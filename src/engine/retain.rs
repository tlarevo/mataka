//! Retain pipeline. Mirrors upstream's 3-phase orchestrator in simplified,
//! synchronous form: extract facts (LLM) -> embed -> store facts + FTS (trigger)
//! + entity links. Consolidation into observations is a follow-up worker (THA tier 1).
//!
//! Prompt ported from upstream Hindsight `fact_extraction.py` (MIT License).
//! See `prompts/extraction.md` for the full prompt text.

use crate::chunking;
use crate::llm::LlmClient;
use crate::store::{f32s_to_blob, Store};
use anyhow::Result;
use rusqlite::params;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Ported from upstream Hindsight fact_extraction.py — MIT License.
/// Source: https://github.com/vectorize-io/hindsight @ 29cc1d7
const EXTRACTION_SYSTEM: &str = include_str!("../../prompts/extraction.md");

/// ~2k tokens ≈ 8k chars (4 chars/token MVP estimator from recall.rs)
const CHUNK_SIZE: usize = 8000;
/// ~200 tokens overlap ≈ 800 chars
const CHUNK_OVERLAP: usize = 800;

#[derive(Deserialize, Serialize)]
struct ExtractedFact {
    text: String,
    #[serde(default = "default_type")]
    fact_type: String,
    #[serde(default)]
    entities: Vec<String>,
    #[serde(default)]
    occurred_start: Option<String>,
    #[serde(default)]
    occurred_end: Option<String>,
}
fn default_type() -> String {
    "world".into()
}

#[derive(Deserialize)]
struct Extraction {
    #[serde(default)]
    facts: Vec<ExtractedFact>,
}

pub struct RetainOutcome {
    pub memory_ids: Vec<String>,
    pub fact_count: usize,
}

pub async fn retain(
    store: &Store,
    llm: &LlmClient,
    bank_id: &str,
    content: &str,
    context: Option<&str>,
    tags: &[String],
    metadata: &Value,
) -> Result<RetainOutcome> {
    store.ensure_bank(bank_id)?;

    // Event Date for temporal resolution — injected into each chunk's user message.
    let event_date = chrono::Utc::now();
    let event_date_str = format!(
        "{} ({})",
        event_date.format("%A, %B %d, %Y"),
        event_date.format("%Y-%m-%d")
    );

    // Chunk long inputs before extraction
    let chunks = chunking::chunk_text(content, CHUNK_SIZE, CHUNK_OVERLAP);
    let total_chunks = chunks.len();

    let mut all_facts: Vec<ExtractedFact> = Vec::new();

    for (i, chunk) in chunks.iter().enumerate() {
        // Build user message with Event Date, matching upstream's _build_user_message
        let user_message = format!(
            "Extract facts from the following text chunk.\n\
             Chunk: {}/{}\n\
             Event Date: {}\n\
             Context: {}\n\
             \n\
             Text:\n{}",
            i + 1,
            total_chunks,
            event_date_str,
            context.unwrap_or("none"),
            chunk
        );

        let raw = llm.chat(EXTRACTION_SYSTEM, &user_message, true).await?;
        let cleaned = raw
            .trim()
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();
        let extraction: Extraction = serde_json::from_str(cleaned).unwrap_or(Extraction {
            facts: vec![],
        });
        all_facts.extend(extraction.facts);
    }

    // Dedupe identical fact texts within the batch (cross-chunk dedup)
    {
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        all_facts.retain(|f| seen.insert(f.text.clone()));
    }

    // Fixture dump for Looper eval harness (THA-136)
    if let Ok(dump_dir) = std::env::var("MATAKA_DUMP_EXTRACTIONS") {
        if !dump_dir.is_empty() {
            dump_extraction_fixtures(&dump_dir, content, &llm.model, &all_facts).ok();
        }
    }

    if all_facts.is_empty() {
        return Ok(RetainOutcome {
            memory_ids: vec![],
            fact_count: 0,
        });
    }

    // Phase 2: embed all facts in one batch
    let texts: Vec<String> = all_facts.iter().map(|f| f.text.clone()).collect();
    let embeddings = llm.embed(&texts).await?;

    // Phase 3: store facts + entity links (FTS kept in sync by trigger)
    let now = chrono::Utc::now().to_rfc3339();
    let tags_json = serde_json::to_string(tags)?;
    let meta_json = metadata.to_string();
    let mut ids = Vec::with_capacity(all_facts.len());

    for (fact, emb) in all_facts.iter().zip(embeddings.iter()) {
        let id = uuid::Uuid::new_v4().to_string();
        {
            let conn = store.conn.lock().unwrap();
            conn.execute(
                "INSERT INTO memory_units(id, bank_id, text, fact_type, context, occurred_start, occurred_end, mentioned_at, tags, metadata, embedding, created_at)
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12)",
                params![
                    id, bank_id, fact.text,
                    if fact.fact_type == "experience" { "experience" } else { "world" },
                    context, fact.occurred_start, fact.occurred_end, now, tags_json, meta_json,
                    f32s_to_blob(emb), now
                ],
            )?;
        }
        for entity in &fact.entities {
            let eid = store.upsert_entity(bank_id, entity)?;
            store.conn.lock().unwrap().execute(
                "INSERT OR IGNORE INTO unit_entities(unit_id, entity_id) VALUES (?1,?2)",
                params![id, eid],
            )?;
        }
        ids.push(id);
    }

    Ok(RetainOutcome {
        fact_count: ids.len(),
        memory_ids: ids,
    })
}

/// Write extraction fixtures as JSONL for Looper eval harness (THA-70..76).
/// Each record: { input, model, output_json } — consumable by Looper as
/// eval cases where the frontier model output is reference and Ornith 9B
/// is candidate.
fn dump_extraction_fixtures(
    dir: &str,
    input: &str,
    model: &str,
    facts: &[ExtractedFact],
) -> Result<()> {
    use std::fs::OpenOptions;
    use std::io::Write;

    std::fs::create_dir_all(dir)?;
    let path = std::path::Path::new(dir).join("extraction_fixtures.jsonl");
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;

    let record = serde_json::json!({
        "input": input,
        "model": model,
        "output_json": { "facts": facts },
    });
    writeln!(file, "{}", serde_json::to_string(&record)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_valid_extraction() {
        let json = r#"{"facts":[{"text":"Alice likes coffee","fact_type":"world","entities":["Alice"],"occurred_start":null,"occurred_end":null}]}"#;
        let e: Extraction = serde_json::from_str(json).unwrap();
        assert_eq!(e.facts.len(), 1);
        assert_eq!(e.facts[0].text, "Alice likes coffee");
        assert_eq!(e.facts[0].fact_type, "world");
        assert!(e.facts[0].occurred_end.is_none());
    }

    #[test]
    fn parse_with_occurred_end() {
        let json = r#"{"facts":[{"text":"Project ran Jan-Mar 2024","fact_type":"world","entities":["Project"],"occurred_start":"2024-01-01","occurred_end":"2024-03-31"}]}"#;
        let e: Extraction = serde_json::from_str(json).unwrap();
        assert_eq!(e.facts[0].occurred_end.as_deref(), Some("2024-03-31"));
    }

    #[test]
    fn empty_facts_is_ok() {
        let json = r#"{"facts":[]}"#;
        let e: Extraction = serde_json::from_str(json).unwrap();
        assert!(e.facts.is_empty());
    }

    #[test]
    fn defaults_for_missing_fields() {
        let json = r#"{"facts":[{"text":"test fact"}]}"#;
        let e: Extraction = serde_json::from_str(json).unwrap();
        assert_eq!(e.facts[0].fact_type, "world");
        assert!(e.facts[0].entities.is_empty());
        assert!(e.facts[0].occurred_start.is_none());
        assert!(e.facts[0].occurred_end.is_none());
    }
}
