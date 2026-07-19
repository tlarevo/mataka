//! Retain pipeline. Mirrors upstream's 3-phase orchestrator in simplified,
//! synchronous form: extract facts (LLM) -> embed -> store facts + FTS (trigger)
//! + entity links. Consolidation into observations is a follow-up worker (THA tier 1).

use crate::llm::LlmClient;
use crate::store::{f32s_to_blob, Store};
use anyhow::Result;
use rusqlite::params;
use serde::Deserialize;
use serde_json::Value;

const EXTRACTION_SYSTEM: &str = "You are a fact extraction engine for an agent memory system. \
Extract discrete, self-contained facts from the input. Classify each as 'world' (facts about \
the world or other people) or 'experience' (the agent's own first-person experiences). \
Resolve pronouns. Extract named entities per fact. If a time is stated or implied, set \
occurred_start as ISO 8601. Respond with ONLY a JSON object: \
{\"facts\": [{\"text\": str, \"fact_type\": \"world\"|\"experience\", \"entities\": [str], \"occurred_start\": str|null}]}";

#[derive(Deserialize)]
struct ExtractedFact {
    text: String,
    #[serde(default = "default_type")]
    fact_type: String,
    #[serde(default)]
    entities: Vec<String>,
    #[serde(default)]
    occurred_start: Option<String>,
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

    // Phase 1: LLM extraction (tolerate fenced JSON from small local models)
    let raw = llm.chat(EXTRACTION_SYSTEM, content, true).await?;
    let cleaned = raw.trim().trim_start_matches("```json").trim_start_matches("```").trim_end_matches("```").trim();
    let extraction: Extraction = serde_json::from_str(cleaned).unwrap_or(Extraction {
        facts: vec![ExtractedFact {
            text: content.to_string(),
            fact_type: "world".into(),
            entities: vec![],
            occurred_start: None,
        }],
    });
    if extraction.facts.is_empty() {
        return Ok(RetainOutcome { memory_ids: vec![], fact_count: 0 });
    }

    // Phase 2: embed all facts in one batch
    let texts: Vec<String> = extraction.facts.iter().map(|f| f.text.clone()).collect();
    let embeddings = llm.embed(&texts).await?;

    // Phase 3: store facts + entity links (FTS kept in sync by trigger)
    let now = chrono::Utc::now().to_rfc3339();
    let tags_json = serde_json::to_string(tags)?;
    let meta_json = metadata.to_string();
    let mut ids = Vec::with_capacity(extraction.facts.len());

    for (fact, emb) in extraction.facts.iter().zip(embeddings.iter()) {
        let id = uuid::Uuid::new_v4().to_string();
        {
            let conn = store.conn.lock().unwrap();
            conn.execute(
                "INSERT INTO memory_units(id, bank_id, text, fact_type, context, occurred_start, mentioned_at, tags, metadata, embedding, created_at)
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)",
                params![
                    id, bank_id, fact.text,
                    if fact.fact_type == "experience" { "experience" } else { "world" },
                    context, fact.occurred_start, now, tags_json, meta_json,
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

    Ok(RetainOutcome { fact_count: ids.len(), memory_ids: ids })
}
