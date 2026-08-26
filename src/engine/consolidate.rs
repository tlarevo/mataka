//! Consolidation pipeline — merges related raw facts into deduplicated,
//! evidence-grounded observations. The feature that makes this "memory that
//! learns" rather than plain RAG.
//!
//! Prompt ported from upstream Hindsight `consolidation/prompts.py` (MIT License).
//! See `prompts/consolidation.md` for the full prompt text.

use crate::llm::LlmClient;
use crate::store::{blob_to_f32s, cosine, f32s_to_blob, Store};
use anyhow::Result;
use rusqlite::params;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Ported from upstream Hindsight consolidation/prompts.py — MIT License.
/// Source: https://github.com/vectorize-io/hindsight
const CONSOLIDATION_SYSTEM: &str = include_str!("../../prompts/consolidation.md");

/// Cosine similarity threshold for grouping related facts.
const SIMILARITY_THRESHOLD: f32 = 0.80;

/// Minimum group size to trigger consolidation.
const MIN_GROUP_SIZE: usize = 2;

#[derive(Deserialize, Serialize)]
struct ConsolidationResult {
    observation_text: String,
    source_ids: Vec<String>,
    supersedes: bool,
}

/// A fact candidate for grouping.
struct FactCandidate {
    id: String,
    text: String,
    embedding: Vec<f32>,
}

pub struct ConsolidateOutcome {
    pub observations_created: usize,
    pub observation_ids: Vec<String>,
}

/// Full-bank consolidation pass — the explicit `/consolidate` endpoint.
/// Unbounded: merges every qualifying group. Background follow-on use must
/// go through [`consolidate_scoped`] instead.
pub async fn consolidate(
    store: &Store,
    llm: &LlmClient,
    bank_id: &str,
) -> Result<ConsolidateOutcome> {
    consolidate_scoped(store, llm, bank_id, &[], usize::MAX).await
}

/// Budget for retain follow-on consolidation passes. Each merge is a full
/// chat completion (~5-15s of local inference), so a follow-on can never
/// fan out unbounded.
pub const FOLLOW_ON_MAX_LLM_CALLS: usize = 8;

/// Bounded consolidation pass for the retain follow-on chain.
///
/// - `focus_ids`: non-empty → only groups containing at least one of these
///   units (the facts just retained) are considered.
/// - `max_llm_calls`: hard cap on LLM merge calls this pass.
///
/// Full-bank unbounded passes used to run after every async retain. On a
/// ~9.3k-fact bank that meant hundreds of fresh chat calls per retain
/// (group membership drifts as facts accumulate, so exact-set idempotency
/// never converged), pegging the local inference server near 80% CPU — the
/// 2026-08-26 incident. Scoping + capping bounds follow-on work to the new
/// facts only.
pub async fn consolidate_scoped(
    store: &Store,
    llm: &LlmClient,
    bank_id: &str,
    focus_ids: &[String],
    max_llm_calls: usize,
) -> Result<ConsolidateOutcome> {
    store.ensure_bank(bank_id)?;

    let bank = store.get_bank(bank_id)?;

    // 1. Load all world/experience facts with embeddings
    let candidates: Vec<FactCandidate> = {
        let conn = bank.read_conn();
        let mut stmt = conn.prepare(
            "SELECT id, text, embedding FROM memory_units
             WHERE bank_id=?1 AND fact_type IN ('world', 'experience') AND embedding IS NOT NULL",
        )?;
        let rows: Vec<FactCandidate> = stmt
            .query_map(params![bank_id], |r| {
                Ok(FactCandidate {
                    id: r.get(0)?,
                    text: r.get(1)?,
                    embedding: blob_to_f32s(&r.get::<_, Vec<u8>>(2)?),
                })
            })?
            .filter_map(|r| r.ok())
            .collect();
        rows
    };

    if candidates.len() < MIN_GROUP_SIZE {
        return Ok(ConsolidateOutcome {
            observations_created: 0,
            observation_ids: vec![],
        });
    }

    // 2. Group by cosine similarity (greedy single-linkage)
    let groups = find_similar_groups(&candidates);

    // Facts already serving as a source of some observation. A group whose
    // members are all covered was already consolidated — skip it. Replaces
    // the old exact-source-set check, which never converged once group
    // membership drifted by even one fact (21.7k observations from 2.4k
    // distinct sets on the live bank).
    let mut covered: HashSet<String> = {
        let conn = bank.read_conn();
        let mut stmt = conn.prepare(
            "SELECT DISTINCT os.source_unit_id FROM observation_sources os
             JOIN memory_units mu ON mu.id = os.observation_id
             WHERE mu.bank_id = ?1 AND mu.fact_type = 'observation'",
        )?;
        let ids: HashSet<String> = stmt
            .query_map(params![bank_id], |r| r.get::<_, String>(0))?
            .filter_map(|r| r.ok())
            .collect();
        ids
    };

    let focus: Option<HashSet<&str>> = if focus_ids.is_empty() {
        None
    } else {
        Some(focus_ids.iter().map(|s| s.as_str()).collect())
    };

    // 3. For each group, call LLM and create observation
    let mut total_created = 0usize;
    let mut all_obs_ids = Vec::new();
    let mut llm_calls = 0usize;

    for group in &groups {
        if llm_calls >= max_llm_calls {
            break;
        }
        if group.len() < MIN_GROUP_SIZE {
            continue;
        }

        // Collect source IDs for this group
        let source_ids: Vec<String> = group.iter().map(|&i| candidates[i].id.clone()).collect();

        // Focus filter: background passes only touch groups with new facts.
        if let Some(f) = &focus {
            if !source_ids.iter().any(|id| f.contains(id.as_str())) {
                continue;
            }
        }

        // Convergent idempotency: every member already consolidated.
        if source_ids.iter().all(|id| covered.contains(id)) {
            continue;
        }

        llm_calls += 1;
        let outcome = consolidate_group(store, llm, bank_id, &candidates, group).await?;
        if let Some(obs_id) = outcome {
            total_created += 1;
            all_obs_ids.push(obs_id);
            covered.extend(source_ids.iter().cloned());
        }
    }

    Ok(ConsolidateOutcome {
        observations_created: total_created,
        observation_ids: all_obs_ids,
    })
}

/// Delete all observations from a bank and their provenance records.
pub fn delete_observations(store: &Store, bank_id: &str) -> Result<usize> {
    store.ensure_bank(bank_id)?;
    let bank = store.get_bank(bank_id)?;
    let conn = bank.get_write_conn();
    // observation_sources rows are cascade-deleted by FK, but be explicit
    conn.execute(
        "DELETE FROM observation_sources WHERE observation_id IN
         (SELECT id FROM memory_units WHERE bank_id=?1 AND fact_type='observation')",
        params![bank_id],
    )?;
    let n = conn.execute(
        "DELETE FROM memory_units WHERE bank_id=?1 AND fact_type='observation'",
        params![bank_id],
    )?;
    Ok(n)
}

/// Greedy single-linkage grouping: for each candidate, find all others with
/// cosine ≥ threshold. O(n²) — fine for ~100k units.
fn find_similar_groups(candidates: &[FactCandidate]) -> Vec<Vec<usize>> {
    let n = candidates.len();
    let mut assigned = vec![false; n];
    let mut groups = Vec::new();

    for i in 0..n {
        if assigned[i] {
            continue;
        }
        let mut group = vec![i];
        assigned[i] = true;

        for j in (i + 1)..n {
            if assigned[j] {
                continue;
            }
            let sim = cosine(&candidates[i].embedding, &candidates[j].embedding);
            if sim >= SIMILARITY_THRESHOLD {
                group.push(j);
                assigned[j] = true;
            }
        }

        groups.push(group);
    }

    groups
}

/// Consolidate a group of facts via LLM (or mock), insert observation + provenance.
async fn consolidate_group(
    store: &Store,
    llm: &LlmClient,
    bank_id: &str,
    candidates: &[FactCandidate],
    group_indices: &[usize],
) -> Result<Option<String>> {
    let group_facts: Vec<&FactCandidate> = group_indices.iter().map(|&i| &candidates[i]).collect();

    // Build user message with fact list
    let facts_json: Vec<serde_json::Value> = group_facts
        .iter()
        .map(|f| serde_json::json!({"id": f.id, "text": f.text}))
        .collect();
    let user_message = format!(
        "Consolidate the following related facts into a single observation:\n\n{}",
        serde_json::to_string_pretty(&facts_json)?
    );

    let raw = llm.chat(CONSOLIDATION_SYSTEM, &user_message, true).await?;
    let cleaned = raw
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();

    let result: ConsolidationResult = match serde_json::from_str(cleaned) {
        Ok(r) => r,
        Err(_) => return Ok(None),
    };

    if result.observation_text.is_empty() || result.source_ids.is_empty() {
        return Ok(None);
    }

    // Provenance may only cite facts in this group. The LLM occasionally
    // returns hallucinated IDs, which broke the FK constraint and aborted
    // the whole pass ("consolidation follow-on failed: FOREIGN KEY
    // constraint failed").
    let group_id_set: HashSet<&str> = group_facts.iter().map(|f| f.id.as_str()).collect();
    let source_ids: Vec<String> = result
        .source_ids
        .into_iter()
        .filter(|id| group_id_set.contains(id.as_str()))
        .collect();
    if source_ids.is_empty() {
        return Ok(None);
    }

    // Embed the observation
    let embeddings = llm
        .embed(std::slice::from_ref(&result.observation_text))
        .await?;
    let emb = embeddings.into_iter().next().unwrap_or_default();

    // Insert observation as memory_units row — one transaction so a failure
    // can never leave an observation without its provenance.
    let obs_id = uuid::Uuid::new_v4().to_string();
    let now = chrono::Utc::now().to_rfc3339();
    {
        let bank = store.get_bank(bank_id)?;
        let conn = bank.get_write_conn();
        conn.execute_batch("BEGIN")?;
        let tx_result = (|| -> Result<()> {
            conn.execute(
                "INSERT INTO memory_units(id, bank_id, text, fact_type, context, occurred_start, occurred_end, mentioned_at, tags, metadata, embedding, created_at)
                 VALUES (?1,?2,?3,'observation',NULL,NULL,NULL,?4,'[]','{}',?5,?4)",
                params![obs_id, bank_id, result.observation_text, now, f32s_to_blob(&emb)],
            )?;
            for source_id in &source_ids {
                conn.execute(
                    "INSERT OR IGNORE INTO observation_sources(observation_id, source_unit_id) VALUES (?1,?2)",
                    params![obs_id, source_id],
                )?;
            }
            Ok(())
        })();
        conn.execute_batch(if tx_result.is_ok() {
            "COMMIT"
        } else {
            "ROLLBACK"
        })?;
        tx_result?;
    }

    Ok(Some(obs_id))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_similar_groups_basic() {
        // Two identical embeddings → same group
        let candidates = vec![
            FactCandidate {
                id: "1".into(),
                text: "Alice likes hiking".into(),
                embedding: vec![1.0, 0.0, 0.0],
            },
            FactCandidate {
                id: "2".into(),
                text: "Alice enjoys hiking".into(),
                embedding: vec![0.99, 0.1, 0.0],
            },
            FactCandidate {
                id: "3".into(),
                text: "Bob likes pizza".into(),
                embedding: vec![0.0, 0.0, 1.0],
            },
        ];
        let groups = find_similar_groups(&candidates);
        // Should have 2 groups: {0,1} and {2}
        assert_eq!(groups.len(), 2);
        assert!(groups.iter().any(|g| g.len() == 2));
        assert!(groups.iter().any(|g| g.len() == 1));
    }

    #[test]
    fn find_similar_groups_all_different() {
        let candidates = vec![
            FactCandidate {
                id: "1".into(),
                text: "a".into(),
                embedding: vec![1.0, 0.0],
            },
            FactCandidate {
                id: "2".into(),
                text: "b".into(),
                embedding: vec![0.0, 1.0],
            },
        ];
        let groups = find_similar_groups(&candidates);
        assert_eq!(groups.len(), 2); // orthogonal → separate groups
    }

    #[test]
    fn find_similar_groups_empty() {
        let candidates = vec![];
        let groups = find_similar_groups(&candidates);
        assert!(groups.is_empty());
    }

    #[test]
    fn parse_consolidation_result() {
        let json = r#"{"observation_text":"Alice enjoys outdoor activities","source_ids":["1","2"],"supersedes":true}"#;
        let r: ConsolidationResult = serde_json::from_str(json).unwrap();
        assert_eq!(r.observation_text, "Alice enjoys outdoor activities");
        assert_eq!(r.source_ids, vec!["1", "2"]);
        assert!(r.supersedes);
    }

    // ── Integration tests (success criteria) ──────────────────────────────

    use crate::llm::LlmClient;
    use crate::store::{f32s_to_blob, Store};
    use rusqlite::params;
    use std::path::PathBuf;

    fn test_store() -> (Store, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.keep();
        let store = Store::open(&path).unwrap();
        (store, path)
    }

    fn test_llm() -> LlmClient {
        LlmClient::from_env() // defaults to "mock" provider
    }

    /// Insert a fact directly into the store with a known ID and embedding.
    fn insert_fact(store: &Store, bank_id: &str, id: &str, text: &str) {
        store.ensure_bank(bank_id).unwrap();
        let bank = store.get_bank(bank_id).unwrap();
        let conn = bank.get_write_conn();
        let emb = crate::llm::mock_embed(text);
        let now = chrono::Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO memory_units(id, bank_id, text, fact_type, context, occurred_start, occurred_end, mentioned_at, tags, metadata, embedding, created_at)
             VALUES (?1,?2,?3,'world',NULL,NULL,NULL,?4,'[]','{}',?5,?4)",
            params![id, bank_id, text, now, f32s_to_blob(&emb)],
        )
        .unwrap();
    }

    /// Helper: count rows matching a query.
    fn count_rows(store: &Store, bank_id: &str, sql: &str) -> i64 {
        let bank = store.get_bank(bank_id).unwrap();
        let conn = bank.read_conn();
        conn.query_row(sql, params![bank_id], |r| r.get(0))
            .unwrap_or(0)
    }

    #[tokio::test]
    async fn consolidation_yields_observation_linked_to_both_sources() {
        let (store, _dir) = test_store();
        let llm = test_llm();
        let bank = "test-consolidate-1";

        // Insert two similar facts with controlled embeddings
        insert_fact(&store, bank, "f1", "Alice likes hiking");
        insert_fact(&store, bank, "f2", "Alice likes hiking trails");

        assert_eq!(
            count_rows(
                &store,
                bank,
                "SELECT COUNT(*) FROM memory_units WHERE bank_id=?1 AND fact_type='world'"
            ),
            2
        );

        // Run consolidation
        let outcome = consolidate(&store, &llm, bank).await.unwrap();
        assert!(
            outcome.observations_created >= 1,
            "expected at least 1 observation, got {}",
            outcome.observations_created
        );

        // Verify observation exists
        let obs_count = count_rows(
            &store,
            bank,
            "SELECT COUNT(*) FROM memory_units WHERE bank_id=?1 AND fact_type='observation'",
        );
        assert!(obs_count >= 1, "expected at least 1 observation row");

        // Verify provenance: observation is linked to both source facts
        let obs_id = &outcome.observation_ids[0];
        let bank_db = store.get_bank(bank).unwrap();
        let conn = bank_db.read_conn();
        let source_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM observation_sources WHERE observation_id=?1",
                params![obs_id],
                |r| r.get(0),
            )
            .unwrap();
        assert!(
            source_count >= 2,
            "observation should be linked to ≥2 source facts, got {source_count}"
        );
        // Original facts still exist (supplement, not replace)
        assert_eq!(
            count_rows(
                &store,
                bank,
                "SELECT COUNT(*) FROM memory_units WHERE bank_id=?1 AND fact_type='world'"
            ),
            2
        );
    }

    #[tokio::test]
    async fn recall_observation_type_and_prefer_observations() {
        let (store, _dir) = test_store();
        let llm = test_llm();
        let bank = "test-recall-obs";

        insert_fact(&store, bank, "f1", "Alice likes hiking");
        insert_fact(&store, bank, "f2", "Alice likes hiking trails");
        consolidate(&store, &llm, bank).await.unwrap();

        // Recall with types=["observation"] should return observation
        let results = crate::engine::recall::recall(
            &store,
            &llm,
            bank,
            "Alice hiking",
            &["observation".into()],
            "mid",
            4096,
            &[],
            "any",
            false,
        )
        .await
        .unwrap();
        assert!(
            results.iter().any(|r| r.fact_type == "observation"),
            "recall with types=[observation] should return an observation"
        );

        // Recall with prefer_observations=true: source facts suppressed
        let results_pref = crate::engine::recall::recall(
            &store,
            &llm,
            bank,
            "Alice hiking",
            &["world".into(), "experience".into(), "observation".into()],
            "mid",
            4096,
            &[],
            "any",
            true,
        )
        .await
        .unwrap();
        let has_obs = results_pref.iter().any(|r| r.fact_type == "observation");
        let has_world = results_pref.iter().any(|r| r.fact_type == "world");
        assert!(has_obs, "prefer_observations should include observations");
        assert!(
            !has_world,
            "prefer_observations should suppress source world facts"
        );
    }

    #[tokio::test]
    async fn delete_observations_removes_only_observation_rows_and_provenance() {
        let (store, _dir) = test_store();
        let llm = test_llm();
        let bank = "test-delete-obs";

        insert_fact(&store, bank, "f1", "Alice likes hiking");
        insert_fact(&store, bank, "f2", "Alice likes hiking trails");
        consolidate(&store, &llm, bank).await.unwrap();

        let obs_before = count_rows(
            &store,
            bank,
            "SELECT COUNT(*) FROM memory_units WHERE bank_id=?1 AND fact_type='observation'",
        );
        assert!(obs_before >= 1);

        let deleted = delete_observations(&store, bank).unwrap();
        assert_eq!(deleted as i64, obs_before);

        // No observations left
        assert_eq!(
            count_rows(
                &store,
                bank,
                "SELECT COUNT(*) FROM memory_units WHERE bank_id=?1 AND fact_type='observation'"
            ),
            0
        );
        // No provenance left
        assert_eq!(
            count_rows(&store, bank, "SELECT COUNT(*) FROM observation_sources"),
            0
        );
        // Original facts still intact
        assert_eq!(
            count_rows(
                &store,
                bank,
                "SELECT COUNT(*) FROM memory_units WHERE bank_id=?1 AND fact_type='world'"
            ),
            2
        );
    }

    #[tokio::test]
    async fn consolidate_idempotent_no_duplicates() {
        let (store, _dir) = test_store();
        let llm = test_llm();
        let bank = "test-idempotent";

        insert_fact(&store, bank, "f1", "Alice likes hiking");
        insert_fact(&store, bank, "f2", "Alice likes hiking trails");

        let _first = consolidate(&store, &llm, bank).await.unwrap();
        let obs_count_after_first = count_rows(
            &store,
            bank,
            "SELECT COUNT(*) FROM memory_units WHERE bank_id=?1 AND fact_type='observation'",
        );

        let second = consolidate(&store, &llm, bank).await.unwrap();
        let obs_count_after_second = count_rows(
            &store,
            bank,
            "SELECT COUNT(*) FROM memory_units WHERE bank_id=?1 AND fact_type='observation'",
        );

        assert!(
            second.observations_created == 0,
            "second consolidate should create 0 new observations, got {}",
            second.observations_created
        );
        assert_eq!(
            obs_count_after_first, obs_count_after_second,
            "observation count should not change on second consolidate"
        );
    }

    // ── 2026-08-26 CPU-incident regression tests ─────────────────────────
    // The retain follow-on used to run an unbounded full-bank pass — hundreds
    // of LLM calls per retain on a real bank, pegging the local inference
    // server. These pin the follow-on contract: scoped + budget-capped.

    #[tokio::test]
    async fn follow_on_scoped_pass_only_touches_focused_group() {
        let (store, _dir) = test_store();
        let llm = test_llm();
        let bank = "test-scoped-focus";

        // Two unrelated similar pairs → two independent groups.
        insert_fact(&store, bank, "a1", "Alice likes hiking");
        insert_fact(&store, bank, "a2", "Alice likes hiking trails");
        insert_fact(&store, bank, "b1", "Bob likes pizza");
        insert_fact(&store, bank, "b2", "Bob really likes pizza");

        // Focus the pass on group A only (as a retain follow-on would).
        let outcome = consolidate_scoped(&store, &llm, bank, &["a1".into(), "a2".into()], 100)
            .await
            .unwrap();

        assert_eq!(
            outcome.observations_created, 1,
            "focused pass should consolidate exactly the focused group"
        );
        // Provenance cites only group-A facts.
        assert_eq!(
            count_rows(
                &store,
                bank,
                "SELECT COUNT(*) FROM observation_sources WHERE source_unit_id IN ('a1','a2')"
            ),
            count_rows(&store, bank, "SELECT COUNT(*) FROM observation_sources"),
            "no provenance outside the focused group"
        );
    }

    #[tokio::test]
    async fn follow_on_pass_caps_llm_calls() {
        let (store, _dir) = test_store();
        let llm = test_llm();
        let bank = "test-scoped-cap";

        // Three independent similar pairs → three groups, three potential calls.
        insert_fact(&store, bank, "a1", "Alice likes hiking");
        insert_fact(&store, bank, "a2", "Alice likes hiking trails");
        insert_fact(&store, bank, "b1", "Bob likes pizza");
        insert_fact(&store, bank, "b2", "Bob really likes pizza");
        insert_fact(&store, bank, "c1", "Carol likes sailing");
        insert_fact(&store, bank, "c2", "Carol enjoys sailing boats");

        let outcome = consolidate_scoped(&store, &llm, bank, &[], 2)
            .await
            .unwrap();

        assert_eq!(
            outcome.observations_created, 2,
            "max_llm_calls=2 must stop the pass at 2 merges, not 3"
        );
    }

    #[tokio::test]
    async fn follow_on_pass_skips_groups_already_covered() {
        let (store, _dir) = test_store();
        let llm = test_llm();
        let bank = "test-scoped-covered";

        insert_fact(&store, bank, "f1", "Alice likes hiking");
        insert_fact(&store, bank, "f2", "Alice likes hiking trails");

        let first = consolidate_scoped(&store, &llm, bank, &[], 100)
            .await
            .unwrap();
        assert_eq!(first.observations_created, 1);

        // Same facts, second pass: every member is already a provenance
        // source, so nothing is re-consolidated — even though group
        // membership can drift, coverage converges.
        let second = consolidate_scoped(&store, &llm, bank, &[], 100)
            .await
            .unwrap();
        assert_eq!(
            second.observations_created, 0,
            "already-covered facts must not trigger another LLM merge"
        );
    }
}
