//! Recall pipeline — the TEMPR analog:
//!   semantic (cosine over stored embeddings)
//!   bm25     (SQLite FTS5 native bm25())
//!   graph    (entity co-occurrence expansion)
//!   temporal (recency-ordered slice)
//! fused with RRF (k=60), then packed to a token budget.
//! Cross-encoder reranking is deliberately deferred (tier-1, ONNX feature flag).

use super::fusion::{cap_per_source, reciprocal_rank_fusion, RetrievalResult};
use crate::llm::LlmClient;
use crate::store::{blob_to_f32s, cosine, Store};
use crate::tokens;
use anyhow::Result;
use rusqlite::params;
use serde::Serialize;

#[derive(Serialize, Clone)]
pub struct RecallResult {
    pub id: String,
    pub text: String,
    #[serde(rename = "type")]
    pub fact_type: String,
    pub context: Option<String>,
    pub occurred_start: Option<String>,
    pub tags: serde_json::Value,
    pub score: f32,
    pub sources: Vec<&'static str>,
}

pub struct Budget;
impl Budget {
    /// Per-arm candidate counts per budget level (upstream: low/mid/high).
    pub fn candidates(level: &str) -> usize {
        match level {
            "low" => 20,
            "high" => 100,
            _ => 50,
        }
    }
}

/// Pure function: does a unit's tag set pass the filter?
pub fn tags_pass(unit_tags: &[String], filter: &[String], mode: &str) -> bool {
    if filter.is_empty() {
        // exact + empty filter → only untagged pass
        return match mode {
            "exact" => unit_tags.is_empty(),
            _ => true, // all other modes pass with no constraint
        };
    }
    let is_untagged = unit_tags.is_empty();
    match mode {
        "any" => {
            // OR match, includes untagged
            if is_untagged {
                return true;
            }
            filter.iter().any(|t| unit_tags.contains(t))
        }
        "all" => {
            // AND match, includes untagged
            if is_untagged {
                return true;
            }
            filter.iter().all(|t| unit_tags.contains(t))
        }
        "any_strict" => {
            // OR match, excludes untagged
            if is_untagged {
                return false;
            }
            filter.iter().any(|t| unit_tags.contains(t))
        }
        "all_strict" => {
            // AND match, excludes untagged
            if is_untagged {
                return false;
            }
            filter.iter().all(|t| unit_tags.contains(t))
        }
        "exact" => {
            // Set-equality, excludes untagged (unless filter is empty)
            if is_untagged {
                return false;
            }
            unit_tags.len() == filter.len() && filter.iter().all(|t| unit_tags.contains(t))
        }
        _ => true, // unknown mode: pass (defensive)
    }
}

#[allow(clippy::too_many_arguments)]
pub async fn recall(
    store: &Store,
    llm: &LlmClient,
    bank_id: &str,
    query: &str,
    types: &[String],
    budget: &str,
    max_tokens: usize,
    filter_tags: &[String],
    tags_match: &str,
    prefer_observations: bool,
) -> Result<Vec<RecallResult>> {
    let n = Budget::candidates(budget);
    let type_filter: Vec<String> = if types.is_empty() {
        vec!["world".into(), "experience".into()]
    } else {
        types.to_vec()
    };

    // --- Arm 1: semantic ---
    let qvec = llm.embed(&[query.to_string()]).await?.remove(0);
    let bank = store.get_bank(bank_id)?;
    let semantic = {
        let conn = bank.read_conn();
        let mut stmt = conn.prepare(
            "SELECT id, embedding FROM memory_units WHERE bank_id=?1 AND embedding IS NOT NULL AND quarantined_at IS NULL",
        )?;
        let mut scored: Vec<RetrievalResult> = stmt
            .query_map(params![bank_id], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, Vec<u8>>(1)?))
            })?
            .filter_map(|row| row.ok())
            .map(|(id, blob)| RetrievalResult {
                unit_id: id,
                score: cosine(&qvec, &blob_to_f32s(&blob)),
            })
            .collect();
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored
    };

    // --- Arm 2: BM25 via FTS5 ---
    let fts_query: String = query
        .split_whitespace()
        .map(|w| {
            w.chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
        })
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join(" OR ");
    let bm25 = if fts_query.is_empty() {
        vec![]
    } else {
        let conn = bank.read_conn();
        let mut stmt = conn.prepare(
            "SELECT mu.id, bm25(memory_fts) AS s FROM memory_fts
             JOIN memory_units mu ON mu.rowid = memory_fts.rowid
             WHERE memory_fts MATCH ?1 AND mu.bank_id = ?2 AND mu.quarantined_at IS NULL ORDER BY s LIMIT ?3",
        )?;
        let v: Vec<RetrievalResult> = stmt
            .query_map(params![fts_query, bank_id, n as i64], |r| {
                Ok(RetrievalResult {
                    unit_id: r.get(0)?,
                    score: -r.get::<_, f64>(1)? as f32, // FTS5 bm25() is lower-is-better
                })
            })?
            .filter_map(|x| x.ok())
            .collect();
        v
    };

    // --- Arm 3: graph (units sharing entities with top lexical/semantic seeds) ---
    let seed_ids: Vec<String> = semantic
        .iter()
        .take(5)
        .chain(bm25.iter().take(5))
        .map(|r| r.unit_id.clone())
        .collect();
    let graph = if seed_ids.is_empty() {
        vec![]
    } else {
        let conn = bank.read_conn();
        let placeholders = seed_ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!(
            "SELECT ue2.unit_id, COUNT(*) AS overlap FROM unit_entities ue1
             JOIN unit_entities ue2 ON ue1.entity_id = ue2.entity_id
             JOIN memory_units mu ON mu.id = ue2.unit_id AND mu.quarantined_at IS NULL
             WHERE ue1.unit_id IN ({placeholders}) AND ue2.unit_id NOT IN ({placeholders})
             GROUP BY ue2.unit_id ORDER BY overlap DESC LIMIT {n}"
        );
        let mut stmt = conn.prepare(&sql)?;
        let bind: Vec<&dyn rusqlite::ToSql> = seed_ids
            .iter()
            .chain(seed_ids.iter())
            .map(|s| s as &dyn rusqlite::ToSql)
            .collect();
        let v: Vec<RetrievalResult> = stmt
            .query_map(&bind[..], |r| {
                Ok(RetrievalResult {
                    unit_id: r.get(0)?,
                    score: r.get::<_, i64>(1)? as f32,
                })
            })?
            .filter_map(|x| x.ok())
            .collect();
        v
    };

    // --- Arm 4: temporal (recency) ---
    let temporal = {
        let conn = bank.read_conn();
        let mut stmt = conn.prepare(
            "SELECT id FROM memory_units WHERE bank_id=?1 AND quarantined_at IS NULL
             ORDER BY COALESCE(occurred_start, created_at) DESC LIMIT ?2",
        )?;
        let v: Vec<RetrievalResult> = stmt
            .query_map(params![bank_id, n as i64], |r| {
                Ok(RetrievalResult {
                    unit_id: r.get(0)?,
                    score: 0.0,
                })
            })?
            .filter_map(|x| x.ok())
            .collect();
        v
    };

    // --- Fuse + hydrate + pack to budget ---
    let fused = reciprocal_rank_fusion([
        cap_per_source(semantic, n),
        cap_per_source(bm25, n),
        cap_per_source(graph, n),
        cap_per_source(temporal, n),
    ]);

    // --- prefer_observations: collect source IDs from observations in results ---
    let suppressed_sources: std::collections::HashSet<String> = if prefer_observations {
        let conn2 = bank.read_conn();
        let mut sources = std::collections::HashSet::new();
        for cand in &fused {
            let ft: String = conn2
                .query_row(
                    "SELECT fact_type FROM memory_units WHERE id=?1",
                    params![cand.unit_id],
                    |r| r.get(0),
                )
                .unwrap_or_default();
            if ft == "observation" {
                let mut stmt = conn2
                    .prepare(
                        "SELECT source_unit_id FROM observation_sources WHERE observation_id=?1",
                    )
                    .unwrap();
                let ids: Vec<String> = stmt
                    .query_map(params![cand.unit_id], |r| r.get(0))
                    .unwrap()
                    .filter_map(|r| r.ok())
                    .collect();
                for id in ids {
                    sources.insert(id);
                }
            }
        }
        sources
    } else {
        std::collections::HashSet::new()
    };

    let conn = bank.read_conn();
    let mut results = Vec::new();
    let mut token_used = 0usize;
    for cand in fused {
        let row = conn.query_row(
            "SELECT text, fact_type, context, occurred_start, tags FROM memory_units WHERE id=?1 AND quarantined_at IS NULL",
            params![cand.unit_id],
            |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, Option<String>>(2)?,
                    r.get::<_, Option<String>>(3)?,
                    r.get::<_, String>(4)?,
                ))
            },
        );
        let Ok((text, ft, ctx, occ, tags)) = row else {
            continue;
        };
        if !type_filter.contains(&ft) {
            continue;
        }
        // Suppress source facts when prefer_observations is active
        if prefer_observations
            && (ft == "world" || ft == "experience")
            && suppressed_sources.contains(&cand.unit_id)
        {
            continue;
        }
        // Apply tag filter
        let unit_tags: Vec<String> = serde_json::from_str(&tags).unwrap_or_default();
        if !filter_tags.is_empty() && !tags_pass(&unit_tags, filter_tags, tags_match) {
            continue;
        }
        let est_tokens = tokens::count(&text) + 8; // per-item overhead (matches upstream)

        if token_used + est_tokens > max_tokens {
            break;
        }
        token_used += est_tokens;
        results.push(RecallResult {
            id: cand.unit_id,
            text,
            fact_type: ft,
            context: ctx,
            occurred_start: occ,
            tags: serde_json::from_str(&tags).unwrap_or(serde_json::json!([])),
            score: cand.rrf_score,
            sources: cand.sources,
        });
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(s: &[&str]) -> Vec<String> {
        s.iter().map(|s| s.to_string()).collect()
    }

    // --- any (OR, includes untagged) ---
    #[test]
    fn any_untagged() {
        assert!(tags_pass(&[], &v(&["a"]), "any"));
    }
    #[test]
    fn any_partial_overlap() {
        assert!(tags_pass(&v(&["a"]), &v(&["a", "b"]), "any"));
    }
    #[test]
    fn any_full_overlap() {
        assert!(tags_pass(&v(&["a", "b"]), &v(&["a"]), "any"));
    }
    #[test]
    fn any_superset() {
        assert!(tags_pass(&v(&["a", "b", "c"]), &v(&["a", "b"]), "any"));
    }
    #[test]
    fn any_exact_match() {
        assert!(tags_pass(&v(&["a", "b"]), &v(&["a", "b"]), "any"));
    }

    // --- all (AND, includes untagged) ---
    #[test]
    fn all_untagged() {
        assert!(tags_pass(&[], &v(&["a"]), "all"));
    }
    #[test]
    fn all_partial_overlap() {
        assert!(!tags_pass(&v(&["a"]), &v(&["a", "b"]), "all"));
    }
    #[test]
    fn all_full_overlap() {
        assert!(tags_pass(&v(&["a", "b"]), &v(&["a"]), "all"));
    }
    #[test]
    fn all_superset() {
        assert!(tags_pass(&v(&["a", "b", "c"]), &v(&["a", "b"]), "all"));
    }
    #[test]
    fn all_exact_match() {
        assert!(tags_pass(&v(&["a", "b"]), &v(&["a", "b"]), "all"));
    }

    // --- any_strict (OR, excludes untagged) ---
    #[test]
    fn any_strict_untagged() {
        assert!(!tags_pass(&[], &v(&["a"]), "any_strict"));
    }
    #[test]
    fn any_strict_partial_overlap() {
        assert!(tags_pass(&v(&["a"]), &v(&["a", "b"]), "any_strict"));
    }
    #[test]
    fn any_strict_full_overlap() {
        assert!(tags_pass(&v(&["a", "b"]), &v(&["a"]), "any_strict"));
    }
    #[test]
    fn any_strict_superset() {
        assert!(tags_pass(
            &v(&["a", "b", "c"]),
            &v(&["a", "b"]),
            "any_strict"
        ));
    }
    #[test]
    fn any_strict_exact_match() {
        assert!(tags_pass(&v(&["a", "b"]), &v(&["a", "b"]), "any_strict"));
    }

    // --- all_strict (AND, excludes untagged) ---
    #[test]
    fn all_strict_untagged() {
        assert!(!tags_pass(&[], &v(&["a"]), "all_strict"));
    }
    #[test]
    fn all_strict_partial_overlap() {
        assert!(!tags_pass(&v(&["a"]), &v(&["a", "b"]), "all_strict"));
    }
    #[test]
    fn all_strict_full_overlap() {
        assert!(tags_pass(&v(&["a", "b"]), &v(&["a"]), "all_strict"));
    }
    #[test]
    fn all_strict_superset() {
        assert!(tags_pass(
            &v(&["a", "b", "c"]),
            &v(&["a", "b"]),
            "all_strict"
        ));
    }
    #[test]
    fn all_strict_exact_match() {
        assert!(tags_pass(&v(&["a", "b"]), &v(&["a", "b"]), "all_strict"));
    }

    // --- exact (set-equality, excludes untagged; empty filter → only untagged) ---
    #[test]
    fn exact_untagged() {
        assert!(!tags_pass(&[], &v(&["a"]), "exact"));
    }
    #[test]
    fn exact_partial_overlap() {
        assert!(!tags_pass(&v(&["a"]), &v(&["a", "b"]), "exact"));
    }
    #[test]
    fn exact_full_overlap() {
        assert!(!tags_pass(&v(&["a", "b"]), &v(&["a"]), "exact"));
    }
    #[test]
    fn exact_superset() {
        assert!(!tags_pass(&v(&["a", "b", "c"]), &v(&["a", "b"]), "exact"));
    }
    #[test]
    fn exact_exact_match() {
        assert!(tags_pass(&v(&["a", "b"]), &v(&["a", "b"]), "exact"));
    }

    // --- exact + empty filter: only untagged pass ---
    #[test]
    fn exact_empty_filter_untagged() {
        assert!(tags_pass(&[], &[], "exact"));
    }
    #[test]
    fn exact_empty_filter_tagged() {
        assert!(!tags_pass(&v(&["a"]), &[], "exact"));
    }
}
