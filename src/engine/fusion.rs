//! Reciprocal Rank Fusion — direct port of hindsight_api/engine/search/fusion.py (MIT).
//! score(d) = Σ over arms of 1 / (k + rank(d)), k = 60.
//! Arms are ordered [semantic, bm25, graph, temporal] to match upstream.

use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct RetrievalResult {
    pub unit_id: String,
    pub score: f32, // arm-native score (cosine, bm25, ...)
}

#[derive(Clone, Debug)]
pub struct MergedCandidate {
    pub unit_id: String,
    pub rrf_score: f32,
    pub sources: Vec<&'static str>,
    pub semantic: Option<f32>,
    pub keyword: Option<f32>,
}

pub const ARM_NAMES: [&str; 4] = ["semantic", "bm25", "graph", "temporal"];
pub const RRF_K: f32 = 60.0;

/// Cap each arm's list before fusion (upstream: cap_per_source), so one arm
/// can't flood the fused set.
pub fn cap_per_source(results: Vec<RetrievalResult>, cap: usize) -> Vec<RetrievalResult> {
    results.into_iter().take(cap).collect()
}

pub fn reciprocal_rank_fusion(result_lists: [Vec<RetrievalResult>; 4]) -> Vec<MergedCandidate> {
    let mut merged: HashMap<String, MergedCandidate> = HashMap::new();

    for (arm_idx, list) in result_lists.iter().enumerate() {
        let arm = ARM_NAMES[arm_idx];
        for (rank, r) in list.iter().enumerate() {
            let entry = merged.entry(r.unit_id.clone()).or_insert(MergedCandidate {
                unit_id: r.unit_id.clone(),
                rrf_score: 0.0,
                sources: vec![],
                semantic: None,
                keyword: None,
            });
            entry.rrf_score += 1.0 / (RRF_K + (rank as f32) + 1.0);
            entry.sources.push(arm);
            match arm {
                "semantic" => entry.semantic = Some(r.score),
                "bm25" => entry.keyword = Some(r.score),
                _ => {}
            }
        }
    }

    let mut out: Vec<MergedCandidate> = merged.into_values().collect();
    out.sort_by(|a, b| {
        b.rrf_score
            .partial_cmp(&a.rrf_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out
}
