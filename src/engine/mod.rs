pub mod fusion;
pub mod recall;
pub mod retain;

use crate::llm::LlmClient;
use crate::store::Store;
use anyhow::Result;
use rusqlite::params;

/// Reflect: disposition-aware answer grounded in recalled memories.
/// Upstream runs an agentic tool loop over Mental Models -> Observations -> Raw
/// Facts; MVP collapses this to single-shot recall + generation. Same request/
/// response contract, upgrade path is internal.
pub async fn reflect(
    store: &Store,
    llm: &LlmClient,
    bank_id: &str,
    query: &str,
    budget: &str,
) -> Result<String> {
    let memories = recall::recall(
        store,
        llm,
        bank_id,
        query,
        &["world".into(), "experience".into(), "observation".into()],
        budget,
        2048,
        &[],
        "any",
    )
    .await?;

    let (mission, disposition) = {
        let conn = store.conn.lock().unwrap();
        conn.query_row(
            "SELECT mission, disposition FROM banks WHERE bank_id=?1",
            params![bank_id],
            |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)),
        )
        .unwrap_or((String::new(), "{}".into()))
    };

    let context_block: String = memories
        .iter()
        .map(|m| format!("- [{}] {}", m.fact_type, m.text))
        .collect::<Vec<_>>()
        .join("\n");

    let system = format!(
        "You are the reasoning layer of an agent memory bank.\nMission: {}\nDisposition: {}\n\
         Answer strictly grounded in the memories below. If memories don't cover it, say so.\n\nMemories:\n{}",
        mission, disposition, context_block
    );
    llm.chat(&system, query, false).await
}
