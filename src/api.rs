//! HTTP surface — path-and-schema compatible with Hindsight's dataplane
//! (contract pinned in contract/openapi-0.8.4.json). Tier-0 routes only;
//! see PARITY.md for the full 74-route tiering.

use crate::engine;
use crate::llm::LlmClient;
use crate::store::Store;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Json, Response},
    routing::{delete, get, post, put},
    Router,
};
use rusqlite::params;
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::Arc;

pub struct AppState {
    pub store: Store,
    pub llm: LlmClient,
}

type S = Arc<AppState>;

pub fn router(state: S) -> Router {
    Router::new()
        .route("/health", get(|| async { Json(json!({"status": "ok"})) }))
        .route("/version", get(|| async {
            Json(json!({"version": "0.1.0-mataka", "compat": "hindsight-0.8.4"}))
        }))
        .route("/v1/default/banks", get(list_banks))
        .route("/v1/default/banks/:bank_id", put(put_bank).patch(patch_bank).delete(delete_bank))
        .route("/v1/default/banks/:bank_id/stats", get(bank_stats))
        .route("/v1/default/banks/:bank_id/memories", post(retain_memories).delete(delete_memories))
        .route("/v1/default/banks/:bank_id/memories/list", get(list_memories))
        .route("/v1/default/banks/:bank_id/memories/recall", post(recall_memories))
        .route("/v1/default/banks/:bank_id/memories/:memory_id", get(get_memory).delete(delete_memory))
        .route("/v1/default/banks/:bank_id/reflect", post(reflect_bank))
        .route("/v1/default/banks/:bank_id/entities", get(list_entities))
        .with_state(state)
}

struct ApiError(anyhow::Error);
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"detail": self.0.to_string()}))).into_response()
    }
}
impl<E: Into<anyhow::Error>> From<E> for ApiError {
    fn from(e: E) -> Self {
        Self(e.into())
    }
}

// ---------- banks ----------

async fn list_banks(State(s): State<S>) -> Result<Json<Value>, ApiError> {
    let conn = s.store.conn.lock().unwrap();
    let mut stmt = conn.prepare("SELECT bank_id, name, mission, created_at FROM banks ORDER BY created_at")?;
    let banks: Vec<Value> = stmt
        .query_map([], |r| {
            Ok(json!({
                "bank_id": r.get::<_, String>(0)?,
                "name": r.get::<_, Option<String>>(1)?,
                "mission": r.get::<_, String>(2)?,
                "created_at": r.get::<_, String>(3)?,
            }))
        })?
        .filter_map(|x| x.ok())
        .collect();
    Ok(Json(json!({"banks": banks})))
}

#[derive(Deserialize, Default)]
struct BankBody {
    name: Option<String>,
    mission: Option<String>,
    disposition: Option<Value>,
}

async fn put_bank(State(s): State<S>, Path(bank_id): Path<String>, body: Option<Json<BankBody>>) -> Result<Json<Value>, ApiError> {
    let b = body.map(|Json(b)| b).unwrap_or_default();
    let now = chrono::Utc::now().to_rfc3339();
    s.store.conn.lock().unwrap().execute(
        "INSERT INTO banks(bank_id, name, mission, disposition, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?5)
         ON CONFLICT(bank_id) DO UPDATE SET
           name=COALESCE(excluded.name, name),
           mission=COALESCE(NULLIF(excluded.mission,''), mission),
           updated_at=excluded.updated_at",
        params![
            bank_id,
            b.name.clone().unwrap_or_else(|| bank_id.clone()),
            b.mission.clone().unwrap_or_default(),
            b.disposition.map(|d| d.to_string()).unwrap_or_else(|| "{}".into()),
            now
        ],
    )?;
    Ok(Json(json!({"bank_id": bank_id, "status": "ok"})))
}

async fn patch_bank(state: State<S>, path: Path<String>, body: Option<Json<BankBody>>) -> Result<Json<Value>, ApiError> {
    put_bank(state, path, body).await
}

async fn delete_bank(State(s): State<S>, Path(bank_id): Path<String>) -> Result<Json<Value>, ApiError> {
    s.store.conn.lock().unwrap().execute("DELETE FROM banks WHERE bank_id=?1", params![bank_id])?;
    Ok(Json(json!({"bank_id": bank_id, "status": "deleted"})))
}

async fn bank_stats(State(s): State<S>, Path(bank_id): Path<String>) -> Result<Json<Value>, ApiError> {
    Ok(Json(s.store.bank_stats(&bank_id)?))
}

// ---------- retain ----------

#[derive(Deserialize)]
struct RetainItem {
    content: String,
    #[serde(default)]
    context: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    metadata: Value,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum RetainRequest {
    Batch { items: Vec<RetainItem> },
    Single(RetainItem),
}

async fn retain_memories(
    State(s): State<S>,
    Path(bank_id): Path<String>,
    Json(req): Json<RetainRequest>,
) -> Result<Json<Value>, ApiError> {
    let items = match req {
        RetainRequest::Batch { items } => items,
        RetainRequest::Single(i) => vec![i],
    };
    let mut all_ids = Vec::new();
    let mut total_facts = 0;
    for item in &items {
        let out = engine::retain::retain(
            &s.store, &s.llm, &bank_id, &item.content,
            item.context.as_deref(), &item.tags, &item.metadata,
        )
        .await?;
        total_facts += out.fact_count;
        all_ids.extend(out.memory_ids);
    }
    Ok(Json(json!({
        "bank_id": bank_id,
        "items_processed": items.len(),
        "facts_extracted": total_facts,
        "memory_ids": all_ids,
        "status": "completed"
    })))
}

// ---------- recall ----------

#[derive(Deserialize)]
struct RecallRequest {
    query: String,
    #[serde(default)]
    types: Option<Vec<String>>,
    #[serde(default = "default_budget")]
    budget: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
}
fn default_budget() -> String {
    "mid".into()
}
fn default_max_tokens() -> usize {
    4096
}

async fn recall_memories(
    State(s): State<S>,
    Path(bank_id): Path<String>,
    Json(req): Json<RecallRequest>,
) -> Result<Json<Value>, ApiError> {
    let results = engine::recall::recall(
        &s.store, &s.llm, &bank_id, &req.query,
        &req.types.unwrap_or_default(), &req.budget, req.max_tokens,
    )
    .await?;
    Ok(Json(json!({"results": results})))
}

// ---------- reflect ----------

#[derive(Deserialize)]
struct ReflectRequest {
    query: String,
    #[serde(default = "default_reflect_budget")]
    budget: String,
}
fn default_reflect_budget() -> String {
    "low".into()
}

async fn reflect_bank(
    State(s): State<S>,
    Path(bank_id): Path<String>,
    Json(req): Json<ReflectRequest>,
) -> Result<Json<Value>, ApiError> {
    let text = engine::reflect(&s.store, &s.llm, &bank_id, &req.query, &req.budget).await?;
    Ok(Json(json!({"text": text, "bank_id": bank_id})))
}

// ---------- memories CRUD ----------

#[derive(Deserialize)]
struct ListQuery {
    #[serde(default = "default_limit")]
    limit: i64,
    #[serde(default)]
    offset: i64,
}
fn default_limit() -> i64 {
    50
}

async fn list_memories(
    State(s): State<S>,
    Path(bank_id): Path<String>,
    Query(q): Query<ListQuery>,
) -> Result<Json<Value>, ApiError> {
    let conn = s.store.conn.lock().unwrap();
    let mut stmt = conn.prepare(
        "SELECT id, text, fact_type, occurred_start, created_at FROM memory_units
         WHERE bank_id=?1 ORDER BY created_at DESC LIMIT ?2 OFFSET ?3",
    )?;
    let items: Vec<Value> = stmt
        .query_map(params![bank_id, q.limit, q.offset], |r| {
            Ok(json!({
                "id": r.get::<_, String>(0)?,
                "text": r.get::<_, String>(1)?,
                "type": r.get::<_, String>(2)?,
                "occurred_start": r.get::<_, Option<String>>(3)?,
                "created_at": r.get::<_, String>(4)?,
            }))
        })?
        .filter_map(|x| x.ok())
        .collect();
    Ok(Json(json!({"items": items})))
}

async fn get_memory(State(s): State<S>, Path((bank_id, memory_id)): Path<(String, String)>) -> Result<Response, ApiError> {
    let conn = s.store.conn.lock().unwrap();
    let row = conn.query_row(
        "SELECT text, fact_type, context, occurred_start, tags, metadata, created_at
         FROM memory_units WHERE bank_id=?1 AND id=?2",
        params![bank_id, memory_id],
        |r| {
            Ok(json!({
                "id": memory_id,
                "text": r.get::<_, String>(0)?,
                "type": r.get::<_, String>(1)?,
                "context": r.get::<_, Option<String>>(2)?,
                "occurred_start": r.get::<_, Option<String>>(3)?,
                "tags": serde_json::from_str::<Value>(&r.get::<_, String>(4)?).unwrap_or(json!([])),
                "metadata": serde_json::from_str::<Value>(&r.get::<_, String>(5)?).unwrap_or(json!({})),
                "created_at": r.get::<_, String>(6)?,
            }))
        },
    );
    match row {
        Ok(v) => Ok(Json(v).into_response()),
        Err(_) => Ok((StatusCode::NOT_FOUND, Json(json!({"detail": "memory not found"}))).into_response()),
    }
}

async fn delete_memory(State(s): State<S>, Path((bank_id, memory_id)): Path<(String, String)>) -> Result<Json<Value>, ApiError> {
    s.store.conn.lock().unwrap().execute(
        "DELETE FROM memory_units WHERE bank_id=?1 AND id=?2",
        params![bank_id, memory_id],
    )?;
    Ok(Json(json!({"status": "deleted"})))
}

async fn delete_memories(State(s): State<S>, Path(bank_id): Path<String>) -> Result<Json<Value>, ApiError> {
    let n = s.store.conn.lock().unwrap().execute(
        "DELETE FROM memory_units WHERE bank_id=?1",
        params![bank_id],
    )?;
    Ok(Json(json!({"deleted": n})))
}

async fn list_entities(State(s): State<S>, Path(bank_id): Path<String>) -> Result<Json<Value>, ApiError> {
    let conn = s.store.conn.lock().unwrap();
    let mut stmt = conn.prepare(
        "SELECT e.id, e.canonical_name, COUNT(ue.unit_id) FROM entities e
         LEFT JOIN unit_entities ue ON ue.entity_id = e.id
         WHERE e.bank_id=?1 GROUP BY e.id ORDER BY COUNT(ue.unit_id) DESC",
    )?;
    let items: Vec<Value> = stmt
        .query_map(params![bank_id], |r| {
            Ok(json!({
                "id": r.get::<_, String>(0)?,
                "name": r.get::<_, String>(1)?,
                "mention_count": r.get::<_, i64>(2)?,
            }))
        })?
        .filter_map(|x| x.ok())
        .collect();
    Ok(Json(json!({"entities": items})))
}
