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
    routing::{get, post, put},
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
        .route(
            "/version",
            get(|| async { Json(json!({"version": "0.1.0-mataka", "compat": "hindsight-0.8.4"})) }),
        )
        .route("/v1/default/banks", get(list_banks))
        .route(
            "/v1/default/banks/:bank_id",
            put(put_bank).patch(patch_bank).delete(delete_bank),
        )
        .route("/v1/default/banks/:bank_id/stats", get(bank_stats))
        .route(
            "/v1/default/banks/:bank_id/memories",
            post(retain_memories).delete(delete_memories),
        )
        .route(
            "/v1/default/banks/:bank_id/memories/list",
            get(list_memories),
        )
        .route(
            "/v1/default/banks/:bank_id/memories/recall",
            post(recall_memories),
        )
        .route(
            "/v1/default/banks/:bank_id/memories/:memory_id",
            get(get_memory).delete(delete_memory),
        )
        .route("/v1/default/banks/:bank_id/vet", post(vet_bank))
        .route("/v1/default/banks/:bank_id/reflect", post(reflect_bank))
        .route("/v1/default/banks/:bank_id/entities", get(list_entities))
        .route(
            "/v1/default/banks/:bank_id/operations",
            get(list_operations),
        )
        .route(
            "/v1/default/banks/:bank_id/operations/:operation_id",
            get(get_operation_status).delete(cancel_operation),
        )
        .route(
            "/v1/default/banks/:bank_id/operations/:operation_id/retry",
            post(retry_operation),
        )
        .with_state(state)
}

struct ApiError(anyhow::Error);
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"detail": self.0.to_string()})),
        )
            .into_response()
    }
}
impl<E: Into<anyhow::Error>> From<E> for ApiError {
    fn from(e: E) -> Self {
        Self(e.into())
    }
}

// ---------- banks ----------

async fn list_banks(State(s): State<S>) -> Result<Json<Value>, ApiError> {
    let banks = s.store.list_banks()?;
    Ok(Json(json!({"banks": banks})))
}

#[derive(Deserialize, Default)]
struct BankBody {
    name: Option<String>,
    mission: Option<String>,
    disposition: Option<Value>,
}

async fn put_bank(
    State(s): State<S>,
    Path(bank_id): Path<String>,
    body: Option<Json<BankBody>>,
) -> Result<Json<Value>, ApiError> {
    let b = body.map(|Json(b)| b).unwrap_or_default();
    Store::validate_bank_id(&bank_id)?;
    s.store.ensure_bank(&bank_id)?;
    s.store.upsert_bank(
        &bank_id,
        &b.name.clone().unwrap_or_else(|| bank_id.clone()),
        &b.mission.clone().unwrap_or_default(),
        &b.disposition
            .map(|d| d.to_string())
            .unwrap_or_else(|| "{}".into()),
    )?;
    Ok(Json(json!({"bank_id": bank_id, "status": "ok"})))
}

async fn patch_bank(
    state: State<S>,
    path: Path<String>,
    body: Option<Json<BankBody>>,
) -> Result<Json<Value>, ApiError> {
    put_bank(state, path, body).await
}

async fn delete_bank(
    State(s): State<S>,
    Path(bank_id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    s.store.delete_bank(&bank_id)?;
    Ok(Json(json!({"bank_id": bank_id, "status": "deleted"})))
}

async fn bank_stats(
    State(s): State<S>,
    Path(bank_id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    Ok(Json(s.store.bank_stats(&bank_id)?))
}

// ---------- retain ----------

#[derive(Deserialize, serde::Serialize)]
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
struct RetainRequest {
    items: Vec<RetainItem>,
    #[serde(default)]
    #[serde(rename = "async")]
    run_async: bool,
}

async fn retain_memories(
    State(s): State<S>,
    Path(bank_id): Path<String>,
    Json(req): Json<RetainRequest>,
) -> Result<Json<Value>, ApiError> {
    let items = req.items;
    let items_count = items.len();
    if req.run_async {
        // Create operation row, spawn background task, return 202
        let op_id = s.store.create_operation(&bank_id, "retain", &json!({"items": items}))?;
        let state = Arc::clone(&s);
        let bank = bank_id.clone();
        let oid = op_id.clone();
        tokio::spawn(async move {
            let _ = state.store.update_operation_status(&bank, &oid, "running", None);
            let result = run_retain_background(&state.store, &state.llm, &bank, &items).await;
            match result {
                Ok(out) => {
                    let detail = json!({
                        "facts_extracted": out.fact_count,
                        "memory_ids": out.memory_ids,
                    });
                    let _ = state.store.update_operation_status(&bank, &oid, "completed", Some(&detail));
                }
                Err(e) => {
                    let detail = json!({"error": e.to_string()});
                    let _ = state.store.update_operation_status(&bank, &oid, "failed", Some(&detail));
                }
            }
        });
        Ok(Json(json!({
            "success": true,
            "bank_id": bank_id,
            "items_count": items_count,
            "async": true,
            "operation_id": op_id,
        })))
    } else {
        // Synchronous: run inline, return results
        for item in &items {
            let _out = engine::retain::retain(
                &s.store,
                &s.llm,
                &bank_id,
                &item.content,
                item.context.as_deref(),
                &item.tags,
                &item.metadata,
            )
            .await?;
        }
        Ok(Json(json!({
            "success": true,
            "bank_id": bank_id,
            "items_count": items.len(),
            "async": false,
            "operation_id": null,
        })))
    }
}

async fn run_retain_background(
    store: &Store,
    llm: &LlmClient,
    bank_id: &str,
    items: &[RetainItem],
) -> Result<engine::retain::RetainOutcome, anyhow::Error> {
    let mut all_ids = Vec::new();
    let mut total_facts = 0;
    for item in items {
        let out = engine::retain::retain(
            store,
            llm,
            bank_id,
            &item.content,
            item.context.as_deref(),
            &item.tags,
            &item.metadata,
        )
        .await?;
        total_facts += out.fact_count;
        all_ids.extend(out.memory_ids);
    }
    Ok(engine::retain::RetainOutcome {
        fact_count: total_facts,
        memory_ids: all_ids,
    })
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
    #[serde(default)]
    tags: Option<Vec<String>>,
    #[serde(default = "default_tags_match")]
    tags_match: String,
    #[serde(default)]
    tag_groups: Option<serde_json::Value>,
}
fn default_tags_match() -> String {
    "any".into()
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
    if req.tag_groups.is_some() {
        return Err(ApiError(anyhow::anyhow!("tag_groups not implemented (501)")));
    }
    let tags = req.tags.unwrap_or_default();
    let results = engine::recall::recall(
        &s.store,
        &s.llm,
        &bank_id,
        &req.query,
        &req.types.unwrap_or_default(),
        &req.budget,
        req.max_tokens,
        &tags,
        &req.tags_match,
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
    let bank = s.store.get_bank(&bank_id)?;
    let conn = bank.read_conn();
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

async fn get_memory(
    State(s): State<S>,
    Path((bank_id, memory_id)): Path<(String, String)>,
) -> Result<Response, ApiError> {
    let bank = s.store.get_bank(&bank_id)?;
    let conn = bank.read_conn();
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
        Err(_) => Ok((
            StatusCode::NOT_FOUND,
            Json(json!({"detail": "memory not found"})),
        )
            .into_response()),
    }
}

async fn delete_memory(
    State(s): State<S>,
    Path((bank_id, memory_id)): Path<(String, String)>,
) -> Result<Json<Value>, ApiError> {
    let bank = s.store.get_bank(&bank_id)?;
    let conn = bank.get_write_conn();
    conn.execute(
        "DELETE FROM memory_units WHERE bank_id=?1 AND id=?2",
        params![bank_id, memory_id],
    )?;
    Ok(Json(json!({"status": "deleted"})))
}

async fn delete_memories(
    State(s): State<S>,
    Path(bank_id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let bank = s.store.get_bank(&bank_id)?;
    let conn = bank.get_write_conn();
    let n = conn.execute(
        "DELETE FROM memory_units WHERE bank_id=?1",
        params![bank_id],
    )?;
    Ok(Json(json!({"deleted": n})))
}

async fn list_entities(
    State(s): State<S>,
    Path(bank_id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let bank = s.store.get_bank(&bank_id)?;
    let conn = bank.read_conn();
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

// ---------- operations ----------

#[derive(Deserialize)]
struct OperationsQuery {
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    #[serde(rename = "type")]
    op_type: Option<String>,
    #[serde(default = "default_ops_limit")]
    limit: i64,
    #[serde(default)]
    offset: i64,
}
fn default_ops_limit() -> i64 {
    20
}

async fn list_operations(
    State(s): State<S>,
    Path(bank_id): Path<String>,
    Query(q): Query<OperationsQuery>,
) -> Result<Json<Value>, ApiError> {
    let (ops, total) = s.store.list_operations(
        &bank_id,
        q.status.as_deref(),
        q.op_type.as_deref(),
        q.limit,
        q.offset,
    )?;
    Ok(Json(json!({
        "bank_id": bank_id,
        "total": total,
        "limit": q.limit,
        "offset": q.offset,
        "operations": ops,
    })))
}

async fn get_operation_status(
    State(s): State<S>,
    Path((bank_id, operation_id)): Path<(String, String)>,
) -> Result<Json<Value>, ApiError> {
    match s.store.get_operation(&bank_id, &operation_id)? {
        Some(op) => {
            let status = op["status"].as_str().unwrap_or("unknown");
            let detail = &op["detail"];
            let result_metadata = if status == "completed" || status == "failed" {
                Some(detail)
            } else {
                None
            };
            Ok(Json(json!({
                "operation_id": operation_id,
                "status": status,
                "operation_type": op["operation_type"],
                "created_at": op["created_at"],
                "updated_at": op["updated_at"],
                "completed_at": if status == "completed" || status == "failed" { op["updated_at"].clone() } else { Value::Null },
                "error_message": detail.get("error"),
                "result_metadata": result_metadata,
            })))
        }
        None => Ok(Json(json!({
            "operation_id": operation_id,
            "status": "not_found",
        }))),
    }
}

async fn cancel_operation(
    State(s): State<S>,
    Path((bank_id, operation_id)): Path<(String, String)>,
) -> Result<Json<Value>, ApiError> {
    let deleted = s.store.delete_operation(&bank_id, &operation_id)?;
    Ok(Json(json!({
        "success": deleted,
        "message": if deleted {
            format!("Operation {operation_id} cancelled")
        } else {
            format!("Operation {operation_id} not found or already running")
        },
        "operation_id": operation_id,
    })))
}

async fn retry_operation(
    State(s): State<S>,
    Path((bank_id, operation_id)): Path<(String, String)>,
) -> Result<Json<Value>, ApiError> {
    // Fetch original payload and op_type
    let op = s.store.get_operation(&bank_id, &operation_id)?
        .ok_or_else(|| anyhow::anyhow!("operation not found"))?;
    let status = op["status"].as_str().unwrap_or("");
    if status != "failed" {
        return Err(anyhow::anyhow!("can only retry failed operations").into());
    }
    let _op_type = op["operation_type"].as_str().unwrap_or("retain");
    let payload = s.store.get_operation_payload(&bank_id, &operation_id)?
        .unwrap_or(json!({}));
    // Reset to pending
    s.store.update_operation_status(&bank_id, &operation_id, "pending", None)?;
    // Spawn background task
    let state = Arc::clone(&s);
    let bank = bank_id.clone();
    let oid = operation_id.clone();
    let items_value = payload.get("items").cloned().unwrap_or(json!([]));
    tokio::spawn(async move {
        let _ = state.store.update_operation_status(&bank, &oid, "running", None);
        let items: Vec<RetainItem> = serde_json::from_value(items_value).unwrap_or_default();
        let result = run_retain_background(&state.store, &state.llm, &bank, &items).await;
        match result {
            Ok(out) => {
                let detail = json!({
                    "facts_extracted": out.fact_count,
                    "memory_ids": out.memory_ids,
                });
                let _ = state.store.update_operation_status(&bank, &oid, "completed", Some(&detail));
            }
            Err(e) => {
                let detail = json!({"error": e.to_string()});
                let _ = state.store.update_operation_status(&bank, &oid, "failed", Some(&detail));
            }
        }
    });
    Ok(Json(json!({
        "success": true,
        "message": format!("Operation {operation_id} queued for retry"),
        "operation_id": operation_id,
    })))
}

// ---------- vet ----------

#[derive(Deserialize)]
struct VetRequest {
    #[serde(default)]
    fix: bool,
}

async fn vet_bank(
    State(s): State<S>,
    Path(bank_id): Path<String>,
    Json(req): Json<VetRequest>,
) -> Result<Json<Value>, ApiError> {
    let engine = crate::vet::VetEngine::new()?;
    let items = s.store.scan_units_for_vet(&bank_id)?;
    let mut findings: Vec<serde_json::Value> = Vec::new();
    let mut fixed_count = 0u64;

    for item in &items {
        let result = engine.scan(&item.value);
        if result.detections.is_empty() {
            continue;
        }
        // Report rule_id + location only — never the secret itself
        for det in &result.detections {
            findings.push(json!({
                "rule_id": det.rule_id,
                "count": det.count,
                "source": item.source,
                "field": item.field,
                "unit_id": item.id,
            }));
        }

        if req.fix {
            let (redacted, _) = engine.vet_text(&item.value, crate::vet::VetMode::Redact)?;
            match item.field.as_str() {
                "text" => {
                    // Redact text + re-embed atomically
                    let emb = s.llm.embed(std::slice::from_ref(&redacted)).await?;
                    s.store.redact_unit(&bank_id, &item.id, &redacted, &emb[0])?;
                }
                "context" | "metadata" => {
                    // Update the specific column directly
                    s.store.quarantine_unit(&bank_id, &item.id)?;
                    let bank = s.store.get_bank(&bank_id)?;
                    let conn = bank.get_write_conn();
                    let sql = format!(
                        "UPDATE memory_units SET {}=?1 WHERE id=?2",
                        item.field
                    );
                    conn.execute(&sql, rusqlite::params![redacted, item.id])?;
                    s.store.unquarantine_unit(&bank_id, &item.id)?;
                }
                "content" => {
                    // Document content — update documents table
                    let bank = s.store.get_bank(&bank_id)?;
                    let conn = bank.get_write_conn();
                    conn.execute(
                        "UPDATE documents SET content=?1 WHERE id=?2",
                        rusqlite::params![redacted, item.id],
                    )?;
                }
                _ => {}
            }
            fixed_count += 1;
        }
    }

    Ok(Json(json!({
        "success": true,
        "bank_id": bank_id,
        "findings_count": findings.len(),
        "findings": findings,
        "fixed": fixed_count,
    })))
}
