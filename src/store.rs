//! SQLite storage layer. Replaces Hindsight's embedded PostgreSQL + pgvector.
//! - Vectors: f32-LE BLOBs, brute-force cosine (fine to ~100k units; swap to sqlite-vec later)
//! - Lexical: FTS5 with built-in BM25 (replaces ParadeDB/VectorChord pg_search)
//! - Graph: entities + unit_entities join tables (replaces PG entity_links)

use anyhow::Result;
use rusqlite::{params, Connection};
use serde_json::{json, Value};
use std::sync::Mutex;

pub struct Store {
    pub conn: Mutex<Connection>,
}

pub const SCHEMA: &str = r#"
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS banks (
  bank_id     TEXT PRIMARY KEY,
  name        TEXT,
  mission     TEXT DEFAULT '',
  disposition TEXT DEFAULT '{}',   -- JSON: {skepticism, literalism, empathy} 1-5
  config      TEXT DEFAULT '{}',
  created_at  TEXT NOT NULL,
  updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_units (
  id             TEXT PRIMARY KEY,
  bank_id        TEXT NOT NULL REFERENCES banks(bank_id) ON DELETE CASCADE,
  text           TEXT NOT NULL,
  fact_type      TEXT NOT NULL DEFAULT 'world',  -- world | experience | observation
  context        TEXT,
  occurred_start TEXT,
  occurred_end   TEXT,
  mentioned_at   TEXT,
  tags           TEXT NOT NULL DEFAULT '[]',
  metadata       TEXT NOT NULL DEFAULT '{}',
  document_id    TEXT,
  embedding      BLOB,
  created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_units_bank ON memory_units(bank_id, fact_type);
CREATE INDEX IF NOT EXISTS idx_units_time ON memory_units(bank_id, occurred_start);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
  text, content='memory_units', content_rowid='rowid', tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS units_ai AFTER INSERT ON memory_units BEGIN
  INSERT INTO memory_fts(rowid, text) VALUES (new.rowid, new.text);
END;
CREATE TRIGGER IF NOT EXISTS units_ad AFTER DELETE ON memory_units BEGIN
  INSERT INTO memory_fts(memory_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
END;
CREATE TRIGGER IF NOT EXISTS units_au AFTER UPDATE OF text ON memory_units BEGIN
  INSERT INTO memory_fts(memory_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
  INSERT INTO memory_fts(rowid, text) VALUES (new.rowid, new.text);
END;

CREATE TABLE IF NOT EXISTS entities (
  id             TEXT PRIMARY KEY,
  bank_id        TEXT NOT NULL REFERENCES banks(bank_id) ON DELETE CASCADE,
  canonical_name TEXT NOT NULL,
  created_at     TEXT NOT NULL,
  UNIQUE(bank_id, canonical_name)
);

CREATE TABLE IF NOT EXISTS unit_entities (
  unit_id   TEXT NOT NULL REFERENCES memory_units(id) ON DELETE CASCADE,
  entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
  PRIMARY KEY (unit_id, entity_id)
);
CREATE INDEX IF NOT EXISTS idx_ue_entity ON unit_entities(entity_id);

CREATE TABLE IF NOT EXISTS documents (
  id         TEXT PRIMARY KEY,
  bank_id    TEXT NOT NULL REFERENCES banks(bank_id) ON DELETE CASCADE,
  content    TEXT,
  metadata   TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS operations (
  id         TEXT PRIMARY KEY,
  bank_id    TEXT NOT NULL,
  op_type    TEXT NOT NULL,          -- retain | consolidate | refresh
  status     TEXT NOT NULL,          -- pending | running | completed | failed
  detail     TEXT DEFAULT '{}',
  payload    TEXT DEFAULT '{}',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
"#;

pub fn f32s_to_blob(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for f in v {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}

pub fn blob_to_f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

impl Store {
    pub fn open(path: &str) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.execute_batch(SCHEMA)?;
        // Migrate: add payload column if missing (guarded)
        let has_payload: bool = conn
            .prepare("PRAGMA table_info(operations)")?
            .query_map([], |row| row.get::<_, String>(1))? // column name at index 1
            .filter_map(|r| r.ok())
            .any(|name| name == "payload");
        if !has_payload {
            conn.execute_batch("ALTER TABLE operations ADD COLUMN payload TEXT DEFAULT '{}'")?;
        }
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    pub fn ensure_bank(&self, bank_id: &str) -> Result<()> {
        let now = chrono::Utc::now().to_rfc3339();
        self.conn.lock().unwrap().execute(
            "INSERT INTO banks(bank_id, name, created_at, updated_at) VALUES (?1, ?1, ?2, ?2)
             ON CONFLICT(bank_id) DO NOTHING",
            params![bank_id, now],
        )?;
        Ok(())
    }

    pub fn upsert_entity(&self, bank_id: &str, name: &str) -> Result<String> {
        let canonical = name.trim().to_lowercase();
        let conn = self.conn.lock().unwrap();
        let existing: Option<String> = conn
            .query_row(
                "SELECT id FROM entities WHERE bank_id=?1 AND canonical_name=?2",
                params![bank_id, canonical],
                |r| r.get(0),
            )
            .ok();
        if let Some(id) = existing {
            return Ok(id);
        }
        let id = uuid::Uuid::new_v4().to_string();
        conn.execute(
            "INSERT INTO entities(id, bank_id, canonical_name, created_at) VALUES (?1,?2,?3,?4)",
            params![id, bank_id, canonical, chrono::Utc::now().to_rfc3339()],
        )?;
        Ok(id)
    }

    pub fn bank_stats(&self, bank_id: &str) -> Result<Value> {
        let conn = self.conn.lock().unwrap();
        let count = |sql: &str| -> i64 {
            conn.query_row(sql, params![bank_id], |r| r.get(0))
                .unwrap_or(0)
        };
        Ok(serde_json::json!({
            "bank_id": bank_id,
            "total_memories": count("SELECT COUNT(*) FROM memory_units WHERE bank_id=?1"),
            "world_facts": count("SELECT COUNT(*) FROM memory_units WHERE bank_id=?1 AND fact_type='world'"),
            "experience_facts": count("SELECT COUNT(*) FROM memory_units WHERE bank_id=?1 AND fact_type='experience'"),
            "observations": count("SELECT COUNT(*) FROM memory_units WHERE bank_id=?1 AND fact_type='observation'"),
            "entities": count("SELECT COUNT(*) FROM entities WHERE bank_id=?1"),
            "documents": count("SELECT COUNT(*) FROM documents WHERE bank_id=?1"),
        }))
    }
    // ---- operations ----

    pub fn create_operation(
        &self,
        bank_id: &str,
        op_type: &str,
        payload: &Value,
    ) -> Result<String> {
        let id = uuid::Uuid::new_v4().to_string();
        let now = chrono::Utc::now().to_rfc3339();
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO operations(id, bank_id, op_type, status, detail, payload, created_at, updated_at)
             VALUES (?1, ?2, ?3, 'pending', '{}', ?4, ?5, ?5)",
            params![id, bank_id, op_type, payload.to_string(), now],
        )?;
        Ok(id)
    }

    pub fn update_operation_status(
        &self,
        operation_id: &str,
        status: &str,
        detail: Option<&Value>,
    ) -> Result<()> {
        let now = chrono::Utc::now().to_rfc3339();
        let conn = self.conn.lock().unwrap();
        if let Some(d) = detail {
            conn.execute(
                "UPDATE operations SET status=?1, detail=?2, updated_at=?3 WHERE id=?4",
                params![status, d.to_string(), now, operation_id],
            )?;
        } else {
            conn.execute(
                "UPDATE operations SET status=?1, updated_at=?2 WHERE id=?3",
                params![status, now, operation_id],
            )?;
        }
        Ok(())
    }

    pub fn get_operation(&self, operation_id: &str) -> Result<Option<Value>> {
        let conn = self.conn.lock().unwrap();
        let row = conn
            .query_row(
                "SELECT id, bank_id, op_type, status, detail, payload, created_at, updated_at
                 FROM operations WHERE id=?1",
                params![operation_id],
                |r| {
                    let id: String = r.get(0)?;
                    let bank_id: String = r.get(1)?;
                    let op_type: String = r.get(2)?;
                    let status: String = r.get(3)?;
                    let detail: String = r.get(4)?;
                    let payload: String = r.get(5)?;
                    let created_at: String = r.get(6)?;
                    let updated_at: String = r.get(7)?;
                    Ok(json!({
                        "operation_id": id,
                        "bank_id": bank_id,
                        "operation_type": op_type,
                        "status": status,
                        "detail": serde_json::from_str::<Value>(&detail).unwrap_or(json!({})),
                        "payload": serde_json::from_str::<Value>(&payload).unwrap_or(json!({})),
                        "created_at": created_at,
                        "updated_at": updated_at,
                    }))
                },
            )
            .ok();
        Ok(row)
    }

    pub fn list_operations(
        &self,
        bank_id: &str,
        status: Option<&str>,
        op_type: Option<&str>,
        limit: i64,
        offset: i64,
    ) -> Result<(Vec<Value>, i64)> {
        let conn = self.conn.lock().unwrap();
        let mut where_clauses = vec!["bank_id = ?1".to_string()];
        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = vec![Box::new(bank_id.to_string())];
        if let Some(s) = status {
            let idx = param_values.len() + 1;
            where_clauses.push(format!("status = ?{idx}"));
            param_values.push(Box::new(s.to_string()));
        }
        if let Some(t) = op_type {
            let idx = param_values.len() + 1;
            where_clauses.push(format!("op_type = ?{idx}"));
            param_values.push(Box::new(t.to_string()));
        }
        let where_sql = where_clauses.join(" AND ");
        let count_sql = format!("SELECT COUNT(*) FROM operations WHERE {where_sql}");
        let total: i64 = {
            let params_refs: Vec<&dyn rusqlite::types::ToSql> =
                param_values.iter().map(|p| p.as_ref()).collect();
            conn.query_row(&count_sql, params_refs.as_slice(), |r| r.get(0))?
        };
        let query_sql = format!(
            "SELECT id, bank_id, op_type, status, detail, created_at, updated_at
             FROM operations WHERE {where_sql}
             ORDER BY created_at DESC LIMIT ?{0} OFFSET ?{1}",
            param_values.len() + 1,
            param_values.len() + 2,
        );
        param_values.push(Box::new(limit));
        param_values.push(Box::new(offset));
        let params_refs: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|p| p.as_ref()).collect();
        let mut stmt = conn.prepare(&query_sql)?;
        let rows = stmt
            .query_map(params_refs.as_slice(), |r| {
                let id: String = r.get(0)?;
                let _bank_id: String = r.get(1)?;
                let op_type: String = r.get(2)?;
                let status: String = r.get(3)?;
                let _detail: String = r.get(4)?;
                let created_at: String = r.get(5)?;
                let updated_at: String = r.get(6)?;
                Ok(json!({
                    "id": id,
                    "task_type": op_type,
                    "items_count": 0,
                    "document_id": null,
                    "filename": null,
                    "created_at": created_at,
                    "updated_at": updated_at,
                    "status": status,
                    "error_message": null,
                }))
            })?
            .filter_map(|r| r.ok())
            .collect::<Vec<_>>();
        Ok((rows, total))
    }

    pub fn delete_operation(&self, operation_id: &str) -> Result<bool> {
        let conn = self.conn.lock().unwrap();
        let affected = conn.execute(
            "DELETE FROM operations WHERE id=?1 AND status NOT IN ('running')",
            params![operation_id],
        )?;
        Ok(affected > 0)
    }

    pub fn get_operation_payload(&self, operation_id: &str) -> Result<Option<Value>> {
        let conn = self.conn.lock().unwrap();
        let row = conn
            .query_row(
                "SELECT payload FROM operations WHERE id=?1",
                params![operation_id],
                |r| {
                    let payload: String = r.get(0)?;
                    Ok(serde_json::from_str::<Value>(&payload).unwrap_or(json!({})))
                },
            )
            .ok();
        Ok(row)
    }

}
