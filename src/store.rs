//! SQLite storage layer. Replaces Hindsight's embedded PostgreSQL + pgvector.
//! - Vectors: f32-LE BLOBs, brute-force cosine (fine to ~100k units; swap to sqlite-vec later)
//! - Lexical: FTS5 with built-in BM25 (replaces ParadeDB/VectorChord pg_search)
//! - Graph: entities + unit_entities join tables (replaces PG entity_links)

use anyhow::Result;
use rusqlite::{params, Connection};
use serde_json::Value;
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
}
