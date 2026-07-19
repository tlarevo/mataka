//! SQLite storage layer — per-bank sharded files + read pool.
//! Replaces Hindsight's embedded PostgreSQL + pgvector.
//! - Vectors: f32-LE BLOBs, brute-force cosine (fine to ~100k units; swap to sqlite-vec later)
//! - Lexical: FTS5 with built-in BM25 (replaces ParadeDB/VectorChord pg_search)
//! - Graph: entities + unit_entities join tables (replaces PG entity_links)
//!
//! Each bank gets its own SQLite file (`{data_dir}/{bank_id}.db`) with:
//!   - 1 write connection behind a Mutex (PRAGMA busy_timeout=5000, WAL)
//!   - N=4 read-only connections behind a simple free-list
//!
//! A tiny registry DB (`{data_dir}/_registry.db`) holds the banks table
//! for cross-bank listing. Legacy single-file mode (MATAKA_DB=path/to/file.db)
//! falls back to the original Mutex<Connection> for backward compatibility.

use anyhow::Result;
use rusqlite::{params, Connection};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use parking_lot::{Mutex, RwLock};
use std::sync::Arc;

// ─── Per-bank schema (identical to original, minus banks table) ──────────

pub const BANK_SCHEMA: &str = r#"
PRAGMA journal_mode=WAL;
PRAGMA busy_timeout=5000;

CREATE TABLE IF NOT EXISTS memory_units (
  id             TEXT PRIMARY KEY,
  bank_id        TEXT NOT NULL,
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
  bank_id        TEXT NOT NULL,
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
  bank_id    TEXT NOT NULL,
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

CREATE TABLE IF NOT EXISTS observation_sources (
  observation_id TEXT NOT NULL REFERENCES memory_units(id) ON DELETE CASCADE,
  source_unit_id TEXT NOT NULL REFERENCES memory_units(id) ON DELETE CASCADE,
  PRIMARY KEY (observation_id, source_unit_id)
);
CREATE INDEX IF NOT EXISTS idx_os_source ON observation_sources(source_unit_id);
"#;

// ─── Legacy single-file Store (backward compat for :memory: / single .db) ──

pub struct LegacyStore {
    pub conn: Mutex<Connection>,
}

impl LegacyStore {
    pub fn open(path: &str) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.execute_batch(LEGACY_SCHEMA)?;
        let has_payload: bool = conn
            .prepare("PRAGMA table_info(operations)")?
            .query_map([], |row| row.get::<_, String>(1))?
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
        self.conn.lock().execute(
            "INSERT INTO banks(bank_id, name, created_at, updated_at) VALUES (?1, ?1, ?2, ?2)
             ON CONFLICT(bank_id) DO NOTHING",
            params![bank_id, now],
        )?;
        Ok(())
    }

    pub fn upsert_entity(&self, bank_id: &str, name: &str) -> Result<String> {
        let canonical = name.trim().to_lowercase();
        let conn = self.conn.lock();
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
        let conn = self.conn.lock();
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

    pub fn create_operation(
        &self,
        bank_id: &str,
        op_type: &str,
        payload: &Value,
    ) -> Result<String> {
        let id = uuid::Uuid::new_v4().to_string();
        let now = chrono::Utc::now().to_rfc3339();
        let conn = self.conn.lock();
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
        let conn = self.conn.lock();
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
        let conn = self.conn.lock();
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
        let conn = self.conn.lock();
        let mut where_clauses = vec!["bank_id = ?1".to_string()];
        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> =
            vec![Box::new(bank_id.to_string())];
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
        let conn = self.conn.lock();
        let affected = conn.execute(
            "DELETE FROM operations WHERE id=?1 AND status NOT IN ('running')",
            params![operation_id],
        )?;
        Ok(affected > 0)
    }

    pub fn get_operation_payload(&self, operation_id: &str) -> Result<Option<Value>> {
        let conn = self.conn.lock();
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
    pub fn upsert_bank(
        &self,
        bank_id: &str,
        name: &str,
        mission: &str,
        disposition: &str,
    ) -> Result<()> {
        let now = chrono::Utc::now().to_rfc3339();
        self.conn.lock().execute(
            "INSERT INTO banks(bank_id, name, mission, disposition, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?5)
             ON CONFLICT(bank_id) DO UPDATE SET
               name=COALESCE(excluded.name, name),
               mission=COALESCE(NULLIF(excluded.mission,''), mission),
               disposition=excluded.disposition,
               updated_at=excluded.updated_at",
            params![bank_id, name, mission, disposition, now],
        )?;
        Ok(())
    }

    pub fn get_bank_info(&self, bank_id: &str) -> Result<Option<Value>> {
        let row = self
            .conn
            .lock()
            .query_row(
                "SELECT bank_id, name, mission, disposition, created_at FROM banks WHERE bank_id=?1",
                params![bank_id],
                |r| {
                    Ok(json!({
                        "bank_id": r.get::<_, String>(0)?,
                        "name": r.get::<_, Option<String>>(1)?,
                        "mission": r.get::<_, String>(2)?,
                        "disposition": serde_json::from_str::<Value>(&r.get::<_, String>(3)?).unwrap_or(json!({})),
                        "created_at": r.get::<_, String>(4)?,
                    }))
                },
            )
            .ok();
        Ok(row)
    }

    pub fn list_banks(&self) -> Result<Vec<Value>> {
        let conn = self.conn.lock();
        let mut stmt =
            conn.prepare("SELECT bank_id, name, mission, created_at FROM banks ORDER BY created_at")?;
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
        Ok(banks)
    }

    pub fn delete_bank(&self, bank_id: &str) -> Result<()> {
        self.conn
            .lock()
            .execute("DELETE FROM banks WHERE bank_id=?1", params![bank_id])?;
        Ok(())
    }
}

const LEGACY_SCHEMA: &str = r#"
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS banks (
  bank_id     TEXT PRIMARY KEY,
  name        TEXT,
  mission     TEXT DEFAULT '',
  disposition TEXT DEFAULT '{}',
  config      TEXT DEFAULT '{}',
  created_at  TEXT NOT NULL,
  updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_units (
  id             TEXT PRIMARY KEY,
  bank_id        TEXT NOT NULL REFERENCES banks(bank_id) ON DELETE CASCADE,
  text           TEXT NOT NULL,
  fact_type      TEXT NOT NULL DEFAULT 'world',
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
  op_type    TEXT NOT NULL,
  status     TEXT NOT NULL,
  detail     TEXT DEFAULT '{}',
  payload    TEXT DEFAULT '{}',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS observation_sources (
  observation_id TEXT NOT NULL REFERENCES memory_units(id) ON DELETE CASCADE,
  source_unit_id TEXT NOT NULL REFERENCES memory_units(id) ON DELETE CASCADE,
  PRIMARY KEY (observation_id, source_unit_id)
);
CREATE INDEX IF NOT EXISTS idx_os_source ON observation_sources(source_unit_id);
"#;

// ─── Per-bank database ──────────────────────────────────────────────────

pub struct BankDb {
    write_conn: Mutex<Connection>,
    read_conns: Vec<Mutex<Connection>>,
    next_read: std::sync::atomic::AtomicUsize,
    _bank_dir: PathBuf,
}

const READ_POOL_SIZE: usize = 4;

impl BankDb {
    fn new(path: &Path) -> Result<Self> {
        // Write connection
        let wc = Connection::open(path)?;
        wc.execute_batch("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;")?;
        wc.execute_batch(BANK_SCHEMA)?;

        // Read pool — each connection independently lockable for true concurrency
        let mut rc = Vec::with_capacity(READ_POOL_SIZE);
        for _ in 0..READ_POOL_SIZE {
            let url = format!("file:{}?mode=ro", path.display());
            let conn = Connection::open_with_flags(
                &url,
                rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY
                    | rusqlite::OpenFlags::SQLITE_OPEN_URI,
            )?;
            rc.push(Mutex::new(conn));
        }

        Ok(Self {
            write_conn: Mutex::new(wc),
            read_conns: rc,
            next_read: std::sync::atomic::AtomicUsize::new(0),
            _bank_dir: path
                .parent()
                .unwrap_or(Path::new("."))
                .to_path_buf(),
        })
    }

    /// Round-robin a read connection. Each call returns a different mutex guard,
    /// enabling true concurrent reads across the pool.
    pub fn read_conn(&self) -> parking_lot::MutexGuard<'_, Connection> {
        use std::sync::atomic::Ordering;
        let idx = self.next_read.fetch_add(1, Ordering::Relaxed) % self.read_conns.len();
        self.read_conns[idx].lock()
    }

    /// Acquire the write connection.
    pub fn get_write_conn(&self) -> parking_lot::MutexGuard<'_, Connection> {
        self.write_conn.lock()
    }
}

// ─── Store: bank registry + lazy bank opener ────────────────────────────

pub struct Store {
    data_dir: PathBuf,
    registry_conn: Mutex<Connection>,
    banks: RwLock<HashMap<String, Arc<BankDb>>>,
    legacy: Option<LegacyStore>,
}

impl Store {
    /// Open in per-bank sharded mode. `data_dir` is the directory for all `.db` files.
    pub fn open(data_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(data_dir)?;
        let reg_path = data_dir.join("_registry.db");
        let rc = Connection::open(&reg_path)?;
        rc.execute_batch("PRAGMA journal_mode=WAL;")?;
        rc.execute_batch(
            "CREATE TABLE IF NOT EXISTS banks (
                bank_id     TEXT PRIMARY KEY,
                name        TEXT,
                mission     TEXT DEFAULT '',
                disposition TEXT DEFAULT '{}',
                config      TEXT DEFAULT '{}',
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );",
        )?;
        Ok(Self {
            data_dir: data_dir.to_path_buf(),
            registry_conn: Mutex::new(rc),
            banks: RwLock::new(HashMap::new()),
            legacy: None,
        })
    }

    /// Open in legacy single-file mode for backward compatibility.
    pub fn open_legacy(db_path: &str) -> Result<Self> {
        let legacy = LegacyStore::open(db_path)?;
        let stub = Connection::open_in_memory()?;
        Ok(Self {
            data_dir: std::path::PathBuf::new(),
            registry_conn: Mutex::new(stub),
            banks: RwLock::new(HashMap::new()),
            legacy: Some(legacy),
        })
    }

    /// Get or open a BankDb for the given bank. Lazily creates on first access.
    pub fn get_bank(&self, bank_id: &str) -> Result<Arc<BankDb>> {
        if self.legacy.is_some() {
            anyhow::bail!("get_bank not available in legacy mode");
        }
        {
            let map = self.banks.read();
            if let Some(b) = map.get(bank_id) {
                return Ok(Arc::clone(b));
            }
        }
        let path = self.data_dir.join(format!("{bank_id}.db"));
        let bank = Arc::new(BankDb::new(&path)?);
        let mut map = self.banks.write();
        if let Some(b) = map.get(bank_id) {
            return Ok(Arc::clone(b));
        }
        map.insert(bank_id.to_string(), Arc::clone(&bank));
        Ok(bank)
    }

    /// Sanitize bank_id: allow [A-Za-z0-9._-] only.
    pub fn validate_bank_id(id: &str) -> Result<()> {
        if id.is_empty() {
            anyhow::bail!("bank_id cannot be empty");
        }
        if !id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'_' || b == b'-')
        {
            anyhow::bail!("bank_id contains invalid characters (only [A-Za-z0-9._-] allowed)");
        }
        Ok(())
    }

    // ---- Registry operations (banks table lives in _registry.db) ----

    pub fn ensure_bank(&self, bank_id: &str) -> Result<()> {
        if let Some(legacy) = &self.legacy {
            return legacy.ensure_bank(bank_id);
        }
        let now = chrono::Utc::now().to_rfc3339();
        self.registry_conn.lock().execute(
            "INSERT INTO banks(bank_id, name, created_at, updated_at) VALUES (?1, ?1, ?2, ?2)
             ON CONFLICT(bank_id) DO NOTHING",
            params![bank_id, now],
        )?;
        Ok(())
    }

    pub fn upsert_bank(
        &self,
        bank_id: &str,
        name: &str,
        mission: &str,
        disposition: &str,
    ) -> Result<()> {
        if let Some(legacy) = &self.legacy {
            return legacy.upsert_bank(bank_id, name, mission, disposition);
        }
        let now = chrono::Utc::now().to_rfc3339();
        self.registry_conn.lock().execute(
            "INSERT INTO banks(bank_id, name, mission, disposition, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?5)
             ON CONFLICT(bank_id) DO UPDATE SET
               name=COALESCE(excluded.name, name),
               mission=COALESCE(NULLIF(excluded.mission,''), mission),
               disposition=excluded.disposition,
               updated_at=excluded.updated_at",
            params![bank_id, name, mission, disposition, now],
        )?;
        Ok(())
    }

    pub fn get_bank_info(&self, bank_id: &str) -> Result<Option<Value>> {
        if let Some(legacy) = &self.legacy {
            return legacy.get_bank_info(bank_id);
        }
        let row = self
            .registry_conn
            .lock()
            .query_row(
                "SELECT bank_id, name, mission, disposition, created_at FROM banks WHERE bank_id=?1",
                params![bank_id],
                |r| {
                    Ok(json!({
                        "bank_id": r.get::<_, String>(0)?,
                        "name": r.get::<_, Option<String>>(1)?,
                        "mission": r.get::<_, String>(2)?,
                        "disposition": serde_json::from_str::<Value>(&r.get::<_, String>(3)?).unwrap_or(json!({})),
                        "created_at": r.get::<_, String>(4)?,
                    }))
                },
            )
            .ok();
        Ok(row)
    }

    pub fn list_banks(&self) -> Result<Vec<Value>> {
        if let Some(legacy) = &self.legacy {
            return legacy.list_banks();
        }
        let conn = self.registry_conn.lock();
        let mut stmt =
            conn.prepare("SELECT bank_id, name, mission, created_at FROM banks ORDER BY created_at")?;
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
        Ok(banks)
    }

    pub fn delete_bank(&self, bank_id: &str) -> Result<()> {
        if let Some(legacy) = &self.legacy {
            return legacy.delete_bank(bank_id);
        }
        // 1. Remove from registry
        self.registry_conn
            .lock()
            .execute("DELETE FROM banks WHERE bank_id=?1", params![bank_id])?;
        // 2. Close cached BankDb (drop all connections)
        self.banks.write().remove(bank_id);
        // 3. Remove files from disk (best-effort)
        let base = self.data_dir.join(bank_id);
        let _ = std::fs::remove_file(base.with_extension("db"));
        let _ = std::fs::remove_file(base.with_extension("db-wal"));
        let _ = std::fs::remove_file(base.with_extension("db-shm"));
        Ok(())
    }

    // ---- Per-bank data operations (route to BankDb) ----

    pub fn upsert_entity(&self, bank_id: &str, name: &str) -> Result<String> {
        if let Some(legacy) = &self.legacy {
            return legacy.upsert_entity(bank_id, name);
        }
        let canonical = name.trim().to_lowercase();
        let bank = self.get_bank(bank_id)?;
        let conn = bank.read_conn();
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
        drop(conn);
        let wconn = bank.get_write_conn();
        wconn.execute(
            "INSERT INTO entities(id, bank_id, canonical_name, created_at) VALUES (?1,?2,?3,?4)",
            params![id, bank_id, canonical, chrono::Utc::now().to_rfc3339()],
        )?;
        Ok(id)
    }

    pub fn bank_stats(&self, bank_id: &str) -> Result<Value> {
        if let Some(legacy) = &self.legacy {
            return legacy.bank_stats(bank_id);
        }
        let bank = self.get_bank(bank_id)?;
        let conn = bank.read_conn();
        let count = |sql: &str| -> i64 {
            conn.query_row(sql, params![bank_id], |r| r.get(0))
                .unwrap_or(0)
        };
        Ok(json!({
            "bank_id": bank_id,
            "total_memories": count("SELECT COUNT(*) FROM memory_units WHERE bank_id=?1"),
            "world_facts": count("SELECT COUNT(*) FROM memory_units WHERE bank_id=?1 AND fact_type='world'"),
            "experience_facts": count("SELECT COUNT(*) FROM memory_units WHERE bank_id=?1 AND fact_type='experience'"),
            "observations": count("SELECT COUNT(*) FROM memory_units WHERE bank_id=?1 AND fact_type='observation'"),
            "entities": count("SELECT COUNT(*) FROM entities WHERE bank_id=?1"),
            "documents": count("SELECT COUNT(*) FROM documents WHERE bank_id=?1"),
        }))
    }

    // ---- Operations (per-bank) ----

    pub fn create_operation(
        &self,
        bank_id: &str,
        op_type: &str,
        payload: &Value,
    ) -> Result<String> {
        if let Some(legacy) = &self.legacy {
            return legacy.create_operation(bank_id, op_type, payload);
        }
        let id = uuid::Uuid::new_v4().to_string();
        let now = chrono::Utc::now().to_rfc3339();
        let bank = self.get_bank(bank_id)?;
        let conn = bank.get_write_conn();
        conn.execute(
            "INSERT INTO operations(id, bank_id, op_type, status, detail, payload, created_at, updated_at)
             VALUES (?1, ?2, ?3, 'pending', '{}', ?4, ?5, ?5)",
            params![id, bank_id, op_type, payload.to_string(), now],
        )?;
        Ok(id)
    }

    pub fn update_operation_status(
        &self,
        bank_id: &str,
        operation_id: &str,
        status: &str,
        detail: Option<&Value>,
    ) -> Result<()> {
        if let Some(legacy) = &self.legacy {
            return legacy.update_operation_status(operation_id, status, detail);
        }
        let now = chrono::Utc::now().to_rfc3339();
        let bank = self.get_bank(bank_id)?;
        let conn = bank.get_write_conn();
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

    pub fn get_operation(&self, bank_id: &str, operation_id: &str) -> Result<Option<Value>> {
        if let Some(legacy) = &self.legacy {
            return legacy.get_operation(operation_id);
        }
        let bank = self.get_bank(bank_id)?;
        let conn = bank.read_conn();
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
        if let Some(legacy) = &self.legacy {
            return legacy.list_operations(bank_id, status, op_type, limit, offset);
        }
        let bank = self.get_bank(bank_id)?;
        let conn = bank.read_conn();
        let mut where_clauses = vec!["bank_id = ?1".to_string()];
        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> =
            vec![Box::new(bank_id.to_string())];
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

    pub fn delete_operation(&self, bank_id: &str, operation_id: &str) -> Result<bool> {
        if let Some(legacy) = &self.legacy {
            return legacy.delete_operation(operation_id);
        }
        let bank = self.get_bank(bank_id)?;
        let conn = bank.get_write_conn();
        let affected = conn.execute(
            "DELETE FROM operations WHERE id=?1 AND status NOT IN ('running')",
            params![operation_id],
        )?;
        Ok(affected > 0)
    }

    pub fn get_operation_payload(
        &self,
        bank_id: &str,
        operation_id: &str,
    ) -> Result<Option<Value>> {
        if let Some(legacy) = &self.legacy {
            return legacy.get_operation_payload(operation_id);
        }
        let bank = self.get_bank(bank_id)?;
        let conn = bank.read_conn();
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

// ─── Helpers ────────────────────────────────────────────────────────────

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_bank_id_ok() {
        assert!(Store::validate_bank_id("my-bank_1.0").is_ok());
        assert!(Store::validate_bank_id("abc").is_ok());
    }

    #[test]
    fn validate_bank_id_rejects() {
        assert!(Store::validate_bank_id("").is_err());
        assert!(Store::validate_bank_id("has space").is_err());
        assert!(Store::validate_bank_id("slash/").is_err());
        assert!(Store::validate_bank_id("special!@#").is_err());
    }

    #[test]
    fn cosine_identical() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((cosine(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((cosine(&a, &b)).abs() < 1e-6);
    }

    #[test]
    fn f32_blob_roundtrip() {
        let v = vec![1.0f32, -2.5, 0.0];
        let b = f32s_to_blob(&v);
        let v2 = blob_to_f32s(&b);
        assert_eq!(v, v2);
    }
}
