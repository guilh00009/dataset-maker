//! SQLite journal of finished translation units. Content-addressed, so a re-run (after a
//! crash, a cancel, or a different mix of the same data) never pays twice for the same text.

use anyhow::Result;
use rusqlite::{Connection, OptionalExtension, params};
use std::path::Path;
use std::sync::Mutex;

pub struct Journal {
    conn: Mutex<Connection>,
}

impl Journal {
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(p) = path.parent() {
            std::fs::create_dir_all(p)?;
        }
        let conn = Connection::open(path)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS units (
                key INTEGER PRIMARY KEY,
                text TEXT NOT NULL,
                created INTEGER NOT NULL DEFAULT (strftime('%s','now'))
            );",
        )?;
        Ok(Self { conn: Mutex::new(conn) })
    }

    pub fn open_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch("CREATE TABLE units (key INTEGER PRIMARY KEY, text TEXT NOT NULL, created INTEGER NOT NULL DEFAULT 0);")?;
        Ok(Self { conn: Mutex::new(conn) })
    }

    pub fn get(&self, key: u64) -> Option<String> {
        let c = self.conn.lock().ok()?;
        c.query_row("SELECT text FROM units WHERE key = ?1", params![key as i64], |r| r.get::<_, String>(0)).optional().ok().flatten()
    }

    pub fn put(&self, key: u64, text: &str) -> Result<()> {
        let c = self.conn.lock().map_err(|_| anyhow::anyhow!("journal poisoned"))?;
        c.execute("INSERT OR REPLACE INTO units (key, text) VALUES (?1, ?2)", params![key as i64, text])?;
        Ok(())
    }

    pub fn count(&self) -> u64 {
        self.conn
            .lock()
            .ok()
            .and_then(|c| c.query_row("SELECT COUNT(*) FROM units", [], |r| r.get::<_, i64>(0)).ok())
            .unwrap_or(0) as u64
    }

    pub fn clear(&self) -> Result<()> {
        let c = self.conn.lock().map_err(|_| anyhow::anyhow!("journal poisoned"))?;
        c.execute("DELETE FROM units", [])?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persists_across_reopen() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("j.sqlite");
        {
            let j = Journal::open(&p).unwrap();
            assert!(j.get(1).is_none());
            j.put(1, "olá").unwrap();
            j.put(u64::MAX, "max").unwrap();
        }
        let j = Journal::open(&p).unwrap();
        assert_eq!(j.get(1).as_deref(), Some("olá"));
        assert_eq!(j.get(u64::MAX).as_deref(), Some("max"));
        assert_eq!(j.count(), 2);
    }
}
