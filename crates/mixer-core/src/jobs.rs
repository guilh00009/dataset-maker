//! Cancellation / pause / progress primitives shared by every long-running step.

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

#[derive(Debug, thiserror::Error)]
#[error("cancelled")]
pub struct Cancelled;

/// Control handle: cancel + pause.
#[derive(Clone, Default)]
pub struct Ctl {
    cancel: Arc<AtomicBool>,
    pause: Arc<AtomicBool>,
}

impl Ctl {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::SeqCst);
    }
    pub fn pause(&self) {
        self.pause.store(true, Ordering::SeqCst);
    }
    pub fn resume(&self) {
        self.pause.store(false, Ordering::SeqCst);
    }
    pub fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::SeqCst)
    }
    pub fn is_paused(&self) -> bool {
        self.pause.load(Ordering::SeqCst)
    }

    /// Blocking check (for sync code). Returns `Err(Cancelled)` if cancelled and
    /// blocks while paused.
    pub fn check(&self) -> Result<(), Cancelled> {
        while self.is_paused() && !self.is_cancelled() {
            std::thread::sleep(Duration::from_millis(100));
        }
        if self.is_cancelled() { Err(Cancelled) } else { Ok(()) }
    }

    /// Async check (for tokio code).
    pub async fn check_async(&self) -> Result<(), Cancelled> {
        while self.is_paused() && !self.is_cancelled() {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        if self.is_cancelled() { Err(Cancelled) } else { Ok(()) }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Update {
    pub phase: String,
    pub done: u64,
    pub total: u64,
    pub message: String,
    /// Free-form numeric stats (tokens/sec, failed count, …).
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub stats: serde_json::Map<String, serde_json::Value>,
}

impl Update {
    pub fn new(phase: &str, done: u64, total: u64, message: impl Into<String>) -> Self {
        Self { phase: phase.into(), done, total, message: message.into(), stats: Default::default() }
    }
}

/// Progress callback.
pub type Progress = Arc<dyn Fn(Update) + Send + Sync>;

pub fn no_progress() -> Progress {
    Arc::new(|_| {})
}
