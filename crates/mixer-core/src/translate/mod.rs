//! The translation engine: many concurrent requests to an OpenAI-compatible server, within
//! limits, with chunking, validation, retries and a resumable content-addressed journal.

pub mod client;
pub mod estimate;
pub mod journal;
pub mod prompt;
pub mod protect;

use crate::export::{SampleTranslator, Translated};
use crate::jobs::{Ctl, Progress, Update};
use crate::mix::{MixPlan, SourceData};
use crate::model::{Role, Sample, est_tokens};
use crate::recipe::{Recipe, Structured, TranslateOptions};
use anyhow::{Result, bail};
use client::{ChatRequest, ChatResponse, LlmClient, LlmError};
use futures::future::{BoxFuture, join_all};
use futures::stream::StreamExt;
use governor::{Quota, RateLimiter, clock::DefaultClock, state::{InMemoryState, NotKeyed}};
use journal::Journal;
use prompt::{Language, PROMPT_VERSION};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{HashMap, VecDeque};
use std::num::NonZeroU32;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};
use tokio::sync::{OnceCell, Semaphore};
use xxhash_rust::xxh3::xxh3_64;

type Limiter = RateLimiter<NotKeyed, InMemoryState, DefaultClock>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Failure {
    pub source: usize,
    pub row: u64,
    pub reason: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StatsSnapshot {
    pub requests: u64,
    pub retries: u64,
    pub http_errors: u64,
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub tokens_per_sec: f64,
    pub samples_done: u64,
    pub samples_failed: u64,
    pub units_cached: u64,
    pub units_translated: u64,
    pub in_flight: i64,
    pub elapsed_secs: f64,
    pub last_error: String,
    pub recent_failures: Vec<Failure>,
}

#[derive(Default)]
struct Stats {
    requests: AtomicU64,
    retries: AtomicU64,
    http_errors: AtomicU64,
    tokens_in: AtomicU64,
    tokens_out: AtomicU64,
    samples_done: AtomicU64,
    samples_failed: AtomicU64,
    units_cached: AtomicU64,
    units_translated: AtomicU64,
    in_flight: AtomicI64,
    last_error: Mutex<String>,
    failures: Mutex<VecDeque<Failure>>,
    first_request: OnceLock<Instant>,
}

struct CallError {
    msg: String,
    fatal: bool,
}

/// Parts of the cache key that depend only on the translation settings.
#[derive(Clone)]
pub struct KeyCtx {
    cfg_hash: u64,
}

impl KeyCtx {
    pub fn new(t: &TranslateOptions) -> Self {
        let s = format!(
            "v{PROMPT_VERSION}|{}|{}|{}|{}|{}|{:.2}",
            t.source_language.trim().to_lowercase(),
            t.target_language.trim().to_lowercase(),
            t.model.trim(),
            t.instructions.trim(),
            t.glossary.trim(),
            t.temperature
        );
        Self { cfg_hash: xxh3_64(s.as_bytes()) }
    }
    pub fn seg_key(&self, seg: &str, names_hash: u64) -> u64 {
        xxh3_64(&[seg.as_bytes(), &self.cfg_hash.to_le_bytes(), &names_hash.to_le_bytes()].concat())
    }
}

pub fn names_hash(names: &[String]) -> u64 {
    xxh3_64(names.join("\u{1}").as_bytes())
}

/// Persona template variables that end up in the exported text (only these are translated).
pub fn persona_fields_in_template(template: &str) -> Vec<String> {
    let rx = regex::Regex::new(r"\{\{\s*([a-z_]+)\s*\}\}").unwrap();
    rx.captures_iter(template).map(|c| c[1].to_string()).collect()
}

pub struct Translator {
    cfg: TranslateOptions,
    client: LlmClient,
    journal: Arc<Journal>,
    sem: Semaphore,
    rpm: Option<Limiter>,
    tpm: Option<(Limiter, u32)>,
    stats: Stats,
    ctl: Ctl,
    keys: KeyCtx,
    src: Language,
    tgt: Language,
    persona_fields: Vec<String>,
    structured_ok: AtomicBool,
    fatal: Mutex<Option<String>>,
    budget_hit: AtomicBool,
    fields: Mutex<HashMap<u64, Arc<OnceCell<std::result::Result<String, String>>>>>,
}

impl Translator {
    pub fn new(recipe: &Recipe, token: Option<String>, journal: Arc<Journal>, ctl: Ctl) -> Result<Arc<Self>> {
        let cfg = recipe.translate.clone();
        if cfg.base_url.trim().is_empty() {
            bail!("set the LLM base URL first (for llama.cpp: http://127.0.0.1:8080/v1)");
        }
        let client = LlmClient::new(&cfg.base_url, &cfg.model, token, Duration::from_secs(cfg.timeout_secs.max(5)))?;
        let rpm = NonZeroU32::new(cfg.requests_per_minute).map(|n| RateLimiter::direct(Quota::per_minute(n)));
        let tpm = NonZeroU32::new(cfg.tokens_per_minute).map(|n| (RateLimiter::direct(Quota::per_minute(n)), n.get()));
        let conc = cfg.concurrency.clamp(1, 256);
        Ok(Arc::new(Self {
            src: prompt::language(&cfg.source_language),
            tgt: prompt::language(&cfg.target_language),
            keys: KeyCtx::new(&cfg),
            persona_fields: persona_fields_in_template(&recipe.persona.template),
            structured_ok: AtomicBool::new(cfg.structured != Structured::Off),
            sem: Semaphore::new(conc),
            client,
            journal,
            rpm,
            tpm,
            stats: Stats::default(),
            ctl,
            cfg,
            fatal: Mutex::new(None),
            budget_hit: AtomicBool::new(false),
            fields: Mutex::new(HashMap::new()),
        }))
    }

    pub fn snapshot(&self) -> StatsSnapshot {
        let s = &self.stats;
        let elapsed = s.first_request.get().map_or(0.0, |t| t.elapsed().as_secs_f64());
        let tokens_out = s.tokens_out.load(Ordering::Relaxed);
        StatsSnapshot {
            requests: s.requests.load(Ordering::Relaxed),
            retries: s.retries.load(Ordering::Relaxed),
            http_errors: s.http_errors.load(Ordering::Relaxed),
            tokens_in: s.tokens_in.load(Ordering::Relaxed),
            tokens_out,
            tokens_per_sec: if elapsed > 1.0 { tokens_out as f64 / elapsed } else { 0.0 },
            samples_done: s.samples_done.load(Ordering::Relaxed),
            samples_failed: s.samples_failed.load(Ordering::Relaxed),
            units_cached: s.units_cached.load(Ordering::Relaxed),
            units_translated: s.units_translated.load(Ordering::Relaxed),
            in_flight: s.in_flight.load(Ordering::Relaxed),
            elapsed_secs: elapsed,
            last_error: s.last_error.lock().map(|e| e.clone()).unwrap_or_default(),
            recent_failures: s.failures.lock().map(|f| f.iter().cloned().collect()).unwrap_or_default(),
        }
    }

    /// A configuration error (bad token, unknown endpoint) or the token budget stopped the run.
    pub fn stop_reason(&self) -> Option<String> {
        if let Some(f) = self.fatal.lock().ok().and_then(|f| f.clone()) {
            return Some(f);
        }
        if self.budget_hit.load(Ordering::Relaxed) {
            return Some("token budget reached".into());
        }
        None
    }

    fn payload_budget(&self) -> usize {
        ((self.cfg.context_budget as f64 - 350.0) / 2.3).max(200.0) as usize
    }

    fn out_tokens(&self, est_total_in: usize, est_payload: usize) -> u32 {
        if self.cfg.max_output_tokens > 0 {
            return self.cfg.max_output_tokens;
        }
        let want = (est_payload as f64 * 2.2) as u32 + 96;
        let room = (self.cfg.context_budget as i64 - est_total_in as i64).max(64) as u32;
        want.min(room).max(64)
    }

    fn note_error(&self, msg: &str) {
        if let Ok(mut e) = self.stats.last_error.lock() {
            *e = msg.chars().take(400).collect();
        }
    }

    /// One HTTP call with concurrency limit, rate limits, pause/cancel and retries.
    async fn call(&self, mut req: ChatRequest) -> std::result::Result<ChatResponse, CallError> {
        let est_in: usize = req.messages.iter().map(|(_, c)| est_tokens(c) + 4).sum();
        let mut attempt = 0u32;
        loop {
            if self.ctl.check_async().await.is_err() {
                return Err(CallError { msg: "cancelled".into(), fatal: true });
            }
            if let Some(f) = self.stop_reason() {
                return Err(CallError { msg: f, fatal: true });
            }
            if self.cfg.token_budget > 0
                && self.stats.tokens_in.load(Ordering::Relaxed) + self.stats.tokens_out.load(Ordering::Relaxed) >= self.cfg.token_budget
            {
                self.budget_hit.store(true, Ordering::Relaxed);
                return Err(CallError { msg: "token budget reached".into(), fatal: true });
            }
            if let Some(l) = &self.rpm {
                l.until_ready().await;
            }
            if let Some((l, burst)) = &self.tpm {
                let n = (est_in as u32 + req.max_tokens / 2).clamp(1, *burst);
                let _ = l.until_n_ready(NonZeroU32::new(n).unwrap()).await;
            }
            let result = {
                let _permit = self.sem.acquire().await.expect("semaphore closed");
                self.stats.first_request.get_or_init(Instant::now);
                self.stats.requests.fetch_add(1, Ordering::Relaxed);
                self.stats.in_flight.fetch_add(1, Ordering::Relaxed);
                let r = self.client.chat(&req).await;
                self.stats.in_flight.fetch_sub(1, Ordering::Relaxed);
                r
            };
            match result {
                Ok(r) => {
                    let tin = r.prompt_tokens.map_or(est_in as u64, |t| t as u64);
                    let tout = r.completion_tokens.map_or_else(|| est_tokens(&r.content) as u64, |t| t as u64);
                    self.stats.tokens_in.fetch_add(tin, Ordering::Relaxed);
                    self.stats.tokens_out.fetch_add(tout, Ordering::Relaxed);
                    return Ok(r);
                }
                Err(e) => {
                    self.stats.http_errors.fetch_add(1, Ordering::Relaxed);
                    self.note_error(&e.to_string());
                    // the server does not support `response_format`: retry without it
                    if let LlmError::Http { status: 400 | 422 | 500, body, .. } = &e {
                        if req.json_schema.is_some() && self.cfg.structured == Structured::Auto {
                            let b = body.to_lowercase();
                            if b.contains("response_format") || b.contains("json_schema") || b.contains("schema") || b.contains("grammar") {
                                self.structured_ok.store(false, Ordering::Relaxed);
                                req.json_schema = None;
                                continue;
                            }
                        }
                    }
                    if e.fatal() {
                        let msg = match &e {
                            LlmError::Http { status: 401 | 403, .. } => format!("The server rejected the API token ({e})"),
                            _ => format!("{e}. Check the base URL (it should end with /v1) and the model name"),
                        };
                        if let Ok(mut f) = self.fatal.lock() {
                            f.get_or_insert(msg.clone());
                        }
                        return Err(CallError { msg, fatal: true });
                    }
                    if !e.retriable() || attempt >= self.cfg.max_retries {
                        return Err(CallError { msg: e.to_string(), fatal: false });
                    }
                    attempt += 1;
                    self.stats.retries.fetch_add(1, Ordering::Relaxed);
                    let base = (self.cfg.retry_backoff_ms as f64 / 1000.0) * (1u64 << (attempt - 1).min(6)) as f64;
                    let jitter = 0.75 + (std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.subsec_nanos()) % 500) as f64 / 1000.0;
                    let mut wait = Duration::from_secs_f64((base * jitter).min(60.0));
                    if let Some(ra) = e.retry_after() {
                        wait = wait.max(ra);
                    }
                    // sleep in small steps so cancel stays responsive
                    let until = Instant::now() + wait;
                    while Instant::now() < until {
                        if self.ctl.is_cancelled() {
                            break;
                        }
                        tokio::time::sleep(Duration::from_millis(100).min(until - Instant::now())).await;
                    }
                }
            }
        }
    }

    // ───────────── segments ─────────────

    async fn translate_single(&self, src: &str, expected: &[usize], key: u64) -> std::result::Result<String, String> {
        let sys = prompt::system_prompt(&self.src, &self.tgt, &self.cfg.instructions, &self.cfg.glossary, false);
        let est_payload = est_tokens(src);
        let est_total = est_payload + est_tokens(&sys) + 8;
        let mut last = String::from("no attempt");
        for attempt in 0..3u32 {
            let req = ChatRequest {
                messages: vec![("system".into(), sys.clone()), ("user".into(), src.to_string())],
                temperature: (self.cfg.temperature + 0.15 * attempt as f32).min(1.0),
                max_tokens: self.out_tokens(est_total, est_payload),
                json_schema: None,
            };
            match self.call(req).await {
                Ok(r) => {
                    if r.finish_reason.as_deref() == Some("length") {
                        last = "response was cut off (raise the max output tokens or lower the context budget)".into();
                        continue;
                    }
                    let out = prompt::clean_single(&r.content, src);
                    match prompt::validate(src, &out, expected, &self.src, &self.tgt) {
                        Ok(()) => {
                            let _ = self.journal.put(key, &out);
                            self.stats.units_translated.fetch_add(1, Ordering::Relaxed);
                            return Ok(out);
                        }
                        Err(e) => last = e,
                    }
                }
                Err(e) => {
                    if e.fatal {
                        return Err(e.msg);
                    }
                    last = e.msg;
                }
            }
        }
        Err(last)
    }

    async fn run_chunk(&self, segs: &[Seg], idxs: Vec<usize>) -> Vec<(usize, std::result::Result<String, String>)> {
        if idxs.len() == 1 {
            let i = idxs[0];
            return vec![(i, self.translate_single(&segs[i].src, &segs[i].expected, segs[i].key).await)];
        }
        let items: Vec<(usize, &str)> = idxs.iter().enumerate().map(|(k, &i)| (k, segs[i].src.as_str())).collect();
        let user = prompt::chunk_user_message(&items);
        let sys = prompt::system_prompt(&self.src, &self.tgt, &self.cfg.instructions, &self.cfg.glossary, true);
        let est_payload = est_tokens(&user);
        let est_total = est_payload + est_tokens(&sys) + 8;
        let schema = if self.structured_ok.load(Ordering::Relaxed) { Some(prompt::json_schema()) } else { None };
        let req = ChatRequest {
            messages: vec![("system".into(), sys), ("user".into(), user)],
            temperature: self.cfg.temperature,
            max_tokens: self.out_tokens(est_total, est_payload),
            json_schema: schema,
        };
        let mut done: Vec<(usize, std::result::Result<String, String>)> = Vec::new();
        let mut redo: Vec<usize> = Vec::new();
        match self.call(req).await {
            Ok(r) if r.finish_reason.as_deref() != Some("length") => match prompt::parse_chunk_response(&r.content) {
                Some(parsed) => {
                    let map: HashMap<usize, String> = parsed.into_iter().collect();
                    for (k, &i) in idxs.iter().enumerate() {
                        match map.get(&k) {
                            Some(out) => match prompt::validate(&segs[i].src, out.trim(), &segs[i].expected, &self.src, &self.tgt) {
                                Ok(()) => {
                                    let out = out.trim().to_string();
                                    let _ = self.journal.put(segs[i].key, &out);
                                    self.stats.units_translated.fetch_add(1, Ordering::Relaxed);
                                    done.push((i, Ok(out)));
                                }
                                Err(_) => redo.push(i),
                            },
                            None => redo.push(i),
                        }
                    }
                }
                None => redo.extend(idxs.iter().copied()),
            },
            Ok(_) => redo.extend(idxs.iter().copied()),
            Err(e) if e.fatal => {
                return idxs.iter().map(|&i| (i, Err(e.msg.clone()))).collect();
            }
            Err(_) => redo.extend(idxs.iter().copied()),
        }
        // anything the batch got wrong is retried one by one
        let singles = join_all(redo.iter().map(|&i| async move { (i, self.translate_single(&segs[i].src, &segs[i].expected, segs[i].key).await) })).await;
        done.extend(singles);
        done
    }

    async fn translate_segs(&self, segs: &[Seg]) -> Vec<std::result::Result<String, String>> {
        let mut results: Vec<Option<std::result::Result<String, String>>> = vec![None; segs.len()];
        let mut todo: Vec<usize> = Vec::new();
        for (i, s) in segs.iter().enumerate() {
            match self.journal.get(s.key) {
                Some(t) => {
                    self.stats.units_cached.fetch_add(1, Ordering::Relaxed);
                    results[i] = Some(Ok(t));
                }
                None => todo.push(i),
            }
        }
        let budget = self.payload_budget();
        let mut chunks: Vec<Vec<usize>> = Vec::new();
        let mut cur: Vec<usize> = Vec::new();
        let mut cur_tok = 0usize;
        for i in todo {
            let t = est_tokens(&segs[i].src) + 14;
            if !cur.is_empty() && (cur_tok + t > budget || cur.len() >= 16) {
                chunks.push(std::mem::take(&mut cur));
                cur_tok = 0;
            }
            cur.push(i);
            cur_tok += t;
        }
        if !cur.is_empty() {
            chunks.push(cur);
        }
        let outs = join_all(chunks.into_iter().map(|c| self.run_chunk(segs, c))).await;
        for group in outs {
            for (i, r) in group {
                results[i] = Some(r);
            }
        }
        results.into_iter().map(|r| r.unwrap_or_else(|| Err("not translated".into()))).collect()
    }

    /// Translate several texts (messages) in as few requests as the context budget allows.
    pub async fn translate_texts(&self, texts: &[String], names: &[String]) -> std::result::Result<Vec<String>, String> {
        let nh = names_hash(names);
        let max_chars = (self.payload_budget() as f64 * 3.0) as usize;
        enum Piece {
            /// Nothing to translate in this piece (kept verbatim, with its separator).
            Keep(String),
            /// Index into `all` plus the separator that followed the segment.
            Seg(usize, String),
        }
        struct UnitPlan {
            prot: Option<protect::Protected>,
            pieces: Vec<Piece>,
        }
        let mut all: Vec<Seg> = Vec::new();
        let mut plans: Vec<UnitPlan> = Vec::with_capacity(texts.len());
        for t in texts {
            let prot = protect::protect(t, names);
            if !protect::has_translatable_text(&prot.text) {
                plans.push(UnitPlan { prot: None, pieces: vec![] });
                continue;
            }
            let mut pieces = Vec::new();
            for (seg, sep) in prompt::split_segments(&prot.text, max_chars) {
                if !protect::has_translatable_text(&seg) {
                    pieces.push(Piece::Keep(format!("{seg}{sep}")));
                    continue;
                }
                let key = self.keys.seg_key(&seg, nh);
                let expected = protect::tokens_in(&seg);
                all.push(Seg { src: seg, expected, key });
                pieces.push(Piece::Seg(all.len() - 1, sep));
            }
            plans.push(UnitPlan { prot: Some(prot), pieces });
        }
        let results = self.translate_segs(&all).await;
        let mut out = Vec::with_capacity(texts.len());
        for (t, p) in texts.iter().zip(plans) {
            let Some(prot) = p.prot else {
                out.push(t.clone());
                continue;
            };
            let mut joined = String::new();
            for piece in p.pieces {
                match piece {
                    Piece::Keep(s) => joined.push_str(&s),
                    Piece::Seg(idx, sep) => match &results[idx] {
                        Ok(s) => {
                            joined.push_str(s);
                            joined.push_str(&sep);
                        }
                        Err(e) => return Err(e.clone()),
                    },
                }
            }
            out.push(protect::restore(&joined, &prot.spans));
        }
        Ok(out)
    }

    /// Translate one long field (persona description…) once, even if many samples ask at the same time.
    async fn translate_field(&self, text: &str, names: &[String]) -> std::result::Result<String, String> {
        let key = self.keys.seg_key(text, names_hash(names)) ^ 0xF1E1_D000;
        let cell = {
            let mut m = self.fields.lock().expect("fields lock");
            m.entry(key).or_insert_with(|| Arc::new(OnceCell::new())).clone()
        };
        let r = cell
            .get_or_init(|| async { self.translate_texts(&[text.to_string()], names).await.map(|mut v| v.remove(0)) })
            .await
            .clone();
        if r.is_err() {
            if let Ok(mut m) = self.fields.lock() {
                m.remove(&key);
            }
        }
        r
    }

    pub async fn translate_sample(&self, sample: Sample) -> Translated {
        let r = self.translate_sample_inner(sample.clone()).await;
        match r {
            Ok(s) => {
                self.stats.samples_done.fetch_add(1, Ordering::Relaxed);
                Translated::Ok(s)
            }
            Err(reason) => {
                self.stats.samples_failed.fetch_add(1, Ordering::Relaxed);
                if reason != "cancelled" {
                    self.note_error(&reason);
                    if let Ok(mut f) = self.stats.failures.lock() {
                        if f.len() >= 200 {
                            f.pop_front();
                        }
                        f.push_back(Failure { source: sample.source, row: sample.row, reason: reason.clone() });
                    }
                }
                Translated::Failed(reason)
            }
        }
    }

    async fn translate_sample_inner(&self, sample: Sample) -> std::result::Result<Sample, String> {
        if let Some(f) = self.stop_reason() {
            return Err(f);
        }
        let mut out = sample.clone();
        let mut names: Vec<String> = Vec::new();
        if let Some(p) = &sample.persona {
            names.extend(p.char_name.clone());
            names.extend(p.user_name.clone());
        }
        // persona fields that actually end up in the system prompt
        if self.cfg.translate_persona {
            if let Some(p) = out.persona.as_mut() {
                let used = |f: &str| self.persona_fields.iter().any(|x| x == f);
                if used("description") && !p.description.trim().is_empty() {
                    p.description = self.translate_field(&p.description.clone(), &names).await?;
                }
                if used("definitions") && !p.definitions.trim().is_empty() {
                    p.definitions = self.translate_field(&crate::render::clean_definitions(&p.definitions), &names).await?;
                }
                if used("scenario") && !p.scenario.trim().is_empty() {
                    p.scenario = self.translate_field(&p.scenario.clone(), &names).await?;
                }
                if used("greeting") && !p.greeting.trim().is_empty() {
                    p.greeting = self.translate_field(&p.greeting.clone(), &names).await?;
                }
            }
        }
        if self.cfg.roles.contains(&Role::System) {
            if let Some(s) = out.system.clone().filter(|s| !s.trim().is_empty()) {
                out.system = Some(self.translate_field(&s, &names).await?);
            }
        }
        // conversation turns (batched into chunks)
        let idxs: Vec<usize> = out.messages.iter().enumerate().filter(|(_, m)| self.cfg.roles.contains(&m.role)).map(|(i, _)| i).collect();
        if !idxs.is_empty() {
            let texts: Vec<String> = idxs.iter().map(|&i| out.messages[i].content.clone()).collect();
            let tr = self.translate_texts(&texts, &names).await?;
            for (i, t) in idxs.into_iter().zip(tr) {
                out.messages[i].content = t;
            }
        }
        if out.messages.is_empty() {
            if let Some(t) = out.text.clone() {
                out.text = Some(self.translate_texts(&[t], &names).await?.remove(0));
            }
        }
        if out.id == sample.id {
            out.id = out.compute_id();
        }
        Ok(out)
    }
}

struct Seg {
    src: String,
    expected: Vec<usize>,
    key: u64,
}

impl SampleTranslator for Translator {
    fn translate(&self, sample: Sample) -> BoxFuture<'_, Translated> {
        Box::pin(self.translate_sample(sample))
    }
    fn concurrency(&self) -> usize {
        self.cfg.concurrency.clamp(1, 256)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct JobSummary {
    pub total: u64,
    pub done: u64,
    pub failed: u64,
    pub stopped: Option<String>,
    pub stats: StatsSnapshot,
}

fn progress_update(t: &Translator, done: u64, total: u64) -> Update {
    let s = t.snapshot();
    let mut u = Update::new("translate", done, total, format!("Translated {done}/{total}"));
    u.stats.insert("stats".into(), serde_json::to_value(&s).unwrap_or(json!({})));
    u
}

/// Translate every sample of the plan (filling the journal). Export later reads the same cache.
pub async fn run_job(
    translator: Arc<Translator>,
    sources: &[SourceData],
    plan: &MixPlan,
    only_first: Option<usize>,
    ctl: &Ctl,
    progress: &Progress,
) -> JobSummary {
    let sources: HashMap<usize, &SourceData> = sources.iter().map(|s| (s.index, s)).collect();
    let entries: Vec<_> = plan.entries.iter().copied().take(only_first.unwrap_or(usize::MAX)).collect();
    let total = entries.len() as u64;
    let conc = translator.concurrency() * 2;
    let mut done = 0u64;
    let mut failed = 0u64;
    let mut stopped = None;
    let tr = translator.clone();
    let srcs = &sources;
    let mut stream = futures::stream::iter(entries.into_iter())
        .map(|e| {
            let tr = tr.clone();
            async move {
                let sd = srcs.get(&(e.source as usize))?;
                let s = sd.sample(e.row)?;
                Some(tr.translate_sample(s).await)
            }
        })
        .buffered(conc);
    let mut last = Instant::now();
    while let Some(r) = stream.next().await {
        done += 1;
        if let Some(Translated::Failed(_)) = r {
            failed += 1;
        }
        if ctl.is_cancelled() {
            stopped = Some("cancelled".to_string());
            break;
        }
        if let Some(reason) = translator.stop_reason() {
            stopped = Some(reason);
            break;
        }
        if last.elapsed().as_millis() > 250 {
            progress(progress_update(&translator, done, total));
            last = Instant::now();
        }
    }
    progress(progress_update(&translator, done, total));
    JobSummary { total, done, failed, stopped, stats: translator.snapshot() }
}
