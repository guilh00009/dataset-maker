//! Application layer shared by the desktop app (Tauri) and the HTTP bridge: it owns the
//! session state, runs background jobs and exposes everything as `call(command, args) -> JSON`.

mod secrets;

use anyhow::{Context, Result, anyhow, bail};
use mixer_core::detect::{Detection, Mapping, normalize};
use mixer_core::export::{self, ExportJob, SampleTranslator, Translated};
use mixer_core::jobs::{Ctl, Progress, Update};
use mixer_core::mix::{self, HIST_BOUNDS, MixPlan, Scan, SourceData};
use mixer_core::pipeline;
use mixer_core::recipe::{Format, Recipe, SourceEntry, SourceLocation};
use mixer_core::render;
use mixer_core::source::FileFormat;
use mixer_core::translate::{self, Translator, client::TestReport, journal::Journal};
use mixer_core::{hf, model::Sample};
use secrets::Secrets;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::future::Future;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Instant;
use tokio::sync::{Mutex, broadcast};

// ───────────────────────────── events & jobs ─────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct JobView {
    pub id: String,
    pub kind: String,
    pub label: String,
    /// running | paused | done | error | cancelled
    pub state: String,
    pub update: Option<Update>,
    pub result: Option<Value>,
    pub error: Option<String>,
    pub elapsed_secs: f64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    Job(JobView),
    /// Session state changed; the UI refetches `get_state`.
    State,
}

struct Job {
    view: JobView,
    ctl: Ctl,
    started: Instant,
    translator: Option<Arc<Translator>>,
}

// ───────────────────────────── state ─────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    pub theme: String,
}

impl Default for Settings {
    fn default() -> Self {
        Self { theme: "system".into() }
    }
}

#[derive(Default)]
struct Slot {
    status: String, // opening | ready | error
    error: Option<String>,
    loaded: Option<Arc<SourceData>>,
    detection: Option<Detection>,
    rows: usize,
    bytes: u64,
    format: Option<FileFormat>,
    job: Option<String>,
}

struct PlanState {
    plan: Arc<MixPlan>,
    key: u64,
    sources: Vec<(usize, Arc<SourceData>)>,
}

struct St {
    recipe: Recipe,
    slots: Vec<Slot>,
    plan: Option<PlanState>,
    scans: HashMap<String, (u64, Arc<Scan>)>,
    settings: Settings,
    last_test: Option<TestReport>,
}

struct Inner {
    cache_dir: PathBuf,
    config_dir: PathBuf,
    st: Mutex<St>,
    jobs: StdMutex<HashMap<String, Job>>,
    job_order: StdMutex<Vec<String>>,
    tx: broadcast::Sender<Event>,
    secrets: Secrets,
    journal: Arc<Journal>,
    seq: AtomicU64,
}

#[derive(Clone)]
pub struct Api {
    inner: Arc<Inner>,
}

fn hash_of<T: Hash>(t: &T) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    t.hash(&mut h);
    h.finish()
}

fn err_string(e: &anyhow::Error) -> String {
    format!("{e:#}")
}

fn is_cancel(e: &anyhow::Error) -> bool {
    e.downcast_ref::<mixer_core::jobs::Cancelled>().is_some() || format!("{e:#}") == "cancelled"
}

fn default_out_dir() -> String {
    dirs::home_dir().unwrap_or_else(|| PathBuf::from(".")).join("dataset-mixer-output").to_string_lossy().to_string()
}

fn slug(s: &str) -> String {
    let t: String = s.chars().map(|c| if c.is_alphanumeric() { c.to_ascii_lowercase() } else { '-' }).collect();
    t.trim_matches('-').to_string()
}

impl Api {
    pub fn new(cache_dir: Option<PathBuf>, config_dir: Option<PathBuf>) -> Result<Self> {
        let cache_dir = cache_dir.unwrap_or_else(pipeline::default_cache_dir);
        let config_dir = config_dir
            .or_else(|| std::env::var_os("MIXER_CONFIG_DIR").map(PathBuf::from))
            .or_else(|| dirs::config_dir().map(|d| d.join("dataset-mixer")))
            .unwrap_or_else(|| PathBuf::from(".mixer-config"));
        std::fs::create_dir_all(&cache_dir)?;
        std::fs::create_dir_all(&config_dir)?;
        let journal = Arc::new(Journal::open(&pipeline::journal_path(&cache_dir))?);
        let settings: Settings = std::fs::read(config_dir.join("settings.json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
        let mut recipe: Recipe = std::fs::read(config_dir.join("session.json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
        if recipe.export.out_dir == "out" {
            recipe.export.out_dir = default_out_dir();
        }
        let slots = recipe.sources.iter().map(|_| Slot { status: "opening".into(), ..Default::default() }).collect();
        let (tx, _) = broadcast::channel(2048);
        Ok(Self {
            inner: Arc::new(Inner {
                secrets: Secrets::new(&config_dir),
                cache_dir,
                config_dir,
                st: Mutex::new(St { recipe, slots, plan: None, scans: HashMap::new(), settings, last_test: None }),
                jobs: StdMutex::new(HashMap::new()),
                job_order: StdMutex::new(vec![]),
                tx,
                journal,
                seq: AtomicU64::new(1),
            }),
        })
    }

    /// Re-open the sources of the restored session (call once, inside the runtime).
    pub async fn init(&self) {
        let n = self.inner.st.lock().await.recipe.sources.len();
        for i in 0..n {
            self.spawn_open(i).await;
        }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<Event> {
        self.inner.tx.subscribe()
    }

    fn emit(&self, e: Event) {
        let _ = self.inner.tx.send(e);
    }

    fn state_changed(&self) {
        self.emit(Event::State);
    }

    // ───────── jobs ─────────

    fn spawn_job<F, Fut>(&self, kind: &str, label: String, ctl: Ctl, translator: Option<Arc<Translator>>, f: F) -> String
    where
        F: FnOnce(Progress) -> Fut + Send + 'static,
        Fut: Future<Output = Result<Value>> + Send + 'static,
    {
        let id = format!("job-{}", self.inner.seq.fetch_add(1, Ordering::SeqCst));
        let view = JobView { id: id.clone(), kind: kind.into(), label, state: "running".into(), update: None, result: None, error: None, elapsed_secs: 0.0 };
        {
            let mut jobs = self.inner.jobs.lock().unwrap();
            jobs.insert(id.clone(), Job { view: view.clone(), ctl: ctl.clone(), started: Instant::now(), translator });
            let mut order = self.inner.job_order.lock().unwrap();
            order.push(id.clone());
            // keep the registry small: forget the oldest finished jobs
            while order.len() > 40 {
                let old = order.remove(0);
                if jobs.get(&old).is_some_and(|j| j.view.state != "running" && j.view.state != "paused") {
                    jobs.remove(&old);
                } else {
                    order.push(old);
                    break;
                }
            }
        }
        self.emit(Event::Job(view));
        let api = self.clone();
        let jid = id.clone();
        let last_emit = Arc::new(StdMutex::new(Instant::now() - std::time::Duration::from_secs(1)));
        let progress: Progress = {
            let api = api.clone();
            let jid = jid.clone();
            Arc::new(move |u: Update| {
                let mut le = last_emit.lock().unwrap();
                let final_update = u.total > 0 && u.done >= u.total;
                if le.elapsed().as_millis() < 120 && !final_update {
                    return;
                }
                *le = Instant::now();
                drop(le);
                api.update_job(&jid, |v| v.update = Some(u.clone()));
            })
        };
        tokio::spawn(async move {
            let r = f(progress).await;
            api.update_job(&jid, |v| match r {
                Ok(val) => {
                    v.state = "done".into();
                    v.result = Some(val);
                }
                Err(e) if is_cancel(&e) => v.state = "cancelled".into(),
                Err(e) => {
                    v.state = "error".into();
                    v.error = Some(err_string(&e));
                }
            });
            api.state_changed();
        });
        id
    }

    fn update_job(&self, id: &str, f: impl FnOnce(&mut JobView)) {
        let view = {
            let mut jobs = self.inner.jobs.lock().unwrap();
            let Some(j) = jobs.get_mut(id) else { return };
            f(&mut j.view);
            if j.ctl.is_paused() && j.view.state == "running" {
                j.view.state = "paused".into();
            }
            j.view.elapsed_secs = j.started.elapsed().as_secs_f64();
            j.view.clone()
        };
        self.emit(Event::Job(view));
    }

    fn job_ctl(&self, id: &str) -> Result<Ctl> {
        self.inner.jobs.lock().unwrap().get(id).map(|j| j.ctl.clone()).ok_or_else(|| anyhow!("unknown job `{id}`"))
    }

    fn jobs_view(&self) -> Vec<JobView> {
        let jobs = self.inner.jobs.lock().unwrap();
        let order = self.inner.job_order.lock().unwrap();
        order
            .iter()
            .filter_map(|id| jobs.get(id))
            .map(|j| {
                let mut v = j.view.clone();
                if v.state == "running" || v.state == "paused" {
                    v.elapsed_secs = j.started.elapsed().as_secs_f64();
                }
                if let Some(t) = &j.translator {
                    if v.state == "running" || v.state == "paused" {
                        v.update.get_or_insert_with(Update::default).stats.insert("stats".into(), serde_json::to_value(t.snapshot()).unwrap_or(json!({})));
                    }
                }
                v
            })
            .collect()
    }

    // ───────── persistence ─────────

    async fn save_session(&self) {
        let st = self.inner.st.lock().await;
        let _ = std::fs::write(self.inner.config_dir.join("session.json"), serde_json::to_vec_pretty(&st.recipe).unwrap_or_default());
    }

    fn save_settings(&self, s: &Settings) {
        let _ = std::fs::write(self.inner.config_dir.join("settings.json"), serde_json::to_vec_pretty(s).unwrap_or_default());
    }

    fn llm_token(&self) -> Option<String> {
        self.inner.secrets.get("llm_token")
    }
    fn hf_token(&self) -> Option<String> {
        self.inner.secrets.get("hf_token")
    }

    // ───────── sources ─────────

    async fn spawn_open(&self, index: usize) {
        let (entry, ctl) = {
            let mut st = self.inner.st.lock().await;
            let Some(entry) = st.recipe.sources.get(index).cloned() else { return };
            if let Some(old) = st.slots.get(index).and_then(|s| s.job.clone()) {
                if let Ok(c) = self.job_ctl(&old) {
                    c.cancel();
                }
            }
            let slot = &mut st.slots[index];
            *slot = Slot { status: "opening".into(), ..Default::default() };
            (entry, Ctl::new())
        };
        let api = self.clone();
        let cache = self.inner.cache_dir.clone();
        let hf_token = self.hf_token();
        let id = entry.id.clone();
        let label = format!("Opening {}", entry.id);
        let ctl2 = ctl.clone();
        let job = self.spawn_job("open_source", label, ctl, None, move |progress| async move {
            let r: Result<()> = async {
                let path = pipeline::resolve_location(&entry.location, &cache, &hf_token, &ctl2, &progress).await?;
                let (eid, mapping, c2, p2, ctl3) = (entry.id.clone(), entry.mapping.clone(), cache.clone(), progress.clone(), ctl2.clone());
                let l = tokio::task::spawn_blocking(move || pipeline::open_source(index, &eid, &path, mapping, &c2, &ctl3, &p2)).await??;
                let mut st = api.inner.st.lock().await;
                // the source may have been removed or moved while we were busy
                if st.recipe.sources.get(index).map(|e| e.id.as_str()) == Some(id.as_str()) {
                    let mut data = l.data;
                    data.index = index;
                    st.slots[index] = Slot {
                        status: "ready".into(),
                        error: None,
                        loaded: Some(Arc::new(data)),
                        detection: Some(l.detection),
                        rows: l.rows,
                        bytes: l.bytes,
                        format: Some(l.format),
                        job: None,
                    };
                }
                Ok(())
            }
            .await;
            if let Err(e) = &r {
                if !is_cancel(e) {
                    let mut st = api.inner.st.lock().await;
                    if st.recipe.sources.get(index).map(|e| e.id.as_str()) == Some(id.as_str()) {
                        st.slots[index] = Slot { status: "error".into(), error: Some(err_string(e)), ..Default::default() };
                    }
                }
            }
            api.state_changed();
            r.map(|_| json!({}))
        });
        if let Some(slot) = self.inner.st.lock().await.slots.get_mut(index) {
            if slot.status == "opening" {
                slot.job = Some(job);
            }
        }
    }

    async fn add_source(&self, args: Value) -> Result<Value> {
        let kind = args["kind"].as_str().unwrap_or("file");
        let location = match kind {
            "hf" => SourceLocation::Hf {
                repo: args["repo"].as_str().context("repo is required")?.to_string(),
                file: args["file"].as_str().context("file is required")?.to_string(),
                revision: args["revision"].as_str().map(|s| s.to_string()).filter(|s| !s.is_empty() && s != "main"),
            },
            _ => SourceLocation::File { path: args["path"].as_str().context("path is required")?.to_string() },
        };
        let (index, id) = {
            let mut st = self.inner.st.lock().await;
            let base = match &location {
                SourceLocation::Hf { repo, file, .. } => {
                    let short = repo.split('/').next_back().unwrap_or(repo);
                    if hf::is_data_file(file) && file.rsplit('/').next().is_some_and(|f| !f.starts_with("train") && !f.starts_with("0000")) {
                        format!("{short}/{}", file.rsplit('/').next().unwrap_or(file))
                    } else {
                        short.to_string()
                    }
                }
                l => l.label(),
            };
            let mut id = base.clone();
            let mut n = 2;
            while st.recipe.sources.iter().any(|s| s.id == id) {
                id = format!("{base} ({n})");
                n += 1;
            }
            st.recipe.sources.push(SourceEntry { id: id.clone(), location, weight: 50.0, count: None, mapping: None, enabled: true });
            st.slots.push(Slot { status: "opening".into(), ..Default::default() });
            (st.recipe.sources.len() - 1, id)
        };
        self.spawn_open(index).await;
        self.save_session().await;
        self.state_changed();
        Ok(json!({"id": id}))
    }

    async fn remove_source(&self, id: &str) -> Result<Value> {
        {
            let mut st = self.inner.st.lock().await;
            let i = st.recipe.sources.iter().position(|s| s.id == id).context("unknown source")?;
            if let Some(job) = st.slots[i].job.clone() {
                if let Ok(c) = self.job_ctl(&job) {
                    c.cancel();
                }
            }
            st.recipe.sources.remove(i);
            st.slots.remove(i);
            st.scans.remove(id);
            // indices follow positions
            let mut rebuilt: Vec<Option<Arc<SourceData>>> = Vec::new();
            for (k, slot) in st.slots.iter().enumerate() {
                rebuilt.push(slot.loaded.as_ref().map(|d| {
                    let mut c = (**d).clone();
                    c.index = k;
                    Arc::new(c)
                }));
            }
            for (slot, r) in st.slots.iter_mut().zip(rebuilt) {
                slot.loaded = r;
            }
            st.plan = None;
        }
        self.save_session().await;
        self.state_changed();
        Ok(json!({}))
    }

    async fn update_source(&self, args: Value) -> Result<Value> {
        let id = args["id"].as_str().context("id is required")?.to_string();
        {
            let mut st = self.inner.st.lock().await;
            let i = st.recipe.sources.iter().position(|s| s.id == id).context("unknown source")?;
            let e = &mut st.recipe.sources[i];
            if let Some(w) = args.get("weight").and_then(|v| v.as_f64()) {
                e.weight = w.max(0.0);
            }
            if let Some(c) = args.get("count") {
                e.count = c.as_u64();
            }
            if let Some(en) = args.get("enabled").and_then(|v| v.as_bool()) {
                e.enabled = en;
            }
            let mut mapping_changed = false;
            if let Some(m) = args.get("mapping") {
                e.mapping = if m.is_null() { None } else { Some(serde_json::from_value::<Mapping>(m.clone()).context("invalid mapping")?) };
                mapping_changed = true;
            }
            if mapping_changed {
                let effective = st.recipe.sources[i].mapping.clone().or_else(|| st.slots[i].detection.as_ref().map(|d| d.best.clone()));
                if let (Some(m), Some(d)) = (effective, st.slots[i].loaded.clone()) {
                    let mut c = (*d).clone();
                    c.mapping = m;
                    st.slots[i].loaded = Some(Arc::new(c));
                }
                st.scans.remove(&id);
            }
        }
        self.save_session().await;
        self.state_changed();
        Ok(json!({}))
    }

    async fn set_options(&self, args: Value) -> Result<Value> {
        {
            let mut st = self.inner.st.lock().await;
            let mut v = serde_json::to_value(&st.recipe)?;
            let obj = v.as_object_mut().unwrap();
            for key in ["name", "seed", "mix", "filters", "split", "persona", "export", "translate"] {
                if let Some(x) = args.get(key) {
                    obj.insert(key.into(), x.clone());
                }
            }
            st.recipe = serde_json::from_value(v).context("invalid options")?;
        }
        self.save_session().await;
        self.state_changed();
        Ok(json!({}))
    }

    // ───────── state view ─────────

    fn plan_key(recipe: &Recipe, slots: &[Slot]) -> u64 {
        let mut s = String::new();
        s.push_str(&serde_json::to_string(&json!({
            "seed": recipe.seed, "mix": recipe.mix, "filters": recipe.filters, "split": recipe.split,
            "sources": recipe.sources.iter().map(|e| json!({"id": e.id, "w": e.weight, "c": e.count, "en": e.enabled, "m": e.mapping})).collect::<Vec<_>>(),
        }))
        .unwrap_or_default());
        for sl in slots {
            s.push_str(&format!("|{}:{}", sl.status, sl.rows));
            if let Some(d) = &sl.loaded {
                s.push_str(&serde_json::to_string(&d.mapping).unwrap_or_default());
            }
        }
        hash_of(&s)
    }

    fn plan_view(ps: &PlanState, current_key: u64) -> Value {
        let p = &ps.plan;
        json!({
            "total": p.entries.len(), "train": p.entries.len() - p.val_count, "validation": p.val_count,
            "tokens_total": p.tokens_total, "histogram": p.histogram, "bounds": HIST_BOUNDS,
            "per_source": p.per_source, "stale": ps.key != current_key,
        })
    }

    async fn get_state(&self) -> Result<Value> {
        let st = self.inner.st.lock().await;
        let key = Self::plan_key(&st.recipe, &st.slots);
        let sources: Vec<Value> = st
            .recipe
            .sources
            .iter()
            .zip(st.slots.iter())
            .map(|(e, s)| {
                json!({
                    "id": e.id, "label": e.location.label(), "location": e.location, "status": s.status, "error": s.error,
                    "rows": s.rows, "bytes": s.bytes, "format": s.format, "job": s.job,
                    "detection": s.detection.as_ref().map(|d| json!({"best": d.best, "candidates": d.candidates, "fields": d.fields, "sampled": d.sampled})),
                    "mapping_is_custom": e.mapping.is_some(),
                    "effective_mapping": s.loaded.as_ref().map(|d| &d.mapping),
                })
            })
            .collect();
        let secrets = &self.inner.secrets;
        Ok(json!({
            "recipe": st.recipe,
            "sources": sources,
            "plan": st.plan.as_ref().map(|p| Self::plan_view(p, key)),
            "settings": {
                "theme": st.settings.theme,
                "has_llm_token": secrets.has("llm_token"), "has_hf_token": secrets.has("hf_token"),
                "llm_token_store": secrets.location("llm_token"), "hf_token_store": secrets.location("hf_token"),
                "cache_dir": self.inner.cache_dir, "config_dir": self.inner.config_dir,
            },
            "jobs": self.jobs_view(),
            "formats": Format::ALL.iter().map(|f| json!({"id": f, "label": f.label()})).collect::<Vec<_>>(),
            "journal_units": self.inner.journal.count(),
            "last_test": st.last_test,
        }))
    }

    // ───────── mixing ─────────

    async fn compute_mix(&self) -> Result<Value> {
        let (recipe, sources) = {
            let st = self.inner.st.lock().await;
            let sources: Vec<(usize, Arc<SourceData>)> = st
                .recipe
                .sources
                .iter()
                .enumerate()
                .filter(|(_, e)| e.enabled)
                .filter_map(|(i, _)| st.slots[i].loaded.clone().map(|d| (i, d)))
                .collect();
            let enabled = st.recipe.sources.iter().filter(|e| e.enabled).count();
            if enabled == 0 {
                bail!("enable at least one source first");
            }
            if sources.len() < enabled {
                bail!("some sources are still opening or failed to open");
            }
            (st.recipe.clone(), sources)
        };
        let api = self.clone();
        let ctl = Ctl::new();
        let c2 = ctl.clone();
        let label = "Computing mix".to_string();
        let id = self.spawn_job("compute_mix", label, ctl, None, move |progress| async move {
            let mut scans: Vec<Arc<Scan>> = Vec::new();
            for (_, sd) in &sources {
                let skey = hash_of(&(serde_json::to_string(&recipe.filters)?, serde_json::to_string(&sd.mapping)?));
                let cached = api.inner.st.lock().await.scans.get(&sd.id).filter(|(k, _)| *k == skey).map(|(_, s)| s.clone());
                let scan = match cached {
                    Some(s) => s,
                    None => {
                        let (sdc, f, c, p) = (sd.clone(), recipe.filters.clone(), c2.clone(), progress.clone());
                        let s = Arc::new(tokio::task::spawn_blocking(move || mix::scan_source(&sdc, &f, &c, &p)).await??);
                        api.inner.st.lock().await.scans.insert(sd.id.clone(), (skey, s.clone()));
                        s
                    }
                };
                scans.push(scan);
            }
            let (r2, src2, sc2, c3) = (recipe.clone(), sources.clone(), scans.clone(), c2.clone());
            let plan = tokio::task::spawn_blocking(move || {
                let pairs: Vec<(&SourceData, &Scan)> = src2.iter().zip(sc2.iter()).map(|((_, sd), sc)| (sd.as_ref(), sc.as_ref())).collect();
                mix::build_plan(&r2, &pairs, &c3)
            })
            .await??;
            let mut st = api.inner.st.lock().await;
            let key = Self::plan_key(&st.recipe, &st.slots);
            let ps = PlanState { plan: Arc::new(plan), key, sources };
            let view = Self::plan_view(&ps, key);
            st.plan = Some(ps);
            Ok(view)
        });
        Ok(json!({"job_id": id}))
    }

    // ───────── preview ─────────

    async fn preview(&self, args: Value) -> Result<Value> {
        let offset = args["offset"].as_u64().unwrap_or(0) as usize;
        let count = (args["count"].as_u64().unwrap_or(10) as usize).clamp(1, 50);
        let source_id = args["source"].as_str().map(|s| s.to_string());
        let (recipe, picks): (Recipe, Vec<(usize, u32, Arc<SourceData>)>) = {
            let st = self.inner.st.lock().await;
            let mut picks = Vec::new();
            let total;
            if let Some(sid) = &source_id {
                let i = st.recipe.sources.iter().position(|s| &s.id == sid).context("unknown source")?;
                let sd = st.slots[i].loaded.clone().context("source is not ready")?;
                total = sd.prepared.len();
                for pos in offset..(offset + count).min(total) {
                    picks.push((pos, pos as u32, sd.clone()));
                }
            } else {
                let ps = st.plan.as_ref().context("compute the mix first")?;
                total = ps.plan.entries.len();
                for pos in offset..(offset + count).min(total) {
                    let e = ps.plan.entries[pos];
                    let sd = ps.sources.iter().find(|(i, _)| *i == e.source as usize).map(|(_, s)| s.clone()).context("source missing")?;
                    picks.push((pos, e.row, sd));
                }
            }
            let _ = total;
            (st.recipe.clone(), picks)
        };
        let total = if let Some(sid) = &source_id {
            let st = self.inner.st.lock().await;
            st.recipe.sources.iter().position(|s| &s.id == sid).and_then(|i| st.slots[i].loaded.as_ref().map(|d| d.prepared.len())).unwrap_or(0)
        } else {
            self.inner.st.lock().await.plan.as_ref().map_or(0, |p| p.plan.entries.len())
        };
        let journal = self.inner.journal.clone();
        let cache_tr = if recipe.translate.enabled || journal.count() > 0 { Translator::new_cache_only(&recipe, journal).ok() } else { None };
        let mut out = Vec::new();
        for (pos, row, sd) in picks {
            let raw = sd.prepared.row_value(row as usize).ok();
            let sample: Option<Sample> = raw.as_ref().and_then(|r| normalize(r, &sd.mapping, sd.index, row as u64));
            let (original, formatted) = match &sample {
                Some(s) => (Some(render::finalize(s, &recipe.persona)), export::preview_row(&recipe, s).ok()),
                None => (None, None),
            };
            let translated = match (&cache_tr, &sample) {
                (Some(t), Some(s)) => match t.translate(s.clone()).await {
                    Translated::Ok(ts) => Some(json!({"sample": render::finalize(&ts, &recipe.persona), "formatted": export::preview_row(&recipe, &ts).ok()})),
                    Translated::Failed(_) => None,
                },
                _ => None,
            };
            out.push(json!({
                "position": pos, "row": row, "source": sd.id, "raw": raw, "original": original, "formatted": formatted, "translated": translated,
                "persona": sample.as_ref().and_then(|s| s.persona.clone()),
            }));
        }
        Ok(json!({"items": out, "total": total}))
    }

    // ───────── translation ─────────

    async fn llm_token_or(&self, args: &Value) -> Option<String> {
        args["token"].as_str().map(|s| s.to_string()).filter(|s| !s.trim().is_empty()).or_else(|| self.llm_token())
    }

    async fn llm_test(&self, args: Value) -> Result<Value> {
        let base = args["base_url"].as_str().unwrap_or("").to_string();
        let model = args["model"].as_str().unwrap_or("").to_string();
        let token = self.llm_token_or(&args).await;
        let rep = translate::client::test_connection(&base, &model, token).await;
        self.inner.st.lock().await.last_test = Some(rep.clone());
        Ok(serde_json::to_value(rep)?)
    }

    async fn estimate(&self) -> Result<Value> {
        let (recipe, sources, plan, last) = {
            let st = self.inner.st.lock().await;
            let ps = st.plan.as_ref().context("compute the mix first")?;
            (st.recipe.clone(), ps.sources.iter().map(|(_, s)| (**s).clone()).collect::<Vec<_>>(), ps.plan.clone(), st.last_test.clone())
        };
        let journal = self.inner.journal.clone();
        let est = tokio::task::spawn_blocking(move || {
            translate::estimate::estimate(&recipe, &sources, &plan, Some(&journal), last.as_ref().and_then(|l| l.tokens_per_sec), last.as_ref().and_then(|l| l.slots), &Ctl::new())
        })
        .await??;
        Ok(serde_json::to_value(est)?)
    }

    async fn start_translation(&self, args: Value) -> Result<Value> {
        let (recipe, sources, plan) = {
            let st = self.inner.st.lock().await;
            let ps = st.plan.as_ref().context("compute the mix first")?;
            if Self::plan_key(&st.recipe, &st.slots) != ps.key {
                bail!("the mix changed since it was computed; compute it again first");
            }
            (st.recipe.clone(), ps.sources.iter().map(|(_, s)| (**s).clone()).collect::<Vec<_>>(), ps.plan.clone())
        };
        let ctl = Ctl::new();
        let translator = Translator::new(&recipe, self.llm_token(), self.inner.journal.clone(), ctl.clone())?;
        let only_first = args["only_first"].as_u64().map(|n| n as usize);
        let c2 = ctl.clone();
        let t2 = translator.clone();
        let label = match only_first {
            Some(n) => format!("Translating a trial of {n} samples"),
            None => "Translating".to_string(),
        };
        let id = self.spawn_job("translate", label, ctl, Some(translator), move |progress| async move {
            let summary = translate::run_job(t2, &sources, &plan, only_first, &c2, &progress).await;
            if summary.stopped.as_deref() == Some("cancelled") {
                bail!("cancelled");
            }
            Ok(serde_json::to_value(summary)?)
        });
        Ok(json!({"job_id": id}))
    }

    async fn start_export(&self, args: Value) -> Result<Value> {
        let (mut recipe, sources, plan) = {
            let st = self.inner.st.lock().await;
            let ps = st.plan.as_ref().context("compute the mix first")?;
            if Self::plan_key(&st.recipe, &st.slots) != ps.key {
                bail!("the mix changed since it was computed; compute it again first");
            }
            (st.recipe.clone(), ps.sources.iter().map(|(_, s)| (**s).clone()).collect::<Vec<_>>(), ps.plan.clone())
        };
        if let Some(d) = args["out_dir"].as_str().filter(|d| !d.trim().is_empty()) {
            recipe.export.out_dir = d.to_string();
        }
        // fail fast on a bad custom template
        export::custom_template(&recipe.export)?;
        let ctl = Ctl::new();
        let translator: Option<Arc<Translator>> =
            if recipe.translate.enabled { Some(Translator::new(&recipe, self.llm_token(), self.inner.journal.clone(), ctl.clone())?) } else { None };
        let out_dir = PathBuf::from(&recipe.export.out_dir);
        let c2 = ctl.clone();
        let t2 = translator.clone();
        let id = self.spawn_job("export", format!("Exporting to {}", out_dir.display()), ctl, translator, move |progress| async move {
            let tr: Option<Arc<dyn SampleTranslator>> = t2.map(|t| t as Arc<dyn SampleTranslator>);
            let m = export::export(ExportJob { recipe: &recipe, sources: &sources, plan: &plan, out_dir: out_dir.clone(), translator: tr }, &c2, &progress).await?;
            let mut v = serde_json::to_value(&m)?;
            v["out_dir"] = json!(out_dir);
            Ok(v)
        });
        Ok(json!({"job_id": id}))
    }

    // ───────── misc commands ─────────

    async fn fs_list(&self, args: Value) -> Result<Value> {
        let path = args["path"].as_str().filter(|p| !p.is_empty()).map(PathBuf::from).or_else(dirs::home_dir).unwrap_or_else(|| PathBuf::from("/"));
        let path = if path.is_dir() { path } else { path.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| PathBuf::from("/")) };
        let path = path.canonicalize().unwrap_or(path);
        let mut entries = Vec::new();
        for e in std::fs::read_dir(&path).with_context(|| format!("cannot read {}", path.display()))?.flatten().take(5000) {
            let name = e.file_name().to_string_lossy().to_string();
            if name.starts_with('.') {
                continue;
            }
            let Ok(md) = e.metadata() else { continue };
            entries.push(json!({"name": name, "is_dir": md.is_dir(), "size": md.len(), "is_data": !md.is_dir() && hf::is_data_file(&name) || name.ends_with(".txt") || name.ends_with(".md")}));
        }
        entries.sort_by(|a, b| {
            (b["is_dir"].as_bool(), a["name"].as_str().map(|s| s.to_lowercase())).cmp(&(a["is_dir"].as_bool(), b["name"].as_str().map(|s| s.to_lowercase())))
        });
        Ok(json!({"path": path, "parent": path.parent(), "home": dirs::home_dir(), "entries": entries}))
    }

    async fn hf_list(&self, args: Value) -> Result<Value> {
        let input = args["input"].as_str().context("enter a dataset id or URL")?;
        let r = hf::parse_ref(input)?;
        let rev = args["revision"].as_str().map(|s| s.to_string()).filter(|s| !s.is_empty()).or(r.revision.clone());
        let files = hf::list_files(&r.repo, rev.as_deref(), &self.hf_token()).await?;
        let data: Vec<_> = files.iter().filter(|f| hf::is_data_file(&f.path)).collect();
        Ok(json!({"repo": r.repo, "revision": rev.unwrap_or_else(|| "main".into()), "file": r.file, "files": data, "all_files": files.len()}))
    }

    async fn set_settings(&self, args: Value) -> Result<Value> {
        let mut store = None;
        if let Some(t) = args.get("llm_token") {
            store = Some(self.inner.secrets.set("llm_token", t.as_str()));
        }
        if let Some(t) = args.get("hf_token") {
            store = Some(self.inner.secrets.set("hf_token", t.as_str()));
        }
        if let Some(theme) = args["theme"].as_str() {
            let mut st = self.inner.st.lock().await;
            st.settings.theme = theme.to_string();
            self.save_settings(&st.settings);
        }
        self.state_changed();
        Ok(json!({"store": store}))
    }

    async fn tokenizer_config(&self, args: Value) -> Result<Value> {
        let text = if let Some(t) = args["text"].as_str().filter(|t| !t.trim().is_empty()) {
            t.to_string()
        } else if let Some(p) = args["path"].as_str().filter(|t| !t.trim().is_empty()) {
            std::fs::read_to_string(p).with_context(|| format!("cannot read {p}"))?
        } else if let Some(repo) = args["repo"].as_str().filter(|t| !t.trim().is_empty()) {
            let repo = repo.trim().trim_start_matches("https://huggingface.co/").trim_matches('/');
            hf::fetch_text("", repo, "tokenizer_config.json", &self.hf_token()).await?
        } else {
            bail!("paste tokenizer_config.json, give a file path, or a model id");
        };
        let (template, bos, eos) = export::parse_tokenizer_config(&text)?;
        Ok(json!({"template": template, "bos_token": bos, "eos_token": eos}))
    }

    async fn save_recipe(&self, args: Value) -> Result<Value> {
        let path = PathBuf::from(args["path"].as_str().context("path is required")?);
        let toml = self.inner.st.lock().await.recipe.to_toml()?;
        if let Some(p) = path.parent() {
            std::fs::create_dir_all(p)?;
        }
        std::fs::write(&path, toml)?;
        Ok(json!({"path": path}))
    }

    async fn load_recipe(&self, args: Value) -> Result<Value> {
        let path = args["path"].as_str().context("path is required")?;
        let recipe = Recipe::from_toml(&std::fs::read_to_string(path).with_context(|| format!("cannot read {path}"))?)?;
        let n = {
            let mut st = self.inner.st.lock().await;
            for s in &st.slots {
                if let Some(j) = &s.job {
                    if let Ok(c) = self.job_ctl(j) {
                        c.cancel();
                    }
                }
            }
            st.slots = recipe.sources.iter().map(|_| Slot { status: "opening".into(), ..Default::default() }).collect();
            st.recipe = recipe;
            st.plan = None;
            st.scans.clear();
            st.recipe.sources.len()
        };
        for i in 0..n {
            self.spawn_open(i).await;
        }
        self.save_session().await;
        self.state_changed();
        Ok(json!({}))
    }

    async fn recipe_toml(&self) -> Result<Value> {
        Ok(json!({"toml": self.inner.st.lock().await.recipe.to_toml()?}))
    }

    async fn new_recipe(&self) -> Result<Value> {
        {
            let mut st = self.inner.st.lock().await;
            for s in &st.slots {
                if let Some(j) = &s.job {
                    if let Ok(c) = self.job_ctl(j) {
                        c.cancel();
                    }
                }
            }
            let mut r = Recipe::default();
            r.export.out_dir = default_out_dir();
            // keep the connection settings: they rarely change between recipes
            r.translate.base_url = st.recipe.translate.base_url.clone();
            r.translate.model = st.recipe.translate.model.clone();
            st.recipe = r;
            st.slots.clear();
            st.plan = None;
            st.scans.clear();
        }
        self.save_session().await;
        self.state_changed();
        Ok(json!({}))
    }

    /// Single entry point used by both the Tauri command and the HTTP bridge.
    pub async fn call(&self, cmd: &str, args: Value) -> std::result::Result<Value, String> {
        let r: Result<Value> = async {
            match cmd {
                "get_state" => self.get_state().await,
                "add_source" => self.add_source(args).await,
                "remove_source" => self.remove_source(args["id"].as_str().context("id is required")?).await,
                "update_source" => self.update_source(args).await,
                "retry_source" => {
                    let id = args["id"].as_str().context("id is required")?;
                    let i = self.inner.st.lock().await.recipe.sources.iter().position(|s| s.id == id).context("unknown source")?;
                    self.spawn_open(i).await;
                    self.state_changed();
                    Ok(json!({}))
                }
                "set_options" => self.set_options(args).await,
                "compute_mix" => self.compute_mix().await,
                "preview" => self.preview(args).await,
                "llm_test" => self.llm_test(args).await,
                "llm_models" => {
                    let c = mixer_core::translate::client::LlmClient::new(args["base_url"].as_str().unwrap_or(""), "", self.llm_token_or(&args).await, std::time::Duration::from_secs(20))?;
                    Ok(json!({"models": c.list_models().await.map_err(|e| anyhow!(e.to_string()))?}))
                }
                "estimate_translation" => self.estimate().await,
                "start_translation" => self.start_translation(args).await,
                "export" => self.start_export(args).await,
                "pause_job" => {
                    let id = args["job_id"].as_str().context("job_id is required")?;
                    self.job_ctl(id)?.pause();
                    self.update_job(id, |v| v.state = "paused".into());
                    Ok(json!({}))
                }
                "resume_job" => {
                    let id = args["job_id"].as_str().context("job_id is required")?;
                    self.job_ctl(id)?.resume();
                    self.update_job(id, |v| v.state = "running".into());
                    Ok(json!({}))
                }
                "cancel_job" => {
                    let id = args["job_id"].as_str().context("job_id is required")?;
                    let c = self.job_ctl(id)?;
                    c.resume();
                    c.cancel();
                    Ok(json!({}))
                }
                "fs_list" => self.fs_list(args).await,
                "hf_list" => self.hf_list(args).await,
                "set_settings" => self.set_settings(args).await,
                "tokenizer_config" => self.tokenizer_config(args).await,
                "save_recipe" => self.save_recipe(args).await,
                "load_recipe" => self.load_recipe(args).await,
                "recipe_toml" => self.recipe_toml().await,
                "new_recipe" => self.new_recipe().await,
                "clear_translation_cache" => {
                    self.inner.journal.clear()?;
                    self.state_changed();
                    Ok(json!({}))
                }
                other => bail!("unknown command `{other}`"),
            }
        }
        .await;
        r.map_err(|e| err_string(&e))
    }

    pub fn default_slug(name: &str) -> String {
        slug(name)
    }
}

/// Convenience for tests and the CLI: wait for a job to finish and return its result.
pub async fn wait_job(api: &Api, job_id: &str) -> std::result::Result<Value, String> {
    let mut rx = api.subscribe();
    // the job may already be finished
    let check = |api: &Api| -> Option<std::result::Result<Value, String>> {
        let jobs = api.inner.jobs.lock().unwrap();
        let j = jobs.get(job_id)?;
        match j.view.state.as_str() {
            "done" => Some(Ok(j.view.result.clone().unwrap_or(Value::Null))),
            "error" => Some(Err(j.view.error.clone().unwrap_or_default())),
            "cancelled" => Some(Err("cancelled".into())),
            _ => None,
        }
    };
    loop {
        if let Some(r) = check(api) {
            return r;
        }
        match tokio::time::timeout(std::time::Duration::from_millis(250), rx.recv()).await {
            Ok(Err(broadcast::error::RecvError::Closed)) => return Err("closed".into()),
            _ => {}
        }
    }
}
