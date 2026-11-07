//! Orchestration shared by the CLI and the app: resolve sources, open + detect, scan, plan,
//! translate and export.

use crate::detect::{Detection, Mapping, detect_prepared};
use crate::export::{self, ExportJob, Manifest, SampleTranslator};
use crate::hf;
use crate::jobs::{Ctl, Progress, Update};
use crate::mix::{self, MixPlan, Scan, SourceData};
use crate::recipe::{Recipe, SourceLocation};
use crate::source::{self, FileFormat};
use crate::translate::{Translator, journal::Journal};
use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub fn default_cache_dir() -> PathBuf {
    std::env::var_os("MIXER_CACHE_DIR")
        .map(PathBuf::from)
        .or_else(|| dirs::cache_dir().map(|d| d.join("dataset-mixer")))
        .unwrap_or_else(|| PathBuf::from(".mixer-cache"))
}

pub fn journal_path(cache_dir: &Path) -> PathBuf {
    cache_dir.join("translations.sqlite")
}

pub struct Loaded {
    pub data: SourceData,
    pub detection: Detection,
    pub rows: usize,
    pub bytes: u64,
    pub format: FileFormat,
}

/// Turn a source location into a local file path (downloading from Hugging Face if needed).
pub async fn resolve_location(loc: &SourceLocation, cache_dir: &Path, hf_token: &Option<String>, ctl: &Ctl, progress: &Progress) -> Result<PathBuf> {
    match loc {
        SourceLocation::File { path } => {
            let p = PathBuf::from(shellexpand_home(path));
            if !p.exists() {
                bail!("{} does not exist", p.display());
            }
            Ok(p)
        }
        SourceLocation::Hf { repo, file, revision } => hf::download(repo, file, revision.as_deref(), hf_token, cache_dir, ctl, progress).await,
    }
}

fn shellexpand_home(p: &str) -> String {
    if let Some(rest) = p.strip_prefix("~/") {
        if let Some(h) = dirs::home_dir() {
            return h.join(rest).to_string_lossy().to_string();
        }
    }
    p.to_string()
}

/// Open a local file: materialize, index and detect its format (blocking, CPU/IO heavy).
pub fn open_source(index: usize, id: &str, path: &Path, mapping: Option<Mapping>, cache_dir: &Path, ctl: &Ctl, progress: &Progress) -> Result<Loaded> {
    let prepared = source::prepare(path, &cache_dir.join("prepared"), ctl, progress)?;
    progress(Update::new("detect", 0, 0, "Detecting format"));
    let detection = detect_prepared(&prepared);
    let rows = prepared.len();
    let bytes = prepared.bytes;
    let format = prepared.format;
    let mapping = mapping.unwrap_or_else(|| detection.best.clone());
    let data = SourceData { index, id: id.to_string(), prepared: Arc::new(prepared), mapping };
    Ok(Loaded { data, detection, rows, bytes, format })
}

pub struct RunOptions {
    pub cache_dir: PathBuf,
    pub hf_token: Option<String>,
    pub llm_token: Option<String>,
    /// Overrides `recipe.export.out_dir`.
    pub out_dir: Option<PathBuf>,
    pub translate: bool,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self { cache_dir: default_cache_dir(), hf_token: None, llm_token: None, out_dir: None, translate: true }
    }
}

/// Scan every enabled source and build the mix.
pub async fn build_mix(recipe: &Recipe, loaded: &[(usize, Arc<SourceData>)], ctl: &Ctl, progress: &Progress) -> Result<(MixPlan, Vec<Scan>)> {
    let mut scans = Vec::new();
    for (_, sd) in loaded {
        let sd = sd.clone();
        let filters = recipe.filters.clone();
        let (c, p) = (ctl.clone(), progress.clone());
        let scan = tokio::task::spawn_blocking(move || mix::scan_source(&sd, &filters, &c, &p)).await??;
        scans.push(scan);
    }
    let pairs: Vec<(&SourceData, &Scan)> = loaded.iter().zip(scans.iter()).map(|((_, sd), sc)| (sd.as_ref(), sc)).collect();
    let plan = mix::build_plan(recipe, &pairs, ctl)?;
    Ok((plan, scans))
}

/// Run a recipe from start to finish (what `mixer run` does).
pub async fn run_recipe(recipe: &Recipe, opts: &RunOptions, ctl: &Ctl, progress: &Progress) -> Result<Manifest> {
    let enabled: Vec<usize> = recipe.sources.iter().enumerate().filter(|(_, s)| s.enabled).map(|(i, _)| i).collect();
    if enabled.is_empty() {
        bail!("the recipe has no enabled sources");
    }
    let mut loaded: Vec<(usize, Arc<SourceData>)> = Vec::new();
    for &i in &enabled {
        let entry = &recipe.sources[i];
        progress(Update::new("prepare", 0, 0, format!("Opening {}", entry.id)));
        let path = resolve_location(&entry.location, &opts.cache_dir, &opts.hf_token, ctl, progress).await?;
        let (id, mapping, cache) = (entry.id.clone(), entry.mapping.clone(), opts.cache_dir.clone());
        let (c, p) = (ctl.clone(), progress.clone());
        let l = tokio::task::spawn_blocking(move || open_source(i, &id, &path, mapping, &cache, &c, &p))
            .await?
            .with_context(|| format!("opening source `{}`", entry.id))?;
        progress(Update::new("detect", 1, 1, format!("{}: {}", entry.id, l.detection.best.summary)));
        loaded.push((i, Arc::new(l.data)));
    }
    let (plan, _) = build_mix(recipe, &loaded, ctl, progress).await?;
    let sources: Vec<SourceData> = loaded.iter().map(|(_, s)| (**s).clone()).collect();

    let translator: Option<Arc<dyn SampleTranslator>> = if recipe.translate.enabled && opts.translate {
        std::fs::create_dir_all(&opts.cache_dir)?;
        let journal = Arc::new(Journal::open(&journal_path(&opts.cache_dir))?);
        let t = Translator::new(recipe, opts.llm_token.clone(), journal, ctl.clone())?;
        Some(t as Arc<dyn SampleTranslator>)
    } else {
        None
    };
    let out_dir = opts.out_dir.clone().unwrap_or_else(|| PathBuf::from(&recipe.export.out_dir));
    export::export(ExportJob { recipe, sources: &sources, plan: &plan, out_dir, translator }, ctl, progress).await
}
