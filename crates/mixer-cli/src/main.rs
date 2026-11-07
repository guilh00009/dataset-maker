//! `mixer`: headless dataset mixing, a mock LLM server and the local HTTP bridge for the UI.

mod bridge;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use mixer_core::jobs::{Ctl, Progress, Update};
use mixer_core::pipeline::{self, RunOptions};
use mixer_core::recipe::Recipe;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "mixer", version, about = "Mix, translate and export LLM training datasets")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run a recipe (TOML) from start to finish.
    Run {
        recipe: PathBuf,
        /// Output directory (overrides the recipe).
        #[arg(long)]
        out: Option<PathBuf>,
        /// Skip translation even if the recipe enables it.
        #[arg(long)]
        no_translate: bool,
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        /// Environment variable holding the LLM API token.
        #[arg(long, default_value = "MIXER_LLM_TOKEN")]
        llm_token_env: String,
    },
    /// Show how a file (or a Hugging Face dataset file) is detected and what a few rows look like.
    Inspect {
        /// Path, or `hf:owner/name/file.jsonl`
        source: String,
        #[arg(long, default_value_t = 2)]
        samples: usize,
        #[arg(long)]
        cache_dir: Option<PathBuf>,
    },
    /// Check an OpenAI-compatible server (llama.cpp, vLLM, …): models, token and a tiny translation.
    TestLlm {
        #[arg(long, default_value = "http://127.0.0.1:8080/v1")]
        url: String,
        #[arg(long, default_value = "")]
        model: String,
        #[arg(long, env = "MIXER_LLM_TOKEN")]
        token: Option<String>,
    },
    /// Start a mock OpenAI-compatible server (for trying the app without a model).
    MockLlm {
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8089)]
        port: u16,
        #[arg(long, default_value = "mock-translator-7b")]
        model: String,
        #[arg(long)]
        api_key: Option<String>,
        #[arg(long, default_value_t = 4)]
        slots: usize,
        #[arg(long, default_value_t = 20)]
        latency_ms: u64,
        #[arg(long, default_value_t = 2)]
        ms_per_token: u64,
        /// Probability of answering 429 with Retry-After.
        #[arg(long, default_value_t = 0.0)]
        rate_429: f64,
        #[arg(long, default_value_t = 0.0)]
        rate_500: f64,
        /// Also serve a fake Hugging Face Hub from this directory (<owner>__<name>/…).
        #[arg(long)]
        hf_dir: Option<PathBuf>,
    },
    /// Serve the UI and its API on localhost (the same code the desktop app uses).
    Serve {
        /// Directory with the built UI (app/ui/dist).
        #[arg(long)]
        ui: Option<PathBuf>,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8787)]
        port: u16,
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        #[arg(long)]
        config_dir: Option<PathBuf>,
    },
}

fn console_progress() -> Progress {
    let last = std::sync::Mutex::new((String::new(), std::time::Instant::now() - std::time::Duration::from_secs(5)));
    Arc::new(move |u: Update| {
        let mut l = last.lock().unwrap();
        let done = u.total > 0 && u.done >= u.total;
        if l.1.elapsed().as_millis() < 500 && !done && l.0 == u.phase {
            return;
        }
        *l = (u.phase.clone(), std::time::Instant::now());
        if u.total > 0 {
            eprintln!("[{}] {} ({:.0}%)", u.phase, u.message, 100.0 * u.done as f64 / u.total as f64);
        } else {
            eprintln!("[{}] {}", u.phase, u.message);
        }
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Run { recipe, out, no_translate, cache_dir, llm_token_env } => {
            let text = std::fs::read_to_string(&recipe).with_context(|| format!("cannot read {}", recipe.display()))?;
            let r = Recipe::from_toml(&text).with_context(|| format!("invalid recipe {}", recipe.display()))?;
            let opts = RunOptions {
                cache_dir: cache_dir.unwrap_or_else(pipeline::default_cache_dir),
                hf_token: std::env::var("HF_TOKEN").ok(),
                llm_token: std::env::var(&llm_token_env).ok(),
                out_dir: out,
                translate: !no_translate,
            };
            let ctl = Ctl::new();
            let c2 = ctl.clone();
            tokio::spawn(async move {
                let _ = tokio::signal::ctrl_c().await;
                eprintln!("\nStopping… (finished translations are kept; run again to resume)");
                c2.cancel();
            });
            let m = pipeline::run_recipe(&r, &opts, &ctl, &console_progress()).await?;
            println!("Wrote {} rows in {:.1}s", m.rows_written, m.duration_secs);
            for f in &m.files {
                println!("  {:<32} {:>10} rows  {:>12} bytes", f.name, f.rows, f.bytes);
            }
            if m.translation_failed > 0 {
                println!("  {} samples could not be translated", m.translation_failed);
            }
        }
        Cmd::Inspect { source, samples, cache_dir } => {
            let cache = cache_dir.unwrap_or_else(pipeline::default_cache_dir);
            let ctl = Ctl::new();
            let progress = console_progress();
            let path = if let Some(rest) = source.strip_prefix("hf:") {
                let r = mixer_core::hf::parse_ref(rest)?;
                let file = r.file.context("give a file: hf:owner/name/file.jsonl")?;
                mixer_core::hf::download(&r.repo, &file, r.revision.as_deref(), &std::env::var("HF_TOKEN").ok(), &cache, &ctl, &progress).await?
            } else {
                PathBuf::from(&source)
            };
            let l = tokio::task::spawn_blocking({
                let (c, p, ctl) = (cache.clone(), progress.clone(), ctl.clone());
                move || pipeline::open_source(0, "source", &path, None, &c, &ctl, &p)
            })
            .await??;
            println!("format      : {:?}", l.format);
            println!("rows        : {}", l.rows);
            println!("size        : {:.1} MB", l.bytes as f64 / 1e6);
            println!("detected    : {} ({:.0}% confidence)", l.detection.best.summary, l.detection.best.confidence * 100.0);
            for c in l.detection.candidates.iter().skip(1) {
                println!("  alt       : {} ({:.0}%)", c.summary, c.confidence * 100.0);
            }
            let recipe = Recipe::default();
            let step = (l.rows / samples.max(1)).max(1);
            for k in 0..samples.min(l.rows) {
                if let Some(s) = l.data.sample((k * step) as u32) {
                    println!("\n--- row {} ---", k * step);
                    for row in mixer_core::export::preview_row(&recipe, &s)? {
                        println!("{}", row["text"].as_str().map(|t| t.to_string()).unwrap_or_else(|| row.to_string()));
                    }
                }
            }
        }
        Cmd::TestLlm { url, model, token } => {
            let r = mixer_core::translate::client::test_connection(&url, &model, token).await;
            println!("{}", serde_json::to_string_pretty(&r)?);
            if !r.ok {
                std::process::exit(1);
            }
        }
        Cmd::MockLlm { host, port, model, api_key, slots, latency_ms, ms_per_token, rate_429, rate_500, hf_dir } => {
            let cfg = mixer_mock::MockConfig { model: model.clone(), api_key, slots, base_latency_ms: latency_ms, ms_per_token, rate_429, rate_500, ..Default::default() };
            let addr: std::net::SocketAddr = format!("{host}:{port}").parse()?;
            let (bound, _state, handle) = mixer_mock::serve_full(cfg, addr, hf_dir).await?;
            println!("Mock LLM listening on http://{bound}/v1  (model: {model})");
            println!("Point the app at this URL to try translation without a real model.");
            handle.await?;
        }
        Cmd::Serve { ui, host, port, cache_dir, config_dir } => {
            let ui = ui.or_else(|| {
                ["app/ui/dist", "ui/dist", "dist"].iter().map(PathBuf::from).find(|p| p.join("index.html").exists())
            });
            if ui.is_none() {
                eprintln!("note: no built UI found (build it with `npm run build` in app/ui, or pass --ui); only /api is served");
            }
            let api = mixer_api::Api::new(cache_dir, config_dir)?;
            api.init().await;
            bridge::serve(api, ui, &host, port).await?;
        }
    }
    Ok(())
}
