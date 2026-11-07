//! Exporters: chat-template formats, containers (JSONL / JSON / Parquet), sharding, manifest.

use crate::jobs::{Ctl, Progress, Update};
use crate::mix::{MixPlan, PlanEntry, SourcePlan, SourceData};
use crate::model::{Message, Role, Sample};
use crate::recipe::{Container, ExportOptions, Format, OnFail, OutputMode, Recipe};
use crate::render;
use anyhow::{Context, Result, bail};
use futures::future::BoxFuture;
use futures::stream::StreamExt;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

// ───────────────────────────── formatting ─────────────────────────────

/// Compiled custom Jinja chat template.
pub struct CustomTemplate {
    env: minijinja::Environment<'static>,
    bos: String,
    eos: String,
}

impl CustomTemplate {
    pub fn new(template: &str, bos: &str, eos: &str) -> Result<Self> {
        let mut env = minijinja::Environment::new();
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", |msg: String| -> Result<String, minijinja::Error> {
            Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, msg))
        });
        env.add_function("strftime_now", |_fmt: String| -> String { "2026-01-01".to_string() });
        env.add_template_owned("chat", template.to_string()).context("invalid chat template")?;
        Ok(Self { env, bos: bos.into(), eos: eos.into() })
    }

    pub fn render(&self, messages: &[Message]) -> Result<String> {
        let msgs: Vec<Value> = messages.iter().map(|m| json!({"role": m.role.as_str(), "content": m.content})).collect();
        let t = self.env.get_template("chat")?;
        Ok(t.render(minijinja::context! {
            messages => minijinja::Value::from(minijinja::value::Serde(&msgs)),
            bos_token => self.bos.clone(),
            eos_token => self.eos.clone(),
            add_generation_prompt => false,
        })?)
    }
}

/// Pull `chat_template`, `bos_token` and `eos_token` out of a HF `tokenizer_config.json`.
pub fn parse_tokenizer_config(text: &str) -> Result<(String, String, String)> {
    let v: Value = serde_json::from_str(text)?;
    let tok = |k: &str| -> String {
        match v.get(k) {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Object(o)) => o.get("content").and_then(|c| c.as_str()).unwrap_or("").to_string(),
            _ => String::new(),
        }
    };
    let template = match v.get("chat_template") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(a)) => a
            .iter()
            .find(|x| x.get("name").and_then(|n| n.as_str()) == Some("default"))
            .or_else(|| a.first())
            .and_then(|x| x.get("template"))
            .and_then(|t| t.as_str())
            .unwrap_or("")
            .to_string(),
        _ => String::new(),
    };
    if template.is_empty() {
        bail!("no chat_template found in the tokenizer config");
    }
    Ok((template, tok("bos_token"), tok("eos_token")))
}

fn merge_consecutive(msgs: &[Message]) -> Vec<Message> {
    let mut out: Vec<Message> = Vec::with_capacity(msgs.len());
    for m in msgs {
        match out.last_mut() {
            Some(l) if l.role == m.role => {
                l.content.push_str("\n\n");
                l.content.push_str(&m.content);
            }
            _ => out.push(m.clone()),
        }
    }
    out
}

pub fn transcript(s: &Sample) -> String {
    if let Some(t) = &s.text {
        if s.messages.is_empty() {
            return t.clone();
        }
    }
    s.messages
        .iter()
        .map(|m| {
            let label = match m.role {
                Role::System => "System",
                Role::User => "User",
                Role::Assistant => "Assistant",
                Role::Tool => "Tool",
            };
            format!("{label}: {}", m.content)
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn chatml(msgs: &[Message]) -> String {
    let mut out = String::new();
    for m in msgs {
        out.push_str("<|im_start|>");
        out.push_str(m.role.as_str());
        out.push('\n');
        out.push_str(&m.content);
        out.push_str("<|im_end|>\n");
    }
    out
}

fn llama3(msgs: &[Message]) -> String {
    let mut out = String::from("<|begin_of_text|>");
    for m in msgs {
        let role = if m.role == Role::Tool { "ipython" } else { m.role.as_str() };
        out.push_str("<|start_header_id|>");
        out.push_str(role);
        out.push_str("<|end_header_id|>\n\n");
        out.push_str(&m.content);
        out.push_str("<|eot_id|>");
    }
    out
}

fn mistral(msgs: &[Message]) -> String {
    let mut out = String::from("<s>");
    let mut pending: Vec<String> = Vec::new();
    for m in msgs {
        match m.role {
            Role::System | Role::User | Role::Tool => pending.push(m.content.clone()),
            Role::Assistant => {
                if !pending.is_empty() {
                    out.push_str("[INST] ");
                    out.push_str(&pending.join("\n\n"));
                    out.push_str(" [/INST]");
                    pending.clear();
                }
                out.push(' ');
                out.push_str(&m.content);
                out.push_str("</s>");
            }
        }
    }
    if !pending.is_empty() {
        out.push_str("[INST] ");
        out.push_str(&pending.join("\n\n"));
        out.push_str(" [/INST]");
    }
    out
}

fn gemma(msgs: &[Message]) -> String {
    let mut out = String::from("<bos>");
    let mut sys: Option<String> = None;
    for m in merge_consecutive(msgs) {
        match m.role {
            Role::System => sys = Some(m.content),
            Role::User | Role::Tool => {
                out.push_str("<start_of_turn>user\n");
                if let Some(s) = sys.take() {
                    out.push_str(&s);
                    out.push_str("\n\n");
                }
                out.push_str(&m.content);
                out.push_str("<end_of_turn>\n");
            }
            Role::Assistant => {
                if let Some(s) = sys.take() {
                    out.push_str("<start_of_turn>user\n");
                    out.push_str(&s);
                    out.push_str("<end_of_turn>\n");
                }
                out.push_str("<start_of_turn>model\n");
                out.push_str(&m.content);
                out.push_str("<end_of_turn>\n");
            }
        }
    }
    out
}

fn phi3(msgs: &[Message]) -> String {
    let mut out = String::new();
    for m in msgs {
        let tag = if m.role == Role::Tool { "user" } else { m.role.as_str() };
        out.push_str(&format!("<|{tag}|>\n{}<|end|>\n", m.content));
    }
    out
}

fn metharme(msgs: &[Message]) -> String {
    let mut out = String::new();
    for m in msgs {
        let tag = match m.role {
            Role::System => "system",
            Role::User | Role::Tool => "user",
            Role::Assistant => "model",
        };
        out.push_str(&format!("<|{tag}|>{}", m.content));
    }
    out
}

fn sharegpt_role(r: Role) -> &'static str {
    match r {
        Role::System => "system",
        Role::User => "human",
        Role::Assistant => "gpt",
        Role::Tool => "tool",
    }
}

/// Format one finalized sample into the JSON row of the chosen format.
/// Returns `None` when the sample cannot be represented (e.g. no assistant turn for Alpaca).
pub fn format_sample(fmt: Format, s: &Sample, custom: Option<&CustomTemplate>) -> Option<Value> {
    // Raw-text samples carry no turns.
    if s.messages.is_empty() {
        let t = s.text.as_ref()?;
        return Some(match fmt {
            Format::PromptCompletion => json!({"prompt": "", "completion": t}),
            _ => json!({"text": t}),
        });
    }
    let msgs = &s.messages;
    Some(match fmt {
        Format::Messages => json!({"messages": msgs.iter().map(|m| json!({"role": m.role.as_str(), "content": m.content})).collect::<Vec<_>>()}),
        Format::Sharegpt => json!({"conversations": msgs.iter().map(|m| json!({"from": sharegpt_role(m.role), "value": m.content})).collect::<Vec<_>>()}),
        Format::Chatml => json!({"text": chatml(msgs)}),
        Format::Llama3 => json!({"text": llama3(msgs)}),
        Format::Mistral => json!({"text": mistral(msgs)}),
        Format::Gemma => json!({"text": gemma(msgs)}),
        Format::Phi3 => json!({"text": phi3(msgs)}),
        Format::Metharme => json!({"text": metharme(msgs)}),
        Format::Text => json!({"text": transcript(s)}),
        Format::Custom => json!({"text": custom?.render(msgs).ok()?}),
        Format::Alpaca => {
            let sys = msgs.iter().find(|m| m.role == Role::System).map(|m| m.content.clone());
            let user = msgs.iter().find(|m| m.role == Role::User)?;
            let asst = msgs.iter().skip_while(|m| m.role != Role::User).find(|m| m.role == Role::Assistant)?;
            let mut v = json!({"instruction": user.content, "input": "", "output": asst.content});
            if let Some(sys) = sys {
                v["system"] = json!(sys);
            }
            v
        }
        Format::PromptCompletion => {
            let last = msgs.iter().rposition(|m| m.role == Role::Assistant)?;
            let history = &msgs[..last];
            let users = history.iter().filter(|m| m.role != Role::System).count();
            let prompt = if users == 1 && history.iter().filter(|m| m.role == Role::System).count() == 0 {
                history.iter().find(|m| m.role == Role::User)?.content.clone()
            } else {
                let tmp = Sample { messages: history.to_vec(), ..s.clone() };
                transcript(&tmp)
            };
            json!({"prompt": prompt, "completion": msgs[last].content})
        }
    })
}

// ───────────────────────────── containers ─────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInfo {
    pub name: String,
    pub split: String,
    pub rows: u64,
    pub bytes: u64,
    pub xxh3: String,
}

enum RowFile {
    Jsonl { w: BufWriter<File> },
    Json { w: BufWriter<File>, first: bool },
    Parquet(ParquetFile),
}

struct ParquetFile {
    path: PathBuf,
    buf: Vec<u8>,
    buffered: usize,
    writer: Option<parquet::arrow::ArrowWriter<File>>,
    schema: Option<arrow_schema::SchemaRef>,
}

const PARQUET_BATCH: usize = 8192;

impl ParquetFile {
    fn flush(&mut self) -> Result<()> {
        if self.buffered == 0 {
            return Ok(());
        }
        let schema = match &self.schema {
            Some(s) => s.clone(),
            None => {
                let (schema, _) = arrow_json::reader::infer_json_schema(&mut std::io::Cursor::new(&self.buf), None)?;
                Arc::new(schema)
            }
        };
        if self.writer.is_none() {
            let props = parquet::file::properties::WriterProperties::builder()
                .set_compression(parquet::basic::Compression::ZSTD(parquet::basic::ZstdLevel::default()))
                .build();
            self.writer = Some(parquet::arrow::ArrowWriter::try_new(File::create(&self.path)?, schema.clone(), Some(props))?);
            self.schema = Some(schema.clone());
        }
        let reader = arrow_json::ReaderBuilder::new(schema).with_batch_size(PARQUET_BATCH).build(std::io::Cursor::new(&self.buf))?;
        for batch in reader {
            self.writer.as_mut().unwrap().write(&batch?)?;
        }
        self.buf.clear();
        self.buffered = 0;
        Ok(())
    }
}

impl RowFile {
    fn create(path: &Path, c: Container) -> Result<Self> {
        Ok(match c {
            Container::Jsonl => RowFile::Jsonl { w: BufWriter::with_capacity(1 << 20, File::create(path)?) },
            Container::Json => {
                let mut w = BufWriter::with_capacity(1 << 20, File::create(path)?);
                w.write_all(b"[\n")?;
                RowFile::Json { w, first: true }
            }
            Container::Parquet => RowFile::Parquet(ParquetFile { path: path.to_path_buf(), buf: Vec::new(), buffered: 0, writer: None, schema: None }),
        })
    }

    fn write(&mut self, v: &Value) -> Result<()> {
        match self {
            RowFile::Jsonl { w } => {
                serde_json::to_writer(&mut *w, v)?;
                w.write_all(b"\n")?;
            }
            RowFile::Json { w, first } => {
                if !*first {
                    w.write_all(b",\n")?;
                }
                *first = false;
                serde_json::to_writer(&mut *w, v)?;
            }
            RowFile::Parquet(p) => {
                serde_json::to_writer(&mut p.buf, v)?;
                p.buf.push(b'\n');
                p.buffered += 1;
                if p.buffered >= PARQUET_BATCH {
                    p.flush()?;
                }
            }
        }
        Ok(())
    }

    fn close(self) -> Result<()> {
        match self {
            RowFile::Jsonl { mut w } => w.flush()?,
            RowFile::Json { mut w, .. } => {
                w.write_all(b"\n]\n")?;
                w.flush()?;
            }
            RowFile::Parquet(mut p) => {
                p.flush()?;
                match p.writer.take() {
                    Some(w) => {
                        w.close()?;
                    }
                    None => {
                        // zero rows: write an empty parquet file with a single text column
                        let schema = Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new("text", arrow_schema::DataType::Utf8, true)]));
                        parquet::arrow::ArrowWriter::try_new(File::create(&p.path)?, schema, None)?.close()?;
                    }
                }
            }
        }
        Ok(())
    }
}

fn ext(c: Container) -> &'static str {
    match c {
        Container::Jsonl => "jsonl",
        Container::Json => "json",
        Container::Parquet => "parquet",
    }
}

fn hash_file(path: &Path) -> Result<(u64, String)> {
    let mut f = File::open(path)?;
    let mut h = xxhash_rust::xxh3::Xxh3::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut total = 0u64;
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
        total += n as u64;
    }
    Ok((total, format!("{:016x}", h.digest())))
}

/// Writes rows of one split into one or several (sharded) files.
struct SplitWriter {
    dir: PathBuf,
    split: String,
    container: Container,
    shard_size: Option<u64>,
    cur: Option<(RowFile, PathBuf)>,
    cur_rows: u64,
    shards: Vec<(PathBuf, u64)>,
    total_rows: u64,
}

impl SplitWriter {
    fn new(dir: &Path, split: &str, container: Container, shard_size: Option<u64>) -> Self {
        Self { dir: dir.to_path_buf(), split: split.into(), container, shard_size: shard_size.filter(|s| *s > 0), cur: None, cur_rows: 0, shards: vec![], total_rows: 0 }
    }

    fn write(&mut self, v: &Value) -> Result<()> {
        if self.cur.is_none() {
            let name = format!("{}-{:05}.{}.part", self.split, self.shards.len(), ext(self.container));
            let path = self.dir.join(name);
            self.cur = Some((RowFile::create(&path, self.container)?, path));
            self.cur_rows = 0;
        }
        self.cur.as_mut().unwrap().0.write(v)?;
        self.cur_rows += 1;
        self.total_rows += 1;
        if self.shard_size.is_some_and(|s| self.cur_rows >= s) {
            self.rotate()?;
        }
        Ok(())
    }

    fn rotate(&mut self) -> Result<()> {
        if let Some((f, p)) = self.cur.take() {
            f.close()?;
            self.shards.push((p, self.cur_rows));
            self.cur_rows = 0;
        }
        Ok(())
    }

    fn finish(mut self) -> Result<Vec<FileInfo>> {
        self.rotate()?;
        if self.shards.is_empty() && self.total_rows == 0 {
            return Ok(vec![]);
        }
        let n = self.shards.len();
        let mut out = Vec::new();
        for (i, (p, rows)) in self.shards.iter().enumerate() {
            let name = if self.shard_size.is_some() {
                format!("{}-{:05}-of-{:05}.{}", self.split, i, n, ext(self.container))
            } else {
                format!("{}.{}", self.split, ext(self.container))
            };
            let dst = self.dir.join(&name);
            std::fs::rename(p, &dst)?;
            let (bytes, xxh3) = hash_file(&dst)?;
            out.push(FileInfo { name, split: self.split.clone(), rows: *rows, bytes, xxh3 });
        }
        Ok(out)
    }
}

// ───────────────────────────── manifest & job ─────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub tool: String,
    pub created_unix: u64,
    pub recipe: Recipe,
    pub files: Vec<FileInfo>,
    pub per_source: Vec<SourcePlan>,
    pub rows_written: u64,
    pub samples_skipped: u64,
    pub translation_failed: u64,
    pub duration_secs: f64,
}

/// Result of translating one sample.
pub enum Translated {
    Ok(Sample),
    Failed(String),
}

pub trait SampleTranslator: Send + Sync {
    fn translate(&self, sample: Sample) -> BoxFuture<'_, Translated>;
    fn concurrency(&self) -> usize;
}

pub struct ExportJob<'a> {
    pub recipe: &'a Recipe,
    pub sources: &'a [SourceData],
    pub plan: &'a MixPlan,
    pub out_dir: PathBuf,
    pub translator: Option<Arc<dyn SampleTranslator>>,
}

struct Ctx {
    fmt: Format,
    custom: Option<CustomTemplate>,
    persona: crate::recipe::PersonaOptions,
    window: Option<usize>,
    bilingual: bool,
}

fn rows_for(ctx: &Ctx, translated: &Sample, original: Option<&Sample>) -> Vec<Value> {
    let fin = render::finalize(translated, &ctx.persona);
    let windows = match ctx.window {
        Some(w) => render::window(&fin, w),
        None => vec![fin],
    };
    let orig_windows = if ctx.bilingual { original.map(|o| render::finalize(o, &ctx.persona)) } else { None };
    windows
        .iter()
        .filter_map(|w| {
            let mut v = format_sample(ctx.fmt, w, ctx.custom.as_ref())?;
            if let (Some(o), Some(obj)) = (&orig_windows, v.as_object_mut()) {
                if let Some(ov) = format_sample(ctx.fmt, o, ctx.custom.as_ref()) {
                    obj.insert("original".into(), ov);
                }
            }
            Some(v)
        })
        .collect()
}

/// Format a single sample the way the exporter would (used by the UI preview).
pub fn preview_row(recipe: &Recipe, s: &Sample) -> Result<Vec<Value>> {
    let custom = custom_template(&recipe.export)?;
    let ctx = Ctx {
        fmt: recipe.export.format,
        custom,
        persona: recipe.persona.clone(),
        window: recipe.filters.window_tokens,
        bilingual: false,
    };
    Ok(rows_for(&ctx, s, None))
}

pub fn custom_template(o: &ExportOptions) -> Result<Option<CustomTemplate>> {
    if o.format != Format::Custom {
        return Ok(None);
    }
    let t = o.custom_template.as_deref().filter(|t| !t.trim().is_empty()).context("choose a Jinja chat template for the custom format")?;
    Ok(Some(CustomTemplate::new(t, &o.bos_token, &o.eos_token)?))
}

pub async fn export(job: ExportJob<'_>, ctl: &Ctl, progress: &Progress) -> Result<Manifest> {
    let started = Instant::now();
    let r = job.recipe;
    let dir = job.out_dir.clone();
    std::fs::create_dir_all(&dir).with_context(|| format!("cannot create {}", dir.display()))?;
    let ctx = Ctx {
        fmt: r.export.format,
        custom: custom_template(&r.export)?,
        persona: r.persona.clone(),
        window: r.filters.window_tokens,
        bilingual: r.translate.enabled && r.translate.output_mode == OutputMode::Bilingual,
    };
    let sources: std::collections::HashMap<usize, &SourceData> = job.sources.iter().map(|s| (s.index, s)).collect();

    let total = job.plan.entries.len() as u64;
    let done = AtomicU64::new(0);
    let skipped = AtomicU64::new(0);
    let failed = AtomicU64::new(0);
    let mut files: Vec<FileInfo> = Vec::new();
    let mut rows_written = 0u64;

    let mut last_report = Instant::now();
    let report = |progress: &Progress, extra: &str| {
        let d = done.load(Ordering::Relaxed);
        let mut u = Update::new("export", d, total, format!("Exporting {d}/{total}{extra}"));
        u.stats.insert("skipped".into(), json!(skipped.load(Ordering::Relaxed)));
        u.stats.insert("failed".into(), json!(failed.load(Ordering::Relaxed)));
        progress(u);
    };

    let splits: [(&str, &[PlanEntry]); 2] = [("train", job.plan.train()), ("validation", job.plan.validation())];
    for (name, entries) in splits {
        if entries.is_empty() {
            continue;
        }
        let mut w = SplitWriter::new(&dir, name, r.export.container, r.export.shard_size);
        if let Some(tr) = job.translator.as_ref().filter(|_| r.translate.enabled) {
            let conc = tr.concurrency().max(1) * 2;
            let ctx_ref = &ctx;
            let sources_ref = &sources;
            let tr = tr.clone();
            let on_fail = r.translate.on_fail;
            let mut stream = futures::stream::iter(entries.iter().copied())
                .map(|e| {
                    let tr = tr.clone();
                    let failed = &failed;
                    let skipped = &skipped;
                    async move {
                        let Some(sd) = sources_ref.get(&(e.source as usize)) else { return vec![] };
                        let Some(sample) = sd.sample(e.row) else {
                            skipped.fetch_add(1, Ordering::Relaxed);
                            return vec![];
                        };
                        let original = sample.clone();
                        match tr.translate(sample).await {
                            Translated::Ok(t) => rows_for(ctx_ref, &t, Some(&original)),
                            Translated::Failed(_) => {
                                failed.fetch_add(1, Ordering::Relaxed);
                                match on_fail {
                                    OnFail::Drop => vec![],
                                    OnFail::KeepOriginal => rows_for(ctx_ref, &original, None),
                                }
                            }
                        }
                    }
                })
                .buffered(conc);
            while let Some(rows) = stream.next().await {
                ctl.check_async().await?;
                for v in rows {
                    w.write(&v)?;
                }
                done.fetch_add(1, Ordering::Relaxed);
                if last_report.elapsed().as_millis() > 200 {
                    report(progress, "");
                    last_report = Instant::now();
                }
            }
        } else {
            for chunk in entries.chunks(4096) {
                ctl.check()?;
                let out: Vec<Vec<Value>> = chunk
                    .par_iter()
                    .map(|e| {
                        let Some(sd) = sources.get(&(e.source as usize)) else { return vec![] };
                        match sd.sample(e.row) {
                            Some(s) => rows_for(&ctx, &s, None),
                            None => {
                                skipped.fetch_add(1, Ordering::Relaxed);
                                vec![]
                            }
                        }
                    })
                    .collect();
                for rows in out {
                    for v in rows {
                        w.write(&v)?;
                    }
                }
                done.fetch_add(chunk.len() as u64, Ordering::Relaxed);
                if last_report.elapsed().as_millis() > 200 {
                    report(progress, "");
                    last_report = Instant::now();
                }
            }
        }
        rows_written += w.total_rows;
        files.extend(w.finish()?);
    }
    report(progress, "");

    let manifest = Manifest {
        tool: format!("dataset-mixer {}", env!("CARGO_PKG_VERSION")),
        created_unix: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs()),
        recipe: r.clone(),
        files,
        per_source: job.plan.per_source.clone(),
        rows_written,
        samples_skipped: skipped.load(Ordering::Relaxed),
        translation_failed: failed.load(Ordering::Relaxed),
        duration_secs: started.elapsed().as_secs_f64(),
    };
    std::fs::write(dir.join("manifest.json"), serde_json::to_vec_pretty(&manifest)?)?;
    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Sample {
        Sample {
            id: 1,
            source: 0,
            row: 0,
            system: None,
            messages: vec![
                Message::new(Role::System, "Be brief."),
                Message::new(Role::User, "Hi"),
                Message::new(Role::Assistant, "Hello!"),
                Message::new(Role::User, "Bye"),
                Message::new(Role::Assistant, "Ciao"),
            ],
            text: None,
            persona: None,
        }
    }

    fn text_of(fmt: Format) -> String {
        format_sample(fmt, &sample(), None).unwrap()["text"].as_str().unwrap().to_string()
    }

    #[test]
    fn chatml_exact() {
        assert_eq!(
            text_of(Format::Chatml),
            "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello!<|im_end|>\n<|im_start|>user\nBye<|im_end|>\n<|im_start|>assistant\nCiao<|im_end|>\n"
        );
    }

    #[test]
    fn llama3_exact() {
        let t = text_of(Format::Llama3);
        assert!(t.starts_with("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nBe brief.<|eot_id|>"));
        assert!(t.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\nCiao<|eot_id|>"));
    }

    #[test]
    fn mistral_gemma_phi_metharme() {
        assert_eq!(text_of(Format::Mistral), "<s>[INST] Be brief.\n\nHi [/INST] Hello!</s>[INST] Bye [/INST] Ciao</s>");
        assert_eq!(
            text_of(Format::Gemma),
            "<bos><start_of_turn>user\nBe brief.\n\nHi<end_of_turn>\n<start_of_turn>model\nHello!<end_of_turn>\n<start_of_turn>user\nBye<end_of_turn>\n<start_of_turn>model\nCiao<end_of_turn>\n"
        );
        assert!(text_of(Format::Phi3).starts_with("<|system|>\nBe brief.<|end|>\n<|user|>\nHi<|end|>\n"));
        assert_eq!(text_of(Format::Metharme), "<|system|>Be brief.<|user|>Hi<|model|>Hello!<|user|>Bye<|model|>Ciao");
    }

    #[test]
    fn structured_formats() {
        let v = format_sample(Format::Sharegpt, &sample(), None).unwrap();
        assert_eq!(v["conversations"][1]["from"], "human");
        assert_eq!(v["conversations"][2]["from"], "gpt");
        let v = format_sample(Format::Alpaca, &sample(), None).unwrap();
        assert_eq!(v["instruction"], "Hi");
        assert_eq!(v["output"], "Hello!");
        assert_eq!(v["system"], "Be brief.");
        let v = format_sample(Format::PromptCompletion, &sample(), None).unwrap();
        assert_eq!(v["completion"], "Ciao");
        assert!(v["prompt"].as_str().unwrap().contains("User: Bye"));
        let v = format_sample(Format::Messages, &sample(), None).unwrap();
        assert_eq!(v["messages"].as_array().unwrap().len(), 5);
    }

    #[test]
    fn custom_jinja_matches_chatml_template() {
        let tpl = "{% for m in messages %}{{'<|im_start|>' + m['role'] + '\\n' + m['content'] + '<|im_end|>' + '\\n'}}{% endfor %}";
        let c = CustomTemplate::new(tpl, "", "").unwrap();
        let v = format_sample(Format::Custom, &sample(), Some(&c)).unwrap();
        assert_eq!(v["text"].as_str().unwrap(), text_of(Format::Chatml));
        // HF style template with bos/eos and python methods
        let tpl2 = "{{ bos_token }}{% for m in messages %}{{ m['role'].upper() }}: {{ m['content']|trim }}{{ eos_token }}\n{% endfor %}";
        let c = CustomTemplate::new(tpl2, "<s>", "</s>").unwrap();
        assert!(c.render(&sample().messages).unwrap().starts_with("<s>SYSTEM: Be brief.</s>\n"));
    }

    #[test]
    fn tokenizer_config_parsing() {
        let (t, b, e) = parse_tokenizer_config(r#"{"chat_template":"{{messages}}","bos_token":"<s>","eos_token":{"content":"</s>"}}"#).unwrap();
        assert_eq!((t.as_str(), b.as_str(), e.as_str()), ("{{messages}}", "<s>", "</s>"));
        let (t, _, _) = parse_tokenizer_config(r#"{"chat_template":[{"name":"x","template":"A"},{"name":"default","template":"B"}]}"#).unwrap();
        assert_eq!(t, "B");
        assert!(parse_tokenizer_config("{}").is_err());
    }

    #[test]
    fn writes_sharded_jsonl_json_and_parquet() {
        for c in [Container::Jsonl, Container::Json, Container::Parquet] {
            let d = tempfile::tempdir().unwrap();
            let mut w = SplitWriter::new(d.path(), "train", c, Some(4));
            for i in 0..10 {
                w.write(&json!({"text": format!("row {i}"), "n": i})).unwrap();
            }
            let files = w.finish().unwrap();
            assert_eq!(files.len(), 3, "{c:?}");
            assert_eq!(files.iter().map(|f| f.rows).sum::<u64>(), 10);
            assert_eq!(files[0].name, format!("train-00000-of-00003.{}", ext(c)));
            assert!(d.path().join(&files[2].name).exists());
        }
    }
}
