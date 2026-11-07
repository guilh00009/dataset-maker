//! Hugging Face Hub datasets: listing repository files and resumable downloads.
//!
//! Works with datasets that ship a single big file and a loader script (like
//! `PygmalionAI/PIPPA`), not only with Parquet-converted viewer datasets. The endpoint can be
//! overridden with `HF_ENDPOINT`, which is also how the tests point at a local mock Hub.

use crate::jobs::{Ctl, Progress, Update};
use anyhow::{Context, Result, bail};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

pub fn endpoint() -> String {
    std::env::var("HF_ENDPOINT").ok().filter(|s| !s.trim().is_empty()).unwrap_or_else(|| "https://huggingface.co".into()).trim_end_matches('/').to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HfFile {
    pub path: String,
    pub size: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HfRef {
    pub repo: String,
    pub file: Option<String>,
    pub revision: Option<String>,
}

/// Accepts `owner/name`, `hf:owner/name`, `hf:owner/name/path/file.jsonl`,
/// `hf://datasets/owner/name/path`, and huggingface.co URLs (`…/datasets/o/n[/blob|resolve|tree/rev/path]`).
pub fn parse_ref(input: &str) -> Result<HfRef> {
    let mut s = input.trim().to_string();
    for p in ["hf://datasets/", "hf-file:", "hf:", "huggingface:"] {
        if let Some(r) = s.strip_prefix(p) {
            s = r.to_string();
            break;
        }
    }
    if let Some(i) = s.find("huggingface.co/") {
        s = s[i + "huggingface.co/".len()..].to_string();
        s = s.trim_start_matches("datasets/").to_string();
    }
    let s = s.split(['?', '#']).next().unwrap_or("").trim_matches('/').to_string();
    let parts: Vec<&str> = s.split('/').filter(|p| !p.is_empty()).collect();
    if parts.len() < 2 {
        bail!("`{input}` is not a Hugging Face dataset id (expected owner/name)");
    }
    let repo = format!("{}/{}", parts[0], parts[1]);
    let rest = &parts[2..];
    if rest.is_empty() {
        return Ok(HfRef { repo, file: None, revision: None });
    }
    if matches!(rest[0], "blob" | "resolve" | "tree") && rest.len() >= 2 {
        let mut rev = rest[1].to_string();
        let mut idx = 2;
        // revisions like refs%2Fconvert%2Fparquet or refs/convert/parquet
        if rev == "refs" && rest.len() >= 4 && rest[2] == "convert" {
            rev = format!("refs/convert/{}", rest[3]);
            idx = 4;
        }
        let rev = rev.replace("%2F", "/");
        let file = rest[idx..].join("/");
        return Ok(HfRef { repo, file: if file.is_empty() { None } else { Some(file) }, revision: Some(rev) });
    }
    Ok(HfRef { repo, file: Some(rest.join("/")), revision: None })
}

fn client(timeout: Duration) -> Result<reqwest::Client> {
    Ok(reqwest::Client::builder().timeout(timeout).connect_timeout(Duration::from_secs(20)).user_agent("dataset-mixer").build()?)
}

fn auth(rb: reqwest::RequestBuilder, token: &Option<String>) -> reqwest::RequestBuilder {
    match token.as_deref().map(str::trim).filter(|t| !t.is_empty()) {
        Some(t) => rb.bearer_auth(t),
        None => rb,
    }
}

fn enc(s: &str) -> String {
    s.replace('/', "%2F")
}

fn explain(status: reqwest::StatusCode, repo: &str, authed: bool) -> String {
    match status.as_u16() {
        401 | 403 => {
            if authed {
                format!("Hugging Face refused access to `{repo}` (HTTP {status}). The dataset may be gated: accept its terms on the website and check your token.")
            } else {
                format!("`{repo}` needs a Hugging Face token (HTTP {status}). Add one in Settings.")
            }
        }
        404 => format!("Dataset `{repo}` (or that file/revision) was not found on Hugging Face."),
        _ => format!("Hugging Face answered HTTP {status} for `{repo}`."),
    }
}

/// List files of a dataset repository (recursively).
pub async fn list_files(repo: &str, revision: Option<&str>, token: &Option<String>) -> Result<Vec<HfFile>> {
    let c = client(Duration::from_secs(60))?;
    let rev = revision.unwrap_or("main");
    let mut url = Some(format!("{}/api/datasets/{repo}/tree/{}?recursive=true&expand=false", endpoint(), enc(rev)));
    let mut out = Vec::new();
    for _ in 0..200 {
        let Some(u) = url.take() else { break };
        let resp = auth(c.get(&u), token).send().await.with_context(|| format!("cannot reach {}", endpoint()))?;
        if !resp.status().is_success() {
            bail!(explain(resp.status(), repo, token.is_some()));
        }
        let next = resp
            .headers()
            .get(reqwest::header::LINK)
            .and_then(|l| l.to_str().ok())
            .and_then(|l| l.split(',').find(|p| p.contains("rel=\"next\"")))
            .and_then(|p| p.split('<').nth(1))
            .and_then(|p| p.split('>').next())
            .map(|s| s.to_string());
        let items: Vec<serde_json::Value> = resp.json().await?;
        for it in items {
            if it["type"] == "file" {
                let size = it["lfs"]["size"].as_u64().or_else(|| it["size"].as_u64()).unwrap_or(0);
                if let Some(p) = it["path"].as_str() {
                    out.push(HfFile { path: p.to_string(), size });
                }
            }
        }
        url = next;
    }
    out.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(out)
}

/// Fetch a small text file of a Hub repository (e.g. a model's `tokenizer_config.json`).
pub async fn fetch_text(repo_type_path: &str, repo: &str, file: &str, token: &Option<String>) -> Result<String> {
    let c = client(Duration::from_secs(60))?;
    let url = format!("{}/{}{repo}/resolve/main/{file}", endpoint(), repo_type_path);
    let resp = auth(c.get(&url), token).send().await.with_context(|| format!("cannot reach {}", endpoint()))?;
    if !resp.status().is_success() {
        bail!(explain(resp.status(), repo, token.is_some()));
    }
    Ok(resp.text().await?)
}

/// Files that look like data (what the UI offers in the picker).
pub fn is_data_file(path: &str) -> bool {
    let p = path.to_lowercase();
    [".jsonl", ".json", ".ndjson", ".parquet", ".csv", ".tsv", ".txt", ".jsonl.gz", ".json.gz", ".jsonl.zst", ".csv.gz", ".txt.gz", ".gz", ".zst"]
        .iter()
        .any(|e| p.ends_with(e))
        && !p.ends_with("tokenizer.json")
        && !p.ends_with("config.json")
        && !p.ends_with("dataset_infos.json")
        && !p.ends_with("package.json")
}

fn cache_path(cache_dir: &Path, repo: &str, revision: &str, file: &str) -> PathBuf {
    let rev: String = revision.chars().map(|c| if c.is_alphanumeric() || c == '.' || c == '-' { c } else { '_' }).collect();
    cache_dir.join("hf").join(repo.replace('/', "__")).join(rev).join(file)
}

/// Download one file (resumable). Returns the local path.
pub async fn download(
    repo: &str,
    file: &str,
    revision: Option<&str>,
    token: &Option<String>,
    cache_dir: &Path,
    ctl: &Ctl,
    progress: &Progress,
) -> Result<PathBuf> {
    let rev = revision.unwrap_or("main");
    let dest = cache_path(cache_dir, repo, rev, file);
    std::fs::create_dir_all(dest.parent().unwrap())?;
    let url = format!("{}/datasets/{repo}/resolve/{}/{}", endpoint(), enc(rev), file.split('/').map(|s| s.replace(' ', "%20")).collect::<Vec<_>>().join("/"));
    let c = client(Duration::from_secs(3600 * 6))?;

    // already complete?
    let remote_len = match auth(c.head(&url), token).send().await {
        Ok(r) if r.status().is_success() => r.headers().get(reqwest::header::CONTENT_LENGTH).and_then(|v| v.to_str().ok()).and_then(|v| v.parse::<u64>().ok()),
        Ok(r) if matches!(r.status().as_u16(), 401 | 403 | 404) => bail!(explain(r.status(), repo, token.is_some())),
        _ => None,
    };
    if let Ok(m) = std::fs::metadata(&dest) {
        if m.len() > 0 && remote_len.map_or(true, |l| l == m.len()) {
            progress(Update::new("download", m.len(), m.len(), format!("{file} (cached)")));
            return Ok(dest);
        }
    }

    let part = dest.with_extension(format!("{}.part", dest.extension().and_then(|e| e.to_str()).unwrap_or("dl")));
    let mut attempt = 0u32;
    loop {
        ctl.check()?;
        let have = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
        let mut rb = auth(c.get(&url), token);
        if have > 0 {
            rb = rb.header(reqwest::header::RANGE, format!("bytes={have}-"));
        }
        let step: Result<()> = async {
            let resp = rb.send().await.with_context(|| format!("cannot reach {}", endpoint()))?;
            let status = resp.status();
            if status.as_u16() == 416 {
                return Ok(()); // the part file already holds everything
            }
            if !status.is_success() {
                bail!(explain(status, repo, token.is_some()));
            }
            let resumed = status.as_u16() == 206;
            let total = resp.content_length().map(|l| l + if resumed { have } else { 0 });
            let mut f = if resumed {
                std::fs::OpenOptions::new().append(true).open(&part)?
            } else {
                std::fs::File::create(&part)?
            };
            let mut done = if resumed { have } else { 0 };
            let mut last = Instant::now();
            let started = Instant::now();
            let base = done;
            let mut stream = resp.bytes_stream();
            while let Some(chunk) = stream.next().await {
                ctl.check_async().await?;
                let chunk = chunk.context("download interrupted")?;
                f.write_all(&chunk)?;
                done += chunk.len() as u64;
                if last.elapsed().as_millis() > 200 {
                    let mbps = (done - base) as f64 / 1e6 / started.elapsed().as_secs_f64().max(0.001);
                    let mut u = Update::new("download", done, total.unwrap_or(0), format!("Downloading {file} · {mbps:.1} MB/s"));
                    u.stats.insert("mbps".into(), serde_json::json!(mbps));
                    progress(u);
                    last = Instant::now();
                }
            }
            f.flush()?;
            Ok(())
        }
        .await;
        match step {
            Ok(()) => break,
            Err(e) => {
                if e.downcast_ref::<crate::jobs::Cancelled>().is_some() {
                    return Err(e);
                }
                let msg = e.to_string();
                let permanent = msg.contains("not found") || msg.contains("needs a Hugging Face token") || msg.contains("refused access");
                attempt += 1;
                if permanent || attempt > 6 {
                    return Err(e);
                }
                tokio::time::sleep(Duration::from_secs(1 << attempt.min(4))).await;
            }
        }
    }
    if let Some(l) = remote_len {
        let got = std::fs::metadata(&part)?.len();
        if got != l {
            bail!("download of {file} is incomplete ({got} of {l} bytes); try again to resume");
        }
    }
    std::fs::rename(&part, &dest)?;
    let size = std::fs::metadata(&dest)?.len();
    progress(Update::new("download", size, size, format!("Downloaded {file}")));
    Ok(dest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_references() {
        let r = parse_ref("PygmalionAI/PIPPA").unwrap();
        assert_eq!((r.repo.as_str(), r.file, r.revision), ("PygmalionAI/PIPPA", None, None));
        let r = parse_ref("hf:PygmalionAI/PIPPA/pippa_deduped.jsonl").unwrap();
        assert_eq!(r.file.as_deref(), Some("pippa_deduped.jsonl"));
        let r = parse_ref("https://huggingface.co/datasets/PygmalionAI/PIPPA/blob/main/pippa_deduped.jsonl").unwrap();
        assert_eq!((r.file.as_deref(), r.revision.as_deref()), (Some("pippa_deduped.jsonl"), Some("main")));
        let r = parse_ref("https://huggingface.co/datasets/a/b/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet").unwrap();
        assert_eq!(r.revision.as_deref(), Some("refs/convert/parquet"));
        assert_eq!(r.file.as_deref(), Some("default/train/0000.parquet"));
        let r = parse_ref("https://huggingface.co/datasets/a/b").unwrap();
        assert_eq!(r.repo, "a/b");
        assert!(parse_ref("justaname").is_err());
    }

    #[test]
    fn data_file_filter() {
        assert!(is_data_file("pippa_deduped.jsonl"));
        assert!(is_data_file("data/train-00000-of-00002.parquet"));
        assert!(!is_data_file("README.md"));
        assert!(!is_data_file("tokenizer.json"));
        assert!(!is_data_file("PIPPA.py"));
    }
}
