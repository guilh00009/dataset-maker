//! End to end: files in, shuffled mix out, in several formats, deterministic, with translation
//! and a mocked Hugging Face Hub.

use mixer_core::jobs::{Ctl, no_progress};
use mixer_core::pipeline::{RunOptions, run_recipe};
use mixer_core::recipe::*;
use serde_json::{Value, json};
use std::path::Path;

fn write_jsonl(path: &Path, rows: impl Iterator<Item = Value>) {
    let s: String = rows.map(|r| r.to_string() + "\n").collect();
    std::fs::write(path, s).unwrap();
}

fn pippa_rows(n: usize) -> impl Iterator<Item = Value> {
    (0..n).map(|i| {
        let bot = ["Rose", "Bob", "Mira"][i % 3];
        json!({
            "submission_timestamp": 1690000000000i64 + i as i64, "categories": null, "bot_id": format!("b{}", i % 3),
            "bot_name": bot, "bot_greeting": format!("Hello, I am {bot}!"),
            "bot_definitions": "{{char}}: Hello there.\nEND_OF_DIALOG\n{{user}}: Hi!", "bot_description": "A friendly person who loves the garden.",
            "conversation": [
                {"message": format!("Hello, I am {bot}!"), "is_human": false},
                {"message": format!("hi {{{{char}}}}, question {i}"), "is_human": true},
                {"message": "Welcome to the shop, {{user}}!", "is_human": false},
            ]
        })
    })
}

fn sharegpt_rows(n: usize) -> impl Iterator<Item = Value> {
    (0..n).map(|i| json!({"id": i, "conversations": [{"from": "human", "value": format!("what is {i}?")}, {"from": "gpt", "value": format!("it is number {i}")}]}))
}

fn base_recipe(dir: &Path) -> Recipe {
    let a = dir.join("pippa.jsonl");
    let b = dir.join("sharegpt.jsonl");
    write_jsonl(&a, pippa_rows(300));
    write_jsonl(&b, sharegpt_rows(100));
    let mut r = Recipe::default();
    r.seed = 7;
    r.sources = vec![
        SourceEntry { id: "pippa".into(), location: SourceLocation::File { path: a.to_string_lossy().into() }, weight: 1.0, count: None, mapping: None, enabled: true },
        SourceEntry { id: "sharegpt".into(), location: SourceLocation::File { path: b.to_string_lossy().into() }, weight: 1.0, count: None, mapping: None, enabled: true },
    ];
    r
}

fn opts(dir: &Path, out: &str) -> RunOptions {
    RunOptions { cache_dir: dir.join("cache"), out_dir: Some(dir.join(out)), ..Default::default() }
}

fn read_lines(p: &Path) -> Vec<Value> {
    std::fs::read_to_string(p).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect()
}

#[tokio::test]
async fn mixes_two_formats_with_weights_and_names_substituted() {
    let d = tempfile::tempdir().unwrap();
    let mut r = base_recipe(d.path());
    r.export.format = Format::Chatml;
    let m = run_recipe(&r, &opts(d.path(), "out"), &Ctl::new(), &no_progress()).await.unwrap();
    // 50/50 limited by the smaller source: 100 + 100
    assert_eq!(m.rows_written, 200);
    assert_eq!(m.per_source[0].taken, 100);
    assert_eq!(m.per_source[1].taken, 100);
    let rows = read_lines(&d.path().join("out/train.jsonl"));
    assert_eq!(rows.len(), 200);
    let texts: Vec<&str> = rows.iter().map(|r| r["text"].as_str().unwrap()).collect();
    assert!(texts.iter().all(|t| t.starts_with("<|im_start|>")));
    // roleplay rows: persona became a system prompt, placeholders became real names
    let rp: Vec<&&str> = texts.iter().filter(|t| t.contains("You are")).collect();
    assert!(rp.len() >= 50);
    for t in &rp {
        assert!(!t.contains("{{char}}") && !t.contains("{{user}}"), "{t}");
        assert!(!t.contains("END_OF_DIALOG"));
        assert!(t.contains("User: Hi!"), "fallback user name used: {t}");
    }
    // shuffled: both sources interleaved, not grouped
    let first_half_rp = texts[..100].iter().filter(|t| t.contains("You are")).count();
    assert!(first_half_rp > 20 && first_half_rp < 80, "{first_half_rp}");
    // manifest on disk
    let man: Value = serde_json::from_slice(&std::fs::read(d.path().join("out/manifest.json")).unwrap()).unwrap();
    assert_eq!(man["rows_written"], 200);
    assert_eq!(man["files"][0]["name"], "train.jsonl");
}

#[tokio::test]
async fn same_seed_same_bytes_different_seed_differs() {
    let d = tempfile::tempdir().unwrap();
    let mut r = base_recipe(d.path());
    r.mix.total = Some(150);
    r.export.format = Format::Messages;
    run_recipe(&r, &opts(d.path(), "a"), &Ctl::new(), &no_progress()).await.unwrap();
    run_recipe(&r, &opts(d.path(), "b"), &Ctl::new(), &no_progress()).await.unwrap();
    let a = std::fs::read(d.path().join("a/train.jsonl")).unwrap();
    let b = std::fs::read(d.path().join("b/train.jsonl")).unwrap();
    assert_eq!(a, b, "byte-identical output for the same recipe and seed");
    r.seed = 8;
    run_recipe(&r, &opts(d.path(), "c"), &Ctl::new(), &no_progress()).await.unwrap();
    assert_ne!(a, std::fs::read(d.path().join("c/train.jsonl")).unwrap());
}

#[tokio::test]
async fn counts_split_filters_dedup_and_shards() {
    let d = tempfile::tempdir().unwrap();
    let mut r = base_recipe(d.path());
    // duplicate rows in the second source
    let dup = d.path().join("dups.jsonl");
    write_jsonl(&dup, sharegpt_rows(10).chain(sharegpt_rows(10)));
    r.sources.push(SourceEntry { id: "dups".into(), location: SourceLocation::File { path: dup.to_string_lossy().into() }, weight: 1.0, count: Some(1000), mapping: None, enabled: true });
    r.sources[0].count = Some(30);
    r.sources[1].count = Some(40);
    r.filters.dedup = DedupMode::Exact;
    r.split.validation = 0.1;
    r.export.format = Format::Sharegpt;
    r.export.shard_size = Some(25);
    let m = run_recipe(&r, &opts(d.path(), "out"), &Ctl::new(), &no_progress()).await.unwrap();
    // dups.jsonl has 10 unique rows after dedup; the first 10 sharegpt rows collide with them
    // (same content) and are removed from whichever source comes later in the list.
    let total: u64 = m.per_source.iter().map(|s| s.taken).sum();
    assert_eq!(m.rows_written, total);
    assert!(m.per_source[2].stats.duplicates >= 10, "{:?}", m.per_source[2].stats);
    let train: u64 = m.files.iter().filter(|f| f.split == "train").map(|f| f.rows).sum();
    let val: u64 = m.files.iter().filter(|f| f.split == "validation").map(|f| f.rows).sum();
    assert_eq!(train + val, total);
    assert_eq!(val, (total as f64 * 0.1).floor() as u64);
    assert!(m.files.iter().any(|f| f.name.starts_with("train-00000-of-0000")));
    // no row is in both splits
    let mut seen = std::collections::HashSet::new();
    for f in &m.files {
        for row in read_lines(&d.path().join("out").join(&f.name)) {
            assert!(seen.insert(row.to_string()), "duplicate row across splits/shards");
        }
    }
}

#[tokio::test]
async fn all_containers_work() {
    let d = tempfile::tempdir().unwrap();
    for (c, name) in [(Container::Json, "train.json"), (Container::Parquet, "train.parquet")] {
        let mut r = base_recipe(d.path());
        r.mix.total = Some(40);
        r.export.container = c;
        r.export.format = Format::Messages;
        let out = format!("out-{name}");
        let m = run_recipe(&r, &opts(d.path(), &out), &Ctl::new(), &no_progress()).await.unwrap();
        assert_eq!(m.files[0].name, name);
        assert_eq!(m.files[0].rows, 40);
        // the produced file can be read back by our own loader (round trip)
        let back = mixer_core::source::prepare(&d.path().join(&out).join(name), &d.path().join("cache2"), &Ctl::new(), &no_progress()).unwrap();
        assert_eq!(back.len(), 40);
        assert!(back.row_value(0).unwrap()["messages"].is_array());
    }
}

#[tokio::test]
async fn translation_end_to_end_with_mock_server() {
    let d = tempfile::tempdir().unwrap();
    let (addr, mock, _h) = mixer_mock::serve(mixer_mock::MockConfig { base_latency_ms: 1, ms_per_token: 0, rate_429: 0.1, retry_after_secs: 0.01, ..Default::default() }, "127.0.0.1:0".parse().unwrap()).await.unwrap();
    let mut r = base_recipe(d.path());
    r.sources.truncate(1);
    r.mix.total = Some(30);
    r.export.format = Format::Messages;
    r.persona.fallback_user_name = "Guilherme".into();
    r.translate.enabled = true;
    r.translate.base_url = format!("http://{addr}/v1");
    r.translate.model = "mock-translator-7b".into();
    r.translate.retry_backoff_ms = 10;
    r.translate.concurrency = 4;
    let m = run_recipe(&r, &opts(d.path(), "out"), &Ctl::new(), &no_progress()).await.unwrap();
    assert_eq!(m.rows_written, 30);
    assert_eq!(m.translation_failed, 0);
    let rows = read_lines(&d.path().join("out/train.jsonl"));
    for row in &rows {
        let msgs = row["messages"].as_array().unwrap();
        let all: String = msgs.iter().map(|m| m["content"].as_str().unwrap()).collect::<Vec<_>>().join("\n");
        assert!(all.contains("Bem-vindo"), "{all}");
        assert!(all.contains("Guilherme"), "user name filled in after translation: {all}");
        assert!(!all.contains("{{"), "no placeholders left: {all}");
        let name = ["Rose", "Bob", "Mira"].iter().find(|n| msgs[0]["content"].as_str().unwrap().contains(*n));
        assert!(name.is_some(), "character name survives untranslated: {all}");
        assert_eq!(msgs[0]["role"], "system");
    }
    // the persona text is shared by all rows of one character, so it was sent at most 3 times
    let persona_reqs = mock.request_log().iter().filter(|l| l.contains("friendly person")).count();
    assert!(persona_reqs <= 3, "{persona_reqs}");
    // running it again hits only the journal
    let before = mock.stats().requests;
    run_recipe(&r, &opts(d.path(), "out2"), &Ctl::new(), &no_progress()).await.unwrap();
    assert_eq!(mock.stats().requests, before, "second run is served entirely from the journal");
    assert_eq!(std::fs::read(d.path().join("out/train.jsonl")).unwrap(), std::fs::read(d.path().join("out2/train.jsonl")).unwrap());
}

#[tokio::test]
async fn bilingual_and_keep_original_modes() {
    let d = tempfile::tempdir().unwrap();
    let (addr, _mock, _h) = mixer_mock::serve(mixer_mock::MockConfig { base_latency_ms: 1, ms_per_token: 0, ..Default::default() }, "127.0.0.1:0".parse().unwrap()).await.unwrap();
    let mut r = base_recipe(d.path());
    r.sources.truncate(1);
    r.mix.total = Some(5);
    r.export.format = Format::Messages;
    r.translate.enabled = true;
    r.translate.base_url = format!("http://{addr}/v1");
    r.translate.model = "mock-translator-7b".into();
    r.translate.output_mode = OutputMode::Bilingual;
    let m = run_recipe(&r, &opts(d.path(), "out"), &Ctl::new(), &no_progress()).await.unwrap();
    assert_eq!(m.rows_written, 5);
    let rows = read_lines(&d.path().join("out/train.jsonl"));
    assert!(rows[0]["original"]["messages"].is_array());
    assert!(rows[0]["original"]["messages"][3]["content"].as_str().unwrap().contains("Welcome to the shop"));

    // server that always fails + keep_original policy
    r.translate.base_url = "http://127.0.0.1:1/v1".into(); // nothing listens here
    r.translate.max_retries = 0;
    r.translate.on_fail = OnFail::KeepOriginal;
    r.translate.output_mode = OutputMode::Replace;
    r.translate.model = "other-model".into(); // different cache key
    let m = run_recipe(&r, &opts(d.path(), "out3"), &Ctl::new(), &no_progress()).await.unwrap();
    assert_eq!(m.rows_written, 5);
    assert_eq!(m.translation_failed, 5);
    r.translate.on_fail = OnFail::Drop;
    let m = run_recipe(&r, &opts(d.path(), "out4"), &Ctl::new(), &no_progress()).await.unwrap();
    assert_eq!(m.rows_written, 0);
}

#[tokio::test]
async fn huggingface_download_resumes_and_detects_pippa_layout() {
    let d = tempfile::tempdir().unwrap();
    let hub = d.path().join("hub/PygmalionAI__PIPPA");
    std::fs::create_dir_all(&hub).unwrap();
    write_jsonl(&hub.join("pippa_deduped.jsonl"), pippa_rows(200));
    std::fs::write(hub.join("README.md"), "# PIPPA").unwrap();
    let (addr, _s, _h) = mixer_mock::serve_full(Default::default(), "127.0.0.1:0".parse().unwrap(), Some(d.path().join("hub"))).await.unwrap();
    // SAFETY: tests in this file that read HF_ENDPOINT run in this single test only.
    unsafe { std::env::set_var("HF_ENDPOINT", format!("http://{addr}")) };

    let files = mixer_core::hf::list_files("PygmalionAI/PIPPA", None, &None).await.unwrap();
    let data: Vec<_> = files.iter().filter(|f| mixer_core::hf::is_data_file(&f.path)).collect();
    assert_eq!(data.len(), 1);
    assert_eq!(data[0].path, "pippa_deduped.jsonl");

    // pretend an earlier download was interrupted halfway
    let full = std::fs::read(hub.join("pippa_deduped.jsonl")).unwrap();
    let cache = d.path().join("cache");
    let part_dir = cache.join("hf/PygmalionAI__PIPPA/main");
    std::fs::create_dir_all(&part_dir).unwrap();
    std::fs::write(part_dir.join("pippa_deduped.jsonl.part"), &full[..full.len() / 2]).unwrap();
    let got = mixer_core::hf::download("PygmalionAI/PIPPA", "pippa_deduped.jsonl", None, &None, &cache, &Ctl::new(), &no_progress()).await.unwrap();
    assert_eq!(std::fs::read(&got).unwrap(), full, "resumed download is byte-identical");

    let mut r = Recipe::default();
    r.sources = vec![SourceEntry {
        id: "PIPPA".into(),
        location: SourceLocation::Hf { repo: "PygmalionAI/PIPPA".into(), file: "pippa_deduped.jsonl".into(), revision: None },
        weight: 1.0,
        count: Some(50),
        mapping: None,
        enabled: true,
    }];
    let m = run_recipe(&r, &RunOptions { cache_dir: cache, out_dir: Some(d.path().join("out")), ..Default::default() }, &Ctl::new(), &no_progress()).await.unwrap();
    assert_eq!(m.rows_written, 50);
    assert!(read_lines(&d.path().join("out/train.jsonl"))[0]["text"].as_str().unwrap().contains("You are"));

    let missing = mixer_core::hf::download("PygmalionAI/PIPPA", "nope.jsonl", None, &None, &d.path().join("c2"), &Ctl::new(), &no_progress()).await;
    assert!(missing.unwrap_err().to_string().contains("not found"));
}
