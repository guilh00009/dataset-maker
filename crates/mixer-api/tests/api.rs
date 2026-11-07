//! Drives the API exactly like the UI does.

use mixer_api::{Api, wait_job};
use serde_json::{Value, json};

fn rows(n: usize) -> String {
    (0..n)
        .map(|i| {
            json!({
                "bot_name": "Rose", "bot_greeting": "Hello, welcome to the shop!", "bot_description": "A kind florist who loves the garden.",
                "bot_definitions": "{{char}}: Hello.\n{{user}}: Hi.",
                "conversation": [{"message": "Hello, welcome to the shop!", "is_human": false}, {"message": format!("hello {{{{char}}}}, number {i}"), "is_human": true}, {"message": "Thank you, my friend.", "is_human": false}]
            })
            .to_string()
                + "\n"
        })
        .collect()
}

async fn job(api: &Api, r: Value) -> Value {
    wait_job(api, r["job_id"].as_str().unwrap()).await.unwrap()
}

#[tokio::test]
async fn full_ui_flow() {
    let d = tempfile::tempdir().unwrap();
    let api = Api::new(Some(d.path().join("cache")), Some(d.path().join("config"))).unwrap();
    api.init().await;
    let f = d.path().join("pippa-like.jsonl");
    std::fs::write(&f, rows(60)).unwrap();

    let (addr, mock, _h) = mixer_mock::serve(mixer_mock::MockConfig { base_latency_ms: 1, ms_per_token: 0, api_key: Some("sk-test".into()), ..Default::default() }, "127.0.0.1:0".parse().unwrap()).await.unwrap();

    // browse files like the picker does
    let ls = api.call("fs_list", json!({"path": d.path()})).await.unwrap();
    assert!(ls["entries"].as_array().unwrap().iter().any(|e| e["name"] == "pippa-like.jsonl" && e["is_data"] == true));

    api.call("add_source", json!({"kind": "file", "path": f})).await.unwrap();
    // wait until it is ready
    let mut state = Value::Null;
    for _ in 0..100 {
        state = api.call("get_state", json!({})).await.unwrap();
        if state["sources"][0]["status"] == "ready" {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    assert_eq!(state["sources"][0]["status"], "ready", "{state}");
    assert_eq!(state["sources"][0]["rows"], 60);
    let best = &state["sources"][0]["detection"]["best"];
    assert!(best["confidence"].as_f64().unwrap() > 0.9);
    assert_eq!(best["char_name"], "bot_name");
    assert!(best["summary"].as_str().unwrap().contains("persona"));

    // options
    api.call("set_options", json!({"export": {"format": "chatml", "container": "jsonl", "out_dir": d.path().join("out")}, "split": {"validation": 0.1}, "persona": {"fallback_user_name": "Guilherme"}})).await.unwrap();
    let r = api.call("compute_mix", json!({})).await.unwrap();
    let plan = job(&api, r).await;
    assert_eq!(plan["total"], 60);
    assert_eq!(plan["validation"], 6);

    // changing a mix setting marks the plan as stale
    api.call("update_source", json!({"id": state["sources"][0]["id"], "count": 20})).await.unwrap();
    assert_eq!(api.call("get_state", json!({})).await.unwrap()["plan"]["stale"], true);
    let r = api.call("compute_mix", json!({})).await.unwrap();
    assert_eq!(job(&api, r).await["total"], 20);

    // preview: original, formatted, no translation yet
    let pv = api.call("preview", json!({"offset": 0, "count": 3})).await.unwrap();
    let it = &pv["items"][0];
    assert!(it["formatted"][0]["text"].as_str().unwrap().contains("Guilherme"));
    assert!(it["translated"].is_null());
    assert_eq!(pv["total"], 20);

    // connection settings: wrong token first, then the right one
    let base = format!("http://{addr}/v1");
    let bad = api.call("llm_test", json!({"base_url": base, "model": "mock-translator-7b", "token": "nope"})).await.unwrap();
    assert_eq!(bad["ok"], false);
    api.call("set_settings", json!({"llm_token": "sk-test"})).await.unwrap();
    let ok = api.call("llm_test", json!({"base_url": base, "model": "mock-translator-7b"})).await.unwrap();
    assert_eq!(ok["ok"], true, "{ok}");
    assert!(api.call("get_state", json!({})).await.unwrap()["settings"]["has_llm_token"].as_bool().unwrap());

    api.call("set_options", json!({"translate": {"enabled": true, "base_url": base, "model": "mock-translator-7b", "target_language": "pt-BR", "concurrency": 3}})).await.unwrap();
    let r = api.call("compute_mix", json!({})).await.unwrap();
    job(&api, r).await;
    let est = api.call("estimate_translation", json!({})).await.unwrap();
    assert_eq!(est["samples"], 20);
    assert!(est["input_tokens"].as_u64().unwrap() > 0 && est["requests"].as_u64().unwrap() > 0);
    assert!(est["eta_secs"].is_number(), "ETA uses the measured speed: {est}");

    // trial of 5, then everything; the trial's work is reused
    let r = api.call("start_translation", json!({"only_first": 5})).await.unwrap();
    let trial = job(&api, r).await;
    assert_eq!(trial["done"], 5);
    assert_eq!(trial["failed"], 0);
    let pv = api.call("preview", json!({"offset": 0, "count": 2})).await.unwrap();
    let tr = &pv["items"][0]["translated"];
    assert!(tr["formatted"][0]["text"].as_str().unwrap().to_lowercase().contains("bem-vindo"), "{tr}");
    let r = api.call("start_translation", json!({})).await.unwrap();
    let full = job(&api, r).await;
    assert_eq!(full["done"], 20);
    assert!(full["stats"]["units_cached"].as_u64().unwrap() > 0);

    // export (reads the cache), manifest and files on disk
    let r = api.call("export", json!({})).await.unwrap();
    let m = job(&api, r).await;
    assert_eq!(m["rows_written"], 20);
    assert_eq!(m["translation_failed"], 0);
    let train = std::fs::read_to_string(d.path().join("out/train.jsonl")).unwrap();
    assert_eq!(train.lines().count(), 18);
    assert!(train.to_lowercase().contains("bem-vindo") && train.contains("Guilherme") && !train.contains("{{"));
    assert!(d.path().join("out/validation.jsonl").exists());
    assert!(d.path().join("out/manifest.json").exists());
    let before = mock.stats().requests;
    let r = api.call("export", json!({"out_dir": d.path().join("out2")})).await.unwrap();
    job(&api, r).await;
    assert_eq!(mock.stats().requests, before, "export after translation does not call the server again");

    // recipe round trip
    let rp = d.path().join("recipe.toml");
    api.call("save_recipe", json!({"path": rp})).await.unwrap();
    let toml = std::fs::read_to_string(&rp).unwrap();
    assert!(toml.contains("[[source]]") && !toml.contains("sk-test"), "the token must never be written to a recipe");
    api.call("new_recipe", json!({})).await.unwrap();
    assert_eq!(api.call("get_state", json!({})).await.unwrap()["sources"].as_array().unwrap().len(), 0);
    api.call("load_recipe", json!({"path": rp})).await.unwrap();
    for _ in 0..100 {
        state = api.call("get_state", json!({})).await.unwrap();
        if state["sources"][0]["status"] == "ready" {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    assert_eq!(state["sources"][0]["status"], "ready");
    assert_eq!(state["recipe"]["source"][0]["count"].as_u64().or(state["recipe"]["sources"][0]["count"].as_u64()), Some(20));

    // errors are readable
    let e = api.call("add_source", json!({"kind": "file", "path": "/nope/missing.jsonl"})).await;
    assert!(e.is_ok());
    for _ in 0..50 {
        state = api.call("get_state", json!({})).await.unwrap();
        if state["sources"].as_array().unwrap().iter().any(|s| s["status"] == "error") {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    assert!(state["sources"].as_array().unwrap().iter().any(|s| s["status"] == "error" && s["error"].as_str().unwrap().contains("does not exist")), "{state}");
    assert!(api.call("nonsense", json!({})).await.unwrap_err().contains("unknown command"));
}

#[tokio::test]
async fn tokenizer_config_and_pause_cancel() {
    let d = tempfile::tempdir().unwrap();
    let api = Api::new(Some(d.path().join("cache")), Some(d.path().join("config"))).unwrap();
    let t = api.call("tokenizer_config", json!({"text": r#"{"chat_template":"{{ messages|length }}","bos_token":"<s>","eos_token":"</s>"}"#})).await.unwrap();
    assert_eq!(t["bos_token"], "<s>");
    assert!(api.call("tokenizer_config", json!({"text": "{}"})).await.is_err());
    assert!(api.call("pause_job", json!({"job_id": "job-999"})).await.is_err());
}
