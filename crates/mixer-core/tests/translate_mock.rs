//! Integration tests: the real translation pipeline against the mock OpenAI-compatible server.

use mixer_core::export::{SampleTranslator, Translated};
use mixer_core::jobs::Ctl;
use mixer_core::model::{Message, Persona, Role, Sample};
use mixer_core::recipe::{Recipe, Structured};
use mixer_core::translate::{Translator, journal::Journal};
use mixer_mock::{MockConfig, MockState};
use std::sync::Arc;

async fn start(cfg: MockConfig) -> (String, Arc<MockState>) {
    let (addr, state, _h) = mixer_mock::serve(cfg, "127.0.0.1:0".parse().unwrap()).await.unwrap();
    (format!("http://{addr}/v1"), state)
}

fn recipe(url: &str, model: &str) -> Recipe {
    let mut r = Recipe::default();
    r.translate.enabled = true;
    r.translate.base_url = url.into();
    r.translate.model = model.into();
    r.translate.concurrency = 4;
    r.translate.max_retries = 6;
    r.translate.timeout_secs = 10;
    r.translate.retry_backoff_ms = 20;
    r
}

fn sample(i: u64) -> Sample {
    let mut s = Sample {
        id: 0,
        source: 0,
        row: i,
        system: None,
        messages: vec![
            Message::new(Role::User, format!("Hello Rose, I want to buy some flowers today. Number {i}.")),
            Message::new(Role::Assistant, "*smiles* Welcome to the shop! Visit https://example.com/flowers for the garden and `code()` too."),
            Message::new(Role::User, "Thank you, my friend. How much is the red bouquet?"),
            Message::new(Role::Assistant, "Hello {{user}}, it is very good and nice."),
        ],
        text: None,
        persona: Some(Persona {
            char_name: Some("Rose".into()),
            description: "A kind florist who loves the garden and the flowers.".into(),
            definitions: "{{char}}: Hello, welcome to the shop.\n{{user}}: Thank you very much.".into(),
            ..Default::default()
        }),
    };
    s.id = s.compute_id();
    s
}

fn assert_translated(s: &Sample) {
    let all: String = s.messages.iter().map(|m| m.content.clone()).collect::<Vec<_>>().join("\n");
    assert!(all.contains("Bem-vindo"), "{all}");
    assert!(all.contains("https://example.com/flowers"), "URL must survive: {all}");
    assert!(all.contains("`code()`"), "inline code must survive: {all}");
    assert!(all.contains("{{user}}"), "placeholder must survive: {all}");
    assert!(all.contains("Rose"), "name must survive: {all}");
    assert_eq!(s.messages.len(), 4);
    assert_eq!(s.messages[0].role, Role::User);
    let p = s.persona.as_ref().unwrap();
    assert!(p.description.contains("florista"), "{}", p.description);
    assert!(p.definitions.contains("{{char}}:"), "{}", p.definitions);
    assert_eq!(p.char_name.as_deref(), Some("Rose"));
}

#[tokio::test]
async fn translates_with_protected_spans() {
    let (url, mock) = start(MockConfig { base_latency_ms: 1, ..Default::default() }).await;
    let r = recipe(&url, "mock-translator-7b");
    let t = Translator::new(&r, None, Arc::new(Journal::open_memory().unwrap()), Ctl::new()).unwrap();
    match t.translate(sample(1)).await {
        Translated::Ok(s) => assert_translated(&s),
        Translated::Failed(e) => panic!("failed: {e}"),
    }
    // all four messages travelled in a single batched request (+1 each for description/definitions)
    assert!(mock.stats().requests <= 4, "{:?}", mock.stats());
}

#[tokio::test]
async fn survives_rate_limits_errors_garbage_and_missing_items() {
    let (url, mock) = start(MockConfig {
        base_latency_ms: 1,
        rate_429: 0.25,
        retry_after_secs: 0.05,
        rate_500: 0.15,
        rate_malformed: 0.2,
        rate_drop_item: 0.3,
        rate_untranslated: 0.15,
        ..Default::default()
    })
    .await;
    let r = recipe(&url, "mock-translator-7b");
    let t = Translator::new(&r, None, Arc::new(Journal::open_memory().unwrap()), Ctl::new()).unwrap();
    let futs: Vec<_> = (0..20).map(|i| t.translate(sample(i))).collect();
    let results = futures::future::join_all(futs).await;
    let mut ok = 0;
    for res in results {
        match res {
            Translated::Ok(s) => {
                assert_translated(&s);
                ok += 1;
            }
            Translated::Failed(e) => panic!("sample failed: {e}"),
        }
    }
    assert_eq!(ok, 20);
    let st = mock.stats();
    assert!(st.sent_429 > 0 && st.sent_500 > 0, "faults were injected: {st:?}");
    let snap = t.snapshot();
    assert!(snap.retries > 0, "{snap:?}");
}

#[tokio::test]
async fn never_exceeds_the_concurrency_limit() {
    let (url, mock) = start(MockConfig { slots: 16, base_latency_ms: 40, ms_per_token: 0, ..Default::default() }).await;
    let mut r = recipe(&url, "mock-translator-7b");
    r.translate.concurrency = 3;
    let t = Translator::new(&r, None, Arc::new(Journal::open_memory().unwrap()), Ctl::new()).unwrap();
    let futs: Vec<_> = (0..30).map(|i| t.translate(sample(i))).collect();
    futures::future::join_all(futs).await;
    let st = mock.stats();
    assert!(st.max_in_flight <= 3, "limit exceeded: {st:?}");
    assert!(st.max_in_flight >= 2, "should actually run in parallel: {st:?}");
}

#[tokio::test]
async fn api_token_is_sent_and_bad_token_is_a_clear_fatal_error() {
    let (url, mock) = start(MockConfig { api_key: Some("sk-secret".into()), base_latency_ms: 1, ..Default::default() }).await;
    let r = recipe(&url, "mock-translator-7b");
    let j = Arc::new(Journal::open_memory().unwrap());

    let bad = Translator::new(&r, Some("wrong".into()), j.clone(), Ctl::new()).unwrap();
    match bad.translate(sample(1)).await {
        Translated::Failed(e) => assert!(e.to_lowercase().contains("token"), "{e}"),
        _ => panic!("must fail"),
    }
    assert!(bad.stop_reason().is_some());
    assert_eq!(mock.stats().requests, 1, "a rejected token must not be retried");

    let good = Translator::new(&r, Some("sk-secret".into()), j, Ctl::new()).unwrap();
    assert!(matches!(good.translate(sample(1)).await, Translated::Ok(_)));

    // the connection test reports the same problem with a hint
    let rep = mixer_core::translate::client::test_connection(&url, "mock-translator-7b", Some("wrong".into())).await;
    assert!(!rep.ok && rep.hint.unwrap().to_lowercase().contains("token"));
    let rep = mixer_core::translate::client::test_connection(&url, "mock-translator-7b", Some("sk-secret".into())).await;
    assert!(rep.ok, "{rep:?}");
    assert_eq!(rep.models, vec!["mock-translator-7b".to_string()]);
    assert_eq!(rep.slots, Some(4));
    assert!(rep.sample_output.unwrap().contains("Olá"));
}

#[tokio::test]
async fn wrong_model_name_is_reported() {
    let (url, _mock) = start(MockConfig { base_latency_ms: 1, ..Default::default() }).await;
    let r = recipe(&url, "does-not-exist");
    let t = Translator::new(&r, None, Arc::new(Journal::open_memory().unwrap()), Ctl::new()).unwrap();
    match t.translate(sample(1)).await {
        Translated::Failed(e) => assert!(e.contains("404") || e.to_lowercase().contains("model"), "{e}"),
        _ => panic!("must fail"),
    }
}

#[tokio::test]
async fn falls_back_when_server_rejects_response_format() {
    let (url, mock) = start(MockConfig { reject_response_format: true, base_latency_ms: 1, ..Default::default() }).await;
    let mut r = recipe(&url, "mock-translator-7b");
    r.translate.structured = Structured::Auto;
    let t = Translator::new(&r, None, Arc::new(Journal::open_memory().unwrap()), Ctl::new()).unwrap();
    for i in 0..3 {
        assert!(matches!(t.translate(sample(i)).await, Translated::Ok(_)));
    }
    // asked once, learned, never asked again
    assert!(mock.stats().rejected_response_format <= 2, "{:?}", mock.stats());
}

#[tokio::test]
async fn resumes_after_a_crash_without_resending_finished_units() {
    let dir = tempfile::tempdir().unwrap();
    let jpath = dir.path().join("journal.sqlite");
    let (url, mock) = start(MockConfig { base_latency_ms: 1, ..Default::default() }).await;
    let r = recipe(&url, "mock-translator-7b");

    // first "run": translate samples 0..10, then the process "dies"
    {
        let t = Translator::new(&r, None, Arc::new(Journal::open(&jpath).unwrap()), Ctl::new()).unwrap();
        for i in 0..10 {
            assert!(matches!(t.translate(sample(i)).await, Translated::Ok(_)));
        }
    }
    let after_first = mock.stats();

    // second run over samples 0..14 with a fresh process and the same journal file
    let t = Translator::new(&r, None, Arc::new(Journal::open(&jpath).unwrap()), Ctl::new()).unwrap();
    for i in 0..14 {
        assert!(matches!(t.translate(sample(i)).await, Translated::Ok(_)));
    }
    let after_second = mock.stats();
    let snap = t.snapshot();
    assert!(snap.units_cached >= 10 * 4, "finished units come from the journal: {snap:?}");
    // only the 4 new samples (one batched request each; persona text is cached) hit the server
    assert!(after_second.requests - after_first.requests <= 4, "{after_first:?} -> {after_second:?}");
    // and the persona (same for every sample) was translated exactly once across both runs
    let log = mock.request_log();
    let persona_requests = log.iter().filter(|l| l.contains("kind florist")).count();
    assert_eq!(persona_requests, 1, "persona description translated once, got {persona_requests}");
}

#[tokio::test]
async fn long_messages_are_split_and_reassembled() {
    let (url, mock) = start(MockConfig { base_latency_ms: 1, ms_per_token: 0, ..Default::default() }).await;
    let mut r = recipe(&url, "mock-translator-7b");
    r.translate.context_budget = 1024; // payload budget of about 290 tokens
    let paragraphs: Vec<String> = (0..30).map(|i| format!("The story of the garden and the flowers, part {i}, is very good.")).collect();
    let long = paragraphs.join("\n\n");
    let mut s = sample(1);
    s.messages = vec![Message::new(Role::User, "Hello"), Message::new(Role::Assistant, long.clone())];
    s.persona = None;
    let t = Translator::new(&r, None, Arc::new(Journal::open_memory().unwrap()), Ctl::new()).unwrap();
    let out = match t.translate(s).await {
        Translated::Ok(s) => s,
        Translated::Failed(e) => panic!("{e}"),
    };
    let reply = &out.messages[1].content;
    assert_eq!(reply.matches("\n\n").count(), 29, "paragraph structure preserved");
    assert!(reply.contains("A história do jardim") || reply.contains("história"), "{reply}");
    assert!(mock.stats().requests > 2, "split into several requests");
}

#[tokio::test]
async fn cancel_stops_work() {
    let (url, _mock) = start(MockConfig { base_latency_ms: 200, ..Default::default() }).await;
    let r = recipe(&url, "mock-translator-7b");
    let ctl = Ctl::new();
    let t = Translator::new(&r, None, Arc::new(Journal::open_memory().unwrap()), ctl.clone()).unwrap();
    ctl.cancel();
    match t.translate(sample(1)).await {
        Translated::Failed(e) => assert_eq!(e, "cancelled"),
        _ => panic!("must be cancelled"),
    }
}
