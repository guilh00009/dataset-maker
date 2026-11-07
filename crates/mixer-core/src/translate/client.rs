//! Minimal OpenAI-compatible client (llama.cpp `llama-server`, vLLM, Ollama, LM Studio,
//! OpenRouter, OpenAI, …). Only three endpoints are used: `GET /models`,
//! `POST /chat/completions` and (llama.cpp only) `GET /props`.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::time::{Duration, Instant};

/// Accept `host:port`, `http://host:port`, `…/v1`, `…/v1/` and `…/v1/chat/completions`.
pub fn normalize_base(url: &str) -> String {
    let mut u = url.trim().trim_end_matches('/').to_string();
    if u.is_empty() {
        return u;
    }
    if !u.contains("://") {
        u = format!("http://{u}");
    }
    for suffix in ["/chat/completions", "/completions", "/models"] {
        if let Some(s) = u.strip_suffix(suffix) {
            u = s.to_string();
        }
    }
    // A bare host (no path at all) gets `/v1`.
    let after_scheme = u.split("://").nth(1).unwrap_or("");
    if !after_scheme.contains('/') {
        u.push_str("/v1");
    }
    u
}

/// Server root (without `/v1`) for non-standard endpoints such as llama.cpp's `/props`.
pub fn server_root(base: &str) -> String {
    base.trim_end_matches('/').trim_end_matches("/v1").to_string()
}

#[derive(Debug, Clone)]
pub enum LlmError {
    /// Connection refused, DNS failure, reset…
    Network(String),
    Timeout,
    Http { status: u16, body: String, retry_after: Option<Duration> },
    /// 200 OK but the payload is not what we expect.
    BadResponse(String),
}

impl std::fmt::Display for LlmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LlmError::Network(e) => write!(f, "network error: {e}"),
            LlmError::Timeout => write!(f, "request timed out"),
            LlmError::Http { status, body, .. } => {
                let b: String = body.chars().take(300).collect();
                write!(f, "HTTP {status}: {b}")
            }
            LlmError::BadResponse(e) => write!(f, "unexpected response: {e}"),
        }
    }
}

impl std::error::Error for LlmError {}

impl LlmError {
    pub fn retriable(&self) -> bool {
        match self {
            LlmError::Network(_) | LlmError::Timeout => true,
            LlmError::Http { status, .. } => matches!(status, 408 | 409 | 425 | 429 | 500 | 502 | 503 | 504 | 529),
            LlmError::BadResponse(_) => false,
        }
    }
    /// Errors that mean the configuration is wrong; retrying cannot help.
    pub fn fatal(&self) -> bool {
        matches!(self, LlmError::Http { status, .. } if matches!(status, 401 | 403 | 404))
    }
    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            LlmError::Http { retry_after, .. } => *retry_after,
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub messages: Vec<(String, String)>,
    pub temperature: f32,
    pub max_tokens: u32,
    /// JSON schema to force structured output (OpenAI `response_format: json_schema`).
    pub json_schema: Option<Value>,
}

#[derive(Debug, Clone, Default)]
pub struct ChatResponse {
    pub content: String,
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub finish_reason: Option<String>,
}

#[derive(Clone)]
pub struct LlmClient {
    http: reqwest::Client,
    pub base: String,
    pub model: String,
    token: Option<String>,
}

impl LlmClient {
    pub fn new(base_url: &str, model: &str, token: Option<String>, timeout: Duration) -> anyhow::Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(timeout)
            .connect_timeout(Duration::from_secs(15))
            .pool_idle_timeout(Duration::from_secs(30))
            .user_agent("dataset-mixer")
            .build()?;
        Ok(Self { http, base: normalize_base(base_url), model: model.trim().to_string(), token: token.filter(|t| !t.trim().is_empty()) })
    }

    fn req(&self, method: reqwest::Method, url: &str) -> reqwest::RequestBuilder {
        let mut r = self.http.request(method, url);
        if let Some(t) = &self.token {
            r = r.bearer_auth(t.trim());
        }
        r
    }

    async fn send(&self, rb: reqwest::RequestBuilder) -> Result<Value, LlmError> {
        let resp = rb.send().await.map_err(|e| {
            if e.is_timeout() { LlmError::Timeout } else { LlmError::Network(format_err(&e)) }
        })?;
        let status = resp.status();
        let retry_after = resp
            .headers()
            .get(reqwest::header::RETRY_AFTER)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.trim().parse::<f64>().ok())
            .map(|s| Duration::from_secs_f64(s.clamp(0.0, 600.0)));
        let text = resp.text().await.map_err(|e| if e.is_timeout() { LlmError::Timeout } else { LlmError::Network(format_err(&e)) })?;
        if !status.is_success() {
            return Err(LlmError::Http { status: status.as_u16(), body: text, retry_after });
        }
        serde_json::from_str(&text).map_err(|e| LlmError::BadResponse(format!("invalid JSON ({e}): {}", text.chars().take(200).collect::<String>())))
    }

    pub async fn list_models(&self) -> Result<Vec<String>, LlmError> {
        let v = self.send(self.req(reqwest::Method::GET, &format!("{}/models", self.base))).await?;
        let arr = v.get("data").and_then(|d| d.as_array()).or_else(|| v.get("models").and_then(|d| d.as_array()));
        let Some(arr) = arr else { return Err(LlmError::BadResponse("no `data` array in /models".into())) };
        Ok(arr
            .iter()
            .filter_map(|m| m.get("id").or_else(|| m.get("name")).or_else(|| m.get("model")).and_then(|x| x.as_str()))
            .map(|s| s.to_string())
            .collect())
    }

    /// llama.cpp `/props`; `None` on any other server.
    pub async fn props(&self) -> Option<Value> {
        let url = format!("{}/props", server_root(&self.base));
        self.send(self.req(reqwest::Method::GET, &url)).await.ok()
    }

    pub async fn chat(&self, r: &ChatRequest) -> Result<ChatResponse, LlmError> {
        let mut body = json!({
            "model": self.model,
            "messages": r.messages.iter().map(|(role, content)| json!({"role": role, "content": content})).collect::<Vec<_>>(),
            "temperature": r.temperature,
            "max_tokens": r.max_tokens,
            "stream": false,
        });
        if let Some(schema) = &r.json_schema {
            body["response_format"] = json!({
                "type": "json_schema",
                "json_schema": {"name": "translations", "strict": true, "schema": schema},
            });
        }
        let v = self.send(self.req(reqwest::Method::POST, &format!("{}/chat/completions", self.base)).json(&body)).await?;
        let choice = v.get("choices").and_then(|c| c.get(0)).ok_or_else(|| LlmError::BadResponse("no choices".into()))?;
        let content = choice
            .get("message")
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .or_else(|| choice.get("text").and_then(|c| c.as_str()))
            .ok_or_else(|| LlmError::BadResponse("no message content".into()))?
            .to_string();
        let usage = v.get("usage");
        Ok(ChatResponse {
            content,
            prompt_tokens: usage.and_then(|u| u.get("prompt_tokens")).and_then(|x| x.as_u64()).map(|x| x as u32),
            completion_tokens: usage.and_then(|u| u.get("completion_tokens")).and_then(|x| x.as_u64()).map(|x| x as u32),
            finish_reason: choice.get("finish_reason").and_then(|x| x.as_str()).map(|s| s.to_string()),
        })
    }
}

fn format_err(e: &reqwest::Error) -> String {
    use std::error::Error;
    let mut s = e.to_string();
    let mut src = e.source();
    while let Some(x) = src {
        s.push_str(": ");
        s.push_str(&x.to_string());
        src = x.source();
    }
    s
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TestReport {
    pub ok: bool,
    pub base_url: String,
    pub models: Vec<String>,
    pub model_found: Option<bool>,
    pub models_error: Option<String>,
    pub latency_ms: Option<u64>,
    pub tokens_per_sec: Option<f64>,
    pub sample_output: Option<String>,
    pub slots: Option<u32>,
    pub error: Option<String>,
    /// A human readable hint for the most common mistakes.
    pub hint: Option<String>,
}

/// "Test connection": list models, then run a tiny chat completion.
pub async fn test_connection(base_url: &str, model: &str, token: Option<String>) -> TestReport {
    let mut rep = TestReport { base_url: normalize_base(base_url), ..Default::default() };
    let client = match LlmClient::new(base_url, model, token, Duration::from_secs(90)) {
        Ok(c) => c,
        Err(e) => {
            rep.error = Some(e.to_string());
            return rep;
        }
    };
    match client.list_models().await {
        Ok(m) => {
            rep.model_found = if model.trim().is_empty() { None } else { Some(m.iter().any(|x| x == model.trim())) };
            rep.models = m;
        }
        Err(e) => {
            if e.fatal() {
                rep.error = Some(e.to_string());
                rep.hint = Some(match &e {
                    LlmError::Http { status: 401 | 403, .. } => "The server rejected the API token. Check the token (llama-server --api-key).".into(),
                    _ => "The URL answered but `/models` was not found. Make sure the base URL ends with /v1.".into(),
                });
                return rep;
            }
            rep.models_error = Some(e.to_string());
            if matches!(e, LlmError::Network(_) | LlmError::Timeout) {
                rep.error = Some(e.to_string());
                rep.hint = Some("Cannot reach the server. Is llama-server running, and is the host/port right?".into());
                return rep;
            }
        }
    }
    if let Some(p) = client.props().await {
        rep.slots = p.get("total_slots").and_then(|s| s.as_u64()).map(|s| s as u32);
    }
    let started = Instant::now();
    let req = ChatRequest {
        messages: vec![
            ("system".into(), "You are a translator. Reply with the translation only.".into()),
            ("user".into(), "Translate to Portuguese: Hello, how are you today?".into()),
        ],
        temperature: 0.0,
        max_tokens: 64,
        json_schema: None,
    };
    match client.chat(&req).await {
        Ok(r) => {
            let secs = started.elapsed().as_secs_f64();
            rep.latency_ms = Some((secs * 1000.0) as u64);
            let toks = r.completion_tokens.unwrap_or_else(|| (r.content.chars().count() as f64 / 3.6).ceil() as u32);
            rep.tokens_per_sec = if secs > 0.0 && toks > 0 { Some(toks as f64 / secs) } else { None };
            rep.sample_output = Some(r.content.trim().to_string());
            rep.ok = true;
            if rep.model_found == Some(false) {
                rep.hint = Some("The server answered, but the model name is not in its model list; it may still work if the server ignores the name.".into());
            }
        }
        Err(e) => {
            rep.hint = Some(match &e {
                LlmError::Http { status: 401 | 403, .. } => "The server rejected the API token.".to_string(),
                LlmError::Http { status: 404, .. } => "Not found: check the model name and that the base URL ends with /v1.".to_string(),
                LlmError::Http { status: 400, body, .. } if body.to_lowercase().contains("model") => "The server does not know this model name.".to_string(),
                LlmError::Network(_) | LlmError::Timeout => "Cannot reach the server.".to_string(),
                _ => String::new(),
            })
            .filter(|h| !h.is_empty());
            rep.error = Some(e.to_string());
        }
    }
    rep
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_url_normalization() {
        assert_eq!(normalize_base("127.0.0.1:8080"), "http://127.0.0.1:8080/v1");
        assert_eq!(normalize_base("http://localhost:8080/"), "http://localhost:8080/v1");
        assert_eq!(normalize_base("http://localhost:8080/v1/"), "http://localhost:8080/v1");
        assert_eq!(normalize_base("https://api.example.com/v1/chat/completions"), "https://api.example.com/v1");
        assert_eq!(normalize_base("https://openrouter.ai/api/v1"), "https://openrouter.ai/api/v1");
        assert_eq!(server_root("http://h:1/v1"), "http://h:1");
    }
}
