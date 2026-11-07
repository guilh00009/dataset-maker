//! Local HTTP bridge: `POST /api/call/{cmd}` and `GET /api/events` (Server-Sent Events).
//! It exposes the same `Api` the desktop app uses, which is how the UI is previewed in a
//! browser and how a future web version would work.

use anyhow::Result;
use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event as SseEvent, KeepAlive, Sse},
    },
    routing::{get, post},
};
use futures::Stream;
use mixer_api::Api;
use serde_json::{Value, json};
use std::convert::Infallible;
use std::path::PathBuf;
use tokio_stream::{StreamExt, wrappers::BroadcastStream};
use tower_http::services::{ServeDir, ServeFile};

async fn call(State(api): State<Api>, Path(cmd): Path<String>, body: Option<Json<Value>>) -> Response {
    let args = body.map(|b| b.0).unwrap_or(Value::Null);
    match api.call(&cmd, args).await {
        Ok(v) => Json(json!({"ok": true, "result": v})).into_response(),
        Err(e) => (StatusCode::OK, Json(json!({"ok": false, "error": e}))).into_response(),
    }
}

async fn events(State(api): State<Api>) -> Sse<impl Stream<Item = Result<SseEvent, Infallible>>> {
    let stream = BroadcastStream::new(api.subscribe()).filter_map(|e| e.ok()).map(|e| Ok(SseEvent::default().data(serde_json::to_string(&e).unwrap_or_default())));
    Sse::new(stream).keep_alive(KeepAlive::default())
}

pub async fn serve(api: Api, ui: Option<PathBuf>, host: &str, port: u16) -> Result<()> {
    let mut app = Router::new().route("/api/call/{cmd}", post(call)).route("/api/events", get(events)).route("/api/health", get(|| async { "ok" })).with_state(api);
    if let Some(dir) = ui {
        let index = dir.join("index.html");
        app = app.fallback_service(ServeDir::new(&dir).fallback(ServeFile::new(index)));
    }
    let listener = tokio::net::TcpListener::bind(format!("{host}:{port}")).await?;
    println!("Dataset Mixer on http://{}", listener.local_addr()?);
    axum::serve(listener, app).await?;
    Ok(())
}
