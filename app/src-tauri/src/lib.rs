//! Desktop shell: the whole application lives in `mixer-api`; this only exposes it to the UI.
//!
//! * `call(cmd, args)`: the single command the UI uses (same names as the HTTP bridge)
//! * `mixer-event`: job progress and state-change events pushed to the UI

use mixer_api::Api;
use serde_json::Value;
use tauri::{Emitter, Manager, State};
use tokio::sync::broadcast::error::RecvError;

#[tauri::command]
async fn call(api: State<'_, Api>, cmd: String, args: Option<Value>) -> Result<Value, String> {
    api.call(&cmd, args.unwrap_or(Value::Null)).await
}

pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            let api = Api::new(None, None)?;
            let handle = app.handle().clone();
            let worker = api.clone();
            tauri::async_runtime::spawn(async move {
                let mut rx = worker.subscribe();
                worker.init().await;
                loop {
                    match rx.recv().await {
                        Ok(event) => {
                            let _ = handle.emit("mixer-event", event);
                        }
                        Err(RecvError::Lagged(_)) => continue,
                        Err(RecvError::Closed) => break,
                    }
                }
            });
            app.manage(api);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![call])
        .run(tauri::generate_context!())
        .expect("error while running Dataset Mixer");
}
