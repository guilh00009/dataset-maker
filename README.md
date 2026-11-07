# Dataset Mixer

Point it at the datasets you want, mix and shuffle them, translate them with your own LLM, and export in a
training format (ChatML, Llama 3, Mistral, Gemma, Phi-3, Metharme, ShareGPT, Alpaca, OpenAI messages, any
Jinja chat template…).

* **Rust** for everything that touches data. **Tauri 2** desktop app. **ViraUI** (Core, free tier) for the interface.
* Built for **very large datasets** (PIPPA and bigger): files are memory-mapped and indexed on disk, never loaded whole.
* **Auto-detects the format** of almost anything (chat logs, instruction sets, roleplay cards, DPO pairs, text that
  already has a chat template, CSV/Parquet/JSON/JSONL, compressed files). A mapping editor exists for the rare miss.
* **Translate** with any OpenAI-compatible server (llama.cpp `llama-server`, vLLM, Ollama, LM Studio, OpenRouter…):
  you give a base URL, a model name and (optionally) a token.

Screenshots of every page are in [`docs/screenshots`](docs/screenshots).

## Run it

```bash
# 1. the desktop app (needs the Tauri system libraries on Linux: webkit2gtk-4.1, gtk3, libsoup3)
cd app/ui && npm install && cd ../..
cargo run -p dataset-mixer-app            # or: npx --prefix app/ui tauri dev   /   tauri build for an installer

# 2. the same UI in a browser (no desktop libraries needed)
(cd app/ui && npm install && npm run build)
cargo run -p mixer-cli -- serve           # http://127.0.0.1:8787

# 3. headless
cargo run -p mixer-cli -- run examples/recipes/pippa-chatml.toml
```

Try it without any model: `cargo run -p mixer-cli -- mock-llm` starts a fake OpenAI-compatible server on
`http://127.0.0.1:8089/v1` that "translates" English to Portuguese with a small dictionary. The app talks to it
exactly as it talks to a real server, so switching to your llama.cpp only means changing the URL and model name.

### Your llama.cpp server

```bash
llama-server -m your-model.gguf --port 8080 --parallel 4 -c 16384 [--api-key SECRET]
```

On the **Translate** page: Base URL `http://127.0.0.1:8080/v1`, the model name, the token if you set `--api-key`,
then **Test connection** (it lists models, measures speed, reads the number of slots from llama.cpp and suggests
the parallelism). `Context budget` is the number of tokens per request: use about `-c` divided by `--parallel`.

How translation works:

* Many requests run in parallel up to your limit (plus optional requests/minute and tokens/minute limits);
  rate-limit and server errors are retried with backoff, honouring `Retry-After`.
* Long chats are split to fit your context; each request asks for JSON and every answer is validated (same number
  of messages, protected spans intact, not still in the source language). A bad batch is retried message by
  message, then the row is dropped or kept (your choice).
* **Never translated:** code, URLs, LaTeX, `{{char}}` / `{{user}}` and the character/user names. Names are
  filled in at export, so "Rose" never becomes "Rosa".
* Each character card is translated once and reused for all its conversations.
* **Resumable:** every finished piece goes into a SQLite cache keyed by content, language and model. Stop,
  crash or re-run: nothing is paid twice.

### Recipes

A recipe is the whole mix in one TOML file (see [`examples/recipes`](examples/recipes)). The UI edits it, the
CLI runs it unchanged, and the same recipe + seed always produces byte-identical output. Tokens are never stored
in it: use the keychain (the app), or the `MIXER_LLM_TOKEN` / `HF_TOKEN` environment variables (the CLI).

## Layout

```
crates/mixer-core   data model, auto-detection, indexing, mixing, exporters, translation engine, Hugging Face
crates/mixer-api    session state, background jobs, secrets: one `call(command, args)` used by app and bridge
crates/mixer-cli    `mixer run | inspect | test-llm | mock-llm | serve`
crates/mixer-mock   mock OpenAI-compatible server (+ fake Hugging Face Hub) for development and tests
app/src-tauri       Tauri 2 desktop shell
app/ui              React 19 + @viraui/react interface (talks to a `Backend`: Tauri or HTTP)
scripts/            demo.sh + screenshots.mjs (drives the real UI), gen_big.py (stress-test data)
```

`cargo test` runs the whole suite: format detection on 25+ shapes, exporters, mixing, the translation pipeline
against the mock with injected faults (429, 500, garbage JSON, missing items, untranslated answers, timeouts),
kill-and-resume, resumable Hugging Face downloads against a fake Hub, and the API flow the UI uses.

## Notes

* ViraUI is proprietary (free Core tier, private use). The UI uses the published package and its themes.
* Token counts for filters and budgets are estimates (about 3.6 characters per token), not a real tokenizer.
* Zip/tar archives are not unpacked (gzip and zstd files are).
