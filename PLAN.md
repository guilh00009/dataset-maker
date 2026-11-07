# Dataset Mixer — Implementation Plan

A **desktop app** (plus a headless CLI) for building LLM fine-tuning and training datasets. You choose the source datasets, set how much of each to take, shuffle them together, translate the result with an LLM if needed, and export it in a training format (ChatML and others).

- **Language:** Rust for everything that touches data (loading, mixing, translation, export).
- **UI:** a minimal design that uses the free **ViraUI Core** components and built-in theme.

---

## 0. Confirmed decisions

| Topic | Decision |
|---|---|
| Platform | **Desktop only for now** (Tauri 2). Built so a **web version can be added later** without rewriting anything (§1.1). |
| Translation LLM | **Any OpenAI-compatible server**, primarily **llama.cpp `llama-server`** run locally or on a remote machine. You set three things: **Base URL**, **Model name**, **API token** (optional, matches `llama-server --api-key`). |
| Testing without your server | Development uses a **mock OpenAI-compatible server**, but the app always takes the **same real HTTP code path**. The mock is just another URL; production code has no mock-specific branches. When you point the app at your llama.cpp server, it works unchanged (§6.6). |
| Input formats | **Automatic detection that works with almost any format.** You only point to the file or HF id. The manual mapping editor is just a fallback for when detection isn't confident (§3.2). |
| Dataset scale | **Very large datasets**, e.g. **PIPPA** (`PygmalionAI/PIPPA`: 349 MB JSONL, ~26k conversations, ~1M messages, persona fields, `viewer: false`). Multi-GB inputs must work without loading everything into RAM (§3.3). |
| Translation default | **English → Portuguese (Brazil)**: source `en`, target `pt-BR`, both changeable on the Translate page. Character and user **names are never translated**. |
| Usage / licensing | **Private, personal use only.** The ViraUI Core free tier is fine for this. |
| Roleplay placeholders | `{{char}}` and `{{user}}` are **replaced with the real character name and user name** on export (§3.2). |
| Look | "Keep it beautiful": a minimal ViraUI design, light and dark. **Screenshots of every page** are delivered when it's ready (§10). |

---

## 1. Architecture

ViraUI ships only as React components (`@viraui/react`, built on Base UI, React ≥ 19), so the plan pairs a Rust backend with a React view:

| Layer | Tech | Responsibility |
|---|---|---|
| `crates/mixer-core` | Rust library | Data model, loaders, schema detection, indexing, mixing/shuffle, filters, exporters, translation engine. No UI code. |
| `crates/mixer-cli` | Rust + `clap` | Headless runs of saved "recipes" (`mixer run recipe.toml`), plus `mixer mock-llm` (§6.6). |
| `app/src-tauri` | Rust + **Tauri 2** | Desktop shell. Exposes `mixer-core` to the UI as Tauri commands and progress events. |
| `app/ui` | React 19 + Vite + TypeScript + `@viraui/react` | View only: forms, previews, progress. Holds no dataset data. Rust sends rows one page at a time. |

**Why Tauri and not a pure-Rust UI (Leptos or Dioxus)?** The ViraUI components only work in React. Tauri keeps all the logic in Rust, and the final binary is small (it uses the system webview).

### 1.1 Ready for a web version later (not built now)
- The UI never calls Tauri directly. It talks to a small `Backend` TypeScript interface (`invoke(cmd, args)` and `subscribe(jobId, onEvent)`).
- Shipped now: `TauriBackend`.
- Later: an `HttpBackend` that talks to an `axum` server wrapping the same `mixer-core` (REST + Server-Sent Events). Adding the web version then means adding that server, with no UI rewrite.
- A minimal dev-only version of this HTTP bridge is used to take the browser screenshots (§10). It is not shipped as a product.

### 1.2 Recipe = the single source of truth
Every mix is a serializable `Recipe` (TOML). The UI edits it, and the CLI runs it unchanged, so any result can be reproduced from its recipe and seed.

```toml
seed = 42
[[source]]
id = "hf-file:PygmalionAI/PIPPA/pippa_deduped.jsonl"
weight = 0.5            # or: count = 20000
mapping = "auto"        # auto-detects the PIPPA persona schema
[[source]]
id = "file:./my_chats.jsonl"
weight = 0.5
[filters]
min_turns = 3
max_tokens = 8192
dedup = "exact"
[split]
validation = 0.02
[export]
format = "chatml"
container = "jsonl"     # jsonl | parquet | json
shard_size = 100000
[translate]
enabled = true
base_url = "http://127.0.0.1:8080/v1"
model = "qwen2.5-14b-instruct-q4_k_m"
# token lives in the OS keychain, never in the recipe
source_language = "en"
target_language = "pt-BR"
concurrency = 4
```

---

## 2. Data model (`mixer-core::model`)

```rust
pub enum Role { System, User, Assistant, Tool }
pub struct Message { pub role: Role, pub content: String }
pub struct Sample {
    pub id: u64,                 // xxh3 hash of the normalized content (dedup + translation cache key)
    pub source: SourceId,
    pub kind: SampleKind,        // Chat(Vec<Message>) | Text(String)
    pub persona: Option<Persona>,// name / description / definitions / greeting (roleplay datasets)
    pub meta: serde_json::Map<String, Value>,
}
```

All inputs are normalized into `Sample`, and all exporters read from `Sample`. `Persona` is kept as structured data until export, so it can be translated field by field and rendered into a system prompt with an editable template.

---

## 3. Loading sources (`mixer-core::source`)

### 3.1 Where data comes from
- **Local files:** JSONL, JSON (array), CSV, Parquet, plain-text folders. Libraries: `serde_json` (`simd-json` for hot paths), `csv`, `arrow` + `parquet`.
- **Hugging Face Hub**, two paths:
  1. **Parquet path:** resolve through the datasets-server `/parquet` endpoint (for datasets that have the viewer).
  2. **Direct repo file path:** list the repo's files and download a specific one (e.g. `pippa_deduped.jsonl`). This is **required for PIPPA** and other datasets with a loader script or no viewer.

  Both download with `hf-hub`, which uses the local HF cache, shows progress, resumes interrupted downloads, and supports gated datasets through an HF token.

### 3.2 Format auto-detection (works with almost any dataset)
You just point the app at a file or a Hugging Face id. Detection is **automatic and heuristic**, not limited to a fixed list of known formats. A manual mapping editor exists only as a fallback.

**Step 1: file format sniffing.** Uses the file's content and magic bytes, never the extension. Recognizes:
- JSONL / NDJSON.
- A JSON array, or a JSON object that wraps the rows (e.g. `{"data": [...]}`).
- CSV/TSV (delimiter and header sniffed).
- Parquet, Arrow/Feather.
- Plain text.
- Any of the above compressed with gzip, zstd, or bz2, or packed in a zip/tar archive.

**Step 2: structure detection** on a sample of rows (about 200, spread across the file):
1. **Flatten each row into paths** (`data.messages[].content`, …) and look for candidate fields, at any nesting depth.
2. **Conversation detector:** find arrays of objects whose keys look like *role* and *content*, and score them by how well they fit.
   - Role-like keys: `role`, `from`, `speaker`, `author`, `sender`, `name`, `is_human`, `is_user`, `is_bot`, …
   - Content-like keys: `content`, `value`, `text`, `message`, `msg`, `utterance`, `response`, …
   - Role values are mapped through a synonym table: `human`, `user`, `USER`, `prompter`, `you` → user; `gpt`, `bot`, `assistant`, `model`, `AI`, `char`, the character's name → assistant; `system` → system.
   - Booleans like `is_human` are handled too.
   - If roles are missing, the detector infers them from alternation and the greeting position.
3. **Arrays of plain strings** (`["hi", "hello!", …]`) become alternating user/assistant turns.
4. **Pair detector:** for flat rows, pick the best prompt-like field (`instruction`, `prompt`, `question`, `query`, `input`, `context`, `human`) and the best response-like field (`output`, `response`, `answer`, `completion`, `assistant`, `chosen`). An extra `input`/`context` field is folded into the prompt.
5. **Preference datasets:** `chosen`/`rejected` rows (including when they hold full conversations) export the *chosen* side, or keep both for DPO export later.
6. **Persona and system detector:** fields like `system`, `system_prompt`, `persona`, `character`, `char_name`, `bot_name`, `description`, `definitions`, `scenario`, `greeting`, `first_mes` go into `Persona`/system.
7. **Pre-formatted text parser:** if the only content is a `text` field that already contains a chat template, it is **parsed back into turns**. Recognized templates: ChatML `<|im_start|>`, Llama 2/3, Mistral `[INST]`, Gemma, Phi, Metharme, Vicuna `USER:/ASSISTANT:`, Alpaca `### Instruction:/### Response:`, `Human:/Assistant:`. Otherwise it stays a raw-text sample.
8. **Scoring:** every candidate interpretation gets a confidence score, based on coverage across the sampled rows and how well the types and roles fit. The best one wins.

**Step 3: confirm.**
- The source card shows the detected interpretation (e.g. *"Conversation · `conversation[].message` · role from `is_human` · persona from 4 fields · 98%"*) and a preview of three normalized samples.
- If confidence is low (< 80%) or there are several close candidates, the **mapping editor opens pre-filled** with the top suggestions. You pick fields from dropdowns and see the result live.
- The chosen mapping is saved in the recipe.

**Tests:** a fixture corpus of 25+ real-world shapes (OpenAI messages, ShareGPT, PIPPA, Alpaca, Dolly, OASST, UltraChat, Capybara, SillyTavern / character-card logs, DPO pairs, pre-formatted ChatML/Llama text, CSV Q&A, nested JSON, …), each with its expected normalized output.

Common shapes it will handle out of the box:

| Shape | Example keys | Normalization |
|---|---|---|
| OpenAI messages | `messages[{role, content}]` | direct |
| ShareGPT | `conversations[{from, value}]` | `human→user`, `gpt→assistant` |
| **PIPPA / persona chat** | `bot_name`, `bot_description`, `bot_definitions`, `bot_greeting`, `conversation[{message, is_human}]` | `is_human` → user or assistant. The greeting becomes the first assistant turn. Persona fields go into `Persona`. |
| Alpaca / Dolly | `instruction`, `input`/`context`, `output`/`response` | single turn |
| Prompt/completion, Q&A | `prompt`/`completion`, `question`/`answer` | single turn |
| Preference (DPO) | `prompt`, `chosen`, `rejected` | the chosen side |
| Pre-formatted text | `text` containing ChatML / Llama / Alpaca markers | parsed into turns |
| Raw text | `text` | `Text` sample |

**PIPPA specifics:**
- The persona becomes the system prompt through an editable template, e.g. `"{{bot_name}}: {{bot_description}}\n\n{{bot_definitions}}"`.
- **`{{char}}` and `{{user}}` are replaced with real names on export (the default).** This applies to every text field: the system prompt, definitions, greeting, and all messages. The same rule applies to any persona dataset with these placeholders, not just PIPPA. Name variants `{{Char}}`, `{{CHAR}}`, `<BOT>`, `<USER>`, and `{char}` are recognized too.
  - **`{{char}}`** becomes the character's real name from the dataset (`bot_name` in PIPPA, or `char_name` / `name` / `character` in other datasets, found by auto-detection).
  - **`{{user}}`** becomes the user's real name when the dataset has one (e.g. `user_name`, the `name` on user turns in SillyTavern-style logs, or a persona field). PIPPA has no user-name field, so a **fallback name** is used. It's set in the Mix page and defaults to `User`. There's an optional **"random name per conversation"** pool so the model doesn't learn one fixed name.
  - Names are **never translated:** placeholders stay protected during translation (§6.3) and are replaced only at export, so a name like "Rose" isn't turned into "Rosa".
- Character.AI's `END_OF_DIALOG` markers in the definitions are cleaned up.
- An option drops conversations shorter than N turns (PIPPA recommends at least 3).

### 3.3 Very large datasets
- **Nothing is fully loaded into memory.**
  - On first open, a JSONL file is **memory-mapped** (`memmap2`) and scanned with `memchr` to build a **line offset index** (`Vec<u64>`, 8 bytes per row).
  - Parquet uses row-group and row offsets instead.
  - The index is saved in the app cache, keyed by file size + mtime + hash of the first MB, so reopening a file is instant.
- **Parallel parsing** with `rayon` for detection, filtering, stats, and export. Workers read rows by offset straight from the memory map.
- **The mixer only reorders row positions** (`(source, offset)`), never full rows. Shuffling 10M rows needs about 160 MB.
- **Preview** reads only the rows shown on screen, by offset, so it stays fast on huge files.
- **Export** streams rows to disk in shuffled order, keeping memory flat.
- **Progress, cancel, and errors:** every long step (download, index, filter, export, translate) runs as a background job with progress and a cancel button. Malformed lines are counted and logged, not fatal.
- **Very long conversations** (PIPPA has some with hundreds of messages) are handled by `max_tokens` filtering, or optionally **split into windows** of at most N tokens that keep the system prompt.

---

## 4. Mixing and shuffling (`mixer-core::mix`)

**Per-source quota:** set either a `weight` or an absolute `count`. Strategies:
- **Proportional:** use each source as-is.
- **Fixed count:** sample N rows without replacement.
- **Upsample:** repeat rows, with a cap.
- **Temperature sampling:** `p_i ∝ n_i^(1/T)`.

**Shuffle:**
- A Fisher–Yates shuffle over the global row index, using `ChaCha8Rng` seeded from the recipe. The same recipe and seed always give the same output.
- Optional interleave mode.

**Filters:**
- Minimum/maximum turns.
- Minimum/maximum tokens, counted with `tokenizers` (any HF `tokenizer.json`) or a fast characters÷4 estimate.
- Drop empty assistant turns.
- Language detection (`whatlang`).
- Regex include/exclude.

**Dedup:** exact, by xxh3 hash. MinHash near-dedup comes later.

**Split:** train/validation by ratio, deterministic from the seed.

**Stats:** rows per source (requested vs. actual), token histogram, turns per conversation, duplicates and malformed rows removed.

---

## 5. Export formats (`mixer-core::export`)

Every format implements `trait Exporter { fn write(&mut self, s: &Sample) -> Result<()>; fn finish(self) -> Result<Manifest>; }`.

| Format | Output shape |
|---|---|
| **ChatML** | `{"text": "<\|im_start\|>system\n…<\|im_end\|>\n<\|im_start\|>user\n…"}` |
| OpenAI / HF messages | `{"messages":[{"role":…,"content":…}]}` (works with TRL, Axolotl, Unsloth) |
| ShareGPT | `{"conversations":[{"from":"human","value":…}]}` |
| Alpaca | `{"instruction","input","output"}` (warns when it drops turns) |
| Llama 3 | `<\|begin_of_text\|><\|start_header_id\|>user<\|end_header_id\|>…<\|eot_id\|>` |
| Mistral | `<s>[INST] … [/INST] …</s>` |
| Gemma | `<start_of_turn>user\n…<end_of_turn>` |
| Phi-3 | `<\|user\|>\n…<\|end\|>` |
| Metharme (Pygmalion) | `<\|system\|>…<\|user\|>…<\|model\|>…` |
| Prompt/completion | `{"prompt","completion"}` |
| Raw text | `{"text": …}` or `.txt` |
| **Custom template** | Any Jinja chat template, rendered with `minijinja`. It can load `chat_template` from a model's `tokenizer_config.json` on the Hub. |

**Containers:** JSONL (default), Parquet, JSON, with optional sharding (`train-00000-of-00004.jsonl`).

**Manifest:** each export writes a `manifest.json` containing the recipe, seed, per-source counts, and hashes.

---

## 6. The **Translate** button (`mixer-core::translate`)

### 6.1 Connection settings (Settings → LLM, also editable on the Translate page)

| Field | Example | Notes |
|---|---|---|
| **Base URL** | `http://127.0.0.1:8080/v1` (local llama.cpp) · `https://my-server:8080/v1` (remote) | Any OpenAI-compatible server. Paths are normalized so both `…/v1` and the bare host work. |
| **Model name** | `qwen2.5-14b-instruct` | Sent as `model`. A dropdown is filled from `GET /v1/models`, and you can still type any name. |
| **API token** | `sk-…` (optional) | Sent as `Authorization: Bearer …`. Stored in the OS keychain (`keyring`). Leave it empty if your llama.cpp server has no `--api-key`. |

- **Test connection** button: calls `GET /v1/models`, then sends a tiny chat completion. It shows latency, tokens/sec, and the exact error message if something fails (unreachable, 401 bad token, unknown model).
- **llama.cpp extras (optional):** when the server is llama.cpp, the app reads `GET /props` to find the number of parallel slots (`--parallel`) and suggests that as the concurrency. If `/props` doesn't exist (vLLM, OpenAI, etc.), the app just skips this.
- **Request body:** standard `POST {base}/chat/completions` (`model`, `messages`, `temperature`, `max_tokens`, optional `response_format`), so the same code works with llama.cpp, vLLM, Ollama, LM Studio, OpenRouter, and OpenAI.

### 6.2 Limits and throughput

| Setting | Default | Implementation |
|---|---|---|
| Max concurrent requests | 4, or the slot count from llama.cpp `/props` | `tokio::sync::Semaphore` |
| Requests/minute, tokens/minute | off (local server) | `governor` rate limiters |
| Retries | 5 | Exponential backoff with jitter. Honors `Retry-After` on 429 and 5xx. Retries on connection reset and timeout. |
| Request timeout | 600 s | Long default for slow local generation |
| Max output tokens | auto (about 2× the input) | Sent as `max_tokens` |
| Budget cap | off | Optional cap in tokens (or $ if you enter a price) |

### 6.3 How a sample is translated
1. **Chunking:** a conversation is split into chunks that fit a configurable **context budget** (default 4096 input tokens, set to match the server's `--ctx-size`), keeping the order. Short conversations take one request. PIPPA-length ones take several.
2. **Structured output:** each chunk is sent as a JSON array of `{role, content}`. The model must return an array with the **same length and roles**. If the server supports `response_format` with `json_schema`, it is used (llama.cpp enforces it with a grammar). Otherwise the app falls back to prompt instructions plus strict parsing.
3. **Validation:** role count and order match, no empty messages, and the text is in the target language (`whatlang`).
   - If validation fails, the chunk is retried message by message.
   - If it still fails, the sample is marked `failed`. You choose whether failed samples are dropped or kept in the original language.
4. **Content protection:** code blocks, inline code, URLs, `{{char}}`/`{{user}}` and the character/user names themselves (replaced with real names only at export, §3.2), `{placeholders}`, `*action*` markers (optional), and LaTeX are replaced with opaque tokens (`⟦0⟧`) before sending and restored afterwards. A sample whose response loses a token fails validation.
5. **Persona fields** (`bot_description`, `bot_definitions`, `bot_greeting`) are translated too. Each **unique persona is translated only once** and reused (PIPPA has about 1k personas across 26k conversations).
6. **Options:**
   - Choose which roles to translate.
   - Custom instructions or glossary (e.g. "keep character names").
   - Temperature (default 0.2).
   - Output mode: **replace** the original, or **bilingual** (keep both versions).

### 6.4 Reliability for multi-day runs
- **Resumable:** every finished chunk is written to a **SQLite journal** (`rusqlite`, WAL mode), keyed by `(sample hash, chunk index, target language, model, prompt version)`. After a crash, a reboot, or a cancel, a run continues exactly where it stopped, and nothing is sent twice.
- **Pre-flight estimate dialog:** shows the number of samples and chunks, estimated input and output tokens, and an **ETA** based on the measured tokens/sec from Test connection.
- **Try-first option:** translate a random 50 samples, review them in Preview, then start the full run.
- **Live progress:** done, failed, and in-flight counts, tokens/sec, ETA, and the last error. Pause, resume, or cancel at any time.
- **Pipeline order:** translation runs after mixing and filtering, so only rows that will actually be exported are translated.

### 6.5 Request flow (per chunk)
`queue → acquire semaphore → wait for rate limiter → protect placeholders → POST /chat/completions → parse JSON → validate → restore placeholders → write to journal → emit progress event`

### 6.6 Mock server (for development; real code path)
- `mixer mock-llm --port 8089` is a small `axum` server that implements `GET /v1/models`, `POST /v1/chat/completions`, and `GET /props`, just like llama.cpp.
- It returns structurally valid "translations" (it prefixes each message and keeps placeholder tokens). It can **simulate** latency, 429 responses with `Retry-After`, 500 errors, timeouts, malformed JSON, and a required Bearer token.
- In development and tests, the app is pointed at `http://127.0.0.1:8089/v1` like any other server. Integration tests run the whole translation pipeline against it, including killing and resuming a run.
- **When you switch to your real setup**, you only change Base URL, Model, and Token in Settings. The real build has no other mock code path.

---

## 7. UI and design (ViraUI Core)

**Packages:**
- `@viraui/react` and its peer `@base-ui/react`.
- `@viraui/foundation` for the built-in `vira.css` theme.
- `@viraui/react/preflight.css`.
- **Geist Variable** font.
- Import order: theme CSS → fonts → preflight. Mount the Dialog, Tooltip, and Toast providers once in the app shell.

**Look**, taken from viraui.dev's own tokens:
- Neutral oklab grayscale: background `oklab(97.6% 0 0)` (light) or `oklab(0% 0 0)` (dark) through `light-dark()`.
- Hairline borders (`--global-contrast`, about 11% alpha).
- `--radius-m: .75rem`, spacing scale `--space-xs…5xl`.
- Accent colors only for status (`--highlight-green`, `--highlight-yellow`, `--highlight-red`).
- Lots of whitespace, one primary action per screen, and smooth but reduced-motion-aware transitions.

> **Licensing note:** ViraUI is proprietary, and Core is free on npm. This project is for private use, so the free tier is fine. The app uses the published package and its built-in themes, not the marketing site's own files.

**Layout:**
- A sidebar app shell with a top-bar recipe name, a job indicator (background download, index, or translate progress), and a primary **Export** button.
- Sidebar pages:

| Page | Contents | ViraUI components |
|---|---|---|
| **Sources** | Add a source by dropping files, browsing, or pasting an HF id (then pick a file or split). Each source card shows the detected schema badge (e.g. *PIPPA persona*), row count, size, index progress, and a mapping editor. | `Surface`, `Stack`, `Grid`, `TextField`, `Select`, `Badge`, `Chip`, `Dialog`, `Skeleton`, `LinearProgress` |
| **Mix** | Per-source weight `Slider` or count `NumberField`, strategy `ToggleGroup`, seed, filters, split. Live composition `Meter` bars and a token histogram. | `Slider`, `NumberField`, `ToggleGroup`, `Switch`, `Meter`, `Accordion` |
| **Preview** | Virtualized, paginated sample viewer with chat bubbles and a persona header. Tabs for *Raw / Formatted / Translated*, with a side-by-side original vs. translation view. | `Tabs`, `Prose`, `Text`, `IconButton`, `Tooltip` |
| **Translate** | Connection card (URL, model, token, **Test connection**), target language, limits, options. **Translate** button → estimate `Dialog` → live progress (`LinearProgress`, `Gauge` for tokens/sec, counters), pause/resume/cancel, failures list. | `Button`, `TextField`, `Autocomplete`, `NumberField`, `LinearProgress`, `Gauge`, `Toast`, `Dialog` |
| **Export** | Format `Select` with a live preview of one formatted sample, container, sharding, output folder, Export button with progress. | `Select`, `Radio`, `Button`, `Toast` |
| **Settings** | LLM connection defaults, HF token (keychain), theme (system/light/dark), cache location and size. | `TextField`, `Switch`, `Fieldset` |

**UI ↔ Rust (`Backend` interface → Tauri commands):** `add_source`, `inspect_source`, `set_mapping`, `update_recipe`, `compute_mix`, `preview(page, view)`, `llm_test_connection`, `llm_list_models`, `estimate_translation`, `start_translation` / `pause` / `resume` / `cancel`, `export`, `save_recipe` / `load_recipe`, `list_jobs`. Long jobs emit `progress://{job_id}` events.

---

## 8. Repository layout

```
dataset-maker/
├─ Cargo.toml                # workspace
├─ crates/
│  ├─ mixer-core/src/{model,source,detect,index,mix,filter,export,translate,recipe,jobs}.rs
│  └─ mixer-cli/src/main.rs  # run, inspect, mock-llm
├─ app/
│  ├─ src-tauri/             # Tauri 2 crate, depends on mixer-core
│  └─ ui/                    # Vite + React 19 + @viraui/react, Backend interface
├─ examples/recipes/*.toml   # incl. pippa-chatml.toml, pippa-translate-ptbr.toml
└─ tests/fixtures/           # small sharegpt/alpaca/openai/pippa/parquet samples
```

---

## 9. Milestones

| # | Deliverable | Done when |
|---|---|---|
| **M1 Core** | Data model, local loaders, memory-mapped JSONL offset index, heuristic auto-detection (§3.2) with the 25+ shape fixture corpus, weighted mixing with seeded shuffle, ChatML / OpenAI-messages / ShareGPT exporters, CLI `mixer run` | Same recipe and seed give byte-identical output. A 1 GB synthetic JSONL indexes and exports with memory use staying flat. |
| **M2 Translate engine** | OpenAI-compatible client, semaphore, rate limits, retries, chunking, placeholder protection, persona dedup, SQLite journal, `mixer mock-llm` | The full pipeline passes against the mock with 429, 500, timeout, and malformed-JSON faults injected. A run killed midway resumes without re-sending finished chunks. |
| **M3 App** | Tauri + ViraUI shell and all six pages wired to core, background jobs, Test connection | You can mix PIPPA with a local file, translate through the mock, and export ChatML entirely from the GUI. |
| **M4 Formats and Hub** | HF Hub (Parquet path and direct-file path), all template exporters, custom Jinja templates, Parquet output, sharding, manifest | `PygmalionAI/PIPPA` downloads and is detected. Template output matches `apply_chat_template` for the Llama 3, Mistral, Gemma, and Qwen fixtures. |
| **M5 Polish** | Filters, dedup, split, stats, conversation windowing, light and dark themes, Linux / macOS / Windows bundles | Screenshots delivered (§10). Release builds pass in CI. |
| Later | Web version (`axum` + `HttpBackend`), MinHash near-dedup, push to Hub, quality scoring | — |

---

## 10. Verification and screenshots

- **Automated tests:**
  - Unit tests for every detector and exporter, using golden files.
  - Property tests (`proptest`): shuffling keeps every row, a split never puts the same row in both sets, and weights add up.
  - Translation integration tests against `mixer mock-llm`, including fault injection and kill/resume.
- **Real-server check:** if a llama.cpp binary and a small GGUF model can be downloaded in the build environment, the translation flow is also run once against a real `llama-server`. If not, the report says so plainly, and §6.6 explains why switching to your server only needs a Settings change.
- **Screenshots** (Playwright, 2× scale, light and dark) of Sources (with PIPPA loaded), Mix, Preview (original vs. translation), Translate (connection test and a run in progress), Export, and Settings. The headless build environment has no display, so screenshots are taken of the same React UI in Chromium through the dev-only HTTP bridge (§1.1) against the real `mixer-core`. The desktop app renders the identical UI.
- **Report:** what was built, test results, and anything that's still missing.

---

## 11. Open questions
None. All of them are answered in §0.

---

## 12. Build status (end of the overnight build)

Everything in §0–§10 is implemented and exercised by tests or by the screenshot flow, with these differences:

| Plan said | What was built |
|---|---|
| `tokenizers` / tiktoken for token counts | A fast estimate (about 3.6 characters per token) for filters, budgets and ETAs. A real tokenizer can be added behind the same `est_tokens` function. |
| zip/tar archives | Not unpacked. gzip and zstd are (also bzip2 is not). |
| keychain only | OS keychain through the `keyring` crate, with a private `0600` file fallback when no keychain exists (the UI says which one is in use). |
| Real `llama-server` check | **Not run here**: GitHub release downloads are blocked in the build sandbox. The client speaks the plain OpenAI-compatible API (`/v1/models`, `/v1/chat/completions`, llama.cpp `/props`) and is tested against a mock that implements the same API with injected faults. |
| `hf-hub` crate | Own resumable downloader (token support, Range resume, retries), tested against a fake Hub; also verified against the **real Hugging Face Hub** with `PygmalionAI/PIPPA` (256.8 MB, 16,832 conversations, detected at 100%). |
| Playwright smoke test of the desktop window | The same React UI is driven in headless Chromium against the real Rust backend (`scripts/demo.sh`), and the Tauri shell is built and launched under a virtual display. |
| Installers (macOS/Windows/Linux) | Not built here (only `cargo build` of the Linux shell). `tauri build` produces them on each OS. |
| MinHash near-dedup, push to Hub, LLM judge | Still "later", as planned. |

Measured on a 1.5 GB / 418k-row PIPPA-shaped file (release build, 4 cores): full scan + filter + dedup + shuffle +
export of 20,000 ChatML rows in about 4 s with **under 75 MB of memory**. 300 real PIPPA conversations were
translated through the pipeline (chunked JSON requests, 8 in parallel, 3% injected 429s) with 0 failures.
