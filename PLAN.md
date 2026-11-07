# Dataset Mixer — Implementation Plan

A desktop app (plus a headless CLI) for building LLM fine-tuning and training datasets. You choose the source datasets, set how much of each to take, shuffle them together, translate the result with an LLM if needed, and export it in a training format (ChatML and others).

- **Language:** Rust for everything that touches data (loading, mixing, translation, export).
- **UI:** a minimal design that uses the free **ViraUI Core** components and built-in theme.

---

## 1. Architecture

ViraUI ships only as React components (`@viraui/react`, built on Base UI, React ≥ 19), so the plan pairs a Rust backend with a React view:

| Layer | Tech | Responsibility |
|---|---|---|
| `crates/mixer-core` | Rust library | Data model, loaders, schema detection, mixing/shuffle, filters, exporters, translation engine. No UI code. |
| `crates/mixer-cli` | Rust + `clap` | Headless runs of saved "recipes" (for scripting, servers, CI). |
| `app/src-tauri` | Rust + **Tauri 2** | Desktop shell. Exposes `mixer-core` to the UI as Tauri commands and progress events. |
| `app/ui` | React 19 + Vite + TypeScript + `@viraui/react` | View only: forms, previews, progress. Holds no dataset data. Rust sends rows one page at a time. |

**Why Tauri and not a pure-Rust UI (Leptos or Dioxus)?** The ViraUI components only work in React. A Leptos UI could reuse ViraUI's CSS tokens (`@viraui/foundation`), but every component would have to be rebuilt by hand. Tauri keeps 100% of the logic in Rust and still gets the real ViraUI components. The final binary is small (it uses the system webview, not a bundled Chromium).

### Recipe = the single source of truth
Every mix is a serializable `Recipe` (TOML/JSON). The UI edits it, and the CLI runs it unchanged (`mixer run recipe.toml`), so any result can be reproduced from its recipe and seed.

```toml
seed = 42
[[source]]
id = "hf:teknium/OpenHermes-2.5"
weight = 0.6            # or: count = 50000
mapping = "auto"        # or explicit column mapping
[[source]]
id = "file:./my_chats.jsonl"
weight = 0.4
[filters]
min_turns = 1
max_tokens = 4096
dedup = "exact"
[split]
validation = 0.02
[export]
format = "chatml"
container = "jsonl"     # jsonl | parquet | json
shard_size = 100000
[translate]             # optional
enabled = false
target_language = "pt-BR"
```

---

## 2. Data model (`mixer-core::model`)

```rust
pub enum Role { System, User, Assistant, Tool }
pub struct Message { pub role: Role, pub content: String }
pub struct Sample {
    pub id: u64,                 // xxh3 hash of the normalized content (dedup + translation cache key)
    pub source: SourceId,
    pub kind: SampleKind,        // Chat(Vec<Message>) | Text(String)  (Text = pretraining/completion)
    pub meta: serde_json::Map<String, Value>,
}
```

All inputs are normalized into `Sample`, and all exporters read from `Sample`. Supporting a new format means adding one adapter, not N×M converters.

---

## 3. Loading sources (`mixer-core::source`)

**Local files:**
- JSONL, JSON (array), CSV, Parquet, plain-text folders.
- Libraries: `serde_json`, `csv`, `arrow` + `parquet`.

**Hugging Face Hub:**
- Point to a dataset by repo id plus optional config, split, and revision.
- Resolve its Parquet files through the datasets-server `/parquet` endpoint, then download them with the `hf-hub` crate (uses the local HF cache, supports gated datasets through an HF token).

**Schema auto-detection,** with a manual column-mapping override in the UI:

| Detected shape | Example keys |
|---|---|
| OpenAI messages | `messages[{role, content}]` |
| ShareGPT | `conversations[{from, value}]` (`human/gpt/system`) |
| Alpaca | `instruction`, `input`, `output` |
| Prompt/completion | `prompt`, `completion` / `question`, `answer` |
| Raw text | `text` |

**Scaling to large datasets:**
- Loaders are streaming iterators.
- The mixer works on an **index** (`(source, file, row-group, offset)`) instead of on samples held in memory, so shuffling 10M rows needs only a few hundred MB.
- Full rows are read only for preview, filtering, translation, and export.

---

## 4. Mixing and shuffling (`mixer-core::mix`)

**Per-source quota:** set either a `weight` (a share of the total) or an absolute `count`. Strategies:
- **Proportional:** use each source as-is.
- **Fixed count:** sample N rows without replacement.
- **Upsample:** repeat rows, with a cap (e.g. at most 3×).
- **Temperature sampling:** `p_i ∝ n_i^(1/T)`, to balance large and small sources.

**Shuffle:**
- A Fisher–Yates shuffle over the global index, using `rand_chacha::ChaCha8Rng` seeded from the recipe, so results are reproducible.
- Optional "interleave" mode: round-robin across sources, then a local shuffle.

**Filters** (run before or after sampling, configurable):
- Minimum/maximum number of turns.
- Minimum/maximum tokens, counted with `tokenizers` (any HF tokenizer.json) or `tiktoken-rs`.
- Drop empty assistant turns.
- Language filter, using `whatlang`.
- Regex include/exclude.

**Dedup:**
- Exact dedup by xxh3 hash.
- Phase 2: near-duplicate removal with MinHash LSH.

**Split:** train/validation by ratio, deterministic from the seed.

**Stats:** rows per source (requested vs. actual), token histogram, average turns, and duplicates removed. These feed the UI meters.

---

## 5. Export formats (`mixer-core::export`)

Every format implements `trait Exporter { fn write(&mut self, s: &Sample) -> Result<()>; fn finish(self) -> Result<Manifest>; }`.

| Format | Output shape |
|---|---|
| **ChatML** | `{"text": "<\|im_start\|>system\n…<\|im_end\|>\n<\|im_start\|>user\n…"}` |
| OpenAI / HF messages | `{"messages":[{"role":…,"content":…}]}` (works with TRL `SFTTrainer`, Axolotl, Unsloth) |
| ShareGPT | `{"conversations":[{"from":"human","value":…}]}` |
| Alpaca | `{"instruction","input","output"}` (single-turn only, warns when it drops turns) |
| Llama 3 | `<\|begin_of_text\|><\|start_header_id\|>user<\|end_header_id\|>…<\|eot_id\|>` |
| Mistral | `<s>[INST] … [/INST] …</s>` |
| Gemma | `<start_of_turn>user\n…<end_of_turn>` |
| Phi-3 | `<\|user\|>\n…<\|end\|>` |
| Prompt/completion | `{"prompt","completion"}` |
| Raw text (pretraining) | `{"text": …}` or `.txt` |
| **Custom template** | Any Jinja chat template, rendered with `minijinja`. It can load `chat_template` straight from a model's `tokenizer_config.json` on the Hub, so any model is supported. |

**Containers:** JSONL (default), Parquet, JSON, with optional sharding (`train-00000-of-00004.jsonl`).

**Manifest:** each export writes a `manifest.json` containing the recipe, seed, per-source counts, and hashes, so any export can be traced back to how it was made.

**Phase 3:** a "Push to Hugging Face Hub" option.

---

## 6. The **Translate** button (`mixer-core::translate`)

Translates the whole mixed dataset (or one source) into a target language by sending many concurrent requests to an LLM API, within limits you set.

### Providers
- **OpenAI-compatible `/v1/chat/completions`:** one adapter covers OpenAI, OpenRouter, Groq, Together, DeepSeek, Mistral, Ollama, LM Studio, and vLLM. You only change the base URL.
- **Anthropic Messages API:** native adapter.
- HTTP is handled by `reqwest` and `tokio`.
- API keys are stored in the OS keychain with the `keyring` crate. They are never written to recipes or logs.

### Limits and throughput

| Setting | Implementation |
|---|---|
| Max concurrent requests | `tokio::sync::Semaphore` |
| Requests/minute and tokens/minute | `governor` rate limiters (tokens are estimated before sending) |
| Retries | Exponential backoff with jitter. Honors `Retry-After` on 429 and 5xx responses. |
| Per-request timeout | Configurable |
| Budget cap | Stops when estimated spend reaches a $ limit or a token limit |

### How a sample is translated
1. **One request per conversation.** The conversation is sent as JSON, and the model must return the **same JSON structure** (same number of messages and the same roles), via JSON mode or a structured-output schema where the provider supports it.
2. **Validation:** the role order must match, nothing may be empty, and the output must actually be in the target language (`whatlang` check). If validation fails, the sample is retried message by message. If it still fails, it is marked `failed` and left out of the export (or kept in the original language, if you choose that).
3. **Content protection:** code blocks, inline code, URLs, `{placeholders}`, and LaTeX are swapped for tokens before sending and restored afterwards, so code is never translated.
4. **Options:**
   - Choose which roles to translate (e.g. skip the system prompt).
   - Custom instructions (tone, glossary).
   - Choose the model.
   - Output mode: *replace* the original, or *bilingual* (keep both versions).

### Reliability
- **Resumable:** every finished sample is written to a SQLite journal (`rusqlite`), keyed by `(sample hash, target language, model)`. After a crash or cancel, a run picks up where it stopped, and re-running a mix never pays twice for the same sample.
- **Pre-flight estimate:** before starting, a dialog shows the sample count, estimated input and output tokens, estimated cost (based on a per-model price you enter), and ETA.
- **Live progress:** Tauri events report done, failed, and in-flight counts, tokens/sec, and spend so far. You can pause, resume, or cancel.
- **Pipeline order:** by default, translation runs **after** mixing and filtering, so only the rows that will actually be exported are translated (much cheaper).

---

## 7. UI and design (ViraUI Core)

**Packages:**
- `@viraui/react` and its peer `@base-ui/react`.
- `@viraui/foundation` for the built-in theme (`vira.css`; `cinder`, `sunburst`, and `vira-condensed` also exist).
- `@viraui/react/preflight.css`.
- **Geist Variable** font.
- Import order at the root: theme CSS → fonts → preflight. Mount the Dialog, Tooltip, and Toast providers once in the app shell.

**Look**, taken from viraui.dev's own tokens:
- Neutral oklab grayscale. Background `oklab(97.6% 0 0)` (light) or `oklab(0% 0 0)` (dark), switched automatically through `light-dark()`.
- Hairline borders (`--global-contrast`, about 11% alpha).
- `--radius-m: .75rem`, spacing scale `--space-xs…5xl`.
- Accent colors only for status (`--highlight-green`, `--highlight-yellow`, `--highlight-red`).
- Lots of whitespace and one primary action per screen.

> **Licensing note:** ViraUI is proprietary (`SEE LICENSE IN LICENSE`, terms at viraui.dev/terms-conditions). Core is free on npm. Use the published package and its built-in themes rather than copying the marketing site's CSS or assets. If this repo will be public or redistributed, read the terms first.

**Layout:**
- A sidebar app shell with a top-bar recipe name and a primary **Export** button.
- Sidebar pages (each built from ViraUI components):

| Page | Contents | ViraUI components |
|---|---|---|
| **Sources** | Add a source by dropping files or pasting an HF id. Each source gets a card showing its detected schema, row count, and a mapping editor. | `Surface`, `Stack`, `Grid`, `TextField`, `Select`, `Badge`, `Chip`, `Dialog`, `Skeleton` |
| **Mix** | Per-source `Slider` for weight (or `NumberField` for count), a strategy `ToggleGroup`, seed, filters, split. Live `Meter` bars show the resulting composition. | `Slider`, `NumberField`, `ToggleGroup`, `Switch`, `Meter`, `Accordion` |
| **Preview** | Paginated sample viewer that renders chats as bubbles. Tabs for *Raw / Formatted / Translated*. | `Tabs`, `Prose`, `Text`, `IconButton`, `Tooltip` |
| **Translate** | Provider, model, target language, and limits (concurrency, RPM, TPM, budget). **Translate** button → estimate `Dialog` → live `LinearProgress`, stats, pause/cancel. | `Button`, `Select`, `Autocomplete`, `NumberField`, `LinearProgress`, `Gauge`, `Toast` |
| **Export** | Format `Select` with a live preview of one formatted sample, container, sharding, output folder. | `Select`, `RadioGroup`, `Button`, `Toast` |
| **Settings** | API keys (keychain), HF token, theme toggle, cache location. | `TextField`, `Switch`, `Fieldset` |

**UI ↔ Rust (Tauri commands):** `add_source`, `inspect_source`, `set_mapping`, `update_recipe`, `compute_mix` (returns stats), `preview(page, view)`, `estimate_translation`, `start_translation` / `pause` / `resume` / `cancel`, `export(recipe, path)`, `save_recipe` / `load_recipe`. Long jobs emit `progress://{job_id}` events.

---

## 8. Repository layout

```
dataset-maker/
├─ Cargo.toml                # workspace
├─ crates/
│  ├─ mixer-core/src/{model,source,detect,mix,filter,export,translate,recipe}.rs
│  └─ mixer-cli/src/main.rs
├─ app/
│  ├─ src-tauri/             # Tauri 2 crate, depends on mixer-core
│  └─ ui/                    # Vite + React 19 + @viraui/react
├─ examples/recipes/*.toml
└─ tests/fixtures/           # tiny sharegpt/alpaca/openai/parquet samples
```

---

## 9. Milestones

| # | Deliverable | Done when |
|---|---|---|
| **M1 Core** | Data model, local loaders (JSONL/JSON/CSV/Parquet), schema detection, weighted mixing with seeded shuffle, ChatML and OpenAI-messages exporters, CLI `mixer run recipe.toml` | The same recipe and seed give a byte-identical output. Fixture tests pass. |
| **M2 App shell** | Tauri + ViraUI shell, Sources, Mix, Preview, and Export pages wired to core | You can mix two local files and export ChatML from the GUI. |
| **M3 Translate** | Provider adapters, semaphore and rate limits, retries, SQLite journal, estimate dialog, live progress, pause/resume | A 1,000-row translation over an OpenAI-compatible API and over a local Ollama model survives being killed mid-run and resumes without re-sending finished rows. |
| **M4 Formats and Hub** | HF Hub sources, all template exporters, custom Jinja / `tokenizer_config.json` templates, Parquet output, sharding, manifest | Output matches `tokenizer.apply_chat_template` for the Llama 3, Mistral, Gemma, and Qwen fixtures. |
| **M5 Polish** | Filters (tokens, language, regex), dedup, train/val split, stats, dark mode, installers (macOS, Windows, Linux) via Tauri bundler | Release builds produced in CI. |
| Later | MinHash near-dedup, push to Hub, bilingual mixes, quality scoring with an LLM judge | — |

---

## 10. Testing

- **Unit tests:** every detector and exporter, using golden files in `tests/fixtures`.
- **Property tests (`proptest`):** shuffling keeps every row, weights add up, and a split never puts the same row in both train and validation.
- **Translation tests:** run against a mock HTTP server (`wiremock`). It injects 429 responses with `Retry-After`, timeouts, and malformed JSON, and checks that limits are respected, retries happen, and the journal resumes correctly.
- **UI:** Playwright smoke test against `tauri dev`.

---

## 11. Open questions

1. Desktop app only, or should the UI also be servable in a browser? With an `axum` server mode, the same React UI can talk HTTP instead of Tauri IPC.
2. Which LLM provider or model do you expect to use for translation most often? That determines which adapter and defaults to tune first.
3. Is the main use SFT (chat) datasets, or do you also need pretraining (raw text) mixes at large scale (100M+ rows)? The latter changes the indexing design.
4. Will the repo be public? See the ViraUI licensing note.
