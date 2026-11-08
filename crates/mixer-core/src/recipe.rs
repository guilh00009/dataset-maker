//! A `Recipe` is the single source of truth for a mix. The UI edits it and the
//! CLI runs it unchanged.

use crate::detect::Mapping;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recipe {
    #[serde(default = "default_name")]
    pub name: String,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default, rename = "source")]
    pub sources: Vec<SourceEntry>,
    #[serde(default)]
    pub mix: MixOptions,
    #[serde(default)]
    pub filters: Filters,
    #[serde(default)]
    pub split: Split,
    #[serde(default)]
    pub persona: PersonaOptions,
    #[serde(default)]
    pub export: ExportOptions,
    #[serde(default)]
    pub translate: TranslateOptions,
}

fn default_name() -> String {
    "Untitled mix".into()
}
fn default_seed() -> u64 {
    42
}

impl Default for Recipe {
    fn default() -> Self {
        Self {
            name: default_name(),
            seed: default_seed(),
            sources: vec![],
            mix: MixOptions::default(),
            filters: Filters::default(),
            split: Split::default(),
            persona: PersonaOptions::default(),
            export: ExportOptions::default(),
            translate: TranslateOptions::default(),
        }
    }
}

impl Recipe {
    pub fn from_toml(s: &str) -> anyhow::Result<Self> {
        Ok(toml::from_str(s)?)
    }
    pub fn to_toml(&self) -> anyhow::Result<String> {
        Ok(toml::to_string_pretty(self)?)
    }
}

/// Where a source comes from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SourceLocation {
    /// A local file path.
    File { path: String },
    /// A file inside a Hugging Face dataset repository.
    Hf {
        repo: String,
        file: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        revision: Option<String>,
    },
}

impl SourceLocation {
    pub fn label(&self) -> String {
        match self {
            SourceLocation::File { path } => std::path::Path::new(path)
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| path.clone()),
            SourceLocation::Hf { repo, file, .. } => format!("{repo}/{file}"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceEntry {
    pub id: String,
    pub location: SourceLocation,
    /// Relative share of the mix (used when `count` is not set).
    #[serde(default = "default_weight")]
    pub weight: f64,
    /// Absolute number of rows to take. Overrides `weight`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub count: Option<u64>,
    /// Manual mapping; `None` means "use the auto-detected one".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mapping: Option<Mapping>,
    #[serde(default = "default_true")]
    pub enabled: bool,
}

fn default_weight() -> f64 {
    1.0
}
fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Strategy {
    /// Use weights/counts as given.
    Proportional,
    /// Weights are rescaled with `p_i ∝ n_i^(1/T)`.
    Temperature,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShuffleMode {
    Full,
    Interleave,
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixOptions {
    #[serde(default = "default_strategy")]
    pub strategy: Strategy,
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    /// Total number of rows wanted. `None` = as many as the weights allow.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total: Option<u64>,
    /// Maximum number of times one row may be repeated when a source is
    /// asked for more rows than it has. 1 = never repeat.
    #[serde(default = "default_upsample")]
    pub max_upsample: u32,
    #[serde(default = "default_shuffle")]
    pub shuffle: ShuffleMode,
}

fn default_strategy() -> Strategy {
    Strategy::Proportional
}
fn default_temperature() -> f64 {
    1.0
}
fn default_upsample() -> u32 {
    1
}
fn default_shuffle() -> ShuffleMode {
    ShuffleMode::Full
}

impl Default for MixOptions {
    fn default() -> Self {
        Self {
            strategy: default_strategy(),
            temperature: default_temperature(),
            total: None,
            max_upsample: 1,
            shuffle: ShuffleMode::Full,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DedupMode {
    None,
    Exact,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Filters {
    #[serde(default = "default_min_turns")]
    pub min_turns: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_turns: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_tokens: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
    #[serde(default = "default_true")]
    pub drop_empty_messages: bool,
    #[serde(default = "default_dedup")]
    pub dedup: DedupMode,
    /// ISO 639-3 code (`eng`, `por`, …) the sample must be written in.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_regex: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exclude_regex: Option<String>,
    /// Split conversations longer than this many (estimated) tokens into windows.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window_tokens: Option<usize>,
}

fn default_min_turns() -> usize {
    1
}
fn default_dedup() -> DedupMode {
    DedupMode::Exact
}

impl Default for Filters {
    fn default() -> Self {
        Self {
            min_turns: 1,
            max_turns: None,
            min_tokens: None,
            max_tokens: None,
            drop_empty_messages: true,
            dedup: DedupMode::Exact,
            language: None,
            include_regex: None,
            exclude_regex: None,
            window_tokens: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Split {
    /// Fraction in [0, 0.5) of rows moved to the validation set.
    #[serde(default)]
    pub validation: f64,
}

impl Default for Split {
    fn default() -> Self {
        Self { validation: 0.0 }
    }
}

/// How roleplay placeholders and persona fields become text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonaOptions {
    /// Template for the generated system prompt. Lines whose variables are all
    /// empty are dropped. Variables: char_name, user_name, description,
    /// definitions, scenario, greeting, system.
    #[serde(default = "default_persona_template")]
    pub template: String,
    /// Replace `{{char}}` / `{{user}}` (and variants) with real names on export.
    #[serde(default = "default_true")]
    pub substitute_names: bool,
    /// Name used for `{{user}}` when the dataset has none.
    #[serde(default = "default_user_name")]
    pub fallback_user_name: String,
    /// Pick a random user name per conversation from this pool instead of the fallback.
    #[serde(default)]
    pub user_name_pool: Vec<String>,
    /// Name used for `{{char}}` when the dataset has none.
    #[serde(default = "default_char_name")]
    pub fallback_char_name: String,
}

pub fn default_persona_template() -> String {
    "{{system}}\nYou are {{char_name}}. {{description}}\n\n{{definitions}}\n\nScenario: {{scenario}}".into()
}
fn default_user_name() -> String {
    "User".into()
}
fn default_char_name() -> String {
    "Assistant".into()
}

impl Default for PersonaOptions {
    fn default() -> Self {
        Self {
            template: default_persona_template(),
            substitute_names: true,
            fallback_user_name: default_user_name(),
            user_name_pool: vec![],
            fallback_char_name: default_char_name(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Container {
    Jsonl,
    Json,
    Parquet,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Format {
    Chatml,
    Messages,
    Sharegpt,
    Alpaca,
    Llama3,
    Mistral,
    Gemma,
    Phi3,
    Metharme,
    PromptCompletion,
    Text,
    Custom,
}

impl Format {
    pub const ALL: [Format; 12] = [
        Format::Chatml,
        Format::Messages,
        Format::Sharegpt,
        Format::Alpaca,
        Format::Llama3,
        Format::Mistral,
        Format::Gemma,
        Format::Phi3,
        Format::Metharme,
        Format::PromptCompletion,
        Format::Text,
        Format::Custom,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            Format::Chatml => "ChatML",
            Format::Messages => "OpenAI messages",
            Format::Sharegpt => "ShareGPT",
            Format::Alpaca => "Alpaca",
            Format::Llama3 => "Llama 3",
            Format::Mistral => "Mistral",
            Format::Gemma => "Gemma",
            Format::Phi3 => "Phi-3",
            Format::Metharme => "Metharme",
            Format::PromptCompletion => "Prompt / completion",
            Format::Text => "Raw text",
            Format::Custom => "Custom Jinja template",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportOptions {
    #[serde(default = "default_format")]
    pub format: Format,
    #[serde(default = "default_container")]
    pub container: Container,
    /// Rows per shard; `None` = single file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shard_size: Option<u64>,
    #[serde(default = "default_out_dir")]
    pub out_dir: String,
    /// Jinja chat template for `Format::Custom` (HF `chat_template` syntax).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub custom_template: Option<String>,
    #[serde(default = "default_bos")]
    pub bos_token: String,
    #[serde(default = "default_eos")]
    pub eos_token: String,
}

fn default_format() -> Format {
    Format::Chatml
}
fn default_container() -> Container {
    Container::Jsonl
}
fn default_out_dir() -> String {
    "out".into()
}
fn default_bos() -> String {
    String::new()
}
fn default_eos() -> String {
    String::new()
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            format: Format::Chatml,
            container: Container::Jsonl,
            shard_size: None,
            out_dir: default_out_dir(),
            custom_template: None,
            bos_token: default_bos(),
            eos_token: default_eos(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputMode {
    /// Replace the original text with the translation.
    Replace,
    /// Keep both: the translation is exported and the original is added to the metadata
    /// (exports `original_*` fields where the format has room for them).
    Bilingual,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OnFail {
    Drop,
    KeepOriginal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Thinking {
    /// Ask the server not to think (Qwen3 `enable_thinking=false`, `/no_think`) and still strip any reasoning.
    Off,
    /// Let the model think: extra output budget is added and reasoning is stripped from the answer.
    Allow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Structured {
    /// Try `response_format` json_schema first; remember if the server rejects it.
    Auto,
    On,
    Off,
}

/// Connection + behaviour of the translation step. The API token never lives in
/// the recipe (it is kept in the OS keychain or passed at runtime).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslateOptions {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_base_url")]
    pub base_url: String,
    #[serde(default)]
    pub model: String,
    #[serde(default = "default_source_lang")]
    pub source_language: String,
    #[serde(default = "default_target_lang")]
    pub target_language: String,
    #[serde(default = "default_concurrency")]
    pub concurrency: usize,
    /// Requests per minute; 0 = unlimited.
    #[serde(default)]
    pub requests_per_minute: u32,
    /// Tokens per minute (estimated); 0 = unlimited.
    #[serde(default)]
    pub tokens_per_minute: u32,
    #[serde(default = "default_retries")]
    pub max_retries: u32,
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    /// First retry delay; doubles on every further attempt (capped at 60 s).
    #[serde(default = "default_backoff")]
    pub retry_backoff_ms: u64,
    #[serde(default = "default_temp")]
    pub temperature: f32,
    /// Max estimated input tokens per request.
    #[serde(default = "default_ctx")]
    pub context_budget: usize,
    /// 0 = automatic (about twice the input).
    #[serde(default)]
    pub max_output_tokens: u32,
    /// Stop after this many (estimated) tokens in total; 0 = unlimited.
    #[serde(default)]
    pub token_budget: u64,
    #[serde(default = "default_roles")]
    pub roles: Vec<crate::model::Role>,
    #[serde(default = "default_true")]
    pub translate_persona: bool,
    #[serde(default)]
    pub instructions: String,
    /// "Source term = translation" lines, one per line.
    #[serde(default)]
    pub glossary: String,
    #[serde(default = "default_output_mode")]
    pub output_mode: OutputMode,
    #[serde(default = "default_on_fail")]
    pub on_fail: OnFail,
    #[serde(default = "default_structured")]
    pub structured: Structured,
    #[serde(default = "default_thinking")]
    pub thinking: Thinking,
    /// Extra output tokens per request reserved for reasoning when `thinking = allow`.
    #[serde(default = "default_thinking_tokens")]
    pub thinking_tokens: u32,
}

fn default_base_url() -> String {
    "http://127.0.0.1:8080/v1".into()
}
fn default_source_lang() -> String {
    "en".into()
}
fn default_target_lang() -> String {
    "pt-BR".into()
}
fn default_concurrency() -> usize {
    4
}
fn default_retries() -> u32 {
    5
}
fn default_timeout() -> u64 {
    600
}
fn default_backoff() -> u64 {
    1000
}
fn default_temp() -> f32 {
    0.2
}
fn default_ctx() -> usize {
    4096
}
fn default_roles() -> Vec<crate::model::Role> {
    vec![crate::model::Role::User, crate::model::Role::Assistant, crate::model::Role::System]
}
fn default_output_mode() -> OutputMode {
    OutputMode::Replace
}
fn default_on_fail() -> OnFail {
    OnFail::Drop
}
fn default_thinking() -> Thinking {
    Thinking::Off
}
fn default_thinking_tokens() -> u32 {
    3000
}
fn default_structured() -> Structured {
    Structured::Auto
}

impl Default for TranslateOptions {
    fn default() -> Self {
        Self {
            enabled: false,
            base_url: default_base_url(),
            model: String::new(),
            source_language: default_source_lang(),
            target_language: default_target_lang(),
            concurrency: default_concurrency(),
            requests_per_minute: 0,
            tokens_per_minute: 0,
            max_retries: default_retries(),
            timeout_secs: default_timeout(),
            retry_backoff_ms: default_backoff(),
            temperature: default_temp(),
            context_budget: default_ctx(),
            max_output_tokens: 0,
            token_budget: 0,
            roles: default_roles(),
            translate_persona: true,
            instructions: String::new(),
            glossary: String::new(),
            output_mode: default_output_mode(),
            on_fail: default_on_fail(),
            structured: default_structured(),
            thinking: default_thinking(),
            thinking_tokens: default_thinking_tokens(),
        }
    }
}
