//! Heuristic format auto-detection.
//!
//! Given a sample of rows (any JSON shape) this module works out which fields hold the
//! conversation / prompt-response pair / raw text / persona, scores every candidate
//! interpretation by how many rows it can actually normalize, and returns the best ones.
//! The result is a serializable [`Mapping`] that the UI can show and edit.

use crate::model::{Message, Persona, Role, Sample};
use crate::source::Prepared;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::sync::OnceLock;

// ───────────────────────────── mapping types ─────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "how", rename_all = "snake_case")]
pub enum RoleSpec {
    /// Role is a string in `key` (`user`, `human`, `gpt`, a speaker name, …).
    Key { key: String },
    /// Role is a boolean in `key`; `true` means the human/user side unless `invert`.
    Bool { key: String, #[serde(default)] invert: bool },
    /// No role information: turns alternate user, assistant, user, …
    Alternating,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Layout {
    /// Array of message objects.
    Conversation { path: String, content_key: String, role: RoleSpec },
    /// Array of plain strings, alternating user/assistant.
    StringTurns { path: String },
    /// Single prompt/response pair (prompt may be assembled from several fields).
    Pair { prompt: Vec<String>, response: String },
    /// A single text field.
    Text { path: String },
    /// A text field that already contains a chat template (ChatML, Llama, Alpaca, …).
    Preformatted { path: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Mapping {
    pub layout: Layout,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub char_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub definitions: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub greeting: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scenario: Option<String>,
    /// 0..=1, how well this mapping explains the sampled rows.
    #[serde(default)]
    pub confidence: f32,
    /// Human readable one-liner.
    #[serde(default)]
    pub summary: String,
}

impl Mapping {
    fn new(layout: Layout) -> Self {
        Self {
            layout,
            system: None,
            char_name: None,
            user_name: None,
            description: None,
            definitions: None,
            greeting: None,
            scenario: None,
            confidence: 0.0,
            summary: String::new(),
        }
    }
    pub fn has_persona(&self) -> bool {
        self.char_name.is_some()
            || self.user_name.is_some()
            || self.description.is_some()
            || self.definitions.is_some()
            || self.greeting.is_some()
            || self.scenario.is_some()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldInfo {
    pub path: String,
    /// `string`, `number`, `bool`, `object`, `array<object>`, `array<string>`, `array`
    pub kind: String,
    pub presence: f32,
    pub example: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Detection {
    pub best: Mapping,
    pub candidates: Vec<Mapping>,
    pub fields: Vec<FieldInfo>,
    pub sampled: usize,
}

// ───────────────────────────── key vocabularies ─────────────────────────────

const CONTENT_KEYS: &[&str] = &[
    "content", "value", "text", "message", "msg", "utterance", "body", "response", "mes", "line",
    "sentence", "string", "turn",
];
const ROLE_KEYS: &[&str] = &["role", "from", "speaker", "author", "sender", "name", "user", "who", "type", "actor"];
const BOOL_ROLE_KEYS: &[(&str, bool)] = &[
    ("is_human", false),
    ("is_user", false),
    ("human", false),
    ("is_bot", true),
    ("is_assistant", true),
    ("is_ai", true),
    ("is_model", true),
];
const CONV_PATH_HINTS: &[&str] =
    &["messages", "conversations", "conversation", "dialog", "dialogue", "chat", "turns", "history", "utterances", "log", "chosen"];

const PROMPT_KEYS: &[&str] = &[
    "instruction", "prompt", "question", "query", "human", "user", "problem", "source", "src", "input", "context", "inputs",
    "text_a", "request", "task", "message",
];
const RESPONSE_KEYS: &[&str] = &[
    "output", "response", "answer", "completion", "assistant", "gpt", "bot", "chosen", "target", "solution", "reply", "tgt",
    "outputs", "generated", "gold", "result",
];
const TEXT_KEYS: &[&str] = &["text", "content", "body", "document", "article", "passage", "raw", "story", "doc", "contents", "sentence"];

const SYSTEM_KEYS: &[&str] = &["system", "system_prompt", "sys_prompt", "system_message", "systemprompt", "sys"];
const CHAR_NAME_KEYS: &[&str] = &["bot_name", "char_name", "character_name", "name_of_character", "character", "char", "npc_name", "persona_name", "bot"];
const USER_NAME_KEYS: &[&str] = &["user_name", "human_name", "player_name", "username", "user_persona_name"];
const DESC_KEYS: &[&str] = &[
    "bot_description", "char_persona", "character_description", "char_description", "description", "persona", "personality",
    "profile", "bio", "about", "character_persona",
];
const DEF_KEYS: &[&str] = &["bot_definitions", "definitions", "example_dialogue", "mes_example", "dialogue_examples", "example_dialog", "examples", "example_messages"];
const GREETING_KEYS: &[&str] = &["bot_greeting", "greeting", "first_mes", "first_message", "first_msg", "intro", "char_greeting"];
const SCENARIO_KEYS: &[&str] = &["scenario", "world_scenario", "setting"];

// ───────────────────────────── helpers ─────────────────────────────

pub fn get_path<'a>(v: &'a Value, path: &str) -> Option<&'a Value> {
    let mut cur = v;
    for part in path.split('.') {
        if part.is_empty() {
            continue;
        }
        cur = match cur {
            Value::Object(m) => m.get(part)?,
            Value::Array(a) => a.get(part.parse::<usize>().ok()?)?,
            _ => return None,
        };
    }
    Some(cur)
}

/// Turn a JSON value into text: strings as-is, content-part arrays joined,
/// objects with a text-like key unwrapped, numbers stringified.
pub fn as_text(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        Value::Array(a) => {
            let parts: Vec<String> = a.iter().filter_map(as_text).filter(|s| !s.is_empty()).collect();
            if parts.is_empty() { None } else { Some(parts.join("\n")) }
        }
        Value::Object(m) => {
            for k in ["text", "content", "value", "message"] {
                if let Some(x) = m.get(k) {
                    return as_text(x);
                }
            }
            None
        }
        Value::Null => None,
    }
}

fn lower_eq(a: &str, b: &str) -> bool {
    a.eq_ignore_ascii_case(b)
}

pub fn map_role_str(s: &str) -> Option<Role> {
    match s.trim().to_lowercase().as_str() {
        "user" | "human" | "prompter" | "customer" | "client" | "question" | "person" | "you" | "input" | "usr"
        | "humans" | "users" => Some(Role::User),
        "assistant" | "gpt" | "bot" | "ai" | "model" | "chatbot" | "agent" | "character" | "char" | "response" | "output"
        | "claude" | "chatgpt" | "bing" | "gemini" | "llm" | "bots" | "assistant_response" => Some(Role::Assistant),
        "system" | "sys" | "developer" => Some(Role::System),
        "tool" | "function" | "observation" | "function_response" | "ipython" | "tool_response" | "function_call" => Some(Role::Tool),
        _ => None,
    }
}

fn walk<'a>(v: &'a Value, prefix: &str, depth: usize, out: &mut Vec<(String, &'a Value)>) {
    if let Value::Object(m) = v {
        for (k, val) in m {
            let p = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
            out.push((p.clone(), val));
            if depth < 3 && val.is_object() {
                walk(val, &p, depth + 1, out);
            }
        }
    }
}

fn kind_of(v: &Value) -> &'static str {
    match v {
        Value::String(_) => "string",
        Value::Number(_) => "number",
        Value::Bool(_) => "bool",
        Value::Object(_) => "object",
        Value::Null => "null",
        Value::Array(a) => {
            if a.iter().all(|x| x.is_object()) && !a.is_empty() {
                "array<object>"
            } else if a.iter().all(|x| x.is_string()) && !a.is_empty() {
                "array<string>"
            } else {
                "array"
            }
        }
    }
}

fn short(s: &str) -> String {
    let s = s.replace('\n', " ");
    if s.chars().count() > 80 { format!("{}…", s.chars().take(80).collect::<String>()) } else { s }
}

// ───────────────────────────── normalization ─────────────────────────────

/// Resolve an unknown speaker name to a role. Speakers equal to the character name are
/// the assistant; otherwise the first distinct speaker is the user, the second the assistant.
struct Speakers {
    map: HashMap<String, Role>,
    char_name: Option<String>,
}

impl Speakers {
    fn resolve(&mut self, name: &str) -> Role {
        if let Some(r) = map_role_str(name) {
            return r;
        }
        let key = name.trim().to_lowercase();
        if let Some(r) = self.map.get(&key) {
            return *r;
        }
        let role = if self.char_name.as_ref().is_some_and(|c| c.trim().eq_ignore_ascii_case(name.trim())) {
            Role::Assistant
        } else if !self.map.values().any(|r| *r == Role::User) {
            Role::User
        } else {
            Role::Assistant
        };
        self.map.insert(key, role);
        role
    }
}

fn opt_text(row: &Value, path: &Option<String>) -> Option<String> {
    let p = path.as_ref()?;
    let t = as_text(get_path(row, p)?)?;
    let t = t.trim().to_string();
    if t.is_empty() { None } else { Some(t) }
}

fn push_msg(out: &mut Vec<Message>, role: Role, content: &str) {
    let c = content.trim();
    if !c.is_empty() {
        out.push(Message::new(role, c));
    }
}

/// Apply a mapping to one row. Returns `None` when the row does not fit the mapping.
pub fn normalize(row: &Value, m: &Mapping, source: usize, row_idx: u64) -> Option<Sample> {
    let mut system = opt_text(row, &m.system);
    let persona = if m.has_persona() {
        let p = Persona {
            char_name: opt_text(row, &m.char_name),
            user_name: opt_text(row, &m.user_name),
            description: opt_text(row, &m.description).unwrap_or_default(),
            definitions: opt_text(row, &m.definitions).unwrap_or_default(),
            greeting: opt_text(row, &m.greeting).unwrap_or_default(),
            scenario: opt_text(row, &m.scenario).unwrap_or_default(),
        };
        if p.is_empty() { None } else { Some(p) }
    } else {
        None
    };
    let char_name = persona.as_ref().and_then(|p| p.char_name.clone());
    let mut messages: Vec<Message> = Vec::new();
    let mut text: Option<String> = None;

    match &m.layout {
        Layout::Conversation { path, content_key, role } => {
            let arr = get_path(row, path)?.as_array()?;
            let mut speakers = Speakers { map: HashMap::new(), char_name: char_name.clone() };
            for (i, el) in arr.iter().enumerate() {
                match el {
                    Value::Object(o) => {
                        let content = o
                            .get(content_key)
                            .and_then(as_text)
                            .or_else(|| CONTENT_KEYS.iter().find_map(|k| o.get(*k).and_then(as_text)));
                        let Some(content) = content else { continue };
                        let r = match role {
                            RoleSpec::Key { key } => match o.get(key) {
                                Some(Value::String(s)) => speakers.resolve(s),
                                Some(other) => match as_text(other) {
                                    Some(s) => speakers.resolve(&s),
                                    None => alt_role(i),
                                },
                                None => alt_role(i),
                            },
                            RoleSpec::Bool { key, invert } => {
                                let b = match o.get(key) {
                                    Some(Value::Bool(b)) => *b,
                                    Some(Value::String(s)) => s.eq_ignore_ascii_case("true"),
                                    Some(Value::Number(n)) => n.as_i64() == Some(1),
                                    _ => false,
                                };
                                if b != *invert { Role::User } else { Role::Assistant }
                            }
                            RoleSpec::Alternating => alt_role(i),
                        };
                        push_msg(&mut messages, r, &content);
                    }
                    Value::String(s) => push_msg(&mut messages, alt_role(i), s),
                    _ => {}
                }
            }
        }
        Layout::StringTurns { path } => {
            let arr = get_path(row, path)?.as_array()?;
            for (i, el) in arr.iter().enumerate() {
                if let Some(s) = as_text(el) {
                    push_msg(&mut messages, alt_role(i), &s);
                }
            }
        }
        Layout::Pair { prompt, response } => {
            let mut parts = Vec::new();
            for p in prompt {
                if let Some(t) = get_path(row, p).and_then(as_text) {
                    let t = t.trim();
                    if !t.is_empty() {
                        parts.push(t.to_string());
                    }
                }
            }
            let resp = get_path(row, response).and_then(as_text)?;
            if parts.is_empty() {
                return None;
            }
            push_msg(&mut messages, Role::User, &parts.join("\n\n"));
            push_msg(&mut messages, Role::Assistant, &resp);
        }
        Layout::Text { path } => {
            let t = get_path(row, path).and_then(as_text)?;
            let t = t.trim();
            if t.is_empty() {
                return None;
            }
            text = Some(t.to_string());
        }
        Layout::Preformatted { path } => {
            let t = get_path(row, path).and_then(as_text)?;
            match parse_chat_text(&t) {
                Some((sys, msgs)) => {
                    if system.is_none() {
                        system = sys;
                    }
                    messages = msgs;
                }
                None => {
                    let t = t.trim();
                    if t.is_empty() {
                        return None;
                    }
                    text = Some(t.to_string());
                }
            }
        }
    }

    // A greeting that is not already the first assistant turn is prepended.
    if let Some(p) = &persona {
        let g = p.greeting.trim();
        if !g.is_empty() && !messages.is_empty() {
            let first = &messages[0];
            let already = first.role == Role::Assistant;
            if !already && first.role == Role::User {
                messages.insert(0, Message::new(Role::Assistant, g));
            }
        }
    }

    if messages.is_empty() && text.is_none() {
        return None;
    }
    let mut s = Sample { id: 0, source, row: row_idx, system, messages, text, persona };
    s.id = s.compute_id();
    Some(s)
}

fn alt_role(i: usize) -> Role {
    if i % 2 == 0 { Role::User } else { Role::Assistant }
}

// ───────────────────────────── pre-formatted chat text ─────────────────────────────

fn rx(cell: &'static OnceLock<Regex>, pat: &str) -> &'static Regex {
    cell.get_or_init(|| Regex::new(pat).expect("valid regex"))
}

fn strip_end_tokens(s: &str) -> &str {
    let mut s = s.trim();
    for t in ["<|im_end|>", "<|eot_id|>", "<end_of_turn>", "<|end|>", "</s>", "<|endoftext|>", "<eos>", "<|end_of_text|>"] {
        if let Some(x) = s.strip_suffix(t) {
            s = x.trim();
        }
    }
    s.trim_start_matches("<s>").trim_start_matches("<bos>").trim()
}

fn tag_split(text: &str, re: &Regex) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let matches: Vec<_> = re.captures_iter(text).collect();
    for (i, c) in matches.iter().enumerate() {
        let whole = c.get(0).unwrap();
        let end = matches.get(i + 1).map_or(text.len(), |n| n.get(0).unwrap().start());
        let role = c.get(1).map_or("", |m| m.as_str()).to_string();
        out.push((role, strip_end_tokens(&text[whole.end()..end]).to_string()));
    }
    out
}

fn build_msgs(parts: Vec<(String, String)>) -> Option<(Option<String>, Vec<Message>)> {
    let mut system = None;
    let mut msgs = Vec::new();
    for (r, c) in parts {
        if c.is_empty() {
            continue;
        }
        let role = map_role_str(&r)?;
        if role == Role::System {
            system = Some(c);
        } else {
            msgs.push(Message::new(role, c));
        }
    }
    let has_user = msgs.iter().any(|m| m.role == Role::User);
    let has_asst = msgs.iter().any(|m| m.role == Role::Assistant);
    if msgs.len() >= 2 && has_user && has_asst { Some((system, msgs)) } else { None }
}

/// Try to parse text that already contains a chat template back into turns.
pub fn parse_chat_text(text: &str) -> Option<(Option<String>, Vec<Message>)> {
    static CHATML: OnceLock<Regex> = OnceLock::new();
    static LLAMA3: OnceLock<Regex> = OnceLock::new();
    static GEMMA: OnceLock<Regex> = OnceLock::new();
    static PHI: OnceLock<Regex> = OnceLock::new();
    static ALPACA: OnceLock<Regex> = OnceLock::new();
    static ROLES: OnceLock<Regex> = OnceLock::new();

    if text.contains("<|im_start|>") {
        let re = rx(&CHATML, r"<\|im_start\|>\s*([A-Za-z_]+)[ \t]*\n?");
        return build_msgs(tag_split(text, re));
    }
    if text.contains("<|start_header_id|>") {
        let re = rx(&LLAMA3, r"<\|start_header_id\|>\s*([A-Za-z_]+)\s*<\|end_header_id\|>\s*");
        return build_msgs(tag_split(text, re));
    }
    if text.contains("<start_of_turn>") {
        let re = rx(&GEMMA, r"<start_of_turn>\s*([A-Za-z_]+)[ \t]*\n?");
        return build_msgs(tag_split(text, re));
    }
    if text.contains("[INST]") {
        let mut system = None;
        let mut msgs = Vec::new();
        for seg in text.split("[INST]").skip(1) {
            let (user, asst) = seg.split_once("[/INST]")?;
            let mut user = user.trim().to_string();
            if let (Some(a), Some(b)) = (user.find("<<SYS>>"), user.find("<</SYS>>")) {
                if a < b {
                    system = Some(user[a + 7..b].trim().to_string());
                    user = user[b + 8..].trim().to_string();
                }
            }
            let user = strip_end_tokens(&user).to_string();
            let asst = strip_end_tokens(asst).to_string();
            if !user.is_empty() {
                msgs.push(Message::new(Role::User, user));
            }
            if !asst.is_empty() {
                msgs.push(Message::new(Role::Assistant, asst));
            }
        }
        return if msgs.len() >= 2 { Some((system, msgs)) } else { None };
    }
    if text.contains("<|user|>") || text.contains("<|model|>") || text.contains("<|assistant|>") {
        let re = rx(&PHI, r"<\|(system|user|assistant|model)\|>\s*");
        return build_msgs(tag_split(text, re));
    }
    if text.contains("### Instruction:") && text.contains("### Response:") {
        let re = rx(&ALPACA, r"(?s)###\s*Instruction:\s*(.*?)\s*(?:###\s*Input:\s*(.*?)\s*)?###\s*Response:\s*(.*)$");
        let c = re.captures(text)?;
        let mut prompt = c.get(1)?.as_str().trim().to_string();
        if let Some(i) = c.get(2) {
            prompt.push_str("\n\n");
            prompt.push_str(i.as_str().trim());
        }
        let resp = strip_end_tokens(c.get(3)?.as_str()).to_string();
        if prompt.is_empty() || resp.is_empty() {
            return None;
        }
        return Some((None, vec![Message::new(Role::User, prompt.trim()), Message::new(Role::Assistant, resp)]));
    }
    let re = rx(&ROLES, r"(?m)^(?:###\s*)?(Human|HUMAN|User|USER|Assistant|ASSISTANT|System|SYSTEM|GPT|Bot|AI)[ \t]*:[ \t]*");
    if re.find_iter(text).count() >= 2 {
        return build_msgs(tag_split(text, re));
    }
    None
}

// ───────────────────────────── detection ─────────────────────────────

#[derive(Default)]
struct PathStat {
    count: usize,
    kinds: BTreeMap<&'static str, usize>,
    example: String,
    total_len: usize,
}

#[derive(Default)]
struct ArrStat {
    rows: usize,
    elems: usize,
    key_counts: BTreeMap<String, usize>,
    key_string: BTreeMap<String, usize>,
    key_bool: BTreeMap<String, usize>,
    str_values: BTreeMap<String, BTreeMap<String, usize>>,
}

/// Evenly spaced sample of up to `n` parsed rows.
pub fn sample_rows(p: &Prepared, n: usize) -> Vec<(usize, Value)> {
    let len = p.len();
    if len == 0 {
        return vec![];
    }
    let n = n.min(len);
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        let i = if n == 1 { 0 } else { k * (len - 1) / (n - 1) };
        if let Ok(v) = p.row_value(i) {
            out.push((i, v));
        }
    }
    out
}

pub fn detect_prepared(p: &Prepared) -> Detection {
    let rows = sample_rows(p, 200);
    detect_rows(&rows.iter().map(|(_, v)| v.clone()).collect::<Vec<_>>())
}

pub fn detect_rows(rows: &[Value]) -> Detection {
    let n = rows.len().max(1);
    // 1. gather per-path statistics
    let mut stats: BTreeMap<String, PathStat> = BTreeMap::new();
    let mut arrs: BTreeMap<String, ArrStat> = BTreeMap::new();
    for row in rows {
        let mut paths = Vec::new();
        walk(row, "", 0, &mut paths);
        for (p, v) in paths {
            let st = stats.entry(p.clone()).or_default();
            st.count += 1;
            *st.kinds.entry(kind_of(v)).or_default() += 1;
            if st.example.is_empty() {
                st.example = match v {
                    Value::String(s) => short(s),
                    Value::Array(a) => a.first().map(|x| short(&x.to_string())).unwrap_or_default(),
                    other => short(&other.to_string()),
                };
            }
            if let Value::String(s) = v {
                st.total_len += s.chars().count();
            }
            if let Value::Array(a) = v {
                let objs: Vec<&serde_json::Map<String, Value>> = a.iter().filter_map(|x| x.as_object()).collect();
                if !objs.is_empty() {
                    let ast = arrs.entry(p.clone()).or_default();
                    ast.rows += 1;
                    for o in objs {
                        ast.elems += 1;
                        for (k, x) in o {
                            *ast.key_counts.entry(k.clone()).or_default() += 1;
                            match x {
                                Value::String(s) => {
                                    *ast.key_string.entry(k.clone()).or_default() += 1;
                                    let vals = ast.str_values.entry(k.clone()).or_default();
                                    if vals.len() < 64 {
                                        *vals.entry(s.clone()).or_default() += 1;
                                    }
                                }
                                Value::Array(_) | Value::Object(_) => {
                                    if as_text(x).is_some() {
                                        *ast.key_string.entry(k.clone()).or_default() += 1;
                                    }
                                }
                                Value::Bool(_) => *ast.key_bool.entry(k.clone()).or_default() += 1,
                                _ => {}
                            }
                        }
                    }
                }
            }
        }
    }

    let fields: Vec<FieldInfo> = stats
        .iter()
        .map(|(p, s)| {
            let kind = s.kinds.iter().max_by_key(|(_, c)| **c).map(|(k, _)| *k).unwrap_or("null");
            FieldInfo { path: p.clone(), kind: kind.to_string(), presence: s.count as f32 / n as f32, example: s.example.clone() }
        })
        .collect();

    let string_paths: Vec<&String> = stats
        .iter()
        .filter(|(_, s)| s.kinds.get("string").copied().unwrap_or(0) * 2 >= s.count && s.count * 10 >= n)
        .map(|(p, _)| p)
        .collect();
    let leaf = |p: &str| p.rsplit('.').next().unwrap_or(p).to_string();

    // 2. persona / system fields (shared by every layout)
    let used_by_layout = |l: &Layout| -> Vec<String> {
        match l {
            Layout::Pair { prompt, response } => prompt.iter().cloned().chain(std::iter::once(response.clone())).collect(),
            Layout::Text { path } | Layout::Preformatted { path } | Layout::StringTurns { path } => vec![path.clone()],
            Layout::Conversation { path, .. } => vec![path.clone()],
        }
    };
    let pick = |vocab: &[&str], used: &[String], need_presence: f32| -> Option<String> {
        for want in vocab {
            if let Some(p) = string_paths.iter().find(|p| lower_eq(&leaf(p), want) && !used.contains(p)) {
                if stats[*p].count as f32 / n as f32 >= need_presence {
                    return Some((*p).clone());
                }
            }
        }
        None
    };
    let attach_persona = |m: &mut Mapping| {
        let used = used_by_layout(&m.layout);
        m.system = pick(SYSTEM_KEYS, &used, 0.1);
        let mut u = used.clone();
        u.extend(m.system.clone());
        m.char_name = pick(CHAR_NAME_KEYS, &u, 0.3);
        m.greeting = pick(GREETING_KEYS, &u, 0.3);
        m.definitions = pick(DEF_KEYS, &u, 0.3);
        // `description` / `scenario` etc. are too generic to trust on their own: only treat them as
        // persona information when a character name, greeting or definitions field was also found.
        if m.char_name.is_some() || m.greeting.is_some() || m.definitions.is_some() {
            m.description = pick(DESC_KEYS, &u, 0.3);
            m.scenario = pick(SCENARIO_KEYS, &u, 0.3);
            m.user_name = pick(USER_NAME_KEYS, &u, 0.3);
        }
        if m.char_name.as_deref().is_some_and(|p| p == m.system.clone().unwrap_or_default()) {
            m.char_name = None;
        }
    };

    // 3. candidate layouts
    let mut candidates: Vec<(Mapping, f32, String)> = Vec::new(); // mapping, quality, description

    // 3a. conversations
    for (path, a) in &arrs {
        if a.rows * 4 < n || a.elems == 0 {
            continue;
        }
        let tot = a.elems as f32;
        let content = CONTENT_KEYS
            .iter()
            .find_map(|c| a.key_string.iter().find(|(k, cnt)| lower_eq(k, c) && **cnt as f32 / tot >= 0.8).map(|(k, _)| k.clone()));
        let Some(content_key) = content else { continue };
        // role: boolean key, string key with known values, or alternating
        let mut role = RoleSpec::Alternating;
        let mut quality = 0.8f32;
        'bool: for (bk, invert) in BOOL_ROLE_KEYS {
            if let Some((k, cnt)) = a.key_bool.iter().find(|(k, _)| lower_eq(k, bk)) {
                if *cnt as f32 / tot >= 0.8 {
                    role = RoleSpec::Bool { key: k.clone(), invert: *invert };
                    quality = 1.0;
                    break 'bool;
                }
            }
        }
        if role == RoleSpec::Alternating {
            for rk in ROLE_KEYS {
                if let Some((k, cnt)) = a.key_string.iter().find(|(k, _)| lower_eq(k, rk) && **k != content_key) {
                    if (*cnt as f32 / tot) < 0.8 {
                        continue;
                    }
                    let vals = &a.str_values[k];
                    let total: usize = vals.values().sum();
                    let known: usize = vals.iter().filter(|(v, _)| map_role_str(v).is_some()).map(|(_, c)| *c).sum();
                    let distinct = vals.len();
                    if known as f32 / total.max(1) as f32 >= 0.9 {
                        role = RoleSpec::Key { key: k.clone() };
                        quality = 1.0;
                        break;
                    } else if distinct <= 8 {
                        // speaker names
                        role = RoleSpec::Key { key: k.clone() };
                        quality = 0.85;
                        break;
                    }
                }
            }
        }
        let mean_len = a.elems as f32 / a.rows as f32;
        if mean_len < 1.5 {
            quality *= 0.5;
        }
        let name_bonus = CONV_PATH_HINTS.iter().any(|h| lower_eq(&leaf(path), h));
        let mut q = quality * if name_bonus { 1.0 } else { 0.96 };
        if lower_eq(&leaf(path), "rejected") {
            q *= 0.8;
        }
        let mut m = Mapping::new(Layout::Conversation { path: path.clone(), content_key: content_key.clone(), role: role.clone() });
        attach_persona(&mut m);
        let role_desc = match &role {
            RoleSpec::Key { key } => format!("role from `{key}`"),
            RoleSpec::Bool { key, .. } => format!("role from boolean `{key}`"),
            RoleSpec::Alternating => "alternating roles".to_string(),
        };
        candidates.push((m, q, format!("Conversation · `{path}[].{content_key}` · {role_desc}")));
    }

    // 3b. arrays of strings
    for (path, s) in &stats {
        let arr_str = s.kinds.get("array<string>").copied().unwrap_or(0);
        if arr_str * 4 >= n && arr_str * 2 >= s.count {
            let mut m = Mapping::new(Layout::StringTurns { path: path.clone() });
            attach_persona(&mut m);
            candidates.push((m, 0.8, format!("Alternating turns · `{path}[]`")));
        }
    }

    // 3c. prompt / response pair
    {
        let leafs: Vec<(&String, String)> = string_paths.iter().map(|p| (*p, leaf(p))).collect();
        let find = |vocab: &[&str], exclude: &[String]| -> Option<String> {
            for want in vocab {
                if let Some((p, _)) = leafs.iter().find(|(p, l)| lower_eq(l, want) && !exclude.contains(*p)) {
                    return Some((*p).clone());
                }
            }
            None
        };
        if let Some(resp) = find(RESPONSE_KEYS, &[]) {
            let mut prompt: Vec<String> = Vec::new();
            if let Some(first) = find(PROMPT_KEYS, &[resp.clone()]) {
                prompt.push(first.clone());
                // Alpaca style: instruction + input (+ context) are folded into one prompt.
                if lower_eq(&leaf(&first), "instruction") || lower_eq(&leaf(&first), "question") {
                    for extra in ["input", "context"] {
                        if let Some((p, _)) = leafs.iter().find(|(p, l)| {
                            lower_eq(l, extra) && !prompt.contains(*p) && **p != resp && stats[*p].total_len > 0
                        }) {
                            prompt.push((*p).clone());
                        }
                    }
                }
                let mut m = Mapping::new(Layout::Pair { prompt: prompt.clone(), response: resp.clone() });
                attach_persona(&mut m);
                let exact = RESPONSE_KEYS[..6].iter().any(|k| lower_eq(&leaf(&resp), k));
                candidates.push((
                    m,
                    if exact { 0.92 } else { 0.84 },
                    format!("Prompt/response · {} → `{resp}`", prompt.iter().map(|p| format!("`{p}`")).collect::<Vec<_>>().join(" + ")),
                ));
            }
        }
    }

    // 3d. text fields (pre-formatted or raw)
    {
        let mut best: Option<(&String, f32)> = None;
        for p in &string_paths {
            let l = leaf(p);
            let avg = stats[*p].total_len as f32 / stats[*p].count.max(1) as f32;
            let mut score = avg.min(4000.0) / 4000.0;
            if TEXT_KEYS.iter().any(|k| lower_eq(&l, k)) {
                score += 1.0;
            }
            if best.map_or(true, |(_, s)| score > s) {
                best = Some((p, score));
            }
        }
        if let Some((p, _)) = best {
            let ok = rows.iter().filter(|r| get_path(r, p).and_then(as_text).is_some_and(|t| !t.trim().is_empty())).count();
            if ok * 2 >= n {
                let parsed = rows
                    .iter()
                    .filter(|r| get_path(r, p).and_then(as_text).is_some_and(|t| parse_chat_text(&t).is_some()))
                    .count();
                if parsed * 2 >= ok && parsed > 0 {
                    let mut m = Mapping::new(Layout::Preformatted { path: (*p).clone() });
                    attach_persona(&mut m);
                    candidates.push((m, 0.95, format!("Pre-formatted chat text in `{p}` (parsed back into turns)")));
                } else {
                    let mut m = Mapping::new(Layout::Text { path: (*p).clone() });
                    attach_persona(&mut m);
                    candidates.push((m, 0.7, format!("Raw text · `{p}`")));
                }
            }
        }
    }

    // 4. score every candidate by actually normalizing the sampled rows
    let mut scored: Vec<Mapping> = Vec::new();
    for (mut m, quality, desc) in candidates {
        let ok = rows.iter().enumerate().filter(|(i, r)| normalize(r, &m, 0, *i as u64).is_some()).count();
        let coverage = ok as f32 / n as f32;
        m.confidence = (coverage * quality).clamp(0.0, 1.0);
        let persona_desc = if m.has_persona() {
            let mut f = Vec::new();
            if let Some(x) = &m.char_name {
                f.push(format!("name `{x}`"));
            }
            if let Some(x) = &m.description {
                f.push(format!("description `{x}`"));
            }
            if let Some(x) = &m.definitions {
                f.push(format!("definitions `{x}`"));
            }
            if let Some(x) = &m.greeting {
                f.push(format!("greeting `{x}`"));
            }
            if let Some(x) = &m.scenario {
                f.push(format!("scenario `{x}`"));
            }
            format!(" · persona ({})", f.join(", "))
        } else {
            String::new()
        };
        let sys_desc = m.system.as_ref().map(|s| format!(" · system `{s}`")).unwrap_or_default();
        m.summary = format!("{desc}{sys_desc}{persona_desc}");
        scored.push(m);
    }
    scored.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
    scored.dedup_by(|a, b| a.layout == b.layout);

    let best = scored.first().cloned().unwrap_or_else(|| {
        // Nothing recognisable: fall back to the longest string field (or the whole row as JSON text).
        let path = string_paths.iter().max_by_key(|p| stats[**p].total_len).map(|p| (*p).clone()).unwrap_or_else(|| "text".into());
        let mut m = Mapping::new(Layout::Text { path });
        m.summary = "No known structure recognised; using the longest text field".into();
        m
    });
    Detection { best, candidates: scored.into_iter().take(6).collect(), fields, sampled: rows.len() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn best(rows: Vec<Value>) -> (Mapping, Vec<Sample>) {
        let d = detect_rows(&rows);
        let samples = rows.iter().enumerate().filter_map(|(i, r)| normalize(r, &d.best, 0, i as u64)).collect();
        (d.best, samples)
    }

    fn roles(s: &Sample) -> Vec<&'static str> {
        s.messages.iter().map(|m| m.role.as_str()).collect()
    }

    #[test]
    fn openai_messages() {
        let rows = (0..10)
            .map(|i| json!({"messages":[{"role":"system","content":"be nice"},{"role":"user","content":format!("q{i}")},{"role":"assistant","content":"a"}]}))
            .collect();
        let (m, s) = best(rows);
        assert!(m.confidence > 0.9, "{m:?}");
        assert_eq!(roles(&s[0]), ["system", "user", "assistant"]);
    }

    #[test]
    fn sharegpt() {
        let rows = (0..10)
            .map(|i| json!({"id":i,"conversations":[{"from":"human","value":"hi"},{"from":"gpt","value":"hello"}]}))
            .collect();
        let (m, s) = best(rows);
        assert!(m.confidence > 0.9);
        assert_eq!(roles(&s[0]), ["user", "assistant"]);
    }

    #[test]
    fn pippa_persona() {
        let rows: Vec<Value> = (0..10)
            .map(|i| {
                json!({
                    "submission_timestamp": 1690000000000i64 + i, "categories": null, "bot_id": "x",
                    "bot_name": "Rose", "bot_greeting": "Hello there!", "bot_definitions": "{{char}}: hi\n{{user}}: yo",
                    "bot_description": "A florist.",
                    "conversation": [
                        {"message":"Hello there!","is_human":false},
                        {"message":"hi Rose","is_human":true},
                        {"message":"hello!","is_human":false}
                    ]
                })
            })
            .collect();
        let (m, s) = best(rows);
        assert!(m.confidence > 0.9, "{m:?}");
        assert_eq!(m.char_name.as_deref(), Some("bot_name"));
        assert_eq!(roles(&s[0]), ["assistant", "user", "assistant"]);
        let p = s[0].persona.as_ref().unwrap();
        assert_eq!(p.char_name.as_deref(), Some("Rose"));
        assert_eq!(p.description, "A florist.");
    }

    #[test]
    fn greeting_prepended_when_missing() {
        let rows: Vec<Value> = (0..5)
            .map(|_| json!({"char_name":"Bob","greeting":"Hey","conversation":[{"speaker":"user","text":"yo"},{"speaker":"bob","text":"sup"}]}))
            .collect();
        let (_, s) = best(rows);
        assert_eq!(roles(&s[0]), ["assistant", "user", "assistant"]);
    }

    #[test]
    fn alpaca_and_dolly() {
        let rows: Vec<Value> = (0..10).map(|i| json!({"instruction":format!("do {i}"),"input":"ctx","output":"done"})).collect();
        let (m, s) = best(rows);
        assert!(matches!(m.layout, Layout::Pair { .. }));
        assert!(s[0].messages[0].content.contains("ctx"));
        let rows: Vec<Value> = (0..10).map(|_| json!({"instruction":"i","context":"c","response":"r","category":"x"})).collect();
        let (m, s) = best(rows);
        assert!(matches!(m.layout, Layout::Pair { .. }), "{m:?}");
        assert_eq!(roles(&s[0]), ["user", "assistant"]);
    }

    #[test]
    fn qa_columns_and_text() {
        let rows: Vec<Value> = (0..10).map(|_| json!({"question":"why?","answer":"because"})).collect();
        assert!(matches!(best(rows).0.layout, Layout::Pair { .. }));
        let rows: Vec<Value> = (0..10).map(|i| json!({"text":format!("some long document number {i} with words"),"label":1})).collect();
        let (m, s) = best(rows);
        assert!(matches!(m.layout, Layout::Text { .. }));
        assert!(s[0].is_text());
    }

    #[test]
    fn dpo_chosen_strings_and_conversations() {
        let rows: Vec<Value> = (0..10).map(|_| json!({"prompt":"p","chosen":"good","rejected":"bad"})).collect();
        let (m, s) = best(rows);
        match &m.layout {
            Layout::Pair { response, .. } => assert_eq!(response, "chosen"),
            other => panic!("{other:?}"),
        }
        assert_eq!(s[0].messages[1].content, "good");
        let rows: Vec<Value> = (0..10)
            .map(|_| json!({"chosen":[{"role":"user","content":"p"},{"role":"assistant","content":"good"}],"rejected":[{"role":"user","content":"p"},{"role":"assistant","content":"bad"}]}))
            .collect();
        let (m, s) = best(rows);
        match &m.layout {
            Layout::Conversation { path, .. } => assert_eq!(path, "chosen"),
            other => panic!("{other:?}"),
        }
        assert_eq!(s[0].messages[1].content, "good");
    }

    #[test]
    fn nested_and_string_arrays() {
        let rows: Vec<Value> = (0..10)
            .map(|_| json!({"data":{"dialog":[{"speaker":"A","utterance":"hi"},{"speaker":"B","utterance":"yo"},{"speaker":"A","utterance":"bye"}]}}))
            .collect();
        let (m, s) = best(rows);
        assert!(m.confidence > 0.7, "{m:?}");
        assert_eq!(roles(&s[0]), ["user", "assistant", "user"]);
        let rows: Vec<Value> = (0..10).map(|_| json!({"dialog":["hello","hi there","how are you","fine"]})).collect();
        let (m, s) = best(rows);
        assert!(matches!(m.layout, Layout::StringTurns { .. }));
        assert_eq!(roles(&s[0]), ["user", "assistant", "user", "assistant"]);
    }

    #[test]
    fn preformatted_chatml_llama_alpaca() {
        let chatml = "<|im_start|>system\nbe brief<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nhello<|im_end|>";
        let rows: Vec<Value> = (0..5).map(|_| json!({"text": chatml})).collect();
        let (m, s) = best(rows);
        assert!(matches!(m.layout, Layout::Preformatted { .. }), "{m:?}");
        assert_eq!(s[0].system.as_deref(), Some("be brief"));
        assert_eq!(roles(&s[0]), ["user", "assistant"]);

        let alpaca = "Below is an instruction.\n\n### Instruction:\nSay hi\n\n### Response:\nHi!";
        let (_, msgs) = parse_chat_text(alpaca).unwrap();
        assert_eq!(msgs[0].content, "Say hi");
        let mistral = "<s>[INST] hello [/INST] hi there</s><s>[INST] more [/INST] sure</s>";
        let (_, msgs) = parse_chat_text(mistral).unwrap();
        assert_eq!(msgs.len(), 4);
        let vicuna = "USER: hi\nASSISTANT: hello\nUSER: bye\nASSISTANT: ciao";
        assert_eq!(parse_chat_text(vicuna).unwrap().1.len(), 4);
        let llama3 = "<|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nhello<|eot_id|>";
        assert_eq!(parse_chat_text(llama3).unwrap().1.len(), 2);
        let gemma = "<start_of_turn>user\nhi<end_of_turn>\n<start_of_turn>model\nhello<end_of_turn>";
        assert_eq!(parse_chat_text(gemma).unwrap().1[1].role, Role::Assistant);
        let metharme = "<|system|>You are Bob<|user|>hi<|model|>hello";
        let (sys, msgs) = parse_chat_text(metharme).unwrap();
        assert_eq!(sys.as_deref(), Some("You are Bob"));
        assert_eq!(msgs.len(), 2);
    }

    #[test]
    fn content_parts_and_unknown_speakers() {
        let rows: Vec<Value> = (0..5)
            .map(|_| json!({"messages":[{"role":"user","content":[{"type":"text","text":"look"}]},{"role":"assistant","content":"ok"}]}))
            .collect();
        let (_, s) = best(rows);
        assert_eq!(s[0].messages[0].content, "look");
        let rows: Vec<Value> = (0..5)
            .map(|_| json!({"chat":[{"name":"Alice","message":"hi"},{"name":"Bob","message":"yo"},{"name":"Alice","message":"k"}]}))
            .collect();
        let (_, s) = best(rows);
        assert_eq!(roles(&s[0]), ["user", "assistant", "user"]);
    }

    #[test]
    fn unrecognised_falls_back_to_text() {
        let rows: Vec<Value> = (0..5).map(|i| json!({"foo": format!("some words {i}"), "n": i})).collect();
        let d = detect_rows(&rows);
        assert!(matches!(d.best.layout, Layout::Text { .. }));
    }
}
