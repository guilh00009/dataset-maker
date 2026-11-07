//! Normalized data model. Every input format is converted into `Sample`;
//! every exporter reads from `Sample`.

use serde::{Deserialize, Serialize};
use xxhash_rust::xxh3::xxh3_64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl Role {
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self { role, content: content.into() }
    }
}

/// Structured roleplay / character information. Kept structured until export so
/// it can be translated field by field and rendered with an editable template.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Persona {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub char_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_name: Option<String>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub definitions: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub greeting: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub scenario: String,
}

impl Persona {
    pub fn is_empty(&self) -> bool {
        self.char_name.is_none()
            && self.user_name.is_none()
            && self.description.is_empty()
            && self.definitions.is_empty()
            && self.greeting.is_empty()
            && self.scenario.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sample {
    /// xxh3 of the normalized content; used for dedup and as translation cache key.
    pub id: u64,
    /// Index of the source in the recipe.
    pub source: usize,
    /// Row number within the (materialized) source.
    pub row: u64,
    /// Explicit system prompt coming from the dataset (before persona rendering).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub messages: Vec<Message>,
    /// Raw text for pretraining style samples.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub persona: Option<Persona>,
}

impl Sample {
    pub fn is_text(&self) -> bool {
        self.text.is_some() && self.messages.is_empty()
    }

    pub fn compute_id(&self) -> u64 {
        let mut buf = String::new();
        if let Some(s) = &self.system {
            buf.push_str("S:");
            buf.push_str(s);
            buf.push('\u{1}');
        }
        if let Some(p) = &self.persona {
            if let Some(n) = &p.char_name {
                buf.push_str(n);
            }
            buf.push('\u{2}');
            buf.push_str(&p.description);
            buf.push('\u{2}');
            buf.push_str(&p.definitions);
            buf.push('\u{2}');
            buf.push_str(&p.greeting);
            buf.push('\u{2}');
            buf.push_str(&p.scenario);
            buf.push('\u{1}');
        }
        for m in &self.messages {
            buf.push_str(m.role.as_str());
            buf.push(':');
            buf.push_str(&m.content);
            buf.push('\u{1}');
        }
        if let Some(t) = &self.text {
            buf.push_str("T:");
            buf.push_str(t);
        }
        xxh3_64(buf.as_bytes())
    }

    /// Number of conversational turns (non-system messages).
    pub fn turns(&self) -> usize {
        self.messages.iter().filter(|m| m.role != Role::System).count()
    }

    pub fn char_len(&self) -> usize {
        let mut n = self.system.as_ref().map_or(0, |s| s.chars().count());
        if let Some(p) = &self.persona {
            n += p.description.chars().count()
                + p.definitions.chars().count()
                + p.greeting.chars().count()
                + p.scenario.chars().count();
        }
        n += self.messages.iter().map(|m| m.content.chars().count()).sum::<usize>();
        n += self.text.as_ref().map_or(0, |t| t.chars().count());
        n
    }

    pub fn est_tokens(&self) -> usize {
        est_tokens_from_chars(self.char_len()) + self.messages.len() * 4
    }
}

/// Cheap, tokenizer-free token estimate (works across languages well enough for
/// budgeting, filtering and ETA calculations).
pub fn est_tokens_from_chars(chars: usize) -> usize {
    ((chars as f64) / 3.6).ceil() as usize
}

pub fn est_tokens(s: &str) -> usize {
    est_tokens_from_chars(s.chars().count())
}
