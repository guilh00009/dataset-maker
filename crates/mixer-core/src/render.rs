//! Turning a normalized `Sample` into its final conversational form:
//! persona → system prompt (via an editable template), `{{char}}`/`{{user}}` → real names,
//! and optional windowing of very long conversations.

use crate::model::{Message, Role, Sample, est_tokens};
use crate::recipe::PersonaOptions;
use regex::Regex;
use std::sync::OnceLock;

fn char_rx() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(r"(?i)\{\{\s*(?:char|bot|character|npc|assistant)\s*\}\}|<BOT>|\{char\}|%char%").unwrap()
    })
}
fn user_rx() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"(?i)\{\{\s*(?:user|human|player|you)\s*\}\}|<USER>|\{user\}|%user%").unwrap())
}

/// True when text contains any `{{char}}` / `{{user}}` style placeholder.
pub fn has_placeholders(text: &str) -> bool {
    char_rx().is_match(text) || user_rx().is_match(text)
}

/// Replace placeholder variants with real names.
pub fn substitute(text: &str, char_name: &str, user_name: &str) -> String {
    if !text.contains('{') && !text.contains('<') && !text.contains('%') {
        return text.to_string();
    }
    let t = char_rx().replace_all(text, regex::NoExpand(char_name));
    user_rx().replace_all(&t, regex::NoExpand(user_name)).into_owned()
}

/// Names that apply to this sample: (character, user).
pub fn resolve_names(sample: &Sample, opts: &PersonaOptions) -> (String, String) {
    let c = sample
        .persona
        .as_ref()
        .and_then(|p| p.char_name.clone())
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| opts.fallback_char_name.clone());
    let u = sample
        .persona
        .as_ref()
        .and_then(|p| p.user_name.clone())
        .filter(|s| !s.trim().is_empty())
        .or_else(|| {
            if opts.user_name_pool.is_empty() {
                None
            } else {
                Some(opts.user_name_pool[(sample.id % opts.user_name_pool.len() as u64) as usize].clone())
            }
        })
        .unwrap_or_else(|| opts.fallback_user_name.clone());
    (c, u)
}

/// Character.AI style definitions contain `END_OF_DIALOG` markers between examples.
pub fn clean_definitions(s: &str) -> String {
    s.lines()
        .map(|l| if l.trim() == "END_OF_DIALOG" { "" } else { l })
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

fn var_rx() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"\{\{\s*([a-z_]+)\s*\}\}").unwrap())
}

/// Render the system prompt from the template. Lines whose variables are all empty are dropped.
pub fn render_system(sample: &Sample, opts: &PersonaOptions, char_name: &str, user_name: &str) -> Option<String> {
    let has_persona = sample.persona.as_ref().is_some_and(|p| !p.is_empty());
    if !has_persona {
        return sample.system.clone().filter(|s| !s.trim().is_empty());
    }
    let p = sample.persona.as_ref().unwrap();
    let get = |name: &str| -> String {
        match name {
            "char_name" => p.char_name.clone().unwrap_or_default(),
            "user_name" => p.user_name.clone().unwrap_or_default(),
            "description" => p.description.clone(),
            "definitions" => clean_definitions(&p.definitions),
            "scenario" => p.scenario.clone(),
            "greeting" => p.greeting.clone(),
            "system" => sample.system.clone().unwrap_or_default(),
            _ => String::new(),
        }
    };
    // `char_name`/`user_name` fall back to the configured names so "You are X." never vanishes
    // for datasets whose character name is simply unknown (the line is only dropped if the
    // fallback is empty too).
    let mut out: Vec<String> = Vec::new();
    for line in opts.template.lines() {
        let mut any_var = false;
        let mut any_filled = false;
        let rendered = var_rx().replace_all(line, |c: &regex::Captures| {
            any_var = true;
            let mut v = get(&c[1]);
            if v.trim().is_empty() && &c[1] == "char_name" && p.char_name.is_some() {
                v = char_name.to_string();
            }
            if !v.trim().is_empty() {
                any_filled = true;
            }
            v
        });
        if any_var && !any_filled {
            continue;
        }
        out.push(rendered.into_owned());
    }
    let joined = out.join("\n");
    // collapse 3+ newlines
    static NL: OnceLock<Regex> = OnceLock::new();
    let nl = NL.get_or_init(|| Regex::new(r"\n{3,}").unwrap());
    let text = nl.replace_all(&joined, "\n\n").trim().to_string();
    let _ = user_name;
    if text.is_empty() { None } else { Some(text) }
}

/// Final form of a sample, ready for formatting: persona folded into a leading system
/// message and placeholders replaced by real names.
pub fn finalize(sample: &Sample, opts: &PersonaOptions) -> Sample {
    let (c, u) = resolve_names(sample, opts);
    let sys = render_system(sample, opts, &c, &u);
    let sub = |t: &str| if opts.substitute_names { substitute(t, &c, &u) } else { t.to_string() };

    let mut messages: Vec<Message> = Vec::with_capacity(sample.messages.len() + 1);
    if let Some(s) = sys {
        let s = sub(&s);
        if !s.trim().is_empty() {
            messages.push(Message::new(Role::System, s));
        }
    }
    for m in &sample.messages {
        if m.role == Role::System && messages.first().is_some_and(|f| f.role == Role::System) && messages[0].content == m.content {
            continue;
        }
        messages.push(Message::new(m.role, sub(&m.content)));
    }
    Sample {
        id: sample.id,
        source: sample.source,
        row: sample.row,
        system: None,
        messages,
        text: sample.text.as_ref().map(|t| sub(t)),
        persona: None,
    }
}

/// Split a long conversation into windows of at most `max_tokens` (estimated), always
/// repeating the leading system message. Windows start on a user turn when possible.
pub fn window(sample: &Sample, max_tokens: usize) -> Vec<Sample> {
    if sample.messages.is_empty() || sample.est_tokens() <= max_tokens {
        return vec![sample.clone()];
    }
    let (sys, rest): (Option<&Message>, &[Message]) = match sample.messages.first() {
        Some(m) if m.role == Role::System => (Some(m), &sample.messages[1..]),
        _ => (None, &sample.messages[..]),
    };
    let base = sys.map_or(0, |m| est_tokens(&m.content) + 4);
    let budget = max_tokens.saturating_sub(base).max(32);
    let mut windows: Vec<Vec<Message>> = Vec::new();
    let mut cur: Vec<Message> = Vec::new();
    let mut cur_tok = 0usize;
    for m in rest {
        let t = est_tokens(&m.content) + 4;
        if cur_tok + t > budget && !cur.is_empty() {
            // Close the window after its last assistant turn; the remainder starts the next one.
            if let Some(pos) = cur.iter().rposition(|x| x.role == Role::Assistant) {
                let tail = cur.split_off(pos + 1);
                windows.push(std::mem::replace(&mut cur, tail));
                cur_tok = cur.iter().map(|x| est_tokens(&x.content) + 4).sum();
            }
        }
        cur.push(m.clone());
        cur_tok += t;
    }
    if !cur.is_empty() {
        windows.push(cur);
    }
    windows
        .into_iter()
        .filter(|w| w.iter().any(|m| m.role == Role::Assistant))
        .map(|w| {
            let mut messages = Vec::with_capacity(w.len() + 1);
            if let Some(s) = sys {
                messages.push(s.clone());
            }
            messages.extend(w);
            let mut s = Sample { messages, ..sample.clone() };
            s.id = s.compute_id();
            s
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Persona;

    fn persona_sample() -> Sample {
        Sample {
            id: 7,
            source: 0,
            row: 0,
            system: None,
            messages: vec![
                Message::new(Role::Assistant, "Hi {{user}}, I am {{char}}."),
                Message::new(Role::User, "hello {{char}}"),
                Message::new(Role::Assistant, "<BOT> says hi to <USER>"),
            ],
            text: None,
            persona: Some(Persona {
                char_name: Some("Rose".into()),
                user_name: None,
                description: "A florist.".into(),
                definitions: "{{char}}: hi\nEND_OF_DIALOG\n{{user}}: yo".into(),
                greeting: String::new(),
                scenario: String::new(),
            }),
        }
    }

    #[test]
    fn names_are_substituted_everywhere() {
        let f = finalize(&persona_sample(), &PersonaOptions::default());
        assert_eq!(f.messages[0].role, Role::System);
        let sys = &f.messages[0].content;
        assert!(sys.contains("You are Rose. A florist."), "{sys}");
        assert!(sys.contains("Rose: hi"), "{sys}");
        assert!(sys.contains("User: yo"), "{sys}");
        assert!(!sys.contains("END_OF_DIALOG"));
        assert!(!sys.contains("Scenario"), "empty scenario line is dropped: {sys}");
        assert_eq!(f.messages[1].content, "Hi User, I am Rose.");
        assert_eq!(f.messages[3].content, "Rose says hi to User");
        assert!(!f.messages.iter().any(|m| has_placeholders(&m.content)));
    }

    #[test]
    fn user_name_pool_is_deterministic() {
        let opts = PersonaOptions { user_name_pool: vec!["Ann".into(), "Bo".into(), "Cy".into()], ..Default::default() };
        let (_, a) = resolve_names(&persona_sample(), &opts);
        let (_, b) = resolve_names(&persona_sample(), &opts);
        assert_eq!(a, b);
        assert!(["Ann", "Bo", "Cy"].contains(&a.as_str()));
    }

    #[test]
    fn substitution_can_be_disabled() {
        let opts = PersonaOptions { substitute_names: false, ..Default::default() };
        let f = finalize(&persona_sample(), &opts);
        assert!(f.messages[1].content.contains("{{user}}"));
    }

    #[test]
    fn windowing_keeps_system_and_ends_on_assistant() {
        let mut msgs = vec![Message::new(Role::System, "sys")];
        for i in 0..20 {
            msgs.push(Message::new(Role::User, format!("question number {i} {}", "x".repeat(200))));
            msgs.push(Message::new(Role::Assistant, format!("answer number {i} {}", "y".repeat(200))));
        }
        let s = Sample { id: 1, source: 0, row: 0, system: None, messages: msgs, text: None, persona: None };
        let w = window(&s, 400);
        assert!(w.len() > 3);
        for win in &w {
            assert_eq!(win.messages[0].role, Role::System);
            assert_eq!(win.messages.last().unwrap().role, Role::Assistant);
            assert!(win.est_tokens() <= 600);
        }
        let total: usize = w.iter().map(|x| x.messages.len() - 1).sum();
        assert_eq!(total, 40, "no message lost");
    }
}
