//! Protecting spans that must never be translated (code, URLs, placeholders, names, LaTeX).
//! Spans are swapped for opaque tokens `⟦n⟧` before sending and restored afterwards.

use regex::Regex;
use std::sync::OnceLock;

pub const OPEN: char = '⟦';
pub const CLOSE: char = '⟧';

fn base_rx() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(concat!(
            r"(?s)```.*?```",                          // fenced code
            r"|`[^`\n]+`",                             // inline code
            r#"|https?://[^\s<>"'\)\]]+"#,             // URLs
            r"|\{\{[^{}\n]{1,60}\}\}",                 // {{char}} / {{user}} / {{anything}}
            r"|<BOT>|<USER>",                          // legacy placeholders
            r"|\{[A-Za-z_][A-Za-z0-9_]{0,30}\}",       // {placeholder}
            r"|(?s)\$\$.*?\$\$",                       // display math
            r"|(?s)\\\[.*?\\\]",                       // \[ ... \]
            r"|(?s)\\\(.*?\\\)",                       // \( ... \)
            r"|⟦\d+⟧",                       // tokens already present in the source
        ))
        .unwrap()
    })
}

fn token_rx() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new("\u{27e6}(\\d+)\u{27e7}").unwrap())
}

#[derive(Debug, Clone)]
pub struct Protected {
    pub text: String,
    pub spans: Vec<String>,
}

fn names_rx(names: &[String]) -> Option<Regex> {
    let mut ns: Vec<String> = names.iter().map(|n| n.trim().to_string()).filter(|n| n.chars().count() >= 2).collect();
    ns.sort_by_key(|n| std::cmp::Reverse(n.len()));
    ns.dedup();
    if ns.is_empty() {
        return None;
    }
    let alt = ns.iter().map(|n| regex::escape(n)).collect::<Vec<_>>().join("|");
    Regex::new(&format!(r"(?i)\b(?:{alt})\b")).ok()
}

/// Replace every protected span with `⟦n⟧`.
pub fn protect(text: &str, names: &[String]) -> Protected {
    let mut spans: Vec<String> = Vec::new();
    let step1 = base_rx().replace_all(text, |c: &regex::Captures| {
        spans.push(c[0].to_string());
        format!("{OPEN}{}{CLOSE}", spans.len() - 1)
    });
    let text = match names_rx(names) {
        Some(rx) => rx
            .replace_all(&step1, |c: &regex::Captures| {
                spans.push(c[0].to_string());
                format!("{OPEN}{}{CLOSE}", spans.len() - 1)
            })
            .into_owned(),
        None => step1.into_owned(),
    };
    Protected { text, spans }
}

/// Put the original spans back.
pub fn restore(text: &str, spans: &[String]) -> String {
    token_rx()
        .replace_all(text, |c: &regex::Captures| {
            c[1].parse::<usize>().ok().and_then(|i| spans.get(i).cloned()).unwrap_or_else(|| c[0].to_string())
        })
        .into_owned()
}

/// Token ids appearing in `text`, in order.
pub fn tokens_in(text: &str) -> Vec<usize> {
    token_rx().captures_iter(text).filter_map(|c| c[1].parse().ok()).collect()
}

/// Is there anything left to translate once the protected tokens are removed?
pub fn has_translatable_text(protected: &str) -> bool {
    token_rx().replace_all(protected, " ").chars().any(|c| c.is_alphabetic())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protects_and_restores_roundtrip() {
        let src = "Hi {{user}}, see https://x.org/a?b=1 and `code()` then ```py\nprint(1)\n``` — I'm Rose. $$x^2$$ {name}";
        let p = protect(src, &["Rose".into()]);
        assert!(!p.text.contains("https"));
        assert!(!p.text.contains("Rose"));
        assert!(!p.text.contains("{{user}}"));
        assert_eq!(restore(&p.text, &p.spans), src);
        assert_eq!(tokens_in(&p.text).len(), p.spans.len());
    }

    #[test]
    fn names_are_whole_words_and_case_insensitive() {
        let p = protect("Rosemary met rose and ROSE.", &["Rose".into()]);
        assert!(p.text.starts_with("Rosemary"), "{}", p.text);
        assert_eq!(p.spans, vec!["rose".to_string(), "ROSE".to_string()]);
    }

    #[test]
    fn nothing_to_translate_detection() {
        assert!(!has_translatable_text("\u{27e6}0\u{27e7} \u{27e6}1\u{27e7} 123 !!"));
        assert!(has_translatable_text("\u{27e6}0\u{27e7} hello"));
    }

    #[test]
    fn existing_tokens_survive() {
        let p = protect("odd \u{27e6}7\u{27e7} text", &[]);
        assert_eq!(restore(&p.text, &p.spans), "odd \u{27e6}7\u{27e7} text");
    }
}
