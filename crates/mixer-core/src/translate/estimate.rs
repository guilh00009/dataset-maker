//! Pre-flight estimate shown before a translation run: how many samples, requests and tokens,
//! how much is already cached, and a rough ETA.

use super::{KeyCtx, journal::Journal, names_hash, persona_fields_in_template, prompt, protect};
use crate::jobs::Ctl;
use crate::mix::{MixPlan, SourceData};
use crate::model::{Role, est_tokens};
use crate::recipe::Recipe;
use anyhow::Result;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Estimate {
    pub samples: u64,
    pub messages: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub requests: u64,
    pub persona_units: u64,
    pub persona_tokens: u64,
    /// Share of the translatable messages already present in the journal (sampled).
    pub cached_fraction: f64,
    pub eta_secs: Option<f64>,
    pub extrapolated: bool,
}

#[derive(Default)]
struct Acc {
    samples: u64,
    messages: u64,
    tokens: u64,
    requests: u64,
    persona: HashMap<u64, u32>,
}

pub fn estimate(
    recipe: &Recipe,
    sources: &[SourceData],
    plan: &MixPlan,
    journal: Option<&Journal>,
    measured_tps: Option<f64>,
    slots: Option<u32>,
    ctl: &Ctl,
) -> Result<Estimate> {
    let t = &recipe.translate;
    let src: HashMap<usize, &SourceData> = sources.iter().map(|s| (s.index, s)).collect();
    let fields = persona_fields_in_template(&recipe.persona.template);
    let payload_budget = ((t.context_budget as f64 - 350.0) / 2.3).max(200.0) as usize;
    let prompt_tokens = est_tokens(&prompt::system_prompt(
        &prompt::language(&t.source_language),
        &prompt::language(&t.target_language),
        &t.instructions,
        &t.glossary,
        true,
    ));

    // Very large mixes are estimated from an even subset and scaled up.
    const CAP: usize = 120_000;
    let all = &plan.entries;
    let step = (all.len() / CAP).max(1);
    let picked: Vec<_> = all.iter().step_by(step).copied().collect();
    let scale = all.len() as f64 / picked.len().max(1) as f64;

    let acc = picked
        .par_iter()
        .fold(Acc::default, |mut a, e| {
            let Some(sd) = src.get(&(e.source as usize)) else { return a };
            let Some(s) = sd.sample(e.row) else { return a };
            a.samples += 1;
            let mut payload = 0usize;
            for m in &s.messages {
                if t.roles.contains(&m.role) {
                    payload += est_tokens(&m.content) + 14;
                    a.messages += 1;
                }
            }
            if let Some(tx) = &s.text {
                if s.messages.is_empty() {
                    payload += est_tokens(tx) + 14;
                    a.messages += 1;
                }
            }
            if payload > 0 {
                a.requests += payload.div_ceil(payload_budget) as u64;
            }
            a.tokens += payload as u64;
            if t.translate_persona {
                if let Some(p) = &s.persona {
                    let mut add = |name: &str, text: &str| {
                        if fields.iter().any(|f| f == name) && !text.trim().is_empty() {
                            let h = xxhash_rust::xxh3::xxh3_64(text.as_bytes());
                            a.persona.entry(h).or_insert(est_tokens(text) as u32);
                        }
                    };
                    add("description", &p.description);
                    add("definitions", &p.definitions);
                    add("scenario", &p.scenario);
                    add("greeting", &p.greeting);
                }
            }
            a
        })
        .reduce(Acc::default, |mut a, b| {
            a.samples += b.samples;
            a.messages += b.messages;
            a.tokens += b.tokens;
            a.requests += b.requests;
            a.persona.extend(b.persona);
            a
        });
    ctl.check()?;

    // persona units are shared between samples: scale only the part that is not saturated
    let persona_units = acc.persona.len() as f64 * if scale > 1.0 { scale.min(2.0) } else { 1.0 };
    let persona_tokens: f64 = acc.persona.values().map(|v| *v as f64).sum::<f64>() * if scale > 1.0 { scale.min(2.0) } else { 1.0 };

    // sampled cache hit rate
    let mut hits = 0usize;
    let mut probed = 0usize;
    if let Some(j) = journal {
        let keys = KeyCtx::new(t);
        'outer: for e in picked.iter().take(80) {
            let Some(sd) = src.get(&(e.source as usize)) else { continue };
            let Some(s) = sd.sample(e.row) else { continue };
            let mut names = Vec::new();
            if let Some(p) = &s.persona {
                names.extend(p.char_name.clone());
                names.extend(p.user_name.clone());
            }
            let nh = names_hash(&names);
            for m in s.messages.iter().filter(|m| t.roles.contains(&m.role) && m.role != Role::System) {
                let prot = protect::protect(&m.content, &names);
                if !protect::has_translatable_text(&prot.text) {
                    continue;
                }
                let segs = prompt::split_segments(&prot.text, (payload_budget as f64 * 3.0) as usize);
                probed += 1;
                if segs.iter().all(|(s, _)| j.get(keys.seg_key(s, nh)).is_some()) {
                    hits += 1;
                }
                if probed >= 400 {
                    break 'outer;
                }
            }
        }
    }
    let cached_fraction = if probed > 0 { hits as f64 / probed as f64 } else { 0.0 };

    let msg_tokens = acc.tokens as f64 * scale * (1.0 - cached_fraction);
    let requests = acc.requests as f64 * scale * (1.0 - cached_fraction) + persona_units;
    let input_tokens = msg_tokens + persona_tokens + requests * prompt_tokens as f64;
    let output_tokens = (msg_tokens + persona_tokens) * 1.2;
    let eta_secs = measured_tps.filter(|v| *v > 0.0).map(|tps| {
        let conc = (t.concurrency.max(1) as f64).min(slots.unwrap_or(t.concurrency as u32).max(1) as f64);
        output_tokens / (tps * conc.powf(0.7))
    });
    Ok(Estimate {
        samples: all.len() as u64,
        messages: (acc.messages as f64 * scale) as u64,
        input_tokens: input_tokens as u64,
        output_tokens: output_tokens as u64,
        requests: requests.ceil() as u64,
        persona_units: persona_units as u64,
        persona_tokens: persona_tokens as u64,
        cached_fraction,
        eta_secs,
        extrapolated: scale > 1.0,
    })
}
