//! Scanning sources (filters, dedup, stats), computing per-source quotas and building the
//! shuffled mix. The mixer only ever reorders *row positions*, never full rows, so shuffling
//! tens of millions of rows needs a few hundred MB at most.

use crate::detect::{Mapping, normalize};
use crate::jobs::{Ctl, Progress, Update};
use crate::model::Sample;
use crate::recipe::{DedupMode, Filters, MixOptions, Recipe, ShuffleMode, Strategy};
use crate::source::Prepared;
use anyhow::{Result, bail};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;

/// A prepared source plus the mapping used to normalize its rows.
#[derive(Clone)]
pub struct SourceData {
    pub index: usize,
    pub id: String,
    pub prepared: Arc<Prepared>,
    pub mapping: Mapping,
}

impl SourceData {
    pub fn sample(&self, row: u32) -> Option<Sample> {
        let v = self.prepared.row_value(row as usize).ok()?;
        normalize(&v, &self.mapping, self.index, row as u64)
    }
}

pub const HIST_BOUNDS: [u32; 9] = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384];

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SourceStats {
    pub total: u64,
    pub valid: u64,
    pub malformed: u64,
    pub unmatched: u64,
    pub filtered_turns: u64,
    pub filtered_tokens: u64,
    pub filtered_language: u64,
    pub filtered_regex: u64,
    pub duplicates: u64,
    pub tokens_total: u64,
    pub turns_total: u64,
    /// Counts per bucket of `HIST_BOUNDS` (+ one overflow bucket).
    pub histogram: Vec<u64>,
}

impl SourceStats {
    pub fn avg_tokens(&self) -> f64 {
        if self.valid == 0 { 0.0 } else { self.tokens_total as f64 / self.valid as f64 }
    }
    pub fn avg_turns(&self) -> f64 {
        if self.valid == 0 { 0.0 } else { self.turns_total as f64 / self.valid as f64 }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ScanRow {
    pub row: u32,
    pub id: u64,
    pub tokens: u32,
}

pub struct Scan {
    pub rows: Vec<ScanRow>,
    pub stats: SourceStats,
}

/// Filters with their regexes compiled once.
pub struct CompiledFilters {
    f: Filters,
    include: Option<Regex>,
    exclude: Option<Regex>,
    lang: Option<whatlang::Lang>,
}

impl CompiledFilters {
    pub fn new(f: &Filters) -> Result<Self> {
        let include = f.include_regex.as_deref().filter(|s| !s.is_empty()).map(Regex::new).transpose()
            .map_err(|e| anyhow::anyhow!("invalid include regex: {e}"))?;
        let exclude = f.exclude_regex.as_deref().filter(|s| !s.is_empty()).map(Regex::new).transpose()
            .map_err(|e| anyhow::anyhow!("invalid exclude regex: {e}"))?;
        let lang = match f.language.as_deref().filter(|s| !s.is_empty()) {
            Some(code) => Some(whatlang::Lang::from_code(code).ok_or_else(|| anyhow::anyhow!("unknown language code `{code}` (use ISO 639-3 like eng, por, spa)"))?),
            None => None,
        };
        Ok(Self { f: f.clone(), include, exclude, lang })
    }
}

#[derive(Debug)]
enum Reject {
    Malformed,
    Unmatched,
    Turns,
    Tokens,
    Language,
    Regex,
}

fn sample_text(s: &Sample, cap: usize) -> String {
    let mut out = String::new();
    for m in &s.messages {
        if out.len() >= cap {
            break;
        }
        out.push_str(&m.content);
        out.push('\n');
    }
    if let Some(t) = &s.text {
        out.push_str(t);
    }
    if out.len() > cap {
        let mut end = cap;
        while !out.is_char_boundary(end) {
            end -= 1;
        }
        out.truncate(end);
    }
    out
}

fn scan_one(src: &SourceData, f: &CompiledFilters, row: u32) -> std::result::Result<ScanRow, Reject> {
    let v = src.prepared.row_value(row as usize).map_err(|_| Reject::Malformed)?;
    let s = normalize(&v, &src.mapping, src.index, row as u64).ok_or(Reject::Unmatched)?;
    if !s.is_text() {
        if s.turns() < f.f.min_turns {
            return Err(Reject::Turns);
        }
        if let Some(max) = f.f.max_turns {
            if s.turns() > max {
                return Err(Reject::Turns);
            }
        }
        if f.f.drop_empty_messages && s.messages.iter().any(|m| m.content.trim().is_empty()) {
            return Err(Reject::Unmatched);
        }
    }
    let tokens = s.est_tokens();
    if f.f.min_tokens.is_some_and(|m| tokens < m) || f.f.max_tokens.is_some_and(|m| tokens > m) {
        return Err(Reject::Tokens);
    }
    if f.lang.is_some() || f.include.is_some() || f.exclude.is_some() {
        let text = sample_text(&s, 4000);
        if let Some(want) = f.lang {
            match whatlang::detect(&text) {
                Some(info) if info.lang() == want => {}
                _ => return Err(Reject::Language),
            }
        }
        if f.include.as_ref().is_some_and(|r| !r.is_match(&text)) {
            return Err(Reject::Regex);
        }
        if f.exclude.as_ref().is_some_and(|r| r.is_match(&text)) {
            return Err(Reject::Regex);
        }
    }
    Ok(ScanRow { row, id: s.id, tokens: tokens as u32 })
}

/// Scan every row of a source in parallel: normalize, filter, hash, collect stats.
pub fn scan_source(src: &SourceData, filters: &Filters, ctl: &Ctl, progress: &Progress) -> Result<Scan> {
    let cf = CompiledFilters::new(filters)?;
    let n = src.prepared.len();
    if n > u32::MAX as usize {
        bail!("sources with more than 4 billion rows are not supported");
    }
    let mut stats = SourceStats { total: n as u64, histogram: vec![0; HIST_BOUNDS.len() + 1], ..Default::default() };
    let mut rows: Vec<ScanRow> = Vec::with_capacity(n.min(1 << 24));
    const CHUNK: usize = 32_768;
    let mut start = 0usize;
    while start < n {
        ctl.check()?;
        let end = (start + CHUNK).min(n);
        let part: Vec<std::result::Result<ScanRow, Reject>> =
            (start..end).into_par_iter().map(|i| scan_one(src, &cf, i as u32)).collect();
        for r in part {
            match r {
                Ok(sr) => {
                    stats.valid += 1;
                    stats.tokens_total += sr.tokens as u64;
                    let b = HIST_BOUNDS.iter().position(|b| sr.tokens <= *b).unwrap_or(HIST_BOUNDS.len());
                    stats.histogram[b] += 1;
                    rows.push(sr);
                }
                Err(Reject::Malformed) => stats.malformed += 1,
                Err(Reject::Unmatched) => stats.unmatched += 1,
                Err(Reject::Turns) => stats.filtered_turns += 1,
                Err(Reject::Tokens) => stats.filtered_tokens += 1,
                Err(Reject::Language) => stats.filtered_language += 1,
                Err(Reject::Regex) => stats.filtered_regex += 1,
            }
        }
        start = end;
        progress(Update::new("scan", start as u64, n as u64, format!("Scanning {}", src.id)));
    }
    Ok(Scan { rows, stats })
}

// ───────────────────────────── plan ─────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanEntry {
    pub source: u16,
    pub row: u32,
    pub tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourcePlan {
    pub source: usize,
    pub id: String,
    pub available: u64,
    pub requested: u64,
    pub taken: u64,
    pub share: f64,
    pub stats: SourceStats,
}

pub struct MixPlan {
    /// Final order. The last `val_count` entries form the validation split.
    pub entries: Vec<PlanEntry>,
    pub val_count: usize,
    pub per_source: Vec<SourcePlan>,
    pub tokens_total: u64,
    pub histogram: Vec<u64>,
}

impl MixPlan {
    pub fn train(&self) -> &[PlanEntry] {
        &self.entries[..self.entries.len() - self.val_count]
    }
    pub fn validation(&self) -> &[PlanEntry] {
        &self.entries[self.entries.len() - self.val_count..]
    }
}

/// Unbiased-enough uniform integer in `[0, n)`.
fn below(rng: &mut ChaCha8Rng, n: usize) -> usize {
    ((rng.next_u64() as u128 * n as u128) >> 64) as usize
}

pub fn shuffle<T>(v: &mut [T], rng: &mut ChaCha8Rng) {
    for i in (1..v.len()).rev() {
        let j = below(rng, i + 1);
        v.swap(i, j);
    }
}

#[derive(Debug, Clone)]
pub struct QuotaInput {
    pub available: u64,
    pub weight: f64,
    pub count: Option<u64>,
}

/// Decide how many rows to take from each source.
/// Returns `(requested, taken)` per source. `taken` respects the upsample cap.
pub fn quotas(inputs: &[QuotaInput], mix: &MixOptions) -> Vec<(u64, u64)> {
    let cap = |avail: u64| avail.saturating_mul(mix.max_upsample.max(1) as u64);
    let mut taken = vec![0u64; inputs.len()];
    let mut requested = vec![0u64; inputs.len()];

    // fixed counts first
    let mut weighted: Vec<usize> = Vec::new();
    for (i, inp) in inputs.iter().enumerate() {
        match inp.count {
            Some(c) => {
                requested[i] = c;
                taken[i] = c.min(cap(inp.available));
            }
            None => weighted.push(i),
        }
    }
    if weighted.is_empty() {
        return requested.into_iter().zip(taken).collect();
    }

    // effective weights
    let w: Vec<f64> = weighted
        .iter()
        .map(|&i| {
            let base = inputs[i].weight.max(0.0);
            match mix.strategy {
                Strategy::Proportional => base,
                Strategy::Temperature => base * (inputs[i].available.max(1) as f64).powf(1.0 / mix.temperature.max(0.05)),
            }
        })
        .collect();
    let wsum: f64 = w.iter().sum();
    if wsum <= 0.0 {
        return requested.into_iter().zip(taken).collect();
    }
    let share: Vec<f64> = w.iter().map(|x| x / wsum).collect();

    // total for the weighted group
    let total = match mix.total {
        Some(t) => {
            let fixed: u64 = taken.iter().sum();
            t.saturating_sub(fixed)
        }
        None => {
            // largest total such that no weighted source needs more rows than it can supply
            let mut t = f64::INFINITY;
            for (k, &i) in weighted.iter().enumerate() {
                if share[k] > 0.0 {
                    t = t.min(cap(inputs[i].available) as f64 / share[k]);
                }
            }
            if t.is_finite() { t.floor() as u64 } else { 0 }
        }
    };

    // water-filling: redistribute what capped sources cannot supply
    let mut got = vec![0u64; weighted.len()];
    let mut active: Vec<usize> = (0..weighted.len()).filter(|&k| share[k] > 0.0).collect();
    let mut remaining = total;
    for k in 0..weighted.len() {
        requested[weighted[k]] = (share[k] * total as f64).round() as u64;
    }
    for _ in 0..8 {
        if remaining == 0 || active.is_empty() {
            break;
        }
        let ash: f64 = active.iter().map(|&k| share[k]).sum();
        let mut next_active = Vec::new();
        let mut used = 0u64;
        for &k in &active {
            let want = ((share[k] / ash) * remaining as f64).round() as u64;
            let room = cap(inputs[weighted[k]].available).saturating_sub(got[k]);
            let add = want.min(room);
            got[k] += add;
            used += add;
            if got[k] < cap(inputs[weighted[k]].available) {
                next_active.push(k);
            }
        }
        remaining = remaining.saturating_sub(used);
        if used == 0 {
            break;
        }
        active = next_active;
    }
    for (k, &i) in weighted.iter().enumerate() {
        taken[i] = got[k];
    }
    requested.into_iter().zip(taken).collect()
}

/// Build the final shuffled plan from scanned sources.
pub fn build_plan(recipe: &Recipe, scans: &[(&SourceData, &Scan)], ctl: &Ctl) -> Result<MixPlan> {
    if scans.is_empty() {
        bail!("add at least one enabled source first");
    }
    // 1. global exact dedup (first occurrence wins, in source order)
    let mut dup_counts = vec![0u64; scans.len()];
    let mut valid: Vec<Vec<ScanRow>> = Vec::with_capacity(scans.len());
    match recipe.filters.dedup {
        DedupMode::Exact => {
            let total: usize = scans.iter().map(|(_, s)| s.rows.len()).sum();
            let mut seen: HashSet<u64> = HashSet::with_capacity(total);
            for (k, (_, sc)) in scans.iter().enumerate() {
                ctl.check()?;
                let mut keep = Vec::with_capacity(sc.rows.len());
                for r in &sc.rows {
                    if seen.insert(r.id) {
                        keep.push(*r);
                    } else {
                        dup_counts[k] += 1;
                    }
                }
                valid.push(keep);
            }
        }
        DedupMode::None => {
            for (_, sc) in scans {
                valid.push(sc.rows.clone());
            }
        }
    }

    // 2. quotas
    let inputs: Vec<QuotaInput> = scans
        .iter()
        .enumerate()
        .map(|(k, (sd, _))| {
            let e = &recipe.sources[sd.index];
            QuotaInput { available: valid[k].len() as u64, weight: e.weight, count: e.count }
        })
        .collect();
    let q = quotas(&inputs, &recipe.mix);
    let wsum: f64 = inputs.iter().map(|i| i.weight.max(0.0)).sum();

    // 3. pick rows per source
    let mut per_source_entries: Vec<Vec<PlanEntry>> = Vec::with_capacity(scans.len());
    for (k, (sd, _)) in scans.iter().enumerate() {
        ctl.check()?;
        let mut rng = ChaCha8Rng::seed_from_u64(recipe.seed ^ ((sd.index as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)));
        let rows = &valid[k];
        let take = q[k].1 as usize;
        let mut pool: Vec<usize> = (0..rows.len()).collect();
        let mut chosen: Vec<usize> = Vec::with_capacity(take);
        let full = if rows.is_empty() { 0 } else { take / rows.len() };
        for _ in 0..full {
            chosen.extend_from_slice(&pool);
        }
        let rem = take - full * rows.len();
        // partial Fisher–Yates for the remainder
        for i in 0..rem.min(pool.len()) {
            let j = i + below(&mut rng, pool.len() - i);
            pool.swap(i, j);
            chosen.push(pool[i]);
        }
        let mut entries: Vec<PlanEntry> = chosen
            .into_iter()
            .map(|ix| PlanEntry { source: sd.index as u16, row: rows[ix].row, tokens: rows[ix].tokens })
            .collect();
        if recipe.mix.shuffle == ShuffleMode::None {
            entries.sort_by_key(|e| e.row);
        } else {
            shuffle(&mut entries, &mut rng);
        }
        per_source_entries.push(entries);
    }

    // 4. combine
    let mut rng = ChaCha8Rng::seed_from_u64(recipe.seed.wrapping_mul(0xD1B5_4A32_D192_ED03) ^ 0x5151);
    let mut entries: Vec<PlanEntry> = Vec::with_capacity(per_source_entries.iter().map(|v| v.len()).sum());
    match recipe.mix.shuffle {
        ShuffleMode::Full => {
            for v in &per_source_entries {
                entries.extend_from_slice(v);
            }
            shuffle(&mut entries, &mut rng);
        }
        ShuffleMode::None => {
            for v in &per_source_entries {
                entries.extend_from_slice(v);
            }
        }
        ShuffleMode::Interleave => {
            // smooth weighted round robin across sources
            let sizes: Vec<f64> = per_source_entries.iter().map(|v| v.len() as f64).collect();
            let total: f64 = sizes.iter().sum();
            let mut pos = vec![0usize; per_source_entries.len()];
            let mut credit = vec![0f64; per_source_entries.len()];
            for _ in 0..(total as usize) {
                for (k, s) in sizes.iter().enumerate() {
                    credit[k] += s / total;
                }
                let best = (0..credit.len())
                    .filter(|&k| pos[k] < per_source_entries[k].len())
                    .max_by(|&a, &b| credit[a].partial_cmp(&credit[b]).unwrap_or(std::cmp::Ordering::Equal));
                if let Some(k) = best {
                    entries.push(per_source_entries[k][pos[k]]);
                    pos[k] += 1;
                    credit[k] -= 1.0;
                }
            }
        }
    }

    // 5. split
    let val_frac = recipe.split.validation.clamp(0.0, 0.5);
    let val_count = ((entries.len() as f64) * val_frac).floor() as usize;

    // 6. report
    let mut hist = vec![0u64; HIST_BOUNDS.len() + 1];
    let mut tokens_total = 0u64;
    for e in &entries {
        tokens_total += e.tokens as u64;
        let b = HIST_BOUNDS.iter().position(|b| e.tokens <= *b).unwrap_or(HIST_BOUNDS.len());
        hist[b] += 1;
    }
    let per_source = scans
        .iter()
        .enumerate()
        .map(|(k, (sd, sc))| {
            let mut stats = sc.stats.clone();
            stats.duplicates = dup_counts[k];
            stats.valid = stats.valid.saturating_sub(dup_counts[k]);
            SourcePlan {
                source: sd.index,
                id: sd.id.clone(),
                available: valid[k].len() as u64,
                requested: q[k].0,
                taken: per_source_entries[k].len() as u64,
                share: if wsum > 0.0 { recipe.sources[sd.index].weight.max(0.0) / wsum } else { 0.0 },
                stats,
            }
        })
        .collect();
    Ok(MixPlan { entries, val_count, per_source, tokens_total, histogram: hist })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(avail: &[u64], weights: &[f64], total: Option<u64>) -> Vec<u64> {
        let inputs: Vec<QuotaInput> =
            avail.iter().zip(weights).map(|(a, w)| QuotaInput { available: *a, weight: *w, count: None }).collect();
        let mix = MixOptions { total, ..Default::default() };
        quotas(&inputs, &mix).into_iter().map(|(_, t)| t).collect()
    }

    #[test]
    fn auto_total_respects_weights_and_availability() {
        // 50/50 but one source only has 100 rows -> total 200
        assert_eq!(q(&[1000, 100], &[1.0, 1.0], None), vec![100, 100]);
        // 3:1
        assert_eq!(q(&[1000, 1000], &[3.0, 1.0], None), vec![1000, 333]);
    }

    #[test]
    fn explicit_total_redistributes_shortfall() {
        let r = q(&[1000, 50], &[1.0, 1.0], Some(400));
        assert_eq!(r[1], 50);
        assert_eq!(r[0], 350);
        assert_eq!(r.iter().sum::<u64>(), 400);
    }

    #[test]
    fn counts_and_upsample() {
        let inputs = vec![
            QuotaInput { available: 10, weight: 1.0, count: Some(25) },
            QuotaInput { available: 100, weight: 1.0, count: Some(5) },
        ];
        let mix = MixOptions { max_upsample: 2, ..Default::default() };
        let r = quotas(&inputs, &mix);
        assert_eq!(r[0], (25, 20));
        assert_eq!(r[1], (5, 5));
    }

    #[test]
    fn temperature_flattens() {
        let inputs = vec![
            QuotaInput { available: 10_000, weight: 1.0, count: None },
            QuotaInput { available: 100, weight: 1.0, count: None },
        ];
        let prop = quotas(&inputs, &MixOptions { total: Some(1000), ..Default::default() });
        let temp = quotas(&inputs, &MixOptions { strategy: Strategy::Temperature, temperature: 4.0, total: Some(1000), ..Default::default() });
        assert!(temp[1].1 > prop[1].1 || prop[1].1 == 0 || temp[1].1 >= 100);
        assert_eq!(temp.iter().map(|x| x.1).sum::<u64>(), 1000.min(10_100));
    }

    #[test]
    fn shuffle_is_deterministic_permutation() {
        let mut a: Vec<u32> = (0..1000).collect();
        let mut b = a.clone();
        shuffle(&mut a, &mut ChaCha8Rng::seed_from_u64(7));
        shuffle(&mut b, &mut ChaCha8Rng::seed_from_u64(7));
        assert_eq!(a, b);
        let mut sorted = a.clone();
        sorted.sort();
        assert_eq!(sorted, (0..1000).collect::<Vec<_>>());
        assert_ne!(a, (0..1000).collect::<Vec<_>>());
    }
}
