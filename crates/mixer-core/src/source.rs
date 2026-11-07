//! Opening data sources of (almost) any file format.
//!
//! Every input is *materialized* into a JSONL file (or used in place when it is
//! already JSONL), then memory-mapped and indexed by line offset. Nothing is
//! ever fully loaded into memory, so multi-GB files work.

use crate::jobs::{Ctl, Progress, Update};
use anyhow::{Context, Result, bail};
use memmap2::Mmap;
use serde_json::Value;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use xxhash_rust::xxh3::xxh3_64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FileFormat {
    Jsonl,
    JsonArray,
    JsonWrapped,
    Csv,
    Tsv,
    Parquet,
    Text,
    Directory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Compression {
    None,
    Gzip,
    Zstd,
}

/// A prepared (materialized + indexed) source. Rows are read with positional reads, so memory
/// stays flat however large the file is (only the 8-byte-per-row line index lives in RAM).
pub struct Prepared {
    pub original: PathBuf,
    pub jsonl: PathBuf,
    pub format: FileFormat,
    pub compression: Compression,
    pub bytes: u64,
    file: File,
    /// Start offset of every non-empty line.
    offsets: Vec<u64>,
}

impl std::fmt::Debug for Prepared {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Prepared")
            .field("original", &self.original)
            .field("rows", &self.offsets.len())
            .field("format", &self.format)
            .finish()
    }
}

#[cfg(unix)]
fn read_at(f: &File, buf: &mut [u8], off: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    f.read_exact_at(buf, off)
}

#[cfg(windows)]
fn read_at(f: &File, mut buf: &mut [u8], mut off: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        let n = f.seek_read(buf, off)?;
        if n == 0 {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }
        buf = &mut buf[n..];
        off += n as u64;
    }
    Ok(())
}

impl Prepared {
    pub fn len(&self) -> usize {
        self.offsets.len()
    }
    pub fn is_empty(&self) -> bool {
        self.offsets.is_empty()
    }

    /// The bytes of one row (without its line ending).
    pub fn row_bytes(&self, i: usize) -> Result<Vec<u8>> {
        let start = self.offsets[i];
        let end = self.offsets.get(i + 1).copied().unwrap_or(self.bytes);
        let mut buf = vec![0u8; (end - start) as usize];
        read_at(&self.file, &mut buf, start).with_context(|| format!("row {i}: read error"))?;
        while buf.last().is_some_and(|b| b.is_ascii_whitespace()) {
            buf.pop();
        }
        Ok(buf)
    }

    pub fn row_value(&self, i: usize) -> Result<Value> {
        serde_json::from_slice(&self.row_bytes(i)?).with_context(|| format!("row {i}: invalid JSON"))
    }
}

#[cfg(unix)]
fn release_range(m: &Mmap, start: usize, end: usize) {
    const PAGE: usize = 4096;
    let s = start.div_ceil(PAGE) * PAGE;
    let e = end / PAGE * PAGE;
    if e > s {
        // SAFETY: the mapping is read-only and file-backed; dropping its pages cannot change data
        // (they are simply faulted in again from the file if touched later).
        unsafe {
            let _ = m.unchecked_advise_range(memmap2::UncheckedAdvice::DontNeed, s, e - s);
        }
    }
}

fn cache_key(path: &Path, extra: &str) -> String {
    let meta = std::fs::metadata(path).ok();
    let size = meta.as_ref().map_or(0, |m| m.len());
    let mtime = meta
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_secs());
    let canon = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    format!("{:016x}", xxh3_64(format!("v2|{}|{size}|{mtime}|{extra}", canon.display()).as_bytes()))
}

fn sniff_compression(head: &[u8]) -> Compression {
    if head.starts_with(&[0x1f, 0x8b]) {
        Compression::Gzip
    } else if head.starts_with(&[0x28, 0xb5, 0x2f, 0xfd]) {
        Compression::Zstd
    } else {
        Compression::None
    }
}

fn read_head(path: &Path, n: usize) -> Result<Vec<u8>> {
    let mut f = File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    let mut buf = vec![0u8; n];
    let mut got = 0;
    while got < n {
        let r = f.read(&mut buf[got..])?;
        if r == 0 {
            break;
        }
        got += r;
    }
    buf.truncate(got);
    Ok(buf)
}

/// Decide the text format from the file's content (never from its extension).
fn sniff_format(path: &Path) -> Result<FileFormat> {
    if path.is_dir() {
        return Ok(FileFormat::Directory);
    }
    let head = read_head(path, 256 * 1024)?;
    if head.starts_with(b"PAR1") {
        return Ok(FileFormat::Parquet);
    }
    if head.is_empty() {
        bail!("{} is empty", path.display());
    }
    let text = String::from_utf8_lossy(&head);
    let trimmed = text.trim_start_matches('\u{feff}').trim_start();
    let first = trimmed.chars().next().unwrap_or(' ');
    if first == '[' {
        return Ok(FileFormat::JsonArray);
    }
    if first == '{' {
        // JSONL if the first full line parses as an object on its own...
        let mut lines = trimmed.lines();
        let first_line = lines.next().unwrap_or("");
        if let Ok(Value::Object(o)) = serde_json::from_str::<Value>(first_line) {
            // ...except a single-line document that wraps its rows (`{"data":[...]}`).
            let rest_empty = lines.all(|l| l.trim().is_empty()) && head.len() < 256 * 1024;
            const WRAPPERS: &[&str] = &["data", "rows", "items", "examples", "instances", "train", "records", "dataset", "entries", "results", "samples"];
            let wrapper = o.iter().any(|(k, v)| WRAPPERS.contains(&k.to_lowercase().as_str()) && (v.is_array() || v.is_object()));
            if rest_empty && wrapper {
                return Ok(FileFormat::JsonWrapped);
            }
            return Ok(FileFormat::Jsonl);
        }
        return Ok(FileFormat::JsonWrapped);
    }
    // CSV / TSV: parse the head with the real CSV parser and require a consistent field count.
    for d in [b',', b'\t', b';', b'|'] {
        if let Some(n) = csv_consistent_fields(&head, d) {
            let _ = n;
            return Ok(if d == b'\t' { FileFormat::Tsv } else { FileFormat::Csv });
        }
    }
    // A JSON string per line ("hello"\n"world")?
    if first == '"' {
        return Ok(FileFormat::Jsonl);
    }
    Ok(FileFormat::Text)
}

/// Number of fields if every record in `head` (ignoring a possibly truncated last one)
/// has the same count of at least two fields.
fn csv_consistent_fields(head: &[u8], delim: u8) -> Option<usize> {
    let mut rdr = csv::ReaderBuilder::new().delimiter(delim).has_headers(false).flexible(true).from_reader(head);
    let recs: Vec<usize> = rdr.records().take(25).filter_map(|r| r.ok()).map(|r| r.len()).collect();
    let usable = if head.len() >= 256 * 1024 && recs.len() > 1 { &recs[..recs.len() - 1] } else { &recs[..] };
    if usable.len() >= 2 && usable[0] >= 2 && usable.iter().all(|c| *c == usable[0]) { Some(usable[0]) } else { None }
}

fn csv_delimiter(path: &Path, fmt: FileFormat) -> u8 {
    if fmt == FileFormat::Tsv {
        return b'\t';
    }
    let head = read_head(path, 256 * 1024).unwrap_or_default();
    for d in [b',', b';', b'|'] {
        if csv_consistent_fields(&head, d).is_some() {
            return d;
        }
    }
    b','
}

fn decompress_to(src: &Path, comp: Compression, dst: &Path, ctl: &Ctl) -> Result<()> {
    let f = BufReader::with_capacity(1 << 20, File::open(src)?);
    let mut out = BufWriter::with_capacity(1 << 20, File::create(dst)?);
    let mut reader: Box<dyn Read> = match comp {
        Compression::Gzip => Box::new(flate2::read::MultiGzDecoder::new(f)),
        Compression::Zstd => Box::new(zstd::stream::read::Decoder::new(f)?),
        Compression::None => Box::new(f),
    };
    let mut buf = vec![0u8; 1 << 20];
    loop {
        ctl.check()?;
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])?;
    }
    out.flush()?;
    Ok(())
}

/// Recursively find the longest array of rows inside a JSON document.
fn find_rows(v: Value) -> Option<Vec<Value>> {
    fn best<'a>(v: &'a Value, depth: usize, out: &mut Option<&'a Vec<Value>>) {
        match v {
            Value::Array(a) if !a.is_empty() && a.iter().all(|x| x.is_object() || x.is_string() || x.is_array()) => {
                if out.map_or(true, |o| a.len() > o.len()) {
                    *out = Some(a);
                }
            }
            Value::Object(m) if depth < 4 => {
                for (_, x) in m {
                    best(x, depth + 1, out);
                }
            }
            _ => {}
        }
    }
    let mut found = None;
    best(&v, 0, &mut found);
    let rows = found?.clone();
    // HF datasets-server style: [{"row_idx":0,"row":{...}}]
    let unwrapped: Vec<Value> = rows
        .into_iter()
        .map(|r| match &r {
            Value::Object(m) if m.contains_key("row") && m.get("row").is_some_and(|x| x.is_object()) && m.len() <= 3 => {
                m["row"].clone()
            }
            _ => r,
        })
        .collect();
    Some(unwrapped)
}

fn write_value_line<W: Write>(out: &mut W, v: &Value) -> Result<()> {
    let row = match v {
        Value::Object(_) => v.clone(),
        // Strings / arrays become a one-field object so detection can see them.
        Value::String(s) => serde_json::json!({ "text": s }),
        other => serde_json::json!({ "data": other }),
    };
    serde_json::to_writer(&mut *out, &row)?;
    out.write_all(b"\n")?;
    Ok(())
}

fn json_array_to_jsonl(src: &Path, dst: &Path, ctl: &Ctl) -> Result<()> {
    use serde::de::{Deserializer as _, SeqAccess, Visitor};
    struct V<'a, W: Write> {
        out: &'a mut W,
        ctl: &'a Ctl,
    }
    impl<'de, W: Write> Visitor<'de> for V<'_, W> {
        type Value = ();
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("a JSON array")
        }
        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<(), A::Error> {
            while let Some(v) = seq.next_element::<Value>()? {
                self.ctl.check().map_err(serde::de::Error::custom)?;
                write_value_line(self.out, &v).map_err(serde::de::Error::custom)?;
            }
            Ok(())
        }
    }
    let reader = BufReader::with_capacity(1 << 20, File::open(src)?);
    let mut out = BufWriter::with_capacity(1 << 20, File::create(dst)?);
    let mut de = serde_json::Deserializer::from_reader(SkipBom(reader));
    de.deserialize_seq(V { out: &mut out, ctl })?;
    out.flush()?;
    Ok(())
}

/// Strips a UTF-8 BOM if present.
struct SkipBom<R: Read>(BufReader<R>);
impl<R: Read> Read for SkipBom<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let peek = self.0.fill_buf()?;
        if peek.starts_with(&[0xEF, 0xBB, 0xBF]) {
            self.0.consume(3);
        }
        self.0.read(buf)
    }
}

fn json_wrapped_to_jsonl(src: &Path, dst: &Path) -> Result<()> {
    let size = std::fs::metadata(src)?.len();
    if size > 3 << 30 {
        bail!("wrapped JSON object larger than 3 GiB is not supported; convert it to JSONL first");
    }
    let text = std::fs::read_to_string(src)?;
    let v: Value = serde_json::from_str(text.trim_start_matches('\u{feff}'))?;
    let rows = find_rows(v).context("no array of rows found inside the JSON document")?;
    let mut out = BufWriter::new(File::create(dst)?);
    for r in rows {
        write_value_line(&mut out, &r)?;
    }
    out.flush()?;
    Ok(())
}

fn cell_value(cell: &str) -> Value {
    let t = cell.trim();
    if (t.starts_with('[') && t.ends_with(']')) || (t.starts_with('{') && t.ends_with('}')) {
        if let Ok(v) = serde_json::from_str::<Value>(t) {
            return v;
        }
    }
    Value::String(cell.to_string())
}

fn csv_to_jsonl(src: &Path, dst: &Path, delim: u8, ctl: &Ctl) -> Result<()> {
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(delim)
        .flexible(true)
        .from_reader(BufReader::with_capacity(1 << 20, File::open(src)?));
    let headers: Vec<String> = rdr
        .headers()?
        .iter()
        .enumerate()
        .map(|(i, h)| if h.trim().is_empty() { format!("col{i}") } else { h.trim_start_matches('\u{feff}').to_string() })
        .collect();
    let mut out = BufWriter::with_capacity(1 << 20, File::create(dst)?);
    for rec in rdr.records() {
        ctl.check()?;
        let rec = match rec {
            Ok(r) => r,
            Err(_) => continue,
        };
        let mut m = serde_json::Map::new();
        for (h, c) in headers.iter().zip(rec.iter()) {
            m.insert(h.clone(), cell_value(c));
        }
        serde_json::to_writer(&mut out, &Value::Object(m))?;
        out.write_all(b"\n")?;
    }
    out.flush()?;
    Ok(())
}

fn parquet_to_jsonl(src: &Path, dst: &Path, ctl: &Ctl) -> Result<()> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    let f = File::open(src)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(f)?.with_batch_size(4096).build()?;
    let out = BufWriter::with_capacity(1 << 20, File::create(dst)?);
    let mut w = arrow_json::LineDelimitedWriter::new(out);
    for batch in reader {
        ctl.check()?;
        w.write(&batch?)?;
    }
    w.finish()?;
    w.into_inner().flush()?;
    Ok(())
}

fn text_to_jsonl(src: &Path, dst: &Path, ctl: &Ctl) -> Result<()> {
    // Plain text: documents are separated by <|endoftext|> if present, otherwise
    // paragraphs are grouped into chunks of about 8000 characters.
    let text = std::fs::read_to_string(src).or_else(|_| {
        std::fs::read(src).map(|b| String::from_utf8_lossy(&b).to_string())
    })?;
    let mut out = BufWriter::new(File::create(dst)?);
    let mut emit = |s: &str| -> Result<()> {
        let s = s.trim();
        if !s.is_empty() {
            serde_json::to_writer(&mut out, &serde_json::json!({ "text": s }))?;
            out.write_all(b"\n")?;
        }
        Ok(())
    };
    if text.contains("<|endoftext|>") {
        for d in text.split("<|endoftext|>") {
            ctl.check()?;
            emit(d)?;
        }
    } else if text.chars().count() <= 8000 {
        emit(&text)?;
    } else {
        let mut cur = String::new();
        for para in text.split("\n\n") {
            ctl.check()?;
            if cur.chars().count() + para.chars().count() > 8000 && !cur.is_empty() {
                emit(&cur)?;
                cur.clear();
            }
            cur.push_str(para);
            cur.push_str("\n\n");
        }
        emit(&cur)?;
    }
    drop(emit);
    out.flush()?;
    Ok(())
}

fn is_data_file(p: &Path) -> bool {
    let name = p.file_name().map(|s| s.to_string_lossy().to_lowercase()).unwrap_or_default();
    !name.starts_with('.')
        && [".jsonl", ".json", ".ndjson", ".csv", ".tsv", ".parquet", ".txt", ".md", ".gz", ".zst"]
            .iter()
            .any(|e| name.ends_with(e))
}

/// Convert any supported source into a JSONL file inside `cache_dir`
/// (or return the original path unchanged when it already is plain JSONL).
fn materialize(path: &Path, cache_dir: &Path, ctl: &Ctl, progress: &Progress) -> Result<(PathBuf, FileFormat, Compression)> {
    std::fs::create_dir_all(cache_dir)?;
    let key = cache_key(path, "mat");
    let head = if path.is_file() { read_head(path, 8)? } else { vec![] };
    let comp = sniff_compression(&head);

    // Directory: concatenate every data file.
    if path.is_dir() {
        let dst = cache_dir.join(format!("{key}.jsonl"));
        if dst.exists() {
            return Ok((dst, FileFormat::Directory, Compression::None));
        }
        let tmp = cache_dir.join(format!("{key}.jsonl.tmp"));
        let mut out = BufWriter::new(File::create(&tmp)?);
        let mut files: Vec<PathBuf> = std::fs::read_dir(path)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_file() && is_data_file(p))
            .collect();
        files.sort();
        if files.is_empty() {
            bail!("directory {} contains no readable data files", path.display());
        }
        for (i, f) in files.iter().enumerate() {
            ctl.check()?;
            progress(Update::new("prepare", i as u64, files.len() as u64, format!("Reading {}", f.display())));
            let (p, _, _) = materialize(f, cache_dir, ctl, progress)?;
            let mut r = BufReader::new(File::open(p)?);
            let mut last = b'\n';
            let mut buf = vec![0u8; 1 << 20];
            loop {
                let n = r.read(&mut buf)?;
                if n == 0 {
                    break;
                }
                out.write_all(&buf[..n])?;
                last = buf[n - 1];
            }
            if last != b'\n' {
                out.write_all(b"\n")?;
            }
        }
        out.flush()?;
        drop(out);
        std::fs::rename(&tmp, &dst)?;
        return Ok((dst, FileFormat::Directory, Compression::None));
    }

    // Compressed: decompress first, then continue with the decompressed file.
    let (work_path, comp_used) = if comp != Compression::None {
        let dst = cache_dir.join(format!("{key}.decomp"));
        if !dst.exists() {
            progress(Update::new("prepare", 0, 0, "Decompressing"));
            let tmp = cache_dir.join(format!("{key}.decomp.tmp"));
            decompress_to(path, comp, &tmp, ctl)?;
            std::fs::rename(&tmp, &dst)?;
        }
        (dst, comp)
    } else {
        (path.to_path_buf(), Compression::None)
    };

    let fmt = sniff_format(&work_path)?;
    if fmt == FileFormat::Jsonl && comp_used == Compression::None {
        return Ok((work_path, fmt, comp_used));
    }
    let dst = cache_dir.join(format!("{key}.jsonl"));
    if dst.exists() {
        return Ok((dst, fmt, comp_used));
    }
    progress(Update::new("prepare", 0, 0, format!("Converting {fmt:?} to JSONL")));
    let tmp = cache_dir.join(format!("{key}.jsonl.tmp"));
    match fmt {
        FileFormat::Jsonl => {
            std::fs::copy(&work_path, &tmp)?;
        }
        FileFormat::JsonArray => json_array_to_jsonl(&work_path, &tmp, ctl)?,
        FileFormat::JsonWrapped => json_wrapped_to_jsonl(&work_path, &tmp)?,
        FileFormat::Csv | FileFormat::Tsv => {
            let d = csv_delimiter(&work_path, fmt);
            csv_to_jsonl(&work_path, &tmp, d, ctl)?
        }
        FileFormat::Parquet => parquet_to_jsonl(&work_path, &tmp, ctl)?,
        FileFormat::Text => text_to_jsonl(&work_path, &tmp, ctl)?,
        FileFormat::Directory => unreachable!(),
    }
    std::fs::rename(&tmp, &dst)?;
    Ok((dst, fmt, comp_used))
}

/// Build the line offset index of a JSONL file (blank lines are skipped).
fn build_offsets(map: &Mmap, ctl: &Ctl, progress: &Progress) -> Result<Vec<u64>> {
    let data: &[u8] = &map[..];
    let mut offsets: Vec<u64> = Vec::with_capacity(data.len() / 512 + 16);
    let mut pos = 0usize;
    let total = data.len() as u64;
    let mut next_report = 0usize;
    while pos < data.len() {
        let end = memchr::memchr(b'\n', &data[pos..]).map_or(data.len(), |p| pos + p);
        let line = &data[pos..end];
        if line.iter().any(|b| !b.is_ascii_whitespace()) {
            offsets.push(pos as u64);
        }
        pos = end + 1;
        if pos >= next_report {
            #[cfg(unix)]
            release_range(map, next_report.saturating_sub(64 << 20), pos);
            next_report = pos + (64 << 20);
            ctl.check()?;
            progress(Update::new("index", pos.min(data.len()) as u64, total, "Indexing rows"));
        }
    }
    Ok(offsets)
}

fn load_or_build_index(jsonl: &Path, file: &File, cache_dir: &Path, ctl: &Ctl, progress: &Progress) -> Result<Vec<u64>> {
    let key = cache_key(jsonl, "idx");
    let idx_path = cache_dir.join(format!("{key}.idx"));
    if let Ok(mut f) = File::open(&idx_path) {
        let mut hdr = [0u8; 16];
        if f.read_exact(&mut hdr).is_ok() && &hdr[0..8] == b"MIXIDX01" {
            let n = u64::from_le_bytes(hdr[8..16].try_into().unwrap()) as usize;
            let mut raw = vec![0u8; n * 8];
            if f.read_exact(&mut raw).is_ok() {
                return Ok(raw.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect());
            }
        }
    }
    // SAFETY: read-only mapping used for one sequential pass; cache files are never modified in place.
    let map = unsafe { Mmap::map(file)? };
    #[cfg(unix)]
    let _ = map.advise(memmap2::Advice::Sequential);
    let offsets = build_offsets(&map, ctl, progress)?;
    drop(map);
    let tmp = idx_path.with_extension("idx.tmp");
    let mut w = BufWriter::new(File::create(&tmp)?);
    w.write_all(b"MIXIDX01")?;
    w.write_all(&(offsets.len() as u64).to_le_bytes())?;
    for o in &offsets {
        w.write_all(&o.to_le_bytes())?;
    }
    w.flush()?;
    drop(w);
    std::fs::rename(tmp, idx_path)?;
    Ok(offsets)
}

/// Open any supported file or directory: materialize to JSONL, mmap, index.
pub fn prepare(path: &Path, cache_dir: &Path, ctl: &Ctl, progress: &Progress) -> Result<Prepared> {
    if !path.exists() {
        bail!("{} does not exist", path.display());
    }
    let (jsonl, format, compression) = materialize(path, cache_dir, ctl, progress)?;
    let mut file = File::open(&jsonl)?;
    let bytes = file.seek(SeekFrom::End(0))?;
    if bytes == 0 {
        bail!("{} has no rows", path.display());
    }
    let offsets = load_or_build_index(&jsonl, &file, cache_dir, ctl, progress)?;
    if offsets.is_empty() {
        bail!("{} has no rows", path.display());
    }
    progress(Update::new("index", bytes, bytes, format!("{} rows", offsets.len())));
    Ok(Prepared { original: path.to_path_buf(), jsonl, format, compression, bytes, file, offsets })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jobs::no_progress;

    fn prep(dir: &Path, name: &str, content: &[u8]) -> Prepared {
        let p = dir.join(name);
        std::fs::write(&p, content).unwrap();
        prepare(&p, &dir.join("cache"), &Ctl::new(), &no_progress()).unwrap()
    }

    #[test]
    fn jsonl_in_place() {
        let d = tempfile::tempdir().unwrap();
        let p = prep(d.path(), "a.dat", b"{\"a\":1}\n\n{\"a\":2}\r\n{\"a\":3}");
        assert_eq!(p.len(), 3);
        assert_eq!(p.row_value(1).unwrap()["a"], 2);
        assert_eq!(p.row_value(2).unwrap()["a"], 3);
        assert_eq!(p.format, FileFormat::Jsonl);
    }

    #[test]
    fn json_array_and_wrapped() {
        let d = tempfile::tempdir().unwrap();
        let p = prep(d.path(), "a.json", br#"[{"q":"hi","a":"yo"},{"q":"x","a":"y"}]"#);
        assert_eq!(p.len(), 2);
        assert_eq!(p.format, FileFormat::JsonArray);
        let p = prep(d.path(), "b.json", br#"{"version":1,"data":{"rows":[{"q":"1"},{"q":"2"},{"q":"3"}]}}"#);
        assert_eq!(p.len(), 3);
        assert_eq!(p.format, FileFormat::JsonWrapped);
    }

    #[test]
    fn csv_and_json_cells() {
        let d = tempfile::tempdir().unwrap();
        let p = prep(d.path(), "a.csv", b"q,a,conv\nhello,world,\"[{\"\"role\"\":\"\"user\"\",\"\"content\"\":\"\"x\"\"}]\"\nfoo,bar,\n");
        assert_eq!(p.len(), 2);
        assert!(p.row_value(0).unwrap()["conv"].is_array());
    }

    #[test]
    fn gzip_jsonl() {
        use flate2::{Compression as C, write::GzEncoder};
        let d = tempfile::tempdir().unwrap();
        let mut e = GzEncoder::new(Vec::new(), C::default());
        e.write_all(b"{\"a\":1}\n{\"a\":2}\n").unwrap();
        let p = prep(d.path(), "a.jsonl.gz", &e.finish().unwrap());
        assert_eq!(p.len(), 2);
        assert_eq!(p.compression, Compression::Gzip);
    }

    #[test]
    fn plain_text() {
        let d = tempfile::tempdir().unwrap();
        let p = prep(d.path(), "a.txt", b"just some text\nwith lines");
        assert_eq!(p.len(), 1);
        assert_eq!(p.format, FileFormat::Text);
    }
}
