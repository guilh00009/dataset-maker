import { useEffect, useMemo, useState } from 'react';
import { Button, Chip, Dialog, IconButton, Stack, Surface, Switch, Text, Textfield, Title } from '@viraui/react';
import { ArrowLeft, ArrowRotate, Cloud, File, Folder, FolderOpen, Plus, Sliders, Trash, Xmark } from '@viraui/icons/react';
import { useStore } from '../store';
import { Card, Code, ConfidenceChip, Empty, JobProgress, PageHeader, Pick, Stat } from '../lib/ui';
import { fmtBytes, fmtNum } from '../lib/format';

// ───────────────────────── file picker ─────────────────────────

export function FilePicker({ open, onOpenChange, onPick, mode = 'file', title = 'Choose a file', startPath }: { open: boolean; onOpenChange: (o: boolean) => void; onPick: (path: string) => void; mode?: 'file' | 'dir'; title?: string; startPath?: string }) {
  const { call } = useStore();
  const [data, setData] = useState<any>(null);
  const [path, setPath] = useState(startPath ?? '');
  const [typed, setTyped] = useState('');
  const [err, setErr] = useState<string | null>(null);
  const [all, setAll] = useState(false);

  const load = async (p: string) => {
    try {
      const d = await call('fs_list', { path: p });
      setData(d);
      setPath(d.path);
      setTyped(d.path);
      setErr(null);
    } catch (e: any) {
      setErr(e.message);
    }
  };
  useEffect(() => { if (open) load(startPath ?? path); /* eslint-disable-next-line */ }, [open]);

  const entries: any[] = (data?.entries ?? []).filter((e: any) => e.is_dir || (mode === 'file' && (all || e.is_data)));
  const join = (n: string) => (path.endsWith('/') ? path + n : `${path}/${n}`);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <Dialog.Sheet title={title} description={mode === 'file' ? 'Pick a JSONL, JSON, CSV, Parquet or text file (compressed files work too).' : 'Pick a folder.'} maxContentWidth="64ch" showCloseButton>
        <Stack rowGap="m" fullWidth>
          <Stack direction="row" columnGap="s" vAlign="end" fullWidth expandChildren>
            <Textfield label="Location" value={typed} onChange={(e) => setTyped(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && load(typed)} fullWidth />
            <Button variant="secondary" data-grow="false" onClick={() => load(typed)}>Go</Button>
          </Stack>
          {err ? <Text tone="danger" size="s">{err}</Text> : null}
          <Surface border="all" radius="m" overflow="auto" style={{ blockSize: '20rem' }}>
            <Stack rowGap="2xs" hPadding="xs" vPadding="xs">
              {data?.parent ? (
                <Button variant="flat" size="s" addon={<ArrowLeft aria-hidden />} onClick={() => load(data.parent)}>Up one folder</Button>
              ) : null}
              {entries.map((e) => (
                <Button
                  key={e.name}
                  variant="flat"
                  size="s"
                  addon={e.is_dir ? <Folder aria-hidden /> : <File aria-hidden />}
                  onClick={() => (e.is_dir ? load(join(e.name)) : onPick(join(e.name)))}
                >
                  {e.name}{e.is_dir ? '/' : `  ·  ${fmtBytes(e.size)}`}
                </Button>
              ))}
              {!entries.length && !err ? <Text tone="muted" size="s" hPadding="m" vPadding="m">Nothing to show here.</Text> : null}
            </Stack>
          </Surface>
          <Stack direction="row" columnGap="m" hAlign="space-between" vAlign="center">
            {mode === 'file' ? <Switch checked={all} onCheckedChange={setAll} label="Show all files" /> : <span />}
            {mode === 'dir' ? <Button onClick={() => onPick(path)}>Use this folder</Button> : null}
          </Stack>
        </Stack>
      </Dialog.Sheet>
    </Dialog>
  );
}

// ───────────────────────── Hugging Face ─────────────────────────

function HfDialog({ open, onOpenChange, initial }: { open: boolean; onOpenChange: (o: boolean) => void; initial?: string }) {
  const { call, notify, refresh } = useStore();
  const [input, setInput] = useState(initial ?? '');
  const [parquet, setParquet] = useState(false);
  const [busy, setBusy] = useState(false);
  const [res, setRes] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => { if (open && initial) { setInput(initial); find(initial); } /* eslint-disable-next-line */ }, [open]);

  const find = async (text = input, rev?: string) => {
    setBusy(true); setErr(null);
    try {
      const r = await call('hf_list', { input: text, revision: rev ?? (parquet ? 'refs/convert/parquet' : '') });
      setRes(r);
      if (r.file) await add(r.repo, r.file, r.revision);
    } catch (e: any) { setErr(e.message); setRes(null); } finally { setBusy(false); }
  };
  const add = async (repo: string, file: string, revision: string) => {
    await call('add_source', { kind: 'hf', repo, file, revision });
    notify('Source added', `${repo}/${file} is downloading`);
    onOpenChange(false);
    refresh();
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <Dialog.Sheet title="Add from Hugging Face" description="Paste a dataset id or URL. Big files download once, resume if interrupted, and are cached." maxContentWidth="64ch" showCloseButton>
        <Stack rowGap="m" fullWidth>
          <Stack direction="row" columnGap="s" vAlign="end" fullWidth expandChildren>
            <Textfield label="Dataset" placeholder="PygmalionAI/PIPPA" value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && find()} fullWidth />
            <Button data-grow="false" loading={busy} onClick={() => find()}>Find files</Button>
          </Stack>
          <Switch checked={parquet} onCheckedChange={(v) => { setParquet(v); if (res) find(input, v ? 'refs/convert/parquet' : 'main'); }} label="Use Hugging Face's Parquet conversion" description="For datasets with a viewer. Datasets with a single big file (like PIPPA) should keep this off." />
          {err ? <Text tone="danger" size="s">{err}</Text> : null}
          {res ? (
            <Stack rowGap="s">
              <Text size="s" tone="muted">{res.files.length} data file{res.files.length === 1 ? '' : 's'} in {res.repo} · {res.revision}</Text>
              <Surface border="all" radius="m" overflow="auto" style={{ maxBlockSize: '18rem' }}>
                <Stack rowGap="2xs" hPadding="xs" vPadding="xs">
                  {res.files.map((f: any) => (
                    <Stack key={f.path} direction="row" vAlign="center" hAlign="space-between" columnGap="m" hPadding="s">
                      <Stack rowGap="2xs">
                        <Text size="s" family="mono">{f.path}</Text>
                        <Text size="xs" tone="muted">{fmtBytes(f.size)}</Text>
                      </Stack>
                      <Button size="s" variant="secondary" addon={<Plus aria-hidden />} onClick={() => add(res.repo, f.path, res.revision)}>Add</Button>
                    </Stack>
                  ))}
                  {!res.files.length ? <Text tone="muted" size="s" hPadding="m" vPadding="m">No data files found. Try the Parquet conversion switch.</Text> : null}
                </Stack>
              </Surface>
              {res.files.some((f: any) => f.path.includes('deduped')) ? <Text size="xs" tone="muted">Tip: files named “deduped” are usually the cleaned version.</Text> : null}
            </Stack>
          ) : null}
        </Stack>
      </Dialog.Sheet>
    </Dialog>
  );
}

// ───────────────────────── mapping editor ─────────────────────────

const NONE = '__none';
const PERSONA_FIELDS: [string, string][] = [
  ['system', 'System prompt'],
  ['char_name', 'Character name'],
  ['user_name', 'User name'],
  ['description', 'Description'],
  ['definitions', 'Example dialogue'],
  ['greeting', 'Greeting'],
  ['scenario', 'Scenario'],
];

function PathSelect({ label, value, onChange, options, allowNone = true, description }: { label: string; value?: string | null; onChange: (v: string | null) => void; options: string[]; allowNone?: boolean; description?: string }) {
  const opts = options.includes(value ?? '') || !value ? options : [value!, ...options];
  const items = [...(allowNone ? [{ value: NONE, label: '— none —' }] : []), ...opts.map((o) => ({ value: o, label: o }))];
  return <Pick label={label} description={description} value={value ?? NONE} options={items} onChange={(v) => onChange(v === NONE ? null : v)} />;
}

function MappingEditor({ src, open, onOpenChange }: { src: any; open: boolean; onOpenChange: (o: boolean) => void }) {
  const { call, refresh, notify } = useStore();
  const det = src.detection;
  const [m, setM] = useState<any>(null);
  const [raw, setRaw] = useState<string>('');
  useEffect(() => {
    if (!open) return;
    setM(JSON.parse(JSON.stringify(src.effective_mapping ?? det?.best)));
    call('preview', { source: src.id, offset: 0, count: 1 }).then((r) => setRaw(JSON.stringify(r.items?.[0]?.raw ?? {}, null, 2))).catch(() => setRaw(''));
    /* eslint-disable-next-line */
  }, [open]);
  if (!m || !det) return null;

  const fields: any[] = det.fields;
  const strings = fields.filter((f) => f.kind === 'string').map((f) => f.path);
  const arrays = fields.filter((f) => f.kind === 'array<object>').map((f) => f.path);
  const arrStr = fields.filter((f) => f.kind === 'array<string>').map((f) => f.path);
  const L = m.layout;
  const setLayout = (patch: any) => setM({ ...m, layout: { ...L, ...patch } });
  const kinds: [string, string][] = [['conversation', 'Conversation (list of messages)'], ['string_turns', 'List of strings (alternating turns)'], ['pair', 'Prompt → response'], ['text', 'Raw text'], ['preformatted', 'Text that already has a chat template']];

  const changeKind = (k: any) => {
    if (k === L.kind) return;
    const first = (a: string[]) => a[0] ?? '';
    const layout =
      k === 'conversation' ? { kind: k, path: first(arrays), content_key: 'content', role: { how: 'key', key: 'role' } }
      : k === 'string_turns' ? { kind: k, path: first(arrStr) }
      : k === 'pair' ? { kind: k, prompt: [first(strings)], response: strings[1] ?? first(strings) }
      : { kind: k, path: first(strings) };
    setM({ ...m, layout });
  };
  const apply = async (mapping: any) => {
    try { await call('update_source', { id: src.id, mapping }); notify('Mapping updated', 'Compute the mix again to apply it.'); onOpenChange(false); refresh(); } catch (e: any) { notify('Could not update', e.message); }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <Dialog.Sheet title={`Mapping · ${src.id}`} description="Tell the mixer which fields hold the conversation. Auto-detection already picked the most likely one; change it only if it is wrong." maxContentWidth="76ch" showCloseButton>
        <Stack rowGap="l" fullWidth>
          {det.candidates.length > 1 ? (
            <Pick label="Other interpretations found" placeholder="Pick a suggestion…" value="" options={det.candidates.map((c: any, i: number) => ({ value: String(i), label: `${Math.round(c.confidence * 100)}% · ${c.summary}` }))} onChange={(v) => setM(JSON.parse(JSON.stringify(det.candidates[Number(v)])))} />
          ) : null}

          <Pick label="The data is" value={L.kind} options={kinds.map(([value, label]) => ({ value, label }))} onChange={changeKind} />

          {L.kind === 'conversation' ? (
            <Stack rowGap="m">
              <PathSelect label="Messages list" value={L.path} options={arrays} allowNone={false} onChange={(v) => setLayout({ path: v ?? '' })} />
              <Stack direction="row" columnGap="m" expandChildren>
                <Textfield label="Text key inside each message" value={L.content_key} onChange={(e) => setLayout({ content_key: e.target.value })} />
                <Pick label="Who is speaking" value={L.role.how} onChange={(v) => setLayout({ role: v === 'key' ? { how: 'key', key: 'role' } : v === 'bool' ? { how: 'bool', key: 'is_human' } : { how: 'alternating' } })}
                  options={[{ value: 'key', label: 'A text field (role, from, speaker…)' }, { value: 'bool', label: 'A true/false field (is_human…)' }, { value: 'alternating', label: 'Alternating: user, assistant, user…' }]} />
              </Stack>
              {L.role.how !== 'alternating' ? <Textfield label="Speaker key" description={L.role.how === 'bool' ? 'true means the human side' : 'values like user / human / assistant / gpt, or speaker names'} value={L.role.key} onChange={(e) => setLayout({ role: { ...L.role, key: e.target.value } })} /> : null}
            </Stack>
          ) : null}
          {L.kind === 'string_turns' ? <PathSelect label="List of strings" value={L.path} options={arrStr} allowNone={false} onChange={(v) => setLayout({ path: v ?? '' })} /> : null}
          {L.kind === 'pair' ? (
            <Stack rowGap="m">
              <Stack direction="row" columnGap="m" expandChildren>
                <PathSelect label="Prompt field" value={L.prompt[0]} options={strings} allowNone={false} onChange={(v) => setLayout({ prompt: [v ?? '', ...L.prompt.slice(1)].filter(Boolean) })} />
                <PathSelect label="Extra prompt field" description="e.g. input / context" value={L.prompt[1]} options={strings} onChange={(v) => setLayout({ prompt: [L.prompt[0], ...(v ? [v] : [])] })} />
              </Stack>
              <PathSelect label="Response field" value={L.response} options={strings} allowNone={false} onChange={(v) => setLayout({ response: v ?? '' })} />
            </Stack>
          ) : null}
          {L.kind === 'text' || L.kind === 'preformatted' ? <PathSelect label="Text field" value={L.path} options={strings} allowNone={false} onChange={(v) => setLayout({ path: v ?? '' })} /> : null}

          <Stack rowGap="s">
            <Title size="6" render={<h3 />}>Character & system fields</Title>
            <Text size="s" tone="muted">Roleplay datasets keep the character card in separate columns. They are turned into a system prompt and {'{{char}}'} / {'{{user}}'} get real names.</Text>
            <Stack direction="row" wrap columnGap="m" rowGap="m" expandChildren>
              {PERSONA_FIELDS.map(([k, l]) => <PathSelect key={k} label={l} value={m[k]} options={strings} onChange={(v) => setM({ ...m, [k]: v })} />)}
            </Stack>
          </Stack>

          {raw ? (
            <Stack rowGap="xs">
              <Text size="s" weight="semibold">First row, as stored</Text>
              <Code maxHeight="14rem">{raw}</Code>
            </Stack>
          ) : null}
          <Stack direction="row" columnGap="s" hAlign="space-between" wrap>
            <Button variant="flat" onClick={() => apply(null)}>Reset to auto-detected</Button>
            <Stack direction="row" columnGap="s">
              <Dialog.Close render={<Button variant="secondary" />}>Cancel</Dialog.Close>
              <Button onClick={() => apply(m)}>Apply mapping</Button>
            </Stack>
          </Stack>
        </Stack>
      </Dialog.Sheet>
    </Dialog>
  );
}

// ───────────────────────── source card ─────────────────────────

function SourceCard({ src }: { src: any }) {
  const { call, refresh, notify, jobById } = useStore();
  const [mapOpen, setMapOpen] = useState(false);
  const job = jobById(src.job);
  const best = src.effective_mapping ?? src.detection?.best;
  const loc = src.location;
  const where = loc.kind === 'hf' ? `huggingface.co/datasets/${loc.repo}` : loc.path;

  return (
    <Card
      title={src.id}
      description={<Text size="s" tone="muted" family="mono" whiteSpace="nowrap" style={{ overflowWrap: 'anywhere', whiteSpace: 'normal' }}>{where}</Text>}
      actions={
        <>
          {src.status === 'ready' ? <Switch aria-label={`Use ${src.id}`} checked={src.enabled ?? true} onCheckedChange={async (v) => { await call('update_source', { id: src.id, enabled: v }); refresh(); }} /> : null}
          {src.status === 'ready' ? <IconButton aria-label="Edit mapping" icon={<Sliders aria-hidden />} variant="flat" onClick={() => setMapOpen(true)} /> : null}
          <IconButton aria-label="Remove source" icon={<Trash aria-hidden />} variant="flat" onClick={async () => { await call('remove_source', { id: src.id }); notify('Source removed', src.id); refresh(); }} />
        </>
      }
    >
      {src.status === 'opening' ? <JobProgress job={job} label={`Opening ${src.id}`} /> : null}
      {src.status === 'error' ? (
        <Stack rowGap="s">
          <Text tone="danger" size="s">{src.error}</Text>
          <Stack direction="row"><Button size="s" variant="secondary" addon={<ArrowRotate aria-hidden />} onClick={async () => { await call('retry_source', { id: src.id }); refresh(); }}>Try again</Button></Stack>
        </Stack>
      ) : null}
      {src.status === 'ready' ? (
        <Stack rowGap="m">
          <Stack direction="row" columnGap="xl" rowGap="m" wrap>
            <Stat label="Rows" value={fmtNum(src.rows)} />
            <Stat label="Size" value={fmtBytes(src.bytes)} />
            <Stat label="Format" value={String(src.format ?? '').replace('_', ' ')} />
          </Stack>
          <Surface color={2} radius="m" hPadding="m" vPadding="m">
            <Stack rowGap="s">
              <Stack direction="row" columnGap="s" vAlign="center" wrap>
                <ConfidenceChip value={best?.confidence ?? 0} />
                {src.mapping_is_custom ? <Chip variant="outline">custom mapping</Chip> : <Chip variant="outline">auto-detected</Chip>}
              </Stack>
              <Text size="s" family="mono" style={{ overflowWrap: 'anywhere' }}>{best?.summary}</Text>
              {(best?.confidence ?? 1) < 0.8 ? <Text size="s" tone="warning">Not sure about this one. Open the mapping editor to check the fields.</Text> : null}
            </Stack>
          </Surface>
        </Stack>
      ) : null}
      {mapOpen ? <MappingEditor src={src} open={mapOpen} onOpenChange={setMapOpen} /> : null}
    </Card>
  );
}

// ───────────────────────── page ─────────────────────────

function looksLikeHf(s: string) {
  const t = s.trim();
  return /^(hf:|https?:\/\/(www\.)?huggingface\.co)/i.test(t) || (/^[\w.-]+\/[\w.-]+(\/.*)?$/.test(t) && !t.startsWith('.') && !t.startsWith('/') && !t.startsWith('~') && !/\.(jsonl?|csv|tsv|parquet|txt|gz|zst)$/i.test(t.split('/').slice(0, 2).join('/')));
}

export function Sources() {
  const { state, call, notify, refresh } = useStore();
  const [picker, setPicker] = useState(false);
  const [hf, setHf] = useState(false);
  const [hfInitial, setHfInitial] = useState('');
  const [quick, setQuick] = useState('');

  const sources: any[] = useMemo(() => state.sources.map((s: any, i: number) => ({ ...s, enabled: state.recipe.sources[i]?.enabled ?? true })), [state]);
  const totalRows = sources.filter((s) => s.enabled && s.status === 'ready').reduce((a, s) => a + s.rows, 0);

  const addFile = async (path: string) => {
    setPicker(false);
    try { await call('add_source', { kind: 'file', path }); refresh(); } catch (e: any) { notify('Could not add', e.message); }
  };
  const submitQuick = async () => {
    const t = quick.trim();
    if (!t) return;
    if (looksLikeHf(t)) { setHfInitial(t); setHf(true); setQuick(''); return; }
    await addFile(t);
    setQuick('');
  };

  return (
    <>
      <PageHeader
        title="Sources"
        description="Point the mixer at the datasets you want to include. The format is detected automatically: chat logs, instruction sets, roleplay cards, plain text, spreadsheets."
        actions={
          <>
            <Button variant="secondary" addon={<Cloud aria-hidden />} onClick={() => { setHfInitial(''); setHf(true); }}>Hugging Face</Button>
            <Button addon={<FolderOpen aria-hidden />} onClick={() => setPicker(true)}>Add file</Button>
          </>
        }
      />
      <Stack direction="row" columnGap="s" vAlign="end" fullWidth expandChildren>
        <Textfield label="Quick add" placeholder="/path/to/data.jsonl   or   PygmalionAI/PIPPA" value={quick} onChange={(e) => setQuick(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && submitQuick()} fullWidth />
        <Button variant="secondary" data-grow="false" onClick={submitQuick} disabled={!quick.trim()}>Add</Button>
      </Stack>

      {sources.length === 0 ? (
        <Empty title="No datasets yet" description="Add a file from your computer or paste a Hugging Face dataset id. Large files are fine: they are indexed on disk, not loaded into memory.">
          <Stack direction="row" columnGap="s" wrap>
            <Button variant="secondary" addon={<Cloud aria-hidden />} onClick={() => { setHfInitial('PygmalionAI/PIPPA'); setHf(true); }}>Try PIPPA</Button>
            <Button addon={<FolderOpen aria-hidden />} onClick={() => setPicker(true)}>Add file</Button>
          </Stack>
        </Empty>
      ) : (
        <>
          <Stack rowGap="m">{sources.map((s) => <SourceCard key={s.id} src={s} />)}</Stack>
          <Stack direction="row" hAlign="space-between" vAlign="center" wrap columnGap="m">
            <Text tone="muted" size="s">{sources.length} source{sources.length === 1 ? '' : 's'} · {fmtNum(totalRows)} rows available</Text>
            <Button addon={<Plus aria-hidden />} variant="flat" onClick={() => (location.hash = '/mix')}>Continue to Mix</Button>
          </Stack>
        </>
      )}
      <FilePicker open={picker} onOpenChange={setPicker} onPick={addFile} />
      {hf ? <HfDialog open={hf} onOpenChange={setHf} initial={hfInitial} /> : null}
    </>
  );
}
