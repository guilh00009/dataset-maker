import { useEffect, useState } from 'react';
import { Button, Chip, NumberField, Stack, Surface, Text, Textarea, Textfield, ToggleGroup, ToggleTextButton } from '@viraui/react';
import { Download, FolderOpen } from '@viraui/icons/react';
import { useStore } from '../store';
import { Card, Code, Empty, JobProgress, PageHeader, Pick, Stat } from '../lib/ui';
import { fmtBytes, fmtDuration, fmtNum } from '../lib/format';
import { FilePicker } from './Sources';

function formatPreview(f: any[] | null | undefined) {
  if (!f || !f.length) return '(no sample to show: compute the mix first)';
  const row = f[0];
  const keys = Object.keys(row).filter((k) => k !== 'original');
  return typeof row.text === 'string' && keys.length === 1 ? row.text : JSON.stringify(row, null, 2);
}

export function Export() {
  const { state, call, runJob, notify, setOptions, jobById, refresh } = useStore();
  const r = state.recipe;
  const e = r.export;
  const plan = state.plan;
  const set = (patch: any) => setOptions({ export: { ...e, ...patch } });
  const [sample, setSample] = useState<any[] | null>(null);
  const [pickDir, setPickDir] = useState(false);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [model, setModel] = useState('');
  const [loadingTpl, setLoadingTpl] = useState(false);
  const running = [...state.jobs].reverse().find((j: any) => j.kind === 'export' && (j.state === 'running' || j.state === 'paused'));
  const job = jobById(running?.id);

  useEffect(() => {
    if (!plan) { setSample(null); return; }
    const id = window.setTimeout(() => {
      call('preview', { offset: 0, count: 1 }).then((p) => setSample(p.items?.[0]?.formatted ?? null)).catch(() => setSample(null));
    }, 450);
    return () => window.clearTimeout(id);
  }, [call, plan?.total, plan?.stale, e.format, e.custom_template, e.bos_token, e.eos_token, r.persona.template, r.persona.substitute_names, r.filters.window_tokens]);

  const loadTemplate = async () => {
    setLoadingTpl(true);
    try {
      const looksLikePath = model.startsWith('/') || model.startsWith('~') || model.endsWith('.json');
      const t = await call('tokenizer_config', looksLikePath ? { path: model } : { repo: model });
      set({ custom_template: t.template, bos_token: t.bos_token, eos_token: t.eos_token, format: 'custom' });
      notify('Chat template loaded', model);
    } catch (err: any) { notify('Could not load the template', err.message); } finally { setLoadingTpl(false); }
  };

  const start = async () => {
    setBusy(true); setResult(null);
    try {
      const m = await runJob('export');
      setResult(m);
      notify('Export finished', `${fmtNum(m.rows_written)} rows written`);
    } catch (err: any) { if (err.message !== 'cancelled') notify('Export failed', err.message); } finally { setBusy(false); refresh(); }
  };

  if (!state.sources.length) {
    return (
      <>
        <PageHeader title="Export" description="Write the mix in the format your trainer expects." />
        <Empty title="Nothing to export yet" description="Add sources and compute the mix first."><Button onClick={() => (location.hash = '/sources')}>Go to Sources</Button></Empty>
      </>
    );
  }

  const ready = plan && !plan.stale;
  return (
    <>
      <PageHeader
        title="Export"
        description="Write the mix in the format your trainer expects, in one file or in shards."
        actions={<Button addon={<Download aria-hidden />} loading={busy} disabled={!ready} onClick={start}>Export</Button>}
      />
      {!plan ? <Text size="s" tone="warning">Compute the mix first (Mix page).</Text> : null}
      {plan?.stale ? <Text size="s" tone="warning">The mix changed since it was computed. Compute it again before exporting.</Text> : null}
      {r.translate.enabled ? (
        <Surface color={2} radius="m" hPadding="m" vPadding="m">
          <Stack direction="row" columnGap="s" vAlign="center" wrap>
            <Chip variant="green">translation on</Chip>
            <Text size="s">{r.translate.source_language} → {r.translate.target_language}. Anything not yet in the cache is translated during the export, using your server.</Text>
          </Stack>
        </Surface>
      ) : null}
      {running ? <JobProgress job={job} label="Exporting" /> : null}

      <Card title="Format" description="Chat templates are written as a single `text` field, ready for SFT trainers that expect a formatted string. Structured formats keep the messages as data.">
        <Stack rowGap="m">
          <Stack direction="row" columnGap="l" rowGap="m" wrap expandChildren>
            <Pick label="Format" value={e.format} onChange={(v) => set({ format: v })} minWidth="16rem" options={state.formats.map((f: any) => ({ value: f.id, label: f.label }))} />
            <Stack rowGap="xs">
              <Text size="s" weight="semibold">File type</Text>
              <Surface border="all" radius="m" hPadding="2xs" vPadding="2xs" style={{ alignSelf: 'start' }}>
                <ToggleGroup aria-label="File type" value={[e.container]} onValueChange={(v: string[]) => v[0] && set({ container: v[0] })}>
                  <ToggleTextButton value="jsonl" variant="flat" pressedVariant="secondary" size="s" radius="auto">JSONL</ToggleTextButton>
                  <ToggleTextButton value="json" variant="flat" pressedVariant="secondary" size="s" radius="auto">JSON</ToggleTextButton>
                  <ToggleTextButton value="parquet" variant="flat" pressedVariant="secondary" size="s" radius="auto">Parquet</ToggleTextButton>
                </ToggleGroup>
              </Surface>
            </Stack>
            <NumberField label="Rows per file" min={0} step={10000} value={e.shard_size ?? 0} onValueChange={(v: number | null) => set({ shard_size: v ? Math.round(v) : null })} description="0 = a single file" />
          </Stack>

          {e.format === 'custom' ? (
            <Stack rowGap="m">
              <Stack direction="row" columnGap="s" vAlign="end" wrap>
                <Textfield label="Load a model's chat template" placeholder="Qwen/Qwen2.5-7B-Instruct   or   /path/tokenizer_config.json" value={model} onChange={(ev) => setModel(ev.target.value)} />
                <Button variant="secondary" loading={loadingTpl} disabled={!model.trim()} onClick={loadTemplate}>Load</Button>
              </Stack>
              <Textarea label="Jinja chat template" rows={8} resize="vertical" value={e.custom_template ?? ''} onChange={(ev) => set({ custom_template: ev.target.value })} description="The same template syntax Hugging Face uses (messages, bos_token, eos_token)." />
              <Stack direction="row" columnGap="l" wrap>
                <Textfield label="BOS token" value={e.bos_token} onChange={(ev) => set({ bos_token: ev.target.value })} />
                <Textfield label="EOS token" value={e.eos_token} onChange={(ev) => set({ eos_token: ev.target.value })} />
              </Stack>
            </Stack>
          ) : null}

          <Stack rowGap="xs">
            <Text size="s" weight="semibold">What one row looks like</Text>
            <Code maxHeight="20rem">{formatPreview(sample)}</Code>
          </Stack>
        </Stack>
      </Card>

      <Card title="Destination">
        <Stack direction="row" columnGap="s" vAlign="end" wrap>
          <Textfield label="Output folder" value={e.out_dir} onChange={(ev) => set({ out_dir: ev.target.value })} description="Files: train, validation (if split), and manifest.json describing exactly how they were made." />
          <Button variant="secondary" addon={<FolderOpen aria-hidden />} onClick={() => setPickDir(true)}>Browse</Button>
        </Stack>
      </Card>
      <FilePicker open={pickDir} onOpenChange={setPickDir} mode="dir" title="Choose the output folder" startPath={e.out_dir} onPick={(p) => { set({ out_dir: p }); setPickDir(false); }} />

      {result ? (
        <Card title="Done" description={`Written to ${result.out_dir}`} actions={<Chip variant="green">{fmtNum(result.rows_written)} rows</Chip>}>
          <Stack rowGap="m">
            <Stack direction="row" columnGap="xl" rowGap="m" wrap>
              <Stat label="Rows" value={fmtNum(result.rows_written)} />
              <Stat label="Files" value={result.files.length} />
              <Stat label="Took" value={fmtDuration(result.duration_secs)} />
              {result.translation_failed ? <Stat label="Not translated" value={fmtNum(result.translation_failed)} /> : null}
            </Stack>
            <Surface color={2} radius="m" hPadding="m" vPadding="m">
              <Stack rowGap="xs">
                {result.files.map((f: any) => (
                  <Stack key={f.name} direction="row" columnGap="m" hAlign="space-between" wrap>
                    <Text size="s" family="mono">{f.name}</Text>
                    <Text size="s" tone="muted" family="mono">{fmtNum(f.rows)} rows · {fmtBytes(f.bytes)} · {f.xxh3.slice(0, 8)}</Text>
                  </Stack>
                ))}
                <Text size="xs" tone="muted" family="mono">manifest.json</Text>
              </Stack>
            </Surface>
          </Stack>
        </Card>
      ) : null}
    </>
  );
}
