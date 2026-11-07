import { useEffect, useState } from 'react';
import { Button, Chip, IconButton, Meter, NumberField, Slider, Stack, Surface, Switch, Text, Textarea, Textfield, ToggleGroup, ToggleTextButton } from '@viraui/react';
import { ArrowRight, Dice, Layers } from '@viraui/icons/react';
import { useStore } from '../store';
import { Card, Empty, JobProgress, PageHeader, Pick, Stat } from '../lib/ui';
import { fmtNum, fmtTokens } from '../lib/format';

const PT_TEMPLATE = '{{system}}\nVocê é {{char_name}}. {{description}}\n\n{{definitions}}\n\nCenário: {{scenario}}';
const EN_TEMPLATE = '{{system}}\nYou are {{char_name}}. {{description}}\n\n{{definitions}}\n\nScenario: {{scenario}}';

function num(v: number | null, fallback?: number) {
  return v == null || Number.isNaN(v) ? fallback : v;
}

function SourceRow({ s, plan, enabledCount }: { s: any; plan?: any; enabledCount: number }) {
  const { call, refresh } = useStore();
  const entry = s.entry;
  const [w, setW] = useState<number>(entry.weight);
  useEffect(() => setW(entry.weight), [entry.weight]);
  const commit = async (patch: any) => { await call('update_source', { id: s.id, ...patch }); refresh(); };
  const fixed = entry.count != null;
  return (
    <Surface color={2} radius="m" hPadding="m" vPadding="m">
      <Stack rowGap="m">
        <Stack direction="row" hAlign="space-between" vAlign="center" columnGap="m" wrap>
          <Stack direction="row" columnGap="s" vAlign="center" wrap>
            <Text weight="semibold">{s.id}</Text>
            <Chip variant="outline">{fmtNum(s.rows)} rows</Chip>
            {!entry.enabled ? <Chip variant="secondary">off</Chip> : null}
          </Stack>
          {plan ? (
            <Text size="s" tone="muted" family="mono">
              takes {fmtNum(plan.taken)} of {fmtNum(plan.available)} usable
            </Text>
          ) : null}
        </Stack>
        {entry.enabled ? (
          <Stack direction="row" columnGap="l" rowGap="m" vAlign="end" wrap>
            <Stack minWidth="16rem" data-grow="true">
              <Slider
                label={fixed ? 'Weight (ignored: fixed rows set)' : 'Weight (relative to the other sources)'}
                min={0}
                max={100}
                step={1}
                value={w}
                disabled={fixed}
                renderValue={(_f: any, v: any) => `${v}`}
                onValueChange={(v: any) => setW(Array.isArray(v) ? v[0] : v)}
                onValueCommitted={(v: any) => commit({ weight: Array.isArray(v) ? v[0] : v })}
              />
            </Stack>
            <NumberField
              label="Or exactly this many rows"
              min={0}
              step={100}
              value={entry.count ?? null}
              onValueChange={(v: number | null) => commit({ count: v == null ? null : Math.round(v) })}
              description="Empty = use the share"
            />
          </Stack>
        ) : null}
        {plan && entry.enabled ? (
          <Meter aria-label={`${s.id} usage`} value={plan.taken} min={0} max={Math.max(plan.available, plan.taken, 1)} />
        ) : null}
      </Stack>
    </Surface>
  );
}

function Result({ plan, stale }: { plan: any; stale: boolean }) {
  const labels = (plan.bounds as number[]).map((b) => `≤ ${b >= 1024 ? b / 1024 + 'K' : b}`).concat(['more']);
  const maxH = Math.max(...plan.histogram, 1);
  return (
    <Card
      title="Result"
      description={stale ? 'Settings changed since this was computed. Compute again to refresh.' : 'This is exactly what will be exported (before translation).'}
      actions={stale ? <Chip variant="yellow">out of date</Chip> : <Chip variant="green">up to date</Chip>}
    >
      <Stack rowGap="l">
        <Stack direction="row" columnGap="xl" rowGap="m" wrap>
          <Stat label="Rows" value={fmtNum(plan.total)} />
          <Stat label="Train" value={fmtNum(plan.train)} />
          <Stat label="Validation" value={fmtNum(plan.validation)} />
          <Stat label="Tokens (est.)" value={fmtTokens(plan.tokens_total)} hint={plan.total ? `${fmtNum(plan.tokens_total / plan.total)} per row` : undefined} />
        </Stack>
        <Stack rowGap="m">
          <Text weight="semibold" size="s">Where the rows come from</Text>
          {plan.per_source.map((p: any) => (
            <Stack key={p.id} rowGap="xs">
              <Meter label={p.id} renderValue={(_f: any, v: any) => `${fmtNum(p.taken)} rows · ${Math.round(plan.total ? (100 * p.taken) / plan.total : 0)}%`} value={p.taken} min={0} max={Math.max(plan.total, 1)} />
              <Text size="xs" tone="muted">
                {fmtNum(p.stats.total)} in file → {fmtNum(p.available)} usable
                {p.stats.duplicates ? ` · ${fmtNum(p.stats.duplicates)} duplicates` : ''}
                {p.stats.unmatched ? ` · ${fmtNum(p.stats.unmatched)} not recognised` : ''}
                {p.stats.malformed ? ` · ${fmtNum(p.stats.malformed)} broken lines` : ''}
                {p.stats.filtered_turns ? ` · ${fmtNum(p.stats.filtered_turns)} too short/long` : ''}
                {p.stats.filtered_tokens ? ` · ${fmtNum(p.stats.filtered_tokens)} outside token limits` : ''}
                {p.stats.filtered_language ? ` · ${fmtNum(p.stats.filtered_language)} wrong language` : ''}
                {p.stats.filtered_regex ? ` · ${fmtNum(p.stats.filtered_regex)} regex` : ''}
              </Text>
            </Stack>
          ))}
        </Stack>
        <Stack rowGap="s">
          <Text weight="semibold" size="s">Length of the rows (tokens)</Text>
          <Stack direction="row" columnGap="m" rowGap="s" wrap>
            {plan.histogram.map((h: number, i: number) => (
              <Stack key={i} minWidth="7rem" data-grow="true">
                <Meter label={labels[i]} value={h} min={0} max={maxH} renderValue={() => fmtNum(h)} />
              </Stack>
            ))}
          </Stack>
        </Stack>
      </Stack>
    </Card>
  );
}

export function Mix() {
  const { state, runJob, notify, setOptions, jobById, refresh } = useStore();
  const r = state.recipe;
  const [jobId, setJobId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const ready: any[] = state.sources.map((s: any, i: number) => ({ ...s, entry: r.sources[i] })).filter((s: any) => s.status === 'ready');
  const enabled = ready.filter((s) => s.entry.enabled);
  const plan = state.plan;
  const job = jobById(jobId);

  const compute = async () => {
    setBusy(true);
    try {
      await runJob('compute_mix');
      await refresh();
    } catch (e: any) { notify('Could not compute the mix', e.message); } finally { setBusy(false); }
  };
  useEffect(() => {
    // find the running job for the progress bar
    const j = state.jobs.filter((x: any) => x.kind === 'compute_mix' && x.state === 'running').pop();
    setJobId(j ? j.id : null);
  }, [state.jobs]);

  const set = (section: string, patch: any) => setOptions({ [section]: { ...r[section], ...patch } });

  if (!state.sources.length) {
    return (
      <>
        <PageHeader title="Mix" description="Decide how much of each source goes in, how it is shuffled, and what gets filtered out." />
        <Empty title="Add a source first" description="The mix needs at least one dataset.">
          <Button onClick={() => (location.hash = '/sources')}>Go to Sources</Button>
        </Empty>
      </>
    );
  }

  const strategy = r.mix.strategy as string;
  const planBySource = (id: string) => plan?.per_source.find((p: any) => p.id === id);

  return (
    <>
      <PageHeader
        title="Mix"
        description="Decide how much of each source goes in, how it is shuffled, and what gets filtered out."
        actions={
          <Button addon={<Layers aria-hidden />} loading={busy} disabled={!enabled.length || ready.length < state.sources.length} onClick={compute}>
            Compute mix
          </Button>
        }
      />
      {busy ? <JobProgress job={job} label="Computing mix" /> : null}
      {ready.length < state.sources.length ? <Text size="s" tone="warning">Some sources are still opening or failed. Fix them in Sources first.</Text> : null}

      <Card title="Composition" description="Each source gets a share of the final mix, or an exact number of rows. Rows are picked at random without repeating; the mix stops when the smallest source runs out unless you set a total.">
        <Stack rowGap="m">
          {ready.map((s) => <SourceRow key={s.id} s={s} plan={planBySource(s.id)} enabledCount={enabled.length} />)}
        </Stack>
        <Stack direction="row" columnGap="l" rowGap="m" wrap vAlign="end">
          <NumberField label="Total rows" description="Empty = as many as the shares allow" min={1} step={1000} value={r.mix.total ?? null} onValueChange={(v: number | null) => set('mix', { total: v == null ? null : Math.round(v) })} />
          <NumberField label="Repeat small sources up to" description="1 = never repeat rows" min={1} max={20} step={1} value={r.mix.max_upsample} onValueChange={(v: number | null) => set('mix', { max_upsample: Math.max(1, Math.round(num(v, 1)!)) })} />
          <Stack rowGap="xs">
            <Text size="s" weight="semibold">Balance</Text>
            <Surface border="all" radius="m" hPadding="2xs" vPadding="2xs">
              <ToggleGroup aria-label="Strategy" value={[strategy]} onValueChange={(v: string[]) => v[0] && set('mix', { strategy: v[0] })}>
                <ToggleTextButton value="proportional" variant="flat" pressedVariant="secondary" size="s" radius="auto">Use my shares</ToggleTextButton>
                <ToggleTextButton value="temperature" variant="flat" pressedVariant="secondary" size="s" radius="auto">Flatten big sources</ToggleTextButton>
              </ToggleGroup>
            </Surface>
          </Stack>
          {strategy === 'temperature' ? (
            <NumberField label="Temperature" description="Higher = flatter" min={1} max={10} step={0.5} value={r.mix.temperature} onValueChange={(v: number | null) => set('mix', { temperature: num(v, 1) })} />
          ) : null}
        </Stack>
      </Card>

      {plan ? <Result plan={plan} stale={plan.stale} /> : null}

      <Card title="Shuffle & split" description="The same seed always gives exactly the same mix.">
        <Stack direction="row" columnGap="l" rowGap="m" wrap vAlign="end">
          <Stack direction="row" columnGap="xs" vAlign="end">
            <NumberField label="Seed" min={0} step={1} value={r.seed} onValueChange={(v: number | null) => setOptions({ seed: Math.max(0, Math.round(num(v, 0)!)) })} />
            <IconButton aria-label="Random seed" icon={<Dice aria-hidden />} variant="secondary" onClick={() => setOptions({ seed: Math.floor(Math.random() * 1_000_000) })} />
          </Stack>
          <Pick label="Order" value={r.mix.shuffle} onChange={(v) => set('mix', { shuffle: v })} minWidth="14rem"
            options={[{ value: 'full', label: 'Fully shuffled' }, { value: 'interleave', label: 'Interleaved by source' }, { value: 'none', label: 'Keep file order' }]} />
          <Stack minWidth="16rem" data-grow="true">
            <Slider label="Validation split" min={0} max={20} step={1} value={Math.round(r.split.validation * 100)} renderValue={(_f: any, v: any) => `${v}%`} onValueChange={(v: any) => set('split', { validation: (Array.isArray(v) ? v[0] : v) / 100 })} />
          </Stack>
        </Stack>
      </Card>

      <Card title="Filters" description="Applied to every source before rows are picked. Token counts are estimates.">
        <Stack rowGap="m">
          <Stack direction="row" columnGap="l" rowGap="m" wrap>
            <NumberField label="Min turns" min={0} value={r.filters.min_turns} onValueChange={(v: number | null) => set('filters', { min_turns: Math.max(0, Math.round(num(v, 1)!)) })} description="PIPPA advises 3+" />
            <NumberField label="Max turns" min={1} value={r.filters.max_turns ?? null} onValueChange={(v: number | null) => set('filters', { max_turns: v == null ? null : Math.round(v) })} />
            <NumberField label="Min tokens" min={0} value={r.filters.min_tokens ?? null} onValueChange={(v: number | null) => set('filters', { min_tokens: v == null ? null : Math.round(v) })} />
            <NumberField label="Max tokens" min={1} value={r.filters.max_tokens ?? null} onValueChange={(v: number | null) => set('filters', { max_tokens: v == null ? null : Math.round(v) })} />
            <NumberField label="Split long chats into windows of" min={256} step={256} value={r.filters.window_tokens ?? null} onValueChange={(v: number | null) => set('filters', { window_tokens: v == null ? null : Math.round(v) })} description="Tokens. The system prompt is kept in every window" />
          </Stack>
          <Stack direction="row" columnGap="l" rowGap="m" wrap vAlign="end">
            <Pick label="Only keep language" value={r.filters.language ?? '__any'} onChange={(v) => set('filters', { language: v === '__any' ? null : v })}
              options={[{ value: '__any', label: 'Any language' }, ...[['eng', 'English'], ['por', 'Portuguese'], ['spa', 'Spanish'], ['fra', 'French'], ['deu', 'German'], ['ita', 'Italian'], ['rus', 'Russian'], ['jpn', 'Japanese'], ['kor', 'Korean'], ['cmn', 'Chinese']].map(([value, label]) => ({ value, label }))]} />
            <Textfield label="Must match (regex)" value={r.filters.include_regex ?? ''} onChange={(e) => set('filters', { include_regex: e.target.value || null })} />
            <Textfield label="Must not match (regex)" value={r.filters.exclude_regex ?? ''} onChange={(e) => set('filters', { exclude_regex: e.target.value || null })} />
          </Stack>
          <Stack direction="row" columnGap="xl" rowGap="s" wrap>
            <Switch label="Remove exact duplicates" checked={r.filters.dedup === 'exact'} onCheckedChange={(v) => set('filters', { dedup: v ? 'exact' : 'none' })} />
            <Switch label="Drop empty messages" checked={r.filters.drop_empty_messages} onCheckedChange={(v) => set('filters', { drop_empty_messages: v })} />
          </Stack>
        </Stack>
      </Card>

      <Card title="Characters & names" description="Roleplay datasets keep a character card next to the chat. It becomes the system prompt, and {{char}} / {{user}} are replaced with real names, after translation, so names are never translated.">
        <Stack rowGap="m">
          <Textarea label="System prompt template" rows={5} resize="vertical" value={r.persona.template} onChange={(e) => set('persona', { template: e.target.value })}
            description="Variables: {{system}} {{char_name}} {{user_name}} {{description}} {{definitions}} {{scenario}} {{greeting}}. Lines whose variables are all empty are dropped." />
          <Stack direction="row" columnGap="s" wrap>
            <Button size="s" variant="secondary" onClick={() => set('persona', { template: EN_TEMPLATE })}>English template</Button>
            <Button size="s" variant="secondary" onClick={() => set('persona', { template: PT_TEMPLATE })}>Portuguese template</Button>
          </Stack>
          <Switch label="Replace {{char}} and {{user}} with real names" checked={r.persona.substitute_names} onCheckedChange={(v) => set('persona', { substitute_names: v })} />
          <Stack direction="row" columnGap="l" rowGap="m" wrap>
            <Textfield label="Name for {{user}} when the dataset has none" value={r.persona.fallback_user_name} onChange={(e) => set('persona', { fallback_user_name: e.target.value })} />
            <Textfield label="Name for {{char}} when the dataset has none" value={r.persona.fallback_char_name} onChange={(e) => set('persona', { fallback_char_name: e.target.value })} />
          </Stack>
          <Textarea label="Random user names (optional)" rows={3} resize="vertical" placeholder={'Alex\nSam\nJordan'} value={(r.persona.user_name_pool ?? []).join('\n')}
            onChange={(e) => set('persona', { user_name_pool: e.target.value.split('\n').map((x: string) => x.trim()).filter(Boolean) })}
            description="One per line. Each conversation gets one of them, so the model does not learn a single user name." />
        </Stack>
      </Card>
      <Stack direction="row" hAlign="end">
        <Button variant="flat" addon={<ArrowRight aria-hidden />} addonPosition="end" onClick={() => (location.hash = '/preview')}>Preview rows</Button>
      </Stack>
    </>
  );
}
