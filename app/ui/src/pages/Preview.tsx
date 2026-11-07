import { useCallback, useEffect, useState } from 'react';
import { Button, Chip, IconButton, NumberField, Skeleton, Stack, Surface, Text, ToggleGroup, ToggleTextButton } from '@viraui/react';
import { ArrowLeft, ArrowRight } from '@viraui/icons/react';
import { useStore } from '../store';
import { Code, Empty, PageHeader, Pick } from '../lib/ui';
import { fmtNum } from '../lib/format';

const PAGE = 4;

function Bubble({ role, children }: { role: string; children: string }) {
  const isUser = role === 'user';
  const isSystem = role === 'system';
  return (
    <Stack hAlign={isUser ? 'end' : 'start'} fullWidth>
      <Surface
        color={isSystem ? 3 : isUser ? 2 : 1}
        border="all"
        radius="m"
        hPadding="m"
        vPadding="s"
        style={{ maxInlineSize: isSystem ? '100%' : '85%' }}
      >
        <Stack rowGap="xs">
          <Text size="xs" transform="uppercase" tracking="0.05em" tone="muted" weight="semibold">{role}</Text>
          <Text size="s" whiteSpace="pre-wrap" style={{ overflowWrap: 'anywhere' }}>{children}</Text>
        </Stack>
      </Surface>
    </Stack>
  );
}

function Chat({ messages }: { messages: { role: string; content: string }[] }) {
  return (
    <Stack rowGap="s" fullWidth>
      {messages.map((m, i) => <Bubble key={i} role={m.role}>{m.content}</Bubble>)}
    </Stack>
  );
}

function formattedText(f: any[] | null | undefined) {
  if (!f || !f.length) return '(this sample cannot be represented in the chosen format)';
  return f.map((row) => (typeof row.text === 'string' && Object.keys(row).filter((k) => k !== 'original').length === 1 ? row.text : JSON.stringify(row, null, 2))).join('\n\n— next window —\n\n');
}

function Item({ it, tab }: { it: any; tab: string }) {
  const msgs = it.original?.messages ?? [];
  const persona = it.persona;
  return (
    <Surface color={1} border="all" radius="l" hPadding="l" vPadding="l">
      <Stack rowGap="m">
        <Stack direction="row" columnGap="s" rowGap="xs" vAlign="center" wrap>
          <Chip variant="outline">#{fmtNum(it.position + 1)}</Chip>
          <Chip variant="secondary">{it.source}</Chip>
          <Text size="xs" tone="muted" family="mono">row {fmtNum(it.row)}</Text>
          {persona?.char_name ? <Chip variant="purple">{persona.char_name}</Chip> : null}
          {it.translated ? <Chip variant="green">translated</Chip> : null}
        </Stack>
        {tab === 'original' ? (
          it.original ? <Chat messages={msgs} /> : <Text tone="warning">This row could not be understood with the current mapping.</Text>
        ) : null}
        {tab === 'original' && !msgs.length && it.original?.text ? <Text whiteSpace="pre-wrap" size="s">{it.original.text}</Text> : null}
        {tab === 'formatted' ? <Code>{formattedText(it.formatted)}</Code> : null}
        {tab === 'translated' ? (
          it.translated ? (
            <Stack direction="row" columnGap="l" rowGap="l" wrap expandChildren vAlign="start">
              <Stack rowGap="s" minWidth="18rem">
                <Text size="s" weight="semibold" tone="muted">Original</Text>
                <Chat messages={msgs} />
              </Stack>
              <Stack rowGap="s" minWidth="18rem">
                <Text size="s" weight="semibold" tone="muted">Translation</Text>
                <Chat messages={it.translated.sample.messages} />
              </Stack>
            </Stack>
          ) : (
            <Stack rowGap="s">
              <Text tone="muted" size="s">Not translated yet. Run a trial on the Translate page and it will show up here.</Text>
              <Stack direction="row"><Button size="s" variant="secondary" onClick={() => (location.hash = '/translate')}>Go to Translate</Button></Stack>
            </Stack>
          )
        ) : null}
        {tab === 'raw' ? <Code>{JSON.stringify(it.raw, null, 2)}</Code> : null}
      </Stack>
    </Surface>
  );
}

export function Preview() {
  const { state, call, notify } = useStore();
  const plan = state.plan;
  const [scope, setScope] = useState<string>(plan ? '__mix' : state.sources.find((s: any) => s.status === 'ready')?.id ?? '__mix');
  const [offset, setOffset] = useState(0);
  const [tab, setTab] = useState('original');
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const ready = state.sources.filter((s: any) => s.status === 'ready');
  const usable = scope === '__mix' ? !!plan : ready.some((s: any) => s.id === scope);

  const load = useCallback(async () => {
    if (!usable) return;
    setLoading(true);
    try {
      const r = await call('preview', { source: scope === '__mix' ? undefined : scope, offset, count: PAGE });
      setData(r);
      setErr(null);
    } catch (e: any) { setErr(e.message); } finally { setLoading(false); }
  }, [call, scope, offset, usable]);

  useEffect(() => { load(); }, [load, plan?.total, plan?.stale, state.journal_units, state.recipe.export.format, state.recipe.persona.template, state.recipe.persona.substitute_names]);

  if (!ready.length) {
    return (
      <>
        <PageHeader title="Preview" description="Look at real rows before you export." />
        <Empty title="Nothing to preview" description="Add a source first."><Button onClick={() => (location.hash = '/sources')}>Go to Sources</Button></Empty>
      </>
    );
  }
  const total = data?.total ?? 0;
  const pages = Math.max(1, Math.ceil(total / PAGE));
  const page = Math.floor(offset / PAGE) + 1;

  return (
    <>
      <PageHeader title="Preview" description="Check how rows are understood, how they will look in the output format, and how they translate." />
      <Stack direction="row" columnGap="l" rowGap="m" vAlign="end" wrap>
        <Pick label="Show rows from" value={scope} onChange={(v) => { setScope(v); setOffset(0); }} minWidth="18rem"
          options={[{ value: '__mix', label: plan ? `The mix (${fmtNum(plan.total)} rows)` : 'The mix (compute it first)' }, ...ready.map((s: any) => ({ value: s.id, label: `${s.id} · file order` }))]} />
        <NumberField label="Go to row" min={1} max={Math.max(total, 1)} value={offset + 1} onValueChange={(v: number | null) => v != null && setOffset(Math.max(0, Math.min(total - 1, Math.round(v) - 1)))} />
        <Stack direction="row" columnGap="xs" vAlign="center">
          <IconButton aria-label="Previous page" icon={<ArrowLeft aria-hidden />} variant="secondary" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))} />
          <Text size="s" tone="muted" family="mono" hPadding="s" whiteSpace="nowrap">{page} / {fmtNum(pages)}</Text>
          <IconButton aria-label="Next page" icon={<ArrowRight aria-hidden />} variant="secondary" disabled={offset + PAGE >= total} onClick={() => setOffset(offset + PAGE)} />
        </Stack>
      </Stack>

      <Surface border="all" radius="m" hPadding="2xs" vPadding="2xs" style={{ alignSelf: 'start' }}>
        <ToggleGroup aria-label="What to show" value={[tab]} onValueChange={(v: string[]) => v[0] && setTab(v[0])}>
          <ToggleTextButton value="original" variant="flat" pressedVariant="secondary" size="s" radius="auto">Original</ToggleTextButton>
          <ToggleTextButton value="formatted" variant="flat" pressedVariant="secondary" size="s" radius="auto">Output format</ToggleTextButton>
          <ToggleTextButton value="translated" variant="flat" pressedVariant="secondary" size="s" radius="auto">Translation</ToggleTextButton>
          <ToggleTextButton value="raw" variant="flat" pressedVariant="secondary" size="s" radius="auto">Raw row</ToggleTextButton>
        </ToggleGroup>
      </Surface>

      {err ? <Text tone="danger">{err}</Text> : null}
      {!usable ? <Empty title="Compute the mix first" description="The preview of the mix shows rows in their final shuffled order."><Button onClick={() => (location.hash = '/mix')}>Go to Mix</Button></Empty> : null}
      {usable && loading && !data ? <Skeleton height="12rem" radius="l" /> : null}
      {usable && data ? (
        <Stack rowGap="m">
          {scope === '__mix' && plan?.stale ? <Text size="s" tone="warning">The mix is out of date. Compute it again to see current settings.</Text> : null}
          {data.items.map((it: any) => <Item key={`${it.source}-${it.row}-${it.position}`} it={it} tab={tab} />)}
        </Stack>
      ) : null}
    </>
  );
}
