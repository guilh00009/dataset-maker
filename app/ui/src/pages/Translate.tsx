import { useEffect, useMemo, useState } from 'react';
import { Autocomplete, Button, Chip, Dialog, Gauge, GaugeArc, GaugeText, GaugeTrack, GaugeValue, Meter, NumberField, Stack, Surface, Switch, Text, Textarea, Textfield, ToggleGroup, ToggleTextButton } from '@viraui/react';
import { CirclePause, CirclePlay, CircleStop, Key, Language, Server, Trash } from '@viraui/icons/react';
import { useStore } from '../store';
import { Card, Empty, JobProgress, PageHeader, Pick, Stat } from '../lib/ui';
import { LANGUAGES, fmtDuration, fmtNum, fmtTokens, pct } from '../lib/format';

function LanguageSelect({ label, value, onChange }: { label: string; value: string; onChange: (v: string) => void }) {
  const known = LANGUAGES.some((l) => l.value === value);
  const options = [...(known ? [] : [{ value, label: value }]), ...LANGUAGES.map((l) => ({ value: l.value, label: `${l.label} (${l.value})` }))];
  return <Pick label={label} value={value} options={options} onChange={onChange} minWidth="16rem" />;
}

function Connection() {
  const { state, call, notify, refresh, setOptions } = useStore();
  const t = state.recipe.translate;
  const test = state.last_test;
  const [token, setToken] = useState('');
  const [testing, setTesting] = useState(false);
  const [models, setModels] = useState<string[]>(test?.models ?? []);
  const saved = state.settings.has_llm_token;
  const set = (patch: any) => setOptions({ translate: { ...t, ...patch } });

  const run = async () => {
    setTesting(true);
    try {
      const r = await call('llm_test', { base_url: t.base_url, model: t.model, token: token || undefined });
      if (r.models?.length) setModels(r.models);
      if (r.ok && !t.model && r.models?.length === 1) set({ model: r.models[0] });
      if (r.ok && r.slots && t.concurrency === 4 && r.slots !== 4) set({ concurrency: r.slots });
      refresh();
    } catch (e: any) { notify('Test failed', e.message); } finally { setTesting(false); }
  };
  const saveToken = async () => {
    const r = await call('set_settings', { llm_token: token });
    setToken('');
    notify('Token saved', r.store === 'keychain' ? 'Stored in your system keychain.' : 'Stored in a private file (no keychain available here).');
    refresh();
  };
  const removeToken = async () => { await call('set_settings', { llm_token: null }); refresh(); };

  return (
    <Card
      title="Connection"
      description="Any OpenAI-compatible server works: llama.cpp (llama-server), vLLM, Ollama, LM Studio, OpenRouter, OpenAI…"
      actions={test ? (test.ok ? <Chip variant="green">connected</Chip> : <Chip variant="red">not connected</Chip>) : undefined}
    >
      <Stack rowGap="m">
        <Stack direction="row" columnGap="m" rowGap="m" wrap expandChildren>
          <Textfield label="Base URL" placeholder="http://127.0.0.1:8080/v1" value={t.base_url} onChange={(e) => set({ base_url: e.target.value })} description="Local llama.cpp is usually http://127.0.0.1:8080/v1" />
          <Autocomplete
            label="Model name"
            placeholder="qwen2.5-14b-instruct"
            value={t.model}
            onValueChange={(v: string) => set({ model: v })}
            items={models}
            itemToStringValue={(m: string) => m}
            empty="Run “Test connection” to list the server's models, or type the name."
            description="Sent as the `model` field. llama.cpp accepts any name."
          >
            {(m: string) => <Autocomplete.Item key={m} value={m}>{m}</Autocomplete.Item>}
          </Autocomplete>
        </Stack>
        <Stack direction="row" columnGap="s" vAlign="end" wrap>
          <Textfield
            label="API token"
            type="password"
            autoComplete="off"
            placeholder={saved ? '•••••••••••• (saved)' : 'Optional · llama-server --api-key'}
            value={token}
            onChange={(e) => setToken(e.target.value)}
            description={saved ? `Saved in ${state.settings.llm_token_store === 'keychain' ? 'your system keychain' : 'a private local file'}. It is never written to recipes or exports.` : 'Leave empty if your server does not need one.'}
            startAddon={<Key aria-hidden />}
          />
          <Button variant="secondary" disabled={!token.trim()} onClick={saveToken}>Save token</Button>
          {saved ? <Button variant="flat" onClick={removeToken}>Remove</Button> : null}
          <Button addon={<Server aria-hidden />} loading={testing} onClick={run} disabled={!t.base_url.trim()}>Test connection</Button>
        </Stack>
        {test ? (
          <Surface color={2} radius="m" hPadding="m" vPadding="m">
            <Stack rowGap="s">
              {test.ok ? (
                <Stack direction="row" columnGap="xl" rowGap="s" wrap>
                  <Stat label="Latency" value={`${test.latency_ms} ms`} />
                  <Stat label="Speed" value={test.tokens_per_sec ? `${test.tokens_per_sec.toFixed(1)} tok/s` : '–'} />
                  <Stat label="Parallel slots" value={test.slots ?? '–'} hint={test.slots ? 'from llama.cpp' : 'not reported'} />
                  <Stat label="Models" value={test.models.length || '–'} />
                </Stack>
              ) : null}
              {test.sample_output ? <Text size="s" family="mono">“Translate to Portuguese: Hello, how are you today?” → {test.sample_output}</Text> : null}
              {test.error ? <Text tone="danger" size="s">{test.error}</Text> : null}
              {test.hint ? <Text tone={test.ok ? 'warning' : undefined} size="s">{test.hint}</Text> : null}
              {test.models_error && !test.error ? <Text tone="muted" size="xs">Could not list models ({test.models_error}); that is fine for some servers.</Text> : null}
            </Stack>
          </Surface>
        ) : null}
      </Stack>
    </Card>
  );
}

function Settings() {
  const { state, setOptions } = useStore();
  const t = state.recipe.translate;
  const slots = state.last_test?.slots;
  const set = (patch: any) => setOptions({ translate: { ...t, ...patch } });
  const n = (v: number | null, d = 0) => (v == null || Number.isNaN(v) ? d : Math.round(v));
  return (
    <>
      <Card title="Languages" description="Names, code, links, LaTeX and {{char}} / {{user}} placeholders are protected and never translated.">
        <Stack direction="row" columnGap="l" rowGap="m" wrap expandChildren>
          <LanguageSelect label="Translate from" value={t.source_language} onChange={(v) => set({ source_language: v })} />
          <LanguageSelect label="Translate to" value={t.target_language} onChange={(v) => set({ target_language: v })} />
        </Stack>
      </Card>

      <Card title="Limits" description="How hard to push the server. Requests are sent in parallel up to the limit and retried with backoff when the server is busy.">
        <Stack direction="row" columnGap="l" rowGap="m" wrap>
          <NumberField label="Parallel requests" min={1} max={256} value={t.concurrency} onValueChange={(v: number | null) => set({ concurrency: Math.max(1, n(v, 1)) })} description={slots ? `Your server reports ${slots} slots` : 'Match llama-server --parallel'} />
          <NumberField label="Requests per minute" min={0} value={t.requests_per_minute} onValueChange={(v: number | null) => set({ requests_per_minute: n(v) })} description="0 = no limit" />
          <NumberField label="Tokens per minute" min={0} step={1000} value={t.tokens_per_minute} onValueChange={(v: number | null) => set({ tokens_per_minute: n(v) })} description="0 = no limit" />
          <NumberField label="Retries" min={0} max={20} value={t.max_retries} onValueChange={(v: number | null) => set({ max_retries: n(v, 5) })} />
          <NumberField label="Timeout (s)" min={5} value={t.timeout_secs} onValueChange={(v: number | null) => set({ timeout_secs: Math.max(5, n(v, 600)) })} />
          <NumberField label="Context budget" min={512} step={512} value={t.context_budget} onValueChange={(v: number | null) => set({ context_budget: Math.max(512, n(v, 4096)) })} description="Tokens per request. Match llama-server -c divided by parallel slots" />
          <NumberField label="Max output tokens" min={0} step={256} value={t.max_output_tokens} onValueChange={(v: number | null) => set({ max_output_tokens: n(v) })} description="0 = automatic" />
          <NumberField label="Stop after tokens" min={0} step={100000} value={t.token_budget} onValueChange={(v: number | null) => set({ token_budget: n(v) })} description="0 = no budget" />
          <NumberField label="Temperature" min={0} max={1.5} step={0.1} value={t.temperature} onValueChange={(v: number | null) => set({ temperature: v ?? 0.2 })} />
        </Stack>
      </Card>

      <Card title="What to translate">
        <Stack rowGap="m">
          <Stack rowGap="xs">
            <Text size="s" weight="semibold">Messages from</Text>
            <Surface border="all" radius="m" hPadding="2xs" vPadding="2xs" style={{ alignSelf: 'start' }}>
              <ToggleGroup aria-label="Roles to translate" multiple value={t.roles} onValueChange={(v: string[]) => set({ roles: v })}>
                <ToggleTextButton value="system" variant="flat" pressedVariant="secondary" size="s" radius="auto">System</ToggleTextButton>
                <ToggleTextButton value="user" variant="flat" pressedVariant="secondary" size="s" radius="auto">User</ToggleTextButton>
                <ToggleTextButton value="assistant" variant="flat" pressedVariant="secondary" size="s" radius="auto">Assistant</ToggleTextButton>
              </ToggleGroup>
            </Surface>
          </Stack>
          <Switch label="Translate character cards" description="Description and example dialogue. Each unique card is translated once and reused." checked={t.translate_persona} onCheckedChange={(v) => set({ translate_persona: v })} />
          <Stack direction="row" columnGap="l" rowGap="m" wrap expandChildren>
            <Pick label="Result" value={t.output_mode} onChange={(v) => set({ output_mode: v })} minWidth="16rem" options={[{ value: 'replace', label: 'Replace the original text' }, { value: 'bilingual', label: 'Keep both (adds an “original” field)' }]} />
            <Pick label="If a row cannot be translated" value={t.on_fail} onChange={(v) => set({ on_fail: v })} minWidth="16rem" options={[{ value: 'drop', label: 'Leave it out of the export' }, { value: 'keep_original', label: 'Keep it in the original language' }]} />
            <Pick label="Structured output" value={t.structured} onChange={(v) => set({ structured: v })} minWidth="16rem" options={[{ value: 'auto', label: 'Automatic (JSON schema if supported)' }, { value: 'on', label: 'Always request a JSON schema' }, { value: 'off', label: 'Never (prompt only)' }]} />
          </Stack>
          <Stack direction="row" columnGap="l" rowGap="m" wrap vAlign="end">
            <Pick label="Thinking models" value={t.thinking} onChange={(v) => set({ thinking: v })} minWidth="20rem" description="Reasoning is always removed from the result"
              options={[{ value: 'off', label: 'Ask the server not to think (faster)' }, { value: 'allow', label: 'Let the model think (extra output budget)' }]} />
            {t.thinking === 'allow' ? <NumberField label="Tokens reserved for thinking" min={256} step={500} value={t.thinking_tokens} onValueChange={(v: number | null) => set({ thinking_tokens: Math.max(256, n(v, 3000)) })} description="Added to every request's output limit" /> : null}
          </Stack>
          <Textarea label="Extra instructions" rows={3} resize="vertical" placeholder="Use informal Brazilian Portuguese. Keep roleplay actions between *asterisks*." value={t.instructions} onChange={(e) => set({ instructions: e.target.value })} />
          <Textarea label="Glossary" rows={3} resize="vertical" placeholder={'shop = loja\nmana = mana'} value={t.glossary} onChange={(e) => set({ glossary: e.target.value })} description="One “term = translation” per line" />
        </Stack>
      </Card>
    </>
  );
}

function EstimateSheet({ open, onOpenChange, est, onStart }: { open: boolean; onOpenChange: (o: boolean) => void; est: any; onStart: (n?: number) => void }) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <Dialog.Sheet title="Ready to translate" description="This is what the run will do. Rows that were already translated before are reused for free." maxContentWidth="60ch" showCloseButton>
        {est ? (
          <Stack rowGap="l" fullWidth>
            <Stack direction="row" columnGap="xl" rowGap="m" wrap>
              <Stat label="Rows" value={fmtNum(est.samples)} />
              <Stat label="Requests" value={fmtNum(est.requests)} />
              <Stat label="Tokens in" value={fmtTokens(est.input_tokens)} />
              <Stat label="Tokens out" value={fmtTokens(est.output_tokens)} />
            </Stack>
            <Stack rowGap="xs">
              <Text size="s">
                <b>{Math.round(est.cached_fraction * 100)}%</b> of the messages are already in your translation cache.
                {est.persona_units ? ` ${fmtNum(est.persona_units)} character cards are translated once each.` : ''}
              </Text>
              <Text size="s" tone="muted">
                {est.eta_secs != null ? `Rough time at your server's measured speed: ${fmtDuration(est.eta_secs)}.` : 'Run “Test connection” to get a time estimate.'}
                {est.extrapolated ? ' (Estimated from a sample of the mix.)' : ''}
              </Text>
            </Stack>
            <Stack direction="row" columnGap="s" hAlign="end" wrap>
              <Dialog.Close render={<Button variant="flat" />}>Cancel</Dialog.Close>
              <Button variant="secondary" onClick={() => onStart(50)}>Try 50 rows first</Button>
              <Button onClick={() => onStart()}>Translate everything</Button>
            </Stack>
          </Stack>
        ) : null}
      </Dialog.Sheet>
    </Dialog>
  );
}

function RunPanel() {
  const { state, call, runJob, notify, refresh } = useStore();
  const t = state.recipe.translate;
  const plan = state.plan;
  const [est, setEst] = useState<any>(null);
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const jobs: any[] = state.jobs;
  const running = [...jobs].reverse().find((j) => j.kind === 'translate' && (j.state === 'running' || j.state === 'paused'));
  const last = [...jobs].reverse().find((j) => j.kind === 'translate' && ['done', 'error', 'cancelled'].includes(j.state));
  const shown = running ?? last;
  const stats = shown?.update?.stats?.stats ?? shown?.result?.stats;
  const u = shown?.update;
  const done = running ? u?.done ?? 0 : shown?.result?.done ?? u?.done ?? 0;
  const total = running ? u?.total ?? 0 : shown?.result?.total ?? u?.total ?? 0;
  const percent = pct(done, total);
  const remaining = stats && stats.tokens_per_sec > 0 && total > done && done >= 12 ? (shown.elapsed_secs / done) * (total - done) : null;

  const prepare = async () => {
    setBusy(true);
    try { setEst(await call('estimate_translation')); setOpen(true); } catch (e: any) { notify('Cannot estimate', e.message); } finally { setBusy(false); }
  };
  const start = async (n?: number) => {
    setOpen(false);
    try {
      const r: any = await runJob('start_translation', { only_first: n });
      if (r.stopped) notify('Translation stopped', r.stopped);
      else notify('Translation finished', `${fmtNum(r.done - r.failed)} translated${r.failed ? `, ${fmtNum(r.failed)} failed` : ''}.`);
    } catch (e: any) { if (e.message !== 'cancelled') notify('Translation failed', e.message); }
    refresh();
  };
  const ctl = async (cmd: string) => { await call(cmd, { job_id: shown.id }); refresh(); };
  const failures: any[] = stats?.recent_failures ?? [];
  const canRun = plan && !plan.stale && t.base_url.trim();

  return (
    <Card
      title="Run"
      description="Translate the whole mix now (it fills the cache; export then reuses it), or let the export translate on the fly."
      actions={
        running ? (
          <Stack direction="row" columnGap="s">
            {running.state === 'paused'
              ? <Button size="m" variant="secondary" addon={<CirclePlay aria-hidden />} onClick={() => ctl('resume_job')}>Resume</Button>
              : <Button size="m" variant="secondary" addon={<CirclePause aria-hidden />} onClick={() => ctl('pause_job')}>Pause</Button>}
            <Button size="m" variant="destructive" addon={<CircleStop aria-hidden />} onClick={() => ctl('cancel_job')}>Cancel</Button>
          </Stack>
        ) : (
          <Button addon={<Language aria-hidden />} loading={busy} disabled={!canRun} onClick={prepare}>Translate</Button>
        )
      }
    >
      {!plan ? <Text size="s" tone="warning">Compute the mix first (Mix page): translation runs on exactly the rows that will be exported.</Text> : null}
      {plan?.stale ? <Text size="s" tone="warning">The mix changed. Compute it again before translating.</Text> : null}
      {shown ? (
        <Stack rowGap="l">
          <Stack direction="row" columnGap="xl" rowGap="l" vAlign="center" wrap>
            <Stack minWidth="9rem" maxWidth="12rem">
              <Gauge value={percent} min={0} max={100} label="Translation progress">
                <GaugeTrack width={26} />
                <GaugeArc width={26} />
                <GaugeValue fontSize={72} weight="semibold" decimals={0} />
                <GaugeText fontSize={22} y={80}>{running ? (shown.state === 'paused' ? 'paused' : 'running') : shown.state}</GaugeText>
              </Gauge>
            </Stack>
            <Stack rowGap="m" data-grow="true" minWidth="16rem">
              <Stack direction="row" columnGap="xl" rowGap="m" wrap>
                <Stat label="Done" value={`${fmtNum(done)} / ${fmtNum(total)}`} />
                <Stat label="Failed" value={fmtNum(stats?.samples_failed ?? shown.result?.failed)} />
                <Stat label="Speed" value={stats?.tokens_per_sec ? `${stats.tokens_per_sec.toFixed(1)} tok/s` : '–'} />
                <Stat label="Time left" value={running ? fmtDuration(remaining) : fmtDuration(shown.elapsed_secs)} hint={running ? undefined : 'total'} />
              </Stack>
              {running ? <JobProgress job={shown} label="Translating" /> : null}
              {stats ? (
                <Meter label="Requests in flight" value={stats.in_flight ?? 0} min={0} max={Math.max(t.concurrency, 1)} renderValue={() => `${stats.in_flight ?? 0} / ${t.concurrency}`} />
              ) : null}
            </Stack>
          </Stack>
          {stats ? (
            <Stack direction="row" columnGap="xl" rowGap="s" wrap>
              <Text size="xs" tone="muted">{fmtNum(stats.requests)} requests</Text>
              <Text size="xs" tone="muted">{fmtNum(stats.retries)} retries</Text>
              <Text size="xs" tone="muted">{fmtNum(stats.units_cached)} from cache</Text>
              <Text size="xs" tone="muted">{fmtNum(stats.units_translated)} new</Text>
              <Text size="xs" tone="muted">{fmtTokens(stats.tokens_in)} tokens in · {fmtTokens(stats.tokens_out)} out</Text>
            </Stack>
          ) : null}
          {shown.state === 'error' || shown.result?.stopped ? <Text tone="danger" size="s">{shown.error ?? shown.result?.stopped}</Text> : null}
          {stats?.last_error && (running || shown.result?.failed) ? <Text size="xs" tone="muted">Last error: {stats.last_error}</Text> : null}
          {failures.length ? (
            <Stack rowGap="xs">
              <Text size="s" weight="semibold">Recent failures</Text>
              {failures.slice(-5).map((f: any, i: number) => <Text key={i} size="xs" family="mono" tone="muted">{f.source}:{f.row} · {f.reason}</Text>)}
            </Stack>
          ) : null}
        </Stack>
      ) : null}
      <EstimateSheet open={open} onOpenChange={setOpen} est={est} onStart={start} />
    </Card>
  );
}

export function Translate() {
  const { state, call, refresh, notify, setOptions } = useStore();
  const t = state.recipe.translate;
  const units = state.journal_units as number;
  const [confirmClear, setConfirmClear] = useState(false);
  useEffect(() => {
    if (!confirmClear) return;
    const id = window.setTimeout(() => setConfirmClear(false), 4000);
    return () => window.clearTimeout(id);
  }, [confirmClear]);

  return (
    <>
      <PageHeader title="Translate" description="Translate the whole mix into another language with your own model. Requests are sent in parallel within the limits you set, and every finished piece is saved, so a stopped run picks up where it left off." />
      <Connection />
      <RunPanel />
      <Settings />
      <Card title="Translate while exporting" description="With this on, Export translates whatever is not already in the cache. With it off, Export writes the original text (even if you translated it here).">
        <Stack rowGap="m">
          <Switch label="Use translations in the export" checked={t.enabled} onCheckedChange={(v) => setOptions({ translate: { ...t, enabled: v } })} />
          <Stack direction="row" columnGap="m" vAlign="center" wrap>
            <Text size="s" tone="muted">{fmtNum(units)} translated pieces in the cache</Text>
            <Button size="s" variant="flat" addon={<Trash aria-hidden />} disabled={!units} onClick={async () => {
              if (!confirmClear) return setConfirmClear(true);
              await call('clear_translation_cache'); setConfirmClear(false); notify('Translation cache cleared'); refresh();
            }}>{confirmClear ? 'Click again to confirm' : 'Clear cache'}</Button>
          </Stack>
        </Stack>
      </Card>
    </>
  );
}
