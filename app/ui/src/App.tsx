import { useEffect, useState } from 'react';
import { Button, IconButton, Stack, Surface, Text, Title, ToggleGroup, ToggleButton, Chip, Tooltip, LinearProgress } from '@viraui/react';
import { Folder, Layers, Eye, Language, Download, Gear, Sun, Moon, DarkLight, Shuffle } from '@viraui/icons/react';
import { useStore } from './store';
import { fmtDuration, pct } from './lib/format';
import { Sources } from './pages/Sources';
import { Mix } from './pages/Mix';
import { Preview } from './pages/Preview';
import { Translate } from './pages/Translate';
import { Export } from './pages/Export';
import { Settings } from './pages/Settings';

const ROUTES = [
  { id: 'sources', label: 'Sources', icon: <Folder aria-hidden />, el: <Sources /> },
  { id: 'mix', label: 'Mix', icon: <Layers aria-hidden />, el: <Mix /> },
  { id: 'preview', label: 'Preview', icon: <Eye aria-hidden />, el: <Preview /> },
  { id: 'translate', label: 'Translate', icon: <Language aria-hidden />, el: <Translate /> },
  { id: 'export', label: 'Export', icon: <Download aria-hidden />, el: <Export /> },
  { id: 'settings', label: 'Settings', icon: <Gear aria-hidden />, el: <Settings /> },
] as const;

function useRoute() {
  const get = () => (location.hash.replace(/^#\/?/, '') || 'sources').split('?')[0];
  const [r, setR] = useState(get);
  useEffect(() => {
    const on = () => setR(get());
    window.addEventListener('hashchange', on);
    return () => window.removeEventListener('hashchange', on);
  }, []);
  return [r, (id: string) => { location.hash = `/${id}`; }] as const;
}

export const go = (id: string) => { location.hash = `/${id}`; };

function ActiveJobs() {
  const { jobs } = useStore();
  const active = Object.values(jobs).filter((j) => (j.state === 'running' || j.state === 'paused') && j.kind !== 'open_source');
  if (!active.length) return null;
  return (
    <Surface color={2} radius="m" hPadding="m" vPadding="m">
      <Stack rowGap="m">
        {active.map((j) => {
          const u = j.update;
          const p = u && u.total > 0 ? pct(u.done, u.total) : null;
          return (
            <Stack key={j.id} rowGap="xs">
              <Stack direction="row" hAlign="space-between" columnGap="s">
                <Text size="s" weight="semibold">{j.label}</Text>
                {j.state === 'paused' ? <Chip variant="yellow">paused</Chip> : <Text size="xs" tone="muted" family="mono">{fmtDuration(j.elapsed_secs)}</Text>}
              </Stack>
              <LinearProgress aria-label={j.label} value={p as any} min={0} max={100} />
            </Stack>
          );
        })}
      </Stack>
    </Surface>
  );
}

function ThemeToggle() {
  const { state, call, refresh } = useStore();
  const theme = state?.settings?.theme ?? 'system';
  useEffect(() => {
    const el = document.documentElement;
    if (theme === 'light' || theme === 'dark') el.setAttribute('data-mode', theme);
    else el.removeAttribute('data-mode');
  }, [theme]);
  const set = async (t: string) => { await call('set_settings', { theme: t }); refresh(); };
  return (
    <Surface border="all" radius="m" hPadding="2xs" vPadding="2xs">
      <ToggleGroup aria-label="Color mode" value={[theme]} onValueChange={(v: string[]) => v[0] && set(v[0])}>
        <ToggleButton value="light" aria-label="Light" restingIcon={<Sun aria-hidden />} variant="flat" size="s" pressedVariant="secondary" radius="auto" />
        <ToggleButton value="dark" aria-label="Dark" restingIcon={<Moon aria-hidden />} variant="flat" size="s" pressedVariant="secondary" radius="auto" />
        <ToggleButton value="system" aria-label="System" restingIcon={<DarkLight aria-hidden />} variant="flat" size="s" pressedVariant="secondary" radius="auto" />
      </ToggleGroup>
    </Surface>
  );
}

export function App() {
  const [route, nav] = useRoute();
  const { state, error } = useStore();
  const current = ROUTES.find((r) => r.id === route) ?? ROUTES[0];
  const badge = (id: string): string | null => {
    if (!state) return null;
    if (id === 'sources') return state.sources.length ? String(state.sources.length) : null;
    return null;
  };
  return (
    <div className="app-shell">
      <nav className="app-nav" aria-label="Main">
        <Stack rowGap="l" hPadding="m" vPadding="m" fullHeight>
          <Stack direction="row" columnGap="s" vAlign="center" vPadding={[0, 's']}>
            <Surface color="primary" radius="m" hPadding="s" vPadding="s">
              <Shuffle aria-hidden />
            </Surface>
            <Stack rowGap="2xs">
              <Title size="6" render={<span />} weight={600}>Dataset Mixer</Title>
              <Text size="xs" tone="muted">mix · translate · export</Text>
            </Stack>
          </Stack>
          <Stack rowGap="2xs" fullWidth>
            {ROUTES.map((r) => {
              const b = badge(r.id);
              return (
                <Button
                  key={r.id}
                  variant={r.id === current.id ? 'secondary' : 'flat'}
                  size="m"
                  addon={r.icon}
                  className="nav-item"
                  onClick={() => nav(r.id)}
                  aria-current={r.id === current.id ? 'page' : undefined}
                >
                  {r.label}{b ? ` · ${b}` : ''}
                </Button>
              );
            })}
          </Stack>
          <Stack rowGap="m" vAlign="end" fullHeight>
            <ActiveJobs />
            <Stack direction="row" hAlign="space-between" vAlign="center">
              <Tooltip>
                <Tooltip.Trigger render={<span />}><Text size="xs" tone="muted">v0.1 · private</Text></Tooltip.Trigger>
                <Tooltip.Content>Desktop build · everything runs locally</Tooltip.Content>
              </Tooltip>
              <ThemeToggle />
            </Stack>
          </Stack>
        </Stack>
      </nav>
      <main className="app-main">
        <Stack hPadding="xl" vPadding="xl" rowGap="xl" maxWidth="80rem" hAlign="stretch">
          {error ? (
            <Surface color={2} border="all" radius="m" hPadding="m" vPadding="m">
              <Text tone="danger">{error}</Text>
            </Surface>
          ) : null}
          {state ? current.el : <Text tone="muted">Loading…</Text>}
        </Stack>
      </main>
    </div>
  );
}
