import type { ReactNode } from 'react';
import { Stack, Surface, Text, Title, Chip, LinearProgress, Select } from '@viraui/react';
import type { JobView } from '../backend';
import { pct } from './format';

/** Page title block + primary actions. */
export function PageHeader({ title, description, actions }: { title: string; description?: ReactNode; actions?: ReactNode }) {
  return (
    <Stack direction="row" columnGap="l" rowGap="m" vAlign="end" hAlign="space-between" wrap fullWidth>
      <Stack rowGap="xs" maxWidth="62ch">
        <Title size="3" render={<h1 />}>{title}</Title>
        {description ? <Text tone="muted" size="m">{description}</Text> : null}
      </Stack>
      {actions ? <Stack direction="row" columnGap="s" rowGap="s" vAlign="center" wrap>{actions}</Stack> : null}
    </Stack>
  );
}

/** Titled card: Surface for chrome, Stack for rhythm. */
export function Card({ title, description, actions, children, color = 1 }: { title?: ReactNode; description?: ReactNode; actions?: ReactNode; children?: ReactNode; color?: 1 | 2 | 3 }) {
  return (
    <Surface color={color} border="all" radius="l" hPadding="l" vPadding="l">
      <Stack rowGap="m">
        {title || actions ? (
          <Stack direction="row" columnGap="m" vAlign="start" hAlign="space-between" wrap>
            <Stack rowGap="2xs">
              {title ? <Title size="5" render={<h2 />}>{title}</Title> : null}
              {description ? <Text size="s" tone="muted" maxWidth="70ch">{description}</Text> : null}
            </Stack>
            {actions ? <Stack direction="row" columnGap="s" vAlign="center">{actions}</Stack> : null}
          </Stack>
        ) : null}
        {children}
      </Stack>
    </Surface>
  );
}

export function Stat({ label, value, hint }: { label: string; value: ReactNode; hint?: ReactNode }) {
  return (
    <Stack rowGap="xs">
      <Text size="xs" tone="muted" transform="uppercase" tracking="0.04em">{label}</Text>
      <Text size="xl" weight="semibold" family="mono" whiteSpace="nowrap">{value}</Text>
      {hint ? <Text size="xs" tone="muted">{hint}</Text> : null}
    </Stack>
  );
}

export function ConfidenceChip({ value }: { value: number }) {
  const p = Math.round(value * 100);
  const variant = p >= 90 ? 'green' : p >= 60 ? 'yellow' : 'red';
  return <Chip variant={variant}>{p}% match</Chip>;
}

export function JobProgress({ job, label }: { job?: JobView; label?: string }) {
  if (!job) return null;
  const u = job.update;
  const known = u && u.total > 0;
  return (
    <Stack rowGap="xs" fullWidth>
      {known ? (
        <LinearProgress aria-label={label ?? job.label} value={pct(u!.done, u!.total)} min={0} max={100} />
      ) : (
        <LinearProgress aria-label={label ?? job.label} value={null as any} min={0} max={100} />
      )}
      <Text size="xs" tone="muted" whiteSpace="nowrap">{u?.message || job.label}</Text>
    </Stack>
  );
}

export function Empty({ title, description, children }: { title: string; description?: ReactNode; children?: ReactNode }) {
  return (
    <Surface border="all" borderStyle="dashed" radius="l" hPadding="xl" vPadding="xl">
      <Stack rowGap="m" hAlign="center" vAlign="center">
        <Title size="5" render={<h2 />}>{title}</Title>
        {description ? <Text tone="muted" align="center" maxWidth="56ch">{description}</Text> : null}
        {children}
      </Stack>
    </Surface>
  );
}

export function Mono({ children }: { children: ReactNode }) {
  return <Text family="mono" size="s">{children}</Text>;
}

/** Preformatted block in a Surface (no custom colours). */
export function Code({ children, maxHeight = '22rem' }: { children: ReactNode; maxHeight?: string }) {
  return (
    <Surface color={2} radius="m" hPadding="m" vPadding="m" overflow="auto" style={{ maxBlockSize: maxHeight }}>
      <Text render={<pre />} family="mono" size="s" lineHeight="m" whiteSpace="pre-wrap" style={{ margin: 0, overflowWrap: 'anywhere' }}>{children}</Text>
    </Surface>
  );
}

export type Opt = { value: string; label: string };

/** Select that shows option labels (not raw values) in the trigger. */
export function Pick({ label, value, onChange, options, description, minWidth = '12rem', placeholder }: { label?: ReactNode; value: string; onChange: (v: string) => void; options: Opt[]; description?: ReactNode; minWidth?: string; placeholder?: string }) {
  return (
    <Stack minWidth={minWidth}>
      <Select label={label} description={description} placeholder={placeholder} value={value} items={options} onValueChange={(v: any) => v != null && onChange(v)}>
        {options.map((o) => <Select.Option key={o.value} value={o.value}>{o.label}</Select.Option>)}
      </Select>
    </Stack>
  );
}
