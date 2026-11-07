import { useState } from 'react';
import { Button, Stack, Surface, Text, Textfield } from '@viraui/react';
import { FolderOpen, Key, Trash } from '@viraui/icons/react';
import { useStore } from '../store';
import { Card, Code, PageHeader, Pick } from '../lib/ui';
import { fmtNum } from '../lib/format';
import { FilePicker } from './Sources';

export function Settings() {
  const { state, call, notify, refresh, setOptions } = useStore();
  const s = state.settings;
  const r = state.recipe;
  const [hf, setHf] = useState('');
  const [path, setPath] = useState('');
  const [pick, setPick] = useState<'load' | 'save' | null>(null);
  const [toml, setToml] = useState<string | null>(null);
  const [confirmNew, setConfirmNew] = useState(false);

  const saveHf = async () => {
    const res = await call('set_settings', { hf_token: hf });
    setHf('');
    notify('Hugging Face token saved', res.store === 'keychain' ? 'Stored in your system keychain.' : 'Stored in a private local file.');
    refresh();
  };
  const savePath = path.trim() || `${s.config_dir}/recipes/${(r.name || 'recipe').toLowerCase().replace(/[^a-z0-9]+/g, '-')}.toml`;
  const where = (x: string | null) => (x === 'keychain' ? 'your system keychain' : x === 'file' ? 'a private local file' : '');

  return (
    <>
      <PageHeader title="Settings" description="Appearance, access tokens, saved recipes and storage." />

      <Card title="Appearance">
        <Pick label="Theme" value={s.theme} minWidth="16rem" onChange={async (v) => { await call('set_settings', { theme: v }); refresh(); }}
          options={[{ value: 'system', label: 'Match my system' }, { value: 'light', label: 'Light' }, { value: 'dark', label: 'Dark' }]} />
      </Card>

      <Card title="Access tokens" description="Stored in your system keychain when there is one, otherwise in a private file in the config folder. Never written to recipes, exports or logs. The LLM token is on the Translate page.">
        <Stack rowGap="m">
          <Stack direction="row" columnGap="s" vAlign="end" wrap>
            <Textfield label="Hugging Face token" type="password" autoComplete="off" startAddon={<Key aria-hidden />} placeholder={s.has_hf_token ? '•••••••••••• (saved)' : 'hf_… (only for private or gated datasets)'} value={hf} onChange={(e) => setHf(e.target.value)}
              description={s.has_hf_token ? `Saved in ${where(s.hf_token_store)}.` : 'Public datasets like PIPPA do not need one.'} />
            <Button variant="secondary" disabled={!hf.trim()} onClick={saveHf}>Save</Button>
            {s.has_hf_token ? <Button variant="flat" onClick={async () => { await call('set_settings', { hf_token: null }); refresh(); }}>Remove</Button> : null}
          </Stack>
        </Stack>
      </Card>

      <Card title="Recipe" description="A recipe is everything about this mix in one small TOML file: sources, weights, filters, names, translation settings (never tokens). The command line runs it unchanged: mixer run recipe.toml">
        <Stack rowGap="m">
          <Textfield label="Name" value={r.name} onChange={(e) => setOptions({ name: e.target.value })} />
          <Stack direction="row" columnGap="s" vAlign="end" wrap>
            <Textfield label="File" placeholder={savePath} value={path} onChange={(e) => setPath(e.target.value)} />
            <Button variant="secondary" addon={<FolderOpen aria-hidden />} onClick={() => setPick('load')}>Browse…</Button>
          </Stack>
          <Stack direction="row" columnGap="s" wrap>
            <Button variant="secondary" onClick={async () => { const p = await call('save_recipe', { path: savePath }); notify('Recipe saved', p.path); }}>Save recipe</Button>
            <Button variant="secondary" disabled={!path.trim()} onClick={async () => { try { await call('load_recipe', { path }); notify('Recipe loaded', path); refresh(); } catch (e: any) { notify('Could not load', e.message); } }}>Load recipe</Button>
            <Button variant="flat" onClick={async () => setToml(toml == null ? (await call('recipe_toml')).toml : null)}>{toml == null ? 'Show as TOML' : 'Hide TOML'}</Button>
            <Button variant={confirmNew ? 'destructive' : 'flat'} onClick={async () => {
              if (!confirmNew) { setConfirmNew(true); window.setTimeout(() => setConfirmNew(false), 4000); return; }
              await call('new_recipe'); setConfirmNew(false); setToml(null); refresh();
            }}>{confirmNew ? 'Click again to start over' : 'New recipe'}</Button>
          </Stack>
          {toml != null ? <Code maxHeight="24rem">{toml}</Code> : null}
        </Stack>
      </Card>
      <FilePicker open={pick != null} onOpenChange={() => setPick(null)} title="Choose a recipe file" onPick={(p) => { setPath(p); setPick(null); }} />

      <Card title="Storage" description="Converted copies of your datasets, Hugging Face downloads and the translation cache live here. Safe to delete; they are rebuilt when needed.">
        <Stack rowGap="m">
          <Surface color={2} radius="m" hPadding="m" vPadding="m">
            <Stack rowGap="xs">
              <Text size="xs" tone="muted">Cache folder</Text>
              <Text size="s" family="mono" style={{ overflowWrap: 'anywhere' }}>{s.cache_dir}</Text>
              <Text size="xs" tone="muted">Config folder</Text>
              <Text size="s" family="mono" style={{ overflowWrap: 'anywhere' }}>{s.config_dir}</Text>
            </Stack>
          </Surface>
          <Stack direction="row" columnGap="m" vAlign="center" wrap>
            <Text size="s" tone="muted">{fmtNum(state.journal_units)} translated pieces cached</Text>
            <Button size="s" variant="flat" addon={<Trash aria-hidden />} disabled={!state.journal_units} onClick={async () => { await call('clear_translation_cache'); refresh(); notify('Translation cache cleared'); }}>Clear translation cache</Button>
          </Stack>
        </Stack>
      </Card>
    </>
  );
}
