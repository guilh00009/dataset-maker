// Drives the real UI in headless Chromium (against `mixer serve` + the mock LLM) and takes screenshots.
// Usage: node scripts/screenshots.mjs <base-url> <out-dir> <demo-data-dir> <mock-llm-url>
import { createRequire } from 'node:module';
import path from 'node:path';
import fs from 'node:fs';
const require = createRequire(import.meta.url);
const { chromium } = require('playwright');

const [base, out, data, llm] = process.argv.slice(2);
fs.mkdirSync(out, { recursive: true });
const W = 1440;
const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: W, height: 900 }, deviceScaleFactor: 2, colorScheme: 'light' });
const page = await ctx.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
page.on('console', (m) => { if (m.type() === 'error') errors.push(`console: ${m.text()}`); });

let n = 0;
const shot = async (name, height = 900) => {
  await page.setViewportSize({ width: W, height });
  await page.waitForTimeout(450);
  const file = path.join(out, `${String(++n).padStart(2, '0')}-${name}.png`);
  await page.screenshot({ path: file });
  console.log('shot', file);
};
const nav = async (label) => { await page.locator('nav.app-nav').getByRole('button', { name: new RegExp(`^${label}`) }).click(); await page.waitForTimeout(300); };
const waitText = (t, timeout = 60000) => page.getByText(t).first().waitFor({ timeout });
const closeSheet = async () => { await page.keyboard.press('Escape'); await page.waitForTimeout(500); };

await page.goto(base);
await page.waitForSelector('text=Sources');
await shot('sources-empty');

// ── add the three demo datasets through "Quick add" ──
for (const f of ['pippa-sample.jsonl', 'sharegpt-sample.jsonl', 'alpaca-sample.json']) {
  await page.getByLabel('Quick add').fill(path.join(data, f));
  await page.getByLabel('Quick add').press('Enter');
  await page.waitForTimeout(400);
}
await page.waitForFunction(() => document.querySelectorAll('[aria-label^="Use "]').length >= 3 || document.body.innerText.includes('match'), null, { timeout: 60000 });
await page.waitForFunction(() => (document.body.innerText.match(/% match/g) || []).length >= 3, null, { timeout: 60000 });
await shot('sources', 1500);

// mapping editor
await page.getByRole('button', { name: 'Edit mapping' }).first().click();
await page.waitForTimeout(700);
await shot('mapping-editor', 1500);
await closeSheet();

// ── mix ──
await nav('Mix');
await page.getByRole('button', { name: 'Portuguese template' }).click();
await page.waitForTimeout(400);
await page.getByRole('button', { name: 'Compute mix' }).click();
await waitText('Result');
await page.waitForTimeout(600);
await shot('mix', 2600);

// ── preview ──
await nav('Preview');
await page.waitForSelector('text=PIPPA', { timeout: 1 }).catch(() => {});
await page.waitForTimeout(800);
await shot('preview-original', 1500);
await page.getByRole('button', { name: 'Output format' }).click();
await page.waitForTimeout(600);
await shot('preview-formatted', 1300);

// ── translate ──
await nav('Translate');
await page.getByLabel('Base URL').fill(llm);
await page.getByLabel('Model name').fill('mock-translator-7b');
await page.keyboard.press('Escape');
await page.waitForTimeout(300);
await page.getByRole('button', { name: 'Test connection' }).click();
await page.getByText('connected', { exact: true }).waitFor({ timeout: 30000 });
await page.waitForTimeout(500);
await shot('translate-connection', 2400);

await page.getByRole('main').getByRole('button', { name: 'Translate', exact: true }).click();
await waitText('Ready to translate');
await page.waitForTimeout(700);
await shot('translate-estimate', 1100);
await page.getByRole('button', { name: 'Translate everything' }).click();
await page.waitForTimeout(900);
await page.getByText('Run', { exact: true }).first().scrollIntoViewIfNeeded().catch(() => {});
await page.waitForTimeout(2500);
await shot('translate-running', 2200);
await page.waitForFunction(() => /done\b/i.test(document.body.innerText) && !document.body.innerText.includes('Pause'), null, { timeout: 180000 }).catch(() => {});
await page.waitForTimeout(1500);
await shot('translate-done', 2200);

// ── preview of the translation ──
await nav('Preview');
await page.getByRole('button', { name: 'Translation', exact: true }).click();
await page.waitForTimeout(900);
await shot('preview-translated', 1700);

// ── export ──
await nav('Export');
await page.waitForTimeout(800);
await shot('export', 1800);
await page.getByRole('main').getByRole('button', { name: 'Export', exact: true }).first().click();
await waitText('Written to', 120000);
await page.waitForTimeout(700);
await shot('export-done', 2000);

// ── settings ──
await nav('Settings');
await page.waitForTimeout(500);
await shot('settings', 1700);

// ── dark mode ──
await page.getByRole('button', { name: 'Dark' }).click();
await page.waitForTimeout(500);
await shot('dark-settings', 1700);
for (const [label, name, h] of [['Sources', 'dark-sources', 1500], ['Mix', 'dark-mix', 2600], ['Preview', 'dark-preview', 1500], ['Translate', 'dark-translate', 2400], ['Export', 'dark-export', 1800]]) {
  await nav(label);
  await page.waitForTimeout(700);
  await shot(name, h);
}
console.log(errors.length ? `PAGE ERRORS:\n${errors.join('\n')}` : 'no page errors');
await browser.close();
