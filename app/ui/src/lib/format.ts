export const fmtNum = (n: number | undefined | null) => (n == null ? '–' : Math.round(n).toLocaleString('en-US'));
export const fmtBytes = (b: number | undefined | null) => {
  if (b == null) return '–';
  const u = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  let v = b;
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return `${v >= 100 || i === 0 ? Math.round(v) : v.toFixed(1)} ${u[i]}`;
};
export const fmtDuration = (s: number | undefined | null) => {
  if (s == null || !isFinite(s)) return '–';
  if (s < 60) return `${Math.max(1, Math.round(s))}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${Math.round(s % 60)}s`;
  const h = Math.floor(m / 60);
  if (h < 48) return `${h}h ${m % 60}m`;
  return `${Math.floor(h / 24)}d ${h % 24}h`;
};
export const fmtTokens = (n: number | undefined | null) => {
  if (n == null) return '–';
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return String(Math.round(n));
};
export const pct = (a: number, b: number) => (b > 0 ? Math.min(100, (100 * a) / b) : 0);

export const LANGUAGES: { value: string; label: string }[] = [
  ['en', 'English'], ['pt-BR', 'Portuguese (Brazil)'], ['pt-PT', 'Portuguese (Portugal)'], ['es', 'Spanish'], ['fr', 'French'], ['de', 'German'],
  ['it', 'Italian'], ['nl', 'Dutch'], ['pl', 'Polish'], ['ru', 'Russian'], ['uk', 'Ukrainian'], ['tr', 'Turkish'], ['ar', 'Arabic'],
  ['hi', 'Hindi'], ['id', 'Indonesian'], ['vi', 'Vietnamese'], ['th', 'Thai'], ['ja', 'Japanese'], ['ko', 'Korean'], ['zh-CN', 'Chinese (Simplified)'], ['zh-TW', 'Chinese (Traditional)'],
  ['sv', 'Swedish'], ['da', 'Danish'], ['fi', 'Finnish'], ['no', 'Norwegian'], ['cs', 'Czech'], ['el', 'Greek'], ['he', 'Hebrew'], ['ro', 'Romanian'], ['hu', 'Hungarian'],
].map(([value, label]) => ({ value, label }));
