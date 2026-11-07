import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { Toast } from '@viraui/react';
import { backend, type JobView } from './backend';

export const toastManager = Toast.createToastManager();

type Store = {
  state: any | null;
  error: string | null;
  jobs: Record<string, JobView>;
  call: <T = any>(cmd: string, args?: any) => Promise<T>;
  /** Run a long command that returns { job_id } and resolve with the job result. */
  runJob: <T = any>(cmd: string, args?: any) => Promise<T>;
  refresh: () => Promise<void>;
  notify: (title: string, description?: string) => void;
  /** Debounced option updates: patch the recipe sections. */
  setOptions: (patch: Record<string, any>) => void;
  jobById: (id?: string | null) => JobView | undefined;
};

const Ctx = createContext<Store>(null as any);
export const useStore = () => useContext(Ctx);

export function StoreProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<any | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [jobs, setJobs] = useState<Record<string, JobView>>({});
  const waiters = useRef(new Map<string, { resolve: (v: any) => void; reject: (e: Error) => void }[]>());
  const jobsRef = useRef<Record<string, JobView>>({});
  const timer = useRef<number | null>(null);
  const pendingPatch = useRef<Record<string, any>>({});
  const patchTimer = useRef<number | null>(null);

  const refresh = useCallback(async () => {
    try {
      const s = await backend.call('get_state');
      // the recipe serializes its source list as `source` (TOML table name)
      s.recipe.sources = s.recipe.source ?? [];
      setState(s);
      setError(null);
      setJobs((old) => {
        const next = { ...old };
        for (const j of s.jobs as JobView[]) next[j.id] = { ...next[j.id], ...j };
        jobsRef.current = next;
        return next;
      });
    } catch (e: any) {
      setError(e.message ?? String(e));
    }
  }, []);

  const settle = useCallback((j: JobView) => {
    const w = waiters.current.get(j.id);
    if (!w || !['done', 'error', 'cancelled'].includes(j.state)) return;
    waiters.current.delete(j.id);
    for (const x of w) j.state === 'done' ? x.resolve(j.result) : x.reject(new Error(j.state === 'cancelled' ? 'cancelled' : j.error ?? 'failed'));
  }, []);

  useEffect(() => {
    refresh();
    const off = backend.onEvent((e) => {
      if (e.type === 'state') {
        if (timer.current) window.clearTimeout(timer.current);
        timer.current = window.setTimeout(refresh, 80);
      } else if (e.type === 'job') {
        const j = e as unknown as JobView & { type: string };
        setJobs((old) => {
          const next = { ...old, [j.id]: j };
          jobsRef.current = next;
          return next;
        });
        settle(j);
      }
    });
    const poll = window.setInterval(() => { if (document.visibilityState === 'visible') refresh(); }, 4000);
    return () => { off(); window.clearInterval(poll); };
  }, [refresh, settle]);

  const notify = useCallback((title: string, description?: string) => {
    toastManager.add({ title, description, timeout: 5000 });
  }, []);

  const call = useCallback(async <T,>(cmd: string, args: any = {}) => backend.call<T>(cmd, args), []);

  const runJob = useCallback(
    async <T,>(cmd: string, args: any = {}) => {
      const { job_id } = await backend.call<{ job_id: string }>(cmd, args);
      return new Promise<T>((resolve, reject) => {
        const existing = jobsRef.current[job_id];
        if (existing && ['done', 'error', 'cancelled'].includes(existing.state)) {
          return existing.state === 'done' ? resolve(existing.result) : reject(new Error(existing.error ?? existing.state));
        }
        const l = waiters.current.get(job_id) ?? [];
        l.push({ resolve, reject });
        waiters.current.set(job_id, l);
        // the job may have finished before we registered: re-check shortly
        window.setTimeout(refresh, 400);
      });
    },
    [refresh],
  );

  const setOptions = useCallback(
    (patch: Record<string, any>) => {
      // optimistic local update so inputs feel instant, then persist (debounced)
      setState((s: any) => (s ? { ...s, recipe: { ...s.recipe, ...patch } } : s));
      pendingPatch.current = { ...pendingPatch.current, ...patch };
      if (patchTimer.current) window.clearTimeout(patchTimer.current);
      patchTimer.current = window.setTimeout(async () => {
        const p = pendingPatch.current;
        pendingPatch.current = {};
        try { await backend.call('set_options', p); } catch (e: any) { notify('Could not save settings', e.message); refresh(); }
      }, 250);
    },
    [notify, refresh],
  );

  const jobById = useCallback((id?: string | null) => (id ? jobs[id] : undefined), [jobs]);

  const value = useMemo<Store>(() => ({ state, error, jobs, call, runJob, refresh, notify, setOptions, jobById }), [state, error, jobs, call, runJob, refresh, notify, setOptions, jobById]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}
