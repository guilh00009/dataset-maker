// The UI never talks to Tauri or HTTP directly: everything goes through `Backend`.
// Desktop app  -> TauriBackend (invoke + events)
// Browser/dev  -> HttpBackend  (POST /api/call/{cmd} + Server-Sent Events from `mixer serve`)
// A future hosted web version only needs the same HTTP routes.

export type JobView = {
  id: string;
  kind: string;
  label: string;
  state: 'running' | 'paused' | 'done' | 'error' | 'cancelled';
  update?: { phase: string; done: number; total: number; message: string; stats?: Record<string, any> } | null;
  result?: any;
  error?: string | null;
  elapsed_secs: number;
};

export type AppEvent = { type: 'job'; [k: string]: any } | { type: 'state' };

export interface Backend {
  call<T = any>(cmd: string, args?: any): Promise<T>;
  onEvent(cb: (e: AppEvent) => void): () => void;
  readonly kind: 'tauri' | 'http';
}

class HttpBackend implements Backend {
  kind = 'http' as const;
  async call<T = any>(cmd: string, args: any = {}): Promise<T> {
    const r = await fetch(`/api/call/${cmd}`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(args) });
    if (!r.ok) throw new Error(`The local server answered HTTP ${r.status}. Is "mixer serve" running?`);
    const j = await r.json();
    if (!j.ok) throw new Error(j.error);
    return j.result as T;
  }
  onEvent(cb: (e: AppEvent) => void) {
    let es: EventSource | null = null;
    let closed = false;
    const open = () => {
      if (closed) return;
      es = new EventSource('/api/events');
      es.onmessage = (m) => {
        try { cb(JSON.parse(m.data)); } catch { /* ignore */ }
      };
      es.onerror = () => {
        es?.close();
        setTimeout(open, 1500);
      };
    };
    open();
    return () => { closed = true; es?.close(); };
  }
}

class TauriBackend implements Backend {
  kind = 'tauri' as const;
  async call<T = any>(cmd: string, args: any = {}): Promise<T> {
    const { invoke } = await import('@tauri-apps/api/core');
    return (await invoke('call', { cmd, args })) as T;
  }
  onEvent(cb: (e: AppEvent) => void) {
    let un: (() => void) | null = null;
    let closed = false;
    import('@tauri-apps/api/event').then(({ listen }) =>
      listen<AppEvent>('mixer-event', (e) => cb(e.payload)).then((u) => {
        if (closed) u(); else un = u;
      }),
    );
    return () => { closed = true; un?.(); };
  }
}

export const backend: Backend = typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window ? new TauriBackend() : new HttpBackend();
