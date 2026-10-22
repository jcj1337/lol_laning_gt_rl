// The typed API client. Every path is same-origin (/api), every shape comes from the generated schema
// (src/api/schema.d.ts, from web/openapi.json, from the FastAPI app), and every failure is an ApiError that carries
// the server's status and reason, or status 0 when the server could not be reached or did not answer in time.
import type { components } from './schema';

type S = components['schemas'];

export class ApiError extends Error {
  constructor(public readonly status: number, public readonly kind: string, message: string) {
    super(message);
    this.name = 'ApiError';
  }
  /** The request never produced an answer: the server may or may not have acted on it. */
  get unanswered(): boolean { return this.status === 0; }
}

const TIMEOUT_MS = 15_000;

async function call<T>(path: string, init: RequestInit & { timeout?: number } = {}): Promise<T> {
  const { timeout, ...rest } = init;
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), timeout ?? TIMEOUT_MS);
  let r: Response;
  try {
    r = await fetch(path, { ...rest, signal: ctl.signal, cache: 'no-store' });
  } catch {
    throw new ApiError(0, ctl.signal.aborted ? 'timeout' : 'network',
      ctl.signal.aborted ? 'The server did not answer in time.' : 'The server could not be reached.');
  } finally {
    clearTimeout(timer);
  }
  let body: unknown = null;
  try { body = await r.json(); } catch { /* not JSON */ }
  if (!r.ok) {
    const e = (body ?? {}) as Partial<S['ErrorBody']>;
    throw new ApiError(r.status, e.error ?? `http_${r.status}`, e.message ?? `HTTP ${r.status}`);
  }
  return body as T;
}

const get = <T>(path: string, timeout?: number) => call<T>(path, { timeout });
const post = <T>(path: string, body: unknown, timeout?: number) =>
  call<T>(path, { method: 'POST', body: JSON.stringify(body), headers: { 'Content-Type': 'application/json' }, timeout });

export const api = {
  status: () => get<S['Status']>('/api/status'),
  createSession: (body: S['CreateRequest']) => post<S['CreateResponse']>('/api/sessions', body),
  view: (sid: string) => get<S['View']>(`/api/sessions/${sid}`),
  submitTurn: (sid: string, turn: number, code: number) =>
    post<S['TurnResult']>(`/api/sessions/${sid}/turns`, { turn, code } satisfies S['SubmitRequest']),
  history: (sid: string, from = 0, to?: number) =>
    get<S['History']>(`/api/sessions/${sid}/history?from=${from}${to === undefined ? '' : `&to=${to}`}`),
  stop: (sid: string) => post<S['StopResponse']>(`/api/sessions/${sid}/stop`, {}),
  record: (sid: string) => get<S['Recording']>(`/api/sessions/${sid}/record`),
  simulate: (body: S['SimulateRequest']) => post<S['GameBundle']>('/api/simulations', body, 120_000),
  replays: () => get<S['ReplayEntry'][]>('/api/replays'),
  replay: (name: string) => get<S['GameBundle']>(`/api/replays/${encodeURIComponent(name)}`),
  guides: () => get<S['GuideEntry'][]>('/api/guides'),
  guide: (id: string) => get<S['Guide']>('/api/guides/' + encodeURIComponent(id)),
};

export type Api = typeof api;
