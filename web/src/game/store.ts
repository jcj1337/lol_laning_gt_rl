// The application's state and every action on it, outside React: one store that components read through
// useSyncExternalStore. It owns the current game and its controller, the live session, the UI flags, the layout
// decision, and the talk with the server. Three guarantees live here:
//   * stale responses never act: every request carries the generation it was made in, and a response from an
//     earlier game (or an earlier session) is dropped;
//   * an unanswered request never leaves the screen guessing: the client re-reads the server's view and appends what
//     it missed before it offers a retry (the server resolves each turn once, so a retry of the same code is safe);
//   * a reload resumes the session the server still holds, and a lost session offers a clear way out.
import { api as realApi, ApiError } from '../api/client';
import type { Api } from '../api/client';
import { Controller, Live } from './controller';
import { prefersReducedMotion } from './frames';
import type { BeliefMode, Game, ModelId, Names, ReplayEntry, Status, View } from './types';
import { gameFromBundle } from './types';
import { AGENTS, nameSets } from './words';
import type { Lane } from '../scene/lane';

export type Mode = 'play' | 'watch';
export type Layout = 'dock' | 'rail' | 'console';
export type LayoutChoice = Layout | 'auto';
export type Theme = 'auto' | 'light' | 'dark';
export const LAYOUTS: Layout[] = ['dock', 'rail', 'console'];
export const THEMES: Theme[] = ['auto', 'light', 'dark'];
export interface Selection { kind: 'lane' | 'recall' | 'ward' | null; wave: number | null; beh: number | null; pressure: boolean }
export interface LiveSession { sid: string; opp: ModelId; belief: BeliefMode; cap: number }
export interface Problem { kind: 'lost' | 'engine' | 'unanswered' | 'error'; message: string; retry?: () => void }

const KEYS = { theme: 'lanegame-demo-theme', ticks: 'lanegame-demo-ticks', session: 'lanegame-demo-session' };
export const empty = (): Selection => ({ kind: null, wave: null, beh: null, pressure: false });

const hasDOM = typeof document !== 'undefined';
const raf = (fn: () => void): void => { if (typeof requestAnimationFrame === 'function') requestAnimationFrame(fn); else fn(); };

function storage(): Storage | null { try { return window.localStorage; } catch { return null; } }
function loadJSON<T>(key: string): T | null { try { const v = storage()?.getItem(key); return v ? (JSON.parse(v) as T) : null; } catch { return null; } }
function saveJSON(key: string, v: unknown): void { try { if (v === null) storage()?.removeItem(key); else storage()?.setItem(key, JSON.stringify(v)); } catch { /* unavailable */ } }

export interface Params {
  mode: Mode; opp: ModelId; belief: BeliefMode; seed: number | null; game: string | null; view: number | null;
  layout: LayoutChoice; theme: Theme | null; ticks: 'auto' | 'manual' | null; still: boolean; history: boolean;
  graph: boolean; detail: string; speed: number; fresh: boolean;
}

export function readParams(search = typeof location === 'undefined' ? '' : location.search): Params {
  const q = new URLSearchParams(search);
  const num = (k: string): number | null => (q.has(k) && q.get(k) !== '' && !isNaN(+q.get(k)!) ? +q.get(k)! : null);
  return {
    mode: q.get('mode') === 'watch' ? 'watch' : 'play',
    opp: q.get('opp') === 'om:0' ? 'om:0' : 'sp:0',
    belief: q.get('belief') === 'shown' ? 'shown' : 'hidden',
    seed: num('seed'), game: q.get('game'), view: num('view'),
    layout: (LAYOUTS as string[]).includes(q.get('layout') ?? '') ? (q.get('layout') as Layout) : 'auto',
    theme: (THEMES as string[]).includes(q.get('theme') ?? '') ? (q.get('theme') as Theme) : null,
    ticks: q.get('ticks') === 'auto' ? 'auto' : q.get('ticks') === 'manual' ? 'manual' : null,
    still: q.has('still'), history: q.get('history') === 'open', graph: q.get('graph') === 'expanded',
    detail: q.get('detail') ?? '', speed: num('speed') ?? 1, fresh: q.has('fresh'),
  };
}

export class AppStore {
  // ---- observation
  private version = 0;
  private listeners = new Set<() => void>();
  subscribe = (fn: () => void): (() => void) => { this.listeners.add(fn); return () => { this.listeners.delete(fn); }; };
  getSnapshot = (): number => this.version;
  emit(): void { this.version++; for (const fn of this.listeners) fn(); }

  // ---- state
  params: Params;
  game: Game | null = null;
  ctrl: Controller | null = null;
  mode: Mode = 'play';
  live: LiveSession | null = null;
  engine: Status | null = null;
  replays: ReplayEntry[] = [];
  names: Names = { A: 'A', B: 'B' };
  short: Names = { A: 'A', B: 'B' };
  head: Names = { A: 'A', B: 'B' };
  sel: Selection = empty();
  busy = false;
  error = '';
  problem: Problem | null = null;
  loading: string | null = null;
  recovered: 'ok' | 'gone' | null = null;
  hover: number | null = null;
  events = false;
  histOpen = false;
  graphBig = false;
  detailsOpen = false;
  detailsSection: string | null = null;
  dialog: 'newGame' | 'newSim' | 'about' | null = null;
  layoutChoice: LayoutChoice = 'auto';
  layout: Layout = 'dock';
  arr: Layout = 'dock';
  phone = false;
  theme: Theme = 'auto';
  ticksAuto = false;
  forcedRun = false;
  still = false;
  lane: Lane | null = null;
  private gen = 0;
  private started = false;
  private unsubscribeCtrl: (() => void) | null = null;

  api: Api;

  constructor(params: Params = readParams(), apiImpl: Api = realApi) {
    this.params = params;
    this.api = apiImpl;
    this.still = params.still;
  }

  // ---- start-up
  async start(): Promise<void> {
    if (this.started) return;
    this.started = true;
    const P = this.params;
    this.applyTheme(P.theme ?? (loadJSON<Theme>(KEYS.theme) ?? 'auto'));
    this.layoutChoice = P.layout;
    this.ticksAuto = (P.ticks ?? loadJSON<string>(KEYS.ticks)) === 'auto';
    if (prefersReducedMotion() || P.still) document.body.classList.add('reduced');
    this.arrange(true);
    try { this.engine = await this.api.status(); } catch { this.engine = null; }
    try { this.replays = this.engine?.ready ? await this.api.replays() : []; } catch { this.replays = []; }
    this.emit();
    if (P.mode === 'watch') {
      if (P.game) await this.watch(P.game, P.view ?? 0);
      else if (this.replays.length) await this.watch(this.replays[0].name, P.view ?? 0);
      else this.showIdle();
    } else if (!P.fresh && await this.resume()) {
      /* the previous game continues */
    } else if (this.engine?.ready && (P.seed !== null || P.fresh || !this.recovered)) {
      await this.newGame(P.opp, P.belief, P.seed);
    } else {
      this.showIdle();
    }
    if (P.detail) this.openDetails(P.detail);
    if (P.graph) this.setGraphBig(true);
    if (P.history) this.setHistory(true);
    // `fresh` and `seed` act once: a reload of the page resumes the game the server still holds
    try {
      const u = new URL(location.href);
      if (u.searchParams.has('fresh') || u.searchParams.has('seed')) { u.searchParams.delete('fresh'); u.searchParams.delete('seed'); history.replaceState(null, '', u); }
    } catch { /* the address cannot change here */ }
    document.body.dataset.ready = '1';
  }

  /** No game on screen: the initial position of a research game as a backdrop, or nothing. */
  private async showIdle(): Promise<void> {
    this.mode = this.params.mode;
    this.live = null;
    if (this.replays.length) {
      try {
        const b = await this.api.replay(this.replays[0].name);
        const G: Game = { ...gameFromBundle(b), name: 'idle', positions: [b.positions[0]], turns: [], beliefs: null };
        this.install(G, new Controller(G, { mode: 'watch', view: 0, env: this.env() }));
        return;
      } catch { /* fall through */ }
    }
    this.emit();
  }

  /** Resume the session the browser remembers, if the server still holds it. */
  private async resume(): Promise<boolean> {
    const saved = loadJSON<LiveSession>(KEYS.session);
    if (!saved || !this.engine?.ready) return false;
    const gen = ++this.gen;
    try {
      const v = await this.api.view(saved.sid);
      const h = await this.api.history(saved.sid);
      if (gen !== this.gen) return true;
      const G = this.liveGame(v, h.positions, h.turns, h.beliefs ? { types: Object.values(h.beliefs)[0].types, posteriors: Object.fromEntries(Object.entries(h.beliefs).map(([s, b]) => [s, b.posteriors])) } : null);
      this.mode = 'play';
      this.live = { ...saved, cap: v.turn_cap };
      const c = new Live(G, v.turn_cap, { belief: v.belief_mode, speed: this.params.speed, env: this.env() });
      c.adopt(v);
      this.recovered = 'ok';
      this.install(G, c);
      return true;
    } catch (e) {
      if (gen !== this.gen) return true;
      if (e instanceof ApiError && e.status === 404) { saveJSON(KEYS.session, null); this.recovered = 'gone'; return false; }
      this.problem = { kind: e instanceof ApiError && e.unanswered ? 'unanswered' : 'error', message: (e as Error).message,
        retry: () => { this.problem = null; void this.resume().then((ok) => { if (!ok) void this.newGame(); }); } };
      this.emit();
      return true;
    }
  }

  private liveGame(v: View, positions: Game['positions'], turns: Game['turns'], beliefs: Game['beliefs']): Game {
    return {
      name: 'live', label: '', positions, turns, beliefs,
      meta: { provenance: { kind: 'human_game' }, seats: v.seats, seeds: v.seeds, marks: v.marks, belief_mode: v.belief_mode, turn_cap: v.turn_cap },
      lane_exit_position: v.lane_exit_position, read_position: v.read_position,
    };
  }

  env() { return { reduced: () => prefersReducedMotion() || this.still, ticksAuto: () => this.ticksAuto, forcedRun: () => this.forcedRun }; }

  // ---- games
  async newGame(opp: ModelId = 'sp:0', belief: BeliefMode = 'hidden', seed: number | null = null): Promise<void> {
    this.ctrl?.pause();
    const gen = ++this.gen;
    this.mode = 'play';
    this.loading = 'Starting a game…';
    this.problem = null;
    this.dialog = null;
    this.emit();
    try {
      const d = await this.api.createSession({ opponent: opp, belief, ...(seed !== null ? { seed } : {}) });
      if (gen !== this.gen) return;
      const v = d.view;
      const bel = v.belief ? { types: Object.values(v.belief)[0].types, posteriors: Object.fromEntries(Object.entries(v.belief).map(([s, b]) => [s, [b.posterior]])) } : null;
      const G = this.liveGame(v, d.history.positions, [], bel);
      this.live = { sid: d.session, opp, belief, cap: v.turn_cap };
      saveJSON(KEYS.session, this.live);
      this.recovered = null;
      const c = new Live(G, v.turn_cap, { belief, speed: this.ctrl ? this.ctrl.speed : this.params.speed, env: this.env() });
      c.adopt(v);
      this.install(G, c);
    } catch (e) {
      if (gen !== this.gen) return;
      this.failed(e, () => this.newGame(opp, belief, seed));
    } finally {
      if (gen === this.gen) { this.loading = null; this.emit(); }
    }
  }

  async watch(name: string, view = 0): Promise<void> {
    this.ctrl?.pause();
    const gen = ++this.gen;
    this.mode = 'watch';
    this.loading = 'Loading the game…';
    this.problem = null;
    this.dialog = null;
    this.emit();
    try {
      const b = await this.api.replay(name);
      if (gen !== this.gen) return;
      this.live = null;
      const G = gameFromBundle(b);
      this.install(G, new Controller(G, { mode: 'watch', view, speed: this.ctrl ? this.ctrl.speed : this.params.speed, env: this.env() }));
    } catch (e) {
      if (gen !== this.gen) return;
      this.failed(e, () => this.watch(name, view));
    } finally {
      if (gen === this.gen) { this.loading = null; this.emit(); }
    }
  }

  async simulate(a: ModelId = 'om:0', b: ModelId = 'sp:0'): Promise<void> {
    this.ctrl?.pause();
    const gen = ++this.gen;
    this.mode = 'watch';
    this.loading = 'The engine is playing 150 turns…';
    this.problem = null;
    this.dialog = null;
    this.emit();
    try {
      const bundle = await this.api.simulate({ a, b, turns: 150 });
      if (gen !== this.gen) return;
      this.live = null;
      const G = gameFromBundle(bundle);
      G.label = `New game: ${AGENTS[a].name} vs. ${AGENTS[b].name.toLowerCase()}`;
      this.install(G, new Controller(G, { mode: 'watch', view: 0, speed: this.ctrl ? this.ctrl.speed : this.params.speed, env: this.env() }));
    } catch (e) {
      if (gen !== this.gen) return;
      this.failed(e, () => this.simulate(a, b));
    } finally {
      if (gen === this.gen) { this.loading = null; this.emit(); }
    }
  }

  private failed(e: unknown, retry: () => void): void {
    const err = e instanceof ApiError ? e : null;
    const kind: Problem['kind'] = err?.status === 503 ? 'engine' : err?.status === 404 ? 'lost' : err?.unanswered ? 'unanswered' : 'error';
    this.problem = { kind, message: (e as Error).message, retry: () => { this.problem = null; this.emit(); retry(); } };
    this.emit();
  }

  private install(G: Game, ctrl: Controller): void {
    this.ctrl?.dispose();
    this.unsubscribeCtrl?.();
    this.game = G;
    this.ctrl = ctrl;
    this.sel = empty();
    this.error = '';
    this.busy = false;                                        // whatever the replaced game was doing is over
    this.hover = null;
    const sets = nameSets(G, this.mode === 'play' && !!this.live);
    this.names = sets.names; this.short = sets.short; this.head = sets.head;
    this.unsubscribeCtrl = ctrl.subscribe(() => this.emit());
    this.setDur();
    this.emit();
  }

  // ---- the live game
  /** The engine resolves the turn; the page then shows it (stepped, or on its own) and only after that offers the
   * next decision. */
  async submit(code: number | null): Promise<void> {
    const c = this.ctrl;
    if (!(c instanceof Live) || !c.canSubmit || this.busy || code === null || code === undefined || !this.live) return;
    const gen = this.gen, sid = this.live.sid, t = c.progress + 1;
    this.busy = true;
    this.error = '';
    this.problem = null;
    this.emit();
    try {
      let res;
      try {
        res = await this.api.submitTurn(sid, t, code);
      } catch (e) {
        if (gen !== this.gen) return;
        await this.reconcile(e, sid, gen, () => this.submit(code));
        return;
      }
      if (gen !== this.gen) return;
      this.sel = empty();
      const anim = c.append(res);
      this.emit();
      await anim;
      if (gen !== this.gen) return;
      if (c.stopped) await this.finish(sid, gen);
    } finally {
      if (gen === this.gen) { this.busy = false; this.emit(); }
    }
  }

  /** A submission failed or went unanswered: re-read the server before deciding what the visitor sees. */
  private async reconcile(e: unknown, sid: string, gen: number, retry: () => void): Promise<void> {
    const c = this.ctrl as Live;
    const err = e instanceof ApiError ? e : null;
    if (err?.status === 422) { this.error = err.message; return; }                 // illegal: nothing changed
    if (err?.status === 404) { this.lost(err.message); return; }
    try {
      const v = await this.api.view(sid);
      if (gen !== this.gen) return;
      const advanced = v.position > c.progress;
      if (advanced) {
        const h = await this.api.history(sid, c.progress, v.position);
        if (gen !== this.gen) return;
        c.appendHistory(h, v);
        this.sel = empty();
      } else c.adopt(v);
      if (c.stopped) { await this.finish(sid, gen); return; }
      if (err?.status === 409 || err?.status === 410) { this.error = err.message; return; }
      if (advanced) return;                                   // the server had resolved it; the turn is on screen now
      this.problem = { kind: 'unanswered', message: `${(e as Error).message} Nothing changed on the server; you can try the same action again.`,
        retry: () => { this.problem = null; this.emit(); retry(); } };
    } catch (e2) {
      if (gen !== this.gen) return;
      if (e2 instanceof ApiError && e2.status === 404) { this.lost(e2.message); return; }
      this.problem = { kind: 'unanswered', message: `${(e2 as Error).message} The game may or may not have advanced; reconnect to find out.`,
        retry: () => { this.problem = null; this.emit(); void this.reconcile(e2, sid, this.gen, retry); } };
    }
  }

  private lost(message: string): void {
    saveJSON(KEYS.session, null);
    this.problem = { kind: 'lost', message: `${message} Start a new game to continue.`, retry: () => { this.problem = null; this.dialog = 'newGame'; this.emit(); } };
  }

  async continueAway(): Promise<void> {
    this.forcedRun = true;
    try {
      while (this.ctrl instanceof Live && this.ctrl.forced() && !this.error && !this.problem) await this.submit(45);
    } finally {
      this.forcedRun = false;
    }
  }

  private async finish(sid: string, gen: number): Promise<void> {
    const d = await this.api.stop(sid);
    if (gen !== this.gen) return;
    const c = this.ctrl as Live;
    c.conclude(d.recording.beliefs ?? null, d.view);
    this.game!.recording = d.recording;
    this.emit();
  }

  async stopGame(): Promise<void> {
    const c = this.ctrl;
    if (!(c instanceof Live) || c.stopped || this.busy || !this.live) return;
    const gen = this.gen, sid = this.live.sid;
    this.busy = true;
    this.emit();
    try { await this.finish(sid, gen); } catch (e) { if (gen === this.gen) this.failed(e, () => this.stopGame()); } finally { if (gen === this.gen) { this.busy = false; this.emit(); } }
  }

  /** The visitor's own recording, as the server serializes it (replayable with lanegame.demo.session.replay). */
  async downloadRecord(): Promise<void> {
    if (!this.live) return;
    const sid = this.live.sid;
    try {
      const rec = await this.api.record(sid);
      const blob = new Blob([JSON.stringify(rec, null, 1)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      if (!hasDOM) return;
      const a = document.createElement('a');
      a.href = url;
      a.download = `laning-game-${sid.slice(0, 8)}.json`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (e) {
      this.failed(e, () => this.downloadRecord());
    }
  }

  // ---- selection
  selectionCode(): number | null {
    const m = this.ctrl?.menu(), s = this.sel;
    if (!m) return null;
    if (s.kind === 'recall') return 12;
    if (s.kind === 'ward') return 30;
    if (s.kind === 'lane' && s.wave !== null && s.beh !== null) return m.lane_codes[`${s.wave},${s.beh},${s.pressure ? 1 : 0}`] ?? null;
    return null;
  }
  choose(k: string): void {
    this.error = '';
    const s = this.sel.kind === 'lane' ? { ...this.sel } : empty();
    if (k === 'recall' || k === 'ward') this.sel = { ...empty(), kind: k };
    else {
      s.kind = 'lane';
      if (k[0] === 'w') s.wave = +k.slice(1);
      else if (k[0] === 'b') s.beh = +k.slice(1);
      else s.pressure = !s.pressure;
      if (s.pressure && s.beh !== null && ![0, 2].includes(s.beh)) s.beh = null;
      this.sel = s;
    }
    this.emit();
  }
  setSelection(sel: Partial<Selection>): void { this.sel = { ...empty(), ...sel }; this.emit(); }

  // ---- panels, layout, theme
  setHover(p: number | null): void { if (p !== this.hover) { this.hover = p; this.emit(); } }
  setEvents(on: boolean): void { this.events = on; this.emit(); }
  openDetails(section: string | null = null): void { this.detailsOpen = true; this.detailsSection = section; this.emit(); }
  closeDetails(): void { this.detailsOpen = false; this.emit(); }
  openDialog(d: AppStore['dialog']): void { this.dialog = d; this.emit(); }
  closeDialog(): void { this.dialog = null; this.emit(); }

  setGraphBig(big: boolean, quiet = false): void {
    this.graphBig = !!big;
    if (this.graphBig && this.arr === 'console' && this.histOpen) this.setHistory(false, true);
    if (hasDOM) document.body.classList.toggle('graph-big', this.graphBig);
    if (!quiet) this.refit(); else this.emit();
  }
  setHistory(open: boolean, quiet = false): void {
    this.histOpen = !!open && this.arr !== 'rail';
    if (this.histOpen && this.arr === 'console' && this.graphBig) this.setGraphBig(false, true);
    if (hasDOM) document.body.classList.toggle('hist-open', this.histOpen);
    if (!quiet) this.refit(); else this.emit();
  }
  /** The lane is fitted again around whatever the surface covers (after a layout change, never as turns advance). */
  refit(): void {
    this.emit();
    raf(() => { if (this.lane && this.ctrl && this.lane.layout()) this.lane.update(this.ctrl.frame); this.emit(); });
  }
  chosenLayout(): Layout {
    if (this.layoutChoice !== 'auto') return this.layoutChoice;
    if (!hasDOM) return 'dock';
    return innerWidth > innerHeight && innerHeight <= 520 ? 'rail' : 'dock';
  }
  /** The arrangement for the window: a phone held sideways gets the rail, a narrow window the console form of the
   * dock; returns whether anything changed. */
  arrange(force = false): boolean {
    const lay = this.chosenLayout(), narrow = hasDOM && innerWidth < 700;
    const arr: Layout = lay === 'dock' && narrow ? 'console' : lay;
    const phone = arr === 'rail' ? hasDOM && innerHeight < 560 : arr === 'console' && narrow;
    if (!force && lay === this.layout && arr === this.arr && phone === this.phone) return false;
    this.layout = lay; this.arr = arr; this.phone = phone;
    if (hasDOM) {
      const B = document.body.classList;
      for (const l of LAYOUTS) B.toggle(`lay-${l}`, l === arr);
      B.toggle('phone', phone);
      B.toggle('rail-phone', phone && arr === 'rail');
    }
    if (this.lane) this.lane.labels = arr !== 'rail' && !phone;
    this.setHistory(false, true);
    this.setGraphBig(false, true);
    this.emit();
    return true;
  }
  setLayout(name: string): void {
    this.layoutChoice = (LAYOUTS as string[]).includes(name) ? (name as Layout) : 'auto';
    try {
      const u = new URL(location.href);
      if (this.layoutChoice === 'auto') u.searchParams.delete('layout'); else u.searchParams.set('layout', this.layoutChoice);
      history.replaceState(null, '', u);
    } catch { /* the address cannot change here */ }
    this.arrange(true);
    this.refitLane();
  }
  refitLane(): void {
    raf(() => { if (this.lane && this.ctrl) { this.lane.layout(true); this.lane.update(this.ctrl.frame); } this.emit(); });
  }
  applyTheme(name: Theme, remember = false): void {
    this.theme = THEMES.includes(name) ? name : 'auto';
    if (hasDOM) {
      if (this.theme === 'auto') delete document.documentElement.dataset.theme;
      else document.documentElement.dataset.theme = this.theme;
    }
    if (remember) saveJSON(KEYS.theme, this.theme);
    this.emit();
  }
  isDark(): boolean { return this.theme === 'dark' || (this.theme === 'auto' && hasDOM && matchMedia('(prefers-color-scheme: dark)').matches); }
  setTicksAuto(on: boolean, remember = false): void {
    this.ticksAuto = !!on;
    if (remember) saveJSON(KEYS.ticks, this.ticksAuto ? 'auto' : 'manual');
    if (this.ctrl?.stepping) this.ctrl.setAuto(this.ticksAuto);
    this.emit();
  }
  setSpeed(x: number): void { this.ctrl?.setSpeed(x); this.setDur(); this.emit(); }
  private setDur(): void { if (this.ctrl && hasDOM) document.documentElement.style.setProperty('--dur', `${Math.round((420 * 0.9) / this.ctrl.speed)}ms`); }

  // ---- keys
  key(e: KeyboardEvent): boolean {
    const c = this.ctrl;
    if (!c) return false;
    const live = this.mode === 'play' && !!this.live;
    switch (e.key) {
      case 'ArrowLeft': c.prev(); return true;
      case 'ArrowRight': c.next(); return true;
      case 'Home': c.first(); return true;
      case 'End': c.latest(); return true;
      case 'Escape':
        if (this.dialog) this.closeDialog();
        else if (this.detailsOpen) this.closeDetails();
        else if (this.graphBig) this.setGraphBig(false);
        else if (this.histOpen) this.setHistory(false);
        else c.returnToCurrent();
        return true;
      case ' ':
        if (c.stepping && !c.playing) { c.advance(); return true; }
        if (live && !c.stopped) return false;
        c.toggle();
        return true;
      case 'Enter':
        if (live && c.canSubmit) { void this.submit(this.selectionCode()); return true; }
        return false;
      default: return false;
    }
  }
}
