// The history controller: two indices kept apart, as the M6 plan requires (§5.1, rule 13):
//   progress  the number of resolved turns (a running human game grows it; a bot game has all of them)
//   view      the position on screen; view < progress is a historical view
// Position p is the state after p turns (0 = the initial position); turn t moves position t - 1 to position t.
// Navigation never changes progress and never submits. A turn is shown frame by frame (start, tick 1, tick 2, end),
// stepped by the visitor or played on its own; a jump cancels the stepping and shows the end state. Reduced motion
// skips the frames but keeps the pace. A walk (playTo) shows each position up to a target without the frames. The
// class is a typed port of the approved prototype's controller, with the environment injected so it can be tested
// without a browser.
import { BASE_FRAME_MS, Frame, positionFrame, prefersReducedMotion, SPEEDS, Speed, TICK_MS, turnSubframes, WALK_MS } from './frames';
import type { BeliefMode, Game, History, Menu, Position, RecordingBeliefs, Seat, Turn, TurnResult, View } from './types';

export interface Env {
  reduced(): boolean;        // reduced motion or a still screenshot: no frames
  ticksAuto(): boolean;      // the visitor's setting: play the frames of a turn on their own
  forcedRun(): boolean;      // forced away turns are being played: frames always run on their own
}

export const defaultEnv: Env = { reduced: () => prefersReducedMotion(), ticksAuto: () => false, forcedRun: () => false };

export interface Step { t: number; frames: Frame[]; i: number; auto: boolean; resolve: () => void; fromPlay: boolean }

export interface ControllerOptions { mode?: 'play' | 'watch'; belief?: BeliefMode; view?: number; speed?: number; env?: Env }

const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x));

export class Controller {
  game: Game;
  n: number;
  mode: 'play' | 'watch';
  beliefMode: BeliefMode;
  stopped: boolean;
  progress: number;
  view: number;
  speed: Speed;
  playing = false;
  frame: Frame;
  step: Step | null = null;
  env: Env;
  private listeners = new Set<() => void>();
  private timer: ReturnType<typeof setTimeout> | undefined;
  private timer2: ReturnType<typeof setTimeout> | undefined;
  private anim = 0;
  private target: number | null = null;

  constructor(game: Game, opts: ControllerOptions = {}) {
    this.game = game;
    this.n = game.turns.length;
    this.mode = opts.mode ?? 'watch';
    this.beliefMode = opts.belief ?? 'hidden';
    this.env = opts.env ?? defaultEnv;
    this.stopped = true;                                     // a whole bot game has no decision left
    this.progress = this.n;
    this.view = clamp(opts.view ?? 0, 0, this.progress);
    const speed = opts.speed ?? 1;
    this.speed = (SPEEDS as readonly number[]).includes(speed) ? (speed as Speed) : 1;
    this.frame = positionFrame(game, this.view);
  }

  // ---- observation
  subscribe(fn: () => void): () => void { this.listeners.add(fn); return () => { this.listeners.delete(fn); }; }
  protected emit(): void { for (const fn of this.listeners) fn(); }

  // ---- what the screen may show
  get historical(): boolean { return this.view < this.progress; }
  get atCurrent(): boolean { return this.view === this.progress; }
  get stepping(): boolean { return this.step !== null; }
  /** The position a walk is heading for, or null. */
  get walkingTo(): number | null { return this.target; }
  get canSubmit(): boolean { return false; }
  menu(): Menu | null { return null; }
  forced(): boolean { return false; }
  beliefVisible(): boolean { return !!this.game.beliefs && (this.stopped || this.beliefMode === 'shown'); }
  belief(p: number): { types: string[]; bySeat: Partial<Record<Seat, number[]>> } | null {
    if (!this.beliefVisible() || p > this.progress) return null;
    const out: Partial<Record<Seat, number[]>> = {};
    for (const [seat, rows] of Object.entries(this.game.beliefs!.posteriors)) out[seat as Seat] = rows[p];
    return { types: this.game.beliefs!.types, bySeat: out };
  }
  turnAt(t: number): Turn | null { return t >= 1 && t <= this.progress ? this.game.turns[t - 1] : null; }
  position(p: number): Position | null { return p <= this.progress ? this.game.positions[p] : null; }
  /** While a turn is being stepped its result is not settled: the shown position is the one before it. */
  resolving(): Step | null { const s = this.step; return s && s.i < s.frames.length - 1 ? s : null; }
  shownView(): number { const s = this.resolving(); return s ? s.t - 1 : this.view; }

  // ---- navigation (never changes progress)
  goto(p: number, animate = false): Promise<void> | void {
    this.pause();
    p = clamp(p, 0, this.progress);
    if (animate && p === this.view + 1) return this.animateTurn(p);
    this.cancelAnim();
    this.view = p;
    this.frame = positionFrame(this.game, p);
    this.emit();
  }
  first(): void { this.goto(0); }
  prev(): void { this.goto(this.view - 1); }
  next(): void { if (this.step) this.advance(); else this.goto(this.view + 1, true); }
  latest(): void { this.goto(this.progress); }
  returnToCurrent(): void { this.goto(this.progress); }

  // ---- playback (Watch and the review of a stopped game)
  play(): void {
    if (this.step) { this.playing = true; this.step.auto = true; this.show(); return; }   // carry on from this frame
    if (this.view >= this.progress) {
      if (this.mode === 'play' && !this.stopped) return;      // a running game has nothing ahead to play back
      this.view = 0;                                          // Watch, or the review of a stopped game: restart
    }
    this.playing = true;
    this.emit();
    this.playStep();
  }
  pause(): void {
    if (!this.playing) return;
    this.playing = false;
    this.target = null;
    clearTimeout(this.timer);
    if (this.step) { this.step.auto = false; clearTimeout(this.timer2); }   // stop between frames
    this.emit();
  }
  toggle(): void { if (this.playing) this.pause(); else this.play(); }
  setSpeed(x: number): void { if ((SPEEDS as readonly number[]).includes(x)) { this.speed = x as Speed; this.emit(); } }
  private playStep(): void {
    if (!this.playing) return;
    if (this.view >= this.progress) { this.playing = false; this.emit(); return; }
    const t = this.view + 1;
    void this.animateTurn(t).then(() => {
      if (!this.playing || this.target !== null) return;     // paused, or a walk replaced the playback
      // without the frames, wait as long as they would have shown, so a speed setting means one pace
      const wait = this.env.reduced() ? turnSubframes(this.game, t).length * TICK_MS : BASE_FRAME_MS * 0.8;
      this.timer = setTimeout(() => this.playStep(), wait / this.speed);
    });
  }
  /** Walk forward to position `to`, one position at a time without the frames, then stop. */
  playTo(to: number): void {
    this.pause();
    this.cancelAnim();
    to = clamp(to, 0, this.progress);
    if (to <= this.view) { this.goto(to); return; }
    this.playing = true;
    this.target = to;
    this.emit();
    this.timer = setTimeout(() => this.walkStep(), WALK_MS / this.speed);
  }
  private walkStep(): void {
    if (this.target === null) return;
    this.view++;
    this.frame = positionFrame(this.game, this.view);
    if (this.view >= this.target) { this.playing = false; this.target = null; }
    else this.timer = setTimeout(() => this.walkStep(), WALK_MS / this.speed);
    this.emit();
  }

  // ---- a turn shown frame by frame
  protected animateTurn(t: number): Promise<void> {
    if (this.env.reduced()) {
      this.cancelAnim();
      this.view = t;
      this.frame = positionFrame(this.game, t);
      this.emit();
      return Promise.resolve();
    }
    return this.begin(t, this.playing || this.env.ticksAuto() || this.env.forcedRun());
  }
  /** Step through a resolved turn on request, from its start. */
  stepTurn(t: number): void {
    if (t < 1 || t > this.progress) return;
    this.pause();
    void this.begin(t, this.env.ticksAuto());
  }
  private begin(t: number, auto: boolean): Promise<void> {
    this.cancelAnim();
    this.view = t;
    return new Promise((resolve) => {
      this.step = { t, frames: turnSubframes(this.game, t), i: 0, auto, resolve, fromPlay: this.playing };
      this.show();
    });
  }
  private show(): void {
    const s = this.step!;
    clearTimeout(this.timer2);
    this.frame = s.frames[s.i];
    this.emit();
    if (s.auto) this.timer2 = setTimeout(() => this.advance(), TICK_MS / this.speed);
  }
  /** The next frame; after the end frame, the turn is done. */
  advance(): void {
    const s = this.step;
    if (!s) return;
    if (s.i < s.frames.length - 1) { s.i++; this.show(); } else this.finish();
  }
  /** Straight to the end frame (its panel stays until Continue). */
  skip(): void {
    const s = this.step;
    if (s && s.i < s.frames.length - 1) { s.i = s.frames.length - 1; this.show(); }
  }
  setAuto(on: boolean): void {
    const s = this.step;
    if (!s) return;
    s.auto = on || this.playing;
    if (s.auto) this.show();
    else { clearTimeout(this.timer2); this.emit(); }
  }
  private finish(): void {
    const s = this.step!;
    this.step = null;
    clearTimeout(this.timer2);
    this.emit();
    s.resolve();
    if (this.playing && !s.fromPlay) this.timer = setTimeout(() => this.playStep(), (BASE_FRAME_MS * 0.8) / this.speed);
  }
  protected cancelAnim(): void {
    this.anim++;
    clearTimeout(this.timer2);
    if (this.step) { const s = this.step; this.step = null; s.resolve(); }
  }
  /** Stop every timer (when the controller is replaced). */
  dispose(): void { this.playing = false; clearTimeout(this.timer); this.cancelAnim(); this.listeners.clear(); }
}

/** A human game that grows one resolved turn at a time. `n` is the turn cap; the menu is the engine's menu for the
 * current decision; nothing past `progress` exists on the client. */
export class Live extends Controller {
  menuNow: Menu | null = null;

  constructor(game: Game, cap: number, opts: ControllerOptions = {}) {
    super(game, { ...opts, mode: 'play', view: 0 });
    this.n = cap;
    this.progress = game.turns.length;
    this.view = clamp(opts.view ?? this.progress, 0, this.progress);
    this.frame = positionFrame(game, this.view);
    this.stopped = false;
  }

  get canSubmit(): boolean { return !this.stopped && this.atCurrent && this.progress < this.n && this.step === null; }
  menu(): Menu | null { return this.menuNow; }
  forced(): boolean { return this.canSubmit && this.menuNow !== null && this.menuNow.forced !== null; }
  awayRun(): number { return this.menuNow && this.menuNow.away ? this.menuNow.away.returns_in : 0; }

  /** Take the server's view of the decision point (after a submission, a resync or a reload). */
  adopt(view: View): void {
    this.game.meta.marks = view.marks;
    this.game.meta.seeds = view.seeds;
    this.stopped = view.state === 'stopped';
    this.menuNow = this.stopped ? null : (view.menu ?? null);
  }

  /** One resolved turn from the server (a turn result): append it and show it. */
  append(res: TurnResult): Promise<void> {
    this.appendTurn(res.turn, res.position, res.belief ? Object.fromEntries(Object.entries(res.belief).map(([s, b]) => [s, b.posterior])) : null);
    this.adopt(res.view);
    return this.animateTurn(this.progress);
  }

  /** Resolved turns the client missed (a reload or an unanswered request), from a history request. */
  appendHistory(h: History, view: View): number {
    let added = 0;
    for (let i = 0; i < h.turns.length; i++) {
      const t = h.turns[i].t;
      if (t !== this.progress + 1) continue;                 // only turns that continue what the client has
      const row: Partial<Record<Seat, number[]>> | null = h.beliefs
        ? Object.fromEntries(Object.entries(h.beliefs).map(([s, b]) => [s, b.posteriors[t - h.from]])) : null;
      this.appendTurn(h.turns[i], h.positions[t - h.from], row);
      added++;
    }
    this.adopt(view);
    if (added) { this.view = this.progress; this.frame = positionFrame(this.game, this.view); }
    this.emit();
    return added;
  }

  private appendTurn(turn: Turn, position: Position, belief: Partial<Record<Seat, number[]>> | null): void {
    const G = this.game;
    G.turns.push(turn);
    G.positions.push(position);
    if (belief && G.beliefs) for (const [s, row] of Object.entries(belief)) if (row) (G.beliefs.posteriors[s] ??= []).push(row);
    this.progress = G.turns.length;
  }

  /** After the stop: the whole belief history and the seeds are public. */
  conclude(beliefs: RecordingBeliefs | null | undefined, view: View): void {
    if (beliefs) this.game.beliefs = beliefs;
    this.adopt(view);
    this.stopped = true;
    this.menuNow = null;
    this.emit();
  }
}
