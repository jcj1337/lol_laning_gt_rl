// M6a prototype core: data loading, the history controller and playback, shared by every direction.
//
// Throwaway prototype code (docs/m6/m6a_decision_record.md). It reads frame files exported by
// `python -m lanegame.experiments.m6a export` (the demo's frame contract, lanegame/demo/frames.py) and computes no
// game rule: every number it shows comes from those files.
//
// Two indices are kept apart, as the production app must keep them apart:
//   progress  the number of resolved turns in the game being played (Play) or the whole recording (Watch, Review)
//   view      the position on screen; view < progress is a historical view
// Position p is the state after p turns (0 = the initial position); turn t moves position t - 1 to position t.
//
// Play is STAGED against a verified recording: a visitor may select any legal action, but the recorded action is the
// one that resolves, and the controller reports both. The data file contains the whole recorded game, which the real
// client never would; only the backend (M6b) resolves arbitrary choices and withholds the future.

export const SPEEDS = [0.5, 1, 2, 4];
export const BASE_FRAME_MS = 420;            // one sub-frame (start, tick 1, tick 2, end) at 1x

export function params() {
  const q = new URLSearchParams(location.search);
  const num = (k, d) => (q.has(k) && q.get(k) !== '' && !isNaN(+q.get(k)) ? +q.get(k) : d);
  return {
    game: q.get('game') || 'h2h_om_s0_vs_sp_s0',
    mode: q.get('mode') || 'play',               // play | watch | review
    progress: num('progress', null),              // Play: resolved turns to start from (screenshots, demos)
    view: num('view', null),
    detail: q.get('detail') || '',                // which optional panel is open, '' = closed
    belief: q.get('belief') || 'hidden',          // Play: hidden (default) | shown
    speed: num('speed', 1),
    theme: q.get('theme') || '',
    still: q.has('still'),                        // screenshots: no animation
  };
}

export async function loadIndex(base = '../data') {
  const r = await fetch(`${base}/index.json`);
  return r.json();
}

export async function loadGame(name, base = '../data') {
  const r = await fetch(`${base}/${name}.json`);
  if (!r.ok) throw new Error(`cannot load ${name}: ${r.status}`);
  return r.json();
}

export const prefersReducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

// ------------------------------------------------------------------------------------------- names and text
export function seatNames(game, mode) {
  const seat = (s) => game.meta.seats['AB'.indexOf(s)];
  const label = (x) => (x.kind === 'model' ? `${x.label} (${x.id})` : x.label || x.id);
  if (mode === 'play') return { A: 'You', B: label(seat('B')) };
  return { A: label(seat('A')), B: label(seat('B')) };
}

export function shortNames(game, mode) {
  const s = (x) => (x.kind === 'model' ? x.id : x.id);
  if (mode === 'play') return { A: 'You', B: s(game.meta.seats[1]) };
  return { A: s(game.meta.seats[0]), B: s(game.meta.seats[1]) };
}

// Fill the {A} / {B} placeholders of the outcome texts. A seat named "You" takes the second person ("Your wave",
// "you were ganked", "You are back"); the outcome texts use the past tense otherwise, so no other verb changes.
export function fillText(text, names) {
  let out = text;
  for (const key of ['A', 'B']) {
    const name = names[key];
    const p = `\\{${key}\\}`;
    if (name !== 'You') { out = out.replace(new RegExp(p, 'g'), name); continue; }
    const start = '(^|[.!?]\\s+)';
    out = out
      .replace(new RegExp(`${start}${p}'s`, 'g'), '$1Your').replace(new RegExp(`${p}'s`, 'g'), 'your')
      .replace(new RegExp(`${start}${p} was\\b`, 'g'), '$1You were').replace(new RegExp(`${p} was\\b`, 'g'), 'you were')
      .replace(new RegExp(`${start}${p} is\\b`, 'g'), '$1You are').replace(new RegExp(`${p} is\\b`, 'g'), 'you are')
      .replace(new RegExp(`${start}${p}`, 'g'), '$1You').replace(new RegExp(p, 'g'), 'you');
  }
  return out;
}

export const gold = (x) => (x >= 0 ? '+' : '−') + Math.abs(x).toFixed(2);
export const gold0 = (x) => x.toFixed(1);
export const hpText = (x) => `${Math.round(x)}`;

const WAVE_SHORT = { 0: 'LH', 2: 'TN', 3: 'SL', 1: 'SH' };
const BEH_SHORT = { 0: 'Hold', 1: 'Yield', 2: 'Trade', 3: 'Safe' };

// A compact code label for a history row: "SH·Trade", "SL·Hold+P", "Recall", "Ward", "Away".
export function codeShort(parts) {
  if (parts.kind === 'absent') return 'Away';
  if (parts.kind === 'recall') return 'Recall';
  if (parts.kind === 'ward') return 'Ward';
  return `${WAVE_SHORT[parts.wave]}·${BEH_SHORT[parts.behaviour]}${parts.objective === 1 ? '+P' : ''}`;
}

// ------------------------------------------------------------------------------------------- frames
// A drawable frame: the lane, each champion's cell, and the status shown with it. HP, gold and XP change at the end
// of a turn (turn-level outcomes), so tick frames show the pre-turn status.
export function positionFrame(game, p) {
  const pos = game.positions[p];
  const cells = {}, ward = {};
  for (const n of ['A', 'B']) {
    cells[n] = pos.champions[n].cell;
    ward[n] = pos.champions[n].at_ward;
  }
  return { p, sub: 'position', label: p === 0 ? 'initial position' : `after turn ${p}`, lane: pos.lane, cells,
           at_ward: ward, status: pos.champions, towers: pos.towers, tick: null };
}

export function turnSubframes(game, t) {
  const tr = game.turns[t - 1];
  const pre = positionFrame(game, t - 1);
  const out = [{ ...pre, sub: 'start', label: `turn ${t}: start`, turn: t }];
  tr.ticks.forEach((tk, j) => {
    out.push({ p: t - 1, sub: 'tick', label: `turn ${t}: tick ${j + 1}`, turn: t,
               lane: { packets: tk.packets, siege: tk.siege, contact: tk.contact }, cells: tk.cells,
               at_ward: tk.at_ward, status: pre.status, towers: pre.towers, tick: tk });
  });
  out.push({ ...positionFrame(game, t), sub: 'end', label: `turn ${t}: end`, turn: t });
  return out;
}

// ------------------------------------------------------------------------------------------- the controller
export class Controller extends EventTarget {
  constructor(game, opts = {}) {
    super();
    this.game = game;
    this.n = game.turns.length;
    this.mode = opts.mode || 'play';
    this.beliefMode = opts.belief || 'hidden';
    this.stopped = this.mode !== 'play';
    this.progress = this.mode === 'play' ? clamp(opts.progress ?? 0, 0, this.n) : this.n;
    if (this.progress >= this.n) this.stopped = true;          // a game at its cap has no decision left
    this.view = clamp(opts.view ?? (this.mode === 'play' ? this.progress : 0), 0, this.progress);
    this.speed = SPEEDS.includes(opts.speed) ? opts.speed : 1;
    this.playing = false;
    this.frame = positionFrame(game, this.view);
    this.lastSubmission = null;       // {t, selected, recorded, matched}
    this.still = !!opts.still;
    this._timer = null;
    this._anim = 0;
  }

  // ---- what the screen may show
  get historical() { return this.view < this.progress; }
  get atCurrent() { return this.view === this.progress; }
  get canSubmit() { return this.mode === 'play' && !this.stopped && this.atCurrent && this.progress < this.n; }
  menu() { return this.game.menus_A[this.progress] ?? null; }   // Play: the decision at the current position (none at the cap)
  forced() { return this.canSubmit && this.menu() !== null && this.menu().forced !== null; }
  awayRun() {                                                // how many forced turns follow from the current one
    let k = 0;
    while (this.progress + k < this.n && this.game.menus_A[this.progress + k].forced !== null) k++;
    return k;
  }
  beliefVisible() {
    if (!this.game.beliefs) return false;
    return this.stopped || this.beliefMode === 'shown';
  }
  belief(p) {                                                // never beyond progress, never while hidden
    if (!this.beliefVisible() || p > this.progress) return null;
    const out = {};
    for (const [seat, rows] of Object.entries(this.game.beliefs.posteriors)) out[seat] = rows[p];
    return { types: this.game.beliefs.types, bySeat: out };
  }
  turnAt(t) { return t >= 1 && t <= this.progress ? this.game.turns[t - 1] : null; }
  position(p) { return p <= this.progress ? this.game.positions[p] : null; }

  // ---- navigation (never changes progress)
  goto(p, { animate = false } = {}) {
    this.pause();
    p = clamp(p, 0, this.progress);
    if (animate && p === this.view + 1) return this._animateTurn(p);
    this._cancelAnim();
    this.view = p;
    this.frame = positionFrame(this.game, p);
    this._emit();
  }
  first() { this.goto(0); }
  prev() { this.goto(this.view - 1); }
  next() { this.goto(this.view + 1, { animate: true }); }
  latest() { this.goto(this.progress); }
  returnToCurrent() { this.goto(this.progress); }

  // ---- playback (Watch and Review; also history in Play)
  play() {
    if (this.view >= this.progress) {
      if (this.mode === 'play' && !this.stopped) return;     // a running game has nothing ahead to play back
      this.view = 0;                                          // Watch, or the review of a stopped game: restart
    }
    this.playing = true;
    this._emit();
    this._step();
  }
  pause() {
    if (!this.playing) return;
    this.playing = false;
    clearTimeout(this._timer);
    this._emit();
  }
  toggle() { this.playing ? this.pause() : this.play(); }
  setSpeed(x) { if (SPEEDS.includes(x)) { this.speed = x; this._emit(); } }
  _step() {
    if (!this.playing) return;
    if (this.view >= this.progress) { this.playing = false; this._emit(); return; }
    this._animateTurn(this.view + 1).then(() => {
      if (!this.playing) return;
      this._timer = setTimeout(() => this._step(), BASE_FRAME_MS * 0.8 / this.speed);
    });
  }

  // ---- staged submission (Play)
  submit(code) {
    if (!this.canSubmit) return null;
    const m = this.menu();
    if (!m.legal.includes(code)) return null;                 // the real backend explains and refuses
    const t = this.progress + 1;
    const recorded = this.game.turns[t - 1].codes[0];
    this.progress = t;
    this.lastSubmission = { t, selected: code, recorded, matched: code === recorded };
    if (this.progress >= this.n) this.stopped = true;
    this._animateTurn(t);
    return this.lastSubmission;
  }
  async continueAway() {                                      // one control, one resolution per turn
    while (this.forced()) {
      this.submit(this.menu().forced);
      await this._waitAnim();
    }
  }
  stop() {
    if (this.mode !== 'play' || this.stopped) return;
    this.pause();
    this.stopped = true;
    this._emit();
  }

  // ---- animation of one turn: start, tick 1, tick 2, end
  _animateTurn(t) {
    this._cancelAnim();
    const id = ++this._anim;
    const frames = turnSubframes(this.game, t);
    this.view = t;
    const reduced = prefersReducedMotion() || this.still;
    if (reduced) {
      this.frame = frames[frames.length - 1];
      this._emit();
      return Promise.resolve();
    }
    return new Promise((resolve) => {
      let i = 0;
      const tick = () => {
        if (id !== this._anim) return resolve();
        this.frame = frames[i];
        this._emit({ animating: i < frames.length - 1 });
        i++;
        if (i < frames.length) this._timer2 = setTimeout(tick, BASE_FRAME_MS / this.speed);
        else resolve();
      };
      tick();
    });
  }
  _waitAnim() { return new Promise((r) => setTimeout(r, (prefersReducedMotion() || this.still) ? 0 : BASE_FRAME_MS * 4.2 / this.speed)); }
  _cancelAnim() { this._anim++; clearTimeout(this._timer2); }
  _emit(extra = {}) { this.dispatchEvent(new CustomEvent('change', { detail: extra })); }
}

function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }

// ------------------------------------------------------------------------------------------- keyboard
// Left / Right step, Home / End jump, Space plays or pauses, Escape returns to the current turn. Keys are ignored
// while focus is in a form field.
export function bindKeys(ctrl, extra = {}) {
  window.addEventListener('keydown', (e) => {
    if (e.target.closest('input, textarea, select, [contenteditable]')) return;
    if (e.altKey || e.ctrlKey || e.metaKey) return;
    const k = e.key;
    const act = {
      ArrowLeft: () => ctrl.prev(), ArrowRight: () => ctrl.next(), Home: () => ctrl.first(), End: () => ctrl.latest(),
      Escape: () => ctrl.returnToCurrent(),
      ' ': () => (ctrl.mode === 'play' && !ctrl.stopped ? null : ctrl.toggle()),
      ...extra,
    }[k];
    if (act && !(k === ' ' && e.target.closest('button'))) {
      const r = act();
      if (r !== null) e.preventDefault();
    }
  });
}

// ------------------------------------------------------------------------------------------- belief labels
export const BELIEF_LABELS = {
  g_farm: 'Farmer (scripted)', g_safe: 'Safe (scripted)', g_bully: 'Bully (scripted)', g_shove: 'Shover (scripted)',
  fz_guard_sh: 'Freeze guard (scripted)', g_shove_ward: 'Shove + ward (scripted)', peer: 'Peer policy (learned)',
  attacker: 'Attacker (learned)', freezer: 'Freeze specialist (learned)', gapper: 'Large-gap specialist (learned)',
  short: 'Short-horizon policy (learned)', mixed: 'Changing mixture', unknown: 'Unknown',
};
export const BELIEF_NOTE = 'The modelling agent compares your play with 13 opponent types: 6 scripted styles, 5 ' +
  'learned policies, a mixture and "unknown". None of them is a human. In shown mode you can try to steer it on ' +
  'purpose; M5 never tested deliberate deception.';

export const STAGED_NOTE = 'Prototype: Play is staged against a verified recording. You can select any legal action, ' +
  'but the recorded action resolves. The real app resolves your own choice on the server.';
