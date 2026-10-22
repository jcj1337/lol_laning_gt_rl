// The M6a demo as one game view with one glass surface: a top-down lane fills the window, and every control sits on a
// single translucent surface over it, in one of three layouts (?layout=dock|rail|console; About switches them). The
// lane is fitted around whatever the surface covers. Play is live: every action goes to the local engine
// (prototypes/m6a/serve.py, lanegame/demo/service.py), which resolves it with the tested session core and returns the
// frames shown here. Watch plays back verified recordings (data/*.json) or a new simulation from the engine. This file
// computes no game rule.

import * as core from './core.js';

window.__errors = [];
addEventListener('error', (e) => window.__errors.push(String(e.message || e)));
addEventListener('unhandledrejection', (e) => window.__errors.push(String((e.reason && e.reason.message) || e.reason)));

const P = core.params();
const Q = new URLSearchParams(location.search);
const $ = (id) => document.getElementById(id);
const NS = 'http://www.w3.org/2000/svg';
// Minions are drawn as balls, one per HP point: the engine tracks each wave's HP, not single minions, and 1 HP of
// minions dying pays the gold of one HP.
const SEATS = ['A', 'B'];
// Team colours and the few scene colours drawn from script come from the theme's CSS variables (readTheme), so a light
// or dark theme changes them in one place.
const COL = { A: '#2563a8', B: '#c0392b' };
const SOFT = { A: '#dbe6f4', B: '#f6dcd8' };
const TH = { fill: 0.18, neutral: '#8a8f96', muted: '#555b62', siege: '#ffffff' };
function readTheme() {
  const cs = getComputedStyle(document.documentElement), v = (k) => cs.getPropertyValue(k).trim();
  COL.A = v('--a'); COL.B = v('--b'); SOFT.A = v('--a-soft'); SOFT.B = v('--b-soft');
  Object.assign(TH, { fill: +v('--g-fill') || 0.18, neutral: v('--g-neutral'), muted: v('--muted'), siege: v('--siege-edge') });
}
const WAVE = { 0: 'Last hit', 2: 'Thin', 3: 'Slow push', 1: 'Shove' };
const BEH = { 0: 'Hold', 1: 'Yield', 2: 'Trade', 3: 'Safe' };
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const sgn = (x, d = 2) => (x > 0 ? '+' : x < 0 ? '−' : '') + Math.abs(x).toFixed(d);
const turns = (k) => `${k} turn${k === 1 ? '' : 's'}`;
const other = (n) => (n === 'A' ? 'B' : 'A');
const empty = () => ({ kind: null, wave: null, beh: null, pressure: false });

const S = { G: null, ctrl: null, mode: 'play', live: null, engine: null, index: null, names: {}, short: {},
            sel: empty(), busy: false, error: '', hover: null, events: false, rows: -1, last: {}, lane: null,
            histOpen: false, graphBig: false, layout: 'dock', layoutChoice: 'auto', arr: 'dock', phone: false, arrKey: '', theme: 'auto' };
const LAYOUTS = ['dock', 'rail', 'console'];
const THEMES = ['auto', 'light', 'dark'];
const THEME_KEY = 'm6a-glass-theme';
const TICKS_KEY = 'm6a-glass-ticks';

// ------------------------------------------------------------------------------------------- engine
async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  let d = null;
  try { d = await r.json(); } catch { /* not JSON */ }
  if (!r.ok) throw Object.assign(new Error((d && d.message) || `HTTP ${r.status}`), { status: r.status });
  return d;
}

// Turns resolved frame by frame: the start (both actions shown), tick 1, tick 2, the end. The viewer steps through them
// (Next tick) unless ticks play on their own: by choice, during Watch's Play, and for forced away turns. A jump (a
// history row, the graph, First, Previous, Latest, Return) cancels a step and shows the end state. Reduced motion and
// ?still skip the frames, as before; "Show the ticks" steps through a turn on request. Only the frames of the shared
// controller are shown; nothing is computed here.
const TICK_MS = 1100;                                 // one frame at 1x when ticks play on their own
class Stepper extends core.Controller {
  constructor(game, opts) { super(game, opts); this.step = null; }
  get stepping() { return this.step !== null; }
  get canSubmit() { return super.canSubmit && this.step === null; }      // no submission while a turn is being shown
  _animateTurn(t) {
    if (core.prefersReducedMotion() || this.still) return super._animateTurn(t);
    return this._begin(t, this.playing || S.ticksAuto || S.forcedRun);
  }
  stepTurn(t) {                                       // step through a turn on request, from its start
    if (t < 1 || t > this.progress) return;
    this.pause();
    this._begin(t, S.ticksAuto);
  }
  _begin(t, auto) {
    this._cancelAnim();
    this.view = t;
    return new Promise((resolve) => {
      this.step = { t, frames: core.turnSubframes(this.game, t), i: 0, auto, resolve, fromPlay: this.playing };
      this._show();
    });
  }
  _show() {
    const s = this.step;
    clearTimeout(this._timer2);
    this.frame = s.frames[s.i];
    this._emit({ animating: true });
    if (s.auto) this._timer2 = setTimeout(() => this.advance(), TICK_MS / this.speed);
  }
  advance() {                                         // the next frame; after the end frame, the turn is done
    const s = this.step;
    if (!s) return;
    if (s.i < s.frames.length - 1) { s.i++; this._show(); } else this._finish();
  }
  skip() {                                            // straight to the end frame (its panel stays until Continue)
    const s = this.step;
    if (s && s.i < s.frames.length - 1) { s.i = s.frames.length - 1; this._show(); }
  }
  setAuto(on) {
    const s = this.step;
    if (!s) return;
    s.auto = on || this.playing;
    if (s.auto) this._show();
    else { clearTimeout(this._timer2); this._emit(); }
  }
  _finish() {
    const s = this.step;
    this.step = null;
    clearTimeout(this._timer2);
    this._emit();
    s.resolve();
    if (this.playing && !s.fromPlay) this._timer = setTimeout(() => this._step(), core.BASE_FRAME_MS * 0.8 / this.speed);
  }
  _cancelAnim() {
    super._cancelAnim();
    if (this.step) { const s = this.step; this.step = null; s.resolve(); }
  }
  next() { return this.step ? this.advance() : super.next(); }
  play() {
    if (this.step) { this.playing = true; this.step.auto = true; this._show(); return; }   // carry on from this frame
    super.play();
  }
  pause() {
    const was = this.playing;
    super.pause();
    if (was && this.step) { this.step.auto = false; clearTimeout(this._timer2); this._emit(); }   // stop between frames
  }
}

// The shared controller, extended for a game that grows one resolved turn at a time. `n` is the turn cap; the menu
// is the engine's menu for the current decision; nothing past `progress` exists on the client.
class Live extends Stepper {
  constructor(game, cap, opts) {
    super(game, { ...opts, mode: 'play', progress: 0, view: 0 });
    this.n = cap;
    this.stopped = false;
    this.menuNow = null;
  }
  menu() { return this.menuNow; }
  awayRun() { return this.menuNow && this.menuNow.away ? this.menuNow.away.returns_in : 0; }
  submit() { return null; }                       // the staged path of the old prototypes is never used here
  append(res) {
    const G = this.game;
    G.turns.push(res.turn);
    G.positions.push(res.position);
    if (res.belief && G.beliefs) for (const [s, b] of Object.entries(res.belief)) G.beliefs.posteriors[s].push(b.posterior);
    G.meta.marks = res.view.marks;
    this.progress = G.turns.length;
    this.stopped = res.view.state === 'stopped';
    this.menuNow = this.stopped ? null : res.view.menu;
    return this._animateTurn(this.progress);
  }
}

// ------------------------------------------------------------------------------------------- words
// Plain names for every seat. The research ids (sp:0 is M5's self-play agent of seed 0, om:0 its opponent-modelling
// agent) and the bots' rules appear only in Details.
const AGENTS = {
  'sp:0': { name: 'Self-play agent', short: 'Self-play', about: 'learned only by playing against itself' },
  'om:0': { name: 'Adaptive agent', short: 'Adaptive',
            about: 'keeps a running guess about what kind of opponent it faces and adjusts to it' },
};
const BOTS = {
  g_shove: { name: 'Always-push bot', short: 'Push bot', about: 'a simple rule-based bot that always pushes its wave as hard as it can' },
  g_farm: { name: 'Farming bot', short: 'Farm bot', about: 'a simple rule-based bot that only farms' },
  g_safe: { name: 'Cautious bot', short: 'Safe bot', about: 'a simple rule-based bot that farms from a safe distance' },
  g_bully: { name: 'Aggressive bot', short: 'Bully bot', about: 'a simple rule-based bot that attacks whenever it can' },
};
function who(x) {
  if (x.kind === 'human') return { name: 'You', short: 'You' };
  if (x.kind === 'model') return AGENTS[x.id] || { name: x.label, short: x.label };
  return BOTS[x.id] || { name: 'Rule-based bot', short: 'Bot' };
}
const GAME_NAMES = {
  h2h_om_s0_vs_sp_s0: 'Self-play agent vs. adaptive agent',
  om_s0_vs_g_shove: 'Always-push bot vs. adaptive agent',
  sp_s0_vs_g_shove: 'Self-play agent vs. always-push bot',
  sim_om_vs_sp: 'Adaptive agent vs. self-play agent',
};
function seatLabel(x) { return who(x).name; }

function setNames() {
  const [a, b] = S.G.meta.seats.map(seatLabel);
  const dup = a === b;
  S.names = { A: dup ? `${a} (A)` : a, B: dup ? `${b} (B)` : b };
  const short = (x) => who(x).short;
  const [sa, sb] = S.G.meta.seats.map(short);
  S.short = S.mode === 'play' && S.live ? { A: 'You', B: 'the opponent' } : { A: sa === sb ? `${sa} A` : sa, B: sa === sb ? `${sb} B` : sb };
  S.head = S.mode === 'play' && S.live ? { A: 'You', B: 'Opponent' } : S.short;
}

const cap = (t) => t.replace(/(^|[.!?]\s+)(the opponent|you)\b/g, (m, a, b) => a + b[0].toUpperCase() + b.slice(1));
const fill = (t) => cap(core.fillText(t, S.short));

function actionWords(p) {
  if (p.kind === 'absent') return 'Away';
  if (p.kind === 'recall') return 'Recall';
  if (p.kind === 'ward') return 'Ward';
  return `${WAVE[p.wave]}, ${BEH[p.behaviour]}${p.objective === 1 ? ', Pressure' : ''}`;
}

function phrase(e) {
  const t = phraseRaw(e);
  return t && cap(t);
}

function phraseRaw(e) {
  const n = S.short[e.seat], o = S.short[other(e.seat)];
  const you = n === 'You';
  switch (e.kind) {
    case 'trade': return e.data.result === 'hit' ? `${n} hit ${o === 'You' ? 'you' : o} (${e.data.damage})`
      : e.data.result === 'miss' ? `${you ? 'Your' : n + "'s"} trade missed` : null;
    case 'crash': return `${you ? 'Your' : n + "'s"} wave crashed`;
    case 'tower_shot': return `Tower hit ${you ? 'you' : n}`;
    case 'gank': return `${you ? 'You were' : n + ' was'} ganked`;
    case 'death': return `${n} died`;
    case 'plate': return `${n} broke a plate`;
    case 'recall': return `${n} recalled`;
    case 'recall_interrupted': return `${you ? 'Your' : n + "'s"} recall was cut`;
    case 'ward': return `${n} warded`;
    case 'ward_interrupted': return `${you ? 'Your' : n + "'s"} ward was cut`;
    default: return null;
  }
}

function glyph(n) {
  return n === 'A'
    ? `<svg class="glyph" viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="7" fill="${COL.A}"/><text x="8" y="11.5" text-anchor="middle">A</text></svg>`
    : `<svg class="glyph" viewBox="0 0 16 16" aria-hidden="true"><rect x="3" y="3" width="10" height="10" rx="1" transform="rotate(45 8 8)" fill="${COL.B}"/><text x="8" y="11.5" text-anchor="middle">B</text></svg>`;
}

// ------------------------------------------------------------------------------------------- the scene (top-down lane)
function svg(tag, attrs = {}, parent = null) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}

// What the surface covers: the floating top bar, the surface itself and any sheet open out of it, each with a margin.
// These change only on a resize or when a section is opened or closed, never as turns advance.
function obstacles() {
  const m = 12;
  return ['topbar', 'surface', 'zHist', 'graphBig'].map($).filter((e) => e && !e.hidden && e.offsetParent !== null)
    .map((e) => e.getBoundingClientRect()).filter((r) => r.width > 0 && r.height > 0)
    .map((r) => ({ x0: r.left - m, y0: r.top - m, x1: r.right + m, y1: r.bottom + m }));
}

// The largest rectangle of the window left after cutting each obstacle away on the side that keeps the most area. The
// lane is centred on it when it can be, and the legend sits in its top-left corner.
function freeRect(obs) {
  const e = 14, area = (r) => Math.max(0, r.x1 - r.x0) * Math.max(0, r.y1 - r.y0);
  let r = { x0: e, y0: e, x1: innerWidth - e, y1: innerHeight - e };
  for (const o of [...obs].sort((p, q) => area(q) - area(p))) {
    if (o.x1 <= r.x0 || o.x0 >= r.x1 || o.y1 <= r.y0 || o.y0 >= r.y1) continue;
    r = [{ ...r, x1: o.x0 }, { ...r, x0: o.x1 }, { ...r, y1: o.y0 }, { ...r, y0: o.y1 }].reduce((p, q) => (area(q) > area(p) ? q : p));
  }
  return r;
}

const hits = (a, b, dx = 0, dy = 0) => a.x0 + dx < b.x1 && a.x1 + dx > b.x0 && a.y0 + dy < b.y1 && a.y1 + dy > b.y0;
const overlap = (a, b) => Math.max(0, Math.min(a.x1, b.x1) - Math.max(a.x0, b.x0)) * Math.max(0, Math.min(a.y1, b.y1) - Math.max(a.y0, b.y0));

// A centre for a drawing whose boxes `ex` (relative to its centre) all stay inside `B` and clear of every obstacle,
// the candidate nearest the centre of `pref` first; null when there is none. Testing every box, not one bounding
// rectangle, lets the ends of the diagonal road reach past a surface that only covers the middle of an edge.
function placeDrawing(ex, B, obs, pref) {
  const lo = { x: B.x0 - ex.x0, y: B.y0 - ex.y0 }, hi = { x: B.x1 - ex.x1, y: B.y1 - ex.y1 };
  if (lo.x > hi.x || lo.y > hi.y) return null;
  const cl = (v, a, b) => Math.max(a, Math.min(b, v));
  const px = cl((pref.x0 + pref.x1) / 2 - (ex.x0 + ex.x1) / 2, lo.x, hi.x), py = cl((pref.y0 + pref.y1) / 2 - (ex.y0 + ex.y1) / 2, lo.y, hi.y);
  const step = Math.max(6, Math.min(20, Math.max(hi.x - lo.x, hi.y - lo.y) / 50));
  const axis = (a, b) => { const out = []; for (let v = a; v < b; v += step) out.push(v); out.push(b); return out; };
  const cands = [[px, py]];
  for (const x of axis(lo.x, hi.x)) for (const y of axis(lo.y, hi.y)) cands.push([x, y]);
  cands.sort((p, q) => (p[0] - px) ** 2 + (p[1] - py) ** 2 - (q[0] - px) ** 2 - (q[1] - py) ** 2);
  for (const [x, y] of cands) if (!ex.boxes.some((b) => obs.some((o) => hits(b, o, x, y)))) return { x, y };
  return null;
}

// The simulator's topology drawn from above, on a diagonal: nine cells on one road from A's base at the lower left
// (cell -4) to B's base at the upper right (+4), with A's tower at -2, the lane 0 and B's tower at +2, and the ward spot
// W beside cell 0 on the upper-left side. One mapping places everything: P(u, v) = centre + u (cos t, -sin t) +
// v (sin t, cos t), u along the road (cells x spacing), v across it (negative: the upper-left side). A's minions walk the
// upper-left half and B's the lower-right half; A stands on the upper-left edge, B on the lower-right edge. A's tower sits
// just off the lower-right edge and B's just off the upper-left edge (point-symmetric), clear of a champion standing on
// its own edge. Shapes of the road (band, cell dividers, bases) are rotated; text, balls, tokens and floats stay upright.
// Nothing else is drawn: no other paths or areas. With `labels` on (dock and console), each champion carries a small
// status chip (name, level, HP) placed beside its token where it covers nothing.
const BUSH_NOTE = 'The bush hides no one in this simulator: the other side still sees where this champion is. The '
  + 'see-through token only marks that it stands in the bush.';
const CHIP = { w: 108, h: 34 };                               // the largest chip, for fitting the drawing
const tokHalf = (n, r) => (n === 'B' ? r * 1.17 : r) + 1;    // half the token's width: A a circle, B a diamond
const STEEP = (35 * Math.PI) / 180;                           // above this angle B's base label goes above the base
class Lane {
  constructor(el) { this.el = el; this.packets = new Map(); this.fxKey = ''; this.key = ''; this.labels = true; }

  // every length of the drawing at scale a (the parent's rules: the road grows faster than the cells, capped at 2.1;
  // a wide road takes clusters of at most 6 across and 4 rows; the balls fit a cell)
  sizes(a) {
    const sp = 115 * a, b = Math.min(2.1, 1.7 * a);
    const hr = Math.round(44 * b), rt = Math.round(17 * b), tok = Math.round(12.5 * b), wr = Math.round(15 * b);
    const maxCols = hr >= 75 ? 6 : 8, maxRows = maxCols === 6 ? 4 : 3;
    const R = Math.max(3, Math.min(7.2, (sp * 0.85) / (maxCols * 2.45), (hr - 6) / (maxRows * 2.45)));
    const halfH = hr + Math.round(10 * b), half = Math.min(halfH, sp * 0.46);
    // the ward spot: a rounded patch with an eye in the middle and room for a warding champion at each end
    const we = Math.max(7, Math.round(wr * 0.72)), ph = 2 * Math.max(tok + 7, we + 11), pw = 2 * (we + 2 * tok + 12);
    return { a, sp, b, hr, rt, tok, wr, we, pw, ph, maxCols, maxRows, R, D: R * 2.45, halfH, half,
             tw: hr + Math.round(tok * 1.2) + 8 + rt,          // the towers' distance from the centre line
             wv: hr + Math.round(ph / 2 + 40 * b) };            // the ward spot's distance from the centre line
  }

  // every screen box the scene may draw into, relative to the drawing's centre
  extents(g, t) {
    const ca = Math.cos(t), sa = Math.sin(t), boxes = [];
    const P = (u, v) => [u * ca + v * sa, -u * sa + v * ca];
    const box = ([x, y], w, h, dy = 0, dx = 0) => boxes.push({ x0: x + dx - w / 2, x1: x + dx + w / 2, y0: y + dy - h / 2, y1: y + dy + h / 2 });
    const baseDown = g.half * sa + g.halfH * ca;
    for (const s of [-1, 1]) {
      for (let i = 0; i < 4; i++) {                                              // the rotated base, in four slices
        const v0 = -g.halfH + (i * g.halfH) / 2, v1 = v0 + g.halfH / 2;
        const cs = [[-g.half, v0], [g.half, v0], [-g.half, v1], [g.half, v1]].map(([uu, vv]) => P(s * 4 * g.sp + uu, vv));
        boxes.push({ x0: Math.min(...cs.map((q) => q[0])), x1: Math.max(...cs.map((q) => q[0])),
                     y0: Math.min(...cs.map((q) => q[1])), y1: Math.max(...cs.map((q) => q[1])) });
      }
      box(P(s * 4 * g.sp, 0), 100, 16, s > 0 && t > STEEP ? -baseDown - 12 : baseDown + 12);   // base label
    }
    const cw = g.maxCols * g.D + 46, chh = g.maxRows * g.D, th = this.labels ? 2 * g.tok + 8 : 2 * g.tok + 48;
    const chip = (p, n, r) => box(p, CHIP.w, CHIP.h, 0, (n === 'A' ? -1 : 1) * (tokHalf(n, r) + 5 + CHIP.w / 2));
    for (let c = -4; c <= 4; c++) {
      for (const s of [-1, 1]) {
        box(P(c * g.sp, s * g.hr / 2), cw, chh);                                // a full cluster with its count
        box(P(c * g.sp, s * g.hr), 2 * g.tok + 8, th);                          // a champion (with its floats)
        if (this.labels) chip(P(c * g.sp, s * g.hr), s < 0 ? 'A' : 'B', g.tok);
      }
    }
    for (const [c, s] of [[-2, 1], [2, -1]]) {
      const tp = P(c * g.sp, s * g.tw), r = g.rt + 8;
      box(tp, 2 * r, 2 * r);
      box(tp, 90, 16, s * (g.rt + 18));                                         // tower label
      const off = g.rt + 14 + (g.maxCols * g.D) / 2, side = -Math.sign(c);    // crashed minions, on the lane side
      box([tp[0] + side * off * ca, tp[1] - side * off * sa], g.maxCols * g.D + 30, chh);
    }
    const w = P(0, -g.wv);
    box(w, g.pw + 4, g.ph + 4);                                                 // the ward spot
    box(w, Math.max(g.pw, 180), 22, -g.ph / 2 - 13);                            // its vision marks
    box([w[0] - g.pw / 2 - 50, w[1]], 84, 18);                                  // "Ward spot", on the open side
    if (this.labels) {
      for (const n of SEATS) {
        box([w[0] + (n === 'A' ? -1 : 1) * (g.pw / 2 - g.tok - 5), w[1]], CHIP.w, CHIP.h, 0,          // at the ward spot
            (n === 'A' ? -1 : 1) * (g.ph / 2 + 7 + CHIP.w / 2));
        chip(P((n === 'A' ? -4 : 4) * g.sp, (n === 'A' ? 1 : -1) * g.hr / 2), n, g.tok);             // away, at the base
      }
    }
    return { boxes, x0: Math.min(...boxes.map((q) => q.x0)), x1: Math.max(...boxes.map((q) => q.x1)),
             y0: Math.min(...boxes.map((q) => q.y0)), y1: Math.max(...boxes.map((q) => q.y1)) };
  }

  layout(force = false) {
    const W = innerWidth, H = innerHeight, obs = obstacles(), free = freeRect(obs);
    const key = `${W}x${H}:${+this.labels}:${obs.map((o) => [o.x0, o.y0, o.x1, o.y1].map(Math.round).join(',')).join(';')}`;
    if (key === this.key && !force) return false;
    this.key = key;
    this.W = W; this.H = H; this.safe = free;
    // the legend of the balls sits in the free area's top-left corner, so the drawing keeps clear of it as well
    const short = free.x1 - free.x0 < 640;
    this.legend = { x: free.x0 + 6, y: free.y0 + 12, text: short ? 'Each ball is 1 HP of minions' : 'Each ball is 1 HP of minions (a new wave is 8)' };
    this.block = [...obs, { x0: free.x0, y0: free.y0, x1: free.x0 + (short ? 200 : 300), y1: free.y0 + 22 }];
    const e = 14, B = { x0: e, y0: e, x1: W - e, y1: H - e };
    this.bounds = B;
    // the angle follows the free area's diagonal, kept between 25 and 55 degrees (a tall area, as on an upright phone,
    // takes the steeper ones; other angles are tried if it does not fit); then the largest scale at which every box finds
    // a place
    const t0 = Math.round(Math.max(25, Math.min(55, (Math.atan2(free.y1 - free.y0, free.x1 - free.x0) * 180) / Math.PI)));
    const angles = [...new Set([t0, 25, 40, 32, ...(t0 > 40 ? [48, 55] : [])])].map((d) => (d * Math.PI) / 180);
    const scales = [];
    for (let a = 1.6; a >= 0.3; a *= 0.97) scales.push(a);
    const memo = new Map();
    const fit = (i) => {
      if (!memo.has(i)) {
        const g = this.sizes(scales[i]);
        let r = null;
        for (const t of angles) {
          const ex = this.extents(g, t), c = placeDrawing(ex, B, this.block, free);
          if (c) { r = { g, t, c }; break; }
        }
        memo.set(i, r);
      }
      return memo.get(i);
    };
    let lo = 0, hi = scales.length - 1, best;
    if (fit(hi)) {                                       // a smaller drawing fits wherever a larger one does
      while (lo < hi) { const mid = (lo + hi) >> 1; if (fit(mid)) hi = mid; else lo = mid + 1; }
      best = fit(lo);
    } else {
      const g = this.sizes(scales[hi]), t = angles[0], ex = this.extents(g, t);
      best = { g, t, c: { x: (free.x0 + free.x1) / 2 - (ex.x0 + ex.x1) / 2, y: (free.y0 + free.y1) / 2 - (ex.y0 + ex.y1) / 2 } };
    }
    const { g, t, c } = best;
    Object.assign(this, g, { k: g.b, t, ca: Math.cos(t), sa: Math.sin(t) });
    this.cx = c.x;
    this.cy = c.y;
    this.el.setAttribute('viewBox', `0 0 ${W} ${H}`);
    this.packets.clear();
    this.build();
    return true;
  }

  P(u, v) { return { x: this.cx + u * this.ca + v * this.sa, y: this.cy - u * this.sa + v * this.ca }; }
  at(c, v = 0) { return this.P(c * this.sp, v); }
  towerAt(n) { return n === 'A' ? this.at(-2, this.tw) : this.at(2, -this.tw); }
  wAt() { return this.at(0, -this.wv); }

  // The bush at the ward spot, in the bases' style (a translucent green fill, a thin green outline): one silhouette whose
  // top edge is a row of grass spikes.
  drawBush(g, w, pw, ph) {
    const yb = w.y + ph * 0.40, L = w.x - pw * 0.46, Rr = w.x + pw * 0.46;
    const tips = [[-0.40, 0.62], [-0.27, 0.86], [-0.13, 0.74], [0, 0.98], [0.14, 0.78], [0.28, 0.9], [0.41, 0.6]];
    let d = `M${L} ${yb} Q${L - pw * 0.02} ${yb - ph * 0.3} ${w.x + tips[0][0] * pw} ${yb - tips[0][1] * ph}`;
    for (let i = 1; i < tips.length; i++) {
      const [x0] = tips[i - 1], [x1, h1] = tips[i];
      d += ` L${w.x + ((x0 + x1) / 2) * pw} ${yb - ph * 0.42} L${w.x + x1 * pw} ${yb - h1 * ph}`;
    }
    d += ` Q${Rr + pw * 0.02} ${yb - ph * 0.3} ${Rr} ${yb} Q${w.x} ${yb + ph * 0.12} ${L} ${yb} Z`;
    svg('path', { d, class: 'bush-shape' }, g);
  }

  // Ball positions of a cluster of `hp` balls, relative to its centre, filled column by column from the back (index 0)
  // to the front, so lost HP leaves from the front edge. `dir` is +1 for a side marching towards B (A), -1 for B.
  balls(hp, dir) {
    const mc = this.maxCols || 8;
    const cols = hp <= 4 ? Math.max(hp, 1) : hp <= 8 ? 4 : hp <= 12 ? 6 : hp <= 24 ? mc : Math.ceil(hp / (mc === 6 ? 4 : 3));
    const rows = Math.max(1, Math.ceil(hp / cols)), D = this.D;
    const w = cols * D, h = rows * D, out = [];
    for (let i = 0; i < hp; i++) {
      const c = Math.floor(i / rows), r = i % rows;
      out.push([dir * (-w / 2 + D / 2 + c * D), -h / 2 + D / 2 + r * D]);
    }
    return { w, h, pos: out };
  }

  // Draw (or update) one cluster: balls appear and fade with the HP the engine reports for this frame.
  cluster(key, n, hp, cx, cy, opts = {}) {
    const dir = n === 'A' ? 1 : -1, L = this.balls(hp, dir);
    let it = this.packets.get(key);
    if (!it) {
      const g = svg('g', { class: 'mv cl' }, opts.parent || this.gPk);
      it = { g, list: [], label: svg('text', { class: 'dot-count', y: 4 }, g), title: svg('title', {}, g) };
      this.packets.set(key, it);
      g.style.transform = `translate(${cx}px, ${cy}px)`;
    }
    const reduced = document.body.classList.contains('reduced');
    while (it.list.length > hp) {                     // HP lost: the front balls fade out
      const b = it.list.pop();
      if (reduced) b.remove();
      else { b.classList.add('gone'); setTimeout(() => b.remove(), 450); }
    }
    while (it.list.length < hp) {
      const [dx, dy] = L.pos[it.list.length];
      const b = svg('circle', { r: this.R, class: 'mb' + (reduced ? '' : ' new') }, it.g);
      b.style.transform = `translate(${dx}px, ${dy}px)`;
      if (!reduced) requestAnimationFrame(() => requestAnimationFrame(() => b.classList.remove('new')));
      it.list.push(b);
    }
    it.list.forEach((b, i) => {
      b.style.transform = `translate(${L.pos[i][0]}px, ${L.pos[i][1]}px)`;
      b.setAttribute('fill', COL[n]);
      b.setAttribute('fill-opacity', opts.rear ? 0.55 : 1);
      b.setAttribute('stroke', opts.siege ? TH.siege : 'none');
    });
    it.label.textContent = hp;
    it.label.setAttribute('x', dir * -(L.w / 2 + 5));
    it.label.setAttribute('text-anchor', dir > 0 ? 'end' : 'start');
    it.title.textContent = opts.title || '';
    it.g.style.transform = `translate(${cx}px, ${cy}px)`;
    it.w = L.w;
    it.h = L.h;
    return it;
  }

  build() {
    const s = this.el, { W, H, hr, half, halfH, sp } = this;
    const deg = (-this.t * 180) / Math.PI;
    s.innerHTML = '';
    svg('rect', { x: 0, y: 0, width: W, height: H, class: 'field' }, s);
    const live = S.mode === 'play' && S.live;
    const word = { A: live ? 'Your' : 'A', B: live ? 'Opponent' : 'B' };
    const pts = (list) => list.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ');
    const line = (p, q, cls, extra = {}) => svg('line', { x1: p.x, y1: p.y, x2: q.x, y2: q.y, class: cls, ...extra }, s);
    const lab = (p, t, dy = 0, cls = 'l-label', anchor = 'middle') => {
      const e = svg('text', { x: p.x, y: p.y + dy, 'text-anchor': anchor, class: cls }, s); e.textContent = t; return e; };
    // the road band, its cells and centre line
    const u0 = -4 * sp - half, u1 = 4 * sp + half;
    svg('polygon', { points: pts([this.P(u0, -hr), this.P(u1, -hr), this.P(u1, hr), this.P(u0, hr)]), class: 'road' }, s);
    for (let c = -3.5; c <= 3.5; c += 1) line(this.at(c, -hr + 3), this.at(c, hr - 3), 'cellsep');
    line(this.P(u0 + 10, 0), this.P(u1 - 10, 0), 'midline');
    // bases, rotated with the road; labels upright below them
    const baseDown = half * this.sa + halfH * this.ca;
    for (const n of SEATS) {
      const c = this.at(n === 'A' ? -4 : 4);
      const g = svg('g', { class: 'base', transform: `translate(${c.x.toFixed(1)} ${c.y.toFixed(1)}) rotate(${deg.toFixed(2)})` }, s);
      svg('rect', { x: -half, y: -halfH, width: 2 * half, height: 2 * halfH, rx: 10 * this.k, fill: SOFT[n], stroke: COL[n], 'stroke-width': 1.2 }, g);
      svg('title', {}, g).textContent = `${word[n]} base: waves start here; an away champion waits here`;
      lab(c, `${word[n]} base`, n === 'B' && this.t > STEEP ? -baseDown - 8 : baseDown + 16);
    }
    // the ward spot W beside cell 0: a bush in the bases' style (a translucent green fill and a thin green outline),
    // with a few spikes so it reads as grass; joined to the road's
    // edge by a dashed path. A warding champion stands at one end of it.
    const w = this.wAt(), { pw, ph } = this;
    line(this.at(0, -this.wv + ph / 2), this.at(0, -hr), 'wpath');
    const wg = svg('g', { class: 'wnode' }, s);
    this.drawBush(wg, w, pw, ph);
    svg('title', {}, wg).textContent = 'The ward spot: a bush beside the middle of the lane. A champion here can place a ward, which gives vision for a few turns';
    lab({ x: w.x - pw / 2 - 8, y: w.y }, 'Ward spot', 4.5, 'l-label ward', 'end');
    // towers just off the road's edge: a body seen from above; the plate ring is drawn per frame
    this.towerG = {};
    for (const n of SEATS) {
      const tp = this.towerAt(n);
      const g = svg('g', { class: 'tower' }, s);
      svg('circle', { cx: tp.x, cy: tp.y, r: this.rt, fill: SOFT[n], stroke: COL[n], 'stroke-width': 1.6 }, g);
      svg('circle', { cx: tp.x, cy: tp.y, r: this.rt * 0.42, fill: COL[n], opacity: 0.85 }, g);
      this.towerG[n] = { g, title: svg('title', {}, g) };
      lab(tp, `${word[n]} tower`, n === 'A' ? this.rt + 22 : -this.rt - 14);
    }
    // the unit of the balls, once, in the free upper-left corner of the scene
    const lg = svg('g', { transform: `translate(${this.legend.x} ${this.legend.y})` }, s);
    svg('circle', { r: Math.min(this.R, 6), cx: Math.min(this.R, 6), cy: -4, class: 'legend-dot' }, lg);
    const lt = svg('text', { x: 2 * Math.min(this.R, 6) + 7, y: 0, class: 'l-label' }, lg);
    lt.textContent = this.legend.text;
    this.gPlates = svg('g', {}, s);
    this.gSiege = svg('g', {}, s);
    this.gPk = svg('g', {}, s);
    this.gContact = svg('g', {}, s);
    this.gChips = svg('g', {}, s);
    this.gPins = svg('g', {}, s);
    this.gFx = svg('g', {}, s);
    this.pins = {};
    // what a status chip must not cover, apart from the waves and champions of the frame: towers and their labels, the
    // ward spot with its label and vision marks, the base labels, the legend and the surface
    const bx = (p, bw, bh, dy = 0, dx = 0) => ({ x0: p.x + dx - bw / 2, x1: p.x + dx + bw / 2, y0: p.y + dy - bh / 2, y1: p.y + dy + bh / 2 });
    this.fixed = [...this.block];
    for (const n of SEATS) {
      const tp = this.towerAt(n), r = this.rt + 8;
      this.fixed.push(bx(tp, 2 * r, 2 * r), bx(tp, 90, 18, n === 'A' ? this.rt + 18 : -this.rt - 18));
      this.fixed.push(bx(this.at(n === 'A' ? -4 : 4), 100, 18, n === 'B' && this.t > STEEP ? -baseDown - 12 : baseDown + 12));
    }
    this.fixed.push(bx(w, pw + 2, ph + 2), bx(w, Math.max(pw, 180), 22, -ph / 2 - 13), bx(w, 84, 18, 0, -pw / 2 - 50));
    this.chips = {};
    if (this.labels) {
      for (const n of SEATS) {
        const g = svg('g', { class: 'mv tlab' }, this.gChips);
        const bg = svg('rect', { class: 'bg', rx: 9, height: CHIP.h, width: CHIP.w }, g);
        const t1 = svg('text', { x: 8, y: 14 }, g);
        const nm = svg('tspan', { class: 'nm' }, t1), lv = svg('tspan', { class: 'lv' }, t1);
        const bar0 = svg('rect', { class: 'bar0', x: 8, y: 21, width: 48, height: 6, rx: 3 }, g);
        const bar = svg('rect', { x: 8, y: 21, height: 6, rx: 3, fill: COL[n] }, g);
        const t2 = svg('text', { x: 62, y: 27.5 }, g);
        const hp = svg('tspan', { class: 'hp' }, t2), fx = svg('tspan', { class: 'fx', dx: 5 }, t2);
        const aw = svg('text', { class: 'aw', x: 8, y: 27.5 }, g);
        this.chips[n] = { g, bg, nm, lv, bar0, bar, t2, hp, fx, aw, t1, title: svg('title', {}, g), placed: false };
      }
    }
    for (const n of SEATS) {
      const g = svg('g', { class: 'mv tok' }, this.gPins);
      const r = this.tok;
      const head = n === 'A' ? svg('circle', { r, fill: COL[n], 'stroke-width': 2, class: 'head' }, g)
        : svg('rect', { x: -r * 0.82, y: -r * 0.82, width: r * 1.64, height: r * 1.64, rx: 2, transform: 'rotate(45)', fill: COL[n], 'stroke-width': 2, class: 'head' }, g);
      const letter = svg('text', { y: 4.3, 'text-anchor': 'middle', class: 'pin-letter' }, g);
      letter.textContent = n;
      this.pins[n] = { g, head, title: svg('title', {}, g) };
    }
  }

  arc(cx, cy, r, a0, a1) {
    const p = (a) => [cx + r * Math.cos((a * Math.PI) / 180), cy + r * Math.sin((a * Math.PI) / 180)];
    const [x0, y0] = p(a0), [x1, y1] = p(a1);
    return `M${x0.toFixed(1)} ${y0.toFixed(1)} A${r} ${r} 0 ${a1 - a0 > 180 ? 1 : 0} 1 ${x1.toFixed(1)} ${y1.toFixed(1)}`;
  }

  update(f) {
    const { hr, k } = this, occ = [];
    const clBox = (x, y, it) => occ.push({ x0: x - it.w / 2 - 22, x1: x + it.w / 2 + 22, y0: y - it.h / 2, y1: y + it.h / 2 });
    // plates around each tower: a ring of three; a lost plate is a dashed outline
    this.gPlates.innerHTML = '';
    for (const n of SEATS) {
      const t = f.towers[n], c = this.towerAt(n), rr = this.rt + 5;
      for (let i = 0; i < t.plates_total; i++) {
        const a0 = -90 + i * 120 + 10, lost = i < t.plates_lost;
        svg('path', { d: this.arc(c.x, c.y, rr, a0, a0 + 100), fill: 'none', stroke: COL[n], 'stroke-width': lost ? 1.2 : 4,
                      'stroke-dasharray': lost ? '3 3' : null, opacity: lost ? 0.6 : 1, 'stroke-linecap': 'round' }, this.gPlates);
      }
      const left = t.plates_total - t.plates_lost;
      this.towerG[n].title.textContent = `${S.names[n]}'s tower: ${left} of ${t.plates_total} plates left, damage ${t.damage} of ${t.damage_max}`;
      // a placed ward: that side's eye in its colour above the patch, with the turns of vision left
      const vis = f.status[n].vision;
      if (vis > 0) {
        const w = this.wAt(), g = svg('g', {}, this.gPlates);
        const bg = svg('rect', { y: -9, height: 18, rx: 9, class: 'wbadge' }, g);
        svg('path', { d: 'M-9 0 Q0 -7 9 0 Q0 7 -9 0 Z', fill: COL[n] }, g);
        svg('circle', { r: 2.6, class: 'wbadge-pupil' }, g);
        const t = svg('text', { x: 14, y: 4, class: 'wcount', fill: COL[n] }, g);
        t.textContent = turns(vis);
        const bw = 32 + t.getComputedTextLength();               // A's badge left of the eye, B's right of it
        bg.setAttribute('x', -14);
        bg.setAttribute('width', bw);
        g.setAttribute('transform', `translate(${w.x + (n === 'A' ? -bw + 8 : 18)} ${w.y - this.ph / 2 - 13})`);
        svg('title', {}, g).textContent = `${S.names[n]} placed a ward: vision for ${turns(vis)} more`;
      }
    }
    // crashed minions, beside the tower they attack, on its lane side (towards the middle of the road)
    for (const n of SEATS) {
      const [, hp] = f.lane.siege[n], key = `S${n}`;
      if (hp > 0) {
        const tn = other(n), tw = this.towerAt(tn), L = this.balls(hp, 1), side = tn === 'A' ? 1 : -1;
        const off = this.rt + 14 + L.w / 2, x = tw.x + side * off * this.ca, y = tw.y - side * off * this.sa;
        clBox(x, y, this.cluster(key, n, hp, x, y, { parent: this.gSiege, siege: true,
          title: `${hp} HP of ${S.names[n]}'s minions crashed under the tower, hitting it` }));
      } else if (this.packets.has(key)) { this.packets.get(key).g.remove(); this.packets.delete(key); }
    }
    // walking waves: A's on the upper-left half of the road, B's on the lower-right half
    const seen = new Set(['SA', 'SB']);
    for (const n of SEATS) {
      const v = n === 'A' ? -hr / 2 + 1 : hr / 2 - 1;
      let lastX = null, stack = 0;
      f.lane.packets[n].forEach(([cx, hp, front], i) => {
        const key = `${n}${i}`;
        seen.add(key);
        stack = cx === lastX ? stack + 1 : 0;
        lastX = cx;
        const p = this.P(cx * this.sp + (n === 'A' ? -1 : 1) * stack * 14, v);
        clBox(p.x, p.y, this.cluster(key, n, hp, p.x, p.y, { rear: !front,
          title: `${S.names[n]}'s ${front ? 'leading wave' : 'wave behind'}: ${hp} HP of minions` }));
      });
    }
    for (const [key, it] of this.packets) if (!seen.has(key)) { it.g.remove(); this.packets.delete(key); }
    // where the two waves fight
    this.gContact.innerHTML = '';
    if (f.lane.contact !== null && f.lane.contact !== undefined) {
      const p = this.at(f.lane.contact), r = 7.5 * Math.min(1, k + 0.1);
      svg('circle', { cx: p.x, cy: p.y, r, class: 'contact' }, this.gContact);
      svg('path', { d: `M${p.x - r * 0.45} ${p.y - r * 0.45} l${r * 0.9} ${r * 0.9} m0 ${-r * 0.9} l${-r * 0.9} ${r * 0.9}`, class: 'contact-x',
                    'stroke-width': 1.5, 'stroke-linecap': 'round' }, this.gContact);
      svg('title', {}, this.gContact).textContent = 'The two waves are fighting here';
      occ.push({ x0: p.x - r - 2, x1: p.x + r + 2, y0: p.y - r - 2, y1: p.y + r + 2 });
    }
    // champions: A on the upper-left edge of its cell, B on the lower-right edge; beside W when warding; away, faded
    // at its own base in the half its waves do not use
    const pos = {};
    for (const n of SEATS) {
      const cell = f.cells[n];
      if (cell !== null && cell !== undefined) pos[n] = { ...this.at(cell, n === 'A' ? -hr : hr), mode: 'lane' };
      else if (f.at_ward[n]) { const w = this.wAt(); pos[n] = { x: w.x + (n === 'A' ? -1 : 1) * (this.pw / 2 - this.tok - 5), y: w.y, mode: 'ward' }; }
      else pos[n] = { ...this.at(n === 'A' ? -4 : 4, n === 'A' ? hr / 2 : -hr / 2), mode: 'away' };
    }
    for (const n of SEATS) {
      const p = pos[n], pin = this.pins[n], st = f.status[n];
      pin.g.style.transform = `translate(${p.x}px, ${p.y}px)`;
      pin.g.style.opacity = p.mode === 'away' ? 0.42 : 1;
      pin.g.classList.toggle('inbush', p.mode === 'ward');       // see-through, outline kept (the away look is faded and dashed)
      pin.head.setAttribute('stroke-dasharray', p.mode === 'away' ? '3 3' : '');
      pin.title.textContent = `${S.names[n]}: ${p.mode === 'away' ? (st.status === 'dead' ? 'dead' : 'recalling') + ', back in ' + turns(st.returns_in)
        : p.mode === 'ward' ? `in the bush at the ward spot, level ${st.level}, ${st.hp} of ${st.hp_max} HP. ${BUSH_NOTE}`
        : 'level ' + st.level + ', ' + st.hp + ' of ' + st.hp_max + ' HP'}`;
    }
    // with labels on, the turn's HP losses and minion hits are written in the chips instead of floating by the tokens
    const chipFx = { A: '', B: '' };
    if (this.labels && f.sub === 'tick' && f.tick) for (const n of SEATS) { if (f.tick.minion_fire[n].units > 0 && pos[n].mode === 'lane') chipFx[n] = 'minions hit'; }
    if (this.labels && f.sub === 'end' && f.turn) {
      for (const n of SEATS) { const h = S.G.turns[f.turn - 1].delta[n].hp_taken; if (h > 0 && pos[n].mode !== 'away') chipFx[n] = `−${h} HP`; }
    }
    if (this.labels) this.placeChips(f, pos, occ, chipFx);
    // per-tick facts at the ticks, the turn's HP losses at its end
    const key = `${f.turn || ''}:${f.sub}:${f.label}`;
    if (key === this.fxKey) return;
    this.fxKey = key;
    this.gFx.innerHTML = '';
    const away = (n) => (n === 'A' ? -1 : 1) * (this.tok + 9) + (n === 'A' ? 0 : 8);   // above A's token, below B's
    if (f.sub === 'tick' && f.tick) {
      for (const n of SEATS) {
        const val = f.tick.wave_removed[n], ef = f.lane.packets[other(n)].find((q) => q[2]);
        if (val > 0 && ef) {
          const ew = this.balls(ef[1], 1).w, toRight = n === 'A';
          const p = this.at(ef[0], other(n) === 'A' ? -hr / 2 : hr / 2);
          const t = svg('text', { x: p.x + (toRight ? -ew / 2 - 6 : ew / 2 + 6), y: p.y + 4,
                                  'text-anchor': toRight ? 'end' : 'start', class: 'float' }, this.gFx);
          t.textContent = `−${val}`;
        }
        if (!this.labels && f.tick.minion_fire[n].units > 0 && pos[n].mode === 'lane') {
          const t = svg('text', { x: pos[n].x, y: pos[n].y + away(n), 'text-anchor': 'middle', class: 'float hit' }, this.gFx);
          t.textContent = 'minions hit';
        }
      }
    } else if (f.sub === 'end' && f.turn) {
      const tr = S.G.turns[f.turn - 1];
      for (const n of SEATS) {
        const h = tr.delta[n].hp_taken;
        if (!this.labels && h > 0 && pos[n].mode !== 'away') {
          const y = pos[n].mode === 'ward' ? pos[n].y - this.ph / 2 - 6 : pos[n].y + away(n);
          const t = svg('text', { x: pos[n].x, y, 'text-anchor': 'middle', class: 'float hit' }, this.gFx);
          t.textContent = `−${h} HP`;
        }
      }
    }
  }

  // Each champion's chip goes to the first spot around its token that covers nothing (towers, the ward spot, labels,
  // the frame's waves, the other champion and its chip, the surface); if every spot covers something, to the one that
  // covers least. A's spots start on its left (outside the road), B's on its right.
  placeChips(f, pos, occ, fx) {
    const h = CHIP.h, ext = (n) => tokHalf(n, this.tok);
    const ORDER = { A: ['left', 'upleft', 'downleft', 'up', 'down', 'upright', 'right', 'downright'],
                    B: ['right', 'downright', 'upright', 'down', 'up', 'downleft', 'left', 'upleft'] };
    const spot = (k, w, e) => ({ left: [-e - 5 - w, -h / 2], right: [e + 5, -h / 2], up: [-w / 2, -e - 5 - h], down: [-w / 2, e + 5],
                                 upleft: [-e - 2 - w, -e - 2 - h], upright: [e + 2, -e - 2 - h], downleft: [-e - 2 - w, e + 2], downright: [e + 2, e + 2] }[k]);
    const B = this.bounds, tokBox = (n) => { const e = ext(n) + 2; return { x0: pos[n].x - e, x1: pos[n].x + e, y0: pos[n].y - e, y1: pos[n].y + e }; };
    const taken = [];
    this.chipBoxes = {};
    for (const n of SEATS) {
      const c = this.chips[n], st = f.status[n], p = pos[n], present = st.status === 'present';
      c.nm.textContent = S.head[n];
      c.lv.textContent = ` · Level ${st.level}`;
      for (const e of [c.bar0, c.bar, c.t2]) e.style.display = present ? '' : 'none';
      c.aw.style.display = present ? 'none' : '';
      if (present) {
        c.bar.setAttribute('width', (48 * Math.max(0, Math.min(1, st.hp / st.hp_max))).toFixed(1));
        c.hp.textContent = st.hp;
        c.fx.textContent = fx[n];
      } else c.aw.textContent = `${st.status === 'dead' ? 'Dead' : 'Recalling'}, back in ${turns(st.returns_in)}`;
      const w = Math.ceil(Math.max(c.t1.getComputedTextLength(), present ? 54 + c.t2.getComputedTextLength() : c.aw.getComputedTextLength())) + 16;
      c.bg.setAttribute('width', w);
      c.title.textContent = `${S.names[n]}: level ${st.level}, ${present ? `${st.hp} of ${st.hp_max} HP` : st.status === 'dead' ? 'dead' : 'recalling'}`;
      c.g.style.opacity = p.mode === 'away' ? 0.8 : 1;
      const block = [...this.fixed, ...occ, ...taken, tokBox(other(n))];
      let best = null;
      for (const k of ORDER[n]) {
        const [dx, dy] = spot(k, w, p.mode === 'ward' ? Math.max(ext(n), this.ph / 2 + 2) : ext(n));   // clear of the patch
        const b = { x0: p.x + dx, y0: p.y + dy, x1: p.x + dx + w, y1: p.y + dy + h };
        let cost = 0;
        for (const o of block) cost += overlap(b, o);
        cost += 4 * (w * h - overlap(b, B));                    // off the screen counts four times
        if (!best || cost < best.cost) best = { b, cost };
        if (cost === 0) break;
      }
      taken.push(best.b);
      this.chipBoxes[n] = best.b;
      if (!c.placed) { c.g.style.transition = 'none'; c.placed = true; }
      c.g.style.transform = `translate(${best.b.x0}px, ${best.b.y0}px)`;
      if (c.g.style.transition) requestAnimationFrame(() => { c.g.style.transition = ''; });
    }
  }
}

// ------------------------------------------------------------------------------------------- the frames of a turn, in words
// Words for a tick come only from that tick's facts in the trace (lanegame/demo/frames.py): each champion's cell, the
// waves and their contact, the HP each champion removed, the HP that crashed in that tick, minion fire on a champion.
// Turn-level events (trades, tower shots at champions, ganks, deaths, Recalls, wards, levels, gold) appear only at the
// end frame, in the turn's own outcome text.
const Cap = (t) => t[0].toUpperCase() + t.slice(1);
const own = (n) => (S.short[n] === 'You' ? 'your' : `${S.short[n]}'s`);          // your, the opponent's, Self-play's
const they = (n) => Cap(S.short[n]);
const verb = (n, you, other) => (S.short[n] === 'You' ? you : other);
function cellName(c) {
  if (c === 0) return 'the lane';
  const n = c < 0 ? 'A' : 'B', d = Math.abs(c);
  return d === 1 ? `${own(n)} side of the lane` : d === 2 ? `${own(n)} tower` : d === 3 ? `behind ${own(n)} tower` : `${own(n)} base`;
}
const toCell = (c) => (Math.abs(c) === 3 ? cellName(c) : `to ${cellName(c)}`);    // "behind your tower" takes no "to"
const placeOf = (f, n) => (f.at_ward[n] ? 'W' : f.cells[n] === null || f.cells[n] === undefined ? null : f.cells[n]);
const frontOf = (f, n) => f.lane.packets[n].find((q) => q[2]) || null;

function tickWords(prev, f) {
  const out = [], tk = f.tick;
  for (const n of SEATS) {
    const a = placeOf(prev, n), b = placeOf(f, n);
    if (b === null || a === b) continue;
    if (b === 'W') out.push(`${they(n)} ${verb(n, 'walk', 'walks')} into the bush.`);
    else if (a === 'W') out.push(`${they(n)} ${verb(n, 'leave', 'leaves')} the bush and ${verb(n, 'move', 'moves')} ${toCell(b)}.`);
    else out.push(`${they(n)} ${verb(n, 'move', 'moves')} ${toCell(b)}.`);
  }
  const ct = f.lane.contact;
  if (ct !== null && ct !== undefined) {
    out.push(`${Cap(own('A'))} wave and ${own('B')} wave ${prev.lane.contact === ct ? 'keep fighting' : 'fight'} at ${cellName(ct)}.`);
  } else {
    for (const n of SEATS) {
      const a = frontOf(prev, n), b = frontOf(f, n);
      if (a && b && a[0] !== b[0] && !tk.crashed[n]) out.push(`${Cap(own(n))} wave walks ${toCell(b[0])}.`);
    }
  }
  const r = tk.wave_removed;
  if (r.A > 0 && r.B > 0) out.push(`${they('A')} ${verb('A', 'remove', 'removes')} ${r.A} HP of enemy minions; ${S.short.B} ${verb('B', 'remove', 'removes')} ${r.B}.`);
  else for (const n of SEATS) if (r[n] > 0) out.push(`${they(n)} ${verb(n, 'remove', 'removes')} ${r[n]} HP of enemy minions.`);
  for (const n of SEATS) {
    if (tk.crashed[n] > 0) out.push(`${Cap(own(n))} wave crashes into ${own(other(n))} tower (${tk.crashed[n]} HP).`);
    else {
      const was = prev.lane.siege[n][1], now = f.lane.siege[n][1];
      if (now < was) out.push(now > 0 ? `${Cap(own(n))} crashed minions under ${own(other(n))} tower drop to ${now} HP.`
        : `${Cap(own(n))} crashed minions under ${own(other(n))} tower are gone.`);
    }
  }
  for (const n of SEATS) if (tk.minion_fire[n].units > 0) out.push(`${Cap(own(other(n)))} minions hit ${S.short[n] === 'You' ? 'you' : S.short[n]}.`);
  return out.length ? out : ['Nothing changes in this tick.'];
}

// The heading, text and (at the end) outcomes of frame i of turn t.
function frameWords(t, frames, i, maxOut = 3) {
  const f = frames[i], tr = S.G.turns[t - 1], nt = tr.ticks.length;
  if (f.sub === 'start') {
    const chose = (n, first) => {
      const pt = tr.parts[SEATS.indexOf(n)], who = first ? they(n) : S.short[n];
      return pt.kind === 'absent' ? `${who} ${verb(n, 'are', 'is')} away` : `${who} chose ${actionWords(pt)}`;
    };
    return { head: `Both actions revealed`, text: `${chose('A', true)}; ${chose('B', false)}. `
      + `${nt === 2 ? 'Two ticks' : `${nt} ticks`} follow: the champions move, and the waves walk, fight and lose HP.` };
  }
  if (f.sub === 'tick') return { head: `Tick ${f.tick.tick} of ${nt}`, text: tickWords(frames[i - 1], f).join(' ') };
  const major = tr.outcomes.filter((e) => e.major), extra = tr.outcomes.length - Math.min(maxOut, major.length);
  return { head: `End of turn ${t}`,
           text: 'Trades, tower shots at champions, ganks, deaths, Recalls, wards, levels and gold are settled at the end of a turn.',
           outcomes: major.length ? major.slice(0, maxOut).map((e) => fill(e.text)) : ['No trades, crashes, Recalls or deaths.'],
           more: extra > 0 ? `+${extra} more` : null };
}

// While a turn is being stepped, its result is not settled yet: the graph, the history's outcome line and Details stay
// at the position before it until the end frame.
function resolving() { const s = S.ctrl.step; return s && s.i < s.frames.length - 1 ? s : null; }
function shownView() { const s = resolving(); return s ? s.t - 1 : S.ctrl.view; }
function shownProgress() { const s = resolving(); return s && S.live && s.t === S.ctrl.progress ? s.t - 1 : S.ctrl.progress; }

// ------------------------------------------------------------------------------------------- top bar
function renderTop() {
  $('modePlay').setAttribute('aria-pressed', String(S.mode === 'play'));
  $('modeWatch').setAttribute('aria-pressed', String(S.mode === 'watch'));
  const ctx = $('context');
  if (S.mode === 'play') {
    const opp = S.live ? S.G.meta.seats[1] : null;
    ctx.innerHTML = `<button class="btn" id="btnNew"${S.engine ? '' : ' disabled'}>New game</button>`;
    $('btnNew').addEventListener('click', () => $('newGame').showModal());
  } else {
    const opt = (g) => `<option value="${g.name}"${S.G && S.G.name === g.name ? ' selected' : ''}>${esc(GAME_NAMES[g.name] || g.label)}</option>`;
    const all = S.index ? S.index.games : [];
    let games = `<optgroup label="Recorded research games">${all.filter((g) => g.kind === 'research_record').map(opt).join('')}</optgroup>`
      + `<optgroup label="Generated for this demo">${all.filter((g) => g.kind !== 'research_record').map(opt).join('')}`
      + (S.G && S.G.name === 'new simulation' ? `<option value="__sim" selected>${esc(S.G.label)}</option>` : '') + '</optgroup>';
    ctx.innerHTML = `<label class="sel"><span class="sr-only">Game</span><select id="gamePick">${games}</select></label>
      <button class="btn" id="btnSim"${S.engine ? '' : ' disabled title="Needs the local engine"'}><span class="long">New game between agents</span><span class="short">New game</span></button>`;
    $('gamePick').addEventListener('change', (e) => { if (e.target.value !== '__sim') watch(e.target.value); });
    $('btnSim').addEventListener('click', () => $('newSim').showModal());
  }
  const kind = S.G ? S.G.meta.provenance.kind : null;
  $('recType').textContent = { research_record: 'Recorded research game', new_simulation: 'Generated game', human_game: 'Your game, live' }[kind] || '';
  $('recType').title = 'One game is an example, not evidence for M5\'s findings. See "About this game".';
}

// ------------------------------------------------------------------------------------------- status, banner, summary
function renderStatus(f) {
  for (const n of SEATS) {
    const st = f.status[n];
    const pct = Math.max(0, Math.min(100, (100 * st.hp) / st.hp_max));
    const body = st.status === 'present'
      ? `<span class="bar" role="img" aria-label="HP ${st.hp} of ${st.hp_max}"><i style="width:${pct}%;background:${COL[n]}"></i></span><span class="num">${st.hp} / ${st.hp_max} HP</span>`
      : `<span class="away">${st.status === 'dead' ? 'Dead' : 'Recalling'}, back in ${turns(st.returns_in)}</span>`;
    $(`st${n}`).innerHTML = `<div class="st-row">${glyph(n)}<span class="nm" title="${esc(S.names[n])}">${esc(S.names[n])}</span><span class="lvl">Level ${st.level}</span></div>
      <div class="st-row2">${body}</div>`;
  }
  const c = S.ctrl, v = c.view;
  // while a turn is stepped the tick count lives in the step panel only
  let lab = v === 0 ? 'Start' : f.sub === 'start' || f.sub === 'tick' ? `Turn ${f.turn}` : `After turn ${v}`;
  if (S.live && !c.stopped && !c.historical && !c.stepping) lab = `Turn ${c.progress + 1} of ${c.n} · your move`;
  else if (S.live && c.stopped && v === c.progress) lab = `Stopped after turn ${c.progress}`;
  $('turnLabel').textContent = lab;
}

function renderBanner() {
  const c = S.ctrl, hist = !!S.live && !c.stopped && c.historical;
  $('banner').hidden = !hist;
  $('turnPill').hidden = hist;
  document.body.classList.toggle('historical', hist);
  if (hist) $('bannerText').textContent = `Viewing ${c.view === 0 ? 'the start' : 'turn ' + c.view} · the game is at turn ${c.progress + 1}`;
}

function renderSummary(f) {
  const c = S.ctrl, v = c.view;
  $('summary').hidden = S.mode === 'watch';
  if (S.mode === 'watch') return renderTurnCard(f);
  const box = $('summary');
  if (v === 0) { box.innerHTML = `<span class="muted">Start of the game. Both champions begin at their own tower.</span>`; return; }
  const tr = S.G.turns[v - 1];
  if (f.sub === 'start' || f.sub === 'tick') {
    // the turn label beside this line already names the turn
    box.innerHTML = `<span>${glyph('A')}${esc(actionWords(tr.parts[0]))}</span><span>${glyph('B')}${esc(actionWords(tr.parts[1]))}</span>`;
    return;
  }
  const major = tr.outcomes.filter((e) => e.major);
  const minor = tr.outcomes.length - major.length;
  const text = major.length ? major.slice(0, 3).map((e) => esc(fill(e.text))).join(' · ') : 'No trades, crashes, Recalls or deaths.';
  const more = tr.outcomes.length - Math.min(3, major.length);
  box.innerHTML = `<b>Turn ${v}</b><span class="sum-text">${text}</span>`
    + (more > 0 || minor ? `<button class="linkbtn" id="sumMore">${more > 0 ? `+${more} more` : 'All events'}</button>` : '')
    + (c.stepping ? '' : '<button class="linkbtn" id="sumTicks">Show the ticks</button>');
  $('sumMore')?.addEventListener('click', () => openDetails('events'));
  $('sumTicks')?.addEventListener('click', () => S.ctrl.stepTurn(v));
}

// ------------------------------------------------------------------------------------------- history
function renderHistory() {
  const list = $('hList'), c = S.ctrl;
  if (S.rows > c.progress || list.children.length === 0) { list.innerHTML = ''; S.rows = -1; }
  list.querySelector('.pending')?.remove();
  for (let t = S.rows + 1; t <= c.progress; t++) {
    const li = document.createElement('li');
    li.dataset.p = t;
    if (t === 0) {
      li.className = 'row start';
      li.innerHTML = `<span class="n">0</span><span class="a">Start</span>`;
    } else {
      const tr = S.G.turns[t - 1];
      const ph = tr.outcomes.filter((e) => e.major).map(phrase).filter(Boolean);
      const a = actionWords(tr.parts[0]), b = actionWords(tr.parts[1]);
      li.className = 'row';
      li.innerHTML = `<span class="n">${t}</span><span class="a" title="${esc(a)}">${esc(a)}</span><span class="b" title="${esc(b)}">${esc(b)}</span>`
        + (ph.length ? `<span class="o">${esc(ph.slice(0, 2).join(' · '))}${ph.length > 2 ? ` +${ph.length - 2}` : ''}</span>` : '');
      li.setAttribute('aria-label', `Turn ${t}: ${S.names.A} ${a}; ${S.names.B} ${b}. ${ph.join('. ')}`);
    }
    li.addEventListener('click', () => S.ctrl.goto(+li.dataset.p));
    list.appendChild(li);
  }
  S.rows = c.progress;
  if (S.live && !c.stopped) {
    const li = document.createElement('li');
    li.className = 'row pending';
    li.innerHTML = `<span class="n">${c.progress + 1}</span><span class="a">${c.forced() ? 'Away' : 'Your move'}</span><span class="b" title="Chosen before your move; revealed with it">Chosen</span>`;
    list.appendChild(li);
  }
}

function highlightRows() {
  const list = $('hList');
  list.querySelectorAll('.viewed, .hover').forEach((e) => { e.classList.remove('viewed', 'hover'); e.removeAttribute('aria-current'); });
  const row = list.querySelector(`[data-p="${S.ctrl.view}"]`);
  if (row) {
    row.classList.add('viewed');
    row.setAttribute('aria-current', 'true');
    const lr = list.getBoundingClientRect(), rr = row.getBoundingClientRect();
    if (rr.top < lr.top || rr.bottom > lr.bottom) list.scrollTop += rr.top - lr.top - lr.height / 2;
  }
  if (S.hover !== null) list.querySelector(`[data-p="${S.hover}"]`)?.classList.add('hover');
  list.querySelectorAll('.resolving').forEach((e) => e.classList.remove('resolving'));
  const rs = resolving();
  if (rs) list.querySelector(`[data-p="${rs.t}"]`)?.classList.add('resolving');      // its outcomes come at the end frame
}

function renderNav() {
  const c = S.ctrl;
  $('btnFirst').disabled = $('btnPrev').disabled = c.view === 0;
  $('btnNext').disabled = $('btnLast').disabled = c.view >= c.progress;
}

// ------------------------------------------------------------------------------------------- gold-lead graph
function niceStep(m, target = 3) {
  for (const s of [0.5, 1, 2, 5, 10, 20, 25, 50, 100]) if (m / s <= target) return s;
  return 200;
}

// Blue above zero when A (you) has earned more gold, red below when B has. The y-axis is symmetric and at least ±2
// gold, so small changes stay small. The x-axis spans the turns played so far in a running game (at least 20), and the
// whole game otherwise. The compact graph lives in the surface; the large one in a sheet or section opened from it.
function renderGraph() {
  drawGraph($('graph'));
  if (S.graphBig) drawGraph($('graphB'));
}

// A plot lower than 90 px is a sparkline: the zero line, the two colours and the selected turn, with the value written
// beside it (#gVal) instead of axes and labels.
function drawGraph(el) {
  const c = S.ctrl, G = S.G, id = el.id;
  const H = Math.max(20, el.clientHeight || 140), mini = H < 90;
  const ahead = { A: S.live ? 'You' : S.short.A, B: S.live ? 'Opponent' : S.short.B };
  const label = (p) => {
    const v = G.positions[p].lead_A;
    return `${p === 0 ? 'Start' : 'Turn ' + p}: ${Math.abs(v) < 0.005 ? 'even' : `${v > 0 ? ahead.A : ahead.B} +${Math.abs(v).toFixed(2)}`}`;
  };
  // the value beside a sparkline is written first: in the console it shares the row, so it sets the plot's width
  const sv = shownView(), prog = shownProgress();
  if (id === 'graph') $('gVal').textContent = mini ? label(S.hover !== null && S.hover <= prog ? S.hover : sv) : '';
  const W = Math.max(120, el.clientWidth);
  const M = mini ? { l: 4, r: 4, t: 4, b: 4 } : { l: 30, r: 8, t: 21, b: 17 };
  const running = !!S.live && !c.stopped;
  const n = running ? Math.max(20, Math.ceil((prog + 1) / 10) * 10) : Math.max(prog, 1);
  const X = (p) => M.l + ((W - M.l - M.r) * p) / n;
  let mx = 0;
  for (let p = 0; p <= prog; p++) mx = Math.max(mx, Math.abs(G.positions[p].lead_A));
  const span = Math.max(mx, 2), step = niceStep(span, 2.5), top = step * Math.ceil(span / step);
  const Y = (v) => M.t + ((H - M.t - M.b) * (top - v)) / (2 * top);
  const y0 = Y(0);
  const pts = [];
  for (let p = 0; p <= prog; p++) pts.push(`${X(p).toFixed(1)} ${Y(G.positions[p].lead_A).toFixed(1)}`);
  const line = `M${pts.join(' L')}`;
  const area = `${line} L${X(prog).toFixed(1)} ${y0} L${X(0)} ${y0} Z`;
  const o = [];
  // a tie (the line on zero) stays neutral: each colour is clipped a pixel away from the baseline
  o.push(`<defs><clipPath id="${id}Up"><rect x="0" y="0" width="${W}" height="${y0 - 1}"/></clipPath>`
    + `<clipPath id="${id}Dn"><rect x="0" y="${y0 + 1}" width="${W}" height="${H}"/></clipPath></defs>`);
  for (let v = -top; v <= top + 1e-9; v += step) {
    const y = Y(v), zero = Math.abs(v) < 1e-9;
    if (mini && !zero) continue;
    o.push(`<line x1="${M.l}" x2="${W - M.r}" y1="${y}" y2="${y}" class="${zero ? 'g-zero' : 'g-grid'}"/>`);
    if (!mini) o.push(`<text x="${M.l - 6}" y="${y + 4}" text-anchor="end" class="g-axis">${zero ? '0' : Math.abs(v).toFixed(step < 1 ? 1 : 0)}</text>`);
  }
  const xs = niceStep(n, 6);
  if (!mini) for (let p = 0; p <= n; p += xs) o.push(`<text x="${X(p)}" y="${H - 4}" text-anchor="middle" class="g-axis">${p}</text>`);
  o.push(`<path d="${area}" fill="${COL.A}" fill-opacity="${TH.fill}" clip-path="url(#${id}Up)"/>`
    + `<path d="${area}" fill="${COL.B}" fill-opacity="${TH.fill}" clip-path="url(#${id}Dn)"/>`);
  o.push(`<path d="${line}" class="g-line" stroke="${TH.neutral}"/>`);
  o.push(`<path d="${line}" class="g-line" stroke="${COL.A}" clip-path="url(#${id}Up)"/>`
    + `<path d="${line}" class="g-line" stroke="${COL.B}" clip-path="url(#${id}Dn)"/>`);
  if (S.events && !mini) {
    for (const [p, lab] of [[94, 'Lane exit (94)'], [114, 'Turn 114']]) {
      if (p > n) continue;
      o.push(`<line x1="${X(p)}" x2="${X(p)}" y1="${M.t}" y2="${H - M.b}" class="g-mark"/><text x="${X(p) + 4}" y="${M.t + 24}" class="g-axis g-marklab">${lab}</text>`);
    }
    for (let t = 1; t <= prog; t++) {
      for (const e of G.turns[t - 1].outcomes) {
        const x = X(t), y = e.seat === 'A' ? M.t + 6 : H - M.b - 6, col = COL[e.seat];
        if (e.kind === 'death') o.push(`<path d="M${x - 3.5} ${y - 3.5} l7 7 m0 -7 l-7 7" stroke="${col}" stroke-width="1.8"/>`);
        else if (e.kind === 'recall') o.push(`<path d="M${x} ${y - 4} l4 7 h-8 z" fill="none" stroke="${col}" stroke-width="1.2"/>`);
        else if (e.kind === 'crash') o.push(`<rect x="${x - 2.3}" y="${y - 2.3}" width="4.6" height="4.6" fill="${col}"/>`);
        else if (e.kind === 'plate') o.push(`<rect x="${x - 3}" y="${y - 3}" width="6" height="6" transform="rotate(45 ${x} ${y})" class="g-plate" stroke="${col}" stroke-width="1.3"/>`);
      }
    }
  }
  const mark = (p, cls) => {
    const v = G.positions[p].lead_A, x = X(p), text = label(p);
    const dot = Math.abs(v) < 0.005 ? TH.muted : v > 0 ? COL.A : COL.B;
    if (mini) return `<g class="${cls}"><line x1="${x}" x2="${x}" y1="${M.t - 2}" y2="${H - M.b + 2}"/><circle cx="${x}" cy="${Y(v)}" r="3" style="stroke:${dot}"/></g>`;
    const w = text.length * 6.4 + 14, lx = Math.max(M.l, Math.min(W - M.r - w, x - w / 2));
    return `<g class="${cls}"><line x1="${x}" x2="${x}" y1="${M.t}" y2="${H - M.b}"/><circle cx="${x}" cy="${Y(v)}" r="4" style="stroke:${dot}"/>
      <rect x="${lx}" y="1" width="${w}" height="17" rx="3"/><text x="${lx + w / 2}" y="13" text-anchor="middle">${esc(text)}</text></g>`;
  };
  o.push(mark(sv, `g-sel${c.historical && running ? ' past' : ''}`));
  if (S.hover !== null && S.hover !== sv && S.hover <= prog) o.push(mark(S.hover, 'g-hover'));
  if (!mini) {
    o.push(`<text x="${M.l + 6}" y="${M.t + 12}" class="g-side" fill="${COL.A}">${esc(ahead.A)} ahead</text>`);
    o.push(`<text x="${M.l + 6}" y="${H - M.b - 6}" class="g-side" fill="${COL.B}">${esc(ahead.B)} ahead</text>`);
  }
  el.innerHTML = `<svg class="${mini ? 'g-mini' : ''}" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="Gold lead by turn. ${esc($('gNote').textContent)}">${o.join('')}</svg>`;
  el._geo = { M, W, n };
}

function graphTurn(ev) {
  const el = ev.currentTarget, g = el._geo;
  if (!g) return null;
  const r = el.getBoundingClientRect();
  const p = Math.round(((ev.clientX - r.left - g.M.l) / (r.width - g.M.l - g.M.r)) * g.n);
  return Math.max(0, Math.min(shownProgress(), p));
}

// The large graph: a sheet above the dock or beside the rail, a section inside the console. The full history: the
// dock's history column grows upward, the console opens it as a section; the rail always shows it. In the console the
// two sections take turns. Opening or closing one refits the lane once, so it stays clear of the surface.
function setGraphBig(big, quiet = false) {
  S.graphBig = !!big;
  if (S.graphBig && S.arr === 'console' && S.histOpen) setHistory(false, true);
  $('graphBig').hidden = !S.graphBig;
  document.body.classList.toggle('graph-big', S.graphBig);
  $('gExpand').textContent = S.graphBig ? 'Collapse' : 'Expand';
  $('gExpand').setAttribute('aria-expanded', String(S.graphBig));
  if (!quiet) refit();
}

function setHistory(open, quiet = false) {
  S.histOpen = !!open && S.arr !== 'rail';
  if (S.histOpen && S.arr === 'console' && S.graphBig) setGraphBig(false, true);
  document.body.classList.toggle('hist-open', S.histOpen);
  $('zHist').classList.toggle('pop', S.arr === 'dock' && S.histOpen);
  const t = S.arr === 'console' ? 'Hide the turns' : S.histOpen ? 'Show fewer turns' : 'Show all turns';
  for (const b of [$('hToggle'), $('cHist')]) b.setAttribute('aria-expanded', String(S.histOpen));
  $('hToggle').title = t;
  $('hToggle').setAttribute('aria-label', t);
  if (!quiet) refit();
}

function refit() {
  if (!S.ctrl) return;
  if (S.lane.layout()) S.lane.update(S.ctrl.frame);
  highlightRows();
  renderGraph();
}

// The layout in use: the one forced by ?layout or the About switcher, or else one chosen from the window, again on every
// resize or turn of a phone: a phone held sideways (wide and at most 520 px high) gets the rail, everything else the dock.
function chosenLayout() {
  if (S.layoutChoice !== 'auto') return S.layoutChoice;
  return innerWidth > innerHeight && innerHeight <= 520 ? 'rail' : 'dock';
}

// Move the shared zones into an arrangement. On a phone the rail scrolls inside and keeps its main button in a footer,
// and the dock takes the console's compact, full-width form with a status strip (no chips on such a small lane). Only
// the top bar (inside the rail, floating otherwise) and the turn buttons (in the bottom bar of the compact forms, under
// the history otherwise) change parents; the rest is CSS. Returns whether anything changed.
function arrange(force = false) {
  const lay = chosenLayout(), narrow = innerWidth < 700;
  const arr = lay === 'dock' && narrow ? 'console' : lay;
  const phone = arr === 'rail' ? innerHeight < 560 : arr === 'console' && narrow;
  const key = `${lay}:${arr}:${phone}`;
  if (key === S.arrKey && !force) return false;
  Object.assign(S, { arrKey: key, layout: lay, arr, phone });
  const B = document.body.classList;
  for (const l of LAYOUTS) B.toggle(`lay-${l}`, l === arr);
  B.toggle('phone', phone);
  B.toggle('rail-phone', phone && arr === 'rail');
  if (arr === 'rail') $('zScroll').prepend($('topbar'));
  else $('hud').prepend($('topbar'));
  if (arr === 'console') $('zBar').prepend($('nav'));
  else $('zHist').append($('nav'));
  $('graphBig').classList.toggle('pop', arr !== 'console');
  S.lane.labels = arr !== 'rail' && !phone;
  setHistory(false, true);
  setGraphBig(false, true);
  if (S.ctrl) renderControls();
  return true;
}

function applyLayout(name) {
  S.layoutChoice = LAYOUTS.includes(name) ? name : 'auto';
  try {
    const u = new URL(location.href);
    if (S.layoutChoice === 'auto') u.searchParams.delete('layout'); else u.searchParams.set('layout', S.layoutChoice);
    window.history.replaceState(null, '', u);
  } catch { /* the address cannot change here */ }
  arrange(true);
  if (S.ctrl) { S.lane.layout(true); renderAll(); }
}

// Light, dark, or the system's choice (auto). A choice made in About is remembered in this browser; ?theme= forces one.
function applyTheme(name, remember = false) {
  S.theme = THEMES.includes(name) ? name : 'auto';
  if (S.theme === 'auto') delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = S.theme;
  if (remember) { try { localStorage.setItem(THEME_KEY, S.theme); } catch { /* storage unavailable */ } }
  readTheme();
  if (S.ctrl) { S.lane.layout(true); renderHeads(); S.rows = -1; renderAll(); }
}

function graphNote() {
  $('gNote').textContent = 'How much more gold one side has earned so far. Not a chance of winning.';
}

// ------------------------------------------------------------------------------------------- bottom controls
// On a phone's rail the main button (Play turn, Continue, Return, or the playback row) and Stop move to a footer that
// stays in view while the rail scrolls.
function renderControls() {
  // the step panel takes the actions' place at their height, so nothing moves when a turn starts or ends
  const zm = document.querySelector('.z-main'), stepping = S.ctrl && S.ctrl.stepping;
  if (stepping && !S.mainH) S.mainH = zm.offsetHeight;
  if (!stepping) S.mainH = 0;
  zm.style.minHeight = stepping && S.mainH ? `${S.mainH}px` : '';
  renderActions();
  const foot = $('zFoot');
  foot.replaceChildren();
  if (!(S.phone && S.arr === 'rail')) return;
  const box = $('controls'), main = box.querySelector('.submit') || box.querySelector('.pb'), text = box.querySelector('.steptext');
  if (text) foot.append(text);                      // a stepped turn's words stay in view with its button
  if (main) foot.append(main);
  const skip = box.querySelector('#btnSkip');
  if (skip) foot.append(skip);
  const stop = box.querySelector('#btnStop');
  if (stop) foot.append(stop);
}

function renderActions() {
  const box = $('controls'), c = S.ctrl;
  box.className = 'controls';
  if (S.mode === 'play' && !S.live) {
    box.innerHTML = `<p class="note">${S.engine ? 'Starting a game…' : 'Play needs the local engine: run <code>prototypes/m6a/serve.py</code>, then reload. Watch works without it.'}</p>`;
    return;
  }
  if (c.stepping) return renderStep(box);
  if (S.mode === 'watch' || c.stopped) return renderPlayback(box);
  const m = c.menu(), t = c.progress + 1, review = c.historical;
  const err = S.error ? `<span class="err" role="alert">${esc(S.error)}</span>` : '';
  if (m.forced !== null) {
    // the status line above the lane already says why and for how long; here the choices are greyed and one button
    // plays the away turns, one engine turn at a time
    const dis = (label, cls = 'opt') => `<button class="${cls}" aria-disabled="true" tabindex="-1">${esc(label)}</button>`;
    box.innerHTML = `<div class="act">
      <div class="rows">
        <div class="r"><span class="lbl">Wave</span><div class="opts">${[0, 2, 3, 1].map((w) => dis(WAVE[w])).join('')}</div></div>
        <div class="r"><span class="lbl">Behaviour</span><div class="opts">${[0, 1, 2, 3].map((b) => dis(BEH[b])).join('')}</div></div>
        <div class="r"><span class="lbl">Objective</span><div class="opts one">${dis('Pressure')}</div></div>
      </div>
      <div class="side">
        <div class="singles">${dis('Recall', 'single')}${dis('Ward', 'single')}</div>
        ${review ? '<button class="btn primary submit" id="btnBack">Return to current turn</button>'
          : `<button class="btn primary submit" id="btnAway"${S.busy ? ' disabled' : ''}
          aria-label="Continue: your champion is away, so each away turn is played for you">Continue</button>`}
        <div class="foot"><span></span><button class="linkbtn" id="btnStop">Stop game</button></div>
      </div>
    </div>
    <div class="help">${err}</div>`;
    $('btnAway')?.addEventListener('click', continueAway);
    $('btnBack')?.addEventListener('click', () => S.ctrl.returnToCurrent());
    $('btnStop').addEventListener('click', stopGame);
    return;
  }
  const s = S.sel, R = {};
  const pr = m.objectives.find((o) => o.id === 1);
  if (s.pressure && !pr.legal) s.pressure = false;
  const btn = (key, label, on, reason, help, extra = '') => {
    R[key] = reason ? `${label} is unavailable. ${reason}` : `${label}. ${help}`;
    return `<button class="opt${on ? ' on' : ''}" data-k="${key}" aria-pressed="${on}" ${reason || review ? 'aria-disabled="true"' : ''}
      aria-label="${esc(R[key])}" ${extra}>${esc(label)}</button>`;
  };
  const waves = m.waves.map((w) => btn(`w${w.id}`, WAVE[w.id], s.kind === 'lane' && s.wave === w.id, null, w.help)).join('');
  const behs = m.behaviours.map((b) => btn(`b${b.id}`, BEH[b.id], s.kind === 'lane' && s.beh === b.id,
    s.pressure && ![0, 2].includes(b.id) ? 'Pressure combines with Hold or Trade only.' : null, b.help)).join('');
  const press = btn('p', 'Pressure', s.kind === 'lane' && s.pressure, pr.legal ? null : pr.reason, pr.help);
  const [rc, wd] = m.singles;
  const single = (x, kind) => `<button class="single${s.kind === kind ? ' on' : ''}" data-k="${kind}" aria-pressed="${s.kind === kind}"
      ${x.legal && !review ? '' : 'aria-disabled="true"'} aria-label="${esc(x.legal ? `${x.name}. ${x.help}` : `${x.name} is unavailable. ${x.reason}`)}">
      <span class="nm">${x.name}</span><span class="why">${esc(x.legal ? (kind === 'recall' ? 'Back to base' : 'In the bush') : x.reason)}</span></button>`;
  R.recall = rc.legal ? `Recall. ${rc.help}` : `Recall is unavailable. ${rc.reason}`;
  R.ward = wd.legal ? `Ward. ${wd.help}` : `Ward is unavailable. ${wd.reason}`;
  const code = selectionCode(m);
  const ok = code !== null && m.legal.includes(code) && !c.historical && !S.busy;
  const what = s.kind === 'recall' ? 'Recall' : s.kind === 'ward' ? 'Ward'
    : s.wave !== null && s.beh !== null ? `${WAVE[s.wave]}, ${BEH[s.beh]}${s.pressure ? ', Pressure' : ''}` : null;
  const off = [rc, wd].filter((x) => !x.legal).map((x) => esc(x.reason)).join(' ');
  // in the bush, Recall is not allowed: say how to leave rather than only why Recall is off
  const inBush = S.G.positions[c.progress]?.champions?.A?.at_ward;
  const bush = 'You are in the bush. Any wave action with a behaviour walks you back to the lane; '
    + 'choose Safe to step back to your side, then you can Recall next turn.';
  const base = review ? `You are looking at an earlier turn. Return to turn ${t} to choose.`
    : err || (inBush ? bush : off) || (what ? 'Point at any choice to see what it does.' : 'Choose a wave action and a behaviour, or Recall or Ward.');
  box.innerHTML = `<div class="act">
      <div class="rows">
        <div class="r"><span class="lbl">Wave</span><div class="opts">${waves}</div></div>
        <div class="r"><span class="lbl">Behaviour</span><div class="opts">${behs}</div></div>
        <div class="r"><span class="lbl">Objective</span><div class="opts one">${press}</div><span class="why">${esc(pr.legal ? 'Combines with Hold or Trade.' : pr.reason)}</span></div>
      </div>
      <div class="side">
        <div class="singles">${single(rc, 'recall')}${single(wd, 'ward')}</div>
        ${review ? '<button class="btn primary submit" id="btnBack">Return to current turn</button>'
          : `<button class="btn primary submit" id="btnSubmit" ${ok ? '' : 'disabled'}>${what ? `Play turn ${t}: ${esc(what)}` : `Play turn ${t}`}</button>`}
        <div class="foot"><span class="muted">The opponent has already chosen.</span><button class="linkbtn" id="btnStop">Stop game</button></div>
      </div>
    </div>
    <div class="help" id="help">${base}</div>`;
  const help = $('help');
  box.querySelectorAll('[data-k]').forEach((b) => {
    const k = b.dataset.k;
    const show = () => { help.textContent = R[k]; };
    const back = () => { help.innerHTML = base; };
    b.addEventListener('mouseenter', show); b.addEventListener('focus', show);
    b.addEventListener('mouseleave', back); b.addEventListener('blur', back);
    b.addEventListener('click', () => {
      if (review) return;
      if (b.getAttribute('aria-disabled') === 'true') { help.textContent = R[k]; return; }
      choose(k);
    });
  });
  $('btnSubmit')?.addEventListener('click', () => submit(selectionCode(c.menu())));
  $('btnBack')?.addEventListener('click', () => S.ctrl.returnToCurrent());
  $('btnStop').addEventListener('click', stopGame);
}

function selectionCode(m) {
  const s = S.sel;
  if (s.kind === 'recall') return 12;
  if (s.kind === 'ward') return 30;
  if (s.kind === 'lane' && s.wave !== null && s.beh !== null) return m.lane_codes[`${s.wave},${s.beh},${s.pressure ? 1 : 0}`] ?? null;
  return null;
}

function choose(k) {
  S.error = '';
  const s = S.sel.kind === 'lane' ? { ...S.sel } : empty();
  if (k === 'recall' || k === 'ward') S.sel = { ...empty(), kind: k };
  else {
    s.kind = 'lane';
    if (k[0] === 'w') s.wave = +k.slice(1);
    else if (k[0] === 'b') s.beh = +k.slice(1);
    else s.pressure = !s.pressure;
    if (s.pressure && s.beh !== null && ![0, 2].includes(s.beh)) s.beh = null;
    S.sel = s;
  }
  renderControls();
  $('controls').querySelector(`[data-k="${k}"]`)?.focus({ preventScroll: true });
}

// Watch: the selected turn's actions and outcomes in the controls area (outcomes only once the turn has ended)
function renderTurnCard(f) {
  const box = $('turnCard');
  if (!box) return;
  const v = S.ctrl.view;
  if (v === 0) { box.innerHTML = '<p class="muted">Start of the game. Both champions begin at their own tower.</p>'; return; }
  const tr = S.G.turns[v - 1];
  const acts = SEATS.map((n) => `<span>${glyph(n)}<b>${esc(S.short[n])}</b> ${esc(actionWords(tr.parts[SEATS.indexOf(n)]))}</span>`).join('');
  let list;
  if (f.sub === 'start' || f.sub === 'tick') list = '<li class="muted">Resolving: the waves and champions move; outcomes follow at the end of the turn.</li>';
  else {
    const major = tr.outcomes.filter((e) => e.major), minor = tr.outcomes.length - major.length;
    list = (major.length ? major.slice(0, 4).map((e) => `<li>${esc(fill(e.text))}</li>`).join('') : '<li class="muted">No trades, crashes, Recalls or deaths.</li>')
      + (major.length > 4 || minor ? `<li><button class="linkbtn" id="tcMore">${major.length > 4 ? `+${major.length - 4} more` : `${minor} smaller event${minor === 1 ? '' : 's'}`}</button></li>` : '');
  }
  box.innerHTML = `<div class="tc-acts"><b>Turn ${v}</b>${acts}<button class="linkbtn" id="tcTicks">Show the ticks</button></div><ul class="tc-list">${list}</ul>`;
  $('tcMore')?.addEventListener('click', () => openDetails('events'));
  $('tcTicks').addEventListener('click', () => S.ctrl.stepTurn(v));
}

// The frame of the turn being stepped: what it shows, Next tick (Space or →), Skip to the end, and the automatic mode.
function renderStep(box) {
  const c = S.ctrl, s = c.step, last = s.frames.length - 1, watching = S.mode === 'watch' || c.stopped;
  const W = frameWords(s.t, s.frames, s.i, watching ? 2 : 3);
  const label = s.i < last - 1 ? 'Next tick' : s.i === last - 1 ? 'Show the result' : watching ? 'Continue' : `Continue to turn ${s.t + 1}`;
  const focused = document.activeElement && (document.activeElement.id === 'btnTick' || document.activeElement === document.body);
  const dots = s.frames.map((_, j) => `<i class="${j < s.i ? 'done' : j === s.i ? 'now' : ''}"></i>`).join('');
  const out = W.outcomes ? `<ul class="st-out">${W.outcomes.map((o) => `<li>${esc(o)}</li>`).join('')}`
    + `${W.more ? `<li><button class="linkbtn" id="stMore">${W.more}</button></li>` : ''}</ul>` : '';
  box.className = 'controls' + (watching ? ' watch' : '');
  box.innerHTML = `${watching ? pbRow(c, false) : ''}<div class="act stepper">
      <div class="steptext" aria-live="polite">
        <div class="st-head"><b>${esc(W.head)}</b><span class="st-dots" aria-hidden="true">${dots}</span></div>
        <p class="st-desc">${esc(W.text)}</p>${out}
      </div>
      <div class="side">
        <label class="tick-auto"><input type="checkbox" id="tickAuto"${S.ticksAuto ? ' checked' : ''}> Play ticks automatically</label>
        <button class="btn primary submit" id="btnTick" title="${label} (Space or right arrow)">${esc(label)}</button>
        <div class="foot">${s.i < last ? '<button class="linkbtn" id="btnSkip">Skip to the end of the turn</button>' : '<span></span>'}</div>
      </div>
    </div>`;
  if (watching) wirePb(box, c);
  $('btnTick').addEventListener('click', () => c.advance());
  $('btnSkip')?.addEventListener('click', () => c.skip());
  $('tickAuto').addEventListener('change', (e) => setTicksAuto(e.target.checked, true));
  $('stMore')?.addEventListener('click', () => openDetails('events'));
  if (focused) $('btnTick').focus({ preventScroll: true });
}

function setTicksAuto(on, remember = false) {
  S.ticksAuto = !!on;
  if (remember) { try { localStorage.setItem(TICKS_KEY, S.ticksAuto ? 'auto' : 'manual'); } catch { /* storage unavailable */ } }
  if (S.ctrl && S.ctrl.stepping) S.ctrl.setAuto(S.ticksAuto);
  else if (S.ctrl) renderControls();
}

const pbRow = (c, primary = true) => `<div class="pb">
      <button class="btn${primary ? ' primary' : ''}" id="btnPlay">${c.playing ? 'Pause' : c.view >= c.progress ? 'Play from the start' : 'Play'}</button>
      <div class="seg speed" role="group" aria-label="Playback speed">${core.SPEEDS.map((x) => `<button aria-pressed="${x === c.speed}" data-s="${x}">${x}×</button>`).join('')}</div>
      <span class="muted pb-pos">${c.view === 0 ? 'Start' : 'Turn ' + c.view} of ${c.progress}</span>
    </div>`;
function wirePb(box, c) {
  $('btnPlay').addEventListener('click', () => { c.toggle(); });
  box.querySelectorAll('[data-s]').forEach((b) => b.addEventListener('click', () => { c.setSpeed(+b.dataset.s); setDur(); renderControls(); }));
}

function renderPlayback(box) {
  const c = S.ctrl;
  box.classList.add('watch');
  let extra = '';
  if (S.live && c.stopped) {
    const pos = S.G.positions[c.progress];
    const mk = S.G.meta.marks || {};
    const marks = [mk.lane_exit && `Gold lead at the lane exit (turn 94): ${sgn(mk.lane_exit.panel.D0)}.`,
                   mk.lead_114 && `After 114 turns: ${sgn(mk.lead_114.lead_A)} (this game only; M5 averaged it over many games).`].filter(Boolean);
    extra = `<div class="stopsum"><b>${c.progress >= c.n ? `The game reached the ${c.n}-turn limit.` : `You stopped the game after turn ${c.progress}.`}</b>
      Stopping declares no winner; the lane shows the state at the stop. Gold lead ${sgn(pos.lead_A)}.
      ${marks.join(' ')} <button class="linkbtn" id="btnNew2">New game</button></div>`;
  }
  box.innerHTML = `${pbRow(c)}${extra}${S.mode === 'watch' ? '<div class="turncard" id="turnCard" aria-live="polite"></div>' : ''}`;
  if (S.mode === 'watch') renderTurnCard(c.frame);
  wirePb(box, c);
  $('btnNew2')?.addEventListener('click', () => $('newGame').showModal());
}

// ------------------------------------------------------------------------------------------- details and about
let detailsOpen = false;
function openDetails(section) {
  detailsOpen = true;
  $('details').hidden = false;
  $('detailsBtn').setAttribute('aria-expanded', 'true');
  renderDetails(section);
}
function closeDetails() { detailsOpen = false; $('details').hidden = true; $('detailsBtn').setAttribute('aria-expanded', 'false'); }

function renderDetails(open) {
  if (!detailsOpen) return;
  const c = S.ctrl, v = shownView(), G = S.G, pos = G.positions[v], tr = v > 0 ? G.turns[v - 1] : null;
  const prev = {};
  $('dBody').querySelectorAll('details').forEach((d) => { prev[d.dataset.s] = d.open; });
  const sec = (key, title, html) => `<details data-s="${key}" ${(open ? open === key : prev[key] ?? key === 'events') ? 'open' : ''}><summary>${title}</summary>${html}</details>`;
  const events = tr ? `<ol class="ev">${tr.outcomes.map((e) => `<li class="${e.major ? '' : 'minor'}">${esc(fill(e.text))}</li>`).join('') || '<li>No events.</li>'}</ol>
      <p class="muted">In the resolver's order. Waves and fights move tick by tick; trades, tower shots, ganks, deaths, plates and Recalls are known for the whole turn.</p>`
    : '<p class="muted">The start: no turn played yet.</p>';
  const side = (n) => {
    const st = pos.champions[n], ln = pos.lane, front = ln.packets[n].find((q) => q[2]);
    return `<tr><th colspan="2">${glyph(n)} ${esc(S.names[n])}</th></tr>
      <tr><td>Front wave</td><td>${front ? `${front[1]} HP at cell ${front[0] > 0 ? '+' : ''}${front[0]}` : 'none'}</td></tr>
      <tr><td>Waves behind</td><td>${ln.packets[n].filter((q) => !q[2]).map((q) => `${q[1]} HP`).join(', ') || 'none'}</td></tr>
      <tr><td>Crashed under the enemy tower</td><td>${ln.siege[n][1]} HP</td></tr>
      <tr><td>Wave damage per turn</td><td>${st.capacity ?? '–'} HP</td></tr>
      <tr><td>Budgets for the next turn</td><td>${Object.entries(st.wave_budgets || {}).map(([k, x]) => `${esc(k)} ${x}`).join(', ')}</td></tr>
      ${tr ? `<tr><td>Enemy HP removed, tick 1 + 2</td><td>${tr.ticks.map((k) => k.wave_removed[n]).join(' + ')}</td></tr>` : ''}`;
  };
  let ticks = '';
  if (tr) {
    const pk = (l) => l.map((q) => `${q[1]}@${q[0] > 0 ? '+' : ''}${q[0]}`).join(' ') || '–';
    ticks = `<p class="muted">Waves are listed as HP@cell (global cells, A's tower at −2). Cells are where each champion stands after the frame.</p>
      <table class="dt"><thead><tr><th>Frame</th><th>A waves</th><th>B waves</th><th>Cells A / B</th><th>Minion hits A / B</th></tr></thead><tbody>
      ${core.turnSubframes(G, v).map((f) => `<tr><td>${f.sub === 'tick' ? 'tick ' + f.tick.tick : f.sub}</td><td>${pk(f.lane.packets.A)}</td><td>${pk(f.lane.packets.B)}</td>
      <td>${f.cells.A ?? (f.at_ward.A ? 'W' : '–')} / ${f.cells.B ?? (f.at_ward.B ? 'W' : '–')}</td>
      <td>${f.tick ? `${f.tick.minion_fire.A.units} / ${f.tick.minion_fire.B.units}` : ''}</td></tr>`).join('')}</tbody></table>`;
  }
  const champ = (n) => {
    const st = pos.champions[n];
    return `<tr><th colspan="2">${glyph(n)} ${esc(S.names[n])}</th></tr>
      <tr><td>Level, XP</td><td>${st.level}, ${st.xp}${st.xp_next ? ` (next level at ${st.xp_next})` : ''}</td></tr>
      <tr><td>Gold earned</td><td>${st.gold_earned.toFixed(2)}</td></tr>
      <tr><td>Unspent, invested</td><td>${st.gold_unspent.toFixed(2)}, ${st.invested.toFixed(2)}</td></tr>
      <tr><td>Attack power</td><td>${st.attack_power.toFixed(3)}</td></tr>
      <tr><td>Ward vision</td><td>${st.vision ? turns(st.vision) : 'none'}</td></tr>`;
  };
  const bl = c.belief(v);
  const GUESS = { g_farm: 'Farming bot', g_safe: 'Cautious bot', g_bully: 'Aggressive bot', g_shove: 'Always-push bot',
                  fz_guard_sh: 'Freezing bot', g_shove_ward: 'Push-and-ward bot', peer: 'A trained agent like itself',
                  attacker: 'An agent trained to attack', freezer: 'An agent that holds the wave', gapper: 'An agent that presses big leads',
                  short: 'A short-sighted trained agent', mixed: 'A mix of the bots', unknown: 'None of these' };
  const NOTE = 'The adaptive agent compares how its opponent plays with 13 kinds of player: 6 simple bots, 5 trained '
    + 'agents, a mix of the bots, and "none of these". None of them is a person. When its guess is shown you can try to '
    + 'mislead it; the research never tested that.';
  let belief;
  if (!G.beliefs && !(S.live && S.live.belief === 'hidden' && S.G.meta.seats[1].arm === 'om')) belief = '<p class="muted">Only the adaptive agent keeps a guess about its opponent, and it is not playing here.</p>';
  else if (!bl) belief = `<p class="muted">Hidden in this game, as chosen when it started. It appears when the game ends.</p><p class="muted">${esc(NOTE)}</p>`;
  else belief = Object.entries(bl.bySeat).map(([s, post]) => `<p>The adaptive agent's guess about ${S.names[other(s)] === 'You' ? 'your style' : esc(S.names[other(s)].toLowerCase())} ${v === 0 ? 'at the start' : 'after turn ' + v}:</p>
      <div class="belief">${bl.types.map((t, i) => [t, post[i]]).sort((x, y) => y[1] - x[1]).map(([t, p]) =>
        `<span>${esc(GUESS[t] || t)}</span><span class="bb"><i style="width:${(100 * p).toFixed(1)}%"></i></span><span class="num">${(100 * p).toFixed(1)}%</span>`).join('')}</div>`).join('')
      + `<p class="muted">${esc(NOTE)}</p>`;
  const m = G.meta, sd = m.seeds || {}, mk = m.marks || {};
  const seat = (x) => `<tr><td>${x.seat}</td><td>${x.kind === 'human' ? 'You' : x.kind === 'model'
      ? `${esc(who(x).name)}<br><span class="muted">research id ${esc(x.id)}: M5's ${x.arm === 'om' ? 'opponent-modelling' : 'pure self-play'} agent, seed ${x.seed}</span><br><span class="mono">file SHA-256 ${esc(x.file_sha256.slice(0, 16))}…</span>`
      : `${esc(who(x).name)}<br><span class="muted">rule (${esc(x.id)}): ${esc(x.rule)}</span>`}</td></tr>`;
  const record = `<table class="dt"><tbody>${m.seats.map(seat).join('')}
      <tr><td>Record type</td><td>${esc($('recType').textContent)}</td></tr>
      <tr><td>Chance seed</td><td class="mono">${sd.chance ?? 'revealed when the game stops'}</td></tr>
      <tr><td>Policy seed</td><td class="mono">${sd.policy ?? 'revealed when the game stops'}</td></tr>
      ${mk.lane_exit ? `<tr><td>Gold lead at the lane exit (turn 94)</td><td>${sgn(mk.lane_exit.panel.D0)}</td></tr>` : ''}
      ${mk.lead_114 ? `<tr><td>Gold lead after 114 turns</td><td>${sgn(mk.lead_114.lead_A)}</td></tr>` : ''}</tbody></table>`;
  $('dTitle').textContent = v === 0 ? 'Details: start' : `Details: turn ${v}`;
  $('dBody').innerHTML = sec('events', 'Turn events', events)
    + sec('waves', 'Waves and ticks', `<table class="dt"><tbody>${side('A')}${side('B')}</tbody></table>${ticks}`)
    + sec('champs', 'Champions', `<table class="dt"><tbody>${champ('A')}${champ('B')}</tbody></table>`)
    + sec('belief', 'Belief', belief) + sec('record', 'Game record', record);
}

function openAbout() {
  const G = S.G, m = G.meta, kind = m.provenance.kind;
  const intro = { research_record: 'A game recorded in the research and checked by replaying it exactly.',
                  new_simulation: 'A game the engine just played between two trained agents.',
                  human_game: 'Your game against a trained agent. Every action you choose is the one the engine plays.' }[kind];
  const title = GAME_NAMES[G.name] || G.label || '';
  $('aboutBody').innerHTML = `<h2 id="aboutTitle">About this game</h2>
    <p>${title ? `<b>${esc(title)}.</b> ` : ''}${intro} A single game shows how the agents play. It is an example, not a measurement.</p>
    <h3>The players</h3>
    <ul><li><b>Self-play agent</b>: ${AGENTS['sp:0'].about}.</li>
    <li><b>Adaptive agent</b>: ${AGENTS['om:0'].about}. Half of its training games were against fixed play styles.</li>
    <li><b>Always-push bot</b>: ${BOTS.g_shove.about}. It is not trained.</li></ul>
    <p class="muted">Both agents are the final models of the research's last training run, and both pick their actions with
    some randomness, as they did in the research.</p>
    <h3>What the research found</h3>
    <ul><li>Against simple bots like the ones it trained against, the adaptive agent ends the lane far ahead: about 9 gold
    more than the self-play agent manages.</li>
    <li>Against strong opponents, including the self-play agent, it does slightly worse: about 0.4 to 0.6 gold.</li>
    <li>Opponents trained specially to beat them can beat both.</li></ul>
    <p>Neither agent is better overall, and neither was ever tested against people. The full numbers are in the research
    report (docs/m5/m5e_closeout_report.md).</p>
    <h3>Layout</h3>
    <div class="lay-pick"><div class="seg" role="group" aria-label="Layout">${['auto', ...LAYOUTS].map((l) =>
      `<button data-lay="${l}" aria-pressed="${l === S.layoutChoice}">${l[0].toUpperCase() + l.slice(1)}</button>`).join('')}</div>
      <span class="muted">Auto uses the rail on a phone held sideways and the dock elsewhere. The game carries on when you switch.</span></div>
    <h3>Theme</h3>
    <div class="lay-pick"><div class="seg" role="group" aria-label="Theme">${THEMES.map((t) =>
      `<button data-theme-pick="${t}" aria-pressed="${t === S.theme}">${t[0].toUpperCase() + t.slice(1)}</button>`).join('')}</div>
      <span class="muted">Auto follows your system setting.</span></div>
    <h3>Ticks</h3>
    <div class="lay-pick"><label class="tick-auto"><input type="checkbox" id="aboutTicks"${S.ticksAuto ? ' checked' : ''}> Play ticks automatically</label>
      <span class="muted">Off: after each turn you step through its two ticks yourself (Next tick). Watch's Play and
      away turns always play them on their own.</span></div>
    <form method="dialog"><button class="btn" autofocus>Close</button></form>`;
  $('aboutTicks').addEventListener('change', (e) => setTicksAuto(e.target.checked, true));
  $('aboutBody').querySelectorAll('[data-lay]').forEach((b) => b.addEventListener('click', () => { $('about').close(); applyLayout(b.dataset.lay); }));
  $('aboutBody').querySelectorAll('[data-theme-pick]').forEach((b) => b.addEventListener('click', () => { $('about').close(); applyTheme(b.dataset.themePick, true); }));
  $('about').showModal();
}

// ------------------------------------------------------------------------------------------- actions
// The engine resolves the turn; the page then shows it (stepped, or on its own) and only after that offers the next
// decision. `appended` is told as soon as the resolved turn is on the client.
async function submit(code, appended) {
  const c = S.ctrl;
  if (!(c instanceof Live) || !c.canSubmit || S.busy || code === null || code === undefined) { appended?.(); return; }
  S.busy = true;
  S.error = '';
  renderControls();
  try {
    const res = await api(`/api/sessions/${S.live.sid}/turns`, { turn: c.progress + 1, code });
    S.sel = empty();
    const anim = c.append(res);
    renderAll();
    appended?.();
    await anim;
    if (c.stopped) await finish();
  } catch (e) {
    S.error = e.message;
  } finally {
    S.busy = false;
    appended?.();
    renderAll();
  }
}

async function continueAway() {                     // away turns play their ticks on their own
  S.forcedRun = true;
  try { while (S.ctrl instanceof Live && S.ctrl.forced() && !S.error) await submit(45); } finally { S.forcedRun = false; }
}

async function finish() {
  const d = await api(`/api/sessions/${S.live.sid}/stop`, {});
  const G = S.G;
  G.beliefs = d.recording.beliefs || null;
  G.meta.seeds = d.recording.seeds;
  G.meta.marks = d.view.marks;
  S.ctrl.stopped = true;
  S.ctrl.menuNow = null;
  S.rows = Infinity;                                    // rebuild the history without the pending row
}

async function stopGame() {
  if (!(S.ctrl instanceof Live) || S.busy) return;
  S.busy = true;
  try { await finish(); } catch (e) { S.error = e.message; } finally { S.busy = false; renderAll(); }
}

async function newGame(opp = 'sp:0', belief = 'hidden', seed = null) {
  S.ctrl?.pause();
  const d = await api('/api/sessions', { opponent: opp, belief, ...(seed !== null ? { seed } : {}) });
  const v = d.view;
  const G = { name: 'live', label: '', positions: d.history.positions, turns: [],
              meta: { seats: v.seats, provenance: { kind: 'human_game' }, marks: v.marks, seeds: v.seeds },
              beliefs: v.belief ? { types: v.belief.B.types, posteriors: { B: [v.belief.B.posterior] } } : null };
  S.mode = 'play';
  S.live = { sid: d.session, opp, belief, cap: v.turn_cap };
  const c = new Live(G, v.turn_cap, { belief, speed: S.ctrl ? S.ctrl.speed : P.speed, still: P.still });
  c.menuNow = v.menu;
  install(G, c);
}

async function watch(name, view = 0) {
  S.ctrl?.pause();
  const G = await core.loadGame(name);
  S.mode = 'watch';
  S.live = null;
  install(G, new Stepper(G, { mode: 'watch', view, speed: S.ctrl ? S.ctrl.speed : P.speed, still: P.still }));
}

async function simulate(a = 'om:0', b = 'sp:0') {
  S.ctrl?.pause();
  $('controls').innerHTML = '<p class="note">The engine is playing 150 turns…</p>';
  $('zFoot').replaceChildren();
  const G = await api('/api/simulations', { a, b });
  G.label = `New game: ${AGENTS[a].name} vs. ${AGENTS[b].name.toLowerCase()}`;
  S.mode = 'watch';
  S.live = null;
  install(G, new Stepper(G, { mode: 'watch', view: 0, speed: S.ctrl ? S.ctrl.speed : P.speed, still: P.still }));
}

// ------------------------------------------------------------------------------------------- wiring
function setDur() { document.documentElement.style.setProperty('--dur', `${Math.round((core.BASE_FRAME_MS * 0.9) / S.ctrl.speed)}ms`); }

function install(G, ctrl) {
  S.G = G;
  S.ctrl = ctrl;
  S.sel = empty();
  S.error = '';
  S.hover = null;
  S.rows = -1;
  $('hList').innerHTML = '';
  setNames();
  renderHeads();
  graphNote();
  renderTop();                                          // the top bar's width is part of what the lane must avoid
  S.lane.layout(true);
  ctrl.addEventListener('change', () => { if (S.ctrl === ctrl) onChange(); });
  setDur();
  renderAll();
}

function renderHeads() {
  $('hA').innerHTML = `${glyph('A')}<span title="${esc(S.names.A)}">${esc(S.head.A)}</span>`;
  $('hB').innerHTML = `${glyph('B')}<span title="${esc(S.names.B)}">${esc(S.head.B)}</span>`;
}

function renderAll() {
  const f = S.ctrl.frame;
  S.lane.update(f);
  renderStatus(f);
  renderHistory();
  highlightRows();
  renderSummary(f);
  renderBanner();
  renderControls();
  renderNav();
  renderGraph();
  renderDetails();
  $('laneText').textContent = laneText(f);
  S.last = { view: S.ctrl.view, progress: S.ctrl.progress, stopped: S.ctrl.stopped, playing: S.ctrl.playing, step: stepKey() };
}

const stepKey = () => { const s = S.ctrl.step; return s ? `${s.t}:${s.i}:${+s.auto}` : ''; };

function onChange() {
  const c = S.ctrl, f = c.frame, L = S.last;
  S.lane.update(f);
  renderStatus(f);
  renderSummary(f);
  if (c.progress !== L.progress || c.stopped !== L.stopped) renderHistory();
  const sk = stepKey();
  if (c.view !== L.view || c.progress !== L.progress || c.stopped !== L.stopped || c.playing !== L.playing || sk !== L.step) {
    highlightRows(); renderBanner(); renderControls(); renderNav(); renderGraph();
  }
  if (f.sub === 'end' || f.sub === 'position' || sk !== L.step) { renderDetails(); $('laneText').textContent = laneText(f); }
  S.last = { view: c.view, progress: c.progress, stopped: c.stopped, playing: c.playing, step: sk };
}

function laneText(f) {
  const out = [$('turnLabel').textContent + '.'];
  for (const n of SEATS) {
    const st = f.status[n], front = f.lane.packets[n].find((q) => q[2]);
    const at = f.at_ward[n] ? 'in the bush at the ward spot (the bush hides no one here)' : `cell ${f.cells[n]}`;
    out.push(`${S.names[n]}: level ${st.level}, ${st.status === 'present' ? `${st.hp} of ${st.hp_max} HP, ${at}` : st.status}; wave ${front ? `${front[1]} HP at cell ${front[0]}` : 'none'}.`);
  }
  return out.join(' ');
}

function bindKeys() {
  addEventListener('keydown', (e) => {
    if (e.target.closest('input, select, textarea, dialog[open]') || e.altKey || e.ctrlKey || e.metaKey) return;
    const c = S.ctrl;
    if (!c) return;
    const act = {
      ArrowLeft: () => c.prev(), ArrowRight: () => c.next(), Home: () => c.first(), End: () => c.latest(),
      Escape: () => (detailsOpen ? closeDetails() : S.graphBig ? setGraphBig(false) : S.histOpen ? setHistory(false) : c.returnToCurrent()),
      ' ': () => (c.stepping && !c.playing ? c.advance() : S.live && !c.stopped ? null : c.toggle()),
      Enter: () => (S.live && c.canSubmit && !e.target.closest('button') ? submit(selectionCode(c.menu())) : null),
    }[e.key];
    if (act && !(e.key === ' ' && e.target.closest('button'))) { if (act() !== null) e.preventDefault(); }
  });
}

async function main() {
  S.lane = new Lane($('lane'));
  let stored = null;
  try { stored = localStorage.getItem(THEME_KEY); } catch { /* storage unavailable */ }
  applyTheme(THEMES.includes(Q.get('theme')) ? Q.get('theme') : stored);
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { if (S.theme === 'auto') applyTheme('auto'); });
  S.layoutChoice = LAYOUTS.includes(Q.get('layout')) ? Q.get('layout') : 'auto';
  let ticks = null;
  try { ticks = localStorage.getItem(TICKS_KEY); } catch { /* storage unavailable */ }
  S.ticksAuto = (['auto', 'manual'].includes(Q.get('ticks')) ? Q.get('ticks') : ticks) === 'auto';
  arrange(true);
  bindKeys();
  $('modePlay').addEventListener('click', () => { if (S.mode !== 'play') (S.engine ? newGame(Q.get('opp') || 'sp:0', 'hidden') : playUnavailable()); });
  $('modeWatch').addEventListener('click', () => { if (S.mode !== 'watch') watch(S.index.games[0].name); });
  $('aboutBtn').addEventListener('click', openAbout);
  $('detailsBtn').addEventListener('click', () => (detailsOpen ? closeDetails() : openDetails()));
  $('dClose').addEventListener('click', closeDetails);
  $('btnFirst').addEventListener('click', () => S.ctrl.first());
  $('btnPrev').addEventListener('click', () => S.ctrl.prev());
  $('btnNext').addEventListener('click', () => S.ctrl.next());
  $('btnLast').addEventListener('click', () => S.ctrl.latest());
  $('btnReturn').addEventListener('click', () => S.ctrl.returnToCurrent());
  $('gEvents').addEventListener('change', (e) => { S.events = e.target.checked; renderGraph(); });
  $('gExpand').addEventListener('click', () => setGraphBig(!S.graphBig));
  $('gCollapse').addEventListener('click', () => setGraphBig(false));
  $('hToggle').addEventListener('click', () => setHistory(!S.histOpen));
  $('cHist').addEventListener('click', () => setHistory(!S.histOpen));
  setHistory(Q.get('history') === 'open', true);
  for (const gp of [$('graph'), $('graphB')]) {
    gp.addEventListener('pointermove', (e) => { const p = graphTurn(e); if (p !== S.hover) { S.hover = p; renderGraph(); highlightRows(); } });
    gp.addEventListener('pointerleave', () => { S.hover = null; renderGraph(); highlightRows(); });
    gp.addEventListener('click', (e) => { const p = graphTurn(e); if (p !== null) S.ctrl.goto(p); });
  }
  $('ngForm').addEventListener('submit', (e) => {
    if (e.submitter && e.submitter.value === 'ok') {
      const f = new FormData(e.target);
      newGame(f.get('opp'), f.get('opp') === 'om:0' ? f.get('belief') : 'hidden').catch((x) => { S.error = x.message; renderControls(); });
    }
  });
  $('nsForm').addEventListener('submit', (e) => {
    if (e.submitter && e.submitter.value === 'ok') { const f = new FormData(e.target); simulate(f.get('a'), f.get('b')); }
  });
  let rz;
  addEventListener('resize', () => {
    clearTimeout(rz);
    rz = setTimeout(() => { if (!S.ctrl) return; arrange(); S.lane.layout(true); renderAll(); }, 120);
  });
  if (core.prefersReducedMotion() || P.still) document.body.classList.add('reduced');

  S.index = await core.loadIndex();
  try { const st = await api('/api/status'); S.engine = st.ready ? st : null; } catch { S.engine = null; }
  window.__demo = {
    newGame: (o = 'sp:0', b = 'hidden', seed = null) => newGame(o, b, seed),
    submit: (code, wait = true) => (wait ? submit(code) : new Promise((r) => { submit(code, r); })),
    step: () => { S.ctrl.advance(); return Promise.resolve(); }, skip: () => { S.ctrl.skip(); return Promise.resolve(); },
    quick: async (code) => {                         // play a turn and finish showing it at once (tests and screenshots)
      await new Promise((r) => { submit(code, r); });
      if (S.ctrl.step) { S.ctrl.skip(); S.ctrl.advance(); }
      while (S.busy) await new Promise((r) => setTimeout(r, 20));
    },
    stepTurn: (t) => { S.ctrl.stepTurn(t); return Promise.resolve(); }, ticks: (m) => { setTicksAuto(m === 'auto'); return Promise.resolve(); },
    select: (sel) => { S.sel = { ...empty(), ...sel }; renderControls(); }, continueAway, stop: stopGame,
    goto: (p) => { S.ctrl.goto(p); return Promise.resolve(); }, watch, simulate,
    hover: (p) => { S.hover = p; renderGraph(); highlightRows(); }, details: (s) => openDetails(s),
    history: (open) => { setHistory(open); return Promise.resolve(); }, graph: (big) => { setGraphBig(big); return Promise.resolve(); },
    layout: (name) => { applyLayout(name); return Promise.resolve(); },
    theme: (name) => { applyTheme(name); return Promise.resolve(); },
    state: () => ({
      layout: S.layout, arrangement: S.arr, phone: S.phone, layoutChoice: S.layoutChoice, theme: S.theme,
      dark: S.theme === 'dark' || (S.theme === 'auto' && matchMedia('(prefers-color-scheme: dark)').matches), lane: { scale: +S.lane.a.toFixed(3), angle: Math.round((S.lane.t * 180) / Math.PI) },
      mode: S.mode, progress: S.ctrl.progress, view: S.ctrl.view, stopped: S.ctrl.stopped, live: !!S.live, sid: S.live && S.live.sid,
      turns: S.G.turns.map((t) => t.codes), menu: S.ctrl.menu ? S.ctrl.menu() : null, error: S.error, busy: S.busy,
      canSubmit: S.ctrl.canSubmit, ticksAuto: S.ticksAuto, playing: S.ctrl.playing,
      step: S.ctrl.step ? { t: S.ctrl.step.t, i: S.ctrl.step.i, n: S.ctrl.step.frames.length, sub: S.ctrl.frame.sub, auto: S.ctrl.step.auto,
                            head: document.querySelector('.st-head b')?.textContent, text: document.querySelector('.st-desc')?.textContent } : null }),
  };
  const mode = P.mode === 'watch' ? 'watch' : 'play';
  if (mode === 'watch') await watch(P.game, P.view ?? 0);
  else if (S.engine) await newGame(Q.get('opp') || 'sp:0', Q.get('belief') || 'hidden', Q.has('seed') ? +Q.get('seed') : null);
  else await playUnavailable();
  if (P.detail) openDetails(P.detail);
  if (Q.get('graph') === 'expanded') setGraphBig(true);
  if (S.histOpen) refit();
  document.body.dataset.ready = '1';
}

async function playUnavailable() {
  const G = await core.loadGame('h2h_om_s0_vs_sp_s0');
  const G0 = { ...G, name: 'unavailable', positions: [G.positions[0]], turns: [], beliefs: null };
  S.mode = 'play';
  S.live = null;
  install(G0, new Stepper(G0, { mode: 'watch', view: 0, still: P.still }));
}

main();
