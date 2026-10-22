// The scene: the simulator's topology drawn from above, on a diagonal, as the approved M6a design draws it. Nine
// cells on one road from A's base at the lower left (cell -4) to B's base at the upper right (+4), A's tower at -2,
// the lane 0, B's tower at +2, and the ward spot W (a bush) beside cell 0 on the upper-left side. One mapping places
// everything: P(u, v) = centre + u (cos t, -sin t) + v (sin t, cos t), u along the road (cells x spacing), v across
// it (negative: the upper-left side). A's minions walk the upper-left half and B's the lower-right half; A stands on
// the upper-left edge, B on the lower-right edge. Minions are balls, one per HP point: the engine tracks each wave's HP,
// not single minions. Nothing else is drawn: no terrain, no river, no paths the simulator does not have.
//
// The drawing is fitted around whatever the glass surface covers (the top bar, the surface, an open sheet), at the
// largest scale at which every box of the drawing finds a place, and refitted only on a resize or when a section
// opens or closes, never as turns advance. This file computes no game rule: it draws the frames the server sent.
// Imperative SVG, mounted by LaneView.tsx; colours come from the theme's CSS variables.
import type { Frame } from '../game/frames';
import type { Names, Seat, Turn } from '../game/types';
import { SEATS, other } from '../game/types';
import { BUSH_NOTE, turnsWord } from '../game/words';

const NS = 'http://www.w3.org/2000/svg';
const COL: Record<Seat, string> = { A: 'var(--a)', B: 'var(--b)' };
const SOFT: Record<Seat, string> = { A: 'var(--a-soft)', B: 'var(--b-soft)' };
const CHIP = { w: 108, h: 34 };                               // the largest chip, for fitting the drawing
const STEEP = (35 * Math.PI) / 180;                           // above this angle B's base label goes above the base
const tokHalf = (n: Seat, r: number) => (n === 'B' ? r * 1.17 : r) + 1;

export interface Box { x0: number; y0: number; x1: number; y1: number }
export interface LaneContext { names: Names; head: Names; live: boolean; turns: Turn[] }

interface Sizes {
  a: number; sp: number; b: number; hr: number; rt: number; tok: number; wr: number; we: number; pw: number; ph: number;
  maxCols: number; maxRows: number; R: number; D: number; halfH: number; half: number; tw: number; wv: number;
}
interface Extents extends Box { boxes: Box[] }
interface Cluster { g: SVGGElement; list: SVGCircleElement[]; label: SVGTextElement; title: SVGTitleElement; w: number; h: number }
interface Chip {
  g: SVGGElement; bg: SVGRectElement; nm: SVGTSpanElement; lv: SVGTSpanElement; bar0: SVGRectElement; bar: SVGRectElement;
  t2: SVGTextElement; hp: SVGTSpanElement; fx: SVGTSpanElement; aw: SVGTextElement; t1: SVGTextElement; title: SVGTitleElement; placed: boolean;
}
interface Pin { g: SVGGElement; head: SVGElement; title: SVGTitleElement }
type Pt = { x: number; y: number };

function svg<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number | null | undefined> = {}, parent: Element | null = null): SVGElementTagNameMap[K] {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined) e.setAttribute(k, String(v));
  if (parent) parent.appendChild(e);
  return e;
}

/** What the surface covers: the floating top bar, the surface itself and any sheet open out of it, each with a
 * margin. These change only on a resize or when a section is opened or closed. */
export function obstacles(ids: string[] = ['topbar', 'surface', 'zHist', 'graphBig']): Box[] {
  const m = 12;
  return ids.map((id) => document.getElementById(id)).filter((e): e is HTMLElement => !!e && !e.hidden && e.offsetParent !== null)
    .map((e) => e.getBoundingClientRect()).filter((r) => r.width > 0 && r.height > 0)
    .map((r) => ({ x0: r.left - m, y0: r.top - m, x1: r.right + m, y1: r.bottom + m }));
}

/** The largest rectangle of the window left after cutting each obstacle away on the side that keeps the most area. */
function freeRect(obs: Box[]): Box {
  const e = 14, area = (r: Box) => Math.max(0, r.x1 - r.x0) * Math.max(0, r.y1 - r.y0);
  let r: Box = { x0: e, y0: e, x1: innerWidth - e, y1: innerHeight - e };
  for (const o of [...obs].sort((p, q) => area(q) - area(p))) {
    if (o.x1 <= r.x0 || o.x0 >= r.x1 || o.y1 <= r.y0 || o.y0 >= r.y1) continue;
    r = [{ ...r, x1: o.x0 }, { ...r, x0: o.x1 }, { ...r, y1: o.y0 }, { ...r, y0: o.y1 }].reduce((p, q) => (area(q) > area(p) ? q : p));
  }
  return r;
}

const hits = (a: Box, b: Box, dx = 0, dy = 0) => a.x0 + dx < b.x1 && a.x1 + dx > b.x0 && a.y0 + dy < b.y1 && a.y1 + dy > b.y0;
const overlap = (a: Box, b: Box) => Math.max(0, Math.min(a.x1, b.x1) - Math.max(a.x0, b.x0)) * Math.max(0, Math.min(a.y1, b.y1) - Math.max(a.y0, b.y0));

/** A centre for a drawing whose boxes `ex` (relative to its centre) all stay inside `B` and clear of every obstacle,
 * the candidate nearest the centre of `pref` first; null when there is none. */
function placeDrawing(ex: Extents, B: Box, obs: Box[], pref: Box): Pt | null {
  const lo = { x: B.x0 - ex.x0, y: B.y0 - ex.y0 }, hi = { x: B.x1 - ex.x1, y: B.y1 - ex.y1 };
  if (lo.x > hi.x || lo.y > hi.y) return null;
  const cl = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));
  const px = cl((pref.x0 + pref.x1) / 2 - (ex.x0 + ex.x1) / 2, lo.x, hi.x), py = cl((pref.y0 + pref.y1) / 2 - (ex.y0 + ex.y1) / 2, lo.y, hi.y);
  const step = Math.max(6, Math.min(20, Math.max(hi.x - lo.x, hi.y - lo.y) / 50));
  const axis = (a: number, b: number) => { const out: number[] = []; for (let v = a; v < b; v += step) out.push(v); out.push(b); return out; };
  const cands: [number, number][] = [[px, py]];
  for (const x of axis(lo.x, hi.x)) for (const y of axis(lo.y, hi.y)) cands.push([x, y]);
  cands.sort((p, q) => (p[0] - px) ** 2 + (p[1] - py) ** 2 - (q[0] - px) ** 2 - (q[1] - py) ** 2);
  for (const [x, y] of cands) if (!ex.boxes.some((b) => obs.some((o) => hits(b, o, x, y)))) return { x, y };
  return null;
}

export class Lane {
  el: SVGSVGElement;
  ctx: () => LaneContext;
  labels = true;
  packets = new Map<string, Cluster>();
  private fxKey = '';
  private key = '';
  W = 0; H = 0; a = 1; sp = 0; k = 1; t = 0; ca = 1; sa = 0; cx = 0; cy = 0;
  hr = 0; rt = 0; tok = 0; wr = 0; we = 0; pw = 0; ph = 0; maxCols = 8; maxRows = 3; R = 5; D = 12; halfH = 0; half = 0; tw = 0; wv = 0;
  safe: Box = { x0: 0, y0: 0, x1: 0, y1: 0 };
  bounds: Box = { x0: 0, y0: 0, x1: 0, y1: 0 };
  legend = { x: 0, y: 0, text: '' };
  block: Box[] = [];
  fixed: Box[] = [];
  chipBoxes: Partial<Record<Seat, Box>> = {};
  private towerG: Partial<Record<Seat, { g: SVGGElement; title: SVGTitleElement }>> = {};
  private chips: Partial<Record<Seat, Chip>> = {};
  private pins: Partial<Record<Seat, Pin>> = {};
  private gPlates!: SVGGElement; private gSiege!: SVGGElement; private gPk!: SVGGElement; private gContact!: SVGGElement;
  private gChips!: SVGGElement; private gPins!: SVGGElement; private gFx!: SVGGElement;

  constructor(el: SVGSVGElement, ctx: () => LaneContext) { this.el = el; this.ctx = ctx; }

  /** Every length of the drawing at scale a (the road grows faster than the cells, capped at 2.1; a wide road takes
   * clusters of at most 6 across and 4 rows; the balls fit a cell). */
  sizes(a: number): Sizes {
    const sp = 115 * a, b = Math.min(2.1, 1.7 * a);
    const hr = Math.round(44 * b), rt = Math.round(17 * b), tok = Math.round(12.5 * b), wr = Math.round(15 * b);
    const maxCols = hr >= 75 ? 6 : 8, maxRows = maxCols === 6 ? 4 : 3;
    const R = Math.max(3, Math.min(7.2, (sp * 0.85) / (maxCols * 2.45), (hr - 6) / (maxRows * 2.45)));
    const halfH = hr + Math.round(10 * b), half = Math.min(halfH, sp * 0.46);
    const we = Math.max(7, Math.round(wr * 0.72)), ph = 2 * Math.max(tok + 7, we + 11), pw = 2 * (we + 2 * tok + 12);
    return { a, sp, b, hr, rt, tok, wr, we, pw, ph, maxCols, maxRows, R, D: R * 2.45, halfH, half,
      tw: hr + Math.round(tok * 1.2) + 8 + rt, wv: hr + Math.round(ph / 2 + 40 * b) };
  }

  /** Every screen box the scene may draw into, relative to the drawing's centre. */
  extents(g: Sizes, t: number): Extents {
    const ca = Math.cos(t), sa = Math.sin(t), boxes: Box[] = [];
    const P = (u: number, v: number): [number, number] => [u * ca + v * sa, -u * sa + v * ca];
    const box = ([x, y]: [number, number], w: number, h: number, dy = 0, dx = 0) => boxes.push({ x0: x + dx - w / 2, x1: x + dx + w / 2, y0: y + dy - h / 2, y1: y + dy + h / 2 });
    const baseDown = g.half * sa + g.halfH * ca;
    for (const s of [-1, 1]) {
      for (let i = 0; i < 4; i++) {                                              // the rotated base, in four slices
        const v0 = -g.halfH + (i * g.halfH) / 2, v1 = v0 + g.halfH / 2;
        const cs = [[-g.half, v0], [g.half, v0], [-g.half, v1], [g.half, v1]].map(([uu, vv]) => P(s * 4 * g.sp + uu, vv));
        boxes.push({ x0: Math.min(...cs.map((q) => q[0])), x1: Math.max(...cs.map((q) => q[0])), y0: Math.min(...cs.map((q) => q[1])), y1: Math.max(...cs.map((q) => q[1])) });
      }
      box(P(s * 4 * g.sp, 0), 100, 16, s > 0 && t > STEEP ? -baseDown - 12 : baseDown + 12);   // base label
    }
    const cw = g.maxCols * g.D + 46, chh = g.maxRows * g.D, th = this.labels ? 2 * g.tok + 8 : 2 * g.tok + 48;
    const chip = (p: [number, number], n: Seat, r: number) => box(p, CHIP.w, CHIP.h, 0, (n === 'A' ? -1 : 1) * (tokHalf(n, r) + 5 + CHIP.w / 2));
    for (let c = -4; c <= 4; c++) {
      for (const s of [-1, 1]) {
        box(P(c * g.sp, (s * g.hr) / 2), cw, chh);                              // a full cluster with its count
        box(P(c * g.sp, s * g.hr), 2 * g.tok + 8, th);                          // a champion (with its floats)
        if (this.labels) chip(P(c * g.sp, s * g.hr), s < 0 ? 'A' : 'B', g.tok);
      }
    }
    for (const [c, s] of [[-2, 1], [2, -1]] as const) {
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
        box([w[0] + (n === 'A' ? -1 : 1) * (g.pw / 2 - g.tok - 5), w[1]], CHIP.w, CHIP.h, 0, (n === 'A' ? -1 : 1) * (g.ph / 2 + 7 + CHIP.w / 2));
        chip(P((n === 'A' ? -4 : 4) * g.sp, ((n === 'A' ? 1 : -1) * g.hr) / 2), n, g.tok);             // away, at the base
      }
    }
    return { boxes, x0: Math.min(...boxes.map((q) => q.x0)), x1: Math.max(...boxes.map((q) => q.x1)), y0: Math.min(...boxes.map((q) => q.y0)), y1: Math.max(...boxes.map((q) => q.y1)) };
  }

  /** Fit the drawing to the window and the surface; returns whether anything changed. */
  layout(force = false): boolean {
    const W = innerWidth, H = innerHeight, obs = obstacles(), free = freeRect(obs);
    const key = `${W}x${H}:${+this.labels}:${obs.map((o) => [o.x0, o.y0, o.x1, o.y1].map(Math.round).join(',')).join(';')}`;
    if (key === this.key && !force) return false;
    this.key = key;
    this.W = W; this.H = H; this.safe = free;
    const short = free.x1 - free.x0 < 640;
    this.legend = { x: free.x0 + 6, y: free.y0 + 12, text: short ? 'Each ball is 1 HP of minions' : 'Each ball is 1 HP of minions (a new wave is 8)' };
    this.block = [...obs, { x0: free.x0, y0: free.y0, x1: free.x0 + (short ? 200 : 300), y1: free.y0 + 22 }];
    const e = 14, B: Box = { x0: e, y0: e, x1: W - e, y1: H - e };
    this.bounds = B;
    const t0 = Math.round(Math.max(25, Math.min(55, (Math.atan2(free.y1 - free.y0, free.x1 - free.x0) * 180) / Math.PI)));
    const angles = [...new Set([t0, 25, 40, 32, ...(t0 > 40 ? [48, 55] : [])])].map((d) => (d * Math.PI) / 180);
    const scales: number[] = [];
    for (let a = 1.6; a >= 0.3; a *= 0.97) scales.push(a);
    const memo = new Map<number, { g: Sizes; t: number; c: Pt } | null>();
    const fit = (i: number) => {
      if (!memo.has(i)) {
        const g = this.sizes(scales[i]);
        let r: { g: Sizes; t: number; c: Pt } | null = null;
        for (const t of angles) {
          const ex = this.extents(g, t), c = placeDrawing(ex, B, this.block, free);
          if (c) { r = { g, t, c }; break; }
        }
        memo.set(i, r);
      }
      return memo.get(i)!;
    };
    let lo = 0, hi = scales.length - 1, best: { g: Sizes; t: number; c: Pt };
    if (fit(hi)) {                                       // a smaller drawing fits wherever a larger one does
      while (lo < hi) { const mid = (lo + hi) >> 1; if (fit(mid)) hi = mid; else lo = mid + 1; }
      best = fit(lo)!;
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

  P(u: number, v: number): Pt { return { x: this.cx + u * this.ca + v * this.sa, y: this.cy - u * this.sa + v * this.ca }; }
  at(c: number, v = 0): Pt { return this.P(c * this.sp, v); }
  towerAt(n: Seat): Pt { return n === 'A' ? this.at(-2, this.tw) : this.at(2, -this.tw); }
  wAt(): Pt { return this.at(0, -this.wv); }

  /** The bush at the ward spot: one silhouette whose top edge is a row of grass spikes, in the bases' style. */
  private drawBush(g: SVGGElement, w: Pt, pw: number, ph: number): void {
    const yb = w.y + ph * 0.4, L = w.x - pw * 0.46, Rr = w.x + pw * 0.46;
    const tips = [[-0.4, 0.62], [-0.27, 0.86], [-0.13, 0.74], [0, 0.98], [0.14, 0.78], [0.28, 0.9], [0.41, 0.6]];
    let d = `M${L} ${yb} Q${L - pw * 0.02} ${yb - ph * 0.3} ${w.x + tips[0][0] * pw} ${yb - tips[0][1] * ph}`;
    for (let i = 1; i < tips.length; i++) {
      const [x0] = tips[i - 1], [x1, h1] = tips[i];
      d += ` L${w.x + ((x0 + x1) / 2) * pw} ${yb - ph * 0.42} L${w.x + x1 * pw} ${yb - h1 * ph}`;
    }
    d += ` Q${Rr + pw * 0.02} ${yb - ph * 0.3} ${Rr} ${yb} Q${w.x} ${yb + ph * 0.12} ${L} ${yb} Z`;
    svg('path', { d, class: 'bush-shape' }, g);
  }

  /** Ball positions of a cluster of `hp` balls, relative to its centre, filled column by column from the back to the
   * front, so lost HP leaves from the front edge. `dir` is +1 for a side marching towards B (A), -1 for B. */
  balls(hp: number, dir: number): { w: number; h: number; pos: [number, number][] } {
    const mc = this.maxCols || 8;
    const cols = hp <= 4 ? Math.max(hp, 1) : hp <= 8 ? 4 : hp <= 12 ? 6 : hp <= 24 ? mc : Math.ceil(hp / (mc === 6 ? 4 : 3));
    const rows = Math.max(1, Math.ceil(hp / cols)), D = this.D;
    const w = cols * D, h = rows * D, out: [number, number][] = [];
    for (let i = 0; i < hp; i++) {
      const c = Math.floor(i / rows), r = i % rows;
      out.push([dir * (-w / 2 + D / 2 + c * D), -h / 2 + D / 2 + r * D]);
    }
    return { w, h, pos: out };
  }

  /** Draw (or update) one cluster: balls appear and fade with the HP the engine reports for this frame. */
  private cluster(key: string, n: Seat, hp: number, cx: number, cy: number, opts: { parent?: SVGGElement; rear?: boolean; siege?: boolean; title?: string } = {}): Cluster {
    const dir = n === 'A' ? 1 : -1, L = this.balls(hp, dir);
    let it = this.packets.get(key);
    if (!it) {
      const g = svg('g', { class: 'mv cl' }, opts.parent || this.gPk);
      it = { g, list: [], label: svg('text', { class: 'dot-count', y: 4 }, g), title: svg('title', {}, g), w: 0, h: 0 };
      this.packets.set(key, it);
      g.style.transform = `translate(${cx}px, ${cy}px)`;
    }
    const reduced = document.body.classList.contains('reduced');
    while (it.list.length > hp) {                     // HP lost: the front balls fade out
      const b = it.list.pop()!;
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
      b.setAttribute('fill-opacity', opts.rear ? '0.55' : '1');
      b.setAttribute('stroke', opts.siege ? 'var(--siege-edge)' : 'none');
    });
    it.label.textContent = String(hp);
    it.label.setAttribute('x', String(dir * -(L.w / 2 + 5)));
    it.label.setAttribute('text-anchor', dir > 0 ? 'end' : 'start');
    it.title.textContent = opts.title || '';
    it.g.style.transform = `translate(${cx}px, ${cy}px)`;
    it.w = L.w;
    it.h = L.h;
    return it;
  }

  private build(): void {
    const s = this.el, { W, H, hr, half, halfH, sp } = this;
    const deg = (-this.t * 180) / Math.PI;
    s.innerHTML = '';
    svg('rect', { x: 0, y: 0, width: W, height: H, class: 'field' }, s);
    const live = this.ctx().live;
    const word: Record<Seat, string> = { A: live ? 'Your' : 'A', B: live ? 'Opponent' : 'B' };
    const pts = (list: Pt[]) => list.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ');
    const line = (p: Pt, q: Pt, cls: string) => svg('line', { x1: p.x, y1: p.y, x2: q.x, y2: q.y, class: cls }, s);
    const lab = (p: Pt, t: string, dy = 0, cls = 'l-label', anchor = 'middle') => {
      const e = svg('text', { x: p.x, y: p.y + dy, 'text-anchor': anchor, class: cls }, s); e.textContent = t; return e; };
    const u0 = -4 * sp - half, u1 = 4 * sp + half;
    svg('polygon', { points: pts([this.P(u0, -hr), this.P(u1, -hr), this.P(u1, hr), this.P(u0, hr)]), class: 'road' }, s);
    for (let c = -3.5; c <= 3.5; c += 1) line(this.at(c, -hr + 3), this.at(c, hr - 3), 'cellsep');
    line(this.P(u0 + 10, 0), this.P(u1 - 10, 0), 'midline');
    const baseDown = half * this.sa + halfH * this.ca;
    for (const n of SEATS) {
      const c = this.at(n === 'A' ? -4 : 4);
      const g = svg('g', { class: 'base', transform: `translate(${c.x.toFixed(1)} ${c.y.toFixed(1)}) rotate(${deg.toFixed(2)})` }, s);
      svg('rect', { x: -half, y: -halfH, width: 2 * half, height: 2 * halfH, rx: 10 * this.k, fill: SOFT[n], stroke: COL[n], 'stroke-width': 1.2 }, g);
      svg('title', {}, g).textContent = `${word[n]} base: waves start here; an away champion waits here`;
      lab(c, `${word[n]} base`, n === 'B' && this.t > STEEP ? -baseDown - 8 : baseDown + 16);
    }
    const w = this.wAt(), { pw, ph } = this;
    line(this.at(0, -this.wv + ph / 2), this.at(0, -hr), 'wpath');
    const wg = svg('g', { class: 'wnode' }, s);
    this.drawBush(wg, w, pw, ph);
    svg('title', {}, wg).textContent = 'The ward spot: a bush beside the middle of the lane. A champion here can place a ward, which gives vision for a few turns';
    lab({ x: w.x - pw / 2 - 8, y: w.y }, 'Ward spot', 4.5, 'l-label ward', 'end');
    this.towerG = {};
    for (const n of SEATS) {
      const tp = this.towerAt(n);
      const g = svg('g', { class: 'tower' }, s);
      svg('circle', { cx: tp.x, cy: tp.y, r: this.rt, fill: SOFT[n], stroke: COL[n], 'stroke-width': 1.6 }, g);
      svg('circle', { cx: tp.x, cy: tp.y, r: this.rt * 0.42, fill: COL[n], opacity: 0.85 }, g);
      this.towerG[n] = { g, title: svg('title', {}, g) };
      lab(tp, `${word[n]} tower`, n === 'A' ? this.rt + 22 : -this.rt - 14);
    }
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
    const bx = (p: Pt, bw: number, bh: number, dy = 0, dx = 0): Box => ({ x0: p.x + dx - bw / 2, x1: p.x + dx + bw / 2, y0: p.y + dy - bh / 2, y1: p.y + dy + bh / 2 });
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
    this.fxKey = '';
  }

  private arc(cx: number, cy: number, r: number, a0: number, a1: number): string {
    const p = (a: number) => [cx + r * Math.cos((a * Math.PI) / 180), cy + r * Math.sin((a * Math.PI) / 180)];
    const [x0, y0] = p(a0), [x1, y1] = p(a1);
    return `M${x0.toFixed(1)} ${y0.toFixed(1)} A${r} ${r} 0 ${a1 - a0 > 180 ? 1 : 0} 1 ${x1.toFixed(1)} ${y1.toFixed(1)}`;
  }

  /** Draw one frame. */
  update(f: Frame): void {
    if (!this.gPlates) return;
    const { hr, k } = this, occ: Box[] = [], C = this.ctx();
    const clBox = (x: number, y: number, it: Cluster) => occ.push({ x0: x - it.w / 2 - 22, x1: x + it.w / 2 + 22, y0: y - it.h / 2, y1: y + it.h / 2 });
    this.gPlates.innerHTML = '';
    for (const n of SEATS) {
      const t = f.towers[n], c = this.towerAt(n), rr = this.rt + 5;
      for (let i = 0; i < t.plates_total; i++) {
        const a0 = -90 + i * 120 + 10, lost = i < t.plates_lost;
        svg('path', { d: this.arc(c.x, c.y, rr, a0, a0 + 100), fill: 'none', stroke: COL[n], 'stroke-width': lost ? 1.2 : 4,
          'stroke-dasharray': lost ? '3 3' : null, opacity: lost ? 0.6 : 1, 'stroke-linecap': 'round' }, this.gPlates);
      }
      const left = t.plates_total - t.plates_lost;
      this.towerG[n]!.title.textContent = `${C.names[n]}'s tower: ${left} of ${t.plates_total} plates left, damage ${t.damage} of ${t.damage_max}`;
      const vis = f.status[n].vision;
      if (vis > 0) {
        const w = this.wAt(), g = svg('g', {}, this.gPlates);
        const bg = svg('rect', { y: -9, height: 18, rx: 9, class: 'wbadge' }, g);
        svg('path', { d: 'M-9 0 Q0 -7 9 0 Q0 7 -9 0 Z', fill: COL[n] }, g);
        svg('circle', { r: 2.6, class: 'wbadge-pupil' }, g);
        const tx = svg('text', { x: 14, y: 4, class: 'wcount', fill: COL[n] }, g);
        tx.textContent = turnsWord(vis);
        const bw = 32 + tx.getComputedTextLength();
        bg.setAttribute('x', '-14');
        bg.setAttribute('width', String(bw));
        g.setAttribute('transform', `translate(${w.x + (n === 'A' ? -bw + 8 : 18)} ${w.y - this.ph / 2 - 13})`);
        svg('title', {}, g).textContent = `${C.names[n]} placed a ward: vision for ${turnsWord(vis)} more`;
      }
    }
    for (const n of SEATS) {
      const [, hp] = f.lane.siege[n], key = `S${n}`;
      if (hp > 0) {
        const tn = other(n), tw = this.towerAt(tn), L = this.balls(hp, 1), side = tn === 'A' ? 1 : -1;
        const off = this.rt + 14 + L.w / 2, x = tw.x + side * off * this.ca, y = tw.y - side * off * this.sa;
        clBox(x, y, this.cluster(key, n, hp, x, y, { parent: this.gSiege, siege: true, title: `${hp} HP of ${C.names[n]}'s minions crashed under the tower, hitting it` }));
      } else if (this.packets.has(key)) { this.packets.get(key)!.g.remove(); this.packets.delete(key); }
    }
    const seen = new Set(['SA', 'SB']);
    for (const n of SEATS) {
      const v = n === 'A' ? -hr / 2 + 1 : hr / 2 - 1;
      let lastX: number | null = null, stack = 0;
      f.lane.packets[n].forEach(([cx, hp, front], i) => {
        const key = `${n}${i}`;
        seen.add(key);
        stack = cx === lastX ? stack + 1 : 0;
        lastX = cx;
        const p = this.P(cx * this.sp + (n === 'A' ? -1 : 1) * stack * 14, v);
        clBox(p.x, p.y, this.cluster(key, n, hp, p.x, p.y, { rear: !front, title: `${C.names[n]}'s ${front ? 'leading wave' : 'wave behind'}: ${hp} HP of minions` }));
      });
    }
    for (const [key, it] of this.packets) if (!seen.has(key)) { it.g.remove(); this.packets.delete(key); }
    this.gContact.innerHTML = '';
    if (f.lane.contact !== null && f.lane.contact !== undefined) {
      const p = this.at(f.lane.contact), r = 7.5 * Math.min(1, k + 0.1);
      svg('circle', { cx: p.x, cy: p.y, r, class: 'contact' }, this.gContact);
      svg('path', { d: `M${p.x - r * 0.45} ${p.y - r * 0.45} l${r * 0.9} ${r * 0.9} m0 ${-r * 0.9} l${-r * 0.9} ${r * 0.9}`, class: 'contact-x', 'stroke-width': 1.5, 'stroke-linecap': 'round' }, this.gContact);
      svg('title', {}, this.gContact).textContent = 'The two waves are fighting here';
      occ.push({ x0: p.x - r - 2, x1: p.x + r + 2, y0: p.y - r - 2, y1: p.y + r + 2 });
    }
    const pos = {} as Record<Seat, Pt & { mode: 'lane' | 'ward' | 'away' }>;
    for (const n of SEATS) {
      const cell = f.cells[n];
      if (cell !== null && cell !== undefined) pos[n] = { ...this.at(cell, n === 'A' ? -hr : hr), mode: 'lane' };
      else if (f.at_ward[n]) { const w = this.wAt(); pos[n] = { x: w.x + (n === 'A' ? -1 : 1) * (this.pw / 2 - this.tok - 5), y: w.y, mode: 'ward' }; }
      else pos[n] = { ...this.at(n === 'A' ? -4 : 4, n === 'A' ? hr / 2 : -hr / 2), mode: 'away' };
    }
    for (const n of SEATS) {
      const p = pos[n], pin = this.pins[n]!, st = f.status[n];
      pin.g.style.transform = `translate(${p.x}px, ${p.y}px)`;
      pin.g.style.opacity = p.mode === 'away' ? '0.42' : '1';
      pin.g.classList.toggle('inbush', p.mode === 'ward');
      pin.head.setAttribute('stroke-dasharray', p.mode === 'away' ? '3 3' : '');
      pin.title.textContent = `${C.names[n]}: ${p.mode === 'away' ? (st.status === 'dead' ? 'dead' : 'recalling') + ', back in ' + turnsWord(st.returns_in)
        : p.mode === 'ward' ? `in the bush at the ward spot, level ${st.level}, ${st.hp} of ${st.hp_max} HP. ${BUSH_NOTE}`
          : 'level ' + st.level + ', ' + st.hp + ' of ' + st.hp_max + ' HP'}`;
    }
    const chipFx: Record<Seat, string> = { A: '', B: '' };
    if (this.labels && f.sub === 'tick' && f.tick) for (const n of SEATS) { if (f.tick.minion_fire[n].units > 0 && pos[n].mode === 'lane') chipFx[n] = 'minions hit'; }
    if (this.labels && f.sub === 'end' && f.turn) {
      for (const n of SEATS) { const h = C.turns[f.turn - 1]?.delta[n].hp_taken ?? 0; if (h > 0 && pos[n].mode !== 'away') chipFx[n] = `−${h} HP`; }
    }
    if (this.labels) this.placeChips(f, pos, occ, chipFx);
    const key = `${f.turn || ''}:${f.sub}:${f.label}`;
    if (key === this.fxKey) return;
    this.fxKey = key;
    this.gFx.innerHTML = '';
    const away = (n: Seat) => (n === 'A' ? -1 : 1) * (this.tok + 9) + (n === 'A' ? 0 : 8);
    if (f.sub === 'tick' && f.tick) {
      for (const n of SEATS) {
        const val = f.tick.wave_removed[n], ef = f.lane.packets[other(n)].find((q) => q[2]);
        if (val > 0 && ef) {
          const ew = this.balls(ef[1], 1).w, toRight = n === 'A';
          const p = this.at(ef[0], other(n) === 'A' ? -hr / 2 : hr / 2);
          const t = svg('text', { x: p.x + (toRight ? -ew / 2 - 6 : ew / 2 + 6), y: p.y + 4, 'text-anchor': toRight ? 'end' : 'start', class: 'float' }, this.gFx);
          t.textContent = `−${val}`;
        }
        if (!this.labels && f.tick.minion_fire[n].units > 0 && pos[n].mode === 'lane') {
          const t = svg('text', { x: pos[n].x, y: pos[n].y + away(n), 'text-anchor': 'middle', class: 'float hit' }, this.gFx);
          t.textContent = 'minions hit';
        }
      }
    } else if (f.sub === 'end' && f.turn) {
      const tr = C.turns[f.turn - 1];
      for (const n of SEATS) {
        const h = tr?.delta[n].hp_taken ?? 0;
        if (!this.labels && h > 0 && pos[n].mode !== 'away') {
          const y = pos[n].mode === 'ward' ? pos[n].y - this.ph / 2 - 6 : pos[n].y + away(n);
          const t = svg('text', { x: pos[n].x, y, 'text-anchor': 'middle', class: 'float hit' }, this.gFx);
          t.textContent = `−${h} HP`;
        }
      }
    }
  }

  /** Each champion's chip goes to the first spot around its token that covers nothing; if every spot covers
   * something, to the one that covers least. A's spots start on its left (outside the road), B's on its right. */
  private placeChips(f: Frame, pos: Record<Seat, Pt & { mode: string }>, occ: Box[], fx: Record<Seat, string>): void {
    const h = CHIP.h, ext = (n: Seat) => tokHalf(n, this.tok), C = this.ctx();
    const ORDER: Record<Seat, string[]> = { A: ['left', 'upleft', 'downleft', 'up', 'down', 'upright', 'right', 'downright'],
      B: ['right', 'downright', 'upright', 'down', 'up', 'downleft', 'left', 'upleft'] };
    const spot = (k: string, w: number, e: number): [number, number] => ({ left: [-e - 5 - w, -h / 2], right: [e + 5, -h / 2], up: [-w / 2, -e - 5 - h], down: [-w / 2, e + 5],
      upleft: [-e - 2 - w, -e - 2 - h], upright: [e + 2, -e - 2 - h], downleft: [-e - 2 - w, e + 2], downright: [e + 2, e + 2] } as Record<string, [number, number]>)[k];
    const B = this.bounds, tokBox = (n: Seat): Box => { const e = ext(n) + 2; return { x0: pos[n].x - e, x1: pos[n].x + e, y0: pos[n].y - e, y1: pos[n].y + e }; };
    const taken: Box[] = [];
    this.chipBoxes = {};
    for (const n of SEATS) {
      const c = this.chips[n]!, st = f.status[n], p = pos[n], present = st.status === 'present';
      c.nm.textContent = C.head[n];
      c.lv.textContent = ` · Level ${st.level}`;
      for (const e of [c.bar0, c.bar, c.t2]) e.style.display = present ? '' : 'none';
      c.aw.style.display = present ? 'none' : '';
      if (present) {
        c.bar.setAttribute('width', (48 * Math.max(0, Math.min(1, st.hp / st.hp_max))).toFixed(1));
        c.hp.textContent = String(st.hp);
        c.fx.textContent = fx[n];
      } else c.aw.textContent = `${st.status === 'dead' ? 'Dead' : 'Recalling'}, back in ${turnsWord(st.returns_in)}`;
      const w = Math.ceil(Math.max(c.t1.getComputedTextLength(), present ? 54 + c.t2.getComputedTextLength() : c.aw.getComputedTextLength())) + 16;
      c.bg.setAttribute('width', String(w));
      c.title.textContent = `${C.names[n]}: level ${st.level}, ${present ? `${st.hp} of ${st.hp_max} HP` : st.status === 'dead' ? 'dead' : 'recalling'}`;
      c.g.style.opacity = p.mode === 'away' ? '0.8' : '1';
      const block = [...this.fixed, ...occ, ...taken, tokBox(other(n))];
      let best: { b: Box; cost: number } | null = null;
      for (const k of ORDER[n]) {
        const [dx, dy] = spot(k, w, p.mode === 'ward' ? Math.max(ext(n), this.ph / 2 + 2) : ext(n));
        const b: Box = { x0: p.x + dx, y0: p.y + dy, x1: p.x + dx + w, y1: p.y + dy + h };
        let cost = 0;
        for (const o of block) cost += overlap(b, o);
        cost += 4 * (w * h - overlap(b, B));                    // off the screen counts four times
        if (!best || cost < best.cost) best = { b, cost };
        if (cost === 0) break;
      }
      taken.push(best!.b);
      this.chipBoxes[n] = best!.b;
      if (!c.placed) { c.g.style.transition = 'none'; c.placed = true; }
      c.g.style.transform = `translate(${best!.b.x0}px, ${best!.b.y0}px)`;
      if (c.g.style.transition) requestAnimationFrame(() => { c.g.style.transition = ''; });
    }
  }

  /** The screen boxes of what the scene draws for a frame: for the overlap check of the browser verification. */
  objectBoxes(f: Frame): Record<string, Box> {
    const out: Record<string, Box> = {};
    const r = (el: Element | null | undefined) => { const b = el?.getBoundingClientRect(); return b && b.width > 0 ? { x0: b.left, y0: b.top, x1: b.right, y1: b.bottom } : null; };
    for (const n of SEATS) {
      const pin = r(this.pins[n]?.g); if (pin) out[`champion_${n}`] = pin;
      const tw = r(this.towerG[n]?.g); if (tw) out[`tower_${n}`] = tw;
      const chip = r(this.chips[n]?.g); if (chip && this.labels) out[`chip_${n}`] = chip;
    }
    for (const [key, it] of this.packets) { const b = r(it.g); if (b) out[`wave_${key}`] = b; }
    const w = r(this.el.querySelector('.wnode')); if (w) out.ward_spot = w;
    void f;
    return out;
  }
}
