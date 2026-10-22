// The refined M6a demo: prototype C's lane, prototype A's action bar, a readable gold-lead graph.
// Play is live: every action goes to the local engine (prototypes/m6a/serve.py, lanegame/demo/service.py), which
// resolves it with the tested session core and returns the frames shown here. Watch plays back verified recordings
// (data/*.json) or a new simulation from the engine. This file computes no game rule.

import * as core from '../shared/core.js';

window.__errors = [];
addEventListener('error', (e) => window.__errors.push(String(e.message || e)));
addEventListener('unhandledrejection', (e) => window.__errors.push(String((e.reason && e.reason.message) || e.reason)));

const P = core.params();
const Q = new URLSearchParams(location.search);
const $ = (id) => document.getElementById(id);
const NS = 'http://www.w3.org/2000/svg';
// Minions are drawn as balls, one per HP point (the engine tracks each wave's HP, not single minions; 1 HP of minions
// dying pays the gold of one HP). ?minions=blocks shows the earlier HP blocks for comparison.
const BALLS = new URLSearchParams(location.search).get('minions') !== 'blocks';
const SEATS = ['A', 'B'];
const COL = { A: '#2563a8', B: '#c0392b' };
const SOFT = { A: '#dbe6f4', B: '#f6dcd8' };
const WAVE = { 0: 'Last hit', 2: 'Thin', 3: 'Slow push', 1: 'Shove' };
const BEH = { 0: 'Hold', 1: 'Yield', 2: 'Trade', 3: 'Safe' };
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const sgn = (x, d = 2) => (x > 0 ? '+' : x < 0 ? '−' : '') + Math.abs(x).toFixed(d);
const turns = (k) => `${k} turn${k === 1 ? '' : 's'}`;
const other = (n) => (n === 'A' ? 'B' : 'A');
const empty = () => ({ kind: null, wave: null, beh: null, pressure: false });

const S = { G: null, ctrl: null, mode: 'play', live: null, engine: null, index: null, names: {}, short: {},
            sel: empty(), busy: false, error: '', hover: null, events: false, rows: -1, last: {}, lane: null };

// ------------------------------------------------------------------------------------------- engine
async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  let d = null;
  try { d = await r.json(); } catch { /* not JSON */ }
  if (!r.ok) throw Object.assign(new Error((d && d.message) || `HTTP ${r.status}`), { status: r.status });
  return d;
}

// The shared controller, extended for a game that grows one resolved turn at a time. `n` is the turn cap; the menu
// is the engine's menu for the current decision; nothing past `progress` exists on the client.
class Live extends core.Controller {
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

// ------------------------------------------------------------------------------------------- the lane (prototype C's, trimmed)
function svg(tag, attrs = {}, parent = null) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}

class Lane {
  constructor(el) { this.el = el; this.vw = 0; this.packets = new Map(); this.fxKey = ''; }

  layout(force = false) {
    const compact = (this.el.clientWidth || 1000) < 640;
    const vw = compact ? 480 : 1000;
    if (vw === this.vw && !force) return;
    this.vw = vw;
    this.compact = compact;
    this.Y = BALLS
      ? (compact ? { w: 34, pill: 96, head: 126, siege: 150, road: 176, ra: 182, rb: 218, rowH: 32, label: 290, vh: 302 }
                 : { w: 34, pill: 80, head: 110, siege: 140, road: 158, ra: 164, rb: 210, rowH: 42, label: 282, vh: 294 })
      : (compact ? { w: 34, pill: 96, head: 128, siege: 160, road: 184, ra: 192, rb: 222, rowH: 22, label: 282, vh: 292 }
                 : { w: 34, pill: 86, head: 120, siege: 150, road: 172, ra: 180, rb: 208, rowH: 20, label: 264, vh: 274 });
    this.R = compact ? 3.4 : 5.2;                      // ball radius and pitch
    this.D = compact ? 8.4 : 12.6;
    this.el.setAttribute('viewBox', `0 0 ${vw} ${this.Y.vh}`);
    this.sp = (vw / 2 - (compact ? 30 : 56)) / 4;
    this.packets.clear();
    this.build();
  }

  x(c) { return this.vw / 2 + c * this.sp; }

  // Ball positions of a cluster of `hp` balls, relative to its centre, filled column by column from the back (index 0)
  // to the front, so lost HP leaves from the front edge. `dir` is +1 for a side marching right (A), -1 for B.
  balls(hp, dir) {
    const cols = hp <= 4 ? Math.max(hp, 1) : hp <= 8 ? 4 : hp <= 12 ? 6 : hp <= 24 ? 8 : Math.ceil(hp / 3);
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
      const g = svg('g', { class: 'mv' }, opts.parent || this.gPk);
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
      b.setAttribute('stroke', opts.siege ? '#fff' : 'none');
    });
    it.label.textContent = hp;
    it.label.setAttribute('x', dir * -(L.w / 2 + 5));
    it.label.setAttribute('text-anchor', dir > 0 ? 'end' : 'start');
    it.title.textContent = opts.title || '';
    it.g.style.transform = `translate(${cx}px, ${cy}px)`;
    it.w = L.w;
    return it;
  }
  pw(hp) { return Math.min(this.sp * 0.92, Math.max(this.compact ? 22 : 24, (this.compact ? 10 : 14) + hp * (this.compact ? 1.9 : 3.3))); }

  build() {
    const s = this.el, Y = this.Y;
    s.innerHTML = '';
    const defs = svg('defs', {}, s);
    for (const n of SEATS) {
      const p = svg('pattern', { id: `hatch${n}`, width: 6, height: 6, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' }, defs);
      svg('rect', { width: 6, height: 6, fill: SOFT[n] }, p);
      svg('rect', { width: 2.4, height: 6, fill: COL[n] }, p);
    }
    const roadBot = Y.rb + Y.rowH + 8;
    svg('rect', { x: this.x(-4) - 22, y: Y.road, width: this.x(4) - this.x(-4) + 44, height: roadBot - Y.road, rx: 10, fill: '#ecebe6' }, s);
    svg('line', { x1: this.x(-4) - 8, x2: this.x(4) + 8, y1: (Y.ra + Y.rowH + Y.rb) / 2, y2: (Y.ra + Y.rowH + Y.rb) / 2, stroke: '#d3d1c9', 'stroke-dasharray': '6 7' }, s);
    for (let c = -4; c <= 4; c++) svg('line', { x1: this.x(c), x2: this.x(c), y1: roadBot + 2, y2: roadBot + 7, stroke: '#c6c3ba' }, s);
    const live = S.mode === 'play' && S.live;
    const word = { A: live ? 'Your' : 'A', B: live ? 'Opponent' : 'B' };
    const lab = (c, t) => { const e = svg('text', { x: this.x(c), y: Y.label, 'text-anchor': 'middle', class: 'l-label' }, s); e.textContent = t; return e; };
    if (!this.compact) { lab(-4, `${word.A} base`); lab(4, `${word.B} base`); }
    lab(0, 'Lane');
    if (BALLS) {
      const lg = svg('g', { transform: `translate(${this.x(-4) - 18} ${Y.w - 6})` }, s);
      svg('circle', { r: this.R, cx: 0, cy: 0, fill: '#6b7077' }, lg);
      const lt = svg('text', { x: 9, y: 4, class: 'l-label' }, lg);
      lt.textContent = this.compact ? '= 1 HP of minions' : 'Each ball is 1 HP of minions (a new wave is 8)';
    }
    this.towerLabel = { A: lab(-2, ''), B: lab(2, '') };
    this.towerWord = { A: `${word.A} tower`, B: `${word.B} tower` };
    const wx = this.x(0);
    svg('line', { x1: wx, x2: wx, y1: Y.w + 13, y2: Y.road - 2, stroke: '#c6c3ba', 'stroke-dasharray': '3 5' }, s);
    svg('rect', { x: wx - 12, y: Y.w - 12, width: 24, height: 24, rx: 4, transform: `rotate(45 ${wx} ${Y.w})`, fill: '#fff', stroke: '#8d8a80', 'stroke-width': 1.4 }, s);
    const wt = svg('text', { x: wx, y: Y.w + 4.5, 'text-anchor': 'middle', class: 'l-label strong' }, s);
    wt.textContent = 'W';
    if (!this.compact) {
      const wl = svg('text', { x: wx + 22, y: Y.w + 4, class: 'l-label' }, s);
      wl.textContent = 'Vision node';
    }
    for (const n of SEATS) {
      const sign = n === 'A' ? -1 : 1, bx = this.x(4 * sign), tx = this.x(2 * sign), r = Y.road;
      svg('path', { d: `M${bx - 28} ${r} V ${r - 38} h 10 v -8 h 8 v 8 h 20 v -8 h 8 v 8 h 10 V ${r} Z`, fill: SOFT[n], stroke: COL[n], 'stroke-width': 1.2 }, s);
      const top = r - 92;
      svg('path', { d: `M${tx - 20} ${r} L ${tx - 15} ${top + 14} H ${tx - 20} V ${top} h 8 v 6 h 8 v -6 h 8 v 6 h 8 v -6 h 8 V ${top + 14} H ${tx + 15} L ${tx + 20} ${r} Z`, fill: SOFT[n], stroke: COL[n], 'stroke-width': 1.3 }, s);
    }
    this.gTower = svg('g', {}, s);
    this.gSiege = svg('g', {}, s);
    this.gPk = svg('g', {}, s);
    this.gContact = svg('g', {}, s);
    this.gPins = svg('g', {}, s);
    this.gFx = svg('g', {}, s);
    this.pins = {};
    for (const n of SEATS) {
      const g = svg('g', { class: 'mv' }, this.gPins);
      const stem = svg('line', { x1: 0, x2: 0, y1: 15, y2: Y.road - Y.head, stroke: COL[n], 'stroke-width': 2.2, 'stroke-linecap': 'round' }, g);
      const head = n === 'A' ? svg('circle', { r: 15, fill: COL[n], stroke: '#fff', 'stroke-width': 2 }, g)
        : svg('rect', { x: -12, y: -12, width: 24, height: 24, rx: 2, transform: 'rotate(45)', fill: COL[n], stroke: '#fff', 'stroke-width': 2 }, g);
      const letter = svg('text', { y: 5, 'text-anchor': 'middle', class: 'pin-letter' }, g);
      letter.textContent = n;
      this.pins[n] = { g, stem, head };
    }
  }

  update(f) {
    const Y = this.Y;
    this.gTower.innerHTML = '';
    for (const n of SEATS) {
      const t = f.towers[n], tx = this.x(n === 'A' ? -2 : 2);
      for (let i = 0; i < t.plates_total; i++) {
        const lost = i < t.plates_lost;
        svg('rect', { x: tx - 13, y: Y.road - 54 + i * 13, width: 26, height: 7, rx: 1.5, fill: lost ? '#fff' : COL[n],
                      stroke: COL[n], 'stroke-dasharray': lost ? '2.5 2' : null }, this.gTower);
      }
      const left = t.plates_total - t.plates_lost;
      this.towerLabel[n].textContent = `${this.towerWord[n]} · ${left}/${t.plates_total} plates`;
      const vis = f.status[n].vision;
      if (vis > 0) {
        const g = svg('g', { transform: `translate(${this.x(0) + (n === 'A' ? -26 : 26)} ${Y.w})` }, this.gTower);
        svg('path', { d: 'M-7 0 Q0 -6 7 0 Q0 6 -7 0 Z', fill: '#fff', stroke: COL[n], 'stroke-width': 1.3 }, g);
        svg('circle', { r: 2.3, fill: COL[n] }, g);
        const ti = svg('title', {}, g);
        ti.textContent = `${S.names[n]}: ward vision for ${turns(vis)}`;
      }
    }
    if (!BALLS) this.gSiege.innerHTML = '';
    for (const n of SEATS) {
      const [sx, hp] = f.lane.siege[n];
      if (BALLS) {
        const key = `S${n}`;
        if (hp > 0) {
          // beside the base of the tower it attacks, on the side it came from
          const w = this.balls(hp, 1).w, h = this.balls(hp, 1).h, side = n === 'A' ? -1 : 1;
          this.cluster(key, n, hp, this.x(sx) + side * (24 + w / 2), Y.road - h / 2 - 4, { parent: this.gSiege, siege: true,
            title: `${hp} HP of ${S.names[n]}'s minions crashed under the tower, hitting it` });
        } else if (this.packets.has(key)) { this.packets.get(key).g.remove(); this.packets.delete(key); }
        continue;
      }
      if (hp <= 0) continue;
      const w = Math.min(this.sp * 0.8, 18 + hp * 2.6), x = this.x(sx);
      svg('rect', { x: x - w / 2, y: Y.siege, width: w, height: 17, rx: 4, fill: `url(#hatch${n})`, stroke: COL[n] }, this.gSiege);
      const t = svg('text', { x, y: Y.siege + 12.5, 'text-anchor': 'middle', class: 'pk-text dark' }, this.gSiege);
      t.textContent = hp;
      const ti = svg('title', {}, t);
      ti.textContent = `${hp} HP of ${S.names[n]}'s minions crashed under the tower`;
    }
    const seen = new Set(['SA', 'SB']);
    for (const n of SEATS) {
      const row = n === 'A' ? Y.ra : Y.rb;
      let lastX = null, stack = 0;
      f.lane.packets[n].forEach(([cx, hp, front], i) => {
        const key = `${n}${i}`;
        seen.add(key);
        if (BALLS) {
          stack = cx === lastX ? stack + 1 : 0;
          lastX = cx;
          this.cluster(key, n, hp, this.x(cx) + (n === 'A' ? -1 : 1) * stack * 14, row + Y.rowH / 2, { rear: !front,
            title: `${S.names[n]}'s ${front ? 'leading wave' : 'wave behind'}: ${hp} HP of minions` });
          return;
        }
        stack = cx === lastX ? stack + 1 : 0;
        lastX = cx;
        const w = this.pw(hp);
        let it = this.packets.get(key);
        if (!it) {
          const g = svg('g', { class: 'mv' }, this.gPk);
          it = { g, r: svg('rect', { height: Y.rowH, rx: 5 }, g), t: svg('text', { class: 'pk-text', y: Y.rowH / 2 + 4.5, 'text-anchor': 'middle' }, g) };
          this.packets.set(key, it);
          g.style.transform = `translate(${this.x(cx) - w / 2}px, ${row}px)`;
        }
        it.r.setAttribute('width', w);
        it.r.setAttribute('fill', front ? COL[n] : SOFT[n]);
        it.r.setAttribute('stroke', COL[n]);
        it.t.setAttribute('x', w / 2);
        it.t.setAttribute('class', `pk-text${front ? '' : ' dark'}`);
        it.t.textContent = hp;
        it.g.style.transform = `translate(${this.x(cx) - w / 2}px, ${row - stack * 4}px)`;
      });
    }
    for (const [k, it] of this.packets) if (!seen.has(k)) { it.g.remove(); this.packets.delete(k); }
    this.gContact.innerHTML = '';
    if (f.lane.contact !== null && f.lane.contact !== undefined) {
      const cx = this.x(f.lane.contact), cy = Y.rb + Y.rowH + 12;
      svg('line', { x1: cx, x2: cx, y1: Y.ra - 4, y2: cy - 8, stroke: '#2b2d31', 'stroke-dasharray': '2 3', opacity: 0.5 }, this.gContact);
      svg('circle', { cx, cy, r: 8, fill: '#fff', stroke: '#2b2d31' }, this.gContact);
      svg('path', { d: `M${cx - 3.5} ${cy - 3.5} l7 7 m0 -7 l-7 7`, stroke: '#2b2d31', 'stroke-width': 1.6, 'stroke-linecap': 'round' }, this.gContact);
      const ti = svg('title', {}, this.gContact);
      ti.textContent = 'The two waves are fighting here';
    }
    const pos = {};
    for (const n of SEATS) {
      const cell = f.cells[n];
      if (cell !== null && cell !== undefined) pos[n] = { x: this.x(cell), y: Y.head, mode: 'lane' };
      else if (f.at_ward[n]) pos[n] = { x: this.x(0) + (n === 'A' ? -48 : 48), y: Y.w, mode: 'ward' };
      else pos[n] = { x: this.x(n === 'A' ? -4 : 4), y: Y.head, mode: 'away' };
    }
    if (pos.A.mode === 'lane' && pos.B.mode === 'lane' && Math.abs(pos.A.x - pos.B.x) < 30) { pos.A.x -= 22; pos.B.x += 22; }
    for (const n of SEATS) {
      const p = pos[n], pin = this.pins[n];
      pin.g.style.transform = `translate(${p.x}px, ${p.y}px)`;
      pin.g.style.opacity = p.mode === 'away' ? 0.4 : 1;
      pin.stem.style.opacity = p.mode === 'ward' ? 0 : 1;
      pin.head.setAttribute('stroke-dasharray', p.mode === 'away' ? '3 3' : '');
    }
    // per-tick facts at the ticks, the turn's HP losses at its end
    const key = `${f.turn || ''}:${f.sub}:${f.label}`;
    if (key === this.fxKey) return;
    this.fxKey = key;
    this.gFx.innerHTML = '';
    if (f.sub === 'tick' && f.tick) {
      for (const n of SEATS) {
        const v = f.tick.wave_removed[n], ef = f.lane.packets[other(n)].find((q) => q[2]);
        if (v > 0 && ef) {
          const ew = BALLS ? this.balls(ef[1], 1).w : this.pw(ef[1]), left = n === 'A';
          const t = svg('text', { x: this.x(ef[0]) + (left ? ew / 2 + 5 : -ew / 2 - 5), y: (n === 'A' ? Y.rb : Y.ra) + Y.rowH / 2 + 4,
                                  'text-anchor': left ? 'start' : 'end', class: 'float' }, this.gFx);
          t.textContent = `−${v}`;
        }
        if (f.tick.minion_fire[n].units > 0 && pos[n].mode === 'lane') {
          const t = svg('text', { x: pos[n].x, y: Y.pill + 4, 'text-anchor': 'middle', class: 'float hit' }, this.gFx);
          t.textContent = 'minions hit';
        }
      }
    } else if (f.sub === 'end' && f.turn) {
      const tr = S.G.turns[f.turn - 1];
      const close = Math.abs(pos.A.x - pos.B.x) < 80;
      for (const n of SEATS) {
        const h = tr.delta[n].hp_taken;
        if (h > 0 && pos[n].mode !== 'away') {
          const dx = close ? (n === 'A' ? -6 : 6) : 0, anchor = close ? (n === 'A' ? 'end' : 'start') : 'middle';
          const t = svg('text', { x: pos[n].x + dx, y: pos[n].mode === 'ward' ? Y.w - 20 : Y.pill + 4, 'text-anchor': anchor, class: 'float hit' }, this.gFx);
          t.textContent = `−${h} HP`;
        }
      }
    }
  }
}

// ------------------------------------------------------------------------------------------- top bar
function renderTop() {
  $('modePlay').setAttribute('aria-pressed', String(S.mode === 'play'));
  $('modeWatch').setAttribute('aria-pressed', String(S.mode === 'watch'));
  const ctx = $('context');
  if (S.mode === 'play') {
    const opp = S.live ? S.G.meta.seats[1] : null;
    const bel = opp && opp.arm === 'om' ? ` · its guess about your style is ${S.live.belief}` : '';
    ctx.innerHTML = (opp ? `<span>Opponent: <b>${esc(who(opp).name)}</b><span class="muted">${esc(bel)}</span></span>` : '')
      + `<button class="btn" id="btnNew"${S.engine ? '' : ' disabled'}>New game</button>`;
    $('btnNew').addEventListener('click', () => $('newGame').showModal());
  } else {
    const opt = (g) => `<option value="${g.name}"${S.G && S.G.name === g.name ? ' selected' : ''}>${esc(GAME_NAMES[g.name] || g.label)}</option>`;
    const all = S.index ? S.index.games : [];
    let games = `<optgroup label="Recorded research games">${all.filter((g) => g.kind === 'research_record').map(opt).join('')}</optgroup>`
      + `<optgroup label="Generated for this demo">${all.filter((g) => g.kind !== 'research_record').map(opt).join('')}`
      + (S.G && S.G.name === 'new simulation' ? `<option value="__sim" selected>${esc(S.G.label)}</option>` : '') + '</optgroup>';
    ctx.innerHTML = `<label class="sel"><span class="sr-only">Game</span><select id="gamePick">${games}</select></label>
      <button class="btn" id="btnSim"${S.engine ? '' : ' disabled title="Needs the local engine"'}>New game between agents</button>`;
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
    $(`st${n}`).innerHTML = `${glyph(n)}<span class="nm" title="${esc(S.names[n])}">${esc(S.names[n])}</span><span class="lvl">Level ${st.level}</span>${body}`;
  }
  const c = S.ctrl, v = c.view;
  let lab = v === 0 ? 'Start' : f.sub === 'start' || f.sub === 'tick' ? `Turn ${f.turn}${f.sub === 'tick' ? `, tick ${f.tick.tick} of 2` : ''}` : `After turn ${v}`;
  if (S.live && !c.stopped && !c.historical) lab = `Turn ${c.progress + 1} of ${c.n}: your move`;
  $('turnLabel').textContent = lab;
}

function renderBanner() {
  const c = S.ctrl, hist = !!S.live && !c.stopped && c.historical;
  $('banner').hidden = !hist;
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
    box.innerHTML = `<b>Turn ${v}</b><span>${glyph('A')}${esc(actionWords(tr.parts[0]))}</span><span>${glyph('B')}${esc(actionWords(tr.parts[1]))}</span>`;
    return;
  }
  const major = tr.outcomes.filter((e) => e.major);
  const minor = tr.outcomes.length - major.length;
  const text = major.length ? major.slice(0, 3).map((e) => esc(fill(e.text))).join(' · ') : 'No trades, crashes, Recalls or deaths.';
  const more = tr.outcomes.length - Math.min(3, major.length);
  box.innerHTML = `<b>Turn ${v}</b><span class="sum-text">${text}</span>`
    + (more > 0 || minor ? `<button class="linkbtn" id="sumMore">${more > 0 ? `+${more} more` : 'All events'}</button>` : '');
  $('sumMore')?.addEventListener('click', () => openDetails('events'));
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
// whole game otherwise.
function renderGraph() {
  const el = $('graph'), c = S.ctrl, G = S.G;
  const W = Math.max(300, el.clientWidth), H = el.clientHeight || 172;
  const M = { l: 40, r: 12, t: 22, b: 20 };
  const prog = c.progress, running = !!S.live && !c.stopped;
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
  const ahead = { A: S.live ? 'You' : S.short.A, B: S.live ? 'Opponent' : S.short.B };
  const o = [];
  // a tie (the line on zero) stays neutral: each colour is clipped a pixel away from the baseline
  o.push(`<defs><clipPath id="gUp"><rect x="0" y="0" width="${W}" height="${y0 - 1}"/></clipPath>`
    + `<clipPath id="gDn"><rect x="0" y="${y0 + 1}" width="${W}" height="${H}"/></clipPath></defs>`);
  for (let v = -top; v <= top + 1e-9; v += step) {
    const y = Y(v), zero = Math.abs(v) < 1e-9;
    o.push(`<line x1="${M.l}" x2="${W - M.r}" y1="${y}" y2="${y}" class="${zero ? 'g-zero' : 'g-grid'}"/>`);
    o.push(`<text x="${M.l - 6}" y="${y + 4}" text-anchor="end" class="g-axis">${zero ? '0' : Math.abs(v).toFixed(step < 1 ? 1 : 0)}</text>`);
  }
  const xs = niceStep(n, 6);
  for (let p = 0; p <= n; p += xs) o.push(`<text x="${X(p)}" y="${H - 4}" text-anchor="middle" class="g-axis">${p}</text>`);
  o.push(`<path d="${area}" fill="${COL.A}" fill-opacity="0.18" clip-path="url(#gUp)"/>`
    + `<path d="${area}" fill="${COL.B}" fill-opacity="0.18" clip-path="url(#gDn)"/>`);
  o.push(`<path d="${line}" class="g-line" stroke="#8a8f96"/>`);
  o.push(`<path d="${line}" class="g-line" stroke="${COL.A}" clip-path="url(#gUp)"/>`
    + `<path d="${line}" class="g-line" stroke="${COL.B}" clip-path="url(#gDn)"/>`);
  if (S.events) {
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
        else if (e.kind === 'plate') o.push(`<rect x="${x - 3}" y="${y - 3}" width="6" height="6" transform="rotate(45 ${x} ${y})" fill="#fff" stroke="${col}" stroke-width="1.3"/>`);
      }
    }
  }
  const mark = (p, cls) => {
    const v = G.positions[p].lead_A, x = X(p);
    const lead = Math.abs(v) < 0.005 ? 'even' : `${v > 0 ? ahead.A : ahead.B} +${Math.abs(v).toFixed(2)}`;
    const label = `${p === 0 ? 'Start' : 'Turn ' + p}: ${lead}`;
    const w = label.length * 6.4 + 14, lx = Math.max(M.l, Math.min(W - M.r - w, x - w / 2));
    const dot = Math.abs(v) < 0.005 ? '#5b6168' : v > 0 ? COL.A : COL.B;
    return `<g class="${cls}"><line x1="${x}" x2="${x}" y1="${M.t}" y2="${H - M.b}"/><circle cx="${x}" cy="${Y(v)}" r="4" style="stroke:${dot}"/>
      <rect x="${lx}" y="1" width="${w}" height="17" rx="3"/><text x="${lx + w / 2}" y="13" text-anchor="middle">${esc(label)}</text></g>`;
  };
  o.push(mark(c.view, `g-sel${c.historical && running ? ' past' : ''}`));
  if (S.hover !== null && S.hover !== c.view && S.hover <= prog) o.push(mark(S.hover, 'g-hover'));
  o.push(`<text x="${M.l + 6}" y="${M.t + 12}" class="g-side" fill="${COL.A}">${esc(ahead.A)} ahead</text>`);
  o.push(`<text x="${M.l + 6}" y="${H - M.b - 6}" class="g-side" fill="${COL.B}">${esc(ahead.B)} ahead</text>`);
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="Gold lead by turn. ${esc($('gNote').textContent)}">${o.join('')}</svg>`;
  el._geo = { M, W, n };
}

function graphTurn(ev) {
  const g = $('graph')._geo;
  if (!g) return null;
  const r = $('graph').getBoundingClientRect();
  const p = Math.round(((ev.clientX - r.left - g.M.l) / (r.width - g.M.l - g.M.r)) * g.n);
  return Math.max(0, Math.min(S.ctrl.progress, p));
}

function graphNote() {
  $('gNote').textContent = 'How much more gold one side has earned so far. Not a chance of winning.';
}

// ------------------------------------------------------------------------------------------- bottom controls
function renderControls() {
  const box = $('controls'), c = S.ctrl;
  box.className = 'controls';
  if (S.mode === 'play' && !S.live) {
    box.innerHTML = `<p class="note">${S.engine ? 'Starting a game…' : 'Play needs the local engine: run <code>prototypes/m6a/serve.py</code>, then reload. Watch works without it.'}</p>`;
    return;
  }
  if (S.mode === 'watch' || c.stopped) return renderPlayback(box);
  const m = c.menu(), t = c.progress + 1;
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
        <button class="btn primary submit" id="btnAway"${S.busy || c.historical ? ' disabled' : ''}
          aria-label="Continue: your champion is away, so each away turn is played for you">Continue</button>
        <div class="foot"><span></span><button class="linkbtn" id="btnStop">Stop game</button></div>
      </div>
    </div>
    <div class="help">${err}</div>`;
    $('btnAway').addEventListener('click', continueAway);
    $('btnStop').addEventListener('click', stopGame);
    return;
  }
  const s = S.sel, R = {};
  const pr = m.objectives.find((o) => o.id === 1);
  if (s.pressure && !pr.legal) s.pressure = false;
  const btn = (key, label, on, reason, help, extra = '') => {
    R[key] = reason ? `${label} is unavailable. ${reason}` : `${label}. ${help}`;
    return `<button class="opt${on ? ' on' : ''}" data-k="${key}" aria-pressed="${on}" ${reason ? 'aria-disabled="true"' : ''}
      aria-label="${esc(R[key])}" ${extra}>${esc(label)}</button>`;
  };
  const waves = m.waves.map((w) => btn(`w${w.id}`, WAVE[w.id], s.kind === 'lane' && s.wave === w.id, null, w.help)).join('');
  const behs = m.behaviours.map((b) => btn(`b${b.id}`, BEH[b.id], s.kind === 'lane' && s.beh === b.id,
    s.pressure && ![0, 2].includes(b.id) ? 'Pressure combines with Hold or Trade only.' : null, b.help)).join('');
  const press = btn('p', 'Pressure', s.kind === 'lane' && s.pressure, pr.legal ? null : pr.reason, pr.help);
  const [rc, wd] = m.singles;
  const single = (x, kind) => `<button class="single${s.kind === kind ? ' on' : ''}" data-k="${kind}" aria-pressed="${s.kind === kind}"
      ${x.legal ? '' : 'aria-disabled="true"'} aria-label="${esc(x.legal ? `${x.name}. ${x.help}` : `${x.name} is unavailable. ${x.reason}`)}">
      <span class="nm">${x.name}</span><span class="why">${esc(x.legal ? (kind === 'recall' ? 'Back to base' : 'Vision at W') : x.reason)}</span></button>`;
  R.recall = rc.legal ? `Recall. ${rc.help}` : `Recall is unavailable. ${rc.reason}`;
  R.ward = wd.legal ? `Ward. ${wd.help}` : `Ward is unavailable. ${wd.reason}`;
  const code = selectionCode(m);
  const ok = code !== null && m.legal.includes(code) && !c.historical && !S.busy;
  const what = s.kind === 'recall' ? 'Recall' : s.kind === 'ward' ? 'Ward'
    : s.wave !== null && s.beh !== null ? `${WAVE[s.wave]}, ${BEH[s.beh]}${s.pressure ? ', Pressure' : ''}` : null;
  const off = [rc, wd].filter((x) => !x.legal).map((x) => esc(x.reason)).join(' ');
  const base = err || off || (what ? 'Point at any choice to see what it does.' : 'Choose a wave action and a behaviour, or Recall or Ward.');
  box.innerHTML = `<div class="act">
      <div class="rows">
        <div class="r"><span class="lbl">Wave</span><div class="opts">${waves}</div></div>
        <div class="r"><span class="lbl">Behaviour</span><div class="opts">${behs}</div></div>
        <div class="r"><span class="lbl">Objective</span><div class="opts one">${press}</div><span class="why">${esc(pr.legal ? 'Combines with Hold or Trade.' : pr.reason)}</span></div>
      </div>
      <div class="side">
        <div class="singles">${single(rc, 'recall')}${single(wd, 'ward')}</div>
        <button class="btn primary submit" id="btnSubmit" ${ok ? '' : 'disabled'}>${what ? `Play turn ${t}: ${esc(what)}` : `Play turn ${t}`}</button>
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
      if (b.getAttribute('aria-disabled') === 'true') { help.textContent = R[k]; return; }
      choose(k);
    });
  });
  $('btnSubmit').addEventListener('click', () => submit(selectionCode(c.menu())));
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
  box.innerHTML = `<div class="tc-acts"><b>Turn ${v}</b>${acts}</div><ul class="tc-list">${list}</ul>`;
  $('tcMore')?.addEventListener('click', () => openDetails('events'));
}

function renderPlayback(box) {
  const c = S.ctrl;
  box.classList.add('watch');
  const speeds = core.SPEEDS.map((x) => `<button aria-pressed="${x === c.speed}" data-s="${x}">${x}×</button>`).join('');
  let extra = '';
  if (S.live && c.stopped) {
    const pos = S.G.positions[c.progress];
    const mk = S.G.meta.marks || {};
    const marks = [mk.lane_exit && `Gold lead at the lane exit (turn 94): ${sgn(mk.lane_exit.panel.D0)}.`,
                   mk.lead_114 && `After 114 turns: ${sgn(mk.lead_114.lead_A)} (this game only; M5 averaged it over many games).`].filter(Boolean);
    extra = `<div class="stopsum"><b>${c.progress >= c.n ? `The game reached the ${c.n}-turn limit.` : `You stopped the game after turn ${c.progress}.`}</b>
      Stopping declares no winner; the lane above shows the state at the stop. Gold lead ${sgn(pos.lead_A)}.
      ${marks.join(' ')} <button class="linkbtn" id="btnNew2">New game</button></div>`;
  }
  box.innerHTML = `<div class="pb">
      <button class="btn primary" id="btnPlay">${c.playing ? 'Pause' : c.view >= c.progress ? 'Play from the start' : 'Play'}</button>
      <div class="seg speed" role="group" aria-label="Playback speed">${speeds}</div>
      <span class="muted pb-pos">${c.view === 0 ? 'Start' : 'Turn ' + c.view} of ${c.progress}</span>
    </div>${extra}${S.mode === 'watch' ? '<div class="turncard" id="turnCard" aria-live="polite"></div>' : ''}`;
  if (S.mode === 'watch') renderTurnCard(c.frame);
  $('btnPlay').addEventListener('click', () => { c.toggle(); });
  box.querySelectorAll('[data-s]').forEach((b) => b.addEventListener('click', () => { c.setSpeed(+b.dataset.s); setDur(); renderControls(); }));
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
  const c = S.ctrl, v = c.view, G = S.G, pos = G.positions[v], tr = v > 0 ? G.turns[v - 1] : null;
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
    <form method="dialog"><button class="btn">Close</button></form>`;
  $('about').showModal();
}

// ------------------------------------------------------------------------------------------- actions
async function submit(code) {
  const c = S.ctrl;
  if (!(c instanceof Live) || !c.canSubmit || S.busy || code === null || code === undefined) return;
  S.busy = true;
  S.error = '';
  renderControls();
  try {
    const res = await api(`/api/sessions/${S.live.sid}/turns`, { turn: c.progress + 1, code });
    S.sel = empty();
    const anim = c.append(res);
    renderAll();
    await anim;
    if (c.stopped) await finish();
  } catch (e) {
    S.error = e.message;
  } finally {
    S.busy = false;
    renderAll();
  }
}

async function continueAway() {
  while (S.ctrl instanceof Live && S.ctrl.forced() && !S.error) await submit(45);
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
  install(G, new core.Controller(G, { mode: 'watch', view, speed: S.ctrl ? S.ctrl.speed : P.speed, still: P.still }));
}

async function simulate(a = 'om:0', b = 'sp:0') {
  S.ctrl?.pause();
  $('controls').innerHTML = '<p class="note">The engine is playing 150 turns…</p>';
  const G = await api('/api/simulations', { a, b });
  G.label = `New game: ${AGENTS[a].name} vs. ${AGENTS[b].name.toLowerCase()}`;
  S.mode = 'watch';
  S.live = null;
  install(G, new core.Controller(G, { mode: 'watch', view: 0, speed: S.ctrl ? S.ctrl.speed : P.speed, still: P.still }));
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
  $('hA').innerHTML = `${glyph('A')}<span title="${esc(S.names.A)}">${esc(S.head.A)}</span>`;
  $('hB').innerHTML = `${glyph('B')}<span title="${esc(S.names.B)}">${esc(S.head.B)}</span>`;
  graphNote();
  S.lane.layout(true);
  ctrl.addEventListener('change', () => { if (S.ctrl === ctrl) onChange(); });
  setDur();
  renderTop();
  renderAll();
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
  S.last = { view: S.ctrl.view, progress: S.ctrl.progress, stopped: S.ctrl.stopped, playing: S.ctrl.playing };
}

function onChange() {
  const c = S.ctrl, f = c.frame, L = S.last;
  S.lane.update(f);
  renderStatus(f);
  renderSummary(f);
  if (c.progress !== L.progress || c.stopped !== L.stopped) renderHistory();
  if (c.view !== L.view || c.progress !== L.progress || c.stopped !== L.stopped || c.playing !== L.playing) {
    highlightRows(); renderBanner(); renderControls(); renderNav(); renderGraph();
  }
  if (f.sub === 'end' || f.sub === 'position') { renderDetails(); $('laneText').textContent = laneText(f); }
  S.last = { view: c.view, progress: c.progress, stopped: c.stopped, playing: c.playing };
}

function laneText(f) {
  const out = [$('turnLabel').textContent + '.'];
  for (const n of SEATS) {
    const st = f.status[n], front = f.lane.packets[n].find((q) => q[2]);
    out.push(`${S.names[n]}: level ${st.level}, ${st.status === 'present' ? `${st.hp} of ${st.hp_max} HP, cell ${f.cells[n] ?? 'W'}` : st.status}; wave ${front ? `${front[1]} HP at cell ${front[0]}` : 'none'}.`);
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
      Escape: () => (detailsOpen ? closeDetails() : c.returnToCurrent()),
      ' ': () => (S.live && !c.stopped ? null : c.toggle()),
      Enter: () => (S.live && c.canSubmit && !e.target.closest('button') ? submit(selectionCode(c.menu())) : null),
    }[e.key];
    if (act && !(e.key === ' ' && e.target.closest('button'))) { if (act() !== null) e.preventDefault(); }
  });
}

async function main() {
  S.lane = new Lane($('lane'));
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
  const gp = $('graph');
  gp.addEventListener('pointermove', (e) => { const p = graphTurn(e); if (p !== S.hover) { S.hover = p; renderGraph(); highlightRows(); } });
  gp.addEventListener('pointerleave', () => { S.hover = null; renderGraph(); highlightRows(); });
  gp.addEventListener('click', (e) => { const p = graphTurn(e); if (p !== null) S.ctrl.goto(p); });
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
  addEventListener('resize', () => { clearTimeout(rz); rz = setTimeout(() => { if (!S.ctrl) return; S.lane.layout(); renderAll(); }, 120); });
  if (core.prefersReducedMotion() || P.still) document.body.classList.add('reduced');

  S.index = await core.loadIndex();
  try { const st = await api('/api/status'); S.engine = st.ready ? st : null; } catch { S.engine = null; }
  window.__demo = {
    newGame: (o = 'sp:0', b = 'hidden', seed = null) => newGame(o, b, seed), submit: (code) => submit(code),
    select: (sel) => { S.sel = { ...empty(), ...sel }; renderControls(); }, continueAway, stop: stopGame,
    goto: (p) => { S.ctrl.goto(p); return Promise.resolve(); }, watch, simulate,
    hover: (p) => { S.hover = p; renderGraph(); highlightRows(); }, details: (s) => openDetails(s), state: () => ({
      mode: S.mode, progress: S.ctrl.progress, view: S.ctrl.view, stopped: S.ctrl.stopped, live: !!S.live, sid: S.live && S.live.sid,
      turns: S.G.turns.map((t) => t.codes), menu: S.ctrl.menu ? S.ctrl.menu() : null, error: S.error }),
  };
  const mode = P.mode === 'watch' ? 'watch' : 'play';
  if (mode === 'watch') await watch(P.game, P.view ?? 0);
  else if (S.engine) await newGame(Q.get('opp') || 'sp:0', Q.get('belief') || 'hidden', Q.has('seed') ? +Q.get('seed') : null);
  else await playUnavailable();
  if (P.detail) openDetails(P.detail);
  document.body.dataset.ready = '1';
}

async function playUnavailable() {
  const G = await core.loadGame('h2h_om_s0_vs_sp_s0');
  const G0 = { ...G, name: 'unavailable', positions: [G.positions[0]], turns: [], beliefs: null };
  S.mode = 'play';
  S.live = null;
  install(G0, new core.Controller(G0, { mode: 'watch', view: 0, still: P.still }));
}

main();
