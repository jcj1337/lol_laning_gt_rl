// Lane Study, prototype C ("story"): timeline-first and narrated, for a newcomer.
// Throwaway prototype code (docs/m6/m6a_decision_record.md). Every number comes from the exported frame files through
// ../shared/core.js; this file computes no game rule. Play is staged against a verified recording (core.STAGED_NOTE).

import * as core from '../shared/core.js';

window.__errors = [];
window.addEventListener('error', (e) => window.__errors.push(String(e.message || e)));
window.addEventListener('unhandledrejection', (e) => window.__errors.push(String((e.reason && e.reason.message) || e.reason)));

const P = core.params();
const $ = (id) => document.getElementById(id);
const NS = 'http://www.w3.org/2000/svg';
const SEATS = ['A', 'B'];
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

let G, IDX, ctrl, lane;
let isPlay, NAMES, SHORT;
const ui = { sel: { kind: null, wave: null, beh: null, pressure: false }, tab: '', drawer: false,
             lastView: -1, lastProgress: -1, lastStopped: null, matchFor: null, rows: 0 };

// ------------------------------------------------------------------------------------------- words
const other = (n) => (n === 'A' ? 'B' : 'A');
const you = (n) => isPlay && n === 'A';
const subj = (n) => (you(n) ? 'You' : SHORT[n]);
const subjL = (n) => (you(n) ? 'you' : SHORT[n]);
const poss = (n) => (you(n) ? 'Your' : `${SHORT[n]}'s`);
const possL = (n) => (you(n) ? 'your' : `${SHORT[n]}'s`);
const plural = (k, w) => `${k} ${w}${k === 1 ? '' : 's'}`;
const sgn = (x, d = 2) => (x > 0 ? '+' : x < 0 ? '−' : '±') + Math.abs(x).toFixed(d);

// The frame files phrase outcomes for third persons ("{A} was ganked"). In Play seat A is "You": fix person and
// possessives locally (see the report: core.fillText has no second person).
function narrate(text) {
  let s = text;
  if (isPlay) {
    s = s.replace(/^\{A\}'s/, 'Your').replace(/\{A\}'s/g, 'your')
      .replace(/^\{A\} was\b/, 'You were').replace(/^\{A\} is\b/, 'You are')
      .replace(/; \{A\} gains/, '; you gain')
      .replace(/^\{A\}/, 'You').replace(/\{A\}/g, 'you');
  }
  return core.fillText(s, SHORT);
}

function cellWords(x) {
  if (x === null || x === undefined) return 'the Vision node W';
  const side = x < 0 ? 'A' : 'B';
  if (x === 0) return 'the lane (midpoint)';
  if (Math.abs(x) === 4) return `${possL(side)} base`;
  if (Math.abs(x) === 3) return `behind ${possL(side)} tower`;
  if (Math.abs(x) === 2) return `${possL(side)} tower`;
  return `${possL(side)} side of the lane`;
}

function phrase(e) {
  const n = e.seat, o = other(n);
  switch (e.kind) {
    case 'trade': return e.data.result === 'hit' ? `${subj(n)} hit ${subjL(o)}` : e.data.result === 'miss' ? `${poss(n)} trade missed` : null;
    case 'crash': return `${poss(n)} wave crashed`;
    case 'tower_shot': return `Tower shot ${subjL(n)}`;
    case 'gank': return you(n) ? 'You were ganked' : `${SHORT[n]} ganked`;
    case 'death': return `${subj(n)} died`;
    case 'plate': return `${subj(n)} broke a plate`;
    case 'recall': return `${subj(n)} recalled`;
    case 'recall_interrupted': return `${poss(n)} Recall cut`;
    case 'ward': return `${subj(n)} warded`;
    case 'ward_interrupted': return `${poss(n)} ward cut`;
    default: return null;
  }
}

function rowSummary(tr) {
  const ph = tr.outcomes.filter((e) => e.major).map(phrase).filter(Boolean);
  if (!ph.length) return { text: 'Farming', major: false };
  const more = ph.length > 2 ? ` +${ph.length - 2}` : '';
  return { text: ph.slice(0, 2).join(' · ') + more, major: true };
}

function glyph(n, cls = 'glyph') {
  return n === 'A'
    ? `<svg class="${cls}" viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="6" fill="var(--a)"/></svg>`
    : `<svg class="${cls}" viewBox="0 0 16 16" aria-hidden="true"><rect x="3.4" y="3.4" width="9.2" height="9.2" rx="1.2" transform="rotate(45 8 8)" fill="var(--b)"/></svg>`;
}

function waveCaption(n, ln) {
  const pk = ln.packets[n];
  const front = pk.find((p) => p[2]);
  const sg = ln.siege[n][1];
  const parts = [];
  if (front) {
    let where = 'walking';
    if (ln.contact !== null && ln.contact !== undefined) {
      const c = ln.contact;
      where = c === 0 ? 'fighting at the lane' : `fighting on ${possL(c < 0 ? 'A' : 'B')} side`;
    }
    parts.push(`${poss(n)} wave: ${front[1]} HP, ${where}`);
  } else {
    parts.push(`${poss(n)} wave: none in the lane`);
  }
  const rears = pk.filter((p) => !p[2]).length;
  if (rears) parts.push(`${rears} more behind`);
  if (sg > 0) parts.push(`${sg} HP crashed under ${possL(other(n))} tower`);
  return parts.join(' · ');
}

function statusCaption(n, st, atWard) {
  if (st.status === 'recall') return `${subj(n)} ${you(n) ? 'are' : 'is'} recalling · back in ${plural(st.returns_in, 'turn')}`;
  if (st.status === 'dead') return `${subj(n)} died · back in ${plural(st.returns_in, 'turn')}`;
  if (atWard) return `${subj(n)} ${you(n) ? 'are' : 'is'} at the Vision node`;
  return '';
}

// ------------------------------------------------------------------------------------------- the lane (SVG)
function svg(tag, attrs = {}, parent = null) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}

class Lane {
  constructor(el) { this.el = el; this.vw = 0; this.packets = new Map(); this.caps = new Map(); this.fxKey = ''; }

  layout() {
    const w = this.el.clientWidth || 1000;
    const compact = w < 720;
    const vw = compact ? 480 : 1000;
    if (vw === this.vw) return false;
    this.vw = vw;
    this.compact = compact;
    this.vh = compact ? 392 : 330;
    this.el.setAttribute('viewBox', `0 0 ${vw} ${this.vh}`);
    this.el.classList.toggle('compact', compact);
    this.sp = (vw / 2 - (compact ? 30 : 56)) / 4;
    this.Y = compact
      ? { w: 52, head: 180, pill: 144, road: 236, ra: 244, rb: 274, rowH: 22, siege: 212, capA: 326, capB: 348, label: 378 }
      : { w: 52, head: 146, pill: 110, road: 198, ra: 206, rb: 234, rowH: 20, siege: 176, capA: 288, capB: 307, label: 325 };
    this.packets.clear();
    this.caps.clear();
    this.buildStatic();
    return true;
  }

  x(c) { return this.vw / 2 + c * this.sp; }

  pw(hp) { return Math.min(this.sp * 0.92, Math.max(this.compact ? 22 : 24, (this.compact ? 10 : 14) + hp * (this.compact ? 1.9 : 3.3))); }

  buildStatic() {
    const s = this.el, Y = this.Y, vw = this.vw;
    s.innerHTML = '';
    const defs = svg('defs', {}, s);
    for (const [n, col] of [['A', '#1e6f68'], ['B', '#ad4a27']]) {
      const p = svg('pattern', { id: `hatch${n}`, width: 6, height: 6, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' }, defs);
      svg('rect', { width: 6, height: 6, fill: n === 'A' ? '#d7ebe6' : '#f5ddd0' }, p);
      svg('rect', { width: 2.6, height: 6, fill: col }, p);
    }
    const road = svg('linearGradient', { id: 'road', x1: 0, y1: 0, x2: 0, y2: 1 }, defs);
    svg('stop', { offset: 0, 'stop-color': '#eadfcb' }, road);
    svg('stop', { offset: 1, 'stop-color': '#e2d5bd' }, road);
    const tA = svg('linearGradient', { id: 'twA', x1: 0, y1: 0, x2: 0, y2: 1 }, defs);
    svg('stop', { offset: 0, 'stop-color': '#cfe5df' }, tA);
    svg('stop', { offset: 1, 'stop-color': '#b9d7cf' }, tA);
    const tB = svg('linearGradient', { id: 'twB', x1: 0, y1: 0, x2: 0, y2: 1 }, defs);
    svg('stop', { offset: 0, 'stop-color': '#f3d6c7' }, tB);
    svg('stop', { offset: 1, 'stop-color': '#e9c1ac' }, tB);

    // horizon and road
    svg('path', { d: `M0 ${Y.road - 6} Q ${vw * 0.25} ${Y.road - 26} ${vw * 0.5} ${Y.road - 14} T ${vw} ${Y.road - 10} V ${Y.road} H 0 Z`, fill: '#ebe1cf', opacity: 0.6 }, s);
    const roadTop = Y.road, roadBot = Y.rb + Y.rowH + 8;
    svg('rect', { x: this.x(-4) - 22, y: roadTop, width: this.x(4) - this.x(-4) + 44, height: roadBot - roadTop, rx: 12, fill: 'url(#road)' }, s);
    svg('line', { x1: this.x(-4) - 10, x2: this.x(4) + 10, y1: (Y.ra + Y.rowH + Y.rb) / 2, y2: (Y.ra + Y.rowH + Y.rb) / 2, stroke: '#d4c5aa', 'stroke-width': 1.2, 'stroke-dasharray': '6 7' }, s);
    for (let c = -4; c <= 4; c++) {
      svg('line', { x1: this.x(c), x2: this.x(c), y1: roadBot + 2, y2: roadBot + 8, stroke: '#c9b99d', 'stroke-width': 1.2 }, s);
    }
    // cell labels
    const lab = (c, t, strong) => {
      const e = svg('text', { x: this.x(c), y: Y.label, 'text-anchor': 'middle', class: `l-label${strong ? ' strong' : ''}` }, s);
      e.textContent = t;
    };
    const mine = isPlay ? 'Your' : "A's", theirs = isPlay ? 'Their' : "B's";
    if (!this.compact) { lab(-4, `${mine} base`); lab(4, `${theirs} base`); }
    lab(0, 'The lane', true);
    this.towerLabel = { A: svg('text', { x: this.x(-2), y: Y.label, 'text-anchor': 'middle', class: 'l-label strong' }, s),
                        B: svg('text', { x: this.x(2), y: Y.label, 'text-anchor': 'middle', class: 'l-label strong' }, s) };
    this.towerWord = { A: `${mine} tower`, B: `${theirs} tower` };

    // W node
    const wx = this.x(0);
    svg('line', { x1: wx, x2: wx, y1: Y.w + 14, y2: Y.road - 2, stroke: '#bfae92', 'stroke-width': 1.4, 'stroke-dasharray': '3 5' }, s);
    svg('rect', { x: wx - 13, y: Y.w - 13, width: 26, height: 26, rx: 5, transform: `rotate(45 ${wx} ${Y.w})`, fill: '#fffaf0', stroke: '#9f8f75', 'stroke-width': 1.6 }, s);
    const wt = svg('text', { x: wx, y: Y.w + 5, 'text-anchor': 'middle', class: 'l-label strong' }, s);
    wt.textContent = 'W';
    const wl = svg('text', { x: wx + 24, y: Y.w + 4, 'text-anchor': 'start', class: 'l-label' }, s);
    wl.textContent = 'Vision node';

    // bases and towers
    for (const n of SEATS) {
      const sign = n === 'A' ? -1 : 1;
      const bx = this.x(4 * sign), tx = this.x(2 * sign), fill = n === 'A' ? 'url(#twA)' : 'url(#twB)', st = n === 'A' ? '#1e6f68' : '#ad4a27';
      const r = Y.road;
      svg('path', { d: `M${bx - 30} ${r} V ${r - 44} h 10 v -9 h 9 v 9 h 22 v -9 h 9 v 9 h 10 V ${r} Z`, fill, stroke: st, 'stroke-width': 1.4, 'stroke-linejoin': 'round', opacity: 0.9 }, s);
      svg('path', { d: `M${bx - 7} ${r} v -16 a 7 7 0 0 1 14 0 v 16 z`, fill: '#fffaf0', opacity: 0.85 }, s);
      const top = r - 104;
      svg('path', { d: `M${tx - 22} ${r} L ${tx - 16} ${top + 16} H ${tx - 22} V ${top} h 9 v 6 h 8 v -6 h 10 v 6 h 8 v -6 h 9 V ${top + 16} H ${tx + 16} L ${tx + 22} ${r} Z`, fill, stroke: st, 'stroke-width': 1.5, 'stroke-linejoin': 'round', opacity: 0.82 }, s);
      svg('rect', { x: tx - 3.5, y: top + 30, width: 7, height: 16, rx: 3.5, fill: '#fffaf0', opacity: 0.9 }, s);
    }
    this.gTower = svg('g', {}, s);
    this.gSiege = svg('g', {}, s);
    this.gPk = svg('g', {}, s);
    this.gContact = svg('g', {}, s);
    this.gPins = svg('g', {}, s);
    this.gFx = svg('g', {}, s);
    this.gCaps = svg('g', {}, s);
    this.pins = {};
    for (const n of SEATS) this.pins[n] = this.makePin(n);
  }

  makePin(n) {
    const Y = this.Y, g = svg('g', { class: 'mv' }, this.gPins);
    const col = n === 'A' ? '#1e6f68' : '#ad4a27';
    const shadow = svg('ellipse', { cx: 0, cy: Y.road - Y.head + 4, rx: 15, ry: 4, fill: 'rgba(60,40,20,.16)' }, g);
    const stem = svg('line', { x1: 0, x2: 0, y1: 16, y2: Y.road - Y.head, stroke: col, 'stroke-width': 2.4, 'stroke-linecap': 'round' }, g);
    const head = n === 'A'
      ? svg('circle', { r: 17, fill: col, stroke: '#fffaf0', 'stroke-width': 2.5 }, g)
      : svg('rect', { x: -14, y: -14, width: 28, height: 28, rx: 3, transform: 'rotate(45)', fill: col, stroke: '#fffaf0', 'stroke-width': 2.5 }, g);
    const letter = svg('text', { y: 5, 'text-anchor': 'middle', class: 'pin-letter' }, g);
    letter.textContent = n;
    const py = Y.pill - Y.head;
    const pillBg = svg('rect', { x: -32, y: py - 8, width: 64, height: 16, rx: 8, fill: '#fffaf0', stroke: '#d2c4ad' }, g);
    const pillFill = svg('rect', { x: -31, y: py - 7, width: 62, height: 14, rx: 7, fill: n === 'A' ? '#cfe5df' : '#f3d6c7' }, g);
    const pillText = svg('text', { y: py + 4, 'text-anchor': 'middle', class: 'pin-hp' }, g);
    return { g, shadow, stem, head, letter, pillBg, pillFill, pillText };
  }

  update(f) {
    const Y = this.Y;
    // towers: plates as bands on the tower body (the top band breaks first)
    this.gTower.innerHTML = '';
    for (const n of SEATS) {
      const t = f.towers[n], tx = this.x(n === 'A' ? -2 : 2), col = n === 'A' ? '#1e6f68' : '#ad4a27';
      for (let i = 0; i < t.plates_total; i++) {
        const lost = i < t.plates_lost, y = Y.road - 58 + i * 14;
        svg('rect', { x: tx - 15 - i * 0.6, y, width: 30 + i * 1.2, height: 8, rx: 2, fill: lost ? '#fffaf0' : col, opacity: lost ? 0.9 : 0.85,
                      stroke: col, 'stroke-width': 1.1, 'stroke-dasharray': lost ? '2.5 2' : null }, this.gTower);
      }
      const vis = f.status[n].vision;
      if (vis > 0) {
        const ex = this.x(0) + (n === 'A' ? -15 : 15), ey = Y.w - 25;
        const g = svg('g', { transform: `translate(${ex} ${ey})` }, this.gTower);
        svg('path', { d: 'M-8 0 Q0 -7 8 0 Q0 7 -8 0 Z', fill: '#fffaf0', stroke: col, 'stroke-width': 1.4 }, g);
        svg('circle', { r: 2.6, fill: col }, g);
        const tt = svg('text', { x: n === 'A' ? -11 : 11, y: 4, 'text-anchor': n === 'A' ? 'end' : 'start', class: `l-cap ${n.toLowerCase()}` }, g);
        tt.textContent = vis;
        const ti = svg('title', {}, g);
        ti.textContent = `${poss(n)} ward gives vision for ${plural(vis, 'more turn')}`;
      }
      const left = t.plates_total - t.plates_lost;
      this.towerLabel[n].textContent = this.compact ? `${this.towerWord[n]} · ${left}/${t.plates_total}` : `${this.towerWord[n]} · ${left} of ${t.plates_total} plates`;
    }
    // crashed masses (under the ENEMY tower)
    this.gSiege.innerHTML = '';
    for (const n of SEATS) {
      const [sx, hp] = f.lane.siege[n];
      if (hp <= 0) continue;
      const w = Math.min(this.sp * 0.8, 18 + hp * 2.6), x = this.x(sx);
      svg('rect', { x: x - w / 2, y: Y.siege, width: w, height: 18, rx: 5, fill: `url(#hatch${n})`, stroke: n === 'A' ? '#155a54' : '#8c3717', 'stroke-width': 1.3 }, this.gSiege);
      const t = svg('text', { x, y: Y.siege + 13, 'text-anchor': 'middle', class: 'pk-text dark' }, this.gSiege);
      t.textContent = hp;
    }
    // packets
    const seen = new Set();
    for (const n of SEATS) {
      const row = n === 'A' ? Y.ra : Y.rb;
      const list = f.lane.packets[n];
      let lastX = null, stack = 0;
      list.forEach((pk, i) => {
        const [cx, hp, front] = pk;
        const key = `${n}${i}`;
        seen.add(key);
        stack = cx === lastX ? stack + 1 : 0;
        lastX = cx;
        const w = this.pw(hp);
        let it = this.packets.get(key);
        if (!it) {
          const g = svg('g', { class: 'mv' }, this.gPk);
          const r = svg('rect', { class: 'pk-rect', y: 0, height: Y.rowH, rx: 6 }, g);
          const chev = svg('path', { fill: 'none', 'stroke-width': 1.6, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, g);
          const t = svg('text', { class: 'pk-text', y: Y.rowH / 2 + 4.5, 'text-anchor': 'middle' }, g);
          it = { g, r, chev, t };
          this.packets.set(key, it);
          g.style.transform = `translate(${this.x(cx) - w / 2}px, ${row}px)`;
        }
        const col = n === 'A' ? '#1e6f68' : '#ad4a27', soft = n === 'A' ? '#d7ebe6' : '#f5ddd0';
        it.r.style.width = `${w}px`;
        it.r.setAttribute('width', w);
        it.r.setAttribute('fill', front ? col : soft);
        it.r.setAttribute('stroke', col);
        it.r.setAttribute('stroke-width', front ? 0 : 1.2);
        const cy = Y.rowH / 2;
        const ch = w < 34 ? '' : n === 'A' ? `M${w - 9} ${cy - 4} l 4 4 l -4 4` : `M9 ${cy - 4} l -4 4 l 4 4`;
        it.chev.setAttribute('d', ch);
        it.chev.setAttribute('stroke', front ? 'rgba(255,255,255,.75)' : col);
        it.t.setAttribute('x', w / 2);
        it.t.setAttribute('class', `pk-text${front ? '' : ' dark'}`);
        it.t.textContent = hp;
        it.g.style.transform = `translate(${this.x(cx) - w / 2}px, ${row - stack * 4}px)`;
        it.g.style.opacity = 1;
        it.g.dataset.x = this.x(cx);
        it.g.dataset.y = row;
      });
    }
    for (const [k, it] of this.packets) if (!seen.has(k)) { it.g.remove(); this.packets.delete(k); }
    // contact
    this.gContact.innerHTML = '';
    if (f.lane.contact !== null && f.lane.contact !== undefined) {
      const cx = this.x(f.lane.contact), cy = Y.rb + Y.rowH + 12;
      svg('line', { x1: cx, x2: cx, y1: Y.ra - 4, y2: cy - 9, stroke: '#2a2420', 'stroke-width': 1.2, 'stroke-dasharray': '2 3', opacity: 0.55 }, this.gContact);
      svg('circle', { cx, cy, r: 9, fill: '#fffaf0', stroke: '#2a2420', 'stroke-width': 1.3 }, this.gContact);
      svg('path', { d: `M${cx - 4} ${cy - 4} L ${cx + 4} ${cy + 4} M ${cx + 4} ${cy - 4} L ${cx - 4} ${cy + 4}`, stroke: '#2a2420', 'stroke-width': 1.8, 'stroke-linecap': 'round' }, this.gContact);
    }
    // champions
    const pos = {};
    for (const n of SEATS) {
      const st = f.status[n], cell = f.cells[n], atW = f.at_ward[n];
      let x, y, mode;
      if (cell !== null && cell !== undefined) { x = this.x(cell); y = Y.head; mode = 'lane'; }
      else if (atW) { x = this.x(0) + (n === 'A' ? -58 : 116); y = Y.w; mode = 'ward'; }
      else { x = this.x(n === 'A' ? -4 : 4); y = Y.head + (this.compact ? 6 : 10); mode = 'away'; }
      pos[n] = { x, y, mode, st };
    }
    if (pos.A.mode === 'lane' && pos.B.mode === 'lane' && Math.abs(pos.A.x - pos.B.x) < 30) { pos.A.x -= 28; pos.B.x += 28; }
    const close = pos.A.mode === pos.B.mode && pos.A.mode !== 'away' && Math.abs(pos.A.x - pos.B.x) < 70;
    for (const n of SEATS) {
      const p = pos[n], pin = this.pins[n];
      pin.g.style.transform = `translate(${p.x}px, ${p.y}px)`;
      const away = p.mode === 'away';
      pin.g.style.opacity = away ? 0.5 : 1;
      pin.stem.style.opacity = p.mode === 'lane' || away ? 1 : 0;
      pin.shadow.style.opacity = p.mode === 'lane' ? 1 : 0;
      pin.head.setAttribute('stroke-dasharray', away ? '4 3' : '');
      const frac = Math.max(0, Math.min(1, p.st.hp / p.st.hp_max));
      const shift = close ? (n === 'A' ? -16 : 16) : 0;
      for (const e of [pin.pillBg, pin.pillFill]) e.setAttribute('transform', `translate(${shift} 0)`);
      pin.pillText.setAttribute('x', shift);
      const py = p.mode === 'ward' ? 40 : Y.pill - Y.head;
      pin.pillBg.setAttribute('y', py - 8);
      pin.pillFill.setAttribute('y', py - 7);
      pin.pillText.setAttribute('y', py + 4);
      pin.pillFill.setAttribute('width', Math.max(0, 62 * frac));
      pin.pillFill.setAttribute('fill', frac < 0.35 ? (n === 'A' ? '#9fcfc4' : '#eeb59c') : (n === 'A' ? '#cfe5df' : '#f3d6c7'));
      pin.pillText.textContent = away ? (p.st.status === 'dead' ? 'dead' : 'away') : `${p.st.hp} HP`;
    }
    // captions
    const caps = [];
    for (const n of SEATS) {
      const front = f.lane.packets[n].find((q) => q[2]);
      const sg = f.lane.siege[n];
      const anchorX = front ? this.x(front[0]) : (sg[1] > 0 ? this.x(sg[0]) : this.x(n === 'A' ? -3 : 3));
      caps.push({ key: `w${n}`, text: waveCaption(n, f.lane), x: anchorX, y: n === 'A' ? Y.capA : Y.capB, cls: n.toLowerCase() });
      const sc = statusCaption(n, pos[n].st, pos[n].mode === 'ward');
      if (sc) {
        const away = pos[n].mode === 'away';
        const ward = pos[n].mode === 'ward';
        const yy = ward ? Y.w + 5 : away ? pos[n].y + 5 : Y.pill - 16;
        const xx = away ? pos[n].x + (n === 'A' ? 26 : -26) : ward ? pos[n].x + (n === 'A' ? -24 : 24) : pos[n].x;
        const anc = away ? (n === 'A' ? 'start' : 'end') : ward ? (n === 'A' ? 'end' : 'start') : 'middle';
        caps.push({ key: `s${n}`, text: ward ? 'at the Vision node' : sc, x: xx, y: yy, cls: n.toLowerCase(), anchor: anc });
      }
    }
    const want = new Set(caps.map((c) => c.key));
    for (const [k, e] of this.caps) if (!want.has(k)) { e.remove(); this.caps.delete(k); }
    for (const c of caps) {
      let e = this.caps.get(c.key);
      const anchor = c.anchor || 'middle';
      const text = c.text;
      const approx = text.length * (this.compact ? 7.6 : 6.6);
      let x = c.x;
      if (anchor === 'middle') x = Math.max(approx / 2 + 6, Math.min(this.vw - approx / 2 - 6, x));
      if (anchor === 'start' && x + approx > this.vw - 6) x = this.vw - 6 - approx;
      if (anchor === 'end' && x - approx < 6) x = 6 + approx;
      if (!e || e.textContent !== text) {
        if (e) e.remove();
        e = svg('text', { class: `l-cap ${c.cls} fade` }, this.gCaps);
        e.textContent = text;
        this.caps.set(c.key, e);
      }
      e.setAttribute('x', x);
      e.setAttribute('y', c.y);
      e.setAttribute('text-anchor', anchor);
    }
    // per-tick facts (tick frames) and turn-level HP outcomes (end frame only)
    const key = `${f.turn || ''}:${f.sub}:${f.label}`;
    if (key !== this.fxKey) {
      this.fxKey = key;
      this.gFx.innerHTML = '';
      if (f.sub === 'tick' && f.tick) {
        for (const n of SEATS) {
          const v = f.tick.wave_removed[n];
          const enemyFront = f.lane.packets[other(n)].find((q) => q[2]);
          if (v > 0 && enemyFront) {
            const ew = this.pw(enemyFront[1]);
            const left = other(n) === 'B';
            const t = svg('text', { x: this.x(enemyFront[0]) + (left ? -ew / 2 - 6 : ew / 2 + 6), y: (other(n) === 'A' ? Y.ra : Y.rb) + Y.rowH / 2 + 5,
                                    'text-anchor': left ? 'end' : 'start', class: `float ${n.toLowerCase()}` }, this.gFx);
            t.textContent = `−${v} HP`;
          }
          const mf = f.tick.minion_fire[n];
          if (mf.units > 0) {
            const g0 = svg('g', { transform: `translate(${pos[n].x + (n === 'A' ? -24 : 24)} ${pos[n].y + 22})` }, this.gFx);
            const g = svg('g', { class: 'spark' }, g0);
            svg('path', { d: 'M0 -9 L2.5 -2.5 L9 0 L2.5 2.5 L0 9 L-2.5 2.5 L-9 0 L-2.5 -2.5 Z', fill: '#c48a22', stroke: '#7a4b00', 'stroke-width': 0.8 }, g);
            const t = svg('text', { x: n === 'A' ? -12 : 12, y: 4, class: 'l-cap n', 'text-anchor': n === 'A' ? 'end' : 'start' }, g);
            t.textContent = 'minions hit';
          }
        }
      } else if (f.sub === 'end' && f.turn) {
        const tr = G.turns[f.turn - 1];
        for (const n of SEATS) {
          const h = tr.delta[n].hp_taken;
          if (h > 0 && pos[n].mode !== 'away') {
            const t = svg('text', { x: pos[n].x + (close ? (n === 'A' ? -16 : 16) : 0), y: (pos[n].mode === 'ward' ? Y.w : Y.pill) - 16, 'text-anchor': 'middle', class: 'float hit' }, this.gFx);
            t.textContent = `−${h} HP`;
          }
        }
      }
    }
  }
}

// ------------------------------------------------------------------------------------------- HUD and text equivalent
function renderHud(f) {
  for (const n of SEATS) {
    const st = f.status[n];
    const hpPct = Math.max(0, Math.min(100, (100 * st.hp) / st.hp_max));
    const xpPct = st.xp_next ? Math.max(0, Math.min(100, (100 * (st.xp - st.xp_level)) / (st.xp_next - st.xp_level))) : 100;
    const away = st.status !== 'present'
      ? `<div class="hud-away">${st.status === 'dead' ? 'Dead' : 'Recalling'} · back in ${plural(st.returns_in, 'turn')}</div>`
      : st.vision > 0 ? `<div class="hud-vis">Ward: vision for ${plural(st.vision, 'more turn')}</div>` : '';
    $(`hud${n}`).innerHTML = `
      <div class="hud-top">${glyph(n)}<span class="hud-name" title="${esc(NAMES[n])}">${esc(NAMES[n])}</span><span class="lvl">Level ${st.level}</span></div>
      <div class="bar hp" role="img" aria-label="${esc(NAMES[n])} HP ${st.hp} of ${st.hp_max}"><i style="width:${hpPct}%"></i></div>
      <div class="hud-line"><span><b>${st.hp}</b> / ${st.hp_max} HP</span><span>earned <b>${st.gold_earned.toFixed(2)}</b> gold</span></div>
      <div class="bar xp" aria-hidden="true"><i style="width:${xpPct}%"></i></div>
      ${away}`;
  }
}

function laneText(f) {
  const out = [f.p === 0 && f.sub === 'position' ? 'Initial position.' : `${f.label}.`];
  for (const n of SEATS) {
    const st = f.status[n];
    const where = st.status !== 'present' ? statusCaption(n, st, false) : `${subj(n)} at ${cellWords(f.cells[n] ?? (f.at_ward[n] ? null : undefined))}, ${st.hp} of ${st.hp_max} HP, level ${st.level}`;
    out.push(`${where}.`);
  }
  for (const n of SEATS) out.push(`${waveCaption(n, f.lane)}.`);
  return out.join(' ');
}

// ------------------------------------------------------------------------------------------- timeline
function renderTimeline() {
  const el = $('timeline');
  const W = Math.max(320, el.clientWidth), H = 94, n = G.turns.length, prog = ctrl.progress;
  const padL = 28, padR = 10;
  const X = (p) => padL + ((W - padL - padR) * p) / n;
  const yMid = 30, amp = 18;
  let m = 1;
  for (let p = 0; p <= prog; p++) m = Math.max(m, Math.abs(G.positions[p].lead_A));
  const Ylead = (v) => yMid - (v / m) * amp;
  let path = '';
  for (let p = 0; p <= prog; p++) path += `${p ? 'L' : 'M'}${X(p).toFixed(1)} ${Ylead(G.positions[p].lead_A).toFixed(1)}`;
  const parts = [];
  parts.push(`<rect x="${padL}" y="8" width="${W - padL - padR}" height="${H - 24}" rx="6" fill="#fbf7ef"/>`);
  if (ctrl.mode === 'play' && !ctrl.stopped && prog < n) {
    parts.push(`<pattern id="fut" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="7" height="7" fill="#f4ecdf"/><rect width="2" height="7" fill="#e6dac6"/></pattern>`);
    parts.push(`<rect x="${X(prog)}" y="8" width="${X(n) - X(prog)}" height="${H - 24}" fill="url(#fut)"/>`);
    if (X(n) - X(prog) > 110) parts.push(`<text class="t-note" x="${(X(prog) + X(n)) / 2}" y="${yMid + 4}" text-anchor="middle">not played yet</text>`);
  }
  parts.push(`<line x1="${padL}" x2="${W - padR}" y1="${yMid}" y2="${yMid}" stroke="#d2c4ad" stroke-dasharray="3 4"/>`);
  const narrowT = W < 520;
  for (const [p, lab] of [[94, narrowT ? '94 exit' : 'lane exit · 94'], [114, narrowT ? '114 lead' : 'observed lead · 114']]) {
    if (p > n) continue;
    parts.push(`<line x1="${X(p)}" x2="${X(p)}" y1="8" y2="${H - 16}" stroke="#c48a22" stroke-width="1.2" stroke-dasharray="2 3"/>`);
    parts.push(`<text class="t-cap" x="${X(p) + (p === 94 ? -4 : 4)}" y="${H - 4}" text-anchor="${p === 94 ? 'end' : 'start'}" fill="#7a4b00">${lab}</text>`);
  }
  for (let p = 0; p <= n; p += 25) {
    if (p > 94 - (narrowT ? 26 : 14) && p < 114 + (narrowT ? 36 : 22)) continue;
    parts.push(`<text class="t-axis" x="${X(p)}" y="${H - 4}" text-anchor="middle">${p}</text>`);
  }
  const noteRight = X(ctrl.view) < W / 2;
  const note = narrowT ? 'earned-gold lead, not a score' : `line: earned-gold lead (${isPlay ? 'you' : 'A'} − ${isPlay ? SHORT.B : 'B'}) · not a score or a win chance`;
  parts.push(`<text class="t-note" x="${noteRight ? W - padR - 6 : padL + 6}" y="1" text-anchor="${noteRight ? 'end' : 'start'}">${esc(note)}</text>`);
  parts.push(`<path d="${path}" fill="none" stroke="#2a2420" stroke-width="1.6" stroke-linejoin="round"/>`);
  // event markers (shape-coded), one row per seat
  const rowY = { A: 58, B: 70 };
  for (const s of SEATS) parts.push(`<text class="t-row" x="${padL - 6}" y="${rowY[s] + 3.5}" text-anchor="end">${s}</text>`);
  for (let t = 1; t <= prog; t++) {
    for (const e of G.turns[t - 1].outcomes) {
      const x = X(t), y = rowY[e.seat], col = e.seat === 'A' ? '#1e6f68' : '#ad4a27';
      if (e.kind === 'death') parts.push(`<path d="M${x - 4} ${y - 4} L${x + 4} ${y + 4} M${x + 4} ${y - 4} L${x - 4} ${y + 4}" stroke="${col}" stroke-width="2.2" stroke-linecap="round"/>`);
      else if (e.kind === 'recall') parts.push(`<path d="M${x} ${y - 4.5} L${x + 4.5} ${y + 3.5} L${x - 4.5} ${y + 3.5} Z" fill="none" stroke="${col}" stroke-width="1.4"/>`);
      else if (e.kind === 'crash') parts.push(`<rect x="${x - 2.6}" y="${y - 2.6}" width="5.2" height="5.2" fill="${col}" opacity=".75"/>`);
      else if (e.kind === 'plate') parts.push(`<rect x="${x - 3.4}" y="${y - 3.4}" width="6.8" height="6.8" transform="rotate(45 ${x} ${y})" fill="#fffaf0" stroke="${col}" stroke-width="1.5"/>`);
    }
  }
  // cursor
  const v = ctrl.view, cx = X(v), lead = G.positions[v].lead_A;
  parts.push(`<line x1="${cx}" x2="${cx}" y1="6" y2="${H - 16}" stroke="${ctrl.historical ? '#c48a22' : '#2a2420'}" stroke-width="2"/>`);
  parts.push(`<circle cx="${cx}" cy="${Ylead(lead)}" r="4.5" fill="#fffaf0" stroke="${ctrl.historical ? '#c48a22' : '#2a2420'}" stroke-width="2"/>`);
  const lab = `${v === 0 ? 'start' : 'turn ' + v} · ${sgn(lead)}`;
  const lw = lab.length * 6.4 + 12;
  const lx = Math.max(padL, Math.min(W - padR - lw, cx - lw / 2));
  parts.push(`<rect x="${lx}" y="-12" width="${lw}" height="17" rx="8.5" fill="${ctrl.historical ? '#c48a22' : '#2a2420'}"/>`);
  parts.push(`<text x="${lx + lw / 2}" y="0" text-anchor="middle" style="font:600 11px var(--sans);fill:#fff">${lab}</text>`);
  if (ctrl.mode === 'play' && ctrl.historical) {
    parts.push(`<path d="M${X(prog)} 6 V ${H - 16}" stroke="#2a2420" stroke-width="1.4" stroke-dasharray="3 2"/>`);
    parts.push(`<text class="t-cap" x="${X(prog) + 4}" y="46">now</text>`);
  }
  el.innerHTML = `<svg viewBox="0 -14 ${W} ${H + 14}" height="${H + 14}" aria-hidden="true">${parts.join('')}</svg>`;
  el.setAttribute('aria-valuemax', String(prog));
  el.setAttribute('aria-valuenow', String(v));
  el.setAttribute('aria-valuetext', `${v === 0 ? 'Initial position' : 'After turn ' + v}; earned-gold lead ${sgn(lead)}`);
  el._X = { padL, padR, W, n };
}

function timelinePos(ev) {
  const el = $('timeline'), r = el.getBoundingClientRect(), g = el._X;
  if (!g) return null;
  const p = Math.round(((ev.clientX - r.left - g.padL) / (r.width - g.padL - g.padR)) * g.n);
  return Math.max(0, Math.min(ctrl.progress, p));
}

// ------------------------------------------------------------------------------------------- history
function historyHeads() {
  $('historyCols').innerHTML = `<span>#</span><span>${glyph('A')}${esc(SHORT.A)}</span><span>${glyph('B')}${esc(SHORT.B)}</span>`;
}

function renderHistory() {
  const list = $('historyList');
  if (list.children.length === 0 || ui.rows > ctrl.progress) {
    list.innerHTML = '';
    ui.rows = -1;
  }
  list.querySelector('.pending')?.remove();
  for (let t = ui.rows + 1; t <= ctrl.progress; t++) {
    const li = document.createElement('li');
    li.className = 'hrow' + (t === 0 ? ' start' : '');
    li.dataset.p = t;
    li.tabIndex = -1;
    if (t === 0) {
      li.innerHTML = `<span class="n">0</span><span class="act">Initial position</span>`;
      li.setAttribute('aria-label', 'Initial position');
    } else {
      const tr = G.turns[t - 1];
      const sum = rowSummary(tr);
      li.innerHTML = `<span class="n">${t}</span><span class="act a">${core.codeShort(tr.parts[0])}</span><span class="act b">${core.codeShort(tr.parts[1])}</span>
        <span class="sum${sum.major ? ' major' : ''}">${esc(sum.text)}</span>`;
      li.setAttribute('aria-label', `Turn ${t}: ${subj('A')} ${tr.names[0]}, ${subj('B')} ${tr.names[1]}. ${sum.text}`);
    }
    li.addEventListener('click', () => { ctrl.goto(+li.dataset.p); closeDrawer(); });
    list.appendChild(li);
  }
  ui.rows = ctrl.progress;
  list.querySelectorAll('.now').forEach((e) => e.remove());
  if (ctrl.mode === 'play' && !ctrl.stopped) {
    const last = list.querySelector(`[data-p="${ctrl.progress}"]`);
    if (last) last.insertAdjacentHTML('beforeend', '<span class="now">now</span>');
    const li = document.createElement('li');
    li.className = 'hrow pending';
    li.innerHTML = `<span class="n">${ctrl.progress + 1}</span><span class="act a">${ctrl.forced() ? 'away' : 'your choice…'}</span><span class="act b">✓ chosen</span>
      <span class="sum">${esc(SHORT.B)} has chosen; both are revealed together</span>`;
    li.setAttribute('aria-label', `Turn ${ctrl.progress + 1}, not played yet. ${SHORT.B} has already chosen; both actions are revealed together.`);
    list.appendChild(li);
  }
}

function highlightHistory() {
  const list = $('historyList');
  list.querySelectorAll('.viewed').forEach((e) => { e.classList.remove('viewed'); e.removeAttribute('aria-current'); });
  const row = list.querySelector(`[data-p="${ctrl.view}"]`);
  if (row) {
    row.classList.add('viewed');
    row.setAttribute('aria-current', 'true');
    const lr = list.getBoundingClientRect(), rr = row.getBoundingClientRect();
    if (rr.top < lr.top + 4 || rr.bottom > lr.bottom - 4) list.scrollTop += rr.top - lr.top - lr.height / 2 + rr.height / 2;
  }
}

// ------------------------------------------------------------------------------------------- the narrated card
function renderCard(f) {
  const card = $('card'), v = ctrl.view;
  if (v === 0) {
    card.innerHTML = `<div class="kicker">The starting position</div><h3>Before the first turn</h3>
      <p class="quiet">Both champions start at their own tower. The first waves spawn at each base and walk toward the lane,
      where they meet and fight. ${isPlay && !ctrl.stopped && ctrl.progress === 0 ? 'Choose your first action on the right.' : ''}</p>`;
    return;
  }
  const tr = G.turns[v - 1];
  const played = `<div class="played">
      <span class="who">${glyph('A')}${esc(subj('A'))}</span><span class="what">${esc(tr.names[0])}</span>
      <span class="who">${glyph('B')}${esc(subj('B'))}</span><span class="what">${esc(tr.names[1])}</span></div>`;
  if (f.sub === 'start' || f.sub === 'tick') {
    const k = f.sub === 'start' ? 0 : f.tick.tick;
    card.innerHTML = `<div class="kicker">Turn ${v} of ${G.turns.length}</div><h3>Both actions revealed</h3>${played}
      <div class="resolving"><span class="ticks-dots">${[0, 1, 2].map((i) => `<i class="${i <= k ? 'on' : ''}"></i>`).join('')}</span>
      ${f.sub === 'start' ? 'The turn begins' : `Tick ${k} of 2: the waves and champions move`}. Outcomes appear when the turn ends.</div>`;
    return;
  }
  const major = tr.outcomes.filter((e) => e.major);
  const minor = tr.outcomes.length - major.length;
  const story = major.length
    ? `<ul class="story">${major.map((e) => `<li class="${e.seat ? e.seat.toLowerCase() : ''}">${esc(narrate(e.text))}</li>`).join('')}</ul>`
    : `<p class="quiet">A quiet turn: both champions farmed. ${esc(waveCaption('A', G.positions[v].lane))}.</p>`;
  const deltas = SEATS.map((n) => {
    const d = tr.delta[n];
    const hp = d.hp_taken > 0 ? `<span>HP −${d.hp_taken}</span>` : (d.hp > 0 ? `<span>HP +${d.hp}</span>` : '');
    return `<div class="delta"><b>${glyph(n)}${esc(subj(n))}</b><span>gold ${sgn(d.gold)}</span><span>XP +${d.xp}</span>${hp}</div>`;
  }).join('');
  let match = '';
  const ls = ctrl.lastSubmission;
  if (isPlay && ls && ls.t === v) {
    const nm = (c) => actionName(c);
    match = ls.matched
      ? `<div class="matchline ok">Your selection, ${esc(nm(ls.selected))}, is the recorded action, so this is what the recording shows.</div>`
      : `<div class="matchline">Staged prototype: you selected ${esc(nm(ls.selected))}, but the recording's action, ${esc(nm(ls.recorded))}, resolved. The real app resolves your own choice.</div>`;
  }
  let mark = '';
  const mk = G.meta.marks || {};
  if (v === 94 && mk.lane_exit) mark = `<div class="marknote">Turn 94 is the lane exit, where M5 read each game's economy. Here the earned-gold lead is ${sgn(mk.lane_exit.panel.D0)}.</div>`;
  if (v === 114 && mk.lead_114) mark = `<div class="marknote">${esc(mk.lead_114.note)} Here: ${sgn(mk.lead_114.lead_A)}.</div>`;
  card.innerHTML = `<div class="card-top"><div class="kicker">Turn ${v} of ${G.turns.length}</div>
    <button class="btn quiet-btn" id="btnDetail">Show every detail${minor ? ` · ${plural(minor, 'smaller event')}` : ''}</button></div>
    <h3>What happened in turn ${v}</h3>${played}${story}${mark}${match}<div class="deltas">${deltas}</div>`;
  $('btnDetail').addEventListener('click', () => openSheet('events'));
}

function actionName(code) {
  if (code === 45) return 'Absent';
  const m = G.menu_static;
  if (code === 12) return 'Recall';
  if (code === 30) return 'Ward';
  for (const [k, c] of Object.entries(m.lane_codes)) {
    if (c !== code) continue;
    const [w, b, o] = k.split(',').map(Number);
    return `${m.waves.find((x) => x.id === w).name} + ${m.behaviours.find((x) => x.id === b).name}${o ? ' + Pressure' : ''}`;
  }
  return `code ${code}`;
}

// ------------------------------------------------------------------------------------------- the composer
function selectionCode(menu) {
  const s = ui.sel, m = G.menu_static;
  if (s.kind === 'recall') return 12;
  if (s.kind === 'ward') return 30;
  if (s.kind === 'lane' && s.wave !== null && s.beh !== null) return m.lane_codes[`${s.wave},${s.beh},${s.pressure ? 1 : 0}`];
  return null;
}

function validateSelection(menu) {
  const s = ui.sel, r = menu.reasons || {};
  if (s.kind === 'recall' && r.recall) ui.sel = { kind: null, wave: null, beh: null, pressure: false };
  if (s.kind === 'ward' && r.ward) ui.sel = { kind: null, wave: null, beh: null, pressure: false };
  if (s.pressure && r.pressure) s.pressure = false;
  if (s.pressure && s.beh !== null && ![0, 2].includes(s.beh)) s.beh = null;
}

function renderComposer() {
  const box = $('composer');
  box.classList.remove('locked');
  if (isPlay && ctrl.stopped) return renderSummary(box);
  if (ctrl.mode !== 'play') return renderWatchPanel(box);
  const menu = ctrl.menu();
  const t = ctrl.progress + 1;
  if (menu.forced !== null) {
    const k = ctrl.awayRun();
    const why = menu.away.status === 'dead' ? 'Your champion died and is away from the lane.' : 'Your champion is recalling to base.';
    box.innerHTML = `<div class="comp-head"><h3>Turn ${t}: you are away</h3>${chosenBadge()}</div>
      <div class="awaybox"><div class="big">${why} Back in ${plural(menu.away.returns_in, 'turn')}.</div>
      <p class="panel-sub">Absent is your only action until you return. One click plays every away turn, one turn at a time,
      each with ${esc(SHORT.B)}'s action chosen first.</p></div>
      <div class="comp-foot"><button class="btn primary" id="btnAway">Continue — you are away for ${plural(k, 'turn')}
      <svg class="arrow" viewBox="0 0 16 16"><path d="M3 8h10M9 4l4 4-4 4"/></svg></button>${stagedPill()}</div>`;
    $('btnAway').addEventListener('click', () => ctrl.continueAway());
    lockIfHistorical(box);
    return;
  }
  validateSelection(menu);
  const m = G.menu_static, r = menu.reasons, s = ui.sel;
  const budgets = G.positions[ctrl.progress].champions.A.wave_budgets || {};
  const pressureOn = s.kind === 'lane' && s.pressure;
  const hints = {};
  const card = (key, attrs, name, sub, help, checked, reason, role = 'radio') => {
    const dis = !!reason;
    hints[key] = dis ? `<b>${esc(name)} is unavailable.</b> ${esc(reason)}` : `<b>${esc(name)}.</b> ${esc(help)}`;
    const label = dis ? `${name}. Unavailable: ${reason}` : `${name}. ${help}`;
    return `<button class="choice${role === 'switch' ? ' toggle' : ''}" role="${role}" aria-checked="${checked}" aria-disabled="${dis}" data-hint="${key}" ${attrs}
      aria-label="${esc(label)}"><span class="nm">${role === 'switch' ? '<span class="sw" aria-hidden="true"></span>' : ''}${esc(name)}</span>
      <span class="sub${dis ? ' why' : ''}">${esc(dis ? reason : sub)}</span></button>`;
  };
  const waves = m.waves.map((w) => card(`w${w.id}`, `data-wave="${w.id}"`, w.name,
    budgets[w.name] !== undefined ? `${budgets[w.name]} HP of minions` : '', w.help, s.kind === 'lane' && s.wave === w.id, null)).join('');
  const SUBS = { 0: 'stand and farm', 1: 'farm, step away', 2: 'attack the champion', 3: 'farm from behind' };
  const behs = m.behaviours.map((b) => {
    const reason = pressureOn && ![0, 2].includes(b.id) ? 'Pressure combines with Hold or Trade only.' : null;
    return card(`b${b.id}`, `data-beh="${b.id}"`, b.name, SUBS[b.id], b.help, s.kind === 'lane' && s.beh === b.id, reason);
  }).join('');
  const pr = m.objectives.find((o) => o.id === 1);
  const pressure = card('p', 'id="tgPressure"', 'Pressure', 'damage their tower', pr.help, pressureOn, r.pressure, 'switch');
  const singles = m.singles.map((x) => card(`s${x.code}`, `data-single="${x.code}"`, x.name, x.code === 12 ? 'back to base' : 'vision at W', x.help,
    s.kind === (x.code === 12 ? 'recall' : 'ward'), x.code === 12 ? r.recall : r.ward)).join('');
  const code = selectionCode(menu);
  const legal = code !== null && menu.legal.includes(code);
  let preview = 'Choose a wave action and a behaviour, or Recall or Ward instead. Point at any choice to see what it does.';
  if (s.kind === 'recall') preview = `<b>Planned: Recall.</b> ${esc(m.singles[0].help)}`;
  else if (s.kind === 'ward') preview = `<b>Planned: Ward.</b> ${esc(m.singles[1].help)}`;
  else if (s.kind === 'lane' && (s.wave !== null || s.beh !== null)) {
    const w = s.wave !== null ? m.waves.find((x) => x.id === s.wave) : null;
    const b = s.beh !== null ? m.behaviours.find((x) => x.id === s.beh) : null;
    const name = [w && w.name, b && b.name, pressureOn && 'Pressure'].filter(Boolean).join(' + ');
    const bits = [w && w.help, b && b.help, pressureOn && pr.help].filter(Boolean).join(' ');
    preview = legal ? `<b>Planned: ${esc(name)}.</b> ${esc(bits)}` : `<b>${esc(name)}</b> · now choose ${w ? 'a behaviour' : 'a wave action'}.`;
  }
  const beliefChip = ctrl.beliefVisible() ? `<button class="btn quiet-btn" id="btnBelief">Belief: shown</button>` : '';
  box.innerHTML = `<div class="comp-head"><h3>Your move · turn ${t}</h3>${chosenBadge()}</div>
    <div class="group"><div class="group-label" id="lgW">Wave</div><div class="choices" role="radiogroup" aria-labelledby="lgW">${waves}</div></div>
    <div class="group"><div class="group-label" id="lgB">Behaviour</div><div class="choices" role="radiogroup" aria-labelledby="lgB">${behs}</div></div>
    <div class="group"><div class="group-label">Or instead</div><div class="extras">${pressure}${singles}</div></div>
    <div class="preview" id="preview" aria-live="polite">${preview}</div>
    <div class="comp-foot"><button class="btn primary" id="btnSubmit" aria-disabled="${!legal}">Play turn ${t}
      <svg class="arrow" viewBox="0 0 16 16"><path d="M3 8h10M9 4l4 4-4 4"/></svg></button>
      <button class="btn" id="btnStop">Stop the game</button>${beliefChip}${stagedPill()}</div>`;
  const pv = $('preview');
  box.querySelectorAll('[data-hint]').forEach((b) => {
    const show = () => { pv.innerHTML = hints[b.dataset.hint]; pv.classList.add('hinting'); };
    const back = () => { pv.innerHTML = preview; pv.classList.remove('hinting'); };
    b.addEventListener('mouseenter', show);
    b.addEventListener('mouseleave', back);
    b.addEventListener('focus', show);
    b.addEventListener('blur', back);
  });
  box.querySelectorAll('[data-wave]').forEach((b) => b.addEventListener('click', () => pick({ wave: +b.dataset.wave })));
  box.querySelectorAll('[data-beh]').forEach((b) => b.addEventListener('click', () => { if (b.getAttribute('aria-disabled') !== 'true') pick({ beh: +b.dataset.beh }); }));
  box.querySelectorAll('[data-single]').forEach((b) => b.addEventListener('click', () => {
    if (b.getAttribute('aria-disabled') === 'true') return;
    ui.sel = { kind: +b.dataset.single === 12 ? 'recall' : 'ward', wave: null, beh: null, pressure: false };
    renderComposer();
    $('composer').querySelector(`[data-single="${b.dataset.single}"]`)?.focus({ preventScroll: true });
  }));
  $('tgPressure').addEventListener('click', () => {
    if (r.pressure) return;
    pick({ pressure: !pressureOn });
  });
  $('btnSubmit').addEventListener('click', submit);
  $('btnStop').addEventListener('click', stopGame);
  $('btnBelief')?.addEventListener('click', () => openSheet('belief'));
  lockIfHistorical(box);
}

function pick(change) {
  const s = ui.sel.kind === 'lane' ? { ...ui.sel } : { kind: 'lane', wave: null, beh: null, pressure: false };
  Object.assign(s, change);
  s.kind = 'lane';
  ui.sel = s;
  const focusKey = Object.keys(change)[0];
  renderComposer();
  const sel = focusKey === 'wave' ? `[data-wave="${change.wave}"]` : focusKey === 'beh' ? `[data-beh="${change.beh}"]` : '#tgPressure';
  $('composer').querySelector(sel)?.focus({ preventScroll: true });
}

function chosenBadge() {
  return `<span class="chosen"><span class="lockdot"><svg viewBox="0 0 10 10" aria-hidden="true"><rect x="2" y="4.5" width="6" height="4.5" rx="1"/><path d="M3.3 4.6V3.4a1.7 1.7 0 0 1 3.4 0v1.2" fill="none" stroke="#8c3717" stroke-width="1.1"/></svg></span>
    ${esc(SHORT.B)} has already chosen · both are revealed together</span>`;
}

function stagedPill() {
  return `<span class="staged" title="${esc(core.STAGED_NOTE)}"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 1.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13zm.8 10H7.2V7h1.6zm0-6H7.2V4h1.6z"/></svg>
    <span>Staged prototype: the recording's action resolves, not necessarily yours.</span></span>`;
}

function lockIfHistorical(box) {
  if (!ctrl.historical) return;
  box.classList.add('locked');
  const b = box.querySelector('#btnSubmit');
  if (b) b.setAttribute('aria-disabled', 'true');
}

function submit() {
  if (!ctrl.canSubmit) return;
  const code = selectionCode(ctrl.menu());
  if (code === null || !ctrl.menu().legal.includes(code)) return;
  ctrl.submit(code);
}

function stopGame() {
  ctrl.stop();
  ctrl.mode = 'review';            // core.play() refuses to restart in a stopped Play game (see the report)
  renderAll();
}

function renderWatchPanel(box) {
  const kind = G.meta.provenance.kind;
  const seat = (n) => {
    const x = G.meta.seats['AB'.indexOf(n)];
    const ds = x.kind === 'model' ? `${x.arm === 'om' ? 'M5 arm A' : 'M5 arm B'}, seed ${x.seed}; sampled as M5 evaluated it` : `Scripted rule: ${x.rule}`;
    return `<div class="seatcard">${glyph(n)}<span class="nm">${esc(n)} · ${esc(NAMES[n])}</span><span class="ds">${esc(ds)}</span></div>`;
  };
  box.innerHTML = `<div class="comp-head"><h3>${isPlay ? 'Reviewing your game' : kind === 'new_simulation' ? 'Watching a new simulation' : 'Watching a recorded game'}</h3></div>
    <div class="seats">${seat('A')}${seat('B')}</div>
    <div class="legend" aria-label="Legend">
      <span>${glyph('A')} seat A (circle)</span><span>${glyph('B')} seat B (diamond)</span>
      <span><svg viewBox="0 0 16 16" aria-hidden="true"><rect x="1" y="4" width="14" height="8" rx="3" fill="#6f9e98"/></svg>a wave (its HP)</span>
      <span><svg viewBox="0 0 16 16" aria-hidden="true"><rect x="1" y="4" width="14" height="8" rx="3" fill="url(#lgh)" stroke="#155a54"/><defs><pattern id="lgh" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="4" height="4" fill="#d7ebe6"/><rect width="1.6" height="4" fill="#1e6f68"/></pattern></defs></svg>crashed under a tower</span>
      <span><svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="6" fill="#fffaf0" stroke="#2a2420"/><path d="M5.5 5.5l5 5m0-5l-5 5" stroke="#2a2420" stroke-width="1.5"/></svg>waves fighting</span>
    </div>
    <p class="panel-sub">Play it with the button beside the timeline, step with ← and →, or click any turn. The recording was
    generated first; playback only shows it.</p>
    <div class="btnrow"><button class="btn" id="btnBelief2">The belief</button><button class="btn" id="btnAbout2">About this game</button></div>`;
  $('btnBelief2').addEventListener('click', () => openSheet('belief'));
  $('btnAbout2').addEventListener('click', openAbout);
}

function renderSummary(box) {
  const p = ctrl.progress, pos = G.positions[p], n = G.turns.length, a = pos.champions.A, b = pos.champions.B;
  const capped = p >= n;
  const row = (k, va, vb) => `<span class="k">${k}</span><span>${va}</span><span>${vb}</span>`;
  const status = (st) => (st.status === 'present' ? 'in lane' : st.status === 'dead' ? `dead, back in ${st.returns_in}` : `recalling, back in ${st.returns_in}`);
  const marks = G.meta.marks || {};
  let mk = '';
  if (p >= 94 && marks.lane_exit) mk += `<li>At the lane exit (turn 94) the earned-gold lead was <b>${sgn(marks.lane_exit.panel.D0)}</b>.</li>`;
  if (p >= 114 && marks.lead_114) mk += `<li>After 114 turns it was <b>${sgn(marks.lead_114.lead_A)}</b>: this game's value. M5 averaged this lead over many games; one game is an example, not an expectation.</li>`;
  if (mk) mk = `<ul class="marklist">${mk}</ul>`;
  box.innerHTML = `<div><h3 class="panel-title">${capped ? `The demo's turn cap: the game stopped after ${n} turns` : `You stopped the game after turn ${p}`}</h3>
    <p class="panel-sub">Stopping is not an ending of the game and declares no winner. The state when it stopped:</p></div>
    <div class="sumgrid"><span></span><span class="h">${glyph('A')} you</span><span class="h">${glyph('B')} ${esc(SHORT.B)}</span>
      ${row('Earned gold', a.gold_earned.toFixed(2), b.gold_earned.toFixed(2))}
      ${row('Level · XP', `${a.level} · ${a.xp}`, `${b.level} · ${b.xp}`)}
      ${row('HP', `${a.hp} / ${a.hp_max}`, `${b.hp} / ${b.hp_max}`)}
      ${row('Status', status(a), status(b))}
      <span class="k">Earned-gold lead</span><span class="wide"><b>${sgn(pos.lead_A)}</b> (you − ${esc(SHORT.B)}) · ${esc(waveCaption('A', pos.lane))}</span></div>
    ${mk}
    <div class="btnrow"><button class="btn primary" id="btnReview">Review from the start</button><button class="btn" id="btnBelief3">${esc(SHORT.B)}'s belief about you</button></div>`;
  $('btnReview').addEventListener('click', () => { ctrl.first(); ctrl.play(); });
  $('btnBelief3').addEventListener('click', () => openSheet('belief'));
}

// ------------------------------------------------------------------------------------------- banner, nav
function renderBanner() {
  const hist = ctrl.mode === 'play' && !ctrl.stopped && ctrl.historical;
  $('banner').hidden = !hist;
  $('stage').classList.toggle('historical', hist);
  $('stageTag').hidden = !hist;
  if (hist) {
    $('bannerText').innerHTML = `<strong>You are looking at ${ctrl.view === 0 ? 'the initial position' : 'turn ' + ctrl.view}.</strong>
      <span>The game is waiting at turn ${ctrl.progress + 1}. Actions are disabled until you return.</span>`;
    $('stageTag').textContent = `Viewing ${ctrl.view === 0 ? 'the initial position' : 'turn ' + ctrl.view} · the game is at turn ${ctrl.progress + 1}`;
  }
}

function renderNav() {
  const v = ctrl.view, n = G.turns.length;
  let where = v === 0 ? 'Initial position' : `After turn ${v}`;
  let sub;
  if (ctrl.mode === 'play' && !ctrl.stopped) sub = ctrl.historical ? `An earlier turn · the game is at turn ${ctrl.progress + 1}` : `Deciding turn ${ctrl.progress + 1} · the demo stops at ${n}`;
  else sub = `${isPlay ? 'Your stopped game' : G.meta.provenance.kind === 'new_simulation' ? 'New simulation' : 'Recorded game'} · ${ctrl.progress} turns`;
  $('navWhere').innerHTML = `${where}<small>${esc(sub)}</small>`;
  const live = ctrl.mode === 'play' && !ctrl.stopped;
  $('btnFirst').disabled = v === 0;
  $('btnPrev').disabled = v === 0;
  $('btnNext').disabled = v >= ctrl.progress;
  $('btnLast').disabled = v >= ctrl.progress;
  const pb = $('btnPlay');
  pb.disabled = live && v >= ctrl.progress;
  pb.classList.toggle('playing', ctrl.playing);
  pb.setAttribute('aria-label', ctrl.playing ? 'Pause' : 'Play');
  for (const b of $('speed').children) b.setAttribute('aria-pressed', String(+b.dataset.s === ctrl.speed));
}

// ------------------------------------------------------------------------------------------- details sheet
const TABS = [['events', 'Events'], ['waves', 'Wave numbers'], ['ticks', 'Tick trace'], ['belief', 'Belief']];

function openSheet(tab) {
  ui.tab = TABS.some((t) => t[0] === tab) ? tab : 'events';
  $('sheet').hidden = false;
  renderSheet();
  $('sheetTabs').querySelector('[aria-selected="true"]')?.focus({ preventScroll: true });
}
function closeSheet() { ui.tab = ''; $('sheet').hidden = true; }

function renderSheet() {
  if ($('sheet').hidden) return;
  $('sheetTabs').innerHTML = TABS.map(([k, l]) => `<button class="tab" role="tab" aria-selected="${k === ui.tab}" data-tab="${k}">${l}</button>`).join('');
  $('sheetTabs').querySelectorAll('.tab').forEach((b) => b.addEventListener('click', () => { ui.tab = b.dataset.tab; renderSheet(); }));
  const v = ctrl.view, body = $('sheetBody');
  const tr = v > 0 ? G.turns[v - 1] : null;
  const head = `<p class="notebox">${v === 0 ? 'The initial position.' : `Turn ${v}: ${esc(subj('A'))} played ${esc(tr.names[0])}; ${esc(subj('B'))} played ${esc(tr.names[1])}.`}</p>`;
  if (ui.tab === 'events') {
    body.innerHTML = head + (tr ? `<table class="dt"><thead><tr><th>#</th><th>Seat</th><th>Event</th><th></th></tr></thead><tbody>
      ${tr.outcomes.map((e, i) => `<tr><td class="mono">${i + 1}</td><td>${e.seat ? glyph(e.seat) : ''}</td><td>${esc(narrate(e.text))}</td>
      <td><span class="tag${e.major ? ' major' : ''}">${e.major ? 'major' : 'minor'}</span></td></tr>`).join('') || '<tr><td colspan="4">No events.</td></tr>'}
      </tbody></table><p class="notebox" style="margin-top:10px">Events follow the resolver's order. Wave movement and fights happen tick by tick; trades, tower shots, ganks, deaths, plates and Recalls are known only for the whole turn.</p>` : '');
  } else if (ui.tab === 'waves') {
    const pos = G.positions[v];
    const side = (n) => {
      const st = pos.champions[n], ln = pos.lane, front = ln.packets[n].find((q) => q[2]);
      const removed = tr ? tr.ticks.map((k) => k.wave_removed[n]).join(' + ') : '—';
      return `<div><h4>${glyph(n)} ${esc(poss(n))} wave</h4><table class="dt"><tbody>
        <tr><td>Front</td><td>${front ? `${front[1]} HP at ${esc(cellWords(front[0]))}` : 'none'}</td></tr>
        <tr><td>Waves behind</td><td>${ln.packets[n].filter((q) => !q[2]).map((q) => `${q[1]} HP`).join(', ') || 'none'}</td></tr>
        <tr><td>Crashed under the enemy tower</td><td>${ln.siege[n][1]} HP</td></tr>
        <tr><td>Wave damage per turn (capacity)</td><td>${st.capacity ?? '—'} HP</td></tr>
        <tr><td>Next turn's budgets</td><td>${Object.entries(st.wave_budgets).map(([k, x]) => `${esc(k)} ${x}`).join(' · ')}</td></tr>
        ${tr ? `<tr><td>Enemy HP removed in turn ${v} (per tick)</td><td>${removed}</td></tr><tr><td>Farm gold in turn ${v}</td><td>${tr.delta[n].farm.toFixed(3)}</td></tr>` : ''}
        </tbody></table></div>`;
    };
    body.innerHTML = head + `<div class="cols2">${side('A')}${side('B')}</div>`;
  } else if (ui.tab === 'ticks') {
    if (!tr) { body.innerHTML = head + '<p>No turn has been played yet.</p>'; return; }
    const fr = core.turnSubframes(G, v);
    const pk = (l) => l.map((q) => `${q[1]}@${q[0] > 0 ? '+' : ''}${q[0]}${q[2] ? '' : '·'}`).join(' ') || '—';
    body.innerHTML = head + `<table class="dt"><thead><tr><th>Frame</th><th>${glyph('A')} waves</th><th>${glyph('B')} waves</th><th>Crashed A / B</th><th>Contact</th><th>Cells A / B</th><th>HP removed A / B</th><th>Minion fire on A / B</th></tr></thead><tbody>
      ${fr.map((f) => `<tr><td>${esc(f.label.replace(`turn ${v}: `, ''))}</td><td class="mono">${pk(f.lane.packets.A)}</td><td class="mono">${pk(f.lane.packets.B)}</td>
        <td class="mono">${f.lane.siege.A[1]} / ${f.lane.siege.B[1]}</td><td class="mono">${f.lane.contact ?? '—'}</td>
        <td class="mono">${f.cells.A ?? (f.at_ward.A ? 'W' : '—')} / ${f.cells.B ?? (f.at_ward.B ? 'W' : '—')}</td>
        <td class="mono">${f.tick ? `${f.tick.wave_removed.A} / ${f.tick.wave_removed.B}` : ''}</td>
        <td class="mono">${f.tick ? `${f.tick.minion_fire.A.units || ''} ${f.tick.minion_fire.A.cause || ''} / ${f.tick.minion_fire.B.units || ''} ${f.tick.minion_fire.B.cause || ''}` : ''}</td></tr>`).join('')}
      </tbody></table><p class="notebox" style="margin-top:10px">Waves as HP@cell (global cells: −2 is ${esc(possL('A'))} tower, +2 ${esc(possL('B'))}); · marks a wave behind the front. HP, gold and XP change only at the end of the turn.</p>`;
  } else {
    const bl = ctrl.belief(v);
    if (!G.beliefs) { body.innerHTML = '<p>No seat in this game models its opponent.</p>'; return; }
    if (!bl) {
      body.innerHTML = `<p class="notebox">${esc(SHORT.B)}'s belief about you is hidden in this game: you chose hidden mode at the start
        (the default). The whole belief history becomes available when the game stops.</p><p class="notebox">${esc(core.BELIEF_NOTE)}</p>`;
      return;
    }
    body.innerHTML = Object.entries(bl.bySeat).map(([s, post]) => {
      const rows = bl.types.map((t, i) => [t, post[i]]).sort((x, y) => y[1] - x[1]);
      return `<h4>What ${esc(SHORT[s])} believes about ${esc(subjL(other(s)))} ${v === 0 ? 'at the start' : `after turn ${v}`}</h4>
        <div class="belief">${rows.map(([t, p]) => `<span>${esc(core.BELIEF_LABELS[t] || t)}</span><span class="bb${s === 'A' ? ' alt' : ''}"><i style="width:${(100 * p).toFixed(1)}%"></i></span><span class="v">${(100 * p).toFixed(1)}%</span>`).join('')}</div>`;
    }).join('');
    body.insertAdjacentHTML('afterbegin', `<p class="notebox">${esc(core.BELIEF_NOTE)}</p>`);
  }
}

// ------------------------------------------------------------------------------------------- about
function openAbout() {
  const m = G.meta, prov = m.provenance;
  const seat = (x) => x.kind === 'model'
    ? `<li><b>${esc(x.seat)} · ${esc(x.label)} (${esc(x.id)})</b>: ${x.arm === 'om' ? 'keeps a Bayesian belief about its opponent\'s style and sees the opponent\'s previous action; half of its training games were against fixed opponents.' : 'trained only against itself.'}
        <div class="ids">file SHA-256 ${esc(x.file_sha256)}</div></li>`
    : `<li><b>${esc(x.seat)} · ${esc(x.label)}</b>: a hand-written rule, "${esc(x.rule)}"</li>`;
  $('aboutBody').innerHTML = `<h2 id="aboutTitle">About this game</h2>
    <p>${esc(G.label)}. ${prov.kind === 'research_record' ? 'A verified M5 recording, re-simulated exactly before export.' : 'A new simulation generated by the demo\'s session core.'}
    ${esc(prov.note || '')}</p>
    <h3>The players</h3><ul>${m.seats.map(seat).join('')}</ul>
    <p>Seeds: chance ${esc(m.seeds.chance)}; policy ${esc(m.seeds.policy)}, ${esc(m.seeds.policy_rule)}.</p>
    <h3>What M5 found (all six seeds, 95% intervals)</h3>
    <ul>
      <li>Both agents keep four registered strategic findings: conditional wave management, conditional freezing and the reply to freezes, selective trading, and Recall timing.</li>
      <li>The opponent-modelling package gains against library-like scripted styles: +9.26 [+8.95, +9.56] gold at the lane exit.</li>
      <li>Against the same competent opponents it does worse than pure self-play: −0.63 [−0.98, −0.27], and −0.44 [−0.72, −0.16] head to head.</li>
      <li>Dedicated attackers exploit both agents (lower bounds +2.96 against opponent modelling, +1.15 against pure self-play); whether one is easier to exploit is inconclusive.</li>
    </ul>
    <p>The gain and the cost stand side by side; neither agent is better overall. M5 never evaluated play against humans, so a game here shows how the policies respond to one person and measures nothing about humans in general.</p>
    <p><button class="btn" id="aboutClose">Close</button></p>`;
  $('aboutClose').addEventListener('click', () => $('about').close());
  $('about').showModal();
}

// ------------------------------------------------------------------------------------------- drawer, header
function openDrawer() { ui.drawer = true; $('history').classList.add('open'); $('scrim').hidden = false; $('turnsToggle').setAttribute('aria-expanded', 'true'); }
function closeDrawer() { if (!ui.drawer) return; ui.drawer = false; $('history').classList.remove('open'); $('scrim').hidden = true; $('turnsToggle').setAttribute('aria-expanded', 'false'); }

function header() {
  const pick = $('gamePicker');
  pick.innerHTML = IDX.games.map((g) => `<option value="${g.name}"${g.name === P.game ? ' selected' : ''}>${esc(g.label)} · ${g.kind === 'research_record' ? 'research record' : 'new simulation'}</option>`).join('');
  pick.addEventListener('change', () => go({ game: pick.value }));
  $('modePlay').setAttribute('aria-pressed', String(isPlay));
  $('modeWatch').setAttribute('aria-pressed', String(!isPlay));
  $('modePlay').addEventListener('click', () => go({ mode: 'play' }));
  $('modeWatch').addEventListener('click', () => go({ mode: 'watch' }));
  const kind = G.meta.provenance.kind;
  $('prov').innerHTML = `<span class="badge ${kind === 'research_record' ? 'research' : 'sim'}" title="${esc(G.meta.provenance.note || '')}"><span class="dot"></span>${kind === 'research_record' ? 'Verified research record' : kind === 'new_simulation' ? 'New simulation' : 'Your game'}</span>
    <span class="prov-note">One game: an example, not evidence</span>`;
  $('aboutBtn').addEventListener('click', openAbout);
  $('turnsToggle').addEventListener('click', () => (ui.drawer ? closeDrawer() : openDrawer()));
  $('scrim').addEventListener('click', closeDrawer);
}

function go(change) {
  const q = new URLSearchParams({ game: P.game, mode: isPlay ? 'play' : 'watch', ...change });
  location.search = q.toString();
}

// ------------------------------------------------------------------------------------------- wiring
function setDur() { document.documentElement.style.setProperty('--dur', `${Math.round((core.BASE_FRAME_MS * 0.92) / ctrl.speed)}ms`); }

function renderAll() {
  lane.update(ctrl.frame);
  renderHud(ctrl.frame);
  renderHistory();
  highlightHistory();
  renderTimeline();
  renderCard(ctrl.frame);
  renderComposer();
  renderBanner();
  renderNav();
  renderSheet();
  ui.lastView = ctrl.view; ui.lastProgress = ctrl.progress; ui.lastStopped = ctrl.stopped;
}

function onChange() {
  const f = ctrl.frame;
  lane.update(f);
  renderHud(f);
  const vc = ctrl.view !== ui.lastView, pc = ctrl.progress !== ui.lastProgress, sc = ctrl.stopped !== ui.lastStopped;
  if (pc || sc) { renderHistory(); }
  if (vc || pc || sc) { highlightHistory(); renderTimeline(); renderBanner(); renderComposer(); }
  renderCard(f);
  renderNav();
  if (f.sub === 'end' || f.sub === 'position') {
    $('laneText').textContent = laneText(f);
    renderSheet();
  }
  ui.lastView = ctrl.view; ui.lastProgress = ctrl.progress; ui.lastStopped = ctrl.stopped;
}

async function main() {
  [IDX, G] = await Promise.all([core.loadIndex(), core.loadGame(P.game)]);
  isPlay = P.mode === 'play';
  const mode = P.mode === 'watch' ? 'watch' : P.mode === 'review' ? 'review' : 'play';
  NAMES = core.seatNames(G, isPlay ? 'play' : 'watch');
  SHORT = core.shortNames(G, isPlay ? 'play' : 'watch');
  if (isPlay) NAMES.A = 'You';
  ctrl = new core.Controller(G, { mode, progress: P.progress, view: P.view, belief: P.belief, speed: P.speed, still: P.still });
  if (isPlay && ctrl.progress >= G.turns.length) { ctrl.stopped = true; ctrl.mode = 'review'; }
  if (core.prefersReducedMotion() || P.still) document.body.classList.add('reduced');
  document.title = `Lane Study · ${isPlay ? 'Play' : 'Watch'} · prototype C`;
  header();
  historyHeads();
  lane = new Lane($('lane'));
  lane.layout();
  $('speed').innerHTML = core.SPEEDS.map((s) => `<button data-s="${s}" aria-pressed="false" aria-label="Speed ${s} times">${s}×</button>`).join('');
  for (const b of $('speed').children) b.addEventListener('click', () => { ctrl.setSpeed(+b.dataset.s); setDur(); renderNav(); });
  setDur();
  $('btnFirst').addEventListener('click', () => ctrl.first());
  $('btnPrev').addEventListener('click', () => ctrl.prev());
  $('btnNext').addEventListener('click', () => ctrl.next());
  $('btnLast').addEventListener('click', () => ctrl.latest());
  $('btnPlay').addEventListener('click', () => ctrl.toggle());
  $('btnReturn').addEventListener('click', () => ctrl.returnToCurrent());
  $('sheetClose').addEventListener('click', closeSheet);
  const tl = $('timeline');
  let drag = false;
  tl.addEventListener('pointerdown', (e) => { drag = true; tl.setPointerCapture(e.pointerId); const p = timelinePos(e); if (p !== null) ctrl.goto(p); });
  tl.addEventListener('pointermove', (e) => { if (!drag) return; const p = timelinePos(e); if (p !== null && p !== ctrl.view) ctrl.goto(p); });
  tl.addEventListener('pointerup', () => { drag = false; });
  ctrl.addEventListener('change', onChange);
  core.bindKeys(ctrl, { Enter: () => (ctrl.canSubmit && !document.activeElement.closest('button') ? submit() : null) });
  window.addEventListener('keydown', (e) => { if (e.key === 'Escape') { closeSheet(); closeDrawer(); } });
  let rz;
  window.addEventListener('resize', () => { clearTimeout(rz); rz = setTimeout(() => { if (lane.layout()) lane.update(ctrl.frame); renderTimeline(); }, 80); });
  renderAll();
  $('laneText').textContent = laneText(ctrl.frame);
  if (P.detail) openSheet(P.detail);
  // hooks for screenshots and demos
  window.__story = {
    ctrl, G,
    select(sel) {
      if ('single' in sel) ui.sel = { kind: sel.single === 12 ? 'recall' : 'ward', wave: null, beh: null, pressure: false };
      else ui.sel = { kind: 'lane', wave: sel.wave ?? null, beh: sel.beh ?? null, pressure: !!sel.pressure };
      renderComposer();
    },
    submit, stop: stopGame, openSheet, closeSheet, openAbout, openDrawer,
    showFrame(t, i) { ctrl.view = t; ctrl.frame = core.turnSubframes(G, t)[i]; onChange(); },
  };
  requestAnimationFrame(() => { document.body.dataset.ready = '1'; });
}

main().catch((e) => { window.__errors.push(String(e && e.stack || e)); document.body.dataset.ready = '1'; });
