// Lane Arena: M6a prototype A (throwaway). A game-like lane over the shared core (../shared/core.js).
// Every number on screen comes from the exported frame files; nothing here computes a game rule.
import * as C from '../shared/core.js';

const P = C.params();
const $ = (id) => document.getElementById(id);
const NS = 'http://www.w3.org/2000/svg';
const S = { index: null, game: null, ctrl: null, names: null, short: null, mode: P.mode === 'watch' ? 'watch' : 'play',
  sel: { wave: null, beh: null, obj: 0, single: null }, drawerTab: P.detail || null, summaryOpen: false,
  lastProgress: -1, lastView: -1, toastedTurn: 0, gameName: P.game };

// ------------------------------------------------------------------------------------------------ geometry
const X = (c) => 600 + c * 128;                   // global cell -> SVG x (bases at -4, +4)
const ROAD_Y = 316, ROAD_H = 110;
const PK_Y = { A: 288, B: 346 };                  // packet rows on the road
const CH_Y = 176, W_Y = 76;                       // champion row; the Vision node W
const TOWER_Y = 416;
const SEAT_COL = { A: 'var(--a)', B: 'var(--b)' };

function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
const h = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html !== undefined) e.innerHTML = html; return e; };
// core.fillText puts "You" in place of {A}; a visitor needs "your" and "you" mid-sentence
function fill(text) {
  if (S.short.A !== 'You') return C.fillText(text, S.short);
  return C.fillText(text.replace(/^\{A\}'s/, 'Your').replace(/\{A\}'s/g, 'your').replace(/^\{A\}/, 'You').replace(/\{A\}/g, 'you'), S.short);
}
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

// seat glyphs: A = circle, B = hexagon (shape as well as colour)
function hexPath(r) {
  const pts = [];
  for (let i = 0; i < 6; i++) { const a = Math.PI / 6 + i * Math.PI / 3; pts.push(`${(r * Math.cos(a)).toFixed(1)},${(r * Math.sin(a)).toFixed(1)}`); }
  return `M${pts.join('L')}Z`;
}
const glyphSvg = (seat, size = 20) => seat === 'A'
  ? `<svg class="hud-glyph" viewBox="-12 -12 24 24" width="${size}" height="${size}" aria-hidden="true"><circle r="10" fill="var(--a-deep)" stroke="var(--a)" stroke-width="2"/><text y="4" text-anchor="middle" font-size="11" font-weight="700" fill="#eafffb" font-family="Barlow Condensed, sans-serif">A</text></svg>`
  : `<svg class="hud-glyph" viewBox="-12 -12 24 24" width="${size}" height="${size}" aria-hidden="true"><path d="${hexPath(10.5)}" fill="var(--b-deep)" stroke="var(--b)" stroke-width="2"/><text y="4" text-anchor="middle" font-size="11" font-weight="700" fill="#fff4ec" font-family="Barlow Condensed, sans-serif">B</text></svg>`;

// ------------------------------------------------------------------------------------------------ static lane art
function buildLane() {
  const svg = $('lane');
  [...svg.querySelectorAll(':scope > :not(title)')].forEach((n) => n.remove());
  const defs = el('defs', {}, svg);
  defs.innerHTML = `
    <radialGradient id="g-ground" cx="50%" cy="55%" r="75%">
      <stop offset="0" stop-color="#173a2f"/><stop offset=".55" stop-color="#0f2721"/><stop offset="1" stop-color="#081613"/></radialGradient>
    <linearGradient id="g-road" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#5a5040"/><stop offset=".12" stop-color="#4a4234"/><stop offset=".5" stop-color="#3c362b"/>
      <stop offset=".88" stop-color="#4a4234"/><stop offset="1" stop-color="#2c271f"/></linearGradient>
    <linearGradient id="g-river" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="#123b2a" stop-opacity="0"/><stop offset=".5" stop-color="#1f5a3d" stop-opacity=".55"/><stop offset="1" stop-color="#123b2a" stop-opacity="0"/></linearGradient>
    <linearGradient id="g-tower-a" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#6fe7da"/><stop offset="1" stop-color="#1c6f67"/></linearGradient>
    <linearGradient id="g-tower-b" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#ffb07f"/><stop offset="1" stop-color="#8f3a17"/></linearGradient>
    <linearGradient id="g-stone" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#4c5a5e"/><stop offset=".5" stop-color="#6e7c80"/><stop offset="1" stop-color="#3f4b4f"/></linearGradient>
    <radialGradient id="g-base-a" r=".6"><stop offset="0" stop-color="#3ed6c4" stop-opacity=".55"/><stop offset="1" stop-color="#3ed6c4" stop-opacity="0"/></radialGradient>
    <radialGradient id="g-base-b" r=".6"><stop offset="0" stop-color="#ff8b4f" stop-opacity=".55"/><stop offset="1" stop-color="#ff8b4f" stop-opacity="0"/></radialGradient>
    <radialGradient id="g-brush" r=".6"><stop offset="0" stop-color="#2f6a3a"/><stop offset="1" stop-color="#173d22"/></radialGradient>
    <filter id="f-grain" x="-20%" y="-20%" width="140%" height="140%">
      <feTurbulence type="fractalNoise" baseFrequency=".9" numOctaves="2" seed="7" result="n"/>
      <feColorMatrix in="n" type="matrix" values="0 0 0 0 .5  0 0 0 0 .55  0 0 0 0 .45  0 0 0 .09 0"/>
      <feComposite in2="SourceGraphic" operator="in"/></filter>
    <filter id="f-blot" x="-10%" y="-10%" width="120%" height="120%"><feTurbulence type="fractalNoise" baseFrequency=".012" numOctaves="3" seed="3"/>
      <feColorMatrix type="matrix" values="0 0 0 0 .06  0 0 0 0 .18  0 0 0 0 .12  0 0 0 .55 -.1"/></filter>
    <filter id="f-shadow" x="-30%" y="-30%" width="160%" height="170%"><feDropShadow dx="0" dy="5" stdDeviation="5" flood-color="#000" flood-opacity=".55"/></filter>
    <filter id="f-glow" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="6"/></filter>
    <pattern id="p-hatch-a" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="8" height="8" fill="#0f4d47"/><rect width="3" height="8" fill="#3ed6c4" opacity=".6"/></pattern>
    <pattern id="p-hatch-b" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(-45)"><rect width="8" height="8" fill="#5c2410"/><rect width="3" height="8" fill="#ff8b4f" opacity=".65"/></pattern>`;
  const bg = el('g', { id: 'terrain' }, svg);
  el('rect', { x: -600, y: -300, width: 2400, height: 1100, fill: 'url(#g-ground)' }, bg);
  el('rect', { x: -600, y: -300, width: 2400, height: 1100, filter: 'url(#f-blot)' }, bg);
  // a soft glade band through the middle (where the two sides meet), and rocks / grass tufts as texture
  el('rect', { x: 380, y: -300, width: 440, height: 1100, fill: 'url(#g-river)', opacity: .6 }, bg);
  const rnd = mulberry(11);
  for (let i = 0; i < 70; i++) {
    const x = -100 + rnd() * 1400, y = -40 + rnd() * 580;
    if (y > ROAD_Y - ROAD_H / 2 - 20 && y < ROAD_Y + ROAD_H / 2 + 14) continue;
    if (rnd() < .55) el('path', { d: `M${x} ${y}q3 -9 6 0q2 -7 5 0`, stroke: '#2f6b48', 'stroke-width': 2, fill: 'none', opacity: .45 + rnd() * .3 }, bg);
    else el('ellipse', { cx: x, cy: y, rx: 4 + rnd() * 9, ry: 2 + rnd() * 5, fill: '#2a3b39', opacity: .5 }, bg);
  }
  // the road
  const road = el('g', { id: 'road' }, svg);
  el('path', { d: `M-40 ${ROAD_Y - ROAD_H / 2 + 4} C 300 ${ROAD_Y - ROAD_H / 2 - 6}, 900 ${ROAD_Y - ROAD_H / 2 - 6}, 1240 ${ROAD_Y - ROAD_H / 2 + 4} L1240 ${ROAD_Y + ROAD_H / 2 - 2} C 900 ${ROAD_Y + ROAD_H / 2 + 8}, 300 ${ROAD_Y + ROAD_H / 2 + 8}, -40 ${ROAD_Y + ROAD_H / 2 - 2} Z`,
    fill: 'url(#g-road)', filter: 'url(#f-shadow)' }, road);
  el('rect', { x: -40, y: ROAD_Y - ROAD_H / 2, width: 1280, height: ROAD_H, fill: '#fff', filter: 'url(#f-grain)', opacity: .7 }, road);
  // cell tiles: faint separators and labels below the road
  const labels = {};
  for (let c = -4; c <= 4; c++) {
    if (c > -4) el('line', { x1: X(c) - 64, x2: X(c) - 64, y1: ROAD_Y - ROAD_H / 2 + 10, y2: ROAD_Y + ROAD_H / 2 - 8, stroke: '#2a251d', 'stroke-width': 2, 'stroke-dasharray': '2 7', opacity: .8 }, road);
    const t = el('text', { x: X(c), y: ROAD_Y - ROAD_H / 2 - 9, 'text-anchor': 'middle', 'font-size': 13, fill: c === 0 ? '#cfe6cf' : '#8fa79c', 'letter-spacing': '.12em', opacity: .85 }, road);
    t.textContent = (labels[c] || '').toUpperCase();
    const n = el('text', { x: X(c), y: ROAD_Y + ROAD_H / 2 - 7, 'text-anchor': 'middle', 'font-size': 10.5, fill: '#8a7e66', opacity: .9, 'letter-spacing': '.08em' }, road);
    n.textContent = c === -4 ? 'A BASE' : (c === 4 ? 'B BASE' : (c === 0 ? 'LANE 0' : (c > 0 ? `+${c}` : `${c}`)));
  }
  // the Vision node W: a brush patch above the lane cell, joined by a trail
  const w = el('g', { id: 'wnode' }, svg);
  el('path', { d: `M${X(0)} ${ROAD_Y - ROAD_H / 2} C ${X(0) - 10} ${W_Y + 70}, ${X(0) + 12} ${W_Y + 50}, ${X(0)} ${W_Y + 28}`, stroke: '#4c4536', 'stroke-width': 9, fill: 'none', 'stroke-linecap': 'round', opacity: .8 }, w);
  for (const [dx, dy, r] of [[-44, 6, 26], [-18, -10, 30], [16, -8, 31], [44, 8, 25], [0, 14, 28], [-30, 22, 20], [32, 24, 20]]) {
    el('circle', { cx: X(0) + dx, cy: W_Y + dy, r, fill: 'url(#g-brush)', stroke: '#245433', 'stroke-width': 2 }, w);
  }
  for (let i = 0; i < 16; i++) {
    const a = rnd() * Math.PI * 2, rr = 20 + rnd() * 34;
    el('path', { d: `M${X(0) + Math.cos(a) * rr} ${W_Y + Math.sin(a) * rr * .6 + 6}l-3 -10l3 4l3 -8l1 9`, stroke: '#4f9a5c', 'stroke-width': 1.6, fill: 'none', opacity: .7 }, w);
  }
  const wl = el('text', { x: X(0) + 76, y: W_Y - 18, 'font-size': 13, fill: '#a9d4ac', 'letter-spacing': '.1em' }, w);
  wl.textContent = 'W · VISION NODE';
  el('g', { id: 'ward-marks' }, w);
  // bases
  for (const [seat, c] of [['A', -4], ['B', 4]]) {
    const g = el('g', { id: `base-${seat}` }, svg);
    el('ellipse', { cx: X(c), cy: ROAD_Y, rx: 92, ry: 78, fill: `url(#g-base-${seat.toLowerCase()})` }, g);
    el('ellipse', { cx: X(c), cy: ROAD_Y + 4, rx: 48, ry: 30, fill: '#1d2a2b', stroke: seat === 'A' ? '#3ed6c4' : '#ff8b4f', 'stroke-width': 2.5, opacity: .95 }, g);
    el('ellipse', { cx: X(c), cy: ROAD_Y + 2, rx: 32, ry: 18, fill: seat === 'A' ? '#14524b' : '#5a2611', opacity: .9 }, g);
    el('path', { d: seat === 'A' ? hexPathAt(X(c), ROAD_Y - 2, 0) : hexPathAt(X(c), ROAD_Y - 2, 1), fill: seat === 'A' ? '#3ed6c4' : '#ff8b4f', opacity: .8 }, g);
  }
  // towers (static bodies; plates and damage are dynamic)
  for (const [seat, c] of [['A', -2], ['B', 2]]) {
    const g = el('g', { id: `tower-${seat}`, filter: 'url(#f-shadow)' }, svg);
    const x = X(c);
    el('ellipse', { cx: x, cy: TOWER_Y + 30, rx: 44, ry: 10, fill: '#000', opacity: .35 }, g);
    el('path', { d: `M${x - 28} ${TOWER_Y + 30} L${x - 21} ${TOWER_Y - 18} L${x + 21} ${TOWER_Y - 18} L${x + 28} ${TOWER_Y + 30} Z`, fill: 'url(#g-stone)', stroke: '#26302f', 'stroke-width': 2 }, g);
    for (let k = 0; k < 3; k++) el('line', { x1: x - 25 + k * 1.5, x2: x + 25 - k * 1.5, y1: TOWER_Y + 24 - k * 15, y2: TOWER_Y + 24 - k * 15, stroke: '#3a4547', 'stroke-width': 1.5 }, g);
    el('path', { d: `M${x - 27} ${TOWER_Y - 18} h54 v-9 h-8 v5 h-7.5 v-5 h-7.5 v5 h-8 v-5 h-7.5 v5 h-7.5 v-5 h-8 z`, fill: '#5d6b6e', stroke: '#26302f', 'stroke-width': 1.5 }, g);
    el('circle', { cx: x + (seat === 'A' ? 36 : -36), cy: TOWER_Y - 8, r: 10, fill: seat === 'A' ? '#3ed6c4' : '#ff8b4f', filter: 'url(#f-glow)', opacity: .7 }, g);
    el('circle', { cx: x + (seat === 'A' ? 36 : -36), cy: TOWER_Y - 8, r: 6, fill: seat === 'A' ? '#3ed6c4' : '#ff8b4f', stroke: '#0c1416', 'stroke-width': 1.5 }, g);
    el('g', { id: `tower-dyn-${seat}` }, svg);
  }
  el('g', { id: 'siege-layer' }, svg);
  el('g', { id: 'packet-layer' }, svg);
  el('g', { id: 'contact-layer' }, svg);
  el('g', { id: 'champ-layer' }, svg);
  el('g', { id: 'float-layer' }, svg);
  S.pk = { A: [], B: [] };
  S.champ = {};
  for (const seat of ['A', 'B']) S.champ[seat] = buildToken(seat);
}
function hexPathAt(cx, cy, flip) {
  const r = 12, pts = [];
  for (let i = 0; i < 6; i++) { const a = Math.PI / 6 + i * Math.PI / 3; pts.push(`${(cx + r * Math.cos(a)).toFixed(1)},${(cy + r * Math.sin(a) * .7).toFixed(1)}`); }
  return flip ? `M${pts.join('L')}Z` : `M${cx - 10} ${cy}a10 7 0 1 0 20 0a10 7 0 1 0 -20 0`;
}
function mulberry(a) { return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }

// ------------------------------------------------------------------------------------------------ dynamic lane objects
function buildToken(seat) {
  const layer = $('champ-layer');
  const g = el('g', { class: 'mv', 'data-seat': seat }, layer);
  const R = 27;
  el('ellipse', { cx: 0, cy: R + 8, rx: 26, ry: 7, fill: '#000', opacity: .45 }, g);
  el('circle', { r: R + 7, fill: 'none', stroke: '#0b1418', 'stroke-width': 7, opacity: .9 }, g);
  const ring = el('circle', { r: R + 7, fill: 'none', stroke: '#62de8b', 'stroke-width': 5, 'stroke-linecap': 'round', transform: 'rotate(-90)', class: 'fade' }, g);
  const body = seat === 'A' ? el('circle', { r: R, fill: 'var(--a-deep)', stroke: 'var(--a)', 'stroke-width': 3 }, g)
    : el('path', { d: hexPath(R + 2), fill: 'var(--b-deep)', stroke: 'var(--b)', 'stroke-width': 3 }, g);
  el('circle', { r: R - 7, fill: 'none', stroke: '#fff', 'stroke-opacity': .12, 'stroke-width': 2 }, g);
  const letter = el('text', { y: 8, 'text-anchor': 'middle', 'font-size': 24, 'font-weight': 700, fill: '#f4fffd' }, g);
  letter.textContent = seat;
  const lv = el('g', { transform: `translate(${R - 2},${R - 6})` }, g);
  el('circle', { r: 12, fill: '#0d1a1e', stroke: '#e7c46a', 'stroke-width': 2 }, lv);
  const lvt = el('text', { y: 4.5, 'text-anchor': 'middle', 'font-size': 13, 'font-weight': 700, fill: '#ffe39a' }, lv);
  const tagG = el('g', { transform: `translate(0,${R + 26})` }, g);
  const tagBg = el('rect', { x: -50, y: -11, width: 100, height: 21, rx: 7, fill: 'rgba(7,14,17,.82)', stroke: '#2e4851' }, tagG);
  const tag = el('text', { y: 4, 'text-anchor': 'middle', 'font-size': 12.5, fill: '#dce8ea', 'font-weight': 600 }, tagG);
  const away = el('g', { opacity: 0, class: 'fade' }, g);
  el('rect', { x: -48, y: -R - 34, width: 96, height: 22, rx: 7, fill: 'rgba(7,14,17,.88)', stroke: '#4a5f66' }, away);
  const awayT = el('text', { y: -R - 18, 'text-anchor': 'middle', 'font-size': 12.5, fill: '#ffd6a0', 'font-weight': 700, 'letter-spacing': '.04em' }, away);
  return { g, ring, body, lvt, tag, tagBg, away, awayT, R };
}

function packetGlyph(seat) {
  const g = el('g', { class: 'mv' }, $('packet-layer'));
  const shape = el('path', {}, g);
  const shine = el('path', { fill: '#fff', opacity: .12 }, g);
  const txt = el('text', { 'text-anchor': 'middle', 'font-size': 17, 'font-weight': 700, y: 6, fill: '#fff' }, g);
  const sub = el('text', { 'text-anchor': 'middle', 'font-size': 9.5, y: -21, fill: '#cfe0e0', opacity: .85, 'letter-spacing': '.12em' }, g);
  return { g, shape, shine, txt, sub, seat, opacity: 1 };
}
function packetShape(p, hp, front) {
  const w = 30 + hp * 2.4, hh = 12 + Math.min(hp, 24) * .22;     // size grows with HP (never individual minions)
  const dir = p.seat === 'A' ? 1 : -1;
  const tip = front ? 14 : 4;
  const x0 = -w / 2, x1 = w / 2;
  const d = dir === 1
    ? `M${x0} ${-hh} L${x1} ${-hh} L${x1 + tip} 0 L${x1} ${hh} L${x0} ${hh} Q${x0 - 6} 0 ${x0} ${-hh} Z`
    : `M${x1} ${-hh} L${x0} ${-hh} L${x0 - tip} 0 L${x0} ${hh} L${x1} ${hh} Q${x1 + 6} 0 ${x1} ${-hh} Z`;
  p.shape.setAttribute('d', d);
  p.shape.setAttribute('fill', p.seat === 'A' ? '#13675f' : '#8a3615');
  p.shape.setAttribute('stroke', p.seat === 'A' ? '#5ff0de' : '#ffa06c');
  p.shape.setAttribute('stroke-width', front ? 2.4 : 1.4);
  p.shine.setAttribute('d', `M${x0 + 4} ${-hh + 3} L${x1 - 4} ${-hh + 3} L${x1 - 8} ${-hh + 8} L${x0 + 6} ${-hh + 8} Z`);
  p.txt.textContent = hp;
  p.sub.textContent = front ? 'FRONT' : '';
  p.sub.setAttribute('y', p.seat === 'A' ? -hh - 6 : hh + 13);
}

function syncPackets(seat, list) {
  // match old glyphs to new packets: deaths and crashes remove the front, spawns add at the rear
  const old = S.pk[seat];
  while (old.length > list.length) {
    const gone = old.shift();
    gone.g.style.opacity = 0;
    setTimeout(() => gone.g.remove(), 400);
  }
  while (old.length < list.length) {
    const p = packetGlyph(seat);
    const [x] = list[old.length];
    p.g.style.transition = 'none';
    p.g.style.opacity = 0;
    p.g.style.transform = `translate(${X(x)}px, ${PK_Y[seat]}px)`;
    p.g.getBoundingClientRect();
    p.g.style.transition = '';
    old.push(p);
  }
  // packets of one side in the same cell are offset so both stay readable
  const seen = {};
  list.forEach(([x, hp, front], i) => {
    const p = old[i];
    const k = seen[x] = (seen[x] || 0) + 1;
    const dx = (k - 1) * (seat === 'A' ? -22 : 22);
    packetShape(p, hp, front);
    p.g.style.opacity = front ? 1 : .78;
    p.g.style.transform = `translate(${X(x) + dx}px, ${PK_Y[seat] + (k - 1) * 6}px)`;
    p.g.setAttribute('aria-label', `${seat}'s minions, ${hp} HP${front ? ', front' : ''}`);
  });
}

function drawSiege(lane, tick) {
  const layer = $('siege-layer');
  layer.innerHTML = '';
  for (const seat of ['A', 'B']) {
    const [x, hp] = lane.siege[seat];
    if (!hp) continue;
    const g = el('g', { transform: `translate(${X(x) + (seat === 'A' ? -62 : 62)}, ${ROAD_Y + 40})` }, layer);
    const w = 26 + hp * 2;
    el('path', { d: `M${-w / 2} 10 L${-w / 2 + 6} -12 L${-w / 4} -4 L0 -16 L${w / 4} -5 L${w / 2 - 5} -13 L${w / 2} 10 Z`,
      fill: `url(#p-hatch-${seat.toLowerCase()})`, stroke: seat === 'A' ? '#6ff3e2' : '#ffb27f', 'stroke-width': 2 }, g);
    const t = el('text', { y: 33, 'text-anchor': 'middle', 'font-size': 12.5, 'font-weight': 700, fill: seat === 'A' ? '#aef7ee' : '#ffd3b5' }, g);
    t.textContent = `CRASHED ${hp} HP`;
    const t2 = el('text', { y: 4, 'text-anchor': 'middle', 'font-size': 14, 'font-weight': 700, fill: '#fff' }, g);
    t2.textContent = hp;
  }
  if (tick) {
    for (const seat of ['A', 'B']) {
      if (tick.crashed[seat]) {
        const x = seat === 'A' ? X(2) : X(-2);
        floater(x, ROAD_Y - 4, `crash ${tick.crashed[seat]} HP`, seat === 'A' ? '#8ff5e8' : '#ffc29b');
      }
    }
  }
}

function drawContact(lane) {
  const layer = $('contact-layer');
  layer.innerHTML = '';
  if (lane.contact === null || lane.contact === undefined) return;
  const g = el('g', { transform: `translate(${X(lane.contact)}, ${ROAD_Y})` }, layer);
  el('line', { x1: 0, x2: 0, y1: -40, y2: 40, stroke: '#e7c46a', 'stroke-width': 2, 'stroke-dasharray': '3 4', opacity: .55 }, g);
  el('circle', { r: 11, fill: '#1e1a12', stroke: '#e7c46a', 'stroke-width': 2 }, g);
  el('path', { d: 'M-5 -5 L5 5 M5 -5 L-5 5', stroke: '#ffe39a', 'stroke-width': 2.2, 'stroke-linecap': 'round' }, g);
}

function drawTowers(towers) {
  for (const seat of ['A', 'B']) {
    const tw = towers[seat];
    const c = seat === 'A' ? -2 : 2;
    const g = $(`tower-dyn-${seat}`);
    g.innerHTML = '';
    const x = X(c);
    // plates: three pips on the tower body; a lost plate is cracked
    for (let i = 0; i < tw.plates_total; i++) {
      const lost = i < tw.plates_lost;
      const py = TOWER_Y + 17 - i * 13;
      const pg = el('g', { transform: `translate(${x}, ${py})` }, g);
      el('rect', { x: -13, y: -4.5, width: 26, height: 9, rx: 2, fill: lost ? '#2a2f30' : (seat === 'A' ? '#3ed6c4' : '#ff8b4f'), stroke: lost ? '#555' : '#0c1416', 'stroke-width': 1.3, opacity: lost ? .9 : .95 }, pg);
      if (lost) el('path', { d: 'M-8 -4 l4 4 l-3 2 l5 2.5 M5 -4 l-2 4 l4 2', stroke: '#9aa', 'stroke-width': 1.2, fill: 'none' }, pg);
    }
    const bw = 84, frac = Math.min(1, tw.damage / tw.damage_max);
    el('rect', { x: x - bw / 2, y: TOWER_Y + 46, width: bw, height: 5, rx: 2.5, fill: '#1b2427' }, g);
    if (frac > 0) el('rect', { x: x - bw / 2, y: TOWER_Y + 46, width: bw * frac, height: 5, rx: 2.5, fill: '#ff5d6c' }, g);
    const t = el('text', { x, y: TOWER_Y + 66, 'text-anchor': 'middle', 'font-size': 12, fill: '#a9bbb8', 'letter-spacing': '.06em' }, g);
    t.textContent = `${seat}'S TOWER · ${tw.plates_total - tw.plates_lost}/${tw.plates_total} PLATES${tw.damage ? ` · DMG ${tw.damage}` : ''}`;
  }
}

function champPlacement(frame) {
  // a present champion stands on its cell; at W on the Vision node; an absent one waits at its base
  const out = {};
  for (const seat of ['A', 'B']) {
    const st = frame.status[seat];
    const present = frame.sub === 'tick' ? frame.cells[seat] !== null || frame.at_ward[seat] : st.status === 'present';
    let x, y, where;
    if (frame.at_ward[seat] && present) { x = X(0); y = W_Y + 4; where = 'w'; }
    else if (present && frame.cells[seat] !== null) { x = X(frame.cells[seat]); y = CH_Y; where = 'lane'; }
    else { x = X(seat === 'A' ? -4 : 4); y = CH_Y - 6; where = 'base'; }
    out[seat] = { x, y, where, present };
  }
  if (out.A.where === out.B.where && Math.abs(out.A.x - out.B.x) < 110) {
    const mid = (out.A.x + out.B.x) / 2, sa = out.A.x <= out.B.x ? -1 : 1;
    out.A.x = mid + sa * 54; out.B.x = mid - sa * 54;
  }
  return out;
}

function drawChamps(frame, view) {
  const place = champPlacement(frame);
  for (const seat of ['A', 'B']) {
    const tk = S.champ[seat], st = frame.status[seat], pl = place[seat];
    tk.g.style.transform = `translate(${pl.x}px, ${pl.y}px)`;
    const frac = Math.max(0, st.hp / st.hp_max);
    const circ = 2 * Math.PI * (tk.R + 7);
    tk.ring.setAttribute('stroke-dasharray', `${(circ * frac).toFixed(1)} ${circ.toFixed(1)}`);
    tk.ring.setAttribute('stroke', frac < .35 ? '#ff5d6c' : (frac < .65 ? '#e7c46a' : '#62de8b'));
    tk.lvt.textContent = st.level;
    tk.tag.textContent = `${C.hpText(st.hp)} HP · ${st.gold_earned.toFixed(1)}g`;
    const away = st.status !== 'present' && frame.sub !== 'tick' || (frame.sub === 'tick' && !pl.present);
    tk.g.style.opacity = away ? .5 : 1;
    tk.g.style.filter = away ? 'grayscale(.85)' : '';
    tk.away.setAttribute('opacity', away ? 1 : 0);
    if (away) tk.awayT.textContent = st.status === 'dead' ? `✕ DEAD · BACK IN ${st.returns_in}` : (st.status === 'recall' ? `⟲ RECALL · BACK IN ${st.returns_in}` : 'AWAY');
    tk.g.setAttribute('aria-label', `${S.names[seat]}: level ${st.level}, ${st.hp} of ${st.hp_max} HP`);
  }
  // a placed ward glows at W while the seat's vision lasts
  const wm = $('ward-marks');
  wm.innerHTML = '';
  ['A', 'B'].forEach((seat, i) => {
    const v = frame.status[seat].vision;
    if (!v) return;
    const g = el('g', { transform: `translate(${X(0) + (i ? 54 : -54)}, ${W_Y - 30})` }, wm);
    el('circle', { r: 14, fill: seat === 'A' ? '#3ed6c4' : '#ff8b4f', opacity: .25, filter: 'url(#f-glow)' }, g);
    el('path', { d: 'M0 -11 L6 0 L0 11 L-6 0 Z', fill: seat === 'A' ? '#3ed6c4' : '#ff8b4f', stroke: '#fff', 'stroke-width': 1 }, g);
    const t = el('text', { y: 26, 'text-anchor': 'middle', 'font-size': 11, fill: '#dce8ea', 'font-weight': 600 }, g);
    t.textContent = `${seat} ward · ${v}`;
  });
}

function floater(x, y, text, color) {
  const g = el('g', { transform: `translate(${x}, ${y})` }, $('float-layer'));
  const inner = el('g', { class: 'floater' }, g);
  el('rect', { x: -text.length * 3.9 - 8, y: -13, width: text.length * 7.8 + 16, height: 22, rx: 7, fill: 'rgba(7,14,17,.85)', stroke: color }, inner);
  const t = el('text', { y: 3, 'text-anchor': 'middle', 'font-size': 13, 'font-weight': 700, fill: color }, inner);
  t.textContent = text;
  const keep = P.still || C.prefersReducedMotion();
  if (!keep) setTimeout(() => g.remove(), 950);
  else inner.classList.remove('floater');
}

function drawTickFloaters(frame) {
  $('float-layer').innerHTML = '';
  if (frame.sub !== 'tick' || !frame.tick) return;
  const place = champPlacement(frame);
  for (const seat of ['A', 'B']) {
    const tk = frame.tick;
    if (tk.wave_removed[seat] > 0) floater(place[seat].x, place[seat].y - 50, `−${tk.wave_removed[seat]} minion HP`, seat === 'A' ? '#8ff5e8' : '#ffc29b');
    if (tk.minion_fire[seat].units > 0) floater(place[seat].x, place[seat].y + 66, `hit by minions ×${tk.minion_fire[seat].units}`, '#ff8f9a');
  }
}

// ------------------------------------------------------------------------------------------------ HUD, ribbon, banner
function cellWords(seat, st) {
  if (st.status === 'dead') return `<span class="warn">Dead</span> · back in ${st.returns_in}`;
  if (st.status === 'recall') return `<span class="rec">Recalling</span> · back in ${st.returns_in}`;
  if (st.at_ward) return 'At the Vision node W';
  const own = seat === 'A' ? st.cell : -st.cell;
  return { '-2': 'At own tower', '-1': 'Own side of the lane', 0: 'At the lane (midpoint)', 1: 'Enemy side of the lane', 2: 'At the enemy tower' }[own] || `cell ${st.cell}`;
}
function hudName(seat) {
  const x = S.game.meta.seats['AB'.indexOf(seat)];
  if (S.mode === 'play' && seat === 'A') return 'You';
  return x.kind === 'model' ? x.label : `Scripted ${x.id}`;
}
function hudId(seat) {
  const x = S.game.meta.seats['AB'.indexOf(seat)];
  return x.kind === 'model' && !(S.mode === 'play' && seat === 'A') ? `<span class="idchip">${esc(x.id)}</span>` : '';
}
function drawHud(frame) {
  for (const seat of ['A', 'B']) {
    const st = frame.status[seat];
    const hpf = st.hp / st.hp_max;
    const xpf = st.xp_next ? (st.xp - st.xp_level) / (st.xp_next - st.xp_level) : 1;
    $(`hud-${seat}`).innerHTML = `
      <div class="hud-head">${glyphSvg(seat)}<span class="hud-name" title="${esc(S.names[seat])}">${esc(hudName(seat))}</span>${hudId(seat)}<span class="lvl">Lv ${st.level}</span></div>
      <div class="bar hp ${hpf < .35 ? 'low' : ''}" title="HP"><i style="width:${(hpf * 100).toFixed(1)}%"></i></div>
      <div class="hud-row"><span>HP <b>${C.hpText(st.hp)}</b> / ${st.hp_max}</span><span>XP ${st.xp}${st.xp_next ? ' / ' + st.xp_next : ''}</span></div>
      <div class="bar xp" title="XP to the next level"><i style="width:${(xpf * 100).toFixed(1)}%"></i></div>
      <div class="hud-row"><span>Earned <b>${st.gold_earned.toFixed(2)}</b> g</span><span>unspent ${st.gold_unspent.toFixed(1)} · in items ${st.invested.toFixed(1)}</span></div>
      <div class="hud-status">${cellWords(seat, st)}${st.vision ? ` · vision ${st.vision}` : ''}</div>`;
  }
}

const OUT_ICON = { death: '✕', trade: '⚔', recall: '⟲', recall_interrupted: '⟲', ward: '◆', ward_interrupted: '◇', plate: '▣', crash: '▲',
  tower_shot: '✦', gank: '!', return: '↩', level_up: '↑', invest: '$', minion_fire: '•', tower_damage: '▪', wave_lost: '·', merge: '+' };
function outChip(e) {
  const cls = e.kind === 'death' ? 'death' : (e.kind === 'trade' && e.data && e.data.result === 'hit' ? 'hit' : (e.kind.startsWith('recall') || e.kind === 'return' ? 'recall' : ''));
  return `<span class="out-chip ${cls}"><span class="ic" aria-hidden="true">${OUT_ICON[e.kind] || '•'}</span>${esc(fill(e.text))}</span>`;
}
function drawRibbon(frame) {
  const r = $('ribbon');
  const g = S.game;
  const t = frame.turn || (frame.p > 0 ? frame.p : 0);
  if (!t) {
    r.innerHTML = `<span class="ribbon-turn">Initial position</span><span class="ribbon-outs"><span class="out-chip">Both champions at their own tower, a fresh wave at each base. No turn resolved yet.</span></span>`;
    return;
  }
  const tr = g.turns[t - 1];
  const acts = `<span class="ribbon-acts"><span class="act-pill a">${esc(S.short.A)} · ${esc(tr.names[0])}</span><span class="act-pill b">${esc(S.short.B)} · ${esc(tr.names[1])}</span></span>`;
  if (frame.sub === 'start' || frame.sub === 'tick') {
    r.innerHTML = `<span class="ribbon-turn">Turn ${t} · resolving</span>${acts}<span class="ribbon-outs"><span class="out-chip">${frame.sub === 'tick' ? `tick ${frame.tick.tick} of ${tr.ticks.length}: minions move and fight` : 'both actions revealed'}</span></span>`;
    return;
  }
  const major = tr.outcomes.filter((e) => e.major);
  const minor = tr.outcomes.length - major.length;
  r.innerHTML = `<span class="ribbon-turn">Turn ${t}</span>${acts}<span class="ribbon-outs">${major.length ? major.map(outChip).join('') : '<span class="out-chip">Quiet turn: farming only</span>'}</span>` +
    `<button class="ribbon-more" data-open="events">${minor ? `+${minor} minor · ` : ''}all events</button>`;
  r.querySelector('.ribbon-more').onclick = () => openDrawer('events');
}

function drawFrameTag(frame) {
  const tag = $('frame-tag');
  if (S.ctrl.historical && S.ctrl.mode === 'play' && !S.ctrl.stopped) { tag.hidden = true; return; }
  tag.hidden = false;
  tag.textContent = frame.sub === 'tick' ? `Turn ${frame.turn} · tick ${frame.tick.tick} of ${S.game.turns[frame.turn - 1].ticks.length}`
    : (frame.sub === 'start' ? `Turn ${frame.turn} · actions revealed` : (frame.p === 0 ? 'Initial position' : `After turn ${frame.p}`));
}

function drawBanner() {
  const c = S.ctrl, b = $('hist-banner');
  const show = c.historical && c.mode === 'play' && !c.stopped;     // Watch and review have no decision point
  b.hidden = !show;
  $('lane').parentElement.classList.toggle('historical', show);
  if (show) {
    $('hist-text').textContent = c.mode === 'play' && !c.stopped
      ? `Viewing ${c.view === 0 ? 'the initial position' : `turn ${c.view}`} — not the current turn (${c.progress + 1})`
      : `Viewing ${c.view === 0 ? 'the initial position' : `turn ${c.view}`} of ${c.progress}`;
    $('return-btn').textContent = c.mode === 'play' && !c.stopped ? 'Return to current turn' : 'Jump to the latest';
  }
}

function srSummary(frame) {
  const parts = [`${frame.p === 0 ? 'Initial position' : `Position after turn ${frame.p}`}.`];
  for (const seat of ['A', 'B']) {
    const st = frame.status[seat];
    parts.push(`${S.names[seat]}: level ${st.level}, ${st.hp} of ${st.hp_max} HP, ${cellWords(seat, st).replace(/<[^>]+>/g, '')}, earned ${st.gold_earned.toFixed(2)} gold.`);
  }
  const pos = S.game.positions[frame.p];
  parts.push(`Wave: ${pos.wave_status}. Earned-gold lead A minus B: ${C.gold(pos.lead_A)}.`);
  $('sr-summary').textContent = parts.join(' ');
}

// ------------------------------------------------------------------------------------------------ the command bar
const WAVE_METER = { 0: 1, 2: 2, 3: 3, 1: 4 };
const ICONS = {
  recall: '<svg viewBox="0 0 32 32" aria-hidden="true"><circle cx="16" cy="16" r="12" fill="none" stroke="#8fb7ff" stroke-width="2.5" stroke-dasharray="58 18"/><path d="M16 9v8l5 3" stroke="#cfe0ff" stroke-width="2.5" fill="none" stroke-linecap="round"/></svg>',
  ward: '<svg viewBox="0 0 32 32" aria-hidden="true"><path d="M16 4 L24 16 L16 28 L8 16 Z" fill="#2f6a3a" stroke="#9fe0a5" stroke-width="2"/><circle cx="16" cy="16" r="3.5" fill="#d8ffd9"/></svg>',
};
function selectedCode() {
  const s = S.sel, st = S.game.menu_static;
  if (s.single !== null) return s.single;
  if (s.wave === null || s.beh === null) return null;
  return st.lane_codes[`${s.wave},${s.beh},${s.obj}`];
}
function codeName(code) {
  const st = S.game.menu_static;
  if (code === 12) return 'Recall';
  if (code === 30) return 'Ward';
  if (code === 45) return 'Absent';
  for (const [k, v] of Object.entries(st.lane_codes)) if (v === code) {
    const [w, b, o] = k.split(',').map(Number);
    return `${st.waves.find((x) => x.id === w).name} + ${st.behaviours.find((x) => x.id === b).name}${o ? ' + Pressure' : ''}`;
  }
  return `code ${code}`;
}
function drawCommand() {
  const c = S.ctrl, box = $('command');
  box.innerHTML = '';
  const info = h('div', 'cmd-info');
  box.appendChild(info);
  if (S.mode === 'watch' || c.stopped && S.mode !== 'play') return drawTransport(box, info);
  if (c.stopped) {
    info.innerHTML = `<span class="tag">Stopped</span><span>The session stopped after turn ${c.progress}. Review it with the turn list, autoplay or the arrow keys.</span>`;
    const body = h('div', 'forced');
    body.innerHTML = `<svg class="big-glyph" viewBox="0 0 64 64" aria-hidden="true"><rect x="14" y="14" width="36" height="36" rx="6" fill="#16262d" stroke="#e7c46a" stroke-width="3"/></svg>
      <div><h3>Session stopped · no winner declared</h3><p>Stopping is not an ending of the game. The summary describes the state at the stop; the record replays bit for bit in the real app.</p></div>`;
    const b = h('button', 'btn gold continue', 'Open the summary');
    b.onclick = () => { S.summaryOpen = true; render(); };
    body.appendChild(b);
    box.appendChild(body);
    return;
  }
  // Play
  const ls = c.lastSubmission;
  const staged = `<span class="tag staged" tabindex="0" title="${esc(C.STAGED_NOTE)}" aria-label="${esc(C.STAGED_NOTE)}">Staged prototype</span>`;
  if (ls) {
    const tr = S.game.turns[ls.t - 1];
    info.innerHTML = `${staged}<span class="reveal">Turn ${ls.t} revealed: <b>You</b> ${esc(tr.names[0])} · <b>${esc(S.short.B)}</b> ${esc(tr.names[1])}</span>` +
      (ls.matched ? '<span class="match ok">✓ your choice matched the recording</span>'
        : `<span class="match no">⚠ you chose ${esc(codeName(ls.selected))}; the recorded ${esc(codeName(ls.recorded))} resolved</span>`);
  } else {
    info.innerHTML = `${staged}<span>Both champions choose at the same time. The opponent has already chosen; its action is revealed with yours.</span>`;
  }
  const m = c.menu();
  if (c.forced() || (c.historical && m && m.forced !== null)) {
    const run = c.awayRun();
    const body = h('div', 'forced');
    const dead = m.away && m.away.status === 'dead';
    body.innerHTML = `<svg class="big-glyph" viewBox="0 0 64 64" aria-hidden="true">${dead
      ? '<circle cx="32" cy="32" r="26" fill="#2a1418" stroke="#ff5d6c" stroke-width="3"/><path d="M22 22l20 20M42 22L22 42" stroke="#ff8f9a" stroke-width="4" stroke-linecap="round"/>'
      : '<circle cx="32" cy="32" r="26" fill="#15213a" stroke="#8fb7ff" stroke-width="3"/><path d="M32 18v15l9 6" stroke="#cfe0ff" stroke-width="4" fill="none" stroke-linecap="round"/>'}</svg>
      <div><h3>${dead ? 'You died' : 'You are recalling'} · back in ${m.away ? m.away.returns_in : run} turn${(m.away ? m.away.returns_in : run) === 1 ? '' : 's'}</h3>
      <p>${esc(m.reasons.absent || 'Absent is the only action until your champion returns.')} Each away turn still resolves on its own, with the opponent's action fixed first.</p></div>`;
    const b = h('button', 'btn gold continue', `Continue (away ${run} turn${run === 1 ? '' : 's'})`);
    b.onclick = () => c.continueAway();
    b.disabled = !c.canSubmit;
    b.setAttribute('aria-keyshortcuts', 'Enter');
    body.appendChild(b);
    box.appendChild(body);
    if (c.historical) addVeil(box);
    return;
  }
  const st = S.game.menu_static;
  const reasons = m ? m.reasons : {};
  const legal = new Set(m ? m.legal : []);
  const body = h('div', 'cmd-body');
  const rows = h('div', 'rows');
  // wave row
  const wrow = h('div', 'row');
  wrow.innerHTML = '<div class="row-k">Wave<small>least to most damage</small></div>';
  const wopts = h('div', 'opts');
  st.waves.forEach((w, i) => {
    const b = h('button', 'opt');
    b.setAttribute('aria-pressed', S.sel.wave === w.id && S.sel.single === null);
    b.title = w.help;
    b.setAttribute('aria-keyshortcuts', String(i + 1));
    b.innerHTML = `<span class="meter" aria-hidden="true">${[1, 2, 3, 4].map((k) => `<i class="${k <= WAVE_METER[w.id] ? 'on' : ''}" style="height:${4 + k * 3}px"></i>`).join('')}</span><span class="lbl">${w.name}</span><span class="tick" aria-hidden="true"></span>`;
    b.onclick = () => { S.sel.wave = w.id; S.sel.single = null; drawCommand(); };
    wopts.appendChild(b);
  });
  wrow.appendChild(wopts);
  // behaviour row
  const brow = h('div', 'row');
  brow.innerHTML = '<div class="row-k">Behaviour<small>where you stand</small></div>';
  const bopts = h('div', 'opts');
  st.behaviours.forEach((bh, i) => {
    const b = h('button', 'opt');
    const blocked = S.sel.obj === 1 && !(st.objectives[1].with || []).includes(bh.name);
    b.disabled = blocked;
    b.setAttribute('aria-pressed', S.sel.beh === bh.id && S.sel.single === null);
    b.title = blocked ? 'Pressure combines with Hold or Trade only.' : bh.help;
    b.setAttribute('aria-label', `${bh.name}${blocked ? ', unavailable: Pressure combines with Hold or Trade only' : ''}`);
    b.setAttribute('aria-keyshortcuts', String(i + 5));
    b.innerHTML = `<span class="lbl">${bh.name}${blocked ? '<br><span class="why">needs Hold or Trade with Pressure</span>' : ''}</span><span class="tick" aria-hidden="true"></span>`;
    b.onclick = () => { S.sel.beh = bh.id; S.sel.single = null; drawCommand(); };
    bopts.appendChild(b);
  });
  brow.appendChild(bopts);
  // objective row
  const orow = h('div', 'row obj-row');
  orow.innerHTML = '<div class="row-k">Objective<small>optional</small></div>';
  const oopts = h('div', 'opts');
  const pb = h('button', 'opt');
  const pr = reasons.pressure;
  pb.disabled = !!pr;
  pb.setAttribute('aria-pressed', S.sel.obj === 1 && S.sel.single === null);
  pb.title = pr || st.objectives[1].help;
  pb.setAttribute('aria-label', `Pressure${pr ? ', unavailable: ' + pr : ''}`);
  pb.setAttribute('aria-keyshortcuts', 'P');
  pb.innerHTML = '<span class="lbl">Pressure</span><span class="tick" aria-hidden="true"></span>';
  pb.onclick = () => {
    S.sel.obj = S.sel.obj ? 0 : 1; S.sel.single = null;
    if (S.sel.obj && S.sel.beh !== null && ![0, 2].includes(S.sel.beh)) S.sel.beh = null;
    drawCommand();
  };
  oopts.appendChild(pb);
  const why = h('div', 'obj-why');
  why.innerHTML = pr ? `<span aria-hidden="true">ⓘ</span><span>${esc(pr)}</span>` : `<span style="color:var(--muted)">${esc(st.objectives[1].help)} With Hold or Trade only.</span>`;
  oopts.appendChild(why);
  orow.appendChild(oopts);
  rows.append(wrow, brow, orow);
  // standalone actions and the submit
  const side = h('div', 'side');
  const singles = h('div', 'singles');
  for (const sg of st.singles) {
    const key = sg.code === 12 ? 'recall' : 'ward';
    const r = reasons[key];
    const b = h('button', 'single');
    b.disabled = !!r || !legal.has(sg.code);
    b.setAttribute('aria-pressed', S.sel.single === sg.code);
    b.title = r || sg.help;
    b.setAttribute('aria-label', `${sg.name}${r ? ', unavailable: ' + r : ''}`);
    b.setAttribute('aria-keyshortcuts', key === 'recall' ? 'R' : 'V');
    b.innerHTML = `${ICONS[key]}<b>${sg.name}</b><small>${esc(r ? shortReason(r) : (key === 'recall' ? 'Back to base: full HP, gold invested' : 'Vision at W: fewer ganks, 6 turns'))}</small>`;
    b.onclick = () => { S.sel.single = S.sel.single === sg.code ? null : sg.code; drawCommand(); };
    singles.appendChild(b);
  }
  const code = selectedCode();
  const valid = code !== null && code !== undefined && legal.has(code);
  const sub = h('button', 'submit');
  sub.disabled = !valid || !c.canSubmit;
  sub.setAttribute('aria-keyshortcuts', 'Enter');
  sub.innerHTML = `<span><span class="s-k">Lock in turn ${c.progress + 1}</span><span class="s-v">${valid ? esc(codeName(code)) : (S.sel.wave === null && S.sel.single === null ? 'Choose a wave action' : (S.sel.beh === null && S.sel.single === null ? 'Choose a behaviour' : 'Not available'))}</span></span><span class="kbd">Enter</span>`;
  sub.onclick = () => submit();
  const ready = h('div', 'opp-ready', `<svg class="lock" viewBox="0 0 16 16" aria-hidden="true"><rect x="3" y="7" width="10" height="7" rx="1.5" fill="currentColor"/><path d="M5 7V5a3 3 0 0 1 6 0v2" stroke="currentColor" stroke-width="1.6" fill="none"/></svg><span><b>${esc(S.short.B)}</b> has chosen; hidden until you lock in</span>`);
  side.append(singles, h('div', '', ''));
  side.lastChild.append(sub, ready);
  side.lastChild.style.display = 'grid'; side.lastChild.style.gap = '6px';
  body.append(rows, side);
  box.appendChild(body);
  if (c.historical) addVeil(box);
}
function shortReason(r) {
  if (/Vision node W/.test(r) && /Recall/.test(r)) return 'Not from the Vision node W';
  if (/Recall needs/.test(r)) return 'Only on your side of the lane';
  if (/steps from/.test(r)) return 'Too far from W this turn';
  return r;
}
function addVeil(box) {
  const v = h('div', 'paused-veil');
  v.innerHTML = `<div><p>Viewing history: action selection is paused.</p><p style="color:var(--muted);font-weight:500;font-size:13px;margin-top:4px">The game is waiting at turn ${S.ctrl.progress + 1}; browsing changes nothing.</p></div>`;
  const b = h('button', 'btn gold', 'Return to current turn');
  b.onclick = () => S.ctrl.returnToCurrent();
  v.firstChild.appendChild(b);
  b.style.marginTop = '10px';
  box.appendChild(v);
}
function submit() {
  const code = selectedCode();
  if (code === null || code === undefined) return;
  const r = S.ctrl.submit(code);
  if (r) S.sel = { wave: null, beh: null, obj: 0, single: null };
}

function drawTransport(box, info) {
  const c = S.ctrl, g = S.game;
  const kind = g.meta.provenance.kind === 'research_record' ? 'Verified M5 recording' : 'New simulation';
  info.innerHTML = `<span class="tag">${kind}</span><span>Two bots play; the game was simulated first and is played back here. ${c.view ? `Viewing turn ${c.view} of ${c.n}.` : 'At the initial position.'}</span>`;
  const t = h('div', 'transport');
  const big = h('button', 'bigplay');
  big.setAttribute('aria-label', c.playing ? 'Pause (Space)' : 'Play (Space)');
  big.innerHTML = c.playing ? '<svg viewBox="0 0 24 24"><rect x="5" y="4" width="5" height="16" rx="1.5" fill="currentColor"/><rect x="14" y="4" width="5" height="16" rx="1.5" fill="currentColor"/></svg>'
    : '<svg viewBox="0 0 24 24"><path d="M7 4l13 8-13 8z" fill="currentColor"/></svg>';
  big.onclick = () => c.toggle();
  const right = h('div', 'scrub');
  const range = h('input');
  range.type = 'range'; range.min = 0; range.max = c.progress; range.value = c.view; range.step = 1;
  range.setAttribute('aria-label', `Position, 0 to ${c.progress}`);
  range.oninput = () => c.goto(+range.value);
  const marks = h('div', 'marks');
  for (const [p, lab] of [[0, 'start'], [g.lane_exit_position, 'lane exit · 94'], [g.read_position, 'lead read · 114'], [c.progress, `${c.progress}`]]) {
    if (p > c.progress) continue;
    const s = h('span', '', lab);
    s.style.left = `calc(${(p / c.progress) * 100}% + ${8 - (p / c.progress) * 16}px)`;
    marks.appendChild(s);
  }
  const row = h('div', 't-row');
  const tr = c.view ? g.turns[c.view - 1] : null;
  row.innerHTML = `<span>${tr ? `<b style="color:var(--text)">Turn ${c.view}</b> · ${esc(S.short.A)} ${esc(tr.names[0])} · ${esc(S.short.B)} ${esc(tr.names[1])}` : 'Press play or use the arrow keys'}</span>`;
  row.appendChild(speedButtons());
  right.append(range, marks, row);
  t.append(big, right);
  box.appendChild(t);
}

function speedButtons() {
  const wrap = h('div', 'speeds');
  wrap.setAttribute('role', 'radiogroup');
  wrap.setAttribute('aria-label', 'Playback speed');
  for (const s of C.SPEEDS) {
    const b = h('button', '', `${s}×`);
    b.setAttribute('role', 'radio');
    b.setAttribute('aria-checked', S.ctrl.speed === s);
    b.onclick = () => S.ctrl.setSpeed(s);
    wrap.appendChild(b);
  }
  return wrap;
}

// ------------------------------------------------------------------------------------------------ history panel
function evIcons(tr) {
  const ks = new Set(tr.outcomes.filter((e) => e.major).map((e) => (e.kind === 'trade' ? `trade_${e.data.result}` : e.kind)));
  const order = [['death', '✕'], ['trade_hit', '⚔'], ['recall', '⟲'], ['plate', '▣'], ['ward', '◆'], ['crash', '▲']];
  return order.filter(([k]) => ks.has(k)).map(([, i]) => i).join('');
}
function buildHistory() {
  const c = S.ctrl, g = S.game, list = $('hist-list');
  list.innerHTML = '';
  const add = (t) => {
    const li = h('li', 'hrow');
    li.dataset.p = t;
    li.tabIndex = -1;
    if (t === 0) {
      li.innerHTML = '<span class="n">0</span><span class="ca" style="color:var(--muted)">Initial position</span><span></span><span></span>';
    } else {
      const tr = g.turns[t - 1];
      const ca = C.codeShort(tr.parts[0]), cb = C.codeShort(tr.parts[1]);
      li.innerHTML = `<span class="n">${t}</span><span class="ca ${ca === 'Away' ? 'away' : ''}" title="${esc(tr.names[0])}">${ca}</span><span class="cb ${cb === 'Away' ? 'away' : ''}" title="${esc(tr.names[1])}">${cb}</span><span class="ev" aria-hidden="true">${evIcons(tr)}</span>`;
      li.setAttribute('aria-label', `Turn ${t}: ${S.short.A} ${tr.names[0]}, ${S.short.B} ${tr.names[1]}`);
    }
    li.onclick = () => c.goto(t);
    list.appendChild(li);
    if (t === g.lane_exit_position && g.meta.marks.lane_exit) {
      const pnl = g.meta.marks.lane_exit.panel;
      const m = h('li', 'hmark', `<b>Lane exit · 94</b><span>lead ${C.gold(pnl.D0)} · Lv ${pnl.level_A}/${pnl.level_B}</span>`);
      m.title = 'Position 94 is the lane exit (about level 14). The lead is earned gold A − B; XP, HP and the wave are separate.';
      list.appendChild(m);
    }
    if (t === g.read_position && g.meta.marks.lead_114) {
      const m = h('li', 'hmark read', `<b>Lead after 114</b><span>${C.gold(g.meta.marks.lead_114.lead_A)} · one game</span>`);
      m.title = g.meta.marks.lead_114.note;
      list.appendChild(m);
    }
  };
  for (let t = 0; t <= c.progress; t++) add(t);
  if (S.mode === 'play' && !c.stopped && c.progress < c.n) {
    const li = h('li', 'hrow pending');
    li.dataset.p = c.progress + 1;
    li.innerHTML = `<span class="n">${c.progress + 1}</span><span class="ca">choosing…</span><span class="cb">ready ✓</span><span class="ev"></span>`;
    li.setAttribute('aria-label', `Turn ${c.progress + 1}: your decision; the opponent has chosen`);
    li.onclick = () => c.returnToCurrent();
    list.appendChild(li);
  }
  $('col-a').textContent = S.short.A;
  $('col-b').textContent = S.short.B;
}
function markHistory() {
  const c = S.ctrl, list = $('hist-list');
  let cur = null;
  for (const li of list.querySelectorAll('.hrow')) {
    const p = +li.dataset.p;
    if (li.classList.contains('pending')) {
      li.classList.toggle('live', c.atCurrent && !c.stopped);
      continue;
    }
    const isCur = p === c.view;
    li.setAttribute('aria-current', isCur ? 'true' : 'false');
    if (isCur) cur = li;
  }
  if (cur) {
    const top = cur.offsetTop - list.offsetTop, bot = top + cur.offsetHeight;
    if (top < list.scrollTop + 4) list.scrollTop = top - 30;
    else if (bot > list.scrollTop + list.clientHeight - 4) list.scrollTop = bot - list.clientHeight + 30;
  }
  const pb = $('play-btn');
  pb.innerHTML = c.playing ? '❚❚ Pause' : '▶ Autoplay';
  pb.disabled = !c.playing && c.view >= c.progress && S.mode === 'play';
  pb.setAttribute('aria-label', c.playing ? 'Pause autoplay (Space)' : 'Autoplay through the turns (Space)');
  $('nav-first').disabled = $('nav-prev').disabled = c.view <= 0;
  $('nav-next').disabled = $('nav-last').disabled = c.view >= c.progress;
  const sp = $('speeds');
  sp.innerHTML = '';
  sp.append(...speedButtons().childNodes);
}

// ------------------------------------------------------------------------------------------------ drawer
const TABS = [['events', 'Events'], ['wave', 'Wave'], ['ticks', 'Ticks'], ['belief', 'Belief'], ['exit', 'Lane exit']];
function openDrawer(tab) { S.drawerTab = tab || S.drawerTab || 'events'; renderDrawer(); }
function closeDrawer() { S.drawerTab = null; renderDrawer(); }
function renderDrawer() {
  const d = $('drawer');
  const open = !!S.drawerTab;
  d.hidden = !open;
  $('detail-btn').setAttribute('aria-expanded', open);
  if (!open) return;
  const tabs = $('tabs');
  tabs.innerHTML = '';
  for (const [k, lab] of TABS) {
    const b = h('button', '', lab);
    b.setAttribute('role', 'tab');
    b.setAttribute('aria-selected', S.drawerTab === k);
    b.onclick = () => openDrawer(k);
    tabs.appendChild(b);
  }
  const body = $('drawer-body');
  const c = S.ctrl, g = S.game, p = c.view;
  const tr = p > 0 ? g.turns[p - 1] : null;
  const pos = g.positions[p];
  const nm = S.short;
  let html = '';
  if (S.drawerTab === 'events') {
    if (!tr) html = '<p class="note">The initial position: no turn has resolved.</p>';
    else {
      html += `<h3>Turn ${p}</h3><p class="note">${esc(nm.A)}: ${esc(tr.names[0])} (${tr.modes[0]}) · ${esc(nm.B)}: ${esc(tr.names[1])} (${tr.modes[1]}). Outcomes in the resolver's order; tick facts are in the tick trace.</p>`;
      html += `<ul class="ev-list">${tr.outcomes.map((e) => `<li class="${e.major ? '' : 'minor'}"><span aria-hidden="true">${OUT_ICON[e.kind] || '•'}</span><span>${e.major ? '<b>' : ''}${esc(fill(e.text))}${e.major ? '</b>' : ''}</span></li>`).join('') || '<li><span></span><span>No events: both champions farmed.</span></li>'}</ul>`;
      html += `<h3>Changes this turn</h3><table class="tbl"><tr><th></th><th>gold</th><th>farm</th><th>XP</th><th>HP</th><th>HP taken</th></tr>${['A', 'B'].map((s) => `<tr><td>${esc(nm[s])}</td><td>${C.gold(tr.delta[s].gold)}</td><td>${tr.delta[s].farm.toFixed(2)}</td><td>+${tr.delta[s].xp}</td><td>${tr.delta[s].hp >= 0 ? '+' : ''}${tr.delta[s].hp}</td><td>${tr.delta[s].hp_taken}</td></tr>`).join('')}</table>`;
      html += `<p class="note">Reward to A this turn: ${C.gold(tr.reward_A)} gold (earned A − earned B).</p>`;
    }
  } else if (S.drawerTab === 'wave') {
    html += `<h3>Position ${p}: the wave</h3><p class="note">${esc(pos.wave_status)}.</p>`;
    html += `<table class="tbl"><tr><th>side</th><th>packets (cell: HP)</th><th>crashed</th></tr>${['A', 'B'].map((s) => `<tr><td>${esc(nm[s])}</td><td>${pos.lane.packets[s].map(([x, hp, f]) => `${x > 0 ? '+' : ''}${x}: ${hp}${f ? ' (front)' : ''}`).join(', ') || 'none'}</td><td>${pos.lane.siege[s][1]} HP</td></tr>`).join('')}</table>`;
    html += `<h3>Wave budgets from this position</h3><table class="tbl"><tr><th></th><th>capacity</th>${Object.keys(pos.champions.A.wave_budgets).map((k) => `<th>${k}</th>`).join('')}<th>attack power</th></tr>${['A', 'B'].map((s) => { const st = pos.champions[s]; return `<tr><td>${esc(nm[s])}</td><td>${st.capacity ?? '—'}</td>${Object.values(st.wave_budgets).map((v) => `<td>${v}</td>`).join('')}<td>${st.attack_power.toFixed(2)}</td></tr>`; }).join('')}</table>`;
    html += `<p class="note">HP of enemy minions each wave action would remove this turn (the engine's budget; a connected trader spends at most half its capacity).</p>`;
    html += `<h3>Towers</h3><table class="tbl"><tr><th></th><th>damage taken</th><th>plates lost</th></tr>${['A', 'B'].map((s) => `<tr><td>${s}'s tower</td><td>${pos.towers[s].damage} / ${pos.towers[s].damage_max}</td><td>${pos.towers[s].plates_lost} / ${pos.towers[s].plates_total}</td></tr>`).join('')}</table>`;
  } else if (S.drawerTab === 'ticks') {
    if (!tr) html = '<p class="note">The initial position has no ticks.</p>';
    else {
      html += `<h3>Turn ${p}: the authoritative tick trace</h3><p class="note">The engine records these facts per tick. Trades, tower shots, ganks, deaths and level-ups happen per turn and are listed under Events.</p>`;
      html += `<table class="tbl"><tr><th>tick</th><th>${esc(nm.A)} packets</th><th>${esc(nm.B)} packets</th><th>cells A/B</th><th>HP removed</th><th>minion hits</th></tr>${tr.ticks.map((tk) => `<tr><td>${tk.tick}</td><td>${tk.packets.A.map(([x, hp]) => `${x}:${hp}`).join(' ') || '—'}</td><td>${tk.packets.B.map(([x, hp]) => `${x}:${hp}`).join(' ') || '—'}</td><td>${tk.at_ward.A ? 'W' : (tk.cells.A ?? '—')}/${tk.at_ward.B ? 'W' : (tk.cells.B ?? '—')}</td><td>${tk.wave_removed.A}/${tk.wave_removed.B}</td><td>${tk.minion_fire.A.units}/${tk.minion_fire.B.units}</td></tr>`).join('')}</table>`;
      html += `<p class="note">Crashed into towers this turn: ${tr.ticks.map((tk) => `tick ${tk.tick} ${tk.crashed.A}/${tk.crashed.B}`).join(', ')} (A/B HP). Contact: ${tr.ticks.map((tk) => tk.contact ?? 'none').join(', ')}.</p>`;
    }
  } else if (S.drawerTab === 'belief') {
    const b = c.belief(p);
    if (!g.beliefs) html = '<p class="note">Neither seat in this game models its opponent.</p>';
    else if (!b) {
      html = `<div class="hidden-note"><b>The belief is hidden.</b><br>The belief mode was chosen at the start of this session (hidden by default). It becomes available after the session stops.</div><p class="note">${esc(C.BELIEF_NOTE)}</p>`;
    } else {
      for (const [seat, post] of Object.entries(b.bySeat)) {
        const other = seat === 'A' ? 'B' : 'A';
        const rows = b.types.map((t, i) => [t, post[i]]).sort((x, y) => y[1] - x[1]);
        html += `<h3>${esc(nm[seat])}'s belief about ${esc(nm[other])} · position ${p}</h3>`;
        html += rows.map(([t, v], i) => `<div class="belief-row ${i === 0 ? 'top' : ''}"><span>${i === 0 ? '<b>' : ''}${esc(C.BELIEF_LABELS[t] || t)}${i === 0 ? '</b>' : ''}</span><span class="bb"><i style="width:${(v * 100).toFixed(1)}%"></i></span><span>${(v * 100).toFixed(1)}%</span></div>`).join('');
      }
      html += `<p class="note">${esc(C.BELIEF_NOTE)}</p>`;
    }
  } else if (S.drawerTab === 'exit') {
    const mk = g.meta.marks;
    html += '<h3>Lane exit · position 94</h3>';
    if (c.progress >= g.lane_exit_position && mk.lane_exit) {
      const q = mk.lane_exit.panel;
      html += `<table class="tbl"><tr><th></th><th>${esc(nm.A)}</th><th>${esc(nm.B)}</th></tr>
        <tr><td>level</td><td>${q.level_A}</td><td>${q.level_B}</td></tr><tr><td>XP</td><td>${q.xp_A}</td><td>${q.xp_B}</td></tr>
        <tr><td>HP (share)</td><td>${(q.hp_A * 100).toFixed(0)}%</td><td>${(q.hp_B * 100).toFixed(0)}%</td></tr>
        <tr><td>earned gold</td><td>${q.earned_A.toFixed(2)}</td><td>${q.earned_B.toFixed(2)}</td></tr>
        <tr><td>turns since return</td><td>${q.since_return_A}</td><td>${q.since_return_B}</td></tr></table>
        <p class="note">Earned-gold lead A − B at the lane exit: ${C.gold(q.D0)}. XP, HP and the wave are shown beside it, never folded into it.</p>`;
    } else html += '<p class="note">Not reached yet in this game.</p>';
    html += '<h3>Lead after 114 turns</h3>';
    if (c.progress >= g.read_position && mk.lead_114) html += `<p><b>${C.gold(mk.lead_114.lead_A)}</b> (A − B)</p><p class="note">${esc(mk.lead_114.note)}</p>`;
    else html += '<p class="note">Not reached yet in this game.</p>';
  }
  body.innerHTML = html;
}

// ------------------------------------------------------------------------------------------------ summary, provenance
function renderSummary() {
  const card = $('summary'), c = S.ctrl;
  const show = S.mode === 'play' && c.stopped && S.summaryOpen;
  card.hidden = !show;
  $('lane').parentElement.classList.toggle('dimmed', show);
  if (!show) return;
  const pos = S.game.positions[c.progress];
  const reason = c.progress >= c.n ? 'the turn cap of this demo was reached' : 'you stopped';
  const col = (seat) => {
    const st = pos.champions[seat];
    return `<div class="sum-col"><h3>${glyphSvg(seat, 22)}${esc(S.names[seat])}</h3><dl>
      <dt>Level</dt><dd>${st.level}</dd><dt>XP</dt><dd>${st.xp}</dd><dt>HP</dt><dd>${st.hp} / ${st.hp_max}</dd>
      <dt>Status</dt><dd>${cellWords(seat, st).replace(/<[^>]+>/g, '')}</dd><dt>Earned gold</dt><dd>${st.gold_earned.toFixed(2)}</dd>
      <dt>Unspent / in items</dt><dd>${st.gold_unspent.toFixed(2)} / ${st.invested.toFixed(2)}</dd></dl></div>`;
  };
  const mk = S.game.meta.marks;
  card.innerHTML = `<h2 id="summary-title">Session stopped after turn ${c.progress}</h2>
    <p class="sub">Stopped because ${reason}. No winner is declared: stopping is not an ending of the game, and the earned-gold lead is not a complete score.</p>
    <div class="sum-grid">${col('A')}${col('B')}</div>
    <div class="sum-note"><b>The state at the stop.</b> Earned-gold lead A − B ${C.gold(pos.lead_A)}. Wave: ${esc(pos.wave_status)}.
    ${c.progress >= 94 && mk.lane_exit ? ` Lane exit (position 94): lead ${C.gold(mk.lane_exit.panel.D0)}.` : ' The game stopped before the lane exit at position 94.'}
    ${c.progress >= 114 && mk.lead_114 ? ` Lead after 114 turns: ${C.gold(mk.lead_114.lead_A)} (one game).` : ''}</div>
    <div class="sum-note">${esc(S.game.meta.provenance.note || '')} A visitor's game is one example and measures nothing about play against humans in general; M5 never evaluated play against humans.</div>
    <div class="sum-actions"><button class="btn gold" id="sum-review">Review the game</button><button class="btn" disabled title="The real app (M6b) offers the record download">Download record (M6b)</button></div>`;
  $('sum-review').onclick = () => { S.summaryOpen = false; c.goto(0); render(); };
}

function renderProv() {
  const g = S.game, chip = $('prov-chip');
  const research = g.meta.provenance.kind === 'research_record';
  chip.className = `chip prov ${research ? '' : 'sim'}`;
  chip.innerHTML = `<span class="dot" aria-hidden="true"></span>${research ? 'Verified M5 recording' : 'New simulation'}<span class="info">· an example ⓘ</span>`;
  const pop = $('prov-pop');
  const src = g.meta.provenance.source || {};
  pop.innerHTML = `<h3>${research ? 'A verified M5 recording' : 'A new simulation'}</h3>
    <p>${esc(g.meta.provenance.note || '')}</p>
    <p>One game is an example, not evidence for or against M5's aggregate findings, which rest on all six seeds and many games per comparison. Neither agent is shown to be better overall.</p>
    <div class="kv">${g.meta.seats.map((s) => `<span>Seat ${s.seat}</span><b>${esc(s.kind === 'model' ? `${s.label} (${s.id}), sampled · ${s.file_sha256.slice(0, 12)}…` : s.label + (s.rule ? ' — ' + s.rule : ''))}</b>`).join('')}
    <span>Seeds</span><b>chance ${g.meta.seeds.chance} · policy ${g.meta.seeds.policy} (${esc(g.meta.seeds.policy_rule)})</b>
    ${src.file ? `<span>Source</span><b>${esc(src.file)}</b><span>Selection</span><b>${esc((src.m5 && src.m5.rule) || '')}</b>` : ''}
    <span>Verified</span><b>${esc(JSON.stringify(g.verification))}</b></div>`;
  chip.onclick = () => { const open = pop.hidden; pop.hidden = !open; chip.setAttribute('aria-expanded', open); };
}

// ------------------------------------------------------------------------------------------------ render loop
function render() {
  const c = S.ctrl, f = c.frame;
  if (c.progress !== S.lastProgress) { buildHistory(); S.lastProgress = c.progress; }
  syncPackets('A', f.lane.packets.A);
  syncPackets('B', f.lane.packets.B);
  drawSiege(f.lane, f.sub === 'tick' ? f.tick : null);
  drawContact(f.lane);
  drawTowers(f.towers);
  drawChamps(f, c.view);
  drawTickFloaters(f);
  drawHud(f);
  drawRibbon(f);
  drawFrameTag(f);
  drawBanner();
  markHistory();
  if (c.view !== S.lastView || f.sub === 'end' || f.sub === 'position') drawCommand();
  if (c.view !== S.lastView) { srSummary(c.frame.sub === 'position' || c.frame.sub === 'end' ? c.frame : C.positionFrame(S.game, c.view)); }
  // turn-level outcomes appear as toasts when a turn finishes resolving live
  if (f.sub === 'end' && f.turn && f.turn !== S.toastedTurn && !c.historical) {
    S.toastedTurn = f.turn;
    toasts(S.game.turns[f.turn - 1]);
  }
  S.lastView = c.view;
  $('turn-stat').textContent = S.mode === 'play' && !c.stopped ? `${c.progress + 1} / ${c.n}` : `${c.view} / ${c.n}`;
  const lead = S.game.positions[f.sub === 'tick' || f.sub === 'start' ? f.p : f.p].lead_A;
  const ls = $('lead-stat');
  ls.textContent = C.gold(lead);
  ls.className = `stat-v ${lead > 0.005 ? 'pos' : (lead < -0.005 ? 'neg' : '')}`;
  for (const b of $('mode-seg').children) b.setAttribute('aria-checked', b.dataset.mode === S.mode);
  renderSummary();
  if (S.drawerTab) renderDrawer();
}

function toasts(tr) {
  if (P.still) return;
  const box = $('toasts');
  box.innerHTML = '';
  for (const e of tr.outcomes.filter((x) => x.major).slice(0, 3)) {
    const t = h('div', `toast ${e.kind === 'death' ? 'death' : ''}`, `${OUT_ICON[e.kind] || '•'} ${esc(fill(e.text))}`);
    box.appendChild(t);
    setTimeout(() => t.remove(), 2500);
  }
}

async function start(name, mode) {
  S.gameName = name;
  S.mode = mode;
  S.game = await C.loadGame(name);
  S.names = C.seatNames(S.game, S.mode);
  S.short = C.shortNames(S.game, S.mode);
  if (S.ctrl) S.ctrl.pause();
  const first = !S.ctrl;
  S.ctrl = new C.Controller(S.game, { mode: S.mode, progress: first ? P.progress : 0, view: first ? P.view : null,
    belief: P.belief, speed: P.speed, still: P.still });
  S.ctrl.addEventListener('change', render);
  S.sel = { wave: null, beh: null, obj: 0, single: null };
  S.lastProgress = -1; S.lastView = -1; S.toastedTurn = S.ctrl.view; S.summaryOpen = false;
  buildLane();
  renderProv();
  render();
  const q = new URLSearchParams(location.search);
  q.set('game', name); q.set('mode', mode);
  history.replaceState(null, '', `?${q}`);
}

async function main() {
  if (P.still) document.body.classList.add('still');
  S.index = await C.loadIndex();
  const pick = $('game-pick');
  for (const g of S.index.games) {
    const o = document.createElement('option');
    o.value = g.name;
    o.textContent = `${g.kind === 'research_record' ? 'Verified' : 'New sim'} · ${g.label}`;
    pick.appendChild(o);
  }
  pick.value = P.game;
  pick.onchange = () => start(pick.value, S.mode);
  for (const b of $('mode-seg').children) b.onclick = () => start(S.gameName, b.dataset.mode);
  $('nav-first').onclick = () => S.ctrl.first();
  $('nav-prev').onclick = () => S.ctrl.prev();
  $('nav-next').onclick = () => S.ctrl.next();
  $('nav-last').onclick = () => S.ctrl.latest();
  $('play-btn').onclick = () => S.ctrl.toggle();
  $('return-btn').onclick = () => S.ctrl.returnToCurrent();
  $('detail-btn').onclick = () => (S.drawerTab ? closeDrawer() : openDrawer('events'));
  $('drawer-close').onclick = () => closeDrawer();
  C.bindKeys({ prev: () => S.ctrl.prev(), next: () => S.ctrl.next(), first: () => S.ctrl.first(), latest: () => S.ctrl.latest(),
    returnToCurrent: () => (S.drawerTab ? closeDrawer() : S.ctrl.returnToCurrent()), toggle: () => S.ctrl.toggle(),
    get mode() { return S.ctrl.mode; }, get stopped() { return S.ctrl.stopped; } }, {
    Enter: () => {
      const a = document.activeElement;
      if (a && a.tagName === 'BUTTON' && !a.classList.contains('submit')) return null;   // let a focused button act
      if (S.ctrl.forced()) S.ctrl.continueAway(); else if (S.ctrl.canSubmit) submit(); else return null;
    },
    d: () => (S.drawerTab ? closeDrawer() : openDrawer('events')),
    p: () => { if (!S.ctrl.canSubmit) return null; document.querySelector('.obj-row .opt')?.click(); },
    r: () => { if (!S.ctrl.canSubmit) return null; document.querySelectorAll('.single')[0]?.click(); },
    v: () => { if (!S.ctrl.canSubmit) return null; document.querySelectorAll('.single')[1]?.click(); },
    ...Object.fromEntries([1, 2, 3, 4, 5, 6, 7, 8].map((k) => [String(k), () => {
      if (!S.ctrl.canSubmit) return null;
      const rows = document.querySelectorAll('.rows .row');
      const btn = rows[k <= 4 ? 0 : 1]?.querySelectorAll('.opt')[(k - 1) % 4];
      if (btn && !btn.disabled) btn.click();
    }])),
  });
  document.addEventListener('click', (e) => {
    const pop = $('prov-pop');
    if (!pop.hidden && !e.target.closest('#prov-pop, #prov-chip')) { pop.hidden = true; $('prov-chip').setAttribute('aria-expanded', 'false'); }
  });
  await start(P.game, S.mode);
  if (P.detail) openDrawer(P.detail);
  window.__app = {
    get ctrl() { return S.ctrl; },
    select(w, b, o = 0) { S.sel = { wave: w, beh: b, obj: o, single: null }; drawCommand(); },
    single(code) { S.sel = { wave: null, beh: null, obj: 0, single: code }; drawCommand(); },
    submitRecorded() { const t = S.ctrl.progress + 1; return S.ctrl.submit(S.game.turns[t - 1].codes[0]); },
    stop() { S.ctrl.stop(); S.summaryOpen = true; render(); },
    showFrame(t, i) { S.ctrl.view = t; S.ctrl.frame = C.turnSubframes(S.game, t)[i]; render(); },
    openDrawer, closeDrawer,
  };
  document.body.dataset.ready = '1';
}

// core.bindKeys calls the controller's own methods; it receives the controller-like object above so the drawer's
// Escape closes the drawer first.
main().catch((e) => { window.__errors.push(String(e && e.stack || e)); document.body.dataset.ready = '1'; });
