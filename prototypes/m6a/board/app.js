// Prototype B · Board. A schematic tactical lane, a chess.com-style move list and a compact action builder.
// Throwaway M6a prototype: every number comes from the exported frame files (../data), read through ../shared/core.js.
import * as core from '../shared/core.js';

const P = core.params();
const $ = (id) => document.getElementById(id);
const NS = 'http://www.w3.org/2000/svg';
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

let index, game, ctrl, names, shorts, T;
const sel = { wave: null, beh: null, obj: 0, single: null };
let selFor = -1;                         // the progress the selection belongs to
let detailOpen = !!P.detail, detailTab = P.detail || 'events';
let lastSeq = null;                      // [turn, sub-frame index] of the frame on screen
let listBuiltFor = -1, listStopped = null;
let notice = '';                         // a short builder notice ("Pressure reset: ...")

// ------------------------------------------------------------------------------------------- geometry (viewBox 1000 x 300)
const X0 = 23, CW = 106;
const cx = (x) => X0 + (x + 4) * CW + CW / 2;
const Y = { bandTop: 72, bandBot: 244, aRow: 98, aSiege: 118, ch: 158, bSiege: 198, bRow: 220, w: 31 };
const HPW = 3.4;                         // px per minion HP

// ------------------------------------------------------------------------------------------- text helpers
function fill(text) {
  let s = text;
  for (const n of ['A', 'B']) {
    const nm = shorts[n], you = nm === 'You';
    s = s.replace(new RegExp(`\\{${n}\\}'s`, 'g'), you ? 'your' : `${nm}'s`);
    if (you) s = s.replace(new RegExp(`\\{${n}\\} was\\b`, 'g'), 'you were').replace(new RegExp(`\\{${n}\\} is\\b`, 'g'), 'you are');
    s = s.replace(new RegExp(`\\{${n}\\}`, 'g'), you ? 'you' : nm);
  }
  return /^you/.test(s) ? s[0].toUpperCase() + s.slice(1) : s;
}
const g2 = (x) => (Math.round(x * 100) / 100).toFixed(2);
const signed = (x) => (x > 0.004 ? '+' : x < -0.004 ? '−' : '±') + g2(Math.abs(x));
const plural = (n, w) => `${n} ${w}${n === 1 ? '' : 's'}`;
const cellName = (x) => {
  const play = ctrl.mode === 'play';
  const m = { '-4': play ? 'Your base' : 'A base', '-2': play ? 'Your tower' : 'A tower', 0: 'Lane',
              2: play ? 'Enemy tower' : 'B tower', 4: play ? 'Enemy base' : 'B base' };
  return m[x] ?? '';
};
const cellWord = (x) => (x === null ? 'W' : x > 0 ? `+${x}` : x < 0 ? `−${-x}` : '0');

function glyph(n, size = 20) {
  return n === 'A'
    ? `<svg class="glyph A" viewBox="0 0 20 20" width="${size}" height="${size}" aria-hidden="true"><circle cx="10" cy="10" r="8.5"/><text x="10" y="10.5" text-anchor="middle" dominant-baseline="central" font-size="10" font-weight="700" fill="#fff" font-family="Inter, system-ui, sans-serif">A</text></svg>`
    : `<svg class="glyph B" viewBox="0 0 20 20" width="${size}" height="${size}" aria-hidden="true"><rect x="3.6" y="3.6" width="12.8" height="12.8" transform="rotate(45 10 10)"/><text x="10" y="10.5" text-anchor="middle" dominant-baseline="central" font-size="9.5" font-weight="700" fill="#fff" font-family="Inter, system-ui, sans-serif">B</text></svg>`;
}

// ------------------------------------------------------------------------------------------- boot
async function boot() {
  if (P.still) document.body.classList.add('still');
  if (P.theme) document.documentElement.dataset.theme = P.theme;
  index = await core.loadIndex();
  const known = index.games.map((g) => g.name);
  game = await core.loadGame(known.includes(P.game) ? P.game : known[0]);
  T = game.turns[0].ticks.length;
  const mode = ['play', 'watch', 'review'].includes(P.mode) ? P.mode : 'play';
  ctrl = new core.Controller(game, { mode, belief: P.belief, progress: P.progress ?? 0, view: P.view ?? undefined,
                                      speed: P.speed, still: P.still });
  names = core.seatNames(game, ctrl.mode);
  shorts = core.shortNames(game, ctrl.mode);
  document.documentElement.style.setProperty('--tween', `${Math.round(core.BASE_FRAME_MS * 0.68 / ctrl.speed)}ms`);
  buildChrome();
  buildBoardStatic();
  ctrl.addEventListener('change', render);
  core.bindKeys(ctrl, {
    Enter: () => (ctrl.canSubmit && !ctrl.forced() && currentCode() !== null ? submit() : null),
    1: () => pick('wave', 0), 2: () => pick('wave', 2), 3: () => pick('wave', 3), 4: () => pick('wave', 1),
    q: () => pick('beh', 0), w: () => pick('beh', 1), e: () => pick('beh', 2), r: () => pick('beh', 3),
    p: () => pick('obj', sel.obj === 1 ? 0 : 1),
    d: () => toggleDetail(),
  });
  render();
  if (document.fonts) document.fonts.ready.then(() => markList());
  const bw = document.querySelector('.boardWrap');
  if (bw.scrollWidth > bw.clientWidth) bw.scrollLeft = (bw.scrollWidth - bw.clientWidth) / 2;
  window.addEventListener('resize', () => markList());
  window.__app = { ctrl, pick, submit, openDetail, toggleDetail, select: pick, stop: () => ctrl.stop() };
  requestAnimationFrame(() => { document.body.dataset.ready = '1'; });
}

// ------------------------------------------------------------------------------------------- header, nav and fixed chrome
function buildChrome() {
  const picker = $('gamePicker');
  for (const [kind, title] of [['research_record', 'Verified M5 recordings'], ['new_simulation', 'New simulations']]) {
    const og = document.createElement('optgroup');
    og.label = title;
    for (const g of index.games.filter((x) => x.kind === kind)) {
      const o = document.createElement('option');
      o.value = g.name;
      o.textContent = g.label;
      if (g.name === game.name) o.selected = true;
      og.append(o);
    }
    if (og.children.length) picker.append(og);
  }
  picker.addEventListener('change', () => go({ game: picker.value, progress: '', view: '' }));
  for (const b of $('modeSwitch').querySelectorAll('button')) {
    const on = (b.dataset.mode === 'play') === (ctrl.mode === 'play');
    b.setAttribute('aria-checked', String(on));
    b.addEventListener('click', () => go({ mode: b.dataset.mode, progress: '', view: '' }));
  }
  const kind = game.meta.provenance.kind;
  const badge = $('provBadge');
  badge.className = `badge ${kind === 'new_simulation' ? 'sim' : ''}`;
  badge.innerHTML = `<span class="dot"></span>${ctrl.mode === 'play' ? 'Staged on a ' : ''}${kind === 'research_record' ? (ctrl.mode === 'play' ? 'verified M5 recording' : 'Verified M5 recording') : (ctrl.mode === 'play' ? 'new simulation' : 'New simulation')}`;
  const pv = game.meta.provenance, sd = game.meta.seeds;
  const seat = (s) => { const x = game.meta.seats['AB'.indexOf(s)]; return `${s}: ${x.kind === 'model' ? `${x.label} (${x.id})` : x.label}`; };
  $('aboutPop').innerHTML = `<h3>${esc(game.label)}</h3><p>${esc(pv.note)}</p>
    ${ctrl.mode === 'play' ? `<p><b>Prototype.</b> ${esc(core.STAGED_NOTE)}</p>` : ''}
    <dl><dt>Kind</dt><dd>${kind === 'research_record' ? 'Verified M5 recording (re-simulated, every state digest checked)' : 'New simulation by the session core'}</dd>
    <dt>Seats</dt><dd>${esc(seat('A'))}<br>${esc(seat('B'))}</dd>
    ${pv.source ? `<dt>Source</dt><dd class="mono">${esc(pv.source.file)}</dd><dt>SHA-256</dt><dd class="mono">${esc(pv.source.sha256.slice(0, 32))}…</dd>` : ''}
    <dt>Seeds</dt><dd class="mono">chance ${sd.chance} · policy ${sd.policy}</dd><dt>Seed rule</dt><dd>${esc(sd.policy_rule)}</dd>
    <dt>Turns</dt><dd>${game.turns.length} (cap ${game.meta.turn_cap})</dd></dl>`;
  const about = $('aboutBtn');
  about.addEventListener('click', () => {
    const open = $('aboutPop').hidden;
    $('aboutPop').hidden = !open;
    about.setAttribute('aria-expanded', String(open));
  });
  document.addEventListener('click', (e) => {
    if (!e.target.closest('.prov')) { $('aboutPop').hidden = true; about.setAttribute('aria-expanded', 'false'); }
  });
  $('colA').textContent = shorts.A;
  $('colB').textContent = shorts.B;
  $('bFirst').addEventListener('click', () => ctrl.first());
  $('bPrev').addEventListener('click', () => ctrl.prev());
  $('bNext').addEventListener('click', () => ctrl.next());
  $('bLast').addEventListener('click', () => ctrl.latest());
  $('bPlay').addEventListener('click', () => ctrl.toggle());
  const sp = $('speed');
  sp.innerHTML = `<span class="lbl">Speed</span>` + core.SPEEDS.map((s) => `<button role="radio" data-s="${s}">${s}×</button>`).join('');
  sp.querySelectorAll('button').forEach((b) => b.addEventListener('click', () => {
    ctrl.setSpeed(+b.dataset.s);
    document.documentElement.style.setProperty('--tween', `${Math.round(core.BASE_FRAME_MS * 0.68 / ctrl.speed)}ms`);
  }));
  $('detailBtn').addEventListener('click', () => toggleDetail());
  $('moves').addEventListener('click', (e) => {
    const b = e.target.closest('[data-p]');
    if (!b) return;
    const p = +b.dataset.p;
    if (p > ctrl.progress) ctrl.returnToCurrent(); else ctrl.goto(p);
  });
}

function go(over) {
  const q = new URLSearchParams(location.search);
  for (const [k, v] of Object.entries(over)) { if (v === '' || v === null) q.delete(k); else q.set(k, v); }
  location.search = q.toString();
}

// ------------------------------------------------------------------------------------------- the board: static layer
function el(tag, attrs = {}, parent = null, text = null) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text !== null) e.textContent = text;
  if (parent) parent.append(e);
  return e;
}

function buildBoardStatic() {
  const g = $('gGrid');
  g.innerHTML = '';
  for (let x = -4; x <= 4; x++) {
    const cls = Math.abs(x) === 4 ? 'base' : Math.abs(x) === 2 ? 'tower' : (x + 4) % 2 ? 'alt' : '';
    el('rect', { x: X0 + (x + 4) * CW, y: Y.bandTop, width: CW, height: Y.bandBot - Y.bandTop, class: `cellBg ${cls}` }, g);
    if (x > -4) el('line', { x1: X0 + (x + 4) * CW, x2: X0 + (x + 4) * CW, y1: Y.bandTop, y2: Y.bandBot, class: 'cellLine' }, g);
    const nm = cellName(x);
    el('text', { x: cx(x), y: 279, class: `cellLbl ${x === 0 || Math.abs(x) === 2 ? 'strong' : ''}` }, g, nm || '·');
    el('text', { x: cx(x), y: 295, class: 'cellNum' }, g, x > 0 ? `+${x}` : x < 0 ? `−${-x}` : '0');
  }
  el('rect', { x: X0, y: Y.bandTop, width: 9 * CW, height: Y.bandBot - Y.bandTop, rx: 6, class: 'laneEdge' }, g);
  // the Vision node W, off the lane, attached to the lane cell 0
  el('line', { x1: cx(0), x2: cx(0), y1: Y.w + 17, y2: Y.bandTop, class: 'wEdge' }, g);
  el('circle', { cx: cx(0), cy: Y.w, r: 17, class: 'wNode' }, g);
  el('text', { x: cx(0), y: Y.w, class: 'wTxt' }, g, 'W');
  el('text', { x: cx(0) + 26, y: Y.w + 4, class: 'wSub', 'text-anchor': 'start' }, g, 'Vision node');
}

// ------------------------------------------------------------------------------------------- the board: dynamic layers
const pkEls = { A: [], B: [] };
const chEls = {};

function drawTowers(frame) {
  const g = $('gStatic');
  g.innerHTML = '';
  for (const [n, x] of [['A', -2], ['B', 2]]) {
    const tw = frame.towers[n];
    const t = el('g', { class: `tower ${n}`, transform: `translate(${cx(x)}, 0)` }, g);
    el('path', { d: 'M-13,37 V11 H-8 V16 H-2.5 V11 H2.5 V16 H8 V11 H13 V37 Z', class: 'body', 'stroke-linejoin': 'round' }, t);
    el('rect', { x: -3, y: 26, width: 6, height: 11, rx: 3, class: 'crenel' }, t);
    const left = tw.plates_total - tw.plates_lost;
    for (let i = 0; i < tw.plates_total; i++) {
      el('circle', { cx: -12 + i * 12, cy: 46, r: 4, class: `plate ${i < left ? 'on' : 'off'}` }, t);
    }
    el('text', { x: 0, y: 63, class: 'towerTxt' }, t, `${tw.damage} / ${tw.damage_max}`);
    el('title', {}, t, `${n === 'A' ? names.A : names.B}'s tower: ${left} of ${tw.plates_total} plates left, damage ${tw.damage} of ${tw.damage_max}`);
  }
  // wards: each seat's own vision timer, drawn beside W
  for (const [n, dx, anchor] of [['A', -26, 'end'], ['B', 26, 'start']]) {
    const v = frame.status[n].vision;
    if (v > 0) el('text', { x: cx(0) + dx, y: Y.w - 8, class: 'wardPip', 'text-anchor': anchor, fill: `var(--${n.toLowerCase()})` }, g,
                  `${n === 'A' && shorts.A === 'You' ? 'your' : n} ward ${v}`);
  }
}

function drawContact(frame) {
  const g = $('gContact');
  g.innerHTML = '';
  const c = frame.lane.contact;
  if (c === null || c === undefined) return;
  const gx = cx(c);
  const k = el('g', { class: 'contact' }, g);
  el('rect', { x: gx - 7, y: Y.bandTop + 1, width: 14, height: Y.bandBot - Y.bandTop - 2, class: 'glow' }, k);
  el('line', { x1: gx, x2: gx, y1: Y.aRow + 9, y2: Y.ch - 34 }, k);
  el('line', { x1: gx, x2: gx, y1: Y.ch + 34, y2: Y.bRow - 9 }, k);
  el('rect', { x: gx - 28, y: Y.bandBot + 2, width: 56, height: 14, rx: 7, class: 'knot' }, k);
  el('text', { x: gx, y: Y.bandBot + 12.5 }, k, 'CONTACT');
  el('title', {}, k, `The two fronts are in contact at ${c}: their minions fight`);
}

function packetGroup(n, pk, y) {
  const [x, hp, front] = pk;
  const w = Math.max(6, hp * HPW);
  const g = document.createElementNS(NS, 'g');
  g.setAttribute('class', `pk ${n} ${front ? 'front' : ''}`);
  const x0 = -w / 2;
  el('rect', { x: x0, y: -7, width: w, height: 14, rx: 3, class: 'bar' }, g);
  for (let s = 8; s < hp; s += 8) el('line', { x1: x0 + s * HPW, x2: x0 + s * HPW, y1: -6, y2: 6, class: 'seg' }, g);
  if (front) {
    const tip = n === 'A' ? `M${w / 2 + 2},-6 L${w / 2 + 9},0 L${w / 2 + 2},6 Z` : `M${-w / 2 - 2},-6 L${-w / 2 - 9},0 L${-w / 2 - 2},6 Z`;
    el('path', { d: tip, class: 'chev' }, g);
  }
  const ty = n === 'A' ? -11 : 21;
  el('text', { x: 0, y: ty, class: 'hp', 'text-anchor': 'middle' }, g, `${hp}`);
  el('title', {}, g, `${n === 'A' ? names.A : names.B}: ${front ? 'front' : 'rear'} minion packet, ${hp} HP, at cell ${cellWord(x)}`);
  return g;
}

function drawPackets(frame, animate) {
  const layer = $('gPackets');
  for (const n of ['A', 'B']) {
    const list = frame.lane.packets[n];
    const y = n === 'A' ? Y.aRow : Y.bRow;
    const same = animate && pkEls[n].length === list.length;
    // stack packets of one side that share a cell
    const seen = {};
    const pos = list.map((p) => { const k = p[0]; seen[k] = (seen[k] || 0) + 1; return [p[0], (seen[k] - 1) * (n === 'A' ? -22 : 22)]; });
    if (!same) {
      for (const e of pkEls[n]) e.remove();
      pkEls[n] = list.map((p, i) => {
        const g = packetGroup(n, p, y);
        g.style.transition = 'none';
        g.style.transform = `translate(${cx(pos[i][0])}px, ${y + pos[i][1]}px)`;
        layer.append(g);
        if (animate) g.classList.add('enter');
        requestAnimationFrame(() => requestAnimationFrame(() => { g.style.transition = ''; }));
        return g;
      });
    } else {
      list.forEach((p, i) => {
        const old = pkEls[n][i];
        const g = packetGroup(n, p, y);
        g.style.transform = old.style.transform;
        layer.replaceChild(g, old);
        pkEls[n][i] = g;
        // eslint-disable-next-line no-unused-expressions
        g.getBoundingClientRect();
        g.style.transform = `translate(${cx(pos[i][0])}px, ${y + pos[i][1]}px)`;
      });
    }
  }
  // crashed masses under the enemy tower (A's under B's tower at +2, B's under A's at -2)
  const sg = $('gStatic');
  for (const n of ['A', 'B']) {
    const [x, hp] = frame.lane.siege[n];
    if (!hp) continue;
    const w = Math.max(8, hp * HPW);
    const y = n === 'A' ? Y.aSiege : Y.bSiege;
    const g = el('g', { class: `siege ${n}`, transform: `translate(${cx(x)}, ${y})` }, sg);
    el('rect', { x: -w / 2, y: -6, width: w, height: 12, rx: 2, class: 'blk' }, g);
    el('text', { x: w / 2 + 5, y: 4, 'text-anchor': 'start' }, g, `crashed ${hp}`);
    el('title', {}, g, `${n === 'A' ? names.A : names.B}'s crashed minions under the enemy tower: ${hp} HP`);
  }
}

function champPlace(frame, n) {
  if (frame.at_ward[n]) return { x: cx(0) + (n === 'A' ? -24 : 24), y: Y.w + 2, where: 'W' };
  const c = frame.cells[n];
  if (c === null || c === undefined) return { x: cx(n === 'A' ? -4 : 4), y: Y.ch, where: 'away' };
  const other = n === 'A' ? 'B' : 'A';
  const shared = !frame.at_ward[other] && frame.cells[other] === c;
  return { x: cx(c) + (shared ? (n === 'A' ? -21 : 21) : 0), y: Y.ch, where: c, shared };
}

function drawChamps(frame, animate) {
  const layer = $('gChamps');
  for (const n of ['A', 'B']) {
    const st = frame.status[n];
    const pl = champPlace(frame, n);
    let g = chEls[n];
    if (!g) {
      g = el('g', { class: `ch ${n}` }, layer);
      chEls[n] = g;
    }
    const away = pl.where === 'away';
    g.setAttribute('class', `ch ${n} ${away ? 'away' : ''}`);
    const frac = st.hp_max ? st.hp / st.hp_max : 0;
    g.innerHTML = '';
    if (n === 'A') el('circle', { r: 16, class: 'shape' }, g);
    else el('rect', { x: -13, y: -13, width: 26, height: 26, rx: 2, transform: 'rotate(45)', class: 'shape' }, g);
    el('text', { class: 'letter', y: 0.5 }, g, n);
    const bw = pl.shared || pl.where === 'W' ? 36 : 46;
    el('rect', { x: -bw / 2, y: 21, width: bw, height: 5, rx: 2.5, class: 'hpBg' }, g);
    el('rect', { x: -bw / 2, y: 21, width: Math.max(0, bw * frac), height: 5, rx: 2.5, class: `hpFg ${frac < 0.3 ? 'low' : ''}` }, g);
    const left = (pl.shared && n === 'A') || (pl.where === 'W' && n === 'A');
    el('text', { x: left ? -22 : 22, y: -8, class: 'lvl', 'text-anchor': left ? 'end' : 'start' }, g, `L${st.level}`);
    el('text', { x: 0, y: -22, class: 'tag' }, g, shorts[n]);
    if (away) {
      el('text', { x: 0, y: 39, class: 'tag' }, g, st.status === 'dead' ? `dead · back in ${st.returns_in}` : `at base · back in ${st.returns_in}`);
    }
    el('title', {}, g, `${names[n]}: level ${st.level}, ${st.hp} of ${st.hp_max} HP, ${away ? (st.status === 'dead' ? `dead, back in ${plural(st.returns_in, 'turn')}` : `recalling, back in ${plural(st.returns_in, 'turn')}`) : pl.where === 'W' ? 'at the Vision node W' : `at cell ${cellWord(pl.where)}`}`);
    if (!animate) g.style.transition = 'none';
    g.style.transform = `translate(${pl.x}px, ${pl.y}px)`;
    if (!animate) requestAnimationFrame(() => requestAnimationFrame(() => { g.style.transition = ''; }));
  }
}

function drawFx(frame) {
  const g = $('gFx');
  g.innerHTML = '';
  const tk = frame.tick;
  if (!tk) return;
  for (const n of ['A', 'B']) {
    const o = n === 'A' ? 'B' : 'A';
    const rem = tk.wave_removed[n];
    const ef = frame.lane.packets[o][0];
    if (rem > 0 && ef) {
      const x = cx(ef[0]) + (o === 'A' ? -1 : 1) * 0;
      const y = o === 'A' ? Y.aRow - 26 : Y.bRow + 34;
      const t = el('g', { transform: `translate(${x}, ${y})` }, g);
      el('rect', { x: -28, y: -9, width: 56, height: 15, rx: 7.5, class: 'fxPill' }, t);
      el('text', { x: 0, y: 2.5, class: 'fx removed', 'text-anchor': 'middle' }, t, `${shorts[n] === 'You' ? 'you' : n} −${rem}`);
      el('title', {}, t, `In tick ${tk.tick}, ${names[n]} removed ${rem} HP of ${names[o]}'s minions`);
    }
    const f = tk.minion_fire[n];
    if (f.units > 0) {
      const pl = champPlace(frame, n);
      const t = el('g', { transform: `translate(${pl.x + 34}, ${pl.y + 12})` }, g);
      el('rect', { x: -2, y: -9, width: 70, height: 15, rx: 7.5, class: 'fxPill' }, t);
      el('text', { x: 33, y: 2.5, class: 'fx fire', 'text-anchor': 'middle' }, t, `minions ×${f.units}`);
      el('title', {}, t, `In tick ${tk.tick}, ${names[o]}'s minions hit ${names[n]} (${f.units} unit${f.units === 1 ? '' : 's'}, ${f.cause})`);
    }
  }
}

function seqOf(frame) {
  if (frame.sub === 'start') return [frame.turn, 0];
  if (frame.sub === 'tick') return [frame.turn, frame.tick.tick];
  if (frame.sub === 'end') return [frame.turn, T + 1];
  return [frame.p, T + 1];                         // a position frame is the end of the turn that reached it
}

function drawBoard() {
  const f = ctrl.frame;
  const s = seqOf(f);
  const animate = !P.still && !core.prefersReducedMotion() && lastSeq !== null &&
    ((s[0] === lastSeq[0] && s[1] === lastSeq[1] + 1) || (s[1] === 0 && lastSeq[0] === s[0] - 1 && lastSeq[1] === T + 1));
  lastSeq = s;
  drawTowers(f);
  drawContact(f);
  drawPackets(f, animate);
  drawChamps(f, animate);
  drawFx(f);
  // the frame tag: where in the turn we are
  const steps = Array.from({ length: T + 2 }, (_, i) => `<i class="${i <= s[1] && (f.sub !== 'position' || i === T + 1) ? 'on' : ''}"></i>`).join('');
  const lbl = f.sub === 'position' ? (f.p === 0 ? 'Initial position' : `Position after turn ${f.p}`) :
              f.sub === 'start' ? `Turn ${f.turn} · start` : f.sub === 'tick' ? `Turn ${f.turn} · tick ${f.tick.tick} of ${T}` : `Turn ${f.turn} · end`;
  $('frameTag').innerHTML = `<span>${lbl}</span><span class="steps" aria-hidden="true">${steps}</span>`;
  const txt = laneText(f);
  $('laneText').textContent = txt;
  $('boardDesc').textContent = txt;
}

function laneText(f) {
  const parts = [f.sub === 'position' ? (f.p === 0 ? 'Initial position.' : `After turn ${f.p}.`) : `${$('frameTag').innerText || ''}.`];
  for (const n of ['A', 'B']) {
    const pk = f.lane.packets[n];
    const st = f.status[n];
    const pl = champPlace(f, n);
    const where = pl.where === 'away' ? (st.status === 'dead' ? `dead, back in ${plural(st.returns_in, 'turn')}` : `at base, back in ${plural(st.returns_in, 'turn')}`)
                  : pl.where === 'W' ? 'at the Vision node' : `at cell ${cellWord(pl.where)}`;
    parts.push(`${names[n]}: ${where}, ${st.hp} of ${st.hp_max} HP, level ${st.level}.`);
    parts.push(pk.length ? `${names[n]}'s minions: ${pk.map((p) => `${p[2] ? 'front' : 'rear'} ${p[1]} HP at ${cellWord(p[0])}`).join(', ')}.` : `${names[n]} has no minions in the lane.`);
    if (f.lane.siege[n][1]) parts.push(`${names[n]}'s crashed minions under the enemy tower: ${f.lane.siege[n][1]} HP.`);
  }
  if (f.lane.contact !== null && f.lane.contact !== undefined) parts.push(`The waves are in contact at ${f.lane.contact}.`);
  return parts.join(' ');
}

// ------------------------------------------------------------------------------------------- status strip
function seatLabel(n) {
  if (shorts[n] === 'You') return 'You';
  const x = game.meta.seats['AB'.indexOf(n)];
  return x.kind === 'model' ? x.label : `Scripted ${x.id}`;
}
function seatSub(n) {
  const x = game.meta.seats['AB'.indexOf(n)];
  if (shorts[n] === 'You') return `seat ${n} · staged`;
  return x.kind === 'model' ? `${x.id} · seat ${n}` : `scripted · seat ${n}`;
}
function drawStatus() {
  const f = ctrl.frame, st = f.status;
  const row = (n) => {
    const s = st[n];
    const hpf = s.hp_max ? s.hp / s.hp_max : 0;
    const xpf = s.xp_next ? (s.xp - s.xp_level) / (s.xp_next - s.xp_level) : 1;
    const away = s.status !== 'present';
    const state = s.status === 'dead' ? `<span class="statePill dead">Dead, back in ${s.returns_in}</span>`
      : s.status === 'recall' ? `<span class="statePill recall">At base, back in ${s.returns_in}</span>`
      : `<span class="statePill">${f.at_ward[n] ? 'At the Vision node' : f.cells[n] !== null && f.cells[n] !== undefined ? `In lane, cell ${cellWord(f.cells[n])}` : 'In lane'}${s.vision ? ` · ward ${s.vision}` : ''}</span>`;
    const tw = f.towers[n];
    return `<div class="srow" role="group" aria-label="${esc(names[n])}">
      <div class="sname">${glyph(n)}<span class="nn"><span class="n">${esc(seatLabel(n))}</span><small>${esc(seatSub(n))}</small></span></div>
      <div class="metric"><span class="k">HP</span><span class="v">${s.hp} <small>/ ${s.hp_max}</small></span><span class="meter ${n}"><i class="${hpf < 0.3 ? 'low' : ''}" style="width:${(hpf * 100).toFixed(1)}%"></i></span></div>
      <div class="metric"><span class="k">Level ${s.level}</span><span class="v">${s.xp} <small>XP${s.xp_next ? ` / ${s.xp_next}` : ' (max)'}</small></span><span class="meter xp"><i style="width:${(xpf * 100).toFixed(1)}%"></i></span></div>
      <div class="metric"><span class="k">Earned</span><span class="v">${g2(s.gold_earned)}</span></div>
      <div class="metric gold2"><span class="k">Unspent</span><span class="v">${g2(s.gold_unspent)}</span></div>
      <div class="metric gold3"><span class="k">Invested</span><span class="v">${g2(s.invested)}</span></div>
      <div>${state}</div>
      <div class="metric"><span class="k">Tower</span><span class="v">${tw.plates_total - tw.plates_lost}/${tw.plates_total} <small>plates</small></span></div>
    </div>`;
  };
  const pos = ctrl.game.positions[f.sub === 'position' || f.sub === 'end' ? f.p : f.p];
  const lead = pos.lead_A;
  const leader = lead > 0.004 ? names.A : lead < -0.004 ? names.B : 'Even';
  $('status').innerHTML = `<div class="rows">${row('A')}${row('B')}</div>
    <div class="leadBox" aria-label="Earned-gold lead"><span class="k">Earned-gold lead</span>
      <span class="v">${leader === 'Even' ? 'Even' : `${esc(shorts[lead > 0 ? 'A' : 'B'])} ${signed(Math.abs(lead)).replace('+', '+')}`}</span>
      <span class="s">A − B: ${signed(lead)} gold</span></div>`;
}

// ------------------------------------------------------------------------------------------- the outcome line
function tagsOf(tr) {
  const out = [];
  for (const e of tr.outcomes) {
    if (e.kind === 'death') out.push(['DIED', 'd', fill(e.text)]);
    else if (e.kind === 'trade' && e.data && e.data.result === 'hit') out.push(['HIT', 'h', fill(e.text)]);
    else if (e.kind === 'crash') out.push(['CRASH', '', fill(e.text)]);
    else if (e.kind === 'recall') out.push(['RCL', '', fill(e.text)]);
    else if (e.kind === 'ward') out.push(['WARD', '', fill(e.text)]);
    else if (e.kind === 'plate') out.push(['PLATE', '', fill(e.text)]);
  }
  const order = ['DIED', 'PLATE', 'HIT', 'CRASH', 'RCL', 'WARD'];
  out.sort((a, b) => order.indexOf(a[0]) - order.indexOf(b[0]));
  const uniq = [];
  for (const t of out) if (!uniq.some((u) => u[0] === t[0])) uniq.push(t);
  return uniq.slice(0, 2);
}

function drawOutcome() {
  const f = ctrl.frame;
  const t = f.sub === 'position' ? f.p : f.turn;
  const box = $('outcome');
  if (!t) {
    box.innerHTML = `<span class="tn">Start</span><span class="evs"><span class="quiet">Initial position: both champions at their towers, a fresh wave at each base.</span></span>`;
    return;
  }
  const tr = ctrl.game.turns[t - 1];
  const acts = `<span class="acts"><b class="A">${esc(shorts.A)}</b> ${esc(core.codeShort(tr.parts[0]))} · <b class="B">${esc(shorts.B)}</b> ${esc(core.codeShort(tr.parts[1]))}</span>`;
  let evs;
  if (f.sub === 'start' || f.sub === 'tick') {
    const tk = f.tick;
    evs = tk ? `<span class="quiet">Resolving · tick ${tk.tick} of ${T}: ${['A', 'B'].map((n) => tk.wave_removed[n] ? `${shorts[n] === 'You' ? 'you' : shorts[n]} removed ${tk.wave_removed[n]} HP of minions` : '').filter(Boolean).join(' · ') || 'the waves move'}</span>`
             : `<span class="quiet">Both actions revealed; resolving the turn…</span>`;
  } else {
    const major = tr.outcomes.filter((e) => e.major).map((e) => esc(fill(e.text)));
    evs = major.length ? major.join('<span class="sep">·</span>') : `<span class="quiet">No major outcome: ${['A', 'B'].map((n) => `${esc(shorts[n])} ${tr.delta[n].gold >= 0 ? '+' : ''}${g2(tr.delta[n].gold)} gold`).join(', ')}</span>`;
  }
  box.innerHTML = `<span class="tn">Turn ${t}</span>${acts}<span class="evs">${evs}</span>
    <button class="ghost small more" aria-controls="detail" aria-expanded="${detailOpen}">${detailOpen ? 'Hide details' : 'All details'}</button>`;
  box.querySelector('.more').addEventListener('click', () => openDetail(detailOpen ? null : 'events'));
}

// ------------------------------------------------------------------------------------------- the banner
function drawBanner() {
  const b = $('banner');
  b.className = 'banner';
  const pg = ctrl.progress, v = ctrl.view;
  if (ctrl.mode === 'play' && !ctrl.stopped && ctrl.historical) {
    b.classList.add('hist');
    b.innerHTML = `<span class="lead-in">Viewing ${v === 0 ? 'the initial position' : `the position after turn ${v}`}</span>
      <span class="grow">The game is at turn ${pg + 1}. Browsing history changes nothing; actions are disabled here.</span>
      <button class="returnBtn" id="returnBtn">Return to current turn ${pg + 1} <span class="mono">Esc</span></button>`;
    $('returnBtn').addEventListener('click', () => ctrl.returnToCurrent());
  } else if (ctrl.mode === 'play' && !ctrl.stopped) {
    const forced = ctrl.forced();
    b.innerHTML = `<span class="lead-in">Turn ${pg + 1} · ${forced ? 'you are away' : 'your decision'}</span>
      <span class="pill ready" title="The opponent's action for this turn is fixed. It is revealed when both actions resolve.">Opponent has chosen</span>
      <span class="grow">${forced ? 'Absent is your only action until you return.' : 'Pick a wave action and a behaviour, or Recall or Ward. Both actions resolve together.'}</span>
      <button class="ghost small" id="stopBtn">Stop game</button>`;
    $('stopBtn').addEventListener('click', () => ctrl.stop());
  } else if (ctrl.mode === 'play') {
    b.classList.add('stopped');
    b.innerHTML = `<span class="lead-in">Game stopped after turn ${pg}</span>
      <span class="grow">Review: step through any turn or autoplay from the start. No winner is declared.</span>
      <span class="pill">${pg >= ctrl.n ? '150-turn demo cap' : 'You stopped'}</span>`;
  } else {
    b.innerHTML = `<span class="lead-in">Watching</span><span class="grow">${esc(names.A)} (A) and ${esc(names.B)} (B) · ${ctrl.n} recorded turns · ${ctrl.playing ? `autoplay at ${ctrl.speed}×` : 'paused'}</span>
      <span class="pill">${game.meta.provenance.kind === 'research_record' ? 'Verified M5 recording' : 'New simulation'} · an example, not evidence</span>`;
  }
}

// ------------------------------------------------------------------------------------------- the move list
function buildList() {
  const ol = $('moves');
  const pg = ctrl.progress;
  const live = ctrl.mode === 'play' && !ctrl.stopped;
  const rows = [`<li><button class="mv start" data-p="0"><span class="no">0</span><span class="c">Start</span><span class="c"></span><span class="tg"></span></button></li>`];
  const lx = game.lane_exit_position, rp = game.read_position;
  for (let t = 1; t <= pg; t++) {
    const tr = game.turns[t - 1];
    const tg = tagsOf(tr).map(([s, c, title]) => `<i class="${c}" title="${esc(title)}">${s}</i>`).join('');
    const ca = core.codeShort(tr.parts[0]), cb = core.codeShort(tr.parts[1]);
    const label = `Turn ${t}: ${shorts.A} ${tr.names[0]}, ${shorts.B} ${tr.names[1]}${tagsOf(tr).length ? '; ' + tagsOf(tr).map((x) => x[2]).join('; ') : ''}`;
    rows.push(`<li><button class="mv" data-p="${t}" aria-label="${esc(label)}"><span class="no">${t}.</span><span class="c ${tr.parts[0].kind === 'absent' ? 'abs' : ''}">${esc(ca)}</span><span class="c ${tr.parts[1].kind === 'absent' ? 'abs' : ''}">${esc(cb)}</span><span class="tg">${tg}</span></button></li>`);
    if (t === lx) rows.push(`<li class="mvMark" aria-label="Lane exit at turn ${lx}"><span>Lane exit · turn ${lx}</span></li>`);
    if (t === rp) {
      const l = game.positions[rp].lead_A;
      rows.push(`<li class="mvMark lead" title="${esc(game.meta.marks.lead_114 ? game.meta.marks.lead_114.note : '')}"><span>Lead after ${rp} turns: ${esc(shorts.A)} ${signed(l)}</span></li>`);
    }
  }
  if (live && pg < ctrl.n) {
    const forced = ctrl.forced();
    rows.push(`<li><button class="mv pending" data-p="${pg + 1}" aria-label="Turn ${pg + 1}: your decision; the opponent has chosen"><span class="no">${pg + 1}.</span><span class="c">${forced ? 'Away' : 'Your move'}</span><span class="c wait">chosen ●</span><span class="tg"></span></button></li>`);
  }
  ol.innerHTML = rows.join('');
  listBuiltFor = pg;
}

function markList() {
  if (listBuiltFor !== ctrl.progress || listStopped !== ctrl.stopped) { buildList(); listStopped = ctrl.stopped; }
  const ol = $('moves');
  const v = ctrl.view;
  const live = ctrl.mode === 'play' && !ctrl.stopped;
  let target = null;
  for (const b of ol.querySelectorAll('.mv')) {
    const p = +b.dataset.p;
    const on = p === v && !(b.classList.contains('pending'));
    b.classList.toggle('viewed', on);
    b.setAttribute('aria-current', on ? 'true' : 'false');
    if (on) target = b;
    if (b.classList.contains('pending')) {
      b.classList.toggle('viewed', false);
      if (live && v === ctrl.progress && !target) target = b;
    }
  }
  if (target) {
    const r = target.getBoundingClientRect(), o = ol.getBoundingClientRect();
    if (r.top < o.top + 4 || r.bottom > o.bottom - 4) ol.scrollTop += (r.top - o.top) - o.height / 2 + r.height / 2;
  }
  if (live && v === ctrl.progress) ol.scrollTop = ol.scrollHeight;
}

function drawNav() {
  const live = ctrl.mode === 'play' && !ctrl.stopped;
  $('bFirst').disabled = ctrl.view === 0;
  $('bPrev').disabled = ctrl.view === 0;
  $('bNext').disabled = ctrl.view >= ctrl.progress;
  $('bLast').disabled = ctrl.view >= ctrl.progress;
  const canPlay = ctrl.playing || ctrl.view < ctrl.progress || !live;
  $('bPlay').disabled = !canPlay;
  $('bPlay').setAttribute('aria-label', ctrl.playing ? 'Pause (Space)' : 'Autoplay (Space)');
  $('playIcon').innerHTML = ctrl.playing ? '<path d="M5 4h3.5v12H5zM11.5 4H15v12h-3.5z"/>' : '<path d="M6 4l10 6-10 6z"/>';
  for (const b of $('speed').querySelectorAll('button')) b.setAttribute('aria-checked', String(+b.dataset.s === ctrl.speed));
  const v = ctrl.view;
  $('navInfo').innerHTML = live
    ? `<span>Viewing <b>${v === 0 ? 'start' : `after turn ${v}`}</b></span><span>Current <b>turn ${ctrl.progress + 1}</b></span>`
    : `<span>Viewing <b>${v === 0 ? 'start' : `turn ${v}`}</b> of ${ctrl.progress}</span><span>${ctrl.playing ? `<b>Playing</b> ${ctrl.speed}×` : 'Paused'}</span>`;
}

// ------------------------------------------------------------------------------------------- the bottom region
function currentCode() {
  if (!ctrl.canSubmit) return null;
  const m = ctrl.menu();
  if (m.forced !== null) return m.forced;
  let code = null;
  if (sel.single !== null) code = sel.single;
  else if (sel.wave !== null && sel.beh !== null) code = game.menu_static.lane_codes[`${sel.wave},${sel.beh},${sel.obj}`];
  return code !== null && m.legal.includes(code) ? code : null;
}

function pick(kind, id) {
  if (!ctrl.canSubmit || ctrl.forced()) return null;
  const m = ctrl.menu();
  notice = '';
  if (kind === 'single') {
    const reason = id === 12 ? m.reasons.recall : m.reasons.ward;
    if (reason) return null;
    sel.single = sel.single === id ? null : id;
    if (sel.single !== null) { sel.wave = null; sel.beh = null; sel.obj = 0; }
  } else {
    sel.single = null;
    if (kind === 'wave') sel.wave = id;
    if (kind === 'beh') {
      sel.beh = id;
      if (sel.obj === 1 && ![0, 2].includes(id)) { sel.obj = 0; notice = 'Pressure was cleared: it combines with Hold or Trade only.'; }
    }
    if (kind === 'obj') {
      if (id === 1 && (m.reasons.pressure || (sel.beh !== null && ![0, 2].includes(sel.beh)))) return null;
      sel.obj = id;
    }
  }
  render();
  return true;
}

function submit() {
  const code = currentCode();
  if (code === null) return null;
  ctrl.submit(code);
  sel.wave = sel.beh = sel.single = null; sel.obj = 0;
  return true;
}

function drawBottom() {
  const act = $('act');
  if (selFor !== ctrl.progress) { sel.wave = sel.beh = sel.single = null; sel.obj = 0; selFor = ctrl.progress; notice = ''; }
  if (ctrl.mode === 'play' && !ctrl.stopped) {
    if (ctrl.forced() || (ctrl.historical && game.menus_A[ctrl.progress].forced !== null)) act.innerHTML = awayHtml();
    else act.innerHTML = builderHtml();
  } else if (ctrl.mode === 'play') {
    act.innerHTML = summaryHtml();
  } else {
    act.innerHTML = playbackHtml();
  }
  act.querySelectorAll('[data-kind]').forEach((b) => b.addEventListener('click', () => pick(b.dataset.kind, +b.dataset.id)));
  const sb = act.querySelector('#submitBtn');
  if (sb) sb.addEventListener('click', submit);
  const cb = act.querySelector('#continueBtn');
  if (cb) cb.addEventListener('click', () => ctrl.continueAway());
  const sp = act.querySelector('.spark');
  if (sp) sp.addEventListener('click', (e) => {
    const r = sp.getBoundingClientRect();
    const p = Math.round(((e.clientX - r.left) / r.width) * ctrl.progress);
    ctrl.goto(Math.max(0, Math.min(ctrl.progress, p)));
  });
  const rb = act.querySelector('#replayBtn');
  if (rb) rb.addEventListener('click', () => { ctrl.first(); ctrl.play(); });
  drawDetail();
}

function optBtn(kind, item, pressed, reason, key) {
  const dis = !!reason;
  const lbl = `${item.name}${dis ? ` (unavailable: ${reason})` : ''}${item.help ? `. ${item.help}` : ''}`;
  return `<button class="opt" data-kind="${kind}" data-id="${item.id ?? item.code}" aria-pressed="${pressed}" ${dis ? 'disabled' : ''}
    aria-label="${esc(lbl)}" title="${esc(dis ? reason : item.help || '')}">${esc(item.name)}${key ? `<span class="k">${key}</span>` : ''}</button>`;
}

function builderHtml() {
  const hist = ctrl.historical;
  const m = game.menus_A[ctrl.progress];
  const ms = game.menu_static;
  const active = !hist;
  const pr = m.reasons.pressure;
  const behBlocks = sel.beh !== null && ![0, 2].includes(sel.beh);
  const wKeys = { 0: '1', 2: '2', 3: '3', 1: '4' }, bKeys = { 0: 'Q', 1: 'W', 2: 'E', 3: 'R' };
  const waves = ms.waves.map((w) => optBtn('wave', w, sel.wave === w.id, active ? null : 'viewing history', wKeys[w.id])).join('');
  const behs = ms.behaviours.map((b) => optBtn('beh', b, sel.beh === b.id, active ? null : 'viewing history', bKeys[b.id])).join('');
  const objs = ms.objectives.map((o) => {
    const reason = !active ? 'viewing history' : o.id === 1 ? (pr || (behBlocks ? 'Pressure combines with Hold or Trade only.' : null)) : null;
    return optBtn('obj', { ...o, name: o.id === 0 ? 'None' : o.name }, sel.single === null && sel.obj === o.id && active, reason, o.id === 1 ? 'P' : '');
  }).join('');
  const singles = ms.singles.map((s) => {
    const reason = !active ? 'viewing history' : s.code === 12 ? m.reasons.recall : m.reasons.ward;
    return optBtn('single', s, sel.single === s.code, reason, '');
  }).join('');
  const why = [];
  if (active) {
    if (pr) why.push(['Pressure', pr]);
    if (m.reasons.recall) why.push(['Recall', m.reasons.recall]);
    if (m.reasons.ward) why.push(['Ward', m.reasons.ward]);
    if (!pr && behBlocks) why.push(['Pressure', 'combines with Hold or Trade only.']);
  }
  const code = currentCode();
  let plan;
  if (hist) plan = `<span class="none">You are viewing history. Return to turn ${ctrl.progress + 1} to choose.</span>`;
  else if (sel.single !== null) {
    const s = ms.singles.find((x) => x.code === sel.single);
    plan = `<b>${esc(s.name)}</b>. ${esc(s.help)}`;
  } else if (sel.wave !== null || sel.beh !== null) {
    const w = ms.waves.find((x) => x.id === sel.wave), b = ms.behaviours.find((x) => x.id === sel.beh);
    plan = `<b>${w ? esc(w.name) : '<span class="none">wave?</span>'} + ${b ? esc(b.name) : '<span class="none">behaviour?</span>'}${sel.obj === 1 ? ' + Pressure' : ''}</b><br>${w ? esc(w.help) + ' ' : ''}${b ? esc(b.help) : ''}`;
  } else plan = '<span class="none">Nothing selected yet. Choose a wave action and a behaviour, or a single action.</span>';
  const ls = ctrl.lastSubmission;
  let result = '';
  if (ls && ls.t === ctrl.progress && !hist && ls.selected !== 45) {
    const tr = game.turns[ls.t - 1];
    const selName = labelOf(ls.selected), recName = tr.names[0];
    result = ls.matched
      ? `<div class="result match">Turn ${ls.t}: your choice, ${esc(recName)}, matched the recording.</div>`
      : `<div class="result miss">Turn ${ls.t}: you chose ${esc(selName)}; the recording's ${esc(recName)} resolved (staged).</div>`;
  }
  return `<div class="builder">
    <div class="bMain">
      <div class="bRow"><span class="lbl">Wave</span><div class="opts" role="group" aria-label="Wave action">${waves}</div></div>
      <div class="bRow"><span class="lbl">Behaviour</span><div class="opts" role="group" aria-label="Behaviour">${behs}</div></div>
      <div class="bRow"><span class="lbl">Objective</span><div class="opts" role="group" aria-label="Objective" style="grid-template-columns:repeat(4,minmax(0,1fr))">${objs}</div></div>
      <div class="why" aria-live="polite">${notice ? `<div><span class="x">!</span>${esc(notice)}</div>` : ''}${why.map(([k, r]) => `<div><span class="x">×</span><b>${k}</b> unavailable: ${esc(r)}</div>`).join('')}
        <div class="staged"><span aria-hidden="true">⚑</span><span><b>Staged prototype.</b> ${esc(core.STAGED_NOTE.replace(/^Prototype: /, ''))}</span></div></div>
    </div>
    <div class="bSide">
      <span class="lbl">Or a single action</span>
      <div class="singles" role="group" aria-label="Single actions">${singles}</div>
      <div class="plan" aria-live="polite">${plan}${result}</div>
      <button class="submit" id="submitBtn" ${code === null ? 'disabled' : ''} aria-label="Submit turn ${ctrl.progress + 1}">Submit turn ${ctrl.progress + 1} <span class="kbd">Enter</span></button>
    </div>
  </div>`;
}

function labelOf(code) {
  const ms = game.menu_static;
  const s = ms.singles.find((x) => x.code === code);
  if (s) return s.name;
  if (code === 45) return 'Absent';
  for (const [k, c] of Object.entries(ms.lane_codes)) {
    if (c === code) {
      const [w, b, o] = k.split(',').map(Number);
      return `${ms.waves.find((x) => x.id === w).name} + ${ms.behaviours.find((x) => x.id === b).name}${o === 1 ? ' + Pressure' : ''}`;
    }
  }
  return String(code);
}

function awayHtml() {
  const m = game.menus_A[ctrl.progress];
  const a = m.away || { status: 'recall', returns_in: 1 };
  const run = ctrl.awayRun();
  const dead = a.status === 'dead';
  const pips = Array.from({ length: Math.max(a.returns_in, 1) }, (_, i) => `<i class="${i < a.returns_in ? 'on' : ''}"></i>`).join('');
  const hist = ctrl.historical;
  return `<div class="awayCard">
    <div class="awayMain">
      <h3>${dead ? 'You died' : 'You are at base'} · back in lane in ${plural(a.returns_in, 'turn')}</h3>
      <p>${esc(m.reasons.absent || '')} Continue resolves the away turns one at a time; before each one the opponent's action is fixed, exactly as on any other turn.</p>
      <div class="count" aria-hidden="true">${pips}</div>
      <p class="staged" style="color:var(--muted)">Staged prototype: the recording's turns resolve.</p>
    </div>
    <div class="awaySide">
      <button class="submit" id="continueBtn" ${hist ? 'disabled' : ''}>Continue · away ${plural(run, 'turn')}</button>
      <span style="font-size:12px;color:var(--muted)">${hist ? 'Return to the current turn to continue.' : `Turns ${ctrl.progress + 1} to ${ctrl.progress + run} play as Absent for you.`}</span>
    </div>
  </div>`;
}

function summaryHtml() {
  const pg = ctrl.progress;
  const pos = game.positions[pg];
  const c = pos.champions;
  const capped = pg >= ctrl.n;
  const rowv = (k, f) => `<span class="k">${k}</span><span>${f('A')}</span><span>${f('B')}</span>`;
  const st = (n) => (c[n].status === 'present' ? 'in lane' : c[n].status === 'dead' ? `dead, back in ${c[n].returns_in}` : `at base, back in ${c[n].returns_in}`);
  const lx = game.lane_exit_position, rp = game.read_position;
  const ex = game.meta.marks.lane_exit, l114 = game.meta.marks.lead_114;
  const marks = [];
  if (pg >= lx && ex) {
    const pn = ex.panel;
    marks.push(`<div class="markBox"><h4>Lane exit · turn ${lx}</h4>Earned-gold lead ${signed(pn.D0)} (A − B) · levels ${pn.level_A} / ${pn.level_B} · XP ${signed(pn.xp_diff).replace('.00', '')} · HP ${Math.round(pn.hp_A * 100)}% / ${Math.round(pn.hp_B * 100)}%</div>`);
  } else marks.push(`<div class="markBox"><h4>Lane exit · turn ${lx}</h4>Not reached in this game.</div>`);
  if (pg >= rp && l114) marks.push(`<div class="markBox"><h4>Lead after ${rp} turns</h4>${esc(shorts.A)} ${signed(l114.lead_A)} gold. ${esc(l114.note)}</div>`);
  return `<div class="summary">
    <div class="sumMain">
      <h3>Game stopped after turn ${pg} · ${capped ? 'the 150-turn demo cap' : 'you stopped'}</h3>
      <p>Stopping is not an ending of the game and declares no winner. This is the state when the game stopped.</p>
      <div class="sumGrid"><span class="h"></span><span class="h">${esc(names.A)}</span><span class="h">${esc(names.B)}</span>
        ${rowv('Earned gold', (n) => g2(c[n].gold_earned))}
        ${rowv('Unspent · invested', (n) => `${g2(c[n].gold_unspent)} · ${g2(c[n].invested)}`)}
        ${rowv('Level (XP)', (n) => `${c[n].level} (${c[n].xp})`)}
        ${rowv('HP', (n) => `${c[n].hp} / ${c[n].hp_max}`)}
        ${rowv('State', st)}
        ${rowv('Own tower plates', (n) => `${pos.towers[n].plates_total - pos.towers[n].plates_lost} of ${pos.towers[n].plates_total}`)}
        <span class="k">Waves</span><span style="grid-column: span 2">${esc(pos.wave_status)}</span>
      </div>
    </div>
    <div class="sumSide">${marks.join('')}
      <div>${game.beliefs ? 'The opponent\'s belief history is now available under Details → Belief.' : ''} ${esc(game.meta.provenance.kind === 'research_record' ? 'In the real app this would be your own game: an example, not evidence for M5\'s findings.' : '')}</div>
      <button class="ghost" id="replayBtn">Replay from the start</button>
    </div>
  </div>`;
}

function spark() {
  const n = ctrl.progress, W = 280, H = 110, mid = H / 2;
  const leads = game.positions.slice(0, n + 1).map((p) => p.lead_A);
  const mx = Math.max(1, ...leads.map(Math.abs));
  const px = (p) => (p / Math.max(1, n)) * W, py = (l) => mid - (l / mx) * (mid - 8);
  const d = leads.map((l, p) => `${p ? 'L' : 'M'}${px(p).toFixed(1)},${py(l).toFixed(1)}`).join('');
  const area = `${d}L${px(n)},${mid}L0,${mid}Z`;
  const mk = [game.lane_exit_position, game.read_position].filter((x) => x <= n)
    .map((x) => `<line class="mk" x1="${px(x)}" x2="${px(x)}" y1="4" y2="${H - 4}"/><text class="mkT" x="${px(x) + 3}" y="12">${x}</text>`).join('');
  const v = ctrl.view;
  return `<svg class="spark" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label="Earned-gold lead of ${esc(names.A)} over the game; click to jump">
    <line class="zero" x1="0" x2="${W}" y1="${mid}" y2="${mid}"/><path class="area" d="${area}"/><path class="path" d="${d}"/>${mk}
    <line class="cur" x1="${px(v)}" x2="${px(v)}" y1="0" y2="${H}"/><circle class="curDot" cx="${px(v)}" cy="${py(leads[v])}" r="3"/></svg>`;
}

function helpOf(pt) {
  const ms = game.menu_static;
  if (pt.kind === 'absent') return 'Away from the lane: Absent.';
  if (pt.kind === 'recall') return ms.singles.find((x) => x.code === 12).help;
  if (pt.kind === 'ward') return ms.singles.find((x) => x.code === 30).help;
  const w = ms.waves.find((x) => x.id === pt.wave), b = ms.behaviours.find((x) => x.id === pt.behaviour);
  return `${w.name}: ${w.help} ${b.name}: ${b.help}${pt.objective === 1 ? ' Pressure: ' + ms.objectives.find((o) => o.id === 1).help : ''}`;
}

function playbackHtml() {
  const f = ctrl.frame;
  const t = f.sub === 'position' ? f.p : f.turn;
  const tr = t ? game.turns[t - 1] : null;
  const ended = f.sub === 'position' || f.sub === 'end';
  const items = tr ? (ended ? tr.outcomes : []) : [];
  let list = items.length
    ? items.map((e) => `<li class="${e.major ? 'major' : 'minor'}">${glyph(e.seat || 'A', 16).replace('class="glyph', 'class="sd glyph')}<span>${esc(fill(e.text))}</span></li>`).join('')
    : `<li class="minor"><span>${!tr ? 'The initial position. Press play or step forward.' : ended ? 'No outcome beyond farming this turn.' : 'Resolving the turn: its outcomes appear when it ends.'}</span></li>`;
  if (tr && ended) {
    list += `<li class="minor deltas">${['A', 'B'].map((n) => `<span>${esc(shorts[n])}: ${tr.delta[n].gold >= 0 ? '+' : ''}${g2(tr.delta[n].gold)} gold, +${tr.delta[n].xp} XP${tr.delta[n].hp_taken ? `, −${tr.delta[n].hp_taken} HP` : ''}</span>`).join('<span class="sep">·</span>')}</li>`;
  } else if (tr) {
    list += tr.parts.map((pt, i) => `<li class="minor">${glyph('AB'[i], 16).replace('class="glyph', 'class="sd glyph')}<span>${esc(helpOf(pt))}</span></li>`).join('');
  }
  return `<div class="playback">
    <div class="pbMain">
      <div class="pbHead"><h3>${t ? `Turn ${t}` : 'Start'} <span style="color:var(--muted);font-weight:500">of ${ctrl.progress}</span></h3>
        <span class="sub">${f.sub === 'tick' ? `tick ${f.tick.tick} of ${T}` : f.sub === 'start' ? 'both actions revealed' : t ? 'turn resolved' : ''}</span></div>
      <div class="pbActs">${['A', 'B'].map((n, i) => `<div class="pbAct">${glyph(n, 18)}<span class="who">${esc(shorts[n])}</span><span class="what">${tr ? esc(tr.names[i]) : '—'}</span></div>`).join('')}</div>
      <ul class="pbList" aria-live="polite">${list}</ul>
    </div>
    <div class="pbSide"><span class="lbl">Earned-gold lead of ${esc(shorts.A)}, this game</span>${spark()}
      <span class="sparkNote">Marks at the lane exit (94) and turn 114. One game: an example, not an expectation.</span></div>
  </div>`;
}

// ------------------------------------------------------------------------------------------- the detail panel
const TABS = [['events', 'Events'], ['wave', 'Wave'], ['ticks', 'Ticks'], ['belief', 'Belief'], ['exit', 'Lane exit'], ['record', 'Record']];

function openDetail(tab) {
  if (tab === null) { detailOpen = false; } else { detailOpen = true; detailTab = tab || detailTab; }
  render();
}
function toggleDetail() { openDetail(detailOpen ? null : detailTab); return true; }

function drawDetail() {
  const d = $('detail');
  $('detailBtn').setAttribute('aria-expanded', String(detailOpen));
  $('detailBtn').textContent = detailOpen ? 'Close details' : 'Details: events, wave, ticks, belief';
  d.hidden = !detailOpen;
  if (!detailOpen) return;
  const bv = ctrl.beliefVisible() && !!game.beliefs;
  const tabs = TABS.map(([k, l]) => {
    const dis = k === 'belief' && !bv;
    return `<button role="tab" data-tab="${k}" aria-selected="${detailTab === k}" ${dis ? 'aria-disabled="true"' : ''} title="${dis ? 'Hidden during play in this game; shown after the game stops' : ''}">${l}${dis ? ' (hidden)' : ''}</button>`;
  }).join('');
  d.innerHTML = `<div class="tabs" role="tablist" aria-label="Details">${tabs}<span class="spacer"></span><button class="ghost small close" id="closeDetail" aria-label="Close details">${ctrl.canSubmit ? 'Back to your action' : 'Close'}</button></div>
    <div class="tabBody" role="tabpanel">${tabBody(detailTab)}</div>`;
  d.querySelectorAll('[role="tab"]').forEach((b) => b.addEventListener('click', () => { detailTab = b.dataset.tab; render(); }));
  $('closeDetail').addEventListener('click', () => openDetail(null));
}

function viewTurn() {
  const f = ctrl.frame;
  return f.sub === 'position' ? f.p : f.turn;
}

function tabBody(tab) {
  const t = viewTurn();
  const tr = t ? game.turns[t - 1] : null;
  if (tab === 'events') {
    if (!tr) return '<p class="note">No turn has been played at the initial position.</p>';
    const ev = tr.outcomes.map((e) => `<li class="${e.major ? 'major' : 'minor'}">${glyph(e.seat || 'A', 16)}<span class="kind">${esc(e.kind)}</span><span class="txt">${esc(fill(e.text))}</span></li>`).join('');
    const dl = (n) => tr.delta[n];
    return `<p class="note">Turn ${t}, in the engine's order of resolution. HP is in the engine's units (a level-1 champion has 600); gold in resources.</p>
      <div style="display:grid;grid-template-columns:minmax(0,1.3fr) minmax(0,1fr);gap:20px">
      <ul class="evList">${ev || '<li class="minor"><span></span><span></span><span class="txt">Farming only.</span></li>'}</ul>
      <table class="t"><thead><tr><th></th><th>${esc(shorts.A)}</th><th>${esc(shorts.B)}</th></tr></thead><tbody>
        <tr><td>Action</td><td>${esc(tr.names[0])}</td><td>${esc(tr.names[1])}</td></tr>
        <tr><td>Mode</td><td>${esc(tr.modes[0])}</td><td>${esc(tr.modes[1])}</td></tr>
        <tr><td>Gold earned</td><td>${g2(dl('A').gold)}</td><td>${g2(dl('B').gold)}</td></tr>
        <tr><td>XP gained</td><td>${dl('A').xp}</td><td>${dl('B').xp}</td></tr>
        <tr><td>HP change</td><td>${dl('A').hp}</td><td>${dl('B').hp}</td></tr>
        <tr><td>HP taken</td><td>${dl('A').hp_taken}</td><td>${dl('B').hp_taken}</td></tr>
      </tbody></table></div>`;
  }
  if (tab === 'wave') {
    if (!tr) return '<p class="note">The wave budgets of a turn are computed from its start state; step to a turn.</p>';
    const pre = game.positions[t - 1].champions;
    const order = ['Last Hit', 'Thin', 'Slow Push', 'Shove'];
    const chosen = (i) => (tr.parts[i].kind === 'lane' ? game.menu_static.waves.find((w) => w.id === tr.parts[i].wave).name : null);
    const rows = ['A', 'B'].map((n, i) => {
      const c = chosen(i);
      const rem = tr.ticks.map((k) => k.wave_removed[n]);
      return `<tr><td>${glyph(n, 14)} ${esc(shorts[n])}</td><td>${pre[n].capacity ?? '—'}</td>${order.map((o) => `<td class="mono" ${o === c ? 'style="font-weight:700;text-decoration:underline"' : ''}>${pre[n].wave_budgets[o] ?? '—'}</td>`).join('')}
        <td>${c ? esc(c) : esc(tr.names[i])}</td><td class="mono">${rem.join(' + ')} = ${rem.reduce((a, b) => a + b, 0)}</td><td>${esc(tr.modes[i])}</td></tr>`;
    }).join('');
    return `<p class="note">The budget is what each wave action may spend on enemy minions this turn (from the start-of-turn lane and power); "removed" is the HP the engine applied, tick by tick. Underlined: the chosen action.</p>
      <table class="t"><thead><tr><th>Seat</th><th>Capacity</th>${order.map((o) => `<th>${o}</th>`).join('')}<th>Chosen</th><th>Removed per tick</th><th>Mode</th></tr></thead><tbody>${rows}</tbody></table>`;
  }
  if (tab === 'ticks') {
    if (!tr) return '<p class="note">No tick trace at the initial position.</p>';
    const pk = (l) => (l.length ? l.map((p) => `${cellWord(p[0])}:${p[1]}${p[2] ? '*' : ''}`).join(' ') : '—');
    const rows = tr.ticks.map((k) => `<tr><td>${k.tick}</td><td class="mono">${pk(k.packets.A)}</td><td class="mono">${pk(k.packets.B)}</td>
      <td class="mono">${k.siege.A[1] || '—'} / ${k.siege.B[1] || '—'}</td><td class="mono">${k.contact ?? '—'}</td>
      <td class="mono">${k.at_ward.A ? 'W' : k.cells.A ?? 'away'} / ${k.at_ward.B ? 'W' : k.cells.B ?? 'away'}</td>
      <td class="mono">${k.wave_removed.A} / ${k.wave_removed.B}</td><td class="mono">${k.minion_fire.A.units || '—'} / ${k.minion_fire.B.units || '—'}</td>
      <td class="mono">${k.crashed.A || '—'} / ${k.crashed.B || '—'}</td></tr>`).join('');
    return `<p class="note">The authoritative tick trace of turn ${t} (global cells; * marks the front packet). Values are A / B. Trades, tower shots, ganks, deaths and level-ups happen per turn and are not placed at a tick.</p>
      <table class="t"><thead><tr><th>Tick</th><th>A packets</th><th>B packets</th><th>Crashed (A / B)</th><th>Contact</th><th>Cells</th><th>Removed</th><th>Minion fire</th><th>Crashed this tick</th></tr></thead><tbody>${rows}</tbody></table>`;
  }
  if (tab === 'belief') {
    const b = ctrl.belief(ctrl.view);
    if (!b) return `<p class="note">${game.beliefs ? 'Hidden. In this game the opponent\'s belief stays hidden during play (the mode is chosen at the start). It becomes available when the game stops.' : 'No seat in this game keeps a belief: only the opponent-modelling agent does.'}</p>`;
    const blocks = Object.entries(b.bySeat).map(([seat, post]) => {
      const other = seat === 'A' ? 'B' : 'A';
      const mx = Math.max(...post);
      const rows = b.types.map((ty, i) => `<span>${esc(core.BELIEF_LABELS[ty] || ty)}</span><span class="bar ${seat}"><i style="width:${(post[i] * 100).toFixed(1)}%;${post[i] === mx ? '' : 'opacity:.55'}"></i></span><span class="pct">${(post[i] * 100).toFixed(1)}%</span>`).join('');
      return `<p class="note"><b>${esc(names[seat])}</b>'s belief about ${esc(names[other])} ${ctrl.view === 0 ? 'before any turn (the uniform prior)' : `after turn ${ctrl.view}`}. Fixed order; the darkest bar is the most likely type.</p><div class="beliefGrid">${rows}</div>`;
    }).join('');
    return `${blocks}<p class="note" style="margin-top:10px">${esc(core.BELIEF_NOTE)}</p>`;
  }
  if (tab === 'exit') {
    const ex = game.meta.marks.lane_exit, l = game.meta.marks.lead_114;
    const lx = game.lane_exit_position, rp = game.read_position;
    if (ctrl.progress < lx || !ex) return `<p class="note">The lane exit is marked at turn ${lx}. This game has not reached it yet.</p>`;
    const p = ex.panel;
    const pc = (x) => `${Math.round(x * 100)}%`;
    const st = (x) => ['present', 'at base (recall)', 'dead'][x];
    return `<p class="note">Turn ${lx} is where M5 read the lane exit. These are this game's values there; M5's findings average many games and all six seeds.</p>
      <div class="kv"><span class="k"></span><b>${esc(shorts.A)}</b><b>${esc(shorts.B)}</b>
        <span class="k">Earned gold</span><span>${g2(p.earned_A)}</span><span>${g2(p.earned_B)}</span>
        <span class="k">Level · XP</span><span>${p.level_A} · ${p.xp_A}</span><span>${p.level_B} · ${p.xp_B}</span>
        <span class="k">HP</span><span>${pc(p.hp_A)}</span><span>${pc(p.hp_B)}</span>
        <span class="k">State</span><span>${st(p.status_A)}</span><span>${st(p.status_B)}</span>
        <span class="k">Turns since last return</span><span>${p.since_return_A}</span><span>${p.since_return_B}</span>
        <span class="k">Unspent · invested</span><span>${g2(p.unspent_A)} · ${g2(p.invested_A)}</span><span>${g2(p.unspent_B)} · ${g2(p.invested_B)}</span>
        <span class="k">Earned-gold lead (A − B)</span><span>${signed(p.D0)}</span><span></span>
        <span class="k">Lead after ${rp} turns</span><span>${ctrl.progress >= rp && l ? signed(l.lead_A) : 'not reached'}</span><span></span>
      </div>${ctrl.progress >= rp && l ? `<p class="note" style="margin-top:8px">${esc(l.note)}</p>` : ''}`;
  }
  if (tab === 'record') {
    const m = game.meta;
    return `<div class="kv" style="grid-template-columns:160px 1fr">
      <span class="k">Kind</span><span>${m.provenance.kind === 'research_record' ? 'Verified M5 recording' : 'New simulation'}</span>
      <span class="k">Note</span><span>${esc(m.provenance.note)}</span>
      ${m.provenance.source ? `<span class="k">Source</span><span class="mono">${esc(m.provenance.source.file)}</span><span class="k">File SHA-256</span><span class="mono">${esc(m.provenance.source.sha256)}</span>` : ''}
      <span class="k">Configuration</span><span class="mono">${esc(m.config.name)} · ${esc(m.config.digest.slice(0, 16))}…</span>
      <span class="k">Seeds</span><span class="mono">chance ${m.seeds.chance}, policy ${m.seeds.policy} (${esc(m.seeds.policy_rule)})</span>
      ${m.seats.map((s) => `<span class="k">Seat ${s.seat}</span><span class="mono">${esc(s.kind === 'model' ? `${s.id} · ${s.label} · file ${s.file_sha256.slice(0, 16)}… · ${s.sampling}` : `${s.label}: ${s.rule || ''}`)}</span>`).join('')}
      <span class="k">Verification</span><span class="mono">${esc(JSON.stringify(game.verification))}</span></div>`;
  }
  return '';
}

// ------------------------------------------------------------------------------------------- render
function render() {
  drawBoard();
  drawStatus();
  drawOutcome();
  drawBanner();
  drawNav();
  drawBottom();
  markList();
}

boot().catch((e) => { window.__errors.push(String(e.stack || e)); document.body.dataset.ready = '1'; throw e; });
