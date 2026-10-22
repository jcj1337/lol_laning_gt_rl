// Plain words for everything on screen: the players' names (the research ids appear only in Details), the actions,
// the outcomes, and what each frame of a turn shows. Words for a tick come only from that tick's recorded facts;
// turn-level events (trades, tower shots at champions, ganks, deaths, Recalls, wards, levels, gold) appear only at the
// end frame, in the turn's own outcome text.
import type { Frame } from './frames';
import type { Arrow } from './moves';
import type { ChampionMove, CodeParts, Game, Names, Outcome, Seat, SeatPublic, WaveMove } from './types';
import { SEATS, other } from './types';

export const WAVE: Record<number, string> = { 0: 'Last hit', 2: 'Thin', 3: 'Slow push', 1: 'Shove' };
export const BEH: Record<number, string> = { 0: 'Hold', 1: 'Yield', 2: 'Trade', 3: 'Safe' };
export const WAVE_ORDER = [0, 2, 3, 1];
export const BEH_ORDER = [0, 1, 2, 3];
export const PRESSURE_BEHS = [0, 2];
export const RECALL_CODE = 12;
export const WARD_CODE = 30;
export const ABSENT_CODE = 45;

export const AGENTS: Record<string, { name: string; short: string; about: string }> = {
  'sp:0': { name: 'Self-play agent', short: 'Self-play', about: 'learned only by playing against itself' },
  'om:0': { name: 'Adaptive agent', short: 'Adaptive', about: 'keeps a running guess about what kind of opponent it faces and adjusts to it' },
};
export const BOTS: Record<string, { name: string; short: string; about: string }> = {
  g_shove: { name: 'Always-push bot', short: 'Push bot', about: 'a simple rule-based bot that always pushes its wave as hard as it can' },
  g_farm: { name: 'Farming bot', short: 'Farm bot', about: 'a simple rule-based bot that only farms' },
  g_safe: { name: 'Cautious bot', short: 'Safe bot', about: 'a simple rule-based bot that farms from a safe distance' },
  g_bully: { name: 'Aggressive bot', short: 'Bully bot', about: 'a simple rule-based bot that attacks whenever it can' },
};
export const GAME_NAMES: Record<string, string> = {
  h2h_om_s0_vs_sp_s0: 'Self-play agent vs. adaptive agent',
  om_s0_vs_g_shove: 'Always-push bot vs. adaptive agent',
  sp_s0_vs_g_shove: 'Self-play agent vs. always-push bot',
};

export function who(x: SeatPublic): { name: string; short: string } {
  if (x.kind === 'human') return { name: 'You', short: 'You' };
  if (x.kind === 'model') return AGENTS[x.id ?? ''] ?? { name: x.label ?? x.id ?? 'Agent', short: x.label ?? 'Agent' };
  return BOTS[x.id ?? ''] ?? { name: 'Rule-based bot', short: 'Bot' };
}

export function gameTitle(game: Game): string {
  return GAME_NAMES[game.name] || game.label || '';
}

export function recordType(kind: string | undefined): string {
  return ({ research_record: 'Recorded research game', new_simulation: 'Generated game', human_game: 'Your game, live' } as Record<string, string>)[kind ?? ''] || '';
}

/** The three name sets: full names, the short names used in sentences, and the history's column heads. */
export function nameSets(game: Game, live: boolean): { names: Names; short: Names; head: Names } {
  const [a, b] = game.meta.seats.map((x) => who(x).name);
  const dup = a === b;
  const names: Names = { A: dup ? `${a} (A)` : a, B: dup ? `${b} (B)` : b };
  const [sa, sb] = game.meta.seats.map((x) => who(x).short);
  const short: Names = live ? { A: 'You', B: 'the opponent' } : { A: sa === sb ? `${sa} A` : sa, B: sa === sb ? `${sb} B` : sb };
  const head: Names = live ? { A: 'You', B: 'Opponent' } : short;
  return { names, short, head };
}

// Fill the {A} / {B} placeholders of the server's outcome texts. A seat named "You" takes the second person ("Your
// wave", "you were ganked", "You are back"); the texts use the past tense otherwise, so no other verb changes.
export function fillText(text: string, names: Names): string {
  let out = text;
  for (const key of ['A', 'B'] as const) {
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

export const cap = (t: string): string => t.replace(/(^|[.!?]\s+)(the opponent|you)\b/g, (_m, a: string, b: string) => a + b[0].toUpperCase() + b.slice(1));
export const Cap = (t: string): string => (t ? t[0].toUpperCase() + t.slice(1) : t);
export const fill = (t: string, short: Names): string => cap(fillText(t, short));
export const sgn = (x: number, d = 2): string => (x > 0 ? '+' : x < 0 ? '−' : '') + Math.abs(x).toFixed(d);
export const turnsWord = (k: number): string => `${k} turn${k === 1 ? '' : 's'}`;

export function actionWords(p: CodeParts): string {
  if (p.kind === 'absent') return 'Away';
  if (p.kind === 'recall') return 'Recall';
  if (p.kind === 'ward') return 'Ward';
  return `${WAVE[p.wave ?? 0]}, ${BEH[p.behaviour ?? 0]}${p.objective === 1 ? ', Pressure' : ''}`;
}

/** A short phrase for a major outcome in a history row, or null for the ones the row does not show. */
export function phrase(e: Outcome, short: Names): string | null {
  if (!e.seat) return null;
  const n = short[e.seat], o = short[other(e.seat)];
  const you = n === 'You';
  const d = (e.data ?? {}) as Record<string, number | string>;
  let t: string | null;
  switch (e.kind) {
    case 'trade': t = d.result === 'hit' ? `${n} hit ${o === 'You' ? 'you' : o} (${d.damage})` : d.result === 'miss' ? `${you ? 'Your' : n + "'s"} trade missed` : null; break;
    case 'crash': t = `${you ? 'Your' : n + "'s"} wave crashed`; break;
    case 'tower_shot': t = `Tower hit ${you ? 'you' : n}`; break;
    case 'gank': t = `${you ? 'You were' : n + ' was'} ganked`; break;
    case 'death': t = `${n} died`; break;
    case 'plate': t = `${n} broke a plate`; break;
    case 'recall': t = `${n} recalled`; break;
    case 'recall_interrupted': t = `${you ? 'Your' : n + "'s"} recall was cut`; break;
    case 'ward': t = `${n} warded`; break;
    case 'ward_interrupted': t = `${you ? 'Your' : n + "'s"} ward was cut`; break;
    default: t = null;
  }
  return t && cap(t);
}

// ---- the frames of a turn, in words
const own = (n: Seat, short: Names) => (short[n] === 'You' ? 'your' : `${short[n]}'s`);
const they = (n: Seat, short: Names) => Cap(short[n]);
const verb = (n: Seat, short: Names, you: string, oth: string) => (short[n] === 'You' ? you : oth);

function cellName(c: number, short: Names): string {
  if (c === 0) return 'the lane';
  const n: Seat = c < 0 ? 'A' : 'B', d = Math.abs(c);
  return d === 1 ? `${own(n, short)} side of the lane` : d === 2 ? `${own(n, short)} tower` : d === 3 ? `behind ${own(n, short)} tower` : `${own(n, short)} base`;
}
const toCell = (c: number, short: Names) => (Math.abs(c) === 3 ? cellName(c, short) : `to ${cellName(c, short)}`);
const placeOf = (f: Frame, n: Seat): number | 'W' | null => (f.at_ward[n] ? 'W' : f.cells[n] === null || f.cells[n] === undefined ? null : f.cells[n]);
const frontOf = (f: Frame, n: Seat) => f.lane.packets[n].find((q) => q[2]) || null;

export function tickWords(prev: Frame, f: Frame, short: Names): string[] {
  const out: string[] = [], tk = f.tick!;
  for (const n of SEATS) {
    const a = placeOf(prev, n), b = placeOf(f, n);
    if (b === null || a === b) continue;
    if (b === 'W') out.push(`${they(n, short)} ${verb(n, short, 'walk', 'walks')} into the bush.`);
    else if (a === 'W') out.push(`${they(n, short)} ${verb(n, short, 'leave', 'leaves')} the bush and ${verb(n, short, 'move', 'moves')} ${toCell(b, short)}.`);
    else out.push(`${they(n, short)} ${verb(n, short, 'move', 'moves')} ${toCell(b, short)}.`);
  }
  const ct = f.lane.contact;
  if (ct !== null && ct !== undefined) {
    out.push(`${Cap(own('A', short))} wave and ${own('B', short)} wave ${prev.lane.contact === ct ? 'keep fighting' : 'fight'} at ${cellName(ct, short)}.`);
  } else {
    for (const n of SEATS) {
      const a = frontOf(prev, n), b = frontOf(f, n);
      if (a && b && a[0] !== b[0] && !tk.crashed[n]) out.push(`${Cap(own(n, short))} wave walks ${toCell(b[0], short)}.`);
    }
  }
  const r = tk.wave_removed;
  if (r.A > 0 && r.B > 0) out.push(`${they('A', short)} ${verb('A', short, 'remove', 'removes')} ${r.A} HP of enemy minions; ${short.B} ${verb('B', short, 'remove', 'removes')} ${r.B}.`);
  else for (const n of SEATS) if (r[n] > 0) out.push(`${they(n, short)} ${verb(n, short, 'remove', 'removes')} ${r[n]} HP of enemy minions.`);
  for (const n of SEATS) {
    if (tk.crashed[n] > 0) out.push(`${Cap(own(n, short))} wave crashes into ${own(other(n), short)} tower (${tk.crashed[n]} HP).`);
    else {
      const was = prev.lane.siege[n][1], now = f.lane.siege[n][1];
      if (now < was) out.push(now > 0 ? `${Cap(own(n, short))} crashed minions under ${own(other(n), short)} tower drop to ${now} HP.`
        : `${Cap(own(n, short))} crashed minions under ${own(other(n), short)} tower are gone.`);
    }
  }
  for (const n of SEATS) if (tk.minion_fire[n].units > 0) out.push(`${Cap(own(other(n), short))} minions hit ${short[n] === 'You' ? 'you' : short[n]}.`);
  return out.length ? out : ['Nothing changes in this tick.'];
}

export interface FrameWords { head: string; text: string; outcomes?: string[]; more?: string | null }

/** The heading, text and (at the end) outcomes of frame i of turn t. */
export function frameWords(game: Game, t: number, frames: Frame[], i: number, short: Names, maxOut = 3): FrameWords {
  const f = frames[i], tr = game.turns[t - 1], nt = tr.ticks.length;
  if (f.sub === 'start') {
    const chose = (n: Seat, first: boolean) => {
      const pt = tr.parts[n === 'A' ? 0 : 1], w = first ? they(n, short) : short[n];
      return pt.kind === 'absent' ? `${w} ${verb(n, short, 'are', 'is')} away` : `${w} chose ${actionWords(pt)}`;
    };
    return { head: 'Both actions revealed', text: `${chose('A', true)}; ${chose('B', false)}. `
      + `${nt === 2 ? 'Two ticks' : `${nt} ticks`} follow: the champions move, and the waves walk, fight and lose HP.` };
  }
  if (f.sub === 'tick') return { head: `Tick ${f.tick!.tick} of ${nt}`, text: tickWords(frames[i - 1], f, short).join(' ') };
  const major = tr.outcomes.filter((e) => e.major), extra = tr.outcomes.length - Math.min(maxOut, major.length);
  return { head: `End of turn ${t}`,
    text: 'Trades, tower shots at champions, ganks, deaths, Recalls, wards, levels and gold are settled at the end of a turn.',
    outcomes: major.length ? major.slice(0, maxOut).map((e) => fill(e.text, short)) : ['No trades, crashes, Recalls or deaths.'],
    more: extra > 0 ? `+${extra} more` : null };
}

// ---- the lane's arrows: a sentence for each (the hover title and the screen-reader text) and a label for each kind
// except a plain walk. {X} stands for the seat, so a seat named "You" reads "Your leading wave" (fillText).
type ArrowWords = { text: string; label: string | null };
const CHAMPION_MOVE: Record<ChampionMove['kind'], ArrowWords> = {
  walk: { text: '{X} moved this turn', label: null },
  recall: { text: '{X} recalled to base', label: 'Recall' },
  return: { text: '{X} came back from base', label: 'Back from base' },
  death: { text: '{X} died and went back to base', label: 'Died' },
};
const WAVE_MOVE: Record<WaveMove['kind'], ArrowWords> = {
  walk: { text: "{X}'s leading wave walked forward", label: null },
  crash: { text: "{X}'s leading wave crashed into the tower", label: 'Crash' },
};

export function arrowWords(a: Arrow, names: Names): ArrowWords {
  const w = a.what === 'champion' ? CHAMPION_MOVE[a.kind] : WAVE_MOVE[a.kind];
  return { text: fillText(w.text.replace('{X}', `{${a.seat}}`), names), label: w.label };
}

export const GUESS: Record<string, string> = {
  g_farm: 'Farming bot', g_safe: 'Cautious bot', g_bully: 'Aggressive bot', g_shove: 'Always-push bot',
  fz_guard_sh: 'Freezing bot', g_shove_ward: 'Push-and-ward bot', peer: 'A trained agent like itself',
  attacker: 'An agent trained to attack', freezer: 'An agent that holds the wave', gapper: 'An agent that presses big leads',
  short: 'A short-sighted trained agent', mixed: 'A mix of the bots', unknown: 'None of these',
};
export const BELIEF_NOTE = 'The adaptive agent compares how its opponent plays with 13 kinds of player: 6 simple bots, 5 trained '
  + 'agents, a mix of the bots, and "none of these". None of them is a person. When its guess is shown you can try to '
  + 'mislead it; the research never tested that.';
export const BUSH_NOTE = 'The bush hides no one in this simulator: the other side still sees where this champion is. The '
  + 'see-through token only marks that it stands in the bush.';
export const GRAPH_NOTE = 'How much more gold one side has earned so far. Not a chance of winning.';
export const EXAMPLE_NOTE = "One game is an example, not evidence for M5's findings. See About.";
export const DOCS_URL = 'https://github.com/jcj1337/lol_game_theory_laning/blob/main/docs/README.md';
/** A position in words: the state after turn p, or the start. */
export const afterTurn = (p: number): string => (p === 0 ? 'the start' : `after turn ${p}`);
