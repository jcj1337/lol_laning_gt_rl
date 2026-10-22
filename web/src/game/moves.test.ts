// The lane's arrows describe the turn that produced the position on screen: a position after turn p and the end frame
// of turn t carry that turn's moves; the initial position and a turn's start and tick frames carry none.
import { describe, expect, it } from 'vitest';
import { positionFrame, turnSubframes } from './frames';
import { arrowsFor } from './moves';
import type { Game, Position, SeatMoves, Turn } from './types';
import { arrowWords } from './words';

function position(p: number): Position {
  const champ = { level: 1, hp: 600, hp_max: 600, xp: 0, xp_level: 0, xp_next: 10, gold_earned: 0, gold_unspent: 0, invested: 0, attack_power: 1,
    status: 'present' as const, returns_in: 0, cell: -2, at_ward: false, vision: 0, capacity: 4, wave_budgets: {} };
  return { p, k: p, lane: { packets: { A: [[-3, 8, true]], B: [[3, 8, true]] }, siege: { A: [-2, 0], B: [2, 0] }, contact: null },
    champions: { A: champ, B: { ...champ, cell: 2 } }, towers: { A: { damage: 0, damage_max: 15, plates_lost: 0, plates_total: 3 }, B: { damage: 0, damage_max: 15, plates_lost: 0, plates_total: 3 } },
    lead_A: 0, wave_status: 'separated' };
}

const still: SeatMoves = { champion: null, wave: null };

function turn(t: number, A: SeatMoves = still, B: SeatMoves = still): Turn {
  const tick = { tick: 1, packets: { A: [], B: [] }, siege: { A: [-2, 0] as [number, number], B: [2, 0] as [number, number] }, contact: null, cells: { A: -2, B: 2 },
    at_ward: { A: false, B: false }, wave_removed: { A: 0, B: 0 }, crashed: { A: 0, B: 0 }, minion_fire: { A: { cause: '', units: 0 }, B: { cause: '', units: 0 } } };
  return { t, k: t, codes: [2, 2], names: ['x', 'y'], parts: [{ kind: 'lane', wave: 0, behaviour: 0, objective: 0 }, { kind: 'lane', wave: 0, behaviour: 0, objective: 0 }],
    modes: ['farm', 'farm'], ticks: [tick, { ...tick, tick: 2 }], outcomes: [], delta: { A: { gold: 0, farm: 0, xp: 0, hp: 0, hp_taken: 0 }, B: { gold: 0, farm: 0, xp: 0, hp: 0, hp_taken: 0 } },
    reward_A: 0, moves: { A, B } };
}

function game(turns: Turn[], positions = turns.length + 1): Game {
  return { name: 'g', label: '', meta: { provenance: { kind: 'new_simulation' }, seats: [{ seat: 'A', kind: 'model' }, { seat: 'B', kind: 'model' }],
    seeds: { chance: 1, chance_chosen: false, policy: 2, policy_rule: 'given' }, marks: {}, belief_mode: 'hidden', turn_cap: 150 },
    positions: Array.from({ length: positions }, (_, p) => position(p)), turns, beliefs: null, lane_exit_position: 94, read_position: 114 };
}

// Turn 1: A walks to the lane while its wave walks; B's wave crashes. Turn 2: B recalls; nothing else moves.
const G = game([
  turn(1, { champion: { kind: 'walk', from: -2, to: 0 }, wave: { kind: 'walk', from: -3, to: -1 } }, { champion: null, wave: { kind: 'crash', from: -1, to: -2 } }),
  turn(2, still, { champion: { kind: 'recall', from: 2, to: 'base' }, wave: null }),
]);
const TURN1 = [
  { seat: 'A', what: 'champion', kind: 'walk', from: -2, to: 0 },
  { seat: 'A', what: 'wave', kind: 'walk', from: -3, to: -1 },
  { seat: 'B', what: 'wave', kind: 'crash', from: -1, to: -2 },
];
const TURN2 = [{ seat: 'B', what: 'champion', kind: 'recall', from: 2, to: 'base' }];

describe('arrowsFor', () => {
  it("gives a position frame after turn p that turn's moves, champion A, wave A, champion B, wave B, skipping null moves", () => {
    expect(arrowsFor(positionFrame(G, 1), G.turns)).toEqual(TURN1);
    expect(arrowsFor(positionFrame(G, 2), G.turns)).toEqual(TURN2);
  });

  it("gives the end frame of turn t that turn's moves", () => {
    const frames = turnSubframes(G, 2);
    expect(frames[frames.length - 1].sub).toBe('end');
    expect(arrowsFor(frames[frames.length - 1], G.turns)).toEqual(TURN2);
    expect(arrowsFor(turnSubframes(G, 1).at(-1)!, G.turns)).toEqual(TURN1);
  });

  it('gives none for the initial position, nor for a turn\'s start and tick frames, which animate the motion themselves', () => {
    expect(arrowsFor(positionFrame(G, 0), G.turns)).toEqual([]);
    const frames = turnSubframes(G, 1);
    expect(frames.map((f) => f.sub)).toEqual(['start', 'tick', 'tick', 'end']);
    for (const f of frames.slice(0, -1)) expect(arrowsFor(f, G.turns)).toEqual([]);
  });

  it('gives none when the turn that produced the frame is not known', () => {
    const short = game(G.turns.slice(0, 1), 3);
    expect(arrowsFor(positionFrame(short, 2), short.turns)).toEqual([]);
    expect(arrowsFor(positionFrame(G, 2), [])).toEqual([]);
  });
});

describe('arrowWords', () => {
  it("names the seat, in the second person for a seat named You, and labels every kind but a plain walk", () => {
    const [walk, wave, crash] = arrowsFor(positionFrame(G, 1), G.turns);
    const names = { A: 'You', B: 'Self-play agent' };
    expect(arrowWords(walk, names)).toEqual({ text: 'You moved this turn', label: null });
    expect(arrowWords(wave, names)).toEqual({ text: 'Your leading wave walked forward', label: null });
    expect(arrowWords(crash, names)).toEqual({ text: "Self-play agent's leading wave crashed into the tower", label: 'Crash' });
    expect(arrowWords(arrowsFor(positionFrame(G, 2), G.turns)[0], names)).toEqual({ text: 'Self-play agent recalled to base', label: 'Recall' });
  });
});
