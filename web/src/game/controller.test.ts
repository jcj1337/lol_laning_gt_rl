// The history rules of plan §5.1 and rule 13, on the controller alone: progress and the viewed position are separate,
// browsing never changes progress, submission is impossible while browsing, and a history catch-up appends only the
// turns that continue what the client has. A walk shows each position up to its target and stops there, and playback
// without motion keeps the pace of the animated turns.
import { afterEach, describe, expect, it, vi } from 'vitest';
import { Controller, Live } from './controller';
import { TICK_MS, WALK_MS } from './frames';
import type { Game, History, Position, Turn, View } from './types';

const env = { reduced: () => true, ticksAuto: () => false, forcedRun: () => false };

function position(p: number, lead = 0): Position {
  const champ = { level: 1, hp: 600, hp_max: 600, xp: 0, xp_level: 0, xp_next: 10, gold_earned: 0, gold_unspent: 0, invested: 0, attack_power: 1,
    status: 'present' as const, returns_in: 0, cell: -2, at_ward: false, vision: 0, capacity: 4, wave_budgets: {} };
  return { p, k: p, lane: { packets: { A: [[-3, 8, true]], B: [[3, 8, true]] }, siege: { A: [-2, 0], B: [2, 0] }, contact: null },
    champions: { A: champ, B: { ...champ, cell: 2 } }, towers: { A: { damage: 0, damage_max: 15, plates_lost: 0, plates_total: 3 }, B: { damage: 0, damage_max: 15, plates_lost: 0, plates_total: 3 } },
    lead_A: lead, wave_status: 'separated' };
}

function turn(t: number, a = 2, b = 2): Turn {
  const tick = { tick: 1, packets: { A: [], B: [] }, siege: { A: [-2, 0] as [number, number], B: [2, 0] as [number, number] }, contact: null, cells: { A: -2, B: 2 },
    at_ward: { A: false, B: false }, wave_removed: { A: 0, B: 0 }, crashed: { A: 0, B: 0 }, minion_fire: { A: { cause: '', units: 0 }, B: { cause: '', units: 0 } } };
  return { t, k: t, codes: [a, b], names: ['x', 'y'], parts: [{ kind: 'lane', wave: 0, behaviour: 0, objective: 0 }, { kind: 'lane', wave: 0, behaviour: 0, objective: 0 }],
    modes: ['farm', 'farm'], ticks: [tick, { ...tick, tick: 2 }], outcomes: [], delta: { A: { gold: 0, farm: 0, xp: 0, hp: 0, hp_taken: 0 }, B: { gold: 0, farm: 0, xp: 0, hp: 0, hp_taken: 0 } }, reward_A: 0,
    moves: { A: { champion: null, wave: null }, B: { champion: null, wave: null } } };
}

function game(n: number): Game {
  return { name: 'g', label: '', meta: { provenance: { kind: 'new_simulation' }, seats: [{ seat: 'A', kind: 'model' }, { seat: 'B', kind: 'model' }],
    seeds: { chance: 1, chance_chosen: false, policy: 2, policy_rule: 'given' }, marks: {}, belief_mode: 'hidden', turn_cap: 150 },
    positions: Array.from({ length: n + 1 }, (_, p) => position(p)), turns: Array.from({ length: n }, (_, i) => turn(i + 1)), beliefs: null,
    lane_exit_position: 94, read_position: 114 };
}

const view = (position: number, state: 'awaiting_action' | 'stopped' = 'awaiting_action'): View => ({
  session: 'x', kind: 'human_game', state, stop_reason: null, position, next_turn: position + 1, turn_cap: 150, lane_exit_position: 94, read_position: 114,
  seats: [{ seat: 'A', kind: 'human' }, { seat: 'B', kind: 'model' }], belief_mode: 'hidden', seeds: { chance: null, chance_chosen: false, policy: null, policy_rule: 'drawn' },
  current: position_(position), marks: {}, human_seat: 'A', opponent_ready: true,
  menu: { legal: [2], forced: null, away: null, waves: [], behaviours: [], objectives: [], singles: [], lane_codes: {} },
});
const position_ = position;

describe('Controller (a whole bot game)', () => {
  it('browses without ever changing progress, and clamps the view', () => {
    const c = new Controller(game(10), { mode: 'watch', env });
    expect(c.progress).toBe(10);
    expect(c.view).toBe(0);
    c.goto(7);
    expect(c.view).toBe(7);
    c.next(); c.next(); c.next(); c.next();
    expect(c.view).toBe(10);
    expect(c.progress).toBe(10);
    c.first();
    expect(c.view).toBe(0);
    c.prev();
    expect(c.view).toBe(0);
    c.latest();
    expect(c.view).toBe(10);
    expect(c.canSubmit).toBe(false);
  });

  it('shows a belief only when the mode or the stop allows it', () => {
    const G = game(2);
    G.beliefs = { types: ['a', 'b'], posteriors: { B: [[0.5, 0.5], [0.6, 0.4], [0.7, 0.3]] } };
    expect(new Controller(G, { mode: 'watch', env }).belief(2)?.bySeat.B).toEqual([0.7, 0.3]);
    const live = new Live(G, 150, { belief: 'hidden', env });
    expect(live.belief(2)).toBeNull();
    live.stopped = true;
    expect(live.belief(2)?.bySeat.B).toEqual([0.7, 0.3]);
    expect(new Live(G, 150, { belief: 'shown', env }).belief(1)?.bySeat.B).toEqual([0.6, 0.4]);
  });
});

describe('Controller timing', () => {
  afterEach(() => { vi.useRealTimers(); });

  it('walks to a position one position per interval and stops there', () => {
    vi.useFakeTimers();
    const c = new Controller(game(10), { mode: 'watch', view: 3, env });
    c.playTo(7);
    expect(c.view).toBe(3);
    expect(c.playing).toBe(true);
    expect(c.walkingTo).toBe(7);
    const seen: number[] = [];
    for (let i = 0; i < 3; i++) { vi.advanceTimersByTime(WALK_MS); seen.push(c.view); }
    expect(seen).toEqual([4, 5, 6]);
    expect(c.playing).toBe(true);
    vi.advanceTimersByTime(WALK_MS);
    expect(c.view).toBe(7);
    expect(c.playing).toBe(false);
    expect(c.walkingTo).toBeNull();
    vi.advanceTimersByTime(WALK_MS * 10);
    expect(c.view).toBe(7);
    expect(c.frame.p).toBe(7);
  });

  it('stops a walk on any jump', () => {
    vi.useFakeTimers();
    const c = new Controller(game(10), { mode: 'watch', view: 3, env });
    c.playTo(9);
    vi.advanceTimersByTime(WALK_MS);
    expect(c.view).toBe(4);
    c.goto(5);
    expect(c.playing).toBe(false);
    expect(c.walkingTo).toBeNull();
    expect(c.view).toBe(5);
    vi.advanceTimersByTime(WALK_MS * 10);
    expect(c.view).toBe(5);
  });

  it('plays without motion at the pace of the animated turns', async () => {
    vi.useFakeTimers();
    const c = new Controller(game(10), { mode: 'watch', env });
    const turnMs = 4 * TICK_MS;                                // a turn of two ticks: start, tick 1, tick 2, end
    c.play();
    expect(c.view).toBe(1);
    await vi.advanceTimersByTimeAsync(turnMs - 1);
    expect(c.view).toBe(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(c.view).toBe(2);
    await vi.advanceTimersByTimeAsync(turnMs);
    expect(c.view).toBe(3);
    expect(c.playing).toBe(true);
    c.pause();
  });
});

describe('Live (a growing human game)', () => {
  it('keeps the decision point and the viewed position apart', async () => {
    const G = game(0);
    const c = new Live(G, 150, { env });
    c.adopt(view(0));
    expect(c.canSubmit).toBe(true);
    await c.append({ turn: turn(1), position: position(1, 0.5), view: view(1) });
    await c.append({ turn: turn(2), position: position(2, 0.25), view: view(2) });
    expect(c.progress).toBe(2);
    expect(c.view).toBe(2);
    expect(c.canSubmit).toBe(true);
    c.goto(1);
    expect(c.historical).toBe(true);
    expect(c.canSubmit).toBe(false);
    expect(c.progress).toBe(2);
    expect(c.position(3)).toBeNull();
    c.returnToCurrent();
    expect(c.view).toBe(2);
    expect(c.canSubmit).toBe(true);
    c.adopt(view(2, 'stopped'));
    expect(c.stopped).toBe(true);
    expect(c.canSubmit).toBe(false);
    expect(c.menu()).toBeNull();
  });

  it('appends from a history only the turns that continue what it has', () => {
    const G = game(1);
    const c = new Live(G, 150, { env });
    c.adopt(view(1));
    const h: History = { session: 'x', from: 0, to: 3, positions: [position(0), position(1), position(2), position(3)], turns: [turn(1), turn(2), turn(3)] };
    expect(c.appendHistory(h, view(3))).toBe(2);
    expect(c.progress).toBe(3);
    expect(G.turns.map((t) => t.t)).toEqual([1, 2, 3]);
    expect(c.view).toBe(3);
    expect(c.appendHistory(h, view(3))).toBe(0);
    expect(c.progress).toBe(3);
  });

  it('never plays back past the decision point of a running game', () => {
    const c = new Live(game(3), 150, { env });
    c.adopt(view(3));
    c.play();
    expect(c.playing).toBe(false);
    expect(c.view).toBe(3);
    c.adopt(view(3, 'stopped'));
    c.play();
    expect(c.playing).toBe(true);
    c.pause();
    expect(c.playing).toBe(false);
    c.dispose();
  });
});
