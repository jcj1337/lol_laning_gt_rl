// The store against a fake server: a late response from an earlier game never acts on the new one, an unanswered
// submission re-reads the server and appends what it missed (or offers a retry when nothing changed), and a lost
// session is reported as such.
import { describe, expect, it } from 'vitest';
import { ApiError } from '../api/client';
import type { Api } from '../api/client';
import { AppStore, readParams } from './store';
import type { CreateResponse, History, Position, Turn, TurnResult, View } from './types';

function position(p: number): Position {
  const champ = { level: 1, hp: 600, hp_max: 600, xp: 0, xp_level: 0, xp_next: 10, gold_earned: 0, gold_unspent: 0, invested: 0, attack_power: 1,
    status: 'present' as const, returns_in: 0, cell: -2, at_ward: false, vision: 0, capacity: 4, wave_budgets: {} };
  return { p, k: p, lane: { packets: { A: [], B: [] }, siege: { A: [-2, 0], B: [2, 0] }, contact: null }, champions: { A: champ, B: champ },
    towers: { A: { damage: 0, damage_max: 15, plates_lost: 0, plates_total: 3 }, B: { damage: 0, damage_max: 15, plates_lost: 0, plates_total: 3 } }, lead_A: 0, wave_status: '' };
}
function turn(t: number, a: number): Turn {
  return { t, k: t, codes: [a, 2], names: ['x', 'y'], parts: [{ kind: 'lane', wave: 0, behaviour: 0, objective: 0 }, { kind: 'lane', wave: 0, behaviour: 0, objective: 0 }],
    modes: [], ticks: [], outcomes: [], delta: { A: { gold: 0, farm: 0, xp: 0, hp: 0, hp_taken: 0 }, B: { gold: 0, farm: 0, xp: 0, hp: 0, hp_taken: 0 } }, reward_A: 0 };
}
const view = (sid: string, p: number): View => ({
  session: sid, kind: 'human_game', state: 'awaiting_action', stop_reason: null, position: p, next_turn: p + 1, turn_cap: 150, lane_exit_position: 94, read_position: 114,
  seats: [{ seat: 'A', kind: 'human' }, { seat: 'B', kind: 'model', id: 'sp:0', label: 'Pure self-play', arm: 'sp' }], belief_mode: 'hidden',
  seeds: { chance: null, chance_chosen: false, policy: null, policy_rule: 'drawn' }, current: position(p), marks: {}, human_seat: 'A', opponent_ready: true,
  menu: { legal: [2, 7], forced: null, away: null, waves: [], behaviours: [], objectives: [], singles: [], lane_codes: {} },
});
const delay = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** A fake server with one session per create; `slow` delays the next matching call. */
function fakeApi(): Api & { sessions: Record<string, { turns: Turn[] }>; slow: Record<string, number>; failNext: Record<string, Error> } {
  let n = 0;
  const sessions: Record<string, { turns: Turn[] }> = {};
  const slow: Record<string, number> = {};
  const failNext: Record<string, Error> = {};
  const gate = async (name: string) => {
    const ms = slow[name] ?? 0;
    if (ms) { delete slow[name]; await delay(ms); }
    const err = failNext[name];
    if (err) { delete failNext[name]; throw err; }
  };
  const api: Api = {
    status: async () => ({ ready: true, models: [], simulations: 'ready', replays: 0, sessions: 0 }),
    createSession: async (): Promise<CreateResponse> => {
      const sid = `s${++n}`.padEnd(32, '0');                 // numbered in the order the requests were made
      await gate('createSession');
      sessions[sid] = { turns: [] };
      return { session: sid, view: view(sid, 0), history: { session: sid, from: 0, to: 0, positions: [position(0)], turns: [] } };
    },
    view: async (sid) => { await gate('view'); if (!sessions[sid]) throw new ApiError(404, 'unknown_session', 'no such session'); return view(sid, sessions[sid].turns.length); },
    submitTurn: async (sid, t, code): Promise<TurnResult> => {
      await gate('submitTurn');
      const s = sessions[sid];
      if (!s) throw new ApiError(404, 'unknown_session', 'no such session');
      if (t === s.turns.length + 1) s.turns.push(turn(t, code));
      else if (t > s.turns.length + 1) throw new ApiError(409, 'turn_out_of_order', 'out of order');
      else if (s.turns[t - 1].codes[0] !== code) throw new ApiError(409, 'turn_conflict', `turn ${t} is already resolved with another action`);
      return { turn: s.turns[t - 1], position: position(t), view: view(sid, s.turns.length) };
    },
    history: async (sid, from = 0, to?): Promise<History> => {
      const s = sessions[sid];
      const end = to ?? s.turns.length;
      return { session: sid, from, to: end, positions: Array.from({ length: end - from + 1 }, (_, i) => position(from + i)), turns: s.turns.slice(from, end) };
    },
    stop: async () => { throw new Error('not used'); },
    record: async () => { throw new Error('not used'); },
    simulate: async () => { throw new Error('not used'); },
    replays: async () => [],
    replay: async () => { throw new Error('not used'); },
  };
  return Object.assign(api, { sessions, slow, failNext });
}

const params = () => ({ ...readParams(''), still: true });

describe('AppStore', () => {
  it('drops a late response from a game that was replaced', async () => {
    const api = fakeApi();
    const S = new AppStore(params(), api);
    api.slow.createSession = 60;
    const first = S.newGame('sp:0');                         // answers after 60 ms
    await delay(5);
    const second = S.newGame('om:0');                        // answers at once
    await Promise.all([first, second]);
    expect(S.live?.sid.startsWith('s2')).toBe(true);
    expect(S.loading).toBeNull();
    expect(S.game?.turns.length).toBe(0);
  });

  it('drops a late turn result after a new game started', async () => {
    const api = fakeApi();
    const S = new AppStore(params(), api);
    await S.newGame('sp:0');
    const a = S.live!.sid;
    api.slow.submitTurn = 60;
    const p = S.submit(7);
    await delay(5);
    await S.newGame('sp:0');
    await p;
    expect(S.live!.sid).not.toBe(a);
    expect(S.game!.turns.length).toBe(0);
    expect(S.ctrl!.progress).toBe(0);
    expect(api.sessions[a].turns.length).toBe(1);             // the old session did resolve it; the screen ignored it
    expect(S.busy).toBe(false);
  });

  it('re-reads the server after an unanswered submission and appends what it missed', async () => {
    const api = fakeApi();
    const S = new AppStore(params(), api);
    await S.newGame('sp:0');
    const sid = S.live!.sid;
    // the request reached the server, the answer was lost
    api.failNext.submitTurn = new ApiError(0, 'network', 'The server could not be reached.');
    api.sessions[sid].turns.push(turn(1, 7));
    await S.submit(7);
    expect(S.ctrl!.progress).toBe(1);
    expect(S.game!.turns[0].codes[0]).toBe(7);
    expect(S.problem).toBeNull();
    expect(S.ctrl!.canSubmit).toBe(true);
    // the request never reached the server: nothing changed, and a retry is offered
    api.failNext.submitTurn = new ApiError(0, 'timeout', 'The server did not answer in time.');
    await S.submit(2);
    expect(S.ctrl!.progress).toBe(1);
    expect(S.problem?.kind).toBe('unanswered');
    S.problem!.retry!();
    await delay(5);
    expect(S.ctrl!.progress).toBe(2);
    expect(S.problem).toBeNull();
  });

  it('reports an illegal action without changing anything, and a lost session as lost', async () => {
    const api = fakeApi();
    const S = new AppStore(params(), api);
    await S.newGame('sp:0');
    api.failNext.submitTurn = new ApiError(422, 'illegal_action', 'Recall needs you on your own side of the lane.');
    await S.submit(12);
    expect(S.error).toContain('Recall needs');
    expect(S.ctrl!.progress).toBe(0);
    expect(S.problem).toBeNull();
    delete api.sessions[S.live!.sid];
    await S.submit(2);
    expect(S.problem?.kind).toBe('lost');
    expect(S.ctrl!.progress).toBe(0);
  });

  it('keeps the selection rules: Pressure only with Hold or Trade', async () => {
    const S = new AppStore(params(), fakeApi());
    await S.newGame('sp:0');
    S.choose('w1'); S.choose('b3'); S.choose('p');
    expect(S.sel).toEqual({ kind: 'lane', wave: 1, beh: null, pressure: true });
    S.choose('b2');
    expect(S.sel.beh).toBe(2);
    S.choose('recall');
    expect(S.sel.kind).toBe('recall');
    expect(S.selectionCode()).toBe(12);
  });
});
