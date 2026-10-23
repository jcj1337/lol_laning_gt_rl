// The store against a fake server: a late response from an earlier game never acts on the new one, an unanswered
// submission re-reads the server and appends what it missed (or offers a retry when nothing changed), a failed stop at
// the turn cap offers a retry, and a lost session is reported as such. A sequence walks to its events on the controller
// (Next and Play stop at each one), goes with the next game, and a sequence opened over a live game leaves the way back
// to it.
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ApiError } from '../api/client';
import type { Api } from '../api/client';
import { WALK_MS } from './frames';
import { AppStore, readParams } from './store';
import type { CreateResponse, GameBundle, Guide, History, Position, Recording, StopResponse, Turn, TurnResult, View } from './types';

function position(p: number): Position {
  const champ = { level: 1, hp: 600, hp_max: 600, xp: 0, xp_level: 0, xp_next: 10, gold_earned: 0, gold_unspent: 0, invested: 0, attack_power: 1,
    status: 'present' as const, returns_in: 0, cell: -2, at_ward: false, vision: 0, capacity: 4, wave_budgets: {} };
  return { p, k: p, lane: { packets: { A: [], B: [] }, siege: { A: [-2, 0], B: [2, 0] }, contact: null }, champions: { A: champ, B: champ },
    towers: { A: { damage: 0, damage_max: 15, plates_lost: 0, plates_total: 3 }, B: { damage: 0, damage_max: 15, plates_lost: 0, plates_total: 3 } }, lead_A: 0, wave_status: '' };
}
function turn(t: number, a: number): Turn {
  return { t, k: t, codes: [a, 2], names: ['x', 'y'], parts: [{ kind: 'lane', wave: 0, behaviour: 0, objective: 0 }, { kind: 'lane', wave: 0, behaviour: 0, objective: 0 }],
    modes: [], ticks: [], outcomes: [], delta: { A: { gold: 0, farm: 0, xp: 0, hp: 0, hp_taken: 0 }, B: { gold: 0, farm: 0, xp: 0, hp: 0, hp_taken: 0 } }, reward_A: 0,
    moves: { A: { champion: null, wave: null }, B: { champion: null, wave: null } } };
}
const view = (sid: string, p: number): View => ({
  session: sid, kind: 'human_game', state: 'awaiting_action', stop_reason: null, position: p, next_turn: p + 1, turn_cap: 150, lane_exit_position: 94, read_position: 114,
  seats: [{ seat: 'A', kind: 'human' }, { seat: 'B', kind: 'model', id: 'sp:0', label: 'Pure self-play', arm: 'sp' }], belief_mode: 'hidden',
  seeds: { chance: null, chance_chosen: false, policy: null, policy_rule: 'drawn' }, current: position(p), marks: {}, human_seat: 'A', opponent_ready: true,
  menu: { legal: [2, 7], forced: null, away: null, waves: [], behaviours: [], objectives: [], singles: [], lane_codes: {} },
});
const delay = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** A bot game of 30 turns as the server bundles it (the store never reads the recording). */
function bundle(name: string, kind: 'research_record' | 'new_simulation'): GameBundle {
  return {
    name, label: name, positions: Array.from({ length: 31 }, (_, p) => position(p)), turns: Array.from({ length: 30 }, (_, i) => turn(i + 1, 2)),
    beliefs: null, lane_exit_position: 94, read_position: 114, recording: {} as GameBundle['recording'],
    meta: { provenance: { kind }, seats: [{ seat: 'A', kind: 'model', id: 'om:0', label: 'Adaptive', arm: 'om' }, { seat: 'B', kind: 'model', id: 'sp:0', label: 'Self-play', arm: 'sp' }],
      seeds: { chance: 0, chance_chosen: true, policy: 0, policy_rule: 'fixed' }, marks: {}, belief_mode: 'hidden', turn_cap: 30,
      status: { state: 'stopped', stop_reason: 'turn_cap', turns: 30 }, lead_A: 0, config: {}, policy_rule: 'fixed' },
  };
}
const GUIDE: Guide = {
  id: 'g', title: 'A sequence', intro: 'The introduction.', replay: 'r', start: 5, end: 20,
  checkpoints: [{ id: 'a', position: 8, title: 'First', text: 'One.' }, { id: 'b', position: 12, title: 'Second', text: 'Two.', highlight: { kind: 'wave', seat: 'B' } },
    { id: 'c', position: 20, title: 'Third', text: 'Three.' }],
};

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
    status: async () => ({ ready: true, models: [], simulations: 'ready', replays: 0, sessions: 0, guides: 0 }),
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
    simulate: async () => bundle('sim', 'new_simulation'),
    replays: async () => [],
    replay: async (name) => bundle(name, 'research_record'),
    guides: async () => [{ id: GUIDE.id, title: GUIDE.title, replay: GUIDE.replay, checkpoints: GUIDE.checkpoints.length }],
    guide: async (id) => { if (id !== GUIDE.id) throw new ApiError(404, 'unknown_guide', `no sequence named '${id}'`); return GUIDE; },
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

  it('offers a retry when the stop at the turn cap fails, and the retry shows the recording', async () => {
    const api = fakeApi();
    const S = new AppStore(params(), api);
    await S.newGame('sp:0');
    const submit = api.submitTurn;
    const stopped = (v: View): View => ({ ...v, state: 'stopped', stop_reason: 'turn_cap', menu: null });
    api.submitTurn = async (sid, t, code) => { const r = await submit(sid, t, code); return { ...r, view: stopped(r.view) }; };
    const recording = { turns: [[7, 2, 'digest']] } as unknown as Recording;
    let stops = 0;
    api.stop = async (sid) => {
      if (++stops === 1) throw new ApiError(0, 'timeout', 'The server did not answer in time.');
      return { recording, view: stopped(view(sid, 1)) } as unknown as StopResponse;
    };
    await S.submit(7);
    expect(S.ctrl!.progress).toBe(1);
    expect(S.ctrl!.stopped).toBe(true);
    expect(S.problem?.kind).toBe('unanswered');
    expect(S.game!.recording).toBeUndefined();
    expect(S.busy).toBe(false);
    S.problem!.retry!();
    await delay(5);
    expect(stops).toBe(2);
    expect(S.problem).toBeNull();
    expect(S.game!.recording).toBe(recording);
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

describe('sequences', () => {
  afterEach(() => { vi.useRealTimers(); });

  it('opens from ?sequence=<id> in Watch, and ?guide= is not an address', () => {
    expect(readParams('?sequence=s')).toMatchObject({ mode: 'watch', guide: 's' });
    expect(readParams('?guide=s')).toMatchObject({ mode: 'play', guide: null });
  });

  it('walks to the events on the replay and knows when the view has left one', async () => {
    vi.useFakeTimers();
    const S = new AppStore(params(), fakeApi());
    await S.openGuide('g');
    expect(S.mode).toBe('watch');
    expect(S.game?.name).toBe('r');
    expect(S.ctrl!.view).toBe(5);
    expect(S.ctrl!.playing).toBe(false);
    expect(S.guide).toMatchObject({ at: null, notes: true });
    expect(S.guideAtEvent()).toBe(true);
    for (const [at, p] of [[0, 8], [1, 12], [2, 20]]) {
      S.guideNext();
      expect(S.guide!.at).toBe(at);
      vi.runAllTimers();
      expect(S.ctrl!.view).toBe(p);
      expect(S.guideAtEvent()).toBe(true);
    }
    S.guideNext();                                            // nothing past the last event
    expect(S.guide!.at).toBe(2);
    expect(S.ctrl!.playing).toBe(false);
    expect(S.guideHighlight()).toBeNull();
    S.guidePrev();                                            // back is a jump, not a walk
    expect(S.ctrl!.view).toBe(12);
    expect(S.ctrl!.playing).toBe(false);
    expect(S.guideHighlight()).toEqual({ kind: 'wave', seat: 'B' });
    S.ctrl!.goto(13);                                         // off the event: the note stays, the ring goes
    expect(S.guide!.at).toBe(1);
    expect(S.guideAtEvent()).toBe(false);
    expect(S.guideHighlight()).toBeNull();
    S.guideReturn();
    expect(S.ctrl!.view).toBe(12);
    expect(S.guideHighlight()).toEqual({ kind: 'wave', seat: 'B' });
    S.guidePrev();
    S.ctrl!.goto(9);
    expect(S.guide!.at).toBe(0);
    expect(S.guideAtEvent()).toBe(false);
    S.guideReturn();
    expect(S.ctrl!.view).toBe(8);
    S.guidePrev();
    expect(S.guide!.at).toBeNull();
    expect(S.ctrl!.view).toBe(5);
  });

  it('Next event plays the turns up to the event, with the note waiting until the view gets there', async () => {
    vi.useFakeTimers();
    const S = new AppStore(params(), fakeApi());
    await S.openGuide('g');
    S.guideNext();
    expect(S.guide!.at).toBe(0);
    expect(S.ctrl!.view).toBe(5);
    expect(S.guideWalking()).toBe(true);
    expect(S.guideAtEvent()).toBe(false);
    expect(S.guideHighlight()).toBeNull();
    const seen: number[] = [];
    while (S.ctrl!.playing) { vi.advanceTimersByTime(WALK_MS); seen.push(S.ctrl!.view); }
    expect(seen).toEqual([6, 7, 8]);
    expect(S.guideWalking()).toBe(false);
    expect(S.guideAtEvent()).toBe(true);
  });

  it('Play walks to the next event and stops there; pressed during the walk it pauses', async () => {
    vi.useFakeTimers();
    const S = new AppStore(params(), fakeApi());
    await S.openGuide('g');
    S.guideSelect(0);
    expect(S.ctrl!.view).toBe(8);
    S.togglePlay();
    expect(S.guide!.at).toBe(1);
    expect(S.guideWalking()).toBe(true);
    vi.advanceTimersByTime(WALK_MS * 4);
    expect(S.ctrl!.view).toBe(12);
    expect(S.ctrl!.playing).toBe(false);
    expect(S.guideAtEvent()).toBe(true);
    vi.advanceTimersByTime(WALK_MS * 10);
    expect(S.ctrl!.view).toBe(12);
    S.togglePlay();
    expect(S.guide!.at).toBe(2);
    vi.advanceTimersByTime(WALK_MS * 2);
    expect(S.ctrl!.view).toBe(14);
    expect(S.key({ key: ' ' } as KeyboardEvent)).toBe(true);   // Space is the same button
    expect(S.ctrl!.playing).toBe(false);
    expect(S.guideWalking()).toBe(false);
    vi.advanceTimersByTime(WALK_MS * 10);
    expect(S.ctrl!.view).toBe(14);
    expect(S.guide!.at).toBe(2);
  });

  it('Play past the last event walks on to the end of the recording, the sequence still open', async () => {
    vi.useFakeTimers();
    const S = new AppStore(params(), fakeApi());
    await S.openGuide('g');
    S.guideSelect(2);
    S.togglePlay();
    expect(S.ctrl!.playing).toBe(true);
    expect(S.ctrl!.walkingTo).toBe(30);
    expect(S.guideWalking()).toBe(false);
    vi.advanceTimersByTime(WALK_MS * 10);
    expect(S.ctrl!.view).toBe(30);
    expect(S.ctrl!.playing).toBe(false);
    expect(S.guide).toMatchObject({ at: 2, notes: true });
    S.togglePlay();                                           // "Play from the start": the sequence's, from its introduction
    expect(S.ctrl!.view).toBe(5);
    expect(S.guide!.at).toBe(0);
    expect(S.guideWalking()).toBe(true);
    vi.advanceTimersByTime(WALK_MS * 3);
    expect(S.ctrl!.view).toBe(8);
    expect(S.guideAtEvent()).toBe(true);
  });

  it('Play without a sequence is ordinary playback', async () => {
    vi.useFakeTimers();
    const S = new AppStore(params(), fakeApi());
    await S.watch('x');
    S.togglePlay();
    expect(S.ctrl!.playing).toBe(true);
    expect(S.ctrl!.walkingTo).toBeNull();
    expect(S.ctrl!.view).toBe(1);
    S.togglePlay();
    expect(S.ctrl!.playing).toBe(false);
  });

  it('hides the notes, and drops the sequence when another game is installed', async () => {
    const S = new AppStore(params(), fakeApi());
    await S.openGuide('g');
    S.guideSelect(1);
    expect(S.guideHighlight()).not.toBeNull();
    S.setGuideNotes(false);
    expect(S.guide).toMatchObject({ at: 1, notes: false });
    expect(S.guideHighlight()).toBeNull();
    await S.watch('other');
    expect(S.guide).toBeNull();
    expect(S.game?.name).toBe('other');
    await S.openGuide('g');
    await S.simulate('om:0', 'sp:0');
    expect(S.guide).toBeNull();
    expect(S.game?.meta.provenance.kind).toBe('new_simulation');
    await S.openGuide('g');
    await S.newGame('sp:0');
    expect(S.guide).toBeNull();
    expect(S.mode).toBe('play');
  });

  it("reports an unknown sequence as an error in the server's words, and changes nothing else", async () => {
    const S = new AppStore(params(), fakeApi());
    await S.openGuide('g');
    S.guideSelect(0);
    await S.openGuide('nope');
    expect(S.problem?.kind).toBe('error');
    expect(S.problem?.message).toBe("no sequence named 'nope'");
    expect(S.loading).toBeNull();
    expect(S.game?.name).toBe('r');
    expect(S.ctrl!.view).toBe(8);
    expect(S.guide).toMatchObject({ at: 0 });
  });

  it('returns to the live game a sequence was opened over, or offers a new one', async () => {
    const api = fakeApi();
    const S = new AppStore(params(), api);
    S.engine = await api.status();
    await S.newGame('sp:0');
    const sid = S.live!.sid;
    await S.openGuide('g');
    expect(S.mode).toBe('watch');
    expect(S.live).toBeNull();
    expect(api.sessions[sid]).toBeDefined();                  // the session is still on the server
    await S.returnToPlay();
    expect(S.mode).toBe('play');
    expect(S.live?.sid).toBe(sid);
    expect(S.ctrl!.canSubmit).toBe(true);
    expect(S.guide).toBeNull();
    expect(S.dialog).toBeNull();
    const T = new AppStore(params(), fakeApi());
    T.engine = await api.status();
    await T.openGuide('g');
    await T.returnToPlay();
    expect(T.mode).toBe('watch');
    expect(T.dialog).toBe('newGame');
  });
});
