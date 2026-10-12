"""M4 scenarios (plan WP8): the parked-front harness of design §6.4 and replayable witnesses of the ten design
scenarios (§15) and of the reachable situations the validation stages need (Freeze, Shove, XP denial, Recall, item
purchases, wave crashes, tower plates).

Parked-front harness. `hold_run(e, b_wave, phase, a_wave)` starts from design §6.4's configuration (the defender
A's wave just killed, B's front of e HP at A's -1, both champions present: A at its tower, B at its front) and plays
the M4 resolver without chance events: A Freezes (Hold, maintained) whenever Freeze is legal, or always plays Slow
Push; B plays Slow Push or Shove with Hold. It returns the turn of B's first crash into A's tower (None if none).

Witnesses. A scenario is a predicate on the environment state. `find_witnesses` plays seeded scripted pairings
from s0 and records, for every scenario, the first state satisfying it together with its witness (the environment
seed and the action prefix). `replay(cfg, witness)` reproduces the state exactly (lanegame.env.starts).
"""

from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .champions_m4 import M4Side, M4State, initial_m4, level, resolve_turn_m4
from .config import B_HOLD, F, GameConfig, SH, SP, champ_act
from .mechanics import wave_actions
from .progression import tables
from .state import State

A, B = 0, 1
WITNESS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "m4_scenarios.json")


# ---------------------------------------------------------------------- the parked front (design §6.4)
def parked_state(e: int, cfg: GameConfig) -> Tuple[State, M4State]:
    lane = State(0, 0, (), 0, 1, e, (), 0, 0, 0, 0, 0)
    t = tables(cfg)
    m4 = M4State(M4Side(-cfg.tower, t.hp_max[1]), M4Side(1, t.hp_max[1]))
    return lane, m4


def hold_run(e: int, b_wave: int, phase: int, a_wave: int, cfg: GameConfig, turns: int = 24,
             lane: Optional[State] = None, m4: Optional[M4State] = None) -> dict:
    """Design §6.4 measurement on the implemented resolver; returns first crash turn, Freeze-legal turns, held turns,
    maintained turns and the turns on which the hold was broken."""
    if lane is None:
        lane, m4 = parked_state(e, cfg)
    first, legal_turns, held, maintained, broken = None, 0, 0, 0, 0
    for i in range(turns):
        k = phase + i
        legal = F in wave_actions(lane, A, cfg)
        legal_turns += legal
        wa = F if (a_wave == F and legal) else SP
        t = resolve_turn_m4(lane, m4, k, champ_act(wa, B_HOLD), champ_act(b_wave, B_HOLD), cfg, u=None)
        held += t.lane.events["freeze_held"][B] > 0
        broken += t.lane.events["freeze_broken"][B] > 0
        maintained += t.freeze_maintained[A]
        if t.lane.crash_hp[B] > 0 and first is None:
            first = i
        lane, m4 = t.lane.state, t.champ
    return {"first_crash": first, "freeze_legal": legal_turns, "held": held, "maintained": maintained,
            "broken": broken}


# design §6.4 / results/m4_design/m4_hold.txt: first crash turn (None = never in 24 turns), per (e, B wave, phase),
# for A = Freeze and A = Slow Push (lane-only prototype)
DESIGN_HOLD_TABLE = {
    # (e, wave): ([Freeze by phase 0..3], [Slow Push by phase 0..3])
    (2, SP): ([None] * 4, [None] * 4), (3, SP): ([None] * 4, [None] * 4), (4, SP): ([None] * 4, [None] * 4),
    (5, SP): ([None] * 4, [0, None, None, None]),
    (6, SP): ([12, 11, 10, 9], [0, 15, None, None]), (7, SP): ([8, 7, 10, 9], [0, 4, 14, 13]),
    (8, SP): ([8, 7, 6, 5], [0, 4, 3, 2]), (9, SP): ([4, 3, 6, 5], [0, 3, 3, 2]),
    (2, SH): ([None] * 4, [0] * 4), (3, SH): ([6, 5, 6, 5], [0] * 4), (4, SH): ([4, 3, 4, 3], [0] * 4),
    (5, SH): ([4, 3, 4, 3], [0] * 4), (6, SH): ([0] * 4, [0] * 4), (7, SH): ([0] * 4, [0] * 4),
    (8, SH): ([0] * 4, [0] * 4), (9, SH): ([0] * 4, [0] * 4),
}


# ---------------------------------------------------------------------- scenario predicates (A's perspective)
def _present(m4: M4State, s: int) -> bool:
    return m4.side(s).absent == 0


def _lv(m4, s, t):
    return level(m4.side(s), t)


def _freeze_legal(lane, s, cfg) -> bool:
    return F in wave_actions(lane, s, cfg)


def _enemy_front_at(lane, s):
    """(cell in s's own coordinates, HP) of the enemy front, or None."""
    fx, fh = lane.side(1 - s)[:2]
    return (-fx, fh) if fh > 0 else None


def scenario_predicates(cfg: GameConfig) -> Dict[str, Callable]:
    t = tables(cfg)
    T = cfg.tower

    def equal_power_freeze(lane, m4, ph):
        ef = _enemy_front_at(lane, A)
        return (_present(m4, A) and _present(m4, B) and ef is not None and ef[0] == -1 and 3 <= ef[1] <= 5
                and m4.a.pos in (-2, -1) and abs(_lv(m4, A, t) - _lv(m4, B, t)) <= 1 and m4.a.items == m4.b.items
                and _lv(m4, B, t) < t.level_cap)

    def strong_freezer(lane, m4, ph):
        return (_present(m4, A) and _present(m4, B) and _freeze_legal(lane, A, cfg)
                and _lv(m4, A, t) >= _lv(m4, B, t) + 2 and m4.a.items >= m4.b.items + 1)

    def weak_breaker(lane, m4, ph):
        ef = _enemy_front_at(lane, A)
        return (_present(m4, A) and _present(m4, B) and ef is not None and ef[0] == -1 and ef[1] >= 3
                and (_lv(m4, A, t) > _lv(m4, B, t) or m4.a.items > m4.b.items))

    def skill_vs_economy(lane, m4, ph):
        return (_present(m4, A) and _present(m4, B) and _lv(m4, B, t) >= _lv(m4, A, t) + 2
                and m4.b.items >= m4.a.items + 1)

    def low_hp_recall_to_buy(lane, m4, ph):
        me = m4.a
        return (_present(m4, A) and me.pos <= cfg.recall_line and me.hp <= t.hp_max[_lv(m4, A, t)] // 3
                and me.items < t.item_tiers and me.gold >= t.prices[me.items])

    def recall_cheap(lane, m4, ph):
        me = m4.a
        own = lane.side(A)
        return (_present(m4, A) and _present(m4, B) and me.pos <= cfg.recall_line and own[1] > 0 and own[0] >= 1
                and me.items < t.item_tiers and me.gold >= t.prices[me.items])

    def recall_expensive(lane, m4, ph):
        me = m4.a
        ef = _enemy_front_at(lane, A)
        return (_present(m4, A) and _present(m4, B) and me.pos <= cfg.recall_line and ef is not None
                and ef[0] == -1 and ef[1] >= 5 and me.hp >= t.hp_max[_lv(m4, A, t)] * 2 // 3)

    def crash_bounce_plates(lane, m4, ph):
        return lane.asg > 0 and m4.a.tower_dmg >= 800 and m4.a.tower_dmg < cfg.plate_thresholds[-1]

    def levelup_in_combat(lane, m4, ph):
        L = _lv(m4, A, t)
        me = m4.a
        return (_present(m4, A) and _present(m4, B) and L < t.level_cap and t.xp_cum[L + 1] - me.xp <= 8
                and me.hp <= t.hp_max[L] // 4 and abs(me.pos + m4.b.pos) <= cfg.trade_range + 1)

    def beyond_turn_32(lane, m4, ph):          # needs the turn index: checked by the search loop (k >= 64)
        return _present(m4, A) and _present(m4, B)

    def caps(lane, m4, ph):
        return all(_lv(m4, s, t) == t.level_cap and m4.side(s).items == t.item_tiers for s in (A, B))

    def xp_denial(lane, m4, ph):
        # B pushed back to its own tower cell while A's minions die near A's tower: A threatens from the lane
        return (_present(m4, A) and _present(m4, B) and m4.b.pos == -T and m4.a.pos >= 0
                and _freeze_legal(lane, A, cfg))

    def item_purchase(lane, m4, ph):
        me = m4.a
        return me.absent == 1 and me.items < t.item_tiers and me.gold >= t.prices[me.items]

    def wave_crash(lane, m4, ph):
        return lane.asg >= 8

    def tower_plate(lane, m4, ph):
        return lane.asg > 0 and any(0 < th - m4.a.tower_dmg <= 200 for th in cfg.plate_thresholds)

    return {"1_equal_power_freeze": equal_power_freeze, "2_strong_freezer": strong_freezer,
            "3_weak_breaker": weak_breaker, "4_skill_vs_economy": skill_vs_economy,
            "5_low_hp_recall_to_buy": low_hp_recall_to_buy, "6a_recall_cheap": recall_cheap,
            "6b_recall_expensive": recall_expensive, "7_crash_bounce_plates": crash_bounce_plates,
            "8_levelup_in_combat": levelup_in_combat, "9_beyond_turn_32": beyond_turn_32, "10_caps": caps,
            "xp_denial": xp_denial, "item_purchase": item_purchase, "wave_crash": wave_crash,
            "tower_plate": tower_plate}


SCENARIO_CONFIGS = {"4_skill_vs_economy": {"skill_a": 0.7, "skill_b": 0.3}}
MIN_TURN = {"9_beyond_turn_32": 64}


def scenario_config(name: str, base: Optional[GameConfig] = None) -> GameConfig:
    base = base if base is not None else GameConfig.m4()
    kw = SCENARIO_CONFIGS.get(name)
    return base.with_(**kw) if kw else base


# ---------------------------------------------------------------------- witness search
def find_witnesses(base: Optional[GameConfig] = None, games: int = 400, turns: int = 400, seed: int = 20261011,
                   names: Optional[Sequence[str]] = None) -> Dict[str, dict]:
    """Seeded scripted pairings from s0; the first state satisfying each scenario, with its witness."""
    from .env import LaneEnv, UniformAgent
    from .env.agents_m4 import gate_attackers, gate_defenders, m4_suite
    base = base if base is not None else GameConfig.m4()
    pool = list(m4_suite().values()) + list(gate_defenders().values()) + list(gate_attackers().values())
    pool.append(UniformAgent())
    names = list(names) if names is not None else list(scenario_predicates(base))
    found: Dict[str, dict] = {}
    rng = np.random.default_rng(seed)
    for cfg_key in sorted({json.dumps(SCENARIO_CONFIGS.get(n, {}), sort_keys=True) for n in names}):
        kw = json.loads(cfg_key)
        cfg = base.with_(**kw) if kw else base
        todo = [n for n in names if json.dumps(SCENARIO_CONFIGS.get(n, {}), sort_keys=True) == cfg_key]
        preds = {n: scenario_predicates(cfg)[n] for n in todo}
        env = LaneEnv(cfg, max_turns=turns, cache_size=20_000)
        for g in range(games):
            if all(n in found for n in todo):
                break
            x, y = pool[int(rng.integers(len(pool)))], pool[int(rng.integers(len(pool)))]
            env_seed = int(rng.integers(2 ** 31))
            arng = np.random.default_rng(env_seed + 1)
            obs = env.reset(seed=env_seed)
            if hasattr(x, "reset"):
                x.reset()
            if hasattr(y, "reset"):
                y.reset()
            actions: List[List[int]] = []
            while True:
                k = env.k
                for n in todo:
                    if n not in found and k >= MIN_TURN.get(n, 0) and preds[n](env.state, env.champ, k % cfg.phase_period):
                        found[n] = {"config": kw, "witness": {"seed": env_seed, "actions": [list(a) for a in actions]},
                                    "agents": [x.name, y.name], "k": k,
                                    "lane": env.state.to_list(), "champ": env.champ.to_list(),
                                    "phase": k % cfg.phase_period}
                a, b = x.act(obs[0], arng), y.act(obs[1], arng)
                r = env.step(a, b)
                for ag, s in ((x, 0), (y, 1)):
                    if hasattr(ag, "notify"):
                        ag.notify(r.info, s)
                actions.append([int(a), int(b)])
                if r.done:
                    break
                obs = r.obs
    return found


def save_witnesses(found: Dict[str, dict], path: str = WITNESS_FILE) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"schema": "lanegame.m4_scenarios", "version": 1, "scenarios": found}, f, indent=1)


def load_witnesses(path: str = WITNESS_FILE) -> Dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    assert d["schema"] == "lanegame.m4_scenarios"
    return d["scenarios"]


def replay(name: str, rec: dict, base: Optional[GameConfig] = None):
    """The environment in the witnessed state of scenario `name` (raises if the witness does not reproduce it)."""
    from .env.starts import replay_witness
    cfg = scenario_config(name, base)
    env = replay_witness(cfg, rec["witness"])
    if env.state.to_list() != rec["lane"] or env.champ.to_list() != rec["champ"] or env.k != rec["k"]:
        raise AssertionError(f"witness of {name} does not reproduce its state")
    return env
