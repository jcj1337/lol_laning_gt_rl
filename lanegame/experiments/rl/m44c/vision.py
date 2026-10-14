"""Part 5: is Vision a profitable multi-turn investment before sustained Pressure?

Mechanics (reference §6.4, §8.4). A siege objective is legal while the own crashed minions are under the enemy tower and
the champion can reach the enemy tower cell this turn; it moves the champion to the enemy tower cell (+2, gank exposure
+2). Pressure adds floor(400 AP*) centi-units of tower damage (plates at 1,600 / 3,600 / 6,000, 1.0 resource each).
Vision sets a 6-turn timer; on the following 6 turns (not the current one) the side's gank OCCURRENCE is multiplied by
0.25. Vision has no other effect; it costs the turn's Pressure.

States
    plates (NATURAL)     siege objective legal, focal present, no active vision, enemy plates left
    exhausted (NATURAL)  the same with the enemy plates exhausted (Pressure then only damages a 6,000-clamped tower)
    active (NATURAL)     siege objective legal with vision already active (Vision first is mostly redundant here)
    init (INITIALIZED)   the M4.3c siege start (k 24, level 7, A's 14-HP crash under B's tower, A in B's tower cell)
                         with A's power {equal, x1.76} and A's HP {100%, 40%}

Plans (at most 12 turns; every plan stops as soon as no siege objective is legal: the siege is over or out of reach;
the behaviour and wave keep the policy's own conditional distribution)
    policy            the learned policy
    Pressure          Pressure on every legal turn
    Vision>Pressure   Vision on the branch turn, then Pressure on every legal turn
    Conditional       Vision first only if no vision is active and the own crashed mass is >= 12 HP (a siege that
                      lasts several turns: the tower removes 6 HP per turn), else Pressure at once; then Pressure
    No objective      no siege objective while the siege lasts (the champion farms / defends instead)
Opponent protocols: policy and modal.
"""

from __future__ import annotations

from typing import Dict, List

from ....champions import objective_reachable
from ....champions_m4 import context
from ....config import NONE, PRESSURE, VISION
from ....diagnostics.branch import FollowPolicy, observation
from ....state import State
from ..m44.actions import full_distribution, masses
from ..m44.records import power_ratio
from . import plans as P
from .common import OFFSET, PRIMARY, STATE_SEED
from .runner import Experiment
from .states import collect, initialized

MAX_TURNS = 12
LONG_SIEGE_HP = 12
INIT_GRID = [(gap, hp) for gap in ("equal", "large") for hp in (1.0, 0.4)]


def _obj_legal(obs) -> bool:
    return objective_reachable(obs.state, obs.champ, obs.side, obs.cfg)


def _over(obs, t, mem):
    return "siege over" if not _obj_legal(obs) else False


PRESS = P.force({"objective": PRESSURE}, "Pressure")
VIS = P.force({"objective": VISION}, "Vision")


def conditional_first(obs) -> P.Decision:
    long_siege = obs.state.siege_of(obs.side) >= LONG_SIEGE_HP
    return VIS if (obs.state.vision_of(obs.side) == 0 and long_siege) else PRESS


def candidates(situation, cfg, snap, seat) -> Dict[str, object]:
    def plan(name, rule):
        return P.StrategicPlan(name, rule, max_turns=MAX_TURNS, stop=_over)
    return {"policy": FollowPolicy(),
            "Pressure": plan("Pressure", P.const_rule(PRESS)),
            "Vision>Pressure": plan("Vision>Pressure", P.first_then(VIS, PRESS)),
            "Conditional": plan("Conditional", lambda obs, t, mem: conditional_first(obs) if t == 0 else PRESS),
            "No objective": plan("No objective", P.const_rule(P.force({"objective": NONE}, "no objective")))}


def describe(snap, s: int, cfg) -> dict:
    ctx = context(cfg)
    t = ctx.t
    me, en = snap.champ.side(s), snap.champ.side(1 - s)
    from ....progression import plates_broken
    n = len(cfg.plate_thresholds)
    return {"siege_hp": snap.lane.siege_of(s), "vision": snap.lane.vision_of(s), "pos": me.pos,
            "hp": me.hp / t.hp_max[t.level_by_xp[min(me.xp, t.xp_cap)]], "power": power_ratio(ctx, snap.champ, s),
            "level": t.level_by_xp[min(me.xp, t.xp_cap)], "plates_left": n - plates_broken(me.tower_dmg, t),
            "tower_dmg": me.tower_dmg, "enemy_absent": en.absent > 0}


def predicates(cfg):
    base = lambda d, t, s: d["obj_legal"][t, s] > 0
    return {"plates": lambda d, t, s: base(d, t, s) and d["vision"][t, s] == 0 and d["plates_left"][t, s] > 0,
            "exhausted": lambda d, t, s: base(d, t, s) and d["vision"][t, s] == 0 and d["plates_left"][t, s] == 0,
            "active": lambda d, t, s: base(d, t, s) and d["vision"][t, s] > 0}


SIZES = {"plates": 24, "exhausted": 24, "active": 12}


def states(cfg, agent, env, seed, phase) -> Dict[str, List[dict]]:
    out = collect(cfg, agent, predicates(cfg), SIZES, STATE_SEED + OFFSET["vision"] + 100 * seed +
                  (0 if phase == PRIMARY else 50), game_turns=500, max_games=120, per_game=2, describe=describe)
    return out


def states_init(cfg, agent, env, seed, phase) -> Dict[str, List[dict]]:
    out = []
    for gap, hp in INIT_GRID:
        snap = initialized(cfg, "siege", gap, hp=(hp, 1.0))
        out.append({"snapshot": snap, "seat": 0, "turn": snap.k, "game": -1, "natural": False,
                    "start": f"siege/{gap}/hpA{hp:.1f}", "grid": {"gap": gap, "hp": hp}, "row": {},
                    "desc": describe(snap, 0, cfg)})
    return {"init": out}


def frequency(agent, cfg, snap, s) -> dict:
    m = masses(full_distribution(agent, observation(cfg, snap, s)))
    return {"none": float(m["obj"][0]), "Pressure": float(m["obj"][1]), "Vision": float(m["obj"][2]),
            "Recall": float(m["recall"]), "Trade": float(m["beh"][2])}


def extra(cfg):
    def fn(infos, f):
        H = min(20, len(infos))
        prot = siege = legal = 0
        for inf in infos[:H]:
            before = State.from_list(inf["state_before"])
            pressed = inf["effective_objectives"][f] == PRESSURE
            prot += int(pressed and before.vision_of(f) > 0)
            siege += int(before.siege_of(f) > 0)
        return {"press_protected20": float(prot), "siege_turns20": float(siege)}
    return fn


def experiment(name: str) -> Experiment:
    common = dict(offset="vision", candidates=candidates, frequency=frequency, ep_h=(5, 10, 20, 40), extra=extra,
                  notes={"max_turns": MAX_TURNS, "long_siege_hp": LONG_SIEGE_HP, "init_grid": INIT_GRID})
    if name == "vision":
        return Experiment(name=name, sizes=dict(SIZES), states=states,
                          protocols={"plates": ("policy", "modal"), "exhausted": ("policy",), "active": ("policy",)},
                          n_roll={k: 24 for k in SIZES}, **common)
    if name == "vision_init":
        return Experiment(name=name, sizes={"init": len(INIT_GRID)}, states=states_init,
                          protocols={"init": ("policy", "modal")}, n_roll={"init": 48}, initialized=("init",),
                          chunk=4, **common)
    raise ValueError(name)


def run(envs=None, seeds=None, workers: int = 11):
    from .runner import run as _run
    a = _run("vision", "vision", envs, seeds, (PRIMARY,), workers)
    _run("vision", "vision_init", envs, seeds, (PRIMARY,), workers)
    return a


__all__ = ["experiment", "run", "candidates", "conditional_first", "INIT_GRID", "SIZES"]
