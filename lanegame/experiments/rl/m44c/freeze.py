"""Part 3: is maintaining a freeze worth more than breaking it, over the freeze's whole duration?

Freeze state (M4.4a §4.2, `m44.records.holds_freeze`): both fronts alive and fighting, the enemy front at the freezer's
own -1 or deeper (the freezer holds the enemy wave next to its own tower). The focal side is the FREEZER.

States
    held (NATURAL)   the focal side is present and has held the freeze for at least 2 consecutive turns (an active
                     freeze, not a one-turn touch; the run length so far is recorded, and the subset >= 4 turns, the
                     M4.4a "established" threshold, is reported separately)
    init (INITIALIZED) the M4.3c outside-tower freeze start (k 51; A holds B's 5-HP wave at A's -1, a 6-HP B mass under
                     A's tower, A in its tower cell, B at its +1) on a grid: power gap {equal, A x1.76 = stronger
                     freezer, B x1.76 = weaker freezer} x level {4, 10} x frozen-on HP {100%, 50%} x plates {left,
                     exhausted}; 24 states, outside the training distribution

Freezer plans (wave component only; behaviour and objective keep the policy's own conditional distribution given that
wave; every plan stops as soon as the freeze state ends, never forcing a wave action after it, and lasts at most 10
turns; an absent freezer ends it)
    policy          the learned policy
    Preserve        the documented threshold controller (reference §4.8, scripted freezers): Thin if the fronts are in
                    contact, the own front is at the own -1 or deeper and the enemy front has >= 14 HP, else Last Hit
    Preserve-Thin   Thin on every held turn (the four-action freeze keeper `thin4`: Thin restores window-strength parity)
    Break           Shove on every held turn (crash the held wave out of the freeze)
    Conditional     Preserve while the freeze holds AND the frozen-on champion cannot contest it cheaply (it is away,
                    or below 50% HP, or the freezer's HP-aware duel edge is positive); otherwise Shove (release it into
                    a crash). A transparent comparator, not a learned policy
Opponent (the frozen-on side): protocols policy / modal and the styles breaker (Shove while frozen on), passive (Last
Hit while frozen on), stepback (Safe while frozen on) of `opponents`.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ....champions_m4 import context
from ....config import B_SAFE, W_HOLD, W_SHOVE, W_THIN
from ....diagnostics.branch import FollowPolicy, observation
from ...m4.zoning_natural import duel_edge
from ..m44.actions import full_distribution, masses
from ..m44.records import holds_freeze, power_ratio
from . import plans as P
from .common import INIT_SEED, OFFSET, PRIMARY, STATE_SEED
from .runner import Experiment
from .states import collect, initialized

FREEZE_HP = 14
MAX_TURNS = 10
NAT_PROTOCOLS = ("policy", "modal", "breaker", "passive", "stepback")
INIT_PROTOCOLS = ("policy", "breaker", "passive", "stepback")
GRID = [(gap, level, hp, plates) for gap in ("equal", "large", "b_large") for level in (4, 10) for hp in (1.0, 0.5)
        for plates in ("left", "exhausted")]


# ---------------------------------------------------------------------------------------------- plans
def _held(obs) -> bool:
    return holds_freeze(obs.state, obs.side)


def _ended(obs, t, mem):
    return "freeze ended" if not _held(obs) else False


def controller_wave(obs) -> int:
    lane, s = obs.state, obs.side
    ofx, ofh = lane.side(s)[:2]
    efx, efh = lane.side(1 - s)[:2]
    contact = ofh > 0 and efh > 0 and ofx + efx >= -1
    return W_THIN if (contact and ofx <= -1 and efh >= FREEZE_HP) else W_HOLD


def favorable(obs) -> bool:
    """The frozen-on champion cannot contest the freeze cheaply: away, below 50% HP, or the freezer has the edge."""
    cfg, m4, s = obs.cfg, obs.champ, obs.side
    en = m4.side(1 - s)
    if en.absent > 0:
        return True
    t = context(cfg).t
    if en.hp <= 0.5 * t.hp_max[t.level_by_xp[min(en.xp, t.xp_cap)]]:
        return True
    return duel_edge(cfg, obs.state, m4, s) > 0


def preserve_rule(obs, t, mem):
    w = controller_wave(obs)
    return P.force({"wave": w}, "Preserve: " + ("Thin" if w == W_THIN else "Last Hit"))


def conditional_rule(obs, t, mem):
    if favorable(obs):
        return preserve_rule(obs, t, mem)
    return P.force({"wave": W_SHOVE}, "release: Shove")


def candidates(situation, cfg, snap, seat) -> Dict[str, object]:
    def plan(name, rule, desc):
        return P.StrategicPlan(name, rule, max_turns=MAX_TURNS, stop=_ended, description=desc)
    return {"policy": FollowPolicy(),
            "Preserve": plan("Preserve", preserve_rule, "threshold controller while held"),
            "Preserve-Thin": plan("Preserve-Thin", P.const_rule(P.force({"wave": W_THIN}, "Thin")), "Thin while held"),
            "Break": plan("Break", P.const_rule(P.force({"wave": W_SHOVE}, "Shove")), "Shove while held"),
            "Conditional": plan("Conditional", conditional_rule, "Preserve while favourable, else Shove")}


# ---------------------------------------------------------------------------------------------- states
def predicates(cfg):
    def held(d, t, s):
        return t > 0 and d["held"][t, s] > 0 and d["held"][t - 1, s] > 0
    return {"held": held}


def describe(snap, s: int, cfg) -> dict:
    ctx = context(cfg)
    t = ctx.t
    me, en = snap.champ.side(s), snap.champ.side(1 - s)
    ofx, ofh = snap.lane.side(s)[:2]
    efx, efh = snap.lane.side(1 - s)[:2]
    from ....progression import plates_broken
    n = len(cfg.plate_thresholds)
    return {"power": power_ratio(ctx, snap.champ, s), "edge": duel_edge(cfg, snap.lane, snap.champ, s),
            "hp": me.hp / t.hp_max[t.level_by_xp[min(me.xp, t.xp_cap)]],
            "ehp": en.hp / t.hp_max[t.level_by_xp[min(en.xp, t.xp_cap)]], "enemy_absent": en.absent > 0,
            "own_front": [int(ofx), int(ofh)], "enemy_front": [int(-efx), int(efh)], "en_siege": snap.lane.siege_of(1 - s),
            "level": t.level_by_xp[min(me.xp, t.xp_cap)], "plates_left": n - plates_broken(me.tower_dmg, t),
            "own_plates_left": n - plates_broken(en.tower_dmg, t)}


def states(cfg, agent, env, seed, phase) -> Dict[str, List[dict]]:
    return collect(cfg, agent, predicates(cfg), {"held": 24}, STATE_SEED + OFFSET["freeze"] + 100 * seed +
                   (0 if phase == PRIMARY else 50), game_turns=500, max_games=120, per_game=2, describe=describe)


def states_init(cfg, agent, env, seed, phase) -> Dict[str, List[dict]]:
    return {"init": init_states(cfg)}


def init_states(cfg) -> List[dict]:
    out = []
    for i, (gap, level, hp, plates) in enumerate(GRID):
        snap = initialized(cfg, "freeze", gap, level=level, hp=(1.0, hp))
        if plates == "exhausted":
            theta = cfg.plate_thresholds[-1]
            snap = snap.with_side(0, tower_dmg=theta).with_side(1, tower_dmg=theta)
        assert holds_freeze(snap.lane, 0)
        out.append({"snapshot": snap, "seat": 0, "turn": snap.k, "game": -1, "natural": False,
                    "start": f"freeze/{gap}/L{level}/hp{hp:.1f}/plates_{plates}",
                    "grid": {"gap": gap, "level": level, "opp_hp": hp, "plates": plates},
                    "row": {}, "desc": describe(snap, 0, cfg)})
    return out


def frequency(agent, cfg, snap, s) -> dict:
    m = masses(full_distribution(agent, observation(cfg, snap, s)))
    mo = masses(full_distribution(agent, observation(cfg, snap, 1 - s)))
    names = ("Last Hit", "Thin", "Slow Push", "Shove")
    out = {n: float(m["wave"][i]) for i, n in enumerate(names)}
    out.update({f"opp_{n}": float(mo["wave"][i]) for i, n in enumerate(names)})
    out["Recall"] = float(m["recall"])
    out["opp_Safe"] = float(mo["beh"][B_SAFE])
    out["opp_Recall"] = float(mo["recall"])
    return out


def extra(cfg):
    def fn(infos, f):
        o = 1 - f
        b = infos[0]["champ_before"]
        e = infos[min(39, len(infos) - 1)]["champ_after"]
        lead0 = (b[f][4] - b[o][4]) / cfg.gold_scale
        lead1 = (e[f][4] - e[o][4]) / cfg.gold_scale
        return {"inv_lead_change40": lead1 - lead0}
    return fn


def experiment(name: str) -> Experiment:
    if name == "freeze":
        sizes, protocols, nroll, fn = {"held": 24}, {"held": NAT_PROTOCOLS}, {"held": 24}, states
    elif name == "freeze_init":
        sizes, protocols, nroll, fn = {"init": len(GRID)}, {"init": INIT_PROTOCOLS}, {"init": 24}, states_init
    else:
        raise ValueError(name)
    return Experiment(name=name, offset="freeze", sizes=sizes, protocols=protocols, states=fn,
                      candidates=candidates, frequency=frequency, n_roll=nroll, ep_h=(5, 10, 20, 40),
                      track_freeze=True, extra=extra, initialized=("init",),
                      notes={"max_turns": MAX_TURNS, "freeze_hp": FREEZE_HP, "grid": GRID})


def run(envs=None, seeds=None, phases=(PRIMARY,), workers: int = 11, name: str = "freeze", protocols=None):
    from .runner import run as _run
    return _run("freeze", "freeze", envs, seeds, phases, workers, protocols, out_name=name)


def run_init(envs=None, seeds=None, workers: int = 11):
    from .runner import run as _run
    return _run("freeze", "freeze_init", envs, seeds, (PRIMARY,), workers)


__all__ = ["experiment", "run", "run_init", "candidates", "controller_wave", "favorable", "GRID"]
