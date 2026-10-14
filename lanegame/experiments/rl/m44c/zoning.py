"""Part 4: why does the stronger champion not convert its advantage into resource denial?

The focal side is the STRONGER champion (seat A in the initialized states). Zoning in this game can only work through
the threat of trades (reference §5.3: an opponent's presence never reduces collection; the defender loses gold only to
reach, the Safe / retreat posture, trading, the own-tower rate and absence), so the experiments measure what the threat
actually costs the defender, never "chose Safe" alone.

States
    init (INITIALIZED)  the M4.3c harness starts `neutral` (k 9, level 3, both champions and fronts at the lane cell)
                        and `freeze` (k 51, level 10, A holds B's wave at A's -1) with A's power lead {equal, x1.24
                        (16 vs 4 invested), x1.76 (50 invested), x2.04 (50 invested and +3 levels)} and B's HP
                        {100%, 50%}: 16 states, outside the training distribution
    edge (NATURAL)      the focal side has a large HP-aware duel edge (zoning_natural.duel_edge >= 0.15: its expected
                        exchange takes a much larger share of the enemy's current HP than the enemy's of its own), both
                        champions present, the enemy outside its own tower cell and within catch distance (|pos + enemy
                        pos| <= 3). HP gaps are common in natural play, attack-power gaps are not (M4.4b §5.5)
    matrix (INITIALIZED) neutral starts {equal, x1.76, x2.04} x B's HP {100%, 50%}; both behaviours fixed for 10 turns
                        (waves: each side's own conditional choice), the behaviour matrix of the mechanical diagnosis

Stronger-side plans (20 turns at most, then the policy; behaviour / wave as stated, the other components the policy's
own conditional distribution)
    policy          the learned policy
    Farm            Last Hit + Hold (passive farming)
    Threaten        Trade when the enemy can be caught this turn without a dive and the duel edge is positive (the
                    M4.3c "hunt" rule), else Hold
    Bully           Trade every turn the enemy is present (dives included)
    Freeze+threat   the freeze controller's wave (Thin / Last Hit: keeps the wave on the own side) with the Threaten
                    behaviour
    Shove+Pressure  Shove, with Pressure whenever it is legal (pushing for plates instead of denial)
Weaker-side responses (its continuation agent): policy, ignore (Hold while the enemy is within reach and own HP > 40%),
respect (Safe while the enemy is within reach), shover (Shove every turn), ref:respect4 (the scripted conceding farmer).
"""

from __future__ import annotations

from typing import Dict, List

from ....champions_m4 import context
from ....config import B_HOLD, B_SAFE, B_TRADE, B_YIELD, PRESSURE, W_HOLD, W_SHOVE
from ....diagnostics.branch import FollowPolicy, observation
from ...m43c.strategies import behavior_rule
from ...m4.zoning_natural import duel_edge
from ..m44.actions import full_distribution, masses
from ..m44.records import power_ratio
from . import opponents as O
from . import plans as P
from .common import OFFSET, PRIMARY, STATE_SEED
from .freeze import controller_wave
from .runner import Experiment
from .states import collect, initialized

MAX_TURNS = 20
EDGE = 0.15
INIT_GRID = [(sc, gap, hp) for sc in ("neutral", "freeze") for gap in ("equal", "moderate", "large", "large_levels")
             for hp in (1.0, 0.5)]
MATRIX_GRID = [(gap, hp) for gap in ("equal", "large", "large_levels") for hp in (1.0, 0.5)]
RESPONSES = ("policy", "ignore", "respect", "shover", "ref:respect4")


# ---------------------------------------------------------------------------------------------- plans
def _hunt(obs) -> int:
    return behavior_rule("hunt", obs.cfg, obs.state, obs.champ, obs.side)


def threaten_rule(obs, t, mem):
    b = _hunt(obs)
    return P.force({"behavior": b}, "Threaten: " + ("Trade" if b == B_TRADE else "Hold"))


def bully_rule(obs, t, mem):
    if obs.champ.side(1 - obs.side).absent > 0:
        return P.force({"behavior": B_HOLD}, "Bully: enemy away, Hold")
    return P.force({"behavior": B_TRADE}, "Bully: Trade")


def freeze_threat_rule(obs, t, mem):
    w, b = controller_wave(obs), _hunt(obs)
    return P.force({"wave": w, "behavior": b}, "Freeze+threat")


def shove_pressure_rule(obs, t, mem):
    if P.possible_many(obs.legal, {"wave": W_SHOVE, "objective": PRESSURE}):
        return P.force({"wave": W_SHOVE, "objective": PRESSURE}, "Shove + Pressure")
    return P.force({"wave": W_SHOVE}, "Shove")


def candidates(situation, cfg, snap, seat) -> Dict[str, object]:
    if situation == "matrix":
        return {f"A_{n}": P.StrategicPlan(f"A_{n}", P.const_rule(P.force({"behavior": b}, n)), max_turns=10,
                                          on_absent="continue")
                for n, b in (("Hold", B_HOLD), ("Trade", B_TRADE))} | {
                "A_Threaten": P.StrategicPlan("A_Threaten", threaten_rule, max_turns=10, on_absent="continue")}

    def plan(name, rule):
        return P.StrategicPlan(name, rule, max_turns=MAX_TURNS, on_absent="continue")
    return {"policy": FollowPolicy(),
            "Farm": plan("Farm", P.const_rule(P.force({"wave": W_HOLD, "behavior": B_HOLD}, "Farm"))),
            "Threaten": plan("Threaten", threaten_rule),
            "Bully": plan("Bully", bully_rule),
            "Freeze+threat": plan("Freeze+threat", freeze_threat_rule),
            "Shove+Pressure": plan("Shove+Pressure", shove_pressure_rule)}


def matrix_opp_plans(situation):
    return {f"B_{n}": (lambda b=b, n=n: O.behavior_plan(b, 10, f"B_{n}"))
            for n, b in (("Hold", B_HOLD), ("Yield", B_YIELD), ("Safe", B_SAFE))}


# ---------------------------------------------------------------------------------------------- states
def describe(snap, s: int, cfg) -> dict:
    ctx = context(cfg)
    t = ctx.t
    me, en = snap.champ.side(s), snap.champ.side(1 - s)
    return {"power": power_ratio(ctx, snap.champ, s), "edge": duel_edge(cfg, snap.lane, snap.champ, s),
            "hp": me.hp / t.hp_max[t.level_by_xp[min(me.xp, t.xp_cap)]],
            "ehp": en.hp / t.hp_max[t.level_by_xp[min(en.xp, t.xp_cap)]], "pos": me.pos, "enemy_pos": en.pos,
            "level": t.level_by_xp[min(me.xp, t.xp_cap)], "elevel": t.level_by_xp[min(en.xp, t.xp_cap)],
            "invested": me.items / cfg.gold_scale, "einvested": en.items / cfg.gold_scale}


def init_states(cfg) -> List[dict]:
    out = []
    for sc, gap, hp in INIT_GRID:
        snap = initialized(cfg, sc, gap, hp=(1.0, hp))
        out.append({"snapshot": snap, "seat": 0, "turn": snap.k, "game": -1, "natural": False,
                    "start": f"{sc}/{gap}/hpB{hp:.1f}", "grid": {"scenario": sc, "gap": gap, "opp_hp": hp},
                    "row": {}, "desc": describe(snap, 0, cfg)})
    return out


def matrix_states(cfg) -> List[dict]:
    out = []
    for gap, hp in MATRIX_GRID:
        snap = initialized(cfg, "neutral", gap, hp=(1.0, hp))
        out.append({"snapshot": snap, "seat": 0, "turn": snap.k, "game": -1, "natural": False,
                    "start": f"neutral/{gap}/hpB{hp:.1f}", "grid": {"scenario": "neutral", "gap": gap, "opp_hp": hp},
                    "row": {}, "desc": describe(snap, 0, cfg)})
    return out


def edge_pred(d, t, s):
    return (d["en_present"][t, s] > 0 and d["edge"][t, s] >= EDGE and d["epos"][t, s] < 2 and d["dist"][t, s] <= 3)


def states_init(cfg, agent, env, seed, phase):
    return {"init": init_states(cfg)}


def states_matrix(cfg, agent, env, seed, phase):
    return {"matrix": matrix_states(cfg)}


def states_natural(cfg, agent, env, seed, phase):
    return collect(cfg, agent, {"edge": edge_pred}, {"edge": 24}, STATE_SEED + OFFSET["zoning"] + 100 * seed +
                   (0 if phase == PRIMARY else 50), game_turns=500, max_games=120, per_game=2, describe=describe)


def frequency(agent, cfg, snap, s) -> dict:
    m = masses(full_distribution(agent, observation(cfg, snap, s)))
    mo = masses(full_distribution(agent, observation(cfg, snap, 1 - s)))
    names = ("Hold", "Yield", "Trade", "Safe")
    out = {n: float(m["beh"][i]) for i, n in enumerate(names)}
    out.update({f"opp_{n}": float(mo["beh"][i]) for i, n in enumerate(names)})
    out["Recall"], out["opp_Recall"] = float(m["recall"]), float(mo["recall"])
    out["Shove"], out["opp_Shove"] = float(m["wave"][3]), float(mo["wave"][3])
    return out


def extra(cfg):
    T = cfg.tower

    def fn(infos, f):
        o = 1 - f
        H = min(MAX_TURNS, len(infos))
        beh = [inf["behaviors"][o] for inf in infos[:H]]
        present = [not inf["away"][o] for inf in infos[:H]]
        n = max(1, sum(present))
        pos_o = [inf["champ_end"][o] for inf, p in zip(infos[:H], present) if p]
        pos_f = [inf["champ_end"][f] for inf in infos[:H] if not inf["away"][f]]
        return {"opp_safe20": sum(b == B_SAFE for b, p in zip(beh, present) if p) / n,
                "opp_yield20": sum(b == B_YIELD for b, p in zip(beh, present) if p) / n,
                "opp_trade20": sum(b == B_TRADE for b, p in zip(beh, present) if p) / n,
                "opp_threatened20": sum(bool(inf["threatened"][o]) for inf in infos[:H]) / n,
                "opp_pos20": sum(pos_o) / len(pos_o) if pos_o else float("nan"),
                "foc_pos20": sum(pos_f) / len(pos_f) if pos_f else float("nan"),
                "opp_tower_cell20": sum(p == -T for p in pos_o) / n,
                "foc_tower_shot20": sum(inf["tower_damage"][f] > 0 for inf in infos[:H]),
                "opp_present20": float(sum(present))}
    return fn


def experiment(name: str) -> Experiment:
    common = dict(offset="zoning", candidates=candidates, frequency=frequency, ep_h=(5, 10, 20, 40), extra=extra,
                  notes={"max_turns": MAX_TURNS, "edge": EDGE, "init_grid": INIT_GRID, "matrix_grid": MATRIX_GRID})
    if name == "zoning":
        return Experiment(name=name, sizes={"init": len(INIT_GRID)}, protocols={"init": RESPONSES},
                          states=states_init, n_roll={"init": 24}, initialized=("init",), **common)
    if name == "zoning_nat":
        return Experiment(name=name, sizes={"edge": 24}, protocols={"edge": ("policy", "respect")},
                          states=states_natural, n_roll={"edge": 24}, **common)
    if name == "zoning_mech":
        return Experiment(name=name, sizes={"matrix": len(MATRIX_GRID)},
                          protocols={"matrix": ("B_Hold", "B_Yield", "B_Safe")}, states=states_matrix,
                          n_roll={"matrix": 32}, initialized=("matrix",), opp_plans=matrix_opp_plans, **common)
    raise ValueError(name)


def run_init(envs=None, seeds=None, workers: int = 11):
    from .runner import run
    return run("zoning", "zoning", envs, seeds, (PRIMARY,), workers)


def run_natural(envs=None, seeds=None, workers: int = 11):
    from .runner import run
    return run("zoning", "zoning_nat", envs, seeds, (PRIMARY,), workers)


def run_mechanics(envs=None, seeds=None, workers: int = 11):
    from .runner import run
    return run("zoning", "zoning_mech", envs, seeds, (PRIMARY,), workers)


__all__ = ["experiment", "run_init", "run_natural", "run_mechanics", "candidates", "INIT_GRID", "MATRIX_GRID"]
