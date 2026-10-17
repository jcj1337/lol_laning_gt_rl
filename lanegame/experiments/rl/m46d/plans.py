"""Adaptive plans around a learned policy (workstreams A and B).

A plan restricts the CRN policy's own distribution to a slice of action components chosen by a state rule, then
samples it with the SAME per-turn uniform as the unrestricted policy (`m46a.policies.BehaviorPlan`), so a plan and its
baseline share their randomness turn by turn. Rules read the START-OF-TURN state only (never the opponent's current
action). A plan is active while turn index i < n, the champion is present, its HP is above `release_hp`, and its
condition holds (memoryless: it re-engages if the condition holds again within n turns); otherwise the turn is the
policy's own, so every plan keeps the policy's Recall, disengagement and conversion outside its active turns.

Freezer rules (seat holding the freeze; condition "held" = the seat holds a freeze):
    guard_sel    wave Last Hit / Thin (the policy's choice between them); Trade when the enemy is present, catchable
                 without a dive, walked up (standing at our -1 or deeper), our HP >= 50% and our HP-aware duel edge
                 >= 0; else Hold                                                     n 20, release HP 0.40
    guard_walk   EXPLORATORY (amendment 3): as guard_sel without the duel-edge condition (Trade any walked-up,
                 catchable enemy while our HP >= 50%)                                n 20, release HP 0.40
    passive      wave Last Hit / Thin; behaviour Hold                                n 20, release HP 0.40
    guard_rigid  Thin + Trade (the M4.6c "guard", reference)                         n 20, release HP 0.40
    push         wave Shove, behaviour the policy's (Pressure allowed): release the wave   n 2, no condition
Breaker rules (seat frozen on; condition "frozen" = the enemy holds a freeze on the seat):
    commit       Shove + Hold (walk up and break)                                    n 10, release HP 0.25
    walk_lh      Last Hit + Hold (stand with the wave, do not break)                  n 10, release HP 0.30
    contest_sel  Shove + Hold when our HP >= 60% and (the enemy is away, below 50% HP, or its duel edge on us
                 <= 0.05); else Last Hit + Safe                                      n 10, release HP 0.30
    concede      Last Hit + Safe until the enemy is away or below 50% HP (then the policy)   n 10, release HP 0.30
    reset        walk back (Safe) and Recall as soon as legal, then the policy (`m46a.policies.ResetPlan`)  n 10
For the neutral and pushing-wave controls (no freeze), breaker plans drop the condition and run for n turns
(commit 3, the others 6).
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Set

import numpy as np

from ....config import (B_HOLD, B_RECALL, B_SAFE, B_TRADE, CHAMP_ABSENT, CHAMP_WARD, VISION, W_HOLD, W_SHOVE, W_THIN,
                        champ_act, champ_behavior, champ_objective, champ_wave)

LH, SH, TN = W_HOLD, W_SHOVE, W_THIN


# ---------------------------------------------------------------------------------------------- state helpers
def hp(cfg, m4, s) -> float:
    from ...m45b.strategies import hp_frac
    me = m4.side(s)
    return 1.0 if me.absent else hp_frac(cfg, m4, s)


def present(m4, s) -> bool:
    return m4.side(s).absent == 0


def catchable(cfg, m4, s) -> bool:
    from ...m45b.strategies import catchable as c
    return c(cfg, m4, s)


def edge(cfg, lane, m4, s) -> float:
    from ...m4.zoning_natural import duel_edge
    return duel_edge(cfg, lane, m4, s)


def enemy_cell(m4, s):
    from ...m45b.strategies import enemy_cell as e
    return e(m4, s)


def held(cfg, lane, m4, s) -> bool:
    from ..m44.records import holds_freeze
    return holds_freeze(lane, s)


def frozen(cfg, lane, m4, s) -> bool:
    from ..m44.records import holds_freeze
    return holds_freeze(lane, 1 - s)


CONDS: Dict[Optional[str], Optional[Callable]] = {None: None, "held": held, "frozen": frozen}


# ---------------------------------------------------------------------------------------------- rules
# a rule returns {"wave": set, "beh": set} (components allowed), or None (the policy's own turn)
def r_guard_sel(cfg, lane, m4, s):
    ec = enemy_cell(m4, s)
    ok = (present(m4, 1 - s) and ec is not None and ec != 99 and ec <= -1 and catchable(cfg, m4, s)
          and hp(cfg, m4, s) >= 0.5 and edge(cfg, lane, m4, s) >= 0.0)
    return {"wave": {LH, TN}, "beh": {B_TRADE if ok else B_HOLD}}


def r_guard_walk(cfg, lane, m4, s):
    ec = enemy_cell(m4, s)
    ok = (present(m4, 1 - s) and ec is not None and ec != 99 and ec <= -1 and catchable(cfg, m4, s)
          and hp(cfg, m4, s) >= 0.5)
    return {"wave": {LH, TN}, "beh": {B_TRADE if ok else B_HOLD}}


def r_passive(cfg, lane, m4, s):
    return {"wave": {LH, TN}, "beh": {B_HOLD}}


def r_guard_rigid(cfg, lane, m4, s):
    return {"wave": {TN}, "beh": {B_TRADE}}


def r_push(cfg, lane, m4, s):
    return {"wave": {SH}}


def r_commit(cfg, lane, m4, s):
    return {"wave": {SH}, "beh": {B_HOLD}}


def r_walk_lh(cfg, lane, m4, s):
    return {"wave": {LH}, "beh": {B_HOLD}}


def r_contest_sel(cfg, lane, m4, s):
    o = 1 - s
    weak_enemy = (not present(m4, o)) or hp(cfg, m4, o) < 0.5 or edge(cfg, lane, m4, o) <= 0.05
    if hp(cfg, m4, s) >= 0.6 and weak_enemy:
        return {"wave": {SH}, "beh": {B_HOLD}}
    return {"wave": {LH}, "beh": {B_SAFE}}


def r_concede(cfg, lane, m4, s):
    o = 1 - s
    if (not present(m4, o)) or hp(cfg, m4, o) < 0.5:
        return None
    return {"wave": {LH}, "beh": {B_SAFE}}


RULES: Dict[str, Callable] = {"guard_sel": r_guard_sel, "guard_walk": r_guard_walk, "passive": r_passive, "guard_rigid": r_guard_rigid,
                              "push": r_push, "commit": r_commit, "walk_lh": r_walk_lh, "contest_sel": r_contest_sel,
                              "concede": r_concede}


def code_matches(code: int, comps: dict) -> bool:
    if code in (CHAMP_ABSENT, CHAMP_WARD) or champ_behavior(code) == B_RECALL or champ_objective(code) == VISION:
        return False
    if "wave" in comps and champ_wave(code) not in comps["wave"]:
        return False
    if "beh" in comps and champ_behavior(code) not in comps["beh"]:
        return False
    return True


class Plan:
    """The module docstring's adaptive plan around a CRN policy `own` (m45c.horizon.CRNNet)."""

    def __init__(self, name: str, own, rule: Callable, n: int, cond: Optional[str] = None, release_hp: float = 0.4):
        self.name, self.own, self.rule, self.n = name, own, rule, int(n)
        self.cond, self.release_hp = CONDS[cond], float(release_hp)
        self.forced = 0

    def decide(self, cfg, lane, m4, s: int, i: int):
        if (i >= self.n or not present(m4, s) or hp(cfg, m4, s) <= self.release_hp
                or (self.cond is not None and not self.cond(cfg, lane, m4, s))):
            return self.own.decide(cfg, lane, m4, s, i)
        comps = self.rule(cfg, lane, m4, s)
        if comps is None:
            return self.own.decide(cfg, lane, m4, s, i)
        legal, p = self.own.probs(cfg, lane, m4, s, i)
        sl = [j for j, c in enumerate(legal) if code_matches(int(c), comps)]
        if not sl:
            return self.own.decide(cfg, lane, m4, s, i)
        q = p[sl]
        tot = float(q.sum())
        q = q / tot if tot > 0 else np.full(len(sl), 1.0 / len(sl))
        j = min(int(np.searchsorted(np.cumsum(q), self.own.u[i], side="right")), len(sl) - 1)
        self.forced += 1
        return legal[sl[j]], f"[{self.name}]"


# plan specifications: name -> (rule, n, condition, release HP); "policy" and "reset" are special
FREEZER = {"policy": None,
           "guard_sel": ("guard_sel", 20, "held", 0.4),
           "guard_walk": ("guard_walk", 20, "held", 0.4),
           "passive": ("passive", 20, "held", 0.4),
           "guard_rigid": ("guard_rigid", 20, "held", 0.4),
           "push": ("push", 2, None, 0.0)}
BREAKER = {"policy": None,
           "commit": ("commit", 10, "frozen", 0.25),
           "walk_lh": ("walk_lh", 10, "frozen", 0.3),
           "contest_sel": ("contest_sel", 10, "frozen", 0.3),
           "concede": ("concede", 10, "frozen", 0.3),
           "concede3": ("concede", 3, "frozen", 0.3),
           "reset": ("reset", 10, None, 0.0)}
# the principal matrix (pre-registration §3): the four freezer and five breaker options of the brief
A_FREEZER = {k: FREEZER[k] for k in ("policy", "guard_sel", "passive", "push")}
A_BREAKER = {k: BREAKER[k] for k in ("policy", "commit", "contest_sel", "concede", "reset")}
BREAKER_CONTROL = {"policy": None,
                   "commit": ("commit", 3, None, 0.25),
                   "walk_lh": ("walk_lh", 6, None, 0.3),
                   "contest_sel": ("contest_sel", 6, None, 0.3),
                   "concede": ("concede", 6, None, 0.3),
                   "concede3": ("concede", 3, None, 0.3),
                   "reset": ("reset", 10, None, 0.0)}


def build(spec, own):
    """A seat policy from a plan specification (None = the policy itself)."""
    if spec is None:
        return own
    rule, n, cond, rel = spec
    if rule == "reset":
        from ..m46a.policies import ResetPlan
        return ResetPlan(own, n)
    return Plan(rule, own, RULES[rule], n, cond, rel)


__all__ = ["Plan", "RULES", "FREEZER", "BREAKER", "A_FREEZER", "A_BREAKER", "BREAKER_CONTROL", "build", "code_matches", "held", "frozen", "hp"]
