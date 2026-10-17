"""Plans on a trained policy for M4.6a (pre-registration §2, E2 and E5), in the m45b engine's policy interface
`decide(cfg, lane, m4, s, i) -> (code, why)`.

`CRNValueNet` is `m45c.horizon.CRNNet` that also keeps the critic's value of its last observation (E6).

`BehaviorPlan` forces components of the action for the first `n` turns and leaves the others to the policy: the policy's
distribution over its legal codes is restricted to the codes with the forced components (Ward and Recall excluded),
renormalized (uniform over the slice if the policy gives it no mass), and sampled with the SAME per-turn uniform as the
unforced policy, so a plan and its baseline share their randomness turn by turn. While the champion is at or below
`release_hp` of its max HP (or absent, or the slice is empty) the turn is the policy's own. Rules are functions of the
start-of-turn state only:

    farm           behaviour Hold
    threaten       Trade if the enemy is catchable without a dive and the duel edge is positive, else Hold (the
                   M4.3c "hunt" rule, graph-aware catchability of M4.5b)
    edge           Trade if catchable without a dive, own HP >= 60% and duel edge >= 0.05, else Hold
    bully          Trade whenever catchable without a dive, else Hold
    freeze_threat  the threaten behaviour with the freeze controller's wave (Thin when the fronts touch at our -1 or
                   deeper and the enemy front holds >= 14 HP, else Last Hit)

Candidate specifications (tuples, picklable) are built by `build(spec, own)`:
    ("policy",)                         the policy itself
    ("beh", rule, n, release_hp)        BehaviorPlan
    ("script", strategy, n)             an M4.5b scripted strategy for n turns, then the policy (m45c horizon "rule")
    ("fixed", code, n)                  a fixed code for n turns while legal (m45c horizon "fixed")
State variants (E5) are start-state transforms named in VARIANTS.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

from ....config import (B_HOLD, B_RECALL, B_TRADE, CHAMP_ABSENT, CHAMP_WARD, VISION, W_HOLD, W_THIN, champ_behavior,
                        champ_objective, champ_wave)
from ..m45c import horizon as HZ

FREEZE_HP = 14
EDGE_HP, EDGE_MIN = 0.6, 0.05
RELEASE_HP = 0.4


class CRNValueNet(HZ.CRNNet):
    """CRNNet whose forward pass also stores the critic value (resources) of the observation it decided on."""

    def probs(self, cfg, lane, m4, s: int, i: int):
        import torch
        from ....diagnostics.branch import Snapshot, observation
        obs = observation(cfg, Snapshot(lane, m4, self.k0 + i, (cfg.skill_a, cfg.skill_b)), s)
        with torch.no_grad():
            logits, v = self.net(torch.as_tensor(obs.vector, dtype=torch.float32).unsqueeze(0))
        self.last_value = float(v.reshape(-1)[0]) / 0.1
        lg = logits[0].double().numpy()[list(obs.legal)]
        p = np.exp(lg - lg.max())
        return obs.legal, p / p.sum()


# ---------------------------------------------------------------------------------------------- rules
def _hp(cfg, m4, s) -> float:
    from ...m45b.strategies import hp_frac
    return hp_frac(cfg, m4, s)


def _catchable(cfg, m4, s) -> bool:
    from ...m45b.strategies import catchable
    return catchable(cfg, m4, s)


def _edge(cfg, lane, m4, s) -> float:
    from ...m4.zoning_natural import duel_edge
    return duel_edge(cfg, lane, m4, s)


def controller_wave(lane, s: int) -> int:
    ofx, ofh = lane.side(s)[:2]
    efx, efh = lane.side(1 - s)[:2]
    contact = ofh > 0 and efh > 0 and ofx + efx >= -1
    return W_THIN if (contact and ofx <= -1 and efh >= FREEZE_HP) else W_HOLD


def rule_farm(cfg, lane, m4, s):
    return {"behavior": B_HOLD}


def rule_threaten(cfg, lane, m4, s):
    ok = m4.side(1 - s).absent == 0 and _edge(cfg, lane, m4, s) > 0 and _catchable(cfg, m4, s)
    return {"behavior": B_TRADE if ok else B_HOLD}


def rule_edge(cfg, lane, m4, s):
    ok = (m4.side(1 - s).absent == 0 and _catchable(cfg, m4, s) and _hp(cfg, m4, s) >= EDGE_HP
          and _edge(cfg, lane, m4, s) >= EDGE_MIN)
    return {"behavior": B_TRADE if ok else B_HOLD}


def rule_bully(cfg, lane, m4, s):
    return {"behavior": B_TRADE if _catchable(cfg, m4, s) else B_HOLD}


def rule_freeze_threat(cfg, lane, m4, s):
    return {"wave": controller_wave(lane, s), **rule_threaten(cfg, lane, m4, s)}


def rule_trade(cfg, lane, m4, s):
    return {"behavior": B_TRADE}


def rule_safe(cfg, lane, m4, s):
    from ....config import B_SAFE
    return {"behavior": B_SAFE}


RULES: Dict[str, Callable] = {"farm": rule_farm, "threaten": rule_threaten, "edge": rule_edge, "bully": rule_bully,
                              "freeze_threat": rule_freeze_threat, "trade": rule_trade, "safe": rule_safe,
                              "hold": rule_farm}


def matches(code: int, comps: Dict[str, int]) -> bool:
    if code in (CHAMP_ABSENT, CHAMP_WARD) or champ_behavior(code) == B_RECALL or champ_objective(code) == VISION:
        return False
    if "behavior" in comps and champ_behavior(code) != comps["behavior"]:
        return False
    if "wave" in comps and champ_wave(code) != comps["wave"]:
        return False
    return True


class BehaviorPlan:
    """The module docstring's restricted-distribution plan around a CRN policy `own`."""

    def __init__(self, name: str, own: HZ.CRNNet, rule: Callable, n: int, release_hp: float = RELEASE_HP):
        self.name, self.own, self.rule, self.n, self.release_hp = name, own, rule, int(n), float(release_hp)
        self.forced = 0

    def decide(self, cfg, lane, m4, s: int, i: int):
        if i >= self.n or m4.side(s).absent > 0 or _hp(cfg, m4, s) <= self.release_hp:
            return self.own.decide(cfg, lane, m4, s, i)
        legal, p = self.own.probs(cfg, lane, m4, s, i)
        comps = self.rule(cfg, lane, m4, s)
        sl = [j for j, c in enumerate(legal) if matches(int(c), comps)]
        if not sl:
            return self.own.decide(cfg, lane, m4, s, i)
        q = p[sl]
        tot = float(q.sum())
        q = q / tot if tot > 0 else np.full(len(sl), 1.0 / len(sl))
        j = min(int(np.searchsorted(np.cumsum(q), self.own.u[i], side="right")), len(sl) - 1)
        self.forced += 1
        return legal[sl[j]], f"[{self.name}] {comps}"

    def inner_policies(self):
        return (self.own,)                               # M5 opt-in hook (m45b.engine.observe_turn)


class ResetPlan:
    """M4.6b: reset now. While present and not yet recalled (turn index < n): Recall when it is legal, otherwise the
    policy restricted to Safe (walk back; the wave left to the policy); after the Recall, or from turn n, the policy."""

    def __init__(self, own: HZ.CRNNet, n: int):
        self.own, self.n, self.done, self.forced = own, int(n), False, 0
        self.safe = BehaviorPlan("safe", own, rule_safe, n, 0.0)

    def decide(self, cfg, lane, m4, s: int, i: int):
        from ....champions import champion_legal_actions
        from ....config import W_HOLD as _LH, champ_act
        if i == 0:                                       # a new rollout (the engine also queries turn 0 beforehand)
            self.done = False
        if self.done or i >= self.n or m4.side(s).absent > 0:
            return self.own.decide(cfg, lane, m4, s, i)
        rec = champ_act(_LH, B_RECALL)
        if rec in champion_legal_actions(lane, m4, s, cfg):
            self.done = True
            self.forced += 1
            return rec, "[reset] Recall"
        self.forced += 1
        return self.safe.decide(cfg, lane, m4, s, i)

    def inner_policies(self):
        return (self.own,)                               # M5 opt-in hook (m45b.engine.observe_turn)


def build(cspec: tuple, own: HZ.CRNNet):
    kind = cspec[0]
    if kind == "reset":
        return ResetPlan(own, cspec[1])
    if kind == "hz":                                     # an m45c horizon candidate spec (fixed / rule, with a condition)
        return HZ.build_candidate(tuple(cspec[1]), own)
    if kind == "policy":
        return own
    if kind == "beh":
        _, rule, n, rel = cspec
        return BehaviorPlan(rule, own, RULES[rule], n, rel)
    if kind == "script":
        _, name, n = cspec
        return HZ.build_candidate(("rule", name, n, None), own)
    if kind == "fixed":
        _, code, n = cspec
        return HZ.build_candidate(("fixed", int(code), n, None), own)
    raise ValueError(cspec)


# ---------------------------------------------------------------------------------------------- state variants
def variant_invest(cfg, start):
    """E5-shop: the focal side A's unspent gold moved into its investment (nothing else changes)."""
    lane, m4, k = start
    a = m4.a
    return lane, m4._replace(a=a._replace(items=a.items + a.gold, gold=0)), k


def variant_hp(cfg, start, bars: float = 0.3):
    """E5-hp: the focal side A's HP lowered by `bars` of its max HP (at least 1 centi-HP left)."""
    from ....champions_m4 import context
    lane, m4, k = start
    t = context(cfg).t
    a = m4.a
    L = t.level_by_xp[min(a.xp, t.xp_cap)]
    hp = max(1, a.hp - int(round(bars * t.hp_max[L])))
    return lane, m4._replace(a=a._replace(hp=hp)), k


VARIANTS: Dict[str, Callable] = {"orig": lambda cfg, st: st, "inv": variant_invest, "hp-0.3": variant_hp}


__all__ = ["CRNValueNet", "BehaviorPlan", "RULES", "build", "VARIANTS", "controller_wave", "matches", "RELEASE_HP"]
