"""State features (exposure, analysis) and observation-derived strength features (the explicit-feature arm D).

State features of seat s (`StateFeatures.side`), M4.6d conventions:
    duel        (1.05^(L_s - L_o) M(S_s) / M(S_o))^2: damage per exchange in victim bars, s over o (m46d.freezes.Strength)
    hp, hp_o    HP fractions of the level's max HP (raw; `present` says whether the champion is in lane)
    held        s holds a freeze (m44.records.holds_freeze); frozen: the opponent holds one on s
    held_hp     the enemy front's HP while s holds a freeze (the held wave)
    catch_o     the opponent can reach s's start cell without a dive this turn (m45b.strategies.catchable)
    dist        champion distance along the lane (both in lane cells; None with W or absence)

Observation-derived features (`explicit`): deterministic functions of the 90-entry observation, so arm D adds NO
information; it only presents a relationship in a form a small network can use directly.
    log_duel    2 (logit(ap_own) - logit(ap_enemy)); logit of the attack-power entry ap / (ap + ap_ref) is
                log(ap / ap_ref), so this is 2 log(AP_own / AP_enemy) = log of the duel ratio (exact up to float32 and
                the AP table's rounding)
    hp_own, hp_enemy, log_hp_ratio = log(max(hp_own, 0.05) / max(hp_enemy, 0.05)), log_duel_hp = log_duel +
    log_hp_ratio (the HP-adjusted duel), both_present x log_duel
"""

from __future__ import annotations

from typing import Dict

import numpy as np

EXPLICIT_NAMES = ("log_duel", "hp_own", "hp_enemy", "log_hp_ratio", "log_duel_hp", "present_log_duel")


class StateFeatures:
    def __init__(self, cfg):
        from ....champions_m4 import context
        from ..m46d.freezes import Strength
        self.cfg = cfg
        self.t = context(cfg).t
        self.st = Strength(cfg)
        self.sc = cfg.gold_scale

    def level(self, m4, s: int) -> int:
        t = self.t
        return int(t.level_by_xp[min(m4.side(s).xp, t.xp_cap)])

    def hp(self, m4, s: int) -> float:
        me = m4.side(s)
        return me.hp / self.t.hp_max[self.level(m4, s)]

    def duel(self, m4, s: int) -> float:
        o = 1 - s
        d, _, _ = self.st.parts(self.level(m4, s), m4.side(s).items / self.sc, self.level(m4, o),
                                m4.side(o).items / self.sc)
        return float(d)

    def side(self, lane, m4, s: int) -> Dict[str, float]:
        from ...m45b.strategies import catchable
        from ..m44.records import holds_freeze
        from ....positional import WARD
        o = 1 - s
        me, en = m4.side(s), m4.side(o)
        held = holds_freeze(lane, s)
        dist = None
        if not me.absent and not en.absent and me.pos != WARD and en.pos != WARD:
            dist = abs(me.pos + en.pos)                       # the enemy's cell in s's coordinates is -en.pos
        return {"L": self.level(m4, s), "L_o": self.level(m4, o), "S": me.items / self.sc,
                "S_o": en.items / self.sc, "g": me.gold / self.sc, "hp": self.hp(m4, s), "hp_o": self.hp(m4, o),
                "present": float(me.absent == 0), "present_o": float(en.absent == 0), "duel": self.duel(m4, s),
                "held": float(held), "frozen": float(holds_freeze(lane, o)),
                "held_hp": float(lane.side(o)[1]) if held else 0.0,
                "catch_o": float(catchable(self.cfg, m4, o)), "dist": -1.0 if dist is None else float(dist),
                "pos": float(me.pos)}


# ---------------------------------------------------------------------------------------------- observation side
def obs_index(cfg) -> Dict[str, int]:
    from ....env.observation_m4 import m4_feature_names
    names = m4_feature_names(cfg)
    return {n: i for i, n in enumerate(names)}


def _logit(x):
    x = np.clip(np.asarray(x, dtype=np.float64), 1e-9, 1 - 1e-9)
    return np.log(x / (1.0 - x))


def explicit(X: np.ndarray, idx: Dict[str, int]) -> np.ndarray:
    """(N, 6) observation-derived strength features (module docstring), from (N, 90) observations."""
    X = np.asarray(X, dtype=np.float64)
    ld = 2.0 * (_logit(X[:, idx["own_champ_attack_power"]]) - _logit(X[:, idx["enemy_champ_attack_power"]]))
    ho, he = X[:, idx["own_champ_hp_frac"]], X[:, idx["enemy_champ_hp_frac"]]
    lhr = np.log(np.maximum(ho, 0.05)) - np.log(np.maximum(he, 0.05))
    both = (X[:, idx["own_champ_absent"]] == 0) & (X[:, idx["enemy_champ_absent"]] == 0)
    return np.stack([ld, ho, he, lhr, ld + lhr, both * ld], axis=1).astype(np.float32)


__all__ = ["StateFeatures", "obs_index", "explicit", "EXPLICIT_NAMES"]
