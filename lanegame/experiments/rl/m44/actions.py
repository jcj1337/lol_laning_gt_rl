"""The 61-code action space of the four-action presets: decoding, the codes that can ever be legal, and the
probability masses of a policy by component.

Encoding (lanegame.config; nothing here redefines it, every decoding goes through champ_wave / champ_behavior /
champ_objective):

    codes 0..44    wave in {Last Hit (slot 0), Shove (slot 1), Thin (slot 2)} x behaviour x objective,
                   a = wave + 3 behaviour + 15 objective
    code  45       ABSENT (the only legal code while the champion is away)
    codes 46..60   Slow Push (slot 3) x behaviour x objective, a = 46 + behaviour + 5 objective

The wave slots are NOT contiguous in the code space and their numbering is not their order of wave damage: the
canonical order used for reporting is WAVE_ORDER = (Last Hit, Thin, Slow Push, Shove), from the least to the most wave
damage. Legality (champions.champion_legal_actions): a present champion has the 4 x 4 wave x {Hold, Yield, Trade, Safe}
codes; with an own crash under the enemy tower and the champion able to reach that tower, also 4 waves x {Hold, Trade}
x {Pressure, Vision}; at or behind its recall line, the single Recall code champ_act(Last Hit, Recall) = 12. So only
16 + 16 + 1 + 1 = 34 of the 61 codes can ever be legal (`EVER_LEGAL`); the other 27 (Yield / Safe with an objective,
Recall with another wave or an objective) exist only because the encoding is a product, are masked in every state and
never receive probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Sequence

import numpy as np

from ....config import (B_HOLD, B_RECALL, B_SAFE, B_TRADE, B_YIELD, CHAMP_ABSENT, CHAMP_BEHAVIORS, NONE, PRESSURE,
                        VISION, W_HOLD, W_SHOVE, W_SLOWPUSH, W_THIN, GameConfig, champ_act, champ_behavior,
                        champ_objective, champ_wave)

WAVE_ORDER = (W_HOLD, W_THIN, W_SLOWPUSH, W_SHOVE)           # least -> most wave damage
WAVE_NAMES = ("Last Hit", "Thin", "Slow Push", "Shove")
WAVE_KEYS = ("LH", "TN", "SL", "SH")
WAVE_INDEX = {w: i for i, w in enumerate(WAVE_ORDER)}         # wave slot -> reporting index
BEH_NAMES = ("Hold", "Yield", "Trade", "Safe", "Recall")       # indexed by the behaviour code B_*
OBJ_NAMES = ("none", "Pressure", "Vision")                     # indexed by the objective code
RECALL = champ_act(W_HOLD, B_RECALL)                           # 12
N_CODES = 61


@dataclass(frozen=True)
class Decoded:
    code: int
    absent: bool
    recall: bool
    wave: int              # wave slot (W_HOLD / W_SHOVE / W_THIN / W_SLOWPUSH); -1 when absent
    behavior: int          # B_*; -1 when absent
    objective: int         # NONE / PRESSURE / VISION; -1 when absent

    @property
    def wave_index(self) -> int:
        """Index into WAVE_ORDER (-1 when absent; Recall reports its Last Hit slot)."""
        return WAVE_INDEX.get(self.wave, -1)

    def label(self) -> str:
        if self.absent:
            return "Absent"
        if self.recall:
            return "Recall"
        s = f"{WAVE_NAMES[self.wave_index]} + {BEH_NAMES[self.behavior]}"
        return s + (f" + {OBJ_NAMES[self.objective]}" if self.objective else "")


def decode(a: int) -> Decoded:
    """Decode one champion-mode code of the 61-code space (the authoritative config helpers)."""
    a = int(a)
    if not 0 <= a < N_CODES:
        raise ValueError(f"action code {a} outside the 61-code space")
    if a == CHAMP_ABSENT:
        return Decoded(a, True, False, -1, -1, -1)
    b = champ_behavior(a)
    return Decoded(a, False, b == B_RECALL, champ_wave(a), b, champ_objective(a))


def _ever_legal() -> FrozenSet[int]:
    s = {champ_act(w, b) for w in WAVE_ORDER for b in CHAMP_BEHAVIORS}
    s |= {champ_act(w, b, o) for w in WAVE_ORDER for b in (B_HOLD, B_TRADE) for o in (PRESSURE, VISION)}
    s |= {RECALL, CHAMP_ABSENT}
    return frozenset(s)


EVER_LEGAL: FrozenSet[int] = _ever_legal()
NEVER_LEGAL: FrozenSet[int] = frozenset(range(N_CODES)) - EVER_LEGAL


def action_table(cfg: GameConfig) -> List[dict]:
    """Every code of cfg's action space with its decoding, label and whether the legality rules can ever offer it."""
    out = []
    for a in cfg.action_codes():
        d = decode(a)
        out.append({"code": a, "label": cfg.label(a), "name": d.label(), "wave": d.wave_index, "behavior": d.behavior,
                    "objective": d.objective, "absent": d.absent, "recall": d.recall, "ever_legal": a in EVER_LEGAL})
    return out


def check_four_action_space(cfg: GameConfig) -> None:
    """Raise unless cfg is a four-action configuration with the 61-code space this module describes."""
    if not (cfg.is_m4 and cfg.natural_waves and cfg.wave_slowpush):
        raise ValueError("not a four-action configuration (needs ruleset m4, natural waves and wave_slowpush)")
    if tuple(cfg.action_codes()) != tuple(range(N_CODES)):
        raise ValueError("unexpected action space size")


# ---------------------------------------------------------------------------------------------- masses
def masses(p_full: np.ndarray) -> Dict[str, np.ndarray]:
    """Probability masses of a full-length (61) distribution by component: wave (4, WAVE_ORDER, Recall excluded),
    behaviour (5, B_* incl. Recall), objective (3) over present, non-Recall codes, plus recall and absent."""
    p = np.asarray(p_full, dtype=float)
    if len(p) != N_CODES:
        raise ValueError(f"expected a {N_CODES}-entry distribution, got {len(p)}")
    wave, beh, obj = np.zeros(4), np.zeros(5), np.zeros(3)
    rec = absent = 0.0
    for a in np.flatnonzero(p > 0):
        a = int(a)
        if a in NEVER_LEGAL:
            raise ValueError(f"probability {p[a]:.3g} on code {a}, which no state can make legal")
        if a == CHAMP_ABSENT:
            absent += p[a]
            continue
        b = champ_behavior(a)
        beh[b] += p[a]
        if b == B_RECALL:
            rec += p[a]
            continue
        wave[WAVE_INDEX[champ_wave(a)]] += p[a]
        obj[champ_objective(a)] += p[a]
    return {"wave": wave, "beh": beh, "obj": obj, "recall": rec, "absent": absent}


def full_distribution(agent, obs) -> np.ndarray:
    """Full-length probability vector of `agent` at `obs` (0 on illegal codes); agents expose `probabilities` (a
    NetAgent) or `distribution` over obs.legal (scripted agents: one-hot)."""
    f = getattr(agent, "probabilities", None)
    if callable(f):
        return np.asarray(f(obs), dtype=float)
    p = np.zeros(len(obs.mask))
    p[list(obs.legal)] = np.asarray(agent.distribution(obs), dtype=float)
    return p


def codes_with(legal: Sequence[int], wave: int = None, behavior: int = None, objective: int = None) -> List[int]:
    """The legal present, non-Recall codes whose components equal the given ones (None = any)."""
    out = []
    for a in legal:
        if a in (CHAMP_ABSENT, RECALL):
            continue
        if wave is not None and champ_wave(a) != wave:
            continue
        if behavior is not None and champ_behavior(a) != behavior:
            continue
        if objective is not None and champ_objective(a) != objective:
            continue
        out.append(int(a))
    return out


__all__ = ["WAVE_ORDER", "WAVE_NAMES", "WAVE_KEYS", "WAVE_INDEX", "BEH_NAMES", "OBJ_NAMES", "RECALL", "N_CODES",
           "Decoded", "decode", "EVER_LEGAL", "NEVER_LEGAL", "action_table", "check_four_action_space", "masses",
           "full_distribution", "codes_with", "B_HOLD", "B_YIELD", "B_TRADE", "B_SAFE", "B_RECALL", "NONE", "PRESSURE",
           "VISION", "W_HOLD", "W_THIN", "W_SLOWPUSH", "W_SHOVE"]
