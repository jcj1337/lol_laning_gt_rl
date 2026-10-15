"""Investment curves of M4.6b: the arms and their mapping tables (pure formulas, no gameplay).

The current curve (`GameConfig.m45_calibrated()`, invest_curve "saturating"): M(S) = 1 + A (1 - sqrt(K / (K + S))),
A = invest_gain = 1.5, K = invest_scale = 16 resources. M multiplies attack power AP*(L, S) = 1.05^(L-1) M(S) (trade
damage dealt, wave-clear capacity C = round(4 AP*), Pressure) and divides every damage taken (invest_durability).

ARMS maps an arm name to the GameConfig overrides of `m45_calibrated()`; candidates use existing fields only. They are
filled in from the audit (audit.py) before any candidate gameplay outcome (pre-registration).
"""

from __future__ import annotations

import math
from typing import Dict

ARMS: Dict[str, dict] = {
    "baseline": {"overrides": {}, "label": "current: saturating A 1.5, K 16",
                 "formula": "1 + 1.5 (1 - sqrt(16 / (16 + S)))", "role": "reference"},
    # pre-registered (docs/m4/m4_6/m4_6b_preregistration.md §2): amplitudes solved so that the discount-weighted mean of
    # ln(M_arm / M_baseline) over the audit's natural investment distribution is 0 (|residual| <= 0.001)
    "C1": {"overrides": {"invest_gain": 3.24, "invest_scale": 64.0}, "label": "C1 slow saturation: A 3.24, K 64",
           "formula": "1 + 3.24 (1 - sqrt(64 / (64 + S)))", "role": "modest candidate"},
    "C2": {"overrides": {"invest_gain": 9.53, "invest_scale": 256.0}, "label": "C2 slower saturation: A 9.53, K 256",
           "formula": "1 + 9.53 (1 - sqrt(256 / (256 + S)))", "role": "larger candidate"},
    "C3": {"overrides": {"invest_curve": "power", "invest_alpha": 0.016, "invest_p": 1.0},
           "label": "C3 linear reference: 1 + 0.016 S", "formula": "1 + 0.016 S", "role": "reference (no diminishing returns)"},
}
ARM_NAMES = tuple(ARMS)


def multiplier(arm: str, S: float) -> float:
    """M(S) of an arm (float; the engine evaluates it exactly in integers)."""
    from .common import arm_cfg
    from ....progression import investment
    cfg = arm_cfg(arm)
    inv = investment(cfg)
    return inv.factor(int(round(S * cfg.gold_scale)))


def m_formula(curve: str, S: float, gain: float = 1.5, scale: float = 16.0, alpha: float = 0.045,
              p: float = 0.9) -> float:
    if curve == "saturating":
        return 1.0 + gain * (1.0 - math.sqrt(scale / (scale + S)))
    return 1.0 + alpha * ((1.0 + S) ** p - 1.0)


def elasticity(arm: str, S: float, h: float = 0.01) -> float:
    """d ln M / d ln S: the relative AP gain per relative investment lead (finite difference)."""
    if S <= 0:
        return 0.0
    a, b = multiplier(arm, S * (1 - h)), multiplier(arm, S * (1 + h))
    return (math.log(b) - math.log(a)) / (math.log(1 + h) - math.log(1 - h))


__all__ = ["ARMS", "ARM_NAMES", "multiplier", "m_formula", "elasticity"]
