"""Wave-only interventions on a learned policy (M4.3b follow-up, brief §2).

The plan-level counterfactuals of `plans.py` force COMPLETE actions (Thin + Hold position, Hold + Hold, ...), so a
measured loss of a slow push or a freeze mixes two things: the wave management under test and the loss of the
policy's adaptive champion behaviour (Trade, Safe, Yield, Recall, Pressure) for the plan's duration. This module
intervenes on the WAVE component only and leaves every other decision to the policy, through an explicit, reproducible
rule on the policy's probability vector at each turn.

The action space (`config.champ_act`): a = wave + 3 behaviour + 15 objective for a present champion (45 codes), plus
ABSENT. Under natural waves (`mechanics.wave_actions`) all three wave components are legal on every turn a champion is
present, and the legality of the behaviour / objective part does not depend on the wave (Pressure / Vision are legal
with Hold or Trade when the own crashed mass is under the enemy tower and the champion can reach it; Recall is the
single code champ_act(W_HOLD, B_RECALL) at or behind the recall line). Therefore replacing the wave of any legal
present-champion code other than Recall gives a legal code, and `intervene` never needs a fallback.

Two intervention rules, both defined on the policy's probability vector p over obs.legal:

  marginal      component replacement: every action a (other than Recall) sends its probability mass to
                (w, behaviour(a), objective(a)). The behaviour / objective is drawn from the policy's MARGINAL over
                those components, whatever wave it would have paired them with.
  conditional   policy-conditional: the mass of the actions whose wave is w is renormalised, i.e. the behaviour /
                objective is drawn from the policy's CONDITIONAL distribution given that it chooses wave w (what the
                policy pairs with that wave when it chooses it itself).

Both keep Recall exactly at the policy's own probability (a Recall channel has no wave management; it is the decision
the brief asks to preserve), and both leave an absent champion's single ABSENT action alone. The rules differ when
the policy's behaviour is correlated with its wave choice (e.g. Clear + Trade against Hold + Safe); reporting both
brackets the intervention. Neither rule changes the opponent, the continuation after the intervention, or the
chance streams: a `WavePlan` reshapes p on every turn of the intervention window and the rollout samples from the
reshaped vector with the same uniform draw (`diagnostics.branch.Plan.transform`), so candidates stay paired.

Wave functions (obs, t) -> wave component or None (no intervention this turn):

  const(w)              Hold, Thin or Clear every turn
  slow_push_schedule    Thin for the first 2 turns, then Hold (the schedule of plans.slow_push)
  freeze_controller     the M4.2e-C threshold controller: Thin while the enemy front stands at own -1 or deeper with
                        >= FREEZE_THRESHOLD HP, else Hold
  reactive_slow_push    the zoning_natural "slow_push" wave rule: in contact, Thin while the own front is at most
                        2 HP ahead of the enemy front, else Hold; no contact: Hold (for whole games)
  shove_away            Clear while the enemy champion is absent, else Hold

`WavePlan(name, wave_fn, turns, rule)` is the branch plan (DEFER on every turn of the window, including turns the
champion is absent, so an intervention of N turns means "for the next N turns, whenever present"; then the policy
resumes). `InterventionAgent(agent, wave_fn, rule)` is the same policy as a whole-game environment agent (its
`distribution` is the reshaped vector; `act` samples it with the environment's inverse-CDF rule), for the
opponent-dependent whole-game comparisons.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np

from ....config import (B_HOLD, B_RECALL, CHAMP_ABSENT, W_CLEAR, W_HOLD, W_THIN, champ_act, champ_behavior,
                        champ_objective, champ_wave)
from ....diagnostics.branch import DEFER, Plan
from ....env.agents import sample

RECALL = champ_act(W_HOLD, B_RECALL)
RULES = ("marginal", "conditional")
FREEZE_THRESHOLD = 14
WAVE_OF = {"hold": W_HOLD, "thin": W_THIN, "clear": W_CLEAR}
WAVE_NAME = {W_HOLD: "Hold", W_THIN: "Thin", W_CLEAR: "Clear"}


# ---------------------------------------------------------------------------------------------- the rules
def intervene(p, legal: Sequence[int], wave: Optional[int], rule: str) -> np.ndarray:
    """The reshaped probability vector over `legal` (see the module docstring). `wave` None: p unchanged."""
    p = np.asarray(p, dtype=float)
    legal = tuple(int(a) for a in legal)
    if wave is None or legal == (CHAMP_ABSENT,):
        return p
    if len(p) != len(legal):
        raise ValueError(f"p has {len(p)} entries for {len(legal)} legal actions")
    if rule not in RULES:
        raise ValueError(f"unknown intervention rule {rule!r}; choose from {RULES}")
    idx = {a: i for i, a in enumerate(legal)}
    q = np.zeros_like(p)
    if rule == "marginal":
        for i, a in enumerate(legal):
            if a == RECALL or a == CHAMP_ABSENT:
                q[i] += p[i]
                continue
            a2 = champ_act(wave, champ_behavior(a), champ_objective(a))
            j = idx.get(a2)
            if j is None:                                   # cannot happen under natural waves; never silently drop mass
                raise ValueError(f"wave replacement of action {a} gives {a2}, which is not legal (legal {legal})")
            q[j] += p[i]
    else:
        r = float(p[idx[RECALL]]) if RECALL in idx else 0.0
        sl = [i for i, a in enumerate(legal) if a != RECALL and a != CHAMP_ABSENT and champ_wave(a) == wave]
        if not sl:
            raise ValueError(f"no legal action with wave {wave} (legal {legal})")
        mass = float(p[sl].sum())
        if mass > 0.0:
            q[sl] = p[sl] * (1.0 - r) / mass
        else:                                               # degenerate slice (never observed): uniform over it
            q[sl] = (1.0 - r) / len(sl)
        if RECALL in idx:
            q[idx[RECALL]] = r
    return q


def wave_replacement_is_legal(legal: Sequence[int]) -> bool:
    """True iff every legal present-champion code other than Recall stays legal under every wave replacement
    (the invariant `intervene` relies on; checked by the tests over sampled states)."""
    s = set(legal)
    for a in legal:
        if a in (RECALL, CHAMP_ABSENT):
            continue
        for w in (W_HOLD, W_THIN, W_CLEAR):
            if champ_act(w, champ_behavior(a), champ_objective(a)) not in s:
                return False
    return True


# ---------------------------------------------------------------------------------------------- wave functions
def const(wave: int) -> Callable:
    def fn(obs, t):
        return wave
    fn.__name__ = f"const_{WAVE_NAME[wave]}"
    return fn


def slow_push_schedule(obs, t) -> int:
    if t is None:
        raise ValueError("slow_push_schedule is a branch plan (needs the turn index), not a whole-game rule")
    return W_THIN if t < 2 else W_HOLD


def _fronts(obs):
    """(own front x, own front HP, enemy front x in OWN coordinates, enemy front HP) of the observing seat."""
    s = obs.side
    own_fx, own_fh = obs.state.side(s)[:2]
    en_fx, en_fh = obs.state.side(1 - s)[:2]
    return own_fx, own_fh, -en_fx, en_fh


def freeze_controller(obs, t) -> int:
    _, _, en_x, en_fh = _fronts(obs)
    return W_THIN if (en_fh > 0 and en_x <= -1 and en_fh >= FREEZE_THRESHOLD) else W_HOLD


def reactive_slow_push(obs, t) -> int:
    own_fx, own_fh, en_x, en_fh = _fronts(obs)
    contact = own_fh > 0 and en_fh > 0 and own_fx - en_x >= -1          # fronts fighting (zoning_natural.wave_rule)
    if not contact:
        return W_HOLD
    return W_THIN if own_fh <= en_fh + 2 else W_HOLD


def shove_away(obs, t) -> int:
    return W_CLEAR if obs.champ.side(1 - obs.side).absent > 0 else W_HOLD


WAVE_FNS: Dict[str, Callable] = {"hold": const(W_HOLD), "thin": const(W_THIN), "clear": const(W_CLEAR),
                                 "slow_push": slow_push_schedule, "freeze": freeze_controller,
                                 "reactive_push": reactive_slow_push, "shove_away": shove_away}
GAME_FNS = ("hold", "thin", "clear", "reactive_push", "freeze", "shove_away")      # usable in whole games


# ---------------------------------------------------------------------------------------------- plan and agent
class WavePlan(Plan):
    """Intervene on the wave component for `turns` turns of the branch (DEFER throughout; the policy's own
    distribution is reshaped by `intervene` on every present turn), then hand control back for good."""

    def __init__(self, name: str, wave_fn: Callable, turns: int, rule: str):
        if rule not in RULES:
            raise ValueError(f"unknown intervention rule {rule!r}")
        self.name, self.wave_fn, self.turns, self.rule = name, wave_fn, int(turns), rule

    def act(self, obs, t: int) -> Optional[int]:
        return DEFER if t < self.turns else None

    def transform(self, obs, t: int, p: np.ndarray) -> np.ndarray:
        if obs.legal == (CHAMP_ABSENT,):
            return p
        return intervene(p, obs.legal, self.wave_fn(obs, t), self.rule)


def wave_plan(kind: str, rule: str, turns: int, name: Optional[str] = None) -> WavePlan:
    return WavePlan(name or f"{rule[0]}:{kind}", WAVE_FNS[kind], turns, rule)


class InterventionAgent:
    """`agent` with its wave component forced by `wave_fn(obs, None)` under `rule`, as an environment agent."""

    def __init__(self, agent, wave_fn: Callable, rule: str, name: Optional[str] = None):
        if rule not in RULES:
            raise ValueError(f"unknown intervention rule {rule!r}")
        self.base, self.wave_fn, self.rule = agent, wave_fn, rule
        self.name = name or f"{getattr(agent, 'name', 'agent')}[{rule}:{getattr(wave_fn, '__name__', 'wave')}]"

    def distribution(self, obs) -> np.ndarray:
        p = np.asarray(self.base.distribution(obs), dtype=float)
        if obs.legal == (CHAMP_ABSENT,):
            return p
        return intervene(p, obs.legal, self.wave_fn(obs, None), self.rule)

    def probabilities(self, obs) -> np.ndarray:
        """Full-length vector (0 on illegal actions), as a NetAgent's."""
        full = np.zeros(len(obs.mask))
        full[list(obs.legal)] = self.distribution(obs)
        return full

    def act(self, obs, rng: Optional[np.random.Generator] = None) -> int:
        p = self.distribution(obs)
        if rng is None:
            return obs.legal[int(np.argmax(p))]
        return obs.legal[sample(rng, p)]

    def reset(self) -> None:
        f = getattr(self.base, "reset", None)
        if callable(f):
            f()

    def notify(self, info, side) -> None:
        f = getattr(self.base, "notify", None)
        if callable(f):
            f(info, side)


__all__ = ["intervene", "wave_replacement_is_legal", "const", "slow_push_schedule", "freeze_controller",
           "reactive_slow_push", "shove_away", "WAVE_FNS", "GAME_FNS", "WavePlan", "wave_plan", "InterventionAgent",
           "RULES", "RECALL", "FREEZE_THRESHOLD", "WAVE_OF", "WAVE_NAME"]
