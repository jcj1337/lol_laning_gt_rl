"""Multi-turn strategic plans over the existing legal actions (M4.4c Part 1.1).

M4.4b's counterfactuals changed ONE component on ONE turn and returned control to the policy. A freeze, a slow push,
Vision before Pressure or a retreat before a Recall last several turns, so M4.4c compares *plans*: rules evaluated on
every turn of a branch, with preconditions, stop conditions, a maximum duration and explicit fallbacks.

A `StrategicPlan` is a `diagnostics.branch.Plan` (the M4.4a rollout protocol): on every turn while it is active its
`rule(obs, t, mem)` returns one decision

    code(a, label)                  play the complete action code a (must be legal; else the fallback)
    force(components, label)        reshape the policy's distribution so that the given components take the given
                                    values ({"wave": W_THIN}, {"behavior": B_SAFE}, {"wave": W_HOLD, "behavior":
                                    B_TRADE}, ...); every other component keeps the policy's own conditional
                                    distribution (the M4.4a "conditional" rule; `m44.interventions.intervene` for one
                                    component, the same rule on the joint slice for several)
    policy(label)                   the policy's own distribution this turn; the plan stays active
    stop(label)                     hand control back to the policy for good

and `stop(obs, t, mem)` (checked from t = 1 on, before the rule) ends the plan when its situation is over (the freeze
ended, the siege is gone, the champion left the lane). An absent champion ends the plan by default. A decision that is
not legal at that turn falls back to the policy for that turn (`fallback="policy"`, counted) or ends the plan
(`fallback="stop"`); a plan never plays an illegal or invented action. At the branch turn (t = 0) the first decision must
be legal (`applicable`): candidates are built only where they exist, as in M4.4a.

Pairing. A code decision still consumes the seat's single uniform of the turn and a forced distribution is sampled with
it (the rollout loop of `diagnostics.branch`), so every candidate sees the same environment chance and sampling streams
turn by turn: common random numbers until the trajectories diverge.

Every rollout's decisions are logged (`last`: turns applied, fallbacks, policy turns, the stop turn and reason), so a
reader can see how often a plan actually did what its name says.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ....config import B_HOLD, B_RECALL, B_SAFE, B_TRADE, CHAMP_ABSENT
from ....diagnostics.branch import DEFER, Plan
from ..m44.actions import RECALL, codes_with
from ..m44.interventions import intervene, possible

Decision = Tuple


def code(a: int, label: str = "") -> Decision:
    return ("code", int(a), label)


def force(components: Dict[str, int], label: str = "") -> Decision:
    return ("force", dict(components), label)


def policy(label: str = "") -> Decision:
    return ("policy", label)


def stop(label: str = "") -> Decision:
    return ("stop", label)


def possible_many(legal, components: Dict[str, int]) -> bool:
    """Some legal code has every given component value (Recall: the Recall code is legal)."""
    legal = tuple(int(a) for a in legal)
    if legal == (CHAMP_ABSENT,):
        return False
    if components.get("behavior") == B_RECALL:
        return RECALL in legal and len(components) == 1
    if len(components) == 1:
        (c, v), = components.items()
        return possible(legal, c, v)
    return len(codes_with(legal, **components)) > 0


def reshape(p: np.ndarray, legal, components: Dict[str, int], wave_rule: str = "conditional") -> np.ndarray:
    """The policy's distribution p over `legal` restricted to the codes with the given components and renormalized
    (uniform over that slice when the policy gives it no mass). One component: exactly `m44.interventions.intervene`."""
    legal = tuple(int(a) for a in legal)
    if len(components) == 1:
        (c, v), = components.items()
        return intervene(p, legal, c, v, wave_rule if c == "wave" else "conditional")
    sl = [legal.index(a) for a in codes_with(legal, **components)]
    if not sl:
        raise ValueError(f"no legal code with {components} (legal {legal})")
    q = np.zeros_like(np.asarray(p, dtype=float))
    mass = float(np.asarray(p)[sl].sum())
    q[sl] = np.asarray(p)[sl] / mass if mass > 0 else 1.0 / len(sl)
    return q


class StrategicPlan(Plan):
    """A multi-turn plan (module docstring). `rule(obs, t, mem)` -> decision; `stop(obs, t, mem)` -> falsy or a reason;
    `mem` is a per-rollout dict the rule may use (reset on every branch turn t = 0)."""

    def __init__(self, name: str, rule: Callable, max_turns: int = 10, stop: Optional[Callable] = None,
                 on_absent: str = "stop", fallback: str = "policy", wave_rule: str = "conditional",
                 description: str = ""):
        if fallback not in ("policy", "stop", "raise") or on_absent not in ("stop", "continue"):
            raise ValueError("fallback in {policy, stop, raise}, on_absent in {stop, continue}")
        self.name, self.rule, self.max_turns, self.stop_fn = name, rule, int(max_turns), stop
        self.on_absent, self.fallback, self.wave_rule, self.description = on_absent, fallback, wave_rule, description
        self.history: List[dict] = []          # one summary per rollout (`last` is the latest)
        self._reset()

    # ------------------------------------------------------------------ per rollout
    def _reset(self) -> None:
        self.mem: dict = {}
        self.done = False
        self.pending: Optional[Dict[str, int]] = None
        self.log: List[Tuple[int, str, str]] = []

    def _finish(self, t: int, status: str, why: str) -> None:
        self.done = True
        self.log.append((t, status, why))

    @property
    def last(self) -> dict:
        """Summary of the current / latest rollout's decisions."""
        n = {k: sum(1 for _, s, _ in self.log if s == k) for k in ("applied", "fallback", "policy", "absent")}
        end = next(((t, w) for t, s, w in self.log if s == "stopped"), (None, ""))
        return {**n, "stop_turn": end[0], "stop_reason": end[1],
                "active_turns": n["applied"] + n["fallback"] + n["policy"] + n["absent"]}

    def _fallback(self, obs, t: int, label: str) -> Optional[int]:
        if self.fallback == "raise":
            raise ValueError(f"{self.name}: {label} is not legal at turn {obs.k} (legal {obs.legal})")
        if self.fallback == "stop":
            self._finish(t, "stopped", f"not legal: {label}")
            return None
        self.pending = None
        self.log.append((t, "fallback", label))
        return DEFER

    # ------------------------------------------------------------------ Plan protocol
    def act(self, obs, t: int) -> Optional[int]:
        if t == 0:
            self._reset()
        if self.done:
            return None
        if obs.legal == (CHAMP_ABSENT,):
            if self.on_absent == "stop":
                self._finish(t, "stopped", "champion absent")
                return None
            self.pending = None
            self.log.append((t, "absent", ""))
            return DEFER
        if t >= self.max_turns:
            self._finish(t, "stopped", "window")
            return None
        if t > 0 and self.stop_fn is not None:
            why = self.stop_fn(obs, t, self.mem)
            if why:
                self._finish(t, "stopped", why if isinstance(why, str) else "stop condition")
                return None
        d = self.rule(obs, t, self.mem)
        kind = d[0]
        if kind == "stop":
            self._finish(t, "stopped", d[1])
            return None
        if kind == "policy":
            self.pending = None
            self.log.append((t, "policy", d[1]))
            return DEFER
        if kind == "code":
            a, label = int(d[1]), d[2]
            if a not in obs.legal:
                return self._fallback(obs, t, label or f"code {a}")
            self.pending = None
            self.log.append((t, "applied", label))
            return a
        if kind == "force":
            comps, label = d[1], d[2]
            if not possible_many(obs.legal, comps):
                return self._fallback(obs, t, label or str(comps))
            if comps.get("behavior") == B_RECALL:          # Recall is a single code, never a distribution slice
                self.pending = None
                self.log.append((t, "applied", label))
                return RECALL
            self.pending = comps
            self.log.append((t, "applied", label))
            return DEFER
        raise ValueError(f"{self.name}: unknown decision {d!r}")

    def transform(self, obs, t: int, p: np.ndarray) -> np.ndarray:
        if self.pending is None or obs.legal == (CHAMP_ABSENT,):
            return p
        return reshape(p, obs.legal, self.pending, self.wave_rule)

    # ------------------------------------------------------------------ preconditions
    def applicable(self, obs) -> bool:
        """The branch-turn decision exists at this state (legal code / possible components / the policy's own turn)."""
        if obs.legal == (CHAMP_ABSENT,):
            return False
        d = self.rule(obs, 0, {})
        if d[0] == "code":
            return int(d[1]) in obs.legal
        if d[0] == "force":
            return possible_many(obs.legal, d[1])
        return d[0] == "policy"


# ---------------------------------------------------------------------------------------------- common rules
def const_rule(decision: Decision) -> Callable:
    def rule(obs, t, mem):
        return decision
    return rule


def first_then(first: Decision, then: Decision) -> Callable:
    """`first` on the branch turn, `then` afterwards."""
    def rule(obs, t, mem):
        return first if t == 0 else then
    return rule


def one_turn(name: str, decision: Decision, description: str = "") -> StrategicPlan:
    """The M4.4b-style single-turn intervention as a StrategicPlan (the policy from turn 1 on)."""
    return StrategicPlan(name, const_rule(decision), max_turns=1, description=description or f"{name} on the branch turn")


RECALL_D = code(RECALL, "Recall")
SAFE_D = force({"behavior": B_SAFE}, "Safe")
HOLD_D = force({"behavior": B_HOLD}, "Hold")
TRADE_D = force({"behavior": B_TRADE}, "Trade")


__all__ = ["code", "force", "policy", "stop", "possible_many", "reshape", "StrategicPlan", "const_rule", "first_then",
           "one_turn", "RECALL_D", "SAFE_D", "HOLD_D", "TRADE_D", "RECALL"]
