"""Game layer: payoffs, gank risk, crash-reward choices and reachable states.

One decision turn k consists of two stages:

  MOVE stage    both players simultaneously pick a strategic action from their
                legal sets; each suffers an independent gank check (expected
                loss L * p) and the lane advances deterministically (mechanics).
  CHOICE stage  if a side crashed during the turn it picks one reward option
                for the crash size (resources vs. vision). If both crashed the
                choices are simultaneous. Choices are publicly observed before
                turn k+1.

All payoffs are written from player A's point of view; B receives the negative
(zero-sum relative resource advantage).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from .config import F, SH, SP, GameConfig
from .mechanics import TurnResult, initial_state, legal_actions, step_turn
from .state import State

A, B = 0, 1


# ---------------------------------------------------------------- gank model
def gank_probability(state: State, s: int, action: int, cfg: GameConfig, exposure: int) -> float:
    """p_s = clip(b0 + b_pos*(e+T)/(2T) + b_act[a] - b_str*dsigma, 0, p_max) * (1 - rho*[v_s > 0]).

    e is side s's exposure for the turn: the most advanced own-perspective
    position its leading wave occupies during the turn (the champion travels
    with its wave), clipped to [-T, T]. dsigma is the stack difference of the
    two leading waves at the start of the turn.
    """
    own_h = state.ah if s == A else state.bh
    opp_h = state.bh if s == A else state.ah
    dsig = cfg.stacks(own_h) - cfg.stacks(opp_h)
    p = (cfg.gank_base
         + cfg.gank_position * (exposure + cfg.tower) / (2 * cfg.tower)
         + cfg.gank_action[action]
         - cfg.gank_strength * dsig)
    p = min(max(p, 0.0), cfg.gank_max)
    v = state.va if s == A else state.vb
    if v > 0:
        p *= (1.0 - cfg.vision_reduction)
    return p


def champion_exposure(reach: int, action: int, cfg: GameConfig) -> int:
    cap = cfg.posture_cap[action]
    return reach if cap is None else min(reach, cap)


def turn_gank_probabilities(state: State, a: int, b: int, res: TurnResult, cfg: GameConfig):
    return (gank_probability(state, A, a, cfg, champion_exposure(res.exposure[A], a, cfg)),
            gank_probability(state, B, b, cfg, champion_exposure(res.exposure[B], b, cfg)))


# ----------------------------------------------------------- crash choices
def crash_options(crash_hp: int, cfg: GameConfig) -> Tuple[Tuple[float, bool], ...]:
    """Reward options (resources, vision) available after a crash of total HP crash_hp."""
    if crash_hp <= 0:
        return ((0.0, False),)
    return cfg.crash_options[cfg.size_category(crash_hp)]


def apply_choice(state: State, opt_a: Tuple[float, bool], opt_b: Tuple[float, bool],
                 cfg: GameConfig) -> Tuple[State, float]:
    """Apply crash-reward choices; return (next state, payoff to A)."""
    va = cfg.vision_turns if opt_a[1] else state.va
    vb = cfg.vision_turns if opt_b[1] else state.vb
    return state._replace(va=va, vb=vb), opt_a[0] - opt_b[0]


@dataclass(frozen=True)
class Transition:
    """Outcome of a MOVE-stage joint action, before the CHOICE stage."""
    pre_state: State                    # lane after the turn, vision counted down
    gank_payoff: float                  # expected gank payoff to A: L * (p_B - p_A)
    gank_p: Tuple[float, float]         # gank probabilities (p_A, p_B) for this turn
    options_a: Tuple[Tuple[float, bool], ...]
    options_b: Tuple[Tuple[float, bool], ...]
    crash_hp: Tuple[int, int]

    def choice_outcomes(self, cfg: GameConfig):
        """Yield (i, j, next_state, crash payoff to A) for every option pair."""
        for i, oa in enumerate(self.options_a):
            for j, ob in enumerate(self.options_b):
                s2, r = apply_choice(self.pre_state, oa, ob, cfg)
                yield i, j, s2, r


class LaneGame:
    """Finite-horizon zero-sum stochastic game with transition caching."""

    def __init__(self, cfg: GameConfig):
        cfg.validate()
        self.cfg = cfg
        self._cache: Dict[tuple, Transition] = {}

    @property
    def horizon(self) -> int:
        return self.cfg.horizon

    def initial_state(self) -> State:
        return initial_state(self.cfg)

    def legal(self, state: State, s: int) -> Tuple[int, ...]:
        return legal_actions(state, s, self.cfg)

    def transition(self, state: State, k: int, a: int, b: int) -> Transition:
        # The lane dynamics depend on k only through the spawn phase.
        phase = k % self.cfg.spawn_every_turns
        key = (state, phase, a, b)
        tr = self._cache.get(key)
        if tr is None:
            if a not in self.legal(state, A) or b not in self.legal(state, B):
                raise ValueError(f"illegal joint action ({a}, {b}) in {state}")
            res: TurnResult = step_turn(state, k, a, b, self.cfg)
            pa, pb = turn_gank_probabilities(state, a, b, res, self.cfg)
            tr = Transition(res.state, self.cfg.gank_loss * (pb - pa), (pa, pb),
                            crash_options(res.crash_hp[A], self.cfg),
                            crash_options(res.crash_hp[B], self.cfg),
                            res.crash_hp)
            self._cache[key] = tr
        return tr

    def successors(self, state: State, k: int) -> Iterable[State]:
        for a in self.legal(state, A):
            for b in self.legal(state, B):
                tr = self.transition(state, k, a, b)
                for _, _, s2, _ in tr.choice_outcomes(self.cfg):
                    yield s2

    def reachable(self, horizon: int | None = None, max_states: int = 5_000_000,
                  root: State | None = None, k0: int = 0) -> List[List[State]]:
        """States reachable at each decision index k under some play.

        Returns layers with len(layers) == horizon; layers[k] is sorted for
        determinism and empty for k < k0. By default the root is the standard
        initial state at k0 = 0. Raises if the total exceeds max_states, a
        safety limit against uncontrolled enumeration.
        """
        K = self.horizon if horizon is None else horizon
        root = self.initial_state() if root is None else root
        layers: List[List[State]] = [[] for _ in range(k0)] + [[root]]
        total = 1
        for k in range(k0, K - 1):
            nxt = set()
            for s in layers[k]:
                nxt.update(self.successors(s, k))
            layer = sorted(nxt)
            total += len(layer)
            if total > max_states:
                raise RuntimeError(f"reachable state count exceeded {max_states} at k={k + 1}")
            layers.append(layer)
        return layers
