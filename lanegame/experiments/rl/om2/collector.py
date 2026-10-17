"""The OM2 training collector (design §5): the M4.6c mixed start distribution, a library opponent mix, and one belief per
learner seat through the trainer's opt-in hooks (`SharedCollector.extra_obs / augment / observe_step`).

Matchups. With probability P_LIB a game is against a fixed opponent drawn uniformly from TRAIN_OPPONENTS, the learner
in a uniformly drawn seat (a HISTORICAL matchup: only the learner's transitions are samples; the opponent's name is
"lib:<name>"); otherwise the trainer's own sampler decides (100% current-policy self-play in the M4.6c settings). The
draws use the collector's matchup generator, so both arms of a seed play the same matchups in the same order.

Beliefs. Every environment keeps a Belief per seat (about the other seat), reset at every game start. After each step,
the TREATMENT arm updates the belief of every learner seat with the opponent's action and the opponent's start-of-turn
state; the learner's network input is its observation followed by the belief's features. The CONTROL arm's input is its
observation followed by the prior (constant); it computes no belief.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from . import common as C


class LibrarySampler:
    def __init__(self, base, names, agents, p_lib: float):
        self.base, self.names, self.agents, self.p_lib = base, list(names), list(agents), float(p_lib)
        self.counts: Dict[str, int] = {}

    def sample(self, rng: np.random.Generator):
        from ....rl.shared import HISTORICAL, LEARNER, Matchup
        if rng.random() < self.p_lib:
            j = int(rng.integers(len(self.agents)))
            seat = int(rng.integers(2))
            opp = self.agents[j]
            m = Matchup(HISTORICAL, (LEARNER, opp) if seat == 0 else (opp, LEARNER), f"lib:{self.names[j]}", seat)
        else:
            m = self.base.sample(rng)
        self.counts[m.opponent] = self.counts.get(m.opponent, 0) + 1
        return m

    def describe(self) -> dict:
        return {"p_lib": self.p_lib, "opponents": self.names, "draws": dict(self.counts)}


def om_collector_cls(arm: str, p_lib: float = C.P_LIB, lam: float = C.LAMBDA):
    """A collector class for one arm (bound to the M4.6c archive and start mixture)."""
    if arm not in C.ARMS:
        raise ValueError(arm)
    from ..m46c.train import collector as mixed
    base = mixed()

    class OMCollector(base):
        ARM, P_LIB, LAMBDA = arm, float(p_lib), float(lam)
        extra_obs = C.K

        def __init__(self, game_cfg, cfg, sampler, seed: int):
            from ....opponent import Belief, prior_features
            if not game_cfg.obs_last_action:
                raise ValueError("OM2 trains with obs_last_action on")
            self.types = C.belief_types()
            self.treatment = self.ARM == "treatment"
            self.prior = prior_features(C.K)
            self.beliefs = [[Belief(C.K, self.LAMBDA), Belief(C.K, self.LAMBDA)] for _ in range(cfg.n_envs)]
            self.lib = LibrarySampler(sampler, C.TRAIN_OPPONENTS, [C.agent(n) for n in C.TRAIN_OPPONENTS],
                                      self.P_LIB)
            self.belief_updates = 0
            self._pending: List[tuple] = []                   # (env, learner seat, likelihood item) not yet applied
            super().__init__(game_cfg, cfg, self.lib, seed)

        def _reset(self, i: int) -> None:
            # a new game: drop the old game's unapplied updates of this environment, then reset its beliefs
            self._pending = [p for p in self._pending if p[0] != i]
            super()._reset(i)
            for b in self.beliefs[i]:
                b.reset()

        def _flush(self) -> None:
            """Apply every pending update in one batched likelihood evaluation (each belief gets at most one update
            per step, so the order of application is immaterial)."""
            if not self._pending:
                return
            from ....opponent import loglik
            ll = loglik(self.types, self.game_cfg, [p[2] for p in self._pending])
            for (i, s, _), row in zip(self._pending, ll):
                self.beliefs[i][s].update(row)
            self.belief_updates += len(self._pending)
            self._pending = []

        def augment(self, i: int, seat: int, vector: np.ndarray) -> np.ndarray:
            if self.treatment:
                self._flush()
                f = self.beliefs[i][seat].features()
            else:
                f = self.prior
            return np.concatenate([np.asarray(vector, dtype=np.float32), f])

        def observe_step(self, i: int, actions, result) -> None:
            if not self.treatment:
                return
            from ....rl.shared import LEARNER
            ctrl = self.match[i].controllers
            for s in (0, 1):
                if ctrl[s] is LEARNER:
                    o = self.obs[i][1 - s]                    # the opponent's start-of-turn observation
                    self._pending.append((i, s, (o.state, o.champ, 1 - s, o.k, int(actions[1 - s]), o.vector,
                                                 o.legal)))

        def describe_starts(self) -> dict:
            d = super().describe_starts()
            d["om2"] = {"arm": self.ARM, "belief_types": list(C.BELIEF_TYPES), "lambda": self.LAMBDA,
                        "library": self.lib.describe(), "belief_updates": self.belief_updates}
            return d

    return OMCollector


__all__ = ["LibrarySampler", "om_collector_cls"]
