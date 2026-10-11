"""Exact forward analysis of a policy pair: state occupancy and path statistics.

Because transitions are deterministic given actions and choices, the
distribution over states at turn k under mixed policies is computed exactly by
propagating probability mass forward (no sampling). Every payoff component is
accumulated separately, so the expected total decomposes exactly as

  value = (farm_A - farm_B) + (crash_A - crash_B) - (gank_loss_A - gank_loss_B).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .config import GameConfig, posture_of, wave_of, WAVE_SHORT
from .game import LaneGame, apply_choice
from .policy import Policy
from .state import State


def _pair():
    return [0.0, 0.0]


@dataclass
class PathStats:
    value: float = 0.0                         # expected total payoff to A
    crash_resources: List[float] = field(default_factory=_pair)
    gank_loss: List[float] = field(default_factory=_pair)
    farm: List[float] = field(default_factory=_pair)            # collected
    farm_available: List[float] = field(default_factory=_pair)  # opportunities created
    crashes: List[List[float]] = field(default_factory=lambda: [[0.0] * 3, [0.0] * 3])
    crash_hp: List[float] = field(default_factory=_pair)        # expected HP crashed
    vision_choices: List[float] = field(default_factory=_pair)
    contest: List[float] = field(default_factory=_pair)         # expected number of Contest turns
    action_freq: List[Dict[str, float]] = field(default_factory=lambda: [defaultdict(float), defaultdict(float)])
    wave_freq: List[Dict[str, float]] = field(default_factory=lambda: [defaultdict(float), defaultdict(float)])
    occupancy: List[Dict[State, float]] = field(default_factory=list)

    def to_dict(self, horizon: int):
        f = lambda v: [float(x) for x in v]
        return {
            "value": float(self.value),
            "farm": f(self.farm),
            "farm_available": f(self.farm_available),
            "farm_missed": [float(a - c) for a, c in zip(self.farm_available, self.farm)],
            "crash_resources": f(self.crash_resources),
            "gank_loss": f(self.gank_loss),
            "crashes_by_size": {"A": f(self.crashes[0]), "B": f(self.crashes[1])},
            "crash_hp": f(self.crash_hp),
            "vision_choices": f(self.vision_choices),
            "contest_share": [float(c) / horizon for c in self.contest],
            "action_freq": [{k: float(v) / horizon for k, v in d.items()} for d in self.action_freq],
            "wave_freq": [{k: float(v) / horizon for k, v in d.items()} for d in self.wave_freq],
            "support_size": [len(o) for o in self.occupancy],
        }


def path_statistics(game: LaneGame, pol_a: Policy, pol_b: Policy, keep_occupancy=False,
                    root: State | None = None, k0: int = 0) -> PathStats:
    cfg: GameConfig = game.cfg
    st = PathStats()
    dist: Dict[State, float] = {game.initial_state() if root is None else root: 1.0}
    for k in range(k0, game.horizon):
        if keep_occupancy:
            st.occupancy.append(dict(dist))
        nxt: Dict[State, float] = defaultdict(float)
        for s, w in dist.items():
            la, lb = game.legal(s, 0), game.legal(s, 1)
            pa = pol_a.move(k, s, 0, la)
            pb = pol_b.move(k, s, 1, lb)
            for side, (leg, p) in enumerate(((la, pa), (lb, pb))):
                for i, a in enumerate(leg):
                    st.action_freq[side][cfg.label(a)] += w * p[i]
                    st.wave_freq[side][WAVE_SHORT[wave_of(a)]] += w * p[i]
                    if posture_of(a) == 0:
                        st.contest[side] += w * p[i]
            for i, a in enumerate(la):
                for j, b in enumerate(lb):
                    p = w * pa[i] * pb[j]
                    if p == 0:
                        continue
                    tr = game.transition(s, k, a, b)
                    st.value += p * tr.payoff
                    for side in (0, 1):
                        st.gank_loss[side] += p * cfg.gank_loss * tr.gank_p[side]
                        st.farm[side] += p * tr.farm[side]
                        st.farm_available[side] += p * tr.farm_available[side]
                    ca = (pol_a.choose(k, tr.pre_state, 0, tr.options_a, tr.options_b)
                          if len(tr.options_a) > 1 else np.ones(1))
                    cb = (pol_b.choose(k, tr.pre_state, 1, tr.options_b, tr.options_a)
                          if len(tr.options_b) > 1 else np.ones(1))
                    for side, hp in enumerate(tr.crash_hp):
                        if hp > 0:
                            st.crashes[side][cfg.size_category(hp)] += p
                            st.crash_hp[side] += p * hp
                    for ii, oa in enumerate(tr.options_a):
                        for jj, ob in enumerate(tr.options_b):
                            q = p * ca[ii] * cb[jj]
                            if q == 0:
                                continue
                            s2, r = apply_choice(tr.pre_state, oa, ob, cfg)
                            st.value += q * r
                            st.crash_resources[0] += q * oa[0]
                            st.crash_resources[1] += q * ob[0]
                            st.vision_choices[0] += q * (oa[1] and len(tr.options_a) > 1)
                            st.vision_choices[1] += q * (ob[1] and len(tr.options_b) > 1)
                            nxt[s2] += q
        dist = nxt
    return st
