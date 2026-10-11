"""Exact forward analysis of a policy pair: state occupancy and path statistics.

Because transitions are deterministic given actions and choices, the
distribution over states at turn k under mixed policies is computed exactly by
propagating probability mass forward (no sampling). Every payoff component is
accumulated separately, so the expected total decomposes exactly as

  value = (farm_A - farm_B) + (plates_A - plates_B) + (crash_A - crash_B) - (gank_loss_A - gank_loss_B).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .config import GameConfig, PRESSURE, VISION, objective_of, posture_of, wave_of, WAVE_SHORT
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
    vision_choices: List[float] = field(default_factory=_pair)  # "choice" mode: vision options taken
    contest: List[float] = field(default_factory=_pair)         # expected number of Contest turns
    action_freq: List[Dict[str, float]] = field(default_factory=lambda: [defaultdict(float), defaultdict(float)])
    wave_freq: List[Dict[str, float]] = field(default_factory=lambda: [defaultdict(float), defaultdict(float)])
    occupancy: List[Dict[State, float]] = field(default_factory=list)
    # siege objectives ("objective" mode)
    plates: List[float] = field(default_factory=_pair)          # plate income (resources)
    plate_count: List[float] = field(default_factory=_pair)     # expected plates claimed
    pressure_turns: List[float] = field(default_factory=_pair)  # expected Apply Pressure turns
    vision_turns_set: List[float] = field(default_factory=_pair)  # expected Establish Vision turns
    siege_turns: List[float] = field(default_factory=_pair)     # expected turns with an eligible siege
    siege_safe_turns: List[float] = field(default_factory=_pair)  # ... spent disengaged (Safe)
    thinned_hp: List[float] = field(default_factory=_pair)      # expected enemy HP thinned
    thin_farm: List[float] = field(default_factory=_pair)       # expected farm collected from thinning
    protected_turns: List[float] = field(default_factory=_pair)  # turns started with active vision
    protected_contest: List[float] = field(default_factory=_pair)  # ... in which the player contested
    unprotected_contest: List[float] = field(default_factory=_pair)
    bounces: List[float] = field(default_factory=_pair)         # expected bounces (>= 2 waves released)
    bounce_hp: List[float] = field(default_factory=_pair)       # expected HP of released defender waves
    sieges_started: List[float] = field(default_factory=_pair)  # crashes onto an empty tower

    def to_dict(self, horizon: int):
        f = lambda v: [float(x) for x in v]
        prot = [float(c) / p if p > 1e-12 else None for c, p in zip(self.protected_contest, self.protected_turns)]
        unp = [float(c) / (horizon - p) if horizon - p > 1e-12 else None
               for c, p in zip(self.unprotected_contest, self.protected_turns)]
        return {
            "value": float(self.value),
            "farm": f(self.farm),
            "farm_available": f(self.farm_available),
            "farm_missed": [float(a - c) for a, c in zip(self.farm_available, self.farm)],
            "crash_resources": f(self.crash_resources),
            "plates": f(self.plates),
            "plate_count": f(self.plate_count),
            "gank_loss": f(self.gank_loss),
            "crashes_by_size": {"A": f(self.crashes[0]), "B": f(self.crashes[1])},
            "crash_hp": f(self.crash_hp),
            "vision_choices": f(self.vision_choices),
            "pressure_turns": f(self.pressure_turns),
            "vision_turns_set": f(self.vision_turns_set),
            "siege_turns": f(self.siege_turns),
            "siege_safe_turns": f(self.siege_safe_turns),
            "thinned_hp": f(self.thinned_hp),
            "thin_farm": f(self.thin_farm),
            "protected_turns": f(self.protected_turns),
            "p_contest_protected": prot,
            "p_contest_unprotected": unp,
            "bounces": f(self.bounces),
            "bounce_hp": f(self.bounce_hp),
            "sieges_started": f(self.sieges_started),
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
    W = cfg.wave_hp
    for k in range(k0, game.horizon):
        if keep_occupancy:
            st.occupancy.append(dict(dist))
        nxt: Dict[State, float] = defaultdict(float)
        for s, w in dist.items():
            la, lb = game.legal(s, 0), game.legal(s, 1)
            pa = pol_a.move(k, s, 0, la)
            pb = pol_b.move(k, s, 1, lb)
            for side, (leg, p) in enumerate(((la, pa), (lb, pb))):
                protected = s.vision_of(side) > 0
                st.protected_turns[side] += w * protected
                if s.siege_of(side) > 0 and cfg.objectives:
                    st.siege_turns[side] += w
                for i, a in enumerate(leg):
                    q = w * p[i]
                    st.action_freq[side][cfg.label(a)] += q
                    st.wave_freq[side][WAVE_SHORT[wave_of(a)]] += q
                    if posture_of(a) == 0:
                        st.contest[side] += q
                        if protected:
                            st.protected_contest[side] += q
                        else:
                            st.unprotected_contest[side] += q
                    elif s.siege_of(side) > 0 and cfg.objectives:
                        st.siege_safe_turns[side] += q
                    o = objective_of(a)
                    if o == PRESSURE:
                        st.pressure_turns[side] += q
                    elif o == VISION:
                        st.vision_turns_set[side] += q
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
                        st.plates[side] += p * tr.plates[side]
                        st.plate_count[side] += p * (tr.plates[side] / cfg.plate_value if cfg.plate_value else 0.0)
                        st.thinned_hp[side] += p * tr.thinned[side]
                        if tr.thinned[side]:
                            # thinned minions die beyond the enemy tower cell (own x >= T)
                            from .game import collection_efficiency
                            act_s = a if side == 0 else b
                            act_o = b if side == 0 else a
                            st.thin_farm[side] += p * cfg.farm_per_hp * tr.thinned[side] * collection_efficiency(
                                cfg.tower + 1, posture_of(act_s), posture_of(act_o), cfg)
                        # bounce: the enemy's crashed mass was cleared this turn and this side's
                        # leading wave is released with at least two waves of HP
                        enemy_siege_before = s.siege_of(1 - side)
                        enemy_siege_after = tr.pre_state.siege_of(1 - side)
                        own_front_hp = tr.pre_state.ah if side == 0 else tr.pre_state.bh
                        if enemy_siege_before > 0 and enemy_siege_after == 0 and own_front_hp >= 2 * W:
                            st.bounces[side] += p
                            st.bounce_hp[side] += p * own_front_hp
                        if tr.crash_hp[side] > 0 and s.siege_of(side) == 0:
                            st.sieges_started[side] += p
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
