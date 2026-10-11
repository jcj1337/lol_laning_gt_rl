"""A simultaneous-move, two-player environment around the existing laning game.

    env = LaneEnv(cfg)
    obs_a, obs_b = env.reset(seed=42)
    result = env.step(action_a, action_b)        # both actions at once; nothing is revealed in between

One `step` is one decision turn of the exact game, resolved by the existing engine: the lane advances with
`mechanics.step_turn`, farm and plates are paid as in `game.LaneGame.transition`, and the two gank checks
are REALIZED (one Bernoulli draw per player per turn from the environment's own generator) instead of
being charged as expected losses. The per-turn reward to A is therefore

    r = (farm_A - farm_B) + (plates_A - plates_B) + L * (ganked_B - ganked_A) + (crash_A - crash_B),

B receives -r (zero-sum by construction), and the expectation of an episode's total over the chance draws
equals the exact value of the two policies (`best_response.evaluate`), which the tests check.

Crash-reward choices. In the "choice" siege mode of the earlier configurations (`GameConfig.v2()`,
`GameConfig.v1()`) a side that crashed picks one reward option after the turn (simultaneously if both
crashed). That is a second, strategic decision stage; the environment resolves it inside the same `step`
through the `choosers` callbacks `(k, pre_state, side, opts_self, opts_opp) -> index`, which
`agents.play_episode` wires to the agents' `choose` methods. Without a chooser the greedy default
(most resources) is used and `StepResult.info["choice_default_used"]` records that the trajectory is then
NOT a faithful rollout of an arbitrary policy. The siege model ("objective" mode) has no such stage.

Randomness. `reset(seed)` creates the environment's chance generator; it is used only for the gank draws,
in a fixed order (A then B), and never for agents' action sampling. Two rollouts with the same seed and the
same action sequences produce identical chance draws; the uniform draws themselves are reported in
`info["gank_draws"]` so that policies can be compared on common random numbers.

Champion mode (`GameConfig.champions()`, roadmap M2a). The environment additionally holds the champion
state (`env.champ`, lanegame.champions.ChampState), legality comes from `champion_legal_actions`, and one
step is `champions.resolve_turn` (persistent positions, Hold / Yield / Trade / Safe, physical reach); farm,
plates, exposure and gank probabilities come from it, and the gank checks are realized exactly as above.
There is no crash-choice stage (the siege model has none). Observations carry `champ` as well.

The lane step is deterministic, so it can be cached by (state, k mod phase_period, a, b). The cache here is
bounded (LRU, `cache_size` entries; 0 disables it); the unbounded cache of `LaneGame` is for enumeration.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from ..config import B_TRADE, BEHAVIOR_NAMES, GameConfig, objective_of
from ..game import apply_choice, champion_exposure, crash_options, turn_farm, turn_gank_probabilities
from ..mechanics import TurnResult, initial_state, step_turn
from ..state import State
from .observation import legal_mask, legal_set, n_actions, obs_size, observe

A, B = 0, 1

# (k, pre-choice state, side, own options, opponent's options) -> index into own options
Chooser = Callable[[int, State, int, tuple, tuple], int]


def greedy_chooser(k: int, pre_state: State, side: int, opts_self, opts_opp) -> int:
    """Default crash-reward choice ("choice" siege mode only): the option with the most resources."""
    return int(np.argmax([o[0] for o in opts_self]))


@dataclass(eq=False)
class Observation:
    """What one player sees at the start of a turn. `vector` and `mask` are functions of (k, state, side)."""
    k: int
    side: int
    state: State                 # the shared lane state (perfect information)
    legal: Tuple[int, ...]       # legal action codes of this side, sorted (the engine's own legal set)
    vector: np.ndarray           # own-perspective features, float32, length env.obs_size
    mask: np.ndarray             # bool, length env.n_actions, True exactly on `legal`
    champ: Optional[object] = None   # champion mode: the shared ChampState (perfect information)


@dataclass(eq=False)
class StepResult:
    obs: Tuple[Observation, Observation]    # observations of (A, B) at the start of the next turn
    rewards: Tuple[float, float]            # (r, -r): realized payoff to A this turn and its negative
    terminated: bool                        # the horizon was reached
    truncated: bool                         # max_turns was reached before the horizon
    info: dict                              # every event and payoff component of the turn (JSON-able)

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


class LaneEnv:
    """Two-player simultaneous-move environment over `GameConfig` (siege model, farming revision or v1)."""

    def __init__(self, cfg: GameConfig, max_turns: Optional[int] = None,
                 choosers: Optional[Tuple[Optional[Chooser], Optional[Chooser]]] = None,
                 cache_size: int = 0):
        cfg.validate()
        if max_turns is not None and max_turns <= 0:
            raise ValueError("max_turns must be positive (or None)")
        self.cfg = cfg
        self.max_turns = max_turns
        self.choosers = choosers
        self.n_actions = n_actions(cfg)
        self.obs_size = obs_size(cfg)
        self._cache_size = int(cache_size)
        self._cache: "OrderedDict[tuple, TurnResult]" = OrderedDict()
        self._rng: Optional[np.random.Generator] = None
        self._state: Optional[State] = None
        self._champ = None                      # champion mode: ChampState
        self._k = 0
        self._k0 = 0
        self._done = True
        self._legal: Tuple[Tuple[int, ...], Tuple[int, ...]] = ((), ())

    # ------------------------------------------------------------------ introspection
    @property
    def horizon(self) -> int:
        return self.cfg.horizon

    @property
    def k(self) -> int:
        return self._k

    @property
    def k0(self) -> int:
        return self._k0

    @property
    def state(self) -> Optional[State]:
        return self._state

    @property
    def champ(self):
        """Champion state (champion mode), else None."""
        return self._champ

    @property
    def champion_mode(self) -> bool:
        return self.cfg.champion_mode

    @property
    def done(self) -> bool:
        return self._done

    @property
    def chance_rng(self) -> Optional[np.random.Generator]:
        return self._rng

    def action_label(self, a: int) -> str:
        return self.cfg.label(a)

    def legal(self, side: int) -> Tuple[int, ...]:
        """Legal action codes of `side` at the current turn (the engine's `legal_actions`)."""
        self._require_episode()
        return self._legal[side]

    def masks(self) -> Tuple[np.ndarray, np.ndarray]:
        self._require_episode()
        return (legal_mask(self._state, A, self.cfg, self.n_actions, self._champ),
                legal_mask(self._state, B, self.cfg, self.n_actions, self._champ))

    def observe(self) -> Tuple[Observation, Observation]:
        """Both players' observations of the current turn (a function of (k, state) only)."""
        self._require_episode()
        return self._observe()

    # ------------------------------------------------------------------ episode control
    def reset(self, seed=None, root: Optional[State] = None, k0: int = 0, champ=None) -> Tuple[Observation, Observation]:
        """Start an episode at decision index k0 (default: the standard initial state at k0 = 0).

        `seed` seeds the environment's chance generator (int, SeedSequence or None for fresh entropy).
        Champion mode: `champ` is the starting champion state (default: both at their own tower, full HP)."""
        if not 0 <= k0 < self.cfg.horizon:
            raise ValueError(f"k0 must be in [0, {self.cfg.horizon}); got {k0}")
        if champ is not None and not self.cfg.champion_mode:
            raise ValueError("a champion state was given but the configuration is not in champion mode")
        self._rng = np.random.default_rng(seed)
        self._state = initial_state(self.cfg) if root is None else root
        if self.cfg.champion_mode:
            from ..champions import initial_champions
            self._champ = initial_champions(self.cfg) if champ is None else champ
        self._k = self._k0 = int(k0)
        self._done = False
        return self._observe()

    def step(self, action_a: int, action_b: int) -> StepResult:
        """Resolve one turn from both players' simultaneous actions."""
        if self._state is None or self._done:
            raise RuntimeError("no running episode: call reset() before step(), and after the episode has ended")
        a, b = int(action_a), int(action_b)
        la, lb = self._legal
        if a not in la:
            raise ValueError(f"illegal action {a} for A at turn {self._k} (legal: {la})")
        if b not in lb:
            raise ValueError(f"illegal action {b} for B at turn {self._k} (legal: {lb})")
        cfg, s, k = self.cfg, self._state, self._k
        champ_before, ct = self._champ, None

        # ---- MOVE stage: deterministic lane step, farm, plates (exactly LaneGame.transition's numbers;
        #      champion mode: champions.resolve_turn)
        if cfg.champion_mode:
            from ..champions import resolve_turn
            ct = resolve_turn(s, champ_before, k, a, b, cfg, lane_step=self._lane_turn)
            res = ct.lane
            pa, pb = ct.gank_p
            avail, coll = ct.farm_available, ct.farm
            exposure = ct.exposure
        else:
            res = self._lane_turn(s, k, a, b)
            pa, pb = turn_gank_probabilities(s, a, b, res, cfg)
            avail, coll = turn_farm(res, a, b, cfg)
            exposure = (champion_exposure(res.exposure[A], a, cfg), champion_exposure(res.exposure[B], b, cfg))
        plates = (cfg.plate_value * res.plates[A], cfg.plate_value * res.plates[B])
        farm_payoff = coll[A] - coll[B]
        plate_payoff = plates[A] - plates[B]

        # ---- realized ganks: one Bernoulli draw per player, A first, from the environment's stream
        ua = self._rng.random()
        ub = self._rng.random()
        ga, gb = bool(ua < pa), bool(ub < pb)
        gank_payoff = cfg.gank_loss * (int(gb) - int(ga))

        # ---- CHOICE stage ("choice" siege mode; single trivial option otherwise)
        opts_a = crash_options(res.crash_hp[A], cfg)
        opts_b = crash_options(res.crash_hp[B], cfg)
        i = j = 0
        default_used = False
        if len(opts_a) > 1:
            i, d = self._choose(A, k, res.state, opts_a, opts_b)
            default_used = default_used or d
        if len(opts_b) > 1:
            j, d = self._choose(B, k, res.state, opts_b, opts_a)
            default_used = default_used or d
        s2, crash_payoff = apply_choice(res.state, opts_a[i], opts_b[j], cfg)

        reward = farm_payoff + plate_payoff + gank_payoff + crash_payoff
        self._state = s2
        if ct is not None:
            self._champ = ct.champ
        self._k = k + 1
        terminated = self._k >= cfg.horizon
        truncated = (not terminated) and self.max_turns is not None and (self._k - self._k0) >= self.max_turns
        self._done = terminated or truncated
        obs = self._observe()
        info = {
            "k": k,
            "actions": (a, b),
            "action_labels": (cfg.label(a), cfg.label(b)),
            "legal": (la, lb),
            "reward": reward,
            "farm_payoff": farm_payoff,
            "plate_payoff": plate_payoff,
            "gank_payoff": gank_payoff,
            "crash_payoff": crash_payoff,
            "farm": coll,
            "farm_available": avail,
            "plates": plates,
            "plate_count": res.plates,
            "gank_p": (pa, pb),
            "gank_draws": (ua, ub),
            "ganked": (ga, gb),
            "crash_hp": res.crash_hp,
            "crash_options": (opts_a, opts_b),
            "choices": (i, j),
            "choice_default_used": default_used,
            "exposure": exposure,
            "objectives": ct.objectives if ct is not None else (objective_of(a), objective_of(b)),
            "vision_set": res.vision_set,
            "vision": (s2.va, s2.vb),
            "plates_pending": (s2.ap, s2.bp),
            "thinned": res.thinned,
            "events": res.events,
            "kills": res.kills,
            "state_before": s.to_list(),
            "state_pre_choice": res.state.to_list(),
            "state_after": s2.to_list(),
        }
        if ct is not None:
            info.update({
                "behaviors": ct.behaviors,
                "behavior_labels": tuple(BEHAVIOR_NAMES[x] for x in ct.behaviors),
                "effective_objectives": ct.effective_objectives,
                "champ_start": ct.start,
                "champ_end": ct.end,
                "farm_spot": ct.farm_spot,
                "away": ct.away,
                "recalled": ct.recalled,
                "recall_interrupted": ct.interrupted,
                "threatened": ct.threatened,
                "trade_attempted": tuple(x == B_TRADE for x in ct.behaviors),
                "trade_connected": ct.connected,
                "modes": ct.modes,
                "champion_reach": ct.reach,
                "lane_codes": ct.lane_codes,
                "champ_before": champ_before.to_list(),
                "champ_after": ct.champ.to_list(),
            })
        return StepResult(obs, (reward, -reward), terminated, truncated, info)

    # ------------------------------------------------------------------ internals
    def _require_episode(self) -> None:
        if self._state is None:
            raise RuntimeError("no episode: call reset() first")

    def _observe(self) -> Tuple[Observation, Observation]:
        s, k, cfg, n, ch = self._state, self._k, self.cfg, self.n_actions, self._champ
        la, lb = legal_set(s, A, cfg, ch), legal_set(s, B, cfg, ch)
        self._legal = (la, lb)
        ma = np.zeros(n, dtype=bool)
        ma[list(la)] = True
        mb = np.zeros(n, dtype=bool)
        mb[list(lb)] = True
        return (Observation(k, A, s, la, observe(s, k, A, cfg, ch), ma, ch),
                Observation(k, B, s, lb, observe(s, k, B, cfg, ch), mb, ch))

    def _lane_turn(self, s: State, k: int, a: int, b: int, reach=None) -> TurnResult:
        if self._cache_size <= 0:
            return step_turn(s, k, a, b, self.cfg, champion_reach=reach)
        key = (s, k % self.cfg.phase_period, a, b, reach)
        res = self._cache.get(key)
        if res is None:
            res = step_turn(s, k, a, b, self.cfg, champion_reach=reach)
            self._cache[key] = res
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(key)
        return res

    def _choose(self, side: int, k: int, pre_state: State, opts_self, opts_opp) -> Tuple[int, bool]:
        ch = self.choosers[side] if self.choosers is not None else None
        if ch is None:
            return greedy_chooser(k, pre_state, side, opts_self, opts_opp), True
        i = int(ch(k, pre_state, side, opts_self, opts_opp))
        if not 0 <= i < len(opts_self):
            raise ValueError(f"chooser of side {side} returned {i}; {len(opts_self)} options")
        return i, False
