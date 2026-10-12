"""A simultaneous-move, two-player environment around the existing laning game.

    env = LaneEnv(cfg)
    obs_a, obs_b = env.reset(seed=42)
    result = env.step(action_a, action_b)        # both actions at once; nothing is revealed in between

One `step` is one decision turn of the exact game, resolved by the existing engine: the lane advances with
`mechanics.step_turn`, farm and plates are paid as in `game.LaneGame.transition`, and the two gank checks
are REALIZED (one Bernoulli draw per player per turn from the environment's own generator) instead of
being charged as expected losses. The per-turn reward to A is therefore

    r = (farm_A - farm_B) + (plates_A - plates_B) + L * (ganked_B - ganked_A) + (crash_A - crash_B)

(champion mode: r = (farm_A - farm_B) + (plates_A - plates_B) + (bounty_A - bounty_B); a gank costs HP and
possibly a death, never resources),

B receives -r (zero-sum by construction), and the expectation of an episode's total over the chance draws
equals the exact value of the two policies (`best_response.evaluate`), which the tests check.

Crash-reward choices. In the "choice" siege mode of the earlier configurations (`GameConfig.v2()`,
`GameConfig.v1()`) a side that crashed picks one reward option after the turn (simultaneously if both
crashed). That is a second, strategic decision stage; the environment resolves it inside the same `step`
through the `choosers` callbacks `(k, pre_state, side, opts_self, opts_opp) -> index`, which
`agents.play_episode` wires to the agents' `choose` methods. Without a chooser the greedy default
(most resources) is used and `StepResult.info["choice_default_used"]` records that the trajectory is then
NOT a faithful rollout of an arbitrary policy. The siege model ("objective" mode) has no such stage.

Variance reduction (legacy configurations only, optional): `gank_reward="expected"` pays the gank term as its
conditional expectation L * (p_B - p_A) instead of the realized L * (ganked_B - ganked_A). In the legacy game a gank
changes nothing but the payoff (the lane transition does not depend on it), so every conditional expected reward,
every expected return and every policy gradient is unchanged (Rao-Blackwellization); only the Bernoulli noise is
removed. The draws are still made and reported in `info`. Champion mode refuses it (ganks change HP and absence).

Randomness. `reset(seed)` creates the environment's chance generator; it is used only for the gank draws,
in a fixed order (A then B), and never for agents' action sampling. Two rollouts with the same seed and the
same action sequences produce identical chance draws; the uniform draws themselves are reported in
`info["gank_draws"]` so that policies can be compared on common random numbers.

Champion mode (`GameConfig.champions()`, roadmap M2). The environment additionally holds the champion
state (`env.champ`, lanegame.champions.ChampState), legality comes from `champion_legal_actions`, and one
step is `champions.resolve_turn` (positions, Hold / Yield / Trade / Safe / Recall, absence, trades, tower
shots, ganks with HP damage, death). Each champion-mode turn draws exactly N_CHANCE uniforms from the
chance generator (champions.U_* layout), whatever the actions, so two policies are compared on identical
chance draws turn by turn; `info["chance"]` reports them. There is no crash-choice stage (the siege model
has none). Observations carry `champ` as well.

The lane step is deterministic, so it can be cached by (state, k mod phase_period, a, b). The cache here is
bounded (LRU, `cache_size` entries; 0 disables it); the unbounded cache of `LaneGame` is for enumeration.

Continuing time (rulesets "m2c" and "m4", `cfg.time_mode == "continuing"`; M4 design §10). There is no terminal
state: `terminated` is always False, `horizon` is None and is never used, and an episode ends only by truncation at
`max_turns` (a time limit of the run, not of the game: training bootstraps it, evaluation truncates a discounted
sum). `reset(k0)` accepts any k0 >= 0 and records only k0 mod phase_period, so `env.k` counts turns from that
phase; the dynamics and the observation depend on k only through k mod phase_period. Stepping after a truncation
raises until `reset`.

Ruleset "m4" (`GameConfig.m4()`): one step is `champions_m4.resolve_turn_m4`; `env.champ` is the M4State (positions,
centi-HP, absence, XP, items, spendable gold, tower damage). The reward is the exact ledger difference
`reward_units / gold_scale` (farm + plate gold + kill bounty); `info` carries the integer components
(`farm_units`, `plate_units`, `bounty_units`, `reward_units`) whose identity is exact, and the M4 events (levels, XP,
purchases, tower damage, plates, Freeze maintenance, the tower-adjacent shot).
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
    cfg: Optional[GameConfig] = None   # the configuration being played (skills are part of it)


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
                 cache_size: int = 0, gank_reward: str = "realized"):
        cfg.validate()
        if gank_reward not in ("realized", "expected"):
            raise ValueError("gank_reward must be 'realized' or 'expected'")
        if gank_reward == "expected" and cfg.champion_mode:
            raise ValueError("gank_reward='expected' applies to the legacy configurations only: in champion mode a "
                             "gank changes the state (HP, death), so its realized outcome cannot be replaced")
        self.gank_reward = gank_reward
        if max_turns is not None and max_turns <= 0:
            raise ValueError("max_turns must be positive (or None)")
        if cfg.continuing and gank_reward != "realized":
            raise ValueError("continuing rulesets are champion games: gank_reward must be 'realized'")
        self.cfg = cfg
        self.max_turns = max_turns
        self._ctx = None
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
    def horizon(self) -> Optional[int]:
        """The terminal horizon K (finite time); None under continuing time."""
        return None if self.cfg.continuing else self.cfg.horizon

    @property
    def continuing(self) -> bool:
        return self.cfg.continuing

    @property
    def turns(self) -> int:
        """Turns played since the last reset."""
        return self._k - self._k0

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
        Champion mode: `champ` is the starting champion state (default: both at their own tower, full HP; M4: the
        M4State of champions_m4.initial_m4). Continuing time: any k0 >= 0, of which only k0 mod phase_period is kept."""
        if self.cfg.continuing:
            if k0 < 0:
                raise ValueError(f"k0 must be >= 0; got {k0}")
            k0 = int(k0) % self.cfg.phase_period
        elif not 0 <= k0 < self.cfg.horizon:
            raise ValueError(f"k0 must be in [0, {self.cfg.horizon}); got {k0}")
        if champ is not None and not self.cfg.champion_mode:
            raise ValueError("a champion state was given but the configuration is not in champion mode")
        self._rng = np.random.default_rng(seed)
        self._state = initial_state(self.cfg) if root is None else root
        if self.cfg.is_m4:
            from ..champions_m4 import M4State, context, initial_m4
            if champ is not None and not isinstance(champ, M4State):
                raise TypeError("ruleset m4: the champion state must be a champions_m4.M4State")
            self._champ = initial_m4(self.cfg) if champ is None else champ
            self._ctx = context(self.cfg)
        elif self.cfg.champion_mode:
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

        if cfg.is_m4:
            return self._step_m4(a, b)

        # ---- MOVE stage: deterministic lane step, farm, plates (exactly LaneGame.transition's numbers;
        #      champion mode: champions.resolve_turn)
        if cfg.champion_mode:
            from ..champions import N_CHANCE, resolve_turn
            u = self._rng.random(N_CHANCE)
            ct = resolve_turn(s, champ_before, k, a, b, cfg, lane_step=self._lane_turn, u=u)
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
        #      (champion mode: the gank entries of the turn's chance vector, realized by resolve_turn)
        if ct is None:
            ua = self._rng.random()
            ub = self._rng.random()
            ga, gb = bool(ua < pa), bool(ub < pb)
        else:
            from ..champions import U_GANK
            ua, ub = float(u[U_GANK[A]]), float(u[U_GANK[B]])
            ga, gb = bool(ct.ganked[A]), bool(ct.ganked[B])
        # legacy: the realized gank costs gank_loss resources; champion mode: the gank's cost is its HP damage and
        # a possible death (absence), so the reward has no gank term, and kill bounties (default 0) are separate
        if ct is not None:
            gank_payoff = 0.0
        elif self.gank_reward == "expected":                 # legacy only: E[realized | s, a, b] (variance reduction)
            gank_payoff = cfg.gank_loss * (pb - pa)
        else:
            gank_payoff = cfg.gank_loss * (int(gb) - int(ga))
        bounty_payoff = 0.0 if ct is None else ct.bounty[A] - ct.bounty[B]

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

        reward = farm_payoff + plate_payoff + gank_payoff + crash_payoff + bounty_payoff
        self._state = s2
        if ct is not None:
            self._champ = ct.champ
        self._k = k + 1
        terminated = (not cfg.continuing) and self._k >= cfg.horizon
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
            "bounty_payoff": bounty_payoff,
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
                "hit_p": ct.hit_p,
                "double_p": ct.double_p,
                "tower_p": ct.tower_p,
                "trade_damage": ct.trade_damage,
                "tower_damage": ct.tower_damage,
                "damage_taken": ct.damage_taken,
                "gank_damage_p": ct.gank_damage_p,
                "gank_damage": ct.gank_damage,
                "died": ct.died,
                "bounty": ct.bounty,
                "chance": [float(x) for x in u],
                "modes": ct.modes,
                "champion_reach": ct.reach,
                "lane_codes": ct.lane_codes,
                "champ_before": champ_before.to_list(),
                "champ_after": ct.champ.to_list(),
            })
        return StepResult(obs, (reward, -reward), terminated, truncated, info)

    # ------------------------------------------------------------------ ruleset m4
    def _step_m4(self, a: int, b: int) -> StepResult:
        from ..champions import N_CHANCE, U_GANK
        from ..champions_m4 import resolve_turn_m4
        cfg, s, k, champ_before = self.cfg, self._state, self._k, self._champ
        la, lb = self._legal
        u = self._rng.random(N_CHANCE)
        ct = resolve_turn_m4(s, champ_before, k, a, b, cfg, lane_step=self._lane_turn, u=u, ctx=self._ctx)
        res = ct.lane
        s2 = res.state
        self._state, self._champ = s2, ct.champ
        self._k = k + 1
        truncated = self.max_turns is not None and (self._k - self._k0) >= self.max_turns
        self._done = truncated
        obs = self._observe()
        scale = self._ctx.scale
        fu, pu, bu = ct.farm_units, ct.plate_units, ct.bounty_units
        reward = ct.reward
        trivial = ((0.0, False),)
        info = {
            "k": k, "phase": k % cfg.phase_period,
            "actions": (a, b), "action_labels": (cfg.label(a), cfg.label(b)), "legal": (la, lb),
            "reward": reward, "reward_units": ct.reward_units,
            "farm_payoff": (fu[0] - fu[1]) / scale, "plate_payoff": (pu[0] - pu[1]) / scale,
            "gank_payoff": 0.0, "crash_payoff": 0.0, "bounty_payoff": (bu[0] - bu[1]) / scale,
            "farm_units": fu, "plate_units": pu, "bounty_units": bu, "earned_units": ct.earned_units,
            "farm": ct.farm, "farm_available": ct.farm_available, "plates": ct.plate_gold,
            "plate_count": ct.plates_broken,
            "gank_p": ct.gank_p, "gank_draws": (float(u[U_GANK[0]]), float(u[U_GANK[1]])), "ganked": ct.ganked,
            "crash_hp": res.crash_hp, "crash_options": (trivial, trivial), "choices": (0, 0),
            "choice_default_used": False, "exposure": ct.exposure, "objectives": ct.objectives,
            "vision_set": res.vision_set, "vision": (s2.va, s2.vb), "plates_pending": (s2.ap, s2.bp),
            "thinned": res.thinned, "events": res.events, "kills": res.kills,
            "state_before": s.to_list(), "state_pre_choice": s2.to_list(), "state_after": s2.to_list(),
            "behaviors": ct.behaviors, "behavior_labels": tuple(BEHAVIOR_NAMES[x] for x in ct.behaviors),
            "effective_objectives": ct.effective_objectives, "champ_start": ct.start, "champ_end": ct.end,
            "farm_spot": ct.farm_spot, "away": ct.away, "recalled": ct.recalled,
            "recall_interrupted": ct.interrupted, "threatened": ct.threatened,
            "trade_attempted": tuple(x == B_TRADE for x in ct.behaviors), "trade_connected": ct.connected,
            "hit_p": ct.hit_p, "double_p": ct.double_p, "tower_p": ct.tower_p, "adjacent_shot": ct.adjacent_shot,
            "trade_damage": ct.trade_damage, "tower_damage": ct.tower_damage, "damage_taken": ct.damage_taken,
            "gank_damage_p": ct.gank_damage_p, "gank_damage": ct.gank_damage, "died": ct.died, "bounty": ct.bounty,
            "chance": [float(x) for x in u], "modes": ct.modes, "champion_reach": ct.reach,
            "lane_codes": ct.lane_codes, "champ_before": champ_before.to_list(), "champ_after": ct.champ.to_list(),
            # ---- M4 events
            "waves": ct.waves, "effective_waves": ct.effective_waves, "freeze_maintained": ct.freeze_maintained,
            "levels_before": ct.levels_before, "levels": ct.levels_after, "xp_gain": ct.xp_gain,
            "xp_discarded": ct.xp_discarded, "level_up": ct.level_up,
            "xp": (ct.champ.a.xp, ct.champ.b.xp), "items": (ct.champ.a.items, ct.champ.b.items),
            "gold": (ct.champ.a.gold, ct.champ.b.gold), "tower_dmg": (ct.champ.a.tower_dmg, ct.champ.b.tower_dmg),
            "tower_hits": ct.tower_hits, "tower_damage_dealt": ct.tower_damage_dealt,
            "plates_broken": ct.plates_broken, "plate_gold": ct.plate_gold, "purchases": ct.purchases,
            "returned": ct.returned, "gold_unspendable": ct.gold_clamped, "invested": ct.invested,
            "pinned": ct.pinned,                                              # position_rule "pin"/"standoff"
        }
        return StepResult(obs, (reward, -reward), False, truncated, info)

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
        return (Observation(k, A, s, la, observe(s, k, A, cfg, ch), ma, ch, cfg),
                Observation(k, B, s, lb, observe(s, k, B, cfg, ch), mb, ch, cfg))

    def _lane_turn(self, s: State, k: int, a: int, b: int, reach=None, budget=None) -> TurnResult:
        if self._cache_size <= 0:
            return step_turn(s, k, a, b, self.cfg, champion_reach=reach, champion_budget=budget)
        key = (s, k % self.cfg.phase_period, a, b, reach, budget)
        res = self._cache.get(key)
        if res is None:
            res = step_turn(s, k, a, b, self.cfg, champion_reach=reach, champion_budget=budget)
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
