"""Policies of both arms for every M5 instrument (design §5.4: belief and last-action handling).

The om arm's network is history-dependent: its input is the obs_last_action observation (the 90-entry state encoding
and the opponent's previous action) followed by the belief's 13 features. The sp arm's network is the stateless
90-input policy of every earlier instrument. Two interfaces:

    engine loops   (m45b.engine.play, m45c.horizon.rollout, m46a.engine.rollout, diagnostics.lane_exit.continuation):
                   `crn_policy(spec, k0, seed, seat, horizon, ...)` -> an `m45c.horizon.CRNNet` for a 90-input network,
                   an `OMPolicy` for a 116-input one. Both sample by the inverse CDF of their policy with the uniform
                   u[i] of default_rng([seed, 7, seat]) (common random numbers across candidates and arms). The engines
                   run on `m45_calibrated()` (90 entries; the dynamics are those of the om configuration exactly, only
                   the observation differs); an OMPolicy builds its own obs_last_action observation from the
                   previous codes it has observed.
    s0 games       (`games.play`): the OM2 evaluation's players (`om2.evaluate.ArmPlayer` / `PiPlayer`), extended
                   with the belief modes below.

Belief and last-action conventions (identical to training, `om2.collector`):
    * a game or continuation that starts without a recorded history (s0, an archive or constructed state, a
      constructed cell) starts with the uniform prior and "no previous action" (-1, -1): exactly what training does at
      every game start, s0 and archive alike;
    * a continuation branched from a recorded om-arm game starts from the belief log posterior and the previous
      codes recorded at that turn (`start_info`), the same for every candidate of a paired comparison; a mirrored
      state swaps the two seats' records;
    * after every turn the belief of seat s is updated with the opponent's code and the opponent's start-of-turn
      observation (the likelihoods of `opponent.loglik` with lambda 0.97, eps 0.05); a turn on which the opponent
      could only be Absent changes nothing. The update uses only what both players saw: the start-of-turn state and the
      code just played. `decide` is pure (it never changes the belief), so the instruments may call it outside the
      turn loop;
    * a critic bootstrap (`value_at`) reads the input the policy would see at that state, belief included.
Belief modes (OMPolicy and the s0 players): "live" (the trained behaviour), "prior" (the 13 inputs held at the
uniform prior), ("table", T) (the inputs at turn k replaced by row min(k, len(T) - 1) of a fixed table, e.g. the mean
belief of the policy's own self-play: "no adaptation, typical inputs"; M5d's adaptation-exploitation test).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from . import common as C

_NETS: Dict[str, object] = {}
_TYPES = {}


def net_of(spec: str):
    """The network of an M5 policy specification or a checkpoint path (digest verified, cached per process); 90 (sp),
    103 (an attacker: the obs_last_action observation, no belief) or 116 (om) inputs, 61 outputs."""
    if spec not in _NETS:
        from ....rl import load_checkpoint
        path = spec if (spec.endswith(".pt") and not spec.startswith("spec:")) else C.policy_path(spec)
        net, _, _ = load_checkpoint(path)
        if net.n_actions != 61 or net.obs_size not in (C.OBS_SP, C.OBS_OM, C.NET_IN_OM):
            raise ValueError(f"{path}: {net.obs_size} inputs / {net.n_actions} outputs: not an M5 policy")
        net.eval()
        _NETS[spec] = net
    return _NETS[spec]


def install(spec: str, net) -> None:
    """Register an in-memory network under a specification (tests, smoke networks)."""
    _NETS[spec] = net


def belief_types():
    if "t" not in _TYPES:
        from ..om2 import common as O2
        _TYPES["t"] = O2.belief_types()
    return _TYPES["t"]


_TABLES: Dict[str, np.ndarray] = {}


def resolve_mode(mode):
    """"table:<path.npy>" -> ("table", the array) (cached); other modes unchanged."""
    if isinstance(mode, str) and mode.startswith("table:"):
        p = mode[6:]
        if p not in _TABLES:
            _TABLES[p] = np.load(p)
        return ("table", _TABLES[p])
    return mode


def _features_of(mode, belief, k: int) -> np.ndarray:
    from ....opponent import prior_features
    if mode == "live":
        return belief.features()
    if mode == "prior":
        return prior_features(C.K)
    if isinstance(mode, tuple) and mode[0] == "table":
        T = mode[1]
        return np.asarray(T[min(int(k), len(T) - 1)], dtype=np.float32)
    raise ValueError(f"unknown belief mode {mode!r}")


def observation_of(c, lane, m4, k: int, last, seat: int):
    """`diagnostics.branch.observation(c, Snapshot(lane, m4, k, (c.skill_a, c.skill_b), last), seat)` without
    rebuilding the configuration on every call (Snapshot.config returns a copy equal to `c` when the skills are c's):
    the same Observation, value for value (tested)."""
    from ....env.core import Observation
    from ....env.observation import legal_set, n_actions, observe_full
    legal = legal_set(lane, seat, c, m4)
    mask = np.zeros(n_actions(c), dtype=bool)
    mask[list(legal)] = True
    return Observation(int(k), seat, lane, legal, observe_full(lane, int(k), seat, c, m4, last), mask, m4, c)


class OMPolicy:
    """An om-arm network with its belief and the opponent's previous code, as an engine policy (module docstring). A
    103-input network (an M5d attacker) is the same without the belief: it tracks the previous codes only."""

    def __init__(self, net, k0: int, seed: int, seat: int, horizon: int, start_info: Optional[dict] = None,
                 greedy: bool = False, mode="live", name: str = "om"):
        from ....opponent import Belief
        if net.obs_size not in (C.OBS_OM, C.NET_IN_OM):
            raise ValueError(f"OMPolicy needs a {C.OBS_OM}- or {C.NET_IN_OM}-input network, not {net.obs_size}")
        self.has_belief = net.obs_size == C.NET_IN_OM
        self.net, self.k0, self.seat, self.greedy, self.name = net, int(k0), int(seat), bool(greedy), name
        self.mode = resolve_mode(mode)
        self.u = np.random.default_rng([int(seed), 7, int(seat)]).random(int(horizon) + 1)
        self.cfg_om = C.arm_cfg("om")
        self.types = belief_types()
        self.belief = Belief(C.K, C.LAMBDA)
        self.last: Tuple[int, int] = (-1, -1)
        if start_info:
            if start_info.get("lp") is not None:
                self.belief.lp = np.asarray(start_info["lp"], dtype=np.float64).copy()
                self.belief.updates = int(start_info.get("updates", 0))
            if start_info.get("last") is not None:
                self.last = (int(start_info["last"][0]), int(start_info["last"][1]))

    # -- the input
    def _obs(self, lane, m4, k: int, seat: int):
        return observation_of(self.cfg_om, lane, m4, k, self.last, seat)

    def features(self, k: int) -> np.ndarray:
        if not self.has_belief:
            return np.zeros(0, dtype=np.float32)
        return _features_of(self.mode, self.belief, k)

    def input(self, lane, m4, k: int):
        o = self._obs(lane, m4, k, self.seat)
        return o, np.concatenate([o.vector.astype(np.float32), self.features(k)])

    # -- the engine interface
    def probs(self, cfg, lane, m4, s: int, i: int):
        import torch
        if s != self.seat:
            raise ValueError(f"OMPolicy of seat {self.seat} asked to play seat {s}")
        o, x = self.input(lane, m4, self.k0 + i)
        with torch.no_grad():
            logits, _ = self.net(torch.as_tensor(x).unsqueeze(0))
        lg = logits[0].double().numpy()[list(o.legal)]
        p = np.exp(lg - lg.max())
        return o.legal, p / p.sum()

    def decide(self, cfg, lane, m4, s: int, i: int):
        legal, p = self.probs(cfg, lane, m4, s, i)
        if len(legal) == 1:
            return legal[0], "only"
        if self.greedy:
            return legal[int(np.argmax(p))], "modal"
        j = min(int(np.searchsorted(np.cumsum(p), self.u[i], side="right")), len(p) - 1)
        return legal[j], "sample"

    def observe_turn(self, cfg, lane, m4, k: int, codes) -> None:
        """The turn that started in (lane, m4, k) ended with `codes` (A's, B's): update the belief with the opponent's
        code on the opponent's start-of-turn observation (as om2.collector), then remember the codes."""
        opp = 1 - self.seat
        if self.mode == "live" and self.has_belief:
            from ....opponent import loglik
            o = self._obs(lane, m4, k, opp)
            item = (o.state, o.champ, opp, o.k, int(codes[opp]), o.vector, o.legal)
            self.belief.update(loglik(self.types, self.cfg_om, [item])[0])
        self.last = (int(codes[0]), int(codes[1]))

    def value_at(self, cfg, lane, m4, k: int, s: int = 0) -> float:
        import torch
        if s != self.seat:
            raise ValueError("value_at reads the policy's own seat")
        _, x = self.input(lane, m4, k)
        with torch.no_grad():
            return float(self.net.value(torch.as_tensor(x).unsqueeze(0))[0]) / 0.1

    def state(self) -> dict:
        return {"lp": self.belief.lp.tolist(), "updates": int(self.belief.updates), "last": list(self.last)}


def crn_policy(spec: str, k0: int, seed: int, seat: int, horizon: int, start_info: Optional[dict] = None,
               greedy: bool = False, mode="live"):
    """The engine policy of a specification: CRNNet (90 inputs, stateless) or OMPolicy (103 / 116 inputs, history)."""
    from ..m45c import horizon as HZ
    net = net_of(spec)
    if net.obs_size == C.OBS_SP:
        return HZ.CRNNet(net, k0, seed, seat, horizon, greedy=greedy, name=spec)
    return OMPolicy(net, k0, seed, seat, horizon, (start_info or {}).get(seat) if start_info else None, greedy, mode,
                    name=spec)


def critic_of(pol, spec: str):
    """The critic a rollout bootstraps with: the om policy object itself (its value reads its belief), else the net."""
    return pol if isinstance(pol, OMPolicy) else net_of(spec)


def mirror_info(info: Optional[dict]) -> Optional[dict]:
    """Seat records {0: {...}, 1: {...}} of a state whose seats are swapped (focal seat B -> seat A)."""
    if not info:
        return info
    out = {}
    for s, rec in info.items():
        r = dict(rec)
        if r.get("last") is not None:
            r["last"] = [r["last"][1], r["last"][0]]
        out[1 - int(s)] = r
    return out


__all__ = ["observation_of", "net_of", "install", "belief_types", "OMPolicy", "crn_policy", "critic_of", "mirror_info"]
