"""Opponent types (OM2, docs/opponent_modelling/om2_design.md §3).

A BELIEF TYPE reports the probability of every legal action of the opponent's seat in a state (`probs`), the likelihood
a modeller assigns to an observed action:
    ScriptedType   a deterministic scripted strategy (lanegame.experiments.m45b.strategies); the likelihood trembles:
                   (1 - eps) on the scripted action, eps spread evenly over the legal codes
    NetType        a learned network on the state encoding (its first `net.obs_size` observation entries); its exact
                   masked-softmax distribution; `probs_batch` evaluates many states in one forward pass
    UniformType    the catch-all "unknown": every legal action equally likely (novel play)
    LibraryMixtureType  the catch-all "mixed": a different library type each turn (per-turn mixtures of styles)
Every type can also PLAY a seat of a `LaneEnv` (`agent()`: an object with `.name` and `.act(obs, rng)`, the interface
of the trainer's frozen opponents). Opponents that are not belief types (training and evaluation only):
    MixtureAgent   each turn plays agent a with probability p, else b (a draw from the seat's generator)
    SwitchAgent    plays agent a while the observation's turn index k < turn, then b
Learned agents read the first `net.obs_size` entries of the observation, so 90-input networks play unchanged in an
`obs_last_action` environment (whose extra entries come after the state encoding).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..champions import champion_legal_actions


def _decide(strategy, cfg, lane, m4, s: int) -> int:
    if hasattr(strategy, "decide"):
        return int(strategy.decide(cfg, lane, m4, s, 0)[0])
    return int(strategy.act(cfg, lane, m4, s, 0))


def _sample(rng: Optional[np.random.Generator], p: np.ndarray) -> int:
    if rng is None:
        return int(np.argmax(p))
    from ..env.agents import sample
    return int(sample(rng, p))


# ---------------------------------------------------------------------------------------------- agents
class ScriptedAgent:
    def __init__(self, strategy, name: str):
        self.strategy, self.name = strategy, name

    def act(self, obs, rng=None) -> int:
        a = _decide(self.strategy, obs.cfg, obs.state, obs.champ, obs.side)
        if a not in obs.legal:
            raise ValueError(f"{self.name}: illegal scripted action {a}")
        return a


class NetAgent:
    """A frozen network acting on the prefix of the observation it was trained on (sampling with the seat's
    generator, or greedily without one)."""

    def __init__(self, net, name: str):
        self.net, self.name = net, name

    def distribution(self, obs) -> np.ndarray:
        import torch
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(obs.vector[:self.net.obs_size], dtype=np.float32)).unsqueeze(0)
            logits, _ = self.net(x)
        lg = logits[0].double().numpy()[list(obs.legal)]
        p = np.exp(lg - lg.max())
        return p / p.sum()

    def act(self, obs, rng=None) -> int:
        return int(obs.legal[_sample(rng, self.distribution(obs))])


class MixtureAgent:
    def __init__(self, a, b, p: float, name: str):
        self.a, self.b, self.p, self.name = a, b, float(p), name

    def act(self, obs, rng=None) -> int:
        u = rng.random() if rng is not None else 0.0
        return (self.a if u < self.p else self.b).act(obs, rng)


class SwitchAgent:
    def __init__(self, a, b, turn: int, name: str):
        self.a, self.b, self.turn, self.name = a, b, int(turn), name

    def act(self, obs, rng=None) -> int:
        return (self.a if obs.k < self.turn else self.b).act(obs, rng)


# ---------------------------------------------------------------------------------------------- belief types
class ScriptedType:
    kind = "script"

    def __init__(self, name: str, eps: float = 0.05, strategy=None):
        if strategy is None:
            from ..experiments.m45b.strategies import get
            strategy = get(name)
        self.name, self.eps, self.strategy = name, float(eps), strategy

    def probs(self, cfg, lane, m4, s: int, k: int) -> Tuple[Tuple[int, ...], np.ndarray]:
        legal = tuple(int(c) for c in champion_legal_actions(lane, m4, s, cfg))
        p = np.full(len(legal), self.eps / len(legal))
        p[legal.index(_decide(self.strategy, cfg, lane, m4, s))] += 1.0 - self.eps
        return legal, p

    def agent(self) -> ScriptedAgent:
        return ScriptedAgent(self.strategy, self.name)


class NetType:
    kind = "net"

    def __init__(self, name: str, net):
        self.name, self.net = name, net

    def _obs(self, cfg, lane, m4, s: int, k: int):
        from ..diagnostics.branch import Snapshot, observation
        base = cfg.with_(obs_last_action=False) if cfg.obs_last_action else cfg
        return observation(base, Snapshot(lane, m4, k, (cfg.skill_a, cfg.skill_b)), s)

    def probs_batch(self, cfg, items: Sequence[tuple]) -> List[Tuple[Tuple[int, ...], np.ndarray]]:
        """[(legal, p)] for items (lane, m4, seat, k), one forward pass."""
        import torch
        obs = [self._obs(cfg, *it) for it in items]
        with torch.no_grad():
            x = torch.as_tensor(np.stack([o.vector[:self.net.obs_size] for o in obs]).astype(np.float32))
            logits, _ = self.net(x)
        lg = logits.double().numpy()
        out = []
        for o, row in zip(obs, lg):
            legal = tuple(int(c) for c in o.legal)
            z = row[list(legal)]
            p = np.exp(z - z.max())
            out.append((legal, p / p.sum()))
        return out

    def probs(self, cfg, lane, m4, s: int, k: int):
        return self.probs_batch(cfg, [(lane, m4, s, k)])[0]

    def probs_vectors(self, vectors: np.ndarray, legals: Sequence[Sequence[int]]):
        """[(legal, p)] from precomputed observation vectors (their first net.obs_size entries must be the state
        encoding of the opponent's seat, e.g. the first 90 entries of an obs_last_action observation) and legal sets."""
        import torch
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(vectors, dtype=np.float32)[:, :self.net.obs_size])
            logits, _ = self.net(x)
        lg = logits.double().numpy()
        out = []
        for legal, row in zip(legals, lg):
            legal = tuple(int(c) for c in legal)
            z = row[list(legal)]
            p = np.exp(z - z.max())
            out.append((legal, p / p.sum()))
        return out

    def agent(self) -> NetAgent:
        return NetAgent(self.net, self.name)


class LibraryMixtureType:
    """The catch-all "mixed": each turn the opponent acts like a different library type, drawn uniformly from
    `components` (indices into the belief's type list). Its likelihood is the mean of those types' probabilities
    (computed by belief.loglik). It explains a per-turn mixture of styles better than any single style does, while
    a consistent style beats it."""
    kind = "library_mixture"

    def __init__(self, components, name: str = "mixed"):
        self.components, self.name = tuple(int(c) for c in components), name


class UniformType:
    kind = "uniform"

    def __init__(self, name: str = "unknown"):
        self.name = name

    def probs(self, cfg, lane, m4, s: int, k: int):
        legal = tuple(int(c) for c in champion_legal_actions(lane, m4, s, cfg))
        return legal, np.full(len(legal), 1.0 / len(legal))


__all__ = ["ScriptedType", "NetType", "UniformType", "LibraryMixtureType", "ScriptedAgent", "NetAgent", "MixtureAgent", "SwitchAgent"]
