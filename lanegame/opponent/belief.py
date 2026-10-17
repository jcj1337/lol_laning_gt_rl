"""The Bayesian belief about the opponent (OM2, docs/opponent_modelling/om2_design.md §3).

A log posterior over K belief types (lanegame.opponent.types; the last is usually the catch-all "unknown"), uniform
prior. After each turn the opponent's observed action updates it:

    log post <- normalize(lam * log post + log P_type(action | the opponent's start-of-turn state))

lam = 1 is exact Bayes; lam < 1 (forgetting) decays the accumulated evidence by lam per informative turn, an effective
memory of about 1 / (1 - lam) turns, so the belief can follow an opponent that changes style. A turn on which every
type assigns the observed action the same probability (in particular an Absent turn, the only legal action of an away
opponent) carries no information and changes nothing, not even by forgetting. `features()` is the posterior, the input
the opponent-modelling policy receives.

`loglik(types, cfg, items)` evaluates the K types on many (lane, m4, opponent seat, k, observed code) items at once:
one batched forward pass per learned type, one scripted decision per (scripted type, item). A "library_mixture" type
(types.LibraryMixtureType) is derived: the mean of its component types' probabilities of the observed action.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


class Belief:
    def __init__(self, n_types: int, lam: float = 0.97):
        if not 0.0 < lam <= 1.0:
            raise ValueError("lam must be in (0, 1]")
        self.n, self.lam = int(n_types), float(lam)
        self.reset()

    def reset(self) -> None:
        self.lp = np.full(self.n, -np.log(self.n))
        self.updates = 0

    def update(self, loglik: np.ndarray) -> None:
        loglik = np.asarray(loglik, dtype=np.float64)
        if np.ptp(loglik) == 0.0:                       # uninformative: every type gave the same probability
            return
        lp = self.lam * self.lp + loglik
        m = lp.max()
        self.lp = lp - (m + np.log(np.exp(lp - m).sum()))
        self.updates += 1

    def posterior(self) -> np.ndarray:
        return np.exp(self.lp)

    def features(self) -> np.ndarray:
        return self.posterior().astype(np.float32)


def prior_features(n_types: int) -> np.ndarray:
    """The features of an un-updated belief (the control arm's constant input)."""
    return np.full(n_types, 1.0 / n_types, dtype=np.float32)


def loglik(types: Sequence, cfg, items: Sequence[tuple]) -> np.ndarray:
    """[len(items), K] log P_type(code | state) for items (lane, m4, seat, k, code) or (lane, m4, seat, k, code,
    vector, legal), the last two the opponent seat's precomputed observation vector (state encoding first) and legal
    set, which learned types then use instead of rebuilding the observation."""
    P = np.zeros((len(items), len(types)))
    if not items:
        return P
    pre = all(len(it) >= 7 for it in items)
    vecs = np.stack([np.asarray(it[5], dtype=np.float32) for it in items]) if pre else None
    derived = []
    for j, t in enumerate(types):
        if getattr(t, "kind", "") == "library_mixture":
            derived.append(j)
            continue
        if pre and hasattr(t, "probs_vectors"):
            res = t.probs_vectors(vecs, [it[6] for it in items])
        elif hasattr(t, "probs_batch"):
            res = t.probs_batch(cfg, [it[:4] for it in items])
        else:
            res = [t.probs(cfg, *it[:4]) for it in items]
        for n, ((legal, p), it) in enumerate(zip(res, items)):
            P[n, j] = float(p[legal.index(int(it[4]))])
    for j in derived:                                  # the per-turn mixture of its component types
        P[:, j] = P[:, list(types[j].components)].mean(1)
    return np.log(np.maximum(P, 1e-300))


__all__ = ["Belief", "prior_features", "loglik"]
