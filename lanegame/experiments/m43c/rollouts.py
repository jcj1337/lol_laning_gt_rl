"""Paired counterfactual rollouts that record the objective channels (common.channels).

`channel_rollout` is `lanegame.diagnostics.branch.rollout` with the same stream layout (environment chance, focal
sampling, opponent sampling from one SeedSequence; exactly one uniform per seat per turn; the same inverse-CDF
sampling and the same `Plan` semantics, including the distribution-level `transform` hook), so that its gold channel
reproduces `branch.rollout(...).ret` exactly. It additionally returns the discounted XP and level-progress channels and
per-side totals.

Extended absence (death-timer diagnostic, design document §D2). `extra_absence = {seat: n}` keeps that seat's
champion away for n more turns after its engine timer runs out: on the turn the engine would return it (the timer
reaches 0, it shops and refills its HP), the diagnostic writes absent = min(n_left, 3) back into the champion record
and re-observes, so the champion sits out n more turns. The observation's absence entry stays inside its trained range
(absent / champion_absent_max <= 1). This is a measurement device for "what would an absence of d turns cost", applied
outside the engine; it is NOT a death-timer rule of the game (the engine and every preset are unchanged).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np

from ...diagnostics.branch import DEFER, Plan, ProbCache, Snapshot, restore, sample_index
from ...env import LaneEnv
from ...env.agents import episode_streams
from .common import channels


@dataclass
class ChannelRollout:
    G: float                       # discounted gold channel (= the current objective's return)
    X: float                       # discounted credited-XP channel
    Lam: float                     # discounted level-progress channel
    turns: int
    # the same channels truncated at the Part C termination (the turn either tower's cumulative damage reaches the last
    # plate threshold, that turn included); equal to the untruncated sums when no termination occurs in the window
    G_T: float = 0.0
    X_T: float = 0.0
    Lam_T: float = 0.0
    term_turn: Optional[int] = None
    gold: tuple = (0.0, 0.0)       # undiscounted earned resources (focal, opponent)
    xp: tuple = (0, 0)             # undiscounted credited XP
    xp_discarded: tuple = (0, 0)
    deaths: tuple = (0, 0)
    recalls: tuple = (0, 0)
    plates: tuple = (0, 0)
    absent_turns: tuple = (0, 0)
    extra: Dict[str, float] = field(default_factory=dict)


def channel_rollout(env: LaneEnv, snap: Snapshot, focal: int, plan: Optional[Plan], agents: Sequence, seed,
                    cache: Optional[ProbCache] = None, window: int = 300, gamma: Optional[float] = None,
                    extra_absence: Optional[Dict[int, int]] = None, on_turn=None) -> ChannelRollout:
    """One rollout of `window` turns from `snap` (see the module docstring). `plan=None` follows the agents."""
    from ...progression import tables
    cfg = env.cfg
    g = cfg.gamma if gamma is None else float(gamma)
    t_tab = tables(cfg)
    scale = cfg.gold_scale
    cache = cache if cache is not None else ProbCache()
    ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    _, env_ss, rng_a, rng_b = episode_streams(ss)
    rngs = (rng_a, rng_b)
    obs = restore(env, snap, env_ss)
    o = 1 - focal
    live = plan is not None
    G = X = Lam = 0.0
    G_T = X_T = Lam_T = 0.0
    theta = cfg.plate_thresholds[-1]
    term_turn = 0 if max(snap.champ.a.tower_dmg, snap.champ.b.tower_dmg) >= theta else None
    w = 1.0
    gold, xp, disc, deaths, recalls, plates, absent = [0.0, 0.0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]
    extra_left = dict(extra_absence or {})
    for t in range(window):
        acts = [0, 0]
        for s in (0, 1):
            ob = obs[s]
            p = cache.distribution(agents[s], ob)
            pa = None
            if s == focal and live:
                pa = plan.act(ob, t)
                if pa is None:
                    live = False
                elif pa == DEFER:
                    q = np.asarray(plan.transform(ob, t, p), dtype=float)
                    if q.shape != p.shape or not np.all(np.isfinite(q)) or q.sum() <= 0 or (q < 0).any():
                        raise ValueError(f"plan {plan.name!r} returned an invalid distribution")
                    p = q
                elif pa not in ob.legal:
                    raise ValueError(f"plan {plan.name!r} chose illegal action {pa} (legal {ob.legal})")
            u = rngs[s].random()                              # exactly one draw per seat per turn
            a = ob.legal[sample_index(p, u)]
            if pa is not None and pa != DEFER:
                a = int(pa)
            acts[s] = a
        r = env.step(acts[0], acts[1])
        info = r.info
        cg, cx, cl = channels(info, focal, t_tab, scale)
        G += w * cg
        X += w * cx
        Lam += w * cl
        if term_turn is None:
            G_T += w * cg
            X_T += w * cx
            Lam_T += w * cl
            if max(info["champ_after"][0][6], info["champ_after"][1][6]) >= theta:
                term_turn = t + 1
        w *= g
        for i, s in enumerate((focal, o)):
            gold[i] += info["earned_units"][s] / scale
            xp[i] += info["xp_gain"][s]
            disc[i] += info["xp_discarded"][s]
            deaths[i] += int(info["died"][s])
            recalls[i] += int(info["recalled"][s])
            plates[i] += info["plates_broken"][s]
            absent[i] += int(info["away"][s])
        if on_turn is not None:
            on_turn(t, info)
        obs = r.obs
        # the extended-absence device (module docstring): re-impose absence on the turn the engine returns the seat
        changed = False
        for s, left in list(extra_left.items()):
            if left > 0 and info["returned"][s]:
                n = min(left, cfg.champion_absent_max)
                env._champ = env._champ.replace_side(s, absent=n)
                extra_left[s] = left - n
                changed = True
        if changed:
            obs = env._observe()
    return ChannelRollout(G, X, Lam, window, G_T, X_T, Lam_T, term_turn, tuple(gold), tuple(xp), tuple(disc),
                          tuple(deaths), tuple(recalls), tuple(plates), tuple(absent))


def paired(cfg, snap: Snapshot, focal: int, candidates: Dict[str, Optional[Plan]], agents: Sequence, n: int, root: int,
           key: Sequence[int] = (), cache: Optional[ProbCache] = None, window: int = 300,
           variants: Optional[Dict[str, tuple]] = None) -> Dict[str, dict]:
    """Paired rollouts of every candidate from `snap` on the roots of `branch.rollout_seed(root, m, *key)`.
    `variants` optionally maps a candidate name to (snapshot, extra_absence) replacing the start state (the death-timer
    diagnostic starts its variants from modified copies of the same state, still on the same streams).
    Returns {candidate: {"G", "X", "Lam": arrays over rollouts, plus per-side totals}}."""
    from ...diagnostics.branch import rollout_seed
    env = LaneEnv(snap.config(cfg), cache_size=20_000)
    cache = cache if cache is not None else ProbCache()
    out = {}
    for name, plan in candidates.items():
        s0, extra = (snap, None)
        if variants and name in variants:
            s0, extra = variants[name]
        rs = [channel_rollout(env, s0, focal, plan, agents, rollout_seed(root, m, *key), cache, window,
                              extra_absence=extra) for m in range(n)]
        out[name] = {"G": np.array([r.G for r in rs]), "X": np.array([r.X for r in rs]),
                     "Lam": np.array([r.Lam for r in rs]),
                     "G_T": np.array([r.G_T for r in rs]), "X_T": np.array([r.X_T for r in rs]),
                     "Lam_T": np.array([r.Lam_T for r in rs]),
                     "term_turn": np.array([np.nan if r.term_turn is None else r.term_turn for r in rs], dtype=float),
                     "gold": np.array([r.gold for r in rs]), "xp": np.array([r.xp for r in rs], dtype=float),
                     "xp_discarded": np.array([r.xp_discarded for r in rs], dtype=float),
                     "deaths": np.array([r.deaths for r in rs], dtype=float),
                     "recalls": np.array([r.recalls for r in rs], dtype=float),
                     "plates": np.array([r.plates for r in rs], dtype=float),
                     "absent": np.array([r.absent_turns for r in rs], dtype=float)}
    return out


def diff_stats(res: Dict[str, dict], c: str, base: str, lam: float = 0.0, mu: float = 0.0, terminal: bool = False):
    """Paired mean and standard error of J(c) - J(base) for J = G + lam X + mu Lam (terminal: the sums truncated at the
    Part C termination)."""
    a, b = res[c], res[base]
    sfx = "_T" if terminal else ""
    d = ((a["G" + sfx] - b["G" + sfx]) + lam * (a["X" + sfx] - b["X" + sfx])
         + mu * (a["Lam" + sfx] - b["Lam" + sfx]))
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")


__all__ = ["ChannelRollout", "channel_rollout", "paired", "diff_stats"]
