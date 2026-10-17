"""Paired rollouts that keep every turn's reward (the M4.6a battery design with full reward streams).

One job = one start state x a list of seat-policy combinations x K paired rollouts of T turns. Rollout r of every
combination uses the chance stream default_rng(seed_r) and the per-turn sampling uniforms of `m45c.horizon.CRNNet(net,
k0, seed_r, seat)` for both seats (the same draw pattern as `m46a.engine.rollout`), so combinations differ only by their
intervention. Stored per combination: seat A's reward per turn r [K, T] (float32, resources), both seats' levels at
the start of each turn L [K, T, 2] (int8), how many turns each seat's plan forced, and both seats' turn-0 action codes.
Any discount, window or level-triggered cut is computed afterwards from the same streams; nothing is bootstrapped with
a critic.

Seat-policy specifications (tuples, picklable):
    ("policy",)                          the job's network itself (CRN sampling)
    ("net", spec)                        another network (CRN sampling, the same uniforms)
    ("ref", name)                        a fixed M4.5c reference strategy (m45c.references.SUITE)
    ("m46a", cspec)                      an M4.6a plan around the job's network (m46a.policies.build: beh, fixed,
                                         reset, script, hz)
    ("m46d", name)                       an M4.6d freeze / breaker plan (m46d.plans.FREEZER / BREAKER)
State variants applied to the start before the rollouts (seat A's side; VARIANTS): "inv+4" (+4 invested resources),
"hp-0.3" (0.3 bars of HP lost), "shop" (unspent gold invested now).
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Sequence

import numpy as np

from ....champions import N_CHANCE
from . import common as C


# ---------------------------------------------------------------------------------------------- variants
def variant_inv(cfg, start, res: float = 4.0):
    lane, m4, k = start
    a = m4.a
    return lane, m4._replace(a=a._replace(items=a.items + int(round(res * cfg.gold_scale)))), k


def variant_hp(cfg, start):
    from ..m46a.policies import variant_hp as _hp
    return _hp(cfg, start, 0.3)


def variant_shop(cfg, start):
    from ..m46a.policies import variant_invest
    return variant_invest(cfg, start)


VARIANTS = {"orig": lambda cfg, st: st, "inv+4": variant_inv, "hp-0.3": variant_hp, "shop": variant_shop}


# ---------------------------------------------------------------------------------------------- seat policies
def seat_policy(spec: tuple, net, k0: int, seed: int, seat: int, T: int):
    from ..m45c import horizon as HZ
    kind = spec[0]
    own = HZ.CRNNet(net, k0, seed, seat, T)
    if kind == "policy":
        return own
    if kind == "net":
        return HZ.CRNNet(C.net_of(spec[1]), k0, seed, seat, T)
    if kind == "ref":
        from ...m45b.strategies import get
        return get(spec[1])
    if kind == "m46a":
        from ..m46a import policies as P
        return P.build(tuple(spec[1]), own)
    if kind == "m46d":
        from ..m46d import plans as PD
        table = PD.FREEZER if spec[1] in PD.FREEZER else PD.BREAKER
        return PD.build(table[spec[1]], own)
    raise ValueError(spec)


# ---------------------------------------------------------------------------------------------- one game
def stream_game(cfg, start, pa, pb, seed: int, T: int):
    """(r [T] seat A's reward per turn, L [T, 2] levels at the start of each turn, (code A, code B) of turn 0) of one
    game."""
    from ....champions_m4 import context, resolve_turn_m4
    from ...m45b import engine as E
    lane, m4, k0 = start
    ctx = context(cfg)
    tb = ctx.t
    ls = E.lane_cache(cfg)
    rng = np.random.default_rng(int(seed))
    r = np.zeros(T, dtype=np.float32)
    L = np.zeros((T, 2), dtype=np.int8)
    first = (-1, -1)
    for i in range(T):
        u = rng.random(N_CHANCE)
        L[i, 0] = tb.level_by_xp[min(m4.a.xp, tb.xp_cap)]
        L[i, 1] = tb.level_by_xp[min(m4.b.xp, tb.xp_cap)]
        a = E.decide(pa, cfg, lane, m4, 0, i)
        b = E.decide(pb, cfg, lane, m4, 1, i)
        if i == 0:
            first = (a, b)
        t = resolve_turn_m4(lane, m4, k0 + i, a, b, cfg, lane_step=ls, u=u, ctx=ctx)
        r[i] = t.reward
        lane, m4 = t.lane.state, t.champ
    return r, L, first


def self_play_state(cfg, net, start, seed: int, turns: int, net_b=None):
    """The state reached after `turns` turns of CRN self-play (seat B `net_b` or the same network) from `start`."""
    from ....champions_m4 import context, resolve_turn_m4
    from ...m45b import engine as E
    from ..m45c import horizon as HZ
    lane, m4, k0 = start
    ctx = context(cfg)
    ls = E.lane_cache(cfg)
    pa = HZ.CRNNet(net, k0, seed, 0, turns)
    pb = HZ.CRNNet(net_b if net_b is not None else net, k0, seed, 1, turns)
    rng = np.random.default_rng(int(seed))
    for i in range(turns):
        u = rng.random(N_CHANCE)
        a = E.decide(pa, cfg, lane, m4, 0, i)
        b = E.decide(pb, cfg, lane, m4, 1, i)
        t = resolve_turn_m4(lane, m4, k0 + i, a, b, cfg, lane_step=ls, u=u, ctx=ctx)
        lane, m4 = t.lane.state, t.champ
    return lane, m4, k0 + turns


# ---------------------------------------------------------------------------------------------- the paired job
def stream_job(job: dict) -> tuple:
    """`job`: key, path, env, spec, start (list form) or natural (seed, turns), combos {name: [variant, a_spec,
    b_spec]}, K, T, root, label. Writes an npz to `path` (resumable) and returns (key, info)."""
    path = job["path"]
    if os.path.exists(path):
        return job["key"], {"path": path, "cached": True}
    cfg = C.setup(job["env"])
    net = C.net_of(job["spec"])
    if job.get("natural"):
        from ...m45b import states as S
        nsd, nturns = job["natural"]
        start = self_play_state(cfg, net, S.s0(cfg), nsd, nturns)
    else:
        start = C.state_from_list(job["start"])
    T, K = int(job["T"]), int(job["K"])
    seeds = [int(job["root"]) * 1000 + r for r in range(K)]
    names = list(job["combos"])
    R = np.zeros((len(names), K, T), dtype=np.float32)
    Lv = np.zeros((len(names), K, T, 2), dtype=np.int8)
    forced = np.zeros((len(names), K, 2), dtype=np.int32)
    first = np.zeros((len(names), K, 2), dtype=np.int16)
    t0 = time.perf_counter()
    for ci, nm in enumerate(names):
        var, a_spec, b_spec = job["combos"][nm]
        st = VARIANTS[var](cfg, start)
        k0 = st[2]
        for r, sd in enumerate(seeds):
            pa = seat_policy(tuple(a_spec), net, k0, sd, 0, T)
            pb = seat_policy(tuple(b_spec), net, k0, sd, 1, T)
            R[ci, r], Lv[ci, r], first[ci, r] = stream_game(cfg, st, pa, pb, sd, T)
            forced[ci, r] = (getattr(pa, "forced", 0), getattr(pb, "forced", 0))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".{os.getpid()}.tmp.npz"
    lane, m4, k = start
    np.savez_compressed(tmp, r=R, L=Lv, forced=forced, first=first, combos=np.asarray(names),
                        label=np.asarray(job.get("label", "")),
                        start_turn=np.asarray(int(k)), start_levels=np.asarray([int(x) for x in _levels(cfg, m4)]),
                        elapsed=np.asarray(time.perf_counter() - t0))
    os.replace(tmp, path)
    return job["key"], {"path": path, "elapsed_s": time.perf_counter() - t0}


def _levels(cfg, m4):
    from ....champions_m4 import context
    tb = context(cfg).t
    return tb.level_by_xp[min(m4.a.xp, tb.xp_cap)], tb.level_by_xp[min(m4.b.xp, tb.xp_cap)]


def load(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["combos"] = [str(x) for x in d["combos"]]
    d["label"] = str(d["label"])
    return d


def run_sorted(fn, js: List[dict], workers: int, label: str):
    js = sorted(js, key=lambda j: -int(j["K"]) * int(j["T"]) * len(j["combos"]))
    t0 = time.perf_counter()
    res = C.run_jobs(fn, js, workers, label)
    print(f"  [{label}] {len(js)} jobs in {time.perf_counter() - t0:.0f} s", flush=True)
    return dict(res)


# ---------------------------------------------------------------------------------------------- scoring
def discounted(r: np.ndarray, gamma: float) -> np.ndarray:
    """r [..., T] -> sum_t gamma^t r_t over the last axis."""
    return (r * C.disc(gamma, r.shape[-1])).sum(-1)


def window_sum(r: np.ndarray, lo: int, hi: int) -> np.ndarray:
    return r[..., lo:hi].sum(-1)


def split_at(r: np.ndarray, cut, gamma: float):
    """(pre, post) discounted sums of r [K, T] split at per-rollout turn indices `cut` [K] (or a scalar)."""
    T = r.shape[-1]
    w = C.disc(gamma, T)
    x = r * w
    cut = np.broadcast_to(np.asarray(cut), x.shape[:-1])
    m = np.arange(T)[None, :] < cut[..., None]
    return (x * m).sum(-1), (x * ~m).sum(-1)


def level_cut(L: np.ndarray, lv: int = 14) -> np.ndarray:
    """L [K, T, 2] -> [K] the first turn index at which the LOWER of the two levels reaches lv (T if never)."""
    m = L.min(-1) >= lv
    T = L.shape[-2]
    return np.where(m.any(-1), m.argmax(-1), T)


__all__ = ["VARIANTS", "seat_policy", "stream_game", "self_play_state", "stream_job", "load", "run_sorted",
           "discounted", "window_sum", "split_at", "level_cut"]
