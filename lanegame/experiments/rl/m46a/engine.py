"""The M4.6a paired rollout job (the M4.5c battery's design, generalized to behaviour plans, state variants and
substituted networks) and the extended ledger.

One job = one start state x one opponent protocol x every candidate x K paired rollouts. Rollout r of every candidate
uses the chance stream default_rng(seed_r) and the per-turn sampling uniforms of `horizon.CRNNet(net, k0, seed_r,
seat)` for both seats, seed_r = roll_root * 100 + r (E4: roll_root * 1000 + 100 * idx + r): candidates differ only by
their intervention. Candidates (dict name -> spec):
    ("policy",) / ("beh", ...) / ("script", ...) / ("fixed", ...)   policies.build around the focal CRN policy
    ("variant", variant, inner_spec)                                 the start state transformed (policies.VARIANTS),
                                                                     then inner_spec
    ("net", spec_or_path)                                            a different network in seat A (E4), same CRN
                                                                     uniforms
Opponent protocols for seat B: "policy" (the evaluated network, sampling), "modal", a scripted M4.5b strategy name,
or "net:<spec_or_path>".

Recorded per candidate: G_h for every rollout at the checkpoints, Q_h = G_h + 0.99^h V(s_h) with the evaluated policy's
critic (when `critic`), and for the ledger checkpoints the candidate mean and the paired difference to the baseline
(the first candidate) of every LEDGER key (mean and SD over the K paired rollouts), plus the number of turns a plan
actually forced. Results are written as gzipped JSON to `path` and read back if present (resumable).

Extended ledger (`XLedger`): the m45b ledger plus unc_away (farm lost while away; unc - unc_away = farm lost while
present: reach, posture, trading).
"""

from __future__ import annotations

import os
import time
from typing import Dict, Sequence

import numpy as np

from ....champions import N_CHANCE
from .common import GAMMA, load_gz, net_of, save_gz, setup_worker

LEDGER_KEYS = ("earned", "farm", "farm_avail", "unc", "unc_away", "bounty", "plate", "xp", "xp_lost", "d_total",
               "d_trade", "d_min", "d_gank", "d_tower", "deaths", "recalls", "rec_intr", "away", "trade_att",
               "trade_conn", "trade_dealt", "held", "frozen", "crash_hp", "plates", "n_safe", "n_yield", "n_trade")
BASE_CP = (1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 60, 75, 100, 150, 200, 300)
BASE_LCP = (5, 10, 20, 50, 60, 100, 150, 300)


def checkpoints(horizon: int):
    cps = tuple(c for c in BASE_CP if c <= horizon)
    return cps if horizon in cps else cps + (horizon,)


def ledger_checkpoints(horizon: int):
    return tuple(c for c in BASE_LCP if c <= horizon)


def XLedger(cfg, start):
    from ...m45b import engine as E

    class _X(E.Ledger):
        def __init__(self, cfg, start):
            super().__init__(cfg, start)
            self.v["unc_away"] = [0.0, 0.0]

        def add(self, t, lane_before):
            super().add(t, lane_before)
            sc = self.cfg.gold_scale
            for s in (0, 1):
                if t.away[s]:
                    self.v["unc_away"][s] += t.farm_available[s] - t.farm_units[s] / sc
    return _X(cfg, start)


def rollout(cfg, start, pa, pb, seed: int, horizon: int, cps, lcps, critic=None) -> Dict[int, dict]:
    """One game from `start`; {checkpoint: {"G", ["Q", "V"], ledger keys _A/_B at ledger checkpoints}}."""
    from ....champions_m4 import context, resolve_turn_m4
    from ...m45b import engine as E
    from ..m45c.horizon import critic_value
    lane, m4, k0 = start
    ctx = context(cfg)
    ls = E.lane_cache(cfg)
    rng = np.random.default_rng(int(seed))
    led = XLedger(cfg, start)
    cps, lcps = set(cps), set(lcps)
    out: Dict[int, dict] = {}
    G = 0.0
    for i in range(horizon):
        u = rng.random(N_CHANCE)
        a = E.decide(pa, cfg, lane, m4, 0, i)
        b = E.decide(pb, cfg, lane, m4, 1, i)
        t = resolve_turn_m4(lane, m4, k0 + i, a, b, cfg, lane_step=ls, u=u, ctx=ctx)
        led.add(t, lane)
        G += GAMMA ** i * t.reward
        lane, m4 = t.lane.state, t.champ
        n = i + 1
        if n in cps:
            row = {"G": G}
            if critic is not None:
                V = critic_value(critic, cfg, lane, m4, k0 + n, 0)
                row["V"], row["Q"] = V, G + GAMMA ** n * V
            if n in lcps:
                for kk in LEDGER_KEYS:
                    row[f"{kk}_A"], row[f"{kk}_B"] = led.v[kk][0], led.v[kk][1]
            out[n] = row
    return out


def _opponent(proto: str, net, k0: int, seed: int, horizon: int):
    from ..m45c import horizon as HZ
    from ...m45b.strategies import get
    if proto == "policy":
        return HZ.CRNNet(net, k0, seed, 1, horizon)
    if proto == "modal":
        return HZ.CRNNet(net, k0, seed, 1, horizon, greedy=True)
    if proto.startswith("net:"):
        return HZ.CRNNet(net_of(proto[4:]), k0, seed, 1, horizon)
    return get(proto)


def _seat_a(cspec, net, cfg, start, k0, seed, horizon):
    """(policy for seat A, start state) of a candidate."""
    from ..m45c import horizon as HZ
    from . import policies as P
    if cspec[0] == "variant":
        st = P.VARIANTS[cspec[1]](cfg, start)
        pol, _ = _seat_a(cspec[2], net, cfg, st, k0, seed, horizon)
        return pol, st
    if cspec[0] == "net":
        return HZ.CRNNet(net_of(cspec[1]), k0, seed, 0, horizon), start
    own = HZ.CRNNet(net, k0, seed, 0, horizon)
    return P.build(tuple(cspec), own), start


def paired_job(job: dict) -> tuple:
    """`job`: spec, family, idx, start, proto, horizon, K, roll (list of K seeds), cands (dict), path, label,
    critic (bool). Returns (key, record)."""
    path = job["path"]
    key = f"{job['family']}|{job['idx']}|{job['proto']}|{job['horizon']}|{job.get('tag', '')}"
    if path and os.path.exists(path):
        return key, load_gz(path)
    cfg = setup_worker()
    from ...m45b import engine as E
    net = net_of(job["spec"])
    start = job["start"]
    k0 = start[2]
    H = int(job["horizon"])
    cps, lcps = checkpoints(H), ledger_checkpoints(H)
    keys = [f"{k}_{x}" for k in LEDGER_KEYS for x in ("A", "B")]
    cands = job["cands"]
    critic = net if job.get("critic", True) else None
    t0 = time.perf_counter()
    G, Q, L, forced, first = {}, {}, {}, {}, {}
    for name, cspec in cands.items():
        g = {c: [] for c in cps}
        q = {c: [] for c in cps}
        led = {c: np.full((len(job["roll"]), len(keys)), np.nan) for c in lcps}
        f_count, codes = [], []
        for r, sd in enumerate(job["roll"]):
            pa, st = _seat_a(cspec, net, cfg, start, k0, sd, H)
            pb = _opponent(job["proto"], net, k0, sd, H)
            codes.append(int(E.decide(pa, cfg, st[0], st[1], 0, 0)))
            if hasattr(pa, "forced"):
                pa.forced = 0
            out = rollout(cfg, st, pa, pb, sd, H, cps, lcps, critic)
            f_count.append(int(getattr(pa, "forced", 0)))
            for c, row in out.items():
                g[c].append(row["G"])
                if "Q" in row:
                    q[c].append(row["Q"])
                if c in led:
                    led[c][r] = [row.get(kk, np.nan) for kk in keys]
        G[name] = {str(c): v for c, v in g.items()}
        Q[name] = {str(c): v for c, v in q.items()} if critic is not None else None
        L[name] = led
        forced[name] = f_count
        first[name] = codes
    base = list(cands)[0]
    ledger = {}
    for name in cands:
        ledger[name] = {}
        for c in lcps:
            d = L[name][c] - L[base][c]
            ledger[name][str(c)] = {"mean": np.nanmean(L[name][c], 0).tolist(), "diff": np.nanmean(d, 0).tolist(),
                                    "diff_sd": (np.nanstd(d, 0, ddof=1) if len(job["roll"]) > 1
                                                else np.zeros(len(keys))).tolist()}
    rec = {"family": job["family"], "idx": job["idx"], "proto": job["proto"], "horizon": H, "spec": job["spec"],
           "label": job.get("label", ""), "tag": job.get("tag", ""), "K": len(job["roll"]), "k0": int(k0),
           "keys": keys, "cands": {n: list(map(str, s)) for n, s in cands.items()}, "baseline": base, "G": G, "Q": Q,
           "ledger": ledger, "forced": forced, "first_codes": first, "elapsed_s": time.perf_counter() - t0,
           "turns": len(job["roll"]) * len(cands) * H}
    if path:
        save_gz(path, rec)
    return key, rec


# ---------------------------------------------------------------------------------------------- analysis helpers
def per_state_diff(rec: dict, cand: str, stat: str, h: int, base: str = None) -> float:
    """The paired mean (candidate - baseline) of statistic G or Q at horizon h for one job record."""
    base = base or rec["baseline"]
    src = rec[stat]
    if src is None:
        return float("nan")
    a, b = src[cand].get(str(h)), src[base].get(str(h))
    if not a or not b:
        return float("nan")
    return float(np.mean(np.asarray(a) - np.asarray(b)))


def per_state_mean(rec: dict, cand: str, stat: str, h: int) -> float:
    src = rec[stat]
    if src is None or not src[cand].get(str(h)):
        return float("nan")
    return float(np.mean(src[cand][str(h)]))


def ledger_diff(rec: dict, cand: str, key: str, h: int) -> float:
    j = rec["keys"].index(key)
    e = rec["ledger"][cand].get(str(h))
    return float(e["diff"][j]) if e else float("nan")


def ledger_mean(rec: dict, cand: str, key: str, h: int) -> float:
    j = rec["keys"].index(key)
    e = rec["ledger"][cand].get(str(h))
    return float(e["mean"][j]) if e else float("nan")


def roll_seeds(root: int, K: int, idx: int = None):
    if idx is None:
        return [root * 100 + r for r in range(K)]
    return [root * 1000 + 100 * idx + r for r in range(K)]


__all__ = ["LEDGER_KEYS", "rollout", "paired_job", "checkpoints", "ledger_checkpoints", "per_state_diff",
           "per_state_mean", "ledger_diff", "ledger_mean", "roll_seeds", "XLedger"]
