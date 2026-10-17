"""Full-state natural self-play records (descriptive stage; workstreams A, B and the critic).

A game g of policy seed s (environment `arm`) starts at s0 and uses the chance stream default_rng(sd) and each seat's
CRN sampling uniforms default_rng([sd, 7, seat]) (`m45c.horizon.CRNNet`), sd = REC_ROOT * 1000 + 100 s + g: the same
streams for every policy and environment, so games pair across arms. Both seats are the same network (self-play,
sampling), exactly the pairing its critic is trained to predict.

Per turn and seat (arrays [T, 2] unless noted; cells in each side's OWN coordinates, the Ward node W = 99):
    lane      fx, fh (front cell / HP), nrear, rear_near (rear packets within 2 cells behind the front), mass (crashed
              mass under the enemy tower), vis
    champion  pos, hp (fraction; 1 while away), L, S (invested), g (unspent), absent, C (wave capacity), xp
    action    code, wave (0 LH, 1 SH, 2 TN, 3 SL; -1 none), beh (0 Hold, 1 Yield, 2 Trade, 3 Safe, 4 Recall; -1),
              obj (0 none, 1 Pressure, 2 Vision/Ward)
    policy    p_wave [T, 2, 4], p_beh [T, 2, 5], p_ward, p_press (marginal probabilities of the turn's distribution),
              V (critic, resources, own perspective)
    outcome   earned, farm, avail, bounty, plate, xpg, died, recalled, intr, conn, dealt (bars of the victim),
              taken (bars), dmin (bars), away, crash, held (holds_freeze after the turn), held0 (before)
    reward    r [T] (to seat A)
Returned as numpy arrays in one dict per game; a policy's games are saved together (npz, compressed).
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np

from ....champions import N_CHANCE
from ....config import (B_RECALL, CHAMP_ABSENT, CHAMP_WARD, PRESSURE, VISION, champ_behavior, champ_objective,
                        champ_wave)
from . import common as C

GAMES, TURNS = 48, 400
PAIR_KEYS = ("fx", "fh", "nrear", "rear_near", "mass", "vis", "pos", "hp", "L", "S", "g", "absent", "C", "xp", "code",
             "wave", "beh", "obj", "p_ward", "p_press", "V", "earned", "farm", "avail", "bounty", "plate", "xpg",
             "died", "recalled", "intr", "conn", "dealt", "taken", "dmin", "away", "crash", "held", "held0", "tower")


class PVNet:
    """A CRN network policy that also returns its marginals and critic value (`m45c.horizon.CRNNet` sampling)."""

    def __init__(self, net, k0: int, seed: int, seat: int, horizon: int):
        self.net, self.k0 = net, int(k0)
        self.u = np.random.default_rng([int(seed), 7, int(seat)]).random(int(horizon) + 1)

    def forward(self, cfg, lane, m4, s: int, i: int):
        import torch
        from ....diagnostics.branch import Snapshot, observation
        obs = observation(cfg, Snapshot(lane, m4, self.k0 + i, (cfg.skill_a, cfg.skill_b)), s)
        with torch.no_grad():
            logits, v = self.net(torch.as_tensor(obs.vector, dtype=torch.float32).unsqueeze(0))
        lg = logits[0].double().numpy()[list(obs.legal)]
        p = np.exp(lg - lg.max())
        return list(obs.legal), p / p.sum(), float(v.reshape(-1)[0]) / C.REWARD_SCALE

    def pick(self, legal, p, i: int) -> int:
        if len(legal) == 1:
            return int(legal[0])
        j = min(int(np.searchsorted(np.cumsum(p), self.u[i], side="right")), len(p) - 1)
        return int(legal[j])

    def decide(self, cfg, lane, m4, s: int, i: int):
        legal, p, _ = self.forward(cfg, lane, m4, s, i)
        return self.pick(legal, p, i), "sample"


def marginals(legal, p):
    pw, pb = np.zeros(4), np.zeros(5)
    pward = ppress = 0.0
    for c, q in zip(legal, p):
        c = int(c)
        if c == CHAMP_ABSENT:
            continue
        if c == CHAMP_WARD:
            pward += q
            continue
        b = champ_behavior(c)
        pb[b] += q
        if b != B_RECALL:
            pw[champ_wave(c)] += q
        if champ_objective(c) == PRESSURE:
            ppress += q
    return pw, pb, pward, ppress


def lane_side(lane, s: int):
    fx, fh, rears, mass, vis, _ = lane.side(s)
    near = sum(1 for r in rears if fh > 0 and 0 < fx - r <= 2) if fh > 0 else 0
    return fx, fh, len(rears), near, mass, vis


def play_record(cfg, net, start, sd: int, turns: int, net_b=None) -> Dict[str, np.ndarray]:
    """One game from `start` (seat A `net`, seat B `net_b` or the same), full per-turn record (module docstring)."""
    from ....champions_m4 import context, resolve_turn_m4
    from ...m45b import engine as E
    from ..m44.records import holds_freeze
    lane, m4, k0 = start
    ctx = context(cfg)
    tb = ctx.t
    ls = E.lane_cache(cfg)
    sc = cfg.gold_scale
    pols = (PVNet(net, k0, sd, 0, turns), PVNet(net_b if net_b is not None else net, k0, sd, 1, turns))
    rng = np.random.default_rng(sd)
    T = turns
    d = {k: np.zeros((T, 2)) for k in PAIR_KEYS}
    d["p_wave"] = np.zeros((T, 2, 4))
    d["p_beh"] = np.zeros((T, 2, 5))
    d["r"] = np.zeros(T)
    d["k"] = np.zeros(T, dtype=np.int64)
    for i in range(T):
        u = rng.random(N_CHANCE)
        codes = []
        for s in (0, 1):
            fx, fh, nr, near, mass, vis = lane_side(lane, s)
            me = m4.side(s)
            L = int(tb.level_by_xp[min(me.xp, tb.xp_cap)])
            row = d
            row["fx"][i, s], row["fh"][i, s], row["nrear"][i, s], row["rear_near"][i, s] = fx, fh, nr, near
            row["mass"][i, s], row["vis"][i, s] = mass, vis
            row["pos"][i, s] = me.pos
            row["hp"][i, s] = 1.0 if me.absent else me.hp / tb.hp_max[L]
            row["L"][i, s], row["S"][i, s], row["g"][i, s] = L, me.items / sc, me.gold / sc
            row["absent"][i, s], row["xp"][i, s] = me.absent, me.xp
            row["C"][i, s] = ctx.capacity(L, me.items)
            row["held0"][i, s] = holds_freeze(lane, s)
            row["tower"][i, s] = me.tower_dmg
            legal, p, v = pols[s].forward(cfg, lane, m4, s, i)
            c = pols[s].pick(legal, p, i)
            codes.append(c)
            pw, pb, pward, ppress = marginals(legal, p)
            d["p_wave"][i, s], d["p_beh"][i, s] = pw, pb
            row["p_ward"][i, s], row["p_press"][i, s], row["V"][i, s] = pward, ppress, v
            row["code"][i, s] = c
            if c == CHAMP_ABSENT:
                row["wave"][i, s] = row["beh"][i, s] = -1
                row["obj"][i, s] = 0
            elif c == CHAMP_WARD:
                row["wave"][i, s], row["beh"][i, s], row["obj"][i, s] = -1, 0, VISION
            else:
                b = champ_behavior(c)
                row["beh"][i, s] = b
                row["wave"][i, s] = -1 if b == B_RECALL else champ_wave(c)
                row["obj"][i, s] = champ_objective(c)
        t = resolve_turn_m4(lane, m4, k0 + i, codes[0], codes[1], cfg, lane_step=ls, u=u, ctx=ctx)
        after = t.lane.state
        lv = t.levels_before
        for s in (0, 1):
            o = 1 - s
            H = tb.hp_max[lv[s]]
            d["earned"][i, s] = t.earned_units[s] / sc
            d["farm"][i, s] = t.farm_units[s] / sc
            d["avail"][i, s] = t.farm_available[s]
            d["bounty"][i, s] = t.bounty_units[s] / sc
            d["plate"][i, s] = t.plate_units[s] / sc
            d["xpg"][i, s] = t.xp_gain[s]
            d["died"][i, s] = t.died[s]
            d["recalled"][i, s] = t.recalled[s]
            d["intr"][i, s] = t.interrupted[s]
            d["conn"][i, s] = t.connected[s]
            d["dealt"][i, s] = t.trade_damage[s] / tb.hp_max[lv[o]]
            d["taken"][i, s] = t.damage_taken[s] / H
            d["dmin"][i, s] = t.minion_damage[s] / H
            d["away"][i, s] = t.away[s]
            d["crash"][i, s] = t.lane.crash_hp[s]
            d["held"][i, s] = holds_freeze(after, s)
        d["r"][i] = t.reward
        d["k"][i] = k0 + i
        lane, m4 = after, t.champ
    return d


def path(spec: str, arm: str) -> str:
    return os.path.join(C.CKPT, "records", arm, spec.replace(":", "_").replace("/", "_") + ".npz")


def record_job(args) -> tuple:
    """(spec, arm, seed index s, games, turns): s0 self-play records (resumable)."""
    spec, arm, s, games, turns = args
    p = path(spec, arm)
    if os.path.exists(p):
        return (spec, arm), {"path": p, "cached": True}
    cfg = C.setup(arm)
    from ...m45b import states as S
    net = C.net_of(spec)
    games_d = [play_record(cfg, net, S.s0(cfg), C.REC_ROOT * 1000 + 100 * s + g, turns) for g in range(games)]
    out = {k: np.stack([g[k] for g in games_d]) for k in games_d[0]}
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp.npz"
    np.savez_compressed(tmp, **out)
    os.replace(tmp, p)
    return (spec, arm), {"path": p}


def load(spec: str, arm: str) -> Dict[str, np.ndarray]:
    with np.load(path(spec, arm)) as z:
        return {k: z[k] for k in z.files}


def record_jobs(arms=("C1", "baseline"), games: int = GAMES, turns: int = TURNS):
    return [(C.general(arm, s), arm, s, games, turns) for arm in arms for s in C.SEEDS]


__all__ = ["PVNet", "play_record", "record_job", "record_jobs", "load", "path", "marginals", "GAMES", "TURNS"]
