"""E3: power-gap self-play of each final (pre-registration §2 E3), the semi-natural gap states for E2 / E4, and the
critic aggregates for E6.

Cells: the M4.5b zoning situation "neutral" at levels 6 / 10 / 14 x A's lead equal / x1.24 / x1.5 / x1.76 / x2.2
(investment only) x orientation (0: A stronger; 1: the mirrored state, B stronger). Games g = 0..47 of 300 turns, both
seats the final with CRN sampling; the chance and sampling seeds depend on (level, orientation, g) only, so every gap
is paired with the equal-power game of the same seeds. Games 0-31 are the TRAINING split (E2's semi-natural family,
E4's archives), 32-47 the HELD-OUT split (E4's evaluation states).

Per game, from the stronger side's perspective (s* = stronger seat, w = weaker): G_h of s* (discounted) and the
undiscounted lead at h = 20 / 50 / 100 / 300, and per side the ledger at those horizons. Per turn while both are
present and s*'s power ratio is >= 1.25: s*'s Trade attempt / connection, cell, w's Yield / Safe, w at its tower cell,
w's farm lost while s* can catch it without a dive vs otherwise.

Semi-natural states: per game of a gap >= x1.5, one seeded draw among turns 3..80 where both are present and s*'s power
ratio is still >= 1.4, mirrored so that the stronger side is seat A.

E6 critic aggregates (both seats): for t <= 200, the error V(s_t) - (sum_{j<100} 0.99^j r_{t+j} + 0.99^100 V(s_{t+100}))
by the seat's turn class (Trade connected, Trade not connected, other), as sums of error, |error|, error^2, target,
target^2 and counts.

Continuity (E3-P3): the M4.5c stage `health.gap_job` (viewer neutral, level 3, A x1.76) re-run on new seeds.
"""

from __future__ import annotations

import time
from typing import Dict, List

import numpy as np

from ....champions import N_CHANCE
from ....config import B_SAFE, B_TRADE, B_YIELD
from ....positional import WARD
from .common import (CKPT, E3_ROOT, FINALS, GAMMA, load_gz, meta, net_of, run_jobs, save_gz, save_json, setup_worker,
                     spec, state_to_list)

LEVELS = (6, 10, 14)
GAPS = ("equal", "x1.24", "x1.5", "x1.76", "x2.2")
GAMES, TRAIN_GAMES, TURNS = 48, 32, 300
HS = (20, 50, 100, 300)
SN_GAPS = ("x1.5", "x1.76", "x2.2")
SN_TURNS = (3, 80)
SN_POWER = 1.4
KEYS = ("earned", "farm", "farm_avail", "unc", "unc_away", "bounty", "plate", "xp", "xp_lost", "deaths", "recalls",
        "away", "trade_att", "trade_conn", "d_total", "trade_dealt", "held", "frozen")
E6_CLASSES = ("trade_conn", "trade_unconn", "other")
E6_H = 100


def game_seed(level: int, orient: int, g: int) -> int:
    return E3_ROOT * 1000 + 1000 * level + 100 * orient + g


def start_of(cfg, level: int, gap: str, orient: int):
    from . import states as ST
    st = ST.zoning_gap_state(cfg, "neutral", level, gap)
    return ST.mirror(st) if orient == 1 else st


def _path(seed: int, level: int, gap: str, orient: int) -> str:
    import os
    return os.path.join(CKPT, "e3", f"s{seed}", f"L{level}_{gap}_o{orient}.json.gz")


def game(cfg, net, start, seed: int, strong: int, collect_states: bool) -> dict:
    from ....champions_m4 import context, resolve_turn_m4
    from ...m45b import engine as E
    from ...m45b.strategies import catchable
    from ..m44.records import power_ratio
    from .engine import XLedger
    from .policies import CRNValueNet
    lane, m4, k0 = start
    ctx = context(cfg)
    tb = ctx.t
    ls = E.lane_cache(cfg)
    rng = np.random.default_rng(int(seed))
    pa, pb = CRNValueNet(net, k0, seed, 0, TURNS), CRNValueNet(net, k0, seed, 1, TURNS)
    led = XLedger(cfg, start)
    w = 1 - strong
    sign = 1.0 if strong == 0 else -1.0
    snaps, G, lead = {}, 0.0, 0.0
    rew, vals, cls = [], [[], []], [[], []]
    gap_turns = {"n": 0, "s_trade_att": 0, "s_trade_conn": 0, "s_pos": 0.0, "s_lane": 0, "s_W": 0, "w_pos": 0.0,
                 "w_lane": 0, "w_W": 0, "w_yield": 0, "w_safe": 0, "w_tower": 0,
                 "w_unc_near": 0.0, "n_near": 0, "w_unc_far": 0.0, "n_far": 0}
    cand = []
    for i in range(TURNS):
        u = rng.random(N_CHANCE)
        a = E.decide(pa, cfg, lane, m4, 0, i)
        b = E.decide(pb, cfg, lane, m4, 1, i)
        vals[0].append(pa.last_value)
        vals[1].append(pb.last_value)
        both = m4.a.absent == 0 and m4.b.absent == 0
        pr = power_ratio(ctx, m4, strong) if both else 1.0
        near = both and catchable(cfg, m4, strong)
        if collect_states and both and SN_TURNS[0] <= i <= SN_TURNS[1] and pr >= SN_POWER:
            cand.append((i, lane, m4, k0 + i))
        t = resolve_turn_m4(lane, m4, k0 + i, a, b, cfg, lane_step=ls, u=u, ctx=ctx)
        led.add(t, lane)
        for s in (0, 1):
            cls[s].append(0 if (t.behaviors[s] == B_TRADE and t.connected[s]) else
                          (1 if t.behaviors[s] == B_TRADE else 2))
        if both and pr >= 1.25:
            gt = gap_turns
            gt["n"] += 1
            gt["s_trade_att"] += t.behaviors[strong] == B_TRADE
            gt["s_trade_conn"] += bool(t.connected[strong])
            for side, key in ((strong, "s"), (w, "w")):       # lane cells only: the Ward node W is stored as 99
                if t.start[side] == WARD:
                    gt[f"{key}_W"] += 1
                else:
                    gt[f"{key}_pos"] += t.start[side]
                    gt[f"{key}_lane"] += 1
            gt["w_yield"] += t.behaviors[w] == B_YIELD
            gt["w_safe"] += t.behaviors[w] == B_SAFE
            gt["w_tower"] += t.start[w] == -cfg.tower
            unc = t.farm_available[w] - t.farm_units[w] / cfg.gold_scale
            if near:
                gt["w_unc_near"] += unc
                gt["n_near"] += 1
            else:
                gt["w_unc_far"] += unc
                gt["n_far"] += 1
        rew.append(t.reward)
        G += GAMMA ** i * t.reward
        lead += t.reward
        lane, m4 = t.lane.state, t.champ
        n = i + 1
        if n in HS:
            snaps[n] = {"G": sign * G, "lead": sign * lead,
                        **{f"{kk}_s": led.v[kk][strong] for kk in KEYS}, **{f"{kk}_w": led.v[kk][w] for kk in KEYS}}
    # E6 critic aggregates
    r = np.asarray(rew)
    e6 = {s: {c: np.zeros(6) for c in E6_CLASSES} for s in (0, 1)}
    disc = GAMMA ** np.arange(E6_H)
    for s in (0, 1):
        rs = r if s == 0 else -r
        v = np.asarray(vals[s])
        for t0 in range(0, TURNS - E6_H):
            if m4 is None:
                break
            target = float(np.dot(disc, rs[t0:t0 + E6_H]) + GAMMA ** E6_H * v[t0 + E6_H])
            err = v[t0] - target
            c = E6_CLASSES[cls[s][t0]]
            e6[s][c] += np.array([1.0, err, abs(err), err * err, target, target * target])
    states = []
    if collect_states and cand:
        j = int(np.random.default_rng([int(seed), 11]).integers(len(cand)))
        i, ln, mm, kk = cand[j]
        st = (ln, mm, kk)
        if strong == 1:
            st = (ln.mirror(), mm.mirror(), kk)
        states.append({"turn": i, "state": state_to_list(st)})
    return {"snaps": snaps, "gap_turns": gap_turns, "states": states,
            "e6": {str(s): {c: e6[s][c].tolist() for c in E6_CLASSES} for s in (0, 1)}}


def cell_job(args) -> tuple:
    """(final seed, level, gap, orientation): GAMES games (resumable per cell)."""
    import os
    seed, level, gap, orient = args
    path = _path(seed, level, gap, orient)
    if os.path.exists(path):
        return (seed, level, gap, orient), load_gz(path)
    cfg = setup_worker()
    net = net_of(spec(seed))
    start = start_of(cfg, level, gap, orient)
    strong = orient
    t0 = time.perf_counter()
    games = []
    for g in range(GAMES):
        res = game(cfg, net, start, game_seed(level, orient, g), strong, gap in SN_GAPS)
        res["g"] = g
        res["split"] = "train" if g < TRAIN_GAMES else "heldout"
        games.append(res)
    rec = {"seed": seed, "level": level, "gap": gap, "orient": orient, "games": games,
           "elapsed_s": time.perf_counter() - t0}
    save_gz(path, rec)
    return (seed, level, gap, orient), rec


def continuity_job(args) -> tuple:
    s, games, turns, root = args
    setup_worker()
    from ..m45c.health import gap_job
    k, v = gap_job((spec(s), games, turns, root))
    return s, v


def run(workers: int, finals=FINALS) -> dict:
    t0 = time.perf_counter()
    js = [(s, L, g, o) for s in finals for L in LEVELS for g in GAPS for o in (0, 1)]
    res = run_jobs(cell_job, js, workers, "E3 gap self-play")
    cont = dict(run_jobs(continuity_job, [(s, 16, 300, E3_ROOT + 90) for s in finals], workers, "E3 continuity"))
    summ = summarize({k: v for k, v in res}, cont)
    save_json("e3_summary", {"meta": meta(stage="e3", levels=LEVELS, gaps=GAPS, games=GAMES, turns=TURNS),
                             "summary": summ, "elapsed_s": time.perf_counter() - t0})
    return summ


# ---------------------------------------------------------------------------------------------- analysis
def summarize(cells: Dict[tuple, dict], continuity: Dict[int, dict] = None) -> dict:
    """Per final x level x gap (orientations pooled): mean over games of each snapshot quantity, and the paired
    premium over the equal-power game of the same seeds; the M4.5c-style readouts; the E6 aggregates."""
    out = {"per_final": {}, "e6": {}, "continuity": {}}
    for (s, L, gap, o), rec in cells.items():
        out["per_final"].setdefault(str(s), {}).setdefault(str(L), {}).setdefault(gap, {})[str(o)] = rec
    tab = {}
    for s, byL in out["per_final"].items():
        for L, byg in byL.items():
            eq = byg.get("equal")
            for gap, byo in byg.items():
                row = {}
                for h in HS:
                    vals, prem = {}, {}
                    for o, rec in byo.items():
                        for gm in rec["games"]:
                            sn = gm["snaps"][str(h)] if str(h) in gm["snaps"] else gm["snaps"].get(h)
                            e = None
                            if eq and o in eq:
                                eg = eq[o]["games"][gm["g"]]
                                e = eg["snaps"][str(h)] if str(h) in eg["snaps"] else eg["snaps"].get(h)
                            for kk, v in sn.items():
                                vals.setdefault(kk, []).append(v)
                                if e is not None:
                                    prem.setdefault(kk, []).append(v - e[kk])
                    row[str(h)] = {"mean": {k: float(np.mean(v)) for k, v in vals.items()},
                                   "premium": {k: float(np.mean(v)) for k, v in prem.items()},
                                   "premium_sd": {k: float(np.std(v, ddof=1)) for k, v in prem.items()},
                                   "n": len(vals.get("G", []))}
                gt = {}
                for o, rec in byo.items():
                    for gm in rec["games"]:
                        for kk, v in gm["gap_turns"].items():
                            gt[kk] = gt.get(kk, 0) + v
                n = max(gt.get("n", 0), 1)
                row["gap_turns"] = {"turns": gt.get("n", 0), "s_trade_rate": gt.get("s_trade_att", 0) / n,
                                    "s_conn_rate": gt.get("s_trade_conn", 0) / n,
                                    "s_pos": gt.get("s_pos", 0) / max(gt.get("s_lane", 0), 1),
                                    "s_W": gt.get("s_W", 0) / n,
                                    "w_pos": gt.get("w_pos", 0) / max(gt.get("w_lane", 0), 1),
                                    "w_W": gt.get("w_W", 0) / n,
                                    "w_yield": gt.get("w_yield", 0) / n, "w_safe": gt.get("w_safe", 0) / n,
                                    "w_tower": gt.get("w_tower", 0) / n,
                                    "w_unc_near": gt.get("w_unc_near", 0) / max(gt.get("n_near", 0), 1),
                                    "w_unc_far": gt.get("w_unc_far", 0) / max(gt.get("n_far", 0), 1),
                                    "near_share": gt.get("n_near", 0) / n}
                tab.setdefault(s, {}).setdefault(L, {})[gap] = row
    out["table"] = tab
    # E6: pooled over cells, per final and seat class
    e6 = {}
    for (s, L, gap, o), rec in cells.items():
        for gm in rec["games"]:
            for seat, d in gm["e6"].items():
                for c, v in d.items():
                    acc = e6.setdefault(str(s), {}).setdefault(c, np.zeros(6))
                    acc += np.asarray(v)
    for s, d in e6.items():
        out["e6"][s] = {}
        for c, v in d.items():
            n, se, sa, ss, st, st2 = v
            if n <= 1:
                continue
            var_t = st2 / n - (st / n) ** 2
            var_e = ss / n - (se / n) ** 2
            out["e6"][s][c] = {"n": int(n), "bias": se / n, "mae": sa / n, "rmse": float(np.sqrt(ss / n)),
                               "target_sd": float(np.sqrt(max(var_t, 0))),
                               "explained_variance": float(1 - var_e / var_t) if var_t > 0 else float("nan")}
    for s, v in (continuity or {}).items():
        z, c = v["zoning"], v["conditional"]
        out["continuity"][str(s)] = {
            "by_power": {k: {kk: x.get(kk) for kk in ("n", "trade_attempt_rate", "mean_pos", "at_own_tower")}
                         for k, x in z.get("by_power", {}).items()},
            "pressure_cost": z.get("pressure_cost"),
            "weaker_mass": c.get("weaker (power ratio <= 0.8)", {}).get("mass")}
    out.pop("per_final")
    return out


def semi_natural(seed: int, split: str, gaps=SN_GAPS, levels=LEVELS) -> List[dict]:
    """The semi-natural gap states of one final's E3 games ({"start", "level", "gap", "orient", "g", "turn"})."""
    import os
    from .common import state_from_list
    out = []
    for L in levels:
        for gap in gaps:
            for o in (0, 1):
                p = _path(seed, L, gap, o)
                if not os.path.exists(p):
                    continue
                rec = load_gz(p)
                for gm in rec["games"]:
                    if gm["split"] != split:
                        continue
                    for st in gm["states"]:
                        out.append({"start": state_from_list(st["state"]), "level": L, "gap": gap, "orient": o,
                                    "g": gm["g"], "turn": st["turn"]})
    return out


__all__ = ["run", "summarize", "semi_natural", "LEVELS", "GAPS", "HS", "game_seed", "start_of"]


# ---------------------------------------------------------------------------------------------- measurement check
def positions_job(args) -> tuple:
    """(agent spec, games, root): the M4.5c power-gap stage's games (viewer neutral, level 3, A x1.76; M4.4
    play_record on SeedSequence(root, spawn_key=(g,))), the stronger side's mean cell computed as M4.5c did (every
    cell, the Ward node W stored as 99) and on lane cells only, with the share of turns at W (both seats)."""
    agent_spec, games, root = args
    setup_worker()
    from ..m44.metrics import power_class
    from ..m44.records import play_record
    from ..m45c.common import agent_from_spec, env_cfg
    from ..m45c.traces import _gap_start
    cfg = env_cfg()
    x = agent_from_spec(agent_spec, cfg)
    out = {"stronger": {"all": [], "lane": [], "W": []}, "weaker": {"all": [], "lane": [], "W": []}}
    for g in range(games):
        rec = play_record(cfg, (x, x), np.random.SeedSequence(root, spawn_key=(g,)), TURNS, _gap_start(),
                          "neutral/large", "m46a")
        d = rec.data
        for s in (0, 1):
            both = (d["present"][:, s] > 0) & (d["en_present"][:, s] > 0)
            for t in np.flatnonzero(both):
                k = power_class(float(d["power"][t, s]))
                if k == "even":
                    continue
                p = int(d["pos"][t, s])
                o = out[k]
                o["all"].append(p)
                o["W"].append(p == WARD)
                if p != WARD:
                    o["lane"].append(p)
    res = {k: {"turns": len(v["all"]), "mean_all": float(np.mean(v["all"])) if v["all"] else float("nan"),
               "mean_lane": float(np.mean(v["lane"])) if v["lane"] else float("nan"),
               "share_W": float(np.mean(v["W"])) if v["W"] else float("nan"),
               "cells": {c: float(np.mean(np.asarray(v["lane"]) == c)) for c in range(-2, 3)} if v["lane"] else {}}
           for k, v in out.items()}
    return agent_spec, res


def positions(workers: int, finals=FINALS) -> dict:
    specs = [spec(s) for s in finals] + [f"final:full:{s}" for s in finals]
    res = dict(run_jobs(positions_job, [(sp, 16, E3_ROOT + 91) for sp in specs], workers, "E3 positions"))
    save_json("e3_positions", {"meta": meta(stage="e3_positions", games=16, turns=TURNS, root=E3_ROOT + 91,
                                            start="viewer neutral, level 3, A x1.76 (the M4.5c stage)"),
                               "results": res})
    return res


# ---------------------------------------------------------------------------------------------- natural gaps
NAT_GAMES, NAT_TURNS = 16, 600
LEAD_BINS = ((0, 3), (3, 6), (6, 10), (10, 20), (20, 1000))


def natural_gaps_job(args) -> tuple:
    """EXPLORATORY: s0 self-play of one final (CRN sampling, seeds E3_ROOT x 1000 + 900,000 + 100 x final + g): the
    power ratio (stronger / weaker, `m44.records.power_ratio`) on turns with both present, against the cumulative
    undiscounted gold lead at that turn, and the turn's invested-gold gap."""
    s, games = args
    cfg = setup_worker()
    from ....champions_m4 import context, resolve_turn_m4
    from ...m45b import engine as E
    from ...m45b import states as S
    from ..m44.records import power_ratio
    from ..m45c import horizon as HZ
    ctx = context(cfg)
    ls = E.lane_cache(cfg)
    net = net_of(spec(s))
    pr, lead_abs, inv_gap, turn = [], [], [], []
    for g in range(games):
        sd = E3_ROOT * 1000 + 900_000 + 100 * s + g
        lane, m4, k0 = S.s0(cfg)
        pa, pb = HZ.CRNNet(net, k0, sd, 0, NAT_TURNS), HZ.CRNNet(net, k0, sd, 1, NAT_TURNS)
        rng = np.random.default_rng(sd)
        lead = 0.0
        for i in range(NAT_TURNS):
            u = rng.random(N_CHANCE)
            a = E.decide(pa, cfg, lane, m4, 0, i)
            b = E.decide(pb, cfg, lane, m4, 1, i)
            t = resolve_turn_m4(lane, m4, k0 + i, a, b, cfg, lane_step=ls, u=u, ctx=ctx)
            lead += t.reward
            lane, m4 = t.lane.state, t.champ
            if m4.a.absent == 0 and m4.b.absent == 0:
                r = power_ratio(ctx, m4, 0)
                pr.append(max(r, 1.0 / r))
                lead_abs.append(abs(lead))
                inv_gap.append(abs(m4.a.items - m4.b.items) / cfg.gold_scale)
                turn.append(i + 1)
    pr, la, ig, tu = map(np.asarray, (pr, lead_abs, inv_gap, turn))
    out = {"turns": int(len(pr)), "quantiles": {str(q): float(np.quantile(pr, q)) for q in (0.5, 0.9, 0.99, 0.999)},
           "share": {str(x): float((pr >= x).mean()) for x in (1.1, 1.25, 1.5)},
           "lead_quantiles": {str(q): float(np.quantile(la, q)) for q in (0.5, 0.9, 0.99)},
           "by_lead": {f"{lo}-{hi}": {"n": int(((la >= lo) & (la < hi)).sum()),
                                      "mean_ratio": float(pr[(la >= lo) & (la < hi)].mean()) if ((la >= lo) & (la < hi)).any() else float("nan"),
                                      "q90_ratio": float(np.quantile(pr[(la >= lo) & (la < hi)], 0.9)) if ((la >= lo) & (la < hi)).any() else float("nan")}
                       for lo, hi in LEAD_BINS},
           "by_phase": {f"{lo}-{hi}": {"mean_ratio": float(pr[(tu > lo) & (tu <= hi)].mean()),
                                       "share_1.25": float((pr[(tu > lo) & (tu <= hi)] >= 1.25).mean())}
                        for lo, hi in ((0, 50), (50, 150), (150, 300), (300, 600))},
           "inv_gap_quantiles": {str(q): float(np.quantile(ig, q)) for q in (0.5, 0.9, 0.99)}}
    return s, out


def natural_gaps(workers: int, finals=FINALS) -> dict:
    res = dict(run_jobs(natural_gaps_job, [(s, NAT_GAMES) for s in finals], workers, "E3 natural gaps"))
    save_json("e3_natural_gaps", {"meta": meta(stage="e3_natural_gaps", games=NAT_GAMES, turns=NAT_TURNS,
                                               label="exploratory"), "results": res})
    return res
