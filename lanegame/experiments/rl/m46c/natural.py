"""Natural self-play of M4.6c: s0 games of a policy against itself in an environment, with per-turn lead arrays and
per-window lane aggregates (pre-registration §4 "natural").

A game g of seed s uses the chance stream and both seats' CRN sampling uniforms of NAT_ROOT * 1000 + 100 s + g, the
same for every policy and environment, so arms are compared on common random numbers game by game.

Per game: E (cumulative earned gold), S (invested), g (unspent), L (level), present and HP fraction per side and turn
(rounded), and per side and window (all 1-600, early 1-100, mid 101-240, late 241-600): crashed HP, plates, deaths,
Recalls (and interrupted), farm and farm available, Shove turns and wave turns (present, not recalling or warding),
trade attempts and connections, freeze turns held (holds_freeze after the turn) and frozen-on turns, Ward turns and
Wards placed, turns away, capacity C (mean over wave turns), damage taken; freeze episodes (maximal runs of held turns,
side by side); the turn of each plate.
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np

from ....champions import N_CHANCE
from . import common as C

GAMES, TURNS = 48, 600
WINDOWS = {"all": (1, 600), "early": (1, 100), "mid": (101, 240), "late": (241, 600)}
LANE_KEYS = ("crash_hp", "plates", "deaths", "recalls", "rec_intr", "farm", "farm_avail", "shove", "wave_turns",
             "trade_att", "trade_conn", "held", "frozen", "ward", "ward_placed", "away", "cap_sum", "damage_taken",
             "bounty", "plate_gold", "xp")


def path(spec: str, env: str) -> str:
    return os.path.join(C.CKPT, "eval", "natural", env, spec.replace(":", "_").replace("/", "_") + ".json.gz")


def natural_job(args) -> tuple:
    """(policy spec, environment arm, games, turns, seed): self-play records (resumable)."""
    spec, env, games, turns, seed = args
    p = path(spec, env)
    if os.path.exists(p):
        return (spec, env), {"path": p, "cached": True}
    cfg = C.setup(env)
    from ....config import B_RECALL, B_TRADE, SH as W_SH
    from ....champions_m4 import context, resolve_turn_m4
    from ...m45b import engine as E
    from ...m45b import states as S
    from ..m44.records import holds_freeze
    from ..m45c import horizon as HZ
    ctx = context(cfg)
    tb = ctx.t
    ls = E.lane_cache(cfg)
    net = C.net_of(spec)
    sc = cfg.gold_scale
    recs = []
    for g in range(games):
        sd = C.NAT_ROOT * 1000 + 100 * seed + g
        lane, m4, k0 = S.s0(cfg)
        pa, pb = HZ.CRNNet(net, k0, sd, 0, turns), HZ.CRNNet(net, k0, sd, 1, turns)
        rng = np.random.default_rng(sd)
        arr = {k: [[], []] for k in ("E", "S", "g", "L", "P", "hp")}
        per_turn = {k: np.zeros((2, turns)) for k in LANE_KEYS}
        plate_turns = [[], []]
        Ecum = [0.0, 0.0]
        for i in range(turns):
            u = rng.random(N_CHANCE)
            a = E.decide(pa, cfg, lane, m4, 0, i)
            b = E.decide(pb, cfg, lane, m4, 1, i)
            t = resolve_turn_m4(lane, m4, k0 + i, a, b, cfg, lane_step=ls, u=u, ctx=ctx)
            before = m4
            lane, m4 = t.lane.state, t.champ
            for x in (0, 1):
                me = m4.side(x)
                Ecum[x] += t.earned_units[x] / sc
                L = int(tb.level_by_xp[min(me.xp, tb.xp_cap)])
                arr["E"][x].append(round(Ecum[x], 3))
                arr["S"][x].append(round(me.items / sc, 3))
                arr["g"][x].append(round(me.gold / sc, 3))
                arr["L"][x].append(L)
                arr["P"][x].append(int(me.absent == 0))
                arr["hp"][x].append(round(me.hp / tb.hp_max[L], 3) if me.absent == 0 else 1.0)
                played = (not t.away[x]) and t.behaviors[x] != B_RECALL and not t.warding[x]
                pt = per_turn
                pt["crash_hp"][x, i] = t.lane.crash_hp[x]
                pt["plates"][x, i] = t.plates_broken[x]
                if t.plates_broken[x]:
                    plate_turns[x] += [i + 1] * int(t.plates_broken[x])
                pt["deaths"][x, i] = t.died[x]
                pt["recalls"][x, i] = t.recalled[x]
                pt["rec_intr"][x, i] = t.interrupted[x]
                pt["farm"][x, i] = t.farm_units[x] / sc
                pt["farm_avail"][x, i] = t.farm_available[x]
                pt["wave_turns"][x, i] = played
                pt["shove"][x, i] = played and t.waves[x] == W_SH
                pt["trade_att"][x, i] = played and t.behaviors[x] == B_TRADE
                pt["trade_conn"][x, i] = t.connected[x]
                pt["held"][x, i] = holds_freeze(lane, x) and me.absent == 0
                pt["frozen"][x, i] = holds_freeze(lane, 1 - x) and me.absent == 0
                pt["ward"][x, i] = t.warding[x]
                pt["ward_placed"][x, i] = t.ward_placed[x]
                pt["away"][x, i] = t.away[x]
                L0 = int(tb.level_by_xp[min(before.side(x).xp, tb.xp_cap)])
                pt["cap_sum"][x, i] = ctx.capacity(L0, before.side(x).items) if played else 0
                pt["damage_taken"][x, i] = t.damage_taken[x]
                pt["bounty"][x, i] = t.bounty[x]
                pt["plate_gold"][x, i] = t.plate_gold[x]
                pt["xp"][x, i] = t.xp_gain[x]
        lane_agg = {}
        for w, (lo, hi) in WINDOWS.items():
            sl = slice(lo - 1, min(hi, turns))
            lane_agg[w] = {k: [float(per_turn[k][x, sl].sum()) for x in (0, 1)] for k in LANE_KEYS}
        # freeze episodes: maximal runs of held turns
        eps = [[], []]
        for x in (0, 1):
            h = per_turn["held"][x]
            run = 0
            for v in list(h) + [0]:
                if v:
                    run += 1
                elif run:
                    eps[x].append(run)
                    run = 0
        recs.append({"arr": arr, "lane": lane_agg, "freeze_episodes": eps, "plate_turns": plate_turns})
    data = {"spec": spec, "env": env, "trained_on": C.trained_on(spec), "games": games, "turns": turns,
            "seed_root": C.NAT_ROOT, "records": recs}
    C.save_gz(p, data)
    return (spec, env), {"path": p}


# ---------------------------------------------------------------------------------------------- metrics
def metrics(data: dict) -> Dict[str, float]:
    """Per-policy metrics (means over games and both sides) of one natural record set."""
    cfg = C.arm_cfg(data["env"])
    from ....champions_m4 import context
    ctx = context(cfg)
    sc = cfg.gold_scale
    recs = data["records"]
    T = data["turns"]
    out: Dict[str, float] = {}
    # lane metrics per 100 turns (all window), per side
    for w, (lo, hi) in WINDOWS.items():
        n = min(hi, T) - lo + 1
        for k in LANE_KEYS:
            v = [r["lane"][w][k][x] for r in recs for x in (0, 1)]
            out[f"{w}:{k}_per100"] = float(np.mean(v)) * 100.0 / n
        fa = sum(r["lane"][w]["farm_avail"][x] for r in recs for x in (0, 1))
        out[f"{w}:collection"] = sum(r["lane"][w]["farm"][x] for r in recs for x in (0, 1)) / fa if fa else np.nan
        wt = sum(r["lane"][w]["wave_turns"][x] for r in recs for x in (0, 1))
        out[f"{w}:shove_share"] = sum(r["lane"][w]["shove"][x] for r in recs for x in (0, 1)) / wt if wt else np.nan
        out[f"{w}:cap_mean"] = sum(r["lane"][w]["cap_sum"][x] for r in recs for x in (0, 1)) / wt if wt else np.nan
        out[f"{w}:held_share"] = float(np.mean([r["lane"][w]["held"][x] for r in recs for x in (0, 1)])) / n
        out[f"{w}:away_share"] = float(np.mean([r["lane"][w]["away"][x] for r in recs for x in (0, 1)])) / n
    out["plates_by150"] = float(np.mean([sum(1 for p in r["plate_turns"][x] if p <= 150) for r in recs for x in (0, 1)]))
    out["plates_by300"] = float(np.mean([sum(1 for p in r["plate_turns"][x] if p <= 300) for r in recs for x in (0, 1)]))
    eps = [e for r in recs for x in (0, 1) for e in r["freeze_episodes"][x]]
    out["freeze_episodes_per100"] = len(eps) / (len(recs) * 2) * 100.0 / T
    out["freeze_episode_mean_len"] = float(np.mean(eps)) if eps else 0.0
    out["freeze_episode_ge3_share"] = float(np.mean([e >= 3 for e in eps])) if eps else 0.0
    # leads and strength
    E = np.array([[r["arr"]["E"][0], r["arr"]["E"][1]] for r in recs], dtype=float)       # [game, side, turn]
    S = np.array([[r["arr"]["S"][0], r["arr"]["S"][1]] for r in recs], dtype=float)
    L = np.array([[r["arr"]["L"][0], r["arr"]["L"][1]] for r in recs], dtype=float)
    P = np.array([[r["arr"]["P"][0], r["arr"]["P"][1]] for r in recs], dtype=float)
    dE = E[:, 0, :] - E[:, 1, :]
    meanE = E.mean(1)
    for t in (25, 50, 100, 150, 300, 600):
        if t <= T:
            out[f"abs_lead@{t}_median"] = float(np.median(np.abs(dE[:, t - 1])))
            out[f"abs_lead@{t}_q90"] = float(np.quantile(np.abs(dE[:, t - 1]), 0.9))
            out[f"rel_lead@{t}_median"] = float(np.median(np.abs(dE[:, t - 1]) / np.maximum(meanE[:, t - 1], 1e-9)))
            dS = S[:, 0, t - 1] - S[:, 1, t - 1]
            out[f"abs_inv_lead@{t}_median"] = float(np.median(np.abs(dS)))
    for a, b in ((50, 150), (25, 100), (100, 300), (150, 300), (300, 600)):
        if b <= T and dE[:, a - 1].std() > 0:
            out[f"slope_{b}_on_{a}"] = float(np.polyfit(dE[:, a - 1], dE[:, b - 1], 1)[0])
    if T >= 150:
        out["decisive150_share"] = float(np.mean(np.abs(dE[:, 149]) >= 0.15 * meanE[:, 149]))
    # attack and duel ratios on present turns (levels once / twice; investment once / twice)
    Mf = np.vectorize(lambda s: ctx.inv.factor(int(round(float(s) * sc))))
    both = (P[:, 0, :] > 0) & (P[:, 1, :] > 0)
    mA, mB = Mf(S[:, 0, :][both]), Mf(S[:, 1, :][both])
    lvl = (L[:, 0, :][both] - L[:, 1, :][both]) * np.log(1.05)
    ar = np.exp(lvl) * mA / mB
    att = np.maximum(ar, 1 / ar)
    duel = att ** 2
    out["attack_q90"] = float(np.quantile(att, 0.9))
    out["attack_share_1.1"] = float((att >= 1.1).mean())
    out["attack_share_1.25"] = float((att >= 1.25).mean())
    out["duel_q90"] = float(np.quantile(duel, 0.9))
    out["duel_share_1.25"] = float((duel >= 1.25).mean())
    # the gold leader's duel ratio by stage (geometric mean over present turns)
    for name, (lo, hi) in {"early": (20, 50), "mid": (90, 150), "late": (240, 320), "very_late": (450, 600)}.items():
        if hi > T:
            continue
        m = both[:, lo - 1:hi]
        lead_sign = np.sign(dE[:, lo - 1:hi])[m]
        sa, sb = S[:, 0, lo - 1:hi][m], S[:, 1, lo - 1:hi][m]
        la, lb = L[:, 0, lo - 1:hi][m], L[:, 1, lo - 1:hi][m]
        r = np.exp((la - lb) * np.log(1.05)) * Mf(sa) / Mf(sb)
        r = np.where(lead_sign >= 0, r, 1 / r)
        out[f"leader_duel_{name}"] = float(np.exp(np.mean(np.log(r ** 2))))
    out["level_end"] = float(L[:, :, -1].mean())
    out["S_end"] = float(S[:, :, -1].mean())
    out["E_end"] = float(E[:, :, -1].mean())
    return out


__all__ = ["natural_job", "metrics", "path", "GAMES", "TURNS", "WINDOWS", "LANE_KEYS"]
