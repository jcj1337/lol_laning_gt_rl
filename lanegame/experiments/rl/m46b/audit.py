"""The gold-to-strength audit of the CURRENT curve (M4.6b Part 2). Nothing here evaluates a candidate's gameplay.

Formulas (code: progression.Investment, champions_m4.M4Context; reference §7.3, §4.5, §6.2):

    E   earned gold (farm + plates + bounty), cumulative;  E = S + g at all times
    g   unspent gold: earned since the last return to base (the reward counts it the turn it is earned)
    S   invested gold: on every return (the turn a Recall or death absence ends) S <- S + g, g <- 0
    M   M(S) = 1 + 1.5 (1 - sqrt(16 / (16 + S)))          (exact integers; 1 <= M < 2.5)
    AP  AP*(L, S) = 1.05^(L-1) M(S)                       trade damage dealt, wave capacity, Pressure
    H   max HP H(L) = 600 x 1.05^(L-1) centi-HP           (levels only)
    dur every centi-HP taken (trade, tower shot, gank, minions) is divided by M(S)
    C   wave-clear capacity C = round(4 AP*) HP per turn; Shove spends C, Thin / Slow Push up to round(C / 2)

Ratios of two champions (A over B), equal skills:
    attack ratio   AP_A / AP_B = 1.05^dL x M_A / M_B
    M4.4 "power"   (hit_A / u_B) / (hit_B / u_A) = 1.05^(2 dL) x M_A / M_B     (records.power_ratio: damage per enemy HP
                   unit, which counts the level twice but the investment once: it omits durability)
    duel ratio     (damage A deals in B's bars) / (damage B deals in A's bars) = 1.05^(2 dL) x (M_A / M_B)^2

Trajectories. s0 self-play of the six 16.384M finals (CRN sampling, seeds AUDIT_ROOT x 1000 + 100 x final + g), GAMES
games x TURNS turns; per turn and side: S, g, E, level, presence, returns and what they invested.
"""

from __future__ import annotations

import math
import os
import time
from typing import Dict, List

import numpy as np

from ....champions import N_CHANCE
from .common import AUDIT_ROOT, CKPT, FINALS, GAMMA, load_gz, meta, net_of, run_jobs, save_gz, save_json, setup, spec

GAMES, TURNS = 24, 600
STAGES = {"early": (20, 50), "mid": (90, 150), "late": (240, 320), "very late": (450, 600)}
INCOME = 0.40                     # resources per turn: the measured natural income (audit, turns 1-600)
LEAD_TURNS = (5, 10, 20, 40)      # leads expressed as turns of income


def _path(s: int, arm: str = None) -> str:
    if arm is None:
        return os.path.join(CKPT, "audit", f"s{s}.json.gz")
    return os.path.join(CKPT, "natural", arm, f"s{s}.json.gz")


def trajectories_job(args) -> tuple:
    """(final, games[, arm, seed root, turns]): s0 self-play records. Without an arm: the audit (baseline, AUDIT_ROOT);
    with an arm: the S6 natural-start screen (the arm's configuration, the given seed root, the same seeds for every arm)
    with the lane-sanity readouts as well."""
    s, games, *rest = args
    arm = rest[0] if rest else None
    root = rest[1] if len(rest) > 1 else AUDIT_ROOT
    turns = rest[2] if len(rest) > 2 else TURNS
    path = _path(s, arm)
    if os.path.exists(path):
        return s, load_gz(path)
    cfg = setup(arm or "baseline")
    from ....config import B_RECALL, SH as W_SH
    from ....champions_m4 import context, resolve_turn_m4
    from ...m45b import engine as E
    from ...m45b import states as S
    from ..m45c import horizon as HZ
    ctx = context(cfg)
    tb = ctx.t
    ls = E.lane_cache(cfg)
    net = net_of(spec(s))
    sc = cfg.gold_scale
    out = []
    for g in range(games):
        sd = root * 1000 + 100 * s + g
        lane, m4, k0 = S.s0(cfg)
        pa, pb = HZ.CRNNet(net, k0, sd, 0, turns), HZ.CRNNet(net, k0, sd, 1, turns)
        rng = np.random.default_rng(sd)
        rec = {k: [[], []] for k in ("S", "g", "E", "L", "present", "returned", "bought", "hp")}
        lane_k = ("crash_hp", "plates", "deaths", "recalls", "farm", "farm_avail", "shove", "wave_turns", "trade_conn",
                  "cap")
        lan = {k: [[], []] for k in lane_k}
        Ecum = [0.0, 0.0]
        rew = []
        for i in range(turns):
            u = rng.random(N_CHANCE)
            a = E.decide(pa, cfg, lane, m4, 0, i)
            b = E.decide(pb, cfg, lane, m4, 1, i)
            t = resolve_turn_m4(lane, m4, k0 + i, a, b, cfg, lane_step=ls, u=u, ctx=ctx)
            before = m4
            lane, m4 = t.lane.state, t.champ
            rew.append(t.reward)
            for x in (0, 1):
                me = m4.side(x)
                Ecum[x] += t.earned_units[x] / sc
                rec["S"][x].append(round(me.items / sc, 4))
                rec["g"][x].append(round(me.gold / sc, 4))
                rec["E"][x].append(round(Ecum[x], 4))
                rec["L"][x].append(int(tb.level_by_xp[min(me.xp, tb.xp_cap)]))
                rec["present"][x].append(int(me.absent == 0))
                rec["returned"][x].append(int(t.returned[x]))
                rec["bought"][x].append(round((me.items - before.side(x).items) / sc, 4))
                L = tb.level_by_xp[min(me.xp, tb.xp_cap)]
                rec["hp"][x].append(round(me.hp / tb.hp_max[L], 4) if me.absent == 0 else 1.0)
                if arm is not None:                                   # S6 lane-sanity readouts (per turn)
                    lan["crash_hp"][x].append(int(t.lane.crash_hp[x]))
                    lan["plates"][x].append(int(t.plates_broken[x]))
                    lan["deaths"][x].append(int(t.died[x]))
                    lan["recalls"][x].append(int(t.recalled[x]))
                    lan["farm"][x].append(round(t.farm_units[x] / sc, 5))
                    lan["farm_avail"][x].append(round(t.farm_available[x], 5))
                    played = not t.away[x] and t.behaviors[x] != B_RECALL and not t.warding[x]
                    lan["wave_turns"][x].append(int(played))
                    lan["shove"][x].append(int(played and t.waves[x] == W_SH))
                    lan["trade_conn"][x].append(int(t.connected[x]))
                    L0 = tb.level_by_xp[min(before.side(x).xp, tb.xp_cap)]
                    lan["cap"][x].append(int(ctx.capacity(L0, before.side(x).items)))
        rec["reward"] = [round(r, 6) for r in rew]
        if arm is not None:
            rec["lane"] = lan
        out.append(rec)
    data = {"final": s, "games": games, "turns": turns, "seed_root": root, "arm": arm or "baseline", "records": out}
    save_gz(path, data)
    return s, data


# ---------------------------------------------------------------------------------------------- the formulas
def M(S: float) -> float:
    return 1.0 + 1.5 * (1.0 - math.sqrt(16.0 / (16.0 + S)))


def dM(S: float) -> float:
    return 1.5 * 0.5 * math.sqrt(16.0) * (16.0 + S) ** -1.5


def elast(S: float) -> float:
    return S * dM(S) / M(S) if S > 0 else 0.0


def _elast_of(f, S: float, h: float = 1e-3) -> float:
    if S <= 0:
        return 0.0
    return (math.log(f(S * (1 + h))) - math.log(f(S * (1 - h)))) / (math.log(1 + h) - math.log(1 - h))


# ---------------------------------------------------------------------------------------------- analysis
def _arrays(data: Dict[int, dict]):
    """Stacked arrays [game, side, turn] over all finals."""
    keys = ("S", "g", "E", "L", "present", "returned", "bought", "hp")
    out = {k: [] for k in keys}
    out["final"] = []
    for s, d in sorted(data.items()):
        for rec in d["records"]:
            for k in keys:
                out[k].append(np.asarray(rec[k], dtype=float))
            out["final"].append(s)
    return {k: (np.stack(v) if k != "final" else np.asarray(v)) for k, v in out.items()}


def analyse(data: Dict[int, dict], Mfun=None) -> dict:
    """`Mfun`: the arm's M(S) (float resources -> float); the current curve by default."""
    Mv = np.vectorize(Mfun or M)
    A = _arrays(data)
    S, g, E, L, P = A["S"], A["g"], A["E"], A["L"], A["present"]
    n_games, _, T = S.shape
    turns = np.arange(1, T + 1)
    out = {}
    # ---- 1. trajectories by turn
    traj = {}
    for t in (10, 25, 50, 75, 100, 150, 200, 300, 400, 600):
        if t > T:
            continue
        i = t - 1
        both = S[:, :, i].ravel()
        traj[t] = {"S_median": float(np.median(both)), "S_q10": float(np.quantile(both, 0.1)),
                   "S_q90": float(np.quantile(both, 0.9)), "g_median": float(np.median(g[:, :, i])),
                   "E_median": float(np.median(E[:, :, i])), "L_median": float(np.median(L[:, :, i])),
                   "abs_dE_median": float(np.median(np.abs(E[:, 0, i] - E[:, 1, i]))),
                   "abs_dE_q90": float(np.quantile(np.abs(E[:, 0, i] - E[:, 1, i]), 0.9)),
                   "abs_dS_median": float(np.median(np.abs(S[:, 0, i] - S[:, 1, i]))),
                   "abs_dS_q90": float(np.quantile(np.abs(S[:, 0, i] - S[:, 1, i]), 0.9)),
                   "abs_dL_mean": float(np.mean(np.abs(L[:, 0, i] - L[:, 1, i]))),
                   "share_dL_ge1": float(np.mean(np.abs(L[:, 0, i] - L[:, 1, i]) >= 1)),
                   "rel_lead_median": float(np.median(np.abs(E[:, 0, i] - E[:, 1, i]) / np.maximum(E[:, :, i].mean(1), 1e-9)))}
    out["trajectory"] = traj
    # discount-weighted investment level (the objective weights turn t by 0.99^t)
    w = GAMMA ** (turns - 1)
    Sw = np.repeat(w[None, None, :], n_games * 2, axis=0).reshape(n_games, 2, T)
    vals, wts = S.ravel(), Sw.ravel()
    order = np.argsort(vals)
    cw = np.cumsum(wts[order]) / wts.sum()
    out["discount_weighted_S"] = {str(q): float(vals[order][np.searchsorted(cw, q)]) for q in (0.25, 0.5, 0.75, 0.9)}
    out["income_per_turn"] = {f"{a}-{b}": float(np.mean((E[:, :, b - 1] - E[:, :, a - 1]) / (b - a)))
                              for a, b in ((1, 50), (50, 150), (150, 300), (300, 600)) if b <= T}
    # ---- 2. decomposition of the ratio, both present, per stage
    dec = {}
    both = (P[:, 0, :] > 0) & (P[:, 1, :] > 0)
    for name, (a, b) in STAGES.items():
        if b > T:
            continue
        m = both[:, a - 1:b]
        sl = slice(a - 1, b)
        SA, SB, GA, GB, EA, EB = (S[:, 0, sl][m], S[:, 1, sl][m], g[:, 0, sl][m], g[:, 1, sl][m], E[:, 0, sl][m],
                                  E[:, 1, sl][m])
        LA, LB = L[:, 0, sl][m], L[:, 1, sl][m]
        lead = EA - EB                                    # signed earned lead of A
        sign = np.where(lead >= 0, 1.0, -1.0)             # orient every state to the gold leader
        mS = np.log(Mv(SA)) - np.log(Mv(SB))
        mE = np.log(Mv(EA)) - np.log(Mv(EB))
        rel = np.log(np.maximum(EA, 1e-6)) - np.log(np.maximum(EB, 1e-6))
        dL = LA - LB
        lvl = dL * math.log(1.05)
        rows = {"n": int(len(SA)), "S_mean": float(((SA + SB) / 2).mean()), "E_mean": float(((EA + EB) / 2).mean()),
                "abs_lead_mean": float(np.abs(lead).mean()), "abs_lead_q90": float(np.quantile(np.abs(lead), 0.9)),
                "elasticity_at_mean_S": (elast(float(((SA + SB) / 2).mean())) if Mfun is None else
                                         _elast_of(Mfun, float(((SA + SB) / 2).mean()))),
                # all oriented to the gold leader (sign), means of logs, reported as ratios
                "relative_gold_lead": float(np.exp((sign * rel).mean())),
                "M_ratio_if_all_earned_invested": float(np.exp((sign * mE).mean())),
                "M_ratio_actual": float(np.exp((sign * mS).mean())),
                "level_ratio_AP": float(np.exp((sign * lvl).mean())),
                "attack_ratio": float(np.exp((sign * (lvl + mS)).mean())),
                "m44_power_ratio": float(np.exp((sign * (2 * lvl + mS)).mean())),
                "duel_ratio": float(np.exp((sign * (2 * lvl + 2 * mS)).mean())),
                "share_leader_more_invested": float(np.mean(sign * (SA - SB) > 0)),
                "unspent_lead_mean": float((sign * (GA - GB)).mean()),
                "invested_lead_mean": float((sign * (SA - SB)).mean()),
                "abs_dL_mean": float(np.abs(dL).mean())}
        dec[name] = rows
    out["decomposition"] = dec
    # ---- 3. M4.6a's binning by |lead|, re-derived: which turns and investment levels the bins contain
    bins = {}
    for lo, hi in ((0, 3), (3, 6), (6, 10), (10, 20), (20, 1000)):
        lead = np.abs(E[:, 0, :] - E[:, 1, :])
        m = both & (lead >= lo) & (lead < hi)
        if not m.any():
            continue
        tt = np.broadcast_to(turns, lead.shape)[m]
        SA, SB, LA, LB = S[:, 0, :][m], S[:, 1, :][m], L[:, 0, :][m], L[:, 1, :][m]
        pr = np.exp(2 * (LA - LB) * math.log(1.05)) * Mv(SA) / Mv(SB)
        bins[f"{lo}-{hi}"] = {"n": int(m.sum()), "turn_median": float(np.median(tt)),
                              "S_median": float(np.median((SA + SB) / 2)),
                              "m44_power_ratio_mean": float(np.maximum(pr, 1 / pr).mean()),
                              "share_turns_after_150": float(np.mean(tt > 150))}
    out["lead_bins"] = bins
    # ---- 4. persistence and catch-up: does the earned lead grow, and how does the relative lead evolve?
    per = {}
    dE = E[:, 0, :] - E[:, 1, :]
    for t0, t1 in ((50, 100), (100, 200), (200, 400), (300, 600)):
        if t1 > T:
            continue
        x, y = dE[:, t0 - 1], dE[:, t1 - 1]
        slope = float(np.polyfit(x, y, 1)[0]) if x.std() > 0 else float("nan")
        per[f"{t0}->{t1}"] = {"slope_lead_on_lead": slope, "corr": float(np.corrcoef(x, y)[0, 1]),
                              "abs_lead_t0": float(np.abs(x).mean()), "abs_lead_t1": float(np.abs(y).mean()),
                              "rel_lead_t0": float(np.mean(np.abs(x) / E[:, :, t0 - 1].mean(1))),
                              "rel_lead_t1": float(np.mean(np.abs(y) / E[:, :, t1 - 1].mean(1)))}
    out["persistence"] = per
    # ---- 5. returns: when gold becomes power
    R = A["returned"]
    bought = A["bought"]
    rows = {}
    for name, (a, b) in STAGES.items():
        if b > T:
            continue
        sl = slice(a - 1, b)
        r = R[:, :, sl] > 0
        rows[name] = {"returns_per_100_turns": float(r.sum() / (n_games * 2) / (b - a + 1) * 100),
                      "bought_per_return": float(bought[:, :, sl][r].mean()) if r.any() else float("nan"),
                      "unspent_mean": float(g[:, :, sl].mean()), "unspent_q90": float(np.quantile(g[:, :, sl], 0.9))}
    out["returns"] = rows
    return out


def what_leads_buy(stage_S: Dict[str, float], stage_L: Dict[str, int]) -> dict:
    """For each stage's typical (S, level): what a lead of LEAD_TURNS turns of income buys under the current curve:
    the M ratio, the attack and duel ratios, the change in wave capacity C, in Pressure per turn, in the expected trade
    damage per connected exchange (bars of the opponent's HP), and the elasticity. A purchase of size dS by ONE side
    (the marginal value) is the same arithmetic seen from the purchaser."""
    from ....config import GameConfig
    from ....champions_m4 import context
    cfg = GameConfig.m45_calibrated()
    ctx = context(cfg)
    tb = ctx.t
    sc = cfg.gold_scale
    h, q = 0.55, 0.3
    out = {}
    for stage, S0 in stage_S.items():
        L = stage_L[stage]
        base_u = int(round(S0 * sc))
        rows = {}
        for k in LEAD_TURNS:
            d = k * INCOME
            u = int(round((S0 + d) * sc))
            mA, mB = ctx.inv.factor(u), ctx.inv.factor(base_u)
            cA, cB = ctx.capacity(L, u), ctx.capacity(L, base_u)
            # expected damage per connected exchange in the victim's bars, after the victim's durability
            dmg = lambda att, vic: h * ((1 - q) * ctx.hit(1, L, att) + q * ctx.hit(2, L, att)) / (tb.hp_max[L] * ctx.inv.factor(vic))
            rows[f"{k} turns ({d:.2f} res)"] = {
                "M_ratio": mA / mB, "attack_ratio": mA / mB, "duel_ratio": (mA / mB) ** 2,
                "capacity": [cB, cA], "pressure_per_turn": [ctx.pressure(L, base_u), ctx.pressure(L, u)],
                "exchange_bars": {"leader_on_trailer": dmg(u, base_u), "trailer_on_leader": dmg(base_u, u),
                                  "equal": dmg(base_u, base_u)}}
        out[stage] = {"S": S0, "level": L, "M": ctx.inv.factor(base_u), "elasticity": elast(S0), "dM_per_res": dM(S0),
                      "leads": rows}
    return out


def implementation_checks() -> dict:
    """Assumptions in the code that a different curve would touch (read from the implementation, verified here)."""
    from ....config import GameConfig
    from ....env.discounted import max_champion_wave_budget, per_turn_bound, tail_constant
    from ....progression import INVEST_MAX_RESOURCES, investment
    cfg = GameConfig.m45_calibrated()
    inv = investment(cfg)
    ap18 = inv.t.ap[cfg.level_cap][0] / 10_000
    return {"invest_obs_encoding": cfg.invest_obs_encoding, "invest_obs_ref": cfg.invest_obs_ref,
            "invest_obs_gold_ref": cfg.invest_obs_gold_ref,
            "observation_invested_entry": "n / (n + 64), n = whole resources invested: encodes S, not M",
            "observation_attack_power_entry": f"AP* / (AP* + AP(18, 0)), AP(18, 0) = {ap18:.4f}: encodes M; bounded in "
                                             "(0, 1) for any curve, but its relation to the S entry changes with "
                                             "the curve (a trained network sees new combinations)",
            "decode_exact_below_resources": "invested < 23,106, unspent < 8,184 (compressed encoding, float32)",
            "INVEST_MAX_RESOURCES": INVEST_MAX_RESOURCES,
            "M_sup_current": 1 + cfg.invest_gain, "max_wave_budget_current": max_champion_wave_budget(cfg),
            "per_turn_bound_current": per_turn_bound(cfg), "tail_constant_current": tail_constant(cfg),
            "notes": ["wave_hold_cap = 1: Last Hit is 1 HP at every power, so investment never changes Last Hit",
                      "Thin and Slow Push spend only what their window-strength target needs, capped at round(C / 2)",
                      "Shove spends C; a front holds at most 24 HP (+ queued packets), so C above about 24 adds "
                      "little except against stacked waves and crashed masses",
                      "Pressure adds round(400 AP*) centi-units per turn: plates at 1,600 / 3,600 / 6,000",
                      "minion damage on champions falls with level (x 1 / 1.08^(L-1)) and is divided by M(S)",
                      "the reward counts gold when EARNED; investment only changes future earning"]}


def run(workers: int) -> dict:
    t0 = time.perf_counter()
    data = dict(run_jobs(trajectories_job, [(s, GAMES) for s in FINALS], workers, "audit trajectories"))
    a = analyse(data)
    stage_S = {k: float(np.median([(r["S"][0][t] + r["S"][1][t]) / 2 for d in data.values() for r in d["records"]
                                    for t in range((b + c) // 2 - 1, (b + c) // 2)]))
               for k, (b, c) in STAGES.items()}
    stage_L = {k: int(np.median([r["L"][0][(b + c) // 2 - 1] for d in data.values() for r in d["records"]]))
               for k, (b, c) in STAGES.items()}
    a["stage_typical"] = {"S": stage_S, "level": stage_L}
    a["what_leads_buy"] = what_leads_buy(stage_S, stage_L)
    a["implementation"] = implementation_checks()
    a["curve_table"] = {str(S): {"M": M(S), "dM_per_res": dM(S), "elasticity": elast(S),
                                 "dlnM_per_10res": math.log(M(S + 10) / M(S)) if S >= 0 else None}
                        for S in (0, 2, 5, 10, 16, 25, 35, 48, 64, 80, 100, 130, 160, 200, 300, 500)}
    save_json("audit", {"meta": meta(stage="audit", games=GAMES, turns=TURNS, seed_root=AUDIT_ROOT), "analysis": a,
                        "elapsed_s": time.perf_counter() - t0})
    return a


__all__ = ["run", "analyse", "what_leads_buy", "implementation_checks", "M", "dM", "elast", "STAGES"]
