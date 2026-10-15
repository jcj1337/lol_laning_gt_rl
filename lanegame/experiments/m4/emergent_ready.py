"""M4.2d integration: joint feasibility checks of `GameConfig.m4_emergent()` (docs/m4/m4_2/m4_emergent_parallel_report.md).

Mechanical feasibility, not equilibrium play: short deterministic lines (u = None) and a few paired sampled windows
(common random numbers), reusing the workstream drivers (wave_readiness, positioning, zoning). Stages:

    wave      A. stack -> crash -> (no) bounce, counter-play by the defender, Recall timing (wave_readiness lines)
    pressure  B. a stronger champion holding forward without attacking against fixed and reactive opponents
    zoning    C. the M4.2d-C post-integration checklist: one-turn and 6-turn zero-sum matrices of the lane behaviors,
                 the concession flip multiplier m*, under the baseline and the integrated position / zoning rules
    hp        C. sampled HP windows (trades land, recalls, deaths): contest vs retreat at three investment gaps
    recall    D. a Recall that invests unspent gold and returns stronger, then trades; scripted env games

    python -m lanegame.experiments.m4.emergent_ready          JSON to results/m4_emergent_ready/
"""

from __future__ import annotations

import os
import time
from typing import Dict, List

import numpy as np

from ...champions import CHAMP_ABSENT, N_CHANCE, champion_legal_actions
from ...champions_m4 import M4State, context, resolve_turn_m4
from ...config import B_HOLD, B_RECALL, B_SAFE, B_TRADE, B_YIELD, NONE, SP, GameConfig, champ_act
from ...io import dump_json
from . import positioning as po
from . import wave_readiness as wr
from . import zoning as zo
from .common import RESULTS_DIR, meta

OUT = os.path.join(RESULTS_DIR, "m4_emergent_ready")
A, B = 0, 1
TURNS = [0]
# investment gaps (resources invested by A, by B) and the resulting attack multiplier ratio under the preset
GAPS = {"equal": (0, 0), "moderate": (16, 4), "large": (50, 0)}


def preset(**kw) -> GameConfig:
    cfg = GameConfig.m4_emergent(**kw)
    cfg.validate()
    return cfg


def with_invested(m4: M4State, cfg: GameConfig, sa: float, sb: float) -> M4State:
    g = cfg.gold_scale
    return M4State(m4.a._replace(items=int(sa * g)), m4.b._replace(items=int(sb * g)))


def multiplier_ratio(cfg: GameConfig, sa: float, sb: float) -> float:
    from ...progression import investment
    inv = investment(cfg)
    return inv.factor(int(sa * cfg.gold_scale)) / inv.factor(int(sb * cfg.gold_scale))


# ---------------------------------------------------------------------- A. wave behavior
WAVE_LINES = ("hold", "shove_farm", "shove_recall_small_crash", "hold_recall_large_crash", "shove_farm_vs_control",
              "neutral_hold", "neutral_recall", "neutral_recall_punished")


def stage_wave() -> dict:
    out = {}
    for name, cfg in (("m4", GameConfig.m4()), ("m4_emergent", preset())):
        st = wr.starts(cfg)
        out[name] = {}
        for line in WAVE_LINES:
            start, fa, pb, n = wr.LINES[line]
            rows = wr.play(cfg, fa(), pb, n, *st[start])
            s = wr.summarize(rows, cfg)
            s["max_front_hp_A"] = max(r["t"].lane.state.ah for r in rows)
            out[name][line] = s
    return out


# ---------------------------------------------------------------------- B. champion pressure
PRESSURE_LINES = {"A Hold vs B Hold": (po.fixed(B_HOLD), po.fixed(B_HOLD)),
                  "A Hold vs B Yield": (po.fixed(B_HOLD), po.fixed(B_YIELD)),
                  "A Hold vs B respect": (po.fixed(B_HOLD), po.respect),
                  "A Hold+Trade/4 vs B respect": (po.periodic_trade(4), po.respect),
                  "A Hold+Trade/4 vs B Yield": (po.periodic_trade(4), po.fixed(B_YIELD)),
                  "A Yield vs B Yield": (po.fixed(B_YIELD), po.fixed(B_YIELD))}


def stage_pressure() -> dict:
    out = {}
    for name, cfg in (("continuous_m2_presence", preset(position_rule="m2", zoning_rule="presence")),
                      ("m4_emergent", preset())):
        lane, m4, k = wr.end_state(wr.play(cfg, wr.hold, wr.hold, 9))
        out[name] = {}
        for gap in ("equal", "large"):
            m = with_invested(m4, cfg, *GAPS[gap])
            out[name][gap] = {ln: po.summarize(po.play(cfg, pa, pb, po.W, lane, m, k), cfg)
                              for ln, (pa, pb) in PRESSURE_LINES.items()}
    return out


# ---------------------------------------------------------------------- C. zoning checklist (deterministic)
ZONING_CFGS = {"baseline": dict(),
               "standoff_presence": dict(position_rule="standoff", farm_spot_rule="pre_crash_front"),
               "integrated": dict(position_rule="standoff", farm_spot_rule="pre_crash_front", zoning_rule="none")}


def stage_zoning() -> dict:
    """Two-tier power proxy (A x1.2 / x1.44 = the preset's multiplier at S = 6 / 16): the zoning driver's power
    conditions; the rules compared are the position and zoning rules, with everything else the M4 tiers game."""
    st = zo.starts(zo.cfg_of("presence"))
    out = {}
    for name, kw in ZONING_CFGS.items():
        cfg = GameConfig.m4(**kw)
        cfg.validate()
        out[name] = {}
        for n in ("neutral", "forward"):
            paths = {(ba, bb): zo.run_pair(cfg, st[n], ba, bb) for ba in zo.BEHAVIORS for bb in zo.BEHAVIORS}
            res = {"games": {}, "flip": {}}
            for h in (1, zo.HORIZON):
                for p in zo.POWERS:
                    res["games"][f"h{h}/{p}"] = zo.solve(zo.matrix(paths, h, p))
                res["flip"][f"h{h}"] = zo.flip_multiplier(paths, h)
            sm = {f"{zo.BNAME[a]}-{zo.BNAME[b]}": zo.summarize(paths[(a, b)], zo.HORIZON, "equal")
                  for a, b in paths}
            res["b_earned_h6"] = {k: round(v["earned"][B], 4) for k, v in sm.items()}
            res["a_earned_h6"] = {k: round(v["earned"][A], 4) for k, v in sm.items()}
            res["b_xp_h6"] = {k: v["xp"][B] for k, v in sm.items()}
            res["connected_h6"] = {k: v["connected"] for k, v in sm.items()}
            hh = sm["Hold-Hold"]["earned"][B]
            res["concede_ratio_safe"] = round(sm["Hold-Safe"]["earned"][B] / hh, 3) if hh else None
            res["concede_ratio_yield"] = round(sm["Hold-Yield"]["earned"][B] / hh, 3) if hh else None
            out[name][n] = res
    out["turns_note"] = "zoning.TURNS counts these"
    return out


# ---------------------------------------------------------------------- C. sampled HP windows
HP_SEEDS, HP_WINDOW, RECALL_FRAC = 8, 10, 0.4
HP_LINES = (("Hold", "Contest"), ("Trade", "Contest"), ("Trade", "Retreat"), ("Hold", "Retreat"), ("Trade", "Trade"))
POLICY_BEH = {"Hold": B_HOLD, "Contest": B_HOLD, "Trade": B_TRADE, "Retreat": B_SAFE, "Standoff": B_YIELD}


def _policy(name: str):
    beh = POLICY_BEH[name]

    def act(cfg, lane, m4, s):
        legal = champion_legal_actions(lane, m4, s, cfg)
        if legal == (CHAMP_ABSENT,):
            return CHAMP_ABSENT
        me, tb = m4.side(s), context(cfg).t
        if me.hp <= RECALL_FRAC * tb.hp_max[tb.level_by_xp[min(me.xp, tb.xp_cap)]]:
            c = champ_act(SP, B_RECALL)
            return c if c in legal else champ_act(SP, B_SAFE)
        return champ_act(SP, beh, NONE)
    return act


def window(cfg, lane, m4, k0, pa, pb, seed, turns=HP_WINDOW, lam=zo.LAMBDA) -> dict:
    """zoning.hp_window with free policies: earned_A - earned_B + lam (bar_A - bar_B), a full bar while away."""
    rng = np.random.default_rng(seed)
    earned, xp, deaths, conn = [0, 0], [0, 0], [0, 0], [0, 0]
    for i in range(turns):
        u = tuple(rng.random(N_CHANCE))
        t = resolve_turn_m4(lane, m4, k0 + i, pa(cfg, lane, m4, A), pb(cfg, lane, m4, B), cfg, u=u)
        TURNS[0] += 1
        for s in (A, B):
            earned[s] += t.earned_units[s]
            xp[s] += t.xp_gain[s]
            deaths[s] += t.died[s]
            conn[s] += t.connected[s]
        lane, m4 = t.lane.state, t.champ
    tb = context(cfg).t
    bar = [cfg.champion_hp if m4.side(s).absent else
           m4.side(s).hp / tb.hp_unit[tb.level_by_xp[min(m4.side(s).xp, tb.xp_cap)]] for s in (A, B)]
    g = [e / cfg.gold_scale for e in earned]
    return {"earned": g, "xp": xp, "deaths": deaths, "connected": conn, "bar": bar,
            "value": g[A] - g[B] + lam * (bar[A] - bar[B])}


def _agg(rows: List[dict]) -> dict:
    v = [r["value"] for r in rows]
    out = {"value": round(float(np.mean(v)), 3), "value_se": round(float(np.std(v, ddof=1) / np.sqrt(len(v))), 3)}
    for key in ("earned", "xp", "deaths", "connected", "bar"):
        out[key] = [round(float(np.mean([r[key][s] for r in rows])), 3) for s in (A, B)]
    return out


def stage_hp() -> dict:
    cfg = preset()
    st = zo.starts(zo.cfg_of("presence"))
    out = {"multiplier_ratio": {g: round(multiplier_ratio(cfg, *v), 3) for g, v in GAPS.items()}}
    for n in ("neutral", "forward"):
        lane, m4, k = st[n]
        for gap, (sa, sb) in GAPS.items():
            m = with_invested(m4, cfg, sa, sb)
            for pa, pb in HP_LINES:
                rows = [window(cfg, lane, m, k, _policy(pa), _policy(pb), 2000 + sd) for sd in range(HP_SEEDS)]
                out[f"{n}/{gap}/{pa}-{pb}"] = _agg(rows)
    return out


# ---------------------------------------------------------------------- D. recall and investment
def stage_recall() -> dict:
    """D1: from the neutral lane at equal investment, A holds `gold` unspent resources. "recall" = A walks back and
    Recalls now (invests on return), "stay" = A keeps holding; both then Trade on a contesting B for the rest of the
    window (common random numbers). D2: scripted env games gold_recaller vs hold_farmer under the preset."""
    cfg = preset()
    lane0, m40, k0 = wr.end_state(wr.play(cfg, wr.hold, wr.hold, 9))
    out = {}
    for gold in (4, 8, 16):
        m4 = M4State(m40.a._replace(gold=gold * cfg.gold_scale), m40.b)
        res = {}
        for plan in ("recall", "stay"):
            rows = []
            for sd in range(HP_SEEDS):
                rng = np.random.default_rng(3000 + sd)
                lane, m, earned, inv_at_trade = lane0, m4, [0, 0], None
                for i in range(16):
                    u = tuple(rng.random(N_CHANCE))
                    legal = champion_legal_actions(lane, m, A, cfg)
                    if legal == (CHAMP_ABSENT,):
                        a = CHAMP_ABSENT
                    elif plan == "recall" and i < 3 and m.a.gold > 0:
                        a = champ_act(SP, B_RECALL) if champ_act(SP, B_RECALL) in legal else champ_act(SP, B_SAFE)
                    elif i < 4:
                        a = champ_act(SP, B_HOLD)
                    else:
                        a = _policy("Trade")(cfg, lane, m, A)
                    b = _policy("Contest")(cfg, lane, m, B)
                    t = resolve_turn_m4(lane, m, k0 + i, a, b, cfg, u=u)
                    TURNS[0] += 1
                    if i == 4:
                        inv_at_trade = m.a.items / cfg.gold_scale
                    for s in (A, B):
                        earned[s] += t.earned_units[s]
                    lane, m = t.lane.state, t.champ
                tb = context(cfg).t
                bar = [cfg.champion_hp if m.side(s).absent else
                       m.side(s).hp / tb.hp_unit[tb.level_by_xp[min(m.side(s).xp, tb.xp_cap)]] for s in (A, B)]
                g = [e / cfg.gold_scale for e in earned]
                rows.append({"earned": g, "bar": bar, "invested_at_trade": inv_at_trade,
                             "value": g[A] - g[B] + zo.LAMBDA * (bar[A] - bar[B])})
            v = [r["value"] for r in rows]
            res[plan] = {"value": round(float(np.mean(v)), 3),
                         "value_se": round(float(np.std(v, ddof=1) / np.sqrt(len(v))), 3),
                         "invested_at_trade": rows[0]["invested_at_trade"],
                         "earned": [round(float(np.mean([r["earned"][s] for r in rows])), 3) for s in (A, B)]}
        out[f"gold{gold}"] = res
    # D2: scripted games
    from ...env import LaneEnv
    from ...env.agents_m4 import m4_suite
    from ...progression import investment
    suite, inv = m4_suite(), investment(cfg)
    games = {}
    for pa, pb in (("gold_recaller", "hold_farmer"), ("gold_recaller", "aggressive_trader")):
        for seed in (1, 2):
            env = LaneEnv(cfg, max_turns=300)
            obs = env.reset(seed=seed)
            rng = np.random.default_rng(seed)
            tot, deaths, recalls = 0, [0, 0], [0, 0]
            while True:
                r = env.step(suite[pa].act(obs[0], rng), suite[pb].act(obs[1], rng))
                TURNS[0] += 1
                tot += r.info["reward_units"]
                for s in (A, B):
                    deaths[s] += r.info["died"][s]
                    recalls[s] += r.info["recalled"][s]
                if r.done:
                    break
                obs = r.obs
            c = env.champ
            games[f"{pa}-{pb}/s{seed}"] = {
                "reward_A": round(tot / cfg.gold_scale, 3), "deaths": deaths, "recalls": recalls,
                "invested": [round(c.side(s).items / cfg.gold_scale, 2) for s in (A, B)],
                "multiplier": [round(inv.factor(c.side(s).items), 3) for s in (A, B)]}
    out["games"] = games
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    for c in (TURNS, wr.TURNS, zo.TURNS):
        c[0] = 0
    data = {"wave": stage_wave(), "pressure": stage_pressure(), "zoning": stage_zoning(), "hp": stage_hp(),
            "recall": stage_recall()}
    data["budget"] = {"turns": TURNS[0] + wr.TURNS[0] + zo.TURNS[0], "own": TURNS[0], "wave_readiness": wr.TURNS[0],
                      "zoning": zo.TURNS[0], "wall_s": round(time.time() - t0, 1)}
    dump_json({**meta(preset()), **data}, os.path.join(OUT, "emergent_ready.json"))
    print("budget", data["budget"])


if __name__ == "__main__":
    main()
