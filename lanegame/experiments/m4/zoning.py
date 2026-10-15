"""M4.2d-C: combat-driven zoning (docs/m4/m4_2/m4_zoning_report.md).

Controlled counterfactuals of the M4 resolver: from one fixed state, every pair of lane behaviors (Hold, Yield, Trade,
Safe; both waves Slow Push, no objective) is played for one turn and, as a stationary pair, for a short window of
`HORIZON` turns, under each `zoning_rule` ("presence" = the M4 default, "contact", "none"). Nothing is sampled:
farm, XP, positions and the lane are deterministic (u = None, so no trade hit, tower shot or gank happens on the path)
and every chance event enters as its exact expectation, computed turn by turn from the resolver's own probabilities:

    E[trade damage by s]  = h_s (1 - q_s) dmg(1, L_s, N_s) + h_s q_s dmg(2, L_s, N_s)
    E[tower damage to s]  = tower_p_s * tower_shot_damage * u(L_s)
    E[gank damage to s]   = gank_p_s * gank_damage_trials * p_e,s * u(L_s)

On a deterministic path no chance event changes the positions or the farm, so the path does not depend on the item
tiers; the power conditions (A holds 0, 1 or 2 tiers, B none) are therefore evaluated on one path per (state, rule,
pair) by recomputing the damage with A's tier (the driver verifies this invariance on the first turn of every pair).
The value to A of a pair is

    V = (earned_A - earned_B) + lambda * (D_B / u(L_B) - D_A / u(L_A))      resources,

earned = farm + plate gold + bounty, D = expected centi-HP taken, lambda = the value of one HP unit (1/6 of a bar;
M2's estimate 0.22-0.30, default 0.25). The 4 x 4 matrices are solved as zero-sum games (lanegame.matrix_game).
The expectation treats HP linearly: it ignores that cumulative damage beyond the bar is a death (flagged per row).

Two further stages: a diagnostic sensitivity with the connected trader's farm cost removed (trade_farm_factor 1.0;
not a proposal), and sampled HP windows (`hp_window`): MC_SEEDS seeds x MC_WINDOW turns with the engine's own chance
events, so that trades actually land, champions recall at 40% HP and die (common random numbers across conditions).

    python -m lanegame.experiments.m4.zoning          JSON + Markdown to results/m4_zoning/  (4,966 turns)
"""

from __future__ import annotations

import os
from fractions import Fraction
from typing import Dict, List

import numpy as np

from ...champions import AWAY, CHAMP_ABSENT, CHANNEL, FARM, N_CHANCE, RETREAT, TRADING, champion_legal_actions
from ...champions_m4 import M4State, collection_units, context, initial_m4, resolve_turn_m4
from ...config import (B_HOLD, B_RECALL, B_SAFE, B_TRADE, B_YIELD, NONE, SH, SP, GameConfig, champ_act,
                       exact_safe_base)
from ...matrix_game import solve_matrix_game
from ...mechanics import initial_state
from .common import RESULTS_DIR, meta

OUT = os.path.join(RESULTS_DIR, "m4_zoning")
A, B = 0, 1
TURNS = [0]
RULES = ("presence", "contact", "none")
POWERS = {"equal": 0, "moderate": 1, "large": 2}           # A's item tiers; B holds none
BEHAVIORS = (B_HOLD, B_YIELD, B_TRADE, B_SAFE)
BNAME = {B_HOLD: "Hold", B_YIELD: "Yield", B_TRADE: "Trade", B_SAFE: "Safe"}
LAMBDA = 0.25
LAMBDAS = (0.15, 0.25, 0.35)
HORIZON = 6


def cfg_of(rule: str, **kw) -> GameConfig:
    cfg = GameConfig.m4(zoning_rule=rule, **kw)
    cfg.validate()
    return cfg


def step(cfg: GameConfig, lane, m4: M4State, k: int, wa: int, ba: int, wb: int, bb: int):
    a, b = champ_act(wa, ba, NONE), champ_act(wb, bb, NONE)
    assert a in champion_legal_actions(lane, m4, A, cfg) and b in champion_legal_actions(lane, m4, B, cfg)
    TURNS[0] += 1
    return resolve_turn_m4(lane, m4, k, a, b, cfg, u=None)


# ---------------------------------------------------------------------- start states
def starts(cfg: GameConfig) -> Dict[str, tuple]:
    """neutral: 9 turns of mutual Slow Push + Hold from s0 (both champions at the lane, kills at the lane).
    forward: A shoves on turns 5-7, then both hold; at k = 10 A stands at the lane with a 6-HP front and B one cell
    in front of its tower (in trade contact). B's reached front there has 1 HP left (its farm spot is about to walk
    back); it is replaced by a 4-HP front, which makes the configuration stationary under mutual Hold (a controlled
    counterfactual, not a reached state). tower: the same line at k = 13, B in its own tower cell, A at its +1.
    Every state is then put on common ground: both champions at the larger of the two XP values with full HP, no
    unspent gold and no items (the power conditions set A's tiers)."""
    out = {}
    lane, m4 = initial_state(cfg), initial_m4(cfg)
    for k in range(9):
        t = step(cfg, lane, m4, k, SP, B_HOLD, SP, B_HOLD)
        lane, m4 = t.lane.state, t.champ
    out["neutral"] = (lane, m4, 9)
    lane, m4 = initial_state(cfg), initial_m4(cfg)
    for k in range(13):
        t = step(cfg, lane, m4, k, SH if 5 <= k < 8 else SP, B_HOLD, SP, B_HOLD)
        lane, m4 = t.lane.state, t.champ
        if k + 1 == 10:
            assert (lane.ax, lane.ah, lane.bx, lane.bh) == (0, 6, -1, 1) and (m4.a.pos, m4.b.pos) == (0, -1)
            out["forward"] = (lane._replace(bh=cfg.wave_hp), m4, 10)
    out["tower"] = (lane, m4, 13)
    tb = context(cfg).t
    for n, (lane, m4, k) in out.items():
        xp = max(m4.a.xp, m4.b.xp)
        hp = tb.hp_max[tb.level_by_xp[xp]]
        out[n] = (lane, M4State(m4.a._replace(xp=xp, hp=hp, gold=0, items=0),
                                m4.b._replace(xp=xp, hp=hp, gold=0, items=0)), k)
    return out


# ---------------------------------------------------------------------- accounting
def gold_breakdown(t, s: int, cfg: GameConfig) -> Dict[str, Fraction]:
    """Exact attribution (resources) of side s's farm opportunities this turn: collected + base (the contest base's
    last-hit inefficiency, 1 - 0.9, or 1 - 0.5 under the enemy tower) + reach (out of farm reach, away, channeling)
    + posture (Safe / retreat base below the contest base) + zone (the proximity / contact factor) + trade (the
    connected trader's trade_farm_factor). The parts sum to the available farm exactly."""
    q = lambda v: Fraction(repr(float(v)))
    f, T, scale = q(cfg.farm_per_hp), cfg.tower, cfg.gold_scale
    o = 1 - s
    mode, pos = t.modes[s], t.end[s]
    out = {k: Fraction(0) for k in ("available", "collected", "base", "reach", "posture", "zone", "trade")}
    for x, hp in t.lane.kills[s].items():
        cmax = q(cfg.collect_contest) if x < T else q(cfg.collect_contest_enemy_tower)
        out["available"] += f * hp
        out["base"] += f * hp * (1 - cmax)
        reach = cfg.retreat_farm_reach if mode == RETREAT else cfg.farm_reach
        if mode in (AWAY, CHANNEL) or abs(x - pos) > reach:
            out["reach"] += f * hp * cmax
            continue
        b = cmax if mode in (FARM, TRADING) else exact_safe_base(x, cfg)
        out["posture"] += f * hp * (cmax - b)
        got = Fraction(hp * collection_units(x, mode, pos, t.modes[o], t.end[o], cfg), scale)
        out["collected"] += got
        tf = q(cfg.trade_farm_factor) if mode == TRADING else Fraction(1)
        kept = got / (f * hp * b * tf) if b * tf > 0 else Fraction(0)      # 1 - zone
        out["zone"] += f * hp * b * (1 - kept) if b > 0 else Fraction(0)
        out["trade"] += f * hp * b * kept * (1 - tf)
    assert sum(v for k, v in out.items() if k != "available") == out["available"]
    assert out["collected"] == Fraction(t.farm_units[s], scale)
    return out


def expected_damage(t, m4: M4State, s: int, items_a: int, cfg: GameConfig) -> Dict[str, float]:
    """Expected centi-HP side s takes this turn (trade by the opponent, tower shot, gank) with A holding items_a."""
    ctx, tb = context(cfg), context(cfg).t
    o = 1 - s
    lv = t.levels_before
    it_o = items_a if o == A else m4.side(o).items
    trade = 0.0
    if t.connected[o]:
        h, qd = t.hit_p[o], t.double_p[o]
        trade = h * (1 - qd) * ctx.hit(1, lv[o], it_o) + h * qd * ctx.hit(2, lv[o], it_o)
    tower = t.tower_p[s] * cfg.tower_shot_damage * tb.hp_unit[lv[s]]
    gank = t.gank_p[s] * cfg.gank_damage_trials * t.gank_damage_p[s] * tb.hp_unit[lv[s]]
    return {"trade": trade, "tower": tower, "gank": gank, "unit": tb.hp_unit[lv[s]], "hp": m4.side(s).hp}


def run_pair(cfg: GameConfig, start, ba: int, bb: int, horizon: int = HORIZON) -> List[dict]:
    lane, m4, k0 = start
    rows = []
    for i in range(horizon):
        t = step(cfg, lane, m4, k0 + i, SP, ba, SP, bb)
        g = [gold_breakdown(t, s, cfg) for s in (A, B)]
        xp_av = [cfg.xp_per_hp * sum(t.lane.kills[s].values()) for s in (A, B)]
        row = {"end": t.end, "modes": t.modes, "connected": t.connected, "threatened": t.threatened,
               "earned": [t.earned_units[s] / cfg.gold_scale for s in (A, B)],
               "plates": t.plate_units, "gold": [{k: float(v) for k, v in g[s].items()} for s in (A, B)],
               "xp": [t.xp_gain[s] + t.xp_discarded[s] for s in (A, B)], "xp_available": xp_av,
               "dmg": {p: [expected_damage(t, m4, s, n, cfg) for s in (A, B)] for p, n in POWERS.items()}}
        rows.append(row)
        lane, m4 = t.lane.state, t.champ
    return rows


def summarize(rows: List[dict], h: int, power: str, lam: float = LAMBDA) -> dict:
    r = rows[:h]
    earned = [sum(x["earned"][s] for x in r) for s in (A, B)]
    dmg = [sum(x["dmg"][power][s]["trade"] + x["dmg"][power][s]["tower"] + x["dmg"][power][s]["gank"] for x in r)
           for s in (A, B)]
    trade = [sum(x["dmg"][power][s]["trade"] for x in r) for s in (A, B)]
    units = [sum((x["dmg"][power][s]["trade"] + x["dmg"][power][s]["tower"] + x["dmg"][power][s]["gank"])
                 / x["dmg"][power][s]["unit"] for x in r) for s in (A, B)]
    hp0 = [r[0]["dmg"][power][s]["hp"] for s in (A, B)]
    gold = {k: [sum(x["gold"][s][k] for x in r) for s in (A, B)]
            for k in ("available", "collected", "base", "reach", "posture", "zone", "trade")}
    return {"earned": earned, "dmg": dmg, "trade_dmg": trade, "dmg_units": units, "hp0": hp0,
            "lethal": [dmg[s] >= hp0[s] for s in (A, B)], "gold": gold,
            "xp": [sum(x["xp"][s] for x in r) for s in (A, B)],
            "xp_available": [sum(x["xp_available"][s] for x in r) for s in (A, B)],
            "connected": [sum(x["connected"][s] for x in r) for s in (A, B)],
            "threatened": [sum(x["threatened"][s] for x in r) for s in (A, B)],
            "value": earned[A] - earned[B] + lam * (units[B] - units[A])}


def matrix(paths: dict, h: int, power: str, lam: float = LAMBDA) -> np.ndarray:
    return np.array([[summarize(paths[(ba, bb)], h, power, lam)["value"] for bb in BEHAVIORS] for ba in BEHAVIORS])


def solve(M: np.ndarray) -> dict:
    sol = solve_matrix_game(M)
    j_vs = {BNAME[ba]: BNAME[BEHAVIORS[int(np.argmin(M[i]))]] for i, ba in enumerate(BEHAVIORS)}
    return {"value": float(sol.value), "x": [float(v) for v in sol.x], "y": [float(v) for v in sol.y],
            "b_best_response": j_vs}


def flip_multiplier(paths: dict, h: int, lam: float = LAMBDA, grid=np.arange(1.0, 4.001, 0.01)):
    """Linear-in-AP extrapolation from the equal-power path: the smallest A attack multiplier m at which B's best
    response to A's Trade is a conceding behavior (Yield or Safe) rather than a contesting one (Hold, Trade).
    A's trade damage scales by m (ignoring the centi-HP rounding); everything else is unchanged."""
    vals = {}
    for bb in BEHAVIORS:
        s = summarize(paths[(B_TRADE, bb)], h, "equal", lam)
        tr_units_b = sum(x["dmg"]["equal"][B]["trade"] / x["dmg"]["equal"][B]["unit"] for x in paths[(B_TRADE, bb)][:h])
        vals[bb] = (s["value"] - lam * tr_units_b, lam * tr_units_b)        # V(m) = a + m c
    for m in grid:
        best = min(BEHAVIORS, key=lambda bb: vals[bb][0] + m * vals[bb][1])
        if best in (B_YIELD, B_SAFE):
            return float(round(m, 2)), BNAME[best]
    return None, BNAME[min(BEHAVIORS, key=lambda bb: vals[bb][0] + grid[-1] * vals[bb][1])]


def verify_power_invariance(cfg: GameConfig, start) -> None:
    """The deterministic path does not depend on A's items (the premise of evaluating powers on one path)."""
    lane, m4, k = start
    for ba in BEHAVIORS:
        for bb in BEHAVIORS:
            t0 = step(cfg, lane, m4, k, SP, ba, SP, bb)
            t2 = step(cfg, lane, M4State(m4.a._replace(items=2), m4.b), k, SP, ba, SP, bb)
            assert (t0.end, t0.farm_units, t0.xp_gain, t0.lane.state, t0.hit_p) == \
                   (t2.end, t2.farm_units, t2.xp_gain, t2.lane.state, t2.hit_p)


# ------------------------------------------------------------------- HP consequences (sampled, common random numbers)
MC_SEEDS = 10
MC_WINDOW = 10
MC_RULES = ("presence", "none")
MC_STATES = ("neutral", "forward")
MC_LINES = (("Hold", "Contest"), ("Trade", "Contest"), ("Trade", "Concede"))
RECALL_FRAC = 0.4


def _policy(name: str):
    """Scripted lane behavior with the same retreat rule for every policy: at or below RECALL_FRAC of max HP a
    champion Recalls when legal and otherwise walks back (Safe). Hold / Contest = Hold, Trade = Trade, Concede =
    Yield (stands and farms like Hold unless threatened, then backs off)."""
    beh = {"Hold": B_HOLD, "Contest": B_HOLD, "Trade": B_TRADE, "Concede": B_YIELD}[name]

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


def hp_window(cfg: GameConfig, start, items_a: int, pa: str, pb: str, seed: int, lam: float = LAMBDA) -> dict:
    """One sampled window of MC_WINDOW turns with the engine's own chance events (deaths, recalls and bounties
    happen). Terminal value: earned_A - earned_B + lam * (bar_A - bar_B), bar = HP / u(L) of a present champion and
    a full bar (6 units) for one that is away (it returns with full HP; the absence turns left are not charged)."""
    lane, m4, k0 = start
    m4 = M4State(m4.a._replace(items=items_a), m4.b)
    rng = np.random.default_rng(seed)
    fa, fb = _policy(pa), _policy(pb)
    earned, xp, deaths, recalls, conn = [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]
    for i in range(MC_WINDOW):
        u = tuple(rng.random(N_CHANCE))
        t = resolve_turn_m4(lane, m4, k0 + i, fa(cfg, lane, m4, A), fb(cfg, lane, m4, B), cfg, u=u)
        TURNS[0] += 1
        for s in (A, B):
            earned[s] += t.earned_units[s]
            xp[s] += t.xp_gain[s]
            deaths[s] += t.died[s]
            recalls[s] += t.recalled[s]
            conn[s] += t.connected[s]
        lane, m4 = t.lane.state, t.champ
    tb = context(cfg).t
    bar = []
    for s in (A, B):
        me = m4.side(s)
        bar.append(cfg.champion_hp if me.absent else me.hp / tb.hp_unit[tb.level_by_xp[min(me.xp, tb.xp_cap)]])
    g = [e / cfg.gold_scale for e in earned]
    return {"earned": g, "xp": xp, "deaths": deaths, "recalls": recalls, "connected": conn, "bar": bar,
            "value": g[A] - g[B] + lam * (bar[A] - bar[B])}


def run_hp_windows(st: dict) -> dict:
    out = {}
    for rule in MC_RULES:
        cfg = cfg_of(rule)
        for n in MC_STATES:
            for p, items in POWERS.items():
                for pa, pb in MC_LINES:
                    rows = [hp_window(cfg, st[n], items, pa, pb, 1000 + sd) for sd in range(MC_SEEDS)]
                    v = [r["value"] for r in rows]
                    agg = {"value": float(np.mean(v)), "value_se": float(np.std(v, ddof=1) / np.sqrt(len(v)))}
                    for key in ("earned", "xp", "deaths", "recalls", "connected", "bar"):
                        agg[key] = [float(np.mean([r[key][s] for r in rows])) for s in (A, B)]
                    out[f"{rule}/{n}/{p}/{pa}-{pb}"] = agg
    return out


# ---------------------------------------------------------------------- driver
def run() -> dict:
    TURNS[0] = 0
    base = cfg_of("presence")
    st = starts(base)
    out = {"states": {n: {"lane": list(v[0]), "m4": v[1].to_list(), "k": v[2]} for n, v in st.items()},
           "paths": {}, "games": {}, "flip": {}}
    for n in st:
        verify_power_invariance(base, st[n])
    for rule in RULES:
        cfg = cfg_of(rule)
        for n, start in st.items():
            paths = {(ba, bb): run_pair(cfg, start, ba, bb) for ba in BEHAVIORS for bb in BEHAVIORS}
            out["paths"][f"{rule}/{n}"] = {f"{BNAME[a]}-{BNAME[b]}": {f"h{h}": {p: summarize(v, h, p)
                                                                              for p in POWERS}
                                                                       for h in (1, HORIZON)}
                                           for (a, b), v in paths.items()}
            for h in (1, HORIZON):
                for p in POWERS:
                    for lam in LAMBDAS:
                        out["games"][f"{rule}/{n}/h{h}/{p}/l{lam}"] = solve(matrix(paths, h, p, lam))
                out["flip"][f"{rule}/{n}/h{h}"] = flip_multiplier(paths, h)
    # sensitivity (diagnostic only, never a proposal by itself): which term blocks the concession? With the
    # connected trader's farm cost removed (trade_farm_factor 1.0) B's contesting no longer refunds A's farm
    out["tff1"] = {}
    for rule in ("presence", "none"):
        cfg = cfg_of(rule, trade_farm_factor=1.0)
        for n in ("neutral", "forward"):
            paths = {(ba, bb): run_pair(cfg, st[n], ba, bb) for ba in BEHAVIORS for bb in BEHAVIORS}
            for h in (1, HORIZON):
                for p in POWERS:
                    out["tff1"][f"{rule}/{n}/h{h}/{p}"] = solve(matrix(paths, h, p))
                out["tff1"][f"{rule}/{n}/h{h}/flip"] = flip_multiplier(paths, h)
    out["deterministic_turns"] = TURNS[0]
    out["hp_windows"] = run_hp_windows(st)
    out["turns"] = TURNS[0]
    return out


def markdown(res: dict) -> str:
    from ..common import md_table
    lines = [f"# M4.2d-C zoning counterfactuals\n\nSimulation turns: {res['turns']}\n"]
    for key, g in res["games"].items():
        if "/l0.25" not in key:
            continue
        lines.append(f"- `{key}`: value {g['value']:+.3f}; A mix " +
                     ", ".join(f"{BNAME[b]} {x:.2f}" for b, x in zip(BEHAVIORS, g["x"]) if x > 1e-6) + "; B mix " +
                     ", ".join(f"{BNAME[b]} {y:.2f}" for b, y in zip(BEHAVIORS, g["y"]) if y > 1e-6) +
                     f"; B best response to Trade: {g['b_best_response']['Trade']}, to Hold: "
                     f"{g['b_best_response']['Hold']}")
    lines.append("\n## HP consequences (sampled windows, mean over seeds)\n")
    lines.append(md_table(["key", "V", "se", "earned A", "earned B", "B deaths", "B recalls", "A conn", "bar A",
                           "bar B"],
                          [[k, v["value"], v["value_se"], v["earned"][0], v["earned"][1], v["deaths"][1],
                            v["recalls"][1], v["connected"][0], v["bar"][0], v["bar"][1]]
                           for k, v in res["hp_windows"].items()]))
    lines.append("\n## Flip multipliers (B's best response to A's Trade becomes Yield / Safe)\n")
    lines.append(md_table(["key", "m*", "B response"], [[k, v[0], v[1]] for k, v in res["flip"].items()]))
    return "\n".join(lines)


def main():
    res = run()
    res["meta"] = meta(GameConfig.m4())
    import json
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "zoning.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, separators=(",", ":"), default=str)
    md = markdown(res)
    with open(os.path.join(OUT, "zoning.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(md)
    print("turns", res["turns"])


if __name__ == "__main__":
    main()
