"""M4.3c quick wave-control screen (docs/m4/m4_3/m4_3c_wave_control_screen.md): Hold capped at 1 HP per turn, and an
adaptive Thin that only removes the enemy wave's excess, against the M4.2e-C share model.

Three configurations of the research preset (opt-in fields; `m4_lasthit()` itself is unchanged):

    A  baseline       GameConfig.m4_lasthit()                                   Hold round(C/8), Thin round(C/2)
    B  hold1          ... (wave_hold_cap=1)                                     Hold min(1, round(C/8)) = 1
    C  hold1_adaptive ... (wave_hold_cap=1, wave_thin_rule="adaptive")          Thin min(round(C/2), need)

need = mechanics.thin_need: the enemy blob's HP above the largest blob whose minions do not out-fire ours. Clear = C in
all three.

Deterministic scripted play through the M4 resolver with no chance event (u = None, no trades, so no HP is lost and
nobody recalls). Champion power is held FIXED over a window: each side's wave-clear capacity C is that of a (level,
invested resources) tier, computed once by the engine, and every wave slot's budget is taken from it through the
configuration's own budget model (`M4Context.budget_from_capacity`, then the adaptive cap inside `step_turn`). Levels
and XP still evolve for the gold and XP accounting. Starts are the M4.2e-B harness states (`zoning_natural.starts`):

    neutral    turn 9, both 8-HP fronts and both champions at the lane cell
    freeze     turn 51, B's 5-HP front held at A's -1 (a fresh B wave behind it, 6 HP of B's crashed under A's tower),
               A's 8-HP front in A's tower cell, A at -2, B at its +1
    imbalance  the neutral lane with B's front set to 12-24 HP (and A's to 4 or 8)

Stages (all deterministic; one run per cell):
    neutral    both Hold for 60 turns; A Thin or A Clear against B Hold
    freeze     A freezes ("threshold": the M4.2e-C controller, Thin while B's front >= 14 HP at A's -1; "thin": Thin
               every turn) against B walk_up / step_back / retreat (safe_line -2) / clear, 40 turns
    recovery   B concedes (step_back or retreat) for 20 turns, then walks up for 40; against walking up for 60
    imbalance  A Thins (or Holds or Clears) for 12 turns against a larger B wave; B Holds
    games      M4.3b base s0 self-play from s0 under real progression (levels and investment), 16 seeds x 160 turns:
               a screen of a FIXED policy trained under A, not an equilibrium of B or C

Coordinates in records and traces are GLOBAL (A's tower at -2, B's at +2).
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ...config import B_HOLD, B_SAFE, F, SH, SP, W_CLEAR, W_HOLD, W_THIN, GameConfig, champ_act
from .common import mean_ci, meta, save_json, write_md

CONFIGS = ("A", "B", "C")
CONFIG_LABEL = {"A": "baseline", "B": "hold1", "C": "hold1 + adaptive Thin"}
CONFIG_KW = {"A": {}, "B": {"wave_hold_cap": 1}, "C": {"wave_hold_cap": 1, "wave_thin_rule": "adaptive"}}
# power tiers: name -> ((level, invested resources) of A, of B); capacities are computed by the engine
TIERS = {"L3": ((3, 0), (3, 0)), "L3 A+": ((3, 50), (3, 0)),
         "L6": ((6, 0), (6, 0)), "L6 A+": ((6, 50), (6, 0)),
         "L10": ((10, 0), (10, 0)), "L10 A+": ((10, 50), (10, 0)),
         "L13 A+": ((13, 50), (13, 0)), "L18 A++": ((18, 10_000), (18, 0)),
         "L18 +64": ((18, 64), (18, 64))}
LAMBDA = 1 / 8
NEUTRAL_TURNS, FREEZE_TURNS, RECOVERY_TURNS, CONCEDE_TURNS, IMBALANCE_TURNS = 60, 40, 60, 20, 12
FREEZERS = ("threshold", "thin", "no_freeze")
RESPONSES = ("walk_up", "step_back", "retreat", "clear")
IMBALANCES = ((8, 12), (8, 16), (8, 20), (8, 24), (4, 20))      # (A's front, B's front) HP at the lane cell
IMBALANCE_CAPS = (4, 6, 11, 22)                                   # A's capacity: Thin 2, 3, 6, 11 under the share model
IMBALANCE_B_CAP = 6
RECOVERY_TIERS = ("L3", "L10", "L10 A+")
GAME_SPEC, GAME_EPISODES, GAME_TURNS, GAME_SEED = "final:base:0", 16, 160, 76_000   # new root (common.py: 71-75k)
TRACES = (("freeze", "L10", "A", "threshold", "retreat"), ("freeze", "L10", "C", "thin", "retreat"),
          ("freeze", "L13 A+", "A", "threshold", "walk_up"), ("freeze", "L13 A+", "B", "threshold", "walk_up"),
          ("freeze", "L13 A+", "C", "thin", "walk_up"),
          ("freeze", "L10", "C", "thin", "clear"),
          ("imbalance", "8v24", "A", 22, "thin"), ("imbalance", "8v24", "C", 4, "thin"),
          ("imbalance", "8v24", "C", 22, "thin"))
TRACE_TURNS = 16


def cfg_of(name: str, **kw) -> GameConfig:
    return GameConfig.m4_lasthit(**CONFIG_KW[name], **kw)


def capacities(tier: str) -> tuple:
    from ...champions_m4 import context
    cfg = GameConfig.m4_lasthit()
    ctx = context(cfg)
    return tuple(ctx.capacity(L, S * cfg.gold_scale) for L, S in TIERS[tier])


# ---------------------------------------------------------------------------------------------- lane helpers
def packets(lane, cfg) -> list:
    """The start-of-turn packet lists of mechanics.step_turn (own coordinates, leading packet first)."""
    P = [[], []]
    for s in (0, 1):
        fx, fh, rears = lane.side(s)[:3]
        if fh > 0:
            P[s].append([fx, fh])
        P[s].extend([x, cfg.wave_hp] for x in rears)
    return P


def lane_numbers(lane, cfg) -> dict:
    from ...mechanics import _blob_hp
    P = packets(lane, cfg)
    front = [[lane.ax, lane.ah] if lane.ah > 0 else None, [-lane.bx, lane.bh] if lane.bh > 0 else None]
    contact = lane.ax if (lane.ah > 0 and lane.bh > 0 and lane.ax + lane.bx >= -1) else None
    return {"front": front, "blob": [_blob_hp(P[s], cfg) for s in (0, 1)],
            "stock": [max(lane.side(s)[1], 0) + cfg.wave_hp * len(lane.side(s)[2]) + lane.side(s)[3] for s in (0, 1)],
            "siege": [lane.asg, lane.bsg], "contact": contact,
            "held": int(lane.bh > 0 and lane.bx >= 1 and lane.ah > 0 and lane.ax <= -1)}


def strip(lane, m4) -> str:
    """Global cells -4 .. +4: A's packets A<hp>, B's B<hp>, crashed masses [A<hp>] under the tower they siege,
    champions a / b."""
    cells = {x: [] for x in range(-4, 5)}
    if lane.ah > 0:
        cells[lane.ax].append(f"A{lane.ah}")
    for x in lane.ar:
        cells[x].append("A8")
    if lane.bh > 0:
        cells[-lane.bx].append(f"B{lane.bh}")
    for x in lane.br:
        cells[-x].append("B8")
    if lane.asg:
        cells[2].append(f"[A{lane.asg}]")
    if lane.bsg:
        cells[-2].append(f"[B{lane.bsg}]")
    if not m4.a.absent:
        cells[m4.a.pos].append("a")
    if not m4.b.absent:
        cells[-m4.b.pos].append("b")
    return "".join(f"{(' '.join(v) if v else '.'):<11}" for v in cells.values()).rstrip()


# ---------------------------------------------------------------------------------------------- driver
def play(cfg: GameConfig, start, caps, policy: Callable, turns: int, trace: int = 0) -> dict:
    """Scripted play from start = (lane, m4, k) with fixed capacities caps = (C_A, C_B); policy(i, lane, m4) ->
    (code_A, code_B) at turn index i. Returns per-turn series and the first `trace` turns as lane strips."""
    from ...champions_m4 import context, resolve_turn_m4
    from ...mechanics import step_turn, thin_need
    from ..m4.natural_waves import _legal
    ctx = context(cfg)
    lane, m4, k = start

    def lane_step(s, kk, x, y, reach, budget=None):
        b = (ctx.budget_from_capacity(x % 3, caps[0]), ctx.budget_from_capacity(y % 3, caps[1]))
        return step_turn(s, kk, x, y, cfg, champion_reach=reach, champion_budget=b)

    rows, strips = [], [] if trace else None
    if trace:
        strips.append(f"{'turn':<5}" + "".join(f"{x:+d}".ljust(11) for x in range(-4, 5)) + "  A/B damage")
        strips.append(f"{0:<5}{strip(lane, m4)}")
    for i in range(turns):
        a, b = policy(i, lane, m4)
        a, b = _legal(cfg, lane, m4, 0, a), _legal(cfg, lane, m4, 1, b)
        P = packets(lane, cfg)
        need = [thin_need(P, s, cfg) for s in (0, 1)]
        t = resolve_turn_m4(lane, m4, k, a, b, cfg, lane_step=lane_step, u=None)
        w = list(t.effective_waves)
        bud = [ctx.budget_from_capacity(w[s], caps[s]) for s in (0, 1)]
        eff = [min(bud[s], need[s]) if (w[s] == F and cfg.wave_thin_rule == "adaptive") else bud[s] for s in (0, 1)]
        ev = t.lane.events
        rec = {"w": w, "bud": bud, "need": need, "eff": eff,
               "dmg": [ev["champion_hp"][1], ev["champion_hp"][0]],
               "crash": list(t.lane.crash_hp), "gold": [x / cfg.gold_scale for x in t.earned_units],
               "xp": list(t.xp_gain), "end": [t.end[0], -t.end[1]], "away": list(t.away),
               **lane_numbers(t.lane.state, cfg)}
        rows.append(rec)
        lane, m4, k = t.lane.state, t.champ, k + 1
        if trace and i < trace:
            tag = "/".join(f"{'HCT'[w[s]]}{rec['dmg'][s]}" for s in (0, 1))
            strips.append(f"{i + 1:<5}{strip(lane, m4):<99}  {tag}")
    return {"rows": rows, "strips": strips}


def longest(flags: Sequence[int]) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def stats(rows: List[dict], start_held: int = 0) -> dict:
    """Window statistics of one run (A = side 0, B = side 1; values B - A where stated)."""
    s = lambda key, side: float(sum(r[key][side] for r in rows))
    held = [r["held"] for r in rows]
    contacts = [r["contact"] for r in rows if r["contact"] is not None]
    out = {"turns": len(rows), "held": int(sum(held)), "longest_hold": longest(held),
           "first_break": next((i + 1 for i, h in enumerate(held) if not h), None) if start_held else None,
           "gold": [s("gold", 0), s("gold", 1)], "xp": [s("xp", 0), s("xp", 1)],
           "crashes": [sum(1 for r in rows if r["crash"][x] > 0) for x in (0, 1)],
           "crash_hp": [s("crash", 0), s("crash", 1)],
           "dmg": [s("dmg", 0), s("dmg", 1)],
           "contact_turns": len(contacts), "contact_mean": float(np.mean(contacts)) if contacts else None,
           "final_contact": rows[-1]["contact"], "final_stock": rows[-1]["stock"],
           "mean_blob": [float(np.mean([r["blob"][x] for r in rows])) for x in (0, 1)],
           "mean_stock": [float(np.mean([r["stock"][x] for r in rows])) for x in (0, 1)]}
    for x, side in ((0, "a"), (1, "b")):
        for slot, name in ((SP, "hold"), (F, "thin"), (SH, "clear")):
            tr = [r for r in rows if r["w"][x] == slot]
            out[f"{side}_{name}_turns"] = len(tr)
            out[f"{side}_{name}_dmg"] = float(sum(r["dmg"][x] for r in tr))
            if slot == F:
                out[f"{side}_thin_zero"] = sum(1 for r in tr if r["eff"][x] == 0)
                out[f"{side}_thin_capped"] = sum(1 for r in tr if r["eff"][x] < r["bud"][x])
    out["gold_diff_b"] = out["gold"][1] - out["gold"][0]
    out["xp_diff_b"] = out["xp"][1] - out["xp"][0]
    out["value_b"] = out["gold_diff_b"] + LAMBDA * out["xp_diff_b"]
    return out


def series(rows: List[dict]) -> dict:
    """Compact per-turn series for the JSON: blobs, stocks, fronts, held, A's and B's champion damage, A's need."""
    return {k: [r[k] for r in rows] for k in ("blob", "stock", "front", "held", "dmg", "need", "eff", "w", "crash",
                                              "contact")}


# ---------------------------------------------------------------------------------------------- policies
def const(a: int, b: int) -> Callable:
    return lambda i, lane, m4: (a, b)


def freezer(cfg, kind: str) -> Callable:
    """A's wave: the M4.2e-C threshold controller (zoning_natural.wave_rule "freeze"), Thin every turn, or Clear every
    turn (no_freeze, the reference); Hold position."""
    from ..m4.zoning_natural import wave_rule
    if kind == "thin":
        return lambda lane, m4: champ_act(W_THIN, B_HOLD)
    if kind == "no_freeze":
        return lambda lane, m4: champ_act(W_CLEAR, B_HOLD)
    return lambda lane, m4: champ_act(wave_rule("freeze", cfg, lane, m4, 0), B_HOLD)


RESPONSE_CODE = {"walk_up": champ_act(W_HOLD, B_HOLD), "step_back": champ_act(W_HOLD, B_SAFE),
                 "retreat": champ_act(W_HOLD, B_SAFE), "clear": champ_act(W_CLEAR, B_HOLD)}


def response_cfg(cfg_name: str, response: str) -> GameConfig:
    """`retreat` is Safe with safe_line -2 (stay in the own tower cell), a configuration override of its runs only."""
    return cfg_of(cfg_name, **({"safe_line": -2} if response == "retreat" else {}))


# ---------------------------------------------------------------------------------------------- stages
def stage_neutral(start) -> dict:
    out = {}
    for tier in TIERS:
        caps = capacities(tier)
        for c in CONFIGS:
            cfg = cfg_of(c)
            for wa, name in ((W_HOLD, "hold"), (W_THIN, "thin"), (W_CLEAR, "clear")):
                r = play(cfg, start, caps, const(champ_act(wa, B_HOLD), champ_act(W_HOLD, B_HOLD)), NEUTRAL_TURNS)
                out[f"{tier}|{c}|{name}"] = {"caps": caps, "stats": stats(r["rows"]), "series": series(r["rows"])}
    return out


def stage_freeze(start) -> dict:
    out = {}
    for tier in TIERS:
        caps = capacities(tier)
        for c in CONFIGS:
            for fz in FREEZERS:
                for resp in RESPONSES:
                    cfg = response_cfg(c, resp)
                    fa, code_b = freezer(cfg, fz), RESPONSE_CODE[resp]
                    r = play(cfg, start, caps, lambda i, l, m: (fa(l, m), code_b), FREEZE_TURNS)
                    out[f"{tier}|{c}|{fz}|{resp}"] = {"caps": caps, "stats": stats(r["rows"], start_held=1),
                                                      "series": series(r["rows"])}
    return out


def stage_recovery(start) -> dict:
    out = {}
    for tier in RECOVERY_TIERS:
        caps = capacities(tier)
        for c in CONFIGS:
            for fz in ("threshold", "thin"):
                for line in ("walk_up", "step_back", "retreat"):
                    cfg = response_cfg(c, line)
                    fa = freezer(cfg, fz)
                    concede = RESPONSE_CODE[line]
                    up = RESPONSE_CODE["walk_up"]
                    pol = lambda i, l, m: (fa(l, m), concede if i < CONCEDE_TURNS else up)
                    rows = play(cfg, start, caps, pol, RECOVERY_TURNS)["rows"]
                    out[f"{tier}|{c}|{fz}|{line}"] = {"at_concede_end": stats(rows[:CONCEDE_TURNS], 1),
                                                      "at_end": stats(rows, 1),
                                                      "after": stats(rows[CONCEDE_TURNS:])}
    return out


def imbalance_start(start, ha: int, hb: int):
    lane, m4, k = start
    return lane._replace(ah=ha, bh=hb), m4, k


def stage_imbalance(start) -> dict:
    out = {}
    for ha, hb in IMBALANCES:
        st = imbalance_start(start, ha, hb)
        for cap in IMBALANCE_CAPS:
            caps = (cap, IMBALANCE_B_CAP)
            for c, actions in (("A", ("thin",)), ("C", ("thin", "hold", "clear"))):
                cfg = cfg_of(c)
                for name in actions:
                    wa = {"thin": W_THIN, "hold": W_HOLD, "clear": W_CLEAR}[name]
                    rows = play(cfg, st, caps, const(champ_act(wa, B_HOLD), champ_act(W_HOLD, B_HOLD)),
                                IMBALANCE_TURNS)["rows"]
                    over = sum(max(0, r["dmg"][0] - r["need"][0]) for r in rows) if name == "thin" else None
                    out[f"{ha}v{hb}|{cap}|{c}|{name}"] = {
                        "caps": caps, "need0": rows[0]["need"][0], "stats": stats(rows),
                        "dmg_first4": [r["dmg"][0] for r in rows[:4]], "need_first4": [r["need"][0] for r in rows[:4]],
                        "overshoot": over,
                        "contact_at": {str(n): rows[n - 1]["contact"] for n in (4, 8, 12)},
                        "series": series(rows)}
    return out


def traces(starts_: dict) -> dict:
    out = {}
    for sc, tier, c, x, resp in TRACES:
        if sc == "freeze":
            cfg = response_cfg(c, resp)
            fa, code_b = freezer(cfg, x), RESPONSE_CODE[resp]
            r = play(cfg, starts_["freeze"], capacities(tier), lambda i, l, m: (fa(l, m), code_b), TRACE_TURNS,
                     trace=TRACE_TURNS)
            key = f"freeze|{tier}|{c}|{x}|{resp}"
        else:
            ha, hb = (int(v) for v in tier.split("v"))
            cfg = cfg_of(c)
            r = play(cfg, imbalance_start(starts_["neutral"], ha, hb), (x, IMBALANCE_B_CAP),
                     const(champ_act(W_THIN, B_HOLD), champ_act(W_HOLD, B_HOLD)), TRACE_TURNS, trace=TRACE_TURNS)
            key = f"imbalance|{tier}|{c}|cap{x}|{resp}"
        out[key] = {"strips": r["strips"],
                    "rows": [{k: r_[k] for k in ("w", "bud", "need", "eff", "dmg", "blob", "stock", "held", "crash")}
                             for r_ in r["rows"]]}
    return out


# ---------------------------------------------------------------------------------------------- sensitivity
SENSITIVITY_TIERS = ("L3", "L10", "L10 A+", "L18 A++")


def turn_window_need(P, s: int, cfg: GameConfig) -> int:
    """Sensitivity variant of mechanics.thin_need (NOT an engine rule): the same parity target on the HP that fights
    during the turn, i.e. the leading packet plus every packet within ticks_per_turn + 1 cells behind it (a rear walks
    one cell per tick and merges at distance 1), so that a reinforcement arriving this turn counts."""
    def blob(pk):
        if not pk:
            return 0
        fx = pk[0][0]
        return pk[0][1] + sum(p[1] for p in pk[1:] if p[0] >= fx - cfg.ticks_per_turn - 1)
    he, ho = blob(P[1 - s]), blob(P[s])
    fo = max(1, (2 * ho) // cfg.wave_hp) if ho > 0 else 0
    target = (cfg.wave_hp * (fo + 1) - 1) // 2 if fo > 0 else 0
    return max(0, he - target)


def stage_sensitivity(starts_: dict) -> dict:
    """Configuration C with mechanics.thin_need swapped for turn_window_need (restored afterwards): the Thin-every-turn
    freezer against every response, and Thin against Hold from the neutral lane."""
    from ... import mechanics
    orig = mechanics.thin_need
    out = {}
    try:
        mechanics.thin_need = turn_window_need
        for tier in SENSITIVITY_TIERS:
            caps = capacities(tier)
            for resp in RESPONSES:
                cfg = response_cfg("C", resp)
                fa, code_b = freezer(cfg, "thin"), RESPONSE_CODE[resp]
                rows = play(cfg, starts_["freeze"], caps, lambda i, l, m: (fa(l, m), code_b), FREEZE_TURNS)["rows"]
                out[f"freeze|{tier}|{resp}"] = stats(rows, start_held=1)
            rows = play(cfg_of("C"), starts_["neutral"], caps,
                        const(champ_act(W_THIN, B_HOLD), champ_act(W_HOLD, B_HOLD)), NEUTRAL_TURNS)["rows"]
            out[f"neutral|{tier}|thin"] = stats(rows)
    finally:
        mechanics.thin_need = orig
    return out


# ---------------------------------------------------------------------------------------------- learned games
def _game_job(args) -> tuple:
    c, episodes, turns = args
    import torch
    from ...champions_m4 import context
    from ...env import LaneEnv
    from ...env.agents import episode_streams
    from ...env.discounted import game_agents
    from ..rl.m43b.common import agent_from_spec
    torch.set_num_threads(1)
    cfg = cfg_of(c)
    base = context(GameConfig.m4_lasthit())
    ctx = context(cfg)
    t_tab = ctx.t
    ag = agent_from_spec(GAME_SPEC, cfg)
    games = []
    for ep in range(episodes):
        _, env_ss, rng_a, rng_b = episode_streams(np.random.SeedSequence(GAME_SEED, spawn_key=(ep,)))
        agents = game_agents((ag, ag))
        env = LaneEnv(cfg, max_turns=turns, cache_size=20_000)
        obs = env.reset(seed=env_ss)
        acc = {"gold": [0.0, 0.0], "xp": [0, 0], "xp_100": [0, 0], "crashes": [0, 0], "crash_hp": [0, 0], "dmg": [0, 0],
               "hold": [0, 0], "thin": [0, 0], "clear": [0, 0], "hold_over1": [0, 0], "thin_dmg": [0, 0],
               "hold_dmg": [0, 0], "contact": [], "first_L16": [None, None], "first_L11": [None, None]}
        for turn in range(turns):
            a = agents[0].act(obs[0], rng_a)
            b = agents[1].act(obs[1], rng_b)
            r = env.step(a, b)
            i = r.info
            st = r.obs[0].state
            for s in (0, 1):
                acc["gold"][s] += i["earned_units"][s] / cfg.gold_scale
                acc["xp"][s] += i["xp_gain"][s]
                if turn < 100:
                    acc["xp_100"][s] += i["xp_gain"][s]
                acc["crashes"][s] += int(i["crash_hp"][s] > 0)
                acc["crash_hp"][s] += i["crash_hp"][s]
                d = i["events"]["champion_hp"][1 - s]
                acc["dmg"][s] += d
                w = i["effective_waves"][s]
                if not i["away"][s]:
                    acc[("hold", "clear", "thin")[w]][s] += 1
                    if w == SP:
                        acc["hold_dmg"][s] += d
                        side = i["champ_before"][s]
                        L = t_tab.level_by_xp[min(side[3], t_tab.xp_cap)]
                        acc["hold_over1"][s] += int(base.wave_budget(SP, L, side[4]) > 1)
                    elif w == F:
                        acc["thin_dmg"][s] += d
                for L, key in ((16, "first_L16"), (11, "first_L11")):
                    if acc[key][s] is None and i["levels"][s] >= L:
                        acc[key][s] = turn + 1
            if st.ah > 0 and st.bh > 0 and st.ax + st.bx >= -1:
                acc["contact"].append(st.ax)
            if r.done:
                break
            obs = r.obs
        acc["contact_mean"] = float(np.mean(acc["contact"])) if acc["contact"] else None
        acc["contact_off_lane"] = sum(1 for x in acc.pop("contact") if x != 0)
        games.append(acc)
    return c, games


def stage_games(workers: int) -> dict:
    from ..rl.m43b.common import run_jobs
    res = run_jobs(_game_job, [(c, GAME_EPISODES, GAME_TURNS) for c in CONFIGS], workers, "wave_control games")
    out = {}
    for c, games in res:
        summ = {}
        for key in ("gold", "xp", "xp_100", "crashes", "crash_hp", "dmg", "hold", "thin", "clear", "hold_over1", "thin_dmg",
                    "hold_dmg", "first_L16", "first_L11"):
            summ[key] = mean_ci([g[key][s] for g in games for s in (0, 1) if g[key][s] is not None])
        summ["contact_mean"] = mean_ci([g["contact_mean"] for g in games if g["contact_mean"] is not None])
        summ["contact_off_lane"] = mean_ci([g["contact_off_lane"] for g in games])
        summ["abs_gold_diff"] = mean_ci([abs(g["gold"][0] - g["gold"][1]) for g in games])
        out[c] = {"summary": summ, "games": games}
    return out


# ---------------------------------------------------------------------------------------------- run
def run(workers: int = 3) -> dict:
    from ..m4.zoning_natural import starts
    t0 = time.perf_counter()
    st = starts(GameConfig.m4_lasthit())
    out = {"meta": meta(stage="wave_control", configs={c: CONFIG_KW[c] for c in CONFIGS},
                        tiers={k: {"levels_invested": v, "capacities": capacities(k)} for k, v in TIERS.items()},
                        starts={k: st[k][0].to_list() for k in ("neutral", "freeze")}, lam=LAMBDA,
                        game_spec=GAME_SPEC, game_episodes=GAME_EPISODES, game_turns=GAME_TURNS,
                        game_seed_root=GAME_SEED)}
    out["neutral"] = stage_neutral(st["neutral"])
    out["freeze"] = stage_freeze(st["freeze"])
    out["recovery"] = stage_recovery(st["freeze"])
    out["imbalance"] = stage_imbalance(st["neutral"])
    out["traces"] = traces(st)
    out["sensitivity"] = stage_sensitivity(st)
    print(f"  deterministic stages {time.perf_counter() - t0:.1f} s", flush=True)
    out["games"] = stage_games(workers)
    out["meta"]["elapsed_s"] = time.perf_counter() - t0
    save_json("wave_control", out)
    return out


# ---------------------------------------------------------------------------------------------- tables
def _f(x, nd: int = 2, sign: bool = True) -> str:
    if x is None:
        return "–"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def _freeze_cell(s: dict) -> str:
    fb = s["first_break"] if s["first_break"] is not None else ">40"
    return (f"{s['held']}/{s['longest_hold']}/{fb} · {s['crashes'][0]}/{s['crashes'][1]} · "
            f"{_f(s['gold_diff_b'])} / {_f(s['value_b'])}")


def report(out: dict) -> str:
    from ...champions_m4 import context
    L = ["# M4.3c wave-control screen: tables", "",
         "Generated by `python -m lanegame.experiments.m43c wave_control` (lanegame/experiments/m43c/wave_control.py). "
         "Configurations: A = `m4_lasthit()`, B = A + `wave_hold_cap=1`, C = B + `wave_thin_rule=\"adaptive\"`. "
         "Deterministic, no chance events; capacities fixed per tier. Global cells; B's view for values (B − A). "
         "Value = gold + XP/8.", ""]
    # budgets
    L += ["## Tiers and per-turn budgets (Hold / Thin / Clear)", "",
          "| tier | A (level, invested) | B (level, invested) | C_A / C_B | A's budgets, config A | config B, C (C: Thin is a cap) |",
          "|---|---|---|---|---|---|"]
    for tier, v in out["meta"]["tiers"].items():
        ca, cb = v["capacities"]
        bud = {c: [context(cfg_of(c)).budget_from_capacity(w, ca) for w in (SP, F, SH)] for c in ("A", "B")}
        L.append(f"| {tier} | {tuple(v['levels_invested'][0])} | {tuple(v['levels_invested'][1])} | {ca} / {cb} | "
                 f"{'/'.join(map(str, bud['A']))} | {'/'.join(map(str, bud['B']))} |")
    # neutral
    L += ["", "## Scenario A: neutral lane, 60 turns (A's action against B Hold)", "",
          "Cell: A crashes / B crashes · mean contact cell (turns in contact) · A's champion damage per turn · gold B − A.", ""]
    for act in ("hold", "thin", "clear"):
        L += [f"**A {act}**", "", "| tier | A baseline | B hold1 | C hold1 + adaptive |", "|---|---|---|---|"]
        for tier in TIERS:
            cells = []
            for c in CONFIGS:
                s = out["neutral"][f"{tier}|{c}|{act}"]["stats"]
                cm = "–" if s["contact_mean"] is None else f"{s['contact_mean']:+.2f}"
                cells.append(f"{s['crashes'][0]}/{s['crashes'][1]} · {cm} ({s['contact_turns']}) · "
                             f"{s['dmg'][0] / s['turns']:.2f} · {_f(s['gold_diff_b'])}")
            L.append(f"| {tier} | " + " | ".join(cells) + " |")
        L.append("")
    # freeze
    L += ["## Scenarios B and D: established freeze, 40 turns", "",
          "Cell: held turns / longest hold / first turn not held · A crashes / B crashes · B − A gold / gold + XP/8.", ""]
    for fz in FREEZERS:
        L += [f"**A = {fz}**", "", "| tier | B's response | A baseline | B hold1 | C hold1 + adaptive |", "|---|---|---|---|---|"]
        for tier in TIERS:
            for resp in RESPONSES:
                cells = [_freeze_cell(out["freeze"][f"{tier}|{c}|{fz}|{resp}"]["stats"]) for c in CONFIGS]
                L.append(f"| {tier} | {resp} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["**Champion damage applied over the 40 turns (A / B) and A's Thin turns that dealt 0 / were capped**", "",
          "| tier | freezer | B's response | A baseline | C hold1 + adaptive |", "|---|---|---|---|---|"]
    for tier in ("L10", "L10 A+", "L13 A+", "L18 A++"):
        for fz in ("threshold", "thin"):
            for resp in RESPONSES:
                cells = []
                for c in ("A", "C"):
                    s = out["freeze"][f"{tier}|{c}|{fz}|{resp}"]["stats"]
                    cells.append(f"{s['dmg'][0]:.0f} / {s['dmg'][1]:.0f} · Thin {s['a_thin_turns']} "
                                 f"({s.get('a_thin_zero', 0)} zero, {s.get('a_thin_capped', 0)} capped)")
                L.append(f"| {tier} | {fz} | {resp} | " + " | ".join(cells) + " |")
    L += ["", "**Wave HP over time: A's blob / B's blob at the end of turns 1, 5, 10, 15, 20, 30, 40**", "",
          "| tier | config | freezer | B's response | blobs |", "|---|---|---|---|---|"]
    for tier in ("L10", "L13 A+"):
        for c, fz in (("A", "threshold"), ("B", "threshold"), ("A", "thin"), ("C", "thin")):
            for resp in ("walk_up", "retreat"):
                ser = out["freeze"][f"{tier}|{c}|{fz}|{resp}"]["series"]
                pts = " · ".join(f"{ser['blob'][t - 1][0]}/{ser['blob'][t - 1][1]}" for t in (1, 5, 10, 15, 20, 30, 40))
                L.append(f"| {tier} | {c} | {fz} | {resp} | {pts} |")
    # imbalance
    L += ["", "## Scenario C: imbalance correction, 12 turns (A Thins, B Holds at C = 6)", "",
          "Cell: A's damage turns 1–4 [need at the start of turns 1–4] · damage above need over 12 turns · contact "
          "cell after turn 4 / 8 / 12 (– = no contact) · crashes A/B.", "",
          "| A front v B front | C_A | baseline Thin | adaptive Thin | C: Hold | C: Clear |", "|---|---|---|---|---|---|"]
    for ha, hb in IMBALANCES:
        for cap in IMBALANCE_CAPS:
            cells = []
            for c, name in (("A", "thin"), ("C", "thin"), ("C", "hold"), ("C", "clear")):
                v = out["imbalance"][f"{ha}v{hb}|{cap}|{c}|{name}"]
                s = v["stats"]
                ct = "/".join("–" if x is None else f"{x:+d}" for x in v["contact_at"].values())
                over = "" if v["overshoot"] is None else f" · over {v['overshoot']}"
                dmg, need = ",".join(map(str, v["dmg_first4"])), ",".join(map(str, v["need_first4"]))
                cells.append(f"{dmg} [{need}]{over} · {ct} · {s['crashes'][0]}/{s['crashes'][1]}")
            L.append(f"| {ha} v {hb} | {cap} | " + " | ".join(cells) + " |")
    # recovery
    L += ["", "## Recovery: B concedes for 20 turns, then walks up for 40", "",
          "Cell: the conceding line minus the walk-up-throughout line, B − A gold and B − A XP, at turn 20 → turn 60 "
          "(negative = conceding cost B). The walk-up line itself: B − A gold + XP/8 at 20 → 60.", "",
          "| tier | config | freezer | walk-up line | step back: gold, XP | retreat: gold, XP |", "|---|---|---|---|---|---|"]
    for tier in RECOVERY_TIERS:
        for c in CONFIGS:
            for fz in ("threshold", "thin"):
                rv = {ln: out["recovery"][f"{tier}|{c}|{fz}|{ln}"] for ln in ("walk_up", "step_back", "retreat")}
                w = rv["walk_up"]
                cells = [f"{_f(w['at_concede_end']['value_b'])} → {_f(w['at_end']['value_b'])}"]
                for ln in ("step_back", "retreat"):
                    d = {(t, k): rv[ln][t][k] - w[t][k] for t in ("at_concede_end", "at_end")
                         for k in ("gold_diff_b", "xp_diff_b")}
                    cells.append(f"{_f(d['at_concede_end', 'gold_diff_b'])} → {_f(d['at_end', 'gold_diff_b'])}, "
                                 f"{d['at_concede_end', 'xp_diff_b']:+.0f} → {d['at_end', 'xp_diff_b']:+.0f}")
                L.append(f"| {tier} | {c} | {fz} | " + " | ".join(cells) + " |")
    # sensitivity
    if "sensitivity" in out:
        L += ["", "## Sensitivity: the parity target on the HP that fights this turn (configuration C, A Thins every turn)",
              "", "Cell: held / longest · A crashes / B crashes · B − A gold + XP/8; engine estimate → turn-window estimate.",
              "", "| tier | B's response | engine (start-of-turn blobs) | turn window |", "|---|---|---|---|"]
        for tier in SENSITIVITY_TIERS:
            for resp in RESPONSES:
                a = out["freeze"][f"{tier}|C|thin|{resp}"]["stats"]
                b = out["sensitivity"][f"freeze|{tier}|{resp}"]
                cell = lambda s: (f"{s['held']}/{s['longest_hold']} · {s['crashes'][0]}/{s['crashes'][1]} · "
                                  f"{_f(s['value_b'])}")
                L.append(f"| {tier} | {resp} | {cell(a)} | {cell(b)} |")
            a = out["neutral"][f"{tier}|C|thin"]["stats"]
            b = out["sensitivity"][f"neutral|{tier}|thin"]
            cell = lambda s: (f"neutral Thin v Hold: {s['crashes'][0]}/{s['crashes'][1]} crashes, contact "
                              f"{_f(s['contact_mean'])}")
            L.append(f"| {tier} | (neutral) | {cell(a)} | {cell(b)} |")
    # games
    if "games" in out:
        L += ["", f"## Learned self-play under real progression ({GAME_SPEC}, {GAME_EPISODES} seeds x {GAME_TURNS} turns)", "",
              "Per side, mean ± 95% half-width over seeds and seats.", "",
              "| quantity | A baseline | B hold1 | C hold1 + adaptive |", "|---|---|---|---|"]
        rows = (("gold", "gold"), ("xp_100", "XP by turn 100"), ("crashes", "crashes"), ("crash_hp", "crashed HP"),
                ("dmg", "champion damage on minions"), ("hold", "Hold turns"), ("hold_over1", "Hold turns with baseline Hold ≥ 2"),
                ("hold_dmg", "damage on Hold turns"), ("thin", "Thin turns"), ("thin_dmg", "damage on Thin turns"),
                ("clear", "Clear turns"), ("first_L11", "turn of level 11"), ("first_L16", "turn of level 16"),
                ("contact_mean", "mean contact cell (per game)"), ("contact_off_lane", "turns with contact off the lane cell"),
                ("abs_gold_diff", "|gold A − gold B| per game"))
        for key, name in rows:
            cells = []
            for c in CONFIGS:
                m, h = out["games"][c]["summary"][key]
                cells.append(f"{m:.2f} ± {h:.2f}")
            L.append(f"| {name} | " + " | ".join(cells) + " |")
    # traces
    L += ["", "## Traces", "", "Lane strips after each turn (global cells). `A12` = an A packet of 12 HP, `[B6]` = B's "
          "crashed mass under A's tower, `a` / `b` = champions. Right: each side's wave action and the minion HP its "
          "champion removed (H = Hold, C = Clear, T = Thin).", ""]
    for key, v in out["traces"].items():
        L += [f"**{key}**", "", "```"] + v["strips"] + ["```", ""]
    md = "\n".join(L)
    write_md("wave_control", md)
    return md


__all__ = ["run", "play", "stats", "capacities", "cfg_of", "turn_window_need", "CONFIGS", "TIERS", "report"]
