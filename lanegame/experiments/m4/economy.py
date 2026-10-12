"""S4: economic sanity (design §8.5, §15 scenarios 1, 2, 5, 6; audit §2-§4; plan §7 S4).

    python -m lanegame.experiments.m4.economy [--rollouts 60] [--window 1000] [--procs N]

All comparisons are PAIRED counterfactuals: rollout m of every variant uses the same seeds (environment chance and
both agents' streams; lanegame.diagnostics.rollout_seed), so variants see identical chance vectors turn by turn.
Values are discounted over W turns with the game's gamma (W = 1000: the guaranteed truncation error of each value is
<= gamma^W Omega = 0.003). Reported: paired mean difference, its standard error and 95% interval, and the share of
rollouts with a positive difference. Continuations are scripted (approximate champion-game measurements, not
equilibrium values).

  A  item value: a free tier (N + 1 vs N, same gold) and buying now vs at the next return (N + 1 with gold - price
     vs N with gold >= price), at snapshots of mutual edge-trader play
  B  Recall now vs Hold now (then the continuation), at HP 2, 4, 6 sixths of max HP, with and without an affordable
     tier, in recall-cheap (own wave pushing) and recall-expensive (enemy wave at own -1) witnessed states
  C  under a held freeze: the opponent standing forward (+1, Hold) vs back (-1, Safe), against a holding and a
     trading freezer
  D  Recall vs death as a route to the shop (from own -1 and from +1 with a walk-back turn)
  E  gold and XP denial by an equal and by a strong (+2 levels, +1 tier) freezer that advances and trades
  F  plates: plate gold per game by attacker, the share earned while the earner was away (crash-then-recall)
  G  snowballing: P(final discounted margin > 0 | margin after 30 turns), level gaps over time, mirror edge traders
  H  dominance: the scripted M4 suite round robin, protocol E1 from s0, W = 1000, both seats, common random numbers
"""

from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np

from ...champions_m4 import M4State, level
from ...config import B_HOLD, B_RECALL, B_SAFE, B_TRADE, CHAMP_ABSENT, F, NONE, PRESSURE, SH, SP, GameConfig, champ_act
from ...diagnostics import Commit, DEFER, FollowPolicy, FunctionPlan, ProbCache, Snapshot, rollout, rollout_seed
from ...env import LaneEnv, play_episode
from ...env.agents_m4 import M4ScriptAgent, gate_attackers, gate_defenders, m4_suite
from ...progression import tables
from ...scenarios_m4 import load_witnesses, replay
from .common import Timer, Z95, md_table, mean_se, meta, pool_map, save

RECALL = champ_act(SP, B_RECALL)
HOLD = champ_act(SP, B_HOLD)
SAFE = champ_act(SP, B_SAFE)


def agents() -> Dict[str, object]:
    a = dict(m4_suite())
    a.update({"edge_trader": M4ScriptAgent("edge_trader", SP, B_HOLD, PRESSURE, edge_trade=True),
              "freeze_hold": gate_defenders()["freeze"],
              "freeze_trade": M4ScriptAgent("freeze_trade", SP, B_TRADE, freeze=True),
              "strong_freezer": M4ScriptAgent("strong_freezer", SP, B_TRADE, freeze=True, recall_frac=1 / 6),
              "forward": M4ScriptAgent("forward", SP, B_HOLD, NONE),
              "back": M4ScriptAgent("back", SP, B_SAFE, NONE)})
    a.update({"g_" + k: v for k, v in gate_attackers().items()})
    return a


def _plan(spec: str):
    if spec == "follow":
        return FollowPolicy()
    if spec == "recall":
        return Commit(RECALL, "recall")
    if spec == "hold":
        return Commit(HOLD, "hold")
    if spec == "walk_back_recall":
        def fn(obs, t):
            if t == 0:
                return SAFE if SAFE in obs.legal else None
            if t == 1:
                return RECALL if RECALL in obs.legal else None
            return None
        return FunctionPlan("walk_back_recall", fn)
    raise ValueError(spec)


def run_job(job):
    """Paired rollouts of one (snapshot, plan) variant; returns the discounted returns per rollout and counters."""
    cfg = GameConfig.from_dict(job["cfg"])
    snap = Snapshot.from_dict(job["snap"])
    ag = agents()
    pair = (ag[job["agents"][0]], ag[job["agents"][1]])
    env = LaneEnv(snap.config(cfg), cache_size=50_000)
    cache = ProbCache()
    rets, extra = [], []
    for m in range(job["n"]):
        r = rollout(env, snap, job["focal"], _plan(job["plan"]), pair, rollout_seed(job["root"], m, *job["key"]),
                    cache, max_turns=job["window"])
        rets.append(r.ret - job.get("offset", 0.0))
        extra.append({"xp": r.m4["xp"], "farm": list(r.farm), "plates": list(r.plates), "deaths": list(r.deaths),
                      "purchases": r.m4["purchases"]})
    return job["id"], np.array(rets), extra


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    d = a - b
    m, se = mean_se(d)
    return {"diff": m, "se": se, "ci95": [m - Z95 * se, m + Z95 * se], "p_positive": float(np.mean(d > 0)),
            "significant": bool(abs(m) > Z95 * se) if np.isfinite(se) else False, "a": float(a.mean()),
            "b": float(b.mean())}


# ---------------------------------------------------------------------- snapshots
def edge_snapshots(cfg: GameConfig, turns=(20, 40, 80, 150), games=3) -> List[Snapshot]:
    ag = agents()
    env = LaneEnv(cfg, max_turns=max(turns) + 1, cache_size=20_000)
    out = []
    for g in range(games):
        log = play_episode(env, ag["edge_trader"], ag["edge_trader"], 31_000 + g, record=True)
        for k in turns:
            out.append(Snapshot.from_info(log.turns[k], (cfg.skill_a, cfg.skill_b)))
    return out


def witness_snapshot(name: str) -> Snapshot:
    env = replay(name, load_witnesses()[name])
    return Snapshot(env.state, env.champ, env.k, (env.cfg.skill_a, env.cfg.skill_b))


def recall_states(cfg: GameConfig) -> Dict[str, Snapshot]:
    """Recall-legal states: the witnessed recall-cheap / recall-expensive situations plus snapshots of edge-trader
    play in which A stands at or behind the recall line."""
    out = {"cheap": witness_snapshot("6a_recall_cheap"), "expensive": witness_snapshot("6b_recall_expensive")}
    ag = agents()
    env = LaneEnv(cfg, max_turns=200, cache_size=20_000)
    found = 0
    for g in range(20):
        log = play_episode(env, ag["edge_trader"], ag["edge_trader"], 32_000 + g, record=True)
        for i in log.turns[20::23]:
            c = M4State.from_list(i["champ_before"])
            if c.a.absent == 0 and c.a.pos <= cfg.recall_line and c.b.absent == 0:
                out[f"play{found}"] = Snapshot.from_info(i, (cfg.skill_a, cfg.skill_b))
                found += 1
                break
        if found >= 4:
            break
    return out


def _stable(name: str) -> int:
    """A process-independent key for seeds (Python's str hash is salted per process)."""
    import zlib
    return zlib.crc32(name.encode()) % 100_000


def with_a(snap: Snapshot, **kw) -> Snapshot:
    return snap.with_side(0, **kw)


# ---------------------------------------------------------------------- the jobs
def build_jobs(cfg: GameConfig, n: int, window: int) -> Dict[str, dict]:
    t = tables(cfg)
    jobs: Dict[str, dict] = {}
    base = {"cfg": cfg.to_dict(), "n": n, "window": window, "focal": 0}

    def add(jid, snap, plan, agents_, key, root, **kw):
        jobs[jid] = {**base, "id": jid, "snap": snap.to_dict(), "plan": plan, "agents": agents_, "key": key,
                     "root": root, **kw}

    # A: item value at edge-trader snapshots
    for i, s in enumerate(edge_snapshots(cfg)):
        a = s.champ.a
        if a.absent > 0 or a.items >= t.item_tiers:
            continue
        add(f"A|free|{i}|N", s, "follow", ("edge_trader", "edge_trader"), (1, i), 501)
        add(f"A|free|{i}|N+1", with_a(s, items=a.items + 1), "follow", ("edge_trader", "edge_trader"), (1, i), 501)
        price = t.prices[a.items]
        g = max(a.gold, price)
        add(f"A|now|{i}|later", with_a(s, gold=min(g, t.gold_cap[a.items])), "follow",
            ("edge_trader", "edge_trader"), (2, i), 502)
        add(f"A|now|{i}|now", with_a(s, items=a.items + 1, gold=min(g, t.gold_cap[a.items]) - price), "follow",
            ("edge_trader", "edge_trader"), (2, i), 502)
    # B: Recall now vs Hold now, HP 2 / 4 / 6 sixths, affordable vs not
    for name, s in recall_states(cfg).items():
        a = s.champ.a
        L = level(a, t)
        H = t.hp_max[L]
        if a.items >= t.item_tiers:
            s = with_a(s, items=t.item_tiers - 1)
            a = s.champ.a
        price = t.prices[a.items]
        for hp6 in (2, 4, 6):
            for afford in (True, False):
                v = with_a(s, hp=max(1, H * hp6 // 6), gold=price if afford else max(0, price - 1_000_000))
                for plan in ("recall", "hold"):
                    add(f"B|{name}|{hp6}|{'afford' if afford else 'poor'}|{plan}", v, plan,
                        ("edge_trader", "edge_trader"), (3, _stable(name), hp6, afford), 503)
    # C: forward vs back under a held freeze, holding vs trading freezer (B is the focal opponent)
    s1 = witness_snapshot("1_equal_power_freeze")
    for fz in ("freeze_hold", "freeze_trade"):
        for pos_pol, pos in (("forward", 1), ("back", -1)):
            sv = s1.with_side(1, pos=pos)
            add(f"C|{fz}|{pos_pol}", sv, "follow", (fz, pos_pol), (4,), 504, focal=1)
    # D: Recall vs death as a route to the shop
    for name, s in recall_states(cfg).items():
        a = s.champ.a
        if a.items >= t.item_tiers:
            continue
        price = t.prices[a.items]
        L = level(a, t)
        low = with_a(s, hp=max(1, t.hp_max[L] // 6), gold=max(a.gold, price))
        add(f"D|{name}|recall", low, "recall", ("edge_trader", "edge_trader"), (5, _stable(name)), 505)
        dead = with_a(low, hp=0, absent=cfg.death_turns, pos=-cfg.tower)
        add(f"D|{name}|death", dead, "follow", ("edge_trader", "edge_trader"), (5, _stable(name)), 505,
            offset=cfg.kill_bounty)
        fwd = with_a(low, pos=1)
        add(f"D|{name}|walk_back_recall", fwd, "walk_back_recall", ("edge_trader", "edge_trader"),
            (5, _stable(name)), 505)
        add(f"D|{name}|death_forward", dead, "follow", ("edge_trader", "edge_trader"), (5, _stable(name)), 505,
            offset=cfg.kill_bounty)
    # E: denial by an equal and a strong freezer (advances and trades) against an XP keeper
    for strength in ("equal", "strong"):
        s = s1
        if strength == "strong":
            b = s.champ.b
            Lb = level(b, t)
            La = min(t.level_cap, Lb + 2)
            s = with_a(s, xp=t.xp_cum[La], items=min(t.item_tiers, b.items + 1), hp=t.hp_max[La])
        add(f"E|{strength}", s, "follow", ("strong_freezer", "xp_keeper"), (6,), 506, focal=1)
    return jobs


# ---------------------------------------------------------------------- descriptive parts F, G, H
def plates_part(cfg: GameConfig, games: int = 60, turns: int = 400) -> dict:
    ag = agents()
    env = LaneEnv(cfg, max_turns=turns, cache_size=50_000)
    out = {}
    for x, y in (("presser", "hold_farmer"), ("g_stacker", "hold_farmer"), ("presser", "gold_recaller"),
                 ("freeze_breaker", "gold_recaller"), ("g_adaptive_breaker", "freezer")):
        total = away = 0.0
        plates = tau = 0
        for g in range(games):
            log = play_episode(env, ag[x], ag[y], 33_000 + g, record=True)
            for i in log.turns:
                total += i["plate_gold"][0]
                if i["away"][0]:
                    away += i["plate_gold"][0]
            plates += sum(i["plates_broken"][0] for i in log.turns)
            tau += log.m4_final["tower_dmg"][0]
        out[f"{x} vs {y}"] = {"plate_gold_per_game": total / games, "share_while_away": away / total if total else None,
                              "plates_per_game": plates / games, "tower_damage_per_game": tau / games}
    return out


def snowball_part(cfg: GameConfig, games: int = 300, window: int = 1000) -> dict:
    ag = agents()
    env = LaneEnv(cfg, max_turns=window, cache_size=50_000)
    g = cfg.gamma
    early, final, gaps = [], [], {c: [] for c in (30, 60, 100, 150) if c <= window}
    for i in range(games):
        log = play_episode(env, ag["edge_trader"], ag["edge_trader"], 34_000 + i, record=True)
        r = np.asarray(log.rewards)
        early.append(float(r[:30].sum()))
        final.append(float(np.sum(r * g ** np.arange(len(r)))))
        for c in gaps:
            gaps[c].append(log.turns[c - 1]["levels"][0] - log.turns[c - 1]["levels"][1])
    early, final = np.array(early), np.array(final)
    lead = early > 0
    return {"games": games, "p_final_pos_given_early_lead": float(np.mean(final[lead] > 0)) if lead.any() else None,
            "p_final_pos_given_early_deficit": float(np.mean(final[~lead] > 0)) if (~lead).any() else None,
            "corr_early_final": float(np.corrcoef(early, final)[0, 1]),
            "level_gap_abs_mean": {c: float(np.mean(np.abs(v))) for c, v in gaps.items()},
            "level_gap_ge2_share": {c: float(np.mean(np.abs(v) >= 2)) for c, v in gaps.items()}}


def item_diagnosis(cfg: GameConfig, n: int = 40, turns: int = 400) -> dict:
    """Why is a tier worth so little? The free-tier value under continuations of increasing combat intensity, with
    deaths and connected trades; the share of turns on which the two champions' items differ; death causes."""
    import collections
    ag = agents()
    snaps = [s for s in edge_snapshots(cfg) if s.champ.a.absent == 0 and s.champ.a.items < tables(cfg).item_tiers][:6]
    env = LaneEnv(cfg, cache_size=50_000)
    out = {"free_tier": {}, "unequal_item_turns_per_game": {}, "death_causes": {}}
    for pair in (("edge_trader", "edge_trader"), ("aggressive_trader", "aggressive_trader"),
                 ("aggressive_trader", "hold_farmer")):
        diffs, deaths, trades = [], [], []
        for i, s in enumerate(snaps):
            for m in range(n):
                rs = [rollout(env, v, 0, FollowPolicy(), (ag[pair[0]], ag[pair[1]]), rollout_seed(9, m, i),
                              ProbCache(), max_turns=turns, record=True)
                      for v in (s, s.with_side(0, items=s.champ.a.items + 1))]
                diffs.append(rs[1].ret - rs[0].ret)
                deaths.append([list(r.deaths) for r in rs])
                trades.append([sum(t["trade_connected"][0] for t in r.infos) for r in rs])
        m_, se = mean_se(diffs)
        dd = np.array(deaths, dtype=float)
        key = f"{pair[0]} vs {pair[1]}"
        out["free_tier"][key] = {"value": m_, "ci95": Z95 * se, "deaths_without": dd[:, 0].mean(0).tolist(),
                                 "deaths_with": dd[:, 1].mean(0).tolist(),
                                 "a_trades_without_with": np.mean(trades, axis=0).tolist()}
        envp = LaneEnv(cfg, max_turns=turns, cache_size=50_000)
        c, unequal = collections.Counter(), []
        for g in range(30):
            log = play_episode(envp, ag[pair[0]], ag[pair[1]], 37_000 + g, record=True)
            unequal.append(sum(t["items"][0] != t["items"][1] for t in log.turns))
            for t in log.turns:
                for side in (0, 1):
                    if t["died"][side]:
                        c["deaths"] += 1
                        c["with_gank_damage"] += t["gank_damage"][side] > 0
                        c["trade_damage_only"] += t["gank_damage"][side] == 0 and t["tower_damage"][side] == 0
        out["unequal_item_turns_per_game"][key] = float(np.mean(unequal))
        out["death_causes"][key] = dict(c)
    return out


def rr_job(args):
    x, y, episodes, window, seed, cfg_dict = args
    from ...env.discounted import evaluate_discounted
    cfg = GameConfig.from_dict(cfg_dict)
    ag = agents()
    r = evaluate_discounted(cfg, ag[x], ag[y], episodes=episodes, seed=seed, window=window)
    return x, y, r


def round_robin(cfg: GameConfig, episodes: int, window: int, procs: int) -> dict:
    names = list(m4_suite()) + ["edge_trader"]
    jobs = [(x, y, episodes, window, 35_000, cfg.to_dict()) for i, x in enumerate(names) for y in names[i + 1:]]
    res = pool_map(rr_job, jobs, procs)
    M = {x: {y: 0.0 for y in names} for x in names}
    SE = {x: {y: 0.0 for y in names} for x in names}
    tail = None
    for x, y, r in res:
        M[x][y], M[y][x] = r["estimate"], -r["estimate"]
        SE[x][y] = SE[y][x] = r["stderr"]
        tail = r["tail_bound"]["guaranteed"]
    dominant = [x for x in names if all(M[x][y] > 0 for y in names if y != x)]
    return {"names": names, "matrix": M, "stderr": SE, "episodes_per_seat": episodes, "window": window,
            "tail_bound": tail, "dominant": dominant,
            "row_mean": {x: float(np.mean([M[x][y] for y in names if y != x])) for x in names}}


# ---------------------------------------------------------------------- report
def analyse(results: Dict[str, np.ndarray]) -> dict:
    out = {"A": {}, "B": {}, "C": {}, "D": {}, "E": {}}
    keys = list(results)
    for k in keys:
        p = k.split("|")
        if p[0] == "A" and p[-1] in ("N+1", "now"):
            other = "N" if p[-1] == "N+1" else "later"
            out["A"][f"{p[1]}|{p[2]}"] = paired(results[k], results["|".join(p[:-1] + [other])])
        if p[0] == "B" and p[-1] == "recall":
            out["B"]["|".join(p[1:4])] = paired(results[k], results["|".join(p[:-1] + ["hold"])])
        if p[0] == "D" and p[-1] == "recall":
            out["D"][f"{p[1]}|from -1"] = paired(results[k], results["|".join(p[:-1] + ["death"])])
        if p[0] == "D" and p[-1] == "walk_back_recall":
            out["D"][f"{p[1]}|from +1"] = paired(results[k], results["|".join(p[:-1] + ["death_forward"])])
    for fz in ("freeze_hold", "freeze_trade"):
        out["C"][fz] = paired(results[f"C|{fz}|forward"], results[f"C|{fz}|back"])
    out["E"]["strong_minus_equal"] = paired(results["E|strong"], results["E|equal"])
    b = out["B"]
    out["recall_sign_pattern"] = {"positive": [k for k, v in b.items() if v["significant"] and v["diff"] > 0],
                                  "negative": [k for k, v in b.items() if v["significant"] and v["diff"] < 0]}
    out["recall_sign_pattern"]["degenerate"] = not (out["recall_sign_pattern"]["positive"] and
                                                    out["recall_sign_pattern"]["negative"])
    return out


def _rows(d: dict):
    return [[k, f"{v['diff']:+.3f}", f"±{Z95 * v['se']:.3f}", f"{v['p_positive']:.2f}",
             "yes" if v["significant"] else "no", f"{v['a']:+.2f}", f"{v['b']:+.2f}"] for k, v in d.items()]


def markdown(d) -> str:
    a = d["analysis"]
    hdr = ["comparison", "paired diff", "95% CI", "P(diff > 0)", "significant", "value 1", "value 2"]
    L = ["# S4 — Economic sanity (M4.2)", "",
         f"Commit `{d['meta']['commit'][:10]}`. Paired counterfactual rollouts ({d['rollouts']} per variant, common "
         f"random numbers), discounted over W = {d['window']} turns (guaranteed truncation error per value ≤ "
         f"{d['tail_bound']:.4f}). Scripted continuations: approximate champion-game measurements, not equilibrium "
         f"values. Command: `{d['meta']['command']}`.", "",
         "## A — item value (focal A): free tier (N + 1 vs N, same gold) and buying now vs at the next return", "",
         md_table(hdr, _rows(a["A"])), "",
         "## B — Recall now minus Hold now (focal A), by state | HP in sixths | affordable tier", "",
         md_table(hdr, _rows(a["B"])), "",
         f"Recall sign pattern: significantly positive in {len(a['recall_sign_pattern']['positive'])} cells, "
         f"significantly negative in {len(a['recall_sign_pattern']['negative'])} cells → "
         f"**{'DEGENERATE (design finding)' if a['recall_sign_pattern']['degenerate'] else 'both signs present'}**.", "",
         "## C — the opponent under a held freeze: forward (+1, Hold) minus back (−1, Safe), focal B", "",
         md_table(hdr, _rows(a["C"])), "",
         "## D — Recall versus death as the route to the shop (focal A; death includes the 0.5 bounty)", "",
         md_table(hdr, _rows(a["D"])), "",
         "## E — denial by a strong freezer (+2 levels, +1 tier) minus an equal one, focal B (xp_keeper)", "",
         md_table(hdr, _rows(a["E"])), "",
         "### Item diagnosis: a free tier under continuations of increasing combat intensity (400 turns)", "",
         md_table(["continuation", "free-tier value (95% CI)", "deaths A, B without", "deaths A, B with",
                   "A trades without / with", "turns per game with unequal items", "deaths: with gank damage / "
                   "trade damage only"],
                  [[k, f"{v['value']:+.3f} ± {v['ci95']:.3f}", ", ".join(f"{x:.1f}" for x in v["deaths_without"]),
                    ", ".join(f"{x:.1f}" for x in v["deaths_with"]),
                    " / ".join(f"{x:.0f}" for x in v["a_trades_without_with"]),
                    f"{d['item_diagnosis']['unequal_item_turns_per_game'][k]:.1f}",
                    f"{d['item_diagnosis']['death_causes'][k].get('with_gank_damage', 0)} / "
                    f"{d['item_diagnosis']['death_causes'][k].get('trade_damage_only', 0)}"]
                   for k, v in d["item_diagnosis"]["free_tier"].items()]), "",
         "## F — plates", "",
         md_table(["pairing (A vs B)", "plate gold per game (A)", "share earned while A away", "plates per game",
                   "tower damage per game (centi)"],
                  [[k, f"{v['plate_gold_per_game']:.2f}",
                    "-" if v["share_while_away"] is None else f"{v['share_while_away']:.0%}",
                    f"{v['plates_per_game']:.2f}", f"{v['tower_damage_per_game']:.0f}"] for k, v in d["plates"].items()]),
         "", "## G — snowballing (mirror edge traders)", ""]
    s = d["snowball"]
    L += [f"- P(final discounted margin > 0 | ahead after 30 turns) = {s['p_final_pos_given_early_lead']:.2f}; "
          f"| behind = {s['p_final_pos_given_early_deficit']:.2f}; correlation {s['corr_early_final']:.2f} "
          f"({s['games']} games, W = {d['window']})",
          f"- mean |level gap| at turns 30/60/100/150: " + ", ".join(f"{v:.2f}" for v in s["level_gap_abs_mean"].values())
          + "; share with a gap ≥ 2: " + ", ".join(f"{v:.0%}" for v in s["level_gap_ge2_share"].values()), "",
          "## H — scripted round robin (E1 from s0, W = 1000, both seats, CRN): payoff to the row", ""]
    rr = d["round_robin"]
    names = rr["names"]
    rows = [[x] + [("" if x == y else f"{rr['matrix'][x][y]:+.2f}") for y in names] + [f"{rr['row_mean'][x]:+.2f}"]
            for x in names]
    L.append(md_table(["row \\ column"] + names + ["mean"], rows))
    L += ["", f"Dominant scripted strategy (beats every other): {rr['dominant'] or 'none'}. Guaranteed truncation "
          f"bound per value {rr['tail_bound']:.4f}."]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", type=int, default=60)
    ap.add_argument("--window", type=int, default=1000)
    ap.add_argument("--rr-episodes", type=int, default=60)
    ap.add_argument("--procs", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    cfg = GameConfig.m4()
    timer = Timer()
    jobs = build_jobs(cfg, a.rollouts, a.window)
    print(f"{len(jobs)} variant jobs", flush=True)
    res = pool_map(run_job, list(jobs.values()), a.procs)
    results = {jid: rets for jid, rets, _ in res}
    extras = {jid: ex for jid, _, ex in res}
    print(f"counterfactuals done ({timer():.0f} s)", flush=True)
    from ...env.discounted import tail_bound
    d = {"meta": meta(cfg, rollouts=a.rollouts, window=a.window), "rollouts": a.rollouts, "window": a.window,
         "tail_bound": tail_bound(cfg, a.window), "analysis": analyse(results),
         "raw": {k: v.tolist() for k, v in results.items()},
         "counters": {k: {f: np.mean([e[f] for e in v], axis=0).tolist() for f in v[0]} for k, v in extras.items()}}
    d["item_diagnosis"] = item_diagnosis(cfg, n=4 if a.quick else 40, turns=100 if a.quick else 400)
    d["plates"] = plates_part(cfg, games=20 if a.quick else 60)
    d["snowball"] = snowball_part(cfg, games=40 if a.quick else 300, window=a.window)
    print(f"descriptive parts done ({timer():.0f} s)", flush=True)
    d["round_robin"] = round_robin(cfg, 6 if a.quick else a.rr_episodes, a.window, a.procs)
    d["elapsed_s"] = timer()
    save("economy", d, markdown(d))
    sp = d["analysis"]["recall_sign_pattern"]
    print(f"S4 recall sign pattern: +{len(sp['positive'])} -{len(sp['negative'])} degenerate={sp['degenerate']}; "
          f"dominant={d['round_robin']['dominant']} ({timer():.0f} s)")
    return d


if __name__ == "__main__":
    main()
