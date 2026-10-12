"""S2: progression speed (design §9.3; plan §7 S2).

    python -m lanegame.experiments.m4.progression [--games 400] [--turns 400]

400 seeded 400-turn games of mutual hold_farmer (M4 scripted suite): the turn at which each champion first reaches
each level (median and quartiles over both seats), against the design targets 4 / 20 / 59 / 147 for levels 2 / 6 /
11 / 18 (pass: medians within +-25%). The only adjustable parameter on failure is xp_per_hp; a miss is reported, not
tuned. Repeated with aggressive_trader vs hold_farmer, reporting the level gap over time (design expectation: 1-2
levels by turn 60), deaths, and both matchups' item purchase turns (design §8.5: tier 1 around turns 14-20 and tier 2
around 35-50 at learned-play collection rates).
"""

from __future__ import annotations

import argparse

import numpy as np

from ...config import GameConfig
from ...env import LaneEnv, play_episode
from ...env.agents_m4 import m4_suite
from .common import Timer, md_table, meta, pool_map, save

TARGETS = {2: 4, 6: 20, 11: 59, 18: 147}
CHECKPOINTS = (30, 60, 100, 150, 200, 300, 399)


def run_chunk(args):
    x_name, y_name, seeds, turns, cfg_dict = args
    cfg = GameConfig.from_dict(cfg_dict)
    suite = m4_suite()
    env = LaneEnv(cfg, max_turns=turns, cache_size=50_000)
    out = []
    for seed in seeds:
        log = play_episode(env, suite[x_name], suite[y_name], seed, record=True)
        first = [[None] * 19, [None] * 19]
        buy = [[None, None], [None, None]]
        gap = {}
        xp_avail, xp_in_range = [0, 0], [0, 0]
        for t in log.turns:
            k = t["k"]
            for s in (0, 1):
                L = t["levels"][s]
                for l in range(2, L + 1):
                    if first[s][l] is None:
                        first[s][l] = k + 1                    # reached at the end of turn k: playing turn k + 1
                items = t["items"][s]
                for n in range(items):
                    if buy[s][n] is None:
                        buy[s][n] = k + 1
                xp_avail[s] += cfg.xp_per_hp * sum(t["kills"][s].values())
                xp_in_range[s] += t["xp_gain"][s] + t["xp_discarded"][s]      # earned, before the cap
            if (k + 1) in CHECKPOINTS:
                gap[k + 1] = t["levels"][0] - t["levels"][1]
        out.append({"first": first, "buy": buy, "gap": gap, "deaths": list(log.deaths), "farm": list(log.farm),
                    "xp_gain": xp_in_range, "xp_avail": xp_avail, "payoff": log.payoff,
                    "recalls": list(log.recalls)})
    return out


def stats(values):
    v = [x for x in values if x is not None]
    if not v:
        return {"n": 0, "reached": 0.0}
    q = np.quantile(v, [0.25, 0.5, 0.75])
    return {"n": len(v), "reached": len(v) / len(values), "q25": float(q[0]), "median": float(q[1]), "q75": float(q[2])}


def matchup(x, y, games, turns, cfg, procs):
    seeds = list(range(10_000, 10_000 + games))
    chunks = [seeds[i::max(1, procs or 10)] for i in range(max(1, procs or 10))]
    res = [r for part in pool_map(run_chunk, [(x, y, c, turns, cfg.to_dict()) for c in chunks if c], procs) for r in part]
    levels = {}
    for L in range(2, 19):
        both = [g["first"][s][L] for g in res for s in (0, 1)]
        levels[L] = {"both": stats(both), "A": stats([g["first"][0][L] for g in res]),
                     "B": stats([g["first"][1][L] for g in res])}
    buys = {f"tier{n + 1}": {"A": stats([g["buy"][0][n] for g in res]), "B": stats([g["buy"][1][n] for g in res])}
            for n in range(2)}
    gaps = {c: {"mean": float(np.mean([g["gap"][c] for g in res if c in g["gap"]])),
                "sd": float(np.std([g["gap"][c] for g in res if c in g["gap"]]))} for c in CHECKPOINTS}
    absorption = [float(np.sum([g["xp_gain"][s] for g in res]) / max(1, np.sum([g["xp_avail"][s] for g in res])))
                  for s in (0, 1)]
    return {"x": x, "y": y, "games": games, "turns": turns, "levels": levels, "purchases": buys, "level_gap": gaps,
            "xp_absorption": absorption, "deaths_per_game": [float(np.mean([g["deaths"][s] for g in res])) for s in (0, 1)],
            "recalls_per_game": [float(np.mean([g["recalls"][s] for g in res])) for s in (0, 1)],
            "farm_per_turn": [float(np.mean([g["farm"][s] for g in res]) / turns) for s in (0, 1)],
            "payoff_mean": float(np.mean([g["payoff"] for g in res]))}


def markdown(d) -> str:
    m = d["mutual"]
    L = ["# S2 — Progression speed (M4.2)", "",
         f"Commit `{d['meta']['commit'][:10]}`. {m['games']} seeded games of {m['turns']} turns per matchup. "
         f"Command: `{d['meta']['command']}`.", "",
         "## Mutual hold_farmer: turn at which each level is first reached (both seats)", ""]
    rows = []
    for Lv in range(2, 19):
        s = m["levels"][Lv]["both"]
        tgt = TARGETS.get(Lv)
        ok = "" if tgt is None else ("within ±25%" if abs(s["median"] - tgt) <= 0.25 * tgt else "MISS")
        rows.append([Lv, s.get("q25"), s.get("median"), s.get("q75"), f"{s['reached']:.0%}", tgt or "", ok])
    L.append(md_table(["level", "q25", "median", "q75", "reached", "design target", "S2"], rows))
    L += ["", f"Verdict: **{'PASS' if d['pass'] else 'FAIL'}** (medians of levels 2 / 6 / 11 / 18 within ±25% of "
          f"4 / 20 / 59 / 147). XP absorption (XP earned in range, including XP beyond the cap, / XP of all enemy HP "
          f"dying): "
          f"{m['xp_absorption'][0]:.1%} / {m['xp_absorption'][1]:.1%} (design assumed 94%).", ""]
    for key, label in (("mutual", "mutual hold_farmer"), ("trader", "aggressive_trader (A) vs hold_farmer (B)")):
        r = d[key]
        L += [f"## {label}", "",
              f"- deaths per game A {r['deaths_per_game'][0]:.2f}, B {r['deaths_per_game'][1]:.2f}; recalls per game "
              f"A {r['recalls_per_game'][0]:.2f}, B {r['recalls_per_game'][1]:.2f}; farm per turn A "
              f"{r['farm_per_turn'][0]:.3f}, B {r['farm_per_turn'][1]:.3f}; payoff to A (undiscounted, "
              f"{r['turns']} turns) {r['payoff_mean']:+.2f}",
              f"- item purchases (turn of the purchase): tier 1 A median {r['purchases']['tier1']['A'].get('median')}, "
              f"B {r['purchases']['tier1']['B'].get('median')}; tier 2 A {r['purchases']['tier2']['A'].get('median')}, "
              f"B {r['purchases']['tier2']['B'].get('median')}", ""]
        L.append(md_table(["turn", "level gap A − B (mean)", "sd"],
                          [[c, f"{v['mean']:+.2f}", f"{v['sd']:.2f}"] for c, v in r["level_gap"].items()]))
        L.append("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--turns", type=int, default=400)
    ap.add_argument("--procs", type=int, default=0)
    a = ap.parse_args(argv)
    cfg = GameConfig.m4()
    timer = Timer()
    d = {"meta": meta(cfg, games=a.games, turns=a.turns, seeds="10000 .. 10000 + games")}
    d["mutual"] = matchup("hold_farmer", "hold_farmer", a.games, a.turns, cfg, a.procs)
    d["trader"] = matchup("aggressive_trader", "hold_farmer", a.games, a.turns, cfg, a.procs)
    meds = {Lv: d["mutual"]["levels"][Lv]["both"]["median"] for Lv in TARGETS}
    d["targets"] = {str(k): {"target": v, "median": meds[k], "rel_error": (meds[k] - v) / v} for k, v in TARGETS.items()}
    d["pass"] = all(abs(meds[k] - v) <= 0.25 * v for k, v in TARGETS.items())
    d["elapsed_s"] = timer()
    save("progression", d, markdown(d))
    print("S2", "PASS" if d["pass"] else "FAIL", {k: round(v["rel_error"], 3) for k, v in d["targets"].items()},
          f"({timer():.0f} s)")
    return d


if __name__ == "__main__":
    main()
