"""M2e-D/E: the open M2a concerns and degenerate-incentive probes, measured before any default is changed.

Each probe varies one parameter (or a pair) away from `GameConfig.champions()` and reports the matchups that
the parameter is supposed to affect, with 95% confidence intervals (seeded, common random numbers):

  bounty      kill_bounty x retreat_farm_reach grid: value of HP management (recalling at low HP versus never),
              Trade versus Hold, zoning of a Yield by a trader, and the Nash mixture of the scripted meta-game
  trade_farm  trade_farm_factor (opportunity cost of a connected trade)
  probes      degenerate strategies: recall spam, permanent Safe / Trade, vision play, gank exposure switched
              off, tower dives; and the geometry parameters farm_reach and champion_speed
"""

from __future__ import annotations

import itertools

import numpy as np

from .common import base_config, fmt_ci, md_table, meta_nash, round_robin, run_pairs, write

CORE = ["hold", "hold_farmer", "conservative_recaller", "trade", "aggressive_trader", "opportunistic_trader",
        "yield_farmer", "safe_player", "punisher"]


def _cell(rr, a, b):
    i, j = rr["names"].index(a), rr["names"].index(b)
    return rr["matrix"][i][j], rr["stderr"][i][j]


def bounty_grid(episodes: int, seed: int, bounties=(0.0, 0.5, 1.0, 2.0), reaches=(1, 2)) -> dict:
    out = []
    for bounty, reach in itertools.product(bounties, reaches):
        cfg = base_config(kill_bounty=bounty, retreat_farm_reach=reach)
        rr = round_robin(cfg, CORE, episodes, seed)
        nash = meta_nash(np.array(rr["matrix"]), CORE)
        stats = {}
        for r in rr["pairs"]:
            for s, n in enumerate(r["names"]):
                stats.setdefault(n, {"deaths": [], "recalls": [], "farm": [], "bounty": []})
                for m in ("deaths", "recalls", "farm", "bounty"):
                    stats[n][m].append(r[m][s])
        stats = {n: {m: float(np.mean(v)) for m, v in d.items()} for n, d in stats.items()}
        out.append({"kill_bounty": bounty, "retreat_farm_reach": reach, "round_robin": rr, "nash": nash,
                    "per_agent": stats})
        print(f"bounty {bounty} reach {reach}: nash {nash['mixture']} ({rr['elapsed_s']:.0f}s)")
    return {"grid": out, "agents": CORE, "episodes": episodes, "seed": seed}


def bounty_markdown(res: dict) -> str:
    rows = []
    for g in res["grid"]:
        rr = g["round_robin"]
        rec = _cell(rr, "hold_farmer", "hold")
        th = _cell(rr, "trade", "hold")
        ah = _cell(rr, "aggressive_trader", "hold_farmer")
        oh = _cell(rr, "opportunistic_trader", "hold_farmer")
        yz = _cell(rr, "yield_farmer", "aggressive_trader")
        avg = np.array(rr["matrix"]).mean(axis=1)
        best = rr["names"][int(np.argmax(avg))]
        mix = ", ".join(f"{k} {v:.2f}" for k, v in sorted(g["nash"]["mixture"].items(), key=lambda kv: -kv[1]))
        rows.append([g["kill_bounty"], g["retreat_farm_reach"], fmt_ci(*rec), fmt_ci(*th), fmt_ci(*ah), fmt_ci(*oh),
                     fmt_ci(*yz), f"{g['per_agent']['hold']['deaths']:.2f}", best, mix])
    return md_table(["kill_bounty", "retreat_farm_reach", "hold_farmer vs hold (value of recalling at <=2 HP)",
                     "trade vs hold", "aggressive_trader vs hold_farmer", "opportunistic vs hold_farmer",
                     "yield_farmer vs aggressive_trader", "deaths of naive hold / game", "best average",
                     "Nash mixture of the 9-agent meta-game"], rows)


def trade_farm(episodes: int, seed: int, bounty: float, reach: int, factors=(0.25, 0.5, 0.75, 1.0)) -> dict:
    pairs = [("trade", "hold"), ("aggressive_trader", "hold_farmer"), ("opportunistic_trader", "hold_farmer"),
             ("trade", "trade"), ("hold_farmer", "hold_farmer")]
    out = []
    for f in factors:
        cfg = base_config(kill_bounty=bounty, retreat_farm_reach=reach, trade_farm_factor=f)
        recs = run_pairs(cfg, pairs, episodes, seed)
        out.append({"trade_farm_factor": f, "pairs": recs})
    return {"grid": out, "kill_bounty": bounty, "retreat_farm_reach": reach}


def probes(episodes: int, seed: int, bounty: float, reach: int) -> dict:
    base = dict(kill_bounty=bounty, retreat_farm_reach=reach)
    pairs = [("recall_spammer", "hold_farmer"), ("safe_player", "hold_farmer"), ("aggressive_trader", "hold_farmer"),
             ("vision_player", "hold_farmer"), ("shover", "hold_farmer"), ("punisher", "hold_farmer"),
             ("aggressive_trader", "safe_player"), ("hold_farmer", "hold_farmer")]
    variants = {
        "default": base,
        "ganks off": dict(base, champ_gank_position=0.0, champ_gank_behavior=(0.0, 0.0, 0.0, 0.0), champ_gank_shove=0.0,
                          champ_gank_strength=0.0),
        "no tower shot": dict(base, tower_shot_p=0.0, tower_shot_p_busy=0.0),
        "farm_reach 2": dict(base, farm_reach=2, retreat_farm_reach=max(2, reach)),
        "champion_speed 1": dict(base, champion_speed=1),
        "vision 50% / 3 turns": dict(base, vision_reduction=0.5, vision_turns=3),
    }
    out = {}
    for name, kw in variants.items():
        cfg = base_config(**kw)
        out[name] = run_pairs(cfg, pairs, episodes, seed)
        print(f"probe {name} done")
    return {"variants": {k: {kk: (list(v) if isinstance(v, tuple) else v) for kk, v in kw.items()}
                         for k, kw in variants.items()}, "pairs": pairs, "results": out}


def probes_markdown(res: dict) -> str:
    names = list(res["results"])
    rows = []
    for idx, (a, b) in enumerate(res["pairs"]):
        row = [f"{a} vs {b}"]
        for v in names:
            r = res["results"][v][idx]
            row.append(fmt_ci(r["mean"], r["stderr"]))
        rows.append(row)
    lines = [md_table(["matchup (payoff to the first)"] + names, rows), ""]
    rows = []
    for v in names:
        for idx, (a, b) in enumerate(res["pairs"]):
            r = res["results"][v][idx]
            if (a, b) in (("aggressive_trader", "hold_farmer"), ("recall_spammer", "hold_farmer"), ("vision_player", "hold_farmer"),
                          ("shover", "hold_farmer")):
                rows.append([v, f"{a} vs {b}", f"{r['farm'][0]:.2f} / {r['farm'][1]:.2f}", f"{r['plates'][0]:.2f}",
                             f"{r['deaths'][0]:.2f} / {r['deaths'][1]:.2f}", f"{r['recalls'][0]:.2f}",
                             f"{r['ganks'][0]:.2f} / {r['ganks'][1]:.2f}", f"{r['tower_damage_taken'][0]:.2f}",
                             f"{r['objectives'][0]['Vision']:.3f}", f"{r['turns_away'][0]:.2f}"])
    lines.append(md_table(["variant", "matchup", "farm (first / second)", "plates (first)", "deaths", "recalls (first)",
                           "ganks", "tower dmg taken (first)", "Vision share (first)", "turns away (first)"], rows))
    return "\n".join(lines)


def run(episodes: int = 300, seed: int = 11):
    res_b = bounty_grid(episodes, seed)
    chosen_bounty, chosen_reach = 0.5, 2               # the defaults adopted from section 1 (see docs)
    res_t = trade_farm(episodes, seed, chosen_bounty, chosen_reach)
    res_p = probes(episodes, seed, chosen_bounty, chosen_reach)
    rows = []
    for g in res_t["grid"]:
        rows.append([g["trade_farm_factor"]] + [fmt_ci(r["mean"], r["stderr"]) for r in g["pairs"]]
                    + [f"{g['pairs'][0]['farm'][0]:.2f}"])
    md = "\n".join([
        "# Champion mode: M2a concerns and degenerate-incentive probes (M2e-D/E)", "",
        f"Generated by `python -m lanegame experiments-champions concerns`; {episodes} episodes per ordered pair, "
        "the same seed for every pair (common random numbers), K = 32, equal skill. Payoffs are resources per game "
        "to the first-named agent with 95% confidence intervals.", "",
        "## 1. Kill bounty x retreat farm reach", "",
        "Nine agents (naive `hold` never recalls; `hold_farmer` recalls at <= 2 HP; `conservative_recaller` at <= 4; "
        "`trade` never recalls; `aggressive_trader` recalls at 1 HP; `opportunistic_trader`, `yield_farmer`, "
        "`safe_player`, `punisher`), full round robin, both seats.", "",
        bounty_markdown(res_b), "",
        f"## 2. trade_farm_factor (kill_bounty {chosen_bounty}, retreat_farm_reach {chosen_reach})", "",
        md_table(["trade_farm_factor", "trade vs hold", "aggressive_trader vs hold_farmer", "opportunistic vs hold_farmer",
                  "trade vs trade", "hold_farmer vs hold_farmer", "farm of `trade` vs hold"], rows), "",
        f"## 3. Degenerate-strategy probes and parameter checks (kill_bounty {chosen_bounty}, retreat_farm_reach {chosen_reach})", "",
        probes_markdown(res_p), ""])
    write("concerns", {"bounty_grid": res_b, "trade_farm": res_t, "probes": res_p}, md)
    return res_b, res_t, res_p


if __name__ == "__main__":
    run()
