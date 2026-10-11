"""Experiment B - the farming economy: disabled vs. enabled, and its scale.

farm_per_hp is varied from 0 (no farming: only crash rewards and ganks) to 4x the default
(4 resources per fresh wave). Everything else stays at the revised default, so the crash reward
table (1/2/3) and the gank loss (L = 5) are fixed and the experiment shows what happens as farming
becomes more (or less) important relative to them.
"""

from __future__ import annotations

from .. import GameConfig
from ..baselines import all_baselines
from .common import REVISION, eq_summary, exploit_row, fmt_freq, md_table, save, solve
from .probes import probes, solve_probe

K = 32
SCALES = [0.0, 0.125, 0.25, 0.5, 1.0]        # per HP; W = 4, so 0 / 0.5 / 1 / 2 / 4 resources per wave
PROBES = ["B wave at A's freeze line, fighting A's wave", "A 2-wave stack at +1, B wave just spawned",
          "A's 3-wave crash under B's tower", "B's 3-wave crash under A's tower (A defends)"]


def run(horizon: int = K) -> dict:
    res = {}
    for phi in SCALES:
        print("farm_per_hp", phi, flush=True)
        cfg = GameConfig(horizon=horizon, farm_per_hp=phi)
        g, layers, sol, secs = solve(cfg)
        d = eq_summary(g, sol)
        d["seconds"] = secs
        d["baselines"] = {}
        for b in all_baselines():
            row = exploit_row(g, layers, sol, b)
            row.pop("br_path")
            d["baselines"][b.name] = row
        ps = probes(cfg.wave_hp)
        d["probes"] = {n: solve_probe(cfg, ps[n]) for n in PROBES}
        res[phi] = d

    md = [f"# Experiment B: farming economy (K = {horizon})", "",
          "farm_per_hp is the value of one HP of minions; a fresh wave has 4 HP, so the default 0.25 means one "
          "resource per wave. Crash rewards (1/2/3) and the gank loss (5) are unchanged.", "",
          "## Equilibrium structure", "",
          md_table(["farm/HP", "nodes", "mixed", "P(Contest) avg node", "P(Freeze) where legal",
                    "eq path actions (A)", "eq path P(Contest)", "eq crashes A", "vision S/M/L"],
                   [(phi, d["states"], d["mixed"], round(d["nodes"]["contest"], 3),
                     round(d["nodes"]["freeze_prob_when_legal"], 3), fmt_freq(d["path"]["action_freq"][0]),
                     round(d["path"]["contest_share"][0], 3), round(sum(d["path"]["crashes_by_size"]["A"]), 3),
                     "/".join(f"{d['vision'].get(s, float('nan')):.2f}" for s in ("small", "medium", "large")))
                    for phi, d in res.items()]), "",
          "## Equilibrium path economy (per player, whole game)", "",
          md_table(["farm/HP", "farm available", "farm collected", "collection rate", "gank loss"],
                   [(phi, round(d["path"]["farm_available"][0], 2), round(d["path"]["farm"][0], 2),
                     round(d["path"]["farm"][0] / d["path"]["farm_available"][0], 3) if d["path"]["farm_available"][0] else "-",
                     round(d["path"]["gank_loss"][0], 2)) for phi, d in res.items()]), "",
          "## Exploitability of baselines (as A) [farm / crash / gank decomposition of the value vs. best response]", "",
          md_table(["farm/HP"] + [b.name for b in all_baselines()],
                   [(phi, *[f"{r['exploitability']:.2f} [{r['farm']:.1f}/{r['crash']:.1f}/{r['gank']:.1f}]"
                            for r in d["baselines"].values()]) for phi, d in res.items()]), "",
          "## Probe situations (k = 12 of 32): value to A, A's equilibrium strategy, best-Contest minus best-Safe", ""]
    for n in PROBES:
        md += [f"**{n}**", "",
               md_table(["farm/HP", "V*", "A plays", "B plays", "Contest - Safe"],
                        [(phi, round(d["probes"][n]["value"], 3), d["probes"][n]["A"], d["probes"][n]["B"],
                          round(d["probes"][n]["contest_advantage"], 3)) for phi, d in res.items()]), ""]
    save("farm_economy", {"horizon": horizon, "scales": SCALES, "results": res}, "\n".join(md), REVISION)
    return res


if __name__ == "__main__":
    run()
