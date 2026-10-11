"""Candidate mechanics under the revised economy (Slow Push, Freeze, tower, cap rule, HP units, attrition).

Every variant changes exactly one mechanic of the revised default and is evaluated on the same
criteria: behavioural scenarios, reachable events from the standard start (crash sizes, bounces,
held/broken freezes, deleted minions), state-space size and solve time, equilibrium structure, and
baseline exploitability. This is the evidence for the choices documented in the revision report
(section E).
"""

from __future__ import annotations

from .. import GameConfig, LaneGame
from ..baselines import all_baselines
from ..best_response import exploitability
from ..reachability import explore_events
from ..scenarios import all_scenarios, run_scenario
from .common import REVISION, eq_summary, md_table, save, solve

K = 32
W2 = dict(wave_hp=2, farm_per_hp=0.5, freeze_break_threshold=5, freeze_break_push=(0, 2, 0), siege_cap=24)

VARIANTS = {
    "revised default (stack_creep, period 2)": {},
    "Slow Push hold at lane (original rule)": {"slowpush_mode": "hold"},
    "Slow Push creep, 1 cell / spawn interval": {"slowpush_mode": "creep", "slowpush_creep_period": 1},
    "Slow Push creep, 1 cell / 2 intervals": {"slowpush_mode": "creep"},
    "Slow Push stack_creep, period 1": {"slowpush_creep_period": 1},
    "Slow Push half speed past lane": {"slowpush_mode": "half"},
    "Slow Push free (no line)": {"slowpush_mode": "free"},
    "Freeze legal from the lane (original rule)": {"freeze_legal_max": 0},
    "tower: 1 packet per turn (original rule)": {"tower_mode": "packet"},
    "tower fire 2 HP per tick": {"tower_siege_damage": 2},
    "cap rule: delete overflow HP (original rule)": {"cap_rule": "overflow"},
    "queued packets stand idle (do not fight with the front)": {"queue_fights": False},
    "defenders engage crashed minions under tower": {"siege_engage": True},
    "defenders engage + tower fire 2 HP per tick": {"siege_engage": True, "tower_siege_damage": 2},
    "HP units W = 2 (original resolution)": W2,
    "attrition: linear (1 per tick)": {"damage_law": "linear"},
    "attrition: Lanchester ceil": {"damage_law": "lanchester_ceil"},
}


def run(horizon: int = K) -> dict:
    out = {}
    for name, kw in VARIANTS.items():
        print(name, flush=True)
        cfg = GameConfig.v2(horizon=horizon, **kw)
        try:
            g, layers, sol, secs = solve(cfg)
        except RuntimeError as err:                       # unbounded state growth (safety cap hit)
            out[name] = {"overrides": kw, "error": str(err)}
            continue
        summ = eq_summary(g, sol)
        scen = {}
        for sc in all_scenarios(cfg):
            ok, detail, _, _ = run_scenario(sc, cfg)
            scen[sc.name] = {"pass": bool(ok), "detail": detail}
        rep = explore_events(LaneGame(cfg))
        try:          # boundedness: does the set of distinct states keep growing with the horizon?
            d64 = len(set().union(*LaneGame(cfg.with_(horizon=2 * horizon)).reachable(max_states=2_000_000)))
        except RuntimeError:
            d64 = None
        expl = {b.name: exploitability(g, layers, b, 0, sol.value).exploitability for b in all_baselines()}
        out[name] = {"overrides": kw, "seconds": secs, **summ, "distinct_2K": d64, "scenarios": scen,
                     "scenarios_passed": sum(v["pass"] for v in scen.values()),
                     "events": rep.counts, "baseline_exploitability": expl}
    failed = {n: d["error"] for n, d in out.items() if "error" in d}
    out_ok = {n: d for n, d in out.items() if "error" not in d}
    n_sc = len(all_scenarios(GameConfig.v2()))
    ev_keys = ["crash_small", "crash_medium", "crash_large", "crash_onto_siege", "bounce", "freeze_held",
               "freeze_broken", "minions_deleted"]
    md = [f"# Candidate mechanics under the revised economy (K = {horizon})", "",
          "Each row changes one mechanic of the revised default. Scenario checks are the revised scenarios "
          "(some test a specific rule of the default, so a variant that changes that rule fails them by design).",
          "",
          *(["Variants whose reachable state space exceeded the safety cap: " +
             "; ".join(f"{n} ({e})" for n, e in failed.items()), ""] if failed else []),
          f"'distinct at 2K' enumerates the same variant at horizon {2 * horizon}: equal counts mean the set of "
          "distinct states has saturated (bounded state space); a larger count means it keeps growing with the "
          "horizon.", "",
          md_table(["variant", "scenarios", "distinct states", f"distinct at 2K", "nodes", "mixed", "solve s",
                    "eq crashes (A)", "eq P(Contest)", "P(Contest) nodes", "P(Freeze) legal"],
                   [(n, f"{d['scenarios_passed']}/{n_sc}", d["distinct"], d["distinct_2K"] or "> cap", d["states"],
                     d["mixed"],
                     round(d["seconds"], 1), round(sum(d["path"]["crashes_by_size"]["A"]), 3),
                     round(d["path"]["contest_share"][0], 2), round(d["nodes"]["contest"], 3),
                     round(d["nodes"]["freeze_prob_when_legal"], 3)) for n, d in out_ok.items()]), "",
          "## Reachable events from the standard start (number of reachable transitions)", "",
          md_table(["variant"] + ev_keys, [(n, *[d["events"][e] for e in ev_keys]) for n, d in out_ok.items()]), "",
          "## Failed scenario checks", ""]
    for name, d in out_ok.items():
        fails = [f"`{k}` ({v['detail']})" for k, v in d["scenarios"].items() if not v["pass"]]
        md.append(f"- **{name}**: " + ("none" if not fails else "; ".join(fails)))
    md += ["", "## Baseline exploitability (as player A)", "",
           md_table(["variant"] + [b.name for b in all_baselines()],
                    [(n, *[round(v, 2) for v in d["baseline_exploitability"].values()]) for n, d in out_ok.items()])]
    save("model_comparison", {"horizon": horizon, "variants": out}, "\n".join(md), REVISION)
    return out


if __name__ == "__main__":
    run()
