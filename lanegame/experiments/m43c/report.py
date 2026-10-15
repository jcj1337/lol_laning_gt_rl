"""results/m43c/report.md: every table of the M4.3c design study, generated from the stage JSON files."""

from __future__ import annotations

from typing import List

from .common import load, write_md

NA = "—"


def _q(d: dict, digits: int = 0) -> str:
    if not d or d.get("q50") != d.get("q50"):
        return NA
    f = f"{{:.{digits}f}}"
    return f"{f.format(d['q50'])} ({f.format(d['q10'])}–{f.format(d['q90'])})"


def _mc(x, digits: int = 2) -> str:
    m, h = x
    if h != h:
        return f"{m:+.{digits}f}"
    return f"{m:+.{digits}f} ± {h:.{digits}f}"


def progression_md(d: dict) -> List[str]:
    out = ["## 1. Progression audit (`progression.json.gz`)", "",
           f"W = {d['meta']['window']} turns, {d['meta']['episodes']} games per matchup from s0; medians "
           "(10th–90th percentile). Plate turns pool both towers; 'end' = the first turn either tower's last plate "
           "falls (the Part C proposal).", "",
           "| matchup | family | plate 1 turn | plate 2 turn | plate 3 turn | levels at plate 1 | end turn | levels at end "
           "| ended by 150 | both L18 turn | weight after end | Pressure share | deaths / recalls per side | "
           "value to A, W | value to A, to end |", "|---" * 15 + "|"]
    for name, m in d["matchups"].items():
        s = m["summary"]
        out.append(f"| {name} | {m['family']} | {_q(s['plate1_turn'])} | {_q(s['plate2_turn'])} | {_q(s['plate3_turn'])} "
                   f"| {_q(s.get('plate1_levels', {}))} | {_q(s['end_turn'])} | {_q(s['end_levels'])} "
                   f"| {s['end_share_by_150']:.2f} | {_q(s['both18_turn'])} | {s['weight_after_end'][0]:.2f} "
                   f"| {s['pressure_share_of_tower_damage']:.2f} | {s['deaths_per_side']:.1f} / {s['recalls_per_side']:.1f} "
                   f"| {_mc(s['disc_payoff_a'])} | {_mc(s['disc_payoff_a_to_end'])} |")
    return out + [""]


XP_LABS = ("gold", "lam=0.0625", "lam=0.125", "lam=0.25", "mu=1", "mu=2", "gold_T", "lam=0.125_T", "mu=1_T")


def xp_md(d: dict) -> List[str]:
    out = ["## 2. Wave-only interventions under alternative objectives (`xp_cf.json.gz`)", "",
           "Advantage of the intervention over the policy's own play, resources, 300-turn window, 16 paired rollouts per "
           "state; mean over the policies of the per-policy state means (± 1.96 SE over states, pooled). Columns: gold "
           "(the current objective), B1 with lambda = 1/16, 1/8, 1/4 per XP, B2 with mu = 1, 2 per level; the _T "
           "columns truncate every rollout at the Part C termination.", ""]
    for stage in ("rescore", "early"):
        if stage not in d:
            continue
        S = d[stage]["summary"]
        title = ("re-scored M4.3b validation states (base seeds 0-2)" if stage == "rescore"
                 else "new laning-phase states (no tower exhausted, both below level 18)")
        out += [f"### 2.{1 if stage == 'rescore' else 2} {title}", ""]
        if stage == "rescore":
            out += [f"Largest deviation of the gold channel from the stored M4.3b advantages: "
                    f"{d[stage]['max_dev_gold_vs_stored']:.2e}.", ""]
        out += ["| opponent, situation | candidate | n | " + " | ".join(XP_LABS) + " | XP lead change | end after (turns) |",
                "|---" * (len(XP_LABS) + 5) + "|"]
        for cell, rows in S.items():
            for c, e in rows.items():
                vals = []
                for lab in XP_LABS:
                    x = e[lab]
                    m = x["mean_specs"] if x["mean_specs"] == x["mean_specs"] else x["mean"]
                    vals.append(f"{m:+.2f} ± {x['ci_states']:.2f}" if x["ci_states"] == x["ci_states"] else f"{m:+.2f}")
                tt = e["term_turn"]
                out.append(f"| {cell.replace('|', ', ')} | {c} | {e['n_states']} | " + " | ".join(vals)
                           + f" | {e['dX'][0]:+.1f} | {NA if tt is None else f'{tt:.0f}'} |")
        out.append("")
    return out


FREEZE_KEYS = (("gold_diff_b", "B gold − A gold"), ("xp_diff_b", "B XP − A XP"), ("value_b_gold_xp", "B value, lambda 1/8"),
               ("dmg_b", "B HP lost (bars)"), ("deaths_b", "B deaths"), ("ganks_b", "B ganked"), ("trades_a", "A trades"),
               ("held", "turns held"), ("pos_b", "B mean cell"), ("plates_b", "B plates"))


def freeze_md(d: dict) -> List[str]:
    out = ["## 3. Freeze and neutral scenarios (`freeze.json`)", "",
           "64 paired seeds; means ± 1.96 SE; from B's side. Cells: scenario | A's policy | B's policy | gap.", ""]
    for w in ("20", "40"):
        out += [f"### 3.{1 if w == '20' else 2} {w} turns", "",
                "| cell | " + " | ".join(lab for _, lab in FREEZE_KEYS) + " |", "|---" * (len(FREEZE_KEYS) + 1) + "|"]
        for k, v in d["cells"].items():
            s = v[w]
            out.append(f"| {k.replace('|', ' / ')} | " + " | ".join(_mc(s[key]) for key, _ in FREEZE_KEYS) + " |")
        out.append("")
    return out


def death_md(d: dict) -> List[str]:
    out = ["## 4. Absence costs (`death.json`)", "",
           "Value lost relative to the unchanged state (16 paired rollouts of 300 turns per state, 8 states per phase "
           "and policy, base seeds 0-2); deaths include the 0.5 bounty; ± 1.96 SE over states.", "",
           "| phase | turn | level | variant | gold | gold, to the end | lambda 1/16 | lambda 1/8 | lambda 1/4 | mu 1 | "
           "lambda 1/8, to the end | opponent plates | focal absent turns |", "|---" * 13 + "|"]
    for phase, row in d["summary"].items():
        for name in ("recall", "death_1", "death_3", "death_5", "death_8", "death_12"):
            e = row[name]
            out.append(f"| {phase} | {row['turn']:.0f} | {row['level']:.1f} | {name} | {_mc(e['cost_gold'])} | "
                       f"{_mc(e['cost_gold_T'])} | {_mc(e['cost_lam=0.0625'])} | {_mc(e['cost_lam=0.125'])} | "
                       f"{_mc(e['cost_lam=0.25'])} | {_mc(e['cost_mu=1'])} | {_mc(e['cost_lam=0.125_T'])} | "
                       f"{_mc(e['plates_opp'])} | {_mc(e['absent_focal'], 1)} |")
    return out + [""]


def run() -> str:
    parts = ["# M4.3c design study: result tables", "",
             "Generated by `python -m lanegame.experiments.m43c report` from the stage files in `results/m43c/`. The "
             "game is the unchanged `GameConfig.m4_lasthit()`; nothing was trained. Interpretation: "
             "docs/m4/m4_3/m4_3c_economic_objective_design.md.", ""]
    for name, fn in (("progression", progression_md), ("xp_cf", xp_md), ("freeze", freeze_md), ("death", death_md)):
        d = load(name)
        parts += fn(d) if d else [f"(stage {name} not run)", ""]
    text = "\n".join(parts)
    write_md("report", text)
    return text


__all__ = ["run"]
