"""Tables and figures of the M4.3b follow-up: results/rl/m43b/validation/report.md and fig_val_*.png, assembled from
the stage files (wave_cf, games, horizon, freeze). The interpretation is in docs/rl/m4_3b_validation_report.md."""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

from .common import PRIMARY, SEEDS, fmt_ci, md_table
from .validation import (GAME_OPPONENTS, OPPONENTS, POLICIES, SITUATIONS, SYN_CANDS, SYN_GAPS, SYN_STARTS, VAL_DIR,
                         vload)

CAND_LABEL = {"m:hold": "Hold (wave-only, marginal)", "m:thin": "Thin (wave-only, marginal)",
              "m:clear": "Clear (wave-only, marginal)", "m:slow_push": "slow push (wave-only, marginal)",
              "m:freeze": "freeze controller (wave-only, marginal)", "c:hold": "Hold (wave-only, conditional)",
              "c:slow_push": "slow push (wave-only, conditional)", "c:freeze": "freeze controller (wave-only, conditional)",
              "f:hold": "Hold + Hold (full action)", "f:slow_push": "slow push (full action)",
              "f:freeze": "freeze controller (full action)", "m:reactive_push": "reactive slow push (wave-only, marginal)",
              "m:shove_away": "shove while away (wave-only, marginal)", "c:reactive_push": "reactive slow push (wave-only, conditional)"}
SIT_LABEL = {"neutral": "neutral lane", "enemy_pushing": "enemy front at own −1", "freeze_like": "freeze-like"}
OPP_LABEL = {"self": "the policy itself", "farm": "farm (pure last-hitter)", "push_punish": "push_punish (reactive slow push)",
             "clear_punish": "clear_punish (Clear every turn)", "freeze_punish": "freeze_punish (threshold freeze)",
             "attacker": "its own attacker"}


def _cell(s: Optional[dict]) -> str:
    if not s:
        return "n/a"
    return f"{fmt_ci(s['mean_adv'], s['ci'])} ({s['share_pos']:.0%}/{s['share_neg']:.0%})"


# ---------------------------------------------------------------------------------------------- sections
def sec_wave_cf(wc: dict) -> List[str]:
    L = ["## 1. Wave-only interventions at sampled states (`wave_cf.json`)", "",
         f"W = {wc['meta']['window']} turns discounted, {wc['meta']['rollouts']} paired rollouts per candidate, up to "
         f"{wc['meta']['states']} sampled reachable states per cell (from games of the policy against the opponent named "
         "in the cell, in the situation named in the cell). Entry = Q(candidate) − Q(policy's own play), mean over states "
         "± 95% (states are the unit), then (share of states with a significant gain / loss). Candidates: m: = wave-only, "
         "marginal rule; c: = wave-only, conditional rule; f: = the original full-action plan of M4.3b §7.2 on the same "
         f"states and rollout streams. Plan lengths: {wc['meta']['plan_turns']} turns.", ""]
    cands = wc["candidates"]
    for opp in wc["opponents"]:
        L += [f"### 1.{wc['opponents'].index(opp) + 1}. Opponent: {OPP_LABEL.get(opp, opp)}", ""]
        rows = []
        for spec in wc["policies"]:
            for key in wc["situations"]:
                r = wc["results"].get(f"{spec}|{opp}|{key}")
                if r is None:
                    continue
                s = r["summary"]
                rows.append([spec, SIT_LABEL.get(key, key), s["n_states"]] + [_cell(s.get(c)) for c in cands])
        L += [md_table(["policy", "situation", "states"] + cands, rows), ""]
        # across the baseline seeds
        rows = []
        for key in wc["situations"]:
            a = wc["across_seeds"].get(f"{opp}|{key}", {})
            if a:
                rows.append([SIT_LABEL.get(key, key)] + [fmt_ci(a[c]["mean"], a[c]["ci_seeds"]) + f" (n={a[c]['n_seeds']})"
                                                         if c in a else "n/a" for c in cands])
        if rows:
            L += [f"Across the three `{PRIMARY}` training seeds (seeds are the unit; mean ± 1.96 SE, indicative at n = 3):", "",
                  md_table(["situation"] + cands, rows), ""]
    # the confound: full-action minus wave-only on the same states
    L += ["### 1.5. The confound: full-action plan minus the wave-only plan on the same states", "",
          "Positive = the full-action plan did better than its wave-only counterpart; negative = forcing Hold position "
          "cost that much on top of the wave management (mean over states ± 95%, paired per state).", ""]
    rows = []
    for opp in wc["opponents"]:
        for key in wc["situations"]:
            for spec in wc["policies"]:
                r = wc["results"].get(f"{spec}|{opp}|{key}")
                if r is None:
                    continue
                row = [spec, OPP_LABEL.get(opp, opp).split(" (")[0], SIT_LABEL.get(key, key)]
                for k in ("hold", "slow_push", "freeze"):
                    d = np.array([st["adv"][f"f:{k}"][0] - st["adv"][f"m:{k}"][0] for st in r["states"]
                                  if f"f:{k}" in st["adv"] and f"m:{k}" in st["adv"]])
                    row.append(fmt_ci(float(d.mean()), float(1.96 * d.std(ddof=1) / np.sqrt(len(d)))) if len(d) > 1 else "n/a")
                rows.append(row)
    L += [md_table(["policy", "opponent", "situation", "f:hold − m:hold", "f:slow_push − m:slow_push", "f:freeze − m:freeze"], rows), ""]
    return L


def sec_games(g: dict) -> List[str]:
    L = ["## 2. Whole-game value of a wave rule with adaptive behaviour (`games.json`)", "",
         f"W = {g['meta']['window']} turns discounted from s0, {g['meta']['episodes']} paired episodes (both seats), the "
         "policy with its wave component forced by the rule (behaviour, objectives and Recall as the policy decides) "
         "against each opponent. Entry = value(rule) − value(unmodified policy) on the same episode seeds, ± 95% paired. "
         "`none` row = the unmodified policy's value against that opponent.", ""]
    for spec in g["policies"]:
        rows = []
        for opp in g["opponents"]:
            cell = g["diffs"].get(f"{spec}|{opp}")
            if cell is None:
                continue
            rows.append([OPP_LABEL.get(opp, opp), fmt_ci(cell["none"]["mean"], cell["none"]["ci"])]
                        + [fmt_ci(cell[v]["diff"], cell[v]["ci"]) if v in cell else "n/a" for v in g["variants"][1:]])
        L += [f"### 2.{g['policies'].index(spec) + 1}. {spec}", "",
              md_table(["opponent", "none (value)"] + g["variants"][1:], rows), ""]
    rows = []
    for opp in g["opponents"]:
        a = g["across_seeds"].get(opp)
        if a:
            rows.append([OPP_LABEL.get(opp, opp)] + [fmt_ci(a[v]["mean"], a[v]["ci_seeds"]) if v in a else "n/a"
                                                     for v in g["variants"][1:]])
    if rows:
        L += [f"Across the three `{PRIMARY}` seeds (seeds are the unit):", "", md_table(["opponent"] + g["variants"][1:], rows), ""]
    return L


def sec_horizon(h: dict) -> List[str]:
    marks = [str(m) for m in h["meta"]["marks"]]
    tb = h["meta"]["tail_bounds"]
    L = ["## 3. Horizon check: the same rollouts at W = " + " / ".join(marks) + " (`horizon.json`)", "",
         f"The wave_cf states and rollout roots, rolled out for {h['meta']['window']} turns; the discounted sum is read at "
         f"each mark from the same rollout (the W = {marks[0]} column therefore equals the wave_cf estimate exactly). "
         "Guaranteed truncation bounds |J − J_W| ≤ γ^W Ω: " + ", ".join(f"W = {m}: {tb[m]:.3f}" for m in marks) +
         ". 'tail' = the paired difference accumulated between the first and the last mark; 'flips' = states whose "
         "estimate changes sign between them.", ""]
    rows = []
    for spec in h["policies"]:
        for opp in h["opponents"]:
            for key in h["situations"]:
                r = h["results"].get(f"{spec}|{opp}|{key}")
                if r is None:
                    continue
                s = r["summary"]
                for c in r["candidates"]:
                    if c not in s:
                        continue
                    rows.append([spec, OPP_LABEL.get(opp, opp).split(" (")[0], SIT_LABEL.get(key, key), c, s["n_states"]]
                                + [fmt_ci(s[c][m]["mean_adv"], s[c][m]["ci"]) for m in marks]
                                + [fmt_ci(s[c]["tail"]["mean_adv"], s[c]["tail"]["ci"]), s[c]["sign_changes"]])
    L += [md_table(["policy", "opponent", "situation", "candidate", "states"] + [f"W = {m}" for m in marks] + ["tail", "flips"], rows), ""]
    rows = []
    for opp in h["opponents"]:
        for key in h["situations"]:
            a = h["across_seeds"].get(f"{opp}|{key}", {})
            for c, d in a.items():
                rows.append([OPP_LABEL.get(opp, opp).split(" (")[0], SIT_LABEL.get(key, key), c]
                            + [fmt_ci(d[m]["mean"], d[m]["ci_seeds"]) if m in d else "n/a" for m in marks]
                            + [fmt_ci(d["tail"]["mean"], d["tail"]["ci_seeds"]) if "tail" in d else "n/a"])
    if rows:
        L += [f"Across the three `{PRIMARY}` seeds:", "",
              md_table(["opponent", "situation", "candidate"] + [f"W = {m}" for m in marks] + ["tail"], rows), ""]
    return L


def sec_freeze(f: dict) -> List[str]:
    L = ["## 4. Freeze states: classification of the sampled freeze-like states (`freeze.json`)", "",
         f"Each freeze-like state of wave_cf is classified by rolling the wave-only freeze controller for {f['meta']['watch']} "
         "turns against a Holding opponent (farm) and counting the turns the enemy front stands at own −1 with HP > 0: "
         f"maintainable = held ≥ {f['meta']['held']} of {f['meta']['watch']}; favourable = maintainable and the focal "
         "champion at least even in attack power and at ≥ 50% HP; otherwise superficial. The freeze advantages are the "
         "wave_cf values of those states (W = 300, against the opponent of the cell).", ""]
    rows = []
    for key, rs in sorted(f["classes"].items()):
        spec, opp, _ = key.split("|")
        for r in rs:
            rows.append([spec, opp, r["index"], f"{r['held_vs_farm']:.1f}", f"{r['held_vs_self']:.1f}", f"{r['power']:.2f}",
                         f"{r['hp_frac']:.2f}", r["en_fh"], r["own_fh"], r["class"],
                         fmt_ci(*r["adv_m_freeze"]) if r["adv_m_freeze"] else "n/a",
                         fmt_ci(*r["adv_f_freeze"]) if r["adv_f_freeze"] else "n/a",
                         fmt_ci(*r["adv_m_hold"]) if r["adv_m_hold"] else "n/a"])
    L += [md_table(["policy", "opponent", "state", "held vs farm", "held vs self", "power", "HP", "enemy front HP",
                    "own front HP", "class", "m:freeze", "f:freeze", "m:hold"], rows), ""]
    rows = []
    for opp, d in f["class_summary"].items():
        for cls, cd in d.items():
            rows.append([OPP_LABEL.get(opp, opp).split(" (")[0], cls, cd["n"]]
                        + [fmt_ci(cd[c]["mean"], cd[c]["ci"]) if c in cd else "n/a"
                           for c in ("adv_m_freeze", "adv_c_freeze", "adv_f_freeze", "adv_m_hold")])
    L += ["By class (states pooled over the policies; mean ± 95%):", "",
          md_table(["opponent", "class", "states", "m:freeze", "c:freeze", "f:freeze", "m:hold"], rows), ""]
    L += ["## 5. SYNTHETIC sensitivity: constructed starts with assigned power gaps (`freeze.json`, block `synthetic`)", "",
          f"**{f['label_synthetic']}.** The neutral (k = 9) and freeze (k = 51) starts of the M4.2e-B harness "
          "(`zoning_natural.starts`), both champions at the same XP, with `with_gap` applied (equal; moderate = 16 vs 4 "
          "resources invested; large = 50 vs 0; levels = +3 levels; large_levels = both): the focal champion is the one "
          "ahead or the one behind. W = 300, 16 paired rollouts, the candidates against each opponent; the entry is the "
          f"mean over the three `{PRIMARY}` seeds of the per-seed paired advantage ± 1.96 SE over seeds.", ""]
    for start in SYN_STARTS:
        rows = []
        for gap in SYN_GAPS:
            for focal in ("ahead", "behind"):
                for opp in OPPONENTS:
                    cell = f["synthetic_summary"].get(f"{start}|{gap}|{focal}|{opp}")
                    if cell is None:
                        continue
                    rows.append([gap, focal, f"{cell['power']:.2f}", opp]
                                + [fmt_ci(cell[c]["mean"], cell[c]["ci_seeds"]) for c in SYN_CANDS])
        L += [f"### 5.{SYN_STARTS.index(start) + 1}. Start `{start}`", "",
              md_table(["gap", "focal is", "power ratio", "opponent"] + list(SYN_CANDS), rows), ""]
    return L


# ---------------------------------------------------------------------------------------------- figures
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


COLORS = {"m": "#2a6f97", "c": "#61a5c2", "f": "#c0392b"}


def fig_wave_cf(wc: dict) -> Optional[str]:
    plt = _mpl()
    opps, sits = wc["opponents"], wc["situations"]
    cands = [c for c in wc["candidates"]]
    fig, axes = plt.subplots(len(sits), len(opps), figsize=(4.2 * len(opps), 3.0 * len(sits)), squeeze=False, sharey=True)
    for i, key in enumerate(sits):
        for j, opp in enumerate(opps):
            ax = axes[i, j]
            a = wc["across_seeds"].get(f"{opp}|{key}", {})
            x = np.arange(len(cands))
            m = [a[c]["mean"] if c in a else np.nan for c in cands]
            h = [a[c]["ci_seeds"] if c in a else np.nan for c in cands]
            col = [COLORS[c[0]] for c in cands]
            ax.bar(x, m, yerr=h, color=col, capsize=2, width=0.7)
            ax.axhline(0, color="#444", linewidth=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(cands, rotation=60, ha="right", fontsize=7)
            ax.set_title(f"{SIT_LABEL.get(key, key)} vs {opp}", fontsize=9)
            if j == 0:
                ax.set_ylabel("Q(plan) − Q(policy)")
    fig.suptitle(f"Wave-only (m: marginal, c: conditional) and full-action (f:) plans, W = {wc['meta']['window']}; mean over the "
                 f"{PRIMARY} seeds ± 1.96 SE", fontsize=10)
    fig.tight_layout()
    p = os.path.join(VAL_DIR, "fig_val_wave_cf.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def fig_games(g: dict) -> Optional[str]:
    plt = _mpl()
    variants = g["variants"][1:]
    opps = [o for o in g["opponents"] if o in g["across_seeds"]]
    fig, ax = plt.subplots(figsize=(1.6 * len(variants) + 2, 4.2))
    w = 0.8 / max(len(opps), 1)
    cmap = plt.get_cmap("tab10")
    for j, opp in enumerate(opps):
        a = g["across_seeds"][opp]
        m = [a[v]["mean"] if v in a else np.nan for v in variants]
        h = [a[v]["ci_seeds"] if v in a else np.nan for v in variants]
        ax.bar(np.arange(len(variants)) + (j - len(opps) / 2 + 0.5) * w, m, yerr=h, width=w, color=cmap(j), capsize=2, label=opp)
    ax.axhline(0, color="#444", linewidth=0.8)
    ax.set_xticks(np.arange(len(variants)))
    ax.set_xticklabels(variants, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("value(rule) − value(policy)")
    ax.set_title(f"Whole-game value of a wave rule with adaptive behaviour, W = {g['meta']['window']} (mean over {PRIMARY} seeds ± 1.96 SE)",
                 fontsize=9)
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    p = os.path.join(VAL_DIR, "fig_val_games.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def fig_horizon(h: dict) -> Optional[str]:
    plt = _mpl()
    marks = [str(m) for m in h["meta"]["marks"]]
    cells = [(o, k) for o in h["opponents"] for k in h["situations"] if f"{o}|{k}" in h["across_seeds"]]
    fig, axes = plt.subplots(1, len(cells), figsize=(3.4 * len(cells), 3.6), squeeze=False, sharey=True)
    for ax, (opp, key) in zip(axes[0], cells):
        a = h["across_seeds"][f"{opp}|{key}"]
        for c, d in a.items():
            m = [d[mk]["mean"] for mk in marks if mk in d]
            e = [d[mk]["ci_seeds"] for mk in marks if mk in d]
            ax.errorbar([int(mk) for mk in marks if mk in d], m, yerr=e, marker="o", capsize=2, label=c, linewidth=1.2)
        ax.axhline(0, color="#444", linewidth=0.8)
        ax.set_title(f"{SIT_LABEL.get(key, key)} vs {opp}", fontsize=9)
        ax.set_xlabel("window W (turns)")
    axes[0, 0].set_ylabel("Q(plan) − Q(policy)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"The same rollouts read at each window (mean over {PRIMARY} seeds ± 1.96 SE)", fontsize=10)
    fig.tight_layout()
    p = os.path.join(VAL_DIR, "fig_val_horizon.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def fig_synthetic(f: dict) -> Optional[str]:
    plt = _mpl()
    fig, axes = plt.subplots(len(SYN_STARTS), len(OPPONENTS), figsize=(4.0 * len(OPPONENTS), 3.0 * len(SYN_STARTS)),
                             squeeze=False, sharey=True)
    combos = [(g, fo) for g in SYN_GAPS for fo in ("ahead", "behind") if not (g == "equal" and fo == "behind")]
    cmap = plt.get_cmap("tab10")
    for i, start in enumerate(SYN_STARTS):
        for j, opp in enumerate(OPPONENTS):
            ax = axes[i, j]
            w = 0.8 / len(SYN_CANDS)
            for ci, c in enumerate(SYN_CANDS):
                m = [f["synthetic_summary"].get(f"{start}|{g}|{fo}|{opp}", {}).get(c, {}).get("mean", np.nan) for g, fo in combos]
                ax.bar(np.arange(len(combos)) + (ci - len(SYN_CANDS) / 2 + 0.5) * w, m, width=w, color=cmap(ci), label=c if (i, j) == (0, 0) else None)
            ax.axhline(0, color="#444", linewidth=0.8)
            ax.set_xticks(np.arange(len(combos)))
            ax.set_xticklabels([f"{g}\n{fo}" for g, fo in combos], fontsize=6.5)
            ax.set_title(f"SYNTHETIC {start} vs {opp}", fontsize=9)
    axes[0, 0].legend(fontsize=7)
    axes[0, 0].set_ylabel("Q(plan) − Q(policy)")
    fig.suptitle(f"Constructed starts with assigned power gaps (synthetic sensitivity; mean over {PRIMARY} seeds)", fontsize=10)
    fig.tight_layout()
    p = os.path.join(VAL_DIR, "fig_val_synthetic.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------------------------- report
def validation_report() -> str:
    wc, g, h, f = vload("wave_cf"), vload("games"), vload("horizon"), vload("freeze")
    L = ["# M4.3b follow-up: strategic validation of the wave-management conclusions (results tables)", "",
         "Generated by `python -m lanegame.experiments.rl.m43b val_report` from the stage files in this directory; the "
         "interpretation is in `docs/rl/m4_3b_validation_report.md`. Commit: `"
         + (wc or g or h or f or {}).get("meta", {}).get("commit", "unknown")[:10] + "`.", ""]
    figs = []
    if wc:
        L += sec_wave_cf(wc)
        figs.append(fig_wave_cf(wc))
    if g:
        L += sec_games(g)
        figs.append(fig_games(g))
    if h:
        L += sec_horizon(h)
        figs.append(fig_horizon(h))
    if f:
        L += sec_freeze(f)
        figs.append(fig_synthetic(f))
    L += ["## Figures", ""] + [f"- `{os.path.basename(p)}`" for p in figs if p] + [""]
    text = "\n".join(L).rstrip() + "\n"
    os.makedirs(VAL_DIR, exist_ok=True)
    p = os.path.join(VAL_DIR, "report.md")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"wrote {p}", flush=True)
    return text


__all__ = ["validation_report"]
