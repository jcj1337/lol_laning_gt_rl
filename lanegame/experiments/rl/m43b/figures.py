"""Figures of the M4.3b pilot (matplotlib, PNG in results/rl/m43b/). Each figure reads the stage's JSON; stages that
have not run are skipped. Conventions: one fixed colour per condition (never re-assigned), categories inside a panel
in a fixed order, 2 px lines, thin bars, a recessive hairline grid, a legend wherever two or more series share a
panel, labels in text ink (never in the series colour)."""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

from .common import CONDITIONS, OUT, PRIMARY, SEEDS, load
from .references import CURVE_REFS, HEAD_REFS

COND_COLOR = {"base": "#2a78d6", "ent03": "#eb6834", "hist80": "#1baf7a", "mixed": "#eda100"}
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]       # fixed categorical order inside a panel
WAVES, BEHS = ("Hold", "Thin", "Clear"), ("Hold", "Yield", "Trade", "Safe", "Recall")
TEXT, TEXT2, GRID, GRAY, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#9a9994", "#fcfcfb"


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "text.color": TEXT, "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2})
    return plt


def _style(ax, title: str = "", xlabel: str = "", ylabel: str = "", zero: bool = False):
    ax.set_title(title, loc="left", color=TEXT)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    if zero:
        ax.axhline(0, color=GRAY, linewidth=0.8)


def _save(fig, name: str) -> str:
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"{name}.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"wrote {p}", flush=True)
    return p


def _legend(ax, **kw):
    lg = ax.legend(frameon=False, **kw)
    for t in lg.get_texts():
        t.set_color(TEXT2)


def _series(curves: dict) -> Dict[tuple, List[dict]]:
    out: Dict[tuple, List[dict]] = {}
    for p in curves["points"]:
        out.setdefault((p["condition"], p["seed"]), []).append(p)
    for v in out.values():
        v.sort(key=lambda p: p["transitions"])
    return out


def _mean_over_seeds(series: Dict[tuple, List[dict]], cond: str, fn) -> tuple:
    """Common x grid (the transitions of the seeds' points, which coincide) and the mean / per-seed y."""
    xs, ys = None, []
    for (c, s), pts in sorted(series.items()):
        if c != cond:
            continue
        x = np.array([p["transitions"] for p in pts], dtype=float)
        y = np.array([fn(p) for p in pts], dtype=float)
        if xs is None or len(x) < len(xs):
            xs = x
        ys.append((x, y))
    if xs is None:
        return None, None, []
    grid = xs
    aligned = [np.interp(grid, x, y) for x, y in ys]
    return grid, np.mean(aligned, axis=0), aligned


# ---------------------------------------------------------------------------------------------- figures
def fig_curves(curves: dict) -> Optional[str]:
    plt = _mpl()
    series = _series(curves)
    refs = curves["meta"]["refs"]
    fig, axes = plt.subplots(1, len(refs), figsize=(3.6 * len(refs), 3.2), sharey=True)
    for ax, r in zip(np.atleast_1d(axes), refs):
        for c in CONDITIONS:
            grid, mean, per = _mean_over_seeds(series, c, lambda p, r=r: p["vs"][r]["estimate"])
            if grid is None:
                continue
            for y in per:
                ax.plot(grid / 1e6, y, color=COND_COLOR[c], linewidth=0.9, alpha=0.3)
            ax.plot(grid / 1e6, mean, color=COND_COLOR[c], linewidth=2, label=c, solid_capstyle="round")
        _style(ax, f"vs {r}", "agent transitions (millions)", "discounted payoff (resources)" if r == refs[0] else "",
               zero=True)
    _legend(np.atleast_1d(axes)[0], loc="lower right")
    fig.suptitle(f"Learning curves against fixed references (W = {curves['meta']['window']}, "
                 f"{curves['meta']['episodes']} paired episodes; thin lines: seeds, thick: mean)", x=0.01, ha="left",
                 fontsize=10, color=TEXT)
    fig.tight_layout()
    return _save(fig, "fig_curves")


def fig_training(train: dict, curves: Optional[dict]) -> Optional[str]:
    plt = _mpl()
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.1))
    runs = train["runs"]
    for c in CONDITIONS:
        rs = [r for r in runs.values() if r["condition"] == c]
        if not rs:
            continue
        grid = np.array([row[0] for row in rs[0]["curve"]], dtype=float)
        for ax, idx, lab in ((axes[0], 1, "policy entropy (nats)"), (axes[2], 3, "explained variance"),
                             (axes[3], 2, "approximate KL per update")):
            ys = [np.interp(grid, [row[0] for row in r["curve"]], [row[idx] for row in r["curve"]]) for r in rs]
            ax.plot(grid / 1e6, np.mean(ys, axis=0), color=COND_COLOR[c], linewidth=2, label=c)
            for y in ys:
                ax.plot(grid / 1e6, y, color=COND_COLOR[c], linewidth=0.8, alpha=0.3)
    _style(axes[0], "Policy entropy over training", "agent transitions (millions)", "nats")
    _style(axes[2], "Critic explained variance", "agent transitions (millions)", "", zero=True)
    axes[2].set_ylim(-0.5, 1.0)
    _style(axes[3], "Approximate KL per update", "agent transitions (millions)", "")
    axes[3].set_yscale("log")
    if curves is not None:
        series = _series(curves)
        for c in CONDITIONS:
            grid, mean, per = _mean_over_seeds(series, c, lambda p: p["profile"]["norm_entropy"])
            if grid is None:
                continue
            axes[1].plot(grid / 1e6, mean, color=COND_COLOR[c], linewidth=2, label=c)
            for y in per:
                axes[1].plot(grid / 1e6, y, color=COND_COLOR[c], linewidth=0.8, alpha=0.3)
    _style(axes[1], "Entropy / log |legal| on present turns", "agent transitions (millions)", "")
    axes[1].set_ylim(0, 1)
    _legend(axes[0], loc="upper right")
    fig.tight_layout()
    return _save(fig, "fig_training")


def fig_behavior_training(curves: dict) -> Optional[str]:
    plt = _mpl()
    series = _series(curves)
    conds = [c for c in CONDITIONS if any(k[0] == c for k in series)]
    fig, axes = plt.subplots(3, len(conds), figsize=(4.2 * len(conds), 8.4), sharex=True, squeeze=False)
    for j, c in enumerate(conds):
        for i, w in enumerate(WAVES):
            grid, mean, _ = _mean_over_seeds(series, c, lambda p, w=w: p["profile"]["waves"][w])
            axes[0, j].plot(grid / 1e6, mean, color=CAT[i], linewidth=2, label=w)
        for i, b in enumerate(BEHS):
            grid, mean, _ = _mean_over_seeds(series, c, lambda p, b=b: p["profile"]["behaviors"][b])
            axes[1, j].plot(grid / 1e6, mean, color=CAT[i], linewidth=2, label=b)
        for i, (key, lab) in enumerate((("per_game_deaths", "deaths"), ("per_game_recalls", "recalls"),
                                        ("per_game_crashes", "own crashes"))):
            grid, mean, _ = _mean_over_seeds(series, c, lambda p, key=key: p["profile"][key])
            axes[2, j].plot(grid / 1e6, mean, color=CAT[i], linewidth=2, label=lab)
        _style(axes[0, j], f"{c}: wave action shares (self-play)", "", "share of present turns" if j == 0 else "")
        _style(axes[1, j], f"{c}: behaviour shares", "", "share of present turns" if j == 0 else "")
        _style(axes[2, j], f"{c}: events per game and side (460 turns)", "agent transitions (millions)",
               "count" if j == 0 else "")
        axes[0, j].set_ylim(0, 1)
        axes[1, j].set_ylim(0, 1)
    _legend(axes[0, 0], loc="center right")
    _legend(axes[1, 0], loc="center right")
    _legend(axes[2, 0], loc="upper right")
    fig.tight_layout()
    return _save(fig, "fig_behavior_training")


def _self_tables(behavior: dict, cond: str) -> Optional[dict]:
    """Mean of the conditional tables over the seeds of a condition (self-play rows)."""
    tabs = [v["tables"] for k, v in behavior["results"].items()
            if k.startswith(f"final:{cond}:") and k.endswith("|self")]
    if not tabs:
        return None
    out = {}
    for sit in tabs[0]:
        out[sit] = {key: float(np.mean([t[sit][key] for t in tabs])) for key in tabs[0][sit]}
    return out


def fig_conditional(behavior: dict) -> Optional[str]:
    plt = _mpl()
    conds = [c for c in CONDITIONS if _self_tables(behavior, c)]
    if not conds:
        return None
    fig, axes = plt.subplots(2, len(conds), figsize=(5.2 * len(conds), 9.0), squeeze=False)
    sits = [s for s in behavior["situations"] if _self_tables(behavior, conds[0])[s]["n"] >= 50]
    for j, c in enumerate(conds):
        tab = _self_tables(behavior, c)
        y = np.arange(len(sits))
        for row, (cats, keys, title) in enumerate(((WAVES, [f"p_{w}" for w in WAVES], "mean P(wave action)"),
                                                   (BEHS, [f"p_b{b[0]}" for b in BEHS], "mean P(behaviour)"))):
            ax = axes[row, j]
            left = np.zeros(len(sits))
            for i, (cat, key) in enumerate(zip(cats, keys)):
                vals = np.array([tab[s][key] for s in sits])
                ax.barh(y, vals, left=left, color=CAT[i], height=0.62, label=cat, edgecolor=SURFACE, linewidth=1.5)
                left += vals
            ax.set_yticks(y)
            ax.set_yticklabels([f"{s} (n={int(tab[s]['n'])})" for s in sits] if j == 0 else [""] * len(sits),
                               fontsize=7.5)
            ax.invert_yaxis()
            ax.set_xlim(0, 1)
            _style(ax, f"{c}: {title}", "probability mass", "")
            ax.grid(False, axis="y")
            if j == 0:
                _legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=len(cats))
    fig.suptitle("Conditional action probabilities by situation (self-play, present turns, mean over seeds)",
                 x=0.01, ha="left", fontsize=10, color=TEXT)
    fig.tight_layout()
    return _save(fig, "fig_conditional")


def fig_counterfactual(cf: dict) -> Optional[str]:
    plt = _mpl()
    specs = sorted({k.split("|")[0] for k in cf["results"]})
    sits = list(cf["situations"])
    fig, axes = plt.subplots(len(specs), 1, figsize=(11, 2.9 * len(specs)), squeeze=False, sharex=True)
    for ax, spec in zip(axes[:, 0], specs):
        x = np.arange(len(sits))
        for i, w in enumerate(WAVES):
            m, h = [], []
            for key in sits:
                s = cf["results"].get(f"{spec}|{key}", {}).get("summary", {}).get("candidates", {}).get(w)
                m.append(s["mean_adv"] if s else np.nan)
                h.append(s["ci"] if s else np.nan)
            ax.errorbar(x + (i - 1) * 0.22, m, yerr=h, fmt="o", color=CAT[i], markersize=6, capsize=2, linewidth=1.2,
                        label=f"commit {w} (Hold position)", markeredgecolor=SURFACE, markeredgewidth=1)
        share = [cf["results"].get(f"{spec}|{k}", {}).get("summary", {}).get("modal_is_best_share", np.nan) for k in sits]
        ax.set_xticks(x)
        ax.set_xticklabels([f"{k}\nmodal best {sh:.0%}" if np.isfinite(sh) else k for k, sh in zip(sits, share)],
                           fontsize=7.5)
        _style(ax, f"{spec}: paired advantage of committing one wave action over the policy's own move "
                   f"(W = {cf['meta']['window']}, discounted resources; mean over states ± 95%)", "", "resources", zero=True)
    _legend(axes[0, 0], loc="upper right", ncol=3)
    fig.tight_layout()
    return _save(fig, "fig_counterfactual")


def fig_recall(rec: dict) -> Optional[str]:
    plt = _mpl()
    specs = [s for s in rec["death_vs_recall"]]
    spec = f"final:{PRIMARY}:{SEEDS[0]}" if f"final:{PRIMARY}:{SEEDS[0]}" in specs else specs[0]
    classes = list(rec["classes"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.6))
    ax = axes[0]
    x = np.arange(len(classes))
    for i, (key, lab) in enumerate((("recall_vs_stay", "Recall − stay (Hold + Hold)"), ("recall_vs_policy", "Recall − policy's move"),
                                    ("recall_vs_safe", "Recall − walk back (Safe)"))):
        m = [rec["counterfactual"].get(f"{spec}|{c}", {}).get("summary", {}).get(key, {}).get("mean", np.nan) for c in classes]
        h = [rec["counterfactual"].get(f"{spec}|{c}", {}).get("summary", {}).get(key, {}).get("ci", np.nan) for c in classes]
        ax.errorbar(x + (i - 1) * 0.22, m, yerr=h, fmt="o", color=CAT[i], markersize=6, capsize=2, linewidth=1.2,
                    label=lab, markeredgecolor=SURFACE, markeredgewidth=1)
    ns = [rec["counterfactual"].get(f"{spec}|{c}", {}).get("summary", {}).get("n_states", 0) for c in classes]
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c} (n={n})" for c, n in zip(classes, ns)], fontsize=7.5, rotation=20, ha="right")
    _style(ax, f"{spec}: value of Recall at Recall-legal states (W = {rec['meta']['window']}, discounted)", "",
           "resources", zero=True)
    _legend(ax, loc="lower left")
    ax = axes[1]
    dv = rec["death_vs_recall"][spec]["summary"]
    bands = [b for b in ("all", "hp<=40%", "40%<hp<=80%", "hp>80%", "after own crash", "enemy pushing", "gold>=4") if b in dv]
    x = np.arange(len(bands))
    w = 0.34
    ax.bar(x - w / 2, [dv[b]["death_full_cost"] for b in bands], width=w, color=CAT[0], label="death (incl. 0.5 bounty)",
           edgecolor=SURFACE, linewidth=1.5)
    ax.bar(x + w / 2, [dv[b]["recall_full_cost"] for b in bands], width=w, color=CAT[1], label="Recall (incl. the Recall turn)",
           edgecolor=SURFACE, linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n(n={dv[b]['n']})" for b in bands], fontsize=7.5)
    _style(ax, f"{spec}: full cost of a death against a completed Recall from the same state", "",
           "resources (discounted)", zero=True)
    _legend(ax, loc="upper right")
    fig.tight_layout()
    return _save(fig, "fig_recall")


def fig_headline(head: dict) -> Optional[str]:
    plt = _mpl()
    S = head["summary"]
    refs = [r for r in HEAD_REFS if any(r in S["vs_refs"].get(c, {}) for c in CONDITIONS)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    x = np.arange(len(refs) + 1)
    for i, c in enumerate(CONDITIONS):
        if c not in S["vs_refs"]:
            continue
        means = [S["vs_refs"][c][r]["mean"] for r in refs] + [S["vs_initial"].get(c, {}).get("mean", np.nan)]
        ax.plot(x + (i - 1) * 0.2, means, "o", color=COND_COLOR[c], markersize=7, label=c, markeredgecolor=SURFACE,
                markeredgewidth=1)
        for j, r in enumerate(refs):
            for v in S["vs_refs"][c][r]["per_seed"]:
                ax.plot(x[j] + (i - 1) * 0.2, v[0], ".", color=COND_COLOR[c], markersize=4, alpha=0.6)
        for v in S["vs_initial"].get(c, {}).get("per_seed", []):
            ax.plot(x[-1] + (i - 1) * 0.2, v, ".", color=COND_COLOR[c], markersize=4, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(refs + ["initial net"], fontsize=8, rotation=20, ha="right")
    _style(ax, f"Headline values of the finals (W = {head['meta']['window']}, {head['meta']['episodes']} paired episodes; "
               "large: mean over seeds, small: seeds)", "", "discounted payoff to the final (resources)", zero=True)
    _legend(ax, loc="upper left")
    ax = axes[1]
    finals, M = S["cross"]["finals"], np.array(S["cross"]["matrix"], dtype=float)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
    v = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1.0
    im = ax.imshow(M, cmap=cmap, vmin=-v, vmax=v)
    labels = [f.replace("final:", "") for f in finals]
    ax.set_xticks(range(len(finals)))
    ax.set_yticks(range(len(finals)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7.5)
    ax.set_yticklabels(labels, fontsize=7.5)
    for i in range(len(finals)):
        for j in range(len(finals)):
            if i != j and np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center", fontsize=6.5, color=TEXT)
    ax.set_title("Cross-play: payoff to the row final against the column final", loc="left", color=TEXT)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False)
    fig.tight_layout()
    return _save(fig, "fig_headline")


def fig_exploit(ex: dict) -> Optional[str]:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7, 3.4))
    conds = [c for c in CONDITIONS if c in ex["summary"]]
    x = np.arange(len(conds))
    for i, c in enumerate(conds):
        res = [v for k, v in ex["results"].items() if v["condition"] == c]
        for r in res:
            ax.plot(x[i] - 0.12, r["eval"]["head"]["estimate"], "o", color=COND_COLOR[c], markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=1)
            for t in r["transfer"].values():
                ax.plot(x[i] + 0.12, t["estimate"], "s", color=COND_COLOR[c], markersize=5, alpha=0.6)
    ax.plot([], [], "o", color=GRAY, label="attacker vs its own target (W = 1000)")
    ax.plot([], [], "s", color=GRAY, alpha=0.6, label="the same attacker vs the other seeds' finals")
    ax.set_xticks(x)
    ax.set_xticklabels(conds)
    _style(ax, f"Approximate best responses ({ex['meta']['steps']:,} transitions each, warm-started from the target)",
           "", "attacker's discounted payoff (resources)", zero=True)
    _legend(ax, loc="upper right")
    fig.tight_layout()
    return _save(fig, "fig_exploit")


def figures() -> dict:
    out = {}
    train, curves, behavior = load("train"), load("curves"), load("behavior")
    cf, rec, head, ex = load("counterfactual"), load("recall"), load("headline"), load("exploit")
    if curves:
        out["curves"] = fig_curves(curves)
        out["behavior_training"] = fig_behavior_training(curves)
    if train:
        out["training"] = fig_training(train, curves)
    if behavior:
        out["conditional"] = fig_conditional(behavior)
    if cf:
        out["counterfactual"] = fig_counterfactual(cf)
    if rec:
        out["recall"] = fig_recall(rec)
    if head:
        out["headline"] = fig_headline(head)
    if ex:
        out["exploit"] = fig_exploit(ex)
    return out


__all__ = ["figures"]
