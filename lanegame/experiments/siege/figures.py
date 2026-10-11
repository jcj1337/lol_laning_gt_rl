"""Static figures for the siege study (reads results/siege/*.json; run the other experiments first)."""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..common import out_path  # noqa: E402
from .common import SIEGE  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8e5bd6"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _load(name):
    with open(out_path(name + ".json", SIEGE), encoding="utf-8") as f:
        return json.load(f)


def _style(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, name):
    p = out_path(name, SIEGE)
    fig.tight_layout()
    fig.savefig(p, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return p


def fig_bounce():
    """Released defender blob (leading wave plus queued reinforcement) after a 12-HP crash, by tower rate and
    the attacker's siege behaviour; thinning 1 HP per turn; the attacker's own waves wait at the lane (Slow Push)."""
    rows = [r for r in _load("timing")["bounces"] if r["waves"].startswith("Slow Push") and r["thin"] == 1]
    fig, ax = plt.subplots(figsize=(7.2, 4.0), facecolor=SURFACE)
    _style(ax)
    behaviours = ["disengage (Safe)", "vision every turn", "pressure every turn"]
    towers = [1, 2, 3]
    width = 0.25
    for i, beh in enumerate(behaviours):
        ys = [next((r["released_total_hp"] or 0) for r in rows if r["tower"] == d and r["A"] == beh) for d in towers]
        xs = np.arange(len(towers)) + (i - 1) * width
        ax.bar(xs, ys, width=width, color=SERIES[i], label=beh, edgecolor=SURFACE)
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.2, str(y), ha="center", fontsize=8, color=INK2)
    ax.set_xticks(np.arange(len(towers)))
    ax.set_xticklabels([f"{d} HP/tick" for d in towers])
    ax.set_ylabel("HP of the defender's released blob", color=INK2)
    ax.set_xlabel("tower fire on the crashed mass", color=INK2)
    ax.set_title("Bounce after a 12-HP crash (thinning 1 HP per pressure turn)", color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2)
    ax.set_ylim(0, 18)
    return _save(fig, "fig_bounce_by_tower.png")


def fig_vision_sweep():
    """Equilibrium probability of Establish Vision at siege nodes without vision, by effectiveness and duration."""
    res = _load("vision_sweep")["results"]
    rhos = sorted({r["rho"] for r in res.values()})
    Vs = sorted({r["V"] for r in res.values()})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), facecolor=SURFACE)
    for ax, key, title in ((axes[0], "p_vision_no_vision_nodes", "P(Establish Vision) at siege nodes without vision"),
                           (axes[1], "vision_value_vs_pressure", "mean Q(best Vision) - Q(best Pressure) there")):
        _style(ax)
        for i, rho in enumerate(rhos):
            ys = [next(r[key] for r in res.values() if r["rho"] == rho and r["V"] == V) for V in Vs]
            ys = [y if y is not None else np.nan for y in ys]
            ax.plot(Vs, ys, marker="o", color=SERIES[i], linewidth=2, markersize=5, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, label=f"rho = {rho:.2f}")
        ax.set_xticks(Vs)
        ax.set_xlabel("vision duration (turns)", color=INK2)
        ax.set_title(title, color=INK, loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2)
    axes[1].axhline(0, color=INK2, linewidth=0.8)
    return _save(fig, "fig_vision_sweep.png")


def fig_crash_value():
    """Crash gain (Q best Shove - Q best hold) by crash size, model and opponent, 16 turns left."""
    res = _load("crash_value")["results"]
    models = list(res)
    sizes = ["small (1 wave)", "medium (2 waves)", "large (3 waves)"]
    opp = [("vs_equilibrium", "vs equilibrium B"), ("vs_passive_safe", "vs passive_safe B")]

    def gain(acts):
        sh = max(v["Q"] for lab, v in acts.items() if lab.startswith("SH"))
        hold = max(v["Q"] for lab, v in acts.items() if not lab.startswith("SH"))
        return sh - hold
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), facecolor=SURFACE, sharey=True)
    for ax, (okey, otitle) in zip(axes, opp):
        _style(ax)
        width = 0.8 / len(models)
        for i, m in enumerate(models):
            ys = []
            for h in (4, 8, 12):
                r = res[m][f"{h} HP, 16 turns left"]
                ys.append(gain(r[okey]["actions"]))
            xs = np.arange(3) + (i - (len(models) - 1) / 2) * width
            ax.bar(xs, ys, width=width, color=SERIES[i], label=m, edgecolor=SURFACE)
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_xticks(np.arange(3))
        ax.set_xticklabels(sizes, fontsize=8.5)
        ax.set_title(f"Value of crashing now, {otitle}", color=INK, loc="left", fontsize=10)
        ax.set_ylabel("Q(best Shove) - Q(best hold), resources", color=INK2)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK2)
    return _save(fig, "fig_crash_value.png")


def fig_exploitability():
    """Exploitability of the siege baselines and the decomposition of the best responder's gain."""
    d = _load("equilibrium")["baselines"]
    names = sorted(d, key=lambda n: -d[n]["exploitability"])
    comps = [("farm", "farm"), ("plates", "plates"), ("gank", "gank exposure")]
    fig, ax = plt.subplots(figsize=(8.5, 5.0), facecolor=SURFACE)
    _style(ax, "x")
    y = np.arange(len(names))
    pos = np.zeros(len(names))
    neg = np.zeros(len(names))
    for ci, (key, label) in enumerate(comps):
        if key == "farm":
            g = np.array([-(d[n]["br_path"]["farm"][0] - d[n]["br_path"]["farm"][1]) for n in names])
        elif key == "plates":
            g = np.array([-(d[n]["br_path"]["plates"][0] - d[n]["br_path"]["plates"][1]) for n in names])
        else:
            g = np.array([(d[n]["br_path"]["gank_loss"][0] - d[n]["br_path"]["gank_loss"][1]) for n in names])
        left = np.where(g >= 0, pos, neg + g)
        ax.barh(y, np.abs(g), left=left, height=0.6, color=SERIES[ci], label=label, edgecolor=SURFACE)
        pos += np.where(g >= 0, g, 0)
        neg += np.where(g < 0, g, 0)
    for yi, n, end in zip(y, names, pos):
        ax.text(end + 0.3, yi, f"{d[n]['exploitability']:.1f}", va="center", fontsize=8, color=INK2)
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("best responder's gain per 32-turn game (resources); number = exploitability", color=INK2, fontsize=9)
    ax.set_title("Where a best responder gains against each siege baseline", color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower right")
    return _save(fig, "fig_exploitability_decomposition.png")


def fig_siege_decisions():
    """Equilibrium siege decisions by siege mass and by plates left (default model)."""
    g = _load("siege_options")["grouped"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), facecolor=SURFACE, sharey=True)
    for ax, key, title in ((axes[0], "siege mass", "by crashed mass under the enemy tower"),
                           (axes[1], "plates left", "by plate opportunities pending")):
        _style(ax)
        rows = g[key]
        labels = list(rows)
        x = np.arange(len(labels))
        bottom = np.zeros(len(labels))
        for ci, (k, lab) in enumerate((("p_pressure", "Apply Pressure"), ("p_vision", "Establish Vision"), ("p_disengage", "disengage (Safe)"))):
            vals = np.array([rows[l][k] for l in labels])
            ax.bar(x, vals, bottom=bottom, color=SERIES[ci], label=lab, edgecolor=SURFACE, width=0.7)
            bottom += vals
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
        ax.set_title(title, color=INK, loc="left", fontsize=10)
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("probability (mean over reachable siege nodes)", color=INK2)
    axes[1].legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left", bbox_to_anchor=(0.0, 0.98))
    fig.suptitle("Equilibrium siege decision of player A (K = 32)", color=INK, x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = out_path("fig_siege_decisions.png", SIEGE)
    fig.savefig(p, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return p


def run() -> dict:
    paths = []
    for fn in (fig_bounce, fig_vision_sweep, fig_crash_value, fig_exploitability, fig_siege_decisions):
        try:
            paths.append(fn())
        except FileNotFoundError as err:
            print("skipped", fn.__name__, "-", err)
    print("wrote", *paths, sep="\n  ")
    return {"figures": paths}


if __name__ == "__main__":
    run()
