"""Static figures for the revision report (reads results/revision/*.json; run the other experiments first)."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .common import REVISION, out_path  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # categorical slots 1-4, fixed order
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _load(name):
    with open(out_path(name + ".json", REVISION), encoding="utf-8") as f:
        return json.load(f)


def _style(ax, grid_axis="x"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, name):
    p = out_path(name, REVISION)
    fig.tight_layout()
    fig.savefig(p, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return p


def fig_decomposition():
    """Where a best responder's gain comes from: original vs revised model (rung 0 vs rung 7)."""
    ba = _load("before_after")["rungs"]
    rungs = list(ba)
    names = ["aggressive", "uniform", "passive_safe", "freeze_contest", "release_safe", "contest_slowpush",
             "stack_builder"]
    comps = [("farm", "farm"), ("crash", "crash rewards"), ("gank", "gank exposure")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), facecolor=SURFACE, sharey=True)
    for ax, rung, title in ((axes[0], rungs[0], "Original model"), (axes[1], rungs[-1], "Revised model")):
        _style(ax)
        d = ba[rung]["baselines"]
        y = np.arange(len(names))
        pos = np.zeros(len(names))
        neg = np.zeros(len(names))
        for ci, (key, label) in enumerate(comps):
            gain = np.array([-d[n][key] for n in names])      # exploiter's gain from this component
            left = np.where(gain >= 0, pos, neg + gain)
            ax.barh(y, np.abs(gain), left=left, height=0.55, color=SERIES[ci], label=label,
                    edgecolor=SURFACE, linewidth=1.0)
            pos += np.where(gain >= 0, gain, 0)
            neg += np.where(gain < 0, gain, 0)
        tot = [d[n]["exploitability"] for n in names]
        for yi, t, end in zip(y, tot, pos):
            ax.text(end + 0.4, yi, f"{t:.1f}", va="center", ha="left", fontsize=8, color=INK2)
        ax.axvline(0, color=INK2, linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=9, color=INK)
        ax.invert_yaxis()
        ax.set_title(title, color=INK, loc="left", fontsize=10)
        ax.set_xlabel("best responder's gain per 32-turn game (resources)", color=INK2, fontsize=9)
        ax.set_xlim(-6, 30)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper right", ncol=3,
               bbox_to_anchor=(0.99, 0.95))
    fig.suptitle("Exploitability of baseline policies by source of the best responder's gain (number = total)",
                 color=INK, x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = out_path("fig_exploitability_decomposition.png", REVISION)
    fig.savefig(p, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return p


def fig_farm_scale():
    fe = _load("farm_economy")["results"]
    phis = sorted(float(k) for k in fe)
    per_wave = [4 * p for p in phis]
    path = [fe[str(p) if str(p) in fe else repr(p)]["path"]["contest_share"][0] for p in phis]
    nodes = [fe[str(p) if str(p) in fe else repr(p)]["nodes"]["contest"] for p in phis]
    fig, ax = plt.subplots(figsize=(7.2, 4.0), facecolor=SURFACE)
    _style(ax, "y")
    ax.grid(axis="x", visible=False)
    for i, (ys, lab) in enumerate(((path, "equilibrium path"), (nodes, "average over all reachable nodes"))):
        ax.plot(per_wave, ys, color=SERIES[i], linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=lab)
        ax.text(per_wave[-1] + 0.08, ys[-1], lab, va="center", fontsize=8.5, color=INK2)
    ax.set_xlabel("farm value of one fresh wave (resources)", color=INK2)
    ax.set_ylabel("equilibrium probability of Contest", color=INK2)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(-0.1, per_wave[-1] + 1.9)
    ax.set_xticks(per_wave)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower right")
    ax.set_title("Contest vs. Safe as farming becomes more valuable", color=INK, loc="left", fontsize=11)
    return _save(fig, "fig_contest_vs_farm_value.png")


def fig_contest_by_position():
    cs = _load("contest_safe")["grouped"]
    sit = cs["situation"]
    order = ["A's leading wave at -4", "A's leading wave at -3", "A's leading wave at -2", "A's leading wave at -1",
             "A's leading wave at +0", "A's leading wave at +1", "A's leading wave at +2",
             "A has crashed minions under B's tower", "B has crashed minions under A's tower"]
    labels = ["wave at -4 (base)", "wave at -3", "wave at -2 (own tower)", "wave at -1", "wave at 0 (lane)",
              "wave at +1", "wave at +2 (enemy tower)", "own minions crashed\nunder enemy tower",
              "enemy minions crashed\nunder own tower"]
    vals = [sit[k]["p_contest"] for k in order]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), facecolor=SURFACE)
    _style(ax)
    y = np.arange(len(order))
    ax.barh(y, vals, height=0.55, color=SERIES[0])
    for yi, v in zip(y, vals):
        ax.text(v + 0.015, yi, f"{v:.2f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("mean equilibrium probability of Contest (player A, reachable nodes, k <= 24)", color=INK2,
                  fontsize=9)
    ax.set_title("Where the equilibrium contests", color=INK, loc="left", fontsize=11)
    return _save(fig, "fig_contest_by_situation.png")


def fig_horizon():
    hb = _load("horizon_benchmark")["results"]
    Ks = sorted(int(k) for k in hb)
    names = ["aggressive", "passive_safe", "release_safe", "stack_builder"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4), facecolor=SURFACE)
    _style(ax, "y")
    ax.grid(axis="x", visible=False)
    ends = []
    for i, n in enumerate(names):
        ys = [hb[str(K)]["baseline_exploitability"][n] / K for K in Ks]
        ax.plot(Ks, ys, color=SERIES[i], linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=n)
        ends.append([ys[-1], n])
    ends.sort()                                   # spread end labels so they never collide
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.035)
    for y, n in ends:
        ax.text(Ks[-1] * 1.08, y, n, va="center", fontsize=8.5, color=INK2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ks)
    ax.set_xticklabels([str(K) for K in Ks])
    ax.set_xlim(Ks[0] / 1.2, Ks[-1] * 2.2)
    ax.set_xlabel("horizon K (decision turns)", color=INK2)
    ax.set_ylabel("exploitability per turn", color=INK2)
    ax.set_ylim(0, 0.95)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.16))
    ax.set_title("Baseline exploitability per turn vs. horizon (revised model)", color=INK, loc="left", fontsize=11)
    return _save(fig, "fig_exploitability_vs_horizon.png")


def run() -> dict:
    paths = [fig_decomposition(), fig_farm_scale(), fig_contest_by_position(), fig_horizon()]
    print("wrote", *paths, sep="\n  ")
    return {"figures": paths}


if __name__ == "__main__":
    run()
