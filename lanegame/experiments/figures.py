"""Static figures for the report (reads results/*.json; run the other experiments first)."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .common import out_path  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def run() -> dict:
    with open(out_path("horizon_benchmark.json"), encoding="utf-8") as f:
        hb = json.load(f)["results"]
    Ks = sorted(int(k) for k in hb)
    names = list(hb[str(Ks[0])]["baseline_exploitability"])

    # Figure 1: exploitability per turn vs horizon (one line per baseline, direct labels)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    # baselines with identical curves are drawn once with a combined label
    curves = {}
    for n in names:
        ys = tuple(round(hb[str(K)]["baseline_exploitability"][n] / K, 12) for K in Ks)
        curves.setdefault(ys, []).append(n)
    order = sorted(curves.items(), key=lambda kv: -kv[0][-1])
    for i, (ys, ns) in enumerate(order):
        ax.plot(Ks, ys, color=SERIES[i], linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=" = ".join(ns))
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ks)
    ax.set_xticklabels([str(K) for K in Ks])
    ax.set_xlabel("horizon K (agent decisions)", color=INK2)
    ax.set_ylabel("exploitability per decision turn", color=INK2)
    ax.set_title("Baseline exploitability per turn vs. horizon", color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    p1 = out_path("fig_exploitability_vs_horizon.png")
    fig.savefig(p1, dpi=160)
    plt.close(fig)

    # Figure 2: equilibrium values of constructed situations (single series, sorted bars)
    with open(out_path("equilibrium_structure.json"), encoding="utf-8") as f:
        sits = json.load(f)["situations"]
    sits = sorted(sits, key=lambda d: d["value"])
    fig, ax = plt.subplots(figsize=(7.2, 4.4), facecolor=SURFACE)
    _style(ax)
    labels = [d["label"] for d in sits]
    vals = [d["value"] for d in sits]
    ax.barh(range(len(vals)), vals, color=SERIES[0], height=0.6)
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=8.5, color=INK)
    for i, v in enumerate(vals):
        ax.text(v + (0.08 if v >= 0 else -0.08), i, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8, color=INK2)
    ax.set_xlabel("equilibrium value to player A (resources)", color=INK2)
    ax.set_title("Value of lane situations under equilibrium play", color=INK, loc="left", fontsize=11)
    lo, hi = min(vals), max(vals)
    ax.set_xlim(lo - 0.8, hi + 0.8)
    fig.tight_layout()
    p2 = out_path("fig_situation_values.png")
    fig.savefig(p2, dpi=160)
    plt.close(fig)
    print("wrote", p1, p2)
    return {"figures": [p1, p2]}


if __name__ == "__main__":
    run()
