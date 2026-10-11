"""Learning-curve figure of the RL experiments (results/rl/fig_learning_curves.png), drawn from the result JSON
files. Three small multiples, one y-axis each, at most three series per panel (validated categorical slots
1-3), direct labels at the line ends plus a legend; the numbers themselves are in the Markdown tables."""

from __future__ import annotations

import json
import os

import numpy as np

from .common import OUT

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # categorical slots 1-3 (validated, light surface)
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _load(name):
    with open(os.path.join(OUT, f"{name}.json"), encoding="utf-8") as f:
        return json.load(f)


def _style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.set_xlabel("training steps (thousands)", fontsize=9, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def _line(ax, x, ys, color, label):
    ys = np.asarray(ys, dtype=float)
    m = np.nanmean(ys, axis=0)
    ax.plot(x, m, color=color, linewidth=2, label=label)
    if ys.shape[0] > 1:
        ax.fill_between(x, np.nanmin(ys, axis=0), np.nanmax(ys, axis=0), color=color, alpha=0.15, linewidth=0)
    return x[-1], m[-1], label


def _labels(ax, ends, gap=0.06):
    """Direct labels at the line ends, pushed apart vertically to at least `gap` of the y-range."""
    lo, hi = ax.get_ylim()
    step = gap * (hi - lo)
    ends = sorted(ends, key=lambda e: e[1])
    placed = []
    for x, y, label in ends:
        ty = y if not placed else max(y, placed[-1] + step)
        placed.append(ty)
        ax.annotate(label, (x, y), xytext=(x, ty), textcoords="data", fontsize=8, color=INK, va="center",
                    ha="left", annotation_clip=False)


def run():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a, b, c = _load("m3a_scripted"), _load("m3b_selfplay"), _load("m3a_legacy_exact")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), facecolor=SURFACE)
    # (a) M3a single-opponent training returns, mean over seeds (band = min-max over seeds), 10-update smoothing
    ax = axes[0]
    ends_a = []
    for color, opp in zip(SERIES, ["yield_farmer", "aggressive_trader", "hold_farmer"]):
        runs = a["single"][opp]["runs"]
        n = min(len(r["curve"]) for r in runs)
        x = np.array([p[0] for p in runs[0]["curve"][:n]]) / 1000
        ys = [np.convolve([p[1] for p in r["curve"][:n]], np.ones(10) / 10, mode="same") for r in runs]
        ends_a.append(_line(ax, x[5:-5], [y[5:-5] for y in ys], color, f"  vs {opp}"))
    _style(ax, "M3a: PPO training return against one scripted opponent", "mean episode return (resources)")
    _labels(ax, ends_a)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_xlim(right=ax.get_xlim()[1] * 1.25)
    # (b) M3b held-out payoff against fixed references during self-play (equal skill)
    ax = axes[1]
    eq = [r for r in b["champion"] if r["variant"] == "equal"]
    steps = np.array([s for s, _ in eq[0]["evals"]]) / 1000
    ends_b = [_line(ax, steps, [[e[ref] for _, e in r["evals"]] for r in eq], color, label)
              for color, ref, label in zip(SERIES, ["hold_farmer", "aggressive_trader", "m3a_init"],
                                           ["  vs hold_farmer", "  vs aggressive_trader", "  vs own M3a start"])]
    _style(ax, "M3b: self-play policy vs fixed references (held-out)", "payoff per game (resources)")
    _labels(ax, ends_b)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_xlim(right=ax.get_xlim()[1] * 1.3)
    # (c) legacy K = 16: exact exploitability of self-play snapshots
    ax = axes[2]
    leg = b["legacy"]
    variants = [("realized reward, entropy 0.01", [[p["exploitability"] for p in r["series"]] for r in leg],
                 [p["step"] for p in leg[0]["series"]])]
    for ent in (0.01, 0.05):
        rs = [r for r in c["selfplay"] if r["ent_coef"] == ent]
        variants.append((f"expected reward, entropy {ent}", [[p["exploitability"] for p in r["series"]] for r in rs],
                         [p["step"] for p in rs[0]["series"]]))
    ends_c = []
    for color, (label, ys, st) in zip(SERIES, variants):
        n = min(len(y) for y in ys)
        ends_c.append(_line(ax, np.array(st[:n]) / 1000, [y[:n] for y in ys], color, "  " + label))
    uni = 3.74
    ax.axhline(uni, color=MUTED, linewidth=1, linestyle="--")
    ax.annotate("uniform play (exact)", (20, uni), xytext=(0, 4), textcoords="offset points", fontsize=8, color=MUTED)
    _style(ax, "Legacy K = 16: EXACT exploitability of self-play snapshots", "exploitability (resources)")
    _labels(ax, ends_c)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_xlim(right=ax.get_xlim()[1] * 1.45)
    fig.tight_layout()
    path = os.path.join(OUT, "fig_learning_curves.png")
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    run()
