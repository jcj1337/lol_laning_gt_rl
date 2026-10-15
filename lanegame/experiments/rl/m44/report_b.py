"""results/m44b/report.md and results/m44b/figures/*.png: the tables and figures of the M4.4b training report,
generated from train_main.json, eval_main.json.gz, extra_main.json(.gz), traces_main.json and the run histories.

Statistical unit. Between-environment claims use the three training seeds as replicates: per-seed values are
summarised by their mean and a t interval with 2 degrees of freedom (t = 4.30), shown as "mean [lo, hi]" next to
the three seed values. Intervals inside one seed (evaluation games, counterfactual states) are 95% normal intervals
over that seed's evaluation units; they describe one trained network, not the training procedure.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..common import md_table
from .actions import BEH_NAMES, WAVE_NAMES
from .common import ENV_ORDER, OUT_B, SEEDS, history, load_json, write_md

T2 = 4.303                                  # t(0.975, 2 df)
ENV_COLOR = {"four": "#2a78d6", "econ": "#eb6834", "aggro": "#1baf7a"}       # reference palette slots 1-3
WAVE_COLOR = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")                     # slots 1-4 (adjacent-validated)
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
FIG = os.path.join(OUT_B, "figures")
NL = chr(10)                                # the line break inside figure titles


# ---------------------------------------------------------------------------------------------- formatting
def finite(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and np.isfinite(x)


def f2(x, d=2) -> str:
    return f"{x:.{d}f}" if finite(x) else "–"


def sg(x, d=2) -> str:
    return f"{x:+.{d}f}" if finite(x) else "–"


def pct(x, d=0) -> str:
    return f"{100 * x:.{d}f}%" if finite(x) else "–"


def pm(m, h, d=2) -> str:
    if not finite(m):
        return "–"
    return f"{m:+.{d}f} ± {h:.{d}f}" if finite(h) else f"{m:+.{d}f}"


def seed_stats(vals: Sequence[float]) -> Tuple[float, float, float]:
    """(mean, lo, hi): t interval over training seeds (2 df for three seeds)."""
    v = np.asarray([x for x in vals if finite(x)], dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(v) == 1:
        return float(v[0]), float("nan"), float("nan")
    if len(v) == 3:
        t = T2
    else:
        from scipy import stats as _st
        t = float(_st.t.ppf(0.975, len(v) - 1))
    h = t * v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean()), float(v.mean() - h), float(v.mean() + h)


def seeds_cell(vals: Sequence[float], fmt=sg, d=2) -> str:
    """'a / b / c (mean [lo, hi])' over the three seeds."""
    m, lo, hi = seed_stats(vals)
    body = " / ".join(fmt(x, d) if fmt is not pct else pct(x) for x in vals)
    if not finite(m):
        return body
    if fmt is pct:
        return f"{body} ({pct(m)} [{pct(lo)}, {pct(hi)}])" if finite(lo) else f"{body} ({pct(m)})"
    return f"{body} ({fmt(m, d)} [{fmt(lo, d)}, {fmt(hi, d)}])" if finite(lo) else f"{body} ({fmt(m, d)})"


def short(spec: str) -> str:
    p = spec.split(":")
    if p[0] == "final":
        return f"{p[2]}_s{p[3]}"
    if p[0] == "snap":
        return f"{p[2]}_s{p[3]}@{int(p[4]) / 1e6:.2f}M"
    if p[0] == "initial":
        return f"{p[1]}_s{p[2]}@0"
    return spec


# ---------------------------------------------------------------------------------------------- data access
class Data:
    def __init__(self, phase: str = "main", src: str = OUT_B):
        self.phase = phase
        self.train = load_json(f"train_{phase}", src)
        self.eval = load_json(f"eval_{phase}", src)
        self.extra = load_json(f"extra_{phase}", src)
        self.traces = load_json(f"traces_{phase}", src)
        self.recall = load_json(f"recall_{phase}", src)
        self.train_ext = load_json("train_ext", src) if phase == "main" else None
        self.eval_ext = load_json("eval_ext", src) if phase == "main" else None
        self.envs = [e for e in ENV_ORDER if any(f"{e}_s{s}" in (self.train or {}).get("runs", {}) for s in SEEDS)]

    def spec(self, e: str, s: int) -> str:
        return f"final:{self.phase}:{e}:{s}"

    def beh(self, e: str, s: int, opp: str = "self", start: str = "s0", src: str = "eval",
            spec: Optional[str] = None) -> Optional[dict]:
        """The metric block of one behaviour cell (group natural:s0 or initialized:<start>)."""
        d = (self.eval if src == "eval" else self.extra) or {}
        k = f"{e}|{spec or self.spec(e, s)}|{opp}|{start}"
        v = d.get("behavior", {}).get(k)
        if v is None:
            return None
        g = "natural:s0" if start == "s0" else f"initialized:{start}"
        return v["metrics"].get(g)

    def strength(self, e: str, x: str, y: str, src: str = "eval") -> Optional[dict]:
        d = (self.eval if src == "eval" else self.extra) or {}
        if src == "eval":
            return d.get("strength", {}).get(f"{e}|{x}|{y}")
        for block in ("tournament", "transfer"):
            v = d.get(block, {}).get("results", {}).get(f"{e}|{x}|{y}")
            if v is not None:
                return v
        return None

    def cf(self, e: str, s: int, key: str, protocol: str = "policy") -> Optional[dict]:
        return ((self.eval or {}).get("counterfactual", {})).get(f"{e}|{self.spec(e, s)}|{key}|{protocol}")


def get(d, *path, default=float("nan")):
    for p in path:
        if d is None or not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


# ---------------------------------------------------------------------------------------------- figures
def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                         "legend.frameon": False, "savefig.facecolor": SURFACE, "lines.linewidth": 2})
    return plt


def _save(fig, name: str) -> str:
    os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, name)
    fig.savefig(p, dpi=130, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return p.replace("\\", "/")


def _smooth(y: np.ndarray, k: int = 10) -> np.ndarray:
    if len(y) < k:
        return y
    c = np.convolve(y, np.ones(k) / k, mode="valid")
    return np.concatenate([np.full(k - 1, np.nan), c])


def fig_health(D: Data) -> Optional[str]:
    plt = _style()
    keys = (("entropy", "policy entropy (nats)"), ("approx_kl", "approximate KL"),
            ("explained_variance", "explained variance"), ("value_loss", "value loss (scaled)"),
            ("clipfrac", "clip fraction"), ("grad_norm", "gradient norm (pre-clip)"))
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6))
    for ax, (k, title) in zip(axes.flat, keys):
        for e in D.envs:
            for s in SEEDS:
                try:
                    h = history(D.phase, e, s)
                except FileNotFoundError:
                    continue
                x = np.array([r["agent_transitions"] for r in h]) / 1e6
                y = _smooth(np.array([r[k] for r in h], dtype=float))
                ax.plot(x, y, color=ENV_COLOR[e], lw=1.2, alpha=0.85, label=e if s == SEEDS[0] else None)
        ax.set_title(title, loc="left")
        ax.set_xlabel("agent transitions (millions)")
    fig.suptitle("PPO health over training (moving mean of 10 updates; one line per run)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=3, title="environment (3 seeds each)", bbox_to_anchor=(1.0, 1.06))
    return _save(fig, "fig1_ppo_health.png")


def curve_data(D: Data, src: str = "eval") -> Dict[Tuple[str, int, str], List[Tuple[int, float, float]]]:
    """(env, seed, reference) -> [(transitions, value, 1.96 se)] of the learning curve."""
    out: Dict[Tuple[str, int, str], List[Tuple[int, float, float]]] = {}
    for k, v in (D.eval or {}).get("strength", {}).items():
        e, x, y = k.split("|")
        if x.startswith("snap:") and y.startswith("ref:"):
            p = x.split(":")
            out.setdefault((e, int(p[3]), y[4:]), []).append((int(p[4]), v["estimate"], 1.96 * v["stderr"]))
    for k in out:
        out[k].sort()
    return out


def fig_curves(D: Data) -> Optional[str]:
    cd = curve_data(D)
    if not cd:
        return None
    plt = _style()
    refs = sorted({k[2] for k in cd}, key=lambda r: ("farm4", "shove4", "freeze_hunt4", "sh_vis", "sh_dive").index(r)
                  if r in ("farm4", "shove4", "freeze_hunt4", "sh_vis", "sh_dive") else 9)
    fig, axes = plt.subplots(1, len(refs), figsize=(2.6 * len(refs), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, refs):
        for e in D.envs:
            ys = []
            for s in SEEDS:
                pts = cd.get((e, s, r))
                if not pts:
                    continue
                x = np.array([p[0] for p in pts]) / 1e6
                y = np.array([p[1] for p in pts])
                ax.plot(x, y, color=ENV_COLOR[e], lw=0.8, alpha=0.45)
                ys.append(y)
            if ys:
                n = min(len(y) for y in ys)
                ax.plot(x[:n], np.mean([y[:n] for y in ys], axis=0), color=ENV_COLOR[e], lw=2.2,
                        label=e if r == refs[0] else None)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title(f"vs {r}", loc="left")
        ax.set_xlabel("transitions (M)")
    axes[0].set_ylabel("discounted value to the policy\n(resources per game)")
    axes[0].legend(loc="lower right", title="mean of 3 seeds")
    fig.suptitle("Learning curves against fixed scripted references (thin: single seeds; W = 460, 16 paired seeds)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, "fig2_learning_curves.png")


def tournament_matrix(D: Data, e: str, s: int) -> Optional[Tuple[List[str], np.ndarray, np.ndarray]]:
    ex = D.extra or {}
    players = ex.get("tournament", {}).get("players", {}).get(f"{e}_s{s}")
    if not players:
        return None
    n = len(players)
    M = np.full((n, n), np.nan)
    H = np.full((n, n), np.nan)
    for i, a in enumerate(players):
        for j, b in enumerate(players):
            if i == j:
                M[i, j] = 0
                continue
            v = D.strength(e, a, b, "extra")
            if v is not None:
                M[i, j], H[i, j] = v["estimate"], 1.96 * v["stderr"]
            else:
                v = D.strength(e, b, a, "extra")
                if v is not None:
                    M[i, j], H[i, j] = -v["estimate"], 1.96 * v["stderr"]
    return players, M, H


def _label(p: str) -> str:
    q = p.split(":")
    if q[0] == "initial":
        return "0"
    if q[0] == "final":
        return "2.05M"
    return f"{int(q[4]) / 1e6:.2f}M"


def fig_tournament(D: Data) -> Optional[str]:
    mats = {e: [tournament_matrix(D, e, s) for s in SEEDS] for e in D.envs}
    if not any(m for v in mats.values() for m in v):
        return None
    plt = _style()
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
    fig, axes = plt.subplots(1, len(D.envs), figsize=(4.1 * len(D.envs), 3.9))
    axes = np.atleast_1d(axes)
    for ax, e in zip(axes, D.envs):
        ms = [m for m in mats[e] if m]
        if not ms:
            continue
        players = ms[0][0]
        A = np.nanmean([m[1] for m in ms], axis=0)
        lim = max(1.0, np.nanmax(np.abs(A)))
        ax.imshow(A, cmap=cmap, vmin=-lim, vmax=lim)
        labs = [_label(p) for p in players]
        ax.set_xticks(range(len(labs)), labs, rotation=60, fontsize=7)
        ax.set_yticks(range(len(labs)), labs, fontsize=7)
        ax.grid(False)
        for i in range(len(labs)):
            for j in range(len(labs)):
                if i != j and np.isfinite(A[i, j]):
                    ax.text(j, i, f"{A[i, j]:.0f}", ha="center", va="center", fontsize=6, color=INK)
        ax.set_title(f"{e}: row vs column (mean of seeds)", loc="left")
        ax.set_xlabel("opponent checkpoint")
        ax.set_ylabel("checkpoint")
    fig.suptitle("Snapshot tournament: value to the row checkpoint (resources per game; blue = row wins)", x=0.01,
                 ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, "fig3_snapshot_tournament.png")


WAVE_FIG_STATES = (("all present turns", "all"), ("fronts in contact at the lane cell", "neutral"),
                   ("my wave stronger (window strength +1 or more)", "my wave +1"),
                   ("enemy wave stronger (-1 or less)", "enemy +1"), ("enemy wave much stronger (-4 or less)", "enemy +4"),
                   ("holding a freeze (freeze state, I am the freezer)", "holding freeze"),
                   ("frozen on (freeze state at the enemy tower)", "frozen on"),
                   ("my crash under the enemy tower", "my crash"), ("enemy crash under my tower", "enemy crash"),
                   ("enemy champion absent", "enemy absent"))


def wave_by_state(D: Data, e: str, opp: str = "self", start: str = "s0") -> Dict[str, List[Optional[dict]]]:
    out = {}
    for st, _ in WAVE_FIG_STATES:
        out[st] = [get(D.beh(e, s, opp, start), "wave_management", "by_state", st, default=None) for s in SEEDS]
    return out


def fig_waves(D: Data) -> Optional[str]:
    if not D.eval:
        return None
    plt = _style()
    fig, axes = plt.subplots(len(D.envs), 1, figsize=(10, 2.3 * len(D.envs)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, e in zip(axes, D.envs):
        wb = wave_by_state(D, e)
        labels = [lab for _, lab in WAVE_FIG_STATES]
        bottoms = np.zeros(len(labels))
        for i, wn in enumerate(WAVE_NAMES):
            vals = np.array([np.nanmean([get(r, "chose", wn) for r in wb[st] if r]) if any(wb[st]) else np.nan
                             for st, _ in WAVE_FIG_STATES])
            ax.bar(range(len(labels)), vals, bottom=bottoms, color=WAVE_COLOR[i], width=0.72, edgecolor=SURFACE,
                   linewidth=1.5, label=wn if e == D.envs[0] else None)
            for j, v in enumerate(vals):
                if finite(v) and v >= 0.12:
                    ax.text(j, bottoms[j] + v / 2, f"{100 * v:.0f}", ha="center", va="center", fontsize=7, color=INK)
            bottoms += np.nan_to_num(vals)
        ax.set_ylim(0, 1)
        ax.set_ylabel(e, fontweight="bold")
        ax.set_yticks([0, 0.5, 1], ["0", "50%", "100%"])
        ax.grid(axis="x", visible=False)
        ax.set_xticks(range(len(labels)), labels, fontsize=8)
    axes[0].legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0))
    fig.suptitle("Wave action chosen by decision state (natural self-play, mean of 3 seeds; numbers = % of turns)",
                 x=0.01, y=1.04, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, "fig4_wave_by_state.png")


def fig_freeze(D: Data) -> Optional[str]:
    if not D.eval:
        return None
    plt = _style()
    panels = (("established freezes\nper 1000 champion-turns", lambda b: get(b, "freezes", "established_per_1000_turns")[0]
               if isinstance(get(b, "freezes", "established_per_1000_turns", default=None), list) else float("nan")),
              ("useful denial share\nof established freezes", lambda b: get(b, "freezes", "established", "useful_denial_share")),
              ("Shove share while\nholding a freeze", lambda b: get(b, "wave_management", "by_state",
                                                                    "holding a freeze (freeze state, I am the freezer)",
                                                                    "chose", "Shove")),
              ("Shove share while\nfrozen on", lambda b: get(b, "wave_management", "by_state",
                                                             "frozen on (freeze state at the enemy tower)", "chose", "Shove")))
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.9))
    for ax, (title, fn) in zip(axes, panels):
        for i, e in enumerate(D.envs):
            vals = [fn(D.beh(e, s)) for s in SEEDS]
            m = np.nanmean(vals) if any(finite(v) for v in vals) else np.nan
            ax.bar(i, m, color=ENV_COLOR[e], width=0.6, alpha=0.9)
            ax.scatter([i - 0.12, i, i + 0.12], vals, color=INK, s=14, zorder=3)
        ax.set_xticks(range(len(D.envs)), D.envs)
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Freezes in natural self-play (bar: mean of seeds; dots: seeds 0 / 1 / 2)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    return _save(fig, "fig5_freezes.png")


SNAP_METRICS = (("Shove share\n(all turns)", lambda b: get(b, "wave_management", "frequency_per_game", "Shove")[0]
                 if isinstance(get(b, "wave_management", "frequency_per_game", "Shove", default=None), list) else np.nan),
                ("established freezes\nper 1000 turns", lambda b: get(b, "freezes", "established_per_1000_turns")[0]
                 if isinstance(get(b, "freezes", "established_per_1000_turns", default=None), list) else np.nan),
                ("Shove share\nholding a freeze", lambda b: get(b, "wave_management", "by_state",
                                                              "holding a freeze (freeze state, I am the freezer)",
                                                              "chose", "Shove")),
                ("Vision share when legal\n(no active vision)",
                 lambda b: _obj_share(b, "Vision")),
                ("dives\nper 1000 turns", lambda b: get(b, "vision_tower", "per_1000_turns", "dives")))


def _obj_share(b, which: str) -> float:
    o = get(b, "vision_tower", "objective_choice_when_legal_no_active_vision", default=None)
    if not isinstance(o, dict):
        return float("nan")
    n = sum(v["n"] for v in o.values())
    return sum(v["n"] * v[which] for v in o.values() if v["n"]) / n if n else float("nan")


def snapshot_behavior(D: Data) -> Dict[Tuple[str, int], List[Tuple[int, dict]]]:
    out: Dict[Tuple[str, int], List[Tuple[int, dict]]] = {}
    ex = D.extra or {}
    for k, v in ex.get("behavior", {}).items():
        e, spec, opp, start = k.split("|")
        if opp != "self" or start != "s0":
            continue
        p = spec.split(":")
        n = int(p[4]) if p[0] == "snap" else 2_048_000
        out.setdefault((e, int(p[3])), []).append((n, v["metrics"]["natural:s0"]))
    for k in out:
        out[k].sort(key=lambda x: x[0])
    return out


def fig_snapshot_behavior(D: Data) -> Optional[str]:
    sb = snapshot_behavior(D)
    if not sb:
        return None
    plt = _style()
    fig, axes = plt.subplots(1, len(SNAP_METRICS), figsize=(13, 2.9))
    for ax, (title, fn) in zip(axes, SNAP_METRICS):
        for e in D.envs:
            for s in SEEDS:
                pts = sb.get((e, s))
                if not pts:
                    continue
                ax.plot([p[0] / 1e6 for p in pts], [fn(p[1]) for p in pts], color=ENV_COLOR[e], lw=1.3, marker="o",
                        ms=3.5, alpha=0.85, label=e if (s == SEEDS[0] and title == SNAP_METRICS[0][0]) else None)
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.set_xlabel("transitions (M)")
    axes[0].legend(loc="best")
    fig.suptitle("Behaviour of intermediate snapshots in self-play (16 games x 500 turns each; one line per run)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, "fig6_snapshot_behaviour.png")


CF_KEYS = ("freeze_held", "neutral", "oversized_enemy", "siege", "threatened")


def fig_cf(D: Data) -> Optional[str]:
    if not D.eval:
        return None
    plt = _style()
    fig, axes = plt.subplots(1, len(CF_KEYS), figsize=(13, 2.9), sharey=True)
    for ax, key in zip(axes, CF_KEYS):
        name = None
        for i, e in enumerate(D.envs):
            for j, s in enumerate(SEEDS):
                c = D.cf(e, s, key)
                sm = get(c, "summary", default=None)
                if not isinstance(sm, dict) or not sm.get("n_states"):
                    continue
                name = sm["contrast"]
                m, h = sm["mean_contrast"]
                y = i - (j - 1) * 0.2
                ax.errorbar(m, y, xerr=h if finite(h) else None, fmt="o", color=ENV_COLOR[e], ms=5, lw=1.4, capsize=2)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_yticks(range(len(D.envs)), D.envs)
        ax.set_title(f"{key}\n{name or ''}", loc="left", fontsize=8.5)
        ax.set_xlabel("contrast (resources, 300 turns)")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Counterfactual contrasts at sampled states (seed 0 / 1 / 2 top to bottom; ± 95% over states)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, "fig7_counterfactuals.png")


# ---------------------------------------------------------------------------------------------- sections
def sec_training(D: Data) -> List[str]:
    t = D.train
    if not t:
        return ["## 1. Training", "", "_train_main.json missing_"]
    rows = []
    for e in D.envs:
        for s in SEEDS:
            r = t["runs"].get(f"{e}_s{s}")
            if not r:
                continue
            rows.append([f"{e}_s{s}", f"{r['agent_transitions']:,}", r["updates"], f"{r['elapsed_s'] / 60:.1f}",
                         f"{r['transitions_per_s']:,.0f}",
                         f"{r['entropy_first_last'][0]:.2f} → {r['entropy_first_last'][1]:.2f}",
                         f"{r['explained_variance_first_last10'][1]:.2f}",
                         f"{r['approx_kl_mean']:.4f} / {r['approx_kl_max']:.3f}", f"{r['clipfrac_mean']:.3f}",
                         f"{r['grad_norm_mean']:.2f} / {r['grad_norm_max']:.2f}", f"{r['value_absmax_resources']:.1f}",
                         f"{r['value_s0_resources'][0]:+.2f}", r["snapshots"], r["digest"][:12]])
    tp = t["throughput"]
    out = ["## 1. Training", "",
           md_table(["run", "transitions", "updates", "min", "trans./s", "entropy first → last 10", "EV last 10",
                     "KL mean / max", "clipfrac", "grad mean / max", "max abs V", "V(s0)", "snapshots", "digest"], rows),
           "", f"Wall clock {tp['wall_s'] / 60:.1f} min for {len(t['runs'])} runs on {tp['processes']} processes; "
               f"per-process throughput min / mean / max "
               + " / ".join(f"{x:,.0f}" for x in tp["transitions_per_s_per_process_min_mean_max"])
               + f" agent transitions/s; aggregate {tp['aggregate_transitions_per_s']:,.0f} /s.", "",
           f"Pipeline checks: **{sum(t['checks']['checks'].values())} / {len(t['checks']['checks'])} passed**; "
           f"learning checks: {sum(t['checks']['learning'].values())} / {len(t['checks']['learning'])}."]
    failed = [k for k, v in t["checks"]["checks"].items() if not v] + [k for k, v in t["checks"]["learning"].items() if not v]
    if failed:
        out += ["", "Not passed: " + "; ".join(failed)]
    out += ["", md_table(["seed", "initial digest (four / econ / aggro)", "identical"],
                         [[k, " / ".join(d[:12] for d in v["digests"].values()), v["identical_across_environments"]]
                          for k, v in t["initialization"].items()])]
    if t.get("reproducibility"):
        out += ["", "Re-training at the full budget (`repro_main`):", "",
                md_table(["run", "weights digest", "re-run digest", "identical weights", "identical histories"],
                         [[n, r["digest"][:16], r["repeat_digest"][:16], r["identical_weights"], r["identical_histories"]]
                          for n, r in t["reproducibility"].items()])]
    # late-training drift: windows of 50 updates
    rows = []
    for e in D.envs:
        for s in SEEDS:
            try:
                h = history(D.phase, e, s)
            except FileNotFoundError:
                continue
            def w(k, a, b):
                return float(np.mean([r[k] for r in h[a:b]]))
            rows.append([f"{e}_s{s}"] + [f"{w('entropy', a, b):.3f}" for a, b in ((200, 250), (300, 350), (400, 450), (450, 500))]
                        + [f"{w('explained_variance', a, b):.2f}" for a, b in ((200, 250), (450, 500))]
                        + [f"{w('approx_kl', a, b):.4f}" for a, b in ((200, 250), (450, 500))])
    out += ["", "Late-training windows (means over 50 updates; the learning rate anneals linearly to 0 at update 500):", "",
            md_table(["run", "entropy 201-250", "301-350", "401-450", "451-500", "EV 201-250", "EV 451-500",
                      "KL 201-250", "KL 451-500"], rows)]
    return out


def sec_load(D: Data) -> List[str]:
    L = (D.eval or {}).get("load")
    if not L:
        return []
    rows = [[k, v["digest_verified"], v["matches_training_record"],
             ", ".join(f"{o}: {'refused' if r else 'LOADED'}" for o, r in v["refused_other_envs_without_opt_in"].items()),
             v["trained_on"]] for k, v in L.items()]
    return ["## 2. Loading, digests and the compatibility guard", "",
            md_table(["final", "digest verified", "= training record", "other environments (no opt-in)", "trained on"], rows)]


def sec_integrity(D: Data) -> List[str]:
    I = (D.extra or {}).get("integrity")
    if not I:
        return []
    rows = [[short(k.split("|")[1]), v["observations"], v["nonfinite_legal_logits"], v["nonfinite_values"],
             f"{v['value_absmax_resources']:.1f} / {v['omega']:.0f}", v["values_above_omega"],
             v["observations_with_illegal_mass"], v["reward_asymmetries"]] for k, v in I.items()]
    return ["### Integrity of the finals (self-play, 4 games x 500 turns; supplementary)", "",
            md_table(["final", "observations", "non-finite logits", "non-finite V", "max abs V / Omega", "V > Omega",
                      "obs. with mass on illegal codes", "reward asymmetries"], rows), "",
            "Every sampled action was legal: `LaneEnv.step` raises on an illegal code, and every training and "
            "evaluation game completed."]


REF_ORDER = ("farm4", "thin4", "slowpush4", "shove4", "shove_hunt4", "freeze_hunt4", "grow_crash_recall4", "respect4",
             "sh_vis", "sh_dive")


def sec_strength(D: Data) -> List[str]:
    if not D.eval:
        return []
    out = ["## 3. Policy strength", "",
           "Values are `evaluate_discounted` (from s0, each seed played in both seats on the same chance stream; "
           "payoff to the policy in resources of discounted gold, W = 1000 for references and cross-play, 460 for "
           "history and curves). Per seed: estimate ± 95% over 32 (16) paired seeds. The last column summarises the "
           "three training seeds (mean [t interval, 2 df]).", ""]
    for e in D.envs:
        rows = []
        for r in REF_ORDER:
            vals, cells = [], []
            for s in SEEDS:
                v = D.strength(e, D.spec(e, s), f"ref:{r}")
                vals.append(v["estimate"] if v else np.nan)
                cells.append(pm(v["estimate"], 1.96 * v["stderr"]) if v else "–")
            m, lo, hi = seed_stats(vals)
            rows.append([r] + cells + [f"{sg(m)} [{sg(lo)}, {sg(hi)}]"])
        out += [f"### 3.{D.envs.index(e) + 1} `{e}` finals against the references", "",
                md_table(["reference", "seed 0", "seed 1", "seed 2", "seeds: mean [t interval]"], rows), ""]
    # history and initial
    rows = []
    for e in D.envs:
        for s in SEEDS:
            f = D.spec(e, s)
            cells = []
            for k, v in sorted(((k, v) for k, v in D.eval["strength"].items()
                                if k.startswith(f"{e}|{f}|") and (k.split("|")[2].startswith("snap:")
                                                                  or k.split("|")[2].startswith("initial:"))),
                               key=lambda kv: -1 if kv[0].split("|")[2].startswith("initial") else int(kv[0].split(":")[-1])):
                cells.append(f"{short(k.split('|')[2]).split('@')[1]}: {pm(v['estimate'], 1.96 * v['stderr'], 1)}")
            rows.append([f"{e}_s{s}", "; ".join(cells)])
    out += ["### 3.4 Finals against their own past (initial network and snapshots; W = 460, 16 paired seeds)", "",
            md_table(["final", "value to the final against: checkpoint @ transitions"], rows), ""]
    # cross-play
    rows = []
    for e in D.envs:
        for i, a in enumerate(SEEDS):
            for b in SEEDS[i + 1:]:
                v = D.strength(e, D.spec(e, a), D.spec(e, b))
                if v:
                    rows.append([e, f"s{a} vs s{b}", pm(v["estimate"], 1.96 * v["stderr"])])
    out += ["### 3.5 Cross-play between seeds (value to the first)", "", md_table(["env", "pair", "value"], rows), ""]
    return out


def sec_curves(D: Data) -> List[str]:
    cd = curve_data(D)
    if not cd:
        return []
    refs = sorted({k[2] for k in cd})
    out = ["## 4. Learning curves and convergence", "",
           "Snapshot (every 4th, ~102k transitions apart) against the fixed references, W = 460, 16 paired seeds. "
           "Plateau test per (run, reference): least-squares slope of the value over the last 8 curve points "
           "(≈ 1.25M → 2.05M transitions), in resources per million transitions, with its 95% interval from the "
           "regression; and the change from the mean of points at ≈ 1.0–1.25M to the mean of the last three.", ""]
    rows, slopes = [], {}
    for e in D.envs:
        for s in SEEDS:
            for r in refs:
                pts = cd.get((e, s, r))
                if not pts or len(pts) < 8:
                    continue
                x = np.array([p[0] for p in pts], float) / 1e6
                y = np.array([p[1] for p in pts], float)
                xs, ys = x[-8:], y[-8:]
                A = np.vstack([xs, np.ones_like(xs)]).T
                coef, res, *_ = np.linalg.lstsq(A, ys, rcond=None)
                resid = ys - A @ coef
                se = np.sqrt(resid @ resid / (len(xs) - 2) / ((xs - xs.mean()) ** 2).sum())
                mid = y[(x >= 0.95) & (x <= 1.3)]
                slopes.setdefault((e, r), []).append(coef[0])
                rows.append([f"{e}_s{s}", r, f"{y[0]:+.2f}", f"{mid.mean():+.2f}" if len(mid) else "–",
                             f"{y[-3:].mean():+.2f}", f"{coef[0]:+.2f} ± {2.447 * se:.2f}",
                             "rising" if coef[0] - 2.447 * se > 0 else ("falling" if coef[0] + 2.447 * se < 0 else "flat")])
    out += [md_table(["run", "reference", "first point", "≈1.0–1.25M", "last 3", "slope last 8 (/M)", "trend"], rows), ""]
    rows = []
    for (e, r), v in sorted(slopes.items()):
        m, lo, hi = seed_stats(v)
        rows.append([e, r, " / ".join(f"{x:+.2f}" for x in v), f"{sg(m)} [{sg(lo)}, {sg(hi)}]"])
    out += ["Slope over the last 8 points by environment and reference (seeds as replicates):", "",
            md_table(["env", "reference", "slopes s0 / s1 / s2", "mean [t interval]"], rows), ""]
    return out


def sec_tournament(D: Data) -> List[str]:
    out = []
    rows = []
    for e in D.envs:
        for s in SEEDS:
            tm = tournament_matrix(D, e, s)
            if not tm:
                continue
            players, M, H = tm
            n = len(players)
            later_wins = later_sig_loss = pairs = 0
            for i in range(n):
                for j in range(i + 1, n):
                    pairs += 1
                    # M[j, i] = value to the LATER checkpoint j against the earlier i
                    later_wins += int(M[j, i] > 0)
                    later_sig_loss += int(M[j, i] + H[j, i] < 0)
            fin = M[n - 1, :n - 1]
            prev = M[n - 2, :n - 2]
            rows.append([f"{e}_s{s}", f"{later_wins} / {pairs}", later_sig_loss,
                         " ".join(f"{x:+.1f}" for x in fin), f"{M[n - 1, n - 2]:+.2f} ± {H[n - 1, n - 2]:.2f}",
                         f"{M[n - 1, n - 3]:+.2f} ± {H[n - 1, n - 3]:.2f}"])
    if rows:
        lab = ""
        tm = tournament_matrix(D, D.envs[0], SEEDS[0])
        if tm:
            lab = ", ".join(_label(p) for p in tm[0])
        out = ["## 5. Snapshot tournament (supplementary: is improvement persistent?)", "",
               f"Checkpoints per run: {lab}. Every pair played with W = 460, 16 paired seeds. 'Later wins': pairs in "
               "which the later checkpoint has a positive value against the earlier one; 'later loses (sig.)': its "
               "value is below zero by more than the 95% half-width.", "",
               md_table(["run", "later wins", "later loses (sig.)", "final vs each earlier (0 → 1.79M)",
                         "final vs 1.79M", "final vs 1.54M"], rows), ""]
    return out


def sec_waves(D: Data) -> List[str]:
    if not D.eval:
        return []
    out = ["## 6. Wave management (natural self-play: 32 games x 1000 turns per final, both seats)", ""]
    rows = []
    for e in D.envs:
        for wn in WAVE_NAMES:
            vals = [get(D.beh(e, s), "wave_management", "frequency_per_game", wn) for s in SEEDS]
            vals = [v[0] if isinstance(v, list) else np.nan for v in vals]
            rows.append([e, wn, seeds_cell(vals, pct)])
    out += ["### 6.1 Frequencies (share of present, non-Recall turns)", "",
            md_table(["env", "wave action", "seeds 0 / 1 / 2 (mean [t interval])"], rows), ""]
    hdr = ["state"] + [f"{e}: LH / TN / SL / SH (n)" for e in D.envs]
    rows = []
    for st, lab in WAVE_FIG_STATES:
        row = [lab]
        for e in D.envs:
            wb = wave_by_state(D, e)[st]
            shares = [np.nanmean([get(r, "chose", wn) for r in wb if r]) if any(wb) else np.nan for wn in WAVE_NAMES]
            n = sum(r["n"] for r in wb if r)
            row.append(" / ".join(pct(x) for x in shares) + f" ({n:,})")
        rows.append(row)
    out += ["### 6.2 Chosen wave action by decision state (mean of the 3 seeds; n = turns pooled over seeds)", "",
            md_table(hdr, rows), ""]
    # seed spread of Shove in key states
    rows = []
    for st, lab in WAVE_FIG_STATES:
        row = [lab]
        for e in D.envs:
            wb = wave_by_state(D, e)[st]
            row.append(seeds_cell([get(r, "chose", "Shove") if r else np.nan for r in wb], pct))
        rows.append(row)
    out += ["### 6.3 Shove share by state, per seed", "", md_table(["state"] + D.envs, rows), ""]
    # plate phases
    rows = []
    for e in D.envs:
        for ph in ("attack plates left", "attack plates exhausted"):
            b = [get(D.beh(e, s), "plate_phases", ph, default=None) for s in SEEDS]
            rows.append([e, ph, seeds_cell([get(x, "waves", "Shove") for x in b], pct),
                         seeds_cell([get(x, "crash_rate") for x in b], pct),
                         seeds_cell([get(x, "pressure_when_legal") for x in b], pct),
                         seeds_cell([get(x, "vision_when_legal") for x in b], pct),
                         seeds_cell([get(x, "freeze_state_share") for x in b], pct),
                         seeds_cell([get(x, "earned_per_turn") for x in b], f2, 3),
                         sum(int(get(x, "turns", default=0)) for x in b)])
    out += ["### 6.4 By plate phase (from the tower state)", "",
            md_table(["env", "phase", "Shove", "crash rate", "Pressure when legal", "Vision when legal",
                      "freeze-state share", "income / turn", "turns"], rows), ""]
    rows = []
    for e in D.envs:
        ex = [get(D.beh(e, s), "plate_phases", "attack_plates_exhausted_at_turn", default=None) for s in SEEDS]
        rows.append([e, " / ".join(f2(get(x, "median"), 0) for x in ex), " / ".join(str(get(x, "never", default="–")) for x in ex)])
    out += [md_table(["env", "median turn the enemy plates are exhausted (s0 / s1 / s2)", "games never (of 64 sides)"], rows), ""]
    # intent vs effect
    rows = []
    for e in D.envs:
        for wn in WAVE_NAMES:
            eff = [get(D.beh(e, s), "wave_management", "neutral_action_effect_after_4_turns", wn, default=None) for s in SEEDS]
            rows.append([e, wn, " / ".join(str(get(x, "n", default="–")) for x in eff),
                         seeds_cell([get(x, "moved_forward_rate") for x in eff], pct),
                         seeds_cell([get(x, "gained_strength_rate") for x in eff], pct)])
    out += ["### 6.5 Intent versus effect at a neutral lane (4 turns later)", "",
            md_table(["env", "chosen", "n (s0 / s1 / s2)", "meeting point moved toward the enemy", "my window strength +1 or more"], rows), ""]
    # reply to opponent patterns
    rows = []
    for e in D.envs:
        for pat in ("opponent Thinned 3 turns", "opponent Shoved 3 turns", "opponent Last Hit 3 turns", "otherwise"):
            r = [get(D.beh(e, s), "wave_management", "reply_to_opponent_pattern", pat, default=None) for s in SEEDS]
            n = sum(int(get(x, "n", default=0)) for x in r)
            sh = [np.nanmean([get(x, "shares", wn) for x in r if x]) for wn in WAVE_NAMES]
            rows.append([e, pat, f"{n:,}"] + [pct(v) for v in sh])
    out += ["### 6.6 Reply to an opponent pattern (pooled over seeds)", "",
            md_table(["env", "opponent's last 3 turns", "n", "LH", "TN", "SL", "SH"], rows), ""]
    # crashes
    rows = []
    for e in D.envs:
        c = [get(D.beh(e, s), "wave_management", "crashes", default=None) for s in SEEDS]
        rows.append([e, seeds_cell([get(x, "per_100_turns")[0] if isinstance(get(x, "per_100_turns", default=None), list)
                                    else np.nan for x in c], f2),
                     seeds_cell([get(x, "size_mean") for x in c], f2, 1),
                     " / ".join(str(get(x, "size_classes", default={})) for x in c[:1])])
    out += ["### 6.7 Crashes", "", md_table(["env", "crashes per 100 turns", "mean size (HP)", "size classes (seed 0)"], rows), ""]
    return out


def freeze_rows(D: Data, e: str, opp: str = "self", start: str = "s0", src: str = "eval") -> List[list]:
    b = [D.beh(e, s, opp, start, src) for s in SEEDS]
    f = [get(x, "freezes", default=None) for x in b]
    est = [get(x, "established", default=None) for x in f]

    def c(fn, fmt=f2, d=2):
        return seeds_cell([fn(x) for x in est], fmt, d)
    rows = [
        ["established per 1000 champion-turns", seeds_cell([get(x, "established_per_1000_turns")[0]
                                                            if isinstance(get(x, "established_per_1000_turns", default=None), list)
                                                            else np.nan for x in f], f2)],
        ["freeze-state share (holding / frozen on)", " / ".join(f"{pct(get(x, 'held_share'), 1)} · {pct(get(x, 'held_on_share'), 1)}" for x in f)],
        ["established episodes (n)", " / ".join(str(get(x, "n", default=0)) for x in est)],
        ["sustained (>= 10 turns) episodes (n)", " / ".join(str(get(x, "sustained", "n", default=0)) for x in f)],
        ["length mean / p90 (turns)", " / ".join(f"{f2(get(x, 'length_mean'), 1)} · {f2(get(x, 'length_p90'), 0)}" for x in est)],
        ["useful denial share (freezer out-earned)", c(lambda x: get(x, "useful_denial_share"), pct)],
        ["gold lead of the freezer during the episode", c(lambda x: get(x, "gold_diff")[0] if isinstance(get(x, "gold_diff", default=None), list) else np.nan)],
        ["XP lead of the freezer", c(lambda x: get(x, "xp_diff")[0] if isinstance(get(x, "xp_diff", default=None), list) else np.nan, sg, 1)],
        ["frozen-on XP lost for good per episode", c(lambda x: get(x, "frozen_on_xp_missed")[0] if isinstance(get(x, "frozen_on_xp_missed", default=None), list) else np.nan, f2, 1)],
        ["frozen-on uncollected gold per episode", c(lambda x: get(x, "frozen_on_uncollected")[0] if isinstance(get(x, "frozen_on_uncollected", default=None), list) else np.nan)],
        ["gold lead in the 20 turns after", c(lambda x: get(x, "after_gold_diff")[0] if isinstance(get(x, "after_gold_diff", default=None), list) else np.nan)],
        ["freezer Last Hit / Thin share while holding", c(lambda x: get(x, "freezer_preserving_share")[0] if isinstance(get(x, "freezer_preserving_share", default=None), list) else np.nan, pct)],
        ["freezer Shoved on the last 2 held turns", c(lambda x: get(x, "freezer_shoved_at_end"), pct)],
        ["frozen-on side Shoved on the last 2 held turns", c(lambda x: get(x, "frozen_on_shoved_at_end"), pct)],
        ["frozen-on Safe / Yield share", " / ".join(
            f"{pct(get(x, 'frozen_on_beh', 'Safe')[0] if isinstance(get(x, 'frozen_on_beh', 'Safe', default=None), list) else np.nan)}"
            f" · {pct(get(x, 'frozen_on_beh', 'Yield')[0] if isinstance(get(x, 'frozen_on_beh', 'Yield', default=None), list) else np.nan)}" for x in est)],
    ]
    ends = {}
    for x in est:
        for k, v in (get(x, "end", default={}) or {}).items():
            ends[k] = ends.get(k, 0) + v
    tot = max(sum(ends.values()), 1)
    rows.append(["end (pooled over seeds)", ", ".join(f"{k} {100 * v / tot:.0f}%" for k, v in ends.items() if v)])
    pw = {}
    for x in est:
        for k, v in (get(x, "freezer_power", default={}) or {}).items():
            pw[k] = pw.get(k, 0) + v
    rows.append(["freezer power class (pooled)", ", ".join(f"{k} {v}" for k, v in pw.items())])
    return rows


def sec_freezes(D: Data) -> List[str]:
    if not D.eval:
        return []
    out = ["## 7. Freezes (state-based definition of M4.4a §4.2; established = held >= 4 turns)", ""]
    for e in D.envs:
        out += [f"### 7.{D.envs.index(e) + 1} `{e}`, natural self-play (seeds 0 / 1 / 2)", "",
                md_table(["measure", "value"], freeze_rows(D, e)), ""]
    # initialized freeze
    rows = []
    for e in D.envs:
        b = [D.beh(e, s, "self", "freeze/equal") for s in SEEDS]
        f = [get(x, "freezes", default=None) for x in b]
        rows.append([e, " / ".join(f2(get(x, "established", "length_mean"), 1) for x in f),
                     seeds_cell([get(x, "established", "useful_denial_share") for x in f], pct),
                     seeds_cell([get(x, "wave_management", "by_state", "holding a freeze (freeze state, I am the freezer)",
                                     "chose", "Shove") for x in b], pct),
                     seeds_cell([get(x, "wave_management", "by_state", "frozen on (freeze state at the enemy tower)",
                                     "chose", "Shove") for x in b], pct)])
    out += ["### 7.4 INITIALIZED: the scripted outside-tower freeze start (16 games x 300 turns; not natural play)", "",
            md_table(["env", "episode length mean", "useful denial", "Shove holding", "Shove frozen on"], rows), ""]
    # against freeze references (extra)
    if D.extra:
        rows = []
        for e in D.envs:
            for r in ("thin4", "freeze_hunt4", "respect4"):
                b = [D.beh(e, s, r, "s0", "extra") for s in SEEDS]
                fo = [get(x, "wave_management", "by_state", "frozen on (freeze state at the enemy tower)", default=None) for x in b]
                ho = [get(x, "wave_management", "by_state", "holding a freeze (freeze state, I am the freezer)", default=None) for x in b]
                rows.append([e, r, " / ".join(pct(get(x, "held_on_share"), 1) for x in [get(y, "freezes", default=None) for y in b]),
                             " / ".join(str(get(x, "n", default=0)) for x in fo),
                             seeds_cell([get(x, "chose", "Shove") for x in fo], pct),
                             seeds_cell([get(x, "chose", "Shove") for x in ho], pct),
                             seeds_cell([get(x, "freezes", "established", "useful_denial_share") for x in b], pct)])
        out += ["### 7.5 Against scripted freezers and a conceding farmer (supplementary; s0, 12 seeds x 1000 turns, both seats)", "",
                md_table(["env", "opponent", "frozen-on share of turns", "frozen-on turns (n)", "Shove when frozen on",
                          "Shove when holding", "useful denial of its own freezes"], rows), ""]
    return out


def sec_vision(D: Data) -> List[str]:
    if not D.eval:
        return []
    out = ["## 8. Vision, tower pressure and dives (natural self-play)", ""]
    rows = []
    for e in D.envs:
        b = [D.beh(e, s) for s in SEEDS]
        p1 = [get(x, "vision_tower", "per_1000_turns", default=None) for x in b]
        rows.append([e,
                     seeds_cell([_obj_share(x, "Pressure") for x in b], pct),
                     seeds_cell([_obj_share(x, "Vision") for x in b], pct),
                     seeds_cell([get(x, "vision_tower", "extended_pressure_runs", "with_vision_share") for x in b], pct),
                     " / ".join(str(get(x, "vision_tower", "extended_pressure_runs", "n", default=0)) for x in b),
                     seeds_cell([get(x, "vision_established") for x in p1], f2, 1),
                     seeds_cell([get(x, "plates") for x in p1], f2, 1),
                     seeds_cell([get(x, "ganks") for x in p1], f2, 1)])
    out += ["### 8.1 Objectives (siege turns: own crash under the enemy tower, tower reachable)", "",
            md_table(["env", "Pressure when legal (no active vision)", "Vision when legal (no active vision)",
                      "extended Pressure runs with vision", "runs (n)", "vision set / 1000", "plates / 1000",
                      "ganks / 1000"], rows), ""]
    rows = []
    for e in D.envs:
        for k in ("vision", "no vision"):
            fx = [get(D.beh(e, s), "vision_tower", "forward_exposure", k, default=None) for s in SEEDS]
            rows.append([e, k, " / ".join(str(get(x, "n", default=0)) for x in fx),
                         seeds_cell([get(x, "gank_p") for x in fx], f2, 3),
                         seeds_cell([get(x, "realized_gank_bars") for x in fx], f2, 3),
                         seeds_cell([get(x, "death_rate") for x in fx], f2, 3)])
    out += ["### 8.2 Forward exposure (turn ended at own +1 / +2)", "",
            md_table(["env", "vision", "turns (s0 / s1 / s2)", "gank probability", "realized gank damage (bars)",
                      "death rate"], rows), ""]
    rows = []
    for e in D.envs:
        b = [D.beh(e, s) for s in SEEDS]
        p1 = [get(x, "vision_tower", "per_1000_turns", default=None) for x in b]
        rows.append([e, seeds_cell([get(x, "dives") for x in p1], f2, 1), seeds_cell([get(x, "busy_dives") for x in p1], f2, 1),
                     seeds_cell([get(x, "busy_dives") / get(x, "dives") if finite(get(x, "dives")) and get(x, "dives") > 0
                                 else np.nan for x in p1], pct),
                     seeds_cell([get(x, "tower_adjacent_trades") for x in p1], f2, 1),
                     seeds_cell([get(x, "expected_tower_shots") for x in p1], f2, 1),
                     seeds_cell([get(x, "expected_tower_shots_busy") for x in p1], f2, 1)])
    out += ["### 8.3 Dives and tower shots per 1000 champion-turns", "",
            md_table(["env", "dives", "busy dives", "busy share", "tower-adjacent trades", "expected tower shots",
                      "of which busy"], rows), ""]
    rows = []
    for e in D.envs:
        for k in ("busy", "not busy"):
            dv = [get(D.beh(e, s), "vision_tower", "dives", k, default=None) for s in SEEDS]
            rows.append([e, k, " / ".join(str(get(x, "n", default=0)) for x in dv),
                         seeds_cell([get(x, "mean_tower_p") for x in dv], f2), seeds_cell([get(x, "shot_rate") for x in dv], pct),
                         seeds_cell([get(x, "diver_died_within_2") for x in dv], pct),
                         seeds_cell([get(x, "dealt_bars") for x in dv], f2), seeds_cell([get(x, "taken_bars") for x in dv], f2)])
    out += ["### 8.4 Dives: busy versus not busy", "",
            md_table(["env", "dive", "n (s0 / s1 / s2)", "shot probability", "shot rate", "diver died within 2",
                      "dealt (bars)", "taken (bars)"], rows), ""]
    rows = []
    for e in D.envs:
        for which in ("defender_reply_on_dive_turn", "defender_reply_next_turn"):
            tabs = [get(D.beh(e, s), "vision_tower", which, default={}) for s in SEEDS]
            tot = {}
            for t in tabs:
                for k, v in (t or {}).items():
                    tot[k] = tot.get(k, 0) + v
            n = max(sum(tot.values()), 1)
            rows.append([e, which.replace("defender_reply_", "").replace("_", " "), sum(tot.values())]
                        + [pct(tot.get(k, 0) / n) for k in ("Hold", "Yield", "Trade", "Safe", "Recall", "absent")])
    out += ["### 8.5 The dived side's reply (pooled over seeds)", "",
            md_table(["env", "turn", "n", "Hold", "Yield", "Trade", "Safe", "Recall", "absent"], rows), ""]
    # against sh_dive / sh_vis references
    rows = []
    for e in D.envs:
        for r in ("sh_dive", "sh_vis", "shove4"):
            b = [D.beh(e, s, r) for s in SEEDS]
            p1 = [get(x, "vision_tower", "per_1000_turns", default=None) for x in b]
            rows.append([e, r, seeds_cell([get(x, "dives") for x in p1], f2, 1),
                         seeds_cell([get(x, "busy_dives") for x in p1], f2, 1),
                         seeds_cell([_obj_share(x, "Vision") for x in b], pct),
                         seeds_cell([get(x, "zoning", "deaths_per_1000_turns") for x in b], f2, 1)])
    out += ["### 8.6 Against the scripted pressure references (focal = the policy; 12 seeds x 1000 turns, both seats)", "",
            md_table(["env", "opponent", "policy's dives / 1000", "busy dives / 1000", "Vision when legal",
                      "policy deaths / 1000"], rows), ""]
    return out


def sec_zoning(D: Data) -> List[str]:
    if not D.eval:
        return []
    out = ["## 9. Zoning, combat strength, deaths and recalls", ""]
    rows = []
    for e in D.envs:
        b = [D.beh(e, s) for s in SEEDS]
        z = [get(x, "zoning", default=None) for x in b]
        bp = [get(x, "by_power", default={}) for x in z]
        n = [sum(get(y, k, "n", default=0) for k in ("stronger", "even", "weaker")) for y in bp]
        gap = [(get(y, "stronger", "n", default=0) + get(y, "weaker", "n", default=0)) / m if m else np.nan for y, m in zip(bp, n)]
        rows.append([e, seeds_cell(gap, pct), " / ".join(str(get(x, "credible_threat", "threat", "n", default=0)) for x in z),
                     seeds_cell([get(x, "deaths_per_1000_turns") for x in z], f2, 1),
                     seeds_cell([get(x, "low_hp_recalls_per_1000_turns") for x in z], f2, 1),
                     seeds_cell([get(x, "other_recalls_per_1000_turns") for x in z], f2, 1),
                     seeds_cell([get(x, "connected_trades_per_1000_turns") for x in z], f2, 0),
                     seeds_cell([get(x, "death_within_3_after_connected_trade") for x in z], pct)])
    out += ["### 9.1 NATURAL self-play: how often a power gap exists, deaths, recalls, trades", "",
            md_table(["env", "turns with a power gap (ratio >= 1.25 or <= 0.8)", "credible-threat turns (n)",
                      "deaths / 1000", "low-HP recalls / 1000", "other recalls / 1000", "connected trades / 1000",
                      "death within 3 of a connected trade"], rows), ""]
    for label, start, src in (("NATURAL self-play", "s0", "eval"), ("INITIALIZED x1.76 gap (neutral lane, A stronger; 16 games x 300 turns)", "neutral/large", "eval")):
        rows = []
        for e in D.envs:
            z = [get(D.beh(e, s, "self", start), "zoning", default=None) for s in SEEDS]
            for k in ("threat", "weaker, no threat"):
                g = [get(x, "credible_threat", k, default=None) for x in z]
                rows.append([e, k, " / ".join(str(get(x, "n", default=0)) for x in g),
                             seeds_cell([get(x, "behavior", "Safe") for x in g], pct),
                             seeds_cell([get(x, "behavior", "Yield") for x in g], pct),
                             seeds_cell([get(x, "behavior", "Trade") for x in g], pct),
                             seeds_cell([get(x, "mean_pos") for x in g], sg, 2),
                             seeds_cell([get(x, "uncollected_per_turn") for x in g], f2, 3),
                             seeds_cell([get(x, "died_within_4") for x in g], pct),
                             seeds_cell([get(x, "recalled_within_4") for x in g], pct)])
        out += [f"### 9.{2 if start == 's0' else 3} Credible threat, {label}", "",
                md_table(["env", "turns", "n (s0 / s1 / s2)", "Safe", "Yield", "Trade", "mean cell", "uncollected gold / turn",
                          "died within 4", "recalled within 4"], rows), ""]
    rows = []
    for e in D.envs:
        z = [get(D.beh(e, s, "self", "neutral/large"), "zoning", default=None) for s in SEEDS]
        for k in ("stronger", "even", "weaker"):
            bp = [get(x, "by_power", k, default=None) for x in z]
            rows.append([e, k, " / ".join(str(get(x, "n", default=0)) for x in bp),
                         seeds_cell([get(x, "mean_pos") for x in bp], sg), seeds_cell([get(x, "at_own_tower") for x in bp], pct),
                         seeds_cell([get(x, "trade_attempt_rate") for x in bp], pct),
                         seeds_cell([get(x, "collection_rate") for x in bp], pct),
                         seeds_cell([get(x, "death_rate") for x in bp], f2, 3)])
        pc = [get(x, "pressure_cost", default=None) for x in z]
        rows.append([e, "opp. uncollected gold / turn: stronger & near vs otherwise", "",
                     " / ".join(f"{f2(get(x, 'opp_uncollected_when_stronger_near', 'mean'), 3)} vs "
                                f"{f2(get(x, 'opp_uncollected_otherwise', 'mean'), 3)}" for x in pc), "", "", "", ""])
    out += ["### 9.4 INITIALIZED x1.76 gap: by power class of the focal side", "",
            md_table(["env", "class", "turns", "mean cell", "at own tower", "trade attempts", "collection", "death rate"], rows), ""]
    # recalls
    rows = []
    for e in D.envs:
        rc = [get(D.beh(e, s), "recalls", default=None) for s in SEEDS]
        for k in ("recall_low_hp", "recall_other", "death"):
            g = [get(x, k, default=None) for x in rc]
            rows.append([e, k.replace("_", " "), seeds_cell([get(x, "per_1000_turns") for x in g], f2, 1),
                         seeds_cell([get(x, "hp_frac") for x in g], pct), seeds_cell([get(x, "unspent_gold") for x in g], f2, 1),
                         seeds_cell([get(x, "bought") for x in g], f2, 1), seeds_cell([get(x, "away_turns") for x in g], f2, 1),
                         seeds_cell([get(x, "power_gain_share") for x in g], pct),
                         seeds_cell([get(x, "opp_lead_while_away") for x in g], f2, 2),
                         seeds_cell([get(x, "xp_missed_while_away") for x in g], f2, 1)])
    out += ["### 9.5 Recalls, deaths and investment (natural self-play)", "",
            md_table(["env", "event", "per 1000 turns", "HP at the event", "unspent gold", "bought on return",
                      "turns away", "came back relatively stronger", "opponent's gold lead while away",
                      "XP lost while away"], rows), ""]
    rows = []
    for e in D.envs:
        rc = [get(D.beh(e, s), "recalls", "by_investment_lead", default=None) for s in SEEDS]
        for k in ("ahead", "level", "behind"):
            g = [get(x, k, default=None) for x in rc]
            rows.append([e, k, " / ".join(str(get(x, "n", default=0)) for x in g),
                         seeds_cell([get(x, "trade_rate") for x in g], pct), seeds_cell([get(x, "shove_share") for x in g], pct),
                         seeds_cell([get(x, "safe_or_yield_share") for x in g], pct), seeds_cell([get(x, "mean_pos") for x in g], sg),
                         seeds_cell([get(x, "collection_rate") for x in g], pct)])
    out += ["### 9.6 Behaviour by investment lead (invested - enemy invested; ±5 resources; natural self-play)", "",
            md_table(["env", "lead", "turns", "Trade", "Shove", "Safe / Yield", "mean cell", "collection"], rows), ""]
    return out


def cf_alignment(states: List[dict]) -> Tuple[float, int]:
    """Share of states in which the policy put more mass on the side of the contrast that measured better (states
    with a significant contrast only), and their number."""
    hits = n = 0
    for st in states:
        a, b = st["contrast_names"]
        d, se = st["contrast"]
        if abs(d) <= 1.96 * se:
            continue
        fa, fb = st["frequency"].get(a, np.nan), st["frequency"].get(b, np.nan)
        if not (finite(fa) and finite(fb)) or fa == fb:
            continue
        n += 1
        hits += int((d > 0) == (fa > fb))
    return (hits / n if n else float("nan")), n


def sec_cf(D: Data) -> List[str]:
    if not D.eval:
        return []
    out = ["## 10. Counterfactuals (pre-registered: 8 states x 16 paired rollouts x 300 turns per final and situation; "
           "opponent protocol `policy`; continuation = the final in both seats)", ""]
    for key in CF_KEYS:
        rows, means = [], {}
        name = ""
        for e in D.envs:
            for s in SEEDS:
                c = D.cf(e, s, key)
                sm = get(c, "summary", default=None)
                if not isinstance(sm, dict) or not sm.get("n_states"):
                    rows.append([e, s, get(c, "states_found", default=0), get(c, "states_skipped", default=0)] + ["–"] * 8)
                    continue
                name = sm["contrast"]
                a, b = name.split(" - ")
                al, na = cf_alignment(c["states"])
                means.setdefault(e, []).append(sm["mean_contrast"][0])
                rows.append([e, s, sm["n_states"], c["states_skipped"], pm(*sm["mean_contrast"]),
                             f"{pct(sm['share_significant_positive'])} / {pct(sm['share_significant_negative'])}",
                             f"{pct(sm['frequency'][a][0])} / {pct(sm['frequency'][b][0])}",
                             pm(*sm.get("critic_mean_contrast_Q1", [np.nan, np.nan])),
                             pct(sm.get("critic_sign_agreement")),
                             f"{pct(al)} (n={na})",
                             "; ".join(f"{k}: {sg(v[0])}" for k, v in sm["vs_policy"].items()),
                             sm["opponent_first_action_identical_across_candidates"]])
        lab = get(D.cf(D.envs[0], SEEDS[0], key), "label", default=key)
        out += [f"### 10.{CF_KEYS.index(key) + 1} `{key}`: {lab}", "",
                md_table(["env", "seed", "states", "skipped", f"mean contrast {name} (± 95% over states)",
                          "states sig. + / −", f"policy mass {name.replace(' - ', ' / ')}", "critic one-step contrast",
                          "critic sign agreement", "policy favours the better side (sig. states)",
                          "candidate − policy", "opponent first action identical"], rows), ""]
        rr = [[e, " / ".join(sg(x) for x in v), "{} [{}, {}]".format(*[sg(x) for x in seed_stats(v)])] for e, v in means.items()]
        out += [md_table(["env", "seed means", "mean [t interval over seeds]"], rr), ""]
    return out


def sec_snapshot_behavior(D: Data) -> List[str]:
    sb = snapshot_behavior(D)
    if not sb:
        return []
    rows = []
    for e in D.envs:
        for s in SEEDS:
            pts = sb.get((e, s), [])
            for title, fn in SNAP_METRICS:
                rows.append([f"{e}_s{s}", title.replace("\n", " ")] + [f2(fn(m), 3 if "share" in title else 1) for _, m in pts])
    pts = [n for n, _ in next(iter(sb.values()))]
    return ["## 7b. Behaviour of intermediate snapshots (supplementary; self-play 16 games x 500 turns each)", "",
            md_table(["run", "measure"] + [f"{n / 1e6:.2f}M" for n in pts], rows), ""]


def sec_recall(D: Data) -> List[str]:
    R = (D.recall or {}).get("results")
    if not R:
        return []
    rows = []
    for k, v in R.items():
        r = v["interrupted_runs"]
        rows.append([short(k.split("|")[1]), f"{v['attempts_per_1000_turns']:.1f}", pct(v["interrupted_share"]),
                     f"{v['interrupted_per_1000_turns']:.1f}", pct(v["enemy_traded_share"]), pct(v["hp_at_attempt"]),
                     pct(v["hp_at_interrupted"]), pct(v["died_within_3_after_interrupted"]),
                     pct(v["policy_recall_mass_when_interrupted"]), pct(v["policy_safe_mass_when_interrupted"]),
                     f"{r['n']} / {r['mean_length']:.2f} / {r['runs_of_3_or_more']} / {r['max']}"])
    return ["## 9b. Chosen Recalls interrupted by an enemy Trade (supplementary follow-up of the traces; natural "
            "self-play, the first 8 of the 32 behaviour games)", "",
            md_table(["final", "Recalls chosen / 1000", "interrupted", "interrupted / 1000", "enemy chose Trade",
                      "HP at attempt", "HP when interrupted", "died within 3 after an interruption",
                      "policy Recall mass then", "policy Safe mass then", "interrupted runs: n / mean / >= 3 / max"],
                     rows), ""]


KEY_MEASURES = (
    ("Shove share, all turns", lambda b: (get(b, "wave_management", "frequency_per_game", "Shove", default=[np.nan]) or [np.nan])[0], pct),
    ("Shove holding a freeze", lambda b: get(b, "wave_management", "by_state", "holding a freeze (freeze state, I am the freezer)", "chose", "Shove"), pct),
    ("Shove frozen on", lambda b: get(b, "wave_management", "by_state", "frozen on (freeze state at the enemy tower)", "chose", "Shove"), pct),
    ("Shove, enemy crash under my tower", lambda b: get(b, "wave_management", "by_state", "enemy crash under my tower", "chose", "Shove"), pct),
    ("established freezes / 1000", lambda b: (get(b, "freezes", "established_per_1000_turns", default=[np.nan]) or [np.nan])[0], f2),
    ("useful denial share", lambda b: get(b, "freezes", "established", "useful_denial_share"), pct),
    ("Vision when legal (no active vision)", lambda b: _obj_share(b, "Vision"), pct),
    ("dives / 1000", lambda b: get(b, "vision_tower", "per_1000_turns", "dives"), f2),
    ("busy dives / 1000", lambda b: get(b, "vision_tower", "per_1000_turns", "busy_dives"), f2),
    ("deaths / 1000", lambda b: get(b, "zoning", "deaths_per_1000_turns"), f2),
)


def sec_ext(D: Data) -> List[str]:
    """The matched 2x-budget extension, reported beside (never instead of) the pre-registered 2,048,000 results."""
    T, E = D.train_ext, D.eval_ext
    if not T:
        return []
    out = ["## 13. Matched 2x-budget extension (4,096,000 transitions; training-extension policy, reported separately)", "",
           "The same 9 runs (environments, seeds, every PPO setting) at twice the budget; the learning rate anneals to 0 "
           "over the longer run. The 2,048,000-transition checkpoints and results above are untouched and remain the "
           "pre-registered result.", ""]
    rows = []
    for e in D.envs:
        for sd in SEEDS:
            r = T["runs"].get(f"{e}_s{sd}")
            if r:
                rows.append([f"{e}_s{sd}", f"{r['agent_transitions']:,}", f"{r['elapsed_s'] / 60:.1f}",
                             f"{r['entropy_first_last'][0]:.2f} → {r['entropy_first_last'][1]:.2f}",
                             f"{r['explained_variance_first_last10'][1]:.2f}", f"{r['approx_kl_mean']:.4f}",
                             f"{r['grad_norm_max']:.2f}", f"{r['value_s0_resources'][0]:+.2f}", r["digest"][:12]])
    c = T["checks"]
    out += [md_table(["run", "transitions", "min", "entropy", "EV last 10", "KL mean", "grad max", "V(s0)", "digest"], rows),
            "", f"Pipeline checks {sum(c['checks'].values())} / {len(c['checks'])}; learning checks "
                f"{sum(c['learning'].values())} / {len(c['learning'])}; wall clock {T['wall_s'] / 60:.1f} min.", ""]
    if not E:
        return out
    st = E["strength"]
    rows = []
    for e in D.envs:
        for r in REF_ORDER:
            m_main = [get(D.strength(e, D.spec(e, sd), f"ref:{r}"), "estimate") for sd in SEEDS]
            m_ext = [get(st.get(f"{e}|final:ext:{e}:{sd}|ref:{r}"), "estimate") for sd in SEEDS]
            diff = [b - a for a, b in zip(m_main, m_ext)]
            rows.append([e, r, seeds_cell(m_main), seeds_cell(m_ext), seeds_cell(diff)])
    out += ["### 13.1 Finals against the references: 2.05M versus 4.10M (W = 1000, 32 paired seeds, the same seed root)", "",
            md_table(["env", "reference", "2.05M finals", "4.10M finals", "difference (4.10M - 2.05M)"], rows), ""]
    rows = []
    for e in D.envs:
        cells, vals = [], []
        for sd in SEEDS:
            v = st.get(f"{e}|final:ext:{e}:{sd}|final:main:{e}:{sd}")
            vals.append(get(v, "estimate"))
            cells.append(pm(get(v, "estimate"), 1.96 * get(v, "stderr")) if v else "–")
        m, lo, hi = seed_stats(vals)
        own = []
        for sd in SEEDS:
            mid = [k for k in st if k.startswith(f"{e}|final:ext:{e}:{sd}|snap:ext:")]
            own.append(pm(st[mid[0]]["estimate"], 1.96 * st[mid[0]]["stderr"]) if mid else "–")
        rows.append([e] + cells + [f"{sg(m)} [{sg(lo)}, {sg(hi)}]", " / ".join(own)])
    out += ["### 13.2 Extension final against the main final of the same seed (W = 1000, 32 paired seeds) and against its "
            "own 2.05M snapshot (W = 460, 16)", "",
            md_table(["env", "seed 0", "seed 1", "seed 2", "seeds: mean [t interval]", "vs own 2.05M snapshot (s0 / s1 / s2)"],
                     rows), ""]
    rows = []
    for e in D.envs:
        for sd in SEEDS:
            pts = E["curve_snapshots"].get(f"{e}_s{sd}", [])
            for r in ("farm4", "shove4", "freeze_hunt4", "sh_vis", "sh_dive"):
                vals = [get(st.get(f"{e}|snap:ext:{e}:{sd}:{n}|ref:{r}"), "estimate") for n in pts]
                rows.append([f"{e}_s{sd}", r] + [sg(v) for v in vals])
    pts = next(iter(E["curve_snapshots"].values()))
    out += ["### 13.3 Extension learning curve against the core references (W = 460, 16 paired seeds)", "",
            md_table(["run", "reference"] + [f"{n / 1e6:.2f}M" for n in pts], rows), ""]
    hdr = ["env", "measure", "2.05M finals (s0 / s1 / s2)", "4.10M finals (s0 / s1 / s2)"]
    rows = []
    for e in D.envs:
        for name, fn, fmt in KEY_MEASURES:
            a = [fn(D.beh(e, sd)) for sd in SEEDS]
            b = [fn(get(E["behavior"].get(f"{e}|final:ext:{e}:{sd}|self|s0"), "metrics", "natural:s0", default=None))
                 for sd in SEEDS]
            rows.append([e, name, seeds_cell(a, fmt), seeds_cell(b, fmt)])
    out += ["### 13.4 Natural self-play behaviour of the finals, 2.05M versus 4.10M (32 games x 1000 turns, the same seeds)", "",
            md_table(hdr, rows), ""]
    rows = []
    sb = snapshot_behavior(D)
    for e in D.envs:
        for sd in SEEDS:
            main_pts = sb.get((e, sd), [])
            ext_pts = []
            for n in E["curve_snapshots"].get(f"{e}_s{sd}", []):
                v = E["snapshot_behavior"].get(f"{e}|snap:ext:{e}:{sd}:{n}|self|s0")
                if v:
                    ext_pts.append((n, v["metrics"]["natural:s0"]))
            for title, fn in SNAP_METRICS[:3]:
                rows.append([f"{e}_s{sd}", title.replace(NL, " "),
                             " ".join(f"{n / 1e6:.2f}M:{f2(fn(m), 2)}" for n, m in main_pts),
                             " ".join(f"{n / 1e6:.2f}M:{f2(fn(m), 2)}" for n, m in ext_pts)])
    out += ["### 13.5 Behaviour drift over training: main runs (2.05M schedule) and extension runs (4.10M schedule); "
            "self-play 16 games x 500 turns per point", "",
            md_table(["run", "measure", "main run snapshots", "extension run snapshots"], rows), ""]
    return out


def sec_transfer(D: Data) -> List[str]:
    tr = (D.extra or {}).get("transfer")
    if not tr:
        return []
    rows = []
    for k, v in sorted(tr["results"].items()):
        e, x, y = k.split("|")
        rows.append([e, short(x), short(y) if not y.startswith("ref:") else y[4:], pm(v["estimate"], 1.96 * v["stderr"])])
    return ["## 11. TRANSFER TEST (supplementary; cross-environment opt-in, not like-for-like)", "",
            "A final played in another environment sees observations normalized for its own game (absence scale 3 in "
            "`four`, 5 in `econ` / `aggro`) and an economy it was not trained on. Rows with a native final in the "
            "policy column are the same-environment baselines on the same seeds. W = 460, 16 paired seeds.", "",
            md_table(["game", "policy", "opponent", "value to the policy"], rows), ""]


def sec_traces(D: Data) -> List[str]:
    tr = D.traces
    out = []
    if (D.eval or {}).get("traces"):
        rows = [[k, v["label"], v["turns"], v["verify"]["actions_equal"] and v["verify"]["payoff_equal"],
                 f"{v['payoff_A']:+.2f}"] for k, v in D.eval["traces"].items()]
        out += ["## 12. Viewer recordings", "", "Pre-registered (`eval_main`): results/m44b/trajectories/", "",
                md_table(["name", "label", "turns", "replay verified", "payoff to A"], rows), ""]
    if tr:
        rows = []
        for k, v in tr["selected"].items():
            if not v.get("path"):
                rows.append([k, 0, "–", "–", "–"])
                continue
            rows.append([k, v["found"], os.path.basename(v["path"]), f"{v['side']} turns {v['t0']}-{v['t1']} (seed {v['seed']}, {v['scenario']}/{v['gap']})",
                         v["verify"]["actions_equal"] and v["verify"]["payoff_equal"]])
        out += ["Representative games (`traces_main`, state-defined search over the seed-0 finals' self-play):", "",
                md_table(["environment | situation", "candidates found", "file", "where", "replay verified"], rows), ""]
    return out


# ---------------------------------------------------------------------------------------------- main
def report_main(phase: str = "main", src: str = OUT_B, dst: str = OUT_B, figures: bool = True) -> str:
    """Tables to <dst>/report.md, figures to <dst>/figures/ (dst defaults to results/m44b; never the M4.4a folder)."""
    global FIG
    if os.path.abspath(dst) == os.path.abspath(os.path.join("results", "m44a")):
        raise ValueError("report_main must not write into results/m44a (the M4.4a record)")
    FIG = os.path.join(dst, "figures")
    D = Data(phase, src)
    figs = {}
    if figures:
        for name, fn in (("health", fig_health), ("curves", fig_curves), ("tournament", fig_tournament),
                         ("waves", fig_waves), ("freeze", fig_freeze), ("snap", fig_snapshot_behavior), ("cf", fig_cf)):
            try:
                figs[name] = fn(D)
            except Exception as ex:                                  # pragma: no cover - a figure must not stop the tables
                print(f"figure {name} failed: {ex!r}")
                figs[name] = None
    head = ["# M4.4b — controlled PPO self-play training: generated tables", "",
            f"Generated by `python -m lanegame.experiments.rl.m44 report_main` from results/m44b/ "
            f"(commit {get(D.train, 'meta', 'commit', default='?')[:10]}, working tree dirty: "
            f"{get(D.train, 'meta', 'working_tree_dirty', default='?')}). Narrative and interpretation: "
            "docs/m4/m4_4/m4_4b_ppo_training_report.md. Seeds as replicates: 'a / b / c (mean [lo, hi])' = the three "
            "training seeds and a t interval with 2 df.", ""]
    if any(figs.values()):
        head += ["Figures: " + ", ".join(f"[{k}]({os.path.relpath(v, dst).replace(os.sep, '/')})" for k, v in figs.items() if v), ""]
    parts = (head + sec_training(D) + [""] + sec_load(D) + [""] + sec_integrity(D) + [""] + sec_strength(D)
             + sec_curves(D) + sec_tournament(D) + sec_waves(D) + sec_freezes(D) + sec_snapshot_behavior(D)
             + sec_vision(D) + sec_zoning(D) + sec_recall(D) + sec_cf(D) + sec_transfer(D) + sec_traces(D)
             + sec_ext(D))
    return write_md("report", "\n".join(str(x) for x in parts), dst)


__all__ = ["report_main", "Data", "seed_stats", "curve_data", "tournament_matrix", "cf_alignment"]
