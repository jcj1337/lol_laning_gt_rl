"""results/m44c/report.md, results/m44c/summary.json and results/m44c/figures/ from the M4.4c result files.

Every number is computed here from results/m44c/*.json(.gz). Two levels of uncertainty, never mixed:
  * within one trained network: the states are the units (mean of per-state paired differences, 95% t interval);
  * across training seeds: the three per-seed means are the units (95% t interval, 2 degrees of freedom).
Tables print "mean [interval]" across seeds, with the per-seed means "s0 / s1 / s2" where they matter.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import engine as E
from .common import ENV_ORDER, OUT, PHASE_LABEL, SEEDS, fmt, load_json, pct, save_json, t_ci, write_md

FIG = os.path.join(OUT, "figures")
ENV_COLOR = {"aggro": "#2a78d6", "econ": "#eb6834", "four": "#1baf7a"}          # reference palette slots 1-3
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, INK2, MUTED, GRID, BASE, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


# ---------------------------------------------------------------------------------------------- selection
def cells(data: Optional[dict], phase: str = "ext", env: Optional[str] = None, situation: Optional[str] = None,
          protocol: Optional[str] = None) -> Dict[int, dict]:
    """seed -> cell for one (phase, env, situation, protocol)."""
    out = {}
    if not data:
        return out
    for c in data["cells"].values():
        if c["phase"] == phase and (env is None or c["env"] == env) and (situation is None or c["situation"] == situation) \
                and (protocol is None or c["protocol"] == protocol):
            out[c["seed"]] = c
    return out


def per_seed(cs: Dict[int, dict], cand: str, metric: str, ref: Optional[str] = "policy",
             where: Optional[Callable] = None) -> Dict[int, dict]:
    """seed -> across-state mean (t interval) of the candidate's paired difference against `ref` (ref None: its mean)."""
    out = {}
    for seed, c in sorted(cs.items()):
        sts = [st for st in c["states"] if where is None or where(st)]
        out[seed] = E.across_states(sts, cand, metric, ref)
    return out


def seeds_summary(ps: Dict[int, dict]) -> dict:
    vals = [v["mean"] for v in ps.values() if np.isfinite(v["mean"])]
    m, h, n = t_ci(vals)
    return {"per_seed": {int(k): v for k, v in ps.items()}, "mean": m, "hw": h, "n_seeds": n,
            "n_states": int(sum(v["n"] for v in ps.values()))}


def summary(cs, cand, metric, ref="policy", where=None) -> dict:
    return seeds_summary(per_seed(cs, cand, metric, ref, where))


def cell_str(s: dict, d: int = 2, seeds: bool = False, sign: bool = True) -> str:
    if not s or s.get("n_seeds", 0) == 0:
        return "n/a"
    txt = fmt(s["mean"], s["hw"], d, sign)
    if seeds:
        txt += " (" + " / ".join(fmt(v["mean"], None, d, sign) for v in s["per_seed"].values()) + ")"
    return txt


def freq(cs: Dict[int, dict], key: str, where=None) -> dict:
    """Mean policy frequency over states per seed, then across seeds."""
    ps = {}
    for seed, c in sorted(cs.items()):
        v = [st["frequency"].get(key, np.nan) for st in c["states"] if where is None or where(st)]
        m, h, n = t_ci(v)
        ps[seed] = {"mean": m, "hw": h, "n": n}
    return seeds_summary(ps)


def summary_fn(cs: Dict[int, dict], cand: str, fn: Callable, ref: Optional[str] = None, where=None) -> dict:
    """As `summary`, for a derived per-state quantity fn(mean dict) (minus the same for `ref`)."""
    ps = {}
    for seed, c in sorted(cs.items()):
        vals = []
        for st in c["states"]:
            if where is not None and not where(st):
                continue
            r = st["result"]
            if cand not in r["candidates"] or (ref is not None and ref not in r["candidates"]):
                continue
            vals.append(fn(r["mean"][cand]) - (fn(r["mean"][ref]) if ref is not None else 0.0))
        m, h, n = t_ci(vals)
        ps[seed] = {"mean": m, "hw": h, "n": n}
    return seeds_summary(ps)


def pooled(cs: Dict[int, dict], cand: str, metric: str, ref: Optional[str] = "policy", where=None) -> Tuple[float, float, int]:
    """States of all seeds pooled as units (for descriptive stratification only)."""
    sts = [st for c in cs.values() for st in c["states"] if where is None or where(st)]
    return t_ci(E.state_values(sts, cand, metric, ref))


def lead(h: int) -> Callable:
    return lambda m: m[f"gold_f{h}"] - m[f"gold_o{h}"]


def md_table(header: Sequence[str], rows: List[Sequence[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------- figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK, "axes.titlesize": 10,
                         "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.grid": True, "grid.color": GRID,
                         "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
                         "axes.spines.top": False, "axes.spines.right": False})
    return plt


def grouped_ci(ax, groups: Sequence[str], series: Sequence[str], vals, hws, colors, ylabel: str = "",
               zero: bool = True, width: float = 0.8):
    """Grouped bars with t-interval whiskers: vals[s][g]."""
    n = len(series)
    w = width / n
    x = np.arange(len(groups))
    for i, s in enumerate(series):
        v = np.asarray(vals[i], dtype=float)
        h = np.asarray(hws[i], dtype=float)
        ax.bar(x + (i - (n - 1) / 2) * w, v, w * 0.92, color=colors[i], label=s, zorder=2)
        ax.errorbar(x + (i - (n - 1) / 2) * w, v, yerr=np.where(np.isfinite(h), h, 0), fmt="none", ecolor=INK2,
                    elinewidth=0.9, capsize=2, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    if zero:
        ax.axhline(0, color=BASE, linewidth=1, zorder=1)
    ax.set_ylabel(ylabel)


def savefig(fig, name: str) -> str:
    os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------------------------- Part 2: Recall
RECALL_CANDS = ("Recall", "Safe", "Hold", "Safe>Recall", "Retreat")
RECALL_SITS = ("contested", "repeated", "retreat", "safe")


def recall_section(S: dict) -> List[str]:
    data = load_json("recall")
    rep = load_json("recall_rep")
    L = ["## 2. Interrupted Recall (Part 2)", ""]
    if not data:
        return L + ["*recall.json missing*", ""]
    out = S.setdefault("recall", {})
    # state description
    rows = []
    for env in ENV_ORDER:
        for sit in RECALL_SITS:
            cs = cells(data, "ext", env, sit, "policy")
            if not cs:
                continue
            sts = [st for c in cs.values() for st in c["states"]]
            pos = np.array([st["desc"]["pos"] for st in sts])
            epos = np.array([st["desc"]["enemy_pos"] for st in sts])
            ev = np.mean([not st["desc"].get("safe_vs_trade_connected", False) for st in sts])
            rows.append([env, sit, len(sts), f"{np.mean(pos == -2):.0%}", f"{np.mean((pos == -2) & (epos == 2)):.0%}",
                         f"{np.mean([st['row']['hp_frac'] for st in sts]):.2f}",
                         cell_str(freq(cs, "Recall")), cell_str(freq(cs, "Safe")), cell_str(freq(cs, "opp_Trade")),
                         f"{ev:.0%}"])
    L += ["**States** (natural self-play of the 4.10M finals; decision state only). Columns: share of states with the "
          "champion in its own tower cell; share with the enemy already standing in that cell (a diver); mean HP; "
          "policy's Recall and Safe mass; the opponent policy's Trade mass at the same state (= the probability that a "
          "Recall is interrupted when the enemy can reach it); share of states where Safe would not be connected by an "
          "enemy Trade this turn.", "",
          md_table(["env", "states", "n", "in tower cell", "enemy in that cell", "HP", "P(Recall)", "P(Safe)",
                    "opp P(Trade)", "Safe escapes"], rows), ""]
    # main contrasts: candidate - Recall and candidate - policy at G300
    for proto in ("policy", "modal"):
        L += [f"**Discounted return over 300 turns, candidate minus Recall** (opponent protocol `{proto}`; seed mean "
              "[95% t interval over the three seeds]; positive = better than recalling now)", ""]
        rows = []
        for env in ENV_ORDER:
            for sit in RECALL_SITS:
                cs = cells(data, "ext", env, sit, proto)
                if not cs:
                    continue
                r = [env, sit]
                for cand in ("policy", "Safe", "Hold", "Safe>Recall", "Retreat"):
                    s = summary(cs, cand, "G300", "Recall")
                    out[f"{proto}|{env}|{sit}|{cand}-Recall|G300"] = s
                    r.append(cell_str(s))
                rows.append(r)
        L += [md_table(["env", "states", "policy", "Safe", "Hold", "Safe>Recall", "Retreat"], rows), ""]
    # short horizons
    L += ["**Short and long horizons** (`policy` protocol): Safe − Recall at G10 / G50 / G300", ""]
    rows = []
    for env in ENV_ORDER:
        for sit in RECALL_SITS:
            cs = cells(data, "ext", env, sit, "policy")
            if cs:
                rows.append([env, sit] + [cell_str(summary(cs, "Safe", f"G{h}", "Recall")) for h in (10, 50, 300)] +
                            [cell_str(summary(cs, "Retreat", f"G{h}", "Recall")) for h in (10, 50, 300)])
    L += [md_table(["env", "states", "Safe G10", "Safe G50", "Safe G300", "Retreat G10", "Retreat G50",
                    "Retreat G300"], rows), ""]
    # episode outcomes (absolute means per candidate), contested + policy protocol
    keys = [("left_by_recall", "left by Recall (40 t)", 0), ("left_by_death", "left by death", 0),
            ("left_turn", "turns to leave", 1), ("dmg_before_leave", "HP lost before leaving (bars)", 2),
            ("died_f10", "died within 10", 0), ("interrupts_before_leave", "interruptions", 2),
            ("gold_f10", "own gold 10 t", 2), ("gold_o10", "enemy gold 10 t", 2), ("xp_f10", "own XP 10 t", 1),
            ("invest_f40", "invested 40 t", 2)]
    for env in ENV_ORDER:
        for sit in ("contested", "repeated", "retreat"):
            cs = cells(data, "ext", env, sit, "policy")
            if not cs:
                continue
            rows = []
            for cand in ("policy",) + RECALL_CANDS:
                r = [cand]
                for k, _, d in keys:
                    s = summary(cs, cand, k, None)
                    out[f"policy|{env}|{sit}|{cand}|{k}"] = s
                    r.append(pct(s["mean"]) if d == 0 else fmt(s["mean"], None, d, False))
                rows.append(r)
            if env == "aggro" or sit == "contested":
                L += [f"**Episode outcomes, {env}, {sit}** (`policy` protocol; means over states and seeds)", "",
                      md_table(["candidate"] + [n for _, n, _ in keys], rows), ""]
    # the guessing game: Safe - Recall by the opponent's Trade probability
    L += ["**The guessing game**: Safe − Recall (G300) by the opponent's Trade probability q at the state, and the "
          "policy's Recall mass in the same bins (contested + repeated + retreat states pooled per environment, "
          "`policy` protocol; states as units, all seeds)", ""]
    rows = []
    bins = [(0, 0.3, "q < 0.3"), (0.3, 0.6, "0.3 ≤ q < 0.6"), (0.6, 1.01, "q ≥ 0.6")]
    for env in ENV_ORDER:
        sts = [st for sit in ("contested", "repeated", "retreat") for c in cells(data, "ext", env, sit, "policy").values()
               for st in c["states"]]
        for lo, hi, name in bins:
            b = [st for st in sts if lo <= st["frequency"]["opp_Trade"] < hi]
            d = E.state_values(b, "Safe", "G300", "Recall")
            dr = E.state_values(b, "Retreat", "G300", "Recall")
            m, h, n = t_ci(d)
            mr, hr, _ = t_ci(dr)
            pr, _, _ = t_ci([st["frequency"]["Recall"] for st in b])
            ps, _, _ = t_ci([st["frequency"]["Safe"] for st in b])
            out[f"q|{env}|{name}"] = {"safe_minus_recall": [m, h, n], "retreat_minus_recall": [mr, hr],
                                      "p_recall": pr, "p_safe": ps}
            rows.append([env, name, n, fmt(m, h), fmt(mr, hr), pct(pr), pct(ps)])
    L += [md_table(["env", "opponent P(Trade)", "states", "Safe − Recall", "Retreat − Recall", "policy P(Recall)",
                    "policy P(Safe)"], rows), ""]
    # indifference point: linear fit of Safe - Recall on q, bootstrap over states
    from scipy import stats
    L += ["**Indifference point** (same states): least-squares fit Safe − Recall = a + b q; q* = −a / b with a 95% "
          "bootstrap interval over states (2,000 resamples); the policy's Recall / Safe mass above and below q*; the "
          "Spearman correlation of the policy's Recall mass with q; within-state significance (|d| > 1.96 SE over the "
          "24 rollouts); and the fitted regret per decision of the policy's Recall / Safe mixing and of always "
          "recalling, against the better of the two at each state's q (G300, resources)", ""]
    rows = []
    for env in ENV_ORDER:
        sts = [st for sit in ("contested", "repeated", "retreat") for c in cells(data, "ext", env, sit, "policy").values()
               for st in c["states"]]
        if len(sts) < 5:
            continue
        q = np.array([st["frequency"]["opp_Trade"] for st in sts])
        y = np.array(E.state_values(sts, "Safe", "G300", "Recall"))
        pr = np.array([st["frequency"]["Recall"] for st in sts])
        ps = np.array([st["frequency"]["Safe"] for st in sts])
        lr = stats.linregress(q, y)
        qs = -lr.intercept / lr.slope if lr.slope > 0 else float("nan")
        rng = np.random.default_rng(81_399)
        bs = []
        for _ in range(2000):
            i = rng.integers(0, len(q), len(q))
            b = stats.linregress(q[i], y[i])
            bs.append(-b.intercept / b.slope if b.slope > 0 else np.nan)
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        above = q > qs
        rho = stats.spearmanr(q, pr)
        fit = lr.intercept + lr.slope * q
        regret_pol = float(np.mean(np.maximum(fit, 0) - ps * fit))
        regret_rec = float(np.mean(np.maximum(fit, 0)))
        sig_pos = np.mean([d > 1.96 * st["result"]["diff"]["Safe"]["G300"][1] for d, st in zip(y, sts)])
        sig_neg = np.mean([d < -1.96 * st["result"]["diff"]["Safe"]["G300"][1] for d, st in zip(y, sts)])
        out[f"fit|{env}"] = {"a": lr.intercept, "b": lr.slope, "p": lr.pvalue, "q_star": qs, "q_star_ci": [lo, hi],
                             "share_above": float(above.mean()), "p_recall_above": float(pr[above].mean()),
                             "p_recall_below": float(pr[~above].mean()), "p_safe_above": float(ps[above].mean()),
                             "p_safe_below": float(ps[~above].mean()), "spearman": float(rho.statistic),
                             "spearman_p": float(rho.pvalue), "regret_policy": regret_pol, "regret_always_recall": regret_rec,
                             "sig_pos": float(sig_pos), "sig_neg": float(sig_neg), "n": len(sts)}
        rows.append([env, len(sts), f"{lr.intercept:+.2f} + {lr.slope:.2f} q (p = {lr.pvalue:.0e})",
                     f"{qs:.2f} [{lo:.2f}, {hi:.2f}]", pct(above.mean()),
                     f"{pct(pr[above].mean())} / {pct(pr[~above].mean())}", f"{pct(ps[above].mean())} / {pct(ps[~above].mean())}",
                     f"{rho.statistic:+.2f} (p = {rho.pvalue:.3f})", f"{pct(sig_pos)} / {pct(sig_neg)}",
                     f"{regret_pol:.3f} / {regret_rec:.3f}"])
    L += [md_table(["env", "states", "fit", "q*", "states above q*", "P(Recall) above / below", "P(Safe) above / below",
                    "Spearman (q, P(Recall))", "Safe sig. better / worse", "regret: policy / always Recall"], rows), ""]
    # replication 2.05M
    if rep:
        L += ["**Replication on the 2.05M finals** (`policy` protocol): candidate − Recall, G300", ""]
        rows = []
        for env in ("econ", "aggro"):
            for sit in ("contested", "repeated"):
                cs = cells(rep, "main", env, sit, "policy")
                if not cs:
                    continue
                r = [env, sit]
                for cand in ("policy", "Safe", "Hold", "Safe>Recall", "Retreat"):
                    s = summary(cs, cand, "G300", "Recall")
                    out[f"rep|{env}|{sit}|{cand}-Recall|G300"] = s
                    r.append(cell_str(s))
                r.append(cell_str(freq(cs, "Recall")))
                rows.append(r)
        L += [md_table(["env", "states", "policy", "Safe", "Hold", "Safe>Recall", "Retreat", "P(Recall)"], rows), ""]
    L += [integrity(data), ""]
    _recall_figure(data, out)
    return L


def integrity(data: dict) -> str:
    """Weights unchanged by every job; common random numbers held (chance identical across candidates on the first and
    last turn of every state; the opponent's branch-turn action identical across candidates)."""
    cs = list(data["cells"].values())
    ok = all(c["digests_unchanged"] for c in cs)
    chance = all(st["result"]["pairing"]["chance_identical_first_turn"] and
                 st["result"]["pairing"]["chance_identical_last_turn"] for c in cs for st in c["states"])
    opp = all(st["result"]["pairing"]["opponent_first_action_identical"] for c in cs for st in c["states"])
    n = sum(len(c["states"]) for c in cs)
    return (f"Integrity: {n} evaluated states; weights digests unchanged in every job: **{ok}**; chance identical "
            f"across candidates on the first and last turn: **{chance}**; the opponent's branch-turn action identical "
            f"across candidates: **{opp}**; runtime {data['elapsed_s'] / 60:.0f} min.")


def _recall_figure(data, out) -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    sits = ("contested", "repeated", "retreat", "safe")
    vals, hws = [], []
    for env in ENV_ORDER:
        v, h = [], []
        for sit in sits:
            s = summary(cells(data, "ext", env, sit, "policy"), "Safe", "G300", "Recall")
            v.append(s["mean"])
            h.append(s["hw"])
        vals.append(v)
        hws.append(h)
    grouped_ci(ax, sits, ENV_ORDER, vals, hws, [ENV_COLOR[e] for e in ENV_ORDER], "Safe − Recall, G300 (resources)")
    ax.set_title("Safe instead of Recall on the branch turn (seed means, 95% t)")
    ax.legend(loc="upper right", fontsize=8)
    ax = axes[1]
    for env in ENV_ORDER:
        sts = [st for sit in ("contested", "repeated", "retreat") for c in cells(data, "ext", env, sit, "policy").values()
               for st in c["states"]]
        q = [st["frequency"]["opp_Trade"] for st in sts]
        d = E.state_values(sts, "Safe", "G300", "Recall")
        ax.scatter(q, d, s=12, color=ENV_COLOR[env], alpha=0.35, edgecolors=SURF, linewidths=0.5)
        if len(q) > 3:
            b, a = np.polyfit(q, d, 1)
            xs = np.array([0.0, 1.0])
            ax.plot(xs, a + b * xs, color=ENV_COLOR[env], linewidth=2.2, label=f"{env} (fit; q* = {-a / b:.2f})")
    ax.axhline(0, color=BASE, linewidth=1)
    ax.set_xlabel("opponent's Trade probability at the state (= interruption risk)")
    ax.set_ylabel("Safe − Recall per state, G300")
    ax.set_title("Per state: Safe pays when the opponent is likely to Trade")
    ax.legend(fontsize=8, loc="lower right")
    savefig(fig, "fig1_recall.png")


# ---------------------------------------------------------------------------------------------- Part 3: freeze
FREEZE_CANDS = ("Preserve", "Preserve-Thin", "Break", "Conditional")
FREEZE_PROTOS = ("policy", "modal", "breaker", "passive", "stepback")


def freeze_section(S: dict) -> List[str]:
    data, init, rep = load_json("freeze"), load_json("freeze_init"), load_json("freeze_rep")
    L = ["## 3. Freezing (Part 3)", ""]
    if not data:
        return L + ["*freeze.json missing*", ""]
    out = S.setdefault("freeze", {})
    rows = []
    for env in ENV_ORDER:
        cs = cells(data, "ext", env, "held", "policy")
        if not cs:
            continue
        sts = [st for c in cs.values() for st in c["states"]]
        d = [st["desc"] for st in sts]
        rows.append([env, len(sts), f"{np.mean([st['row']['held_run'] for st in sts]):.1f}",
                     f"{np.mean([st['row']['held_run'] >= 4 for st in sts]):.0%}",
                     f"{np.mean([x['enemy_absent'] for x in d]):.0%}",
                     f"{np.mean([x['power'] >= 1.25 for x in d]):.0%} / {np.mean([x['power'] <= 0.8 for x in d]):.0%}",
                     f"{np.mean([x['plates_left'] > 0 for x in d]):.0%}",
                     f"{np.mean([x['ehp'] <= 0.5 for x in d]):.0%}"] +
                    [cell_str(freq(cs, k)) for k in ("Last Hit", "Thin", "Slow Push", "Shove", "opp_Shove")])
    L += ["**States**: the focal side holds a freeze for >= 2 consecutive turns (natural self-play, 4.10M finals). "
          "Columns: mean run length so far; share >= 4 turns (M4.4a 'established'); enemy away; freezer stronger / "
          "weaker (power ratio 1.25 / 0.8); enemy plates left; frozen-on side at <= 50% HP; the freezer's wave masses; "
          "the frozen-on side's Shove mass.", "",
          md_table(["env", "n", "run so far", ">= 4", "enemy away", "stronger / weaker", "plates left", "enemy HP<=50%",
                    "P(LH)", "P(TN)", "P(SL)", "P(SH)", "opp P(SH)"], rows), ""]
    for metric, label in (("G300", "300 turns"), ("G20", "20 turns")):
        L += [f"**Freezer plan minus the learned policy, discounted return over {label}** (rows: opponent protocol / "
              "style of the frozen-on side)", ""]
        rows = []
        for env in ENV_ORDER:
            for proto in FREEZE_PROTOS:
                cs = cells(data, "ext", env, "held", proto)
                if not cs:
                    continue
                r = [env, proto]
                for cand in FREEZE_CANDS:
                    s = summary(cs, cand, metric)
                    out[f"{proto}|{env}|{cand}|{metric}"] = s
                    r.append(cell_str(s))
                s = summary(cs, "Preserve", metric, "Break")
                out[f"{proto}|{env}|Preserve-Break|{metric}"] = s
                r.append(cell_str(s))
                rows.append(r)
        L += [md_table(["env", "opponent", "Preserve", "Preserve-Thin", "Break", "Conditional", "Preserve − Break"],
                       rows), ""]
    keys = [("freeze_len", "freeze turns", 1, None), ("lead20", "gold lead 20 t", 2, lead(20)),
            ("xplost_o20", "enemy XP lost 20 t", 1, None), ("unc_o20", "enemy uncollected gold 20 t", 2, None),
            ("crash_o20", "enemy crash HP 20 t", 1, None), ("crash_f20", "own crash HP 20 t", 1, None),
            ("died_f20", "freezer died", 0, None), ("died_o20", "enemy died", 0, None),
            ("rec_o20", "enemy recalled", 0, None), ("plates_f20", "own plates", 2, None),
            ("inv_lead_change40", "investment lead change 40 t", 2, None)]
    for env in ("aggro", "econ"):
        for proto in ("policy", "stepback", "breaker"):
            cs = cells(data, "ext", env, "held", proto)
            if not cs:
                continue
            rows = []
            for cand in ("policy",) + FREEZE_CANDS:
                r = [cand]
                for k, _, dd, fn in keys:
                    s = summary_fn(cs, cand, fn) if fn else summary(cs, cand, k, None)
                    out[f"ep|{proto}|{env}|{cand}|{k}"] = s
                    r.append(pct(s["mean"]) if dd == 0 else fmt(s["mean"], None, dd, k.startswith("lead") or "change" in k))
                rows.append(r)
            L += [f"**Episode outcomes, {env}, opponent `{proto}`** (means over states and seeds)", "",
                  md_table(["candidate"] + [n for _, n, _, _ in keys], rows), ""]
    L += ["**How the freeze ended** (share of rollouts, `policy` protocol, all seeds; DESCRIPTIVE: an observed outcome, "
          "not a treatment)", ""]
    rows = []
    for env in ENV_ORDER:
        cs = cells(data, "ext", env, "held", "policy")
        for cand in ("policy",) + FREEZE_CANDS:
            tot = {e: 0 for e in E.FREEZE_ENDS}
            for c in cs.values():
                for st in c["states"]:
                    for e, n in st["result"].get("freeze_end", {}).get(cand, {}).items():
                        tot[e] += n
            N = max(1, sum(tot.values()))
            rows.append([env, cand] + [f"{tot[e] / N:.0%}" for e in E.FREEZE_ENDS])
    L += [md_table(["env", "candidate"] + list(E.FREEZE_ENDS), rows), ""]
    strata = [("plates left", lambda st: st["desc"]["plates_left"] > 0),
              ("plates exhausted", lambda st: st["desc"]["plates_left"] == 0),
              ("enemy HP <= 50%", lambda st: st["desc"]["ehp"] <= 0.5),
              ("enemy HP > 50%", lambda st: st["desc"]["ehp"] > 0.5),
              ("enemy away", lambda st: st["desc"]["enemy_absent"]),
              ("freezer stronger (>= 1.25)", lambda st: st["desc"]["power"] >= 1.25),
              ("even power", lambda st: 0.8 < st["desc"]["power"] < 1.25),
              ("run so far >= 4", lambda st: st["row"]["held_run"] >= 4),
              ("run so far 2-3", lambda st: st["row"]["held_run"] < 4),
              ("level <= 9", lambda st: st["desc"]["level"] <= 9), ("level >= 10", lambda st: st["desc"]["level"] >= 10)]
    L += ["**Stratified (descriptive; states of all three seeds pooled as units, `policy` protocol)**: Preserve − policy "
          "and Preserve − Break at G300", ""]
    rows = []
    for name, w in strata:
        r = [name]
        for env in ("aggro", "econ"):
            cs = cells(data, "ext", env, "held", "policy")
            m1, h1, n1 = pooled(cs, "Preserve", "G300", "policy", w)
            m2, h2, _ = pooled(cs, "Preserve", "G300", "Break", w)
            r += [n1, fmt(m1, h1), fmt(m2, h2)]
        rows.append(r)
    L += [md_table(["stratum", "aggro n", "aggro Preserve − policy", "aggro Preserve − Break", "econ n",
                    "econ Preserve − policy", "econ Preserve − Break"], rows), ""]
    L += ["**Does the policy recognise when breaking pays?** Per state: the policy's Shove mass against the measured "
          "Break − Preserve (G300); Spearman rank correlation over the states of all seeds (`policy` protocol)", ""]
    rows = []
    from scipy.stats import spearmanr
    for env in ENV_ORDER:
        sts = [st for c in cells(data, "ext", env, "held", "policy").values() for st in c["states"]]
        sts = [st for st in sts if "Break" in st["result"]["candidates"] and "Preserve" in st["result"]["candidates"]]
        x = [st["frequency"]["Shove"] for st in sts]
        y = E.state_values(sts, "Break", "G300", "Preserve")
        if len(x) == len(y) and len(x) > 3:
            rho, p = spearmanr(x, y)
            out[f"recognise|{env}"] = {"rho": float(rho), "p": float(p), "n": len(x)}
            rows.append([env, len(x), f"{rho:+.2f}", f"{p:.3f}"])
    L += [md_table(["env", "states", "Spearman rho (P(Shove) vs Break − Preserve)", "p"], rows), ""]
    if init:
        L += ["**INITIALIZED freeze grid** (M4.3c freeze start; 24 states per seed; econ and aggro): Preserve − Break and "
              "Preserve − policy at G300, the 20-turn gold-lead difference Preserve − Break and the extra freeze turns, "
              "by factor (states of all seeds pooled as units; each factor averaged over the others)", ""]
        rows = []
        for env in ("aggro", "econ"):
            for proto in ("policy", "stepback", "breaker", "passive"):
                cs = cells(init, "ext", env, "init", proto)
                if not cs:
                    continue
                for factor, levels in (("gap", ("equal", "large", "b_large")), ("level", (4, 10)),
                                       ("opp_hp", (1.0, 0.5)), ("plates", ("left", "exhausted"))):
                    for lv in levels:
                        w = (lambda st, f=factor, v=lv: st["grid"][f] == v)
                        m1, h1, n1 = pooled(cs, "Preserve", "G300", "Break", w)
                        m2, h2, _ = pooled(cs, "Preserve", "G300", "policy", w)
                        sts = [st for c in cs.values() for st in c["states"] if w(st)]
                        dl = [lead(20)(st["result"]["mean"]["Preserve"]) - lead(20)(st["result"]["mean"]["Break"])
                              for st in sts]
                        m3, h3, _ = t_ci(dl)
                        fl = t_ci([st["result"]["mean"]["Preserve"]["freeze_len"] -
                                   st["result"]["mean"]["Break"]["freeze_len"] for st in sts])
                        out[f"init|{env}|{proto}|{factor}={lv}"] = {"pres_break": [m1, h1, n1], "pres_policy": [m2, h2],
                                                                    "lead20": [m3, h3], "freeze_len": list(fl)}
                        rows.append([env, proto, f"{factor} = {lv}", n1, fmt(m1, h1), fmt(m2, h2), fmt(m3, h3),
                                     fmt(fl[0], fl[1], 1)])
        L += [md_table(["env", "opponent", "factor", "n", "Preserve − Break G300", "Preserve − policy G300",
                        "Preserve − Break gold lead 20 t", "extra freeze turns"], rows), ""]
    if rep:
        L += ["**Replication on the 2.05M finals** (`policy` protocol): plan − policy, G300", ""]
        rows = []
        for env in ("econ", "aggro"):
            cs = cells(rep, "main", env, "held", "policy")
            if cs:
                rows.append([env] + [cell_str(summary(cs, c, "G300")) for c in FREEZE_CANDS] +
                            [cell_str(summary(cs, "Preserve", "G300", "Break"))])
                for c in FREEZE_CANDS:
                    out[f"rep|{env}|{c}|G300"] = summary(cs, c, "G300")
        L += [md_table(["env", "Preserve", "Preserve-Thin", "Break", "Conditional", "Preserve − Break"], rows), ""]
    L += [integrity(data), ""]
    _freeze_figure(data)
    return L


def _freeze_figure(data) -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, env in zip(axes, ("aggro", "econ")):
        vals, hws = [], []
        for cand in FREEZE_CANDS:
            v, h = [], []
            for proto in FREEZE_PROTOS:
                s = summary(cells(data, "ext", env, "held", proto), cand, "G300")
                v.append(s["mean"])
                h.append(s["hw"])
            vals.append(v)
            hws.append(h)
        grouped_ci(ax, FREEZE_PROTOS, FREEZE_CANDS, vals, hws, SERIES[:4], "plan − learned policy, G300")
        ax.set_title(f"{env}: freezer plans against five frozen-on opponents")
        ax.set_xlabel("frozen-on side (protocol / style)")
    axes[0].legend(fontsize=8, ncol=2, loc="lower left")
    savefig(fig, "fig2_freeze.png")


# ---------------------------------------------------------------------------------------------- Part 4: zoning
ZONE_CANDS = ("Farm", "Threaten", "Bully", "Freeze+threat", "Shove+Pressure")
ZONE_RESP = ("policy", "ignore", "respect", "shover", "ref:respect4")


def zoning_section(S: dict) -> List[str]:
    init, nat, mech = load_json("zoning"), load_json("zoning_nat"), load_json("zoning_mech")
    L = ["## 4. Strong-side zoning (Part 4)", ""]
    out = S.setdefault("zoning", {})
    if mech:
        L += ["**Behaviour matrix** (INITIALIZED neutral lane; both behaviours fixed for 10 turns, waves each side's own "
              "conditional choice, then the policy). Stronger side A's 10-turn gold lead, B's uncollected gold and HP "
              "lost (bars) over 10 turns, B's deaths, and A's G300; states of all seeds pooled per gap (B at 100% and "
              "50% HP)", ""]
        rows = []
        for env in ENV_ORDER:
            for gap in ("equal", "large", "large_levels"):
                for proto in ("B_Hold", "B_Yield", "B_Safe"):
                    cs = cells(mech, "ext", env, "matrix", proto)
                    if not cs:
                        rows.append([env, gap, proto[2:], "not run (control)", "", ""])
                        continue
                    w = (lambda st, g=gap: st["grid"]["gap"] == g)
                    r = [env, gap, proto[2:]]
                    for cand in ("A_Hold", "A_Trade", "A_Threaten"):
                        sts = [st for c in cs.values() for st in c["states"] if w(st) and cand in st["result"]["mean"]]
                        mm = lambda k: float(np.mean([st["result"]["mean"][cand][k] for st in sts])) if sts else np.nan
                        ld = float(np.mean([lead(10)(st["result"]["mean"][cand]) for st in sts])) if sts else np.nan
                        out[f"matrix|{env}|{gap}|{proto}|{cand}"] = {"lead10": ld, "unc_o10": mm("unc_o10"),
                                                                     "dmg_o10": mm("dmg_o10"), "died_o10": mm("died_o10"),
                                                                     "dmg_f10": mm("dmg_f10"), "G300": mm("G300")}
                        r.append(f"{ld:+.2f} / {mm('unc_o10'):.2f} / {mm('dmg_o10'):.2f} / {mm('died_o10'):.0%} / "
                                 f"{mm('G300'):+.2f}")
                    rows.append(r)
        L += [md_table(["env", "A's gap", "B plays", "A Hold: lead / B unc. / B dmg / B died / G300", "A Trade",
                        "A Threaten"], rows), ""]
    for data, sit, label in ((init, "init", "INITIALIZED power-gap states (16 per seed)"),
                             (nat, "edge", "NATURAL states with a large HP-aware duel edge (24 per seed)")):
        if not data:
            continue
        L += [f"**Stronger-side plan minus the learned policy, G300 — {label}** (rows: the weaker side's response)", ""]
        rows = []
        for env in ENV_ORDER:
            for resp in ZONE_RESP:
                cs = cells(data, "ext", env, sit, resp)
                if not cs:
                    continue
                r = [env, resp]
                for cand in ZONE_CANDS:
                    s = summary(cs, cand, "G300")
                    out[f"{sit}|{resp}|{env}|{cand}|G300"] = s
                    r.append(cell_str(s))
                rows.append(r)
        L += [md_table(["env", "weaker side", *ZONE_CANDS], rows), ""]
        keys = [("lead20", "lead 20 t", lead(20), 2), ("unc_o20", "weaker uncollected gold", None, 2),
                ("xplost_o20", "weaker XP lost", None, 1), ("dmg_o20", "weaker HP lost (bars)", None, 2),
                ("died_o20", "weaker died", None, 0), ("rec_o20", "weaker recalled", None, 0),
                ("opp_safe20", "weaker Safe share", None, 0), ("opp_yield20", "weaker Yield share", None, 0),
                ("opp_tower_cell20", "weaker in tower cell", None, 0), ("tconn_f20", "connected trades", None, 1),
                ("dmg_f20", "stronger HP lost", None, 2), ("died_f20", "stronger died", None, 0),
                ("gold_f20", "stronger gold", None, 2)]
        for env in ("aggro", "econ"):
            for resp in (("policy", "respect", "ignore") if sit == "init" else ("policy", "respect")):
                cs = cells(data, "ext", env, sit, resp)
                if not cs:
                    continue
                rows = []
                for cand in ("policy",) + ZONE_CANDS:
                    r = [cand]
                    for k, _, fn, dd in keys:
                        s = summary_fn(cs, cand, fn) if fn else summary(cs, cand, k, None)
                        out[f"ep|{sit}|{resp}|{env}|{cand}|{k}"] = s
                        r.append(pct(s["mean"]) if dd == 0 else fmt(s["mean"], None, dd, k.startswith("lead")))
                    rows.append(r)
                L += [f"**20-turn outcomes, {env}, weaker side `{resp}`, {sit}** (means over states and seeds)", "",
                      md_table(["stronger plan"] + [n for _, n, _, _ in keys], rows), ""]
        if sit == "init":
            L += ["**By power gap** (INITIALIZED, weaker side = policy; states of all seeds pooled): Threaten − Farm and "
                  "Bully − Farm at G300, and the extra uncollected gold of the weaker side (Threaten − Farm, 20 t)", ""]
            rows = []
            for env in ("aggro", "econ", "four"):
                cs = cells(data, "ext", env, "init", "policy")
                for sc in ("neutral", "freeze"):
                    for gap in ("equal", "moderate", "large", "large_levels"):
                        w = (lambda st, s_=sc, g=gap: st["grid"]["scenario"] == s_ and st["grid"]["gap"] == g)
                        a = pooled(cs, "Threaten", "G300", "Farm", w)
                        b = pooled(cs, "Bully", "G300", "Farm", w)
                        u = pooled(cs, "Threaten", "unc_o20", "Farm", w)
                        dm = pooled(cs, "Threaten", "dmg_o20", "Farm", w)
                        out[f"gap|{env}|{sc}|{gap}"] = {"threaten_farm": list(a), "bully_farm": list(b),
                                                        "unc": list(u), "dmg": list(dm)}
                        rows.append([env, sc, gap, a[2], fmt(*a[:2]), fmt(*b[:2]), fmt(*u[:2]), fmt(*dm[:2])])
            L += [md_table(["env", "scenario", "A's gap", "n", "Threaten − Farm G300", "Bully − Farm G300",
                            "weaker unc. gold (Threaten − Farm)", "weaker HP lost (Threaten − Farm)"], rows), ""]
    if init:
        L += [integrity(init), ""]
        _zoning_figure(init, mech)
    return L


def _zoning_figure(init, mech) -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    vals, hws = [], []
    resps = ("policy", "ignore", "respect", "shover", "ref:respect4")
    plans5 = ("policy", "Threaten", "Bully", "Freeze+threat", "Shove+Pressure")
    for cand in plans5:
        v, h = [], []
        for resp in resps:
            s = summary(cells(init, "ext", "aggro", "init", resp), cand, "G300", "Farm")
            v.append(s["mean"])
            h.append(s["hw"])
        vals.append(v)
        hws.append(h)
    grouped_ci(ax, resps, ("learned policy",) + plans5[1:], vals, hws, SERIES[:5], "plan − Farm, G300 (resources)")
    ax.set_title("aggro, initialized power gaps: stronger-side plans vs farming")
    ax.set_xlabel("weaker side's response")
    ax.legend(fontsize=8, ncol=2, loc="lower left")
    ax = axes[1]
    vals, hws = [], []
    for cand in ("Threaten", "Bully"):
        v, h = [], []
        for resp in resps:
            s = summary(cells(init, "ext", "aggro", "init", resp), cand, "unc_o20", "Farm")
            v.append(s["mean"])
            h.append(s["hw"])
        vals.append(v)
        hws.append(h)
    grouped_ci(ax, resps, ("Threaten", "Bully"), vals, hws, SERIES[1:3], "extra uncollected gold of the weaker side, 20 t")
    ax.set_title("Resource denial actually caused (vs Farm)")
    ax.set_xlabel("weaker side's response")
    ax.legend(fontsize=8)
    savefig(fig, "fig3_zoning.png")


# ---------------------------------------------------------------------------------------------- Part 5: vision
VIS_CANDS = ("Pressure", "Vision>Pressure", "Conditional", "No objective")


def vision_section(S: dict) -> List[str]:
    data, init = load_json("vision"), load_json("vision_init")
    L = ["## 5. Vision before sustained Pressure (Part 5)", ""]
    if not data:
        return L + ["*vision.json missing*", ""]
    out = S.setdefault("vision", {})
    rows = []
    for env in ENV_ORDER:
        for sit in ("plates", "exhausted", "active"):
            cs = cells(data, "ext", env, sit, "policy")
            if not cs:
                continue
            sts = [st for c in cs.values() for st in c["states"]]
            rows.append([env, sit, len(sts), f"{np.mean([st['desc']['siege_hp'] for st in sts]):.1f}",
                         f"{np.mean([st['desc']['siege_hp'] >= 12 for st in sts]):.0%}",
                         f"{np.mean([st['desc']['hp'] for st in sts]):.2f}"] +
                        [cell_str(freq(cs, k)) for k in ("Pressure", "Vision", "none")])
    L += ["**States** (natural siege states; objective legal). Columns: own crashed mass under the enemy tower (HP), "
          "share >= 12 HP, HP, the policy's objective masses.", "",
          md_table(["env", "situation", "n", "siege HP", ">= 12 HP", "HP", "P(Pressure)", "P(Vision)", "P(none)"], rows), ""]
    for metric in ("G300", "G20"):
        L += [f"**Plan minus the learned policy, {metric}**", ""]
        rows = []
        for env in ENV_ORDER:
            for sit, proto in (("plates", "policy"), ("plates", "modal"), ("exhausted", "policy"), ("active", "policy")):
                cs = cells(data, "ext", env, sit, proto)
                if not cs:
                    continue
                r = [env, sit, proto]
                for cand in VIS_CANDS:
                    s = summary(cs, cand, metric)
                    out[f"{sit}|{proto}|{env}|{cand}|{metric}"] = s
                    r.append(cell_str(s))
                s = summary(cs, "Vision>Pressure", metric, "Pressure")
                out[f"{sit}|{proto}|{env}|V-P|{metric}"] = s
                r.append(cell_str(s))
                rows.append(r)
        L += [md_table(["env", "situation", "opponent", *VIS_CANDS, "Vision>Pressure − Pressure"], rows), ""]
    keys = [("press_f20", "Pressure turns", 1), ("press_protected20", "protected Pressure turns", 1),
            ("visset_f20", "Vision set", 2), ("gankp_f20", "expected ganks", 2), ("ganked_f20", "ganks", 2),
            ("gankdmg_f20", "gank damage (bars)", 2), ("died_f20", "died", 0), ("rec_f20", "recalled", 0),
            ("plates_f20", "plates", 2), ("tower_f20", "tower damage (units)", 1), ("gold_f20", "own gold", 2),
            ("siege_turns20", "siege turns", 1)]
    for env in ("aggro", "econ", "four"):
        for sit in ("plates", "exhausted"):
            cs = cells(data, "ext", env, sit, "policy")
            if not cs:
                continue
            rows = []
            for cand in ("policy",) + VIS_CANDS:
                r = [cand]
                for k, _, dd in keys:
                    s = summary(cs, cand, k, None)
                    out[f"ep|{sit}|{env}|{cand}|{k}"] = s
                    r.append(pct(s["mean"]) if dd == 0 else fmt(s["mean"], None, dd, False))
                rows.append(r)
            L += [f"**20-turn outcomes, {env}, {sit}** (`policy` protocol; means over states and seeds)", "",
                  md_table(["plan"] + [n for _, n, _ in keys], rows), ""]
    L += ["**Stratified (descriptive; states of all seeds pooled)**: Vision>Pressure − Pressure at G300 and G20", ""]
    rows = []
    for env in ENV_ORDER:
        for sit in ("plates", "exhausted"):
            cs = cells(data, "ext", env, sit, "policy")
            for name, w in (("siege >= 12 HP", lambda st: st["desc"]["siege_hp"] >= 12),
                            ("siege < 12 HP", lambda st: st["desc"]["siege_hp"] < 12),
                            ("HP >= 60%", lambda st: st["desc"]["hp"] >= 0.6), ("HP < 60%", lambda st: st["desc"]["hp"] < 0.6),
                            ("level <= 9", lambda st: st["desc"]["level"] <= 9),
                            ("level >= 10", lambda st: st["desc"]["level"] >= 10)):
                a = pooled(cs, "Vision>Pressure", "G300", "Pressure", w)
                b = pooled(cs, "Vision>Pressure", "G20", "Pressure", w)
                rows.append([env, sit, name, a[2], fmt(*a[:2]), fmt(*b[:2])])
    L += [md_table(["env", "situation", "stratum", "n", "V>P − P G300", "V>P − P G20"], rows), ""]
    if init:
        L += ["**INITIALIZED siege start** (A's 14-HP crash under B's tower; 48 rollouts per state; seed means)", ""]
        rows = []
        for env in ENV_ORDER:
            for proto in ("policy", "modal"):
                cs = cells(init, "ext", env, "init", proto)
                for gap, hp in (("equal", 1.0), ("equal", 0.4), ("large", 1.0), ("large", 0.4)):
                    w = (lambda st, g=gap, h=hp: st["grid"]["gap"] == g and st["grid"]["hp"] == h)
                    a = summary(cs, "Vision>Pressure", "G300", "Pressure", w)
                    b = summary(cs, "No objective", "G300", "Pressure", w)
                    c = summary(cs, "Pressure", "G300", "policy", w)
                    rows.append([env, proto, gap, f"{hp:.0%}", cell_str(a), cell_str(b), cell_str(c)])
        L += [md_table(["env", "opponent", "A's gap", "A's HP", "Vision>Pressure − Pressure", "No objective − Pressure",
                        "Pressure − policy"], rows), ""]
    L += [integrity(data), ""]
    _vision_figure(data)
    return L


def _vision_figure(data) -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, sit in zip(axes, ("plates", "exhausted")):
        vals, hws = [], []
        for env in ENV_ORDER:
            v, h = [], []
            for cand in VIS_CANDS:
                s = summary(cells(data, "ext", env, sit, "policy"), cand, "G300")
                v.append(s["mean"])
                h.append(s["hw"])
            vals.append(v)
            hws.append(h)
        grouped_ci(ax, VIS_CANDS, ENV_ORDER, vals, hws, [ENV_COLOR[e] for e in ENV_ORDER], "plan − learned policy, G300")
        ax.set_title(f"siege, enemy plates {'left' if sit == 'plates' else 'exhausted'}")
    axes[0].legend(fontsize=8)
    savefig(fig, "fig4_vision.png")


# ---------------------------------------------------------------------------------------------- Part 6: waves
WAVE_CANDS = ("TNx8", "SLx8", "SL>SH", "SHx8", "policy")
WAVE_OPPS = ("opp_LH", "opp_TN", "opp_SL", "opp_SH", "policy")


def waves_section(S: dict) -> List[str]:
    mech, data = load_json("waves_mech"), load_json("waves")
    L = ["## 6. Last Hit / Thin / Slow Push / Shove (Part 6)", ""]
    out = S.setdefault("waves", {})
    if mech:
        L += ["**Budgets and one-turn transitions at natural decision states** (random present turns of self-play, both "
              "seats; the focal side's recorded behaviour / objective and the opponent's recorded action kept; the lane "
              "step has no chance). Share of states where two wave actions have the same budget, and where they leave "
              "the lane in exactly the same state.", ""]
        rows = []
        classes = [("all", lambda r: True), ("own wave stronger (>= +1)", lambda r: r["wave_adv"] >= 1),
                   ("even (|adv| < 1)", lambda r: abs(r["wave_adv"]) < 1), ("enemy stronger (<= -1)", lambda r: r["wave_adv"] <= -1)]
        for env in ENV_ORDER:
            rws = [r for c in mech["cells"].values() if c["env"] == env for r in c["rows"]]
            for name, w in classes:
                rr = [r for r in rws if w(r)]
                if not rr:
                    continue
                b = lambda f: np.mean([f(r["budget"]) for r in rr])
                same = lambda k: np.mean([r["same"][k] for r in rr])
                pw = np.mean([r["pw"] for r in rr], axis=0)
                key = f"mech|{env}|{name}"
                out[key] = {"n": len(rr), "budget_TN_eq_LH": b(lambda x: x["TN"] == x["LH"]),
                            "budget_SL_eq_LH": b(lambda x: x["SL"] == x["LH"]), "budget_SL_eq_TN": b(lambda x: x["SL"] == x["TN"]),
                            "same_LH_TN": same("LH-TN"), "same_TN_SL": same("TN-SL"), "same_LH_SL": same("LH-SL"),
                            "same_SL_SH": same("SL-SH"), "pw": pw.tolist()}
                o = out[key]
                rows.append([env, name, len(rr), pct(o["budget_TN_eq_LH"]), pct(o["budget_SL_eq_LH"]),
                             pct(o["same_LH_TN"]), pct(o["same_TN_SL"]), pct(o["same_LH_SL"]), pct(o["same_SL_SH"]),
                             " / ".join(f"{x:.0%}" for x in pw)])
        L += [md_table(["env", "state class", "n", "D_TN = D_LH", "D_SL = D_LH", "same lane LH/TN", "TN/SL", "LH/SL",
                        "SL/SH", "policy P(LH / TN / SL / SH)"], rows), ""]
        L += ["**Does the policy's choice among Last Hit / Thin / Slow Push depend on whether they differ?** Mean policy "
              "masses in states where Thin leaves the same lane as Last Hit vs where it does not (all states)", ""]
        rows = []
        for env in ENV_ORDER:
            rws = [r for c in mech["cells"].values() if c["env"] == env for r in c["rows"]]
            for name, w in (("LH and TN identical", lambda r: r["same"]["LH-TN"]), ("LH and TN differ", lambda r: not r["same"]["LH-TN"]),
                            ("LH, TN, SL all identical", lambda r: r["same"]["LH-TN"] and r["same"]["TN-SL"])):
                rr = [r for r in rws if w(r)]
                if rr:
                    pw = np.mean([r["pw"] for r in rr], axis=0)
                    out[f"pw|{env}|{name}"] = {"n": len(rr), "pw": pw.tolist()}
                    rows.append([env, name, len(rr)] + [f"{x:.0%}" for x in pw])
        L += [md_table(["env", "states", "n", "P(LH)", "P(TN)", "P(SL)", "P(SH)"], rows), ""]
        for env in ENV_ORDER:
            for lv in ("L3", "L10"):
                m = mech["matrix"][env][lv]
                names = ("LH", "TN", "SL", "SH")
                rows = [[a] + [f"{m[f'{a}|{b}']['gold_diff16']:+.2f} ({int(m[f'{a}|{b}']['crashes'][0])})" for b in names]
                        for a in names]
                out[f"matrix|{env}|{lv}"] = {k: v["gold_diff16"] for k, v in m.items()}
                if env != "econ":
                    L += [f"**Deterministic 16-turn wave matrix, {env}, `neutral` start at level {lv[1:]}**: row "
                          "player's gold lead (its crashes), both Hold, no chance", "",
                          md_table(["row \\ column"] + list(names), rows), ""]
    if data:
        for metric in ("G300", "G20"):
            L += [f"**8-turn wave sequence minus Last Hit x8, {metric}** (rows: the opponent's wave for the same 8 turns)", ""]
            rows = []
            for env in ENV_ORDER:
                for opp in WAVE_OPPS:
                    cs = cells(data, "ext", env, "contact", opp)
                    if not cs:
                        continue
                    r = [env, opp]
                    for cand in WAVE_CANDS:
                        s = summary(cs, cand, metric, "LHx8")
                        out[f"seq|{opp}|{env}|{cand}-LHx8|{metric}"] = s
                        r.append(cell_str(s))
                    rows.append(r)
            L += [md_table(["env", "opponent", "Thin x8", "Slow Push x8", "SL then SH", "Shove x8", "policy"], rows), ""]
        L += ["**Immediate wave effects** (aggro and econ, opponent = policy; means over states and seeds): wave "
              "meeting point (own coordinates; + = toward the enemy tower), window-strength lead, own crash HP, gold lead", ""]
        rows = []
        for env in ("aggro", "econ", "four"):
            cs = cells(data, "ext", env, "contact", "policy")
            if not cs:
                continue
            for cand in ("LHx8", "TNx8", "SLx8", "SL>SH", "SHx8", "policy"):
                r = [env, cand]
                for h in (2, 4, 8, 16):
                    wp = summary(cs, cand, f"wpos{h}", None)["mean"]
                    wa = summary(cs, cand, f"wadv{h}", None)["mean"]
                    r.append(f"{wp:+.2f} / {wa:+.1f}")
                r.append(fmt(summary(cs, cand, "crash_f16", None)["mean"], None, 1, False))
                r.append(fmt(summary_fn(cs, cand, lead(16))["mean"], None, 2))
                rows.append(r)
        L += [md_table(["env", "plan", "pos / lead 2 t", "4 t", "8 t", "16 t", "own crash HP 16 t", "gold lead 16 t"], rows), ""]
        L += ["**Signal to noise** (states of all seeds pooled): |plan − policy| at G300 against the per-rollout standard "
              "deviation of that paired difference (SE x sqrt(n)), and the rollouts needed to see the difference at 2 "
              "standard errors in ONE state, (2 sd / |d|)^2", ""]
        rows = []
        for env in ENV_ORDER:
            cs = cells(data, "ext", env, "contact", "policy")
            sts = [st for c in cs.values() for st in c["states"]]
            for cand in ("LHx8", "TNx8", "SLx8", "SHx8"):
                d = np.array([st["result"]["diff"][cand]["G300"][0] for st in sts if cand in st["result"]["diff"]])
                sd = np.array([st["result"]["diff"][cand]["G300"][1] * np.sqrt(st["result"]["n"]) for st in sts
                               if cand in st["result"]["diff"]])
                keep = sd > 0                                   # identical plans (sd 0) carry no noise information
                d, sd = d[keep], sd[keep]
                if len(d):
                    med = float(np.median(np.abs(d) / sd))
                    need = float(np.median((2 * sd / np.maximum(np.abs(d), 1e-9)) ** 2))
                    out[f"snr|{env}|{cand}"] = {"abs_d": float(np.mean(np.abs(d))), "sd": float(np.mean(sd)),
                                                "median_ratio": med, "median_n_needed": need}
                    rows.append([env, cand, f"{np.mean(np.abs(d)):.2f}", f"{np.mean(sd):.2f}", f"{med:.2f}", f"{need:.0f}"])
        L += [md_table(["env", "plan (vs policy)", "mean |d|", "mean sd", "median |d|/sd", "median rollouts needed"], rows), ""]
        L += [integrity(data), ""]
        _waves_figure(data)
    return L


def _waves_figure(data) -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    hs = (2, 4, 8, 16)
    cs = cells(data, "ext", "aggro", "contact", "policy")
    for i, cand in enumerate(("LHx8", "TNx8", "SLx8", "SL>SH", "SHx8")):
        v = [summary(cs, cand, f"wpos{h}", None)["mean"] for h in hs]
        ax.plot(hs, v, marker="o", markersize=5, linewidth=2, color=SERIES[i], label=cand)
    ax.set_xticks(hs)
    ax.set_xlabel("turns after the branch")
    ax.set_ylabel("wave meeting point (own coordinates)")
    ax.set_title("aggro: where the waves meet under each 8-turn wave plan")
    ax.legend(fontsize=8)
    ax = axes[1]
    vals, hws = [], []
    for env in ENV_ORDER:
        v, h = [], []
        for cand in ("TNx8", "SLx8", "SL>SH", "SHx8"):
            s = summary(cells(data, "ext", env, "contact", "policy"), cand, "G300", "LHx8")
            v.append(s["mean"])
            h.append(s["hw"])
        vals.append(v)
        hws.append(h)
    grouped_ci(ax, ("TNx8", "SLx8", "SL>SH", "SHx8"), ENV_ORDER, vals, hws, [ENV_COLOR[e] for e in ENV_ORDER],
               "plan − Last Hit x8, G300")
    ax.set_title("Long-run value against the learned opponent")
    ax.legend(fontsize=8)
    savefig(fig, "fig5_waves.png")


# ---------------------------------------------------------------------------------------------- mechanics
def mechanics_section(S: dict) -> List[str]:
    from .mechanics import all_tables
    T = all_tables()
    S["mechanics"] = T
    L = ["## 7. Closed-form mechanics (exact one-turn resolutions, no chance)", ""]
    rows = [[r["a_pos"], r["b_pos"], r["a"], r["b"], r["a_end"], r["b_end"], r["interrupted"], r["connected"],
             f"{r['hit_p']:.2f}", f"{r['trader_tower_p']:.2f}"] for r in T["aggro"]["recall"] if r["b"] == "Trade"]
    L += ["**A damaged champion (A, 30% HP) against an enemy Trade, aggro** (cells in own coordinates)", "",
          md_table(["A cell", "B cell", "A plays", "B plays", "A ends", "B ends", "interrupted", "trade connects",
                    "hit p", "B's tower-shot p"], rows), ""]
    for env in ("aggro", "econ"):
        tt = T[env]["trade"]
        L += [f"**Trade hit probability by the defender's behaviour, {env}**: " +
              "; ".join(f"{k} {v:.2f}" for k, v in tt["hit_probability"].items()), ""]
    rows = [[d["level"], d["gap"], d["vs"], f"{d['expected_bars']:.3f}", f"{d['bars_per_unit_hit']:.3f}"]
            for d in T["aggro"]["trade"]["expected_damage"]]
    L += ["**Expected damage of one connected trade on the weaker side** (bars of its max HP; aggro)", "",
          md_table(["level", "A's gap", "defender", "expected bars", "bars per 1-unit hit"], rows), ""]
    rows = [[r["mode"]] + [f"{r[str(x)]:.4f}" for x in (-2, -1, 0, 1, 2)] for r in T["aggro"]["collection"]]
    L += ["**Gold per minion HP by mode and cell (own coordinates), aggro**", "",
          md_table(["mode", "-2", "-1", "0", "+1", "+2"], rows), ""]
    v = T["aggro"]["vision"]
    rows = [[g["behaviour"], g["wave"], f"{g['p']:.3f}", f"{g['p_vision']:.3f}", f"{g['units_if_ganked']:.1f}",
             f"{g['units_per_turn']:.3f}", f"{g['units_per_turn_vision']:.3f}"] for g in v["gank"]]
    L += ["**Gank exposure at the enemy tower (+2), aggro** (HP units; 6 units = one bar)", "",
          md_table(["behaviour", "wave", "P(gank)", "P(gank) with vision", "units if ganked", "units / turn",
                    "units / turn with vision"], rows), ""]
    rows = [[p["level"], p["invested"], p["centi_units_per_turn"], f"{p['turns_first_plate']:.1f}",
             f"{p['turns_all_plates']:.1f}"] for p in v["pressure"]]
    L += ["**Pressure tower damage per turn** (plates at 1,600 / 3,600 / 6,000 centi-units, 1.0 resource each; tower "
          "damage beyond 6,000 is clamped and worth nothing)", "",
          md_table(["level", "invested", "centi-units / turn", "turns to first plate", "turns to all plates"], rows), ""]
    return L


# ---------------------------------------------------------------------------------------------- driver
SECTIONS = [freeze_section, zoning_section, vision_section, waves_section, mechanics_section]


def report() -> dict:
    S: dict = {}
    L = ["# M4.4c generated tables", "",
         "Generated by `python -m lanegame.experiments.rl.m44c report` from results/m44c/*.json(.gz). Interpretation: "
         "docs/m4/m4_4c_strategic_investigation.md. Seed means with the 95% t interval over the three training seeds "
         "(2 df) unless stated; G_h = discounted focal return over h turns (γ = 0.99), resources.", ""]
    for fn in [recall_section] + SECTIONS:
        try:
            L += fn(S)
        except Exception as e:                                # a missing stage must not hide the others
            import traceback
            L += [f"*section {fn.__name__} failed: {e!r}*", "", "```", traceback.format_exc()[-1500:], "```", ""]
    write_md("report", "\n".join(L))
    from ....io import dump_json
    dump_json(S, os.path.join(OUT, "summary.json"))            # never gzipped: the tracked record of every number
    manifest()
    return S


STAGE_COMMANDS = {"recall": "recall", "recall_rep": "recall_rep", "freeze": "freeze", "freeze_init":
                  "freeze_init --envs econ,aggro", "freeze_rep": "freeze_rep", "zoning": "zoning", "zoning_nat": "zoning_nat",
                  "zoning_mech": "zoning_mech", "vision": "vision", "vision_init": "vision", "waves_mech": "waves_mech",
                  "waves": "waves", "traces": "traces"}


def manifest() -> str:
    """results/m44c/MANIFEST.json: sha256, size and regeneration command of every raw results file (the raw per-state
    files are git-ignored as large and regenerable; this file, the report, summary.json and the figures are tracked)."""
    import glob
    import hashlib
    from ....io import dump_json
    files = []
    for p in sorted(glob.glob(os.path.join(OUT, "*.json.gz")) + glob.glob(os.path.join(OUT, "*.json"))):
        name = os.path.basename(p)
        if name in ("summary.json", "MANIFEST.json"):
            continue
        stage = name.split(".")[0]
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        files.append({"file": name, "bytes": os.path.getsize(p), "sha256": h,
                      "command": f"python -m lanegame.experiments.rl.m44c {STAGE_COMMANDS.get(stage, stage)}"})
    out = os.path.join(OUT, "MANIFEST.json")
    dump_json({"study": "M4.4c", "note": "raw per-state results; regenerate with the command (deterministic seeds); "
               "verify with sha256", "files": files}, out)
    return out


__all__ = ["report", "cells", "summary", "freq", "per_seed", "seeds_summary", "md_table", "cell_str"]
