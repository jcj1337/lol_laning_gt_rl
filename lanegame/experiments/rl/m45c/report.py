"""results/m45c/report.md and results/m45c/figures/: every pilot table, generated from the stage JSONs (nothing is
computed here that a stage did not measure, except formatting, differences and the plan's extrapolations).

Figures are small multiples with one series per panel (no legend; the panel title names the series), the reference
palette's first categorical slot for the data and recessive grey for zero lines and grids."""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from .common import OUT, PILOT_SEED, PILOT_STOP, load_json, write_md
from .references import OOD, ROLE, SUITE

FIG = os.path.join(OUT, "figures")
BLUE, GREY, INK, MUTED = "#2a78d6", "#a3a29c", "#0b0b0b", "#52514e"


def _f(x, d=2, sign=True):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def _ci(r: dict, d=2) -> str:
    lo, hi = r["ci95"]
    return f"{r['estimate']:+.{d}f} ± {(hi - lo) / 2:.{d}f}"


def table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    out = ["| " + " | ".join(str(h) for h in header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _point_of(spec: str) -> int:
    p = spec.split(":")
    return 0 if p[0] == "initial" else (PILOT_STOP if p[0] == "final" else int(p[3]))


def _opp_label(o: str) -> str:
    if o.startswith("ref:"):
        return o[4:]
    if o.startswith("ood:"):
        return "M4.4b aggro s0 (OOD)"
    return o


# ---------------------------------------------------------------------------------------------- figures
def _style(ax):
    ax.grid(True, color="#e6e5e0", linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GREY)
    ax.tick_params(colors=MUTED, labelsize=8)


def fig_curves(ev: dict) -> Optional[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cur = ev.get("curve")
    if not cur:
        return None
    opps = [f"ref:{r}" for r in SUITE] + list(OOD)
    n = len(opps)
    cols = 5
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.5 * rows), sharex=True, sharey=True)
    for ax, o in zip(axes.ravel(), opps):
        pts = sorted((_point_of(k.split("|")[0]), v) for k, v in cur.items() if k.split("|")[1] == o)
        x = np.array([p for p, _ in pts]) / 1e6
        y = np.array([v["estimate"] for _, v in pts])
        lo = np.array([v["ci95"][0] for _, v in pts])
        hi = np.array([v["ci95"][1] for _, v in pts])
        ax.axhline(0, color=GREY, linewidth=1)
        ax.fill_between(x, lo, hi, color=BLUE, alpha=0.15, linewidth=0)
        ax.plot(x, y, color=BLUE, linewidth=2, marker="o", markersize=3)
        ax.set_title(_opp_label(o), fontsize=9, color=INK)
        _style(ax)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.supxlabel("agent transitions (millions); 0 = untrained network", fontsize=9, color=MUTED)
    fig.supylabel("discounted lead vs opponent (resources, W = 460)", fontsize=9, color=MUTED)
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, "learning_curves.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def fig_series(h: List[dict], keys: Sequence[tuple], name: str, xlabel: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = 3
    rows = math.ceil(len(keys) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.4 * rows), sharex=True)
    x = np.array([e["agent_transitions"] for e in h]) / 1e6
    for ax, (k, title, scale) in zip(axes.ravel(), keys):
        y = np.array([e[k] for e in h], dtype=float) * scale
        ax.plot(x, y, color=BLUE, linewidth=1.2, alpha=0.35)
        w = max(1, len(y) // 25)
        if len(y) >= w:
            sm = np.convolve(y, np.ones(w) / w, mode="valid")
            ax.plot(x[w - 1:], sm, color=BLUE, linewidth=2)
        ax.set_title(title, fontsize=9, color=INK)
        _style(ax)
    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    fig.supxlabel(xlabel, fontsize=9, color=MUTED)
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, name)
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------------------------- sections
def sec_audit(a: dict) -> str:
    rp = a["random_play"]
    rows = [["preset equals GameConfig.m45_calibrated()", a["preset"]["equals_m45_calibrated"]],
            ["M4.5b decisions retained (last_front, delay 0, growth 1.08, xp_range 2, ward, alpha 50)",
             a["preset"]["retained_as_specified"]],
            ["observation entries / action codes", f"{rp['obs_size']} / {rp['n_actions']}"],
            ["ever-legal codes (all observed in random play)", f"{a['action_space']['ever_legal']} "
             f"(not observed: {rp['ever_legal_not_observed'] or 'none'}; outside: {rp['outside_ever_legal'] or 'none'})"],
            ["observations outside [0, 1] or non-finite", rp["observations_out_of_range"]],
            ["masks different from the legal sets", rp["masks_not_equal_legal"]],
            ["mirror-decoding failures (seat B = mirror of seat A)", rp["mirror_failures"]],
            ["records' Ward-legality rule vs engine", f"{rp['record_ward_rule_mismatches']} mismatches"],
            ["fixed opponents legal from both seats", all(a["references_legal"].values())],
            ["M4.4 finals on M4.5c", "; ".join(sorted(set(a["m44_checkpoints_on_m45c"].values())))],
            ["OOD adapter (drops the two W entries)", "; ".join(f"{k}: {'loads' if v.get('loads_through_adapter') else 'absent'}"
                                                              for k, v in a["ood_adapter"].items())]]
    return "## 1. Interface audit\n\n" + table(["check", "result"], rows) + f"\n\nRandom play: {rp['turns']} turns.\n"


def sec_smoke(s: dict, se: Optional[dict]) -> str:
    rows = [[k, "pass" if v else "**FAIL**"] for k, v in s["checks"].items()]
    rows += [[k + " (informational)", "yes" if v else "no"] for k, v in s["learning"].items()]
    r = s["runs"]["s100"]
    txt = ("## 2. Smoke test (10 updates, own schedule) and bit-identical re-run\n\n" + table(["check", "result"], rows)
           + f"\n\nThroughput {r['transitions_per_s']:,.0f} agent transitions/s (two concurrent processes); peak "
             f"working set {r['memory'].get('peak_mb', float('nan')):.0f} MB.\n")
    if se:
        txt += "\nSmoke evaluation: " + ", ".join(f"{k}: {'pass' if v else 'FAIL'}" for k, v in se["checks"].items())
        txt += "; guard " + ", ".join(f"{k} {v}" for k, v in se["guard"].items()) + ".\n"
    return txt


def sec_config(p: dict) -> str:
    sp = p["spec"]
    pp = sp["ppo"]
    run = p["run"]
    rows = [["environment", sp["environment"]], ["network", f"{sp['network']['obs_size']} → 128 → 128 (tanh) → "
                                                          f"{sp['network']['n_actions']} masked logits + value; from scratch"],
            ["initialization seed", f"{run['seed']} (torch.manual_seed; collector streams seed + 1); initial digest "
                                    f"`{run['initial_digest'][:12]}…`"],
            ["learning rate", f"{pp['lr']} annealed linearly to 0 over {run['schedule_total_steps']:,} transitions; "
                              f"stopped at {run['stop_at']:,} (final lr {run.get('final_lr', 'n/a')})"],
            ["PPO", f"clip {pp['clip_coef']}, value clip {pp['vf_clip']}, vf coef {pp['vf_coef']}, entropy "
                    f"{pp['ent_coef']}, {pp['epochs']} epochs, minibatch {pp['minibatch_size']}, max grad norm "
                    f"{pp['max_grad_norm']}, advantage normalization {pp['norm_adv']}, reward scale {pp['reward_scale']}"],
            ["γ / GAE λ", f"{pp['gamma']} / {pp['gae_lambda']}"],
            ["rollout", f"{pp['n_envs']} environments × {pp['segment_length']} turns × 2 seats = "
                        f"{sp['accounting']['transitions_per_update']:,} agent transitions per update "
                        f"({sp['accounting']['joint_steps_per_update']:,} joint steps; "
                        f"{sp['accounting']['gradient_steps_per_update']} gradient steps)"],
            ["games", f"{pp['game_length']} turns, truncated and bootstrapped (continuing time), first game staggered"],
            ["opponents", f"100% current-policy self-play (current_fraction {sp['shared']['current_fraction']}); "
                          f"snapshots every {sp['shared']['snapshot_every']:,} (evaluation only)"],
            ["budget", f"{run['agent_transitions']:,} agent transitions = {run['env_turns']:,} joint steps = "
                       f"{run['updates']} updates; {run['snapshots']} snapshots"]]
    return "## 3. Pilot configuration\n\n" + table(["setting", "value"], rows) + "\n"


def sec_health(p: dict, h: List[dict]) -> str:
    rows = [[k, "pass" if v else "**FAIL**"] for k, v in p["checks"].items()]
    rows += [[k + " (learning)", "yes" if v else "no"] for k, v in p["learning"].items()]
    blocks = np.array_split(np.arange(len(h)), 10)
    keys = ("entropy", "approx_kl", "clipfrac", "explained_variance", "value_loss", "policy_loss", "grad_norm",
            "grad_norm_max", "value_absmax", "adv_std", "lr", "sps")
    br = []
    for b in blocks:
        e = [h[i] for i in b]
        br.append([f"{e[0]['agent_transitions'] / 1e6:.2f}–{e[-1]['agent_transitions'] / 1e6:.2f}M"]
                  + [f"{np.mean([x[k] for x in e]):.4g}" for k in keys])
    return ("## 4. Pilot training health\n\n" + table(["check", "result"], rows)
            + "\n\nPPO diagnostics by tenth of the run (means over updates; value statistics in scaled units "
              "(× 10 = resources); grad_norm before clipping):\n\n"
            + table(["transitions"] + list(keys), br)
            + "\n\n![PPO diagnostics](figures/ppo_diagnostics.png)\n")


def sec_rollout(h: List[dict]) -> str:
    blocks = np.array_split(np.arange(len(h)), 10)
    cats = ("LH", "TN", "SL", "SH", "Ward", "Recall", "Absent")
    rows = []
    for b in blocks:
        e = [h[i] for i in b]
        rows.append([f"{e[-1]['agent_transitions'] / 1e6:.2f}M"]
                    + [f"{np.mean([x[f'act_cat_{c}'] for x in e]):.3f}" for c in cats]
                    + [f"{np.mean([x[f'act_beh_{k}'] for x in e]):.3f}" for k in ("Hold", "Yield", "Trade", "Safe")]
                    + [f"{np.mean([x['act_pressure_of_lane'] for x in e]):.3f}",
                       f"{np.mean([x['mask_n_legal_mean'] for x in e]):.2f}",
                       f"{np.mean([x['mask_ward_legal_share'] for x in e]):.2f}",
                       f"{np.mean([x['mask_recall_legal_share'] for x in e]):.2f}",
                       f"{np.mean([x['mask_pressure_legal_share'] for x in e]):.3f}"])
    ep_rows = []
    for b in blocks:
        e = [h[i] for i in b]
        t = sum(x["ep_turns"] for x in e)
        g = sum(x["ep_games"] for x in e)
        side_t = 2 * max(t, 1)
        ep_rows.append([f"{e[-1]['agent_transitions'] / 1e6:.2f}M", g, t,
                        f"{1000 * sum(x['ep_deaths'] for x in e) / side_t:.1f}",
                        f"{1000 * sum(x['ep_recalls'] for x in e) / side_t:.1f}",
                        f"{sum(x['ep_recalls_interrupted'] for x in e) / max(sum(x['ep_recalls'] + x['ep_recalls_interrupted'] for x in e), 1):.2f}",
                        f"{1000 * sum(x['ep_trades_attempted'] for x in e) / side_t:.0f}",
                        f"{sum(x['ep_trades_connected'] for x in e) / max(sum(x['ep_trades_attempted'] for x in e), 1):.2f}",
                        f"{1000 * sum(x['ep_ganks'] for x in e) / side_t:.1f}",
                        f"{sum(x['ep_farm'] for x in e) / max(sum(x['ep_farm_available'] for x in e), 1e-9):.3f}",
                        f"{sum(x['ep_level_sum'] for x in e) / max(sum(x['ep_level_n'] for x in e), 1):.1f}"])
    return ("## 5. Training-time action and mask statistics (sampled actions of every rollout)\n\n"
            + table(["by", *cats, "Hold", "Yield", "Trade", "Safe", "Pressure/lane", "legal codes", "Ward legal",
                     "Recall legal", "Pressure legal"], rows)
            + "\n\nFinished training games (per 1000 champion-turns; recalls = completed Recalls; interrupted = share of "
              "Recall attempts interrupted by an enemy Trade; connected share of trade attempts; collection = farm / "
              "farm available; mean final level):\n\n"
            + table(["by", "games", "turns", "deaths", "recalls", "interrupted", "trade att.", "connected", "ganks",
                     "collection", "level"], ep_rows)
            + "\n\n![action shares during training](figures/action_shares.png)\n")


def sec_curve(ev: dict) -> str:
    cur = ev.get("curve")
    if not cur:
        return ""
    opps = [f"ref:{r}" for r in SUITE] + list(OOD)
    pts = sorted({_point_of(k.split("|")[0]) for k in cur})
    rows = []
    for p in pts:
        row = ["untrained" if p == 0 else f"{p / 1e6:.3f}M"]
        vals = []
        for o in opps:
            r = next((v for k, v in cur.items() if _point_of(k.split("|")[0]) == p and k.split("|")[1] == o), None)
            row.append(_ci(r) if r else "")
            if r and o.startswith("ref:"):
                vals.append(r["estimate"])
        row.append(_f(float(np.mean(vals))) if vals else "")
        rows.append(row)
    s = ev["curve_spec"]
    return (f"## 6. Learning curve against fixed opponents\n\nDiscounted lead of the checkpoint against each opponent "
            f"(resources; {s['episodes']} paired seeds, both seats, W = {s['window']}; ± = 95% interval over seeds). "
            "Last column: mean over the nine scripted opponents.\n\n"
            + table(["checkpoint"] + [_opp_label(o) for o in opps] + ["scripted mean"], rows)
            + "\n\n![learning curves](figures/learning_curves.png)\n")


def sec_headline(ev: dict) -> str:
    hd = ev.get("headline")
    if not hd:
        return ""
    s = ev["headline_spec"]
    fin, half = s["checkpoints"]
    rows = []
    for o in [f"ref:{r}" for r in SUITE] + list(OOD):
        a, b = hd.get(f"{fin}|{o}"), hd.get(f"{half}|{o}")
        if not a:
            continue
        d = a["estimate"] - b["estimate"] if b else float("nan")
        hw = 1.96 * math.sqrt(a["stderr"] ** 2 + b["stderr"] ** 2) if b else float("nan")
        rows.append([_opp_label(o), ROLE.get(o[4:], "out of distribution"), _ci(a), _f(a["seat_A"]), _f(a["seat_B"]),
                     _ci(b) if b else "", f"{d:+.2f} ± {hw:.2f}"])
    return (f"## 7. Headline: final (1.024M) and half-way (0.512M) networks\n\n{s['episodes']} paired seeds, W = "
            f"{s['window']}; seat A / seat B = the policy's mean value in each seat; last column = final − half-way "
            "(unpaired 95% interval).\n\n"
            + table(["opponent", "role", "final", "final seat A", "final seat B", "half-way", "final − half-way"], rows) + "\n")


def sec_history(ev: dict) -> str:
    hs = ev.get("history")
    if not hs:
        return ""
    rows = []
    for k, v in sorted(hs.items(), key=lambda kv: _point_of(kv[0].split("|")[1])):
        y = k.split("|")[1]
        rows.append([("untrained network" if y.startswith("initial") else f"snapshot {_point_of(y) / 1e6:.3f}M"),
                     _ci(v), _f(v["seat_A"]), _f(v["seat_B"])])
    s = ev["history_spec"]
    return (f"## 8. History: the final network against its own past\n\n{s['episodes']} paired seeds, W = {s['window']}.\n\n"
            + table(["opponent", "final's value", "seat A", "seat B"], rows) + "\n")


CATS = ("LH", "TN", "SL", "SH", "Ward", "Recall")


def sec_behavior(bh: dict) -> str:
    rec = bh["records"]
    out = ["## 9. Behaviour of the pilot checkpoints\n"]
    rows = []
    for k, v in rec.items():
        x, o = k.split("|")
        if o != "self":
            continue
        m = v["conditional"]["all present decisions"]["mass"]
        rows.append([x] + [f"{m[c]:.3f}" for c in CATS] + [f"{m['beh_' + b]:.3f}" for b in ("Hold", "Yield", "Trade",
                                                                                          "Safe")]
                    + [f"{m['Pressure']:.3f}", f"{m['entropy']:.2f}"])
    out.append("Self-play, all present decisions: mean probability mass by category (lane codes: LH / TN / SL / SH; "
               "behaviours as shares of the lane mass), Pressure mass, mean entropy (nats):\n")
    out.append(table(["checkpoint", *CATS, "Hold", "Yield", "Trade", "Safe", "Pressure", "entropy"], rows))
    fin = f"final:pilot:{PILOT_SEED}"
    cond = rec.get(f"{fin}|self", {}).get("conditional", {})
    crow = []
    for st, v in cond.items():
        m = v.get("mass")
        if not m:
            continue
        crow.append([st, v["n"]] + [f"{m[c]:.2f}" for c in CATS]
                    + [f"{m['beh_' + b]:.2f}" for b in ("Hold", "Yield", "Trade", "Safe")] + [f"{m['Pressure']:.2f}"])
    out.append("\nThe final network in self-play, by decision state (rows overlap; n = decisions):\n")
    out.append(table(["state", "n", *CATS, "Hold", "Yield", "Trade", "Safe", "Pressure"], crow))
    srow = []
    for k, v in rec.items():
        x, o = k.split("|")
        s = v["summary"]
        srow.append([x, o] + [_f(s[q][0], 2 if q in ("collection_rate", "away_share", "freeze_turns_share",
                                                    "frozen_on_share") else 1, sign=False)
                              for q in ("wards_per_1000", "wards_placed_per_1000", "recalls_per_1000",
                                        "deaths_per_1000", "trade_attempts_per_1000", "trades_connected_per_1000",
                                        "pressure_turns_per_1000", "collection_rate", "freeze_turns_share",
                                        "frozen_on_share")]
                    + [_f(s["lead_per_100_turns"][0]), _f(s["level_end"][0], 1, False)])
    out.append("\nGame summaries of the focal policy (per 1000 of its turns unless stated; lead = its gold lead per 100 "
               "turns):\n")
    out.append(table(["policy", "opponent", "wards", "placed", "recalls", "deaths", "trade att.", "connected",
                      "Pressure", "collection", "freezing", "frozen on", "lead/100", "level end"], srow))
    frow = []
    for k, v in rec.items():
        f = v["freezes"]
        x, o = k.split("|")
        est = f["established"]
        frow.append([x, o, est["n"], _f(f["established_per_1000_turns"][0], 1, False),
                     _f(est.get("length_mean", float("nan")), 1, False),
                     _f(est.get("useful_denial_share", float("nan")), 2, False),
                     _f(est["gold_diff"][0]) if est["n"] else "n/a",
                     _f(est["freezer_preserving_share"][0], 2, False) if est["n"] else "n/a",
                     _f(f["held_share"], 3, False)])
    out.append("\nFreezes with the policy as freezer (M4.4 definition: freeze state held >= 4 turns; denial = share of "
               "episodes in which the freezer out-earned the frozen-on side; preserving = Last Hit / Thin share of the "
               "freezer's held turns, an action pattern, not evidence):\n")
    out.append(table(["policy", "opponent", "freezes", "per 1000 turns", "mean length", "denial share",
                      "gold diff", "preserving", "held share"], frow))
    led = bh.get("ledger", {})
    lrow = []
    keys = ("earned", "unc", "d_trade", "d_min", "d_min_ret", "d_min_pos", "d_tower", "d_gank", "deaths", "recalls",
            "rec_intr", "wards", "ward_placed", "ward_intr", "vis_exposed", "held", "frozen", "at_W")
    for k, v in led.items():
        x, o = k.split("|")
        p = v["per_100_turns"]
        lrow.append([x, o] + [_f(p[q][0], 2, False) for q in keys] + [_f(p["G"][0])])
    out.append("\nM4.5b ledger (per 100 turns of the focal policy; unc = farm that died out of reach; d_* = damage taken "
               "in bars by source, minions split into retaliation / positional aggro; G = discounted lead over 300 "
               "turns):\n")
    out.append(table(["policy", "opponent", *keys, "G300"], lrow))
    return "\n".join(out) + "\n"


def sec_perf(p: dict, h: List[dict], prof: Dict[str, dict], sc: Optional[dict], plan: Optional[dict],
             ev: dict, bh: Optional[dict]) -> str:
    run = p["run"]
    out = ["## 10. Measured performance\n"]
    blocks = np.array_split(np.arange(len(h)), 5)
    tr = [[f"{h[b[-1]]['agent_transitions'] / 1e6:.2f}M", f"{np.mean([h[i]['sps'] for i in b]):,.0f}",
           f"{np.mean([h[i]['collect_s'] for i in b]):.2f}", f"{np.mean([h[i]['update_s'] for i in b]):.2f}"]
          for b in blocks]
    el = run["elapsed_s"]
    over = el - run["collect_s"] - run["update_s"]
    out.append(f"Pilot (one process, alone on the machine): {run['agent_transitions']:,} agent transitions in "
               f"{el / 60:.1f} min = **{run['transitions_per_s']:,.0f} agent transitions/s** "
               f"({run['env_turns_per_s']:,.0f} joint steps/s) end to end; collection {run['collect_s'] / 60:.1f} min, "
               f"PPO updates {run['update_s'] / 60:.1f} min, everything else (snapshots, diagnostics, logs) "
               f"{over:.0f} s; peak working set {run['memory'].get('peak_mb', float('nan')):.0f} MB.\n")
    out.append(table(["by", "transitions/s (collect + update)", "collect s / update", "PPO update s / update"], tr))
    if prof:
        rows = []
        for name, d in prof.items():
            m = d["per_update"]
            sh = m["shares"]
            rows.append([name, f"{m['total_s']:.2f}", f"{m['env_step_s']:.2f} ({sh['env_step_s']:.0%})",
                         f"{m['inference_s']:.2f} ({sh['inference_s']:.0%})",
                         f"{m['collect_other_s']:.2f} ({sh['collect_other_s']:.0%})",
                         f"{m['gae_batch_s']:.3f}", f"{m['update_s']:.2f} ({sh['update_s']:.0%})",
                         f"{m['env_turns_per_s_stepping_only']:,.0f}", f"{m['agent_transitions_per_s_end_to_end']:,.0f}",
                         f"{1000 * m['checkpoint_save_s']:.0f} ms / {m['checkpoint_bytes'] / 1024:.0f} KB"])
        out.append("\nOne update (4,096 agent transitions) broken down (instrumented single process, mean of 4 updates "
                   "after a warm-up):\n")
        out.append(table(["network", "total s", "env stepping", "inference", "other collection", "GAE + batch",
                          "PPO update", "joint steps/s (stepping only)", "transitions/s (end to end)",
                          "checkpoint save"], rows))
    if sc:
        rows = [[v["processes"], f"{v['per_process_mean']:,.0f}", f"{v['slowdown_vs_1']:.2f}",
                 f"{v['aggregate_tps']:,.0f}", f"{max(x or 0 for x in v['peak_mb']):.0f}"]
                for v in sc["scaling"].values()]
        out.append("\nConcurrent training processes (short runs from the untrained network, 8 updates each):\n")
        out.append(table(["processes", "transitions/s per process", "relative to 1", "aggregate transitions/s",
                          "peak MB per process"], rows))
    tim = ev.get("timing", {})
    if tim:
        rows = [[k, f"{v['wall_s'] / 60:.1f} min", v["workers"]] for k, v in tim.items()]
        if bh:
            rows.append(["behaviour + ledger", f"{bh['elapsed_s'] / 60:.1f} min", ""])
        out.append("\nEvaluation overhead of the pilot (wall clock):\n")
        out.append(table(["stage", "wall", "workers"], rows))
    if plan:
        rows = []
        for r in plan["estimates"]:
            rows.append([f"{r['budget']:,}", r["updates"], f"{r['one_run_alone_h'] * 60:.0f} min",
                         f"{r['one_run_under_11_h'] * 60:.0f} min", f"{r['seeds3_concurrent_h'] * 60:.0f} min",
                         f"{r['seeds6_concurrent_h'] * 60:.0f} min", f"{r['seeds10_concurrent_h'] * 60:.0f} min",
                         f"{r['seeds3_sequential_h']:.1f} h", f"{r['seeds6_sequential_h']:.1f} h"])
        out.append("\n**Extrapolated** training wall clock (measured pilot throughput × measured concurrency slowdown; "
                   "11 processes at most):\n")
        out.append(table(["transitions per seed", "updates", "1 run alone", "1 run among 11", "3 seeds concurrent",
                          "6 seeds concurrent", "10 seeds concurrent", "3 seeds sequential", "6 seeds sequential"], rows))
        mem = plan["memory_gb_at_full_cache"]
        caches = sorted({c for v in mem.values() for c in v}, key=int, reverse=True)
        out.append("\nMemory of concurrent training processes once their lane caches are full (measured per-process "
                   "working set × processes; this machine has 17 GB):\n")
        out.append(table(["processes"] + [f"lane_cache {int(c):,}" for c in caches],
                         [[k] + [f"{v[c]:.1f} GB" for c in caches] for k, v in mem.items()]))
        rows = []
        for k, o in plan["options"].items():
            rows.append([o["label"], o["seeds"], f"{o['transitions_per_seed']:,}",
                         f"{o['total_agent_transitions'] / 1e6:.1f}M", f"{o['lane_cache']:,}", f"{o['memory_gb']:.1f} GB",
                         f"{o['training_h'] * 60:.0f} min", f"{o['standard_evaluation_h'] * 60:.0f} min",
                         f"{o['strategic_battery_h'] * 60:.0f} min", f"{o['total_h']:.1f} h"])
        out.append("\nCampaign options (**extrapolated**; standard evaluation = curves, headline, history, tournament, "
                   "cross-play and behaviour of the pilot report's §9 design; strategic battery = its counterfactual "
                   "families, at the measured evaluation throughput):\n")
        out.append(table(["option", "seeds", "transitions per seed", "total", "lane cache", "peak memory", "training",
                          "standard evaluation", "strategic battery", "total"], rows))
    return "\n".join(out) + "\n"


def report() -> str:
    from .plan import plan as make_plan
    a = load_json("audit")
    s = load_json("smoke_train")
    se = load_json("smoke_eval")
    p = load_json("pilot_train")
    ev = load_json("pilot_eval") or {}
    bh = load_json("pilot_behavior")
    prof = {n: d for n, d in (("untrained", load_json("bench_profile_initial")), ("pilot final", load_json("bench_profile_final")))
            if d}
    sc = load_json("bench_scaling")
    pre = load_json("pilot_prefix_check")
    from .common import history
    h = history("pilot", PILOT_SEED)
    p["run"]["final_lr"] = f"{h[-1]['lr']:.2e}"
    try:
        pl = make_plan()
    except RuntimeError:
        pl = None
    fig_curves(ev)
    fig_series(h, (("entropy", "policy entropy (nats)", 1), ("approx_kl", "approximate KL", 1),
                   ("clipfrac", "clip fraction", 1), ("explained_variance", "explained variance", 1),
                   ("value_loss", "value loss (scaled)", 1), ("grad_norm", "gradient norm (pre-clip)", 1)),
               "ppo_diagnostics.png", "agent transitions (millions)")
    fig_series(h, tuple((f"act_cat_{c}", f"sampled share: {c}", 1) for c in ("LH", "TN", "SL", "SH", "Ward", "Recall")),
               "action_shares.png", "agent transitions (millions)")
    parts = ["# M4.5c pilot — generated tables\n",
             f"*Generated from results/m45c/*.json; commit {p['meta']['commit'][:10]} "
             f"(working tree dirty: {p['meta']['working_tree_dirty']}). Environment `GameConfig.m45_calibrated()`. "
             "Transitions are AGENT transitions (2 per joint environment step) unless stated. Prose and "
             "interpretation: docs/m4/m4_5c_pilot_report.md.*\n"]
    parts.append(sec_audit(a))
    parts.append(sec_smoke(s, se))
    parts.append(sec_config(p))
    if pre:
        parts.append(f"Determinism of the pilot itself: its first {pre['updates']} updates re-trained in a fresh "
                     f"directory give identical histories ({pre['identical_histories']}) and weights identical to the "
                     f"pilot's snapshot at {pre['stop_at']:,} ({pre['identical_weights_to_pilot_snapshot']}).\n")
    parts.append(sec_health(p, h))
    parts.append(sec_rollout(h))
    parts.append(sec_curve(ev))
    parts.append(sec_headline(ev))
    parts.append(sec_history(ev))
    if bh:
        parts.append(sec_behavior(bh))
    parts.append(sec_perf(p, h, prof, sc, pl, ev, bh))
    return write_md("report", "\n".join(x for x in parts if x))


__all__ = ["report", "fig_curves", "fig_series", "table"]
