"""Generated tables of M5 (results/m5/report.md): the M5a readiness record, the throughput benchmark and the budget;
the pre-registration's amendment chain; after the campaign stages, the training health and runtime, monitoring, the
pre-registered M5d analysis (`analysis.analyse`, with every endpoint's status), the lane-exit panel, S8
(`strategic.assemble`) and the future-stage plan (`plan_future`). Every table states what is complete, partial or
not run; nothing missing is shown as a result."""

from __future__ import annotations

from typing import List, Optional

from . import common as C


def _t(head: List[str], rows: List[list]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def _f(x, d=0):
    try:
        return f"{float(x):,.{d}f}"
    except (TypeError, ValueError):
        return "n/a"


def _ci(st: Optional[dict], d: int = 2) -> str:
    if not st or st.get("n", st.get("n_seeds", 0)) == 0 or st.get("mean") is None:
        return "n/a"
    return C.fci(st, d)


def readiness_section() -> str:
    rec = C.load("readiness") or {}
    parts = rec.get("parts", {})
    lines = ["## M5a readiness", "", f"Protocol: `{(rec.get('protocol') or {}).get('document')}` "
             f"(SHA-256 `{(rec.get('protocol') or {}).get('sha256', '')[:16]}…`).", ""]
    rows = []
    for k in sorted(parts, key=lambda x: int(x[1:])):
        p = parts[k]
        ch = p.get("checks") or {}
        rows.append([k, "pass" if p.get("passed") else "FAIL", f"{sum(bool(v) for v in ch.values())}/{len(ch)}",
                     p.get("time", "")])
    lines += [_t(["check", "result", "items", "time"], rows), ""]
    for k in sorted(parts, key=lambda x: int(x[1:])):
        fails = [n for n, v in (parts[k].get("checks") or {}).items() if not v]
        if fails:
            lines += [f"- {k} failed: " + "; ".join(fails)]
    r9 = (parts.get("R9") or {}).get("data", {})
    if r9.get("configs"):
        rows = [[n, ", ".join(f"{a} x{b}" for a, b in v["mix"].items()),
                 ", ".join(f"{a} {_f(b)}" for a, b in v.get("per_run_by_kind", {}).items()), _f(v.get("aggregate")),
                 _f(v.get("window_s")), _f(max(m for m in v.get("peak_mb", []) if m)) if any(v.get("peak_mb", []))
                 else "n/a"]
                for n, v in r9["configs"].items()]
        lines += ["", "### Concurrency benchmark (R9)", "",
                  _t(["config", "runs", "transitions/s per run (window)", "aggregate", "window s", "peak MB per run"],
                     rows), ""]
    r10 = (parts.get("R10") or {}).get("data", {})
    if r10.get("rates"):
        lines += ["### Instrument costs (R10)", "",
                  _t(["cost class", "turns per worker-second"], [[k, _f(v)] for k, v in sorted(r10["rates"].items())]),
                  ""]
    return "\n".join(lines)


def budget_section() -> str:
    b = C.load("budget")
    if not b:
        return ""
    t = b["throughput"]
    lines = ["## Budget (M5a record: measured rates, design sizes)", "",
             f"Workers: {b['workers']}. Training throughput per run: om {_f(t['om'])}, sp {_f(t['sp'])} transitions/s "
             f"(benchmark om {_f(t['om_bench'])}, sp {_f(t['sp_bench'])}; calibration to full-length runs x"
             f"{t['calibration']:.2f}).", ""]
    tr = b["training"]
    lo, hi = tr.get("wall_hours_range", [tr["wall_hours"], tr["wall_hours"]])
    rows = [["M5b training (12 runs x 32.768M; full-run evidence to conservative calibration)",
             f"{_f(lo, 1)}–{_f(hi, 1)}"],
            ["  the 16.384M midpoints reached after about", f"{_f(lo / 2, 1)}–{_f(hi / 2, 1)}"]]
    for k, v in b["evaluation"].items():
        rows.append([k, _f(v["wall_hours"], 1)])
    for k, v in (b.get("attackers") or {}).items():
        rows.append([f"attackers, {k} budget (18 / 12 attackers; M5a's plan)",
                     f"{_f(v['wall_hours_18'], 1)} / {_f(v['wall_hours_12_without_adaptation_test'], 1)}"])
    lines += [_t(["part", "wall hours"], rows), ""]
    return "\n".join(lines)


def prereg_section() -> str:
    d = C.load("preregistration")
    if not d:
        return ""
    lines = ["## Pre-registration and amendments", "",
             f"Original frozen record: `{d['record']['document']}` SHA-256 `{d['record']['sha256'][:16]}…` "
             f"({d['meta']['timestamp']}).", ""]
    rows = [[a["id"], a["date"], a["previous_sha256"][:16] + "…", a["document"]["sha256"][:16] + "…",
             a["amendment_document"]["document"], a["reason"]] for a in d.get("amendments") or []]
    if rows:
        lines += [_t(["amendment", "date", "replaces", "new version", "rationale", "reason"], rows), ""]
    act = C.active_version(d)
    problems = C.verify_chain(d)
    lines += [f"Active version: {act['version']} (`{act['sha256'][:16]}…`). Chain: "
              f"{'verified' if not problems else 'PROBLEMS: ' + '; '.join(problems)}.", ""]
    return "\n".join(lines)


def training_section() -> str:
    d = C.load("train_main")
    if not d:
        return "## M5b training\n\nNot run.\n"
    w = C.load("wall_main") or {}
    lines = ["## M5b training", "",
             f"Health check: {'all passed' if d['all_passed'] else 'NOT all passed'}; complete runs "
             f"{len(d['complete'])}/12; technical failures: {d['technical_failures']['pairs'] or 'none'}; "
             f"live alarms: {d.get('live_alarms') or 'none'}.", ""]
    if d.get("failed_checks"):
        lines += ["Failed checks: " + "; ".join(d["failed_checks"]), ""]
    rows = []
    for n, r in d["runs"].items():
        if r.get("status") != "complete":
            rows.append([n, r.get("status"), "", "", "", "", "", "", ""])
            continue
        st = d["stability"][n]
        sh = r["shares"]
        rows.append([n, "complete", _f(r["agent_transitions"]), _f(r["updates"]), _f(r["elapsed_s"] / 3600, 2),
                     _f(r["transitions_per_s"]), f"{st['entropy_min']:.2f} / {st['entropy_last']:.2f}",
                     f"{st['approx_kl_max_10_update_mean']:.4f}",
                     f"{sh['game_share_fixed_opponent']:.3f} / {sh['learner_transition_share_fixed_opponent']:.3f}"])
    lines += [_t(["run", "status", "transitions", "updates", "hours", "transitions/s", "entropy min / last",
                  "max 10-update KL", "fixed-opp. game / transition share"], rows), ""]
    if w:
        lines += [f"Campaign wall time: {_f(w.get('wall_s', 0) / 3600, 2)} h ({w.get('workers')} workers; started "
                  f"{w.get('started')}, finished {w.get('finished')}); stopped by the coordinator: "
                  f"{w.get('stopped') or 'no'}.", ""]
    return "\n".join(lines)


def monitoring_section() -> str:
    m = C.load("monitoring_main")
    if not m:
        return "## M5b monitoring\n\nNot run.\n"
    lines = ["## M5b monitoring (never selects a checkpoint, never changes an endpoint)", "",
             "Learning curves: the lead at 114 against the fixed panel (seed mean ± 95% t interval).", ""]
    rows = []
    for k, r in m["curve_summary"].items():
        a, n = k.split("|")
        rows.append([a, f"{int(n) / 1e6:.3f}M", _ci(r.get("panel mean")),
                     *[_ci(r.get(nm), 1) for nm in ("start:s", "base:o", "g_farm", "g_shove", "g_bully")]])
    lines += [_t(["arm", "transitions", "panel mean", "start:s", "base:o", "g_farm", "g_shove", "g_bully"], rows), ""]
    rows = [[k.split("|")[0], k.split("|")[1], k.split("|")[2], _ci(v), v.get("n_seeds"),
             v.get("seeds_final_loses_significantly")] for k, v in m["history"].items()]
    lines += ["History (64 pairs × 500 turns):", "",
              _t(["arm", "comparison", "readout", "mean ± 95%", "seeds", "seeds where the final loses (game 95%)"],
                 rows), ""]
    lines += ["Convergence probe (final − 75% snapshot, lead at 114; \"still improving\" when ≥ +0.25 with L > 0): "
              + "; ".join(f"{a}: {v.get('verdict')} ({_ci(v) if v.get('status') == 'complete' else v.get('status')})"
                          for a, v in m["convergence"].items()), "",
              "Cycling flag (M4.5c's rule: the final losing significantly to an earlier snapshot on ≥ 2 seeds): "
              + "; ".join(f"{a}: {'FLAG' if v['flag'] else 'no'} {v['seeds_by_snapshot']}"
                          for a, v in m["cycling"].items()), ""]
    return "\n".join(lines)


def analysis_section() -> str:
    a = C.load("analysis_main")
    if not a:
        return "## M5d analysis\n\nNot run.\n"
    eps = a["endpoints"]
    lines = ["## M5d pre-registered analysis (amendment 1)", ""]
    h = a["headline"]
    if h.get("outcome"):
        lines += [f"**Headline: outcome {h['outcome']}, {h['name']}** — {h['statement']}. Scope: {h['scope']}.", ""]
    else:
        lines += [f"**Headline: {h['name']}** ({h.get('status')}).", ""]
    rows = []
    for k, e in eps.items():
        ps = " ".join(f"{float(v):+.2f}" for v in e.get("per_seed", {}).values())
        var = e.get("variance") or {}
        rows.append([k, e["role"], e["status"], _ci(e) if e.get("n_seeds") else "n/a", e["class"], ps,
                     f"{var.get('tau', float('nan')):.2f} / {var.get('mean_game_se', float('nan')):.2f}"
                     if var else "n/a", e["description"]])
    lines += [_t(["endpoint", "role", "status", "mean ± 95% (seeds)", "interval class", "per seed", "tau / mean game SE",
                  "definition"], rows), ""]
    sec = a["secondary"]
    hs = sec["holm"]
    lines += [f"Secondary family: {len(sec['family'])} hypotheses, Holm at familywise {sec['alpha']}; "
              f"{'final' if hs['final'] else 'NOT final (pending: ' + ', '.join(hs.get('pending', [])) + ')'}.", ""]
    rows = []
    for ep, r in sec["endpoints"].items():
        hyp = "; ".join(f"{h}: p {x['p']:.4f}, interval {'yes' if x['interval'] else 'no'}, Holm {x['holm']}"
                        for h, x in (r.get("hypotheses") or {}).items())
        rows.append([ep, r["status"], r.get("interval_class", "n/a"), r["claim"], hyp])
    lines += [_t(["endpoint", "status", "unadjusted interval class", "claim after Holm", "hypotheses"], rows), ""]
    lines += [panel_section(a.get("panel") or {}), versus_start_section(a.get("versus_start") or {}),
              switch_section(a.get("switch_windows") or {}), belief_section(a.get("belief") or {})]
    return "\n".join(lines)


def switch_section(sw: dict) -> str:
    if not sw:
        return ""
    rows = [[opp, w, _ci(st), st.get("n_seeds")] for opp, d in sw.items() for w, st in d.items()]
    return "\n".join(["### Exploratory: misspecified opponents by turn window (om − sp, reward summed in the window)", "",
                      _t(["opponent", "window", "mean ± 95%", "seeds"], rows), ""])


def belief_section(b: dict) -> str:
    if not b:
        return ""
    turns = (5, 10, 20, 40, 94)
    rows = []
    for opp, d in b.items():
        for kind in ("true-type mass", "identified", "catch-all mass"):
            cells = [_ci(d.get(f"{kind} at {t}")) for t in turns]
            if any(c != "n/a" for c in cells):
                rows.append([opp, kind, *cells])
    return "\n".join(["### Exploratory: the om finals' belief (descriptive; the recorded traces)", "",
                      _t(["opponent", "quantity", *[f"turn {t}" for t in turns]], rows), ""])


PANEL_ORDER = ("D0 (earned lead at 94)", "E_20 (lead at 114)", "E_20 - D0", "level_diff", "xp_diff", "hp_diff",
               "stock_diff", "focal away (share)", "opponent away (share)", "focal dead (share)",
               "opponent dead (share)", "returns_in focal - opponent", "since_return focal - opponent",
               "unspent focal - opponent", "invested focal - opponent")


def panel_section(p: dict) -> str:
    if not p:
        return ""
    lines = ["### Lane exit: D0 → E_20 with the non-gold panel (turn 94; focal orientation; seed mean ± 95%)", ""]
    for grp, d in p.items():
        q = d["quantities"]
        if not q:
            lines += [f"{grp}: no complete records.", ""]
            continue
        arms = list(next(iter(q.values())).keys())
        rows = [[k, *[_ci(q[k].get(a)) for a in arms]] for k in PANEL_ORDER if k in q]
        lines += [f"**{grp}** ({d['status']}):", "", _t(["quantity", *arms], rows), ""]
    return "\n".join(lines)


def versus_start_section(v: dict) -> str:
    if not v:
        return ""
    rows = [[k, _ci(x), x.get("n_seeds")] for k, x in v.items()]
    return "\n".join(["### Exploratory: each arm against the common start ext/baseline:s (lead at 114)", "",
                      _t(["contrast | family", "mean ± 95%", "seeds"], rows), ""])


def strategic_section() -> str:
    s = C.load("strategic_main")
    if not s:
        return "## M5c strategic evaluation (S8)\n\nNot run (the battery and cells stages await approval).\n"
    lines = ["## M5c strategic evaluation (S8, descriptive)", ""]
    rows = []
    for arm, vs in s["replication"]["replication"].items():
        for v, r in vs.items():
            rows.append([arm, v, r["replication"]["verdict"], f"{r['replication']['n_holds']}/{r['replication']['n_seeds']}",
                         " ".join(f"{float(x):+.2f}" for x in r["per_seed_gap"].values())])
    lines += [_t(["arm", "verdict", "replication", "seeds holding / complete", "per-seed gap"], rows), ""]
    for f, c in s["cross_arm"].items():
        lines += [f"Cross-arm {f}: {c['status']}."]
    return "\n".join(lines) + "\n"


def plan_section() -> str:
    p = C.load("plan_future")
    if not p:
        return ""
    hb = p["horizon_breakdown"]
    from collections import defaultdict
    agg, turns = defaultdict(float), defaultdict(float)
    for r in hb["rows"]:
        k = (r["part"], r["purpose"], r["horizon"])
        agg[k] += r["worker_h"] or 0.0
        turns[k] += r["turns"]
    rows = [[k[0], k[1], k[2], f"{turns[k] / 1e6:.2f}", _f(agg[k] / hb["workers"], 2)] for k in sorted(agg)]
    at = p["attackers"]
    return "\n".join(["## Future stages (proposal, not authorized): work by horizon and purpose", "",
                      _t(["part", "purpose", "horizon", "M turns", "wall h (6 workers)"], rows), "",
                      f"Attackers at the quarter budget: core 12 — training {_f(at['core_12']['training_wall_h'], 1)} h, "
                      f"evaluation {_f(at['core_12']['eval_wall_h'], 1)} h; optional om_table 6 — +"
                      f"{_f(at['optional_om_table_6']['training_wall_h_added'], 1)} h training, +"
                      f"{_f(at['optional_om_table_6']['tables_wall_h'], 2)} h tables, +"
                      f"{_f(at['optional_om_table_6']['eval_wall_h_added'], 1)} h evaluation.", ""])


def runtime_actual() -> dict:
    """Measured wall and worker time of every completed campaign stage against the M5a estimate (budget.json), and the
    ratio used to update the estimates of the stages still to come (same machine, same instruments)."""
    b = C.load("budget") or {}
    out = {}
    w = C.load("wall_main")
    if w and b:
        lo, hi = b["training"]["wall_hours_range"]
        out["M5b training"] = {"wall_h": w["wall_s"] / 3600, "estimate_h": [lo, hi],
                               "run_wall_h": w.get("run_wall_h")}
    for stage, rec, key in (("M5b monitor", "wall_M5b_monitor", "M5b_monitor"),
                            ("M5d families", "wall_M5d_families", "M5d_families")):
        r = C.load(rec)
        e = (b.get("evaluation") or {}).get(key)
        if r and e:
            out[stage] = {"wall_h": r["wall_s"] / 3600, "estimate_wall_h": e["wall_hours"],
                          "worker_h": r["job_seconds"] / 3600, "estimate_worker_h": e["worker_hours"],
                          "worker_ratio": r["job_seconds"] / 3600 / e["worker_hours"],
                          "wall_ratio": r["wall_s"] / 3600 / e["wall_hours"]}
    ev = [v["worker_ratio"] for k, v in out.items() if "worker_ratio" in v]
    out["evaluation_rate_ratio"] = float(sum(ev) / len(ev)) if ev else None
    return out


def runtime_section() -> str:
    r = runtime_actual()
    if not r:
        return ""
    rows = []
    for k, v in r.items():
        if not isinstance(v, dict):
            continue
        est = v.get("estimate_h") or v.get("estimate_wall_h")
        est = f"{est[0]:.1f}–{est[1]:.1f}" if isinstance(est, list) else _f(est, 2)
        rows.append([k, _f(v["wall_h"], 2), est, _f(v.get("worker_ratio"), 2) if "worker_ratio" in v else ""])
    lines = ["## Runtime: measured against the M5a estimate", "",
             _t(["stage", "wall h", "estimate (wall h)", "worker-time ratio (measured / R10 rates)"], rows), ""]
    p = C.load("plan_future")
    ratio = r.get("evaluation_rate_ratio")
    if p and ratio:
        wb = p["horizon_breakdown"]["wall_h_by_part"]
        at = p["attackers"]
        tr = C.load("train_main") or {}
        bt = (C.load("budget") or {}).get("throughput") or {}
        rr = {a: sum(tr["runs"][f"{a}_s{s}"]["transitions_per_s"] for s in C.SEEDS) / len(C.SEEDS) / bt[f"{a}_bench"]
              for a in C.ARMS if tr.get("runs") and f"{a}_bench" in bt}
        lines += [f"Measured evaluation worker time was ×{ratio:.2f} the R10-based estimate (the instruments ran about "
                  f"{(1 - 1 / ratio) * 100:.0f}% slower than the readiness rates). Updated estimates for the stages "
                  f"still to come (R10-based estimate × {ratio:.2f}): M5c battery {wb['M5c battery'] * ratio:.1f} h "
                  f"(estimate {wb['M5c battery']:.1f}), M5c cells {wb['M5c cells'] * ratio:.1f} h "
                  f"({wb['M5c cells']:.1f}), core attacker evaluation "
                  f"{at['core_12']['eval_wall_h'] * ratio:.1f} h; core attacker training "
                  f"{at['core_12']['training_wall_h']:.1f} h (R9 rates; the main training ran at "
                  + ", ".join(f"{a} ×{v:.2f}" for a, v in rr.items()) + " of its R9 benchmark rates).", ""]
    return "\n".join(lines)


def report() -> str:
    text = "\n".join(["# M5 generated tables", "", "*Generated by `python -m lanegame.experiments.rl.m5 report`.*", "",
                      prereg_section(), training_section(), runtime_section(), monitoring_section(),
                      analysis_section(), strategic_section(), plan_section(), readiness_section(), budget_section()])
    C.write_md("report", text)
    return text


__all__ = ["report"]
