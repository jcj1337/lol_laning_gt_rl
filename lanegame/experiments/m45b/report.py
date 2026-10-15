"""results/m45b/report.md: the generated tables of every M4.5b stage, read from results/m45b/*.json(.gz). Each section
states its source file; a missing stage is skipped. `python -m lanegame.experiments.m45b report`."""

from __future__ import annotations

from typing import List

import numpy as np

from . import common as Cm
from .common import fci, fmt, md_table


def _is_ci(x) -> bool:
    """[mean, half-width, n] (n an integer sample count), as `common.ci` returns."""
    return isinstance(x, (list, tuple)) and len(x) == 3 and isinstance(x[2], int) and not isinstance(x[2], bool)


def _f(x, d=2):
    if _is_ci(x):
        return fci(x, d)
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(_f(v, d) for v in x) + "]"
    if isinstance(x, bool):
        return "pass" if x else "FAIL"
    if isinstance(x, (int, float)):
        return f"{x:+.{d}f}"
    return str(x)


def trades_section() -> List[str]:
    d = Cm.load_json("trades")
    if d is None:
        return []
    from .trades import CRITERIA, criteria
    L = ["## Priority B: trading into waves (`trades.json`)", "",
         "Exact one-turn expectations (trade and tower dice enumerated, ganks off). delta = the Trade's HP exchange "
         "(B's loss minus A's, bars of each one's max HP) minus Hold's, same B action.", "",
         "### Pre-registered criteria, baseline", ""]
    crit = criteria(d["table"])
    L.append(md_table(["criterion", "statement", "value", "result"],
                      [[k, CRITERIA[k], _f(v["value"], 3), _f(v["pass"])] for k, v in crit.items()]))
    L += ["", "### Minion-parameter sweep (exact)", ""]
    rows = []
    for s in d["sweep_rows"]:
        o = s["overrides"]
        c = criteria(s["rows"])
        rows.append([o["minion_damage_alpha"], o["minion_damage_growth"], o["minion_retaliation_delay"],
                     sum(v["pass"] for v in c.values()), _f(c["B1"]["value"], 3), _f(c["B2"]["value"], 3),
                     _f(c["B3"]["value"], 3), _f(c["B4"]["value"], 3), _f(c["B6"]["value"], 3)])
    L.append(md_table(["alpha", "growth", "delay", "passed / 6", "B1 delta L2 24 HP", "B2 delta L2 8 HP",
                       "B3 delta L2 x1.76", "B4 L18 share", "B6 delta by wave"], rows))
    L += ["", "### Bench excerpt (baseline, B Hold, full HP)", ""]
    rows = []
    for r in d["table"]:
        if r["b"] != "hold" or r["hp"] != [1.0, 1.0]:
            continue
        la = r.get("A_landed", {})
        rows.append([r["level"], r["wave"], r["cover"], r["gap"], f"{r['dealt_bar'][0]:.3f}",
                     f"{r['minion_bar'][0]:.3f}", f"{r['u_ret'][0]:.2f} / {r['u_pos'][0]:.2f}",
                     f"{r['exchange'] - r['exchange_hold']:+.3f}", f"{la.get('minion_bar', 0):.3f}",
                     f"{r['d_farm'][0]:+.3f}"])
    L.append(md_table(["L", "wave", "cover", "gap", "dealt (bar)", "minion fire on A (bar)", "units ret / pos",
                       "delta", "minion | landed", "farm vs Hold"], rows))
    L += ["", "### Shadow price of HP (paired G, A at full HP minus A at 70%, per bar)", ""]
    L.append(md_table(["level", "continuation", "G20", "G50", "G100", "extra Recalls"],
                      [[r["level"], r["policy"], fci(r["price_G20"]), fci(r["price_G50"]), fci(r["price_G100"]),
                        fci(r["d_recalls"])] for r in d["hp_price"]]))
    L += ["", "### Trade on the first turn versus Hold (bench states, then g_farm v g_farm; paired, 200 seeds)", ""]
    L.append(md_table(["L", "wave", "cover", "gap", "G1", "G5", "G20", "G100", "deaths B"],
                      [[r["level"], r["wave"], r["cover"], r["gap"], fci(r["G1"]), fci(r["G5"]), fci(r["G20"]),
                        fci(r["G100"]), fci(r["deaths_B"])] for r in d["multi_turn"]]))
    nat = Cm.load_json("trades_natural")
    if nat:
        L += ["", "### Trade versus Hold from natural decision states (`trades_natural.json`; states are the units)",
              ""]
        rows = []
        for key, s in nat["settings"].items():
            for g in ("all|all", "<=8|all", "9-16|all", ">=17|all", "all|L<=5", "all|L>=13"):
                if g in s["groups"]:
                    x = s["groups"][g]
                    rows.append([key, g, x["G1"][2], fci(x["G1"]), fci(x["G20"]), fci(x["G50"]),
                                 fci(x["d_total_A@10"]), fci(x["d_total_B@10"])])
        L.append(md_table(["setting", "enemy blob | level", "states", "G1", "G20", "G50", "HP lost A @10",
                           "HP lost B @10"], rows))
    return L + [""]


def freeze_section() -> List[str]:
    d = Cm.load_json("freeze_grid")
    if d is None:
        return []
    L = ["## Priority A: freezing (`freeze_grid.json`, `freeze_local.json`, `freeze_sweep.json`)", "",
         "Freezer strategies x frozen-on responses, paired DEV seeds (40), 100 turns, payoff G100 (A's discounted "
         "lead). Premium = game value minus the value with only the non-freezing rows (g_shove, shove_recall); "
         "'split' = the selection-free split-sample estimate (equilibrium fitted on half the seeds).", ""]
    L += grid_rows(d["grid"])
    loc = Cm.load_json("freeze_local")
    if loc:
        L += ["", "### Local action values at the freeze (first action forced, then fz_guard_sh; paired 200 seeds; "
              "difference to Thin + Hold)", ""]
        rows = []
        for r in loc["local"]:
            st = r["state"]
            rows.append([f"L{st[0]} {st[1]} h{st[2]} hp{st[3]}", r["response"]] +
                        [f"{fmt(r['G5'][k][0], None, 1)} / {fmt(r['G100'][k][0], r['G100'][k][1], 1)}"
                         for k in ("TN+Trade", "LH+Hold", "SH+Hold", "SH+Trade")])
        L.append(md_table(["state", "response", "TN+Trade G5 / G100", "LH+Hold", "SH+Hold", "SH+Trade"], rows))
        rs = loc["reset"]
        L += ["", f"### After the opponent resets ({rs['n_states']} natural states; difference to keeping the guard)",
              ""]
        L.append(md_table(["plan", "G20", "G50", "G100"],
                          [[k, fci(rs["summary"]["G20"][k]), fci(rs["summary"]["G50"][k]),
                            fci(rs["summary"]["G100"][k])] for k in rs["summary"]["G20"]]))
    sw = Cm.load_json("freeze_sweep")
    if sw:
        L += ["", "### Parameter sweep (four states)", ""]
        L += grid_rows(sw["sweep"], with_over=True)
    return L + [""]


def grid_rows(grid, with_over: bool = False) -> List[str]:
    rows = []
    for g in grid:
        st, an = g["state"], g["analysis"]["G100"]
        pr = an["premium"]
        con = g["analysis"]["contrasts_G100"]
        cells = {(c["a"], c["b"]): c for c in g["cells"]}
        row = ([str(g.get("overrides", {}))] if with_over else []) + [
            f"L{st['level']} {st['gap']} h{st['held']} hp{st['hp']}", f"{an['value']:+.2f}",
            f"{pr['premium']:+.2f}", f"{pr['premium_split'][0]:+.2f}",
            ", ".join(f"{k} {v:.2f}" for k, v in an["A"].items()),
            ", ".join(f"{k} {v:.2f}" for k, v in an["B"].items()),
            fci(con["fz_guard_sh - fz_passive_sh"]["wk_shove_sh"]), fci(con["fz_guard_sh - g_shove"]["wk_shove_sh"]),
            f"{cells[('fz_guard_sh', 'wk_shove_sh')]['held_run0']:.1f} / "
            f"{cells[('fz_passive_sh', 'wk_shove_sh')]['held_run0']:.1f} / "
            f"{cells[('fz_passive_sh', 'wk_lh_sh')]['held_run0']:.1f}",
            f"{cells[('fz_guard_sh', 'wk_shove_sh')]['d_min_A@20']:.2f}"]
        rows.append(row)
    head = (["overrides"] if with_over else []) + [
        "state", "value", "premium", "premium split", "A equilibrium", "B equilibrium", "guard - passive v walk-up "
        "Shove", "guard - Shove v walk-up Shove", "held turns: guard v Shove / passive v Shove / passive v LH",
        "minion fire on A @20 (guard v Shove)"]
    return [md_table(head, rows)]


def vision_section() -> List[str]:
    d = Cm.load_json("vision")
    if d is None:
        return []
    L = ["## Priority C: Vision (`vision.json`, `vision_sweep.json`)", "",
         "### One ward from a situation (Ward on the first turn vs the base strategy's action; paired 200 seeds)", ""]
    rows = []
    for r in d["local"]:
        if not r["legal"]:
            continue
        rows.append([r["situation"], r["level"], r["gap"], r["base"], r["opponent"], fci(r["G1"]), fci(r["G20"]),
                     fci(r["G100"]), f"{r['placed_first']:.2f}",
                     f"{r['d_gank_A']['ward']:.2f} / {r['d_gank_A']['base']:.2f}"])
    L.append(md_table(["situation", "L", "gap", "base", "opponent", "G1", "G20", "G100", "placed",
                       "gank dmg ward / base (bars, 100 turns)"], rows))
    L += ["", "### Ward rules against their base strategy (paired 48 seeds, 200 turns)", ""]
    L.append(md_table(["start", "gap", "base", "rule", "opponent", "dG200", "wards", "placed", "interrupted",
                       "exposed share", "gank dmg", "gank deaths"],
                      [[c["start"], c["gap"], c["base"], c["rule"], c["opponent"], fci(c["dG200"]),
                        f"{c['wards']:.1f}", f"{c['placed']:.1f}", f"{c['interrupted']:.1f}",
                        f"{c['exposed_share']:.2f}", f"{c['d_gank'][0]:.2f} / {c['d_gank'][1]:.2f}",
                        f"{c['gank_deaths'][0]:.2f} / {c['gank_deaths'][1]:.2f}"] for c in d["strategies"]["contrasts"]]))
    sw = Cm.load_json("vision_sweep")
    if sw:
        L += ["", "### Vision parameter sweep", ""]
        rows = []
        for s in sw["sweep"]:
            loc = [r for r in s["local"] if r["legal"]]
            st = s["strategies"]
            rows.append([str(s["overrides"]), f"{np.mean([r['G100'][0] for r in loc]):+.3f}",
                         sum(1 for r in loc if r["G100"][0] - r["G100"][1] > 0),
                         sum(1 for r in loc if r["G100"][0] + r["G100"][1] < 0), len(loc),
                         f"{np.mean([c['dG200'][0] for c in st]):+.3f}",
                         sum(1 for c in st if c["dG200"][0] - c["dG200"][1] > 0),
                         sum(1 for c in st if c["dG200"][0] + c["dG200"][1] < 0), len(st)])
        L.append(md_table(["overrides", "one ward: mean dG100", "sig +", "sig -", "cells", "Ward rules: mean dG200",
                           "sig +", "sig -", "cells"], rows))
    return L + [""]


def zoning_section() -> List[str]:
    d = Cm.load_json("zoning")
    if d is None:
        return []
    L = ["## Priority D: Safe, Yield and threat-based zoning (`zoning.json`)", "",
         "### One turn: the price of stepping back (B's farm in resources and XP lost, HP saved in bars)", ""]
    rows = []
    for c in d["one_turn"]:
        rows.append([c["situation"], c["level"], c["gap"]] +
                    [f"{c[k]['farm']:+.3f} / {c[k]['xp']:+.1f} / {c[k]['hp_saved']:+.3f}"
                     for k in ("price_Yield_vs_Hold", "price_Safe_vs_Hold", "price_Yield_vs_Trade",
                               "price_Safe_vs_Trade")])
    L.append(md_table(["situation", "L", "gap", "Yield v A Hold", "Safe v A Hold", "Yield v A Trade",
                       "Safe v A Trade"], rows))
    L += ["", "### Multi-turn decomposition (B's earned resources and XP at 50 turns, paired 64 seeds)", ""]
    rows = []
    for x in d["matrix"]["decomposition"]:
        rows.append([x["situation"], x["level"], x["gap"], fci(x["passive_safe_earned_B@50"]),
                     fci(x["passive_yield_earned_B@50"]), fci(x["combat_bully_earned_B@50"]),
                     fci(x["combat_guard_earned_B@50"]), fci(x["threat_wary_earned_B@50"]),
                     fci(x["threat_wary_xp_B@50"], 1), f"{x['threat_conn_A']:.1f}", f"{x['exploit_safe']:+.2f}",
                     ", ".join(f"{a}: {b}" for a, b in x["B_best_reply_G100"].items())])
    L.append(md_table(["situation", "L", "gap", "always Safe (passive)", "always Yield (passive)",
                       "A bully (combat)", "A guard (combat)", "threat-only (A guard, B wary)",
                       "threat-only XP", "A conn. trades / 100", "exploit always-Safe (G100)",
                       "B best reply (G100)"], rows))
    return L + [""]


def matrix_section() -> List[str]:
    d = Cm.load_json("matrix")
    if d is None:
        return []
    L = ["## Priority E: integrated restricted games (`matrix.json`)", ""]
    for g in d["games"]:
        L += matrix_game(g)
    return L


def matrix_game(g) -> List[str]:
    L = [f"### {g['base']} from {g['start']} ({g['gap']}), {len(g['strategies'])} strategies, "
         f"{g['seeds'][2]} seeds, {g['turns']} turns, payoff {g['metric']}", ""]
    for tag, an in g["analysis"].items():
        L.append(f"- {tag}: value {an['value']:+.2f}; A {_mix(an['A'])}; B {_mix(an['B'])}; weakly dominated: "
                 f"{', '.join(an['dominated_rows']) or 'none'}")
        rm = sorted(an["row_mean"].items(), key=lambda x: -x[1])[:6]
        L.append(f"  - best row means: {', '.join(f'{k} {v:+.2f}' for k, v in rm)}")
    if g.get("bootstrap"):
        b = g["bootstrap"]
        L.append(f"- bootstrap (seeds) support frequency: "
                 f"{', '.join(f'{k} {v:.2f}' for k, v in b['support_freq'].items() if v > 0)}")
    return L + [""]


def _mix(d):
    return ", ".join(f"{k} {v:.2f}" for k, v in d.items())


def reference_section() -> List[str]:
    d = Cm.load_json("reference")
    if d is None:
        return []
    L = ["## A learned reference against the scripted strategies (`reference.json`; M4.4b aggro finals, evaluation "
         "only; on r2 / cal through an out-of-distribution observation adapter)", ""]
    L.append(md_table(["preset", "network", "opponent", "G200 for the network (both seats)", "network deaths",
                       "opponent deaths", "connected trades", "Shove turns"],
                      [[r["base"], r["net_seed"], r["opponent"], fci(r["G200_net"]),
                        f"{r['ledger_net_in_A']['deaths_A']:.2f}", f"{r['ledger_net_in_A']['deaths_B']:.2f}",
                        f"{r['ledger_net_in_A']['trade_conn_A']:.1f}", f"{r['ledger_net_in_A']['n_SH_A']:.1f}"]
                       for r in d["rows"]]))
    return L + [""]


def timing_section(name="timing", title="Priority F: movement and timing diagnostics") -> List[str]:
    d = Cm.load_json(name)
    if d is None:
        return []
    if "totals" not in d:
        d = d.get("timing", {})
    if not d:
        return []
    L = [f"## {title} (`{name}.json`)", ""]
    rows = []
    for kind, a in d["totals"].items():
        n, tp = a["turns"], a.get("twopass_turns", 1)
        c = max(a.get("conn", 0), 1)
        ret = sum(a.get("ret_centi_" + k, 0) for k in ("start", "tick0", "end"))
        rows.append([kind, int(n), f"{a.get('probe_tick_diff', 0) / n:.4f}", f"{a.get('target_tick_diff', 0) / n:.4f}",
                     f"{a.get('twopass_end_diff', 0) / n:.4f}", f"{a.get('twopass_not_fixed', 0) / tp:.3f}",
                     f"{a.get('cross_uncovered', 0) / n:.4f}", f"{a.get('cross_centi', 0) / n:.3f}",
                     f"{a.get('conn_end', 0) / c:.3f}", f"{a.get('ret_centi_end', 0) / max(ret, 1):.3f}",
                     f"{a.get('fire_ticks', 0) / n:.4f}", f"{a.get('wave_preserved', 0) / max(a.get('fire_ticks', 1), 1):.3f}"])
    L.append(md_table(["play", "champion-turns", "probe front error / ch-turn", "of which minion targeting",
                       "two-pass end-cell change / ch-turn", "two-pass not a fixed point / turn",
                       "uncovered crossings / ch-turn", "missed fire cHP / ch-turn", "trades contacting only at the "
                       "end", "their share of retaliation", "minion fire ticks / ch-turn",
                       "fire ticks that kept a dying front alive"], rows))
    rows = []
    for r in d.get("per_pairing", []):
        a = r["acc"]
        n = a["turns"]
        av = max(a.get("avail_farm", 0), 1e-9)
        rows.append([" v ".join(r["pair"]), f"{a.get('probe_tick_diff', 0) / n:.3f}",
                     f"{a.get('target_tick_diff', 0) / n:.4f}", f"{a.get('twopass_end_diff', 0) / n:.3f}",
                     f"{a.get('oor_farm', 0) / av:.3f}" if "oor_farm" in a else "",
                     f"{a.get('collected_farm', 0) / av:.3f}" if "collected_farm" in a else ""])
    if rows:
        L += ["", md_table(["pairing", "probe front error / ch-turn", "targeting only", "two-pass end-cell change",
                            "farm dying out of reach (share of available)", "farm collected (share)"], rows)]
    return L + [""]


def sensitivity_section(name="sensitivity", title="Joint sensitivity") -> List[str]:
    d = Cm.load_json(name)
    if d is None:
        return []
    from .sensitivity import scalar
    keys = ["trade_pass", "trade_B1", "trade_mt_L2_24_G20", "trade_mt_L2_8_G20", "freeze_L10_large_h12_premium_split",
            "freeze_L4_equal_h12_premium_split", "freeze_L10_large_h12_guard_minus_passive_vs_shove",
            "vision_neutral_G50", "vision_frozen_on_G50", "zoning_passive_safe_cost", "zoning_threat_only",
            "matrix_shove_margin", "matrix_n_support"]
    L = [f"## {title} (`{name}.json`)", ""]
    rows = []
    for r in d["rows"] + d.get("one_at_a_time", []):
        rows.append([", ".join(f"{k}={v}" for k, v in r["overrides"].items())] +
                    [_f(r[k]) if k in r else "" for k in keys])
    L.append(md_table(["configuration"] + keys, rows))
    return L + [""]


def candidate_section() -> List[str]:
    d = Cm.load_json("candidate_dev")
    if d is None:
        return []
    from .trades import CRITERIA
    L = ["## The candidate on the DEV battery (`candidate_dev.json`)", "", "### Trade criteria", ""]
    L.append(md_table(["criterion", "statement", "value", "result"],
                      [[k, CRITERIA[k], _f(v["value"], 3), _f(v["pass"])] for k, v in d["trades"]["criteria"].items()]))
    L += ["", "### Freeze grid", ""] + grid_rows(d["freeze"])
    L += ["", "### Ward rules", ""]
    L.append(md_table(["start", "gap", "base", "rule", "opponent", "dG200", "placed", "interrupted"],
                      [[c["start"], c["gap"], c["base"], c["rule"], c["opponent"], fci(c["dG200"]),
                        f"{c['placed']:.1f}", f"{c['interrupted']:.1f}"] for c in d["vision"]["strategies"]["contrasts"]
                       if c["rule"] in ("favorable", "exposed", "always")]))
    L += ["", "### Zoning decomposition", ""]
    L.append(md_table(["situation", "L", "gap", "always Safe", "A guard (combat)", "threat-only", "exploit Safe"],
                      [[x["situation"], x["level"], x["gap"], fci(x["passive_safe_earned_B@50"]),
                        fci(x["combat_guard_earned_B@50"]), fci(x["threat_wary_earned_B@50"]),
                        f"{x['exploit_safe']:+.2f}"] for x in d["zoning"]["matrix"]["decomposition"]]))
    L += [""]
    for g in d["matrix"]:
        L += matrix_game(g)
    return L


def validation_section() -> List[str]:
    d = Cm.load_json("validation")
    if d is None:
        return []
    from .validate import HCRITERIA
    L = ["## Holdout validation (`validation.json`; HOLDOUT seeds 88,000-88,499 and HOLDOUT_GRID states)", ""]
    for name, v in d["configs"].items():
        L += [f"### {name}", ""]
        if "trades" in v:
            L.append(md_table(["criterion", "statement", "value", "result"],
                              [[k, HCRITERIA[k], _f(x["value"], 3), _f(x["pass"])]
                               for k, x in v["trades"]["criteria"].items()]))
            L += [""]
        if "freeze" in v:
            L += grid_rows(v["freeze"]) + [""]
        if "zoning" in v:
            L.append(md_table(["situation", "L", "gap", "always Safe", "A guard (combat)", "threat-only",
                               "exploit Safe"],
                              [[x["situation"], x["level"], x["gap"], fci(x["passive_safe_earned_B@50"]),
                                fci(x["combat_guard_earned_B@50"]), fci(x["threat_wary_earned_B@50"]),
                                f"{x['exploit_safe']:+.2f}"] for x in v["zoning"]["decomposition"]]))
            L += [""]
        if "vision_strategies" in v:
            st = v["vision_strategies"]
            L.append(f"Ward rules (holdout): mean dG200 {np.mean([c['dG200'][0] for c in st]):+.3f}, "
                     f"significantly positive {sum(1 for c in st if c['dG200'][0] - c['dG200'][1] > 0)}, negative "
                     f"{sum(1 for c in st if c['dG200'][0] + c['dG200'][1] < 0)} of {len(st)}")
            L += [""]
        if "matrix" in v:
            L += matrix_game(v["matrix"]["s0"])
            for g in v["matrix"]["natural"]:
                an = g["analysis"]["raw"]
                L.append(f"- natural state (seed {g['seed']}, turn {g['turn']}): value {an['value']:+.2f}; A "
                         f"{_mix(an['A'])}; B {_mix(an['B'])}")
            L += [""]
    return L


def build() -> str:
    L = ["# M4.5b results (generated)", "",
         "Tables of every M4.5b stage, generated by `python -m lanegame.experiments.m45b report` from "
         "`results/m45b/*.json(.gz)`. Interpretation: docs/m4/m4_5/m4_5b_calibration_report.md. Intervals are 95% t "
         "intervals over seeds (paired differences) or over states, as labelled; G<h> = A's discounted gold lead over "
         "h turns (gamma 0.99).", ""]
    for sec in (trades_section, freeze_section, vision_section, zoning_section, matrix_section, reference_section,
                timing_section,
                sensitivity_section,
                lambda: sensitivity_section("sensitivity_cal", "Joint sensitivity on the corrected mechanics"),
                candidate_section,
                lambda: timing_section("candidate_dev", "Movement and timing diagnostics, candidate"),
                validation_section):
        L += sec()
    return "\n".join(L)


def write() -> str:
    return Cm.write_md("report", build())


__all__ = ["build", "write"]
