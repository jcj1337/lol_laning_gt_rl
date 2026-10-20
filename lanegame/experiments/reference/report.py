"""results/reference/report.md: the generated tables that the reference quotes, from results/reference/*.json."""

from __future__ import annotations

from typing import List

from ..m45b.common import fci, md_table
from .common import POLICY, load_json, write_md


def _examples(e: dict) -> List[str]:
    t = e["tables"]
    L = ["## Tables (`examples`)", "",
         md_table(["L", "XP to next", "cumulative XP", "max HP (cHP)", "HP unit", "AP(L, 0)"],
                  [[r["L"], r["xp_next"] or "-", r["xp_cum"], r["hp_max"], r["hp_unit"], f"{r['ap']:.3f}"]
                   for r in t["progression"]]), "",
         md_table(["S invested"] + [str(r["S"]) for r in t["multiplier"]],
                  [["M(S)"] + [f"{r['M']:.3f}" for r in t["multiplier"]]]), "",
         "Wave capacity C and the Thin / Slow Push cap round(C / 2) by level and invested resources (Last Hit is 1 "
         "at every power):", "",
         md_table(["L \\ S"] + [str(S) for S in t["invest"]],
                  [[L] + [f"{t['capacity'][str(L)][str(S)]['C']} / {t['capacity'][str(L)][str(S)]['half']}"
                          for S in t["invest"]] for L in t["levels"]]), ""]
    L += ["## Four wave actions from the neutral start (`examples`)", ""]
    for name, d in e["wave_actions"].items():
        rows = [[r["turn"], "/".join(map(str, r["budgets"])), r["lane"], f"{r['gold'][0]:.3f} / {r['gold'][1]:.3f}",
                 f"{r['xp'][0]} / {r['xp'][1]}", f"{r['crash_hp'][0]} / {r['crash_hp'][1]}",
                 f"{r['minion_cHP'][0]} / {r['minion_cHP'][1]}"] for r in d["rows"][1:]]
        tt = d["totals"]
        L += [f"### A plays {name} + Hold, B plays Last Hit + Hold", "",
              md_table(["turn", "budgets A/B", "lane (global cells)", "gold A/B", "XP A/B", "crash HP A/B",
                        "minion damage on A/B (cHP)"], rows), "",
              f"Totals: gold {tt['gold'][0]:.3f} / {tt['gold'][1]:.3f}, XP {tt['xp'][0]} / {tt['xp'][1]}, crashes "
              f"{tt['crash'][0]} / {tt['crash'][1]}, minion damage {tt['minion_cHP'][0]} / {tt['minion_cHP'][1]} cHP.",
              ""]
    vs = e["wave_actions_vs_shove"]
    L += ["The same four A actions against B on Shove + Hold (A − B gold over 8 turns): "
          + ", ".join(f"{k} {v['gold'][0] - v['gold'][1]:+.4f}" for k, v in vs.items()) + ".", ""]
    L += ["## Freezes (`examples`)", "",
          "Turns of 40 a freeze holds from the M4.5b freeze start (frozen-on side on Last Hit + Hold at its +1):", "",
          md_table(["level", "gap", "freezer rule"] + [f"held {h} HP" for h in (5, 8, 12, 14, 20)],
                   [[L_, g, rule] + [next(r["turns_held"] for r in e["freeze_hold"] if r["level"] == L_
                                          and r["gap"] == g and r["rule"] == rule and r["held_hp"] == h)
                                     for h in (5, 8, 12, 14, 20)]
                    for L_ in (4, 10, 16, 18) for g in ("equal", "large") for rule in ("controller", "thin")]), "",
          "One walk-up Shove against the freeze (share of chance draws after which the freeze still holds):", "",
          md_table(["held HP", "level", "C of the breaker", "kept passively", "guarded", "guard: A / B bars lost"],
                   [[r["held_hp"], r["L"], r["C_B"], f"{r['keep']['survives']:.2f}", f"{r['guard']['survives']:.2f}",
                     f"{r['guard']['A_bars']:.3f} / {r['guard']['B_bars']:.3f}"] for r in e["freeze_break"]]), ""]
    L += ["## One-turn kills, retaliation and low HP (`examples`)", "",
          md_table(["case", "taken (cHP)", "died"],
                   [[f"combination, own hit {r['own_hit']}", r["taken"], r["died"]] for r in e["kills"]["combination"]]
                   + [[f"gank + positional fire of a {r['enemy_front_hp']}-HP front", r["taken"], r["died"]]
                      for r in e["kills"]["positional"]]
                   + [[f"double hit, L{r['La']} S{r['Sa']} on L{r['Lb']} (ratio {r['ratio']:.2f})", r["taken"],
                       r["died"]] for r in e["kills"]["double_hit"]]), "",
          "Retaliation on a level-2 trader: " + "; ".join(
              f"from its {r['start']:+d}, {r['minion_cHP'][0]} cHP" for r in e["retaliation"]) + ".", "",
          md_table(["level, invested", "A plays", "B plays", "P(A dies this turn)", "of it, B's trade alone"],
                   [[f"{r['L']}, {r['S']}", r["a"], r["b"], f"{r['p_die']:.3f}", f"{r['p_die_trade_only']:.3f}"]
                    for r in e["low_hp"]]), ""]
    L += ["## Window strength (`examples`)", "",
          md_table(["state", "S_A / S_B", "Thin need A / B", "Slow Push need A / B"],
                   [[w["state"], f"{w['S'][0]:g} / {w['S'][1]:g}", f"{w['thin_need'][0]} / {w['thin_need'][1]}",
                     f"{w['slowpush_need'][0]} / {w['slowpush_need'][1]}"] for w in e["windows"]]), ""]
    o = e["one_turn"]
    L += ["## A constructed turn with farm, a plate and a kill (`examples`)", "",
          f"Before: `{o['lane_before']}`. After: `{o['lane_after']}`. Actions {o['actions'][0]} / {o['actions'][1]}.",
          "",
          md_table(["", "A", "B"], [["farm", f"{o['farm'][0]:.4f}", f"{o['farm'][1]:.4f}"],
                                    ["plate gold", f"{o['plate_gold'][0]:.4f}", f"{o['plate_gold'][1]:.4f}"],
                                    ["kills", f"{o['kill_gold'][0]:.4f}", f"{o['kill_gold'][1]:.4f}"],
                                    ["earned", f"{o['earned'][0]:.4f}", f"{o['earned'][1]:.4f}"],
                                    ["XP", o["xp_gain"][0], o["xp_gain"][1]],
                                    ["damage taken (cHP)", o["damage_taken"][0], o["damage_taken"][1]],
                                    ["died", o["died"][0], o["died"][1]]]), "",
          f"r_A = {o['reward']:+.4f} ({o['reward_units']:,} ledger units).", ""]
    tw = e["tower"]
    L += ["## Towers (`examples`)", "",
          "Minion hits of an undisturbed crash (units, by the tick of its first turn): " + "; ".join(
              f"{c_['hp']} HP: " + " or ".join(str(p["hits"]) for p in c_["by_parity"]) for c_ in tw["crash"]) + ".",
          "",
          md_table(["level", "invested", "AP*", "C", "Pressure (centi-units)", "Pressure turns to exhaust"],
                   [[p["L"], p["S"], f"{p['ap']:.3f}", p["C"], p["P"], p["turns"]] for p in tw["pressure"]]), ""]
    b = e["bounds"]
    L += ["## Evaluation bounds (`examples`)", "",
          md_table(["preset", "Omega", "terms a / b/(1-gamma) / kills / plates", "per-turn bound"]
                   + [f"gamma^{W} Omega" for W in (50, 150, 300, 460, 500, 1000)],
                   [[k, f"{v['omega']:.2f}", " / ".join(f"{x:.2f}" for x in v["omega_terms"].values()),
                     f"{v['per_turn_bound']:.2f}"] + [f"{v['bound'][str(W)]:.4g}" for W in (50, 150, 300, 460, 500, 1000)]
                    for k, v in b.items()]), ""]
    return L


def _actions(a: dict) -> List[str]:
    r = a["random_play"]
    moved = r["end_cell_depends_on_wave"]
    return [f"## Legality, the observation and the wave choice (`actions`: {r['games']} games of {r['turns']} turns of "
            "uniform random legal play)", "",
            f"Codes that can ever be legal (exhaustive): {a['exhaustive']['n_ever_legal']}, "
            f"{a['exhaustive']['ever_legal']}. Legal-set sizes in play: {r['sizes']}; mismatches against the rule: "
            f"{r['rule_mismatches']} of {r['seat_turns']}.", "",
            f"Observation: range {r['observation']['range']}, decode mismatches "
            f"{r['observation']['decode_mismatches']}, {r['observation']['at_W']} observations at W. Arrivals at W by "
            f"code: {r['W_arrivals_by_code']}. Absence set by a death: {r['death_absence']}.", "",
            md_table(["lane actions", "n", "end cell depends on the wave slot"],
                     [[k, v["n"], f"{100 * v['share']:.1f}%"] for k, v in moved.items()]), "",
            "Scripted Last Hit + Hold, median turn of levels 2 / 6 / 11 / 18: " + "; ".join(
                f"{k} " + " / ".join(f"{x:g}" for x in v.values()) for k, v in a["level_timing"]["median_turn"].items())
            + ".", ""]


def _selfplay(s: dict) -> List[str]:
    cp = s["checkpoints"]
    turns = list(cp)
    L = [f"## Self-play of `{POLICY}` (`selfplay`: {s['n_games']} games of {s['turns']} turns from s0, both seats)",
         "",
         md_table(["turn"] + turns, [["level"] + [f"{cp[n]['level']:.1f}" for n in turns],
                                     ["earned"] + [f"{cp[n]['earned']:.0f}" for n in turns],
                                     ["invested"] + [f"{cp[n]['invested']:.0f}" for n in turns],
                                     ["M(S)"] + [f"{cp[n]['M']:.2f}" for n in turns],
                                     ["tau dealt"] + [f"{cp[n]['tau']:,.0f}" for n in turns]]), "",
         f"Median turn of level 14: {s['L14']['median']:g}; level 18: {s['L18']['median']:g}; both champions at "
         f"level 18: {s['both_L18']['median']:g}. A side's plates fall at median turns "
         + " / ".join(f"{p['median']:g}" for p in s["plates"])
         + f"; both towers are exhausted at median turn {s['both_towers_exhausted']['median']:g}.", "",
         "Per side and 1,000 turns: " + ", ".join(f"{k} {v:.1f}" for k, v in s["per_1000"].items()) + ". Earned "
         "gold: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in s["earned_split"].items())
         + f". Pressure supplies {100 * s['pressure_share']:.1f}% of the tower damage dealt before a tower is "
           "exhausted.", ""]
    for w, label in (("lane", "the lane phase, turns 0-93"), ("all", "all turns")):
        ip = s["in_play"][w]
        L += [f"In {label}: farm " + ", ".join(f"{k} {v:.3f}" for k, v in ip["farm"].items()) + "; XP "
              + ", ".join(f"{k} {v:.3f}" for k, v in ip["xp"].items()) + "; per 1,000 turns "
              + ", ".join(f"{k} {v:.1f}" for k, v in ip["per_1000"].items()) + ".", ""]
    d = s["decisions"]
    L += ["Recalls and deaths: " + ", ".join(f"{k} {v:.1f}" for k, v in d["per_1000"].items())
          + f" per 1,000 turns; {100 * d['recall_interrupted_share']:.0f}% of Recall attempts interrupted; deaths "
          + ", ".join(f"{k} {100 * v:.0f}%" for k, v in d["death_share"].items()) + ". Wave choices: "
          + ", ".join(f"{k} {100 * v:.1f}%" for k, v in d["wave_share"].items()) + ". Freezes: "
          + ", ".join(f"{k} {v:.2f}" for k, v in d["freezes"].items() if not isinstance(v, dict)) + ".", ""]
    return L


def _death(d: dict) -> List[str]:
    rows = [[p, f"{v['turn']:.0f}", f"{v['level']:.1f}", f"{v['absence']:.1f}", fci(v["recall_cost"]),
             fci(v["death_cost"]), fci(v["death_minus_recall"])] for p, v in d["summary"].items()]
    return [f"## The cost of a death and of a Recall (`death`: {d['n_states']} states per phase, {d['n_roll']} paired "
            f"rollouts of {d['window']} turns, `{POLICY}` in both seats)", "",
            md_table(["phase", "mean turn", "mean level", "death absence", "Recall cost", "death cost",
                      "death minus Recall"], rows), "",
            f"Costs are discounted resources lost by the focal side (gamma {d['gamma']}), with 95% intervals over "
            f"states. The death cost includes the {d['kill_gold']:g} gold the opponent earns for the kill.", ""]


def write() -> str:
    L = ["# Reference generator results (`python -m lanegame.experiments.reference`)", "",
         "Engine-computed numbers that docs/m4/laning_game_mathematical_reference.md quotes, on "
         "`GameConfig.m45_calibrated()`.", ""]
    for name, fn in (("examples", _examples), ("actions", _actions), ("selfplay", _selfplay), ("death", _death)):
        data = load_json(name)
        L += fn(data) if data else [f"## `{name}`", "", "Not run yet.", ""]
    return write_md("report", "\n".join(L))


__all__ = ["write"]
