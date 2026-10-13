"""Part 2 and Part 3 tables of the M4.3d study (results/m43d/tower_farm.md, pressure.md) from the crash grids."""

from __future__ import annotations

from . import common as c
from .analysis import Grid, best_replies, farm_profile, fmt, md_table, pct, risk_profile

OTS = ("ot60", "econ", "ot85", "ot90")
OT_LABEL = {"ot60": "0.6", "econ": "0.8", "ot85": "0.85", "ot90": "0.9"}
GAP_LABEL = {"equal": "equal", "large": "A x1.76", "b_large": "A weak (B x1.76)"}
A_UNCONTESTED = ("sh_leave", "sh_recall")
A_FORWARD = ("sh_stay", "sh_press", "sh_vis")
A_CONTEST = ("sh_harass", "sh_hunt", "sh_dive", "sh_dive_edge", "sh_vis_dive", "sh_bully")
A_ALL = ("lh",) + A_UNCONTESTED + A_FORWARD + A_CONTEST
B_ALL = ("lh", "thin", "safe", "respect", "punish", "fight", "shove")


def _profile_row(g: Grid, v, st, gp, a, b, w) -> list:
    p = farm_profile(g, v, st, gp, a, b, "b", w)
    return [fmt(p["ideal"]), fmt(p["eff"], 3), pct(p["tower"]), pct(p["position"]), pct(p["posture"]),
            pct(p["trade"]), pct(p["absence"]), fmt(p["xp_missed"], 1), fmt(p["hp_lost"]), fmt(p["died"]),
            fmt(p["recalled"])]


PROFILE_HDR = ["B ideal farm", "B efficiency", "tower cap", "position", "posture", "trade", "absence", "B XP missed",
               "B HP lost (bars)", "B deaths", "B Recalls"]


def tower_farm_report(g: Grid) -> str:
    L = ["# M4.3d Part 2: farming a crashed wave under the own tower", "",
         "Crash grid `results/m43d/grid_tower_farm.json.gz`: 64 paired games of 100 turns per cell (SeedSequence "
         "79,000), every configuration is `m4_four_actions_econ()` with only `collect_contest_own_tower` changed "
         "(0.6 / 0.8 = econ / 0.85 / 0.9). Starts: S10 / S4 = one turn after A's crash (A's 14-HP mass under B's "
         "tower, A in B's tower cell, B in its tower cell), N10 / N4 = the neutral lane. B efficiency = B's collected "
         "farm / B's ideal farm (every enemy minion HP that died on B's ledger at the lane rate 0.9); the loss shares "
         "tower cap (unconditional) / position / posture / trade / absence complete it to 100%.", ""]
    # 2.1 the neutral reference
    L += ["## 2.1 Reference: the neutral lane (both Last Hit, 40 turns)", ""]
    rows = []
    for v in OTS:
        for st in ("N4", "N10"):
            rows.append([OT_LABEL[v], st] + _profile_row(g, v, st, "equal", "lh", "lh", "40"))
    L += [md_table(["own tower", "start"] + PROFILE_HDR, rows), ""]
    # 2.2 uncontested crash
    for w, title in (("12", "the crash window (12 turns)"), ("40", "40 turns")):
        L += [f"## 2.2 Uncontested: A shoves and leaves or recalls, B last-hits ({title})", ""]
        rows = []
        for st in ("S10", "S4"):
            for a in A_UNCONTESTED + ("sh_stay",):
                for v in OTS:
                    rows.append([st, a, OT_LABEL[v]] + _profile_row(g, v, st, "equal", a, "lh", w) +
                                [fmt(g.m(v, st, "equal", a, "lh", "earned_diff", w))])
        L += [md_table(["start", "A", "own tower"] + PROFILE_HDR + ["A − B gold"], rows), ""]
    # 2.3 contested
    L += ["## 2.3 Contested: A stays forward, harasses or dives; B's responses (S10, 40 turns)", ""]
    for gp in ("equal", "large"):
        L += [f"### {GAP_LABEL[gp]}", ""]
        rows = []
        for a in ("sh_press",) + A_CONTEST:
            for b in ("lh", "safe", "respect", "punish", "fight"):
                for v in ("econ", "ot90"):
                    p = farm_profile(g, v, "S10", gp, a, b, "b", "40")
                    q = risk_profile(g, v, "S10", gp, a, b, "a", "40")
                    rows.append([a, b, OT_LABEL[v], fmt(p["eff"], 3), pct(p["tower"]), pct(p["posture"]),
                                 pct(p["trade"]), pct(p["absence"]), fmt(p["hp_lost"]), fmt(p["died"]),
                                 fmt(q["hp_lost"]), fmt(q["died"]), fmt(g.m(v, "S10", gp, a, b, "earned_diff", "40")),
                                 fmt(g.m(v, "S10", gp, a, b, "G"))])
        L += [md_table(["A", "B", "own tower", "B eff", "tower cap", "posture", "trade", "absence", "B HP lost",
                        "B deaths", "A HP lost", "A deaths", "A − B gold (40)", "J (100)"], rows), ""]
    # 2.4 the gift test
    L += ["## 2.4 Is a crash a gift? B's farm when crashed on, against the neutral lane (N starts, B last-hits)", "",
          "B farm and A − B gold over 40 turns; J = discounted A − B over 100 turns; Δ = the same minus A last-hitting "
          "(no deliberate crash) on the same seeds.", ""]
    rows = []
    for st in ("N10", "N4"):
        for v in OTS:
            fb0 = g.m(v, st, "equal", "lh", "lh", "farm_b")
            j0 = g.m(v, st, "equal", "lh", "lh", "G")
            for a in ("sh_leave", "sh_recall", "sh_stay", "sh_press", "sh_vis"):
                p = farm_profile(g, v, st, "equal", a, "lh", "b", "40")
                dj, cj = g.paired((v, st, "equal", a, "lh", None), (v, st, "equal", "lh", "lh", None), "G")
                rows.append([st, OT_LABEL[v], a, fmt(p["eff"], 3), fmt(p["farm"]), fmt(p["farm"] - fb0),
                             fmt(g.m(v, st, "equal", a, "lh", "plate_a")),
                             fmt(g.m(v, st, "equal", a, "lh", "earned_diff")), f"{dj:+.2f} ± {cj:.2f}"])
    L += [md_table(["start", "own tower", "A", "B eff", "B farm", "Δ B farm", "A plates", "A − B gold (40)",
                    "Δ J vs A last-hits"], rows), ""]
    # 2.5 displacement
    L += ["## 2.5 Permanent or displaced? B's efficiency by window, and the enemy minion HP still alive at the end", ""]
    rows = []
    for v in OTS:
        for a in ("sh_leave", "sh_press"):
            e = [farm_profile(g, v, "S10", "equal", a, "lh", "b", w)["eff"] for w in ("12", "40", "all")]
            pend = g.m(v, "S10", "equal", a, "lh", "xp_pending_change_b", "all")
            rows.append([OT_LABEL[v], a] + [fmt(x, 3) for x in e] + [fmt(pend, 1)])
    L += [md_table(["own tower", "A", "B eff 12", "B eff 40", "B eff 100", "Δ enemy minion HP alive (100)"], rows), ""]
    # 2.6 best replies
    L += ["## 2.6 B's best reply (lowest J to A) to each attacker, and A's value against it", ""]
    for st in ("S10", "S4", "N10"):
        rows = []
        for gp in ("equal", "large", "b_large"):
            for v in OTS:
                br = best_replies(g, v, st, gp, A_ALL, B_ALL)
                rows.append([GAP_LABEL[gp], OT_LABEL[v]] +
                            [f"{br[a]} ({g.m(v, st, gp, a, br[a], 'G'):+.1f})" for a in A_ALL])
        L += [f"### {st}", "", md_table(["gap", "own tower"] + list(A_ALL), rows), ""]
    return "\n".join(L)


def pressure_report(g: Grid) -> str:
    L = ["# M4.3d Part 3: the risks of applying pressure (econ = m4_four_actions_econ())", "",
         "Crash grid cells on econ (64 paired games, 100 turns, SeedSequence 79,000) and the same games with A made "
         "immune to ganks (`a_no_gank`) or to tower shots (`a_no_tower`): the chance draws are identical, the masked "
         "uniforms are set to 'no event'. Bars = HP in units of the champion's max HP, before durability "
         "mitigation.", ""]
    for st in ("S10", "S4", "N10"):
        for gp in ("equal", "large", "b_large"):
            L += [f"## {st}, {GAP_LABEL[gp]}: where A stands and what it costs (B last-hits; 40 turns)", ""]
            rows = []
            for a in A_ALL:
                q = risk_profile(g, "econ", st, gp, a, "lh", "a", "40")
                rows.append([a, pct(q["at+2"]), pct(q["at+1"]), pct(q["at+0"]), pct(q["at-1"]) + "/" + pct(q["at-2"]),
                             fmt(q["exp_gank"]), fmt(q["gank_dmg"]), fmt(q["tower_dmg"]), fmt(q["trade_dmg"]),
                             fmt(q["died"]), fmt(q["recalled"]), fmt(q["away"], 1), fmt(q["plate"]), fmt(q["farm"]),
                             fmt(g.m("econ", st, gp, a, "lh", "earned_diff")), fmt(g.m("econ", st, gp, a, "lh", "G"))])
            L += [md_table(["A", "at +2", "at +1", "at 0", "at -1/-2", "E[gank] bars", "gank bars", "tower bars",
                            "trade bars", "deaths", "Recalls", "away", "plates", "farm", "A − B (40)", "J"], rows), ""]
    L += ["## The economic cost of each hazard to A (paired: J with the hazard minus J with A immune; resources over "
          "100 turns, discounted)", ""]
    for b in ("lh", "punish"):
        rows = []
        for st in ("S10", "S4", "N10"):
            for gp in ("equal", "large", "b_large"):
                for a in A_ALL:
                    if not g.has("econ", st, gp, a, b, "a_no_gank"):
                        continue
                    base = ("econ", st, gp, a, b, None)
                    dg, cg = g.paired(base, ("econ", st, gp, a, b, "a_no_gank"))
                    dt, ct = g.paired(base, ("econ", st, gp, a, b, "a_no_tower"))
                    rows.append([st, GAP_LABEL[gp], a, fmt(g.m("econ", st, gp, a, b, "G")), f"{dg:+.2f} ± {cg:.2f}",
                                 f"{dt:+.2f} ± {ct:.2f}"])
        hdr = ["start", "gap", "A", "J", "cost of ganks", "cost of tower shots"]
        L += [f"### B = {b}", "", md_table(hdr, rows), ""]
    return "\n".join(L)


SENS = ("econ", "tw_aggro", "tw2", "tw2_aggro", "gk_pos16", "gk_dmg08")
SENS_A = ("lh", "sh_press", "sh_vis", "sh_dive", "sh_dive_edge", "sh_vis_dive", "sh_bully")


def sensitivity_report(g: Grid) -> str:
    from .strategic import grid_games
    from .strategies import ATTACKERS, DEFENDERS
    L = ["# M4.3d Part 4: sensitivity of the risk rules (one parameter group per variant, econ otherwise)", "",
         "Same seeds and cells as Parts 2-3 (64 paired games, 100 turns). " +
         "; ".join(f"`{v}`: {c.VARIANTS[v][0]}" for v in SENS[1:]) + ".", ""]
    for st in ("S10", "N10", "S4"):
        for gp in ("equal", "large", "b_large"):
            L += [f"## {st}, {GAP_LABEL[gp]}: A's value J (B last-hits) / A deaths per 40 turns", ""]
            rows = []
            for a in SENS_A:
                row = [a]
                for v in SENS:
                    if g.has(v, st, gp, a, "lh"):
                        row.append(f"{g.m(v, st, gp, a, 'lh', 'G'):+.2f} / {g.m(v, st, gp, a, 'lh', 'died_a'):.2f}")
                    else:
                        row.append("–")
                rows.append(row)
            L += [md_table(["A"] + list(SENS), rows), ""]
    L += ["## The neutral lane and the tower farmer under each variant (N10 / S10, equal, both Last Hit, 40 turns)", ""]
    rows = []
    for v in SENS:
        rows.append([v] + [fmt(g.m(v, st, "equal", "lh", "lh", k)) for st in ("N10", "S10")
                           for k in ("died_b", "hp_lost_b", "away_b")])
    L += [md_table(["variant", "N10 B deaths", "N10 B HP lost", "N10 B away", "S10 B deaths", "S10 B HP lost",
                    "S10 B away"], rows), ""]
    L += ["## Restricted games of the crash scenarios (A = 12 attackers, B = 7 defenders; J)", ""]
    rows = []
    for st in ("S10", "N10", "S4"):
        for gp in ("equal", "large", "b_large"):
            for v in SENS:
                s = grid_games(g, v, st, gp, tuple(ATTACKERS), tuple(DEFENDERS))
                rows.append([st, GAP_LABEL[gp], v, f"{s['value']:+.2f}",
                             ", ".join(f"{k} {p:.2f}" for k, p in s["x"].items()),
                             ", ".join(f"{k} {p:.2f}" for k, p in s["y"].items())])
    L += [md_table(["start", "gap", "variant", "value", "A's equilibrium", "B's equilibrium"], rows), ""]
    return "\n".join(L)


def crash_games_report(g: Grid) -> str:
    from .strategic import grid_games
    from .strategies import ATTACKERS, DEFENDERS
    L = ["# M4.3d Part 5a: restricted games of the crash scenarios (from the tower_farm grid; J over 100 turns)", "",
         "A = the 12 attacker strategies, B = the 7 defender strategies (strategies.py); 64 paired games per cell.", ""]
    rows = []
    for st in ("S10", "S4", "N10", "N4"):
        for gp in ("equal", "large", "b_large"):
            for v in OTS:
                for obj in ("G", "G40"):
                    s = grid_games(g, v, st, gp, tuple(ATTACKERS), tuple(DEFENDERS), obj)
                    rows.append([st, GAP_LABEL[gp], OT_LABEL[v], "J" if obj == "G" else "G40", f"{s['value']:+.2f}",
                                 ", ".join(f"{k} {p:.2f}" for k, p in s["x"].items()),
                                 ", ".join(f"{k} {p:.2f}" for k, p in s["y"].items())])
    L += [md_table(["start", "gap", "own tower", "objective", "value", "A's equilibrium", "B's equilibrium"], rows)]
    return "\n".join(L)


def validation_report(main: Grid, val: Grid) -> str:
    """The decisive quantities of Parts 2-5 on the main seeds (64 games, root 79,000) and on the independent validation
    seeds (128 games, root 79,500). The candidate's main-run cells are the identical configuration `tw_aggro`."""
    from .strategic import grid_games
    from .strategies import ATTACKERS, DEFENDERS
    MAIN = {"econ": "econ", "ot90": "ot90", "econ_aggro": "tw_aggro"}

    def both(fn):
        return [fn(main, MAIN[v]) for v in ("econ", "ot90", "econ_aggro")], \
            [fn(val, v) for v in ("econ", "ot90", "econ_aggro")]
    rows = []

    def add(label, fn, nd=2):
        m, x = both(fn)
        rows.append([label] + [f"{a:+.{nd}f} / {b:+.{nd}f}" for a, b in zip(m, x)])
    add("B efficiency, neutral lane (N10, both Last Hit, 40 turns)",
        lambda g, v: farm_profile(g, v, "N10", "equal", "lh", "lh", "b", "40")["eff"], 3)
    add("B efficiency, uncontested crash (S10, shove and leave, 40 turns)",
        lambda g, v: farm_profile(g, v, "S10", "equal", "sh_leave", "lh", "b", "40")["eff"], 3)
    add("Δ B farm when crashed on (N10, Shove + Pressure minus Last Hit, 40 turns)",
        lambda g, v: g.m(v, "N10", "equal", "sh_press", "lh", "farm_b") - g.m(v, "N10", "equal", "lh", "lh", "farm_b"))
    for st_, gp, a in (("N10", "equal", "sh_press"), ("S10", "equal", "sh_press"), ("S10", "equal", "sh_vis"),
                       ("S10", "equal", "sh_dive"), ("S10", "large", "sh_press"), ("S10", "large", "sh_vis"),
                       ("S10", "large", "sh_dive_edge"), ("S10", "large", "sh_vis_dive")):
        add(f"J of {a} against Last Hit ({st_}, {GAP_LABEL[gp]})",
            lambda g, v, s=st_, p=gp, x=a: g.m(v, s, p, x, "lh", "G"))
    for st_ in ("S10", "N10"):
        for gp in ("equal", "large", "b_large"):
            add(f"value of the crash game ({st_}, {GAP_LABEL[gp]})",
                lambda g, v, s=st_, p=gp: grid_games(g, v, s, p, tuple(ATTACKERS), tuple(DEFENDERS))["value"])
    L = ["# M4.3d validation: main seeds (64 games, SeedSequence 79,000) / independent seeds (128 games, 79,500)", "",
         "Columns: econ (own tower 0.8) / own tower 0.9 / the candidate m4_four_actions_econ_aggro() (main run: the "
         "identical configuration `tw_aggro`). Each cell: main / validation.", "",
         md_table(["quantity", "econ", "0.9", "candidate"], rows), "",
         "## Crash-game equilibria (validation seeds)", ""]
    eq = []
    for st_ in ("S10", "N10"):
        for gp in ("equal", "large", "b_large"):
            for v in ("econ", "ot90", "econ_aggro"):
                s = grid_games(val, v, st_, gp, tuple(ATTACKERS), tuple(DEFENDERS))
                sm = grid_games(main, MAIN[v], st_, gp, tuple(ATTACKERS), tuple(DEFENDERS))
                eq.append([st_, GAP_LABEL[gp], v, ", ".join(f"{k} {p:.2f}" for k, p in sm["x"].items()) + " / " +
                           ", ".join(f"{k} {p:.2f}" for k, p in sm["y"].items()),
                           ", ".join(f"{k} {p:.2f}" for k, p in s["x"].items()) + " / " +
                           ", ".join(f"{k} {p:.2f}" for k, p in s["y"].items())])
    L += [md_table(["start", "gap", "configuration", "main: A / B", "validation: A / B"], eq)]
    return "\n".join(L)


def run() -> None:
    g = Grid("tower_farm")
    c.write_md("tower_farm", tower_farm_report(g))
    c.write_md("crash_games", crash_games_report(g))
    for tags, name, fn in ((("tower_farm", "pressure"), "pressure", pressure_report),
                           (("tower_farm", "sensitivity"), "sensitivity", sensitivity_report)):
        try:
            c.write_md(name, fn(Grid(*tags)))
        except FileNotFoundError as e:
            print(e)
    try:
        c.write_md("validation", validation_report(Grid("tower_farm", "sensitivity"), Grid("validation")))
    except FileNotFoundError as e:
        print(e)


__all__ = ["run", "tower_farm_report", "pressure_report"]
