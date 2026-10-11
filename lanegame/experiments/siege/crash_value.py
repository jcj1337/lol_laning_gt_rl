"""Section 8 - the value of crashing, measured as a counterfactual at the pre-crash decision.

For pre-crash states in which A's packet of h HP (h = 4, 8, 12) stands one cell from B's tower while B's
next wave has just spawned, the Q-values of A's actions are compared from the same state and horizon:
Shove + Contest (crash now and follow), Shove + Safe (crash now, stay back), Slow Push + Contest / Safe
(hold at +1). Each Q is decomposed into immediate farm, plates, gank and continuation value.

Three opponents: B's equilibrium strategy (worst case), a passive B (Slow Push + Safe) and a predictable
stack builder; three models: the previous revision (choice), the siege model at 1 and at 2 HP/tick.
Also reported: the value of the post-crash state itself, and in it the marginal value of Apply Pressure
vs. Establish Vision vs. disengaging, and the value of the full Vision -> Shove -> Crash -> Pressure
sequence relative to immediate pressure, from a state after a first crash.

All probes of one model share one LaneGame (one transition cache), which bounds the memory by the full
game's cache and avoids recomputing transitions between probes.
"""

from __future__ import annotations

import time

import numpy as np

from ... import GameConfig, LaneGame
from ...baselines import passive_safe, stack_builder
from ...best_response import induct
from ...game import apply_choice
from ...policy import Policy
from ...solver import solve_equilibrium
from ...state import State
from .common import K, md_table, save

MODELS = {
    "previous revision (choice, tower 1)": lambda: GameConfig.v2(horizon=K),
    "siege, tower 1": lambda: GameConfig(horizon=K, tower_siege_damage=1),
    "siege, tower 2": lambda: GameConfig(horizon=K, tower_siege_damage=2),
}
SIZES = {4: "small (1 wave)", 8: "medium (2 waves)", 12: "large (3 waves)"}
REMAINING = (16, 24)


def pre_crash_state(h: int) -> State:
    """A's packet (h HP) at +1, B's fresh wave just spawned at its base (so A's Shove crashes next tick)."""
    return State(1, h, (), 0, -4, 4, (), 0, 0, 0)


def _choice_continuation(g, tr, V_next, pol_b, k0, responder_a: bool):
    """Continuation through the choice stage: A picks its best option (responder) against B's fixed chooser,
    or the choice game's value when B is the equilibrium (handled by the solver, not here)."""
    cfg = g.cfg
    if len(tr.options_a) == 1 and len(tr.options_b) == 1:
        return V_next[tr.pre_state] if V_next is not None else 0.0
    C = np.empty((len(tr.options_a), len(tr.options_b)))
    for ii, oa in enumerate(tr.options_a):
        for jj, ob in enumerate(tr.options_b):
            s2, r = apply_choice(tr.pre_state, oa, ob, cfg)
            C[ii, jj] = r + (V_next[s2] if V_next is not None else 0.0)
    yb = pol_b.choose(k0, tr.pre_state, 1, tr.options_b, tr.options_a) if len(tr.options_b) > 1 else np.ones(1)
    return float(np.max(C @ yb))


def q_rows_vs_eq(g: LaneGame, state: State, k0: int) -> dict:
    cfg = g.cfg
    sol = solve_equilibrium(g, root=state, k0=k0)
    la, lb = sol.legal[k0][state]
    y = sol.y[k0][state]
    rows = {}
    for i, a in enumerate(la):
        farm = gank = plate = crash = 0.0
        for j, b in enumerate(lb):
            tr = g.transition(state, k0, a, b)
            farm += y[j] * tr.farm_payoff
            gank += y[j] * tr.gank_payoff
            plate += y[j] * tr.plate_payoff
            crash += y[j] * tr.crash_hp[0]
        q = float(sol.Q[k0][state][i] @ y)
        rows[cfg.label(a)] = {"Q": q, "farm": farm, "plates": plate, "gank": gank, "continuation": q - farm - gank - plate,
                              "crash_hp": crash}
    return {"value": sol.value, "A": {cfg.label(a): round(float(p), 3) for a, p in zip(la, sol.x[k0][state]) if p > 1e-9},
            "B": {cfg.label(b): round(float(p), 3) for b, p in zip(lb, sol.y[k0][state]) if p > 1e-9}, "actions": rows,
            "nodes": sol.stats.states}


def q_rows_vs_policy(g: LaneGame, state: State, k0: int, pol_b: Policy) -> dict:
    """A best-responds to a fixed B from `state`; Q of each first action against B's fixed move."""
    cfg = g.cfg
    layers = g.reachable(root=state, k0=k0)
    res = induct(g, layers, None, pol_b)
    la, lb = g.legal(state, 0), g.legal(state, 1)
    pb = pol_b.move(k0, state, 1, lb)
    V_next = res.V[k0 + 1] if k0 + 1 < g.horizon else None
    rows = {}
    for a in la:
        q = farm = gank = plate = crash = 0.0
        for j, b in enumerate(lb):
            if pb[j] == 0:
                continue
            tr = g.transition(state, k0, a, b)
            cont = _choice_continuation(g, tr, V_next, pol_b, k0, True)
            q += pb[j] * (tr.payoff + cont)
            farm += pb[j] * tr.farm_payoff
            gank += pb[j] * tr.gank_payoff
            plate += pb[j] * tr.plate_payoff
            crash += pb[j] * tr.crash_hp[0]
        rows[cfg.label(a)] = {"Q": q, "farm": farm, "plates": plate, "gank": gank, "continuation": q - farm - gank - plate,
                              "crash_hp": crash}
    return {"value": res.value(g, state, k0), "actions": rows, "nodes": sum(len(l) for l in layers)}


def post_crash_values(g: LaneGame, h: int, k0: int) -> dict:
    """Value of the state right after A's crash of h HP (B's fresh wave walks up and waits), and A's options in it."""
    cfg = g.cfg
    plates = cfg.plate_grant[cfg.size_category(h)] if cfg.objectives else 0
    mass = max(0, h - cfg.tower_siege_damage)                     # the crash lands on tick 1; one tick of fire follows
    st = State(-4, 4, (), mass, -3, 4, (), 0, 0, 0, plates, 0)
    if mass == 0:
        return {"state": st.to_list(), "value": None}
    d = q_rows_vs_eq(g, st, k0 + 1)
    return {"state": st.to_list(), **d}


def sequence_vs_immediate(g: LaneGame, k0: int) -> dict:
    """From a state after a first 12-HP crash (mass 11, B wave waiting, 3 plates): the equilibrium values of
    committing to Establish Vision now vs. Apply Pressure now (Q against B's equilibrium), and the probability
    with which the equilibrium chooses each."""
    st = State(-4, 4, (), 11, -3, 4, (), 0, 0, 0, 3, 0)
    d = q_rows_vs_eq(g, st, k0)
    best = {}
    for kind, pred in (("pressure", lambda lab: "+P" in lab), ("vision", lambda lab: "+V" in lab), ("safe", lambda lab: "-S" in lab)):
        cand = {lab: r["Q"] for lab, r in d["actions"].items() if pred(lab)}
        best[kind] = max(cand.items(), key=lambda t: t[1]) if cand else None
    return {"value": d["value"], "A": d["A"], "best": best}


def run(horizon: int = K) -> dict:
    out = {}
    for mname, make in MODELS.items():
        cfg = make()
        g = LaneGame(cfg)
        out[mname] = {}
        t0 = time.perf_counter()
        for h in SIZES:
            for rem in REMAINING:
                k0 = horizon - rem
                st = pre_crash_state(h)
                vs_eq = q_rows_vs_eq(g, st, k0)
                vs_pas = q_rows_vs_policy(g, st, k0, passive_safe())
                vs_sb = q_rows_vs_policy(g, st, k0, stack_builder())
                post = post_crash_values(g, h, k0)
                out[mname][f"{h} HP, {rem} turns left"] = {"h": h, "remaining": rem, "vs_equilibrium": vs_eq,
                                                            "vs_passive_safe": vs_pas, "vs_stack_builder": vs_sb,
                                                            "post_crash": post}
                print(mname, h, rem, f"{time.perf_counter() - t0:.0f}s", flush=True)
        if cfg.objectives:
            out[mname]["sequence_vs_immediate"] = {str(rem): sequence_vs_immediate(g, horizon - rem) for rem in (12, 16, 24)}
        out[mname]["seconds"] = time.perf_counter() - t0
        out[mname]["cached_transitions"] = len(g._cache)
        del g

    def best_of(rows, pred):
        cand = {lab: r for lab, r in rows.items() if pred(lab)}
        return max(cand.items(), key=lambda t: t[1]["Q"]) if cand else None

    md = [f"# The value of crashing: counterfactuals at the pre-crash decision (K = {horizon})", "",
          "Pre-crash state: A's packet of h HP one cell from B's tower, B's next wave just spawned at its base. "
          "Shove crashes it on the next tick; Slow Push holds it at +1 (a stacked packet creeps only on creep turns). "
          "Q-values are from the same state and horizon; 'crash gain' = Q(best Shove action) - Q(best non-crashing action), "
          "i.e. how much A gains or loses by producing the crash now instead of its best alternative.", ""]
    for mname, d in out.items():
        md += [f"## {mname}", ""]
        rows = []
        for key, r in d.items():
            if key in ("sequence_vs_immediate", "seconds", "cached_transitions"):
                continue
            def gain(acts):
                sh = best_of(acts, lambda lab: lab.startswith("SH"))
                hold = best_of(acts, lambda lab: lab.startswith("SP") or lab.startswith("F"))
                return sh[1]["Q"] - hold[1]["Q"], sh[0], hold[0]
            ge, gp, gs = gain(r["vs_equilibrium"]["actions"]), gain(r["vs_passive_safe"]["actions"]), gain(r["vs_stack_builder"]["actions"])
            rows.append((SIZES[r["h"]], r["remaining"], round(r["vs_equilibrium"]["value"], 2),
                         ", ".join(f"{k} {v}" for k, v in r["vs_equilibrium"]["A"].items()),
                         f"{ge[0]:+.2f} ({ge[1]} vs {ge[2]})", f"{gp[0]:+.2f} ({gp[1]} vs {gp[2]})",
                         f"{gs[0]:+.2f} ({gs[1]} vs {gs[2]})",
                         round(r["post_crash"]["value"], 2) if r["post_crash"].get("value") is not None else "-"))
        md += [md_table(["crash size", "turns left", "V* pre-crash", "A's equilibrium action", "crash gain vs equilibrium B",
                         "crash gain vs passive B", "crash gain vs stack_builder B", "V(post-crash state)"], rows), ""]
        for key, r in d.items():
            if key in ("sequence_vs_immediate", "seconds", "cached_transitions") or r["remaining"] != 16:
                continue
            md += [f"**{SIZES[r['h']]}, 16 turns left: decomposition against B's equilibrium strategy** "
                   f"(B plays {r['vs_equilibrium']['B']})", "",
                   md_table(["A action", "Q", "farm", "plates", "gank", "continuation", "crash HP now"],
                            [(lab, round(x["Q"], 3), round(x["farm"], 3), round(x["plates"], 3), round(x["gank"], 3),
                              round(x["continuation"], 3), round(x["crash_hp"], 1)) for lab, x in r["vs_equilibrium"]["actions"].items()]), "",
                   "**... against passive_safe B**", "",
                   md_table(["A action", "Q", "farm", "plates", "gank", "continuation", "crash HP now"],
                            [(lab, round(x["Q"], 3), round(x["farm"], 3), round(x["plates"], 3), round(x["gank"], 3),
                              round(x["continuation"], 3), round(x["crash_hp"], 1)) for lab, x in r["vs_passive_safe"]["actions"].items()]), ""]
            post = r["post_crash"]
            if post.get("value") is not None:
                md += [f"**... the post-crash state** (A's mass {post['state'][3]} under B's tower, B's wave waiting; "
                       f"V = {post['value']:.3f}; A plays {post['A']}; B plays {post['B']})", "",
                       md_table(["A action", "Q", "farm", "plates", "gank", "continuation"],
                                [(lab, round(x["Q"], 3), round(x["farm"], 3), round(x["plates"], 3), round(x["gank"], 3),
                                  round(x["continuation"], 3)) for lab, x in post["actions"].items()]), ""]
        if "sequence_vs_immediate" in d:
            md += ["**Vision now vs. pressure now, after a first 12-HP crash (mass 11, B wave waiting, 3 plates)**", "",
                   md_table(["turns left", "V*", "A plays", "best Pressure Q", "best Vision Q", "best Safe Q", "Vision - Pressure"],
                            [(rem, round(s["value"], 3), ", ".join(f"{k} {v}" for k, v in s["A"].items()),
                              round(s["best"]["pressure"][1], 3), round(s["best"]["vision"][1], 3), round(s["best"]["safe"][1], 3),
                              round(s["best"]["vision"][1] - s["best"]["pressure"][1], 3))
                             for rem, s in d["sequence_vs_immediate"].items()]), ""]
        md += [f"({d['seconds']:.0f} s, {d['cached_transitions']} cached transitions)", ""]
    save("crash_value", {"horizon": horizon, "results": out}, "\n".join(md))
    return out


if __name__ == "__main__":
    run()
