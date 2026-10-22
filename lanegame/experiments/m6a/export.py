"""The prototypes' data: verified recordings as frame files (prototypes/m6a/data/<name>.json).

Each file holds one game in the demo's frame contract (lanegame/demo/frames.py), produced by re-simulating its codes
in a fresh environment with every state digest checked: the meta block (provenance, seats, seeds, marks), positions
0..n, turns 1..n, the opponent-modelling seats' belief posteriors per position, seat A's action menu per position (for
the prototypes' staged Play mode) and an index of turns by situation.

The menus are seat A's real choices at each recorded position, with the engine's reasons for unavailable ones. The
prototypes stage Play against the recording: whatever a visitor selects, the recorded action resolves, and the
prototype says so. Only the backend (M6b) resolves arbitrary choices.

The situation index is a mechanical scan of each turn's outcomes (`index_situations`); it selects turns to jump to,
not games, and changes nothing in the recordings.
"""

from __future__ import annotations

import os
from typing import Dict, List

from . import common as Cm

SITUATIONS = {
    "quiet": "a run of at least 6 turns without a major outcome",
    "stacked_crash": "a wave larger than one spawn (over 8 HP) crashes into a tower",
    "trade_hit": "a trade that hits",
    "trade_miss": "a trade that reaches its target and misses",
    "trade_no_contact": "a trade that cannot reach its target",
    "recall": "a completed Recall",
    "recall_interrupted": "an interrupted Recall",
    "death": "a champion dies",
    "forced_absent_A": "seat A's only legal action is Absent",
    "ward": "a ward is placed",
    "plate": "a tower plate breaks",
}


def menu_compact(m: dict) -> dict:
    """The per-position part of a menu (the groups' names and help texts are in the file's `menu_static`)."""
    out = {"legal": m["legal"], "forced": m["forced"], "away": m["away"]}
    if m["forced"] is None:
        p = next(o for o in m["objectives"] if o["id"] == 1)
        out["reasons"] = {"pressure": p.get("reason"), "recall": m["singles"][0].get("reason"),
                          "ward": m["singles"][1].get("reason")}
    else:
        out["reasons"] = {"absent": m["waves"][0]["reason"]}
    return out


def menu_static(m: dict) -> dict:
    return {"waves": [{k: w[k] for k in ("id", "name", "help")} for w in m["waves"]],
            "behaviours": [{k: b[k] for k in ("id", "name", "help")} for b in m["behaviours"]],
            "objectives": [{k: o[k] for k in o if k not in ("legal", "reason")} for o in m["objectives"]],
            "singles": [{k: x[k] for k in ("code", "name", "help")} for x in m["singles"]],
            "lane_codes": m["lane_codes"]}


def index_situations(turns: List[dict], menus: List[dict]) -> Dict[str, List[int]]:
    idx: Dict[str, List[int]] = {k: [] for k in SITUATIONS}
    run = 0
    for tr in turns:
        t = tr["t"]
        kinds = [(e["kind"], e.get("data", {})) for e in tr["outcomes"]]
        major = any(e["major"] for e in tr["outcomes"])
        run = 0 if major else run + 1
        if run == 6:
            idx["quiet"].append(t - 5)
        for k, d in kinds:
            if k == "crash" and d.get("hp", 0) > 8:
                idx["stacked_crash"].append(t)
            if k == "trade":
                idx[f"trade_{d['result']}"].append(t)
            if k in ("recall", "recall_interrupted", "death", "ward", "plate"):
                idx[k].append(t)
    for p, m in enumerate(menus):
        if m["forced"] is not None:
            idx["forced_absent_A"].append(p + 1)              # the turn decided at position p
    return {k: sorted(set(v)) for k, v in idx.items()}


def game_file(s, name: str, label: str) -> dict:
    """A stopped GameSession as a prototype data file."""
    from ...demo import actions as ACT
    from ...demo import record as R
    from ...env import LaneEnv
    rec = s.recording()
    secret = s._secret_record()
    positions, turns = R.replay_frames(s.cfg, secret["seeds"]["chance"], rec["turns"])
    env = LaneEnv(s.cfg)
    env.reset(seed=secret["seeds"]["chance"])
    menus, first = [], None
    for i, (a, b, _) in enumerate(rec["turns"]):
        m = ACT.menu(env.state, env.champ, 0, s.cfg)
        first = first or (m if m["forced"] is None else None)
        menus.append(menu_compact(m))
        env.step(a, b)
    meta = {k: rec[k] for k in ("provenance", "config", "seeds", "policy_rule", "seats", "belief_mode", "turn_cap",
                                "status", "lead_A", "marks")}
    return {"schema": "lanegame.demo_prototype_game", "version": 1, "name": name, "label": label, "meta": meta,
            "lane_exit_position": 94, "read_position": 114, "positions": positions, "turns": turns,
            "beliefs": rec.get("beliefs"), "menu_static": menu_static(first), "menus_A": menus,
            "situations": index_situations(turns, menus), "situation_help": SITUATIONS}


def run() -> dict:
    from ...demo import session as S
    reg = Cm.registry()
    out = {}
    labels = {"h2h_om_s0_vs_sp_s0": "Pure self-play (A) against opponent modelling (B): the seed-0 head to head",
              "om_s0_vs_g_shove": "The scripted g_shove (A) against opponent modelling (B)",
              "sp_s0_vs_g_shove": "Pure self-play (A) against the scripted g_shove (B)"}
    for name in Cm.RESEARCH_GAMES:
        s, cmp = S.from_research(reg, os.path.join(Cm.TRACES, f"{name}.json"))
        d = game_file(s, name, labels[name])
        d["verification"] = cmp
        Cm.save_json(name, d, Cm.PROTO_DATA, indent=None)
        out[name] = {"turns": len(d["turns"]), "verification": cmp, "situations": {k: len(v) for k, v in
                                                                                     d["situations"].items()}}
    sim = S.simulate(reg, S.Seat.model("om:0"), S.Seat.model("sp:0"), chance_seed=Cm.SIM_ROOT,
                     policy_seed=Cm.SIM_ROOT + 500)
    d = game_file(sim, "sim_om_vs_sp", "A new simulation: opponent modelling (A) against pure self-play (B)")
    d["verification"] = S.replay(reg, sim.recording())
    Cm.save_json("sim_om_vs_sp", d, Cm.PROTO_DATA, indent=None)
    out["sim_om_vs_sp"] = {"turns": len(d["turns"]), "verification": d["verification"],
                           "situations": {k: len(v) for k, v in d["situations"].items()}}
    index = {"games": [{"name": n, "label": (labels.get(n) or "A new simulation: opponent modelling (A) against "
                                             "pure self-play (B)"), "kind": "new_simulation" if n.startswith("sim")
                        else "research_record"} for n in out]}
    Cm.save_json("index", index, Cm.PROTO_DATA)
    return out


__all__ = ["run", "game_file", "index_situations", "SITUATIONS"]
