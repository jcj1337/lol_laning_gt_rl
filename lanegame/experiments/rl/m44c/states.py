"""Starting states of the M4.4c experiments (Part 1.3: natural versus initialized, never pooled).

NATURAL states come from recorded self-play games of the evaluated final from s0 (`m44.records.play_record` with
snapshots), selected by a predicate on the DECISION STATE ONLY (the start of the turn and the policy's distribution
there; never on what happened afterwards): within each game the matching (turn, seat) pairs are shuffled with a seeded
generator and at most `per_game` are taken, games are added until `n_states` states are found. Nothing is post-selected
on outcomes.

INITIALIZED states are constructed diagnostic starts (the M4.3c harness scenarios with a power gap, a level or HP
changed); they are outside the training distribution and are always labelled.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ....diagnostics.branch import Snapshot
from ..m44.records import play_record

MIN_TURN = 8
ROW_FIELDS = ("present", "en_present", "pos", "epos", "dist", "hp_frac", "ehp_frac", "level", "elevel", "invested",
              "einvested", "gold", "power", "edge", "own_fx", "own_fh", "en_fx", "en_fh", "contact", "wave_pos", "s_own",
              "s_en", "wave_adv", "own_siege", "en_siege", "held", "held_on", "vision", "evision", "plates_left",
              "own_plates_left", "recall_legal", "obj_legal", "pw0", "pw1", "pw2", "pw3", "pb0", "pb1", "pb2", "pb3",
              "pb4", "po0", "po1", "po2", "prec", "entropy")
PREV_FIELDS = ("beh", "recall", "recalled", "held", "held_on", "wave", "obj", "hp_frac")


def _row(d, t: int, s: int) -> dict:
    out = {f: float(d[f][t, s]) for f in ROW_FIELDS}
    o = 1 - s
    for f in ("pb0", "pb1", "pb2", "pb3", "pb4", "prec", "pw0", "pw1", "pw2", "pw3", "po1", "po2"):
        out["opp_" + f] = float(d[f][t, o])                 # the opponent's distribution at the same decision state
    if t > 0:
        for f in PREV_FIELDS:
            out["prev_" + f] = float(d[f][t - 1, s])
        out["prev_opp_beh"] = float(d["beh"][t - 1, o])
    run = 0                                                  # consecutive freeze-state turns up to and including t
    while t - run >= 0 and d["held"][t - run, s] > 0:
        run += 1
    out["held_run"] = float(run)
    return out


def collect(cfg, agent, preds: Dict[str, Callable], n_states: Dict[str, int], seed: int, game_turns: int = 500,
            max_games: int = 80, per_game: int = 3, min_turn: int = MIN_TURN, start=None, start_label: str = "s0",
            describe: Optional[Callable] = None) -> Dict[str, List[dict]]:
    """Natural states of `agent` in self-play (module docstring): `preds[name](d, t, s) -> bool` on the record arrays,
    up to `n_states[name]` per situation. `describe(snapshot, seat, cfg) -> dict` adds situation covariates."""
    rng = np.random.default_rng(seed)
    found: Dict[str, List[dict]] = {k: [] for k in preds}
    games_used = 0
    for g, ss in enumerate(np.random.SeedSequence(seed).spawn(max_games)):
        if all(len(found[k]) >= n_states[k] for k in preds):
            break
        games_used += 1
        rec = play_record(cfg, (agent, agent), np.random.SeedSequence(ss.entropy, spawn_key=ss.spawn_key + (0,)),
                          game_turns, start, start_label, keep_snapshots=True)
        d = rec.data
        for key, pred in preds.items():
            if len(found[key]) >= n_states[key]:
                continue
            cand = [(t, s) for s in (0, 1) for t in range(min_turn, rec.turns)
                    if d["present"][t, s] > 0 and pred(d, t, s)]
            rng.shuffle(cand)
            for t, s in cand[:per_game]:
                st = {"snapshot": rec.snapshots[t], "seat": int(s), "turn": int(t), "game": g, "row": _row(d, t, s),
                      "natural": start is None, "start": start_label}
                if describe is not None:
                    st["desc"] = describe(rec.snapshots[t], int(s), cfg)
                found[key].append(st)
                if len(found[key]) >= n_states[key]:
                    break
    for key in found:
        for st in found[key]:
            st["games_scanned"] = games_used
    return found


def initialized(cfg, scenario: str, gap: str = "equal", level: Optional[int] = None, hp: Optional[Sequence] = None,
                pos: Optional[Sequence] = None, k_offset: int = 0) -> Snapshot:
    """An INITIALIZED snapshot: the M4.3c harness start `scenario` (neutral, pushed, siege, freeze) with the power
    gap (A's lead: equal, moderate x1.24, large x1.76, levels, large_levels x2.04; b_large: B x1.76), both champions at
    `level`, and optionally the HP fractions (A, B) and cells (A, B, own coordinates) replaced."""
    from ...m43c.harness import start as scenario_start
    from ....champions_m4 import context
    lane, m4, k = scenario_start(scenario, cfg, gap, level=level)
    snap = Snapshot(lane, m4, k + k_offset, (cfg.skill_a, cfg.skill_b))
    t = context(cfg).t
    for s in (0, 1):
        kw = {}
        side = m4.side(s)
        if hp is not None and hp[s] is not None:
            L = t.level_by_xp[min(side.xp, t.xp_cap)]
            kw["hp"] = max(1, int(round(hp[s] * t.hp_max[L])))
        if pos is not None and pos[s] is not None:
            kw["pos"] = int(pos[s])
        if kw:
            snap = snap.with_side(s, **kw)
    return snap


def to_json(st: dict) -> dict:
    """A state record without the live Snapshot object (the snapshot as its list form)."""
    out = {k: v for k, v in st.items() if k != "snapshot"}
    out["snapshot"] = st["snapshot"].to_dict()
    return out


__all__ = ["collect", "initialized", "to_json", "ROW_FIELDS", "MIN_TURN"]
