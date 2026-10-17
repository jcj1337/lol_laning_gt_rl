"""Checkpoint states of M4.8: natural states replayed from the M4.6d records, validation games, and the case criteria.

Natural states (Stage 2). Game g of package (env, s) in the M4.6d natural records is replayed from s0 with its game seed
sd = 99,000,000 + 100 s + g (chance stream default_rng(sd), CRN sampling streams (sd, 7, seat)); the replay is checked
against the stored record on every turn (invested and unspent gold, XP and absence of both seats at the start of the
turn, and the turn's reward) and refused on any difference. The state at the start of turn t0 is the checkpoint; the
record after t0 is the game's realized continuation, used only to check the readout's predictions (never to select).

Validation games (Stage 1). New self-play games of every package (seeds VAL_GAME_ROOT; the M4.6d game design), 160
turns, with the state at the start of every turn and the per-turn event ledger.

Cases. Every criterion reads the checkpoint state and the past only (`describe` plus turns since each seat's last
return), never the future. Seat X is the case's focal seat, Y the other; "case X = A" and "case X = B" are mirror
images:
    returner   X is away after a Recall, or came back at most FRESH_TURNS ago and stands at its own -1 or deeper; at
               least 16 HP of enemy minions alive on X's ledger, at least 8 of them at X's own -1 or deeper (farm
               coming to X); Y present, DUE_TURNS or more past its last return, with >= 3 unspent resources or <= 50%
               HP (Y may still have to reset)
    lost       X is dead with >= 2 turns still away, and >= 12 HP of enemy minions at its own -1 or deeper (a wave
               that reaches X's tower before X is back)
    denied     Y holds a freeze on X's wave next to Y's tower and X is present (X's farm stands on X's +1)
    stable     both present with >= 60% HP and <= 2.5 unspent resources, both 3-15 turns past their last return, enemy
               stocks within 8 HP of each other, no crashed mass under either tower, no freeze
    opposing   both present, |gold lead| >= 1 and the trailing side ahead in XP by >= 15 or in HP by >= 0.3 bars
               (X = the gold leader)
    due        (controlled variants) both present, X = A at least 10 turns past its last return with >= 2 unspent
               resources and >= 30% HP, A not frozen on
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from ....diagnostics import lane_exit as LX
from . import common as C

REC_TURNS = 400
VAL_GAMES, VAL_TURNS = 16, 160
VAL_LO, VAL_HI = 30, 120             # selection turns of validation states (the recorded 40 turns after stay in the game)
N_PER_CASE = 4                       # validation states per case, orientation and package (distinct games)
CASES = ("returner", "lost", "denied", "stable", "opposing")
REC_KEYS = ("S", "g", "xp", "absent")


# ---------------------------------------------------------------------------------------------- games
def play(env: str, s: int, sd: int, turns: int, record_horizon: Optional[int] = None) -> dict:
    """Self-play of package (env, s) from s0 with game seed sd (the M4.6d design); states and the event ledger."""
    from ...m45b import states as S
    cfg = C.setup(env)
    spec = C.package(env, s)
    H = record_horizon or turns
    start = S.s0(cfg)
    pa, pb = C.crn(spec, start[2], sd, 0, H), C.crn(spec, start[2], sd, 1, H)
    return LX.continuation(cfg, start, pa, pb, sd, turns, lane_step=C.lane_step(cfg), keep_states=True)


def history(states: Sequence, i: int) -> Dict[str, float]:
    """Turns since each seat's last return at the start of turn i (from the states' absence counters)."""
    out = {}
    for s, sx in ((0, "A"), (1, "B")):
        ab = [int(st[1].side(s).absent) for st in states[: i + 1]]
        out[f"since_return_{sx}"] = float(LX.since_return(ab, i))
    return out


def record_path(env: str, s: int) -> str:
    from ..m46d import records as R
    return R.path(C.package(env, s), env)


def rec_seed(s: int, g: int) -> int:
    from ..m46d import common as D
    return D.REC_ROOT * 1000 + 100 * s + g


def verify(rec: Dict[str, np.ndarray], g: int, game: dict, upto: int, scale: float) -> None:
    """Refuse a replay that differs from the stored record on any turn before `upto`."""
    for i in range(upto):
        m4 = game["states"][i][1]
        for s in (0, 1):
            me = m4.side(s)
            if (abs(me.items / scale - rec["S"][g, i, s]) > 1e-9 or abs(me.gold / scale - rec["g"][g, i, s]) > 1e-9
                    or me.xp != rec["xp"][g, i, s] or me.absent != rec["absent"][g, i, s]):
                raise AssertionError(f"replay differs from the record: game {g} turn {i} seat {s}")
        if game["r"][i] != rec["r"][g, i]:
            raise AssertionError(f"replay reward differs from the record: game {g} turn {i}")


def natural_job(args) -> tuple:
    """(env, s, games, t0s): replay the M4.6d record games to max(t0s), verify, and return the checkpoint states with
    their descriptions and the record's realized continuations (resumable)."""
    env, s, games, t0s = args
    path = os.path.join(C.CKPT, "states", f"natural_{env}_s{s}.json.gz")
    if os.path.exists(path):
        return (env, s), {"path": path, "cached": True}
    cfg = C.setup(env)
    with np.load(record_path(env, s)) as z:
        rec = {k: z[k] for k in ("S", "g", "xp", "absent", "r", "earned", "farm", "avail", "bounty", "plate", "died",
                                 "recalled", "away")}
    upto = max(t0s)
    out = []
    for g in games:
        game = play(env, s, rec_seed(s, g), upto, record_horizon=REC_TURNS)
        verify(rec, g, game, upto, cfg.gold_scale)
        r = rec["r"][g].astype(float)
        cum = np.concatenate([[0.0], np.cumsum(r)])
        for t0 in t0s:
            st = game["states"][t0]
            d = LX.describe(cfg, st, history(game["states"], t0))
            if abs(d["D0"] - cum[t0]) > 1e-9:
                raise AssertionError(f"checkpoint lead {d['D0']} != summed rewards {cum[t0]} (game {g}, t0 {t0})")
            later = {f"lead@{t}": float(cum[t]) for t in (t0 + 5, t0 + 10, t0 + 20, t0 + 30, t0 + 40, 150, 200, 250,
                                                         300, 400) if t <= REC_TURNS and t > t0}
            comp = {}
            for h in C.WINDOWS:
                w = slice(t0, t0 + h)
                for key in ("farm", "avail", "bounty", "plate", "died", "recalled", "away"):
                    v = rec[key][g, w].astype(float).sum(0)
                    comp[f"{key}_A@{h}"], comp[f"{key}_B@{h}"] = float(v[0]), float(v[1])
            out.append({"env": env, "s": s, "g": int(g), "t0": int(t0), "state": C.state_to_list(st), "desc": d,
                        "rec": {"delta": {str(h): float(cum[t0 + h] - cum[t0]) for h in C.WINDOWS}, **later, **comp}})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    C.save_gz(path, out)
    return (env, s), {"path": path, "states": len(out)}


def natural_states(envs=C.ENVS, seeds=C.SEEDS) -> List[dict]:
    out = []
    for env in envs:
        for s in seeds:
            out += C.load_gz(os.path.join(C.CKPT, "states", f"natural_{env}_s{s}.json.gz"))
    return out


# ---------------------------------------------------------------------------------------------- case criteria
def _role(d: dict, x: str) -> Dict[str, float]:
    """The description from seat x's side (x, y = focal and other)."""
    y = "B" if x == "A" else "A"
    sg = 1.0 if x == "A" else -1.0
    keys = ("status", "returns_in", "hp", "unspent", "stock", "near", "enemy_siege", "pos", "holds_freeze",
            "since_return", "level", "xp")
    o = {f"{k}_x": d[f"{k}_{x}"] for k in keys if f"{k}_{x}" in d}
    o.update({f"{k}_y": d[f"{k}_{y}"] for k in keys if f"{k}_{y}" in d})
    o["lead_x"] = sg * d["D0"]
    o["xp_lead_x"] = sg * d["xp_diff"]
    o["hp_lead_x"] = sg * d["hp_diff"] if np.isfinite(d["hp_diff"]) else float("nan")
    return o


def cases_of(d: dict) -> List[tuple]:
    """[(case, X)] of a described checkpoint (module docstring); X in {"A", "B"}."""
    out = []
    for x in ("A", "B"):
        o = _role(d, x)
        y_present = o["status_y"] == 0
        x_back = (o["status_x"] == 1 and o["returns_in_x"] >= 1) or (
            o["status_x"] == 0 and o["since_return_x"] <= C.FRESH_TURNS and o["pos_x"] <= -1)
        if (x_back and o["stock_x"] >= 16 and o["near_x"] >= 8 and y_present and o["since_return_y"] >= C.DUE_TURNS
                and (o["unspent_y"] >= 3.0 or o["hp_y"] <= 0.5)):
            out.append(("returner", x))
        if o["status_x"] == 2 and o["returns_in_x"] >= 2 and o["near_x"] >= 12:
            out.append(("lost", x))
        if o["status_x"] == 0 and o["holds_freeze_y"] > 0:
            out.append(("denied", x))
        if (o["status_x"] == 0 and y_present and o["lead_x"] >= 1.0
                and ((o["xp_lead_x"] <= -15) or (np.isfinite(o["hp_lead_x"]) and o["hp_lead_x"] <= -0.3))):
            out.append(("opposing", x))
    both = d["status_A"] == 0 and d["status_B"] == 0
    if (both and min(d["hp_A"], d["hp_B"]) >= 0.6 and max(d["unspent_A"], d["unspent_B"]) <= 2.5
            and all(3 <= d[f"since_return_{x}"] <= 15 for x in ("A", "B")) and abs(d["stock_diff"]) <= 8
            and d["enemy_siege_A"] == 0 and d["enemy_siege_B"] == 0
            and d["holds_freeze_A"] == 0 and d["holds_freeze_B"] == 0):
        out.append(("stable", "A"))
    if (both and d["since_return_A"] >= 10 and d["unspent_A"] >= 2.0 and d["hp_A"] >= 0.3
            and d["holds_freeze_B"] == 0):
        out.append(("due", "A"))
    return out


def validation_job(args) -> tuple:
    """(env, s): VAL_GAMES self-play games; every checkpoint in [VAL_LO, VAL_HI] that meets a case; the per-turn ledger
    of each game is kept for the recorded continuations (resumable)."""
    env, s = args
    path = os.path.join(C.CKPT, "states", f"validation_{env}_s{s}.json.gz")
    if os.path.exists(path):
        return (env, s), {"path": path, "cached": True}
    cfg = C.setup(env)
    cands, games = [], []
    for g in range(VAL_GAMES):
        sd = C.VAL_GAME_ROOT * 1000 + 100 * s + g
        game = play(env, s, sd, VAL_TURNS)
        r = game["r"]
        cum = np.concatenate([[0.0], np.cumsum(r)])
        games.append({k: np.asarray(game[k]).tolist() for k in ("r", "earned", "farm", "avail", "bounty", "plate",
                                                                "died", "recalled", "returned", "away", "xpg")})
        for i in range(VAL_LO, VAL_HI + 1):
            st = game["states"][i]
            d = LX.describe(cfg, st, history(game["states"], i))
            cs = cases_of(d)
            if cs:
                cands.append({"env": env, "s": s, "g": g, "t0": i, "cases": cs, "state": C.state_to_list(st),
                              "desc": d, "rec": {"delta": {str(h): float(cum[i + h] - cum[i]) for h in C.WINDOWS}}})
    out = {"candidates": cands, "games": games}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    C.save_gz(path, out)
    return (env, s), {"path": path, "candidates": len(cands)}


def load_validation(env: str, s: int) -> dict:
    return C.load_gz(os.path.join(C.CKPT, "states", f"validation_{env}_s{s}.json.gz"))


def select_validation(envs=C.ENVS, seeds=C.SEEDS, n: int = N_PER_CASE) -> List[dict]:
    """Up to n states per (package, case, X), from distinct games, drawn uniformly among the qualifying checkpoints with
    a fixed generator (VAL_PICK): no outcome is read."""
    out = []
    for ei, env in enumerate(envs):
        for s in seeds:
            v = load_validation(env, s)
            for case in CASES + ("due",):
                for x in ("A", "B"):
                    pool = [c for c in v["candidates"] if (case, x) in [tuple(t) for t in c["cases"]]]
                    if not pool:
                        continue
                    rng = np.random.default_rng([C.VAL_PICK, ei, s, CASES.index(case) if case in CASES else 9,
                                                 0 if x == "A" else 1])
                    by_game: Dict[int, List[dict]] = {}
                    for c in pool:
                        by_game.setdefault(c["g"], []).append(c)
                    gs = sorted(by_game)
                    rng.shuffle(gs)
                    for g in gs[:n]:
                        c = by_game[g][int(rng.integers(len(by_game[g])))]
                        out.append({**{k: c[k] for k in ("env", "s", "g", "t0", "state", "desc", "rec")},
                                    "case": case, "X": x})
    return out


__all__ = ["play", "history", "natural_job", "natural_states", "cases_of", "validation_job", "load_validation",
           "select_validation", "CASES", "VAL_GAMES", "VAL_TURNS", "N_PER_CASE"]
