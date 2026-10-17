"""Stage 2 evaluation of the horizon pilot (pre-registration §5). Every family is resumable: each job writes its own
record under results/rl/checkpoints/m47/eval/<family>/... and is read back when present.

Policies of seed s: X in {p/control:s, p/short:s} (the finals; midpoints p/<arm>:s@MIDPOINT where stated) and the
common start ext/baseline:s. Every game is played under the baseline mechanics (gamma does not change play) and SCORED
afterwards from its reward stream under the common objectives: the undiscounted earned-gold lead in elapsed-turn
windows, the discounted lead under gamma 0.99 and under the short gamma (400 or 300 turns, no critic bootstrap).

Families:
    h2h       head to head from s0, short vs control (finals: N_H2H paired-seat games; midpoints: N_H2H_MID), 400 turns
    fixed     each final and the start against fixed opponents from s0: the start ext/baseline:s, the other seeds'
              M4.6c baseline finals ext/baseline:o (o = 2..5) and the M4.5c scripted suite; 400 turns
    natural   self-play records of the finals and the start (m46d.records.play_record; NAT_GAMES games of 400 turns,
              the same game seeds for every policy)
    cells     Stage 1's three paired families (constructed lead cells, natural early states with alternatives, early
              freeze cells) for the finals, on Stage 1's start states and rollout seeds, so they pair with Stage 1's
              ext/baseline:s results exactly (cells, freezes) or by game seed (natural states)
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

from . import common as C
from . import streams as SM

EVAL = os.path.join(C.CKPT, "eval")
T_H2H = 400
N_H2H, N_H2H_MID = 1024, 256
N_START, N_OTHER, N_REF = 256, 64, 32
OTHER_SEEDS = (2, 3, 4, 5)
NAT_GAMES, NAT_TURNS = 96, 400
GAMES_PER_JOB = 32


def finals(s: int) -> Dict[str, str]:
    return {a: f"p/{a}:{s}" for a in C.ARMS}


def _tag(spec: str) -> str:
    return spec.replace("/", "_").replace(":", "_").replace("@", "_at_")


# ---------------------------------------------------------------------------------------------- paired-seat games
def h2h_job(job: dict) -> tuple:
    """`job`: key, path, x (seat spec), y (seat spec), games (list of game indices), root, T. Game m plays x in seat A
    against y in seat B and then y in A against x in B on the same chance seed; stores seat A's reward streams of
    both orientations (rx: x in A; ry: y in A) and the levels of the first orientation."""
    path = job["path"]
    if os.path.exists(path):
        return job["key"], {"path": path, "cached": True}
    cfg = C.setup("baseline")
    from ...m45b import states as S
    start = S.s0(cfg)
    T = int(job["T"])
    games = list(job["games"])
    rx = np.zeros((len(games), T), np.float32)
    ry = np.zeros((len(games), T), np.float32)
    Lx = np.zeros((len(games), T, 2), np.int8)
    dummy = C.net_of(job["x"][1]) if job["x"][0] == "net" else C.net_of(job["y"][1])
    for i, m in enumerate(games):
        sd = int(job["root"]) * 10_000 + m
        pa = SM.seat_policy(tuple(job["x"]), dummy, start[2], sd, 0, T)
        pb = SM.seat_policy(tuple(job["y"]), dummy, start[2], sd, 1, T)
        rx[i], Lx[i], _ = SM.stream_game(cfg, start, pa, pb, sd, T)
        pa = SM.seat_policy(tuple(job["y"]), dummy, start[2], sd, 0, T)
        pb = SM.seat_policy(tuple(job["x"]), dummy, start[2], sd, 1, T)
        ry[i], _, _ = SM.stream_game(cfg, start, pa, pb, sd, T)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, rx=rx, ry=ry, Lx=Lx, games=np.asarray(games))
    os.replace(tmp, path)
    return job["key"], {"path": path}


def _h2h_jobs(family: str, s: int, xname: str, x: tuple, yname: str, y: tuple, n: int, root: int) -> List[dict]:
    js = []
    for b in range(0, n, GAMES_PER_JOB):
        g = list(range(b, min(n, b + GAMES_PER_JOB)))
        d = os.path.join(EVAL, family, f"s{s}", f"{_tag(xname)}__{_tag(yname)}")
        js.append({"key": f"{family}|s{s}|{xname}|{yname}|{b}", "x": x, "y": y, "games": g, "root": root,
                   "T": T_H2H, "path": os.path.join(d, f"{b}.npz")})
    return js


def h2h_jobs(seeds=C.PILOT_SEEDS) -> List[dict]:
    js = []
    for s in seeds:
        f = finals(s)
        js += _h2h_jobs("h2h", s, f["short"], ("net", f["short"]), f["control"], ("net", f["control"]), N_H2H,
                        C.H2H_ROOT)
        ms, mc = f"p/short:{s}@{C.MIDPOINT}", f"p/control:{s}@{C.MIDPOINT}"
        js += _h2h_jobs("h2h", s, ms, ("net", ms), mc, ("net", mc), N_H2H_MID, C.H2H_ROOT + 1)
    return js


def fixed_jobs(seeds=C.PILOT_SEEDS) -> List[dict]:
    from ..m45c.references import SUITE
    js = []
    for s in seeds:
        start = C.general("baseline", s)
        for xname in list(finals(s).values()) + [start]:
            x = ("net", xname)
            if xname != start:
                js += _h2h_jobs("fixed", s, xname, x, start, ("net", start), N_START, C.FIXED_ROOT)
            for o in OTHER_SEEDS:
                other = C.general("baseline", o)
                js += _h2h_jobs("fixed", s, xname, x, other, ("net", other), N_OTHER, C.FIXED_ROOT + 1 + o)
            for r in SUITE:
                js += _h2h_jobs("fixed", s, xname, x, f"ref:{r}", ("ref", r), N_REF, C.FIXED_ROOT + 10)
    return js


def load_pair(family: str, s: int, xname: str, yname: str) -> Dict[str, np.ndarray]:
    d = os.path.join(EVAL, family, f"s{s}", f"{_tag(xname)}__{_tag(yname)}")
    parts = [np.load(os.path.join(d, f)) for f in sorted(os.listdir(d), key=lambda f: int(f.split(".")[0]))]
    out = {k: np.concatenate([p[k] for p in parts]) for k in ("rx", "ry", "Lx", "games")}
    for p in parts:
        p.close()
    return out


# ---------------------------------------------------------------------------------------------- natural records
def natural_path(spec: str) -> str:
    return os.path.join(EVAL, "natural", _tag(spec) + ".npz")


def natural_job(args) -> tuple:
    """(spec, s): NAT_GAMES self-play records of NAT_TURNS turns (m46d format), seeds NAT_ROOT*1000 + 100 s + g."""
    spec, s = args
    p = natural_path(spec)
    if os.path.exists(p):
        return spec, {"path": p, "cached": True}
    cfg = C.setup("baseline")
    from ...m45b import states as S
    from ..m46d.records import play_record
    net = C.net_of(spec)
    games = [play_record(cfg, net, S.s0(cfg), C.NAT_ROOT * 1000 + 100 * s + g, NAT_TURNS) for g in range(NAT_GAMES)]
    out = {k: np.stack([g[k] for g in games]) for k in games[0]}
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + f".{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, **out)
    os.replace(tmp, p)
    return spec, {"path": p}


def natural_jobs(seeds=C.PILOT_SEEDS) -> list:
    return [(spec, s) for s in seeds for spec in list(finals(s).values()) + [C.general("baseline", s)]]


def load_natural(spec: str) -> Dict[str, np.ndarray]:
    with np.load(natural_path(spec)) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------------------------- Stage 1 families
def cell_family_jobs(seeds=C.PILOT_SEEDS) -> List[dict]:
    """Stage 1's paired families re-run for the finals (same start states, combinations and rollout seeds)."""
    from . import delayed as D
    js = []
    for fam, fn in D.FAMILIES.items():
        for j in fn(("baseline",), seeds):
            s = int(j["key"].split("|")[2][1:])
            for arm, spec in finals(s).items():
                k = dict(j)
                k["spec"] = spec
                k["key"] = f"{j['key']}|{arm}"
                k["path"] = os.path.join(EVAL, "cells", fam, arm, f"s{s}", os.path.basename(j["path"]))
                if fam == "natural":                        # the arm's own self-play states, Stage 1's game seeds
                    k["natural"] = tuple(j["natural"])
                js.append(k)
    return js


def cell_path(fam: str, arm: str, s: int, idx) -> str:
    """Path of a Stage 2 cell-family record (arm = control | short) or of its Stage 1 counterpart (arm = start)."""
    from . import delayed as D
    if arm == "start":
        return D.raw(fam, "baseline", s, idx)
    return os.path.join(EVAL, "cells", fam, arm, f"s{s}", f"{idx}.npz")


# ---------------------------------------------------------------------------------------------- dispatch
PARTS = ("h2h", "fixed", "natural", "cells")


def run(workers: int, part: str = None) -> None:
    for p in PARTS:
        if part and p != part:
            continue
        if p == "h2h":
            SM.run_sorted(h2h_job, [dict(j, K=len(j["games"]), combos={"x": 0}) for j in h2h_jobs()], workers,
                          "M4.7 h2h")
        elif p == "fixed":
            SM.run_sorted(h2h_job, [dict(j, K=len(j["games"]), combos={"x": 0}) for j in fixed_jobs()], workers,
                          "M4.7 fixed")
        elif p == "natural":
            C.run_jobs(natural_job, natural_jobs(), workers, "M4.7 natural")
        elif p == "cells":
            SM.run_sorted(SM.stream_job, cell_family_jobs(), workers, "M4.7 cells")


__all__ = ["h2h_jobs", "fixed_jobs", "natural_jobs", "cell_family_jobs", "load_pair", "load_natural", "cell_path",
           "finals", "run", "PARTS", "T_H2H"]
