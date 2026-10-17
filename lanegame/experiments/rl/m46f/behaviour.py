"""Behavioural evaluation of M4.6f (pre-registration "Behaviour"): actual rollouts only, never critic scores.

Parts (each resumable; per-job records under results/rl/checkpoints/m46f/behaviour/<phase>/):
    h2h       E1 head to head from s0 (`env.discounted.evaluate_discounted`, paired seats): separate vs shared
              (64, W 1000); each final vs start:s (64, W 1000); vs the other five starts (16, W 460); vs the scripted
              core suite (16, W 460)
    curve     separate vs shared at the 1.024M / 2.048M / 3.072M snapshots (32, W 460)
    conv      B2 strength use: M4.6c lead cells and M4.6d lane cells (+ the x1.76 diagnostic), K 16 x H 150
    freeze    B3: M4.6d freeze cells, the freezer and (mirrored) the breaker in seat A, K 32 x H 50
    recall    B4: Recall windows, the weak side (mirrored) and the strong side in seat A, K 32 x H 100; plus the first-turn
              action probabilities of each actor
    natural   B5: start:s's natural leader / trailer states (K 16 x H 150) and favourable natural early freezes, the
              freezer and (mirrored) the breaker in seat A (K 16 x H 50)
Paired jobs run the M4.6a engine through `m46c.evaluate.pjob`: candidates {"shared": ("net", shared final), "separate":
("net", separate final)} in seat A, the opponent protocol "net:<final of either arm>" in seat B, the same rollout seeds
for both candidates and both opponents (CRN). The engine's ledger diff is separate - shared.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import common as C

LONG_W, SHORT_W = 1000, 460
CORE_SUITE = ("g_farm", "g_shove", "g_bully", "fz_guard_sh")
K_CONV, H_CONV = 16, 150
K_FZ, H_FZ = 32, 50
K_RW, H_RW = 32, 100
K_NAT, H_NAT, H_NATFZ = 16, 150, 50

CONV_CONDS = ("A+8", "A+16", "edge", "plaus", "hp")
DEF_CONDS = ("B+8", "B+16", "brk")
LANE_CELLS = (("equal", "equal", 1.0), ("edge", "edge", 1.0), ("plaus", "edge", 0.6), ("hp", "equal", 0.6),
              ("brk", "brk", 1.0), ("x1.76", "x1.76", 1.0))
FREEZE_NAMES = ("equal", "edge", "plaus", "hp", "brk", "x1.76")
RW_CONDS, RW_LEVELS, RW_HP = ("equal", "edge", "x1.76"), (5, 8), (0.5, 0.35)


def finals(seed: int, phase: str) -> Dict[str, str]:
    pre = "ext/" if phase == "ext" else ""
    return {a: C.policy_path(f"{pre}{a}:{seed}") for a in C.ARMS}


def edir(phase: str) -> str:
    return os.path.join(C.CKPT, "behaviour", phase)


# ---------------------------------------------------------------------------------------------- head to head
def h2h_job(args) -> tuple:
    """(key, x path, y path or ref:<name>, episodes, seed, window, out path)."""
    key, xs, ys, episodes, seed, window, path = args
    if os.path.exists(path):
        return key, C.load_gz(path)
    import torch
    torch.set_num_threads(1)
    from ....env.discounted import evaluate_discounted
    from ....rl import NetAgent
    cfg = C.env_cfg()
    x = NetAgent(C.net_of(xs), name=xs, copy_net=False)
    if ys.startswith("ref:"):
        from ..m45c.references import reference
        y = reference(ys[4:])
    else:
        y = NetAgent(C.net_of(ys), name=ys, copy_net=False)
    e = evaluate_discounted(cfg, x, y, episodes=episodes, seed=seed, window=window, cache_size=5000)
    d = {k: e[k] for k in ("estimate", "stderr", "ci95", "seat_A", "seat_B", "window", "pairs", "elapsed_s")}
    d.update({"key": key, "x": xs, "y": ys})
    C.save_gz(path, d)
    return key, d


def h2h_jobs(phase: str) -> List[tuple]:
    js = []
    d = os.path.join(edir(phase), "h2h")
    for s in C.SEEDS:
        f = finals(s, phase)
        js.append((f"ivc|s{s}", f["separate"], f["shared"], 64, C.H2H_ROOT, LONG_W, os.path.join(d, f"ivc_s{s}.json.gz")))
        for a in C.ARMS:
            js.append((f"start|{a}|s{s}", f[a], C.start_path(s), 64, C.H2H_ROOT + 1, LONG_W,
                       os.path.join(d, f"start_{a}_s{s}.json.gz")))
            for o in C.SEEDS:
                if o != s:
                    js.append((f"others|{a}|s{s}|o{o}", f[a], C.start_path(o), 16, C.H2H_ROOT + 2, SHORT_W,
                               os.path.join(d, f"others_{a}_s{s}_o{o}.json.gz")))
            for r in CORE_SUITE:
                js.append((f"suite|{a}|s{s}|{r}", f[a], f"ref:{r}", 16, C.H2H_ROOT + 3, SHORT_W,
                           os.path.join(d, f"suite_{a}_s{s}_{r}.json.gz")))
    return js


POWER_EPISODES = 256            # amendment 1: the powered B1 (a new seed root; the registered 64 episodes unchanged)


def h2h_power_jobs(phase: str) -> List[tuple]:
    d = os.path.join(edir(phase), "h2h")
    return [(f"ivc_power|s{s}", finals(s, phase)["separate"], finals(s, phase)["shared"], POWER_EPISODES,
             C.H2H_ROOT + 10, LONG_W, os.path.join(d, f"ivc_power_s{s}.json.gz")) for s in C.SEEDS]


def curve_jobs() -> List[tuple]:
    d = os.path.join(edir("main"), "curve")
    return [(f"curve|s{s}|{n}", C.snapshot_path("separate", s, n), C.snapshot_path("shared", s, n), 32,
             C.H2H_ROOT + 4, SHORT_W, os.path.join(d, f"curve_s{s}_{n}.json.gz"))
            for s in C.SEEDS for n in C.CURVE_POINTS]


# ---------------------------------------------------------------------------------------------- cells
def conv_cells(cfg) -> List[Tuple[str, tuple, dict]]:
    """(label, start, {group, cond}) of the B2 cells (module docstring)."""
    from ..m46c.archive import LEAD_CELLS, SITS, lead_cell
    from ..m46d import cells as CL
    out = []
    for st in ("early", "mid"):
        for sit in SITS:
            for c in LEAD_CELLS:
                out.append((f"lead {st} {sit} {c}", lead_cell(cfg, st, sit, c), {"group": f"lead {st} {sit}",
                                                                                 "cond": c}))
    for wave in ("neutral", "push"):
        for L in (4, 8):
            for name, cond, hpb in LANE_CELLS:
                out.append((f"lane {wave} L{L} {name}", CL.build(cfg, wave, cond, L, hpb),
                            {"group": f"lane {wave} L{L}", "cond": name}))
    return out


def freeze_cells(cfg) -> List[Tuple[str, tuple, dict]]:
    from ..m46d import cells as CL
    out = []
    for cell in CL.grid_a():
        if cell["held"] != 12 or cell["name"] not in FREEZE_NAMES:
            continue
        st = CL.cell_start(cfg, cell)
        lane, m4, k = st
        meta = {"group": f"freeze L{cell['L']}", "cond": cell["name"], "L": cell["L"]}
        out.append((f"freezer L{cell['L']} {cell['name']}", st, {**meta, "role": "freezer"}))
        out.append((f"breaker L{cell['L']} {cell['name']}", (lane.mirror(), m4.mirror(), k), {**meta, "role": "breaker"}))
    return out


def recall_cells(cfg) -> List[Tuple[str, tuple, dict]]:
    from ..m46d.critic import recall_window
    out = []
    for cond in RW_CONDS:
        for L in RW_LEVELS:
            for hpb in RW_HP:
                lane, m4, k = recall_window(cfg, cond, L, hpb)
                meta = {"group": f"recallw L{L} hpB{hpb}", "cond": cond, "L": L, "hp": hpb}
                out.append((f"strong L{L} {cond} hpB{hpb}", (lane, m4, k), {**meta, "role": "strong"}))
                out.append((f"weak L{L} {cond} hpB{hpb}", (lane.mirror(), m4.mirror(), k), {**meta, "role": "weak"}))
    return out


def natural_cells(seed: int) -> List[Tuple[str, tuple, dict, int]]:
    """(label, start, meta, horizon): start:s's M4.6c leader / trailer states and M4.6d favourable natural freezes."""
    from ..m46c import common as C46
    from ..m46d import common as D
    out = []
    for role in ("leader", "trailer"):
        p = os.path.join(C46.CKPT, "eval", "states", "C1", "ext", f"C1_{seed}", f"{role}.json.gz")
        for i, v in enumerate(C.load_gz(p)["states"]):
            out.append((f"natural {role} {i}", C.state_from_list(v), {"role": role, "i": i}, H_NAT))
    nf = C.load_gz(os.path.join(D.CKPT, "natural_freeze_states.json.gz"))[f"ext/C1:{seed}"]["favourable"]
    for i, rec in enumerate(nf):
        lane, m4, k = C.state_from_list(rec["start"])
        out.append((f"natural freezer {i}", (lane, m4, k), {"role": "freezer", "i": i}, H_NATFZ))
        out.append((f"natural breaker {i}", (lane.mirror(), m4.mirror(), k), {"role": "breaker", "i": i}, H_NATFZ))
    return out


def _paired(family: str, phase: str, seed: int, idx: int, label: str, start, meta: dict, K: int, H: int,
            root: int) -> List[dict]:
    from ..m46a import engine as EN
    f = finals(seed, phase)
    cands = {"shared": ("net", f["shared"]), "separate": ("net", f["separate"])}
    js = []
    for opp in C.ARMS:
        js.append({"spec": f[opp], "family": family, "idx": idx, "start": start, "proto": f"net:{f[opp]}",
                   "horizon": H, "roll": EN.roll_seeds(root, K), "cands": cands, "critic": False,
                   "tag": f"s{seed}_vs_{opp}", "arm": C.ENV, "label": label, "meta": meta,
                   "path": os.path.join(edir(phase), family, f"s{seed}", f"{idx}_vs_{opp}.json.gz")})
    return js


def cell_jobs(family: str, phase: str) -> List[dict]:
    cfg = C.env_cfg()
    js = []
    for s in C.SEEDS:
        if family == "conv":
            items = [(lab, st, m, K_CONV, H_CONV) for lab, st, m in conv_cells(cfg)]
            root = C.CELL_ROOT
        elif family == "freeze":
            items = [(lab, st, m, K_FZ, H_FZ) for lab, st, m in freeze_cells(cfg)]
            root = C.CELL_ROOT + 1
        elif family == "recall":
            items = [(lab, st, m, K_RW, H_RW) for lab, st, m in recall_cells(cfg)]
            root = C.CELL_ROOT + 2
        elif family == "natural":
            items = [(lab, st, m, K_NAT, h) for lab, st, m, h in natural_cells(s)]
            root = C.NATCF_ROOT
        else:
            raise ValueError(family)
        for i, (lab, st, m, K, H) in enumerate(items):
            js += _paired(family, phase, s, i, lab, st, m, K, H, root)
    return js


def pjob(job: dict) -> tuple:
    from ..m46c.evaluate import pjob as p46
    return p46({k: v for k, v in job.items() if k != "meta"})


# ---------------------------------------------------------------------------------------------- first-turn actions
def first_turn_job(args) -> tuple:
    """(seed, phase): each actor's first-turn probability mass by behaviour (Hold / Yield / Trade / Safe / Recall) and
    by wave action at the seat-A side of every Recall-window and freeze cell."""
    seed, phase = args
    import torch
    from ....rl.network import MaskedCategorical
    from ..m44 import actions as AC
    from ..m46e.observation import _obs
    cfg = C.setup()
    f = finals(seed, phase)
    nets = {"start": C.net_of(C.start_path(seed)), **{a: C.net_of(p) for a, p in f.items()}}
    beh = np.asarray([AC.decode(c).behavior if not AC.decode(c).absent else -1 for c in range(AC.N_CODES)])
    rec = np.asarray([AC.decode(c).recall for c in range(AC.N_CODES)])
    wave = np.asarray([AC.decode(c).wave_index for c in range(AC.N_CODES)])
    out = {}
    for lab, st, meta in recall_cells(cfg) + freeze_cells(cfg):
        o = _obs(cfg, st, 0)
        x = torch.as_tensor(o.vector, dtype=torch.float32).unsqueeze(0)
        m = torch.as_tensor(o.mask).unsqueeze(0)
        row = {}
        for name, net in nets.items():
            with torch.no_grad():
                p = MaskedCategorical(net(x)[0], m).probs[0].double().numpy()
            row[name] = {"recall": float(p[rec].sum()),
                         **{AC.BEH_NAMES[b]: float(p[(beh == b) & ~rec].sum()) for b in range(4)},
                         **{AC.WAVE_KEYS[w]: float(p[(wave == w) & ~rec].sum()) for w in range(4)}}
        out[lab] = row
    return seed, out


# ---------------------------------------------------------------------------------------------- economy (amendment 2)
ECON_GAMES, ECON_TURNS = 16, 600


def economy_job(args) -> tuple:
    """(tag, checkpoint path, seed, out path): EXPLORATORY (amendment 2). The network playing itself from s0, ECON_GAMES
    x ECON_TURNS on seeds shared by every network of the same s; per side: the M4.5c game summary plus Recall timing
    and spending (unspent gold at Recall, mean unspent gold while present, invested at turns 150 / 300 / 600)."""
    tag, path, seed, out = args
    if os.path.exists(out):
        return tag, C.load_gz(out)
    import torch
    torch.set_num_threads(1)
    from ....rl import NetAgent
    from ..m44.records import play_records
    from ..m45c import behavior as BH
    cfg = C.setup()
    x = NetAgent(C.net_of(path), name=tag, copy_net=False)
    recs = play_records(cfg, x, x, ECON_GAMES, ECON_TURNS, C.NATCF_ROOT + 50 + seed, both_seats=False,
                        env_label="m46f")
    vals = {k: [] for k in ("gold_at_recall", "gold_present_mean", "invested_150", "invested_300", "invested_600",
                            "recalls_gold_lt2_share")}
    for rec in recs:
        d = rec.data
        for s in (0, 1):
            pres = d["present"][:, s] > 0
            rc = d["recalled"][:, s] > 0
            g = d["gold"][:, s]
            if rc.any():
                vals["gold_at_recall"].append(float(g[rc].mean()))
                vals["recalls_gold_lt2_share"].append(float((g[rc] < 2).mean()))
            vals["gold_present_mean"].append(float(g[pres].mean()))
            for t in (150, 300, 600):
                vals[f"invested_{t}"].append(float(d["invested"][min(t, len(g)) - 1, s]))
    res = {"summary": {k: v[0] for k, v in BH.game_summary(recs, None).items()},
           "economy": {k: float(np.mean(v)) if v else float("nan") for k, v in vals.items()}}
    C.save_gz(out, res)
    return tag, res


def economy_jobs(phase: str) -> List[tuple]:
    d = os.path.join(edir(phase), "economy")
    js = []
    for s in C.SEEDS:
        for a, p in finals(s, phase).items():
            js.append((f"{a}_s{s}", p, s, os.path.join(d, f"{a}_s{s}.json.gz")))
        if phase == "main":
            js.append((f"start_s{s}", C.start_path(s), s, os.path.join(d, f"start_s{s}.json.gz")))
    return js


# ---------------------------------------------------------------------------------------------- driver
PARTS = ("h2h", "curve", "conv", "freeze", "recall", "natural")


def run(workers: int, phase: str = "main", part: Optional[str] = None) -> None:
    parts = (part or ",".join(p for p in PARTS if phase == "main" or p != "curve")).split(",")
    for p in parts:
        t0 = time.perf_counter()
        if p == "h2h":
            js = sorted(h2h_jobs(phase), key=lambda j: -j[3] * j[5])
            res = C.run_jobs(h2h_job, js, workers, f"h2h {phase}")
        elif p == "curve":
            js = curve_jobs()
            res = C.run_jobs(h2h_job, js, workers, "curve")
        elif p == "h2h_power":
            js = h2h_power_jobs(phase)
            res = C.run_jobs(h2h_job, js, workers, f"h2h_power {phase}")
        elif p == "economy":
            js = economy_jobs(phase)
            res = C.run_jobs(economy_job, js, workers, f"economy {phase}")
        elif p in ("conv", "freeze", "recall", "natural"):
            js = sorted(cell_jobs(p, phase), key=lambda j: -len(j["roll"]) * j["horizon"])
            res = C.run_jobs(pjob, js, workers, f"{p} {phase}")
            C.save_json(f"cells_{p}_{phase}", {"jobs": [{k: j[k] for k in ("family", "idx", "label", "meta", "tag",
                                                                           "path", "horizon")} | {"K": len(j["roll"])}
                                                        for j in js]})
            if p in ("recall", "freeze") and not os.path.exists(os.path.join(edir(phase), f"first_turn.json.gz")):
                ft = dict(C.run_jobs(first_turn_job, [(s, phase) for s in C.SEEDS], workers, "first-turn actions"))
                C.save_gz(os.path.join(edir(phase), "first_turn.json.gz"), {str(k): v for k, v in ft.items()})
        else:
            raise SystemExit(f"unknown behaviour part {p!r}")
        print(f"  [{p}] {len(js)} jobs in {time.perf_counter() - t0:.0f} s", flush=True)
        C.jdump(os.path.join(C.OUT, "stages", f"behaviour_{phase}_{p}.json"),
                {"part": p, "phase": phase, "jobs": len(js), "wall_s": time.perf_counter() - t0,
                 "finished": time.strftime("%Y-%m-%d %H:%M:%S")})


__all__ = ["h2h_job", "h2h_jobs", "curve_jobs", "conv_cells", "freeze_cells", "recall_cells", "natural_cells",
           "cell_jobs", "pjob", "first_turn_job", "run", "CONV_CONDS", "DEF_CONDS", "PARTS"]
