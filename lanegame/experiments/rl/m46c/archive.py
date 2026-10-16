"""The M4.6c training start archive and the held-out evaluation cells (pre-registration §3).

One archive file serves BOTH arms. Every state is defined by gold, levels, HP fractions, positions and waves, never
by a combat ratio, so the same state means a different strength gap under each curve: the effect under test.

Training archive (seat A is the focal side of the stored state; the collector mirrors half of the draws):
    natural       N_NAT states per M4.5c final: decision states of its s0 self-play in the BASELINE environment
                  (seeds ARCH_ROOT, turns 5-299, both seats, mirrored to A), unmodified
    natural+lead  N_LEAD more such states with A's INVESTED gold raised by d, d log-uniform in [2, 32] resources
                  (5-80 turns of the measured income 0.40)
    zone          constructed M4.5b zoning situations (neutral / A's push / B's push), level uniform in 3-18,
                  invested gold S(L) x U(0.7, 1.3) for both (L 18: S log-uniform in [56, 240]); A's lead d = 0 with
                  probability 1/4, else log-uniform in [2, 32]; HP full for both with probability 1/2, else each
                  U(0.35, 1); a uniformly drawn phase
    freeze        the M4.5b freeze builder (A holds B's wave), level uniform in 3-15, held 5 / 8 / 12 / 16 / 20, B
                  walked up (+1) or at its lane cell (0), A's remaining plates taken with probability 0.2, invested
                  S(L) x U(0.7, 1.3) for both and a lead d log-uniform in [2, 24] on the freezer, the breaker or
                  neither (1/3 each); HP as for zone; phase 51 (the builder's)
S(L) interpolates the M4.6b audit's natural (median level, median investment) path: (3, 0), (6, 4.9), (9, 15),
(12, 25), (14, 35), (17, 50), (18, 56).

Held-out evaluation cells (never in the archive; checked by key): the M4.6b stage cells (early L8 S9, mid L16 S42,
late L18 S108; neutral / A's push / B's push; leads 0, +8 and +16 on either side; phase 2, full HP) and the M4.6b S4
freeze cells (levels 4 / 10 / 15, held 5 / 12, investment equal / freezer +8 / breaker +8). Natural evaluation states
come from each evaluated policy's own self-play on EVAL_STATE_ROOT seeds (evaluate.py).
"""

from __future__ import annotations

import math
import os
import time
from typing import Dict, List, Tuple

import numpy as np

from . import common as C

N_NAT, N_LEAD, N_ZONE, N_FREEZE = 300, 250, 900, 600
NAT_TURNS, NAT_MIN_TURN, NAT_PER_GAME = 300, 5, 10
LEAD_RANGE = (2.0, 32.0)
FREEZE_LEAD_RANGE = (2.0, 24.0)
LEVEL_S_PATH = ((3, 0.0), (6, 4.9), (9, 15.0), (12, 25.0), (14, 35.0), (17, 50.0), (18, 56.0))
LATE_S_RANGE = (56.0, 240.0)
HELD = (5, 8, 12, 16, 20)


def archive_path() -> str:
    return os.path.join(C.CKPT, "archive", "train_archive.json.gz")


def s_of_level(L: int) -> float:
    xs, ys = zip(*LEVEL_S_PATH)
    return float(np.interp(L, xs, ys))


def _loguniform(rng, lo, hi) -> float:
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def _base_s(rng, L: int) -> float:
    if L >= 18:
        return round(_loguniform(rng, *LATE_S_RANGE), 2)
    return round(s_of_level(L) * rng.uniform(0.7, 1.3), 2)


def _hp(rng) -> Tuple[float, float]:
    if rng.random() < 0.5:
        return 1.0, 1.0
    return round(rng.uniform(0.35, 1.0), 3), round(rng.uniform(0.35, 1.0), 3)


def _with_items(cfg, m4, s_a: float, s_b: float):
    sc = cfg.gold_scale
    return m4._replace(a=m4.a._replace(items=int(round(s_a * sc))), b=m4.b._replace(items=int(round(s_b * sc))))


# ---------------------------------------------------------------------------------------------- natural states
def natural_job(args) -> tuple:
    """(final, n, root, offset): n decision states of the final's s0 self-play in the baseline environment."""
    final, n, root, offset = args
    cfg = C.setup("baseline")
    from ...m45b import engine as E
    from ...m45b import states as S
    from ..m45c import horizon as HZ
    net = C.net_of(f"start:{final}")
    out, games = [], 0
    for g in range(400):
        sd = root * 1000 + 100 * final + offset + g
        pa, pb = HZ.CRNNet(net, 0, sd, 0, NAT_TURNS), HZ.CRNNet(net, 0, sd, 1, NAT_TURNS)
        out += E.collect(cfg, S.s0(cfg), pa, pb, [sd], NAT_TURNS, lambda *a: True, per_game=NAT_PER_GAME,
                         min_turn=NAT_MIN_TURN, rng_seed=sd)
        games += 1
        if len(out) >= n:
            break
    return (final, offset), {"states": [C.state_to_list(x["start"]) for x in out[:n]],
                             "turns": [x["turn"] for x in out[:n]], "games": games}


# ---------------------------------------------------------------------------------------------- constructed states
def zone_draw(cfg, rng):
    from ...m45b import states as S
    sit = ("neutral", "a_push", "b_push")[int(rng.integers(3))]
    L = int(rng.integers(3, 19))
    s0 = _base_s(rng, L)
    d = 0.0 if rng.random() < 0.25 else round(_loguniform(rng, *LEAD_RANGE), 2)
    hp = _hp(rng)
    lane, pa, pb, _ = S.ZONING_SITUATIONS[sit]
    m4 = S.champs(cfg, (L, L), (s0 + d, s0), (pa, pb), hp)
    k = int(rng.integers(cfg.phase_period))
    return (lane, m4, k), {"sit": sit, "level": L, "S": s0, "lead": d, "hp": hp, "k": k}


def freeze_draw(cfg, rng):
    from ...m45b import states as S
    L = int(rng.integers(3, 16))
    held = HELD[int(rng.integers(len(HELD)))]
    b_pos = (1, 0)[int(rng.integers(2))]
    plates_left = rng.random() >= 0.2
    hp = _hp(rng)
    lane, m4, k = S.freeze_state(cfg, level=L, gap="equal", held=held, own=8, hp=hp, b_pos=b_pos,
                                 plates_left=plates_left)
    s0 = _base_s(rng, L)
    who = ("freezer", "breaker", "none")[int(rng.integers(3))]
    d = 0.0 if who == "none" else round(_loguniform(rng, *FREEZE_LEAD_RANGE), 2)
    s_a, s_b = (s0 + d, s0) if who == "freezer" else ((s0, s0 + d) if who == "breaker" else (s0, s0))
    return (lane, _with_items(cfg, m4, s_a, s_b), k), {"level": L, "held": held, "b_pos": b_pos,
                                                        "plates_left": plates_left, "S": s0, "lead": d, "who": who,
                                                        "hp": hp}


# ---------------------------------------------------------------------------------------------- evaluation cells
STAGES = {"early": (8, 9.0), "mid": (16, 42.0), "late": (18, 108.0)}
SITS = ("neutral", "a_push", "b_push")
LEAD_CELLS = ("equal", "A+8", "B+8", "A+16", "B+16")
FREEZE_LEVELS, FREEZE_HELD = (4, 10, 15), (5, 12)
FREEZE_INV = ("equal", "freezer+8", "breaker+8")


def lead_cell(cfg, stage: str, sit: str, cell: str, extra: float = 0.0):
    """An M4.6b stage cell (`extra`: a different lead size for the strength-matched supplement)."""
    from ...m45b import states as S
    L, s0 = STAGES[stage]
    d = 0.0 if cell == "equal" else float(cell[2:]) + extra
    s_a, s_b = (s0 + d, s0) if cell.startswith("A") else ((s0, s0 + d) if cell.startswith("B") else (s0, s0))
    lane, pa, pb, _ = S.ZONING_SITUATIONS[sit]
    return lane, S.champs(cfg, (L, L), (s_a, s_b), (pa, pb)), 2


def lead_cells() -> List[Tuple[int, tuple]]:
    return list(enumerate((st, sit, c) for st in STAGES for sit in SITS for c in LEAD_CELLS))


def freeze_cell(cfg, L: int, held: int, inv: str):
    from ...m45b import states as S
    from ...m45b.states import typical_invest
    lane, m4, k = S.freeze_state(cfg, level=L, gap="equal", held=held)
    base = typical_invest(L)
    s_a, s_b = {"equal": (base, base), "freezer+8": (base + 8, base), "breaker+8": (base, base + 8)}[inv]
    return lane, _with_items(cfg, m4, s_a, s_b), k


def freeze_cells() -> List[Tuple[int, tuple]]:
    return list(enumerate((L, h, inv) for L in FREEZE_LEVELS for h in FREEZE_HELD for inv in FREEZE_INV))


def eval_keys(cfg) -> set:
    """Identity keys of every held-out constructed evaluation state, both orientations."""
    keys = set()
    sts = [lead_cell(cfg, *c) for _, c in lead_cells()] + [freeze_cell(cfg, *c) for _, c in freeze_cells()]
    for lane, m4, k in sts:
        keys.add((lane, m4, k % cfg.phase_period))
        keys.add((lane.mirror(), m4.mirror(), k % cfg.phase_period))
    return keys


# ---------------------------------------------------------------------------------------------- build
def build(workers: int = 6, force: bool = False) -> dict:
    """Write the training archive (idempotent unless `force`); returns its description."""
    from ....env.starts import StartState, save_archive
    path = archive_path()
    if os.path.exists(path) and not force:
        return C.load("archive") or {"path": path}
    cfg = C.arm_cfg("baseline")
    t0 = time.perf_counter()
    js = [(f, N_NAT, C.ARCH_ROOT, 0) for f in C.SEEDS] + [(f, N_LEAD, C.ARCH_ROOT, 50) for f in C.SEEDS]
    res = dict(C.run_jobs(natural_job, js, workers, "M4.6c archive natural states"))
    states: List = []
    desc: Dict[str, dict] = {"natural": {}, "natural+lead": {}}
    rng = np.random.default_rng([C.ARCH_CON_ROOT, 1])
    leads = []
    for f in C.SEEDS:
        nat = res[(f, 0)]
        for v in nat["states"]:
            lane, m4, k = C.state_from_list(v)
            states.append(StartState(lane, m4, k % cfg.phase_period, None, f"natural s{f}"))
        desc["natural"][str(f)] = {"states": len(nat["states"]), "games": nat["games"],
                                   "turn_median": float(np.median(nat["turns"]))}
        nl = res[(f, 50)]
        for v in nl["states"]:
            lane, m4, k = C.state_from_list(v)
            d = round(_loguniform(rng, *LEAD_RANGE), 2)
            leads.append(d)
            m4 = m4._replace(a=m4.a._replace(items=m4.a.items + int(round(d * cfg.gold_scale))))
            states.append(StartState(lane, m4, k % cfg.phase_period, None, f"natural+lead s{f} d={d}"))
        desc["natural+lead"][str(f)] = {"states": len(nl["states"]), "games": nl["games"]}
    zrng = np.random.default_rng([C.ARCH_CON_ROOT, 2])
    zd = []
    for _ in range(N_ZONE):
        (lane, m4, k), d = zone_draw(cfg, zrng)
        zd.append(d)
        states.append(StartState(lane, m4, k % cfg.phase_period, None,
                                 f"zone {d['sit']} L{d['level']} S{d['S']} d={d['lead']}"))
    frng = np.random.default_rng([C.ARCH_CON_ROOT, 3])
    fd = []
    for _ in range(N_FREEZE):
        (lane, m4, k), d = freeze_draw(cfg, frng)
        fd.append(d)
        states.append(StartState(lane, m4, k % cfg.phase_period, None,
                                 f"freeze L{d['level']} held{d['held']} {d['who']} d={d['lead']}"))
    held_out = eval_keys(cfg)
    before = len(states)
    states = [s for s in states if s.key() not in held_out]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_archive(path, states, cfg, {"stage": "m46c archive", "roots": [C.ARCH_ROOT, C.ARCH_CON_ROOT],
                                     "sizes": {"natural": N_NAT, "natural+lead": N_LEAD, "zone": N_ZONE,
                                               "freeze": N_FREEZE}})
    labels: Dict[str, int] = {}
    for s in states:
        lb = s.label.split(" ")[0]
        labels[lb] = labels.get(lb, 0) + 1
    out = {"meta": C.meta(stage="archive"), "path": path, "states": len(states), "dropped_eval_collisions": before - len(states),
           "by_label": labels, "natural": desc, "lead_quantiles": {str(q): float(np.quantile(leads, q)) for q in (0.1, 0.5, 0.9)},
           "zone_levels": np.bincount([d["level"] for d in zd], minlength=19).tolist(),
           "zone_lead_zero_share": float(np.mean([d["lead"] == 0 for d in zd])),
           "freeze_who": {w: int(sum(d["who"] == w for d in fd)) for w in ("freezer", "breaker", "none")},
           "elapsed_s": time.perf_counter() - t0}
    C.save_json("archive", out)
    return out


def check_states(n: int = 200) -> dict:
    """Every archived state (a sample of n) resets both arms' environments and has a legal action for both seats."""
    from ....env import LaneEnv
    from ....env.starts import load_archive
    sts = load_archive(archive_path(), C.arm_cfg("baseline"))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(sts), size=min(n, len(sts)), replace=False)
    ok = {}
    for arm in C.ARMS:
        cfg = C.arm_cfg(arm)
        env = LaneEnv(cfg, max_turns=3)
        good = 0
        for i in idx:
            st = sts[int(i)]
            for s in (st, st.mirror()):
                o = env.reset(seed=int(i), root=s.lane, k0=s.phase, champ=s.champ)
                if o[0].mask.any() and o[1].mask.any():
                    r = env.step(int(o[0].legal[0]), int(o[1].legal[0]))
                    good += int(np.isfinite(r.rewards[0]))
        ok[arm] = {"checked": 2 * len(idx), "ok": good}
    return ok


__all__ = ["build", "archive_path", "lead_cell", "lead_cells", "freeze_cell", "freeze_cells", "eval_keys", "STAGES",
           "SITS", "LEAD_CELLS", "check_states", "s_of_level"]
