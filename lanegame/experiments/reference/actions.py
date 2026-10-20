"""Legality, the observation and the reach of the wave choice on `m45_calibrated()`, measured without a learned policy.

  exhaustive    every input `champions.champion_legal_actions` reads (absence 0..5, position -2..+2 or W, own crashed
                mass zero or positive): the codes that can ever be legal and the legal-set size by position and siege
  random_play   uniform random legal play through `LaneEnv`, both seats, N_GAMES games of TURNS turns on seeds
                SEED0.. (the actions drawn from numpy default_rng(ACTION_SEED)). Per seat-turn:
                  * the engine's legal set against the rule of reference §3.3 (mismatches), the codes seen, the sizes;
                  * for a lane action (not ABSENT, Recall or Ward), whether the end cell depends on the wave slot: the
                    turn is re-resolved for each of the four slots under the same behaviour and objective with the
                    opponent's code fixed (movement precedes every chance draw, so the chance vector plays no part);
                    split by behaviour (Pressure separately) and by 50-turn window;
                  * the observation: every entry in [0, 1], `decode_m4` returns the true phase, lane and champion state
                    (the plate counters excepted), the absence entry equals absence / 5, the number at W;
                  * how champions arrive at W (by code), and the absence a death and a Recall set (deaths by the band
                    of the start-of-turn level);
                  * the largest number of rear packets in one cell.
  level_timing  scripted Last Hit + Hold against itself (the first legal code when Last Hit + Hold is not legal) from s0
                with chance events, LEVEL_SEEDS seeds of LEVEL_TURNS turns, on `m45_calibrated()` and `m4_lasthit()`:
                the median turn at which a seat reaches levels 2, 6, 11 and 18
"""

from __future__ import annotations

import collections
from typing import Dict, List

import numpy as np

from .common import cfg

N_GAMES, TURNS, SEED0, ACTION_SEED, CHUNK = 200, 200, 0, 20_261_020, 20
LEVEL_SEEDS, LEVEL_TURNS, LEVELS = 60, 300, (2, 6, 11, 18)


def exhaustive() -> dict:
    from ...champions import champion_legal_actions
    from ...champions_m4 import M4Side, M4State
    from ...mechanics import initial_state
    from ...positional import WARD
    c = cfg()
    lane0 = initial_state(c)
    union, sizes = set(), {}
    for absent in range(0, 6):
        for pos in (-2, -1, 0, 1, 2, WARD):
            for siege in (0, 8):
                side = M4Side(pos, 600, absent)
                legal = champion_legal_actions(lane0._replace(asg=siege), M4State(side, side), 0, c)
                union |= set(legal)
                if absent == 0:
                    sizes[f"{'W' if pos == WARD else pos}{' siege' if siege else ''}"] = len(legal)
    return {"ever_legal": sorted(union), "n_ever_legal": len(union), "size_by_position": sizes}


def _rule(pos: int, absent: int, siege: int) -> set:
    """Reference §3.3."""
    from ...config import (B_HOLD, B_SAFE, B_TRADE, B_YIELD, CHAMP_ABSENT, CHAMP_WARD, F, PRESSURE, SH, SP,
                           W_SLOWPUSH, champ_act)
    from ...positional import WARD, dist
    if absent > 0:
        return {CHAMP_ABSENT}
    waves = (SP, SH, F, W_SLOWPUSH)
    out = {champ_act(w, b) for w in waves for b in (B_HOLD, B_YIELD, B_TRADE, B_SAFE)}
    if siege > 0 and pos != WARD and dist(pos, 2) <= 2:
        out |= {champ_act(w, b, PRESSURE) for w in waves for b in (B_HOLD, B_TRADE)}
    if dist(pos, WARD) <= 2:
        out.add(CHAMP_WARD)
    if pos != WARD and pos <= -1:
        out.add(champ_act(SP, 4))
    return out


def _chunk(seeds: List[int]) -> dict:
    from ...champions_m4 import context, resolve_turn_m4
    from ...config import CHAMP_ABSENT, CHAMP_WARD, F, PRESSURE, SH, SP, W_SLOWPUSH, champ_act, champ_behavior, \
        champ_objective
    from ...env.core import LaneEnv
    from ...env.observation_m4 import decode_m4, m4_feature_names
    from ...positional import WARD
    c = cfg()
    ctx = context(c)
    t = ctx.t
    absent_ix = m4_feature_names(c).index("own_champ_absent")
    names = {0: "Hold", 1: "Yield", 2: "Trade", 3: "Safe"}
    recall = champ_act(SP, 4)
    env = LaneEnv(c, max_turns=TURNS)
    out = collections.defaultdict(collections.Counter)
    seen, lo, hi = set(), 1.0, 0.0
    for seed in seeds:
        rng = np.random.default_rng([ACTION_SEED, seed])
        obs = env.reset(seed=seed)
        done = False
        while not done:
            lane, champ, k = env.state, env.champ, env.k
            codes = [int(rng.choice(obs[0].legal)), int(rng.choice(obs[1].legal))]
            for s in (0, 1):
                me, o = champ.side(s), obs[s]
                legal = set(o.legal)
                seen |= legal
                out["sizes"][len(legal)] += 1
                out["count"]["seat_turns"] += 1
                out["count"]["mismatch"] += int(legal != _rule(me.pos, me.absent, lane.siege_of(s)))
                v = o.vector
                lo, hi = min(lo, float(v.min())), max(hi, float(v.max()))
                phase, dl, dm, _ = decode_m4(v, c)
                want_l, want_m = (lane, champ) if s == 0 else (lane.mirror(), champ.mirror())
                out["count"]["decode_bad"] += int(not (phase == k % 4 and dl == want_l._replace(ap=0, bp=0)
                                                       and dm == want_m))
                out["count"]["absent_entry_bad"] += int(abs(float(v[absent_ix]) - me.absent / c.champion_absent_max)
                                                        > 1e-6)
                out["count"]["at_W"] += int(me.pos == WARD)
                for side in (lane.side(s)[2],):
                    if side:
                        out["count"]["max_rears"] = max(out["count"]["max_rears"],
                                                        max(collections.Counter(side).values()))
                a = codes[s]
                if a in (CHAMP_ABSENT, recall, CHAMP_WARD):
                    continue
                beh, obj = champ_behavior(a), champ_objective(a)
                ends = set()
                for w in (SP, SH, F, W_SLOWPUSH):
                    pair = list(codes)
                    pair[s] = champ_act(w, beh, obj)
                    ends.add(resolve_turn_m4(lane, champ, k, pair[0], pair[1], c, ctx=ctx).end[s])
                moved = int(len(ends) > 1)
                key = names[beh] + (" + Pressure" if obj == PRESSURE else "")
                for k2 in ("all", key, f"turns {50 * (k // 50)}-{50 * (k // 50) + 49}"):
                    out["lane_actions"][k2] += 1
                    out["end_moved"][k2] += moved
            r = env.step(*codes)
            for s in (0, 1):
                a0, a1 = champ.side(s), env.champ.side(s)
                if a1.pos == WARD and a0.pos != WARD:
                    out["W_arrivals"][str(codes[s])] += 1
                if a0.absent == 0 and a1.absent > 0:
                    if codes[s] == recall:
                        out["recall_absence"][str(a1.absent)] += 1
                    else:
                        L0 = t.level_by_xp[min(a0.xp, t.xp_cap)]
                        band = "1-6" if L0 <= 6 else ("7-12" if L0 <= 12 else "13-18")
                        out["death_absence"][f"{band}: {a1.absent}"] += 1
            obs, done = r.obs, r.done
    out = {k: dict(v) for k, v in out.items()}
    out["seen"] = sorted(seen)
    out["range"] = [lo, hi]
    return out


def random_play(workers: int) -> dict:
    from ..m45b.common import run_jobs
    seeds = list(range(SEED0, SEED0 + N_GAMES))
    parts = run_jobs(_chunk, [seeds[i:i + CHUNK] for i in range(0, N_GAMES, CHUNK)], workers, "random play")
    tot: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    seen, lo, hi = set(), 1.0, 0.0
    for p in parts:
        seen |= set(p.pop("seen"))
        r = p.pop("range")
        lo, hi = min(lo, r[0]), max(hi, r[1])
        for k, v in p.items():
            for kk, x in v.items():
                if kk == "max_rears":
                    tot[k][kk] = max(tot[k][kk], x)
                else:
                    tot[k][kk] += x
    moved = {k: {"n": n, "share": tot["end_moved"][k] / n} for k, n in tot["lane_actions"].items()}
    return {"games": N_GAMES, "turns": TURNS, "seeds": [SEED0, SEED0 + N_GAMES - 1], "action_seed": ACTION_SEED,
            "seat_turns": tot["count"]["seat_turns"], "rule_mismatches": tot["count"]["mismatch"],
            "codes_seen": sorted(seen), "sizes": dict(sorted(tot["sizes"].items())),
            "end_cell_depends_on_wave": moved, "observation": {
                "range": [lo, hi], "decode_mismatches": tot["count"]["decode_bad"],
                "absent_entry_mismatches": tot["count"]["absent_entry_bad"], "at_W": tot["count"]["at_W"],
                "max_rears_in_a_cell": tot["count"]["max_rears"]},
            "W_arrivals_by_code": dict(tot["W_arrivals"]), "recall_absence": dict(tot["recall_absence"]),
            "death_absence": dict(sorted(tot["death_absence"].items()))}


def level_timing() -> dict:
    import statistics
    from ...champions_m4 import level
    from ...config import B_HOLD, SP, GameConfig, champ_act
    from ...env.core import LaneEnv
    from ...progression import tables
    lh = champ_act(SP, B_HOLD)
    out = {}
    for name, c in (("m45_calibrated", cfg()), ("m4_lasthit", GameConfig.m4_lasthit())):
        t = tables(c)
        env = LaneEnv(c)
        res = {L: [] for L in LEVELS}
        for seed in range(LEVEL_SEEDS):
            obs = env.reset(seed=seed)
            first = {}
            for n in range(1, LEVEL_TURNS + 1):
                r = env.step(*[lh if lh in o.legal else o.legal[0] for o in obs])
                obs = r.obs
                for s, sd in enumerate((env.champ.a, env.champ.b)):
                    for L in LEVELS:
                        if (s, L) not in first and level(sd, t) >= L:
                            first[(s, L)] = n
            for L in LEVELS:
                res[L] += [first[(s, L)] for s in (0, 1) if (s, L) in first]
        out[name] = {str(L): statistics.median(v) for L, v in res.items()}
    return {"seeds": LEVEL_SEEDS, "turns": LEVEL_TURNS, "median_turn": out}


def run(workers: int) -> dict:
    return {"exhaustive": exhaustive(), "random_play": random_play(workers), "level_timing": level_timing()}


__all__ = ["run", "exhaustive", "random_play", "level_timing"]
