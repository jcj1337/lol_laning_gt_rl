"""M4.5a movement correction (champion_movement_rule "tick"; docs/m4/m4_5a_unified_mechanics_report.md §10):
deterministic scenario demonstrations and paired comparisons of the superseded revision `m45_unified_r1()` (r1) with
the corrected `m45_unified()` (r2). Descriptive checks of the correction, not the M4.5b calibration.

    scenarios   the correction brief's scenarios as hand-built one-turn states, each resolved under three rules:
                "end cell" (r1 without its stationary exemption: champions at their end cells for both ticks and
                designation at the start of each tick, the false positive the exemption hid), r1 and r2; every
                scenario is also resolved mirrored (A and B exchanged) and must give the mirrored result under r2
    paired      along r2 trajectories (uniform random legal play, and the smoke study's scripted pairings), every turn
                is also resolved under r1 and "end cell" from the same state with the same actions and chance:
                positional and retaliation ticks per champion-turn by the victim's movement class (follow = Hold, an
                unthreatened Yield or Safe; chase = Trade; evade = a threatened Yield / Safe; ward; objective =
                Pressure; recall = an interrupted Recall), whether it moved this turn, and, for a follower hit by
                positional aggro under r2, whether the probe (the lane prediction it walks with) also had it ahead of
                its wave on that tick (a genuine exposure) or covered it (a probe mismatch: the authoritative wave fell
                behind the predicted one)
    throughput  turns per second of uniform play, r1 versus r2

Seeds: SeedSequence roots 86,000-86,999 (disjoint from every earlier study, smoke.py uses 85,000-85,999).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ...champions import N_CHANCE, U_TRADE_HIT, mirror_chance
from ...champions_m4 import M4State, context, initial_m4, resolve_turn_m4
from ...config import (B_ABSENT, B_HOLD, B_RECALL, B_SAFE, B_TRADE, B_YIELD, CHAMP_WARD, NONE, PRESSURE, SP,
                       GameConfig, champ_act)
from ...env import LaneEnv
from ...mechanics import step_turn
from ...state import State

SEED = 86_000
A, B = 0, 1
HOLD, TRADE, SAFE, YIELD = (champ_act(SP, b) for b in (B_HOLD, B_TRADE, B_SAFE, B_YIELD))
PRESS, RECALL = champ_act(SP, B_HOLD, PRESSURE), champ_act(SP, B_RECALL)
RULES = ("end cell", "r1", "r2")


def _lane(ax, ah, bx, bh, asg=0) -> State:
    return State(ax, ah, (), asg, bx, bh, (), 0, 0, 0, 0, 0)


def _chance(hit=()) -> Tuple[float, ...]:
    """Exactly the trades of the sides in `hit` land (1 base HP); no gank, no tower shot."""
    u = [0.999] * N_CHANCE
    for s in hit:
        u[U_TRADE_HIT[s]] = 0.0
    return tuple(u)


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    lane: State
    pos: Tuple[int, int]
    actions: Tuple[int, int]
    hit: Tuple[int, ...] = ()


# own coordinates per side; the B entries of `lane` are B's own cells (B's front at its -1 is A's +1)
SCENARIOS: Tuple[Scenario, ...] = (
    Scenario("A1", "Hold follows its winning wave (the brief's example): A and its 16-HP front at 0, B's 2-HP front "
             "at A's +1, which dies on tick 0", _lane(0, 16, -1, 2), (0, -2), (HOLD, HOLD)),
    Scenario("A2", "Hold follows its winning wave from behind (A and its 16-HP front at -1, B's 2-HP front at 0)",
             _lane(-1, 16, 0, 2), (-1, -2), (HOLD, HOLD)),
    Scenario("A3", "Safe follows its winning wave (A and its 16-HP front at -2, B's 2-HP front at A's -1)",
             _lane(-2, 16, 1, 2), (-2, -2), (SAFE, HOLD)),
    Scenario("A4", "Yield (stand-off) in the same lane: it stays one cell behind B's starting front",
             _lane(-2, 16, 1, 2), (-2, -2), (YIELD, HOLD)),
    Scenario("B1", "Deliberate advance: A at 0 (its front at -1) Pressures through B's front at A's +1, which waits "
             "while A's crashed mass is under B's tower", _lane(-1, 8, -1, 8, asg=8), (0, -2), (PRESS, HOLD)),
    Scenario("B2", "The brief's B with walking waves: A at 0 (its front at -1) Trades toward B at its tower while "
             "B's front walks from A's +1 to 0: they swap cells within tick 0 (no shared cell at a tick boundary)",
             _lane(-1, 8, -1, 8), (0, -2), (TRADE, TRADE)),
    Scenario("C1", "Trade chase into B's 16-HP wave (B stands behind it), connected, missed",
             _lane(-2, 8, 1, 16), (-2, -2), (TRADE, HOLD)),
    Scenario("C2", "The same chase, landed: positional aggro, then retaliation", _lane(-2, 8, 1, 16), (-2, -2),
             (TRADE, HOLD), hit=(A,)),
    Scenario("C3", "Trade chase into B's wave; B evades out of contact (the Trade does not connect)",
             _lane(-1, 8, 0, 8), (-1, -1), (TRADE, SAFE)),
    Scenario("D1", "A Holds at B's tower cell far ahead of its wave and walks back: in B's oncoming front's cell "
             "after tick 0, covered by its own wave at the lane cell after tick 1", _lane(-2, 8, -2, 8), (2, -2),
             (HOLD, HOLD)),
    Scenario("D2", "Cover on the same tick: A stands at 0 and both fronts walk into its cell on tick 0",
             _lane(-1, 8, -1, 8), (0, -2), (TRADE, SAFE)),
    Scenario("D3", "No cover: as D2 with A's front far behind (at -3); B's front walks into A's cell on tick 0",
             _lane(-3, 8, -1, 8), (0, -2), (TRADE, SAFE)),
    Scenario("W1", "Ward from +1 walks through B's front at the lane cell (A's front at -1)", _lane(-1, 8, 0, 8),
             (1, -2), (CHAMP_WARD, HOLD)),
    Scenario("S1", "Standing still in B's wave ahead of the own front (unchanged by the correction)",
             _lane(-1, 8, 0, 8), (0, -2), (TRADE, SAFE)),
    Scenario("N1", "Adjacent to B's waiting front, never in its cell", _lane(-2, 8, -1, 8, asg=8), (0, -1),
             (TRADE, TRADE)),
    Scenario("R1", "B dives A's Recall through A's 16-HP front at A's -1 and lands: positional aggro on tick 0, "
             "retaliation from the end cell on tick 1", _lane(-1, 16, 0, 8), (-2, 0), (RECALL, TRADE), hit=(B,)),
)


def _champs(cfg: GameConfig, pa: int, pb: int, level: int = 2) -> M4State:
    t = context(cfg).t
    m = initial_m4(cfg)
    xp, hp = t.xp_cum[level], t.hp_max[level]
    return M4State(m.a._replace(pos=pa, xp=xp, hp=hp), m.b._replace(pos=pb, xp=xp, hp=hp))


def _lane_step(cfg: GameConfig, rule: str, calls: Optional[list] = None, record: bool = False):
    """A resolver lane step for `rule` ("end cell" forces every MinionTarget stationary under r1); `calls` collects
    the TurnResults (the probe first, then the authoritative step)."""
    def ls(s, kk, x, y, reach, budget=None, targets=None):
        if rule == "end cell" and targets is not None:
            targets = tuple(None if g is None else g._replace(stationary=True) for g in targets)
        r = step_turn(s, kk, x, y, cfg, record=record, champion_reach=reach, champion_budget=budget,
                      champion_targets=targets)
        if calls is not None:
            calls.append(r)
        return r
    return ls


CFGS = {"end cell": GameConfig.m45_unified_r1(), "r1": GameConfig.m45_unified_r1(), "r2": GameConfig.m45_unified()}


def resolve(rule: str, lane: State, m4: M4State, k: int, a: int, b: int, u, calls=None, record=False):
    cfg = CFGS[rule]
    return resolve_turn_m4(lane, m4, k, a, b, cfg, lane_step=_lane_step(cfg, rule, calls, record), u=u)


def scenarios(k: int = 10) -> dict:
    out = []
    for sc in SCENARIOS:
        m4 = _champs(CFGS["r2"], *sc.pos)
        u = _chance(sc.hit)
        row = {"key": sc.key, "label": sc.label, "lane": list(sc.lane), "pos": list(sc.pos),
               "actions": [CFGS["r2"].label(a) for a in sc.actions], "hit": list(sc.hit)}
        for rule in RULES:
            t = resolve(rule, sc.lane, m4, k, *sc.actions, u, record=True)
            row[rule] = {"fire": [list(f) for f in t.minion_fire], "units": list(t.minion_hits),
                         "centi": list(t.minion_damage), "end": list(t.end), "connected": list(t.connected),
                         "retaliation": list(t.retaliation), "path": [list(p) for p in t.path] if t.path else None,
                         "fronts": [[tr["A"][0][0] if tr["A"] else None, tr["B"][0][0] if tr["B"] else None]
                                    for tr in t.lane.trace]}
        t = resolve("r2", sc.lane, m4, k, *sc.actions, u)
        tm = resolve("r2", sc.lane.mirror(), m4.mirror(), k, sc.actions[1], sc.actions[0], mirror_chance(list(u)))
        row["mirror_ok"] = bool(tm.minion_fire == t.minion_fire[::-1] and tm.path == t.path[::-1]
                                and tm.champ == t.champ.mirror() and tm.lane.state == t.lane.state.mirror()
                                and tm.reward_units == -t.reward_units)
        out.append(row)
    return {"k": k, "level": 2, "rows": out}


# ---------------------------------------------------------------------------------------------------- paired study
CLASSES = ("follow", "chase", "evade", "ward", "objective", "recall")


def movement_class(t, s: int) -> Optional[str]:
    """The victim's movement this turn (None: away, never a minion target)."""
    if t.away[s]:
        return None
    if t.behaviors[s] == B_RECALL:
        return "recall"
    if t.warding[s]:
        return "ward"
    if t.objectives[s] != NONE:
        return "objective"
    if t.behaviors[s] == B_TRADE and t.behaviors[1 - s] != B_ABSENT:
        return "chase"
    if t.threatened[s]:
        return "evade"
    return "follow"


class _Tally:
    def __init__(self):
        self.c = {cl: {"turns": 0, "moved": 0, **{f"{k}_{r}": 0 for r in RULES for k in ("pos", "ret", "centi")},
                       "r2_pos_moved": 0, "r2_pos_genuine": 0, "r2_pos_mismatch": 0} for cl in CLASSES}
        self.turns = 0
        self.died = {r: 0 for r in RULES}

    def add(self, lane: State, m4: M4State, k: int, a: int, b: int, u, ref=None) -> None:
        res, calls = {}, []
        for rule in RULES:
            res[rule] = resolve(rule, lane, m4, k, a, b, u, calls=calls if rule == "r2" else None)
        r2 = res["r2"]
        if ref is not None:
            assert r2.champ == ref, "the paired r2 resolution must equal the environment's"
        probe = calls[0]
        self.turns += 1
        for rule in RULES:
            self.died[rule] += sum(res[rule].died)
        for s in (A, B):
            cl = movement_class(r2, s)
            if cl is None:
                continue
            d = self.c[cl]
            d["turns"] += 1
            moved = r2.start[s] != r2.end[s]
            d["moved"] += moved
            for rule in RULES:
                f = res[rule].minion_fire[s]
                d[f"pos_{rule}"] += sum(x == "positional" for x in f)
                d[f"ret_{rule}"] += sum(x == "retaliation" for x in f)
                d[f"centi_{rule}"] += res[rule].minion_damage[s]
            for tk, x in enumerate(r2.minion_fire[s]):
                if x != "positional":
                    continue
                d["r2_pos_moved"] += moved
                if cl == "follow":
                    pf = probe.fronts_by_tick[tk][s]
                    covered = pf is not None and pf >= r2.path[s][tk]
                    d["r2_pos_mismatch" if covered else "r2_pos_genuine"] += 1

    def summary(self) -> dict:
        rows = {}
        for cl, d in self.c.items():
            n = max(1, d["turns"])
            rows[cl] = {"champion_turns": d["turns"], "moved_share": d["moved"] / n,
                        **{f"{k}_per_turn_{r}": d[f"{k}_{r}"] / n for r in RULES for k in ("pos", "ret")},
                        **{f"centi_per_turn_{r}": d[f"centi_{r}"] / n for r in RULES},
                        "r2_pos_ticks": d["pos_r2"], "r2_pos_ticks_moved": d["r2_pos_moved"],
                        "r2_follow_genuine": d["r2_pos_genuine"], "r2_follow_probe_mismatch": d["r2_pos_mismatch"]}
        tot = {f"{k}_{r}": sum(d[f"{k}_{r}"] for d in self.c.values()) for r in RULES for k in ("pos", "ret", "centi")}
        n = 2 * self.turns
        return {"turns_resolved": self.turns, "by_class": rows,
                "per_champion_turn": {k: v / n for k, v in tot.items()},
                "deaths_per_champion_turn": {r: v / n for r, v in self.died.items()}}


def paired_random(games: int = 40, turns: int = 300) -> dict:
    cfg = CFGS["r2"]
    env = LaneEnv(cfg, max_turns=turns)
    rng = np.random.default_rng(SEED)
    tally = _Tally()
    for g in range(games):
        env.reset(seed=SEED + g)
        while not env.done:
            lane, m4, k = env.state, env.champ, env.k
            a, b = int(rng.choice(env.legal(0))), int(rng.choice(env.legal(1)))
            r = env.step(a, b)
            tally.add(lane, m4, k, a, b, tuple(r.info["chance"]), ref=env.champ)
    return {"games": games, "turns": turns, **tally.summary()}


def paired_scripted(seeds: int = 8, turns: int = 200) -> dict:
    from ...viewer import policies as pl
    from ...viewer import session as ss
    from .smoke import PAIRINGS
    out = {}
    for pa, pb, scen, gap in PAIRINGS:
        tally = _Tally()
        for i in range(seeds):
            rec = ss.play(ss.Setup("m45_unified", scen, gap, SEED + 500 + i),
                          policies=(pl.POLICIES[pa], pl.POLICIES[pb]), turns=turns)
            for j, t in enumerate(rec.turns):
                a, b = rec.actions[j]
                tally.add(rec.lanes[j], rec.champs[j], rec.k0 + j, a, b, tuple(t.info["chance"]),
                          ref=rec.champs[j + 1])
        out[f"{pa} v {pb} [{scen}, {gap}]"] = tally.summary()
    return {"seeds": seeds, "turns": turns, "pairings": out}


def throughput(turns: int = 3000, repeats: int = 3) -> dict:
    out = {}
    for name in ("m45_unified_r1", "m45_unified"):
        cfg = getattr(GameConfig, name)()
        best = 0.0
        for _ in range(repeats):
            env = LaneEnv(cfg, max_turns=turns)
            env.reset(seed=SEED)
            rng = np.random.default_rng(SEED)
            t0 = time.perf_counter()
            while not env.done:
                env.step(int(rng.choice(env.legal(0))), int(rng.choice(env.legal(1))))
            best = max(best, turns / (time.perf_counter() - t0))
        out[name] = best
    return {"turns_per_second": out, "ratio": out["m45_unified"] / out["m45_unified_r1"], "repeats": repeats}


def run() -> dict:
    t0 = time.time()
    out = {"scenarios": scenarios(), "random": paired_random(), "scripted": paired_scripted(),
           "throughput": throughput()}
    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


# ---------------------------------------------------------------------------------------------------- report
def _fire(f) -> str:
    return " / ".join("·" if not x else ("pos" if x == "positional" else "ret") for x in f)


def report(d: dict) -> str:
    L = ["# M4.5a movement correction: demonstrations and paired comparison (generated)", "",
         "`python -m lanegame.experiments.m45a movement`; design and discussion: "
         "`docs/m4/m4_5a_unified_mechanics_report.md` §10. r1 = `m45_unified_r1()` (champions at their end cells for "
         "the whole turn, positional aggro only on a champion that did not move); r2 = `m45_unified()` "
         "(champion_movement_rule \"tick\"); \"end cell\" = r1 without its stationary exemption (the original false "
         "positive).", ""]
    sc = d["scenarios"]
    L += [f"## Scenarios (one turn, level {sc['level']}, k = {sc['k']}; fire per tick for champion A / B: pos = "
          "positional, ret = retaliation, · = none)", "",
          "| | scenario | actions | end cell | r1 | r2 | r2 path A ; B | mirror |", "|---|---|---|---|---|---|---|---|"]
    for r in sc["rows"]:
        cells = [f"A {_fire(r[x]['fire'][0])}; B {_fire(r[x]['fire'][1])}" for x in RULES]
        p = r["r2"]["path"]
        hit = f" (hit {'AB'[r['hit'][0]]})" if r["hit"] else ""
        L.append(f"| {r['key']} | {r['label']} | {' v '.join(r['actions'])}{hit} | " + " | ".join(cells)
                 + f" | {p[0]} ; {p[1]} | {'ok' if r['mirror_ok'] else 'FAIL'} |")
    L.append("")
    for name, block in (("Uniform random legal play", d["random"]),):
        L += _paired_table(f"{name} ({block['games']} games x {block['turns']} turns, r2 trajectories)", block)
    sp = d["scripted"]
    L += [f"## Scripted pairings ({sp['seeds']} seeds x {sp['turns']} turns, r2 trajectories)", "",
          "| pairing | positional ticks / champion-turn: end cell / r1 / r2 | retaliation r1 / r2 | "
          "minion cHP r1 / r2 | deaths r1 / r2 | r2 follower hits: genuine / probe mismatch |",
          "|---|---|---|---|---|---|"]
    for name, b in sp["pairings"].items():
        pc = b["per_champion_turn"]
        fol = b["by_class"]["follow"]
        L.append(f"| {name} | {pc['pos_end cell']:.4f} / {pc['pos_r1']:.4f} / {pc['pos_r2']:.4f} | "
                 f"{pc['ret_r1']:.4f} / {pc['ret_r2']:.4f} | {pc['centi_r1']:.2f} / {pc['centi_r2']:.2f} | "
                 f"{b['deaths_per_champion_turn']['r1']:.4f} / {b['deaths_per_champion_turn']['r2']:.4f} | "
                 f"{fol['r2_follow_genuine']} / {fol['r2_follow_probe_mismatch']} |")
    L.append("")
    tp = d["throughput"]
    L += ["## Throughput (uniform play, one process, best of {})".format(tp["repeats"]), "",
          "| preset | turns / s |", "|---|---|"] + [f"| `{k}` | {v:,.0f} |" for k, v in tp["turns_per_second"].items()]
    L += ["", f"r2 / r1 = {tp['ratio']:.3f}", "", f"Elapsed: {d.get('elapsed_s', 0)} s.", ""]
    return "\n".join(L)


def _paired_table(title: str, b: dict) -> List[str]:
    L = [f"## {title}", "", "Per champion-turn of each victim movement class (paired: the same state, actions and "
         "chance under each rule).", "",
         "| class | champion-turns | moved | positional: end cell / r1 / r2 | retaliation: r1 / r2 | "
         "minion cHP: r1 / r2 | r2 positional ticks on a moved champion | r2 follower hits: genuine / probe mismatch |",
         "|---|---|---|---|---|---|---|---|"]
    for cl, r in b["by_class"].items():
        L.append(f"| {cl} | {r['champion_turns']} | {r['moved_share']:.2f} | {r['pos_per_turn_end cell']:.4f} / "
                 f"{r['pos_per_turn_r1']:.4f} / {r['pos_per_turn_r2']:.4f} | {r['ret_per_turn_r1']:.4f} / "
                 f"{r['ret_per_turn_r2']:.4f} | {r['centi_per_turn_r1']:.2f} / {r['centi_per_turn_r2']:.2f} | "
                 f"{r['r2_pos_ticks_moved']} of {r['r2_pos_ticks']} | "
                 + (f"{r['r2_follow_genuine']} / {r['r2_follow_probe_mismatch']}" if cl == "follow" else "—") + " |")
    pc = b["per_champion_turn"]
    L += ["", f"All classes: positional ticks per champion-turn {pc['pos_end cell']:.4f} (end cell) / "
          f"{pc['pos_r1']:.4f} (r1) / {pc['pos_r2']:.4f} (r2); retaliation {pc['ret_r1']:.4f} / {pc['ret_r2']:.4f}; "
          f"deaths per champion-turn {b['deaths_per_champion_turn']['r1']:.4f} (r1) / "
          f"{b['deaths_per_champion_turn']['r2']:.4f} (r2).", ""]
    return L


__all__ = ["SCENARIOS", "Scenario", "scenarios", "paired_random", "paired_scripted", "throughput", "run", "report",
           "movement_class", "resolve"]
