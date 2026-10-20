"""Measurements of learned play on `m45_calibrated()` with the M5 self-play policy `sp:0` (common.POLICY).

  selfplay   `sp:0` against itself from s0, N_GAMES games of GAME_TURNS turns on PLAY seeds. Per seat-game: level,
             earned, invested gold, M(S) and tower damage dealt after CHECKPOINTS turns; the turns at which the seat
             reaches levels 14 and 18 and 48 invested resources and breaks each plate; deaths, Recalls, wards placed,
             turns at W; the split of earned gold into farm, plate gold and kills; the share of tower damage that
             Pressure supplies (turns before the tower is exhausted). In the lane phase (turns 0-93) and over the whole
             game: where the farm and XP of the seat's kill ledger go (collected; lost while away, channelling or
             warding, out of reach, or to the collection rate, by mode) and the counts of connected trades, dives,
             tower-adjacent trades, tower shots taken, ganks and their damage, interrupted Recalls, Safe and retreat
             turns, turns with minion fire taken and interrupted wards. Over the whole game (`_events`): Recall
             attempts and the HP at them, the circumstances of every death, the wave slots of lane decisions, and the
             natural freeze episodes. Both seats of a self-play game are samples of the same policy, so statistics
             pool them.
  death      the cost of a death and of a Recall. States: present-champion turns of further `sp:0` self-play games in
             three phases (PHASES: early, mid, late), one per game and phase, the focal seat mirrored to A. From each
             state the focal champion's record is replaced, on the SAME rollout streams, by
               alive    unchanged
               recall   "just recalled": at its tower cell, away recall_turns - 1 more turns, HP kept (full on return)
               death    "just died": HP 0, at its tower cell, away `death_turns` of its level (3 / 4 / 5)
             and every variant is played N_ROLL times for WINDOW turns by `sp:0` in both seats (m45b.engine.play,
             paired seeds: state j of every phase uses ROLL seeds j * N_ROLL onward). cost = G(alive) - G(variant) with G the discounted reward to the focal side over the window;
             the cost of a death adds back the kill gold the opponent received on the death turn, which precedes the
             state. Reported with 95% t intervals over states.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .common import GAMMA, PLAY, POLICY, ROLL, cfg

N_GAMES, GAME_TURNS = 32, 1000
CHECKPOINTS = (50, 94, 100, 150, 200, 300, 500, 1000)
LANE_EXIT = 94                     # the lane phase is turns 0-93 (M4.7)
EVENTS = ("connected", "dive", "adjacent", "tower_shot", "ganked", "gank_cHP", "interrupted", "safe", "retreat",
          "minion_turns", "ward_hit", "turns")
PHASES = {"early": (20, 60), "mid": (80, 130), "late": (200, 300)}
N_STATES, N_ROLL, WINDOW = 24, 16, 300
STATE_SEEDS = PLAY[100:100 + N_STATES]


def _policies(k0: int, seed: int, horizon: int):
    from ..rl.m5.players import crn_policy
    return crn_policy(POLICY, k0, seed, 0, horizon), crn_policy(POLICY, k0, seed, 1, horizon)


def _setup():
    import torch
    torch.set_num_threads(1)


def selfplay_game(seed: int) -> dict:
    """One `sp:0` self-play game from s0 (module docstring); per-seat records."""
    from collections import defaultdict
    from ...champions import N_CHANCE, RETREAT
    from ...champions_m4 import CHANNEL, WARD_MODE, context, initial_m4, resolve_turn_m4
    from ...mechanics import initial_state
    from ...positional import WARD, dist
    from ...progression import investment
    from ..m45b.engine import decide, lane_cache
    _setup()
    c = cfg()
    ctx = context(c)
    tb, inv, g = ctx.t, investment(c), c.gold_scale
    ls = lane_cache(c)
    pa, pb = _policies(0, seed, GAME_TURNS)
    lane, m4 = initial_state(c), initial_m4(c)
    rng = np.random.default_rng(seed)
    seats = [{"cp": {}, "L14": None, "L18": None, "S48": None, "plates": [None, None, None], "deaths": 0,
              "recalls": 0, "wards": 0, "ward_placed": 0, "at_W": 0, "farm": 0, "plate": 0, "kill": 0, "earned": 0,
              "tower_hits": 0, "pressure": 0, "flow": {"lane": defaultdict(float), "all": defaultdict(float)}}
             for _ in (0, 1)]
    ev = {"counts": defaultdict(int), "recall_hp": [], "recalled_hp": [], "death_hp": [], "waves": [0, 0, 0, 0],
          "freeze_waves": [0, 0, 0, 0], "freezes": [], "open": [None, None], "prev_wave": [None, None]}
    for i in range(GAME_TURNS):
        u = rng.random(N_CHANCE)
        a, b = decide(pa, c, lane, m4, 0, i), decide(pb, c, lane, m4, 1, i)
        tau0 = (m4.a.tower_dmg, m4.b.tower_dmg)
        t = resolve_turn_m4(lane, m4, i, a, b, c, lane_step=ls, u=u, ctx=ctx)
        _events(ev, i, lane, m4, (a, b), t, c, tb)
        lane, m4 = t.lane.state, t.champ
        n = i + 1
        for s in (0, 1):
            for w in ("lane", "all") if i < LANE_EXIT else ("all",):
                _flow(seats[s]["flow"][w], t, s, c, ctx, dist, RETREAT, CHANNEL, WARD_MODE)
        for s in (0, 1):
            r, me = seats[s], m4.side(s)
            r["farm"] += t.farm_units[s]
            r["plate"] += t.plate_units[s]
            r["kill"] += t.bounty_units[s]
            r["earned"] += t.earned_units[s]
            r["deaths"] += int(t.died[s])
            r["recalls"] += int(t.recalled[s])
            r["wards"] += int(t.warding[s])
            r["ward_placed"] += int(t.ward_placed[s])
            r["at_W"] += int(not me.absent and me.pos == WARD)
            if tau0[s] + t.tower_damage_dealt[s] < c.plate_thresholds[-1]:
                r["tower_hits"] += 100 * t.tower_hits[s]
                r["pressure"] += t.tower_damage_dealt[s] - 100 * t.tower_hits[s]
            L = t.levels_after[s]
            if r["L14"] is None and L >= 14:
                r["L14"] = n
            if r["L18"] is None and L >= 18:
                r["L18"] = n
            if r["S48"] is None and me.items >= 48 * g:
                r["S48"] = n
            for j, th in enumerate(c.plate_thresholds):
                if r["plates"][j] is None and tau0[s] < th <= me.tower_dmg:
                    r["plates"][j] = n
            if n in CHECKPOINTS:
                r["cp"][n] = {"level": L, "earned": r["earned"] / g, "invested": me.items / g,
                              "M": inv.factor(me.items), "tau": me.tower_dmg}
    for r in seats:
        for key in ("farm", "plate", "kill", "earned"):
            r[key] /= g
        r["flow"] = {w: dict(v) for w, v in r["flow"].items()}
    ev["freezes"] += [e for e in ev.pop("open") if e is not None]
    ev.pop("prev_wave")
    ev["counts"] = dict(ev["counts"])
    return {"seed": seed, "seats": seats, "events": ev}


def _events(ev, i: int, lane, m4, codes, t, c, tb) -> None:
    """One turn of both seats: Recall attempts (HP at the start of the turn), completions and interruptions; the
    circumstances of every death; the wave slot of every present lane decision; Pressure and Ward decisions; and the
    freeze episodes (`m44.records.holds_freeze` at the start of a turn, maximal runs per freezer; "pushed" when the
    breaker's previous wave choice was Shove or Slow Push, M4.6d's definition), with the freezer's wave choices."""
    from ...config import CHAMP_WARD, PRESSURE, SH, SP, W_SLOWPUSH, champ_act
    from ..rl.m44.records import holds_freeze
    n = ev["counts"]
    recall = champ_act(SP, 4)
    for s in (0, 1):
        held = holds_freeze(lane, s)
        if held:
            if ev["open"][s] is None:
                ev["open"][s] = {"start": i, "len": 0, "pushed": ev["prev_wave"][1 - s] in (SH, W_SLOWPUSH)}
            ev["open"][s]["len"] += 1
            if not m4.side(s).absent:
                ev["freeze_waves"][t.waves[s]] += 1
        elif ev["open"][s] is not None:
            ev["freezes"].append(ev["open"][s])
            ev["open"][s] = None
    for s, code in enumerate(codes):
        me, o = m4.side(s), 1 - s
        if me.absent:
            continue
        hp = me.hp / tb.hp_max[tb.level_by_xp[min(me.xp, tb.xp_cap)]]
        n["present"] += 1
        if code == recall:
            n["recall_attempts"] += 1
            ev["recall_hp"].append(hp)
            n["recall_interrupted"] += int(t.interrupted[s])
            if t.recalled[s]:
                n["recalls"] += 1
                ev["recalled_hp"].append(hp)
        elif code == CHAMP_WARD:
            n["wards"] += 1
        else:
            ev["waves"][t.waves[s]] += 1
            n["pressure"] += int(t.objectives[s] == PRESSURE)
        if t.died[s]:
            n["deaths"] += 1
            n["deaths_lane_phase"] += int(i < LANE_EXIT)
            n["death_by_trade"] += int(t.trade_damage[o] > 0)
            n["death_by_gank"] += int(t.gank_damage[s] > 0)
            n["death_by_tower"] += int(t.tower_damage[s] > 0)
            n["death_by_minions"] += int(t.minion_damage[s] > 0)
            n["death_on_interrupted_recall"] += int(code == recall and t.interrupted[s])
            n["death_killer_in_tower_cell"] += int(t.end[o] == c.tower)
            n["death_killer_shot"] += int(t.tower_damage[o] > 0)
            ev["death_hp"].append(hp)
    ev["prev_wave"] = [None if m4.side(s).absent else t.waves[s] for s in (0, 1)]


def _flow(acc, t, s, c, ctx, dist, RETREAT, CHANNEL, WARD_MODE) -> None:
    """Add one turn of seat s to `acc`: where the farm (resources) and XP of its kill ledger went, and its events."""
    mode, end = t.modes[s], t.end[s]
    reach = c.retreat_farm_reach if mode == RETREAT else c.farm_reach
    for x, hp in t.lane.kills[s].items():
        f, xp = c.farm_per_hp * hp, c.xp_per_hp * hp
        acc["farm_avail"] += f
        acc["xp_avail"] += xp
        if t.away[s]:
            acc["farm_away"] += f
            acc["xp_away"] += xp
            continue
        if dist(x, end) > c.xp_range:
            acc["xp_range"] += xp
        if mode in (CHANNEL, WARD_MODE):
            acc["farm_mode"] += f
        elif dist(x, end) > reach:
            acc["farm_reach"] += f
        else:
            got = hp * ctx.collection(x, mode, end, t.modes[1 - s], t.end[1 - s]) / c.gold_scale
            acc["farm_rate"] += f - got
            acc["got_" + mode] += got
            acc["avail_" + mode] += f
    acc["farm_got"] += t.farm_units[s] / c.gold_scale
    acc["xp_got"] += t.xp_gain[s]
    acc["xp_disc"] += t.xp_discarded[s]
    acc["turns"] += 1
    acc["connected"] += int(t.connected[s])
    acc["dive"] += int(t.connected[s] and end == c.tower)
    acc["adjacent"] += int(t.connected[s] and end == c.tower - 1)
    acc["tower_shot"] += int(t.tower_damage[s] > 0)
    acc["ganked"] += int(t.ganked[s])
    acc["gank_cHP"] += t.gank_damage[s]
    acc["interrupted"] += int(t.interrupted[s])
    acc["safe"] += int(mode == "safe")
    acc["retreat"] += int(mode == RETREAT)
    acc["minion_turns"] += int(t.minion_hits[s] > 0)
    acc["ward_hit"] += int(t.ward_interrupted[s])


def _in_play(seats: List[dict]) -> dict:
    """Pooled shares and rates of the `_flow` records, per window."""
    out = {}
    for w in ("lane", "all"):
        tot: Dict[str, float] = {}
        for s in seats:
            for k, v in s["flow"][w].items():
                tot[k] = tot.get(k, 0.0) + v
        fa, xa, turns = tot["farm_avail"], tot["xp_avail"], tot["turns"]
        modes = sorted(k[6:] for k in tot if k.startswith("avail_"))
        out[w] = {"seat_turns": turns,
                  "farm": {"collected": tot["farm_got"] / fa, "away": tot.get("farm_away", 0) / fa,
                           "channel_or_ward": tot.get("farm_mode", 0) / fa, "out_of_reach": tot.get("farm_reach", 0) / fa,
                           "rate": tot.get("farm_rate", 0) / fa},
                  "by_mode": {m: {"share": tot["avail_" + m] / fa, "rate": tot["got_" + m] / tot["avail_" + m]}
                              for m in modes},
                  "xp": {"gained": tot["xp_got"] / xa, "away": tot.get("xp_away", 0) / xa,
                         "out_of_range": tot.get("xp_range", 0) / xa, "discarded": tot.get("xp_disc", 0) / xa},
                  "per_1000": {k: 1000.0 * tot.get(k, 0) / turns for k in EVENTS if k not in ("turns", "gank_cHP")},
                  "gank_cHP_mean": tot.get("gank_cHP", 0) / tot["ganked"] if tot.get("ganked") else None}
    return out


def selfplay(workers: int) -> dict:
    from ..m45b.common import run_jobs
    games = run_jobs(selfplay_game, list(PLAY[:N_GAMES]), workers, "selfplay")
    seats = [s for gm in games for s in gm["seats"]]

    def med(vals):
        """Median, quartiles, 10th / 90th percentiles and range over the events that happened (n of `of`)."""
        v = [x for x in vals if x is not None]
        if not v:
            return {"median": None, "n": 0, "of": len(vals)}
        p = np.percentile(v, [10, 25, 50, 75, 90])
        return {"median": float(p[2]), "q25": float(p[1]), "q75": float(p[3]), "p10": float(p[0]),
                "p90": float(p[4]), "min": float(min(v)), "max": float(max(v)), "n": len(v), "of": len(vals)}

    def game_turn(j: int, pick) -> List:
        return [pick(gm["seats"][0]["plates"][j] or 10 ** 9, gm["seats"][1]["plates"][j] or 10 ** 9) for gm in games]

    cps = {n: {k: float(np.mean([s["cp"][n][k] for s in seats])) for k in ("level", "earned", "invested", "M", "tau")}
           for n in CHECKPOINTS}
    for n in CHECKPOINTS:
        cps[n]["exhausted_share"] = float(np.mean([s["cp"][n]["tau"] >= cfg().plate_thresholds[-1] for s in seats]))
    both18 = [max(gm["seats"][0]["L18"] or 10 ** 9, gm["seats"][1]["L18"] or 10 ** 9) for gm in games]
    per1000 = 1000.0 / GAME_TURNS
    earned = sum(s["earned"] for s in seats)
    hits, press = sum(s["tower_hits"] for s in seats), sum(s["pressure"] for s in seats)
    share = [s["pressure"] / (s["pressure"] + s["tower_hits"]) for s in seats if s["pressure"] + s["tower_hits"]]
    prev, income = (0, 0.0), {}
    for n in CHECKPOINTS:
        income[n] = (cps[n]["earned"] - prev[1]) / (n - prev[0])
        prev = (n, cps[n]["earned"])
    return {"n_games": N_GAMES, "turns": GAME_TURNS, "seeds": [PLAY[0], PLAY[N_GAMES - 1]], "checkpoints": cps,
            "L14": med([s["L14"] for s in seats]), "L18": med([s["L18"] for s in seats]),
            "both_L18": med([x if x < 10 ** 9 else None for x in both18]), "S48": med([s["S48"] for s in seats]),
            "plates": [med([s["plates"][j] for s in seats]) for j in range(3)],
            "first_tower_exhausted": med([x if x < 10 ** 9 else None for x in game_turn(2, min)]),
            "both_towers_exhausted": med([x if x < 10 ** 9 else None for x in game_turn(2, max)]),
            "per_1000": {k: float(np.mean([s[k] for s in seats])) * per1000
                         for k in ("deaths", "recalls", "wards", "ward_placed", "at_W")},
            "earned_split": {k: sum(s[k] for s in seats) / earned for k in ("farm", "plate", "kill")},
            "earned_per_game": earned / len(seats), "income_per_turn": income,
            "pressure_share": press / (press + hits) if press + hits else None,
            "pressure_share_per_seat": med(share), "minion_tower_damage_per_seat": hits / len(seats),
            "in_play": _in_play(seats), "decisions": _decisions(games, len(seats) * GAME_TURNS), "games": games}


def _decisions(games: List[dict], seat_turns: int) -> dict:
    """Pooled `_events` records: Recalls, deaths, decision shares and natural freezes."""
    n: Dict[str, float] = {}
    lists = {"recall_hp": [], "recalled_hp": [], "death_hp": []}
    waves, fwaves, freezes = np.zeros(4), np.zeros(4), []
    for gm in games:
        ev = gm["events"]
        for k, v in ev["counts"].items():
            n[k] = n.get(k, 0) + v
        for k in lists:
            lists[k] += ev[k]
        waves += ev["waves"]
        fwaves += ev["freeze_waves"]
        freezes += ev["freezes"]
    deaths, lane_dec = n.get("deaths", 0), waves.sum()
    lengths = np.array([e["len"] for e in freezes]) if freezes else np.zeros(0)

    def q(v):
        return {"median": float(np.median(v)), "q25": float(np.quantile(v, 0.25)), "q75": float(np.quantile(v, 0.75))}

    names = ("last_hit", "shove", "thin", "slow_push")      # slots 0 (SP), 1 (SH), 2 (F), 3 (W_SLOWPUSH)
    return {"per_1000": {k: 1000.0 * n.get(k, 0) / seat_turns for k in ("recall_attempts", "recalls",
                                                                        "recall_interrupted", "deaths")},
            "recall_interrupted_share": n.get("recall_interrupted", 0) / n["recall_attempts"],
            "recall_attempt_hp": q(lists["recall_hp"]),
            "recalled_hp": {**q(lists["recalled_hp"]),
                            "share_at_least_0.9": float(np.mean(np.array(lists["recalled_hp"]) >= 0.9))},
            "deaths_lane_phase_per_seat_game": n.get("deaths_lane_phase", 0) / (2 * len(games)),
            "death_hp": q(lists["death_hp"]),
            "death_share": {k[6:]: n.get(k, 0) / deaths for k in ("death_by_trade", "death_by_gank", "death_by_tower",
                                                                    "death_by_minions", "death_on_interrupted_recall",
                                                                    "death_killer_in_tower_cell", "death_killer_shot")},
            "wave_share": {nm: float(waves[j] / lane_dec) for j, nm in enumerate(names)},
            "shove_share_of_present": float(waves[1] / n["present"]),
            "pressure_share_of_lane_decisions": n.get("pressure", 0) / lane_dec,
            "ward_share_of_present": n.get("wards", 0) / n["present"],
            "recall_share_of_present": n.get("recall_attempts", 0) / n["present"],
            "freezes": {"episodes_per_100_seat_turns": 100.0 * len(freezes) / seat_turns,
                        "mean_length": float(lengths.mean()) if len(lengths) else None,
                        "single_turn_share": float(np.mean(lengths == 1)) if len(lengths) else None,
                        "four_or_more_share": float(np.mean(lengths >= 4)) if len(lengths) else None,
                        "pushed_in_share": float(np.mean([e["pushed"] for e in freezes])) if freezes else None,
                        "freezer_wave_share": {nm: float(fwaves[j] / fwaves.sum()) for j, nm in enumerate(names)}}}


# ---------------------------------------------------------------------------------------------- death and Recall
def _sample_states(seed: int) -> Dict[str, dict]:
    """One present-champion state per phase from an `sp:0` self-play game (both champions present, a seeded choice
    of turn and seat), mirrored so that the focal seat is A."""
    from ...champions_m4 import initial_m4
    from ...mechanics import initial_state
    from ..m45b.engine import play
    c = cfg()
    pa, pb = _policies(0, seed, max(hi for _, hi in PHASES.values()) + 1)
    g = play(c, (initial_state(c), initial_m4(c), 0), pa, pb, seed, max(hi for _, hi in PHASES.values()) + 1,
             keep_states=True)
    rng = np.random.default_rng([seed, 11])
    out = {}
    for phase, (lo, hi) in PHASES.items():
        cand = [(i, s) for i, (lane, m4, k) in enumerate(g["states"]) if lo <= i <= hi for s in (0, 1)
                if not m4.a.absent and not m4.b.absent]
        i, s = cand[int(rng.integers(len(cand)))]
        lane, m4, k = g["states"][i]
        if s == 1:
            lane, m4 = lane.mirror(), m4.mirror()
        out[phase] = {"start": (lane, m4, k), "turn": i, "seat": s, "seed": seed}
    return out


def _variants(start, c) -> Dict[str, tuple]:
    from ...champions_m4 import context
    lane, m4, k = start
    ctx = context(c)
    me = m4.a
    level = ctx.t.level_by_xp[min(me.xp, ctx.t.xp_cap)]
    rec = me._replace(pos=-c.tower, absent=c.recall_turns - 1)
    dead = me._replace(pos=-c.tower, hp=0, absent=ctx.death_turns(level))
    return {"alive": start, "recall": (lane, m4._replace(a=rec), k), "death": (lane, m4._replace(a=dead), k)}


def _death_job(job) -> dict:
    from ..m45b.engine import play
    _setup()
    c = cfg()
    start, idx = job["start"], job["idx"]
    k0 = start[2]
    seeds = ROLL[idx * N_ROLL:(idx + 1) * N_ROLL]
    if len(seeds) != N_ROLL:
        raise ValueError(f"state {idx}: the ROLL block has no {N_ROLL} seeds left")
    out = {}
    for name, st in _variants(start, c).items():
        vals = []
        for sd in seeds:
            pa, pb = _policies(k0, sd, WINDOW)
            vals.append(play(c, st, pa, pb, sd, WINDOW)["summary"][f"G{WINDOW}"])
        out[name] = vals
    from ...champions_m4 import context
    t = context(c).t
    lv = t.level_by_xp[min(start[1].a.xp, t.xp_cap)]
    return {"phase": job["phase"], "turn": job["turn"], "seed": job["seed"], "level": lv,
            "absence": _variants(start, c)["death"][1].a.absent, "G": out}


def death(workers: int) -> dict:
    from ..m45b.common import ci, run_jobs
    c = cfg()
    sampled = run_jobs(_sample_states, list(STATE_SEEDS), workers, "death states")
    jobs = []
    for phase in PHASES:
        for j, st in enumerate(sampled):
            r = st[phase]
            jobs.append({"phase": phase, "start": r["start"], "turn": r["turn"], "seed": r["seed"], "idx": j})
    res = run_jobs(_death_job, jobs, workers, "death rollouts")
    summary = {}
    for phase in PHASES:
        rows = [r for r in res if r["phase"] == phase]
        alive = [float(np.mean(r["G"]["alive"])) for r in rows]
        rec = [a - float(np.mean(r["G"]["recall"])) for a, r in zip(alive, rows)]
        dth = [a - float(np.mean(r["G"]["death"])) + c.kill_bounty for a, r in zip(alive, rows)]
        summary[phase] = {"turn": float(np.mean([r["turn"] for r in rows])),
                          "level": float(np.mean([r["level"] for r in rows])),
                          "absence": float(np.mean([r["absence"] for r in rows])),
                          "recall_cost": ci(rec), "death_cost": ci(dth),
                          "death_minus_recall": ci([d - r for d, r in zip(dth, rec)])}
    return {"phases": PHASES, "n_states": N_STATES, "n_roll": N_ROLL, "window": WINDOW, "gamma": GAMMA,
            "kill_gold": c.kill_bounty, "summary": summary,
            "rows": [{k: v for k, v in r.items()} for r in res]}


__all__ = ["selfplay", "selfplay_game", "death", "CHECKPOINTS", "PHASES"]
