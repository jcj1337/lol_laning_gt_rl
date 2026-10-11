"""M2e-A: functional correctness of champion mode, checked turn by turn over many seeded episodes.

Every turn of every episode is checked against the rules (no sampling of rules, only of trajectories):
state bounds (cell, HP, absence timer), legal sets and masks, the absence timeline, movement limits, damage
and HP accounting, zero-sum rewards and their decomposition, no remote farming, no remote siege (plates,
thinning, vision only from the enemy tower with an effective objective), no contribution or exposure of an
away champion, observations as pure functions of (k, lane, champions) that are injective over the visited
states (nothing decision-relevant hidden), mirror equivariance of the resolution with mirrored chance
draws, and exact replay from the seed.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List

import numpy as np

from ...champions import (AWAY, CHANNEL, NO_REACH, RETREAT, ChampState, champion_legal_actions, mirror_chance, resolve_turn,
                          swap_seats)
from ...config import CHAMP_ABSENT, NONE, GameConfig
from ...env import LaneEnv, UniformAgent, champion_agents, legal_mask, observe, play_episode
from ...state import State
from .common import base_config, md_table, write

A, B = 0, 1


def check_turn(cfg: GameConfig, lane: State, champ: ChampState, k: int, a: int, b: int, r, seen: Dict) -> List[str]:
    """Violations of the champion-mode rules in one turn (an empty list if none). `seen` maps observation
    bytes to (k, lane, champ) across calls (injectivity)."""
    bad = []
    i = r.info
    T = cfg.tower
    after = ChampState.from_list(i["champ_after"])
    acts = (a, b)
    # observations: pure functions of (k, lane, champ), injective, bounded; masks equal the legal sets
    for s in (A, B):
        v = observe(lane, k, s, cfg, champ)
        if np.any(v < 0) or np.any(v > 1):
            bad.append(f"observation out of [0, 1] (side {s})")
        key = (s, v.tobytes())
        if key in seen and seen[key] != (k, lane, champ):
            bad.append("observation collision: two different states share one vector")
        seen[key] = (k, lane, champ)
        legal = champion_legal_actions(lane, champ, s, cfg)
        if tuple(np.flatnonzero(legal_mask(lane, s, cfg, champ=champ))) != legal:
            bad.append("mask differs from the legal set")
        if acts[s] not in legal or tuple(i["legal"][s]) != legal:
            bad.append("illegal action accepted or legal set mismatch")
        if champ.side(s).absent > 0 and legal != (CHAMP_ABSENT,):
            bad.append("absent champion with a legal set other than {ABSENT}")
        if champ.side(s).absent == 0 and CHAMP_ABSENT in legal:
            bad.append("ABSENT legal for a present champion")
    # rewards
    if r.rewards[0] != -r.rewards[1]:
        bad.append("rewards not zero-sum")
    if abs(i["reward"] - (i["farm_payoff"] + i["plate_payoff"] + i["bounty_payoff"])) > 1e-9 or i["gank_payoff"] != 0:
        bad.append("reward decomposition (farm + plates + bounty, no gank loss) violated")
    for s in (A, B):
        o = 1 - s
        me0, me1 = champ.side(s), after.side(s)
        # state bounds
        if not (-T <= me1.pos <= T and 0 <= me1.hp <= cfg.champion_hp and 0 <= me1.absent <= cfg.champion_absent_max):
            bad.append(f"state out of bounds: {me1}")
        if me1.hp == 0 and me1.absent == 0:
            bad.append("present champion with 0 HP")
        # absence timeline
        if me0.absent > 0:
            if not i["away"][s] or me1.absent != me0.absent - 1 or me1.pos != -T:
                bad.append("absence timer did not count down by one")
            if me1.absent == 0 and me1.hp != cfg.champion_hp:
                bad.append("returning champion without full HP")
        elif i["recalled"][s]:
            if me1.absent != cfg.recall_turns - 1 or me1.pos != -T or me0.pos > cfg.recall_line:
                bad.append("recall timeline / legality violated")
        elif i["died"][s]:
            if me1 != (-T, cfg.champion_hp if cfg.death_turns == 0 else 0, cfg.death_turns):
                bad.append("death did not start the death timeline")
            if me0.hp - i["damage_taken"][s] > 0:
                bad.append("death without lethal damage")
        else:
            if me1.absent != 0 or me1.hp != me0.hp - i["damage_taken"][s] or me1.hp <= 0:
                bad.append("HP accounting of a present champion violated")
            if abs(me1.pos - me0.pos) > cfg.champion_speed:
                bad.append("champion moved faster than champion_speed")
        # damage accounting
        if i["damage_taken"][s] != i["trade_damage"][o] + i["tower_damage"][s] + i["gank_damage"][s]:
            bad.append("damage sources do not add up")
        if i["trade_damage"][s] > 0 and not i["trade_connected"][s]:
            bad.append("trade damage without a connected trade")
        if i["tower_damage"][s] > 0 and not (i["trade_connected"][s] and i["champ_end"][s] == T):
            bad.append("tower shot without a dive")
        if i["gank_damage"][s] > 0 and not i["ganked"][s]:
            bad.append("gank damage without a gank")
        # an away champion takes no part
        if i["away"][s]:
            if (i["farm"][s] != 0 or i["gank_p"][s] != 0 or i["damage_taken"][s] != 0 or i["trade_connected"][s]
                    or i["champion_reach"][s] != NO_REACH or i["modes"][s] != AWAY or i["threatened"][s]
                    or i["effective_objectives"][s] != NONE or i["plate_count"][s] != 0 or i["vision_set"][s]):
                bad.append("an away champion took part in the turn")
        if i["modes"][s] == CHANNEL and (i["farm"][s] != 0 or i["champion_reach"][s] != NO_REACH):
            bad.append("a channeling champion farmed or helped its wave")
        # no remote farming
        if i["farm"][s] > 0:
            reach = cfg.retreat_farm_reach if i["modes"][s] == RETREAT else cfg.farm_reach
            if not any(abs(x - i["champ_end"][s]) <= reach for x, hp in i["kills"][s].items() if hp > 0):
                bad.append("farm collected from out of reach")
        # no remote siege
        if i["plate_count"][s] > 0 or i["thinned"][s] > 0 or i["vision_set"][s]:
            if i["champ_end"][s] != T or i["effective_objectives"][s] == NONE or i["trade_connected"][s]:
                bad.append("siege effect without the champion at the enemy tower")
    return bad


def sweep(cfg: GameConfig, pairs, episodes: int, seed: int, mirror_every: int = 7) -> dict:
    env = LaneEnv(cfg)
    agents = champion_agents(cfg)
    seen: Dict = {}
    violations: Counter = Counter()
    turns = mirror_checks = replays = 0
    rng = np.random.default_rng(seed)
    events = Counter()
    for (na, nb) in pairs:
        for e in range(episodes):
            ep_seed = int(rng.integers(2 ** 31))
            log = play_episode(env, agents[na], agents[nb], seed=ep_seed, record=True)
            prev = None
            for t in log.turns:
                lane, champ = State.from_list(t["state_before"]), ChampState.from_list(t["champ_before"])
                k, (a, b) = t["k"], t["actions"]
                if prev is not None and (prev[0] != lane or prev[1] != champ):
                    violations["state does not chain between turns"] += 1

                class _R:                                    # the StepResult fields check_turn reads
                    rewards = (t["reward"], -t["reward"])
                    info = t
                for v in check_turn(cfg, lane, champ, k, a, b, _R, seen):
                    violations[v] += 1
                if turns % mirror_every == 0:
                    x = resolve_turn(lane, champ, k, a, b, cfg, u=t["chance"])
                    m = resolve_turn(lane.mirror(), champ.mirror(), k, b, a, swap_seats(cfg), u=mirror_chance(t["chance"]))
                    if (m.champ != x.champ.mirror() or m.lane.state != x.lane.state.mirror()
                            or not np.allclose(m.farm, x.farm[::-1]) or m.died != x.died[::-1]):
                        violations["mirror equivariance"] += 1
                    if x.champ.to_list() != t["champ_after"]:
                        violations["resolution not reproducible from the recorded chance draws"] += 1
                    mirror_checks += 1
                for s in (A, B):
                    events["trade_landed"] += t["trade_damage"][s] > 0
                    events["dive"] += t["tower_p"][s] > 0
                    events["gank"] += bool(t["ganked"][s])
                    events["death"] += bool(t["died"][s])
                    events["recall"] += bool(t["recalled"][s])
                    events["recall_interrupted"] += bool(t["recall_interrupted"][s])
                    events["plates"] += t["plate_count"][s]
                    events["vision_set"] += bool(t["vision_set"][s])
                    events["double_death"] += s == 0 and all(t["died"])
                turns += 1
                prev = (State.from_list(t["state_after"]), ChampState.from_list(t["champ_after"]))
            if e == 0:                                       # exact replay from the seed
                again = play_episode(env, agents[na], agents[nb], seed=ep_seed, record=True)
                replays += 1
                if again.rewards != log.rewards or [x["champ_after"] for x in again.turns] != [x["champ_after"] for x in log.turns]:
                    violations["episode not reproducible from its seed"] += 1
    return {"turns": turns, "distinct_observations": len(seen), "mirror_checks": mirror_checks, "replays": replays,
            "violations": dict(violations), "events": dict(events)}


def run(episodes: int = 60, seed: int = 2026):
    from ...env.agents import champion_suite
    suite = [a.name for a in champion_suite()]
    pairs = [("uniform", "uniform"), ("uniform", "aggressive_trader"), ("trade", "trade"), ("shove_trade", "yield")]
    pairs += [(x, y) for x in suite for y in suite][::5]
    out = {}
    for label, cfg in (("equal skill", base_config()), ("asymmetric skill 0.8 / 0.2", base_config(skill_a=0.8, skill_b=0.2)),
                       ("kill bounty 1, recall 3 turns", base_config(kill_bounty=1.0, recall_turns=3))):
        out[label] = sweep(cfg, pairs, episodes, seed)
        print(label, out[label]["turns"], "turns,", sum(out[label]["violations"].values()), "violations")
    rows = [(lab, r["turns"], r["distinct_observations"], r["mirror_checks"], r["replays"],
             sum(r["violations"].values()), ", ".join(f"{k} {v}" for k, v in sorted(r["events"].items())))
            for lab, r in out.items()]
    md = "\n".join([
        "# Champion mode: functional correctness sweep (M2e-A)", "",
        f"Generated by `python -m lanegame experiments-champions invariants`; {episodes} seeded episodes per pair, "
        f"{len(pairs)} agent pairs per configuration (uniform random play, the naive constant behaviors and a "
        "sample of the scripted suite), K = 32. Every turn is checked against every rule listed in "
        "`lanegame/experiments/champions/invariants.py`; mirror equivariance and replay from the recorded chance "
        "draws are checked on every seventh turn; each pair's first episode is replayed from its seed.", "",
        md_table(["configuration", "turns checked", "distinct observations", "mirror checks", "seed replays",
                  "violations", "events observed (both sides)"], rows), "",
        "Violations by kind: " + (", ".join(f"{k}: {v}" for lab, r in out.items() for k, v in r["violations"].items())
                                  or "none") + ".", ""])
    write("invariants", out, md)
    return out


if __name__ == "__main__":
    run()
