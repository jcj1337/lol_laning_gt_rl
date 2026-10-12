"""S1 long-horizon soak (plan §7 S1, design §14): random-legal and scripted play with every invariant checked on
every turn.

    python -m lanegame.experiments.m4.soak [--turns 50000]

Checked per turn: zero-sum rewards; the exact integer reward decomposition; the state invariants of design §3.4;
monotone XP, items and tower damage; XP and gold bounded by the turn's kills; minion conservation (alive = initial +
spawned - removed, no overflow); plate gold paid = v_plate x plates broken; the ledger identity (spendable gold =
earned - purchases - clamped); purchases only on return turns; observations in [0, 1] and exactly decodable; the
per-turn reward bound |r| <= 8.575 (design §10.5.3).
"""

from __future__ import annotations

import argparse

import numpy as np

from ...champions_m4 import check_invariants
from ...config import GameConfig
from ...env import LaneEnv, UniformAgent
from ...env.agents_m4 import gate_attackers, m4_suite
from ...env.discounted import per_turn_bound
from ...env.observation_m4 import decode_m4
from ...mechanics import total_hp
from ... import progression as pg
from .common import Timer, md_table, meta, save


def soak(cfg: GameConfig, turns: int, seed: int, agents=(None, None), decode_every: int = 1) -> dict:
    env = LaneEnv(cfg, max_turns=turns, cache_size=20_000)
    rng = np.random.default_rng(seed)
    ags = [a or UniformAgent() for a in agents]
    obs = env.reset(seed=seed)
    t = pg.tables(cfg)
    alive0 = total_hp(env.state, cfg)
    spawned, removed, plates_paid, earned, spent, clamped = [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]
    prev = env.champ
    bound = per_turn_bound(cfg)
    stats = {"turns": 0, "max_abs_reward": 0.0, "purchases": 0, "deaths": 0, "plates": 0, "freeze_maintained": 0,
             "adjacent_shots": 0, "level_ups": 0, "xp_discarded": 0, "gold_unspendable_units": 0, "decoded": 0}
    while True:
        items_before = (env.champ.a.items, env.champ.b.items)
        r = env.step(ags[0].act(obs[0], rng), ags[1].act(obs[1], rng))
        i = r.info
        assert r.rewards[0] + r.rewards[1] == 0.0
        assert i["reward_units"] == sum(i[k][0] - i[k][1] for k in ("farm_units", "plate_units", "bounty_units"))
        assert abs(r.rewards[0]) <= bound + 1e-12
        cur = env.champ
        check_invariants(cur, cfg)
        for s in (0, 1):
            a, b = prev.side(s), cur.side(s)
            assert b.xp >= a.xp and b.items >= a.items and b.tower_dmg >= a.tower_dmg
            kills = sum(i["kills"][s].values())
            assert i["xp_gain"][s] + i["xp_discarded"][s] <= cfg.xp_per_hp * kills
            assert i["farm"][s] <= cfg.farm_per_hp * kills + 1e-12
            ev = i["events"]
            spawned[s] += ev["spawned_hp"][s]
            removed[s] += ev["killed_hp"][s] + ev["tower_hp"][s] + ev["overflow_hp"][s]
            assert ev["overflow_hp"][s] == 0
            assert total_hp(env.state, cfg)[s] == alive0[s] + spawned[s] - removed[s]
            plates_paid[s] += i["plate_units"][s]
            assert plates_paid[s] == cfg.gold_scale * pg.plates_broken(b.tower_dmg)
            earned[s] += i["earned_units"][s]
            clamped[s] += i["gold_unspendable"][s]
            n = i["purchases"][s]
            assert n == 0 or i["returned"][s]
            spent[s] += sum(t.prices[items_before[s] + j] for j in range(n))
            assert b.gold == earned[s] - spent[s] - clamped[s]
            stats["purchases"] += n
            stats["deaths"] += int(i["died"][s])
            stats["plates"] += i["plates_broken"][s]
            stats["freeze_maintained"] += int(i["freeze_maintained"][s])
            stats["adjacent_shots"] += int(i["adjacent_shot"][s])
            stats["level_ups"] += i["level_up"][s]
            stats["xp_discarded"] += i["xp_discarded"][s]
            stats["gold_unspendable_units"] += i["gold_unspendable"][s]
        if decode_every and stats["turns"] % decode_every == 0:
            for side in (0, 1):
                v = r.obs[side].vector
                assert 0.0 <= v.min() and v.max() <= 1.0
                ph, lane, m4, _ = decode_m4(v, cfg)
                assert lane == (env.state if side == 0 else env.state.mirror())._replace(ap=0, bp=0)
                assert m4 == (cur if side == 0 else cur.mirror())
                stats["decoded"] += 1
        stats["max_abs_reward"] = max(stats["max_abs_reward"], abs(r.rewards[0]))
        stats["turns"] += 1
        prev = cur
        if r.done:
            break
        obs = r.obs
    stats["final"] = {"levels": list(i["levels"]), "items": list(i["items"]), "tower_dmg": list(i["tower_dmg"])}
    stats["per_turn_bound"] = bound
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=50_000)
    a = ap.parse_args(argv)
    cfg = GameConfig.m4()
    timer = Timer()
    runs = {"random legal (50 000 turns)": soak(cfg, a.turns, 1)}
    runs["random legal, skills 0.7 / 0.3"] = soak(cfg.with_(skill_a=0.7, skill_b=0.3), a.turns // 5, 2)
    pool = {**m4_suite(), **{"g_" + k: v for k, v in gate_attackers().items()}}
    for j, (x, y) in enumerate([("freezer", "presser"), ("gold_recaller", "aggressive_trader"),
                                ("g_adaptive_breaker", "freezer"), ("xp_keeper", "g_stacker")]):
        runs[f"{x} vs {y} (2 000 turns)"] = soak(cfg, 2000, 10 + j, (pool[x], pool[y]))
    d = {"meta": meta(cfg, turns=a.turns), "runs": runs, "violations": 0, "elapsed_s": timer()}
    keys = ["turns", "max_abs_reward", "deaths", "purchases", "plates", "freeze_maintained", "adjacent_shots",
            "level_ups", "xp_discarded", "decoded"]
    L = ["# S1 — Long-horizon soak (M4.2)", "",
         f"Commit `{d['meta']['commit'][:10]}`. Every design §14 invariant asserted on every turn (zero-sum, exact "
         f"reward decomposition, §3.4 state invariants, monotone progression, kill bounds, minion conservation, plate "
         f"gold, the ledger identity, purchase timing, observation range and exact decoding, the per-turn reward bound "
         f"{runs['random legal (50 000 turns)']['per_turn_bound']}). **No violation** in "
         f"{sum(r['turns'] for r in runs.values()):,} turns ({timer():.0f} s). Command: `{d['meta']['command']}`.", "",
         md_table(["run"] + keys + ["final levels / items / tower damage"],
                  [[n] + [(f"{r[k]:.3f}" if isinstance(r[k], float) else r[k]) for k in keys]
                   + [f"{r['final']['levels']} / {r['final']['items']} / {r['final']['tower_dmg']}"]
                   for n, r in runs.items()])]
    save("soak", d, "\n".join(L))
    print("S1 soak: no violation", sum(r["turns"] for r in runs.values()), "turns", f"({timer():.0f} s)")
    return d


if __name__ == "__main__":
    main()
