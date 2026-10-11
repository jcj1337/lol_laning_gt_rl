"""Measure the lane economy / combat statistics of champion mode with scripted agents (read-only).

Run from the repo root:  python results/m4_design/scripts/econ_measure.py [episodes32] [episodes200]
Writes econ_results.npz next to this script.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.getcwd())
from lanegame.config import GameConfig  # noqa: E402
from lanegame.env.agents import champion_agents, episode_streams  # noqa: E402
from lanegame.env.core import LaneEnv  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
E32 = int(sys.argv[1]) if len(sys.argv) > 1 else 300
E200 = int(sys.argv[2]) if len(sys.argv) > 2 else 100

MATCHUPS = [("hold_farmer", "hold_farmer"), ("aggressive_trader", "hold_farmer"), ("shover", "hold_farmer"),
            ("shover", "shover"), ("safe_player", "hold_farmer"), ("yield_farmer", "hold_farmer"),
            ("opportunistic_trader", "opportunistic_trader")]
LONG = [("hold_farmer", "hold_farmer"), ("aggressive_trader", "hold_farmer"), ("shover", "hold_farmer"),
        ("shover", "shover")]
NCELL = 7   # own cells -3..3
NDIST = 7   # |x - end| 0..6


def run(name_a, name_b, horizon, episodes, seed):
    cfg = GameConfig.champions(horizon=horizon)
    agents = champion_agents(cfg)
    ag = (agents[name_a], agents[name_b])
    env = LaneEnv(cfg)
    T = cfg.tower
    K = horizon
    avail = np.zeros((episodes, K, 2))           # enemy minion HP died (farm opportunities, in HP)
    farm = np.zeros((episodes, K, 2))            # resources collected
    bycell = np.zeros((episodes, K, 2, NCELL))   # HP died by own-perspective cell
    bydist = np.zeros((episodes, K, 2, NDIST))   # HP died by distance from own end cell (present champion only)
    away = np.zeros((episodes, K, 2), bool)      # champion off the lane this turn (absent or recalled)
    hp0 = np.zeros((episodes, K, 2), int)        # HP at turn start (0 when absent)
    att = np.zeros((episodes, K, 2), bool)
    con = np.zeros((episodes, K, 2), bool)
    dealt = np.zeros((episodes, K, 2))           # trade damage dealt by side s
    taken = np.zeros((episodes, K, 2))
    gdmg = np.zeros((episodes, K, 2))
    tdmg = np.zeros((episodes, K, 2))
    died = np.zeros((episodes, K, 2), bool)
    recalled = np.zeros((episodes, K, 2), bool)
    endc = np.zeros((episodes, K, 2), int)
    plates = np.zeros((episodes, K, 2))
    ss = np.random.SeedSequence(seed)
    for e, child in enumerate(ss.spawn(episodes)):
        _, env_seed, ra, rb = episode_streams(child)
        oa, ob = env.reset(seed=env_seed)
        for k in range(K):
            for s in (0, 1):
                hp0[e, k, s] = env.champ.side(s).hp if env.champ.side(s).absent == 0 else 0
            a = ag[0].act(oa, ra)
            b = ag[1].act(ob, rb)
            r = env.step(a, b)
            inf = r.info
            for s in (0, 1):
                avail[e, k, s] = inf["farm_available"][s] / cfg.farm_per_hp
                farm[e, k, s] = inf["farm"][s]
                aw = bool(inf["away"][s])
                away[e, k, s] = aw
                end = inf["champ_end"][s]
                endc[e, k, s] = end
                for x, h in inf["kills"][s].items():
                    bycell[e, k, s, x + 3] += h
                    if not aw:
                        bydist[e, k, s, abs(x - end)] += h
                att[e, k, s] = inf["trade_attempted"][s]
                con[e, k, s] = inf["trade_connected"][s]
                dealt[e, k, s] = inf["trade_damage"][s]
                taken[e, k, s] = inf["damage_taken"][s]
                gdmg[e, k, s] = inf["gank_damage"][s]
                tdmg[e, k, s] = inf["tower_damage"][s]
                died[e, k, s] = inf["died"][s]
                recalled[e, k, s] = inf["recalled"][s]
                plates[e, k, s] = inf["plates"][s]
            if r.done:
                break
            oa, ob = r.obs
    return dict(avail=avail, farm=farm, bycell=bycell, bydist=bydist, away=away, hp0=hp0, att=att, con=con,
                dealt=dealt, taken=taken, gdmg=gdmg, tdmg=tdmg, died=died, recalled=recalled, endc=endc,
                plates=plates)


def main():
    out = {}
    t0 = time.perf_counter()
    for i, (a, b) in enumerate(MATCHUPS):
        d = run(a, b, 32, E32, 1000 + i)
        for key, v in d.items():
            out[f"h32|{a}|{b}|{key}"] = v
        print(f"h32 {a} vs {b}: {time.perf_counter() - t0:.1f}s", flush=True)
    for i, (a, b) in enumerate(LONG):
        d = run(a, b, 200, E200, 2000 + i)
        for key, v in d.items():
            out[f"h200|{a}|{b}|{key}"] = v
        print(f"h200 {a} vs {b}: {time.perf_counter() - t0:.1f}s", flush=True)
    np.savez_compressed(os.path.join(HERE, "econ_results.npz"), **out)


if __name__ == "__main__":
    main()
