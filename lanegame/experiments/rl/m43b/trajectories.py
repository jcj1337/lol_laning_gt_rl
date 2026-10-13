"""Representative games of the learned policies, exported for the visual debugger (M4.2e-D viewer).

A viewer recording is a pure function of (Setup, action list): `lanegame.viewer.session.play` resets a LaneEnv at
the setup's start state with the setup's INTEGER seed as the chance stream and replays the actions. `record_game`
plays a game exactly that way (env.reset(seed=int), the agents sampling from their own streams), so the exported
file {"schema": "lanegame.viewer_recording", "config", "scenario": "s0", "gap": "equal", "seed", "actions", ...}
replays to the identical trajectory in the viewer ("Recorded trajectory" mode) and in `session.play`.

The exported set: self-play of the seed-0 finals of every condition, the baseline final against the equilibrium-like
references (push_punish, clear_punish) and against crash_recall, a second self-play seed, and the attacker against
its target when the exploit stage has run. Each file carries a compact event timeline for the report.
"""

from __future__ import annotations

import os
import time
from typing import List, Optional, Sequence

import numpy as np

from ....config import GameConfig
from ....env import LaneEnv
from .common import CONDITIONS, OUT, PRIMARY, SEEDS, TRAJ_SEED, WINDOW_DEV, agent_from_spec, attacker_dir, cfg_game, meta, save_json

SCHEMA = "lanegame.viewer_recording"
TRAJ_DIR = os.path.join(OUT, "trajectories")


def record_game(cfg: GameConfig, x, y, seed: int, turns: int) -> dict:
    """One game from s0 with the integer chance seed `seed` (the viewer's convention); returns actions, events and
    summary counters."""
    env = LaneEnv(cfg, max_turns=turns, cache_size=20_000)
    rng_a, rng_b = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2))
    for ag in (x, y):
        if callable(getattr(ag, "reset", None)):
            ag.reset()
    obs = env.reset(seed=seed)
    actions, events, cum = [], [], 0.0
    cnt = {"deaths": [0, 0], "recalls": [0, 0], "crashes": [0, 0], "plates": [0, 0], "trades_connected": [0, 0],
           "farm": [0.0, 0.0], "invested": [0.0, 0.0], "returns_after_death": [0, 0], "returns_after_recall": [0, 0]}
    cause = [None, None]
    sc = cfg.gold_scale
    for t in range(turns):
        a, b = x.act(obs[0], rng_a), y.act(obs[1], rng_b)
        r = env.step(a, b)
        info = r.info
        for ag, s in ((x, 0), (y, 1)):
            nf = getattr(ag, "notify", None)
            if callable(nf):
                nf(info, s)
        actions.append([int(a), int(b)])
        cum += r.rewards[0]
        for s, n in ((0, "A"), (1, "B")):
            if info["died"][s]:
                cnt["deaths"][s] += 1
                cause[s] = "death"
                src = "ganked" if info["gank_damage"][s] > 0 else ("tower" if info["tower_damage"][s] > 0 else "trade")
                events.append([t, f"{n} dies ({src}; HP {info['champ_before'][s][1]} before, gold {info['champ_before'][s][5] / sc:.1f} unspent)"])
            if info["recalled"][s]:
                cnt["recalls"][s] += 1
                cause[s] = "recall"
                events.append([t, f"{n} recalls (HP {info['champ_before'][s][1]}, gold {info['champ_before'][s][5] / sc:.1f} unspent)"])
            if info["recall_interrupted"][s]:
                events.append([t, f"{n}'s recall is interrupted"])
            if info["returned"][s]:
                if cause[s] == "death":
                    cnt["returns_after_death"][s] += 1
                elif cause[s] == "recall":
                    cnt["returns_after_recall"][s] += 1
                cnt["invested"][s] += info["invested"][s] / sc
                events.append([t, f"{n} returns next turn, invests {info['invested'][s] / sc:.1f}"])
            if info["crash_hp"][s]:
                cnt["crashes"][s] += 1
                events.append([t, f"{n} crashes {info['crash_hp'][s]} HP into the enemy tower"])
            if info["plates_broken"][s]:
                cnt["plates"][s] += info["plates_broken"][s]
                events.append([t, f"{n} breaks a plate (+{info['plate_units'][s] / sc:.1f})"])
            if info["level_up"][s]:
                events.append([t, f"{n} reaches level {info['levels'][s]}"])
            cnt["trades_connected"][s] += int(info["trade_connected"][s])
            cnt["farm"][s] += info["farm"][s]
        if r.done:
            break
        obs = r.obs
    labels = [[cfg.label(a), cfg.label(b)] for a, b in actions]
    shares = {}
    for s in (0, 1):
        labs = [l[s] for l in labels if l[s] != "ABSENT"]
        n = max(len(labs), 1)
        shares[s] = {"Hold": sum(l.startswith("HO-") for l in labs) / n, "Thin": sum(l.startswith("TN-") for l in labs) / n,
                     "Clear": sum(l.startswith("CL-") for l in labs) / n,
                     "Trade": sum("-T" in l for l in labs) / n, "Safe": sum("-S" in l for l in labs) / n,
                     "Recall": sum("-R" in l for l in labs) / n, "Pressure": sum(l.endswith("+P") for l in labs) / n}
    return {"actions": actions, "events": events, "payoff_A": cum, "turns": len(actions), "counters": cnt,
            "action_shares": shares, "final": {"levels": list(info["levels"]), "gold_earned_units": None}}


def export(name: str, x_spec: str, y_spec: str, seed: int, turns: int, label: str, cfg: Optional[GameConfig] = None) -> dict:
    from ....io import dump_json
    cfg = cfg if cfg is not None else cfg_game()
    x, y = agent_from_spec(x_spec, cfg), agent_from_spec(y_spec, cfg)
    g = record_game(cfg, x, y, seed, turns)
    rec = {"schema": SCHEMA, "version": 1, "config": "m4_lasthit", "scenario": "s0", "gap": "equal", "seed": seed,
           "names": [x_spec, y_spec], "label": label, **g}
    os.makedirs(TRAJ_DIR, exist_ok=True)
    path = os.path.join(TRAJ_DIR, f"{name}.json")
    dump_json(rec, path)
    rec["path"] = path
    return rec


def planned(conds: Sequence[str], seeds: Sequence[int]) -> List[tuple]:
    s0 = seeds[0]
    plan = [(f"{PRIMARY}_s{s0}_selfplay_a", f"final:{PRIMARY}:{s0}", f"final:{PRIMARY}:{s0}", TRAJ_SEED,
             "baseline final against itself (seed a)"),
            (f"{PRIMARY}_s{s0}_selfplay_b", f"final:{PRIMARY}:{s0}", f"final:{PRIMARY}:{s0}", TRAJ_SEED + 1,
             "baseline final against itself (seed b)"),
            (f"{PRIMARY}_s{s0}_vs_push_punish", f"final:{PRIMARY}:{s0}", "ref:push_punish", TRAJ_SEED,
             "baseline final (A) against the slow-push + punish reference (B)"),
            (f"{PRIMARY}_s{s0}_vs_clear_punish", f"final:{PRIMARY}:{s0}", "ref:clear_punish", TRAJ_SEED,
             "baseline final (A) against the Clear + punish reference (B)"),
            (f"{PRIMARY}_s{s0}_vs_crash_recall", f"final:{PRIMARY}:{s0}", "ref:crash_recall", TRAJ_SEED,
             "baseline final (A) against the crash-and-Recall reference (B)"),
            (f"{PRIMARY}_s{s0}_vs_initial", f"final:{PRIMARY}:{s0}", f"initial:{s0}", TRAJ_SEED,
             "baseline final (A) against its untrained initial network (B)")]
    for c in conds:
        if c != PRIMARY:
            plan.append((f"{c}_s{s0}_selfplay", f"final:{c}:{s0}", f"final:{c}:{s0}", TRAJ_SEED,
                         f"{CONDITIONS[c]['label']}: final against itself"))
    if os.path.exists(os.path.join(attacker_dir(PRIMARY, s0), "final.pt")):
        plan.append((f"attacker_vs_{PRIMARY}_s{s0}", f"attacker:{PRIMARY}:{s0}", f"final:{PRIMARY}:{s0}", TRAJ_SEED,
                     "the approximate best response (A) against the baseline final it was trained on (B)"))
    return plan


def trajectories(conds: Sequence[str] = tuple(CONDITIONS), seeds: Sequence[int] = SEEDS, turns: int = WINDOW_DEV) -> dict:
    t0 = time.perf_counter()
    index = {}
    for name, xs, ys, seed, label in planned(conds, seeds):
        rec = export(name, xs, ys, seed, turns, label)
        index[name] = {k: rec[k] for k in ("path", "names", "label", "seed", "turns", "payoff_A", "counters",
                                           "action_shares", "events")}
        print(f"  {name}: payoff to A {rec['payoff_A']:+.1f}, deaths {rec['counters']['deaths']}, recalls "
              f"{rec['counters']['recalls']}, crashes {rec['counters']['crashes']}", flush=True)
    data = {"meta": meta(stage="trajectories", turns=turns, seed=TRAJ_SEED), "elapsed_s": time.perf_counter() - t0,
            "how_to_view": "streamlit run lanegame/viewer/app.py -> mode 'Recorded trajectory' (files of "
                           "results/rl/m43b/trajectories/), or lanegame.viewer.session.load_recording(path)",
            "recordings": index}
    save_json("trajectories", data)
    return data


__all__ = ["record_game", "export", "trajectories", "SCHEMA", "TRAJ_DIR"]
