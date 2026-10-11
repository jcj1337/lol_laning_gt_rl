"""Evaluation harness (roadmap M3c): head-to-head results in both seats with confidence intervals, round
robins, and approximate best responses.

head_to_head(cfg, x, y) plays `episodes` games with x in seat A and `episodes` with x in seat B, on the SAME
episode seeds (common random numbers: identical chance draws for both seat assignments and for every opponent
evaluated with the same seed), and reports the payoff to x. Mechanical skill belongs to the player, not the
seat: with `skill=(m_x, m_y)` the configuration is set so that x has m_x in either seat.

approximate_best_response(cfg, target) trains a fresh PPO learner against the frozen target only, then evaluates
it on held-out seeds. Its payoff is a LOWER bound on the target's exploitability (the value a true best response
would obtain against it, relative to the game value 0 of the symmetric game): it is limited by the training
budget, the network and the optimizer. It is never a certificate; only the exact solver gives one, on the legacy
configurations.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ..config import GameConfig
from ..env import LaneEnv, run_episodes
from .train import OpponentSampler, train_ppo
from .ppo import PPOConfig
from .agent import NetAgent

Z95 = 1.96
COUNTERS = ("farm", "plates", "ganks", "exposure", "trades_attempted", "trades_connected", "recalls", "turns_away")
CHAMP_COUNTERS = ("trades_landed", "trade_damage_dealt", "damage_taken", "gank_damage_taken", "deaths",
                  "recalls_interrupted", "tower_damage_taken", "bounty")


def _seat_stats(st, seat: int) -> dict:
    """Metrics of the player in `seat` from an EpisodeStats."""
    s = st.summary()
    out = {"farm": s["mean_farm"][seat], "farm_available": s["mean_farm_available"][seat],
           "plates": s["mean_plates"][seat], "ganks": s["mean_ganks"][seat], "exposure": s["mean_exposure"][seat],
           "trades_attempted": s["mean_trades_attempted"][seat], "trades_connected": s["mean_trades_connected"][seat],
           "recalls": s["mean_recalls"][seat], "turns_away": s["mean_turns_away"][seat]}
    if st.counters:
        for c in CHAMP_COUNTERS:
            out[c] = float(st.counters[c][:, seat].mean())
    counts = st.action_counts[seat]
    total = sum(counts.values()) or 1
    beh: Dict[str, float] = {}
    obj = {"Pressure": 0.0, "Vision": 0.0}
    waves = {"SP": 0.0, "SH": 0.0, "F": 0.0}
    for label, n in counts.items():
        if label == "ABSENT":
            beh["ABSENT"] = beh.get("ABSENT", 0.0) + n / total
            continue
        if "-" in label:
            w, rest = label.split("-", 1)
            waves[w] = waves.get(w, 0.0) + n / total
            beh[rest[0]] = beh.get(rest[0], 0.0) + n / total
            if rest.endswith("+P"):
                obj["Pressure"] += n / total
            elif rest.endswith("+V"):
                obj["Vision"] += n / total
        else:
            waves[label] = waves.get(label, 0.0) + n / total
    out["behaviors"] = dict(sorted(beh.items()))
    out["objectives"] = obj
    out["waves"] = waves
    return out


def head_to_head(cfg: GameConfig, x, y, episodes: int = 500, seed: int = 10_000,
                 skill: Optional[Tuple[float, float]] = None) -> dict:
    """Payoff to x against y over both seats (2 x episodes games), with a 95% confidence interval."""
    cfg_a, cfg_b = cfg, cfg
    if skill is not None:
        cfg_a = cfg.with_(skill_a=skill[0], skill_b=skill[1])
        cfg_b = cfg.with_(skill_a=skill[1], skill_b=skill[0])
    t0 = time.perf_counter()
    st_a = run_episodes(LaneEnv(cfg_a), x, y, episodes, seed)          # x in seat A
    st_b = run_episodes(LaneEnv(cfg_b), y, x, episodes, seed)          # x in seat B
    pay = np.concatenate([st_a.payoffs, -st_b.payoffs])
    mean, se = float(pay.mean()), float(pay.std(ddof=1) / np.sqrt(len(pay)))
    sx_a, sx_b = _seat_stats(st_a, 0), _seat_stats(st_b, 1)
    sy_a, sy_b = _seat_stats(st_a, 1), _seat_stats(st_b, 0)

    def merge(p, q):
        out = {}
        for k in p:
            if isinstance(p[k], dict):
                out[k] = {kk: 0.5 * (p[k].get(kk, 0.0) + q[k].get(kk, 0.0)) for kk in set(p[k]) | set(q[k])}
            else:
                out[k] = 0.5 * (p[k] + q[k])
        return out
    return {"x": getattr(x, "name", "x"), "y": getattr(y, "name", "y"), "episodes": 2 * episodes, "seed": seed,
            "skill": list(skill) if skill else [cfg.skill_a, cfg.skill_b],
            "mean": mean, "stderr": se, "ci95": [mean - Z95 * se, mean + Z95 * se], "std": float(pay.std(ddof=1)),
            "win_rate": float(np.mean(pay > 1e-9)), "loss_rate": float(np.mean(pay < -1e-9)),
            "tie_rate": float(np.mean(np.abs(pay) <= 1e-9)),
            "seat_A": {"mean": float(st_a.payoffs.mean()), "stderr": st_a.stderr},
            "seat_B": {"mean": float(-st_b.payoffs.mean()), "stderr": st_b.stderr},
            "x_stats": merge(sx_a, sx_b), "y_stats": merge(sy_a, sy_b), "elapsed_s": time.perf_counter() - t0}


def round_robin(cfg: GameConfig, agents: Dict[str, object], episodes: int = 300, seed: int = 10_000,
                rows: Optional[Sequence[str]] = None) -> dict:
    """M[i][j] = payoff to rows[i] against agents[j] (both seats). Rows default to every agent."""
    names = list(agents)
    rows = list(rows) if rows is not None else names
    M, SE, recs = np.zeros((len(rows), len(names))), np.zeros((len(rows), len(names))), []
    for i, a in enumerate(rows):
        for j, b in enumerate(names):
            if a == b:
                continue
            r = head_to_head(cfg, agents[a], agents[b], episodes, seed)
            M[i, j], SE[i, j] = r["mean"], r["stderr"]
            recs.append(r)
    return {"rows": rows, "columns": names, "matrix": M.tolist(), "stderr": SE.tolist(), "pairs": recs}


def best_scripted_response(cfg: GameConfig, target, scripted: Dict[str, object], episodes: int = 300,
                           seed: int = 10_000) -> dict:
    """The scripted agent with the highest payoff against `target` (both seats)."""
    res = {n: head_to_head(cfg, a, target, episodes, seed) for n, a in scripted.items()}
    best = max(res, key=lambda n: res[n]["mean"])
    return {"best": best, "value": res[best]["mean"], "stderr": res[best]["stderr"],
            "all": {n: {"mean": r["mean"], "stderr": r["stderr"]} for n, r in res.items()}}


def approximate_best_response(cfg: GameConfig, target, ppo_cfg: PPOConfig, eval_episodes: int = 500,
                              eval_seed: int = 77_777, out_dir: Optional[str] = None, name: str = "br",
                              verbose: bool = False) -> dict:
    """Train a fresh PPO learner against the frozen `target` only, evaluate it on held-out seeds (both seats)."""
    res = train_ppo(cfg, ppo_cfg, OpponentSampler([target]), out_dir=out_dir, verbose=verbose, name=name)
    br = NetAgent(res.net, name)
    ev = head_to_head(cfg, br, target, eval_episodes, eval_seed)
    curve = [(h["steps"], h["mean_return"]) for h in res.history]
    return {"target": getattr(target, "name", "target"), "train_steps": ppo_cfg.n_updates * ppo_cfg.batch_size,
            "train_seed": ppo_cfg.seed, "train_elapsed_s": res.elapsed, "value": ev["mean"], "stderr": ev["stderr"],
            "ci95": ev["ci95"], "eval": ev, "curve": curve, "agent": br}
