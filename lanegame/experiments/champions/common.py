"""Shared helpers of the champion-mode experiments: seeded matchups (common random numbers), confidence
intervals, process-parallel round robins, and JSON / Markdown output under results/champions/."""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ...config import BEHAVIOR_SHORT, GameConfig
from ...io import dump_json

OUT = os.path.join("results", "champions")
Z95 = 1.96


def base_config(**kw) -> GameConfig:
    return GameConfig.champions(**kw)


def agent_by_name(name: str, cfg: GameConfig):
    from ...env import champion_agents
    agents = champion_agents(cfg)
    if name not in agents:
        raise KeyError(f"unknown champion agent {name!r}")
    return agents[name]


def behavior_frequencies(action_counts: Dict[str, int]) -> Dict[str, float]:
    """Share of turns per behavior letter (H, Y, T, S, R) and ABSENT, from env action labels like 'SP-H+P'."""
    letters = {v: k for k, v in BEHAVIOR_SHORT.items()}
    out: Dict[str, float] = {}
    total = sum(action_counts.values()) or 1
    for label, n in action_counts.items():
        key = "ABSENT" if label == "ABSENT" else label.split("-")[1][0]
        out[key] = out.get(key, 0.0) + n / total
    return {k: out[k] for k in sorted(out)}


def objective_frequencies(action_counts: Dict[str, int]) -> Dict[str, float]:
    total = sum(action_counts.values()) or 1
    out = {"Pressure": 0.0, "Vision": 0.0}
    for label, n in action_counts.items():
        if label.endswith("+P"):
            out["Pressure"] += n / total
        elif label.endswith("+V"):
            out["Vision"] += n / total
    return out


def wave_frequencies(action_counts: Dict[str, int]) -> Dict[str, float]:
    total = sum(action_counts.values()) or 1
    out = {"SP": 0.0, "SH": 0.0, "F": 0.0}
    for label, n in action_counts.items():
        if label != "ABSENT":
            out[label.split("-")[0]] += n / total
    return out


def summarize(stats) -> dict:
    """EpisodeStats -> a compact JSON-able record with 95% confidence intervals."""
    s = stats.summary()
    n = stats.episodes
    rec = {
        "names": s["names"], "episodes": n, "mean": s["mean"], "stderr": s["stderr"],
        "ci95": [s["mean"] - Z95 * s["stderr"], s["mean"] + Z95 * s["stderr"]], "std": s["std"],
        "win_rate": float(np.mean(stats.payoffs > 1e-9)), "loss_rate": float(np.mean(stats.payoffs < -1e-9)),
        "tie_rate": float(np.mean(np.abs(stats.payoffs) <= 1e-9)),
        "farm": s["mean_farm"], "farm_available": s["mean_farm_available"], "plates": s["mean_plates"],
        "ganks": s["mean_ganks"], "exposure": s["mean_exposure"],
        "trades_attempted": s["mean_trades_attempted"], "trades_connected": s["mean_trades_connected"],
        "yields_threatened": s["mean_yields_threatened"], "recalls": s["mean_recalls"],
        "turns_away": s["mean_turns_away"],
        "behaviors": [behavior_frequencies(c) for c in stats.action_counts],
        "objectives": [objective_frequencies(c) for c in stats.action_counts],
        "waves": [wave_frequencies(c) for c in stats.action_counts],
        "turns_per_second": s["turns_per_second"],
    }
    for c, arr in (stats.counters or {}).items():
        rec[c] = arr.mean(axis=0).tolist()
    return rec


def run_pair(args) -> dict:
    """Worker: (cfg dict, name A, name B, episodes, seed) -> summary record."""
    cfg_d, name_a, name_b, episodes, seed = args
    from ...env import LaneEnv, run_episodes
    cfg = GameConfig.from_dict(cfg_d)
    env = LaneEnv(cfg)
    st = run_episodes(env, agent_by_name(name_a, cfg), agent_by_name(name_b, cfg), episodes, seed)
    return summarize(st)


def run_pairs(cfg: GameConfig, pairs: Sequence[Tuple[str, str]], episodes: int, seed: int,
              workers: Optional[int] = None) -> List[dict]:
    """Every (A, B) pair with the same seed (common random numbers across pairs), in parallel processes."""
    jobs = [(cfg.to_dict(), a, b, episodes, seed) for a, b in pairs]
    workers = workers or min(len(jobs), max(1, (os.cpu_count() or 2) - 2))
    if workers <= 1:
        return [run_pair(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(run_pair, jobs))


def round_robin(cfg: GameConfig, names: Sequence[str], episodes: int, seed: int) -> dict:
    """All ordered pairs. M[i][j] = mean payoff to names[i] against names[j], averaged over both seats
    (i as A against j, and minus the payoff of j as A against i), with a pooled standard error."""
    pairs = [(a, b) for a in names for b in names]
    t0 = time.perf_counter()
    recs = run_pairs(cfg, pairs, episodes, seed)
    by = {(r["names"][0], r["names"][1]): r for r in recs}
    n = len(names)
    M, SE = np.zeros((n, n)), np.zeros((n, n))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            x, y = by[(a, b)], by[(b, a)]
            M[i, j] = 0.5 * (x["mean"] - y["mean"])
            SE[i, j] = 0.5 * np.sqrt(x["stderr"] ** 2 + y["stderr"] ** 2)
    return {"names": list(names), "matrix": M.tolist(), "stderr": SE.tolist(), "pairs": recs,
            "elapsed_s": time.perf_counter() - t0}


def meta_nash(M: np.ndarray, names: Sequence[str]) -> dict:
    """Zero-sum Nash equilibrium of the empirical payoff matrix (row player maximizes)."""
    from ...matrix_game import solve_matrix_game
    sol = solve_matrix_game(np.asarray(M, dtype=float))
    x = np.asarray(sol.x)
    return {"value": float(sol.value), "mixture": {names[i]: float(x[i]) for i in range(len(names)) if x[i] > 1e-6}}


def write(name: str, data: dict, md: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    dump_json(data, os.path.join(OUT, f"{name}.json"))
    with open(os.path.join(OUT, f"{name}.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(f"wrote {OUT}/{name}.json and .md")


def fmt_ci(mean: float, se: float, digits: int = 2) -> str:
    return f"{mean:+.{digits}f} ± {Z95 * se:.{digits}f}"


def md_table(header: Sequence[str], rows: Iterable[Sequence]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)
