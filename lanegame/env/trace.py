"""Episode logs, aggregate statistics and a versioned JSON trace format.

Trace schema ("lanegame.episode", version 1):

{
  "schema": "lanegame.episode", "version": 1,
  "config": {...GameConfig fields...},
  "n_actions": 18, "actions": {"0": "SlowPush+Contest", ...},      # labels of the fixed action space
  "obs_size": 34, "features": ["turn_fraction", ...],
  "state_fields": ["ax", "ah", "ar", "asg", "bx", "bh", "br", "bsg", "va", "vb", "ap", "bp"],
  "episode": {
     "seed": ..., "k0": 0, "horizon": K, "names": ["aggressive", "passive_safe"],
     "payoff": total realized payoff to A, "rewards": [per turn, to A],
     "actions": [[a, b], ...], "farm": [A, B], "farm_available": [A, B], "plates": [A, B],
     "ganks": [A, B] (successful ganks suffered), "crash_resources": [A, B],
     "choice_default_used": false,
     "exposure_sum": [A, B], "trades_attempted": [A, B], "trades_connected": [A, B],
     "yields_threatened": [A, B], "recalls": [A, B], "recalls_interrupted": [A, B], "turns_away": [A, B],
     "trades_landed": [A, B], "trade_damage_dealt": [A, B], "tower_damage_taken": [A, B],
     "damage_taken": [A, B], "gank_damage_taken": [A, B], "deaths": [A, B], "bounty": [A, B],
                                                                   # champion mode only (else 0)
     "turns": [ {StepResult.info of turn k}, ... ]        # only when recorded
  }
}

Champion mode adds "champion_fields": ["pos", "hp", "absent"] and, per turn, the behaviors, chosen and
effective siege objectives, champion cells at the start and end, farm spots, Yield threats, Trade
attempts and contacts, modes, the champions' wave-contribution lines and the champion state before and
after. Each turn record carries the state before and after the turn, both actions and legal sets, the reward
and its decomposition (farm, plates, realized gank payoff, crash reward), farm available and collected,
plates claimed, gank probabilities, the uniform draws and the realized ganks, crash sizes and options
and the choices taken, exposure, siege objectives, vision changes, thinning and the engine's event
counters (spawned, merged, died, crashed, tower_hp, thinned_hp, plates, freeze_held, freeze_broken, ...).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import GameConfig
from ..io import STATE_FIELDS, dump_json
from .observation import feature_names, n_actions, obs_size

CHAMP_COUNTERS = ("recalls_interrupted", "trades_landed", "trade_damage_dealt", "tower_damage_taken", "damage_taken",
                  "gank_damage_taken", "deaths", "bounty")
TRACE_SCHEMA = "lanegame.episode"
TRACE_VERSION = 1


def _pair():
    return [0.0, 0.0]


@dataclass
class EpisodeLog:
    seed: object
    k0: int
    horizon: int
    names: Tuple[str, str] = ("A", "B")
    payoff: float = 0.0                              # realized total payoff to A
    rewards: List[float] = field(default_factory=list)
    actions: List[Tuple[int, int]] = field(default_factory=list)
    farm: List[float] = field(default_factory=_pair)
    farm_available: List[float] = field(default_factory=_pair)
    plates: List[float] = field(default_factory=_pair)
    ganks: List[int] = field(default_factory=lambda: [0, 0])
    crash_resources: List[float] = field(default_factory=_pair)
    choice_default_used: bool = False
    exposure_sum: List[int] = field(default_factory=lambda: [0, 0])
    trades_attempted: List[int] = field(default_factory=lambda: [0, 0])
    trades_connected: List[int] = field(default_factory=lambda: [0, 0])
    yields_threatened: List[int] = field(default_factory=lambda: [0, 0])
    recalls: List[int] = field(default_factory=lambda: [0, 0])
    recalls_interrupted: List[int] = field(default_factory=lambda: [0, 0])
    turns_away: List[int] = field(default_factory=lambda: [0, 0])     # turns off the lane (Recall turn included)
    trades_landed: List[int] = field(default_factory=lambda: [0, 0])  # connected trades that dealt damage
    trade_damage_dealt: List[int] = field(default_factory=lambda: [0, 0])
    tower_damage_taken: List[int] = field(default_factory=lambda: [0, 0])
    damage_taken: List[int] = field(default_factory=lambda: [0, 0])   # every source (trades, tower, ganks)
    gank_damage_taken: List[int] = field(default_factory=lambda: [0, 0])
    deaths: List[int] = field(default_factory=lambda: [0, 0])
    bounty: List[float] = field(default_factory=_pair)                 # kill bounty received
    turns: Optional[List[dict]] = None

    def add(self, result, record: bool = False) -> None:
        info = result.info
        self.payoff += result.rewards[0]
        self.rewards.append(result.rewards[0])
        self.actions.append(tuple(info["actions"]))
        for s in (0, 1):
            self.farm[s] += info["farm"][s]
            self.farm_available[s] += info["farm_available"][s]
            self.plates[s] += info["plates"][s]
            self.ganks[s] += int(info["ganked"][s])
            self.exposure_sum[s] += int(info["exposure"][s])
            if "trade_connected" in info:
                self.trades_attempted[s] += int(info["trade_attempted"][s])
                self.trades_connected[s] += int(info["trade_connected"][s])
                self.yields_threatened[s] += int(info["threatened"][s])
                self.recalls[s] += int(info["recalled"][s])
                self.recalls_interrupted[s] += int(info["recall_interrupted"][s])
                self.turns_away[s] += int(info["away"][s])
                self.trades_landed[s] += int(info["trade_damage"][s] > 0)
                self.trade_damage_dealt[s] += int(info["trade_damage"][s])
                self.tower_damage_taken[s] += int(info["tower_damage"][s])
                self.damage_taken[s] += int(info["damage_taken"][s])
                self.gank_damage_taken[s] += int(info["gank_damage"][s])
                self.deaths[s] += int(info["died"][s])
                self.bounty[s] += float(info["bounty"][s])
        oa, ob = info["crash_options"]
        i, j = info["choices"]
        self.crash_resources[0] += oa[i][0]
        self.crash_resources[1] += ob[j][0]
        self.choice_default_used = self.choice_default_used or bool(info["choice_default_used"])
        if record:
            if self.turns is None:
                self.turns = []
            self.turns.append(info)

    @property
    def length(self) -> int:
        return len(self.rewards)

    def to_dict(self) -> dict:
        d = {"seed": self.seed, "k0": self.k0, "horizon": self.horizon, "names": list(self.names),
             "payoff": self.payoff, "rewards": list(self.rewards), "actions": [list(a) for a in self.actions],
             "farm": list(self.farm), "farm_available": list(self.farm_available), "plates": list(self.plates),
             "ganks": list(self.ganks), "crash_resources": list(self.crash_resources),
             "choice_default_used": self.choice_default_used, "exposure_sum": list(self.exposure_sum),
             "trades_attempted": list(self.trades_attempted), "trades_connected": list(self.trades_connected),
             "yields_threatened": list(self.yields_threatened), "recalls": list(self.recalls),
             "recalls_interrupted": list(self.recalls_interrupted),
             "turns_away": list(self.turns_away), "trades_landed": list(self.trades_landed),
             "trade_damage_dealt": list(self.trade_damage_dealt), "tower_damage_taken": list(self.tower_damage_taken),
             "damage_taken": list(self.damage_taken), "gank_damage_taken": list(self.gank_damage_taken),
             "deaths": list(self.deaths), "bounty": list(self.bounty)}
        if self.turns is not None:
            d["turns"] = self.turns
        return d


def trace_dict(log: EpisodeLog, cfg: GameConfig) -> dict:
    d = {"schema": TRACE_SCHEMA, "version": TRACE_VERSION, "config": cfg.to_dict(),
         "n_actions": n_actions(cfg), "actions": {str(a): cfg.label(a) for a in cfg.action_codes()},
         "obs_size": obs_size(cfg), "features": feature_names(cfg), "state_fields": STATE_FIELDS}
    if cfg.champion_mode:
        d["champion_fields"] = ["pos", "hp", "absent"]
    d["episode"] = log.to_dict()
    return d


def write_trace(log: EpisodeLog, cfg: GameConfig, path: str) -> None:
    dump_json(trace_dict(log, cfg), path)


def read_trace(path: str) -> dict:
    import json
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    assert d.get("schema") == TRACE_SCHEMA, f"not a {TRACE_SCHEMA} file"
    d["config_obj"] = GameConfig.from_dict(d["config"])
    return d


@dataclass
class EpisodeStats:
    """Aggregate of several episodes between the same two agents."""
    names: Tuple[str, str]
    payoffs: np.ndarray                 # realized payoff to A per episode
    farm: np.ndarray                    # (n, 2)
    farm_available: np.ndarray          # (n, 2)
    plates: np.ndarray                  # (n, 2)
    ganks: np.ndarray                   # (n, 2)
    crash_resources: np.ndarray         # (n, 2)
    action_counts: Tuple[Dict[str, int], Dict[str, int]]
    turns: int
    elapsed: float
    obs_size: int
    n_actions: int
    choice_default_used: bool
    exposure: Optional[np.ndarray] = None          # (n, 2) mean champion exposure cell per turn
    trades_attempted: Optional[np.ndarray] = None  # (n, 2)
    trades_connected: Optional[np.ndarray] = None  # (n, 2)
    yields_threatened: Optional[np.ndarray] = None # (n, 2)
    recalls: Optional[np.ndarray] = None           # (n, 2)
    turns_away: Optional[np.ndarray] = None        # (n, 2)
    counters: Optional[Dict[str, np.ndarray]] = None   # champion-mode counters (n, 2) by CHAMP_COUNTERS name
    champion_mode: bool = False
    first_log: Optional[EpisodeLog] = None

    @classmethod
    def from_logs(cls, logs: Sequence[EpisodeLog], elapsed: float, env) -> "EpisodeStats":
        counts = (defaultdict(int), defaultdict(int))
        for log in logs:
            for a, b in log.actions:
                counts[0][env.cfg.label(a)] += 1
                counts[1][env.cfg.label(b)] += 1
        return cls(names=logs[0].names if logs else ("A", "B"),
                   payoffs=np.array([l.payoff for l in logs], dtype=float),
                   farm=np.array([l.farm for l in logs], dtype=float).reshape(len(logs), 2),
                   farm_available=np.array([l.farm_available for l in logs], dtype=float).reshape(len(logs), 2),
                   plates=np.array([l.plates for l in logs], dtype=float).reshape(len(logs), 2),
                   ganks=np.array([l.ganks for l in logs], dtype=float).reshape(len(logs), 2),
                   crash_resources=np.array([l.crash_resources for l in logs], dtype=float).reshape(len(logs), 2),
                   action_counts=(dict(counts[0]), dict(counts[1])),
                   turns=sum(l.length for l in logs), elapsed=elapsed,
                   obs_size=env.obs_size, n_actions=env.n_actions,
                   choice_default_used=any(l.choice_default_used for l in logs),
                   exposure=np.array([[e / max(l.length, 1) for e in l.exposure_sum] for l in logs], dtype=float).reshape(len(logs), 2),
                   trades_attempted=np.array([l.trades_attempted for l in logs], dtype=float).reshape(len(logs), 2),
                   trades_connected=np.array([l.trades_connected for l in logs], dtype=float).reshape(len(logs), 2),
                   yields_threatened=np.array([l.yields_threatened for l in logs], dtype=float).reshape(len(logs), 2),
                   recalls=np.array([l.recalls for l in logs], dtype=float).reshape(len(logs), 2),
                   turns_away=np.array([l.turns_away for l in logs], dtype=float).reshape(len(logs), 2),
                   counters={c: np.array([getattr(l, c) for l in logs], dtype=float).reshape(len(logs), 2)
                             for c in CHAMP_COUNTERS},
                   champion_mode=bool(env.cfg.champion_mode),
                   first_log=next((l for l in logs if l.turns is not None), None))

    @property
    def episodes(self) -> int:
        return int(len(self.payoffs))

    @property
    def mean(self) -> float:
        return float(self.payoffs.mean())

    @property
    def stderr(self) -> float:
        n = len(self.payoffs)
        return float(self.payoffs.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")

    @property
    def episodes_per_second(self) -> float:
        return self.episodes / self.elapsed if self.elapsed > 0 else float("inf")

    @property
    def turns_per_second(self) -> float:
        return self.turns / self.elapsed if self.elapsed > 0 else float("inf")

    def action_frequencies(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        n = max(self.turns, 1)
        return tuple({k: v / n for k, v in sorted(c.items())} for c in self.action_counts)

    def summary(self) -> dict:
        q = np.quantile(self.payoffs, [0.05, 0.25, 0.5, 0.75, 0.95]) if self.episodes else [float("nan")] * 5
        return {
            "names": list(self.names), "episodes": self.episodes, "turns": self.turns,
            "mean": self.mean, "stderr": self.stderr,
            "std": float(self.payoffs.std(ddof=1)) if self.episodes > 1 else float("nan"),
            "p_positive": float(np.mean(self.payoffs > 0)) if self.episodes else float("nan"),
            "quantiles": dict(zip(["q05", "q25", "q50", "q75", "q95"], map(float, q))),
            "mean_farm": self.farm.mean(axis=0).tolist(), "mean_farm_available": self.farm_available.mean(axis=0).tolist(),
            "mean_plates": self.plates.mean(axis=0).tolist(), "mean_ganks": self.ganks.mean(axis=0).tolist(),
            "mean_crash_resources": self.crash_resources.mean(axis=0).tolist(),
            "action_frequencies": [dict(f) for f in self.action_frequencies()],
            "choice_default_used": self.choice_default_used,
            "mean_exposure": self.exposure.mean(axis=0).tolist(),
            "mean_trades_attempted": self.trades_attempted.mean(axis=0).tolist(),
            "mean_trades_connected": self.trades_connected.mean(axis=0).tolist(),
            "mean_yields_threatened": self.yields_threatened.mean(axis=0).tolist(),
            "mean_recalls": self.recalls.mean(axis=0).tolist(),
            "mean_turns_away": self.turns_away.mean(axis=0).tolist(),
            **{f"mean_{c}": self.counters[c].mean(axis=0).tolist() for c in CHAMP_COUNTERS},
            "champion_mode": self.champion_mode,
            "elapsed_s": self.elapsed, "episodes_per_second": self.episodes_per_second,
            "turns_per_second": self.turns_per_second, "obs_size": self.obs_size, "n_actions": self.n_actions,
        }

    def format(self) -> str:
        s = self.summary()
        fa, fb = s["action_frequencies"]
        fmt = lambda d: ", ".join(f"{k} {v:.2f}" for k, v in d.items() if v >= 0.005)
        lines = [
            f"{s['names'][0]} (A) vs {s['names'][1]} (B): {s['episodes']} episodes, {s['turns']} turns",
            f"  payoff to A: mean {s['mean']:+.4f} +- {s['stderr']:.4f} (std {s['std']:.3f}; P(A ahead) {s['p_positive']:.3f}; "
            f"q05 {s['quantiles']['q05']:+.1f}, q50 {s['quantiles']['q50']:+.1f}, q95 {s['quantiles']['q95']:+.1f})",
            f"  farm collected A {s['mean_farm'][0]:.3f} B {s['mean_farm'][1]:.3f} (available {s['mean_farm_available'][0]:.3f} / "
            f"{s['mean_farm_available'][1]:.3f}); plates A {s['mean_plates'][0]:.3f} B {s['mean_plates'][1]:.3f}; "
            f"crash resources A {s['mean_crash_resources'][0]:.3f} B {s['mean_crash_resources'][1]:.3f}",
            f"  ganks suffered per game A {s['mean_ganks'][0]:.3f} B {s['mean_ganks'][1]:.3f}; "
            f"mean exposure cell A {s['mean_exposure'][0]:+.2f} B {s['mean_exposure'][1]:+.2f}",
            f"  actions A: {fmt(fa)}",
            f"  actions B: {fmt(fb)}",
            *([f"  trades per game attempted A {s['mean_trades_attempted'][0]:.2f} B {s['mean_trades_attempted'][1]:.2f}, "
               f"connected A {s['mean_trades_connected'][0]:.2f} B {s['mean_trades_connected'][1]:.2f}; "
               f"Yield threatened A {s['mean_yields_threatened'][0]:.2f} B {s['mean_yields_threatened'][1]:.2f}",
               f"  recalls per game A {s['mean_recalls'][0]:.2f} B {s['mean_recalls'][1]:.2f}; "
               f"turns away A {s['mean_turns_away'][0]:.2f} B {s['mean_turns_away'][1]:.2f}",
               f"  trades landed A {s['mean_trades_landed'][0]:.2f} B {s['mean_trades_landed'][1]:.2f}; "
               f"trade damage dealt A {s['mean_trade_damage_dealt'][0]:.2f} B {s['mean_trade_damage_dealt'][1]:.2f}; "
               f"tower damage taken A {s['mean_tower_damage_taken'][0]:.2f} B {s['mean_tower_damage_taken'][1]:.2f}; "
               f"HP lost A {s['mean_damage_taken'][0]:.2f} B {s['mean_damage_taken'][1]:.2f} "
               f"(to ganks A {s['mean_gank_damage_taken'][0]:.2f} B {s['mean_gank_damage_taken'][1]:.2f}); "
               f"deaths A {s['mean_deaths'][0]:.2f} B {s['mean_deaths'][1]:.2f}"]
              if s["champion_mode"] else []),
            f"  observation size {s['obs_size']}, action space {s['n_actions']}"
            + ("; crash choices by the greedy default" if s["choice_default_used"] else ""),
            f"  {s['elapsed_s']:.2f} s: {s['episodes_per_second']:,.0f} episodes/s, {s['turns_per_second']:,.0f} turns/s",
        ]
        return "\n".join(lines)
