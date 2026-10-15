"""Behaviour diagnostics of M4.5c checkpoints (pilot scale: descriptive readouts that catch obvious learning problems,
not evidence of strategic competence).

Two existing instruments are reused unchanged:

  records   `m44.records.play_records`: per-turn, per-side decision state, the policy's probability masses and the
            outcome, through the real environment (LaneEnv). On this preset two record fields need care:
              * Ward (code 30) decodes as "Last Hit + Hold + Vision" in the M4.4 decomposition. Because Vision is no
                objective in M4.5 (code 30 is the only code with objective Vision), P(Ward) = po2 exactly, and the
                tables below use LH = pw0 - po2 and Hold = pb0 - po2; sampled codes are classified by
                `actions.category` (Ward its own category).
              * `epos` / `dist` treat the Vision node W (cell 99) as a lane cell; no table here uses them, and "at W"
                is read from `pos == 99`.
  ledger    the M4.5b engine (`m45b.engine.play` with `m45b.reference.NetPolicy`): wards chosen / placed /
            interrupted, minion fire on champions by cause, out-of-reach farm, freeze phases, deaths, recalls, on the
            environment's chance stream with integer seeds.

`conditional()` tabulates, for every decision-state class of STATE_CLASSES, the mean probability mass of each action
category and behaviour (and the sampled shares), so "the policy plays X" is always read with the state it plays X in.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ....config import CHAMP_ABSENT
from . import actions as AC
from .common import BEHAVIOR_SEED, LEDGER_SEEDS, agent_from_spec, env_cfg, run_jobs

WARD_CELL = 99
BIG_FRONT = 24                     # HP: a large enemy front (two stacked waves at level 1; M4.5b's "24-HP stack")

STATE_CLASSES: Dict[str, Callable] = {
    "all present decisions": lambda d, s: np.ones(len(d["present"]), bool),
    "waves meet on my half (<= -1)": lambda d, s: d["wave_pos"][:, s] <= -1,
    "waves meet at the lane cell (0)": lambda d, s: d["wave_pos"][:, s] == 0,
    "waves meet on the enemy half (>= 1)": lambda d, s: d["wave_pos"][:, s] >= 1,
    "stronger (power ratio >= 1.25)": lambda d, s: (d["power"][:, s] >= 1.25) & (d["en_present"][:, s] > 0),
    "even power": lambda d, s: (d["power"][:, s] > 0.8) & (d["power"][:, s] < 1.25) & (d["en_present"][:, s] > 0),
    "weaker (power ratio <= 0.8)": lambda d, s: (d["power"][:, s] <= 0.8) & (d["en_present"][:, s] > 0),
    "HP < 35%": lambda d, s: d["hp_frac"][:, s] < 0.35,
    "HP 35-70%": lambda d, s: (d["hp_frac"][:, s] >= 0.35) & (d["hp_frac"][:, s] < 0.7),
    "HP >= 70%": lambda d, s: d["hp_frac"][:, s] >= 0.7,
    "level 1-5": lambda d, s: d["level"][:, s] <= 5,
    "level 6-10": lambda d, s: (d["level"][:, s] >= 6) & (d["level"][:, s] <= 10),
    "level 11-18": lambda d, s: d["level"][:, s] >= 11,
    f"large enemy front (>= {BIG_FRONT} HP)": lambda d, s: d["en_fh"][:, s] >= BIG_FRONT,
    "enemy wave much stronger (window -4 or less)": lambda d, s: d["wave_adv"][:, s] <= -4,
    "holding a freeze (freezer)": lambda d, s: d["held"][:, s] > 0,
    "frozen on": lambda d, s: d["held_on"][:, s] > 0,
    "enemy champion away": lambda d, s: d["en_present"][:, s] == 0,
    "own crash under the enemy tower, Pressure legal": lambda d, s: d["obj_legal"][:, s] > 0,
    "enemy crash under my tower": lambda d, s: d["en_siege"][:, s] > 0,
    "Ward legal, own vision down": lambda d, s: ward_legal(d, s) & (d["vision"][:, s] == 0),
    "Ward legal, own vision up": lambda d, s: ward_legal(d, s) & (d["vision"][:, s] > 0),
    "Recall legal": lambda d, s: d["recall_legal"][:, s] > 0,
    "standing at W": lambda d, s: d["pos"][:, s] == WARD_CELL,
}


def ward_legal(d: Dict[str, np.ndarray], s: int) -> np.ndarray:
    """champions.ward_legal from a record: present and within champion_speed (2) edges of W, i.e. own cell -1, 0, +1
    or W itself (tested against the engine's legal sets)."""
    p = d["pos"][:, s]
    return (d["present"][:, s] > 0) & ((np.abs(p) <= 1) | (p == WARD_CELL))


def masses(d: Dict[str, np.ndarray], s: int, m: np.ndarray) -> Optional[Dict[str, float]]:
    """Mean probability masses of the focal side over the turns `m` (present, with a recorded distribution)."""
    m = m & np.isfinite(d["pw0"][:, s])
    if not m.any():
        return None
    ward = d["po2"][m, s]
    out = {"LH": float(np.mean(d["pw0"][m, s] - ward)), "TN": float(np.mean(d["pw1"][m, s])),
           "SL": float(np.mean(d["pw2"][m, s])), "SH": float(np.mean(d["pw3"][m, s])), "Ward": float(np.mean(ward)),
           "Recall": float(np.mean(d["prec"][m, s]))}
    lane = 1.0 - ward - d["prec"][m, s]
    with np.errstate(invalid="ignore", divide="ignore"):
        beh = {"Hold": (d["pb0"][m, s] - ward) / lane, "Yield": d["pb1"][m, s] / lane, "Trade": d["pb2"][m, s] / lane,
               "Safe": d["pb3"][m, s] / lane}
    for k, v in beh.items():
        out[f"beh_{k}"] = float(np.nanmean(np.where(lane > 1e-9, v, np.nan)))
    out["Pressure"] = float(np.mean(d["po1"][m, s]))
    out["entropy"] = float(np.mean(d["entropy"][m, s]))
    return out


def conditional(records: Sequence, focal: Optional[str] = None) -> Dict[str, dict]:
    """STATE_CLASSES x (n, mean masses, sampled category shares) over the focal sides of `records`."""
    acc: Dict[str, dict] = {k: {"n": 0, "mass": [], "w": [], "codes": []} for k in STATE_CLASSES}
    for rec in records:
        d = rec.data
        for s in (0, 1):
            if focal is not None and rec.names[s] != focal:
                continue
            pres = (d["present"][:, s] > 0) & (d["a"][:, s] != CHAMP_ABSENT)
            for k, f in STATE_CLASSES.items():
                m = pres & f(d, s)
                n = int(m.sum())
                if not n:
                    continue
                acc[k]["n"] += n
                acc[k]["codes"].append(d["a"][m, s].astype(int))
                ms = masses(d, s, m)
                if ms is not None:
                    acc[k]["mass"].append(ms)
                    acc[k]["w"].append(n)
    out = {}
    for k, v in acc.items():
        row = {"n": v["n"]}
        if v["mass"]:
            w = np.asarray(v["w"], float)
            keys = v["mass"][0].keys()
            row["mass"] = {kk: float(np.nansum([x[kk] * wi for x, wi in zip(v["mass"], w)]) / w.sum()) for kk in keys}
        if v["codes"]:
            row["sampled"] = AC.histogram(np.concatenate(v["codes"]))
        out[k] = row
    return out


def dominance(cond: Dict[str, dict]) -> dict:
    """Does one category dominate everywhere? The largest category mass per state class, and the spread of each
    category's mass across the classes with at least 200 decisions (max - min)."""
    cats = ("LH", "TN", "SL", "SH", "Ward", "Recall")
    rows = {k: v["mass"] for k, v in cond.items() if v.get("mass") and v["n"] >= 200}
    top = {k: max(cats, key=lambda c: m[c]) for k, m in rows.items()}
    spread = {c: float(max(m[c] for m in rows.values()) - min(m[c] for m in rows.values())) for c in cats} if rows else {}
    return {"top_category_by_state": top, "top_share_by_state": {k: float(rows[k][top[k]]) for k in rows},
            "mass_spread_across_states": spread}


def game_summary(records: Sequence, focal: Optional[str] = None) -> dict:
    """Per-1000-turn rates and economic outcomes of the focal sides (over games, with 95% intervals)."""
    from .common import mean_ci
    keys = {"wards_per_1000": lambda d, s: 1000 * float((d["a"][:, s] == AC.WARD).mean()),
            "wards_placed_per_1000": lambda d, s: 1000 * float(d["vision_set"][:, s].mean()),
            "recalls_per_1000": lambda d, s: 1000 * float(d["recalled"][:, s].mean()),
            "deaths_per_1000": lambda d, s: 1000 * float(d["died"][:, s].mean()),
            "trade_attempts_per_1000": lambda d, s: 1000 * float(d["trade_att"][:, s].mean()),
            "trades_connected_per_1000": lambda d, s: 1000 * float(d["trade_conn"][:, s].mean()),
            "pressure_turns_per_1000": lambda d, s: 1000 * float(d["pressure_eff"][:, s].mean()),
            "away_share": lambda d, s: float(d["away"][:, s].mean()),
            "collection_rate": lambda d, s: float(d["farm"][:, s].sum() / max(d["farm_avail"][:, s].sum(), 1e-9)),
            "earned_per_100_turns": lambda d, s: 100 * float(d["earned"][:, s].mean()),
            "lead_per_100_turns": lambda d, s: 100 * float((d["earned"][:, s] - d["earned"][:, 1 - s]).mean()),
            "xp_missed_per_100_turns": lambda d, s: 100 * float(d["xp_missed"][:, s].mean()),
            "level_end": lambda d, s: float(d["level"][-1, s]),
            "freeze_turns_share": lambda d, s: float(d["held"][:, s].mean()),
            "frozen_on_share": lambda d, s: float(d["held_on"][:, s].mean()),
            "crashes_per_100_turns": lambda d, s: 100 * float((d["crash"][:, s] > 0).mean())}
    vals: Dict[str, List[float]] = {k: [] for k in keys}
    for rec in records:
        for s in (0, 1):
            if focal is not None and rec.names[s] != focal:
                continue
            for k, f in keys.items():
                vals[k].append(f(rec.data, s))
    return {k: list(mean_ci(v)) for k, v in vals.items()}


# ---------------------------------------------------------------------------------------------- jobs
def records_job(args) -> tuple:
    """(spec, opponent spec or 'self', games, turns, seed) -> conditional table, dominance, game summary and the
    M4.4 freeze / recall metrics, focal = the policy."""
    spec, opp, games, turns, seed = args
    import torch
    torch.set_num_threads(1)
    from ..m44 import metrics as M
    from ..m44.records import play_records
    cfg = env_cfg()
    x = agent_from_spec(spec, cfg)
    y = x if opp == "self" else agent_from_spec(opp, cfg)
    t0 = time.perf_counter()
    recs = play_records(cfg, x, y, games, turns, seed, both_seats=opp != "self", env_label="m45c")
    focal = x.name
    cond = conditional(recs, focal)
    return (spec, opp), {"games": len(recs), "turns": int(sum(r.turns for r in recs)), "conditional": cond,
                         "dominance": dominance(cond), "summary": game_summary(recs, focal),
                         "freezes": M.freezes(recs, focal), "recalls": M.recalls(recs, focal),
                         "elapsed_s": time.perf_counter() - t0}


LEDGER_KEYS = ("earned", "farm", "unc", "farm_avail", "xp", "xp_lost", "d_trade", "d_min", "d_min_ret", "d_min_pos",
               "d_tower", "d_gank", "trade_att", "trade_conn", "trade_dealt", "deaths", "recalls", "rec_intr", "away",
               "wards", "ward_placed", "ward_intr", "vis_turns", "vis_exposed", "ganks", "held", "frozen", "crashes",
               "plates", "capped", "n_hold", "n_yield", "n_trade", "n_safe", "n_recall", "n_LH", "n_TN", "n_SL",
               "n_SH", "at_W", "level_end")


def ledger_job(args) -> tuple:
    """(spec, opponent spec or 'self', seeds, turns): the M4.5b ledger of the policy (seat A, and seat B on the same
    seeds against a fixed opponent; both seats in self-play)."""
    spec, opp, seeds, turns = args
    import torch
    torch.set_num_threads(1)
    from ...m45b import engine as E
    from ...m45b import states as S
    from ...m45b.reference import NetPolicy
    from .common import mean_ci
    cfg = env_cfg()
    x = agent_from_spec(spec, cfg)
    start = S.s0(cfg)
    vals: Dict[str, List[float]] = {k: [] for k in LEDGER_KEYS + ("G",)}
    t0 = time.perf_counter()

    def pol(agent, sd, seat):
        if hasattr(agent, "strategy"):
            return agent.strategy                                  # a StrategyAgent: its Contingent strategy
        return NetPolicy(agent, 0, sd, seat)

    for sd in seeds:
        if opp == "self":
            g = E.play(cfg, start, pol(x, sd, 0), pol(x, sd, 1), sd, turns, check_legal=False)["summary"]
            for side in ("A", "B"):
                for k in LEDGER_KEYS:
                    vals[k].append(g[f"{k}_{side}"])
            vals["G"].append(g[f"G{turns}"] if f"G{turns}" in g else float("nan"))
        else:
            y = agent_from_spec(opp, cfg)
            for seat in (0, 1):
                pa, pb = (pol(x, sd, 0), pol(y, sd, 1)) if seat == 0 else (pol(y, sd, 0), pol(x, sd, 1))
                g = E.play(cfg, start, pa, pb, sd, turns, check_legal=False)["summary"]
                side = "AB"[seat]
                for k in LEDGER_KEYS:
                    vals[k].append(g[f"{k}_{side}"])
                gv = g.get(f"G{turns}", float("nan"))
                vals["G"].append(gv if seat == 0 else -gv)
    per = {k: [v * 100.0 / turns for v in vs] if k not in ("level_end", "G") else vs for k, vs in vals.items()}
    return (spec, opp), {"turns": turns, "seeds": len(seeds), "per_100_turns": {k: list(mean_ci(v)) for k, v in per.items()},
                         "elapsed_s": time.perf_counter() - t0}


def run_records(jobs: list, workers: int) -> Dict[str, dict]:
    return {f"{k[0]}|{k[1]}": v for k, v in run_jobs(records_job, jobs, workers, "behaviour records")}


def run_ledger(jobs: list, workers: int) -> Dict[str, dict]:
    return {f"{k[0]}|{k[1]}": v for k, v in run_jobs(ledger_job, jobs, workers, "ledger")}


__all__ = ["STATE_CLASSES", "ward_legal", "masses", "conditional", "dominance", "game_summary", "records_job",
           "ledger_job", "run_records", "run_ledger", "LEDGER_KEYS", "BIG_FRONT", "WARD_CELL", "BEHAVIOR_SEED",
           "LEDGER_SEEDS"]
