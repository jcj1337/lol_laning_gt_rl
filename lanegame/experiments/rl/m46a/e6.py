"""E6: credit assignment (pre-registration §2 E6), from E2's raw job records and E3's critic aggregates.

    snr      per E2 family / protocol / plan: the mean paired effect (candidate - policy, G at the family horizon), the
             mean over states of the per-rollout SD of that difference (the noise of one paired rollout), their ratio,
             the effect per forced turn, and the number of paired rollouts a single state would need for a 2-SE
             resolution of its effect
    critic   E3: the evaluated critics' bias, error and explained variance on Trade-connected, Trade-unconnected and
             other turns
"""

from __future__ import annotations

import os

import numpy as np

from . import engine as EN
from .common import CKPT, load_gz, load_json, mean_ci, meta, save_json


def _records(stage: str):
    root = os.path.join(CKPT, stage)
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.endswith(".json.gz"):
                yield load_gz(os.path.join(dp, f))


def snr(stage: str = "e2", h: int = 60, select=None) -> dict:
    acc = {}
    for rec in _records(stage):
        if select is not None and not select(rec):
            continue
        base = rec["baseline"]
        for c in rec["G"]:
            if c == base or str(h) not in rec["G"][c]:
                continue
            d = np.asarray(rec["G"][c][str(h)]) - np.asarray(rec["G"][base][str(h)])
            if len(d) < 2:
                continue
            a = acc.setdefault((rec["family"], rec["proto"], c), {"eff": [], "sd": [], "forced": []})
            a["eff"].append(float(d.mean()))
            a["sd"].append(float(d.std(ddof=1)))
            a["forced"].append(float(np.mean(rec["forced"].get(c, [0]))))
    out = {}
    for (fam, proto, c), a in acc.items():
        eff, sd, fo = float(np.mean(a["eff"])), float(np.mean(a["sd"])), float(np.mean(a["forced"]))
        out[f"{fam}|{proto}|{c}"] = {"states": len(a["eff"]), "effect": eff, "rollout_sd": sd,
                                     "snr": abs(eff) / sd if sd > 0 else float("nan"),
                                     "forced_turns": fo, "effect_per_forced_turn": eff / fo if fo > 0 else float("nan"),
                                     "rollouts_for_2se": (2 * sd / abs(eff)) ** 2 if eff != 0 else float("inf")}
    return out


def critic_by_gap() -> dict:
    """E3's critic aggregates re-pooled by power gap (finals and levels pooled; both seats)."""
    acc = {}
    root = os.path.join(CKPT, "e3")
    for dp, _, fs in os.walk(root):
        for f in fs:
            if not f.endswith(".json.gz"):
                continue
            rec = load_gz(os.path.join(dp, f))
            for gm in rec["games"]:
                for seat, d in gm["e6"].items():
                    role = "equal" if rec["gap"] == "equal" else ("stronger" if int(seat) == rec["orient"] else "weaker")
                    for c, v in d.items():
                        a = acc.setdefault((rec["gap"], role, c), np.zeros(6))
                        a += np.asarray(v)
    out = {}
    for (gap, role, c), v in sorted(acc.items()):
        n, se, sa, ss, st, st2 = v
        var_t, var_e = st2 / n - (st / n) ** 2, ss / n - (se / n) ** 2
        out[f"{gap}|{role}|{c}"] = {"n": int(n), "bias": se / n, "mae": sa / n, "rmse": float(np.sqrt(ss / n)),
                                    "target_sd": float(np.sqrt(max(var_t, 0))),
                                    "explained_variance": float(1 - var_e / var_t) if var_t > 0 else float("nan")}
    return out


def run() -> dict:
    e3 = load_json("e3_summary")
    first = lambda r: r["horizon"] != 150             # the exception-3 re-runs repeat the same states
    data = {"meta": meta(stage="e6"), "snr_e2": snr("e2", 60, first), "snr_e5": snr("e5", 50, first),
            "critic_e3": (e3 or {}).get("summary", {}).get("e6"), "critic_by_gap": critic_by_gap()}
    save_json("e6_summary", data)
    return data


__all__ = ["run", "snr"]
