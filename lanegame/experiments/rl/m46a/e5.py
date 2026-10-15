"""E5: the price of a reset and of HP (pre-registration §2 E5). Natural states of each final's self-play, K = 12 paired
rollouts, horizon 50, Q50 primary, protocol "policy".

    recall_full / recall_mid   candidates on the original state (policy, 1t Recall, 1t stay = Last Hit + Hold,
                               1t Shove + Hold) and on the "inv" variant (the focal side's unspent gold already
                               invested: policy, Recall, stay)
    hp_early / hp_late         the state as is against the focal HP lowered by 0.3 bar, both continued by the policy
"""

from __future__ import annotations

import time

from ....config import B_HOLD, B_RECALL, W_HOLD, W_SHOVE, champ_act
from . import families as FA
from .common import E5_STATE_ROOT, FINALS, ROLL_ROOT, load_json, meta, run_jobs, save_json
from .states import collect_job

RECALL = champ_act(W_HOLD, B_RECALL)
STAY = champ_act(W_HOLD, B_HOLD)
SHOVE = champ_act(W_SHOVE, B_HOLD)
N_STATES, K, H = 48, 12, 50
RECALL_CANDS = {"policy": ("variant", "orig", ("policy",)), "recall": ("variant", "orig", ("fixed", RECALL, 1)),
                "stay": ("variant", "orig", ("fixed", STAY, 1)), "shove": ("variant", "orig", ("fixed", SHOVE, 1)),
                "inv:policy": ("variant", "inv", ("policy",)), "inv:recall": ("variant", "inv", ("fixed", RECALL, 1)),
                "inv:stay": ("variant", "inv", ("fixed", STAY, 1))}
HP_CANDS = {"policy": ("variant", "orig", ("policy",)), "hp-0.3": ("variant", "hp-0.3", ("policy",))}
FAMILIES = {"recall_full": RECALL_CANDS, "recall_mid": RECALL_CANDS, "hp_early": HP_CANDS, "hp_late": HP_CANDS}
OFFSETS = {"recall_full": 0, "recall_mid": 1000, "hp_early": 2000, "hp_late": 3000}


def collect(workers: int, finals=FINALS) -> dict:
    js = [(s, f, N_STATES, E5_STATE_ROOT, OFFSETS[f], 1) for s in finals for f in FAMILIES]
    res = run_jobs(collect_job, js, workers, "E5 states")
    return {k: v for k, v in res}


def run(workers: int, finals=FINALS) -> dict:
    t0 = time.perf_counter()
    col = collect(workers, finals)
    jobs, desc = [], {}
    for fam, cands in FAMILIES.items():
        states = {s: [x["start"] for x in col[(s, fam)]["states"]] for s in finals}
        val = {s: FA.validation_indices(f"E5:{fam}", len(v)) for s, v in states.items()}
        jobs += FA.build_jobs("e5", fam, states, cands, ("policy",), K, H, ROLL_ROOT, val)
        desc[fam] = {str(s): {"n": len(col[(s, fam)]["states"]), "games": col[(s, fam)]["games"],
                              "validation": val[s],
                              "states": [{"seed": x["seed"], "turn": x["turn"], "seat": x["seat"]}
                                         for x in col[(s, fam)]["states"]]} for s in finals}
    res = FA.run(jobs, workers, "E5 rollouts")
    out = analyse(res)
    save_json("e5_summary", {"meta": meta(stage="e5", K=K, horizon=H, n_states=N_STATES), "states": desc,
                             "analysis": out, "elapsed_s": time.perf_counter() - t0})
    return out


def analyse(res: dict) -> dict:
    import numpy as np
    from . import engine as EN
    from .common import mean_ci
    out = {}
    for fam in ("recall_full", "recall_mid"):
        a = {}
        for stat in ("Q", "G"):
            for c in ("recall", "stay", "shove", "inv:policy"):
                a[f"{c}-policy|{stat}"] = FA.contrast(res, fam, "policy", c, stat, H)
            a[f"recall-stay|{stat}"] = FA.contrast(res, fam, "policy", "recall", stat, H, base="stay")
            a[f"inv:recall-inv:stay|{stat}"] = FA.contrast(res, fam, "policy", "inv:recall", stat, H, base="inv:stay")
            # the shopping component: (recall - stay | orig) - (recall - stay | inv), per state then per final
            per = {}
            for s, recs in FA.records(res, fam, "policy").items():
                v = [EN.per_state_diff(r, "recall", stat, H, "stay") - EN.per_state_diff(r, "inv:recall", stat, H,
                                                                                         "inv:stay") for r in recs]
                v = [x for x in v if np.isfinite(x)]
                if v:
                    per[s] = float(np.mean(v))
            m, h = mean_ci(list(per.values()))
            a[f"shopping|{stat}"] = {"mean": m, "ci": h, "per_final": per}
        for key in ("away", "deaths", "farm", "earned", "unc", "xp_lost", "d_total"):
            a[f"ledger:recall-policy:{key}@20"] = FA.ledger_contrast(res, fam, "policy", "recall", f"{key}_A", 20)
            a[f"ledger:recall-policy:{key}_B@20"] = FA.ledger_contrast(res, fam, "policy", "recall", f"{key}_B", 20)
        a["validation_Q"] = FA.validate(res, fam, "Q", H, list(RECALL_CANDS))
        out[fam] = a
    for fam in ("hp_early", "hp_late"):
        a = {}
        for stat in ("Q", "G"):
            c = FA.contrast(res, fam, "policy", "hp-0.3", stat, H)
            a[f"hp-0.3|{stat}"] = c
            a[f"price_per_bar|{stat}"] = {"mean": -c["mean"] / 0.3, "ci": c["ci"] / 0.3 if c["ci"] == c["ci"] else c["ci"]}
        for key in ("recalls", "deaths", "away", "earned", "d_total"):
            a[f"ledger:hp-0.3:{key}@20"] = FA.ledger_contrast(res, fam, "policy", "hp-0.3", f"{key}_A", 20)
            a[f"ledger:hp-0.3:{key}@50"] = FA.ledger_contrast(res, fam, "policy", "hp-0.3", f"{key}_A", 50)
        a["validation_Q"] = FA.validate(res, fam, "Q", H, list(HP_CANDS))
        out[fam] = a
    return out


__all__ = ["run", "analyse", "collect", "FAMILIES"]


def extend(workers: int, families=("recall_full",), finals=FINALS) -> dict:
    """Exception 3 (pre-registration §1): a family that failed the pooled validation is re-run at 150 turns for all
    its states (the same states and rollout seeds: the first 50 turns are the 50-turn rollouts themselves)."""
    t0 = time.perf_counter()
    col = collect(workers, finals)
    jobs = []
    for fam in families:
        states = {s: [x["start"] for x in col[(s, fam)]["states"]] for s in finals}
        jobs += FA.build_jobs("e5", fam, states, FAMILIES[fam], ("policy",), K, H, ROLL_ROOT, None, extended=True)
    res = FA.run(jobs, workers, "E5 extended")
    out = {}
    for fam in families:
        a = {}
        for stat in ("Q", "G"):
            for h in (50, 150):
                a[f"recall-stay|{stat}{h}"] = FA.contrast(res, fam, "policy", "recall", stat, h, base="stay")
                a[f"recall-policy|{stat}{h}"] = FA.contrast(res, fam, "policy", "recall", stat, h)
                a[f"inv:recall-inv:stay|{stat}{h}"] = FA.contrast(res, fam, "policy", "inv:recall", stat, h,
                                                                   base="inv:stay")
        # 50 vs 150 on the same trajectories, per (state, candidate) pair
        import numpy as np
        from . import engine as EN
        d50, d150 = [], []
        for recs in FA.records(res, fam, "policy").values():
            for r in recs:
                for c in FAMILIES[fam]:
                    if c == "policy":
                        continue
                    d50.append(EN.per_state_diff(r, c, "Q", 50))
                    d150.append(EN.per_state_diff(r, c, "Q", 150))
        d50, d150 = np.asarray(d50), np.asarray(d150)
        a["50_vs_150"] = {"pairs": int(len(d50)), "mean_change": float(np.mean(d150 - d50)),
                          "slope": float(np.polyfit(d50, d150, 1)[0]) if len(d50) > 2 else float("nan")}
        out[fam] = a
    d = load_json("e5_summary")
    d["extended"] = {"meta": meta(stage="e5_extend", families=list(families), horizon=FA.EXTENDED_H), "analysis": out,
                     "elapsed_s": time.perf_counter() - t0}
    save_json("e5_summary", d)
    return out
