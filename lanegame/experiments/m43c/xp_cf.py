"""Part B / D screen: would counting XP in the objective (or ending the game at the final plate) change the M4.3b
conclusions about withholding wave damage?

Two stages, both on the M4.3b finals and opponents of the wave-only validation (`rl.m43b.validation`), both paired on
common random numbers, both scoring every objective on the SAME rollouts through the channels of `common.channels`:

  rescore   the stored states of `results/rl/m43b/validation/wave_cf.json.gz` (`base` seeds 0-2), re-rolled with the
            study's own roots (VAL_ROLL_SEED + 100 i + 10 j + l, key = state index) for the candidates policy, m:hold,
            m:thin, m:slow_push, m:freeze and f:freeze. The gold channel reproduces the stored advantages exactly (the
            stage records the largest deviation). It also records the phase of each state (turn, levels, plates).
  early     NEW states of the same situations, restricted to the laning phase the Part C proposal keeps: no tower has
            lost its final plate and neither champion is at the level cap (`EARLY` mask), sampled from games of the
            policy against the cell's opponent (roots XP_CF_SEED + ...). Same candidates; the returns are computed both
            over the 300-turn window and truncated at the Part C termination (the turn a tower's final plate falls).

Objectives (design document §B): J = G + lambda X (lambda in common.LAMBDAS, resources per credited XP) and
J = G + mu Lam (mu in common.MUS, resources per level of level progress). The continuation policy and the opponent were
trained on G alone: this is a first-order screen of the objective (does re-pricing the same play change the sign of a
deviation?), not the value of a policy re-trained on the new objective.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from .common import LAMBDAS, MUS, XP_CF_SEED, cfg_game, mean_ci, meta, save_json

SPECS = ("final:base:0", "final:base:1", "final:base:2", "final:ent03:0")
RESCORE_SPECS = ("final:base:0", "final:base:1", "final:base:2")
OPPONENTS = ("self", "farm", "push_punish", "clear_punish")
SITUATIONS = ("neutral", "enemy_pushing", "freeze_like")
MARGINAL = ("hold", "thin", "slow_push", "freeze")
FULL = ("freeze",)
WINDOW = 300
N_STATES, N_ROLL = 8, 16
CHANNELS = ("G", "X", "Lam", "G_T", "X_T", "Lam_T")


def _candidates():
    from ..rl.m43b.validation import candidates
    return candidates(include_full=True, marginal=MARGINAL, conditional=(), full=FULL)


def early_mask(base):
    """The Part C laning phase: no tower exhausted, neither champion at the level cap."""
    def m(R, S):
        return (base(R, S) & (R["tower_dmg"] < 6000) & (R["etower_dmg"] < 6000) & (R["level"] < 18)
                & (R["elevel"] < 18))
    return m


def _phase(snap: dict) -> dict:
    ch = snap["champ"]
    return {"k": snap["k"], "xp": [ch[0][3], ch[1][3]], "tau": [ch[0][6], ch[1][6]],
            "both_cap": min(ch[0][3], ch[1][3]) >= 544, "exhausted": max(ch[0][6], ch[1][6]) >= 6000}


def _state_record(res: Dict[str, dict], cands: Sequence[str]) -> dict:
    """Per candidate: paired means of every channel difference to the policy, the SE of the main objectives, the mean
    termination turn and the death / XP decomposition."""
    from .rollouts import diff_stats
    out = {}
    pol = res["policy"]
    for c in cands:
        if c == "policy":
            continue
        a = res[c]
        rec = {ch: float((a[ch] - pol[ch]).mean()) for ch in CHANNELS}
        rec["se"] = {"gold": diff_stats(res, c, "policy")[1],
                     "lam_1/8": diff_stats(res, c, "policy", lam=1 / 8)[1],
                     "mu_1": diff_stats(res, c, "policy", mu=1.0)[1],
                     "gold_T": diff_stats(res, c, "policy", terminal=True)[1],
                     "lam_1/8_T": diff_stats(res, c, "policy", lam=1 / 8, terminal=True)[1]}
        rec["xp_focal"] = float(a["xp"][:, 0].mean() - pol["xp"][:, 0].mean())
        rec["xp_opp"] = float(a["xp"][:, 1].mean() - pol["xp"][:, 1].mean())
        rec["deaths_focal"] = float(a["deaths"][:, 0].mean() - pol["deaths"][:, 0].mean())
        rec["term_turn"] = float(np.nanmean(a["term_turn"])) if np.isfinite(a["term_turn"]).any() else None
        rec["term_share"] = float(np.isfinite(a["term_turn"]).mean())
        out[c] = rec
    out["policy_term_turn"] = float(np.nanmean(pol["term_turn"])) if np.isfinite(pol["term_turn"]).any() else None
    out["policy_term_share"] = float(np.isfinite(pol["term_turn"]).mean())
    return out


def _seat_agents(agent, opp_agent, seat):
    return (agent, opp_agent) if seat == 0 else (opp_agent, agent)


def _rescore_job(args) -> tuple:
    key, i_spec, j_opp, l_sit, states = args
    import torch
    from ...diagnostics import ProbCache, Snapshot
    from ..rl.m43b.common import agent_from_spec
    from ..rl.m43b.validation import VAL_ROLL_SEED, opponent_agent
    from .rollouts import paired
    torch.set_num_threads(1)
    cfg = cfg_game()
    spec, opp, sit = key.split("|")
    agent = agent_from_spec(spec, cfg)
    opp_agent = opponent_agent(opp, spec, agent, cfg)
    root = VAL_ROLL_SEED + 100 * i_spec + 10 * j_opp + l_sit
    cache = ProbCache()
    cands = _candidates()
    out, dev = [], 0.0
    t0 = time.perf_counter()
    for i, st in enumerate(states):
        snap = Snapshot.from_dict(st["snapshot"])
        seat = st["seat"]
        res = paired(cfg, snap, seat, cands, _seat_agents(agent, opp_agent, seat), N_ROLL, root, key=(i,), cache=cache,
                     window=WINDOW)
        rec = _state_record(res, list(cands))
        for c in cands:
            if c != "policy" and c in st["adv"]:
                dev = max(dev, abs(rec[c]["G"] - st["adv"][c][0]))
        out.append({"seat": seat, "phase": _phase(st["snapshot"]), "candidates": rec})
    return key, {"states": out, "max_dev_gold_vs_stored": dev, "elapsed_s": time.perf_counter() - t0}


def _early_job(args) -> tuple:
    spec, opp, sit, state_seed, roll_seed = args
    import torch
    from ...diagnostics import ProbCache
    from ..rl.m43b.behavior import sample_states, situation_masks
    from ..rl.m43b.common import agent_from_spec
    from ..rl.m43b.validation import opponent_agent
    from .rollouts import paired
    torch.set_num_threads(1)
    cfg = cfg_game()
    agent = agent_from_spec(spec, cfg)
    opp_agent = opponent_agent(opp, spec, agent, cfg)
    base = situation_masks([sit])[sit]
    t0 = time.perf_counter()
    found = sample_states(cfg, agent, {sit: early_mask(base)}, N_STATES, state_seed,
                          opponent=None if opp == "self" else opp_agent)[sit]
    cache = ProbCache()
    cands = _candidates()
    out = []
    for i, (snap, seat, feats) in enumerate(found):
        res = paired(cfg, snap, seat, cands, _seat_agents(agent, opp_agent, seat), N_ROLL, roll_seed, key=(i,),
                     cache=cache, window=WINDOW)
        out.append({"seat": seat, "phase": _phase(snap.to_dict()), "features": feats, "snapshot": snap.to_dict(),
                    "candidates": _state_record(res, list(cands))})
    return f"{spec}|{opp}|{sit}", {"states": out, "n_found": len(found), "elapsed_s": time.perf_counter() - t0}


def objective_of_record(rec: dict, lam: float = 0.0, mu: float = 0.0, terminal: bool = False) -> float:
    s = "_T" if terminal else ""
    return rec["G" + s] + lam * rec["X" + s] + mu * rec["Lam" + s]


def summarize(cells: Dict[str, dict], cands: Sequence[str]) -> dict:
    """Per (opponent, situation): the mean over states (pooled over specs; states are the unit) of every objective's
    advantage, and the across-spec mean +- 1.96 SE (specs are the unit)."""
    out = {}
    keys = sorted({tuple(k.split("|")[1:]) for k in cells})
    for opp, sit in keys:
        rows = {}
        for c in cands:
            if c == "policy":
                continue
            per_spec, pooled = {}, []
            for k, v in cells.items():
                spec, o, s_ = k.split("|")
                if (o, s_) != (opp, sit):
                    continue
                recs = [st["candidates"][c] for st in v["states"] if c in st["candidates"]]
                if recs:
                    per_spec[spec] = recs
                    pooled += recs
            if not pooled:
                continue
            entry = {"n_states": len(pooled)}
            for lab, lam, mu, term in [("gold", 0, 0, False)] + [(f"lam={l:.4g}", l, 0, False) for l in LAMBDAS[1:]] + \
                    [(f"mu={m:g}", 0, m, False) for m in MUS[1:]] + [("gold_T", 0, 0, True), ("lam=0.125_T", 1 / 8, 0, True),
                                                                     ("mu=1_T", 0, 1.0, True)]:
                vals = [objective_of_record(r, lam, mu, term) for r in pooled]
                m, h = mean_ci(vals)
                spec_means = [np.mean([objective_of_record(r, lam, mu, term) for r in recs]) for recs in per_spec.values()]
                ms, hs = mean_ci(spec_means)
                entry[lab] = {"mean": m, "ci_states": h, "mean_specs": ms, "ci_specs": hs, "n_specs": len(spec_means)}
            entry["dX"] = mean_ci([r["X"] for r in pooled])
            entry["xp_focal"] = mean_ci([r["xp_focal"] for r in pooled])
            entry["xp_opp"] = mean_ci([r["xp_opp"] for r in pooled])
            entry["term_share"] = float(np.mean([r["term_share"] for r in pooled]))
            tt = [r["term_turn"] for r in pooled if r["term_turn"] is not None]
            entry["term_turn"] = float(np.mean(tt)) if tt else None
            rows[c] = entry
        out[f"{opp}|{sit}"] = rows
    return out


def run(stages: Sequence[str] = ("rescore", "early"), workers: int = 9) -> dict:
    from ..rl.m43b.common import run_jobs
    from ..rl.m43b.validation import OPPONENTS as VAL_OPP, POLICIES as VAL_POL, SITUATIONS as VAL_SIT, vload
    t0 = time.perf_counter()
    out = {"meta": meta(stage="xp_cf", window=WINDOW, n_roll=N_ROLL, lambdas=list(LAMBDAS), mus=list(MUS),
                        candidates=list(_candidates()))}
    if "rescore" in stages:
        stored = vload("wave_cf")["results"]
        jobs = []
        for key, v in stored.items():
            spec, opp, sit = key.split("|")
            if spec not in RESCORE_SPECS:
                continue
            jobs.append((key, VAL_POL.index(spec), VAL_OPP.index(opp), VAL_SIT.index(sit),
                         [{"snapshot": s["snapshot"], "seat": s["seat"], "adv": s["adv"]} for s in v["states"]]))
        res = dict(run_jobs(_rescore_job, jobs, workers, "rescore"))
        out["rescore"] = {"cells": res, "summary": summarize(res, list(_candidates())),
                          "max_dev_gold_vs_stored": max(v["max_dev_gold_vs_stored"] for v in res.values())}
    if "early" in stages:
        jobs = []
        for i, spec in enumerate(SPECS):
            for j, opp in enumerate(OPPONENTS):
                for l, sit in enumerate(SITUATIONS):
                    jobs.append((spec, opp, sit, XP_CF_SEED + 100 * i + 10 * j + l, XP_CF_SEED + 500 + 100 * i + 10 * j + l))
        res = dict(run_jobs(_early_job, jobs, workers, "early"))
        out["early"] = {"cells": res, "summary": summarize(res, list(_candidates()))}
    out["meta"]["elapsed_s"] = time.perf_counter() - t0
    save_json("xp_cf", out)
    return out


__all__ = ["run", "summarize", "early_mask", "objective_of_record"]
