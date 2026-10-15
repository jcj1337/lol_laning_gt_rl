"""M4.5c stages other than training: the interface audit, the smoke evaluation, the pilot evaluation (curves,
headline, history) and the pilot behaviour diagnostics. Every stage is a set of picklable jobs run in worker processes;
agents are rebuilt in the worker from their specification strings (common.agent_from_spec, with the compatibility
guard). Results are JSON in results/m45c/.
"""

from __future__ import annotations

import time
from typing import Dict, Sequence

import numpy as np

from . import actions as AC
from .common import (BEHAVIOR_SEED, LEDGER_SEEDS, PILOT_SEED, PILOT_STOP, SMOKE_EVAL_SEED, SMOKE_SEED, WORKERS,
                     env_cfg, load_json, meta, save_json)
from .references import CORE, OOD, SUITE


# ---------------------------------------------------------------------------------------------- audit
def audit() -> dict:
    """Static and random-play checks of the M4.5c interface: the frozen preset, the action space and masks, the Ward
    legality the records use, observation range and mirror symmetry, zero-sum rewards, the reference suite's legality,
    the checkpoint guard and the labelled out-of-distribution adapter."""
    import os
    from ....champions import champion_legal_actions
    from ....config import GameConfig
    from ....env import LaneEnv
    from ....env.agents import UniformAgent, episode_streams
    from ....env.observation_m4 import decode_m4
    from ..m44.common import load_policy
    from .common import agent_from_spec
    from .references import reference
    t0 = time.perf_counter()
    cfg = env_cfg()
    AC.check_space(cfg)
    out = {"meta": meta(stage="audit")}
    want = {"farm_spot_rule": "last_front", "minion_retaliation_delay": 0, "minion_damage_growth": 1.08,
            "xp_range": 2, "vision_rule": "ward", "minion_damage_alpha": 50}
    d = cfg.to_dict()
    out["preset"] = {"equals_m45_calibrated": cfg == GameConfig.m45_calibrated(),
                     "retained_fields": {k: d.get(k) for k in want},
                     "retained_as_specified": all(d.get(k) == v for k, v in want.items()),
                     "vision": {k: d[k] for k in sorted(d) if k.startswith(("vision", "ward"))}}
    e = LaneEnv(cfg, max_turns=400, cache_size=5000)
    seen, bad_obs, bad_mask, mirror_bad, ward_bad, turns = set(), 0, 0, 0, 0, 0
    for g in range(4):
        _, env_ss, ra, rb = episode_streams(np.random.SeedSequence(SMOKE_EVAL_SEED, spawn_key=(g,)))
        obs = e.reset(seed=env_ss)
        ua, ub = UniformAgent(), UniformAgent()
        while True:
            for s in (0, 1):
                o = obs[s]
                seen |= set(o.legal)
                v = o.vector
                bad_obs += int(len(v) != 90 or v.min() < 0 or v.max() > 1 or not np.all(np.isfinite(v)))
                bad_mask += int(set(np.flatnonzero(o.mask)) != set(o.legal))
                me = o.champ.side(s)
                rec_rule = me.absent == 0 and (abs(me.pos) <= 1 or me.pos == 99)
                ward_bad += int(rec_rule != (AC.WARD in champion_legal_actions(o.state, o.champ, s, cfg)))
            ph_a, la, ca, _ = decode_m4(obs[0].vector, cfg)
            ph_b, lb, cb, _ = decode_m4(obs[1].vector, cfg)
            mirror_bad += int(not (ph_a == ph_b and lb == la.mirror() and cb == ca.mirror()))
            r = e.step(ua.act(obs[0], ra), ub.act(obs[1], rb))
            turns += 1
            if r.rewards[0] != -r.rewards[1]:
                raise AssertionError("rewards not zero-sum")
            if r.done:
                break
            obs = r.obs
    out["random_play"] = {"turns": turns, "obs_size": e.obs_size, "n_actions": e.n_actions,
                          "observed_legal_codes": sorted(seen), "outside_ever_legal": sorted(seen - AC.EVER_LEGAL),
                          "ever_legal_not_observed": sorted(AC.EVER_LEGAL - seen), "observations_out_of_range": bad_obs,
                          "masks_not_equal_legal": bad_mask, "mirror_failures": mirror_bad,
                          "record_ward_rule_mismatches": ward_bad}
    out["action_space"] = {"codes": AC.N_CODES, "ever_legal": len(AC.EVER_LEGAL), "never_legal": len(AC.NEVER_LEGAL),
                           "table": AC.action_table(cfg)}
    # the reference suite: legal on this preset, both seats, against random play
    refs = {}
    for name in SUITE:
        x = reference(name)
        ok = True
        for seat in (0, 1):
            _, env_ss, ra, rb = episode_streams(np.random.SeedSequence(SMOKE_EVAL_SEED + 1, spawn_key=(seat,)))
            obs = e.reset(seed=env_ss)
            x.reset()
            u = UniformAgent()
            for _ in range(150):
                a = x.act(obs[seat])
                ok &= a in obs[seat].legal
                b = u.act(obs[1 - seat], rb)
                r = e.step(a, b) if seat == 0 else e.step(b, a)
                x.notify(r.info, seat)
                if r.done:
                    break
                obs = r.obs
        refs[name] = ok
    out["references_legal"] = refs
    # checkpoint guard and the OOD adapter
    import glob
    comp = {}
    paths = sorted(glob.glob(os.path.join("results", "rl", "checkpoints", "m44", "ext", "*_s0", "final.pt")))
    for p in paths:
        try:
            load_policy(p, cfg)
            comp[p.replace("\\", "/")] = "loads (DEFECT: must be refused)"
        except ValueError as ex:
            comp[p.replace("\\", "/")] = "refused: shape" if "action outputs" in str(ex) else "refused: configuration"
    out["m44_checkpoints_on_m45c"] = comp
    ood = {}
    for spec in OOD:
        try:
            a = agent_from_spec(spec, cfg)
            o = e.reset(seed=1)[0]
            p = a.probabilities(o)
            ood[spec] = {"loads_through_adapter": True, "drops": a.drop, "mass_on_legal": float(p[list(o.legal)].sum()),
                         "label": "OUT OF DISTRIBUTION"}
        except FileNotFoundError:
            ood[spec] = {"loads_through_adapter": False, "reason": "checkpoint not present"}
    out["ood_adapter"] = ood
    out["elapsed_s"] = time.perf_counter() - t0
    save_json("audit", out)
    return out


# ---------------------------------------------------------------------------------------------- smoke evaluation
def smoke_eval(workers: int = WORKERS) -> dict:
    """Every evaluation instrument end to end on the smoke final at tiny sizes, plus the guard on M4.5 presets."""
    from ..m44.common import load_policy
    from . import behavior as BH
    from . import evaluate as EV
    from .common import final_path
    from ....config import GameConfig
    t0 = time.perf_counter()
    spec = f"final:smoke:{SMOKE_SEED}"
    data = {"meta": meta(stage="smoke_eval")}
    p = final_path("smoke", SMOKE_SEED)
    guard = {"m45_calibrated": "loads"}
    load_policy(p, env_cfg())
    for other in ("m45_unified", "m4_four_actions_econ_aggro"):
        try:
            load_policy(p, getattr(GameConfig, other)())
            guard[other] = "loads (DEFECT: must be refused)"
        except ValueError as ex:
            guard[other] = "refused: shape" if "action outputs" in str(ex) else "refused: configuration"
    data["guard"] = guard
    jobs = [(spec, f"ref:{r}", 2, 100, SMOKE_EVAL_SEED) for r in CORE]
    jobs += [(f"initial:{SMOKE_SEED}", f"ref:{r}", 2, 100, SMOKE_EVAL_SEED) for r in CORE[:2]]
    jobs += [(spec, f"initial:{SMOKE_SEED}", 2, 100, SMOKE_EVAL_SEED), (spec, spec, 2, 100, SMOKE_EVAL_SEED)]
    jobs += [(spec, o, 2, 100, SMOKE_EVAL_SEED) for o in OOD]
    data["strength"] = EV.run(jobs, workers, "smoke strength")
    data["records"] = BH.run_records([(spec, "self", 2, 200, SMOKE_EVAL_SEED), (spec, "ref:g_shove", 1, 200,
                                                                                   SMOKE_EVAL_SEED)], workers)
    data["ledger"] = BH.run_ledger([(spec, "self", LEDGER_SEEDS[:2], 100), (spec, "ref:g_bully", LEDGER_SEEDS[:2], 100)],
                                   workers)
    sp = data["strength"][f"{spec}|{spec}"]["estimate"]
    data["checks"] = {"self-play value exactly 0 (paired seats, same network)": abs(sp) < 1e-9,
                      "guard refuses other configurations": all(v.startswith("refused") for k, v in guard.items()
                                                               if k != "m45_calibrated"),
                      "every instrument produced output": bool(data["records"] and data["ledger"] and data["strength"])}
    data["all_passed"] = all(data["checks"].values())
    data["elapsed_s"] = time.perf_counter() - t0
    save_json("smoke_eval", data)
    return data


# ---------------------------------------------------------------------------------------------- pilot evaluation
def eval_pilot(workers: int = WORKERS, parts: Sequence[str] = ("curve", "headline", "history")) -> dict:
    """Learning curve (snapshots and the untrained network vs the suite and the OOD network), headline (final and
    half-way vs the suite at W = 1000) and history (final vs own snapshots and the untrained network). Merged into
    pilot_eval.json part by part."""
    from . import evaluate as EV
    data = load_json("pilot_eval") or {"meta": meta(stage="eval_pilot")}
    opps = [f"ref:{r}" for r in SUITE] + list(OOD)
    timing = data.setdefault("timing", {})
    for part in parts:
        t0 = time.perf_counter()
        if part == "curve":
            jobs = EV.curve_jobs(opps)
            data["curve"] = EV.run(jobs, workers, "curve")
            data["curve_spec"] = {"points": list(EV.CURVE_POINTS), "episodes": EV.CURVE_EPISODES, "window": EV.DEV_WINDOW}
        elif part == "headline":
            data["headline"] = EV.run(EV.headline_jobs(opps), workers, "headline")
            data["headline_spec"] = {"episodes": EV.HEAD_EPISODES, "window": EV.HEAD_WINDOW,
                                     "checkpoints": [f"final:pilot:{PILOT_SEED}", EV.snap_spec(PILOT_STOP // 2)]}
        elif part == "history":
            data["history"] = EV.run(EV.history_jobs(), workers, "history")
            data["history_spec"] = {"points": list(EV.HIST_POINTS), "episodes": EV.HIST_EPISODES, "window": EV.DEV_WINDOW}
        else:
            raise ValueError(part)
        timing[part] = {"wall_s": time.perf_counter() - t0, "workers": workers}
        save_json("pilot_eval", data)
    return data


BEHAVIOR_CHECKPOINTS = (f"initial:{PILOT_SEED}", f"snap:pilot:{PILOT_SEED}:102400", f"snap:pilot:{PILOT_SEED}:512000",
                        f"final:pilot:{PILOT_SEED}")
BEHAVIOR_OPPONENTS = ("ref:g_shove", "ref:g_farm", "ref:g_bully", "ref:g_safe", "ref:fz_guard_sh", "ref:g_shove_ward")


def behavior_pilot(workers: int = WORKERS) -> dict:
    """Self-play records of the behaviour checkpoints (16 games x 1000 turns, both seats focal), the final against the
    behaviour opponents (8 seeds x 500 turns, both seats), and the M4.5b ledger (self-play 48 seeds x 300 turns; the
    final against four opponents 24 seeds x 300 turns, both seats)."""
    from . import behavior as BH
    t0 = time.perf_counter()
    final = f"final:pilot:{PILOT_SEED}"
    rj = [(c, "self", 16, 1000, BEHAVIOR_SEED) for c in BEHAVIOR_CHECKPOINTS]
    rj += [(final, o, 8, 500, BEHAVIOR_SEED + 1) for o in BEHAVIOR_OPPONENTS]
    lj = [(c, "self", LEDGER_SEEDS[:48], 300) for c in BEHAVIOR_CHECKPOINTS]
    lj += [(final, o, LEDGER_SEEDS[48:72], 300) for o in ("ref:g_shove", "ref:g_bully", "ref:fz_guard_sh",
                                                           "ref:g_shove_ward")]
    data = {"meta": meta(stage="behavior_pilot", checkpoints=list(BEHAVIOR_CHECKPOINTS),
                         opponents=list(BEHAVIOR_OPPONENTS)),
            "records": BH.run_records(rj, workers), "ledger": BH.run_ledger(lj, workers)}
    data["elapsed_s"] = time.perf_counter() - t0
    save_json("pilot_behavior", data)
    return data


__all__ = ["audit", "smoke_eval", "eval_pilot", "behavior_pilot", "BEHAVIOR_CHECKPOINTS", "BEHAVIOR_OPPONENTS"]
