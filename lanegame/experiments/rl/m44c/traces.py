"""Counterfactual pairs for the Streamlit viewer (M4.4c Part 9): two strategies from the SAME state, replayable.

The viewer replays a recording from (preset, scenario, gap, integer chance seed) and the list of actions. A
counterfactual pair is therefore built INSIDE a viewer-seeded game: the natural game of the aggro seed-0 4.10M final
(or an initialized scenario the viewer knows) is replayed up to the branch turn t, then each strategy plays the next
`H` turns in the same environment. The environment's chance vector of every turn does not depend on the actions, so
both branches see identical chance (common random numbers), and the agents' sampling streams are seeded identically
for both branches. Each branch is exported as prefix + continuation and verified by replaying it through the viewer's
own session code (`m44.trace.verify`).

Selection (pre-declared, never the most extreme example): for each situation the first K states that match its
decision-state predicate (one per viewer-seeded game, in seed order) are branched once with both strategies; among the
states where the two branches' H-turn gold leads differ at all (identical branches show nothing), the state with the
MEDIAN absolute difference is exported. A single pair is one sample of a noisy game: it illustrates the mechanism, not
the average effect (the tables do that).

    streamlit run lanegame/viewer/app.py  ->  "Recorded trajectory"  ->  folder results/m44c/trajectories
"""

from __future__ import annotations

import os
import time
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ....diagnostics.branch import DEFER, FollowPolicy, sample_index
from ....env import LaneEnv
from ..m44.actions import decode
from ..m44.common import ENVS
from ..m44.records import holds_freeze, play_record
from ....state import State
from .common import TRACE_SEED, TRAJ_DIR, load, meta, save_json

K = 8
H = 30
SCHEMA = "lanegame.viewer_recording"


def play_branch(cfg, x, seed: int, prefix: Sequence, start, focal: int, plan, horizon: int, stream_seed: int):
    """Replay `prefix` from the viewer start, then `horizon` turns: the focal side follows `plan` then x, the other
    side x; returns (actions of the continuation, infos of prefix + continuation)."""
    env = LaneEnv(cfg, max_turns=len(prefix) + horizon, cache_size=20_000)
    if start is None:
        obs = env.reset(seed=seed)
    else:
        lane, m4, k0 = start
        obs = env.reset(seed=seed, root=lane, k0=k0, champ=m4)
    infos = []
    for a, b in prefix:
        r = env.step(int(a), int(b))
        infos.append(r.info)
        obs = r.obs
    rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(stream_seed).spawn(2)]
    actions, live = [], True
    for t in range(horizon):
        acts = [0, 0]
        for s in (0, 1):
            ob = obs[s]
            p = np.asarray(x.distribution(ob), dtype=float)
            pa = None
            if s == focal and live:
                pa = plan.act(ob, t)
                if pa is None:
                    live = False
                elif pa == DEFER:
                    p = np.asarray(plan.transform(ob, t, p), dtype=float)
            u = rngs[s].random()
            a = ob.legal[sample_index(p, u)]
            if pa is not None and pa != DEFER:
                a = int(pa)
            acts[s] = int(a)
        r = env.step(acts[0], acts[1])
        actions.append(acts)
        infos.append(r.info)
        obs = r.obs
        if r.done:
            break
    return actions, infos


def _lead(infos: List[dict], f: int, cfg) -> float:
    return sum((inf["earned_units"][f] - inf["earned_units"][1 - f]) / cfg.gold_scale for inf in infos)


def _events(infos: List[dict], cfg, t0: int, focal: int, label: str) -> List[list]:
    ev = [[t0, f"BRANCH POINT: {'AB'[focal]} plays '{label}' from here (M4.4c counterfactual)"]]
    for t, inf in enumerate(infos):
        for s, n in ((0, "A"), (1, "B")):
            if inf["died"][s]:
                ev.append([t, f"{n} dies"])
            if inf["recalled"][s]:
                ev.append([t, f"{n} recalls"])
            if inf["recall_interrupted"][s]:
                ev.append([t, f"{n}'s recall is interrupted"])
            if inf["crash_hp"][s]:
                ev.append([t, f"{n} crashes {int(inf['crash_hp'][s])} HP into the enemy tower"])
            if inf["plates_broken"][s]:
                ev.append([t, f"{n} breaks a plate"])
            if inf["vision_set"][s]:
                ev.append([t, f"{n} establishes vision"])
        lane = State.from_list(inf["state_after"])
        prev = State.from_list(inf["state_before"])
        for s, n in ((0, "A"), (1, "B")):
            if holds_freeze(lane, s) and not holds_freeze(prev, s):
                ev.append([t, f"freeze state: {n} holds the enemy wave at its tower"])
    return sorted(ev)


def _situations():
    from . import freeze as F
    from . import recall as R
    from . import vision as V
    from . import waves as W
    from . import zoning as Z
    from ..m44.actions import RECALL
    return {
        "recall": dict(pred=lambda cfg: R.predicates(cfg)["contested"], pair=("Recall", "Safe"),
                       plans=lambda: R.candidates(), desc="a damaged champion that may be interrupted: Recall now vs "
                       "Safe this turn (then the policy)"),
        "freeze": dict(pred=lambda cfg: F.predicates(cfg)["held"], pair=("Preserve", "Break"),
                       plans=lambda: F.candidates("held", None, None, None), desc="a freeze held for 2+ turns: keep it "
                       "(threshold controller) vs break it (Shove)"),
        "vision": dict(pred=lambda cfg: V.predicates(cfg)["plates"], pair=("Vision>Pressure", "Pressure"),
                       plans=lambda: V.candidates("plates", None, None, None), desc="a siege with plates left: Vision "
                       "first, then Pressure vs Pressure at once"),
        "waves": dict(pred=lambda cfg: (lambda d, t, s: W.contact_pred(d, t, s) and d["hp_frac"][t, s] >= 0.5
                                        and d["prec"][t, s] < 0.2), pair=("SLx8", "LHx8"),
                      plans=lambda: W.candidates("contact", None, None, None), desc="fronts in contact: Slow Push vs "
                      "Last Hit for 8 turns"),
        "zoning": dict(start=("neutral", "large"), pair=("Threaten", "Farm"),
                       plans=lambda: Z.candidates("init", None, None, None), desc="INITIALIZED neutral lane, A x1.76: "
                       "threaten (Trade when catchable with the edge) vs farm"),
    }


def situation_pair(key: str, env: str = "aggro", seed: int = 0, phase: str = "ext") -> dict:
    cfg, x = load(env, seed, phase)
    sit = _situations()[key]
    a_name, b_name = sit["pair"]
    cands = []
    for g in range(60):
        vseed = TRACE_SEED + g
        if "start" in sit:
            from ...m43c.harness import start as scenario_start
            sc, gap = sit["start"]
            start = scenario_start(sc, cfg, gap)
            prefix, t, focal, scen = [], 0, 0, (sc, gap)
        else:
            rec = play_record(cfg, (x, x), vseed, 460, viewer_seed=True)
            d = rec.data
            pred = sit["pred"](cfg)
            hits = [(t, s) for t in range(8, rec.turns - H) for s in (0, 1) if d["present"][t, s] > 0 and pred(d, t, s)]
            if not hits:
                continue
            t, focal = hits[0]
            prefix = [[int(d["a"][u, 0]), int(d["a"][u, 1])] for u in range(t)]
            start, scen = None, ("s0", "equal")
        out = {}
        for name in (a_name, b_name):
            plan = sit["plans"]()[name]
            acts, infos = play_branch(cfg, x, vseed, prefix, start, focal, plan, H, stream_seed=vseed * 7 + 1)
            out[name] = {"actions": prefix + acts, "infos": infos, "lead": _lead(infos[t:], focal, cfg)}
        cands.append({"vseed": vseed, "turn": t, "focal": focal, "scen": scen, "branches": out,
                      "diff": out[a_name]["lead"] - out[b_name]["lead"]})
        if len(cands) >= K:
            break
    diffs = np.array([c["diff"] for c in cands])
    differ = [c for c in cands if abs(c["diff"]) > 1e-9] or cands
    ad = np.array([abs(c["diff"]) for c in differ])
    pick = differ[int(np.argsort(ad)[len(ad) // 2])]
    files = []
    for name in (a_name, b_name):
        br = pick["branches"][name]
        infos = br["infos"]
        fname = f"m44c_{key}_{name.replace('>', '_then_').replace('+', '_').replace(' ', '_')}"
        rec = {"schema": SCHEMA, "version": 1, "config": ENVS[env].preset, "scenario": pick["scen"][0],
               "gap": pick["scen"][1], "seed": int(pick["vseed"]), "names": [f"final:{phase}:{env}:{seed}"] * 2,
               "label": (f"M4.4c counterfactual pair ({key}): {sit['desc']}. Branch '{name}' for side "
                         f"{'AB'[pick['focal']]} from turn {pick['turn']}; the other file of the pair plays "
                         f"'{b_name if name == a_name else a_name}' from the same state with the same chance. Selected "
                         f"as the median absolute {H}-turn gold-lead difference among the {len(differ)} of {len(cands)} "
                         f"candidate states where the two branches differ."),
               "actions": br["actions"], "events": _events(infos, cfg, pick["turn"], pick["focal"], name),
               "payoff_A": float(sum(inf["reward"] for inf in infos)), "turns": len(br["actions"]),
               "labels": [[decode(a).label(), decode(b).label()] for a, b in br["actions"]],
               "branch": {"turn": pick["turn"], "focal": pick["focal"], "strategy": name, "horizon": H,
                          "gold_lead_over_horizon": br["lead"]}}
        os.makedirs(TRAJ_DIR, exist_ok=True)
        path = os.path.join(TRAJ_DIR, fname + ".json")
        from ....io import dump_json
        dump_json(rec, path)
        from ..m44.trace import verify
        files.append({"path": path, "strategy": name, "lead": br["lead"], "verify": verify(path)})
    return {"situation": key, "pair": [a_name, b_name], "picked": {"vseed": pick["vseed"], "turn": pick["turn"],
            "focal": pick["focal"], "diff": pick["diff"]}, "candidate_diffs": diffs.tolist(), "files": files}


def run(workers: int = 5) -> dict:
    from .common import run_jobs
    t0 = time.perf_counter()
    keys = list(_situations())
    res = run_jobs(situation_pair, keys, workers, "traces")
    data = {"meta": meta(stage="traces", K=K, H=H, trace_seed=TRACE_SEED), "pairs": res,
            "all_verified": all(f["verify"]["actions_equal"] and f["verify"]["payoff_equal"] for r in res
                                for f in r["files"]), "elapsed_s": time.perf_counter() - t0}
    save_json("traces", data)
    return data


__all__ = ["play_branch", "situation_pair", "run", "K", "H"]
