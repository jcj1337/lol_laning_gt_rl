"""Representative games of M4.5c policies for the viewer: the M4.4b state-defined trace search (`m44.supplement`:
freeze denies / fails / broken, a Slow Push that lands, Ward then siege Pressure, busy tower dives, the weak side under
a credible threat), adapted to M4.5. Prepared for the comprehensive evaluation; `smoke()` exercises it.

Differences from M4.4b: the recordings are stamped `m45_calibrated` (`viewer.session.recording_stamp`); action labels
come from the configuration (`cfg.label`, so code 30 reads "Ward", never "Last Hit + Hold + Vision"); a placed Ward is
an event; the power-gap search starts from the viewer's own "neutral" start with gap "large" (A x1.76), so every
exported game replays exactly in the debugger. Every export is verified by replaying it through the viewer's session
code (`m44.trace.verify`: equal actions and payoff on every turn).

Seeds: search games 94,200+ (self-play from s0) and 94,240+ (power gap), viewer convention (`viewer_seed`).
"""

from __future__ import annotations

import os
import time
from typing import List, Sequence

from .common import OUT, agent_from_spec, env_cfg, meta, run_jobs, save_json

TOUT = os.path.join(OUT, "traces")
TRACE_SEED_F = 94_200
SIZES = dict(selfplay_games=24, selfplay_turns=460, gap_games=12, gap_turns=300)
KINDS = ("freeze_denies", "freeze_fails", "freeze_broken", "slow_push_lands", "vision_then_pressure", "busy_dive",
         "weak_side_threatened")
CONFIG = "m45_calibrated"


def _gap_start():
    from ....viewer import session as ss
    return ss.start_state(ss.Setup(CONFIG, "neutral", "large", 0))


def search_job(args) -> tuple:
    """(spec, kind 's0' | 'gap', games, turns, seed0): record self-play games and return the candidate situations."""
    spec, kind, games, turns, seed0 = args
    import torch
    torch.set_num_threads(1)
    from ..m44.records import play_record
    from ..m44.supplement import _episodes
    cfg = env_cfg()
    x = agent_from_spec(spec, cfg)
    start = _gap_start() if kind == "gap" else None
    cands = []
    for g in range(games):
        seed = seed0 + g
        rec = play_record(cfg, (x, x), seed, turns, start, "neutral/large" if start else "s0", "m45c", viewer_seed=True)
        for s in (0, 1):
            if kind == "gap" and s == 0:
                continue                                  # the gap favours A: B is the weak side
            for k, lst in _episodes(rec, s).items():
                if kind == "gap" and k != "weak_side_threatened":
                    continue
                for c in lst:
                    cands.append({"kind": k, "seed": seed, "scenario": "neutral" if start else "s0",
                                  "gap": "large" if start else "equal", "side": c["side"], "t0": c["t0"],
                                  "t1": c["t1"], "score": c["score"]})
    return (spec, kind), cands


def export_record(rec, name: str, specs: Sequence[str], seed: int, scenario: str, gap: str, label: str,
                  out_dir: str) -> str:
    from ....io import dump_json
    from ....viewer.session import RECORDING_SCHEMA, recording_stamp
    cfg = env_cfg()
    d = rec.data
    actions = [[int(d["a"][t, 0]), int(d["a"][t, 1])] for t in range(rec.turns)]
    events = []
    for t in range(rec.turns):
        for s, n in ((0, "A"), (1, "B")):
            for f, text in (("died", f"{n} dies"), ("recalled", f"{n} recalls (HP {d['hp_frac'][t, s]:.0%})"),
                            ("plates", f"{n} breaks a plate"), ("vision_set", f"{n} places a Ward")):
                if d[f][t, s]:
                    events.append([t, text])
            if d["crash"][t, s]:
                events.append([t, f"{n} crashes {int(d['crash'][t, s])} HP into the enemy tower"])
            if d["held"][t, s] and (t == 0 or not d["held"][t - 1, s]):
                events.append([t, f"freeze state: {n} holds the enemy wave at its tower"])
    rj = {"schema": RECORDING_SCHEMA, "version": 1, **recording_stamp(CONFIG), "scenario": scenario, "gap": gap,
          "seed": int(seed), "names": list(specs), "label": label, "actions": actions, "events": events,
          "payoff_A": float(d["reward"][:, 0].sum()), "turns": rec.turns,
          "labels": [[cfg.label(a), cfg.label(b)] for a, b in actions]}
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.json")
    dump_json(rj, path)
    return path


def traces(specs: Sequence[str], workers: int, sizes: dict = SIZES, out_dir: str = TOUT, tag: str = "") -> dict:
    """Search every spec's self-play (and power-gap) games, export the best-scoring game of each kind, verify it."""
    import torch
    from ..m44.records import play_record
    from ..m44.supplement import excerpt
    from ..m44.trace import verify
    t0 = time.perf_counter()
    js = [(sp, "s0", sizes["selfplay_games"], sizes["selfplay_turns"], TRACE_SEED_F) for sp in specs]
    js += [(sp, "gap", sizes["gap_games"], sizes["gap_turns"], TRACE_SEED_F + 40) for sp in specs]
    found = dict(run_jobs(search_job, js, workers, "trace search"))
    cfg = env_cfg()
    selected = {}
    for sp in specs:
        cands = found.get((sp, "s0"), []) + found.get((sp, "gap"), [])
        x = agent_from_spec(sp, cfg)
        for kind in KINDS:
            cs = sorted([c for c in cands if c["kind"] == kind], key=lambda c: (-c["score"], c["seed"], c["t0"]))
            if not cs:
                selected[f"{sp}|{kind}"] = {"found": 0}
                continue
            c = cs[0]
            torch.set_num_threads(1)
            start = _gap_start() if c["scenario"] != "s0" else None
            turns = sizes["gap_turns"] if start else sizes["selfplay_turns"]
            rec = play_record(cfg, (x, x), c["seed"], turns, start, "neutral/large" if start else "s0", "m45c",
                              viewer_seed=True)
            name = f"m45c{tag}_{sp.replace(':', '_')}_{kind}"
            label = f"M4.5c {sp} self-play; {kind.replace('_', ' ')} by {'AB'[c['side']]} at turns {c['t0']}-{c['t1']}"
            path = export_record(rec, name, (sp, sp), c["seed"], c["scenario"], c["gap"], label, out_dir)
            selected[f"{sp}|{kind}"] = {"found": len(cs), **{k: c[k] for k in ("seed", "side", "t0", "t1", "scenario",
                                                                               "gap")},
                                        "path": path, "label": label, "excerpt": excerpt(rec, c["t0"], c["t1"]),
                                        "verify": verify(path)}
    data = {"meta": meta(stage="traces", specs=list(specs), sizes=dict(sizes)), "selected": selected,
            "all_verified": all(v["verify"]["actions_equal"] and v["verify"]["payoff_equal"]
                                for v in selected.values() if v.get("found")),
            "elapsed_s": time.perf_counter() - t0}
    save_json(f"traces{tag}", data, out_dir)
    return data


def smoke(spec: str = "final:full:0", workers: int = 2) -> dict:
    return traces([spec], workers, dict(selfplay_games=2, selfplay_turns=200, gap_games=1, gap_turns=120),
                  os.path.join(OUT, "battery_smoke", "traces"), "_smoke")


__all__ = ["traces", "smoke", "search_job", "export_record", "KINDS", "SIZES", "TRACE_SEED_F"]
