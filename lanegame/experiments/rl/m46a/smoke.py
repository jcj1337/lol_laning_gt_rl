"""Pipeline smoke test of M4.6a (pass / fail and timing only; no number is reported or used, pre-registration §3).
Nothing is written to the experiment directories (every job here has path None)."""

from __future__ import annotations

import time

import numpy as np


def run() -> dict:
    from .common import env_cfg, setup_worker, spec
    from . import engine as EN
    from . import policies as P
    from .states import freeze_gap_state, zoning_gap_state
    from .e1 import FREEZE_A, FREEZE_B, pair_expectation
    from .e2 import CANDS
    from .e5 import RECALL_CANDS
    checks = {}
    t0 = time.perf_counter()
    cfg = setup_worker()
    st = zoning_gap_state(cfg, "neutral", 10, "x1.76")
    # 1. a duplicated baseline gives exactly zero paired differences; every E2 candidate runs
    cands = {"policy": ("policy",), "policy2": ("policy",), **{k: v for k, v in CANDS.items() if k != "policy"}}
    job = {"spec": spec(0), "family": "smoke", "idx": 0, "start": st, "proto": "policy", "horizon": 12,
           "roll": EN.roll_seeds(1, 2), "cands": cands, "critic": True, "path": None}
    _, rec = EN.paired_job(job)
    checks["crn_identity"] = all(abs(EN.per_state_diff(rec, "policy2", s, 12)) == 0 for s in ("G", "Q"))
    checks["plans_force"] = rec["forced"]["threaten_beh"] != [0, 0] and rec["forced"]["farm_beh"] != [0, 0]
    _, rec2 = EN.paired_job({**job, "proto": "g_wary", "cands": {"policy": ("policy",), "bully": CANDS["bully_beh"]}})
    checks["scripted_opponent"] = rec2["turns"] == 2 * 2 * 12
    # 2. state variants
    fs = (st[0], st[1]._replace(a=st[1].a._replace(gold=3 * cfg.gold_scale)), st[2])
    iv = P.VARIANTS["inv"](cfg, fs)
    checks["variant_inv"] = iv[1].a.gold == 0 and iv[1].a.items == fs[1].a.items + 3 * cfg.gold_scale
    hv = P.VARIANTS["hp-0.3"](cfg, st)
    checks["variant_hp"] = hv[1].a.hp < st[1].a.hp
    _, rec3 = EN.paired_job({**job, "cands": {"policy": RECALL_CANDS["policy"], "inv": RECALL_CANDS["inv:policy"],
                                              "hp": ("variant", "hp-0.3", ("policy",))}, "start": fs})
    checks["variant_jobs"] = rec3["turns"] == 3 * 2 * 12
    # 3. exact one-turn expectation
    fz = freeze_gap_state(cfg, 10, "x1.76", 12)
    e = pair_expectation(cfg, fz, FREEZE_A["guard"], FREEZE_B["break"])
    checks["exact"] = -1e-9 <= e["freeze_holds"] <= 1.0 + 1e-9 and e["branches"] > 1
    # 4. one E3 game
    from .e3 import game, start_of
    from .common import net_of
    g = game(cfg, net_of(spec(0)), start_of(cfg, 10, "x1.76", 1), 123, 1, True)
    checks["e3_game"] = set(g["snaps"]) == {20, 50, 100, 300} and "e6" in g
    # 5. the seat-aware collector equals Collector without an archive (one update, bit-identical weights)
    checks["focal_collector_default"] = _collector_identity()
    out = {"ok": all(checks.values()), "checks": checks, "seconds": time.perf_counter() - t0}
    print(out, flush=True)
    return out


def _collector_identity() -> bool:
    import torch
    from ....rl import OpponentSampler, train_ppo
    from ..m45c.common import agent_from_spec, env_cfg, ppo_config
    from .e4 import focal_collector_cls
    cfg = env_cfg()
    opp = agent_from_spec("final:ext:0", cfg)
    nets = []
    for cls in (None, focal_collector_cls()):
        ppo = ppo_config(5, 4096, lane_cache=500, n_envs=2, segment_length=32)
        r = train_ppo(cfg, ppo, OpponentSampler([opp]), verbose=False, collector_cls=cls)
        nets.append(r.net.state_dict())
    return all(torch.equal(nets[0][k], nets[1][k]) for k in nets[0])


__all__ = ["run"]
