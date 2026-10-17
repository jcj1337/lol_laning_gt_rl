"""M4.6d EXPLORATORY follow-up of the critic ranking (after reading it): at the constructed Recall windows, the weak
side's evasive codes the actor actually uses (Yield) against Last Hit + Safe; same seeds and design as `critic rank`."""
import os

from lanegame.config import B_SAFE, B_YIELD, W_HOLD, W_SHOVE, champ_act
from lanegame.experiments.rl.m46d import common as C
from lanegame.experiments.rl.m46d import critic as CR

if __name__ == "__main__":
    cands = {"LH-Safe": champ_act(W_HOLD, B_SAFE), "SH-Safe": champ_act(W_SHOVE, B_SAFE),
             "LH-Yield": champ_act(W_HOLD, B_YIELD), "SH-Yield": champ_act(W_SHOVE, B_YIELD)}
    cfg = C.arm_cfg("C1")
    js, j = [], 0
    for cond in ("edge", "x1.76"):
        for L in (5, 8):
            for hpb in (0.5, 0.35):
                st = CR.recall_window(cfg, cond, L, hpb)
                for s in C.SEEDS:
                    js.append({"arm": "C1", "spec": C.general("C1", s), "tag": f"s{s}", "family": "recallw_yield",
                               "idx": j, "decider": 1, "start": C.state_to_list(st), "cands": cands, "K": CR.K_RANK,
                               "H": CR.H_RANK, "root": C.CRIT_ROOT + 1, "label": f"recall window L{L} {cond} hpB{hpb}",
                               "path": os.path.join(C.CKPT, "critic", "rank_yield", f"s{s}", f"recallw_{j}_d1.json.gz")})
                j += 1
    C.run_jobs(CR.rank_job, js, 6, "critic ranking: Yield follow-up")
    print("== yield follow-up done", flush=True)
