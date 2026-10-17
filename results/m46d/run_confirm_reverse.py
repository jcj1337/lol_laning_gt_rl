"""M4.6d helper: run the confirmation jobs in REVERSE order (a second runner beside `confirm`, so the two meet in the
middle; each job is skipped when its file exists, results are deterministic). Usage: python results/m46d/run_confirm_reverse.py N"""
import sys

from lanegame.experiments.rl.m46d import analysis as AN
from lanegame.experiments.rl.m46d import common as C
from lanegame.experiments.rl.m46d import matrix as M

if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    sel = AN.selected_contrasts()
    js = [M.mk("conf", it["arm"], it["spec"], it["tag"], it["idx"], C.state_from_list(it["start"]), it["a"], it["b"],
               it["combos"], it["K"], it["H"], C.CONF_ROOT, it["label"]) for it in sel["jobs"]]
    js = sorted(js, key=lambda j: -len(j["combos"]) * j["K"] * j["H"])[::-1]
    C.run_jobs(M.matrix_job, js, workers, "confirmation (reverse)")
    print("== reverse confirm done", flush=True)
