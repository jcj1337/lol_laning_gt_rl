"""Phase D of the diagnostics milestone (docs/openai_five_diagnostics_report.md, section E): stronger approximate
best responses with attacker learning curves.

The previous study's attackers (102,400 and 307,200 steps) were budget-limited: tripling the budget raised the
lower bound for 5 of 9 targets and reordered them. Here every main-study policy (M3b, shared 100/0, shared 80/20;
3 seeds each) is attacked again with the SAME attacker (the existing `train_ppo` against the frozen target only,
default `PPOConfig`, 16 x 64 rollouts, LR annealed over the attacker's budget) for 1,024,000 steps:

  * two warm-started attackers per target (a copy of the target's own weights: payoff 0 at step 0 by symmetry),
    new training seeds 910 and 911 (the previous study used 900-902);
  * one from-scratch attacker for one target per condition (seed 912).

Each attacker is saved at CURVE_STEPS (a checkpoint is never overwritten) and every checkpoint is evaluated against
its target on held-out seeds (SeedSequence(EVAL_ROOT); disjoint from every training stream and from the previous
studies' evaluation roots 10,000-43,000 and 77,777), both seats, common random numbers across checkpoints, so the
difference between two checkpoints of one attacker is a paired comparison. Per-episode payoffs are kept for that.

An attacker's payoff is a LOWER bound on the target's exploitability, never a certificate. The bound is called
saturated when the last increment of the learning curve (614,400 -> 1,024,000 steps) is within its paired 95%
interval of zero AND below 0.10 resources; otherwise it is reported as still rising.

    python -m lanegame.experiments.rl.robustness train     # 21 attackers x 1,024,000 steps (checkpoints git-ignored)
    python -m lanegame.experiments.rl.robustness evaluate  # learning curves on held-out seeds
    python -m lanegame.experiments.rl.robustness report    # results/rl/diagnostics/robustness.{json,md}
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List

import numpy as np

from ...config import GameConfig
from .common import CKPT, md_table
from .openai_five import CONDITIONS, LABEL, SEEDS, fmt_seed_ci, run_dir

OUT = os.path.join("results", "rl", "diagnostics")
CK = os.path.join(CKPT, "diag", "br")
BR_STEPS = 1_024_000
CURVE_STEPS = (25_600, 51_200, 102_400, 204_800, 307_200, 409_600, 614_400, 819_200, 1_024_000)
WARM_SEEDS = (910, 911)
SCRATCH_SEED = 912
SCRATCH_TARGETS = ("m3b_s0", "sp100_s0", "sp80_s0")
EVAL_ROOT = 50_000
CURVE_EPISODES, FINAL_EPISODES = 200, 500
WORKERS = 6


def targets() -> Dict[str, str]:
    """The nine main-study policies (results/rl/checkpoints/of5/main_<condition>_s<seed>/final.pt)."""
    return {f"{c}_s{s}": os.path.join(run_dir("main", c, s), "final.pt") for c in CONDITIONS for s in SEEDS}


def attackers() -> List[tuple]:
    """(attacker id, target name, target path, mode, training seed)."""
    out = []
    for name, path in targets().items():
        for seed in WARM_SEEDS:
            out.append((f"{name}_warm_{seed}", name, path, "warm", seed))
        if name in SCRATCH_TARGETS:
            out.append((f"{name}_scratch_{SCRATCH_SEED}", name, path, "scratch", SCRATCH_SEED))
    return out


def attacker_dir(aid: str) -> str:
    return os.path.join(CK, aid)


def checkpoint_path(aid: str, step: int) -> str:
    return os.path.join(attacker_dir(aid), "curve", f"att_{step:09d}.pt")


# ---------------------------------------------------------------------------------------------- training
def _train_job(args) -> dict:
    aid, name, path, mode, seed = args
    import torch
    from ...rl import OpponentSampler, PPOConfig, load_agent, load_checkpoint, save_checkpoint, train_ppo
    torch.set_num_threads(1)
    cfg = GameConfig.champions()
    out = attacker_dir(aid)
    final = os.path.join(out, "final.pt")
    if os.path.exists(final):
        return {"attacker": aid, "skipped": True}
    if os.path.isdir(out) and os.listdir(out):
        raise FileExistsError(f"{out} holds an unfinished attacker; inspect and remove it before retraining")
    target = load_agent(path, name=name)                       # frozen NetAgent (no gradients)
    net = copy.deepcopy(load_checkpoint(path)[0]) if mode == "warm" else None
    ppo = PPOConfig(total_steps=BR_STEPS, seed=seed)
    meta = {"attacker": aid, "target": name, "target_path": path, "mode": mode, "seed": seed, "ppo": ppo.to_dict()}
    if net is not None:                                        # step 0 of a warm attacker = the target itself
        save_checkpoint(checkpoint_path(aid, 0), net, cfg, {**meta, "step": 0}, exist_ok=False)

    def on_update(u, net_, entry):
        if entry["steps"] in CURVE_STEPS:
            save_checkpoint(checkpoint_path(aid, entry["steps"]), net_, cfg, {**meta, "step": entry["steps"]},
                            exist_ok=False)

    t0 = time.perf_counter()
    res = train_ppo(cfg, ppo, OpponentSampler([target]), out_dir=out, net=net, on_update=on_update, verbose=False,
                    name=aid)
    with open(os.path.join(out, "attacker.json"), "w", encoding="utf-8") as f:
        json.dump({**meta, "elapsed_s": time.perf_counter() - t0, "final_entropy": res.history[-1]["entropy"]}, f,
                  indent=1)
    return {"attacker": aid, "skipped": False, "elapsed_s": time.perf_counter() - t0}


def train(workers: int = WORKERS, only=None):
    jobs = [a for a in attackers() if only is None or a[0] in only]
    t0 = time.perf_counter()
    print(f"robustness train: {len(jobs)} attackers of {BR_STEPS:,} steps on {workers} processes", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_train_job, jobs):
            print(f"  {r['attacker']}: " + ("existing" if r["skipped"] else f"{r['elapsed_s']:.0f} s"), flush=True)
    print(f"robustness train: {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


# ---------------------------------------------------------------------------------------------- evaluation
def _eval_job(args):
    """Payoff of one attacker checkpoint against its target: per-episode payoffs (seat A games then seat B games,
    same episode seeds), so that checkpoints of one attacker can be compared pairwise."""
    aid, step, att_path, target_path, episodes = args
    import torch
    from ...env import LaneEnv, run_episodes
    from ...rl import load_agent
    torch.set_num_threads(1)
    cfg = GameConfig.champions()
    x, y = load_agent(att_path, name=f"{aid}@{step}"), load_agent(target_path, name="target")
    env = LaneEnv(cfg, cache_size=20_000)
    st_a = run_episodes(env, x, y, episodes, EVAL_ROOT)
    st_b = run_episodes(env, y, x, episodes, EVAL_ROOT)
    pay = np.concatenate([st_a.payoffs, -st_b.payoffs])
    out = {"mean": float(pay.mean()), "stderr": float(pay.std(ddof=1) / np.sqrt(len(pay))), "payoffs": pay.tolist(),
           "episodes": int(len(pay))}
    if step == BR_STEPS:                                      # behavior of the final attacker
        from ...rl.evaluate import _seat_stats
        sa, sb = _seat_stats(st_a, 0), _seat_stats(st_b, 1)
        out["behaviors"] = {k: 0.5 * (sa["behaviors"].get(k, 0.0) + sb["behaviors"].get(k, 0.0))
                            for k in set(sa["behaviors"]) | set(sb["behaviors"])}
        out["waves"] = {k: 0.5 * (sa["waves"].get(k, 0.0) + sb["waves"].get(k, 0.0)) for k in sa["waves"]}
        out["objectives"] = {k: 0.5 * (sa["objectives"][k] + sb["objectives"][k]) for k in sa["objectives"]}
        out["recalls"] = 0.5 * (sa["recalls"] + sb["recalls"])
        out["deaths"] = 0.5 * (sa.get("deaths", 0.0) + sb.get("deaths", 0.0))
        out["farm"] = 0.5 * (sa["farm"] + sb["farm"])
        out["plates"] = 0.5 * (sa["plates"] + sb["plates"])
    return (aid, step), out


def evaluate(workers: int = 5):
    t0 = time.perf_counter()
    jobs = []
    for aid, name, path, mode, seed in attackers():
        for step in ((0,) if mode == "warm" else ()) + CURVE_STEPS:
            p = checkpoint_path(aid, step)
            if not os.path.exists(p):
                raise SystemExit(f"missing attacker checkpoint {p}; run the train stage first")
            jobs.append((aid, step, p, path, FINAL_EPISODES if step in (0, BR_STEPS) else CURVE_EPISODES))
    print(f"robustness evaluate: {len(jobs)} checkpoint evaluations", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = dict(ex.map(_eval_job, jobs, chunksize=2))
    data = {"eval_root": EVAL_ROOT, "curve_steps": list(CURVE_STEPS), "elapsed_s": time.perf_counter() - t0,
            "attackers": [list(a) for a in attackers()], "results": {f"{a}|{s}": v for (a, s), v in res.items()}}
    _write("robustness_eval", data, f"# robustness_eval\n\nRaw attacker-curve evaluations; see robustness.md. "
                                    f"{len(jobs)} jobs, {data['elapsed_s'] / 60:.1f} min.\n")
    return data


# ---------------------------------------------------------------------------------------------- report
def paired_diff(p: List[float], q: List[float], n_common: int = None) -> tuple:
    """Mean and 95% half-width of p - q over the common episode seeds. Payoff vectors hold seat-A games then seat-B
    games; evaluations with different episode counts share the first min(n) seeds of each seat block."""
    p, q = np.asarray(p), np.asarray(q)
    n = min(len(p), len(q)) // 2
    d = np.concatenate([p[:n] - q[:n], p[len(p) // 2:len(p) // 2 + n] - q[len(q) // 2:len(q) // 2 + n]])
    return float(d.mean()), float(1.96 * d.std(ddof=1) / np.sqrt(len(d)))


def _prior() -> Dict[str, dict]:
    """The previous study's attacker values for the same targets (results/rl/openai_five)."""
    out = {}
    base = os.path.join("results", "rl", "openai_five")
    try:
        with open(os.path.join(base, "exploit_main.json"), encoding="utf-8") as f:
            ex = json.load(f)["results"]
        with open(os.path.join(base, "exploit_strong.json"), encoding="utf-8") as f:
            st = json.load(f)["results"]
    except FileNotFoundError:
        return out
    for n in targets():
        out[n] = {"warm_102k": max(ex[f"{n}|warm|0"]["value"], ex[f"{n}|warm|1"]["value"]),
                  "scratch_102k": ex[f"{n}|scratch|0"]["value"], "warm_307k": st[n]["strong"]}
    return out


def summarize() -> dict:
    with open(os.path.join(OUT, "robustness_eval.json"), encoding="utf-8") as f:
        ev = json.load(f)
    R = ev["results"]
    per = {}
    for aid, name, path, mode, seed in attackers():
        steps = ((0,) if mode == "warm" else ()) + CURVE_STEPS
        curve = [(s, R[f"{aid}|{s}"]["mean"], R[f"{aid}|{s}"]["stderr"]) for s in steps]
        last, prev = R[f"{aid}|{BR_STEPS}"], R[f"{aid}|{CURVE_STEPS[-3]}"]
        inc, inc_h = paired_diff(last["payoffs"], prev["payoffs"])
        best = max(curve[1:], key=lambda c: c[1])
        per[aid] = {"target": name, "mode": mode, "seed": seed, "curve": curve, "final": last["mean"],
                    "final_se": last["stderr"], "last_increment": inc, "last_increment_ci": inc_h,
                    "saturated": bool(abs(inc) <= inc_h and inc < 0.10), "best_step": best[0], "best": best[1],
                    "final_behavior": {k: last.get(k) for k in ("behaviors", "waves", "objectives", "recalls", "deaths",
                                                                "farm", "plates")}}
    tgt = {}
    for name in targets():
        warm = [per[a] for a in per if per[a]["target"] == name and per[a]["mode"] == "warm"]
        scr = [per[a] for a in per if per[a]["target"] == name and per[a]["mode"] == "scratch"]
        tgt[name] = {"warm_finals": [w["final"] for w in warm], "lower_bound": max(w["final"] for w in warm + scr),
                     "scratch_final": scr[0]["final"] if scr else None,
                     "saturated": all(w["saturated"] for w in warm),
                     "seed_gap": abs(warm[0]["final"] - warm[1]["final"]) if len(warm) == 2 else None}
    return {"per": per, "targets": tgt, "prior": _prior(), "ev": ev}


def report():
    S = summarize()
    per, tgt, prior = S["per"], S["targets"], S["prior"]
    lines = ["# Stronger approximate best responses (Phase D)", "",
             f"Generated by `python -m lanegame.experiments.rl.robustness report`. Attackers: the existing PPO "
             f"best-response trainer (`train_ppo` against the frozen target only, default `PPOConfig`), "
             f"{BR_STEPS:,} steps each; warm-started from the target's weights (training seeds {WARM_SEEDS}) for every "
             f"main-study target, from scratch (seed {SCRATCH_SEED}) for {', '.join(SCRATCH_TARGETS)}. Checkpoints "
             f"evaluated against the target on held-out seeds SeedSequence({EVAL_ROOT}), both seats, "
             f"{CURVE_EPISODES} episodes per seat ({FINAL_EPISODES} at step 0 and at the end). Payoff to the attacker "
             "= lower bound on the target's exploitability (never a certificate). The learning rate anneals over the "
             "whole 1,024,000-step budget, so an intermediate checkpoint is not the same as a separately trained "
             "shorter attacker.", ""]
    hdr = ["attacker"] + [f"{s // 1000}k" if s else "0" for s in (0,) + CURVE_STEPS] + \
          [f"last increment ({CURVE_STEPS[-3] // 1000}k→{BR_STEPS // 1000}k, paired 95%)", "saturated"]
    rows = []
    for aid, p in per.items():
        vals = {s: m for s, m, _ in p["curve"]}
        rows.append([aid] + [f"{vals[s]:+.2f}" if s in vals else "" for s in (0,) + CURVE_STEPS] +
                    [f"{p['last_increment']:+.2f} ± {p['last_increment_ci']:.2f}", "yes" if p["saturated"] else "no"])
    lines += ["## Attacker learning curves (payoff to the attacker)", "", md_table(hdr, rows), ""]
    rows = []
    for name, t in tgt.items():
        pr = prior.get(name, {})
        rows.append([name, f"{pr.get('warm_102k', float('nan')):+.2f}", f"{pr.get('warm_307k', float('nan')):+.2f}",
                     ", ".join(f"{v:+.2f}" for v in t["warm_finals"]),
                     "" if t["scratch_final"] is None else f"{t['scratch_final']:+.2f}",
                     f"{t['lower_bound']:+.2f}", "yes" if t["saturated"] else "no"])
    lines += ["## Exploitability lower bounds per target", "",
              md_table(["target", "previous best warm, 102k", "previous warm, 307k", f"warm attackers, {BR_STEPS // 1000}k",
                        f"scratch, {BR_STEPS // 1000}k", "lower bound (max)", "both warm curves saturated"], rows), ""]
    rows = []
    for c in CONDITIONS:
        names = [f"{c}_s{s}" for s in SEEDS]
        if not all(n in tgt for n in names):
            continue
        rows.append([LABEL[c], fmt_seed_ci([prior[n]["warm_102k"] for n in names]) if prior else "",
                     fmt_seed_ci([prior[n]["warm_307k"] for n in names]) if prior else "",
                     fmt_seed_ci([tgt[n]["lower_bound"] for n in names])])
    lines += ["## Condition means (95% t interval over the 3 training seeds of the targets)", "",
              md_table(["condition", "102k (previous)", "307k (previous)", f"{BR_STEPS // 1000}k (this study)"], rows), ""]
    from scipy.stats import spearmanr
    names = list(targets())
    if prior and len(names) > 2:
        r1 = float(spearmanr([prior[n]["warm_307k"] for n in names], [tgt[n]["lower_bound"] for n in names])[0])
        r0 = float(spearmanr([prior[n]["warm_102k"] for n in names], [tgt[n]["lower_bound"] for n in names])[0])
        lines += [f"Spearman rank correlation of the 9 targets' bounds: 102k vs {BR_STEPS // 1000}k {r0:+.2f}; "
                  f"307k vs {BR_STEPS // 1000}k {r1:+.2f}.", ""]
    rows = []
    for aid, p in per.items():
        b = p["final_behavior"]
        if not b.get("behaviors"):
            continue
        beh = b["behaviors"]
        rows.append([aid] + [f"{beh.get(k, 0.0):.2f}" for k in ("H", "Y", "T", "S", "R")] +
                    [f"{b['waves'].get('SH', 0.0):.2f}", f"{b['waves'].get('F', 0.0):.2f}",
                     f"{b['objectives']['Pressure']:.3f}", f"{b['objectives']['Vision']:.3f}", f"{b['recalls']:.2f}",
                     f"{b['deaths']:.2f}", f"{b['farm']:.2f}", f"{b['plates']:.2f}"])
    lines += ["## Behavior of the final attackers (share of turns; per game)", "",
              md_table(["attacker", "Hold", "Yield", "Trade", "Safe", "Recall", "Shove", "Freeze", "Pressure", "Vision",
                        "recalls", "deaths", "farm", "plates"], rows), ""]
    _write("robustness", {"per_attacker": per, "targets": tgt, "prior": prior}, "\n".join(lines))


def _write(name: str, data: dict, md: str) -> None:
    from ...io import dump_json
    os.makedirs(OUT, exist_ok=True)
    dump_json(data, os.path.join(OUT, f"{name}.json"))
    with open(os.path.join(OUT, f"{name}.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(f"wrote {OUT}/{name}.json and .md", flush=True)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "train"
    {"train": train, "evaluate": evaluate, "report": report}[stage]()
