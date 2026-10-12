"""Phase E of the diagnostics milestone (docs/rl/openai_five_diagnostics_report.md, section F): longer shared-policy
self-play, pure (100/0) versus the OpenAI Five-inspired mixture (80/20), five seeds each.

Same trainer, network, PPO settings, environment (`GameConfig.champions()`, unchanged rules), skill distribution
({0.2, 0.5, 0.8} per seat and episode) and snapshot cadence (25,600 agent transitions) as the main study
(`lanegame.experiments.rl.openai_five`), but 1,024,000 agent transitions per run and seeds 0-4. The learning rate
anneals over the longer budget, so these runs are NOT continuations of the 307,200-transition main runs (whose
first transitions they do not reproduce); both are reported separately, at equal agent transitions.

Evaluation (held-out roots 57,000-57,999; frozen networks; both seats; common random numbers):
  curves    every other snapshot against a fixed panel (hold_farmer, aggressive_trader, the published M3b policy
            pub_sprand_s1 and the main-study policy m3b_s1), 200 episodes per seat
  matrix    every fourth snapshot of a run against every earlier one (forgetting / cycling), 150 episodes per seat
  cross     the 10 final policies against each other and against the 9 main-study policies, 300 episodes per seat
  audit     mirror-game action audit of every final policy at equal and unequal skills (lanegame.diagnostics)
  exploit   one warm-started attacker (the Phase D attacker, `robustness`) per final policy and per 307k-snapshot

    python -m lanegame.experiments.rl.long_selfplay train | evaluate | exploit | report
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List

import numpy as np

from ...config import GameConfig
from .common import md_table
from .openai_five import LABEL, _load as _load_of5, fmt_seed_ci, run_dir, snapshots_of, train_runs

TAG = "long"
BUDGET = 1_024_000
CONDS = ("sp100", "sp80")
SEEDS5 = (0, 1, 2, 3, 4)
OUT = os.path.join("results", "rl", "diagnostics")
ROOT = 57_000
PANEL = ("hold_farmer", "aggressive_trader", "pub_sprand_s1", "m3b_s1")
CURVE_EPISODES, MATRIX_EPISODES, CROSS_EPISODES = 200, 150, 300
MID_STEP = 307_200
WORKERS = 10


def finals() -> Dict[str, str]:
    return {f"long_{c}_s{s}": os.path.join(run_dir(TAG, c, s), "final.pt") for c in CONDS for s in SEEDS5}


def references() -> Dict[str, str]:
    from .openai_five_eval import main_policies, published
    return {**main_policies(), "pub_sprand_s1": published()["pub_sprand_s1"]}


def mid_snapshot(c: str, s: int):
    """The run's snapshot nearest to MID_STEP agent transitions (the main study's budget)."""
    snaps = snapshots_of(run_dir(TAG, c, s))
    return min(snaps, key=lambda tp: abs(tp[0] - MID_STEP))


# ---------------------------------------------------------------------------------------------- train
def train(workers: int = WORKERS):
    runs = train_runs(TAG, BUDGET, conditions=CONDS, seeds=SEEDS5, workers=workers)
    return runs


# ---------------------------------------------------------------------------------------------- evaluate
def evaluate(workers: int = WORKERS):
    from .openai_five_eval import _h2h
    t0 = time.perf_counter()
    refs = references()
    fin = finals()
    jobs, snaps = [], {}
    for c in CONDS:
        for s in SEEDS5:
            n = f"long_{c}_s{s}"
            snaps[n] = snapshots_of(run_dir(TAG, c, s))
            for i, (t, p) in enumerate(snaps[n]):
                if i % 2 == 1 or i == len(snaps[n]) - 1:
                    for o in PANEL:
                        jobs.append((("B", n, t, o), p, refs.get(o, o), CURVE_EPISODES, ROOT + 1, None, "x"))
            sub = [tp for i, tp in enumerate(snaps[n]) if i % 4 == 3 or i == len(snaps[n]) - 1]
            for i, (ti, pi) in enumerate(sub):
                for tj, pj in sub[i + 1:]:
                    jobs.append((("M", n, tj, ti), pj, pi, MATRIX_EPISODES, ROOT + 2, None, "none"))
    names = list(fin)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            jobs.append((("C", a, b), fin[a], fin[b], CROSS_EPISODES, ROOT, None, "none"))
        for m, q in refs.items():
            jobs.append((("C", a, m), fin[a], q, CROSS_EPISODES, ROOT, None, "none"))
    print(f"long evaluate: {len(jobs)} head-to-head jobs", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = dict(ex.map(_h2h, jobs, chunksize=4))
    # mirror-game audits at equal and unequal skill
    from .diagnostics import _audit_job
    ajobs = [((n, "self", sk), p, "self", 150, ROOT + 3, None if sk == "0.5/0.5" else tuple(map(float, sk.split("/"))))
             for n, p in fin.items() for sk in ("0.5/0.5", "0.2/0.8", "0.8/0.2")]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        aud = dict(ex.map(_audit_job, ajobs))
    data = {"policies": fin, "references": refs, "snapshots": snaps, "elapsed_s": time.perf_counter() - t0,
            "results": {"|".join(str(x) for x in k): v for k, v in res.items()},
            "audit": {"|".join(k): v for k, v in aud.items()}}
    _write("long_eval", data, lambda d: f"# long_eval\n\nRaw results; see long_selfplay.md. {len(jobs)} jobs, "
                                        f"{d['elapsed_s'] / 60:.1f} min.\n")
    return data


# ---------------------------------------------------------------------------------------------- exploit
def _att_job(args):
    """One warm-started attacker (the Phase D attacker, fewer checkpoints) against a frozen long-run policy."""
    aid, path, steps, seed = args
    import copy
    import json
    import torch
    from ...env import LaneEnv, run_episodes
    from ...rl import NetAgent, OpponentSampler, PPOConfig, load_agent, load_checkpoint, save_checkpoint, train_ppo
    from .robustness import CK
    torch.set_num_threads(1)
    cfg = GameConfig.champions()
    out = os.path.join(CK, "long_" + aid)
    curve_steps = (steps // 4, steps // 2, steps)
    if not os.path.exists(os.path.join(out, "final.pt")):
        if os.path.isdir(out) and os.listdir(out):
            raise FileExistsError(f"{out} holds an unfinished attacker")
        target = load_agent(path, name=aid)
        net = copy.deepcopy(load_checkpoint(path)[0])
        saved = {}

        def on_update(u, net_, entry):
            if entry["steps"] in curve_steps:
                p = os.path.join(out, "curve", f"att_{entry['steps']:09d}.pt")
                save_checkpoint(p, net_, cfg, {"attacker": aid, "target_path": path, "step": entry["steps"]},
                                exist_ok=False)
        train_ppo(cfg, PPOConfig(total_steps=steps, seed=seed), OpponentSampler([target]), out_dir=out, net=net,
                  on_update=on_update, verbose=False, name=aid)
    target = load_agent(path, name="target")
    env = LaneEnv(cfg, cache_size=20_000)
    curve = []
    for st in curve_steps:
        x = load_agent(os.path.join(out, "curve", f"att_{st:09d}.pt"), name="att")
        eps = 500 if st == steps else 200
        a = run_episodes(env, x, target, eps, ROOT + 4)
        b = run_episodes(env, target, x, eps, ROOT + 4)
        pay = np.concatenate([a.payoffs, -b.payoffs])
        curve.append([st, float(pay.mean()), float(pay.std(ddof=1) / np.sqrt(len(pay))), pay.tolist()])
    return aid, {"curve": curve}


def exploit(steps: int = 1_024_000, workers: int = 6):
    t0 = time.perf_counter()
    jobs = []
    for c in CONDS:
        for s in SEEDS5:
            n = f"long_{c}_s{s}"
            jobs.append((f"{n}_final", os.path.join(run_dir(TAG, c, s), "final.pt"), steps, 920))
            t, p = mid_snapshot(c, s)
            jobs.append((f"{n}_mid", p, steps, 920))
    print(f"long exploit: {len(jobs)} attackers of {steps:,} steps", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = dict(ex.map(_att_job, jobs))
    data = {"steps": steps, "elapsed_s": time.perf_counter() - t0, "results": res,
            "mid_steps": {f"long_{c}_s{s}": mid_snapshot(c, s)[0] for c in CONDS for s in SEEDS5}}
    _write("long_exploit", data, lambda d: f"# long_exploit\n\nRaw results; see long_selfplay.md. "
                                           f"{d['elapsed_s'] / 60:.1f} min.\n")
    return data


# ---------------------------------------------------------------------------------------------- helpers
def _write(name: str, data: dict, md_fn) -> None:
    from .diagnostics import _write as w
    w(name, data, md_fn)


def _load(name: str) -> dict:
    from .diagnostics import _load as ld
    return ld(name)


# ---------------------------------------------------------------------------------------------- report
def _cross(R, a, b):
    k = f"C|{a}|{b}"
    if k in R:
        return R[k]["mean"]
    return -R[f"C|{b}|{a}"]["mean"]


def _fmtp(x):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def report():
    ev = _load("long_eval")
    ex = _load("long_exploit") if os.path.exists(os.path.join(OUT, "long_exploit.json")) else None
    R, A = ev["results"], ev["audit"]
    fin, refs = ev["policies"], ev["references"]
    tr = _load_of5(f"train_{TAG}")
    train = {f"long_{r['condition']}_s{r['seed']}": r for r in tr["runs"]}
    L = ["# Phase E: longer shared-policy self-play (100/0 vs 80/20, 5 seeds, 1,024,000 agent transitions)", "",
         "Generated by `python -m lanegame.experiments.rl.long_selfplay report`. Same environment, trainer, network, "
         "PPO settings and skill distribution as the main study; budget 1,024,000 agent transitions; seeds 0-4. "
         f"Held-out evaluation roots {ROOT}-{ROOT + 4}. Condition `±` = 95% t interval over the 5 training seeds "
         "(t4 = 2.78); per-policy numbers are means over evaluation episodes.", ""]
    rows = []
    for c in CONDS:
        rs = [train[f"long_{c}_s{s}"] for s in SEEDS5]
        rows.append([LABEL[c], f"{np.mean([r['agent_transitions'] for r in rs]):,.0f}",
                     f"{np.mean([r['env_turns'] for r in rs]):,.0f}", f"{np.mean([r['elapsed_s'] for r in rs]):.0f}",
                     "-" if rs[0]["historical_transition_share"] is None else
                     f"{np.mean([r['historical_transition_share'] for r in rs]):.3f}",
                     f"{np.mean([r['diagnostics']['entropy'][-1][1] for r in rs]):.2f}"])
    L += ["## Training cost (mean per seed)", "",
          md_table(["condition", "agent transitions", "environment turns", "train time (s, 1 thread, 10 runs in parallel)",
                    "historical share of transitions", "final entropy"], rows), ""]
    for o in PANEL:
        keys = [k for k in R if k.startswith("B|") and k.endswith("|" + o)]
        steps = sorted({int(round(int(k.split("|")[2]) / 25_600)) * 25_600 for k in keys})
        rows = []
        for st in steps:
            row = [f"{st:,}"]
            for c in CONDS:
                vals = []
                for s in SEEDS5:
                    n = f"long_{c}_s{s}"
                    ks = [k for k in keys if k.split("|")[1] == n and
                          int(round(int(k.split("|")[2]) / 25_600)) * 25_600 == st]
                    if ks:
                        vals.append(R[ks[0]]["mean"])
                row.append(fmt_seed_ci(vals) if len(vals) == len(SEEDS5) else "")
            rows.append(row)
        L += [f"### Snapshots vs `{o}` ({CURVE_EPISODES} episodes per seat)", "",
              md_table(["agent transitions (nominal)"] + [LABEL[c] for c in CONDS], rows), ""]
    rows = []
    for c in CONDS:
        for s in SEEDS5:
            n = f"long_{c}_s{s}"
            M = {(int(k.split("|")[2]), int(k.split("|")[3])): (v["mean"], v["stderr"]) for k, v in R.items()
                 if k.startswith(f"M|{n}|")}
            snaps = sorted({a for a, _ in M} | {b for _, b in M})
            older = sum(1 for (a, b), (m, se) in M.items() if m < -2 * se)
            first, last = snaps[0], snaps[-1]
            trip = ok = 0
            for i in range(len(snaps)):
                for j in range(i + 1, len(snaps)):
                    for k in range(j + 1, len(snaps)):
                        trip += 1
                        a, b = M[(snaps[k], snaps[i])], M[(snaps[j], snaps[i])]
                        ok += a[0] >= b[0] - 2 * np.hypot(a[1], b[1])
            late = [M[(last, b)][0] for b in snaps[:-1] if b >= 512_000]
            rows.append([n, len(M), older, f"{ok / trip if trip else 1:.2f}", f"{M[(last, first)][0]:+.2f}",
                         f"{min(late):+.2f}" if late else "", f"{min(m for m, _ in M.values()):+.2f}"])
    L += ["## Historical checkpoints inside each run (every fourth snapshot)", "",
          "`older wins` = pairs (newer, older) in which the older snapshot wins by > 2 SE; `monotone` = share of "
          "triples i < j < k with payoff(k vs i) >= payoff(j vs i) − 2 SE.", "",
          md_table(["run", "pairs", "older wins", "monotone", "final vs first", "final vs worst snapshot >= 512k",
                    "worst newer-vs-older"], rows), ""]
    main = [n for n in refs if n != "pub_sprand_s1"]
    per = {}
    for n in fin:
        per[n] = {"vs_main": float(np.mean([_cross(R, n, m) for m in main])),
                  "vs_m3b": float(np.mean([_cross(R, n, f"m3b_s{s}") for s in (0, 1, 2)])),
                  "vs_sp100": float(np.mean([_cross(R, n, f"sp100_s{s}") for s in (0, 1, 2)])),
                  "vs_sp80": float(np.mean([_cross(R, n, f"sp80_s{s}") for s in (0, 1, 2)])),
                  "vs_pub": _cross(R, n, "pub_sprand_s1"),
                  "vs_long": float(np.mean([_cross(R, n, m) for m in fin if m != n]))}
    rows = []
    for c in CONDS:
        names = [f"long_{c}_s{s}" for s in SEEDS5]
        rows.append([LABEL[c]] + [fmt_seed_ci([per[n][k] for n in names])
                                  for k in ("vs_main", "vs_m3b", "vs_sp100", "vs_sp80", "vs_pub", "vs_long")])
    d = [per[f"long_sp80_s{s}"]["vs_long"] - per[f"long_sp100_s{s}"]["vs_long"] for s in SEEDS5]
    h2h = [_cross(R, f"long_sp80_s{s}", f"long_sp100_s{s}") for s in SEEDS5]
    wins = sum(_cross(R, f"long_sp80_s{a}", f"long_sp100_s{b}") > 0 for a in SEEDS5 for b in SEEDS5)
    L += ["## Cross-play of the final policies (payoff to the row family)", "",
          md_table(["condition", "vs the 9 main-study (307k) policies", "vs main m3b", "vs main 100/0", "vs main 80/20",
                    "vs pub_sprand_s1", "vs the other long policies"], rows), "",
          f"80/20 − 100/0 (paired by seed) in cross-play against the long pool: {', '.join(f'{x:+.2f}' for x in d)}, "
          f"{fmt_seed_ci(d)}. Direct seed-paired games long_sp80_s<k> vs long_sp100_s<k>: "
          f"{', '.join(f'{x:+.2f}' for x in h2h)}, {fmt_seed_ci(h2h)}. (sp80, sp100) pairs won by 80/20: {wins}/25.",
          ""]
    rows = []
    for c in CONDS:
        for s in SEEDS5:
            n = f"long_{c}_s{s}"
            a = A[f"{n}|self|0.5/0.5"]
            lg, op, st = a["legality"], a["opportunities"], a["states"]
            rows.append([n] + [f"{lg[k]['p_when_legal']:.3f}" for k in ("Hold", "Yield", "Trade", "Safe", "SH")] +
                        [_fmtp(op["recall legal and HP <= 2"]["p"]["Recall"]),
                         _fmtp(op["enemy champion away (punish window)"]["p"]["SH"]),
                         f"{st['deaths_per_game']:.2f}", f"{st['crashes_per_game']:.2f}",
                         f"{st['objective_legal_per_game']:.2f}",
                         f"{A[f'{n}|self|0.2/0.8']['legality']['Trade']['p_when_legal']:.2f}",
                         f"{A[f'{n}|self|0.8/0.2']['legality']['Trade']['p_when_legal']:.2f}"])
    L += ["## Behavior of the final policies (mirror games; probabilities on present turns)", "",
          md_table(["policy", "Hold", "Yield", "Trade", "Safe", "Shove", "Recall given legal, HP <= 2", "Shove given enemy away",
                    "deaths / game", "crashes / game", "objective-legal turns / game", "Trade at skill 0.2 vs 0.8",
                    "Trade at skill 0.8 vs 0.2"], rows), ""]
    if ex is not None:
        E = ex["results"]
        rows = []
        for c in CONDS:
            fins = [E[f"long_{c}_s{s}_final"]["curve"][-1][1] for s in SEEDS5]
            mids = [E[f"long_{c}_s{s}_mid"]["curve"][-1][1] for s in SEEDS5]
            inc = [E[f"long_{c}_s{s}_final"]["curve"][-1][1] - E[f"long_{c}_s{s}_final"]["curve"][-2][1] for s in SEEDS5]
            rows.append([LABEL[c], fmt_seed_ci(mids) + " (" + ", ".join(f"{v:+.2f}" for v in mids) + ")",
                         fmt_seed_ci(fins) + " (" + ", ".join(f"{v:+.2f}" for v in fins) + ")",
                         fmt_seed_ci([f - m for f, m in zip(fins, mids)]), fmt_seed_ci(inc)])
        dd = [E[f"long_sp80_s{s}_final"]["curve"][-1][1] - E[f"long_sp100_s{s}_final"]["curve"][-1][1] for s in SEEDS5]
        L += ["## Approximate exploitability (warm-started attacker, "
              f"{ex['steps']:,} steps, 500 held-out episodes per seat at the end)", "",
              md_table(["condition", "snapshot near 307k transitions", "final (1,024k)", "final − 307k (paired by seed)",
                        "attacker's last increment (half → full budget)"], rows), "",
              f"80/20 − 100/0 final, paired by seed: {', '.join(f'{x:+.2f}' for x in dd)}, {fmt_seed_ci(dd)}.", ""]
    _write("long_selfplay", {"per_policy": per}, "\n".join(L))


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "train"
    {"train": train, "evaluate": evaluate, "exploit": exploit, "report": report}[stage]()
