"""M3c: evaluation of the trained policies and approximate exploitability.

Policies: the M3a mixture policies (`mix`), the M3b equal-skill self-play policies (`sp`) and the M3b
skill-randomized self-play policies (`sprand`), three seeds each. Every evaluation is a both-seat head-to-head on
held-out seeds (SeedSequence(30000); never used in training), the same seeds for every pair (common random
numbers), with 95% confidence intervals.

1. PPO versus the scripted suite (and uniform).
2. PPO versus PPO (all nine policies, cross-seed) and each self-play policy versus its own historical snapshots.
3. Mechanical skill: the skill-randomized policies at every (own, enemy) skill pair in {0.2, 0.5, 0.8}^2 against
   a scripted reference and against another seed of themselves: payoff, trading, exposure, deaths.
4. Approximate best responses: a fresh PPO learner trained against each frozen target (two training seeds),
   evaluated on held-out seeds, compared with the best scripted response to the same target. This bounds the
   target's exploitability FROM BELOW only; it is not a certificate (the exact, certified numbers exist only for
   the legacy configuration: see m3a_legacy_exact and m3b_selfplay section 3).
"""

from __future__ import annotations

import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List

import numpy as np

from ...config import GameConfig
from .common import CKPT, SCRIPTED, ci, md_table, mean_ci, write

SEEDS = [0, 1, 2]
EVAL_SEED = 30_000
SKILLS = (0.2, 0.5, 0.8)


def _policies() -> Dict[str, str]:
    out = {}
    for s in SEEDS:
        out[f"mix_s{s}"] = os.path.join(CKPT, f"m3a_mixture_s{s}", "final.pt")
        out[f"sp_s{s}"] = os.path.join(CKPT, f"m3b_equal_s{s}", "final.pt")
        out[f"sprand_s{s}"] = os.path.join(CKPT, f"m3b_skillrand_s{s}", "final.pt")
    return out


def _resolve(spec: str, cfg):
    from ...env import UniformAgent, champion_agents
    from ...rl import load_agent
    if spec.endswith(".pt"):
        return load_agent(spec)
    if spec == "uniform":
        return UniformAgent()
    return champion_agents(cfg)[spec]


def _h2h(args):
    """Worker: head_to_head(x, y) with x, y given as scripted names or checkpoint paths."""
    key, x, y, episodes, seed, skill = args
    import torch
    from ...rl import head_to_head
    torch.set_num_threads(1)
    cfg = GameConfig.champions()
    r = head_to_head(cfg, _resolve(x, cfg), _resolve(y, cfg), episodes, seed, skill)
    return key, {k: r[k] for k in ("mean", "stderr", "ci95", "win_rate", "loss_rate", "tie_rate", "seat_A", "seat_B",
                                   "x_stats", "y_stats", "skill", "episodes")}


def _br(args):
    """Worker: approximate best response to a frozen target."""
    target, seed, steps, eval_eps = args
    import torch
    from ...rl import PPOConfig, approximate_best_response
    torch.set_num_threads(1)
    cfg = GameConfig.champions()
    r = approximate_best_response(cfg, _resolve(target, cfg), PPOConfig(total_steps=steps, seed=500 + seed),
                                  eval_episodes=eval_eps, eval_seed=EVAL_SEED + 1, name=f"br_s{seed}")
    r.pop("agent")
    r["eval"] = {k: r["eval"][k] for k in ("mean", "stderr", "ci95", "win_rate", "x_stats")}
    return (target, seed), r


def run(episodes: int = 300, br_steps: int = 102_400, br_eval: int = 500, workers: int = 10):
    t0 = time.perf_counter()
    pols = _policies()
    missing = [p for p in pols.values() if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"missing checkpoints (run experiments-rl scripted and selfplay first): {missing}")
    names = list(pols)
    jobs = []
    # 1. PPO vs scripted
    for n in names:
        for o in SCRIPTED + ["uniform"]:
            jobs.append((("vs_scripted", n, o), pols[n], o, episodes, EVAL_SEED, None))
    # 2. PPO vs PPO (upper triangle) and self-play vs own snapshots
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            jobs.append((("ppo", a, b), pols[a], pols[b], episodes, EVAL_SEED, None))
    snap = {}
    for s in SEEDS:
        files = sorted(glob.glob(os.path.join(CKPT, f"m3b_equal_s{s}", "snapshots", "snap_*.pt")))
        snap[s] = files
        for f in files:
            step = int(os.path.basename(f)[5:-3])
            jobs.append((("snap", f"sp_s{s}", step), pols[f"sp_s{s}"], f, episodes, EVAL_SEED, None))
    # 3. skill grid (skill-randomized policies; also the equal-skill policies for contrast)
    for s in SEEDS:
        other = f"sprand_s{(s + 1) % len(SEEDS)}"
        for mx in SKILLS:
            for my in SKILLS:
                jobs.append((("skill_ref", f"sprand_s{s}", mx, my), pols[f"sprand_s{s}"], "hold_farmer", episodes,
                             EVAL_SEED, (mx, my)))
                jobs.append((("skill_self", f"sprand_s{s}", mx, my), pols[f"sprand_s{s}"], pols[other], episodes,
                             EVAL_SEED, (mx, my)))
                jobs.append((("skill_sp", f"sp_s{s}", mx, my), pols[f"sp_s{s}"], "hold_farmer", episodes,
                             EVAL_SEED, (mx, my)))
    # 4. approximate best responses (+ best scripted response to each target, from section 1 or computed here)
    targets = {"mix_s0": pols["mix_s0"], "sp_s0": pols["sp_s0"], "sp_s1": pols["sp_s1"], "sp_s2": pols["sp_s2"],
               "sprand_s0": pols["sprand_s0"], "hold_farmer": "hold_farmer", "punisher": "punisher"}
    for t, spec in targets.items():
        if spec.endswith(".pt"):
            continue
        for o in SCRIPTED + ["uniform"]:
            jobs.append((("scripted_br", t, o), o, spec, episodes, EVAL_SEED, None))
    br_jobs = [(spec, s, br_steps, br_eval) for spec in targets.values() for s in (0, 1)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        fut_br = [ex.submit(_br, j) for j in br_jobs]
        res = dict(ex.map(_h2h, jobs, chunksize=2))
        br = dict(f.result() for f in fut_br)
    elapsed = time.perf_counter() - t0
    spec_name = {v: k for k, v in targets.items()}

    # ---------- 1. vs scripted
    opp_list = SCRIPTED + ["uniform"]
    rows = []
    for n in names:
        vals = [res[("vs_scripted", n, o)]["mean"] for o in opp_list]
        rows.append([n, f"{np.mean(vals[:-1]):+.2f}", f"{min(vals[:-1]):+.2f}"] + [f"{v:+.2f}" for v in vals])
    sec1 = md_table(["policy", "average vs 13 scripted", "worst"] + opp_list, rows)
    fam_rows = []
    for fam in ("mix", "sp", "sprand"):
        avgs = [np.mean([res[("vs_scripted", f"{fam}_s{s}", o)]["mean"] for o in SCRIPTED]) for s in SEEDS]
        worst = [min(res[("vs_scripted", f"{fam}_s{s}", o)]["mean"] for o in SCRIPTED) for s in SEEDS]
        st = [res[("vs_scripted", f"{fam}_s{s}", "hold_farmer")]["x_stats"] for s in SEEDS]
        beh = {k: np.mean([x["behaviors"].get(k, 0.0) for x in st]) for k in ("H", "Y", "T", "S", "R", "ABSENT")}
        fam_rows.append([fam, mean_ci(avgs), mean_ci(worst),
                         f"{np.mean([x['farm'] for x in st]):.2f}", f"{np.mean([x.get('deaths', 0) for x in st]):.2f}",
                         f"{np.mean([x['trades_connected'] for x in st]):.1f}", f"{np.mean([x['recalls'] for x in st]):.2f}",
                         f"{np.mean([x['exposure'] for x in st]):+.2f}",
                         " ".join(f"{k} {v:.2f}" for k, v in beh.items() if v >= 0.005)])
    # ---------- 2. PPO vs PPO
    M = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i < j:
                M[i, j] = res[("ppo", a, b)]["mean"]
                M[j, i] = -M[i, j]
    sec2 = md_table(["row vs column"] + names + ["average"],
                    [[a] + [f"{M[i, j]:+.2f}" if i != j else "" for j in range(len(names))] + [f"{M[i].sum() / (len(names) - 1):+.2f}"]
                     for i, a in enumerate(names)])
    srows = []
    for s in SEEDS:
        for f in snap[s]:
            step = int(os.path.basename(f)[5:-3])
            r = res[("snap", f"sp_s{s}", step)]
            srows.append([f"sp_s{s}", f"{step:,}", ci(r["mean"], r["stderr"])])
    # ---------- 3. skill grid
    def grid(kind, fam):
        rows = []
        for mx in SKILLS:
            row = [f"{mx:.1f}"]
            for my in SKILLS:
                vals = [res[(kind, f"{fam}_s{s}", mx, my)]["mean"] for s in SEEDS]
                row.append(mean_ci(vals))
            rows.append(row)
        return md_table(["own skill \\ enemy skill"] + [f"{m:.1f}" for m in SKILLS], rows)

    def behavior_grid(kind, fam, field):
        rows = []
        for mx in SKILLS:
            row = [f"{mx:.1f}"]
            for my in SKILLS:
                vals = []
                for s in SEEDS:
                    x = res[(kind, f"{fam}_s{s}", mx, my)]["x_stats"]
                    vals.append(x["behaviors"].get("T", 0.0) if field == "trade_share" else x.get(field, 0.0))
                row.append(f"{np.mean(vals):.2f}")
            rows.append(row)
        return md_table(["own skill \\ enemy skill"] + [f"{m:.1f}" for m in SKILLS], rows)
    # ---------- 4. approximate best responses
    brows = []
    br_summary = {}
    for t, spec in targets.items():
        vals = [br[(spec, s)] for s in (0, 1)]
        if spec.endswith(".pt"):
            scripted = {o: res[("vs_scripted", t, o)]["mean"] for o in SCRIPTED + ["uniform"]}
            best_o = min(scripted, key=lambda o: scripted[o])                # target's worst result = best response
            best_v = -scripted[best_o]
            best_se = res[("vs_scripted", t, best_o)]["stderr"]
        else:
            scripted = {o: res[("scripted_br", t, o)]["mean"] for o in SCRIPTED + ["uniform"]}
            best_o = max(scripted, key=lambda o: scripted[o])
            best_v = scripted[best_o]
            best_se = res[("scripted_br", t, best_o)]["stderr"]
        br_summary[t] = {"br_values": [v["value"] for v in vals], "br_stderr": [v["stderr"] for v in vals],
                         "best_scripted": best_o, "best_scripted_value": best_v,
                         "curves": [v["curve"] for v in vals], "train_steps": vals[0]["train_steps"]}
        brows.append([t, ", ".join(ci(v["value"], v["stderr"]) for v in vals), f"{max(v['value'] for v in vals):+.2f}",
                      f"{best_o} {ci(best_v, best_se)}",
                      f"{max(max(v['value'] for v in vals), best_v):+.2f}"])
    md = "\n".join([
        "# M3c: evaluation of the trained policies and approximate exploitability", "",
        f"Generated by `python -m lanegame experiments-rl evaluation` in {elapsed / 60:.1f} min ({workers} processes). "
        f"Champion game, K = 32. Every number is a payoff to the first-named (row) policy per game over BOTH seats, "
        f"{episodes} held-out episodes per seat (SeedSequence({EVAL_SEED}), never used in training), common random "
        "numbers across pairs, 95% confidence intervals (half-widths are ~0.1-0.3 per cell unless shown). "
        "Policies: `mix_s*` = M3a learners trained against the 13 scripted agents; `sp_s*` = M3b self-play "
        "(initialized from `mix_s*`); `sprand_s*` = M3b self-play with randomized skills.", "",
        "## 1. PPO versus the scripted suite", "",
        md_table(["family", "average vs 13 scripted (3 seeds)", "worst matchup", "farm vs hold_farmer",
                  "deaths", "connected trades", "recalls", "mean exposure", "behaviors vs hold_farmer"], fam_rows), "",
        sec1, "",
        "## 2. PPO versus PPO", "",
        "Cross-play matrix (payoff to the row policy):", "", sec2, "",
        "Each equal-skill self-play policy against its own historical snapshots (step 0 = its M3a initialization):", "",
        md_table(["policy", "snapshot step", "payoff vs snapshot"], srows), "",
        "## 3. Mechanical skill (policies trained with randomized skills; skill of the evaluated policy x enemy skill)", "",
        "Payoff against `hold_farmer` (mean over 3 seeds ± CI across seeds):", "", grid("skill_ref", "sprand"), "",
        "Payoff against another skill-randomized seed (mirror matchups have equal skill on the diagonal):", "",
        grid("skill_self", "sprand"), "",
        "Share of turns with the Trade behavior against another skill-randomized seed:", "",
        behavior_grid("skill_self", "sprand", "trade_share"), "",
        "Mean exposure cell against another skill-randomized seed:", "", behavior_grid("skill_self", "sprand", "exposure"), "",
        "Deaths per game against another skill-randomized seed:", "", behavior_grid("skill_self", "sprand", "deaths"), "",
        "For contrast, the equal-skill self-play policies (never trained at other skills) against `hold_farmer`:", "",
        grid("skill_sp", "sp"), "",
        f"## 4. Approximate best responses ({br_steps:,} PPO steps per response, two training seeds, "
        f"{br_eval} held-out episodes per seat)", "",
        "A larger value means a more exploitable target. The PPO response and the best scripted response are both "
        "lower bounds on the target's exploitability; their maximum is the best available estimate. The symmetric "
        "game's value at equal skill is 0, so an unexploitable policy would score <= 0 against every response.", "",
        md_table(["target", "PPO best response (per training seed)", "best PPO response", "best scripted response",
                  "lower bound on exploitability"], brows), ""])
    data = {"policies": pols, "results": {"|".join(str(x) for x in k): v for k, v in res.items()},
            "best_responses": {f"{k[0]}|{k[1]}": v for k, v in br.items()}, "br_summary": br_summary,
            "elapsed_s": elapsed, "episodes": episodes, "br_steps": br_steps}
    write("m3c_evaluation", data, md)
    return data


if __name__ == "__main__":
    run()
