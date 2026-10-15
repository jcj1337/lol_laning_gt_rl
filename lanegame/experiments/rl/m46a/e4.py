"""E4: learned best responses trained from power-gap and freeze starts (pre-registration §2 E4).

Protocol (the M4.5c attacker protocol, `m45c.attackers`, with one change): warm start from the target's weights, PPO
against the frozen target only, 4,096,000 learner transitions, the M4.5c PPO settings (`m45c.common.ppo_config`,
lane cache 3,000: an exact memo), except that every training game starts from a start archive (`start_mix` 1.0), lasts
GAME_LENGTH = 150 turns (bootstrapped truncation; the first game of each environment a seeded random length) and the
learner always plays the archive's FOCAL side: `FocalCollector` mirrors the drawn archive state when the learner sits
in seat B. With start_mix 0 `FocalCollector` is exactly `Collector`.

Archives (per target; constructed states repeated to half of the archive, labelled):
    gap      M4.5b zoning situations neutral / a_push / b_push x levels 6 / 10 / 14 x A's lead x1.5 / x1.76 / x2.2,
             plus the semi-natural states of the target's E3 TRAINING-split games (stronger side focal)
    freeze   the M4.5b freeze builder (levels 4 / 10 / 15 x equal / x1.76 x held 5 / 12 / 20), plus natural freeze
             states of the target's self-play (seeds E4_TRAIN_ROOT x 10 + 7000 + g), the freezer focal
Evaluation (held out): the constructed cells with evaluation seeds (K_CON) and HELD-OUT states (gap: E3's held-out
games; freeze: self-play seeds E4_EVAL_ROOT x 10 + 7000 + g), K_NAT each; the target in seat B; seat A played by the
target ("target", the baseline), the attacker, and the same target's M4.5c s0-attacker ("s0att", the control); plus the
gap attacker on the equal-power versions of the constructed cells, and every attacker against its target from s0
(`evaluate_discounted`, 64 paired seeds, W = 460).
"""

from __future__ import annotations

import os
import time
from typing import Dict, List

import numpy as np

from . import engine as EN
from . import families as FA
from .common import (CKPT, E4_EVAL_ROOT, E4_TARGETS, E4_TRAIN_ROOT, env_cfg, load_json, mean_ci, meta, run_jobs,
                     s0_attacker_path, save_json, setup_worker, significant, spec)

BUDGET = 4_096_000
GAME_LENGTH = 150
TRAIN_CACHE = 3000
KINDS = ("gap", "freeze")
H_EVAL = {"gap": 150, "freeze": 100}
PRIMARY_H = {"gap": 100, "freeze": 50}
K_CON, K_NAT, N_EVAL_NAT = 16, 8, 48
N_FREEZE_TRAIN = 240
GAP_CELLS = [(sit, L, gap) for sit in ("neutral", "a_push", "b_push") for L in (6, 10, 14) for gap in ("x1.5", "x1.76", "x2.2")]
EQUAL_CELLS = [(sit, L, "equal") for sit in ("neutral", "a_push", "b_push") for L in (6, 10, 14)]
FREEZE_CELLS = [(L, gap, held) for L in (4, 10, 15) for gap in ("equal", "x1.76") for held in (5, 12, 20)]


def attacker_dir(kind: str, target: int, root: str = CKPT) -> str:
    return os.path.join(root, "e4", f"{kind}_s{target}")


def archive_path(kind: str, target: int, split: str) -> str:
    return os.path.join(CKPT, "e4", "archives", f"{kind}_s{target}_{split}.json.gz")


# ---------------------------------------------------------------------------------------------- seat-aware starts
def focal_collector_cls():
    from ....rl.train import Collector

    class FocalCollector(Collector):
        """Collector whose archive starts put the learner on the archive's focal side (seat A of the stored state):
        the state is mirrored when the learner draws seat B. Every draw is GameTimes.reset's."""

        def _reset(self, i: int) -> None:
            from ....rl.train import EpisodeLog, episode_streams
            ss = self._episode_seeds.spawn(1)[0]
            self._episodes_started += 1
            _, env_ss, rng_a, rng_b = episode_streams(ss)
            self.opp[i] = self.sampler.sample(self.rng)
            self.seat[i] = int(self.rng.integers(2))
            if self.cfg.skill_levels is not None:
                raise ValueError("FocalCollector does not support skill_levels")
            self.opp_rng[i] = rng_b if self.seat[i] == 0 else rng_a
            self._game_opp[i] = self.opp[i]
            tm, env = self.times, self.envs[i]
            if not tm.continuing:
                self.obs[i] = env.reset(seed=env_ss)
            else:
                env.max_turns = tm.first_lengths[i] if tm.games_started[i] == 0 else tm.game_length
                tm.games_started[i] += 1
                if tm.start_mix > 0 and tm.start_rng.random() < tm.start_mix:
                    st = tm.archive[int(tm.start_rng.integers(len(tm.archive)))]
                    tm.start_counts["archive"] += 1
                    if self.seat[i] == 1:
                        st = st.mirror()
                        tm.start_counts["mirrored"] = tm.start_counts.get("mirrored", 0) + 1
                    self.obs[i] = env.reset(seed=env_ss, root=st.lane, k0=st.phase, champ=st.champ)
                else:
                    tm.start_counts["initial"] += 1
                    self.obs[i] = env.reset(seed=env_ss)
            names = ("learner", getattr(self.opp[i], "name", "opponent"))
            self.logs[i] = EpisodeLog(seed=None, k0=0, horizon=env.horizon,
                                      names=names if self.seat[i] == 0 else names[::-1])
            self.ret[i] = 0.0
    return FocalCollector


# ---------------------------------------------------------------------------------------------- archives
def _freeze_states_job(args) -> tuple:
    target, n, root, offset = args
    from .states import collect_job
    return collect_job((target, "freeze", n, root, offset, 2))


def build_archives(workers: int, targets=E4_TARGETS) -> dict:
    """Write the training and evaluation archives / state sets of every target (idempotent)."""
    from ....env.starts import StartState, save_archive
    from .common import save_gz, state_to_list
    from .e3 import semi_natural
    from .states import freeze_gap_state, zoning_gap_state
    cfg = env_cfg()
    need = [t for t in targets if not os.path.exists(archive_path("freeze", t, "eval"))]
    fr = {}
    if need:
        js = [(t, N_FREEZE_TRAIN, E4_TRAIN_ROOT, 7000) for t in need] + \
             [(t, N_EVAL_NAT, E4_EVAL_ROOT, 7000) for t in need]
        res = run_jobs(_freeze_states_job, js, workers, "E4 freeze states")
        for (k, v), j in zip(res, js):
            fr[(j[0], "train" if j[2] == E4_TRAIN_ROOT else "eval")] = v
    desc = {}
    for t in targets:
        if os.path.exists(archive_path("freeze", t, "eval")):
            continue
        con_gap = [zoning_gap_state(cfg, *c) for c in GAP_CELLS]
        semi_tr = [x["start"] for x in semi_natural(t, "train")]
        semi_ev = semi_natural(t, "heldout")
        rng = np.random.default_rng([E4_EVAL_ROOT, t, 1])
        ev_idx = sorted(rng.choice(len(semi_ev), size=min(N_EVAL_NAT, len(semi_ev)), replace=False))
        rep = max(1, len(semi_tr) // len(con_gap))
        gap_states = [StartState(l, m, k % cfg.phase_period, None, f"constructed {c}")
                      for _ in range(rep) for c, (l, m, k) in zip(GAP_CELLS, con_gap)]
        gap_states += [StartState(l, m, k % cfg.phase_period, None, "semi-natural E3 train") for l, m, k in semi_tr]
        os.makedirs(os.path.dirname(archive_path("gap", t, "train")), exist_ok=True)
        save_archive(archive_path("gap", t, "train"), gap_states, cfg,
                     {"stage": "m46a e4", "target": spec(t), "constructed_repeats": rep,
                      "constructed": len(GAP_CELLS), "semi_natural": len(semi_tr)})
        save_gz(archive_path("gap", t, "eval"), {"semi_natural": [
            {"start": state_to_list(semi_ev[i]["start"]), **{k: semi_ev[i][k] for k in ("level", "gap", "orient", "g",
                                                                                      "turn")}} for i in ev_idx]})
        con_fr = [freeze_gap_state(cfg, *c) for c in FREEZE_CELLS]
        nat_tr = [x["start"] for x in fr[(t, "train")]["states"]]
        rep_f = max(1, len(nat_tr) // len(con_fr))
        fz_states = [StartState(l, m, k % cfg.phase_period, None, f"constructed {c}")
                     for _ in range(rep_f) for c, (l, m, k) in zip(FREEZE_CELLS, con_fr)]
        fz_states += [StartState(l, m, k % cfg.phase_period, None, "natural freeze (train seeds)")
                      for l, m, k in nat_tr]
        save_archive(archive_path("freeze", t, "train"), fz_states, cfg,
                     {"stage": "m46a e4", "target": spec(t), "constructed_repeats": rep_f,
                      "constructed": len(FREEZE_CELLS), "natural": len(nat_tr),
                      "games": fr[(t, "train")]["games"]})
        save_gz(archive_path("freeze", t, "eval"), {"natural": [
            {"start": state_to_list(x["start"]), "seed": x["seed"], "turn": x["turn"], "seat": x["seat"]}
            for x in fr[(t, "eval")]["states"]], "games": fr[(t, "eval")]["games"]})
        desc[str(t)] = {"gap_train": len(gap_states), "gap_semi_train": len(semi_tr), "gap_eval_semi": len(ev_idx),
                        "freeze_train": len(fz_states), "freeze_nat_train": len(nat_tr),
                        "freeze_eval_nat": len(fr[(t, "eval")]["states"])}
    if desc:
        old = load_json("e4_archives") or {}
        old.update(desc)
        save_json("e4_archives", old)
    return desc


# ---------------------------------------------------------------------------------------------- training
def train_job(args) -> tuple:
    """(kind, target, steps, out_dir or None): one attacker (resumable: an existing final.pt is kept)."""
    kind, target, steps, out_dir = args
    import torch
    torch.set_num_threads(1)
    if out_dir and os.path.exists(os.path.join(out_dir, "final.pt")):
        return (kind, target), {"skipped": "final.pt exists", "dir": out_dir}
    from ....rl import OpponentSampler, PolicyValueNet, train_ppo
    from ..m45c.common import agent_from_spec, ppo_config
    cfg = env_cfg()
    tgt = agent_from_spec(spec(target), cfg)
    net = PolicyValueNet(tgt.net.obs_size, tgt.net.n_actions, tgt.net.hidden)
    net.load_state_dict(tgt.net.state_dict())
    seed = E4_TRAIN_ROOT + (0 if kind == "gap" else 10) + target
    ppo = ppo_config(seed, steps, lane_cache=TRAIN_CACHE, start_archive=archive_path(kind, target, "train"),
                     start_mix=1.0, game_length=GAME_LENGTH)
    t0 = time.perf_counter()
    res = train_ppo(cfg, ppo, OpponentSampler([tgt]), out_dir=out_dir, net=net, verbose=False,
                    name=f"m46a_{kind}_s{target}", collector_cls=focal_collector_cls())
    hist = res.history
    curve = [[h["steps"], h["mean_return"]] for h in hist if np.isfinite(h.get("mean_return", np.nan))]
    return (kind, target), {"steps": steps, "train_s": time.perf_counter() - t0, "seed": seed, "ppo": ppo.to_dict(),
                            "updates": len(hist), "curve": curve[::max(1, len(curve) // 80)],
                            "entropy_first_last": [hist[0]["entropy"], hist[-1]["entropy"]] if hist else None,
                            "kl_max": float(max(h["approx_kl"] for h in hist)) if hist else None,
                            "grad_norm_max": float(max(h.get("grad_norm", 0) for h in hist)) if hist else None,
                            "finite": bool(all(np.isfinite(h.get("value_loss", 0)) and np.isfinite(h.get("policy_loss", 0))
                                               for h in hist))}


def train(workers: int, targets=E4_TARGETS, kinds=KINDS, steps: int = BUDGET) -> dict:
    t0 = time.perf_counter()
    js = [(k, t, steps, attacker_dir(k, t)) for k in kinds for t in targets]
    res = run_jobs(train_job, js, workers, "E4 training")
    old = load_json("e4_training") or {"runs": {}}
    for (k, t), v in res:
        if "skipped" not in v:
            old["runs"][f"{k}_s{t}"] = v
    old["meta"] = meta(stage="e4_train", budget=steps, game_length=GAME_LENGTH)
    old["elapsed_s_last"] = time.perf_counter() - t0
    save_json("e4_training", old)
    return old


# ---------------------------------------------------------------------------------------------- evaluation
def eval_jobs(targets=E4_TARGETS, kinds=KINDS) -> list:
    from .common import load_gz, state_from_list
    from .states import freeze_gap_state, zoning_gap_state
    cfg = env_cfg()
    js = []
    for t in targets:
        for kind in kinds:
            att = os.path.join(attacker_dir(kind, t), "final.pt")
            cands = {"target": ("net", spec(t)), "attacker": ("net", att), "s0att": ("net", s0_attacker_path(t))}
            H = H_EVAL[kind]
            sets = []
            if kind == "gap":
                sets.append(("con", [zoning_gap_state(cfg, *c) for c in GAP_CELLS], K_CON))
                sets.append(("equal", [zoning_gap_state(cfg, *c) for c in EQUAL_CELLS], K_CON))
                ev = load_gz(archive_path("gap", t, "eval"))["semi_natural"]
                sets.append(("semi", [state_from_list(x["start"]) for x in ev], K_NAT))
            else:
                sets.append(("con", [freeze_gap_state(cfg, *c) for c in FREEZE_CELLS], K_CON))
                ev = load_gz(archive_path("freeze", t, "eval"))["natural"]
                sets.append(("nat", [state_from_list(x["start"]) for x in ev], K_NAT))
            for name, sts, K in sets:
                fam = f"E4{kind}_{name}"
                for i, st in enumerate(sts):
                    js.append({"spec": spec(t), "family": fam, "idx": i, "start": st, "proto": "policy",
                               "horizon": H, "roll": EN.roll_seeds(E4_EVAL_ROOT, K, i), "cands": cands,
                               "critic": False, "tag": f"s{t}",
                               "path": os.path.join(CKPT, "e4", "eval", fam, f"s{t}", f"{i}.json.gz"),
                               "label": f"{fam} {i}"})
    return js


def s0_eval_job(args) -> tuple:
    kind, target, path = args
    setup_worker()
    from ....env.discounted import evaluate_discounted
    from ..m45c.common import agent_from_spec
    cfg = env_cfg()
    e = evaluate_discounted(cfg, agent_from_spec(path, cfg), agent_from_spec(spec(target), cfg), episodes=64,
                            seed=E4_EVAL_ROOT + 50, window=460)
    return (kind, target), {k: e[k] for k in ("estimate", "stderr", "ci95", "seat_A", "seat_B", "window")}


def evaluate(workers: int, targets=E4_TARGETS, kinds=KINDS) -> dict:
    t0 = time.perf_counter()
    res = FA.run(eval_jobs(targets, kinds), workers, "E4 evaluation")
    s0j = [(k, t, os.path.join(attacker_dir(k, t), "final.pt")) for k in kinds for t in targets]
    s0j += [("s0att", t, s0_attacker_path(t)) for t in targets]
    s0 = {f"{k}_s{t}": v for (k, t), v in run_jobs(s0_eval_job, s0j, workers, "E4 s0 evaluation")}
    out = analyse(res, targets)
    d = load_json("e4_summary") or {}                     # merge: earlier kinds and exploratory results are kept
    an = d.get("analysis", {})
    an.update(out)
    an["s0_eval"] = {**an.get("s0_eval", {}), **s0}
    d["analysis"] = an
    d.setdefault("eval_meta", {})["+".join(kinds)] = meta(stage="e4_eval", targets=list(targets), K_con=K_CON,
                                                          K_nat=K_NAT, horizons=H_EVAL, primary=PRIMARY_H,
                                                          elapsed_s=time.perf_counter() - t0)
    save_json("e4_summary", d)
    return out


def analyse(res: dict, targets=E4_TARGETS) -> dict:
    out = {}
    fams = sorted({r["family"] for r in res.values()})
    for fam in fams:
        kind = "gap" if fam.startswith("E4gap") else "freeze"
        F = {}
        for h in (20, 50, 100, H_EVAL[kind]):
            for cand, base in (("attacker", "target"), ("s0att", "target"), ("attacker", "s0att")):
                per = {}
                for t in targets:
                    recs = [r for r in res.values() if r["family"] == fam and r["tag"] == f"s{t}"]
                    v = [EN.per_state_diff(r, cand, "G", h, base) for r in recs]
                    v = [x for x in v if np.isfinite(x)]
                    if v:
                        m, hw = mean_ci(v)
                        per[t] = {"mean": m, "ci": hw, "n": len(v), "sig": significant(m, hw)}
                m, hw = mean_ci([x["mean"] for x in per.values()])
                F[f"{cand}-{base}|G{h}"] = {"mean": m, "ci": hw, "per_target": per,
                                             "k_pos": sum(x["sig"] > 0 for x in per.values()),
                                             "k_neg": sum(x["sig"] < 0 for x in per.values())}
            absg = {}
            for t in targets:
                recs = [r for r in res.values() if r["family"] == fam and r["tag"] == f"s{t}"]
                absg[t] = {c: float(np.nanmean([EN.per_state_mean(r, c, "G", h) for r in recs]))
                           for c in ("target", "attacker", "s0att")}
            F[f"abs|G{h}"] = absg
        ch = {}
        hh = PRIMARY_H[kind] if PRIMARY_H[kind] in EN.ledger_checkpoints(H_EVAL[kind]) else 50
        for cand in ("attacker", "s0att"):
            for key in ("earned_A", "farm_A", "earned_B", "unc_B", "unc_away_B", "away_B", "deaths_B", "recalls_B",
                        "xp_lost_B", "deaths_A", "d_total_A", "trade_att_A", "trade_conn_A", "held_A", "crash_hp_B",
                        "plates_A", "n_safe_B"):
                per = {}
                for t in targets:
                    recs = [r for r in res.values() if r["family"] == fam and r["tag"] == f"s{t}"]
                    v = [EN.ledger_diff(r, cand, key, hh) for r in recs]
                    v = [x for x in v if np.isfinite(x)]
                    if v:
                        per[t] = float(np.mean(v))
                m, hw = mean_ci(list(per.values()))
                ch[f"{cand}:{key}@{hh}"] = {"mean": m, "ci": hw, "per_target": per}
            # the target's own absolute channel levels (baseline)
        for key in ("earned_A", "earned_B", "away_B", "deaths_B", "unc_B", "held_A", "trade_conn_A"):
            per = {}
            for t in targets:
                recs = [r for r in res.values() if r["family"] == fam and r["tag"] == f"s{t}"]
                v = [EN.ledger_mean(r, "target", key, hh) for r in recs]
                if v:
                    per[t] = float(np.nanmean(v))
            ch[f"target_abs:{key}@{hh}"] = {"mean": float(np.mean(list(per.values()))) if per else float("nan"),
                                            "per_target": per}
        F["channels"] = ch
        out[fam] = F
    return out


# ---------------------------------------------------------------------------------------------- smoke
def smoke(target: int = 0) -> dict:
    """Pipeline check only (4 updates of 2,048 learner transitions, no checkpoint): pass / fail, time and memory."""
    import psutil
    t0 = time.perf_counter()
    for kind in KINDS:
        if not os.path.exists(archive_path(kind, target, "train")):
            raise SystemExit("build the archives first (stage e4_archives)")
    _, r = train_job(("gap", target, 2 * 4096, None))
    proc = psutil.Process()
    return {"ok": r["updates"] == r["steps"] // 2048 and r["finite"], "updates": r["updates"], "seconds": time.perf_counter() - t0,
            "rss_gb": proc.memory_info().rss / 1e9, "transitions_per_s": r["steps"] / max(r["train_s"], 1e-9)}


__all__ = ["focal_collector_cls", "build_archives", "train", "train_job", "evaluate", "eval_jobs", "analyse", "smoke",
           "attacker_dir", "archive_path", "BUDGET", "GAME_LENGTH"]


POOLS = {"E4gap_pooled": ("E4gap_con", "E4gap_semi"), "E4freeze_pooled": ("E4freeze_con", "E4freeze_nat")}


def reanalyse(targets=E4_TARGETS) -> dict:
    """The analysis from the saved evaluation records, with the amendment's pooled families (constructed + held-out
    states of a target, equal weight per state); merged into results/m46a/e4_summary.json."""
    from .common import load_gz
    res = {}
    root = os.path.join(CKPT, "e4", "eval")
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.endswith(".json.gz"):
                r = load_gz(os.path.join(dp, f))
                res[os.path.join(dp, f)] = r
    for pool, members in POOLS.items():
        for k, r in list(res.items()):
            if r["family"] in members:
                q = dict(r)
                q["family"] = pool
                res[k + "|" + pool] = q
    out = analyse(res, targets)
    d = load_json("e4_summary") or {"analysis": {}}
    s0 = d.get("analysis", {}).get("s0_eval")
    d["analysis"] = {**out, **({"s0_eval": s0} if s0 else {})}
    d["reanalysed"] = meta(stage="e4_reanalyse")
    save_json("e4_summary", d)
    return d["analysis"]


def transfer(workers: int, targets=E4_TARGETS, kind: str = "gap", K: int = 8) -> dict:
    """EXPLORATORY (after the E4-P1 result, to separate better strong-side play from target-specific exploitation):
    each attacker in seat A against every OTHER final in seat B, from the constructed cells, paired against that
    final's own play in seat A (the same states, chance and sampling streams). Horizon 100; G100."""
    from .common import FINALS
    from .states import freeze_gap_state, zoning_gap_state
    cfg = env_cfg()
    cells = GAP_CELLS if kind == "gap" else FREEZE_CELLS
    sts = [zoning_gap_state(cfg, *c) for c in cells] if kind == "gap" else [freeze_gap_state(cfg, *c) for c in cells]
    js = []
    for t in targets:
        att = os.path.join(attacker_dir(kind, t), "final.pt")
        for o in FINALS:
            if o == t:
                continue
            for i, st in enumerate(sts):
                js.append({"spec": spec(o), "family": f"T{kind}", "idx": i, "start": st, "proto": "policy",
                           "horizon": 100, "roll": EN.roll_seeds(E4_EVAL_ROOT + 70, K, i),
                           "cands": {"own": ("net", spec(o)), "attacker": ("net", att)}, "critic": False,
                           "tag": f"a{t}_o{o}",
                           "path": os.path.join(CKPT, "e4", "transfer", kind, f"a{t}_o{o}", f"{i}.json.gz")})
    res = FA.run(js, workers, f"E4 transfer {kind}")
    out = {}
    for t in targets:
        per = {}
        for o in FINALS:
            if o == t:
                continue
            v = [EN.per_state_diff(r, "attacker", "G", 100, "own") for r in res.values() if r["tag"] == f"a{t}_o{o}"]
            m, hw = mean_ci(v)
            per[o] = {"mean": m, "ci": hw, "sig": significant(m, hw)}
        m, hw = mean_ci([x["mean"] for x in per.values()])
        out[str(t)] = {"mean": m, "ci": hw, "per_opponent": per,
                       "k_pos": sum(x["sig"] > 0 for x in per.values()), "k_neg": sum(x["sig"] < 0 for x in per.values())}
    for key in ("earned_A", "earned_B", "deaths_B", "away_B", "trade_conn_A", "deaths_A"):
        vals = [EN.ledger_diff(r, "attacker", key, 100) for r in res.values()]
        out[f"channel:{key}@100"] = float(np.nanmean(vals))
    d = load_json("e4_summary") or {}
    d.setdefault("exploratory", {})[f"transfer_{kind}"] = {"meta": meta(stage="e4_transfer", kind=kind, K=K),
                                                         "analysis": out}
    save_json("e4_summary", d)
    return out
