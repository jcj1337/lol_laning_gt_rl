"""Q1: the measurement path from game state to observation, critic input, stored value and reported return.

    pairs        controlled single-feature pairs: from three base states (freeze L5, neutral lane L8 and L15, equal
                 strength, full HP) one feature of seat A is changed at a time; checked: exactly the intended
                 observation entries change, in seat A's OWN block and seat B's ENEMY block, by the intended amount;
                 recorded: the critic's change for both seats (resources) and the actor's total-variation change
    reachable    on reachable states (self-play from s0 and from archive starts, the archive sample's variants and
                 the M4.6d cells): mirror semantics observe(s, side 1) == observe(mirror(s), side 0); exact decoding
                 (decode_m4) back to the full Markov state; injectivity (distinct states -> distinct observations);
                 entries in [0, 1]; saturation of the investment entries; the plates-pending lane fields (not in the
                 observation) are always 0 under the M4 rules; the critic's zero-sum consistency V_A + V_B
    ranges       the natural spread of the strength entries against the change one "edge" or "x1.76" makes
    compat       every evaluated checkpoint: digest-verified load, 90 inputs / 61 actions, its run.json game
                 configuration equal to the C1 arm's, reward scale 0.1, gamma 0.99
    grad         the shared trunk: gradient norms of the value term (vf_coef x value loss) and the policy term
                 (clipped surrogate - ent_coef x entropy) on the trunk, on training-like batches (the M4.6c mixed-start
                 collector and the run's own PPO configuration, at the final weights; no update is applied)
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List

import numpy as np

from . import common as C
from .features import StateFeatures, explicit, obs_index

TOL = 1e-6


# ---------------------------------------------------------------------------------------------- helpers
def _obs(cfg, start, s: int) -> np.ndarray:
    from ....diagnostics.branch import Snapshot, observation
    lane, m4, k = start
    return observation(cfg, Snapshot(lane, m4, k, (cfg.skill_a, cfg.skill_b)), s)


def _net_eval(net, vecs):
    import torch
    with torch.no_grad():
        lg, v = net(torch.as_tensor(np.stack(vecs), dtype=torch.float32))
    return lg.double().numpy(), v.double().numpy() / C.REWARD_SCALE


def _probs(logits, legal):
    z = logits[list(legal)]
    p = np.exp(z - z.max())
    return dict(zip(legal, p / p.sum()))


def _tv(p, q) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def base_states(cfg) -> Dict[str, tuple]:
    from ..m46d import cells as CL
    return {"freeze L5": CL.build(cfg, "freeze", "equal", 5, 1.0), "neutral L8": CL.build(cfg, "neutral", "equal", 8, 1.0),
            "neutral L15": CL.build(cfg, "neutral", "equal", 15, 1.0)}


def perturbations(cfg) -> Dict[str, tuple]:
    """name -> (function(start) -> start, required and optional seat-A OWN entries (names without 'own_'))."""
    from ....champions_m4 import context
    t = context(cfg).t
    sc = cfg.gold_scale
    T = cfg.tower

    def lev(m4):
        return int(t.level_by_xp[min(m4.a.xp, t.xp_cap)])

    def a_(st, **kw):
        lane, m4, k = st
        return lane, m4._replace(a=m4.a._replace(**kw)), k

    def inv(d):
        return lambda st: a_(st, items=st[1].a.items + int(round(d * sc)))

    def gold(d):
        return lambda st: a_(st, gold=st[1].a.gold + int(round(d * sc)))

    def level_up(st):
        m4 = st[1]
        L = lev(m4)
        return a_(st, xp=int(t.xp_cum[L + 1]), hp=int(round(m4.a.hp / t.hp_max[L] * t.hp_max[L + 1])))

    def xp_half(st):
        L = lev(st[1])
        return a_(st, xp=int(t.xp_cum[L] + t.xp_next[L] // 2))

    def hp(f):
        return lambda st: a_(st, hp=max(1, int(round(f * t.hp_max[lev(st[1])]))))

    def hp_b(st):
        lane, m4, k = st
        Lb = int(t.level_by_xp[min(m4.b.xp, t.xp_cap)])
        return lane, m4._replace(b=m4.b._replace(hp=int(round(0.6 * t.hp_max[Lb])))), k

    def away(st):
        return a_(st, absent=2, pos=-T)

    def step_fwd(st):
        return a_(st, pos=min(st[1].a.pos + 1, T - 1))

    def wave(st):
        lane, m4, k = st
        return lane._replace(ah=max(1, lane.ah // 2)), m4, k

    def vision(st):
        lane, m4, k = st
        return lane._replace(va=3), m4, k

    def tower(st):
        return a_(st, tower_dmg=st[1].a.tower_dmg + cfg.plate_thresholds[0])

    def phase(st):
        return st[0], st[1], st[2] + 1

    c = "champ_"
    return {
        "invested +1": (inv(1), {c + "invested_whole", c + "attack_power"}, set()),
        "invested +3": (inv(3), {c + "invested_whole", c + "attack_power"}, set()),
        "invested +8": (inv(8), {c + "invested_whole", c + "attack_power"}, set()),
        "invested +32": (inv(32), {c + "invested_whole", c + "attack_power"}, set()),
        "invested +0.5": (inv(0.5), {c + "invested_frac", c + "attack_power"}, {c + "invested_whole"}),
        "unspent gold +1": (gold(1), {c + "gold_whole"}, set()),
        "unspent gold +4": (gold(4), {c + "gold_whole"}, set()),
        "level +1": (level_up, {c + "level", c + "hp_max", c + "attack_power"}, {c + "hp_frac"}),
        "XP within level": (xp_half, {c + "xp_progress"}, set()),
        "HP 60%": (hp(0.6), {c + "hp_frac"}, set()),
        "HP 30%": (hp(0.3), {c + "hp_frac"}, set()),
        "enemy HP 60%": (hp_b, {"ENEMY " + c + "hp_frac"}, set()),
        "away 2 turns": (away, {c + "absent"}, {"CELL"}),
        "one cell forward": (step_fwd, {"CELL"}, set()),
        "own front HP halved": (wave, {"front_hp"}, set()),
        "vision 3 turns": (vision, {"vision"}, set()),
        "first plate dealt": (tower, {"tower_dmg", "plates_broken", "plates_left"}, {"plate_progress"}),
        "phase + 1": (phase, {"PHASE"}, set()),
    }


def _changed(v0, v1, names) -> Dict[str, float]:
    d = v1 - v0
    return {names[i]: float(d[i]) for i in np.flatnonzero(np.abs(d) > TOL)}


def _expected_ok(changed: Dict[str, float], req: set, opt: set, perspective: str) -> bool:
    """Seat A ('own' block) or seat B ('enemy' block): every required entry changed, nothing outside required |
    optional changed. Tokens: entry names without the block prefix; 'ENEMY x' = the other block; CELL = the
    champion's cell one-hot; PHASE = the phase one-hot."""
    own, other = ("own_", "enemy_") if perspective == "A" else ("enemy_", "own_")
    every = req | opt

    def name_of(tok):
        return other + tok[6:] if tok.startswith("ENEMY ") else own + tok

    allowed = {name_of(t) for t in every if t not in ("CELL", "PHASE")}

    def is_cell(n):
        return n == own + "champ_W" or (n.startswith(own + "champ_x") and n[len(own) + 7:len(own) + 8] in "+-")

    for n in changed:
        if n.startswith("phase_"):
            if "PHASE" not in every:
                return False
        elif is_cell(n):
            if "CELL" not in every:
                return False
        elif n not in allowed:
            return False
    for t in req:
        if t == "PHASE":
            ok = any(n.startswith("phase_") for n in changed)
        elif t == "CELL":
            ok = any(is_cell(n) for n in changed)
        else:
            ok = name_of(t) in changed
        if not ok:
            return False
    return True


def pairs(spec: str) -> List[dict]:
    from ....env.observation_m4 import m4_feature_names
    cfg = C.setup(C.ARM)
    net = C.net_of(spec)
    names = m4_feature_names(cfg)
    out = []
    for bname, st in base_states(cfg).items():
        oa0, ob0 = _obs(cfg, st, 0), _obs(cfg, st, 1)
        lg0, v0 = _net_eval(net, [oa0.vector, ob0.vector])
        pa0 = _probs(lg0[0], oa0.legal)
        for pname, (fn, req, opt) in perturbations(cfg).items():
            st1 = fn(st)
            oa1, ob1 = _obs(cfg, st1, 0), _obs(cfg, st1, 1)
            lg1, v1 = _net_eval(net, [oa1.vector, ob1.vector])
            ca = _changed(oa0.vector, oa1.vector, names)
            cb = _changed(ob0.vector, ob1.vector, names)
            out.append({"base": bname, "change": pname, "A_changed": ca, "B_changed": cb,
                        "A_ok": _expected_ok(ca, req, opt, "A"), "B_ok": _expected_ok(cb, req, opt, "B"),
                        "dV_A": float(v1[0] - v0[0]), "dV_B": float(v1[1] - v0[1]),
                        "tv_actor_A": _tv(pa0, _probs(lg1[0], oa1.legal))})
    return out


# ---------------------------------------------------------------------------------------------- reachable states
def reachable_states(n_s0: int = 4, n_arch: int = 24) -> List[tuple]:
    """Start-of-turn states of self-play of ext/C1:0 (s0 games of 400 turns, archive-start games of 150 turns), the
    archive sample's variants and the M4.6d calibration cells."""
    from ....env.starts import load_archive
    from ...m45b import states as S
    from ..m46c.archive import archive_path
    from . import labels as LB
    cfg = C.setup(C.ARM)
    net = C.net_of(C.POLICIES[0])
    out: List[tuple] = []

    def grab(i, lane, m4, obs, vals):
        out.append((lane, m4, None))

    ks = []
    for g in range(n_s0):
        n0 = len(out)
        LB.play(cfg, net, S.s0(cfg), C.OBS_ROOT * 1000 + g, 400, grab)
        ks += list(range(400))
        assert len(out) - n0 == 400
    arch = load_archive(archive_path(), cfg)
    rng = np.random.default_rng(C.OBS_ROOT)
    for g in range(n_arch):
        st = arch[int(rng.integers(len(arch)))]
        if rng.random() < 0.5:
            st = st.mirror()
        LB.play(cfg, net, (st.lane, st.champ, st.phase), C.OBS_ROOT * 1000 + 100 + g, 150, grab)
        ks += [st.phase + i for i in range(150)]
    states = [(lane, m4, k) for (lane, m4, _), k in zip(out, ks)]
    for item in LB.archive_sample():
        states += list(LB.variants(cfg, item).values())
    states += [st for _, _, st, _ in LB._m46d_cells(cfg)]
    return states


def reachable(spec: str) -> dict:
    from ....env.observation_m4 import decode_m4
    from ....env.observation import observe
    cfg = C.setup(C.ARM)
    net = C.net_of(spec)
    states = reachable_states()
    P = cfg.phase_period
    mirror_bad = decode_bad = plates_nonzero = out_of_range = 0
    beyond: List[float] = []
    seen: Dict[bytes, tuple] = {}
    collisions = 0
    vsum, vecs = [], []
    inv_max = gold_max = 0.0
    idx = obs_index(cfg)
    for lane, m4, k in states:
        va = observe(lane, k, 0, cfg, m4)
        vb = observe(lane, k, 1, cfg, m4)
        vm = observe(lane.mirror(), k, 0, cfg, m4.mirror())
        mirror_bad += int(not np.array_equal(vb, vm))
        if lane.ap or lane.bp:
            plates_nonzero += 1
        try:
            ph, l2, m2, _ = decode_m4(va, cfg)
            decode_bad += int(not (ph == k % P and l2 == lane._replace(ap=0, bp=0) and m2 == m4))
        except ValueError:                           # beyond the compressed entries' exact range (decode_m4 raises)
            beyond.append(max(m4.a.items, m4.b.items) / cfg.gold_scale)
        out_of_range += int(va.min() < 0 or va.max() > 1)
        key = va.tobytes()
        sk = (lane, m4, k % P)
        if key in seen and seen[key] != sk:
            collisions += 1
        seen.setdefault(key, sk)
        inv_max = max(inv_max, float(va[idx["own_champ_invested_whole"]]), float(va[idx["enemy_champ_invested_whole"]]))
        gold_max = max(gold_max, float(va[idx["own_champ_gold_whole"]]), float(va[idx["enemy_champ_gold_whole"]]))
        vecs += [va, vb]
    V = []
    for b in range(0, len(vecs), 4096):
        V.append(_net_eval(net, vecs[b:b + 4096])[1])
    V = np.concatenate(V).reshape(-1, 2)
    vsum = V.sum(1)
    distinct_states = len({(lane, m4, k % P) for lane, m4, k in states})
    return {"states": len(states), "distinct_states": distinct_states, "distinct_observations": len(seen),
            "mirror_mismatches": mirror_bad, "decode_mismatches": decode_bad, "observation_collisions": collisions,
            "beyond_exact_decode": len(beyond), "beyond_exact_decode_invested": beyond,
            "plates_pending_nonzero": plates_nonzero, "entries_out_of_range": out_of_range,
            "invested_entry_max": inv_max, "gold_entry_max": gold_max,
            "V_sum_mean": float(vsum.mean()), "V_sum_absmean": float(np.abs(vsum).mean()),
            "V_sum_q": [float(x) for x in np.quantile(vsum, (0.05, 0.5, 0.95))],
            "V_A_sd": float(V[:, 0].std())}


# ---------------------------------------------------------------------------------------------- entry ranges
def ranges() -> dict:
    """Natural spread (s0 self-play of ext/C1:0..2 from the labelled natural set when present, else the reachable
    sample) of the strength entries, against the change of the M4.6d edge / x1.76 contrasts."""
    from ..m46d import cells as CL
    from ....env.observation_m4 import m4_feature_names
    cfg = C.setup(C.ARM)
    names = m4_feature_names(cfg)
    idx = {n: i for i, n in enumerate(names)}
    from . import labels as LB
    vecs = []
    for spec in C.POLICIES[:3]:
        for b in range(0, C.NAT_GAMES, 8):
            p = os.path.join(C.CKPT, "labels", LB._tag(spec), f"natural_{b}.npz")
            if os.path.exists(p):
                vecs.append(np.load(p)["obs"][:, 0])
    X = np.concatenate(vecs) if vecs else None
    keys = ("own_champ_level", "own_champ_hp_frac", "own_champ_hp_max", "own_champ_invested_whole",
            "own_champ_gold_whole", "own_champ_attack_power")
    out = {"natural_states": 0 if X is None else int(len(X)), "entries": {}}
    contrasts = {}
    for L in (3, 5, 8):
        for cond in ("edge", "x1.76"):
            e = _obs(cfg, CL.build(cfg, "freeze", "equal", L, 1.0), 0).vector
            v = _obs(cfg, CL.build(cfg, "freeze", cond, L, 1.0), 0).vector
            contrasts[f"L{L} {cond}"] = {k: float(v[idx[k]] - e[idx[k]]) for k in keys}
            xa = explicit(np.stack([e, v]), obs_index(cfg))
            contrasts[f"L{L} {cond}"]["log_duel (explicit)"] = float(xa[1, 0] - xa[0, 0])
    for k in keys:
        rec = {}
        if X is not None:
            x = X[:, idx[k]]
            rec = {"q05": float(np.quantile(x, 0.05)), "q50": float(np.median(x)), "q95": float(np.quantile(x, 0.95)),
                   "sd": float(x.std())}
            if X is not None:
                d = np.abs(np.diff(X[:, idx[k]]))
                rec["typical_turn_change"] = float(np.median(d[d > 0])) if (d > 0).any() else 0.0
        out["entries"][k] = rec
    if X is not None:
        ex = explicit(X, obs_index(cfg))
        sf_duel = []
        out["natural_log_duel_q"] = [float(x) for x in np.quantile(ex[:, 0], (0.05, 0.25, 0.5, 0.75, 0.95))]
    out["contrasts"] = contrasts
    # the explicit log duel equals the state's duel ratio (checked on the M4.6d cells)
    sf = StateFeatures(cfg)
    errs = []
    for _, _, st, _ in __import__(f"{__package__}.labels", fromlist=["_m46d_cells"])._m46d_cells(cfg):
        lane, m4, k = st
        x = explicit(_obs(cfg, st, 0).vector[None], obs_index(cfg))[0, 0]
        errs.append(abs(float(x) - float(np.log(sf.duel(m4, 0)))))
    out["explicit_log_duel_max_abs_error"] = float(max(errs))
    return out


# ---------------------------------------------------------------------------------------------- compatibility
def compat() -> List[dict]:
    from ..m46c import common as C46
    cfg = C.arm_cfg(C.ARM).to_dict()
    out = []
    for arm in ("C1", "baseline"):
        for s in C.SEEDS:
            spec = f"ext/{arm}:{s}"
            path = C46.policy_path(spec)
            run = json.load(open(os.path.join(os.path.dirname(path), "run.json"), encoding="utf-8"))
            net = C.net_of(spec)
            want = C.arm_cfg(arm).to_dict()
            out.append({"spec": spec, "inputs": net.obs_size, "actions": net.n_actions,
                        "hidden": list(net.hidden), "game_config_matches_arm": run["game_config"] == want,
                        "reward_scale": run["ppo"]["reward_scale"], "gamma": run["ppo"]["gamma"],
                        "vf_coef": run["ppo"]["vf_coef"], "vf_clip": run["ppo"]["vf_clip"],
                        "gae_lambda": run["ppo"]["gae_lambda"], "ent_coef": run["ppo"]["ent_coef"],
                        "current_fraction": run["shared"]["current_fraction"],
                        "C1_config_identical_obs_layout": len(cfg) > 0})
    return out


# ---------------------------------------------------------------------------------------------- gradient share
def grad_job(spec: str) -> tuple:
    """Value-term and policy-term gradient norms on the shared trunk, over the minibatches of training-like batches."""
    import torch
    from ....rl.ppo import PPOConfig, compute_gae
    from ....rl.network import MaskedCategorical
    from ....rl.selfplay import SnapshotPool
    from ....rl.shared import MatchupSampler, rollout_batch
    from ..m46c import common as C46
    from ..m46c.archive import archive_path
    from ..m46c.collector import mixed_collector_cls
    torch.set_num_threads(1)
    arm = spec.split("/")[1].split(":")[0]
    cfg = C.arm_cfg(arm)
    run = json.load(open(os.path.join(os.path.dirname(C46.policy_path(spec)), "run.json"), encoding="utf-8"))
    pc = PPOConfig.from_dict(run["ppo"])
    net = C.net_of(spec)
    coll_cls = mixed_collector_cls(archive_path(), C46.ARCHIVE_BETA, C46.ARCHIVE_LENGTH, C46.MIRROR_P)
    coll = coll_cls(cfg, pc, MatchupSampler(SnapshotPool(8, keep_first=False), 1.0), C.GRAD_ROOT)
    gen = torch.Generator().manual_seed(C.GRAD_ROOT)
    trunk = [p for p in net.trunk.parameters()]
    rows = []
    for rnd in range(4):
        roll, _, _ = coll.collect(net, gen)
        if rnd < 2:                                  # warm-up: the first games are staggered starts
            continue
        batch, adv, ret = rollout_batch(roll, pc)
        n = batch["obs"].shape[0]
        perm = torch.randperm(n, generator=gen)
        for b in range(0, n, pc.minibatch_size):
            idx = perm[b:b + pc.minibatch_size]
            logits, v = net(batch["obs"][idx])
            dist = MaskedCategorical(logits, batch["mask"][idx])
            w = batch["choice"][idx]
            nw = w.sum().clamp(min=1.0)
            a = batch["advantages"][idx]
            m = (a * w).sum() / nw
            sd = (((a - m) ** 2 * w).sum() / nw).sqrt()
            a = (a - m) / (sd + 1e-8)
            ratio = (dist.log_prob(batch["actions"][idx]) - batch["logprobs"][idx]).exp()
            pg = (torch.max(-a * ratio, -a * ratio.clamp(1 - pc.clip_coef, 1 + pc.clip_coef)) * w).sum() / nw
            ent = (dist.entropy() * w).sum() / nw
            vl = 0.5 * ((v - batch["returns"][idx]) ** 2).mean()
            gp = torch.autograd.grad(pg - pc.ent_coef * ent, trunk, retain_graph=True)
            gv = torch.autograd.grad(pc.vf_coef * vl, trunk, retain_graph=True)
            gvh = torch.autograd.grad(pc.vf_coef * vl, list(net.value_head.parameters()))
            fp = torch.cat([g.reshape(-1) for g in gp])
            fv = torch.cat([g.reshape(-1) for g in gv])
            rows.append((float(fp.norm()), float(fv.norm()), float(torch.dot(fp, fv) / (fp.norm() * fv.norm() + 1e-12)),
                         float(torch.cat([g.reshape(-1) for g in gvh]).norm()), float(vl)))
    r = np.asarray(rows)
    return spec, {"minibatches": int(len(r)), "policy_trunk_norm": float(r[:, 0].mean()),
                  "value_trunk_norm": float(r[:, 1].mean()), "ratio_value_to_policy": float((r[:, 1] / r[:, 0]).mean()),
                  "cosine": float(r[:, 2].mean()), "value_head_norm": float(r[:, 3].mean()),
                  "value_loss_scaled": float(r[:, 4].mean()),
                  "value_loss_resources2": float(r[:, 4].mean() / C.REWARD_SCALE ** 2)}


# ---------------------------------------------------------------------------------------------- driver
def run(workers: int) -> None:
    t0 = time.perf_counter()
    res = {"meta": C.meta(stage="observation")}
    res["pairs"] = {spec: rows for spec, rows in zip(C.POLICIES, C.run_jobs(pairs, list(C.POLICIES), workers, "pairs"))}
    res["reachable"] = reachable(C.POLICIES[0])
    res["ranges"] = ranges()
    res["compat"] = compat()
    specs = list(C.POLICIES) + [f"ext/baseline:{s}" for s in C.SEEDS[:3]]
    res["grad"] = dict(C.run_jobs(grad_job, specs, workers, "gradient share"))
    res["elapsed_s"] = time.perf_counter() - t0
    C.save_json("observation", res)
    print(f"  observation checks done in {res['elapsed_s']:.0f} s", flush=True)


__all__ = ["pairs", "reachable", "ranges", "compat", "grad_job", "run", "perturbations", "base_states"]
