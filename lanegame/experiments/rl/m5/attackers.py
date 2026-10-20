"""M5d's dedicated attackers (design §8.4): learned best responses to a frozen M5 final, reported as exploitability
LOWER bounds (limited by the attacker's budget, network and optimizer; never a certificate).

Proposal (every item is in the pre-registration's attacker section and is a review decision):
    target       a frozen final: om (its belief live, updated from the attacker's actions exactly as in deployment), sp,
                 or for the adaptation test om_table (the om final with its belief inputs replaced by the mean belief of
                 its own self-play at that turn: the same network without in-game adaptation). Access: the attacker is
                 INITIALIZED from the target's weights (see start), so the procedure is not black-box as a whole; during
                 play it sees only the game state, the target's previous action and its own belief, never the target's
                 weights, belief, logits or gradients.
    observation  the om input format for every attacker: the obs_last_action observation (103: the state and the
                 target's previous action) plus the attacker's own belief over the OM2 library (13), the same
                 information channel the om arm has, for attackers of either arm
    start        a copy of the target in that format (parity, as M4.5c: an om target's own weights; an sp target widened
                 with zero weights on the 26 new inputs), so a gain measures the target's exploitability, not the
                 attacker's catching up; om_table attackers start as the om final
    objective    the training objective unchanged: the zero-sum gold-difference reward, gamma 0.99, continuing time, the
                 learner in a uniformly drawn seat of every game, only the learner trains. No reward for deception or
                 for moving the target's belief; whether exploiting the adaptation emerges is measured, not rewarded
    games        s0 starts, 512-turn bootstrapped games (the M4.5c attacker protocol), the M4.6c PPO settings, a fresh
                 Adam at 3e-4 annealed to 0 over the attacker's budget, PPO seed ATTACK_TRAIN_ROOT + 10 * kind + s
    budget       ATTACK_BUDGET agent transitions. The denominator is the target's M5 continuation budget, 32,768,000:
                 the 24,576,000 prior transitions are shared by both arms' targets and inherited by the attacker
                 through its start. Recommended (M5a, for cost): QUARTER = 8,192,000 (M4.5c's ratio); the
                 alternative is HALF = 16,384,000
    evaluation   independent games from s0 (ATTACK_EVAL_ROOT), N_EVAL paired-seat pairs: the lead at turn 114 (lane exit)
                 and the discounted lead over 500 turns (the objective); transfer: the attacker against the other arm's
                 final of its seed and against the common start; the target's belief trace against the attacker
                 (`games.cell_job` stores it as f0__opp_trace in the om-target cells)

Runs (amendment 1's recovery rules of the main training, pre-registration §3, applied to the attackers): every run is
watched by `health.RunMonitor` (a DEFECT alarm ends it; an INVESTIGATE alarm is recorded and the run continues), the
coordinator `health.run_campaign` logs progress (results/m5/progress_attack_main.json) and stops after more than two
failed seeds, an interrupted run is preserved under attack/<phase>/partial/ and restarted from its start
(`restart_incomplete`; no mid-run resume exists) and must reproduce the preserved history prefix. `check` is the
post-hoc health and identity check (results/m5/attack_check_<phase>.json); `require_ok` gates the evaluation on it.

The collector keeps one belief per learner seat (the attacker's) and one target instance per game (the target's belief
lives in it); both beliefs are updated after every step from the other seat's code on its start-of-turn observation,
batched across environments before any decision of the next step.

Scope (amendment 1, 2026-10-17). The CORE attacker stage is 12 attackers: CORE_KINDS = (om, sp) x six seeds at the
quarter budget, still awaiting its own approval (--confirm-m5-attackers). The six om_table attackers (OPTIONAL_KINDS)
are an optional follow-up that needs a separate decision (--include-om-table on top of that flag). Their reading is
limited: om - om_table compares two approximate lower bounds, so a difference can come from what each attacker learned,
not only from the targets; removing the live belief can change the target's competence and creates state-belief
combinations the network never saw in training; neither isolates a causal adaptation vulnerability, and neither is
evidence of deliberate deception (the attackers are stateless). The live/table cross-evaluation (each attacker against
the other target form) is implemented in `eval_jobs` when both kinds are included.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

import numpy as np

from . import common as C
from . import players as PL

AOUT = os.path.join(C.CKPT, "attack")
HALF_BUDGET = 16_384_000        # half of the target's M5 continuation budget (32,768,000)
QUARTER_BUDGET = 8_192_000      # a quarter of it (M4.5c's ratio)
ATTACK_BUDGET = QUARTER_BUDGET  # the M5a recommendation (design §7.1, §11): quarter, for cost; half is the alternative
TARGET_KINDS = ("om", "sp", "om_table")    # every kind (index = the PPO seed offset; unchanged)
CORE_KINDS = ("om", "sp")                   # the core attacker stage (12 attackers)
OPTIONAL_KINDS = ("om_table",)              # the optional adaptation ablation (6 attackers), a separate decision
N_EVAL = 128
SMOKE_UPDATES = 8


# ---------------------------------------------------------------------------------------------- the frozen target
class TargetAgent:
    """A frozen M5 final as a training opponent; one instance per game (its belief, if any, starts at the prior)."""

    def __init__(self, net, mode="live", name: str = "target"):
        from ....opponent import Belief
        self.net, self.mode, self.name = net, mode, name
        self.has_belief = net.obs_size == C.NET_IN_OM
        self.belief = Belief(C.K, C.LAMBDA) if self.has_belief else None

    def input(self, obs) -> np.ndarray:
        v = np.asarray(obs.vector, dtype=np.float32)
        if self.net.obs_size == C.OBS_SP:
            return v[:C.OBS_SP]
        if not self.has_belief:
            return v
        return np.concatenate([v, PL._features_of(self.mode, self.belief, obs.k)])

    def act(self, obs, rng=None) -> int:
        import torch
        from ....env.agents import sample
        if len(obs.legal) == 1:
            return int(obs.legal[0])
        with torch.no_grad():
            logits, _ = self.net(torch.as_tensor(self.input(obs)).unsqueeze(0))
        lg = logits[0].double().numpy()[list(obs.legal)]
        p = np.exp(lg - lg.max())
        p /= p.sum()
        return int(obs.legal[sample(rng, p)]) if rng is not None else int(obs.legal[int(np.argmax(p))])


def target_spec(kind: str, seed: int, phase: str = "main") -> str:
    return f"{'' if phase == 'main' else phase + '/'}{'om' if kind == 'om_table' else kind}:{seed}"


def target_net(kind: str, seed: int, phase: str = "main", spec: Optional[str] = None):
    return PL.net_of(spec or target_spec(kind, seed, phase))


def table_path(seed: int, phase: str = "main") -> str:
    return os.path.join(AOUT, "tables", phase, f"om_s{seed}.npy")


def target_mode(kind: str, seed: int, phase: str = "main"):
    if kind != "om_table":
        return "live"
    p = table_path(seed, phase)
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p}: build the om final's self-play belief table first (stage attack_tables)")
    return PL.resolve_mode(f"table:{p}")


def self_play_table(spec: str, games: int = 32, T: int = 512, root: int = C.ATTACK_EVAL_ROOT + 50) -> np.ndarray:
    """[T, K]: the mean belief features an om policy holds at each turn of its own self-play from s0 (both seats)."""
    from . import games as G
    acc = np.zeros((T, C.K))
    n = 0
    for g in range(games):
        sd = G.game_seed(root, 0, g)
        rec = _belief_series(spec, sd, T)
        acc += rec
        n += 2
    return (acc / n).astype(np.float32)


def _belief_series(spec: str, sd: int, T: int) -> np.ndarray:
    from ....champions import N_CHANCE
    from ....champions_m4 import context, resolve_turn_m4
    from ...m45b import engine as E
    from ...m45b import states as S
    cfg = C.arm_cfg("sp")
    ls = E.lane_cache(cfg)
    ctx = context(cfg)
    lane, m4, k0 = S.s0(cfg)
    pols = [PL.crn_policy(spec, k0, sd, s, T) for s in (0, 1)]
    rng = np.random.default_rng(int(sd))
    out = np.zeros((T, C.K))
    for i in range(T):
        for p in pols:
            out[i] += p.features(k0 + i)
        u = rng.random(N_CHANCE)
        a, b = E.decide(pols[0], cfg, lane, m4, 0, i), E.decide(pols[1], cfg, lane, m4, 1, i)
        t = resolve_turn_m4(lane, m4, k0 + i, a, b, cfg, lane_step=ls, u=u, ctx=ctx)
        for p in pols:
            E.observe_turn(p, cfg, lane, m4, k0 + i, (a, b))
        lane, m4 = t.lane.state, t.champ
    return out


# ---------------------------------------------------------------------------------------------- the collector
def attack_collector_cls(kind: str, seed: int, phase: str = "main", spec: Optional[str] = None):
    from ....rl.shared import HISTORICAL, LEARNER, Matchup, SharedCollector

    class AttackSampler:
        def __init__(self):
            self.net, self.mode = target_net(kind, seed, phase, spec), target_mode(kind, seed, phase)
            self.counts: Dict[str, int] = {}

        def sample(self, rng):
            s = int(rng.integers(2))
            t = TargetAgent(self.net, self.mode, f"target:{kind}:{seed}")
            m = Matchup(HISTORICAL, (LEARNER, t) if s == 0 else (t, LEARNER), f"target:{kind}", s)
            self.counts[m.opponent] = self.counts.get(m.opponent, 0) + 1
            return m

        def describe(self) -> dict:
            return {"target": f"{kind}:{seed}", "draws": dict(self.counts)}

    class AttackCollector(SharedCollector):
        extra_obs = C.K

        def __init__(self, game_cfg, cfg, sampler, seed_: int):
            from ....opponent import Belief
            if not game_cfg.obs_last_action:
                raise ValueError("attackers train with obs_last_action on")
            self.types = PL.belief_types()
            self.beliefs = [Belief(C.K, C.LAMBDA) for _ in range(cfg.n_envs)]
            self._pending: List[tuple] = []           # (env, "att" | "tgt", item)
            self.belief_updates = {"att": 0, "tgt": 0}
            self.target_sampler = AttackSampler()
            super().__init__(game_cfg, cfg, self.target_sampler, seed_)

        def _reset(self, i: int) -> None:
            self._pending = [p for p in self._pending if p[0] != i]
            super()._reset(i)
            self.beliefs[i].reset()

        def _target(self, i: int):
            return next(c for c in self.match[i].controllers if c is not LEARNER)

        def _flush(self) -> None:
            if not self._pending:
                return
            from ....opponent import loglik
            ll = loglik(self.types, self.game_cfg, [p[2] for p in self._pending])
            for (i, who, _), row in zip(self._pending, ll):
                (self.beliefs[i] if who == "att" else self._target(i).belief).update(row)
                self.belief_updates[who] += 1
            self._pending = []

        def augment(self, i: int, seat: int, vector: np.ndarray) -> np.ndarray:
            self._flush()
            return np.concatenate([np.asarray(vector, dtype=np.float32), self.beliefs[i].features()])

        def observe_step(self, i: int, actions, result) -> None:
            mt = self.match[i]
            s = mt.learner_seat
            o = self.obs[i][1 - s]                    # the target's start-of-turn observation
            self._pending.append((i, "att", (o.state, o.champ, 1 - s, o.k, int(actions[1 - s]), o.vector, o.legal)))
            t = self._target(i)
            if t.has_belief and t.mode == "live":
                a = self.obs[i][s]                    # the attacker's start-of-turn observation
                self._pending.append((i, "tgt", (a.state, a.champ, s, a.k, int(actions[s]), a.vector, a.legal)))

        def describe_starts(self) -> dict:
            return {"attack": {"target": self.target_sampler.describe(), "belief_updates": dict(self.belief_updates)}}

    return AttackCollector


def initial_net(kind: str, seed: int, phase: str = "main", spec: Optional[str] = None):
    """A copy of the target in the om input format (module docstring)."""
    from ....rl import PolicyValueNet
    from ....rl.network import widen_inputs
    src = target_net(kind, seed, phase, spec)
    if src.obs_size == C.NET_IN_OM:
        net = PolicyValueNet(src.obs_size, src.n_actions, src.hidden)
        net.load_state_dict(src.state_dict())
        return net
    return widen_inputs(src, C.NET_IN_OM)


def ppo_config(kind: str, seed: int, total: int):
    from ....rl import PPOConfig
    from ..m46c import common as C46
    return PPOConfig.continuing(C.arm_cfg("om"), total_steps=int(total), n_envs=C46.N_ENVS,
                                seed=C.ATTACK_TRAIN_ROOT + 10 * TARGET_KINDS.index(kind) + seed, torch_threads=1,
                                ent_coef=C46.ENT_COEF, hidden=C46.HIDDEN, lane_cache=C46.LANE_CACHE,
                                game_length=C46.GAME_LENGTH, start_mix=0.0)


def attacker_dir(kind: str, seed: int, phase: str = "main") -> str:
    return os.path.join(AOUT, phase, f"{kind}_s{seed}")


def target_identity(kind: str, seed: int, phase: str = "main", spec: Optional[str] = None) -> dict:
    """The target checkpoint: its specification, path, file SHA-256 and stored weights digest."""
    from ....rl import checkpoint_info
    sp = spec or target_spec(kind, seed, phase)
    p = sp if (sp.endswith(".pt") and not sp.startswith("spec:")) else C.policy_path(sp)
    return {"spec": sp, "path": p.replace("\\", "/"), "file_sha256": C.file_sha256(p),
            "stored_digest": checkpoint_info(p)["digest"]}


def train_job(args) -> dict:
    """(kind, seed, total, stop or None, phase, overwrite[, target spec, out dir[, monitor]]): one attacker under
    `health.RunMonitor` (unless monitor is False); writes summary.json. The optional target spec replaces the M5 final
    (readiness stand-ins). The summary records the target's identity and its in-memory weights digest before and after
    training (the target must stay frozen)."""
    kind, seed, total, stop, phase, overwrite, *rest = args
    spec = rest[0] if rest else None
    out = (rest[1] if len(rest) > 1 else None) or attacker_dir(kind, seed, phase) + ("_smoke" if stop else "")
    monitor = rest[2] if len(rest) > 2 else True
    t0 = time.perf_counter()
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        import torch
        torch.set_num_threads(1)
        from ....rl import checkpoint_info, train_shared, weights_digest
        from ..m43b.train import summarize_history
        from ..m45c.train import rollout_summary
        from ..m46c.train import shared_config
        from .health import RunMonitor
        from .train import Stats
        cfg = C.setup("om")
        ppo = ppo_config(kind, seed, total)
        target = target_identity(kind, seed, phase, spec)
        tnet = target_net(kind, seed, phase, spec)         # the object the collector plays (net_of caches it)
        target["digest_start"] = weights_digest(tnet.state_dict())
        net = initial_net(kind, seed, phase, spec)
        init_digest = weights_digest(net.state_dict())
        res = train_shared(cfg, ppo, shared_config(), out_dir=out, net=net, name=f"m5_attack_{kind}_s{seed}",
                           verbose=False, overwrite=overwrite, stop_at=stop, rollout_stats=Stats("om", ppo),
                           collector_cls=attack_collector_cls(kind, seed, phase, spec),
                           on_update=RunMonitor(out) if monitor else None)
        end_epoch = time.time()
        target["digest_end"] = weights_digest(tnet.state_dict())
        h = res.history
        s = summarize_history(h, ppo.reward_scale)
        s.pop("curve", None)
        s.update({"status": "complete", "name": f"{kind}_s{seed}", "kind": kind, "seed": seed, "phase": phase,
                  "schedule_total_steps": int(total), "stop_at": stop, "budget": stop or int(total),
                  "agent_transitions": res.agent_transitions, "env_turns": res.env_turns,
                  "elapsed_s": res.elapsed, "transitions_per_s": res.agent_transitions / res.elapsed,
                  "steady_transitions_per_s": float(np.median([e["sps"] for e in h[len(h) // 3:]])) if h else None,
                  "historical_return_last": h[-1]["historical_return"] if h else None,
                  "digest": weights_digest(res.net.state_dict()), "initial_digest": init_digest,
                  "run_dir": out, "target": target, "target_spec": target["spec"], "ppo": ppo.to_dict(),
                  "rollout": rollout_summary(h), "monitored": bool(monitor), "started": started,
                  "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "wall_s": time.perf_counter() - t0,
                  "train_start_epoch": end_epoch - res.elapsed, "train_end_epoch": end_epoch,
                  "collector": (checkpoint_info(os.path.join(out, "final.pt")).get("metadata") or {}).get("starts")})
    except BaseException as ex:                            # noqa: BLE001 - recorded, never swallowed silently
        import traceback
        from .health import StopCondition
        from .train import history
        s = {"status": "stopped" if isinstance(ex, StopCondition) else "failed", "name": f"{kind}_s{seed}",
             "kind": kind, "seed": seed, "phase": phase, "run_dir": out, "error": repr(ex),
             "traceback": traceback.format_exc(), "started": started, "wall_s": time.perf_counter() - t0,
             "updates_written": len(history(out))}
    if os.path.isdir(out):
        C.jdump(os.path.join(out, "summary.json"), s)
    return s


# ---------------------------------------------------------------------------------------------- the stage
def _complete(d: str) -> bool:
    p = os.path.join(d, "summary.json")
    return os.path.exists(p) and (C.jload(p) or {}).get("status") == "complete"


def plan_runs(kinds=CORE_KINDS, seeds=C.SEEDS, phase: str = "main", restart_incomplete: bool = False,
              total: int = ATTACK_BUDGET, out: str = C.OUT) -> List[tuple]:
    """The attacker runs still to do, the longer om-target runs first: complete runs are skipped; an incomplete
    directory is refused unless restart_incomplete, in which case it is preserved under attack/<phase>/partial/ first
    (runs are never overwritten)."""
    from .train import preserve_partial
    todo = []
    for kind in kinds:
        for s in seeds:
            d = attacker_dir(kind, s, phase)
            if _complete(d):
                continue
            if os.path.isdir(d) and os.listdir(d):
                if not restart_incomplete:
                    raise SystemExit(f"{d} exists and is incomplete: inspect it, then pass --restart-incomplete to "
                                     "preserve it under partial/ and restart the run (runs are never overwritten)")
                preserve_partial(d, f"attack_{phase}", f"attack_{kind}_s{s}", root=os.path.join(AOUT, phase, "partial"),
                                 out=out)
            todo.append((kind, s, int(total), None, phase, False))
    return todo


def train(workers: int, kinds=CORE_KINDS, seeds=C.SEEDS, phase: str = "main", restart_incomplete: bool = False) -> dict:
    """The attacker stage under the main training's failure policy (`health.run_campaign`) with the progress and
    resource logs, then `check`. The caller has checked the user's approval (--confirm-m5-attackers) and the gate."""
    from ..m46c.train import ResourceLog
    from . import health as HL
    todo = plan_runs(kinds, seeds, phase, restart_incomplete)
    names = [f"{j[0]}_s{j[1]}" for j in todo]
    q = ((C.load("budget") or {}).get("attackers") or {}).get("quarter") or {}
    rates = {"om": (q.get("tps") or {}).get("om_target", 620.0), "sp": (q.get("tps") or {}).get("sp_target", 767.0)}
    est = ((C.load("plan_future") or {}).get("attackers") or {}).get("core_12", {}).get("training_wall_h")
    log = ResourceLog(os.path.join(C.OUT, f"resources_attack_{phase}.jsonl"))
    prog = HL.Progress(f"attack_{phase}", names, workers, ATTACK_BUDGET, rates=rates, estimate=est,
                       dir_of=lambda n: attacker_dir(n.split("_s")[0], int(n.split("_s")[1]), phase))
    log.start()
    prog.start()
    t0 = time.perf_counter()
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        out = HL.run_campaign(train_job, todo, names, workers)
    finally:
        log.stop_flag.set()
        prog.stop_flag.set()
        try:
            prog.write()
        except Exception:                                  # pragma: no cover
            pass
    rec = {"runs": names, "workers": workers, "started": started, "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
           "wall_s": time.perf_counter() - t0, "status": {n: r.get("status") for n, r in out["results"].items()},
           "stopped": out["stopped"], "run_wall_h": {n: (r.get("wall_s") or 0) / 3600 for n, r in out["results"].items()},
           "estimate_wall_h": est, "rates_estimated": rates}
    C.save_json(f"wall_attack_{phase}", rec)
    return check(phase, kinds, seeds)


# ---------------------------------------------------------------------------------------------- health check and gate
def restart_checks(phase: str = "main", out_dir: str = C.OUT) -> Dict[str, dict]:
    """Every attacker restart of this phase (results/m5/restarts.json): the restart must reproduce the preserved prefix."""
    from .train import prefix_check
    out = {}
    for rec in (C.load("restarts", out_dir) or {}).get("restarts", []):
        if rec.get("phase") != f"attack_{phase}":
            continue
        kind, s = rec["run"][len("attack_"):].split("_s")
        out[f"{rec['run']} ({rec['moved_to'].split('/')[-1]})"] = prefix_check(rec["moved_to"],
                                                                                 attacker_dir(kind, int(s), phase))
    return out


def check(phase: str = "main", kinds=CORE_KINDS, seeds=C.SEEDS, total: int = ATTACK_BUDGET, save: bool = True,
          out: str = C.OUT, target_ids: Optional[Dict[str, dict]] = None) -> dict:
    """The attackers' health and identity check, rebuilt from the run directories (the main training's checks where
    they apply, pre-registration §3): complete, the budget within one update, finite statistics, no termination and
    one truncation per game, a bounded critic, no illegal action, the runtime monitor agreeing with the post-hoc scan,
    no stop condition; the initial weights = the designed copy of the target; the target = the approved final (file
    SHA-256 recorded by the training check) and frozen (its in-memory weights equal to the file's before and after
    training); s0 starts and 512-turn games; every game against the target, the learner's seats balanced; the
    attacker's belief updated on every turn and the target's on every turn exactly when it is a live om target; every
    restart reproducing its preserved prefix. `target_ids`: {target spec: {"file_sha256"}} in place of the training
    check's identities (readiness stand-ins)."""
    from ....env.discounted import tail_constant
    from ....rl import weights_digest
    from ..m45c.full import stability_scan
    from . import health as HL
    from .train import history
    if target_ids is None:
        tc = C.load("train_main") or {}
        target_ids = {f"{a}:{s}": ((tc.get("identities") or {}).get(f"{a}_s{s}") or {}).get("final") or {}
                      for a in C.ARMS for s in C.SEEDS}
    runs, checks, scans, live, ids = {}, {}, {}, {}, {}
    one_update = 16 * 128                                   # one learner seat per game: n_envs x rollout length
    omega = tail_constant(C.arm_cfg("om"))
    for kind in kinds:
        for s in seeds:
            name = f"{kind}_s{s}"
            d = attacker_dir(kind, s, phase)
            r = C.jload(os.path.join(d, "summary.json"))
            h = history(d)
            scans[name] = stability_scan(h)
            live[name] = HL.alarms(d)
            if r is None or r.get("status") != "complete":
                runs[name] = {"status": (r or {}).get("status") or ("missing" if not h else f"incomplete ({len(h)})"),
                              "error": (r or {}).get("error"), "updates_written": len(h),
                              "alarms": [x["alarm"] for x in live[name]]}
                continue
            runs[name] = r
            fin = os.path.join(d, "final.pt")
            ids[name] = {"path": fin.replace("\\", "/"), "file_sha256": C.file_sha256(fin)}
            tg = r["target"]
            col = (r.get("collector") or {}).get("attack") or {}
            draws = (col.get("target") or {}).get("draws") or {}
            bu = col.get("belief_updates") or {}
            seats = r["transitions_seat_a"] + r["transitions_seat_b"]
            ppo = r.get("ppo") or {}
            checks.update({
                f"{name}: complete, budget reached (within one update)":
                    total <= r["agent_transitions"] < total + one_update if r.get("stop_at") is None
                    else r["agent_transitions"] >= r["stop_at"],
                f"{name}: finite losses and gradients": bool(r["finite_losses_and_gradients"]),
                f"{name}: never terminated; one truncation per finished game": r["terminations"] == 0 and
                r["truncations"] == r["games_finished"],
                f"{name}: critic bounded (|V| <= Omega)": r["value_absmax_resources"] <= omega,
                f"{name}: no illegal action sampled": r["rollout"].get("illegal_sampled_total", 1) == 0,
                f"{name}: runtime monitor attached and agreeing with the post-hoc scan": bool(r.get("monitored")) and
                sorted(x["alarm"] for x in live[name]) == sorted(scans[name].get("alarms", [])),
                f"{name}: no pre-registered stop condition": not scans[name]["alarms"],
                f"{name}: initial weights = the target copied into the om format":
                    r["initial_digest"] == weights_digest(initial_net(kind, s, phase, tg["spec"]).state_dict()),
                f"{name}: target = the approved final (file SHA-256)":
                    tg["file_sha256"] == (target_ids.get(tg["spec"]) or {}).get("file_sha256"),
                f"{name}: target frozen (in-memory weights = the file's, before and after)":
                    tg["digest_start"] == tg["digest_end"] == tg["stored_digest"],
                f"{name}: s0 starts, 512-turn games, schedule {total}": ppo.get("start_mix") == 0.0 and
                ppo.get("game_length") == 512 and r["schedule_total_steps"] == total,
                f"{name}: every game against the target": set(draws) == {f"target:{kind}"} and
                sum(draws.values()) >= r["games_finished"],
                f"{name}: the learner's seats balanced (within 5%)":
                    seats > 0 and abs(r["transitions_seat_a"] - r["transitions_seat_b"]) <= 0.05 * seats,
                f"{name}: attacker belief updated on every turn": bu.get("att") == r["env_turns"],
                f"{name}: target belief updated on every turn iff a live om target":
                    bu.get("tgt") == (r["env_turns"] if kind == "om" else 0)})
    rs = restart_checks(phase, out)
    for k, v in rs.items():
        checks[f"restart {k}: reproduces the preserved history prefix"] = bool(v["identical"])
    complete = [n for n, r in runs.items() if r.get("status") == "complete"]
    tech = {n: r for n, r in runs.items() if r.get("status") != "complete"}
    data = {"meta": C.meta(stage="attack_check", phase=phase), "runs": runs, "stability": scans, "checks": checks,
            "live_alarms": {n: v for n, v in live.items() if v}, "complete": complete,
            "technical_failures": {n: {"status": r["status"], "error": r.get("error"), "alarms": r.get("alarms")}
                                   for n, r in tech.items()},
            "restarts": rs, "identities": ids,
            "failed_checks": [k for k, v in checks.items() if not v],
            "all_passed": bool(checks) and all(checks.values()) and len(complete) == len(kinds) * len(seeds)}
    if save:
        C.save_json(f"attack_check_{phase}", data, out)
    return data


def require_ok(phase: str = "main", kinds=CORE_KINDS, out: str = C.OUT) -> dict:
    """The gate of the attacker evaluation: the attacker check exists, every run of `kinds` is complete, every failed
    check has a recorded resolution as a training property (`health.resolve` with phase "attack_<phase>"), and the
    evaluated attackers are the files the check recorded (SHA-256)."""
    from . import health as HL
    d = C.load(f"attack_check_{phase}", out)
    if not d:
        raise SystemExit(f"no attacker check (results/m5/attack_check_{phase}.json): run stage attack_check first")
    names = [f"{k}_s{s}" for k in kinds for s in C.SEEDS]
    missing = [n for n in names if n not in d.get("complete", [])]
    if missing:
        raise SystemExit(f"attackers incomplete or failed ({missing}): evaluation refused; investigate (§3 rules)")
    res = HL.resolutions(f"attack_{phase}", out)
    open_ = [k for k, v in d["checks"].items() if not v
             and (res.get(k) or {}).get("classification") != "training property (not a defect)"]
    if open_:
        raise SystemExit("failed attacker checks without a recorded resolution: " + "; ".join(open_))
    bad = [n for n in names if C.file_sha256(d["identities"][n]["path"]) != d["identities"][n]["file_sha256"]]
    if bad:
        raise SystemExit(f"attacker checkpoints differ from the check's identities: {bad}")
    return d


def eval_jobs(kinds=CORE_KINDS, seeds=C.SEEDS, phase: str = "main", N: int = N_EVAL) -> List[dict]:
    """The attackers' independent evaluation cells (games.cell_job): each attacker against its target, the other arm's
    final and the common start (the target's belief trace is recorded in the om-target cells). When both om and
    om_table are included, the live/table cross-evaluation is added: the om attacker against the table-belief target
    ("cross_table", on the games of the om_table attacker's own target cell) and the om_table attacker against the live
    target ("cross_live", on the games of the om attacker's own target cell). Each target form is then played by both
    attackers on the same games, so target-specific tuning can be told apart from the channel."""
    js = []
    for s in seeds:
        for kind in kinds:
            att = ("net", f"spec:{os.path.join(attacker_dir(kind, s, phase), 'final.pt')}")
            tgt_kind = "om" if kind == "om_table" else kind
            mode = f"table:{table_path(s, phase)}" if kind == "om_table" else "live"
            targets = {"target": ("net", f"{tgt_kind}:{s}", mode),
                       "other_arm": ("net", f"{'sp' if tgt_kind == 'om' else 'om'}:{s}"),
                       "start": ("net", f"start:{s}")}
            if "om" in kinds and "om_table" in kinds and kind in ("om", "om_table"):
                if kind == "om":
                    targets["cross_table"] = ("net", f"om:{s}", f"table:{table_path(s, phase)}")
                else:
                    targets["cross_live"] = ("net", f"om:{s}", "live")
            for c, (name, opp) in enumerate(targets.items()):
                if name == "cross_table":
                    cell = 100 * s + 10 * TARGET_KINDS.index("om_table")
                elif name == "cross_live":
                    cell = 100 * s + 10 * TARGET_KINDS.index("om")
                else:
                    cell = 100 * s + 10 * TARGET_KINDS.index(kind) + c
                js.append({"key": f"attack|{phase}|{kind}|s{s}|{name}",
                           "path": os.path.join(AOUT, "eval", phase, f"{kind}_s{s}__{name}.npz"),
                           "focals": [att], "opponent": opp, "N": N, "T": 500, "root": C.ATTACK_EVAL_ROOT,
                           "cell": cell})
    return js


__all__ = ["TargetAgent", "attack_collector_cls", "initial_net", "ppo_config", "attacker_dir", "train_job",
           "target_identity", "plan_runs", "train", "check", "require_ok", "restart_checks",
           "eval_jobs", "self_play_table", "ATTACK_BUDGET", "HALF_BUDGET", "QUARTER_BUDGET", "TARGET_KINDS", "N_EVAL",
           "CORE_KINDS", "OPTIONAL_KINDS"]
