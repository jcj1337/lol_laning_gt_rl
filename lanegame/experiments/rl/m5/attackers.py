"""M5d's dedicated attackers (design §8.4): learned best responses to a frozen M5 final, reported as exploitability
LOWER bounds (limited by the attacker's budget, network and optimizer; never a certificate).

Proposal (every item is in the pre-registration's attacker section and is a review decision):
    target       a frozen final: om (its belief live, updated from the attacker's actions exactly as in deployment), sp,
                 or for the adaptation test om_table (the om final with its belief inputs replaced by the mean belief of
                 its own self-play at that turn: the same network without in-game adaptation). Black-box access only:
                 the attacker plays the target; it never reads the target's weights, belief or gradients.
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

The collector keeps one belief per learner seat (the attacker's) and one target instance per game (the target's belief
lives in it); both beliefs are updated after every step from the other seat's code on its start-of-turn observation,
batched across environments before any decision of the next step.
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
TARGET_KINDS = ("om", "sp", "om_table")
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


def train_job(args) -> dict:
    """(kind, seed, total, stop or None, phase, overwrite[, target spec, out dir]): one attacker; writes summary.json.
    The optional target spec replaces the M5 final (readiness stand-ins)."""
    kind, seed, total, stop, phase, overwrite, *rest = args
    spec = rest[0] if rest else None
    out = (rest[1] if len(rest) > 1 else None) or attacker_dir(kind, seed, phase) + ("_smoke" if stop else "")
    t0 = time.perf_counter()
    try:
        import torch
        torch.set_num_threads(1)
        from ....rl import checkpoint_info, train_shared, weights_digest
        from ..m46c.train import shared_config
        cfg = C.setup("om")
        ppo = ppo_config(kind, seed, total)
        net = initial_net(kind, seed, phase, spec)
        res = train_shared(cfg, ppo, shared_config(), out_dir=out, net=net, name=f"m5_attack_{kind}_s{seed}",
                           verbose=False, overwrite=overwrite, stop_at=stop,
                           collector_cls=attack_collector_cls(kind, seed, phase, spec))
        end_epoch = time.time()
        h = res.history
        s = {"status": "complete", "kind": kind, "seed": seed, "phase": phase, "agent_transitions": res.agent_transitions,
             "updates": len(h), "elapsed_s": res.elapsed, "transitions_per_s": res.agent_transitions / res.elapsed,
             "steady_transitions_per_s": float(np.median([e["sps"] for e in h[len(h) // 3:]])) if h else None,
             "historical_return_last": h[-1]["historical_return"] if h else None,
             "digest": weights_digest(res.net.state_dict()), "run_dir": out, "wall_s": time.perf_counter() - t0,
             "target_spec": spec or target_spec(kind, seed, phase), "train_start_epoch": end_epoch - res.elapsed,
             "train_end_epoch": end_epoch,
             "collector": (checkpoint_info(os.path.join(out, "final.pt")).get("metadata") or {}).get("starts")}
    except BaseException as ex:                            # noqa: BLE001
        import traceback
        s = {"status": "failed", "kind": kind, "seed": seed, "error": repr(ex), "traceback": traceback.format_exc(),
             "wall_s": time.perf_counter() - t0}
    if os.path.isdir(out):
        C.jdump(os.path.join(out, "summary.json"), s)
    return s


def eval_jobs(kinds=("om", "sp"), seeds=C.SEEDS, phase: str = "main", N: int = N_EVAL) -> List[dict]:
    """The attackers' independent evaluation cells (games.cell_job): each attacker against its target, the other arm's
    final and the common start; plus, for om targets, the om final against the attacker with its belief traced."""
    js = []
    for s in seeds:
        for kind in kinds:
            att = ("net", f"spec:{os.path.join(attacker_dir(kind, s, phase), 'final.pt')}")
            tgt_kind = "om" if kind == "om_table" else kind
            mode = f"table:{table_path(s, phase)}" if kind == "om_table" else "live"
            targets = {"target": ("net", f"{tgt_kind}:{s}", mode),
                       "other_arm": ("net", f"{'sp' if tgt_kind == 'om' else 'om'}:{s}"),
                       "start": ("net", f"start:{s}")}
            for c, (name, opp) in enumerate(targets.items()):
                js.append({"key": f"attack|{phase}|{kind}|s{s}|{name}",
                           "path": os.path.join(AOUT, "eval", phase, f"{kind}_s{s}__{name}.npz"),
                           "focals": [att], "opponent": opp, "N": N, "T": 500, "root": C.ATTACK_EVAL_ROOT,
                           "cell": 100 * s + 10 * TARGET_KINDS.index(kind) + c})
    return js


__all__ = ["TargetAgent", "attack_collector_cls", "initial_net", "ppo_config", "attacker_dir", "train_job",
           "eval_jobs", "self_play_table", "ATTACK_BUDGET", "HALF_BUDGET", "QUARTER_BUDGET", "TARGET_KINDS", "N_EVAL"]
