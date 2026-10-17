"""OM2 evaluation (design §6; families frozen in the pre-registration).

Game m of a cell starts from s0 with chance stream default_rng(sd) and is played twice, the evaluated network in seat A
and in seat B (paired seats). An OM2 network ("arm player") samples by the inverse CDF of its policy with the uniform
u[i] of default_rng([sd, 7, seat]) (as `m45c.horizon.CRNNet`), on its input = observe_full(state, previous actions)
followed by its belief's features (treatment) or the prior (control); its belief is updated after every turn with the
opponent's action, exactly as in training. Fixed opponents act through the agent interface with the generator
default_rng([sd, 11, seat]). Both arms of a seed meet every opponent on the same games and streams.

Families: E (exploitation: archetypes in the library and held-out scripted variants), R1 (competent learned opponents:
the other seeds' adopted baseline policies and held-out learned instances), R2 (misspecified: held-out mixtures and
switchers), R3 (head to head, treatment against control). Values: the evaluated seat's lead at turn 114 (the lane-exit
readout E_20 at turn 94 in expectation), the lane lead at 94, the lead at T. The treatment's belief is recorded at
CHECK_TURNS (mechanism).
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import numpy as np

from ....champions import N_CHANCE
from ....champions_m4 import context, resolve_turn_m4
from . import common as C

T_EVAL = 150
READ_TURN = 114
LANE_END = 94
CHECK_TURNS = (1, 3, 5, 10, 20, 40, 60, 94)
N_PAIRS = 64
FAMILIES = {
    "E": list(C.ARCHETYPES) + list(C.HELDOUT_SCRIPTED),
    "R1": [f"base:{o}" for o in C.COMPETENT_SEEDS] + list(C.HELDOUT_LEARNED),
    "R2": list(C.HELDOUT_MIXTURES) + list(C.HELDOUT_SWITCHERS),
}


class ArmPlayer:
    """An OM2 network with its belief (treatment) or the prior (control); CRN sampling."""

    def __init__(self, net, arm: str, seed: int, seat: int, T: int, types):
        from ....opponent import Belief, prior_features
        self.net, self.arm, self.types = net, arm, types
        self.u = np.random.default_rng([int(seed), 7, int(seat)]).random(int(T) + 1)
        self.belief = Belief(C.K, C.LAMBDA)
        self.prior = prior_features(C.K)

    def features(self) -> np.ndarray:
        return self.belief.features() if self.arm == "treatment" else self.prior

    def decide(self, obs, i: int) -> int:
        import torch
        x = np.concatenate([obs.vector.astype(np.float32), self.features()])
        with torch.no_grad():
            logits, _ = self.net(torch.as_tensor(x).unsqueeze(0))
        lg = logits[0].double().numpy()[list(obs.legal)]
        p = np.exp(lg - lg.max())
        p /= p.sum()
        if len(obs.legal) == 1:
            return int(obs.legal[0])
        j = min(int(np.searchsorted(np.cumsum(p), self.u[i], side="right")), len(p) - 1)
        return int(obs.legal[j])

    def observe(self, cfg, opp_obs, opp_code: int) -> None:
        if self.arm != "treatment":
            return
        from ....opponent import loglik
        it = (opp_obs.state, opp_obs.champ, opp_obs.side, opp_obs.k, int(opp_code), opp_obs.vector, opp_obs.legal)
        self.belief.update(loglik(self.types, cfg, [it])[0])


class PiPlayer(ArmPlayer):
    """A network without opponent information (90 inputs: the prefix of the observation), with the same CRN sampling
    as an arm player; used by the exploratory reference (explore.py)."""

    def __init__(self, net, seed: int, seat: int, T: int, types):
        super().__init__(net, "pi", seed, seat, T, types)

    def features(self) -> np.ndarray:
        return np.zeros(0, dtype=np.float32)

    def decide(self, obs, i: int) -> int:
        class _O:
            pass
        o = _O()
        o.vector, o.legal = obs.vector[:self.net.obs_size], obs.legal
        return super().decide(o, i)


def play(cfg, a_spec: tuple, b_spec: tuple, sd: int, T: int, types, lane_step=None, collect: bool = False) -> dict:
    """One game from s0. Seat specs: ("arm", net, arm), ("pi", net) (a 90-input network, CRN sampling) or
    ("opp", name). Returns r [T] (seat A's reward) and the belief log posteriors of each arm seat at CHECK_TURNS."""
    from ...m45b import states as S
    from ....diagnostics.branch import Snapshot, observation
    lane, m4, k0 = S.s0(cfg)
    ctx = context(cfg)
    players = []
    for seat, spec in ((0, a_spec), (1, b_spec)):
        if spec[0] == "arm":
            players.append(ArmPlayer(spec[1], spec[2], sd, seat, T, types))
        elif spec[0] == "pi":
            players.append(PiPlayer(spec[1], sd, seat, T, types))
        else:
            players.append((C.agent(spec[1]), np.random.default_rng([int(sd), 11, seat])))
    rng = np.random.default_rng(int(sd))
    last = (-1, -1)
    r = np.zeros(T)
    trace = {s: np.zeros((len(CHECK_TURNS), C.K)) for s in (0, 1) if isinstance(players[s], ArmPlayer)}
    checks = {t: j for j, t in enumerate(CHECK_TURNS)}
    seen = []
    for i in range(T):
        u = rng.random(N_CHANCE)
        snap = Snapshot(lane, m4, k0 + i, (cfg.skill_a, cfg.skill_b), last)
        obs = (observation(cfg, snap, 0), observation(cfg, snap, 1))
        codes = []
        for s in (0, 1):
            p = players[s]
            if isinstance(p, ArmPlayer):
                if collect and len(obs[s].legal) > 1:
                    seen.append((obs[s].vector.copy(), tuple(obs[s].legal)))
                codes.append(p.decide(obs[s], i))
            else:
                codes.append(int(p[0].act(obs[s], p[1])))
        for s in (0, 1):
            if isinstance(players[s], ArmPlayer):
                players[s].observe(cfg, obs[1 - s], codes[1 - s])
                if (i + 1) in checks:
                    trace[s][checks[i + 1]] = players[s].belief.lp
        t = resolve_turn_m4(lane, m4, k0 + i, codes[0], codes[1], cfg, lane_step=lane_step, u=u, ctx=ctx)
        r[i] = t.reward
        lane, m4 = t.lane.state, t.champ
        last = (codes[0], codes[1])
    return {"r": r, "trace": trace, "seen": seen}


def cell_job(j: dict) -> tuple:
    """`j`: key, path, seed s, arms (list), opponent (name) or "h2h", N, T, cell. Every arm plays the opponent on the
    same N paired-seat games; h2h plays treatment vs control."""
    path = j["path"]
    if os.path.exists(path):
        return j["key"], {"path": path, "cached": True}
    from ...m45b import engine as E
    cfg = C.setup()
    ls = E.lane_cache(cfg)
    types = C.belief_types()
    nets = {a: C.om_net(p) for a, p in j["nets"].items()}
    N, T = int(j["N"]), int(j["T"])
    out: Dict[str, np.ndarray] = {}
    t0 = time.perf_counter()
    pairs = [("treatment", "control")] if j["opponent"] == "h2h" else [(a, None) for a in nets]
    for arm, other in pairs:
        R = np.zeros((N, 2, T), dtype=np.float32)
        tr = np.zeros((N, 2, len(CHECK_TURNS), C.K), dtype=np.float32)
        for m in range(N):
            sd = int(C.EVAL_ROOT) * 10 ** 7 + int(j["cell"]) * 10 ** 4 + m
            me = ("arm", nets[arm], arm)
            opp = ("arm", nets[other], other) if other else ("opp", j["opponent"])
            for o, (sa, sb) in enumerate(((me, opp), (opp, me))):
                g = play(cfg, sa, sb, sd, T, types, ls)
                R[m, o] = g["r"] if o == 0 else -g["r"]
                seat = 0 if o == 0 else 1
                if seat in g["trace"]:
                    tr[m, o] = g["trace"][seat]
        out[f"{arm}__r"] = R
        out[f"{arm}__trace"] = tr
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, **out, elapsed=np.asarray(time.perf_counter() - t0))
    os.replace(tmp, path)
    return j["key"], {"path": path, "elapsed": time.perf_counter() - t0}


def final_path(arm: str, s: int, phase: str = "main") -> str:
    return os.path.join(C.run_dir(arm, s, phase), "final.pt")


def jobs(seeds=C.PILOT_SEEDS, phase: str = "main", N: int = N_PAIRS, T: int = T_EVAL, nets=None) -> List[dict]:
    js = []
    for s in seeds:
        net_paths = nets(s) if nets else {a: final_path(a, s, phase) for a in C.ARMS}
        cell = 0
        for fam, opps in list(FAMILIES.items()) + [("R3", ["h2h"])]:
            for o in opps:
                key = f"eval|{phase}|s{s}|{fam}|{o}"
                js.append({"key": key, "path": os.path.join(C.CKPT, "eval", phase, f"s{s}", f"{fam}__{o.replace(':', '_')}.npz"),
                           "seed": s, "opponent": o, "family": fam, "nets": net_paths, "N": N, "T": T,
                           "cell": s * 100 + cell})
                cell += 1
    return js


def load(path: str) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def pair_lead(R: np.ndarray, t: int = READ_TURN) -> np.ndarray:
    return R[:, :, :t].sum(-1).mean(-1)


# ---------------------------------------------------------------------------------------------- belief-swap probes
def probes(net, states_obs: List[tuple], types_n: int = None) -> dict:
    """How much the policy's action distribution depends on the belief: for observations (vector, legal), the mean
    total-variation distance between the policies with the belief set to each type's one-hot and with the prior."""
    import torch
    from ....opponent import prior_features
    K = C.K
    tv = np.zeros(K)
    for vec, legal in states_obs:
        base = np.concatenate([vec.astype(np.float32), prior_features(K)])
        xs = [base]
        for k in range(K):
            oh = np.zeros(K, dtype=np.float32)
            oh[k] = 1.0
            xs.append(np.concatenate([vec.astype(np.float32), oh]))
        with torch.no_grad():
            lg, _ = net(torch.as_tensor(np.stack(xs)))
        lg = lg.double().numpy()[:, list(legal)]
        p = np.exp(lg - lg.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        tv += 0.5 * np.abs(p[1:] - p[0]).sum(1)
    tv /= max(1, len(states_obs))
    return {C.BELIEF_TYPES[k]: float(tv[k]) for k in range(K)}


def run(workers: int, seeds=C.PILOT_SEEDS, phase: str = "main") -> dict:
    js = sorted(jobs(seeds, phase), key=lambda x: -(2 if x["opponent"] != "h2h" else 1))
    t0 = time.perf_counter()
    C.run_jobs(cell_job, js, workers, f"OM2 evaluation ({phase})")
    return {"jobs": len(js), "wall_s": time.perf_counter() - t0}


__all__ = ["ArmPlayer", "PiPlayer", "play", "cell_job", "jobs", "run", "load", "pair_lead", "probes", "FAMILIES", "N_PAIRS",
           "T_EVAL", "READ_TURN", "CHECK_TURNS"]
