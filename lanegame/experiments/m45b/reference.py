"""A learned reference opponent for the restricted games (evaluation only; nothing is trained).

The M4.4b aggro finals (4,096,000 transitions, training seeds 0-2; results/rl/checkpoints/m44/ext, git-ignored) play
against scripted strategies from s0 on DEV seeds, in both seats, 200 turns:

  native     on their own preset `m4_four_actions_econ_aggro()` (88-entry observation)
  transfer   on an M4.5 preset through an adapter that drops the two W entries of the 90-entry observation (an
             OUT-OF-DISTRIBUTION test: the network never saw minion targeting, the reach ball or XP range 2, and code 30,
             its siege-Vision code, is now the Ward action)

The question is only whether a conditional policy beats the scripted Shove that dominates every restricted scripted
game, i.e. whether that dominance is a property of the game or of the scripted strategy set. Sampling uses each seat's
own stream (numpy default_rng([seed, 7, seat])).
"""

from __future__ import annotations

import os
from typing import List

import numpy as np

from . import common as Cm
from . import engine as E
from . import states as S
from . import strategies as ST

OPPONENTS = ("g_shove", "g_shove_sr", "g_farm", "g_guard", "grow_shove4", "wk_shove_reset_sh")


class NetPolicy:
    """A NetAgent as an engine policy; `drop` = observation entries removed before the network (the W entries)."""

    def __init__(self, agent, k0: int, seed: int, seat: int, drop=()):
        self.agent, self.k0, self.drop = agent, k0, list(drop)
        self.rng = np.random.default_rng([seed, 7, seat])

    def decide(self, cfg, lane, m4, s, i):
        import torch
        from ...diagnostics.branch import Snapshot, observation
        obs = observation(cfg, Snapshot(lane, m4, self.k0 + i, (cfg.skill_a, cfg.skill_b)), s)
        v = np.delete(obs.vector, self.drop) if self.drop else obs.vector
        with torch.no_grad():
            logits, _ = self.agent.net(torch.as_tensor(v, dtype=torch.float32).unsqueeze(0))
            logits = logits[0].double()
            logits = torch.where(torch.as_tensor(obs.mask), logits, torch.full_like(logits, -np.inf))
            p = torch.softmax(logits, dim=-1).numpy()[list(obs.legal)]
        c = np.cumsum(p)
        j = min(int(np.searchsorted(c, self.rng.random() * c[-1], side="right")), len(c) - 1)
        return obs.legal[j], "net"


def available() -> bool:
    from ..rl.m44.common import final_path
    return all(os.path.exists(final_path("ext", "aggro", s)) for s in (0, 1, 2))


def _job(args):
    base, net_seed, opp, seeds, turns = args
    import torch
    torch.set_num_threads(1)
    from ...env.observation_m4 import m4_feature_names
    from ..rl.m44.common import agent_from_spec
    cfg = Cm.cfg(base)
    agent = agent_from_spec(f"final:ext:aggro:{net_seed}", Cm.cfg("aggro"))
    drop = [i for i, n in enumerate(m4_feature_names(cfg)) if n.endswith("champ_W")]
    start = S.s0(cfg)
    vals, led = [], []
    for sd in seeds:
        g = E.play(cfg, start, NetPolicy(agent, 0, sd, 0, drop), ST.get(opp), sd, turns, check_legal=False)["summary"]
        vals.append(g["G200"])
        led.append(g)
        g = E.play(cfg, start, ST.get(opp), NetPolicy(agent, 0, sd, 1, drop), sd, turns, check_legal=False)["summary"]
        vals.append(-g["G200"])
    keys = ("deaths_A", "deaths_B", "trade_conn_A", "trade_dealt_A", "d_min_A", "wards_A", "farm_A", "farm_B",
            "plates_A", "plates_B", "n_SH_A", "n_trade_A")
    return {"base": base, "net_seed": net_seed, "opponent": opp, "G200_net": Cm.ci(vals),
            "ledger_net_in_A": {k: float(np.mean([x[k] for x in led])) for k in keys}}


def run(bases=("aggro", "r2", "cal"), seeds=Cm.DEV[:32], turns: int = 200, workers: int = Cm.WORKERS) -> List[dict]:
    jobs = [(b, s, o, tuple(seeds), turns) for b in bases for s in (0, 1, 2) for o in OPPONENTS]
    return Cm.run_jobs(_job, jobs, workers, "learned reference")


__all__ = ["run", "available", "NetPolicy", "OPPONENTS"]
