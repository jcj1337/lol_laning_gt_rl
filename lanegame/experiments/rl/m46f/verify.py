"""Initial-equivalence verification of the two arms (pre-registration §2).

For each seed s, on representative observations (both seats of: s0 self-play of start:s, every 5th turn of 4 games of
200 turns; 64 states drawn from the M4.6c training archive; the 42 M4.6d calibration cells):
    * the action probabilities (masked softmax over the legal codes) and the values of the start network, the shared
      arm's initial network and the separate arm's initial network are compared: they must be equal exactly;
    * the separate arm's actor and critic parameters share no storage, and changing the critic trunk leaves the action
      probabilities unchanged;
    * gradient routing on one training-like minibatch: the policy(+entropy) loss reaches no critic parameter and the
      value loss no actor parameter in the separate arm; in the shared arm the value loss does reach the trunk;
    * a checkpoint round trip restores the class and the digest; the start checkpoint still loads as PolicyValueNet.
"""

from __future__ import annotations

import os
import tempfile
from typing import Dict, List, Sequence

import numpy as np

from . import common as C


def observations(seed: int) -> List[object]:
    """Representative observations (module docstring) as env Observation objects."""
    from ....env.starts import load_archive
    from ...m45b import states as S
    from ..m46c.archive import archive_path
    from ..m46d.critic import _cell_states
    from ..m46e import labels as LB
    from ..m46e.observation import _obs
    cfg = C.setup()
    net = C.net_of(C.start_path(seed))
    out = []

    def grab(i, lane, m4, obs, vals):
        if i % 5 == 0:
            out.extend(obs)

    for g in range(4):
        LB.play(cfg, net, S.s0(cfg), C.EQUIV_ROOT * 1000 + 10 * seed + g, 200, grab)
    arch = load_archive(archive_path(), cfg)
    pick = np.random.default_rng([C.EQUIV_ROOT, seed]).choice(len(arch), size=64, replace=False)
    for i in pick:
        st = arch[int(i)]
        out.extend(_obs(cfg, (st.lane, st.champ, st.phase), s) for s in (0, 1))
    for _, _, st, _ in _cell_states(cfg):
        out.extend(_obs(cfg, st, s) for s in (0, 1))
    return out


def _eval(net, obs) -> Dict[str, np.ndarray]:
    import torch
    from ....rl.network import MaskedCategorical
    x = torch.as_tensor(np.stack([o.vector for o in obs]), dtype=torch.float32)
    m = torch.as_tensor(np.stack([o.mask for o in obs]))
    with torch.no_grad():
        lg, v = net(x)
        p = MaskedCategorical(lg, m).probs
        v2 = net.value(x)
    return {"probs": p.numpy(), "value": v.numpy(), "value_fn": v2.numpy()}


def gradient_routing(net, obs) -> Dict[str, float]:
    """Gradient norms of the policy(+entropy) and value losses on each parameter group (one minibatch, a stand-in
    advantage and return: routing depends on the graph, not on the numbers)."""
    import torch
    from ....rl.network import MaskedCategorical
    x = torch.as_tensor(np.stack([o.vector for o in obs]), dtype=torch.float32)
    m = torch.as_tensor(np.stack([o.mask for o in obs]))
    g = np.random.default_rng(C.EQUIV_ROOT)
    adv = torch.as_tensor(g.normal(size=len(obs)), dtype=torch.float32)
    ret = torch.as_tensor(g.normal(size=len(obs)), dtype=torch.float32)
    groups = {"trunk": list(net.trunk.parameters()), "policy_head": list(net.policy_head.parameters()),
              "value_head": list(net.value_head.parameters())}
    if hasattr(net, "critic_trunk"):
        groups["critic_trunk"] = list(net.critic_trunk.parameters())
    params = [p for ps in groups.values() for p in ps]
    out = {}
    for lname in ("policy", "value"):
        lg, v = net(x)
        dist = MaskedCategorical(lg, m)
        if lname == "policy":
            act = dist.mode()
            loss = -(adv * dist.log_prob(act)).mean() - 0.03 * dist.entropy().mean()
        else:
            loss = 0.5 * 0.5 * ((v - ret) ** 2).mean()
        gs = torch.autograd.grad(loss, params, allow_unused=True)
        i = 0
        for gname, ps in groups.items():
            norms = [0.0 if gg is None else float(gg.norm()) for gg in gs[i:i + len(ps)]]
            out[f"{lname}->{gname}"] = float(np.sqrt(np.sum(np.square(norms))))
            i += len(ps)
    return out


def check_seed(seed: int) -> dict:
    import torch
    from ....rl import PolicyValueNet, SeparateCriticNet, load_checkpoint, save_checkpoint, weights_digest
    from .train import actor_digest, initial_net
    obs = observations(seed)
    start = C.net_of(C.start_path(seed))
    sh, sep = initial_net("shared", seed), initial_net("separate", seed)
    e0, e1, e2 = _eval(start, obs), _eval(sh, obs), _eval(sep, obs)
    rec = {"seed": seed, "observations": len(obs),
           "max_abs_dprob_shared": float(np.abs(e1["probs"] - e0["probs"]).max()),
           "max_abs_dprob_separate": float(np.abs(e2["probs"] - e0["probs"]).max()),
           "max_abs_dvalue_shared": float(np.abs(e1["value"] - e0["value"]).max()),
           "max_abs_dvalue_separate": float(np.abs(e2["value"] - e0["value"]).max()),
           "value_fn_equals_forward": bool(np.array_equal(e2["value"], e2["value_fn"])),
           "classes": [type(sh).__name__, type(sep).__name__],
           "parameters": [int(sum(p.numel() for p in sh.parameters())), int(sum(p.numel() for p in sep.parameters()))],
           "start_actor_digest": actor_digest(start), "separate_actor_digest": actor_digest(sep)}
    ptr_a = {p.data_ptr() for p in sep.actor_parameters()}
    ptr_c = {p.data_ptr() for p in sep.critic_parameters()}
    rec["shared_storage"] = len(ptr_a & ptr_c)
    probe = SeparateCriticNet.from_shared(start)
    with torch.no_grad():
        for p in probe.critic_trunk.parameters():
            p.add_(0.05)
    e3 = _eval(probe, obs)
    rec["critic_change_moves_actor"] = float(np.abs(e3["probs"] - e0["probs"]).max())
    rec["critic_change_moves_value"] = float(np.abs(e3["value"] - e0["value"]).max())
    mb = obs[:256]
    rec["routing_separate"] = gradient_routing(sep, mb)
    rec["routing_shared"] = gradient_routing(sh, mb)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "sep.pt")
        dg = save_checkpoint(path, sep, C.env_cfg(), {"kind": "verify"})
        back, _, _ = load_checkpoint(path)
        rec["roundtrip_class"] = type(back).__name__
        rec["roundtrip_digest_equal"] = weights_digest(back.state_dict()) == dg == weights_digest(sep.state_dict())
    old, _, _ = load_checkpoint(C.start_path(seed))
    rec["start_loads_as"] = type(old).__name__
    rs, rh = rec["routing_separate"], rec["routing_shared"]
    rec["checks"] = {
        f"s{seed}: shared arm initial = start (probabilities, values)": rec["max_abs_dprob_shared"] == 0.0
        and rec["max_abs_dvalue_shared"] == 0.0,
        f"s{seed}: separate arm initial = start (probabilities, values)": rec["max_abs_dprob_separate"] == 0.0
        and rec["max_abs_dvalue_separate"] == 0.0,
        f"s{seed}: separate actor parameters = start actor": rec["start_actor_digest"] == rec["separate_actor_digest"],
        f"s{seed}: actor and critic share no storage": rec["shared_storage"] == 0,
        f"s{seed}: changing the critic trunk leaves the actor unchanged": rec["critic_change_moves_actor"] == 0.0
        and rec["critic_change_moves_value"] > 0.0,
        f"s{seed}: separate: policy loss reaches no critic parameter": rs["policy->critic_trunk"] == 0.0
        and rs["policy->value_head"] == 0.0 and rs["policy->trunk"] > 0.0,
        f"s{seed}: separate: value loss reaches no actor parameter": rs["value->trunk"] == 0.0
        and rs["value->policy_head"] == 0.0 and rs["value->critic_trunk"] > 0.0,
        f"s{seed}: shared: value loss reaches the shared trunk": rh["value->trunk"] > 0.0,
        f"s{seed}: checkpoint round trip (class, digest)": rec["roundtrip_class"] == "SeparateCriticNet"
        and rec["roundtrip_digest_equal"],
        f"s{seed}: start checkpoint loads as PolicyValueNet": rec["start_loads_as"] == "PolicyValueNet",
    }
    return rec


def run(seeds: Sequence[int] = C.SEEDS, save: bool = True) -> dict:
    recs = [check_seed(s) for s in seeds]
    checks = {}
    for r in recs:
        checks.update(r["checks"])
    out = {"meta": C.meta(stage="verify"), "seeds": list(seeds), "per_seed": recs, "checks": checks,
           "all_passed": all(checks.values())}
    if save:
        C.save_json("verify" if len(seeds) == len(C.SEEDS) else "verify_smoke", out)
    return out


__all__ = ["observations", "gradient_routing", "check_seed", "run"]
