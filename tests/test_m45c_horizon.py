"""M4.5c horizon study: the rollout recorder and the paired-counterfactual machinery.

CRN policies draw their sampling uniform from (seed, seat, turn index) only; a short rollout is an exact prefix of a
long one (so every horizon is read from the same trajectory); the recorder's G_h equals the M4.5b engine's; Q_h is
G_h + 0.99^h times the critic's value of the state actually reached; one-turn candidates force exactly one action and
sustained plans stop when their condition fails; finished jobs are reused (resumable). Uses an untrained network (no
checkpoint needed). Skipped without PyTorch."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lanegame import GameConfig                                                          # noqa: E402
from lanegame.experiments.m45b import engine as E                                        # noqa: E402
from lanegame.experiments.m45b import states as S                                        # noqa: E402
from lanegame.experiments.rl.m45c import horizon as H                                    # noqa: E402
from lanegame.rl import PolicyValueNet                                                   # noqa: E402

CFG = GameConfig.m45_calibrated()


def net(seed=0):
    torch.manual_seed(seed)
    return PolicyValueNet(90, 61, (32, 32))


START = S.freeze_state(CFG, level=10, gap="large", held=12)


def pols(n, sd, h, k0=START[2]):
    return H.CRNNet(n, k0, sd, 0, h), H.CRNNet(n, k0, sd, 1, h)


def test_crn_uniforms_depend_only_on_seed_seat_and_turn():
    n = net()
    a, b = H.CRNNet(n, 0, 5, 0, 50), H.CRNNet(n, 7, 5, 0, 80)
    assert np.array_equal(a.u[:50], b.u[:50])
    assert not np.array_equal(a.u[:50], H.CRNNet(n, 0, 5, 1, 50).u[:50])
    lane, m4, k = START
    assert a.decide(CFG, lane, m4, 0, 3) == H.CRNNet(n, k - k + 0, 5, 0, 50).decide(CFG, lane, m4, 0, 3)


def test_short_rollout_is_an_exact_prefix_and_matches_the_engine():
    n = net()
    pa, pb = pols(n, 11, 40)
    long = H.rollout(CFG, START, pa, pb, 11, 40, (5, 10, 20, 30, 40), critic=n)
    pa, pb = pols(n, 11, 40)
    short = H.rollout(CFG, START, pa, pb, 11, 20, (5, 10, 20), critic=n)
    for c in (5, 10, 20):
        assert long["cp"][c] == short["cp"][c]
    pa, pb = pols(n, 11, 40)
    g = E.play(CFG, START, pa, pb, 11, 40, keep_states=True)
    for c in (5, 10, 20):
        assert long["cp"][c]["G"] == pytest.approx(g["summary"][f"G{c}"], abs=1e-12)
    # Q_h = G_h + 0.99^h V(s_h), V the critic's value of the state actually reached (seat A's observation)
    lane, m4, k = g["states"][20]
    v = H.critic_value(n, CFG, lane, m4, k, 0)
    assert long["cp"][20]["Q"] == pytest.approx(long["cp"][20]["G"] + 0.99 ** 20 * v, abs=1e-9)
    assert long["cp"][20]["V"] == pytest.approx(v, abs=1e-9)


def test_one_turn_candidate_forces_one_action_and_plans_stop_with_their_condition():
    n = net()
    own, pb = pols(n, 3, 30)
    one = H.build_candidate(H.FAMILY_CANDS["F_freeze_c"]["1t:break"], own)
    lane, m4, _ = START
    code = H.FAMILY_CANDS["F_freeze_c"]["1t:break"][1]
    assert E.decide(one, CFG, lane, m4, 0, 0) == code
    assert E.decide(one, CFG, lane, m4, 0, 1) == E.decide(own, CFG, lane, m4, 0, 1)        # handed back after 1 turn
    keep = H.build_candidate(H.FAMILY_CANDS["F_freeze_c"]["plan:keep"], own)
    assert E.decide(keep, CFG, lane, m4, 0, 5) == H.FAMILY_CANDS["F_freeze_c"]["plan:keep"][1]   # held: keeps
    lane2 = S.vision_state(CFG, "neutral", level=10)[0]                                     # no freeze: hands back
    assert not H.cond_held(CFG, lane2, m4, 0)
    assert E.decide(keep, CFG, lane2, m4, 0, 5) == E.decide(own, CFG, lane2, m4, 0, 5)


def test_state_jobs_are_paired_and_resumable(tmp_path, monkeypatch):
    monkeypatch.setattr(H, "PARTIAL", str(tmp_path))
    monkeypatch.setattr(H, "K", 2)
    monkeypatch.setattr(H, "SPEC", "initial:100")
    monkeypatch.setattr(H, "FAMILY_CANDS", {"F_freeze_c": {"1t:break": H.FAMILY_CANDS["F_freeze_c"]["1t:break"]}})
    key, d = H.state_job(("F_freeze_c", 0, START, "policy", 50, "test"))
    assert key == "F_freeze_c|0|policy" and set(d["GQ"]) == {"policy", "1t:break"}
    assert all(len(v) == 2 for v in d["GQ"]["policy"]["G"].values())
    assert d["series"]["policy"]["heldA"][0] in (0.0, 0.5, 1.0)
    # paired: a candidate identical to the policy in a rollout (it sampled the forced code) has the same G there
    for r in range(2):
        if d["first_codes"]["policy"][r] == d["first_codes"]["1t:break"][r]:
            assert d["GQ"]["policy"]["G"]["50"][r] == d["GQ"]["1t:break"]["G"]["50"][r]
    calls = []
    monkeypatch.setattr(H, "rollout", lambda *a, **k: calls.append(1))
    key2, d2 = H.state_job(("F_freeze_c", 0, START, "policy", 50, "test"))
    assert not calls and d2["GQ"] == d["GQ"]                                             # reused, not re-run
