"""M5a: the campaign drivers, the history-dependent policy path, the opt-in turn hook and the pre-registered statistics.

Tests that need git-ignored checkpoints or records skip when those files are absent."""

import os

import numpy as np
import pytest

from lanegame.experiments.rl.m5 import analysis as AN
from lanegame.experiments.rl.m5 import common as C

START0 = os.path.join("results", "rl", "checkpoints", "m46c", "train", "ext", "baseline", "s0", "final.pt")
OM2_T0 = os.path.join("results", "rl", "checkpoints", "om2", "train", "main", "treatment", "s0", "final.pt")
ARCHIVE = os.path.join("results", "rl", "checkpoints", "m46c", "archive", "train_archive.json.gz")
need_start = pytest.mark.skipif(not (os.path.exists(START0) and os.path.exists(ARCHIVE)),
                                reason="needs the git-ignored ext/baseline checkpoints and the M4.6c archive")
need_om2 = pytest.mark.skipif(not os.path.exists(OM2_T0), reason="needs the git-ignored OM2 checkpoints")


# ---------------------------------------------------------------------------------------------- seeds and specs
def test_seed_block_is_disjoint_from_every_other_study():
    from lanegame.experiments.rl.m5.readiness import scan_roots
    lo, hi = C.SEED_BLOCK
    assert all(lo <= v <= hi for v in C.ROOTS.values())
    assert len(set(C.ROOTS.values())) == len(C.ROOTS)
    found = scan_roots()
    assert len(found) > 20                                  # the scan actually reads the experiment packages
    assert not [(p, v) for p, vs in found.items() for v in vs if lo <= v <= hi]


def test_policy_specifications():
    assert C.policy_path("start:3").replace("\\", "/").endswith("m46c/train/ext/baseline/s3/final.pt")
    assert C.policy_path("om:2").replace("\\", "/").endswith("m5/train/main/om/s2/final.pt")
    assert C.policy_path("smoke/sp:0").replace("\\", "/").endswith("m5/train/smoke/sp/s0/final.pt")
    assert C.policy_path("om2:treatment:1").replace("\\", "/").endswith("om2/train/main/treatment/s1/final.pt")
    assert C.policy_path("spec:x/y.pt") == "x/y.pt"
    assert [C.arm_of(s) for s in ("om:0", "sp:0@16384000", "start:0", "om2:control:0", "bench/om:1")] == \
        ["om", "sp", "sp", "om", "om"]
    assert C.net_inputs("om") == 116 and C.net_inputs("sp") == 90
    assert C.arm_cfg("om").obs_last_action and not C.arm_cfg("sp").obs_last_action
    assert C.arm_cfg("om").with_(obs_last_action=False) == C.arm_cfg("sp")    # the same game, a longer observation


def test_campaign_stages_refuse_without_their_flag():
    from lanegame.experiments.rl.m5.__main__ import main
    for st in ("train", "families", "battery", "attackers"):
        with pytest.raises(SystemExit, match="needs --confirm-m5"):
            main([st])


# ---------------------------------------------------------------------------------------------- the opt-in turn hook
class _Counting:
    def __init__(self):
        self.calls = []

    def decide(self, cfg, lane, m4, s, i):
        from lanegame.champions import champion_legal_actions
        return int(champion_legal_actions(lane, m4, s, cfg)[0]), ""

    def observe_turn(self, cfg, lane, m4, k, codes):
        self.calls.append((k, tuple(codes)))


def test_observe_turn_reaches_each_stateful_policy_once_and_ignores_the_rest():
    from lanegame.experiments.m45b import engine as E
    from lanegame.experiments.m45b import states as S
    cfg = C.arm_cfg("sp")
    own = _Counting()
    plan = E.Then("p", E.Fixed("f", 0, own), own, 3)          # own is reachable twice: told once per turn
    opp = _Counting()
    E.play(cfg, S.s0(cfg), plan, opp, seed=5, turns=6)
    assert [k for k, _ in own.calls] == list(range(6)) and len(opp.calls) == 6
    assert own.calls == opp.calls                             # both seats see the same (A, B) codes of each turn
    E.observe_turn(lambda *a: 0, cfg, None, None, 0, (0, 0))   # a plain function: untouched, no error


def test_stateless_rollouts_are_unchanged_by_the_hook():
    """The hook adds nothing for stateless policies: two plays of scripted strategies agree with a manual loop."""
    from lanegame.champions import N_CHANCE
    from lanegame.champions_m4 import context, resolve_turn_m4
    from lanegame.experiments.m45b import engine as E
    from lanegame.experiments.m45b import states as S
    from lanegame.experiments.m45b.strategies import get
    cfg = C.arm_cfg("sp")
    pa, pb = get("g_shove"), get("g_bully")
    out = E.play(cfg, S.s0(cfg), pa, pb, seed=11, turns=40, keep_rows=True)
    lane, m4, k0 = S.s0(cfg)
    rng, ctx = np.random.default_rng(11), context(cfg)
    rs = []
    for i in range(40):
        u = rng.random(N_CHANCE)
        a, b = E.decide(pa, cfg, lane, m4, 0, i), E.decide(pb, cfg, lane, m4, 1, i)
        t = resolve_turn_m4(lane, m4, k0 + i, a, b, cfg, lane_step=E.lane_cache(cfg), u=u, ctx=ctx)
        rs.append(t.reward)
        lane, m4 = t.lane.state, t.champ
    assert [r["r"] for r in out["rows"]] == rs


def test_mirror_info_swaps_the_seats_records():
    info = {0: {"lp": [0.0], "last": [3, 7]}, 1: {"lp": [1.0], "last": [3, 7]}}
    m = __import__("lanegame.experiments.rl.m5.players", fromlist=["mirror_info"]).mirror_info(info)
    assert m[1]["lp"] == [0.0] and m[0]["lp"] == [1.0] and m[0]["last"] == [7, 3] and m[1]["last"] == [7, 3]


# ---------------------------------------------------------------------------------------------- the om policy path
@need_start
def test_widened_start_computes_the_start_exactly_in_batches():
    import torch
    from lanegame.experiments.rl.m5.readiness import probe_states
    from lanegame.experiments.rl.m5.train import initial_net
    from lanegame.experiments.rl.m46c import common as C46
    X = np.stack([v for v, _ in probe_states(n_games=1, turns=40)]).astype(np.float32)
    src, w = C46.net_of("ext/baseline:0"), initial_net("om", 0)
    extra = np.random.default_rng(0).dirichlet(np.ones(13), len(X)).astype(np.float32)
    with torch.no_grad():
        a, b = src(torch.as_tensor(X[:, :90])), w(torch.as_tensor(np.concatenate([X, extra], 1)))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


@need_start
def test_game_starts_are_common_to_both_arms():
    from lanegame.experiments.rl.m5.readiness import start_schedule
    a, b = start_schedule("om", 40), start_schedule("sp", 40)
    assert a == b and any(x.startswith("archive") for x in a) and any(x.startswith("initial") for x in a)


@need_start
def test_training_inputs_equal_the_evaluation_policys_inputs():
    """Every om training input of one rollout (s0 and archive games, game starts, previous actions, beliefs) and every
    truncation bootstrap is rebuilt by players.OMPolicy from the start-of-turn states and past codes only."""
    from lanegame.experiments.rl.m5.readiness import BELIEF_TOL, probe_inputs
    om = probe_inputs("om")
    assert om["width"] == 116 and om["prior_at_every_game_start"] and om["belief_moves_later"]
    assert om["inputs_reconstructed"] > 1000 and om["bootstraps_reconstructed"] > 0 and om["mismatches"] == 0
    assert om["max_abs_diff_by_block"]["state"] == 0 and om["max_abs_diff_by_block"]["last"] == 0
    assert om["max_abs_diff_by_block"]["belief"] <= BELIEF_TOL
    sp = probe_inputs("sp")
    assert sp["width"] == 90 and sp["inputs_are_the_observation"]


@need_start
@need_om2
def test_m5_games_reproduce_the_om2_evaluation_games():
    from lanegame.experiments.m45b import engine as E
    from lanegame.experiments.rl.m5 import games as G
    from lanegame.experiments.rl.om2 import common as O2
    from lanegame.experiments.rl.om2 import evaluate as OE
    cfg = O2.setup()
    types = O2.belief_types()
    net = O2.om_net(OM2_T0)
    for opp in ("g_bully", "mix_bully_farm"):
        g1 = OE.play(cfg, ("arm", net, "treatment"), ("opp", opp), 4242, 60, types, E.lane_cache(cfg))
        g2 = G.play(("net", "om2:treatment:0"), ("opp", opp), 4242, 60, panel=False)
        assert np.array_equal(g1["r"].astype(np.float32), g2["r"])
        assert np.allclose(g1["trace"][0], g2["trace"][0], atol=1e-5)


@need_om2
def test_om_policy_decide_is_pure_and_observe_turn_moves_the_input():
    from lanegame.experiments.m45b import states as S
    from lanegame.experiments.m45b.strategies import get
    from lanegame.experiments.rl.m5 import players as PL
    cfg = C.arm_cfg("sp")
    pol = PL.crn_policy("om2:treatment:0", 0, 3, 0, 10)
    assert isinstance(pol, PL.OMPolicy) and pol.has_belief
    lane, m4, k = S.s0(cfg)
    x0 = pol.input(lane, m4, k)[1]
    for i in range(4):
        pol.decide(cfg, lane, m4, 0, i)
    assert pol.belief.updates == 0 and np.array_equal(pol.input(lane, m4, k)[1], x0)
    pol.observe_turn(cfg, lane, m4, k, (int(pol.decide(cfg, lane, m4, 0, 0)[0]),
                                        int(get("g_bully").decide(cfg, lane, m4, 1, 0)[0])))
    x1 = pol.input(lane, m4, k)[1]
    assert np.array_equal(x0[:90], x1[:90]) and not np.array_equal(x0[90:], x1[90:])
    assert abs(float(x1[103:].sum()) - 1.0) < 1e-5


def test_belief_modes():
    from lanegame.experiments.rl.m5 import players as PL
    from lanegame.opponent import Belief, prior_features
    b = Belief(13, 0.97)
    b.update(np.log(np.linspace(0.1, 1.0, 13)))
    T = np.tile(np.eye(13, dtype=np.float32)[2], (5, 1))
    assert np.array_equal(PL._features_of("prior", b, 3), prior_features(13))
    assert np.array_equal(PL._features_of(("table", T), b, 99), T[-1])
    assert np.allclose(PL._features_of("live", b, 0), b.features())


# ---------------------------------------------------------------------------------------------- cells and the battery
CELL_REF = os.path.join("results", "rl", "checkpoints", "m46c", "eval", "battery_ext", "baseline", "bat_baseline_s0",
                        "0.json.gz")


@need_start
@pytest.mark.skipif(not os.path.exists(CELL_REF), reason="needs the git-ignored M4.6c weaker-side records")
def test_cells_paired_job_reproduces_an_m46c_record():
    from lanegame.experiments.rl.m5.readiness import reproduce_weak
    out = reproduce_weak(K=1)
    assert out and all(v["exact"] for v in out.values())


def test_family_jobs_have_unique_paths_and_seeds():
    from lanegame.experiments.rl.m5 import evaluate as EV
    js = EV.family_jobs() + EV.monitor_jobs()
    assert len({j["path"] for j in js}) == len(js)
    fam = [j for j in EV.family_jobs()]
    assert len({(j["root"], j["cell"]) for j in fam}) == len(fam)
    f = EV.families(0)
    assert len(f["S_in"]) == 6 and len(f["C_cross"]) == 10 and ("net", "om:0") not in f["C_cross"]
    assert all(o[1] != "base:0" for o in f["C_hist"])        # never the seed's own start as a competent opponent


# ---------------------------------------------------------------------------------------------- statistics
def _st(per):
    return AN.interval(dict(enumerate(per)))


def test_classifications():
    assert AN.classify_gain(_st([2.0, 2.4, 1.8, 2.2, 2.1, 1.9])) == "demonstrated"
    assert AN.classify_gain(_st([0.10, 0.12, 0.08, 0.11, 0.09, 0.10])) == "detected, below materiality"
    assert AN.classify_gain(_st([-0.05, 0.05, 0.0, 0.02, -0.03, 0.01])) == "no material gain"
    assert AN.classify_ni(_st([0.0, 0.05, -0.05, 0.02, -0.02, 0.01])) == "preserved"
    assert AN.classify_ni(_st([-0.10, -0.12, -0.08, -0.11, -0.09, -0.10])) == "preserved (a small loss detected)"
    assert AN.classify_ni(_st([-0.6, -0.5, -0.7, -0.55, -0.65, -0.6])) == "material loss"
    assert AN.classify_ni(_st([-0.8, 0.6, -0.3, 0.4, -0.6, 0.5])) == "inconclusive"
    assert AN.classify_exploit(_st([0.5, 0.6, 0.4, 0.55, 0.45, 0.5])) == "increase demonstrated"
    assert AN.classify_exploit(_st([0.0, 0.05, -0.05, 0.02, -0.02, 0.01])) == "no material difference"
    assert AN.headline("demonstrated", "preserved", "preserved")["outcome"] == 1
    assert AN.headline("demonstrated", "material loss", "preserved")["outcome"] == 2
    assert AN.headline("demonstrated", "inconclusive", "preserved")["outcome"] == 3
    assert AN.headline("no material gain", "inconclusive", "preserved")["outcome"] == 4
    assert AN.headline("no material gain", "material loss", "preserved")["outcome"] == 5


def test_holm_and_variance_split_and_resolution():
    assert AN.holm({"a": 0.001, "b": 0.02, "c": 0.04}) == {"a": True, "b": True, "c": True}
    assert AN.holm({"a": 0.001, "b": 0.03, "c": 0.04}) == {"a": True, "b": False, "c": False}
    v = AN.variance_split({0: 1.0, 1: 2.0, 2: 3.0}, {0: 0.1, 1: 0.1, 2: 0.1})
    assert abs(v["between_seed_sd"] - 1.0) < 1e-12 and v["tau"] < 1.0 and v["game_share_of_variance"] < 0.02
    r = AN.ni_resolution(0.0, 0.05, 6, sims=4000)
    assert r["preserved"] > 0.99
    r = AN.ni_resolution(-1.0, 0.05, 6, sims=4000)
    assert r["material_loss"] > 0.99
    assert abs(sum(v for k, v in AN.ni_resolution(0.0, 0.3, 6, sims=4000).items() if k != "mean_half_width") - 1) < 1e-9


def test_secondary_endpoints_are_pending_without_records():
    """Amendment 1: missing records never shrink the family; without records nothing is claimed and Holm is pending."""
    out = AN.secondary({})
    assert not out["holm"]["final"] and len(out["holm"]["pending"]) == len(AN.SECONDARY_HYPOTHESES)
    assert all(v["claim"] == "not available" for v in out["endpoints"].values())
    st = _st([0.3, 0.2, 0.4, 0.25, 0.35, 0.3])
    assert AN.t_pvalue_one_sided(st, 0.0, "greater") < 0.001 < AN.t_pvalue_one_sided(st, 0.0, "less")


def test_replication_rule():
    r = AN.replication({0: 0.1, 1: 0.15, 2: 0.3, 3: -0.2, 4: 0.0, 5: 0.19})
    assert r["n_holds"] == 5 and r["replicates"]
    assert not AN.replication({0: 0.3, 1: 0.25, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0})["replicates"]


# ---------------------------------------------------------------------------------------------- training bookkeeping
def test_shares_and_prefix_check(tmp_path):
    import json
    from lanegame.experiments.rl.m5 import train as T
    h = [{"agent_transitions": 3000, "rollout_transitions_historical": 1000, "games_current": 2, "games_historical": 2,
          "rollout_env_turns": 2048},
         {"agent_transitions": 6100, "rollout_transitions_historical": 1100, "games_current": 1, "games_historical": 1,
          "rollout_env_turns": 2048}]
    s = T.shares(h)
    assert s["agent_transitions"] == 6100 and abs(s["learner_transition_share_fixed_opponent"] - 2100 / 6100) < 1e-12
    assert s["game_share_fixed_opponent"] == 0.5
    for name, hist in (("a", h), ("b", h), ("c", [h[0], {**h[1], "entropy": 9.0}])):
        d = tmp_path / name
        d.mkdir()
        (d / "history.jsonl").write_text("\n".join(json.dumps({**e, "entropy": e.get("entropy", 1.0)}) for e in hist))
    assert T.prefix_check(str(tmp_path / "a"), str(tmp_path / "b"))["identical"]
    assert not T.prefix_check(str(tmp_path / "a"), str(tmp_path / "c"))["identical"]


def test_schedule_wall():
    from lanegame.experiments.rl.m5.bench import schedule_wall
    assert schedule_wall([6, 6, 6, 3, 3, 3], 3) == 9
    assert schedule_wall([5] * 12, 6) == 10
