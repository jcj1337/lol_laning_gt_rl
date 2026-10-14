"""M4.4c experimental infrastructure (docs/m4/m4_4c_strategic_investigation.md): multi-turn strategic plans, paired
multi-turn counterfactuals, complete-action Recall comparisons, opponent protocols and styles, episode ledgers, wave
budget identities, seed roots and viewer traces. No checkpoint files: networks are the untrained initial networks of a
training seed (bit-identical to the start of the M4.4b runs)."""

import numpy as np
import pytest

from lanegame.champions_m4 import resolve_turn_m4
from lanegame.config import (B_HOLD, B_RECALL, B_SAFE, B_TRADE, CHAMP_ABSENT, W_HOLD, W_SHOVE, W_THIN, champ_act,
                             champ_behavior, champ_wave)
from lanegame.diagnostics.branch import FollowPolicy, ProbCache, observation, rollout, rollout_seed
from lanegame.env import LaneEnv
from lanegame.experiments.rl.m44 import common as C
from lanegame.experiments.rl.m44.actions import RECALL
from lanegame.experiments.rl.m44.records import holds_freeze, play_record
from lanegame.experiments.rl.m44c import engine as E
from lanegame.experiments.rl.m44c import opponents as O
from lanegame.experiments.rl.m44c import plans as P
from lanegame.experiments.rl.m44c.states import initialized


@pytest.fixture(scope="module")
def setup():
    cfg = C.env_cfg("aggro")
    x = C.agent_from_spec("initial:aggro:1", cfg)
    rec = play_record(cfg, (x, x), np.random.SeedSequence(81_990), 80, keep_snapshots=True)
    return cfg, x, rec


def _recall_state(cfg):
    """An initialized state where A (30% HP) stands at its -1 with Recall legal and B at the lane cell."""
    return initialized(cfg, "neutral", hp=(0.3, 1.0), pos=(-1, 0))


# ---------------------------------------------------------------------------------------------- plans
def test_a_plan_decision_that_is_not_legal_falls_back_to_the_policy_and_is_logged():
    cfg = C.env_cfg("aggro")
    x = C.agent_from_spec("initial:aggro:1", cfg)
    snap = initialized(cfg, "neutral")                       # A stands at the lane cell: Recall is not legal
    ob = observation(cfg, snap, 0)
    assert RECALL not in ob.legal
    assert not P.StrategicPlan("bad", P.const_rule(P.code(RECALL, "Recall")), max_turns=3).applicable(ob)
    plan = P.StrategicPlan("later", lambda o, t, m: P.HOLD_D if t == 0 else P.code(RECALL, "Recall"), max_turns=4)
    assert plan.applicable(ob)
    env = LaneEnv(snap.config(cfg))
    for m in range(4):
        r = rollout(env, snap, 0, plan, (x, x), rollout_seed(1, m), max_turns=6, record=True)
        assert plan.log[0][:2] == (0, "applied") and champ_behavior(r.infos[0]["actions"][0]) == B_HOLD
        for t, status, _ in plan.log[1:]:
            a = r.infos[t]["actions"][0]
            assert a in r.infos[t]["legal"][0]                 # never an illegal or invented action
            if status == "applied":
                assert a == RECALL
            elif status == "fallback":
                assert RECALL not in r.infos[t]["legal"][0]   # the fallback happens exactly when Recall is illegal
        assert plan.last["applied"] + plan.last["fallback"] + plan.last["absent"] <= 4


def test_a_plan_stops_when_its_situation_ends_and_never_acts_after_it():
    cfg = C.env_cfg("aggro")
    x = C.agent_from_spec("initial:aggro:2", cfg)
    snap = initialized(cfg, "freeze")
    assert holds_freeze(snap.lane, 0)
    from lanegame.experiments.rl.m44c.freeze import candidates
    plan = candidates("held", cfg, snap, 0)["Break"]
    env = LaneEnv(snap.config(cfg))
    r = rollout(env, snap, 0, plan, (x, x), rollout_seed(2, 0), max_turns=15, record=True)
    log = plan.last
    assert log["stop_reason"] in ("freeze ended", "window", "champion absent")
    if log["stop_reason"] == "freeze ended":
        t_end = log["stop_turn"]
        from lanegame.state import State
        assert not holds_freeze(State.from_list(r.infos[t_end]["state_before"]), 0)
        assert all(holds_freeze(State.from_list(r.infos[t]["state_before"]), 0) for t in range(t_end))
        for t in range(t_end):                               # while held, the forced wave was played
            assert champ_wave(r.infos[t]["actions"][0]) == W_SHOVE or r.infos[t]["actions"][0] == CHAMP_ABSENT


def test_forcing_several_components_keeps_the_policys_conditional_choice_of_the_rest(setup):
    cfg, x, rec = setup
    ob = observation(cfg, rec.snapshots[30], 1)
    p = np.asarray(x.distribution(ob), dtype=float)
    q = P.reshape(p, ob.legal, {"wave": W_THIN, "behavior": B_HOLD})
    legal = list(ob.legal)
    sl = [i for i, a in enumerate(legal) if a not in (RECALL, CHAMP_ABSENT) and champ_wave(a) == W_THIN
          and champ_behavior(a) == B_HOLD]
    assert np.isclose(q.sum(), 1.0) and set(np.flatnonzero(q > 0)) <= set(sl)
    assert np.allclose(q[sl], p[sl] / p[sl].sum())
    one = P.reshape(p, ob.legal, {"behavior": B_SAFE})
    from lanegame.experiments.rl.m44.interventions import intervene
    assert np.allclose(one, intervene(p, ob.legal, "behavior", B_SAFE))


# ---------------------------------------------------------------------------------------------- Recall
def test_recall_comparisons_use_complete_legal_codes():
    cfg = C.env_cfg("aggro")
    x = C.agent_from_spec("initial:aggro:1", cfg)
    snap = _recall_state(cfg)
    ob = observation(cfg, snap, 0)
    assert RECALL in ob.legal
    from lanegame.experiments.rl.m44c.recall import candidates
    c = candidates()
    assert all(v.applicable(ob) for k, v in c.items() if k != "policy")
    env = LaneEnv(snap.config(cfg))
    for name in ("Recall", "Safe", "Hold", "Safe>Recall", "Retreat"):
        for m in range(4):
            r = rollout(env, snap, 0, c[name], (x, x), rollout_seed(3, m), max_turns=3, record=True)
            a0 = r.infos[0]["actions"][0]
            assert a0 in r.infos[0]["legal"][0]
            if name == "Recall":
                assert a0 == RECALL
            elif name in ("Safe", "Safe>Recall"):
                assert a0 != RECALL and champ_behavior(a0) == B_SAFE
            elif name == "Hold":
                assert a0 != RECALL and champ_behavior(a0) == B_HOLD
    # the mechanics the Recall study rests on: an interrupted Recall is hit at 0.55, Safe evades to the tower cell
    rec_vs_trade = resolve_turn_m4(snap.lane, snap.champ, snap.k, RECALL, champ_act(W_HOLD, B_TRADE), cfg, u=None)
    safe_vs_trade = resolve_turn_m4(snap.lane, snap.champ, snap.k, champ_act(W_HOLD, B_SAFE),
                                    champ_act(W_HOLD, B_TRADE), cfg, u=None)
    assert rec_vs_trade.interrupted[0] and rec_vs_trade.connected[1] and rec_vs_trade.hit_p[1] == pytest.approx(0.55)
    assert safe_vs_trade.end[0] == -2 and not safe_vs_trade.connected[1]


def test_the_reach_geometry_of_an_interruption():
    from lanegame.experiments.rl.m44c.recall import reach_cell
    cfg = C.env_cfg("aggro")
    assert reach_cell(cfg, -2, 0) and reach_cell(cfg, -2, 2)        # only by ending in the tower cell (a dive)
    assert not reach_cell(cfg, -2, -1)                             # a dive from the enemy's -1 needs 3 cells
    assert reach_cell(cfg, -1, -2) and reach_cell(cfg, -1, 1)


# ---------------------------------------------------------------------------------------------- engine
def test_paired_candidates_share_chance_and_the_opponents_first_action(setup):
    cfg, x, rec = setup
    snap = rec.snapshots[40]
    cands = {"policy": FollowPolicy(), "policy2": FollowPolicy(),
             "Shove": P.one_turn("Shove", P.force({"wave": W_SHOVE}))}
    r = E.evaluate(cfg, snap, 0, cands, (x, x), 4, 81_991, key=(1,), window=25)
    assert all(r["pairing"].values())
    for k in r["mean"]["policy"]:                             # identical plans -> identical outcomes (CRN)
        a, b = r["mean"]["policy"][k], r["mean"]["policy2"][k]
        assert (np.isnan(a) and np.isnan(b)) or a == b
    assert r["diff"]["policy2"]["G20"][0] == 0.0


def test_opponent_protocols_are_reproducible_and_modal_is_the_most_probable_action(setup):
    cfg, x, rec = setup
    snap = rec.snapshots[50]
    for proto in ("policy", "modal", "breaker", "respect"):
        agents, plan = O.setup(proto, x, cfg, snap, 0)
        cands = {"policy": FollowPolicy(), "Thin": P.one_turn("Thin", P.force({"wave": W_THIN}))}
        r1 = E.evaluate(cfg, snap, 0, cands, agents, 3, 81_992, key=(2,), window=20, opponent_plan=plan)
        agents, plan = O.setup(proto, x, cfg, snap, 0)
        r2 = E.evaluate(cfg, snap, 0, cands, agents, 3, 81_992, key=(2,), window=20, opponent_plan=plan)
        assert r1["mean"] == r2["mean"] or all(
            (np.isnan(r1["mean"][c][k]) and np.isnan(r2["mean"][c][k])) or r1["mean"][c][k] == r2["mean"][c][k]
            for c in r1["mean"] for k in r1["mean"][c])
    plan = O.modal_plan(x, cfg, snap, 1)
    ob = observation(cfg, snap, 1)
    p = np.asarray(x.distribution(ob))
    assert plan.action == ob.legal[int(np.argmax(p))]


def test_evaluation_never_changes_the_network_weights(setup):
    from lanegame.rl import weights_digest
    cfg, x, rec = setup
    d0 = weights_digest(x.net.state_dict())
    E.evaluate(cfg, rec.snapshots[25], 1, {"policy": FollowPolicy(), "Hold": P.one_turn("Hold", P.HOLD_D)}, (x, x), 2,
               81_993, window=15)
    assert weights_digest(x.net.state_dict()) == d0


def test_the_ledger_summary_is_consistent_with_the_zero_sum_reward(setup):
    cfg, x, rec = setup
    snap = rec.snapshots[35]
    env = LaneEnv(snap.config(cfg))
    r = rollout(env, snap, 1, FollowPolicy(), (x, x), rollout_seed(4, 0), max_turns=40, record=True)
    s = E.summarize(r.infos, 1, cfg, ep_h=(10, 40), ret_h=(10, 40))
    rewards_b = [-inf["reward"] for inf in r.infos]
    assert s["gold_f40"] - s["gold_o40"] == pytest.approx(sum(rewards_b), abs=1e-9)
    assert s["G40"] == pytest.approx(sum(cfg.gamma ** t * v for t, v in enumerate(rewards_b)), abs=1e-9)
    assert s["G40"] == pytest.approx(r.ret, abs=1e-9)
    for inf in r.infos:
        assert inf["earned_units"][0] - inf["earned_units"][1] == inf["reward_units"]


def test_across_state_contrasts_use_the_requested_reference():
    st = {"result": {"candidates": ["policy", "Recall", "Safe"], "ref": "policy",
                     "mean": {"policy": {"G300": 1.0}, "Recall": {"G300": 0.25}, "Safe": {"G300": 2.0}},
                     "diff": {"Recall": {"G300": [-0.75, 0.1]}, "Safe": {"G300": [1.0, 0.1]}}}}
    assert E.state_values([st], "Safe", "G300", "policy") == [1.0]
    assert E.state_values([st], "policy", "G300", "Recall") == [0.75]     # the stored reference against another one
    assert E.state_values([st], "Safe", "G300", "Recall") == [1.75]
    assert E.state_values([st], "Recall", "G300", "Recall") == [0.0]
    assert E.state_values([st], "Safe", "G300", None) == [2.0]


def test_the_freeze_end_classifier_reports_one_known_type(setup):
    cfg = C.env_cfg("econ")
    x = C.agent_from_spec("initial:econ:0", cfg)
    snap = initialized(cfg, "freeze")
    env = LaneEnv(snap.config(cfg))
    for m in range(3):
        r = rollout(env, snap, 0, FollowPolicy(), (x, x), rollout_seed(5, m), max_turns=40, record=True)
        assert E.freeze_end(r.infos, 0, cfg) in E.FREEZE_ENDS


# ---------------------------------------------------------------------------------------------- waves
def test_thin_and_slow_push_reduce_to_last_hit_when_the_own_wave_is_not_weaker():
    from lanegame.experiments.rl.m44c.waves import budgets, one_turn
    cfg = C.env_cfg("aggro")
    snap = initialized(cfg, "neutral")                       # symmetric: equal window strengths
    b = budgets(cfg, snap.lane, snap.champ, snap.k, 0)
    assert b["LH"] == 1 and b["TN"] == b["LH"] and b["SL"] == b["LH"] + 1 and b["SH"] == b["C"]
    tr = one_turn(cfg, snap.lane, snap.champ, snap.k, 0, champ_act(W_HOLD, B_HOLD), champ_act(W_HOLD, B_HOLD))
    assert tr["LH"]["lane"] == tr["TN"]["lane"]
    m = __import__("lanegame.experiments.rl.m44c.waves", fromlist=["matrix"]).matrix(cfg, 3, turns=8)
    assert m["TN|LH"]["gold_diff8"] == 0.0 and m["LH|LH"]["gold_diff8"] == 0.0


# ---------------------------------------------------------------------------------------------- seeds, traces, CLI
def test_m44c_seed_roots_are_disjoint_from_training_and_earlier_evaluation():
    from lanegame.experiments.rl.m44c.common import ALL_ROOTS, TRACE_SEED
    m44_roots = {C.BEHAVIOR_SEED, C.SCENARIO_SEED, C.CF_STATE_SEED, C.CF_ROLL_SEED, C.REF_SEED, C.TRACE_SEED,
                 C.XPLAY_SEED, C.SMOKE_EVAL_SEED}
    assert all(81_000 <= r < 82_000 for r in ALL_ROOTS) and 81_000 <= TRACE_SEED < 82_000
    assert not (set(ALL_ROOTS) & m44_roots) and len(set(ALL_ROOTS)) == len(ALL_ROOTS)
    assert min(ALL_ROOTS) > max(C.SEEDS) + 1                  # training uses torch / numpy seeds 0-3


def test_a_counterfactual_branch_replays_exactly_in_the_viewer(tmp_path):
    from lanegame.experiments.rl.m44.trace import verify
    from lanegame.experiments.rl.m44c.traces import SCHEMA, play_branch
    from lanegame.io import dump_json
    cfg = C.env_cfg("aggro")
    x = C.agent_from_spec("initial:aggro:1", cfg)
    rec = play_record(cfg, (x, x), 81_995, 30, viewer_seed=True)
    prefix = [[int(rec.data["a"][t, 0]), int(rec.data["a"][t, 1])] for t in range(20)]
    for plan in (FollowPolicy(), P.StrategicPlan("Thin", P.const_rule(P.force({"wave": W_THIN})), max_turns=5)):
        acts, infos = play_branch(cfg, x, 81_995, prefix, None, 0, plan, 10, stream_seed=7)
        path = str(tmp_path / f"b_{plan.name}.json")
        dump_json({"schema": SCHEMA, "version": 1, "config": "m4_four_actions_econ_aggro", "scenario": "s0",
                   "gap": "equal", "seed": 81_995, "actions": prefix + acts,
                   "payoff_A": float(sum(i["reward"] for i in infos))}, path)
        v = verify(path)
        assert v["actions_equal"] and v["payoff_equal"] and v["turns"] == 30
    # the prefix of a FollowPolicy branch is the recorded game itself
    assert [i["actions"] for i in infos[:20]] == [tuple(a) for a in prefix]


def test_finished_jobs_survive_a_failure_and_are_reused(tmp_path, monkeypatch):
    from lanegame.experiments.rl.m44c import common as MC
    monkeypatch.setattr(MC, "PARTIAL", str(tmp_path))
    calls, crash = [], [True]

    def fn(j):
        calls.append(j)
        if j == ("b", 2) and crash[0]:
            raise RuntimeError("crash after one job")
        return j[1] * 10
    jobs = [("a", 1), ("b", 2), ("c", 3)]
    with pytest.raises(RuntimeError):
        MC.run_resumable(fn, jobs, 1, "t")
    assert calls == [("a", 1), ("b", 2)]                      # job a finished and was written before the crash
    calls.clear()
    crash[0] = False
    assert MC.run_resumable(fn, jobs, 1, "t") == [10, 20, 30]
    assert calls == [("b", 2), ("c", 3)]                      # job a was reused, not re-run
    MC.clear_partial("t")
    assert not (tmp_path / "t").exists()


def test_the_cli_offers_every_m44c_stage():
    from lanegame.experiments.rl.m44c.__main__ import STAGES
    assert {"recall", "freeze", "freeze_init", "zoning", "zoning_nat", "zoning_mech", "vision", "waves_mech", "waves",
            "traces", "report"} <= set(STAGES)
