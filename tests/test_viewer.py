"""M4.2e-D viewer: the recordings, replays, coordinates, summaries and tick traces it shows agree with the engine."""

import random

import pytest

from lanegame.champions import CHAMP_ABSENT
from lanegame.champions_m4 import context
from lanegame.config import B_HOLD, W_CLEAR, W_HOLD, W_THIN, champ_act
from lanegame.env import LaneEnv
from lanegame.experiments.m4 import zoning_natural as zn
from lanegame.mechanics import step_turn
from lanegame.viewer import policies as pl
from lanegame.viewer import session as ss

A, B = 0, 1


def matchup(name, seed=3):
    sc, gap, pa, pb, n, _ = pl.MATCHUPS[name]
    return ss.play(ss.Setup(scenario=sc, gap=gap, seed=seed), policies=(pl.POLICIES[pa], pl.POLICIES[pb]), turns=n)


ALL = list(pl.MATCHUPS)


def strip(info):
    return {k: v for k, v in info.items() if k != "lane_trace"}


@pytest.mark.parametrize("name", ALL)
def test_recording_matches_plain_engine(name):
    """A recording equals the trajectory of a plain LaneEnv (no instrumentation) fed the same start, seed, actions."""
    rec = matchup(name)
    lane, m4, k0 = ss.start_state(rec.setup)
    env = LaneEnv(rec.cfg)
    env.reset(seed=rec.setup.seed, root=lane, champ=m4, k0=k0)
    for j, (a, b) in enumerate(rec.actions):
        assert (env.state, env.champ) == (rec.lanes[j], rec.champs[j])
        assert (env.legal(A), env.legal(B)) == rec.legal[j]
        r = env.step(a, b)
        assert strip(r.info) == strip(rec.turns[j].info)
    assert (env.state, env.champ) == (rec.lanes[-1], rec.champs[-1])


def test_replay_is_deterministic_and_branching_keeps_the_prefix():
    rec = matchup("7. Power advantage")
    again = ss.play(rec.setup, prefix=rec.actions)
    assert [strip(t.info) for t in again.turns] == [strip(t.info) for t in rec.turns]
    # scrubbing reads in any order and changes nothing
    order = list(range(rec.n + 1))
    random.Random(0).shuffle(order)
    before = [strip(t.info) for t in rec.turns]
    for i in order:
        ss.summary(rec, i, A), ss.lane_objects(rec.lanes[i], rec.cfg)
    assert [strip(t.info) for t in rec.turns] == before
    # branch at turn 15: the first 15 turns replay exactly; a different action then continues from the same RNG
    cut = 15
    br = ss.play(rec.setup, prefix=rec.actions[:cut], policies=(pl.POLICIES["farm"], pl.POLICIES["farm"]), turns=5)
    assert [strip(t.info) for t in br.turns[:cut]] == before[:cut]
    assert [t.info["chance"] for t in br.turns] == [t.info["chance"] for t in rec.turns[:cut + 5]]


def test_tick_recording_changes_nothing():
    rec = matchup("3. Wave clear")
    lane, m4, k0 = ss.start_state(rec.setup)
    env = LaneEnv(rec.cfg, record_ticks=False)
    env.reset(seed=rec.setup.seed, root=lane, champ=m4, k0=k0)
    for j, (a, b) in enumerate(rec.actions):
        info = env.step(a, b).info
        assert "lane_trace" not in info
        assert info == strip(rec.turns[j].info)


@pytest.mark.parametrize("name", ALL)
def test_tick_trace_is_the_authoritative_lane_step(name):
    """The recorded trace is the trace of the lane step that produced the end state (not of the resolver's probe):
    re-running mechanics.step_turn with the authoritative arguments reproduces it, and its last tick agrees with the
    end-of-turn lane (crashed masses always; packets on turns without a spawn)."""
    rec = matchup(name)
    cfg = rec.cfg
    ctx = context(cfg)
    for j, t in enumerate(rec.turns):
        info, lane = t.info, rec.lanes[j]
        tr = info["lane_trace"]
        assert len(tr) == cfg.ticks_per_turn
        budget = tuple(ctx.wave_budget(info["waves"][s], info["levels_before"][s], rec.champs[j].side(s).items)
                       for s in (A, B))
        res = step_turn(lane, info["k"], *info["lane_codes"], cfg, record=True,
                        champion_reach=tuple(info["champion_reach"]), champion_budget=budget)
        assert res.state == rec.lanes[j + 1]
        assert res.trace == tr
        end = rec.lanes[j + 1]
        assert tr[-1]["siege"] == [end.asg, end.bsg]
        if info["events"]["spawned"] == [0, 0]:
            for s in (A, B):
                fx, fh, rears = end.side(s)[:3]
                want = sorted(([[fx, fh]] if fh > 0 else []) + [[x, cfg.wave_hp] for x in rears])
                assert sorted(tr[-1][ss.SIDES[s]]) == want


def test_global_coordinates_mirror_b():
    rec = matchup("2. Slow push")
    cfg = rec.cfg
    for i in range(rec.n + 1):
        lane, m4 = rec.lanes[i], rec.champs[i]
        o, om = ss.lane_objects(lane, cfg), ss.lane_objects(lane.mirror(), cfg)
        # A's own coordinates are global; B's are negated
        assert [(x, h) for x, h, _ in o["A"]["packets"]] == \
            ([(lane.ax, lane.ah)] if lane.ah > 0 else []) + [(x, cfg.wave_hp) for x in lane.ar]
        assert [x for x, _, _ in o["B"]["packets"]] == ([-lane.bx] if lane.bh > 0 else []) + [-x for x in lane.br]
        assert o["A"]["siege"] == (cfg.tower, lane.asg) and o["B"]["siege"] == (-cfg.tower, lane.bsg)
        # swapping the players reflects the picture
        assert [(-x, h, f) for x, h, f in om["B"]["packets"]] == o["A"]["packets"]
        assert ss.champion_x(m4, A) == m4.a.pos and ss.champion_x(m4, B) == -m4.b.pos
        if ss.contact_x(lane) is not None:
            assert ss.contact_x(lane.mirror()) == -ss.contact_x(lane)


def test_summaries_agree_with_engine_state_and_ledger():
    rec = matchup("6. Crash and Recall")                 # s0-free scenario with an equal gap: no initial gold
    cfg, sc = rec.cfg, rec.cfg.gold_scale
    t = context(cfg).t
    earned = [0, 0]
    clamped = [0, 0]
    for i in range(rec.n + 1):
        for s in (A, B):
            d, me = ss.summary(rec, i, s), rec.champs[i].side(s)
            assert d["hp"] == me.hp and d["xp"] == me.xp and d["absent"] == me.absent
            assert d["level"] == t.level_by_xp[min(me.xp, t.xp_cap)] and d["hp_max"] == t.hp_max[d["level"]]
            assert d["gold_spendable"] == me.gold / sc and d["invested"] == me.items / sc
            assert d["gold_earned"] == pytest.approx(earned[s] / sc)
            # income = spendable + invested (+ anything clamped), the scenario starts with neither
            assert me.gold + me.items + clamped[s] == earned[s]
        if i < rec.n:
            info = rec.turns[i].info
            for s in (A, B):
                assert info["earned_units"][s] == info["farm_units"][s] + info["plate_units"][s] + info["bounty_units"][s]
                earned[s] += info["earned_units"][s]
                clamped[s] += info["gold_unspendable"][s]
    tl = ss.timeline(rec)
    assert tl["advantage"][-1] == pytest.approx(sum(t.info["reward"] for t in rec.turns))


def test_legal_actions_respected_and_illegal_manual_action_rejected():
    for name in ALL:
        rec = matchup(name)
        for j, (a, b) in enumerate(rec.actions):
            assert a in rec.legal[j][0] and b in rec.legal[j][1]
    rec = ss.play(ss.Setup(), prefix=[(champ_act(W_HOLD, B_HOLD), champ_act(W_HOLD, B_HOLD))])
    assert rec.n == 1
    with pytest.raises(ValueError, match="illegal action"):
        ss.play(ss.Setup(), prefix=[(CHAMP_ABSENT, champ_act(W_HOLD, B_HOLD))])


def test_policies_match_zoning_natural_and_explain_fallbacks():
    """decide() plays exactly zoning_natural.Policy.act (same rules) and every note is non-empty."""
    rec = matchup("7. Power advantage")
    for name, pol in pl.POLICIES.items():
        if isinstance(pol, pl.CrashRecallPolicy):
            continue
        ref = zn.Policy(name, pol.behavior, pol.wave, pol.pressure)
        for i in range(rec.n + 1):
            for s in (A, B):
                code, note = pol.decide(rec.cfg, rec.lanes[i], rec.champs[i], s)
                assert code == ref.act(rec.cfg, rec.lanes[i], rec.champs[i], s) and note


def test_policies_are_state_dependent():
    def labels(name, s):
        return [t.info["action_labels"][s] for t in matchup(name).turns]
    push = labels("2. Slow push", A)
    assert {"TN-H", "HO-H"} <= set(push)                      # Thins, then Holds
    crash = matchup("6. Crash and Recall")
    waves = {t.info["waves"][A] for t in crash.turns}
    assert {W_THIN, W_CLEAR} <= waves and any(t.info["recalled"][A] for t in crash.turns)
    j = next(j for j, t in enumerate(crash.turns) if t.info["recalled"][A])
    assert any(crash.lanes[x].asg > 0 for x in range(max(0, j - 3), j + 1))   # Recall follows the own crash
    freeze = labels("4. Freeze", A)
    assert len(set(freeze)) > 1
    assert labels("1. Passive farming", A) != push


def test_frame_views_follow_the_engine():
    """Every frame's packets come from the lane states or the authoritative tick trace, in global coordinates, and
    the contact and movement arrows are consistent with them."""
    rec = matchup("6. Crash and Recall")
    cfg = rec.cfg
    for i in range(rec.n + 1):
        nf = ss.n_frames(rec, i)
        assert nf == (1 if i == rec.n else cfg.ticks_per_turn + 2)
        for f in range(nf):
            v = ss.frame_view(rec, i, f)
            if f == 0 or i == rec.n:
                want, j = rec.lanes[i], i
            elif f <= cfg.ticks_per_turn:
                tk, j = rec.turns[i].info["lane_trace"][f - 1], i + 1
                want = None
            else:
                want, j = rec.lanes[i + 1], i + 1
            if want is not None:
                assert v["packets"] == {n: ss.lane_objects(want, cfg)[n]["packets"] for n in ss.SIDES}
                assert v["siege"]["A"] == (cfg.tower, want.asg) and v["siege"]["B"] == (-cfg.tower, want.bsg)
                assert v["contact"] == ss.contact_x(want)
            else:
                assert sorted(x for x, _, _ in v["packets"]["A"]) == sorted(x for x, _ in tk["A"])
                assert sorted(-x for x, _, _ in v["packets"]["B"]) == sorted(x for x, _ in tk["B"])
                assert v["siege"] == {"A": (cfg.tower, tk["siege"][0]), "B": (-cfg.tower, tk["siege"][1])}
            assert v["state_index"] == j
            for n in ss.SIDES:
                assert v["champions"][n] == ss.summary(rec, j, ss.SIDES.index(n))
                mv = v["moves"][n]
                if mv is not None:
                    assert mv[1] == ss.fronts({m: {"packets": v["packets"][m]} for m in ss.SIDES})[n]
            assert v["towers"]["A"]["damage"] == rec.champs[j].b.tower_dmg
            assert v["towers"]["B"]["damage"] == rec.champs[j].a.tower_dmg


def test_render_builds_every_frame():
    pytest.importorskip("plotly")
    from lanegame.viewer import render
    rec = matchup("6. Crash and Recall")
    for i in range(rec.n + 1):
        for f in range(ss.n_frames(rec, i)):
            v = ss.frame_view(rec, i, f)
            fig = render.lane_figure(v, rec.cfg)
            hover = next(t for t in fig.data if t.name == "packets")
            # one hover point per tower, per leading packet, per reinforcement cell and per crashed mass
            cells = {n: len({x for x, _, fr in v["packets"][n] if not fr}) + sum(1 for *_, fr in v["packets"][n] if fr)
                     + (1 if v["siege"][n][1] else 0) for n in ss.SIDES}
            assert len(hover.x) == 2 + cells["A"] + cells["B"]
            champs = next(t for t in fig.data if t.name == "champions")
            for n, s in (("A", 0), ("B", 1)):
                d = v["champions"][n]
                if not d["absent"]:
                    assert abs(champs.x[s] - d["pos_global"]) <= 0.25
    render.timeline_figure(ss.timeline(rec), 0)
