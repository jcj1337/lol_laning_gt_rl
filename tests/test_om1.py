"""OM1: opponent types' likelihoods, the Bayesian belief, the switcher and the selection rule.
Checkpoint-dependent tests skip when the git-ignored files are absent."""

import os

import numpy as np
import pytest

from lanegame import GameConfig
from lanegame.config import B_HOLD, B_TRADE, CHAMP_ABSENT, CHAMP_WARD, PRESSURE, SH, SP, champ_act
from lanegame.experiments.m45b import states as S
from lanegame.experiments.rl.om1 import common as C
from lanegame.experiments.rl.om1 import library as LB
from lanegame.experiments.rl.om1 import stages as ST

CFG = GameConfig.m45_calibrated()
HAVE_NETS = all(os.path.exists(p) for p in C.LEARNED.values()) and os.path.exists(C.pi_path(0))


def test_coarse_signal_hides_only_the_wave_choice():
    assert LB.coarse(champ_act(SH, B_HOLD)) == LB.coarse(champ_act(SP, B_HOLD))
    assert LB.coarse(champ_act(SH, B_HOLD)) != LB.coarse(champ_act(SH, B_TRADE))
    assert LB.coarse(champ_act(SH, B_HOLD, PRESSURE)) != LB.coarse(champ_act(SH, B_HOLD))
    assert LB.coarse(CHAMP_ABSENT) == LB.COARSE_ABSENT and LB.coarse(CHAMP_WARD) == LB.COARSE_WARD


def test_scripted_likelihood_trembles_around_the_scripted_action():
    lane, m4, k = S.s0(CFG)
    t = LB.ScriptType("g_bully", eps=0.1)
    legal, p = t.probs(CFG, lane, m4, 0, k)
    a = t.strategy.decide(CFG, lane, m4, 0, 0)[0]
    assert p.sum() == pytest.approx(1.0)
    assert p[legal.index(a)] == pytest.approx(0.9 + 0.1 / len(legal))
    assert np.all(p > 0)


class _Fixed:
    """A fake type with a fixed distribution over the legal codes (for the Bayes check)."""
    kind = "net"

    def __init__(self, weights):
        self.w = weights

    def probs(self, cfg, lane, m4, s, k):
        from lanegame.champions import champion_legal_actions
        legal = tuple(champion_legal_actions(lane, m4, s, cfg))
        p = np.array([self.w.get(c, 1.0) for c in legal], float)
        return legal, p / p.sum()


def test_belief_is_exact_bayes_and_ignores_absent_turns():
    lane, m4, k = S.s0(CFG)
    from lanegame.champions import champion_legal_actions
    legal = champion_legal_actions(lane, m4, 1, CFG)
    c0, c1 = legal[0], legal[1]
    t1, t2 = _Fixed({c0: 3.0}), _Fixed({c1: 3.0})
    b = LB.Belief([t1, t2])
    b.update(CFG, lane, m4, 1, k, c0)
    b.update(CFG, lane, m4, 1, k, c0)
    _, p1 = t1.probs(CFG, lane, m4, 1, k)
    _, p2 = t2.probs(CFG, lane, m4, 1, k)
    l1, l2 = p1[list(legal).index(c0)] ** 2, p2[list(legal).index(c0)] ** 2
    assert b.posterior()[0] == pytest.approx(l1 / (l1 + l2))
    away = m4.replace_side(1, absent=2, pos=-CFG.tower)
    before = b.full.copy()
    b.update(CFG, lane, away, 1, k, CHAMP_ABSENT)
    assert np.array_equal(before, b.full) and b.updates == 2


def test_selection_rule_requires_a_material_gain_over_pi():
    menu, types = ("pi", "x", "y"), ("t1", "t2")
    V = {("pi", "t1"): 1.0, ("x", "t1"): 1.2, ("y", "t1"): 0.0,          # x better by 0.2 < M: keep pi
         ("pi", "t2"): 0.0, ("x", "t2"): 0.1, ("y", "t2"): 1.0}          # y better by 1.0: switch to y
    d = ST.choose(V, menu, types, 0.25)
    assert d["rstar"] == {"t1": "pi", "t2": "y"}
    assert d["rfix"] == "pi"                                             # mean pi 0.5, x 0.65, y 0.5


def test_mixture_player_is_reproducible_by_stream():
    t = LB.make_type("mix_bully_safe")
    lane, m4, k = S.s0(CFG)
    a = [t.player(k, 7, 1, 30).decide(CFG, lane, m4, 1, i)[0] for i in range(30)]
    b = [t.player(k, 7, 1, 30).decide(CFG, lane, m4, 1, i)[0] for i in range(30)]
    assert a == b


@pytest.mark.skipif(not HAVE_NETS, reason="checkpoints absent")
def test_learned_likelihood_matches_the_sampling_distribution():
    pytest.importorskip("torch")
    cfg = C.setup()
    lane, m4, k = S.s0(cfg)
    t = LB.make_type("attacker")
    legal, p = t.probs(cfg, lane, m4, 1, k)
    l2, p2 = t.player(k, 3, 1, 10).probs(cfg, lane, m4, 1, 0)
    assert tuple(legal) == tuple(int(c) for c in l2) and np.allclose(p, p2) and p.sum() == pytest.approx(1.0)


@pytest.mark.skipif(not HAVE_NETS, reason="checkpoints absent")
def test_paired_self_play_is_zero_and_the_switcher_equals_pi_until_it_switches():
    pytest.importorskip("torch")
    from lanegame.experiments.rl.om1 import games as G
    cfg = C.setup()
    sd = C.game_seed(C.SMOKE_ROOT, 9, 1)
    a = G.play(cfg, ("type", "g_shove"), "g_shove", sd, 30, 0)["r"]
    b = G.play(cfg, ("type", "g_shove"), "g_shove", sd, 30, 1)["r"]
    assert abs(a.sum() + b.sum()) < 1e-9
    rstar = {t: "pi" for t in C.LIBRARY}
    rstar["g_farm"] = "g_bully"
    g = G.play(cfg, ("switch", 0, rstar, C.THETA), "g_farm", sd, 30, 1)
    p = G.play(cfg, ("type", "pi:0"), "g_farm", sd, 30, 1)["r"]
    f = g["switch"]["first_switch"]
    n = f if f >= 0 else 30
    assert np.array_equal(g["r"][:n], p[:n])
    never = {t: "pi" for t in C.LIBRARY}
    g2 = G.play(cfg, ("switch", 0, never, C.THETA), "g_farm", sd, 30, 1)
    assert np.array_equal(g2["r"], p) and g2["switch"]["first_switch"] == -1
