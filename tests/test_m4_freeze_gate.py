"""S3b required Freeze-gate cases (plan §7 S3b: F+, SHx, F0) on fixed scenarios with fixed chance vectors.

Fixed scenarios: the design §6.4 parked starts (the prototype's harness) and the deterministic reachable scenario set of
lanegame.experiments.m4.freeze_gate (witnessed Freeze-legal states). Fixed chance vectors: no chance event (every
uniform 1.0), as in the design prototype. Policies: the S3b gate defenders and attackers (lanegame.env.agents_m4).

Two of the plan's required statements do not hold on the implemented environment. They are kept as STRICT expected
failures (they would flag if they started to hold) and are reported in docs/m4/m4_2/m4_implementation_report.md:
  * SHx "the freeze breaks within 6 turns" against a presser with a 3-5 HP front: measured 6-9 turns (the reach-loss
    of a Shove + Hold breaker on the turns its probe predicts a crash, see tests/test_m4_integration.py);
  * F0 "a 6-8 HP front against the stacker: Freeze's return is below Slow Push's" on the parked starts: Freeze is
    better there (the freezer keeps the stacker farming under its tower and trades it with the home advantage). On the
    reachable 6-8 HP scenarios the sign is mixed, and the existence of scenarios where Freeze is worse is asserted.
"""

import numpy as np
import pytest

from lanegame import GameConfig
from lanegame.champions_m4 import M4State
from lanegame.env import LaneEnv
from lanegame.env.agents_m4 import gate_attackers, gate_defenders
from lanegame.scenarios_m4 import DESIGN_HOLD_TABLE, parked_state
from lanegame.config import SP
from lanegame.state import State

M4 = GameConfig.m4()
H = 120


class _NoEvents:
    def random(self, n=None):
        return np.ones(n)


def play(lane, m4, phase, dn, an, turns=H):
    """Deterministic gate rollout: (discounted return to A, B's first crash, B's tower damage dealt)."""
    env = LaneEnv(M4, max_turns=turns)
    obs = env.reset(seed=0, root=lane, k0=phase, champ=m4)
    env._rng = _NoEvents()
    d, a = gate_defenders()[dn], gate_attackers()[an]
    if hasattr(a, "reset"):
        a.reset()
    disc, w, first, tau = 0.0, 1.0, None, 0
    for t in range(turns):
        r = env.step(d.act(obs[0]), a.act(obs[1]))
        if hasattr(a, "notify"):
            a.notify(r.info, 1)
        disc += w * r.rewards[0]
        w *= M4.gamma
        if r.info["crash_hp"][1] > 0 and first is None:
            first = t
        tau += r.info["tower_damage_dealt"][1]
        obs = r.obs
    return disc, first, tau


def parked(e, ph, dn, an):
    lane, m4 = parked_state(e, M4)
    return play(lane, m4, ph, dn, an)


# ---------------------------------------------------------------------- F+
@pytest.mark.parametrize("e", [5, 6])
def test_f_plus_freeze_denies_the_creep_crash_and_pays(e):
    fz_crash, sp_crash = DESIGN_HOLD_TABLE[(e, SP)]
    checked = 0
    for ph in range(4):
        f, s = parked(e, ph, "freeze", "park_hold"), parked(e, ph, "slowpush", "park_hold")
        assert f[0] >= s[0] - 1e-9
        if sp_crash[ph] is not None:                      # Slow Push lets the creeping front crash ...
            assert f[0] > s[0]                            # ... and Freeze's discounted return is higher
            design_later = fz_crash[ph] is None or fz_crash[ph] > sp_crash[ph]
            if design_later:                              # where design §6.4 reports a denied / later crash
                assert f[1] is None or f[1] > s[1]        # (6 HP phase 1 is the table's own earlier-crash cell)
            checked += 1
    assert checked >= 1


# ---------------------------------------------------------------------- SHx
@pytest.mark.parametrize("e", [3, 4, 5])
def test_sh_x_presser_breaks_the_freeze_and_deals_more_tower_damage(e):
    for ph in range(4):
        f, s = parked(e, ph, "freeze", "presser"), parked(e, ph, "slowpush", "presser")
        assert f[1] is not None                           # the freeze is broken
        assert f[2] > s[2]                                # B's tower damage under Freeze exceeds Slow Push's


@pytest.mark.xfail(strict=True, reason="design finding: the presser breaks a 3-5 HP freeze in 6-9 turns, not <= 6")
def test_sh_x_freeze_breaks_within_six_turns():
    for e in (3, 4, 5):
        for ph in range(4):
            assert parked(e, ph, "freeze", "presser")[1] <= 6


# ---------------------------------------------------------------------- F0
@pytest.mark.parametrize("e", [3, 4])
def test_f0_lane_stable_fronts_give_identical_returns(e):
    for ph in range(4):
        f, s = parked(e, ph, "freeze", "park_hold"), parked(e, ph, "slowpush", "park_hold")
        assert f[0] == pytest.approx(s[0], abs=1e-12) and f[1] is None and s[1] is None


@pytest.mark.xfail(strict=True, reason="design finding: on the parked starts Freeze beats Slow Push against the stacker")
def test_f0_stacker_parked_starts_freeze_below_slow_push():
    for e in (6, 7, 8):
        for ph in range(4):
            assert parked(e, ph, "freeze", "stacker")[0] < parked(e, ph, "slowpush", "stacker")[0]


def test_f0_stacker_reachable_scenarios_freeze_is_worse_in_some():
    from lanegame.experiments.m4.freeze_gate import find_scenarios
    sel = [s for s in find_scenarios(M4) if -s["lane"][4] == -1 and 6 <= s["lane"][5] <= 8]
    assert len(sel) >= 20
    diffs = []
    for s in sel:
        lane, m4 = State.from_list(s["lane"]), M4State.from_list(s["champ"])
        f = play(lane, m4, s["phase"], "freeze", "stacker")[0]
        p = play(lane, m4, s["phase"], "slowpush", "stacker")[0]
        diffs.append(f - p)
    d = np.array(diffs)
    assert np.sum(d < -0.05) >= 1                         # Freeze not beneficial in some reachable scenarios
    assert np.sum(d > 0.05) >= 1                          # ... and beneficial in others: no dominance either way
