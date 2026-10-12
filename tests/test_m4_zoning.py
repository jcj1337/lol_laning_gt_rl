"""M4.2d-C: the opt-in `zoning_rule` (docs/m4/m4_zoning_report.md). "presence" (default) is the M2 proximity zone;
"contact" zones only a collector hit by a connected enemy Trade; "none" never zones."""

from fractions import Fraction

import numpy as np
import pytest

from lanegame import GameConfig
from lanegame.champions import (AWAY, CHANNEL, FARM, N_CHANCE, RETREAT, SAFE_MODE, TRADING, champion_collection,
                                mirror_chance, swap_seats)
from lanegame.champions_m4 import check_invariants, collection_units, resolve_turn_m4
from lanegame.config import B_HOLD, B_TRADE, B_YIELD, SP, champ_act
from lanegame.experiments.m4 import zoning as zx

RULES = ("presence", "contact", "none")
CFG = {r: GameConfig.m4(zoning_rule=r) for r in RULES}
M4 = GameConfig.m4()
T = M4.tower
MODES = (FARM, TRADING, RETREAT, SAFE_MODE, AWAY, CHANNEL)
CELLS = range(-T, T + M4.thin_reach + 1)          # kill cells (thinned minions can stand behind the enemy tower)
POS = range(-T, T + 1)


def _gold_only_fields(t):
    return (t.farm, t.farm_units, t.earned_units, t.reward_units, t.reward, t.champ.a.gold, t.champ.b.gold,
            t.gold_clamped)


def test_default_is_presence_and_the_field_is_validated():
    assert GameConfig.m4().zoning_rule == "presence" and GameConfig().zoning_rule == "presence"
    for r in RULES:
        CFG[r].validate()
    with pytest.raises(ValueError, match="zoning_rule"):
        GameConfig.m4(zoning_rule="threat").validate()
    for preset in (GameConfig.champions, GameConfig.champions_continuing, GameConfig):
        with pytest.raises(ValueError, match="zoning_rule"):
            preset(zoning_rule="none").validate()


def test_presence_is_exactly_the_m2_collection_rule():
    """Default preservation, exhaustively: the "presence" ledger value is the M2 `champion_collection` share."""
    f, scale = M4.farm_per_hp, M4.gold_scale
    for x in CELLS:
        for mode in MODES:
            for pos in POS:
                for om in MODES:
                    for op in POS:
                        got = collection_units(x, mode, pos, om, op, M4)
                        assert got == collection_units(x, mode, pos, om, op, CFG["presence"])
                        assert got == pytest.approx(champion_collection(x, mode, pos, om, op, M4) * f * scale, abs=1)


def test_none_and_contact_rules_exhaustively():
    """none = the unzoned value whatever the opponent does; contact = the zoned value iff the opponent's Trade
    connected (opponent in trading mode), else unzoned. "none" is never below either other rule; "contact" can be
    below "presence" (a connected trader zones even a kill more than farm_reach from it)."""
    q = lambda v: Fraction(repr(float(v)))
    for x in CELLS:
        for mode in MODES:
            for pos in POS:
                unz = collection_units(x, mode, pos, AWAY, -T, M4)                 # an absent opponent zones nothing
                zone = q(M4.zone_contest) if x > -T else q(M4.zone_under_tower)
                zoned = int(Fraction(unz) * (1 - zone))
                for om in MODES:
                    for op in POS:
                        pres = collection_units(x, mode, pos, om, op, CFG["presence"])
                        none = collection_units(x, mode, pos, om, op, CFG["none"])
                        cont = collection_units(x, mode, pos, om, op, CFG["contact"])
                        assert none == unz
                        assert cont == (zoned if om == TRADING else unz)
                        assert pres <= none and cont <= none
                        if mode not in (AWAY, CHANNEL) and abs(x - pos) <= M4.farm_reach and unz:
                            assert none > 0                                     # reach and posture still decide


def test_the_rule_changes_gold_only_and_is_mirror_equivariant():
    """From the same state, actions and chance vector the three rules give the same positions, lane, HP, XP,
    trades, ganks and tower damage; only farm gold (and what is derived from it) differs. Each rule is
    mirror-equivariant (swap seats, reflect the lane, mirror the chance vector)."""
    from lanegame.env import LaneEnv, UniformAgent
    asym = M4.with_(skill_a=0.8, skill_b=0.3)
    env = LaneEnv(asym, max_turns=300)
    rng, nrng = np.random.default_rng(1), np.random.default_rng(7)
    ag = UniformAgent()
    obs = env.reset(seed=5)
    differs, zoned_trades = 0, 0
    for _ in range(300):
        lane, m4, k = env.state, env.champ, env.k
        a, b = ag.act(obs[0], rng), ag.act(obs[1], rng)
        u = tuple(float(v) for v in nrng.random(N_CHANCE))
        ts = {r: resolve_turn_m4(lane, m4, k, a, b, asym.with_(zoning_rule=r), u=u) for r in RULES}
        p = ts["presence"]
        for r in ("contact", "none"):
            t = ts[r]
            assert (t.end, t.modes, t.lane.state, t.damage_taken, t.died, t.xp_gain, t.levels_after,
                    t.tower_damage_dealt, t.plate_units, t.bounty_units) == \
                   (p.end, p.modes, p.lane.state, p.damage_taken, p.died, p.xp_gain, p.levels_after,
                    p.tower_damage_dealt, p.plate_units, p.bounty_units)
            if r == "none":
                assert all(t.farm_units[s] >= p.farm_units[s] for s in (0, 1))
            assert all(t.farm_units[s] <= ts["none"].farm_units[s] for s in (0, 1))
            m = resolve_turn_m4(lane.mirror(), m4.mirror(), k, b, a, swap_seats(asym.with_(zoning_rule=r)),
                                u=mirror_chance(u))
            assert m.champ == t.champ.mirror() and m.farm_units == t.farm_units[::-1]
            check_invariants(t.champ, asym)
        differs += _gold_only_fields(ts["none"]) != _gold_only_fields(p)
        zoned_trades += any(ts["contact"].connected) and ts["contact"].farm_units != ts["none"].farm_units
        r_ = env.step(a, b)
        obs = r_.obs
        if r_.done:
            break
    assert differs > 20 and zoned_trades > 0


@pytest.fixture(scope="module")
def starts():
    return zx.starts(GameConfig.m4())


def _turn(rule, start, ba, bb):
    lane, m4, k = start
    return resolve_turn_m4(lane, m4, k, champ_act(SP, ba), champ_act(SP, bb), CFG[rule], u=None)


def test_new_behavior_at_the_lane(starts):
    """Neutral state (both champions and every kill at the lane): mutual Hold collects the full contest base under
    "none" (0.45 = 2 HP x 0.25 x 0.9) instead of 0.315; retreating from a Trade still costs the Yield champion
    its contest posture (0.20 at Safe efficiency 0.4), and connecting a Trade still halves the trader's farm."""
    s = starts["neutral"]
    mu = M4.gold_scale
    assert _turn("presence", s, B_HOLD, B_HOLD).farm_units == (315_000, 315_000)
    assert _turn("none", s, B_HOLD, B_HOLD).farm_units == (450_000, 450_000)
    ty = _turn("none", s, B_TRADE, B_YIELD)
    assert ty.threatened[1] and not ty.connected[0] and ty.farm_units == (450_000, 200_000)
    th = {r: _turn(r, s, B_TRADE, B_HOLD) for r in RULES}
    assert all(t.connected == (True, False) for t in th.values())
    assert th["none"].farm_units == (225_000, 450_000)             # trader x 0.5; the victim is not zoned
    assert th["contact"].farm_units == (225_000, 315_000)          # the victim of a connected Trade is zoned
    assert th["presence"].farm_units == (157_500, 315_000)         # both zoned by proximity
    assert sum(_turn("none", s, B_HOLD, B_HOLD).farm_units) <= 2 * 0.5 * mu


def test_gold_attribution_sums_exactly_and_has_no_zone_part_under_none(starts):
    cfg = CFG["none"]
    for name, st in starts.items():
        for ba in zx.BEHAVIORS:
            for bb in zx.BEHAVIORS:
                for row in zx.run_pair(cfg, st, ba, bb, horizon=2):
                    for s in (0, 1):
                        g = row["gold"][s]
                        assert g["zone"] == 0.0
                        assert abs(sum(v for k, v in g.items() if k != "available") - g["available"]) < 1e-12
                        assert row["xp"][s] <= row["xp_available"][s]
