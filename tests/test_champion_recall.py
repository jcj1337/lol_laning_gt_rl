"""Champion mode M2b: Recall and absence. Legality, the exact timeline, configurable duration, what an away
champion does not do, interruption by an enemy trade, and the economics of recall timing (no explicit
penalty: the cost is the farm and wave control given up)."""

import numpy as np
import pytest

from lanegame import GameConfig
from lanegame.champions import (AWAY, CHANNEL, FARM, NO_REACH, RECALL, TRADING, ChampSide, ChampState,
                                champion_collection, champion_legal_actions, initial_champions, recall_legal,
                                resolve_turn, swap_seats)
from lanegame.config import B_HOLD, B_RECALL, B_SAFE, B_TRADE, CHAMP_ABSENT, SH, SP, champ_act
from lanegame.env import LaneEnv, feature_names, play_episode
from lanegame.mechanics import step_turn
from lanegame.state import State

CFG = GameConfig.champions(horizon=16)
T = CFG.tower
CONTACT = State(0, 4, (), 0, 0, 4, (), 0, 0, 0)                 # fronts fighting at the lane (x = 0)
SIEGE_A = State(-4, 4, (), 8, -3, 4, (), 0, 0, 0, 2, 0)         # A's 8 HP under B's tower, B's wave waiting at -3
PUSHED_A = State(-4, 4, (), 0, 1, 8, (), 0, 0, 0, 0, 0)         # B's 8-HP wave walking into A's tower


def ch(pa, pb, hpa=6, hpb=6, absa=0, absb=0):
    return ChampState(ChampSide(pa, hpa, absa), ChampSide(pb, hpb, absb))


def H(w=SP):
    return champ_act(w, B_HOLD)


def S(w=SP):
    return champ_act(w, B_SAFE)


def TR(w=SP):
    return champ_act(w, B_TRADE)


# ------------------------------------------------------------------ legality
def test_recall_is_legal_only_near_own_tower_as_one_code():
    for pos in range(-T, T + 1):
        champ = ch(pos, 0)
        legal = champion_legal_actions(CONTACT, champ, 0, CFG)
        assert (RECALL in legal) == (pos <= CFG.recall_line) == recall_legal(champ, 0, CFG)
        assert CHAMP_ABSENT not in legal
        assert all(champ_act(w, B_RECALL) not in legal for w in (SH, 2))     # only the Slow Push code
    assert RECALL == champ_act(SP, B_RECALL) == 12


def test_absent_champion_has_only_the_absent_action():
    for left in (1, 2, 3):
        champ = ch(-T, 0, absa=left)
        assert champion_legal_actions(CONTACT, champ, 0, CFG) == (CHAMP_ABSENT,)
        assert not recall_legal(champ, 0, CFG)
        assert RECALL not in champion_legal_actions(CONTACT, ch(-1, 0, absb=left), 1, CFG)
    env = LaneEnv(CFG)
    oa, ob = env.reset(seed=0, root=CONTACT, k0=2, champ=ch(-T, 0, absa=1))
    assert oa.legal == (CHAMP_ABSENT,) and oa.mask.sum() == 1 and oa.mask[CHAMP_ABSENT]
    with pytest.raises(ValueError):
        env.step(H(), ob.legal[0])                                # a normal action while absent
    idx = {n: i for i, n in enumerate(feature_names(CFG))}
    assert oa.vector[idx["own_champ_absent"]] == pytest.approx(1 / CFG.champion_absent_max)
    assert ob.vector[idx["enemy_champ_absent"]] == pytest.approx(1 / CFG.champion_absent_max)


def test_forward_champion_must_retreat_before_recalling():
    """No teleport from the enemy tower: a champion at +T walks back (Safe) at champion_speed cells per turn and
    recalls on a later turn."""
    env = LaneEnv(CFG)
    oa, ob = env.reset(seed=0, root=CONTACT, k0=2, champ=ch(T, 0))
    turns = 0
    while RECALL not in oa.legal:
        r = env.step(S(), H())
        oa, ob = r.obs
        turns += 1
        assert env.champ.a.pos >= T - turns * CFG.champion_speed
    assert turns == 2 and env.champ.a.pos <= CFG.recall_line          # +2 -> 0 -> -1, then Recall


def test_config_validation():
    with pytest.raises(AssertionError):
        GameConfig.champions(recall_turns=0).validate()
    with pytest.raises(AssertionError):
        GameConfig.champions(recall_turns=5).validate()             # absence timer beyond the observation scale
    GameConfig.champions(recall_turns=4, champion_absent_max=3).validate()


# ------------------------------------------------------------------ the timeline
@pytest.mark.parametrize("recall_turns", [1, 2, 3])
def test_recall_timeline_is_exact(recall_turns):
    """Recall on turn t: off the lane on t, forced ABSENT on t+1 .. t+recall_turns-1, back at its own tower with
    full HP at the start of t+recall_turns with a normal legal set. Exactly recall_turns turns away."""
    cfg = CFG.with_(recall_turns=recall_turns)
    env = LaneEnv(cfg)
    oa, ob = env.reset(seed=0, root=CONTACT, k0=2, champ=ch(-1, 0, hpa=2))
    assert RECALL in oa.legal
    r = env.step(RECALL, H())
    away = [r.info["away"][0]]
    timers = [env.champ.a.absent]
    assert r.info["recalled"] == (True, False) and r.info["farm"][0] == 0 and r.info["gank_p"][0] == 0
    while env.champ.a.absent > 0:
        oa, ob = r.obs
        assert oa.legal == (CHAMP_ABSENT,) and env.champ.a.pos == -T
        r = env.step(CHAMP_ABSENT, ob.legal[0] if H() not in ob.legal else H())
        away.append(r.info["away"][0])
        timers.append(env.champ.a.absent)
        assert r.info["farm"][0] == 0 and r.info["modes"][0] == AWAY
    oa, _ = r.obs
    assert away == [True] * recall_turns and len(away) == recall_turns
    assert timers == list(range(recall_turns - 1, -1, -1))
    assert env.champ.a == ChampSide(-T, cfg.champion_hp, 0)          # back at the tower, healed
    assert env.k == 2 + recall_turns and RECALL in oa.legal and H() in oa.legal


def test_episode_can_end_during_absence():
    env = LaneEnv(CFG)
    oa, ob = env.reset(seed=0, root=CONTACT, k0=CFG.horizon - 2, champ=ch(-T, 0, absa=3))
    r = env.step(CHAMP_ABSENT, H())
    r = env.step(r.obs[0].legal[0], H())
    assert r.terminated and env.champ.a.absent == 1                   # still away when the horizon is reached
    with pytest.raises(RuntimeError):
        env.step(CHAMP_ABSENT, H())


# ------------------------------------------------------------------ an away champion takes no part
def test_away_champion_takes_no_part_in_the_lane():
    """No farm, no wave contribution (the lane step equals Slow Push with no reach), no zoning, no trade, no
    objective, not gankable; the opponent's Trade has nobody to chase and farms like Hold."""
    champ = ch(-T, 0, absa=1)
    for b in (H(), H(SH), TR(), TR(SH)):
        t = resolve_turn(CONTACT, champ, 2, CHAMP_ABSENT, b, CFG)
        assert t.away == (True, False) and t.modes[0] == AWAY and t.reach[0] == NO_REACH
        assert t.farm[0] == 0 and t.gank_p[0] == 0 and t.connected == (False, False)
        assert t.effective_objectives[0] == 0 and t.lane_codes[0] == SP
        ref = step_turn(CONTACT, 2, SP, t.lane_codes[1], CFG, champion_reach=(NO_REACH, t.reach[1]))
        assert t.lane.state == ref.state and t.lane.kills == ref.kills
        for x in range(-T - 2, T + 3):                                # the opponent collects unzoned
            assert champion_collection(x, FARM, t.end[1], AWAY, -T, CFG) >= champion_collection(x, FARM, t.end[1], FARM, 0, CFG)
        th = resolve_turn(CONTACT, champ, 2, CHAMP_ABSENT, champ_act(b % 3, B_HOLD), CFG)
        assert t.end[1] == th.end[1] and t.farm[1] == th.farm[1]     # Trade without a target = Hold
    t = resolve_turn(SIEGE_A, ch(-T, 0, absa=1), 2, CHAMP_ABSENT, H(), CFG)   # its siege runs without it
    assert t.lane.plates == (0, 0) and t.lane.thinned == (0, 0) and t.lane.vision_set == (False, False)
    assert t.lane.events["tower_hp"][0] > 0                           # the tower keeps clearing the crashed mass


def test_opponent_farms_freely_while_the_champion_is_away():
    stay = resolve_turn(CONTACT, ch(-1, 0), 2, H(), H(), CFG)
    gone = resolve_turn(CONTACT, ch(-1, 0), 2, RECALL, H(), CFG)
    assert gone.recalled[0] and gone.farm[0] == 0 < stay.farm[0]
    assert gone.farm[1] > stay.farm[1]                                # nobody zones B any more


# ------------------------------------------------------------------ interruption
def test_enemy_trade_interrupts_a_recall_in_reach():
    t = resolve_turn(CONTACT, ch(-1, 0), 2, RECALL, TR(), CFG)
    assert t.interrupted == (True, False) and not t.recalled[0] and not t.away[0]
    assert t.modes == (CHANNEL, TRADING) and t.connected == (False, True) and t.end[0] == -1
    assert t.farm[0] == 0 and t.reach[0] == NO_REACH and t.gank_p[0] > 0
    assert t.champ.a == ChampSide(-1, 6, 0)                           # still in lane, plays next turn
    # a dive interrupts a recall at the tower
    t = resolve_turn(CONTACT, ch(-T, 0), 2, RECALL, TR(), CFG)
    assert t.interrupted[0] and t.end == (-T, T)
    # out of reach (4 cells, champion_speed 2 + trade_range 1 = 3): the recall succeeds
    t = resolve_turn(CONTACT, ch(-T, -T), 2, RECALL, TR(), CFG)
    assert t.recalled[0] and not t.interrupted[0] and t.modes[1] == FARM and t.champ.a.absent == CFG.recall_turns - 1
    # a non-trader never interrupts
    for b in (H(), S(), champ_act(SP, 1)):
        assert resolve_turn(CONTACT, ch(-1, 0), 2, RECALL, b, CFG).recalled[0]


def test_both_recall_and_mirror_equivariance():
    t = resolve_turn(CONTACT, ch(-1, -1), 2, RECALL, RECALL, CFG)
    assert t.recalled == (True, True) and t.farm == (0.0, 0.0) and t.lane.state == step_turn(
        CONTACT, 2, SP, SP, CFG, champion_reach=(NO_REACH, NO_REACH)).state
    asym = CFG.with_(skill_a=0.8, skill_b=0.3)
    for lane, champ, a, b in ((CONTACT, ch(-1, 0), RECALL, TR()), (CONTACT, ch(-T, 0, absa=2), CHAMP_ABSENT, TR()),
                              (PUSHED_A, ch(-1, 0), RECALL, H()), (CONTACT, ch(-1, -1), RECALL, RECALL)):
        x = resolve_turn(lane, champ, 2, a, b, asym)
        m = resolve_turn(lane.mirror(), champ.mirror(), 2, b, a, swap_seats(asym))
        assert m.champ == x.champ.mirror() and m.lane.state == x.lane.state.mirror()
        assert m.interrupted == x.interrupted[::-1] and m.recalled == x.recalled[::-1] and m.modes == x.modes[::-1]
        assert np.allclose(m.farm, x.farm[::-1]) and np.allclose(m.gank_p, x.gank_p[::-1])


# ------------------------------------------------------------------ economics of timing
def _window(lane, champ, k0, first_a, turns=6):
    """Payoff to A over `turns` turns: A plays first_a, then Hold (ABSENT while away); B Holds."""
    total = 0.0
    for k in range(k0, k0 + turns):
        la = champion_legal_actions(lane, champ, 0, CFG)
        a = first_a if k == k0 else (CHAMP_ABSENT if la == (CHAMP_ABSENT,) else H())
        assert a in la
        t = resolve_turn(lane, champ, k, a, H(), CFG)
        total += t.farm[0] - t.farm[1] + CFG.plate_value * (t.lane.plates[0] - t.lane.plates[1])
        lane, champ = t.lane.state, t.champ
    return total


def test_well_timed_recall_costs_less_than_a_poorly_timed_one():
    """Recalling while one's own crashed wave occupies the enemy tower costs (almost) nothing; recalling while the
    enemy wave walks into one's own tower gives up the farm of that crash. No explicit recall penalty exists."""
    for k0 in (2, 3):
        well = _window(SIEGE_A, ch(-1, -2), k0, H()) - _window(SIEGE_A, ch(-1, -2), k0, RECALL)
        poor = _window(PUSHED_A, ch(-1, 0), k0, H()) - _window(PUSHED_A, ch(-1, 0), k0, RECALL)
        assert well == pytest.approx(0.0, abs=1e-9)
        assert poor > 0.5 and poor > well + 0.5


# ------------------------------------------------------------------ episodes
class _RecallAtAgent:
    """Hold, Recall on the given turns when legal (Safe to get there), ABSENT while away."""
    def __init__(self, turns, name="recall_at"):
        self.turns, self.name = set(turns), name

    def act(self, obs, rng):
        if obs.legal == (CHAMP_ABSENT,):
            return CHAMP_ABSENT
        if obs.k in self.turns:
            return RECALL if RECALL in obs.legal else S()
        return H()


def test_episodes_with_recalls_are_complete_zero_sum_and_reproducible():
    env = LaneEnv(CFG)
    a, b = _RecallAtAgent({3, 4, 9, 10}), _RecallAtAgent({6, 7})
    l1 = play_episode(env, a, b, seed=4, record=True)
    l2 = play_episode(env, a, b, seed=4, record=True)
    assert l1.actions == l2.actions and l1.rewards == l2.rewards and l1.length == CFG.horizon
    assert l1.recalls[0] >= 1 and l1.recalls[1] >= 1
    assert l1.turns_away == [CFG.recall_turns * l1.recalls[0], CFG.recall_turns * l1.recalls[1]]
    for t in l1.turns:
        assert t["reward"] == pytest.approx(t["farm_payoff"] + t["plate_payoff"] + t["gank_payoff"])
        for s in (0, 1):
            if t["away"][s]:
                assert t["farm"][s] == 0 and t["gank_p"][s] == 0 and not t["ganked"][s]
                assert t["actions"][s] in (RECALL, CHAMP_ABSENT)
