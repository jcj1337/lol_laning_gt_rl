"""Champion mode M2d: ganks as events with HP damage, death and respawn. Occurrence versus damage, vision,
exposure, the death timeline, recall versus death, simultaneous deaths, zero-sum accounting without the
legacy expected-loss term, and the economics of dying in good and bad wave states."""

import numpy as np
import pytest

from lanegame import GameConfig
from lanegame.champions import (N_CHANCE, RECALL, U_GANK, U_GANK_DAMAGE, U_TRADE_DOUBLE, U_TRADE_HIT, ChampSide,
                                ChampState, champion_gank_probability, champion_legal_actions, gank_damage_distribution,
                                gank_damage_p, mirror_chance, resolve_turn, swap_seats)
from lanegame.config import B_HOLD, B_SAFE, B_TRADE, B_YIELD, CHAMP_ABSENT, SH, SP, champ_act
from lanegame.env import LaneEnv, champion_agents, play_episode, run_episodes
from lanegame.state import State

CFG = GameConfig.champions(horizon=16)
T = CFG.tower
CONTACT = State(0, 4, (), 0, 0, 4, (), 0, 0, 0)
SIEGE_A = State(-4, 4, (), 8, -3, 4, (), 0, 0, 0, 2, 0)         # A's crash sieges B's tower
PUSHED_A = State(-4, 4, (), 0, 1, 8, (), 0, 0, 0, 0, 0)         # B's 8-HP wave walks into A's tower
ONE = (1.0,) * N_CHANCE


def ch(pa, pb, hpa=6, hpb=6, absa=0, absb=0):
    return ChampState(ChampSide(pa, hpa, absa), ChampSide(pb, hpb, absb))


def H(w=SP):
    return champ_act(w, B_HOLD)


def S(w=SP):
    return champ_act(w, B_SAFE)


def TR(w=SP):
    return champ_act(w, B_TRADE)


def ganked(side, damage_u=0.0, base=ONE):
    """Chance vector in which `side` is ganked (damage drawn with damage_u) and nothing else happens."""
    u = list(base)
    u[U_GANK[side]] = 0.0
    u[U_GANK_DAMAGE[side]] = damage_u
    return u


# ------------------------------------------------------------------ occurrence
def test_gank_occurrence_defaults_and_ordering():
    p = lambda beh, e, w=SP, lane=CONTACT: champion_gank_probability(lane, 0, w, beh, CFG, e)
    assert p(B_HOLD, -T) == pytest.approx(0.01) and p(B_SAFE, -T) == 0.0          # no unconditional base rate
    assert p(B_HOLD, 0) == pytest.approx(0.07) and p(B_TRADE, 0) == pytest.approx(0.09)
    assert p(B_HOLD, T) == pytest.approx(0.13) and p(B_HOLD, 0, SH) == pytest.approx(0.09)
    for e in range(-T, T + 1):
        assert p(B_TRADE, e) >= p(B_HOLD, e) == p(B_YIELD, e) >= p(B_SAFE, e)
        if e < T:
            assert p(B_HOLD, e + 1) > p(B_HOLD, e)
        assert p(B_HOLD, e, lane=CONTACT._replace(va=3)) == pytest.approx(0.25 * p(B_HOLD, e))   # vision
    assert p(B_SAFE, -1) < 0.05                                                   # vision is not needed there


def test_exposure_is_not_erased_by_retreating():
    t = resolve_turn(CONTACT, ch(T, 0), 2, S(), H(), CFG)
    assert t.exposure[0] == T and t.gank_p[0] == pytest.approx(champion_gank_probability(CONTACT, 0, SP, B_SAFE, CFG, T))
    assert t.gank_p[0] > resolve_turn(CONTACT, ch(-1, 0), 2, S(), H(), CFG).gank_p[0]


# ------------------------------------------------------------------ damage
def test_gank_damage_distribution_and_its_inputs():
    for pe in (0.0, 0.2, 0.5, 0.8, 1.0):
        d = gank_damage_distribution(pe, CFG)
        assert len(d) == CFG.gank_damage_trials + 1 and sum(d) == pytest.approx(1.0)
        assert sum(i * q for i, q in enumerate(d)) == pytest.approx(CFG.gank_damage_trials * pe)
    assert gank_damage_p(0, B_HOLD, -T, CFG) == pytest.approx(0.2)
    assert gank_damage_p(0, B_HOLD, 0, CFG) == pytest.approx(0.5) and gank_damage_p(0, B_HOLD, T, CFG) == pytest.approx(0.8)
    assert gank_damage_p(0, B_SAFE, 0, CFG) == gank_damage_p(0, B_YIELD, 0, CFG) == pytest.approx(0.3)
    assert gank_damage_p(0, B_TRADE, 0, CFG) == pytest.approx(0.5)
    hi, lo = CFG.with_(skill_a=1.0), CFG.with_(skill_a=0.0)
    assert gank_damage_p(0, B_HOLD, 0, hi) < gank_damage_p(0, B_HOLD, 0, CFG) < gank_damage_p(0, B_HOLD, 0, lo)


def test_full_hp_champion_never_dies_to_one_gank_but_can_escape_unhurt():
    assert CFG.gank_damage_trials < CFG.champion_hp
    for beh_a, start in ((H(), T), (TR(), T), (H(), 0)):
        t = resolve_turn(CONTACT, ch(start, 0), 2, beh_a, H(), CFG, u=ganked(0, damage_u=0.999999))
        assert t.ganked[0] and t.gank_damage[0] <= CFG.gank_damage_trials and not t.died[0]
    t = resolve_turn(CONTACT, ch(-1, 0), 2, S(), H(), CFG, u=ganked(0, damage_u=0.0))
    assert t.ganked[0] and t.gank_damage[0] == 0 and t.champ.a.hp == 6       # a gank may deal nothing


def test_vision_reduces_occurrence_only():
    lane, vis = CONTACT, CONTACT._replace(va=3)
    a = resolve_turn(lane, ch(0, 0), 2, H(), H(), CFG)
    b = resolve_turn(vis, ch(0, 0), 2, H(), H(), CFG)
    assert b.gank_p[0] == pytest.approx((1 - CFG.vision_reduction) * a.gank_p[0])
    assert b.gank_p[1] == pytest.approx(a.gank_p[1])
    for damage_u in np.linspace(0.01, 0.99, 9):
        x = resolve_turn(lane, ch(0, 0), 2, H(), H(), CFG, u=ganked(0, damage_u))
        y = resolve_turn(vis, ch(0, 0), 2, H(), H(), CFG, u=ganked(0, damage_u))
        assert x.gank_damage_p == y.gank_damage_p and x.gank_damage == y.gank_damage


def test_gank_damage_frequencies_match_the_binomial():
    """10,000 seeded environment steps: realized gank damage given occurrence follows Binomial(4, p_e)."""
    cfg = CFG.with_(champ_gank_base=1.0, champ_gank_max=1.0)         # force occurrence to measure damage alone
    env = LaneEnv(cfg)
    counts = np.zeros(cfg.gank_damage_trials + 1)
    n = 10_000
    for i in range(n):
        env.reset(seed=i, root=CONTACT, k0=2, champ=ch(0, 0))
        r = env.step(H(), H())
        assert r.info["ganked"][0]
        counts[r.info["gank_damage"][0]] += 1
    p = np.array(gank_damage_distribution(r.info["gank_damage_p"][0], cfg))
    se = np.sqrt(p * (1 - p) / n)
    assert np.all(np.abs(counts / n - p) < 4.5 * se + 1e-12), (counts / n, p)


def test_ganks_under_own_tower_are_rare_and_light():
    t = resolve_turn(PUSHED_A, ch(-T, 0), 2, H(), H(), CFG)
    assert t.exposure[0] == -T and t.gank_p[0] <= CFG.champ_gank_behavior[B_HOLD] + CFG.champ_gank_strength * CFG.max_stacks + 1e-12
    assert sum(i * q for i, q in enumerate(gank_damage_distribution(gank_damage_p(0, B_HOLD, -T, CFG), CFG))) < 1.0


# ------------------------------------------------------------------ death and respawn
def test_death_while_low_and_the_respawn_timeline():
    """HP 1, traded for 1: dies at the end of turn t (keeps its farm), ABSENT on t+1..t+3, back at its tower with
    full HP on t+4."""
    u = list(ONE)
    u[U_TRADE_HIT[1]] = 0.0
    k_death = 2
    t = resolve_turn(CONTACT, ch(0, 0, hpa=1), k_death, H(), TR(), CFG, u=u)
    assert t.connected[1] and t.died == (True, False) and t.farm[0] > 0      # the farm of the death turn is kept
    env = LaneEnv(CFG)
    env.reset(seed=0, root=t.lane.state, k0=k_death + 1, champ=t.champ)
    assert env.champ.a == ChampSide(-T, 0, CFG.death_turns)
    missed = 0
    while env.champ.a.absent > 0:
        oa, ob = env.observe()
        assert oa.legal == (CHAMP_ABSENT,)
        r = env.step(CHAMP_ABSENT, H() if H() in ob.legal else ob.legal[0])
        missed += 1
        assert r.info["away"][0] and r.info["farm"][0] == 0 and r.info["gank_p"][0] == 0
    assert missed == CFG.death_turns and env.k == k_death + 1 + CFG.death_turns
    assert env.champ.a == ChampSide(-T, CFG.champion_hp, 0) and H() in env.observe()[0].legal


def test_simultaneous_deaths_from_trades_and_ganks():
    u = list(ONE)
    for s in (0, 1):
        u[U_TRADE_HIT[s]] = 0.99 * CFG.trade_hit_base                # both trades land 1 HP
        u[U_GANK[s]] = 0.0                                          # and both are ganked
        u[U_GANK_DAMAGE[s]] = 0.999999                              # for heavy damage
    t = resolve_turn(CONTACT, ch(0, 0, hpa=3, hpb=4), 2, TR(), TR(), CFG, u=u)
    assert t.died == (True, True) and t.trade_damage == (1, 1)
    assert t.champ == ch(-T, -T, 0, 0, CFG.death_turns, CFG.death_turns)
    m = resolve_turn(CONTACT.mirror(), ch(0, 0, hpa=4, hpb=3), 2, TR(), TR(), swap_seats(CFG), u=mirror_chance(u))
    assert m.died == (True, True) and m.champ == t.champ.mirror()


def test_damage_sources_add_up_to_a_kill():
    """A trade (1) and a gank (2) together kill a 3-HP champion that neither kills alone."""
    u = list(ONE)
    u[U_TRADE_HIT[1]] = 0.99 * CFG.trade_hit_base
    u[U_GANK[0]] = 0.0
    pe = gank_damage_p(0, B_HOLD, 0, CFG)
    u[U_GANK_DAMAGE[0]] = sum(gank_damage_distribution(pe, CFG)[:2]) + 1e-9      # damage exactly 2
    t = resolve_turn(CONTACT, ch(0, 0, hpa=3), 2, H(), TR(), CFG, u=u)
    assert t.trade_damage[1] == 1 and t.gank_damage[0] == 2 and t.died[0]
    t = resolve_turn(CONTACT, ch(0, 0, hpa=4), 2, H(), TR(), CFG, u=u)
    assert not t.died[0] and t.champ.a.hp == 1


def test_recall_versus_death():
    """A successful recall leaves before anything can hit it; an interrupted recall can be killed."""
    u = list(ONE)
    u[U_TRADE_HIT[1]] = u[U_TRADE_DOUBLE[1]] = 0.0
    u[U_GANK[0]] = 0.0
    t = resolve_turn(CONTACT, ch(-T, -T, hpa=1), 2, RECALL, TR(), CFG, u=u)     # trader out of reach
    assert t.recalled[0] and not t.died[0] and not t.ganked[0] and t.champ.a.absent == CFG.recall_turns - 1
    t = resolve_turn(CONTACT, ch(-1, 0, hpa=2), 2, RECALL, TR(), CFG, u=u)      # interrupted and killed
    assert t.interrupted[0] and t.died[0] and t.champ.a.absent == CFG.death_turns


def test_episode_ends_during_death_absence():
    env = LaneEnv(CFG)
    env.reset(seed=0, root=CONTACT, k0=CFG.horizon - 1, champ=ch(-T, 0, hpa=0, absa=CFG.death_turns))
    r = env.step(CHAMP_ABSENT, H())
    assert r.terminated and env.champ.a.absent == CFG.death_turns - 1


# ------------------------------------------------------------------ rewards
def test_reward_has_no_gank_loss_and_is_zero_sum_with_bounties():
    agents = champion_agents(CFG)
    for bounty in (0.0, 1.0):
        cfg = CFG.with_(kill_bounty=bounty)
        env = LaneEnv(cfg)
        log = play_episode(env, agents["trade"], agents["hold"], seed=3, record=True)
        deaths = [0, 0]
        for t in log.turns:
            assert t["gank_payoff"] == 0.0 and t["crash_payoff"] == 0
            assert t["reward"] == pytest.approx(t["farm_payoff"] + t["plate_payoff"] + t["bounty_payoff"])
            assert t["bounty_payoff"] == pytest.approx(bounty * (int(t["died"][1]) - int(t["died"][0])))
            deaths = [deaths[s] + int(t["died"][s]) for s in (0, 1)]
        assert sum(log.rewards) == pytest.approx(log.payoff) and log.deaths == deaths
        assert log.bounty == [bounty * deaths[1], bounty * deaths[0]]
    st = run_episodes(LaneEnv(CFG), agents["trade"], agents["hold"], 30, seed=5)
    assert st.counters["deaths"][:, 1].mean() > 0                    # a non-recalling holder does die
    assert np.all(st.counters["deaths"] <= CFG.horizon / (CFG.death_turns + 1) + 1)


def test_reproducible_with_fixed_seeds():
    agents = champion_agents(CFG)
    env = LaneEnv(CFG)
    l1 = play_episode(env, agents["shove_trade"], agents["yield"], seed=21, record=True)
    l2 = play_episode(env, agents["shove_trade"], agents["yield"], seed=21, record=True)
    assert l1.rewards == l2.rewards and l1.deaths == l2.deaths and l1.gank_damage_taken == l2.gank_damage_taken
    assert [t["champ_after"] for t in l1.turns] == [t["champ_after"] for t in l2.turns]


# ------------------------------------------------------------------ death economics
def _absence_cost(lane, k0, turns=6):
    """Payoff to A over `turns` turns (A Holds when present, B Holds), with A present at -T versus A dead (away
    death_turns turns, starting now). No chance events."""
    def run(champ):
        total = 0.0
        ln, c = lane, champ
        for k in range(k0, k0 + turns):
            la = champion_legal_actions(ln, c, 0, CFG)
            a = CHAMP_ABSENT if la == (CHAMP_ABSENT,) else H()
            t = resolve_turn(ln, c, k, a, H(), CFG)
            total += t.farm[0] - t.farm[1] + CFG.plate_value * (t.lane.plates[0] - t.lane.plates[1])
            ln, c = t.lane.state, t.champ
        return total
    return run(ch(-T, -T)) - run(ch(-T, -T, hpa=0, absa=CFG.death_turns))


def test_death_into_an_enemy_crash_costs_more_than_a_death_while_sieging():
    """The cost of a death is lost farm and wave control, so it depends on the wave state (no death penalty)."""
    for k0 in (2, 3):
        good, bad = _absence_cost(SIEGE_A, k0), _absence_cost(PUSHED_A, k0)
        assert bad > good + 0.5 and good >= -1e-9
