"""Champion mode M2e: the validation gate in test form. A shortened invariant sweep (every rule, every turn),
the scripted suite, and the behavioral properties the M2e experiments established: degenerate strategies
lose, Yield is Hold against a non-trader, Hold is not a free win against Trade, a trade must pay its opportunity
cost, skill improves execution, and the legacy configurations are untouched. Research-size runs live in
lanegame/experiments/champions (results/champions/)."""

import numpy as np
import pytest

from lanegame import GameConfig
from lanegame.env import LaneEnv, champion_agents, champion_suite, play_episode, run_episodes
from lanegame.env.observation import obs_size
from lanegame.experiments.champions.invariants import sweep

CFG = GameConfig.champions()


def test_invariant_sweep_has_no_violations():
    pairs = [("uniform", "uniform"), ("aggressive_trader", "hold_farmer"), ("shover", "yield_farmer"),
             ("trade", "trade"), ("opportunistic_trader", "punisher"), ("recall_spammer", "vision_player")]
    for cfg in (CFG, CFG.with_(skill_a=0.9, skill_b=0.1, kill_bounty=0.0, recall_turns=3)):
        r = sweep(cfg, pairs, episodes=6, seed=3)
        assert r["violations"] == {}, r["violations"]
        assert r["turns"] == 6 * len(pairs) * cfg.horizon
        assert r["events"]["death"] > 0 and r["events"]["recall"] > 0 and r["events"]["trade_landed"] > 0


def test_scripted_suite_is_complete_and_deterministic():
    names = [a.name for a in champion_suite()]
    assert names == ["hold_farmer", "yield_farmer", "aggressive_trader", "safe_player", "shover",
                     "conservative_recaller", "opportunistic_trader", "vision_player", "punisher"]
    agents = champion_agents(CFG)
    env = LaneEnv(CFG)
    log = play_episode(env, agents["opportunistic_trader"], agents["hold_farmer"], seed=4, record=True)
    for t in log.turns:                                     # every action is legal and the distribution one-hot
        assert t["actions"][0] in t["legal"][0]
    oa, _ = env.reset(seed=0)
    for a in champion_suite():
        p = a.distribution(oa)
        assert p.sum() == 1 and p.max() == 1 and a.act(oa) == oa.legal[int(np.argmax(p))]


def _payoff(a, b, cfg=CFG, episodes=150, seed=8):
    agents = champion_agents(cfg)
    st = run_episodes(LaneEnv(cfg), agents[a], agents[b], episodes, seed)
    return st.mean, st.stderr, st


def test_degenerate_strategies_lose():
    """Recall spam and permanent Safe are heavily punished; the farmer is not beaten by permanent Trade."""
    for bad in ("recall_spammer", "safe_player"):
        m, se, _ = _payoff(bad, "hold_farmer")
        assert m + 4 * se < -5
    m, se, _ = _payoff("aggressive_trader", "hold_farmer", episodes=300)
    assert m < 1.0                                          # trading is not a free win against a careful farmer
    m, se, _ = _payoff("aggressive_trader", "yield_farmer")
    assert m - 4 * se > 3                                   # ... but it does exploit passive evasion


def test_yield_is_exactly_hold_against_a_non_trader():
    _, _, y = _payoff("yield", "hold_farmer", episodes=60)
    _, _, h = _payoff("hold", "hold_farmer", episodes=60)
    assert np.allclose(y.payoffs, h.payoffs)


def test_trade_opportunity_cost_matters():
    """Without the farm penalty a connected trade would be a free bonus and Trade would beat the farmer."""
    free, se_free, _ = _payoff("aggressive_trader", "hold_farmer", CFG.with_(trade_farm_factor=1.0), episodes=300)
    paid, se_paid, _ = _payoff("aggressive_trader", "hold_farmer", episodes=300)
    assert free - paid > 1.0 and free - 3 * se_free > 0


def test_skill_improves_execution():
    lo = _payoff("aggressive_trader", "aggressive_trader", CFG.with_(skill_a=0.3, skill_b=0.7), episodes=100)[2]
    hi = _payoff("aggressive_trader", "aggressive_trader", CFG.with_(skill_a=0.7, skill_b=0.3), episodes=100)[2]
    rate = lambda st, s: st.counters["trades_landed"][:, s].sum() / st.trades_connected[:, s].sum()
    assert rate(hi, 0) > rate(lo, 0) + 0.1 and rate(hi, 1) < rate(lo, 1) - 0.1
    assert hi.mean > lo.mean


def test_legacy_configurations_are_untouched():
    for cfg, n in ((GameConfig(), 34), (GameConfig.v2(), 34), (GameConfig.v1(), 32)):
        assert not cfg.champion_mode and obs_size(cfg) == n
        assert cfg.gank_base == 0.15 and cfg.gank_loss == 5.0      # the legacy expected-loss gank model
