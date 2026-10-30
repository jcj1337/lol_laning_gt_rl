"""The facts the demo's frames state about damage and lost minion packets (lanegame/demo/frames.py), checked against
the engine.

Damage: the resolver sums a seat's trade, tower, gank and minion damage and divides the sum by that seat's durability
M(S) (champions_m4 step 9b), so the per-source outcomes quote damage, and an `hp_lost` outcome gives the HP removed
whenever the two differ. Lost packets: each tick frame's `packets_lost` counts the seat's minion packets that died in
that tick (crashes excluded). The engine counts deaths only per turn (`info["events"]["died"]`), so the per-tick
oracle wraps mechanics._tick_natural in the test and reads the counter before and after each tick.

The replay tests re-simulate the three verified research replays from their recordings to read the engine's `info`
beside the frames the service serves; they skip without the 7 checkpoints of lanegame/demo/manifest.json
or without the trace files."""

import os

import numpy as np
import pytest

from lanegame import mechanics as M
from lanegame.champions_m4 import M4State, context
from lanegame.config import GameConfig
from lanegame.demo import frames as F
from lanegame.demo import record as R
from lanegame.demo import registry as REG
from lanegame.env import LaneEnv

TRACES = os.path.join("results", "m5", "traces")
H2H = "h2h_om_s0_vs_sp_s0"


@pytest.fixture(scope="module")
def replays():
    if not REG.Registry().available():
        pytest.skip("needs the 7 checkpoints of lanegame/demo/manifest.json")
    if not os.path.isfile(os.path.join(TRACES, f"{H2H}.json")):
        pytest.skip("results/m5/traces is not on disk")
    from lanegame.demo.service import DemoService
    svc = DemoService(registry=REG.Registry().load())
    return {e["name"]: svc.replay(e["name"]) for e in svc.replays()}


def engine_turns(bundle):
    """(cfg, the engine's info, the served turn frame) for every turn of a replay, re-simulated from its recording."""
    rec = bundle["recording"]
    cfg = R.config_of(rec)
    env = LaneEnv(cfg, record_ticks=True)
    env.reset(seed=int(rec["seeds"]["chance"]))
    for (a, b, digest), frame in zip(rec["turns"], bundle["turns"]):
        info = env.step(int(a), int(b)).info
        assert R.state_digest(env.state, env.champ, env.k) == digest
        yield cfg, info, frame


def damage_taken_by(outcomes, seat):
    """The per-source damage the outcomes quote for one seat: the other seat's hit, a tower shot, a gank, minion fire."""
    return sum(o["data"]["damage"] for o in outcomes
               if (o["kind"] in ("tower_shot", "gank", "minion_fire") and o["seat"] == seat)
               or (o["kind"] == "trade" and o["seat"] != seat and o["data"]["result"] == "hit"))


def test_quoted_damage_is_what_the_resolver_divides_by_durability(replays):
    turns = differ = 0
    for bundle in replays.values():
        for cfg, info, frame in engine_turns(bundle):
            ctx = context(cfg)
            before = M4State.from_list(info["champ_before"])
            for s, n in enumerate(F.SEATS):
                damage = damage_taken_by(frame["outcomes"], n)
                resolver_sum = (info["trade_damage"][1 - s] + info["tower_damage"][s] + info["gank_damage"][s]
                                + info["minion_damage"][s])
                assert damage == resolver_sum, (bundle["name"], frame["t"], n)
                hp = frame["delta"][n]["hp_taken"]
                assert ctx.mitigate(damage, before.side(s).items) == hp, (bundle["name"], frame["t"], n)
                lost = [o for o in frame["outcomes"] if o["kind"] == "hp_lost" and o["seat"] == n]
                if damage != hp:
                    differ += 1
                    assert lost == [{"kind": "hp_lost", "seat": n, "major": False, "data": {"damage": damage, "hp": hp},
                                     "text": f"{{{n}}} lost {hp} HP from {damage} damage; invested gold reduces "
                                             "damage taken"}], (bundle["name"], frame["t"], n)
                else:
                    assert not lost, (bundle["name"], frame["t"], n)
            assert not any(o["text"].endswith(" HP") for o in frame["outcomes"]
                           if o["kind"] in ("trade", "tower_shot", "gank", "minion_fire")), frame["t"]
            turns += 1
    assert turns == 450 and differ > 0, (turns, differ)

    H = replays[H2H]["turns"]

    def shown(t, kind, seat):
        return [o["data"] for o in H[t - 1]["outcomes"] if o["kind"] == kind and o["seat"] == seat]

    assert shown(35, "tower_shot", "A") == [{"damage": 141}] and shown(35, "hp_lost", "A") == [{"damage": 141, "hp": 108}]
    assert H[34]["delta"]["A"]["hp_taken"] == 108
    assert shown(50, "trade", "A")[0]["damage"] == 420 and shown(50, "hp_lost", "B") == [{"damage": 420, "hp": 303}]
    kinds = [o["kind"] for o in H[49]["outcomes"]]
    assert kinds.index("hp_lost") > max(kinds.index("trade"), kinds.index("minion_fire")), "after the damage outcomes"


def test_packets_lost_add_up_to_the_engines_deaths(replays):
    ticks = with_loss = 0
    for bundle in replays.values():
        for _, info, frame in engine_turns(bundle):
            for s, n in enumerate(F.SEATS):
                assert sum(tk["packets_lost"][n] for tk in frame["ticks"]) == info["events"]["died"][s], \
                    (bundle["name"], frame["t"], n)
            ticks += len(frame["ticks"])
            with_loss += sum(1 for tk in frame["ticks"] for n in F.SEATS if tk["packets_lost"][n] > 0)
    assert ticks == 900 and with_loss > 0, (ticks, with_loss)


def _random_games(seeds=(1, 2, 3), turns=150):
    cfg = GameConfig.m45_calibrated()
    for seed in seeds:
        rng = np.random.default_rng(seed)
        env = LaneEnv(cfg, record_ticks=True)
        env.reset(seed=seed)
        for _ in range(turns):
            a, b = (int(rng.choice(env.legal(s))) for s in (0, 1))
            yield cfg, env.step(a, b).info


def test_packets_lost_are_each_ticks_deaths(monkeypatch, replays):
    """Per tick, against the engine's own death counter, over the three replays and three games of random legal
    play."""
    real = M._tick_natural

    def counted(*args):
        trace, ev = args[6], args[7]
        before = list(ev["died"])
        real(*args)
        if trace is not None:
            trace[-1]["died_in_tick"] = [ev["died"][s] - before[s] for s in (0, 1)]

    monkeypatch.setattr(M, "_tick_natural", counted)
    games = [((cfg, info) for cfg, info, _ in engine_turns(b)) for b in replays.values()] + [_random_games()]
    ticks = deaths = 0
    for game in games:
        for cfg, info in game:
            for got, tk in zip(F.packets_lost(cfg, info), info["lane_trace"], strict=True):
                assert list(got) == tk["died_in_tick"], (info["k"], got, tk["died_in_tick"])
                ticks += 1
                deaths += sum(got)
    assert ticks == 900 + 900 and deaths > 0, (ticks, deaths)
