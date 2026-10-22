"""The per-turn moves the lane's arrows draw (lanegame/demo/frames.moves; the Turn frame's `moves`).

For each seat, a champion move (walk, recall, return or death, between global cells, the ward spot "W" and "base") and
a leading-wave move (walk, or crash into the enemy tower). A wave move follows the leading packet the turn started
with, so it exists only when the seat lost no packet and crashed none, or when the crash was its only loss. The first
test drives the rule with hand-made frames and needs no engine; the others check it against the three verified
research replays and pin the turns the first guided replay walks through."""

import os

import pytest

from lanegame.config import GameConfig
from lanegame.demo import frames as F
from lanegame.demo import registry as REG

TRACES = os.path.join("results", "m5", "traces")


def _champ(status="present", cell=0, at_ward=False):
    return {"status": status, "cell": cell if status == "present" and not at_ward else None, "at_ward": at_ward}


def _view(a_champ, b_champ, a_packets, b_packets):
    return {"champions": {"A": a_champ, "B": b_champ}, "lane": {"packets": {"A": a_packets, "B": b_packets}}}


def _info(died=(0, 0), crash=(0, 0), champ_died=(False, False), recalled=(False, False), returned=(False, False)):
    return {"events": {"died": list(died)}, "crash_hp": list(crash), "died": list(champ_died),
            "recalled": list(recalled), "returned": list(returned)}


def test_the_rule_on_hand_made_frames():
    cfg = GameConfig.m45_calibrated()
    pre = _view(_champ(cell=-2), _champ(cell=1), [[-1, 8, True], [-4, 8, False]], [[1, 6, True]])
    post = _view(_champ(cell=0), _champ(cell=1), [[0, 8, True]], [[2, 6, True]])
    m = F.moves(cfg, _info(), pre, post)
    assert m["A"] == {"champion": {"kind": "walk", "from": -2, "to": 0}, "wave": {"kind": "walk", "from": -1, "to": 0}}
    assert m["B"]["champion"] is None, "a champion that stayed has no move"
    assert m["B"]["wave"] == {"kind": "walk", "from": 1, "to": 2}
    lost = F.moves(cfg, _info(died=(1, 0)), pre, post)
    assert lost["A"]["wave"] is None, "after a lost packet the end front may be another wave: no wave move"
    crash = F.moves(cfg, _info(crash=(20, 0)), pre, post)
    assert crash["A"]["wave"] == {"kind": "crash", "from": -1, "to": cfg.tower}
    assert F.moves(cfg, _info(died=(1, 0), crash=(20, 0)), pre, post)["A"]["wave"] is None
    away = _view(_champ(status="recall"), _champ(status="dead"), [[-1, 8, True]], [[1, 6, True]])
    rec = F.moves(cfg, _info(recalled=(True, False), champ_died=(False, True)), pre, away)
    assert rec["A"]["champion"] == {"kind": "recall", "from": -2, "to": "base"}
    assert rec["B"]["champion"] == {"kind": "death", "from": 1, "to": "base"}
    back = F.moves(cfg, _info(returned=(True, False)), away, _view(_champ(cell=-2), _champ(status="dead"),
                                                                   [[-1, 8, True]], [[1, 6, True]]))
    assert back["A"]["champion"] == {"kind": "return", "from": "base", "to": -2}
    assert back["B"]["champion"] is None, "a champion away for the whole turn has no move"
    ward = F.moves(cfg, _info(), pre, _view(_champ(at_ward=True), _champ(cell=1), [[0, 8, True]], [[2, 6, True]]))
    assert ward["A"]["champion"] == {"kind": "walk", "from": -2, "to": "W"}


@pytest.fixture(scope="module")
def replays():
    if not REG.Registry().available():
        pytest.skip("needs the 7 git-ignored checkpoints of lanegame/demo/manifest.json")
    if not os.path.isfile(os.path.join(TRACES, "sp_s0_vs_g_shove.json")):
        pytest.skip("results/m5/traces is not on disk")
    from lanegame.demo.service import DemoService
    svc = DemoService(registry=REG.Registry().load())
    return {e["name"]: svc.replay(e["name"]) for e in svc.replays()}


def test_every_move_agrees_with_the_recorded_states(replays):
    """Over every turn of the three replays: waves only walk forward and at most three units, a crash move sits on a
    recorded crash, and a champion's move matches its start and end states."""
    counted = {"walk": 0, "crash": 0, "recall": 0, "return": 0, "death": 0}
    for name, b in replays.items():
        P = b["positions"]
        for tr in b["turns"]:
            t = tr["t"]
            for n, sign in (("A", 1), ("B", -1)):
                c, w = tr["moves"][n]["champion"], tr["moves"][n]["wave"]
                mine = {o["kind"] for o in tr["outcomes"] if o["seat"] == n}
                pre, post = P[t - 1]["champions"][n], P[t]["champions"][n]
                if w is not None:
                    counted[w["kind"]] += 1
                    if w["kind"] == "walk":
                        assert 0 < sign * (w["to"] - w["from"]) <= 3, (name, t, n, w)
                        assert "wave_lost" not in mine and "crash" not in mine, (name, t, n)
                    else:
                        assert "crash" in mine and w["to"] == sign * 2, (name, t, n, w)
                if c is None:
                    continue
                counted[c["kind"]] += 0 if c["kind"] == "walk" else 1
                if c["kind"] == "walk":
                    assert pre["status"] == post["status"] == "present" and c["from"] != c["to"], (name, t, n)
                elif c["kind"] == "recall":
                    assert "recall" in mine and post["status"] == "recall" and c["to"] == "base", (name, t, n)
                elif c["kind"] == "return":
                    assert pre["status"] != "present" and post["status"] == "present", (name, t, n)
                    assert c["to"] == post["cell"], (name, t, n)
                else:
                    assert "death" in mine and post["status"] == "dead", (name, t, n)
    assert all(counted.values()), f"every kind of move occurs in the replays: {counted}"


def test_the_first_guides_turns(replays):
    T = replays["sp_s0_vs_g_shove"]["turns"]

    def moves(t, n):
        return T[t - 1]["moves"][n]

    assert moves(74, "A")["champion"] == {"kind": "walk", "from": 0, "to": "W"}, "the agent walks to the ward spot"
    assert moves(77, "A")["champion"] == {"kind": "walk", "from": 0, "to": 2}, "and to the bot's tower to trade"
    assert moves(80, "A")["wave"] == {"kind": "walk", "from": -1, "to": 0}, "the stacked wave reaches the lane"
    assert moves(80, "B")["wave"] is None, "the bot's wave died"
    assert moves(81, "A")["wave"] == {"kind": "crash", "from": 0, "to": 2}, "and crashes into the bot's tower"
    assert moves(82, "B")["champion"] == {"kind": "recall", "from": 2, "to": "base"}
    assert moves(83, "B")["champion"] == {"kind": "return", "from": "base", "to": 2}
    H = replays["h2h_om_s0_vs_sp_s0"]["turns"]
    assert H[23]["moves"]["A"]["champion"] == {"kind": "recall", "from": -2, "to": "base"}
    assert H[24]["moves"]["A"]["champion"] == {"kind": "return", "from": "base", "to": -2}
    D = replays["om_s0_vs_g_shove"]["turns"]
    assert D[10]["moves"]["A"]["champion"] == {"kind": "death", "from": -1, "to": "base"}
