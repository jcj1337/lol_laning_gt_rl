"""M6c step 1: guided replays (lanegame/demo/guides.py; the service's `guides` and `guide`; the API's /api/guides).

A guide is a JSON file under lanegame/demo/guides/ with notes over one verified research recording, checked at the
boundary: `read_guides` for the format's rules and `check_guide` against the recording's game bundle. These tests cover
the rules (each refusal names the file and the rule), the bundle rules against the real seed-0 recording, the API and
the status count, and the facts the first guide's notes quote, read from the recording itself and compared with the
numbers as the notes print them. The tests that need the
engine skip without the 7 checkpoints of lanegame/demo/manifest.json or without the trace file."""

import copy
import json
import os
import re
from dataclasses import replace
from typing import get_args

import pytest

from lanegame.demo import guides as G
from lanegame.demo import registry as REG
from lanegame.demo import schemas as M
from lanegame.demo.service import DemoService, ServiceError

H2H = "h2h_om_s0_vs_sp_s0"
FIRST = "stacking-to-crashing"
SHOVE = "sp_s0_vs_g_shove"                     # the first guide's recording
SHIPPED = os.path.join(G.GUIDES_DIR, f"{FIRST}.json")


@pytest.fixture(scope="module")
def reg():
    if not REG.Registry().available():
        pytest.skip("needs the 7 checkpoints of lanegame/demo/manifest.json")
    return REG.Registry().load()


@pytest.fixture(scope="module")
def service(reg):
    return DemoService(registry=reg)


@pytest.fixture(scope="module")
def bundle(service):
    if not os.path.isfile(os.path.join("results", "m5", "traces", f"{H2H}.json")):
        pytest.skip("results/m5/traces is not on disk")
    return service.replay(H2H)

@pytest.fixture(scope="module")
def guide_bundle(service):
    if not os.path.isfile(os.path.join("results", "m5", "traces", f"{SHOVE}.json")):
        pytest.skip("results/m5/traces is not on disk")
    return service.replay(SHOVE)


@pytest.fixture(scope="module")
def client(service):
    """The app with the loaded service, bot games in this process, no rate limits and no frontend."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from lanegame.demo.api import create_app
    app = create_app(service, sim_workers=0, static_dir=None, limits={}, replays=True)
    with TestClient(app) as c:
        yield c


def shipped() -> dict:
    with open(SHIPPED, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------------------------- the format
def test_the_shipped_guides_parse_and_are_ordered():
    guides = G.read_guides()
    assert guides
    assert [g.id for g in guides] == sorted(f[:-5] for f in os.listdir(G.GUIDES_DIR) if f.endswith(".json"))
    g = next(g for g in guides if g.id == FIRST)
    assert (g.replay, g.start, g.end) == (SHOVE, 74, 84)
    assert [c.position for c in g.checkpoints] == [77, 80, 81, 82, 84]
    d = g.public()
    assert set(d) == {"id", "title", "intro", "replay", "start", "end", "checkpoints"}
    assert all(set(c) == {"id", "position", "title", "text", "highlight"} for c in d["checkpoints"])
    assert d["checkpoints"][0]["highlight"] == {"kind": "champion", "seat": "B"}
    M.Guide.model_validate(d)


def test_malformed_guides_are_refused_with_the_rule_named(tmp_path):
    base, messages = shipped(), []

    def refused(edit, file_stem=None) -> str:
        d = copy.deepcopy(base)
        edit(d)
        folder = tmp_path / str(len(messages))
        folder.mkdir()
        file = f"{file_stem or d['id']}.json"
        (folder / file).write_text(json.dumps(d), encoding="utf-8")
        with pytest.raises(G.GuideError) as e:
            G.read_guides(str(folder))
        assert file in str(e.value)
        messages.append(str(e.value))
        return messages[-1]

    def top(key, value):
        return lambda d: d.__setitem__(key, value)

    def checkpoint(i, key, value):
        return lambda d: d["checkpoints"][i].__setitem__(key, value)

    assert "notes" in refused(top("notes", "a typo"))
    refused(top("id", "Wave-crash-recall"))
    refused(lambda d: None, file_stem="other-name")
    refused(lambda d: d["checkpoints"].reverse())
    refused(checkpoint(1, "id", "no-way-to-base"))
    refused(top("start", 78))
    refused(top("end", 83))
    refused(checkpoint(0, "text", "x" * 401))
    refused(checkpoint(0, "highlight", {"kind": "wave"}))
    refused(checkpoint(0, "highlight", {"kind": "ward_spot", "seat": "A"}))
    assert "minion" in refused(checkpoint(0, "highlight", {"kind": "minion", "seat": "A"}))
    assert "nope" in refused(top("replay", "nope"))
    refused(top("schema", "lanegame.demo_recording"))
    refused(top("checkpoints", []))
    rules = [re.search(r"rule (\d)", m) for m in messages]
    assert all(rules), messages
    assert {int(r.group(1)) for r in rules} == set(range(1, 9)), "every rule names itself in a refusal"
    assert len({m.split(".json: ", 1)[1] for m in messages}) >= 10


# ---------------------------------------------------------------------------------------------- against the game
def test_bundle_rules_are_checked_against_the_real_replay(bundle, guide_bundle):
    g = next(g for g in G.read_guides() if g.id == FIRST)
    G.check_guide(g, guide_bundle)

    def refused(guide, *words, against=bundle):
        with pytest.raises(G.GuideError) as e:
            G.check_guide(guide, against)
        assert all(w in str(e.value) for w in words), str(e.value)

    def one(position, kind, seat):
        return replace(g, replay=H2H, end=None,
                       checkpoints=(G.Checkpoint("x", position, "t", "a note", G.Highlight(kind, seat)),))

    assert len(bundle["turns"]) == len(guide_bundle["turns"]) == 150
    refused(replace(g, checkpoints=g.checkpoints[:-1] + (replace(g.checkpoints[-1], position=151),)), "151",
            against=guide_bundle)
    refused(one(24, "champion", "A"), "champion", "24")
    refused(one(23, "crashed", "B"), "crashed", "23")
    G.check_guide(one(18, "wave", "A"), bundle)
    assert bundle["positions"][123]["lane"]["packets"]["A"] == []
    refused(one(123, "wave", "A"), "wave", "123")
    G.check_guide(one(123, "tower", "A"), bundle)
    G.check_guide(one(123, "ward_spot", None), bundle)


def test_the_api_serves_the_guides(client, service, reg, guide_bundle, tmp_path):
    idx = client.get("/api/guides").json()
    entry = next(e for e in idx if e["id"] == FIRST)
    assert set(entry) == set(M.GuideEntry.model_fields)
    assert entry["checkpoints"] == 5 and entry["replay"] == guide_bundle["name"] == SHOVE
    r = client.get(f"/api/guides/{FIRST}")
    assert r.status_code == 200, r.text
    M.Guide.model_validate(r.json())
    assert r.json() == service.guide(FIRST)
    r = client.get("/api/guides/nope")
    assert r.status_code == 404 and r.json()["error"] == "unknown_guide"
    assert client.get("/api/status").json()["guides"] == 1
    assert set(get_args(M.GuideHighlight.model_fields["kind"].annotation)) == set(G.HIGHLIGHT_KINDS)
    svc = DemoService(registry=reg, traces=str(tmp_path))
    assert svc.guides() == []
    with pytest.raises(ServiceError) as e:
        svc.guide(FIRST)
    assert (e.value.status, e.value.kind) == (404, "unknown_guide") and SHOVE in e.value.message


# ---------------------------------------------------------------------------------------------- the notes' facts
def test_the_first_guides_facts_match_the_recording(guide_bundle):
    """Every number the first guide's notes quote, read from its recording (P[p] is the state after p turns, turn t is
    T[t - 1], a front packet has a true third element, lead_A is positive when A leads), and each quoted number found
    in its note as the browser prints it (toFixed rounds an exact tie up: 3.625 is shown as 3.63)."""
    from decimal import ROUND_HALF_UP, Decimal

    from lanegame.config import GameConfig
    from lanegame.experiments.m4 import zoning_natural as zn

    P, T = guide_bundle["positions"], guide_bundle["turns"]
    notes = {c.id: c.text for c in next(g for g in G.read_guides() if g.id == FIRST).checkpoints}
    cfg = GameConfig.m45_calibrated()
    LAST_HIT, THIN, SLOW_PUSH, HOLD, TRADE, SAFE, PRESSURE = 0, 2, 3, 0, 2, 3, 1

    def about(x):
        return pytest.approx(x, abs=1e-9)

    def shown(x):
        return str(Decimal(abs(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

    def front(p, seat):
        return next([x, h] for x, h, f in P[p]["lane"]["packets"][seat] if f)

    def champ(p, seat):
        return P[p]["champions"][seat]

    def outcome(t, kind, seat):
        return next(o for o in T[t - 1]["outcomes"] if o["kind"] == kind and o["seat"] == seat)

    def part(t, seat):
        return T[t - 1]["parts"][0 if seat == "A" else 1]

    def removed(t, seat):
        return sum(tk["wave_removed"][seat] for tk in T[t - 1]["ticks"])

    def quoted(note, *values):
        for v in values:
            assert str(v) in notes[note], f"{note} should quote {v}"

    assert [s.get("id") for s in guide_bundle["meta"]["seats"]] == ["sp:0", "g_shove"], "A is the agent, B the bot"
    b = champ(74, "B")
    assert (b["hp"], b["hp_max"]) == (274, 1026) and b["hp"] / b["hp_max"] <= zn.RECALL_FRAC == 0.4
    for t in (75, 77):
        assert part(t, "B")["kind"] == "recall" and outcome(t, "recall_interrupted", "B")
        assert part(t, "A")["behaviour"] == TRADE and outcome(t, "trade", "A")["data"]["result"] in ("hit", "miss")
    assert champ(77, "B")["hp"] == 105 and T[76]["delta"]["A"]["hp_taken"] == 125
    quoted("no-way-to-base", "274 of 1026", "40%", "turns 75 and 77", "105 HP", "125 HP")

    for t in (78, 79):
        assert (part(t, "B")["wave"], part(t, "B")["behaviour"]) == (LAST_HIT, SAFE)
        assert (part(t, "A")["wave"], part(t, "A")["behaviour"]) == (SLOW_PUSH, HOLD if t == 78 else TRADE)
    assert part(80, "B")["kind"] == "recall" and outcome(80, "recall_interrupted", "B")
    assert (part(80, "A")["wave"], part(80, "A")["behaviour"]) == (THIN, TRADE)
    assert all(removed(t, "B") == 0 for t in range(75, 82)), "the bot's champion removed none of A's minions"
    assert [removed(t, "A") for t in (78, 79, 80)] == [1, 1, 1]
    assert front(77, "A") == [-1, 22] and outcome(79, "wave_lost", "B") and outcome(80, "wave_lost", "B")
    assert front(80, "A") == [0, 20] and all(x >= 4 for x, _, _ in P[80]["lane"]["packets"]["B"])
    quoted("stack", "turns 78 to 80", "turn 80", "22 HP", "20 HP")

    assert outcome(81, "crash", "A")["data"]["hp"] == 20 and P[81]["lane"]["siege"]["A"] == [2, 20]
    assert P[80]["lane"]["siege"]["A"][1] == 0
    quoted("crash", "turn 81", "20 HP")

    assert (part(82, "A")["wave"], part(82, "A")["behaviour"], part(82, "A")["objective"]) == (THIN, HOLD, PRESSURE)
    assert outcome(82, "tower_damage", "A")["data"]["damage"] == 1602 and cfg.plate_thresholds[0] == 1600
    assert outcome(82, "plate", "A")["data"]["gold"] == about(1.0) and outcome(82, "recall", "B")
    assert T[81]["delta"]["A"]["gold"] == about(1.3375) and T[81]["delta"]["B"]["gold"] == about(0.0)
    assert P[81]["lead_A"] == about(1.6375) and P[82]["lead_A"] == about(2.975)
    quoted("plate", "turn 82", "1602", "1600", "+1.00", shown(1.3375), shown(1.6375), shown(2.975))

    assert part(83, "B")["kind"] == "absent" and champ(84, "B")["status"] == "present"
    assert [outcome(t, "tower_damage", "A")["data"]["damage"] for t in (83, 84)] == [1402, 1302]
    assert outcome(84, "plate", "A")["data"]["gold"] == about(1.0) and cfg.plate_thresholds[1] == 3600
    assert P[84]["towers"]["B"]["plates_lost"] == 2 and P[84]["towers"]["B"]["damage"] == 4306
    assert P[75]["lead_A"] == about(-1.0325) and P[84]["lead_A"] == about(3.625)
    quoted("second-plate", "1402", "1302", "3600", "turn 84", "turn 75", shown(-1.0325), shown(3.625))
