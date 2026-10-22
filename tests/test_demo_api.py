"""M6b: the demo API over the session core (lanegame/demo/api.py; docs/m6/m6_plan.md §3 rules 1 to 5, 7, 8, 10 to 13
through the API, §8.2 lifecycle and limits).

The core's own rules are tested in test_demo_session.py; these tests cover what the API layer adds: the typed
contract (every real shape validates against schemas.py, and the committed OpenAPI file matches the app), the HTTP
statuses, session expiry and recovery, rate and size limits, the static frontend, and the simulation worker process.
The checkpoint-dependent tests skip without the 7 files of lanegame/demo/manifest.json, so a skipped run proves
nothing about them."""

import json
import os
import threading
import time

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from lanegame.demo import registry as REG  # noqa: E402
from lanegame.demo import schemas as M  # noqa: E402
from lanegame.demo import session as S  # noqa: E402
from lanegame.demo.api import MAX_BODY, create_app  # noqa: E402
from lanegame.demo.service import DemoService, ServiceError  # noqa: E402

OPENAPI = os.path.join("web", "openapi.json")
TRACES = os.path.join("results", "m5", "traces")


@pytest.fixture(scope="module")
def reg():
    if not REG.Registry().available():
        pytest.skip("needs the 7 git-ignored checkpoints of lanegame/demo/manifest.json")
    return REG.Registry().load()


@pytest.fixture(scope="module")
def service(reg):
    return DemoService(registry=reg)


@pytest.fixture(scope="module")
def client(service):
    """The app with the loaded service, bot games in this process, no rate limits and no frontend."""
    app = create_app(service, sim_workers=0, static_dir=None, limits={}, replays=True)
    with TestClient(app) as c:
        yield c


def keys_of(x, acc=None):
    """Every key that carries a value (the API serializes an absent optional field as null)."""
    acc = set() if acc is None else acc
    if isinstance(x, dict):
        for k, v in x.items():
            if v is not None:
                acc.add(k)
            keys_of(v, acc)
    elif isinstance(x, (list, tuple)):
        for v in x:
            keys_of(v, acc)
    return acc


def play(c: TestClient, sid: str, turns: int, pick=None):
    """Submit `turns` turns through the API with a fixed stand-in for the visitor; returns the turn results."""
    out = []
    for _ in range(turns):
        v = c.get(f"/api/sessions/{sid}").json()
        if v["state"] == "stopped":
            break
        m, t = v["menu"], v["next_turn"]
        code = m["forced"] if m["forced"] is not None else (pick(t, m) if pick else m["legal"][(t * 7 + 3) % len(m["legal"])])
        r = c.post(f"/api/sessions/{sid}/turns", json={"turn": t, "code": code})
        assert r.status_code == 200, r.text
        out.append(r.json())
    return out


# ---------------------------------------------------------------------------------------------- the contract
def test_the_openapi_schema_on_disk_is_the_apps():
    """`web/openapi.json` (the source of the frontend's generated types) equals the app's current schema; regenerate
    it with `python -m lanegame.demo --openapi web/openapi.json` after a contract change."""
    assert os.path.isfile(OPENAPI), "web/openapi.json is missing; write it with python -m lanegame.demo --openapi"
    with open(OPENAPI, encoding="utf-8") as f:
        on_disk = json.load(f)
    assert on_disk == create_app(static_dir=None).openapi()


def test_the_models_accept_the_cores_real_output_and_refuse_extras(reg):
    s = S.GameSession(reg, (S.Seat.human(), S.Seat.model("om:0")), belief_mode="shown")
    M.View.model_validate(s.view())
    for t in range(1, 6):
        legal = s.env.legal(0)
        r = s.submit(t, legal[(t * 7 + 3) % len(legal)])
        M.TurnResult.model_validate(r)
        assert set(r["turn"]["ticks"][0]) == set(M.Tick.model_fields)
    M.History.model_validate(s.history())
    M.Recording.model_validate(s.recording())
    s.stop()
    M.Summary.model_validate(s.summary())
    M.GameBundle.model_validate(S.simulate(reg, S.Seat.model("sp:0"), S.Seat.model("om:0"), turns=8)
                                .bundle("new_simulation", "x"))
    bad = dict(s.view(), pending_code=7)
    with pytest.raises(Exception):
        M.View.model_validate(bad)


# ---------------------------------------------------------------------------------------------- a live game
def test_status_and_models(client):
    st = client.get("/api/status").json()
    assert st["ready"] and st["simulations"] == "in_process" and st["turn_cap"] == 150
    assert {m["id"] for m in client.get("/api/models").json()} == {"sp:0", "om:0"}


def test_a_live_game_through_the_api(client, reg):
    g = client.post("/api/sessions", json={"opponent": "om:0", "belief": "hidden"})
    assert g.status_code == 200, g.text
    g = g.json()
    sid, v = g["session"], g["view"]
    assert v["opponent_ready"] is True and v["human_seat"] == "A" and v["next_turn"] == 1
    assert not keys_of(g) & {"belief", "beliefs", "pending", "logits", "probs"}
    assert v["seeds"]["chance"] is None and v["seeds"]["policy"] is None
    code = v["menu"]["legal"][0]
    r = client.post(f"/api/sessions/{sid}/turns", json={"turn": 1, "code": code})
    assert r.status_code == 200 and r.json()["turn"]["codes"][0] == code and r.json()["view"]["position"] == 1
    first = r.json()
    # a duplicate returns the stored result; conflicts, future turns and illegal codes are refused with their status
    again = client.post(f"/api/sessions/{sid}/turns", json={"turn": 1, "code": code})
    assert again.status_code == 200 and again.json()["turn"] == first["turn"] and again.json()["position"] == first["position"]
    for body, status, kind in (({"turn": 1, "code": code + 1}, 409, "turn_conflict"),
                               ({"turn": 5, "code": code}, 409, "turn_out_of_order"),
                               ({"turn": 2, "code": 45}, 422, "illegal_action"),
                               ({"turn": 2, "code": 99}, 422, "illegal_action"),
                               ({"turn": 2, "code": True}, 400, "bad_request"),
                               ({"turn": "2", "code": code}, 400, "bad_request"),
                               ({"turn": 2}, 400, "bad_request")):
        r = client.post(f"/api/sessions/{sid}/turns", json=body)
        assert (r.status_code, r.json()["error"]) == (status, kind), r.text
        assert r.json()["message"]
    assert client.get(f"/api/sessions/{sid}").json()["position"] == 1
    # the illegal code's message is the menu's reason
    r = client.post(f"/api/sessions/{sid}/turns", json={"turn": 2, "code": 45})
    assert "Absent" in r.json()["message"]
    # history: resolved positions only, no beliefs in hidden mode
    assert client.get(f"/api/sessions/{sid}/history", params={"from": 0, "to": 9}).status_code == 403
    h = client.get(f"/api/sessions/{sid}/history", params={"from": 0, "to": 1}).json()
    assert [p["p"] for p in h["positions"]] == [0, 1] and len(h["turns"]) == 1 and h.get("beliefs") is None
    assert h["turns"][0] == first["turn"] and h["positions"][1] == first["position"]
    # the record withholds the seeds and the hidden beliefs while active
    rec = client.get(f"/api/sessions/{sid}/record").json()
    assert rec["seeds"]["chance"] is None and rec["seeds"]["policy"] is None and rec.get("beliefs") is None
    # stop: no winner, the seeds and the belief history revealed, the recording replays
    out = client.post(f"/api/sessions/{sid}/stop")
    assert out.status_code == 200, out.text
    out = out.json()
    assert out["summary"]["stop_reason"] == "visitor" and "no winner" in out["summary"]["note"]
    assert not ({"winner", "won", "lost", "score"} & (set(out["summary"]) | set(out["summary"]["marks"])))
    assert out["recording"]["seeds"]["chance"] is not None and len(out["recording"]["beliefs"]["posteriors"]["B"]) == 2
    assert S.replay(reg, out["recording"])["turns"] == 1
    rec = client.get(f"/api/sessions/{sid}/record").json()
    assert rec == out["recording"]
    assert client.post(f"/api/sessions/{sid}/turns", json={"turn": 2, "code": code}).status_code == 410
    assert client.get(f"/api/sessions/{sid}").json()["state"] == "stopped"
    assert client.get("/api/sessions/" + "0" * 32).status_code == 404
    assert client.get("/api/sessions/not-an-id").status_code == 404
    assert client.post("/api/sessions", json={"opponent": "g_shove"}).status_code == 400


def test_history_after_a_reload_equals_the_turn_results_and_hides_the_future(client):
    g = client.post("/api/sessions", json={"opponent": "sp:0"}).json()
    sid = g["session"]
    results = play(client, sid, 14, pick=lambda t, m: 12 if (t == 3 and 12 in m["legal"]) else m["legal"][-1])
    h = client.get(f"/api/sessions/{sid}/history").json()
    assert [p["p"] for p in h["positions"]] == list(range(15))
    assert h["turns"] == [r["turn"] for r in results]
    assert [p for p in h["positions"][1:]] == [r["position"] for r in results]
    assert h["positions"][0] == g["history"]["positions"][0]
    assert any(t["parts"][0]["kind"] == "absent" for t in h["turns"]), "the Recall forced an Absent turn"
    v = client.get(f"/api/sessions/{sid}").json()
    assert v["position"] == 14 and v["next_turn"] == 15
    assert client.get(f"/api/sessions/{sid}/history", params={"from": 14, "to": 15}).status_code == 403


def test_forced_absent_turns_take_one_code_at_a_time(client):
    g = client.post("/api/sessions", json={"opponent": "sp:0"}).json()
    sid, v = g["session"], g["view"]
    assert 12 in v["menu"]["legal"], "Recall is legal at the own tower at the start"
    r = client.post(f"/api/sessions/{sid}/turns", json={"turn": 1, "code": 12}).json()
    m = r["view"]["menu"]
    assert m["forced"] == 45 and m["away"] == {"status": "recall", "returns_in": 1} and m["legal"] == [45]
    assert all(not w["legal"] and w["reason"] for w in m["waves"] + m["behaviours"])
    bad = client.post(f"/api/sessions/{sid}/turns", json={"turn": 2, "code": 7})
    assert bad.status_code == 422 and "recalling" in bad.json()["message"]
    ok = client.post(f"/api/sessions/{sid}/turns", json={"turn": 2, "code": 45}).json()
    assert ok["turn"]["parts"][0]["kind"] == "absent" and ok["view"]["menu"]["forced"] is None


def test_shown_belief_mode_carries_the_belief_and_hidden_mode_does_not(client):
    for mode, visible in (("shown", True), ("hidden", False)):
        g = client.post("/api/sessions", json={"opponent": "om:0", "belief": mode}).json()
        sid = g["session"]
        has = lambda d, k: d.get(k) is not None  # noqa: E731 (an absent optional field is serialized as null)
        assert has(g["view"], "belief") is visible and has(g["history"], "beliefs") is visible
        r = play(client, sid, 3)[-1]
        assert has(r, "belief") is visible and has(r["view"], "belief") is visible
        h = client.get(f"/api/sessions/{sid}/history").json()
        assert has(h, "beliefs") is visible
        if visible:
            assert len(h["beliefs"]["B"]["posteriors"]) == 4 and len(r["belief"]["B"]["posterior"]) == 13
        out = client.post(f"/api/sessions/{sid}/stop").json()
        assert len(out["recording"]["beliefs"]["posteriors"]["B"]) == 4 and has(out["history"], "beliefs")


def test_concurrent_duplicate_submissions_resolve_the_turn_once_through_the_api(client):
    g = client.post("/api/sessions", json={"opponent": "sp:0"}).json()
    sid, code = g["session"], g["view"]["menu"]["legal"][2]
    results, errors = [], []

    def go():
        try:
            r = client.post(f"/api/sessions/{sid}/turns", json={"turn": 1, "code": code})
            results.append((r.status_code, r.json()["view"]["position"] if r.status_code == 200 else None))
        except Exception as e:                                  # pragma: no cover
            errors.append(e)
    threads = [threading.Thread(target=go) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and results == [(200, 1)] * 8
    assert len(client.get(f"/api/sessions/{sid}/record").json()["turns"]) == 1


# ---------------------------------------------------------------------------------------------- Watch
def test_a_simulation_bundle_is_labelled_consistent_and_replays(client, reg):
    r = client.post("/api/simulations", json={"a": "sp:0", "b": "om:0", "turns": 20})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["meta"]["provenance"]["kind"] == "new_simulation" and b["recording"]["provenance"]["kind"] == "new_simulation"
    assert len(b["positions"]) == 21 and len(b["turns"]) == 20 and len(b["beliefs"]["posteriors"]["B"]) == 21
    assert [t["codes"] for t in b["turns"]] == [t[:2] for t in b["recording"]["turns"]]
    assert b["positions"][-1]["lead_A"] == pytest.approx(b["meta"]["lead_A"], abs=1e-6)
    assert b["meta"]["seeds"]["chance"] is not None and b["meta"]["seeds"]["policy"] is not None
    assert S.replay(reg, b["recording"])["turns"] == 20
    assert client.post("/api/simulations", json={"a": "sp:0", "b": "sp:0", "turns": 0}).status_code == 400
    assert client.post("/api/simulations", json={"a": "g_shove"}).status_code == 400


def test_the_research_replays_are_verified_bundles(client):
    if not os.path.isfile(os.path.join(TRACES, "h2h_om_s0_vs_sp_s0.json")):
        pytest.skip("results/m5/traces is not on disk")
    idx = client.get("/api/replays").json()
    assert [e["name"] for e in idx] == ["h2h_om_s0_vs_sp_s0", "om_s0_vs_g_shove", "sp_s0_vs_g_shove"]
    assert all(e["kind"] == "research_record" and e["turns"] == 150 for e in idx)
    b = client.get("/api/replays/h2h_om_s0_vs_sp_s0").json()
    assert b["meta"]["provenance"]["kind"] == "research_record" and b["meta"]["provenance"]["source"]["sha256"]
    assert b["verification"]["actions"] == 150 and b["verification"]["belief_max_abs_diff"] == 0.0
    with open(os.path.join(TRACES, "h2h_om_s0_vs_sp_s0.json"), encoding="utf-8") as f:
        assert b["verification"]["payoff_A"] == json.load(f)["payoff_A"]
    assert len(b["positions"]) == 151 and b["meta"]["seeds"]["chance"] == b["meta"]["seeds"]["policy"]
    assert client.get("/api/replays/nope").status_code == 404
    assert client.get("/api/status").json()["replays"] == 3


# ---------------------------------------------------------------------------------------------- lifecycle and limits
def test_expired_sessions_are_dropped_and_recovery_is_a_clear_404(reg):
    svc = DemoService(registry=reg, idle_seconds=0.05)
    app = create_app(svc, sim_workers=0, static_dir=None, limits={}, replays=False, sweep_seconds=0.02)
    with TestClient(app) as c:
        sid = c.post("/api/sessions", json={}).json()["session"]
        assert c.get(f"/api/sessions/{sid}").status_code == 200
        time.sleep(0.3)                                         # the sweeper runs every 0.02 s
        r = c.get(f"/api/sessions/{sid}")
        assert r.status_code == 404 and r.json()["error"] == "unknown_session" and "expired" in r.json()["message"]
        assert svc.count() == 0


def test_the_session_cap_drops_the_least_recently_active(reg):
    svc = DemoService(registry=reg, max_sessions=2)
    a = svc.create()["session"]
    b = svc.create()["session"]
    svc.view(a)                                                 # a is now more recent than b
    c = svc.create()["session"]
    assert svc.count() == 2
    with pytest.raises(ServiceError) as e:
        svc.view(b)
    assert e.value.status == 404
    assert svc.view(a)["session"] == a and svc.view(c)["session"] == c


def test_rate_and_size_limits(reg):
    app = create_app(DemoService(registry=reg), sim_workers=0, static_dir=None, replays=False,
                     limits={"sessions": (0.0, 3), "turns": (1000.0, 1000), "simulations": (0.0, 1)})
    with TestClient(app) as c:
        codes = [c.post("/api/sessions", json={}).status_code for _ in range(4)]
        assert codes == [200, 200, 200, 429]
        assert c.post("/api/simulations", json={"turns": 2}).status_code == 200
        r = c.post("/api/simulations", json={"turns": 2})
        assert r.status_code == 429 and r.json()["error"] == "rate_limited"
        big = c.post("/api/sessions", content=b"{" + b" " * MAX_BODY + b"}", headers={"content-type": "application/json"})
        assert big.status_code == 413


def test_the_api_without_an_engine_says_so(tmp_path):
    svc = DemoService(load=False, traces=str(tmp_path))
    assert svc.registry is None and svc.error
    app = create_app(svc, sim_workers=1, static_dir=None, limits={})
    with TestClient(app) as c:
        st = c.get("/api/status").json()
        assert st["ready"] is False and st["simulations"] == "unavailable" and st["models"] == []
        assert c.post("/api/sessions", json={}).status_code == 503
        assert c.post("/api/simulations", json={}).status_code == 503
        assert c.get("/api/replays").status_code == 503


def test_the_built_frontend_is_served_with_the_single_page_fallback(tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>app</title>", encoding="utf-8")
    (dist / "assets" / "x.js").write_text("console.log(1)", encoding="utf-8")
    (dist / "icon.svg").write_text("<svg/>", encoding="utf-8")
    app = create_app(DemoService(load=False), sim_workers=0, static_dir=str(dist), limits={})
    with TestClient(app) as c:
        assert c.get("/").text.startswith("<!doctype html>")
        assert c.get("/play?x=1").text.startswith("<!doctype html>")
        assert c.get("/assets/x.js").text == "console.log(1)" and c.get("/icon.svg").text == "<svg/>"
        assert c.get("/api/nope").status_code == 404 and c.get("/api/nope").json()["error"] == "not_found"
        assert c.get("/../pyproject.toml").text.startswith("<!doctype html>")
    app = create_app(DemoService(load=False), sim_workers=0, static_dir=str(tmp_path / "none"))
    with TestClient(app) as c:
        assert "not built" in c.get("/").text


# ---------------------------------------------------------------------------------------------- the worker process
def test_the_simulation_worker_process_generates_a_replayable_game_without_touching_this_process(reg):
    """A bot game from the separate worker process: the bundle validates and replays here, and this process's
    service created no session for it (rule 3: nothing mutable is shared)."""
    svc = DemoService(registry=reg)
    app = create_app(svc, sim_workers=1, static_dir=None, limits={}, replays=False)
    with TestClient(app) as c:
        t0 = time.time()
        while c.get("/api/status").json()["simulations"] == "starting":
            assert time.time() - t0 < 120, "the worker did not start"
            time.sleep(0.2)
        assert c.get("/api/status").json()["simulations"] == "ready"
        before = svc.count()
        r = c.post("/api/simulations", json={"a": "om:0", "b": "sp:0", "turns": 30})
        assert r.status_code == 200, r.text
        b = M.GameBundle.model_validate(r.json())
        assert len(b.turns) == 30 and svc.count() == before
        assert S.replay(reg, r.json()["recording"])["turns"] == 30
        # the same seeds in this process give the same game: the worker's engine is the same engine
        rec = r.json()["recording"]
        here = S.simulate(reg, S.Seat.model("om:0"), S.Seat.model("sp:0"), turns=30,
                          chance_seed=rec["seeds"]["chance"], policy_seed=rec["seeds"]["policy"]).recording()
        assert here["turns"] == rec["turns"] and here["lead_A"] == rec["lead_A"]
