"""M6a: the demo's headless session core (lanegame/demo; docs/m6/m6_plan.md §3 rules 1 to 5, 7, 10 to 12, §11).

The checks that need the 7 git-ignored checkpoints of lanegame/demo/manifest.json skip when any is missing, so a
skipped run proves nothing about rules 1 to 5; the manifest and tampering checks need no checkpoint."""

import copy
import hashlib
import json
import os
import threading

import numpy as np
import pytest

from lanegame.demo import actions as ACT
from lanegame.demo import frames as F
from lanegame.demo import record as R
from lanegame.demo import registry as REG
from lanegame.demo import session as S

TRACES = os.path.join("results", "m5", "traces")
H2H = os.path.join(TRACES, "h2h_om_s0_vs_sp_s0.json")
need_ckpt = pytest.mark.skipif(not REG.Registry().available(),
                               reason="needs the 7 git-ignored checkpoints of lanegame/demo/manifest.json")


@pytest.fixture(scope="module")
def reg():
    if not REG.Registry().available():
        pytest.skip("needs the 7 git-ignored checkpoints of lanegame/demo/manifest.json")
    return REG.Registry().load()


def chooser(t: int, legal):
    """A fixed stand-in for a human: a legal code picked by the turn number."""
    legal = list(legal)
    return legal[(t * 7 + 3) % len(legal)]


def play_human(s: S.GameSession, turns: int):
    for _ in range(turns):
        if s.state == S.STOPPED:
            break
        t = s.position + 1
        s.submit(t, chooser(t, s.env.legal(s.human)))


def snapshot(s: S.GameSession) -> dict:
    """Everything a rejected or read-only request must leave unchanged."""
    out = {"turns": copy.deepcopy(s.turns), "k": s.env.k, "digest": R.state_digest(s.env.state, s.env.champ, s.env.k),
           "rng": json.dumps(s.env.chance_rng.bit_generator.state, sort_keys=True), "pending": dict(s._pending),
           "state": s.state, "beliefs": copy.deepcopy(s.beliefs)}
    for seat, p in s._bots.items():
        if hasattr(p, "belief"):
            out[f"lp{seat}"] = p.belief.lp.copy().tolist()
            out[f"last{seat}"] = p.last
        if hasattr(p, "rng"):
            out[f"srng{seat}"] = json.dumps(p.rng.bit_generator.state, sort_keys=True)
        if hasattr(p, "u"):
            out[f"u{seat}"] = hashlib.sha256(p.u.tobytes()).hexdigest()
    return out


def keys_of(x, acc=None):
    acc = set() if acc is None else acc
    if isinstance(x, dict):
        for k, v in x.items():
            acc.add(k)
            keys_of(v, acc)
    elif isinstance(x, (list, tuple)):
        for v in x:
            keys_of(v, acc)
    return acc


# ---------------------------------------------------------------------------------------------- rule 7: identity
def test_the_manifest_is_m5s_inventory_and_names_every_file_the_cards_need():
    inv = json.load(open(os.path.join("results", "m5", "inventory.json"), encoding="utf-8"))["networks"]
    by_path = {}
    for v in inv.values():
        by_path.setdefault(v["path"], set()).add(v["file_sha256"])
    files = REG.read_manifest()
    assert len(files) == 7
    for rel, d in files.items():
        assert by_path.get(rel) == {d["sha256"]}, rel
    r = REG.Registry()
    assert set(r.cards) == {"sp:0", "om:0"}
    assert set(r.cards["om:0"].dependencies) == set(REG.LIBRARY_FILES.values())
    from lanegame.experiments.rl.om2 import common as O2
    assert {k: v.replace("\\", "/") for k, v in O2.LEARNED.items()} == REG.LIBRARY_FILES
    assert REG.SCRIPTED_STYLES == O2.ARCHETYPES


def _fake_tree(tmp_path, tamper=None, missing=None):
    """The 7 manifest paths with stand-in bytes and a manifest of their hashes; optionally alter or drop one file."""
    files = {}
    for i, rel in enumerate(REG.read_manifest()):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        b = f"stand-in {i}".encode()
        p.write_bytes(b)
        files[rel] = {"sha256": hashlib.sha256(b).hexdigest(), "role": "test"}
    man = tmp_path / "manifest.json"
    man.write_text(json.dumps({"schema": REG.MANIFEST_SCHEMA, "version": 1, "files": files}), encoding="utf-8")
    if tamper:
        (tmp_path / tamper).write_bytes(b"altered")
    if missing:
        (tmp_path / missing).unlink()
    return str(tmp_path), str(man)


@pytest.mark.parametrize("what", ["tamper", "missing"])
def test_an_altered_or_missing_checkpoint_is_refused_before_anything_is_unpickled(tmp_path, monkeypatch, what):
    import torch
    calls = []
    monkeypatch.setattr(torch, "load", lambda *a, **k: calls.append(1) or (_ for _ in ()).throw(AssertionError))
    last = list(REG.read_manifest())[-1]                    # a library file, checked after the two finals
    root, man = _fake_tree(tmp_path, **({"tamper": last} if what == "tamper" else {"missing": last}))
    with pytest.raises(REG.ChecksumError, match="manifest|missing"):
        REG.Registry(root=root, manifest=man).load()
    assert calls == []                                      # every file is checked before the first torch.load


@need_ckpt
def test_a_tampered_real_checkpoint_copy_is_refused(tmp_path, monkeypatch):
    import shutil
    import torch
    for rel in REG.read_manifest():
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(rel, dst)
    p = tmp_path / REG.LIBRARY_FILES["gapper"]
    b = bytearray(p.read_bytes())
    b[len(b) // 2] ^= 1
    p.write_bytes(bytes(b))
    calls = []
    real = torch.load
    monkeypatch.setattr(torch, "load", lambda *a, **k: calls.append(1) or real(*a, **k))
    with pytest.raises(REG.ChecksumError):
        REG.Registry(root=str(tmp_path)).load()
    assert calls == []


def test_the_session_constants_are_m5s():
    from lanegame.experiments.rl.m5 import games as G
    assert (S.LANE_EXIT, S.READ_TURN, S.TURN_CAP) == (G.LANE_END, G.READ_TURN, G.T_EVAL)


# ---------------------------------------------------------------------------------------------- rule 5: research games
@need_ckpt
@pytest.mark.parametrize("name", ["h2h_om_s0_vs_sp_s0", "om_s0_vs_g_shove", "sp_s0_vs_g_shove"])
def test_the_seed0_research_recordings_reproduce_exactly_through_the_session_core(reg, name):
    path = os.path.join(TRACES, f"{name}.json")
    d = json.load(open(path, encoding="utf-8"))
    s, cmp = S.from_research(reg, path)
    assert [t[:2] for t in s.turns] == d["actions"]
    assert S.running_total(s.rewards) == d["payoff_A"]                       # exactly, as M5 summed it
    assert cmp.get("belief_max_abs_diff", 0.0) <= 1e-5                         # float32 records
    rec = s.recording()
    assert rec["provenance"]["kind"] == "research_record"
    assert rec["provenance"]["source"]["sha256"] == R.file_sha256(path)
    assert rec["seeds"]["chance"] == rec["seeds"]["policy"] == d["seed"]       # M5's convention
    assert S.replay(reg, rec)["digests_checked"] == 150
    if "lead_114_A" in d.get("m5", {}):
        assert s.marks["lead_114"]["lead_A"] == pytest.approx(d["m5"]["lead_114_A"], abs=1e-4)


@need_ckpt
def test_bot_seats_sample_as_m5_evaluated_them_and_do_not_play_greedily(reg):
    s = S.GameSession(reg, (S.Seat.model("sp:0"), S.Seat.model("om:0")), chance_seed=5, policy_seed=6)
    assert all(getattr(p, "greedy", None) is False for p in s._bots.values())
    s.run()
    modal_differs = 0
    re = S.GameSession.rebuild(reg, s._secret_record(), upto=0)
    for t, (a, b, _) in enumerate(s.turns):
        for seat, code in ((0, a), (1, b)):
            legal, p = re._bots[seat].probs(re.cfg, re.env.state, re.env.champ, seat, t)
            modal_differs += int(legal[int(np.argmax(p))] != code)
        re._resolve(None)
    assert modal_differs > 0                                    # some sampled actions are not the most probable


# ---------------------------------------------------------------------------------------------- rule 1: simultaneity
@need_ckpt
@pytest.mark.parametrize("opp", ["sp:0", "om:0"])
def test_the_opponent_code_is_fixed_before_the_human_acts_and_equals_a_fresh_replay(reg, opp):
    s = S.GameSession(reg, (S.Seat.human(), S.Seat.model(opp)), chance_seed=110_801, policy_seed=110_802)
    play_human(s, 11)
    while len(s.env.legal(1)) < 2 or s.forced() is not None:      # a turn where the opponent has a real choice
        play_human(s, 1)
    t = s.position + 1
    fixed = s._pending[1]
    rec = s._secret_record()
    legal = s.env.legal(0)
    assert len(legal) >= 16
    for code in legal:
        b = S.GameSession.rebuild(reg, rec, upto=t - 1)
        assert b._pending[1] == fixed                            # a fresh replay of the history computes the same code
        b.submit(t, code)
        assert b.turns[-1][1] == fixed                           # whatever the human plays, the opponent's code holds


@need_ckpt
def test_the_belief_updates_after_both_codes_with_the_pre_turn_state(reg):
    from lanegame.experiments.rl.m5 import players as PL
    s = S.GameSession(reg, (S.Seat.human(), S.Seat.model("om:0")), chance_seed=110_803, policy_seed=110_804)
    pol = s._bots[1]
    before = pol.belief.lp.copy()
    view = s.view()
    assert "belief" not in view and np.array_equal(pol.belief.lp, before)    # deciding changed nothing
    lane, m4, k = s.env.state, s.env.champ, s.env.k
    code = chooser(1, s.env.legal(0))
    ref = PL.OMPolicy(pol.net, 0, 110_804, 1, s.turn_cap)
    ref.observe_turn(s.cfg, lane, m4, k, (code, s._pending[1]))           # the M5 update on the pre-turn state
    s.submit(1, code)
    assert np.array_equal(pol.belief.lp, ref.belief.lp) and pol.last == ref.last


# ---------------------------------------------------------------------------------------------- rule 2: legality
@need_ckpt
@pytest.mark.parametrize("opp", ["sp:0", "om:0"])
def test_illegal_or_malformed_submissions_change_nothing(reg, opp):
    s = S.GameSession(reg, (S.Seat.human(), S.Seat.model(opp)), chance_seed=110_805, policy_seed=110_806)
    play_human(s, 6)
    t = s.position + 1
    illegal = [c for c in range(61) if c not in s.env.legal(0)]
    before = snapshot(s)
    for bad in illegal + [61, -1, 999, True, False, "3", 3.0, None, [1]]:
        with pytest.raises(S.IllegalAction):
            s.submit(t, bad)
        assert snapshot(s) == before
    for bad_turn in (t + 1, 0, -1, True, "1", 1.0):
        with pytest.raises(S.TurnOutOfOrder):
            s.submit(bad_turn, s.env.legal(0)[0])
        assert snapshot(s) == before
    s.submit(t, s.env.legal(0)[0])                               # and the session then plays on normally
    assert s.position == t


# ---------------------------------------------------------------------------------------------- rule 3: isolation
@need_ckpt
def test_interleaved_sessions_equal_sequential_ones(reg):
    def make():
        return [S.GameSession(reg, (S.Seat.human(), S.Seat.model("sp:0")), chance_seed=11, policy_seed=12),
                S.GameSession(reg, (S.Seat.human(), S.Seat.model("om:0")), chance_seed=11, policy_seed=13),
                S.GameSession(reg, (S.Seat.model("om:0"), S.Seat.human()), chance_seed=14, policy_seed=15),
                S.GameSession(reg, (S.Seat.model("om:0"), S.Seat.scripted("g_bully")), chance_seed=16,
                              policy_seed=17, turn_cap=60)]

    def step(s):
        if s.human is None:
            s.advance(1)
        else:
            play_human(s, 1)

    seq = make()
    for s in seq:
        for _ in range(60):
            step(s)
    inter = make()
    for _ in range(60):
        for s in inter:
            step(s)
    for a, b in zip(seq, inter):
        a.stop(), b.stop()
        ra, rb = a.recording(), b.recording()
        assert ra == rb and len(ra["turns"]) == 60


# ---------------------------------------------------------------------------------------------- rule 4: duplicates
@need_ckpt
def test_a_duplicate_submission_resolves_once_and_a_conflicting_one_is_refused(reg):
    s = S.GameSession(reg, (S.Seat.human(), S.Seat.model("om:0")), chance_seed=110_807, policy_seed=110_808)
    play_human(s, 4)
    t = s.position + 1
    code = chooser(t, s.env.legal(0))
    first = s.submit(t, code)
    after = snapshot(s)
    again = s.submit(t, code)
    assert snapshot(s) == after and s.position == t
    assert again["turn"] == first["turn"] and again["position"] == first["position"]
    other = next(c for c in range(61) if c != code)
    with pytest.raises(S.TurnConflict):
        s.submit(t, other)
    old = s.submit(2, s.turns[1][0])                              # an older duplicate is answered by replay
    assert old["turn"]["codes"] == s.turns[1][:2] and snapshot(s) == after
    with pytest.raises(S.TurnOutOfOrder):
        s.submit(t + 2, code)


@need_ckpt
def test_concurrent_duplicate_submissions_resolve_the_turn_once(reg):
    s = S.GameSession(reg, (S.Seat.human(), S.Seat.model("sp:0")), chance_seed=110_809, policy_seed=110_810)
    code = s.env.legal(0)[0]
    out, errs = [], []
    go = threading.Barrier(8)

    def worker():
        go.wait()
        try:
            out.append(s.submit(1, code)["turn"])
        except Exception as e:                                   # pragma: no cover - would fail the test
            errs.append(e)
    th = [threading.Thread(target=worker) for _ in range(8)]
    for x in th:
        x.start()
    for x in th:
        x.join()
    assert not errs and s.position == 1 and len(s.turns) == 1
    assert all(o == out[0] for o in out)


# ---------------------------------------------------------------------------------------------- rule 5: replay
@need_ckpt
@pytest.mark.parametrize("opp", ["sp:0", "om:0"])
def test_a_full_human_session_replays_bit_for_bit_and_a_changed_record_does_not(reg, opp):
    s = S.GameSession(reg, (S.Seat.human(), S.Seat.model(opp)), chance_seed=110_811, policy_seed=110_812)
    play_human(s, 150)
    assert s.state == S.STOPPED and s.stop_reason == "turn_cap" and s.position == 150
    rec = s.recording()
    assert rec["seeds"]["policy"] == 110_812 and rec["provenance"]["kind"] == "human_game"
    assert S.replay(reg, json.loads(json.dumps(rec)))["digests_checked"] == 150
    bad = copy.deepcopy(rec)
    t = next(i for i, x in enumerate(bad["turns"]) if x[1] != 45)
    bad["turns"][t][1] = next(c for c in (0, 1, 2, 3) if c != bad["turns"][t][1])
    with pytest.raises((R.ReplayMismatch, ValueError)):
        S.replay(reg, bad)
    bad = copy.deepcopy(rec)
    bad["turns"][40][2] = "0" * 16
    with pytest.raises(R.ReplayMismatch):
        S.replay(reg, bad)


@need_ckpt
def test_new_simulations_are_labelled_and_replay(reg):
    s = S.simulate(reg, S.Seat.model("om:0"), S.Seat.model("sp:0"), turns=40, chance_seed=1, policy_seed=2)
    rec = s.recording()
    assert rec["provenance"]["kind"] == "new_simulation" and "not evidence" in rec["provenance"]["note"]
    assert rec["seeds"] == {"chance": 1, "chance_chosen": True, "policy": 2, "policy_rule": "given by the caller"}
    assert S.replay(reg, rec)["turns"] == 40
    with pytest.raises(ValueError):
        S.simulate(reg, S.Seat.human(), S.Seat.model("sp:0"))


# ---------------------------------------------------------------------------------------------- hidden information
@need_ckpt
def test_views_never_carry_the_stored_code_or_a_secret(reg):
    s = S.GameSession(reg, (S.Seat.human(), S.Seat.model("om:0")))
    play_human(s, 9)
    v = json.dumps(s.view(), sort_keys=True)
    keys = keys_of(s.view())
    assert not keys & {"pending", "probs", "logits", "probabilities", "chance_uniforms", "u", "belief", "beliefs"}
    sd = s.seeds()
    assert sd["chance"] is None and sd["policy"] is None and not sd["chance_chosen"]
    other = next(c for c in s.env.legal(1) if c != s._pending[1]) if len(s.env.legal(1)) > 1 else s._pending[1]
    s._pending[1] = other                                         # a different secret changes no visible byte
    assert json.dumps(s.view(), sort_keys=True) == v
    rec = s.recording()
    assert rec["seeds"]["chance"] is None and rec["seeds"]["policy"] is None and "beliefs" not in rec
    res = s._result(s.position)
    for part in (s.history(), res["turn"], res["position"]):
        assert not keys_of(part) & {"belief", "beliefs", "chance", "gank_draws", "probs", "logits", "seeds"}
    assert not keys_of(res) & {"belief", "beliefs", "probs", "logits"} and res["view"]["seeds"] == sd
    with pytest.raises(S.HiddenInformation):
        S.GameSession.rebuild(reg, rec)


@need_ckpt
def test_history_cannot_change_the_session_or_reach_past_the_resolved_turns(reg):
    s = S.GameSession(reg, (S.Seat.human(), S.Seat.model("om:0")), chance_seed=110_813, policy_seed=110_814)
    play_human(s, 25)
    before = snapshot(s)
    full = s.history()
    assert [p["p"] for p in full["positions"]] == list(range(26))
    assert [t["t"] for t in full["turns"]] == list(range(1, 26))
    for a in (0, 7, 24, 25):
        part = s.history(a, 25)
        assert part["positions"] == full["positions"][a:] and part["turns"] == full["turns"][a:]
    assert snapshot(s) == before
    for bad in ((0, 26), (5, 3), (-1, 2)):
        with pytest.raises((S.HiddenInformation, ValueError)):
            s.history(*bad)
    assert snapshot(s) == before
    assert full["positions"][25] == s.view()["current"]           # returning to the current turn restores it
    t = s.position + 1
    s.submit(t, chooser(t, s.env.legal(0)))                       # and the game continues where it was
    assert s.position == 26


@need_ckpt
def test_belief_modes(reg):
    hidden = S.GameSession(reg, (S.Seat.human(), S.Seat.model("om:0")), chance_seed=3, policy_seed=4)
    shown = S.GameSession(reg, (S.Seat.human(), S.Seat.model("om:0")), chance_seed=3, policy_seed=4,
                          belief_mode="shown")
    for s in (hidden, shown):
        play_human(s, 12)
    assert "belief" not in hidden.view() and "beliefs" not in hidden.history() and "beliefs" not in hidden.recording()
    sv = shown.view()["belief"]["B"]
    assert len(sv["posterior"]) == 13 and sv["types"][-2:] == ["mixed", "unknown"]
    assert len(shown.history()["beliefs"]["B"]["posteriors"]) == 13          # positions 0..12
    assert hidden.turns == shown.turns                                        # the mode changes what is shown only
    hidden.stop()
    rec = hidden.recording()
    assert rec["belief_mode"] == "hidden" and len(rec["beliefs"]["posteriors"]["B"]) == 13
    assert rec["beliefs"]["posteriors"]["B"] == shown.recording()["beliefs"]["posteriors"]["B"]


# ---------------------------------------------------------------------------------------------- rule 10 and 11
@need_ckpt
def test_stopping_declares_no_winner_and_reveals_the_seeds(reg):
    s = S.GameSession(reg, (S.Seat.human(), S.Seat.model("sp:0")))
    play_human(s, 20)
    summ = s.stop()
    assert summ["stop_reason"] == "visitor" and summ["turns"] == 20
    import re
    text = json.dumps(summ).lower().replace("declares no winner", "")
    assert not re.search(r"(winner|wins?|won|lose|loses|lost|victory|defeat)", text)
    assert s.seeds()["policy"] is not None and s.seeds()["chance"] is not None
    with pytest.raises(S.SessionStopped):
        s.submit(21, s.env.legal(0)[0])
    assert S.replay(reg, s.recording())["turns"] == 20


@need_ckpt
def test_the_lane_exit_marks(reg):
    from lanegame.diagnostics import lane_exit as LX
    s = S.GameSession(reg, (S.Seat.model("sp:0"), S.Seat.model("om:0")), chance_seed=21, policy_seed=22)
    s.advance(94)
    assert s.marks["lane_exit"]["position"] == 94 and "lead_114" not in s.marks
    assert s.marks["lane_exit"]["panel"]["D0"] == pytest.approx(LX.lead(s.env.champ, s.cfg), abs=1e-4)
    s.advance(20)
    m = s.marks["lead_114"]
    assert m["position"] == 114 and m["lead_A"] == pytest.approx(S.running_total(s.rewards[:114]), abs=1e-6)
    assert m["lead_A"] == pytest.approx(LX.lead(s.env.champ, s.cfg), abs=1e-6)    # from s0 the lead is the reward sum
    assert "example" in m["note"]


# ---------------------------------------------------------------------------------------------- frames and the menu
@pytest.fixture(scope="module")
def viewer_rec():
    from lanegame.viewer import session as VS
    setup, actions, _ = VS.load_recording(os.path.join(TRACES, "om_s0_vs_g_shove.json"))
    return VS.play(setup, prefix=actions, prefix_note="recorded")


def test_frames_equal_the_viewers_and_hold_only_per_tick_facts_in_ticks(viewer_rec):
    from lanegame.viewer import session as VS
    rec = viewer_rec
    cfg = rec.cfg
    pos = [F.position(cfg, rec.lanes[i], rec.champs[i], rec.k0 + i, i) for i in range(rec.n + 1)]
    for i in range(rec.n + 1):
        for s, n in enumerate("AB"):
            v = VS.summary(rec, i, s)
            c = pos[i]["champions"][n]
            assert (c["level"], c["hp"], c["xp"]) == (v["level"], v["hp"], v["xp"])
            assert c["gold_earned"] == pytest.approx(v["gold_earned"], abs=1e-4)
    for i in range(rec.n):
        info = rec.turns[i].info
        trn = F.turn(cfg, info, pos[i], pos[i + 1])
        assert len(trn["ticks"]) == cfg.ticks_per_turn
        for f in range(1, cfg.ticks_per_turn + 1):
            fv, tk = VS.frame_view(rec, i, f), trn["ticks"][f - 1]
            assert tk["packets"] == {n: [[int(x), int(h), bool(b)] for x, h, b in fv["packets"][n]] for n in "AB"}
            assert tk["contact"] == fv["contact"]
            for s, n in enumerate("AB"):
                if not info["away"][s]:                          # the viewer draws a returning champion at its tower
                    assert tk["cells"][n] == fv["champions"][n]["pos_global"]
                else:
                    assert tk["cells"][n] is None
        tick_keys = keys_of(trn["ticks"])
        assert not tick_keys & {"trade", "death", "gank", "recall", "plate", "level_up", "tower_shot"}
        assert not keys_of(trn) & {"chance", "gank_draws"}
        assert pos[i + 1]["lead_A"] == pytest.approx(sum(rec.turns[j].info["reward"] for j in range(i + 1)), abs=1e-6)


def test_the_menu_marks_exactly_the_engines_legal_codes_and_explains_the_rest(viewer_rec):
    from lanegame.champions import champion_legal_actions
    rec = viewer_rec
    cfg = rec.cfg
    seen_reasons = set()
    for i in range(rec.n + 1):
        lane, m4 = rec.lanes[i], rec.champs[i]
        for s in (0, 1):
            legal = set(champion_legal_actions(lane, m4, s, cfg))
            m = ACT.menu(lane, m4, s, cfg)                       # asserts equality with the engine internally
            assert set(m["legal"]) == legal
            for code in range(61):
                why = ACT.explain(lane, m4, s, cfg, code)
                assert (why is None) == (code in legal)
                if why:
                    seen_reasons.add(why.split(":")[0][:40])
    assert len(seen_reasons) >= 4                                # away, Pressure, Recall, Vision at least


# ---------------------------------------------------------------------------------------------- the local service
@need_ckpt
def test_the_service_plays_a_live_game_and_maps_refusals_to_statuses(reg):
    from lanegame.demo.service import DemoService, ServiceError
    svc = DemoService(registry=reg)
    assert svc.status()["ready"] and {m["id"] for m in svc.status()["models"]} == {"sp:0", "om:0"}
    g = svc.create("om:0", "hidden")
    sid = g["session"]
    assert [p["p"] for p in g["history"]["positions"]] == [0]
    assert not keys_of(g) & {"belief", "beliefs", "pending"}
    code = g["view"]["menu"]["legal"][0]
    r = svc.submit(sid, 1, code)
    assert r["turn"]["t"] == 1 and r["turn"]["codes"][0] == code and r["view"]["position"] == 1
    for args, status in (((sid, 1, code + 1), 409), ((sid, 5, code), 409), ((sid, 2, 33), 422), ((sid, 2, True), 422),
                         (("0" * 32, 2, code), 404)):
        with pytest.raises(ServiceError) as e:
            svc.submit(*args)
        assert e.value.status == status
    with pytest.raises(ServiceError) as e:
        svc.history(sid, 0, 9)
    assert e.value.status == 403
    with pytest.raises(ServiceError) as e:
        svc.create("g_shove")
    assert e.value.status == 400
    out = svc.stop(sid)
    assert out["summary"]["stop_reason"] == "visitor" and len(out["recording"]["beliefs"]["posteriors"]["B"]) == 2
    with pytest.raises(ServiceError) as e:
        svc.submit(sid, 2, code)
    assert e.value.status == 410
    assert S.replay(reg, json.loads(json.dumps(out["recording"])))["turns"] == 1
    sim = svc.simulate("sp:0", "om:0", 12)
    assert sim["meta"]["provenance"]["kind"] == "new_simulation" and len(sim["turns"]) == 12
    assert len(sim["positions"]) == 13 and len(sim["beliefs"]["posteriors"]["B"]) == 13


def test_drawn_seeds_stay_exact_as_browser_json_numbers():
    assert all(0 <= S._draw_seed() < 2 ** 53 for _ in range(2000))
