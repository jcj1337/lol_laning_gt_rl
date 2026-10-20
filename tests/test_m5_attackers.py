"""M5e: the core attacker stage's recovery path and evaluation gate, the tolerant history reader, and the attacker
readouts beside S6 on synthetic records. No checkpoint is needed; the real-checkpoint path is readiness part C1."""

import json
import os

import numpy as np
import pytest

from lanegame.experiments.rl.m5 import analysis as AN
from lanegame.experiments.rl.m5 import attackers as AT
from lanegame.experiments.rl.m5 import common as C


def _dump(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def test_incomplete_attacker_runs_are_refused_then_preserved_and_replanned(tmp_path, monkeypatch):
    monkeypatch.setattr(AT, "AOUT", str(tmp_path / "attack"))
    d = AT.attacker_dir("om", 2)
    os.makedirs(d)
    with open(os.path.join(d, "history.jsonl"), "w", encoding="utf-8") as f:
        f.write(json.dumps({"update": 1}) + "\n")
    done = AT.attacker_dir("sp", 2)
    os.makedirs(done)
    _dump(os.path.join(done, "summary.json"), {"status": "complete"})
    with pytest.raises(SystemExit, match="incomplete"):
        AT.plan_runs(seeds=(2,))
    out = str(tmp_path / "out")
    todo = AT.plan_runs(seeds=(2,), restart_incomplete=True, out=out)
    assert [j[:3] for j in todo] == [("om", 2, AT.ATTACK_BUDGET)]          # the complete sp run is not repeated
    rec = C.load("restarts", out)["restarts"][-1]
    assert (rec["phase"], rec["run"], rec["updates_preserved"]) == ("attack_main", "attack_om_s2", 1)
    assert os.path.exists(os.path.join(rec["moved_to"], "history.jsonl")) and not os.path.exists(d)
    assert AT.plan_runs(seeds=(2,), out=out) == todo


def test_the_attacker_evaluation_needs_a_passed_or_resolved_check_and_unchanged_files(tmp_path):
    from lanegame.experiments.rl.m5 import health as HL
    out = str(tmp_path)
    with pytest.raises(SystemExit, match="no attacker check"):
        AT.require_ok(out=out)
    names = [f"{k}_s{s}" for k in AT.CORE_KINDS for s in C.SEEDS]
    ids = {}
    for n in names:
        p = tmp_path / f"{n}.pt"
        p.write_bytes(n.encode())
        ids[n] = {"path": str(p), "file_sha256": C.file_sha256(str(p))}
    rec = {"complete": names, "identities": ids,
           "checks": {f"{n}: no pre-registered stop condition": True for n in names}}
    C.save_json("attack_check_main", rec, out)
    assert AT.require_ok(out=out)
    rec["checks"]["om_s1: no pre-registered stop condition"] = False
    C.save_json("attack_check_main", rec, out)
    with pytest.raises(SystemExit, match="without a recorded resolution"):
        AT.require_ok(out=out)
    HL.resolve("attack_main", "om_s1: no pre-registered stop condition", "training property (not a defect)",
               "entropy fell slowly; no defect found", out=out)
    assert AT.require_ok(out=out)
    bad = {**rec, "checks": {**rec["checks"], "restart attack_om_s0 (x): reproduces the preserved history prefix": False}}
    C.save_json("attack_check_main", bad, out)
    with pytest.raises(SystemExit, match="without a recorded resolution"):
        AT.require_ok(out=out)
    C.save_json("attack_check_main", rec, out)
    (tmp_path / "sp_s0.pt").write_bytes(b"changed")
    with pytest.raises(SystemExit, match="differ from the check"):
        AT.require_ok(out=out)
    C.save_json("attack_check_main", {**rec, "complete": names[1:]}, out)
    with pytest.raises(SystemExit, match="incomplete or failed"):
        AT.require_ok(out=out)


def test_history_drops_only_a_truncated_last_line(tmp_path):
    from lanegame.experiments.rl.m5 import train as T
    p = tmp_path / "history.jsonl"
    p.write_text('{"update": 1}\n{"update": 2}\n{"upd', encoding="utf-8")
    assert [e["update"] for e in T.history(str(tmp_path))] == [1, 2]
    p.write_text('{"update": 1}\n{"upd\n{"update": 3}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        T.history(str(tmp_path))


def _cell(path, lead, late=0.0, N=4, T=500, opp_type=None):
    """A games.cell_job record whose focal 0 leads by `lead` at turn 114 (paid on turn 0) and earns `late` more on
    turn 200 (the discounted readout only)."""
    from lanegame.experiments.rl.m5 import games as G
    os.makedirs(os.path.dirname(path), exist_ok=True)
    R = np.zeros((N, 2, T), dtype=np.float32)
    R[:, :, 0] = lead + np.linspace(-0.01, 0.01, 2 * N).reshape(N, 2)
    if late:
        R[:, :, 200] = late
    pan = np.zeros((N, 2, len(G.PANEL_KEYS)), dtype=np.float32)
    pan[..., G.PANEL_KEYS.index("D0")] = 0.5 * lead
    out = {"f0__r": R, "f0__panel": pan, "f0__trace": np.zeros((N, 2, len(G.CHECK_TURNS), C.K), np.float32)}
    if opp_type is not None:
        lp = np.full((N, 2, len(G.CHECK_TURNS), C.K), -20.0, dtype=np.float32)
        lp[..., list(C.BELIEF_TYPES).index(opp_type)] = 0.0
        out["f0__opp_trace"] = lp
    np.savez(path, **out)


def test_attacker_readouts_beside_s6_have_the_registered_signs(tmp_path):
    root = str(tmp_path / "eval")
    att = str(tmp_path / "attack")
    a = {s: 1.0 + 0.1 * s for s in C.SEEDS}
    b = {s: 0.4 + 0.05 * s for s in C.SEEDS}
    for s in C.SEEDS:
        g = {**AN.seed_paths(s, root), **AN.attack_paths(s, att)}
        _cell(g["attack:om"][0], a[s], late=1.0, opp_type="unknown")
        _cell(g["attack:sp"][0], b[s])
        _cell(g["attack:om:other_arm"][0], 0.2)
        _cell(g["attack:sp:other_arm"][0], -0.1)
        _cell(g["attack:om:start"][0], 0.5)
        _cell(g["attack:sp:start"][0], 1.2)
        _cell(g["h2h"][0], -0.4, T=150)                      # om:s's lead against sp:s
        _cell(g["hist:om:start"][0], 0.2)
        _cell(g["hist:sp:start"][0], 0.9)
    AN.clear_cache()

    def paths_of(s):
        return {**AN.seed_paths(s, root), **AN.attack_paths(s, att)}
    e = {k: AN.endpoint(k, paths_of) for k in AN.ATTACK_ENDPOINTS if k != "X_table"}
    assert all(v["status"] == "complete" for v in e.values())
    for s in C.SEEDS:
        assert e["S6"]["per_seed"][str(s)] == pytest.approx(a[s] - b[s], abs=1e-6)
        assert e["X_om"]["per_seed"][str(s)] == pytest.approx(a[s], abs=1e-6)
        assert e["S6_disc"]["per_seed"][str(s)] == pytest.approx(a[s] + 0.99 ** 200 - b[s], abs=1e-5)
    assert e["T_om_other_gain"]["mean"] == pytest.approx(0.2 - (-0.4), abs=1e-6)
    assert e["T_sp_other_gain"]["mean"] == pytest.approx(-0.1 - 0.4, abs=1e-6)   # sp:s led om:s by +0.4
    assert e["T_om_start_gain"]["mean"] == pytest.approx(0.3, abs=1e-6)
    assert e["T_sp_start_gain"]["mean"] == pytest.approx(0.3, abs=1e-6)
    assert e["S6"]["class"] == "increase demonstrated" and e["X_om"]["class"].startswith("descriptive (positive")
    assert e["T_sp_other_gain"]["class"] == "negative (U < 0)"
    bel = AN.attack_belief(paths_of)
    assert bel["catch-all mass at 94"]["mean"] == pytest.approx(1.0) and bel["modal type unknown at 94"]["mean"] == 1.0
    pan = AN.attack_panel(paths_of)
    assert pan["attack:om"]["status"] == "complete"
    assert pan["attack:om"]["quantities"]["D0 (earned lead at 94)"]["mean"] == pytest.approx(0.5 * np.mean(list(
        a.values())), abs=1e-6)


def test_s6_completes_the_holm_family_with_fifteen_hypotheses():
    per = {"ni": {s: -0.6 + 0.01 * s for s in C.SEEDS}, "gain": {s: 8.0 + 0.1 * s for s in C.SEEDS},
           "exploit": {s: 0.01 * (s - 2.5) for s in C.SEEDS}}
    kinds = {"S1": "ni", "S2": "ni", "S3": "ni", "S4": "ni", "S5a": "gain", "S5b": "ni", "S6": "exploit"}
    eps = {k: {"status": "complete", "kind": v, "class": AN.CLASSIFIERS[v](AN.interval(per[v])),
               **AN.interval(per[v]), "per_seed": {str(s): x for s, x in per[v].items()}} for k, v in kinds.items()}
    out = AN.secondary(eps)
    assert out["holm"]["final"] and out["holm"]["m"] == 15 and not out["holm"].get("pending")
    assert out["endpoints"]["S6"]["claim"] == "no material difference"
    assert out["endpoints"]["S2"]["claim"] == "material loss"
    pending = AN.secondary({k: v for k, v in eps.items() if k != "S6"})
    assert pending["holm"]["m"] == 15 and len(pending["holm"]["pending"]) == 3


def test_the_trace_rule_picks_the_upper_median_game_by_the_focal_lead(tmp_path):
    from lanegame.experiments.rl.m5 import traces as TR
    R = np.zeros((3, 2, 150), dtype=np.float32)
    leads = np.array([[5.0, -1.0], [0.5, 2.0], [-3.0, 1.0]])            # sorted: -3, -1, 0.5, 1, 2, 5
    R[:, :, 0] = leads
    R[:, :, 120] = 100.0                                                  # after turn 114: ignored by the rule
    p = str(tmp_path / "cell.npz")
    np.savez(p, f0__r=R)
    sel = TR.median_game(p)
    assert (sel["pair"], sel["orientation"], sel["lead_114"], sel["games"]) == (2, 1, 1.0, 6)
    assert sel["cell_mean_lead_114"] == pytest.approx(leads.mean())
    names = [c[2] for c in TR.trace_cells()]
    assert len(names) == 15 and len(set(names)) == 15 and sum(n.startswith("attacker_") for n in names) == 12
