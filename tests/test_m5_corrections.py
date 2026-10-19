"""M5a correction pass (amendment 1, 2026-10-17): the amendment chain, the corrected classifications, headline and
multiplicity rules, completeness, the S8 / monitoring / panel analysis on synthetic records, runtime health monitoring,
the coordinator's failure policy, the evaluation gate and the attacker job definitions. No checkpoint is needed except
where marked."""

import gzip
import json
import os

import numpy as np
import pytest

from lanegame.experiments.rl.m5 import analysis as AN
from lanegame.experiments.rl.m5 import common as C


# ---------------------------------------------------------------------------------------------- the amendment chain
def _write(p, text):
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def test_amendment_chain_validates_the_active_version_and_rejects_undocumented_changes(tmp_path):
    out, doc, des = str(tmp_path / "out"), str(tmp_path / "prereg.md"), str(tmp_path / "design.md")
    _write(doc, "# protocol v0\n")
    _write(des, "# design\n")
    d = C.write_prereg(out=out, doc=doc, design=des)
    v0 = d["record"]["sha256"]
    assert C.require_prereg(out, doc)["record"]["sha256"] == v0 and C.active_version(d)["version"] == 0
    _write(doc, "# protocol v1\n")                                   # an undocumented edit
    with pytest.raises(SystemExit, match="undocumented change"):
        C.require_prereg(out, doc)
    am = str(tmp_path / "amendment_1.md")
    _write(am, "why\n")
    with pytest.raises(SystemExit, match="not archived"):            # the old version was not archived before the edit
        C.amend("r", am, out=out, doc=doc, design=des)
    _write(doc, "# protocol v0\n")
    C.archive_active(out, doc)
    _write(doc, "# protocol v1\n")
    d = C.amend("correction pass", am, "training", out=out, doc=doc, design=des)
    assert d["record"]["sha256"] == v0                                # the original frozen record is preserved
    a = d["amendments"][0]
    assert a["id"] == 1 and a["previous_sha256"] == v0 and a["reason"] == "correction pass" and a["date"]
    assert C.active_version(d)["sha256"] == a["document"]["sha256"] != v0
    assert C.require_prereg(out, doc)["active"]["version"] == 1
    assert open(C.version_path(v0, out), encoding="utf-8").read() == "# protocol v0\n"
    with pytest.raises(SystemExit, match="unchanged"):
        C.amend("again", am, out=out, doc=doc, design=des)
    _write(am, "edited rationale\n")                                  # the amendment's own document is pinned too
    with pytest.raises(SystemExit, match="amendment document"):
        C.require_prereg(out, doc)
    _write(am, "why\n")
    _write(C.version_path(v0, out), "tampered\n")                     # so is the archive of the superseded version
    with pytest.raises(SystemExit, match="not archived intact"):
        C.require_prereg(out, doc)
    _write(C.version_path(v0, out), "# protocol v0\n")
    C.archive_active(out, doc)
    _write(doc, "# protocol v2\n")
    d = C.amend("second", am, out=out, doc=doc, design=des)
    assert [x["id"] for x in d["amendments"]] == [1, 2] and not C.verify_chain(d, out)
    assert d["amendments"][1]["previous_sha256"] == d["amendments"][0]["document"]["sha256"]
    assert C.require_prereg(out, doc)["active"]["version"] == 2


def test_the_real_preregistration_chain_verifies():
    d = C.load("preregistration")
    if not d:
        pytest.skip("no pre-registration record")
    assert not C.verify_chain(d)
    assert C.require_prereg()


# ---------------------------------------------------------------------------------------------- classifications
def _st(v):
    return AN.interval(dict(enumerate(v)))


def test_gain_classes_separate_ruled_out_from_inconclusive():
    assert AN.classify_gain(_st([2.0, 2.4, 1.8, 2.2, 2.1, 1.9])) == "demonstrated"
    assert AN.classify_gain(_st([0.10, 0.12, 0.08, 0.11, 0.09, 0.10])) == "detected, below materiality"
    assert AN.classify_gain(_st([0.05, 0.4, 0.05, 0.3, 0.02, 0.2])) == "detected, materiality unresolved"
    assert AN.classify_gain(_st([-0.05, 0.05, 0.0, 0.02, -0.03, 0.01])) == "no material gain"
    assert AN.classify_gain(_st([-1.0, -1.1, -0.9, -1.05, -0.95, -1.0])) == "loss"
    assert AN.classify_gain(_st([-0.5, 1.5, -0.2, 1.2, 0.0, 0.8])) == "inconclusive"
    assert AN.classify_gain(_st([0.3, 1.5, 0.1, 1.2, -0.4, 0.8])) == "inconclusive"   # mean >= 0.25, L <= 0


def test_headline_never_turns_an_inconclusive_gain_into_no_material_gain():
    h = AN.headline("inconclusive", "preserved", "preserved")
    assert h["outcome"] == 6 and h["name"] == "gain not demonstrated" and h["competence"] == "preserved"
    assert AN.headline("detected, materiality unresolved", "inconclusive", "preserved")["outcome"] == 6
    assert AN.headline("no material gain", "inconclusive", "preserved")["outcome"] == 4
    assert AN.headline("detected, below materiality", "preserved", "preserved")["outcome"] == 4
    assert AN.headline("loss", "preserved", "preserved")["outcome"] == 4
    assert AN.headline("inconclusive", "material loss", "preserved")["outcome"] == 5
    assert AN.headline("no material gain", "preserved", "material loss")["outcome"] == 5
    assert AN.headline("demonstrated", "preserved (a small loss detected)", "preserved")["outcome"] == 1
    assert AN.headline("demonstrated", "material loss", "preserved")["outcome"] == 2
    h = AN.headline("demonstrated", "loss, materiality unresolved", "preserved")
    assert h["outcome"] == 3 and h["competence"] == "loss detected, materiality unresolved"
    assert "S6" in h["scope"] and "S8" in h["scope"]                  # the headline states what it does not cover


def test_interval_rules_and_one_sided_tests_agree():
    """L > c  <=>  the one-sided test of H0 mean <= c rejects at 0.025 (df 5); U < c likewise; TOST <=> both bounds."""
    rng = np.random.default_rng(0)
    for _ in range(300):
        st = _st(rng.normal(rng.normal(0, 0.4), rng.uniform(0.05, 0.6), 6))
        for c in (-0.25, 0.0, 0.25):
            assert (st["L"] > c) == (AN.p_one_sided(st, c, ">") < 0.025)
            assert (st["U"] < c) == (AN.p_one_sided(st, c, "<") < 0.025)
        assert (st["L"] > -0.25 and st["U"] < 0.25) == (AN.p_equivalence(st) < 0.025)
    st = _st([0.3, 0.2, 0.4, 0.25, 0.35, 0.3])
    assert AN.p_one_sided(st, 0.0, ">") < 0.001 < AN.p_one_sided(st, 0.0, "<")


def test_holm_needs_the_whole_family_and_secures_only_what_pending_results_cannot_change():
    assert AN.holm({"a": 0.001, "b": 0.02, "c": 0.04}) == {"a": True, "b": True, "c": True}
    assert AN.holm({"a": 0.001, "b": 0.03, "c": 0.04}) == {"a": True, "b": False, "c": False}
    with pytest.raises(ValueError):
        AN.holm({"a": 0.001, "b": float("nan")})
    s = AN.holm_status({"a": 0.001, "b": 0.02, "c": None, "d": None, "e": 0.3})
    assert not s["final"] and s["status"]["a"] == "secured" and s["status"]["b"].startswith("pending")
    assert s["status"]["c"] == "pending (no data)" and s["status"]["e"] == "not rejectable"
    for pend in ([0.0], [1.0], [0.0, 0.0, 0.0], [0.9, 0.01]):              # brute force: secured stays rejected,
        full = {"a": 0.001, "b": 0.02, "e": 0.3, **{f"x{i}": p for i, p in enumerate(pend)}}   # p > alpha never is
        r = AN.holm(full)
        assert r["a"] and not r["e"]
    assert AN.holm_status({"a": 0.001, "b": 0.02})["final"]
    assert len(AN.SECONDARY_HYPOTHESES) == 15                         # the registered family (S1-S6)
    assert {h[0] for h in AN.SECONDARY_HYPOTHESES} == {"S1", "S2", "S3", "S4", "S5a", "S5b", "S6"}


def test_secondary_claims_need_the_interval_and_holm():
    def ep(kind, vals):
        st = _st(vals)
        return {"status": "complete", "kind": kind, "class": AN.CLASSIFIERS[kind](st), **st}
    eps = {k: ep("ni", [0.0, 0.02, -0.01, 0.01, 0.0, -0.02]) for k in ("S1", "S2", "S3", "S4", "S5b")}
    eps["S5a"] = ep("gain", [1.0, 1.1, 0.9, 1.05, 0.95, 1.0])
    out = AN.secondary(eps)                                           # S6 (attackers) not run: the family is pending
    assert not out["holm"]["final"] and "S6: increase" in out["holm"]["pending"]
    assert out["endpoints"]["S6"]["claim"] == "not available"
    assert out["endpoints"]["S1"]["claim"].startswith("preserved")
    assert out["endpoints"]["S1"]["determined"] and "determined" in out["endpoints"]["S1"]["claim"]
    eps["S6"] = ep("exploit", [0.01, -0.01, 0.0, 0.02, -0.02, 0.0])
    out = AN.secondary(eps)
    assert out["holm"]["final"] and out["endpoints"]["S6"]["claim"] == "no material difference"
    eps["S2"] = ep("ni", [-0.6, 0.6, -0.3, 0.4, -0.5, 0.5])
    assert AN.secondary(eps)["endpoints"]["S2"]["claim"] == "inconclusive"
    declined = AN.secondary({k: v for k, v in eps.items() if k != "S6"}, declined=("S6",))
    assert declined["holm"]["final"]                                  # a declined stage enters with p = 1


def test_replication_never_counts_a_missing_seed():
    r = AN.replication({0: 0.1, 1: 0.15, 2: 0.3, 3: -0.2, 4: 0.0, 5: 0.19})
    assert r["n_holds"] == 5 and r["replicates"] and r["complete"]
    assert not AN.replication({0: 0.3, 1: 0.25, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0})["replicates"]
    p = AN.replication({0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0})
    assert p["verdict"] == "partial (4/6 seeds)" and not p["replicates"]
    assert AN.replication({0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0})["verdict"] == "replicates"
    assert AN.replication({0: 0.5, 1: 0.5, 2: 0.0})["verdict"] == "does not replicate"


# ---------------------------------------------------------------------------------------------- synthetic M5d records
def _cell(path, n_focal, values, T, rng, N=4, panel_scale=0.8):
    """A games.cell_job record: focal f's lead at every horizon is values[f] (+ game noise) on turn 0."""
    from lanegame.experiments.rl.m5 import games as G
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = {}
    for f in range(n_focal):
        R = np.zeros((N, 2, T), dtype=np.float32)
        R[:, :, 0] = values[f] + rng.normal(0, 0.05, (N, 2))
        pan = np.zeros((N, 2, len(G.PANEL_KEYS)), dtype=np.float32)
        pan[..., G.PANEL_KEYS.index("D0")] = panel_scale * values[f]
        pan[..., G.PANEL_KEYS.index("level_diff")] = 0.1 * values[f]
        pan[..., G.PANEL_KEYS.index("status_A")] = (f == 0)
        out[f"f{f}__r"], out[f"f{f}__panel"] = R, pan
        out[f"f{f}__trace"] = np.zeros((N, 2, 8, 13), dtype=np.float32)
    np.savez(path, **out)


EFFECTS = {"fam:S_in": 1.0, "fam:S_out": 0.8, "fam:M": -0.05, "fam:C_hist": 0.0, "fam:C_cross": 0.02,
           "long:C_hist": 0.0, "long:C_cross": 0.0, "mid:S_in": 0.5, "mid:S_out": 0.5, "mid:C_hist": 0.0}


def _fixture(root, rng, skip=()):
    for s in C.SEEDS:
        g = AN.seed_paths(s, root)
        for grp, ps in g.items():
            for p in ps:
                if p in skip:
                    continue
                noise = rng.normal(0, 0.03)
                T = 500 if grp in ("long:C_hist", "long:C_cross", "h2h_long") else 150
                if grp in EFFECTS:
                    nf = 2 if grp.startswith("mid:") else 3
                    e = EFFECTS[grp] + noise
                    _cell(p, nf, [e, 0.0, -0.3][:nf], T, rng)
                else:
                    _cell(p, 1, [0.01 + noise], T, rng)


def test_analysis_end_to_end_on_synthetic_records(tmp_path):
    root, out = str(tmp_path / "eval"), str(tmp_path / "out")
    _fixture(root, np.random.default_rng(1))
    a = AN.analyse(root=root, attack_root=str(tmp_path / "none"), out=out)
    e = a["endpoints"]
    assert all(e[k]["status"] == "complete" for k in ("P1", "P2a", "P2b", "S1", "S2", "S3", "S4", "S5a", "S5b"))
    assert e["P1"]["class"] == "demonstrated" and abs(e["P1"]["mean"] - 0.9) < 0.1
    assert e["P2a"]["class"] == "preserved" and e["P2b"]["class"] == "preserved"
    assert a["headline"]["outcome"] == 1
    assert e["S6"]["status"] == "missing" and not a["secondary"]["holm"]["final"]
    assert "historical" in e["S5b"]["description"].lower() and "not P2a" in e["S5b"]["description"]
    p = a["panel"]["fam:S_in"]
    assert p["status"] == "complete"
    q = p["quantities"]
    assert abs(q["D0 (earned lead at 94)"]["om - sp"]["mean"] - 0.8 * 1.0) < 0.1
    assert abs(q["E_20 - D0"]["om"]["mean"] - 0.2) < 0.1 and q["focal away (share)"]["om"]["mean"] == 1.0
    assert a["versus_start"]["om - start | S_in"]["mean"] > 1.0
    assert os.path.exists(os.path.join(out, "analysis_main.json"))


def test_switch_windows_and_belief_readouts_on_synthetic_records(tmp_path):
    from lanegame.experiments.rl.m5 import evaluate as EV
    root = str(tmp_path / "eval")
    rng = np.random.default_rng(6)
    _fixture(root, rng)
    types = list(C.BELIEF_TYPES)
    for s in C.SEEDS:
        g = AN.seed_paths(s, root)
        for j, opp in enumerate(EV.families(s)["M"]):          # om gains 0.01 per turn only after the switch
            d = dict(np.load(g["fam:M"][j]))
            t = AN._switch_turn(opp[1]) or 0
            d["f0__r"], d["f1__r"] = np.zeros_like(d["f0__r"]), np.zeros_like(d["f1__r"])
            d["f0__r"][:, :, t:] = 0.01
            np.savez(g["fam:M"][j], **d)
        for j, opp in enumerate(EV.families(s)["S_in"]):       # the belief puts its mass on the true type
            d = dict(np.load(g["fam:S_in"][j]))
            lp = np.full(d["f0__trace"].shape, -5.0, dtype=np.float32)
            lp[..., types.index(opp[1])] = 0.0
            d["f0__trace"] = lp
            np.savez(g["fam:S_in"][j], **d)
    AN.clear_cache()
    paths_of = lambda s: AN.seed_paths(s, root)
    sw = AN.switch_windows(paths_of)
    w = sw["its:g_farm:40:o"]
    assert abs(w["before the switch (0-39)"]["mean"]) < 1e-6 and w["switch + 0-19 (40-59)"]["mean"] == pytest.approx(0.2)
    assert "turns 0-113" in sw["mix_bully_farm"]
    b = AN.belief_mechanism(paths_of)
    assert b["S_in | g_farm"]["identified at 10"]["mean"] == 1.0 and b["S_in | g_farm"]["true-type mass at 94"]["mean"] > 0.9
    assert "catch-all mass at 20" in b["M | mix_bully_farm"]


def test_missing_records_leave_endpoints_partial_and_the_headline_unavailable(tmp_path):
    root = str(tmp_path / "eval")
    skip = {AN.seed_paths(5, root)["fam:S_in"][2]}
    _fixture(root, np.random.default_rng(2), skip)
    a = AN.analyse(root=root, attack_root=str(tmp_path / "none"), save=False)
    e = a["endpoints"]["P1"]
    assert e["status"] == "partial (5/6)" and e["class"] == "partial: not classified" and "5" in e["missing"]
    assert a["headline"]["outcome"] is None and "incomplete" in a["headline"]["name"]
    assert a["endpoints"]["P2a"]["status"] == "complete"
    assert a["panel"]["fam:S_in"]["status"] == "partial (5/6)"


def test_report_tables_from_synthetic_analysis(tmp_path, monkeypatch):
    from lanegame.experiments.rl.m5 import report as RP
    root, out = str(tmp_path / "eval"), str(tmp_path / "out")
    _fixture(root, np.random.default_rng(3))
    a = AN.analyse(root=root, attack_root=str(tmp_path / "none"), save=False)
    a = json.loads(json.dumps(a, default=float))
    monkeypatch.setattr(C, "load", lambda name, o=None: a if name == "analysis_main" else None)
    text = RP.analysis_section()
    assert "Headline: outcome 1" in text and "| P1 |" in text and "NOT final" in text
    assert "D0 (earned lead at 94)" in text and "S5b" in text


def test_monitoring_and_convergence_on_synthetic_records(tmp_path):
    from lanegame.experiments.rl.m5 import evaluate as EV
    root = str(tmp_path / "eval")
    rng = np.random.default_rng(4)
    for a in C.ARMS:
        for s in C.SEEDS:
            for n in C.CURVE_POINTS:
                for q in range(len(EV.REF_PANEL)):
                    _cell(os.path.join(root, "curve", f"{a}_s{s}", f"{n}_{q}.npz"), 1, [n / 1e7], 150, rng)
            for fname, v in ((f"vs_{C.BUDGET // 4}.npz", 1.0), (f"vs_{C.MIDPOINT}.npz", 0.6),
                             (f"vs_{C.P75}.npz", 0.5 if a == "om" else 0.0 + rng.normal(0, 0.05)),
                             ("vs_start.npz", 2.0)):
                _cell(os.path.join(root, "history", f"{a}_s{s}", fname), 1, [v + rng.normal(0, 0.02)], 500, rng)
    m = AN.monitoring(root=root, save=False)
    assert m["convergence"]["om"]["verdict"].startswith("still improving")
    assert m["convergence"]["sp"]["verdict"] == "not shown to be still improving"
    assert m["curve_summary"][f"om|{C.BUDGET}"]["panel mean"]["mean"] > m["curve_summary"]["om|2048000"]["panel mean"]["mean"]
    assert not m["cycling"]["om"]["flag"]


# ---------------------------------------------------------------------------------------------- S8 on synthetic records
def _battery_record(root, tag, fam, idx, proto, horizon, gaps, K=4, rng=None):
    from lanegame.experiments.rl.m45c import battery as B
    from lanegame.experiments.rl.m45c.horizon_report import intended
    rng = rng or np.random.default_rng(0)
    cands = {"policy": None, **{n: list(s) for n, s in B.candidates(fam).items()}}
    cps = B.checkpoints(horizon)
    GQ = {}
    for n in cands:
        base = 0.0 if n == "policy" else gaps.get(n, -0.5)
        row = {str(c): (base + rng.normal(0, 0.01, K)).tolist() for c in cps}
        GQ[n] = {"G": row, "Q": row}
    first = {n: [intended(s) if s else 0] * K for n, s in cands.items()}
    rec = {"family": fam, "protocol": proto, "horizon": horizon, "index": idx, "GQ": GQ, "candidates": cands,
           "first_codes": first, "K": K}
    p = B.job_path(tag, fam, idx, proto, horizon, root)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with gzip.open(p, "wt", encoding="utf-8") as f:
        json.dump(rec, f)


def test_battery_family_gap_coverage_and_the_validation_path(tmp_path):
    from lanegame.experiments.rl.m45c import battery as B
    from lanegame.experiments.rl.m5 import strategic as ST
    root = str(tmp_path / "battery")
    fam = "H_zone_c"                                                   # constructed: 12 cells, every cell validated
    cand = next(iter(B.candidates(fam)))
    n = len(B.GRIDS[fam])
    rng = np.random.default_rng(5)
    for i in range(n):
        for p in B.FAMILIES[fam][3]:
            _battery_record(root, "om_s0", fam, i, p, 300 if p == "policy" else 50, {cand: 0.1}, rng=rng)
    jobs = B.load("om_s0", root)
    g = ST.family_gap(jobs, fam)
    assert g["status"] == "ok" and g["best"] == cand and abs(g["gap"] - 0.1) < 0.02 and g["statistic"] == "G50"
    assert g["validation"]["verdict"] == "pass"
    cov = ST.battery_coverage("om_s0", jobs, root)
    assert cov[fam]["records"] == "complete" and cov["A_wave"]["records"] == "missing"
    os.remove(B.job_path("om_s0", fam, 3, "g_wary", 50, root))
    cov = ST.battery_coverage("om_s0", B.load("om_s0", root), root)
    assert cov[fam]["records"] == "partial" and "3|g_wary" in cov[fam]["missing_records"]
    seed = ST.battery_seed("om_s0", root)
    assert seed["families"][fam]["status"] == "partial records"
    # a failed validation: the 300-turn value contradicts the 50-turn one -> pending until the 150-turn re-run exists
    root2 = str(tmp_path / "battery2")
    for i in range(n):
        for p in B.FAMILIES[fam][3]:
            _battery_record(root2, "om_s0", fam, i, p, 300 if p == "policy" else 50, {cand: 0.1}, rng=rng)
        rec_p = B.job_path("om_s0", fam, i, "policy", 300, root2)
        with gzip.open(rec_p, "rt", encoding="utf-8") as f:
            r = json.load(f)
        for c in r["GQ"]:
            if c != "policy":
                for X in ("G", "Q"):
                    r["GQ"][c][X]["300"] = (np.asarray(r["GQ"][c][X]["300"]) - 2.0 - i * 0.3).tolist()
        with gzip.open(rec_p, "wt", encoding="utf-8") as f:
            json.dump(r, f)
    g = ST.family_gap(B.load("om_s0", root2), fam)
    assert g["validation"]["verdict"].startswith("fail") and g["status"].startswith("pending")
    assert ST.failed_families("om_s0", root2) == [fam]
    assert ST.validate_at(B.load("om_s0", root2), fam, 150)["verdict"] == "fail"     # 300 was shifted, 150 was not
    assert ST.validate_at(B.load("om_s0", root), fam, 150)["verdict"] == "pass"
    assert g["validation_150_diagnostic"]["verdict"] == "fail"
    fake = [{"start": ("lane", "m4", 30 + i), "info": None, "seat": 0, "turn": 30 + i, "seed": i} for i in range(3)]
    ext = ST.extension_jobs("om:0", "om_s0", [fam, "A_wave"], {"A_wave": fake}, root2)
    assert len(ext) == n * len(B.FAMILIES[fam][3]) + 3 * len(B.FAMILIES["A_wave"][3])
    assert {j[7] for j in ext} == {B.EXTENDED_H} and {j[2] for j in ext} == {fam, "A_wave"}
    paths = {B.job_path(j[1], j[2], j[3], j[6], j[7], j[10]) for j in ext}
    assert len(paths) == len(ext) and all(p.endswith("__h150.json.gz") for p in paths)
    assert not any(os.path.exists(p) for p in paths)                   # the 300-turn records are never overwritten
    for i in range(n):
        for p in B.FAMILIES[fam][3]:
            _battery_record(root2, "om_s0", fam, i, p, B.EXTENDED_H, {cand: 0.3}, rng=rng)
    g = ST.family_gap(B.load("om_s0", root2), fam)
    assert g["extended"] and g["status"] == "ok" and g["statistic"] == "G150" and abs(g["gap"] - 0.3) < 0.02


def _paired_record(path, cands, G, meta=None, K=4, H=50):
    rec = {"G": {c: {str(h): [G[c]] * K for h in (1, 50, 100, 150) if h <= H} for c in cands},
           "meta": meta or {}, "label": "", "keys": [], "ledger": {}}
    C.save_gz(path, rec)


def test_weak_side_and_cross_arm_cells_on_synthetic_records(tmp_path):
    from lanegame.experiments.rl.m46c.evaluate import STRONG_CANDS, WEAK_CANDS
    from lanegame.experiments.rl.m5 import strategic as ST
    root = str(tmp_path / "cells")
    cells = ST.weak_cells()
    for i, c in cells.items():
        cands = WEAK_CANDS if c[0] == "weak" else STRONG_CANDS
        G = {k: (0.0 if k == "policy" else (0.15 if k == "hold" and c[3] == 16 else -0.4)) for k in cands}
        _paired_record(os.path.join(root, "weak", "sp_s0", f"{i}.json.gz"), cands, G)
    w = ST.weak_gap(ST._load_dir("weak", "sp_s0", root))
    assert w["status"] == "ok" and abs(w["gap"] - 0.15) < 1e-9 and w["worst_lead"] == "+16"
    os.remove(os.path.join(root, "weak", "sp_s0", "0.json.gz"))
    assert ST.weak_gap(ST._load_dir("weak", "sp_s0", root))["status"] == "partial records"
    # a cross-arm family: om - sp = +0.2 in every cell, for seeds 0..4 only (seed 5 missing)
    n = ST.expected_items("freeze")
    from lanegame.experiments.rl.m46f import behaviour as BH
    metas = [m for _, _, m in BH.freeze_cells(C.arm_cfg("sp"))]
    for s in range(5):
        for opp in C.ARMS:
            for i in range(n):
                _paired_record(os.path.join(root, "freeze", f"s{s}_vs_{opp}", f"{i}.json.gz"), ("om", "sp"),
                               {"om": 0.2, "sp": 0.0}, metas[i])
    x = ST.cross_arm("freeze", root=root)
    assert x["status"] == "partial (5/6)" and x["missing_records"] == {"5": 2 * n}
    st = x["stats"]["om - sp | all | G50"]
    assert abs(st["mean"] - 0.2) < 1e-9 and st["n_seeds"] == 5


# ---------------------------------------------------------------------------------------------- runtime health
def _entry(u, **kw):
    e = {"update": u, "agent_transitions": 4096 * u, "policy_loss": 0.0, "value_loss": 0.1, "entropy": 1.0,
         "approx_kl": 0.01, "clipfrac": 0.1, "grad_norm": 0.4, "grad_norm_max": 0.5, "explained_variance": 0.5,
         "illegal_sampled": 0, "never_legal_offered": 0}
    e.update(kw)
    return e


def test_run_monitor_matches_the_post_hoc_scan_and_stops_only_on_defects(tmp_path):
    from lanegame.experiments.rl.m45c.full import stability_scan
    from lanegame.experiments.rl.m5 import health as HL
    hist = [_entry(u) for u in range(1, 40)]
    for u in range(12, 22):
        hist[u - 1]["approx_kl"] = 0.08                               # sustained KL (investigate)
    hist[30]["entropy"] = 0.2                                         # entropy floor (investigate)
    mon = HL.RunMonitor(str(tmp_path / "a"))
    for e in hist:
        mon(e)                                                        # investigate alarms never stop the run
    live = sorted(a["alarm"] for a in HL.alarms(str(tmp_path / "a")))
    assert live == sorted(stability_scan(hist)["alarms"]) and len(live) == 2
    kl = next(a for a in HL.alarms(str(tmp_path / "a")) if "KL" in a["alarm"])
    assert kl["class"] == "investigate" and kl["update"] == 17         # updates 8-17: (6 x 0.08 + 4 x 0.01) / 10 > 0.05
    early = [_entry(u, approx_kl=0.2 if u == 3 else 0.01) for u in range(1, 30)]
    mon = HL.RunMonitor(str(tmp_path / "b"))
    for e in early:
        mon(e)                                                        # one high KL is not "sustained"
    assert HL.alarms(str(tmp_path / "b")) == [] and stability_scan(early)["alarms"] == []
    mon = HL.RunMonitor(str(tmp_path / "c"))
    mon(_entry(1))
    with pytest.raises(HL.StopCondition, match="illegal action"):
        mon(_entry(2, illegal_sampled=3))
    mon = HL.RunMonitor(str(tmp_path / "d"))
    with pytest.raises(HL.StopCondition, match="non-finite"):
        mon(_entry(1, value_loss=float("nan")))
    assert HL.alarms(str(tmp_path / "d"))[0]["class"] == "defect"


def test_monitored_training_is_bit_identical_to_the_unmonitored_pin(tmp_path):
    """The OM2 trainer pin (test_om2) reproduced with the runtime monitor attached."""
    torch = pytest.importorskip("torch")
    from lanegame import GameConfig
    from lanegame.experiments.rl.m5.health import RunMonitor
    from lanegame.rl import PolicyValueNet, SharedConfig, train_shared, weights_digest
    from lanegame.rl.ppo import PPOConfig
    base = GameConfig.m45_calibrated()
    torch.set_num_threads(1)
    torch.manual_seed(3)
    ppo = PPOConfig.continuing(base, total_steps=2 * 4 * 32 * 2, n_envs=4, segment_length=32, game_length=40,
                               minibatch_size=64, seed=11, lane_cache=500)
    mon = RunMonitor(str(tmp_path))
    res = train_shared(base, ppo, SharedConfig(current_fraction=0.7, snapshot_every=128, pool_size=2),
                       net=PolicyValueNet(90, 61), verbose=False, on_update=mon)
    assert weights_digest(res.net.state_dict()) == "4e0c2d1acc42d30661e583f267eac8d5bf79843ce6c1dcf3f47bd603650c045a"
    assert mon.n == len(res.history) and mon.n > 0


def test_tail_reading_and_campaign_projection(tmp_path):
    from lanegame.experiments.rl.m5 import health as HL
    p = tmp_path / "history.jsonl"
    p.write_text("\n".join(json.dumps(_entry(u, sps=100.0 + u, elapsed=float(u))) for u in range(1, 501)) + "\n")
    t = HL.tail_entries(str(p), n=5, chunk=256)
    assert [e["update"] for e in t] == [496, 497, 498, 499, 500]
    runs = {"om_s0": {"state": "running", "eta_h": 2.0}, "om_s1": {"state": "running", "eta_h": 3.0},
            "sp_s0": {"state": "queued"}, "sp_s1": {"state": "queued"}}
    assert HL.project_wall(runs, 2, 3600 * 1000, {"sp": 1000.0}) == pytest.approx(4.0)   # 2 + 1, 3 + 1


def test_coordinator_stops_after_more_than_two_failed_pairs():
    from lanegame.experiments.rl.m5 import health as HL
    names = ["om_s0", "om_s1", "om_s2", "om_s3", "om_s4", "om_s5"]
    jobs = [(n, st, 0.0) for n, st in zip(names, ("failed", "complete", "stopped", "complete", "complete", "complete"))]
    out = HL.run_campaign(HL._probe_job, jobs, names, 2)
    assert out["stopped"] is None and HL.failed_pairs(out["results"]) == [0, 2]
    jobs = [(n, "failed", 0.0) for n in names[:3]] + [(n, "complete", 5.0) for n in names[3:]]
    out = HL.run_campaign(HL._probe_job, jobs, names, 3)
    assert out["stopped"] is not None and out["stopped"]["failed_pairs"] == [0, 1, 2]


def test_evaluation_gate_refuses_failed_unresolved_or_changed_training(tmp_path):
    from lanegame.experiments.rl.m5 import health as HL
    out = str(tmp_path)
    with pytest.raises(SystemExit, match="no training check"):
        HL.require_training_ok(out=out)
    ck = tmp_path / "final.pt"
    ck.write_bytes(b"weights")
    runs = {f"{a}_s{s}": {"status": "complete"} for s in C.SEEDS for a in C.ARMS}
    ident = {n: {"final": {"path": str(ck), "file_sha256": C.file_sha256(str(ck))}} for n in runs}
    d = {"runs": runs, "complete": list(runs), "checks": {"om_s0: no pre-registered stop condition": False},
         "identities": ident}
    C.save_json("train_main", d, out)
    with pytest.raises(SystemExit, match="without a recorded resolution"):
        HL.require_training_ok(out=out)
    with pytest.raises(SystemExit):
        HL.resolve("main", "om_s0: no pre-registered stop condition", "fine", "x", out=out)
    HL.resolve("main", "om_s0: no pre-registered stop condition", "training property (not a defect)",
               "entropy fell slowly; no defect found", out=out)
    assert HL.require_training_ok(out=out)
    ck.write_bytes(b"changed")
    with pytest.raises(SystemExit, match="differ from the training check"):
        HL.require_training_ok(out=out)
    runs["sp_s3"] = {"status": "stopped"}
    C.save_json("train_main", {**d, "runs": runs}, out)
    with pytest.raises(SystemExit, match="incomplete or failed"):
        HL.require_training_ok(out=out)


def test_restart_prefix_checks_are_enforced(tmp_path, monkeypatch):
    from lanegame.experiments.rl.m5 import train as T
    a, b = tmp_path / "partial", tmp_path / "run"
    for d, ent in ((a, 1.0), (b, 1.0)):
        d.mkdir()
        (d / "history.jsonl").write_text("\n".join(json.dumps(_entry(u, entropy=ent)) for u in range(1, 4)))
    monkeypatch.setattr(C, "load", lambda name, *x: {"restarts": [{"run": "om_s0", "phase": "main",
                                                                    "moved_to": str(a)}]} if name == "restarts" else None)
    monkeypatch.setattr(C, "run_dir", lambda arm, s, phase="main": str(b))
    r = T.restart_checks("main")
    assert list(r.values())[0]["identical"]
    (b / "history.jsonl").write_text("\n".join(json.dumps(_entry(u, entropy=2.0)) for u in range(1, 4)))
    assert not list(T.restart_checks("main").values())[0]["identical"]


# ---------------------------------------------------------------------------------------------- the future stages
def test_attacker_core_and_optional_job_definitions():
    from lanegame.experiments.rl.m5 import attackers as AT
    core = AT.eval_jobs(AT.CORE_KINDS)
    assert len(core) == 12 * 3 and not any("om_table" in j["key"] or "cross" in j["key"] for j in core)
    full = AT.eval_jobs(AT.CORE_KINDS + AT.OPTIONAL_KINDS)
    assert len(full) == 18 * 3 + 12 and len({j["path"] for j in full}) == len(full)
    for s in C.SEEDS:
        by = {j["key"].split("|", 2)[2]: j for j in full if f"|s{s}|" in j["key"]}
        ct, tt = by[f"om|s{s}|cross_table"], by[f"om_table|s{s}|target"]
        assert ct["opponent"][2].startswith("table:") and (ct["root"], ct["cell"]) == (tt["root"], tt["cell"])
        cl, ol = by[f"om_table|s{s}|cross_live"], by[f"om|s{s}|target"]
        assert cl["opponent"][2] == "live" and (cl["root"], cl["cell"]) == (ol["root"], ol["cell"])
    own = [j for j in full if "cross" not in j["key"]]
    assert len({(j["root"], j["cell"]) for j in own}) == len(own)


def test_cli_refuses_the_optional_tables_without_their_flag():
    from lanegame.experiments.rl.m5.__main__ import main
    with pytest.raises(SystemExit, match="include-om-table"):
        main(["attack_tables", "--confirm-m5-attackers"])


def test_horizon_breakdown_reproduces_the_budget_plan():
    from lanegame.experiments.rl.m5 import evaluate as EV
    rows = EV.battery_breakdown()
    total = sum(r["standard_turns"] + r["validation_turns_beyond_h"] for r in rows)
    assert total == pytest.approx(EV.battery_turns("om"))
    cells = EV.cells_breakdown()
    plan = EV.cells_turns()
    assert sum(r["turns"] for r in cells) == pytest.approx(sum(plan.values()))
