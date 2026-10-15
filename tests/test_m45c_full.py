"""M4.5c full campaign: the training stage (gated, never overwrites, failure records, alarms, reproducibility
comparison), the limited health evaluation's design (reserved seeds, checkpoints on the snapshot schedule and the
409,600 grid, paired seeds) and its pre-registered rules, and the revised strategic battery (exceptions, validation
subset, job design, validation rule, gate, and one real paired job on the pilot policy when its checkpoint exists).
Skipped without PyTorch."""

import gzip
import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lanegame.experiments.rl.m45c import battery as B                                     # noqa: E402
from lanegame.experiments.rl.m45c import common as C                                      # noqa: E402
from lanegame.experiments.rl.m45c import full as F                                        # noqa: E402
from lanegame.experiments.rl.m45c import health as H                                      # noqa: E402
from lanegame.experiments.rl.m45c import health_report as R                               # noqa: E402


# ---------------------------------------------------------------------------------------------- training stage
def test_campaign_specification():
    assert F.FULL_SEEDS == (0, 1, 2, 3, 4, 5) and F.FULL_BUDGET == 8_192_000 and F.FULL_MIDPOINT == 4_096_000
    assert F.FULL_BUDGET % C.TRANSITIONS_PER_UPDATE == 0 and F.FULL_BUDGET // C.TRANSITIONS_PER_UPDATE == 2000
    assert F.PPO_KW == {"lane_cache": 5000}
    sched = C.snapshot_schedule(F.FULL_BUDGET)
    assert len(sched) == 320 and F.FULL_MIDPOINT in sched and sched[-1] == F.FULL_BUDGET
    js = F.jobs()
    assert [j[1] for j in js] == [0, 1, 2, 3, 4, 5, 0] and js[-1][4] == F.REPEAT_TAG
    assert all(j[0] == "full" and j[2] == F.FULL_BUDGET and j[3] is None for j in js)
    p = C.ppo_config(0, F.FULL_BUDGET, **F.PPO_KW)
    assert p.lane_cache == 5000 and p.total_steps == F.FULL_BUDGET and p.lr == 3e-4 and p.anneal_lr


def test_train_full_is_gated_and_refuses_existing_runs(tmp_path, monkeypatch):
    from lanegame.experiments.rl.m45c.__main__ import main
    with pytest.raises(SystemExit, match="confirm-m45c-full"):
        main(["train_full"])
    monkeypatch.setattr(C, "CKPT", str(tmp_path))
    d = C.run_dir("full", 3)
    os.makedirs(d)
    open(os.path.join(d, "history.jsonl"), "w").close()
    with pytest.raises(SystemExit, match="never overwrites"):
        F.train((3,), False, "test")


def test_a_failed_run_writes_an_error_record(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "CKPT", str(tmp_path))
    os.makedirs(C.run_dir("smoke", 7, "_x"))

    def boom(args):
        raise RuntimeError("simulated failure")
    import lanegame.experiments.rl.m45c.train as T
    monkeypatch.setattr(T, "train_one", boom)
    r = F._job(("smoke", 7, 4096, None, "_x"))
    assert r["status"] == "failed" and "simulated failure" in r["error"]
    with open(os.path.join(C.run_dir("smoke", 7, "_x"), "summary.json")) as f:
        assert json.load(f)["status"] == "failed"


def _hist(n=30, **over):
    h = [{"approx_kl": 0.005, "entropy": 1.8, "grad_norm": 0.6, "grad_norm_max": 1.0, "policy_loss": -0.01,
          "value_loss": 0.005, "clipfrac": 0.06, "explained_variance": 0.4, "illegal_sampled": 0,
          "never_legal_offered": 0} for _ in range(n)]
    for k, (i, v) in over.items():
        h[i][k] = v
    return h


def test_stability_scan_flags_exactly_the_pre_registered_conditions():
    assert F.stability_scan(_hist())["alarms"] == []
    assert F.stability_scan(_hist(approx_kl=(5, 0.2)))["alarms"] == []          # one spike is not "sustained"
    h = _hist()
    for e in h[10:20]:
        e["approx_kl"] = 0.06
    assert any("KL" in a for a in F.stability_scan(h)["alarms"])
    assert any("non-finite" in a for a in F.stability_scan(_hist(value_loss=(3, float("nan"))))["alarms"])
    assert any("gradient" in a for a in F.stability_scan(_hist(grad_norm_max=(3, 150.0)))["alarms"])
    assert any("entropy" in a for a in F.stability_scan(_hist(entropy=(29, 0.2)))["alarms"])
    assert any("illegal" in a for a in F.stability_scan(_hist(illegal_sampled=(2, 1)))["alarms"])


# ---------------------------------------------------------------------------------------------- health evaluation
def test_health_design_uses_reserved_seeds_and_existing_checkpoints():
    sched = set(C.snapshot_schedule(F.FULL_BUDGET))
    assert set(H.CHECKPOINTS) <= sched and all(n % 409_600 == 0 for n in H.CHECKPOINTS)
    assert H.LATE_QUARTER in H.CHECKPOINTS and F.FULL_MIDPOINT in H.CHECKPOINTS and H.CHECKPOINTS[-1] == F.FULL_BUDGET
    seeds = [H.CURVE_SEED_F, H.H2H_SEED_F, H.HIST_SEED_F, H.CYCLE_SEED_F, H.CROSS_SEED_F, H.BEHAVIOR_SEED_F,
             *H.LEDGER_SEEDS_F]
    assert all(C.FULL_RESERVED[0] <= s <= C.FULL_RESERVED[1] for s in seeds)
    assert H.spec(2, "initial") == "initial:2" and H.spec(2, F.FULL_BUDGET) == "final:full:2"
    assert H.spec(2, 409_600) == "snap:full:2:409600"
    for part in H.PARTS:
        js = H.JOBS[part]()
        keys = [f"{j[0]}|{j[1]}" for j in js]
        assert len(keys) == len(set(keys)), part                 # results are keyed by (x, y)
        assert len({j[4] for j in js}) == 1                      # one seed per part: every comparison is paired
    assert len(H.cross_jobs()) == 15 and len(H.h2h_jobs()) == 12
    assert len(H.curve_jobs()) == 6 * 13 * 10


def test_pre_registered_rules():
    seeds = (0, 1, 2)
    cv = {"final_minus_q3": {"per_seed": {"0": {"scripted": 0.1}, "1": {"scripted": 0.5}, "2": {"scripted": 0.05}}}}
    hh = {"final_vs_q3": {"per_seed": {"0": {"estimate": 0.1, "half": 0.3}, "1": {"estimate": 0.2, "half": 0.3},
                                       "2": {"estimate": 0.25, "half": 0.2}}},
          "final_vs_mid": {"per_seed": {s: {"estimate": 0.8, "half": 0.3} for s in "012"}, "pooled": [0.8, 0.2]}}
    cy = {"stop_criterion_met": False}
    cr = {"seeds_disagree": False}
    r = R.rules(cv, hh, cy, cr, None, seeds)
    assert r["plateau"]["0"]["plateaued"]                         # both conditions hold
    assert not r["plateau"]["1"]["plateaued"]                     # the curve moved 0.5 over the last quarter
    assert not r["plateau"]["2"]["plateaued"]                     # the head-to-head interval excludes 0
    assert not r["extension"]["criterion_met"]                    # only 2 runs not plateaued (needs >= 3)
    hh["final_vs_q3"]["per_seed"]["0"]["estimate"] = 0.6
    r = R.rules(cv, hh, cy, cr, None, seeds)
    assert r["extension"]["criterion_met"] and len(r["extension"]["seeds_individually_above_0.5_and_significant"]) == 3


def test_cross_play_matrix_is_antisymmetric():
    ev = {"cross": {f"final:full:{a}|final:full:{b}": {"estimate": a - b + 0.5, "ci95": [a - b, a - b + 1.0]}
                    for a in range(3) for b in range(a + 1, 3)}}
    cr = R.cross(ev, (0, 1, 2))
    M = np.asarray(cr["matrix"])
    assert np.allclose(M[np.triu_indices(3, 1)], -M.T[np.triu_indices(3, 1)])
    assert cr["pairs"] == 3 and cr["max_abs"] == pytest.approx(1.5) and cr["seeds_disagree"] is False


# ---------------------------------------------------------------------------------------------- battery
def test_battery_horizons_follow_the_pre_registered_exceptions():
    assert B.family_horizon("B_freeze") == B.family_horizon("F_freeze_c") == 60      # 20-turn plans + 40
    for fam in ("A_wave", "A_wave_early", "B_frozen", "C_trade", "D_ward", "E_recall", "G_trade_c", "H_zone_c",
                "I_ward_c"):
        assert B.family_horizon(fam) == 50                        # plans of up to 10 turns stay at 50
    assert {B.primary_statistic(f) for f in B.FAMILIES if B.FAMILIES[f][0] == "constructed"} == {"G"}
    assert {B.primary_statistic(f) for f in B.FAMILIES if B.FAMILIES[f][0] != "constructed"} == {"Q"}
    assert B.checkpoints(60)[-1] == 60 and 50 in B.checkpoints(60) and B.checkpoints(300)[-1] == 300


def test_battery_validation_subset_is_fixed_and_sized():
    for n, m in ((48, 8), (24, 8), (100, 15), (5, 5), (0, 0)):
        idx = B.validation_indices("A_wave", n)
        assert len(idx) == m and idx == B.validation_indices("A_wave", n) and len(set(idx)) == m
    assert B.validation_indices("A_wave", 48) != B.validation_indices("C_trade", 48)


def test_battery_job_design():
    fake = {f: [{"start": ("lane", "m4", 5), "seed": 1, "turn": 2, "seat": 0}] * 48 for f in ("A_wave", "B_freeze")}
    js = B.jobs("final:full:0", "s0_final", fake, ("A_wave", "B_freeze", "G_trade_c"))
    val = set(B.validation_indices("A_wave", 48))
    for j in js:
        spec, tag, fam, idx, st, proto, h, label, K, root = j
        if fam == "G_trade_c":
            assert K == B.K_CONSTRUCTED and h == (300 if proto == "policy" else 50)
        else:
            assert K == B.K_NATURAL
            long = proto == "policy" and idx in set(B.validation_indices(fam, 48))
            assert h == (300 if long else B.family_horizon(fam))
    assert {j[5] for j in js if j[2] == "B_freeze"} == set(B.FREEZE_OPP)
    assert sum(1 for j in js if j[2] == "A_wave" and j[6] == 300) == len(val)
    ext = B.jobs("x", "t", fake, ("A_wave",), extended=("A_wave",))
    assert {j[6] for j in ext} == {B.EXTENDED_H}
    paths = {B.job_path("t", j[2], j[3], j[5], j[6]) for j in js}
    assert len(paths) == len(js)                                   # one resumable file per job
    v = B.volume()
    assert v["total_turns"] > 0 and set(v["per_family"]) == set(B.FAMILIES)


def _vjob(fam, idx, d50, d300, K=12, cand="1t:trade"):
    rng = np.random.default_rng(idx)
    base = rng.normal(size=K)
    noise = rng.normal(scale=0.05, size=K)
    g = {"policy": {"50": base.tolist(), "300": base.tolist()},
         cand: {"50": (base + d50 + noise).tolist(), "300": (base + d300 + noise).tolist()}}
    return {"family": fam, "protocol": "policy", "horizon": 300, "index": idx,
            "candidates": {"policy": None, cand: ["fixed", 0, 1, None]},
            "first_codes": {"policy": [0] * K, cand: [0] * K},
            "GQ": {c: {"G": g[c], "Q": g[c]} for c in g}}


def test_battery_validation_rule():
    ok = [_vjob("C_trade", i, d, d) for i, d in enumerate(np.linspace(-1, 1, 8))]
    assert B.validate(ok, "C_trade")["verdict"] == "pass"
    biased = [_vjob("C_trade", i, d, d - 0.6) for i, d in enumerate(np.linspace(-1, 1, 8))]
    assert B.validate(biased, "C_trade")["verdict"].startswith("fail")
    flipped = [_vjob("C_trade", i, d, -d) for i, d in enumerate(np.linspace(-1, 1, 8))]
    r = B.validate(flipped, "C_trade")
    assert r["verdict"].startswith("fail") and r["contradictions"] and not r["slope_ok"]


def test_battery_is_gated():
    from lanegame.experiments.rl.m45c.__main__ import main
    with pytest.raises(SystemExit, match="confirm-m45c-battery"):
        main(["battery"])


@pytest.mark.skipif(not os.path.exists(C.final_path("pilot", C.PILOT_SEED)), reason="pilot checkpoint not present")
def test_battery_state_job_is_paired_and_resumable(tmp_path):
    cfg = C.env_cfg()
    st = B.constructed_state(cfg, "G_trade_c", B.GRIDS["G_trade_c"][0])
    args = ("final:pilot:100", "t", "G_trade_c", 0, st, "policy", 5, "constructed", 2, str(tmp_path))
    key, d = B.state_job(args)
    assert d["K"] == 2 and set(d["GQ"]) == {"policy", *B.candidates("G_trade_c")}
    assert all(len(d["GQ"][c]["G"]["5"]) == 2 for c in d["GQ"])
    # the policy baseline is the evaluated policy itself: re-running the same job from disk returns identical data
    with gzip.open(B.job_path("t", "G_trade_c", 0, "policy", 5, str(tmp_path)), "rt") as f:
        assert json.load(f)["GQ"] == d["GQ"]
    _, d2 = B.state_job(args)
    assert d2["GQ"] == d["GQ"]


def test_comprehensive_parts_and_attackers_are_gated():
    from lanegame.experiments.rl.m45c.__main__ import main
    with pytest.raises(SystemExit, match="confirm-m45c-comprehensive"):
        main(["health_eval", "--health-parts", "headline"])
    with pytest.raises(SystemExit, match="confirm-m45c-attackers"):
        main(["attackers"])
    from lanegame.experiments.rl.m45c import attackers as A
    assert A.ATTACK_BUDGET_F == F.FULL_BUDGET // 2
    assert C.FULL_RESERVED[0] <= A.ATTACK_TRAIN_SEED_F and A.ATTACK_EVAL_SEED_F <= C.FULL_RESERVED[1]
    sched = set(C.snapshot_schedule(F.FULL_BUDGET))
    assert len(H.TOURNAMENT) == 9 and set(H.TOURNAMENT) <= sched and len(H.tournament_jobs()) == 6 * 36
    rest = {int(j[0].split(":")[-1]) for j in H.curve_rest_jobs() if j[0].startswith("snap")}
    assert rest | set(H.CHECKPOINTS) == set(H.GRID_POINTS) and not rest & set(H.CHECKPOINTS)


def test_battery_report_pools_policies_and_reports_validation(tmp_path):
    for t, shift in (("a", 0.0), ("b", 0.2)):
        d = tmp_path / t / "partial"
        d.mkdir(parents=True)
        for i, x in enumerate(np.linspace(-1, 1, 8)):
            j = _vjob("C_trade", i, x + shift, x + shift)
            with gzip.open(d / f"C_trade__{i}__policy.json.gz", "wt") as f:
                json.dump(j, f)
    text = B.report(["a", "b"], str(tmp_path))
    assert "## C_trade" in text and "a: pass" in text and "b: pass" in text
    r = B.pooled_over_policies({t: B.analyze(B.load(t, str(tmp_path))) for t in "ab"}, "C_trade", "policy",
                               "1t:trade", "Q", 50)
    assert r["n"] == 2 and r["mean"] == pytest.approx(0.1, abs=0.05)


@pytest.mark.skipif(not os.path.exists(C.final_path("pilot", C.PILOT_SEED)), reason="pilot checkpoint not present")
def test_trace_export_replays_exactly_in_the_viewer(tmp_path):
    from lanegame.experiments.rl.m44.records import play_record
    from lanegame.experiments.rl.m44.trace import verify
    from lanegame.experiments.rl.m45c import traces as TR
    cfg = C.env_cfg()
    x = C.agent_from_spec("final:pilot:100", cfg)
    for start, scen, gap in ((None, "s0", "equal"), (TR._gap_start(), "neutral", "large")):
        rec = play_record(cfg, (x, x), 94_299, 40, start, scen, "m45c", viewer_seed=True)
        path = TR.export_record(rec, f"t_{scen}", ("p", "p"), 94_299, scen, gap, "test", str(tmp_path))
        v = verify(path)
        assert v["actions_equal"] and v["payoff_equal"] and v["config"] == "m45_calibrated"
        with open(path) as f:
            d = json.load(f)
        assert d["config"] == "m45_calibrated" and all("Vision" not in a and "Vision" not in b for a, b in d["labels"])


@pytest.mark.skipif(not os.path.exists(C.final_path("pilot", C.PILOT_SEED)), reason="pilot checkpoint not present")
def test_battery_mechanics_are_exact_one_turn_expectations():
    cfg = C.env_cfg()
    st = B.constructed_state(cfg, "G_trade_c", B.GRIDS["G_trade_c"][1])
    (_, fam), d = B.mechanics_job(("final:pilot:100", "G_trade_c", [(0, st)]))
    row = d["states"][0]
    assert fam == "G_trade_c" and set(row["codes"]) == {"policy (modal)", *B.candidates("G_trade_c")}
    assert row["mean"]["policy (modal)"]["earned"] is not None
    agg = d["aggregate"]["policy (modal)"]
    assert agg["immediate_lead_diff"] == 0 and agg["same_code_as_policy"] == 1.0


def test_extension_campaign_scales_every_fraction_and_compares_budgets():
    sched = set(C.snapshot_schedule(F.EXT_BUDGET))
    assert F.CAMPAIGNS["ext"] == (16_384_000, F.EXT_OUT, False) and "ext" in C.PHASES
    assert H.EXT.checkpoints == tuple(2 * n for n in H.MAIN.checkpoints) and set(H.EXT.checkpoints) <= sched
    assert H.EXT.mid == F.EXT_MIDPOINT == 8_192_000 and H.EXT.q3 == 12_288_000 and set(H.EXT.grid) <= sched
    assert H.spec(3, "final", H.EXT) == "final:ext:3" and H.spec(3, 8_192_000, H.EXT) == "snap:ext:3:8192000"
    vs = H.vs_main_jobs()
    assert len(vs) == 36 and {j[0].split(":")[1] for j in vs} == {"ext"} and {j[1].split(":")[1] for j in vs} == {"full"}
    an = H.anneal_jobs()
    assert all(j[0].startswith("final:full:") and j[1].endswith(":8192000") for j in an) and len(an) == 36
    assert {"vs_main", "anneal", "historical", "headline_main"} <= set(H.COMPREHENSIVE)
    assert all(j[0].startswith("final:full:") for j in H.headline_main_jobs())
    from lanegame.experiments.rl.m45c import attackers as A
    assert A.attacker_dir(2, phase="ext").endswith(os.path.join("attack", "ext_s2"))
