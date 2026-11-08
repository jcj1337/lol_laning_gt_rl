"""M4.4b analysis code (docs/archive/m4/m4_4/m4_4b_ppo_training_report.md): the recall / investment metric, viewer-convention
game records and their replay (s0 and initialized starts), the integrity check of a network, the seed statistics and
the counterfactual alignment of the report, the guard that keeps the M4.4b report out of the M4.4a folder, and the
CLI stages. No training and no checkpoint files: networks are the untrained initial networks of a seed."""

import numpy as np
import pytest

from experiments.m44 import common as C
from experiments.m44 import metrics as M
from experiments.m44 import records as R
from experiments.m44 import supplement as S


def test_the_recall_metric_counts_every_absence_and_its_return():
    from experiments.m44.references import reference_suite
    cfg = C.env_cfg("econ")
    s = reference_suite()
    rec = R.play_record(cfg, (s["grow_crash_recall4"], s["shove_hunt4"]), np.random.SeedSequence(4), 240)
    out = M.recalls([rec])
    for side in (0, 1):
        one = M.recalls([rec], focal=rec.names[side])
        d = rec.data
        n_recall = int((d["recalled"][:, side] > 0).sum())
        n_death = int((d["died"][:, side] > 0).sum())
        assert one["recall_low_hp"]["n"] + one["recall_other"]["n"] == n_recall
        assert one["death"]["n"] == n_death
    assert out["recall_low_hp"]["n"] + out["recall_other"]["n"] + out["death"]["n"] > 0
    for k in ("recall_low_hp", "recall_other", "death"):
        g = out[k]
        if g["n"] - g["censored"] > 0:
            assert g["away_turns"] >= 0 and g["bought"] >= 0
    assert set(out["by_investment_lead"]) == {"ahead", "level", "behind"}
    assert "recalls" in M.all_metrics([rec])["natural:s0"]


@pytest.mark.parametrize("scenario,gap", [("s0", "equal"), ("neutral", "large")])
def test_viewer_seed_records_replay_exactly_in_the_debugger(tmp_path, scenario, gap):
    from experiments.m43c.harness import start as scenario_start
    from experiments.m44.trace import verify
    cfg = C.env_cfg("aggro")
    x = C.agent_from_spec("initial:aggro:1", cfg)
    start = None if scenario == "s0" else scenario_start(scenario, cfg, gap)
    rec = R.play_record(cfg, (x, x), 80_777, 40, start, "s0" if start is None else f"{scenario}/{gap}", "aggro",
                        viewer_seed=True)
    acts = [[int(rec.data["a"][t, 0]), int(rec.data["a"][t, 1])] for t in range(rec.turns)]
    p = S.export_record(rec, acts, "t", "aggro", ("initial:aggro:1",) * 2, 80_777, scenario, gap, "test", str(tmp_path))
    v = verify(p)
    assert v["actions_equal"] and v["payoff_equal"] and v["turns"] == 40
    ex = S.excerpt(rec, 10, 12)
    assert [r["turn"] for r in ex] == list(range(7, 19)) and set(ex[0]) == {"turn", "A", "B"}


def test_a_viewer_seed_record_is_the_game_record_game_plays():
    from experiments.m43b.trajectories import record_game
    cfg = C.env_cfg("econ")
    x = C.agent_from_spec("initial:econ:2", cfg)
    rec = R.play_record(cfg, (x, x), 80_778, 30, viewer_seed=True)
    g = record_game(cfg, x, x, 80_778, 30)
    assert [[int(rec.data["a"][t, 0]), int(rec.data["a"][t, 1])] for t in range(30)] == g["actions"]
    assert abs(float(rec.data["reward"][:, 0].sum()) - g["payoff_A"]) < 1e-9


def test_the_integrity_check_of_a_network_finds_no_violation():
    _, out = S.integrity_job(("four", "initial:four:0", 1, 30, 80_779))
    assert out["observations"] == 60
    for k in ("nonfinite_legal_logits", "nonfinite_values", "values_above_omega", "observations_with_illegal_mass",
              "reward_asymmetries"):
        assert out[k] == 0
    assert out["max_illegal_mass"] == 0.0


def test_nearest_snapshots_follow_update_boundaries(tmp_path, monkeypatch):
    d = tmp_path / "runs" / "checkpoints" / "m44" / "main" / "econ_s0" / "snapshots"
    d.mkdir(parents=True)
    for n in (28_672, 53_248, 77_824):
        (d / f"snap_{n:09d}.pt").write_bytes(b"")
    monkeypatch.chdir(tmp_path)
    assert S.nearest_snapshots("main", "econ", 0, (25_600, 51_200, 76_800)) == [28_672, 53_248, 77_824]
    assert S.nearest_snapshots("main", "econ", 0, (25_600, 30_000)) == [28_672]

