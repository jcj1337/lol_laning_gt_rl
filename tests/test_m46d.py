"""M4.6d instrumentation: plans, cells, the paired matrix job, the episode classifier and the decomposition design."""

import os

import numpy as np
import pytest

from lanegame.experiments.rl.m46d import cells as CL
from lanegame.experiments.rl.m46d import common as C
from lanegame.experiments.rl.m46d import plans as P

torch = pytest.importorskip("torch")

HAVE_C1 = os.path.exists(C.C46.policy_path(C.general("C1", 0)))
HAVE_SPEC = os.path.exists(C.specialist("gap", 0)) and os.path.exists(C.C46.policy_path(C.general("baseline", 0)))
needs_c1 = pytest.mark.skipif(not HAVE_C1, reason="M4.6c ext checkpoints absent (git-ignored)")
needs_spec = pytest.mark.skipif(not HAVE_SPEC, reason="M4.6a specialist / M4.6c checkpoints absent")


def test_strength_conditions_match_their_description():
    from lanegame.experiments.rl.m46d.freezes import Strength
    cfg = C.arm_cfg("C1")
    st = Strength(cfg)
    for L in CL.A_LEVELS:
        La, Lb, Sa, Sb = CL.strength(cfg, "edge", L)
        duel, inv, lev = st.parts(La, Sa, Lb, Sb)
        assert 1.19 <= duel <= 1.29 and lev == pytest.approx(1.05 ** 2)
        La, Lb, Sa, Sb = CL.strength(cfg, "x1.76", L)
        assert st.parts(La, Sa, Lb, Sb)[0] == pytest.approx(1.76 ** 2, rel=0.01)
        La, Lb, Sa, Sb = CL.strength(cfg, "brk", L)
        assert st.parts(La, Sa, Lb, Sb)[0] < 1 / 1.19


def test_cells_are_legal_states_with_the_intended_layout():
    from lanegame.champions import champion_legal_actions
    from lanegame.experiments.rl.m44.records import holds_freeze
    cfg = C.arm_cfg("C1")
    for cell in CL.grid_a() + CL.grid_b():
        lane, m4, k = CL.cell_start(cfg, cell)
        assert champion_legal_actions(lane, m4, 0, cfg) and champion_legal_actions(lane, m4, 1, cfg)
        assert holds_freeze(lane, 0) == (cell["wave"] == "freeze")
        hb = P.hp(cfg, m4, 1)
        assert hb == pytest.approx(cell["hp_b"], abs=0.01)


def test_episode_classifier_on_a_synthetic_record():
    from lanegame.experiments.rl.m46d.freezes import Strength, episodes
    T = 8
    keys = ("held0", "crash", "fh", "fx", "L", "S", "hp", "absent", "pos", "rear_near", "nrear", "mass", "C", "V",
            "wave", "beh", "earned", "farm", "avail", "bounty", "plate", "xpg", "died", "recalled", "away", "taken",
            "intr")
    d = {k: np.zeros((1, T, 2)) for k in keys}
    d["p_wave"] = np.zeros((1, T, 2, 4))
    d["p_beh"] = np.zeros((1, T, 2, 5))
    d["k"] = np.arange(T)[None]
    d["L"][:] = 5
    d["hp"][:] = 1.0
    d["fh"][:] = 8
    d["held0"][0, 1:5, 0] = 1                       # A holds turns 1-4: one maintained episode
    d["wave"][0, 0, 1] = 1                          # B shoved on the turn before: pushed-in origin
    d["crash"][0, 4, 1] = 5                         # B's wave crashed on the episode's last turn
    eps = episodes(d, 0, Strength(C.arm_cfg("C1")))
    assert len(eps) == 1
    e = eps[0]
    assert (e["t0"], e["n"], e["end"], e["origin"], e["f"]) == (1, 4, "crashed_in", "pushed-in", 0)
    assert e["preserving"]                          # wave 0 (Last Hit) on every present turn


def test_plan_restricts_to_its_slice_and_releases():
    cfg = C.arm_cfg("C1")

    class Fake:
        def __init__(self):
            self.u = np.full(50, 0.5)

        def probs(self, cfg, lane, m4, s, i):
            from lanegame.champions import champion_legal_actions
            legal = champion_legal_actions(lane, m4, s, cfg)
            return legal, np.full(len(legal), 1.0 / len(legal))

        def decide(self, cfg, lane, m4, s, i):
            legal, p = self.probs(cfg, lane, m4, s, i)
            return legal[int(self.u[i] * len(legal))], "fake"

    from lanegame.config import champ_behavior, champ_wave
    lane, m4, k = CL.build(cfg, "freeze", "equal", 5)
    own = Fake()
    plan = P.build(P.BREAKER["commit"], own)
    c, _ = plan.decide(cfg, lane, m4, 1, 0)
    assert champ_wave(c) == P.SH and champ_behavior(c) == 0
    assert plan.decide(cfg, lane, m4, 1, 10)[0] == own.decide(cfg, lane, m4, 1, 10)[0]       # after n turns
    low = m4._replace(b=m4.b._replace(hp=1))
    assert plan.decide(cfg, lane, low, 1, 0)[0] == own.decide(cfg, lane, low, 1, 0)[0]       # released at low HP
    fz = P.build(P.FREEZER["passive"], own)
    c, _ = fz.decide(cfg, lane, m4, 0, 0)
    assert champ_wave(c) in (P.LH, P.TN) and champ_behavior(c) == 0


@needs_c1
def test_matrix_job_policy_pair_equals_the_engine_and_is_deterministic(tmp_path):
    from lanegame.experiments.rl.m46a import engine as EN
    from lanegame.experiments.rl.m45c import horizon as HZ
    from lanegame.experiments.rl.m46d import matrix as M
    cfg = C.setup("C1")
    st = CL.build(cfg, "freeze", "edge", 5, 0.6)
    job = M.mk("test", "C1", C.general("C1", 0), "t", 0, st, P.A_FREEZER, P.A_BREAKER,
               [("policy", "policy"), ("guard_sel", "concede")], 2, 12, 7)
    job["path"] = str(tmp_path / "a.npz")
    M.matrix_job(job)
    d1 = M.load(job["path"])
    net = C.net_of(C.general("C1", 0))
    sd = 7 * 1000
    out = EN.rollout(cfg, st, HZ.CRNNet(net, st[2], sd, 0, 12), HZ.CRNNet(net, st[2], sd, 1, 12), sd, 12,
                     M.cps_for(12), (), None)
    assert d1["G"][0, 0, -1] == pytest.approx(out[12]["G"], abs=1e-12)
    job["path"] = str(tmp_path / "b.npz")
    M.matrix_job(job)
    assert np.array_equal(M.load(job["path"])["G"], d1["G"])


@needs_spec
def test_switch_design_identity(tmp_path):
    """S1 (the specialist at turn 0 only, then the general) and U0 (the specialist's choice at turn 0) are the same
    candidate: equal returns rollout by rollout."""
    from lanegame.experiments.rl.m46d import specialists as SP
    j = dict(SP.decomp_jobs()[0])
    j.update({"K": 2, "H": 12, "path": str(tmp_path / "d.npz")})
    SP.decomp_job(j)
    z = np.load(j["path"])
    names = [str(x) for x in z["names"]]
    assert np.array_equal(z["G"][names.index("S1")], z["G"][names.index("U0")])
