"""The run stages of the M4.3d study (see __main__). Each stage stores its results under results/m43d/ and can be
re-run: cells already stored are kept (grid.run and strategic.run merge)."""

from __future__ import annotations

from . import common as c
from . import grid

TOWER_FARM_CONFIGS = ("ot60", "econ", "ot85", "ot90")
PRESSURE_B = ("lh", "safe", "punish", "fight")
PRESSURE_MASKS = ("a_no_gank", "a_no_tower")
# Part 4: the risk variants selected by the Part 3 diagnosis (report §5); each changes one existing parameter group
SENSITIVITY_CONFIGS = ("tw_aggro", "tw2", "tw2_aggro", "gk_pos16", "gk_dmg08")
SENSITIVITY_STARTS = ("S10", "S4", "N10")
# Part 5: the restricted games of the M4.3c scenarios on these configurations
STRATEGIC_CONFIGS = ("base4", "econ", "ot85", "ot90", "econ_aggro")
# Part 6: the independent validation seed set (128 games) on the comparison set
VALIDATION_CONFIGS = ("econ", "ot90", "econ_aggro")
VALIDATION_SCENARIOS = ("F10g", "F10gl", "N10", "N10g", "N3", "P10g", "F4")


def tower_farm(workers: int = 11, variants=None) -> dict:
    """Part 2: the crash grid on the own-tower rates (econ otherwise)."""
    return grid.run("tower_farm", variants or TOWER_FARM_CONFIGS, workers=workers)


def pressure(workers: int = 11, variants=None) -> dict:
    """Part 3: the attacker immune to ganks / to tower shots, paired with the unmasked games of the tower_farm grid
    (same root, same seeds, same cells)."""
    return grid.run("pressure", variants or ("econ",), b_set=PRESSURE_B, masks=PRESSURE_MASKS, workers=workers)


def sensitivity(workers: int = 11, variants=None) -> dict:
    """Part 4: the crash grid (full A x B sets) on the risk variants; the same seeds as Parts 2 and 3."""
    return grid.run("sensitivity", variants or SENSITIVITY_CONFIGS, starts=SENSITIVITY_STARTS, workers=workers)


def strategic(workers: int = 11, variants=None) -> dict:
    """Part 5: the M4.3c restricted games (plus Vision, dive and Recall strategies) on each configuration."""
    from . import strategic as sg
    return sg.run(variants or STRATEGIC_CONFIGS, workers=workers)


def validation(workers: int = 11, variants=None) -> dict:
    """Part 6: the key restricted games and crash cells on an independent seed set (128 games)."""
    from . import strategic as sg
    out = sg.run(variants or VALIDATION_CONFIGS, scenarios=VALIDATION_SCENARIOS, workers=workers,
                 root=c.STRATEGIC_VALIDATION_SEED, episodes=128, tag="_val")
    grid.run("validation", variants or VALIDATION_CONFIGS, starts=("S10", "N10"), root=c.VALIDATION_SEED,
             episodes=128, workers=workers)
    return out


def traces(workers: int = 11, variants=None) -> str:
    from . import traces as tr
    return tr.run()


def report(workers: int = 11, variants=None) -> None:
    from . import report_strategic, report_tower
    report_tower.run()
    report_strategic.run()


__all__ = ["tower_farm", "pressure", "sensitivity", "strategic", "validation", "traces", "report"]
