"""M4.5b: integrated calibration and strategic evaluation of the M4.5a unified positional mechanics
(docs/m4/m4_5b_calibration_report.md).

Question: do the combined strategic and economic incentives of `GameConfig.m45_unified()` (revision 2) make sense,
and which provisional numbers, if any, should change? Nothing here trains a network or changes the reward: every
measurement is a paired, common-random-number simulation of scripted strategies (`strategies`) from constructed or
naturally reached states (`states`), through the real M4 resolver (`engine`).

    python -m lanegame.experiments.m45b <stage> [...]        stages: see __main__.STAGES

Modules
    common       seeds (development 87,000-87,499, holdout 88,000-88,499), configurations, statistics, I/O, job runner
    engine       the paired game loop (env-identical chance stream), the per-side ledger, state collection, branching
    states       parametric start states (freeze grid, trade bench, Vision states, zoning states) and the holdout split
    strategies   contingent scripted strategies (freeze keepers, weak-side responses, Ward rules, Recall rules)
    freeze       Priority A: strong-side freezing, zoning and Recall economics
    trades       Priority B: minion damage, retaliation and wave dynamics
    vision       Priority C: Vision positioning and economics
    zoning       Priority D: Safe, Yield and threat-based zoning
    matrix       Priority E: integrated strategy payoff matrices and restricted-game equilibria
    timing       Priority F: wave-probe accuracy, crossing between ticks, trade-contact timing
    sensitivity  the joint sensitivity analysis and the candidate selection
    validate     the holdout evaluation of the selected candidate against the revision-2 baseline
    demos        viewer recordings (results/m45b/trajectories)
    report       results/m45b/report.md
"""
