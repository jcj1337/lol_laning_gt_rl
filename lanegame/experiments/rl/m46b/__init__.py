"""M4.6b: investment audit and candidate-calibration screen, with the M4.6a measurement fixes
(docs/m4/m4_6/m4_6b_investment_report.md; protocol docs/m4/m4_6/m4_6b_preregistration.md). The calibrated baseline
`GameConfig.m45_calibrated()` is unchanged; candidate investment curves are opt-in configurations built from existing
fields (`invest_curve`, `invest_gain`, `invest_scale`, `invest_alpha`, `invest_p`) and never become a default.

Modules
    common     arms (the baseline and the candidate curves), seeds, paths, worker set-up, I/O
    verify     reproductions that verify the measurement fixes against stored M4.5c records
    audit      the gold-to-strength mapping: formulas, natural trajectories, the x1.01 explanation, what leads buy
    curves     candidate definitions and their mapping tables (pure formulas)
    screen     gameplay screens of every arm (exact one-turn ledger, paired rollouts, natural starts)
    report     generated tables
CLI: `python -m lanegame.experiments.rl.m46b <stage> [--workers N]`.
"""
