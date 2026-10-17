"""M4.6e: the focused observation, training-exposure and value-learning diagnostic (docs/m4/m4_6/m4_6e_report.md).
Diagnostic only: no PPO, no actor training, no rule, observation, reward or default change.

Modules: common (definitions, seeds), features (state and observation-derived strength features), observation (Q1:
the measurement path from state to critic), exposure (Q2: archive starts and reconstructed training occupancy), labels
(Q3: policy-specific no-bootstrap return labels and the horizon check), fit (Q4: the supervised value study), ranking
(look-ahead rankings of the fitted predictors at the M4.6d decision states), report, and the CLI
(`python -m lanegame.experiments.rl.m46e <stage> ...`).
"""
