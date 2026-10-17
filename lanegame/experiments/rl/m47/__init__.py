"""M4.7: the early-lane horizon study (docs/m4/m4_7/m4_7_horizon_report.md).

Stage 1 measures what part of current play happens before about level 14, how the matched M4.6c baseline and C1
policies differ there, and how much of the benefit of early actions and advantages arrives after that phase. Stage 2
(conditional on Stage 1) continues matched baseline policies under the current discount (gamma 0.99) and under one
shorter discount, an opt-in keyword override of `GameConfig.gamma`, and compares their play. Run as
`python -m lanegame.experiments.rl.m47 <stage> [...]`; see `__main__`.
"""
