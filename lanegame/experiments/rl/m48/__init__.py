"""M4.8: lane-exit evaluation (docs/m4/m4_8/m4_8_lane_exit_report.md; protocol docs/m4/m4_8/m4_8_protocol.md).

EVALUATION ONLY. The lane-exit economic readout E_h = D0 + expected change of the earned-gold lead over h continuation
turns (`lanegame.diagnostics.lane_exit`) is validated on selected natural cases, mirrored cases and controlled variants
(Stage 1), then applied to natural checkpoints of the matched M4.6c baseline and C1 policies around the pre-level-14
boundary (Stage 2). No training, no rule, reward, objective or default changes. Run as
`python -m lanegame.experiments.rl.m48 <stage> [...]`; see `__main__`.
"""
