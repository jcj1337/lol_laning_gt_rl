# M4 documentation

The living description of the current game, the research preset `m45_calibrated()`, is
[laning_game_mathematical_reference.md](laning_game_mathematical_reference.md). It stays at this level because every
milestone refers to it. Its version before the 2026-10-20 rewrite, which described `m4_lasthit()` with the research
preset's differences in separate sections, is archived beside it as
[laning_game_mathematical_reference_m4_lasthit.md](laning_game_mathematical_reference_m4_lasthit.md); reports written
before that date cite its section numbers. Everything else is grouped by milestone; documents are historical records of their milestone
and are not edited after the fact (corrections live in later reports and in the reference's §12).

| folder | milestone | documents |
|---|---|---|
| [m4_1](m4_1/) | M4.1 integrated design | [integrated design](m4_1/m4_integrated_design.md), [design audit](m4_1/m4_design_audit.md) |
| [m4_2](m4_2/) | M4.2 implementation and calibration (M4.2a–d) | [implementation plan](m4_2/m4_implementation_plan.md), [implementation report](m4_2/m4_implementation_report.md), [M4.2b calibration](m4_2/m4_calibration_report.md), [M4.2c continuous investment](m4_2/m4_continuous_investment_report.md), [M4.2d investment readiness](m4_2/m4_investment_readiness_report.md), [wave readiness](m4_2/m4_wave_readiness_report.md), [positioning](m4_2/m4_positioning_report.md), [zoning](m4_2/m4_zoning_report.md), [parallel integration](m4_2/m4_emergent_parallel_report.md) |
| [m4_3](m4_3/) | M4.3c–d economics and wave design | [economic objective design](m4_3/m4_3c_economic_objective_design.md), [XP-range screen](m4_3/m4_3c_xp_range_screen.md), [wave-control screen](m4_3/m4_3c_wave_control_screen.md), [four-action redesign](m4_3/m4_3c_four_action_wave_redesign.md), [strategic economics calibration](m4_3/m4_3c_strategic_economics_calibration.md), [M4.3d tower pressure](m4_3/m4_3d_tower_pressure_calibration.md) |
| [m4_4](m4_4/) | M4.4 PPO on the four-action economies | [M4.4a readiness](m4_4/m4_4a_ppo_readiness.md), [M4.4b training](m4_4/m4_4b_ppo_training_report.md), [M4.4c strategic investigation](m4_4/m4_4c_strategic_investigation.md) |
| [m4_5](m4_5/) | M4.5 unified mechanics and the full PPO campaign | [M4.5a spec](m4_5/m4_5a_unified_mechanics_spec.md), [M4.5a report](m4_5/m4_5a_unified_mechanics_report.md), [M4.5b calibration](m4_5/m4_5b_calibration_report.md), [M4.5c pilot](m4_5/m4_5c_pilot_report.md), [runtime audit](m4_5/m4_5c_runtime_audit.md), [horizon study](m4_5/m4_5c_horizon_study.md), [training](m4_5/m4_5c_training_report.md), [research report](m4_5/m4_5c_research_report.md) |
| [m4_6](m4_6/) | M4.6 design review, diagnosis, the investment audit, the C1 training pilot, the freeze / concession diagnosis, the value diagnostic and the separate-critic pilot | [design review](m4_6/m4_6_design_review.md), [M4.6a pre-registration](m4_6/m4_6a_preregistration.md), [M4.6a report](m4_6/m4_6a_report.md), [M4.6b pre-registration](m4_6/m4_6b_preregistration.md), [M4.6b investment audit and candidate screen](m4_6/m4_6b_investment_report.md), [M4.6c pre-registration](m4_6/m4_6c_preregistration.md), [M4.6c C1-versus-baseline training pilot](m4_6/m4_6c_report.md), [M4.6d pre-registration](m4_6/m4_6d_preregistration.md), [M4.6d freezes, concession and the specialist gap](m4_6/m4_6d_report.md), [M4.6e pre-registration](m4_6/m4_6e_preregistration.md), [M4.6e observation, training exposure and value learnability](m4_6/m4_6e_report.md), [M4.6f pre-registration](m4_6/m4_6f_preregistration.md), [M4.6f separate-critic continuation pilot](m4_6/m4_6f_report.md) |
| [m4_7](m4_7/) | M4.7 early-lane horizon study (after the closeout, before the final M5 campaign) | [M4.7 pre-registration](m4_7/m4_7_preregistration.md), [M4.7 horizon study](m4_7/m4_7_horizon_report.md) |
| [m4_8](m4_8/) | M4.8 lane-exit evaluation (evaluation only; validates the lane-exit readout and applies it to the matched baseline/C1 policies) | [M4.8 protocol](m4_8/m4_8_protocol.md), [M4.8 lane-exit evaluation](m4_8/m4_8_lane_exit_report.md) |

The M4-wide closeout record sits at this level, beside the reference:
[m4_closeout_report.md](m4_closeout_report.md) (2026-10-16). It contains:
- the strategic acceptance assessment of every laning goal;
- configuration recommendations (the user chose the recommended baseline and shared critic on 2026-10-17);
- accepted limitations and corrections to the M4.6e–f interpretations;
- M5's verified commitments and proposed scope.

Related documents elsewhere: M4.2e natural waves in [../natural_waves](../natural_waves/), the M4.3a–b PPO pilot and
earlier RL reports in [../rl](../rl/), Phase 1 in [../phase1](../phase1/).

Moved on 2026-10-15 from a flat `docs/m4/` (every link and path reference in code, tests, documentation and the README
was updated; generated files under `results/` keep the old flat paths they were written with).
