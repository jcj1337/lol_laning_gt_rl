# Documentation index

Three documents are the entry points every milestone refers to:

| document | what it is |
|---|---|
| [m4/laning_game_mathematical_reference.md](m4/laning_game_mathematical_reference.md) | **the living mathematical reference of the current game** (`GameConfig.m4_lasthit()`): state, actions, wave mechanics, farming and XP, combat, progression, towers, reward, the RL objective, worked examples; updated with every rule change |
| [design_philosophy.md](design_philosophy.md) | the standard every mechanic is judged against: simplify how League works without changing why its laning strategies work |
| [rl_research_roadmap.md](rl_research_roadmap.md) | **the canonical roadmap**: a current-position summary at the top (milestones M1–M6, the M4.1–M4.4 subphases, what is implemented / experimental / adopted / planned / unresolved, the M4.4a–d plan), then one section per milestone (A–AF) with the decision and the link to its full report |

The reports are grouped by phase. Within a folder the files are listed in the order they were written.

## phase1/ — the exact game (Phase 1)

| document | milestone |
|---|---|
| [phase1_implementation_report.md](phase1/phase1_implementation_report.md) | the original Phase 1 model (`GameConfig.v1()`) |
| [phase1_revision_report.md](phase1/phase1_revision_report.md) | the farming revision (`GameConfig.v2()`) |
| [phase1_siege_vision_study.md](phase1/phase1_siege_vision_study.md) | the siege and vision study (`GameConfig()`) |

## rl/ — the RL environment, the champion game and self-play (M1–M3, OpenAI Five studies)

| document | milestone |
|---|---|
| [m2_m3_implementation_report.md](rl/m2_m3_implementation_report.md) | M1–M3c: environment, champion game, PPO, self-play, evaluation |
| [overnight_implementation_log.md](rl/overnight_implementation_log.md) | the running log of the M2–M3 implementation |
| [openai_five_adaptation.md](rl/openai_five_adaptation.md) | methodology notes for the OpenAI Five-inspired study |
| [openai_five_progress.md](rl/openai_five_progress.md) | its progress log |
| [openai_five_selfplay_report.md](rl/openai_five_selfplay_report.md) | shared-policy self-play with a current/historical mixture |
| [openai_five_diagnostics_progress.md](rl/openai_five_diagnostics_progress.md) | the diagnostics milestone's progress log |
| [openai_five_diagnostics_report.md](rl/openai_five_diagnostics_report.md) | diagnostics of the learned policies, exploitability, long runs |
| [m4_3a_rl_infrastructure_report.md](rl/m4_3a_rl_infrastructure_report.md) | M4.3a: RL infrastructure audit, PettingZoo conformance, PPO reference checks, the smoke test on `m4_lasthit()` |
| [m4_3b_ppo_pilot_report.md](rl/m4_3b_ppo_pilot_report.md) | M4.3b: the PPO self-play pilot on `m4_lasthit()`: learning curves, headline evaluation, exploitability, conditional behaviour, counterfactuals, Recall / death economics, the mixed-start follow-up |
| [m4_3b_validation_report.md](rl/m4_3b_validation_report.md) | M4.3b follow-up: wave-only interventions with adaptive champion behaviour, opponent-dependent value of slow pushing and freezing, the evaluation horizon, freeze-state classification, synthetic power gaps; research-artifact manifest and cleanup proposal |
| [ppo_training_explained.md](rl/ppo_training_explained.md) | how the PPO self-play pipeline works, for a reader learning PPO: the 88-feature input, the 61 masked outputs, shared-policy self-play, actor and critic, one worked training step with real numbers, training versus evaluation, limitations (M4.4a) |

## m4/ — the integrated champion economy (M4.1 design, M4.2 implementation and studies, M4.3c–M4.3d calibration, M4.4a PPO readiness, M4.4b controlled training)

| document | milestone |
|---|---|
| [m4_integrated_design.md](m4/m4_integrated_design.md) | M4.1: the mathematical specification |
| [m4_implementation_plan.md](m4/m4_implementation_plan.md) | M4.1: work packages and validation stages |
| [m4_design_audit.md](m4/m4_design_audit.md) | M4.1: the design review |
| [m4_implementation_report.md](m4/m4_implementation_report.md) | M4.2: implementation and pre-training validation (`GameConfig.m4()`) |
| [m4_calibration_report.md](m4/m4_calibration_report.md) | M4.2b: ganks, items, the Shove → Recall cycle |
| [m4_continuous_investment_report.md](m4/m4_continuous_investment_report.md) | M4.2c: continuous gold investment (`m4_continuous()`) |
| [m4_wave_readiness_report.md](m4/m4_wave_readiness_report.md) | M4.2d-A: wave bounce and Freeze breaking |
| [m4_investment_readiness_report.md](m4/m4_investment_readiness_report.md) | M4.2d-D: the saturating investment curve and compressed observation |
| [m4_positioning_report.md](m4/m4_positioning_report.md) | M4.2d-B: independent champion positioning |
| [m4_zoning_report.md](m4/m4_zoning_report.md) | M4.2d-C: combat-driven zoning (scripted waves) |
| [m4_emergent_parallel_report.md](m4/m4_emergent_parallel_report.md) | M4.2d integration (`m4_emergent()`) |
| [laning_game_mathematical_reference.md](m4/laning_game_mathematical_reference.md) | the living reference of the current game (M4.3c Part A; see above) |
| [m4_3c_economic_objective_design.md](m4/m4_3c_economic_objective_design.md) | M4.3c design: gold + XP objective, termination at the final plate, game-length audit, freeze incentives, death timers (PROPOSED, awaiting review) |
| [m4_3c_xp_range_screen.md](m4/m4_3c_xp_range_screen.md) | M4.3c quick screen of `xp_range` 3 / 2 / 1 (freeze denial, neutral-lane control): keep 3, investigate another mechanic |
| [m4_3c_wave_control_screen.md](m4/m4_3c_wave_control_screen.md) | M4.3c quick screen of Hold capped at 1 HP and an adaptive (target-based) Thin: adopt Hold1 on its own, not adaptive Thin yet |
| [m4_3c_four_action_wave_redesign.md](m4/m4_3c_four_action_wave_redesign.md) | M4.3c four-action wave management (EXPERIMENTAL `m4_four_actions()`): Last Hit / Thin / Slow Push / Clear (now Shove) with window-strength targets, 61-code action space; mechanically ready for PPO, Clear dominance unchanged |
| [m4_3c_strategic_economics_calibration.md](m4/m4_3c_strategic_economics_calibration.md) | M4.3c stabilization: Shove terminology, the four-action visual debugger, mixed-action sequences; economic diagnosis (the own-tower "crash tax" and break-even trades make Shove every side's best reply); opt-in XP reward, death schedule and terminal rule; EXPERIMENTAL candidate `m4_four_actions_econ()` |
| [m4_3d_tower_pressure_calibration.md](m4/m4_3d_tower_pressure_calibration.md) | M4.3d tower farming, pressure and gank risk: the own-tower rate 0.8 makes an uncontested crash farm-neutral for the defender (0.9 makes it a gift); gank risk is already the main price of pressure; the tower's "busy" discount under-prices a strong champion's dive; EXPERIMENTAL candidate `m4_four_actions_econ_aggro()` |
| [m4_4a_ppo_readiness.md](m4/m4_4a_ppo_readiness.md) | M4.4a four-action PPO readiness: the three-environment training specification, checkpoint guard, 61-code decoding, state-based strategic metrics (wave management, freezes, zoning, vision / dives, plate phases), paired counterfactuals, smoke tests, measured M4.4b cost; GO for M4.4b |
| [m4_4b_ppo_training_report.md](m4/m4_4b_ppo_training_report.md) | M4.4b controlled PPO self-play on the three four-action economies (3 seeds × 2M transitions, matched 2× extension): PPO health and convergence, strength against references and own history, state-dependent wave management, freezes and their denial, vision and dives (econ vs aggro), zoning, recalls, the counterfactual scorecard, representative viewer games; CONDITIONAL GO for M4.4c (in review) |

## natural_waves/ — natural wave mechanics and the lane candidate (M4.2e)

| document | milestone |
|---|---|
| [m4_2e_natural_waves_report.md](natural_waves/m4_2e_natural_waves_report.md) | M4.2e-A: the natural-wave prototype (`m4_natural_prototype()`) |
| [m4_2e_natural_waves_final_review.md](natural_waves/m4_2e_natural_waves_final_review.md) | M4.2e-A.1: the final natural-wave preset (`m4_natural()`) |
| [m4_2e_zoning_report.md](natural_waves/m4_2e_zoning_report.md) | M4.2e-B: design philosophy and combat-driven zoning (`m4_lane()`) |
| [m4_2e_wave_actions_report.md](natural_waves/m4_2e_wave_actions_report.md) | M4.2e-C: wave actions as shares of one capacity, the half-wave law (`m4_lasthit()`) |
| [m4_2e_debugger.md](natural_waves/m4_2e_debugger.md) | M4.2e-D: the local visual simulation debugger (`streamlit run lanegame/viewer/app.py`) |

Raw results and the commands that produced them are under `results/`, one folder per study; the drivers are in
`lanegame/experiments/`.
