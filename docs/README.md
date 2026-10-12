# Documentation index

Two documents live at the top level because every milestone refers to them:

| document | what it is |
|---|---|
| [design_philosophy.md](design_philosophy.md) | the standard every mechanic is judged against: simplify how League works without changing why its laning strategies work |
| [rl_research_roadmap.md](rl_research_roadmap.md) | the research roadmap; one section per milestone (A–Y so far) with the decision and the link to its full report |

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

## m4/ — the integrated champion economy (M4.1 design, M4.2 implementation and the M4.2b–M4.2d studies)

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
