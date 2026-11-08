# Research archive

The original reports, plans, pre-registrations and logs of every milestone, from the first exact game (October 2026)
to the network capacity study (2026-10-29). Start with the [project history](history.md), which condenses them: how
the project evolved, the decisions on reward and rules, what failed, the corrections made later, the contradictions
between sources and the lessons. The measured results are collected in [findings](../findings.md).

## Reading archived documents

- **Unchanged.** Every file here is byte-identical to the original; [MOVES.md](MOVES.md) gives each one's SHA-256.
  Pre-registrations must stay that way, because their hashes are part of the studies' records (M5's amendment chain
  checks its documents on every campaign stage).
- **Links use the old layout.** The reports were written when the code was in `lanegame/experiments/`, results in
  `results/<study>/` and the living reference in `docs/m4/laning_game_mathematical_reference.md`. Links between
  archived reports still work, because the directory structure is kept. Links to code, results or the reference point
  to old paths. [MOVES.md](MOVES.md) maps them, and everything resolves at the pre-reorganization commit
  [`f613f0a0`](https://github.com/jcj1337/lol_game_theory_laning/tree/f613f0a00a2d46fe717de7e8c79077fa9e0310c3).
- **Commands use the old layout.** `python -m lanegame.experiments.rl.m5 report` is now
  `python -m experiments.m5 report`, and `python -m lanegame experiments-siege all` is
  `python -m experiments.siege all`. Studies whose packages were removed run only at `f613f0a0`.
- **Corrections are elsewhere.** Reports are never edited after the fact. Later reports, the roadmap's summary,
  [history §6](history.md#6-corrections-made-after-the-fact) and the [model](../model.md)'s §12 record corrections.
- **Tables.** Each study's generated tables (`report.md` and companions) and pre-registration records are kept in
  `artifacts/<study>/`; its raw JSON outputs were removed and are at `f613f0a0` under `results/<study>/`.

## The roadmap

[rl_research_roadmap.md](rl_research_roadmap.md) is the project's running record: the original plan (sections A to
K), then every milestone's outcome in order (L to BG), and in its summary the user's decisions with their dates. It
is the place to find who decided what and when. [docs_index_2026-10-29.md](docs_index_2026-10-29.md) is the
documentation index as it stood before the reorganization.

## Documents by era

Code column: **kept** means part of the study's package is still in `experiments/` (as instruments the final
campaign or the tests use); **removed** means its code is only at `f613f0a0`.

### Phase 1: the exact game (2026-10-10 to 10-11)

| document | contents | tables | code |
|---|---|---|---|
| [phase1_implementation_report.md](phase1/phase1_implementation_report.md) | the original exact game (v1), its solution and exploitability, the audit of the Q-learning prototype | `artifacts/v1/` | removed |
| [phase1_revision_report.md](phase1/phase1_revision_report.md) | the farming revision (v2): farm in the payoff, Contest/Safe, ablations | `artifacts/revision/` | removed |
| [phase1_siege_vision_study.md](phase1/phase1_siege_vision_study.md) | siege objectives and the default model `GameConfig()` | `artifacts/siege/` | kept (`experiments/siege/`) |

The Q-learning prototype the project started from is in [`legacy/`](../../legacy/).

### M1 to M3: the first learning game (2026-10-11)

| document | contents | tables | code |
|---|---|---|---|
| [m2_m3_implementation_report.md](rl/m2_m3_implementation_report.md) | M2 champions (Recall, trades, ganks, death) and M3 PPO, snapshot self-play, exact calibration | `artifacts/rl/`, `artifacts/champions/` | removed |
| [overnight_implementation_log.md](rl/overnight_implementation_log.md) | the running log of the M2b to M3c session | | |
| [openai_five_adaptation.md](rl/openai_five_adaptation.md) | what OpenAI Five documents and what was adapted | | |
| [openai_five_progress.md](rl/openai_five_progress.md) | code audit and work log of the shared self-play study | | |
| [openai_five_selfplay_report.md](rl/openai_five_selfplay_report.md) | snapshot pool against shared 100/0 and 80/20 self-play | `artifacts/rl/openai_five/` | removed |
| [openai_five_diagnostics_progress.md](rl/openai_five_diagnostics_progress.md) | work log of the diagnostics study | | |
| [openai_five_diagnostics_report.md](rl/openai_five_diagnostics_report.md) | action audit, counterfactuals, 1M-step attackers, long runs | `artifacts/rl/diagnostics/` | removed (the library part is `lanegame/diagnostics/`) |
| [ppo_training_explained.md](rl/ppo_training_explained.md) | an explainer of the shared PPO pipeline (written at M4.4a) | | |

### M4.1 to M4.3: the integrated game (2026-10-11 to 10-13)

| document | contents | tables | code |
|---|---|---|---|
| [m4_integrated_design.md](m4/m4_1/m4_integrated_design.md) | the M4 specification: continuing time, economy, progression, plates | `artifacts/m4_design/` | removed |
| [m4_design_audit.md](m4/m4_1/m4_design_audit.md) | a critical audit of that design | | |
| [m4_implementation_plan.md](m4/m4_2/m4_implementation_plan.md) | work packages WP0 to WP10 and validation stages | | |
| [m4_implementation_report.md](m4/m4_2/m4_implementation_report.md) | the implementation of `m4()` and its validation | `artifacts/m4/` | removed |
| [m4_calibration_report.md](m4/m4_2/m4_calibration_report.md) | gank severity, item strength, the Shove to Recall cycle | `artifacts/m4_calibration/` | removed |
| [m4_continuous_investment_report.md](m4/m4_2/m4_continuous_investment_report.md) | `m4_continuous()`: all gold invested on Recall | `artifacts/m4_continuous_investment/` | removed |
| [m4_wave_readiness_report.md](m4/m4_2/m4_wave_readiness_report.md) | the missing wave bounce; the pre-crash farm spot | | removed |
| [m4_positioning_report.md](m4/m4_2/m4_positioning_report.md) | stand-off positioning | | removed |
| [m4_zoning_report.md](m4/m4_2/m4_zoning_report.md) | the zoning inventory and `zoning_rule` | `artifacts/m4_zoning/` | removed |
| [m4_investment_readiness_report.md](m4/m4_2/m4_investment_readiness_report.md) | the saturating investment curve, compressed observation | | removed |
| [m4_emergent_parallel_report.md](m4/m4_2/m4_emergent_parallel_report.md) | four parallel workstreams and `m4_emergent()` (workstream A's own report is only on branch `prototype/m4-emergent-wave`) | | removed |
| [m4_2e_natural_waves_report.md](natural_waves/m4_2e_natural_waves_report.md) | the natural-waves prototype | | kept (`experiments/m4/natural_waves.py`) |
| [m4_2e_natural_waves_final_review.md](natural_waves/m4_2e_natural_waves_final_review.md) | the final `m4_natural()` | | kept |
| [m4_2e_zoning_report.md](natural_waves/m4_2e_zoning_report.md) | combat-driven zoning, `m4_lane()`, the design philosophy | | kept (`experiments/m4/zoning_natural.py`) |
| [m4_2e_wave_actions_report.md](natural_waves/m4_2e_wave_actions_report.md) | capacity-share wave actions, `m4_lasthit()` | | kept (`experiments/m4/wave_actions.py`) |
| [m4_2e_debugger.md](natural_waves/m4_2e_debugger.md) | the Streamlit lane debugger (retired; the recording format lives on in `lanegame/viewer/`) | | removed |
| [m4_3a_rl_infrastructure_report.md](rl/m4_3a_rl_infrastructure_report.md) | PettingZoo adapter, PPO reference checks, smoke test | | library |
| [m4_3b_ppo_pilot_report.md](rl/m4_3b_ppo_pilot_report.md) | the first PPO pilot on `m4_lasthit()`: Clear dominance | `artifacts/rl/m43b/` | kept (`experiments/m43b/`, part) |
| [m4_3b_validation_report.md](rl/m4_3b_validation_report.md) | wave-only interventions confirm it is the game's incentive | `artifacts/rl/m43b/` | kept (part) |
| [m4_3c_economic_objective_design.md](m4/m4_3/m4_3c_economic_objective_design.md) | the gold plus XP and termination proposal (not adopted); the first mathematical reference | | |
| [m4_3c_xp_range_screen.md](m4/m4_3/m4_3c_xp_range_screen.md) | XP range screen | `artifacts/m43c/` | removed |
| [m4_3c_wave_control_screen.md](m4/m4_3/m4_3c_wave_control_screen.md) | Hold cap and adaptive Thin | `artifacts/m43c/` | kept (`experiments/m43c/wave_control.py`) |
| [m4_3c_four_action_wave_redesign.md](m4/m4_3/m4_3c_four_action_wave_redesign.md) | `m4_four_actions()` | `artifacts/m43c/` | kept (`experiments/m43c/four_actions.py`) |
| [m4_3c_strategic_economics_calibration.md](m4/m4_3/m4_3c_strategic_economics_calibration.md) | the crash-tax diagnosis and `m4_four_actions_econ()` | `artifacts/m43c/` | kept (part) |
| [m4_3d_tower_pressure_calibration.md](m4/m4_3/m4_3d_tower_pressure_calibration.md) | tower farming, pressure, dives; `m4_four_actions_econ_aggro()` | `artifacts/m43d/` | kept (`experiments/m43d/strategies.py`) |
| [m4/README.md](m4/README.md) | the M4 report index as it stood | | |
| [laning_game_mathematical_reference_m4_lasthit.md](m4/laning_game_mathematical_reference_m4_lasthit.md) | the mathematical reference as it described `m4_lasthit()`, before the 2026-10-20 rewrite; earlier reports cite its sections | | |

### M4.4 to M4.5: learned strategy and the research preset (2026-10-13 to 10-15)

| document | contents | tables | code |
|---|---|---|---|
| [m4_4a_ppo_readiness.md](m4/m4_4/m4_4a_ppo_readiness.md) | PPO readiness and the strategic instruments | `artifacts/m44a/` | kept (`experiments/m44/`, part) |
| [m4_4b_ppo_training_report.md](m4/m4_4/m4_4b_ppo_training_report.md) | controlled training across three economies | `artifacts/m44b/` | kept (part) |
| [m4_4c_strategic_investigation.md](m4/m4_4/m4_4c_strategic_investigation.md) | multi-turn paired counterfactuals on 5,982 states | `artifacts/m44c/` | kept (`experiments/m44c/`, part) |
| [m4_5a_unified_mechanics_spec.md](m4/m4_5/m4_5a_unified_mechanics_spec.md) | the specification of `m45_unified()` | | |
| [m4_5a_unified_mechanics_report.md](m4/m4_5/m4_5a_unified_mechanics_report.md) | the implementation, revisions 1 and 2 | `artifacts/m45a/` | kept (`experiments/m45a/`, part) |
| [m4_5b_calibration_report.md](m4/m4_5/m4_5b_calibration_report.md) | the integrated calibration, the farm-spot defect, `m45_calibrated()` | `artifacts/m45b/` | kept (`experiments/m45b/`) |
| [m4_5c_pilot_report.md](m4/m4_5/m4_5c_pilot_report.md) | the one-seed pilot and the campaign plan | `artifacts/m45c/` | kept (`experiments/m45c/`, part) |
| [m4_5c_runtime_audit.md](m4/m4_5/m4_5c_runtime_audit.md) | the measured cost plan | | |
| [m4_5c_horizon_study.md](m4/m4_5/m4_5c_horizon_study.md) | the 50-turn counterfactual method and its validation | `artifacts/m45c/horizon/` | kept |
| [m4_5c_training_report.md](m4/m4_5/m4_5c_training_report.md) | six seeds at 8.192M | `artifacts/m45c/full/` | kept |
| [m4_5c_research_report.md](m4/m4_5/m4_5c_research_report.md) | extension, battery, attackers, conclusions | `artifacts/m45c/` | kept |

### M4.6 to M4.8: diagnosis and the closeout (2026-10-15 to 10-17)

| document | contents | tables | code |
|---|---|---|---|
| [m4_6_design_review.md](m4/m4_6/m4_6_design_review.md) | the proposed single cause of cheap freezes and weak zoning | | |
| [m4_6a_preregistration.md](m4/m4_6/m4_6a_preregistration.md), [m4_6a_report.md](m4/m4_6/m4_6a_report.md) | zoning and freeze diagnosis, gap specialists | `artifacts/m46a/` | kept (`experiments/m46a/`, part) |
| [m4_6b_preregistration.md](m4/m4_6/m4_6b_preregistration.md), [m4_6b_investment_report.md](m4/m4_6/m4_6b_investment_report.md) | the investment audit and candidate curves | `artifacts/m46b/` | kept (part) |
| [m4_6c_preregistration.md](m4/m4_6/m4_6c_preregistration.md), [m4_6c_report.md](m4/m4_6/m4_6c_report.md) | the C1 against baseline pilot | `artifacts/m46c/` | kept (part) |
| [m4_6d_preregistration.md](m4/m4_6/m4_6d_preregistration.md), [m4_6d_report.md](m4/m4_6/m4_6d_report.md) | favourable freezes, concession, specialists, critic blindness | `artifacts/m46d/` | kept (part) |
| [m4_6e_preregistration.md](m4/m4_6/m4_6e_preregistration.md), [m4_6e_report.md](m4/m4_6/m4_6e_report.md) | observation, exposure and value learnability | `artifacts/m46e/` | kept (part) |
| [m4_6f_preregistration.md](m4/m4_6/m4_6f_preregistration.md), [m4_6f_report.md](m4/m4_6/m4_6f_report.md) | the separate-critic pilot | `artifacts/m46f/` | kept (part) |
| [m4_closeout_report.md](m4/m4_closeout_report.md) | the M4 acceptance table, five corrected overstatements, M5 proposal | | |
| [m4_7_preregistration.md](m4/m4_7/m4_7_preregistration.md), [m4_7_horizon_report.md](m4/m4_7/m4_7_horizon_report.md) | the early-lane horizon study | `artifacts/m47/` | removed |
| [m4_8_protocol.md](m4/m4_8/m4_8_protocol.md), [m4_8_lane_exit_report.md](m4/m4_8/m4_8_lane_exit_report.md) | the lane-exit readout E_h | `artifacts/m48/` | removed (the readout is `lanegame/diagnostics/lane_exit.py`) |

### Opponent modelling (2026-10-17)

| document | contents | tables | code |
|---|---|---|---|
| [om1_protocol.md](opponent_modelling/om1_protocol.md), [om1_feasibility_report.md](opponent_modelling/om1_feasibility_report.md) | identifiability and value of information | `artifacts/om1/` | kept (`experiments/om1/common.py` only) |
| [om2_design.md](opponent_modelling/om2_design.md), [om2_preregistration.md](opponent_modelling/om2_preregistration.md), [om2_pilot_report.md](opponent_modelling/om2_pilot_report.md) | the belief, the previous-action observation, the matched pilot | `artifacts/om2/` | kept (`experiments/om2/`; the belief is `lanegame/opponent/`) |

### M5: the final campaign (2026-10-17 to 10-19)

| document | contents | tables | code |
|---|---|---|---|
| [m5_design.md](m5/m5_design.md) | arms, lineage, budget, decisions | `artifacts/m5/` (complete) | kept (`experiments/m5/`) |
| [m5_preregistration.md](m5/m5_preregistration.md) | the protocol, version 1 (frozen; the chain is in `artifacts/m5/preregistration.json`) | | |
| [m5_amendment_1.md](m5/m5_amendment_1.md) | the correction pass before training | | |
| [m5a_readiness_protocol.md](m5/m5a_readiness_protocol.md), [m5a_readiness_report.md](m5/m5a_readiness_report.md) | readiness checks R0 to R10 and the budget | | |
| [m5b_initial_report.md](m5/m5b_initial_report.md) | training and the primary endpoints (outcome 2) | | |
| [m5c_strategic_report.md](m5/m5c_strategic_report.md) | the strategic battery and limitation cells | | |
| [m5e_closeout_report.md](m5/m5e_closeout_report.md) | the final report: attackers, Holm family, corrections, configuration for M6 | | |

The newcomer's account of M5 is the living [research overview](../research_overview.md).

### M6: the demo (2026-10-19 to 10-22)

| document | contents |
|---|---|
| [m6_plan.md](m6/m6_plan.md) | the plan: milestones M6a to M6e, 13 correctness rules, wording rules (§9); M6d's remaining items and M6e are not done |
| [m6a_decision_record.md](m6/m6a_decision_record.md) | the user's requirements, the prototypes and the approved interface design |
| [m6a_technical_report.md](m6/m6a_technical_report.md) | the session core and its verification |
| [m6b_report.md](m6/m6b_report.md) | the web application and its browser verification |
| [m6c_guided_replays.md](m6/m6c_guided_replays.md) | the Sequences format and "Stacking to Crashing" |

The demo's code is `lanegame/demo/` and `web/`; its verification records are in `artifacts/m6/`.

### After M6

| document | contents | tables | code |
|---|---|---|---|
| [repository_audit_2026-10-22.md](audit/repository_audit_2026-10-22.md) | a read-only audit: defects, research-validity risks, wording drift, design limits | | |
| [nc1_preregistration.md](capacity/nc1_preregistration.md), [nc1_report.md](capacity/nc1_report.md) | the network capacity study (experimental; review open) | `artifacts/nc1/` | kept (`experiments/nc1/`) |
