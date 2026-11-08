# The reorganization of 2026-11-08

The repository was restructured around the final research: a library split by concept, the study packages moved out
of it and pruned, curated records in `artifacts/`, generated outputs in a git-ignored `runs/`, and the milestone
documents moved into this archive. No rule, parameter, preset, reward, objective, checkpoint or stored result changed.

The tree before the change is commit
[`f613f0a0`](https://github.com/jcj1337/lol_game_theory_laning/tree/f613f0a00a2d46fe717de7e8c79077fa9e0310c3).
[moves.json](moves.json) lists every path: moved unchanged (with SHA-256), moved and edited (with both hashes),
removed (with the reason) and added.

## Path map

| before | after |
|---|---|
| `lanegame/{config,state,mechanics,game,champions,champions_m4,progression,positional,lane_enum,render}.py` | `lanegame/engine/` |
| `lanegame/{solver,matrix_game,best_response,policy,analysis,baselines,simulate,reachability,scenarios}.py` | `lanegame/exact/` |
| `lanegame/scenarios_m4.py`, `lanegame/data/m4_scenarios.json` | `tests/m4_scenarios.py`, `tests/data/` (test support) |
| `lanegame/viewer/policies.py` | `experiments/m43b/viewer_policies.py` |
| `lanegame/viewer/app.py`, `render.py` (the Streamlit UI) | removed |
| `lanegame/experiments/rl/<study>/`, `lanegame/experiments/<study>/` | `experiments/<study>/` (pruned) |
| `python -m lanegame.experiments.rl.<study> ...` | `python -m experiments.<study> ...` |
| `python -m lanegame experiments-siege ...` | `python -m experiments.siege ...` |
| `results/rl/checkpoints/` (git-ignored) | `runs/checkpoints/` (git-ignored) |
| the 7 demo networks in `results/rl/checkpoints/` | `artifacts/checkpoints/` (same relative paths) |
| `results/<study>/` | `artifacts/<study>/` (curated) |
| `results/<file>` (the v1 study's tables) | `artifacts/v1/` |
| `results/siege/solution_K32.json` (68 MB) | `artifacts/siege/solution_K32_summary.json` |
| `docs/m4/laning_game_mathematical_reference.md` | `docs/model.md` |
| `docs/m6/research_overview.md` | `docs/research_overview.md` |
| `docs/rl_research_roadmap.md` | `docs/archive/rl_research_roadmap.md` |
| `docs/README.md` | `docs/archive/docs_index_2026-10-29.md` (a new `docs/README.md` replaces it) |
| `docs/{phase1,rl,m4,natural_waves,opponent_modelling,m5,m6,audit,capacity}/` | `docs/archive/` (same structure) |
| `requirements.txt` | removed (it duplicated `pyproject.toml`) |

Records written before this date keep the old paths; `lanegame.paths.relocated` applies this map, and M5's gates use it
through `experiments/m5/common.resolve`.

## What was verified

- Every archived document (90 files) and every curated artifact but one (365 of 366) is byte-identical to its
  original; the SHA-256 values are in moves.json. The exception is `artifacts/m6/m6d_overview_check.json`, regenerated because the
  overview it checks moved (48 of 48 claims still pass; only the document paths in it changed).
- Three documents that existed only on disk, because the old `.gitignore` ignored `docs/`, are now tracked:
  `research_overview.md` and the NC1 pre-registration and report (their hashes match the files before the move).
- Regenerating records from the new layout reproduces them: `python -m experiments.m5 report` rebuilds
  `artifacts/m5/report.md` with every number identical, and `python -m experiments.reference examples actions report`
  reproduces the model's examples value for value (only the template strings naming paths, the commit and the timing
  differ, so the original bytes were kept).
- The M5 pre-registration chain verifies (`python -m experiments.m5 prereg_check`), and the training check re-hashes
  all 12 runs' checkpoints from the new locations.
- The K = 32 siege solution re-solves to the recorded value and 172,907 states (`pytest --runslow`).

## Removed

| what | files | why |
|---|---|---|
| study code | 194 modules | nothing maintained imports them: the v1 and revision experiments, the M2 champion experiments, the M3 and OpenAI Five scripts, most of the M4.2 to M4.3 study packages, M4.7 and M4.8's drivers, OM1's stages, and every study's report generators and command-line entry points |
| the Streamlit UI | `lanegame/viewer/app.py`, `render.py`, two skill journeys | superseded by the web demo, as the M6 plan scheduled; the recording format it used stays in `lanegame/viewer/session.py` |
| the M6a glass prototype | 5 files in `legacy/` | it did not run; `web/` implements the approved design, and [m6a_decision_record.md](m6/m6a_decision_record.md) records it |
| results | 1,548 files, 326 MB | raw per-job records, partial results, run logs, launcher scripts, prototype and journey screenshots and the 68 MB K = 32 solution, all regenerable; each study's tables, figures and protocol records are kept in `artifacts/` |
| tests | 109 test functions, 2 whole files | they tested study code that was removed (report generators, stage gates, study-specific analyses); see below |

The local, git-ignored data was moved, not deleted: 51,805 checkpoints from `results/rl/checkpoints/` to
`runs/checkpoints/`, the M4.3b checkpoint bundle to `runs/m43b_artifacts/` and twelve M4.4c raw records to
`runs/m44c/`.

### Retired tests

Every retired test exercised code removed with its study. Tests of the engine, the presets, the environment, PPO, the
opponent belief, the lane-exit readout and the demo were all kept. Where a kept test needed a small helper from a
removed study module, the helper moved into the test file (the M4.2d-D investment curve's `READY`, the M4.2d-A
scripted driver, the M4.2e zoning start states), and the four-action, Hold-cap, A3-sequence, M4.5a movement and
M4.5b timing and trade modules stayed because their tests pin mechanics of the research preset.

| file | retired | what they tested |
|---|---|---|
| `test_m43d_tower_pressure.py` | all 11 | the M4.3d ledgers, crash grid and audit tables |
| `test_om1.py` | all 7 | OM1's own belief, library and stage code (OM2's belief in `lanegame/opponent/` is tested by `test_om2.py`) |
| `test_m44c_strategic.py` | 15 of 16 | M4.4c's engine, plans, states and CLI |
| `test_m47.py` | 10 of 11 | M4.7's streams, returns and training variants (the GAE truncation test stays) |
| `test_m43c_diagnostics.py` | 9 of 11 | M4.3c's rollouts, examples, freeze and XP-range screens |
| `test_m48.py` | 7 of 18 | M4.8's stage code (the lane-exit readout's tests stay) |
| `test_m4_calibration.py` | 6 of 7 | the M4.2b candidate preset, never adopted |
| `test_m45c_full.py` | 6 of 18 | M4.5c's CLI gates and health-report rules |
| `test_rl_m43b.py` | 6 of 11 | M4.3b's behaviour, recall, archive and plan analyses |
| `test_diagnostics.py` | 4 of 24 | the M3 diagnostics study's overlays and attackers |
| `test_m44b_analysis.py` | 4 of 9 | M4.4b's report and CLI |
| `test_m46f.py` | 3 of 12 | M4.6f's training, analysis and verification stages |
| `test_rl_m43b_validation.py` | 3 of 13 | M4.3b's validation stage |
| `test_m4_emergent_ready.py`, `test_m4_farm_spot_rule.py`, `test_m45a_mechanics.py`, `test_m45a_movement.py`, `test_viewer.py` | 2 each | study findings of M4.2d, the M4.5a demonstrations and the Streamlit renderer |
| `test_champion_validation.py`, `test_m43c_economics.py`, `test_m44a_readiness.py`, `test_m45c_training.py`, `test_m46d.py`, `test_m46e.py`, `test_m4_freeze_gate.py`, `test_m4_zoning.py` | 1 each | a removed study module or CLI gate each |

The suite went from 1,617 passed tests (plus 4 slow skips and 2 expected failures) to 1,494 with the same skips and
expected failures; the difference is the retired tests and their parametrized cases.
