# Experiments

One package per study, named after its milestone. Run from the repository root:

```bash
python -m experiments.<study> <stage> [<stage> ...] [--workers N]
```

The library is `lanegame/`; this package holds the study drivers and the research instruments built on it. The
[archive](../docs/archive/README.md) has every study's report, and [methodology](../docs/methodology.md) explains the
instruments.

## Studies you can run

| package | study | stages | records |
|---|---|---|---|
| `m5` | the final campaign: opponent modelling against pure self-play | `python -m experiments.m5 --help` lists them; cheap ones include `prereg_check`, `train_check`, `analyse`, `report`, `reproduce` | `artifacts/m5/` |
| `nc1` | network capacity (experimental) | `smoke prereg q1 q3 q2data q2fit q2behaviour q2analyse train evaluate q1pilot report`; training needs `--confirm-nc1-train` | `artifacts/nc1/` |
| `siege` | Phase 1: the exact game's siege study | `all` or `benchmark timing controlled siege_options crash_value vision_sweep ablations sensitivity equilibrium verification figures` | `artifacts/siege/` |
| `reference` | the engine-computed examples quoted by [the model](../docs/model.md) | `examples actions selfplay death report` | `artifacts/reference/` |
| `m6a` | the demo's research recordings re-checked, and its costs measured | `verify measure` | `artifacts/m6/` |
| `m6b` | the demo's visitor journey in headless Edge | `browser` (needs `cd web && npm run build`) | `artifacts/m6/m6b_browser.json`, screenshots in `runs/m6/m6b/` |
| `m6c` | the Sequences in headless Edge, and the site screenshots | `browser site` | `artifacts/m6/m6c_browser.json`, `artifacts/m6/site/` |
| `m6d` | every number of [the research overview](../docs/research_overview.md) checked against the M5 records | `overview` | `artifacts/m6/m6d_overview_check.json` |

Stages that cost real compute (training campaigns, batteries, attackers) refuse to start without a confirm flag such
as `--confirm-m5-train`. Passing one records that the compute was approved. Every M5 campaign stage also requires
the pre-registration chain to verify.

## Instruments kept from earlier studies

These packages are what remains of earlier studies after the reorganization of 2026-11-08: the modules that the final
campaign, the demo or the tests import. They have no command-line entry point. Their remaining stages, report
generators and drivers are at commit
[`f613f0a0`](https://github.com/jcj1337/lol_game_theory_laning/tree/f613f0a00a2d46fe717de7e8c79077fa9e0310c3).

| package | study | what it provides now |
|---|---|---|
| `m4` | M4.2e natural waves, zoning, wave actions | the scripted reactive policies and start states of `zoning_natural`, used by the scripted-strategy chain |
| `m43b` | M4.3b first PPO pilot | `run_jobs`, interventions, reference suites, trajectory export, the viewer's scripted policies |
| `m43c` | M4.3c four-action redesign and screens | the four-action strategy library and harness; `four_actions`, `wave_control` and `sequences` back the preset tests |
| `m43d` | M4.3d tower pressure | scripted strategy lookup |
| `m44` | M4.4 PPO readiness and training | action groups, metrics, records, counterfactuals, viewer recordings of trained play |
| `m44c` | M4.4c strategic investigation | multi-turn plans and opponents |
| `m45a` | M4.5a unified mechanics | hand-built start states; `movement` backs the revision-2 tests |
| `m45b` | M4.5b calibration | the rollout engine (`engine.play`), scripted strategies for `m45_calibrated()`, state builders; `timing` and `trades` back the calibration tests |
| `m45c` | M4.5c campaign | the strategic battery, horizon rollouts and critic values, training and health checks reused by M5 |
| `m46a` to `m46f` | M4.6 diagnosis studies | engines, state families, behaviour plans, the M4.6c archive starts and collector, cells, critic probes, feature fits (used by M5 and NC1) |
| `om1` | OM1 feasibility | shared definitions used by OM2 |
| `om2` | OM2 pilot | the treatment collector, the belief library, training and evaluation, reused as M5's arm A |

`common.py` holds the Markdown and environment helpers of the Phase 1 and M4.2 studies.

## Outputs

A study writes its curated summaries to `artifacts/<study>/` and everything else (checkpoints, raw per-job records,
partial results) to the git-ignored `runs/`, mostly `runs/checkpoints/<study>/`. Records written before 2026-11-08 name
files by the old layout (`results/...`); `lanegame.paths.relocated` maps them. A new study should write to `runs/`
and promote only what it publishes into `artifacts/`.
