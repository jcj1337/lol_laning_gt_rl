# M4.3b research artifacts: what is where, and how to reproduce

This directory holds the results of the M4.3b PPO self-play pilot and of its follow-up (`validation/`). The policy
for generated artifacts (adopted in the follow-up, brief §7):

| kind | where | tracked in Git |
|---|---|---|
| source, tests, configuration, reports | `lanegame/`, `tests/`, `docs/` | yes |
| stage result files (JSON), figures, logs, representative trajectories | `results/rl/m43b/` | yes; JSON files ≥ 1 MB are gzipped (`*.json.gz`) |
| the training start archive of the `mixed` condition (857 witnessed states) | `results/rl/m43b/archive_train.json.gz` | yes (188 KB gzipped; 7 MB raw) |
| trained networks: finals, attackers, run configs, training histories | `results/rl/checkpoints/m43b/` | **no** (`.gitignore`); preserved through the bundle below |
| training snapshots (80 per run, 150 MB) | `results/rl/checkpoints/m43b/*/snapshots/` | no; regenerable by retraining with the recorded seeds, needed only for the learning curves and the self-play progress matchups |
| the checkpoint bundle | `results/rl/m43b_artifacts/m43b_checkpoints_<commit>.tar.gz` (+ `.sha256`) | no; keep a copy outside the working tree (release asset, shared drive) |
| the manifest | `results/rl/m43b/MANIFEST.json` | yes |

Every reader in the package accepts a plain or a gzipped JSON file (`common.load`, `common.read_json`,
`env.starts.load_archive`), so gzipping a result never breaks a stage, a figure or the report.

## The manifest

`python -m lanegame.experiments.rl.m43b manifest` writes `MANIFEST.json` and the bundle. The manifest lists:

- every result file with its size and SHA-256 and, where the file carries a `meta` block, the commit, timestamp,
  stage, window, seeds and the command that produced it;
- every checkpoint file (finals, attackers, run configs, histories; snapshots on request) with its size, SHA-256
  and the **weights digest** that `lanegame.rl.load_agent` verifies on load (equal digests ⇔ identical weights),
  so a number in a report can be traced to the exact network it was computed with;
- the PPO and shared-self-play configuration of every run (`run.json`).

`manifest.verify_bundle(path)` checks a bundle against the manifest without extracting it.

## Which commit produced what

| results | commit | stage command |
|---|---|---|
| `train.json`, `curves.json`, `headline.json`, `exploit.json`, `behavior.json`, `counterfactual.json.gz`, `recall.json.gz`, `plans.json`, `trajectories.json`, `trajectories/`, `archive.json`, `archive_train.json.gz`, figures `fig_*.png`, `report.md` | 1ad8848 (M4.3a) at run time, committed as 98a4341 (M4.3b) | `python -m lanegame.experiments.rl.m43b <stage> --workers 9` (see `MANIFEST.json` per file) |
| `validation/*.json`, `validation/fig_val_*.png`, `validation/report.md` | the follow-up commit (the `meta.commit` field of each file) | `python -m lanegame.experiments.rl.m43b val_wave_cf val_horizon val_games val_freeze val_report --workers 9` |

Seeds: training seeds 0, 1, 2 (attackers 700 +); every evaluation stream is a `SeedSequence` root in 61,000–69,999
(`common.py`; the follow-up uses 67,500–67,999, `validation.py`). All evaluation stages are resumable (a result
already in the stage file is reused) and deterministic given the checkpoints.

## Reproducing without retraining

1. Check out the commit named in the stage file's `meta.commit`.
2. Extract the bundle into the repository root (it unpacks to `results/rl/checkpoints/m43b/...`) and verify it:
   `python -c "from lanegame.experiments.rl.m43b.manifest import verify_bundle; print(verify_bundle('results/rl/m43b_artifacts/m43b_checkpoints_98a4341.tar.gz'))"`.
3. Run the stage command. The learning curves (`curves`) and the self-play progress matchups (`headline`, the
   `vs_snapshots` block) need the snapshots, which are only in a `_full` bundle or in a fresh training run.

## Retraining

`python -m lanegame.experiments.rl.m43b train --workers 9` (13 min for the nine pilot runs on a 12-CPU
workstation) reproduces the runs bit-for-bit only on the same PyTorch build and CPU; on another machine expect
different weights with the same statistics. The `mixed` condition needs `archive_train.json.gz`.
