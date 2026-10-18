# lanegame verification map

This folder is the maintained source for verifying lanegame's user-facing behaviour. Read this index before driving
anything, then use the matching feature file as the recipe. The harness is `vl.py` (see `../SKILL.md`). Every command
below assumes:

```bash
VL=".venv/Scripts/python.exe .claude/skills/verify-lanegame/scripts/vl.py"
RUN=$($VL new-run | sed -n 's/^RUN=//p')
```

## Baseline preconditions

- Run from the repo root, with the venv at `.venv/` and lanegame installed editable.
- `$VL doctor --run "$RUN"` exits 0, and its "repo untouched since new-run" line is PASS.
- CLI journeys need nothing else. Every `$VL cli` call runs in `RUN/work/cli`, so relative output paths stay out of
  the repo.
- Viewer journeys also need `$VL viewer start`, `$VL browser start` and `$VL browser open` to have succeeded in this
  run. A fresh `browser open` lands in Strategy matchup on the first matchup, `M4.5-1. Ward from a neutral lane`, at
  `Turn 0 of 30 · start · k = 1 · equal power · seed 0`.
- Never drive a viewer or browser this run did not start, and never pass a `--confirm-*` flag.

## Driving conventions

- Start each recipe from its stated preconditions. For the viewer, set every setup field the journey depends on
  explicitly: Streamlit session state carries values across modes and matchups.
- Treat commands as literal. Keep quoted texts, keys and flags unchanged.
- Prefer widget keys (`--key`) and exact visible text over `--contains` and `--nth`.
- Read selectbox and number values with `browser widgets`, never from the page text.
- The state line is the second line of `browser text --scope main` (`Turn i of n · frame · k = … · gap · seed s`).
- Exact-game journeys use `--horizon 8`, and training journeys use `--steps 4096 --n-envs 4`. Bigger budgets add
  nothing to a proof and can take hours.

## Proof and skip reporting

- CLI proof is the `$VL cli` transcript: the command, exit code, stdout, stderr and the written files with SHA-256.
- UI proof is the `actions.log` entries for the actions, plus `browser text` or `browser widgets` after each one that
  changes state, plus a `browser screenshot` of the end state (its PNG shows the sidebar title "Lane debugger").
- Determinism proof pairs two runs with the same seed and the same arguments, and compares their hashes or numbers.
- Record the feature file and sub-feature ID with every artifact label, for example `--label env-demo-trace-a`.
- Report a journey that could not run with the attempted command and the unmet precondition. Never report it as
  verified through a different entry point.
- Report a journey whose observed end state differs from its file as a FAIL with the observed state, even when the
  file's Gotchas already record the failure.

## Feature entry contract

Each feature file starts with an H1 title and one paragraph describing the user-visible behaviour, then has exactly
four H2 sections in this order:

1. `Sub-features`: short IDs, one line each.
2. `How to get to it (user POV)`: every user entry point.
3. `Driving it with vl.py`: starts with `Preconditions:`, then labelled bullets that pair each user action with an
   exact command and the observable result.
4. `Gotchas`: traps that can waste or invalidate a run, including journeys known to fail and when that was observed.

Keep implementation details out. Name only user paths, stable handles, required state, commands and observable proof.

## Features

- [Exact game](./exact-game.md): solve, exploitability, the scenario checks, Monte Carlo against exact, and
  reachability, for the siege, revised and v1 rulesets.
- [Environment demo](./env-demo.md): scripted agents through the RL environment, exact agreement, episode traces,
  same-seed determinism, and champion mode.
- [Training and evaluation](./rl-train-eval.md): shared-policy PPO self-play to a checkpoint, bit-identical
  retraining, and head-to-head evaluation.
- [Viewer replay](./viewer-replay.md): Strategy matchup and Recorded trajectory playback, turn and tick navigation,
  and the detail tabs.
- [Viewer manual play](./viewer-manual.md): playing side A by hand, undo, auto-play, branching, and taking over from
  a matchup.
