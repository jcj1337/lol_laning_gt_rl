# Environment demo

The environment demo lets a user pit two scripted agents against each other through the simultaneous-move RL
environment. It prints the payoff distribution, farm, ganks and action frequencies, optionally checks the Monte Carlo
mean against the exact expected payoff, and writes the first episode as a trace. It runs on the exact-game rulesets and
on champion mode, which has no exact solver.

## Sub-features

- `env-demo-pair` plays a scripted pair for N episodes and prints the statistics block.
- `env-demo-exact` adds the exact expected payoff and the z-score of the Monte Carlo mean (`--exact`).
- `env-demo-trace` writes the first episode as `lanegame.episode` JSON (`--trace`).
- `env-demo-determinism` reproduces: the same seed and arguments give the same trace, byte for byte.
- `env-demo-champions` plays champion-mode agents (`--model champions`).
- `env-demo-bad-agent` rejects an unknown agent name with the list of valid names.

## How to get to it (user POV)

- `python -m lanegame env-demo [--model siege|revised|v1|champions] --a <agent> --b <agent> [--horizon K] [--episodes N] [--seed S] [--exact] [--trace FILE]`

## Driving it with vl.py

Preconditions:

- A run exists (`RUN`) and `$VL doctor --run "$RUN"` exits 0. No viewer or browser is needed.

- **Pair, exact and trace.** Run `$VL cli --run "$RUN" --label env-demo-trace-a -- env-demo --horizon 8 --episodes 200 --seed 42 --exact --trace trace42.json`.
  Exit 0. Stdout contains `aggressive (A) vs passive_safe (B): 200 episodes, 1600 turns`,
  `payoff to A: mean -4.0600 +- 0.5511`, `observation size 34, action space 18`, and
  `exact expected payoff -4.0600; Monte Carlo -4.0600 +- 0.5511 (z = +0.00)`; |z| must be below 3. The transcript lists
  `trace42.json` with its SHA-256.
- **Trace content.** Open `$RUN/evidence/env-demo-trace-a.files/trace42.json`. It is JSON with
  `"schema": "lanegame.episode"`, `"version": 1` and a `config` whose `horizon` is 8.
- **Same seed, same trace.** Run `$VL cli --run "$RUN" --label env-demo-trace-b -- env-demo --horizon 8 --episodes 200 --seed 42 --trace trace42b.json`.
  Then run `sha256sum "$RUN"/evidence/env-demo-trace-a.files/trace42.json "$RUN"/evidence/env-demo-trace-b.files/trace42b.json`.
  The two hashes are equal (`885f1c19…` on the reference run), and the payoff line matches the first run.
- **Different seed.** Run `$VL cli --run "$RUN" --label env-demo-seed43 -- env-demo --horizon 8 --episodes 200 --seed 43`.
  Exit 0. The run completes with the same block structure, and its payoff line is reported beside seed 42's.
- **Champion mode.** Run `$VL cli --run "$RUN" --label env-demo-champions -- env-demo --model champions --a trade --b hold --episodes 50 --seed 7`.
  Exit 0. The first line ends `no exact solver in champion mode`. Stdout contains
  `trade (A) vs hold (B): 50 episodes, 1600 turns`, `payoff to A: mean +0.7728 +- 0.2288`, the trades, recalls and
  deaths lines, and `observation size 50, action space 46`.
- **Unknown agent.** Run `$VL cli --run "$RUN" --label env-demo-bad-agent -- env-demo --model champions --a nosuch --episodes 5`.
  Exit 1. Stderr is `unknown champion-mode agent 'nosuch'; choose from [...]`, a list that includes `hold`, `trade`,
  `shove_trade` and the `legacy:` names.
- **Proof.** The `env-demo-*` transcripts, the two trace copies with equal hashes, and a clean repo guard.

## Gotchas

- `--exact` enumerates the reachable game. It is cheap at K <= 16 and slow at the default K = 32. Keep `--horizon 8`.
- Naming `equilibrium` as an agent makes the demo solve the game first, which takes minutes and gigabytes at K = 32
  for siege. Avoid it unless `--horizon` is small.
- The last stats line (`… s: … episodes/s, … turns/s`) is timing and differs between runs. Compare payoff lines and
  trace hashes, not whole stdout files.
- Champion mode ignores `--exact`: there is no exact solver for it.
- Reference numbers were observed on 2026-10-18 at commit 82568827. A mechanics change can move them; a same-seed
  mismatch between two runs of one checkout cannot, and is a bug.
