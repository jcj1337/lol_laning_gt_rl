# Exact game

The exact game (Phase 1) lets a user solve the wave-management game by backward induction, print the equilibrium at
the start state, measure how exploitable the scripted baselines are, run the hand-built scenario checks, compare Monte
Carlo play with the exact expected payoff, and list which events are reachable. It covers three rulesets: `siege`
(default, `GameConfig()`), `revised` (`GameConfig.v2()`) and `v1`.

## Sub-features

- `exact-solve` solves a ruleset at horizon K, prints V*(s0), the node counts and the certified gap, and writes the
  solution JSON.
- `exact-exploit` prints every baseline's exploitability and its payoff against the equilibrium.
- `exact-scenarios` runs the scenario checks, prints PASS or FAIL per scenario, and exits 1 if any fails.
- `exact-simulate` plays two policies by Monte Carlo and prints the exact payoff beside the estimate.
- `exact-reach` counts reachable decision nodes and prints a witness plan for each event.

## How to get to it (user POV)

- `python -m lanegame solve [--model siege|revised|v1] [--horizon K] [--out FILE] [--no-q]`
- `python -m lanegame exploit [--model ...] [--horizon K]`
- `python -m lanegame scenarios [--model ...]`
- `python -m lanegame simulate --a <policy> --b <policy> [--model ...] [--horizon K] [--episodes N] [--seed S]`
- `python -m lanegame reach [--model ...] [--horizon K]`

## Driving it with vl.py

Preconditions:

- A run exists (`RUN`) and `$VL doctor --run "$RUN"` exits 0. No viewer or browser is needed.

- **Solve siege.** Run `$VL cli --run "$RUN" --label exact-solve-siege -- solve --horizon 8 --out solve_siege_K8.json --no-q`.
  Exit 0. Stdout starts `[siege] horizon 8: V*(s0) = 0; 189 decision nodes, 16 mixed; max certified gap` followed by a
  value at most 1e-9. The next line is the equilibrium at s0, then `wrote solve_siege_K8.json`. The transcript lists
  `solve_siege_K8.json` with its SHA-256.
- **Solve the other rulesets.** Run `$VL cli --run "$RUN" --label exact-solve-revised -- solve --model revised --horizon 8 --out solve_revised_K8.json --no-q`
  and `$VL cli --run "$RUN" --label exact-solve-v1 -- solve --model v1 --horizon 8 --out solve_v1_K8.json --no-q`.
  Both exit 0 with `V*(s0) = 0`: revised reports 193 decision nodes (16 mixed), v1 reports 69 (4 mixed). Each writes
  its JSON.
- **Exploitability.** Run `$VL cli --run "$RUN" --label exact-exploit --tail 20 -- exploit --horizon 8`. Exit 0. The
  first line is `[siege] V*(s0) = 0`, then one row per baseline. Every exploitability is >= 0 and every "vs
  equilibrium" value is <= 0. On the reference run: `uniform` 1.150 / -1.077, `aggressive` 4.283 / -4.098,
  `stack_builder` 0.335 / -0.320.
- **Scenario checks.** Run `$VL cli --run "$RUN" --label exact-scenarios --tail 5 -- scenarios`. Exit 0. Then run
  `grep -c '^\[FAIL\]' "$RUN/evidence/exact-scenarios.stdout.txt"` (prints 0) and
  `grep -c '^\[PASS\]' "$RUN/evidence/exact-scenarios.stdout.txt"` (prints the scenario count, 23 on the reference
  run). The first scenario is
  `[PASS] even_lane_fight (reachable start)`.
- **Monte Carlo against exact.** Run `$VL cli --run "$RUN" --label exact-simulate -- simulate --horizon 8 --a aggressive --b passive_safe --episodes 2000 --seed 0`.
  Exit 0. The first line is `exact -4.0600; Monte Carlo -4.1275 +- 0.1802`. The estimate is within 3 standard errors
  of the exact value.
- **Reachability.** Run `$VL cli --run "$RUN" --label exact-reach -- reach --horizon 8`. Exit 0. The first line is
  `[siege] 189 reachable decision nodes, 126 distinct states`, the same node count `solve` reported. Each event row
  shows a transition count, the first turn, and a witness plan or `-`.
- **Unknown policy.** Run `$VL cli --run "$RUN" --label exact-simulate-bad -- simulate --horizon 8 --a nosuch`. The
  exit code is non-zero and stderr names `unknown policy 'nosuch'` with the valid baseline names.
- **Proof.** The `exact-*` transcripts, `exact-solve-*.files/*.json` and a clean repo guard.

## Gotchas

- Siege (the default) at the default `--horizon 32` takes minutes and gigabytes. Always pass `--horizon 8`.
- Without `--out`, `solve` writes `results/<ruleset>/solution_K<K>.json` relative to the working directory. Under
  `vl.py` that is inside `RUN/work/cli`, but pass `--out` anyway so the evidence names the file.
- V*(s0) = 0 is expected: the game is symmetric. A non-zero value at the start state is a finding.
- `scenarios` prints a lane diagram per scenario and its output is long. Assert with `grep` on the stdout file, not on
  the printed tail.
- The reference numbers above were observed on 2026-10-18 at commit 82568827. A rule change can legitimately move
  them. Check the change against `docs/m4/laning_game_mathematical_reference.md` before calling it a regression, and
  never update the digest pins in `tests/test_legacy_pins.py` to match.
