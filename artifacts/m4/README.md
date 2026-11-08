# M4.2 validation results

Produced by the drivers in `lanegame/experiments/m4/` on branch `implementation/m4-integrated`. Every driver writes
the raw measurements as JSON first (with the commit, the command line, the configuration and the seeds), then a
Markdown summary. All runs are seeded and reproducible on one machine (scripted agents are deterministic; paired
rollouts share environment seeds). Run from the repository root with the project virtual environment.

| File | Stage | Command | Wall time (12-core machine) |
|---|---|---|---|
| `soak.{json,md}` | S1 long-horizon soak (68 000 turns, every invariant asserted every turn) | `python -m lanegame.experiments.m4.soak --turns 50000` | 30 s |
| `progression.{json,md}` | S2 progression speed (400 + 400 seeded 400-turn games) | `python -m lanegame.experiments.m4.progression --games 400 --turns 400 --procs 8` | 10 s |
| `freeze_gate.{json,md}` | S3a mechanics + S3b Freeze gate (163 scenarios x 21 policy pairs x 50 paired rollouts x 120 turns; exhaustive lane comparison) | `python -m lanegame.experiments.m4.freeze_gate --rollouts 50 --turns 120 --procs 10` | 12.5 min |
| `economy.{json,md}` | S4 economic sanity (paired counterfactuals, W = 1000; item diagnosis; plates; snowballing; E1 round robin) | `python -m lanegame.experiments.m4.economy --rollouts 60 --window 1000 --rr-episodes 60 --procs 10` | 5.5 min |
| `benchmark.{json,md}` | throughput and memory (single thread, idle machine) | `python -m lanegame.experiments.m4.benchmark --turns 40000` | 3 min |
| `smoke.{json,md}` | bounded training smoke checks (m2c stepping stone, M4 integration); NOT research results | `python -m lanegame.experiments.m4.smoke --m2c-steps 300000 --m4-steps 50000` | 1 min |

The test suite (S0, S1 unit and integration tests): `python -m pytest -q --runslow` (679 passed, 2 strict expected
failures recording design findings; about 19 minutes, of which the exhaustive enumerations B1, O1, O2 and the K = 32
re-solve take about 11).

The interpretation of these results is in [docs/m4/m4_implementation_report.md](../../docs/m4/m4_implementation_report.md).
