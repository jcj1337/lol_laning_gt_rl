# M4.3a PPO smoke test on `GameConfig.m4_lasthit()` — integration checks, NOT research results

Commit `fbd69e10c9`, command `python.exe -m lanegame.experiments.rl.m43a_smoke all --workers 5`. Shared-policy self-play PPO (`lanegame.rl.shared`), `PPOConfig.continuing` (gamma 0.99, lambda 0.95, segments of 128 turns, games of 512 turns with seeded staggered first games, reward scale 0.1, value clip 0.5), 16 environments, snapshots every 25,600 agent transitions, pool 8, one torch thread per run; the runs were trained in parallel worker processes (throughput is per process under that load).

## Training runs

| run | current fraction | seed | agent transitions | env turns | updates | wall-clock s | transitions/s | env turns/s | collect share | snapshots |
|---|---|---|---|---|---|---|---|---|---|---|
| sp100_s0 | 1.0 | 0 | 409,600 | 204,800 | 100 | 76 | 5,388 | 2,694 | 0.74 | 16 |
| sp80_s0 | 0.8 | 0 | 410,001 | 223,232 | 109 | 87 | 4,700 | 2,559 | 0.77 | 16 |
| repro_s0a | 1.0 | 0 | 102,400 | 51,200 | 25 | 21 | 4,816 | 2,408 | 0.77 | 4 |
| repro_s0b | 1.0 | 0 | 102,400 | 51,200 | 25 | 21 | 4,793 | 2,397 | 0.76 | 4 |
| repro_s1 | 1.0 | 1 | 102,400 | 51,200 | 25 | 21 | 4,842 | 2,421 | 0.76 | 4 |

### Numerical health

| run | finite | entropy first → last | policy loss first → last | value loss first → last 10 | explained variance first → last 10 | grad norm mean / max (pre-clip) | clip frac | approx KL mean / max |
|---|---|---|---|---|---|---|---|---|
| sp100_s0 | True | 2.58 → 0.83 | -0.008 → -0.000 | 0.014 → 0.002 | -2.04 → 0.70 | 0.40 / 1.05 | 0.028 | 0.0028 / 0.0114 |
| sp80_s0 | True | 2.58 → 0.80 | -0.008 → -0.000 | 0.014 → 0.002 | -2.04 → 0.75 | 0.39 / 3.96 | 0.028 | 0.0027 / 0.0107 |
| repro_s0a | True | 2.58 → 1.41 | -0.008 → -0.000 | 0.014 → 0.002 | -2.04 → 0.62 | 0.47 / 1.28 | 0.045 | 0.0047 / 0.0089 |
| repro_s0b | True | 2.58 → 1.41 | -0.008 → -0.000 | 0.014 → 0.002 | -2.04 → 0.62 | 0.47 / 1.28 | 0.045 | 0.0047 / 0.0089 |
| repro_s1 | True | 2.58 → 1.26 | -0.007 → -0.001 | 0.018 → 0.002 | -2.84 → 0.58 | 0.47 / 1.34 | 0.053 | 0.0051 / 0.0120 |

### Critic scale (resources; the scaled units of training divided by the reward scale)

| run | V̂(s0) | mean V̂ (last 10 rollouts) | max |V̂| (whole run) | std of λ-returns (last 10) | mean |r| per turn | Ω (|V| bound) |
|---|---|---|---|---|---|---|
| sp100_s0 | -0.25 | +0.02 | 10.66 | 1.17 | 0.511 | 107.04 |
| sp80_s0 | +0.03 | +0.03 | 10.66 | 1.19 | 0.592 | 107.04 |
| repro_s0a | +0.17 | +0.27 | 10.66 | 1.09 | 0.482 | 107.04 |
| repro_s0b | +0.17 | +0.27 | 10.66 | 1.09 | 0.482 | 107.04 |
| repro_s1 | +0.56 | +0.15 | 10.76 | 1.02 | 0.462 | 107.04 |

By symmetry the self-play value of s0 is 0 (the critic sees the same observation from both seats).

### Seats, masks and time semantics

| run | transitions seat A / B | historical transitions | single-choice (absent) transitions | games finished | historical games | truncations | terminations | mean seat-A return (current games) | mean learner return (historical games) |
|---|---|---|---|---|---|---|---|---|---|
| sp100_s0 | 204,800 / 204,800 | 0 | 39,866 | 400 | 0 | 400 | 0 | +0.42 | n/a |
| sp80_s0 | 204,280 / 205,721 | 36,463 | 44,062 | 436 | 71 | 436 | 0 | -0.02 | +9.03 |
| repro_s0a | 51,200 / 51,200 | 0 | 6,790 | 100 | 0 | 100 | 0 | -0.03 | n/a |
| repro_s0b | 51,200 / 51,200 | 0 | 6,790 | 100 | 0 | 100 | 0 | -0.03 | n/a |
| repro_s1 | 51,200 / 51,200 | 0 | 5,943 | 98 | 0 | 98 | 0 | +0.19 | n/a |

Masks: `LaneEnv.step` raises on any illegal action, so a completed run is itself the proof that every sampled action was legal; the recorded masks are checked against the engine's legal sets in `tests/test_rl_m4_lasthit.py`. Truncations: one per finished game (the game time limit), none treated as a termination; the bootstrap values are checked in the same test file.

### Reproducibility

| run | seed | weights digest (SHA-256, first 16) | torch | commit |
|---|---|---|---|---|
| sp100_s0 | 0 | abefaf9d31998179 | 2.14.1+cpu | fbd69e10c9 |
| sp80_s0 | 0 | 73098fa15862a6d0 | 2.14.1+cpu | fbd69e10c9 |
| repro_s0a | 0 | 69069a2dd1507f72 | 2.14.1+cpu | fbd69e10c9 |
| repro_s0b | 0 | 69069a2dd1507f72 | 2.14.1+cpu | fbd69e10c9 |
| repro_s1 | 1 | dd01f810ee5294d8 | 2.14.1+cpu | fbd69e10c9 |

## Checks

| check | passed |
|---|---|
| sp100_s0: finite losses, gradients and critic statistics | True |
| sp100_s0: never terminated; one truncation per finished game | True |
| sp100_s0: critic bounded by the return scale (|V| <= Omega) | True |
| sp100_s0: explained variance improved and is positive | True |
| sp100_s0: gradient norms finite and of order 1 (max < 100) | True |
| sp100_s0: approximate KL small (mean < 0.05) | True |
| sp80_s0: finite losses, gradients and critic statistics | True |
| sp80_s0: never terminated; one truncation per finished game | True |
| sp80_s0: critic bounded by the return scale (|V| <= Omega) | True |
| sp80_s0: explained variance improved and is positive | True |
| sp80_s0: gradient norms finite and of order 1 (max < 100) | True |
| sp80_s0: approximate KL small (mean < 0.05) | True |
| sp100_s0: both seats trained equally (exact) | True |
| sp100_s0: no historical games | True |
| sp80_s0: historical share near 0.2 of games after the first snapshot | True |
| sp80_s0: both seats trained (seat shares within 0.45-0.55) | True |
| reproducibility: same configuration and seed give bit-identical weights | True |
| reproducibility: ... and identical histories | True |
| reproducibility: a different seed gives different weights | True |

**All checks passed: True.**

## Evaluation readout (protocol E1, development window)

development window W = 460: the truncation error is bounded by the guaranteed tail bound; 16 paired episodes per matchup: an integration readout, not a strategy measurement. Guaranteed truncation bound at W = 460: 1.05 resources; the Monte Carlo interval is the 95% paired CI over seeds. Payoff to x (discounted, resources).

| x | y | estimate | 95% CI | x in seat A | x in seat B |
|---|---|---|---|---|---|
| sp100_s0 | hold_farmer | +11.31 | [+10.92, +11.70] | +11.21 | +11.40 |
| sp100_s0 | slow_pusher | +10.44 | [+10.01, +10.87] | +10.60 | +10.28 |
| sp100_s0 | trading_pusher | +17.37 | [+17.03, +17.71] | +17.17 | +17.57 |
| sp80_s0 | hold_farmer | +11.22 | [+10.85, +11.60] | +11.29 | +11.16 |
| sp80_s0 | slow_pusher | +10.42 | [+9.97, +10.86] | +10.31 | +10.53 |
| sp80_s0 | trading_pusher | +18.89 | [+18.39, +19.39] | +19.02 | +18.76 |
| initial_s0 | hold_farmer | -5.09 | [-5.87, -4.30] | -5.25 | -4.92 |
| initial_s0 | slow_pusher | -6.62 | [-7.38, -5.85] | -6.55 | -6.69 |
| initial_s0 | trading_pusher | +6.00 | [+4.99, +7.01] | +5.22 | +6.78 |
| sp100_s0 | sp80_s0 | -0.80 | [-1.27, -0.33] | -0.59 | -1.02 |
| sp100_s0 | initial_s0 | +11.62 | [+11.09, +12.16] | +11.95 | +11.30 |
| sp80_s0 | initial_s0 | +12.02 | [+11.43, +12.61] | +12.37 | +11.68 |

### What the policies do in self-play (8 games of 460 turns, present turns)

| policy | Hold / Clear / Thin | H / Y / T / S / R | Pressure / Vision / none | absent | recalls | connected trades | deaths | farm | plates |
|---|---|---|---|---|---|---|---|---|---|
| sp100_s0 | 0.08 / 0.82 / 0.10 | 0.54 / 0.14 / 0.27 / 0.05 / 0.00 | 0.04 / 0.01 / 0.95 | 0.11 | 0.1 | 91.4 | 17.5 | 159.5 | 3.0 |
| sp80_s0 | 0.03 / 0.82 / 0.15 | 0.58 / 0.11 / 0.27 / 0.04 / 0.00 | 0.03 / 0.01 / 0.96 | 0.11 | 1.4 | 87.8 | 17.1 | 171.9 | 3.0 |
| initial_s0 | 0.36 / 0.32 / 0.32 | 0.25 / 0.23 / 0.24 / 0.22 / 0.05 | 0.02 / 0.02 / 0.96 | 0.09 | 18.2 | 64.9 | 8.4 | 144.1 | 2.5 |

The behaviour columns are a readout after a bounded budget; strategic claims need the M4.3b pilot.
