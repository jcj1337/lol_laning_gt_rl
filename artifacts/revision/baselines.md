# Experiment F: baseline policies (K = 32)

Equilibrium value V*(s0) = 0. Baselines take the resource option after a crash (uniform randomizes it).

- **uniform**: uniform over legal combined actions.
- **aggressive**: Shove + Contest every turn.
- **freeze_contest**: Freeze + Contest whenever legal, otherwise Slow Push + Safe.
- **passive_safe**: Slow Push + Safe every turn.
- **stack_builder**: Slow Push + Contest until the leading wave has 5+ HP, then Shove + Contest.
- **release_safe**: Shove + Safe every turn (lets waves walk in, never follows them).
- **contest_slowpush**: Slow Push + Contest every turn (the original always_slowpush).

## Head-to-head (exact expected payoff to the row player A)

| A \ B | equilibrium | uniform | aggressive | freeze_contest | passive_safe | stack_builder | release_safe | contest_slowpush |
|---|---|---|---|---|---|---|---|---|
| equilibrium | 0 | 9.86 | 24.47 | 2.87 | 2.87 | 0.32 | 2.8 | 0.32 |
| uniform | -9.86 | -0 | 22.35 | -6.09 | -5.36 | -8.07 | -0.84 | -8.08 |
| aggressive | -24.47 | -22.35 | 0 | -24.49 | -24.22 | -24.76 | -14.39 | -24.76 |
| freeze_contest | -2.87 | 6.09 | 24.49 | 0 | 0 | -2.38 | 0 | -2.38 |
| passive_safe | -2.87 | 5.36 | 24.22 | 0 | 0 | -2.38 | 0 | -2.38 |
| stack_builder | -0.32 | 8.07 | 24.76 | 2.38 | 2.38 | 0 | 2.38 | 0 |
| release_safe | -2.8 | 0.84 | 14.39 | 0 | 0 | -2.38 | 0 | -2.38 |
| contest_slowpush | -0.32 | 8.08 | 24.76 | 2.38 | 2.38 | 0 | 2.38 | 0 |

## Exploitability and where the exploiter's gain comes from

Decomposition = the baseline's (A minus B) farm, crash resources and gank payoff along the path against its best response; they sum to the value against the best response (= -exploitability).

| baseline (as A) | exploitability | per turn | farm | crash | gank | vs equilibrium B | B's best response plays |
|---|---|---|---|---|---|---|---|
| uniform | 10.08 | 0.315 | -6.08 | 0.06 | -4.06 | -9.86 | F-C 0.05, F-S 0.08, SH-C 0.09, SH-S 0.03, SP-C 0.60, SP-S 0.16 |
| aggressive | 24.86 | 0.777 | -5.24 | 0 | -19.62 | -24.47 | SP-C 0.94, SP-S 0.06 |
| freeze_contest | 9.45 | 0.295 | -4.55 | -5 | 0.1 | -2.87 | SH-C 0.06, SH-S 0.06, SP-C 0.38, SP-S 0.50 |
| passive_safe | 9.42 | 0.294 | -5.37 | -5 | 0.95 | -2.87 | SH-C 0.06, SH-S 0.06, SP-C 0.41, SP-S 0.47 |
| stack_builder | 5.78 | 0.181 | 2.47 | -6 | -2.25 | -0.32 | SH-C 0.12, SH-S 0.06, SP-C 0.31, SP-S 0.50 |
| release_safe | 7.69 | 0.24 | -11.69 | 0 | 4 | -2.8 | SH-C 0.09, SP-C 0.78, SP-S 0.12 |
| contest_slowpush | 6.59 | 0.206 | 2.36 | -7 | -1.95 | -0.32 | SH-C 0.19, SP-C 0.28, SP-S 0.53 |

## Equilibrium vs. best response: how much of the available exploitation does the equilibrium capture?

'captured' = (value the equilibrium earns against the baseline) / exploitability. The equilibrium guarantees V* against everything but does not seek out mistakes; the gap is what an opponent model could add (the Phase 2 question).

| baseline | exploitability | equilibrium earns | captured |
|---|---|---|---|
| uniform | 10.08 | 9.86 | 98% |
| aggressive | 24.86 | 24.47 | 98% |
| freeze_contest | 9.45 | 2.87 | 30% |
| passive_safe | 9.42 | 2.87 | 30% |
| stack_builder | 5.78 | 0.32 | 6% |
| release_safe | 7.69 | 2.8 | 36% |
| contest_slowpush | 6.59 | 0.32 | 5% |

## Crash, farm and positioning behaviour along the best-response path (per game)

| baseline | baseline crashes S/M/L | best response crashes S/M/L | baseline farm (avail) | best response farm (avail) | baseline P(Contest) | best response P(Contest) |
|---|---|---|---|---|---|---|
| uniform | 0.1/0.0/0.0 | 0.0/0.0/0.0 | 5.3 (15.8) | 11.4 (15.4) | 0.5 | 0.73 |
| aggressive | 0.0/0.0/0.0 | 0.0/0.0/0.0 | 6.0 (16.0) | 11.3 (15.0) | 1 | 0.94 |
| freeze_contest | 0.0/0.0/0.0 | 0.0/1.0/1.0 | 5.8 (12.2) | 10.4 (16.0) | 0.16 | 0.44 |
| passive_safe | 0.0/0.0/0.0 | 0.0/1.0/1.0 | 5.4 (12.2) | 10.8 (16.0) | 0 | 0.47 |
| stack_builder | 0.0/0.0/0.0 | 1.0/1.0/1.0 | 9.4 (12.8) | 7.0 (16.0) | 1 | 0.44 |
| release_safe | 0.0/0.0/0.0 | 0.0/0.0/0.0 | 2.1 (15.8) | 13.7 (15.5) | 0 | 0.88 |
| contest_slowpush | 0.0/0.0/0.0 | 0.0/2.0/1.0 | 9.4 (12.5) | 7.0 (16.0) | 1 | 0.47 |

## Realized payoff distributions (5000 sampled games each, seed 7; farm is an expected reward, ganks are sampled)

| matchup | mean | stderr | std | q05 | q50 | q95 | ganks A/B |
|---|---|---|---|---|---|---|---|
| equilibrium vs equilibrium | 0.178 | 0.233 | 16.47 | -25 | 0 | 30 | [6.9, 6.93] |
| uniform vs best response | -10.02 | 0.228 | 16.11 | -36.15 | -10.55 | 16.17 | [6.73, 5.93] |
| aggressive vs best response | -24.8 | 0.237 | 16.77 | -50.24 | -25.24 | 4.76 | [9.38, 5.47] |
| freeze_contest vs best response | -9.757 | 0.22 | 15.52 | -34.55 | -9.54 | 15.45 | [5.8, 5.76] |
| passive_safe vs best response | -9.705 | 0.22 | 15.54 | -35.37 | -10.36 | 14.64 | [5.7, 5.84] |
| stack_builder vs best response | -5.899 | 0.224 | 15.85 | -33.53 | -3.53 | 21.47 | [6.45, 5.97] |
| release_safe vs best response | -7.461 | 0.218 | 15.41 | -31.69 | -6.69 | 18.31 | [5.51, 6.36] |
| contest_slowpush vs best response | -6.763 | 0.226 | 16 | -34.64 | -4.64 | 20.36 | [6.45, 6.03] |

Equilibrium path per player: farm 9.65 of 15.50 available, gank loss 34.70, no crashes; P(Contest) 0.91.
