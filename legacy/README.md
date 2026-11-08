# Legacy

The tabular Q-learning prototype the project started from (December 2025 to January 2026): `LoL_Laning.ipynb` and
the modules in `src/`. It is kept unchanged and does not use `lanegame`.

Phase 1 audited it and replaced it with an exactly solvable game. The audit found Q-values keyed on floats, no time
index or terminal flag, a fixed rock-paper-scissors payoff, and rewards that were not zero-sum on 53.8% of steps
([v1 report](../docs/archive/phase1/phase1_implementation_report.md) §B). The original model `GameConfig.v1()` keeps
the prototype's three wave actions and its expected gank loss, with packet dynamics in place of the payoff matrix.
