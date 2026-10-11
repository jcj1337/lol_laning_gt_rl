# Laning as a Stochastic Game: Exploitation vs. Exploitability

A quantitative research project on decision-making under strategic uncertainty. The central question:

> **When should an agent exploit an opponent's predictable behavior, and how much risk should it accept when its opponent model might be wrong?**

League of Legends wave management is the testbed. Two players manage minion waves and their own positioning. The project models this as a finite-horizon, two-player, zero-sum stochastic game. That gives an exact, unexploitable baseline (the equilibrium) against which opponent-modelling and exploitation strategies can be measured. The mechanics are deliberately stylized; no parameter is claimed to be an empirical League statistic.

## Roadmap

| Phase | Content | Status |
|---|---|---|
| **1. Mathematical foundation** | Stochastic game, mechanics, exact equilibrium solver, best responses, exploitability | **revised** ([revision report](docs/phase1_revision_report.md); original: [Phase 1 report](docs/phase1_implementation_report.md)) |
| 2. Quantitative research | Bayesian opponent modelling, exploitation vs. exploitability, risk-constrained policies, misspecification, sensitivity | next |
| 3. Interactive dashboard | React/TypeScript site for exploring strategies and results | planned |

## The revised model at a glance

- **Actions.** Each turn both players simultaneously choose a *wave-management* action (Slow Push, Shove, Freeze) and a *positioning* action (Contest, Safe): up to 6 actions each.
- **Lane.** Minion waves are explicit packets with 4 HP per wave. They spawn every 4 ticks, walk, merge (up to a 3-wave cap; extra waves queue behind and fight with the stack), and fight with discrete Lanchester attrition. Slow Push holds an unstacked wave at the river until reinforcements arrive and then creeps forward. Freeze holds the enemy wave just outside the freezer's tower, is legal only there, and breaks under enough pressure. A crashed wave stays under the tower as HP mass that tower fire removes at 1 HP/tick, so bigger crashes occupy the tower longer. Meanwhile the defender's reinforcements gather behind the tower and bounce back out.
- **Economy (event-based).** Every HP of minions that dies is a farm opportunity for the side that killed it, including minions killed by a tower. The collector receives it times a *collection efficiency*, which depends on where the minions died, the collector's posture and whether the opponent contests (zoning). A crash also gives the crasher a choice between resources and two turns of vision. Each turn, each player risks a gank whose probability depends on champion exposure (Contest follows the wave; Safe stays at its tower), action, wave strength and vision. The payoff is relative resources, zero-sum.
- **Solver.** Backward induction with exact stage games (saddle detection, otherwise a HiGHS LP), certified duality gaps, separate crash-choice games, exact best responses and exploitability. The original Phase 1 model is still available, bit-for-bit, as `GameConfig.v1()`.

## Main findings (revised default, 32 turns; details and caveats in the report)

- **Farming changes what exploitation means.** In the original model, almost all of every baseline's exploitability came from gank exposure. Now best responses against passive, freezing and stack-building baselines gain mostly through farm and crash differences. Only the always-Shove baseline is still exploited mainly through ganks.
- **The symmetric equilibrium is an even, contested lane.** The waves fight continuously at the river, both players Contest 91% of turns, and nobody crashes. Off the path, it contests most when its wave fights just in front of its own tower (0.91) and retreats to Safe after crashing or when its wave sits under the enemy tower (0.19–0.25).
- **Contest has thresholds.** With farming off, the equilibrium is entirely passive (Slow Push + Safe). Contest dominates from about one resource per wave. A gank loss of 7 (instead of 5) flips the equilibrium path back to Safe.
- **Crashing is not free.** The defender farms crashed minions under its tower, so a 3-wave crash has a negative continuation value for the crasher. It gets more negative as farm becomes more valuable.
- **Equilibrium play leaves money on the table.** The equilibrium extracts 98% of what a best responder could take from Shove-happy or random opponents, but only 5–36% against passive, freezing or stack-building ones. That gap is the Phase 2 research question.
- **Outcomes are noisy.** A single game has a standard deviation of about 16 resources, against exploitation edges of 6–25 per game.

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate      # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest                                      # full suite, both models, ~1 min
python -m lanegame solve --horizon 32                 # writes results/revision/solution_K32.json
python -m lanegame scenarios                          # traced mechanics scenarios with PASS/FAIL
python -m lanegame reach                              # event reachability from the standard start, with witnesses
python -m lanegame exploit --horizon 32               # exploitability of baseline policies
python -m lanegame experiments all                    # regenerate every revision result (results/revision/)
python -m lanegame experiments-v1 all                 # regenerate the original Phase 1 results (results/)
```

Every command accepts `--model v1` to run the original model instead.

## Repository layout

```
lanegame/                 engine (pure Python + NumPy/SciPy, no presentation code)
  config.py               all model parameters (GameConfig; GameConfig.v1() = original model)
  state.py                Markov state and canonical encoding (own-perspective, mirror = swap)
  mechanics.py            packet movement, merging, fights, crashes, tower fire, spawns, kill accounting
  game.py                 legal actions, farming, gank model, crash-reward choices, reachable states
  matrix_game.py          zero-sum matrix games: saddle detection, HiGHS LP, certificates, support enumeration
  solver.py               backward induction -> values, mixed strategies, Q matrices, choice policies
  best_response.py        exact policy evaluation, best responses, exploitability
  policy.py, baselines.py policy interface; revised and original baseline policies
  analysis.py             exact forward occupancy and path statistics (farm / crash / gank decomposition)
  reachability.py         which lane events are reachable from the standard start, with witness sequences
  simulate.py             seeded Monte Carlo with realized gank outcomes
  scenarios.py, render.py named behavioral scenarios and text traces of the lane
  io.py                   JSON export format for solutions (for Phase 2/3)
  experiments/            revision experiments A-F and benchmarks (results/revision/)
  experiments/v1/         the original Phase 1 experiments, pinned to GameConfig.v1() (results/)
tests/                    pytest suite (both models)
results/                  original Phase 1 outputs (historical); results/revision/ = revised model
docs/                     phase1_implementation_report.md (original), phase1_revision_report.md (revision)
legacy/                   original Q-learning project (env, agents, notebook), kept unchanged for reference
```

## The original projects

- **Phase 1, original version.** Wave actions only, with no farming and one-packet-per-turn tower consumption. It is documented in [docs/phase1_implementation_report.md](docs/phase1_implementation_report.md); its results are in `results/` and it is reproducible with `GameConfig.v1()`.
- **The first Q-learning prototype** is in [legacy/](legacy/), audited in section B of that report.
