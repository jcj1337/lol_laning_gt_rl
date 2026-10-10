# Laning as a Stochastic Game: Exploitation vs. Exploitability

A quantitative research project on decision-making under strategic uncertainty. The central question:

> **When should an agent exploit an opponent's predictable behavior, and how much risk should it accept when its opponent model might be wrong?**

League of Legends wave management is the testbed. Two players manage minion waves by choosing to Slow Push, Shove or Freeze. The project models this as a finite-horizon, two-player, zero-sum stochastic game. That gives an exact, unexploitable baseline (the equilibrium) against which opponent-modelling and exploitation strategies can be measured. The mechanics are deliberately stylized; no parameter is claimed to be an empirical League statistic.

## Roadmap

| Phase | Content | Status |
|---|---|---|
| **1. Mathematical foundation** | Stochastic game, mechanics, exact equilibrium solver, best responses, exploitability | **complete** ([report](docs/phase1_implementation_report.md)) |
| 2. Quantitative research | Bayesian opponent modelling, exploitation vs. exploitability, risk-constrained policies, misspecification, sensitivity | next |
| 3. Interactive dashboard | React/TypeScript site for exploring strategies and results | planned |

## Phase 1 at a glance

- **Model.** Minion waves are explicit packets that spawn every 4 ticks, walk, merge, fight (discrete Lanchester attrition), and crash into towers. Towers consume one packet per decision turn, which produces wave bounces. Legal actions depend on the state (Freeze only with enemy minions on your half). A crash earns a choice between resources and vision. Per-turn gank risk depends on exposure, action, wave size and vision. The payoff is A's resources minus B's.
- **Solver.** Backward induction with exact stage games: saddle-point detection, otherwise a HiGHS LP, with certified duality gaps. Crash choices are solved as separate choice nodes. Best responses and exploitability are exact for any time-dependent Markov policy.
- **Verification.** 68 tests cover:
  - mechanics invariants: packet conservation, no crossing, zero-sum, mirror symmetry;
  - known matrix games;
  - an LP-free reference solver and a naive tree-recursion minimax;
  - zero exploitability (subgame perfection) of the equilibrium;
  - Monte Carlo vs. exact values.
- **Performance.** The game is solved exactly for 16, 32 and 64 decision turns in 0.1–0.6 s, and for 256 turns in 2.5 s.
- **Findings** (default parameters):
  - **Passive symmetric equilibrium.** Symmetric equilibrium play is passive and never crashes. This is robust across 7 alternative collision models and 23 parameter perturbations.
  - **Rich off-path play.** Off-path strategy depends on wave position, size and reinforcement timing (52 strictly mixed nodes).
  - **Exploitation is mostly exposure.** A best responder extracts 1.9–19.5 resources per 32-turn game from simple scripted policies, mostly by letting them overextend.
  - **Noisy outcomes.** Single-game outcomes have a standard deviation of about 16 resources, so exploitation edges are hard to detect.

  See the [Phase 1 report](docs/phase1_implementation_report.md) for the full specification, design history, results and limitations.

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate      # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest                                      # 68 tests, ~7 s
python -m lanegame solve --horizon 32                 # writes results/solution_K32.json
python -m lanegame scenarios                          # traced mechanics scenarios
python -m lanegame exploit --horizon 32               # exploitability of baseline policies
python -m lanegame experiments all                    # regenerate every result (~90 s)
```

## Repository layout

```
lanegame/                 Phase 1 engine (pure Python + NumPy/SciPy, no presentation code)
  config.py               all model parameters (GameConfig, JSON-serializable)
  state.py                Markov state and canonical encoding (own-perspective, mirror = swap)
  mechanics.py            packet movement, merging, fights, crashes, tower consumption, spawns
  game.py                 legal actions, gank model, crash-reward choices, reachable states
  matrix_game.py          zero-sum matrix games: saddle detection, HiGHS LP, certificates, support enumeration
  solver.py               backward induction -> values, mixed strategies, Q matrices, choice policies
  best_response.py        exact policy evaluation, best responses, exploitability
  policy.py, baselines.py policy interface; uniform / always-shove / always-slow-push / defensive-freeze / stack-then-shove
  analysis.py             exact forward occupancy and path statistics
  simulate.py             seeded Monte Carlo with realized gank outcomes
  scenarios.py, render.py named behavioral scenarios and text traces of the lane
  io.py                   JSON export format for solutions (for Phase 2/3)
  experiments/            reproducible experiments (python -m lanegame experiments <name>|all)
tests/                    pytest suite
results/                  generated tables (.md), data (.json) and figures (.png)
docs/                     phase1_implementation_report.md
legacy/                   original Q-learning project (env, agents, notebook), kept unchanged for reference
```

## The original project

The first version of this repository (now in [legacy/](legacy/)) trained a tabular Q-learning agent on a simplified lane model. Its audit is in section B of the Phase 1 report. In brief:
- it used float-valued state keys and had no time index;
- rewards were not zero-sum and some could be collected repeatedly;
- it used a fixed rock-paper-scissors payoff;
- it was evaluated only against scripted opponents.

The legacy notebook still runs from inside `legacy/`.
