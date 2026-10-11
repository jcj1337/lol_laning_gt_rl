# Laning as a Stochastic Game: Exploitation vs. Exploitability

A quantitative research project on decision-making under strategic uncertainty. The central question:

> **When should an agent exploit an opponent's predictable behavior, and how much risk should it accept when its opponent model might be wrong?**

League of Legends wave management is the testbed. Two players manage minion waves, their own positioning and, when they have crashed a wave into the enemy tower, how to use that siege. The project models this as a finite-horizon, two-player, zero-sum stochastic game. That gives an exact, unexploitable baseline (the equilibrium) against which opponent-modelling and exploitation strategies can be measured. The mechanics are deliberately stylized; no parameter is claimed to be an empirical League statistic.

## Roadmap

| Phase | Content | Status |
|---|---|---|
| **1. Mathematical foundation** | Stochastic game, mechanics, exact equilibrium solver, best responses, exploitability | **complete: siege revision** ([siege & vision study](docs/phase1_siege_vision_study.md); earlier: [farming revision report](docs/phase1_revision_report.md), [original Phase 1 report](docs/phase1_implementation_report.md)) |
| 2. Quantitative research | Bayesian opponent modelling, exploitation vs. exploitability, risk-constrained policies, misspecification, sensitivity | next |
| 3. Interactive dashboard | React/TypeScript site for exploring strategies and results | planned |

The transition from the exact game to multi-agent reinforcement learning is planned in [docs/rl_research_roadmap.md](docs/rl_research_roadmap.md). Implemented: Milestone 1, a simultaneous-move RL environment around the existing game (`lanegame/env/`); Milestone 2, the champion game (`GameConfig.champions()`: persistent positions, Hold / Yield / Trade / Safe, Recall and absence, stochastic trades with mechanical skill, tower shots, ganks with HP damage, death and respawn), validated in `results/champions/`; and Milestones 3a-3c, masked PPO, frozen-snapshot self-play and an evaluation harness with approximate best responses (`lanegame/rl/`, `results/rl/`). See [docs/m2_m3_implementation_report.md](docs/m2_m3_implementation_report.md).

## The current model at a glance (siege revision)

- **Actions.** Each turn both players simultaneously choose a *wave-management* action (Slow Push, Shove, Freeze) and a *positioning* action (Contest, Safe). A player whose crashed minions are under the enemy tower at the start of the turn also chooses a *siege objective* with Contest: **Apply Pressure** (claim a tower plate and thin the defender's wave gathering behind the tower) or **Establish Vision** (a persistent gank-risk reduction from the next turn on); Safe during a siege means disengaging. Up to 6 actions per player per turn; no information about the opponent's current choice.
- **Lane.** Minion waves are explicit packets with 4 HP per wave. They spawn every 4 ticks, walk, merge (up to a 3-wave cap; extra waves queue behind and fight with the stack), and fight with discrete Lanchester attrition. Slow Push holds an unstacked wave at the lane until reinforcements arrive and then creeps forward. Freeze holds the enemy wave just outside the freezer's tower, is legal only there, and breaks under enough pressure. A crashed wave stays under the tower as HP mass that tower fire removes at 1 HP/tick (2 / 4 / 6 turns for a 1 / 2 / 3-wave crash); the defender's reinforcements gather behind the tower and bounce back out when the mass is gone, unless the attacker thins them.
- **Economy (event-based).** Every HP of minions that dies is a farm opportunity for the side that killed it, including minions killed by a tower or thinned by a pressuring champion; the collector receives it times a collection efficiency that depends on where the minions died, its posture and whether the opponent contests. Each crash opens 1 / 2 / 3 plate opportunities (small / medium / large) worth one resource each, claimable one per Apply Pressure turn while that siege lasts; crashes pay nothing by themselves. Each turn each player risks a gank whose probability depends on champion exposure, action, wave strength and vision (75% reduction for the 6 turns after vision is established). The payoff is relative resources, zero-sum.
- **Solver.** Backward induction with exact stage games (saddle detection, otherwise a HiGHS LP), certified duality gaps, exact best responses and exploitability, exact forward path statistics. The two earlier models are available bit-for-bit as `GameConfig.v2()` (farming revision) and `GameConfig.v1()` (original).

## Main findings of the siege study (K = 32 unless stated; details and caveats in the report)

- **A big crash is no longer a liability.** In the farming revision the state "my 3-wave crash sits under the enemy tower" was worth −5.8 to the crasher; with siege objectives it is worth +6.1, and the first thing to do in it is Establish Vision, the second Apply Pressure.
- **Vision first, pressure second.** Whenever a player sieges without active vision, the equilibrium establishes vision (probability 0.94–1.00 across 50–90% effectiveness and 4–8 turns); with vision active it applies pressure (0.7–0.8). The Vision → Shove → Crash → Pressure sequence is reachable from the standard start with the protected pressure 3–4 turns after the vision.
- **Thinning weakens bounces without erasing them.** Continuous pressure on a 3-wave crash thins 6 HP: the defender releases a 10-HP wave instead of a capped 12-HP wave plus a queued reinforcement. In solved play the large-bounce problem disappears for a different reason: the best responder prefers small and medium crashes it can siege.
- **Tower fire stays at 1 HP/tick.** At 2 or 3 HP/tick sieges last one turn, every siege decision collapses to vision, and Apply Pressure is never used by a best responder; the state space is 6–10× smaller, which makes 2 HP/tick the cheap variant for sweeps.
- **The symmetric equilibrium is unchanged:** an even, contested, crash-free lane fight (Slow Push + Contest 84%) in every configuration tested. The new mechanics act off the path and in exploitation.
- **Predictable play is much more exploitable, through a new channel.** Exploitability of passive_safe rises from 9.4 to 21.5; the best responder crashes small waves, sieges them for vision and plates, and is protected for 24 of 32 turns, so its gain is now mostly its own reduced gank exposure. Always-Shove (24.9) and Shove + Safe (7.7) are unchanged.
- **Costs.** 172,907 decision nodes, 2.7M transitions, 3 GB and about 4 minutes per K = 32 solve, against 14,849 nodes and 10 seconds before; the vision timer is the main new state variable, and a per-tower plate budget (rejected) would have multiplied the space again.

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate      # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest                                      # full suite (458 tests incl. champion mode and RL), ~6 min
python -m lanegame solve --horizon 32                 # siege model -> results/siege/solution_K32.json (~4 min, 3 GB)
python -m lanegame scenarios                          # 23 traced mechanics scenarios with PASS/FAIL
python -m lanegame reach --horizon 24                 # lane and siege events reachable from the standard start, with witnesses
python -m lanegame strategy --min-crash-hp 4          # Vision -> Shove -> Crash -> Pressure witness with a turn-by-turn trace
python -m lanegame exploit --horizon 24               # exploitability of the siege baselines
python -m lanegame env-demo --a aggressive --b passive_safe --episodes 200 --seed 42   # two scripted agents through the RL environment
python -m lanegame env-demo --model champions --a opportunistic_trader --b hold_farmer   # the champion game (M2)
python -m lanegame experiments-champions all          # champion-mode validation (results/champions/, ~6 min)
pip install torch                                     # PPO needs PyTorch (optional extra "rl")
python -m lanegame rl-train --opponents hold_farmer,aggressive_trader --steps 100000 --out results/rl/checkpoints/demo
python -m lanegame rl-eval results/rl/checkpoints/demo/final.pt   # both seats, held-out seeds, 95% CIs
python -m lanegame experiments-rl scripted legacy selfplay evaluation   # M3a-M3c experiments (results/rl/)
python -m lanegame rl-shared --current-fraction 0.8 --skill-levels 0.2,0.5,0.8 --out results/rl/checkpoints/demo_shared
                                                      # shared-policy self-play, 80% current / 20% past versions
python -m lanegame experiments-rl openai_five          # the OpenAI Five-inspired study (results/rl/openai_five/)
python -m lanegame experiments-siege all              # regenerate every siege-study result (results/siege/)
python -m lanegame experiments all                    # regenerate the farming-revision results (results/revision/)
python -m lanegame experiments-v1 all                 # regenerate the original Phase 1 results (results/)
```

Every command accepts `--model siege|revised|v1` (default `siege`).

## Repository layout

```
lanegame/                 engine (pure Python + NumPy/SciPy, no presentation code)
  config.py               all model parameters (GameConfig; .v2() = farming revision, .v1() = original model)
  state.py                Markov state (lane, siege mass, vision timers, plate opportunities) and canonical encoding
  mechanics.py            packet movement, merging, fights, thinning, crashes, tower fire, spawns, plates, vision timers
  game.py                 legal actions incl. siege objectives, farming, plates, gank model, reachable states
  matrix_game.py          zero-sum matrix games: saddle detection, HiGHS LP, certificates, support enumeration
  solver.py               backward induction -> values, mixed strategies, Q matrices
  best_response.py        exact policy evaluation, best responses, exploitability
  policy.py, baselines.py policy interface; baselines with siege tendencies (pressure / vision / safe); earlier baselines
  analysis.py             exact forward occupancy and path statistics (farm / plates / gank decomposition, siege use, bounces)
  reachability.py         reachable lane and siege events with witnesses; the Vision -> Shove -> Crash -> Pressure search
  simulate.py             seeded Monte Carlo with realized gank outcomes
  scenarios.py, render.py named behavioral scenarios (23) and text traces of the lane
  io.py                   JSON export format for solutions (version 3)
  champions.py            champion mode (roadmap M2): positions, Hold/Yield/Trade/Safe/Recall, absence, trades, skill,
                          tower shots, ganks with HP damage, death and respawn
  env/                    simultaneous-move RL environment (roadmap M1): LaneEnv, own-view observations and
                          masks, scripted agents, exact-policy bridge, episode traces, optional PettingZoo adapter
  rl/                     masked PPO, frozen-snapshot self-play, shared-policy self-play with a current / historical
                          mixture (shared.py), evaluation and approximate best responses (M3; torch)
  experiments/champions/  champion-mode validation (results/champions/)
  experiments/rl/         RL experiments (results/rl/; checkpoints in results/rl/checkpoints/, not tracked)
  experiments/siege/      the siege study's experiments (results/siege/)
  experiments/            the farming revision's experiments, pinned to GameConfig.v2() (results/revision/)
  experiments/v1/         the original Phase 1 experiments, pinned to GameConfig.v1() (results/)
tests/                    pytest suite (all three models)
results/                  original Phase 1 outputs; results/revision/ = farming revision; results/siege/ = this study
docs/                     phase1_implementation_report.md, phase1_revision_report.md, phase1_siege_vision_study.md,
                          rl_research_roadmap.md, m2_m3_implementation_report.md, overnight_implementation_log.md
legacy/                   original Q-learning project (env, agents, notebook), kept unchanged for reference
```

## The earlier models

- **Farming revision** (`GameConfig.v2()`): event-based farming, Contest/Safe positioning, mass-based tower fire, a post-crash choice between crash resources and two turns of 50% vision. Documented in [docs/phase1_revision_report.md](docs/phase1_revision_report.md), results in `results/revision/`.
- **Original Phase 1** (`GameConfig.v1()`): wave actions only, no farming, one-packet-per-turn tower consumption. Documented in [docs/phase1_implementation_report.md](docs/phase1_implementation_report.md), results in `results/`.
- **The first Q-learning prototype** is in [legacy/](legacy/), audited in section B of the original report.
