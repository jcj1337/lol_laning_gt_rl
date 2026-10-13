# Laning as a Two-Player Game: Exact Equilibria and Self-Play Reinforcement Learning

A quantitative research project that models League of Legends laning as a simplified, sequential, two-player
zero-sum game and studies the strategies that arise in it, first with exact game theory and now with multi-agent
reinforcement learning.

## The research question

Two champions share one lane. Each turn both choose, simultaneously, how to handle the minion wave (last-hit, thin
it, clear it), where to stand (hold, yield, trade, play safe), and whether to go home. The simulator resolves
minion fights, tower fire, crashes and bounces, trades whose damage depends on level and items, ganks, recalls, XP,
gold, item purchases and tower plates. None of these is strategically interesting by itself. What makes laning a
game is how they interact: clearing a wave early makes it crash under the enemy tower, which opens a recall window
and denies the opponent farm, but exposes the pusher to a gank; a level lead turns a trade from a bluff into a
threat; a frozen wave starves the opponent only while it is held; plates reward sustained pressure. Those
interactions are what good League players exploit, and every one of them emerges from the rules rather than being
scripted.

**Can an RL agent learn to take advantage of those interactions without us explicitly teaching it League of Legends strategies?**

The agents see only the game's observations and legal actions, and their reward is the economic outcome: the
difference in gold earned, nothing else. No reward is given for freezing, slow pushing, recalling at the right time
or any other named strategy. Whether such behaviour appears is the thing being measured, through
quantitative evaluation of the learned policies: their discounted value against fixed references, their
exploitability (approximate best responses), and audits of what they actually do, compared with the sampled
equilibria of a scripted strategy space.

**Status, stated plainly.** The environment has been built, calibrated and validated (through M4.2, below), the RL
infrastructure has been audited (M4.3a), and the first self-play pilot has been run and analysed (M4.3b). PPO
self-play learns policies that beat every scripted reference of the project's strongest strategy space by 9–20
resources and keep improving against their own earlier versions; their decisions are state-dependent on the
behaviour dimension (trade at the lane, step back at the tower, recall at low HP and after the own crash, pressure
in sieges), and their dominant wave action, clearing, is shown by counterfactual rollouts to be a best reply to an
opponent that clears: slow pushes and held freezes lose one to three resources against the learned policies. Recall
is learned and timed where the game rewards it. What the pilot does not show is the emergence of League's
wave-control strategies; the evidence says the calibrated game does not reward them against a clearing opponent,
and the next milestone (M4.3c) is a design investigation judged by the same counterfactual harness.

## How we study it

1. **An exact game as the foundation (Phase 1).** Wave management alone is a finite-horizon zero-sum stochastic
   game that we solve exactly: equilibrium values and mixed strategies, exact best responses and exploitability.
   This gives an unexploitable baseline and calibrates the RL environment, since a network policy on that game has
   an exactly computable value.
2. **A richer game that is not exactly solvable (M2 and M4).** Champions, HP, trading, recalls, then the integrated
   M4 economy: continuing time with a discounted objective (gamma = 0.99, no last turn to exploit), XP and levels,
   gold and items, durability plates, natural wave dynamics in which pushes, freezes, crashes and bounces follow
   from one damage law.
3. **Multi-agent reinforcement learning.** A simultaneous-move environment with own-perspective observations and
   legal-action masks, exposed through the PettingZoo Parallel API; a PyTorch policy/value network; a masked PPO
   implementation (CleanRL-style, audited against independent references); shared-policy self-play in which one
   network plays both seats and trains on both seats' experience, against the current policy or a mixture with
   frozen past versions; continuing-time rollouts with bootstrapped truncations; discounted evaluation with a
   guaranteed truncation bound.
4. **Quantitative evaluation.** Head-to-head and discounted values with confidence intervals and paired seeds,
   approximate best responses as exploitability lower bounds, action audits and counterfactual rollouts, and the
   exact solver where it applies.

Every mechanic is judged against one standard, [docs/design_philosophy.md](docs/design_philosophy.md): simplify how
League of Legends works without changing why its laning strategies work. The mechanics are deliberately stylized; no
parameter is claimed to be an empirical League statistic.

## Roadmap

| Phase | Content | Status |
|---|---|---|
| **1. Mathematical foundation** | Stochastic game, mechanics, exact equilibrium solver, best responses, exploitability | **complete: siege revision** ([siege & vision study](docs/phase1/phase1_siege_vision_study.md); earlier: [farming revision report](docs/phase1/phase1_revision_report.md), [original Phase 1 report](docs/phase1/phase1_implementation_report.md)) |
| **M1–M3. RL environment, champion game, PPO and self-play** | `lanegame/env/`, `GameConfig.champions()`, masked PPO, snapshot and shared-policy self-play, evaluation | **complete** ([M2/M3 report](docs/rl/m2_m3_implementation_report.md), [OpenAI Five-style self-play](docs/rl/openai_five_selfplay_report.md), [diagnostics](docs/rl/openai_five_diagnostics_report.md)) |
| **M4.1–M4.2. Integrated champion economy and natural waves** | `GameConfig.m4()` and the presets up to `m4_lasthit()` | **complete** ([design](docs/m4/m4_integrated_design.md), [implementation](docs/m4/m4_implementation_report.md), [natural waves](docs/natural_waves/m4_2e_natural_waves_final_review.md), [wave actions](docs/natural_waves/m4_2e_wave_actions_report.md)) |
| **M4.3a. RL infrastructure audit and smoke test** | PettingZoo conformance, PPO reference checks, bounded self-play on `m4_lasthit()` | **complete** ([report](docs/rl/m4_3a_rl_infrastructure_report.md)) |
| **M4.3b. Self-play pilot and strategic diagnostics** | 12 PPO runs on `m4_lasthit()`, reactive reference suite, learning curves, cross-play, exploitability, conditional behaviour, counterfactuals, Recall / death economics, mixed starts | **complete** ([report](docs/rl/m4_3b_ppo_pilot_report.md)) |
| M4.3c. Design investigation of Clear dominance, population attackers | Which mechanics would make wave control pay against a clearing opponent? | next |
| 2. Opponent modelling | Bayesian opponent models, exploitation vs. exploitability, risk | planned |
| 3. Interactive dashboard | A site for exploring strategies and results | planned |

The research roadmap with one section per milestone is [docs/rl_research_roadmap.md](docs/rl_research_roadmap.md);
the documentation index is [docs/README.md](docs/README.md).

## The games

**The exact game (Phase 1, `GameConfig()`).** Each turn both players simultaneously choose a wave-management action
(Slow Push, Shove, Freeze) and a positioning action (Contest, Safe); a player whose crashed minions are under the
enemy tower also chooses a siege objective (Apply Pressure or Establish Vision). Minion waves are explicit packets
that spawn, walk, merge and fight with discrete Lanchester attrition; a crashed wave sits under the tower as HP mass
that tower fire removes. The economy is event-based: every HP of minions that dies is a farm opportunity for the
side that killed it, plates are claimed one per pressure turn, ganks are a per-turn risk reduced by vision, and the
payoff is relative resources, zero-sum. The solver is backward induction with exact stage games (saddle detection,
otherwise a HiGHS LP), certified duality gaps, exact best responses and exploitability. Main findings of the siege
study (K = 32): a big crash becomes an asset once it can be sieged (+6.1 instead of −5.8 to the crasher); the
equilibrium establishes vision first and applies pressure second; the symmetric equilibrium is an even, contested,
crash-free lane fight; predictable play is much more exploitable (passive_safe 21.5 instead of 9.4) through the
siege channel. Details and caveats in the [study](docs/phase1/phase1_siege_vision_study.md). The two earlier models
are available bit-for-bit as `GameConfig.v2()` (farming revision) and `GameConfig.v1()` (original).

**The M4 game (`GameConfig.m4()` and its presets).** A continuing game with no terminal state, discounted with
gamma = 0.99 and trained in bootstrapped segments; champions with persistent positions, centi-HP and attack power
scaled by level and items; XP within 3 cells and levels 1–18; an exact micro-resource gold ledger; purchases on
return to base; persistent tower damage with three plates; ganks with HP damage, death and respawn; a lossless,
decoder-proven 88-feature observation; 46 actions. The M4.2 milestones added opt-in presets on top of `m4()`:
`m4_emergent()` (positioning, zoning and investment readiness), `m4_natural()` (natural wave mechanics: Hold, Thin
and Clear are champion damage on the wave, and pushes, freezes, crashes and bounces follow from the damage law),
`m4_lane()` (danger as consequences instead of fixed collection rates; investment buys durability) and
`m4_lasthit()`, the M4.3 target: Hold, Thin and Clear as shares (1/8, 1/2, 1) of one wave-clear capacity, so that
normal last-hitting pushes a little and thinning grows with power, on a half-wave Lanchester law. Every legacy
configuration is bit-identical, guarded by `tests/test_legacy_pins.py`.

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate      # macOS/Linux: source .venv/bin/activate
pip install -e ".[test,rl]"                          # numpy, scipy, matplotlib, pytest; torch, pettingzoo, gymnasium
python -m pytest                                      # full suite (exact game, champion game, M4, RL): 909 tests, ~8 min
python -m lanegame solve --horizon 32                 # exact siege model -> results/siege/solution_K32.json (~4 min, 3 GB)
python -m lanegame scenarios                          # 23 traced mechanics scenarios with PASS/FAIL
python -m lanegame exploit --horizon 24               # exploitability of the siege baselines
python -m lanegame env-demo --a aggressive --b passive_safe --episodes 200 --seed 42   # scripted agents through the RL environment
python -m lanegame env-demo --model champions --a opportunistic_trader --b hold_farmer   # the champion game (M2)
python -m lanegame rl-shared --model m4_lasthit --current-fraction 1.0 --steps 409600 --out results/rl/checkpoints/demo_m4
                                                      # shared-policy self-play PPO on the M4.3 target (continuing time)
python -m lanegame experiments-rl m43a_smoke          # the M4.3a smoke test (results/rl/m43a_smoke.md, ~5 min)
python -m lanegame.experiments.rl.m43b train curves headline exploit behavior counterfactual plans recall trajectories figures report
                                                      # the M4.3b pilot stage by stage (results/rl/m43b/, ~2 h on 9 processes)
python -m lanegame rl-shared --current-fraction 0.8 --skill-levels 0.2,0.5,0.8 --out results/rl/checkpoints/demo_shared
                                                      # the champion game, 80% current / 20% past versions
python -m lanegame rl-eval results/rl/checkpoints/demo_shared/final.pt   # champion game: both seats, held-out seeds, 95% CIs
python -m lanegame experiments-rl openai_five         # the OpenAI Five-inspired study (results/rl/openai_five/)
python -m lanegame experiments-siege all              # regenerate every siege-study result (results/siege/)
streamlit run lanegame/viewer/app.py                  # the visual lane debugger (pip install -e ".[viewer]"); mode
                                                      # "Recorded trajectory" replays results/rl/m43b/trajectories/*.json
```

The exact-game commands accept `--model siege|revised|v1` (default `siege`).

Using the environment from Python, natively or through PettingZoo:

```python
from lanegame import GameConfig
from lanegame.env import LaneEnv
from lanegame.env.pettingzoo_adapter import LaneParallelEnv

env = LaneEnv(GameConfig.m4_lasthit(), max_turns=512)
obs_a, obs_b = env.reset(seed=0)                 # own-perspective float32 vectors, legal masks, the shared state
result = env.step(obs_a.legal[0], obs_b.legal[0])   # both actions at once; rewards (r, -r); info with every event

pz = LaneParallelEnv(GameConfig.m4_lasthit(), max_turns=512)
observations, infos = pz.reset(seed=0)           # {"A": {"observation", "action_mask"}, "B": ...}
```

## Repository layout

```
lanegame/                 engine (pure Python + NumPy/SciPy, no presentation code)
  config.py               all model parameters (GameConfig and its presets: v1, v2, siege, champions, m4 ... m4_lasthit)
  state.py                Markov lane state and canonical encoding
  mechanics.py            packet movement, merging, fights, thinning, crashes, tower fire, spawns, plates, vision timers
  game.py                 legal actions incl. siege objectives, farming, plates, gank model, reachable states
  matrix_game.py          zero-sum matrix games: saddle detection, HiGHS LP, certificates, support enumeration
  solver.py               backward induction -> values, mixed strategies, Q matrices
  best_response.py        exact policy evaluation, best responses, exploitability
  policy.py, baselines.py policy interface; scripted baselines
  analysis.py             exact forward occupancy and path statistics
  reachability.py         reachable lane and siege events with witnesses
  simulate.py             seeded Monte Carlo with realized gank outcomes
  scenarios.py, render.py named behavioral scenarios (23) and text traces of the lane
  io.py                   JSON export format for solutions (version 3)
  champions.py            champion mode (M2): positions, Hold/Yield/Trade/Safe/Recall, absence, trades, skill, ganks, death
  champions_m4.py         the M4 ruleset's state and resolver (Freeze maintenance, XP/levels, items, ledger, plates)
  progression.py          M4 integer tables: max HP, HP unit, attack power, damage, XP thresholds, purchases
  lane_enum.py            exhaustive reachable-lane enumeration and the enumerated bounds M4 relies on
  scenarios_m4.py         M4 parked-front harness and replayable scenario witnesses (data/m4_scenarios.json)
  env/                    the simultaneous-move RL environment: core.py (LaneEnv), observation.py / observation_m4.py
                          (own-view layouts, the lossless M4 decoder), agents.py / agents_m4.py (scripted agents),
                          trace.py (episode logs), discounted.py (E1/E2 evaluation with the guaranteed tail bound),
                          starts.py (witnessed start archives), pettingzoo_adapter.py (the PettingZoo Parallel API)
  rl/                     PyTorch: network.py (policy/value MLP, masked categorical), ppo.py (GAE, clipped update),
                          train.py (collector, PPO loop, continuing-time game times), selfplay.py (snapshot pool),
                          shared.py (shared-policy self-play, current / historical mixture), agent.py (NetAgent,
                          checkpoints with digests), evaluate.py (head-to-head, round robin, approximate best responses)
  diagnostics/            action audits and paired counterfactual rollouts from restorable states
  viewer/                 the Streamlit lane debugger (M4.2e-D)
  experiments/rl/         RL studies, the M4.3a smoke test and the M4.3b pilot package m43b/ (results/rl/; checkpoints in
                          results/rl/checkpoints/, untracked)
  experiments/m4/         M4 validation and calibration drivers (results/m4*/)
  experiments/champions/  champion-mode validation (results/champions/)
  experiments/siege/      the siege study (results/siege/); experiments/, experiments/v1/: the earlier exact models
tests/                    pytest suite: exact models, champion game, M4 mechanics, environment, PettingZoo, PPO references
results/                  outputs of every study, one folder per study, with the commands that produced them
docs/                     design_philosophy.md, rl_research_roadmap.md and, by phase, phase1/, rl/, m4/, natural_waves/
legacy/                   the original Q-learning prototype, kept unchanged for reference
```

## The earlier models

- **Farming revision** (`GameConfig.v2()`): event-based farming, Contest/Safe positioning, mass-based tower fire, a
  post-crash choice between crash resources and two turns of 50% vision. Documented in
  [docs/phase1/phase1_revision_report.md](docs/phase1/phase1_revision_report.md), results in `results/revision/`.
- **Original Phase 1** (`GameConfig.v1()`): wave actions only, no farming, one-packet-per-turn tower consumption.
  Documented in [docs/phase1/phase1_implementation_report.md](docs/phase1/phase1_implementation_report.md), results
  in `results/`.
- **The first Q-learning prototype** is in [legacy/](legacy/), audited in section B of the original report.
