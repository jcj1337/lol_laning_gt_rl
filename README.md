# Laning as a Two-Player Game

Can a reinforcement learning agent learn League of Legends laning strategy without anyone teaching it? This project
builds a simplified model of the laning phase, solves a small version of it exactly with game theory, and trains
self-play agents on the full version. The reward function is modelled as the gold they earn minus the gold their opponent
earns. No strategy is explicitly coded, agents need to discover
them themselves. 

## The question

Every game of League of Legends opens with the laning phase. Two opposing players share a lane while waves of
computer-controlled minions march toward each other and fight. Players earn gold by landing the killing blow on enemy
minions (last-hitting) and can spend this gold in base to get stronger.

The general rules of the game : 

- Pushing your wave into the enemy tower costs your opponent minions and gives you time to go back to base. It also
  leaves you far from your own tower, where the enemy jungler can ambush you (a gank).
- A level lead / item lead means youre stronger than your opponent and will win an even fight
- Holding the wave just outside your own tower (a freeze) starves your opponent of gold, but only while you can keep
  it there.
- Damaging the enemy tower early pays extra gold through tower plates.

In real LoL, good players exploit these interactions all the time. This project asks simply put whether a computer (RL agent) can learn to exploit them as well. Just like a humanm, it sees only the game state, its legal moves and prioritizes building advantges (getting gold/xp) to "win". If freezing, well-timed trips to base or selective trading show up, the agents found them on their own.

## What we found

- **The small game has an exact answer.** Wave management on its own is small enough to solve exactly. In its
  equilibrium, the pair of strategies neither player can improve on, the lane stays even and contested. An opponent can
  punish every fixed, predictable strategy, some of them heavily.
- **Self-play agents found real laning strategies on their own.** They manage the wave differently depending on the
  situation, freeze when freezing pays and break the opponent's freeze, pick their fights, and time their trips back to
  base. We tested each behaviour by restoring the exact game moment, making a different choice and comparing how the
  game played out.
- **They also skip strategies that don't pay in this model.** Wards, long freezes and bullying a competent opponent
  cost more than they return here, and the agents learned to avoid them.
- **Modelling the opponent is a trade-off.** The final experiment gave one agent a running Bayesian estimate of its
  opponent's play style. Against predictable opponents it built about three times the lead of a pure self-play agent,
  and two and a half times the lead against styles it never met in training. Against strong opponents it was weaker,
  and it lost head-to-head to the pure self-play agent in all six matched pairs.
- **Neither final agent is unbeatable.** Attacker agents, each trained against one frozen final agent, found ways to
  beat both kinds. Each attack worked only against its own target.
- **Some strategies only partly emerge.** Freezing pays only in a narrow early window, leads tend to fade later in the
  lane, and the agents don't fully exploit large power gaps.


## How we study it

1. **Solve a small version exactly.** The wave-management game is a two-player game in which one side's gain is the
   other's loss. Over 32 turns it has 172,907 decision points, few enough for backward induction and linear
   programming to find the exact equilibrium, the exact best response to any fixed strategy, and how much an
   opponent can exploit each strategy. These exact answers also give a ground truth for testing the learning code.
2. **Build a fuller game that can't be solved.** The full model adds champions with health, levels and experience,
   gold and spending, trades, ganks, deaths, trips back to base, tower plates, a ward spot and minions that attack
   champions. Both players move at the same time each turn, choosing from 61 moves. The game has no fixed end, and we
   score who leaves the laning phase ahead (around turn 94, when the champions reach level 14). The environment
   implements PettingZoo, the standard interface for multi-agent RL.
3. **Train with self-play.** One neural network plays both sides and learns with PPO, a standard RL algorithm. OpenAI
   Five trained its Dota 2 agents with the same shared-policy self-play. An early study tested OpenAI Five's mix of
   80% games against the current network and 20% against past versions. The later researcg train sagainst the
   current network only.
4. **Measure carefully.** Agents play scripted opponents and each other on matched random seeds, and every result
   comes with a confidence interval over independent training runs. Attacker agents trained to beat one frozen agent
   show how exploitable a strategy is. Counterfactual replays restore an exact game moment, change one decision and play on,
   which shows whether a behaviour actually pays. 

Every rule follows the project's [design philosophy](docs/design_philosophy.md), which is to simplify how League works
without changing why its laning strategies work. The numbers are stylized, not real League statistics.

## Try it

### Install

Python 3.10 or later.

```bash
python -m venv .venv
.venv/Scripts/activate              # macOS/Linux: source .venv/bin/activate
pip install -e ".[test,rl,viewer]"
python -m pytest                    # about 1,550 tests, 8 to 11 minutes
```

### Watch recorded games

The repository includes 15 recorded games from the final campaign. The viewer replays them turn by turn in the
browser.

```bash
streamlit run lanegame/viewer/app.py
```

In the sidebar, choose **Recorded trajectory** and set the folder to `results/m5/traces`. A web app where you can play
against the trained agents is planned.

### Solve the small game

```bash
python -m lanegame solve --horizon 16     # about 10 seconds; --horizon 32 takes about 4 minutes and 3 GB
python -m lanegame exploit --horizon 16   # how much a best response gains against each scripted strategy
```

### Use the environment

```python
from lanegame import GameConfig
from lanegame.env import LaneEnv
from lanegame.env.pettingzoo_adapter import LaneParallelEnv

env = LaneEnv(GameConfig.m45_calibrated(), max_turns=512)   # the version of the game the research uses
obs_a, obs_b = env.reset(seed=0)                     # each player's view, 90 numbers plus a mask of legal moves
result = env.step(obs_a.legal[0], obs_b.legal[0])    # both players move at once; the rewards are (r, -r)

pz = LaneParallelEnv(GameConfig.m45_calibrated(), max_turns=512)
observations, infos = pz.reset(seed=0)               # {"A": {"observation", "action_mask"}, "B": ...}
```

### Train an agent

```python
from lanegame import GameConfig
from lanegame.rl import PPOConfig, SharedConfig, train_shared

cfg = GameConfig.m45_calibrated()
ppo = PPOConfig.continuing(cfg, total_steps=102_400, seed=0)
train_shared(cfg, ppo, SharedConfig(current_fraction=1.0), out_dir="results/rl/checkpoints/demo")
```

This short run takes about 30 seconds on one CPU core. The research runs used 8 to 33 million transitions each.

### Reproduce the research

Each milestone has a package under `lanegame/experiments/` that writes its results to `results/<milestone>/`.
[results/m5/reproducibility.md](results/m5/reproducibility.md) lists the commands, seeds and SHA-256 checksums behind
the final campaign. Trained networks are too large for git, so the repository doesn't include them. Expensive stages
refuse to start without a `--confirm-...` flag. The docstring of `lanegame/__main__.py` lists every command-line
command.

## Repository layout

```
lanegame/                 the game, the RL environment, training and analysis
  mechanics.py, game.py   the minion wave and the small exact game
  solver.py               the exact equilibrium solver (backward induction and linear programming)
  champions_m4.py         the full laning model, with progression.py and positional.py
  config.py               every game version and its parameters
  env/                    the RL environment and its PettingZoo adapter
  rl/                     PPO, shared-policy self-play, evaluation and checkpoints
  opponent/               scripted opponent styles and the Bayesian belief about them
  diagnostics/            counterfactual replays and the end-of-lane score
  experiments/            one package per milestone, each writing to results/<milestone>/
  viewer/                 the Streamlit replay viewer
  demo/                   the web demo's game sessions, recordings and frames (no web server yet)
prototypes/               throwaway design prototypes of the web demo (M6a)
tests/                    about 1,550 tests, including bit-for-bit reproducibility checks
results/                  the output of every experiment
docs/                     designs, pre-registrations and reports
legacy/                   the original Q-learning prototype
```

## Documentation

- [docs/README.md](docs/README.md) indexes every design document and report.
- The [research roadmap](docs/rl_research_roadmap.md) records each milestone's decisions and links its report.
- The [mathematical reference](docs/m4/laning_game_mathematical_reference.md) gives every rule and equation
  of the current game.
- The [design philosophy](docs/design_philosophy.md) sets the standard for every mechanic.

## Roadmap

| Stage | What it did | Status |
|---|---|---|
| Phase 1 | Solved a small wave-management game exactly, with equilibria, best responses and exploitability | Complete ([study](docs/phase1/phase1_siege_vision_study.md)) |
| M1 to M3 | Built the RL environment and champion mechanics, wrote masked PPO and self-play, and ran an OpenAI Five-style self-play study | Complete ([study](docs/rl/openai_five_selfplay_report.md)) |
| M4 | Built and calibrated the full laning model, trained PPO self-play on it, and diagnosed what the agents miss | Complete ([closeout](docs/m4/m4_closeout_report.md)) |
| Opponent modelling | Tested whether a Bayesian belief about the opponent's style helps, first with existing agents, then in a matched training pilot | Complete ([pilot](docs/opponent_modelling/om2_pilot_report.md)) |
| M5 | The final training, opponent modelling against pure self-play, in 12 runs of 32.8 million transitions each | Complete, 2026-10-19 ([final report](docs/m5/m5e_closeout_report.md)) |
| M6 | A web app to play against the trained agents, watch them play each other and replay research games, plus reproduction bundles | In progress (the game-session backend is done; the visual design is in review) |
