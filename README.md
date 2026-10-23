# Laning as a Two-Player Game

Can a reinforcement learning agent learn League of Legends laning strategy without anyone teaching it? This project
builds a simplified model of the laning phase, abstracting it exactly with game theory, and trains
self-play agents on the full version. The reward is the gold an agent earns minus the gold its opponent earns; the agents have to discover strategies on their own.

## Rules of The Game

Every game of League of Legends opens with the laning phase. Two opposing players share a lane while waves of
computer-controlled minions march toward each other and fight. Players earn gold by landing the killing blow on enemy
minions (last-hitting) and can spend this gold in base to get stronger.

A few general rules of laning:

- Pushing your wave into the enemy tower costs your opponent minions and gives you time to go back to base. 
- A level or item lead makes you stronger than your opponent, so you will win an otherwise
even fight
- It is harder to get ganked on your "side" of the map, e.g. near your tower
- Damaging the enemy tower early pays extra gold through tower plates, along with
last hitting minions and killing enemy players. 

Good LoL players exploit these interactions all the time. Our agents are limited to represent that of a human player, it sees only the current game state and its legal moves (e.g. you can't "Freeze" if you're on the enemy's side of the map), and it is
rewarded only for building a gold lead. That means strategies like "freezing" a wave, recalling on timers, and warding on timers show up on their own through the agents' training.

Every rule follows the project's [design philosophy](docs/design_philosophy.md), which is to simplify how League works
without changing why its laning strategies work. 

## Current Findings (10/22/2026) :

- **Self-play agents found real laning strategies on their own.** They manage the wave differently depending on the
  situation. They freeze when freezing pays, picking their fights based on game state, and time their trips back to
  base in situations that make sense. Part of testing, was that we tested each behaviour by restoring the exact game moment, making a different choice and comparing how the
  game played out.
- **Agents can skip strategies that don't pay.** Warding on certain timers, unnecessarily long freezes and bullying a competent opponent
  cost more than they return here, and the agents learned to avoid them.
- **Modelling the opponent is a trade-off.** The final experiment gave one agent a running Bayesian estimate of its
  opponent's play style. Against the scripted styles it trained with, it built about three and a half times the lead
  of a pure self-play agent. Against new variants of those styles the gain was nearly as large on average, still good, but against
  styles built on a strong learned policy it gained nothing. Against strong opponents it was weaker, and it lost head
  to head to the pure self-play agent in all six matched pairs.
- **Limitations.** Freezing pays only in a narrow early window, and the agents don't fully exploit large power gaps as much as they probably should. Still exploring why this is. 

## Try it

### Install

Python 3.10 or later.

```bash
python -m venv .venv
.venv/Scripts/activate              # macOS/Linux: source .venv/bin/activate
pip install -e ".[test,rl]"
python -m pytest                    # about 1,600 tests, 9 to 12 minutes
```

### Play against the agents, or watch them

The demo is a web page with two modes. In Play you take one side against a trained agent, with every turn explained. In Watch you follow a match the engine
generates between the two agents, or one of three recorded research games that the server re-simulates and checks
when it starts. The demo needs the trained networks, Node 22.12 or later for the
one-time frontend build, and the `demo` extra.

```bash
pip install -e ".[demo]"
cd web && npm install && npm run build && cd ..
python -m lanegame.demo                 # then open http://127.0.0.1:8765/
```

Watch also has sequences: short walkthroughs of a recorded game in which each event leads to the next. The first,
"Stacking to Crashing", sequence, follows a pattern League players (good ones) know. Trades keep a low-health opponent from going back to
base, the wave nobody clears stacks and crashes into its tower, and the crash lets the agent take two tower plates.
Pick it under **Sequences** in Watch, or open http://127.0.0.1:8765/?mode=watch&sequence=stacking-to-crashing.

### Solve the small game

```bash
python -m lanegame solve --horizon 16     # about 10 seconds; --horizon 32 takes about 4 minutes and 3 GB
python -m lanegame exploit --horizon 16   # how much a best response gains against each scripted strategy
```

### Reproduce the research

Each milestone has a package under `lanegame/experiments/` that writes its results to `results/<milestone>/`.
[results/m5/reproducibility.md](results/m5/reproducibility.md) lists the commands, seeds and SHA-256 checksums behind
the final campaign. Trained networks are too large for git, so the repository doesn't include them.

## Documentation

- A final report on the mathematical model, roadmap, and findings is WIP but on the way.

