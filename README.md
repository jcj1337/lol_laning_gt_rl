# Laning as a Two-Player Game

Can a reinforcement learning agent learn League of Legends laning strategy without anyone teaching it? This project
builds a simplified model of the laning phase, solves a small version of it exactly with game theory, and trains
self-play agents on the full version. The reward is the gold an agent earns minus the gold its opponent earns; the
agents have to discover strategies on their own.

There's a walkthrough of the project, with screenshots, at [jcj1337.github.io/laning.html](https://jcj1337.github.io/laning.html).

## Motivation / Intro

Every game of League of Legends opens with the laning phase. Two opposing players share a lane while waves of
computer-controlled minions march toward each other and fight. Players earn gold by landing the killing blow on enemy
minions (last-hitting) and can spend this gold in base to get stronger.

A few general rules of laning:

- Pushing your wave into the enemy tower costs your opponent minions and gives you time to go back to base.
- A level or item lead makes you stronger than your opponent, so you will win an otherwise even fight.
- It is harder to get ganked on your "side" of the map, for example near your tower.
- Damaging the enemy tower early pays extra gold through tower plates, along with last-hitting minions and killing
  the enemy champion.

Good players exploit these interactions all the time. Our agents see only what a player would, the current game
state and their legal moves (you can't "freeze" the wave from the enemy's side of the map, for example). They are
rewarded only for building a gold lead, so any strategy they use, such as freezing a wave or timing a trip back to
base, has to show up on its own through training.

## What the simulator models

Two champions share one lane of a few cells between two towers. Each turn both choose at the same time what to do
with the minion wave (last-hit, thin it, push slowly or shove it), how to behave towards the other champion (hold,
trade, give ground, play safe) and whether to hit the tower, place a ward or go back to base. Minions fight by fixed
rules; champions earn gold and experience, level up, spend gold on stronger items, take tower plates and can be
ganked or killed. The game runs without a fixed end, and the laning phase ends around turn 94, at level 14.
[docs/model.md](docs/model.md) gives every rule.

## What was applied

- **Game theory.** A small version with only the minion waves is solved exactly: equilibria, best responses and how
  much a fixed style can be exploited.
- **Reinforcement learning.** The full game is too large to solve, so one network trained by PPO plays both sides
  against itself. Its strategies are tested by restoring game states and comparing its play with fixed alternatives,
  and its weaknesses by training dedicated attackers against it.
- **Opponent modelling.** The final campaign compared plain self-play with agents that also keep a Bayesian estimate
  of which known play style the opponent resembles.

## What was found

- In the small exact game, best play is passive. Neither side gains by crashing waves into the tower, while simple
  fixed styles are easy to exploit.
- Self-play agents learned laning behaviour that depends on the situation, each shown to pay against the
  alternatives: managing the wave, freezing it when the opponent backs off, choosing when to trade and when to go
  back to base. They also learned what does not pay here, such as warding or bullying a competent opponent.
- Several times the "missing" strategy turned out to be the game's incentive, not a failure to learn. When the
  agents shoved every wave, shoving really was the best reply until the economy was changed.
- Modelling the opponent pays against recognizable scripted styles (a lead about 9 units of gold larger, where one
  unit is a full wave of minions) but costs against competent opponents (about 0.6 units). Neither kind of agent is
  better overall, and both can be beaten by a trained attacker.

[docs/findings.md](docs/findings.md) has the evidence for each of these, and
[docs/research_overview.md](docs/research_overview.md) explains the final campaign for newcomers.

## Try it

### Install

Python 3.10 or later.

```bash
python -m venv .venv
.venv/Scripts/activate              # macOS/Linux: source .venv/bin/activate
pip install -e ".[test,rl]"
python -m pytest                    # about 1,500 tests, 9 to 12 minutes
```

### Play against or watch the agents

The demo is a web page with two modes. In Play you take one side against a trained agent, with every turn
explained. In Watch you follow a match the engine generates between the two agents, or one of three recorded
research games that the server re-simulates and checks when it starts. The demo needs the 7 trained networks
included in the repository, Node 22.12 or later for the one-time frontend build, and the `demo` extra.

```bash
pip install -e ".[demo]"
cd web && npm install && npm run build && cd ..
python -m lanegame.demo                 # then open http://127.0.0.1:8765/
```

Watch also has Sequences: short walkthroughs of a recorded game in which each event leads to the next. The first,
"Stacking to Crashing", follows a pattern good League players know. Trades keep a low-health opponent from going back
to base, the wave nobody clears stacks and crashes into its tower, and the crash lets the agent take two tower plates.
Pick it under **Sequences** in Watch, or open http://127.0.0.1:8765/?mode=watch&sequence=stacking-to-crashing.

### Reproduce the research

Run everything from the repository root.

```bash
python -m lanegame solve --horizon 32        # the exact game's equilibrium (a few minutes, about 3 GB of memory)
python -m experiments.siege all              # the exact game's studies, written to artifacts/siege/
python -m experiments.reference examples     # the engine-computed examples the model document quotes
python -m experiments.m5 prereg_check report # check the final campaign's protocol and rebuild its report from the records
python -m experiments.m6d overview           # check every number of the research overview against the records
```

The records of every study are in [artifacts/](artifacts/README.md). For the final run,
[artifacts/m5/reproducibility.md](artifacts/m5/reproducibility.md) lists the commands, seeds and SHA-256 checksums.
Only the 7 networks the demo uses are in the repository; the rest are too large for git, and rerunning the campaign
from scratch takes about 36 hours (if on six workers). Training and other expensive stages refuse to start without an
explicit confirm flag.

