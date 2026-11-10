# Laning as a Two-Player Game

Can a reinforcement learning agent learn League of Legends laning strategy without being explicitly taught how to play?

I built a simplified simulator of the laning phase to explore this. The project starts with a smaller game that can be
solved exactly using game theory, then moves to a more complete version where agents learn through self-play using
PPO.

The agents are rewarded based on their gold lead over the opponent. They aren't explicitly taught strategies like
freezing, slow pushing, or timing recalls. The idea is to see whether these behaviours emerge naturally from training.

A more detailed walkthrough, including screenshots, is available [here](https://jcj1337.github.io/laning.html).

## Motivation

In League of Legends, the laning phase is where two opposing players compete for gold and experience while managing
waves of minions.

There are a few basic interactions that make laning interesting:

- Pushing a wave into the enemy tower can cause the opponent to lose minions, while giving you time to recall.
- Having a level or item advantage makes it easier to win trades against your opponent.
- Your position in the lane affects how vulnerable you are to enemy ganks.
- Attacking the enemy tower gives additional gold through tower plates.

These mechanics create strategies that experienced players recognize, like freezing waves near their tower, building
slow pushes, or recalling after crashing a wave.

I wanted to see whether an RL agent could discover these same strategies without being told when or why to use them.

Rather than trying to recreate every mechanic in League, the simulator focuses on the higher-level decisions
involved in laning.

## The simulator

The game consists of two champions sharing a lane, with minion waves moving between two towers.

Both players choose their actions simultaneously each turn. They can manipulate the wave (last-hit, thin, slow push,
shove), interact with their opponent (trade, hold position, play safe), recall, ward, or attack the tower.

The simulator also models gold, experience, leveling, item purchases, tower plates, ganks, and deaths.

The laning phase typically ends around turn 94, at roughly level 14, although there isn't a fixed turn limit.

All of the mechanics and rules are documented in [docs/model.md](docs/model.md).

## Approach

I explored three main approaches:

- **Game theory:** I first built a smaller version of the simulator containing only wave management. Since the state
  space is manageable, I could solve it exactly and study equilibria, best responses, and how exploitable different
  strategies are.
- **Reinforcement learning:** For the full simulator, I used PPO with self-play, where a shared network controls both
  sides. I then evaluated the trained agents by replaying specific states, comparing their decisions against fixed
  alternatives, and training dedicated opponents to exploit them.
- **Opponent modelling:** I also experimented with giving agents a Bayesian estimate of their opponent's play style,
  to see whether adapting to the opponent actually improves performance compared to regular self-play.

## Results

A few interesting things came out of the experiments.

### 1. The exact game favours passive play

In the smaller wave-only game, neither player benefits from aggressively crashing waves into the enemy tower. The
equilibrium is relatively passive, although simple fixed strategies can still be exploited.

### 2. Self-play learned several recognizable laning behaviours

In the full simulator, the agents learned to adjust their decisions depending on the game state. This included
managing waves, freezing when the opponent gave up space, choosing when to trade, and finding opportunities to recall.

These behaviours were evaluated against alternative actions from the same states, rather than just identified from
watching games.

Some strategies, like warding or repeatedly pressuring a competent opponent, didn't provide enough value under the
simulator's rules.

### 3. The simulator's design was iterated upon many times

One of the more interesting parts of the project was figuring out whether a missing strategy was an RL problem or a
problem with the game itself.

For example, earlier versions of the simulator produced agents that shoved almost every wave. After investigating, it
turned out that shoving genuinely was the best response under those rules. The economy had to be adjusted before other
strategies became worthwhile.

This was a recurring theme: an agent not learning a particular strategy doesn't necessarily mean the training failed.

### 4. Opponent modelling results

The opponent-modelling agents performed noticeably better against recognizable scripted play styles, gaining roughly
9 additional gold units compared to regular self-play (one unit represents a full minion wave).

However, they performed slightly worse against competent learned opponents, losing about 0.6 units relative to the
baseline.

Overall, neither approach was consistently better, and both could still be exploited by specifically trained attackers.

The full results and supporting experiments are in [docs/findings.md](docs/findings.md). There's also a more accessible
summary of the final training campaign in [docs/research_overview.md](docs/research_overview.md).

## Try it

### Installation

Requires Python 3.10 or later.

```bash
python -m venv .venv
.venv/Scripts/activate              # macOS/Linux: source .venv/bin/activate
pip install -e ".[test,rl]"
python -m pytest                    # ~1,500 tests, 9–12 minutes
```

### Play against the agents

I also built a small web demo for interacting with the trained agents.

There are two modes:

- **Play:** Play one side of the lane against a trained agent, with explanations of what happens each turn.
- **Watch:** Watch matches between trained agents or replay one of three recorded research games. The server
  re-simulates and verifies the recorded games when it starts.

The demo uses 7 trained networks included in the repository. It also requires Node 22.12 or later for the initial
frontend build.

```bash
pip install -e ".[demo]"
cd web && npm install && npm run build && cd ..
python -m lanegame.demo
```

Then open [http://127.0.0.1:8765/](http://127.0.0.1:8765/).

Watch mode also includes **Sequences**, which are short walkthroughs of interesting moments from recorded games.

For example, *Stacking to Crashing* shows how an agent uses trades to prevent a low-health opponent from recalling,
allowing a minion wave to build up and eventually crash into the tower. This creates an opportunity to take two tower
plates.

You can find it under **Sequences** in Watch mode, or open it
[directly](http://127.0.0.1:8765/?mode=watch&sequence=stacking-to-crashing).

### Reproducing the experiments

All commands should be run from the repository root.

```bash
python -m lanegame solve --horizon 32        # Exact-game equilibrium (~3 GB memory)
python -m experiments.siege all              # Exact-game experiments
python -m experiments.reference examples     # Examples used in the model documentation
python -m experiments.m5 prereg_check report # Verify and rebuild the final campaign report
python -m experiments.m6d overview           # Verify the research overview against recorded results
```

The experimental records are stored in [artifacts/](artifacts/README.md).

For the final training campaign, [artifacts/m5/reproducibility.md](artifacts/m5/reproducibility.md) contains the
commands, random seeds, and SHA-256 checksums needed for reproduction.

Only the 7 networks used by the demo are included in the repository, since the remaining checkpoints are too large for
Git.

Reproducing the entire training campaign takes approximately 36 hours with six workers. Expensive training stages
require an explicit confirmation flag to avoid running them accidentally.
