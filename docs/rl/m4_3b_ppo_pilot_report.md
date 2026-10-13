# M4.3b — PPO self-play pilot and strategic learning diagnostics on `GameConfig.m4_lasthit()`

Branch `development/m4-3b-ppo-pilot`, on top of `main` at `1ad8848` (M4.3a). 2026-10-12.

**Purpose.** The first serious investigation of the README's research question on the M4 game: *can an RL agent
learn to take advantage of the strategic interactions of the laning simulator without being taught League
strategies?* M4.3a established that the trainer is correct and that a 400k-transition smoke run learns a narrow
clear-and-trade policy. This milestone asks three questions: (1) is PPO self-play learning stronger policies; (2) are
the learned decisions state-dependent, or a small set of dominant actions; (3) where strategies are missing, is that
the game, the reward, or the training process. No mechanic, reward, observation or hyperparameter of the game was
changed; every preset's digest pin is unchanged.

**Deliverables.** A reproducible pilot (`lanegame.experiments.rl.m43b`, 12 training runs with checkpoints every 25,600
transitions); evaluation against a stronger reference suite with learning curves, a headline W = 1000 evaluation,
cross-play and bounded exploitability; a conditional-behaviour and counterfactual analysis of what the policies do
and whether it depends on the state; the Recall / death economics study; a mixed-start follow-up; exported
trajectories for the viewer; this report; tests; the roadmap and README updates.

## Contents

1. Design of the pilot
2. Training: budgets, throughput, PPO health
3. Is PPO learning? Learning curves and the headline evaluation
4. Cross-play and exploitability
5. Exploration: the entropy coefficient and the historical mixture
6. What the policies do: conditional behaviour
7. Counterfactual first moves: is Clear chosen because it is right?
8. Recall and death economics
9. The mixed-start follow-up
10. Representative trajectories
11. Answers to the three questions
12. Limitations
13. Files, tests, reproduction
14. Recommendations for M4.3c

## 1. Design of the pilot

### 1.1 Conditions and budgets

| condition | self-play | entropy coefficient | training starts | seeds | budget (agent transitions) | role |
|---|---|---|---|---|---|---|
| `base` | 100% current policy, both seats trained | 0.01 | s0 | 0, 1, 2 | 2,048,000 | the baseline (the brief's §2) |
| `ent03` | 100% current | **0.03** | s0 | 0, 1, 2 | 2,048,000 | the exploration control (§3) |
| `hist80` | **80% current / 20% historical** (uniform over the last 8 snapshots) | 0.01 | s0 | 0, 1, 2 | 2,048,000 | the bounded historical-opponent comparison (§8) |
| `mixed` | 100% current | 0.01 | **50% s0 / 50% a witnessed archived state** | 0, 1, 2 | 2,048,000 | the follow-up (§7), run after the pilot |

The budget is twice the roadmap's 1M proposal: M4.3a measured about 5,000 transitions/s per process, so 2M per run
costs 11–13 minutes per run and the nine pilot runs trained in 12.8 minutes in nine parallel processes; the M4.3a
entropy collapse happened inside the first 400k transitions, so the longer budget is what shows whether behaviour
keeps changing after it. Every run is `lanegame.rl.train_shared` with `PPOConfig.continuing` unchanged: γ = 0.99
(the game's), λ = 0.95, segments of 128 turns, games of 512 turns with seeded staggered first games, bootstrapped
truncations from the pre-reset observation, reward scale 0.1, value clip 0.5, learning rate 3e-4 annealed over the
budget, 4 epochs, minibatch 256, clip 0.2, 16 environments, one torch thread; action masking, the economic reward,
own-perspective observations and shared parameters, checkpoint digests and provenance all as audited in M4.3a. The
conditions differ only in the entropy coefficient, the current/historical fraction and the start distribution.

### 1.2 Reference opponents

The M4.3a readout used the scripted natural-wave suite, which every trained policy beat by 10–19 resources. This
milestone evaluates against the REACTIVE policies of the M4.2e-B/C policy games (`zoning_natural.POLICIES`: they read
HP, contact, the lane and the siege, Recall at 40% HP and reset while the enemy is away), wrapped as environment
agents (`m43b.references.ReactiveAgent`, deterministic, a function of the observation), plus the crash-and-Recall
viewer policy and the M4.3a `trading_pusher` for continuity:

| reference | plays | why it is in the suite |
|---|---|---|
| `farm` | Hold the wave, Hold position | the pure last-hitter: the weak passive floor |
| `push_punish` | slow push (Thin while ≤ 2 HP ahead) + punish an opponent in reach + Pressure | the equal-power equilibrium line of M4.2e-C §6.4 (both sides slow-push) |
| `clear_punish` | Clear every turn + punish + Pressure | the stronger side's equilibrium line (Clear + punish) |
| `bully_shove` | Trade every turn the enemy is on the lane; Clear while it is away; Pressure | the aggressive line |
| `freeze_punish` | the threshold freeze controller + punish | the wave-control line the geometry makes hardest |
| `respect` | Safe while the enemy starts within reach, else stand off | the zoned champion's best reply to bare threats |
| `crash_recall` | build a stack, Clear it into the tower, walk back and Recall | the League Recall window |
| `trading_pusher` | slow push + duel-edge trades (M4.3a suite) | continuity with the smoke test |

The four in bold-faced use on the learning curves are `farm`, `push_punish`, `clear_punish` and `bully_shove`; the
headline uses all eight, the untrained initial network of each seed, cross-play between every pair of finals and
the attackers.

### 1.3 Evaluation protocol

Every value is the discounted payoff of protocol E1 (`lanegame.env.discounted.evaluate_discounted`): from s0, the
game's γ = 0.99, both seats on the same seeds (paired), the guaranteed truncation bound γ^W Ω with Ω = 107.04.

- **Development** (learning curves, snapshot profiles, counterfactual windows): W = 460 (bound 1.05 resources), 16
  paired episodes per matchup, one fixed evaluation seed for every curve point so that points are paired.
- **Headline** (every number of §3–§5 and §9 that ranks policies): W = 1000 (bound 0.005), 32 paired episodes per
  matchup, a separate evaluation seed.
- Two kinds of uncertainty are kept apart throughout: the paired 95% interval of one matchup over evaluation
  episodes, and the spread across the three independently trained seeds of a condition (mean ± 1.96 SE over seeds;
  n = 3, indicative only).

All evaluation streams derive from fresh SeedSequence roots (61,000–69,999), disjoint from training and from the
earlier studies.

### 1.4 Instruments

| instrument | module | what it measures |
|---|---|---|
| learning curves and behaviour over training | `curves` | every 4th snapshot (every 102,400 transitions) and the final against the four curve references; a self-play profile per point (action shares, mean category probabilities, entropy and entropy / log |legal| on present turns, deaths, recalls, crashes, returns after death / Recall) |
| headline | `evaluate` | finals × references, finals × initial network, cross-play of all 12 finals |
| exploitability | `exploit` | one PPO attacker per final: `train_ppo` against the frozen target under continuing time, warm-started from the target's weights (M3c), 1,024,000 transitions, evaluated at W = 460 and 1000 and against the other seeds' finals (transfer) |
| conditional behaviour | `behavior` | one row per present turn (24 self-play games of 460 turns per final, both seats; 12 games against `push_punish` and `clear_punish`): the state from the champion's side, the policy's category probabilities, entropy, the action played, what followed; 21 named, overlapping situations; the mutual information between the realized action and a coarse state partition |
| counterfactual first moves | `behavior.counterfactual` | from 8 sampled reachable self-play states per situation (11 situations), paired rollouts (`lanegame.diagnostics.branch`, common random numbers) of every legal first move (Hold / Thin / Clear with Hold position, Trade, Safe, Yield, Pressure, Recall) against the policy's own move, 16 rollouts per candidate over 300 turns, discounted |
| Recall / death economics | `recall` | frequencies; counterfactual Recalls at Recall-legal states in six classes; the value of "just died" against "just recalled" from the same state (paired state variants) |
| start archive | `archive` | witnessed reachable states stratified by situation, mirror-closed, verified by replay |
| trajectories | `trajectories` | viewer recordings (integer chance seed + action list) of representative games |

## 2. Training: budgets, throughput, PPO health

Nine pilot runs (`python -m lanegame.experiments.rl.m43b train --workers 9`), 2,048,000 agent transitions each
(1,000 updates of 16 environments × 128 turns × 2 seats; 1,024,000 environment turns; 2,000 games of 512 turns),
12.8 minutes wall-clock, 2,700–3,300 agent transitions/s per process under the nine-process load (the single-process
rate of M4.3a was about 5,400). The historical condition collected 2,050k transitions (its historical games yield
one trainable seat per turn). The three `mixed` runs trained afterwards in three processes.

| run | entropy first → last | explained variance (last 10) | KL mean / max | grad norm mean / max (pre-clip) | clip fraction | V̂(s0) | max \|V̂\| | std of λ-returns (last 10) | single-choice share | historical games |
|---|---|---|---|---|---|---|---|---|---|---|
| base_s0 | 2.58 → 0.53 | 0.59 | 0.0025 / 0.0104 | 0.52 / 1.74 | 0.029 | +0.08 | 10.7 | 0.92 | 6.0% | 0 / 2000 |
| base_s1 | 2.58 → 0.56 | 0.68 | 0.0023 / 0.0114 | 0.53 / 1.86 | 0.028 | +0.08 | 10.8 | 1.03 | 9.0% | 0 / 2000 |
| base_s2 | 2.58 → 0.53 | 0.64 | 0.0023 / 0.0114 | 0.51 / 1.81 | 0.027 | −0.14 | 10.1 | 1.02 | 6.4% | 0 / 2000 |
| ent03_s0 | 2.58 → 1.10 | 0.63 | 0.0029 / 0.0100 | 0.57 / 1.43 | 0.031 | −0.03 | 10.7 | 1.08 | 7.9% | 0 / 2000 |
| ent03_s1 | 2.58 → 1.05 | 0.63 | 0.0029 / 0.0100 | 0.56 / 1.48 | 0.031 | −0.09 | 10.8 | 1.06 | 8.2% | 0 / 2000 |
| ent03_s2 | 2.58 → 0.97 | 0.62 | 0.0030 / 0.0103 | 0.56 / 1.59 | 0.032 | +0.07 | 10.1 | 1.05 | 7.6% | 0 / 2000 |
| hist80_s0 | 2.58 → 0.68 | 0.66 | 0.0026 / 0.0104 | 0.55 / 8.69 | 0.030 | +0.12 | 10.7 | 0.97 | 5.8% | 408 / 2208 |
| hist80_s1 | 2.58 → 0.60 | 0.70 | 0.0023 / 0.0114 | 0.54 / 10.78 | 0.027 | +0.19 | 10.8 | 1.11 | 9.7% | 434 / 2221 |
| hist80_s2 | 2.58 → 0.57 | 0.73 | 0.0024 / 0.0114 | 0.54 / 7.94 | 0.028 | −0.23 | 10.1 | 1.06 | 8.3% | 432 / 2219 |
| mixed_s0 | 2.58 → 0.42 | 0.62 | 0.0027 / 0.0119 | 0.55 / 1.78 | 0.031 | +0.00 | 10.7 | 1.11 | 9.0% | 0 / 2000 (1,018 archive starts) |
| mixed_s1 | 2.58 → 0.59 | 0.67 | 0.0025 / 0.0104 | 0.49 / 1.77 | 0.030 | +0.00 | 10.2 | 0.85 | 5.6% | 0 / 2000 (1,047) |
| mixed_s2 | 2.58 → 0.59 | 0.73 | 0.0025 / 0.0101 | 0.51 / 1.64 | 0.029 | −0.09 | 9.9 | 1.02 | 6.4% | 0 / 2000 (987) |

All M4.3a health checks pass on every run (finite losses and gradients, no termination and one truncation per game,
|V̂| ≤ Ω, explained variance positive, gradient norms of order 1, KL < 0.05, equal seat counts in the 100% runs;
`results/rl/m43b/train.json`). The entropy of `base` falls to about 0.6 by 400k transitions and then **stays there**
(0.62, 0.64, 0.67, 0.75, 0.62, 0.63, 0.54, 0.56 nats at 400k-step intervals for seed 0): it does not collapse further
over the next 1.6M transitions. `ent03` holds at 1.0–1.1 nats. The historical condition's gradient spikes (max 8–11)
occur in rollouts of historical games (a snapshot's play produces off-distribution samples); its mean norm is the
baseline's. The critic stays on the return scale in every run (|V̂| ≤ 10.8 against Ω = 107; V̂(s0) within ±0.25 of the
symmetric value 0).

## 3. Is PPO learning? Learning curves and the headline evaluation

### 3.1 Learning curves (development window W = 460, 16 paired episodes, `results/rl/m43b/fig_curves.png`)

Discounted payoff of the snapshot against the reference, mean over seeds ± 1.96 SE over seeds:

| condition | transitions | vs `farm` | vs `push_punish` | vs `clear_punish` | vs `bully_shove` |
|---|---|---|---|---|---|
| initial network | 0 | −2.49 | −1.99 | −7.42 | +6.04 |
| base | 102,400 | +10.96 ± 0.12 | +14.11 ± 0.38 | +6.41 ± 0.76 | +14.58 ± 0.28 |
| base | 409,600 | +11.58 ± 0.47 | +15.23 ± 0.48 | +6.58 ± 0.52 | +15.40 ± 0.57 |
| base | 1,024,000 | +10.91 ± 0.55 | +14.64 ± 1.56 | +7.45 ± 1.19 | +15.71 ± 1.64 |
| base | 2,048,000 | +10.89 ± 0.76 | +14.14 ± 2.20 | +8.49 ± 1.02 | +15.60 ± 0.38 |
| ent03 | 2,048,000 | +11.58 ± 0.22 | +15.27 ± 0.18 | +8.94 ± 0.19 | +16.37 ± 0.27 |
| hist80 | 2,048,000 | +11.73 ± 0.75 | +15.01 ± 0.78 | +8.62 ± 1.67 | +15.81 ± 1.04 |

Three things the curves say:

- **Almost all of the measurable progress against the references happens in the first 100k transitions** (5% of
  the budget). The untrained network loses to the farmer and the slow pusher and loses heavily to the Clear line;
  after 50 updates the policy beats all four by 6–15 resources and the curves are flat afterwards, within the seed
  spread, except against `clear_punish`, the strongest reference, where the value keeps creeping up (+6.4 → +8.5 for
  `base`, +8.9 for `ent03`). The references are deterministic scripted policies; once a policy dominates them, they
  cannot show further self-play progress, which is why §3.2 also plays the finals against their own earlier
  snapshots.
- **Behaviour keeps changing long after the reference values stop moving** (`fig_behavior_training.png`). At 100k
  the policy is 88% Clear with no Recall (1 per game and side, 11 deaths); between 100k and 400k it discovers
  Recall (17.6 per 460-turn game and side at 410k) and its deaths fall to 4; over the next 1.6M transitions Clear
  recedes from 0.88 to 0.75, Thin and Hold grow from 0.08 / 0.04 to 0.13 / 0.12, Trade settles at 0.27, Safe at
  0.14, Yield at 0.16, Recall at 0.04 of present turns (14 per game), deaths at 6 and own crashes at 38 per game and
  side. The M4.3a smoke test (same seed, same configuration, budget 409,600) reported no Recall at all: its learning
  rate had annealed to zero before the discovery, which these runs make at a fifth of their schedule. Budget-dependent
  schedules make short runs unrepresentative of long ones, as M4.3a's report already warned.
- **The three conditions are indistinguishable against the references at this budget** at the seed level; `ent03`
  is 0.5–1.1 higher against every reference with a third of the seed spread, a difference inside the baseline's
  seed interval. The headline evaluation (§3.2) and the cross-play (§4) are the proper comparison.

### 3.2 Headline evaluation (W = 1000, 32 paired episodes, `fig_headline.png`, `results/rl/m43b/report.md` §3)

Discounted payoff of the final to the reference (guaranteed truncation bound 0.005). Mean over the three seeds
± 1.96 SE over seeds; the per-seed paired intervals over evaluation episodes have half-widths of 0.2–0.7.

| condition | `farm` | `push_punish` | `clear_punish` | `bully_shove` | `freeze_punish` | `respect` | `crash_recall` | `trading_pusher` | initial net | seed SD (mean over refs) | episode half-width (mean) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base | +11.33 ± 0.76 | +14.43 ± 1.50 | **+9.14 ± 0.75** | +16.18 ± 0.58 | +15.12 ± 2.01 | +10.47 ± 0.26 | +10.87 ± 0.72 | +18.23 ± 1.49 | +12.05 ± 0.11 | 0.89 | 0.38 |
| ent03 | +11.93 ± 0.26 | +15.72 ± 0.05 | **+9.61 ± 0.10** | +16.77 ± 0.53 | +16.52 ± 0.20 | +11.06 ± 0.21 | +11.20 ± 0.33 | +18.68 ± 0.32 | +12.34 ± 0.13 | 0.22 | 0.38 |
| hist80 | +11.91 ± 0.39 | +15.60 ± 0.75 | +8.87 ± 1.04 | +16.43 ± 1.16 | +16.33 ± 1.05 | +11.13 ± 0.31 | +10.93 ± 0.52 | +18.66 ± 1.05 | +11.98 ± 0.41 | 0.69 | 0.36 |

- **Every final beats every reference**, by 8.4 (the weakest seed against `clear_punish`) to 19.8 resources, and
  beats its own untrained network by 12. The strongest reference is the Clear + punish line, the stronger-side
  equilibrium policy of M4.2e-C; the freeze controller, the equal-power slow-push line and the stand-off are beaten
  by 10–16; the aggressive lines by 16–20. For scale: a full fresh wave is worth 1 resource and the lane's income is
  about 0.45 per side per turn, so +9 at γ = 0.99 is about a fifth of the discounted income.
- **Variation across seeds exceeds variation across evaluation episodes** for `base` and `hist80` (seed SD 0.9 and
  0.7 against an episode half-width of 0.4), and not for `ent03` (0.2): the three 0.03 finals are nearly the same
  policy by this measure, the three baseline finals are not. Any claim about a condition therefore rests on the
  seed spread, not on the matchup intervals.
- Against the references `ent03` is ahead of `base` on all eight (by 0.5–1.5), `hist80` on seven; the differences
  are inside the baseline's seed interval on most references and outside it on none by a wide margin. The direct
  comparison is the cross-play.

### 3.3 Self-play progress after the plateau: the final against its own snapshots (W = 1000)

| condition | vs its snapshot at 102,400 | at 409,600 | at 1,024,000 |
|---|---|---|---|
| base | +4.00 ± 0.57 (4.6, 3.6, 3.9) | +1.44 ± 0.45 (1.6, 1.7, 1.0) | +0.53 ± 0.20 (0.7, 0.4, 0.4) |
| ent03 | +3.54 ± 0.86 (3.6, 2.8, 4.3) | +1.63 ± 0.59 (1.0, 1.9, 2.0) | +0.53 ± 0.38 (0.3, 0.9, 0.4) |
| hist80 | +3.14 ± 0.60 (3.6, 2.6, 3.2) | +1.91 ± 0.76 (2.4, 1.2, 2.2) | +1.02 ± 0.50 (1.5, 0.9, 0.7) |

Every final beats every earlier version of itself, by 3–4.6 against the 100k snapshot that already dominated the
references, by 1–2.4 against the 400k snapshot (which had discovered Recall), and by 0.3–1.5 against the midpoint of
the run. Self-play keeps producing a stronger policy with diminishing returns; the references stopped registering it
after 100k because they are dominated, not because learning stopped. This is also the scale of what the second
half of the budget bought: about half a resource of discounted value, against about 12 for the first 100k.

**Answer to question 1.** Yes. PPO self-play learns policies that beat the strongest scripted references of the
M4.2e studies by 9–16 resources, beat their untrained network by 12, and keep improving against their own earlier
versions over the whole budget with diminishing returns. The policies are exploitable by about 1–2 resources by a
dedicated attacker (§4.2), which bounds how far from an equilibrium they are on this measure.

## 4. Cross-play and exploitability

### 4.1 Cross-play between the finals (W = 1000)

Payoff to the row final against the column final, the full 9 × 9 matrix is in `results/rl/m43b/report.md` §3.2;
condition against condition is the mean over the seed pairs (9 across conditions, 6 within) ± 1.96 SE over pairs:

| row \ column | base | ent03 | hist80 |
|---|---|---|---|
| base | 0.00 ± 0.37 | **−0.50 ± 0.22** | −0.30 ± 0.24 |
| ent03 | **+0.50 ± 0.22** | 0.00 ± 0.32 | 0.00 ± 0.17 |
| hist80 | +0.30 ± 0.24 | 0.00 ± 0.17 | 0.00 ± 0.21 |

- The cross-play values are small: no final beats another by more than 1.0 (base_s0 against hist80_s2), and most
  pairs are inside ±0.5. Twelve independently trained networks on this game land within one resource of each
  other in direct play while beating every scripted reference by ten. They are, by this measure, close to one
  another.
- **The 0.03 control beats the baseline** by 0.50 ± 0.22 over the nine seed pairs (every one of the nine pairs is in
  its favour or even) and is even with the historical condition; the historical condition beats the baseline by
  0.30 ± 0.24. Within a condition the seed pairs are even on average, as they must be, with pair-level differences up
  to 0.6 (base_s1 beats base_s2 by 0.57): the three baseline seeds are not interchangeable.
- Ordering by the cross-play, by the references (§3.2) and by exploitability (§4.2) agree: ent03 ≥ hist80 ≥ base,
  with the whole spread inside one resource.

### 4.2 Bounded exploitability (`fig_exploit.png`, `results/rl/m43b/exploit.json`)

One PPO attacker per final: `train_ppo` against the frozen target only, under the same continuing-time semantics as
the targets' training, warm-started from the target's own weights (so that it starts at parity and any gain is the
target's exploitability rather than the attacker's learning speed), 1,024,000 transitions (half the target's
budget; about 17 minutes each under load, since the attacker's collector queries the frozen opponent network one
environment at a time), evaluated on held-out seeds.

| target | attacker's payoff, W = 460 | W = 1000 [95% CI] | the same attacker vs the other seeds' finals |
|---|---|---|---|
| base_s0 | +1.95 | **+1.98** [+1.63, +2.32] | −0.31, −0.13 |
| base_s1 | +1.06 | +1.10 [+0.76, +1.43] | +1.00, +0.49 |
| base_s2 | +1.15 | +1.18 [+0.93, +1.42] | +1.10, +0.21 |
| ent03_s0 | +0.73 | +0.75 [+0.38, +1.11] | +0.56, +1.27 |
| ent03_s1 | +0.93 | +0.95 [+0.61, +1.29] | +0.93, +0.60 |
| ent03_s2 | +0.91 | +0.94 [+0.64, +1.24] | +0.50, +0.29 |
| hist80_s0 | +1.18 | +1.22 [+0.85, +1.60] | +0.09, +0.01 |
| hist80_s1 | +1.41 | +1.44 [+1.10, +1.77] | +0.76, +0.21 |
| hist80_s2 | +1.41 | +1.45 [+1.15, +1.75] | +0.37, −0.19 |

Mean over seeds: `base` +1.42 ± 0.55, `ent03` +0.88 ± 0.13, `hist80` +1.37 ± 0.14 (1.96 SE over seeds). These are
lower bounds on exploitability. They are small on the scale of the game (a final beats the references by 9–16; the
attackers' entropies end at 0.3–0.5 nats, as collapsed as their targets'), and they are seed-specific in part: an
attacker trained on one seed gains 0.2–0.7 on average against the other seeds of the same condition, and the
`base_s0` attacker loses to the other two `base` finals. The 0.03 control is the least exploitable of the three and
its exploits transfer best (0.69), i.e. its three seeds are the most alike. The historical mixture does not reduce
exploitability at this budget (1.37 against 1.42), which is the main thing it was supposed to buy; §5 returns to it.

The attackers' undiscounted training returns reach +3 to +17 per 512-turn game, which is the same effect seen on
the discounted scale: a gain of about 0.02 resources per turn is worth about 2 under γ = 0.99. The exploit is a
small, steady per-turn edge, not a decisive refutation of the target's play.

## 5. Exploration: the entropy coefficient and the historical mixture

### 5.1 Does the entropy coefficient drive the Clear-and-trade convergence?

| at 2,048,000 transitions (mean over seeds) | base (0.01) | ent03 (0.03) | hist80 (0.01, 80/20) |
|---|---|---|---|
| policy entropy over the training batch (nats) | 0.54 | 1.04 | 0.62 |
| entropy / log \|legal\| on present turns (self-play) | 0.21 | 0.41 | 0.24 |
| wave shares Hold / Thin / Clear | 0.12 / 0.13 / 0.75 | 0.16 / 0.21 / 0.63 | 0.16 / 0.14 / 0.71 |
| behaviour shares Hold / Yield / Trade / Safe / Recall | 0.40 / 0.16 / 0.27 / 0.14 / 0.04 | 0.30 / 0.17 / 0.36 / 0.12 / 0.04 | 0.39 / 0.14 / 0.33 / 0.10 / 0.04 |
| deaths / recalls / own crashes per game and side (460 turns) | 6.0 / 14.3 / 38.5 | 8.4 / 15.4 / 43.7 | 9.6 / 11.4 / 39.9 |
| I(wave; state) / H(wave) (§6) | 0.12 | 0.08 | 0.14 |
| headline vs references (mean of the eight) | +13.2 | +13.9 | +13.7 |
| cross-play vs base | — | +0.50 ± 0.22 | +0.30 ± 0.24 |
| exploitability lower bound (W = 1000) | 1.42 ± 0.55 | 0.88 ± 0.13 | 1.37 ± 0.14 |
| seed SD of the reference values | 0.89 | 0.22 | 0.69 |

- The baseline's entropy does **not** collapse: it falls to 0.6 nats by 400k transitions and stays between 0.5 and
  0.75 for the remaining 1.6M, with the normalized entropy on present turns at 0.2 (the policy keeps about a fifth
  of the available randomness, mostly on the behaviour dimension; Clear itself is at 0.75–0.88 from 100k on). The
  M4.3a reading of 0.8 at 400k was this plateau, not the beginning of a collapse.
- Tripling the coefficient roughly doubles the entropy (1.0 nats; 0.41 normalized) and moves the policy **towards
  Thin and Trade**: Clear 0.75 → 0.63, Thin 0.13 → 0.21, Trade 0.27 → 0.36. That is not diversity for its own
  sake: the 0.03 finals are ahead of the baseline against every reference, beat it in direct play by half a
  resource, are the least exploitable of the three conditions and are far more reproducible across seeds (seed SD
  0.22 against 0.89). The extra entropy is spent in the places where the counterfactuals of §7 say the baseline's
  preference costs it something, which is the mark of exploration that found value rather than noise.
- Clear remains the dominant wave action under both coefficients (63–75% of present turns), so the dominance of
  Clear is not an artefact of the exploration setting; whether it is a property of the game is the subject of §7.

### 5.2 The historical mixture at this budget

The 80/20 condition is indistinguishable from the baseline against the references (+0.5 on average, inside the seed
spread), beats it by 0.30 ± 0.24 in cross-play, is exactly as exploitable (1.37 against 1.42, and its attackers'
exploits transfer least across seeds, 0.21), dies more (9.6 per game against 6.0) and trades more (0.33 against
0.27). Its historical games (18% of games, 20% of transitions once the pool exists) produced the only gradient
spikes of the study (pre-clip norms of 8–11 against a mean of 0.55: a snapshot's play is off the current policy's
distribution). The method bought nothing measurable here that the entropy coefficient did not buy more cheaply;
M4.3c's 100/0 against 80/20 comparison should be run at a budget where the pool spans a strategically different
past (the 25,600-transition spacing of this study keeps the pool within the last 10% of the run, where the policy
hardly changes) before the method is judged.

## 6. What the policies do: conditional behaviour

`results/rl/m43b/behavior.json`, `report.md` §5, `fig_conditional.png`. One row per present turn of the final:
24 self-play games of 460 turns per final (both seats; about 20,000 rows per condition), 12 games against
`push_punish` and against `clear_punish`. The probabilities are the policy's own distribution at the decision,
collapsed to categories; the situations overlap by design.

### 6.1 The baseline in self-play, by situation (mean over the three seeds)

| situation | share of turns | P(Hold / Thin / Clear) | P(Hold / Yield / Trade / Safe / Recall) | P(Pressure) | H / log \|legal\| | reward per turn |
|---|---|---|---|---|---|---|
| all present turns | 1.00 | 0.12 / 0.13 / **0.75** | 0.39 / 0.16 / 0.27 / 0.14 / 0.03 | 0.03 | 0.21 | +0.05 |
| neutral lane (fronts fighting at the lane cell) | 0.33 | 0.05 / 0.13 / 0.82 | 0.30 / 0.23 / **0.39** / 0.05 / 0.02 | 0.00 | 0.20 | 0.00 |
| own wave at own +1 (next to the enemy tower) | 0.05 | 0.13 / 0.19 / 0.68 | 0.35 / 0.10 / 0.30 / 0.21 / 0.04 | 0.00 | 0.24 | −0.17 |
| enemy wave at own −1 (next to the own tower) | 0.04 | 0.04 / 0.08 / **0.88** | 0.44 / 0.19 / 0.05 / **0.31** / 0.01 | 0.00 | 0.25 | +0.33 |
| defending (enemy mass under the own tower) | 0.03 | 0.05 / 0.04 / **0.91** | **0.71** / 0.08 / 0.01 / 0.16 / 0.03 | 0.00 | 0.17 | −0.22 |
| sieging (own mass under the enemy tower, champion there) | 0.01 | 0.07 / **0.29** / 0.63 | 0.85 / 0.00 / 0.08 / 0.06 / 0.00 | **0.76** | 0.24 | −0.03 |
| freeze-like (enemy front at own −1, own front in the tower cell) | 0.02 | 0.06 / 0.12 / 0.82 | 0.27 / 0.14 / 0.06 / **0.52** / 0.02 | 0.00 | 0.28 | +0.25 |
| enemy champion absent | 0.07 | 0.10 / 0.10 / 0.80 | 0.47 / 0.07 / 0.35 / 0.08 / 0.03 | 0.08 | 0.15 | +0.72 |
| low HP (≤ 40%), enemy present | 0.17 | 0.21 / 0.15 / 0.64 | 0.43 / 0.15 / 0.16 / 0.13 / 0.13 | 0.03 | 0.23 | −0.15 |
| low HP and Recall legal | 0.09 | **0.29** / 0.07 / 0.64 | 0.38 / 0.12 / 0.18 / 0.07 / **0.25** | 0.00 | 0.20 | −0.06 |
| Recall legal, HP ≥ 80% | 0.30 | 0.13 / 0.07 / 0.80 | 0.38 / 0.13 / 0.35 / 0.14 / **0.00** | 0.00 | 0.18 | +0.19 |
| Recall legal, ≥ 4 resources unspent | 0.25 | 0.21 / 0.09 / 0.70 | 0.32 / 0.14 / 0.29 / 0.12 / 0.12 | 0.00 | 0.21 | +0.01 |
| Pressure legal | 0.03 | 0.02 / 0.29 / 0.69 | 0.95 / 0.00 / 0.03 / 0.02 / 0.00 | **0.90** | 0.14 | +0.45 |
| stronger (power ratio ≥ 1.25) / weaker (≤ 0.8) | 0.003 / 0.003 | 0.03 / 0.10 / 0.87 and 0.13 / 0.09 / 0.78 | Hold 0.66 / Trade 0.18 and Hold 0.76 / Recall 0.08 | 0.08 / 0.16 | 0.13 / 0.15 | +0.23 / −0.23 |

What the table says, read with the state-dependence summary (I(action; state) / H(action) over a coarse partition
of enemy presence × contact cell × power × HP × sieges: wave 0.06–0.20, behaviour 0.12–0.15 for the baseline
seeds; 0.01 for the untrained network; 0.98 and 0.44 for the deterministic `clear_punish` and `push_punish`):

- **Clear is the default everywhere, and its probability moves with the state within a band of 0.6–0.9.** It is
  highest when the enemy wave arrives at the own tower (0.88) and when an enemy mass sits under it (0.91): the
  policy clears what comes to it. It is lowest at low HP (0.64) and while sieging (0.63), where **Thin** takes a
  share (0.29 with Pressure: the policy thins the defenders behind the tower while pressing, which is the damage
  Apply Pressure used to carry) and where **Hold** takes a share at low HP (0.21–0.29). The spread of P(Clear)
  across situations is 0.23 (Pressure legal) to 0.98 (defending) for seed 0.
- **The behaviour dimension is where the state dependence lives.** Trade is 0.39 in the neutral lane and 0.01–0.05
  when defending or when the enemy wave arrives; Safe is 0.05 in the neutral lane and 0.31–0.52 when the enemy wave
  or its mass is at the own tower (the policy steps back and lets the tower and its Clear do the work: a defensive
  concession that the M4.2e policies express as "respect"); Hold is 0.71 when defending and 0.85–0.95 while
  sieging or pressing. Apply Pressure is taken on 76–90% of the turns it is legal. **Recall is conditioned on HP**:
  0.00 at HP ≥ 80%, 0.25 at low HP when it is legal, 0.12 when four resources are unspent; it is never a gold
  decision alone (§8).
- **The freeze-like state is not held as a freeze.** With the enemy front at own −1 and the own front in the tower
  cell (2% of turns), the policy Clears (0.82) and stands Safe (0.52): it farms the wave as it arrives and
  destroys it, which the M4.2e-C threshold controller would hold for 20–30 turns with a 1.7× farm edge. Whether
  that is a mistake is a counterfactual question (§7).
- **Power gaps are almost never seen.** Fewer than 0.3% of self-play turns have an attack-power ratio outside
  [0.8, 1.25]: symmetric self-play keeps levels and investments equal, so the policy has essentially no experience
  of being ahead or behind, which the reference games expose it to (§6.2) and the mixed-start follow-up addresses
  (§9).

### 6.2 Against the references: the same policy, different games

| base final vs | P(Clear) | P(Thin) | P(Trade) | P(Safe) | P(Pressure) | deaths / recalls per game | own crashes per game | farm per game | reward per turn |
|---|---|---|---|---|---|---|---|---|---|
| itself | 0.75 | 0.13 | 0.27 | 0.14 | 0.03 | 7.0 / 13.7 | 39 | 165 | +0.05 |
| `push_punish` | 0.69 | 0.26 | 0.18 | 0.07 | **0.40** | **32.1 / 0.6** | **174** | 179 | +0.26 |
| `clear_punish` | 0.72 | 0.18 | 0.23 | 0.20 | 0.14 | 13.9 / 5.3 | 95 | 175 | +0.15 |

Against the slow pusher the policy's game changes completely without any change of weights: it crashes a wave
into the pusher's tower every 2.6 turns (174 crashes against 39 in self-play), applies Pressure on 40% of its turns,
**dies 32 times and recalls under once per 460 turns**, and still wins +0.26 per turn (the pusher dies 3 times
and recalls 30 times in self-play). It has found that against a punisher that trades every contact, the cheapest
reset is to die at the front after cashing in the crash: the bounty is 0.5, the lost farm of three absent turns is
about 1.3, and the walk back to the recall line would cost more than that in crashes forgone. Against the Clear line
the picture is in between (14 deaths, 5 recalls, 95 crashes). The policy is therefore responsive to the opponent in
the direction the economics point; §8 asks whether the response is right.

**Answer to question 2, first part.** The policies are not a constant action: the realized wave and behaviour
choices depend on the state (mutual information 6–20% of the action entropy; references 44–98%), the behaviour
choice is strongly conditioned (Trade at the lane, Safe and Hold at the tower, Recall at low HP, Pressure in a
siege), and the game against each opponent is recognisably different. The wave choice, however, is Clear in 60–90%
of every situation; the state-dependence of the wave dimension is a modulation of one dominant action, not a
selection among three. §7 asks whether that modulation tracks the value of the alternatives.

## 7. Counterfactual first moves: is Clear chosen because it is right?

### 7.1 Single-move deviations (`results/rl/m43b/counterfactual.json`, `report.md` §6, `fig_counterfactual.png`)

From 8 sampled reachable self-play states per situation (10 situations; the seed-0 final of each condition and
the other two baseline seeds), every legal first move is committed and the policy resumes; 16 paired rollouts per
candidate over 300 turns, discounted, the policy as the opponent. The quantity is Q(commit c) − Q(the policy's own
sampled move), mean over the 8 states ± 95% (states are the unit; the per-state standard errors are 0.1–0.4).

| base seed 0 | Hold | Thin | Clear | Hold+Trade | Clear+Trade | Hold+Safe | Recall | policy's modal move (π) |
|---|---|---|---|---|---|---|---|---|
| neutral lane | −0.05 ± 0.42 | −0.14 ± 0.30 | −0.07 ± 0.34 | −0.01 ± 0.24 | +0.01 ± 0.15 | −0.07 ± 0.25 | −0.28 ± 0.33 | Clear (0.76) |
| own front at own +1 | −0.23 ± 0.37 | −0.11 ± 0.30 | −0.16 ± 0.21 | −0.20 ± 0.34 | +0.01 ± 0.27 | +0.02 ± 0.24 | −0.16 | Clear (0.84) |
| enemy front at own −1 | +0.08 ± 0.26 | +0.11 ± 0.24 | −0.01 ± 0.14 | +0.13 ± 0.33 | +0.07 ± 0.35 | +0.34 ± 0.34 | +0.07 ± 0.20 | Clear (0.77) |
| own front stacked | +0.07 ± 0.38 | +0.15 ± 0.27 | +0.12 ± 0.21 | +0.05 ± 0.47 | −0.10 ± 0.43 | +0.13 ± 0.29 | −0.21 ± 0.54 | Clear (0.46) |
| defending (mass under own tower) | −0.02 ± 0.09 | −0.02 ± 0.09 | −0.04 ± 0.07 | 0.00 ± 0.21 | 0.00 ± 0.21 | −0.02 ± 0.09 | −0.55 ± 0.74 | Clear (0.97) |
| freeze-like | −0.37 ± 0.49 | −0.32 ± 0.61 | −0.49 ± 0.45 | −0.32 ± 0.58 | −0.15 ± 0.66 | −0.29 ± 0.65 | −0.86 ± 0.37 | Clear+Safe (0.75) |
| enemy absent | −0.51 ± 0.47 | −0.39 ± 0.42 | −0.25 ± 0.43 | −0.50 ± 0.48 | −0.25 ± 0.43 | −0.31 ± 0.56 | −0.18 | Clear (0.60) |
| low HP | −0.29 ± 0.36 | −0.06 ± 0.25 | +0.09 ± 0.17 | −0.38 ± 0.30 | −0.35 ± 0.27 | −0.11 ± 0.50 | −0.29 ± 0.36 | Clear (0.84) |

(The full tables for the five policies, with Yield, Clear+Safe and the Pressure variants, are in `report.md` §6.)

- **No single first move is worth more than a few tenths of a resource over the policy's own move, in any
  situation, for any of the five policies.** Over 50 (policy, situation) cells and about 500 (state, candidate)
  comparisons the per-candidate means lie within ±0.5 and their intervals include zero, with two consistent
  exceptions: committing to **Recall** where the policy would not (defending, freeze-like, enemy absent, a neutral
  lane at full HP) costs 0.3–1.1, and committing to anything but Clear **while the enemy is absent** costs 0.2–0.5
  (the policy's shove-while-away is right). At the freeze-like state, Hold (−0.37) and Thin (−0.32), the moves a
  freeze needs, are not better than the Clear + Safe the policy plays, given that the policy's own continuation
  will not hold the freeze.
- The policy's modal move is the best of the 8–11 candidates in 0–50% of states, with a mean "regret" of 0.1–0.7
  against the best; this is the expected magnitude of selecting the maximum of ten noisy estimates (about 1.5
  standard errors) and not evidence of a systematic error. The honest statement is that the policy's sampled move
  is within 0.1–0.5 discounted resources of every single-move alternative.
- **Therefore the policy is locally consistent**: its preferences are those its own continuation values, and the
  wave-action modulation of §6 tracks small, real differences (Clear is cheapest where the wave comes to the tower;
  Hold and Thin become competitive at low HP and with a stacked front, exactly where the policy raises them). What a
  single-move test cannot show is whether a *different strategy*, several coordinated turns long, would be worth
  more than the policy's play: a slow push needs a Thin edge followed by holding, a freeze needs the controller for
  twenty turns. §7.2 tests those directly.

### 7.2 Plan-level deviations: slow push, sustained Clear, pure last-hitting, a held freeze, shoving while away

`plans.py`, `results/rl/m43b/plans.json`, `report.md` §6b. The focal champion commits to a multi-turn plan and the
policy resumes when it ends; the opponent is the policy throughout; 8 sampled self-play states per situation, 16
paired rollouts, 300 turns discounted. Plans: **slow_push** = Thin for 2 turns then Hold for 12 (the M4.2e-C slow
push, which crashes a one-turn edge in about 11–15 turns against a Holding opponent); **clear** = Clear for 14
turns; **hold** = Hold for 14 turns; **freeze** = the M4.2e-C threshold controller for 20 turns (Thin while the
enemy front at own −1 holds ≥ 14 HP, else Hold); **shove_away** = Clear while the enemy champion is absent.
Q(plan) − Q(policy), mean over states ± 95%; in brackets the share of states with a significant loss.

| policy | situation | slow_push | clear | hold | freeze | shove_away |
|---|---|---|---|---|---|---|
| base_s0 | neutral lane | **−1.76 ± 0.49** (88%) | −0.93 ± 0.67 (50%) | **−1.84 ± 0.42** (88%) | | |
| base_s0 | own front at own +1 | −1.30 ± 0.81 (62%) | −0.51 ± 0.58 (50%) | −1.36 ± 0.96 (62%) | | |
| base_s0 | freeze-like | | −0.57 ± 0.67 (12%) | −1.73 ± 0.62 (75%) | **−2.39 ± 0.57** (88%) | |
| base_s0 | enemy absent | | | −1.73 ± 0.55 (75%) | | −0.10 ± 0.28 (0%) |
| ent03_s0 | neutral lane | −1.46 ± 0.45 (88%) | −0.72 ± 0.26 (50%) | −1.55 ± 0.37 (88%) | | |
| ent03_s0 | freeze-like | | −0.19 ± 0.30 (12%) | −1.19 ± 0.66 (62%) | −1.71 ± 0.61 (75%) | |
| hist80_s0 | neutral lane | −1.66 ± 0.72 (88%) | −0.21 ± 0.37 (12%) | −2.07 ± 0.48 (100%) | | |
| hist80_s0 | freeze-like | | −0.50 ± 0.46 (38%) | −2.32 ± 0.75 (100%) | −2.76 ± 0.79 (100%) | |
| mixed_s0 | neutral lane | −2.09 ± 0.67 (88%) | −0.52 ± 0.50 (38%) | −1.95 ± 0.56 (88%) | | |
| mixed_s0 | freeze-like | | −0.60 ± 0.48 (25%) | −1.57 ± 0.43 (75%) | −1.82 ± 0.66 (62%) | |

(Every (policy, situation) cell is in `report.md` §6b; the omitted rows have the same sign pattern.)

- **Against the learned policy, every plan that withholds wave damage loses, and loses consistently.** A slow push
  from the neutral lane costs 1.5–2.1 discounted resources (a significant loss in 88% of states for every policy),
  pure last-hitting 1.6–2.1, and the freeze controller from a freeze-like state 1.7–2.8 (62–100% of states). The
  loss is farm: the plan's own farm minus the policy's falls by 1.3–2.5 over the window while deaths are unchanged.
  Sustained Clear costs 0.2–0.9 (the policy's own play is Clear with Trade / Safe modulation, so a plain Clear plan
  is slightly worse); shoving while the enemy is absent is what the policy already does (−0.1 to −0.4, not
  significant).
- **Why.** The slow push and the freeze are wave-control strategies that pay against an opponent who lets the wave
  build: the M4.2e-C probes measured them against a Holder. Against a champion that Clears every turn, the Thin
  edge is removed the turn it is made, the follow-up waves are crashed into one's own tower, and the player who
  withheld damage has paid a wave of farm for nothing. The M4.2e-C policy games had already found that the
  stronger side's equilibrium line is Clear + punish and that against Clear + punish the weaker side's best reply
  is to contest and push back, never to stand back; the learned policies reproduce exactly that line at equal
  power. Clear dominance is a best reply to Clear, and a population of policies that all Clear has no member against
  which slow pushing or freezing pays.
- **This is a game property, not an exploration failure, with one qualification.** The four policies were trained
  under different exploration settings (entropy 0.01 and 0.03), a different opponent distribution (20% historical)
  and a different start distribution (half the games from freezes, sieges, pushes and absences), and all four value
  the alternatives the same way. The qualification is that the value is measured against the policy itself; a
  strategy that only pays against a *different* class of opponent (one that Holds, as the scripted references do:
  §6.2 shows the policy already pushes and dies freely against the slow pusher) would need that opponent in
  training to be discovered, which is what the M4.2e-C equilibria say about the equal-power lane (both sides
  slow-push in equilibrium there, with a value of +0.04). The learned population sits at the Clear-against-Clear
  line instead, and the counterfactuals say that no unilateral move to the slow-push line pays from there.

**Answer to question 2, second part.** The dominance of Clear is state-appropriate: at every sampled state, for
every policy, the policy's own move is within a few tenths of the best single deviation, and every multi-turn
alternative that withholds wave damage loses one to three resources against the policy. The modulation of §6 is
real but small because the alternatives are worth little; the policies have found the line the game's dynamics
reward against an opponent that clears.

## 8. Recall and death economics

`recall.py`, `results/rl/m43b/recall.json`, `report.md` §7, `fig_recall.png`.

### 8.1 The mechanics, stated

| | Recall | death |
|---|---|---|
| when | at or behind own −1, no enemy Trade reaching the champion (else interrupted) | anywhere, any time |
| turns lost | 2 (the Recall turn and one away) | the death turn's end and 3 away |
| returns with | full HP, all unspent gold invested | the same |
| pays | nothing | 0.5 resources of bounty to the opponent |

A death is a Recall from any position, uninterruptible, for one extra turn of absence and half a wave of bounty.
Whether the policies use it as such is an empirical question with two parts: how the policy prices the two, and
whether its prices are right.

### 8.2 How often (self-play, 12 games of 460 turns per final)

| final | Recall legal (share of present turns) | P(Recall \| legal) | chosen \| legal | recalls / game | deaths / game | returns after death / after Recall | gold invested after death / after Recall (per game) | unspent gold carried: at death / at Recall |
|---|---|---|---|---|---|---|---|---|
| base_s0 | 0.61 | 0.060 | 0.059 | 15.2 | 3.9 | 3.9 / 15.1 | 29 / 128 | 6.8 / 8.5 |
| base_s1 | 0.55 | 0.056 | 0.055 | 11.8 | 10.0 | 9.8 / 11.8 | 79 / 91 | 7.5 / 7.7 |
| base_s2 | 0.56 | 0.069 | 0.073 | 15.7 | 6.1 | 6.0 / 15.6 | 48 / 120 | 7.5 / 7.7 |
| ent03_s0 | 0.56 | 0.076 | 0.079 | 14.8 | 8.6 | 8.6 / 14.8 | 66 / 104 | 7.1 / 7.1 |
| hist80_s0 | 0.62 | 0.061 | 0.062 | 12.5 | 5.5 | 5.5 / 12.5 | 51 / 122 | 9.0 / 9.8 |

Recall is legal on well over half of the present turns (the champion is at or behind own −1 most of the time) and
taken on 6–8% of them: 12–16 Recalls per 460-turn game and side, two to four times the number of deaths. In
self-play, **Recall is the main investment route**: 55–80% of the gold invested at base arrives through a Recall,
with 7–10 resources carried on either route. The M4.3a finding of "Recall effectively absent" does not hold at this
budget (§3.1: it is discovered between 100k and 400k transitions).

### 8.3 Is the timing right? Counterfactual Recalls (W = 300, 24 paired rollouts, 8 states per class)

Q(Recall) − Q(the policy's own move), − Q(stay: Hold + Hold), − Q(walk back: Safe), at Recall-legal states of
self-play games, by class; mean over states ± 95%. The policy's own probability of Recall at those states is π.

| base_s0, class | mean HP | unspent gold | π(Recall) | Recall − policy | Recall − stay | Recall − Safe |
|---|---|---|---|---|---|---|
| after the own crash (own mass under the enemy tower) | 0.19 | 7.3 | **0.82** | +0.16 ± 0.17 | **+0.49 ± 0.26** | **+0.52 ± 0.25** |
| low HP (≤ 40%) | 0.23 | 7.3 | **0.56** | −0.04 ± 0.18 | −0.21 ± 0.49 | −0.14 ± 0.45 |
| ≥ 4 resources unspent | 0.63 | 5.9 | 0.17 | −0.11 ± 0.35 | +0.02 ± 0.75 | +0.04 ± 0.74 |
| enemy absent | 0.82 | 4.3 | 0.00 | −0.60 ± 0.46 | −0.22 ± 0.38 | −0.20 ± 0.30 |
| enemy wave at own −1 | 0.83 | 3.4 | 0.00 | −0.71 ± 0.49 | −0.57 ± 0.75 | −0.57 ± 0.71 |
| neutral lane, HP ≥ 80% | 0.94 | 4.7 | 0.00 | **−0.87 ± 0.21** | −0.53 ± 0.40 | −0.50 ± 0.30 |
| any Recall-legal state | 0.79 | 3.0 | 0.00 | −0.59 ± 0.25 | −0.48 ± 0.36 | −0.25 ± 0.18 |

The other four policies (base_s1, base_s2, ent03_s0, hist80_s0) give the same pattern (`report.md` §7.3): Recall
is worth +0.1 to +0.3 over the policy's own move and +0.1 to +0.6 over staying **after the own crash**, within
noise of the policy's move at **low HP** (where the policy already recalls with π = 0.1–0.6), and costs 0.3–0.9
everywhere else, including the cash-ready states (unspent gold ≥ 4: −0.0 to −0.6 against the policy, −0.1 to −0.3
against staying). A regression of Q(Recall) − Q(stay) on the state over all 56 sampled states per policy gives
−0.4 to −0.7 per unit of HP fraction (a Recall is worth about half a resource more at empty HP than at full), +0.4
to +0.6 after the own crash, and 0.00 to +0.05 per resource of unspent gold (R² 0.2–0.4): **the value of a Recall
is an HP and wave-state decision, not a gold decision**, which is what the policy has learned (P(Recall | legal) is
0.00 at HP ≥ 80%, 0.25–0.36 at low HP, 0.3–0.9 after the own crash). The M4.2b and M4.2c calibration finding that
item timing has no measurable value on this game is reproduced by the learned policy's indifference to its gold.

### 8.4 What a death costs, against what a Recall costs, from the same state

From 24 sampled self-play states per policy (both champions present), the focal champion's record is replaced by
"just died" (at its tower, HP 0, absent 3 turns) or "just recalled" (at its tower, absent 1 more turn), the lane
unchanged, and the three variants are rolled out on common random numbers (24 rollouts, 300 turns). Reported:
V(dead) − V(alive) and V(recalled) − V(alive) (downstream only), then the full costs with the terms that precede
the state added back (the 0.5 bounty; the Recall turn at the policy's farm per present turn, 0.36–0.40).

| base_s0 | n | V(dead) − V(alive) | V(recalled) − V(alive) | full cost of a death | full cost of a Recall | death − Recall |
|---|---|---|---|---|---|---|
| all states | 24 | −0.82 ± 0.26 | +0.34 ± 0.34 | 1.32 | 0.03 | **1.29** |
| HP ≤ 40% | 4 | −0.32 ± 0.44 | +1.00 ± 0.76 | 0.82 | −0.63 | 1.45 |
| 40% < HP ≤ 80% | 10 | −0.65 ± 0.35 | +0.50 ± 0.38 | 1.15 | −0.14 | 1.28 |
| HP > 80% | 10 | −1.18 ± 0.38 | −0.09 ± 0.56 | 1.68 | 0.45 | 1.23 |
| ≥ 4 resources unspent | 16 | −0.59 ± 0.28 | +0.53 ± 0.39 | 1.09 | −0.16 | 1.26 |

Across the five policies the full cost of a death is 1.1–1.3 resources and of a completed Recall 0.0–0.55, with
the death premium 0.8–1.3 (`report.md` §7.4). The scale is the lane's income (about 0.4 per present turn): a
death costs three turns of farm plus the bounty; a Recall, whose absence is one turn shorter and which returns a
full bar, costs about nothing on average and is a **gain** of 0.6 at low HP (the fresh bar is worth more than the
turn). Two readings:

- **Death is not cheap, but it is not expensive either.** One resource per death, against a per-turn income of 0.4
  and a per-game farm of 160–180, is a small item on the ledger: 6 deaths per game cost about 7 resources, 4% of
  the income. The kill bounty (0.5) is a third of the cost; the rest is absence. That is the economics the M4.3a
  report asked to audit, and it says that dying is roughly a bad Recall, not a disaster.
- **Death is not a substitute for Recall in self-play**: the policy recalls three times as often as it dies, its
  deaths are mostly ganks at high HP (the exported games show ganks at 115, 204, 275, 414, 639 centi-HP) rather than
  trades it stayed in, and the counterfactual values of §8.3 say its Recall timing is right. Against the punishing
  slow pusher (§6.2) the same policy dies 32 times and recalls once per game: there the deaths are trades, taken
  at the front while crashing waves and taking plates. §8.5 asks whether that is right.

### 8.5 Against the punisher: is dying at the front the right reset?

The same counterfactual Recalls, from Recall-legal states of games of `base_s0` **against** `push_punish` and
`clear_punish` (the reference is the opponent of the rollouts too; `report.md` §7.3b):

| opponent, class | mean HP | π(Recall) | Recall − policy | Recall − stay | Recall − Safe | focal deaths in 300 turns: policy / Recall | policy's farm lead in 300 turns |
|---|---|---|---|---|---|---|---|
| push_punish, low HP | 0.27 | 0.34 | +0.11 ± 0.42 | +0.36 ± 0.40 | −0.09 ± 0.52 | 19.6 / 19.5 | +46 |
| push_punish, after the own crash | 1.00 | 0.01 | −0.35 ± 0.20 | −0.34 ± 0.23 | −0.38 ± 0.26 | 19.5 / 19.3 | +46 |
| push_punish, any | 0.92 | 0.00 | −0.63 ± 0.34 | −0.50 ± 0.33 | −0.18 ± 0.17 | 19.4 / 19.2 | +47 |
| clear_punish, low HP | 0.25 | 0.48 | −0.13 ± 0.14 | −0.23 ± 0.74 | −0.20 ± 0.87 | 5.7 / 5.6 | +23 |
| clear_punish, after the own crash | 0.39 | 0.27 | +0.27 ± 0.31 | +0.45 ± 0.31 | +0.39 ± 0.42 | 5.7 / 5.7 | +23 |
| clear_punish, any | 0.86 | 0.12 | −0.99 ± 0.46 | −0.38 ± 0.63 | −0.33 ± 0.62 | 5.3 / 5.0 | +24 |

Against the punisher the policy dies about 19 times per 300 turns whether or not it Recalls now (a Recall buys
one return at full HP, after which it goes back to the front and the cycle resumes), it out-farms the punisher by
46 resources per 300 turns while doing so, and a Recall at low HP is worth +0.1 over its own move (+0.4 over
staying), i.e. about the death premium of §8.4 once per cycle and nothing more. Its "after the own crash" states
against the punisher are at full HP (it crashes a wave every two turns and has nothing to reset), and recalling
there costs 0.35. Against the Clear line the self-play pattern returns (Recall pays after the own crash, is even
at low HP). The 32 deaths per game against the punisher are therefore not a mispriced reset: each costs about one
resource, the alternative (walk back, Recall, return) costs about as much and forgoes a crash, and the machine
that produces +0.26 per turn keeps running. Deaths are the policy's reset against an opponent that trades every
contact, and the game prices that reset at roughly the Recall it replaces.

**Answer to question 3 for Recall.** The game does reward well-timed Recalls (after the own crash: +0.5 over
staying; at low HP: a fresh bar for a turn), does not reward gold-timed Recalls (the investment curve is too flat
for the timing to matter, as M4.2b/c found), and prices a death at about one resource above a Recall. PPO found
the Recall window the game rewards, uses it 12–16 times a game, and routes most of its investment through it. The
M4.3a absence of Recall was a budget-and-schedule artefact, not a game property and not a learning failure.

## 9. The mixed-start follow-up

**Motivation.** §6.1 found that self-play from s0 almost never visits power gaps, holds a freeze-like state on 2% of
turns and defends a crashed mass on 3%: the training distribution is narrow. The follow-up asks whether starting
half of the training games from a broad archive of reachable states changes what is learned, with the objective
unchanged (every evaluation here is protocol E1 from s0).

**The archive** (`archive.py`, `results/rl/m43b/archive_train.json`, `archive.json`): 857 witnessed states (429
distinct, closed under the seat mirror; every witness replayed and verified) sampled from 400 games of 460 turns
among the eight reference policies (24 random pairings, cycled), the untrained seed-0 network and the baseline
seed-0 final, at turns 12–400, at most three per game and class, stratified by situation with these targets, all
met except "own front in the enemy half", which on this lane is the same state as "own front next to the enemy
tower":

| class | neutral | enemy wave at own −1 | own front at own +1 | stacked front | defending | sieging | freeze-like | enemy absent | low HP | other |
|---|---|---|---|---|---|---|---|---|---|---|
| states (before mirroring) | 60 | 60 | 30 | 30 | 50 | 40 | 40 | 40 | 40 | 50 |

849 of the 857 states come from reference games and 8 from the untrained network (the class targets were filled
before the learned policy's games in the cycle contributed). The training distribution of the `mixed` runs is then
s0 with probability 0.5 and a uniform archive state otherwise: 1,018 / 1,047 / 987 of each run's 2,000 games
started from the archive (`time_semantics.starts` in the checkpoints). The archive is not used anywhere in
evaluation.

**Result: no change.** The three `mixed` finals are indistinguishable from the baseline on every measure:

| measure (mean over seeds) | base | mixed |
|---|---|---|
| headline vs `farm` / `push_punish` / `clear_punish` / `bully_shove` | +11.3 / +14.4 / +9.1 / +16.2 | +11.4 / +15.4 / +8.8 / +16.3 |
| vs `freeze_punish` / `respect` / `crash_recall` / `trading_pusher` | +15.1 / +10.5 / +10.9 / +18.2 | +16.0 / +10.9 / +10.1 / +17.7 |
| vs initial network | +12.1 | +11.8 |
| cross-play vs base / ent03 / hist80 (payoff to mixed) | — | +0.11 ± 0.31 / −0.28 ± 0.25 / −0.23 ± 0.24 |
| final vs own snapshot at 102k / 410k / 1,024k | +4.0 / +1.4 / +0.5 | +3.8 / +2.4 / +0.9 |
| wave shares Hold / Thin / Clear at 2,048k | 0.12 / 0.13 / 0.75 | 0.12 / 0.13 / 0.76 |
| behaviour Hold / Yield / Trade / Safe / Recall | 0.40 / 0.16 / 0.27 / 0.14 / 0.04 | 0.39 / 0.11 / 0.30 / 0.17 / 0.03 |
| deaths / recalls per game and side | 6.0 / 14.3 | 7.2 / 12.3 |
| I(wave; state) / H(wave) | 0.12 | 0.15 |
| seed SD of the reference values | 0.89 | 0.78 |

The diagnostics of §6–§8 on the `mixed` seed-0 final repeat the baseline's: conditional behaviour (Clear 0.76,
Trade 0.30, Safe 0.16, I(wave; state) / H = 0.14), single-move counterfactuals within ±0.4 of the policy's move
everywhere, plan-level losses of 1.2–2.1 for the slow push, the hold and the freeze (§7.2), Recall worth +0.78 over
staying after the own crash (π = 0.97 there) and −0.8 to −1.0 at a full-HP neutral lane (π = 0.00), a death priced
1.3 above a Recall.

Half of the training games starting in freezes, sieges, enemy crashes, stacked fronts, absences and low-HP states
produced the same Clear-dominant policy with the same value from s0, the same reproducibility and the same
learning curve. Two readings are consistent with this and the counterfactuals of §7 decide between them: either the
policy's preferences are already right in those states (so wider coverage had nothing to teach), or the
transitions from those states back into the ordinary lane are short enough that the s0 distribution already
contains their lessons. What it rules out is the hypothesis that the narrow self-play distribution is what keeps
the policy from holding freezes or slow-pushing: given ten times the exposure to those states, it learned the same
thing. The archive and the `mixed` condition stay in the driver for M4.3c (the E2 diagnostic protocol can evaluate
on a held-out archive if a question needs it).

## 10. Representative trajectories

`results/rl/m43b/trajectories/*.json` (`trajectories.json` is the index with event timelines); open them with
`streamlit run lanegame/viewer/app.py`, mode "Recorded trajectory". A recording is the integer chance seed and the
action list; the viewer replays it through the engine, so every tick, HP and ledger entry can be inspected. All
games are 460 turns from s0; "A" is the first named policy.

| recording | A vs B | payoff to A (undiscounted) | deaths A / B | recalls A / B | crashes A / B | A: Hold / Thin / Clear | A: Trade / Safe / Recall / Pressure |
|---|---|---|---|---|---|---|---|
| base_s0_selfplay_a | base_s0 vs itself | +0.7 | 2 / 3 | 17 / 15 | 30 / 35 | 0.11 / 0.09 / 0.81 | 0.18 / 0.18 / 0.04 / 0.03 |
| base_s0_selfplay_b | the same, another seed | +1.4 | 4 / 4 | 15 / 15 | 34 / 32 | 0.12 / 0.09 / 0.79 | 0.22 / 0.19 / 0.04 / 0.03 |
| base_s0_vs_push_punish | base_s0 vs the slow-push + punish reference | +56.8 | **32 / 0** | 1 / 8 | **181 / 0** | 0.12 / 0.43 / 0.45 | 0.08 / 0.12 / 0.01 / **0.39** |
| base_s0_vs_clear_punish | base_s0 vs the Clear + punish reference | +41.7 | 9 / 4 | 12 / 12 | 101 / 16 | 0.15 / 0.17 / 0.67 | 0.23 / 0.21 / 0.03 / 0.13 |
| base_s0_vs_crash_recall | base_s0 vs the crash-and-Recall reference | +43.8 | 5 / 0 | 11 / 6 | 178 / 1 | 0.26 / 0.39 / 0.35 | 0.12 / 0.21 / 0.03 / 0.32 |
| base_s0_vs_initial | base_s0 vs its untrained network | +40.9 | 10 / 4 | 9 / 20 | 132 / 12 | 0.20 / 0.26 / 0.54 | 0.17 / 0.20 / 0.02 / 0.21 |
| ent03_s0_selfplay | ent03_s0 vs itself | −6.0 | 10 / 4 | 15 / 20 | 39 / 42 | 0.17 / 0.21 / 0.62 | 0.32 / 0.10 / 0.05 / 0.02 |
| hist80_s0_selfplay | hist80_s0 vs itself | −1.5 | 6 / 5 | 11 / 10 | 28 / 31 | 0.11 / 0.19 / 0.70 | 0.23 / 0.25 / 0.03 / 0.02 |
| mixed_s0_selfplay | mixed_s0 vs itself | +5.2 | 11 / 10 | 15 / 13 | 50 / 52 | 0.18 / 0.13 / 0.69 | 0.39 / 0.16 / 0.04 / 0.03 |
| attacker_vs_base_s0 | the approximate best response vs base_s0 | +18.9 | 5 / 10 | 10 / 11 | 34 / 29 | 0.16 / 0.19 / 0.66 | 0.19 / **0.34** / 0.03 / 0.01 |

What to look at:

- **Self-play (`base_s0_selfplay_a`).** Both sides Clear almost every turn, so waves crash alternately into each
  tower every few turns (8 HP, then the 1–3 HP remnants) and plates fall at k91, k109, k115. Deaths are ganks at
  high HP (k60: A ganked at 115 centi-HP); Recalls come at k67, k72, k89, k107, k112 with 3–27 resources unspent,
  always right after the own crash or at low HP. The lane never settles: no freeze, no stack above 13 HP.
- **Against the slow pusher (`base_s0_vs_push_punish`).** From k4 the policy crashes a wave every two turns (181
  crashes), stands in B's tower cell with Pressure (39% of its turns), takes the three plates by k42, and dies 32
  times, almost all to trades at the front (k5 at 47 centi-HP, k18 at 99, k50 at 67, k61 at 5), returning each time
  with a fresh bar and its gold invested. B, which only trades what comes into reach, never crashes a wave and
  farms 110 against A's 180. This is the game the single policy plays when the opponent lets the wave build; it is
  the strongest reference-level result and the clearest illustration of death as the policy's reset.
- **Against the Clear line (`base_s0_vs_clear_punish`).** The reference out-Clears the policy (0.83 against
  0.67) but the policy crashes six times more often (101 against 16), takes all three plates, and wins +41.7
  while dying 9 times to ganks at the front (k12 at 41 centi-HP, k34 at 23). Watch k26–k36: four consecutive crashes
  and two plates while B is away on a Recall.
- **The attacker (`attacker_vs_base_s0`).** The exploit is positional: the attacker plays Safe on 34% of its turns
  (its target 10%), dies half as often (5 against 10), loses no plates, and converts the target's forward deaths
  (k63 a trade, k78 the tower) into +18.9 undiscounted. The target over-commits at the front against an opponent
  that declines contact; this is the one strategic hole the pilot documents, and it is small in discounted terms
  (+2.0 at W = 1000, §4.2).
- **The 0.03 control (`ent03_s0_selfplay`).** Visibly more varied: Thin 0.21, Trade 0.32, both sides die to ganks
  (A 10, B 4) and recall 15–20 times, the sides crash 39–42 times each. The game is bloodier and no better.

## 11. Answers to the three questions

**1. Is PPO actually learning stronger policies in the new simulator?** Yes. Twelve independently trained networks
beat every reference of the strongest scripted strategy space the project has by 9–20 discounted resources at
W = 1000 (about a fifth of the lane's discounted income), beat their own untrained network by 12, and keep beating
their own earlier snapshots in direct play across the whole budget (+4 against the 100k snapshot, +1.5 against
400k, +0.5 against 1M). The references stop registering progress after the first 100k transitions because they
are dominated, not because learning stops. A dedicated attacker with half the budget gains 0.75–2.0 against a
final, which bounds how far these policies are from unexploitable on this measure. The three training conditions
and the follow-up land within one resource of each other; the 0.03 entropy coefficient is the best of them on
every instrument (references, cross-play, exploitability, reproducibility) by a small margin.

**2. State-dependent decisions, or a small set of dominant actions?** Both, and the counterfactuals say why. The
behaviour dimension is clearly state-dependent (Trade at the lane and never when defending; Safe when the enemy
wave or its crashed mass reaches the own tower; Hold while defending or sieging; Pressure on 76–90% of the turns
it is legal; Recall at low HP and after the own crash, never at full HP). The wave dimension is Clear on 60–90% of
the turns in every situation, modulated towards Thin while sieging and towards Hold at low HP. Single-move
counterfactuals find the policy's own move within a few tenths of a resource of every alternative at every sampled
state, and plan-level counterfactuals find that every multi-turn alternative that withholds wave damage (a slow
push, pure last-hitting, a held freeze) loses 1–3 resources against the policy. The dominant action is dominant
because it is right against the opponent the policy faces; the modulation is small because the alternatives are
worth little. The policy is not a reflex: it plays a visibly different game against each reference (§6.2), and
the mutual information between its realized actions and the state is a sixth to a fifth of the action entropy.

**3. Where strategies are missing, is it the mechanics, the reward, or the training?**

- *Recall is not missing.* It was absent in the M4.3a smoke test because that run's learning rate had annealed to
  zero before the discovery; at a fifth of a 2M schedule every run discovers it, uses it 12–16 times a game, times
  it where the counterfactuals say it pays (after the own crash, at low HP) and not where it does not (gold alone),
  and routes most of its investment through it. A death is priced about one resource above a Recall by the game
  and by the policy. Training process, specifically the budget-relative schedule: an artefact, now understood.
- *Slow pushing, freezing and the stand-back concession are missing, and that is the game.* Under the M4.2e-C
  rules, against an opponent that Clears every turn, a Thin edge is removed the turn it is made and the player who
  withheld damage has paid a wave of farm for nothing; the plan-level counterfactuals measure the loss at 1.5–2.8
  resources for every policy, trained under two entropy coefficients, with and without historical opponents, from
  s0 and from an archive in which half the games start inside freezes, sieges, pushes and absences. Exploration,
  coverage and the opponent distribution were all varied and the answer did not move. The reward is the economic
  objective these strategies are supposed to serve, and under it they lose. What remains open is whether a
  *population* with a non-Clearing member would make them pay (they pay against the scripted Holders, which is why
  the policy pushes and crashes freely against those), which is a question about the game's equilibrium
  structure, not about PPO: the M4.2e-C policy games put both sides on a slow push at equal power, and the learned
  policies say that from the Clear-against-Clear line no unilateral deviation to that push pays. The two are
  different equilibria of different strategy spaces; the learned one is the one the full game's objective
  supports.
- *The design finding, stated with its evidence and not acted on.* The calibrated game rewards constant wave
  clearing with trades and resets. The mechanics that make this so are known from the M4.2e reports and visible in
  the trajectories: Clear's capacity (4 HP per turn at AP 1, the whole spawn rate; more at AP ≥ 2) removes any edge
  the opponent builds; a crash into the enemy tower costs the crasher nothing it would otherwise have kept (the
  defender collects it at 0.6, the crasher takes plates with Pressure); the danger of standing forward is a gank or
  a trade whose worst outcome, death, costs about one resource; and the kill bounty is half a wave. Three
  calibration levers are already listed for review in the earlier reports (the own-tower collection rate 0.6, the
  kill bounty 0.5, the gank severity 0.6 → 0.4 of M4.2b), and the M4.2e-B study found that a two-wave bounty is the
  one change that produced a stand-back concession. The recommendation (§14) is a design investigation that uses
  the plan-level counterfactual harness against the learned policies as its test, not a recalibration inside this
  milestone.

## 12. Limitations

1. **Budget and seeds.** 2M agent transitions per run and three seeds per condition. The seed spread of the baseline
   (SD 0.9 against the references, pair-level cross-play differences up to 0.6) is of the same size as the condition
   effects, so every condition comparison here is indicative; the ordering ent03 ≥ hist80 ≥ base is consistent
   across three instruments but rests on n = 3. Behaviour was still drifting slowly at 2M (Clear 0.78 → 0.75 over
   the second half); a longer run could move it further.
2. **References are scripted.** The reactive suite is the strongest strategy space the project has, but it is
   finite and deterministic; being 9–20 ahead of it is a statement about these policies and this suite, not about
   optimality. The exploitability numbers (1–2) are lower bounds from one attacker architecture and budget.
3. **Counterfactual windows.** Single-move and plan-level values are discounted sums over 300 turns (γ^300 = 0.05;
   the guaranteed truncation bound is 5.2, far larger than the effects measured, but the paired design cancels the
   shared tail), with the policy as the opponent and the continuation. They measure the value of a deviation
   *against this policy*, not against a best response, and a plan that only pays against other opponents would not
   show. With 8 states and 16 rollouts per candidate the per-cell resolution is about ±0.3; effects below that are
   invisible.
4. **Coverage of the situation classes.** Power gaps, stacked fronts and sieges are rare in self-play (0.3–1% of
   turns); their counterfactual states were sampled from the few that occur and may not represent the class.
5. **One policy class.** A 2 × 128 tanh MLP over the 88-entry observation, shared across seats, with the M3
   hyperparameters. A larger network, a recurrent one, or a different reward scale were not tried; the entropy
   coefficient is the only hyperparameter varied.
6. **The archive of the follow-up** is almost entirely reference-policy states (849 of 857) and does not stratify by
   power gap; the follow-up therefore tests coverage of lane situations, not of leads and deficits.
7. **The Recall / death values are policy-relative.** "What a death costs" is measured under this policy's play
   against itself; a different continuation (e.g. an opponent that punishes absences by pushing) would price it
   differently, and §6.2 shows the policy itself prices it differently against each opponent.
8. **The game is the M4.2e-C candidate**, with its documented caveats (the own-tower collection rate 0.6 and the
   kill bounty 0.5 are calibrated, not derived; the freeze band is 3 HP wide; no stand-back concession at the lane).
   Where this report attributes a behaviour to the game's economics, that attribution inherits those caveats.

## 13. Files, tests, reproduction

| file | content |
|---|---|
| `lanegame/experiments/rl/m43b/common.py` | conditions, budgets, paths, evaluation seeds, agent specifications, statistics |
| `references.py` | `ReactiveAgent` (a `zoning_natural.Policy` as an environment agent) and the reference suite |
| `train.py`, `curves.py`, `evaluate.py`, `exploit.py` | the training grid; snapshot evaluation and profiles; the headline evaluation; the continuing-time attackers |
| `behavior.py` | per-turn features, situations, conditional tables, state dependence, profiles, counterfactual first moves |
| `recall.py` | the Recall / death study |
| `archive.py` | the witnessed, stratified, mirror-closed training start archive of the follow-up |
| `trajectories.py` | viewer recordings |
| `figures.py`, `report.py`, `__main__.py` | figures, the results tables (`results/rl/m43b/report.md`), the stage CLI |
| `lanegame/experiments/rl/m43b_pilot.py`, `lanegame/__main__.py` | `python -m lanegame experiments-rl m43b_pilot` |
| `lanegame/viewer/session.py`, `app.py` | `load_recording`, the "Recorded trajectory" mode (no other viewer change) |
| `plans.py` | the plan-level counterfactuals (slow push, sustained Clear, pure Hold, the freeze controller, shove while away) |
| `tests/test_rl_m43b.py` | 11 tests: the reference agents (legality, determinism, agreement with the policies), features and situation masks, the return-to-go recursion, the state-dependence summary, the candidate set, the dead / recalled state variants and their return schedule, the Recall masks, viewer recordings replaying identically, the continuing-time attacker (frozen target, both seats, warm start), a one-update end-to-end run of the driver, the archive (witnesses verified, mirror-closed, usable as training starts), the plan definitions (legal, bounded, hand back when absent) |
| `results/rl/m43b/` | `train.json`, `curves.json`, `headline.json`, `exploit.json`, `behavior.json`, `counterfactual.json`, `plans.json`, `recall.json`, `archive.json`, `archive_train.json` (the start archive, 857 witnessed states), `trajectories.json`, `report.md`, the figures, `trajectories/*.json`, the logs |
| `results/rl/checkpoints/m43b/` (untracked) | `<cond>_s<seed>/{run.json, history.jsonl, snapshots/, final.pt}`, `attackers/` |

Tests at this commit: `tests/test_rl_m43b.py` (11) passes in about 30 s; the full suite passes (909 passed, 4 skipped, 2 expected failures; 7.5 min on an idle machine, 13 under the training load).

Reproduction (one workstation, 12 logical CPUs; times under the nine-process load):

```bash
python -m pytest tests/test_rl_m43b.py                                   # 11 tests, ~30 s
python -m lanegame.experiments.rl.m43b train --workers 9                 # 13 min
python -m lanegame.experiments.rl.m43b curves exploit headline behavior counterfactual plans recall trajectories figures report --workers 9
python -m lanegame.experiments.rl.m43b archive                           # the follow-up's start archive
python -m lanegame.experiments.rl.m43b train --conditions mixed --workers 3
python -m lanegame.experiments.rl.m43b curves headline behavior figures report --conditions base,ent03,hist80,mixed --workers 9
streamlit run lanegame/viewer/app.py                                     # mode "Recorded trajectory"
```

## 14. Recommendations for M4.3c

1. **Training defaults.** Entropy coefficient 0.03 (or a small sweep 0.02–0.05 at 3 seeds): it is the best condition
   on every instrument and the most reproducible. 2M transitions reaches the equilibrium line; one 8M run would show
   whether the slow drift of the second half (Clear 0.78 → 0.75) continues. Keep `PPOConfig.continuing` otherwise;
   do not change the reward.
2. **Evaluation.** Keep the reactive suite, cross-play, the snapshot-progress matchups and one attacker per final as
   the standard readout (all in the `m43b` driver, resumable). Add attackers trained against a *population* of
   finals and evaluated against all of them, so that seed-specific exploits (§4.2: transfer 0.2–0.7) are separated
   from exploits of the method. Headline numbers stay at W = 1000, E1.
3. **The 100/0 vs 80/20 comparison** should be run with a pool that spans a strategically different past (snapshots
   every 200k, pool 10, so that the 100k Clear-and-die policy and the 400k Recall policy are both in it), with
   exploitability as the primary measure; at the pilot's spacing the mixture bought nothing (§5.2).
4. **A design investigation of Clear dominance, with the counterfactual harness as the test** (§11, question 3).
   Candidate levers, each already motivated in an earlier report: kill bounty 0.5 → 1 and 2 (M4.2e-B: the one
   change that produced stand-back concessions), gank severity 0.6 → 0.4 (M4.2b), the own-tower collection rate
   0.6 → 0.4 (M4.2e-B/C), the Clear share of capacity. For each variant, train 3 seeds at 2M with the 0.03
   coefficient, then ask the two questions this milestone can now ask mechanically: does the plan-level loss of a
   slow push and of a held freeze against the learned policy shrink or change sign, and does the learned wave mix
   move. A variant is interesting only if the learned policies, not the scripted ones, change their line.
5. **Recall and death: no change.** The economics are coherent (a death costs about one resource more than a
   Recall; Recall pays after the own crash and at low HP) and the policies price them correctly. The kill bounty
   enters item 4 for the sake of Clear dominance, not Recall.
6. **Retire the mixed start** unless an archive with power gaps is built: lane-situation coverage was not the
   limiting factor (§9), but the policies have never seen a lead or a deficit (0.3% of turns), and their play at
   unequal power is untested. A cheaper check is evaluation at unequal skills or with a level or gold lead
   assigned at s0 (the `with_gap` states of the zoning harness), which the E2 protocol supports.
7. **Instruments to keep.** `behavior.conditional_tables` and `state_dependence`, the single-move and plan-level
   counterfactuals, the death-against-Recall variants and the viewer recordings answered this milestone's questions;
   they should be the standard diagnostics of every later training milestone.
