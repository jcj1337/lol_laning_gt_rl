# Research overview: what the final campaign found

*Written on 2026-10-22 (M6d) for readers new to the project. It summarizes the final training campaign, M5, and adds
no new analysis. Every result below comes from the M5 records, and `python -m experiments.m6d overview` checks each
one against them. The full account is in the [M5 final report](archive/m5/m5e_closeout_report.md).*

## 1. The question

The project models the laning phase of League of Legends as a two-player game and trains agents on it by self-play.
No strategy is coded in. The only reward is gold. An agent gains what it collects and loses what its opponent
collects. Earlier milestones showed that self-play agents learn recognisable laning behaviour this way, such as
freezing the wave, picking trades and timing trips back to base.

M5, the final campaign, asked three questions, fixed in its
[pre-registration](archive/m5/m5_preregistration.md) before any training:

1. Do the strategic behaviours found earlier still hold in the final agents?
2. What does a model of the opponent add, compared with more plain self-play?
3. Does that model cost anything against strong opponents, and how far can each kind of agent be exploited?

## 2. The game and its score

Two champions share one lane. Each turn both choose one of 61 actions at the same time. An action says what to do
with the minion wave, how to behave towards the other champion, and whether to hit the tower, place a ward or go
back to base. The game runs without a fixed end. The laning phase ends around turn 94, when the champions reach level 14. The
[mathematical reference](model.md) gives every rule.

**The score.** Every result is a gold lead, meaning how much more gold one side has earned than the other after 114
turns from the standard opening. That number estimates the lead at the end of the laning phase (turn 94) plus how
that lead is expected to change over the next 20 turns, which is the readout the project calls E_20
(reference §9.3). Levels, health and the wave are reported beside it and never folded in.

**What one unit of gold means.** The game counts gold in its own unit (reference §5.1, §6.5, §6.7, §8.3):

- one fresh wave of minions, fully collected, is worth 1;
- a tower plate is worth 1, and a kill is worth 2;
- a death costs about 3.5 to 4 in all, of which 2 is the opponent's kill gold;
- a self-play agent earns about 0.39 a turn, about 37 by the end of the laning phase.

So a lead of 9 is about nine waves of minions, and a lead of half a unit is about half a wave.

## 3. What was trained

We trained two kinds of agent, the two **arms**, both continuing from the same six starting points trained in the
previous milestone. Pair s means the two runs that start from the same starting point s and use the same training seed.

- **Opponent modelling (om).** This arm sees the opponent's previous action and a running Bayesian estimate of which
  of eleven known play styles the opponent resembles, with two extra entries for "mixed" and "unknown". Half of its
  training games were self-play, and half were against 15 fixed opponents: scripted bots, earlier learned policies,
  and mixtures and switchers of them.
- **Pure self-play (sp).** This arm trained only against the current version of itself.

There were twelve runs, six per arm, of 32,768,000 agent decisions each. The two arms differ in what the agent sees
and in whom it trains against, so M5 measures the effect of the whole package. It cannot say which of the two changes
caused a given difference.

## 4. How it was measured

Each final agent played many paired games against five groups of opponents, and the result for a pair is the om
agent's lead minus the sp agent's lead against the same opponents.

| group | what it contains |
|---|---|
| library styles | the six scripted bots in the om arm's training library: passive farming, safe play, bullying, always pushing, freezing, and pushing with wards |
| held-out styles | seven styles neither arm met in training: four scripted variants of the library bots, and three conditional styles built on a competent learned policy |
| misspecified | seven opponents that mix or switch styles, including some that copy a library bot and then switch to a competent policy |
| competent historical | ten competent learned policies from before M5 |
| competent finals | the other five pairs' final agents of both arms |

The agents also played each other, om against sp from the same pair. Every opponent was played in 64 pairs of games
with the seats swapped, and the two arms of a pair met in 128 such pairs.

**Statistics.** The six pairs are the units. Each result is a mean over the six pairs with a 95% t interval, and the
per-pair values are given where the sign matters. The pre-registration fixed the decision rules. A gain counts when
its mean is at least +0.25 and its interval lies above zero. A loss of competence counts as material when the
interval lies below zero and the mean is −0.25 or worse.

**Attackers.** For each of the twelve final agents, a separate attacker started as a copy of it and trained for
8,192,000 decisions to beat it while the target stayed frozen. The attacker's lead against its target is a lower
bound on how far that agent can be exploited, since a stronger attack might do better.

**Strategic behaviour.** The battery restores game states, natural and constructed, and compares each agent's own
play with fixed alternative plans from those states. A behaviour holds in a pair when no fixed alternative beats the agent's play by
more than 0.2 gold over 50 turns, in any of its families of states. It replicates when it holds in at least five of
the six pairs.

## 5. Results

### 5.1 The gain against scripted styles

Against scripted styles, the om agents built much larger leads. The registered measure averages the library and
held-out groups. The gain was **+9.26 [+8.95, +9.56]**, positive in all six pairs.

| group | om lead | sp lead | om − sp |
|---|---|---|---|
| library styles | +16.87 | +4.72 | +12.15 [+11.77, +12.54] |
| held-out styles | +10.62 | +4.26 | +6.36 [+5.80, +6.92] |
| misspecified | +9.59 | +4.39 | +5.20 [+4.82, +5.59] |

The held-out group splits in two:

- Against the four scripted variants the gain averaged +11.39, close to the library's. Three of them gave +14.33,
  +14.42 and +13.12, and the fourth +3.70.
- Against the three conditional styles built on a competent policy the differences were −0.18, −0.67 and −0.20, so
  there was no gain.

The gain is therefore largest against opponents that resemble the training library. The misspecified group mixes
and switches styles, so the om arm's estimate cannot describe it exactly. The om agents still gained there, and the
registered test that the gain was preserved passed.

### 5.2 The cost against competent opponents

Against competent opponents the om agents did worse than the sp agents, in every pair.

| comparison | om − sp | pairs below zero |
|---|---|---|
| competent opponents (historical and finals) | −0.63 [−0.98, −0.27] | 6 of 6 |
| head to head, om against sp of the same pair | −0.44 [−0.72, −0.16] | 6 of 6 |
| competent opponents, 500-turn games | −0.60 [−0.85, −0.35] | 6 of 6 |
| head to head, 500-turn games | −0.90 [−1.33, −0.48] | 6 of 6 |
| head to head, om against the other pairs' sp | −0.68 [−0.96, −0.39] | 6 of 6 |

Each of these meets the registered rule for a material loss. Against the competent historical policies, the om
agents' lead was +0.70 and the sp agents' +1.35. Against the competent finals, the om agents' lead was −0.31 and the
sp agents' +0.29.

The pre-registration names this combination outcome 2, "pays at a competence cost". The opponent model **pays
against scripted styles and costs competence** against competent opponents.

### 5.3 How far each arm can be exploited

Every attacker beat its target, in every pair:

| target | attacker's lead (lower bound) |
|---|---|
| om final agents | +2.96 [+0.32, +5.61] |
| sp final agents | +1.15 [+0.34, +1.96] |

The difference between the two bounds, om minus sp, is **+1.81 [−0.56, +4.19]**. That is inconclusive. M5 shows
neither that om is more exploitable nor that the two are equally exploitable. Most of the spread comes from one om
pair, pair 3, whose attacker led by +7.91. The attacks did not carry over. Against any opponent other than its own
target, an attacker did no better than the agent it started from (final report §5.3).

### 5.4 Strategic behaviour

| behaviour | om pairs holding | sp pairs holding | verdict |
|---|---|---|---|
| wave management that depends on the situation | 6 of 6 | 6 of 6 | replicates in both |
| freezing when it pays, and breaking the opponent's freeze | 6 of 6 | 6 of 6 | replicates in both |
| choosing when to trade | 6 of 6 | 6 of 6 | replicates in both |
| Recall timing | 6 of 6 | 6 of 6 | replicates in both |
| giving ground when weaker and threatened | 3 of 6 | 4 of 6 | replicates in neither |

Two details matter for reading this table:

- **Giving ground fails on different evidence in each arm.** The om agents fail in pairs 0, 3 and 4 on the registered
  50-turn measure. The sp agents fail only in pairs 4 and 5, and only on a 150-turn re-run. Those two families failed
  the battery's horizon check, so the registered rule reads them at 150 turns, where the best alternative beat
  the agent's play by +0.212 and +0.329. At 50 turns the margins were only +0.058 and +0.045, so on the 50-turn
  measure pure self-play holds in all six pairs.
- **Freezing holds narrowly in pure self-play.** In sp pairs 0, 1 and 2 the best alternative comes within 0.014,
  0.015 and 0.034 of the 0.2 threshold.

## 6. The two arms side by side

| | opponent modelling (om) | pure self-play (sp) |
|---|---|---|
| gain against scripted styles, om − sp | +9.26 [+8.95, +9.56] | |
| against competent opponents, om − sp | −0.63 [−0.98, −0.27] | |
| head to head, om's lead | −0.44 [−0.72, −0.16] | |
| attacker's lead against it | +2.96 [+0.32, +5.61] | +1.15 [+0.34, +1.96] |
| strategic behaviours that replicate | four of five | four of five |
| still improving when training stopped | not shown | yes |

Neither arm is better overall. The om package gains a large lead against scripted styles like its library and pays
for it against competent play. Both arms are exploitable by a dedicated attacker. The sp arm was still improving at
the end. Its final agents led their own 75% checkpoints by +0.59 [+0.14, +1.05], while the om agents' difference was
−0.08 [−0.40, +0.24].

## 7. Limitations

From the final report's §11:

- **A package comparison.** M5 cannot attribute the competence cost to the opponent estimate, to the fixed-opponent
  games or to the smaller share of self-play.
- **Six pairs, from one lineage.** The training continued earlier runs, the om arm used one opponent library and one
  mix, and the sp arm had not converged.
- **Style groups that favour the library.** The gain is measured mostly against scripted styles the library
  resembles, and the competent opponents come from the same lineage as both arms.
- **Approximate attackers.** Each attacker is one learned best response at a quarter of the training budget, so its
  lead is a lower bound. Two bounds can differ for reasons that belong to the attackers. The attackers cannot test
  deliberate deception of the opponent estimate.
- **The battery's statistic** takes the best of several alternatives, which pushes each gap upward, and its
  constructed states are ones the agents rarely visit.
- **Coverage.** Nothing was measured beyond 500 turns, and M5 never evaluated play against people.

## 8. What the demo shows

The [web demo](../web/README.md) uses the pair-0 finals, `om:0` and `sp:0`. Pair 0 was chosen by its index, as in
every M5 example, and not for its results.

- A single game, whether a recorded one, a new simulation or a visitor's own game, is an example. It shows how the
  agents play and measures nothing.
- A match between the two agents shows how they play, not which is better.
- A game against a person shows how the agents respond to that person. M5 never evaluated play against people.
- The gold lead the demo shows is the earned-gold lead, not a score or a chance of winning.
- When the om agent's estimate is shown, a player can try to steer it on purpose. M5 never tested deliberate
  deception.
- The sequences in Watch, such as "Stacking to Crashing", explain one stretch of one game. They do not show how often
  that stretch happens.

## 9. Corrections to earlier wording

A repository audit on 2026-10-22 ([report](archive/audit/repository_audit_2026-10-22.md)) found four statements in the M5
reports and summaries that this overview words differently. The reports stay as written.

- "+6.4 against held-out scripted variants" is the mean of the whole held-out group. The scripted variants alone
  average +11.39, and the conditional styles show no gain (§5.1).
- "Concession under threat does not replicate in either arm" is right under the registered rule, but pure self-play
  fails only on the 150-turn re-run of two pairs (§5.4).
- "Three seeds sit within 0.03 of the margin" is not quite right, since one of them is 0.034 away (§5.4).
- "om about +0.25 against the start, an interval that includes 0" describes the om arm's gain over its starting
  point against the competent groups. The value is +0.24, and the interval for the competent finals alone,
  [+0.004, +0.55], does not include 0.

## 10. Evidence

- [M5 pre-registration](archive/m5/m5_preregistration.md): the questions, the design and every decision rule.
- [M5 final report](archive/m5/m5e_closeout_report.md): the full results, the limitations (§11) and the further research
  it proposes (§14).
- `artifacts/m5/analysis_main.json`: every endpoint, per pair, with its interval.
- `artifacts/m5/strategic_main.json` and `artifacts/m5/battery/`: the strategic battery.
- `artifacts/m5/monitoring_main.json`: the training checks, including whether each arm was still improving.
- [artifacts/m5/reproducibility.md](../artifacts/m5/reproducibility.md): the commands, seeds and checksums.
