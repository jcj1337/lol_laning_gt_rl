# M5e: the final M5 report

*2026-10-19. Status: **M5 complete.** Every required registered deliverable exists: training and its health checks, the M5d endpoints, the strategic battery and limitation cells (S8), the 12 core attackers and their independent evaluation (S6), the completed secondary Holm family, this report and the reproducibility materials. The optional om_table adaptation ablation was not run. M6, presentation, demonstration and reproducibility, is next.*

- *Protocol:* [m5_preregistration.md](m5_preregistration.md), version 1 (amended once before any training,
  [m5_amendment_1.md](m5_amendment_1.md)). M5e made no amendment.
- *Earlier reports (historical, unchanged):* [m5b_initial_report.md](m5b_initial_report.md) (training and the M5d
  families), [m5c_strategic_report.md](m5c_strategic_report.md) (battery and limitation cells).
- *Design:* [m5_design.md](m5_design.md). *Readiness:* [m5a_readiness_report.md](m5a_readiness_report.md).
- *Generated tables:* `results/m5/report.md`. *Reproducibility:* `results/m5/reproducibility.md` (commands, seeds,
  digests, the artifact inventory).
- *Records:* `results/m5/analysis_main.json` (every endpoint and the Holm family), `attack_check_main.json` (attacker
  health and identities), `strategic_main.json` (S8), `monitoring_main.json`, `train_main.json`,
  `attack_behaviour.json` and `traces.json` (exploratory), `corrections_readiness.json` part C1 (attacker readiness).
- *Decision trail of this stage:* `results/m5/decisions_m5e.tsv`.

## Summary

M5 trained two continuations of the same policy in six matched pairs and compared them as whole packages. Arm A
(`om`) is OM2's opponent-modelling package: belief and previous-action inputs, and half of its games against 15 fixed
opponents. Arm B (`sp`) is pure self-play.

- **Exploitation gain (registered, demonstrated).** The package beats library-like scripted opponents by +9.26 at the
  lane exit (95% interval [+8.95, +9.56], all six seeds). It stays robust to misspecified mixtures and switchers
  (S1 preserved, +5.20).
- **Competence cost (registered, material).** Against competent opponents the package is worse: −0.63 [−0.98, −0.27]
  against historical and other-arm finals, and −0.44 [−0.72, −0.16] head to head, in every seed. The long-horizon,
  cross-seed and midpoint versions (S2, S3, S4, S5b) are material losses too. Pure self-play kept improving over the
  shared start; the package did not.
- **Pre-registered outcome 2: the package pays at a competence cost.** This report gives the two side by side and
  never combines them into one score.
- **Strategic findings (S8).** Both arms retain four of the five M4.5c strategic findings: conditional wave
  management, conditional freezing and the reply to freezes, selective trading, and Recall timing. Concession under
  threat does not replicate in either arm. Retaining the same registered findings does not mean the arms make the same
  choices. They differ in what they achieve against competent play. The om zoning gap is open (suggestive), not
  missed (§4.2).
- **Attackers (S6).** Dedicated attackers found profitable deviations against both arms. Their lower bounds on
  exploitability at the lane exit are +2.96 [+0.32, +5.61] against the om finals and +1.15 [+0.34, +1.96] against the
  sp finals, both positive in all six seeds. The registered contrast, om − sp = +1.81 [−0.56, +4.19], is
  **inconclusive**: the package is not shown to be more exploitable, and equivalence is not shown either. Most of the
  spread comes from one om seed (+7.91). The attacks do not transfer. Against the other arm's final and against the
  common start every attacker does worse than its own starting point.
- **The completed Holm family** (15 hypotheses, final) keeps every earlier claim (S1 preserved; S2, S3, S4, S5b
  material losses; S5a demonstrated) and adds S6 inconclusive.
- **Recommendation for M6.** Present both arms as a trade-off, not a single winner, with the modelling package for
  recognizable, exploitable opponent styles and pure self-play for stronger performance against competent play. M5
  compares packages. It does not say whether the competence cost comes from the belief, the opponent mix or the
  reduced self-play exposure.

| result | value (6 seeds, 95% t interval) | status |
|---|---|---|
| P1 exploitation gain | +9.26 [+8.95, +9.56] | demonstrated (registered) |
| P2a competence against competent opponents | −0.63 [−0.98, −0.27] | material loss (registered) |
| P2b competence head to head | −0.44 [−0.72, −0.16] | material loss (registered) |
| S6 exploitability, om − sp | +1.81 [−0.56, +4.19] | inconclusive (registered, Holm final) |
| attacker lower bound against om | +2.96 [+0.32, +5.61] | descriptive |
| attacker lower bound against sp | +1.15 [+0.34, +1.96] | descriptive |
| strategic findings retained | 4 of 5 in each arm | registered rule (S8) |

## 1. What was trained and compared

M5 compares two ways of continuing the same policy. Both arms start from `ext/baseline:s`, the M4.6c 8.192M
continuation of the M4.5c 16.384M final (24.576M prior transitions), in six matched pairs s = 0 to 5. Each run adds
32,768,000 agent transitions with PPO seed 70,000 + s.

| | arm A, `om` (the opponent-modelling package) | arm B, `sp` (pure self-play) |
|---|---|---|
| environment | `m45_calibrated(obs_last_action=True)` | `m45_calibrated()` |
| network | 116 inputs: the state, the opponent's previous action, the 13-entry belief | 90 inputs, a copy of the start |
| games | 50% current-policy self-play, 50% one of OM2's 15 fixed opponents | 100% current-policy self-play |
| shared | the M4.6c start mixture, PPO settings and schedule; a fresh Adam annealed to 0 | |

**The contrast A − B is a package effect.** The belief inputs, the previous-action input and the opponent
distribution change together. Arm A also gets about a third fewer self-play transitions (21.8M against 32.8M). M5
therefore cannot say whether a difference between the arms comes from the belief information, from the opponent
distribution or from the reduced self-play exposure. Read every competence result below as a package effect.

## 2. Training health and the learning curves

- **Health.** All 12 runs completed and passed 138/138 registered checks. No runtime alarm, no technical failure and no
  restart occurred. Wall time was 10.90 h against an estimate of 9.4 to 11.8 h.
- **Convergence probe.** The probe is the final's lead over its own 75% snapshot at turn 114, over six seeds.
  - The sp finals lead by +0.59, 95% interval [+0.14, +1.05]. The registered rule calls that "still improving", and
    non-convergence is a registered limitation.
  - The om finals lead by −0.08, interval [−0.40, +0.24], so they are not shown to be still improving.
- **Against the common start** (history games, exploratory). The sp finals lead `ext/baseline:s` by +0.94 ± 0.54
  and the om finals by +0.20 ± 0.68. On the competent families (M5d, exploratory) sp gained about +0.87 over the start and om about
  +0.25, an interval that includes 0.
- **What the curves establish.** Pure self-play kept improving over the shared start through the whole budget. The
  package did not show an improvement over the start against competent play. Most of the competence gap below is
  therefore improvement that sp gained and om did not. It is not shown to be a fall below the start.

## 3. The registered endpoints

Seeds are the units. Each value is a mean over six seeds with a 95% t interval (df 5). The tables give the
game-level sampling uncertainty as two separate numbers. The mean game SE is the average standard error of a seed's
value from its games. τ is the between-seed spread that remains after removing it (τ² = max(0, s² − mean SE²)).

### 3.1 Primary endpoints and the headline

| endpoint | om − sp | 95% interval | per seed | τ / mean game SE | class |
|---|---|---|---|---|---|
| P1 exploitation gain (½ S_in + ½ S_out) | +9.26 | [+8.95, +9.56] | +8.94 +9.25 +9.32 +9.47 +9.64 +8.92 | 0.25 / 0.14 | demonstrated |
| P2a competence, competent opponents (½ C_hist + ½ C_cross) | −0.63 | [−0.98, −0.27] | −0.79 −0.70 −0.38 −0.46 −0.24 −1.17 | 0.31 / 0.13 | material loss |
| P2b competence, head to head (om:s vs sp:s) | −0.44 | [−0.72, −0.16] | −0.79 −0.08 −0.52 −0.27 −0.30 −0.67 | 0.02 / 0.27 | material loss |

**The headline is pre-registered outcome 2, "pays at a competence cost".** It covers P1, P2a and P2b only. This
report gives the gain and the cost side by side and never nets them.

- The gain comes from library-like scripted styles. It is +12.2 against the archetypes the library contains and
  +6.4 against held-out scripted variants. Against conditional styles built on a competent policy it is zero or slightly
  negative (−0.2 to −0.7).
- P2b's spread between seeds is almost all game sampling (τ 0.02 against a mean game SE of 0.27). P2a's is mostly
  seed to seed (τ 0.31 against 0.13).

### 3.2 The secondary family, now complete

S6 supplied the last three p-values, so Holm is now final over all 15 hypotheses at familywise 0.05. Every
one-sided test is a seed-level t test with df 5, fixed by amendment 1 before training.

| endpoint | om − sp | 95% interval | unadjusted interval class | claim after Holm | supporting p | M5b interim status |
|---|---|---|---|---|---|---|
| S1 misspecified opponents | +5.20 | [+4.82, +5.59] | preserved | **preserved** | 1.5·10⁻⁷ (preserved) | secured |
| S2 long-horizon competence (discounted) | −0.60 | [−0.85, −0.35] | material loss | **material loss** | 0.0008 (loss) | secured |
| S3 long-horizon head to head (discounted) | −0.90 | [−1.33, −0.48] | material loss | **material loss** | 0.0014 (loss) | secured |
| S4 cross-seed head to head | −0.68 | [−0.96, −0.39] | material loss | **material loss** | 0.0009 (loss) | secured |
| S5a midpoint exploitation gain | +8.29 | [+7.48, +9.11] | demonstrated | **demonstrated** | 7.6·10⁻⁷ (gain) | secured |
| S5b midpoint competence, historical opponents | −0.80 | [−1.20, −0.40] | material loss | **material loss** | 0.0018 (loss) | secured |
| S6 exploitability (attacker lead at 114 against its own target) | +1.81 | [−0.56, +4.19] | inconclusive | **inconclusive** | increase 0.053, decrease 0.95, equivalence 0.92 | pending |

- **The earlier claims are unchanged.** M5b had marked them "secured": each had p ≤ 0.05/15, which Holm rejects
  whatever the pending p-values turn out to be. The final family rejects exactly those six hypotheses and nothing
  else.
- **S6 makes no claim.** No S6 hypothesis is rejected, even before adjustment. The increase test misses at
  p = 0.053, the interval rule for an increase needs L > 0, and the equivalence rule needs the whole interval inside
  ±0.25. This is a registered inconclusive result. It is not evidence that the arms are equally exploitable.
- **Seed and game uncertainty.** S6's between-seed spread is almost all seed to seed: τ 2.22 against a mean game SE of
  0.42 per seed value. More evaluation games would not resolve it. The six attackers themselves differ.

## 4. Strategic evaluation (S8)

### 4.1 Replication

Four of the five M4.5c verdicts replicate in both arms (6 of 6 seeds each, 5 needed): conditional wave management,
conditional freezing and the reply to freezes, selective trading, and Recall timing. Concession under threat does not
replicate in either arm (om 3 of 6, sp 4 of 6). The weaker-side lead cells hold in all twelve finals. The failure
comes only from the constructed zoning family `H_zone_c`. The sp freezing verdict replicates but is fragile. Three
seeds sit within 0.03 of the 0.2 margin, and the pure-rollout G50 reading would fail it.

**Both arms retain the same four registered strategic findings. That is not a claim that they make the same
choices.** The M5c report's §8 says "the om policies still make the same strategic choices" and "differs in how much
it gains from those choices, not in which choices it makes". The evidence does not support that wording:
- A verdict holds when no fixed alternative beats the policy by more than 0.2 in its families. Two policies can pass
  the same criteria with different actions, and neither the battery nor any M5 instrument compares the two arms'
  action distributions state by state.
- The arms differ in what they achieve against competent opponents (P2a, P2b, and every cross-arm limitation family).

The supported statement is that both arms retain four registered strategic findings and differ in achieved returns
against competent opponents.

### 4.2 Constructed zoning is open, not missed

The M5c report's goal table (§7, G9) labels the om arm "**missed** in constructed zoning states". The M4 closeout
defines "missed" as "a tested alternative beats the policy, confirmed", and its own usage reads "confirmed" as an
interval that excludes 0. The om evidence does not meet that rule:

- The best `H_zone_c` alternative for om, a one-turn Shove, beats the policy by **+0.168** pooled over seeds, 95%
  t interval **[−0.006, +0.343]** (df 5; per seed +0.28 +0.06 +0.02 +0.23 +0.42 +0.01).
- The interval includes 0, so the alternative is not confirmed.
- It is the largest of nine alternatives, chosen after the fact, which biases it upward.
- It is positive in all six seeds. That makes it suggestive, not confirmed.

The corrected label is **open (suggestive)**. The registered result is unchanged, and concession under threat still
does not replicate under the registered criterion. A failure to replicate means that some seeds exceeded the 0.2 margin. It is
not the same claim as a confirmed exploitable mistake. For sp the M5c label was already "the zoning verdict fails on
2 seeds", with a pooled best alternative of +0.03.

### 4.3 The cold-start caveat

The cross-arm limitation cells start from constructed states or from the common start's natural states. No om game
produced those states, so the om final enters each cell with the prior belief and no previous action. Two pieces of
evidence weigh against a pure cold-start artefact, and neither removes it:
- The deficit grows from 50 to 150 turns. That does not by itself exclude an early cold-start effect that is then
  carried forward, because a worse start can compound.
- The independent evidence is the M5d games from s0, where both arms start naturally and the om final carries its own
  belief from turn 0. They show the same competence loss (P2a −0.63, P2b −0.44, S4 −0.68).

Read the size of the cross-arm cell deficits (−0.26 to −0.54) with the caveat in mind. The competence conclusion
comes from the s0 evaluations.

## 5. The dedicated attackers (S6)

### 5.1 What ran

- **The design**, as registered (§8) and checked before launch (readiness part C1, 56 of 56):
  - 12 attackers, six against the om finals with their belief live and six against the sp finals;
  - each starts as a copy of its target in the 116-input om format (an sp target widened with zero weights);
  - 8,192,000 agent transitions each, a quarter of a target's M5 continuation budget;
  - PPO seed 70,600 + 10·kind + s, s0 starts, 512-turn bootstrapped games, the M4.6c PPO settings;
  - the plain zero-sum gold-difference reward, γ 0.99, with no deception term.
- **What the attacker sees.** During play it sees the game state, the target's previous action and its own OM2
  belief about the target. It never reads the target's weights, belief or logits. The procedure as a whole is not
  black-box, because every attacker starts from its target's weights.
- **The om targets adapt as in deployment.** An om target updates its belief from the attacker's actions with the
  deployed rule. The audit's replay matched the deployed policy's belief to within 2·10⁻⁵ in log posterior, which is
  float32 batching.
- **Health.** All 12 runs completed with 180 of 180 checks:
  - exactly 8,192,000 transitions and 16,000 games each;
  - finite statistics and no illegal action;
  - minimum entropy 1.22 nats or more, the largest 10-update KL 0.0072, the largest gradient norm 1.8;
  - the target's in-memory weights equal to its file before and after training, so every target stayed frozen;
  - both beliefs updated on every turn, and the learner's seats within 2% of even.
  No runtime alarm, technical failure or restart occurred.
- **Evaluation.** Each attacker played its own target, the other arm's final of its seed and the common start
  `ext/baseline:s`: 128 paired-seat pairs × 500 turns per cell, on seeds disjoint from training and from every other
  M5 cell.
- **Runtime.** Training took 5.96 h (estimate 6.64 h): 3.3 h per om-target attacker and 2.6 h per sp-target
  attacker. Evaluation took 0.52 h (estimate 0.50 h).

### 5.2 Exploitability lower bounds

The attackers optimize the discounted continuing return of 512-turn games. The registered metric is different. It
is the attacker's undiscounted lead at turn 114, the lane-exit readout E_20, in independent 500-turn games. The discounted
500-turn lead is reported beside it.

| | lead at 114 | 95% interval | per seed | positive seeds | τ / mean game SE | discounted 500-turn lead |
|---|---|---|---|---|---|---|
| attacker against om:s | **+2.96** | [+0.32, +5.61] | +3.15 +2.17 +1.49 +7.91 +1.98 +1.08 | 6 of 6 | 2.50 / 0.30 | +2.70 [+0.42, +4.98] |
| attacker against sp:s | **+1.15** | [+0.34, +1.96] | +1.12 +2.30 +0.14 +1.73 +0.98 +0.63 | 6 of 6 | 0.71 / 0.30 | +1.02 [+0.61, +1.44] |
| S6, om − sp | **+1.81** | [−0.56, +4.19] | +2.04 −0.13 +1.35 +6.18 +1.00 +0.45 | 5 of 6 | 2.22 / 0.42 | +1.68 [−0.44, +3.80] |

- **Both arms are exploitable** by attackers of this class, initialization, objective and budget. Each lower bound is
  above 0 in every seed. That is a lower bound, not a measure of true exploitability. A stronger or longer attack
  could find more, and an attacker that found little would not show a target to be safe.
- **The sp bound matches M4.5c.** M4.5c's quarter-budget attackers beat its finals by +1.02 ± 0.48. The sp finals'
  +1.15 is in line with that.
- **The om bound is larger on average but not shown to differ.** Seed 3's attacker gained +7.91. The other five om
  seeds range from +1.08 to +3.15, with a median over all six of +2.08 against sp's +1.05. These medians are
  descriptive, not a test. The registered S6 contrast is inconclusive (§3.2). A difference between two learned lower
  bounds is not the same difference in true exploitability, because each bound depends on what its own attacker
  learned.

### 5.3 Transfer (exploratory)

Each attacker starts as a copy of its target. Its gain over that starting point therefore shows what the attack itself
added against each opponent. The references are the target's own lead against the same opponent, from the M5d head
to head for the other arm and from the M5b history games for the common start.

| attacker | opponent | attacker's lead at 114 | starting point's lead | gain from the attack | positive seeds |
|---|---|---|---|---|---|
| om | its own target om:s | +2.96 | 0 (symmetry) | +2.96 [+0.32, +5.61] | 6 of 6 |
| om | sp:s | −1.39 [−2.10, −0.68] | om:s vs sp:s, −0.44 | −0.95 [−1.71, −0.20] | 1 of 6 |
| om | ext/baseline:s | −0.13 [−0.88, +0.62] | om:s vs start, about +0.2 | −0.33 [−1.08, +0.43] | 3 of 6 |
| sp | its own target sp:s | +1.15 | 0 (symmetry) | +1.15 [+0.34, +1.96] | 6 of 6 |
| sp | om:s | −0.04 [−0.31, +0.23] | sp:s vs om:s, +0.44 | −0.48 [−0.96, −0.00] | 1 of 6 |
| sp | ext/baseline:s | +0.03 [−0.74, +0.79] | sp:s vs start, about +0.9 | −0.92 [−1.56, −0.28] | 0 of 6 |

**The attacks do not transfer.** Against every opponent other than its own target, an attacker does no better than
its starting point. In three of four comparisons it does worse with an interval below 0, one of them only just (the sp
attacker against om:s, upper bound −0.001). In the fourth (the om attacker against the start) the interval includes
0. Each attacker trades
general strength for a target-specific deviation. M4.5c found the same for its attackers. Target-specific exploitation
is not evidence of deliberate deception. The attackers are stateless apart from their belief, and no reward rewarded
moving the target's belief.

### 5.4 What the attackers do differently (exploratory, descriptive)

**Where the gain appears.** The lane-exit panel at turn 94 shows it, from the attacker's side (seed means).

| quantity at turn 94 | against om | against sp |
|---|---|---|
| D0, the earned lead | +2.30 [+0.19, +4.41] | +1.03 [+0.39, +1.66] |
| E_20 − D0 (the expected change over the next 20 turns) | +0.66 [+0.03, +1.30] | +0.12 [−0.15, +0.40] |
| XP difference | +4.6 [−1.3, +10.6] | +3.8 [+0.6, +6.9] |
| share of games with the target dead, against the attacker dead | 0.087 against 0.048 | 0.061 against 0.044 |
| gold invested, attacker − target | +2.5 [+0.4, +4.6] | +0.5 [−0.1, +1.1] |
| wave stock, attacker − target | −1.5 [−2.5, −0.6] | −1.1 [−1.5, −0.7] |

**What they play.** The table compares the attacker with its starting point on the same 16 pairs of its own-target
cell. The values are shares of turns with each action component.

- **Against om targets, in the lane phase:**
  - the attackers yield less (−0.083, 0 of 6 seeds positive) and play Safe less (−0.032, 0 of 6);
  - they hold more (+0.056, 6 of 6) and trade and last-hit somewhat more (+0.054 and +0.060, 5 of 6 each);
  - their om targets recall more (+0.021, 6 of 6).
- **Against sp targets:**
  - the attackers shove less, −0.046 in the lane phase (0 of 6) and −0.087 after turn 114;
  - after turn 114 they slow-push and thin more (+0.039 and +0.030, 6 of 6 each);
  - their sp targets play Safe more after turn 114 (+0.047, 6 of 6) and recall more.

**A descriptive reading.** Against om targets, the attackers hold ground and trade instead of yielding. Their targets
die and recall more often and fall behind in gold invested. Against sp targets, the attackers give up some shoving
for a slower wave and more trades, and gain mostly XP and kills. These descriptions say what changed. They do not
establish why it pays, or that any one change causes the gain.

**The om target's belief about its attacker** stays spread across library types. The most probable type at turn 94 is
"mixed" in 0.29 of games, "short" in 0.22, "peer" in 0.19 and "attacker" in 0.14, and the catch-all mass is about 0.29.
Belief movement alone does not show that an attacker manipulated the belief, or that manipulation caused its gain.

### 5.5 Representative games

A rule fixed before any trace was read picks one game per traced cell. It takes the game whose focal lead at turn 114
is the upper median of the cell's 256 games. Fifteen viewer recordings are in `results/m5/traces/`:
- the 12 core attackers against their own targets;
- the seed-0 head to head om:0 against sp:0;
- om:0 and sp:0 against the archetype g_shove.

Each replay reproduced its stored evaluation record exactly before export, and a LaneEnv replay of its actions gives
the same rewards, which is what the lane debugger does. Read descriptively, the traces agree with the panel. In 4 of
the 6 om-target median games the target dies far more often over 500 turns than its attacker (seed 3: 16 against 4,
seed 0: 14 against 1). The traces select nothing and change no statistic.

## 6. Exploitation against competence

The registered evidence establishes the trade-off, and the attackers leave its third part open.

1. **The gain is real and specific.** P1 +9.26 against library-like scripted styles, robust to mixtures and
   switchers. It is near zero against conditional styles built on a competent policy.
2. **The cost is real and general.** P2a −0.63 and P2b −0.44, in every seed. The long-horizon, cross-seed and midpoint
   versions show it too, and so does every cross-arm limitation family. The cost is mostly improvement that pure
   self-play gained over the shared start and the package did not.
3. **Exploitability is unresolved between the arms.** Both arms are exploitable by target-tuned attackers. The om
   lower bound is larger on average, but the registered contrast is inconclusive because of large seed-to-seed
   differences between attackers. The attackers therefore neither strengthen nor weaken the competence conclusion.
   They add one fact about the package alone. It does not make the agent robust to a dedicated attacker, since its
   lower bound is positive in every seed. Whether it makes the agent easier to exploit than pure self-play remains
   open at this budget.

M5 compares packages. It cannot say whether the competence cost comes from the belief inputs, from playing half its
games against fixed opponents, or from about a third fewer self-play transitions.

## 7. Configuration for M6

**Present both arms, as a trade-off.** Neither configuration is uniformly better, and this report does not
combine the style gain and the competence loss into one weighted score.

| use in M6 | configuration | why |
|---|---|---|
| playing recognizable, exploitable opponent styles | the modelling package, `om:s` (`m45_calibrated(obs_last_action=True)`, 116 inputs, belief live) | P1 +9.26; robust to mixtures and switchers (S1 +5.20) |
| playing competent opponents | pure self-play, `sp:s` (`m45_calibrated()`, 90 inputs) | better head to head and against competent finals (P2a, P2b), in every limitation family; still improving at the end of training |

- **Exploitability.** Both are exploitable by dedicated attackers, by at least +1.15 (sp) and +2.96 (om) at the lane
  exit. The difference is unresolved. A demonstration should not present either arm as unexploitable.
- **Which pair to show.** Quantitative claims use all six seeds. For a single demonstration pair, the seed-0 pair
  (`om:0`, `sp:0`) is the convention every M5 smoke and trace used. It is chosen by index, not by performance. Its
  recordings already exist in `results/m5/traces/`.
- **The environment** stays `m45_calibrated()` for both. M5 changed no mechanics, reward, discount or opponent mix.

## 8. Checkpoint lineage and final identities

Lineage of every M5 policy:
1. M4.5c: a fresh s0-only, 100% self-play schedule per seed, 16.384M transitions.
2. M4.6c: the 8.192M baseline continuation, `results/rl/checkpoints/m46c/train/ext/baseline/s<s>/final.pt`
   (`ext/baseline:s`, 24.576M prior transitions).
3. M5b: each arm continues it for 32,768,000 transitions (sp from a copy, om from the copy widened with zero weights on
   its 26 new inputs). The initial-weights check passed in all 12 runs.
4. M5d attackers: each starts from its target final, copied into the om input format, and adds 8,192,000 transitions.

| seed | om final SHA-256 | sp final SHA-256 | start SHA-256 | attacker against om / against sp |
|---|---|---|---|---|
| 0 | `ba3d26f2e7d35dba…` | `be46074e6a019b6f…` | `e9e2ef86e07eb494…` | `d21622e31ba93600…` / `3700c0c53ae9cb68…` |
| 1 | `8453fa5643fd4fab…` | `aeb5399d5dada0b3…` | `6c17a58951196841…` | `57ca889a817b6594…` / `616fc54da82b2ec3…` |
| 2 | `51f9fb1257ab325e…` | `dc01114c3eebfa38…` | `f010b739c4c7140e…` | `89639b36873adbce…` / `18b681209a674946…` |
| 3 | `a722f1114b54c347…` | `413c316a8e7f9cab…` | `c847b485e7bcd931…` | `c806003e6ed2fd69…` / `ce05ea0b30b56941…` |
| 4 | `86c42974e8c0b2da…` | `bbe0f9ae7f9e61f0…` | `01e66d2d460846f7…` | `0110be70de6473c2…` / `4ed8987da9da5f83…` |
| 5 | `4ec586a6658f8c34…` | `689e94e54006e8ce…` | `2208b5dd36f51d5e…` | `945b08f24835be28…` / `729be685fbb30970…` |

Full digests, paths, snapshots and every other network an M5 stage read are in `results/m5/reproducibility.md` and
`results/m5/inventory.json`. Every evaluation stage verified these file digests before it ran.

## 9. Runtime by stage

| stage | wall h | estimate (wall h) |
|---|---|---|
| M5b training (12 runs × 32.768M) | 10.90 | 9.4 to 11.8 |
| M5b monitoring | 0.56 | 0.50 |
| M5d opponent families and head to head | 1.87 | 1.72 |
| M5c strategic battery | 8.73 | 9.33 |
| M5c limitation cells | 2.11 | 1.50 |
| M5c 150-turn battery re-runs (51 failed validations) | 5.58 | 6.63 |
| M5d core attackers (12 × 8.192M) | 5.96 | 6.64 |
| M5d attacker evaluation (36 cells) | 0.52 | 0.50 |
| **all campaign stages** | **36.20** | |

The M5e checks before and after the attackers added little compute:
- readiness part C1, 103 s for the final attempt and about 6 minutes over its four attempts;
- the four `--runslow` tests, 11.6 minutes;
- the analysis, 4.8 minutes;
- the traces and behaviour replays, 5.6 minutes;
- the inventory, 17 s;
- the full regression suite, 8.5 minutes.
Six workers ran on a 6-core machine throughout. My own concurrent jobs slowed the first attacker wave for about 15
minutes until I stopped them.

## 10. Registered and exploratory results, amendments and deviations

**Registered (pre-registration version 1).**
- The headline from P1, P2a and P2b: outcome 2.
- The 15-hypothesis secondary Holm family, now final: S1 preserved; S2, S3, S4 and S5b material losses; S5a
  demonstrated; S6 inconclusive.
- S8 strategic replication by the registered rule. It is descriptive and outside the Holm family.
- The attackers' design, budget, seeds and evaluation cells (§8).

**Exploratory or descriptive.** None of these is a registered test.
- Each arm's attacker lower bound, and S6 on the discounted readout. This report gives both beside S6, as §8 asks.
- Attacker transfer and the gain over the starting point.
- The attackers' lane-exit panel, behaviour shares and the om target's belief about its attacker.
- The 15 traces.
- From M5b and M5c: each arm against the common start, per-opponent values, switch windows, the belief mechanism, the
  midpoint head to head, the G50 reading and the 150-against-300-turn diagnostic.

**Amendments.** One, amendment 1 (2026-10-17), recorded before any training. M5e made none.

**Deviations from the protocol.** None in the experiment. M5e changed implementation and interpretation only.
- *The attacker stage's runtime handling.* §8 is silent on attacker failures, so M5e applied the main training's §3
  rules:
  - a live health monitor in which a defect alarm ends a run;
  - a coordinator that stops after more than two failed seeds;
  - deterministic restarts that must reproduce the preserved history prefix;
  - a post-hoc check that gates the evaluation.
  Healthy runs are bit-identical with or without the monitor (C1). No alarm or restart occurred. I judged this an
  implementation of the registered recovery rules, which the brief directed, and recorded no amendment. No result
  depends on it, because no alarm, failure or restart occurred.
- *The history reader.* `train.history` now drops a truncated last line, which a process killed while writing would
  leave. This also changes the main training's restart path. A bad line anywhere else still raises, and no M5 run has
  a truncated history.
- *Additive records and readouts.* The evaluation also stores the om target's belief trace about its attacker. The
  analysis adds the readouts that §8 and §9 name but the code lacked: the discounted lead beside S6, each arm's lower
  bound, and transfer. Every earlier endpoint reproduced exactly after the change.
- *Interpretation corrections in this report and the living summaries.* The historical M5c report is unchanged.
  - The om zoning label is now open (suggestive), not missed (§4.2).
  - "The same strategic choices" is now "the same four registered findings" (§4.1).
  - The cold-start reasoning now rests on the s0 evaluations (§4.3).
- *Not run.* The optional om_table adaptation ablation (6 attackers, a separate decision).

## 11. Limitations

- **A package contrast.** M5 cannot attribute the competence cost to the belief, the opponent mix or the reduced
  self-play exposure (§1).
- **Six pairs, continuations.** The lineage contains three schedules and three learning-rate re-warms. The om arm
  trained with one opponent library and one opponent mix. The sp arm had not converged.
- **Library-favourable style families.** M5 measures the exploitation gain mostly against scripted styles that the
  library resembles. The competent opponents come from the same lineage as both arms.
- **Constructed-state cold starts** (§4.3). Cross-arm cells start the om final with a prior belief.
- **Shared specialists.** Only three gap specialists exist (seeds 0, 2 and 4); seeds 1, 3 and 5 reuse a neighbour's,
  so the six-seed counts of the large-gap cells overstate the independent evidence.
- **The battery's replication statistic** takes the largest of several alternatives per family, which biases each
  per-seed gap upward. Constructed families are states the policies rarely visit.
- **Approximate attackers.** Each attacker is one learned best response at a quarter of its target's continuation budget, initialized from the target. Its lead is a lower bound on exploitability, and two lower bounds can differ for reasons that belong to the attackers. The attackers keep no state beyond their belief, so they cannot test deliberate deception, and nothing rewarded moving the target's belief.
- **Coverage.** M5 measures nothing beyond 500 turns.

## 12. Reproducing M5

`python -m lanegame.experiments.rl.m5 reproduce` generates `results/m5/reproducibility.md` from the records. It
lists the protocol version and its hash, the configurations, every seed root, every stage command with its approval
flag, the measured wall times, and the SHA-256 of every network and record file the results depend on.

- **From the saved records.** `python -m lanegame.experiments.rl.m5 prereg_check train_check attack_check analyse
  report` re-derives every table in minutes. It needs the git-ignored records under `results/rl/checkpoints/m5/`
  (evaluation, battery and cell records, and the run directories' summaries) and the checkpoints the gates verify.
- **The campaign.** Retraining and re-evaluating takes about 36 h on six workers. Run the stages in the
  order of the reproducibility table, each behind its approval flag. With the same code and seeds each stage reproduces bit for bit. Readiness parts R4, A1 and
  C1 test this for training and the attackers.
- **The traces.** The `traces` and `attack_behaviour` stages replay evaluation games from their seeds. Each recording opens in the
  lane debugger (`streamlit run lanegame/viewer/app.py`, mode "Recorded trajectory", folder `results/m5/traces`).

## 13. Tests and verification

| check | result |
|---|---|
| a read-only audit of the attacker path and the S6 / Holm analysis by a separate agent that I briefed; its probe scripts stayed in the session scratchpad, not the repo | no defect that would bias S6 or the training; five robustness and reporting gaps, all fixed. Readiness part C1 re-checks the same items on the real finals and is in the repo |
| C1 attacker readiness on the real finals (`results/m5/corrections_readiness.json` part C1) | 56 of 56 on the fourth attempt. The first three stopped on a target-phase bug in the C1 jobs, a numpy-bool record error and a too-strict panel check. All attempts are kept in the record |
| attacker check after training (`results/m5/attack_check_main.json`) | 180 of 180 |
| P1 to S5, the headline and every exploratory readout after the attacker records and code changes | identical to the committed `results/m5/analysis_main.json` at HEAD (`git show HEAD:results/m5/analysis_main.json`); monitoring, S8 and the battery summaries identical apart from metadata |
| every trace replay against its stored record | 15 of 15 exact |
| M5 test files (`tests/test_m5.py`, `test_m5_corrections.py`, new `test_m5_attackers.py`) | 50 passed on the final tree |
| the four `--runslow` tests (`test_legacy_pins.py:76`, `test_m4_observation.py:55` and `:65`, `test_m4_tower.py:81`) | 4 passed in 685 s (`results/m5/slow_tests_m5e.log`), the first time in M5 |
| full regression suite (`results/m5/regression_suite_m5e.log`) | 1,541 passed, 4 skipped (the `--runslow` tests above), 2 expected failures from earlier milestones, 3 warnings from PettingZoo's API checker; 8.5 minutes |

- **Never exercised on real data:**
  - the attacker restart path, tested on a deliberately interrupted readiness run but not needed in the campaign;
  - the coordinator's stop after more than two failed seeds, tested with stand-in jobs only.

## 14. What remains

**M6: presentation, demonstration and reproducibility.**
- Present M5 as the trade-off of §6 and §7: the package's gain against recognizable styles beside pure self-play's
  competence, with both arms' exploitability lower bounds.
- Demonstrate with the seed-0 pair and the 15 viewer recordings. The lane debugger opens them directly.
- Package the reproducibility materials. `results/m5/reproducibility.md` and the inventory list everything a reader
  needs. The git-ignored tree `results/rl/checkpoints/m5/` holds 4.3 GiB: 2.6 GiB of training runs and snapshots,
  0.8 GiB of battery records and 0.75 GiB of attacker runs and evaluation. It must travel with any archive, together
  with the M4.6c starts and the OM2 library networks.

**Optional future research.** These are proposals that would need a new design and approval. None is needed to finish
M5.
- Isolate the cause of the competence cost: a matched arm with the fixed opponents but no belief, or the belief with
  equal self-play exposure.
- The om_table adaptation ablation, read with the limits stated in amendment 1.
- Larger attacker budgets or several attackers per target, which would tighten the exploitability comparison.
- The two gaps both arms share, constructed zoning and the favourable-freeze window.
