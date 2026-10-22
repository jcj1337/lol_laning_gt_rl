# M4.7 — The early-lane horizon study

*2026-10-16. Status: complete, awaiting review. Nothing is staged or committed. No default, preset, rule, reward or
checkpoint changed: the shorter discount was an opt-in keyword override used only in the pilot.*

- Protocol of Stage 2: [m4_7_preregistration.md](m4_7_preregistration.md), frozen before any pilot training run
  (SHA-256 in `results/m47/preregistration.json`).
- Generated tables: `results/m47/report.md`. Code: `lanegame/experiments/rl/m47/`; tests: `tests/test_m47.py`.

**Terms.**
- **Resource:** the gold of one fresh wave; natural income is about 0.40 resources per turn.
- **Lane phase:** turns 0–93. In natural self-play a seat first reaches level 14 at turn 94 (median), so "before level
  14" and "turns 0–93" coincide to within a few turns (§1).
- **H:** the expected duration of the discount's independent-hazard reading, γ = 1 − 1/H. H 100 is the current
  objective (γ 0.99); H 60 is γ 0.98333.
- **G_h:** a seat's discounted (γ 0.99) earned-gold lead over h turns without a critic bootstrap, as in M4.6.
- **Matched M4.6c policies:** `ext/baseline:s` and `ext/C1:s`, the M4.6c 8.192M continuations of the same six M4.5c
  finals in each curve's environment.
- Stage 1 intervals are 95% t intervals over six seeds (5 df); C1 − baseline is paired by seed (and by game).

## Summary

**Practical conclusion: keep the current horizon (γ 0.99) for M5.** Analyse the lane phase with elapsed-turn windows
(turns 0–93) rather than change the objective. The shorter-horizon pilot is pre-registered outcome 5 (regression);
under the analysis code's first, unpaired interval it would have been 6 (inconclusive). Either way, it showed:
- no reliable lane-phase gain;
- a consistent loss later in the game and against fixed opponents;
- a few changed decisions (more freeze holding, fewer Recalls late).

Nothing was adopted; γ 0.99 and the baseline curve remain as they were.

**Needs discussion.**
- **The correction.** The analysis code was corrected to the protocol's paired intervals after I had seen the
  fixed-opponent results (§5.4). This moved the label from 6 to 5; the means did not change.
- **The training distribution, not the objective, is where most of the "emphasis beyond the lane phase" sits.** 83% of
  training transitions are after level 14 under either γ. A discount cannot change that. If the lane phase should
  receive more of the training signal, the lever is the start distribution or game lengths, which keep the objective
  (future work, §6).
- **C1.** In the lane phase the two curves are the same game. C1's documented benefit (mid-game leads) lies after
  level 14, and most of its extra value for early leads is paid after level 14 too. If the research focus is the lane
  phase, this is a further reason the baseline suffices.

**Answers to the study's five questions.**
1. **What part of current play occurs before level 14?** Turns 0–93: level 14 arrives at turn 94 (p10–p90 90–99),
   almost like a clock, in both curves and across the training lineage.
   - That is 17% of training transitions.
   - It carries 61% of the objective's weight from s0. A decision taken inside the lane phase has 36% of its weight
     there.
   - Kills, plates, Pressure and freezes are concentrated in it (1.4–2 times the later rate per turn); income per turn
     is the same as later.
2. **How do baseline and C1 differ during that phase?** Not measurably in timeline, decisions, income, Recalls or
   freezes. The one difference: C1's early earned leads persist from turn 50 to 94, while the baseline's shrink by
   about a fifth (+0.24 ± 0.15 in the conversion slope). C1's extra value for a constructed early lead is mostly
   realized after level 14 (at level 8: +0.27, of which +0.17 after).
3. **Are delayed rewards materially driving the earlier conclusions?** No.
   - The early conclusions were measured mostly with 50-turn statistics, which for early states end inside the lane
     phase.
   - The paired benefits behind them (Shove, Trade, Safe, Recall and reset, HP, early leads, conceding and contesting
     freezes) are realized mostly before level 14. They keep their sign and order under H80, H60 and H40; no tested
     alternative beats the policy under any of them.
   - Delayed rewards matter for the size of advantages gained at levels 8–11 (a quarter to a half paid after
     level 14) and for C1's effect.
4. **Did the shorter objective change meaningful decisions?** A few, and not into better lane play.
   - **In both seeds:** more freezes held in the lane phase (+1.6 points of seat-turns), less Hold, more conceding when
     frozen on; after the lane phase fewer Recalls at lower HP.
   - **Lane phase head to head:** +0.26 ± 0.17 in one seed, −0.09 ± 0.17 in the other.
   - **Cost:** −0.36 to −0.69 under the full objective against fixed opponents (paired intervals ±0.21), mostly after
     the lane phase and against scripted opponents.
5. **Keep, pursue or leave unresolved?** Keep the current horizon. A shorter horizon is not worth pursuing on this
   evidence. The pilot is two continuation seeds, so a small lane-phase benefit from a from-scratch short-horizon
   campaign is not excluded; it would have to outweigh the measured late-game and robustness costs.

---

## 1. What part of current play occurs before level 14?

**Sources.** Stage 1 reuses existing artifacts wherever they exist:
- the M4.6d full-state records of the matched M4.6c policies: 48 self-play games of 400 turns per policy and
  environment, the same game seeds in both curves;
- the M4.6c natural records (other seeds, 600 turns) of the M4.5c finals and both continuations, for the lineage;
- the M4.6e exposure games, which reconstruct the M4.6c training start distribution.

New games were run only where nothing existed: the paired 300-turn rollouts of §3.

### 1.1 The level timeline is close to a clock

| level | turn (baseline) | p10–p90 | C1 − baseline | M4.5c finals (lineage) |
|---|---|---|---|---|
| 6 | 21.5 ± 0.2 | 19–24 | −0.15 ± 0.63 | |
| 8 | 34.9 ± 0.3 | 32–38 | −0.27 ± 0.49 | |
| 10 | 51.4 ± 0.4 | 48–55 | −0.17 ± 0.53 | 50.0 |
| 12 | 71.1 ± 0.5 | 68–75 | −0.02 ± 0.56 | 69.6 |
| **14** | **94.3 ± 0.5** | **90–99** | −0.25 ± 0.37 | 92.7 |
| 16 | 120.5 ± 0.6 | 116–126 | −0.10 ± 0.81 | |
| 18 | 150.0 ± 0.7 | 144–156 | −0.21 ± 0.65 | 148.4 |

- XP comes mostly from minions in range, which both seats collect almost every turn. Levels therefore track elapsed
  turns: the two seats of a game reach level 14 a mean 3.6 turns apart (p90 7), seat A and seat B identically.
- **The curves and the training lineage do not move the timeline.** C1 differs by less than a third of a turn. The
  M4.5c finals, the 4.096M and the 8.192M continuations all reach level 14 within turns 92.5–94.3.
- **Level-triggered and elapsed-turn windows are interchangeable here.** Every comparison below uses common
  elapsed-turn windows anyway; where a level-triggered cut is also reported, it agrees with the elapsed one (§3).
- This matches the earlier audit (levels 9, 14 and 18 at turns 50, 100 and 150) to within a few turns.

### 1.2 The state when a seat reaches a level (baseline; C1 identical within noise)

| level | turn | earned | invested | unspent | HP | deaths | Recalls completed | interrupted | \|earned lead\| |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 51.4 | 20.1 | 15.3 | 4.8 | 0.78 | 0.52 | 2.4 | 1.3 | 1.9 |
| 12 | 71.1 | 28.1 | 23.3 | 4.8 | 0.77 | 0.71 | 3.3 | 1.9 | 2.2 |
| **14** | **94.3** | **37.6** | **32.6** | **5.0** | **0.78** | **0.93** | **4.3** | **2.7** | **2.6** |
| 18 | 150.0 | 60.3 | 54.8 | 5.5 | 0.79 | 1.28 | 6.8 | 4.4 | 3.3 |

By level 14 each seat has earned about 38 resources, invested 33 (21% of what it will have invested by turn 400),
holds about 5 unspent, has completed about four Recalls and died about once. C1 − baseline is within ±0.7 resources
of investment and ±0.4 Recalls at every level.

### 1.3 How much training and how much of the objective lie before level 14

**Training occupancy** (seat-transitions, reconstructed from the M4.6e exposure games of the M4.6c start mixture):

| below level | 8 | 10 | 12 | 14 | 16 |
|---|---|---|---|---|---|
| training share (archive 25%, s0 75%) | 5.5% | 8.5% | 12.4% | **17.3%** | 23.4% |

- s0 games alone (the M4.5c from-scratch campaign trained only from s0): 18.3% before level 14. Archive games: 14.3%
  (their starts have median lower level 14).
- Identical within ±0.5 points for both curves and for C1 snapshots at 1.0M, 4.1M and 8.192M transitions.
- **About 83% of everything the network trains on is after level 14.** A discount does not change this: it
  reweights what each decision is optimized for, not which states are sampled.

**Objective weight** (exact; 1 − γ^t before turn t from s0):

| H | γ | before L10 | before L12 | before L14 | before L18 | lane-phase decisions: weight before L14 | beyond turn 300 |
|---|---|---|---|---|---|---|---|
| **100 (current)** | 0.99 | 0.40 | 0.51 | **0.61** | 0.78 | **0.36** | 0.049 |
| 80 | 0.9875 | 0.48 | 0.59 | 0.69 | 0.85 | 0.42 | 0.023 |
| 60 | 0.98333 | 0.58 | 0.70 | 0.79 | 0.92 | 0.50 | 0.006 |
| 50 | 0.98 | 0.65 | 0.76 | 0.85 | 0.95 | 0.56 | 0.002 |
| 40 | 0.975 | 0.73 | 0.83 | 0.91 | 0.98 | 0.62 | 0.001 |

- The last-but-one column averages, over decisions taken at a uniformly drawn turn of the lane phase, the share of
  each decision's own objective weight that falls before level 14.
- **Under the current objective, 39% of the weight from s0 lies after level 14, and a typical lane-phase decision is
  64% about what happens after the lane phase.**
- The independent-hazard reading of H 100 has median 69 turns. It is broad: it does not concentrate the end of laning
  near turn 100.

### 1.4 What the lane phase looks like (baseline; common elapsed-turn windows)

| per seat | lane (0–93) | after (94–399) |
|---|---|---|
| income per turn | 0.397 | 0.400 |
| kill bounty per turn | 0.019 | 0.010 |
| plate gold per turn | 0.012 | 0.006 |
| deaths per 100 turns | 0.97 | 0.51 |
| Recall choices / completed Recalls per 100 turns | 7.5 / 4.6 | 7.2 / 4.4 |
| Shove share of wave choices | 0.34 | 0.40 |
| Hold / Yield / Trade / Safe share | 0.28 / 0.25 / 0.24 / 0.14 | 0.26 / 0.21 / 0.24 / 0.22 |
| Pressure / Ward share | 0.038 / 0.011 | 0.022 / 0.023 |
| turns holding a freeze | 6.5% | 4.5% |
| HP while present | 0.75 | 0.78 |

- Income is flat after the first 25 turns (0.37 in turns 0–24, then 0.40–0.41).
- **The interactive part of the economy is concentrated in the lane phase:** kills, plates, Pressure and freezes are
  1.4–2 times as frequent per turn as later. After level 14 play shifts toward Safe and Wards and dies half as often.
- Recalls in the lane phase are chosen at lower HP (0.40 vs 0.47) and with less unspent gold (6.3 vs 8.1); 38% are
  interrupted, as later. The first completed Recall comes at turn 15 on average.
- Freeze episodes in the lane phase: 3.1 per seat, mean length 2.0 turns, 23% lasting three turns or more (after:
  1.6 turns, 14%). When frozen on, the seat Shoves (contests) 56% of the time in the lane phase, 51% later.

## 2. How do baseline and C1 differ during that phase?

**Behaviour: no measurable difference.** Paired by seed and game, every lane-window metric differs by less than its
across-seed noise: income +0.004 per turn, Shove +0.006, Trade −0.018, Hold +0.015, Recall choices −0.4 per 100 turns
(each interval includes 0), level times −0.25 turns, investment at level 14 +0.26 ± 0.62.

**Leads: the one lane-phase difference.**

| | baseline | C1 | C1 − baseline |
|---|---|---|---|
| \|earned lead\| at turn 50 / 94 / 150 | 2.28 / 3.29 / 4.31 | 2.33 / 3.64 / 4.81 | +0.04 / +0.35 ± 0.40 / +0.51 ± 0.64 |
| conversion slope, turns 25 → 75 | −0.27 ± 0.17 | −0.21 | +0.07 ± 0.21 |
| **conversion slope, turns 50 → 94** | **−0.20 ± 0.07** | **+0.04** | **+0.24 ± 0.15** |
| conversion slope, turns 94 → 150 | −0.01 ± 0.12 | +0.03 | +0.05 ± 0.20 |

The slope regresses the later change of a game's earned-gold lead on the lead at the first turn. Under the baseline,
a natural lead at turn 50 shrinks by about a fifth by turn 94; under C1 it persists. Natural leads are small either way
(9% of earned gold at turn 94).

**Constructed early leads (§3.1): C1's extra value is mostly delayed.** At level 8, C1 raises the premium of a +8
lead by +0.27 (H100), of which +0.17 arrives after level 14; at level 11, by +0.40, of which +0.28 after level 14.

**Reading.** In the lane phase the two curves produce the same game: the same timeline, the same decisions, the same
income. C1 makes early earned leads persist rather than partly revert, and most of the extra value it gives an early
investment lead is realized after level 14.

## 3. Are delayed rewards materially driving the earlier conclusions?

Two readings are kept apart:
- **State value** (what a state is worth, including whatever the policy later does): §3.3, from existing
  trajectories.
- **Action-dependent strategic benefit** (what an action or an advantage adds, paired on common random numbers against
  the policy itself): §3.1–§3.2, new 300-turn rollouts that keep every turn's reward.

Every rollout in this section follows the γ 0.99 policies. Rescoring their streams under a shorter H measures how an
**unchanged** behaviour scores under a shorter objective (reweighting). It does not show what agents trained for that
objective would do: that is Stage 2.

### 3.1 Starting advantages (constructed lead cells, baseline; K 32 × 3 lane situations × 6 seeds)

Both champions at level L with the natural investment, one side given +4 or +8 invested resources. The premium is
the leader's extra discounted gold over the equal state (both orientations, the M4.6c definition). "After L14" is the
part of the H100 premium earned after the lower level reaches 14 (level-triggered; with the elapsed-turn cut the share
moves by at most two percentage points).

| cell (turn) | G20 | G50 | H100 (300 turns) | after L14 | share after L14 | H80 | H60 | H40 |
|---|---|---|---|---|---|---|---|---|
| L4 +4 (≈ 12) | +0.43 | +0.70 | +0.85 ± 0.14 | +0.10 | 12% | +0.78 | +0.70 | +0.58 |
| L4 +8 | +0.70 | +1.21 | +1.60 ± 0.18 | +0.16 | 10% | +1.48 | +1.31 | +1.05 |
| L8 +4 (≈ 35) | +0.26 | +0.44 | +0.62 ± 0.18 | +0.18 | 29% | +0.56 | +0.49 | +0.39 |
| L8 +8 | +0.37 | +0.72 | +0.98 ± 0.27 | +0.23 | 23% | +0.89 | +0.77 | +0.61 |
| L11 +4 (≈ 60) | +0.14 | +0.32 | +0.44 ± 0.41 | +0.18 | 42% | +0.39 | +0.34 | +0.26 |
| L11 +8 | +0.19 | +0.37 | +0.58 ± 0.26 | +0.34 | 59% | +0.52 | +0.44 | +0.33 |

C1 (same cells): L4 +8 +1.55 (17% after L14), L8 +8 +1.25 (32%), L11 +8 +0.98 (63%).

- **The L8 +8 cell reproduces M4.6c's published early premium** (G50 +0.72 here, +0.63 there, on other seeds).
- **The earlier an advantage, the more of its value stays inside the lane phase:** about 90% at level 4, about 75% at
  level 8, about half at level 11.
- **Under H 60 the premiums shrink by 18–24% and keep their order.** Early leads still matter, and they matter more
  than later ones under every horizon.

### 3.2 Early actions and natural early states

**Natural early states** (24 states per seed, at turns 8–80 of each policy's own self-play; K 12; candidate − policy
for seat A; pooled over states, baseline; C1 in the generated tables):

| alternative | H100 | after L14 | H80 | H60 | H40 | applied |
|---|---|---|---|---|---|---|
| Shove ×1 | −0.08 ± 0.10 | +0.01 ± 0.13 | −0.08 | −0.09 ± 0.09 | −0.09 | 94% |
| Shove ×4 | −0.20 ± 0.31 | +0.05 ± 0.12 | −0.22 | −0.22 ± 0.27 | −0.22 | 94% |
| Trade ×1 | −0.01 ± 0.23 | +0.05 ± 0.09 | −0.03 | −0.06 ± 0.18 | −0.08 | 94% |
| Safe ×1 | −0.02 ± 0.17 | +0.10 ± 0.14 | −0.04 | −0.06 ± 0.11 | −0.10 | 94% |
| reset now (walk back, Recall when legal) | −0.42 ± 0.18 | −0.06 ± 0.15 | −0.41 | −0.41 ± 0.17 | −0.42 | 100% |
| +4 invested (state variant) | +0.38 ± 0.11 | +0.02 ± 0.23 | +0.35 | +0.31 ± 0.09 | +0.25 | 100% |
| −0.3 bars of HP (state variant) | −0.57 ± 0.15 | −0.08 ± 0.10 | −0.56 | −0.55 ± 0.15 | −0.54 | 100% |

"Applied" is the share of rollouts in which the alternative changed the turn-0 decision (the rest coincide with the
policy).

- **No tested alternative beats the policy, under any horizon.** The after-L14 parts are within ±0.13 and none is
  significant. They lean positive for Trade and Safe (+0.05, +0.10): under H100 a Trade or Safe turn is about neutral,
  and under shorter horizons it becomes slightly worse (−0.06 at H60). Rescoring therefore makes the policy's lane
  choices look more correct under a shorter horizon, not less.
- **An early investment advantage pays inside the lane phase.** In natural states up to turn 35, +4 invested resources
  are worth +0.73 ± 0.40, 90% of it before level 14. From turns 50–80 the same +4 is worth +0.03 ± 0.35 under the
  baseline and +0.40 ± 0.21 under C1 (41% of it after level 14).
- **Ranking stability** (the policy and its five alternatives, per state): rescoring under H80 changes the best option
  in 6% of states, under H60 in 13%, under H40 in 28% (Kendall τ against H100: 0.94, 0.85, 0.72; C1 similar). With 12
  rollouts per state many options are within noise of each other, so these shares are upper bounds on real ranking
  changes.

**Early freeze cells** (the M4.6d builder; levels 4 and 8; equal and healthy, or the freezer with M4.6d's plausible
edge against a breaker at 60% HP; K 24).
- Conceding a freeze for 10 turns costs the breaker 0.4–1.8 resources, and resetting 0.1–1.9. Under C1, guarding the
  level-4 freeze against a damaged breaker gains +0.95 ± 0.52 (M4.6d: +0.3 to +0.5 at 50 turns).
- Almost all of it is realized before level 14. Every after-L14 part is within ±0.38; 3 of 48 are nominally
  significant, in both directions, about what chance gives. Every significant total keeps its sign under H60.
- These 300-turn readings are noisy (intervals ±0.3 to ±1.6). They speak to when the value arrives, not to M4.6d's
  finer freeze-window comparisons.

### 3.3 State value: existing trajectories rescored

The M4.6d records keep seat A's reward for every turn of 48 × 400-turn games per policy. From the state at turn t0
the realized return (no bootstrap; the omitted tail beyond turn 400 weighs at most 2–4% under H100) splits exactly at
turn 94.

| H | t0 | SD of the realized return | variance share after L14 | corr(before, after) |
|---|---|---|---|---|
| 100 | 0 | 2.79 | 0.18 | −0.00 ± 0.13 |
| 100 | 25 | 3.13 | 0.24 | −0.04 ± 0.13 |
| 100 | 50 | 3.32 | 0.35 | −0.04 ± 0.10 |
| 100 | 75 | 3.22 | 0.61 | −0.10 ± 0.11 |
| 60 | 0 | 2.07 | 0.06 | +0.01 ± 0.13 |
| 60 | 50 | 2.73 | 0.20 | −0.03 ± 0.12 |

- **In natural play, what happens after level 14 is unrelated to what happened before it** (correlation ≈ 0 at
  every t0 and horizon, both curves). After-L14 rewards add variance to an early state's value, not a systematic
  continuation of the lane phase's outcome.
- The value of a natural early advantage, read observationally (a regression of the realized return on the invested
  lead with the unspent-gold, level and HP differences as covariates), is imprecise: per resource of invested lead at
  turn 50, −0.01 ± 0.22 under the baseline and +0.16 ± 0.20 under C1, mostly before level 14 in both. It is reported
  for completeness; the paired cells of §3.1 are the evidence about advantages.

### 3.4 What this means for the published conclusions

- **The early-phase conclusions were mostly measured inside the lane phase already.** M4.4c–M4.6f judged early
  decisions with 50-turn statistics (G50, or Q50 with a critic tail), and M4.5c's horizon study found no significant
  50-turn conclusion that a 300-turn one contradicted. From a state before turn 44, G50 ends before level 14.
- **Delayed rewards do not drive them.** The paired benefits that back the early "used" verdicts (Shove where it pays,
  Recall timing and the cost of staying, conceding and contesting freezes, the value of early leads and HP) are
  realized mostly before level 14 and keep their sign and order under H80, H60 and H40.
- **Where delayed rewards do matter:**
  - the size (not the sign) of an advantage gained at levels 8–11, a quarter to a half of which is paid after
    level 14;
  - C1's effect: its extra value for an early lead is mostly delayed, and its published benefit (the mid-game +16 lead
    premium, M4.6c) is measured at level 16, entirely after the lane phase;
  - "late leads fade" (G1) is a statement about the post-lane game by definition.
- **What this does not settle.** The policies themselves were trained for γ 0.99, so even a 50-turn measurement reflects
  behaviour chosen with the long horizon in view, and the critic tail of a Q50 statistic is a γ 0.99 value. Whether
  agents trained for a lane-focused objective would behave differently is the pilot's question (§5).

## 4. The pilot's design choices

**Is a pilot justified?** Stage 1 predicts a small effect: the measured early benefits arrive mostly in the lane
phase, and rescoring under shorter horizons reverses nothing. But rescoring holds behaviour fixed. The objective
difference is material: a lane-phase decision has 36% of its weight inside the phase under H100 and 50% under H60.
The channels through which a long horizon can act are real: the delayed quarter to half of mid-lane advantages, and
investment that pays later. So the direct test, whether agents trained for a lane-focused objective choose
differently, is a meaningful and cheap question (about 1.5 h of compute). The user approved the pre-registered budget
before training started.

**The horizon: H 60** (γ 0.98333; [pre-registration](m4_7_preregistration.md) §3).
- It moves the s0 weight before level 14 from 61% to 79% (92% before level 18), and a lane-phase decision's weight
  inside the phase from 36% to 50%.
- It keeps 76–82% of the early premiums and the 20–50-turn payoffs of Recall, HP and leads (G50 is 71–82% of the
  H100 premium of a level-4 or level-8 lead).
- H 80 would move the weights only modestly (69%; the best option changes in 6% of natural states), a weak
  manipulation for a two-seed screen. H 40–50 would discount the lane phase's own 30–50-turn payoffs (the level-4
  premium falls by a third at H 40) and approach the 17-turn credit window GAE's λ 0.95 already imposes.
- The independent-hazard reading of H 60 has mean 60 and median 41 turns. It does not place the end of laning at turn 60
  or at level 14.

**The curve: the baseline.**
- Stage 1 found no lane-phase difference between the curves that would make C1 the better test bed. Same timeline,
  same decisions, same income. C1's one difference is the persistence of early leads.
- The baseline has the full from-scratch evidence base and is the closeout's recommended M5 configuration.
- **What this leaves open about C1.** C1's extra value is largely delayed (§2), which is exactly what a shorter horizon
  discounts: rescored under H60, its extra L8 +8 premium falls from +0.27 to +0.16. Whether a short-horizon C1 policy
  would keep any lane-phase advantage over a short-horizon baseline policy is not tested.

**Not done** (pre-registration §4.1): no random termination on top of the discount, no shorter rollouts with the old
bootstrap, no change to game lengths, starts, rewards or mechanics, no C1 arm.

## 5. Stage 2: the matched horizon pilot

### 5.1 Design (as pre-registered)

| item | control | short |
|---|---|---|
| objective | `m45_calibrated()`, γ 0.99 (H 100) | `m45_calibrated(gamma=1 - 1/60)`, γ 0.98333 (H 60); the only differing field |
| start | `ext/baseline:s`, s ∈ {0, 1} (M4.6c 8.192M baseline continuations) | the same checkpoint |
| everything else | identical: shared `PolicyValueNet`, M4.6c PPO, the M4.6c archive start mixture, fresh Adam at 3·10⁻⁴ annealed to 0, PPO seed 58,300 + s, 4,096,000 agent transitions | |

**Lineage of the four pilot policies:**
1. 16.384M fresh (M4.5c: s0 starts, γ 0.99).
2. 8.192M continuation (M4.6c baseline arm: archive starts, learning rate re-warmed, γ 0.99).
3. 4.096M continuation here (archive starts, learning rate re-warmed, the arm's γ).

The γ override reaches every place that needs agreement:
- `PPOConfig.continuing` copies the game's γ, and `check_time_semantics` refuses a mismatch.
- GAE, the critic's targets and the bootstrapped truncations use the arm's γ: an in-process spy confirmed it in the
  smoke runs. On identical first rollouts, the advantages under γ 0.98333 differ from those under γ 0.99 by 7–10% of
  their RMS before the critic adapts.
- Game lengths stay at 512 / 150 turns with bootstrapped truncations, and no random termination is added.
- The mathematical reference is unchanged: no preset's objective changed. The override is an experimental keyword, as
  C1 is.

### 5.2 Checks and training

- **Smoke** (twice, before freezing): 66 of 66 checks pass. They cover:
  - the configurations differ only in γ, and γ reaches the trainer and GAE;
  - no termination; every truncation is bootstrapped with a nonzero critic value;
  - learning rate 3·10⁻⁴ at the first update; the policy moves;
  - identical first rollouts in both arms, then divergence;
  - a bit-identical re-run;
  - initial weights equal to the start checkpoint.
- **Training:** 4 runs × 4,096,000 transitions, concurrent, 29.5 min wall (about 2,330 transitions/s per run). 50 of
  50 checks pass:
  - exact budget; game and PPO γ recorded per arm;
  - never terminated, one truncation per finished game;
  - critic bounded by the arm's Ω;
  - both seats trained equally, 100% current policy, no illegal action;
  - initial weights equal the start;
  - archive share 0.249–0.252, and both arms of a seed started the same games;
  - midpoint snapshots present; no stop condition.
- **Learning:** entropy, KL (0.0038 mean in every run) and the critic's explained variance moved alike. Explained
  variance was 0.44 in the short arm against 0.47–0.48 in the control by the end.

### 5.3 Results

**Head to head from s0 (primary).** The short arm's paired-seat lead against the control, 1,024 game pairs per seed,
95% within-seed intervals:

| seed | **lane 0–93 (P1)** | 0–24 | 25–49 | 50–74 | 75–93 | after 94–399 | G100 | G60 | G40 |
|---|---|---|---|---|---|---|---|---|---|
| s0 final | **+0.26 ± 0.17** | +0.09 | +0.16 | −0.01 | +0.02 | −0.92 ± 0.28 | +0.10 ± 0.12 | +0.15 ± 0.09 | +0.16 ± 0.06 |
| s1 final | **−0.09 ± 0.17** | −0.04 | +0.01 | −0.03 | −0.04 | −0.09 ± 0.29 | −0.10 ± 0.12 | −0.03 ± 0.09 | +0.02 ± 0.06 |
| s0 midpoint (2.048M, 256 pairs) | −0.16 ± 0.34 | | | | | −1.91 ± 0.55 | −0.26 ± 0.23 | −0.04 ± 0.17 | +0.07 ± 0.13 |
| s1 midpoint | −0.37 ± 0.36 | | | | | −1.79 ± 0.61 | −0.51 ± 0.24 | −0.23 ± 0.18 | −0.07 ± 0.13 |

- **The seeds disagree.** In seed 0 the short policy gains +0.26 in the lane phase, nearly all of it in turns 0–49,
  and pays −0.92 after it. In seed 1 it gains nothing.
- **Even under its own objective (G60) the short arm leads only in seed 0** (+0.15 ± 0.09; seed 1 −0.03 ± 0.09).
- **At the midpoint both short policies were behind**, by about 1.8–1.9 after level 14. They recovered part of it by
  the end of the annealed schedule.
- The level-triggered window agrees with the elapsed one (+0.26 and −0.08).

**Against fixed opponents** (short − control on the same games, opponents and seats; paired 95% intervals):

| seed | opponents | lane 0–93 | after 94–399 | G100 | G60 |
|---|---|---|---|---|---|
| s0 | the common start | +0.32 ± 0.46 | −1.60 ± 0.81 | −0.03 ± 0.30 | +0.03 ± 0.22 |
| s0 | other seeds' M4.6c finals | +0.11 ± 0.52 | −1.31 ± 0.77 | −0.06 ± 0.34 | +0.04 ± 0.25 |
| s0 | nine scripted references | −1.46 ± 0.39 | −1.29 ± 0.73 | −1.26 ± 0.25 | −0.84 ± 0.19 |
| s0 | **pooled fixed set** | −0.72 ± 0.33 | **−1.30 ± 0.53** | **−0.69 ± 0.21** | −0.43 ± 0.16 |
| s1 | the common start | +0.11 ± 0.48 | −0.50 ± 0.78 | +0.11 ± 0.33 | +0.19 ± 0.24 |
| s1 | other seeds' M4.6c finals | −0.35 ± 0.47 | −0.80 ± 0.81 | −0.32 ± 0.33 | −0.13 ± 0.24 |
| s1 | nine scripted references | −0.48 ± 0.43 | −1.67 ± 0.75 | −0.39 ± 0.27 | −0.20 ± 0.20 |
| s1 | **pooled fixed set** | −0.42 ± 0.32 | **−1.26 ± 0.55** | **−0.36 ± 0.21** | −0.17 ± 0.16 |

- **Against learned opponents** the lane phase is unchanged (−0.35 to +0.32, all intervals include 0), and the short
  arm loses ground after it in every comparison (−0.50 to −1.60; significant in seed 0).
- **Against the scripted references** the short arm earns less in both windows. In seed 0 it is below the control
  against all nine references (G100 −0.9 to −1.6). In seed 1 it is below against seven of nine.
  - This holds even under its own objective (G60: −0.84 and −0.20).
  - The control gained against these references over its continuation (seed 0: +0.68 over the start), the short arm
    lost (−0.58).
- The "after" window is 306 turns long. A lead of −1.3 there is about 1% of one side's income in that window: small
  per turn, consistent in direction.

**Behaviour in natural self-play** (96 games × 400 turns per policy on the same game seeds; short − control):

| lane-window metric | threshold | s0 | s1 | pre-registered "meaningful" |
|---|---|---|---|---|
| turns holding a freeze | 0.005 | +0.016 ± 0.007 | +0.016 ± 0.008 | **yes** |
| Hold share | 0.010 | −0.021 ± 0.010 | −0.034 ± 0.009 | **yes** |
| Safe share when frozen on (descriptive) | 0.052 | +0.087 ± 0.038 | +0.089 ± 0.044 | (not a listed metric) |
| Shove share | 0.033 | +0.063 ± 0.012 | +0.022 ± 0.013 | no (s1 below threshold) |
| Trade share | 0.016 | −0.049 ± 0.009 | +0.020 ± 0.009 | no (opposite signs) |
| deaths per 100 turns | 0.14 | −0.49 ± 0.18 | +0.26 ± 0.19 | no (opposite signs) |
| Recall choices per 100 turns | 0.56 | −0.93 ± 0.40 | −0.03 ± 0.48 | no |
| HP when Recall is chosen | 0.037 | +0.049 ± 0.023 | −0.009 ± 0.023 | no |
| unspent gold at Recall | 0.50 | +0.66 ± 0.37 | −0.52 ± 0.32 | no (opposite signs) |
| invested gold at turn 93 | 0.50 | +0.31 ± 0.86 | −0.87 ± 0.81 | no |
| income per turn | 0.0022 | +0.009 ± 0.004 | −0.004 ± 0.005 | no |

- **Two decisions change in both seeds:** the short arm holds freezes more (control 5.4–5.5% of seat-turns, short
  7.0–7.1%), and plays Hold less. It also concedes more when frozen on.
- **After the lane phase,** in both seeds the short arm completes fewer Recalls (−0.24 and −0.71 per 100 turns),
  recalls at lower HP (−0.03 in both) and plays at lower HP throughout (−0.02, −0.035). That is a shorter horizon
  investing less in the future, as expected; it is also where the short arm loses gold.
- **Every other lane-window change is large in one seed and absent or reversed in the other.** The control itself
  moved away from the common start by similar amounts (control − start, in the generated tables), so seed-specific
  changes are not attributed to the horizon.
- The level timeline is unchanged (level 14 at turns 92–95 in every policy).

**Paired families** (Stage 1's states and rollout seeds, each final against itself; one policy per seed, so these are
descriptive and noisy; means over the family's cells):

| seed | policy | lead premium H100 (H60) | best alternative − policy, natural early states, H100 (H60) | forced concession of a freeze, breaker's view |
|---|---|---|---|---|
| s0 | start | +0.90 (+0.70) | +0.26 (+0.14) | −1.75 |
| s0 | control | +0.92 (+0.72) | −0.25 (−0.20) | −1.01 |
| s0 | short | +0.52 (+0.43) | −0.05 (−0.13) | −0.39 |
| s1 | start | +1.00 (+0.80) | +0.22 (+0.12) | −1.12 |
| s1 | control | +0.82 (+0.67) | −0.37 (−0.33) | −1.14 |
| s1 | short | +0.91 (+0.73) | +0.12 (+0.01) | −0.32 |

- **Conversion of plausible advantages:** no consistent change. Seed 0's short policy converts constructed early leads
  less (+0.52 against +0.92), seed 1's about the same (+0.91 against +0.82).
- **No tested one-turn or short-plan alternative materially beats either final under its own objective** (best
  alternative under H60 for the short arm: −0.13 and +0.01). The pilot does not show the short arm finding lane-phase
  options the control misses.
- **Freezes:** forcing the breaker to concede costs much less relative to the short arm's own policy (−0.3 to −0.4)
  than to the control's (−1.0 to −1.1). This matches the natural-play finding that the short arm concedes more when
  frozen on and holds freezes more. Per-cell freeze readings are too noisy at one policy per seed (±1 resource) to say
  whether this is better play under either objective.

### 5.4 Pre-registered classification

| criterion | s0 | s1 | met |
|---|---|---|---|
| B: P1 ≥ +0.25, interval excluding 0 | +0.26 ± 0.17 | −0.09 ± 0.17 | no (one seed) |
| LR: P1 ≤ −0.25 | | | no |
| R: pooled fixed set, G100 ≤ −0.25 (or after window), interval excluding 0 | −0.69 ± 0.21 (after −1.30 ± 0.53) | −0.36 ± 0.21 (after −1.26 ± 0.55) | **yes** |
| meaningful behaviour change | freeze holding up, Hold down | the same | yes (2 of 17 metrics) |

**Outcome 5, regression: keep the current horizon.**

**An analysis correction, reported with both readings** (pre-registration record, amendment of 2026-10-16):
- The frozen protocol specifies intervals over paired game indices: short − control "on the same games" (§5).
- The first analysis code computed the fixed-opponent difference with an unpaired (independent-error) interval. I
  corrected it to the protocol's paired interval after I had seen the head-to-head and fixed-opponent results.
- Under the unpaired interval, seed 1's pooled G100 is −0.36 ± 0.36: the regression flag is not raised, and the
  outcome is 6 (inconclusive).
- The means are identical under both readings. The substantive picture does not depend on the choice: no reliable
  lane-phase gain, and a consistent loss after the lane phase and against scripted opponents.

## 6. Recommendation

**Keep γ 0.99 (H 100) as the M5 objective.** Do not pursue H 60, and do not adopt a new discount or investment curve.
- **Stage 1:** the current objective does not make the published early-lane conclusions depend on post-lane rewards.
- **Stage 2:** an objective focused on the lane phase did not buy reliable lane-phase performance. It cost play later
  in the game and against fixed opponents, and changed a few decisions (freeze holding, conceding, late Recalls) in
  ways the other evidence does not show to be improvements.
- **Comparability:** every M4 result was obtained under γ 0.99.

**How to honour the lane-phase focus in M5 without changing the objective.**
- **Report lane-phase readouts in elapsed-turn windows.** Use turns 0–93, which equal "before level 14" to within a few
  turns; Stage 1 showed level-triggered and elapsed windows agree. Where an early statistic is needed without
  post-lane rewards, use undiscounted or G_h statistics truncated at turn 94, never critic tails.
- **Keep early-game strategic families as constructed cells** (the Stage 1 lead and freeze cells), as M5 already
  plans.

**The horizon question's status: answered for practical purposes, with stated limits** (§7). A from-scratch
short-horizon campaign was not tested, nor was H 80.

**Future work (described, not implemented):**
- **Training occupancy, not the objective.** 83% of training is after level 14. The M4.6c archive start mixture
  already shifts some training to constructed early states. A start distribution weighted toward early natural states,
  or shorter s0 games with the γ 0.99 bootstrap, would give the lane phase more of the training signal without
  changing the objective. It would need its own matched test: the bootstrap makes such games approximate the same
  objective only as well as the critic is right, and the critic is strength-blind (M4.6e).
- **Other ending mechanisms**: a terminal payoff at a level, a tower-based end, or a progression-dependent hazard.
  Each would change the game being studied. Stage 1 gives no reason to prefer one. Post-lane outcomes are
  uncorrelated with lane-phase outcomes in natural play, so a terminal valuation at level 14 would mostly remove
  variance from early-state values. It would still cut off the delayed part of mid-lane advantages, as H 60 discounts
  it. None is recommended now.
- **Bayesian opponent modelling** remains a desired later addition, not designed or implemented here.

## 7. Limitations

- **Two seeds.** Between-seed uncertainty cannot be estimated, and the seeds disagree on the primary (+0.26 vs −0.09).
  The classification required agreement, and only the regression flag and two behaviour changes had it.
- **Continuation, not fresh training.**
  - Both arms continued policies trained for 24.6M transitions at γ 0.99.
  - The short arm was behind at the midpoint (−1.8 to −1.9 after level 14) and recovered part of it by the end.
  - Longer training, or a from-scratch short-horizon campaign, could end elsewhere.
- **The re-warmed learning rate** is matched across arms but adds disruption to both.
- **One horizon (H 60) and one curve (baseline).** H 80 and a short-horizon C1 were not tested.
- **Constructed and natural paired families** are one policy per seed with K 12–32 rollouts per state: descriptive.
- **Fixed opponents** are the M4.6c baseline finals of other seeds and the M4.5c scripted suite. The scripted suite
  is exploitable by design, and the short arm's loss there is the largest single component of the regression flag.
- **The correction** (§5.4) was made after seeing results. Both readings are reported, and they differ only in the
  classification label.
- **Stage 1's rescoring** reweights trajectories of γ 0.99 policies. It measures how unchanged behaviour scores under
  other objectives, not what agents optimized for them would do.
- **Stage 1's training occupancy is reconstructed** from frozen policies (M4.6e), not logged from training.

## 8. Runtime, lineage, checks and files

**Runtime** (wall clock, on 4–6 worker processes at below-normal priority on a 6-core desktop that was in other use
for part of the time):

| stage | wall |
|---|---|
| timeline, occupancy, returns (existing artifacts) | < 1 min |
| Stage 1 paired rollouts (lead cells, natural states, freeze cells; about 16M simulated turns) | 87 min |
| smoke (twice) | 1 min |
| training (4 runs × 4.096M, concurrent) | 29.5 min |
| evaluation (head to head, fixed opponents, natural records, paired families; about 11M turns) | 28.5 min |

**Checks performed.**
- New tests `tests/test_m47.py` (11), all pass. They check:
  - the γ override changes only γ, and the default stays 0.99;
  - the trainer takes the game's γ and refuses a mismatch;
  - GAE receives the short γ, with bootstrapped truncations and no terminations, in a real `train_shared` run;
  - truncation versus termination in GAE;
  - discount arithmetic and splits;
  - the reward-stream engine reproduces the M4.6a engine's G exactly and deterministically;
  - state variants; policy paths;
  - the timeline and behaviour readouts on a real record.
- Full regression suite: 1,455 passed, 4 skipped, 2 xfailed (the 1,444 of the closeout plus the 11 new tests); legacy
  digest pins pass.
- Smoke 66/66 (twice), training 50/50.
- The L8 +8 lead cell reproduces M4.6c's published early premium.
- The pre-registration was frozen before training (SHA-256 cf7dae62…); one dated amendment (§5.4).

**Files.**
- New code: `lanegame/experiments/rl/m47/` (`common`, `timeline`, `occupancy`, `returns`, `streams`, `delayed`,
  `behaviour`, `train`, `evaluate`, `analysis`, `report`, CLI); `tests/test_m47.py`.
- New documents: this report, [the pre-registration](m4_7_preregistration.md).
- Generated outputs (`results/m47/`): `timeline.json`, `occupancy.json`, `returns.json`, `delayed.json`,
  `smoke.json`, `preregistration.json`, `train_main.json`, `analysis.json`, `report.md`, logs, resource log.
- Raw files (git-ignored) under `results/rl/checkpoints/m47/`: the paired-rollout streams, the four pilot runs (with
  snapshots) and the evaluation records.
- Updated: the roadmap, `README.md`, `docs/README.md`, `docs/m4/README.md`.

**Unchanged:**
- every library module, preset, default and existing checkpoint;
- the mathematical reference (no adopted rule, parameter or objective changed; the γ override is an experimental
  keyword, like C1);
- every earlier report.

Nothing is staged or committed.

**Reproduce:**
`python -m lanegame.experiments.rl.m47 timeline occupancy returns delayed delayed_analyse`, then `smoke prereg`, then
`train --confirm-m47-train` (the user's approval of the budget), then `eval analyse report`.
Add `--workers N` and `--priority below_normal` as needed.
