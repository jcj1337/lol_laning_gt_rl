# M4.8 — Lane-exit evaluation

*2026-10-17. Status: complete, awaiting review. Evaluation only: no training, and no rule, reward, objective (γ 0.99,
continuing time), default, preset or checkpoint changed. Nothing is staged or committed.*

- Protocol: [m4_8_protocol.md](m4_8_protocol.md), frozen after the pilot and before any evaluation run (SHA-256 in
  `results/m48/protocol.json`, with one dated amendment that added a supplementary re-check before it ran).
- Generated tables: `results/m48/report.md`. Code: `lanegame/diagnostics/lane_exit.py` (the readout) and
  `lanegame/experiments/rl/m48/` (the study); tests: `tests/test_m48.py`.

**Terms.**
- **Resource:** the gold of one fresh wave; natural income is about 0.40 resources per turn per seat.
- **Earned gold:** invested + unspent (reference §7.2). A return invests the unspent gold, which leaves earned gold
  unchanged, so spending never counts as earning again.
- **D0:** the raw earned-gold lead at a checkpoint.
- **Δ_h:** the expected change of that lead over the next h turns under the continuation policies (undiscounted,
  exact ledger increments).
- **E_h = D0 + Δ_h:** the **lane-exit economic readout**, the expected earned-gold lead h turns after the checkpoint.
  It is not the discounted G_h of earlier studies, not a critic value, and carries no tail.
- **Lane exit:** an evaluation checkpoint in the continuing game, here turn 94, the median first turn at level 14
  (M4.7). The game does not end there.
- **Package:** a curve's environment played by that curve's matched policy (`ext/baseline:s` or `ext/C1:s`, the M4.6c
  8.192M continuations of the same M4.5c finals).
- **Pending items** of a checkpoint:
  - an absent seat's return;
  - the enemy minion HP alive now on each seat's ledger (its "stock"), which will die collected or lost;
  - the next return of a present seat that is **due** (12 or more turns past its last return).

## Summary

**The measure is useful (the pre-registered classification), with E_20 as the default window.**
- Its accounting, orientation and restoration are exact.
- Against the games' own recorded continuations it is unbiased and on scale (slope 0.99 ± 0.18 at h = 20, mean
  residual −0.14 ± 0.20).
- It moves the economic assessment materially in about two thirds of natural lane-exit states.
- The pre-registered rule selects h = 20, about one reset cycle. By then 95% of the pending items have resolved, and
  the readout stays flat to h = 40.

**What it adds beyond raw gold.** Raw gold at the lane exit counts what has been collected, not what is about to be:
the stacked wave coming to one side, the Recall the other side still owes, a reset just completed or a death.
- At turn 94, 65% of the matched policies' natural states carry an adjustment of at least half a wave with a certain
  sign.
- The typical adjustment (SD 1.16 resources, rollout noise removed) is a third of the typical raw lead (3.5).
- The leader changes in 4% of all states, but in 16% of close ones (|D0| < 0.5).
- Reset status, HP and the wave explain about 80% of the adjustments (cross-validated); the raw lead itself explains
  2%. A seat that has already reset gains 0.7–1.2 resources on average.
- The adjustments last: E_20 predicts the recorded lead at turns 150 and 200 better than D0 does.
- XP, HP, absence and the wave stay beside the readout. In 21% of states gold and XP or HP point different ways, and
  there the readout gives no overall verdict.

**A worked wave/Recall example** (§3.2; baseline seed 0, validation game 2, turn 52):
- X has just come back from a Recall with full HP. An enemy wave of 13 HP sits next to its tower.
- Y is 13 turns past its last reset, at 64% HP, holding 6 unspent resources.
- Raw gold says Y leads by 0.18, and Y also leads by 8 XP.
- E_10 = X ahead by **+2.44 ± 0.24**. Over those 10 turns X collects 5.0 of farm against Y's 2.5, and Y spends
  1.9 turns away on its Recall. X's lead rises in 98% of continuations.
- In the game's own continuation, Y recalled seven turns later and X led by 3.65 after ten turns.

**Baseline and C1: no conclusion changes; one earlier statement is refined.**
- At the lane exit the two packages are indistinguishable:
  - the share of material adjustments (64% and 66%);
  - lead conversion over the next 10–40 turns (−0.03 to −0.05 in both);
  - the same states continued by either package (the leader gains +0.02 ± 0.04 under C1);
  - the value of a completed reset (+0.56 and +0.62).
- **M4.7's lead-size comparison** at turn 94 is unchanged (+0.31 ± 0.46 with E_20, against +0.35 ± 0.40 raw).
- **M4.7's 50 → 94 conversion slopes** keep their C1 − baseline contrast (+0.25 ± 0.13). Their levels shift: about
  half of the baseline's apparent shrinkage was pending consequences (−0.20 → −0.10 ± 0.10), and C1's leads grow
  (+0.04 → +0.15 ± 0.09).
- **M4.7's horizon pilot:** its one seed with a lane-phase gain loses it (+0.26 ± 0.17 → +0.17 ± 0.18), which
  reinforces "no reliable lane-phase gain".
- The baseline stays the recommended configuration (§5).

**What remains uncertain.**
- The readout is an expectation under the continuation policies. Two in-distribution alternatives barely moved it,
  but unfamiliar styles were not tested.
- It values gold only. XP, HP and positional advantages (tower damage, vision) remain separate and are not converted.
- Every window also contains new resets, since each seat resets about once per 20 turns.
- Gold leaders with an HP or XP deficit lost somewhat more in their recorded continuations than predicted
  (z ≈ 2.8 in the validation cases; the same direction, not significant, in the natural states). The cause is not
  identified.
- Population claims rest on six training seeds per curve.

**Next step:** step 3 of the agreed sequence, the design of Bayesian opponent modelling (approved direction, not
designed). The lane-exit readout (E_20 with the non-gold panel) is recommended as the lane-phase economic measure for
its pilot and for M5. No default changes.

**Answers to the study's questions.**
1. *How often does raw checkpoint gold give a materially different economic assessment?* In about two thirds of
   natural turn-94 states (64–73% across windows and curves; 54–69% at turns 50, 84 and 104, where K is smaller).
2. *How large and how certain?*
   - The true adjustment has an SD of 1.16 at h = 20, and the middle half of states moves by −0.76 to +0.75.
   - With K = 128 a state's E_20 is known to ±0.39 (median 95% half-width).
   - The leader flips in 4% of all states and 16% of close ones.
3. *Which wave/Recall situations explain them?* A seat that has just come back (+1.2 at h = 20, 23% of states), an
   asymmetric due reset (+0.7, 16%), a death (−1.2 within five turns, then about 0), an enemy-stock asymmetry, and HP
   differences that decide who resets next.
4. *How stable across windows and continuation policies?*
   - Category means are flat after h = 20.
   - Per state, the change from 20 to 40 is 0.22 on average.
   - Other continuation policies correlate 0.96–0.98 with the default.
   - At most 24% of material classifications change between windows 10 and 20 or between policies, mostly near the
     threshold.
5. *Do baseline and C1 differ meaningfully during this phase?* No, on any lane-exit readout.
6. *Does the new measure alter an earlier conclusion?* It refines M4.7's 50 → 94 conversion levels without changing
   the curve contrast, and it reinforces M4.7's "no reliable lane-phase gain". It changes no recommendation.

---

## 1. The readout

At a checkpoint state s (seat A's orientation):

$$E_h(s)=D_0(s)+\mathbb E\Bigl[\sum_{k<h}r_{A,k}\Bigr],\qquad D_0=(S_A+g_A)-(S_B+g_B).$$

**What it is.**
- The expected earned-gold lead h turns after the checkpoint, under the continuation policies.
- By default those are the policies that produced the state, in both seats, with CRN sampling. It estimates outcomes
  under those policies, not optimal play.
- It is estimated from K continuations of the same state, each with its own chance and sampling streams.
- The windows h = 5, 10, 20 (candidates), 30 and 40 (diagnostic) are prefixes of the same 40-turn continuations,
  fixed in advance. No continuation is stopped because one side looks favourable.

**Accounting.** Exact in ledger units in every continuation:
- lead(end of window) − D0 = Σ of the turns' reward units;
- per seat and turn, earned = farm + plate gold + bounty.

**Reported beside it, never folded in** (`lane_exit.describe`, `window_state`):
- levels and the XP difference, at the checkpoint and after h turns;
- HP, absence status (present, away after a Recall, dead) and turns still away;
- turns since each seat's last return;
- the wave: stock, the part of it at the seat's own −1 or deeper ("near"), crashed masses, freezes;
- the within-state uncertainty (the standard error over continuations).

There is no exchange rate between gold, XP, HP and position. Where they point different ways, the readout gives no
overall "who is winning" verdict (§3.3, case 5).

## 2. Design (protocol §3–§4)

| | Stage 1 (validation) | Stage 2 (application) |
|---|---|---|
| states | 469 checkpoints of 192 new self-play games (16 per package), turns 30–120, selected by checkpoint-only case criteria (up to 4 per package × case × orientation, distinct games, fixed generator) | all 48 games per package of the M4.6d natural records, replayed and verified turn by turn; t0 = 94 (primary), 84 and 104 (sensitivity), 50 (M4.7 replication): 2,304 checkpoints |
| continuations | own package, K 128 × 40 turns; mirrored copies of the 96 returner states; controlled variants of the 48 due states | own package, K 128 at t0 = 94 (64 at 84/104, 32 at 50); at t0 = 94 also the other curve's package on every state, an alternative policy (the same curve's seed s + 1 in both seats) on 144 states, and the "already reset" variant on the 168 due states |
| what is checked | exact accounting, orientation and restoration; the readout against each case's mechanism and recorded events | how often and why raw gold misleads; stability across windows, policies and anchors; calibration against the records; baseline vs C1 |

**Precision.** The pilot measured the within-state SD of Δ_h: median 0.82, 1.35, 2.16 and 2.94 at h = 5, 10, 20 and 40.
K = 128 was chosen for a 95% half-width of 0.25 at h = 10 in the median state. It was never extended.

**Budget.** 14.6M simulated turns plus the supplementary re-check; runtime in §7.

## 3. Stage 1: validation

### 3.1 Exactness

| check | result |
|---|---|
| accounting: lead change = Σ reward units, per continuation | exact in every Stage 1 continuation (60,032 own, 12,288 mirrored, 12,288 variants) and every Stage 2 continuation; the code raises on any residual |
| orientation: the mirrored state with mirrored chance and swapped sampling streams | the exact negation in all 12,288 mirrored continuations (max \|r + r_mirror\| = 0) |
| restoration | every one of the 2,304 natural checkpoints replays its stored M4.6d record bit for bit (invested, unspent, XP, absence, reward on every turn) |
| tests | `tests/test_m48.py` (18; §7) |

### 3.2 A worked wave/Recall example

The rule for the example was fixed before any continuation was read: the first returner state (in environment,
seed, game, turn order) whose returner trails by less than 2 resources. It is baseline seed 0, validation game 2,
turn 52, and the returner X is seat B.

| at the checkpoint | X (seat B) | Y (seat A) |
|---|---|---|
| status | present, back from a Recall 1 turn ago, at its own −1 | present at its own +1, 13 turns past its last return |
| HP | 100% | 64% |
| earned (invested + unspent) | 18.91 (18.73 + 0.18) | 19.09 (13.10 + 5.99) |
| level / XP | 10 / 182 | 10 / 190 |
| enemy minion HP alive on its ledger (at its own −1 or deeper) | 21 (13): Y's 13-HP wave is next to X's tower | 8 (0) |

Raw gold says Y leads by 0.18, and Y is also 8 XP ahead. Y has not reset for 13 turns, holds 6 unspent resources and
is at 64% HP.

| h | E_h (X's lead) | 95% ± | pending items resolved | X's XP lead | the game's recorded change |
|---|---|---|---|---|---|
| 5 | +2.09 | 0.13 | 45% | +11.8 | +1.63 |
| 10 | **+2.44** | 0.24 | 83% | +13.6 | +3.65 |
| 20 | +1.78 | 0.31 | 100% | +7.2 | +0.39 |
| 40 | +2.01 | 0.51 | 100% | +8.6 | +1.91 |

**Accounting over the first 10 turns** (means per continuation, X / Y):
- **Farm:** 4.99 / 2.52 from 6.00 / 3.77 available. X collects 82% of its checkpoint stock, Y 35% of its own.
- **Absence:** Y is away 1.91 turns (0.88 Recalls; back by turn 10 in 83% of continuations), X 0.20.
- **Farm lost while away:** 0.03 / 0.31.
- **Bounty and plates:** 0.13 / 0.02 and 0.05 / 0.02.

So X is ahead in E_h because it has already paid for its Recall and the stacked wave is coming to it, while Y still
has to pay for its own. Once the stack is collected, X leads in XP as well (+13.6 at h = 10).

**The game's own continuation** (one realized sample, not used to compute E_h):
- turn 52: X collects 1.30 from the stack;
- turn 59: Y recalls, and is back after turn 60;
- turn 63: X recalls;
- X's lead is +1.63 at turn 57 and +3.65 at turn 62.

E_h is an expectation under the policies; the realized path is one of the outcomes it averages over. In 98% of the
continuations X's lead at h = 10 is above its checkpoint lead.

### 3.3 The cases (folded to the focal seat X)

Δ_h is X's expected gain; "material" = |Δ_h| ≥ 0.5 with its 95% interval excluding 0. The recorded change is the
validation game's own continuation (one sample per state), averaged over the states.

| case | n | X's raw lead | Δ5 | Δ10 | Δ20 | Δ40 | material @10 / @20 | pending resolved by 10 / 20 | recorded Δ10 |
|---|---|---|---|---|---|---|---|---|---|
| 1–2 returner (X back or coming back; stack at X; Y due) | 96 | −2.80 | +1.84 ± 0.15 | **+2.55 ± 0.19** | +2.13 ± 0.16 | +2.13 ± 0.18 | 98% / 99% | 66% / 95% | +2.70 ± 0.38 |
| 3a lost (X dead ≥ 2 more turns; enemy wave at X's side) | 85 | −3.43 | −0.20 ± 0.28 | +0.91 ± 0.25 | +0.88 ± 0.24 | +0.80 ± 0.26 | 73% / 73% | 87% / 98% | +0.81 ± 0.39 |
| 3b denied (Y holds a freeze on X's wave) | 96 | +0.92 | −0.77 ± 0.19 | −1.08 ± 0.21 | −0.96 ± 0.18 | −0.92 ± 0.20 | 72% / 79% | 80% / 95% | −0.99 ± 0.34 |
| 4 stable (both present, healthy, recently reset, balanced stocks) | 48 | −0.70 | −0.01 ± 0.13 | −0.03 ± 0.16 | −0.04 ± 0.14 | −0.17 ± 0.15 | 38% / 35% | 100% / 100% | −0.30 ± 0.32 |
| 5 opposing (X leads in gold, trails in XP or HP) | 96 | +2.97 | −1.56 ± 0.17 | −1.73 ± 0.23 | −1.15 ± 0.21 | −1.23 ± 0.22 | 91% / 79% | 80% / 97% | −2.51 ± 0.50 |
| controlled base: due (A due, ≥ 2 unspent) | 48 | +0.82 | −0.40 ± 0.34 | −0.60 ± 0.35 | −0.55 ± 0.28 | −0.57 ± 0.30 | 65% / 58% | 61% / 91% | −0.46 ± 0.52 |

**Orientation.** Case 2 (X = B) is case 1 mirrored. The two orientations give +2.48 ± 0.25 and +2.62 ± 0.29 at
h = 10. The other two-orientation cases also agree within their intervals; the largest gap is denied, at −1.24 ± 0.29
against −0.92 ± 0.29. Baseline and C1 agree within noise in every case; the largest gap is 0.34, in the 48 stable
states.

**Event accounting over 10 turns** (X / Y means per continuation):

| case | farm | farm available | farm lost while away | bounty X−Y | plates X−Y | turns away | returns |
|---|---|---|---|---|---|---|---|
| returner | 5.01 / 2.62 | 6.33 / 3.84 | 0.06 / 0.39 | +0.34 | −0.18 | 0.30 / 1.81 | 0.14 / 0.68 |
| lost | 4.14 / 2.55 | 6.92 / 3.80 | 1.74 / 0.44 | +0.37 | −1.05 | 2.49 / 1.80 | 1.02 / 0.61 |
| denied | 3.37 / 4.45 | 4.55 / 5.73 | 0.29 / 0.25 | −0.13 | +0.12 | 1.33 / 0.96 | 0.52 / 0.44 |
| stable | 3.86 / 3.85 | 4.87 / 4.86 | 0.15 / 0.15 | −0.01 | −0.03 | 0.85 / 0.86 | 0.36 / 0.37 |
| opposing | 2.94 / 4.16 | 4.52 / 5.43 | 0.58 / 0.15 | −0.27 | −0.24 | 2.13 / 0.60 | 0.81 / 0.21 |
| due | 3.48 / 4.01 | 4.77 / 5.26 | 0.34 / 0.26 | −0.05 | −0.01 | 1.53 / 1.05 | 0.64 / 0.43 |

Each adjustment has the accounting explanation its case predicts:
- **Returner.** X collects the stacked wave (stock 21.7 HP, 12.1 of it at X's side, 79% collected), while Y spends
  1.8 turns away on the reset it still owed. X's XP deficit of −22 closes to +1.5 by h = 10.
- **Lost.** The stack at X's tower dies while X is dead, and Y takes plates.
  - Over 10 turns X loses 1.74 of farm while away, and Y takes 1.05 more plate gold.
  - This puts X at −0.20 after five turns.
  - Then X is back with full HP and its gold invested, Y has to reset, and X ends +0.9 by h = 10. A death here is costly but is also a completed reset; the raw lead counts only the cost.
- **Denied.** The frozen-on seat farms 1.1 less.
- **Stable.** Symmetric on average. Individually, 38% of these states still carry a material adjustment (in either
  direction, from the wave and HP details within the criteria), so even ordinary-looking states are not always
  "settled".
- **Opposing.** A gold leader at 40% HP with 8.5 unspent has to reset (2.1 turns away) and concedes bounty and plates.
  Its lead shrinks by 1.7 by h = 10 and stays 1.2 smaller at h = 40. Raw gold, XP and HP disagree at the checkpoint;
  after the reset the XP and HP edges have mostly turned into gold.

### 3.4 Controlled comparisons: an equal raw lead, a different reset status

Variants of seat A on the 48 due states (A is 15 turns past its last return, with 7.5 unspent and 65% HP). Each
variant has exactly the same D0 because earned = invested + unspent:
- **A_fresh:** A has just come back from a reset (unspent invested, full HP, at its tower).
- **A_away1:** A has just recalled and is away one more turn.

| variant − original | E5 | E10 | E20 | E40 | material @10 |
|---|---|---|---|---|---|
| A_fresh | +0.56 ± 0.20 | **+0.87 ± 0.18** | +0.62 ± 0.13 | +0.66 ± 0.14 | 67% |
| A_away1 | +0.20 ± 0.26 | +0.49 ± 0.23 | +0.36 ± 0.14 | +0.38 ± 0.17 | 63% |

Raw gold scores the three states identically. Under the policies, having already reset is worth +0.6 to +0.9 of
short-run gold, and being one turn from return about half of that. The accounting for A_fresh at h = 10:
- A farms +0.48 more and loses 0.25 less while away;
- A is away 1.07 fewer turns;
- B collects 0.14 less bounty and 0.12 fewer plates.

Baseline and C1 agree (+0.90 and +0.83 at h = 10).

### 3.5 Predictions against recorded continuations (calibration)

The M4.6d records continue every natural checkpoint for 300 more turns, which gives one realized sample of what
happened next. The prediction Δ_h was computed from fresh continuations and never saw that sample. Natural states at
turn 94 (n = 576):

| h | slope of realized on predicted (ideal 1) | mean residual (ideal 0) | variance of the realized change explained | MSE of D0 / E_h for the lead at t0 + h |
|---|---|---|---|---|
| 5 | 1.00 ± 0.08 | +0.01 ± 0.08 | 60% | 2.64 / 1.06 |
| 10 | **0.99 ± 0.13** | −0.02 ± 0.14 | 38% | 5.04 / 3.11 |
| 20 | **0.99 ± 0.18** | −0.14 ± 0.20 | 19% | 7.05 / 5.72 |
| 30 | 0.91 ± 0.20 | −0.12 ± 0.23 | 13% | 8.89 / 7.71 |
| 40 | 1.00 ± 0.21 | +0.10 ± 0.26 | 13% | 11.49 / 9.93 |

At the sensitivity anchors (K 64, so the predictions themselves are noisier):
- turn 104: slopes 0.94, 1.09, 1.06 at h = 5, 10, 20, with residuals within ±0.13;
- turn 84: slopes 0.98, 0.91 and 0.80 ± 0.18 at h = 20, with residuals within ±0.05.

The h = 20 slope at turn 84 is the lowest reading and still overlaps the protocol's band [0.8, 1.2].

- **The predictions are unbiased and on scale.** A predicted change of +1 is followed on average by a realized
  change of +1.
- **What E_h cannot predict is genuine future randomness.** The realized change over 20 turns has an SD of about 2.6;
  E_20 explains a fifth of its variance, and the rest is trades, ganks, rolls and choices still to come.
- **The adjustments persist beyond the window.** E_20 at turn 94 predicts the recorded lead at turn 150 better than
  D0 does: correlation 0.785 against 0.754, and the mean squared error falls by 1.46 ± 0.76. It does so at turn 200
  too (−1.60 ± 0.83); at turn 250 the gain is no longer significant (−1.51 ± 0.98). The pending consequences are
  mostly permanent level shifts of the lead, not transients that later reverse.
- **Validation cases.** The recorded changes agree with the predictions in five of six cases (§3.3, last column). The
  exception is the opposing case: −2.51 ± 0.50 recorded against −1.73 ± 0.23 predicted at h = 10 (z ≈ 2.8; at h = 20,
  −1.84 against −1.15).
  - The natural states that meet the same criterion lean the same way at h = 10, though each anchor's interval
    includes 0:

    | anchor | states | recorded − predicted |
    |---|---|---|
    | turn 94 | 101 | −0.43 ± 0.43 |
    | turn 104 | 103 | −0.46 ± 0.43 |
    | turn 50 | 86 | −0.28 ± 0.40 |
    | turn 84 | 90 | +0.16 ± 0.42 |

  - Across all 576 states the residual has no relation to the lead, the HP difference, the XP difference or the
    prediction itself (every slope within noise).
  - In theory there can be no bias. The replays are exact, the policies are functions of the state, and the
    selection reads only the checkpoint, so the recorded and the simulated continuations are draws from the same
    process.
  - The likeliest explanation is chance across many subgroup comparisons, but a modest subgroup effect is not
    excluded. **Treat E_h for a gold leader with an HP or XP deficit as possibly understating its coming loss by a few
    tenths of a resource.** The cause is not identified.

### 3.6 Windows and when the pending consequences resolve

Coverage of the pending items in the 576 natural turn-94 states (the share of continuations in which the item has
resolved by h):

| pending item | states | by 5 | by 10 | by 20 | by 30 |
|---|---|---|---|---|---|
| an absent seat's return | 106 | 100% | 100% | 100% | 100% |
| the checkpoint's enemy stock (both ledgers) | 576 | 83% | 100% | 100% | 100% |
| a due seat's next return | 316 | 27% | 60% | 90% | 98% |
| all items of the state | 576 | 54% | 78% | **95%** | |

- **The wave resolves within 5–10 turns and an absence by definition; the slow item is the due reset.** A seat that
  owes a Recall takes it within 20 turns in 90% of continuations.
- **New consequences start immediately.** Each seat completes about one reset per 20 turns (0.53 returns per
  continuation by h = 5, 1.02 by 10, 2.03 by 20). Every window therefore also contains resets that were not pending
  at the checkpoint. That is why the readout of a returner state peaks at h = 10 (+2.55) and settles at h = 20 (+2.13):
  by h = 20 the returner's own next reset has mostly happened too.
- **From 20 to 40 the readout is flat.** In every case and natural category, the mean change from h = 20 to 40 is
  within ±0.2. Per state, the mean |Δ_40 − Δ_20| is 0.22, against 0.34 from 10 to 20 and 0.43 from 5 to 10.

### 3.7 Sensitivity to the continuation policy

On the same turn-94 states, with common random numbers:

| continuation | states | h | mean difference | mean \|difference\| | correlation | material classification changes (among material states) | leader changes |
|---|---|---|---|---|---|---|---|
| alternative policy: the same curve's seed s + 1 in both seats | 144 | 10 | −0.03 ± 0.04 | 0.19 | 0.98 | 8% | 0% |
| | | 20 | −0.04 ± 0.05 | 0.25 | 0.96 | 24% | 4% |
| the other curve's package | 576 | 10 | −0.01 ± 0.02 | 0.19 | 0.98 | 8% | 2% |
| | | 20 | +0.01 ± 0.03 | 0.24 | 0.97 | 21% | 2% |

The readout is a property of the state far more than of which of these policies continues it. The classification
changes at h = 20 are mostly states near the 0.5 threshold. Both alternatives are in-distribution policies of the same
lineage; opponents with unfamiliar styles were not tested.

### 3.8 Leader reversals and their accounting

A material leader reversal means E_h's interval excludes 0 on the side opposite to the raw lead.

| states | h = 10 | h = 20 | dominant accounting component (h = 20) |
|---|---|---|---|
| validation cases | 69 of 469 | 49 of 469 | farm available 44, plates 3, farm lost while away 1, bounty 1 |
| natural, turn 94 | 43 of 576 (7.5%) | 25 of 576 (4.3%) | farm available 20, plates 3, farm lost while away 2 |

Every reversal has an accounting explanation, and in almost all of them it is farm: one seat collects more of the
wave that reaches it in the window. The natural reversals at h = 20 are mostly "fresh" states (13 of 25: one seat has
just come back) and due asymmetries.

For example, take the natural state baseline seed 0, game 2, turn 94 ("fresh" category). Seat A leads by 1.04 in raw
gold, but it is at 28% HP with 4.36 unspent, 10 turns past its last return. Seat B has just come back with full HP,
and 20 HP of enemy minions are alive on B's ledger (8 on A's). Over 20 turns:
- A farms 2.06 less;
- A loses 0.84 more farm while away;
- A concedes 0.77 more bounty and 0.70 more plates.

E_20 = −2.48 ± 0.29: B leads by about two and a half waves.

### 3.9 Verdict on the method (protocol §7)

| criterion (protocol §7) | result | met |
|---|---|---|
| V1–V3 exact | accounting, orientation and restoration exact in every continuation and replay | yes |
| calibration at the default window (h = 20) | slope 0.99, interval [0.80, 1.17]; mean residual −0.14 ± 0.20 | yes |
| the Stage 1 adjustments are accounted for by recorded events matching the cases' mechanisms | yes in every case (§3.3); the opposing case's recorded mean deviates from its prediction, and natural states lean the same way (§3.5) | yes, with that caveat |
| it adds information beyond D0 | at turn 94, 65% of states carry a material, certain adjustment at h = 20; true SD 1.16 (thresholds: 10%, 0.25) | yes |
| not too noisy | median 95% half-width 0.14 / 0.24 / 0.39 at h = 5 / 10 / 20 (threshold 0.5) | yes |
| stable | at most 24% of material classifications change between continuation policies or between windows 10 and 20 (threshold 50%) | yes |

**Pre-registered classification: useful.**

**Default window (protocol §6): E_20.**
- h = 5 fails coverage (54%) and residual drift (dead +1.15, away after a Recall +0.28, fresh +0.25).
- h = 10 fails coverage (78%: due resets are 60% resolved) and passes the rest.
- h = 20 meets all three criteria: coverage 95%, no category with excess drift above 0.15, median ±0.39.
- No category with 30 or more states is an exception at h = 20.

## 4. Stage 2: the matched baseline and C1 policies at the lane exit

The states are the 576 natural turn-94 checkpoints: 48 games × 6 seeds × 2 curves, each curve's own self-play.
- **Levels:** the lower level averages 13.4. Both seats are at level 14 or above in 36% of states, at least one seat
  in 82%; seat levels p10/p50/p90 are 13/14/14.
- **Raw lead:** |D0| averages 3.47; it is under 0.5 in 7.5% of states.

### 4.1 How often raw checkpoint gold gives a materially different assessment

"Material" = |Δ_h| ≥ 0.5 with the state's 95% interval excluding 0. Training seeds are the units: six per curve,
t intervals, C1 − baseline paired by seed.

| h | material: baseline | C1 | C1 − baseline | leader reversals | SD of the true adjustment | mean \|Δ_h\| | median 95% ± per state |
|---|---|---|---|---|---|---|---|
| 5 | 67% | 66% | −1 ± 6 pts | 6.9% | 1.26 | 0.95 | 0.14 |
| 10 | 73% | 68% | −5 ± 6 pts | 7.5% | 1.40 | 1.08 | 0.24 |
| **20** | **64%** | **66%** | +2 ± 6 pts | **4.3%** | **1.16** | **0.92** | 0.39 |
| 40 | 65% | 66% | +1 ± 4 pts | 3.6% | 1.22 | 0.98 | 0.53 |

- **Raw gold at the lane exit is usually not the economic position.** In about two thirds of natural turn-94 states
  the expected lead 20 turns later differs from the raw lead by at least half a wave, with a certain sign.
- **The typical adjustment is large relative to the lead.** The between-state SD of the true adjustment is 1.16
  resources (rollout noise removed), about a third of the mean |D0| of 3.5. The middle half of states moves by
  −0.76 to +0.75.
- **The leader rarely changes, except in close states.** That happens in 4.3% of states at h = 20, by the size of
  the raw lead:
  - 16% when |D0| < 0.5 (where E_20 gives a certain sign in 63% of states);
  - 11% when |D0| is between 0.5 and 2;
  - 0.3% when |D0| > 2.
- **The same at turns 84, 104 and 50.** The SD of the true adjustment at h = 20 is 1.13 (turn 84), 1.15 (94),
  1.11 (104) and 1.20 (50). The share of material, certain adjustments is 61% (84), 65% (94), 59% (104) and 54% (50);
  it is lower where K is smaller, because the criterion includes certainty. C1 − baseline is within ±0.07 at every
  anchor.

### 4.2 Which situations explain the adjustments

Each state is classified by its pending situation (priority order) and folded to the category's focal seat X:

| category (turn 94) | share | X's raw lead | Δ5 | Δ10 | Δ20 | Δ40 | material @20 | Δ40 − Δ20 |
|---|---|---|---|---|---|---|---|---|
| one seat dead (X) | 9% | −3.72 | −1.24 ± 0.51 | −0.19 ± 0.45 | −0.05 ± 0.36 | −0.18 ± 0.39 | 69% | −0.13 |
| one seat away after a Recall (X) | 8% | −0.44 | −0.05 ± 0.14 | +0.25 ± 0.16 | +0.25 ± 0.17 | +0.14 ± 0.21 | 38% | −0.12 |
| X back ≤ 2 turns ago, the other present | 23% | −1.17 | +0.99 ± 0.15 | +1.34 ± 0.20 | **+1.16 ± 0.17** | +1.15 ± 0.18 | 77% | −0.01 |
| due asymmetry (Y due, X back ≤ 6 turns ago) | 16% | −0.63 | +0.84 ± 0.21 | +0.97 ± 0.25 | **+0.70 ± 0.20** | +0.74 ± 0.21 | 61% | +0.04 |
| enemy-stock asymmetry ≥ 16 HP (X: the larger stock) | 2% | −2.43 | +0.49 ± 0.42 | +0.62 ± 0.73 | +1.27 ± 0.56 | +1.10 ± 0.61 | 70% | −0.18 |
| both away (X = A) | 2% | +1.85 | +0.50 ± 0.73 | +0.29 ± 0.84 | +0.23 ± 0.96 | +0.22 ± 1.05 | 80% | −0.02 |
| ordinary (X = A) | 41% | −0.16 | +0.04 ± 0.13 | −0.02 ± 0.15 | −0.06 ± 0.12 | −0.05 ± 0.13 | 63% | +0.01 |

- **The seat that has already reset gains 0.7–1.2 resources.** That is a freshly returned seat (23% of states,
  +1.16 by h = 20) or a seat that reset recently while the other is due (16%, +0.70). This is the motivating example's
  situation in its common natural form. Over 10 turns the fresh seat farms 1.19 more and spends 0.9 fewer turns away.
- **A death is a costly but completed reset.** The readout for the dead seat is −1.24 after five turns. Over the
  first ten:
  - the wave at its tower dies while it is away (1.21 of farm lost);
  - the opponent takes 0.90 more plates.
  It then regains nearly all of it as the opponent has to reset in turn: −0.05 by h = 20. The raw lead already
  includes the bounty, and on average it is about right 20 turns later. It is not right after 5.
- **Ordinary states carry adjustments in both directions.** No seat is away, freshly back or facing a due asymmetry,
  yet 63% of them are individually material at h = 20, in both directions, from HP and wave differences. Their
  average is zero.

**What explains the variance across states.** 5-fold cross-validated R² of Δ̂_h on the checkpoint's features (seat A
minus seat B); in-sample R² is within 0.02. Rollout noise is 1–3% of the variance.

| h | pending features (reset status, unspent gold, stocks, freeze) | strength features (investment, level, HP) | D0 itself | all |
|---|---|---|---|---|
| 10 | 0.57 | 0.39 | 0.02 | 0.85 |
| 20 | 0.64 | 0.29 | 0.02 | 0.80 |

A finer split (descriptive, not pre-registered; cross-validated R² at h = 20):

| feature group | R² |
|---|---|
| reset status (absence, freshness, due, unspent gold) | 0.39 |
| wave (stocks, near stock, crashed mass, freeze) | 0.48 |
| HP | 0.17 |
| investment and level | 0.23 |
| reset + wave + HP together | 0.80, the same as all features |

Investment and level add nothing once reset status, the wave and HP are known.

- **The adjustment is about what is pending, not about the size of the lead.** D0 explains 2%: leads at the lane
  exit do not snowball, and they revert only slightly (3–5%, §4.4b).
- At h = 10 the largest coefficients are:
  - being away after a Recall or dead: about +3.2 for the absent seat, net of −0.8 per turn still away;
  - the enemy stock: +0.77 per resource of minions alive on the seat's ledger;
  - HP: +2.6 per bar of a present seat's HP lead.
- HP acts mostly through who has to reset next; it is a strength feature only in name.

### 4.3 Stability

- **Across windows** (natural turn-94 states):

  | windows | mean \|change\| per state | material classification changes |
  |---|---|---|
  | 5 → 10 | 0.43 | 13% |
  | 10 → 20 | 0.34 | 18% |
  | 20 → 40 | 0.22 | 14% |

  Category means change by at most 0.2 after h = 20 (§4.2, last column).
- **Across continuation policies:** §3.7. The correlation is 0.96–0.98 and the mean differences are within ±0.04.
- **Across anchors:** the category means at turn 84 and 104 match turn 94 (at h = 20: fresh +0.99 and +1.11, due
  asymmetry +0.62 and +0.76, away after a Recall +0.40 and +0.43, dead +0.03 and −0.11, ordinary −0.04 and −0.09).
  The variance components match too (cross-validated R² at h = 20: pending 0.59 / 0.64 / 0.60 and all features
  0.78 / 0.80 / 0.74 at turns 84 / 94 / 104), and so does the calibration (§3.5).

### 4.4 Baseline and C1

State distributions (each curve's own games) are kept apart from outcomes from matched states (the same states
continued by both packages). Six training seeds per curve; C1 − baseline paired by seed.

**(a) State distributions at turn 94.**

| quantity | baseline | C1 | C1 − baseline |
|---|---|---|---|
| \|D0\| (raw lead) | 3.29 ± 0.21 | 3.64 ± 0.46 | +0.35 ± 0.40 |
| \|E_10\| | 3.30 ± 0.20 | 3.60 ± 0.45 | +0.30 ± 0.41 |
| \|E_20\| | 3.27 ± 0.20 | 3.58 ± 0.45 | +0.31 ± 0.46 |
| share of states with a material adjustment at h = 20 | 64% | 66% | +2 ± 6 pts |

The two curves produce the same lane-exit situations:
- category shares within ±2 points (for example, fresh 24% and 23%, ordinary 41% and 41%);
- |XP difference| 13.0 and 12.9;
- |HP difference| 0.29 and 0.29;
- the lower level 13.3 and 13.4;
- seats away 10% and 10%.

**(b) Lead conversion after the lane exit** (each package continuing its own states; slope of Δ_h on D0):

| h | baseline | C1 | C1 − baseline |
|---|---|---|---|
| 10 | −0.051 ± 0.025 | −0.049 ± 0.059 | +0.002 ± 0.060 |
| 20 | −0.042 ± 0.024 | −0.038 ± 0.047 | +0.004 ± 0.058 |
| 40 | −0.035 ± 0.024 | −0.026 ± 0.055 | +0.008 ± 0.053 |

Leads at the lane exit shrink by 3–5% over the next 10–40 turns in both curves.

**(c) Matched states.** All 576 states were continued by both packages, each in its own environment. The measured
quantity is the C1 package minus the baseline package on common random numbers. It contrasts the packages
(environment plus its trained policy); it is not the isolated effect of the investment curve.

| h | the leader's extra gain under C1 | slope difference | mean difference |
|---|---|---|---|
| 10 | −0.02 ± 0.03 | −0.005 ± 0.007 | −0.01 ± 0.02 |
| 20 | +0.02 ± 0.04 | +0.007 ± 0.011 | −0.01 ± 0.03 |
| 40 | +0.04 ± 0.07 | +0.016 ± 0.018 | −0.02 ± 0.06 |

**(d) The value of an already completed reset** (A_fresh − original, turn-94 states where A is due):
- baseline +0.56 ± 0.17 and C1 +0.62 ± 0.07 at h = 20;
- the difference is +0.06 ± 0.20.

**Reading.** At the lane exit the two packages are the same game on every readout:
- the share and size of the pending adjustments;
- the lead's short-run conversion;
- the outcome of identical states;
- the value of having reset.

C1's documented effect (persisting early leads, the mid-game lead premium) does not show here: at turn 94 and over
the following 40 turns it is not measurable.

### 4.5 Earlier conclusions re-checked

The raw readouts below reproduce M4.7's published values exactly from the same records and games, so any change comes
from the readout alone.

| earlier conclusion | raw (as published) | with the lane-exit readout | does the conclusion change? |
|---|---|---|---|
| **M4.7 §2: \|earned lead\| at turn 94**, baseline / C1 / C1 − baseline | 3.29 / 3.64 / +0.35 ± 0.40 | E_20: 3.27 / 3.58 / +0.31 ± 0.46 | **no**: no reliable difference in lead size at the lane exit |
| **M4.7 §2: conversion slope of the lead from turn 50 to 94**, baseline / C1 / C1 − baseline | −0.20 ± 0.07 / +0.04 ± 0.17 / +0.24 ± 0.15 | E_20 at both ends: −0.10 ± 0.10 / **+0.15 ± 0.09** / +0.25 ± 0.13 (E_10: −0.09 ± 0.11 / +0.13 ± 0.11 / +0.22 ± 0.15) | **the contrast stands, the levels shift** (below) |
| **M4.7 pilot primary P1** (short arm's lane lead, head to head), seed 0 / seed 1 | +0.26 ± 0.17 / −0.09 ± 0.17 | E_20: +0.17 ± 0.18 / −0.14 ± 0.17 (E_20 − raw: −0.09 ± 0.06 / −0.05 ± 0.06) | **reinforced**: no reliable lane-phase gain in either seed (below) |

- **M4.7's conversion slope.** M4.7 said "under the baseline a natural lead at turn 50 shrinks by about a fifth by
  turn 94; under C1 it persists".
  - With the readout at both ends, about half of the baseline's shrinkage was pending consequences, and its interval
    now includes 0.
  - C1's leads grow by about a seventh, with the interval excluding 0 at E_20.
  - The C1 − baseline contrast, the part M4.7 drew its conclusion from, is unchanged.
  - The corrected reading: in economic-readout terms, early leads change little under the baseline and grow somewhat
    under C1 between turns 50 and 94.
  - This is a correction of a historical statement and is recorded here and in the roadmap's §AU; the M4.7 report is
    not edited.
- **M4.7's pilot primary (supplementary re-check, protocol amendment).**
  - The 4,096 head-to-head games were replayed exactly (deviation 1·10⁻⁷, float32 storage) and continued K 8 × 20
    turns from turn 94 by the same pair.
  - In seed 0 the readout lowers the short arm's lead by 0.07 ± 0.05 within 5 turns and by 0.09 ± 0.06 within 20.
    - The 5-turn part is consequences still pending at turn 94: they favoured the short arm at that instant and went
      the control's way.
    - Over 20 turns the change also contains the arms' ordinary income difference. In M4.7 the short arm earned less
      after the lane phase, about −0.003 per turn in this seed.
  - The seed's E_20 lead, +0.17 ± 0.18, no longer excludes 0 and falls further below the benefit threshold of +0.25.
  - Seed 1 is unchanged.
  - M4.7's classification (outcome 5) rests on the fixed-opponent regression, which this does not touch. Its
    statement "no reliable lane-phase gain" is now true of both seeds individually.
- **Unchanged:** the C1-versus-baseline lane-phase equivalence (M4.7 §2) and the closeout's recommendation of the
  baseline.

## 5. Recommendation

**Default readout: E_20, presented beside the raw lead and the non-gold dimensions.** The pre-registered rule selects
h = 20. It is about one reset cycle: the wave and absences resolve by h = 5–10, and a due seat's reset by h = 20 in 90%
of continuations. After h = 20 the readout is flat to h = 40. A lane-exit assessment should read:

| item | content |
|---|---|
| raw lead | D0 (earned = invested + unspent) |
| **economic readout** | **E_20 ± 95%** (K 128 per state for state-level claims; K 8–16 suffices for means over hundreds of states) |
| short read (diagnostic) | E_10, and the share of pending items resolved by 10 and by 20 |
| beside it, never converted | levels and XP difference, HP, absence and turns still away, turns since each seat's last return, enemy stocks and crashed masses, freezes |
| verdict | "economically ahead" only when E_20's interval excludes 0; "no overall verdict" when XP or HP point the other way (21% of natural turn-94 states) |

- **Exceptions:** none among the categories with 30 or more states. For a seat that is dead or away at the
  checkpoint, E_5 and E_10 understate its position (−1.24 at h = 5 for the dead seat, −0.05 at h = 20). Do not read
  the short windows alone there.
- **Not a winner score.** E_20 is an expectation of earned gold under the continuation policies. It is not a value of
  the game, and it says nothing about XP, HP or position beyond what they turn into within 20 turns.

**Configuration recommendations** (no default, preset, objective or checkpoint changed; for the user's decision):
1. **Objective.** Keep the continuing objective (earned-gold reward, γ 0.99, no termination), as decided. M4.8 shows
   that the lane exit can be evaluated without changing the game.
2. **Investment curve.** The baseline `m45_calibrated()` remains the recommended M5 configuration, as at the closeout.
   - At the lane exit the two packages are indistinguishable on every readout (§4.4). M4.8 gives no reason to prefer
     C1, which stays a provisional opt-in.
   - The one curve difference M4.7 found before turn 94 (C1's early leads persist) survives the readout (§4.5).
   - The C1-or-baseline decision remains open.
3. **Evaluation in the opponent-modelling pilot and M5.** Add the lane-exit readout at turn 94 (E_20 with the table
   above) to the lane-phase readouts. Use raw turn-94 gold for no state-level claim, and the lane-exit readout for
   every lane-phase economic comparison. The code is `lanegame.diagnostics.lane_exit`, which needs only a state and
   the two continuation policies.
4. **Precision budget.** K 128 × 20 turns per state (about 2,600 simulated turns, about 1 s on one worker) for ±0.39 at
   h = 20; K 8–16 for population means.

**Next step:** step 3 of the agreed sequence, the design of Bayesian opponent modelling (approved direction, not
designed). The lane-exit readout gives its pilot a lane-phase outcome measure that is robust to reset timing. M4.8
does not change the agreed order.

## 6. Limitations

- **A gold readout only.** E_h settles pending farm and reset consequences in gold. It does not value XP, levels, HP,
  absence or position (wave state, tower damage, vision). At turn 94, 21% of natural states have the gold leader
  trailing by at least 15 XP or 0.3 bars of HP; there the readout gives no overall verdict, and nothing in this study
  converts those dimensions into gold.
- **Low uncertainty in E_h is not superiority in every dimension.** A certain E_20 lead says that seat is expected to
  have more earned gold 20 turns later under these policies, nothing more.
- **It depends on the continuation policies.** By default the policies that produced the state continue it.
  - Two in-distribution alternatives (another training seed of the same curve; the other curve's package) moved it
    little (correlation 0.96–0.98).
  - Opponents with unfamiliar styles, best responses or human-like play were not tested.
  - E_h is an expectation under those policies, not a game-theoretic value.
- **Short windows do not settle everything, and new consequences start at once.** Each seat completes about one
  reset every 20 turns, so every window also contains resets that were not pending at the checkpoint.
  - h = 20 is about one reset cycle: the original pending items have resolved in 95% of continuations, and the
    readout is flat to h = 40.
  - In the remaining 5%, mostly due seats that delay their Recall, something is still pending.
- **The prediction is accurate on average but not deterministic.** E_20 explains a fifth of the realized 20-turn
  change. The rest is genuine future randomness.
  - A single game's realized lead can be far from E_h (§3.2: +0.39 realized against +1.96 expected at h = 20).
  - Gold leaders with an HP or XP deficit lost somewhat more in their recorded continuations than predicted:
    z ≈ 2.8 in the validation sample, and the same direction, not significant, at three of four natural anchors
    (§3.5). The cause is not identified.
- **Selected cases are not representative.** Stage 1's cases were chosen to contain specific situations. The
  motivating returner configuration is rare at turn 94 (6 of 576 states); its common natural forms are the fresh and
  due-asymmetry categories (39% of states together).
- **One anchor family.** Checkpoints at fixed elapsed turns (94, with 84, 104 and 50 as checks), not at a level. Level
  14 arrives within turns 90–99 for most games (M4.7), and the lower level at turn 94 averages 13.4.
- **Population claims rest on six training seeds per curve.** Rollouts and the 48 game states per seed measure
  within-seed uncertainty only.
- **Cost.** 128 continuations × 20 turns ≈ 2,600 simulated turns per state for ±0.39 (95%) at h = 20, about 1 s of one
  worker. Population means need far fewer: K = 8 suffices when hundreds of states are averaged (§4.5).

## 7. Runtime, checks and files

**Runtime.** Wall clock on 5 worker processes at below-normal priority, on a 6-core desktop that was in other use
throughout:

| stage | wall | simulated turns |
|---|---|---|
| pilot (24 states, K 64) | 18 s | 61k |
| states: 2,304 natural checkpoints replayed and verified, 192 validation games | 27 s | 0.1M |
| Stage 1 continuations (469 states; mirrors and variants) | 645 s | 3.4M |
| Stage 2 continuations, t0 = 94 (576 states, 1–4 combinations) | 1,334 s | 6.8M |
| Stage 2 continuations, t0 = 50, 84, 104 (1,728 states) | 1,077 s | 4.4M |
| supplementary re-check (4,096 replays to turn 94, K 8 × 20 turns) | 337 s | 1.0M |
| analysis and tables | 70 s | |

That is about 1 h 10 min of compute in all. Throughput was about 2,400–5,400 turns/s on the wall clock (the pilot's
4,300 on a quieter machine).

**Checks.**
- `tests/test_m48.py` (18 tests) pass. They cover:
  - accounting with investments (earned = invested + unspent, spending is not earning, the lead change equals the
    summed rewards);
  - a restored checkpoint continuing identically, and replays reproducing the M4.6d records;
  - chance streams independent of the actions, equal seeds identical, paired differences vanishing;
  - mirrored continuations as exact negations;
  - windows as prefixes of the same continuations, and components reconciling with the readout;
  - resolution times, FIFO stock collection, variants keeping D0, case criteria and their mirror symmetry;
  - a full job with mirrors and window sums;
  - the M4.7 head-to-head replay;
  - the analysis helpers (pending items, categories, cross-validated R², robust slopes).
- Full regression suite: 1,473 passed, 4 skipped, 2 xfailed in 11 min (M4.7's 1,455 plus the 18 new tests); the legacy
  digest pins pass.
- The protocol was frozen before any evaluation run (SHA-256 57c99a63…). One dated amendment added the
  supplementary re-check before it ran.
- The raw 50 → 94 slopes and the turn-94 |lead| reproduce M4.7's published values exactly from the same records.

**Files.**
- New code:
  - `lanegame/diagnostics/lane_exit.py` (the reusable readout);
  - `lanegame/experiments/rl/m48/` (`common`, `states`, `families`, `rollouts`, `analysis`, `validate`, `apply`,
    `recheck`, `report`, CLI);
  - `tests/test_m48.py`.
- New documents: this report, [the protocol](m4_8_protocol.md).
- Generated (`results/m48/`): `pilot.json`, `protocol.json`, `validation.json`, `application.json`, `recheck.json`,
  `report.md`, `eval.log`, `recheck.log`. Raw continuation records and checkpoint states (git-ignored) are under
  `results/rl/checkpoints/m48/`.
- Updated:
  - the roadmap, `README.md`, `docs/README.md`, `docs/m4/README.md`, `CLAUDE.md`;
  - the mathematical reference (§9.1, §9.3 and a §13 line: documentation of an evaluation convention and of the
    no-termination decision; no rule changed);
  - the `lanegame.diagnostics` package docstring.

**Unchanged:** every rule, preset, default, objective, library module other than the new `diagnostics/lane_exit.py`,
checkpoint and earlier report. Nothing is staged or committed.

**Reproduce:**

```
python -m lanegame.experiments.rl.m48 pilot protocol states select val nat recheck analyse report --workers 5
```

Add `--priority below_normal` if needed. There is no training stage and no `--confirm` flag.
