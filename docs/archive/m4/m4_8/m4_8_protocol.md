# M4.8 protocol: lane-exit evaluation

*2026-10-16. Frozen (SHA-256 in `results/m48/protocol.json`) after the throughput and variance pilot and the
outcome-free state construction, before any Stage 1 or Stage 2 continuation. Already seen at freezing time:
- the pilot's throughput and within-state spreads (24 pilot states, `results/m48/pilot.json`);
- the prevalence of the case criteria among validation and natural checkpoints (counts only).

No continuation of a validation or natural checkpoint had been run, and no recorded continuation had been read.*

## 1. Purpose and scope

Assess who leaves the lane in the better economic position without mistaking pending farm collection and Recall timing
for advantages. **Evaluation only.** The reward (the earned-gold lead), the objective (γ 0.99, continuing time) and
every rule, default, preset and checkpoint stay unchanged. There is no termination at level 14 and no fixed turn
deadline; a lane exit is an evaluation checkpoint. There is no training, critic experiment or opponent modelling.

## 2. The readout

At a checkpoint state s (seat A's orientation; resources):
- **Earned gold** = invested + unspent (reference §7.2). Investing on a return moves unspent into invested, so spending
  never counts as earning again.
- **D0** = earned_A − earned_B, the raw lead at the checkpoint.
- **Δ_h** = E[Σ_{t<h} r_t], the expected change over the next h turns under the continuation policies (undiscounted
  exact ledger increments). It is estimated by K repeated continuations.
- **E_h = D0 + Δ_h**, the lane-exit economic readout: the expected earned-gold lead h turns later.
  - It is not the discounted G_h of earlier studies, nor a critic value. No tail is added.
  - It estimates outcomes under the continuation policies, not optimal play.

**Reported beside it, never folded in** (no exchange rate between dimensions):
- XP difference and levels;
- HP and absence status (present, away after a Recall, dead; turns still away);
- turns since each seat's last return;
- the wave: enemy minion HP alive on each seat's ledger (stock), the part at its own −1 or deeper (near), crashed
  masses, freezes;
- the uncertainty of E_h.

When gold, XP and HP disagree, no overall "who is winning" verdict is given.

Code: `lanegame/diagnostics/lane_exit.py` (readout, ledger, accounting) and `lanegame/experiments/rl/m48/` (the study).

## 3. Policies, states and seeds

**Policies.** The matched M4.6c 8.192M continuations `ext/baseline:s` and `ext/C1:s`, s = 0..5: equal lineage and
budgets. A *package* is a curve's environment played by that curve's policy of seed s.

**Stage 1 (validation) states.**
- 16 new self-play games of 160 turns per package (seeds 59,100,000 + 100 s + g).
- Every checkpoint at turns 30–120 is classified by checkpoint-only criteria (`m48/states.py`); X is the focal seat:
  - **returner (X = A: case 1; X = B: case 2, the mirrored situation):**
    - X is away after a Recall, or came back at most 2 turns ago and stands at its own −1 or deeper;
    - at least 16 HP of enemy minions are alive on X's ledger, at least 8 of them at X's own −1 or deeper;
    - Y is present, 12 or more turns past its last return, with at least 3 unspent resources or at most 50% HP.
  - **lost (case 3):** X is dead with at least 2 turns still away, and at least 12 HP of enemy minions are at its
    own −1 or deeper.
  - **denied (case 3):** Y holds a freeze on X's wave and X is present.
  - **stable (case 4):**
    - both present, each with at least 60% HP and at most 2.5 unspent resources;
    - both 3–15 turns past their last return;
    - enemy stocks within 8 HP of each other;
    - no crashed mass under either tower, no freeze.
  - **opposing (case 5):** both present; X leads in gold by at least 1 resource while trailing by at least 15 XP or
    0.3 bars of HP.
  - **due (controlled variants):** both present; A is at least 10 turns past its last return, with at least 2 unspent
    resources and at least 30% HP, and is not frozen on.
- **Selection:** up to 4 states per package × case × orientation, from distinct games, drawn uniformly among the
  qualifying checkpoints with a fixed generator (seed 59,250). No outcome is read.

**Stage 2 (application) states.**
- All 48 games per package of the M4.6d natural records (game seeds 99,000,000 + 100 s + g), replayed and verified
  against the stored record on every turn.
- Checkpoints at elapsed turn **t0 = 94** (primary: the median first turn at level 14 in M4.7), 84 and 104
  (sensitivity) and 50 (replication of M4.7's 50 → 94 slope). Actual levels are recorded.
- The records' realized continuations are used only to check predictions, never to select.
- Stage 2's case subsets use the same checkpoint criteria. They are reported apart from the representative results.

**Continuation seeds.** Continuation r of state i uses seed 59,{050 | 200 | 300}·10⁹ + 1000 i + r for the chance stream
and both sampling streams. The seed is identical for every package, alternative policy and variant of the state:
common random numbers.

## 4. Continuations, windows and repetitions

- **Length.** T = 40 turns. The windows h = 5, 10, 20 are the candidates, and 30 and 40 are diagnostic. All windows
  are prefixes of the same continuations. They are fixed in advance and never stopped early.
- **Default continuation:** the package that produced the state (both seats, CRN sampling).
- **Sensitivity** (t0 = 94):
  - the other curve's package on the same state (the matched-state contrast);
  - on a representative subset (games g ≡ 0 mod 4: 12 states per package), an alternative continuation policy: the
    same curve's policy of training seed s + 1 (mod 6) in both seats.
- **Controlled variants** of seat A (constructed; D0 unchanged by construction):
  - A_fresh: A has just come back from a reset (unspent invested, full HP, at its tower);
  - A_away1: A has just recalled and is away one more turn.
  - They are run on Stage 1 *due* states (A_fresh and A_away1) and on the Stage 2 t0 = 94 states that meet the *due*
    criterion (A_fresh).
- **Mirrors:** every Stage 1 returner state is also run mirrored (mirrored state, mirrored chance, swapped streams).
  It must give exactly −E_h.

**Repetitions.**

| quantity | value |
|---|---|
| pilot within-state SD of Δ_h, median (p75), h = 5 / 10 / 20 / 40 | 0.82 (1.05) / 1.35 (1.81) / 2.16 (2.43) / 2.94 (3.26) |
| precision target | 95% half-width ≤ 0.25 at h = 10 for the median state |
| K at t0 = 94 and in Stage 1 | 128 (median half-widths ≈ 0.14 / 0.23 / 0.37 / 0.51 at h = 5 / 10 / 20 / 40) |
| K at t0 = 84 and 104 | 64 |
| K at t0 = 50 | 32 (regressions only) |

- **Bounded:** K is fixed per family in advance and never extended after any result. The achieved precision is
  reported.
- **Budget:** 14.6M simulated turns (Stage 1: 469 states, 3.4M; Stage 2: 2,304 states, 11.2M), about 1 h at the
  pilot's 4,300 turns/s.

## 5. Readouts

**Stage 1 (validation).**
- **V1 accounting:** every continuation satisfies lead(final) − D0 = Σ reward units exactly. Per seat, earned = farm
  + plate + bounty.
- **V2 orientation:** each mirrored continuation is the exact negation of its original.
- **V3 restoration:** natural checkpoints replay their records exactly.
- **V4 per case:**
  - E_h and Δ_h by window (states as units within a case; packages pooled, curves also shown);
  - the event accounting: farm collected and available, farm lost while away, bounties, plates, deaths, Recalls,
    returns, absent turns, XP;
  - when the pending consequences resolve: the away or due seat's next return, and the turn the checkpoint's enemy
    stock has died on each ledger (first in, first out by HP);
  - the non-gold dimensions at the checkpoint and after h turns;
  - the recorded continuation of the validation game, as one realized sample.
- **V5 controlled variants:** E_h(variant) − E_h(orig) on common random numbers.
- **V6 calibration** (Stage 2's natural states, against the records' realized continuations):
  - the slope of the realized change on the predicted Δ_h (ideal 1);
  - the mean residual (ideal 0);
  - the share of the realized change's variance that the prediction explains.
- **V7 window stability:**
  - mean |Δ_20 − Δ_10| and |Δ_40 − Δ_20|;
  - the share of states whose material classification or leader changes between windows;
  - new consequences after h: deaths, Recalls and returns that start after h.
- **V8 reversals:** every material leader reversal (sign E_h ≠ sign D0, E_h's interval excluding 0) is decomposed
  into its accounting components, and its dominant component is named.

**Stage 2 (application; t0 = 94 unless stated; training seeds are the units for population claims, rollouts measure
within-state uncertainty).**
- **S1 how often raw gold misleads:**
  - per package, the share of states with a material adjustment: |Δ_h| ≥ 0.5 and its 95% interval excluding 0;
  - the share with a material leader reversal;
  - mean |Δ_h|;
  - the between-state SD of the true adjustment, √(var Δ̂_h − mean SE²);
  - means over 6 seeds per curve with t intervals.
- **S2 which situations explain them:**
  - categories by the checkpoint's pending situation (one seat dead, away after a Recall, freshly back, stock
    asymmetry ≥ 16 HP, ordinary), folded to the focal seat;
  - frequency, mean folded Δ_h and the share of material adjustments;
  - the variance of Δ̂_h explained by pending features (absence, freshness, due, stocks) against strength features
    (investment, level, HP).
- **S3 stability:**
  - across windows (V7);
  - across continuation policies: the mean absolute difference and the share of states whose classification changes,
    other package vs own, alternative vs own;
  - across anchors (84, 94, 104).
- **S4 baseline vs C1.** State distributions are kept apart from outcomes from matched states:
  - (a) **distributions:** |D0|, |E_h|, category shares, levels, XP and HP spreads; per seed, C1 − baseline paired by
    seed;
  - (b) **lead conversion** under each package's own continuation: the per-seed OLS slope of Δ_h on D0;
  - (c) **matched states:** the same states continued by both packages: the mean of sign(D0)·(Δ_h^C1 − Δ_h^base) and
    the slope difference. This is the contrast of the packages (environment plus its trained policy), not the
    isolated effect of the investment curve;
  - (d) **earlier conclusions:**
    - M4.7's |earned lead| at turn 94 (baseline vs C1, +0.35 ± 0.40), recomputed with |E_h|;
    - M4.7's conversion slope from turn 50 to 94 (−0.20 vs +0.04), recomputed with E_10 at both ends.
- **Materiality:** 0.5 resources for one state's adjustment; 0.25 for a population mean (M4.7's M).

## 6. Default window and presentation rule

The default window is the smallest h ∈ {5, 10, 20} that meets all of:
- (a) **coverage:** among natural t0 = 94 states with a pending item, the pending items have resolved in ≥ 80% of
  continuations by h. The pending items are:
  - an absent seat's return;
  - the checkpoint's enemy stock dying on each ledger;
  - the next return of a seat that is present and due (≥ 12 turns past its last return).
- (b) **residual drift:** for every pending category with at least 30 states at t0 = 94 (pooled over packages), the mean
  folded increment Δ_40 − Δ_h differs from the ordinary category's by at most 0.25.
- (c) **precision:** the median within-state 95% half-width of E_h at K = 128 is at most 0.5.

Categories that fail (b) at the chosen h are listed as exceptions. If no candidate meets (a)–(c), the recommendation is
a multi-window readout (E_10 and E_20 with resolution flags), not a single horizon.

## 7. What would make the method useful, limited, inadequate or inconclusive

- **Useful:** all of the following hold:
  - V1–V3 exact;
  - calibration at the default window (or at 10 and 20 for a multi-window readout): the 95% interval of the slope
    overlaps [0.8, 1.2] and the mean residual is within ±0.25;
  - the Stage 1 adjustments are accounted for by recorded events whose dominant components match the cases'
    mechanisms;
  - the readout adds information beyond D0: at t0 = 94, at least 10% of states carry a material and certain
    adjustment, or the between-state SD of the true adjustment is at least 0.25.
- **Valid, limited added value:** valid as above, but the added information is below both thresholds.
- **Inadequate:** any of the following:
  - an accounting, orientation or restoration check fails;
  - calibration fails;
  - every candidate window's median half-width exceeds 0.5;
  - more than half of the material classifications change between the continuation policies or between windows 10
    and 20.
- **Inconclusive:** anything else.

If the method is inadequate, Stage 2 reports only the parts that remain defensible (the event accounting and the
distributions), not an economic "winner" readout.

## 8. Known limitations

- E_h depends on the continuation policies; an alternative policy is tested only on a subset.
- Short continuations do not settle positional advantages (wave state, tower damage) or the value of XP and HP.
- Pending consequences resolve while new ones start; the diagnostic windows show how much of the later change is new.
- Two curves × six seeds: population claims rest on six training seeds per curve. Rollouts and game states measure
  within-seed uncertainty only.
