# M4.6e — Observation, training exposure and value learnability: why the critic is blind to strength

*2026-10-16. Status: diagnostic complete, awaiting review. No PPO or actor training ran. No rule, observation,
reward, parameter or default changed, and no checkpoint was modified. C1 (`m45_calibrated(invest_gain=3.24,
invest_scale=64.0)`) stays a provisional opt-in and is not adopted.*

*Protocol: [m4_6e_preregistration.md](m4_6e_preregistration.md), frozen (SHA-256 in
`results/m46e/preregistration.json`) before any supervised fit; one dated post-hoc amendment (§1.3). Generated tables:
`results/m46e/report.md`. Previous stages: [M4.6a](m4_6a_report.md), [M4.6b](m4_6b_investment_report.md),
[M4.6c](m4_6c_report.md), [M4.6d](m4_6d_report.md). Nothing is staged or committed by this work.*

**Terms.**
- **Resources and labels.** A *resource* is the gold of one fresh wave. *G_h* is a seat's discounted (γ 0.99)
  earned-gold lead over h turns, in resources, with no critic bootstrap. Labels are G300.
- **Duel ratio** = (1.05^ΔL · M_A / M_B)² (M4.6d): damage per exchange in victim bars, stronger over weaker. "Edge" is
  one level plus 3 invested resources early (duel ×1.21–1.27). "×1.76" is an investment multiplier ratio of 1.76
  (duel ×3.1).
- **Policies.** The six 8.192M C1 general policies `ext/C1:s`, each playing itself, so every label is the return the
  policy's own critic is trained to predict.
- **Arms.**
  - A: the existing critic.
  - B: a linear readout fitted on the frozen trunk.
  - C: a fresh value network on the observation.
  - D: C plus six observation-derived strength features.
  - L: a linear (ridge) reference on the observation plus the same features.
- **Statistics.** Intervals are 95% t intervals over the six policies unless stated.

---

## Summary

**The four questions, answered.**

1. **Does the observation expose what strength-dependent values need?**
   - **Yes, correctly and losslessly.** Every controlled single-feature change moves exactly the intended entries in
     both seats' observations (54 / 54).
   - On 5,915 reachable states the observation is exactly decodable to the full Markov state, injective (no two states
     share an observation) and mirror-consistent.
   - The duel ratio is an exact function of two entries (error 6×10⁻⁵), and a ×1.76 gap moves the investment entry by
     1.4–2 natural standard deviations.
   - There is no missing information, no incorrect encoding and no state aliasing. Configurations, reward scale and
     checkpoints are consistent.
2. **How much exposure did the problematic families get?** (Reconstructed from frozen snapshots; per 8.192M run, in
   joint turns, each of which trains both roles.)
   - Plausible early edges: common (≈ 137,000, 3.4% of turns).
   - Large gaps (duel ≥ 2): rare (≈ 1,300, 0.03%, all from archive starts).
   - Extreme gaps like M4.6d's ×1.76 cells (duel ≥ 3): **none**. The training archive itself contains no such state.
   - Favourable early freezes with a damaged breaker: ≈ 1,400 (0.03%).
   - The weak-side Safe / Yield family: ≈ 18,000 (0.4%).
3. **Can their values be learned?**
   - **From the observation, yes.** A fresh network of the critic's shape (C), trained on about 20,000 labelled
     examples per policy, recovers 71–87% of the strength premiums of the M4.6d cells. The critic recovers 1–2% of the
     edge, ×1.76 and gap premiums, and 38% of the plaus premium, which includes an HP gap. C captures 84% of the
     explainable variance at held-out constructed states (critic: 32%).
   - **From the trained network's representation, no.** A linear readout of the trunk's last layer (B) recovers 2–3%:
     the layer the value head reads keeps HP (R² 0.92) but has largely discarded strength (R² 0.18, against 1.00 from
     the observation and 0.75 from the first layer).
4. **Smallest justified next step.** **None on the learning side: treat the critic's strength blindness as a
   documented limitation and proceed to M4 closeout.**
   - Better values do not materially improve the ranking of the decisions M4.6d flagged: the freezer's Spearman goes
     from 0.09 to 0.14–0.17, against a pre-registered threshold of +0.2. Elsewhere there is no gain.
   - The weak-side miss is already ranked correctly by the existing critic.
   - Recommendation **3** of the brief, by the pre-registered rule.

**What this narrows, and what it does not.**
- The blindness is not in the observation, the labels or the encoding. It is in what the shared actor–critic trunk
  represents, combined with almost no exposure to large gaps.
- Which part of PPO value training is responsible is not isolated. Candidates: the trunk shaped by the policy loss
  (the value term is under 1% of the trunk gradient), bootstrapped targets, rare states.
- Nothing here measures actor improvement.

---

## 1. What was run

### 1.1 Stages

| stage | content | simulated turns | wall clock (6 workers) |
|---|---|---|---|
| observation | 18 single-feature changes × 3 base states × 6 policies; 5,921 reachable states (decode, mirror, injectivity); entry ranges; compatibility of 12 checkpoints; gradient shares on 9 policies (the M4.6c mixed-start collector, 2 batches of 4,096 transitions after 2 warm-up batches) | ≈ 0.1M | 21 s |
| exposure | archive composition (4,800 states, both orientations, both curves); reconstructed occupancy for 12 policy-times (C1 seeds 0–2 at 1.0M, 4.1M, 8.192M; baseline finals 0–2), each 192 archive games × 150 turns + 48 s0 games × 512 turns | 0.64M | 2 min |
| labels | natural: 6 policies × 32 games × 600 turns; constructed: 6 policies × 679 paired starts × 2 rollouts × 450 turns; tail: 3 policies × 8 cells × 32 × 800 turns | 4.4M | 10.5 min |
| fit | 6 policies × 4 splits × (ridge B and L, C × 3 seeds, D × 3 seeds) | — | 3.3 min |
| ranking | replay of turn 0 of the 288 stored M4.6d ranking jobs (decision states × candidates × 16 rollouts) with every predictor | ≈ 0.03M | 14 s |
| analyse, report | readouts, trunk probes, tables | — | 1 min |

The simulation used about 18 minutes of wall clock in all. The whole task (reading, design, implementation, runs,
analysis, documentation) took about 2.5 hours.

### 1.2 The label sets (per policy; identical start states and seeds for all six)

- **Natural.** 32 s0 self-play games of 600 turns, with a label every 10 turns from turn 5 to 295: 960 states, 1,920
  examples (both seats).
- **Constructed.** A fixed stratified sample of 200 states of the M4.6c *training* archive (40 natural, 40
  natural+lead, 60 zone, 60 freeze). Each comes with paired variants on common seeds:
  - `orig`: as archived;
  - `eq`: the archived investment lead removed;
  - `lvl`: A one level up, same HP fraction;
  - `hpB`: B at 0.6 of its HP;
  - `big`: A's investment multiplier ×1.3–1.9 over B's (duel ×1.7–3.5).

  That is 679 starts with 2 rollouts of 450 turns each, labelled every 15 turns from turn 0 to 150: 14,938 states,
  29,876 examples.
- **External test.** The M4.6d calibration states of the same policies, never used for fitting: 12 freeze, 12 lane and
  18 gap cells, plus 16 natural lead / even states, with K 32 labels (G300; natural also G500). These cells are
  outside the training archive's distribution: the archive has no ×1.76 states.
- **Groups.** One natural game, or one archive state with all its variants, seats, rollouts and trajectory points. 25%
  of the groups are held out (the same groups for every policy) and 15% of the rest serve as validation.

### 1.3 Deviations and amendments

| item | what happened |
|---|---|
| `big` variant added before labelling | the archive's leads turned out to be modest strength gaps under C1 (99th percentile duel ×1.4–2.3), so a paired large-gap variant was added to make the study cover the M4.6d range. Done before any label existed; part of the frozen protocol |
| amendment 1 (post hoc) | 19 of the 120 `big` variants need more than 240 invested resources, two of them over 1,000: B was already near C1's saturation (M ≤ 4.24), so a ×1.3 multiplier ratio is unreachable in play. A sensitivity split `reach` drops them from training. Every readout moves by at most 0.03 (generated tables). The pre-registered splits stay primary |
| two bugs found before any reading | the exposure stage first crashed on a missing output directory, and the decode check on that one unreachable variant. Both were fixed in the instrumentation and re-run; the pairs check's cell-name matcher was fixed before its first full run |
| a readout bug found by its test | stacking both seats' premiums with opposite signs cancelled them; premiums are now oriented as the stronger side's gain in both seats. Fixed before the fits were read |

## 2. Q1 — the measurement path

### 2.1 From state to observation

The full Markov state of an M4 turn is the lane `State`, the `M4State` of both champions and k mod P. The
observation (90 entries, `observation_m4.py`) encodes all of it in each seat's own perspective. The investment and
unspent-gold entries are compressed (n / (n + 64) and n / (n + 8)), exact up to 23,106 and 8,184 resources. The
attack-power entry ap / (ap + ap_ref) combines level and investment.

| check (5,921 reachable states: s0 and archive-start self-play, the archive sample's variants, the M4.6d cells) | result |
|---|---|
| distinct states / distinct observations / collisions | 5,915 / 5,915 / **0** |
| exact decode back to the Markov state | **0** mismatches (one constructed `big` variant at 32,812 invested resources lies beyond the exact range: amendment 1) |
| mirror: observe(s, seat B) = observe(mirror(s), seat A) | **0** mismatches |
| entries outside [0, 1]; plates-pending fields (not encoded) non-zero | 0; 0 (always 0 under the M4 rules, so nothing is lost) |
| largest investment / gold entry | 0.998 (the unreachable variant) / 0.79: no saturation in play |

**Controlled single-feature pairs.** From three base states (freeze L5, neutral L8, neutral L15), 18 changes of seat A
were made, each touching one feature: investment, unspent gold, level, XP within a level, own and enemy HP, absence,
position, wave HP, vision, a plate and the phase. **All 54 change exactly the intended entries**, in seat A's own block
and seat B's enemy block, by the intended amounts. Examples at neutral L8:
- +8 invested: invested entry +0.082, AP +0.027;
- +1 level: level +0.059, max-HP +0.031, AP +0.012;
- enemy at 60% HP: the enemy HP entry −0.400.

**Is strength "present but hard to use"?** It is present at an ordinary scale.
- A ×1.76 contrast moves the investment entry by +0.27 to +0.41, against a natural standard deviation of 0.19.
- An edge contrast moves level, max HP, investment and AP by +0.02 to +0.06 each.
- The log duel ratio is an exact function of the two attack-power entries (max error 6×10⁻⁵). A ridge regression on
  the raw observation decodes it with held-out R² 1.00.
- Natural play rarely produces large values: the 5–95% range of the natural log duel is −0.17 to +0.16 (duel ×0.84 to
  ×1.17). A ×1.76 gap (log duel 1.13) lies far outside it.

**Conclusion on hidden state.** No two different states share an observation, and the game is stationary in k mod P,
so a state-specific value error cannot come from aliasing. Given the evaluated policy pair, the value is a function of
the observation.

### 2.2 From observation to critic, value and return

| item | finding |
|---|---|
| architecture (`rl/network.py`) | **one trunk for actor and critic**: 90 → 128 → 128 (tanh), then a linear policy head (61) and a linear value head (1). The critic has no layer of its own beyond its 129 head parameters |
| scaling | value head in 0.1-scaled resources (reward_scale 0.1); every evaluation divides by 0.1; M4.6d's natural-state slope of realized return on V is 0.985 |
| training objective (`ppo.py`) | λ-returns (γ 0.99, λ 0.95: target horizon about 17 turns), value clipping ±0.5 scaled (±5 resources), vf_coef 0.5, advantages normalized for the policy loss only |
| checkpoints | all 12 ext finals load digest-verified, with 90 inputs / 61 actions, and their run.json game configuration equals the arm's; reward scale 0.1, γ 0.99, λ 0.95, vf_coef 0.5, vf_clip 0.5, entropy 0.03, 100% current-policy self-play |
| zero-sum consistency | V_A + V_B over the reachable states: mean +0.12, mean absolute 0.30, 5–95% −0.43 to +0.73 (minor, as in M4.6d) |
| label horizon (§4) | G300 is within ±0.05 of G800 on the problematic constructed cells |

**What the critic responds to.** The critic's change for single-feature changes, in resources, mean over the six
policies:

| change of seat A | freeze L5 | neutral L8 | neutral L15 |
|---|---|---|---|
| +8 invested | +0.03 | +0.03 | +0.02 |
| +32 invested | +0.10 | +0.10 | +0.05 |
| +1 level | +0.01 | +0.00 | +0.00 |
| +4 unspent gold | −0.05 | −0.10 | −0.09 |
| own HP to 60% | −0.29 | −0.48 | −0.49 |
| enemy HP to 60% | +0.47 | +0.49 | +0.47 |
| own front HP halved | +0.33 | +0.41 | +0.40 |
| first enemy plate dealt | −0.22 | −0.25 | −0.25 |

- The critic is sensitive to HP, wave and plate state, nearly insensitive to investment, and blind to level.
- It assigns unspent gold a small negative value, although unspent gold becomes strength at the next base visit.
- For scale: M4.6d measured +0.83 realized for an edge and +5.8 for ×1.76.

### 2.3 What the shared trunk represents

Ridge probes from each layer, with held-out R² on the main split's test groups (six policies):

| target | from the observation | from hidden layer 1 | from hidden layer 2 (the value head's only input) |
|---|---|---|---|
| log duel ratio | 1.00 ± 0.00 | 0.75 ± 0.03 | **0.18 ± 0.02** |
| own HP fraction | 1.00 ± 0.00 | 0.99 ± 0.00 | 0.92 ± 0.03 |

- A ×1.76 contrast moves hidden layer 2 by only 0.10 ± 0.01 of the natural states' mean distance from their centroid.
- **The trained trunk keeps HP and progressively discards strength.** The value head can only read layer 2.

**Gradient shares on training-like batches.** These use the M4.6c mixed-start collector and each run's own PPO
configuration, at the final weights, without applying an update.
- The value term (vf_coef × value loss) has a trunk-gradient norm of 0.6–0.9% of the policy term's.
- Their cosine is ≈ 0: +0.01 on average, −0.02 to +0.02 per policy, baseline policies alike.
- So the trunk is shaped almost entirely by the policy objective. The small value term follows from the 0.1 reward
  scale (value errors of ±0.1–0.3 in scaled units) against normalized advantages in the policy loss.
- This is a property of the configuration, not a bug. It is consistent with the trunk dropping features the actor
  barely uses, but it does not prove that this is why strength is dropped.
- Note: "shared" here means the actor and critic sharing layers. Nothing in this study bears on the shared *policy*
  (one network for both seats).

## 3. Q2 — training exposure

### 3.1 Direct measurements

**Start counts** (final.pt metadata of the six C1 8.192M runs, mean per run; the baseline runs are identical, since
they use the same seeds and collector):

| start | games started | joint turns of finished games | share |
|---|---|---|---|
| s0 | 6,003 | 3,064,758 | 74.9% |
| archive: natural | 2,557 | 383,106 | 9.4% |
| archive: natural+lead | 2,146 | 321,580 | 7.9% |
| archive: zone | 1,303 | 195,227 | 4.8% |
| archive: freeze | 852 | 127,590 | 3.1% |

**The archive itself**, every state in both orientations under the C1 curve (start state only):

| label | plausible early edge | large (duel ≥ 2) | extreme (≥ 3) | favourable freeze | ... with a damaged breaker | weak-side window | stronger side's duel q50 / q90 / q99 |
|---|---|---|---|---|---|---|---|
| freeze (600) | 23% | 0.2% | **0%** | 7.0% | 2.2% | 1.2% | 1.10 / 1.42 / 1.88 |
| zone (900) | 20% | 3.2% | **0%** | 6.7% | 1.1% | 6.7% | 1.11 / 1.52 / 2.33 |
| natural+lead (1,500) | 6% | 1.4% | **0%** | 0.6% | 0.2% | 3.0% | 1.11 / 1.44 / 2.22 |
| natural (1,800) | 6% | 0% | **0%** | 0.4% | 0.1% | 0.7% | 1.05 / 1.18 / 1.39 |

"25% archive transitions" was therefore not 25% exposure to strength gaps:
- the archive's leads were defined in gold, and under C1 most are modest duel ratios;
- no archived state reaches M4.6d's ×1.76 / gap cells;
- the baseline curve gives the same picture.

### 3.2 Reconstructed occupancy (an estimate)

- **Method.** M4.6c trained with 100% current-policy self-play, so a frozen snapshot playing itself from the training
  start distribution (archive starts with mirroring, 150 turns, or s0, 512 turns; archive turns weighted to the
  measured 25%) reproduces the training state distribution of that moment, up to drift within a snapshot interval.
  Final-policy rollouts do not reproduce earlier training.
- **Roles.** Each joint turn trains both roles once, so stronger and weaker roles are exposed equally by construction.

| family (either seat as the stronger / freezing side) | C1 @1.0M | C1 @4.1M | C1 final | baseline final | share from archive starts (C1 final) |
|---|---|---|---|---|---|
| plausible early edge (duel 1.10–1.5, lower level ≤ 9) | 3.16% (129,000) | 3.19% (130,000) | 3.35% (137,000) | 3.34% | 14% |
| ... duel 1.25–1.5 | 0.89% | 0.92% | 0.91% (37,000) | 1.10% | 21% |
| large gap (duel ≥ 2) | 0.041% | 0.025% | **0.032% (1,300)** | 0.023% | 100% |
| extreme gap (duel ≥ 3) | 0 | 0 | **0** | 0 | — |
| favourable early freeze (M4.6d definition) | 0.15% | 0.11% | 0.10% (4,300) | 0.15% | 31% |
| ... breaker at ≤ 70% HP, freezer level ≤ 8 | 0.047% | 0.034% | **0.033% (1,400)** | 0.065% | 27% |
| weak-side Safe / Yield window (25–55% HP, stronger opponent ×1.2+ within one cell and catchable) | 0.55% | 0.47% | **0.44% (18,000)** | 0.32% | 43% |
| weak side ≤ 60% HP, Recall legal, stronger opponent catchable | 2.1% | 1.9% | 1.9% (78,000) | 1.3% | 37% |
| any freeze state | 11.2% | 10.0% | 10.6% | 10.5% | 25% |

(In brackets: joint turns per 8.192M run, occupancy × 4.096M.) The rare estimates rest on few events per policy-time
(large gaps 27–49, favourable freezes with a damaged breaker 4–15 in s0 games plus 7–15 in archive games), so they are
precise to about a factor of 1.5, which does not change any conclusion.

**After an archive start the advantage persists, but the special situation does not.**
- Mean |log duel| after a zone start stays at 0.17 for 20 turns, 0.16 at 50, 0.12 at 100 and 0.09 at 149: investment
  is permanent and erodes only as both sides invest.
- Large gaps fall from 4% of zone games at the start to 2% by turn 5 and 0% by turn 50.
- Freeze starts lose their freeze at once: 100% → 27% at turn 1 → 11–20% thereafter, about the rate of ordinary play.
- Favourable freezes vanish within two turns (16% → 4% → 0%, re-forming only sporadically).
- Plausible early edges in s0 games occur only in the first 100 turns, while the lower level is ≤ 9 (21% of those
  turns).

**The M4.6d diagnostic states against the training distribution** (C1 final; neighbourhood: the same kind, level ±1,
duel within ×1.15, the weaker side's HP within ±0.15; joint turns per run):

| M4.6d states | training neighbourhood |
|---|---|
| freeze cells, equal power | 1,700–5,100 |
| freeze cells, edge / plaus | 280–1,650 |
| neutral / push lane cells, edge | 16,000–42,000 |
| Recall windows, edge | 4,900–14,000 |
| gap cells ×1.5 (duel 2.0–2.6) | 0–920 |
| **every ×1.76 state (freeze, lane, gap, Recall window)** | **0** |

### 3.3 Answer to Q2

- **Large and extreme gaps:** the critic had essentially no training signal. M4.6d's ×1.76 states lie outside
  everything training produced, archive included. That alone predicts a poor value there.
- **Plausible early edges:** common (≈ 137,000 joint turns per run, mostly early s0 play), yet the critic's edge
  premium is still ≈ 0 (§5). Exposure is not the explanation for the edge case.
- **Favourable freezes with a damaged breaker** (≈ 1,400 per run) and the **weak-side window** (≈ 18,000) are rare
  but not absent.

## 4. Q3 — return labels

- **Policy-specific.** Every label is one policy playing itself in the C1 environment, from the labelled seat, in
  resources; fits are per policy and never pooled.
- **Reproduction.** The rollout core reproduces a stored M4.6d calibration rollout to 10⁻⁹ (test), and the M4.6d
  external labels are reused unchanged.
- **Horizon and residual tail on the problematic constructed cells.** Three policies, K 32, H 800, seeds independent of
  M4.6d's. The critic plays no part, so this is independent of the critic under audit.

| cell | G50 | G150 | G300 | G800 | G800 − G300 |
|---|---|---|---|---|---|
| freeze L5 equal | +1.41 | +1.59 | +1.68 | +1.69 | +0.01 ± 0.08 |
| freeze L5 edge | +2.00 | +2.65 | +2.75 | +2.77 | +0.02 ± 0.03 |
| freeze L5 plaus | +2.72 | +3.32 | +3.34 | +3.31 | −0.03 ± 0.03 |
| freeze L8 plaus | +2.52 | +3.02 | +2.97 | +2.96 | −0.01 ± 0.06 |
| freeze L5 ×1.76 | +5.21 | +7.09 | +7.30 | +7.33 | +0.03 ± 0.04 |
| neutral L8 ×1.76 | +4.46 | +6.52 | +6.71 | +6.76 | +0.05 ± 0.02 |
| gap neutral L6 ×1.5 | +2.37 | +3.61 | +3.73 | +3.73 | +0.00 ± 0.02 |
| gap neutral L10 ×1.76 | +4.24 | +6.51 | +6.88 | +6.90 | +0.01 ± 0.02 |

- **Tail.** Beyond 800 turns the remainder is at most γ^800 · |V| ≈ 3×10⁻⁴ · |V|, below 0.003 for |V| ≤ 10. **G300 is
  within about ±0.05 of the infinite-horizon value at these states.** It is a finite-horizon approximation with that practical uncertainty, not
  an exact label.
- **Noise.** Per-rollout noise dominates: the SD of G300 is 2.9–3.3, and the noise floor of a two-rollout mean is
  4.6 resources².
- **Timing.** Most of a gap's value is realized within 100–150 turns: 70% within 50 turns at ×1.76.
- **Paired equal / advantaged states.** These separate the generic constructed-state error from the strength-specific
  one. The critic's equal-power freeze cells carry a small sign-varying bias (M4.6d: −0.3 to +0.6), whereas its premium
  error is about the whole premium (§5).

## 5. Q4 — the supervised value study

### 5.1 Main split (pre-registered primary readouts)

**P1. Strength-premium recovery on the M4.6d cells.** Ratio = Σ predicted / Σ realized premium over the contrasts and
both seats. Realized premiums: edge +0.83 ± 0.32, plaus +1.29 ± 0.24, ×1.76 +5.77 ± 0.42, gap cells +4.94 ± 0.30.

| arm | edge | plaus | ×1.76 | gap cells | rule: edge+plaus ≥ 0.6 | rule: large+gap ≥ 0.6 |
|---|---|---|---|---|---|---|
| A critic | 0.01 ± 0.02 | 0.38 ± 0.13 | 0.02 ± 0.02 | 0.02 ± 0.02 | 0.17 ✗ | 0.02 ✗ |
| B trunk readout | 0.02 ± 0.02 | 0.49 ± 0.20 | 0.03 ± 0.02 | 0.03 ± 0.03 | 0.22 ✗ | 0.03 ✗ |
| C fresh network | 0.71 ± 0.43 | 0.87 ± 0.09 | 0.83 ± 0.16 | 0.85 ± 0.11 | **0.75 ✓** | **0.84 ✓** |
| D + strength features | 1.67 ± 1.02 | 1.38 ± 0.21 | 1.10 ± 0.15 | 1.15 ± 0.10 | 1.48 ✓ (over) | 1.13 ✓ |
| L linear reference | 1.13 ± 0.59 | 1.05 ± 0.19 | 0.95 ± 0.11 | 0.97 ± 0.07 | 1.06 ✓ | 0.96 ✓ |

**P2. Natural accuracy.**

| arm | held-out natural games: MAE | M4.6d natural states (K 32, G500): MAE | bias | slope | R² |
|---|---|---|---|---|---|
| A critic | 2.14 ± 0.53 | **0.42 ± 0.10** | +0.01 | 1.01 | **0.77** |
| B | 2.18 | 0.55 | +0.01 | 0.89 | 0.61 |
| C | 2.23 (×1.04 of A: rule ✓) | 0.60 ± 0.12 | −0.04 | 0.78 | 0.53 |
| D | 2.23 | 0.59 ± 0.07 | +0.04 | 0.77 | 0.55 |
| L | 2.20 | 0.50 ± 0.12 | −0.00 | 0.90 | 0.63 |

- **Constructed held-out states.** Held-out constructed starts are labelled with two-rollout means (noise floor 4.6).
  The share of the explainable variance captured is: critic 0.32 ± 0.05, B 0.32, **C 0.84 ± 0.07**, D 0.88, L 0.86.
  Seat A's bias on the M4.6d cells: critic freeze −1.99, lane −1.66, gap −4.64 (M4.6d's numbers reproduced); C +0.24,
  −0.17, −0.86.

**P3. Paired premiums on held-out archive groups** (seat A's gain, CRN-paired, mean realized / mean predicted):

| contrast | realized | critic | B | C | D | L |
|---|---|---|---|---|---|---|
| investment lead (orig − eq) | +0.59 ± 0.42 | +0.02 | +0.02 | +0.53 | +0.70 | +0.63 |
| one level (lvl − orig) | +0.27 ± 0.46 | −0.01 | −0.01 | +0.03 | +0.50 | +0.27 |
| B's HP × 0.6 (hpB − orig) | −0.35 ± 0.64 | +0.45 | +0.53 | +0.51 | +0.57 | +0.42 |
| big gap (big − orig) | +4.74 ± 0.43 | +0.11 | +0.14 | +3.52 | +4.34 | +3.87 |

- The level and HP contrasts on these archive states are too noisy to evaluate: their intervals include 0.
- C predicts the investment premium (+0.53 against +0.59 realized). Its level premium is near 0 (+0.03), where D and
  L, which see the duel ratio directly, predict +0.27 to +0.50. The realized level premium on these states is not
  distinguishable from 0, so this contrast cannot decide between them.

**Stability.** The three fit seeds of C give 0.66–0.76 (edge), 0.81–0.94 (plaus) and 0.82–0.84 (×1.76). D's seeds
give 1.60–1.71, 1.32–1.49 and 1.08–1.12. Ridge strengths are interior to the grid (B: 1,000–3,000).

### 5.2 Stronger generalization checks

| split | C: edge / plaus / ×1.76 / gap | D | L | reading |
|---|---|---|---|---|
| strength (no training example beyond duel ×1.6) | 0.39 / 0.70 / 0.41 / 0.41 | 1.36 / 1.16 / 0.84 / 0.91 | 0.78 / 0.84 / 0.72 / 0.74 | the fresh network extrapolates poorly beyond the strength range it saw; the explicit log-duel features extrapolate far better (an inductive bias, not information) |
| family (no constructed freeze group in training) | 0.67 / 0.88 / 0.82 / 0.84 | 1.64 / 1.33 / 1.07 / 1.14 | 1.16 / 1.09 / 0.96 / 0.97 | unchanged from main: the strength value transfers across state families |
| reach (amendment 1) | 0.72 / 0.89 / 0.84 / 0.86 | 1.66 / 1.37 / 1.09 / 1.15 | 1.10 / 1.03 / 0.93 / 0.95 | the unreachable variants do not matter |

### 5.3 Do better values rank the M4.6d decisions better?

- **Method.** Turn 0 of every stored M4.6d ranking rollout was replayed, matching the stored critic look-ahead to
  1.2×10⁻⁶. The look-ahead r0 + γV̂(s1) was then recomputed with each predictor and compared with the stored K 16,
  150-turn rollouts.

| decision | rollout spread | critic: Spearman / same best | B | C | D | L |
|---|---|---|---|---|---|---|
| freezer's first move at a freeze (12 states) | 0.59 | 0.09 / 18% | 0.09 / 24% | 0.14 / 29% | 0.17 / 25% | 0.04 / 22% |
| breaker's first move at a freeze (12) | 0.64 | 0.38 / 35% | 0.38 / 37% | 0.43 / 36% | 0.24 / 29% | 0.44 / 42% |
| stronger side at a Recall window (8) | 0.67 | 0.43 / 58% | 0.38 / 58% | 0.30 / 54% | 0.25 / 40% | 0.31 / 48% |
| weak side at a Recall window (8) | 0.77 | 0.44 / 77% | 0.38 / 75% | 0.30 / 69% | 0.44 / 71% | 0.31 / 62% |
| weak side, Safe vs Yield codes (8) | 0.77 | 0.52 / 69% | 0.47 / 67% | 0.45 / 73% | 0.47 / 65% | 0.58 / 73% |

- **No material gain.** Against the pre-registered materiality of +0.2 Spearman or +15 points of same-best agreement,
  the freezer improves by +0.05 to +0.08 and +7 to +11 points, below both thresholds. Every other decision is unchanged
  or slightly worse.
- **Why (an explanation, consistent with the table and not separately tested).**
  - The candidates of one decision start from the same state and differ by one turn, which changes HP, the wave and
    positions, not investment or level.
  - So a strength-dependent value error is nearly the same for every candidate and cancels in the advantage.
  - What separates the options is the one-turn consequence. Neither the critic nor the fitted predictors resolve it at
    this label budget: their predicted spreads are 0.12–0.24, against 0.59 in the rollouts.
- **The weak-side miss.** M4.6d's actor–critic mismatch (Safe instead of Yield) is not a value problem: the existing
  critic already ranks it correctly.

## 6. Interpretation

| explanation | verdict | evidence |
|---|---|---|
| the observation lacks strength information or encodes it wrongly | **ruled out** | §2.1: exact, lossless, mirror-consistent, duel an exact function of two entries; a fresh network learns the premiums |
| hidden states alias, so labels mix different values | **ruled out** | no observation collisions on 5,915 states; the observation is the Markov state |
| the premiums are label or horizon artifacts | **ruled out** | G300 within ±0.05 of G800 on the problematic cells; paired variants on common seeds |
| the trained representation cannot express strength value | **supported** | B recovers 2–3%, also with a weak penalty (5%); log duel R² falls from 1.00 (observation) to 0.18 (layer 2) while HP stays 0.92 |
| the value is learnable from this observation | **supported on the tested distribution** | C recovers 71–87% with about 20,000 training examples per policy; D and L 95–167%; transfers across families; extrapolation beyond the trained strength range needs the explicit features |
| exposure explains the large-gap blindness | **supported** | duel ≥ 3 never occurs in training, duel ≥ 2 in 0.03% of turns |
| exposure explains the edge blindness | **not supported** | plausible early edges occupy 3.4% of training turns (≈ 137,000 joint turns per run) and are still valued at ≈ 0 |
| which part of PPO value training failed | **not isolated** | candidates: a trunk shaped by the policy objective (the value term ≈ 0.7% of the trunk gradient), bootstrapped λ-targets with a horizon of about 17 turns against a premium realized over 50–150 turns, and the small size of an edge premium relative to return noise. These experiments do not separate them |
| a better critic would fix the actor's gaps | **not supported** | better values do not rank the freezer's options materially better; the weak-side miss is already ranked correctly by the critic |

These comparisons narrow the explanation; they do not identify a unique cause. In particular, B's failure says the
trained trunk does not carry strength *linearly*. It does not say PPO could not have learned it with another value
loss, horizon or architecture, and nothing here measures how an actor would play against a better critic.

## 7. Recommendation

**Decision 3: treat the remaining limitation as acceptable and proceed to M4 closeout.** No learning pilot is
recommended now, and no implementation defect was found.

**The pre-registered rule.**
- Condition 1 holds: C recovers strength value with natural MAE ×1.04.
- **Condition 2 fails**: there is no material ranking improvement at the flagged decisions.
- Condition 3 is moot.

**The stopping principle agrees.**
- The weakness is confined to states the natural game seldom or never produces: large gaps 0.03%, extreme gaps none,
  favourable freezes with a damaged breaker 0.03%.
- The one place better values would plausibly matter, the freezer's choice, did not improve.
- The weak-side miss is an actor issue the critic already sees.
- The critic is calibrated where the game goes: natural R² 0.77, slope 1.0.
- None of this undermines the central claim (shared-policy PPO self-play learns the conditional laning strategies
  documented in M4.4–M4.6d), provided evaluation keeps using rollouts rather than the critic in strength-gap states,
  as M4.5c–M4.6d already do.

**If the question is reopened later** (named, not recommended, and not authorized):
- **Candidate.** The evidence-backed intervention is **a value function with its own representation**: a separate
  critic trunk on the unchanged observation, or a value-loss weight large enough to shape the shared trunk. Explicit
  strength features would be an inductive-bias variant to test second, not first. The reason: the information is in
  the observation, and a separately trained network learns it.
- **Test design.** A single controlled pilot against an unchanged matched control, on value calibration in strength
  states, on the M4.6d decisions and on ordinary-play regression.
- **Expected benefit.** Given §5.3, the expected decision benefit is small; it would be justified mainly if M5 needs
  accurate absolute values in strength-gap states.

**Limitations.**
- **The fitted networks are diagnostics, not replacement critics.** On the precise natural states they are worse than
  the critic (MAE 0.60 against 0.42; R² 0.53 against 0.77). They had about 1,900 natural examples per policy against
  the critic's millions of transitions.
- **Labels.** Single-rollout labels are noisy (SD ≈ 3). The supervised set is about 32,000 examples per policy (about
  20,000 for training), drawn
  from a 200-state archive sample with constructed variants. Natural archive states taken from the same game are not
  grouped together, a minor leak in the held-out readouts that does not affect the external M4.6d test.
- **Exposure** is reconstructed from three seeds at three snapshot times, and the rarest families rest on 4–49 events
  per policy-time.
- **Ranking** reuses M4.6d's K 16 rollouts, with 4–6 candidates per state.
- **Scope.**
  - One environment (C1) for the supervised study; M4.6d found the baseline critics equally blind, and this study does
    not re-test them.
  - No result here concerns shared-policy (two-seat) interference, a from-scratch campaign, or C1's adoption.

## 8. Reproduction, files and tests

**Commands** (from the repository root; every stage is resumable, with per-job records under
`results/rl/checkpoints/m46e/`):

```bash
P="python -m lanegame.experiments.rl.m46e"
$P labels --workers 6                 # natural + constructed labels and the 800-turn tail check (10.5 min)
$P prereg                             # freeze docs/m4/m4_6/m4_6e_preregistration.md (SHA-256)
$P fit --workers 6                    # arms B, L, C, D x 4 splits x 6 policies (3.3 min)
$P ranking --workers 6                # replay of the M4.6d ranking jobs with every predictor (14 s)
$P exposure --workers 6               # archive composition, reconstructed occupancy (2 min)
$P observation --workers 6            # Q1 checks and gradient shares (21 s)
$P analyse report                     # results/m46e/analysis.json and report.md (1 min)
python -m pytest tests/test_m46e.py   # 9 tests
```

`results/m46e/run_queue.sh` records the fit → ranking → exposure → observation sequence as run.

**Requirements.** The M4.6c ext checkpoints and snapshots, the M4.6c training archive and the M4.6d calibration and
ranking records (all git-ignored, all unchanged).

**Seeds:** the range 95,000–95,999:
- observation 95,000; exposure 95,100; natural labels 95,200; constructed labels 95,300;
- tail 95,400; fits and splits 95,500; archive sample 95,600; gradient batches 95,700.

**New files:**
- `lanegame/experiments/rl/m46e/`: `common`, `features`, `observation`, `exposure`, `labels`, `fit`, `analysis`,
  `ranking`, `report`, the CLI;
- `tests/test_m46e.py`;
- `docs/m4/m4_6/m4_6e_preregistration.md` and this report;
- `results/m46e/`:
  - `preregistration.json` (with amendment 1), `observation.json`, `exposure.json`, `analysis.json`, `ranking.json`;
  - the generated `report.md`, the stage logs and `run_queue.sh`.

**Modified:**
- `docs/rl_research_roadmap.md` (summary, milestone table, the forward plan, §AQ);
- `docs/m4/README.md`, `docs/README.md` (index rows);
- `README.md` (roadmap table).

No library code, rule, preset, default, reward, observation, test pin or checkpoint changed. The mathematical
reference is unchanged, because no rule or objective changed.

**Tests** (`tests/test_m46e.py`, 9). Run on the final code with the M4.6a–d and shared-trainer suites, 51 pass. They
cover:
- single-feature pairs change exactly the intended entries, and the entry check rejects a wrong block;
- the explicit log duel equals the state's duel ratio;
- the family classifier on a synthetic state;
- the label variants change only their quantity;
- the group split is deterministic and keeps groups whole;
- the ridge recovers a linear target;
- paired premiums on synthetic data;
- the rollout core reproduces an M4.6d calibration rollout (skipped without the records).
