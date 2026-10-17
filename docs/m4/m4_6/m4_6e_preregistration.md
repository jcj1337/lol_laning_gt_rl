# M4.6e — pre-registration of the supervised value study

*2026-10-16. Frozen (SHA-256 in `results/m46e/preregistration.json`) after the observation, exposure and label stages
were implemented and the labels were being generated, and before any supervised fit. Already seen at freezing time:
the controlled single-feature pairs (Q1). All 54 pass the encoding check. Under C1 the critic moves +0.07 for +8
invested resources and about 0 for a full level, against ±0.4–0.6 for HP and wave changes. No fit, readout or
exposure number had been seen.*

## Scope

A diagnostic of value learnability. There is no PPO, no actor training, and no change to any rule, observation,
reward or default. C1 (`m45_calibrated(invest_gain=3.24, invest_scale=64.0)`) is the evaluation environment and is
not adopted.

## Labels (policy-specific)

- **Policies.** The six 8.192M C1 general policies `ext/C1:s`, each playing itself.
- **Label.** G300 = the seat's discounted (γ 0.99) earned-gold lead over 300 turns, no bootstrap, in resources.
- **Sets.** Each policy gets its own set; there is no pooling across policies.
  - Natural: 32 s0 games of 600 turns, labels every 10 turns from turn 5 to 295.
  - Constructed: a fixed stratified sample of 200 M4.6c training-archive states (40 natural, 40 natural+lead,
    60 zone, 60 freeze), each with paired variants (orig, lead removed, A one level up, B at 0.6 of its HP,
    A's investment multiplier ×1.3–1.9 over B's). Two rollouts of 450 turns per variant on common seeds, labelled
    every 15 turns from turn 0 to 150.
  - External test: the M4.6d calibration states of each policy (constructed freeze / lane / gap cells and natural
    lead / even states, K 32, H 300 / 500), never used for fitting.
- **Horizon.** The residual beyond 300 turns is measured separately on eight problematic constructed cells (three
  policies, K 32, H 800, new seeds).

## Arms (per policy)

- **A.** The existing critic, unchanged (V / 0.1).
- **B.** A ridge readout on the frozen trunk's last hidden layer (128 tanh units). The ridge strength is chosen on
  validation groups.
- **C.** A freshly initialized value MLP of the critic's shape (90 → 128 → 128 → 1, tanh) on the observation. Adam,
  early stopping on validation groups, three fit seeds.
- **D.** C with six observation-derived strength features appended (log duel ratio from the two attack-power entries,
  both HP fractions, log HP ratio, HP-adjusted log duel, both-present × log duel). These are deterministic functions of
  the observation: D tests an inductive bias, not added information.
- **Reference.** A ridge model on the observation plus the same six features.

Both seats of every labelled state enter as two examples (seat B: −G300). Natural and constructed examples carry
equal total weight.

## Splits (groups keep mirrored seats, variants, rollouts and trajectory points of one source together)

1. **main.** 25% of groups held out (natural: by game; constructed: by archive state), the same groups for every
   policy; 15% of the remaining groups for validation.
2. **strength.** main, with every training example whose duel ratio is ≥ ×1.6 in either direction removed; tested
   on the held-out examples beyond ×1.6 and the M4.6d large-gap cells.
3. **family.** main, with every constructed freeze group removed from training; tested on the freeze groups and the
   M4.6d freeze cells.

## Primary readouts (main split; mean and 95% t interval over the six policies)

- **P1. Strength-premium recovery** on the M4.6d cells (realized from their K 32 labels).
  - Ratio = Σ predicted premium / Σ realized premium, for:
    - "edge": edge − equal, freeze L3/5/8 and lane L4/8;
    - "plaus": plaus − equal, freeze;
    - "large": ×1.76 − equal, freeze and lane.
  - Also the absolute bias on the ×1.5 / ×1.76 gap cells.
- **P2. Natural accuracy.**
  - MAE on the held-out natural games.
  - Bias and slope on the M4.6d natural states (G500 labels).
- **P3. Paired premiums** on held-out archive groups: lead, level, HP, big. Mean predicted / mean realized, CRN-paired.
- **Secondary.**
  - Calibration slope, R² and the label-noise floor (from the paired rollouts).
  - The strength and family splits.
  - Look-ahead ranking at the M4.6d decision states. The fitted predictor replaces the critic in r0 + γV(s1); the
    readouts are Spearman, sign agreement and the same best action against the stored K 16 rollouts.

## Decision rules (fixed now)

**An arm "recovers strength value"** when all three hold:
- the P1 ratio is ≥ 0.6 for edge + plaus together;
- the P1 ratio is ≥ 0.6 for large together with the gap cells' value;
- its natural MAE is ≤ 1.10 × arm A's on the same held-out natural points.

**Interpretation.** These narrow the explanation; none identifies a unique cause.

| result | reading |
|---|---|
| B recovers | the trained representation encodes strength value linearly; the value head and PPO value training did not use it |
| B fails, C recovers | the observation supports the value on this distribution; the trained trunk does not represent it linearly |
| only D recovers | an inductive-bias / scaling issue, not missing information |
| none recovers | label noise, data coverage or capacity limits; no learning pilot is justified on this evidence |

**Recommendation rule.** Recommend at most one targeted learning pilot, and only if all three hold:
1. Some arm recovers strength value without natural regression.
2. With that predictor, the look-ahead ranking at the M4.6d decision states improves materially (freezer-at-a-freeze
   Spearman +0.2 or more, or same-best-action +15 points or more).
3. The exposure analysis identifies a mechanism a single training-side change could act on.

Otherwise, document the limitation and proceed to M4 closeout (stopping principle). A concrete measurement or
implementation defect found by Q1 takes precedence: then the recommendation is a narrowly scoped fix.
