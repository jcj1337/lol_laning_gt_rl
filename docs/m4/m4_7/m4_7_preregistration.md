# M4.7 Stage 2: pre-registration of the matched horizon pilot

*2026-10-16. Frozen (SHA-256 in `results/m47/preregistration.json`) after Stage 1 and the pilot smoke test, before
any pilot training run. Already seen at freezing time: every Stage 1 result
([report](m4_7_horizon_report.md) §1–§4, tables in `results/m47/report.md`), and the pipeline checks and throughput of
a 10-update smoke run of both arms (66 of 66 checks pass, run twice). No trained pilot network existed.*

## 1. Question

Does a shorter effective horizon (a smaller discount in the game's objective) change the agents' meaningful
early-lane decisions, and does it make them play the lane phase (turns 0–93, before about level 14) better against
the current-horizon agents, without unacceptable regressions in ordinary play?

This is a screening pilot. It does not adopt a discount, does not change any default or preset, and does not launch a
campaign. Bayesian opponent modelling is out of scope.

## 2. What Stage 1 established (the reasons for this design)

- **The lane phase is a fixed clock.** In natural self-play of the matched M4.6c 8.192M policies a seat reaches level 10
  at turn 51, level 12 at 71 and level 14 at 94 (p10–p90 90–99), in both curves (C1 − baseline −0.25 ± 0.37 turns) and
  across the training lineage (the M4.5c finals: 93).
- **Most training happens after it.** About 17% of training transitions (reconstructed occupancy, M4.6c start mixture)
  are before level 14; 83% after.
- **The objective weights it moderately.** Under γ 0.99, 61% of the objective's weight from s0 lies before turn 94; a
  decision taken inside the lane phase has on average 36% of its weight there.
- **Play in the lane phase is similar to later play** after the first 25 turns (income 0.41 vs 0.40 per turn, Shove
  35–39% vs 39–41% of wave choices), with more Safe and Wards and fewer deaths later.
- **Delayed rewards in measured benefits are modest and do not reverse anything** (paired 300-turn rollouts of the
  γ 0.99 policies, split at level 14):
  - constructed early leads: about 90% of a level-4 lead's premium arrives before level 14, about 75% of a level-8
    lead's, about half of a level-11 lead's; under H 60 the premiums shrink by 18–24% and keep their order;
  - natural early states: no tested alternative (Shove ×1, Shove ×4, Trade ×1, Safe ×1, reset now) beats the policy
    under H 100 or H 60; the after-L14 parts of these benefits are within ±0.13 and not significant;
  - early freeze cells: conceding and resetting cost 0.4–1.9 resources, almost all before level 14.
- **What Stage 1 cannot show.** Rescoring follows behaviour trained for γ 0.99. Whether agents trained for a
  lane-focused objective would choose differently (Recall and spending, trading for leads, freeze defence) needs
  training: the question this pilot answers.
- **Baseline and C1 are indistinguishable in the lane phase** except that C1's early earned-gold leads persist from turn
  50 to 94 while baseline's shrink by about a fifth (slope +0.04 vs −0.20; difference +0.24 ± 0.15).

## 3. The treatment horizon

**H = 60, γ = 1 − 1/60 = 0.98333** (the control is γ 0.99, H = 100).
- **It moves the objective materially toward the lane phase.** From s0, 79% of the weight lies before level 14 (61%
  now) and 92% before level 18 (78% now). A lane-phase decision has on average half its weight inside the phase (36%
  now).
- **It keeps the lane phase's own payoffs.** The measured early benefits mostly arrive within 20–50 turns (G50 is
  71–82% of the H 100 premium of a level-4 or level-8 lead; reset and HP costs are mostly realized by turn 20), and
  H 60 keeps 76–82% of the early premiums.
- **H 80 is too small a manipulation for a two-seed screen**: 69% of the s0 weight before level 14, and rescoring
  changes the best option in 6% of natural early states (13% at H 60).
- **H 40–50 discounts the lane phase's own 30–50-turn payoffs** (the level-4 lead's premium falls by a third at H 40)
  and approaches the roughly 17-turn credit window that GAE's λ 0.95 already imposes.
- If H 60 changes nothing meaningful, H 80 would not be expected to; if it does, H 80 is the natural follow-up.

| H | γ | weight from s0 before L10 / L12 / L14 / L18 | in-phase decisions' weight before L14 | weight beyond turn 300 |
|---|---|---|---|---|
| 100 (control) | 0.99 | 0.40 / 0.51 / 0.61 / 0.78 | 0.36 | 0.049 |
| 80 | 0.9875 | 0.48 / 0.59 / 0.69 / 0.85 | 0.42 | 0.023 |
| **60 (treatment)** | 0.98333 | 0.58 / 0.70 / 0.79 / 0.92 | 0.50 | 0.006 |
| 50 | 0.98 | 0.65 / 0.76 / 0.85 / 0.95 | 0.56 | 0.002 |
| 40 | 0.975 | 0.73 / 0.83 / 0.91 / 0.98 | 0.62 | 0.001 |

The independent-hazard reading of γ = 1 − 1/H is an ending time with mean H and median 0.69 H (41 turns for H 60).
It is broad: it does not concentrate the end of laning near turn 60 or near level 14.

## 4. Design

### 4.1 Arms (two matched seed pairs)

| item | control | short |
|---|---|---|
| game configuration | `GameConfig.m45_calibrated()` (γ 0.99) | `GameConfig.m45_calibrated(gamma=1 - 1/60)`: an opt-in keyword override of the existing `gamma` field; every other field identical (checked) |
| start | `ext/baseline:s`, the M4.6c 8.192M baseline continuation of seed s, s ∈ {0, 1} | the same checkpoint |
| everything else | identical: the shared `PolicyValueNet` (90 → 128 → 128 tanh); M4.6c's PPO (100% current-policy shared self-play, λ 0.95, 128-turn segments, reward scale 0.1, value clip 0.5, vf_coef 0.5, 4 epochs, minibatch 256, clip 0.2, entropy 0.03, gradient norm 0.5, 16 environments); M4.6c's mixed start distribution (the same archive, archive start probability 0.532, archive games 150 turns, s0 games 512, mirroring ½); a fresh Adam with lr 3·10⁻⁴ annealed linearly to 0 over the run; PPO seed 58,300 + s in both arms (the same start-state sequence, and the same first rollout) | |

**The objective change is applied consistently.**
- The trainer takes γ from the game configuration (`PPOConfig.continuing`), so the returns, the GAE advantages, the
  critic's targets and the bootstrapped truncations all use the arm's γ.
- `check_time_semantics` refuses a training γ that differs from the game's.
- Every evaluation scores both arms under the same objectives, from the same reward streams (§5).

**What is not done.**
- No random termination: the discount alone is the shorter objective, and game lengths stay at 512 and 150 turns,
  ending in bootstrapped truncations.
- No shorter rollout with the old bootstrap.
- No reward, mechanic, observation or start-distribution change.
- No second investment curve: the baseline is the established curve, and Stage 1 found no lane-phase difference that
  would favour C1 for this question.

**Lineage of every pilot policy.**
1. 16.384M fresh (M4.5c, s0 starts, γ 0.99).
2. 8.192M continuation (M4.6c baseline arm, archive starts, learning rate re-warmed, γ 0.99).
3. 4.096M continuation in this pilot (archive starts, learning rate re-warmed, γ of the arm).

### 4.2 Budget

- 4,096,000 agent transitions per run (1,000 updates, 50% of the starting policies' last continuation), four runs
  concurrent. The second smoke run measured about 2,900 transitions/s per run with three concurrent runs on a quiet
  machine, so the four runs take about 25–35 minutes.
- The budget is the screening size the study brief proposed. The measured cost would allow more, but a larger first
  budget would be chosen before seeing any result for no stated reason; the midpoint head to head reports whether the
  arms were still separating.
- Snapshots every 25,600 transitions. The midpoint (2,048,000) is evaluated head to head only, to show the direction of
  travel.
- **No extension rule.** If the result is inconclusive, the report says so; no further training starts from this
  protocol.

### 4.3 Smoke checks (all passed before freezing)

The configurations differ only in γ; the control equals `m45_calibrated()`; `PPOConfig.continuing` takes each arm's γ
without a mismatch opt-in; run.json records the arm's γ for the game and for PPO; GAE receives the arm's γ and λ 0.95
(an in-process spy), no termination reaches GAE, every truncation is bootstrapped with a nonzero critic value, and the
advantages differ from those under γ 0.99 on the same data; the first update runs at lr 3·10⁻⁴ and the tenth at the
schedule's value; the policy moves (KL > 0, below 0.05); the first rollout is identical in both arms and the arms
diverge from the first update; the short arm's re-run is bit-identical; initial weights equal `ext/baseline:0`; archive
games start; no illegal action; no stop condition.

## 5. Evaluation

All games are played under the baseline mechanics (play does not depend on γ). Every game stores the per-turn reward
stream, so each readout below is computed from the same trajectories. **No score is bootstrapped with a critic.** Raw
discounted returns under different γ are never compared with each other; every comparison is between policies under
one common objective.

Common objectives: the undiscounted earned-gold lead in elapsed-turn windows (0–24, 25–49, 50–74, 75–93, 94–149,
150–249, 250–399; "lane" = 0–93, "after" = 94–399); a level-triggered window (until the lower level reaches 14); and
the discounted lead under each H ∈ {100, 80, 60, 50, 40} over 400 turns (300 in the paired families).

| family | design | readouts |
|---|---|---|
| **H2H (primary)** | short vs control finals from s0, paired seats (each policy in each seat on the same chance seed), 1,024 game pairs per seed, 400 turns; midpoints 256 pairs | **P1 = the short arm's lane lead (turns 0–93)**; every window, the level-triggered window, G under every H |
| fixed opponents | each final (and the common start) from s0 against the common start (256 pairs), the other seeds' M4.6c baseline finals `ext/baseline:o`, o = 2–5 (64 pairs each), and the nine M4.5c scripted references (32 pairs each) | lane and after leads, G100, G60; short − control on the same games |
| natural behaviour | 96 self-play games of 400 turns per policy (short, control, start), the same game seeds | per-game metrics in the lane and after windows (§6.2): Recall choices and completions, HP and unspent gold at Recall, wave and behaviour shares, deaths, freezes held, the frozen-on side's Shove and Safe, Wards, Pressure, plates, investment and unspent gold at turn 94, income |
| paired families | Stage 1's three families for each final: constructed lead cells (L4, L8, L11 × +4, +8 × three lanes, K 32), natural early states (24 per seed, own self-play, Shove ×1, Shove ×4, Trade ×1, Safe ×1, reset, +4 invested, −0.3 HP; K 12), early freeze cells (levels 4 and 8, freezer and breaker plans, K 24); 300 turns | premiums (conversion of plausible advantages), the alternatives' benefits against each policy, freeze defence and contesting, all under H100 and H60; Stage 1's start-policy results on the same states and seeds as the reference |

Units: the two training seeds are reported separately; within a seed, 95% normal intervals over games (paired game
indices). The classification requires both seeds to agree. Seeds: 58,000–58,999.

## 6. Criteria

### 6.1 Leads

- **Materiality M = 0.25 resources** (about 2.5 turns of one side's income over the 94-turn lane window; the
  lane-window income threshold of §6.2 corresponds to 0.21).
- **Benefit (B):** P1 ≥ +M in both seeds, with each seed's interval excluding 0.
- **Lane regression (LR):** P1 ≤ −M in both seeds, intervals excluding 0.
- **Ordinary-play regression (R):** against the pooled fixed opponents (other-seed finals and scripted references),
  short − control ≤ −M in both seeds with intervals excluding 0, under G100 over 400 turns or in the after window.

### 6.2 Behaviour

A lane-window metric changes **meaningfully** when short − control has the same sign in both seeds, exceeds the
metric's threshold in both, and each seed's interval excludes 0. Thresholds are the across-seed standard deviations
of the six independently trained M4.6c baseline policies (Stage 1 records), so a meaningful change is larger than the
difference between two separately trained baseline seeds:

| metric (lane window, per game, both seats) | threshold |
|---|---|
| Recall choices per 100 seat-turns | 0.56 |
| completed Recalls per 100 seat-turns | 0.48 |
| HP fraction when Recall is chosen | 0.037 |
| unspent gold when Recall is chosen | 0.50 |
| Shove share of wave choices | 0.033 |
| Hold share of behaviour choices | 0.010 |
| Yield share | 0.017 |
| Trade share | 0.016 |
| Safe share | 0.014 |
| deaths per 100 seat-turns | 0.14 |
| share of seat-turns holding a freeze | 0.005 |
| Shove share when frozen on (contesting) | 0.062 |
| Ward share of present turns | 0.004 |
| Pressure share of present turns | 0.004 |
| invested gold at turn 93 | 0.50 |
| unspent gold at turn 93 | 0.50 |
| income per turn | 0.0022 |

The code computes them from the Stage 1 records (`behaviour.thresholds`), unrounded; the table rounds them.

### 6.3 Classification and recommendation

| outcome | condition | recommendation (not an adoption) |
|---|---|---|
| 1 benefit | B and not R | pursue the shorter horizon: confirm with more seeds, ideally from scratch, before any adoption |
| 2 trade-off | B and R | a modelling decision for the user: the lane phase gains, ordinary play loses |
| 3 behaviour only | P1 small or not significant in both seeds, at least one meaningful behaviour change, no R | unresolved; keep the current horizon unless the behaviour change itself is wanted |
| 4 no meaningful change | P1 small or not significant in both seeds, no meaningful behaviour change, no R | keep the current horizon |
| 5 regression | LR, or R without B | keep the current horizon |
| 6 inconclusive | anything else (for example, the seeds disagree) | the question stays unresolved |

Every other readout (windows, fixed-opponent details, paired families) is descriptive and cannot change the outcome.

## 7. Limitations known in advance

- **Two seeds.** Between-seed intervals are not computed (one degree of freedom); a pattern is claimed only when both
  seeds agree.
- **Continuation, not fresh training.** Both arms start from policies trained for 24.6M transitions at γ 0.99. The
  short arm must re-fit its critic to the new objective and may adapt only partly within 4.096M; a null result bounds
  the effect of this continuation, not of a from-scratch campaign.
- **The re-warmed learning rate** costs both arms some strength at the start (about 0.5 resources over the first million
  transitions, M4.6c); the matched control absorbs it.
- **Occupancy is unchanged.** A smaller γ reweights each decision's objective; it does not change which states training
  samples (83% after level 14 in both arms).
- **GAE λ 0.95** already limits the advantage's direct credit window to about 17 turns; a longer horizon acts through
  the critic, whose strength blindness (M4.6e) limits how much a horizon change can move strength-dependent decisions.
- **One curve.** The result says nothing directly about C1.
