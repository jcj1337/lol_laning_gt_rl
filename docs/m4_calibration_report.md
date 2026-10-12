# M4.2b Calibration Report — Gank Severity, Item Strength and the Shove → Recall Cycle

*Branch `calibration/m4-2b` (from `implementation/m4-integrated`, 3b15b41), 2026-10-12. Exploratory calibration under
a fixed compute budget. **The M4 defaults are unchanged** (`GameConfig.m4()` still has `gank_damage_position = 0.60`,
`item_mult = 1.20`). The candidates below are presented for review. Raw results: [results/m4_calibration/](../results/m4_calibration/README.md).
Driver: `lanegame/experiments/m4/calibration.py`. Every number is a scripted-agent measurement, not an equilibrium value.*

## Summary

- **Ganks are not the main reason pushing loses.** Over 400 turns against hold_farmer, the presser loses −17.1 at
  baseline. The payoff decomposes into farm **−16.2**, bounties −1.2 and plates +0.4. Cutting the conditional gank
  damage from 0.60 to 0.30 cuts the presser's deaths from 2.7 to 1.0 per 100 turns, but improves its value by only
  **+1.5**. Gank *frequency* is unchanged; only *severity* falls.
- **The shove penalty is mostly lane economics.** Two mechanisms cost the pusher farm, and no parameter in scope
  touches either:
  - when a shoved wave crashes, the defender farms it under its own tower (0.7–0.9 resources per turn in the traces)
    while the pusher collects nothing;
  - after a push the lane settles into the stable parked configuration of audit §1.1. The pusher then farms next to
    the enemy tower at efficiency 0.5, and the defender farms at its own tower.

  A persistent shover farms **14.8 vs 32.2 per 100 turns**.
- **Items are weak for three reasons, so "both" understates it:**
  1. **No trades, no value.** Items act only on trade damage (and Pressure). In passive play a free tier is worth
     exactly 0.
  2. **Insufficient power at breakpoints.** At 1.20, the first tier does not reduce hits-to-kill at levels 6 and 11
     (6 → 6).
  3. **Short-lived leads.** In trading matchups an item lead lasts 7–11 turns, because deaths and recalls bring both
     champions to the shop.

  A higher multiplier raises the value of an item where trades happen. Over a controlled 40-turn window it rises from
  +0.62 to +1.25 to +1.62 at 1.20 / 1.35 / 1.50. It does **not** make affordability matter to the Recall decision:
  the interaction is ≈ 0 ± 0.8 at every multiplier.
- **The conditional Shove → Recall → Purchase agent** (new, deterministic) beats the permanent presser by +15. It still
  loses to every passive or trading baseline (−4 to −7 per 400 turns). The counterfactuals show decisions with
  opposite signs, except one: **Shove before Recall never significantly improved the outcome** in any tested state.
- **The confirmed effects of the candidates (W = 1,000, 30 seeds) are modest:**
  - gank 0.40 adds +1.26 [+1.12, +1.39] for the presser;
  - item 1.35 adds +2.30 [+1.73, +2.87] for aggressive_trader against hold_farmer. This is the largest effect of any
    candidate, and it favours aggressive trading over passive farming;
  - no matchup changes sign, and no strategy becomes dominant: aggressive trading still loses to its edge-trading
    counters (−1.68 ± 0.81).
- **Freeze breaking.** The prototype (experiment only) uses the pre-crash front as the farm spot. It reproduces the
  design's lane-only parked table in **all 64 cells**, and it is worth +1.49 ± 0.11 to the presser even against a
  non-freezer. It is a rule change and is proposed separately (§9).
- **Recommendation (for review, not adopted):** `gank_damage_position = 0.40`, `item_mult = 1.35`, with
  `0.40 / 1.20` as the conservative alternative. Neither fixes the passive-equilibrium risk. The levers that matter
  are structural: crash farm, the farm-spot rule, and item persistence (D4).
- **Budget:** 1,976,360 ledgered turns; 64 s wall clock on 8 processes; 346 s summed worker CPU. In addition, about
  0.1M turns of development smoke runs and state searches ran outside the ledger, so the ~2M turn cap was reached.
  The full `--runslow` suite was **not** run (§8).

---

## 1. Setup and discipline

- **Candidates.** Each candidate is `GameConfig.m4(gank_damage_position=g, item_mult=i)`, tag `g{g}_i{i}`. Nothing
  else changes.
- **Pairing.** Every configuration plays the same seeds from the same start states (common random numbers). Head-to-
  head values put the focal agent in both seats on each seed, as `evaluate_discounted` does. So a mirror matchup
  scores exactly 0, and candidate − baseline differences are paired.
- **Counterfactual branches.** These use `diagnostics.rollout` with `rollout_seed` streams: identical chance vectors
  turn by turn until the trajectories diverge.
- **Windows and truncation.** The guaranteed truncation bound is γ^W·Ω, with Ω = 68.34:

  | W | 1,000 | 400 | 200 |
  |---|---|---|---|
  | guaranteed bound per value | 0.003 | 1.23 | 9.2 |

  The bound applies per value. Paired differences in practice lie far inside it, but the W = 200 / 400 numbers are
  **windowed** measurements, labelled as exploratory.
- **Tail-bound applicability (checked).** Ω depends only on the farm values, the bounty, the plates and the lane's
  A_max. The lane signature is unchanged by `gank_damage_position` and `item_mult`, so `recorded_bounds` applies and
  Ω is identical for every candidate (asserted in `tests/test_m4_calibration.py`). The per-turn bound rises with
  `item_mult` through the Pressure table and is recomputed from the configuration.
- **Two-stage sampling:**
  - screening with 8–16 paired seeds at W = 40–460;
  - finalist confirmation with 30 paired seeds at W = 1,000.

  Non-significance is **never** read as equivalence.

## 2. Budget and the complete list of runs

| Stage | Purpose | Configs | Scale | Turns | Wall s | CPU s |
|---|---|---|---|---|---|---|
| A (invalid, discarded) | gank screening; the seat-B seed stream was wrongly advanced (mirror ≠ 0). Fixed and rerun | 3 | — | 115,200 | 3.8 | 18.6 |
| A | gank screening | g ∈ {0.60, 0.40, 0.30}, i 1.20 | 4 matchups × 12 seeds × 2 seats × 400 | 115,200 | 4.0 | 19.3 |
| B1 | intrinsic item value | i ∈ {1.20, 1.35, 1.50}, g 0.60 | 4 snapshots × 3 continuations × 12 seeds × 40 turns (+ 200 turns for edge traders) | 92,160 | 3.4 | 19.3 |
| B2 | realized item value | same | 4 matchups × 12 seeds × 2 × 400 | 115,200 | 4.0 | 18.5 |
| R | affordability × Recall | same | 3 states × HP {4, 6}/6 × afford/poor × recall/hold × 12 × 200 | 172,800 | 5.7 | 36.2 |
| CD | combined candidates + cycle head-to-head | baseline, g0.40_i1.35, g0.30_i1.35, g0.40_i1.50 | 11 matchups × 8 seeds × 2 × 400 | 281,600 | 7.3 | 48.1 |
| DCF | cycle counterfactual plans | baseline, g0.40_i1.35 | 5 states × 2 HP × 2 gold × 4 plans × 10 × 200 | 320,000 | 9.8 | 68.9 |
| F | finalist confirmation + snowballing | baseline, g0.40_i1.20, g0.40_i1.35 | 3 matchups × 30 seeds × 2 × 1,000; 50 mirror games × 300 | 585,000 | 16.8 | 95.6 |
| FB | Freeze-breaking prototype | baseline | 64-cell parked table; 5 matchups × 2 variants × 8 × 2 × 400 | 64,000 | 4.0 | 16.0 |
| S4 | targeted S4 checks of the recommendation | baseline, g0.40_i1.35 | dominance 2 matchups × 12 × 2 × 400; Recall-sign 48 variants × 8 × 200 | 115,200 | 5.5 | 5.6* |
| **Total (ledger)** | | | | **1,976,360** | **64.3** | **345.9** |

\* The Recall-sign part of S4 ran through `economy.run_job`; its worker CPU time was not captured.

**Outside the ledger:** about 0.1M turns of development smoke runs and of state searches (snapshot generation for B1,
R, DCF and S4).

**Parameter combinations tested (gank damage position / item multiplier):**
- 0.60 / 1.20 (baseline);
- 0.40 / 1.20 and 0.30 / 1.20 (Phase A; 0.40 / 1.20 again in F);
- 0.60 / 1.35 and 0.60 / 1.50 (Phase B);
- 0.40 / 1.35, 0.30 / 1.35 and 0.40 / 1.50 (Phase C, three of the four allowed combinations; 0.30 / 1.50 was not run).

The budget is the binding constraint. Wall clock was never close to the 15-minute cap.

## 3. Baseline measurements (M4 defaults; W = 400 unless noted)

| Matchup (x vs hold_farmer) | value to x | x ganks /100 | gank trials per gank (of 4) | x deaths /100 (all with gank damage) | x recalls /100 | x farm /100 | hold farm /100 | x plates /100 | payoff components (farm / plate / bounty) |
|---|---|---|---|---|---|---|---|---|---|
| hold_farmer (mirror) | 0.00 | 6.80 | 1.95 | 0.46 | 1.80 | 30.4 | 30.4 | 0 | 0 / 0 / 0 |
| presser | −17.05 ± 0.29 | 11.85 | 3.08 | 2.74 | 3.09 | 14.8 | 32.2 | 0.68 | −16.23 / +0.40 / −1.21 |
| cond_shove | −15.80 ± 0.26 | 10.35 | 2.78 | 1.88 | 2.75 | 15.6 | 34.0 | 0.75 | −16.79 / +1.86 / −0.87 |
| edge_trader | +0.50 ± 0.44 | 7.50 | 2.14 | 1.09 | 2.56 | 25.9 | 26.0 | 0 | −0.24 / 0 / +0.74 |
| shove_recall (new, CD) | −7.12 ± 2.00 | — | — | 0.58 | 2.31 | 24.0 | 32.0 | 0.38 | −7.31 / +0.36 / −0.16 |
| aggressive_trader (F, W = 1,000) | +2.39 ± 0.64 | — | — | 2.52 | 3.01 | 17.5 | 16.3 | 0 | +0.57 / 0 / +1.82 |

In every Phase A row, every death of x involved gank damage (gank deaths = deaths).

## 4. Phase A — gank severity vs gank frequency

| g (damage position) | presser value | Δ vs 0.60 (paired) | presser ganks /100 | trials per gank | presser deaths /100 | cond_shove Δ | edge_trader Δ | hold mirror deaths /100 |
|---|---|---|---|---|---|---|---|---|
| 0.60 | −17.05 | — | 11.85 | 3.08 | 2.74 | — | — | 0.46 |
| 0.40 | −15.92 | **+1.13 [+0.93, +1.32]** | 12.36 | 2.34 | 1.52 | **+0.89 [+0.66, +1.12]** | +0.30 [−0.39, +0.99] | 0.29 |
| 0.30 | −15.52 | **+1.52 [+1.36, +1.69]** | 12.84 | 1.94 | 1.02 | **+1.36 [+1.02, +1.70]** | +0.25 [−0.26, +0.76] | 0.22 |

**Confirmed (12 seeds × 2 seats; at W = 1,000 with 30 seeds: +1.26 [+1.12, +1.39] at 0.40):**
- **Frequency vs severity.** The parameter changes severity only.
  - Gank frequency is unchanged; it rises slightly, because a champion that survives spends more turns forward.
  - Conditional damage falls from 3.1 to 1.9 trials of 4 under the enemy tower.
  - Gank deaths of forward players fall 2–3×.
- **Is forward punishment economically disproportionate?** Only partly. Even with severity halved, the presser keeps
  ≥ 90% of its loss, because that loss is farm (§3, payoff components).
  - The opportunity pushing creates is small: plates are worth +0.4 (presser) to +1.9 (cond_shove) per 400 turns.
  - The farm pushing gives up is −16.
  - The imbalance lies in the lane economy (crashed waves farmed by the defender, the parked configuration), not in
    gank severity.

## 5. Phase B — items

### B1: intrinsic value (controlled)

**Setup.**
- Start states: 4 reachable edge-trader snapshots (both champions present, focal below the item cap).
- Gold: set to 0 for both, so neither can buy early in the window.
- Treatment: the focal gets +1 tier.
- Sample: 48 paired rollouts per cell. The 40-turn window holds the item lead for ~31 of 40 turns.

| continuation | i 1.20 | i 1.35 | i 1.50 |
|---|---|---|---|
| hold_farmer mirror, W 40 | 0.000 (exactly; no trades) | 0.000 | 0.000 |
| edge_trader mirror, W 40 | −0.73 ± 0.54 | −0.25 ± 0.53 | +0.05 ± 0.68 |
| edge_trader mirror, W 200 | +0.22 ± 0.90 | +0.71 ± 1.05 | +1.76 ± 1.12 |
| aggressive_trader vs hold_farmer, W 40 | **+0.62 ± 0.40** | **+1.25 ± 0.43** | **+1.62 ± 0.50** |

**Mechanism** (counters in `stage_B1.json`). With a tier the edge trader trades more: connected trades rise from 9.9
to 14.7 per 40 turns, and trade damage dealt from 1.28 to 2.13 max-HP, at 1.20. It also stands forward more and dies more (1.71 → 2.00). Over 40 turns
the extra aggression cancels the extra damage; at 1.50 and W = 200 the item wins (+1.76, borderline).

**Breakpoints (analytic, hits to kill from full HP, single hits, equal levels; tiers N0 / N1 / N2):**

| level | 1.20 | 1.35 | 1.50 |
|---|---|---|---|
| L6 | 6 / 6 / 5 | 6 / 5 / 4 | 6 / 5 / 3 |
| L11 | 6 / 6 / 5 | 6 / 5 / 4 | 6 / 5 / 3 |
| L18 | 7 / 5 / 5 | 7 / 5 / 4 | 7 / 4 / 3 |

At 1.20 the first tier is invisible in kills through the mid levels. At 1.50 two tiers halve the kill time (2.25× damage).

### B2: realized value (normal purchases from s0, 400 turns)

| matchup | first / second purchase (median turn) | turns with unequal items | item-lead spell (mean turns) | early buyer's next 60 turns |
|---|---|---|---|---|
| hold mirror | 43 / 55.5 | 12% | 38.7 | +0.61 ± 0.65 (no combat value) |
| gold_recaller vs hold | 35 / 48 | 10% | 28.0 | +0.14 ± 0.60 |
| edge mirror (1.20 / 1.35 / 1.50) | 35–35.5 / 56–59 | 4.3 / 4.0 / 3.5% | 8.9 / 8.3 / 7.2 | −0.59 ± 0.90 / +1.14 ± 1.39 / +0.43 ± 1.73 |
| aggressive vs hold (1.20 / 1.35 / 1.50) | 27.5 / 60.5–61 | 5.3 / 4.3 / 4.5% | 11.5 / 9.5 / 10.1 | +0.30 ± 1.06 / −0.64 ± 1.31 / −2.00 ± 1.60 |

- **Purchase timing does not depend on the multiplier**, because income is unchanged.
- **Item leads are shortest exactly where items have value.** In trading play they last 7–11 turns; base visits
  through deaths and recalls equalize items.
- **Earlier purchases:** no measurable advantage at this sample size (inconclusive, n = 20–24 games per cell).
- **aggressive_trader vs hold_farmer:** the value rises with the multiplier: +1.99, +3.95, +5.49 (W = 400).

### Does purchasing influence the Recall decision?

**Stage R** compares Recall-now minus Hold-now with an affordable tier against the same comparison without one. The
interaction, pooled over 3 states × 2 HP levels (72 pairs, W = 200, edge-trader continuations), is:

| multiplier | 1.20 | 1.35 | 1.50 |
|---|---|---|---|
| interaction | +0.05 ± 0.78 | +0.02 ± 0.83 | −0.28 ± 0.73 |

No detectable effect at any multiplier. Combined with S4-B (§10), affordability still does not move the Recall sign;
HP does.

**Answer to the assignment's question: items are weak because of insufficient power at breakpoints AND short-lived
leads, and fundamentally because they act only through trades.** Raising the multiplier fixes the first part only.
It makes items matter in fights, but it neither lengthens leads nor creates a Recall-to-buy motive.

## 6. Phase C — combined candidates (W = 400, 8 seeds × 2 seats; paired vs baseline)

| matchup | baseline | g0.40_i1.35 Δ | g0.30_i1.35 Δ | g0.40_i1.50 Δ |
|---|---|---|---|---|
| presser vs hold | −17.14 | **+1.18** [+0.95, +1.41] | **+1.64** [+1.37, +1.92] | **+1.32** [+1.16, +1.48] |
| cond_shove vs hold | −15.83 | **+0.97** | **+1.40** | **+0.98** |
| shove_recall vs hold | −7.12 | +1.62 [−0.73, +3.98] | +0.97 [−1.56, +3.50] | +1.62 [−0.73, +3.98] |
| aggressive_trader vs hold | +2.06 | **+2.69** [+1.70, +3.68] | **+2.99** [+1.63, +4.34] | **+4.06** [+3.11, +5.02] |
| edge_trader vs hold | +0.39 | **+0.80** [+0.35, +1.24] | **+1.24** [+0.73, +1.75] | **+1.25** [+0.58, +1.91] |
| gold_recaller vs hold | −0.68 | +0.08 | −0.06 | +0.08 |

**Interactions:**
- **Reduced gank severity and stronger items compound for traders.** Trading at full strength is also safer when ganks
  hurt less. aggressive_trader's edge over hold_farmer triples at g0.40_i1.50 (+2.06 → +6.13), and hold_farmer deaths
  rise from 6.1 to 8.9 per 100 turns. This is the **potentially harmful interaction**: overpowered trading against
  passive play.
- **Items need trades.** For trade-free matchups g0.40_i1.35 and g0.40_i1.50 give *identical* values, because items
  change nothing without trades.
- **No sign changes.** No matchup changes sign under any combination, so strategic diversity does not increase; the
  ranking shifts slightly toward trading.
- **Snowballing (F, 50 mirror edge-trader games per config, 300 turns):**

  | config | P(win \| ahead at 30) | correlation | level gap ≥ 2 at turn 100 |
  |---|---|---|---|
  | baseline | 0.75 | 0.65 | 0% |
  | g0.40_i1.20 | 0.56 | 0.46 | 2% |
  | g0.40_i1.35 | 0.67 | 0.63 | 6% |

  **Inconclusive** at n = 50 (SE ≈ 0.09). No configuration approaches the audit's 90% threshold.

## 7. Phase D — the conditional Shove → Recall → Purchase cycle

### The agent

`agents_m4.ShoveRecallAgent` is deterministic and stateless. Its rules, in order:
1. Low-HP Recall.
2. When a tier is affordable:
   - stay while an enemy wave threatens the tower;
   - Recall once the own wave has crashed or stands at x ≥ 1;
   - otherwise Shove first (at HP ≥ 2/3 with a favourable wave);
   - otherwise Recall at HP ≤ 1/2, or keep farming.
3. When not affordable:
   - farm;
   - push a wave already parked at x ≥ 1 through ("finish the push");
   - Shove a stacked wave (≥ 8 HP).

It never looks at the opponent's identity, and was not tuned against any baseline. One rule was added after the
first smoke run: "finish the push". Without it, a single Shove trapped the agent in the parked configuration for the
rest of the game (−46 vs −36 per 400 turns against hold_farmer; development run).

### Head-to-head (baseline rules)

| opponent | W = 400, 8 seeds | W = 1,000, 30 seeds |
|---|---|---|
| hold_farmer | −7.12 ± 2.00 | −5.93 ± 1.03 |
| presser | **+15.19 ± 0.49** | — |
| gold_recaller | −5.26 ± 2.85 | — |
| edge_trader | −4.14 ± 2.20 | — |
| freezer | −6.06 ± 2.36 | — |

The variant with Pressure scores −6.42 ± 1.58 against hold_farmer. The candidates do not change these numbers
significantly; at W = 1,000 the change is −0.53 [−1.78, +0.72].

### Counterfactuals

**Setup.**
- States: 5 reachable states (wave even / own wave pushing / enemy wave threatening; items 0 or 1).
- Variants of each state: HP full or half × gold affordable or 0.
- Plans of the focal agent (then the cycle policy against hold_farmer, W = 200, 10 seeds):
  - farm 8 turns;
  - Recall now;
  - Shove, then Recall;
  - Shove 4 turns, then the policy.
- Configurations: baseline and g0.40_i1.35.

Opposite signs, as requested:

| decision | evidence (significant paired differences, 95%) |
|---|---|
| **Recall to purchase is better** | even state, full HP, affordable: Recall − farm = **+7.09 ± 2.81** (items 0), **+7.04 ± 2.77** (items 1); enemy wave threatening, full HP, affordable: **+4.06 ± 3.78** (g0.40_i1.35: +5.61) |
| **Staying and farming is better** | even state, half HP, affordable: **−0.37 ± 0.27**; under g0.40_i1.35, own wave pushing, affordable: **−1.95 ± 1.92** (full HP), **−3.26 ± 2.23** (half HP); even, full HP, not affordable: **−2.45 ± 2.20** |
| **Shove before Recall improves the outcome** | **not found.** 0 significantly positive cells. Best: +1.17 ± 1.64 (baseline) and +1.41 ± 3.63 (candidate). Significantly negative in 5 of 20 baseline cells: threatening wave −5.09 / −5.19; even, half HP −5.32 |
| **Shove is too risky** | threatening wave, half HP: Shove − farm = **−5.44 ± 3.27** (candidate); even state, half HP, items 1: **−1.59 ± 0.58** |
| *(extra)* **Shove pays: finishing a parked push** | own wave pushing: Shove − farm is positive in 8 of 8 baseline cells (+1.5 to **+8.8**), significant in 7 |

**Caveat (confound).** In these trade-free continuations a tier has no combat value. The large Recall − farm gaps in
affordable states therefore come from how the *continuation* behaves with gold: delaying lets the cycle agent's own
shove-before-recall run later, which is costly. They do not come from the item. The counterfactuals show the
*decision structure* of the cycle, not item value.

## 8. Targeted code changes and tests

**1. Stateful evaluation (`env/discounted.py`)**
- `_play` now calls each agent's `reset()` before a game and `notify(info, side)` after every turn, when the agent
  defines them.
- A stateful agent playing both seats is deep-copied for seat B (`game_agents`).
- An optional `on_step(info)` observer was added for counters.
- Stateless agents are unaffected (the existing evaluator tests pass unchanged).
- **Regression test** (`test_m4_time.py::test_stateful_agents_get_lifecycle_hooks_in_the_discounted_evaluator`). From
  the witnessed equal-power freeze, the AdaptiveBreaker Slow Pushes until the first maintained Freeze and Shoves after
  it. It starts every game unaware, even with stale memory. Without `notify` it never Shoves. Repeated evaluations on
  the same seeds agree.

**2. Training discount factor (`rl/ppo.py`)**
- New field `PPOConfig.allow_gamma_mismatch` (default False).
- Under continuing time, a training γ different from the game's γ now **raises** `ValueError`, unless the field is
  True. With the opt-in the run is silent, and the choice is recorded in the run configuration.
- Backward compatibility:
  - configurations saved without the field load as False;
  - finite (legacy) games are unaffected;
  - `PPOConfig.continuing` always matches.
- The old warning test now asserts the error and the silent opt-in.

**3. New agent:** `ShoveRecallAgent` (`env/agents_m4.py`). Stateless, so it is safe for the rollout probability cache.

**4. New test file:** `tests/test_m4_calibration.py` (11 tests):
- the defaults are unchanged;
- each candidate validates, and the recorded lane bounds and Ω are unchanged;
- invariant soaks with exact observation decoding on random and cycle play (S1 checklist: zero-sum, exact reward
  decomposition, per-turn bound, minion conservation, ledger, purchase timing);
- resolver mirror equivariance under `mirror_chance` and `swap_seats`;
- deterministic replay with and without the lane cache;
- legality, statelessness and cycling of the new agent;
- the keep-reach prototype reproduces the design table, and stays experiment-only.

**Test results:**
- `tests/test_m4_calibration.py` and the changed tests: all pass.
- Default suite (`pytest tests`, slow tests skipped): **687 passed, 4 skipped, 2 xfailed** in 6 min 51 s. The two
  strict expected failures of M4.2 are preserved, and the legacy pins pass.
- **Not run:** the full `pytest --runslow` suite (~19 min), and with it the slow exhaustive observation and lane-bound
  enumerations (O1/O2, B1). They are lane-only and unaffected by the two parameters; final full-suite verification
  is **pending**.

## 9. Freeze-breaking finding (M4.2 §5, finding 1)

**Mechanism.** In the probe pass (design §5 step 1, M2's unchanged rule), the breaker's front crashes. Its farm spot
becomes its next packet near its own base, so its Hold walks back and loses reach. In the authoritative step the
hold then stands.

**Prototype.** A `LaneEnv` subclass in the experiment module rewrites only the probe's result: a champion whose front
crashes in the probe keeps its pre-turn front cell as its farm spot. The engine is not changed.

- **Consistent with the intended model?** The rule is implemented exactly as specified. Its effect is a two-pass
  artefact the design did not intend: the design's §6.4 break windows assumed full reach. With keep-reach, the
  parked-front table equals the design's lane-only table in **all 64 cells**, including every Shove row (e.g. 4 HP:
  [4, 3, 4, 3] instead of [9, 8, 9, 8]).
- **Does it materially affect the calibrated results?** Moderately: about as much as the gank change.

  | matchup | keep-reach − current |
  |---|---|
  | presser vs hold_farmer | **+1.49 ± 0.11** |
  | presser vs freezer | +1.09 ± 0.70 |
  | freeze_breaker vs freezer | +0.64 ± 0.66 |
  | shove_recall vs freezer | −0.76 ± 1.57 |

  W = 400, 8 seeds. The rule fires on *every* crash, not only against Freeze. No sign changes.
- **Is a correction justified?** Yes, as a separately reviewed rule change. Smallest implementation:
  - **The new field.** An M4 economy field `farm_spot_rule ∈ {"m2", "pre_crash_front"}` (default "m2"), added to
    `M4_ECONOMY_FIELDS` so that legacy presets reject it.
  - **The resolver change.** In `resolve_turn_m4` step 1:
    `if cfg.farm_spot_rule == "pre_crash_front" and probe.crash_hp[s] > 0 and lane.side(s)[1] > 0:
    spot[s] = clip(lane.side(s)[0])` — about 4 lines.
  - **Tests:**
    - the S3a Shove rows equal `DESIGN_HOLD_TABLE`;
    - the WP4 M2-equivalence and legacy pins (default rule);
    - mirror equivariance;
    - one soak;
    - the G3 tick check.
  - **Acceptance criteria.** The strict expected failure "presser breaks within 6 turns" would then likely pass (not
    verified), so it must be reconciled at the same time.

**Not done:** the rule is unchanged, and the G3 findings and both strict expected failures are preserved.

## 10. Targeted S4 checks of the recommendation (g0.40_i1.35)

- **Dominance** (W = 400, 12 seeds). aggressive_trader against its S4 counters scores:

  | opponent | baseline | candidate |
  |---|---|---|
  | freezer | −3.39 ± 0.66 | −1.68 ± 0.81 |
  | edge_trader | −3.39 ± 0.66 | −1.68 ± 0.81 |

  The two rows are identical because Freeze is never legal against this trader. Paired, the candidate gives
  aggressive trading +1.71 [+1.05, +2.37], but it still loses. **No dominant strategy.**
- **Recall sign pattern** (S4-B, reduced: 4 states × HP 2/4/6 × afford/poor, 8 rollouts, W = 200):
  - 0 significantly positive cells and 5 significantly negative ones (all in the recall-expensive state);
  - afford − poor ≈ +0.01.

  At this size the positive low-HP cells of the full baseline run (3 of 36 at 60 rollouts, W = 1,000) cannot be
  detected. **Inconclusive.** The full S4 battery on the candidate was not run (budget).

## 11. Findings by confidence

**Confirmed:**
- Gank severity explains a small part of the shove penalty: +1.1 to +1.5 of a −17 deficit per 400 turns, and
  +1.26 [+1.12, +1.39] at W = 1,000. Farm displacement explains most of it.
- Gank frequency is unaffected by `gank_damage_position`; conditional severity and gank deaths fall 2–3×.
- An item tier has exactly zero value in trade-free play, and positive value growing with the multiplier when the
  holder trades into a passive opponent.
- Item leads last 7–11 turns in trading play. Purchase timing is independent of the multiplier.
- `item_mult` 1.35 increases the aggressive trader's edge over a passive farmer by +2.30 [+1.73, +2.87] at W = 1,000.
- The cycle agent beats the permanent presser by +15 and loses to passive and trading baselines at baseline rules.
- The keep-reach prototype reproduces the design's parked table exactly.
- Both targeted code fixes pass their tests.

**Plausible (needs more samples):**
- the item's value to an edge trader turns positive over longer windows at 1.50 (+1.76 ± 1.12, W = 200);
- the cycle agent gains about +1.6 under g0.40_i1.35 at W = 400; this is not confirmed at W = 1,000 (−0.53).

**Inconclusive:**
- whether an earlier purchase yields a later advantage;
- the snowballing differences between configurations;
- the S4-B Recall positives on the candidate;
- keep-reach for the cycle agent and freeze_breaker.

**Not performed (budget):**
- the full S4 battery and the S3 gate on any candidate;
- 0.30 / 1.50;
- W = 1,000 confirmation of the Phase D counterfactuals;
- keep-reach under the candidates;
- the full slow suite.

## 12. Recommendation

**For review (not adopted as defaults):** `gank_damage_position = 0.40`, `item_mult = 1.35`, with every other field
unchanged.

**Why these values:**
- **Gank 0.40.**
  - It halves forward-position deaths (presser 3.1 → 1.5 per 100 turns at W = 1,000).
  - It keeps gank risk: frequency is unchanged, and the expected loss is 2.4 of 6 sixths of max HP per gank under the
    enemy tower.
  - It is neutral for passive play, and confirmed beneficial to forward play.
  - 0.30 helps slightly more, but a gank under the enemy tower would then hurt exactly as much as a baseline gank at
    the lane cell (p_e = 0.5, 2 of 6 sixths). Forward position would carry extra danger only through frequency.
    0.40 is the smaller change with most of the effect.
- **Item 1.35.**
  - It removes the first-tier breakpoint at levels 6–11.
  - It doubles the controlled item value in trading.
  - It does not create dominance: the counters still win.
  - 1.50 makes two tiers a 2.25× damage multiplier (3-hit kills), triples the aggressive trader's edge in
    combination with lower ganks, and shows the largest snowball-adjacent shift.

**Conservative alternative:** `0.40 / 1.20`, if reviewers do not want to strengthen trading against passive play.

**What this recommendation does not do.** It does **not** make shove or siege play profitable. It does not make
Recall-to-buy a real decision. The passive-equilibrium risk (audit §2.1) remains, and is mostly structural. Levers
ranked by measured effect, for the D4 / M4.3 review:
1. **Crash and parked-lane farm economics.** The defender farms the crashed wave at full efficiency under its own
   tower, and the pusher farms at 0.5 next to the enemy tower. This is −16 per 400 turns for a shover; no candidate
   moves it by more than 10%.
2. **The farm-spot rule** (§9): +1.5 for pushing, plus faithful Freeze-break windows.
3. **Item persistence (D4).** A multiplier cannot lengthen 7–11-turn leads. Options to evaluate: prices relative to
   income, tiers that survive deaths longer in effect, or an item effect outside trades (e.g. farm or HP). Each is a
   design change and needs the S4 item section rerun.

Do not begin M4.3 until reviewers decide on these. If the recommended values are adopted, rerun S4 (items, Recall)
and the S3 gate on them first (`python -m lanegame.experiments.m4.economy`, `... .freeze_gate`), plus the full
`--runslow` suite.

## 13. Reproduction

Results files are written to `results/m4_calibration/stage_*.json`. Stages charge `budget.json`, and a stage is
refused once the budget is spent.

```
python -m lanegame.experiments.m4.calibration --stage A
python -m lanegame.experiments.m4.calibration --stage B1
python -m lanegame.experiments.m4.calibration --stage B2
python -m lanegame.experiments.m4.calibration --stage R
python -m lanegame.experiments.m4.calibration --stage CD  --configs g0.40_i1.35,g0.30_i1.35,g0.40_i1.50
python -m lanegame.experiments.m4.calibration --stage DCF --configs g0.40_i1.35
python -m lanegame.experiments.m4.calibration --stage F   --configs g0.40_i1.20,g0.40_i1.35
python -m lanegame.experiments.m4.calibration --stage FB
python -m lanegame.experiments.m4.calibration --stage S4  --configs g0.40_i1.35
python -m lanegame.experiments.m4.calibration --stage summary
python -m pytest tests/test_m4_calibration.py tests/test_m4_time.py
```

**Determinism.** Stages are deterministic given the commit; seeds are fixed in the driver. To rerun from scratch,
delete `budget.json` first: the ledger keeps the totals of the runs above, so the stages would otherwise be refused.
