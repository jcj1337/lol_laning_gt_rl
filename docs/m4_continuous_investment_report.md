# M4.2c — Continuous gold investment prototype

Branch `experiment/m4-continuous-investment` (from `calibration/m4-2b`, 8ee9669). A quick prototype and diagnostics
iteration, not a milestone. **The validated two-tier `GameConfig.m4()` is unchanged and stays the default**; the
experiment is the opt-in preset `GameConfig.m4_continuous(invest_alpha=…)`.

Results: `results/m4_continuous_investment/` (`power`, `combat`, `recall`, `response`, `benchmark`, `shapes` JSON;
`summary.md`; `budget.json`). Driver: `python -m lanegame.experiments.m4.investment --stage …`.

## Summary

- **Mechanically sound and computationally cheap.** Every return to base (after a Recall or a death) invests all
  unspent gold, even 1 ledger unit. Under the continuous preset the gold ledger stays exact (unspent + invested =
  cumulative earned, with nothing clamped), rewards stay zero-sum and are never touched by spending, mirror symmetry
  holds, and throughput matches the baseline (≈4.1k vs 4.0k turns/s).
- **Strategically, not yet more useful than two tiers.**
  1. Recall timing for gold has no detectable value with the existing scripted agents. They go back to base every
     ~6–12 turns for HP reasons anyway, so gold never accumulates (2.7–4.5 resources per investment).
  2. Investment only matters through trades (and Pressure). Without trades its effect is exactly zero.
  3. Combat outcomes stay **breakpoint-quantized**: +15% damage often changes nothing, and a few hundredths of a
     resource can flip a duel at levels whose base damage sits just past a breakpoint.
  4. Opponents do react to an investment lead (`edge_trader` stops trading and spends more time away). But the
     leader's gold return is not significantly positive in that mirror, because its own extra trades cost farm.
- **Main calibration risk: unbounded power growth.** With p = 0.9 and no survivability counterpart, attack power
  reaches ×4.3 at 120 invested resources (≈400 turns of scripted play) and ×6–10 at 1,000 turns. A landed 2-HP hit
  one-shots a full-HP same-level champion at S ≈ 68, and a 1-HP hit at S ≈ 188. Both are within the W = 1,000
  evaluation window. The two-tier system caps at ×1.44.
- **Recommendation:** keep it as an experimental option. Before any training use, recalibrate the curve shape
  (p ≈ 0.5–0.7, or a log curve) and review the combat quantization and agent issues below.

## 1. Implemented mechanics and files changed

| file | change |
|---|---|
| `lanegame/config.py` | new fields `item_model` (`"tiers"` default / `"continuous"`), `invest_alpha` (0.045), `invest_p` (0.90), `invest_obs_gold_ref` and `invest_obs_ref` (1024 resources each); preset `GameConfig.m4_continuous()`; property `continuous_items`. Validation: `"continuous"` requires `item_tiers = 0`; the invest fields must stay at their defaults under `"tiers"`; all five are M4 economy fields (legacy / m2c reject them). |
| `lanegame/progression.py` | `Investment` (fixed-point multiplier, attack power, hit and Pressure damage) and `investment(cfg)`; `INVEST_ONE = 10^12`, `INVEST_MAX_RESOURCES = 10^6`. |
| `lanegame/champions_m4.py` | `M4Context.hit` / `.pressure` dispatch on the item model; under `"continuous"` the `items` slot holds invested ledger units (`M4Side.invested` alias), no G_rem clamp, and the return turn invests all unspent gold; `M4Turn.invested`; invariants. |
| `lanegame/env/observation_m4.py` | continuous encoding of the five item entries (same length, 88 entries); decoder. |
| `lanegame/env/agents_m4.py` | duel edge uses `M4Context.hit`; `cash_ready()`: "next tier affordable", or under `"continuous"` "unspent gold ≥ `invest_gold`" (default 4.0, the baseline's first price). |
| `lanegame/env/core.py`, `env/trace.py`, `diagnostics/branch.py` | `info["invested"]`; trace field name `invested`; rollout counter `invested`. |
| `lanegame/env/discounted.py` | per-turn bound under `"continuous"` (§7). |
| `lanegame/experiments/m4/investment.py` | the experiment driver. |
| `tests/test_m4_continuous_investment.py` | 15 focused tests; `tests/test_m4_config.py` table extended with the new fields. |

**Representation choice.** The cumulative investment is stored in the existing `items` slot of `M4Side`, with an
`invested` alias. I did not add an eighth field: the M4 state's seven-field list form is stored in
`data/m4_scenarios.json`, in snapshots and traces, and is compared to them (`scenarios_m4.replay`,
`diagnostics.champ_from_list`). Adding a field would have changed every one of those. The cost of this choice is that
the slot's meaning depends on `item_model`. Any tier-table lookup on a continuous state fails loudly with an
`IndexError`, because the slot holds millions of ledger units; it cannot be silently misread.

**Timing.** Investment happens exactly where the baseline shops: at the end of the turn on which the absence timer
reaches 0 (the last away turn after a Recall, or the last death turn), after that turn's gold. So gold earned while
away (plate gold) is invested too. Recall travel, interruption and absence are unchanged. Death does not lose
investment.

## 2. Power-scaling formula

    AP(L, S) = AP_base(L) · [1 + α ((1 + S)^p − 1)],    S = invested ledger units / gold_scale (resources)

Implemented in integers: M(u) = 10^12 + round(10^12 α ((1 + u/10^6)^p − 1)), AP* = AP(L, 0) · M(u) (basis points
× 10^12), and

    dmg*(D, L, u) = ⌊(D · 100 · AP* + 5000 · 10^12) / 10^16⌋    (centi-HP, the baseline's rounding)
    P*(L, u)      = ⌊(400 · AP* + 5000 · 10^12) / 10^16⌋        (Pressure tower damage, centi-units)

M(0) = 10^12 exactly, so a champion with no investment has the baseline's tier-0 numbers bit for bit (tested). The
curve has dAP/dS = α p (1 + S)^(p−1) per resource. It is monotone and mildly concave, with no saturation point.
Nothing else is scaled: no survivability, no gank or tower changes.

## 3. Investment / power table (AP(L, S) / AP(L, 0), p = 0.90; level-independent)

| S (resources) | α = 0.03 | α = 0.045 | α = 0.06 | two-tier baseline |
|---:|---:|---:|---:|---|
| 0 | 1.000 | 1.000 | 1.000 | 1.000 (N = 0) |
| 1 | 1.026 | 1.039 | 1.052 | 1.000 |
| 2 | 1.051 | 1.076 | 1.101 | 1.000 |
| 4 | 1.098 | 1.147 | 1.195 | 1.200 (N = 1, 4 spent) |
| 7 | 1.165 | 1.247 | 1.330 | 1.200 |
| 10 | 1.230 | 1.344 | 1.459 | 1.440 (N = 2, 10 spent; cap) |
| 14 | 1.313 | 1.470 | 1.626 | 1.440 |
| 20 | 1.435 | 1.652 | 1.869 | 1.440 |
| 30 | 1.630 | 1.945 | 2.259 | 1.440 |
| 50 | 2.003 | 2.504 | 3.005 | 1.440 |
| 80 | 2.536 | 3.304 | 4.072 | 1.440 |
| 120 | 3.217 | 4.326 | 5.434 | 1.440 |
| 200 | 4.518 | 6.277 | 8.036 | 1.440 |
| 400 | 7.576 | 10.864 | 14.152 | 1.440 |

Scale of S in play: scripted champions invest 72–125 resources in 400 turns and 143–314 in 1,000 turns (benchmark
games). At equal spend α = 0.045 is slightly weaker than the tiers up to S = 10, and α = 0.06 roughly matches them.
Beyond 10 resources the continuous curve keeps growing while the tiers stop.

## 4. Combat breakpoints

Analytic, no simulation (`combat.json`). Both champions are at the same level with full HP. "Duel" means both trade
every turn in a neutral lane with the M2 hit probability 0.55 and double-hit probability 0.30, and reports P(A kills
first) / P(B kills first) / P(same turn). A is the higher-investment side.

**α = 0.045, level 6** (levels 1, 11 and 18 are in `combat.json`):

| A vs B (resources) | 1-HP hit A / B (centi-HP) | hits to kill, 1-HP hits | exchange ratio | duel A / B / same |
|---|---|---|---:|---|
| 1 vs 0 | 133 / 128 | 6 / 6 | 1.039 | 0.454 / 0.444 / 0.102 |
| 4 vs 0 | 146 / 128 | 6 / 6 | 1.144 | 0.454 / 0.444 / 0.102 |
| 7 vs 0 | 159 / 128 | 5 / 6 | 1.244 | 0.596 / 0.304 / 0.100 |
| 7 vs 4 | 159 / 146 | 5 / 6 | 1.087 | 0.591 / 0.310 / 0.100 |
| 14 vs 7 | 188 / 159 | 5 / 5 | 1.181 | 0.444 / 0.444 / 0.113 |
| tiers 1 vs 0 | 153 / 128 | 6 / 6 | 1.197 | 0.454 / 0.444 / 0.102 |
| tiers 2 vs 0 | 184 / 128 | 5 / 6 | 1.440 | 0.596 / 0.304 / 0.100 |

**Duel odds are a step function of hits-to-kill**, not of damage. At level 6, A's first-kill probability against
the lower-investment side:

| pair | α = 0.03 | α = 0.045 | α = 0.06 |
|---|---|---|---|
| 1 vs 0 | 0.454 (6/6 hits) | 0.454 (6/6) | 0.454 (6/6) |
| 4 vs 0 | 0.454 (6/6) | 0.454 (6/6) | 0.454 (6/6) |
| 7 vs 0 | 0.454 (6/6) | 0.596 (5/6) | 0.596 (5/6) |
| 7 vs 4 | 0.449 (6/6) | 0.591 (5/6) | 0.591 (5/6) |
| 14 vs 7 | 0.591 (5/6) | 0.444 (5/5) | 0.601 (4/5) |

So the duel value of an investment lead is non-monotone in α: 14 vs 7 is worth +15 points at α = 0.03, nothing at
0.045, and +31 at 0.06.

**Breakpoints** (smallest investment at which a 1-HP hit kills a same-level full-HP champion in one hit fewer).
Because HP and attack power both grow by 1.05^L, the breakpoints are nearly level-independent:

| α | 6 → 5 hits | 5 → 4 | 4 → 3 | 2-HP hit one-shot | 1-HP hit one-shot |
|---|---:|---:|---:|---:|---:|
| 0.03 | 8.4–8.7 | 23.1–23.3 | 49.6–50.0 | ≈107 | ≈295 |
| 0.045 | 5.4–5.7 | 14.8–15.0 | 31.8–32.0 | ≈68 | ≈188 |
| 0.06 | 4.0–4.2 | 10.8–11.0 | 23.2–23.4 | ≈50 | ≈137 |

Two artefacts of the rounded HP and AP tables:

- **Knife edges.** At level 18 the base 1-HP hit (229 centi-HP) needs 7 hits for 1,375 HP (6.004 hits). An investment
  of 0.03 resources (α = 0.045) makes it 6 hits and moves the duel from 0.453 / 0.453 to 0.583 / 0.325. The same
  happens for 2-HP hits at level 6 (0.023 resources).
- **Quantization.** Hit damage is in whole centi-HP. The smallest investment that changes a 2-HP hit is 0.005–0.06
  resources at α = 0.045 (level-dependent); 1 gold always changes a hit by ≥ 3 centi-HP. The underlying AP* rises
  for every ledger unit (§7), and the rounded damage follows within a few hundredths of a resource. Small
  investments do not disappear in rounding. The coarse quantity is the HP pool measured in hits (5–7 hits from full
  HP), not the damage.

In actual play HP persists between trades and is also reduced by ganks and tower shots, which smooths these steps
somewhat. The rollouts below show that.

## 5. Recall-versus-farm experiments

Setup: the two witnessed recall situations, `6a_recall_cheap` (own wave pushing) and `6b_recall_expensive` (enemy
wave at own −1). I varied A's HP (3/6, 6/6), unspent gold (1, 4, 7, 10) and B's investment (0 or 7; continuous only).
The first move was Recall or Hold, then both seats play `edge_trader` (Hold, trade on a positive duel edge, low-HP
recall at 1/3). Returns are discounted (γ = 0.99) over 120 turns, 10 paired seeds per cell (same seeds for every
variant of a state), α = 0.045. Values are the paired Recall − Hold difference with 95% half-widths.

| model | state | HP | B inv. | 1 gold | 4 gold | 7 gold | 10 gold | gold interaction (10 vs 1) | first investment under Hold (median turn) |
|---|---|---|---|---|---|---|---|---|---:|
| cont | cheap | 3/6 | 0 | −0.25 ± 1.05 | −0.05 ± 1.24 | −2.69 ± 1.52 | −1.28 ± 1.57 | −1.03 ± 1.55 | 5.5 |
| cont | cheap | 3/6 | 7 | −0.65 ± 2.30 | +1.32 ± 2.02 | +1.15 ± 1.20 | +0.85 ± 1.25 | +1.49 ± 2.94 | 5.5 |
| cont | cheap | 6/6 | 0 | −0.52 ± 1.45 | +1.10 ± 1.12 | +1.23 ± 1.63 | −1.34 ± 2.12 | −0.83 ± 2.60 | 11.0 |
| cont | cheap | 6/6 | 7 | +0.08 ± 1.17 | +0.27 ± 1.69 | +0.38 ± 2.07 | +0.06 ± 2.29 | −0.02 ± 2.35 | 6.0 |
| cont | expensive | 3/6 | 0 | +0.08 ± 0.59 | −0.25 ± 0.14 | −0.10 ± 0.15 | −0.15 ± 0.06 | −0.22 ± 0.60 | 6.0 |
| cont | expensive | 3/6 | 7 | −0.15 ± 0.05 | −0.06 ± 0.23 | −0.43 ± 0.50 | +0.10 ± 0.54 | +0.25 ± 0.54 | 6.0 |
| cont | expensive | 6/6 | 0 | −0.06 ± 0.23 | −0.15 ± 0.04 | −0.09 ± 0.56 | +0.01 ± 0.36 | +0.07 ± 0.45 | 12.5 |
| cont | expensive | 6/6 | 7 | +0.06 ± 0.39 | +0.09 ± 0.45 | −0.18 ± 0.00 | −0.08 ± 0.12 | −0.15 ± 0.43 | 8.0 |
| tiers | cheap | 3/6 | 0 | −0.84 ± 1.57 | −0.89 ± 1.60 | −2.53 ± 1.71 | −2.24 ± 1.60 | −1.40 ± 2.08 | 5.5 |
| tiers | cheap | 6/6 | 0 | +0.68 ± 1.01 | +0.66 ± 1.51 | −0.46 ± 1.17 | +0.23 ± 1.46 | −0.45 ± 1.45 | 11.0 |
| tiers | expensive | 3/6 | 0 | −0.40 ± 0.45 | −0.18 ± 0.00 | −0.18 ± 0.00 | −0.18 ± 0.00 | +0.23 ± 0.45 | 6.0 |
| tiers | expensive | 6/6 | 0 | −0.18 ± 0.00 | −0.18 ± 0.00 | −0.18 ± 0.00 | −0.18 ± 0.00 | −0.00 ± 0.00 | 12.5 |

The "gold interaction" is (Recall − Hold at 10 gold) − (Recall − Hold at 1 gold). It isolates the value of
converting the gold now from the HP and time effects of the Recall itself.

Findings:

- **Gold has no detectable effect on the Recall decision in either model.** Every gold interaction includes zero.
  The cause is in the continuation: under Hold, the `edge_trader` returns to base anyway within a median 5.5–12.5
  turns (HP loss from trades and ganks, or death), and invests then. "Recall now" therefore only moves the investment
  a few turns earlier. That is worth far less than the rollout noise of the recall-cheap state (±1–3).
- **The recall-expensive state is nearly deterministic.** Both plans re-merge and the difference is the farm of the
  turn spent recalling (−0.18). This is identical under tiers.
- **Costs of Recall-now** (continuous cells, mean of the paired differences): +0.1 away turns, −1.8 XP, ±1 farm, and
  no systematic change in deaths. Per-cell values are in `recall.json` (`away_A`, `farm_A`, `xp_A`, `deaths_*`,
  `trade_dmg_*`).
- **Against the tiers.** Under tiers, gold below 4 cannot be spent, and gold between prices is spent only in
  4 / 6 lumps. Under continuous investment every return converts everything. That removes the "dead gold" problem, but
  with these agents it makes no measurable difference, because their natural return cadence already converts gold
  quickly (2.7–4.5 resources per investment for edge / aggressive traders, 6.75 for `gold_recaller`).

## 6. Strategic consequences of an investment lead

Setup: two reachable snapshots of mutual `edge_trader` play (levels 7 and 13). B is set to A's XP, both are at full
HP with no unspent gold, and A gets a lead of 0, 4, 7 or 14 invested resources (continuous, each α) or 0 / 1 / 2 tiers
(baseline). Rollouts are 110 turns, discounted, 8 seeds per snapshot (16 per cell), paired against the zero-lead
variant on the same seeds. Values are lead − even, per rollout, from A's side.

| model, lead | matchup (A–B) | return to A | trades A / B | trade dmg dealt, in max HP, A / B | deaths A / B | farm A / B | B away turns | B retreat/safe turns |
|---|---|---|---|---|---|---|---|---|
| α 0.045, 4 | edge–edge | −0.70 ± 1.84 | +11.4 / −8.6 | +2.08 / −1.12 | +0.44 / +1.06 | −2.32 / −1.39 | +4.0 | +0.9 |
| α 0.045, 7 | edge–edge | −0.58 ± 1.64 | +13.0 / −11.4 | +2.97 / −1.42 | +0.25 / +1.56 | −1.54 / −0.48 | +5.4 | +0.5 |
| α 0.045, 14 | edge–edge | +0.65 ± 1.75 | +18.0 / −17.1 | +5.25 / −2.54 | −0.12 / +2.56 | −0.69 / −1.13 | +8.1 | +0.7 |
| α 0.06, 14 | edge–edge | **+2.40 ± 1.31** | +16.6 / −16.7 | +5.81 / −2.74 | −0.88 / +3.56 | +0.86 / −1.47 | +10.2 | +0.6 |
| tiers, 1 (4 gold) | edge–edge | **−1.26 ± 0.90** | +14.1 / −4.4 | +2.17 / −0.12 | +1.12 / +0.81 | −3.32 / −1.70 | +4.0 | +1.1 |
| tiers, 2 (10 gold) | edge–edge | −1.65 ± 1.33 | +19.1 / −9.6 | +3.35 / −0.68 | +1.31 / +0.75 | −3.38 / −0.92 | +5.1 | +0.9 |
| α 0.03, 4 / 7 / 14 | aggressive–hold | +1.22 / +1.53 / +3.20 (± 0.7–1.0) | | | | | | |
| α 0.045, 4 / 7 / 14 | aggressive–hold | **+1.64 / +2.79 / +3.34** (± 0.9–1.1) | −1.3 … −4.1 / 0 | +1.1 … +2.9 / 0 | −0.4 … −1.1 / +1.2 … +2.7 | | +2.6 … +5.8 | ≈ 0 |
| α 0.06, 4 / 7 / 14 | aggressive–hold | +1.74 / +2.33 / +4.31 (± 1.0–1.4) | | | | | | |
| tiers, 1 / 2 | aggressive–hold | +2.03 / +2.21 (± 1.1) | | | | | | |
| any α, any lead | gold_recaller mirror | **0.00 ± 0.00** | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 | 0 |
| tiers, 1 / 2 | gold_recaller mirror | +0.46 / +0.81 (± 0.2) | 0 / 0 | | | | | |

Rows for every α and lead are in `response.json` and `summary.md`.

- **Leads have consequences, but only through trades.**
  - A lead makes the trading opponent concede trades: the `edge_trader` B trades 7–17 times less per 110 turns.
  - B loses fights: +0.9 to +3.6 deaths, and +3.7 to +10 turns away.
  - B gives up little farm: +0.2 to −1.5.
  - B almost never moves to Safe (≤ 1 turn). Scripted agents cannot "concede the lane" while present; they only
    stop trading.
- **Return to the leader is positive only against a passive opponent.**
  - Against `hold_farmer`, which never trades back, the return grows with the lead and with α: +1.6 / +2.8 / +3.3
    at α = 0.045. The continuous 4-gold lead (+1.6) is close to the 1-tier (4-gold) baseline (+2.0).
  - In the `edge_trader` mirror the leader's return is not significantly positive, except α = 0.06 with a 14
    resource lead. The leader's extra trades cost its own farm (trading collects at `trade_farm_factor`), and kills
    pay only 0.5. The same over-trading appears under tiers, where it is significantly negative (−1.26 ± 0.90 for one
    tier).
- **Without trades, investment is worthless.** The `gold_recaller` mirror never trades and does not Pressure in this
  window: every continuous lead gives exactly the same trajectory. The small tier effect (+0.46 / +0.81) is not
  combat. It comes from the recall rule: with a tier owned, the next price is 6 instead of 4, so that agent recalls
  later. Under continuous investment the trigger (4 unspent gold) does not depend on what is already invested.
- **Limitation of the scripted strategies.** Each reacts to investment only through the duel-edge trade trigger,
  which is myopic: it ignores farm lost while trading and the bounty's small size. None weighs Recall timing against
  power, and none concedes farm deliberately. I did not add economic rewards or train agents (out of scope). Whether
  a learning agent would exploit an investment lead is open.

## 7. Observation, precision and boundedness

- **Ledger.** Plain Python integers in ledger units (10^-6 resources). There is no clamp under `"continuous"`
  (`gold_unspendable` ≡ 0), and unspent + invested = cumulative earned on every turn (tested over 600 turns, two
  matchups). Investing is not a reward event: the reward is `earned_A − earned_B`, and a return turn pays the same
  reward with or without gold to invest (tested).
- **Power precision.** The multiplier is fixed point at 10^12. It is strictly increasing per ledger unit, with
  increments ≥ 10,172 units at α = 0.045, ≥ 6.8 k at α = 0.03, and 40,500 at S = 0. The float64 pow error is a few
  units, far below these increments. This holds over the supported range S ≤ `INVEST_MAX_RESOURCES` = 10^6 resources,
  tested on a random grid plus the end points. **Beyond 10^6 resources `Investment` raises `ValueError`**: this is a
  numerical-validity guard, not a game cap (it is ≈3 × 10^6 turns of farming away), and nothing is silently capped
  below it. AP* reaches ~10^20 at the guard, which is exact in Python integers.
- **Observation contract.** I kept the same length (88 entries). The five item entries become invested whole
  resources / R_S, unspent whole resources / R_G, the two sub-resource remainders / 10^6, and a bounded power feature
  AP* / (AP* + AP(18, 0)), with R_S = R_G = 1024 resources.
  - **Exact decoding holds only while S < 1024 and G < 1024 resources.** It is tested on reachable states and the
    boundary values, and `float32` represents every remainder and every whole part n/1024 exactly.
  - At or beyond a reference the whole-part entry saturates at 1.0. `observe_m4` still returns a valid [0, 1] vector,
    because the game has no cap, but the observation is no longer injective, and `decode_m4` raises.
  - In scripted play 1024 invested resources is about 3,000–7,000 turns of one game: outside W = 1,000 but reachable
    in very long continuing-time runs. A never-recalling, never-dying agent could reach 1024 unspent gold sooner
    (the benchmark maximum was 25.6).
  - **Consequence beyond the references:** the `diagnostics.ProbCache` key (the observation vector) could merge
    distinct states, and a policy cannot tell them apart. So **global losslessness is not achieved**; it holds on a
    bounded region.
  - A second weakness for learning: one resource moves the whole-part feature by only 1/1024. The power feature is
    the smooth signal. For M4.3 a log-scaled feature would be better, at the cost of one extra entry.
- **Payoff and truncation bounds.** The guarantee constant Ω = 68.34 (γ^1000 Ω = 0.0030) is unchanged: it counts
  farm, bounty frequency and plate gold, none of which depends on attack power. The per-turn bound
  (`per_turn_bound`) assumed Pressure damage bounded by AP(18, 2 tiers). With unbounded AP one Pressure turn may cross
  every plate, so it now uses all three plates under `"continuous"` (8.575 → 9.575 resources). Both remain valid.

## 8. Simulation usage, wall time, tests

| stage | turns | wall (s) | worker CPU (s) |
|---|---:|---:|---:|
| power, combat, shapes (analytic) | 0 | <1 | — |
| recall (96 variants × 10 × 120) | 115,200 | 5.26 | 34.8 |
| response snapshots + response (90 variants × 8 × 110) | 79,320 | 3.37 | 20.0 |
| benchmark (3 × 1,000 turns per model) | 6,000 | 1.48 | 1.9 |
| development smoke runs and new tests (approximate) | ≈3,100 | ≈1 | — |
| **total** | **≈203,600** | **≈11.1** | **≈57.7** |

This is within the 500,000-turn / 180 s hard cap and slightly above the 200,000 target (because of the benchmark and
smoke runs). Ledger: `results/m4_continuous_investment/budget.json`. The pre-existing focused test files I ran as
baseline regression checks are not counted.

Focused tests:

- `tests/test_m4_continuous_investment.py`: 15 passed.
- Regression files: `test_m4_config`, `test_m4_observation`, `test_m4_resolve`, `test_m4_economy`,
  `test_legacy_pins`, `test_m4_env`, `test_m4_diagnostics`, `test_m4_calibration`, `test_m4_combat`,
  `test_m4_progression` (130 passed with the new file, 3 skipped), and `test_m4_integration`, `test_m4_scenarios`,
  `test_m4_freeze_gate`, `test_env` (116 passed, 2 known strict xfails).

The full `--runslow` suite was not run (assignment §5).

Section A checks map to the new tests:

| check | test |
|---|---|
| 1 gold (1 ledger unit and 1 resource) and 7 gold invest exactly | `test_a_return_invests_exactly_the_unspent_gold` |
| Recall with 1 gold through the environment | `test_recall_with_one_gold_invests_one_through_the_environment` |
| Investments accumulate, and equal totals give equal power | `test_investments_accumulate_and_equal_totals_give_equal_power` |
| Investment persists through death | `test_investment_persists_through_death_and_is_invested_on_the_death_return` |
| Monotone power, neutral at zero | `test_power_is_strictly_increasing_per_ledger_unit_and_neutral_at_zero` |
| Zero-sum and spending-neutral rewards, exact ledger | `test_rewards_are_zero_sum_ledger_exact_and_mirror_symmetric` |
| Mirror symmetry of one turn | `test_mirror_symmetry_of_one_turn` |

## 9. Comparison with the two-tier system

| aspect | two tiers (`m4()`) | continuous (`m4_continuous()`) |
|---|---|---|
| purchase rule | 4 then 6 resources at base; gold beyond the remaining prices is clamped away | everything at every return; nothing is lost |
| power range | ×1.00 / 1.20 / 1.44, capped | ×1 → unbounded (×4.3 at 120, ×8.6 at 300 resources) |
| "dead gold" | yes (below 4; between prices; after both tiers it is unspendable) | none |
| Recall incentive from gold (scripted) | not detectable | not detectable |
| duel effect of a 4-gold lead (L6) | 0 (misses the breakpoint) | 0 at all three α (at L1, α = 0.06 crosses it: 0.475 / 0.420) |
| return of a lead vs a passive opponent | +2.0 (1 tier), +2.2 (2 tiers) | +1.6 (4), +2.8 (7), +3.3 (14) at α = 0.045 |
| late-game lethality | bounded (6 → 5 hits) | one 2-HP hit kills at S ≈ 68, one 1-HP hit at S ≈ 188 (α = 0.045) |
| state / observation | 2 small integers; exact decoding everywhere | unbounded integers; exact only below the references |
| throughput | 4.0k turns/s | 4.1k turns/s |

## 10. Recommendation for further calibration

1. **Do not adopt it as the default.** It is mechanically sound and cheap, but in these diagnostics it is not yet
   strategically more useful than two tiers. It removes dead gold, but it does not create a measurable Recall-timing
   decision with the current agents, and it adds an unbounded late-game power curve.
2. **Fix the curve shape before tuning α.** p = 0.9 is nearly linear. Matched at S = 10 (×1.344), other shapes give
   (`shapes.json`, analytic):

   | shape | α | ×S=4 | ×S=50 | ×S=120 | ×S=300 | 2-HP one-shot at S | 1-HP one-shot at S |
   |---|---:|---:|---:|---:|---:|---:|---:|
   | p = 0.9 | 0.045 | 1.15 | 2.50 | 4.32 | 8.60 | 68 | 189 |
   | p = 0.7 | 0.079 | 1.17 | 2.16 | 3.19 | 5.21 | 106 | 382 |
   | p = 0.5 | 0.149 | 1.18 | 1.91 | 2.49 | 3.43 | 208 | 1,201 |
   | α ln(1 + S) | 0.144 | 1.23 | 1.56 | 1.69 | 1.82 | never | never |

   I suggest a next sweep over p ∈ {0.5, 0.6, 0.7} at fixed early strength. Measure late-window lethality and
   snowballing with the existing `response` stage at S ≈ 100–300, about 80k turns.
3. **Review the combat quantization separately.** With 5–7 hits from full HP, duel odds move in steps, which produces
   knife edges (0.03 resources at L18) and non-monotone value in α. Options for that review: finer HP units, or
   adjusting H(L) and AP so that the base hits-to-kill is not just past a breakpoint. Both are outside this assignment.
4. **The Recall-timing question needs agents that can make it.** The scripted agents return every ~6–12 turns for HP
   reasons and judge trades myopically. The cheapest next evidence is a parameter sweep of the `gold_recaller` /
   `shove_recall` threshold (`invest_gold` ∈ {2, 4, 8, 16}) against fixed opponents, before any PPO (M4.3).
5. **Before any training use**, add a log-scaled invested-gold feature. Either size the observation references to the
   training window, or accept and document the non-injective tail.
