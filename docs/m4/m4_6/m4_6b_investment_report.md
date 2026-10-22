# M4.6b — Investment audit, candidate-curve screen and measurement fixes

*2026-10-15. Status: complete, awaiting review. The calibrated baseline `GameConfig.m45_calibrated()` is unchanged; the
candidate curves are opt-in configurations of existing fields; no PPO training ran; nothing is staged or committed.
Protocol: [m4_6b_preregistration.md](m4_6b_preregistration.md), frozen after the audit of the current curve and before
any candidate outcome (SHA-256 in `results/m46b/preregistration.json`, with one dated amendment for an exploratory
follow-up). Generated tables: `results/m46b/report.md`. Previous stage: [m4_6a_report.md](m4_6a_report.md).*

**Terms.**
- A *resource* is the gold of one fresh wave. Natural income is about 0.40 resources per turn, so 20 turns of income is
  about 8 resources.
- *S* is a champion's invested gold. *M(S)* is its investment multiplier.
- The *attack ratio* is A's attack power over B's.
- The *duel ratio* is the damage A deals per exchange in B's bars over the reverse. It counts investment twice,
  because investment also divides damage taken.
- *Elasticity* is d ln M / d ln S: the relative strength a relative investment lead buys.
- *G_h* is the discounted gold lead over h turns, the training objective truncated at h.
- *The policy* means the six 16.384M M4.5c finals.

---

## Summary

**The answer.** Earned gold leads look weak mainly because natural leads are small relative to accumulated investment,
and the curve's saturation compresses them further. The ×1.01 that M4.6a reported is a late-game reading.
- Three strength-matched alternative curves make mid-game leads clearly more consequential. In the powered follow-up
  they raise the lead premium about 1.6–2.2×.
- None passes the pre-registered rule. Each raises late wave capacity beyond its bound, and each raises the premium of
  an early lead.

**Recommendation: retain the current curve for now, and settle the two failed criteria before any pilot.** If the
review accepts their costs, C1 is the candidate to pilot (§9).

**Measurement fixes.** Each was verified by exact reproduction of stored M4.5c records: 24 of 24 non-position readouts
per final, and 840 of 840 battery jobs.
- **Positions.** The stronger side stands just behind the lane cell, not one to three cells forward. The Ward node was
  being averaged in.
- **Behaviour-only interventions.** With the wave left to the policy, walking up to a freeze costs −0.02 to −0.13, not
  −0.22. The trading decision costs about −0.08, not −0.34.
- **Economic channels.** Farm lost while away, bounty and plates are now separate, as an accounting identity.

**Why ×1.01.**
1. Natural leads are 10% of earned gold at turn 50 and 4% at turn 300. They persist in absolute size while both sides
   keep earning.
2. The curve's elasticity, 0.17 falling to 0.09, compresses a relative lead 6–11×.
3. Before turn 50 the leader's lead sits in unspent gold, so the leader is not yet stronger (×0.99).
4. M4.6a's 10–20 lead bin had its median at turn 418, with 161 invested.

**What leads buy now.** Twenty turns of income buy ×1.25 in a duel around turn 35, ×1.06 around turn 120 and ×1.02
around turn 270. The value of buying earlier, rather than of the gold itself, is about zero under every curve.

**The candidates.** C1 and C2 saturate more slowly, and C3 is a linear reference. All are opt-in and strength-matched,
so each is weaker than the baseline at small investment and stronger at large investment.
- **Leverage.**
  - A mid-game lead buys ×1.11–1.16 in a duel instead of ×1.06.
  - In natural self-play, leads keep their size from turn 100 to 300 instead of shrinking by a quarter.
  - The powered follow-up's mid-game premium is ×1.6 for C1, ×2.2 for C2 and ×2.2 for C3, all significant.
- **Unchanged.**
  - One-turn incentives.
  - Freeze contests: contesting stays correct, and stepping back is the worst reply.
  - Recall decisions, and the equal lane, within 5%.
  - The weaker side's best response: stay and farm, with its own policy best of all.
- **Costs.**
  - Late wave capacity rises by 3 / 5 / 7 HP.
  - Deaths fall 7–15%.
  - The trailer at a 40-turn lead loses about twice as much.
  - Late resets cost more.

**Scorecard.**

| criterion | C1 | C2 | C3 |
|---|---|---|---|
| D1 leads consequential (pre-registered screen) | not met, underpowered | not met | not met |
| the same test on the exploratory follow-up | mid only | met | met |
| D2 not decisive | fails on the early premium | fails | fails |
| D3 equal lane | passes | passes | passes |
| D4 waves and towers | fails: late capacity +3 HP | fails: +5 HP | fails: +7 HP |

**Caveats.** All screens use policies trained on the current curve, so they are out of distribution under a
candidate. A nonsignificant difference is not evidence of equivalence.

---

## 1. Measurement fixes and the conclusions they change

### 1.1 What was fixed

| fix | where | before | after |
|---|---|---|---|
| positions | `rl/m44/metrics.py`: `zoning`, the freeze episodes, the investment-lead table | every cell averaged, with the off-lane Ward node W stored as 99 | lane cells only (`lane_mean`); Ward occupancy reported separately (`ward_share`) |
| champion distance and the enemy's cell | `rl/m44/records.py` | \|pos + enemy pos\| and −enemy pos, wrong whenever either champion stood at W | movement-graph distance; W kept as W (identical on lane cells) |
| economic channels | `m45b/engine.py` (the ledger), `rl/m45c/horizon.py` (the battery's ledger keys) | farm lost while away was not separated; the battery did not record farm available, bounty or plates | `unc_away` in the base ledger; the battery records `farm_avail`, `unc_away`, `bounty`, `plate`, `trade_att` |
| behaviour-only interventions | `rl/m45c/horizon.py` (battery candidates), `rl/m46a/policies.py` | the battery's Trade / Hold / Safe / walk-up / step-back / bully candidates also forced a Last Hit wave | behaviour-only twins for each, wave left to the policy (`1t:trade_beh`, `1t:hold_beh`, `1t:safe_beh`, `1t:walk_up_beh`, `1t:step_back_beh`, `plan:bully_behx10`, `plan:threaten_behx10`, `plan:farm_behx10`); the originals kept as labelled scripted references |

The channels are an **accounting decomposition**, not a causal attribution. They obey exact identities, tested in
`tests/test_m46b.py`:
- earned = farm + bounty + plates;
- farm = available − lost, where lost = lost while present + lost while away.

The difference of two runs' channels says where a difference in gold shows up, not what caused it. Each resource
appears in exactly one channel of one side, so nothing is double-counted.

### 1.2 Verification by reproduction (no M4.5c re-run)

- **V1, M4.5c's power-gap stage on its original seeds:**
  - every one of the 24 non-position readouts per final reproduced exactly, for all six finals;
  - the stronger side's cell went from **+0.93 … +2.95 to −0.13 … −0.26**, with the Ward node holding 1.2–3.2% of its
    turns;
  - the equal-power class went from +1.03 … +2.23 to −0.69 … −0.74;
  - credible-threat counts moved by 1–5%, because the distance is now correct at W.
- **V2, battery families C_trade, B_frozen and H_zone_c for finals s0 and s3 on the battery's own seeds:**
  - **840 of 840 jobs reproduced the stored M4.5c raw values exactly** (G and Q of every rollout of every original
    candidate);
  - the new twins and channels were added alongside.

### 1.3 Corrected interpretations

From V2, two finals, the family's primary statistic at 50 turns; indicative, not a replacement campaign.

| earlier conclusion | corrected reading |
|---|---|
| M4.5c: "the stronger side presses forward (+2.0)", "+1.30 → +2.03 between budgets" | the stronger side stands just behind the lane cell (−0.1 to −0.3), the weaker side about a cell further back; the change was Warding |
| M4.5c: "walking up to a freeze (accepting it) costs about −0.22" | with the policy's own wave the walk-up costs **−0.02 to −0.13** (twin minus the Last Hit version: +0.15 to +0.35). Stepping back still costs −0.12 to −0.46. Walking up to contest a freeze is a reasonable choice |
| M4.5c: "Trading into a heavier wave costs −0.34; the policies avoid it" | standing in that wave at all costs about −0.3 (Hold −0.28 to −0.31). The trading decision itself (Trade − Hold, both behaviour only) is about **−0.08**, within noise per final (−0.23, +0.07) |
| M4.5c: "sustained bullying costs against learned play" | about 0.4 of the scripted bully's loss is its forced Last Hit wave (twin − scripted +0.44), as M4.6a found |
| M4.6a: "a 10–20 resource gold lead corresponds to about ×1.01 power" | correct as measured, but a late-game statement: §3 explains it |

---

## 2. The current investment system

**The formulas** (`progression.Investment`, `champions_m4`, reference §7.3):

| quantity | rule |
|---|---|
| earned E, unspent g, invested S | E = S + g. Gold is earned every turn and paid as reward then; it becomes S only on a **return** (the turn a Recall or a death ends): S ← S + g, g ← 0 |
| multiplier | M(S) = 1 + 1.5 (1 − √(16 / (16 + S))), exact integers, 1 ≤ M < 2.5 |
| attack power | AP*(L, S) = 1.05^(L−1) M(S): trade damage, wave-clear capacity C = round(4 AP*), Pressure = round(400 AP*) per turn |
| durability | every centi-HP taken (trades, tower shots, ganks, minions) is divided by M(S) |
| max HP | 600 × 1.05^(L−1): levels only |

**In words.** Gold buys a multiplier that applies twice in a duel: once to the damage you deal and once, as armour,
to the damage you take. So for two champions of the same level the duel ratio is (M_A / M_B)². Each level adds 5% to
both attack and HP. The multiplier is concave and bounded:
- half of its 1.5 headroom is bought by 48 resources (about 120 turns of income);
- the first resource adds 0.047 to M, the hundredth 0.0024.

**What equal investment does.** If both champions have the same M, their exchanges are unchanged whatever M is: 0.119
bars per exchange in every arm (S1). Higher equal investment only makes both champions stronger against **waves**
(capacity), **towers** (Pressure) and the **environment** (ganks, tower shots, minions, divided by M).

---

## 3. Why natural gold leads look like ×1.01

Audit: s0 self-play of the six finals, 24 games × 600 turns each (`results/m46b/audit.json`).

**Natural investment levels and leads:**

| turn | invested S (median) | unspent g | level | \|earned lead\| median (q90) | relative lead |
|---|---|---|---|---|---|
| 50 | 15 | 4.6 | 9 | 2.0 (5.3) | 10% |
| 100 | 35 | 4.5 | 14 | 3.0 (7.3) | 8% |
| 150 | 56 | 4.4 | 18 | 4.2 (9.5) | 7% |
| 300 | 116 | 4.8 | 18 | 5.4 (11.2) | 4% |
| 600 | 234 | 4.4 | 18 | 5.8 (15.9) | 2.4% |

The objective's discount weights put the median of investment at 22 (quartiles 6 and 50). Leads persist: the slope of
the lead 50 turns later on the lead now is 0.76–0.98. But both sides keep earning about 0.40 per turn, so a lead's
**relative** size falls by a factor of 4 between turns 50 and 600.

**From a gold lead to strength.** Geometric means over present turns, oriented to the gold leader; each column applies
one more step of the chain:

| stage (turns) | relative gold lead | if all earned gold were invested: M ratio | actual M ratio | level factor | attack ratio | duel ratio |
|---|---|---|---|---|---|---|
| early (20–50) | ×1.150 | ×1.025 | **×0.990** | ×1.011 | ×1.001 | ×1.002 |
| mid (90–150) | ×1.087 | ×1.014 | ×1.013 | ×1.009 | ×1.022 | ×1.045 |
| late (240–320) | ×1.052 | ×1.006 | ×1.006 | ×1.000 | ×1.006 | ×1.013 |
| very late (450–600) | ×1.034 | ×1.003 | ×1.003 | ×1.000 | ×1.003 | ×1.006 |

The ×1.01 has four separable causes:
1. **Leads are small relative to investment.** A typical leader is 3–15% ahead in earned gold, and the share falls as
   both sides invest. This is not catch-up by the trailing side: leads persist in absolute terms, while the base
   grows.
2. **Saturation compresses them 6–11×.** The elasticity is 0.17 up to about 50 invested, 0.12 at 100 and 0.09 at 200.
   A 9% lead mid-game becomes a 1.4% multiplier lead.
3. **Unspent gold and Recall timing cancel the effect early.** Before turn 50 the gold leader is less invested than
   the trailer in 53% of turns, because its lead sits in unspent gold (+1.9 of a 1.8 average lead). Its actual M
   ratio is ×0.99. From mid-game on, unspent gold explains almost nothing (the leader is more invested in 72–82% of
   turns).
4. **Levels add a little early and nothing late.** A level gap of 0.27–0.43 on average early and mid gives ×1.009–1.011
   of attack; levels converge at 18 by about turn 145.

**Measurement and sampling.** M4.6a binned by |earned lead| and reported the M4.4 "power ratio" (levels counted
twice, investment once, durability omitted; folded as max(r, 1/r)). Its 10–20 resource bin has its median at **turn
418, with 161 invested on average; 96% of its turns are after turn 150**. The ×1.01 is therefore a statement about the
late game. The bin with the *largest* ratio is the smallest-lead bin (×1.026), because that is where the early-game
level gaps sit.

---

## 4. What a purchase buys

**What a lead buys** (current curve; stage-typical states from the audit; leads in turns of measured income):

| stage | typical S, level | lead (turns of income) | M ratio | duel ratio | wave capacity C (trailer → leader) | bars per exchange: leader on trailer / trailer on leader |
|---|---|---|---|---|---|---|
| early | 9, L8 | 20 (8 res) | ×1.12 | ×1.25 | 7 → 8 | 0.135 / 0.106 |
| early | 9, L8 | 40 (16 res) | ×1.20 | ×1.45 | 7 → 9 | 0.146 / 0.098 |
| mid | 42, L16 | 20 | ×1.03 | ×1.06 | 14 → 15 | 0.123 / 0.116 |
| mid | 42, L16 | 40 | ×1.05 | ×1.11 | 14 → 15 | 0.126 / 0.113 |
| late | 108, L18 | 20 | ×1.009 | ×1.017 | 18 → 18 | 0.120 / 0.118 |
| late | 108, L18 | 40 | ×1.017 | ×1.033 | 18 → 18 | 0.121 / 0.117 |

(Equal champions exchange 0.119 bars at every stage.)

**Two different questions.**
- **Marginal value of a purchase.** Once returned to base, a purchase adds dM/dS × gold to the buyer's M: 0.024 per
  resource early, 0.0067 mid, 0.0022 late. M4.6a's E5 measured its value in the Recall context, and S5 here confirms
  it in every arm: the "shopping component" is about zero. But that measures the value of investing 10–20 turns
  **earlier** (the next natural return comes anyway), not the value of the gold. That is small under any curve.
- **Relative strength of two invested champions.** This is the ratio M_A / M_B, which depends on the lead's size
  relative to S and on the elasticity. It is what decides trades.

**The combined combat effect.** Investment improves damage and durability together, so the duel ratio is the square
of the M ratio. A ×1.05 multiplier lead is a ×1.10 duel ratio. Per exchange the effect is still small: at mid-game a
40-turn lead moves the damage per exchange by ±0.007 bars, about 0.01 resources at the measured price of HP (1.2–1.5
per bar). Leads matter through accumulation over many trades, into kills and resets, not per turn.

**Implementation assumptions that alternative curves touch** (all checked; none breaks):

| item | finding |
|---|---|
| observation | the invested entry is n / (n + 64) of the gold, not of M, so it is unchanged. The attack-power entry AP* / (AP* + AP(18, 0)) stays in (0, 1) under any curve, but its pairing with the invested entry changes: a network trained on the baseline sees combinations it never saw |
| exactness | the saturating curve is exact integer arithmetic for any A, K (`invest_gain` ≤ 10, K a whole number of ledger units); the power curve is float with fixed-point rounding (deterministic); supported up to 10⁶ resources |
| bounds | the per-turn reward bound grows with the largest wave capacity (baseline 24 HP, 11.75 resources; C1 40, 13.6; C2 98, 20.1; linear unbounded: 408 from the lane-HP bound, 55.0). The evaluation's truncation constant uses the per-turn bound only for the first turn; the discounted tail constant is unchanged |
| wave actions | Last Hit is capped at 1 HP at every power; Thin and Slow Push spend only what their window target needs; Shove spends C. A higher C changes Shove and the caps, and a crashed mass dies faster |

---

## 5. The candidates and why

**Selection rule** (fixed in the pre-registration before any candidate outcome):
- raise the elasticity over the range that carries the objective's weight, about 0–120 invested;
- keep diminishing returns, except in one labelled reference;
- match the baseline's overall strength: the discount-weighted mean of ln(M_candidate / M_baseline) over the natural
  investment distribution is zero (every arm within ±0.001).

Every arm is an existing configuration field of `GameConfig.m45_calibrated(**overrides)` (`rl/m46b/curves.py`). No game
code changed.

| arm | fields | M at 5 / 35 / 116 / 234 invested | elasticity at 15 / 35 / 116 / 234 | role |
|---|---|---|---|---|
| baseline | none: saturating, A 1.5, K 16 | 1.19 / 1.66 / 1.98 / 2.12 | 0.18 / 0.17 / 0.12 / 0.08 | reference |
| **C1** slow saturation | `invest_gain=3.24, invest_scale=64.0` | 1.12 / 1.64 / 2.31 / 2.74 | 0.21 / 0.28 / 0.27 / 0.22 | modest candidate: 1.7–2.4× the elasticity, gentle decline, bounded (M < 4.24) |
| **C2** slower saturation | `invest_gain=9.53, invest_scale=256.0` | 1.09 / 1.59 / 2.62 / 3.64 | 0.20 / 0.34 / 0.47 / 0.45 | larger candidate: nearly linear in the relevant range, bounded (M < 10.5) |
| **C3** linear reference | `invest_curve="power", invest_alpha=0.016, invest_p=1.0` | 1.08 / 1.56 / 2.86 / 4.73 | 0.19 / 0.36 / 0.65 / 0.79 | no diminishing returns, unbounded; not a candidate for adoption |

**What matching costs.** A curve with a higher elasticity and the same overall strength must be weaker than the
baseline at small investment and stronger at large investment. Equal champions exchange the same damage in every arm,
so the difference shows only against waves, towers and the environment (S1, stage-typical states):

| arm | stage | M | wave capacity C | turns of Pressure per plate | duel ratio, +8 lead | +16 lead |
|---|---|---|---|---|---|---|
| baseline | early | 1.300 | 7 | 2.73 | ×1.254 | ×1.445 |
| baseline | mid | 1.712 | 14 | 1.40 | ×1.058 | ×1.108 |
| baseline | late | 1.961 | 18 | 1.11 | ×1.017 | ×1.033 |
| C1 | early | 1.206 | 7 | 2.95 | ×1.271 | ×1.531 |
| C1 | mid | 1.722 | 14 | 1.40 | ×1.107 | ×1.208 |
| C1 | late | 2.264 | **21** | 0.96 | ×1.040 | ×1.077 |
| C2 | early | 1.163 | 7 | 3.05 | ×1.252 | ×1.519 |
| C2 | mid | 1.697 | 14 | 1.42 | ×1.142 | ×1.287 |
| C2 | late | 2.538 | **23** | 0.86 | ×1.069 | ×1.139 |
| C3 | early | 1.144 | 6 | 3.11 | ×1.236 | ×1.498 |
| C3 | mid | 1.672 | 14 | 1.44 | ×1.159 | ×1.330 |
| C3 | late | 2.728 | **25** | 0.80 | ×1.096 | ×1.196 |

The candidates roughly double (C1) to triple (C3) what a mid- or late-game lead buys. They leave the early-game ratio
where it is and raise late wave capacity by 3–7 HP.

---

## 6. Gameplay results

All screens use the six baseline-trained finals; §8 states what that limits.

### 6.1 One-turn incentives (S2, exact ledger)

The exact ledger enumerates every chance outcome of one turn. HP is priced at 1.5 resources per bar, the M4.6a
measurement. Neutral situation, A ahead by 0 / +8 / +16 invested:

| stage | contrast | baseline | C1 | C2 | C3 |
|---|---|---|---|---|---|
| early | A: Trade − Hold, against Hold | +0.20 / +0.23 / +0.25 | +0.20 / +0.23 / +0.25 | +0.19 / +0.22 / +0.25 | +0.19 / +0.22 / +0.25 |
| mid | A: Trade − Hold, against Hold | +0.250 / +0.256 / +0.261 | +0.250 / +0.261 / +0.271 | +0.251 / +0.264 / +0.277 | +0.250 / +0.265 / +0.280 |
| late | A: Trade − Hold, against Hold | +0.257 / +0.258 / +0.260 | +0.259 / +0.263 / +0.267 | +0.261 / +0.267 / +0.273 | +0.262 / +0.271 / +0.280 |
| mid | B: Safe − Hold, against Trade | −0.18 / −0.18 / −0.17 | −0.18 / −0.17 / −0.16 | −0.18 / −0.17 / −0.16 | −0.18 / −0.17 / −0.15 |

- **Leads barely move one-turn trading incentives.** Mid-game, a 40-turn lead adds 0.011 to Trade − Hold under the
  baseline and 0.030 under the linear reference, on a base of 0.25.
- **Equal investment changes nothing.** The equal cells differ across arms by at most 0.01.
- **The weaker side should take a trade rather than run from it** for one turn in every arm (Safe − Hold against Trade
  is −0.15 to −0.24).
- **Freezes are identical across arms with one exception.**
  - At level 4 the guard survives a break in every arm and cell. At levels 10 and 15 no freeze survives a break in any
    arm. A walk-up never breaks a freeze.
  - For the breaker, breaking beats walking up by +0.44 at level 4 and +0.78 to +0.81 at levels 10 and 15, in every
    arm.
  - The exception: at level 4, held 5, breaker +8, the keep survives the break under C2 and C3 but not under the
    baseline and C1, because the breaker's wave capacity there is one HP lower.

Whatever a curve changes, it changes through accumulation over many turns, not through the incentives of a single turn.

### 6.2 Purchased leads (S3, 50-turn paired rollouts)

**Design.**
- Each cell starts from a constructed state with one side's invested gold raised by 8 or 16 resources (20 or 40 turns
  of income), in three situations (neutral, A's push, B's push).
- Six finals × 16 paired rollouts per cell; the statistic is G50.
- Per-final means, a t interval across the finals, and cross-arm differences paired by final.

**The lead premium.** The leader in seat A; its G50 in the lead cell minus the equal cell; policy against policy.

| stage | lead | baseline | C1 | C2 | C3 | C1 − baseline | C2 − baseline | C3 − baseline |
|---|---|---|---|---|---|---|---|---|
| early | +8 | +0.39 ± 0.25 | +1.02 ± 0.56 | +0.97 ± 0.28 | +0.74 ± 0.50 | +0.64 ± 0.71 | **+0.58 ± 0.43** | +0.35 ± 0.56 |
| early | +16 | +1.03 ± 0.42 | +1.71 ± 0.70 | +1.82 ± 0.34 | +1.30 ± 0.26 | +0.69 ± 0.91 | +0.79 ± 0.69 | +0.27 ± 0.58 |
| mid | +8 | +0.33 ± 0.48 | +0.51 ± 0.18 | +0.59 ± 0.52 | +0.67 ± 0.54 | +0.18 ± 0.49 | +0.26 ± 0.54 | +0.34 ± 0.70 |
| mid | +16 | +0.79 ± 0.36 | +0.68 ± 0.49 | +0.76 ± 0.60 | +0.99 ± 0.47 | −0.11 ± 0.64 | −0.03 ± 0.79 | +0.20 ± 0.61 |
| late | +8 | +0.24 ± 0.25 | −0.03 ± 0.31 | +0.23 ± 0.60 | +0.13 ± 0.62 | −0.26 ± 0.38 | −0.00 ± 0.49 | −0.11 ± 0.61 |
| late | +16 | +0.09 ± 0.33 | +0.11 ± 0.27 | +0.49 ± 0.41 | +0.22 ± 0.57 | +0.02 ± 0.45 | +0.40 ± 0.48 | +0.13 ± 0.83 |

- **Mid-game point estimates rise with the elasticity** (+0.33 to +0.51 / +0.59 / +0.67 at +8), as the mapping
  predicts. Every mid and late cross-arm interval includes zero.
- **Late-game premiums do not rise** at +8.
- **The screen was underpowered for D1.** M4.6a's E3 measured about 7.6 G100 per unit of ln attack ratio. From that
  scaling, the expected cross-arm differences are 0.1–0.35 per cell, against intervals of ±0.4–0.9. The exploratory
  follow-up in §6.6 has 8× the rollouts.
- **The early +8 premium is 1.9–2.6× the baseline's under every candidate**, significantly so under C2. The likely
  mechanism, derived from the formulas rather than measured, is that an early cell's 50-turn horizon runs into the
  mid-game. Both sides invest about 20 more resources in that time, and under the candidates the same 8-resource gap
  keeps more of its strength:

| arm | M ratio of an 8-resource gap at the start (17 vs 9) | after 50 turns of equal income (37 vs 29) |
|---|---|---|
| baseline | ×1.120 | ×1.044 |
| C1 | ×1.127 | ×1.070 |
| C2 | ×1.119 | ×1.083 |
| C3 | ×1.112 | ×1.087 |

  A lead that stays meaningful for longer is what the design intent asks for. As written, D2's early-premium test
  counts it against the candidates (§7).

**The objective beside its gold and XP channels.** The leader's 40-turn lead (+16), with the lead cell minus the equal
cell, under policy play over 50 turns. The premium is discounted; the channels are undiscounted accounting totals:

| stage | arm | premium (G50) | leader earned | trailer earned | leader farm | leader plates | trailer's kill bounty | leader deaths | leader XP | trailer XP |
|---|---|---|---|---|---|---|---|---|---|---|
| early | baseline | +1.03 | +0.69 | −0.51 | +0.43 | +0.30 | −0.22 | −0.11 | +2.3 | −0.1 |
| early | C1 | +1.71 | +1.07 | −1.08 | +0.48 | +0.45 | −0.58 | −0.29 | +3.1 | −1.0 |
| early | C2 | +1.82 | +1.25 | −1.12 | +0.61 | +0.44 | −0.47 | −0.23 | +3.8 | −2.5 |
| early | C3 | +1.30 | +0.87 | −0.75 | +0.39 | +0.41 | −0.38 | −0.19 | +2.3 | −0.5 |
| mid | baseline | +0.79 | +0.55 | −0.46 | +0.18 | +0.20 | −0.10 | −0.05 | 0 | 0 |
| mid | C1 | +0.68 | +0.48 | −0.40 | +0.36 | +0.15 | −0.22 | −0.11 | 0 | 0 |
| mid | C2 | +0.76 | +0.44 | −0.51 | +0.21 | +0.20 | −0.20 | −0.10 | 0 | 0 |
| mid | C3 | +0.99 | +0.75 | −0.50 | +0.26 | +0.30 | −0.10 | −0.05 | 0 | 0 |

- **The leader converts through farm, plates and survival.** It gains its own farm and plates, and the trailer kills
  it less often, so the trailer earns less bounty.
- **XP barely moves.** Early it shifts by 1–4 XP over 50 turns, at most about one turn of wave XP (4 XP while
  holding at a wave). Mid-game both
  champions reach the level cap within the horizon, so XP does not move at all.
- **These are accounting totals.** They say where the gold difference appears, not what caused it.

**Conversion.** This is threaten minus Hold only, for the leader, behaviour only, over 10 turns.
- Forced pressure is worse than farming in every arm and cell.
- At +16 the contrast is −0.60 ± 0.53 (mid) and −0.57 ± 0.36 (late) under the baseline. Under C1 it is −0.48 / −0.35,
  under C2 −0.58 / −0.48 and under C3 −0.32 / −0.38.
- No cross-arm change is significant. The policy does better than both forced plans in every arm.
- The candidates do not make bullying pay at these lead sizes for baseline-trained play.

**The trailing side.** The trailer is in seat A. The table gives its G50 under the policy and, in brackets, the change
against the baseline:

| stage | lead | baseline | C1 | C2 | C3 |
|---|---|---|---|---|---|
| early | +8 | −0.83 ± 0.32 | −0.76 ± 0.27 (+0.07 ± 0.36) | −0.84 ± 0.28 (−0.00 ± 0.29) | −0.87 ± 0.16 (−0.04 ± 0.35) |
| early | +16 | −0.95 ± 0.35 | −1.45 ± 0.48 (**−0.50 ± 0.36**) | −1.69 ± 0.32 (**−0.74 ± 0.43**) | −1.41 ± 0.27 (−0.45 ± 0.58) |
| mid | +8 | −0.31 ± 0.48 | −0.38 ± 0.49 (−0.07 ± 0.43) | −0.50 ± 0.30 (−0.19 ± 0.63) | −0.73 ± 0.27 (−0.42 ± 0.47) |
| mid | +16 | −0.46 ± 0.32 | −0.96 ± 0.21 (**−0.50 ± 0.38**) | −0.99 ± 0.47 (−0.53 ± 0.54) | −1.35 ± 0.64 (**−0.89 ± 0.78**) |
| late | +8 | −0.13 ± 0.28 | −0.12 ± 0.32 (+0.01 ± 0.46) | −0.20 ± 0.26 (−0.07 ± 0.43) | −0.06 ± 0.23 (+0.08 ± 0.33) |
| late | +16 | −0.28 ± 0.21 | −0.06 ± 0.53 (+0.22 ± 0.55) | −0.29 ± 0.31 (−0.00 ± 0.27) | −0.36 ± 0.36 (−0.07 ± 0.42) |

At a 40-turn lead, early and mid, the trailer loses about twice as much under the candidates, and four of these
changes are significant. Seen from the trailer's side, this is the clearest evidence in the screen that the candidates
make leads more consequential.

The trailer's forced responses at +16 (candidate − policy; 10 turns, behaviour only, the wave left to the policy):

| stage | arm | Hold only (stay and farm) | Safe only (retreat) | Trade only (fight back) | reset (Safe, then Recall) |
|---|---|---|---|---|---|
| mid | baseline | −0.29 ± 0.41 | −1.41 ± 0.39 | −1.26 ± 0.42 | −1.31 ± 0.65 |
| mid | C1 | −0.06 ± 0.38 | −1.04 ± 0.20 | −1.41 ± 0.22 | −0.94 ± 0.48 |
| mid | C2 | −0.13 ± 0.47 | −1.34 ± 0.52 | −1.68 ± 0.50 | −1.15 ± 0.56 |
| mid | C3 | +0.04 ± 0.76 | −1.09 ± 0.62 | −1.55 ± 0.64 | −0.97 ± 0.73 |
| late | baseline | −0.30 ± 0.27 | −1.39 ± 0.22 | −1.89 ± 0.29 | −0.69 ± 0.56 |
| late | C1 | −0.64 ± 1.01 | −1.77 ± 0.46 | −1.76 ± 0.33 | −1.90 ± 0.37 |
| late | C2 | −0.63 ± 0.80 | −1.84 ± 0.42 | −1.78 ± 0.52 | −1.91 ± 0.81 |
| late | C3 | −0.38 ± 0.51 | −1.84 ± 0.47 | −1.85 ± 0.83 | −2.35 ± 0.66 |

- **The trailer's best forced response is always to stay and farm.** Retreating, fighting back and resetting each
  cost about a resource or more over 10 turns.
- **The policy is at least as good as every forced response** in every arm and cell. The best forced response minus
  the policy is at most +0.06 and never significant.
- **No sign of contest-and-die.** Forced fighting back is not better under any candidate.
- **A late reset costs more under the candidates** (−0.69 under the baseline, −1.90 to −2.35 under the candidates). A
  likely reason is that a missed late wave is worth more when the opponent's wave capacity is higher. That reason is
  an inference, not a measurement.

**Validation.** Six fixed cells per arm were re-run at 150 turns. The 150-turn values track the 50-turn values:

| arm | mean change | slope |
|---|---|---|
| baseline | +0.15 | 1.02 |
| C1 | −0.09 | 0.92 |
| C2 | −0.03 | 0.75 |
| C3 | −0.10 | 0.90 |

The 50-turn readings stand. C2's slope suggests part of its premium fades after 50 turns.

### 6.3 Freeze contests (S4, full 50-turn sequences)

**Design.** Levels 4 and 10, held 5 or 12 turns, with investment equal, freezer +8 or breaker +8. Six finals × 16
paired rollouts. The table gives each candidate minus the policy, as the focal side's G50, averaged over the 12 cells
per arm (the range over cells in brackets):

| arm | breaker: break | walk up | step back | reset | freezer: keep | guard | freeze held, turns in 20 (L4 / L10) |
|---|---|---|---|---|---|---|---|
| baseline | −0.01 [−0.27, +0.41] | +0.02 [−0.45, +0.66] | −1.63 [−2.80, −0.95] | −0.68 [−1.81, +0.09] | +0.01 [−0.49, +0.79] | −0.32 [−1.35, +0.25] | 3.9 / 1.2 |
| C1 | +0.02 [−0.38, +0.70] | −0.12 [−0.62, +0.37] | −1.65 [−2.56, −0.99] | −0.67 [−1.59, +0.01] | −0.17 [−0.56, +0.43] | −0.37 [−1.94, +0.83] | 4.0 / 1.2 |
| C2 | −0.03 [−0.50, +0.30] | −0.07 [−0.67, +0.63] | −1.65 [−2.18, −0.85] | −0.66 [−1.47, +0.30] | −0.12 [−0.68, +0.34] | −0.50 [−1.73, +0.77] | 4.6 / 1.2 |
| C3 | −0.11 [−0.47, +0.18] | −0.15 [−0.55, +0.45] | −1.65 [−2.26, −1.15] | −0.85 [−1.68, +0.09] | −0.12 [−0.54, +0.26] | −0.39 [−1.74, +0.50] | 4.6 / 1.2 |

- **Contesting a freeze stays correct.** In every arm, walking up and breaking are as good as the policy. Stepping back
  is the worst choice in every cell of every arm, including for the weaker breaker.
- **Freezes last as long as before.** Under the policy they hold about 4 turns in 20 at level 4 and about 1 at level
  10. They hold slightly longer at level 4 under C2 and C3, matching the S2 exception.
- **Guarding early.** The largest guard effect is C1's at level 4, held 5, freezer +8: +0.83 ± 0.67, against −0.04 ±
  0.45 under the baseline. It is one nominally significant result among 72 candidate-cell comparisons per arm, so by
  itself it is not evidence.

**A representative sequence:** the weaker breaker at level 4, held 12, with the freezer 8 ahead. The table gives
20-turn readouts: the policy's values, then each forced response's change.

| arm | breaker's response | ΔG50 | breaker earned | freezer earned | breaker deaths | breaker Recalls | turns away | breaker farm lost | freezer's minions crashed (HP) |
|---|---|---|---|---|---|---|---|---|---|
| baseline | policy | 0 | 7.06 | 9.93 | 0.35 | 0.58 | 2.1 | 2.31 | 13.2 |
| baseline | break | −0.22 | −0.03 | −0.07 | −0.01 | +0.04 | +0.1 | +0.05 | −0.9 |
| baseline | walk up | −0.22 | +0.42 | +0.21 | +0.07 | −0.08 | +0.1 | −0.10 | +0.2 |
| baseline | step back | −2.22 | −0.55 | +0.28 | −0.33 | −0.51 | −2.0 | +0.90 | +27.1 |
| baseline | reset | −1.52 | −0.52 | +0.14 | −0.11 | +0.79 | +1.3 | +0.65 | +5.1 |
| C1 | policy | 0 | 6.88 | 9.88 | 0.36 | 0.72 | 2.3 | 2.34 | 11.6 |
| C1 | break | +0.37 | +0.08 | −0.09 | −0.01 | +0.00 | +0.0 | +0.04 | −0.2 |
| C1 | walk up | +0.37 | +0.50 | +0.25 | +0.08 | −0.23 | −0.1 | −0.14 | +1.4 |
| C1 | step back | −0.99 | −0.17 | +0.19 | −0.36 | −0.67 | −2.3 | +0.80 | +29.0 |
| C1 | reset | −0.68 | −0.44 | +0.02 | −0.12 | +0.72 | +1.2 | +0.70 | +4.5 |

The policy already contests: it loses about 3 resources over 20 turns and dies 0.35 times. Walking up costs a little
more HP but gains farm. Stepping back avoids the deaths but gives up the wave, and the freezer's minions crash into the
breaker's tower. That is the structure of a good contest the brief describes, and it is the same under both curves.

### 6.4 Recall and shopping (S5)

The table uses the M4.6a E5 states, K = 12 and G50:

| arm | family | Recall − stay | the same, gold pre-invested | shopping component |
|---|---|---|---|---|
| baseline | full HP | −0.35 ± 0.34 | −0.37 ± 0.19 | +0.02 ± 0.27 |
| baseline | mid HP | +0.33 ± 0.07 | +0.20 ± 0.12 | +0.13 ± 0.14 |
| C1 | full HP | −0.37 ± 0.20 | −0.45 ± 0.26 | +0.09 ± 0.21 |
| C1 | mid HP | +0.36 ± 0.37 | +0.25 ± 0.26 | +0.11 ± 0.37 |
| C2 | full HP | −0.42 ± 0.19 | −0.39 ± 0.22 | −0.03 ± 0.24 |
| C2 | mid HP | +0.29 ± 0.14 | +0.16 ± 0.18 | +0.12 ± 0.24 |
| C3 | full HP | −0.27 ± 0.25 | −0.41 ± 0.25 | +0.14 ± 0.19 |
| C3 | mid HP | +0.40 ± 0.23 | +0.25 ± 0.19 | +0.15 ± 0.13 |

- **Recall keeps its role.** In every arm a full-HP Recall costs about 0.3–0.4 and a mid-HP Recall gains about the
  same, so the reset is not free under any arm (P7).
- **The shopping component is about zero in every arm.** It measures the value of investing 10–20 turns earlier, not
  the value of the gold (§4), so it is small under any curve.

### 6.5 Natural starts (S6)

**Design.** Self-play from the normal start, 16 games × 600 turns per final and arm. This is kept apart from the local
counterfactuals.

| arm | attack ratio q90 | share ≥ 1.25 | duel ratio q90 | duel share ≥ 1.25 | leader's duel ratio: early | mid | late | very late | persistence slope (100 to 300) |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 1.047 | 0.24% | 1.096 | 2.1% | ×1.034 | ×1.041 | ×1.010 | ×1.006 | 0.75 ± 0.40 |
| C1 | 1.061 | 0.07% | 1.125 | 2.5% | ×1.025 | ×1.057 | ×1.029 | ×1.015 | 1.02 ± 0.25 |
| C2 | 1.079 | 0.16% | 1.165 | 3.6% | ×1.017 | ×1.075 | ×1.051 | ×1.030 | 1.19 ± 0.29 |
| C3 | 1.098 | 0.15% | 1.206 | 5.9% | ×1.015 | ×1.082 | ×1.072 | ×1.059 | 1.09 ± 0.36 |

The lane sanity readouts below are per side, and per 100 turns where a rate:

| arm | crashed HP | plates by 150 | plates by 300 | deaths | Recalls | farm collected | Shove share | connected trades | C, turns 240–320 |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 66.4 | 1.91 | 2.73 | 0.64 | 4.23 | 0.761 | 0.372 | 13.8 | 18.0 |
| C1 | 69.0 (+4%) | 1.99 (+4%) | 2.86 | 0.59 (−7%) | 4.14 | 0.761 | 0.365 | 13.6 | 20.6 |
| C2 | 69.5 (+5%) | 1.95 | 2.76 | 0.56 (−13%) | 4.09 | 0.765 | 0.369 | 13.5 | 23.1 |
| C3 | 69.8 (+5%) | 1.95 | 2.80 | 0.54 (−15%) | 4.00 | 0.765 | 0.365 | 13.6 | 24.8 |

- **Leads matter more from mid-game on.** The leader's duel ratio rises mid and late under every candidate, and falls
  slightly early, where the candidates are weaker.
- **Natural leads are not decisive under any arm.** An attack ratio of 1.25 or more occurs in under 0.25% of present
  turns.
- **Leads keep their size.** Under the baseline the turn-100 lead shrinks by a quarter by turn 300. Under the
  candidates it keeps its size: pooled over all games, the slopes are 0.75 / 0.94 / 1.08 / 1.12.
- **The equal lane is intact.** Crashes, plates, Recalls, farm collection, the Shove share and connected trades are all
  within 5% or 0.01.
- **Deaths fall 7–15%,** because a higher late M also divides gank, tower and minion damage.
- **Late wave capacity rises by 2.6–6.8 HP.**

### 6.6 Exploratory: the lead premium with adequate power (P1X)

*Exploratory.* This run was added after S3 proved underpowered, and recorded in a dated amendment to the
pre-registration before its results existed. Its design:
- 128 paired rollouts per cell, 8× the S3 count;
- both orientations: the leader in seat A and in seat B. The premium is half of A's G with the lead minus A's G when
  the opponent leads, which cancels the equal cell and any seat effect;
- policy against policy, 100 turns, the mid and late stages, three situations, six finals;
- the same states and seeds in every arm.

| stage | lead | horizon | baseline | C1 | C2 | C3 | C1 − baseline | C2 − baseline | C3 − baseline |
|---|---|---|---|---|---|---|---|---|---|
| mid | +8 | G50 | +0.30 ± 0.15 | +0.47 ± 0.10 | +0.66 ± 0.18 | +0.67 ± 0.19 | **+0.17 ± 0.11** (×1.58) | **+0.37 ± 0.09** (×2.22) | **+0.37 ± 0.12** (×2.24) |
| mid | +8 | G100 | +0.38 ± 0.19 | +0.58 ± 0.14 | +0.84 ± 0.22 | +0.88 ± 0.28 | **+0.20 ± 0.11** (×1.53) | **+0.46 ± 0.13** (×2.21) | **+0.50 ± 0.16** (×2.33) |
| mid | +16 | G50 | +0.52 ± 0.17 | +0.73 ± 0.14 | +0.93 ± 0.23 | +1.05 ± 0.25 | **+0.20 ± 0.10** (×1.38) | **+0.41 ± 0.10** (×1.78) | **+0.52 ± 0.12** (×2.00) |
| mid | +16 | G100 | +0.59 ± 0.17 | +0.96 ± 0.22 | +1.29 ± 0.30 | +1.45 ± 0.33 | **+0.37 ± 0.06** (×1.63) | **+0.70 ± 0.17** (×2.20) | **+0.86 ± 0.17** (×2.46) |
| late | +8 | G50 | +0.08 ± 0.08 | +0.14 ± 0.04 | +0.23 ± 0.13 | +0.27 ± 0.10 | +0.06 ± 0.06 (×1.67) | **+0.14 ± 0.07** (×2.70) | **+0.19 ± 0.08** (×3.22) |
| late | +8 | G100 | +0.16 ± 0.07 | +0.16 ± 0.06 | +0.33 ± 0.15 | +0.39 ± 0.09 | −0.01 ± 0.06 (×0.97) | **+0.17 ± 0.10** (×2.02) | **+0.23 ± 0.04** (×2.40) |
| late | +16 | G50 | +0.20 ± 0.12 | +0.24 ± 0.07 | +0.35 ± 0.17 | +0.51 ± 0.19 | +0.04 ± 0.06 (×1.19) | +0.15 ± 0.17 (×1.75) | **+0.31 ± 0.16** (×2.55) |
| late | +16 | G100 | +0.32 ± 0.18 | +0.37 ± 0.12 | +0.52 ± 0.12 | +0.84 ± 0.26 | +0.05 ± 0.10 (×1.16) | +0.20 ± 0.16 (×1.65) | **+0.52 ± 0.22** (×2.65) |

The ratio in brackets is the arm's premium over the baseline's. Bold marks a difference whose interval excludes zero.

- **Mid-game leads clearly become more consequential.** Every mid-game difference excludes zero. An 8-resource lead's
  premium rises from +0.30 to +0.47 / +0.66 / +0.67 at 50 turns.
- **Late leads gain under C2 and C3 only.** C2 and C3 raise the late premium by ×2.0–3.2. C1's late gain is small: +0.06
  ± 0.06 at 50 turns, and none at 100.
- **The ratios track the mapping.** At mid-game the S1 duel-ratio leverage predicts ×1.8 / ×2.4 / ×2.6, against
  ×1.6 / ×2.2 / ×2.2 measured. Late, the measured effect is smaller than predicted.
- **Magnitudes stay modest.** Under C2 an 8-resource mid-game lead returns about 0.66 resources of further lead over
  50 turns (0.84 over 100), against 0.30 (0.38) under the current curve.
- **S3's null was a power failure, not an absence of effect.** Applied to these exploratory numbers, D1's rule would be
  met by C2 and C3 at both stages, and by C1 at mid-game only, since its late interval touches zero. The
  pre-registered outcome stands.

---

## 7. Pre-registered scorecard

| criterion | C1 | C2 | C3 |
|---|---|---|---|
| **D1** consequential leads: the +8 premium ≥ 1.5× the baseline's, mid and late, interval of the difference excluding 0 | **fail**: mid +0.51 against ≥ +0.50, but the difference is +0.18 ± 0.49; late −0.03 | **fail**: mid +0.59, difference +0.26 ± 0.54; late +0.23 | **fail**: mid +0.67, difference +0.34 ± 0.70; late +0.13 |
| **D2** not decisive: early premium ≤ 1.5× baseline (≤ +0.58); share ≥ 1.25 at most 5%; persistence slope ≤ 1.2 | **fail** on the early premium (+1.02); share 0.07%; slope 1.02 | **fail** (+0.97); 0.16%; 1.19 | **fail** (+0.74); 0.15%; 1.09 |
| **D3** the equal lane, within ±25% and Shove ±0.10 | pass | pass | pass |
| **D4** waves and towers: C within ±2 HP at S ≤ 116; mid Pressure within ±25% | **fail**: late C +3 HP | **fail**: +5 HP | **fail**: +7 HP |
| P6 trailing side: flag if the trailer's best response loses more than 1.5× the baseline's | **flag** at mid +16 (−0.96 against −0.46, ×2.1) | **flag** (×2.2) | **flag** (×2.9) |
| P7 purchases: shopping component; Recall − stay at full HP above −0.2 is a flag | about zero; −0.37, no flag | about zero; −0.42, no flag | about zero; −0.27, no flag |
| P8 freezes | unchanged | the keep survives one more break cell at level 4 | as C2 |

**Rule outcome: no candidate advances to a pilot.** The scorecard is generated from the stage summaries
(`results/m46b/report.md`, first table). How close each miss is:
- **D4 is a narrow miss for C1.** At level 18, C1 stays within ±2 HP of the baseline's capacity up to 86 invested,
  which typical play reaches around turn 225. At the late stage state, 108 invested, it is one HP over the bound.
  The cause is structural: matching overall strength raises absolute late strength, and wave capacity and Pressure
  scale with it.
- **D1 is not met as registered, but the exploratory follow-up shows the effect.** S3's intervals were too wide for the
  predicted size. With 8× the rollouts in both orientations (§6.6), the mid-game premium rises significantly under all
  three candidates, and the late premium under C2 and C3. Because the follow-up was exploratory, it does not change the
  rule outcome.
- **D2 fails on the early premium only.** That failure follows mechanically from leads keeping their value as
  investment grows (§6.2), which is part of what the design intent asks for. The pre-registration's test does not
  separate "early leads are too decisive" from "leads made early stay meaningful into the mid-game". The decision rule
  is applied as registered anyway.
- **P6 reads as registered, on absolute losses.** On that reading it flags, because the trailer loses about twice as
  much. Read as the room the trailer has to improve on the policy, it does not flag, since the policy remains the
  trailer's best response. A larger loss for the trailer is the mirror of a more consequential lead, so the flag
  qualifies D1 rather than contradicting it.

---

## 8. Trade-offs and uncertainty

- **Matching forces a trade-off.** Equal overall strength with a higher elasticity means weaker early and stronger
  late. Late absolute strength drives wave capacity, Pressure and the environment's damage, so any such candidate moves
  D4 and the death rate. Avoiding that would need the multiplier's effect on waves and towers to be separated from its
  effect on duels. That is a design question for review, not a curve choice.
- **The curve controls only part of the effect.** Natural relative leads are 4–10%. Even at the linear reference's late
  elasticity of 0.63, a 4% lead is about a 2.5% multiplier lead, ×1.05 in a duel. The ×1.01 is mostly a statement
  about how small natural leads are relative to accumulated income.
- **Out of distribution.** The six finals were trained on the baseline. Under a candidate curve, the observation's
  attack-power entry pairs differently with the invested entry. The screens measure how baseline-trained play converts
  and defends leads under another curve, and they find mechanical problems. They cannot say what retrained self-play
  would learn: it could convert leads better, or defend better.
- **Power and equivalence.** With K = 16 and six finals, S3's intervals are ±0.4–0.9 per cell. The follow-up, with
  K = 128 in both orientations, narrows them to ±0.04–0.3. A nonsignificant cross-arm difference is not evidence that
  the arms are equivalent. S4's freeze cells and S5's Recall cells remain at the lower power.
- **Multiplicity.** S3–S5 contain hundreds of cell comparisons. Isolated nominal significance, such as the C1 guard
  cell, is expected by chance. The pre-registered primaries are D1–D4.
- **Horizon.** The primary readout is 50 turns, validated at 150 turns on six cells per arm.
- **Accounting, not causation.** The economic channels say where a difference in gold appears, not why.

---

## 9. Recommendation

**Retain the current curve for now, and settle the two failed criteria before any pilot.**
- **No candidate meets the pre-registered rule.** The rule then calls for retaining the curve or investigating the
  failed criterion.
- **The candidates do what they were built to do.** In the powered follow-up, mid-game leads become 1.6–2.2× more
  consequential, close to what the mapping predicts. One-turn incentives, freeze contests, Recall, the equal lane and
  the weaker side's best response stay as they are. The open question is therefore whether the side effects are
  acceptable, not whether a curve can strengthen leads.
- **Late wave capacity is the first side effect (D4).** It rises by 3 / 5 / 7 HP. Under baseline-trained play, crashes
  and plates stay within 5%, but retrained play could learn to use the extra capacity.
- **A larger early premium is the second (D2).** An early lead's 50-turn premium roughly doubles. Natural early leads
  stay far from decisive: an attack ratio of 1.25 or more occurs on under 0.25% of turns. The registered test also
  cannot tell early decisiveness apart from a lead that simply stays meaningful.
- **No curve removes the main cause of the ×1.01.** Natural leads stay 4–10% of investment. A curve can roughly double
  their leverage, but it cannot make them large.
- **The current curve already gives meaningful early investment.** Twenty turns of income buy ×1.25 in a duel early and
  ×1.06 mid-game, and the marginal value declines gently.

**What to settle first** (proposals for review, none implemented):
1. **D4: is +3 HP of late wave capacity acceptable?** That is C1's cost. If it is not, the variant that avoids it
   separates the combat multiplier from wave capacity and Pressure, for example by keeping the baseline curve for waves
   and towers. That is a rule change and out of scope here.
2. **D2: a sharper early-decisiveness test.** It would use natural early leads, or a horizon that ends before the
   mid-game, so that "decisive early" and "stays meaningful" separate.
3. **How much late-game leverage is wanted.** C1 gains mid-game leverage but little late. C2 roughly doubles the late
   premium at +5 HP of capacity.
4. **The powered instrument as the standard.** Any further curve comparison should use K ≥ 128, both orientations and
   100 turns, as in §6.6.

**If the review accepts the costs, pilot C1.**
- It gains significant mid-game leverage (×1.6) with the smallest D4 miss: one HP late, from about turn 225.
- It keeps diminishing returns and is bounded.
- Its natural early ratios are unchanged, and its lane sanity is within 4–7%.
- Its persistence slope is near 1.

C2 is the alternative if more late-game leverage is wanted, at +5 HP of late capacity.

A matched pilot, with the same budget and seeds as a baseline control, must establish:
- **Strong-side conversion.** Retrained leaders turn +8 and +16 leads into a premium of at least 1.5× the baseline's,
  mid and late, measured with K ≥ 128 in both orientations.
- **Weak-side defence.**
  - The trailer's losses stay within the agreed bound, or the larger loss is accepted explicitly.
  - Staying and contesting remains its best response, and deaths per contest do not rise.
  - Retreat and reset stay viable: a mid-HP Recall stays positive, and the late reset cost is checked.
- **Natural starts.**
  - Attack ratio ≥ 1.25 in at most 5% of turns, and a persistence slope ≤ 1.2.
  - Lane sanity within ±25%.
  - Plates, crashes and the Shove share monitored late, because of D4.

---

## 10. Reproduction and files

Every stage caches its jobs under `results/rl/checkpoints/m46b/`; a re-run reuses them. Summaries are written to
`results/m46b/`.

```
python -m lanegame.experiments.rl.m46b verify_v1 verify_v2 verify_v2_analysis --workers 6
python -m lanegame.experiments.rl.m46b audit --workers 6
python -m lanegame.experiments.rl.m46b prereg
python -m lanegame.experiments.rl.m46b screen_s1 screen_s2 screen_s3 screen_s4 screen_s5 screen_s6 --workers 6
python -m lanegame.experiments.rl.m46b screen_p1x --workers 6
python -m lanegame.experiments.rl.m46b report
python -m pytest tests/test_m46b.py
```

The screen stages refuse to run unless the frozen pre-registration matches its recorded SHA-256. Wall-clock times on
six workers:

| stage | time |
|---|---|
| verify_v2 | 21 min |
| S3 | 13 min |
| S4 | 5 min |
| S5 | 14 min |
| P1X | about 25 min |
| the rest | a few minutes each |

**Measurement code**, shared and backward compatible:
- `lanegame/experiments/rl/m44/metrics.py`: lane-only positions and the Ward share.
- `lanegame/experiments/rl/m44/records.py`: graph distance, with W kept.
- `lanegame/experiments/m45b/engine.py`: the ledger's away channel.
- `lanegame/experiments/rl/m45c/horizon.py`: the battery's ledger keys and the behaviour-only twins.
- `lanegame/experiments/rl/m46a/engine.py`: the base ledger, and the optional arm of a paired job.
- `lanegame/experiments/rl/m46a/policies.py`: the Trade, Safe and Hold rules, the reset plan, and fixed plans.

**New files:**
- `lanegame/experiments/rl/m46b/`: common, curves, verify, audit, screen, report, and the CLI.
- `tests/test_m46b.py`.
- `docs/m4/m4_6/m4_6b_preregistration.md` and this report.
- The `results/m46b/` folder.

**Docs reorganization.** The M4 documents moved into `docs/m4/m4_1/` … `docs/m4/m4_6/`, with an index at
`docs/m4/README.md`. The links and path strings in the READMEs, the roadmap, the mathematical reference, code
docstrings and tests were updated to match. `results/` was not moved.

**Not changed:**
- No game rule, and not `GameConfig.m45_calibrated()`: the core modules differ only in documentation paths.
- No default.
- No checkpoint or stored result: the verification wrote to separate roots.
- No training ran.
- The mathematical reference's CURRENT sections, because no rule or objective changed.

Nothing is staged or committed.

