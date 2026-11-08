# M4.6b — Pre-registration of the investment candidate screen

*Written 2026-10-15 after the audit of the CURRENT curve (`results/m46b/audit.json`) and after the measurement fixes
(V1 verified), before any gameplay outcome of a candidate curve. Frozen by stage `prereg` (SHA-256 in
`results/m46b/preregistration.json`). The calibrated baseline `GameConfig.m45_calibrated()` is unchanged; every
candidate is an opt-in configuration of existing fields. No PPO training. Policies: the six 16.384M M4.5c finals,
trained on the baseline. Under a candidate curve they are out of distribution (the observation's attack-power entry
takes new values for the same invested gold), so these screens find mechanical problems and promising candidates.
They cannot tell what retrained self-play would learn.*

## 1. What the audit established (current curve, informing the choice)

- **Natural investment levels.** The median invested gold is 15 (turn 50), 35 (100), 56 (150), 116 (300) and 234
  (600). The objective's discount-weighted median is 22 (quartiles 6 / 50, 90% 87). Income is about 0.40 resources per
  turn.
- **Leads are small relative to investment.** The leader is ahead 10% (turn 50), 8% (100) and 4% (300) in earned
  gold. Leads persist, but their relative size shrinks because both sides keep earning.
- **The current curve's elasticity** (d ln M / d ln S, the relative strength gained per relative investment lead) is
  0.17 up to 50 invested, 0.12 at 100 and 0.09 at 200. Absolute marginal value falls 20× over the first 100 resources.
- **Why the earlier ×1.01.** It came from late-game sampling (the 10–20 lead bin has its median at turn 418 and 160
  invested), saturation, small relative leads, and, early on, leads held as unspent gold.

## 2. Arms (fixed)

The selection rule. Each candidate raises the elasticity in the relevant investment range (about 0–120 resources),
keeps diminishing returns where stated, and matches the baseline's overall strength. "Overall strength" means the
discount-weighted mean of ln(M_candidate / M_baseline) over the natural investment distribution (all audit states,
weight 0.99^t) is zero, which every arm meets to within ±0.001. Consequently every candidate is weaker than the
baseline at small investment and stronger at large investment. That trade-off is unavoidable for any curve with a
higher elasticity, and the screens measure it.

| arm | fields (`m45_calibrated(**...)`) | M at 5 / 35 / 116 / 234 resources | elasticity at 15 / 35 / 116 / 234 | role |
|---|---|---|---|---|
| baseline | (none): saturating, A 1.5, K 16 | 1.19 / 1.66 / 1.98 / 2.12 | 0.18 / 0.17 / 0.12 / 0.08 | reference |
| **C1** slow saturation | `invest_gain=3.24, invest_scale=64.0` | 1.12 / 1.64 / 2.31 / 2.74 | 0.21 / 0.28 / 0.27 / 0.22 | modest candidate: about 1.7–2.4× the elasticity, gentle decline, bounded (M < 4.24) |
| **C2** slower saturation | `invest_gain=9.53, invest_scale=256.0` | 1.09 / 1.59 / 2.62 / 3.64 | 0.20 / 0.34 / 0.47 / 0.45 | larger candidate: nearly linear in the relevant range, bounded (M < 10.5) |
| **C3** linear reference | `invest_curve="power", invest_alpha=0.016, invest_p=1.0` | 1.08 / 1.56 / 2.86 / 4.73 | 0.19 / 0.36 / 0.65 / 0.79 | reference without diminishing returns, unbounded; not a candidate for adoption |

## 3. Screens

Stages (audit medians, rounded): **early** level 8, S 9; **mid** level 16, S 42; **late** level 18, S 108. Leads are
added to one side's invested gold in turns of the measured income (0.40 / turn): **+8** (20 turns) and **+16** (40
turns). The same gold means different strength under each arm, and that is the point of the comparison.

- **S1 mapping tables** (pure): M, elasticity, the attack and duel ratios a lead buys, wave capacity C, Pressure per
  turn and the expected damage per exchange, per stage and arm.
- **S2 exact one-turn ledger** (M4.6a E1 machinery):
  - zoning situations (neutral, A's push, B's push) × stages × leads {0, +8, +16} for A;
  - freeze cells (levels 4, 10, 15 × held 5 / 12 / 20) × investment {equal, freezer +8, breaker +8};
  - the threat game (Trade − Hold against Hold / Safe, Safe − Hold against Trade, priced at 1.5 per bar), P(freeze
    survives a walk-up Shove against keep / guard), and the "both stronger" effect (equal cells across arms) kept
    apart from the lead effect (lead cell − equal cell within an arm).
- **S3 paired rollouts from purchased leads** (M4.6a engine, K = 16, horizon 50, 10-turn plans, opponent = the
  policy, six finals; 27 cells = situations × stages × leads):
  - **strong focal** (the leader in seat A): the policy, threaten (behaviour only), Hold only (behaviour only);
  - **weak focal** (the trailer in seat A; leads +8 / +16): the policy, Hold only (contest by farming), Safe only
    (retreat), Trade only (fight back), reset (Safe until Recall is legal, then Recall; behaviour only, the wave left to
    the policy).
- **S4 freeze contests** (K = 16, horizon 50; levels 4 and 10 × held 5 / 12 × investment {equal, freezer +8, breaker
  +8}):
  - **freezer focal:** the policy, keep (Thin + Hold while held, ≤ 10 turns), guard (Thin + Trade while held);
  - **breaker focal:** the policy, break (Shove + Hold, one turn), walk up (Hold only), step back (Safe only), reset;
  - the full sequence over 50 turns: resources, HP, freeze held turns, crash, deaths, Recalls, missed income.
- **S5 Recall and shopping** (the M4.6a E5 states, re-collected from baseline self-play: the first 24 per final of
  recall_full and recall_mid; K = 12, horizon 50): Recall − stay, the same with the gold pre-invested, and the
  shopping component, per arm.
- **S6 natural starts** (s0 self-play, 16 games × 600 turns per final and arm; a full-game distribution check, kept
  apart from S2–S5):
  - investment gaps and ratios: attack and duel ratio quantiles, shares ≥ 1.1 and ≥ 1.25, the audit decomposition;
  - lead persistence (the slope of the lead at 300 on the lead at 100);
  - lane sanity: crashed HP and plates by turn 150 and 300, deaths and Recalls per 100 turns, farm collection, the
    Shove share, mean C by stage.

**Statistic.** G_h, the focal seat's discounted gold lead (the training objective truncated at h). The critic is not
used (it was trained on the baseline). Undiscounted gold, XP and the accounting channels are diagnostics:
- earned = farm + bounty + plates;
- the weaker side's farm = available − lost while present − lost while away.

These are accounting identities, not causal attributions. Per final: the mean over cells of the per-cell paired mean;
across the six finals a t interval; cross-arm differences paired by final (the arms share states and seeds).

**Validation.** Six fixed S3 strong-focal cells per arm (seeded choice) under the policy are re-run at 150 turns; the
pooled 50-versus-150 check (mean change, slope, contradictions). If it fails, the affected comparisons are reported at
150 turns too.

## 4. Primary comparisons and decision criteria (fixed)

| id | comparison | criterion for a candidate |
|---|---|---|
| **P1** consequential leads | the lead premium: the leader's G50 (policy against policy) in the +8 cell minus the equal cell, mid and late stages, per arm | **D1**: ≥ 1.5 × the baseline premium in both stages, with the cross-arm difference's interval excluding 0 |
| P2 conversion | threaten − Hold only (strong focal) at +16, mid and late | reported; expected to grow with consequential leads |
| **P3** not decisive | the early-stage premium (+8) and S6: the share of present turns with an attack ratio ≥ 1.25; the persistence slope | **D2**: early premium ≤ 1.5 × baseline; share ≥ 1.25 at most 5%; slope ≤ 1.2 |
| **P4** the equal lane | S6 lane sanity, against the baseline arm | **D3**: crashed HP and plates by turn 150, deaths, Recalls and collection within ±25%, Shove share within ±0.10 |
| **P5** waves and towers | S1 / S6: C at stage-typical states, Pressure per turn at mid stage | **D4**: C within ±2 HP of baseline at S ≤ 116; mid-stage Pressure within ±25% |
| P6 trailing responses | S3 weak focal: the best response's value against the policy's at +16, mid and late | reported; flagged if under a candidate the trailing side's best response loses more than 1.5× what it loses under the baseline |
| P7 purchases | S5: the shopping component and Recall − stay at full HP | reported; a positive shopping component means purchases matter; Recall − stay above −0.2 means the reset becomes nearly free (flag) |
| P8 freezes | S2 / S4: P(guard holds), the breaker's and freezer's best plans | reported |

**Recommendation rule.** A candidate advances to a matched PPO pilot only if D1–D4 all hold; P6–P8 qualify the
recommendation. If no candidate passes, the report recommends retaining the current curve or investigating the
specific failed criterion. Exploratory follow-ups are labelled. No further candidate is constructed after outcomes are
seen.
