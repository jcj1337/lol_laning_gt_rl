# M4.2d-D — Continuous investment and observation readiness

Branch `prototype/m4-investment-readiness` (from `readiness/m4-2d-wave`, f504f5e). This is mostly an analytic
study. **Nothing changes by default.** `GameConfig.m4()` (two tiers) and the M4.2c `GameConfig.m4_continuous()`
defaults (power curve α = 0.045, p = 0.9; linear observation entries) are reproduced bit for bit, and tests check
this. The proposal is opt-in:

    GameConfig.m4_continuous(invest_curve="saturating", invest_gain=1.5, invest_scale=16.0,
                             invest_obs_encoding="compressed", invest_obs_ref=64, invest_obs_gold_ref=8)

(`lanegame.experiments.m4.investment_readiness.READY` holds these arguments.)

Results: `results/m4_investment_readiness/` (`trajectory_power.json`, `trajectory_ready.json`, `curves.json`,
`observation.json`, `budget.json`). Driver: `python -m lanegame.experiments.m4.investment_readiness --stage
{trajectory,curves,observation}`.

## Summary

- **Why the current curve is unsafe.** Investment scales only offence. A champion's own investment therefore does
  not protect it, and what decides lethality is the attacker's absolute multiplier M.
  - With M ≥ 3, a 2-HP hit kills a full-HP champion of the same level. With M ≥ 6, a 1-HP hit does.
  - Under p = 0.9, M = 3 arrives at S ≈ 68 resources. In scripted play that is reached at turn ≈ 250–450, and
    M = 6–9 by turn 1,000.
  - In the edge-trader mirror that game had 147 deaths in 1,000 turns.
- **Lower p alone does not fix it.** p = 0.7 / 0.5 (matched at S = 10) still one-shot at S ≈ 106 / 207, which is
  inside W = 1,000. A power curve weak enough to stay below M = 3 through W (p = 0.5, α = 0.073) halves the early
  value of gold (×1.09 at S = 4, against ×1.20 for the first tier). It still one-shots at S ≈ 800.
- **Primary recommendation: a bounded sqrt-saturating curve, M(S) = 1 + 1.5 (1 − √(16 / (16 + S)))**
  (`invest_curve="saturating"`).
  - It matches the M4.2c curve through the early game: ×1.045 / 1.16 / 1.32 at S = 1 / 4 / 10, against
    1.039 / 1.147 / 1.344.
  - It keeps growing through the whole window: ×1.94 at S = 100, ×2.16 at 300, ×2.31 at 1,000.
  - It never reaches 2.5. So no hit ever one-shots a full-HP champion of the same level (or up to three levels
    lower), and 1-HP hits always need at least 3 hits.
  - Its three hits-to-kill steps (1-HP hits: 6→5 at S ≈ 5, 5→4 at S ≈ 20, 4→3 at S ≈ 128) fall into the early,
    middle and late window. M4.2c's five steps all fall before turn ≈ 600 and end in one-shots.
  - It is computed in exact integers (`math.isqrt`). It is platform independent and strictly increasing per ledger
    unit over the whole supported range.
- **The trade-off is real and documented, not hidden.** With offence-only scaling and an absolute safety bound, a
  late gold lead buys little:
  - 300 vs 150 resources is a ×1.06 damage ratio (M4.2c: ×1.70, with one-shots).
  - Gold leads matter early: ×1.13–1.16 for 4 → 10 → 20 → 40.
  - They also matter around the S ≈ 128 breakpoint: 150 vs 100 moves the L18 duel from 0.42 / 0.42 to
    0.61 / 0.26.
  - Making late leads large would need either a convex curve, which contradicts diminishing returns and early value,
    or a survivability counterpart (§5). I argue the counterpart is **not necessary** for initial PPO training.
- **Observation: opt-in "compressed" entries, same length (88).** Whole invested / unspent resources are encoded as
  n / (n + R) with half-points R_S = 64 and R_G = 8. The encoding:
  - never saturates (an entry stays < 1 up to ≥ 2.7 × 10^8 resources);
  - decodes exactly in float32 up to 23,106 invested and 8,184 unspent resources (verified by enumeration). That is
    above the theoretical W = 1,000 maximum of 9,575 for invested gold, and 70× the observed maximum;
  - resolves the Recall-relevant range: 4 unspent gold is 0.33 instead of 0.004, which is 85× more signal;
  - leaves the sub-resource remainders and the bounded attack-power entry unchanged.
- **Budget:** 4,000 experimental turns (3 × 1,000 under the M4.2c curve, 1 × 1,000 under the primary) plus
  ≈ 1,500 turns in test runs.

## 1. Gold accumulation in play (actual numbers, not 1–10 gold)

`--stage trajectory`: one 1,000-turn game per matchup under the M4.2c curve. The edge mirror was repeated under the
primary curve. Values are cumulative earned resources, A / B. Levels are both champions' levels. Investment equals
earned minus unspent, and unspent gold stays small.

| turn | edge_trader mirror | level | aggressive_trader – hold_farmer | level | gold_recaller mirror | level |
|---:|---|---|---|---|---|---|
| 50 | 8.3 / 9.0 | 9 / 9 | 11.8 / 9.9 | 10 / 9 | 16.1 / 13.7 | 10 / 9 |
| 100 | 18.3 / 14.6 | 14 / 13 | 23.1 / 17.5 | 14 / 13 | 31.8 / 28.2 | 15 / 14 |
| 200 | 38.0 / 39.8 | 18 / 18 | 50.4 / 31.6 | 18 / 18 | 60.1 / 58.8 | 18 / 18 |
| 500 | 95.8 / 111.9 | 18 / 18 | 139.3 / 76.5 | 18 / 18 | 150.3 / 148.4 | 18 / 18 |
| 1000 | 194.2 / 255.3 | 18 / 18 | 328.4 / 147.8 | 18 / 18 | 299.0 / 296.1 | 18 / 18 |
| edge mirror, primary curve, t = 1000 | 216.0 / 200.0 | | | | | |

Other per-game statistics:

| matchup | deaths A / B | investments A / B | mean investment | max unspent | level gap mean / max |
|---|---|---|---|---|---|
| edge mirror (M4.2c curve) | 91 / 56 | 103 / 65 | 1.9 / 3.6 | 5.9 / 19.5 | 0.08 / 1 |
| edge mirror (primary curve) | 52 / 71 | 63 / 88 | 3.4 / 2.2 | 11.2 / 6.3 | 0.11 / 2 |
| aggressive – hold (M4.2c curve) | 14 / 141 | 32 / 151 | 9.7 / 1.0 | 32.6 / 3.3 | 0.28 / 3 |
| gold_recaller mirror (M4.2c curve) | 1 / 2 | 38 / 40 | 7.9 / 7.2 | 23.6 / 29.3 | 0.07 / 1 |

What this means for the curve:

- **Scale.** Income is 0.15–0.33 resources per turn, and roughly constant in time. S ≈ 6–16 at turn 50, 13–32 at
  100, 30–60 at 200, 75–150 at 500, and 150–330 at 1,000. A better farmer than these scripts might reach ≈ 500 in W.
  The hard per-turn bound is 9.575 (§7).
- **Levels cap at 18 by turn ≈ 200.** Level gaps are small (max 3, and only before the cap). From turn ≈ 200 on, the
  only combat asymmetry is investment, so the late-game multiplier is just M(S_A) against M(S_B).
- **Investment events are small and frequent:** 1–10 resources every ≈ 6–30 turns. Unspent gold peaks at
  20–33 resources. This sets the observation scale for unspent gold (§8).

## 2. The current model (M4.2c power curve, α = 0.045, p = 0.9)

M(S) = 1 + α((1 + S)^p − 1) multiplies attack power only. HP H(L) and AP(L, 0) both grow by 1.05^L, so
hits-to-kill between equal levels depend on M alone, up to table rounding. A 1-HP hit is H/6 at M = 1, and a 2-HP
hit is H/3.

| quantity | value (exact integer evaluation, `curves.json`) |
|---|---|
| M at S = 50 / 100 / 300 / 1,000 | 2.50 / 3.82 / 8.61 / 23.5 |
| M reached in the trajectories at turn 500 / 1,000 | 3.2–5.1 / 5.0–8.8 |
| 1-HP hits to kill (L18): 7→6→5→4→3→2→1 | S = 0.03, 5.5, 14.9, 32.0, 68.4, 188.3 |
| 2-HP one-shot of a same-level full-HP champion | S ≈ 68.4 at every level; 57.9 with a 2-level gap, 48.6 with 4 |
| relative advantage 300 vs 150 / 500 vs 300 (L18) | ×1.70 / ×1.52 (both sides already one-shot with 2-HP hits) |

Within W the duel turns into "first landed hit wins": with both champions one-shotting, the L18 duel is
0.31 / 0.31 / 0.38 (same turn) for any investment pair. Gank and tower damage are not scaled, so trade damage
grows ×6–9 relative to every other HP source. Under M4.2c the curve is unbounded; it is not a training-ready model.

## 3. Candidate curves (analytic)

All candidates are evaluated like the implementation: M in fixed point, AP* = AP(L, 0) · M, and damage rounded to
centi-HP. "Matched" means M(10) = 1.344, the M4.2c value. sqrt-sat means M = 1 + A(1 − (1 + S/K)^−½), and hyperbolic
means M = 1 + A S / (S + K). The M4.2c `shapes.json` p = 0.7 / 0.5 / log candidates are included.

**Power multiplier M(S)**

| curve | S=0 | 1 | 4 | 7 | 10 | 20 | 50 | 100 | 200 | 300 | 500 | 1000 | bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| power p0.9 α0.045 (M4.2c) | 1 | 1.039 | 1.147 | 1.247 | 1.344 | 1.652 | 2.504 | 3.820 | 6.277 | 8.610 | 13.06 | 23.53 | none |
| power p0.7 α0.079 | 1 | 1.049 | 1.165 | 1.260 | 1.344 | 1.587 | 2.160 | 2.919 | 4.156 | 5.213 | 7.052 | 10.87 | none |
| power p0.5 α0.149 | 1 | 1.062 | 1.184 | 1.272 | 1.345 | 1.534 | 1.915 | 2.348 | 2.963 | 3.436 | 4.186 | 5.565 | none |
| power p0.5 α0.073 (weak early) | 1 | 1.030 | 1.090 | 1.133 | 1.169 | 1.262 | 1.448 | 1.661 | 1.962 | 2.194 | 2.561 | 3.237 | none |
| log α0.144 | 1 | 1.100 | 1.232 | 1.299 | 1.345 | 1.438 | 1.566 | 1.665 | 1.764 | 1.822 | 1.895 | 1.995 | none (3 at 1.1·10^6) |
| hyperbolic A1.5 K33.6 | 1 | 1.043 | 1.160 | 1.259 | 1.344 | 1.560 | 1.897 | 2.123 | 2.284 | 2.349 | 2.406 | 2.451 | 2.5 |
| sqrt-sat A1.25 K12 | 1 | 1.049 | 1.167 | 1.257 | 1.327 | 1.485 | 1.700 | 1.841 | 1.953 | 2.005 | 2.059 | 2.114 | 2.25 |
| **sqrt-sat A1.5 K16 (primary)** | 1 | 1.045 | 1.158 | 1.249 | 1.323 | 1.500 | 1.761 | 1.943 | 2.092 | 2.162 | 2.236 | 2.312 | 2.5 |
| sqrt-sat A1.75 K24 | 1 | 1.035 | 1.130 | 1.210 | 1.280 | 1.458 | 1.753 | 1.980 | 2.177 | 2.274 | 2.375 | 2.482 | 2.75 |
| two tiers (reference) | 1 | 1 | 1.20 | 1.20 | 1.44 | 1.44 | 1.44 | 1.44 | 1.44 | 1.44 | 1.44 | 1.44 | 1.44 |

**Value of later gold: relative power gain of one 4-resource return at investment S (%)**

| curve | S=0 | 10 | 20 | 50 | 100 | 200 | 300 | 500 | share of M(300)−1 gained after S = 100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| power p0.9 (M4.2c) | 14.7 | 9.3 | 7.2 | 4.4 | 2.7 | 1.5 | 1.1 | 0.7 | 0.63 |
| power p0.5 α0.149 | 18.4 | 6.2 | 4.1 | 2.1 | 1.3 | 0.7 | 0.5 | 0.3 | 0.45 |
| power p0.5 α0.073 | 9.0 | 3.5 | 2.4 | 1.4 | 0.9 | 0.5 | 0.4 | 0.3 | 0.45 |
| log | 23.2 | 3.3 | 1.8 | 0.7 | 0.3 | 0.2 | 0.1 | 0.1 | 0.19 |
| hyperbolic A1.5 | 16.0 | 7.2 | 4.2 | 1.5 | 0.5 | 0.2 | 0.1 | 0.03 | 0.17 |
| sqrt-sat A1.25 K12 | 16.8 | 5.6 | 3.0 | 1.0 | 0.4 | 0.1 | 0.1 | 0.04 | 0.16 |
| **sqrt-sat A1.5 K16** | 15.8 | 6.1 | 3.4 | 1.2 | 0.5 | 0.2 | 0.1 | 0.05 | 0.19 |
| sqrt-sat A1.75 K24 | 13.0 | 6.2 | 3.8 | 1.5 | 0.6 | 0.2 | 0.1 | 0.06 | 0.23 |

**Lethality** (resources at which it first happens, exact; "–" = never within 10^6 resources)

| curve | 2-HP one-shot, equal level | … attacker +2 levels | … +4 levels | 1-HP two-shot (L18) | M ≥ 2 at S |
|---|---:|---:|---:|---:|---:|
| power p0.9 (M4.2c) | 68.4 | 57.9 | 48.6 | 68.4 | 31.9 |
| power p0.7 α0.079 | 105.8 | 85.9 | 69.0 | 105.8 | 40.9 |
| power p0.5 α0.149 | 206.7 | 156.1 | 116.5 | 206.9 | 58.5 |
| power p0.5 α0.073 | 804 | 601 | 444 | 805 | 215 |
| log α0.144 | – | 152,700 | 26,400 | – | 1,036 |
| hyperbolic A1.5 K33.6 | – | – | 1,451 | – | 67.2 |
| sqrt-sat A1.25 K12 | – | – | – | – | 288 |
| **sqrt-sat A1.5 K16** | – | – | 31,200 | – | 128 |
| sqrt-sat A1.75 K24 | – | 75,600 | 888 | – | 107 |

(The "+4 levels" column is the worst case. Level gaps of 4 did not occur in play, and they only arise before the
level cap, when S is still small.)

**Hits-to-kill steps against a same-level full-HP champion, level 18** (S at which each step happens; in brackets,
the extra investment that moves that hit by one more centi-HP there). At L18 the base is 7 one-HP hits
(1,375 / 229 = 6.004) and 4 two-HP hits (1,375 / 458 = 3.002), so the first step of each is the table-rounding
knife edge (§6). Levels 1, 6 and 11 (base 6 / 3, apart from the L6 2-HP knife edge) are in `curves.json`; the
non-knife-edge breakpoints are level-independent to within ±0.4 resources.

| curve | 1-HP hit: 7→6 | 6→5 | 5→4 | 4→3 | 3→2 | 2→1 | 2-HP hit: 4→3 | 3→2 | 2→1 |
|---|---|---|---|---|---|---|---|---|---|
| power p0.9 (M4.2c) | 0.032 (0.11) | 5.50 (0.13) | 14.9 (0.14) | 32.0 (0.15) | 68.4 (0.16) | 188 (0.18) | 0.005 (0.05) | 15.0 (0.07) | 68.4 (0.08) |
| power p0.5 α0.149 | 0.018 (0.06) | 4.41 (0.14) | 17.9 (0.26) | 58.5 (0.45) | 207 (0.85) | – | 0.003 (0.03) | 18.0 (0.13) | 207 (0.42) |
| log | 0.009 (0.03) | 2.95 (0.12) | 30.9 (0.98) | – | – | – | 0.002 (0.02) | 31.2 (0.49) | – |
| **sqrt-sat A1.5 K16** | 0.028 (0.09) | 5.22 (0.14) | 19.9 (0.31) | 128 (2.6) | – | – | 0.005 (0.05) | 20.0 (0.16) | – |
| sqrt-sat A1.75 K24 | 0.036 (0.12) | 6.50 (0.17) | 22.9 (0.33) | 107 (1.5) | – | – | 0.006 (0.06) | 23.0 (0.16) | – |

**Relative advantage and duel odds at level 18** (M ratio A/B; 1-HP hits to kill for A / B; P(A kills first) /
P(B kills first) in the M2 trade duel, hit 0.55, double 0.30)

| S_A vs S_B | M4.2c p0.9 | log | sqrt-sat A1.5 K16 (primary) | sqrt-sat A1.75 K24 |
|---|---|---|---|---|
| 4 vs 0 | 1.15; 6/7; 0.58 / 0.33 | 1.23; 5/7; 0.72 / 0.21 | 1.16; 6/7; 0.58 / 0.33 | 1.13; 6/7; 0.58 / 0.33 |
| 10 vs 4 | 1.17; 5/6; 0.59 / 0.31 | 1.09; 5/5; 0.44 / 0.44 | 1.14; 5/6; 0.59 / 0.31 | 1.13; 5/6; 0.59 / 0.31 |
| 20 vs 10 | 1.23; 4/5; 0.60 / 0.29 | 1.07; 5/5; 0.44 / 0.44 | 1.13; 4/5; 0.60 / 0.29 | 1.14; 5/5; 0.44 / 0.44 |
| 40 vs 20 | 1.35; 3/4; 0.61 / 0.26 | 1.07; 4/5; 0.60 / 0.29 | 1.13; 4/4; 0.44 / 0.44 | 1.15; 4/5; 0.60 / 0.29 |
| 100 vs 50 | 1.53; 2/3; 0.64 / 0.22 | 1.06; 4/4; 0.44 / 0.44 | 1.10; 4/4; 0.44 / 0.44 | 1.13; 4/4; 0.44 / 0.44 |
| 150 vs 100 | 1.33; 2/2 (2-HP one-shots both); 0.41 / 0.41 | 1.04; 4/4 | 1.05; 3/4; 0.61 / 0.26 | 1.06; 3/4; 0.61 / 0.26 |
| 300 vs 150 | 1.70; 1/2; 0.64 / 0.16 | 1.06; 4/4 | 1.06; 3/3; 0.42 / 0.42 | 1.08; 3/3; 0.42 / 0.42 |
| 500 vs 300 | 1.52; 1/1; 0.31 / 0.31 | 1.04; 4/4 | 1.03; 3/3; 0.42 / 0.42 | 1.05; 3/3; 0.42 / 0.42 |

## 4. Choice and the trade-off

The objectives conflict. Every offence-only curve has to pick a point on this trade-off:

1. **Early value.** The first returns (S ≈ 4–10) should matter about as much as a tier: ×1.15–1.34.
2. **No one-shots in ordinary windows.** M < 3 must hold through W at least, and ideally everywhere.
3. **Late gold is not worthless.** Power should keep growing through S ≈ 100–500.
4. **Mild diminishing returns.**

Objectives 1 and 2 fix M(10) ≈ 1.3 and M(500) < 3, so the curve must be strongly concave. Objectives 3 and 4 then
cannot both be strong. Under any curve that respects 2, a late lead is worth at most M_max / M(S_B) (for example
2.5 / 2.03 = 1.23 against S_B = 150).

- **Power curves.** At matched early strength they violate 2 (one-shot at S = 68–207). Made safe for W they violate 1
  (p 0.5 α 0.073 is ×1.09 at S = 4) and still one-shot at S ≈ 800, which is reachable in long continuing-time runs.
  Rejected.
- **log.** It satisfies 1 and, in practice, 2 (M = 3 only at ≈ 10^6 resources). But it front-loads the most (+23% for
  the first 4 resources) and leaves only one hits-to-kill step after S = 3 (at S ≈ 31). It has no strict bound, and
  with float evaluation at 10^12 its per-ledger-unit increment falls below 10 fixed-point units beyond ≈ 14k
  resources. Secondary candidate,
  not implemented.
- **Hyperbolic (q = 1).** Bounded, but the tail decays as S^−2: +0.03% per return at S = 500. Later gold is worth the
  least of all candidates, and +4 levels one-shot at S ≈ 1,450. Rejected.
- **sqrt-saturating (q = ½), A = 1.5, K = 16: primary.**
  - Early curve equal to M4.2c (|ΔM| ≤ 0.03 up to S = 10). The early-game findings of M4.2c (recall timing, a lead
    against a passive opponent) carry over.
  - Strict bound 2.5. With the 1-HP / 2-HP hit structure and level gaps ≤ 3, no full-HP one-shot is possible at any S
    (tested exhaustively over all levels at the guard).
  - The tail decays as S^−1.5. That is the heaviest tail among the bounded candidates at this early strength.
  - The late breakpoint (4 → 3 hits at S ≈ 128, turn ≈ 400–850 in play) sits inside W, so late gold still changes a
    combat outcome once.
  - In the edge mirror, deaths fell from 147 to 123 per 1,000 turns. One game per curve, so this is indicative only.
- **sqrt-sat A = 1.75, K = 24: alternative** if late leverage proves too weak. +10–30% more late leverage (300 vs 150:
  ×1.08). Bound 2.75. A 2-HP one-shot needs a 4-level gap at S ≥ 888, or a 2-level gap at S ≥ 75,600. The early game
  is slightly weaker (×1.13 at S = 4).
- **sqrt-sat A = 1.25, K = 12: conservative.** Bound 2.25, but the late step (S ≈ 289) sits at the edge of W.

## 5. Is a survivability counterpart necessary?

**Not for the stated goal.** The danger comes from absolute offence (M ≥ 3), not from relative advantage. A bound
M_max = 2.5 removes it completely:

- 1-HP hits always need ≥ 3 hits from full HP;
- 2-HP hits need ≥ 2, up to a 3-level gap.

A counterpart (for example "damage taken divided by M_def(S_victim)") would buy a different thing: late leads that
stay large in ratio terms, while absolute lethality between similar champions stays fixed. That changes the
semantics of combat damage. Today hit damage is a function of the attacker only (`M4Context.hit(base, level,
items)`), and it is shared by the duel edge of the scripted agents. A counterpart would also make trades relatively
weaker than the unscaled gank and tower damage. Implementing it means a new two-sided damage function and changes
to `agents_m4` and the invariants. It is not needed while the bounded curve meets the safety goal, so I did not
implement it. It is the natural next step only if PPO runs show that late gold leads are strategically inert.

## 6. The proposed scaling: implementation and numerical properties

    M(S) = 1 + A (1 − √(K / (K + S))),   A = invest_gain = 1.5,  K = invest_scale = 16 resources

Integer form (`progression.Investment`, curve "saturating"), with k = K · gold_scale ledger units, u = invested
ledger units and ONE = `INVEST_ONE_SAT` = 10^24:

    M(u) = ONE + ⌊A · (ONE − isqrt(⌊ONE² k / (k + u)⌋))⌋        (A as an exact rational)
    AP*  = AP(L, 0) · M(u);   dmg* = ⌊(D·100·AP* + 5000·ONE) / (10^4·ONE)⌋;   P* = ⌊(400·AP* + 5000·ONE) / (10^4·ONE)⌋

Properties:

- **No floating point.** `isqrt(⌊x⌋) = ⌊√x⌋` exactly, so M is the exact floor of the real formula. It is
  platform independent; the M4.2c power curve uses `float ** p`, whose last bits depend on libm. It is also
  non-decreasing by construction. Tested against an independent 60-digit decimal evaluation.
- **Strictly increasing per ledger unit** on the whole supported range u ≤ 10^6 resources (`INVEST_MAX_RESOURCES`,
  unchanged). The decrement of ONE·√(k/(k+u)) per unit is ONE·√k / (2(k+u)^1.5) ≥ 5·10^8 for k ≥ 10^6 units (K ≥ 1,
  which the validation enforces). Beyond the guard `Investment` raises, as in M4.2c.
- **Neutral at zero.** M(0) = ONE exactly, so the hit damage and Pressure damage of a champion with no investment
  equal the tier-0 tables (tested).
- **Bounded.** M(10^6 resources) = 2.494. The maximum hit damage is 249 / 499 centi-HP at L1 (H = 600) and
  572 / 1,143 at L18 (H = 1,375).
- **Pressure.** At most 998 (L1) / 2,286 (L18) centi-units per effective turn, against plate thresholds
  1,600 / 3,600 / 6,000. `per_turn_bound` already allows every plate under "continuous" (9.575 resources) and stays
  valid. A bounded curve would allow a tighter bound; I did not change it.
- **Every return raises power.** 1 resource raises M by > 0.02% up to S = 300, and AP* rises for every ledger unit.
  Rounded damage follows within 0.1–0.3 resources early, and within 2.6 (L18) to 5.8 (L1) resources around
  S ≈ 128.
- **Unchanged mechanics.** Automatic spending of all gold at every return, permanent cumulative investment, no tiers
  and no thresholds are untouched (same code path in `champions_m4.resolve_turn_m4`).

**Quantization knife edges are not a property of the curve.**

- At levels where the base hits-to-kill sits just past an integer, any positive investment crosses a breakpoint
  almost immediately:
  - L18: 1,375 / 229 = 6.004 hits; the 6→5 step comes at S = 0.028 under the primary;
  - L6: 2-HP hit 4→3 at S = 0.020.
- This is a rounding artefact of the H(L) and AP(L, 0) tables, as found in M4.2c. "Positive benefit from every
  investment" makes it unavoidable for any curve with M′(0) > 0.
- What the curve controls is the number and placement of the other steps. The primary has three steps per hit size
  spread over W (S ≈ 5 / 20 / 128) instead of five, and none of them ends in a one-shot.
- Fixing the tables (H(L) or u(L) rounding) is outside this workstream (no combat changes).

## 7. Level-growth interaction

H(L) and AP(L, 0) both grow by 1.05^L, so a g-level gap multiplies the effective attacker power by ≈ 1.05^g. Primary
curve, hits to kill a full-HP victim (1-HP / 2-HP hits; `curves.json` → `level_gap_primary`):

| victim level | gap | S = 0 | S = 20 | S = 60 | S = 300 | S = 10^5 |
|---|---:|---|---|---|---|---|
| 11 | 0 | 6 / 3 | 5 / 2 | 4 / 2 | 3 / 2 | 3 / 2 |
| 11 | 2 | 6 / 3 | 4 / 2 | 4 / 2 | 3 / 2 | 3 / 2 |
| 11 | 4 | 5 / 3 | 4 / 2 | 3 / 2 | 3 / 2 | 2 / 1 |
| 6 | 0 | 6 / 4 | 5 / 2 | 4 / 2 | 3 / 2 | 3 / 2 |
| 14 | 4 | 5 / 3 | 4 / 2 | 3 / 2 | 3 / 2 | 2 / 1 |

A 4-level gap yields a one-shot only at S ≈ 31,000. In play, levels cap at 18 by turn ≈ 200, when S ≈ 30–60, and the
largest observed gap was 3. Level growth and investment do not compound into one-shots in any reachable regime.

## 8. Observation checks

**M4.2c encoding (default, unchanged).** The five item entries are:

- invested_whole = min(⌊S⌋, R_S) / R_S and gold_whole = min(⌊G⌋, R_G) / R_G, with R_S = R_G = 1,024;
- the two sub-resource remainders;
- attack_power = AP* / (AP* + AP(18, 0)).

Issues for training:

- **Normalisation.** Unspent gold, the quantity that matters for a Recall decision, is ≤ 33 in play, so the entry is
  ≤ 0.03. One resource moves it by 1/1,024. A network effectively sees nothing. Invested gold reaches only 0.3 by the
  end of W.
- **Saturation and losslessness.** Exact only below 1,024 resources. Beyond that the entry clips to 1.0, the
  observation is no longer injective (`ProbCache` keys could merge distinct states) and `decode_m4` raises. This is
  outside W for scripted play (≤ 330), but not outside the theoretical per-turn bound (9.575 × 1,000).

**Opt-in "compressed" encoding** (`invest_obs_encoding="compressed"`; the length stays **88** and the feature names
are unchanged):

    invested_whole = n_S / (n_S + R_S),  gold_whole = n_G / (n_G + R_G)   (R = half-point; recommended R_S = 64, R_G = 8)

The remainders and attack_power are unchanged. The decoder is n = round(R·x / (1 − x)). The exact range is
n ≤ `compressed_exact_max(R)` = isqrt(R·2^23) − R. Over that range the float32 rounding error times dn/dx stays
≤ 1/4; I verified this by enumerating every n, and the first failure is at ≈ 1.4× the bound.

| R | exact up to (resources) | first decoding failure | entry at 1 / 4 / 8 / 16 / 64 / 300 / 10^6 | one resource moves the entry by (at n = 0 / 16 / 300) | entry reaches 1.0 in float32 at |
|---:|---:|---:|---|---|---:|
| 8 (unspent) | 8,184 | 11,594 | 0.111 / 0.333 / 0.5 / 0.667 / 0.889 / 0.974 / 0.99999 | 0.111 / 0.0133 / 8.4·10^-5 | 2.7·10^8 |
| 64 (invested) | 23,106 | 32,832 | 0.015 / 0.059 / 0.111 / 0.2 / 0.5 / 0.824 / 0.99994 | 0.0154 / 0.0099 / 4.8·10^-4 | 2.1·10^9 |
| 1,024 (linear, M4.2c) | 1,023 | 1,024 (saturated) | 0.001 / 0.004 / 0.008 / 0.016 / 0.063 / 0.293 / 1.0 | 0.00098 everywhere | 1,024 |

- **Bounded, monotone, no saturation.** The entry is < 1 for every reachable value: above the guard of 10^6
  invested resources, and above any unspent amount below 2.7 × 10^8.
- **Exactness where replay needs it.** Exact over 23k invested and 8k unspent resources. That covers W with margin
  for invested gold, even at the theoretical maximum income, and covers any unspent amount a champion that ever goes
  back to base can hold. Beyond those bounds the observation is valid but `decode_m4` raises. The bound is explicit
  and tested.
- **Replay, rewards, determinism.**
  - Replays and traces store the integer state, not observations, so no observation choice affects them.
  - Rewards are ledger differences; investing is not a reward event (tested).
  - The integer curve makes damage platform-independent.
  - A 250-turn game played twice under the recommendation gives identical trajectories, and every observation in it
    decodes to the exact state.
- **NN scale.** In W: invested_whole spans 0.1 (turn 50) → 0.3–0.5 (turn 200) → 0.7–0.84 (turn 1,000), and
  gold_whole spans 0 → 0.33 (4 gold) → 0.8 (32 gold). The attack-power entry is a smooth, bounded measure of
  combat strength: under the primary curve at L18 it runs 0.5 → 0.714.
- **Remaining weakness.** The two sub-resource remainders are sawtooth features with no strategic content. They exist
  only so that `decode_m4` and `ProbCache` stay exact. Dropping or masking them for the policy network would be a
  model-side choice; it would change the observation contract, so I did not do it.
- **Caveat.** Setting `invest_obs_encoding="compressed"` without changing the references keeps R = 1,024 as the
  half-point. That is valid and exact up to 91,657 resources, but poorly scaled. Pass R_S = 64 and R_G = 8 as in
  `READY`.

## 9. Changes

| file | change |
|---|---|
| `lanegame/config.py` | New M4 economy fields, opt-in under `item_model="continuous"`: `invest_curve` (`"power"` default / `"saturating"`), `invest_gain` (1.5), `invest_scale` (16.0 resources), `invest_obs_encoding` (`"linear"` default / `"compressed"`). Validation: each curve's parameters must stay at their defaults under the other curve; all four must stay at their defaults under tiers, legacy and m2c; `0.01 ≤ invest_gain ≤ 10`; `1 ≤ invest_scale ≤ 10^6`; `invest_scale` must be a whole number of ledger units. Added to `M4_ECONOMY_FIELDS`. |
| `lanegame/progression.py` | `Investment(…, curve, gain, scale_res)` with the exact-integer saturating branch; per-instance fixed-point unit `one` (`INVEST_ONE` = 10^12 for "power", unchanged; new `INVEST_ONE_SAT` = 10^24); `hit_damage` / `pressure_damage` / `factor` use `one`; `_investment` / `investment(cfg)` pass the curve. The power path is the same arithmetic as before. |
| `lanegame/env/observation_m4.py` | `compressed_exact_max(ref)`; compressed branch in `observe_m4` and `decode_m4`; the attack-power reference uses `inv.one`, which is identical for "power". |
| `lanegame/experiments/m4/investment_readiness.py` | New driver (trajectory / curves / observation stages, budget ledger, `READY`). |
| `tests/test_m4_investment_readiness.py` | New, 23 tests. |
| `tests/test_m4_config.py` | Field table extended with the four new fields. |

Functions touched: `GameConfig.validate`, `GameConfig._validate_exact_economy`, `Investment.__init__ / multiplier /
factor / attack_power / hit_damage / pressure_damage`, `_investment`, `investment`, `observe_m4`, `decode_m4`
(+ new `compressed_exact_max`). Untouched: wave mechanics, positioning, zoning, agents, PPO, `champions_m4`,
`discounted.per_turn_bound`.

## 10. Tests and budget

- `pytest -q tests/test_m4_investment_readiness.py tests/test_m4_continuous_investment.py tests/test_m4_observation.py
  tests/test_m4_progression.py tests/test_m4_config.py`: **98 passed, 2 skipped**. The new file has 23 tests.
- Regression: `tests/test_legacy_pins.py tests/test_m4_economy.py`: 18 passed, 2 skipped.
- The new tests cover:
  - defaults unchanged and the power curve bit-for-bit;
  - validation of foreign and invalid fields;
  - the saturating curve: neutral at zero, the formula, independent decimal exactness, strict per-unit
    monotonicity, the bound, no one-shot at any level or gap ≤ 3, late-return value;
  - gameplay under the recommendation: invest-all on return, spending-neutral rewards, exact ledger, zero sum,
    deterministic replay, and every in-play observation decoding to the state;
  - the compressed encoding: float32 exactness by enumeration, round trips beyond 1,024, no saturation at the guard,
    decoder refusal beyond the bound, small-gold resolution, and the bounded power entry.

| item | simulated turns |
|---|---:|
| trajectory, M4.2c curve (3 matchups × 1,000) | 3,000 |
| trajectory, primary curve (edge mirror × 1,000) | 1,000 |
| curves, observation stages (analytic) | 0 |
| **experiments total** (`results/m4_investment_readiness/budget.json`) | **4,000** |
| new test file, three runs (2 × 250-turn game each) | ≈ 1,500 |

The total is about 5,500, under the 15,000 hard cap. The 5,000 target counts the experiments only.

## 11. Recommendation

1. **For initial PPO training with continuous investment, use**
   `GameConfig.m4_continuous(invest_curve="saturating", invest_gain=1.5, invest_scale=16.0,
   invest_obs_encoding="compressed", invest_obs_ref=64, invest_obs_gold_ref=8)`.
   It is opt-in. The two-tier `m4()` remains the validated default, and the M4.2c defaults are unchanged.
2. Alternatives, in order:
   - `invest_gain=1.75, invest_scale=24.0`, if late leverage turns out too weak (bound 2.75; unsafe only with
     ≥ 4-level gaps at S ≥ 888);
   - a log curve (unbounded, but practically ≤ 2.0 in W; not implemented);
   - a survivability counterpart (§5), only if late gold leads prove strategically inert in training.

Caveats:

- The late-game value of a gold lead is small by design: ×1.03–1.06 for 1.5–2× leads above S = 150, plus one late
  breakpoint at S ≈ 128.
- The duel knife edges at levels whose base hits-to-kill sit just past an integer (L18, 6.004) remain. They are a
  table-rounding issue for a separate review.
- The trajectory figures come from one game per matchup with scripted agents. PPO agents may farm faster (up to
  ≈ 500 resources in W), but every statement here holds up to the guard of 10^6 resources.
- The theoretical maximum income (9.575 per turn) can exceed the compressed exact range for invested gold after
  ≈ 2,400 turns. That income is unreachable in practice. If it were reached, the observation stays valid and only
  `decode_m4` refuses.
- I did not rerun the M4.2c recall or response rollouts under the new curve (budget). The early-game equivalence
  (|ΔM| ≤ 0.03 up to S = 10) suggests their conclusions carry over for leads ≤ 14.
