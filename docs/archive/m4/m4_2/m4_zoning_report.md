# M4.2d-C — Combat-Driven Zoning (prototype)

Branch `prototype/m4-zoning` (from `readiness/m4-2d-wave` f504f5e). This is a short prototype study. Agent B is
building a positioning system in parallel, so every experiment here uses the **baseline positioning interface**:
Hold/Yield stand at the farm spot, Safe stands at most at the safe line, and Trade chases the nearest contact cell.
Nothing was trained.

- **Driver:** `python -m lanegame.experiments.m4.zoning`. One run is 4,966 simulation turns, about 1 s.
- **Results:** `results/m4_zoning/zoning.json` (raw) and `results/m4_zoning/zoning.md`.
- **Tests:** `tests/test_m4_zoning.py`.
- **Opt-in rule:** `GameConfig.m4(zoning_rule="none")`. The default `"presence"` keeps M4 exactly as it was.

Positions are in each side's own coordinates: own tower −2, the lane 0, enemy tower +2. "Gold" means resources
(1 resource = one fresh wave of farm). A HP *unit* is 1/6 of a bar, u(L) = H(L)/6.

## Summary

- **One automatic proximity effect exists, and only one** (§1). It is the zone factor in `collection_units`:
  - −30% of a kill's gold (−15% in the collector's own tower cell);
  - it applies whenever the opponent is farming or trading within 1 cell of the kill;
  - it is symmetric, needs no combat and ignores combat strength.

  Every other gold loss comes from reach, the Safe/retreat posture, a connected trade, or absence. XP has no zoning
  at all.
- **The prototype is a single opt-in field,** `zoning_rule ∈ {"presence", "contact", "none"}`:
  - `"presence"` (default) is the current rule, unchanged;
  - `"none"` removes the proximity zone;
  - `"contact"` applies the same factor only to a champion whose opponent's Trade connected on it.

  No power-based farming term is introduced. Power still enters only through trade damage.
- **Removing the proximity zone does not by itself create threat-based zoning.** The test is 162 solved zero-sum
  games:
  - 3 states × 3 rules × 3 power levels × 2 horizons × 3 HP valuations;
  - all from identical states, using exact expectations.

  B's equilibrium never includes Yield or Safe in any of them.
  - **B's best response to A's Trade:** it becomes "concede" (Yield) only once A's attack multiplier reaches
    **m\* ≈ 2.0–3.0** (§4.4). The "large" investment condition (2 tiers) is only ×1.44.
  - **Under the enemy tower:** B's best response to a Trade is already Safe at m\* ≤ 1.22 (1.0 over six turns),
    because there a Trade is a dive and loses for A anyway.
- **Why: conceding is expensive in this geometry, whatever the zoning rule** (§4.4). When B yields at the lane:
  - B drops from the contest base 0.9 to the Safe base 0.4, losing **0.25 per turn**;
  - A's Trade then misses, so A gets back the half of its farm a connected Trade costs (**+0.225**).

  Against that, B avoids only ≈ 0.72·m HP units of trade damage, worth 0.18·m at λ = 0.25. The zoning rule moves
  weight between the two cost terms, but their sum hardly changes: 0.467 under `presence`, 0.475 under `none`.
  B's other contesting option, Trade-back, also beats conceding. This stays true even when trading is made free
  (§4.6).
- **What the rule does change** (§4.3):
  1. **Who trades at the lane.**
     - Under `presence` the zone shrinks the farm a trader gives up, so in the neutral state mutual Trade is the
       pure equilibrium at λ ≥ 0.25 at every power level.
     - Under `none`, both Hold at equal and moderate power. The stronger A starts trading only at large power, and B
       absorbs the damage.
     - So under `none`, Trade appears only where power justifies it. It is also a more passive lane.
  2. **Every contested gold flow rises by 1/0.7 = +43%.** Mutual Hold at the lane pays 0.45 instead of 0.315 per
     turn. Positional gaps grow by the same factor: A's wave advantage in the forward state is worth +0.557 instead of
     +0.355 over 6 turns.
  3. **Pushing to the enemy tower becomes slightly worse** (−2.27 vs −2.17 over 6 turns). The pusher no longer taxes
     the defender's tower farm by standing next to it.
- **`"contact"` is not recommended.** It turns every connected Trade into a fixed 30% gold penalty on the victim.
  The result is a forced trading equilibrium: Trade is in A's support in 34 of 36 lane-state games and is B's only
  equilibrium behaviour in all 36. That is a fixed penalty attached to combat, not a consequence of it.
- **Sampled windows with real HP** (trades land, recalls at 40% HP, deaths; §4.5): a permanently trading A still
  does worse against a contesting B than a holding A.
  - In the neutral state under `none`: −1.04 / −0.52 / −0.13 at equal / moderate / large power, against +0.07 for
    Hold.
  - Conceding is always worse for B: +1.9 to +2.7 for A over 10 turns.
  - The lasting economic cost of losing trades is **XP and tempo through absence**: in the forward state B loses
    6.6 XP per 10 turns at large power, against 2.4 at equal power.
- **Recommendation:** keep the default.
  - **Carry `zoning_rule="none"` into the integrated build as the candidate rule.** It is the smallest viable change:
    it removes the only automatic proximity loss and keeps the reach, posture, trade and absence costs.
  - **Accept it only if the post-integration checks in §6 pass,** chiefly a concession flip at m\* ≈ 1.2–1.5.
  - **Do not adopt it on its own.** With baseline positioning it makes the equal-power lane more passive and adds
    about 43% contested gold without replacing the denial it removes.

---

## 1. Inventory of hardcoded collection and zoning effects

Line numbers are at f504f5e (before this branch). Gold per HP is `farm_per_hp · base · (1 − zone) · t`, with
`farm_per_hp` = 0.25 (`config.py:254`), computed exactly in `champions_m4.collection_units` (`champions_m4.py:150–169`).

### 1.1 Loss from being outside farming reach (positional access and absence)

| Where | Value | Condition |
|---|---|---|
| `champions_m4.py:153` | collects 0 | mode `away` (recalled, absent, dead) or `channel` (interrupted Recall) |
| `champions_m4.py:153`, `config.py:318` | collects 0 | kill more than `farm_reach` = 1 cell from the champion's end cell |
| `champions_m4.py:153`, `config.py:322` | reach 2 | a *threatened* Yield (mode `retreat`) collects within `retreat_farm_reach` = 2 |
| `champions_m4.py:337`, `config.py:316` | target ≤ −1 | Safe stands at most at `safe_line` = −1, so kills at +1/+2 are out of reach |
| `champions_m4.py:361–363` | evasion cell | a Yield/Safe champion threatened by a Trade retreats to the furthest cell out of contact |
| `champions_m4.py:403`, `config.py:371` | XP 0 | XP needs presence and a kill within `xp_range` = 3; no other factor |

### 1.2 Loss from choosing Safe or Yield (posture)

| Where | Value | Condition |
|---|---|---|
| `champions_m4.py:160`, `config.py:257–258, 714` | base 0.8 / 0.6 / 0.4 / 0.2 / 0 at kill cell −2 / −1 / 0 / +1 / +2 | mode `safe` (Safe) or `retreat` (threatened Yield); contest base is 0.9 |
| `champions_m4.py:380–381` | wave contribution ≤ −1 | a Safe/retreating champion contributes shove damage and freeze-break push only up to `min(end, safe_line)` (indirect, through the lane) |

A Yield that is not threatened is Hold (no posture loss). Combat-side terms that are not gold, but shape the choice:
the hit chance changes by −0.25 for Yield and −0.40 for Safe (`trade_hit_behavior`, `config.py:301`), and
`gank_damage_defense` applies.

### 1.3 Loss from connected Trades

| Where | Value | Condition |
|---|---|---|
| `champions_m4.py:164–165`, `config.py:323` | × `trade_farm_factor` = 0.5 | the trader's own collection when its Trade connects |
| `champions_m4.py:376–377` | no wave contribution | a connected trader (and its objective is suspended; it cannot maintain a Freeze) |

The victim of a connected Trade loses no gold directly. It loses HP, which leads to a later Recall or death and so
to absence (§1.1).

### 1.4 Automatic loss from an opponent merely being nearby

| Where | Value | Condition |
|---|---|---|
| `champions_m4.py:161–162`, `config.py:259–260` | × (1 − 0.3), or × (1 − 0.15) for a kill in the collector's own tower cell | the opponent is in mode `farm` or `trading` (Hold, unthreatened Yield, any Trade) and within `farm_reach` = 1 of the kill |
| `champions.py:255` (M2 resolver), `game.py:85–88` (legacy) | the same factor | M2: same condition. Legacy: whenever the opponent's posture is Contest, with no distance condition. Not used by M4; untouched |

This is the only automatic effect:

- **No combat needed:** it requires no combat, no threat and no position relative to the collector, only proximity
  to the kill.
- **Symmetric:** in mutual farming both sides lose 30%.
- **Destroys gold:** the zoned gold is not transferred to anyone.
- **Ignores power:** it does not depend on levels, items, HP or skill.

### 1.5 Not zoning (for completeness)

- **Contest base 0.9 (`config.py:255`):** a 10% last-hit inefficiency for everyone. Under the enemy tower it is 0.5
  (`collect_contest_enemy_tower`): positional and champion-independent.
- **The G_rem clamp (`champions_m4.py:482–485`):** unspendable gold, not zoning.
- **Plate gold and the kill bounty:** neither is zoning.

---

## 2. The prototype rule

```
GameConfig.zoning_rule: str = "presence"     # config.py, next to zone_contest / zone_under_tower; ruleset "m4" only

zones(collector at own cell x, opponent mode o_mode at its own cell o_pos) =
    "presence":  o_mode in {farm, trading} and |-x - o_pos| <= farm_reach          (M2, the default)
    "contact":   o_mode == trading               (the opponent's Trade connected on this collector this turn)
    "none":      False
factor = 1 - zone_contest (0.3), or 1 - zone_under_tower (0.15) at x = -T, when zones; else 1
```

**Implementation.**

- **Resolver:** `champions_m4.collection_units` is the only resolver change. Its signature is unchanged, so the
  memo in `M4Context.collection` keys correctly.
- **Validation:** `GameConfig._validate_ruleset` accepts the three values.
- **Ruleset contract:** `zoning_rule` is appended to `M4_ECONOMY_FIELDS`, so the legacy and M2 presets reject any
  non-default value. The M2 and legacy collection functions are untouched.

**What remains under `"none"`.**

- **Kept:** reach (1, or 2 for a retreating Yield); the Safe/retreat posture base; the 0.5 trade factor; absence;
  the under-tower contest base 0.5; and all combat consequences (HP leading to Recall, death and bounty).
- **Not added:** nothing replaces the zone. A champion's gold depends on its own mode and position and on where the
  kill happened. It never depends on the opponent's attack power.

Tests (`tests/test_m4_zoning.py`, 6 tests):

1. Default and validation.
2. Exhaustive equality of `"presence"` with the M2 rule over every (kill cell, mode, position, opponent mode,
   opponent position).
3. Exhaustive semantics of `"none"` and `"contact"`.
4. "Gold only": from the same state, actions and chance vector the three rules give identical positions, lane, HP,
   XP, trades, ganks, tower damage, plates and bounties over 300 random-legal turns. This test also covers mirror
   equivariance per rule and the M4 invariants.
5. The lane behaviour numbers of §4.1.
6. The exact gold attribution of the driver: the parts sum to the available farm, and the zone part is 0 under
   `"none"`.

---

## 3. Experimental setup

**States.** Each state is reached by deterministic play from s0, then put on common ground: equal XP, full HP, no
gold and no items.

| State | k | Lane (own coords) | Champions | What it represents |
|---|---|---|---|---|
| neutral | 9 | fronts at the lane | A 0, B 0 (L3) | symmetric contact; all kills at the lane |
| forward | 10 | A front 0 (6 HP), B front −1 (4 HP) | A 0, B −1 (L4) | A's wave advantage; B one cell back, already in trade contact |
| tower | 13 | A front +1 (6 HP), B front −2 | A +1, B −2 (L5) | B in its own tower cell; A can reach it only by diving |

The forward state comes from the pushed line of M4.2d-A at k = 10 with one change. B's reached front had 1 HP, about
to die and walk its farm spot back. It is replaced by a 4-HP front, which makes the configuration stationary under
mutual Hold. This is a controlled counterfactual, not a reached state.

**Power.**

- **Equal:** neither side has items.
- **Moderate:** A holds 1 tier (attack power ×1.20).
- **Large:** A holds 2 tiers (×1.44).

Levels are equal and B has no items. In the deterministic stage items change only the damage, so each path is played
once and its damage is re-evaluated per power condition. The driver checks this invariance on every first turn.

**Actions.** Both sides play Slow Push with no objective. Each side's behaviour is one of Hold, Yield, Trade or Safe,
giving a 4 × 4 matrix.

**Deterministic stage.**

- **Outcomes:** each pair is played for 1 turn (h1) and, as a stationary pair, for 6 turns (h6). Chance is off
  (u = None).
- **Expectations:** expected trade, tower and gank damage come from the resolver's own probabilities.
- **Value to A:** V = (earned_A − earned_B) + λ (D_B/u_B − D_A/u_A), with λ ∈ {0.15, **0.25**, 0.35} resources per HP
  unit (the M2 estimate is 0.22–0.30).
- **Solution:** each matrix is solved as a zero-sum game.
- **Flip multiplier m\*:** the smallest attack multiplier on A's trade damage, extrapolated linearly from the
  equal-power path, at which B's best response to A's Trade becomes Yield or Safe.

**Sampled stage.**

- **Runs:** 10 seeds × 10 turns, with the engine's own chance events and common random numbers.
- **Policies:** scripted. Both sides Recall at ≤ 40% HP, or walk back (Safe) if Recall is not legal.
- **Lines:** A Hold vs B Contest (Hold), A Trade vs B Contest, and A Trade vs B Concede (Yield).
- **Value:** gold difference + λ · (HP-bar difference at the end of the window).

---

## 4. Results

### 4.1 One turn at the lane (neutral, equal power)

The B loss columns attribute B's uncollected farm exactly. Contest base inefficiency is not shown. Damage is in HP
units and includes the expected gank (0.14 for a Hold champion at the lane).

| A–B | rule | gold A | gold B | B loss: zone / posture / reach | E dmg to B (large) | E dmg to A | V (large) |
|---|---|---|---|---|---|---|---|
| Hold–Hold | presence | 0.315 | 0.315 | 0.135 / 0 / 0 | 0.14 | 0.14 | 0 |
| Hold–Hold | none | 0.450 | 0.450 | 0 / 0 / 0 | 0.14 | 0.14 | 0 |
| Trade–Hold | presence | 0.158 | 0.315 | 0.135 / 0 / 0 | 0.86 (1.17) | 0.18 | +0.012 (+0.091) |
| Trade–Hold | contact | 0.225 | 0.315 | 0.135 / 0 / 0 | 0.86 (1.17) | 0.18 | +0.079 (+0.158) |
| Trade–Hold | none | 0.225 | 0.450 | 0 / 0 / 0 | 0.86 (1.17) | 0.18 | −0.056 (+0.023) |
| Trade–Trade | presence | 0.158 | 0.158 | 0.135 / 0 / 0 (+0.158 trade) | 0.90 (1.21) | 0.90 | 0 (+0.079) |
| Trade–Trade | none | 0.225 | 0.225 | 0 / 0 / 0 (+0.225 trade) | 0.90 (1.21) | 0.90 | 0 (+0.079) |
| Trade–Yield | presence | 0.450 | 0.140 | 0.060 / 0.250 / 0 | 0.08 | 0.18 | +0.286 |
| Trade–Yield | none | 0.450 | 0.200 | 0 / 0.250 / 0 | 0.08 | 0.18 | +0.226 |
| Trade–Safe | any | 0.450 | 0 | 0 / 0 / 0.450 | 0.07 | 0.18 | +0.423 |

**The intended interaction already exists mechanically.** In Trade–Yield, A threatens, B yields, A's Trade never
connects, A farms in full and B keeps only its retreat farm. So A gains without landing an attack.

**B does not choose it.** Conceding costs B 0.25 and gives A back 0.225, for 0.475 in total, while it saves B about
0.18 of HP value at equal power and 0.26 at large power.

### 4.2 Six turns (stationary pairs)

| state | A–B | presence: gold A / B, V (large) | none: gold A / B, V (large) |
|---|---|---|---|
| neutral | Hold–Hold | 1.89 / 1.89, 0 | 2.70 / 2.70, 0 |
| neutral | Trade–Hold | 0.95 / 1.89, +0.07 (+0.54) | 1.35 / 2.70, −0.34 (+0.13) |
| neutral | Trade–Yield | 2.70 / 0.84, +1.61 | 2.70 / 1.20, +1.25 |
| forward | Hold–Hold | 2.36 / 1.89, +0.36 | 3.38 / 2.70, +0.56 |
| forward | Trade–Hold | 1.18 / 1.89, +0.20 (+0.68) | 1.69 / 2.70, −0.10 (+0.37) |
| forward | Trade–Yield | 3.38 / 0.84, +2.28 | 3.38 / 1.20, +1.92 |
| tower | Hold–Hold | 0.88 / 2.68, −2.17 | 1.25 / 3.15, −2.27 |
| tower | Trade–Hold | 0.48 / 2.51, −3.04 (−2.66) | 0.69 / 3.15, −3.48 (−3.09) |

**XP.** In every deterministic window, XP collected equals XP available for both sides: 24 of 24 at the lane, 28 of
28 under the tower. XP range 3 covers every cell the baseline behaviours stand on. **No rule and no behaviour denies
XP in these states.**

### 4.3 Equilibria (λ = 0.25; pure unless a mix is shown)

| state | h | power | presence: A / B (V) | contact: A / B (V) | none: A / B (V) |
|---|---|---|---|---|---|
| neutral | 1 | equal | Trade / Trade (0) | Trade / Trade (0) | Hold / Hold (0) |
| neutral | 1 | moderate | Trade / Trade (+0.036) | Trade / Trade (+0.036) | Hold / Hold (0) |
| neutral | 1 | large | Trade / Trade (+0.079) | Trade / Trade (+0.079) | **Trade / Hold** (+0.023) |
| neutral | 6 | equal | Trade / Trade (0) | Trade / Trade (0) | Hold / Hold (0) |
| neutral | 6 | moderate | Trade / Trade (+0.213) | Trade / Trade (+0.213) | Hold / Hold (0) |
| neutral | 6 | large | Trade / Trade (+0.470) | Trade / Trade (+0.470) | **Trade / Hold** (+0.133) |
| forward | 1 | equal | Trade / Hold (+0.015) | Trade / Trade (+0.017) | Hold / Hold (−0.012) |
| forward | 1 | moderate | Trade / Hold (+0.054) | Trade / Trade (+0.056) | Hold / Hold (−0.012) |
| forward | 1 | large | Trade / Hold (+0.100) | Trade / Trade (+0.102) | **Trade / Hold** (+0.033) |
| forward | 6 | equal | Hold / Trade (+0.287) | Trade / Trade (+0.133) | Hold / Hold (+0.557) |
| forward | 6 | moderate | Trade / Trade (+0.349) | Trade / Trade (+0.349) | Hold / Hold (+0.557) |
| forward | 6 | large | Trade / Trade (+0.609) | Trade / Trade (+0.609) | Hold / Hold (+0.557) |
| tower | 1 | any | Hold / Hold (−0.107) | Hold / Trade (−0.159) | Hold / Hold (−0.130) |
| tower | 6 | any | Hold / Hold (−2.173) | Hold / Hold (−2.270) | Hold / Hold (−2.270) |

**Supports over all three λ** (18 games per cell):

| rule | neutral, A / B support | forward, A / B support | tower, A / B support |
|---|---|---|---|
| presence | H 6, T 12 / H 6, T 12 | H 6, T 12 / H 9, T 9 | H 18 / H 15, T 3 |
| contact | T 18 / T 18 | T 16, H 2 / T 18 | H 18 / H 12, T 6 |
| none | H 10, T 8 / H 12, T 6 | H 12, T 6 / H 15, T 3 | H 18 / H 18 |

**Yield and Safe appear in no equilibrium support in any of the 162 games.**

### 4.4 When does B concede? The flip multiplier

m\* is the attack multiplier on A's trade damage at which B's best response to A's Trade becomes Yield or Safe
(λ = 0.25). The large condition is ×1.44, and ×1.82 with the M4.2b item candidate 1.35.

| state | presence h1 / h6 | contact h1 / h6 | none h1 / h6 |
|---|---|---|---|
| neutral | 2.60 / 2.51 (Yield) | 2.27 / 2.17 (Yield) | 2.58 / 2.49 (Yield) |
| forward | 2.35 / 2.98 (Yield) | 2.03 / 2.65 (Yield) | 2.39 / 2.86 (Yield) |
| tower | 1.22 / 1.00 (Safe) | 1.00 / 1.00 (Safe) | 1.13 / 1.00 (Safe) |

**The concession inequality** (zero-sum, one turn, A Trades). B prefers conceding to contesting iff

```
λ · ΔD_B(m)  >  [G_B(contest) − G_B(concede)]  +  [G_A(vs concede) − G_A(vs contest)]  −  λ · ΔD_A
```

The right-hand side has three parts:

- **B's own retreat loss:** at the lane this is 0.45 − 0.20 = 0.25 under `none`, or 0.315 − 0.14 = 0.175 under
  `presence`.
- **A's farm refund:** when B concedes, A's Trade does not connect, so A's trade cost is refunded: 0.45 − 0.225 =
  0.225, or 0.45 − 0.158 = 0.292.
- **ΔD_A, the damage B's contest deals to A.** It matters only for a Trade-back.

The sum is 0.475 (none) or 0.467 (presence). On the left, ΔD_B ≈ 0.72·m units, which is 0.18·m at λ = 0.25. Hence
m\* ≈ 2.6 under both rules: **the proximity zone is not what blocks threat-based zoning.**

**At the tower** the flip is ≤ 1.2 because A's Trade is a dive: A takes the tower shot (1.45 units per turn), and A
would rather not Trade at all.

### 4.5 Sampled windows with real HP consequences (10 seeds × 10 turns, mean, ±95%)

| rule / state | power | A Hold vs B Contest | A Trade vs B Contest | A Trade vs B Concede | B deaths, recalls (Trade vs Contest) | B XP (Hold–Contest → Trade–Contest) |
|---|---|---|---|---|---|---|
| presence / neutral | equal | +0.07 ± 0.37 | −0.66 ± 0.39 | +2.71 ± 0.28 | 0.2, 0.5 | 39.2 → 34.8 |
| presence / neutral | moderate | +0.07 | −0.10 ± 0.64 | +2.71 | 0.5, 0.2 | 39.2 → 34.4 |
| presence / neutral | large | +0.07 | +0.22 ± 0.86 | +2.71 | 0.5, 0.4 | 39.2 → 31.6 |
| none / neutral | equal | +0.07 ± 0.36 | −1.04 ± 0.48 | +2.13 ± 0.25 | 0.2, 0.5 | 39.2 → 34.8 |
| none / neutral | moderate | +0.07 | −0.52 ± 0.71 | +2.13 | 0.5, 0.2 | 39.2 → 34.4 |
| none / neutral | large | +0.07 | −0.13 ± 0.87 | +2.13 | 0.5, 0.4 | 39.2 → 31.6 |
| presence / forward | equal / large | +1.73 ± 0.23 | +1.52 / +1.74 | +2.26 / +2.31 | 0.7 / 0.8, 0 / 0.1 | 33.4 → 31.0 / 26.8 |
| none / forward | equal / large | +2.51 ± 0.21 | +1.28 / +1.61 | +1.93 / +1.97 | 0.7 / 0.8, 0 / 0.1 | 33.4 → 31.0 / 26.8 |

**Real HP feedback** (deaths and recalls) does not change the picture:

- For B, Contest beats Concede in every cell.
- For A, Hold beats or ties Trade.

**Power shows up in the right place** — as Trade becoming less bad, and as more deaths, recalls and XP lost by B (XP
denied through absence) — but not in time to flip anything at ×1.44.

### 4.6 Sensitivity: which term blocks the concession? (diagnostic, not a proposal)

This sensitivity removes the trader's farm cost (`trade_farm_factor = 1.0`), under `presence` and `none`, in the
neutral and forward states:

- Trade becomes the pure equilibrium for both sides in all 24 games.
- The flip multiplier barely moves (none: 2.27 / 2.17 neutral, 2.03 / 2.24 forward).

The reason is that once trading is free, B's contest option is the *Trade-back*. That only neutralizes the exchange,
and conceding still costs B 0.25 per turn. The binding term is B's own **retreat cost**, set by geometry and posture:

- at the lane, the only out-of-contact cells hold kills only at Safe efficiency 0.4 (vs 0.9);
- `farm_reach` equals `trade_range`, so "in reach of my wave" and "in reach of the enemy champion" coincide.

---

## 5. Interpretation

- **The proximity zone is power-blind and symmetric, so it is a poor stand-in for zoning.** Removing it is
  justified on design grounds:
  - it is the only farm loss that needs no action by anyone;
  - removing it makes all positional and wave gaps 43% larger in gold;
  - it also removes the incentive it created: it made a trader's farm cost artificially small, so the lane traded
    at equal power.
- **The threat already exists in the engine, but conceding is too expensive for it to bite.** An unconnected Trade
  costs A nothing, and a yielding B loses farm without being hit. Threat-based zoning needs a *cheap partial
  concession*: a position that keeps a meaningful share of farm while staying out of the stronger champion's
  contact range. The baseline interface has no such position at the lane; that is exactly what positioning
  (Agent B) must supply.
- **Power must enter only through what A can do to B.** Under `none` it does: trade damage, kill and recall pressure,
  and the XP lost to absence. No farming formula reads attack power.

## 6. Outstanding positioning dependencies — the post-integration checklist

These must be checked on the merged build (positioning from Agent B, wave rules from Agent A), with
`zoning_rule="none"` against `"presence"`, using this driver's machinery: `run_pair`, `matrix`, `solve`,
`flip_multiplier` and `hp_window`, with the new positions added as rows and columns. The 4 × 4 behaviour set must
be extended with the new positioning options.

1. **Default preservation.** `GameConfig.m4()` (presence) must reproduce the merged default bit-for-bit, and all of
   `tests/test_m4_zoning.py` must pass, in particular "the rule changes gold only".
2. **Contest/concede geometry exists.** For a forward A at its aggressive cell, there must be:
   - a B contest cell P_c that is within farm reach of B's kills and in A's contact range;
   - a B concession cell P_s that is within farm (or retreat) reach but out of contact, with gold share
     0 < s(P_s) < s(P_c).

   Report s(P_s) / s(P_c). At the baseline lane it is 0.4 / 0.9 = 0.44, which is too low to concede into.
3. **The flip happens at moderate power.**
   - **From identical states:** with B choosing between Contest-at-P_c, Concede-to-P_s and Trade-back, and A
     between Hold-forward and Trade, compute m\* at h1 and h6 for λ ∈ {0.15, 0.25, 0.35}.
   - **Target:** at λ = 0.25, B's best response to A's Trade is Contest (or Trade-back) at equal power and
     Concede at large power (m\* between ≈ 1.2 and 1.5).
   - **Failure:** if m\* > 2 the threat is still not credible. If m\* ≤ 1, equal-power players concede, which is a
     proximity penalty in disguise.
4. **A benefits without repeatedly attacking.** In the solved games of item 3:
   - A's equilibrium Trade probability must *fall* as B's concession share rises.
   - A's value must rise with power, against equal-power trading.
   - The forward A must earn ≥ its Hold farm when not attacking. A forward position should cost exposure (ganks, the
     tower-adjacent shot), not farm.
5. **Trade-back does not dominate conceding** at the flip: check V(T, Trade-back) against V(T, Concede) (§4.6).
6. **The equal-power lane stays active.** Under `none` with baseline positions, the lane is mutual Hold at equal and
   moderate power (§4.3). The new positions must not leave mutual Hold as the only equilibrium at equal power.
   Re-check this against the M4.2b passive-equilibrium risk.
7. **Economy.** `none` raises contested gold by ≈ 43% (mutual Hold at the lane: 0.45 vs 0.315 per turn).
   - Re-check item timing (tier 1 at 4.0, tier 2 at 6.0) and the Recall cycle.
   - If total gold must be preserved, consider lowering `collect_contest` uniformly, a calibration step that was not
     done here. Never use a power-dependent term for this.
8. **Pushing.** Under the enemy tower, `none` makes the pusher slightly worse off (−0.10 over 6 turns). Confirm with
   Agent A's wave changes that this does not deepen the "pushing loses" finding.
9. **XP.** Concession cells must keep XP (range 3). XP denial must still require absence or a retreat into the own
   tower cell, never mere proximity.
10. **HP dynamics.** Repeat the sampled windows with the new positions to confirm that a conceding B is not simply
    farmed down by Recall pressure, and that a contesting B at large deficit actually loses (deaths, recalls, XP).
11. **Accounting and symmetry.** `gold_breakdown` must sum exactly, the zone part must be 0 under `none`, and mirror
    equivariance must hold with the new positions.

## 7. Simulation turns

| Item | Turns |
|---|---|
| Reported run (`python -m lanegame.experiments.m4.zoning`) | **4,966** |
| — start states | 22 |
| — power-invariance check | 96 |
| — 3 rules × 3 states × 16 pairs × 6 turns | 864 |
| — `trade_farm_factor = 1.0` sensitivity (2 rules × 2 states × 16 × 6) | 384 |
| — sampled HP windows (2 rules × 2 states × 3 powers × 3 lines × 10 seeds × 10 turns) | 3,600 |
| Development re-runs of the driver (identical stages, superseded) | 7,528 |
| Exploratory traces before the driver | 42 |
| **Cumulative experimental** | **12,536** (under the 15,000 cap; over the 5,000 target because of re-runs) |
| Focused tests (`tests/test_m4_zoning.py`, about 1.9k turns per run, 4 runs) | ≈ 7,500 (not experiments) |

## 8. Recommendation and caveats

**Recommendation.**

- **Keep `zoning_rule="presence"` as the default.**
- **Carry `zoning_rule="none"` into the integrated build as the candidate rule.** It is the smallest viable change:
  - it removes the only automatic proximity loss;
  - it keeps every reach, posture, trade and absence cost;
  - it adds no formula of its own.
- **Adopt it only if checklist items 2–6 pass.**
- **Reject `"contact"`.** It is kept in the code only so that §4 can be reproduced.

**This study does not show** that proximity penalties are inherently wrong, or that removing them creates meaningful
zoning. With baseline positioning it does not.

**Caveats.**

- **HP valuation:** HP is valued linearly (λ) in the deterministic stage. The sampled stage adds deaths and recalls,
  but uses only 10 seeds, scripted policies and a 10-turn window. Absence beyond the window is not charged.
- **Coverage:** three states; stationary pairs; Slow Push only; power = items only at equal level and equal skill.
- **The forward state** is a hand-modified counterfactual (§3).
- **Dependence on positioning parameters:** the conclusions depend on the Safe-efficiency slope (0.2 per cell), on
  `farm_reach = trade_range = 1` and on the 0.5 trade factor. These are the parameters Agent B's positioning work
  will replace or reinterpret.
- **Results metadata:** the meta block in `results/m4_zoning/zoning.json` names the parent commit f504f5e, because
  the run was made on the uncommitted working tree of this branch.
