# M4.2e-A.1 — Natural Wave Mechanics: Final Review

Branch `development/m4-natural-waves`, on top of M4.2e-A (356b5b0).

- **Final preset:** `GameConfig.m4_natural()`, opt-in. It is the M4.2e-A prototype with four changes (§3).
- **Prototype, kept for comparison:** `GameConfig.m4_natural_prototype()` reproduces M4.2e-A exactly. Its report is
  [m4_2e_natural_waves_report.md](m4_2e_natural_waves_report.md); its driver and tests now run on this preset.
- **Driver:** `python -m lanegame.experiments.m4.natural_waves_final` (≈ 19k turns, 2 s). Results go to
  `results/m4_natural_waves/natural_waves_final.json`.
- **Tests:** `tests/test_m4_natural_final.py` (19 tests).

**Method.** As in M4.2e-A:

- Every result is a deterministic trajectory from the standard start s0, with no seeded stacks.
- Runs use the M4 resolver with no chance events (`u = None`) and short scripted policies.
- Lane-only traces are used only where noted.
- Champion power is varied through a fixed Clear multiplier on side A (`natural_waves_final.play(mult=...)`). §4.1 maps
  each multiplier to the (level, investment) states that reach it.
- Coordinates are GLOBAL: A's tower is at −2, B's at +2 and the lane cell is 0. `+0:17 r[+4] sg12` means a front at
  cell 0 with 17 HP, one rear packet at +4 and a 12-HP crashed mass.

**Verdict: PASS WITH CAVEATS (§10).**

## Summary

- **The real defect was at the tower, not in the wave HP.** Under the prototype's crash rule, a front entering the enemy
  tower cell next to a defender did not crash. It fought there as a second attacker entity beside the crashed mass.
  As a result:
  - the defender's minions fought the follow-up waves and almost never the crashed mass;
  - the tower fired at both entities;
  - the mass died to tower fire alone and hit the tower for the whole siege.

  That is why sieges lasted 6–13 turns.
- **Siege length does not cause the bounce.** Across a 4× range of tower damage, prototype sieges lasted 1–13 turns,
  yet the bounce came 16–21 turns after the crash every time. The bounce comes from sensible dynamics: the defender
  fights at its own tower, with tower support, next to its spawn.
- **Clear could not crash because of reach, not damage.** At 16 HP per tick Clear stalled exactly like at 2 HP per tick.
  Defenders standing behind the tower were out of reach.
- **The fix is one rule plus one parameter.**
  - Rule: entering the enemy tower cell *is* the crash, so every attacker minion under a tower is part of one siege.
  - Parameter: the tower must out-damage the reinforcement stream, τ · spawn_interval > W. That gives τ = 3.

  Two small companions:
  - a champion standing in the enemy tower cell reaches one cell behind it;
  - Clear's budget scales with attack power, while Thin stays a fixed controlled rate.
- **What this produces:**
  - Sieges last 3–9 turns.
  - Stacked crashes still bounce, 20–22 turns later, with 24 HP.
  - Clear gets faster with power: from a neutral lane it crashes in 5 turns at AP 1 and in 1–2 turns at AP 4.
  - A genuine outside-tower freeze appears. The enemy wave is held at the cell in front of one's tower for 20+ turns
    with a 2.4× farm edge. The opponent's Clear crashes it within 2 turns.
- **Main caveat: the economics of pushing.** Against a passive opponent, slow pushes pay (+4.5 over 80 turns), constant
  pushing is heavily punished (−10.8 to −18.8), and push + Pressure is roughly even. All three hinge on one economic
  parameter, the 0.5 collection rate under the enemy tower. At 1.0 the ranking flips (§8.1). That should be calibrated
  with combat-driven zoning, not here.

## 1. Diagnosis

### 1.1 Tower clearing, crashes and bouncing

**What a prototype siege looked like.** This is the first siege of "Thin until crash, then Hold" (`siege` stage). A
crashes 12 HP at k9.

| k | crashed mass | A (follow-up front) | B front | HP removed by tower / by defenders | B's ticks fighting the follow-up / the mass | mass hits on tower |
|---|---|---|---|---|---|---|
| 9 | 12 | +0:8 | +4:8 | 0 / 0 | 2 / 0 | 0 |
| 10 | 9 | +2:6 | +3:6 | 2 / 1 | 1 / 1 | 1 |
| 11 | 7 | +2:2 | +3:12 | 2 / 0 | 2 / 0 | 2 |
| 12 | 5 | +2:6 | +3:10 | 2 / 0 | 2 / 0 | 2 |
| 13 | 3 | +2:2 | +3:16 | 2 / 0 | 2 / 0 | 2 |
| 14 | 1 | +2:5 | +3:14 | 2 / 0 | 2 / 0 | 2 |
| 15 | 0 | +2:1 | +3:20 | 1 / 0 | 2 / 0 | 1 |

**Over the whole siege:**

- The tower removed 11 HP of the mass; the defenders removed 1.
- The defender's front fought A's follow-up waves on 13 ticks and the mass on 1.
- The tower also fired 13 HP at the follow-up fronts standing in its cell.

**The cause is the crash rule "no enemy front within distance 1".** A follow-up wave walking into B's tower cell finds
B's front behind the tower at +3, so it does not crash. It becomes a second attacker entity in the same cell, and the
lane fight takes precedence over the siege fight. As a result:

- **The defender ignores the mass.** Siege length becomes mass ÷ tower damage, independent of the defender's strength.
- **The tower fires twice per tick**, once at the mass and once at the contesting front.
- **The mass hits the tower on almost every tick**, because it is never engaged. That is where most of the prototype's
  plate income came from: 43 hits in one siege of the 80-turn cycle.
- **The defender is pinned at +3 until the mass dies.** The movement rule stops it there, so the siege also sets when
  the defender may walk out.

**Does the siege drive the bounce? No.** Prototype tower-damage sweep (`siege.tower_sweep`; A crashes, then Holds):

| tower: on mass / on contested front | siege length, turns (small, 4-turn build, 1-turn build) | defender in attacker half after | bounce crash after |
|---|---|---|---|
| 1 / 1 (prototype) | 6, 11, 13 | 14, 16, 17 | 16, 18, 20 |
| 2 / 1 | 3, 6, 8 | 14, 14, 17 | 16, 18, 19 |
| 3 / 1 | 2, 4, 6 | 14, 14, 17 | 16, 18, 19 |
| 4 / 1 | 2, 3, 4 | 14, 14, 17 | 16, 18, 19 |
| 3 / 2 | 1, 3, 5 | 19, 17, 19 | 20, 18, 21 |

With contested damage at 1, siege length varies 3–6× while bounce timing moves by at most 1 turn. A stronger
contested tower (1 → 2) delays the small line's bounce by 4 turns, because it shrinks the crash itself (12 → 5 HP).
So the bounce comes from what happens around the tower, not from the length of the accumulation window:

- A's follow-up waves arrive one at a time.
- B's front fights them with tower support from a cell adjacent to its spawn, so every B reinforcement merges at
  once.
- When an A wave dies before the next one arrives, B walks out with the advantage.

That is the League mechanism ("the tower kills the next wave") and it is sound. The artefacts are the siege's
*duration*, its *plate income* and the *defender's size* when the siege ends: B had 20 HP at τ = 1 and 8–12 HP at
τ = 4.

### 1.2 Fast pushing and champion power

Prototype rules with a stronger Clear (`champion_wave_damage = (0, c, 1)`), Clear against Hold from s0 for 30 turns:

| Clear, HP per tick | A's crashes | A − B |
|---|---|---|
| 2 (prototype) | none | −5.20 |
| 3 | 4 HP at k3; 8 at k17 | −0.96 |
| 4–5 | 3–5 HP ×4 | −1.76 |
| 8, 12, 16 | 2–5 HP ×3–4 | −3.0 to −3.6 |

Even an 8× stronger champion stalls. Its wave reaches B's tower and then stands in the tower cell for 7 turns. B's
fresh wave stands at +3, behind the tower, and:

- A's champion cannot reach it (reach is capped at T);
- every B reinforcement merges into it the tick after spawning;
- A's front takes tower damage plus minion damage.

**The limitation is geometry and reach, not champion damage.** Fixed damage was a second, real gap: wave clear did not
respond to levels or items at all.

### 1.3 Freezing

The prototype's freeze held B's wave *inside* A's tower cell, where the tower did the thinning (§6.3 of the prototype
report). That was possible only because the contact crash rule lets a front stand in the enemy tower cell next to a
defender. A freeze outside tower range was never observed. The reason is not that the mechanics forbid it, but that the
inside-tower state was always reached first.

### 1.4 What was *not* wrong (kept)

- **8 HP per wave.** The timescale argument of M4.2e-A still holds. A fighting front survives the 3-tick reinforcement
  arrival.
- **The integer Lanchester law `floor(h / 8)`.**
  - Lane-only, I compared it with a half-wave resolution law (W = 16 sub-HP, every damage doubled). Crash sizes and
    timings were similar for Hold, Thin, Clear and slow push (`granularity` stage):
    - Thin: first crash 23 HP at k10 against 21.5 at k8;
    - Clear: 15 at k6 against 18 at k6.
  - Its breakpoints are what create the freeze band in §6.
  - Finer HP would cost an XP and ledger rescale (XP per HP would no longer be an integer) for no qualitative gain.
- **Five-cell geometry and spawn timing.** Spawn → tower cell is 2 ticks (half a spawn interval) and spawn → lane is
  4 ticks. That is close to League's proportions, and the outside-tower freeze works on it (§6).
- **The defender's front engaging a siege from behind its tower**, and the champion-presence rules of M4.2e-A.

## 2. Alternatives considered

| alternative | outcome | decision |
|---|---|---|
| Faster tower only (prototype crash rule, τ = 2–4) | Shorter sieges; still two entities, a double-firing tower, the defender ignoring the mass and the reach stall | rejected alone |
| Join rule only: a follow-up merges into an existing crashed mass | Fixes the siege, but a first wave reaching the tower next to a defender still fights inside the tower cell | subsumed by entry |
| **Entry rule:** entering the enemy tower cell is the crash | One siege per tower and one tower target. No front ever fights inside the enemy tower cell, and the contested-attacker tower damage becomes unnecessary | **adopted** |
| Entry rule with τ = 1 | Sieges feed themselves: 42 turns and 196 HP from one turn of Thin (`sustain`) | rejected |
| Entry rule with τ = 2 | τ · 4 = W: a lone follow-up stream is exactly cancelled, so sieges last 18 turns / 100 HP | rejected (marginal) |
| **Entry rule with τ = 3** | τ · 4 = 12 > 8: sieges last 2–10 turns; bounces survive | **adopted** |
| Entry rule with τ = 4 | Sieges last 2–6 turns; freeze hold 24 turns instead of 28; plates even rarer | not needed |
| Uncapped reach for every champion | Under the entry rule no farming champion stands in the enemy tower cell, so only the extra cell for a champion standing in it matters | adopted as `natural_tower_reach = 1` |
| Wave clear unscaled | No progression | rejected |
| AP scaling of Clear **and** Thin | Thin at AP 2 behaves exactly like Clear at AP 1. The controlled option disappears, and freezes and slow pushes would depend on level | rejected |
| **AP scaling of Clear only** (Thin fixed at 1 HP per tick) | Progression lives in fast-push capacity. Slow push and freeze stay level-independent | **adopted** |
| A separate wave-clear curve, or an exponent on AP | No evidence it is needed: AP already spans 1.0–5.6 | not adopted |
| Fractional or accumulated damage, finer HP units | Per-turn budgets give 1-HP granularity (1/8 wave) with no hidden state. Finer HP showed no qualitative change | not adopted |
| Longer lane, separate tower-range cells | The entry rule already gives an outside-tower freeze on the five-cell lane | not needed |
| Raising enemy-tower collection to 1.0 | Flips the push economics (§8.1) | **out of scope**; flagged |

## 3. Final mechanics

**Rules.** Changes to the M4.2e-A rules are in bold.

| step (per tick) | rule |
|---|---|
| 1 movement | Free fronts walk. Contact with the enemy front stops a front. A defender's front within distance 1 of its tower cell stops while the enemy has a crashed mass there. Rears walk. |
| 2 merging | "Block" cap rule (24 HP). |
| **2b entry crash** | **A front entering the enemy tower cell joins the crashed mass at once.** Follow-up waves join the same siege. No front ever fights inside the enemy tower cell. |
| 3 lane fight | Lanchester `floor(h / 8)` for each blob, plus each champion's budget on the enemy front if reachable. |
| 3b siege fight | The defender's front fights the crashed mass: Lanchester both ways, plus the defender champion's budget. |
| 3c, 3d | Apply Pressure thinning. Unspent champion budget strikes the enemy's crashed mass, else the enemy's leading packet. |
| tower | **The tower deals τ = 3 HP per tick to the crashed mass present at the start of the tick.** It has no other target. |
| hits | The crashed mass hits the tower (plates) only on ticks when defenders do not engage it. |

**Champion wave damage per turn:**

| slot | budget per turn (2 ticks) | split |
|---|---|---|
| Hold | 0 | — |
| Thin | 2 (1 HP per tick), fixed | 1, 1 |
| Clear | **round(2 × 2 × AP / 10⁴)**, AP at the start of the turn | ceil first: 5 → 3 + 2 |

- **Reach:** a champion standing in the enemy tower cell (Apply Pressure or Vision) reaches T + 1, the defender's
  siege cell. Farming reach is otherwise unchanged.

**Parameters:**

| parameter | prototype | final | why |
|---|---|---|---|
| `natural_crash_rule` | contact | **entry** | one siege per tower (§1.1) |
| `tower_siege_damage` τ | 1 | **3** | τ · spawn_interval > W, the smallest integer (§2) |
| `natural_tower_reach` | 0 | **1** | consistent with `thin_reach = 1`; matters only under Pressure (§4.4) |
| `wave_clear_scaling`, scaled slots | none | **attack_power, (Clear,)** | §4 |
| everything else | — | unchanged | W = 8, floor law, geometry, spawns, economy, champion combat |

`tower_damage` (the tower on a contested attacker) still exists but can no longer apply under the entry rule.

**Code:**

- `mechanics`: `_crash_check` and `_tower_fire` are split out of `_crash_and_tower_fire`, whose behaviour is
  unchanged; step 2b; `step_turn(champion_budget=...)`.
- `champions_m4`: `M4Context.wave_budget`, the reach cap and the budget passed to the lane step.
- `env.core`: the lane cache keys on the budget.
- `env.discounted`: `max_champion_wave_budget` in the per-turn bound.
- `config`: four fields, validated natural-only; `m4_natural_prototype()`.

## 4. Champion progression

### 4.1 Budget by level and investment (`power.budget_table`)

| level | invested (resources) | AP | Clear budget per turn | Thin |
|---|---|---|---|---|
| 1 | 0 | 1.00 | 4 | 2 |
| 3 | 0 | 1.10 | 4 | 2 |
| 6 | 4 | 1.48 | 6 | 2 |
| 9 | 16 | 2.13 | 9 | 2 |
| 13 | 32 | 2.93 | 12 | 2 |
| 18 | 64 | 4.19 | 17 | 2 |
| 18 | ∞ | 5.59 | 22 | 2 |

### 4.2 From a neutral stacked lane: Clear until the crash, then Hold (`power.from_neutral`)

The Hold/Hold lane at k9 has fronts 13/13. A at the given AP, B at AP 1.

| AP | turns to crash | first crash HP | HP entering the siege | siege length (turns) |
|---|---|---|---|---|
| 1.0 | 5 | 25 | 41 | 6 |
| 1.5 | 3 | 21 | 21 | 4 |
| 2.0 | 3 | 22 | 38 | 6 |
| 3.0 | 2 | 16 | 32 | 5 |
| 4.0 | 2 | 16 | 32 | 5 |

From k10 (fronts 17/17) the turns to crash are 4, 4, 3, 2, 1.

**Reading:** stronger champions crash sooner and smaller. A weak Clear takes longer, so more reinforcements merge
first ("more champion wave damage shortens fights and prevents friendly reinforcements from accumulating"). No crash
size is hard-coded anywhere.

### 4.3 From s0: Clear every turn (`power.from_s0`)

| AP | A's sieges (first crash HP) | A − B over 40 turns |
|---|---|---|
| 1.0 | 11, 12, 12, … every 4 turns (merged waves) | −8.39 |
| 1.5 | 4 (12 entered), 7, 12, … | −8.43 |
| 2.0–4.0 | 7–8 every 2 turns (every fresh wave) | −5.96 to −6.08 |

**How the available strategies change with power:**

- **At level 1**, Clear on an unstacked lane reaches the tower only with merged waves, every 4 turns. It is a weak
  shove that the tower and defenders absorb.
- **At AP ≥ 2**, Clear kills each enemy wave as it leaves its base. Every one of A's waves arrives almost intact, a
  perma-shove.
- **Larger crashes still need a stack:** slow push, or Thin then Clear (16–17 HP at AP 2–4).

**Thin-scaling counterfactual.** Thin scaled with AP at AP 2 gives exactly the same trajectory as Clear at AP 1
(sieges at k4, 8, 12 with 11–12 HP). The controlled option would vanish with power. That is why Thin stays fixed.

### 4.4 Pressure during a siege (`power.pressure_siege`)

After a slow push (Thin at k1, crash of 36 HP at k18), A's behaviour while its mass is under B's tower:

| after the crash | AP 1 | AP 2 | AP 4 |
|---|---|---|---|
| Hold (walk away) | −5.29; B bounces | same | same |
| Hold + Pressure | **+0.89**, 2 plates, no bounce | same | same |
| Clear + Pressure, reach T + 1 | −2.46, 3 plates | −1.85: a 26-turn siege, 132 HP fed | −1.74: the same |
| Clear + Pressure, reach T | −2.46 | −2.46 | −1.85: a 26-turn siege |

A strong enough champion committed under the enemy tower can keep killing the defenders and hold the siege open. The
threshold is AP 2 with the T + 1 reach and AP 4 without it. Plate gold comes from Pressure. That is a power-gated
strategy, not an artefact: the tower still out-kills the follow-up stream, and the champion is what removes the
defenders.

### 4.5 Breaking a freeze with power (`freeze.break_by_power`)

From the freeze state in §6, B plays a single turn of Clear and then Holds:

| B's AP | effect |
|---|---|
| 1 | held 17 more turns |
| 2 | held the full 20 turns |
| 4 | broken at once, with a 14-HP crash |

Sustained Clear breaks it at any power (§6.3).

## 5. Tower dynamics

**Bounce lines** (A crashes, then Holds; B Holds; `bounce`):

| line | prototype: crash HP, siege, tower hits, B front at clear, bounce after | final: crash HP (+ follow-ups), siege, hits, B front, bounce after |
|---|---|---|
| Thin k1 | 16 (+16), 13, 43, 22, 20 turns (22 HP) | 36 (+32), **9**, **5**, 9, **22 turns (24 HP)** |
| Thin k1–4 | 25, 11, 27, 16, 18 (16) | 31 (+24), **7**, **2**, 9, **20 (24)** |
| Thin until crash | 12, 6, 10, 20, 16 (22) | 18, **3**, 3, 6, no bounce within 50 |
| Thin k1–3, Clear k8 | 19, 9, 18, 21, 18 (25) | 23, **4**, 5, 12, no bounce within 50 |

**What changed:**

1. **Sieges are shorter and the tower takes few minion hits.** Defenders engage the mass, and a single tower target
   takes 3 HP per tick. Siege length now depends on crash size *and* defender strength.
2. **Stacked crashes still bounce.** The bounce comes later (20–22 turns) and is larger (24 HP), because the follow-up
   waves join the siege instead of dying one by one in the tower cell. The defender's surplus is built afterwards, from
   the waves the attacker sends after the siege.
3. **Small, fast crashes do not bounce at once, and that is correct.** After a 3-turn siege the attacker still holds a
   12-HP front against the defender's 6, so the lane stays pushed toward the defender. Over 70 turns this line gives:
   - A's stack sits just outside B's tower, contact at +1, for about 20 turns;
   - A crashes 35 HP at k30 (+8, +8, +8 follow-ups);
   - B bounces with 23 HP at k50 and 35 at k66.

   The push–bounce cycle still exists, with a period of about 40 turns.
4. **Crash size sets tower pressure.** A crash survives about W_mass ÷ (3 + defender damage) ticks, and only an
   unengaged mass hits the tower. Plates come from big crashes against a weak or absent defender, or from Pressure.

**Siege length versus bounce timing.** These are now distinct quantities with distinct causes:

- siege length is a fight between the mass and the tower plus defenders;
- bounce timing is set by how the next waves meet.

## 6. Freezing

### 6.1 The outside-tower freeze

Under the entry rule an enemy front can never stand in one's tower cell; it would crash. A freeze must therefore hold
the enemy wave *outside* tower range: the enemy front at −1, one's own front at −2 (one's own tower cell). It is a
fight equilibrium of the integer law. Per spawn interval (4 ticks), with one's own front F and the enemy blob E:

- **F gains 8** (its reinforcement arrives the tick after spawning) and **loses 4 · floor(E / 8)**. It is stable
  while E ≤ 23.
- **E gains 8** (its reinforcement walks 4 cells) and **loses 4 · (floor(F / 8) + Thin)**.

With F ≤ 15 and E in [16, 23], Thin every tick holds both steady. Hold lets E grow by 4 per interval. When F reaches 16,
Hold alone holds E. The freezer steers with Hold and Thin; no rule stops any minion.

### 6.2 Reaching it (`freeze.reach`)

B Thins at the start, then Holds. A plays the bounded freezer: Thin while B's front leads by at least the margin. The
table gives the longest hold at A's side (turns) and A − B over 60 turns.

| B's build | margin | prototype | final |
|---|---|---|---|
| Thin k1 | 6 | 0, −0.68 | 0, −0.68 |
| Thin k1 | 8 | 17, −1.63 | **28, +8.75** |
| Thin k1 | 10 | 18, −1.16 | **38, +6.41** |
| Thin k1–2 | 8 | 0 | 0 |
| Thin k1–2 | 10 | 19, −0.48 | **34, +5.50** |
| Thin k2 | 8 / 10 | 17 / 18 | **28 / 38** |
| Thin k1–4 | 6, 8, 10 | 0 | 0 |

- **Margin 6:** A's Thin holds B's push at the lane. That is a neutral standoff, not a freeze.
- **Thin k1–4:** the same at every margin.
- **Margins 8–10:** B's push lands as a crash (22–24 HP). The freeze forms on the follow-up waves, which A then holds
  at −1.

So a freeze is typically set up by absorbing an enemy crash under one's own tower and then controlling the next waves.
That is the League pattern.

### 6.3 Maintaining and breaking it (`freeze.from_k36`)

The state at k36 is A −2:11 against B −1:19: B's wave is outside A's tower, and A's front is in its own tower cell.

| from k36 (20 turns) | turns held | crashes | farm A / B | A − B |
|---|---|---|---|---|
| A freezer, B Holds | **20** (all) | none | 10.0 / 4.1 | +5.89 |
| A Holds (no control) | 13 | B 35 HP at k50 | 9.5 / 4.2 | +5.26 |
| B Thins | 7 | B 14 HP at k43 | 9.1 / 5.1 | +4.05 |
| B Clears | **1** | B 20 HP at k38, then a shove | 9.7 / 4.3 | +5.36 |
| B Clears once, then Holds | 15 | B 22 HP at k52 | 10.1 / 4.7 | +5.44 |

**Who does the work.** Unlike the prototype's in-tower freeze, the tower does none of it here. The freezer's
occasional Thin does: uncontrolled, the freeze collapses after 13 turns.

**How it breaks.** The opponent's champion stands next to A's tower (at its own +1) and Clears A's small front. B's
wave then walks into A's tower and crashes. The freezer farms the crash and keeps the edge it already banked, but the
lane resets.

**Is it permanent?** It is held as long as the freezer keeps controlling and the opponent does not commit to breaking
it. Each turn spent breaking it costs the breaker farm, and it stands next to the enemy tower in the enemy half, where
gank exposure is highest. Temporary but strategically meaningful, as required.

### 6.4 Interpretation

With one cell between the lane and the tower, "just outside tower range" and "the cell in front of the tower" are the
same cell. That coarseness is the main simplification. In particular there is no room for a freeze that sits further
from the tower.

## 7. Strategic outcomes

### 7.1 Against a Holding opponent, 80 turns (`actions`)

| A's line | prototype A − B | final A − B (A sieges / B sieges) | final, enemy-tower collect 1.0 |
|---|---|---|---|
| Hold | 0 | 0 | 0 |
| Thin always | −2.00 | −10.78 (6 / 0) | −6.59 |
| Clear always | −10.23 | −18.78 (6 / 0) | −4.03 |
| Thin k1 (slow push) | −3.85 | **+4.59** (2 / 2) | +0.40 |
| Thin k1–4 | −0.30 | **+4.45** (2 / 2) | +1.70 |
| Thin k1–4, then Pressure while sieging | −1.23 | −1.63 (4 / 0), 3 plates | **+4.44** |
| Clear, then Clear + Pressure while sieging | −10.74 | −14.03, 3 plates | **+3.04** |

**No action is unconditionally optimal against a passive opponent:**

- A single turn of Thin, the natural slow push, beats Hold. It snowballs into a crash and then a long period with
  the contact on the defender's side.
- Constant aggression loses.
- Pressure converts sieges into plates.

**What changed from the prototype.** Each wave action's value moved: Hold remains the reference, slow pushes now pay,
and constant pushing became much worse. The constant-push penalty is economic (§8.1).

### 7.2 Recall (`recall`, the M4.2e-A stage on the final preset)

| situation | prototype cost | final cost |
|---|---|---|
| neutral lane, Recall at k13 | 1.15 | 1.15 |
| right after own small crash | 0.63 | 0.38 |
| after own large crash | 0.75 | **0.00** |
| after a failed push | 0.50 | 0.69 |
| enemy builds and crashes 31 HP at k14: Recall at k12 (back as it lands) | 0.46 | 0.46 |
| ... Recall at k15 / k17 (during the siege) | 0.90 / 1.35 | **1.69 / 1.80** |

- A Recall right after one's own large crash is free: the siege lasts long enough to cover the absence.
- A Recall during the enemy's crash is more expensive than before. The crashed enemy minions die under one's own tower
  within a few turns, and only a present champion farms them.
- The windows are sharper than in the prototype.

### 7.3 Representative trajectories

- **Neutral (Hold / Hold):** symmetric perpetual contact at the lane, as in the prototype; Hold creates no edge.
- **Slow push:** Thin at k1, then Hold:
  - k18: crash of 36 HP, with 32 more HP from follow-ups;
  - 9-turn siege;
  - k38: B crashes back with 24 HP.
- **Fast push:** at AP ≥ 2, Clear crashes every fresh wave every 2 turns (§4.3).
- **Freeze from s0:** in the freeze setup (B Thins at k1; A freezes with margin 8):

  | k | event |
  |---|---|
  | 22–26 | B stacks against A's Thin at the lane |
  | 27 | contact at −1 (A 3, B 24) |
  | 28 | B crashes 22 HP into A's tower |
  | 30–35 | B's follow-ups are held at −1 against A's front at −2 (A 8–14, B 6–19) |
  | 36+ | the freeze holds (§6.3) |
- **Break:**

  | k | event |
  |---|---|
  | 36–37 | B Clears A's small front at −2; it dies at k37 |
  | 38 | B's 20-HP wave walks into A's tower and crashes |
  | 39–41 | B keeps shoving, 8 HP per crash |

### 7.4 Scripted population (`natural_suite`, 100 turns, seeded env)

| pair | prototype | final | final, collect 1.0 |
|---|---|---|---|
| slow_pusher vs hold_farmer | −2.21 | −14.59 | +4.10 |
| crasher vs hold_farmer | −0.88 | −13.80 | +0.88 |
| natural_freezer vs slow_pusher | −5.23 | **+6.49** | −5.89 |
| clearer vs thinner | −17.25 | −12.89 | +0.98 |
| crash_recaller vs natural_freezer | −1.71 | −5.94 | −1.06 |
| trading_pusher vs hold_farmer | −9.09 | −9.38 | −9.57 |

The scripted agents push permanently (18–37 crashes), so these pairs magnify the economic sensitivity of §8.1.

## 8. Tradeoffs and remaining limitations

1. **The economics of pushing hinge on `collect_contest_enemy_tower` (0.5).**
   - **Where the pusher's kills land.** Under the entry rule the attacker's lane fights against fresh defender waves
     happen at +1/+2. Most of the defender's minions die in the defender's tower cell (collected at 0.5) or behind it
     (out of a farming champion's reach).
   - **Where the defender's kills land.** The defender farms every crashed minion under its own tower at the full
     rate.
   - **Clear always, 60 turns:** A collected 11.9 of 30 available against B's 25.5 of 28.5.
   - **Sensitivity:** at 1.0 the ranking flips. Push + Pressure gains +3 to +4.4 and the freezer loses to the slow
     pusher.
   - **Recommendation:** keep 0.5 for now and calibrate it with combat-driven zoning. The static 0.5 stands in for
     the threat of farming next to the enemy tower, which zoning is meant to model directly, so calibrating it alone
     would double count.
2. **Plates now come mainly from Apply Pressure.** Defenders engage the mass, so minion tower hits fall from 27–43 per
   stacked siege to 2–5. That is coherent, but plate value (1 resource) and `pressure_tower_damage` were calibrated with
   the prototype's minion hits. Revisit them together with item 1.
3. **Large power gaps allow a perma-shove and a perma-siege.** A perma-shove needs AP ≥ 2 Clear; a perma-siege needs
   Clear + Pressure at AP ≥ 2 against AP 1. That is intended snowball, but it is strong. Champion-vs-champion combat
   and zoning (not modelled in this test, where both champions are passive) are the counterplay.
4. **Freezes are reachable mainly after an enemy crash.** They are sensitive to the freezer's margin (6 / 8 / 10 give
   0 / 28 / 38 turns). The integer law makes the control bang-bang. A learning agent must find the band. Scripted
   agents with a fixed margin do so only in some phases.
5. **Spawn-phase sensitivity remains.** Outcomes still depend on which tick a front dies relative to spawns (for
   example the non-monotone one-turn break in §4.5). That is inherent to a 2-tick turn and a 4-tick spawn interval.
6. **One cell between the lane and the tower** (§6.4).
7. **Thin does not scale.** A late-game champion's controlled option is the level-1 rate. Mid-strength control does not
   exist; the action space has three slots.
8. **Simplifications kept from the prototype:**
   - a defender front fights either a lane attacker or the siege in a tick;
   - queued packets add firepower but take no damage;
   - the evaluation bound is analytic (per-turn bound 10.36 against 7.66, from the larger Clear budget; tail constant
     unchanged).
9. **`tower_damage`** (contested attacker) remains a field but cannot apply under the entry rule.

## 9. Correctness, performance and compatibility

**Tests.**

- **New: `tests/test_m4_natural_final.py`, 19 tests.**
  - Preset delta and validation.
  - The entry crash (including next to a defender), follow-ups joining, and single-target tower fire.
  - Tower reach under Pressure.
  - The budget table, its monotonicity and its split.
  - The resolver passing the start-of-turn budget.
  - HP conservation, kill accounting, no front ever left in the enemy tower cell, and mirror symmetry on 1,500 random
    lane turns with random budgets.
  - Resolver mirror equivariance with a stronger seat, invariants, the 88-entry observation and its decoding, and the
    per-turn reward bound on 400 random turns.
  - Env soaks: 2,000 random turns and four scripted pairs; zero sum, exact reward decomposition, minion conservation.
  - Pins of §1, §4, §5, §6 and §7.
- **Prototype:** `tests/test_m4_natural_waves.py` (19 tests) now runs on `m4_natural_prototype()` and passes unchanged.
  Every M4.2e-A number reproduces.
- **Focused regression:** the M4.2e-A set plus env, champions, combat, economy, time and integration tests.
  - Result: **468 passed, 4 skipped** (22 files, 228 s).
  - The first run failed on four parametrizations of `test_m4_config`'s legacy-rejection fixture. It needed alternate
    values for the new fields, as in M4.2e-A; the fix is in the commit.
  - The full suite and `--runslow` were not run.

**Historical compatibility.**

- `GameConfig()`, `m4()`, `m4_continuous()` and `m4_emergent()` are unchanged.
- The new fields default to the M4.2e-A behaviour and are rejected outside natural waves.
- The scripted lane step goes through the same `_crash_and_tower_fire` as before.

**Interfaces.** `LaneStep` callables receive an optional sixth argument (the budget) only when wave-clear scaling is on.
Older five-argument lane steps keep working with every other configuration.

**Observation.** Unchanged, at 88 entries. The budget is a function of level and investment, both observed, so there
is no hidden state.

**Performance (single thread, this machine):**

| preset | lane step | env step |
|---|---|---|
| prototype | 38.2 µs | 207 µs |
| final | 38.7 µs | 206 µs |

**Simulation budget:**

| what | turns |
|---|---|
| exploratory scratch runs | ≈ 14k |
| driver runs (one aborted on a print error after its first stages, one complete at 18.9k) | ≈ 24k |
| **total** | **≈ 38k** |

Every run is deterministic and takes seconds; the full driver runs in 2 s.

## 10. Recommendation

**Is the natural-wave system mechanically complete enough to proceed to combat-driven zoning? Yes.**

| criterion | status |
|---|---|
| natural wave movement | yes: one rule set for every action; a crash is entering tower range |
| meaningful Hold / Thin / Clear | yes: Hold neutral; Thin a fixed controlled rate (slow push, freeze control); Clear the power-scaled fast push |
| natural stacking | yes: one turn of Thin snowballs into a 36-HP crash |
| fast pushing that responds to progression | yes: turns to crash 5 → 1–2 and crash size 25 → 13–16 as AP goes 1 → 4; perma-shove at AP ≥ 2 |
| freezing without forced movement | yes: outside tower range, held 20+ turns under control, broken by the opponent's Clear in ≤ 2 turns |
| crashing and bouncing | yes: sieges 3–9 turns; stacked crashes bounce after 20–22 turns; small crashes start a longer cycle |
| Recall timing | yes: 0.00 after one's own large crash, 1.69–1.80 during the enemy's siege |
| no unconditionally optimal action | yes, against a passive opponent; the margin depends on one economic parameter |
| correctness and performance | yes |

**For the zoning milestone:**

1. **Zone checks on two states.**
   - *Freeze state.* The freezer's champion stands in its own tower cell; the breaker must stand at its own +1, next to
      the enemy tower.
   - *Siege state.* A Pressure champion stands in the enemy tower cell and reaches the defenders.

   Reach is wave control: zoning a champion off the wave now also removes its Clear budget.
2. **Calibrate the push economics there.** That covers `collect_contest_enemy_tower`, plate gold and Pressure damage,
   which should be set once zoning's real cost of farming near the enemy tower exists.
3. **The analytic bound** (per-turn 10.36) should be enumerated or accepted before headline evaluations, as before.

### Final decision: **PASS WITH CAVEATS**

**Why it passes.** The important interactions now emerge from one consistent set of minion rules with no special cases:

- stacking, slow and fast pushes, crashes, bounces;
- a freeze outside tower range;
- Recall windows;
- wave clear that grows with champion power.

The prototype's genuine defects are fixed by one rule and one parameter, each justified by a measured mechanism:

- the split attacker entity in the tower cell;
- the self-limiting siege;
- reach, not damage, limiting pushes.

**Caveats** (§8), all well understood:

1. Push profitability hinges on the enemy-tower collection rate, which should be calibrated with zoning.
2. Plates now come mainly from Pressure.
3. Large power gaps allow a perma-shove or perma-siege.
4. Freezes need a precise control band and are reached mainly after an enemy crash.
5. There is one cell between the lane and the tower.
