# M4.2e-A — Natural Wave Mechanics

Branch `development/m4-natural-waves`, from `integration/m4-emergent-ready` (23e48be).

- **Preset:** `GameConfig.m4_natural()`, opt-in. It is `m4_emergent()` plus natural waves (`wave_dynamics="natural"`)
  on an 8-HP wave. `m4()`, `m4_continuous()` and `m4_emergent()` are unchanged.
- **Driver:** `python -m lanegame.experiments.m4.natural_waves` (0.4 s).
- **Results:** `results/m4_natural_waves/natural_waves.json`.
- **Tests:** `tests/test_m4_natural_waves.py` (19 tests).

**Method.** Every result comes from a deterministic trajectory reached from the standard start s0. There are no
seeded stacks and no hand-placed waves. Two kinds of run were used:

- The M4 resolver with no chance events (`u = None`: no ganks, no trade hits), driven by short scripted policies.
- Lane-only tick traces (`mechanics.step_turn`), with the champion standing at its own front (reach = front + 1).

Nothing was trained or sampled.

**Coordinates.** Trajectories use GLOBAL coordinates:

- A walks toward +. A's tower is at −2, B's at +2, and the lane cell is 0.
- `+0:17 r[+4]` means a front at cell 0 with 17 HP and one rear packet at +4. `sg12` is a 12-HP crashed mass.

**Verdict: PASS WITH CAVEATS (§11).**

## Summary

- **The fix is a timescale fix: the wave is now 8 HP.** At 4 HP a wave in contact dies in 2 ticks under any champion
  damage. A reinforcement needs 3 ticks to reach a stationary front, so nothing can ever accumulate. At 8 HP a
  fighting front survives 4 ticks (Thin) or 8 ticks (Hold). That outlasts the 3-tick reinforcement arrival, so
  reinforcements merge and advantages are stored.
- **The economy is rescaled to keep a wave worth what it was:**
  - farm 0.125 per HP (1 resource per wave);
  - XP 1 per HP (8 XP per wave);
  - crash cap 96 HP;
  - gold ledger unit 1e-7 (so every per-HP value stays exact).

  Nothing else changed: geometry, spawn timing, tower damage (1 HP/tick) and the Lanchester law floor(h / W).
- **Minions follow one set of rules whatever the actions.** Fronts walk, engage on contact, merge and crash. A
  defender's front engages the crashed mass under its tower; the old idle wait is gone. Slow Push parking, creeping,
  Freeze holds and the idle siege wait are all removed.
- **The three wave slots now mean champion damage on enemy minions:** Hold 0, Thin 1 and Clear 2 HP per tick. The
  damage needs presence, reach and availability. A connected Trade, a Recall, absence or a Safe retreat deals none.
- **Every essential interaction emerges from reachable states:**
  - **Neutral:** symmetric perpetual contact at the lane.
  - **Slow push:** one turn of Thin snowballs into a 16-HP crash 17 turns later.
  - **Deliberate crash:** Clear on a built stack crashes 4 turns earlier and 3 HP larger.
  - **Bounce:** every crash is followed by the defender pushing into the attacker's half and crashing back.
  - **Freeze:**
    - Bounded Thin turns an enemy push into a held wave at the own tower, with no crash.
    - A Hold then keeps it there for 21 turns.
    - An enemy Thin or Clear breaks it at once.
  - **Recall:** a well-timed Recall costs 0.46. A badly timed one costs up to 1.35.
- **Caveats (§8, §11):**
  - The freeze is temporary.
  - Clear cannot crash an unstacked wave.
  - Pushing against a passive opponent costs the pusher farm.
  - The evaluation bound is analytic and loose.

## 1. Weaknesses of the previous natural-wave prototype

These refer to M4.2d-A, branch `prototype/m4-emergent-wave`, documented in its report on that branch.

1. **Wrong timescale.**
   - At 4 HP a fresh wave lives 4 ticks in contact with no champion help, and 2 ticks with any champion damage.
   - A reinforcement spawned at the base reaches a stationary front 3 ticks after spawning.
   - So any champion damage ended the fight before the next wave arrived. The winner's remnant (≤ 2 HP) walked on
     alone and died to the next enemy wave.
   - No unseeded trajectory ever held more than one wave. There were no crashes, plates or bounces.
2. **Its "minimal" action was not minimal.**
   - Slow Push was a last hit (+1 on the finishing tick).
   - That shortened a neutral fight from 4 to 3 ticks. That alone prevented the merges a passive lane would have had.
3. **Its Freeze was inert.**
   - "Thin only above 7 HP" never applied, because no blob exceeded 4 HP.
   - Freeze played exactly like Slow Push.
4. **It kept the idle siege wait (rule D7).**
   - A defender front stopped behind its tower without fighting while the tower ate the crash.
   - That is a rule written to manufacture a bounce. The config comment said as much: "which is what lets
     reinforcements accumulate into a bounce".
5. **Champions could only add damage to an ongoing minion fight.**
   - A champion could not clear a wave walking into its tower.
   - It could not clear a crashed mass with no minion of its own.
6. **Farm-spot oscillation** (prototype §5).
   - Because fronts died inside every turn, the farm spot jumped back to the next packet.
   - The champion then lost reach and farm.
   - At 8 HP fronts mostly survive in contact, so the problem largely disappears. `m4_emergent`'s
     `farm_spot_rule="pre_crash_front"` is kept.

## 2. Final minion movement and combat rules

The rules are in `mechanics._tick_natural`; the module docstring gives the rule list. One decision turn = 2 ticks;
waves spawn every 4 ticks at own cell −4. Per tick:

| step | rule |
|---|---|
| 1 movement | Every free front walks one cell. The only things that stop it: contact with the enemy front (distance ≤ 1), or, for a defender, contact with the enemy's crashed mass under its own tower (own x ≥ −3, i.e. within distance 1 of the tower cell). Rears walk, and never into the cell of a front they cannot join. The same for every action. |
| 2 merging | A friendly packet within distance 1 merges into the front if the result fits the cap (24 HP); otherwise it queues directly behind and fights with it ("block" rule, unchanged). |
| 3 contact fight | Each side deals `floor(blob / 8)` (at least 1) minion damage, plus its champion's budget if the enemy front is within the champion's reach, plus the tower (1/tick) on an attacker standing in that tower's cell. |
| 3b siege engagement | A defender front within distance 1 of its tower fights the crashed mass: Lanchester both ways, plus the defender champion's budget. This replaces the idle wait. It is skipped on a tick where that front fights in the lane. |
| 3c Apply Pressure | Thinning by the pressuring champion, unchanged. |
| 3d champion alone | An unspent champion budget strikes the enemy's crashed mass under its own tower, else the enemy's leading packet if reachable. No own minion is needed. |
| tower hits | The crashed mass present at the start of the tick hits the tower for `floor(mass / 8)`, unless it fought defenders that tick (minions attack minions first). |
| 4–5 crash, tower fire | Unchanged. A front in the enemy tower cell with no enemy front within 1 crashes; the tower removes 1 HP/tick from the crashed mass. |

**What the rules guarantee:**

- **Determinism.** Every step is a pure function of the state and the actions.
- **Mirror symmetry.** Verified on 1,500 random lane turns and 400 random resolver turns with asymmetric skills.
- **Exact HP accounting.** Every removed HP is one farm opportunity for the remover's side at the cell where it died.

**State and observation.** The representation is unchanged: one front, fresh-wave rears, crashed mass. So is the
88-entry observation.

**Reviewing the siege waiting rule.** The scripted wait (rule D7, `siege_engage=False`) stops the defender without
contact; it is removed. Under natural waves the defender stops for one reason only: it is within distance 1 of minions
it is fighting.

## 3. Hold / Thin / Clear

The three wave slots are kept, so the action space is unchanged (46 codes). Their meaning changes only under the
new preset.

| slot | historical | natural name | champion damage | fresh wave killed in (W = 8) | other effects |
|---|---|---|---|---|---|
| 0 (SP) | Slow Push | **Hold** | 0 HP/tick | 8 ticks (minions only) | none; Recall's code carries this slot |
| 2 (F) | Freeze | **Thin** | 1 HP/tick | 4 ticks | always legal |
| 1 (SH) | Shove | **Clear** | 2 HP/tick | 3 ticks | keeps Shove's +0.02 gank-exposure term |

- **Targets.** The champion's budget goes to the first reachable target in this order:
  1. the enemy front fighting its own front;
  2. the enemy's crashed mass under its own tower;
  3. the enemy's leading packet.
- **Reach.** A target is reachable when its own-perspective cell is at or behind the champion's reach line. The
  resolver already computes that line:
  - `NO_REACH` while away, channeling or in a connected trade;
  - `min(end, safe_line)` for Safe or retreating;
  - `min(T, end + farm_reach)` while farming.

  No champion code changed.
- **Legality.** All three slots are always legal: `wave_actions`, the masks and `champion_legal_actions`.
- **Freeze maintenance** is off under natural waves (`freeze_maintenance=False`; validation enforces it).
- **Labels.** Under the preset, labels read `HO-H`, `TN-T+P`, `CL-H` and so on.
- **Scripted agents.** `agents_m4.NaturalWaveAgent` and `natural_suite()` provide mechanical rules (hold, thin, clear,
  slow push, bounded-surplus freeze, build-then-clear, crash recall). The historical agents still run, with the slot
  meanings changed.

**Champion presence is enforced.** From the neutral contact at k9, one turn of A's action against B
(`reach` stage):

| A's action | A's reach | champion HP removed from B |
|---|---|---|
| Thin + Hold behavior | +1 | 2 |
| Clear + Hold | +1 | 4 |
| Hold + Hold | +1 | 0 |
| Thin + Trade (connected) | none | 0 (and A farms 0.225 instead of 0.675) |
| Clear + Trade (connected) | none | 0 |
| Thin + Safe | −1 (the lane is out of reach) | 0 |

## 4. Parameters and justification

### 4.1 The timescale problem (section 3 of the brief)

These are lane-only tick traces, Thin (A) against Hold (B), from s0, under identical natural rules. Waves spawn at
ticks 0, 4, 8, …

| W = 4 | | W = 8 | |
|---|---|---|---|
| tick 4 | contact at 0, A 3 / B 3 | tick 4 | contact at 0, A 7 / B 7 |
| tick 5 | A 2 / B 1 (Thin) | tick 7 | A's reinforcement merges: A 12 / B 9 |
| tick 6 | B's front dies; A's remnant 1 HP, its reinforcement still at −2 | tick 11 | merge: A 16 / B 8 (B now deals 1, A deals 2 + 1) |
| tick 7 | A's remnant dies to B's next wave at +1 | tick 15 | merge: A 20 / B 6 |
| tick 9–16 | the same 4-tick cycle: A's fronts die singly near B's tower, max front 4 HP | tick 17 | B's front dies; A's 18-HP blob walks on |
| | | tick 24 | A crashes 15 HP into B's tower |

**Distinguishing spawn from arrival:**

- A wave spawned at tick s reaches the lane cell at s + 4 if it walks freely.
- It joins a front that is stationary at the lane at s + 3, when it reaches distance 1.
- With the fight at the lane, the merge ticks are 7, 11, 15, …

| quantity (ticks) | W = 4 | W = 8 |
|---|---|---|
| fresh wave against fresh wave, Hold | 4 | 8 |
| ... against Thin | 2 | 4 |
| ... against Clear | 2 | 3 |
| reinforcement reaches a stationary front | 3 after spawn | 3 after spawn |
| reinforcement interval | 4 | 4 |

**Diagnosis.** The ratio that matters is how long a front survives in contact against the 3-tick reinforcement
arrival.

- At W = 4 any champion damage kills the front before the reinforcement arrives.
- At W = 8 even a Thinned front survives to absorb every reinforcement.

It is not tower damage, not geometry, and not the merge or collision code. (The prototype had already shown that
longer lanes, slower minions and the ceil damage law do not help at 4 HP.)

### 4.2 Choices

| parameter | value | why |
|---|---|---|
| `wave_hp` | 8 (was 4) | The minimum that lets a fighting front outlive the reinforcement arrival under Thin (4 > 3 ticks). |
| `farm_per_hp`, `xp_per_hp` | 0.125, 1 | 1 resource and 8 XP per wave, as before. |
| `gold_unit` | 1e-7 | 0.125 × 0.9 × 0.85 × 0.5 is not a multiple of 1e-6; 1e-7 keeps the exact ledger (≤ 2^24 for the float32 observation). |
| `hp_cap`, Lanchester law | 24 (3 waves), floor(h / 8) | Both follow `wave_hp`: one unit of firepower per full wave. |
| `siege_cap` | 96 | 12 waves, as 48 was at W = 4 (max observed in soaks: 24). |
| `champion_wave_damage` | (Hold 0, Clear 2, Thin 1) | Thin = one wave's firepower, Clear = two. Hold must not perturb (§1, item 2). |
| tower | 1 HP/tick on crashed mass and on a contested attacker | Unchanged. Faster towers are worse (below). |
| geometry, spawn interval, ticks per turn | unchanged | The five-cell lane supports every interaction (§5). |
| `phase_period` | 4 (unchanged) | The true period is 2 (no creep clock). 4 keeps the observation's phase one-hot and its 88-entry layout. |

**Tower damage, measured** (`bounce` stage, A crashes and then Holds):

| tower: on crash / on contested attacker | small crash: HP, clear turns, defender front at clear, minion tower hits | large crash: same |
|---|---|---|
| **1 / 1 (chosen)** | 12 HP, 6 turns, 20 HP, 10 hits | 25 HP, 11 turns, 16 HP, 27 hits |
| 2 / 1 | 12, 3, **10**, 5 | 25, 6, 19, 15 |
| 2 / 2 | **5**, 1, 6, 1 | 22, 5, 13, 8 |

- Faster tower clearing shortens the window in which the defender accumulates. It halves the crasher's tower hits
  (plates), and it does not bring the bounce sooner (the defender reaches the attacker's half after 14–19 turns in
  every variant).
- A stronger tower on contested attackers shrinks the crashes themselves.

The tower stays at 1/1.

## 5. Representative trajectories

**5.1 Neutral (Hold / Hold).** Symmetric perpetual contact at the lane: fronts 7, 5, 11, 9, 14, 12, 16, 13, 17, 14,
18, … HP on both sides. Each side merges a reinforcement every 2 turns and settles where its blob deals 2 per tick.
Over 40 turns: 0 crashes, A − B = 0, farm 15.975 each.

**5.2 Natural slow push: A Thins on turn 1 only, then both Hold** (`stacking.thin_k1_1`).

| k | A | B | note |
|---|---|---|---|
| 1 | +0:7 | +0:5 | one turn of Thin: +2 HP of edge |
| 5 | +0:15 | +0:11 | both merge every 2 turns; A's edge grows as its blob crosses 16 HP first |
| 11 | +0:24 r[−4] | +0:10 | Lanchester feedback: A's blob deals 2–3/tick to B's 1 |
| 12 | +0:22 r[−2] | +0:5 r[+2] | A holds three waves at the lane |
| 15 | +0:24 r[−1,−4] | +0:1 | ... with a queued reinforcement behind it |
| 16 | +1:22 | +2:5 | B's front dies; A advances |
| 18 | +1:8 sg16 | – | **crash 16 HP** (and 16 more at k19) |
| 20–30 | +2:2..6 | +3:4 → +3:22 | B's fronts engage the mass behind their tower and absorb A's single waves with tower help |
| 32–37 | +1:2 → −4:8 | +2:20 → −1:18 | **bounce:** B's 20-HP blob pushes into A's half |
| 38 | – | −1:8 sg22 | **B crashes 22 HP** (and 8 at k39) |

Over 80 turns the cycle repeats with a period of about 38–40 turns: A crashes at k18, 19 and 57; B at k38, 39 and 77.

**5.3 Crash and bounce: A Thins until it crashes, then Holds** (`bounce.thin_until_crash`).

- **k9:** A crashes 12 HP.
- **k10–16:** B's fronts fight the mass at +3 and grow from 6 to 20 HP. A's single follow-up waves die at +2 to B's
  front plus the tower.
- **k15:** the mass is gone (6 turns). Fronts: A 1, B 20 behind its tower.
- **k17–21:** B's 17–23-HP blob walks out and beats A's single waves.
- **k22–24:** B reaches A's tower.
- **k25:** B crashes 22 HP.
- **k26–35:** A's front engages B's mass at −3, grows to 23 HP, and the cycle reverses.

**5.4 Freeze** (§6.3).

- **k22–25:** the contact is at the lane, with B ahead by 14–16. A Thins only while B leads by ≥ 8.
- **k26:** A's front dies.
- **k27:** fronts meet at −1 (A 3, B 24).
- **k28–29:** B's 19 → 13 HP wave stands in A's tower cell against A's 6 → 2 HP front. That is the freeze state.

## 6. Evidence

### 6.1 Hold, Thin and Clear against a Holding opponent (40 turns, from s0)

| A's action | A − B | farm A / B | XP A / B | A's crashes | contact cells |
|---|---|---|---|---|---|
| Hold | 0.000 | 15.98 / 15.98 | 142 / 142 | – | lane throughout |
| Thin | −2.000 | 14.88 / 16.88 | 143 / 152 | k9: 12 HP | lane → B's tower → +1 (B's bigger wave held there by A's Thin) |
| Clear | −4.700 | 11.73 / 16.43 | 155 / 147 | none | B's tower cell for 7 turns at a time, then back |

**What the table shows:**

- **The three actions differ.**
  - Hold changes nothing.
  - Thin creates a controlled edge that stacks.
  - Clear kills a wave in 3 ticks instead of 8.
- **Clear is not a dominant crash button.**
  - Clearing an unstacked wave ends each fight before reinforcements merge, so A's front reaches B's tower with no
    stack.
  - There it stands in the tower cell, taking tower and minion damage. B's front waits behind the tower at +3, out
    of the reach of A's champion (reach ≤ T).
  - It never crashes and loses the most farm.

### 6.2 Stacking and deliberate crashes

| A's line (B Holds) | A's first crash (k: HP) | later |
|---|---|---|
| Thin k1, then Hold | 18: 16, 19: 16 | B bounces: 38: 22, 39: 8 |
| Thin k1–2 | 16: 15, 17: 16 | B 35: 16 |
| Thin k1–4 | 15: 25 | B 33: 16 |
| Thin k1–6 | 13: 20 | B 31: 25 |
| Thin always | **9: 12** | – |
| Thin k1–3, Hold | 15: 16 | – |
| Thin k1–3, **Clear from k8** | **11: 19** | – |
| Thin k1–3, Clear from k10 | 13: 21 | – |
| Clear always | none | – |

- **Small advantages accumulate.** Larger and longer Thin windows crash sooner.
- **Clear on a built stack is the deliberate crash.** It comes 4 turns earlier and 3 HP larger than letting the stack
  walk in.
- **Timing condition.** A crash needs the attacker's front to enter the enemy tower cell when the defender's next wave
  is still ≥ 2 cells away: at its spawn cell, or not yet spawned. The defender's wave reaches its own tower 1–2 ticks
  after spawning. A crash therefore needs a stack that kills the defender's front within that window. That is why
  stacked blobs crash and single waves do not.

### 6.3 Freeze: maintenance and breakage

**Reaching it.**

- Setup: B Thins on turn 1 only, then Holds. A plays a bounded-surplus rule: Thin while B's front leads by ≥ margin
  HP, else Hold.
- No rule moves minions.

| margin | turns with contact at A's side (≤ −1) in 50 | outcome |
|---|---|---|
| 2, 4, 6 | 0 | the contact stays at the lane forever: A's Thin holds B's push in place at any cell |
| **8** | 17 | B's wave arrives at A's tower with a moderate surplus and is held there; later A pushes out and crashes (k47) |
| 10 | 19 | the same, crash at k43 |
| none (A Holds) | – | B crashes 16 HP at k18 (§6.2, mirrored) |

**Maintaining and breaking it.** From the freeze state at k30 (A −2:2, B −2:13 inside A's tower cell):

| B's action from k30 | turns held | crashes | farm A / B (16 turns) |
|---|---|---|---|
| Hold (A keeps its rule) | 14 | none | 9.34 / 2.48 |
| Hold (A Holds, no Thin), 30 turns | **21** | A pushes out and crashes 21 HP at k57 | 14.88 / 6.95 |
| **Thin** | **1** | B crashes 9 HP at k31 | 8.66 / 4.20 |
| **Clear** | **1** | B crashes 12 HP at k31 | 7.88 / 5.66 |

**Why it holds:**

- B's front stands in A's tower cell. It loses 1 (tower) + 1 (A's minions) per tick and gains 8 per 4 ticks, so it
  stays at 13–17 HP.
- A's own smaller front gains 2 per tick and loses 1. It grows by about 1 HP per turn.

**Why it ends.** Once A's front reaches two waves (16 HP), it deals 2 per tick, B's wave melts, and A pushes out.
That is a natural, temporary freeze:

- It depends on relative HP, reinforcement timing, the tower and position.
- It earns the freezer a large farm edge.
- The opponent's champion breaks it by walking into reach (own +1, next to the enemy tower) and Thinning or
  Clearing the freezer's small front, which then crashes.

**Reading the setup result.** Thin's role is to establish the freeze: it keeps an incoming push from becoming a
crash. Once B's wave sits in the tower cell, the tower does the thinning, and an extra Thin only shortens the hold
(14 turns against 21).

### 6.4 Crash size and bounce (A crashes and then Holds; B Holds)

| line | crash HP | clear turns | fronts when cleared (A / B) | B in A's half after | B's bounce crash |
|---|---|---|---|---|---|
| Thin until crash | 12 | 6 | 1 / 20 | 14 turns | 22 HP after 16 turns |
| Thin k1–3, Clear k8 | 19 | 9 | 5 / 21 | 15 | 25 after 18 |
| Thin k1–4 | 25 | 11 | 4 / 16 | 16 | 16 after 18 |
| Thin k1 (16 + 16 HP) | 32 | 13 | 8 / 22 | 17 | 22 after 20 |

**The mechanism:**

- **While the mass is cleared**, B's waves are stationary in contact with it, so every B reinforcement merges.
- **A's follow-up waves** walk in singly and die to B's front plus the tower.
- **After clearing**, B's blob (16–22 HP) meets A's waves one at a time and wins.
- **No reversal rule exists.** The bounce comes from the tower supplying free damage on the attacker and from the
  defender's front standing still.

**Larger crashes give longer windows:**

- clearing: 6 → 13 turns;
- B reaches A's half after 14 → 17 turns;
- the bounce crash lands after 16 → 20 turns.

## 7. Recall timing

A walks back (Safe) until Recall is legal, Recalls (2 turns away), then Holds. Each cost is measured against the same
line without the Recall.

| situation | Recall turn | missed gold | missed XP | cost (A − B) |
|---|---|---|---|---|
| neutral lane | 13 | 1.150 | 8 | 1.150 |
| right after own small crash (12 HP) | 12 | 0.625 | 6 | 0.625 |
| after own large crash (25 HP; 3 turns to walk back) | 18 | 0.750 | 6 | 0.750 |
| after a failed push (Clear k1–6) | 10 | 0.500 | 6 | 0.500 |
| enemy builds and crashes 25 HP into A's tower at k15: Recall k7 | 7 | 0.575 | 4 | 0.575 |
| ... Recall k12 (back as the crash arrives) | 12 | **0.463** | 4 | **0.463** |
| ... Recall k13 | 13 | 0.675 | 6 | 0.675 |
| ... Recall k15 (as it lands) | 15 | 0.900 | 8 | 0.900 |
| ... Recall k17 / k20 (during the siege) | 17 / 20 | **1.350** | 12 | **1.350** |

- **A Recall after one's own crash is cheap.** Little farm is available while one's wave is dying under the enemy
  tower.
- **A Recall during the enemy's crash is expensive.** The crashed enemy minions die under one's own tower, and only
  a present champion farms them.
- **Hold deals no damage, so the lane at return is identical with or without the Recall.** An opponent punishes an
  absence only by acting during it. Its slow push needs about 10 turns to land, against a 2-turn absence.
  - In the probe where B Thinned for 4 turns from A's neutral Recall, B's crash landed 8 turns after A returned.
  - A farmed it under its own tower and came out +2.3 ahead.
- **A pre-built push is what punishes a bad Recall.** That is the k15–20 rows of the table.

## 8. Mechanical limitations

1. **The freeze is temporary.**
   - The freezer can only shrink the enemy wave; nothing caps its own.
   - The hold ends when its own front outgrows two waves: about 14–21 turns here.
   - Whether the enemy wave stops at −1 (outside tower range) or −2 (inside) depends on spawn parity. Only the
     tower-cell variant was observed. With a single cell between the lane and the tower, the geometry is coarse.
2. **Clear cannot crash an unstacked wave.**
   - Champion reach ends at the enemy tower cell (T).
   - A defender waiting at +3 behind its tower cannot be hit, except by Apply Pressure's thinning during a siege.
   - Crashes therefore need a stack. This is coherent, but it makes Clear mainly a tool to convert stacks, clear
     crashes and break freezes.
3. **The neutral lane is unstable by design.** Any edge (one turn of Thin, about 2 HP) snowballs through the
   Lanchester law into a crash about 17 turns later. A push–bounce oscillation follows, with a period of about 38–40
   turns. Small decisions therefore have large delayed effects. That is good for strategy, but it is a long credit
   assignment horizon for PPO.
4. **Pushing costs farm against a passive opponent:**
   - Thin always: −2.0 over 40 turns.
   - Clear always: −4.7.
   - One turn of Thin: −3.85 over 80 turns.

   The reason: crashed minions are the defender's farm, while the crasher's champion collects at 0.5 under the enemy
   tower. Plates (3 against 2 in the 80-turn cycle), Recall tempo and ganks are the only offsets. This is economics,
   not mechanics; see §10.
5. **Simplifications:**
   - A defender front fights either the lane attacker or the crashed mass in a tick, never both.
   - The tower shoots the crashed mass and a contested attacker in the same tick.
   - Queued packets add firepower but take no damage until they become the front.
6. **Evaluation bound.** The natural lane has no enumerated bounds. `lane_enum.natural_lane_bounds` gives analytic
   ones: A_max 408 HP, so Ω = 107 instead of 68. They are guaranteed but loose. An enumeration is deferred to
   integration.
7. **The probe and farm spot are unchanged.** In the freeze, the freezer's champion stands in its tower cell and the
   opponent's in the enemy tower cell. Positioning that reacts to the wave (§10) is not part of this milestone.

## 9. Performance, tests and budget

**Performance (single thread, this machine):**

| | lane step | env step (resolver + observation, uniform random) |
|---|---|---|
| `m4_emergent` (scripted) | 48.2 µs | 210 µs |
| `m4_natural` | 50.7 µs | 220 µs |

**Tests** (`tests/test_m4_natural_waves.py`, **19 passed**, 1.6 s). They cover:

- **Correctness:**
  - preset values and validation;
  - legacy rejection;
  - legality, masks and labels;
  - identical movement under all 9 action pairs;
  - champion damage by slot and reach;
  - a champion clearing a walking wave and a crashed mass alone;
  - siege engagement and tower hits;
  - HP conservation, kill accounting and mirror equivariance on 1,500 random lane turns;
  - resolver mirror equivariance, invariants and the per-turn reward bound on 400 random turns;
  - lossless observation and decode;
  - env soaks: 2,000 random turns and four scripted natural pairs (zero sum, exact reward decomposition, minion
    conservation, plate gold);
  - evaluation bounds.
- **Pins of §5–7:**
  - neutral symmetry;
  - Hold / Thin / Clear;
  - one-turn-Thin stacking, crash and bounce;
  - Clear on a stack;
  - crash-size bounce windows;
  - freeze reach, maintenance (10+ turns under the freezer rule, 21 turns under Hold) and Thin/Clear breakage;
  - Recall timing;
  - Trade, Safe and Recall reach.

**Focused regressions:** `test_mechanics`, `test_legacy_pins`, `test_m4_config`, `test_m4_emergent_ready`,
`test_m4_env`, `test_m4_tower`, `test_m4_calibration`, `test_invariants`, `test_m4_resolve`, `test_m4_farm_spot_rule`,
`test_wave_mechanics`, `test_m4_observation`, `test_m4_scenarios` and `test_m4_diagnostics`, together with the new
file — **305 passed, 4 skipped** (167 s).

- Historical outputs are byte-identical: the natural-only event key is added only under natural waves.
- One fixture in `test_m4_config` gained alternate values for the two new M4-only fields.
- The full suite and `--runslow` were not run.

**Simulation budget:**

| what | turns |
|---|---|
| exploratory scratch runs (timescale traces, action / stacking / freeze / bounce / tower / Recall probes) | ≈ 2,830 |
| driver `natural_waves` (four runs of the same deterministic 2.5k-turn driver: first, micro-test fix, freeze extension, scripted-agent fix) | ≈ 10,180 |
| performance benchmark (1,000 lane + 1,000 env turns per preset) and 80-turn cycle check | ≈ 2,160 |
| **total** | **≈ 15,190** |

That is above the 10k target and under the 20k cap. The overrun is repetition, not exploration: without the repeated
driver runs and the timing benchmark, the experiment is about 5.4k turns.

## 10. Implications for positioning and zoning integration

1. **Zoning now has something to act on.**
   - Under natural waves, champion presence is wave control: Thin and Clear need reach. A connected Trade, a
     retreat or a Recall removes it.
   - A champion that zones its opponent off the wave also strips its wave-control budget. That is a new,
     non-gold cost of being zoned, and it should change the "concession is not rational" finding of M4.2d.
2. **Reach geometry matters.**
   - Breaking a freeze requires standing next to the enemy tower (own +1).
   - Freezing keeps one's own champion at the own tower.
   - Clearing a defender behind its tower is impossible.

   The positioning model's fixed probe spot (own front) and reach (+1) now decide who can touch which wave. Combat-
   driven zoning should be checked on exactly these states: the freeze state and the post-crash siege.
3. **The economic incentive to push is weak.** It should be revisited with zoning, before PPO. Options, none tested
   here:
   - collection efficiency under the enemy tower;
   - plate value;
   - ganks on a champion standing in the enemy half.
4. **For PPO:**
   - The preset keeps the 46-action space and the 88-entry observation.
   - The 38–40-turn push–bounce cycle and about 17-turn snowball delays fit γ = 0.99 (horizon ≈ 100 turns).
   - The loose analytic Ω must be replaced by an enumerated bound, or accepted, before headline evaluations.

## 11. Verdict: **PASS WITH CAVEATS**

| stopping criterion | status |
|---|---|
| consistent natural movement for every wave | yes (§2; 9 action pairs move identically; no parking, hold or idle wait) |
| distinct Hold / Thin / Clear | yes (0 / 1 / 2 HP per tick; kill in 8 / 4 / 3 ticks; different 40-turn outcomes) |
| small advantages accumulate into larger waves | yes (one Thin turn → 24-HP stack → 16-HP crash) |
| champions can deliberately create crashes | yes (build with Thin, convert with Clear: 19 HP at k11) |
| natural bounces in plausible scenarios | yes (every crash tested; the defender pushes back and crashes 16–25 HP) |
| freezes maintained and broken without forced movement | yes, temporary (14–21 turns; broken in 1 turn by Thin or Clear) |
| Recall timing interacts with wave states | yes (0.46 to 1.35 depending on the enemy crash timing; own-crash Recalls cheap) |
| core correctness | yes (conservation, mirror symmetry, accounting, legality, observation, zero sum) |

**Caveats accepted for integration:**

- The freeze is temporary.
- Clear needs a stack to crash.
- Pushing is economically weak against a passive opponent.
- The analytic evaluation bound is loose.
- One cell separates the lane from the tower.

None needs a new mechanic to proceed to the zoning check. The economic one should be looked at alongside it.

**Next step (not started):** verify combat-driven zoning on top of `m4_natural()` in the freeze and post-crash states.
Then integrate and freeze the game for PPO.
