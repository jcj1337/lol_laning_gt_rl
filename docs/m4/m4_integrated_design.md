# M4 — Integrated Champion Economy and Wave Dynamics: Mathematical Specification

*Milestone M4.1 (design only), branch `design/m4-integrated` from `research/openai-five-diagnostics` (4e68c75),
2026-10-11.*

*Revision 1 (review of the same day), with full accounts in audit §1.9–1.11:*
- *the observation is exactly Markov: lossless lane encoding, exact integer gold (§8.6, §12);*
- *the evaluation tail bound is derived from the simulator (§10.5.3);*
- *critic bootstrapping is documented as an approximation (§10.4, §10.5.4);*
- *the evaluation protocols E1 and E2 are separated (§10.5.1);*
- *Freeze has prototype evidence and a pre-training gate (§6.6);*
- *the ruleset contract adds `legacy`, `m2c` and `m4` (§2).*

*Nothing in this document is implemented. It specifies a new, separately versioned ruleset,
`GameConfig.m4()`. The legacy exact game (`GameConfig()`, `.v2()`, `.v1()`) and the M2 champion game
(`GameConfig.champions()`) are unchanged, and every pinned result stays reproducible.*

*Companion documents:*
- *[m4_implementation_plan.md](m4_implementation_plan.md): work packages, tests, validation gates;*
- *[m4_design_audit.md](m4_design_audit.md): contradictions, exploits, risks, ranked alternatives.*

*Terminology: own-perspective coordinates (own base −4, own tower −2, the lane cell 0, enemy tower +2), as
everywhere in the project. Facts about the current code cite file:line at 4e68c75. Numbers measured for this
document are marked "measured" (scripts in `results/m4_design/`, scripted agents, no training). Every other number is a
provisional modelling parameter.*

---

## Contents

1. Findings that motivate the design
2. Design summary
3. State
4. Actions and legality
5. Resolution of one turn
6. Wave management: Freeze, Slow Push, Shove
7. Positioning, farming (gold) and experience (XP)
8. Combat: levels, items, damage
9. Experience thresholds and level-ups
10. Time, discounting, reward and evaluation
11. Tower durability and plates
12. Observation and masks
13. Parameter table
14. Invariants
15. Worked scenarios (the ten required cases)
16. Alternatives and unresolved decisions
17. Acceptance tests (summary)

---

## 1. Findings that motivate the design

### 1.1 Why Freeze produced the same outcomes as Slow Push

The rules (mechanics.py):

- **Legality** (`wave_actions`, mechanics.py:81–92). Freeze is legal only when the enemy front exists and stands at
  own x ≤ `freeze_legal_max` = −1, i.e. on the cell next to one's own tower or inside it.
- **Own front** (mechanics.py:250–251). Under Freeze the own leading packet advances only while x < `freeze_line`
  = −1. Under Slow Push it walks to the lane cell (x = 0) at full speed and then creeps one cell per 4 turns, and
  only once it is stacked (`_sp_moves`, stack_creep).
- **Hold on the enemy front** (mechanics.py:252–260). An enemy front that *wants* to move from its own x ≥ +1
  into its own +2 (the freezer's tower cell, i.e. a crash) is stopped unless
  `blob_hp + freeze_break_push[enemy wave] ≥ freeze_break_threshold` (10). The push of 4 counts only for a Shove
  whose champion reach covers the wave.
- **Contact** (mechanics.py:130–131, 241–242). Fronts within one cell of each other (`x_A + x_B ≥ −1` in own
  coordinates) fight and do not move. Rear packets always walk.

When Freeze is legal, either the fronts are in contact, or the own front is at −3, at −4 (a fresh spawn) or
absent. Fronts in contact do not move. From these rules, a Freeze can change a turn's outcome only through two
channels:

- **(a) Own-front channel.** It binds only when the enemy front dies on the first tick of the turn while the own
  front stands at −1. Freeze then parks the own front at −1, where the next reinforcement merges into it. Slow
  Push instead walks it to the lane cell.
- **(b) Hold channel.** It binds only when the enemy front is at the freezer's −1, **not in contact**, and *wants*
  to enter the freezer's tower cell, all three at once:
  - under its Slow Push it wants to only on a creep tick (one turn in four) and only with more than 4 HP;
  - under its Shove it wants to every tick, but the hold fails whenever blob + push ≥ 10.

  Net effect: the hold stops a 5–9 HP creeping front or a small shoved front.

**Measured, by exhaustive enumeration of the reachable lane** (lane-only fixed point over (lane state, k mod 4);
legacy reach values and the full champion-reach superset give the same counts):

- Freeze differs from Slow Push in **1,126 of 15,458 Freeze-legal (state, phase, side) situations (7.3%)**.
- In contact states it differs in only 356 of 12,624 (2.8%), each time through channel (a) after a first-tick
  death.
- In non-contact states it differs in 770 of 2,834. In every non-contact state where it does not differ, the enemy
  blob is ≥ 10, so the freeze breaks automatically.
- In random or biased lane-only play, a Freeze-legal turn with the own front out of contact occurs on 0.5–2.4% of
  turns. 96–100% of these follow one pattern: own front killed last turn, fresh spawn at −4, enemy front at −1.

All 72 states of the diagnostics counterfactual were contact states: Freeze-legal states constructed by a Shove
prefix, after which the opponent followed its policy. In those states a one-turn Freeze is equal to Slow Push, so
the measured difference of 0.000 is exactly what the rules predict. **Freeze is not structurally identical to Slow
Push. It is inert in contact, and contact is where play reaches Freeze legality.**

**A second, unexpected finding** (simulation of a front parked at the freezer's −1 for 24 turns, freezer's own
wave just killed):

- Against a Slow Push front of **3–5 HP the configuration is itself stable under mutual Slow Push.** The enemy
  front, unstacked and past the lane, does not want to move, and its reinforcements merge into it before it dies.
  The freezer's fresh waves walk into it at the freezer's −2 for all 24 turns. **Every** freezer minion dies inside
  the freezer's tower cell, under Freeze and under Slow Push alike.

  So "the enemy wave parked next to my tower" is a property of the lane dynamics, not of the Freeze action.
- **What Freeze adds is crash denial.**
  - A Slow Push front of 5–9 HP crashes within 0–15 turns without a Freeze (on its creep ticks). With a Freeze it
    crashes after 3–12 turns, or never for 5 HP.
  - A shoved front of 2–5 HP crashes immediately without a Freeze. With a Freeze it crashes after 3–6 turns, or
    never for 2 HP.
  - A shoved front of 6 HP or more crashes immediately either way.

The economic reason completes the explanation. What Freeze buys (crashes denied or delayed) is worth almost
nothing under M2: a crash pays nothing by itself and per-crash plate opportunities lapse with the siege. The
positional consequences of the parked configuration exist with or without Freeze: the opponent farms next to the
freezer's tower, at efficiency 0.5 for kills inside it.

Two further quirks found by the audit:

- **Freeze is never legal with one's own siege active** (in any reachable state).
- **Fronts freeze in place during an enemy siege.** While the enemy has a siege under one's tower, one's own front
  waits at *any* cell x ≥ −3, not only behind the tower (mechanics.py:243–244; the comment says "waiting behind own
  tower"). This is recorded as decision D7 (§16).

### 1.2 Time model artifacts

- **The horizon is terminal.** `LaneEnv.step` sets `terminated = k + 1 ≥ horizon` (env/core.py:272).
- **The observation contains a countdown.** It includes `k/K` and `(K − k)/K` (env/observation.py:104–105).
- **PPO ignores discounting.** It uses γ = 1 (ppo.py:33–56), which is justified only because the countdown is
  observed.
- **Rollouts are lock-stepped whole episodes.** n_steps = 64 = 2K, and every environment starts at k = 0, so all
  rollouts consist of whole episodes in lock-step. The mid-episode bootstrap path of `compute_gae`, which is
  correct (ppo.py:88–100; train.py:288–304), never runs in any published experiment.

The consequence was a measured last-turns artifact: Shove from turn 28 on beats all 9 policies by +0.11 to +0.60
(diagnostics D.3), because a pushed wave's costs (crash, bounce) fall after turn 32.

### 1.3 Lane economy (measured, `GameConfig.champions()`, scripted agents, 400 games per matchup)

| Quantity | Value |
|---|---|
| Enemy minion HP dying per side | 2.0 per turn in steady state (one 4-HP wave per 2 turns); 1.62 over turns 0–7, while the first wave arrives |
| Gold collected, mutual `hold_farmer` | 0.29 per turn (61% of available) |
| Gold collected, learned policies (diagnostics B) | ≈ 0.20 per turn |
| Enemy minion HP dying within 1 or 2 cells of a present champion, mutual farming | 1.88 per turn; identical for ranges 1 and 2, since every kill happens at the lane cell where the champion stands |
| Same, under pressure | ranges 1 and 2 separate: 1.05 vs 1.52 for a farmer facing a persistent trader; 1.30 vs 1.74 for a shover |
| Long horizon (`horizon=200`) | runs cleanly; the per-turn rates are flat from turn 8 to turn 199, so turn 32 is not special for the lane economy |

### 1.4 Recall and HP

From the diagnostics report (B.1, C, D.1):

- HP is valued at ≈ 0.22–0.30 resources per HP point.
- Recall at HP ≤ 2 loses 0.02–0.47 against the policy; at HP 1 it breaks even.
- Recall restores HP but buys nothing, and it is legal only from cell ≤ −1.

M4 keeps Recall's price (two turns away, start at ≤ −1). It adds what Recall buys: HP restored to a level-scaled
maximum, an item purchase, and a continuing game in which the purchase pays off over many turns.

### 1.5 A defect relevant to Freeze: the trade chase from the tower cell

`resolve_turn` step 2 (champions.py:457–467) sends a chaser that is not in contact to the cell `trade_range` short
of its target. When the target stands on the cell next to the chaser's own tower (target's own +1), that cell is
the chaser's own tower cell (−2). Contact from a tower cell requires the other champion to be in the same cell
(`in_contact`, champions.py:382–387).

**Verified:** a Trade from −2 against an enemy holding at its own +1 ends at −2, not connected. The same Trade
from −1 connects. A freezer standing under its tower can therefore never punish an opponent farming its freeze.
M4 replaces the chase target (§7.4).

### 1.6 The M2 observation is not exactly Markov

Exhaustive enumeration shows that M2's clipped, summed minion grid maps 96 classes of reachable lane states to
shared observations. In 85 of the colliding pairs the states transition differently (§12.3). M4 uses a lossless
lane encoding and an exact gold encoding (§12.1–12.2). The legacy observation is unchanged.

---

## 2. Design summary

**Rulesets and configuration contract** (the details and `validate()` rules are in plan WP0):

| `ruleset` | Presets | Time | Countdown in observation | Status |
|---|---|---|---|---|
| `"legacy"` | `GameConfig()`, `.v1()`, `.v2()`, `.champions()` | finite horizon K, terminal | yes | frozen; bit-identical pins |
| `"m2c"` | `GameConfig.champions_continuing()` | continuing | no | **test fixture only**: M2 mechanics with M4 time semantics, used to validate WP1 before any M4 economy exists; never a research ruleset |
| `"m4"` | `GameConfig.m4()` | continuing | no | the M4 ruleset of this document |


| Element | M2 / M3 (`champions()`) | M4 (`m4()`) |
|---|---|---|
| Time | 32-turn episode, terminal at K, countdown observed, γ = 1 | continuing game; no terminal state; no countdown; objective discounted with γ = 0.99; training in fixed-length segments with bootstrapped truncation |
| Wave actions | SP / SH / F; Freeze holds regardless of the freezer's champion | same three, same legality; **a Freeze is effective only while the freezer maintains it** (present, not trading, within 1 cell of the freeze line); an unmaintained Freeze resolves as Slow Push |
| Gold | farm within 1 cell of the champion; posture/zone efficiencies | unchanged (gold reach 1) |
| XP | none | 2 XP per enemy minion HP dying within **3** cells of a present champion; independent of gold |
| Levels | none | 1–18; XP to the next level 8 + 3(L − 1); anchored to League's per-wave progression |
| HP | 6, integer | centi-HP; `hp_max(L) = round(600·1.05^(L−1))`; a level-up adds only the increase in max HP |
| Damage | trade 0/1/2 HP; gank Binomial(4, p) HP; tower shot 1 HP | same probabilities; a hit deals `D·100·AP(L, N)` centi-HP with `AP = 1.05^(L−1)·1.20^N`; gank and tower damage scale with the victim's max HP |
| Hit probability | skill, behaviour, minions, tower | **unchanged**: power never enters it |
| Tower near-cell | only a dive (end cell +2) draws a tower shot | a connected trade from the cell adjacent to the enemy tower (+1) draws a tower shot with p = 0.5 (0.25 if the tower is busy with one's minions) |
| Chase | target `e − 1` (defect, §1.5) | nearest cell in contact with the target |
| Items | none | one offensive track, 2 tiers (×1.20 attack power each), prices 4 and 6 resources, bought automatically at the end of the last turn away (Recall or death) |
| Gold ledger | reward only | reward unchanged (earned gold); a separate spendable-gold state is reduced by purchases |
| Tower | per-crash plate opportunities that lapse | persistent tower damage per tower; minions (Lanchester damage per tick) and Apply Pressure (attack-power-scaled) both contribute; 3 plates at thresholds 16 / 36 / 60 (durabilities 16, 20, 24); 1 resource per plate when its threshold is crossed |
| Actions | 46 codes | **the same 46 codes**; no purchase or other new action |

The most important changes from M2/M3 are four:

1. The time model.
2. Freeze becomes a maintained, economically consequential stance.
3. Recall gains a purpose: it buys power in a continuing game.
4. Crashes leave persistent damage.

---

## 3. State

### 3.1 Components

The M4 environment state at the start of turn k is

```
S = (Λ, φ, C_A, C_B)        with the per-game constants (m_A, m_B) (mechanical skills)
```

- **Λ, the lane.** The existing `State` (state.py:31–43): per side, the front `(fx, fh)`, the rears, the siege mass
  `sg`, the vision timer `v`, and the plate field `p`. The plate field is always 0 under M4 (§11.4).
- **φ, the phase.** φ = k mod P, P = `phase_period` = 4: the spawn timing and the Slow-Push creep timing. The
  absolute turn k is **not** part of the state.
- **C_s, the champion and progression record of side s** (own perspective):

| Field | Symbol | Type and range | Meaning |
|---|---|---|---|
| `pos` | c_s | int, −2 … 2 | champion cell (−2 while away) |
| `hp` | h_s | int centi-HP, 0 … H(L_s) | current HP; 0 only while dead |
| `absent` | a_s | int, 0 … 3 | turns still to sit out (M2 semantics) |
| `xp` | x_s | int, 0 … X(18) = 544 | experience (clamped at the cap) |
| `items` | N_s | int, 0 … 2 | purchased upgrade tiers |
| `gold` | g_s | int μ (10⁻⁶ resource), 0 … G_rem(N_s) ≤ 10⁷ | spendable gold (§8.6) |
| `tower_dmg` | τ_s | int centi-units, 0 … θ₃ = 6000 | cumulative damage side s has dealt to the **enemy** tower |

The level is derived, not stored: `L_s = level_of(x_s)` (§9.1). So are the max HP `H(L_s)`, the attack power
`AP(L_s, N_s)` and the plates broken `π(τ_s)`.

### 3.2 Initial state

The initial state has:

- the lane `initial_state(cfg)` (one fresh wave at each base);
- φ = 0;
- for each side: c = −2, h = H(1) = 600, a = 0, x = 0, N = 0, g = 0, τ = 0.

### 3.3 Mirror

`mirror(S) = (mirror(Λ), φ, C_B, C_A)`. Under `swap_seats` the skills swap with it. Every rule below is written for
a generic side s in its own coordinates, so the game is mirror-equivariant by construction (§14).

### 3.4 Invariants of a valid state

- 0 ≤ h ≤ H(L).
- h = 0 ⇔ (a > 0 and the champion is dead).
- x ≤ 544.
- N ∈ {0, 1, 2}.
- 0 ≤ g ≤ G_rem(N) := Σ_{i > N} price_i.
- 0 ≤ τ ≤ θ₃.
- The champion is at c = −2 whenever a > 0.

These are checked after every step in the soak tests.

---

## 4. Actions and legality

There is **no change to the action space or the legal sets.** The action code is
`a = wave + 3·behavior + 15·objective` (config.py:92–119), plus ABSENT = 45. Its components:

- wave: SP = 0, SH = 1, F = 2;
- behavior: HOLD, YIELD, TRADE, SAFE, and RECALL (only as code 12);
- objective: NONE, PRESSURE, VISION.

Legality is `champion_legal_actions` (champions.py:235–245):

- Freeze is legal iff the enemy front stands at own x ≤ −1.
- Siege objectives are legal iff the side has crashed mass under the enemy tower and its champion can reach the
  enemy tower this turn.
- Recall is legal iff the champion is present at x ≤ −1.
- ABSENT is the only legal code while a > 0.

Items add **no action**. Purchases are automatic (§8.5), so the purchase decision is the Recall decision itself.
That avoids a purchase-timing encoding and its duplicated or degenerate codes (§8.5 explains why automatic
purchase loses nothing).

---

## 5. Resolution of one turn

The resolution builds on M2's `resolve_turn` (champions.py:413–566). Steps 0–10 keep M2's order. **Bold** marks
what M4 changes or adds. Both actions are committed before anything is resolved; nothing reveals one player's
action to the other.

| Step | What happens |
|---|---|
| 0 | **Away sides** (M2). |
| 1 | **Probe** (M2). The lane is stepped once with every present champion fully supporting its wave. This gives each farm spot. The raw wave codes are used, so a Freeze is treated as intended. |
| 2 | **Targets and movement** (M2), with the **contact-cell chase rule** (§7.4). |
| 2b | **Recall interruption** (M2). |
| 3 | **Evasion** by Yield/Safe (M2). |
| 4 | **Contact and connection** (M2). |
| 5 | **Modes and reach** (M2). Then **Freeze maintenance** (§6.3): `M_s` is computed, and the effective wave is `w̃_s = F if M_s, SP if w_s = F and not M_s, w_s otherwise`. |
| 6 | **Lane step**: `step_turn` with the effective waves `w̃`, the effective objectives and the reach. It also returns the **tower hits** (§11.2). |
| 7 | **Gold** (M2 `champion_collection`, unchanged): `G_s`. |
| 7b | **XP**: `ΔX_s` (§7.3). |
| 8 | **Exposure and gank occurrence** (M2). |
| 9 | **Chance events**: trade hits, trade double hits, tower shots (dive or **adjacent**, §8.4), gank occurrence and gank trials, all from the same 10 uniforms as M2 (§8.7). **Damage in centi-HP** (§8.3). All damage of the turn is simultaneous and summed. |
| 10 | **Death**: a present champion with `h − damage ≤ 0` dies (h = 0, a = death_turns, c = −2). The opponent earns the kill bounty. |
| 11 | **Tower damage and plates** (§11): `τ_s ← min(θ₃, τ_s + 100·hits_s + [effective PRESSURE_s]·P(L_s, N_s))`. Each threshold crossed pays `v_plate` to s. |
| 12 | **XP and level-ups** (§9.2): `x_s ← min(544, x_s + ΔX_s)`. A champion that is alive and present gains `H(L') − H(L)` current HP; a dead champion's HP stays 0. |
| 13 | **Gold ledger**: `g_s ← g_s + G_s + Π_s + B_s` (farm, plate gold, bounty), then clamped to `G_rem(N_s)` (§8.6). |
| 14 | **Absence bookkeeping and return** (M2 `_after_away`). A champion whose timer reaches 0 at the end of this turn **makes its purchases (§8.5)**, then gets `h = H(L)` (with its current level) and plays the next turn at −2. |
| 15 | **Reward**: `r_A = (G_A + Π_A + B_A) − (G_B + Π_B + B_B)`, and `r_B = −r_A`. |

Ordering decisions, all deliberate:

- **Death before level-up (10 before 12).** A lethal turn is lethal even if the dying champion's XP crosses a
  threshold in the same turn. A turn's damage is resolved against the HP the champion had while the damage was
  dealt; this avoids "a level-up saves you" ambiguity in a one-step simultaneous resolution. The dead champion
  keeps its XP and levels; it respawns at the new maximum.
- **Every power-dependent quantity of a turn uses the levels and items held at the start of the turn:** trade
  damage, gank and tower-shot damage, and Pressure tower damage. Level-ups (step 12) and purchases (step 14) take
  effect from the next turn. The turn's resolution therefore never depends on the order in which the two sides'
  XP is credited.
- **XP of the death turn is kept,** as gold already is in M2.
- **Gold is credited before the purchase step (13 before 14).** Plate gold earned while away is spendable on
  return.
- **A recalling champion leaves at once and earns neither gold nor XP** on its Recall turn (M2 semantics). An
  *interrupted* Recall (channeling) is present: it earns XP, but no gold.

---

## 6. Wave management: Freeze, Slow Push, Shove

The lane mechanics stay those of `mechanics.step_turn` with the siege-model defaults:

- 4-HP waves every 2 turns;
- the `block` cap rule, with a cap of 12 HP;
- Lanchester minion damage max(1, ⌊h/4⌋) per tick;
- tower fire of 1 HP per tick on crashed mass;
- defenders waiting behind their tower (`siege_engage = False`).

What changes is who can sustain a Freeze, and why each action has economic consequences.

### 6.1 Slow Push (SP, unchanged)

The front walks to the lane cell at full speed. Past it, it stops while unstacked and creeps one cell per
`spawn_every_turns · slowpush_creep_period` = 4 turns once stacked (> 4 HP). Its effect is to build a wave: the
reinforcements merge (up to 12 HP) and later creep toward the enemy tower, giving a large crash. That crash is now
worth persistent tower damage (§11): a 12-HP crash alone deals 18 minion-damage units = 1,800 centi-units, enough
to break the first plate. Slow Push is the "future pressure" action.

### 6.2 Shove (SH, unchanged rules)

- The front walks without limit.
- When the champion's reach covers the fight, it adds `shove_damage` = 1 HP per tick to it.
- It adds the freeze-break push of 4 when the reach covers the front.

Its effect is to clear faster (more gold opportunities now) and to carry the fight to the enemy tower:

- If the defender has a fresh wave at its tower, the fight happens in the defender's tower cell. The shoved wave
  dies there (tower damage +1 per tick) and does not crash.
- If the defender has no fresh wave there, it crashes early with a small wave. That is less tower damage per crash
  than a stacked wave, but sooner, and it is followed by a bounce. An uncontested Shove is
exactly as effective as in M2.

### 6.3 Freeze (F): the maintenance rule

Freeze keeps its legality (enemy front at own x ≤ −1), its own-front rule (do not advance past ℓ_F = −1) and its
hold rule: the enemy front may not enter the freezer's tower cell unless `blob + push ≥ θ_F = 10`.

**New:** both rules apply only while the freezer *maintains* the freeze:

```
M_s = [w_s = F] ∧ ¬away_s ∧ ¬interrupted_s ∧ ¬connected_s ∧ |c'_s − ℓ_F| ≤ r_gold
```

`c'_s` is the freezer's end cell (step 2–3) and `r_gold` = `farm_reach` = 1. Put plainly, the freezer must:

- be on the lane (not away);
- not be channeling a Recall;
- not be trading (a connected Trade);
- stand within last-hitting reach of the freeze point: cells −2, −1 or 0.

A Freeze that is not maintained resolves as Slow Push (the effective wave `w̃`, step 5).

Why this is the minimal change:

- **It ties the freeze to the champion's presence and attention.** In the existing code the hold ignores where the
  freezer's champion is (mechanics.py:252–260). In M2 only an *absent* champion loses it (forced SP code).
- **It creates the interaction the brief asks for.** Trading the freezer does not break the freeze. Making the
  freezer trade back does. A freezer that steps forward beyond cell 0 to zone, or that chases, gives up the hold
  for that turn.
- **It adds no action and no penalty.** It only conditions an existing effect on the freezer's own decisions.

### 6.4 What a held Freeze does

**The parked configuration.** The enemy front stands at the freezer's −1 with `e` HP. The freezer's fresh waves,
spawned at −4, arrive at −2 two ticks later and fight it there. The enemy's reinforcements merge into its front
(queued HP counts toward `blob`, mechanics.py:167–178).

**Measured duration** over 24 turns; the freezer's front was just killed, all four start phases; lane-only
mechanics. "Crash" = the enemy front enters the freezer's tower cell.

| Enemy front e, enemy action | Freezer plays Freeze (maintained): first crash | Freezer plays Slow Push: first crash |
|---|---|---|
| 2, Slow Push | never; the front is destroyed within 3–4 turns and Freeze becomes illegal | never |
| 3–4, Slow Push | never: parked all 24 turns | never: parked all 24 turns (the configuration is lane-stable) |
| 5, Slow Push | never | turn 0 or never (phase-dependent) |
| 6–7, Slow Push | turns 7–12 (the blob grows and breaks the hold) | turns 0–15 |
| 8–9, Slow Push | turns 3–8 | turns 0–3 |
| 2, Shove | never (24 turns) | immediately |
| 3–5, Shove | turns 3–6 | immediately |
| ≥ 6, Shove | immediately (6 + 4 ≥ 10) | immediately |

So a Freeze is **transient against a growing wave** and **decisive against a small shoved or medium creeping
wave**. It cannot hold an enemy wave of ≥ 10 HP at all, nor a 6–9 HP wave for long. That is the "cannot maintain
every wave indefinitely" property, and it follows from the existing break threshold.

**What the hold changes compared with Slow Push** (enemy front held at −1, freezer at −2 or −1):

- **Crashes are denied or delayed (table above), so no tower damage and no plates for the opponent (§11).** This
  is Freeze's distinct effect. It is nearly worthless under M2 and valuable under M4's persistent tower damage.
- **The opponent's gold stays next to the freezer's tower.** This also happens in the parked configuration under
  Slow Push (§1.1), so it is not a Freeze effect, but Freeze *prolongs* it.
  - The freezer's minions die at the freezer's −2, which is the opponent's +2, under the freezer's tower.
  - The opponent collects them only from its +1 or +2 (gold reach 1), at base efficiency 0.5 (kills inside the
    enemy tower cell, config.py:256), zoned by 0.3 when the freezer is in farm mode within reach.
  - The freezer collects the opponent's minions at its −1 from −2 or −1 at 0.9.
- **The opponent's XP is not moved out of reach.** Kills at the opponent's +2 are within XP range 3 of every
  cell ≥ −1 (§7).
- **The opponent's exposure rises.** Its gold cell +1 has:
  - gank occurrence 0.12·(3/4) = 0.09 plus the behavior term, vs 0.03 at −1;
  - per-trial gank damage 0.65 vs 0.35, or 0.15 if defensive at −1;
  - a tower-adjacent shot (§8.4) if it trades back from there.

**Per-turn gold accounting in the parked configuration**, whichever wave action sustains it (4 HP of each
side's minions die per cycle; 1 resource per 4 HP):

| Opponent's choice | Opponent gold per turn | Freezer gold per turn | Relative, to the freezer |
|---|---|---|---|
| Hold at +1 (zoned) | 0.5·0.7·0.5 = 0.175 | 0.9·0.7·0.5 = 0.315 | +0.14 |
| Safe at −1 (XP only) | 0 | 0.9·0.5 = 0.45 | +0.45 |
| Reference: both Hold at the lane | 0.315 | 0.315 | 0 |

So at equal power the parked configuration is worth +0.14 to +0.45 resources per turn to the defender, depending
on whether the opponent accepts the forward farming position.

**This is a calculation for one configuration under fixed behaviors, not an equilibrium claim.**
- The pushing side should not leave its wave parked there: it can push it in (Shove) or let it bounce.
- Freeze is what stops it from pushing a 5–9 HP or small shoved wave *in*, at the moment when doing so would deal
  persistent tower damage.
- The opponent can still break the freeze (§6.5), and the defender may prefer to push.

### 6.5 Breaking a Freeze

**The rule.** The opponent breaks the hold when `blob + 4·[Shove ∧ reach covers its front] ≥ 10`:

- With a front of 6 HP or more, a Shove whose champion is in reach breaks it at once.
- With 3–5 HP it breaks after 3–6 turns (measured, §6.4). The Shove adds 1 HP per tick to the fight, so the
  freezer's fresh 4-HP waves die faster and the front grows until blob + 4 ≥ 10.
- With 2 HP it does not break within 24 turns.

**Reach requirement.** The breaker's champion must reach its front at its own +1. In farm mode its reach is its end
cell + 1, so it must stand at 0 or +1.

**Exposure while breaking.** At +1 it stands in the freezer's trade range. If the freezer is at −1 it is in the
same global cell. If it trades back from there, it draws the tower-adjacent shot. Its gank occurrence is about
0.12 per turn (position 0.09, Hold 0.01, Shove 0.02).

**No automatic penalty.** Nothing in the rule damages the breaker, and its farm is computed as for any other
champion at its cell. Punishment requires the freezer to Trade (giving up its own maintenance and half its farm
for that turn) or a gank to occur.

### 6.6 Does Freeze become strategically meaningful? Prototype evidence and the pre-training gate

Maintenance and persistent tower damage give Freeze a mechanism. They do not by themselves show that Freeze
matters. That needs multi-turn comparisons, from reachable states, of Freeze against its realistic alternatives.
The full gate is a pre-training requirement (plan §7, S3). A **prototype of it** has already been run on the
unchanged engine (`results/m4_design/scripts/freeze_gate.py`, output `freeze_gate.txt`).

**Prototype method.**

- **Engine.** The M2 champion resolver supplies positions, farm spots, evasion, the lane and gold. On top of it the
  prototype adds the M4 accounting: Freeze maintenance (re-resolved as Slow Push when not maintained), XP within
  3 cells, tower hits plus Pressure damage, and plates at 16/36/60.
- **No chance events.** There are no trades, no ganks and no deaths. Gank *exposure* is reported as expected
  occurrence and expected HP loss, outside the return.
- **Scenarios.** 113 distinct Freeze-legal states for A (lane, champions and phase), reached from s₀ by 6
  scripted-shove prefixes and 400 seeded random legal prefixes.
- **Policies.**
  - The defender A plays **freeze** (Freeze when legal, maintained), **slowpush**, or **cshove** (conditional Shove:
    Shove while its own wave is at least as large as the enemy's).
  - The attacker B plays **park_hold** (Slow Push, Hold), **park_safe** (Slow Push, Safe), **breaker** (Shove,
    Hold), **presser** (Shove, Hold, Pressure when legal) or **stacker** (Slow Push until 8 HP, then Shove;
    Pressure).
  - Every pair is played for 120 turns, scoring the discounted relative return to A (gold plus plate gold,
    γ = 0.99).

**Results** (means over the 113 scenarios; "F vs best" counts the scenarios where Freeze's return beats or loses to
the better of Slow Push and conditional Shove by more than 0.05):

| B policy | Freeze | Slow Push | cond. Shove | F − SP: mean (min, max) | F vs best: better / equal / worse | B crashes per 120 turns (F / SP) | B plates (F / SP) | B gold (F / SP) | B XP (F / SP) |
|---|---|---|---|---|---|---|---|---|---|
| park_hold | 8.05 | 6.09 | −4.93 | +1.96 (0.00, +11.75) | 55 / 58 / 0 | 3.57 / 3.78 | 1.27 / 1.14 | 24.7 / 27.3 | 451 / 460 |
| park_safe | 30.15 | 29.62 | 7.35 | +0.54 (−2.60, +2.44) | 46 / 62 / 5 | 3.57 / 3.78 | 1.27 / 1.14 | 0.0 / 0.0 | 450 / 447 |
| breaker | 10.72 | 10.30 | 8.54 | +0.42 (0.00, +1.25) | 66 / 47 / 0 | 0.61 / 0.28 | 0.00 / 0.00 | 19.7 / 19.9 | 480 / 479 |
| presser | 10.18 | 10.33 | 8.57 | −0.15 (−1.10, +0.35) | 18 / 47 / 48 | 0.61 / 0.28 | 0.58 / 0.02 | 19.7 / 19.9 | 480 / 479 |
| stacker | 6.38 | 6.18 | −2.09 | +0.20 (−1.94, +9.08) | 12 / 58 / 43 | 8.39 / 5.81 | 1.27 / 1.14 | 26.0 / 26.6 | 479 / 448 |

**Readings, each a mechanical observation from these runs:**

1. **Freeze is no longer inert.** It differs from Slow Push in 45–58% of the reachable Freeze-legal scenarios over
   120 turns. The diagnostics' one-turn test found 0%.
2. **Freeze is advantageous against passive and pure-shove opponents.**
   - Against park_hold it is never worse and is better in 55 of 113 scenarios (up to +11.75), when the enemy front
     at A's −1 holds 4–8 HP. It denies creep crashes and plates and keeps B farming under A's tower: B's gold is
     −2.7 per 120 turns, while B's XP is nearly unchanged (451 vs 460). That is the "lose gold, keep XP" pattern.
   - The opponent's exposure rises with it: B's expected gank-HP loss is 24.8 vs A's 3.2 per 120 turns, which is
     outside the return.
3. **Shove breaks Freeze, and breaking it pays when combined with Pressure.**
   - Against breaker and presser, Freeze *invites* crashes: 0.61 vs 0.28 per 120 turns, with a median first crash
     at turn 7 vs never. Holding one's wave at −1 moves the fights to one's own tower, where a reaching Shove's
     +4 push breaks through.
   - A pure breaker does not exploit this, so Freeze is still +0.42.
   - A presser converts it into tower damage (τ 1,150 vs 231) and plates (0.58 vs 0.02). Against the presser,
     Freeze is worse than the best alternative in 48 of 113 scenarios (down to −1.10).
4. **Freeze is not beneficial against a stacking wave in most of the scenarios where it matters.**
   - Holding a 6–8 HP front lets it grow into larger crashes: 8.39 vs 5.81 crashes, worse in 43 scenarios, −1.2 to
     −1.9.
   - It helps only where it denies the stack's first crash, at 4–6 HP and particular phases: +8 to +9 in a few
     scenarios.
5. **Freeze is irrelevant for parked fronts of ≤ 3–4 HP against passive play** (difference 0.00). The
   configuration is lane-stable either way (§6.4).
6. **No action dominates.** Freeze's sign depends on the opponent's policy, which is the precondition for a mixed or
   state-dependent equilibrium role. Conditional Shove by the defender is the worst defender policy here: it gives
   up the tower-side farming position.

**Limits of the prototype.**

- **No trades, no tower-adjacent shot, no realized ganks or deaths.** The opponent's forward exposure would make
  farming a freeze costlier, and the freezer's trades would add to the denial.
- **Scripted, non-adaptive continuations.** No policy switches to Pressure when it sees a freeze.
- **The defender always Holds.**
- **Scenarios from one family of prefixes.**
- **The +8 to +12 extremes are driven by plates,** so they are sensitive to the plate thresholds.

The prototype therefore shows that the proposed mechanics *can* create meaningful, non-dominant Freeze incentives.
It does not show that learned play will use them.

**Pre-training gate (plan §7, S3).** The prototype is re-run on the implemented M4 environment with:

- chance events (paired common random numbers);
- trades and the tower-adjacent shot;
- levels;
- adaptive scripted continuations.

It passes if:

- **G1 (distinct).** Freeze ≠ Slow Push (|Δ| > 0.05) in ≥ 20% of ≥ 100 reachable Freeze-legal scenarios, against at
  least two attacker policies.
- **G2 (non-trivial sign).** There exist attacker policies against which Freeze beats both alternatives by ≥ 0.2 in
  some scenarios, *and* attacker policies against which it is ≥ 0.2 worse than the best alternative in some
  scenarios.
- **G3 (breakable).** Against a reaching Shove with a front of ≥ 3 HP, a maintained Freeze is broken within the
  windows of §6.4–6.5.

**Reported, not gated.** For each scenario and policy pair:

- crash delay;
- gold denial (B's gold, Freeze − Slow Push);
- XP of both sides;
- exposure (expected gank occurrence and HP);
- tower damage and plates;
- discounted return.

No threshold is placed on how often Freeze *should* be chosen.

**If the gate fails.** G1 failing means Freeze is still inert. G2 failing means it dominates or is never useful.
Either is reported as a design finding before training, and the smallest corrections are considered in this order:

1. **Allow the freeze to be set up one cell earlier** (`freeze_legal_max = 0`, enemy front at the lane cell), keeping
   maintenance. Reason: the gate failing G1 would mean the legal window is too short. Earlier legality lengthens it
   without a new rule, and it is redundant only in contact states (§1.1).
2. **Apply the siege-wait correction (D7).** Reason: bounces then reach the defender's side sooner, so Freeze-legal
   states occur more often.
3. **Adjust p_adj.** Reason: it is the only parameter that changes the cost of contesting a freeze without changing
   the wave rules.

Each correction is justified only by the specific failure, and each is measured again by the same gate.

---

## 7. Positioning, farming (gold) and experience (XP)

### 7.1 Kill positions

`TurnResult.kills[s][x]` is the HP of the enemy's minions that died this turn at own-perspective cell x of side s,
from every cause:

- lane fights;
- tower fire on crashed mass;
- defenders fighting a siege;
- thinning.

Each HP is recorded exactly once (mechanics.py:38–46). M4 reads the same record for gold and for XP.

### 7.2 Gold (unchanged from M2)

```
G_s = f · Σ_x kills_s[x] · e_s(x)        with f = farm_per_hp = 0.25 resources per HP
e_s(x) = champion_collection(x, mode_s, c'_s, mode_ō, c'_ō)   (champions.py:249–257)
```

That is:

- e_s(x) = 0 if s is away or channeling, or if |x − c'_s| exceeds the reach. The reach is 1, or 2 for a
  threatened Yield.
- Otherwise e_s(x) is the posture efficiency times (1 − zone). The zone is 0.3, or 0.15 for kills in the
  collector's own tower cell, when the opponent is in farm or trading mode within 1 of the kill.
- A connected trader collects ×0.5.

The **gold reach is 1 cell.**

### 7.3 Experience

```
ΔX_s = ξ · Σ_x kills_s[x] · [ |x − c'_s| ≤ R_xp ] · present_s
present_s = ¬away_s          (a channeling champion is present; a recalled, absent or dead one is not)
ξ = xp_per_hp = 2 XP per HP,  R_xp = 3 cells
```

XP has no efficiency factor, no zoning and no trade factor. It needs only presence within 3 cells. XP is
independent of gold: a champion can earn XP while earning no gold.

**Why R_xp = 3.**

With gold reach 1 and the M2 behaviors, R_xp is the smallest range for which the existing defensive positions keep
XP in the freeze geometry:

- Safe stands at the safe line, −1.
- A threatened Yield retreats out of the trader's range, also to −1 against a freezer at −1.

The freeze kills happen at the opponent's +2, three cells from −1. With R_xp = 2:

- the only XP-without-gold cell would be 0;
- no behavior targets that cell;
- every defensive choice would forfeit XP automatically.

With R_xp = 3, losing XP requires being pushed back to one's own tower cell (−2, four cells from the kill). Against
a freezer at −1, that happens only if the freezer advances to 0 to threaten the cell −1. In passive lane play
(kills at the lane cell) ranges 1, 2 and 3 are the same: 1.88 XP-HP per turn, measured.

### 7.4 Contact-cell chase rule (replaces M2's chase target)

A Trade champion s without an objective, whose opponent is present, chases the opponent's reference cell r (as in
M2: the opponent's intended end cell, or its start cell if it also trades). Its target is

```
target_s = argmin_{x ∈ X(r)} |x − c_s|,   X(r) = { x ∈ [−T, T] : in_contact(x, r) },   ties → the smaller x
```

and it moves at most `champion_speed` cells toward the target. X(r) is never empty: for r = −T (opponent in its own
tower cell) X = {+T}, the dive.

This coincides with M2's rule in every case except the two in which M2's target was a cell from which contact is
impossible:

- a chaser targeting its own tower cell, when the target stands at its own +1;
- the mirror case, a chaser targeting the enemy tower cell while the target stands outside it.

The implementation test classifies every difference from M2 into these cases (plan WP4).

### 7.5 Zones under a held Freeze (five cells suffice)

The freezer's minions die at the opponent's +2. The freezer stands at −1, its own coordinates.

| Opponent cell (own) | Gold (reach 1) | XP (range 3) | Trade contact with a freezer at −1 / at −2 / at 0 | Tower |
|---|---|---|---|---|
| +2 | yes (0.5 base) | yes | yes / yes / yes | inside the enemy tower: a dive |
| +1 | yes (0.5 base) | yes | yes (same global cell) / no (tower-protected) / yes | adjacent: trading back draws the adjacent shot |
| 0 | no | yes | yes / no / yes | — |
| −1 | no | yes | no / no / yes | adjacent to its own tower |
| −2 | no | no | no / no / no (protected) | own tower |

This gives three regimes:

- **gold + XP** (forward, exposed);
- **XP only** (back, safe unless the freezer advances);
- **nothing** (own tower).

**Five champion cells are enough, and no grid change is proposed.** An alternative with a finer grid or ranges
(gold 0, XP 1) is in §16 (A3) and audit §7.

---

## 8. Combat: levels, items, damage

### 8.1 Separation of skill and economy

- **Mechanical skill** enters only the hit probabilities (M2, unchanged):
  `h_i = clip(0.55 + 0.4(m_i − m_j) + β[behavior_j] + 0.05(σ_i − σ_j) − 0.15·[j in own tower cell], 0.05, 0.95)`
  and `q_i = clip(0.3 + 0.2(m_i − m_j), 0, 1)` (champions.py:303–319).
- **Economic power** (level L, item tiers N) enters only the **damage of a successful hit** (attack power) and the
  **maximum HP** (level).
- Neither enters the other's channel.

### 8.2 Tables (fixed point; these exact integers are normative)

```
H(L)     = ⌊600 · 1.05^(L−1) + 1/2⌋                 centi-HP          (max HP; 6 HP at level 1)
u(L)     = ⌊H(L) / 6 + 1/2⌋                         centi-HP          (one "HP unit" at level L)
AP(L, N) = ⌊10⁴ · 1.05^(L−1) · 1.20^N + 1/2⌋         basis points      (attack power; 10⁴ = 1.0)
dmg(D, L, N) = ⌊(D · 100 · AP(L, N) + 5000) / 10⁴⌋   centi-HP          (a hit of D base HP, D ∈ {1, 2})
```

| L | H(L) | u(L) | AP(L,0) | AP(L,1) | AP(L,2) | dmg(1,L,0) | dmg(2,L,0) | dmg(1,L,2) | dmg(2,L,2) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 600 | 100 | 10000 | 12000 | 14400 | 100 | 200 | 144 | 288 |
| 2 | 630 | 105 | 10500 | 12600 | 15120 | 105 | 210 | 151 | 302 |
| 3 | 662 | 110 | 11025 | 13230 | 15876 | 110 | 221 | 159 | 318 |
| 4 | 695 | 116 | 11576 | 13892 | 16670 | 116 | 232 | 167 | 333 |
| 5 | 729 | 122 | 12155 | 14586 | 17503 | 122 | 243 | 175 | 350 |
| 6 | 766 | 128 | 12763 | 15315 | 18378 | 128 | 255 | 184 | 368 |
| 7 | 804 | 134 | 13401 | 16081 | 19297 | 134 | 268 | 193 | 386 |
| 8 | 844 | 141 | 14071 | 16885 | 20262 | 141 | 281 | 203 | 405 |
| 9 | 886 | 148 | 14775 | 17729 | 21275 | 148 | 296 | 213 | 426 |
| 10 | 931 | 155 | 15513 | 18616 | 22339 | 155 | 310 | 223 | 447 |
| 11 | 977 | 163 | 16289 | 19547 | 23456 | 163 | 326 | 235 | 469 |
| 12 | 1026 | 171 | 17103 | 20524 | 24629 | 171 | 342 | 246 | 493 |
| 13 | 1078 | 180 | 17959 | 21550 | 25860 | 180 | 359 | 259 | 517 |
| 14 | 1131 | 188 | 18856 | 22628 | 27153 | 189 | 377 | 272 | 543 |
| 15 | 1188 | 198 | 19799 | 23759 | 28511 | 198 | 396 | 285 | 570 |
| 16 | 1247 | 208 | 20789 | 24947 | 29937 | 208 | 416 | 299 | 599 |
| 17 | 1310 | 218 | 21829 | 26194 | 31433 | 218 | 437 | 314 | 629 |
| 18 | 1375 | 229 | 22920 | 27504 | 33005 | 229 | 458 | 330 | 660 |

The tables are computed once in double precision and stored as integers. The implementation must reproduce them
exactly (WP2 test); every later operation is integer arithmetic. The largest value anywhere is 33,005, so overflow
is impossible.

### 8.3 Damage

**Trade.** A connected trade by i on j draws D ∈ {0, 1, 2} with M2's probabilities `(1 − h_i, h_i(1 − q_i), h_i q_i)`
and deals `dmg(D, L_i, N_i)` centi-HP (0 for D = 0).

**Gank on j.** If it occurs, `X ~ Binomial(4, p_e)` trials (M2 parameters) deal `X · u(L_j)` centi-HP: each trial is
1/6 of the victim's max HP. The gank threat stays stationary over a continuing game, because the jungler is
exogenous and scales "with the game". Without this scaling, ganks would become steadily less lethal as levels
rise.

**Tower shot on j** (dive or adjacent). It deals `u(L_j)` centi-HP: one sixth of max HP, as in M2 at level 1, again
level-invariant.

**Simultaneous.** All of a turn's damage is summed and applied at once (M2). Both champions can die in the same
turn.

**Symmetry.** With equal levels and items, every damage-to-max-HP ratio equals M2's. The tables in §8.2 reproduce
M2 exactly at L = 1, N = 0.

**Outgoing scaling only.** M4 has no defensive item and no damage reduction. Max HP growth with level is the only
defensive progression. This is deliberate: it is the smallest model in which economic power matters. Effective
combat edge (A's expected damage as a fraction of B's max HP, divided by the converse; equal skill unless stated;
measured from the tables):

| Matchup | Edge | Duel edge P(A kills first) − P(B first), random starting HP 40–100% |
|---|---|---|
| +1 level | 1.10 | +0.12 |
| +2 levels | 1.21 | +0.18 |
| +3 levels | 1.34 | +0.29 |
| +1 tier | 1.20 | +0.19 |
| +2 tiers | 1.44 | +0.34 |
| +2 levels +1 tier | 1.46 | +0.34 |
| skill 0.6 vs 0.4, equal economy | 1.43 | +0.38 |
| skill 0.7 vs 0.3 against +2 levels +2 tiers | 1.18 | +0.23 |
| skill 0.6 vs 0.4 against +2 levels +1 tier | 0.98 | ≈ 0 (from full HP 0.449 vs 0.452) |

A one-level lead therefore matters, but less than a 0.2 skill gap. A mechanically stronger player can stay even
against a two-level, one-tier economic lead. **Breakpoint caveat (audit §1.3):** from *full* HP the duel outcome is
a step function of power, because a kill takes only 3–6 hits. +1, +2 and +3 levels give identical full-HP odds;
+1 tier at level 6 is worth +0.01. Averaged over mid-game HP states the effect is monotone, as the table shows. If
validation shows the steps dominate learned play, the documented fix is two half-strength rolls per trade (§16, A8).

### 8.4 Tower-adjacent shot (new)

A connected trader whose end cell is `T − 1` (adjacent to the enemy tower) takes a tower shot with probability:

- `p_adj` = 0.5;
- `p_adj_busy` = 0.25 if its own crashed minions were under that tower this turn. "Busy" uses M2's definition:
  siege at the start of the turn, or a crash during it.

The dive rule (end cell T: p = 1.0, busy 0.5) is unchanged. Only a champion that trades draws the shot; holding,
yielding or standing safe next to the enemy tower never does.

**Rationale.** It is the mechanism that gives a freezer next to its own tower a home advantage in trades. The
opponent farming the freeze at +1 can be traded but cannot trade back on equal terms. It applies to every trade
next to a tower, not only during Freeze, so it is not a Freeze modifier.

**Calibration status.** p_adj is a calibration parameter. Its ablation (on/off) is one of the planned budget-matched
studies (plan §7 S6).

### 8.5 Items, purchase and timing

**Items.** There is one offensive track, with tiers N ∈ {0, 1, 2}. Each tier multiplies attack power by 1.20,
through AP. **Items never change hit probabilities.** Prices are 4.0 and 6.0 resources, i.e. 4 and 6 fresh waves of
perfectly collected farm.

**Calibration of the prices.** At the measured collection rates of 0.20–0.29 resources per turn (§1.3), plus plates
and bounties:

- the first tier is affordable around turns 14–20, the M2 policies' typical first low-HP moment;
- the second is affordable around turns 35–50.

**Purchase rule (automatic, deterministic).** At step 14 of a turn at whose end the champion's absence timer
reaches 0, i.e. on the last turn away, after Recall or death:

```
while N_s < 2 and g_s ≥ price_μ[N_s]:     g_s ← g_s − price_μ[N_s];  N_s ← N_s + 1
```

`g_s` and the prices are integers in micro-resources (§8.6): `price_μ = (4,000,000, 6,000,000)`. The comparison is
exact, with no tolerance. The new attack power applies from the next turn, the champion's first back in the lane.

**Timelines** (M2 defaults recall_turns = 2, death_turns = 3):

- **Recall chosen on t:** away t and t + 1; purchase at the end of t + 1; play t + 2 with full HP H(L) and the new
  tier.
- **Death on t:** away t + 1 … t + 3; purchase at the end of t + 3; play t + 4.

**Why automatic purchase loses nothing.**

- The tiers are sequential and gold has no other use.
- Buying tier N + 1 as soon as one is at base and can afford it is therefore weakly dominant over waiting:
  - it never reduces a future option, because tier N + 2 still needs tier N + 1 first;
  - it only brings the power forward.
- An explicit BUY/SAVE action would be a dominated duplicate. It would add mask states and a degenerate "save"
  branch for PPO to learn away.
- The *meaningful* spending decision is **when to return to base**, i.e. the Recall timing, and that stays fully
  strategic. Mid-lane purchase is impossible.

**Death versus Recall as a way to shop.**

Deliberate death is never a cheaper way to the shop when Recall is available:

- a death costs 3 turns away against 2;
- it pays the opponent the kill bounty (0.5);
- it forfeits 3 turns of XP against 2.

Both routes restore HP and allow the purchase. The only path where death competes is a champion far forward at low
HP, which must spend one turn walking back (Safe/Yield) before Recall is legal. Walk-back plus Recall costs 3 turns,
of which the first still collects XP and some gold, and pays no bounty. Death costs 3 turns of nothing plus the
bounty.

So death is at best close to indifferent: when the walk-back turn itself is very dangerous, or when the kill bounty
is set to 0. The latter is why `kill_bounty` stays at its M2e value of 0.5 (audit §2.2).

### 8.6 Gold ledger (reward vs spendable)

**Earned gold** of side s on turn k: `E_s,k = G_s,k + Π_s,k + B_s,k` (farm, plate gold, kill bounty). It is the
reward source (§10.3) and it is never reduced.

**Spendable gold** is the state variable `g_s`:

- each turn it increases by `E_s,k`;
- it decreases only by purchases;
- it is clamped at `G_rem(N_s) = Σ_{i>N_s} price_i` (10.0, 6.0 and 0.0 resources for N = 0, 1, 2).

The clamp removes only gold that can never be spent, which keeps the state bounded. The amount clamped is reported
as `gold_unspendable` (an `info` statistic, not part of the reward).

**Spending is not a negative reward**, so nothing is double-counted. Identity checked per rollout:

```
Σ_k r_A,k = Σ_k E_A,k − Σ_k E_B,k
g_s = Σ_k E_s,k − Σ purchases_s − Σ clamped_s
```

**Representation: exact integers in micro-resources (μ = 10⁻⁶ resource).** Every earned amount is an exact multiple
of 1 μ:

- One HP of farm is worth `f · base · (1 − zone) · t`, a product of the defaults f = 0.25, base ∈ {0.9, 0.5, 0.8,
  0.6, 0.4, 0.2, 0}, (1 − zone) ∈ {1, 0.7, 0.85} and the trade factor t ∈ {1, 0.5}. Its finest value is
  0.25 · 0.9 · 0.85 · 0.5 = 0.095625 = 95,625 μ.
- Kills are whole HP.
- Plate gold (1.0) and the kill bounty (0.5) are whole numbers of μ.

So the ledger `g_s`, the cumulative earned gold and the reward are all computed as integers in μ. The reward is
converted to a float only when it is emitted: `r = (E_A,μ − E_B,μ) / 10⁶`.

- `validate()` converts each economic parameter to an exact rational from its decimal string. It **rejects** any
  configuration whose per-HP farm values, plate values or bounty are not integer multiples of 1 μ, so the exactness
  is enforced rather than assumed.
- Consequences: purchase decisions are exact integer comparisons; the ledger identities above hold exactly; the
  observation can encode g losslessly (§12.2).
- The legacy rulesets keep their float rewards, unchanged.

### 8.7 Chance vector

M4 uses M2's 10 uniforms per turn (champions.py:169–171) and draws them whatever happens:

- the tower-adjacent shot reuses `U_TOWER[i]`, since a trader is either at T or at T − 1, never both;
- gank damage uses `U_GANK_DAMAGE[i]` with the binomial inverse CDF.

Common random numbers across policies are therefore preserved, and `mirror_chance` is unchanged.

---

## 9. Experience thresholds and level-ups

### 9.1 Thresholds

```
XP to advance from L to L + 1:   n(L) = 8 + 3(L − 1)                      (L = 1 … 17)
cumulative XP to be level L:     X(L) = 8(L − 1) + 3(L − 1)(L − 2)/2      (X(1) = 0, X(18) = 544)
level_of(x) = max{ L ∈ 1..18 : X(L) ≤ x }
```

**Anchor.** A fresh 4-HP wave is 8 XP when all of it dies in range. League's XP to the next level is 280 + 100(L − 1),
and a League wave is ≈ 270 XP. Our cumulative thresholds in waves (X(L)/8) match League's (cumulative / 270) to
within 0.2 waves at every level:

| L | 2 | 3 | 6 | 9 | 11 | 13 | 16 | 18 |
|---|---|---|---|---|---|---|---|---|
| X(L) | 8 | 19 | 70 | 148 | 215 | 294 | 435 | 544 |
| waves, ours | 1.00 | 2.38 | 8.75 | 18.5 | 26.9 | 36.8 | 54.4 | 68.0 |
| waves, League | 1.04 | 2.44 | 8.89 | 18.7 | 27.0 | 36.9 | 54.4 | 68.0 |

The progression is therefore not calibrated to any turn count. It is calibrated to waves of minions, the
simulator's own economy. XP needed per level rises linearly (+3 per level), as in League.

### 9.2 Level-up HP rule

At step 12, let `L = level_of(x_before)` and `L' = level_of(x_after)`.

- **A present champion that survived the turn's damage:** `h ← h + H(L') − H(L)`. This adds the increase in max
  HP, never a heal beyond it. Several levels in one turn add their summed increase.
- **A champion that died this turn:** h stays 0. It respawns at H(L').
- **An absent champion:** it gains no XP (§7.3), so it does not level.

Invariant `h ≤ H(L')`: it holds because `h ≤ H(L)` before the step.

### 9.3 Expected progression speed and how to validate it

At the measured mutual-farming XP rate (1.88 HP per turn within range, i.e. 3.76 XP per turn, plus a 1–2 turn
opening delay):

| Level | 2 | 6 | 11 | 18 |
|---|---|---|---|---|
| Turn, mutual farming (94% absorption) | ≈ 4 | ≈ 20 | ≈ 59 | ≈ 147 |
| Turn, 75% absorption (deaths and recalls as in learned play) | ≈ 5 | ≈ 25 | ≈ 74 | ≈ 183 |

Levels 1–11 take ~60–75 turns and the cap ~150–180 turns. Both lie well inside the discounted objective's weight:
half the weight falls within 69 turns at γ = 0.99, and 87% within 200 (§10.2). Turn 32 has no special status:
level ≈ 7 at mutual farming (≈ 113 XP).

**Validation (plan §7 S2).**

1. 400 seeded 400-turn games of mutual `hold_farmer`. Record the turn each level is first reached and report the
   median and quartiles.
2. **Pass:** medians within ±25% of the table.
3. **On failure,** the single adjustable parameter is ξ (`xp_per_hp`, scaling every level time by 1/ξ). The miss is
   reported, not silently tuned.
4. Repeat with `aggressive_trader` vs `hold_farmer` and record the level gap it produces (expected: 1–2 levels by
   turn 60).

---

## 10. Time, discounting, reward and evaluation

### 10.1 The game is continuing

There is no terminal state:

- `terminated` is always False;
- the observation contains no absolute time or countdown, only the phase φ;
- all dynamics depend on k only through φ, which is already the case (mechanics.py:189, 440–441).

The state space is countable: the lane and champion fields are finite, and gold takes countably many values.
The action sets are finite and the per-turn reward is bounded: minion HP dying per turn is finite (at most a few capped
waves), at most 3 plates can fall per side, and there is at most one bounty per side.

So for 0 ≤ γ < 1 the discounted zero-sum game has a value V*(s), and both players have **stationary** optimal
strategies (Shapley 1953, which extends to countable state spaces for bounded rewards). Time-dependent strategies
are unnecessary. That is precisely why no time feature is needed.

### 10.2 Objective and the role of γ

```
J(π_A, π_B; s₀) = E[ Σ_{k ≥ 0} γ^k r_A,k | s₀ ],   γ = 0.99
```

**What γ changes.** M3 maximized the undiscounted sum of relative gold over 32 turns. M4 maximizes discounted
relative gold over an unbounded future:

- gold k turns ahead is worth γ^k of gold now;
- the weight is half gone by turn 69 (0.99^69 ≈ 0.5), and γ^200 ≈ 0.13, γ^460 ≈ 0.01;
- investments (Recall-to-buy, denying XP, building plates) are valued by their discounted future returns, not by
  how many of them fit before turn 32;
- there is no last turn, so there is no "consume everything before the end" incentive.

γ is **part of the game definition**, not only a training hyperparameter:

- equilibria at γ = 0.98 and γ = 0.995 may differ;
- PPO, every best-response attacker and every evaluator must use the same γ;
- if a lower training γ is ever used for variance reduction, the trained policy optimizes a different objective,
  and that must be reported.

Sensitivity values 0.98 and 0.995 are listed in §13.

### 10.3 Per-turn reward

```
r_A,k = (G_A + Π_A + B_A) − (G_B + Π_B + B_B),     r_B,k = −r_A,k
```

The terms are: farm gold collected (§7.2), plate gold (§11.3) and kill bounty.

Nothing else is rewarded:

- not XP or levels;
- not items, spending or unspent gold;
- not HP, tower damage short of a plate threshold, or Freeze, Recall, zoning or vision as such.

All of these are state. Their value is whatever future gold they cause, discounted.

### 10.4 Training semantics (PPO)

**Games.** A training game starts from a state drawn from the training start distribution ρ_train (below). It ends
only by a time limit `game_length` (default 512 turns). That is a **truncation**, not a game end: the last
transition's GAE target uses the critic's estimate `V̂(s_final)` of the continuation value. The existing code path
does this (train.py:288–297).

**Segments.** Rollouts are fixed-length segments of `segment_length` turns (default 128) per environment. A segment
that ends mid-game bootstraps `V̂(s_T)` (train.py:298–304). Games and segments are not aligned. The first game of
each environment gets a random length in [1, game_length], so the batch is not lock-stepped in game age.

**What bootstrapping does and does not guarantee.** With an exact critic (V̂ = V^π) the truncated and segmented
λ-return targets have the same expectation as the infinite-horizon targets. With an imperfect critic they do not. At
a truncation or segment boundary the one-step target's error is exactly `γ (V̂(s′) − V^π(s′))` for the boundary
state s′, and it propagates backward with weight (γλ)^j. So bootstrapping **approximates** the infinite-horizon
continuation value. It is the standard treatment of time limits in RL, and it is preferable to treating the limit
as termination, which would bias the target by `−γ V^π(s′)` and make the value depend on the unobserved time. It is
**not** unbiased. Three things keep the approximation honest:

- every boundary bootstraps from the true pre-reset state, never from the next game's first observation (plan WP1,
  tests T1.5–T1.8);
- the game length is ≫ 1/(1 − γ), so few transitions sit near a truncation;
- the evaluation (§10.5) never uses the critic for headline numbers.

**γ, λ.** γ = cfg.gamma = 0.99 and λ = 0.95. `compute_gae` is unchanged (ppo.py:88–100): termination drops the
successor value, truncation keeps it, and both stop the λ-recursion. M4 never produces `terminated = True`.

**Scale.**
- Typical per-turn rewards are ±0.1–0.3, so returns are of order ±10–30 resources.
- The guaranteed per-turn bound is |r| ≤ 8.575 (§10.5.3).
- Reward scale (0.1) and value clipping are retuned in the smoke stage.

**Training start distribution ρ_train (configurable, always reported).**

- **Default:** ρ_train = δ_{s₀}, every game starting from the standard initial state.
- **Optional mixture:** ρ_train = (1 − β)·δ_{s₀} + β·Uniform(𝒜_train), where 𝒜_train is an archive of recorded states
  (sieges, parked waves, level and item gaps).
- **Archive conditions:**
  - every archived state carries a witness: the seed and action prefix that reproduce it from s₀, so it is
    reachable;
  - the archive's generating policies and β are stated with every result.

ρ_train is a training device, **not part of the objective.** In the exact game an optimal stationary strategy pair
is optimal from every state simultaneously (§10.1), so a different start distribution does not change the set of
equilibria. With function approximation it changes which states receive gradient, and therefore which policy is
learned. That is the purpose of the archive: the diagnostics found rare states unvisited (D.4).

**What truncation does not do.** Truncation is not part of the game and is unobserved, so a policy cannot exploit
it. A policy that conditioned on "turns until truncation" is impossible: the observation has no time.

### 10.5 Evaluation: objectives, estimators and guarantees

#### 10.5.1 Two evaluation protocols, never mixed

- **E1 (primary): the game's objective.** J(π_A, π_B) = E[Σ_k γ^k r_A,k | s₀] from the standard initial state s₀.
  Every headline result, every policy comparison and every exploitability bound is E1.
- **E2 (secondary, diagnostic): value over a fixed held-out start archive.** `J_𝒜(π_A, π_B) = mean over s ∈ 𝒜_eval of
  E[Σ_k γ^k r_A,k | s]`.
  - 𝒜_eval is disjoint from 𝒜_train, fixed before training, published with its witnesses, and mirror-closed (it
    contains the mirror of each of its states, so a mirror-symmetric policy pair scores 0).
  - E2 measures play in rare situations that E1 under-samples.
  - It is reported separately and labelled. It is never averaged into E1 and never used to rank methods.

A policy trained with archived starts is still evaluated by E1 first. Training with ρ_train ≠ δ_{s₀} is a method
choice whose effect is judged by E1.

#### 10.5.2 Estimator

```
V̂_W = (1/n) Σ_episodes Σ_{k < W} γ^k r_A,k        (both seats, paired common random numbers, as in M3)
```

`V̂_W` estimates the truncated value `J_W = E[Σ_{k<W} γ^k r_A,k]`. The total error has two separately reported
parts:

- the Monte Carlo error (paired confidence interval);
- the truncation error `|J − J_W|`, which is bounded **deterministically** (§10.5.3).

#### 10.5.3 A simulator-derived truncation bound (mathematical guarantee)

All earned amounts are non-negative and `r_A = E_A − E_B`, so for any window

```
|Σ_{k≥W} γ^k r_A,k| ≤ max_s Σ_{k≥W} γ^k E_s,k .
```

Each of the three components of E_s is bounded from the rules (M4 defaults):

1. **Farm.**
   - One HP of the enemy's minions is worth at most f·max(base) = 0.25·0.9 = 0.225 resources.
   - The enemy HP that can die during turns W … W + m − 1 is at most its alive HP at turn W plus the HP it spawns
     meanwhile: ≤ A_max + 4⌈m/2⌉ ≤ A_max + 2m + 2.
   - A_max is the largest alive minion HP of one side (lane plus crashed mass) over all reachable states.
   - **By exhaustive enumeration of the reachable lane, A_max = 32** (`results/m4_design/lane_bounds.txt`). This
     is a computational proof for the default lane parameters; a test re-checks it whenever they change (plan WP3).
   - Summation by parts on the cumulative earnings `C_m ≤ a + b·m` gives
     `Σ_{m≥0} γ^{W+m} G_{W+m} = γ^W (1 − γ) Σ_{m≥1} γ^{m−1} C_m ≤ γ^W (a + b/(1 − γ))`, with a = 0.225·(A_max + 2)
     and b = 0.45.
2. **Kill bounty.**
   - A dead champion sits out death_turns = 3 turns and must play at least one turn before it can die again, so
     one side's bounties are at least 4 turns apart.
   - Hence `Σ_{k≥W} γ^k B_s,k ≤ γ^W · 0.5 / (1 − γ⁴)`.
3. **Plates.** At most 3 per tower, ever, so the plate gold still to come is at most `γ^W · 3 · v_plate`.

Hence

```
|J − J_W| ≤ γ^W · Ω,    Ω = 0.225(A_max + 2) + 0.45/(1 − γ) + 0.5/(1 − γ⁴) + 3·v_plate
```

At γ = 0.99, Ω = 68.3:

| W | 460 | 700 | 900 | 1,000 |
|---|---|---|---|---|
| guaranteed bound γ^W·Ω | 0.671 | 0.060 | 0.0081 | 0.0030 |

**Windows.**
- **Headline E1 numbers use W = 1,000.** The guarantee is then ≤ 0.0030 resources, below any Monte Carlo error we
  can afford.
- **W = 460 is allowed only for development runs.** There it must be reported with the guaranteed bound
  (0.671, loose) **and**, separately labelled as an empirical estimate, the tail measured on a subsample extended
  to W = 1,000. The measured tail is not a guarantee.

**Per-turn bound.** For value-scale design, a single turn's reward is bounded:

- **Farm per turn:**
  - one fight per tick removes at most the enemy front's HP, at most hp_cap = 12;
  - tower fire removes at most 1 HP per tick;
  - thinning removes at most 1 HP per turn;
  - there are no siege fights (`siege_engage = False`);
  - so at most 2·12 + 2 + 1 = 27 enemy HP die per turn, worth at most 0.225·27 = 6.075.
- **Plates per turn:** at most 2. One turn adds at most 2·100·minion_damage(48) + P(18, 2) = 2,400 + 1,320 = 3,720
  centi to τ, while crossing all three thresholds from below θ₁ needs more than θ₃ − θ₁ = 4,400.
- **Bounty:** at most 0.5.

So `|r_k| ≤ 6.075 + 2 + 0.5 = 8.575`. The enumeration's largest observed one-turn kill is 15 HP, within the
analytical 27. This per-turn bound is not used for the tail: γ^W·8.575/(1 − γ) = 857.5·γ^W is about 13 times
weaker than the cumulative bound.

#### 10.5.4 Bootstrapped evaluation (optional, approximate)

`V̂_W + γ^W V̂_critic(s_W)` can shorten W. It is **not** an unbiased estimate of J:

- its bias is `γ^W E[V̂_critic(s_W) − V^{π_A,π_B}(s_W)]`, unknown, and bounded only by the critic's accuracy;
- a critic trained in self-play estimates the value of its own policy pair, not of an arbitrary pairing.

It may be reported only as a labelled approximation, never for headline numbers or exploitability bounds.

#### 10.5.5 Descriptive statistics and exploitability

- **Descriptive statistics.** Report undiscounted relative gold per window (turns 0–63, 64–127, …), levels, items
  and plates over time. These are not the objective.
- **Approximate exploitability.** A PPO best responder trains on the same γ, games and segments against the frozen
  target, then is evaluated by E1 with W = 1,000. Its training start distribution is reported like the target's.
  Following the diagnostics lesson:
  - ≥ 2 warm-started and 1 from-scratch attacker per target, ≥ 1M steps each;
  - learning curves and the last paired increment are reported;
  - the maximum over attackers is a **lower bound**, never a certified Nash exploitability.
- **Equivalent estimator.** Geometric termination with continuation probability γ has the same expected
  undiscounted sum as the discounted objective. It is a valid, higher-variance cross-check.

### 10.6 Long-run behaviour with finite caps

Levels (18), items (2) and plates (3 per tower) are finite. In any play where both champions keep collecting XP
and gold, both sides reach L = 18 and N = 2 after finitely many turns, ~150–250 at typical rates. Plates are
exhausted once both towers' damage counters reach θ₃, or they may never be, if nobody sieges.

**The asymptotic game is the stationary capped lane game.** Once both sides are capped, the state reduces to the
M2-like lane plus capped champions and no plates remain. Its per-turn value contributes at most γ^{k_cap}/(1 − γ)
times its stationary mean. For mirror-symmetric policies that mean is 0.

**No state becomes invalid or unbounded:**

- x is clamped at 544;
- N ≤ 2;
- g is clamped at G_rem(N), so 0 after both tiers;
- τ is clamped at θ₃;
- H and AP are table look-ups.

A 2,000-turn soak test checks this (plan §4, test 11).

---

## 11. Tower durability and plates

### 11.1 State

`τ_s` (§3.1) is the cumulative damage side s has dealt to the enemy tower, in centi-units. It persists for the whole
game. Plates broken: `π(τ) = #{ i : τ ≥ θ_i }`, with thresholds `θ = (1600, 3600, 6000)`, i.e. durabilities 16, 20
and 24 minion-damage units, increasing.

### 11.2 Damage sources

- **Minions.** On every tick, the crashed mass already under the enemy tower at the start of that tick hits the
  tower for `minion_damage(m)` = max(1, ⌊m/4⌋) units. This is the same law as against minions, and the same
  start-of-tick convention as tower fire (mechanics.py:378–385). `TurnResult.tower_hits[s]` is the sum over the
  turn's ticks. A newly crashed mass hits from the next tick on. Totals for one undisturbed crash (tower fire 1 HP
  per tick):

| Crash size | Ticks under the tower | Tower hits (units) | Centi |
|---|---|---|---|
| 4 HP | 4 | 4 | 400 |
| 8 HP | 8 | 9 | 900 |
| 12 HP | 12 | 18 | 1,800 |

- **Champion (Apply Pressure).** An effective Pressure, i.e. legal and not suspended by a connected trade (M2),
  adds `P(L, N) = ⌊(400 · AP(L, N) + 5000)/10⁴⌋` centi per turn: 400 at L1 N0, 613 at L6 N1, 1,320 at L18 N2.
  Thinning (M2) is unchanged.

`τ_s ← min(θ₃, τ_s + 100·tower_hits_s + [effective PRESSURE_s] · P(L_s, N_s))`.

### 11.3 Plate gold

Every threshold crossed on this turn pays `v_plate = 1.0` resources to s: `Π_s = v_plate · (π(τ'_s) − π(τ_s))`. It
is paid whether or not s's champion is present, the simplest rule; the alternative is in §16. Plates never come
back, and there is no fixed-turn plate removal.

**Calibration intent:**

- a single small crash breaks nothing;
- a full 12-HP crash breaks the first plate on its own;
- 2–3 large crashes with some Pressure take all three.

Plates are therefore a reward for *repeated* or *stacked* pushing, accumulated persistently. In League, laning
typically takes 1–3 plates per lane, which this matches in order of magnitude. Turn-level calibration is plan §7
S3–S4.

### 11.4 Relation to the existing plate machinery

- **Crash-plate opportunities are off.** The per-crash opportunities of the siege model (`plate_mode="crash"`,
  `ap`/`bp`) are disabled under M4 by `plate_mode="durability"`: the lane step grants and claims no
  opportunities, and the `ap`/`bp` fields stay 0.
- **Vision and thinning are unchanged.**
- **What a Freeze stops.** It is the only way to stop minion tower damage before it starts, because a held enemy
  front never crashes.
- **What a crash buys.** A crash is worth persistent tower damage even if the attacker cannot stay to Pressure.

---

## 12. Observation and masks

### 12.1 Layout

The observation is own-perspective and fixed length, and every entry is in [0, 1]. It is a function of (φ, S,
side) only and contains no absolute time. Layout (88 entries with the default geometry):

| Block | Entries | Definition |
|---|---|---|
| phase | 4 | one-hot of φ = k mod 4 |
| own lane | 19 | **front cell** one-hot over cells −4 … 4 (all zero = no front); **front HP** / 12; **rear count per cell** n(x) / R_max for x = −4 … 4, with R_max = 4 |
| enemy lane | 19 | the same for the enemy, mirrored into own coordinates |
| own side | 3 | siege mass / 48; vision / 6; siege eligible |
| enemy side | 3 | the same for the opponent |
| own tower progress | 4 | τ_own / θ₃; plates broken π(τ_own)/3; progress to the next plate `(τ − θ_π)/(θ_{π+1} − θ_π)` (1 if all broken); plates remaining (3 − π)/3 |
| enemy tower progress | 4 | the same for τ_enemy (damage dealt to *our* tower) |
| own champion | 16 | cell one-hot (5); h/H(L); H(L)/H(18); absent/3; skill; (L − 1)/17; XP progress `(x − X(L))/n(L)` (0 at L = 18); N/2; **g_μ / 10⁷** (spendable gold, exact, §8.6); **can buy the next tier** `[N < 2 ∧ g ≥ price[N]]`; **can buy both remaining tiers** `[N = 0 ∧ g ≥ price[0] + price[1]]`; AP(L, N)/AP(18, 2) |
| enemy champion | 16 | the same, mirrored |

**Changes from the first M4.1 draft.**

- *Gold.* The draft feature `min(1, g/price[N])` was not injective. At N = 0, gold 4 and gold 9 gave the same
  value, although only 9 also pays for most of the second tier, so after one more wave of farm the two states buy
  different things. It is replaced by the exact balance `g_μ/10⁷`, which is lossless because g ≤ G_rem(0) =
  10 resources = 10⁷ μ, plus two exact affordability bits.
- *Lane.* M2's lane grid (HP per cell / 12, clipped at 1) is replaced by a lossless encoding: front cell, front HP
  and rear counts. §12.3 shows why.

**Removed relative to M2:**

- `k/K` and `(K − k)/K`, the countdown;
- `plates_pending`, which is always 0 under M4;
- `has_front`, which is now the "no front" pattern of the front one-hot.

**Mirror property.** `observe(Λ, φ, B, cfg, C) = observe(mirror(Λ), φ, A, swap_seats(cfg), mirror(C))`.

**Masks.** Unchanged: `legal_mask` = `champion_legal_actions`. No M4 rule adds or removes a legal code.

### 12.2 Markov sufficiency: what is proven and what is approximate

The M4 transition and reward depend only on:

- the state S = (Λ, φ, C_A, C_B) of §3.1;
- the per-game constants (the two skills, observed);
- the two actions;
- the i.i.d. chance vector.

So the observation is a sufficient statistic for decision-making **if S can be recovered from it**.

**Proposition (proven).** On every state reachable from s₀ under the M4 defaults, `observe(φ, S, side)` determines S
exactly. The proof is constructive: a decoder D reads S back from the float32 vector.

- **Phase:** φ is the index of the phase one-hot.
- **Lane, per side:**
  - *Front.* The front cell is the index of the front one-hot. Its HP is `round(12·v)`.
  - *Rears.* Rear positions come from the counts: `n(x) = round(4·v_x)`. Every rear packet holds exactly W = 4 HP;
    `step_turn` asserts this (mechanics.py:464–465).
  - *Siege and vision.* `siege = round(48·v)` and `vision = round(6·v)`. The plate fields are 0 under M4.
- **Champion, per side:**
  - *Integer fields.* `pos` from the one-hot; L = 1 + round(17·v); `absent = round(3·v)`; N = round(2·v).
  - *HP.* `h = round(H(L)·v)`. Distinct h at the same level differ by ≥ 1/H(L) ≥ 1/1,375 ≈ 7·10⁻⁴ in the feature,
    far above the float32 spacing (≤ 6·10⁻⁸).
  - *XP.* `x = X(L) + round(n(L)·v)`, with n(L) ≤ 56. Above the cap, x is clamped at 544 (§10.6), so nothing is lost.
  - *Gold.* `g_μ = round(10⁷·v)`. The float32 rounding error of g_μ/10⁷ ∈ [0, 1] is ≤ 2⁻²⁵ ≈ 3·10⁻⁸, i.e. ≤ 0.3 μ
    after rescaling. That is below the 0.5 μ needed for exact rounding, so every integer balance is recovered.
- **Tower:** `τ = round(6000·v)`, a spacing of 1.7·10⁻⁴.

**Two of these steps rest on bounds that are computed, not derived:**

- **R_max = 4 rear packets in one cell.** Measured as the exact maximum over the exhaustive reachable-lane
  enumeration (`results/m4_design/obs_alias.txt`).
- **siege ≤ 48.** `siege_cap` truncates any excess (mechanics.py:371–373).

The enumeration covers a superset of M4's lane steps. It uses every legal wave, the objectives NONE and PRESSURE, and
every champion reach value. M4's Freeze maintenance only substitutes Slow Push, and tower damage never feeds back
into the lane.

**What holds only approximately, or only conditionally:**

1. **Reachable states and default parameters.** R_max and the decoder's integer bounds were established for the
   default lane parameters. Any change of the lane geometry, wave HP, caps or spawn timing must re-run the
   enumeration test (plan WP7, test O1). The environment also checks n(x) ≤ R_max at run time and **raises** if it is
   exceeded. It never clips silently.
2. **Information versus learnability.** The encoding is lossless, but some distinctions are numerically tiny. Two
   gold balances 1 μ apart differ by 10⁻⁷ in the feature. A network cannot be expected to act on such differences,
   and should not need to: the strategically relevant gold comparisons (can buy now; can buy both) are exact bits.
   Markov sufficiency is proven for *information*. That a network *uses* that information is an empirical matter.
3. **Derived features.** AP, H(L)/H(18), the tower plate counts and the affordability bits are deterministic
   functions of other entries. They add no information and are included only to ease learning.
4. **Skills.** Skills are game constants, not state. They are observed exactly, so a single policy can condition on
   them; a game with unobserved skills would be a different, partially observable game.

### 12.3 Aliasing in the M2 observation (historical finding)

The exhaustive enumeration of the reachable lane (20,572 (state, phase) pairs) shows that **M2's lane grid is not
injective**:

- **Two causes.**
  - Rear packets can share the front's cell: in 96 side-states, rears queued behind a front that has just crashed
    become co-located with the new front.
  - A cell can hold up to 4 rears (16 HP), which the grid clips at 12 HP; this happens in 82 side-states.
- **Effect.** 96 observation classes contain more than one lane state. In 85 of the colliding pairs the states
  have **different transitions** for some action pair.
- **Scale.** This is 0.5% of the reachable lane classes, in rare queue-heavy states.

The M2 observation docstring's statement that "nothing decision-relevant is hidden"
(env/observation.py:26–28) is therefore false in these states.

The legacy observation is **not changed**, because changing it would break the pinned M2/M3 results and checkpoints.
The finding should be cited when interpreting M2 learned policies in queue-heavy (siege and bounce) states. M4 uses
the lossless encoding of §12.1.

---

## 13. Parameter table

These are defaults of `GameConfig.m4()`. All other fields keep their `champions()` defaults. "Exp." means the value
can only be fixed by experiment (audit §6).

| Parameter | Symbol | Units | Default | Plausible range | Justification |
|---|---|---|---|---|---|
| `ruleset` | — | — | `"m4"` | — | version switch; `"legacy"` everywhere else |
| `time_mode` | — | — | `"continuing"` | — | §10 |
| `gamma` | γ | per turn | 0.99 | 0.98–0.995 | half-weight at 69 turns covers levels 1–11; exp. sensitivity |
| `game_length` (training) | — | turns | 512 | 256–1,024 | ≫ 1/(1 − γ); bootstrapped, so not part of the objective |
| `segment_length` (training) | — | turns | 128 | 64–256 | PPO batch shape |
| `eval_window` | W | turns | 1,000 (headline); 460 (development only) | — | guaranteed truncation error ≤ 0.0030 (resp. 0.671) resources, §10.5.3 |
| `gold_unit` | μ | resource | 10⁻⁶ | — | every earned amount is an exact multiple (§8.6); enforced by `validate()` |
| `rear_count_max` | R_max | packets per cell | 4 | — | exhaustive maximum over the reachable lane (§12.2); exceeded → error |
| `xp_per_hp` | ξ | XP/HP | 2 | 1.5–3 | one wave = 8 XP; anchors §9.1 |
| `xp_range` | R_xp | cells | 3 | 2–3 | §7.3 (2 needs a new XP-keeping behavior) |
| `xp_base`, `xp_step` | — | XP | 8, 3 | — | League's 280 + 100(L − 1) per 270-XP wave |
| `level_cap` | — | levels | 18 | — | brief |
| `hp_base` | H(1) | centi-HP | 600 | — | M2's 6 HP |
| `hp_growth` | — | per level | 1.05 | 1.03–1.08 | brief (provisional); exp. |
| `ap_level_growth` | — | per level | 1.05 | 1.03–1.08 | brief (provisional); exp. |
| `item_mult` | — | per tier | 1.20 | 1.10–1.30 | brief; duel edge ≈ +2 levels per tier |
| `item_tiers` | — | — | 2 | 1–3 | brief; keep unless a defect shows |
| `item_prices` | — | resources | (4.0, 6.0) | (3–5, 5–8) | tier 1 at ~turn 15–20, tier 2 at ~35–50 (§8.5); exp. |
| `freeze_maintenance` | — | — | True | — | §6.3; False reproduces M2's unconditional hold (ablation) |
| `freeze_line`, `freeze_legal_max`, `freeze_break_threshold`, `freeze_break_push` | — | — | −1, −1, 10, (0, 4, 0) | unchanged | §6.4: small waves stay parked, 6–9 HP waves break within 3–12 turns, ≥ 10 HP cannot be held |
| `farm_reach` | r_gold | cells | 1 | — | unchanged (M2) |
| `trade_range` | — | cells | 1 | — | unchanged |
| `chase_rule` | — | — | `"contact_cell"` | `"m2"` | §7.4 (defect fix) |
| `tower_adjacent_p`, `tower_adjacent_p_busy` | p_adj | probability | 0.5, 0.25 | 0–0.75 | home advantage near one's own tower; exp. (ablation planned) |
| `tower_shot_damage` | — | u(L_victim) | 1 | — | M2, scaled |
| `gank_damage_scale` | — | — | `"victim_level"` | `"none"`, `"attacker_level"` | §8.3 |
| `plate_mode` | — | — | `"durability"` | — | §11 |
| `plate_thresholds` | θ | centi-units | (1600, 3600, 6000) | ±50% | §11.3 intent; exp. |
| `plate_gold` | v_plate | resources | 1.0 per plate | 0.5–1.5 | one wave of farm, as in the siege study |
| `pressure_tower_damage` | — | centi per turn at AP = 1 | 400 | 200–800 | ≈ one 4-HP crash per Pressure turn; exp. |
| `plate_gold_requires_presence` | — | — | False | True | §16 |
| `kill_bounty` | — | resources | 0.5 | 0.25–1.0 | M2e; must stay > 0 (§8.5) |
| `recall_turns`, `death_turns`, `recall_line` | — | turns / cell | 2, 3, −1 | unchanged | M2 |
| everything else (lane, trades, ganks, vision, thinning) | — | — | M2 `champions()` | — | unchanged |

---

## 14. Invariants

1. **Zero-sum.** `r_A,k + r_B,k = 0` for every turn, so the discounted values sum to zero.
2. **Reward decomposition.** `r_A,k = Σ_s ±(G_s + Π_s + B_s)` exactly, as reported in `info`. Purchases, XP,
   levels, HP and τ never appear in r.
3. **Ledger.** `g_s = Σ_k E_s,k − Σ purchases_s − Σ clamped_s`, and g ∈ [0, G_rem(N_s)].
4. **Monotone progression.** x, N and τ are non-decreasing. L = level_of(x), π = π(τ). Plate gold paid to s in
   total = v_plate·π(τ_s).
5. **Minion conservation.** As M2: spawned = alive + died, including tower and thinned kills; zero overflow under
   `block`. XP and gold are both bounded by `kills`: `ΔX_s ≤ ξ Σ_x kills_s[x]` and `G_s ≤ f Σ_x kills_s[x]`.
6. **HP.** 0 ≤ h ≤ H(L). h = 0 iff dead and absent. A return gives h = H(L).
7. **Mirror equivariance.** Resolving (mirror(S), b, a) with the mirrored chance vector and swapped skills gives the
   mirror of resolving (S, a, b). The same holds for observations.
8. **Skill/power separation.** `hit_p` and `double_p` are invariant to (L, N) of both sides. Damage values are
   invariant to skill.
9. **Phase periodicity.** Transitions from (S, φ) are identical for every k with k mod P = φ.
10. **Legacy pins.** All legacy outputs are bit-identical (plan WP0).
11. **Exact ledger.** Gold, earned gold and the reward numerator are integers in μ, so invariants 2 and 3 hold with
    equality, not up to rounding.
12. **Observation decodability.** `D(observe(φ, S, side)) = S` for every reachable state (§12.2).
13. **Lane bounds.**
    - The rear count per cell is at most R_max = 4, siege ≤ 48 and alive HP ≤ A_max = 32 per side. These are
      established by exhaustive enumeration and checked at run time.
    - At most 27 enemy HP die per turn (analytical; 15 observed).

---

## 15. Worked scenarios (the ten required cases)

Each scenario states what the rules imply mechanically. Where the outcome depends on strategy, it states the
trade-off and marks the behavioral claim as a hypothesis to test. **None of the emergent behaviors is guaranteed by
the rules.**

**1. Equal-power Freeze: the opponent loses gold but stays in XP range.**
- B's 3–5 HP front is parked at A's −1. This configuration is lane-stable (§6.4), and A's maintained Freeze
  additionally stops it from being pushed in. A stands at −2 or −1.
- B's minion income moves to its +2, under A's tower:
  - farming it from +1 yields at most 0.5·0.7 = 35% of it and costs exposure;
  - Safe at −1 yields no gold.
- B's XP is intact from either cell (kills at +2 are within 3 of −1).
- A gains +0.14 (B at +1) to +0.45 (B at −1) resources per turn relative to even lane farming, and B keeps full
  XP.
- **Mechanically guaranteed:** gold separates from XP here.
- **Not guaranteed:** that B accepts this rather than breaking the freeze (scenario 3), or that A's freeze state
  arises at all. Freezes require B to push first, e.g. after A crashes and B's wave bounces.

**2. Strong freezer: the opponent retreats and loses both.**
- A, ahead by 2 levels and 1 tier (duel edge +0.34 at mid-game HP), advances to 0 and Trades.
- Its chase reaches B at −1 (contact-cell rule). B's alternatives:
  - trade back at an unfavorable edge;
  - Yield or Safe: threatened, it evades to −2, its own tower cell, where kills at +2 are 4 cells away. No gold,
    no XP;
  - Hold: walks forward to its farm spot, the cell next to A's tower, and takes A's trades plus the adjacent-shot
    risk if it trades back.
- A pays for the denial:
  - its exposure at 0 is gank occurrence ≈ 0.06 + 0.03 (Trade) vs 0.03 at −1;
  - if B trades back, A's connected trade suspends its maintenance and halves its farm that turn.
- **Hypothesis to test:** B's retreat is rational only when A's edge is large. XP denial should then be
  concentrated in high-edge states (plan S4 counterfactuals).

**3. Weaker champion breaks a Freeze by shoving.**
- B (weaker) plays Shove + Hold. Its champion must stand at 0 or +1 for its push to count.
- A front of 6 HP or more breaks at once; a 3–5 HP front breaks after 3–6 turns (§6.4–6.5).
- B's exposure meanwhile: gank occurrence ≈ 0.12 per turn at +1; A's trades from −1, with B unable to trade back
  without the adjacent shot.
- If A does not trade (or misses) and no gank occurs, B breaks the freeze at no HP cost. Nothing penalizes "Shove
  against Freeze" as such.
- After the break B's wave crashes into A's tower. It deals 4–9 tower units and possibly starts a siege, which
  makes Pressure legal.
- **Hypothesis to test:** a weaker B breaks early, before its front shrinks, or not at all.

**4. Mechanical skill vs economic progression.**
- Skill moves hit probabilities; economy moves damage and max HP (§8.3 table).
- Skill 0.7 vs 0.3 still beats an opponent ahead by 2 levels and 2 tiers (+0.23 duel edge at mid-game HP).
- Skill 0.6 vs 0.4 is even against +2 levels +1 tier.
- Both channels are bounded:
  - the hit probability by [0.05, 0.95];
  - the economic edge by the caps, at most 1.05¹⁷·1.2² ≈ 3.3× damage against 1.05¹⁷ ≈ 2.3× HP, between a capped
    and a level-1 champion; between two champions at equal levels, at most 1.44 (two tiers).
- **Hypothesis to test:** learned policies trade more when ahead in (skill + economy). M2 found skill-conditioned
  aggression; M4 adds an economic axis.

**5. Low-HP champion: stay or Recall to buy.**
- Recall at −1 costs:
  - 2 turns away: own gold (≈ 0.29/turn), XP (≈ 3.8 XP/turn) and zoning; the opponent farmed ≈ 0.41/turn while
    one was away in M2 (diagnostics B);
  - wave and tower risk: the opponent may crash, now with persistent tower damage.
- It gains:
  - HP back to H(L) (M2 valued HP at ≈ 0.25 resources per HP point);
  - the next tier, if g ≥ price.
- The tier is otherwise bought at the next return to base. In M2 learned play the next death came every ≈ 16–19
  turns. Recalling now moves the power forward by roughly that long.
- **Not computable on paper:** the value of +20% damage over ~17 turns. It is measured in plan S4 (paired branches:
  recall with vs without an affordable tier, at HP 2, 4 and 6).

**6. Some recall timings pay, others do not.**
- Mechanically, the cost of a Recall depends on the lane at the moment. It is cheap when:
  - one's own wave is pushing or crashing: the opponent is busy, no enemy crash can happen while away, and one's
    own minions damage the tower meanwhile;
  - and the opponent is itself away.
- It is expensive when:
  - an enemy wave is about to crash into one's tower: minion tower damage, plates and the opponent's Pressure;
  - or the opponent is present to farm freely.
- The gain grows with missing HP and with an affordable tier.
- **Hypothesis to test:** the purchase does not make Recall dominant. Expected sign pattern: recall-to-buy positive
  at low HP with one's own wave pushing, negative at full HP with an enemy crash imminent. Plan S4 must find both
  signs, or the result is reported as a design finding before training.

**7. Repeated Shove: crash, bounce and durable plate damage.** Worked example, at levels 1–2:
- **A plain Shove does not crash by itself.** Measured lane-only: Shove against Slow Push gives 0 crashes in 64
  turns. The shoved wave meets the defender's fresh wave inside the defender's tower cell every cycle and dies
  there; the tower adds 1 HP per tick against it.
- Crashes need an asymmetry:
  - a stacked wave (Slow Push to stack, then Shove);
  - the defender's front killed with no fresh wave yet at the tower;
  - or the defender away, so no champion supports its wave. The M2 attackers' 0.8–1.5 crashes per game came
    mostly from timing shoves this way.
- Say A achieves one: its wave crashes with 2–4 HP after clearing B's front.
- Tower hits 2–4 units (200–400 centi). One Pressure turn adds 400. Now τ_A ≈ 800, half the first plate.
- B's waves wait behind the tower (`siege_engage = False`) while the crashed mass is cleared (1 HP per tick, so 1–2
  turns). The next spawn merges into an 8-HP blob, which then walks out, beats A's fresh 4-HP waves and pushes
  toward A's side. That is the bounce.
- When it reaches A's −1, Freeze becomes legal for A. But an 8-HP front grows under a freeze (§6.4), so A's freeze
  breaks within 3–8 turns.
- A's alternatives:
  - let it crash, giving B tower damage on A's tower;
  - Shove into it to shrink it.
- A second A crash adds to τ_A, which persists. The first plate (1,600) falls on the second or third crash with
  Pressure, and pays A 1.0.
- **Mechanical:** τ never decreases. Plates pay once. The bounce makes Freeze legal for the crasher. Whether
  repeated shoves are profitable depends on the gold, XP and HP spent at the forward position against the plate
  income: an empirical question. The diagnostics found siege plans profitable under M2's per-crash plates.

**8. Level-up during sustained combat.**
- A is at L5 with h = 150/729 centi-HP; B trades and lands an ordinary hit, 128 at B's L6 N0.
- A survives with 22. A's XP crosses X(6) = 70 this turn, so A levels to 6: h = 22 + (766 − 729) = 59/766.
- If B had landed a strong hit (255), A would be at 150 − 255 < 0. A dies, even though its XP also crossed 70:
  death first, §5. A respawns at L6 with h = 766 after 3 turns, and buys if it can afford it.
- Two levels in one turn (possible only from very low XP at high kill volume) add H(L + 2) − H(L).

**9. Continuing beyond turn 32: no last-turn artifact.**
- The rules have no turn index (only φ), the objective has no end, and the observation has no countdown. A
  stationary optimal strategy exists (§10.1), so no optimal play conditions on turn 32 or any other turn.
- The M2 "Shove in the last four turns" gain came from costs falling after a terminal state. In M4 those costs
  arrive, discounted, like any others.
- Training-game truncation at 512 turns is bootstrapped and unobserved, so it cannot be exploited.
- **Acceptance check (plan WP1):** in continuing-time M2 mechanics, the Shove rate of a trained policy is
  stationary across windows.

**10. Caps without overflow or invalid states.**
- After L18, x is clamped at 544 and further XP is discarded (`info` reports it). After N = 2 the clamp sets g = 0;
  further earnings remain rewards and are reported as `gold_unspendable`.
- After θ₃, τ is clamped and Pressure only thins.
- AP ≤ 33,005, H ≤ 1,375, damage ≤ 660 centi: everything fits in int16 except AP, which fits in int32.
- The observation stays in [0, 1]. The XP-progress and affordability features are defined as 0 at the caps.
- Soak test: 10,000 random-legal turns, plus a 2,000-turn scripted game reaching every cap (plan §4, test 11).

---

## 16. Alternatives and unresolved decisions

**Required review before implementation starts** (the defaults are implementable as written). Each lists the
recommended default and the alternative:

| # | Decision | Default | Alternative | Consequence |
|---|---|---|---|---|
| D1 | Objective | continuing, discounted γ = 0.99 | laning-session terminal event (A1) | §10; changes what is optimized |
| D2 | XP range and behaviors | R_xp = 3 with M2 behaviors | R_xp = 2 plus a redefined Yield ("stay one cell behind the wave") (A3) | how easily XP is denied |
| D3 | Tower-adjacent shot | on, 0.5 / 0.25 | off: the freezer's advantage is only exposure plus under-tower efficiency | strength of Freeze gold denial |
| D4 | Item prices and tier count | (4, 6), 2 tiers | (3, 5); 3 tiers (A6) | Recall frequency |
| D5 | Plate thresholds and Pressure damage | (16, 36, 60) units; 4 units per Pressure turn | per-plate value growth; fewer plates | siege incentives |
| D6 | Ganks scaled by the victim's level | yes | by the attacker's level (jungler = opponent's level) (A7) | snowball strength |
| D7 | Fronts during an enemy siege (mechanics.py:243–244) | keep the legacy rule (a front at x ≥ −3 waits while the enemy has a siege), documented | wait only at −3, behind the tower, as the comment intends | bounce timing; the default keeps the siege-model calibration; second Freeze contingency (§6.6) |
| D8 | Training start distribution | ρ_train = δ_{s₀} for the first pilot; the archive mixture (β = 0.25) run as a reported comparison | archive from the start | which states are learned; never the objective (§10.4) |
| D9 | Development evaluation window | W = 460 with guaranteed bound 0.67 plus an empirical tail estimate | always W = 1,000 (bound 0.003) | evaluation cost (~2.2×) vs rigor (§10.5.3) |

D1–D7 are rules of the game. D8 and D9 are methodology. None of them blocks M4.2, which implements only the
ruleset scaffold, the legacy pins and continuing time.

**Alternatives considered, ranked by significance** (the full trade-offs are in the audit, §7):

- **A1. Laning-session terminal event plus continuation value** (instead of a continuing game). The game ends
  when a tower falls, e.g. a fourth durability threshold, or when both champions reach a level, with a symmetric
  terminal valuation of the state (levels, items, HP). *Pros:* finite episodes, interpretable "lane won".
  *Cons:* the terminal valuation is an arbitrary reward on state, which reintroduces end-of-episode incentives
  (rush the terminal event, or avoid it). Not recommended unless continuing play proves untrainable.
- **A2. Average-reward objective.** Rejected. With finite caps the long-run average is decided entirely by the
  stationary capped game, so early progression, the subject of M4, would have zero weight.
- **A3. XP range 2 with an XP-keeping behavior.** Redefine Yield as "stand one cell behind the farm spot", or add a
  sixth behavior. More faithful to League's ranges; adds or changes an action.
- **A4. Explicit purchase action.** Rejected: dominated duplicate (§8.5).
- **A5. Defensive item track or damage reduction.** Deferred. With one track and outgoing scaling the model is
  simplest. Add it only if snowballing proves severe (audit §4).
- **A6. Three tiers, or a cheaper first tier.** Only if Recall stays unprofitable after S4.
- **A7. Gank or tower damage scaled by the attacker's or the game's level.** A stronger brake or accelerator on
  snowballing; measured as an ablation.
- **A8. Two half-strength rolls per trade.** Halves the breakpoint granularity (§8.3) and preserves the mean, but
  changes the trade variance relative to M2.
- **A9. Plate gold only for a present champion.** More League-like; makes minion-only plate breaks while away
  worthless.
- **A10. Freeze legal from the lane cell (x ≤ 0).** Not adopted as the default. In contact states it adds Freeze
  codes that behave like Slow Push (§1.1). It is retained as the **first contingency correction** if the Freeze gate
  fails G1 (§6.6).

---

## 17. Acceptance tests (summary; full list in the plan, §3–§4)

1. Freeze ≠ Slow Push in the crash-timing scenarios of §6.4 (plan §4.1). In passive 3–4 HP parked states they
   are equal, which is asserted and reported, not hidden.
1a. **Freeze gate G1–G3** on ≥ 100 reachable scenarios with multi-turn continuations (§6.6, plan S3). It covers
   situations where Freeze is advantageous, where Shove breaks it profitably, and where it is not beneficial.
2. Freeze cannot hold an enemy front of ≥ 10 HP, and an 8–9 HP front crashes within 8 turns despite it (§4.2).
3. Shove breaks a freeze with no automatic HP or farm penalty; reach is required (§4.3).
4. Gold and XP separate by cell under a held freeze (§4.4). XP denial requires the freezer at 0 or beyond (§4.5).
5. Severe power or HP disadvantage makes surrendering XP rational: an **empirical hypothesis**, tested by paired
   counterfactuals (plan S4), not asserted.
6. Item power changes damage only (§4.6).
7. Level, HP and damage tables are exact. Level-up HP follows §9.2, and death precedes the level-up (§4.7).
8. Purchases are affordable, irreversible, single, timed at the end of the last absent turn, and never rewarded
   negatively (WP6; §4.8–4.9).
9. Mirror equivariance and per-turn zero-sum (§4.12).
10. Truncation and segment boundaries bootstrap from the correct pre-reset state; the GAE matches closed forms for
    exact and deliberately biased critics (plan WP1, T1.4–T1.9). No countdown; observation invariant under
    k → k + P.
11. Legacy pins unchanged (WP0). The `m2c` fixture's per-turn dynamics equal `champions()` (WP1, T1.10).
12. Observation decoder round-trip, exhaustive over the reachable lane and on soak states. The 85 M2-aliased pairs
    are distinguished. Gold balances 4 and 9 at N = 0 are distinguished (plan WP7, O1–O7).
13. The evaluation tail bound's constants (A_max, K ≤ 27, ≤ 2 plates per turn) are re-checked by enumeration
    whenever the lane parameters change (plan WP3).
14. The ledger is exact: integer μ identities over 10,000-turn soaks; `validate()` rejects economies not
    representable in μ (WP6).
