# M4.2e-C — Wave Action Redesign and Mechanical Calibration

Branch `development/m4-natural-waves`, on top of M4.2e-B (203852d). Date: 2026-10-12.

| | |
|---|---|
| **Candidate preset** | `GameConfig.m4_lasthit()`: `m4_lane()` with five opt-in changes (§3). Every historical preset is unchanged (digest-pinned). |
| **Driver** | `python -m lanegame.experiments.m4.wave_actions` (74k deterministic turns, 8 s); results in `results/m4_wave_actions/wave_actions.json`. Sampled policy games: `python -m lanegame.experiments.m4.zoning_natural lasthit` (2.5M turns, 1 min on 11 processes); results in `results/m4_zoning_natural/zoning_natural_lasthit.json.gz`, `summary_lasthit.txt`, `trajectories_lasthit.txt`. |
| **Tests** | `tests/test_m4_wave_actions.py` (16 tests). |
| **Verdict** | **PASS WITH CAVEATS** (§9). |

Coordinates are GLOBAL unless stated (A walks +, A's tower at −2, the lane cell at 0, B's tower at +2). "Gold" means
resources (1 = one fresh 8-HP wave of farm). "A" is the moving or stronger side in every asymmetric probe; matrix
values are to A per 32 turns. "C" is a champion's wave-clear capacity in HP per turn.

**Method.** As in M4.2e-A.1 and M4.2e-B:

- Mechanical claims come from deterministic trajectories (the M4 resolver with no chance events, short scripted
  policies, the standard start s0, no seeded stacks except the freeze band analysis of §6, which is labelled).
- Champion power is varied through an attack-power multiplier that sets the capacity C = round(4 × AP) and every
  slot's share of it; §3 maps multipliers to (level, investment) states. The by-power probes set the lane up at equal
  power first.
- Strategic claims come from the M4.2e-B policy games: thirteen reactive scripted policies per seat, four start
  states, five power gaps, 16 seeds × 32 turns with the engine's chance events, each matrix solved as a zero-sum
  game. The candidate's freeze state and the freezing policy use the new freeze band (§6).
- Alternative action models are evaluated through explicit budget functions on the same probes (§2).

## Summary

- **The previous model's limitation was real and had a mechanical cause.** With Hold at zero damage, a wave never
  moved toward a champion that stopped contesting it, so standing back cost the weaker champion 2.5–6.5 gold per 32
  turns and the stand-at-range concession could not occur (M4.2e-B). Thin at a fixed 2 HP per turn gave a late-game
  champion the level-1 controlled option.
- **The adopted model is three shares of one capacity.** C = round(4 × AP) HP per turn is what the champion can
  remove; Hold uses an eighth of it (1 HP per turn until AP 2.9, 2 until AP 4.4, 3 after), Thin half (2, 4, 9 at AP
  1, 2, 4.2), Clear all (4, 8, 17). In the brief's notation D_Thin = D_Hold + α (D_Clear − D_Hold) with α = 3/7; the
  share form is the same family and simpler to state.
- **Last-hit damage alone broke slow pushing, and the cause was the damage law, not the action model.** Under the
  whole-wave Lanchester law `floor(h / 8)`, fronts between 8 and 15 HP fight alike; once both champions last-hit,
  fronts live in that band and a one-turn edge needs 19 turns to crash from the neutral lane instead of 13. The
  half-wave law (firepower `floor(2h / 8)` per turn, split over the ticks) lets a 12-HP front out-fight an 8-HP one,
  and the slow push compounds again: 15 turns for one Thin turn, 11 for two, 9 for four (the old model: 13, 11, 9).
- **Apply Pressure's own thinning is removed.** A champion in the enemy tower cell already reaches the defenders
  with its wave action; Hold + Pressure with the old thinning is exactly Thin + Pressure without it. One special case
  fewer.
- **What the candidate produces.**
  - The neutral lane is symmetric at every power; a champion three times stronger pushes the lane by last-hitting
    alone (crash in 8 turns), which the old model could not express.
  - A wave drifts to a champion that stands back (crash in 10–12 turns against a Safe or recalling opponent);
    standing back now costs the Safe champion 3.8 instead of 8.9 per 40 turns.
  - Thin scales: from a neutral lane it crashes in 9, 5, 3, 2 turns at AP 1, 2, 3, 4 (the old model: 7 at every AP);
    Clear in 5, 2, 2, 1.
  - Sieges last 2–6 turns at tower 3 and feed themselves at tower 1, as before; crashes of 21–27 HP bounce 16–26
    turns later with 23–29 HP.
  - The outside-tower freeze exists in a narrower band (enemy 12–14 HP against own 8–11), is reached after absorbing
    an enemy crash, holds 20–30 turns with a 1.7× farm edge under a threshold controller, breaks to Clear in 1–2
    turns, and a freezer at AP 3 pushes it out by last-hitting.
  - Recall from a neutral lane costs about one plate against an opponent that takes plates (3.9); after one's own
    crash it is free (−0.2 to −0.6).
- **Zoning (sampled, 16 seeds).** Against a stronger champion that only threatens (punishes or bullies), the weaker
  champion's best reply is now to stand back: it keeps 8–10 of its 13 gold because the wave comes to it, takes no
  damage, and the threat alone is worth −0.2 to +1.7 to the zoner instead of +6.5 to +10.5. The threat converts
  only through wave control (Clear + punish, +6 to +8), which is why the equilibria still have the stronger champion
  Clearing and the weaker one contesting. Every state's value stays monotone in power; the equal lane is even
  (+0.04) and both sides slow-push in it; the pushed, siege and freeze states are worth more at equal power than
  before (+1.4, +1.9, +2.1 against 0.0, +1.2, 0.0).
- **Caveats** (§7): the freeze band is 3 HP wide and needs an absolute threshold, not a margin; a champion at AP 3
  or more cannot hold a freeze; a crash into one's own tower is still collected at 0.6 and a passive opponent that
  never takes plates is punished by drift; the stand-back concession appears as a best reply to threats, not in
  equilibrium; all strategic evidence is scripted-policy evidence.

---

## 1. Weaknesses of the previous Hold / Thin / Clear model

The M4.2e-A.1 model (`m4_natural()`, kept in `m4_lane()`) assigns Hold 0, Thin 2 and Clear round(4 × AP) HP per turn.

| weakness | observed consequence (M4.2e-B and this study) |
|---|---|
| Hold does no wave damage | the lane never moves toward a champion that stops contesting: A Hold against B Safe for 40 turns leaves the contact at the lane on every turn and costs B 8.9 gold (§5); standing back cost the weaker champion 2.5–6.5 gold per 32 turns in every solved policy game, so no credible threat ever produced a concession (M4.2e-B §3) |
| Hold does no wave damage | an absent or recalling opponent does not lose the wave either: A Hold against a recalling B never crashes (§5) |
| Thin is fixed at 2 | Thin from a neutral stacked lane crashes in 7 turns with 19 HP at every attack power from 1 to 4 (§3); a level-18 champion's controlled option is the level-1 rate |
| the three actions are not one capability | Clear scales and the others do not, so the gap between Thin and Clear grows from 2 to 20 HP per turn over progression while the gap between Hold and Thin stays 2 |
| Apply Pressure carries its own thinning | with `natural_tower_reach = 1` the champion reaches the defenders anyway; the extra 1 HP per turn made Hold + Pressure a siege-sustaining line at AP 1 (+5.3 over 44 turns, §4) |

The first two are the conceptual limitation the brief names: a champion farming while the opponent abstains should
develop a pushing advantage. The third and fourth are the progression limitation.

## 2. Alternatives considered

Every model below was run on the same mechanical probes, under both damage laws (`stage_variants`; full table in
`results/m4_wave_actions/wave_actions.json`). Budgets are (Hold, Thin, Clear) per turn.

| model | budgets at AP 1 / AP 3 | neutral symmetric | AP 3 Hold pushes? | slow push from neutral, Thin ×1 / ×2 / ×4 (turns to crash), half law | Thin until crash at AP 1 / 2 / 3 / 4, half law | decision |
|---|---|---|---|---|---|---|
| `m4_lane` (Hold 0, Thin 2) | (0, 2, 4) / (0, 2, 12) | yes | no | 13 / 11 / 9 | 7 / 7 / 7 / 7 | the baseline |
| Hold only (Hold 1, Thin 2) | (1, 2, 4) / (1, 2, 12) | yes | no | 15 / 11 / 9 | 9 / 9 / 9 / 9 | Thin does not scale |
| fixed Hold 1, Thin = 1 + 3/7 (C − 1) | (1, 2, 4) / (1, 6, 12) | yes | **no** | 15 / 11 / 9 | 9 / 5 / 3 / 2 | a stronger champion's last-hitting never moves the lane |
| the brief's formula, α = 1/3 (Hold C/8) | (1, 2, 4) / (2, 5, 12) | yes | yes | 15 / 11 / 9 | 9 / 5 / 3 / 2 | close to the adopted model; Thin 7 at AP 4.2 |
| the brief's formula, α = 1/2 (Hold C/8) | (1, 3, 4) / (2, 7, 12) | yes | yes | 11 / 9 / 5 | 5 / 3 / 2 / 2 | Thin 3 at level 1 changes every A.1 slow-push calibration; Thin ≈ Clear from AP 3 |
| shares 1/4, 1/2 | (1, 2, 4) / (3, 6, 12) | yes | yes (crash in 7 turns) | 15 / 11 / 9 | 9 / 5 / 3 / 2 | Hold 3–6 late: last-hitting clears fresh waves |
| **shares 1/8, 1/2 (adopted)** | (1, 2, 4) / (2, 6, 12) | yes | yes (crash in 8 turns) | 15 / 11 / 9 | 9 / 5 / 3 / 2 | **the level-1 budgets of A.1 plus Hold 1; one capacity, three shares** |

Under the whole-wave law every model with Hold 1 gives 19 / 17 / 13 turns for the slow push (the baseline gives 13 /
11 / 9), which is the damage-law finding of §3.2.

Other alternatives:

| alternative | outcome | decision |
|---|---|---|
| keep the whole-wave law | slow pushes compound only through position: 19 turns for a one-turn edge, fronts never reach the 16-HP firepower step in the neutral lane (they oscillate 9–14) | rejected |
| **half-wave law** `floor(2h / 8)` per turn | a half-wave edge already fights harder; slow push 15 / 11 / 9; the fresh-wave timescale is unchanged (8 HP still deal 1 + 1); no new state | **adopted** |
| rescale the wave to 16 HP (quarter-wave resolution, Hold 1/16 wave) | would widen the freeze band (§6.4) at the cost of an XP, ledger and observation rescale and a float32 ledger limit; M4.2e-A.1 found no qualitative gain from finer HP | not adopted; listed |
| keep Apply Pressure's thinning (1 HP per turn) | redundant with the wave action under `natural_tower_reach = 1`; Hold + Pressure with it equals Thin + Pressure without it exactly (§4) | removed (`thin_damage = 0`) |
| tower damage 1, 2, 4 | 1: a one-turn Thin feeds a 38-turn, 171-HP siege; 2: 10 turns; 4: 4 turns and plates rarer | 3 kept (§4) |
| a margin-based freezer (Thin while enemy − own ≥ m) | never engages in the new band (§6) | replaced by a threshold controller in the harness; no engine change |
| scaling Hold sub-linearly in AP | not run; the eighth share already gives 1 / 2 / 3 over the whole progression | not needed |

## 3. Final damage formulas and progression

**Rules.** `wave_clear_scaling = "capacity"`:

| slot | per-turn budget | at AP 1 |
|---|---|---|
| Clear | C = round(ticks_per_turn × champion_wave_damage[Clear] × AP) = round(4 AP) | 4 |
| Thin | round(wave_thin_fraction × C) = round(C / 2) | 2 |
| Hold | round(wave_hold_fraction × C) = round(C / 8) | 1 |

Round half up on exact rationals; Hold ≤ Thin ≤ Clear and every budget is non-decreasing in C. The budget is spent as
before: split over the two ticks with the earlier tick first, on the enemy front in a lane fight, else the enemy's
crashed mass under one's own tower, else the enemy's leading packet, if within reach. Reach, presence and availability
are unchanged (a connected Trade, a Recall, an absence or a Safe retreat deals nothing).

`damage_law = "lanchester_half"`: a packet's firepower is max(1, floor(2h / W)) HP per turn, split over the ticks
with the earlier tick first and recomputed on the HP it has at each tick: 8 HP deal 1 + 1, 12 HP 2 + 1, 16 HP 2 + 2,
24 HP 3 + 3, and 4–7 HP deal 1 on the first tick only. It applies to lane fights, siege fights and the crashed mass's
hits on the tower.

`thin_damage = 0`: Apply Pressure adds tower damage and plates, and the champion's wave action is its damage on the
defenders behind the tower (`natural_tower_reach = 1` already reaches them).

**Parameters:**

| field | `m4_lane()` | `m4_lasthit()` | why |
|---|---|---|---|
| `wave_clear_scaling` | attack_power | **capacity** | one capacity, three shares |
| `wave_hold_fraction` | — | **0.125** | last-hitting pushes a little at every level, more when far ahead |
| `wave_thin_fraction` | — | **0.5** | Thin grows with power; the level-1 value stays 2 |
| `damage_law` | lanchester | **lanchester_half** | §3.2 |
| `thin_damage` | 1 | **0** | redundant special case |
| everything else | — | unchanged | W = 8, tower 3, entry crash, tower reach, collection rates, durability, 46 actions, 88 observation entries |

**Budgets by level and investment** (`stage_budgets`; AP from the saturating investment curve):

| level | invested | AP | Hold | Thin | Clear | `m4_lane` (Hold, Thin, Clear) |
|---|---|---|---|---|---|---|
| 1 | 0 | 1.00 | 1 | 2 | 4 | 0, 2, 4 |
| 3 | 0 | 1.10 | 1 | 2 | 4 | 0, 2, 4 |
| 6 | 4 | 1.48 | 1 | 3 | 6 | 0, 2, 6 |
| 9 | 16 | 2.13 | 1 | 5 | 9 | 0, 2, 9 |
| 13 | 32 | 2.93 | 2 | 6 | 12 | 0, 2, 12 |
| 18 | 64 | 4.19 | 2 | 9 | 17 | 0, 2, 17 |
| 18 | ∞ | 5.59 | 3 | 11 | 22 | 0, 2, 22 |

**Ticks to kill a fresh 8-HP wave already in contact** (lane only, no reinforcements):

| AP | Hold | Thin | Clear |
|---|---|---|---|
| 1 | 7 | 5 | 3 |
| 2 | 7 | 3 | 2 |
| 4 | 5 | 2 | 1 |

(`m4_lane` at AP 1: 8, 4, 3; its Hold and Thin do not change with AP.)

### 3.1 Capability against durability over a game

Minion durability is fixed (W = 8, no match clock, an accepted simplification), so the ratio of champion wave clear to
minion durability is the attack-power curve itself: 1.0 at level 1, 1.5 at level 6 with one small item, 2.1 at level
9 with one item, 2.9 at level 13, 4.2 at level 18 with four items' worth. In wave units per turn that is Clear 0.5 →
2.1 waves, Thin 0.25 → 1.1, Hold 0.125 → 0.25. A level-1 champion needs minion support to move anything (its Thin is
one extra HP over its last-hitting); a level-9 champion's Thin alone kills a fresh wave in 3 ticks; a level-18
champion's Thin clears a fresh wave in one turn and its Clear removes two. No level-based rule exists anywhere; the
progression is the damage ratio.

### 3.2 Why the damage law had to change

From the neutral lane (fronts 8 / 8 at k9 under the candidate), A Thins once and both Hold:

| law | model | turns to A's crash, Thin ×1 / ×2 / ×4 | first crash HP |
|---|---|---|---|
| whole-wave | `m4_lane` | 13 / 11 / 9 | 36 / 35 / 23 |
| whole-wave | shares (Hold 1) | 19 / 17 / 13 | 27 / 26 / 23 |
| half-wave | `m4_lane` | 13 / 11 / 9 | 24 / 35 / 33 |
| half-wave | **shares (Hold 1)** | **15 / 11 / 9** | 27 / 24 / 22 |

Under the whole-wave law a front of 8–15 HP deals 1 HP per tick whatever its size. Without last-hit damage the
neutral fronts oscillate 14–18 and cross the 16-HP step every other turn; with 1 HP per turn of symmetric last-hit
damage they oscillate 9–14 and never cross it, so a small edge compounds only through position (the smaller front
dies first, the bigger one walks forward). The half-wave law makes 12 HP deal 3 per turn against 8 HP's 2, which is
the Lanchester relationship the design philosophy asks for ("a small edge in a minion fight grows through
reinforcements into a stack"). It changes nothing for a fresh wave against a fresh wave.

## 4. Changes to wave HP, minion damage, towers, timing and other parameters

| parameter | decision | evidence |
|---|---|---|
| W = 8 and spawn timing | unchanged | a fresh front under symmetric last-hitting loses 3 HP per turn and still survives the reinforcement's arrival (8 / 3 = 2.7 turns against 2); the neutral lane is symmetric at every power (§5) |
| minion damage law | half-wave resolution | §3.2 |
| champion last-hit damage | an eighth of C | the only share that is 1 at level 1 and stays below a fresh wave's own firepower until AP 4.4 (§2) |
| Thin | half of C | the level-1 value is the A.1 value; the by-power gradient 9 / 5 / 3 / 2 turns is monotone and saturates last (§3) |
| Clear | unchanged | the fast push was already calibrated in A.1 |
| tower damage τ | 3, unchanged | sieges after a one-turn Thin at τ = 1 / 2 / 3 / 4: 38 / 10 / 6 / 4 turns and 171 / 59 / 43 / 35 HP; at τ = 3 Clear's sieges last 2–3 turns (`stage_sustain`) |
| Apply Pressure thinning | removed | Hold + Pressure with it is Thin + Pressure without it: the same siege line, +4.81, 2 plates (`stage_pressure`, pinned) |
| collection rates, durability, XP, levels, items | unchanged | the zoning values stay monotone in power (§6.6); no dependency revealed |
| the freeze controller in the scripted harness | threshold instead of margin | §6; the engine has no freeze rule |

**The siege after a slow push, by the action taken while sieging** (A Thins at k1 and sieges from its first crash; B
Holds; A − B over 44 turns, plates, siege length):

| AP | walk away (Hold) | Hold + Pressure | Thin + Pressure | Clear + Pressure |
|---|---|---|---|---|
| 1 | −1.84, bounce | +0.59, 1 plate, 6 turns | +4.81, 2 plates | +4.89, 2 plates |
| 2 | +0.56, bounce | +1.78, 2 plates, bounce | +6.94, 3 plates | +9.68, 3 plates, a 34-turn siege of 149 HP |
| 4 | +0.04 | +6.98, 3 plates | +9.41, 3 plates | +9.41, 3 plates |

`m4_lane` at AP 1: −3.41 / +5.34 / +9.00 / +7.39. The decision during a siege is now the wave action (take plates and
let it end, or keep the defenders dead and keep it open), and the perma-siege of A.1 remains a power-gated Clear +
Pressure line at AP 2 or more (caveat 3 of A.1; its counterplay is combat and ganks, as the M4.2e-B trajectory showed).

## 5. Representative behaviour across champion states

### 5.1 The neutral lane (`stage_neutral`)

Hold / Hold from s0 is symmetric under the candidate at every power: A − B = 0, the contact at the lane on all 39
turns, fronts oscillating 8 / 12 (`m4_lane`: 14 / 18), farm 16.65 each per 40 turns (15.98). A fresh wave dies within
its spawn interval, so the neutral lane carries one wave at a time, as League's does.

A stronger A, both Holding:

| A's AP | Hold budget | result |
|---|---|---|
| 1.0–2.0 | 1 | symmetric, no movement |
| 3.0–4.0 | 2 | the lane moves: contact at +1 for 23 of 40 turns, A crashes 17 HP at k8, then 8 per wave |

A champion three times stronger pushes by last-hitting alone, and against a Holder that farms the crash under its
tower this costs the stronger champion 1.2 over 40 turns: strength without wave management is a liability, as in
League. `m4_lane` shows nothing at any power.

### 5.2 Drift toward a champion that stops contesting (`stage_drift`)

| A's Hold budget (lane only, B's champion absent) | A's first crash | contact at +1 |
|---|---|---|
| 0 (`m4_lane`) | never | 0 of 40 turns |
| 1 | k12, 30 HP | 12 |
| 2 | k8, 21 HP | 20 |
| 3–4 | k6, 18 HP | 16–19 |

With champions (A Holds, B plays Safe and reaches only fights at or behind its own −1): under `m4_lane` the contact
stays at the lane and B collects 7.1 of A's 16.0; under the candidate A's wave crashes at k10 with 18 HP, B collects
9.3 of A's 13.1, and A − B falls from +8.88 to +3.81. XP is unaffected (144 each: the drift brings the minions to B).
Against a B that recalls whenever it can, A crashes at k10 and nets +12.4 instead of +16.0 (the crashes die to B's
tower, which B's champion does not farm because it is away). This is the mechanism M4.2e-B hypothesised; its
strategic effect is in §6.6.

### 5.3 Fast and controlled pushes by power (`stage_power`)

From the neutral stacked lane (k9 and k10), until A's first crash:

| A's AP | Clear: turns, HP (k9 / k10) | Thin: turns, HP (k9 / k10) | Thin ×1 then Hold | Thin ×2 then Hold |
|---|---|---|---|---|
| 1.0 | 5, 20 / 4, 15 | 9, 23 / 8, 21 | 15, 27 | 11, 24 |
| 1.5 | 3, 17 / 3, 11 | 5, 17 / 6, 20 | 11, 26 | 9, 24 |
| 2.0 | 2, 8 / 2, 15 | 5, 20 / 4, 15 | 9, 23 | 7, 24 |
| 3.0 | 2, 8 / 2, 16 | 3, 17 / 3, 11 | 5, 19 | 3, 15 |
| 4.0 | 1, 13 / 1, 8 | 2, 8 / 2, 15 | 3, 15 | 3, 16 |

`m4_lane`: Clear 5 → 3 → 3 → 2 → 2 turns; Thin 7 turns and 19 HP at every AP; Thin ×1 13 turns and 36 HP at every AP.

Reading: a level-1 champion's Thin needs 9 turns and the help of three reinforcements to crash; at AP 3 (level 13) it
crashes in 3 turns; at AP 4 Thin shoves every fresh wave. A one-turn Thin slow push takes 15 turns at level 1 and 3
at AP 4. Stronger champions crash sooner and smaller, as in A.1. From s0, Clear at AP 2 or more crashes every fresh
wave every two turns (the perma-shove of A.1, unchanged).

### 5.4 The seven lines against a passive opponent, 80 turns (`stage_actions`)

| A's line | against Hold: `m4_lane` | candidate | against Hold + Pressure: `m4_lane` | candidate |
|---|---|---|---|---|
| Hold | 0 | 0 | 0 | 0 |
| Thin always | +0.30 (15 crashes) | +2.29 (16) | +0.30 | +2.29 |
| Clear always | +1.95 (31) | +2.85 (26) | +1.95 | +2.85 |
| Thin k1 | +1.35 | −3.52 | −7.80 (B 3 plates) | −5.84 (B 1 plate) |
| Thin k1–4 | +2.06 | −1.91 | −9.30 (B 3 plates) | −5.97 (B 2 plates) |
| Thin k1–4, then Pressure | +9.49 (3 plates) | +7.76 (3) | +9.49 | −0.04 (2 against 2) |
| Clear, then Clear + Pressure | +11.14 (3) | +12.90 (3) | +11.14 | +12.90 |

These 80-turn totals are phase-sensitive (they depend on where the push–bounce cycle stands at turn 80 and on who
farms whose crash) and are kept for comparability with A.1; the mechanical measures are in §5.3 and §6. What is
robust: no wave action dominates, a slow push that is not followed up is punished by the bounce (and, against an
opponent that takes plates, more so under the candidate because its own crash is smaller and the bounce larger),
and the follow-up decides the value.

## 6. Effects on slow pushing, freezing, crashing, bouncing, zoning and Recall

### 6.1 Slow pushing

§3.2 and §5.3: one Thin turn from the neutral lane crashes in 15 turns with 27 HP (from s0: at k22 with 27 HP plus 16
of follow-ups); two Thin turns in 11; four in 9. The edge compounds through firepower from 12 HP on. A stronger
champion's slow push is faster (5 turns at AP 3).

### 6.2 Freezing

**The band.** Per spawn interval (two turns) the own front F in the tower cell gains 8 and loses 2 × floor(2E / 8) +
2 × (the enemy's Hold); the enemy front E at −1 gains 8 and loses 2 × floor(2F / 8) + 2 × (own Hold or Thin). At
level 1 (Hold 1, Thin 2):

- F is steady while E ≤ 15 (E's firepower 3 per turn plus the enemy's last-hitting equals F's 8 per interval);
- E is steady at F ∈ [8, 11] with Thin (2 × 2 + 4 = 8) and grows by 2 per interval with Hold;
- so the band is E ∈ [12, 15] against F ∈ [8, 11], and the freezer keeps E there by Thinning when E reaches 14.

A.1's band was E ∈ [16, 23] against F ≤ 15: last-hit damage on both sides and the finer law narrow it by about a wave.
A margin rule (Thin while E − F ≥ 8) never fires inside the new band, so the harness uses an absolute threshold
(`wave_actions.threshold_freezer(14)`: Hold until the contact stands at own −1, then Thin while E ≥ 14).

**Seeded states** (labelled; A's F in its tower cell, B's E at −1, 30 turns, held turns under the controller):

| F, E | Hold only | Thin only | threshold 13 | threshold 14 | margin 8 |
|---|---|---|---|---|---|
| 8, 10 | 12 | 2 | 12 | 12 | 12 |
| 8, 12 | 13 | 8 | 26 | **30** (farm 13.3 against 11.1, no crash) | 28 |
| 8, 14 | 1 | 10 | 26 | 17 | 1 |
| 8, 15 | 1 | 1 | 1 | 1 | 1 |
| 8, 16–24 | 1 | 1 | 1 | 1 | 1: the wave crashes through at once |
| 12, 14–16 | 8–14 | 6 | 8–10 | 8–10 | then A's own 26-HP crash |

(`m4_lane`: F 8 against E 20 is held 26–30 turns by thresholds 14–16 or margin 8; F 12 against E 24 is held 30.)

**Reached from s0** (`stage_freeze`, B pushes per setup then Holds; A plays the controller; 70 turns):

| setup | controller | longest hold | from |
|---|---|---|---|
| B Thin k1–4 | threshold 14–16 | **20 turns** | k50, after absorbing B's 22-HP crash at k48 |
| B Thin k1–4 | threshold 13 | 16 | k12 |
| B Thin k1–2 | threshold | 8 | k48 |
| B Thin k1 / k2 | margin 8–10 | 16 | k24–26 |
| any | margin 8 | 8 | — |

(`m4_lane`: margin 10 holds 48 turns from k22.)

**Maintained, broken, by power** (from the hold start at k50: A −2:2 with a rear wave, B's 8-HP follow-up walking in,
B's crashed 12 HP dying under A's tower; 24 turns):

| from k50 | held | A − B | farm A / B | XP A / B |
|---|---|---|---|---|
| A freezer, B Holds | **20** (all) | +5.17 | 12.6 / 7.4 | 116 / 69 |
| A Holds (uncontrolled) | 20 | +5.17 | 12.6 / 7.4 | the controller never fired: E stayed below 14 |
| B Thins | 3 | −0.56 | | B crashes 8 + 8 at k53–54 |
| B Clears | 1 | +0.34 | | |
| B Clears once (level-6 budget 6) | 1 | +1.50 | | B crashes 12 at k52; A crashes 24 at k66 |
| B Clears once at a level-1 budget (4) | 7 | | | |
| freezer at AP 2 (Hold 1, Thin 4) | 20 | | | |
| freezer at AP 3–4 (Hold 2) | 6 | | | A's own 21-HP crash at k60: its last-hitting pushes the wave out |

Interpretation: the freeze is the League pattern of A.1 (absorb the enemy crash, hold the follow-ups outside tower
range with a farm and XP edge, lose it to a Clear), in a narrower band that needs the right controller, and it cannot
be held by a champion far ahead because its last-hitting alone out-damages the enemy's reinforcement stream. The last
point is new and League-like (waves cannot be frozen by a champion that kills them by farming), and it is a caveat
for learning (§7).

### 6.3 Crashing and bouncing (`stage_bounce`, `stage_sustain`)

| line | crash | siege | tower hits | bounce |
|---|---|---|---|---|
| Thin k1 | k22, 27 HP (+16 follow-ups) | 5 turns | 2 | 16 turns later, 23 HP |
| Thin k1–4 | k10, 21 | 3 | 4 | 26 turns later, 29 HP |
| Thin until crash | k8, 17 | 3 | 2 | 26 turns later, 29 HP |
| Thin k1–3, Clear k8 | k10, 21 (+8) | 3 | 0 | 16 turns later, 27 HP |

`m4_lane`: 36 (+32) / 9 turns / 22 turns later with 24 HP; 31 (+24) / 7 / 20 with 24; 18 / 3 / no bounce in 50; 23 / 4
/ no bounce. Crashes are smaller (the defenders' last-hitting thins the stack as it builds), sieges shorter (the
half-wave law gives a 12-HP defender front 3 HP per turn on the mass), and every line now bounces. The 80-turn cycle of
the one-turn slow push: A crashes at k22, k24, k26; B bounces at k38 (23 HP) and k52 (24); A again at k66 (24): a
period of about 44 turns. Sieges feed themselves at τ = 1 (38 turns, 171 HP from one Thin turn) and not at τ = 3 (6
turns, 43 HP), as in A.1.

### 6.4 Zoning: the policy games (`zoning_natural lasthit`, 16 seeds)

**Equilibrium value to A** (A's trades per turn in brackets; `m4_lane` from the M4.2e-B full run):

| state | equal | moderate ×1.24 | levels ×1.35 | large ×1.76 | large + levels ×2.38 |
|---|---|---|---|---|---|
| neutral | **+0.04** (0.00) / −0.12 | +2.39 (0.38) / +2.14 | +0.37 (0.38) / +1.02 | +4.06 (0.28) / +4.73 | +5.91 (0.24) / +6.50 |
| pushed | **+1.41** (0.29) / −0.02 | +2.40 / +0.95 | +1.42 / +0.40 | +3.33 / +2.67 | +4.04 / +3.79 |
| siege | **+1.85** (0.24) / +1.19 | +3.07 / +3.01 | +3.18 / +1.97 | +4.97 (0.00) / +4.09 | +4.97 / +3.99 |
| freeze | **+2.12** (0.28) / 0.00 | +3.22 / +2.12 | +2.09 / +1.97 | +3.98 / +5.64 | +4.96 / +6.72 |

Standard errors of the cells are 0.8–1.3. Every state's value is monotone in power. At equal power the neutral lane is
even and both sides slow-push (push against push, A never trades, B in contact 90% of turns); the pushed, siege and
freeze states are worth +1.4 to +2.1 to the side that holds them, where `m4_lane` gave 0.0, +1.2 and 0.0. At a large
gap the neutral and freeze values are 0.7–1.7 lower than before: the weaker champion's counterplay improved (below).
The equilibria are otherwise those of M4.2e-B (Clear + punish against Clear + punish; in the siege at ×1.76 the weaker
champion stands back with `respect`, at ×2.38 it pushes back out).

**The cost of standing back** (value to A | B's farm | B's deaths per 32 turns; A's threat policies against B's
replies, neutral state):

| gap, model | A plays | B contests (farm) | B stands off | B respects | B hides at its tower |
|---|---|---|---|---|---|
| ×1.76, `m4_lane` | punish | +0.72 / 9.4 / 0.7 | +0.78 / 9.4 / 0.5 | +6.52 / 7.2 / 0.0 | +10.45 / 3.2 / 0.0 |
| ×1.76, **candidate** | punish | +1.66 / 8.3 / 0.8 | +1.53 / 8.4 / 0.6 | **−0.15 / 9.8 / 0.0** | +0.66 / 10.5 / 0.0 |
| ×1.76, `m4_lane` | bully | +0.81 / 6.5 / 1.6 | +4.94 / 6.2 / 0.4 | +7.14 / 6.4 / 0.0 | +13.34 / 0.2 / 0.0 |
| ×1.76, **candidate** | bully | +0.89 / 5.6 / 1.5 | +3.35 / 5.8 / 0.4 | **−1.53 / 8.2 / 0.0** | −0.09 / 6.8 / 0.1 |
| ×1.76, `m4_lane` | Clear + punish | +7.00 / 10.1 / 0.1 | +6.93 / 10.2 | +7.89 / 9.9 | +7.68 / 9.9 |
| ×1.76, **candidate** | Clear + punish | +6.51 / 10.2 / 0.1 | +6.13 / 10.3 | +7.33 / 9.8 | +8.03 / 9.7 |
| equal, `m4_lane` | punish | −2.10 / 10.8 | −1.58 / 10.5 | +6.14 / 7.3 | +10.09 / 3.3 |
| equal, **candidate** | punish | +0.97 / 9.0 | +1.36 / 8.6 | **+0.02 / 9.8** | +0.64 / 10.3 |

Against a stronger champion that threatens without controlling the wave, standing back ("respect": Safe when the
enemy starts in reach, else the stand-off) is now the weaker champion's best reply: it keeps 9.8 of its farm, is
never hit, and leaves the zoner −0.2 to −1.5 instead of +6.5 to +7.1. The best-reply table says the same: at ×1.76
B's reply to `bully` and `bully_shove` is `respect` (it was `push`), to `punish` it is `push`. Against a champion
that Clears and punishes, contesting and pushing back remain B's best replies (its farm 10.2–10.5 against 9.7–9.8
standing back), so the equilibrium concession criterion of M4.2e-B (a conceding best reply that collects under 80%
of contesting) is still never met: the zoner converts through wave control, as League's does, and standing back is
the answer to a threat that is not backed by a push.

The one-turn flip multipliers (the attack ratio at which B's best reply to a Trade becomes Safe) move from 3.68 to
3.99 in the neutral lane and from 3.1 to 1.0 in the freeze state (the breaker next to the enemy tower now concedes
to a Trade at equal power in the one-turn sense).

**Representative trajectories** (`trajectories_lasthit.txt`, seed 5000, ×1.76):

- *A punishes, B respects.* B is Safe on alternate turns and stands off on the others; it is never hit; its wave
  shrinks under A's last-hitting (A 20 against B 2 by k19) and A's 29-HP stack crashes at k22; B then farms the
  crash under its tower at 0.6 and A, who never Pressures, gives most of the lead back: +0.21 at the end of the
  window.
- *A Clears and punishes, B respects.* A crashes 9 at k11 and 8 HP every two turns, stands in B's tower cell and
  takes three plates by k18; B farms under tower at 0.53 per turn and is never hit; +9.8 at the end: zoning plus wave
  control.
- *A punishes, B farms.* B absorbs 1.3 units per connected turn, keeps farming, and its wave drifts into A's tower
  (19-HP crash at k18); B walks back at 1.2 units and recalls; the window ends −0.4: the threat alone does not pay
  even against a champion that contests.
- *Equal, both slow-push.* Symmetric: fronts 11 / 11, 7 / 7, 9 / 9 … every turn; +0.00.
- *Freeze at equal power.* B's first Clear is ganked at the start; B crashes 11 at k52 anyway, recalls, returns and
  Clears; both trade next to A's tower; both recall at k64; B's 34-HP wave crashes at k66: −5.8 by the end. A freeze
  at equal power is worth holding only while the breaker does not commit (the equilibrium value +2.12 is the
  average over seeds).

### 6.5 Recall timing (`stage_recall`)

Cost of a Recall (the stay line's A − B minus the Recall line's), against a Holder that takes plates when it sieges:

| situation | `m4_lane` | candidate |
|---|---|---|
| neutral lane, Recall at k13 | 1.15 | **3.89** (B's wave drifts into A's tower while A is away and B takes a plate) |
| right after own small crash | 0.68 | −0.19 |
| after own large crash | 0.00 | −0.60 |
| after a failed push | 1.24 | −0.11 |
| enemy builds k1–4 and crashes: Recall at k6 / 11 / 13 / 15 / 17 / 20 | 0.58 / 0.46 / 0.60 / 1.13 / 1.13 / 0.90 | −0.38 / −2.99 / 0.94 / −0.11 / 1.09 / 1.09 |

A Recall is now expensive from a neutral lane and free after one's own crash, which is the League window, and sharper
than before. Against a naive Holder that never takes plates the sign flips (−1.6 from the neutral lane): the drift
makes its wave crash into A's tower while B stands at the lane out of reach of the siege, and A farms the crash. The
enemy-build timings are noisy at the one-turn level (spawn-phase sensitivity, a known caveat).

### 6.6 Scripted population (`stage_scripted`, 100 turns, seeded env)

| pair | `m4_lane` | candidate |
|---|---|---|
| slow_pusher vs hold_farmer | +8.81 | +6.40 |
| crasher vs hold_farmer | +9.44 | +7.78 |
| natural_freezer vs slow_pusher | −7.51 | −4.38 |
| clearer vs thinner | +7.99 | +10.79 |
| crash_recaller vs natural_freezer | −0.24 | +1.50 |
| trading_pusher vs hold_farmer | −9.28 | −0.16 |

Correctness and variety only; the `natural_freezer` agent still uses the margin rule and is a weak freezer under the
candidate.

## 7. Regressions, edge cases and remaining limitations

1. **The freeze band is narrow** (enemy 12–14 HP against own 8–11) and its control is an absolute threshold with
   1-HP granularity. A learning agent must find a 3-HP band; the scripted margin freezer of earlier milestones
   does not. The band width is set by W relative to the last-hit damage; a 16-HP wave would widen it at the cost of
   a rescale (§2).
2. **A champion at AP 3 or more cannot hold a freeze**: its last-hitting (2 HP per turn) out-damages the
   reinforcement stream and pushes the wave out in 6 turns. League-like, but the controlled option of a champion far
   ahead is a slow push, not a freeze.
3. **A crash into one's own tower is still a gift.** The own-tower rate of 0.6 (M4.2e-B) is unchanged, and the drift
   produces crashes against passive opponents by itself. A Holder that never takes plates loses to its own drift
   (A's Recall from a neutral lane *gains* 1.6 against it), and a stronger champion that pushes by last-hitting
   without Pressure loses 1.2 per 40 turns. Against the plate-taking policies of the policy games the sign is right
   (Recall costs 3.9; the pushed and siege states pay). The rate was not recalibrated here (the quick variants
   `lasthit_ot5` and `lasthit_ot7` exist in the harness for a later sensitivity run).
4. **The stand-back concession is a best reply to threats, not an equilibrium outcome.** Against Clear + punish the
   weaker champion still contests. The M4.2e-B criterion (a conceding best reply that collects under 80% of
   contesting) is never met because standing back no longer costs farm, so the criterion itself no longer measures
   what it was built for.
5. **The slow push without a follow-up is punished more** (−3.5 to −6.0 over 80 turns against a Holder) because the
   crash is smaller and the bounce larger; the economics of pushing are the follow-up's (Pressure, Recall, plates).
6. **Phase sensitivity remains**: one-turn differences in Recall timing or Thin timing change outcomes by one or two
   gold; inherent to a 2-tick turn and a 4-tick spawn interval.
7. **The perma-shove and perma-siege of A.1** at AP 2 or more are unchanged (Clear + Pressure, +9.7 and a 34-turn siege
   at AP 2 against a passive Holder); combat and ganks remain the counterplay.
8. **Fronts are smaller everywhere** (neutral 8–12 instead of 14–18; crashes 21–27 instead of 31–36). Plates from
   minion hits are rarer still (2–4 per stacked siege); Pressure is the plate source.
9. **The evaluation bound** stays analytic (per-turn 10.36, unchanged: the half-wave law's larger first-tick share is
   offset by the removed thinning).
10. **All strategic evidence is scripted-policy evidence** over 32-turn windows with a linear health valuation at the
    end; learned play is the real test and was not run.

## 8. Correctness, reproducibility and performance

**Tests** (`tests/test_m4_wave_actions.py`, 16):

- the candidate's delta from `m4_lane()` is exactly `M4_LASTHIT_FIELDS`; every other preset keeps the old model;
  validation of the fractions (0 ≤ hold ≤ thin ≤ 1, capacity-only), of the half-wave law (natural-only) and the legacy
  rejections;
- the budget table above, monotonicity in level and investment, Hold ≤ Thin ≤ Clear everywhere, Clear unchanged from
  `m4_lane`, the resolver passing the capacity budgets (Hold removes 1 HP per turn; a stronger champion removes more);
- the half-wave law's per-tick split for 1–56 HP, its per-turn sum, its tick-independence for every historical preset,
  and a 12-against-8 lane fight pinned tick by tick;
- Pressure thinning off, the wave action reaching the defenders, and Hold + Pressure with the old thinning equal to
  Thin + Pressure without it;
- HP conservation, kill accounting, no front ever left in the enemy tower cell and mirror symmetry on 1,500 random
  lane turns with random budgets; resolver mirror equivariance with a stronger seat, invariants, the 88-entry
  observation and its decoding, and the per-turn reward bound on 400 random turns; an env soak (zero sum, exact
  accounting, minion conservation), deterministic replay and a random-play digest pin of the candidate;
- pins of §5 and §6: neutral symmetry and the stronger champion's push, the drift against a Safe champion, the slow
  push compounding under the half-wave law and not under the whole-wave law, Thin and Clear scaling, the freeze band
  (held, broken, pushed out by a strong freezer), crashes bouncing, sieges not self-sustaining at τ = 3, Recall free
  after one's own crash and a plate from the neutral lane.

**Regression.** The focused suite (the new file plus the natural-wave prototype and final tests, the zoning tests
with their digest pins of `m4()`, `m4_emergent()`, `m4_natural()` and `m4_lane()`, config, tower, env, resolver,
combat, economy, time, observation, legacy pins, wave mechanics, invariants, integration, zoning, positioning,
emergent, progression, scenarios, freeze gate, farm spot, investment readiness, calibration, continuous investment,
diagnostics, siege, env, champions, champion combat and economy tests; 33 files): **633 passed, 4 skipped, 2 expected
failures, in 328 s.** `tests/test_m4_config.py`'s legacy-rejection fixture gained alternate values for the two new
fields, as in every previous milestone. The full suite with `--runslow` was not run.

**Historical compatibility.** `GameConfig()`, `.v1()`, `.v2()`, `.champions()`, `.m4()`, `.m4_continuous()`,
`.m4_emergent()`, `.m4_natural_prototype()`, `.m4_natural()` and `.m4_lane()` are unchanged; the three new fields
default to the old behaviour; `minion_damage` gained an optional tick argument that only the new law reads;
`mechanics.step_turn` and the env interface are unchanged. The zoning harness gained the variants `lane`, `lasthit`,
`lasthit_ot5`, `lasthit_ot7`, a preset-aware freeze start state and a preset-aware freezing policy; its historical
variants and results are untouched.

**Observation and actions.** Unchanged: 46 actions, 88 entries, no hidden state (the budgets are functions of the
observed level and investment).

**Performance** (single thread, this machine): lane step 43.4 µs (`m4_lane` 41.6), env step 246 µs (224).

**Simulation budget.** About 9.7M turns: the deterministic driver and its stage reruns 0.3M, scratch probes during the
design 0.2M, the quick policy games 6.6M, the full 16-seed run 2.5M, trajectories 0.3k. Every run is deterministic
from its seeds and reproduces from the drivers.

## 9. Readiness and recommendation

**Is the revised system ready for integration and PPO evaluation?** Yes, with the caveats of §7.

| criterion | status |
|---|---|
| Hold, Thin, Clear are shares of one capability, distinct at every level | yes: 1 / 2 / 4 at level 1, 3 / 11 / 22 at the cap |
| normal last-hitting pushes against an opponent that abstains | yes: crash in 10–12 turns; a 3× stronger champion pushes the neutral lane |
| Thin manipulates larger waves with progression | yes: 9 → 2 turns from a neutral lane |
| slow pushes, freezes, crashes, bounces emerge from one rule set | yes; the freeze band is narrower and needs a threshold controller |
| sieges bounded; Recall windows | yes: 2–6 turns at τ = 3; free after own crash, a plate from the neutral lane |
| zoning: a threat converts through wave control, standing back answers a bare threat | yes (sampled) |
| no unconditionally optimal action; values monotone in power | yes (sampled) |
| correctness guarantees, interface, throughput | yes |

**For the next milestone (M4.3 PPO smoke test):** train on `m4_lasthit()`; judge the learned play against §6.4 (Clear
plus punish when ahead, slow pushes at equal power, standing back against bare threats, Recall after crashes) and
not against the League pattern of a learned freeze, which needs a 3-HP control band. Before training, decide the
two listed parameters: the own-tower collection rate (0.6; `lasthit_ot5` / `lasthit_ot7` are in the harness) and
whether a 16-HP wave is worth its rescale to widen the freeze band.

### Verdict: **PASS WITH CAVEATS**

**Why it passes.** The redesign is the smallest coherent model that meets the brief: three shares of one capacity,
the level-1 budgets of the validated A.1 model plus one HP of last-hit damage, and a damage law one step finer so
that the edges last-hitting creates can compound. It removes a special case (Pressure thinning) rather than adding
one, adds no state, keeps every historical preset bit-identical, and produces the behaviours the design philosophy
names, including two it could not produce before: a wave that drifts toward a champion that stops contesting, and a
stronger champion that pushes by farming.

**Caveats.** The narrow freeze band and its threshold control; no freeze for a champion far ahead; the own-tower rate
and the drift together still make a crash a gift against passive opponents; the stand-back concession is a best
reply to bare threats and not an equilibrium outcome; scripted evidence only.
