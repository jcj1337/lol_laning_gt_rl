# M4.3c — Quick XP-Range Screen (`xp_range` 3 / 2 / 1)

**Status: SCREEN. Nothing in the game, its presets or the trained agents was changed.** Every number comes from
configuration overrides of the research preset, `GameConfig.m4_lasthit(xp_range=r)`, evaluated with the existing M4.3c
freeze harness, the M4.3b finals and the scripted references (`lanegame/experiments/m43c/xp_range.py`, results in
`results/m43c/xp_range.*`). No PPO was trained. The M4.3b policies were trained at `xp_range = 3`, so how they play under
2 or 1 is a screen of fixed policies, not an equilibrium of the modified game. Rules are cited from the
[mathematical reference](../laning_game_mathematical_reference.md) ("Reference §n").

Branch: `development/m4-3b-validation` (working tree on top of `26f8c83`).

---

## 0. Answer

**Recommendation: keep `xp_range = 3`. Do not try 2 or 1. Investigate another mechanic** (§6).

A smaller range does deny XP to a champion who steps back from a held freeze. But the champion who walks up is never
denied, at any range, and walking up was already the better choice. So the change makes conceding worse; it does not
create a choice between farm and safety. It also takes XP from the freezer and from any champion whose crashed wave is
defended behind the enemy tower. At range 1 it disrupts normal laning.

| question | answer |
|---|---|
| 1. Does `xp_range = 2` create meaningful XP denial during a freeze? | **Only against a champion standing at its own −1 or further back.** During the hold, the freezer's minions die in the freezer's tower cell (+2 for B). B at −1 is 3 cells away, so it is denied at range 2 (+21.7 XP missed per 40 turns). B walking up to +1 is 1 cell away and is never denied. The extra net denial is about 8 XP, worth about 1 resource at λ = 1/8. Stepping back already costs 4.9 gold. |
| 2. Is `xp_range = 1` much stronger, and too restrictive? | Stronger against a step-back (43 XP missed vs 10) and against Clear without a follow-up. **Too restrictive:** in learned self-play 4–9% of all XP is missed and level 16 comes 5–13 turns later. The cause is a farm-spot artifact, not pressure (§2.3). |
| 3. Does a weaker champion now have a reason to concede XP rather than risk combat? | **No.** The advantage of walking up over stepping back grows (equal: 5.5 → 7.6 → 8.1). Against the realistic freezer (×1.76 and punishing) it stays at 2.9 at every range. Walking up still costs only HP: 1.3 bars and 0.23 deaths per 40 turns. |
| 4. Does the freezer's economic advantage improve? | **No.** Against a champion who walks up it gets worse (equal: −0.26 → −1.46 → −1.79), because the freezer itself misses XP. Against the learned breaker nothing changes (−3.4 / −3.1 / −2.9). It improves only against a static step-back, which B would not choose. Not freezing (Clear) still beats freezing at every range. |
| 5. Can an aggressive clearer still break the freeze profitably? | **Yes.** The learned policy gains +4.8 gold at every range. On gold + XP that is +3.37 / +3.12 / +2.87, and the scripted Clear + punish + Pressure gains +4.34 / +4.24 / +3.49. Only Clear *without* following the crash loses XP: −0.21 → −1.18 → −3.04. |
| 6. Is the XP disadvantage persistent? | Missed XP is never recovered before the level cap. From turn 40 to turn 60 the missed-XP gap changes only through new misses (≤ 5 XP), while the delayed part swings by about 20 XP at every range. B's 20-turn deficit when walking up (−33 XP at every range) is only a delay and is gone by turn 40. Over whole games, \|XP diff\| at turn 100 changes by −1 to +5 XP at range 2. |
| 7. Problems in neutral laning? | **Range 2: small but real.** Learned self-play loses ≤ 0.6% of XP and its levels are unchanged. Scripted last-hitters lose 4.8%, and a scripted Clear-vs-Hold clearer loses 12.6 XP per 40 turns: champions miss the kills behind the enemy tower when they do not follow their crash. **Range 1: large**, see 2. |

---

## 1. Compact comparison

40-turn windows from B's side (B is the frozen-on champion), unless stated. Value = gold + λ·XP with λ = 1/8.
Uncertainties are in §3–§4. Changes between ranges in the same cell are paired on the same seeds and have 95%
half-widths of ±0.02–0.8, most below ±0.3 (`results/m43c/xp_range.md`).

| quantity | r = 3 | r = 2 | r = 1 |
|---|---|---|---|
| **Held freeze, A = threshold freezer, equal power** | | | |
| B steps back (−1): B's missed XP | 10.2 | 31.9 | 42.8 |
| B steps back: missed-XP gap (B − A) | −6.8 | −14.8 | −16.2 |
| B steps back: value | −5.21 | −6.16 | −6.34 |
| B walks up (+1): missed-XP gap | −1.9 | **+7.6** | **+10.0** |
| B walks up: value | +0.26 | +1.46 | +1.79 |
| step back − walk up (< 0: contesting is better) | −5.47 | −7.62 | −8.13 |
| learned breaker (base s0): value | +3.37 | +3.12 | +2.87 |
| freezer's own missed XP against a walk-up | 2.8 | 15.2 | 21.8 |
| **Held freeze, A = punishing freezer, A ahead ×1.76** | | | |
| step back − walk up | −2.94 | −2.92 | −2.94 |
| B walks up: value | −0.49 | +0.42 | +1.17 |
| B's best response (learned ent03 s0): value | +3.65 | +3.53 | +2.90 |
| **Neutral control** | | | |
| scripted Hold/Hold: XP missed to range per side | 0.1 | 2.8 | 5.1 |
| scripted Clear vs Hold: clearer's missed XP; XP gap (Hold − Clear) | 0.6; −3.2 | 12.6; +9.0 | 28.0; +22.0 |
| learned self-play (3 finals): XP missed to range, % of available, first 100 turns | 0.0–0.1% | 0.2–0.6% | 4.0–9.4% |
| learned self-play: turn of level 16 | 114–115 | 114–115 | 120–127 |
| scripted last-hitters (`farm` vs `farm`): % missed; turn of level 16 | 0.1%; 114 | 4.8%; 119 | 8.6%; 124 |

---

## 2. Geometry: where minions die and who is in range

A champion that is not away gains 1 XP for every HP of enemy minions that dies within `xp_range` cells of the cell it
ends the turn in, whoever kills the minions (Reference §5.4). The cell is the death cell on the collector's own kill
ledger.

**Missed and delayed XP.** Each side's XP is split into three parts. *Available* is the HP on its ledger. *Missed* is
available XP it did not get because it was out of range or away; this is lost for good. *Pending* is the opponent's
minion HP still alive at the end of the window; this is delayed, and the side gets it later only if it is present and
in range when those minions die. The bookkeeping is exact: available − missed = credited before the cap, pinned by a
test. The histogram of available HP over (champion cell, death cell) prices every range on the same play.

### 2.1 The held freeze

B's front is held at B's +1 (the freezer's −1) and the freezer's front stands in its tower cell, which is B's +2. So
**the freezer's minions die at B's +2**, not +1. Of the 58 HP per game that die while a walking-up B stands at +1, 52
die at +2.

| B's cell | B's response | distance to +2 | denied during the hold at r = 3 | r = 2 | r = 1 |
|---|---|---|---|---|---|
| +1 | walk up (Hold); stand off (Yield) is the same cell here | 1 | no | no | no |
| 0 | — | 2 | no | no | yes |
| −1 | step back (Safe) | 3 | no | **yes** | yes |
| −2 | retreat to the own tower cell (Safe, `safe_line = −2`) | 4 | yes | yes | yes |

So a champion at −1 *is* denied at range 2 during a held freeze, because the freezer's minions die one cell deeper than
the freeze line. A kill at +1 (distance 2, still credited at range 2) happens only once the contact moves forward.

**The denial is self-limiting.** A champion who steps back stops killing the freezer's minions. In Safe mode its reach
line is its own −1 (Reference §4.5), so only its minions do damage, about 2 HP per turn. The freezer cannot stop killing
B's minions, because Hold kills at least 1 HP per turn. The freezer's front therefore grows, and within about 6 turns
the freeze collapses toward B: held for 8.6 of 40 turns when B steps back (6.0 with A ahead), against 18.4 when B walks
up. The freezer's minions that piled up then die near B, in range. Stepping back costs gold, because Safe collects
at most 0.6 and only within one cell (Reference §5.3); most of its XP is only delayed.

### 2.2 Examples

**Example 1 — step back, equal power, first paired seed** (`results/m43c/xp_range.json.gz`, `example`). Cells are in
own coordinates: A's cells are global, B's are mirrored.

```
turn  A cell  B cell  A's front   B's front   A's minions die at (B)   B's XP r=3 / r=2   A's XP
 1     -2      -1     -2 / 6 HP   +1 / 11     +2: 2 HP                 2 / 0              8
 2     -2      -1     -2 / 12     +1 / 7      +2: 2                    2 / 0              4
 4     -2      -1     -2 / 16     +1 / 6      +2: 2                    2 / 0              5
 6     -2      -1     -2 / 21     +1 / 4      +2: 1                    1 / 0              6
 7     -1      -1     -1 / 19      0 / 6      +2: 1, +1: 1             2 / 1              6    the freeze breaks
 8     -1      -2     -1 / 18     -2 / 8      +1: 1                    1 / 0              6    B's wave is at B's tower
```

In turns 1–6, B gets 11 XP at range 3 and 0 at range 2, while A gets 31 at both. The freezer's front grows from 6 to
21 HP because nobody kills it, and in turn 7 the wave starts walking to B.

**Example 2 — walk up, same seed.** B stands at +1 and A's minions die at +2 (3–4 HP per turn). B gets 3–4 XP per turn
at every range, and the freeze holds through all 10 turns shown.

**Example 3 — a crash defended behind the enemy tower** (neutral start, A plays Clear + Hold and B plays Hold + Hold,
first seed). In turn 5, A crashes 12 HP under B's tower. In turns 6–7, A stands at the lane cell (0) while B's
defenders, engaging the crash from B's −3, die at A's +3. That is 3 cells away, so A gets 3 + 2 XP at range 3 and 0 at
range 2. Over 14 turns, A misses 6 XP at range 2.

**Example 4 — the farm-spot snap** (aggregate of `base_s0` self-play, 8 games of 100 turns). A champion at the lane
plays Clear + Hold, and its old front dies during the turn's probe. Its farm spot then snaps to its fresh wave (−4,
clipped to −2), and it moves 2 cells back into its own tower cell. The enemy minions it was fighting die at 0, 2 cells
away. This happens to 58.6 HP per game per 100 turns, about 7.5% of both champions' available XP: credited at range
≥ 2, missed at range 1. An 8-game trace attributes three quarters of it to Clear + Hold played from the lane cell.

### 2.3 Where B's available XP dies (range-3 play, A = threshold freezer, equal power)

HP per 40-turn game, the largest (B cell, death cell) entries. In bold: entries at distance 3, the ones range 2 newly
denies (distance 4 or more is already denied at range 3).

| B's response | largest entries |
|---|---|
| walk up | (+1, +2) 51.9 · (−2, −2) 45.7 · (−2, −1) 24.6 · (−1, 0) 12.5 · (+1, +1) 5.5 · (0, +2) 2.7 |
| step back | (−2, −1) 59.8 · (−2, −2) 46.0 · **(−1, +2) 20.6** · (−1, +3) 10.0 · (−1, +1) 9.6 · (−1, 0) 9.2 |
| retreat | (−2, −1) 66.1 · (−2, −2) 46.0 · (−2, +2) 19.7 · **(−2, +1) 10.2** · (−2, 0) 10.0 · (−2, +3) 4.6 |
| Clear + Hold | (+1, +2) 63.7 · (0, +1) 50.3 · (+1, +1) 15.9 · (0, +2) 12.6 · **(0, +3) 6.5** · (+1, +3) 5.4 |
| learned (base s0) | (+2, +2) 68.4 · (+1, +1) 24.6 · (+1, +2) 21.3 · (0, +1) 17.0 · (+2, +3) 12.9 |

The kills at B's −2 and −1 are minions that die after the freeze collapsed toward B, or after a Recall. The learned
breaker follows its crash to the enemy tower (+2) to Pressure, so it stays in range at every range.

---

## 3. Controlled freeze results

**Setup.** The `freeze` start of the M4.2e-B / M4.3c harness: turn 51, both champions level 10, nothing invested, B's
wave held at A's −1. It is built once under the baseline configuration, so every range starts from the identical state.
There are 64 paired seeds per cell. These are the freeze harness's own seeds, so the range-3 cells reproduce
`results/m43c/freeze.json` exactly (largest deviation 0.0).

A either freezes with the threshold controller ("freezer"), freezes and also trades with a B that starts within reach
("punishing freezer"), or Clears ("no freeze"). B keeps one response for 40 turns, then both sides play Hold/Hold for
20 more turns (the extension). The gaps tested are equal power and A ahead (×1.76 attack power, 50 resources invested
against 0). Every policy Recalls at 40% HP.

Columns give ranges 3 / 2 / 1. "Missed-XP gap" is A's missed XP minus B's (positive favours B); the rest of the XP gap
is delayed. 95% half-widths: gold ±0.2–0.8, XP gap ±1.3–5.2, value ±0.4–0.9. Paired changes between ranges are much
tighter (§1).

**3a. Threshold freezer, equal power**

| B's response | gold B − A | XP B − A | of which missed-XP gap | B missed | A missed | value (gold + XP) |
|---|---|---|---|---|---|---|
| walk up | +0.20 / +0.22 / +0.22 | +0.4 / +9.9 / +12.5 | −1.9 / +7.6 / +10.0 | 4.7 / 7.6 / 11.9 | 2.8 / 15.2 / 21.8 | +0.26 / +1.46 / +1.79 |
| step back | −4.89 / −4.88 / −4.89 | −2.5 / −10.2 / −11.6 | −6.8 / −14.8 / −16.2 | 10.2 / 31.9 / 42.8 | 3.4 / 17.1 / 26.6 | −5.21 / −6.16 / −6.34 |
| retreat | −5.03 / −5.02 / −5.03 | −17.9 / −16.7 / −17.4 | −20.7 / −19.6 / −20.2 | 24.5 / 36.7 / 46.9 | 3.8 / 17.1 / 26.6 | −7.26 / −7.11 / −7.21 |
| Clear + Hold | +0.72 / +0.72 / +0.67 | −7.4 / −15.2 / −29.7 | −5.2 / −12.9 / −27.1 | 5.5 / 13.3 / 31.6 | 0.3 / 0.3 / 4.5 | −0.21 / −1.18 / −3.04 |
| Clear + punish + Pressure | +4.70 / +4.72 / +4.86 | −2.9 / −3.8 / −11.0 | −2.3 / −3.1 / −10.1 | 8.0 / 10.0 / 17.8 | 5.7 / 6.9 / 7.7 | +4.34 / +4.24 / +3.49 |
| learned, base s0 | +4.79 / +4.77 / +4.79 | −11.4 / −13.2 / −15.4 | −6.1 / −7.9 / −10.6 | 7.6 / 9.5 / 13.2 | 1.5 / 1.7 / 2.6 | +3.37 / +3.12 / +2.87 |
| learned, ent03 s0 | +5.85 / +5.83 / +5.89 | −4.6 / −4.8 / −6.5 | +0.4 / +0.3 / −1.8 | 5.1 / 5.2 / 6.8 | 5.5 / 5.5 / 5.0 | +5.28 / +5.23 / +5.07 |

**3b. Punishing freezer, A ahead ×1.76 (the realistic combat-power advantage)**

| B's response | gold B − A | XP B − A | of which missed-XP gap | B missed | A missed | value (gold + XP) |
|---|---|---|---|---|---|---|
| walk up | +0.02 / −0.01 / +0.01 | −4.1 / +3.5 / +9.3 | −5.8 / +1.8 / +7.5 | 8.4 / 10.9 / 14.0 | 2.6 / 12.7 / 21.5 | −0.49 / +0.42 / +1.17 |
| step back | −3.23 / −3.33 / −3.33 | −1.6 / +6.7 / +12.5 | +3.6 / +12.2 / +18.1 | 0.6 / 14.0 / 17.3 | 4.1 / 26.2 / 35.3 | −3.43 / −2.50 / −1.77 |
| retreat | −4.06 / −4.06 / −4.06 | −18.2 / +0.7 / +7.1 | −7.3 / +11.5 / +18.0 | 10.1 / 14.1 / 16.1 | 2.8 / 25.6 / 34.1 | −6.33 / −3.98 / −3.17 |
| Clear + Hold | +0.83 / +0.82 / +0.80 | −14.3 / −21.3 / −25.2 | −10.2 / −17.0 / −20.5 | 11.2 / 18.1 / 33.4 | 1.0 / 1.0 / 12.9 | −0.97 / −1.84 / −2.36 |
| Clear + punish + Pressure | +0.55 / +0.54 / +0.62 | −22.3 / −18.1 / −13.8 | −10.0 / −5.8 / −1.5 | 21.1 / 23.4 / 32.6 | 11.0 / 17.6 / 31.2 | −2.24 / −1.72 / −1.11 |
| learned, base s0 | +5.39 / +5.40 / +5.30 | −18.9 / −19.5 / −25.4 | −13.2 / −13.7 / −19.4 | 15.8 / 16.4 / 23.8 | 2.6 / 2.7 / 4.4 | +3.02 / +2.96 / +2.13 |
| learned, ent03 s0 | +5.29 / +5.26 / +5.24 | −13.1 / −13.8 / −18.7 | −8.2 / −8.9 / −14.3 | 14.2 / 14.7 / 22.0 | 6.0 / 5.8 / 7.7 | +3.65 / +3.53 / +2.90 |

With A ahead, the freezer cannot hold the wave against a B who steps back (held 6 of 40 turns). Its stronger wave
crashes into B's tower, and the freezer, standing at the lane, misses the defenders dying at +3. At range 2 the
step-back therefore loses *less* XP than the freezer does.

**3c. Outcomes that do not depend on the range** (range 3; across ranges they differ by < 0.1 bars, < 0.1 deaths and
< 0.5 held turns)

| scenario | B's response | B's mean cell | B HP lost (bars) | B deaths | B ganked (P per present turn) | trades connected B / A | turns held of 40 |
|---|---|---|---|---|---|---|---|
| freezer, equal | walk up | −0.4 | 0.88 | 0.09 | 2.3 (0.061) | 0 / 0 | 18.4 |
| | step back | −1.6 | 0.08 | 0 | 0.8 (0.016) | 0 / 0 | 8.6 |
| | retreat | −2.0 | 0.03 | 0 | 0.3 (0.007) | 0 / 0 | 10.0 |
| | Clear + Hold | +0.4 | 1.44 | 0.12 | 3.9 (0.107) | 0 / 0 | 12.3 |
| | learned | +1.0 | 2.06 | 1.09 | 4.5 (0.125) | 1.5 / 0 | 11.2 |
| punishing, A ahead | walk up | −0.7 | 1.33 | 0.23 | 1.9 (0.054) | 0 / 3.8 | 15.4 |
| | step back | −1.7 | 0.11 | 0 | 0.8 (0.018) | 0 / 1.0 | 6.1 |
| | Clear + Hold | +0.2 | 2.50 | 0.52 | 3.2 (0.104) | 0 / 7.2 | 13.3 |
| | learned | +1.1 | 4.99 | 3.75 | 3.4 (0.124) | 1.5 / 21.5 | 12.2 |

**Persistence.** Missed XP never comes back, so between turn 40 and turn 60 the missed-XP gap changes only through new
misses. Examples: step back at range 2, −14.8 → −19.4; walk up at range 2, +7.6 → +8.2; Clear + Hold at range 1,
−27.1 → −28.3; learned at range 2, −7.9 → −3.8 (the freezer misses more in the extension). The raw
XP gap, by contrast, falls by about 20 XP during the extension at every range. That is the delayed part: releasing the
freeze sends B's piled-up wave and crashed mass under A's tower, where they die on A's ledger (22 HP pending to A
against 8 to B at turn 40). So the extension is not a neutral catch-up period, and persistence has to be read from the
missed/delayed split.

---

## 4. Neutral-lane control

**4a. Scripted, `neutral` start** (turn 9, both at the lane, 64 paired seeds, 40 turns; values per side, A / B)

| A / B | r | XP | missed to range | missed while away | gold | XP B − A |
|---|---|---|---|---|---|---|
| Hold / Hold | 3 | 156.0 / 156.9 | 0.0 / 0.2 | 3.8 / 3.0 | 16.03 / 16.19 | +0.8 ± 3.6 |
| | 2 | 153.8 / 153.8 | 2.2 / 3.3 | 3.8 / 3.0 | 16.03 / 16.19 | +0.0 ± 4.3 |
| | 1 | 151.6 / 151.3 | 4.5 / 5.7 | 3.8 / 3.1 | 16.04 / 16.19 | −0.2 ± 5.0 |
| Clear / Clear | 3 | 162.6 / 162.7 | 0.1 / 0.2 | 3.2 / 2.8 | 15.65 / 15.41 | +0.1 ± 1.3 |
| | 2 | 159.3 / 160.4 | 3.2 / 2.4 | 3.2 / 2.8 | 15.61 / 15.44 | +1.1 ± 1.5 |
| | 1 | 143.9 / 143.8 | 18.8 / 19.1 | 3.4 / 3.0 | 15.73 / 15.56 | −0.1 ± 1.4 |
| Clear / Hold | 3 | 162.7 / 159.5 | 0.6 / 0.0 | 5.8 / 1.9 | 15.21 / 14.01 | −3.2 ± 1.5 |
| | 2 | 150.8 / 159.8 | **12.6** / 0.0 | 5.5 / 1.9 | 15.21 / 14.08 | **+9.0 ± 1.8** |
| | 1 | 135.5 / 157.5 | **28.0** / 2.4 | 5.4 / 1.9 | 15.22 / 14.10 | **+22.0 ± 1.8** |

**4b. Whole games from the initial state** (160 turns, 32 paired seeds per matchup and range, new seed root 75,000;
values per side, both seats pooled; the 95% half-width of \|XP diff\| is ±1.6–5)

| matchup | XP missed to range by turn 100 (% of available) r = 3 / 2 / 1 | \|XP diff\| at turn 100 | turn of level 11 | turn of level 16 |
|---|---|---|---|---|
| base s0 self-play | 0.5 (0.1%) / 2.3 (0.6%) / 37.0 (9.4%) | 5.9 / 7.8 / 8.8 | 57 / 57 / 63 | 114 / 114 / 127 |
| base s1 self-play | 0.0 / 0.6 (0.2%) / 24.8 (6.3%) | 9.1 / 7.9 / 8.5 | 57 / 57 / 61 | 114 / 114 / 122 |
| ent03 s0 self-play | 0.1 / 1.1 (0.3%) / 15.8 (4.0%) | 8.1 / 8.6 / 7.8 | 57 / 58 / 60 | 115 / 115 / 120 |
| base s0 vs `farm` | 2.6 (0.7%) / 4.4 (1.1%) / 9.4 (2.4%) | 7.9 / 9.9 / 13.8 | 57 / 57 / 59 | 114 / 114 / 116 |
| base s0 vs `clear_punish` | 1.0 (0.2%) / 3.2 (0.8%) / 31.2 (7.9%) | 10.6 / 10.7 / 15.6 | 59 / 59 / 63 | 116 / 117 / 127 |
| `farm` vs `farm` | 0.6 (0.1%) / 18.7 (4.8%) / 33.5 (8.6%) | 11.7 / 17.1 / 22.4 | 58 / 60 / 62 | 114 / 119 / 124 |
| `clear_punish` vs `clear_punish` | 2.2 (0.6%) / 8.1 (2.1%) / 41.5 (10.6%) | 9.2 / 10.4 / 14.9 | 60 / 61 / 67 | 120 / 122 / 137 |

For comparison, XP missed while away is 8–27 per side per 100 turns at every range. The learned policy's gold margin
over the references at turn 160 does not change (against `farm`: +15.75 / +15.73 / +16.13).

The neutral losses come from no positional pressure at all. At range 2 they are the crash geometry: a champion at the
lane while its crash is defended behind the enemy tower (`farm` vs `farm`: 31.8 HP per game at (0, +3)). At range 1
they are the farm-spot snap of Example 4 (base s0 self-play: 58.6 HP per game at (−2, 0)).

---

## 5. Unintended consequences

1. **A tax on crashes the champion does not follow, not on conceding.** At range 2 the largest new losses belong to
   whoever crashes a wave and stays at the lane: the clearer in Clear vs Hold (12.6 XP per 40 turns), the freezer
   once its wave pushes out (13–26), and scripted last-hitters (4.8%). The learned policy is barely touched because it
   walks to the enemy tower to Pressure. This works against Clear-without-Pressure only, not against Clear dominance.
2. **The freezer is taxed as well.** Against a champion who walks up, the freezer's value falls by 1.2 (range 2) and 1.5
   (range 1). With a stronger freezer, range 2 *helps* the champion who steps back.
3. **The denial window is short.** A freeze in this game survives only while the frozen-on champion last-hits. When B
   steps back, the wave reaches B within about 6 turns, whatever the range.
4. **Range 1 breaks ordinary progression.** It turns the farm-spot rule's step back to the fresh wave into an XP loss
   of 2–11% in every matchup, delaying level 16 by up to 17 turns.
5. **Off-distribution levels for the frozen policies** at range 1: by turn 100 they are up to 0.6 levels lower than in
   training (0.9 for the scripted `clear_punish` pair). At range 2 the learned policies stay within 0.1 levels.

Making freezing more profitable was not taken as evidence of a better model. Here neither range makes freezing more
profitable against any response B would choose.

---

## 6. Recommendation

**Keep `xp_range = 3`.** Range 2 does not create the intended dilemma. The cell a contesting champion stands in (+1,
next to the freezer's tower) is 1 cell from where the freezer's minions die, so no XP range can make contesting cost XP,
and the only price of contesting stays HP. What range 2 adds is a penalty on the response nobody should choose (stepping
back), an XP tax on unfollowed crashes, and an XP loss for the freezer. Range 1 also disrupts neutral laning.

**Investigate another mechanic instead.** The screen narrows the causes listed in the design document (§4.1, D1–D4):

- **The price of contesting.** Walking up costs 0.05–0.06 gank probability per turn, 0.75–1.3 HP bars and 0.05–0.23
  deaths per 40 turns, and a death is a cheap reset (design §4.2). The candidates are the existing D1 levers (bounty,
  timer schedule) and D2 (the tower, or the freezer's minions, threatening a champion that trades next to them). They
  can be screened with this harness exactly as here.
- **Freeze stability against a champion who concedes.** Hold kills at least 1 HP per turn and a champion cannot hold its
  own wave back, so a freeze collapses toward a champion who stops last-hitting. As long as that holds, *any* denial
  mechanic, XP or gold, lasts only a few turns before the wave goes to the conceding side. This is a wave-mechanics
  question, to be decided before any denial lever is tuned.

**Stopped here.** No PPO training; the default was not changed.

---

## 7. Reproduction, files, limitations

```bash
python -m lanegame.experiments.m43c xp_range --workers 9      # 42 s; writes results/m43c/xp_range.{json.gz,md}
python -m pytest tests/test_m43c_diagnostics.py -k xp_range
```

| item | content |
|---|---|
| `lanegame/experiments/m43c/xp_range.py` | the screen: configuration per range, per-turn ledger records, window statistics, the two stages, the tables |
| `results/m43c/xp_range.json.gz` | every cell (all windows, summed histograms, per-game values for paired differences, the example traces) and every matchup |
| `results/m43c/xp_range.md` | all tables, including both freezers at both gaps, the no-freeze baseline, paired changes and the full geometry tables |
| `results/m43c/xp_range.log` | run log |
| `tests/test_m43c_diagnostics.py` | ledger identity and histogram pricing at ranges 3, 2 and 1; range 3 reproduces the freeze harness |

**Configuration.** `GameConfig.m4_lasthit(xp_range=r)`. The retreat response also sets `safe_line = −2` in its own cells
only. That would also move a low-HP walk-back to −2, but no side walked back in those cells (B lost 0.03 bars). Engine,
presets, observation and checkpoints are unchanged, and `m4_lasthit().xp_range` stays 3 (pinned by the test).

**Seeds.** Freeze cells use the freeze harness's roots (73,000, keyed by gap and episode), so range 3 reproduces
`freeze.json`. The whole games use the new root 75,000.

**Limitations.**
- There is one freeze start (turn 51, level 10, nothing invested), and each response is fixed for 40 turns.
- The learned policies were trained at range 3. A policy trained at range 2 might learn to follow its crashes, which
  would shrink side effect 1.
- 64 seeds per freeze cell, used for exact reproduction, and 32 per game matchup. Small XP effects of ±2–5 XP per 40
  turns are within noise; the paired changes in §1 are not.
- The value is an undiscounted 40-turn window, as in the earlier freeze harness.
