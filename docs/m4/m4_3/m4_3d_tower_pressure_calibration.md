# M4.3d — Tower Farming, Pressure and Gank Risk

**Status.** Investigation complete; nothing adopted as a default. The own-tower collection rate of the M4.3c candidate
stays **0.8**. One EXPERIMENTAL candidate preset is added, `GameConfig.m4_four_actions_econ_aggro()`, which removes the
tower's "busy" discount for a champion that dives. `m4_lasthit()`, `m4_four_actions()`, `m4_four_actions_econ()`
and every other preset are unchanged, and so are all checkpoints and earlier results. Rules are cited from the
[mathematical reference](../laning_game_mathematical_reference.md) ("Reference §n"); this study added Reference §8.8 and
measured notes in §5.3 and §6.3.

Every number below comes from the engine, through the real environment (`LaneEnv`), with scripted strategies on paired
seeds. Labels: **CURRENT** = `m4_four_actions_econ()` ("econ"), the starting point; **EXPERIMENTAL** = an opt-in
configuration in the code; **RECOMMENDED** = what this report proposes for the next PPO milestone; **DIAGNOSTIC** = a
variant run only to isolate a cause. Reproduction: §11.

---

## 0. Executive findings

**What was actually wrong or missing?** Less than the brief suspected, and not where it suspected.

1. **The own-tower cap is unconditional, but at 0.8 it is not a penalty in practice.** The defender loses 1/9 of
   every crashed HP to the cap even with nobody near it. But normal lane farming also realizes only about 0.90 of the
   lane rate (reach, posture and absence losses), and tower farming has none of those losses. An uncontested crash
   therefore leaves the defender's farm unchanged: 0.91 of the lane rate against 0.90 in the lane, +0.1 to +0.3
   resources per 40 turns. The loss is permanent, not displaced to the bounce (§3).
2. **0.9 overcorrects.** At the lane rate, being crashed on raises the defender's farm about 9% (+1.2 to +1.5 per 40
   turns), taking a tower's plates barely pays (+0.3 J at level 10, −0.1 at level 4), and a ×1.76 lead after a
   crash is worth +0.7 instead of +2.7. Thin beats Shove at equal power (−0.2), and breaking a freeze *helps* the
   freezer (+2.0). Once the plates are gone, shoving loses to Last Hit and to Thin in every neutral-lane game (−0.5 to −2.7). The extra
   action diversity at 0.9 comes from crashes being gifts. 0.85 lies in between on every measure (§3.3, §6).
3. **Interaction-based denial exists, at the tower only through dives.** A defender in its tower cell can be
   traded only by a dive (the contact rule), so "harassment from outside the tower" does not exist in this game.
   Real denial comes from dives (absence 6–12% of the defender's farm), from the defender's own Safe response (4–13%
   posture outside the tower cell) and from deaths. Fighting back from the tower costs the defender more farm than
   it gains (§3.4).
4. **Pressure is not under-penalized by ganks.** An equal champion that keeps pressuring the enemy tower without
   vision spends about 60% of its turns there and loses 2.4 bars, 1.6 Recalls and 0.4 deaths per 40 turns. The paired
   counterfactual prices this at **3.0–4.0 resources per 100 turns**, more than its net gain (+0.5 to +3.2, which is
   the plate gold). Vision halves that price (−3.7 → −1.9) and enters the stronger side's equilibrium. Gank risk
   depends on cell, behaviour and vision as documented, on wave position only through where the champion stands,
   and never on the opponent (§2.4, §4).
5. **The under-priced hazard is the tower for a strong diver.** The tower's "busy" discount (shot probability 0.5
   instead of 1.0 while the diver's minions are under the tower) applies to nearly every dive that occurs in play,
   and it reverses League's tower aggro. Under `m4_four_actions_econ()` a ×1.76 champion's dive against a tower
   farmer pays more than pressuring without diving, on both seed sets (+4.5 against +2.7; independent seeds +3.2
   against +2.6), and it dies no more often (§4.3, §7.3). On the main seeds the dive is even its most profitable
   aggression; on the independent seeds Vision-then-Pressure (+4.1) is.
6. **Gank and tower sensitivity.** Removing the busy discount (`tw_aggro`) changes only dives: the strong dive falls
   to about the value of plain Pressure or below (+2.8 against +2.7; independent seeds +1.9 against +2.6), still
   viable, with twice the deaths; equal dives fall to −4.5. No equilibrium of any restricted game changes. The gank variants (steeper exposure, more lethal ganks) tax every forward position
   and the neutral lane alike, and correct no diagnosed error (§5).
7. **Shove's dominance at equal power is now plate gold, not the farm cap.** With plate gold off (DIAGNOSTIC: a
   tower whose plates have fallen), at 0.8 plain Shove loses to Thin and Shove with Vision is the equilibrium. The
   weak side then concedes to a ×1.76 freezer (cautious 0.69, respect 0.21), which M4.3c could not establish. And
   adding Vision strategies overturns M4.3c's "weak side walks up and Thins": the stronger side pressures under
   vision and the weak side breaks by shoving back (§6).

**Decision.** Keep `collect_contest_own_tower` = 0.8. Change no gank parameter. Remove the busy discount for divers,
as the EXPERIMENTAL candidate `GameConfig.m4_four_actions_econ_aggro()` (`tower_shot_p_busy` 1.0,
`tower_adjacent_p_busy` 0.5; §7). It equals `m4_four_actions_econ()` in every game without a dive, in every
restricted-game equilibrium and in every value of the M4.3c scenarios.

**Recommendation.** The environment is ready for substantial four-action PPO: run it on
`m4_four_actions_econ_aggro()` against `m4_four_actions()` (optionally `m4_four_actions_econ()` as a dive-learning
control), and read Vision use, dive frequency, crash-and-leave frequency and Shove before and after the plates fall
(§10). The remaining structural limits (contact geometry at the tower, ganks independent of the opponent) are
documented in §9, not tuned.

---

## 1. Scope and method

| item | what | where |
|---|---|---|
| audit | the risk and collection rules evaluated by the engine's own functions; a Monte Carlo of the gank and tower rules alone | `experiments/m43d/audit.py` → `results/m43d/audit.{json,md}` |
| farm-loss ledger | each side's lost farm split into **tower cap** (unconditional), **position**, **posture**, **trade** and **absence**; risk ledger (cell, exposure, expected vs realized gank damage, tower and trade damage) | `experiments/m43d/ledger.py` |
| post-crash strategies | Shove and leave / Recall / stay / Pressure / harass / hunt / dive / dive on a positive duel edge / Vision then Pressure / Vision then dive / Trade every turn (only existing actions) | `experiments/m43d/strategies.py` |
| crash grid | configuration × start × power gap × 12 attacker × 7 defender strategies, 64 paired games of 100 turns per cell | `experiments/m43d/grid.py` → `results/m43d/grid_*.json.gz` |
| chance masks | the same games with the attacker immune to ganks or to tower shots (the uniforms are drawn, then set to "no event"): the paired economic cost of each hazard | `experiments/m43d/common.MaskedChance` |
| restricted games | the M4.3c matrix games (freeze, neutral, freeze phase) with Vision, dive and Recall strategies added; the crash-scenario games from the grid | `experiments/m43d/strategic.py` |
| candidate | `GameConfig.m4_four_actions_econ_aggro()` | `lanegame/config.py` |

**Starts** (the M4.3c harness starts, both champions at full HP at the given level, no gold): **S4 / S10** =
`siege` at level 4 / 10, one turn after A's crash. A's 14-HP mass is under B's tower, A is in B's tower cell, A's
next 8-HP front is at the lane cell, and B is in its tower cell with its next wave at its base. **N4 / N10** =
`neutral`: both 8-HP fronts and both champions at the lane cell. **Gaps:** equal; "A ×1.76" (50 resources invested
against none, duel ratio ×3.1); "A weak" (B ×1.76).

**Strategy sets.** Attacker A (all Shove except the control): `lh` (Last Hit, the control), `sh_leave` (Safe while
crashed), `sh_recall` (walk back and Recall while crashed), `sh_stay` (Hold, no objective), `sh_press` (Hold +
Pressure: stands in B's tower cell while sieging), `sh_harass` (Pressure + Trade an enemy that starts in reach, no
dives), `sh_hunt` (Pressure + chase on a positive duel edge, no dives), `sh_dive` (Pressure; Trade while crashed:
dives a defender in its tower cell), `sh_dive_edge` (the same only on a positive duel edge), `sh_vis` (Vision when
none is active, else Pressure), `sh_vis_dive` (Vision first, then dive), `sh_bully` (Trade every turn, in the lane
too). Defender B: `lh` (Last Hit, farms the crash in its tower cell), `thin`, `safe` (never ahead of its −1),
`respect` (Safe while the enemy is in reach, else Yield), `punish` (Trade an attacker that starts in reach,
including a sieger in B's tower cell), `fight` (Thin + Trade back on a non-negative edge), `shove` (Shove + Pressure:
contest by shoving back). Every strategy recalls at 40% HP (the M4.3c pre-rules).

**Volume.** Part 2: 3,360 cells (4 own-tower rates × 4 starts × 3 gaps × 70 pairs at the first run, then the dive
strategies re-played). Part 3: 1,152 masked cells. Part 4: 3,780 cells. Part 5: 4 × 807 + 807 matrix-game cells and
the crash-scenario games. Over 100 million game turns in all. Seeds: SeedSequence roots 79,000 (grids), 79,200
(matrix games), 79,500 / 79,700 (independent validation, 128 games), 79,900 (traces).

**Instrument limits.** A restricted game's equilibrium is an equilibrium of a handful of scripts, not of the laning
game. 64 paired games give about ±0.4–0.6 on a cell's J (the 95% intervals are in the result files). The farm ledger
is exact: its parts sum to the ideal farm in every turn (asserted while playing, and tested).

---

## 2. Current mechanics (Part 1)

All cells are in the acting side's own coordinates (own tower −2, lane 0, enemy tower +2). The code is authoritative;
the numbers are produced by the engine's functions on `m4_four_actions_econ()` (`results/m43d/audit.md`).

### 2.1 Farming and collection

A kill of an enemy minion at own cell $x$ pays the collector in mode $\mu$ standing at $p$
(`champions_m4.collection_units`, Reference §5.3)

$$
F=0.125\cdot\mathbb 1\bigl[\lvert x-p\rvert\le\rho_\mu\bigr]\cdot\min\bigl(b_\mu(x),\ c_{\text{ot}}\,\mathbb 1[x=-2]+\mathbb 1[x\ne-2]\bigr)\cdot\bigl(0.8\ \text{if trading}\bigr),
$$

with $b=0.9$ (farm or trading mode), $b=0.8-0.2(x+2)$ for Safe or retreat (0.8 at −2, 0.6 at −1, 0.4 at 0), reach
$\rho=1$ (2 while retreating), and $c_{\text{ot}}$ = `collect_contest_own_tower` = 0.8. There is no presence zoning
(`zoning_rule = "none"`).

| collector and kill | own-tower rate 0.6 | **0.8** | 0.85 | 0.9 |
|---|---|---|---|---|
| Hold in the own tower cell, kill there | 0.60 | **0.80** | 0.85 | 0.90 |
| Hold at −1, kill in the tower cell | 0.60 | 0.80 | 0.85 | 0.90 |
| Safe in or next to the tower, kill in the tower cell | 0.60 | **0.80** | 0.80 | 0.80 |
| Safe at −1, kill at −1 | 0.60 | 0.60 | 0.60 | 0.60 |
| trading in the tower cell, kill there | 0.48 | 0.64 | 0.68 | 0.72 |
| Hold at the lane, or in the enemy tower cell (kill one cell behind it) | 0.90 | 0.90 | 0.90 | 0.90 |

**Where the defender stands during a siege.** A Hold or Safe champion goes to its farm spot, the cell its own front
reaches in the probe (Reference §5.2). During a siege the defender's front waits at its −3, so the spot is clipped to
the own tower cell. The crashed mass dies there, within reach.

### 2.2 Contact, trades and tower shots

* **Contact** (`champions.in_contact`): within one cell, except that a champion in its **own tower cell can be reached
  only from inside that cell**, i.e. by a dive. A Trade chases the nearest contact cell (two cells per turn).
* **Trade hit** (Reference §6.2): $h=\operatorname{clip}(0.55+\beta_{\text{defender}}+0.05\Delta\sigma-0.15\,\mathbb 1[\text{defender in its tower cell}],0.05,0.95)$
  with $\beta$ = 0 Hold/Trade, −0.25 an evading Yield, −0.40 Safe. A Hold defender in its tower cell is hit with
  0.40 and a Safe one with 0.05. Damage is 1 or 2 base units at the attacker's power, divided by the victim's
  $M(S)$. Only a defender that also Trades hits back.
* **Tower shot** (Reference §6.3): only a **connected trader** is ever shot, with p = 1.0 if it ends in the enemy
  tower cell (a dive) or 0.5 if it ends next to it. Both are **halved while its own crashed minions are under that
  tower or crash this turn** ("busy": 0.5 / 0.25). A shot deals one HP unit $u(L)=H(L)/6$, i.e. 1/6 of a bar.
  A champion that farms, Pressures or sets Vision in the enemy tower cell is never shot.
* **Pressure** (objective 1) moves the champion to the enemy tower cell, adds $\lfloor400AP^\ast\rceil$ to the tower
  damage and is legal only while the own crashed mass is there. **Shove** is a wave action: the full capacity $C$,
  plus 0.02 gank occurrence unless Safe. **Trade** is a behaviour: chase and hit; a connected trader deals no wave
  damage, collects at ×0.8 and suspends its objective.

### 2.3 Ganks and vision

Occurrence (`champions.champion_gank_probability`), with exposure $e=\max(\text{start cell},\text{end cell})$:

$$
p_{\text{gank}}=\operatorname{clip}\Bigl(0.12\tfrac{e+2}{4}+\beta_g+0.02\,\mathbb 1[\text{Shove, not Safe}]-0.01(\sigma_{\text{own}}-\sigma_{\text{enemy}}),\,0,\,0.30\Bigr)\cdot(1-0.75\,\mathbb 1[\text{vision}]),
$$

with $\beta_g$ = 0.01 Hold or Yield, 0.03 Trade, 0 Safe. Damage given a gank (`gank_damage_p`) is
$X\sim\mathrm{Binomial}(4,p_e)$ HP units, $p_e=\operatorname{clip}(0.2+0.6\tfrac{e+2}{4}-0.2\,\mathbb 1[\text{Safe or retreating}],0,1)$.

| exposure | own tower −2 | −1 | lane 0 | +1 | enemy tower +2 |
|---|---|---|---|---|---|
| P(gank): Hold, Last Hit | 0.01 | 0.04 | 0.07 | 0.10 | 0.13 |
| P(gank): Hold, Shove (Pressure) | 0.03 | 0.06 | 0.09 | 0.12 | 0.15 |
| P(gank): Trade, Shove (a dive) | 0.05 | 0.08 | 0.11 | 0.14 | 0.17 |
| P(gank): Safe | 0.00 | 0.03 | 0.06 | 0.09 | 0.12 |
| with Vision (any row) | ×0.25 | | | | |
| E[gank damage], bars (not defensive) | 0.13 | 0.23 | 0.33 | 0.43 | 0.53 |
| P(gank deals 4 units = 2/3 of a bar) | 0.002 | 0.015 | 0.06 | 0.18 | 0.41 |
| expected gank damage per turn, Hold + Shove, bars | 0.004 | 0.014 | 0.030 | 0.052 | 0.080 |

**Vision** (Establish Vision, legal only during the own siege): occurrence ×0.25 for the next 6 turns. It has no
effect on damage.

**Hazard of standing at one cell** (Monte Carlo of the gank and tower rules alone, level 10, per 40 turns; the
champion recalls at 40% HP after walking back with Safe; `audit.hazard`):

| case | turns at the cell | ganks | bars lost | forced Recalls | deaths | turns away |
|---|---|---|---|---|---|---|
| own tower cell, Hold, Last Hit | 40.0 | 0.40 | 0.05 | 0.01 | 0.00 | 0.0 |
| lane cell, Hold, Last Hit | 37.5 | 2.67 | 0.88 | 0.79 | 0.04 | 1.7 |
| enemy −1 (own +1), Hold, Shove | 34.2 | 4.25 | 1.82 | 1.62 | 0.24 | 4.1 |
| enemy tower cell, Hold, Shove (Pressure) | 29.8 | 4.83 | 2.50 | 1.88 | 0.64 | 6.1 |
| the same, Vision always | 37.2 | 1.41 | 0.75 | 0.59 | 0.11 | 1.6 |
| enemy tower cell, dive (tower busy) | 22.3 | 4.46 | 4.09 | 3.43 | 0.90 | 10.2 |
| enemy tower cell, dive (tower free) | 18.2 | 3.94 | 4.97 | 4.17 | 1.16 | 12.6 |
| the same at M = 1.76 (50 invested): Pressure / busy dive | 33.3 / 27.2 | 5.30 / 5.19 | 1.57 / 2.80 | 1.51 / 2.84 | 0.11 / 0.29 | 3.4 / 6.7 |

### 2.4 Answers to the audit questions

1. **Does the defender lose collection merely because minions die under its tower, with no enemy present?** **Yes.**
   `collection_units` caps every kill in the own tower cell at `collect_contest_own_tower` in every mode,
   independent of the opponent, who may be away or dead. Under econ this costs 1/9 of the lane rate on every crashed
   HP. It is the only collection loss that depends on no decision or interaction.
2. **What further losses can actual pressure cause?** Five mechanisms, all from choices and consequences:
   (i) **posture**: Safe or a retreating Yield collects at the Safe base. In the own tower cell that base is 0.8,
   so Safe costs nothing extra at a cap ≤ 0.8 and 1/9 at 0.9; at −1 it is 0.6. (ii) **position**: standing more than
   one cell from where minions die. (iii) **trading**: ×0.8 and no wave damage. (iv) **absence**: HP lost to trades
   or ganks becomes Recalls and deaths. (v) **bounty** on death. A defender in its tower cell cannot be traded except
   by a dive (§2.2). Presence zoning does not exist.
3. **Does an attacker that stays forward face more gank risk?** **Yes.** Occurrence rises linearly from 0.01 at the
   own tower to 0.13–0.17 at the enemy tower, and damage from 0.13 to 0.53 bars per gank.
4. **Does the risk depend on location, behaviour, vision and wave position?** Location: yes (exposure = the furthest
   of the start and end cells). Behaviour: yes (Trade +0.02 over Hold, Safe −0.01 and less damage, Shove +0.02).
   Vision: yes (×0.25 occurrence). **Wave position: only indirectly.** The champion follows its wave through the farm
   spot; the formula's own wave term is the stack difference of the two fronts (±0.01 per stack) and ignores crashed
   masses. The risk also ignores the opponent: its presence, HP, position and power.
5. **How dangerous is pressuring from just outside the enemy tower (own +1)?** 0.12 ganks per turn with Shove,
   2.6 units each on average. Standing there for 40 turns costs about 1.8 bars, 1.6 forced Recalls and 0.24 deaths at
   M = 1. There is no tower shot unless the champion trades (then 0.5, or 0.25 while busy).
6. **How dangerous is diving?** The dive itself costs 1/6 of a bar per turn with probability 1.0, but in play
   **0.5**: a defender stands in its tower cell almost only during the diver's own siege, so nearly every dive is
   "busy". On top come exposure +2 with Trade (0.17 ganks per turn, 0.53 bars each) and a hit chance against the
   defender 0.15 lower (0.40, or 0.05 against Safe).
7. **Duplicated, inconsistent or disproportionate rules.**
   (a) The **busy discount reverses League's tower aggro**: a tower switches to a champion that attacks an allied
   champion whatever minions are present. And because dives happen during sieges, the discounted rate is the rate in
   play.
   (b) The **tower shot is small next to a gank**: 1/6 of a bar against 0.53 bars expected per gank at the enemy
   tower, so a dive's own tower price is about a gank's.
   (c) The Shove gank term (+0.02) and the exposure term both charge shoving. They partly overlap, but each is small.
   (d) Gank risk ignores the opponent laner, so a pusher is as exposed when the defender is dead as when it is
   present.
   (e) The unconditional cap of question 1.
   No rule was found to be wrong in the code. (a) and (b) are calibration questions (§5); (d) is a structural
   limitation (§9).

---

## 3. Tower-farming experiments (Part 2)

Crash grid on `m4_four_actions_econ()` with only `collect_contest_own_tower` changed: **0.6** (the historical
rate), **0.8** (econ), **0.85**, **0.9** (the lane rate). Efficiency = collected / ideal, where the ideal is every
enemy minion HP on the side's ledger at the lane rate 0.9; the loss shares complete it to 100%. Full tables:
`results/m43d/tower_farm.md`.

### 3.1 The reference: what a champion collects in the lane

Both Last Hit from the neutral lane, 40 turns (B; A is symmetric):

| own-tower rate | efficiency | tower cap | position | posture | trade | absence | deaths | Recalls |
|---|---|---|---|---|---|---|---|---|
| 0.6 | 0.872 | 4% | 3% | 4% | 0% | 2% | 0.06 | 0.66 |
| **0.8** | **0.899** | 1% | 3% | 4% | 0% | 2% | 0.06 | 0.66 |
| 0.9 | 0.912 | 0% | 3% | 4% | 0% | 2% | 0.06 | 0.66 |

**Normal farming realizes about 0.90 of the lane rate, not 1.0.** Reach, the Safe posture of the low-HP walk-back,
and absences cost 9–10% of the ideal.

### 3.2 Uncontested: the attacker shoves and leaves, recalls or stays back (B last-hits)

| start | A | window | rate 0.6 | **0.8** | 0.85 | 0.9 | B's other losses |
|---|---|---|---|---|---|---|---|
| S10 | shove and leave | 12 turns | 0.737 (cap 26%) | **0.912** (9%) | 0.956 (4%) | 0.999 (0%) | none |
| S10 | shove and Recall | 12 turns | 0.780 | **0.927** | 0.963 | 1.000 | none |
| S10 | shove and leave | 40 turns | 0.753 | **0.907** | 0.945 | 0.984 | position 1% |
| S4 | shove and leave | 40 turns | 0.778 | **0.918** | 0.953 | 0.988 | absence 1% |
| S4 | shove and Recall | 40 turns | 0.803 | **0.929** | 0.960 | 0.992 | position 1% |

(B's efficiency; in brackets the share lost to the unconditional cap.) Against an absent, recalling or retreating
attacker, **the cap is the defender's only loss**. B misses no XP (0.0–1.1 of about 160), loses 0.01–0.16 bars of HP,
and never dies.

**Permanent, not displaced.** B's efficiency is the same over 12, 40 and 100 turns (0.912 / 0.907 / 0.893 at 0.8 for
`sh_leave`; 0.999 / 0.984 / 0.973 at 0.9). The enemy minion HP still alive at the end does not differ between rates.
The crashed minions die within the window and are collected at the cap; nothing is deferred to the bounce.

### 3.3 Is receiving a crash a gift?

The decisive comparison is B's farm when crashed on against B's farm in the neutral lane on the same seeds (N
starts, B last-hits, 40 turns; Δ J = A's discounted value over 100 turns minus that of A last-hitting):

| start | own-tower rate | A | B efficiency | Δ B farm (40 turns) | A's plates | Δ J (A) |
|---|---|---|---|---|---|---|
| N10 | 0.6 | shove and leave | 0.746 | −2.15 | 0 | +1.17 ± 0.43 |
| N10 | 0.6 | Shove + Pressure | 0.724 | −2.52 | 2.98 | +7.80 ± 0.51 |
| N10 | **0.8** | shove and leave | 0.896 | **+0.11** | 0 | −3.21 ± 0.38 |
| N10 | **0.8** | shove and Recall | 0.901 | +0.28 | 0 | −3.14 ± 0.37 |
| N10 | **0.8** | Shove + Pressure | 0.898 | +0.18 | 2.98 | **+2.81 ± 0.48** |
| N10 | **0.8** | Shove, Vision, Pressure | 0.894 | +0.04 | 2.97 | +4.52 ± 0.45 |
| N10 | 0.85 | shove and leave | 0.934 | +0.68 | 0 | −4.30 ± 0.38 |
| N10 | 0.85 | Shove + Pressure | 0.942 | +0.86 | 2.98 | +1.57 ± 0.48 |
| N10 | 0.9 | shove and leave | 0.971 | **+1.24** | 0 | −5.40 ± 0.39 |
| N10 | 0.9 | Shove + Pressure | 0.986 | +1.53 | 2.98 | **+0.32 ± 0.48** |
| N4 | 0.8 | Shove + Pressure | 0.897 | +0.15 | 2.94 | +2.20 ± 0.43 |
| N4 | 0.9 | Shove + Pressure | 0.977 | +1.39 | 2.94 | −0.14 ± 0.44 |

Readings:

* **At 0.8 an uncontested crash is farm-neutral for the defender** (+0.1 to +0.3 of about 16 per 40 turns). The
  1/9 cap equals what lane farming loses to reach, posture and absence (§3.1). Shoving pays the attacker exactly its
  plates (+2.8 J ≈ 3 plates minus its risk), plus Vision's protection.
* **At 0.9 a crash is a gift**: being crashed on raises the defender's farm about 9% (+1.2 to +1.5 per 40 turns),
  because tower farming has no reach, posture or gank losses. Taking a tower's three plates then barely pays at
  level 10 (+0.3) and loses at level 4 (−0.1).
* **At 0.6 a crash taxes the defender a seventh of its farm**, the M4.3c diagnosis.
* **Shoving and leaving is a large loss from 0.8 up** (−3.1 to −5.4 J); only the 0.6 tax made it pay (+1.2). The
  attacker gives up the defenders its crashed mass kills behind the enemy tower: they die out of its reach (trace 1
  in `results/m43d/traces.md`). Crashing is a means (plates, a Recall window), not an income.

### 3.4 Contested: the attacker stays forward, harasses or dives

S10, level 10, 40 turns. B's efficiency and loss shares, both champions' HP and deaths, A − B gold and J (rates 0.8 /
0.9; all cells in `tower_farm.md` §2.3):

| gap | A | B | B eff. 0.8 / 0.9 | posture 0.8 / 0.9 | trade | absence | B HP lost | A HP lost | A deaths | J 0.8 / 0.9 |
|---|---|---|---|---|---|---|---|---|---|---|
| equal | Pressure | last-hit | 0.904 / 0.996 | 0 / 0 | 0 | 0% | 0.10 | 2.04 | 0.41 | +1.27 / −1.46 |
| equal | Pressure | Safe | 0.862 / 0.862 | 4% / 14% | 0 | 0% | 0.03 | 2.04 | 0.41 | +1.28 / +1.28 |
| equal | Pressure | punish | 0.719 / 0.786 | 0 | 10% | 2% | 0.44 | 3.49 | 0.94 | +2.94 / +1.07 |
| equal | dive | last-hit | 0.869 / 0.929 | 0 | 0 | 6% | 1.05 | 2.59 | 0.66 | −3.09 / −4.80 |
| equal | dive | Safe | 0.797 / 0.797 | 13% / 20% | 0 | 0% | 0.13 | 2.68 | 0.58 | −1.98 / −1.98 |
| ×1.76 | Pressure | last-hit | 0.895 / 0.995 | 0 | 0 | 0% | 0.09 | 1.46 | 0.12 | +2.74 / −0.19 |
| ×1.76 | dive (edge) | last-hit | 0.804 / 0.875 | 1% | 0 | 11% | 1.66 | 1.89 | 0.08 | **+4.47** / +2.34 |
| ×1.76 | dive (edge) | Safe | 0.834 / 0.834 | 7% / 16% | 0 | 1% | 0.24 | 1.94 | 0.09 | +2.29 / +2.29 |
| ×1.76 | Vision, dive | last-hit | 0.783 / 0.859 | 1% | 0 | 12% | 1.93 | 1.26 | 0.03 | +4.14 / +1.94 |

* **Real denial at the tower comes from dives and from the defender's own responses.** An equal diver costs the
  defender 6% to absence; a ×1.76 diver costs 11–12% and 0.5 deaths per 40 turns. Retreating with Safe costs
  posture outside the tower cell (4–13% at 0.8).
* **Responses depend on power.** Against a ×1.76 edge-diver, Safe more than halves the attacker's gain (+4.47 →
  +2.29), so respecting a stronger diver pays. Against an equal diver, standing and last-hitting is the defender's
  better reply (−3.09 against −1.98 for the attacker).
* **Fighting back from the tower does not pay at equal power.** `punish` trades about 22 times per 40 turns. It kills
  the sieger more often (0.94 deaths) but loses 10% of its own farm to the trade factor, 8% to stepping out of
  position, and its wave damage while connected. A's value *rises* (+1.27 → +2.94). Equal champions do not profit
  from repeated trades, on either side.
* **At 0.9 the defender's Safe response costs 14–20% posture.** This is the interaction-based denial that 0.9 would
  create, but no defender chooses Safe unless dived by a stronger champion: against Pressure, last-hitting is free
  and better.

---

## 4. Gank diagnosis (Part 3)

### 4.1 Where an attacker stands and what it costs

Econ, B last-hits, 40 turns (`results/m43d/pressure.md`; bars of max HP before durability mitigation):

| start, gap | A | turns at +2 | gank bars (expected / realized) | tower bars | deaths | Recalls | away | plates | J |
|---|---|---|---|---|---|---|---|---|---|
| S10 equal | last-hit (control) | 0% | 0.69 / 0.67 | 0 | 0.03 | 0.64 | 1.4 | 0 | −2.84 |
| S10 equal | shove and stay back | 0% | 1.59 / 1.56 | 0 | 0.27 | 1.25 | 3.4 | 0 | −5.48 |
| S10 equal | Shove + Pressure | 60% | 2.47 / 2.36 | 0 | 0.41 | 1.58 | 4.8 | 3.00 | +1.27 |
| S10 equal | Shove, Vision, Pressure | 81% | 1.15 / 1.03 | 0 | 0.12 | 0.77 | 2.0 | 3.00 | **+3.13** |
| S10 equal | dive | 33% | 2.10 / 2.04 | 0.86 | 0.66 | 1.98 | 6.5 | 0.03 | −3.09 |
| S10 equal | dive on a positive edge | 60% | 2.41 / 2.28 | 0.30 | 0.41 | 1.78 | 5.0 | 2.89 | +1.73 |
| S10 equal | Vision, then dive | 40% | 1.19 / 1.08 | 0.78 | 0.19 | 1.56 | 3.8 | 0.06 | −0.51 |
| S10 ×1.76 | Shove + Pressure | 77% | 2.70 / 2.62 | 0 | 0.12 | 1.36 | 3.1 | 3.00 | +2.74 |
| S10 ×1.76 | Shove, Vision, Pressure | 92% | 1.14 / 1.04 | 0 | 0.00 | 0.50 | 1.0 | 3.00 | +4.20 |
| S10 ×1.76 | dive on a positive edge | 50% | 2.51 / 2.36 | 1.02 | 0.08 | 2.09 | 4.5 | 1.89 | **+4.47** |
| S10 ×1.76 | Vision, then dive | 53% | 1.15 / 1.07 | 1.18 | 0.03 | 1.34 | 2.7 | 0.44 | +4.14 |
| N10 equal | Shove + Pressure | 51% | 2.33 / 2.34 | 0 | 0.48 | 1.47 | 4.7 | 2.98 | +3.23 |
| N10 equal | Shove, Vision, Pressure | 68% | 1.19 / 1.16 | 0 | 0.28 | 0.72 | 2.4 | 2.97 | +4.94 |
| S4 equal | Shove + Pressure | 52% | – / 2.33 | 0 | 0.47 | 1.52 | – | 2.94 | +0.54 |
| S4 equal | Shove, Vision, Pressure | 52% | – / 1.18 | 0 | 0.23 | 0.75 | – | 2.94 | +1.96 |

Expected and realized gank damage agree within about 5%. The difference is the strategies' reaction (they recall
after damage), not a bias.

### 4.2 The economic cost of each hazard (paired)

The same games with the attacker made immune to ganks or to tower shots. The cost is J with the hazard minus J
without it (resources over 100 turns, discounted, ± 95%), B last-hitting:

| start, gap | A | J | cost of ganks | cost of tower shots |
|---|---|---|---|---|
| S10 equal | last-hit | −2.84 | −0.66 ± 0.34 | 0 |
| S10 equal | Shove + Pressure | +1.27 | **−3.74 ± 0.38** | 0 |
| S10 equal | Shove, Vision, Pressure | +3.13 | −1.86 ± 0.29 | 0 |
| S10 equal | dive | −3.09 | −5.02 ± 0.66 | −2.21 ± 0.60 |
| S10 equal | dive on a positive edge | +1.73 | −3.26 ± 0.55 | −0.80 ± 0.43 |
| S10 equal | Vision, then dive | −0.51 | −1.63 ± 0.43 | −1.43 ± 0.47 |
| S10 ×1.76 | Shove + Pressure | +2.74 | −2.32 ± 0.23 | 0 |
| S10 ×1.76 | Shove, Vision, Pressure | +4.20 | −0.83 ± 0.13 | 0 |
| S10 ×1.76 | dive | +2.59 | −6.15 ± 0.85 | −3.39 ± 0.77 |
| S10 ×1.76 | dive on a positive edge | +4.47 | −4.13 ± 0.83 | −2.01 ± 0.72 |
| S10 ×1.76 | Vision, then dive | +4.14 | −1.82 ± 0.52 | −2.40 ± 0.52 |
| N10 equal | Shove + Pressure | +3.23 | −3.95 | 0 |
| S4 equal | Shove + Pressure | +0.54 | −3.03 | 0 |

### 4.3 Diagnosis

* **Aggressive positioning is not under-penalized by ganks.** An equal champion that pressures the enemy tower
  without vision pays 3.0–4.0 resources of J in ganks, more than its net gain (+0.5 to +3.2). The gain is the plate
  gold, three resources per tower and finite.
* **Exposure and damage are both priced.** Occurrence alone would mislead: the cost comes through Recalls (1.6 per
  40 turns) and deaths (0.41), i.e. absence and bounty. The defender in its tower cell pays almost nothing (0.03
  deaths per 40 turns).
* **Vision is a real risk-management decision.** It halves the gank cost (−3.7 → −1.9) and raises the pressuring
  attacker's value by +1.9 at equal power and +1.5 at ×1.76. It does not remove the risk (0.12 deaths per 40 turns,
  1.0 bar). Placing it costs one Pressure turn.
* **Power buys safety.** Durability divides gank damage: a ×1.76 attacker's gank cost is −2.3 instead of −3.7, and it
  almost never dies.
* **The under-priced hazard is the tower for a strong diver.** Under econ a ×1.76 champion's dive against a
  last-hitting tower farmer (+4.47 on a positive edge, +4.14 under vision) pays more than pressuring (+2.74) and as
  much as pressuring under vision (+4.20). On the independent seeds of §7.3 the dive is +3.21 against +2.55 and +4.07:
  above plain Pressure on both seed sets, level with or below Vision-then-Pressure. It dies no more often than a
  champion pressuring outside dives (0.08 against 0.12 per 40 turns). The tower costs it about 1 bar per 40 turns, because every in-play dive is
  "busy" (p 0.5) and its durability halves the shot. At equal power dives already lose (−3.1), as they should.

---

## 5. Sensitivity experiments (Part 4)

One existing parameter group per variant, everything else econ, the same seeds and cells (`results/m43d/sensitivity.md`):

| variant | change | why tested |
|---|---|---|
| `tw_aggro` | `tower_shot_p_busy` 0.5 → 1.0, `tower_adjacent_p_busy` 0.25 → 0.5 | the diagnosed under-pricing of dives (§4.3); League's tower aggro |
| `tw2` | `tower_shot_damage` 1 → 2 units | the other lever of the same hazard |
| `tw2_aggro` | both | upper bound |
| `gk_pos16` | gank occurrence slope 0.12 → 0.16 | the brief's hypothesis that forward exposure is too cheap |
| `gk_dmg08` | gank damage slope 0.6 → 0.8 (p_e 1.0 at the enemy tower) | the same, through lethality |

J of A against a last-hitting defender, and A's deaths per 40 turns:

| start, gap | A | econ | tw_aggro | tw2 | tw2_aggro | gk_pos16 | gk_dmg08 |
|---|---|---|---|---|---|---|---|
| S10 equal | Shove + Pressure | +1.27 / 0.41 | +1.27 / 0.41 | +1.27 / 0.41 | +1.27 / 0.41 | +0.16 / 0.56 | −0.63 / 1.06 |
| S10 equal | Shove, Vision, Pressure | +3.13 / 0.12 | +3.13 | +3.13 | +3.13 | +2.83 / 0.19 | +2.22 / 0.38 |
| S10 equal | dive on a positive edge | +1.73 / 0.41 | +1.12 / 0.48 | +1.08 / 0.48 | +0.33 / 0.56 | +0.63 / 0.55 | +0.07 / 0.75 |
| S10 equal | Vision, then dive | −0.51 / 0.19 | −2.10 / 0.30 | −1.50 / 0.22 | −3.24 / 0.28 | −1.02 | −1.05 |
| S10 ×1.76 | Shove + Pressure | +2.74 / 0.12 | +2.74 | +2.74 | +2.74 | +2.11 / 0.22 | +2.11 / 0.16 |
| S10 ×1.76 | Shove, Vision, Pressure | +4.20 / 0.00 | +4.20 | +4.20 | +4.20 | +3.94 | +3.91 |
| S10 ×1.76 | dive | +2.59 / 0.09 | −0.10 / 0.20 | +0.35 / 0.20 | −3.62 / 0.59 | +1.57 / 0.14 | +0.74 / 0.38 |
| S10 ×1.76 | dive on a positive edge | +4.47 / 0.08 | **+2.82 / 0.16** | +2.47 / 0.22 | +0.52 / 0.34 | +3.67 / 0.16 | +2.65 / 0.34 |
| S10 ×1.76 | Vision, then dive | +4.14 / 0.03 | **+2.20 / 0.03** | +2.42 / 0.06 | −0.91 / 0.19 | +3.39 | +3.36 |
| N10 ×1.76 | dive on a positive edge | +6.90 / 0.08 | +5.51 / 0.11 | +5.39 / 0.12 | +3.38 / 0.33 | +5.74 | +6.02 |
| N10 equal | both last-hit: B deaths / HP lost | 0.06 / 0.74 | 0.06 / 0.74 | 0.06 / 0.74 | 0.06 / 0.74 | 0.09 / 0.91 | 0.16 / 0.83 |

Readings:

* **The tower variants act only on dives.** Every non-dive cell is identical to econ (a test pins this for the
  candidate). `tw_aggro` turns the strong champion's dive from more profitable than pressuring (+4.47 against +2.74) into a
  viable but riskier option level with it (+2.82, twice the deaths; independent seeds +1.86 against +2.55), below
  shoving with Vision (+4.20), and makes equal-power dives worse (−3.1 → −4.5).
  `tw2` has about the same size of effect but also doubles shots on non-siege dives. `tw2_aggro` makes the strong
  side's dive roughly break-even (+0.5, Vision dive −0.9), against the brief's requirement that justified aggression
  stays viable.
* **The gank variants act on everything.** They make pressure without vision unprofitable at equal power (+1.27 →
  +0.16 or −0.63). They raise deaths in the neutral lane (0.06 → 0.09–0.16 per 40 turns) and at the tower cell, and
  barely move the strong side. There was no diagnosed gank error to correct (§4.3), so they are not adopted.
* **No variant changes an equilibrium of the crash-scenario games** (54 games: `sensitivity.md`, last table). The
  defender's best reply to every pressuring attacker is still to shove back, and dives are never played against it.
  The tower change therefore corrects an *out-of-equilibrium* payoff, the one a learning agent meets when its
  opponent farms passively under its tower.

---

## 6. Strategic evaluation (Part 5)

Two families of restricted zero-sum games, solved on the matrix of mean J (discounted gold difference over 100
turns; G40 tables in the result files). (a) The **crash scenarios** (S4, S10, N4, N10 × three gaps; 12 attacker × 7
defender strategies) are read from the Part 2 grid (`results/m43d/crash_games.md`). (b) The **M4.3c scenarios** are
replayed on each configuration (F = freeze, N = neutral, P = freeze phase; the M4.3c sets plus `sh_vis` for both seats
in F and N, `sh_dive` and `sh_recall` in N; `results/m43d/strategic.md`). An equilibrium of such a game is an
equilibrium of the scripts, not of the laning game.

### 6.1 Scorecard

| configuration | B shoves at equal power (F10 / N10) | B concedes / defends vs ×1.76 (F10g) | A keeps a freeze ahead (F10g / P10g) | A uses Vision (N10g / N3) | value of a ×1.76 lead (F10g / N10g) | value of a ×1.76 lead after a crash (S10) | distinct equilibria |
|---|---|---|---|---|---|---|---|
| four-action baseline (`base4`) | 1.00 / 1.00 | 0.00 / 0.00 | 0.00 / 0.69 | 1.00 / 0.00 | +7.54 / +6.47 | — | 6 |
| **econ (0.8)** | 1.00 / 1.00 | 0.00 / 0.00 | 0.00 / **1.00** | **0.70** / 0.00 | +5.20 / +3.91 | +2.68 | 8 |
| econ, own tower 0.85 | 1.00 / 1.00 | 0.00 / 0.00 | 0.00 / 1.00 | 0.68 / 0.00 | +4.62 / +3.27 | +1.62 | 9 |
| econ, own tower 0.9 | 1.00 / 0.82 | 0.26 / 0.39 | 0.05 / 1.00 | 0.94 / 0.40 | +3.84 / +2.55 | +0.67 | 9 |
| **econ + tower aggro (candidate)** | 1.00 / 1.00 | 0.00 / 0.00 | 0.00 / 1.00 | 0.65 / 0.00 | +5.20 / +3.91 | +2.68 | 8 |

("B shoves" = B's weight on shoving strategies, "concedes" = respect or step back, "defends" = walk up and Thin or
contest; the S10 column is the value of the S10 ×1.76 crash game.)

### 6.2 What each economy rewards

Selected matrix entries at equal power and level 10 (value to the row strategy against the column strategy):

| | base4 | **econ (0.8)** | 0.85 | 0.9 | econ, no plate gold (DIAGNOSTIC) | 0.9, no plate gold (DIAGNOSTIC) |
|---|---|---|---|---|---|---|
| Shove against Last Hit (N10) | +9.24 | +3.10 | +1.84 | +0.58 | +0.39 | −2.12 |
| Shove against Thin (N10) | +6.65 | **+1.80** | +0.80 | **−0.20** | −0.71 | −2.70 |
| Shove + Vision against Last Hit (N10) | +11.06 | +5.11 | +3.74 | +2.37 | +2.38 | −0.35 |
| Slow Push then Shove against Shove (N10) | −6.28 | −1.19 | −0.24 | **+0.72** | +1.30 | +3.19 |
| a freezer against a walk-up (F10) | +1.18 | −0.51 | −0.91 | −1.31 | −0.53 | −1.33 |
| a freezer against a breaker (walk up + Shove) (F10) | −5.37 | **−0.07** | +0.96 | **+1.99** | +2.72 | +4.76 |
| the weak side's equilibrium vs a ×1.76 freezer (F10g) | break | break | break | Thin 0.39, break 0.35, respect 0.26 | **cautious 0.69, respect 0.21**, Vision-break 0.11 | Vision-break 0.81 |

Readings:

* **Equal power: mutual Shove, but no longer because of the tower cap.** At 0.8 a crash is farm-neutral (§3.3), and
  Shove's edge over Last Hit (+3.1) is the plate gold. Without plate gold (DIAGNOSTIC: a tower whose plates have
  fallen) Shove beats Last Hit by only +0.4. Thin then beats plain Shove (−0.7), Shove with Vision is the equilibrium,
  and the weak side's best answer to a ×1.76 freezer becomes conceding (cautious 0.69, respect 0.21). So in a
  continuing game Shove is dominant while the plates last and conditional afterwards.
* **Shove remains a rational counter to Thin and to freezes at 0.8, not at 0.9.** At 0.8 Shove beats Thin by +1.8
  and breaking a freeze neutralizes it (−0.07 to the freezer). At 0.9 Thin beats Shove (−0.2), Slow Push then Shove
  beats Shove (+0.7), and breaking a freeze *helps* the freezer (+2.0), because the breaker's crash is a gift. Without
  plates at 0.9 Shove loses to Last Hit, Thin and Slow Push in every neutral-lane game (−0.5 to −2.7).
* **Vision is a risk-management decision.** It enters the stronger side's equilibrium at 0.8 (F10g, F10gl 1.00; N10g
  0.70; validation 0.24 in N10g, where Shove and Vision-Shove are within noise). It is the equilibrium of both equal
  sides once plates are gone. Its value is the halved gank cost of §4.2.
* **Power converts into income through Shove, Vision and the freeze phase.** At 0.8 the ×1.76 lead is worth +5.2
  (F10g); during freeze phases the stronger freezer harasses (P10g/P10gl: Trade every turn while the freeze holds,
  1.00; validation 0.40, against Shove). The lead is worth less at 0.9 (+3.8, and +0.7 after a crash), where
  receiving the leader's crash costs nothing.
* **More action diversity at 0.9 is not better strategy.** The extra equilibrium weight at 0.9 (Thin, respect,
  freezing at equal power: F4 30% on both seed sets) appears because the crash is a gift. Thin and freezing win by
  inviting the opponent's shove, not by denying anything. The 26% respect of F10g does not reproduce on the
  independent seeds (§7.3).
* **The weak side does not concede while plates are on the tower** at any rate up to 0.85: it breaks by shoving back.
  M4.3c's "walk up and Thin" against a ×1.76 freezer (§8 there) does not survive the addition of the Vision
  strategies. Against `sh_vis` the weak side's best reply is to break (S10 crash games: shove back 1.00).

### 6.3 Against the intended properties

| property (brief, Part 5) | econ (0.8) | 0.9 | candidate |
|---|---|---|---|
| Shove useful but not automatically dominant because of tower-farming penalties | **yes**: the cap no longer drives it (crash farm-neutral); plate gold does, and without plates Shove is conditional | Shove loses to Thin and to Last Hit once plates are gone | as econ |
| defenders farm reasonably well under tower without harassment | yes: 0.91 of the lane rate = lane farming | better than lane farming (0.98–1.00): a gift | as econ |
| attackers must take meaningful risks to deny that farm | yes: denial needs dives or a Safe response; gank cost 3–4 per 100 turns without vision | yes (Safe posture costs 14–20%) | yes, and a strong diver pays the tower |
| a stronger champion can induce defensive responses | yes: Safe halves a ×1.76 diver's gain; no-plate games concede | weakly (respect 0.26, not robust) | yes |
| Shove remains a rational counter to Thin and unfavourable freezes | **yes** (+1.8; breaking neutralizes a freeze) | **no** (−0.2; breaking helps the freezer) | yes |
| Vision is a meaningful risk-management decision | **yes** (+1.5 to +1.9; in the equilibria) | yes | yes |
| strategy depends on power, position, wave state and HP | yes (8 distinct equilibria; crash-state replies differ by gap) | yes (9) | yes (8) |
| diving riskier than standing outside tower range | equal power yes; ×1.76 **no** (dive +4.5 / +3.2 against Pressure +2.7 / +2.6, main / independent seeds) | as econ | **yes** (dive +2.8 / +1.9 against +2.7 / +2.6; twice the deaths) |

---

## 7. Selected candidate

### 7.1 Decision

* **Keep the own-tower collection rate at 0.8.** It is not a large automatic penalty: it equals the realized
  efficiency of lane farming (0.91 against 0.90), so an uncontested crash costs the defender nothing. 0.9 makes
  receiving a crash a 9% gift. It also takes away Shove's role as the counter to Thin and to freezes, halves what a
  power lead is worth after a crash, and once plates are gone it makes shoving a loss against Last Hit and Thin in every neutral-lane game. 0.85
  lies between on every measure and adds nothing qualitative.
* **Do not change any gank parameter.** Ganks are the main and already substantial price of forward positioning
  (3.0–4.0 resources per 100 turns of perma-pressure at equal power, more than its gain). Vision halves that price.
  The gank variants penalize every forward position, the neutral lane included, and correct no diagnosed error.
* **Remove the tower's "busy" discount for a champion that attacks a champion under the tower** (`tw_aggro`). It is
  the one hazard that was under-priced: a strong champion's dive paid more than pressuring without diving, on both
  seed sets, at no extra risk of death. The change is League-faithful, touches only connected trades in or next to
  the enemy tower, and changes no equilibrium.

### 7.2 `GameConfig.m4_four_actions_econ_aggro()` (EXPERIMENTAL CANDIDATE, RECOMMENDED for the next PPO milestone)

`m4_four_actions_econ()` plus:

| field | `m4_four_actions()` | `m4_four_actions_econ()` | **candidate** | rule (Reference §6.3, §8.8) |
|---|---|---|---|---|
| `tower_shot_p_busy` | 0.5 | 0.5 | **1.0** | a dive is shot with probability 1.0 even while the diver's crashed minions are under the tower |
| `tower_adjacent_p_busy` | 0.25 | 0.25 | **0.5** | a connected trader next to the enemy tower: 0.5 likewise |
| `collect_contest_own_tower` | 0.6 | 0.8 | 0.8 | unchanged |
| `trade_farm_factor`, `kill_bounty`, `death_turns_by_level` | 0.5, 0.5, 3 | 0.8, 2.0, 3/4/5 | 0.8, 2.0, 3/4/5 | unchanged |

Everything else (wave mechanics and the four wave actions, ganks, vision, collection, XP, death, Recall, plates, the
gold-only reward, continuing time, the 61-code action space and the 88-entry observation) is
`m4_four_actions_econ()`. A game without a connected trade in or next to the enemy tower is the same game turn by turn
(test). `M4_FOUR_AGGRO_FIELDS` lists the two fields; both are existing champion-layer parameters.

**Comparison with the prior candidate.**

| measure | `m4_four_actions_econ()` | candidate |
|---|---|---|
| every M4.3c restricted game (14 scenarios) | — | **identical values and equilibria** |
| every crash-scenario restricted game (S4, S10, N10 × 3 gaps) | — | identical equilibria and values within 0.02 |
| ×1.76 dive on a positive edge against a tower farmer (S10 / N10) | +4.47 / +6.90 | +2.82 / +5.51 |
| the same, independent seeds (S10) | +3.21 | +1.86 |
| ×1.76 Vision-then-dive (S10; independent seeds) | +4.14; +2.98 | +2.20; +1.41 |
| equal-power dive (S10; independent seeds) | −3.09; −3.72 | −4.49; −4.61 |
| ×1.76 diver's deaths per 40 turns (S10, edge) | 0.08 | 0.16 |
| Shove + Pressure, Shove + Vision, every non-dive play | — | identical |

### 7.3 Validation

* **Independent seeds** (SeedSequence 79,700, 128 games per cell; `strategic.md`, "Validation"): econ, 0.9 and the
  candidate on F4, F10g, F10gl, N3, N10, N10g, P10g.
  * Robust: the econ and candidate equilibria of F4, F10g, F10gl, N10 and N3 (in kind); 0.9's weak-side Thin
    (0.35–0.41) and 0.9's freezing at equal power in F4 (0.31).
  * Not robust (near-ties): the Shove / Vision-Shove split in N10g (0.70 → 0.24); P10g's harassment (1.00 → 0.40,
    against Shove); 0.9's respect in F10g (0.26 → 0).
  * Values within 0.1–0.3 of the main run. The candidate equals econ on these seeds too.
* **Crash grid on independent seeds** (SeedSequence 79,500, 128 games, S10 and N10, three gaps, econ / 0.9 /
  candidate): `results/m43d/validation.md`. Robust: the farm-neutral crash at 0.8 (Δ B farm when
  crashed on +0.18 / +0.00, main / validation) and the gift at 0.9 (+1.53 / +1.43); the uncontested-crash and
  neutral efficiencies (0.907 / 0.909 against 0.899 / 0.914 at 0.8; 0.984 / 0.985 at 0.9); the values of pressure
  and of Vision-then-Pressure within 0.4; every crash-game value within 0.4 and every econ and candidate crash-game
  equilibrium. The 0.9 equilibria keep their defensive mixtures (the weak side's Thin in S10 ×1.76: 0.53 / 0.69).
  **Corrected by validation:** the ×1.76 dive is worth less on the independent seeds (+3.21 instead of +4.47 under
  econ, +1.86 instead of +2.82 under the candidate). It stays above plain Pressure under econ on both seed sets and
  falls to or below it under the candidate, but it is not the strong side's best aggression on the new seeds (Vision
  then Pressure, +4.07, is)
* **Tests.** `tests/test_m43d_tower_pressure.py` (15 tests):
  * the variants change only their fields;
  * the candidate changes only its two fields, plays every non-dive game like econ, and leaves every earlier preset's
    values unchanged;
  * the ledger: the unconditional cap is the only loss of an uncontested tower farmer at each rate, the Safe posture
    costs only above the Safe base, and the trade, absence and position parts are exact; the parts sum to the ideal
    in play;
  * the chance masks keep the stream aligned and remove only the masked events;
  * the post-crash strategies use the documented actions;
  * grid cells reproduce;
  * the audit tables are the engine's rules.

  Full suite: **1042 passed, 4 skipped, 2 xfailed** (1027 before this work; the same skips and expected failures).
* **Mechanical checks of the brief.**
  * Gank probabilities stay in [0, 0.30] (audit tables; nothing changed).
  * Vision works as documented: occurrence ×0.25 for 6 turns, set only by an effective objective; a connected trade
    suspends it, which is why `sh_vis_dive` places Vision first.
  * Tower shots follow the contact rules: only connected traders at the enemy tower or next to it.
  * Collection stays within [0, 0.9] (ledger).
  * Rewards are zero-sum (environment; unchanged).
  * Existing presets keep their behaviour (unchanged files, preset tests).
  * Seeds reproduce (test).

---

## 8. Unintended consequences

1. **Dives at equal power become strictly worse** (−3.1 → −4.5; Vision-then-dive −0.5 → −2.1). An equal champion
   should rarely dive a tower farmer, but a learning agent will now almost never try it.
2. **The tower-adjacent shot doubles during sieges** (0.25 → 0.5). It affects a champion that trades next to the
   enemy tower while its own minions are under it. In the scripted sets the harassment strategies take at most 0.11 bar more per 40 turns and their J moves by
   at most 0.25, but it may matter to PPO.
3. **The unconditional cap stays.** At 0.8 it is offset in value by tower farming's safety, but it is still a rule
   attached to a cell, not to an interaction. The interaction-based denial the brief prefers exists (dives, the Safe
   posture outside the tower cell, absence), but it adds to the cap rather than replacing it.
4. **Shove still dominates at equal power while plates remain** (both sides shove in F4, F10, N10). That is the
   plate gold (3 per tower), a legitimate tower reward. After the plates the game is the DIAGNOSTIC no-plate game,
   where Shove is conditional.
5. **The strong side's best aggression is now Vision-then-Pressure** (+4.2 S10 ×1.76). A PPO agent should learn to
   place vision when ahead; if it does not, it under-uses the most profitable line.

## 9. Remaining limitations

1. **Contact geometry.** A defender in its own tower cell can be reached only by a dive, so "harassment from just
   outside the tower while the defender last-hits" cannot exist: the defender never needs to step out to farm a
   crash. Contested tower farming in this game is diving or nothing. Changing it (e.g. contact with the tower cell
   from the adjacent cell at a hit penalty) is a positioning redesign, outside this study as the brief required;
   evidence here: §3.4, trace "B punishes a sieger".
2. **Ganks ignore the opponent.** Occurrence and damage depend on the victim's cell, behaviour, wave choice and
   vision, never on the enemy laner's presence, HP or position. A pusher is as gankable while its opponent is dead
   as while it is there, and a gank can never kill from full HP (4 units < 6). A two-versus-one model would need a
   new rule.
3. **Vision only during the own siege.** It cannot be placed defensively, so the defender has no vision decision
   while being pressured.
4. **The tower shot is one HP unit.** League's tower damage escalates within seconds. `tw2` (two units) was measured
   (§5), but it changes non-siege dives too and was not needed once the busy discount was removed.
5. **Plates are the only tempo reward of a crash.** There are no roams, objectives or jungle invades, so the value of
   "crash and leave" is a Recall window and nothing else. That is why it loses (§3.3).
6. **Restricted games.** Every equilibrium is over scripted sets. The near-ties of §7.3 move between seed sets. The
   no-plate games are a diagnostic of the late game, not a played state.
7. **The cap is a stand-in.** 0.8 reproduces "tower farming = lane farming" for the current lane geometry and gank
   rates. If either changes (reach, gank exposure near the own tower), the farm-neutral rate changes with it and
   §3.1–§3.3 must be rerun (`python -m lanegame.experiments.m43d tower_farm`).

## 10. Recommendation

1. **The economic game is coherent enough for the next PPO milestone.** Under the candidate:
   * an uncontested crash is farm-neutral;
   * denial requires dives or the defender's own retreat;
   * pressure pays its plates against a real gank price that Vision halves;
   * a strong champion's dive is viable but riskier than pressuring;
   * Shove counters Thin and freezes;
   * the strategy depends on power, wave state and the plates left.
2. **Next milestone: four-action PPO on `m4_four_actions_econ_aggro()`**, with `m4_four_actions()` as the control, 3
   seeds × 2M transitions (M4.3b settings, entropy 0.03), after generalizing the M4.3b behaviour and intervention
   tools to four wave slots (M4.3c §12). An optional third arm on `m4_four_actions_econ()` measures whether PPO
   discovers the cheap dive that the candidate removes.
3. **Read in PPO:**
   * the Vision rate when ahead;
   * dive frequency and the defender's Safe response by power gap;
   * crash-and-leave frequency (it should be rare);
   * Shove frequency before and after a tower's plates fall (the no-plate regime should look different);
   * the weak side's concession against a stronger freezer once plates are gone.
4. **Not recommended:** own-tower rate 0.9 or 0.85, any gank change, `tw2`, a new pressure-based collection penalty.
   If PPO shows defenders being harassed off tower farms unrealistically little, the evidence points at the contact
   geometry (§9.1), not at the collection rate.

---

## 11. Reproduction and files

```bash
python -m lanegame.experiments.m43d audit                          # Part 1, 40 s
python -m lanegame.experiments.m43d tower_farm pressure            # Parts 2-3, ~35 min on 11 workers
python -m lanegame.experiments.m43d sensitivity                    # Part 4, ~23 min
python -m lanegame.experiments.m43d strategic                      # Part 5 (base4, econ, ot85, ot90, econ_aggro), ~25 min
python -c "from lanegame.experiments.m43d import strategic as sg; sg.run(['econ_pg0','ot90_pg0'], scenarios=['N3','N10','F4','F10','N10g','F10g','P10'])"
python -m lanegame.experiments.m43d validation                     # Part 6, ~45 min
python -m lanegame.experiments.m43d traces report                  # tables and traces
python -m pytest tests/test_m43d_tower_pressure.py
```

The first Part 2–3 run defined `sh_dive` as "Trade every turn"; those cells were kept under the name `sh_bully`, and
`sh_dive`, `sh_dive_edge` and `sh_vis_dive` were then played with the definitions above (a re-run reproduces the
stored grids, since every cell is keyed by its strategy name and seeds are per cell).

| file | content |
|---|---|
| `lanegame/config.py` | preset `m4_four_actions_econ_aggro()`, `M4_FOUR_AGGRO_FIELDS` |
| `lanegame/experiments/m43d/` | `common` (variants, chance masks, I/O), `ledger`, `strategies`, `sim`, `grid`, `audit`, `strategic`, `traces`, `analysis`, `report_tower`, `report_strategic`, `stages`, `__main__` |
| `tests/test_m43d_tower_pressure.py` | 15 tests |
| `results/m43d/audit.{json,md}` | Part 1 |
| `results/m43d/grid_tower_farm.json.gz`, `tower_farm.md`, `crash_games.md` | Part 2, crash-scenario games |
| `results/m43d/grid_pressure.json.gz`, `pressure.md` | Part 3 |
| `results/m43d/grid_sensitivity.json.gz`, `sensitivity.md` | Part 4 |
| `results/m43d/strategic_*.json.gz`, `strategic.md` | Part 5 (and `_val`, `_pg0`) |
| `results/m43d/grid_validation.json.gz` | Part 6 crash grid on independent seeds |
| `results/m43d/traces.md` | representative lane traces |
| `docs/m4/laning_game_mathematical_reference.md` | §0.4 row, §5.3 and §6.3 measured notes, new §8.8, glossary, §13 |
| `docs/README.md` | index entry |

### Representative traces

From `results/m43d/traces.md` (global cells; `[A14]` A's crashed mass under B's tower; HP in bars). Shove and leave
at level 10, equal: B collects the crash in its tower cell at the cap while A, backing off to its −1, loses the
defenders its mass kills behind B's tower (B4, B9, B7 at +3):

```
turn  A action  B action    -2    -1    +0    +1         +2    +3   HP a/b    B farm (got/ideal, cap)
0     SH-S      LH-H        A8     .     a     .   [A14],b     B4   1.00/1.00  0.80/0.90 cap 0.10
1     SH-S      LH-H         .     a    A8     .    [A6],b     B9   1.00/1.00  0.80/0.90 cap 0.10
2     SH-S      LH-H        A8     a     .     .    [A7],b     B7   1.00/1.00  0.70/0.79 cap 0.09
3     SH-S      LH-H         .     a    A8     .         b    B14   1.00/1.00  0.70/0.79 cap 0.09
4     SH-H      LH-H        A8     .     .  A4,a      B7,b      .   1.00/1.00  0.45/0.45 cap 0.00
```

Vision, then dive, A ×1.76: one turn of Hold + Vision, then Trade + Pressure under vision (gank probability 0.04
instead of 0.17). B falls to 0.12 of its HP in two turns and has to recall; A takes no gank:

```
turn  A action  B action   HP a/b     A risk
0     SH-H+V    LH-H       1.00/1.00  p=0.15
1     SH-T+P    LH-H       1.00/0.71  p=0.04 trade dealt 0.29
2     SH-T+P    LH-H       1.00/0.12  p=0.04 trade dealt 0.59
3     SH-T+P    LH-S       1.00/0.12  p=0.04
5     SH-H      LH-R       1.00/0.12  p=0.03            (B recalls)
6     SH-H+P    ABSENT     1.00/1.00  plate
```
