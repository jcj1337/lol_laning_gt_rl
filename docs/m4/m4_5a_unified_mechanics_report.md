# M4.5a — Unified positional combat, Vision and laning mechanics: implementation report

*2026-10-14. Branch `main` on d9f92df; nothing staged or committed. Implementation and correctness only: **no PPO
training, no calibration campaign, no change to any earlier preset, reward or result.** Design specification:
[m4_5a_unified_mechanics_spec.md](m4_5a_unified_mechanics_spec.md); the implemented rules in equation form:
[mathematical reference §14](laning_game_mathematical_reference.md#14-experimental-the-m45a-unified-positional-candidate-m45_unified);
generated smoke tables: [`results/m45a/report.md`](../../results/m45a/report.md); viewer recordings:
`results/m45a/trajectories/`; code: `lanegame/positional.py` (new), `lanegame/experiments/m45a/` (new); tests:
`tests/test_m45a_mechanics.py` (new).*

---

## 0. Summary

**What was built.** A new EXPERIMENTAL candidate preset, `GameConfig.m45_unified()` = `m4_four_actions_econ_aggro()`
plus six opt-in rules implemented together:

1. a shared off-lane **Vision node W** on a small movement graph, reached by a contestable **Ward** action (it replaces
   the siege Vision objective);
2. **XP range 2** by graph distance;
3. **minion retaliation** after a champion's Trade actually damages an enemy champion;
4. **positional minion aggro** on a champion standing in the enemy wave ahead of its own;
5. **exclusive per-tick targeting** — a wave fires at minions or at a champion on each tick, never both — with
   **minion damage on champions** that matters at level 2 and fades by level 18;
6. **capped wave maintenance during a connected Trade** (Thin + Trade works; Shove + Trade is halved).

One supporting rule completes them: the champion's wave damage now reaches only one cell around it (a **reach ball**),
which the Shove + Safe and W requirements needed.

**Per-tick targeting.** Implemented as requested, without restructuring the combat engine. The single trade exchange
lands on tick 0; retaliation follows from tick 1; positional aggro is re-checked at the start of every tick. One
deviation from the baseline, found by the demonstrations, is that positional aggro applies only to a champion that did
not move this turn (§3.2).

**Correctness.**

- 73 new tests and an updated regression suite: [§4](#4-tests).
- **Byte-for-byte identity of 12 earlier champion presets** with the committed engine: full `info` records, tick traces
  and observations on random games.
- Mirror equivariance, exact replay, the observation decoder, the masks, the PettingZoo API and the checkpoint guard
  are all tested on the new preset.

**Readiness: GO WITH CAVEATS for M4.5b** ([§9](#9-m45b-readiness)). The mechanics work and are deterministic, but
several provisional numbers will shape the incentives decisively, and M4.5b must measure them together.

---

## 1. Implementation summary

### 1.1 Added

| item | where |
|---|---|
| preset `m45_unified()`, `M45_UNIFIED_FIELDS`, 11 opt-in fields, their contract validation, `CHAMP_WARD` (= 30), the `WARD` label, properties `ward_node`, `minion_targeting`, `m45_mechanics` | `lanegame/config.py` |
| the movement graph: `WARD` (= 99), `dist`, `move`, `in_contact`, `mirror_cell`, `exposure_cell`, `reach_interval`, `reaches`, `MinionTarget`, `minion_damage_centi` | `lanegame/positional.py` (new) |
| per-tick target designation (`_designation`) and firing; reach intervals; `TurnResult.champion_hits`, `champion_fire`; the tick trace's `minion_fire` | `lanegame/mechanics.py` |
| Ward legality (`ward_legal`), graph-aware contact and movement, no siege Vision under the ward rule | `lanegame/champions.py` |
| Ward (mode `"ward"`), graph distances for gold, XP, chase and exposure; the trade rolls before the lane step; the capped trade budget; minion targets; minion damage; ward placement; eight new `M4Turn` fields | `lanegame/champions_m4.py` |
| lane-step cache key with targets; the M4.5a `info` fields (only when an M4.5a rule is on) | `lanegame/env/core.py` |
| the W entry of each champion's cell one-hot (90 entries) and its decoding | `lanegame/env/observation_m4.py` |
| the W node, vision timers, Ward outcomes and minion fire in the lane picture; M4.5a events, details and help text; six M4.5 matchups; the M4.5a starts and Ward strategies | `lanegame/viewer/` |
| start states (`bigwave`, `bigwave_l18`, `ahead`), Ward strategies, demonstrations, the smoke study | `lanegame/experiments/m45a/` (new) |
| `m45_unified` among the named presets of the checkpoint guard | `lanegame/experiments/rl/m44/common.py` |
| design spec, this report, reference §14, roadmap | `docs/` |

### 1.2 Changed for every preset, with identical results

- **The trade rolls (step 9a) now run before the lane step** in `resolve_turn_m4`. They read only the start-of-turn
  state and the end cells, never the lane step's result, so the change is exact.
- **`champions.in_contact` and `champions._move` dispatch to the graph** only when a position is W, which cannot occur
  outside the ward rule.

Both claims are verified: random games on 12 champion presets (`m4_four_actions_econ_aggro`, `m4_four_actions_econ`,
`m4_four_actions`, `m4_lasthit`, `m4_lane`, `m4_natural`, `m4_natural_prototype`, `m4_emergent`, `m4_continuous`, `m4`,
`champions`, `champions_continuing`) produce byte-identical `info` records, tick traces and observations under the
committed engine (exported from `HEAD`) and the modified one. Two of them are pinned in the test suite.

### 1.3 Deliberately unchanged

- Every earlier preset and its defaults, rewards, results and checkpoints.
- The reward: gold earned, zero-sum. No term rewards or penalizes Ward, Trade, Safe or any position.
- The 61-code action space; the chance vector (ten uniforms per turn, drawn whatever happens).
- The probe that sets the farm spots.
- Every gank, tower, plate, Recall, death, investment and progression rule.
- The PPO trainer, its settings and the M4.4 instruments.
- The four-action wave budgets.
- The Safe posture's economics: no penalty was added (§3.7).

---

## 2. Mathematical specification (as implemented)

The full statement, with code references, is the mathematical reference §14. In brief:

**State.** $\mathrm{pos}_s\in\{-2,-1,0,+1,+2,W\}$; W is one node attached to the lane cell 0, the same node for both
sides. Distances: $d(\ell,\ell')=|\ell-\ell'|$, $d(\ell,W)=1+|\ell|$, $d(W,W)=0$. The vision timers $v_A,v_B$ stay per
side.

**Movement and contact.** Shortest paths of at most 2 edges per turn, so W is reached from $-1$, 0 or $+1$ in one turn
and left to the same cells. Contact holds when $d\le1$, keeping the own-tower rule. Chase targets are the nearest node
in contact, with W last on ties. W carries the gank exposure of the own cell 0.

**Ward** (code 30, legal at $-1$, 0, $+1$, W):

- The warder moves to W and is in mode `ward`: no wave damage, no gold, XP as present, trade-exposed like Hold.
- It is placed at the end of the turn unless the enemy's connected Trade dealt positive (mitigated) damage to it, or
  it died.
- It then sets $v_s=6$; for the next 6 turns the gank occurrence is multiplied by 0.25.

**Resolution order:** actions → probe → champion movement → trade contact and rolls (exchange on tick 0) → modes,
reach, budgets, minion targets → lane step tick by tick (designation at the tick's start; movement, merging, entry
crash; designated waves in range fire at their champion; fights with those waves dealing no minion damage) → gold, XP →
ganks → summed damage / $M(S)$ → death → ward placement → plates, progression, ledger, absence, reward.

**Designation on tick $t$.**

- **Retaliation:** the enemy champion's Trade damaged ours and $t\ge1$.
- **Positional aggro:** the enemy champion did not move this turn, stands on a lane cell holding our front, and its
  own front is strictly behind that cell or absent, with no crashed mass of its own there.

**Firing.** The designated wave fires if its front is within 1 of that champion and the tick firepower $f_t>0$. It then
adds $f_t$ units to that champion and deals no minion damage on that tick.

**Damage.** $D=\max(1,\lfloor50m/1.08^{L-1}\rceil)$ centi-HP for $m>0$ units, then divided by $M(S)$ with the rest of the
turn's damage. **Trade budget:** $\min(B_w,\lfloor C/2\rceil)$ with farm-mode reach. **Reach:**
$\{x:d(x,\mathrm{end})\le1\}$ (Safe also $x\le\min(\mathrm{end},-1)$). **XP:** $d(x,\mathrm{end})\le2$.
**Observation:** 90 entries.

---

## 3. Design decisions and deviations from the baseline

### 3.1 Per-tick targeting (the update to the brief)

Implemented, at modest cost: one designation step at the start of each tick and one firing step after the tick's
movement, inside `mechanics._tick_natural`. Every quantity the designation needs is already there:

- the start-of-tick packets and crashed masses;
- the champions' end cells, which are fixed during the ticks;
- the trade outcome, known before the lane step.

Exclusivity is per tick. Spent damage is never reassigned: the units a wave puts on a champion are exactly the
firepower it would have put on minions (`test_targeting_redirects_firepower_without_creating_or_duplicating_damage`).

- **Trade timing.** With one exchange per turn (`trade_exchanges` 1, required), the exchange lands on tick 0 and the
  minions react from the next tick (`minion_retaliation_delay` 1, configurable; 0 reproduces "from the exchange tick").
  A retaliation therefore lasts one tick of a two-tick turn.
- **Zero-firepower ticks.** Under the half-wave law a 1–7-HP blob has no firepower on tick 1. Such a blob is not
  counted as firing and keeps its normal role, such as engaging a crashed mass. The first implementation counted it as
  firing; a test caught this.
- **Turn-locked alternative.** Not adopted. It needs a notion of intended versus possible targets, and it either wastes
  firepower once the champion leaves range or double-counts it across ticks.

### 3.2 Positional aggro requires a stationary champion

In the first demonstrations a Hold champion was hit by the very enemy wave its own wave was beating. The champion
moves before the ticks to its farm spot, which is where its winning wave will be at the end of the turn, so at tick 0
it stood one cell ahead of its own wave in the enemy front's cell.

The rule now applies positional aggro only to a champion that did not move this turn. This is also the brief's
original wording ("starts a turn occupying the enemy wave's position"). A stationary champion is re-checked every
tick, so a wave walking into its cell during tick 0 targets it on tick 1. A champion that moves (follows its wave,
chases) is in transit; if it stays, the next turn counts.

### 3.3 Vision: one Ward code instead of the siege objective

The Vision objective slot is re-used: code 30 (Last Hit slot + Hold + Vision) is Ward, and the other Vision codes are
never legal. This keeps the 61-code space. A wave or behaviour choice would be meaningless while placing, so the action
is a single code, like Recall. Siege Vision is removed under the ward rule, so there is one source of Vision.

### 3.4 Ward details

| choice | why |
|---|---|
| W attached to 0, Ward legal within 2 edges, i.e. from $-1$, 0, $+1$ | follows from the existing speed 2; from the own tower (−2) the champion must first walk forward, which costs a turn of lane position |
| placement completes on the arrival turn | movement and action share the turn everywhere else (objectives, trades) |
| the warder collects no gold, deals no wave damage, gains XP | the opportunity cost the brief asks for, without inventing a partial rate; XP follows the range-2 rule |
| only the enemy champion's positive trade damage, or death, interrupts | the brief: "successfully damages"; ganks and minions are not the enemy's attack |
| W's gank exposure = the lane midpoint, and Hold's behaviour term | W is as exposed as standing at the lane |
| re-warding from W is legal | refreshes the timer; no special case |

**Consequence to note.** Every lane cell is within 2 of 0, so any Trader can reach the cell next to W: every Ward can
be contested. Whether a contest is worth it is a simultaneous guess, like the Recall interruption.

### 3.5 Minion damage scaling

The level resistance $\rho^{L-1}$ is new. The investment term re-uses the existing durability division by $M(S)$
(`invest_durability`, on in this economy), so investment is not modelled twice.

$\alpha$ was raised from the 30 first considered to **50**, because per-tick retaliation lasts one tick. A landing trade
into a 24-HP stack at level 2 then draws 93 cHP of minion fire against a 210-cHP double hit, and 46 cHP against a
105-cHP single hit (demonstration C1). At level 18 with 128 invested on both sides, the same two units cost 27 cHP
before durability and 13 after it, against a 917-cHP double hit (458 after the victim's durability; C2).

Results are never exactly 0: a floor of 1 centi-HP applies before the durability division.

### 3.6 The reach ball (not in the baseline list)

The historical reach is a line with no lower bound, which breaks two of the brief's requirements:

- a champion at $+1$ damaged an enemy front at its $-1$, and the crashed mass under its own tower from anywhere;
- Shove + Safe needs actual reach.

The ball $d\le1$ uses the same graph distance as gold collection, which was already a ball. Safe and a retreating
Yield keep their extra bound $\le\min(\mathrm{end},-1)$. Tests show both the forward case and the Safe case.

### 3.7 Safe and Yield

No resource penalty was added. The integrated rules now make retreating costlier through actual mechanics:

- the reach ball (a Safe champion at $-1$ cannot clear a wave at 0);
- XP range 2 (from $-1$ the kills at $+2$ are lost);
- the existing Safe collection rate.

Safe still evades a Trade with room to retreat (`test_safe_evades_with_room_and_pays_in_farm_not_xp`). Whether this is
enough is an M4.5b question.

### 3.8 Smaller decisions

- Crashed masses never target champions.
- A champion at W is never in a wave's cell, but can be hit by retaliation from a front at 0.
- The probe ignores minion targeting.
- From W with no front, the farm spot is 0.
- The trade-capped budget is recorded in `info["wave_budget"]`.

---

## 4. Tests

| suite | result |
|---|---|
| `tests/test_m45a_mechanics.py` (new) | **73 passed** (73 test items, parametrizations included) |
| `tests/test_m4_config.py` | the contract test's probe values extended with the 11 M4.5a fields (each must be rejected by the legacy rulesets) |
| `tests/test_viewer.py` | the authoritative-trace test re-runs the lane step with the recorded `minion_targets` and applies a placed ward's timer; the policy test skips the Ward wrapper like the other composite strategies |
| full regression suite | **1195 passed, 4 skipped, 2 expected failures** in 7 min 25 s (M4.4c: 1099 passed; +96 = 73 new M4.5a tests, 11 contract cases for the new fields, 12 viewer cases for the six M4.5 matchups); the same three PettingZoo API warnings as before; log `results/m45a/regression.log` |

**What the new tests cover.**

**Movement and Vision**

- Both players reach W symmetrically, and legal sets mirror.
- Entering and leaving W needs movement, with no teleport.
- A landing Trade interrupts a ward; a missed one does not.
- A dying warder places nothing.
- Vision timers are independent, both sides may ward, and vision multiplies the gank occurrence by 0.25 on the next
  turns.
- Gold and XP from W obey graph reach.

**Combat and minions**

- A landing Trade draws retaliation from the next tick (and from tick 0 with delay 0); a miss draws none.
- Mutual landing Trades retaliate on both sides.
- Retaliation needs the trader within one cell of the wave.
- A blob without tick firepower does not fire.
- Positional aggro hits a champion standing ahead of its wave. It ends when the champion falls back or its own front
  covers it, and spares a champion following its winning wave.
- A champion one cell away is targeted only after the wave arrives.
- An own crashed mass covers a pressuring champion.
- Firepower is redirected without being created or duplicated, and minion HP is conserved.
- Damage scales down with level and investment, never reaching 0.
- Minion damage can kill, and the kill pays the bounty.

**Wave management**

- The capped budget of a connected trader (Shove halved, Thin unchanged, the historical rule deals nothing).
- The reach ball against the line.
- Shove + Safe has no reach to a wave at 0.
- Safe evades and pays in farm.
- Thin + Trade at the freezer's tower still thins and draws retaliation.

**Integration**

- Mirror equivariance of the resolver on 400 random states with random chance.
- Environment soaks: invariants, zero-sum rewards, masks equal to the legal sets, and the observation decoding W and
  every lane state.
- Seeded replay is deterministic, and the lane-step cache does not change results.
- The official PettingZoo `parallel_api_test` passes (90 / 61 spaces).
- The checkpoint guard refuses an 88-input network on the new preset (also with `cross_env`) and still loads it on
  its own preset.
- Two historical presets are pinned by digests computed on the committed engine.
- Demonstrations replay exactly.

**Smoke soak** (`results/m45a/report.md`): 40 random games × 300 turns plus 7 scripted pairings × 16 seeds × 200 turns
on both presets, with the invariants and the observation decoder checked every turn. No failure; no rear-packet or
siege bound was exceeded.

---

## 5. Demonstrations

`streamlit run lanegame/viewer/app.py`, then "Recorded trajectory", folder `results/m45a/trajectories`. The same six
games are also in "Strategy matchup" mode as M4.5-1 to M4.5-6, with any seed.

In the lane picture:

- W is the green box above the lane cell 0, and a champion there is drawn inside it.
- "V5" is a champion's vision timer.
- "ward placed" / "ward interrupted" marks its Ward turn.
- The red label under a champion is the minion fire it takes in the frame (units and cause per tick; the end frame adds
  the turn's centi-HP).

| file | scenario of the brief | seed (rule) | what to watch |
|---|---|---|---|
| `m45a_E1_ward_neutral` | E, preemptive Vision | 0 | A wards from the neutral lane (turn 0, then every 7 turns), walks back the next turn; 3 / 3 wards placed; its Ward turns show no farm and no wave damage |
| `m45a_E2_ward_interrupted` | E, contested | 0 (first seed whose first Ward is interrupted) | B's landing hit interrupts A's ward; 2 of 4 wards interrupted; A's minions retaliate on B (6 ticks) |
| `m45a_E3_ward_trade_missed` | E, contested | 2 (first seed whose first Ward meets a connected but missed Trade) | the ward is placed although B connected |
| `m45a_C1_bigwave_level2` | C, early | 2 (first seed whose first connected Trade lands and draws minion fire) | A's 210-cHP double hit into B's 24-HP stack draws 2 units = 93 cHP on tick 1 (turn 0); a 105-cHP hit draws 46 cHP (turn 3) |
| `m45a_C2_bigwave_level18` | C, late | 2 (same rule) | the same turn at level 18, 128 invested each: a 917-cHP hit (458 taken) draws 2 units = 27 cHP, 13 taken after durability |
| `m45a_D1_ahead_of_wave` | D | 0 | A stands in B's wave with its own wave behind and chases a retreating B; B's minions fire at A until the wave walks past |
| `m45a_A1_strong_freeze` | A and B | 0 (first seed with a retaliation on A within 10 turns) | A (×1.76) keeps the freeze with Thin + Trade against a walk-up Shove; 6 retaliation ticks on A, B dies 3 times, both Recall; 40 turns |

Each file replays exactly through the viewer's own session code (`m44.trace.verify`: 7 / 7).

---

## 6. Parameters

| field | default (off) | `m45_unified()` | status |
|---|---|---|---|
| `vision_rule` | `"siege"` | `"ward"` | rule |
| `ward_exposure` | 0 | 0 | **calibrate** (gank exposure of W) |
| `vision_turns`, `vision_reduction` | 6, 0.75 | 6, 0.75 (unchanged) | **calibrate** (provisional baselines) |
| `xp_range` | 3 | 2 | **calibrate** |
| `minion_retaliation`, `minion_positional_aggro` | False | True | rules |
| `minion_range` | 1 | 1 | calibrate if needed |
| `minion_retaliation_delay` | 1 | 1 | **calibrate** (0 / 1) |
| `minion_damage_alpha` | 50 | 50 | **calibrate** (centi-HP per unit at level 1) |
| `minion_damage_growth` | 1.08 | 1.08 | **calibrate** (per-level resistance) |
| `trade_wave_rule` | `"none"` | `"capped"` | rule |
| `trade_wave_fraction` | 0.5 | 0.5 | **calibrate** |
| `wave_reach_rule` | `"line"` | `"ball"` | rule |

Parameters of an inactive rule must stay at their defaults (validation), so a configuration cannot carry an unused
value silently.

**Measured, descriptive** (`results/m45a/report.md`):

- Minion damage per minion-hit turn in random play: 9% of the bar at level 1, 3% at level 10, 1.2% at level 18.
- The damage table: 2 units = 16.7% of the bar at level 1, 0.9% at level 18 with 128 invested.
- Throughput: 3,259 turns per second, 0.88× the aggro preset.

---

## 7. Compatibility

- **Earlier presets:** byte-identical turn records on 12 champion presets (§1.2); pinned digests; legacy rulesets
  reject every M4.5a field. **No result file was overwritten**; `results/m45a/` is new.
- **Checkpoints:** every M4.4 / M4.3 checkpoint keeps loading on its own preset. On `m45_unified()` the guard refuses
  it, because 88 inputs ≠ 90; it does so even with `cross_env=True`, since a shape mismatch is always refused. Its
  `info` records and traces gain M4.5a fields only when an M4.5a rule is on.
- **Viewer:** earlier recordings replay unchanged; `m45_unified` is the first configuration and M4.5-1 the first
  matchup.
- **API:** `step_turn(..., champion_targets=None)`, `LaneEnv._lane_turn(..., targets=None)` and the resolver's
  `lane_step(..., targets=None)` are backward compatible. A custom `lane_step` is called with targets only under an
  M4.5a preset.

---

## 8. Roadmap changes

[`docs/rl_research_roadmap.md`](../rl_research_roadmap.md) now has:

- **M4.5a** (this, completed, in review);
- **M4.5b**, integrated calibration and strategic evaluation (planned);
- **M4.5c**, fresh shared-policy PPO training and evaluation on the calibrated model (planned).

M4.4d's robustness and exploitability work is **postponed and folded into M4.5c's evaluation**: it is not run on a
simulator being revised; the M4.4c checkpoints stay available for historical comparisons. M5 and M6 now depend on
M4.5c.

---

## 9. M4.5b readiness

**GO WITH CAVEATS.** The integrated simulator works:

- every mechanic fires at plausible rates in random and scripted play;
- every invariant holds;
- replay is exact and earlier presets are untouched;
- the viewer shows every new event.

The caveats are about incentives, which are M4.5b's subject.

### 9.1 Unresolved concerns (to measure, not yet to fix)

1. **Every Ward can be contested.** Every lane cell is within 2 of the cell next to W. In the contested demonstration
   half of the wards were interrupted (`ward_always4` vs `bully4`: 9.6 of 19.4 placed). Is W too close, or should a
   connected-but-missed trade still cost the warder something?
2. **Vision's value now costs a lane turn.** `ward_favorable4` vs `farm4`:
   - A placed about 20 wards in 200 turns;
   - its gank deaths fell from 0.31 to 0.06;
   - its gold lead was −2.65 ± 2.23 (not significant).

   Whether 6 turns at ×0.25 is worth a turn of farm is open, as is whether wards are placed when ganks matter (exposed
   positions).
3. **Retaliation lasts one tick.** The exchange lands on tick 0 and the minions react from tick 1, so a retaliation is
   half a turn of firepower; $\alpha$ = 50 compensates provisionally. Delay 0, $\alpha$ and trade into waves of
   8 / 16 / 24 HP by level should be screened together.
4. **Bullies lose more against farmers.** The trading bully lost more gold than in aggro (`bully4` vs `farm4`: −16.4
   vs −13.4) and took about 2,400 cHP of minion fire per 200 turns. Retaliation now protects a farmer standing with
   its wave (farmer deaths 1.81 vs 3.19). Is trading now too expensive early?
5. **Strong-side freeze (Scenarios A/B).** The ×1.76 freezer with Thin + Trade still loses gold to a walk-up Shove,
   though less than in aggro (−3.78 vs −8.77). Its Recalls, about 11 per 200 turns, suggest that holding a freeze
   while trading is HP-expensive. This is the intended Scenario B tension, unmeasured as yet.
6. **Positional aggro is rare and mostly self-inflicted.** It was 0 in passive farming. It comes mainly from chasers
   standing in the enemy wave (≈ 30 ticks per 200 turns for `bully4`). Check that it never punishes ordinary
   last-hitting positions in learned play.
7. **Wave dynamics changed.** A wave firing at a champion spares the enemy's minions (aggro preserves waves) and lets a
   crashed mass hit the tower when the defender's front turns on a champion. Freezes may break differently, and stacks
   may grow; no rear-packet bound was exceeded in the soaks.
8. **Safe economics.** XP range 2, the reach ball and the Safe collection rate are the only costs of stepping back. The
   M4.4c measure (≈ 0.08 resources per turn) should be repeated.
9. **The probe ignores targeting.** The farm spot can differ from where the wave actually ends when minions fire at
   champions. This is a known, documented approximation.

### 9.2 Recommended M4.5b experiments (prioritized)

1. **Trading into waves.** Trade EV (dealt minus minion fire minus the defender's reply) by level (2, 6, 11, 18), wave
   size (8 / 16 / 24 HP) and power gap, swept over $\alpha$ ∈ {30, 50, 80}, $\rho$ ∈ {1.05, 1.08, 1.12} and delay ∈
   {0, 1}. Target: early trades into large waves unattractive, late ones nearly unaffected.
2. **Vision economics.** Paired multi-turn counterfactuals of Ward versus farm from natural and initialized states:
   neutral, forward, defending, before a siege, under a threat. Sweep `vision_turns` {4, 6, 8}, `vision_reduction`
   {0.5, 0.75} and `ward_exposure`. Measure the value of a ward, how often it is contested, and the interruption rate
   by contester behaviour.
3. **Freezes with combat (Scenarios A and B).** The freeze grid of M4.4c (power, level, HP, plates) with Thin + Trade
   freezers against walk-up, Shove and step-back. Sweep `trade_wave_fraction` {0.25, 0.5, 1.0}. Measure freeze
   duration, the lead, HP paid, and the Recall-or-hold decision after the opponent resets.
4. **Safe, Yield and threat-only zoning.** The behaviour matrix of M4.4c under the new rules: the price of stepping
   back, and whether threats now buy denial without kills.
5. **Restricted-game equilibria.** The four-action strategy matrices of M4.3c / M4.3d, plus Ward strategies, on the
   new preset: does any single strategy (Shove) dominate again, and where do the equilibria move?
6. **Positional aggro audit.** Frequency by situation in scripted and random play; no false positives for followers
   of their own wave.
7. **Integrated sensitivity.** Vary the four or five most influential parameters jointly around the chosen point.
   Report each conclusion's robustness.

**Measurements to keep:**

- paired common random numbers, with seeds disjoint from 85,000–85,999;
- discounted gold lead (γ = 0.99), XP lead, deaths, Recalls, wards placed and interrupted;
- minion units and centi-HP by cause and level;
- capped trades, freeze turns, plates;
- every number with intervals over states and, where policies are learned, over seeds.

### 9.3 Reproduction

```bash
python -m lanegame.experiments.m45a all            # demonstrations (+ replay checks) and the smoke study, ~25 s
python -m pytest tests/test_m45a_mechanics.py      # the M4.5a tests, ~5 s
python -m pytest                                   # the full regression suite, ~7.5 min
streamlit run lanegame/viewer/app.py               # the viewer (Recorded trajectory: results/m45a/trajectories)
```
