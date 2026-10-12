# M4.2d-B — Independent Champion Positioning

Branch `prototype/m4-positioning` (from `readiness/m4-2d-wave` f504f5e). This is a short prototype study. It uses
deterministic trajectories of the M4 resolver with no chance events (`u = None`), so HP stays full and nothing dies.
The threat of a trade is reported as its **expected** damage: hit probability × damage of the connected trade, plus
any expected tower shot. Gank exposure is the summed gank probability. Every policy is a fixed scripted rule; nothing
was trained or sampled.

- **Opt-in field:** `GameConfig.m4(position_rule="standoff")`, with `"pin"` as an ablation. The default `"m2"` is
  bit-for-bit unchanged.
- **Driver:** `python -m lanegame.experiments.m4.positioning`
- **Results:** `results/m4_positioning/positioning.json`
- **Tests:** `tests/test_m4_positioning_intent.py`

Cells are in own coordinates: own tower −2, the lane 0, enemy tower +2. "Strong" A holds two item tiers (attack
power ×1.44); B holds none. Start state: s0 after 9 turns of mutual Slow Push, which leaves both fronts and both
champions at the lane.

## Summary

- **Under the current rule, position carries no threat.**
  - At the default geometry a champion's speed (2) equals half the lane. A Trade can reach contact from any start
    cell; only a dive needs a start at or beyond 0.
  - A Yield or Safe defender reacts to the committed Trade in the same turn and always escapes to its tower.
  - So a Trade against Yield never lands. It works as a free zoning move for **any** A, strong or weak: 24 trades
    chosen, 0 connected, farm 0.45 vs 0.14 per turn.
  - Being in contact at the end of a turn has no consequence for the next turn.
- **The prototype adds two rules (`position_rule="standoff"`). There are no new actions and no new observations.**
  - **Pin:** a Yield champion that *starts* the turn in contact with an enemy Trade champion cannot evade. It is
    traded exactly like a Hold.
  - **Stand-off:** Yield farms from the rearmost cell still within farm reach of the enemy's leading minions.
- **Equal champions can now farm without contact.**
  - Mutual Yield gives (−1, −1): out of contact, the same farm as mutual Hold (0.315 per turn each), and 41% less
    gank exposure.
  - Stepping up to the wave (Hold) gains no farm by itself. It puts the opponent in contact, so the opponent is pinned
    if the champion trades next turn.
- **The threat becomes real.**
  - Strong A holds the lane and trades only every 4th turn. Against a Yield B: 6 of 6 trades connect, and B's
    expected damage is 790 centi-HP in 24 turns. Under `m2` the same line connects 0 times.
  - B's real options are to contest (Hold, or Trade back), to stand off and accept being pinned, or to retreat
    (Safe: always evades, but farms 0.14 per turn and leaves A's farm unzoned at 0.45).
  - A reactive B (Safe while it starts in contact, else Yield) follows A's intention with a one-turn lag. When A
    stands off, both stand off with equal farm. When A holds the wave, B plays Safe and A out-farms it by 0.27 per
    turn without trading.
- **Main caveat: whether the threat is worth respecting depends on HP value, not on positioning.**
  - At the diagnostics valuation (0.25 resources per 100 centi-HP), the weaker champion's best one-turn reply to a
    Trade is to trade back, not to retreat. Retreating pays only above ≈ 1.2 resources per 100 centi-HP, which is
    the regime near death.
  - The prototype makes positional threats credible. Making them decisive is a combat or economy calibration
    question for the other workstreams.
- **Recommendation: adopt `"standoff"` as the positioning interface for the M4.2d integration, opt-in, and do not
  make it the default yet.** Re-evaluate it after Agent A's wave system and Agent C's zoning land (§8).

## 1. Why the current positioning cannot carry a threat

There are three structural facts at the default geometry (`tower = 2`, `champion_speed = 2`, `trade_range = 1`,
`farm_reach = 1`):

1. **Reach is not limited by position.**
   - Contact needs |x + y| ≤ 1 (own cells x, y), except that the own tower cell is reachable only from inside it.
   - From any start cell, a trader reaches a contact cell of any target outside its tower within 2 cells.
   - Only the dive into the enemy tower cell (+2) requires a start at ≥ 0.
2. **Evasion is a perfect same-turn reaction.**
   - Step 3 of the resolver moves a threatened Yield or Safe champion from its start cell to the furthest-forward cell
     out of the trader's contact.
   - From any start ≤ 0 the tower cell (−2) is reachable. It is out of contact unless the trader dives, and the
     trader's target is computed before the evasion.
3. **Every farming cell at the lane is in contact.**
   - With kills at 0, the farming cells are −1, 0 and +1.
   - Against a champion at 0, only the tower cell is out of contact, and from there the kills are 2 cells away.

Consequence under `m2` (one-turn matrix at the lane, strong A; full matrix in the JSON, stage `matrix`):

| A \ B | Hold | Yield | Safe | Trade |
|---|---|---|---|---|
| Hold | 0.315 / 0.315 | 0.315 / 0.315 | 0.45 / 0.14 | 0.315 / 0.158, A takes 87 |
| Trade | 0.158 / 0.315, B takes 125 | **0.45 / 0.14, B evades, 0 damage** | 0.45 / 0.00, evades | 0.158 / 0.158, 87 / 125 |

Entries are farm A / farm B for the turn, in resources, and expected centi-HP taken.

- **Trade against Yield never lands.** It costs A nothing except the Trade gank term (+0.02).
- **It does not depend on strength.** An equal A gets the same +0.31.
- "Positional pressure" under `m2` is therefore Trade every turn, which is constant nominal fighting. It is never a
  position held between fights.

## 2. The prototype

Here e is the enemy's leading minion cell at the start of the turn, in own coordinates (minus the enemy front's x).
**`position_rule`** (an M4 field: `validate()` rejects it outside ruleset `m4` and checks its value):

| value | Yield's target cell | Evasion of a threatened Yield | Hit probability against it |
|---|---|---|---|
| `"m2"` (default) | farm spot | always (M2) | Yield term (−0.25) |
| `"pin"` | farm spot | **not if it started in contact** with the trader (`in_contact(c0_yield, c0_trader)`) | **Hold's** (0.0) when pinned |
| `"standoff"` | **`max(−T, min(spot, e − farm_reach))`**; the farm spot if the enemy has no front | as `"pin"` | as `"pin"` |

**Exact rule (resolver step numbers):**

- **Step 2, targets.** Hold uses the farm spot (unchanged). Safe uses `min(spot, safe_line)` (unchanged). Trade uses
  the contact-cell chase toward the opponent's intended end cell (unchanged). Yield uses `yield_target(spot, lane,
  s, cfg)`. Every move is bounded by `champion_speed` (`_move`), and cells are clipped to [−T, T].
- **Step 3, evasion.**
  - A Yield or Safe champion whose end cell is in contact with an enemy Trade champion's end cell evades as before.
  - Exception (`"pin"` and `"standoff"`): a **Yield** whose start cell was in contact with the trader's start cell is
    *pinned*. It stays at its target, keeps mode `farm` (it farms and zones like Hold), and the trade connects
    normally in step 4.
- **Step 9, trade hit probability.** A pinned Yield is hit with Hold's behaviour term: it was caught farming, not
  backing off.
- **Safe is never pinned.** Safe is the committed retreat posture: it always evades and pays for that with Safe
  efficiency, no zoning and no wave contribution.
- **Reporting.** `M4Turn.pinned` and the env info key `"pinned"` record the event.

**Why these two pieces and nothing more:**

- **Pin is the only change that makes a turn-end position matter on the next turn.**
  - Speed cannot do this: it covers half the lane.
  - It is physical: a defender already inside the attacker's reach when the turn begins cannot step out before the
    attacker acts.
  - The attacker still has to choose Trade (the simultaneous choice).
  - The defender can always break contact beforehand, at a farm cost, by playing Safe, or (under `"standoff"`) by
    standing off from an opponent who also stands off.
- **Pin alone (`"pin"`) is not enough.**
  - At the lane every turn starts in contact, so a pinned Yield is exactly a Hold. The `pin` rows of the matrix
    equal the Hold rows.
  - Yield then differs from Hold only after a return from base.
  - There is no configuration in which both champions farm the lane out of contact.
- **The stand-off cell adds that configuration.**
  - At the lane it is −1. It is still within farm reach of the kills at 0, still zones, and is out of contact with
    another stand-off champion (|−1 − 1| = 2).
  - It is defined relative to the **enemy front, not the own spot**.
  - A first draft used spot − 1. In the pushed lane of the wave-readiness study (kills at +2 while the own front
    parks at +1), that draft cost the stand-off champion all its farm.
  - With e − farm_reach, the stand-off cell equals the Hold cell there (identical farm, `farm_cost` stage). So
    standing off never moves a champion out of reach of the contested minions.

**Default preservation.**

- `"m2"` takes exactly the old code path: `yield_target` returns the spot and `pinned` is always False.
- A 600-turn uniform-random digest (M4 with asymmetric skills, plus M4 with `farm_spot_rule="pre_crash_front"`)
  covers the resolver output, the env observations, masks, rewards and every info key. It is identical to commit
  f504f5e.
- All existing focused M4 tests pass unchanged. The only edit to an existing test is the new field's entry in the
  legacy-rejection table of `test_m4_config.py`.

## 3. Action space, masks and observation

- **Action space:** unchanged, 46 codes. No behaviour, wave or objective is added. Under the opt-in, the Yield codes
  (`champ_act(w, B_YIELD)`, codes 3–5) mean "stand off and give way to a trade if I can".
- **Masks:** unchanged. Legality never depends on the rule (tested).
- **Observation:** unchanged, 88 entries.
  - Whether a Yield would be pinned depends only on the two start cells. Both are already observed (cell one-hots of
    both champions).
  - The stand-off cell depends on the farm spot and the enemy front. Both are already part of the lane block.
  - The observation stays Markov and the decoder is untouched.
- **Policies:** policies trained or tuned under `"m2"` must not be reused under `"standoff"`, because the Yield
  semantics differ. Every scripted M4 agent is unaffected: none of them uses Yield.

## 4. Experiments (budget: final driver run 1,575 turns)

### E1 — Equal strength, the same wave (stage `equal`, 24 turns)

| rule | A/B | cells | contact at turn end | farm/turn A, B | expected ganks A, B |
|---|---|---|---|---|---|
| m2, pin | any of Hold/Yield | (0, 0) | 100% | 0.315, 0.315 | 1.68, 1.68 |
| standoff | Hold/Hold | (0, 0) | 100% | 0.315, 0.315 | 1.68, 1.68 |
| standoff | **Yield/Yield** | **(−1, −1)** | **0%** | 0.315, 0.315 | **0.99, 0.99** |
| standoff | Hold/Yield | (0, −1) | 100% | 0.315, 0.315 | 1.68, 0.99 |

- The stand-off costs no farm and no XP at the lane (96 XP each in every line), and it lowers exposure.
- Pushed lane (stage `farm_cost`): every Hold/Yield combination is identical under `m2` and `standoff`. Farm/turn is
  0.20 vs 0.29, which is the parked-lane imbalance of the wave-readiness report, not a positioning effect.

### E2 / E3 — A stronger champion takes the wave and holds it without attacking (stages `matrix`, `deterrence`)

One-turn matrix under `"standoff"` from the configuration "A at the wave, B standing off" (0, −1):

| A \ B | Hold | Yield (stand-off) | Safe | Trade |
|---|---|---|---|---|
| Hold | 0.315 / 0.315 | 0.315 / 0.315 | 0.45 / 0.14 | 0.315 / 0.158, A takes 87 |
| Trade | 0.158 / 0.315, B takes 125 | **0.158 / 0.315, B pinned, takes 125** | 0.45 / 0.00, B evades | 0.158 / 0.158, 87 / 125 |

From the mutual stand-off (−1, −1), the same Trade against Yield is **evadable**: B goes to −2 and farms 0.14
against A's 0.45. A Trade from the stand-off is an attack the defender sees coming.

24-turn lines, strong A:

| line | rule | trades chosen / connected | B evaded / pinned | B expected damage taken | farm/turn A, B |
|---|---|---|---|---|---|
| A Hold, Trade every 4th turn, vs B Yield | m2 | 6 / 0 | 6 / 0 | 0 | 0.349, 0.271 |
| same | **standoff** | 6 / **6** | 0 / **6** | **790** | 0.276, 0.315 |
| same vs B Safe | either | 6 / 0 | 6 / 0 | 0 | 0.45, 0.105 |
| A Trade every turn vs B Yield | m2 | 24 / 0 | 24 / 0 | 0 | 0.45, 0.14 |
| same | standoff | 24 / 24 | 0 / 24 | 3,117 | 0.158, 0.315 |
| A Hold (never trades) vs B Yield | either | 0 / 0 | 0 / 0 | 0 | 0.315, 0.315 |
| A Hold (never trades) vs B "respect" | either | 0 / 0 | 0 / 0 | 0 | **0.45, 0.14** |

"Respect" is an observable rule: Safe while the turn starts in contact with the enemy, else Yield. Findings:

- **Under `standoff` the forward position is a credible threat.** A champion standing at the wave can make a Trade
  land on any turn: the opponent standing off is pinned. Under `m2` the identical line never lands.
- **A can control the lane without trading,** provided B respects the threat. The last row gives A +0.31 per turn
  with zero trades.
  - Under `m2` the same row shows B's respect is a mistake, since Yield is free there.
  - Under `standoff`, respecting is a real choice between 0.175 farm per turn and the trade risk.

### E4 — The weaker champion approaching or retreating

- **From the mutual stand-off:**
  - B moving up to the wave (Hold) puts both champions in contact. Both are then exposed (`matrix`, standoff config,
    row Hold / column Hold).
  - B standing off keeps evasion available.
- **Against A at the wave:**
  - Retreating is Safe at −1. It is the same cell as the stand-off, but in a posture that always evades: on a trade
    B goes to its tower.
  - B cannot re-establish a non-contact farming cell while A stays at the wave. That is the control A buys by
    standing forward.
  - B's way back to safe farming is A standing off again (E5), or contesting.
- **On this five-cell lane a retreat is mostly a posture change, not a cell change.** Safe and the stand-off share −1.
  The cell changes (to −2) only when an actual trade arrives.

### E5 — Opponent movement when the stronger champion changes intention (stage `intent_switch`)

A plays Yield (stand-off) for 8 turns, then Hold (at the wave) for 8, then Yield for 8. B plays "respect":

| block | A cell | B behaviour | B cell | contact | farm/turn A, B |
|---|---|---|---|---|---|
| A stand-off | −1 | Yield (after 1 turn) | −1 | 0% | 0.33, 0.29 |
| A at the wave | 0 | **Safe** | −1 | 100% | **0.43, 0.16** |
| A stand-off | −1 | Yield (after 1 turn) | −1 | 0% | 0.33, 0.29 |

B's behaviour string is `SYYYYYYYY SSSSSSSS YYYYYYY`: B tracks A's intention with a one-turn lag, because it reacts to
observed start cells. Under `m2`, "respect" plays Safe all 24 turns (every turn starts in contact), which is simply
dominated there.

A single Trade from the stand-off, followed by Hold, moves A to the wave. That Trade does not connect (B evades), so
it costs no HP and pushes B into Safe for the rest of the window: 0.45 vs 0.14 per turn.

## 5. Interface for the integrator and Agent C (zoning)

**What determines a champion's cell each turn, given the start cell c0, the farm spot, the start-of-turn lane and
the rule:**

| behaviour | target cell |
|---|---|
| objective (Pressure / Vision) | +T |
| Trade | `contact_cell_target(c0, ref)`; ref = the opponent's intended end cell, or its start cell if it also trades |
| Hold | farm spot (the probe's own leading packet clipped to [−T, T]; `farm_spot_rule` may keep the pre-crash cell) |
| Yield | `yield_target(spot, lane, s, cfg)`: the farm spot, or the stand-off cell under `"standoff"` |
| Safe | `min(spot, safe_line)` |
| Recall | c0 |
| absent | −T |

- Each champion moves at most `champion_speed` toward its target.
- A threatened Yield or Safe then evades to `_evasion_cell` (from c0), **unless** it is a pinned Yield.

**How Trade connects:** in step 4, as before, when the end cells are in contact. Pinning only removes the evasion
step. The trader still has to choose Trade and reach contact within its speed.

**How the opponent's options change:**

- Yield is a real stand-off position that keeps evasion only while the turn starts out of the enemy's contact.
- Safe is the only unconditional escape.
- Hold and Trade are unchanged.

**What zoning sees:** modes and end cells exactly as before.

- A pinned Yield is in mode `farm` at its stand-off cell, so it zones like Hold.
- A stand-off Yield is in mode `farm` one cell behind the contested minions.
- A zoning rule that wants "threat" can read `in_contact(c0_self, c0_enemy)`, the condition under which a trade
  would pin. That keeps zoning a function of observed positions.
- Agent C can develop against the baseline. The stand-off only changes *which* cells occur: −1 against −1 at a
  neutral lane.

## 6. Dependence on the wave system (Agent A)

The rule assumes two things about the wave:

- **the farm spot**: the probe's own leading packet, unchanged;
- **the enemy front at the start of the turn**: `lane.side(o)` x and HP, the cell of the enemy minions nearest the
  own tower.

How the rule would carry over:

- **Natural minion movement:** if the front becomes a different object (continuous, several packets), e should be
  defined as "the enemy's leading minions at the start of the turn". The rule needs nothing else.
- **Champion wave damage:**
  - The stand-off champion stands exactly farm_reach behind the contested minions. Its contribution reach
    (`end + farm_reach`) still covers the fight cell today.
  - If Agent A's champion damage range is shorter than farm_reach, a stand-off champion would no longer damage the
    wave. That is a natural trade-off (standing back = less wave control), but it changes the stand-off's price.
  - This should be re-measured after merging: rerun stages `equal` and `farm_cost`, about 700 turns.
- **No wave physics change here:** at the lane the stand-off contribution line (0) covers the front, so Shove damage
  and freeze-break pushes are unaffected. Farm and XP of the E1 trajectories are identical under all three rules.

## 7. Code touched (merge notes)

- **`lanegame/champions_m4.py`**
  - module docstring: two step notes;
  - import line: `B_HOLD`;
  - `__all__`;
  - new function `yield_target`, placed between `contact_cell_target` and `m2_chase_target`;
  - `M4Turn`: new trailing field `pinned` with a default;
  - `resolve_turn_m4`:
    - step 2: one `elif beh[s] == B_YIELD` branch;
    - step 3: the evasion loop, which now has the pin check and a `yield_target` own-target;
    - step 9: the defender behaviour passed to `trade_hit_probability`;
    - the final `M4Turn(...)` call: one trailing argument.
- **`lanegame/config.py`**
  - field `position_rule` after `farm_spot_rule`;
  - its check in `validate()` after the `farm_spot_rule` check;
  - appended to `M4_ECONOMY_FIELDS`.
- **`lanegame/env/core.py`:** one info key, `"pinned"`.
- **`tests/test_m4_config.py`:** one table entry.
- **New files:**
  - `lanegame/experiments/m4/positioning.py`
  - `tests/test_m4_positioning_intent.py`
  - `results/m4_positioning/positioning.json`
  - this report

Wave physics (`mechanics.py`), collection and zoning (`collection_units`), and combat power are untouched.

## 8. Recommendation and caveats

**Recommendation: adopt `position_rule="standoff"` as the M4.2d positioning interface (opt-in). Keep `"m2"` as the
default until the integrated wave and zoning rules are measured together.** It is the smallest rule found that:

- lets a champion hold a forward position that is a standing threat (the pin);
- lets the opponent answer by contesting, standing off or retreating, each with a farm, exposure or HP price;
- gives equal champions a contact-free farming equilibrium, so that advancing is a decision.

It adds no action, mask or observation entry.

Caveats:

1. **HP value decides whether the threat is respected.**
   - At 0.25 resources per 100 centi-HP, a strong A's Trade against Hold nets +0.16, while forcing B into Safe nets
     +0.31. In the one-turn matrix B's best reply to a Trade is to trade back (+0.10 for A), not to retreat.
   - Retreat wins only above ≈ 1.2 resources per 100 centi-HP, which is near death or with a larger bounty.
   - Positional control "without fighting" therefore needs HP to matter more: death risk, item breakpoints or trade
     damage. That belongs to the combat and economy workstreams.
   - One-turn matrices ignore continuation values (HP carried forward). A learned or solved policy is the real test.
2. **A pinned champion has no in-place escape except Safe.**
   - Against an opponent at the wave, Safe at −1 is the same cell as the stand-off.
   - So on this five-cell lane, "retreat" is a posture, and the tower cell is reached only by evading an actual trade.
   - A larger lane or a slower champion would turn it into real movement. Both are geometry changes, out of scope.
3. **The rule reinterprets Yield.** This is clean for new training, but it breaks comparability with `"m2"` runs that
   used Yield. No current scripted M4 agent uses Yield.
4. **`"pin"` alone is not recommended.** It merges Yield into Hold at the lane.
5. **Untested here:** learned play, the integration with Agent A's wave and Agent C's zoning, and stochastic games
   with deaths. These trajectories are deterministic and HP never changes.

## 9. Budget

| item | turns |
|---|---|
| final driver run (`positioning.json`, budget field) | 1,575 |
| development runs of the driver (three earlier runs, before the windows were shortened to 24 turns and the stand-off cell was redefined) | 7,562 |
| initial exploration trajectory | 24 |
| **experimental total** | **≈ 9,160** |
| verification: default-preservation digest, 2 code versions × 600 turns | 1,200 |

The total is above the 5,000 target and below the 15,000 cap. Most of it is development reruns.

Tests:

```
python -m pytest -q tests/test_m4_config.py tests/test_m4_resolve.py tests/test_m4_combat.py tests/test_m4_env.py \
    tests/test_m4_observation.py tests/test_m4_farm_spot_rule.py tests/test_m4_integration.py \
    tests/test_m4_positioning_intent.py          # 93 passed, 2 skipped
```
