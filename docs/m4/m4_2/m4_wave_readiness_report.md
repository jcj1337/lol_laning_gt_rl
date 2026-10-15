# M4.2d-A — Wave Bounce and Freeze-Breaking Readiness

Branch `readiness/m4-2d-wave` (from `experiment/m4-continuous-investment` 25c40a2). This is a short diagnostic. It
uses deterministic trajectories of the M4 resolver with no chance events (`u = None`): no ganks and no trade hits.
Gank exposure is reported as the summed gank probability. All policies are short scripted rules. Nothing was trained
or sampled.

- **Driver:** `python -m lanegame.experiments.m4.wave_readiness`
- **Results:** `results/m4_wave_readiness/wave_readiness.json`
- **Tests:** `tests/test_m4_farm_spot_rule.py`

**Start states:**
- *pushed*: s0, then A Shoves once on turns 5–7, so A holds a stacked wave at the lane at k = 9;
- *neutral*: s0, then 9 turns of mutual Slow Push, giving symmetric contact at the lane.

Positions are in own coordinates: own tower −2, the lane 0, enemy tower +2, base ±4. Windows are 48 turns (pushed)
or 40 turns (neutral).

## Summary

- **The wave does not bounce.** After a crash, the crasher's next waves park and stack in the defender's half. They
  beat the defender's released waves and crash again every ~11 turns.
- **A push is permanent.** Against a passive defender, the pushing side has no action that brings its own wave
  back. Only an opponent's Shove moves the lane, and that opponent then becomes the pusher and pays.
- **The cause is structural:** the lane has no restoring force (§4). This is a design limitation, not a bug.
- **The Shove → Crash → Recall part works.** A Recall during a large crash is free. A badly timed Recall costs
  about 1 resource and 8 XP.
- **The Bounce → Return part does not exist.** The returning champion finds its wave still parked in the enemy half,
  farming 0.17 per turn against the defender's 0.32.
- **The Freeze-breaking movement artifact is a narrow mechanical bug.** The fix is implemented as the opt-in
  `GameConfig.m4(farm_spot_rule="pre_crash_front")`. Defaults and historical tests are unchanged.
- **Recommendation: BLOCKED** on the wave cycle. The fix is a small mechanical rule change; learned agents cannot
  find a bounce the rules do not contain (§9).

## 1. How wave bouncing currently works

A crash puts the crasher's front into the siege mass under the enemy tower. Two rules then apply:

- The tower removes the mass at 2 HP per turn (`tower_siege_damage` 1 per tick), so 9 HP takes 5 turns.
- The defender's fronts at x ≥ −3 wait while the siege lasts (legacy rule D7), so its new waves merge behind the
  tower.

The design (§15 example 7) expected the defender's merged blob to "beat A's fresh 4-HP waves and push toward A's
side". In the implementation, the crasher's follow-up waves are not fresh single waves:

- **They park.** Under `stack_creep` Slow Push, a front at x ≥ 0 does not walk unless it is stacked. Then it creeps
  one cell per 4 turns.
- **They merge.** Rear packets walk freely and merge into the parked front (`block` cap 12, `queue_fights`).
- **They fight outside tower range.** When the siege clears, the defender's blob (8 HP, two waves) meets the
  crasher's blob (8 + 4 HP, three waves) at the defender's −1. The tower fires only on fronts inside its own cell.
- **The larger blob wins.** Lanchester damage, floor(HP / 4), favours it, so the crasher's wave wins, stacks again
  and re-crashes.

Without a crash the lane is just as stuck. Both sides send 4 HP every 2 turns, and a front in the enemy half stands
still while the enemy's fresh waves walk into it. So the contact point is **neutrally stable at every cell**: A +1
against B −2 trades 2 HP per turn each, indefinitely. This generalizes the parked configuration of audit §1.1.

**Sweep** (`stage_sweep`): 48 one-off Shove windows from s0 against a passive defender, 56+ turns each, under both
farm-spot rules.
- 3 windows crash, and all 3 stay in the re-crash loop.
- 45 settle at A 0 / B −1 with equal farm.
- **0 bounce**: after a crash, B's wave never reaches A's half.

**Counterfactual toggles** (scratch runs, 48 turns, the pushed lines) found no bounce in any case:
- D7 "wait only at −3": identical trajectories, because the defender's fronts are already behind −3 during every
  siege;
- `siege_engage`, `tower_siege_damage = 2`, `tower_damage = 2`, `slowpush_mode = "half"`.

## 2. Representative trajectories (farm_spot_rule = "m2")

Columns: A and B fronts as x/HP, the rear packets and siege mass behind them, champion cells (`away` while
recalled), A's crash HP, and farm per turn.

**Slow Push → Shove → keep farming vs passive** (`shove_farm`):

| k | A front | A rears / siege | B front | B rears / siege | champs A, B | A crash | farm A / B |
|---|---|---|---|---|---|---|---|
| 13 | +1/8 | −4 / 0 | −4/4 | – / 0 | +1, −2 | | 0.26 / 0.32 |
| 14 | +2/4 | −2 / 0 | −3/1 | – / 0 | **−1**, −2 | | **0.00** / 0.90 |
| 15 | 0/4 | −4 / 1 | −4/4 | – / 0 | 0, −2 | 2 | 0.00 / 0.68 |
| 16 | 0/4 | −2 / 0 | −2/4 | – / 0 | 0, −2 | | 0.00 / 0.23 |
| 24 | +2/5 | −2 / 0 | −3/1 | – / 0 | +2, −2 | | 0.26 / 0.77 |
| 25 | 0/4 | −4 / 2 | −4/4 | – / 0 | 0, −2 | 3 | 0.00 / 0.68 |
| 35 | +1/9 | 0, −4 / 0 | −4/4 | – / 0 | +1, −2 | | 0.18 / 0.16 |
| 36 | −2/4 | – / 9 | – | – / 0 | **−1**, −2 | 9 | **0.00** / 0.90 |
| 37 | 0/4 | −4 / 7 | −4/4 | – / 0 | 0, −2 | | 0.00 / 0.45 |
| 38 | 0/4 | −2 / 5 | −3/4 | – / 0 | 0, −2 | | 0.00 / 0.45 |
| 41 | +1/10 | −4 / 0 | −2/5 | −4 / 0 | +1, −2 | | 0.26 / 0.51 |
| 42 | +1/7 | −2 / 0 | −2/5 | – / 0 | +1, −2 | | 0.35 / 0.47 |

- k = 14 and 36 show the farm-spot artifact (§5): the champion walks back on the crash turn and collects nothing.
- k = 41: the siege has cleared. B's 8-HP blob meets A's 10-HP blob at B −2 / A +1 and loses. **No bounce**; A
  crashes again at k = 48.

**Recall timed to the 9-HP crash** (`hold_recall_large_crash`):

| k | A front | A siege | B front | champs A, B | farm A / B |
|---|---|---|---|---|---|
| 28 | −2/4 | 9 (crash) | – | −1, −2 | 0.00 / 0.90 |
| 29 | 0/4 | 7 | −4/4 | away, −2 | 0.00 / 0.45 |
| 30 | 0/4 | 5 | −3/4 | away, −2 | 0.00 / 0.45 |
| 31 | 0/8 | 3 | −3/8 | 0, −2 | 0.00 / 0.45 |
| 33 | +1/10 | 0 | −2/5 | +1, −2 | 0.26 / 0.51 |

A misses nothing while away: no farm is available to it during the siege. It returns to the same pushed lane.

**Lane-control defender** (B: Freeze when legal, else Shove while its front ≥ A's; `shove_farm_vs_control`):
- B's Shove flips the lane. By k = 14 the contact is A −2 / B 0; by k = 16, B's front stands at A's −1.
- B then crashes 2-HP waves every 8 turns (k = 21, 29, …) and loses the farm race, A +4.45 per 48 turns.
- The defender's counter-push is real counterplay, but it is a push, not a bounce.

## 3. Does Shove → Crash → Recall → Bounce → Return work?

Totals over the window: earned A − B (farm + plates), farm A / B, XP A / B. Missed gold/XP is what A could have
farmed during its turns away.

| start | A's line (B passive unless noted) | A − B | farm A / B | XP A / B | A crashes (turn: HP / turns to clear) | missed gold / XP | bounce |
|---|---|---|---|---|---|---|---|
| neutral | Hold / farm continuously | **0.00** | 12.60 / 12.60 | 160 / 160 | – | – | – |
| neutral | Recall at k = 10 (badly timed) | −1.21 | 11.80 / 13.01 | 152 / 160 | – | 1.00 / 8 | – |
| neutral | Recall at k = 10, B Shoves while A is away | −1.44 | 11.80 / 13.23 | 152 / 162 | – | 1.00 / 8 | – |
| pushed | Hold (Slow Push) | −6.08 | 9.21 / 16.29 | 176 / 190 | 17: 3/1, 28: 9/5, 40: 7/4, 53: 3/1 | – | no |
| pushed | Slow Push → Shove → keep farming | −5.33 | 9.40 / 15.73 | 174 / 188 | 15: 2/1, 25: 3/1, 36: 9/5, 48: 7/4 | – | no |
| pushed | Slow Push → Shove → Recall at the 2-HP crash | −6.09 | 8.77 / 15.86 | 166 / 188 | as above | 1.00 / 8 | no |
| pushed | Hold → Recall at the 9-HP crash | −6.08 | 9.21 / 16.29 | 176 / 190 | as Hold | **0.00 / 0** | no |
| pushed | Slow Push → Shove vs the lane-control defender | +4.45 | 15.03 / 10.58 | 190 / 204 | – (B crashes 5×) | – | – |

Against the four questions of the assignment:

- **Does shoving create a favourable Recall window?** Yes, for large crashes. A 9-HP crash leaves A nothing to farm
  for 4–5 turns, so a Recall then is free. A 2–3 HP crash clears in one turn, and a Recall then costs the same as
  from a neutral lane.
- **Can a champion Recall without losing too much farm and XP?** Yes. The worst case measured is about 1.0 resource
  and 8 XP for two turns away. Slow Push fights do not depend on the champion, so an absence does not move the lane.
- **Can the bounced wave be collected after returning?** **No.** There is no bounced wave.
- **Are poorly timed Shove/Recall decisions punished?** Only weakly:
  - a mistimed Recall costs about 1.2;
  - a Shove while the opponent is away adds only −0.23;
  - a Shove commits the pusher to a permanent push, costing about 0.11–0.13 per turn. This is the dominant effect,
    and it is far larger than any timing effect.

**Verdict:** the first half of the cycle (Shove → Crash → Recall) works. The second half (Bounce → Return) does not
exist. In these states, Shove is a one-way commitment: whoever pushes first loses farm until the opponent pushes back.
This explains the M4.2b results:
- the presser at −17 per 400 turns;
- `ShoveRecallAgent` losing to passive play;
- "Shove-before-Recall never helped".

## 4. Structural problem and cause

**Classification:** a **design limitation**. The rules are internally consistent (conservation and positioning are
checked) but do not produce the intended interaction.

**The mechanic responsible:** the lane has no restoring force. The rules that combine to cause it:
1. **`stack_creep` Slow Push parks a front at x ≥ 0** unless it is stacked (`mechanics._sp_moves`). The enemy's waves
   walk freely in their own half.
2. **Equal spawn throughput** on both sides, with fights decided where the fronts meet.
3. **The tower acts only inside its cell and on the crashed mass.** The fight just outside the tower (defender −2
   against attacker +1) is unsupported.
4. **Lanchester damage and merging** favour the side whose waves arrive merged: the parked side.

None of these is a parameter in scope. Increasing farm rewards would hide the effect without creating a bounce.

**Two small proposals for review.** Both were tested as scratch patches of the Slow Push movement condition; the
driver's `stage_candidates` applies them temporarily. The engine is unchanged.

| candidate (one condition in `_tick`'s Slow Push branch) | Hold | Shove → farm | Shove → Recall | vs lane-control | neutral |
|---|---|---|---|---|---|
| current | −6.08 | −5.33 | −6.09 | +4.45 | 0.00 |
| **A: C3 + P2**: an *unstacked* Slow Push front past the lane (x ≥ 1, ≤ 1 wave) keeps walking; *and* while a side's crashed mass is under the enemy tower, its Slow Push fronts do not park | −1.61 | −1.53 | −2.10 | +4.45 | 0.00 |
| **B: P2 only**: no Slow Push parking while the own siege lasts | −1.61 | −6.61 | −7.60 | +4.45 | 0.00 |

- **A** lets a push run its course: stacks still creep and crash (one crash per line), then the leftover single waves
  walk into the tower and the lane returns to the lane cell.
  - *Tradeoff:* it removes the parked configuration of audit §1.1 for fronts of ≤ 4 HP. Freeze then holds a walking
    front rather than a parked one, which changes design §6.4: `DESIGN_HOLD_TABLE`, the S3 gate and the F0/F+ cases
    must be recomputed. That is arguably the fix audit §1.8 asked for (Freeze rarely mattered for 3–4 HP fronts).
- **B** is narrower: it only resets the lane after a crash, and leaves every non-crash parked state, including the
  §6.4 table, as it is.
  - *Tradeoff:* a push that never crashes stays parked forever (the Shove line is unchanged at −6.6).
- **Rejected options:**
  - tower support for a defender fighting inside its own tower cell: no crash ever happens;
  - "no parking past the lane" for every front: removes all stacking and crashes;
  - D7, `siege_engage`, and stronger tower fire or tower damage: no bounce.
- **Neither proposal produces a full League-style bounce**, in which the defender's wave crosses into the crasher's
  half and makes Freeze legal for the crasher. Both return the wave toward the lane. A true bounce would need the
  defender to gain a surplus after the crash, which is a further design question.

## 5. Freeze-breaking correction (farm-spot artifact)

**The artifact.** In the probe pass (design §5 step 1), a front that crashes is replaced by the next packet near the
crasher's base, and that packet becomes the farm spot. The champion therefore walks back, typically from +1 to −1.
That costs it:
- the Shove support on the push;
- XP within range 3;
- all farm on the crash turn: 0 of 0.75–1.0 available on every 9-HP crash in §2.

**Classification:** a narrow **mechanical bug**. The resolver contradicts its own positioning intent: the champion
plays at its wave, and the design §6.4 Freeze-break windows assume full reach. The KeepReachEnv prototype
(`experiments/m4/calibration.py`) was reviewed and is small and equivalent. It is now the opt-in rule
`farm_spot_rule = "pre_crash_front"`: a champion whose pre-turn front exists and crashes in the probe keeps its
pre-turn front cell, clipped to [−T, T], as its farm spot.

| check | result |
|---|---|
| never beyond the enemy tower | spot and end are clipped to [−T, T]; asserted on 600 random-legal turns. The furthest cell is +2 (the enemy tower cell), which the default rule reaches too |
| reproduces intended Freeze breaking | the parked-front table equals `DESIGN_HOLD_TABLE` in all 64 cells and equals the prototype. The presser breaks a 3–5 HP freeze in **3–6 turns** (6, 5, 6, 5 / 4, 3, 4, 3 / 4, 3, 4, 3), against 6–9 under the default. The F0 stacker finding is unchanged (Freeze ≥ Slow Push) |
| forward movement and gank exposure | on crash turns the champion stays at +1/+2 instead of −1/0. Expected ganks over 48 turns rise by 0.13–0.18 (+3–5%) in pushed lines. Neutral lines are identical |
| mirror symmetry and determinism | the resolver is mirror-equivariant under the rule (600 random-legal turns, asymmetric skills); trajectories replay identically |
| ordinary crashes | it fires on every crash, as intended. The pusher gains +0.6 to +0.8 per 48 turns and keeps crash-turn XP (+16 to +18). The first Shove crash comes one turn earlier and is larger (k = 14, 4 HP, instead of k = 15, 2 HP). A Recall right after a crash starts one cell further forward: a small-crash Recall misses 0.5 / 4 XP instead of 1.0 / 8, and a large-crash Recall costs −0.06 more. The lane-control line gives A +3.57 instead of +4.45 (B's crashes keep their reach). **It does not create a bounce** (sweep identical) |

No significant unintended consequence was found, so the rule is implemented. **The default stays "m2"**, so all
historical tests and pins hold, including the strict expected failure of the SHx gate. If the rule is adopted as the
M4 default, that expected failure must be turned into a passing test (the new test already asserts the values).

## 6. Code changes

| file | change |
|---|---|
| `lanegame/config.py` | new field `farm_spot_rule: str = "m2"` (`"m2"` or `"pre_crash_front"`), validated, added to `M4_ECONOMY_FIELDS` (legacy and m2c presets reject it) |
| `lanegame/champions_m4.py` | resolver step 1: three lines applying the rule; docstring |
| `lanegame/experiments/m4/wave_readiness.py` | new deterministic driver: scenarios, sweep, experiment-only candidates |
| `tests/test_m4_farm_spot_rule.py` | new focused tests (below) |
| `tests/test_m4_config.py` | adds the new field's non-default value to the legacy-rejection table |

No change to the engine (`mechanics.py`), defaults, investment, ganks, PPO or the reward.

## 7. Focused test results

- **New tests:** `pytest tests/test_m4_farm_spot_rule.py` — 5 passed:
  - the field is validated;
  - the 64-cell table equals the design and the prototype;
  - mirror equivariance, bounds and firing only on probe crashes;
  - deterministic lines and no bounce (pins the finding);
  - the SHx gate under the rule.
- **Related existing tests:** `test_m4_config`, `test_m4_calibration`, `test_m4_resolve`, `test_legacy_pins`,
  `test_m4_freeze_gate`, `test_m4_scenarios`, `test_m4_env`, `test_m4_time` — all passed (2 strict expected failures
  preserved, 1 skip). Wall time 10.4 s and 22.4 s.
- The full suite and `--runslow` were **not** run (per the assignment).

## 8. Simulation budget

| what | resolver turns | wall |
|---|---|---|
| exploratory scratch runs (traces, strategy comparisons, toggles, both-rule sweep, candidate patches) | ≈ 12,700 | < 10 s total |
| driver (`wave_readiness`: 2 rules × 8 lines, the sweep, 3 candidates × 5 lines) | 4,674 | 0.6 s |
| SHx / F0 gate check under both rules | 8,640 | 1.7 s |
| **experimental total** | **≈ 26,000** (target 10,000; cap 30,000) | ≈ 12 s |
| focused tests (verification, not counted above) | ≈ 7,000 per run of the new file | 10–22 s per batch |

The total is above the 10,000-turn target. Most of the excess came from running the sweep under both rules (6,300)
and the gate check (8,640). No large batteries and no training were run.

## 9. Remaining blockers before PPO, and recommendation

1. **No restoring force in the lane (this report):** **requires a mechanical change** — a design decision on rule A
   or B (§4), or an explicit decision to accept permanent pushes.
   - Learned agents cannot find the Bounce → Return cycle, because no policy of either player produces it. The
     pusher cannot pull its wave back; the defender can only become the pusher.
   - Training now would mostly re-learn "never push first", and Freeze would stay reachable only after the
     opponent's mistake.
2. **Adopting `pre_crash_front` as the M4 default:** small. It is implemented and verified; adopting it means
   reconciling the SHx strict expected failure.
3. **Carried over, not addressed here:**
   - items weak and Recall-to-buy not a real decision (M4.2b/c);
   - the full `--runslow` suite is still pending since M4.2b.

### Recommendation: **BLOCKED**

The specific structural problem is that the lane is neutrally stable: a crash is followed by a re-stack, not a
bounce. This is not a tuning issue, and learned agents cannot work around it, because the bounce is absent from the
rules. The fix is small and mechanical: one condition in the Slow Push movement rule, either proposal A (recommended
for review) or B, as an opt-in M4 field, followed by recomputing `DESIGN_HOLD_TABLE` and the S3 gate. If reviewers
instead accept permanent pushes as intended, the mechanics are usable as **PASS WITH CAVEATS**: the push commitment,
the free large-crash Recall window and the weak Recall punishment are all documented above.
