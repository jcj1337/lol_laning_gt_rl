# M4.2d — Parallel Emergent Mechanics: Workstreams and Integration

Branch `integration/m4-emergent-ready`, created from `readiness/m4-2d-wave` (f504f5e). Date: 2026-10-12.

This assignment ran four workstreams in parallel on the same baseline, then integrated the ones that worked into one
**opt-in** experimental preset, `GameConfig.m4_emergent()`. Historical M4 defaults are unchanged. No PPO was run.

| | |
|---|---|
| **Decision** | **GO WITH CAVEATS.** Freeze `m4_emergent()` and proceed to the M4.3 PPO smoke test. |
| **Main caveat** | The headline interaction (a stronger champion controlling the lane by a credible threat, with the weaker champion conceding) is mechanically representable but **not rational at the current payoffs**. See §6 and §8. |
| **Driver** | `python -m lanegame.experiments.m4.emergent_ready` (6,347 turns, 1.5 s). Results in `results/m4_emergent_ready/emergent_ready.json`. |
| **Tests** | `tests/test_m4_emergent_ready.py` |

## 1. Execution

### How the work ran

- **Four git worktrees**, all from f504f5e, at `C:/Users/light/lol_wt/{A,B,C,D}`.
- **Four subagents ran concurrently**, one per worktree and branch. Each finished in 17–22 minutes of wall time.
  - No agent edited another's directory or the main checkout.
  - Each committed once and stopped. Nothing was pushed.
- **Integration ran after all four had finished**, in the main checkout.

### Branches

| Agent | Branch | Commit | Report | Own verdict | Integration |
|---|---|---|---|---|---|
| A: wave dynamics | `prototype/m4-emergent-wave` | 4211ea0 | `docs/m4_emergent_wave_report.md` (on its branch) | BLOCKED | **Not integrated** |
| B: positioning | `prototype/m4-positioning` | 8424985 | `docs/m4_positioning_report.md` | adopt opt-in | merged (24ae46b) |
| C: zoning | `prototype/m4-zoning` | de51899 | `docs/m4_zoning_report.md` | carry `"none"`, conditional | merged (c805d77) |
| D: investment + observation | `prototype/m4-investment-readiness` | 4a650ce | `docs/m4_investment_readiness_report.md` | adopt opt-in | merged (9b76a4c) |

The merges were run in the order the brief requires (wave, positioning, zoning, investment). The wave step was
skipped. The only conflicts were textual, in two places:

- the `M4_ECONOMY_FIELDS` tuple in `lanegame/config.py`;
- the field table in `tests/test_m4_config.py`.

Both were resolved by taking the union. No behavioural conflicts arose: B edits `resolve_turn_m4` steps 2, 3 and 9;
C edits only `collection_units`; D edits only `progression.Investment` and `observation_m4`.

## 2. Workstream results

### A — Natural wave dynamics: blocked, not integrated

**Rule:** `wave_dynamics="natural"` (opt-in on its branch).

- Every free front walks one cell per tick, regardless of the actions.
- The wave action only sets the champion's extra damage in a contact fight:
  - Slow Push: the last hit only.
  - Shove: every tick.
  - Freeze: only the enemy HP above 7.
- That damage requires presence and reach. A champion that is absent, channeling or in a connected trade adds none.

**What works**

- Determinism, HP conservation and mirror symmetry are verified.
- A large crash now resets the lane instead of re-crashing.

**Blocker:** at wave HP 4, waves never stack.

- A front gains HP only while it stands still, which means in contact, where it loses at least 1 HP per tick.
- The remnant dies before its reinforcement arrives. No unseeded trajectory exceeded 4 HP.
- Consequences:
  - no crashes, plates or bounces;
  - Shove is strictly dominated (−4.18 per 40 turns);
  - Freeze is inert.

**Timing checks**

- Tower damage of 2–3 HP/tick, slower minions, a longer lane and the ceil damage law did not unblock it.
- Only rescaling waves to 8 HP creates stacks. That is a rescale needing economy, plate and Freeze re-tuning, so it
  is a design decision, not a tweak.

**Decision:** not merged, as the brief requires for a blocked workstream. The prototype stays on its branch.

A also found a farm-spot interaction that matters for any later natural-wave work: under natural movement the probe
farm spot falls back on every wave cycle, halving farm. The existing opt-in `farm_spot_rule="pre_crash_front"`
fixes this in the scripted model and is selected by the preset.

### B — Independent positioning: integrated

**Rule:** `position_rule="standoff"`.

- Yield now targets the **stand-off cell** `max(-T, min(spot, e - farm_reach))`: the rearmost cell still in farm
  reach of the enemy front.
- A Yield that *starts* the turn in contact with an enemy Trade is **pinned**: it cannot evade and is hit as a Hold.
- Safe always evades.
- No action-space, mask or observation change (46 codes, 88 features). Yield codes 3–5 change meaning under the
  opt-in, so policies trained under `"m2"` must not be reused.

**Why it was needed:** under `"m2"`, position never carried a threat. Speed 2 covers half the lane, and Yield or Safe
evades any committed Trade in the same turn. So a Trade against Yield landed 0 of 24 times.

**Effect:** a strong A that holds and trades every 4th turn lands 6 of 6 trades on a stand-off B.

### C — Combat-driven zoning: integrated as `"none"`

**Rule:** `zoning_rule` ∈ `"presence"` (default, the M2 rule bit for bit), `"contact"` or `"none"`.

**Inventory of hardcoded gold losses** (C report §1). Only one is *automatic*: a 30% loss (15% in the collector's own
tower cell) whenever the opponent merely farms or trades within farm reach. That gold goes to no one. The others are:

- reach and absence;
- the Safe or retreat posture base;
- the 0.5 farm factor on a connected trade.

**`"none"`** removes the automatic proximity loss and adds no power-based term. Everything else is unchanged:
last-hit inefficiency, reach, posture cost, trade cost and XP.

**Finding:** removing the proximity loss does not by itself create threat-based zoning. With baseline positioning, B
never conceded in 162 solved games.

**Not used:** `"contact"`, which forces trading.

### D — Investment curve and observations: integrated

**Problem with the current curve** (p = 0.9): one 2-HP hit kills a full-health champion of the same level at S ≈ 68.
Real games invest 150–330 resources by turn 1,000. Lowering p (0.7, 0.5) only delays this to S ≈ 106–207.

**New curve:** `invest_curve="saturating"`, `M(S) = 1 + 1.5 (1 - sqrt(16 / (16 + S)))`, in exact integers.

- Values: ×1.045 / 1.16 / 1.32 at S = 1 / 4 / 10; ×1.94 at 100; ×2.16 at 300; never reaches ×2.5.
- No hit kills a full-health champion of the same level, or up to 3 levels lower, at any S.

**Observation:** `invest_obs_encoding="compressed"` stores `n / (n + R)` with R = 64 for invested and R = 8 for
unspent gold.

- Length stays 88 and the entries never saturate.
- Decoding is exact up to 23,106 invested and 8,184 unspent resources.
- 4 unspent gold now reads 0.33 instead of 0.004.

**Trade-off:** late leads buy little (300 vs 150 resources is only ×1.06).

## 3. Integrated rules: `GameConfig.m4_emergent()`

```python
GameConfig.m4_emergent(**overrides)
# = GameConfig.m4_continuous(
#       invest_curve="saturating", invest_gain=1.5, invest_scale=16.0,              # D
#       invest_obs_encoding="compressed", invest_obs_ref=64, invest_obs_gold_ref=8, # D
#       position_rule="standoff",                                                   # B
#       zoning_rule="none",                                                         # C
#       farm_spot_rule="pre_crash_front")                                           # M4.2d-A readiness fix
```

**Everything else is the M4 ruleset:**

- continuing time, γ = 0.99;
- the scripted wave rule (Slow Push `stack_creep`, Freeze maintenance);
- XP and levels 1–18;
- durability plates;
- the contact-cell chase;
- the tower-adjacent shot;
- continuous investment of all unspent gold at every return to base.

**Unchanged defaults:** `GameConfig.m4()` and `GameConfig.m4_continuous()` reproduce their previous behaviour, as
checked by the default-preservation tests of B, C and D and by `test_legacy_pins`.

**Why `"none"` is in the preset even though C's checklist does not pass:**

- The checklist items that fail (items 3, 5 and 6: concession flip, trade-back vs conceding, an active lane at equal
  power) fail under `"presence"` too (§5.C).
- `"none"` removes the only imposed, power-blind loss, in line with the design principle.
- It makes a forward A prefer Hold over Trade (§5.C).
- It raises contested gold by 43%, absorbed by the saturating curve.
- It is reversible with `m4_emergent(zoning_rule="presence")`.

## 4. Integration tests

Focused set:

```
pytest -q tests/test_m4_emergent_ready.py tests/test_m4_config.py tests/test_m4_positioning_intent.py \
  tests/test_m4_zoning.py tests/test_m4_investment_readiness.py tests/test_m4_continuous_investment.py \
  tests/test_m4_observation.py tests/test_m4_resolve.py tests/test_m4_combat.py tests/test_m4_economy.py \
  tests/test_m4_env.py tests/test_m4_farm_spot_rule.py tests/test_m4_integration.py tests/test_m4_progression.py \
  tests/test_m4_time.py tests/test_legacy_pins.py tests/test_invariants.py
```

Result: **230 passed, 3 skipped** (68 s). The full suite and the slow batteries were not run.

`tests/test_m4_emergent_ready.py` (9 tests) covers, under the preset:

| Correctness requirement | Coverage |
|---|---|
| Mirror symmetry | env-driven turns replayed mirrored (equal skills); 600 mirrored resolver turns with unequal skills; mirrored observation equals the other seat's |
| Legal-action masks | `mask == legal == legal_set` every turn; 46 codes |
| Deterministic replay | the same episode twice (with and without the env cache) gives identical actions, rewards and states |
| Gold and XP accounting | `gold + invested` grows by exactly `earned_units`; `xp` grows by exactly `xp_gain`; plate gold equals v·π(τ) (soak) |
| Minion HP conservation and explicit deaths | alive = start + spawned − killed − tower; no overflow (`soak`: 3,000 random turns plus 3 scripted pairs × 600) |
| Champion HP and position invariants | `check_invariants` every turn; speed and tower bounds (B's tests) |
| Observation validity | 88 finite entries in [0, 1]; `decode_m4` recovers the exact lane and M4 state for both seats every turn |
| Zero-sum reward accounting | r_A = −r_B = reward_units / scale = Σ(farm + plate + bounty)_A − (…)_B |

Plus two pins of the joint findings (§5).

## 5. Joint strategic tests (mechanical feasibility, not equilibrium play)

**Setup:** deterministic lines (u = None) and paired sampled windows (8 seeds, common random numbers).

- HP is valued at λ = 0.25 resources per HP unit (1/6 of a bar) where a single number is needed.
- **Investment gaps under the preset:**

  | gap | invested (A, B) | A/B attack multiplier |
  |---|---|---|
  | equal | 0, 0 | ×1.00 |
  | moderate | 16, 4 | ×1.24 |
  | large | 50, 0 | ×1.76 |

### A. Wave behaviour (scripted waves)

**Stacking and crashing.** A small advantage develops into a stack and a crash: from the pushed start, A's front
grows to 11 HP and crashes at 3, 9 and 7 HP within 48 turns.

**Counter-play.** The opponent can counteract with the right wave actions. Against the lane-control defender (Freeze
when legal, else Shove while not behind), the same pusher never crashes; its front never exceeds 4 HP. The defender
pays for this in Shoves: the pusher still out-earns it by 2.40 over 48 turns.

**No bounce.** No bounce occurs in any line: after a crash the crasher's follow-ups park and re-crash, as in M4.2d-A.

**Recall timing still matters:**

| Recall | gold missed | XP missed |
|---|---|---|
| during an own crash | 0.5 | 4 |
| from a neutral lane | 1.0 | 8 |

**Wave behaviour is unchanged by the preset;** only farm levels change (zoning `"none"`).

### B. Champion pressure

From the neutral lane, over 24 turns (`pressure` stage):

| A | B | rule | trades connected | B expected damage (centi-HP) | farm/turn A, B |
|---|---|---|---|---|---|
| Hold | respect (Safe if starting in contact, else Yield) | preset | 0 | 0 | 0.45, 0.20 |
| Hold + Trade every 4th | stand-off Yield | preset, equal | 6 / 6 | 549 | 0.39, 0.45 |
| Hold + Trade every 4th | stand-off Yield | preset, large | 6 / 6 | 967 | 0.39, 0.45 |
| Hold + Trade every 4th | stand-off Yield | `position_rule="m2"` | 0 | 0 (all evaded) | 0.35, 0.27 |
| Yield | Yield | preset | 0 (out of contact) | 0 | 0.45, 0.45; gank exposure −41% |

What this shows:

- **A can hold an aggressive position without attacking.** A holding the wave keeps contact every turn and is never
  forced off it.
- **The threat is real and scales with power.** A stand-off B that stays is caught: the expected damage it takes
  rises from 549 to 967 centi-HP as A's investment lead grows.
- **B has a real alternative.** Retreating avoids all damage but costs farm (B 0.20 vs 0.45 per turn).

### C. Threat-based zoning

**Deterministic counterfactual matrices** (C's checklist; zero-sum 4 × 4 over Hold, Yield, Trade and Safe; power
from item tiers ×1.2 / ×1.44 as a proxy):

| rule set | neutral, 6 turns: equilibrium (equal / large) | forward, 6 turns: equilibrium (all powers) | concession flip m* (neutral, forward) |
|---|---|---|---|
| baseline `m4()` | Trade/Trade, Trade/Trade | Trade/Trade (A Hold at equal) | 2.5, 3.0 → Yield |
| standoff + presence | Yield/Yield, Trade/Yield | Trade/Trade (A Hold at equal) | —, 3.75 → Safe |
| **integrated** (standoff + none) | Yield/Yield, Trade/Yield | **Hold/Hold** (V = +0.56 to A) | —, 3.96 → Safe |

(— marks a stand-off Yield that is not a concession; see the first bullet below.)

- At the neutral lane the stand-off is free. B's best response to a Trade is Yield, which keeps 100% of B's farm, so
  the flip at m = 1.0 there is not a concession.
- In the forward state, conceding means Safe, which keeps 44% of the farm (0.4 / 0.9).
- **The concession flip requires A to hold ×3.4–4.0 B's attack power.** C's target was 1.2–1.5.

**Sampled HP windows** (preset, 10 turns, 8 seeds; trades land, recalls at 40% HP, deaths). Value to A, ± s.e.:

| state / gap | A Hold, B Contest | A Trade, B Contest | A Trade, B Trade back | A Trade, B Retreat | A Hold, B Retreat |
|---|---|---|---|---|---|
| neutral / equal | +0.36 ± 0.24 | −0.69 ± 0.19 | +0.28 ± 0.28 | +4.03 ± 0.20 | +2.36 ± 0.06 |
| neutral / moderate | +0.36 | −0.04 ± 0.39 | +0.26 ± 0.37 | +4.03 | +2.36 |
| neutral / large | +0.36 | **+0.21 ± 0.43** | +0.48 ± 0.35 | +4.03 | +2.36 |
| forward / equal | **+2.53 ± 0.07** | +0.39 ± 0.40 | +1.10 ± 0.32 | +2.81 ± 0.28 | +3.97 ± 0.11 |
| forward / large | **+2.53** | +1.37 ± 0.45 | +1.09 ± 0.25 | +2.81 | +3.97 |

The Retreat columns are identical across gaps because the retreating B never connects, so A's investment never
enters.

Answers to the brief's questions:

- **Do combat advantages matter? Yes.** A's Trade into a contesting B goes from losing (−0.69) to winning (+0.21) as
  the gap grows. Over the same range B's deaths rise from 0.25 to 0.875 per 10 turns and its XP falls from 35 to 31.5.
- **Can the weaker champion rationally choose between contesting and retreating?** The choice exists, but
  **contesting dominates** in every tested condition.
  - In every cell, retreating instead of contesting or trading back raises A's value by 1.4–4.7 resources per
    10 turns.
  - Against a contesting B, A's Trade is worth less to A than simply holding, at every gap.
- **Does retreating sacrifice farm or XP?** It sacrifices farm (B earns 0.13–2.0 retreating vs 2.1–4.2 contesting);
  XP is not lost, since
  the retreat cell stays in XP range.
- **Can the stronger champion benefit from Hold instead of Trade once it has positional control? Yes.** In the
  forward state A Hold earns +2.53, more than A Trade at any gap (+0.39 to +1.37), with no fighting at all. The source
  is wave and position control, not deterrence: B still contests.

### D. Recall and investment

**Favourable windows exist.** A Recall during an own crash costs half as much as one from a neutral lane (§5.A).

**Returning stronger works.** Every return invests all unspent gold: 4 / 8 / 16 resources give ×1.16 / ×1.27 / ×1.44
attack power. In 300-turn scripted games both champions reached 69–118 resources invested (×1.85–1.98).

**Leverage is weak within short windows.** Recalling now with 4–16 gold, then trading for 12 turns, scores −1.2 to
−1.6, against −0.3 to −0.5 for staying. The Recall's cost (missed farm and XP) is not recovered within the window;
the permanent benefit is not charged. The scripted `gold_recaller` loses to `hold_farmer` by 5.8–6.0 per 300 turns.

**Conclusion:** investment matters only through trades, and trades are marginal except at large gaps. This is
consistent with M4.2c.

## 6. Remaining limitations

1. **Threat-based concession is not rational at current payoffs.** This is the main caveat.
   - **The mechanism is in place:**
     - stand-off positions;
     - the pin;
     - damage that grows with power;
     - a Trade that zones a retreating opponent without connecting (A Trade vs B Retreat: +4.0, 0 trades connected).
   - **What breaks it:** the weaker champion never retreats.
     - Retreat costs 0.15–0.4 farm per turn in the sampled windows (up to the full 0.45 deterministically): the Safe
       base is 0.4 against the contest 0.9, and an evasion from a Trade leaves farm reach.
     - Being traded costs B less than that at λ = 0.25. The trader also halves its own farm, so in every sampled cell
       an A that trades a contesting B does worse than an A that just holds.
   - **A conflict between workstreams:** concession needs a multiplier of ×3.4–4.0. D's curve is bounded below ×2.5
     on purpose (no one-shots), and realistic gaps reach at most about ×1.76. So investment alone cannot reach the flip
     in the preset.
   - **This is a coefficient decision for reviewers, not a mechanics defect** (B report caveat 1; C report §4.4).
2. **No bounce: the scripted wave rule keeps pushes permanent** (M4.2d-A). The natural-wave prototype fixes the reset
   but cannot stack at wave HP 4; it needs a wave-HP storage rule or a rescale (A report §9).
3. **The lane is passive at equal power.** At equal power the deterministic equilibrium of the integrated rules is
   mutual Hold or Yield (M4.2b's passive-equilibrium risk). The baseline's mutual Trade has the same character in the
   other direction.
4. **Late investment leads buy little** (D's trade-off), and Recall-to-invest pays back only over horizons longer
   than our windows.
5. **The tests here are deliberately small:**
   - linear HP valuation in the matrices;
   - 10-turn sampled windows that do not charge absence beyond the window;
   - stationary scripted lines;
   - an item-tier power proxy in the matrices;
   - one or two seeds for the scripted games.

## 7. Budget

**Simulation turns**

| | experimental turns | test turns (not budgeted) |
|---|---|---|
| A wave | ≈ 6,950 | — |
| B positioning | ≈ 9,160 (+ 1,200 verification digest) | — |
| C zoning | ≈ 12,536 (4,966 in the reported run) | ≈ 7,500 |
| D investment | ≈ 4,000 | ≈ 1,500 |
| Integration driver | 6,347 | ≈ 7,000 (`test_m4_emergent_ready`) |
| **Total** | **≈ 39,000** | |

- Every workstream stayed under the 15,000 cap.
- Three exceeded the 5,000 target, mostly because they re-ran their driver during development.

**Wall time:** each agent took 17–22 minutes; the integration driver ran in 1.5 s.

**Focused test results**

- Integration set: 230 passed, 3 skipped.
- Per workstream:
  - A: 228 passed, 2 skipped (on its branch);
  - B: 93 passed, 2 skipped;
  - C: 85 + 38 passed, 2 skipped;
  - D: 98 + 18 passed, 4 skipped.

## 8. Does anything block PPO?

**Nothing blocks M4.3 smoke testing:**

- the environment is correct (§4);
- it is fast (about 4k resolver turns per second);
- its observations are bounded and lossless;
- its damage is bounded;
- its decisions matter: wave management, crash-timed Recalls, trading when ahead, and the stand-off versus contest
  choice.

**Expectations for PPO:** it should not be expected to discover threat-based concession under `m4_emergent()` as
frozen. The payoffs currently make contesting correct.

**If that interaction is required before full training,** the smallest follow-up experiment is a 2 × 2 rerun of the
`zoning` and `hp` stages of `emergent_ready` (about 10k turns):

- trade damage scale ×1 or ×2;
- Safe or retreat collection keeping the stand-off farm base, versus the current slope.

It would report the flip m* and the sampled Contest-versus-Retreat values. The choice of which lever to move belongs
to the reviewers.

## Decision

**GO WITH CAVEATS.** Freeze `GameConfig.m4_emergent()` as the experimental ruleset and proceed to M4.3 PPO smoke
testing, with the documented limitations:

- concession is not rational at current payoffs;
- pushes have no bounce;
- the lane is passive at equal power;
- late investment leads are weak.

None of these breaks the environment or its core decisions. The first limits the headline interaction, so we should
decide on it before drawing strategic conclusions from trained policies.
