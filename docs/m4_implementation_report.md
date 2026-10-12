# M4.2 Implementation Report — Integrated Champion Economy and Wave Dynamics

*Branch `implementation/m4-integrated` (from `design/m4-integrated`, 085908d), 2026-10-11. Specification:
[m4_integrated_design.md](m4_integrated_design.md) ("design §n"), [m4_implementation_plan.md](m4_implementation_plan.md)
("plan"), [m4_design_audit.md](m4_design_audit.md) ("audit"). Raw results and commands: [results/m4/](../results/m4/README.md).
No PPO research run was made; the two bounded training runs below are integration smoke checks only.*

## Summary

- **Implemented:** WP0–WP10, every work package of the plan, with their unit tests and the 12 cross-mechanic
  integration tests of plan §4.
- **Legacy (S0):** every legacy configuration is bit-identical. Pins taken on the unchanged engine cover lane steps,
  observations, M2 resolution, a 200-episode rollout, an M3b checkpoint and solver values, and they pass. All 500
  pre-existing tests pass.
- **Mechanics (S1):** 679 tests pass, plus 2 strict expected failures that record design findings. A 68,000-turn
  soak found no invariant violation. The observation decodes exactly on all 20,572 reachable lane states and on
  50,000 full M4 states. The exhaustive lane bounds (A_max = 32, ≤ 4 rear packets per cell, siege ≤ 23) are
  re-verified on the implemented lane.
- **Progression (S2):** passes. The median turns to reach levels 2 / 6 / 11 / 18 are 4 / 19 / 58 / 144, against
  targets of 4 / 20 / 59 / 147 (all within 5%).
- **Freeze gate (S3):** **G1, G2 and the G3 rule pass.** The measured break windows of a Shove breaker are longer than
  the design's lane-only windows. The cause is a mechanism in the unchanged M2 movement rule, explained in §5.
- **Economy (S4):** passes its stated criteria. Recall has a two-sided sign pattern, no scripted strategy dominates,
  and snowballing is moderate. **Two design-level weaknesses are found:**
  - item tiers have almost no measurable value, because item leads last ~16–22 of 400 turns;
  - shove/siege play is strongly unprofitable for scripted agents, which is the audit's "passive equilibrium" risk.
- **Throughput:** M4 runs at 6,444 turns/s without the lane cache and 8,818 with it, against a target of 4,000.
- **Readiness:** M4.3 can start. Two design decisions should first be reviewed (D4 items; the M2 farm-spot interaction
  with Freeze breaking, §7).

---

## 1. Work packages and files

| WP | Delivered | Main files |
|---|---|---|
| WP0 | Ruleset contract `legacy` / `m2c` / `m4`; time fields; 22 M4 economy fields with off defaults; `GameConfig.m4()`, `GameConfig.champions_continuing()`; `validate()` enforces the contract table and rejects economies not representable in micro-resources (exact rationals); legacy pins taken **before** any engine change | `lanegame/config.py`, `tests/legacy_pins.py`, `tests/test_legacy_pins.py`, `tests/test_m4_config.py` |
| WP1 | Continuing time (no terminal, truncation only, `reset(k0)` keeps k0 mod P); `time_features=False` drops the countdown; `PPOConfig.continuing` (γ from the game, λ 0.95, segments of 128, games of 512 with seeded staggered first games); ρ_train archive starts; matchup-change guard; the `gank_reward` passthrough fix; discounted evaluator E1 / E2 with the simulator-derived tail bound (Ω = 68.34); `play_episode` requires `max_turns` | `lanegame/env/core.py`, `env/observation.py`, `env/discounted.py`, `env/starts.py`, `rl/ppo.py`, `rl/train.py`, `rl/shared.py`, `rl/evaluate.py`, `tests/test_m4_time.py` |
| WP2 | Integer tables (H, u, AP, damage, Pressure damage, X(L), n(L)), `level_of`, purchase, gold clamp | `lanegame/progression.py`, `tests/test_m4_progression.py` |
| WP3 | `TurnResult.tower_hits` (start-of-tick siege mass, every mode, additive); `plate_mode="durability"` (no lane plate opportunities); exhaustive lane enumeration and recorded bounds | `lanegame/mechanics.py`, `lanegame/lane_enum.py`, `tests/test_m4_tower.py` |
| WP4–WP6 | `M4Side` / `M4State`; `resolve_turn_m4`, design §5 steps 0–15: contact-cell chase, Freeze maintenance, centi-HP combat scaled by level and items, the tower-adjacent shot, XP within range 3, death before level-up, persistent tower damage and plates, the exact micro-resource ledger with the G_rem clamp, automatic purchases on the last absent turn. `champions.py` is unchanged | `lanegame/champions_m4.py`, `tests/test_m4_resolve.py`, `test_m4_combat.py`, `test_m4_economy.py` |
| WP7 | M4 environment step with the exact integer reward decomposition in `info`; the lossless 88-feature observation and its constructive decoder; run-time guards (raise, never clip); trace version 2 | `env/core.py`, `env/observation_m4.py`, `env/trace.py`, `tests/test_m4_observation.py`, `tests/test_m4_env.py` |
| WP8 | M4 scripted suite (hold_farmer, aggressive_trader, freezer, freeze_breaker, xp_keeper, gold_recaller, presser) and the S3b gate policies (incl. the adaptive breaker); the §6.4 parked-front harness; replayable witnesses for the ten design scenarios + five validation situations | `env/agents_m4.py`, `lanegame/scenarios_m4.py`, `lanegame/data/m4_scenarios.json`, `tests/test_m4_scenarios.py`, `tests/test_m4_integration.py` |
| WP9 | Diagnostics on M4: snapshots carry the M4State, discounted windowed rollouts, a labelled optional critic bootstrap, audit with discounted returns-to-go and phase instead of turns-left | `lanegame/diagnostics/branch.py`, `audit.py`, `tests/test_m4_diagnostics.py` |
| WP10 | Drivers for S1–S4, throughput and memory, and the training smoke checks | `lanegame/experiments/m4/`, `results/m4/`, `tests/test_m4_freeze_gate.py` |

Commits on the branch:
- 198b725 — pins;
- b73a1fa — WP0, WP2, WP3;
- 96b3310 — WP4–WP6;
- 99eacef — WP1, WP7 and the agents;
- 09b4420 — WP8 and the integration tests;
- 4d12fc9 — WP9;
- df5f159 — WP10;
- 1a40cf5 and the final commit — validation fixes, results, documentation.

## 2. Differences from the approved specification

**Mathematical differences (one, a specification contradiction):**

1. **u(14) = 189, not 188.**
   - The design gives the formula u(L) = ⌊H(L)/6 + 1/2⌋ and also a table, and calls both normative. For L = 14,
     H = 1,131, so the formula gives ⌊188.5 + 0.5⌋ = 189, while the table shows 188.
   - The table was printed by the prototype with Python's `round()`, which rounds half to even. This is the only
     half-way case where the two rules differ (L = 5's 121.5 rounds to 122 either way).
   - The implementation follows the formula, consistent with every other half-up rounding rule in §8.2. The WP2 test
     asserts the rest of the table exactly and u(14) = 189, with this explanation.
   - Effect: at level 14 a gank trial or tower shot deals 1 centi-HP more.

**No other rule differs from M4.1.** The following are placement or interface choices (also listed at the top of the plan):
- `game_length`, `segment_length` and the evaluation window are training/evaluation parameters (`PPOConfig`,
  evaluator argument), not `GameConfig` fields. They never change the game.
- The observation layout follows the ruleset, rather than being a separate field.
- `evaluate_discounted` lives in NumPy-only `env/discounted.py`.
- `next_price` returns ledger units.
- The feature `plates_remaining` is named `plates_left`, so that no feature name contains "remaining" (T1.3).

**Implementation details the specification left open:**
- **"Can buy both remaining tiers"** is computed as "can buy the next two tiers". With 2 tiers this is identical to the
  design.
- **The gold feature** is normalised by G_rem(0) = 10⁷ μ, which equals the design's 10⁷ for the default prices.
  `validate()` rejects price totals above 2²⁴ units, where float32 exactness would fail.
- **The `gank_damage_scale` alternatives** of the parameter table ("none", "attacker_level") and
  **`plate_gold_requires_presence`** (A9) are available as configuration options. They are off by default.
- **The adaptive breaker** "observes a maintained Freeze" through each turn's public result (`notify`), because the
  observation does not contain the opponent's last action.
- **All gate policies use the low-HP recall rule**, and the defenders trade on a positive duel edge, so the gate's
  continuations include realistic deaths, recalls and trades. The three defenders share identical behaviour rules and
  differ only in their wave rule.

## 3. Tests and legacy compatibility (S0, S1)

- **Full suite with the slow tests:** `pytest --runslow` gives **679 passed, 2 expected failures** in 18.6 min.
  - It includes all 500 pre-existing tests unchanged.
  - The two expected failures are strict: they flag if they ever start to pass. They encode two plan requirements the
    implemented game does not meet (§5).
- **Legacy pins** (`tests/test_legacy_pins.py`):
  - SHA-256 digests of 2,000 lane steps and observations each for v1, v2, the siege default and champions();
  - 2,000 M2 champion resolutions with chance vectors;
  - a 200-episode scripted rollout;
  - the M3b checkpoint's head-to-head against hold_farmer;
  - the K = 16 solver values of the three conftest models (every node value);
  - the K = 32 siege solution's metadata, plus a slow full re-solve (172,907 states, value 0).

  All digests were computed on the unchanged engine before any M4 code existed, and all pass after the full
  implementation.
- **M2 equivalence (WP4).** With chase rule "m2", unconditional Freeze and no adjacent shot, the M4 resolver
  reproduces M2's resolution on 5,000 random M2 tuples: positions, modes, lane, farm, ×100 damage and deaths. With each
  M4 rule switched on alone, every difference is explained by that rule:
  - contact-cell chase: all differences are the "chaser at its own tower cell" case of design §7.4;
  - Freeze maintenance: every difference is an unmaintained Freeze replayed exactly as M2 with Slow Push in the
    authoritative lane step.
- **`m2c` fixture (T1.10).** The resolution and rewards of `champions_continuing()` equal `champions()` on 5,000
  tuples and 20 episodes. Only termination and the two countdown features differ.
- **S1 checklist.** Each item is asserted by tests and, where marked, on every turn of the soak:
  - zero-sum rewards (per turn);
  - mirror equivariance of the resolver under `mirror_chance` and `swap_seats`, and of the observation;
  - deterministic replay (CRN, snapshot restore);
  - masks equal to the legal sets;
  - simultaneous resolution (both actions committed; the opponent's branch-turn action is independent of the focal
    candidate);
  - minion conservation (alive = initial + spawned − removed, no overflow);
  - progression invariants (monotone x, N, τ; L = level_of(x); plate gold = v·π(τ));
  - purchase timing (only on return turns; Recall t → buy at the end of t + 1; death t → buy at the end of t + 3);
  - the exact ledger (gold = earned − purchases − clamped, in integer μ, every turn);
  - tower durability (monotone, clamped at θ₃);
  - continuing time (T1.1–T1.13);
  - observation reconstruction (O1–O7).
- **Long-horizon soak:** 68,000 turns of random-legal and scripted play, with every design §14 invariant asserted on
  every turn. There were no violations, and every observation decoded exactly. The largest one-turn reward was 1.56,
  within the guaranteed bound of 8.575 (`results/m4/soak.md`).
- **Exhaustive checks on the implemented M4 lane** (slow tests):
  - **B1:** 20,572 reachable (state, phase) pairs; A_max = 32, one-turn kills ≤ 15, siege ≤ 23, tower hits ≤ 11, rear
    packets per cell ≤ 4. This equals the recorded bounds behind the tail bound and the lossless observation.
  - **O1:** the decoder round-trips on every reachable lane state, for both seats.
  - **O2:** all 96 classes of M2 lane states that share one M2 observation get distinct M4 encodings.

## 4. Validation outcomes S0–S4

| Stage | Outcome |
|---|---|
| S0 legacy | **Pass.** All pins bit-identical; 500/500 pre-existing tests pass; the M3b checkpoint loads and plays bit-identically |
| S1 mechanics | **Pass.** Unit and integration tests pass; 68,000-turn soak clean; observation exactly decodable (exhaustive lane; 50,000-state soak; every soak turn) |
| S2 progression | **Pass.** Medians within ±5% of the targets (below); XP absorption 95.1% vs the assumed 94% |
| S3a mechanics | **Pass, with a finding.** The parked-front table reproduces the design exactly for every Slow Push row; Shove rows break later (§5). Exhaustive comparison: Freeze differs from Slow Push in 1,126 of 15,488 Freeze-legal situations (7.3%), the same 1,126 as the M2 reference (which counted 15,458 legal situations) |
| S3b gate | **G1 pass, G2 pass, G3 rule pass**; the G3 break windows are longer than the design's lane-only windows (§5) |
| S4 economy | **Pass on the plan's criteria** (Recall with both signs; no dominant strategy). Two weaknesses are reported as design findings (§6) |

**S2 detail** (`results/m4/progression.md`, 400 games of 400 turns per matchup):
- **Levels.** Median turn of reaching L2 / L6 / L11 / L18 under mutual hold_farmer: 4 / 19 / 58 / 144. The
  interquartile ranges are within ±2 turns up to level 11.
- **Level gap.** aggressive_trader vs hold_farmer has a mean level gap of +0.46 at turn 60 and +1.27 at turn 150. This
  is less than the design's expected "1–2 levels by turn 60". Both reach the cap by turn ~200, which closes the gap.
- **Item purchases** come later than design §8.5 assumed:
  - mutual hold_farmer buys tier 1 at turn ~41–45 and tier 2 at ~61;
  - aggressive_trader vs hold_farmer buys at 31–35 and 60–67.

  The farm rate itself is as assumed (0.30 per turn). The delay is the wait for the next base visit; hold_farmer
  recalls about every 57 turns.

## 5. Freeze evaluation (S3)

**Setup** (`results/m4/freeze_gate.md`):
- 163 distinct reachable Freeze-legal states of the defender, each with a witness. The prototype's prefix families
  were rerun with chance events.
- Defenders: freeze, slowpush, cshove. Attackers: the six required, plus an extra trader.
- 50 paired common-random-number rollouts of 120 turns per scenario and policy pair; γ = 0.99.

**Results per attacker:**

| B policy | Freeze − Slow Push: mean (min, max) | scenarios with \|Δ\| > 0.05 | Freeze vs best alternative: better/equal/worse | B crashes F / SP | B plates F / SP | B gold F / SP |
|---|---|---|---|---|---|---|
| park_hold | +0.89 (0.00, +12.61) | 37% | 60/103/0 | 2.71 / 2.97 | 0.92 / 0.91 | 19.0 / 20.1 |
| park_safe | +0.32 (−9.68, +2.77) | 36% | 56/104/3 | 2.66 / 2.97 | 0.94 / 0.91 | 0.0 / 0.0 |
| breaker | +0.24 (−0.88, +1.58) | 88% | 113/20/30 | 4.97 / 4.70 | 0.78 / 0.22 | 13.5 / 14.0 |
| presser | −0.07 (−0.96, +1.07) | 90% | 50/17/96 | 5.40 / 4.92 | 2.85 / 2.33 | 13.2 / 13.6 |
| stacker | +0.34 (−0.94, +12.61) | 36% | 37/105/21 | 4.25 / 4.13 | 0.88 / 0.87 | 20.1 / 20.7 |
| adaptive_breaker | +0.38 (−5.72, +9.16) | 89% | 75/18/70 | 4.75 / 3.75 | 2.48 / 0.92 | 15.2 / 19.8 |

**Gate verdicts:**

- **G1 (distinct): pass.** |Δ| > 0.05 in at least 20% of the scenarios against all six required attackers. Under a
  stricter reading, requiring a paired difference that is also significant at 2 standard errors, it still passes:
  37% for park_hold, 28% for park_safe, 62% for adaptive_breaker.
- **G2 (non-trivial sign): pass.**
  - Freeze is at least 0.2 better than the best alternative in some scenarios against every required attacker.
  - It is at least 0.2 worse in some scenarios against park_safe, breaker, presser, stacker and adaptive_breaker. For
    example, it is worse against the presser in 96 of 163 scenarios, and against the adaptive breaker in 70.
- **G3 rule: pass.**
  - Checked tick by tick: the gate replays each turn's lane step and compares the holds the rule allows with the holds
    recorded. They are equal on every gate turn, with 0 violations among 60,015 ticks where a reaching Shove had
    blob + 4 ≥ 10.
  - An earlier version of the check used the start-of-turn front. It reported 445 apparent violations, which were all
    legitimate second-tick holds after the front had lost HP. It was replaced, and both runs are otherwise
    bit-identical.

**Reported per pair** (design §6.6) in the results file: crash delay, B's gold under Freeze minus Slow Push, both
sides' XP, exposure (gank occurrence, gank HP, trade HP, tower-shot HP, deaths), tower damage, plates, and Freeze turns
chosen and maintained. A representative reading:
- **Freeze reproduces the prototype's mechanism.** It denies crashes against passive attackers: park_hold crashes
  2.71 vs 2.97 times, and B's gold falls by 1.1 while its XP is nearly unchanged.
- **It invites siege conversion by Shove + Pressure.** Against the adaptive breaker, B deals 4,921 vs 1,771 tower
  damage and breaks 2.48 vs 0.92 plates.

**Findings that deviate from design expectations** (no mechanic was changed; contingencies are not triggered, because
G1 and G2 pass):

1. **A Shove + Hold breaker loses reach on the turns its push would succeed.**
   - In the probe pass (design §5 step 1, M2's unchanged rule), the breaking front crashes. The breaker's farm spot
     becomes its next packet near its base, so its Hold walks back and its reach no longer covers the front. In the
     authoritative step the hold then stands.
   - The breaker therefore pushes only every other turn. On the parked starts, 3–5 HP fronts break after 6–9 turns
     (design: 3–6). Fronts of ≥ 6 HP break at once in phases 0 and 2, but only after 2–6 turns in phases 1 and 3
     (design: at once).
   - On the reachable scenarios, 58% of Freezes against a front of ≥ 3 HP break within 6 turns, 80% within 12, and
     99.9% within the 120-turn window (median 6).
   - The hold rule itself is exact (G3 rule). The design's windows were measured on the lane alone, with full reach.
   - Kept as a strict expected failure ("presser breaks within 6 turns").
2. **The design's own §6.4 table contradicts plan §4.1 in 3 cells.**
   - For a creeping front of 6 HP in phase 1, and of 7 HP in phases 2 and 3, the held front grows and crashes earlier
     under Freeze (11 vs 15, 10 vs 14, 9 vs 13).
   - The resolver reproduces these values exactly. The integration test asserts "strictly later" only where the table
     says so.
3. **F0 against the stacker** (Freeze worse than Slow Push) does not hold on the parked starts: Freeze is better there,
   because the freezer keeps the stacker farming under its tower and trades it with home advantage. On the 26
   reachable scenarios with a 6–8 HP front, the outcome is mixed: Freeze worse in 7, equal in 12, better in 7. The test
   asserts that worse cases exist, and records the parked-start statement as a strict expected failure.
4. **A continuously shoving presser never gets Freeze legality after its own crash.** Its shoved waves meet the bounce
   in mid-lane. The crash → bounce → Freeze-legal chain of scenario 7 appears only under conditional Shove, about 20
   turns after the siege clears.
5. **In scripted play, Freeze is legal only against shovers:** 38 of every 100 turns against breaker or presser, 0
   against hold_farmer, stacker or aggressive_trader. Freeze matters exactly in the rare states the audit identified
   (§2.1).

## 6. Economic sanity (S4)

**Setup** (`results/m4/economy.md`): paired counterfactuals, 60 rollouts per variant, W = 1,000 (guaranteed
truncation error per value ≤ 0.003), scripted continuations. Every number below is an approximate champion-game
measurement, not an equilibrium value.

- **Recall (B).** Recall-now minus Hold-now, in 6 recall-legal states × HP {2, 4, 6}/6 × {affordable, not affordable}:
  - significantly positive in 3 cells, all at HP 2/6;
  - significantly negative in 18 cells (at full HP, and when an enemy wave is about to crash).

  **Both signs are present, so the plan's degeneracy test passes.** But the sign is driven by HP: in no state is
  "affordable" distinguishable from "not affordable".
- **Item value (A + diagnosis).**
  - A free tier, or buying now instead of at the next return, has no significant value: every 95% interval contains 0,
    and all |Δ| are ≤ 0.44.
  - This holds even under constant mutual trading: about 155 connected trades per 400 turns, a free-tier value of
    +0.17 ± 0.23, and the opponent's death count unchanged.
  - **Diagnosis:** the two champions' items differ on only 16–22 of 400 turns. Both tiers cost 10 resources against
    ~0.3 per turn of income, and deaths and recalls bring champions to base often. At equal items, attack power cancels
    (audit §1.4). An item lead is therefore short-lived and buys almost no kills.
  - The item track is weak, not exploitable. Its purchase motive for Recall is negligible: the audit §2.3 risk
    "Recall stays unused / items arrive only at deaths" is confirmed in its item part.
- **Freeze positions (C).** Under a held freeze, the opponent standing forward (+1, Hold) instead of back (−1, Safe)
  is worth +22.0 ± 0.6 over 1,000 turns. Against a trading freezer it is +19.5 ± 0.4. Sitting back to keep XP is
  strongly dominated by farming the freeze forward in these continuations.
- **Recall vs death (D).**
  - Recall from −1 is significantly better than dying in 4 of 5 states, by +0.9 to +1.7.
  - In one state, death is better by 0.85 ± 0.65 from −1 and by 0.93 ± 0.63 after a walk-back. The counters show this
    is downstream trajectory timing: about one more death of A later in the Recall branch.
  - Not a severe exploit, but the design's claim that death is never cheaper does not hold universally.
- **Denial by strength (E).** A strong freezer (+2 levels, +1 tier) that advances and trades denies the xp_keeper no
  more than an equal one (+0.008 ± 0.10). The xp_keeper always evades to its tower cell, so strength never comes into
  play. The keeper loses about 40 resources over 1,000 turns either way: XP and gold denial depend on the victim's
  evasive behaviour, not on the attacker's power.
- **Plates (F).**
  - A presser takes 2.65 plates per 400 turns against hold_farmer; a freeze_breaker 0.28; a stacker 0.
  - No plate gold was earned while away. The crash-then-recall concern (audit §2.4) does not arise in scripted play.
- **Snowballing (G)** (mirror edge traders, 300 games):
  - P(final discounted margin > 0 | ahead after 30 turns) = 0.67; behind: 0.27; correlation 0.59.
  - Level gaps of ≥ 2 occur in ≤ 2% of games.
  - Snowballing is well below the audit's 90% threshold.
- **Dominance (H)** (E1 round robin, W = 1,000, both seats, CRN):
  - No scripted strategy beats all others.
  - The top cluster is hold_farmer, xp_keeper, freezer, gold_recaller and edge_trader (row means +4.3 to +4.8).
  - presser (−12.3) and freeze_breaker (−12.7) lose heavily: standing under the enemy tower exposes them to ganks
    (the presser died 59 times in one 2,000-turn game against hold_farmer), which costs more than plates earn.
  - **Siege pressure is unprofitable for scripted play.** This is the audit's largest risk (§2.1, a passive
    equilibrium), here at the scripted level.

**Severe exploits:** none found. No dominant strategy exists, and degenerate death-shopping and crash-then-recall do
not dominate. The weaknesses are the opposite of exploits: mechanics whose incentives are too weak (items, siege).

## 7. Throughput and performance

(`results/m4/benchmark.md`: single thread, uniformly random legal actions, idle machine.)

| Configuration | env turns/s, no cache | env turns/s, LRU 20,000 | resolver turns/s | observe() µs |
|---|---|---|---|---|
| M2 champions() | 6,933 | 10,283 | 11,582 | 6.1 |
| m2c fixture | 6,998 | 10,449 | 11,854 | 5.6 |
| **M4 m4()** | **6,444** | **8,818** | 11,347 | 10.5 |

- **Target.** The ≥ 4,000 turns/s target is met with a 61% margin. M4 is 7–14% slower than M2.
- **Where the time goes** (profile of the M4 step): the resolver is about 52% of step time, the two observations
  about 23%, and building the large `info` dictionary most of the rest.
- **Memory.**
  - With the lane cache, memory grows with the bounded LRU cache: 50 MB at 17,800 of 20,000 entries.
  - Without it, memory is flat at 0.44 MB over 100,000 turns, so there is no leak.
- **Training throughput:** 6,300 shared-PPO transitions per second on M4 (smoke run).
- **Gate cost:** 20.5M turns took 7 minutes on 10 processes.

## 8. Known limitations and unresolved issues

1. **Item tiers are economically weak (S4).** Review design D4 (prices and tier count; A6) before training. Do not
   tune prices to equalise Recall frequency (audit §2.3). Any change should target the persistence of item
   differences.
2. **Siege and shove play is strongly unprofitable in scripted play**, and Freeze is legal only against shovers. This
   confirms the audit's passive-equilibrium risk. Learned play may differ, but the start-state archive (ρ_train
   mixture) will likely be needed in M4.3.
3. **The M2 farm-spot rule delays Freeze breaking** (§5, finding 1). It is faithful to the specification, which keeps
   M2's steps 1–2. If reviewers want a "reaching Shove" to keep its reach while its front crashes, that is a rule
   change of the resolver (for example, a farm spot that ignores the probe's crash), and was not made.
4. **Plan expectations recorded as contradicted:** the presser's 6-turn window; F0 against the stacker on the parked
   starts; "strictly later in every phase" (contradicted by the design table); the presser bounce chain; the S2 level
   gap at turn 60 (+0.46 instead of 1–2).
5. **The `m2c` stepping stone is weak evidence.** After 300k transitions the `m2c` policy almost never shoves (entropy
   0.27), so "no time-conditioned Shove" holds trivially there. The M4 smoke policy shoves 4–6% of the time and is
   stationary across 64-turn windows (p = 0.27).
6. **S4 counterfactuals are scripted** and cover 4–6 snapshots per question. They are approximate measurements with
   paired confidence intervals, not game values.
7. **Decision D7** (the siege-wait rule) and the gate contingencies were not exercised, because G1 and G2 passed.
8. **Tail-bound constant.** The bound's A_max = 32 is enumerated for the default lane parameters only. A different
   lane signature raises an error until the enumeration is rerun.

## 9. Acceptance and readiness for M4.3

| Completion criterion | Status |
|---|---|
| WP0–WP10 implemented and integrated | **met** |
| Specified unit and integration tests pass | **met** (679 passed; 2 strict expected failures record documented deviations from plan expectations, not code defects) |
| Historical benchmarks unchanged | **met** (pins, 500 legacy tests, K = 16 and K = 32 solver checks, the M3b checkpoint) |
| S0–S4 executed and recorded | **met** (`results/m4/`) |
| Continuing-time and observation guarantees validated | **met** (T1.1–T1.13; tail bound Ω = 68.34 reproduced; O1–O7 including the exhaustive lane) |
| Integrated economic mechanics with meaningful situational consequences | **partly met.** Freeze, Recall (through HP), positioning under a freeze, plates and XP all have state-dependent consequences, but item tiers have no measurable value |
| No severe unexplained dominant exploit | **met** |
| Implementation report | **met** |

**Recommendation.** M4.2 is complete as an implementation and validation milestone, with two design findings that are
open for review.

**M4.3 (S5–S6) can begin**, on these conditions:
- every M4.3 result is reported with the item weakness and the scripted unprofitability of siege as known context;
- the start-state archive is used as a stated comparison (D8);
- reviewers decide beforehand whether D4 (item prices and tiers) should change. If it changes, S4's item and Recall
  sections must be rerun first: `python -m lanegame.experiments.m4.economy` takes about 6 minutes.

The legacy pins must pass on any commit used for an experiment (plan §8).
