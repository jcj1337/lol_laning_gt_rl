# M4 Implementation Plan: Integrated Champion Economy and Wave Dynamics

*Status: plan for review, 2026-10-11 (milestone M4.1, design only). Nothing here is implemented. The rules are
specified in [m4_integrated_design.md](m4_integrated_design.md) (cited as "design §n"); the concerns in
[m4_design_audit.md](m4_design_audit.md) (cited as "audit §n"). Coordinates are own-perspective (own tower −2, the lane
cell 0, enemy tower +2). Code references are to branch `research/openai-five-diagnostics` at 4e68c75.*

---

## 1. Principles

1. **One integrated ruleset, built in dependency order.** M4 is a single versioned ruleset, `GameConfig.m4()`. The
   work packages below are an order of construction. They are not separately released mechanics. No intermediate
   package is a research target, and no PPO result is reported until the validation gate (§7) passes.
2. **Legacy configurations are frozen.** `GameConfig()`, `.v2()` and `.v1()` must stay bit-identical, and so must
   the exact solver, the K = 16 / K = 32 pinned values and `GameConfig.champions()` (M2/M3 and the OpenAI Five
   studies). Every new field defaults to the legacy behaviour. Every new code path is reached only through
   `cfg.ruleset` ∈ {`"m2c"`, `"m4"`}, under the contract of WP0. The M2 `champions.resolve_turn` is not edited; M4 gets its own resolver, which reuses M2's
   pure helpers.
3. **No new actions.** M4 changes state, transitions, observations and time semantics. The 46-code action space and
   the legality functions stay unchanged, apart from the Freeze-maintenance substitution (design §6.3), which is a
   resolution rule and does not affect the mask.
4. **Integer state for everything that gates a threshold.** HP and tower damage are stored in integer centi-units;
   XP and items as integers; gold, earned gold and the reward numerator as integers in micro-resources
   (design §8.6). No comparison needs a tolerance.
5. **Tests before training.** Each package ships its unit tests. Cross-mechanic integration tests and the
   calibration measurements of §7 must pass before any PPO run longer than a smoke test.

## 2. Architecture

```
lanegame/
  config.py            + ruleset contract (legacy | m2c | m4), M4 fields, m4(), champions_continuing()   [WP0]
  progression.py       NEW: XP/level/HP/AP tables, damage, purchase, gold clamp (pure functions)   [WP2]
  mechanics.py         + TurnResult.tower_hits (additive), plate_mode "durability" accepted        [WP3]
  champions.py         unchanged except for importable helpers (no behaviour change; golden test) [WP4]
  champions_m4.py      NEW: M4ChampSide / M4State, resolve_turn_m4 (design §5 steps 0-15)          [WP4-WP6]
  env/core.py          ruleset dispatch, continuing time (no terminal), M4 info fields             [WP1, WP7]
  env/observation.py   M4 layout (no countdown), feature names                                     [WP1, WP7]
  env/agents.py        M4 scripted agents (freezer, breaker, recaller-on-gold, presser)            [WP8]
  env/trace.py         trace schema version 2 (M4 state and events)                                [WP7]
  rl/ppo.py            PPOConfig.gamma per preset, segment fields, reward scale                    [WP1]
  rl/train.py          segment collection, staggered starts, (fix: gank_reward passthrough)        [WP1]
  rl/shared.py         same for the shared collector                                               [WP1]
  rl/evaluate.py       discounted evaluation over a window, start-state distributions              [WP1]
  diagnostics/branch.py  discounted / bootstrapped returns, M4 Snapshot                            [WP9]
  experiments/m4/      NEW: calibration and validation drivers (results/m4/)                       [WP10]
tests/
  test_m4_*.py         NEW, one file per package (below)
  test_legacy_pins.py  NEW: golden hashes of legacy observe/step/solver outputs                     [WP0]
```

### 2.1 Interface contracts

| Contract | Signature (proposed) | Guarantees |
|---|---|---|
| Ruleset switch | `cfg.ruleset in ("legacy", "m4")`; `GameConfig.m4(**kw)` | `"legacy"` reproduces every pinned result; `validate()` rejects M4 fields that are set while `ruleset == "legacy"` |
| Progression tables | `level_of(xp) -> int`, `hp_max(L) -> int` (centi-HP), `hp_unit(L) -> int`, `attack_power(L, N) -> int` (basis points), `hit_damage(base, L, N) -> int`, `next_price(N) -> float or None` | pure, integer, precomputed at import, identical on every platform (tables in design §8.2 are the oracle) |
| Lane step | `step_turn(...)` unchanged signature; `TurnResult.tower_hits: Tuple[int, int]` (minion damage to the enemy tower this turn, minion-damage units) | `tower_hits` is computed in every mode and read only by M4; all other fields are bit-identical |
| M4 state | `M4Side(pos, hp, absent, xp, items, gold, tower_dmg)`, `M4State(a, b)`, `.mirror()`, `to_list/from_list` | own-perspective; `mirror()` swaps sides; invariants of design §3.4 hold after every step |
| M4 turn | `resolve_turn_m4(lane, m4, k, a, b, cfg, lane_step=None, u=None) -> M4Turn` | same chance-vector layout as M2 (`N_CHANCE = 10`, no new uniforms); `M4Turn` has every M2 `ChampTurn` field plus `xp_gain`, `level_up`, `tower_damage_dealt`, `plates_broken`, `plate_gold`, `purchases`, `freeze_maintained` |
| Environment | `LaneEnv(cfg, max_turns=...)`; `step` returns `terminated=False` always under M4; `truncated` when `max_turns` is reached | reward `(r, −r)`; `info` decomposes `r`; the observation has no countdown |
| Evaluation | `evaluate_discounted(cfg, x, y, episodes, seed, gamma, window, protocol)` | protocol E1 (from s₀) or E2 (held-out archive), never mixed; estimates `E Σ_{t<W} γ^t r_t`; reports the paired CI and the guaranteed tail bound γ^W·Ω (design §10.5) |
| Observation decoder (tests only) | `decode_m4(vec, cfg) -> (φ, Λ, C_A, C_B)` | exact inverse of `observe` on reachable states (design §12.2) |

## 3. Work packages

Each package lists its scope, its files, its dependencies and its required tests. Sizes: S ≤ 1 session,
M ≈ 1–2 sessions.

### WP0 — Ruleset scaffold, configuration contract and legacy pins (S)

**Configuration contract.** `config.py` gains three groups of fields, each with a default that reproduces today's
behaviour:

- `ruleset ∈ {"legacy", "m2c", "m4"}` (default `"legacy"`);
- **time fields:** `time_mode ∈ {"finite", "continuing"}` (default `"finite"`), `time_features` (default True),
  `gamma` (default 1.0);
- **M4 economy fields** (design §13; each default means "off"): XP, levels, items, `plate_mode="durability"`
  parameters, Freeze maintenance, chase rule, tower-adjacent shot, damage scaling, gold unit, observation layout.

`validate()` enforces the following table. Any other combination is rejected with a message naming the
offending field.

| `ruleset` | Presets | `time_mode`, `time_features`, `gamma` | M4 economy fields | Pinned |
|---|---|---|---|---|
| `"legacy"` | `GameConfig()`, `.v1()`, `.v2()`, `.champions()` | `"finite"`, True, 1.0 | all at their off defaults | **yes**, bit-identical |
| `"m2c"` | `GameConfig.champions_continuing()` | `"continuing"`, False, 0.99 | all at their off defaults; `champion_mode=True` | no (its dynamics are tested against `champions()`, WP1 T1.10) |
| `"m4"` | `GameConfig.m4()` | `"continuing"`, False, 0.99 | M4 defaults (design §13) | M4 pins created after WP7 |

**Consequences of the contract.**

- The draft's `GameConfig.champions(time_mode="continuing", time_features=False)` is now **invalid**. It would be a
  `"legacy"` config with a non-legacy time field, so `validate()` rejects it.
- The continuing-time stepping stone of WP1 is the separate fixture preset `champions_continuing()`, with ruleset
  `"m2c"`. It is a test fixture for the time semantics only, never a research ruleset: no results under it are
  reported as findings about any game.
- WP0's legacy pins therefore apply to exactly the `"legacy"` presets. The `"m2c"` fixture is outside their scope,
  so there is no conflict.

`to_dict`/`from_dict` round-trip the new fields. Old run directories and checkpoints load, because missing keys take
the legacy defaults and therefore `ruleset="legacy"`.

**`tests/test_legacy_pins.py`** must pass before and after every later package. It pins:

- the SHA-256 of `observe()` vectors and of `step_turn` outputs over a fixed set of 2,000 (state, k, a, b) tuples,
  for v1, v2, the siege default and `champions()`;
- the K = 16 solver values of the three conftest fixtures (already pinned) and the `results/siege/solution_K32.json`
  metadata;
- a 200-episode seeded `champions()` rollout digest (rewards, info fields);
- one M3b checkpoint's `head_to_head` against `hold_farmer` at fixed seeds (bit-identical).

**Other tests:**

- C1. `m4()` and `champions_continuing()` round-trip through JSON.
- C2. `validate()` accepts every row of the contract table and rejects:
  - a `"legacy"` config with `time_mode="continuing"`, `time_features=False` or `gamma < 1`;
  - a `"legacy"` or `"m2c"` config with any M4 economy field switched on;
  - `"m4"` with `positioning=False` or `champion_mode=False`;
  - `"m2c"`/`"m4"` with `time_mode="finite"`.
- C3. Old run directories (`run.json` without the new keys) load as `ruleset="legacy"`.

### WP1 — Continuing time, segments and discounted evaluation (M)

Depends on WP0. This package is built and validated first, because every later measurement depends on it. It is
developed on the `"m2c"` fixture, so the time semantics are validated before any M4 economy exists.

- `env/core.py`:
  - under `time_mode == "continuing"`, `terminated` is always False and `truncated` is set only by `max_turns`;
  - `reset(k0)` accepts any `k0 ≥ 0` and records only `k0 mod P`;
  - `horizon` is not used;
  - `step` after a truncation raises until `reset` is called (as it does today after `done`).
- `env/observation.py`: `time_features=False` drops `k/K` and `(K−k)/K` and keeps the phase one-hot.
- `rl/ppo.py`:
  - `PPOConfig` gains `segment_length` (default None = legacy) and `game_length` (time limit of one game, default
    None);
  - preset constructor `PPOConfig.continuing()`: γ = cfg.gamma, λ = 0.95, reward_scale and vf_clip retuned for
    returns of order 1/(1−γ) (design §10.4);
  - `compute_gae` is unchanged.
- `rl/train.py` / `rl/shared.py`:
  - rollouts are fixed-length segments;
  - a game is reset only at its time limit, a truncation, bootstrapped from the **pre-reset** final observation
    (code exists, train.py:288–297);
  - the first game of each environment gets a seeded random length in `[1, game_length]`;
  - start states are drawn from `ρ_train` (`"initial"`, or the mixture with an archive of witnessed states, design
    §10.4);
  - matchup and controller changes happen only at game boundaries (shared.py:217–219 relies on it), now asserted;
  - fix train.py:240, which drops `gank_reward`.
- `rl/evaluate.py`:
  - `evaluate_discounted(cfg, x, y, episodes, seed, gamma, window, protocol)` implements protocol E1 (from s₀) and
    E2 (from a held-out, mirror-closed archive `𝒜_eval` disjoint from `𝒜_train`); design §10.5.1;
  - it returns the estimate, the paired CI, the guaranteed tail bound γ^W·Ω (§10.5.3) and, optionally, an
    empirical tail estimate on a subsample extended to W = 1,000, labelled as such;
  - `head_to_head` keeps its legacy behaviour.
- `env/agents.py`: `play_episode` requires `max_turns` under continuing time (no infinite loop).

**Tests (`test_m4_time.py`).** The toy trajectories use a stub critic whose values are set by the test.

- **T1.1** Never `terminated` over 2,000 turns of random legal play (`"m2c"`; later also `"m4"`).
- **T1.2** `truncated` exactly at `max_turns`; `step` after it raises until `reset`.
- **T1.3** The observation is invariant under `k → k + P`, and no feature name refers to absolute time.
- **T1.4** With an exact critic on a deterministic toy chain (constant reward c, V = c/(1 − γ)), every GAE advantage
  is 0, including at truncations and segment ends. With a terminal-treated truncation, the same chain gives the
  documented nonzero bias. This test shows why truncation ≠ termination.
- **T1.5** Instrumented environment: the observation passed to `net.value` for a truncation bootstrap is
  byte-identical to the final pre-reset observation. It is never the next game's first observation.
- **T1.6** A segment cut mid-game bootstraps `V̂(s_T)` of the current observation. The λ-recursion does not cross a
  game boundary: a reward spike injected in game 2 leaves game 1's advantages unchanged.
- **T1.7** A biased critic V̂ = V + δ gives a truncation-step advantage of `r + γ(V + δ)(s′) − (V + δ)(s)` exactly.
  This documents that bootstrapping approximates the continuation value and is not unbiased for δ ≠ 0. The test
  asserts the formula, not unbiasedness.
- **T1.8** Forcing a matchup or controller change mid-game raises (the invariant behind shared.py:217–219).
- **T1.9** First-game lengths are seeded, reproducible, within `[1, game_length]`, and not all equal for ≥ 2
  environments. A fixed seed gives the documented lengths.
- **T1.10** `"m2c"` dynamics identity. For 5,000 random (lane, champ, k < 32, a, b, u), `resolve_turn` and the
  rewards under `champions_continuing()` equal those under `champions()`. Only `terminated` and the two countdown
  features differ.
- **T1.11** Evaluator:
  - E1 matches the closed form `c(1 − γ^W)/(1 − γ)` on a constant-reward pair;
  - the reported tail bound equals γ^W·Ω;
  - E1 rejects any start other than s₀;
  - E2 rejects an archive that is not mirror-closed or that overlaps `𝒜_train`.
- **T1.12** `play_episode` under continuing time without `max_turns` raises.
- **T1.13** Regression test: `train_ppo(skill_levels=…)` keeps `gank_reward`.
- **T1.14** The legacy pins.

**Stepping-stone check (fixture, not a research result).**

- Run on `"m2c"` for a 300k-transition smoke run.
- Confirm that a policy without time features shows no time-conditioned Shove: its Shove rate is stationary across
  windows of 64 turns.
- Confirm that the critic's value scale is sane.

### WP2 — Progression tables (S)

Depends on WP0. Can be built in parallel with WP1.

- `progression.py`:
  - integer tables `HP_MAX[L]`, `HP_UNIT[L]`, `AP[L][N]`, `XP_CUM[L]`;
  - `level_of`, `hit_damage`, `level_up_hp`, `next_price`, `gold_cap`, `purchase(items, gold) -> (items', gold',
    bought)`.
- **Tests (`test_m4_progression.py`):**
  - the tables equal design §8.2 / §9.2 exactly;
  - `level_of` is monotone, `level_of(XP_CUM[L]) == L` and `level_of(XP_CUM[L] − 1) == L − 1`, and the cap holds
    at 18 for every xp ≥ 544;
  - `hit_damage(2, …) ∈ {2·hit_damage(1, …) − 1, 2·hit_damage(1, …), 2·hit_damage(1, …) + 1}`;
  - AP is monotone in L and in N;
  - no value exceeds int32;
  - `purchase` is idempotent once it can no longer buy, never buys beyond tier 2, never leaves gold negative,
    and buys tier 1 before tier 2.

### WP3 — Lane additions: tower hits and durability mode (S)

Depends on WP0.

- `mechanics.py`:
  - `TurnResult.tower_hits[s] = Σ_ticks minion_damage(m_s(t))` over the ticks whose start-of-tick mass
    `m_s(t) > 0` (design §11.2);
  - `validate()` accepts `plate_mode="durability"`, under which crash-plate opportunities are never granted and
    Apply Pressure claims nothing in the lane step (`ap`/`bp` stay 0).
- **Tests (`test_m4_tower.py`):**
  - **B1 (slow marker, ~5 min).** Exhaustive reachable-lane enumeration re-checks the constants that the
    evaluation bound and the observation rely on: A_max ≤ 32, one-turn kills ≤ 27, tower hits ≤ 24, rears per cell
    ≤ R_max = 4, siege ≤ 48 (design §10.5.3, §12.2). It runs whenever lane parameters change; the scripts in
    `results/m4_design/scripts/` are the reference implementation;
  - tower hits for crashes of 4, 8 and 12 HP equal 4, 9 and 18 (design §11.2);
  - tower hits are 0 without a siege;
  - every legacy `step_turn` output is unchanged (pins);
  - under durability mode, Apply Pressure leaves `ap`/`bp` at 0.

### WP4 — M4 champion state and resolution skeleton (M)

Depends on WP2 and WP3.

- `champions_m4.py`: `M4Side`, `M4State`, `initial_m4`, `resolve_turn_m4` steps 0–10 of design §5, with:
  - the **contact-cell chase rule** (design §7.4) in place of M2's `target = e − rng`;
  - **Freeze maintenance**: a Freeze that is not maintained resolves as Slow Push (design §6.3);
  - M2's farming (gold), exposure and gank occurrence, reused as functions.
- `champions.py`: if helpers need to be exposed, only as moves with no behaviour change (the pins prove it).
- **Tests (`test_m4_resolve.py`):**
  - with every M4 economy switched off, the M4 resolver reproduces M2 `resolve_turn` on 5,000 random
    (lane, champ, a, b, u) tuples, except where the chase rule or Freeze maintenance differs. Every difference
    must be explained by one of the two rules: the test classifies them and fails on any unexplained difference;
  - chase: a trader at its own tower cell connects with an enemy at that tower's adjacent cell (the M2 defect,
    design §7.4);
  - maintenance: a connected-trading freezer, or one standing 2 cells from the freeze line, does not hold the
    enemy front;
  - an absent freezer never holds;
  - mirror equivariance of `resolve_turn_m4` under `mirror_chance`.

### WP5 — Combat and progression in the resolver (M)

Depends on WP4.

- Steps 9–12 of design §5:
  - centi-HP trade damage scaled by the attacker's AP;
  - gank and tower-shot damage scaled by the victim's HP unit;
  - the tower-adjacent shot;
  - simultaneous damage, then the death check, then XP and level-ups with the added-max-HP rule.
- **Tests (`test_m4_combat.py`):**
  - with mechanical skill fixed, changing N or L changes only the damage values and never `hit_p` or `double_p`;
  - damage equals `hit_damage(D, L, N)` for D ∈ {1, 2};
  - gank damage equals `X · hp_unit(L_victim)`;
  - the level-up HP rule, including multiple level-ups in one turn;
  - a lethal turn kills even when XP crosses a threshold (death first);
  - XP is gained by present champions only (away 0, channeling > 0);
  - XP range 3 vs gold reach 1 on constructed kill positions;
  - the caps: xp clamps at 544, `hp ≤ hp_max(L)`, and no overflow after 10,000 random turns.

### WP6 — Tower durability, plates, gold ledger, items, recall/death purchases (M)

Depends on WP5.

- Steps 11 and 13–15 of design §5:
  - tower damage from minions and Pressure;
  - plate thresholds and plate gold;
  - the gold ledger;
  - automatic purchase at the end of the last absent turn;
  - return with full HP at the new max.
- **Tests (`test_m4_economy.py`):**
  - a plate is paid exactly once when its threshold is crossed, and several plates crossed in one turn are all
    paid;
  - the tower damage counter never decreases and clamps at θ₃;
  - a purchase happens only at the end of an absent turn whose timer reaches 0, after that turn's gold credit;
    it is irreversible and never duplicated (two consecutive return turns are impossible);
  - spending does not change the reward;
  - `Σ_t r_t` equals (earned_A − earned_B) exactly over a 500-turn rollout, in integer μ;
  - `validate()` rejects an economy whose per-HP farm values, plate value or bounty are not integer multiples of
    1 μ (e.g. `farm_per_hp = 0.3333`);
  - the gold clamp never removes gold that could still buy something;
  - death and recall both route through the purchase step;
  - no tier is bought while the champion is present.

### WP7 — Environment integration, observation and trace (M)

Depends on WP1 and WP6.

- `env/core.py`: `ruleset == "m4"` dispatch to `resolve_turn_m4`; `info` adds xp, level, items, gold, tower
  damage, plates broken, plate gold, purchases, freeze maintenance, level-ups.
- `env/observation.py`: the M4 layout (design §12), feature names, mask (unchanged legality).
- `env/trace.py`: version 2 with the M4 fields. Version 1 traces are still readable.
- **Observation tests (`test_m4_observation.py`), design §12.2:**
  - **O1 (slow marker).** Decoder round-trip `D(observe(φ, S, side)) == S`:
    - for the lane block, on every state of the exhaustive reachable-lane enumeration (both seats, all phases);
    - for the full M4 state, on 50,000 states of a random-legal soak.
  - **O2.** The 85 colliding pairs of the M2 grid (`results/m4_design/obs_alias.txt`) have distinct M4 lane
    encodings.
  - **O3.** Gold. Constructed champions at N = 0 with g = 4.0 and 9.0 resources, and with g = price − 1 μ and
    price, give distinct observations. The affordability bits are exact. Every integer balance 0 … 10⁷ μ decodes
    exactly from its float32 feature, checked exhaustively in a vectorized test.
  - **O4.** HP, XP and τ: every integer value in range decodes exactly at every level.
  - **O5.** Run-time guards: a lane state with more than R_max rears in a cell, or siege > 48, raises in `observe`.
    Nothing is clipped silently.
  - **O6.** Mirror property, and invariance under `k → k + P`.
  - **O7.** Completeness: every field of `M4Side`/`State` appears in the decoder. A new state field without an
    observation feature makes O1 fail.
- **Tests (`test_m4_env.py`):**
  - observation entries are in [0, 1] and the length matches `feature_names` (88);
  - `observe(lane, k, B, cfg, m4) == observe(lane.mirror(), k, A, swap_seats(cfg), m4.mirror())`;
  - the mask equals the legal set;
  - `info` decomposes the reward exactly;
  - CRN: identical action sequences and seeds give identical trajectories;
  - a 10,000-turn random-legal-play soak finds no invariant violation (design §14).

### WP8 — M4 scripted agents and constructed scenarios (S/M)

Depends on WP7.

- Agents:
  - `freezer`: Freeze when legal, Hold/Safe within 1 cell of the freeze line, Trade when its duel edge is positive;
  - `freeze_breaker`: Shove + Hold;
  - `xp_keeper`: Safe when the enemy is freezing, else Hold;
  - `gold_recaller`: Recall when gold ≥ next price and HP ≤ h* and the own wave is not under threat;
  - `presser`: Shove, Pressure when sieging;
  - `hold_farmer` and `aggressive_trader`, carried over.
- `scenarios_m4.py`: deterministic constructed states for the ten design scenarios (design §15), each with a
  replayable witness from the standard start (a scripted prefix), as the earlier reachability work did.
- **Tests:** each witness reaches its scenario state; each scenario's mechanical assertions (below, §4) hold.

### WP9 — Diagnostics port (S)

Depends on WP7.

- `diagnostics/branch.py`:
  - the `Snapshot` carries `M4State`;
  - `rollout` returns the discounted return over a window; a critic bootstrap `γ^W V̂(s_W)` is optional and
    labelled as an approximation (design §10.5.4);
  - `audit.py` replaces `ret_to_go`/`turns_left` with discounted values.
- Drop the turn-relative overlays (`late_shove`, k-bins) for M4; use phase-relative bins instead.
- **Tests:** a snapshot restore replays exactly; paired candidates share chance streams (existing spy test,
  ported).

### WP10 — Calibration and validation drivers (M)

Depends on WP8 and WP9. Scripted and analytical only: no PPO.

`experiments/m4/calibrate.py` writes JSON before Markdown under `results/m4/` and covers the measurements of §7.

## 4. Integration tests (cross-mechanic; `test_m4_integration.py`)

Each test runs on a constructed scenario (WP8) with fixed chance vectors, unless it is marked statistical.

1. **Freeze ≠ Slow Push.** Start from the parked configuration (enemy front at own −1, own front just killed),
   reproducing design §6.4's table. Against a Slow Push front of 5–7 HP, and against a Shove front of 2–5 HP, the
   enemy front crashes strictly later under a maintained Freeze than under Slow Push, in every start phase where
   §6.4 reports a difference. Against a 3–4 HP Slow Push front both actions leave it parked for 24 turns: that
   equality is asserted too, as the documented lane-stable case.
2. **Freeze cannot hold everything.** A front of ≥ 10 HP (6 HP or more with a reaching Shove) is never held. An
   8–9 HP Slow Push front crashes within 8 turns despite a maintained Freeze.
3. **Shove breaks a Freeze without automatic punishment.** With the enemy front at 3–5 HP, a Shove + Hold breaker
   whose champion is within reach breaks the freeze within 6 turns when no trade and no gank happen (chance vector
   all 1.0). Its HP and gold efficiency on those turns equal those of the same Shove from the same position without
   a Freeze. The same breaker with its champion out of reach (end cell < 0) breaks later or not at all.
4. **Gold/XP separation.** Under the held freeze, a Safe opponent at own −1 earns XP and no gold. One at +1 earns
   both. One at −2 earns neither.
5. **XP denial needs commitment.** An `xp_keeper` at −1 loses XP only on turns when the freezer ends its turn at
   0 or further forward, i.e. is no longer covered by its tower.
6. **Item affects damage only.** For a grid of (L, N) pairs and fixed skills, `hit_p`/`double_p` are identical and
   the damage scales as `AP`.
7. **Level-up in combat.** Constructed turn: survivor at 50 centi-HP gains a level → `hp = 50 + Δhp_max`; a
   champion at 50 taking 128 dies even though its XP crosses a threshold.
8. **Recall purchase timing.** Recall on turn t with gold ≥ 4: the item is bought at the end of t + 1, the champion
   plays t + 2 with N = 1, full HP at the current level's maximum; gold earned by plates during t, t + 1 counts.
9. **Death purchase timing.** Death on turn t: purchase at the end of t + 3; play on t + 4.
10. **Crash → bounce → plates.** The `presser` vs `hold_farmer` (statistical, 200 paired seeds):
    - tower damage is monotone;
    - plate gold is paid once per threshold;
    - the bounce wave reaches the presser's side within the predicted window;
    - Freeze is legal for the presser on some turns after a bounce.
11. **Continuing play.** 2,000-turn games:
    - no terminal;
    - the caps are reached;
    - after both sides are capped and the plates exhausted, the per-turn reward distribution is stationary
      (blocks of 200 turns agree within sampling error).
12. **Zero-sum and symmetry.** For every test above, `r_A + r_B = 0` per turn. Swapping seats with
    `mirror`/`swap_seats` gives the mirrored trajectory.

## 5. Legacy compatibility and migration

- **Exact game.** Untouched (no dependency on `env`; audit confirmed solver.py:97–116 and best_response.py:194–200
  use `horizon` only through `LaneGame`). The pins of WP0 guard it.
- **M2/M3/OpenAI Five results.** `GameConfig.champions()` keeps the finite 32-turn horizon, the countdown features
  and γ = 1. Its checkpoints keep loading: new config keys get their defaults, and the 50-entry observation is
  unchanged.
- **No checkpoint migration** from M2 to M4. The observation, dynamics and objective differ, so M4 training starts
  from scratch. M2 numbers are never compared with M4 numbers as though they measured the same game. Reports
  cite them as historical context only.
- **Results layout.** `results/m4/` (calibration), `results/rl/m4/` (training), git-ignored checkpoints under
  `results/rl/checkpoints/m4/`.
- **Docs.** The roadmap's section G (three levels, two item tracks, `plates_fall_turn = 24`) is superseded by the
  M4 amendment (roadmap section O) and this plan.

## 6. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Legacy drift through a "harmless" refactor of `champions.py` or `mechanics.py` | medium | pins first (WP0); M4 in a new module; additive fields only |
| Throughput: XP/level/gold bookkeeping and the longer games slow rollouts (M2 env ~6k turns/s uncached) | medium | integer tables, no per-step allocation; benchmark after WP7 (target ≥ 4k turns/s single thread); keep the LRU lane cache |
| Evaluation cost: headline E1 values need W = 1,000 turns per game for a guaranteed tail ≤ 0.003 (design §10.5.3) | high | paired seeds; fewer games with CRN; W = 460 for development only (bound 0.67 plus an empirical tail estimate); critic bootstrapping only as a labelled, biased approximation |
| Start-distribution confusion: archive-trained policies reported on archive starts | medium | E1 (from s₀) for every headline; E2 (held-out, mirror-closed archive) reported separately; ρ_train stated with every result (design §10.5.1) |
| Value scale: returns of order ±10–30 with γ = 0.99 destabilize the critic | medium | reward_scale 0.1, retuned vf_clip, value-target checks in the PPO smoke |
| Early lock-in: rare actions collapse within ~50k transitions (diagnostics D.4), now with more of them (Freeze, Recall-to-buy) | high | start-state archive (diverse resets incl. freezes and sieges) and slower entropy decay as configurable options; report both with and without |
| Calibration may make one mechanic dominant (e.g. Recall-on-afford, freezing every bounce) | medium | §7 gates; adjust parameters only for stated mechanical reasons, never to equalize action frequencies |
| Snowballing makes most games decided early (audit §4) | medium | measure win-margin vs early lead (§7.6); caps and increasing XP thresholds are the built-in brakes |

## 7. Staged validation plan

Each stage must pass before the next starts. All stages except S5 use scripted agents and analytical checks only.

- **S0 — Pins.** The legacy pins pass.
- **S1 — Mechanics.** The unit and integration tests of §3–§4 pass. A 50,000-turn random-legal soak shows no
  invariant violation.
- **S2 — Progression speed.** With mutual `hold_farmer` (M4 agents), 400 seeded 400-turn games:
  - the median turn at which levels 2/6/11/18 are reached lies within ±25% of the design targets (design §9.3:
    ≈ 4 / 20 / 59 / 147 under mutual farming, which absorbs about 94% of the available XP);
  - with the learned-play-like `aggressive_trader` vs `hold_farmer`, report the gap.

  If a target fails, the only adjustable parameter is the XP scale `xp_per_hp`. A miss is reported, not tuned
  away silently.
- **S3 — Wave mechanics and the Freeze gate.**
  - **S3a, mechanics.**
    - The exhaustive lane enumeration (7.3% of Freeze-legal situations differ under M2 lane rules) is repeated
      with Freeze maintenance, and the fraction is reported.
    - A breaker breaks within the predicted time.
    - The freeze breaks with an 8+ HP enemy front.
    - The freeze/bounce statistics are reported: freeze-legal turns per 100, held turns, breaks.
  - **S3b, Freeze gate (pre-training, design §6.6).**
    - **Template.** The prototype `results/m4_design/scripts/freeze_gate.py` is ported to the implemented M4
      environment.
    - **Scenarios.** ≥ 100 distinct reachable Freeze-legal scenarios, each with a witness prefix.
    - **Defender policies:** maintained Freeze, Slow Push, conditional Shove.
    - **Attacker policies:**
      - park_hold, park_safe, breaker, presser, stacker;
      - an **adaptive** breaker that switches to Shove + Pressure when it observes a maintained Freeze.
    - **Play.** H ≥ 120 turns per pair, paired CRN chance events with trades and the tower-adjacent shot, and 50
      paired rollouts per scenario and pair.
    - **Reported per pair:**
      - crash delay;
      - B's gold (Freeze − Slow Push);
      - both sides' XP;
      - exposure (gank occurrence, gank HP, trade HP);
      - tower damage and plates;
      - discounted relative return (E2-style over the scenario set).
    - **Gates:** G1 (distinct in ≥ 20% of scenarios against ≥ 2 attacker policies), G2 (non-trivial sign: some
      ≥ +0.2 and some ≤ −0.2 against the best alternative), G3 (breakable within the §6.4–6.5 windows).
    - **Required test cases** (in `test_m4_freeze_gate.py`, on fixed scenarios from the prototype, with fixed
      chance vectors):
      - **F+ (Freeze advantageous):** a 5–6 HP parked front against park_hold. Freeze denies the creep crash, and
        its discounted return exceeds Slow Push's.
      - **SH× (Shove breaks Freeze):** a 3–5 HP front against presser. The freeze breaks within 6 turns and B's
        tower damage under Freeze exceeds that under Slow Push.
      - **F0 (Freeze not beneficial):**
        - a 3–4 HP front against park_hold: returns identical;
        - a 6–8 HP front against stacker: Freeze's return is below Slow Push's.
    - **On failure.** G1 or G2 failing is a design finding. The contingency corrections of design §6.6 are
      evaluated, in order, by the same gate before any training. No parameter is tuned to make Freeze frequent.
- **S4 — Economic sanity.**
  - Paired counterfactuals (diagnostics `branch`, ported) with scripted continuation policies:
    - the value of an item tier at a fixed state (bought vs not bought);
    - the value of a Recall with and without an affordable tier at HP 2/4/6;
    - the cost of standing at +1 vs −1 under a freeze, with and without a trading freezer.
  - Report these numbers. Recall-to-buy must be positive in some states and negative in others: if it is
    positive or negative in every tested state, report it as a design finding before training (audit §2.3).
- **S5 — PPO smoke and pilot.**
  - Smoke: 100k transitions, 1 seed; finite values, the critic's explained variance > 0, every action family's
    logit receives gradient.
  - Pilot: 1M transitions × 3 seeds of shared self-play (100/0); a learning curve against the M4 scripted suite.
- **S6 — Research runs** (only after S5). Paired-seed comparisons, multiple seeds (≥ 3, preferably 5),
  budget-matched ablations (e.g. Freeze maintenance on/off, tower-adjacent shot on/off, item tier count),
  exploitability probes with ≥ 2 warm + 1 scratch attackers at ≥ 1M steps per target. Report the budget, the
  attacker count and the per-attacker learning curves; the bound is a lower bound, never a certified Nash
  exploitability.

## 8. Conditions before expensive training begins

1. S0–S4 pass, and their results are written to `results/m4/`.
2. The open decisions marked **required** in design §16 have been reviewed, or explicitly accepted at their
   defaults.
3. Throughput is measured: M4 environment turns/s and shared-PPO transitions/s on this machine. The budget of the
   pilot is stated in wall-clock time before it starts.
4. The evaluation protocol is fixed in advance:
   - γ;
   - the window W (1,000 for headline) and its guaranteed tail bound;
   - protocol E1 from s₀, and whether E2 is reported, with the held-out archive 𝒜_eval;
   - the training start distribution ρ_train (δ_{s₀} or the stated mixture), recorded with every run;
   - the episodes per pairing;
   - the attackers per target;
   - the seeds (disjoint from training streams, tested).
5. A scripted-baseline table of the M4 suite (round-robin, discounted values) exists as the fixed reference, so
   the first PPO result can be compared against something.
6. The legacy pins pass on the commit that runs the experiment.
7. The Freeze gate S3b passes, or its failure and the chosen contingency have been reviewed.
8. The observation decoder tests O1–O7 pass: the observation is exactly Markov on the reachable set.
