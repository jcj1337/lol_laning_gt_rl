# M4.4a — Four-Action PPO Readiness, Strategic Diagnostics and Roadmap Update

*2026-10-13. Branch `main` (working tree on f931ae8, uncommitted for review). Infrastructure, instrumentation,
validation and documentation only: **no game rule, preset, reward, observation, action or PPO hyperparameter was
changed**, and the M4.4b training budget was not spent. Tables: `results/m44a/report.md`; M4.4b specification:
`results/m44a/m44b_plan.md`; code: `lanegame/experiments/rl/m44/`; tests: `tests/test_m44a_readiness.py`.
Companion documents: [the PPO pipeline explained](../rl/ppo_training_explained.md) and the updated
[roadmap](../rl_research_roadmap.md) (summary at the top, section AF).*

---

## 1. Executive summary

**Question.** Is the PPO pipeline ready to train shared-policy self-play on the three four-action environments, and
can we tell a policy that learns state-dependent laning strategy from one that merely samples different actions?

**Answer: yes, with the instruments built here.** In detail:

1. **The trainer needed no change.** `lanegame.rl.train_shared` is action-count agnostic. On all three environments
   (`m4_four_actions()`, `m4_four_actions_econ()`, `m4_four_actions_econ_aggro()`) one network plays both seats from
   its own perspective, both seats' transitions are trained, rewards are exactly antisymmetric, sampling respects the
   61-code masks, games are truncated and bootstrapped (never terminated), and runs are bit-for-bit reproducible
   (tested; §2, §7).
2. **Three compatibility gaps were found and closed** (none in the simulator or the PPO algorithm):
   * the M4.3b behaviour and intervention tools hard-code three wave slots and would raise `KeyError` on Slow Push
     (codes 46–60). They are left untouched for the historical study; the new `m44` package replaces them for the
     four-action game;
   * a checkpoint of one four-action preset loaded **silently** on another, because all three have 88 inputs and 61
     outputs although econ / aggro scale the absence entry by 5 instead of 3. `m44.common.load_policy` now refuses
     different shapes always and different configurations unless a cross-environment evaluation is requested;
   * the `rl-shared` CLI and the visual debugger did not offer the two calibrated presets; both now do.
3. **Strategic metrics with state-based definitions** (§4): wave management by decision state, freezes as persisting
   lane states (episodes, controller, end, denial), zoning by its consequences under a credible threat, vision /
   pressure / dives / tower shots, and plate phases read from the tower-damage state. Natural (s0) and initialized
   scenarios are never pooled.
4. **Paired counterfactuals** (§5) intervene on one decision component at saved states, with an explicit opponent
   protocol, common random numbers, a critic readout kept apart from measured returns, and confidence intervals over
   states. Illegal substitutions are refused, never invented.
5. **Smoke tests passed** (§6): the M4.4b matrix (3 environments × 3 seeds) at 1/10 budget trained in 93 s on 9
   processes; 81 / 81 pipeline checks; every final beats its initial network (+5.7 to +10.0 resources per game); the
   whole evaluation pipeline (references, cross-play, history, learning curves, behaviour, counterfactuals, viewer
   traces with replay verification) ran end to end in 40 s at small sizes.
6. **The instruments were demonstrated on non-trivial policies** (the M4.3c pilot finals, 2M transitions;
   exploratory). They reproduce the pilot's published freeze contrast (the econ policy Shoves 13–18% of the turns it
   holds a freeze, 52–60% when frozen on) and show things action counts cannot: the four-action policy switches from
   Pressure (95–98% of siege turns) to Vision (50–58%) once the enemy tower's plates are gone; Slow Push at a neutral
   lane produces a strength lead 4 turns later no more often than Last Hit; and only **16–24% of the pilots' dives are
   "busy" dives**, the only dives `aggro` changes (§6.4).
7. **Cost** (§8): M4.4b training ≈ 15 min wall clock on this 12-thread machine (2.0 CPU-hours for 18.4M transitions),
   evaluation ≈ 37 min on 9 workers, 112 MB of git-ignored checkpoints. No reduction of the design is needed.

**Recommendation: GO for M4.4b** as specified in `results/m44a/m44b_plan.md` (§10), after this review.

**Tested versus hypothetical.** Everything in §2, §3, §6 and §7 is tested (unit tests or recorded smoke runs).
The strategic readings in §6.4 are a *demonstration* on exploratory checkpoints, not M4.4 results. Whether PPO
discovers freezes, zoning or vision-before-pressure on the calibrated economies is the M4.4b question and remains
open.

---

## 2. PPO architecture and environment integration

### 2.1 What was inspected

| component | file | finding |
|---|---|---|
| network | `rl/network.py` | 88 → 128 → 128 (tanh) → 61 logits + value; illegal logits −10⁹ (exact zero probability); unchanged |
| PPO update, GAE | `rl/ppo.py` | clipped surrogate, clipped value loss, entropy bonus, single-choice steps excluded from the policy loss; GAE stops at game boundaries and bootstraps truncations; unchanged (audited in M4.3a against independent references) |
| shared self-play | `rl/shared.py` | one network, both seats' observations in one batched forward pass, both seats' transitions trained in current-policy games, rewards `StepResult.rewards[seat]` (r, −r), matchup fixed per game, snapshots between updates; unchanged |
| time semantics | `rl/train.py` (`GameTimes`), `rl/ppo.py` (`check_time_semantics`) | γ = the game's 0.99 enforced; 512-turn games with a seeded random first length; truncation bootstrapped from the pre-reset observation; unchanged |
| checkpoints | `rl/agent.py` | format 2 with SHA-256 weights digest verified on load; `NetAgent` refuses a 46-output network on a 61-code observation; extended by the M4.4 guard (§3.3) |

### 2.2 What is now tested on every four-action environment

`tests/test_m44a_readiness.py` (31 tests, about 5 s; no real training):

* **Shared policy.** On `four`, `econ` and `aggro`: both seats active and counted equally, no historical games, no
  terminations, rewards exactly antisymmetric, every recorded action legal under its mask and inside the 34
  ever-legal codes, seat B's observation equal (decoded and re-encoded) to seat A's view of the mirrored game, and the
  recorded log-probabilities and values equal to the network's at each seat's observation, so both seats act from the
  same parameters on their own observation within one simultaneous step.
* **Settings.** `m44.common.ppo_config` is identical across environments except the game; γ 0.99 = the game's,
  λ 0.95, segments 128, games 512, entropy 0.03, 16 environments; 4,096 transitions per update, 500 updates per run.
* **Reproducibility.** The same seed and environment give bit-identical weights; another environment gives other
  weights; a seed gives the same initial network in every environment (the paired initialization of §2.3); the
  checkpoint carries the exact configuration and the time semantics.

### 2.3 The M4.4 configuration selection

`lanegame/experiments/rl/m44/common.py` holds the single source of the comparison:

| key | preset | role |
|---|---|---|
| `four` | `GameConfig.m4_four_actions()` | control: the original four-action economy |
| `econ` | `GameConfig.m4_four_actions_econ()` | the calibrated economic candidate (M4.3c) |
| `aggro` | `GameConfig.m4_four_actions_econ_aggro()` | econ + tower retaliation on busy dives (M4.3d); the candidate for substantial training |

One `ppo_config(key, seed)` and one `shared_config()` serve all three: the M4.3b `ent03` condition (the settings of
the M4.3c pilot). Seeds 0, 1, 2 in every environment: since `train_shared` seeds the initialization with the run
seed, run (env, s) and run (env′, s) start from **bit-identical weights** (digests recorded in
`results/m44a/train_smoke.json`) and differ only by the game: a paired design across environments. `econ` and
`aggro` differ only in `tower_shot_p_busy` and `tower_adjacent_p_busy`; `four` and `econ` only in the five
`M4_FOUR_ECON_FIELDS` (tested).

The `rl-shared` command now accepts `--model m4_four_actions_econ` and `--model m4_four_actions_econ_aggro`.

---

## 3. Action and observation compatibility

### 3.1 The 61-code action space

Decoding goes only through the authoritative helpers (`config.champ_act`, `champ_wave`, `champ_behavior`,
`champ_objective`); `m44/actions.py` adds reporting conventions on top:

* the wave slots are not contiguous in the code space (codes 0–44 interleave Last Hit / Shove / Thin; Slow Push is
  46–60) and their slot numbers are not their order of damage, so every report uses the order
  **Last Hit, Thin, Slow Push, Shove** (least to most wave damage);
* **34 codes can ever be legal**: 4 waves × {Hold, Yield, Trade, Safe}; 4 waves × {Hold, Trade} × {Pressure, Vision};
  Recall (code 12); ABSENT (45). The other 27 codes (Yield / Safe with an objective, Recall with another wave or an
  objective) are masked in every state. Random play on each environment (`audit` stage, 1,200 turns each) met all 34
  ever-legal codes and nothing else; observations stayed in [0, 1], masks equalled the legal sets and every turn
  passed the decode-and-mirror check;
* whenever a champion is present, all four wave slots are legal with every legal behaviour / objective, which is what
  makes wave-only interventions always legal (tested over the legal sets met in play).

### 3.2 Observation

The 88-entry layout is the M4 one (`observation_m4`), identical in the three environments; only the absence entry's
scale differs (3 in `four`, 5 in `econ` / `aggro`). The full feature list with normalizations is in
[ppo_training_explained.md §1](../rl/ppo_training_explained.md#1-the-input-the-88-feature-observation).

### 3.3 The checkpoint guard

`m44.common.load_policy(path, cfg, cross_env=False)`:

| situation | result |
|---|---|
| digest mismatch (altered file) | refused (`load_checkpoint`) |
| different observation length or action count (e.g. a 46-output M4.3b network on the 61-code space) | **always refused**, even with `cross_env=True` |
| same shapes, different configuration (e.g. a `four` network on `econ`) | refused unless `cross_env=True`; the differing fields are attached (`agent.compat`) |
| same configuration | loads |

Old checkpoints keep working with their original environments (the M4.3b finals load on `m4_lasthit()`). The
compatibility matrix of every checkpoint on disk is in `results/m44a/report.md` §1.

---

## 4. Strategic metrics and their definitions

Every metric is computed from **game records** (`m44/records.py`): one row per turn and side with the decision state
at the start of the turn, the policy's probability masses by component, the action, and the outcome read from the
environment's `info`. A record of two agents is the same game `evaluate_discounted` plays on that seed (tested). All
positions are in the focal side's own coordinates (own tower −2, the lane cell 0, enemy tower +2). Rates are per
champion-turn of the focal side; "per game" values carry 95% intervals over (game, side).

**Natural versus initialized.** Records from s0 are "natural"; records started from a constructed state (the M4.3c
harness `freeze` start, or `neutral` with A at ×1.76 attack power) are "initialized" and labelled with the scenario.
`metrics.all_metrics` reports each start group separately; they are never pooled (tested).

### 4.1 Wave management (`metrics.wave_management`)

* **Frequencies** of Last Hit / Thin / Slow Push / Shove over present, non-Recall turns, per game (± 95%).
* **By decision state**: all turns; fronts in contact at the lane cell; my wave stronger / even / enemy stronger /
  enemy much stronger (window strength difference ≥ 1, < 1, ≤ −1, ≤ −4: the quantity Thin and Slow Push target,
  `mechanics.window_strength`); waves meeting on my half / the enemy half; holding a freeze; frozen on; my crash under
  the enemy tower; enemy crash under mine; enemy absent. Each row reports the chosen shares and the policy's mean
  probability masses.
* **Crashes**: count, size (small < 12 HP, medium 12–23, large ≥ 24), first crash turn, crashes per 100 turns.
* **Wave position over time**: where the waves meet (the contact cell, else the fronts' midpoint), by turn bin.
* **Wave advantage**: share of turns with my wave stronger / even / weaker.
* **Around a crash**: the crasher's and the defender's wave shares in the 5 turns before and after the crasher's crash.
* **Against a repeated pattern**: the focal side's wave shares after the opponent Thinned / Shoved / Last-Hit three
  turns in a row.
* **Intent versus effect** (an action label is not an outcome): at a neutral lane (fronts in contact at the lane cell),
  for each chosen wave action, the share of cases in which the meeting point moved toward the enemy and in which my
  window strength led by ≥ 1 **four turns later**. Selecting Slow Push and having created a slow push are different
  numbers.

### 4.2 Freezes (`metrics.freeze_episodes`, `metrics.freezes`)

* **Freeze state** F_s (start of a turn): both fronts alive and in contact, and the enemy front at side s's own −1 or
  deeper: s (the *freezer*) holds the enemy wave next to its own tower (reference §4.8; the M4.3c harness's `held_by`
  plus contact). It is a lane state; no action enters the definition.
* **Episode**: a maximal run of consecutive turns in the freeze state. **Established** when it lasts ≥ 4 turns (two
  waves arrive while it holds), **sustained** when ≥ 10.
* **Controller**: the freezer; its power class at the episode start (attack-power ratio ≥ 1.25 stronger, ≤ 0.8 weaker,
  else even), level difference and plate phase.
* **End** (from the turn after the last held turn): `crashed_in` (the held wave crashed into the freezer's tower: the
  frozen-on side pushed through), `held_wave_cleared` (enemy front gone without a crash), `pushed_out` (the fronts still
  fight but no longer at the freezer's −1), `own_wave_died`, `contact_lost`, or `censored` (the record ended first).
* **Economics**: gold and XP earned by each side during the episode (the freezer's lead), the frozen-on side's XP lost
  for good and available gold left uncollected, and the gold difference in the 20 turns after the end.
* **Useful denial**: an established episode in which the freezer out-earned the frozen-on side. Reported with the XP
  lost; never inferred from actions.
* **Preserving versus breaking**: the freezer's share of Last Hit or Thin while holding (an action pattern
  *consistent* with keeping the freeze, explicitly not evidence that it was kept: the episode length is), and whether
  the freezer or the frozen-on side Shoved on the last two held turns.

The four cases the brief separates are separate numbers: (1) natural freezes = episodes in s0 records; (2) a scripted
frozen start = the `initialized:freeze/equal` group; (3) an action that appears to preserve a freeze = the preserving
share; (4) a freeze that denies resources = the useful-denial share and the frozen-on side's losses.

### 4.3 Zoning and combat threat (`metrics.zoning`)

* **By power class** of the focal side (both champions present): mean cell, share in the own tower cell, trade attempt
  and connection rates, damage dealt and taken per connected trade (bars of max HP), collection rate (collected /
  available farm), deaths.
* **Credible threat** on side s: both present, the enemy stronger by attack power (my ratio ≤ 0.8) **and** by the
  HP-aware duel edge (`zoning_natural.duel_edge` < 0), s outside its own tower cell (where only a dive reaches it), and
  the enemy within catch distance (|pos + enemy pos| ≤ 3 = champion speed + trade range). Under threat versus the same
  weaker side's turns without threat: behaviour shares (does it Safe / Yield?), mean cell, gold left uncollected and XP
  lost per turn, death and Recall within 4 turns.
* **Price of applying pressure**: the opponent's uncollected gold when the focal side is stronger and near versus
  otherwise, and the focal side's own collection rate when it Trades versus when it Holds.
* **Deaths and forced Recalls**: deaths, Recalls at ≤ 40% HP and other Recalls, per 1000 champion-turns; deaths within
  3 turns of a connected trade.

### 4.4 Vision and tower pressure (`metrics.vision_tower`)

* Objective choice when Pressure / Vision is legal (own crash under the enemy tower, tower reachable), by power class,
  also restricted to turns without active vision.
* **Vision before extended pressure**: runs of ≥ 3 consecutive Pressure turns, and the share that started with vision
  active or established in the 2 turns before.
* **Forward exposure** (ending at own +1 or +2) with and without vision: gank probability, expected and realized gank
  damage, deaths.
* Ganks per 1000 turns (count, expected and realized damage, gank deaths), vision established, pressure turns, plates
  and the median turn of each plate.
* **Dives**: a connected trade ending in the enemy tower cell; **busy** when the diver's own crash is under that tower
  or crashes that turn (the resolver's condition, `champions_m4` step 9b). Busy and not-busy dives separately: tower
  shot probability, realized shots, the diver's death within 2 turns, damage dealt and taken; the **defender's** reply
  on the dive turn and the next (Hold / Yield / Trade / Safe / Recall / absent). Expected tower shots per 1000 turns,
  busy and not busy. This is where `econ` and `aggro` can differ: only busy dives (and busy tower-adjacent trades) have
  another shot probability.

### 4.5 Plate phases (`metrics.plate_phases`)

From the simulation state, never a turn number: the focal side's **attack phase** (plates left on the enemy tower > 0,
or exhausted) and **defensive phase** (plates left on its own tower), plus the turn its attack plates ran out. Per
phase: wave shares, Pressure and Vision when legal, crash rate, freeze-state share, farm / plate / bounty / total income
per turn. "Shove when it pays little" is the Shove share and crash rate in the exhausted phase. Disabling plate gold
(the M4.3d diagnostic) is a different game; these phases are reached in play.

---

## 5. Counterfactual methodology

`m44/counterfactual.py` asks, at states the policy actually reaches: *would a different legal choice of one decision
component have produced a better discounted economic outcome, under stated continuation policies?*

| step | rule |
|---|---|
| state | a `diagnostics.Snapshot` taken from a recorded game (restorable exactly: tested), focal seat where the situation holds, turn ≥ 8 |
| comparisons | `freeze_held`: Thin vs Shove (wave only); `neutral`: Slow Push vs Last Hit; `oversized_enemy` (4 or more behind in window strength): Shove vs Thin; `siege`: Vision vs Pressure (objective only); `threatened` (credible threat): Trade vs Safe (behaviour only); each with a third alternative and the policy's own action |
| intervention | `interventions.ComponentPlan`: one component forced on the branch turn (or for N turns), the policy's other decisions kept. Wave: "marginal" (component replacement) or "conditional" (the policy's conditional given that wave), Recall kept at the policy's probability; behaviour and objective: "conditional" only (a marginal rule would need illegal codes such as Safe + Pressure) |
| legality | a candidate exists only where `interventions.possible` holds; a state where either side of the contrast is impossible is skipped and counted; an impossible value on the branch turn raises, on a later turn of a multi-turn window it leaves the distribution unchanged and is counted (`skipped_turns`). No illegal combination is ever played (tested over every legal set met in play) |
| opponent | simultaneous moves: the opponent's branch-turn action never depends on the focal choice. Protocol **policy** (default): it samples its own policy from its own observation with its own stream, identical across candidates (tested). Protocol **modal**: it commits to its own most probable action on the branch turn (new `opponent_plan` hook of `diagnostics.branch`, backwards compatible) |
| continuation | both seats follow designated continuation agents after the branch turn (default: the policy itself in both seats) |
| pairing | rollout m of every candidate uses the same environment, focal and opponent streams (`rollout_seed(root, m, key)`), one uniform per seat per turn; identical plans give identical returns (tested) |
| value | the focal seat's discounted return over a window (300 turns in M4.4b, the M4.3b-validated window; 60–150 in the M4.4a demonstrations), γ = 0.99 |

Reported separately for every state and aggregated over states:

* **frequency**: the policy's probability mass on each candidate's component value;
* **critic prediction**: V̂(s), and a one-step prediction per candidate E[r₀ + γ V̂(s₁)] on the same paired streams
  (the critic's own belief, an approximation), with how often its sign agrees with the measurement;
* **measured return**: Monte Carlo means, paired differences against the policy and the contrast with standard errors;
  across states the mean contrast with a 95% interval (states as units) and the shares of states with a significantly
  positive / negative contrast.

**Limitation, stated as the method's definition.** A one-step intervention followed by policy continuation measures
the value of that intervention *under the specified continuation policies against the specified opponent*. It is
not a best response, an equilibrium action or proof of optimal strategy; a positive contrast says the alternative
would have paid here, against this play.

---

## 6. Smoke-test results

### 6.1 Training (`smoke_train`: the M4.4b matrix at 204,800 transitions)

| | four | econ | aggro |
|---|---|---|---|
| per-process throughput (9 concurrent runs) | 2,702–2,801 /s | 2,450–2,472 /s | 2,438–2,463 /s |
| entropy start → last 10 updates | 2.85 → 1.62–1.71 | 2.85 → 2.25–2.38 | 2.85 → 2.29–2.35 |
| explained variance (last 10) | 0.62–0.64 | 0.59–0.63 | 0.59–0.66 |
| mean approximate KL; max pre-clip gradient norm | 0.004; ≤ 1.25 | 0.003; ≤ 0.96 | 0.003; ≤ 0.82 |
| max \|V̂\| (resources) vs return bound Ω | 13.9–15.5 vs 107 | 14.2–14.4 vs 145 | 13.5–14.0 vs 145 |
| V̂(s0) (a symmetric zero-sum state: true value 0) | +0.24 to +0.39 | −0.46 to −0.07 | −0.11 to +0.09 |

All **81 pipeline checks** pass (finite losses and gradients; never terminated, one truncation per finished game;
critic within the return bound; gradient norms of order 1; KL small; both seats trained equally; no historical
games; budget exact; both seats see the same vector at s0) and all 27 informational learning checks (|V̂(s0)| ≤ 0.5,
explained variance positive, entropy decreasing).

### 6.2 Evaluation pipeline (`smoke_eval`, tiny sizes: plumbing, not results)

* **Loading**: every final's digest verified and equal to the training record; every final refused by the other two
  environments without the cross-environment opt-in.
* **Strength**: every final beats its initial network by +5.7 to +10.0 resources per game and its 28,672-transition
  snapshot by +5.2 to +8.1 (paired seats, 2 seeds × W 200: indicative only). Against the references, the `four`
  finals already beat `farm4` and `freeze_hunt4` by ~+10 but lose to `shove4` by 2–3; the econ / aggro finals beat
  `farm4` by +0.8 to +4.2. (The M4.3c pilot finals, 10× the budget, beat `shove4` by +3 to +4: the budget matters.)
* **Behaviour**: every metric family produced, for natural self-play, games against `farm4` / `shove4`, and both
  initialized scenarios.
* **Counterfactuals**: 45 (final, situation) jobs, every state found, no state skipped, opponent's first action
  identical across candidates in every state.
* **Traces**: six recordings (self-play and against `shove4`, one per environment) exported and replayed through the
  viewer's own `session.play` with identical actions and payoffs on every turn.

### 6.3 Viewer integration

`lanegame/viewer/session.py` now lists `m4_four_actions_econ` and `m4_four_actions_econ_aggro` (start states from
`m4_lasthit()` as for `m4_four_actions`). Checkpoint → trace: `python -m lanegame.experiments.rl.m44 smoke_eval` (or
`m44.trace.export`) writes `results/m44a/trajectories/*.json`; open them with `streamlit run lanegame/viewer/app.py` →
"Recorded trajectory" → folder `results/m44a/trajectories`. `m44.trace.verify` checks the round trip.

### 6.4 Demonstration on the M4.3c pilot finals (exploratory checkpoints)

To check that the instruments separate strategy from action frequency on policies that have learned something, the
`demo` stage ran them on the four M4.3c pilot finals (`pilot:base:0/1` = `m4_four_actions()`, `pilot:candidate:0/1`
= `m4_four_actions_econ()`; 2,048,000 transitions each): 16 self-play games × 500 turns, 4 games against `farm4`
and `shove4`, 8 games from each initialized scenario, and 6 states × 8 paired rollouts per counterfactual situation.
Selected readings (`results/m44a/report.md` §4):

| instrument | base (four) | candidate (econ) |
|---|---|---|
| Shove share, all turns | 71–76% | 34–38% |
| Shove while **holding** a freeze / while **frozen on** | 76–90% / 68–81% | 13–18% / 52–60% |
| established freezes per 1000 champion-turns; held share | 0.2–1.4; 4–5% | 8.1–9.5; 10–11% |
| freezer Shoved on the last two held turns | 82–100% | 32–43% |
| useful denial (freezer out-earned the frozen-on side) | 86–100% of 25 episodes | 48–49% of 281 episodes |
| Pressure / Vision on siege turns, enemy plates **left** | 95–98% / 2–4% | 52–66% / 5–25% |
| Pressure / Vision on siege turns, enemy plates **exhausted** | 40–45% / 50–58% | 27–28% / 19–26% |
| Shove share with plates left / exhausted | 80–85% / 66–71% | 32–35% / 34–39% |
| dives per 1000 champion-turns; share that are **busy** | 54–58; 22–24% | 38; 16–23% |
| defender's reply on the dive turn | Hold 36–44%, Yield 26–34%, Trade 12–23% | Recall 37–43%, Safe 10–31%, Trade 15–25% |
| under a credible threat (initialized ×1.76 gap): Safe | 3–4% | 23–24% |
| Slow Push at a neutral lane → my strength lead 4 turns later | 32–49% (Last Hit 32–37%) | 44–47% (Last Hit 43–45%) |

What this shows about the **instruments**:

* The four-action freeze contrast reproduces the M4.3c pilot report's numbers (Shove 13–18% holding, 49–54% frozen
  on) under the new state-based definition, so the definitions agree with the earlier, independently written
  counters.
* Phase-dependent behaviour is visible only because phases come from the tower state: the base policy uses Vision on
  more than half of its siege turns after the plates fall, and almost never before.
* Action labels and outcomes come apart: Slow Push at a neutral lane does not create a strength lead more often than
  Last Hit four turns later (the opponent's simultaneous actions decide), so "chose Slow Push" must never be read as
  "slow-pushed".
* **For the M4.4b design**: in learned play most dives are *not* busy (76–84%), unlike the scripted M4.3d dives. The
  `aggro` change acts only on the busy minority (6–14 per 1000 champion-turns in these policies). M4.4b's sample
  (32 games × 1,000 turns × 2 seats per final) contains roughly 400–900 busy dives per final, enough to measure the
  rate shift, but a small overall econ–aggro difference would be the expected outcome, not a failure.
* Power gaps are rare in natural self-play (0.4–0.9% of the turns with both champions present), so zoning must be
  read from the initialized power-gap scenario, which is reported separately.
* The 12 counterfactual cells (6 states each) give contrasts within ±0.6 resources with intervals covering zero, and
  a critic whose one-step sign agrees with the measurement in 17–100% of states: the machinery works; six states per
  cell decide nothing. M4.4b uses 8 states × 16 rollouts × 300 turns per cell.

None of this is a claim about learned strategy on `aggro`; it is the M4.4b question.

---

## 7. Reproducibility and checkpoint verification

| check | result |
|---|---|
| re-training `four_s0`, `econ_s0`, `aggro_s0` (`smoke_repro`) | identical weights digests and identical histories (every field except wall-clock) |
| initial networks of seeds 0 / 1 / 2 across the three environments | identical digests per seed |
| final checkpoints | SHA-256 digest verified on load, equal to the training record |
| cross-environment loads | refused without the opt-in; 46-output networks refused always |
| records versus the evaluator | a recorded game equals `evaluate_discounted`'s game on the same seed (rewards, turn by turn) |
| network sampling in records | identical to `NetAgent.act` (one uniform, inverse CDF) |
| viewer traces | replay to identical actions and payoffs |
| evaluation seeds | SeedSequence roots 80,000–80,999 (disjoint from every earlier study's 10,000–79,999) |

Checkpoints of M4.4a live in the git-ignored `results/rl/checkpoints/m44/smoke/` (9 runs + 3 repeats, 17 MB);
nothing existing was moved or overwritten.

---

## 8. Performance and expected training cost

Measured on this machine (12 logical CPUs, Windows, Python 3.13, torch 2.14 CPU, one thread per process):

| quantity | value |
|---|---|
| training throughput, 9 concurrent runs | 2,438–2,801 agent transitions/s per process (≈ 20,000 /s aggregate) |
| one M4.4b run (2,048,000 transitions) | 12.2–14.0 min |
| **M4.4b training, 9 runs in parallel** | **≈ 15 min wall clock, 2.0 CPU-hours, 18,432,000 transitions** |
| M4.4b evaluation (scaled from the smoke timings) | ≈ 5.6 CPU-hours, **≈ 37 min on 9 workers** (learning curves 144 CPU-min, counterfactuals 87, references 62, behaviour 27, history 9, cross-play 8) |
| checkpoints | 81 per run × 153 KB ≈ 112 MB in total (git-ignored) |

The planned budget is cheap; no change to the controlled design is needed. If longer runs are wanted after M4.4b,
4× the budget would still be about an hour of training.

---

## 9. Known limitations

1. **Pilot readings are exploratory.** §6.4 uses checkpoints trained before M4.4 at two seeds; it validates the
   instruments, not the economies.
2. **Metric thresholds are conventions.** Established freeze ≥ 4 turns, power classes 1.25 / 0.8, catch distance 3,
   the 4-turn effect horizon and the 20-turn after-window are fixed in `metrics.py` and reported with every table; they
   are not tuned to make results look better and can be varied in M4.4c.
3. **Freeze definition and contact geometry.** A freeze is "the enemy wave held at my −1 with both fronts fighting".
   Because the lane has five cells and a defender in its tower cell can only be reached by a dive (M4.3d §9), a freeze
   denies through the frozen-on side's own choices (walking up, stepping back) rather than through harassment from
   outside the tower; the instruments measure that, they do not change it.
4. **Credible threat** uses a static power ratio and the duel edge at the start of the turn; it does not model what
   the stronger side intends. It identifies situations; the consequences (gold left, deaths) are the evidence.
5. **Counterfactual scope** (§5): values under the given continuation policies; one-step by default; the critic
   readout is an approximation; the modal-opponent protocol is a second protocol, not a worst case.
6. **Natural power gaps are rare** (under 1% of self-play turns), so zoning evidence comes from initialized scenarios
   that the policy never trained on.
7. **Three seeds** give a starting comparison (t with 2 degrees of freedom, 4.30), not a convergence guarantee.
8. **Historical M4.3b modules** (`experiments/rl/m43b/behavior.py`, `intervene.py`) still assume three wave slots by
   design; they belong to the M4.3b record. (Their docstring's "1,000 updates of 16 × 128 × 2" for 2,048,000
   transitions should read 500 updates; left as written.)

---

## 10. Recommendation for M4.4b

**GO**, with the configuration of `results/m44a/m44b_plan.md`:

* 3 environments × seeds 0, 1, 2 × 2,048,000 agent transitions, identical settings, shared-policy self-play (100%
  current-policy), snapshots every 25,600 transitions;
  `python -m lanegame.experiments.rl.m44 train_main --confirm-m44b --workers 9` (≈ 15 min), then
  `python -m lanegame.experiments.rl.m44 eval_main --workers 9` (≈ 37 min);
* read, in this order: PPO health and learning curves (has each run plateaued against fixed references and its own
  snapshots?), seed-to-seed spread within each environment, then the behaviour diagnostics by environment (econ vs
  aggro on busy dives, defender replies, Vision before pressure, deaths; four vs econ / aggro on Shove by state and
  plate phase, freeze episodes and denial, concession under a credible threat), then the counterfactual contrasts;
* treat any between-environment difference as real only when it exceeds the within-environment seed spread.

Conditions and risks: the busy-dive contrast is narrow (§6.4); if policies have not plateaued at 2M, extend the
budget for all environments equally rather than reading unconverged policies; if freezes or zoning do not appear on
the calibrated economies, the counterfactuals decide whether that is the game's incentive (outcome C of M4.3b) or a
training gap.

---

## 11. Reproduction and files

```bash
python -m lanegame.experiments.rl.m44 audit smoke_train smoke_repro smoke_eval demo plan report --workers 9
                                                         # every M4.4a stage, about 5 minutes; results/m44a/
python -m pytest tests/test_m44a_readiness.py            # 31 tests, about 5 s
```

| file | role |
|---|---|
| `lanegame/experiments/rl/m44/` (`common`, `actions`, `records`, `metrics`, `references`, `interventions`, `counterfactual`, `train`, `evaluate`, `trace`, `stages`, `plan`, `report`, `__main__`) | the M4.4 package (new) |
| `lanegame/diagnostics/branch.py` | `opponent_plan` hook in `rollout` / `branch` (backwards compatible; default behaviour unchanged) |
| `lanegame/viewer/session.py` | the two calibrated presets in `CONFIGS` / `START_SOURCE` |
| `lanegame/__main__.py` | `rl-shared --model` accepts the two calibrated presets |
| `tests/test_m44a_readiness.py` | 31 tests (new) |
| `results/m44a/` | `audit.json`, `train_smoke.json`, `eval_smoke.json.gz`, `demo.json`, `m44b_plan.{json,md}`, `report.md`, `trajectories/`, logs, `regression.log` |
| `docs/m4/m4_4a_ppo_readiness.md`, `docs/rl/ppo_training_explained.md` | this report; the PPO pipeline explained |
| `docs/rl_research_roadmap.md`, `docs/README.md`, `README.md`, `docs/m4/laning_game_mathematical_reference.md` | roadmap restructured (M1–M6, M4.1–M4.4d), indexes, a training pointer (§9.2, §13 log) |
