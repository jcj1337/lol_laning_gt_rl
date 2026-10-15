# M4.5c — Shared-policy PPO self-play: pilot, training plan and evaluation readiness

*2026-10-14. Branch `main` on 8dc5acf with the uncommitted M4.5b work; nothing staged or committed. **Pilot only: one
seed, 1,024,000 agent transitions. The full multi-seed campaign has NOT been started and awaits approval of the budget
in §8.** Environment `GameConfig.m45_calibrated()`, unchanged. Code: `lanegame/experiments/rl/m45c/` (new) and two
opt-in arguments of `lanegame.rl.train_shared` (§1). Generated tables: [`results/m45c/report.md`](../../results/m45c/report.md)
(every number below that is not computed in the text comes from there); raw results `results/m45c/*.json`; figures
`results/m45c/figures/`; checkpoints `results/rl/checkpoints/m45c/` (git-ignored). Tests: `tests/test_m45c_training.py`
(new, 16).*

**Transition convention (used throughout).** A *joint step* is one simultaneous move of both seats in one
environment. An *agent transition* is one seat's (observation, action, reward) played by the learning network. In
100% current-policy self-play every joint step yields exactly 2 agent transitions, all of which are *PPO samples*:
the value loss uses all of them, and the policy and entropy terms use the 88–92% with more than one legal action.
`PPOConfig.total_steps`, the trainer's `agent_transitions` and every "transitions" figure of M4.4 and M4.5c are
**agent transitions**: M4.4b's "2,048,000 transitions" were 1,024,000 joint steps. One update = 16 environments ×
128 joint steps × 2 seats = 4,096 agent transitions = 64 gradient steps (4 epochs × 16 minibatches of 256).

---

## 0. Summary

**The pipeline is correct and the pilot learned steadily.** The audit checks pass on the calibrated preset, and so do
all 14 smoke checks and all 12 pilot checks. Training is bit-identical on re-run, and a fresh re-run of the pilot's
first 7 updates reproduces its first snapshot bit for bit.

**Pilot run.** Seed 100, 1,024,000 agent transitions, 4.0 minutes in one process, at **4,277 agent transitions/s**
end to end. The learning-rate schedule is the one of a 4,096,000-transition run (M4.4b's extension budget); the run
stops after the first quarter.

**What it learned.**
- Against nine fixed M4.5b strategies, its discounted lead went from **−7.4** (untrained) to **+2.5** resources
  (W = 460, scripted mean). At W = 1000 the final beats every one of them: +1.0 (passive farming) to +6.1 (always-Safe).
- It beats the out-of-distribution M4.4b network by **+0.67 ± 0.48**.
- It is **still improving at the end**: the final beats its own 0.77M snapshot by +0.67 ± 0.46, and the 0.512M
  snapshot against 9 of 10 opponents.
- Its behaviour is state-conditioned, for example:
  - Recall 31% and Safe 65% at low HP, against 1% and 6% at high HP;
  - Pressure 49% when legal;
  - Trade 39% while holding a freeze, against 12% when frozen on.
- Shove has risen to 32% of decisions and is still drifting. Ward is rare (≈ 1%).

**How it compares with M4.4.** The entropy trajectory matches M4.4b's aggro runs on the same schedule (1.77 at 1.02M
against 1.69–1.77). Strength against fixed opponents rises more slowly than in M4.4b, which plateaued against its own
references by ≈ 0.3M.

**Cost.**
- Environment stepping is 70% of training time, network inference 2%, PPO updates 18%.
- Concurrent processes slow each other: per-process speed is 0.83× the single-process speed with 3 processes and
  0.58× with 11.
- A process whose lane caches are full uses about **1.5 GB**, so 11 full-cache runs would not fit in the 17 GB
  machine. A smaller cache costs about 4% speed and is tested to give identical results.

**Recommendation (§8): 6 seeds × 8,192,000 agent transitions**, with the learning rate annealed over each run (the
M4.4b protocol), 100% current-policy self-play, and a pre-registered midpoint (4.096M) comparison as the extension
criterion. Estimated cost (extrapolated): ≈ 46 min of training on 6 concurrent processes, ≈ 1.4 h of standard
evaluation, ≈ 0.4 h of counterfactual battery. Alternatives: 3 × 4,096,000 (≈ 1 h in all) and 10 × 16,384,000
(≈ 5 h).

**Readiness (§10): GO.** No correctness or stability problem is open. The known limitations (§10) are matters for the
evaluation, not preconditions for training.

---

## 1. Training infrastructure

### 1.1 Reused, unchanged

| component | source | role in M4.5c |
|---|---|---|
| shared-policy PPO | `lanegame.rl.train_shared`, `SharedCollector`, `ppo_update`, `compute_gae`, `MaskedCategorical`, `PolicyValueNet` | the trainer: one network plays both seats from its own perspective; both seats' transitions are PPO samples; masked categorical policy; clipped surrogate, clipped value loss, advantage normalization over choice samples, entropy over choice samples, pre-clip gradient norm logged |
| continuing time | `PPOConfig.continuing` | γ = the game's 0.99, λ 0.95, 128-turn segments, 512-turn bootstrapped games with a staggered first game, reward scale 0.1, value clip 0.5 |
| hyperparameters | the M4.4b settings (M4.3b `ent03`) | identical: entropy 0.03, Adam 3e-4 annealed to 0, 4 epochs, minibatch 256, clip 0.2, max grad norm 0.5, 16 environments, 2 × 128 tanh MLP |
| checkpoint guard | `m44.common.load_policy` | refuses other shapes always, and other configurations of the same shape unless `cross_env=True` |
| evaluator | `env.discounted.evaluate_discounted` | protocol E1, paired seats on one chance stream, guaranteed tail bound |
| behaviour records and metrics | `m44.records.play_records`, `m44.metrics.freezes / recalls` | per-turn decision state, probability masses and outcomes; freeze episodes; Recall economics |
| fixed opponents | M4.5b contingent strategies (`m45b.strategies`) via `m43c.strategies.StrategyAgent` | evaluation only |
| M4.5 ledger and counterfactual engine | `m45b.engine` (play, collect, branch, Then, Fixed), `m45b.reference.NetPolicy` | wards placed / interrupted, minion fire by cause, out-of-reach farm; paired multi-turn counterfactuals with learned continuations |

The 90-input network is **trained from scratch**: `torch.manual_seed(seed)`; the initial digest is recorded. No M4.4
weights are transferred. M4.4 finals are refused by the guard (shape), and they enter evaluation only through the
explicitly labelled out-of-distribution adapter (§1.3).

### 1.2 Changed (small, opt-in, tested)

Two keyword arguments of `train_shared`. Both default to `None`, which keeps the earlier behaviour bit for bit; the
existing trainer tests pass unchanged.

- **`stop_at`** ends a run once that many agent transitions are reached, while the learning rate keeps annealing
  over `total_steps`. The run is then an exact **prefix** of the full-schedule run. The test trains a 3-update prefix
  of a 9-update schedule: its weights equal the full run's snapshot at that point, its history equals the full run's
  first entries, and its learning rate has not reached zero. Why it was needed: with the learning rate annealed to
  zero over the pilot's own budget, the end of any pilot looks stable by construction (M4.4b §3). The prefix design
  keeps the pilot's end informative, and it makes the pilot directly comparable to M4.4b's extension runs, which used
  the same schedule.
- **`rollout_stats(roll, finished)`** merges per-rollout diagnostics into history.jsonl. It is called after the
  update and only reads. Those diagnostics are: action-category shares, legal-mask statistics, illegal-sample and
  never-legal-offer counters, per-seat summed rewards and advantage statistics, and finished-game outcomes. The test
  trains with and without it and gets identical weights and identical history.

Nothing else in `lanegame/rl`, the environment, the preset or any earlier experiment changed. The math reference
needs no update, because no rule, reward or objective changed.

### 1.3 New: `lanegame/experiments/rl/m45c/`

| module | content |
|---|---|
| `common` | the frozen preset, settings, transition accounting, paths, seeds, agent specifications, the `OODNetAgent` adapter (drops the two W entries; labelled OUT OF DISTRIBUTION) |
| `actions` | the M4.5 action space: 27 ever-legal codes; **Ward is its own category**, because the M4.4 decomposition reads code 30 as "Last Hit + Hold + Vision" |
| `references` | the fixed suite (§4) |
| `train` | `RolloutStats`, smoke, pilot, prefix check, throughput profile, concurrency scaling, lane-cache probe |
| `evaluate`, `behavior`, `stages` | learning curves, headline, history; state-conditioned tables, freezes, Recalls, the M4.5b ledger |
| `strategic` | the counterfactual framework of the full evaluation (§9), with a smoke run on the pilot final |
| `plan`, `report` | runtime estimates from the measurements; `results/m45c/report.md` and figures |
| `__main__` | the stages; `train_full` refuses to run without `--confirm-m45c-full`, which is reserved for after the budget approval |

### 1.4 Verified

| property | how |
|---|---|
| shared self-play: both seats from one network, own-perspective observations, both selected before the step | collector code; test: seat B's observation decodes as the mirror of seat A's at every step |
| attribution: each seat gets its own reward, exactly antisymmetric | test plus pilot counter: Σ reward A + Σ reward B = 0 in every one of 250 rollouts |
| both seats contribute equally and carry gradient | pilot: 512,000 transitions per seat exactly; test: the policy gradient of seat-A-only and seat-B-only batches are both nonzero; per-seat advantage std 0.11 / 0.11 |
| masks: illegal actions never sampled; Ward offered; never-legal codes never offered | test; pilot counters: 0 illegal samples and 0 never-legal offers in 1,024,000 transitions; Ward legal in 57–63% of decisions |
| log-probabilities = the masked sampling distribution | test against an independent float64 softmax over the legal logits |
| PPO uses the recorded masks; single-choice steps carry no policy gradient | the batch carries the rollout masks; existing reference tests (`test_rl_reference.py`) |
| no invalid-action fallback | `LaneEnv.step` raises on an illegal code; none raised |
| GAE, value targets, termination / truncation | test on M4.5: truncations exactly at the staggered game limits, bootstrap = the critic's value of the true pre-reset observation for both seats, no λ leakage; pilot: 0 terminations, 1 truncation per finished game (1,000 games); existing references |
| advantage normalization, clipping, entropy, gradient clipping | existing reference tests (loss and gradient equal an independent implementation) |
| reproducibility | smoke re-run bit-identical; pilot prefix bit-identical; lane-cache size never changes results (test: cache 0 / 7 / 2,000) |
| frozen environment | audit: equals `m45_calibrated()`; last_front, delay 0, growth 1.08, xp_range 2, Ward, α 50 retained |

Historical opponent snapshots are taken every 25,600 transitions (40 in the pilot) and used as **evaluation**
opponents (history, §4.3). They are not training opponents: 100% current-policy self-play is the M4.4b baseline, and
M3 / M4.3b found the 80/20 mixture bought nothing.

---

## 2. Smoke-test correctness

Ten updates (40,960 transitions) on their own schedule, plus a bit-identical re-run. All 14 checks pass: finite
statistics, no termination, |V| ≤ Ω, gradient norms, KL, equal seats, 2 transitions per joint step, exact budget, no
illegal sample, no never-legal offer, exact reward antisymmetry, V_A(s0) = V_B(s0), identical weights and identical
histories. The smoke evaluation also passes:
- the guard loads the network on `m45_calibrated`, and refuses it on `m45_unified` (configuration) and on M4.4's
  aggro (shape);
- self-play through the paired evaluator is exactly 0;
- every instrument produced output.

**Issues found and resolved**:

1. *Snapshot naming.* `train_shared` snapshots at the first update that reaches each multiple of 25,600 (28,672,
   53,248, …), so two of the initial curve points named nonexistent snapshots. The evaluation now uses
   `common.snapshot_schedule()`, and the test checks every evaluation point against it. This was an evaluation-plan
   error, not a trainer defect.
2. *Ward in the M4.4 decomposition.* The M4.4 decomposition reads code 30 as Last Hit + Hold + Vision. M4.5c
   classifies codes with `actions.category` and uses P(Ward) = the Vision mass. That mass is exact on this preset,
   because code 30 is its only Vision code (tested on random-play observations).
3. *Records near W.* The M4.4 records' `epos` / `dist` treat W (cell 99) as a lane cell. No M4.5c table uses them,
   and "at W" is read from `pos`.
4. *Recall counters.* `recalled` counts completed Recalls and `recall_interrupted` the interrupted ones. Interrupted
   shares below are therefore interrupted / (completed + interrupted).
5. *Memory.* The per-process memory growth is the lane-step cache (§7), not a leak.

No simulator defect and no trainer defect was found.

---

## 3. Pilot configuration

| setting | value |
|---|---|
| environment | `GameConfig.m45_calibrated()` (90 observation entries, 61 codes, 27 ever legal) |
| network | 90 → 128 → 128 (tanh) → 61 masked logits + value head; from scratch, seed 100 (initial digest `09a4f6c6…`) |
| seeds | training seed 100 (torch init; collector streams seed + 1); deliberately outside the campaign seeds 0–9 |
| schedule | Adam 3e-4 annealed linearly to 0 over 4,096,000 transitions, **stopped at 1,024,000** (final learning rate 2.25e-4) |
| PPO | clip 0.2, value clip 0.5, vf coef 0.5, entropy 0.03, 4 epochs, minibatch 256, max grad norm 0.5, reward scale 0.1, γ 0.99, λ 0.95 |
| rollout | 16 environments × 128-turn segments × 2 seats = 4,096 transitions per update; 250 updates |
| games | 512 turns, truncated and bootstrapped; 1,000 games finished |
| opponents | 100% current-policy self-play; 40 snapshots (evaluation only) |
| budget | 1,024,000 agent transitions = 512,000 joint steps |
| evaluation seeds | roots 92,000–92,999; 93,000–94,999 reserved for the full campaign (unused) |

**Why this budget.** The brief suggested about 250,000–500,000 environment transitions. Read as joint steps, the
pilot sits at the top of that range: 512,000 joint steps, which is 1,024,000 agent transitions. Three reasons:

1. M4.4b's strength against fixed references flattened at about 1M agent transitions. A pilot that reaches that point
   can tell whether M4.5 learns faster or slower.
2. At the throughput measured in the smoke run, the pilot cost 4 minutes.
3. The 25,600-transition snapshots give 40 points on the curve.

---

## 4. Learning results

### 4.1 Fixed-opponent suite

Nine M4.5b strategies (the brief's categories), plus one learned out-of-distribution reference:

| opponent | category |
|---|---|
| g_farm | passive farming |
| g_shove | plain Shove |
| g_shove_sr | Shove with Recall (the s0 equilibrium of M4.5b §7.4) |
| wk_shove_reset_sh | Shove with a frozen-on reset (the r2 s0 equilibrium) |
| g_shove_edge | Shove, selective trades (the asymmetric-start equilibria) |
| g_bully | aggressive Trading |
| g_safe | defensive Safe |
| fz_guard_sh | freeze-oriented (guarded freeze) |
| g_shove_ward | Ward-oriented Shove |
| M4.4b aggro final, seed 0 | OOD through the W-dropping adapter |

### 4.2 Learning curve

Snapshots against the suite: 32 paired seeds, both seats, W = 460.

| checkpoint | scripted mean | g_shove | g_shove_sr | g_bully | g_safe | fz_guard_sh | g_shove_ward | M4.4b (OOD) |
|---|---|---|---|---|---|---|---|---|
| untrained | −7.40 | −6.31 | −6.83 | −9.37 | −6.04 | −7.35 | −6.63 | −11.16 |
| 0.10M | −2.88 | −3.01 | −3.31 | −3.02 | +1.68 | −3.30 | −3.52 | −8.59 |
| 0.20M | −0.15 | −0.62 | −0.55 | +1.89 | +2.70 | −0.63 | −1.13 | −3.70 |
| 0.41M | +1.53 | +0.57 | +0.03 | +5.40 | +4.33 | +0.70 | −0.25 | −1.61 |
| 0.61M | +1.25 | +0.15 | −0.12 | +4.30 | +5.57 | +0.20 | −0.29 | +0.23 |
| 0.82M | +2.13 | +1.50 | +0.84 | +4.03 | +5.66 | +1.74 | +0.20 | +0.09 |
| 0.92M | +2.37 | +1.49 | +1.38 | +3.93 | +6.05 | +1.00 | +1.68 | +1.09 |
| 1.02M (final) | **+2.47** | +1.77 | +1.12 | +5.62 | +6.18 | +1.98 | +0.16 | +0.33 |

(± 0.3–0.9 per cell; full table and figure in `results/m45c/report.md` §6.) The curve has three phases:
- a fast phase up to 0.4M, in which the policy stops losing to everything;
- a slower phase from 0.4M to 1.0M (+1.5 to +2.5 on the scripted mean);
- no plateau yet: the last three points rise by +0.1 to +0.2 per 0.1M.

The OOD M4.4b network goes from −11.2 to about even.

### 4.3 Headline and history

**Headline** (64 paired seeds, W = 1000, tail bound 0.006):

| opponent | final | seat A / seat B | half-way (0.512M) | final − half-way |
|---|---|---|---|---|
| g_farm | +1.02 ± 0.46 | +0.94 / +1.09 | −0.71 | +1.73 ± 0.62 |
| g_shove | +1.60 ± 0.49 | +1.37 / +1.83 | +0.67 | +0.93 ± 0.65 |
| g_shove_sr | +1.58 ± 0.51 | +1.34 / +1.81 | +0.85 | +0.72 ± 0.71 |
| wk_shove_reset_sh | +1.68 ± 0.50 | +1.36 / +2.01 | +0.83 | +0.85 ± 0.68 |
| g_shove_edge | +3.10 ± 0.55 | +3.21 / +2.99 | +1.53 | +1.57 ± 0.72 |
| g_bully | +5.73 ± 0.58 | +5.60 / +5.86 | +4.09 | +1.64 ± 0.74 |
| g_safe | +6.12 ± 0.33 | +6.00 / +6.24 | +4.79 | +1.33 ± 0.47 |
| fz_guard_sh | +1.82 ± 0.49 | +1.51 / +2.14 | +0.80 | +1.03 ± 0.62 |
| g_shove_ward | +1.21 ± 0.33 | +1.38 / +1.04 | −0.31 | +1.52 ± 0.50 |
| M4.4b aggro s0 (OOD) | +0.67 ± 0.48 | +0.45 / +0.88 | −0.83 | +1.49 ± 0.60 |

**History** (the final against its own past; 32 paired seeds, W = 460): against the untrained network +9.99,
0.029M +9.69, 0.10M +6.91, 0.26M +2.29, 0.51M +1.54, **0.77M +0.67 ± 0.46**. The values fall monotonically and stay
positive, with no sign of cycling.

**What improved.**
- Against every fixed opponent from both seats; seat A and seat B agree within about 0.5.
- Against the policy's own past.
- The improvement between the half-way snapshot and the final is significant against 9 of 10 opponents
  (g_shove_sr: +0.72 ± 0.71).

**What did not improve, or is uncertain.**
- On the W = 460 curve, the value against g_shove_ward fell between the 0.92M and 1.02M points (+1.68 → +0.16). The
  W = 1000 headline (+1.21 ± 0.33, 64 seeds) shows this as noise between neighbouring snapshots of about ±1 on 32
  seeds.
- One seed says nothing about seed-to-seed spread. M4.4b found a seed SD of 0.2–0.9.
- The scripted suite is beaten everywhere, so from here on it can only show the policy's margin, not its
  weaknesses. Cross-seed play, historical snapshots and best-response attackers must carry the full campaign's
  strength claims.

**Checkpoint selection.** The pre-registered rule is the **final** checkpoint. It is also the best evaluated
checkpoint on the scripted mean, so the rule and the best checkpoint coincide here.

---

## 5. Training stability

| quantity | start → end (means by tenth, `results/m45c/report.md` §4) | reading |
|---|---|---|
| policy entropy | 2.72 → 1.77 nats (initial 2.86) | smooth decline, flattening after 0.5M; M4.4b aggro on the same schedule: 1.69–1.77 at 1.02M and 1.71–1.74 at 4M. The level is set by the 0.03 entropy coefficient; no collapse |
| approximate KL | 0.0046–0.0057 (max 0.0077) | small and steady; no target-KL early stop needed |
| clip fraction | 0.05–0.07 | ordinary |
| explained variance | 0.52 → 0.61 (0.4M) → 0.41–0.44 | positive throughout and lower than M4.4b's 0.49–0.55 at 1.02M; consistent with M4.5b caveat 4 (a trade's value is dominated by wave timing, so returns are noisier) |
| value loss | 0.004–0.008 (scaled) | stable |
| gradient norm (pre-clip) | 0.40 → 0.64 mean, max 1.49 | rising, so in the second half most minibatches are norm-clipped at 0.5, as in M4.4b; no spikes, far below the 100 alarm |
| critic | V(s0) = +0.106 for both seats (equal by symmetry; true value 0); \|V\| max 12.4 resources ≤ Ω | bounded |
| numerics | every loss, gradient and critic statistic finite | — |
| action distribution | no category above 0.37 overall; state-dependent (§6) | no collapse |

---

## 6. Preliminary behaviour

These are descriptive readouts from self-play records of 16 games × 1000 turns per checkpoint (both seats focal),
games against six opponents, and the M4.5b ledger. They are **not** evidence of strategic competence, because no
counterfactual has been run at scale (§9).

### 6.1 Action distribution

Self-play, mean probability mass:

| checkpoint | Last Hit | Thin | Slow Push | Shove | Ward | Recall | Trade (of lane) | Safe | Yield | entropy |
|---|---|---|---|---|---|---|---|---|---|---|
| untrained | 0.23 | 0.23 | 0.23 | 0.23 | 0.04 | 0.03 | 0.25 | 0.25 | 0.25 | 2.87 |
| 0.10M | 0.33 | 0.25 | 0.19 | 0.15 | 0.01 | 0.08 | 0.34 | 0.20 | 0.26 | 2.45 |
| 0.51M | 0.21 | 0.14 | 0.23 | 0.34 | 0.005 | 0.07 | 0.22 | 0.27 | 0.20 | 1.82 |
| final | 0.16 | 0.22 | 0.22 | 0.32 | 0.008 | 0.07 | 0.25 | 0.20 | 0.30 | 1.83 |

Shove rose from 16% to 32% over training and is still moving (`figures/action_shares.png`). M4.4b's econ / aggro
policies reached 35% at 2M.

### 6.2 State-conditioned behaviour of the final

The untrained network is flat across states, so every contrast below is learned:

| state (n decisions) | what changes |
|---|---|
| HP < 35% (3,025) vs ≥ 70% (19,099) | Recall 0.31 vs 0.01; Safe 0.65 vs 0.06 of lane mass; Trade 0.10 vs 0.30 |
| own crash under the enemy tower, Pressure legal (1,386) | Pressure 0.49 (untrained 0.32); Hold 0.50; Yield / Trade 0.06 each |
| holding a freeze (2,634) | Thin is the largest wave category (0.28); Trade 0.39 and Yield 0.39; Recall 0.01 |
| frozen on (2,482) | Shove 0.37, Recall 0.15, Hold 0.45, Trade 0.12 |
| enemy away (2,173) | Shove 0.49 |
| enemy crash under my tower (1,809) | Yield 0.53 |
| waves meet on the enemy half (7,411) vs my half (7,210) | Recall 0.14 vs 0.01; Safe 0.32 vs 0.16; Trade 0.10 vs 0.24 |
| enemy window strength ≥ 4 above ours (5,546) | Shove 0.45, Last Hit 0.09 |

### 6.3 Economy, freezes, Vision and Recall

| quantity | value |
|---|---|
| deaths per 1000 champion-turns | 17.1 at 0.1M → 7.8 by 1.0M |
| farm collection | 0.69 → 0.76 |
| freezes (≥ 4 held turns) per 1000 turns | 6.8; mean 5.9 turns; the freezer out-earns the frozen-on side in 69% of them (M4.4b econ / aggro: 7.0–7.6 per 1000) |
| Wards | 6.7 per 1000 turns (6.3 placed), down from 37 untrained; Ward mass 1% even where legal with vision down. The suite's Ward-oriented opponent is beaten (+1.21), and the counterfactual value of Wards is untested |
| completed Recalls | 41.6 per 1000 turns: 16 at low HP; 26 at about 64% HP with about 8 unspent gold, a reset that buys in 99% of cases |
| interrupted Recall attempts | about 34% in self-play, 42% against the bully, 41% in late training (M4.4b: 22–38%) |
| minion damage | about 9% of the damage a champion takes (0.19 bars per 100 turns against 1.94 from trades) |

Against **passive** opponents the final sieges. Against g_farm and g_safe, Pressure is 24–26% of its decisions; it
beats g_safe by +6.1, so a passive defender is exploited.

### 6.4 Two properties of the training distribution

These matter for the full campaign:
- **Natural power gaps are almost absent.** 42 of 29,561 decisions had a power ratio outside 0.8–1.25 (M4.4b: 0–1%).
  The "stronger champion zones or freezes" questions (B, C) therefore need initialized power-gap scenarios.
- **The early game is a small share of training.** Levels 1–10 last about 55 turns per game, and level 18 is
  reached in every 512-turn game. That is 6% of the decisions of a 1000-turn evaluation game (94% are at level ≥ 11)
  and about 11% of a 512-turn training game. The full evaluation reports early-game behaviour separately.

---

## 7. Measured performance

All figures are from this machine: 12 logical CPUs, 17 GB, Python 3.13, torch 2.14 CPU, one thread per process.

| measurement | value | source |
|---|---|---|
| pilot, end to end, one process alone | **4,277 agent transitions/s** (2,139 joint steps/s); 1,024,000 in 241.5 s: collection 193 s, PPO updates 44 s, snapshots + diagnostics + logs 4 s | measured |
| per-update throughput over the pilot | 4,300–4,530 per fifth of the run; constant | measured |
| one update (4,096 transitions) | 0.94 s: environment stepping 0.66 s (70%), inference 0.02 s (2%), other collection 0.09 s (9%), GAE + batch 0.002 s, PPO update 0.17 s (18%); same for the untrained and the final network | measured (instrumented) |
| environment stepping alone | about 3,100 joint steps/s | measured |
| checkpoint save | 18 ms, 149 KB; 40 snapshots ≈ 6 MB per 1M transitions | measured |
| concurrent training processes | per process 4,223 / 3,515 / 2,936 / 2,430 transitions/s at 1 / 3 / 6 / 11 processes (×1.00 / 0.83 / 0.70 / 0.58); aggregate 4,173 / 7,570 / 12,499 / 18,238 | measured (short runs) |
| memory per training process | 0.45 GB early; **1.5–1.6 GB** once the 16 lane-step caches (20,000 entries each) are full, after about 14,000 joint steps per environment (≈ 0.45M transitions); 0.56 GB at cache 5,000, 0.37 GB at 2,000 | measured (pilot peak 1.60 GB; cache probe) |
| speed against cache size | collection 0.87 s per rollout at 20,000, 0.91 at 5,000, 0.92 at 2,000 (≈ 4–5% slower end to end); results identical (test) | measured |
| evaluation | about 1,150 simulated turns per worker-second (paired games of network against a scripted strategy); the pilot's curve (130 matchups) took 5.0 min, the headline 3.5 min, history 0.4 min, behaviour + ledger 0.8 min, all on 11 workers | measured |

**Extrapolated** training wall clock: measured pilot throughput × measured concurrency factor, lane cache 20,000. The
"1 run alone" and "among 11" columns are single runs; the seed columns are the whole set of runs.

| transitions per seed | 1 run alone | 1 run among 11 | 3 seeds concurrent | 6 seeds concurrent | 10 seeds concurrent | 6 seeds sequential |
|---|---|---|---|---|---|---|
| 1,000,000 | 4 min | 7 min | 5 min | 6 min | 7 min | 0.4 h |
| 4,000,000 | 16 min | 27 min | 19 min | 22 min | 26 min | 1.6 h |
| 5,000,000 | 19 min | 34 min | 23 min | 28 min | 33 min | 1.9 h |
| 8,000,000 | 31 min | 54 min | 37 min | 45 min | 52 min | 3.1 h |
| 10,000,000 | 39 min | 68 min | 47 min | 56 min | 65 min | 3.9 h |

Memory sets the concurrency limit:
- 6 full-cache processes need 8.9 GB;
- 10 need 14.8 GB and 11 need 16.3 GB, too close to 17 GB.

Above 8 concurrent runs the plan therefore uses `lane_cache = 5,000`: 0.56 GB per process, about 4% slower, results
unchanged. The extrapolation assumes the constant per-update throughput the pilot showed. Its main uncertainty is the
concurrency factor, which was measured on short runs from the untrained network; the pilot itself showed no
dependence of speed on the network.

---

## 8. Full-training recommendation

### 8.1 Evidence the budget rests on

| consideration | finding |
|---|---|
| speed of improvement | fast to 0.4M (−7.4 → +1.5), slower 0.4–1.0M (+1.5 → +2.5) |
| improvement near the end | **yes**: final > 0.77M snapshot (+0.67 ± 0.46); final > half-way against 9 of 10 opponents; the last three curve points still rise. The learning rate was still at 75% of its initial value, so this is not an annealing artefact in either direction |
| instability or collapse | none (§5) |
| M4.5 against M4.4 | entropy decays identically; strength against fixed opponents rises **more slowly** than M4.4b aggro, which plateaued by ≈ 0.3M; the M4.5 critic explains less variance |
| what M4.4 needed | strength flat from ≈ 1M, behaviour still drifting at 2M, the 4M extension +0.6 to +1.5 head-to-head over 2M with a stable qualitative profile |
| cost | training is cheap (minutes per million transitions); evaluation costs more than training |
| longer or more seeds | both are affordable, and they answer different questions. Seeds are the unit of replication for any "PPO discovers X" claim. With 3 seeds a t-interval uses t₂ = 4.30; with 6, t₅ = 2.57, so the same seed SD gives a 2.4× narrower interval, and cross-play has 15 pairs instead of 3. Length is needed because the pilot has not plateaued and behaviour (Shove share, Ward use) is still moving |

### 8.2 Default: 6 seeds × 8,192,000 agent transitions

| item | value |
|---|---|
| seeds | 0, 1, 2, 3, 4, 5 (independent initializations; the pilot's seed 100 is not one of them) |
| transitions per seed | 8,192,000 = 4,096,000 joint steps = 2,000 updates |
| schedule | learning rate 3e-4 annealed linearly to 0 over each run's own 8,192,000 (the M4.4b protocol); every other setting is the pilot's |
| self-play | 100% current-policy; snapshots every 25,600 (320 per run, ≈ 48 MB) kept as historical evaluation opponents |
| evaluation interval | every 409,600 transitions (every 16th snapshot, 21 points with the untrained network) against the 10-opponent suite (32 paired seeds, W = 460); headline at the **midpoint 4.096M** and the final (64 paired seeds, W = 1000) |
| historical comparisons | final against 8 evenly spaced own snapshots and the untrained network; a 9-checkpoint round robin per run (cycling check); cross-play of the 6 finals (15 pairs) |
| concurrency | 6 processes at once, lane cache 20,000, about 8.9 GB peak |
| expected training time | ≈ 46 min (extrapolated) |
| expected standard evaluation | ≈ 1.4 h on 11 workers (curves ≈ 49 min, headline ≈ 20 min, the rest ≈ 17 min) |
| expected strategic battery (§9) | ≈ 0.4 h (6 families × 48 states × 16 rollouts × 100 turns per final) |
| total commitment | ≈ 2.5 h wall clock; 49M agent transitions of training |

**Why this option.**
- **8,192,000 transitions.**
  - It is 8× the point where the pilot was still improving, and 2× M4.4b's longest runs. That margin is for an
    environment that learns more slowly than M4.4.
  - It keeps M4.4b's anneal-to-zero protocol.
  - The midpoint snapshot (4.096M, learning rate at 50%) is a natural replication of M4.4b's extension budget within
    each run. If the final equals the midpoint, 4M would have sufficed and that is reported. If not, the slope over the
    second half answers whether more is needed (§8.4).
- **Six seeds.** The interval narrowing above, and enough seeds to report how many of them show a behaviour, not only
  its average.
- **Cost.** All of it is affordable on this machine in an afternoon.

**Uncertainty.** A one-seed pilot cannot say when PPO converges: the 8M choice is a margin, not a prediction. Also, the
pilot's end used a learning rate of 75% of its initial value; a full run's last quarter uses less than 25%, which
slows late change by construction.

### 8.3 Alternatives

| option | seeds × transitions | cost (extrapolated) | what it would support | what remains open |
|---|---|---|---|---|
| **lower cost** | 3 × 4,096,000 | ≈ 19 min training, ≈ 0.7 h evaluation, ≈ 1 h total | replication of M4.4b's scale on the new mechanics: whether policies beat the fixed suite and condition on state in every seed; comparisons with M4.4b at equal budget | seed intervals use t₂ = 4.30 (wide); the pilot has not plateaued, so "did not learn X" could be a budget limit; 3 cross-play pairs |
| **higher confidence** | 10 × 16,384,000 (lane cache 5,000) | ≈ 1.9 h training, ≈ 3 h evaluation, ≈ 5 h total | a seed-level distribution of discovered behaviours (e.g. "7 of 10 seeds learn guarded freezes"); convergence evidence at 2× the default horizon; 45 cross-play pairs for non-transitivity | still one opponent distribution (self-play); evaluation dominates the cost; a long-run drift could also be self-play cycling, which the tournament has to separate |

**Recommendation: the default.** The lower-cost option risks stopping before the improvement the pilot shows has run
out. The higher-confidence option doubles the horizon and the seed count before we know whether 8M is already a
plateau. If the default's §8.4 criteria show continued improvement or a high seed variance, the higher-confidence
setting is the natural next step.

### 8.4 Extension and stopping criteria (pre-registered for the campaign)

1. **Plateau.** A run has plateaued when both of these hold:
   - over its last quarter (6.1M → 8.2M), the scripted-suite mean on the curve changes by less than ±0.3 (about the
     curve's resolution);
   - the final against its 6.1M snapshot is within ±0.3 with the interval covering 0 (W = 1000, 64 paired seeds).
2. **Extension.** If at least 3 of the 6 seeds do not plateau, **and** the final beats its own 4.096M midpoint by more
   than +0.5 (interval excluding 0), propose a matched 2× extension (16,384,000, all seeds, fresh runs with the same
   settings, as in M4.4b). This needs its own approval.
3. **Behaviour still changing while strength is flat** (the M4.4b pattern). Report the drift: Shove share, Ward use,
   freeze-phase actions and Recall timing by checkpoint. Decide with the counterfactuals of §9 whether the new
   behaviour is better. Drift alone is not a reason to stop, and not a reason to extend.
4. **Stop and investigate** if any of these happen: KL above 0.05 sustained; any non-finite statistic; gradient norms
   above 100; entropy below 0.3 nats; a later snapshot losing significantly to an earlier one in the tournament on at
   least 2 seeds (cycling).
5. **Seeds that disagree.** If cross-play or behaviour separates the seeds into groups (e.g. a cross-play gap above 2
   resources), report them as distinct outcomes, not as an average.
6. **Not criteria.** Self-play reward against the current policy (0 by construction); the frequency of Shove or of
   any action.

---

## 9. Full evaluation design (after training approval)

The principle is the brief's: an action is judged by whether it is economically appropriate in the state where it is
chosen, measured by paired counterfactuals, not by its frequency. Every family runs on natural states from self-play
of each final, and where natural states are rare (§6.4), also on initialized scenarios, labelled as such. Seeds are
held out (93,000–94,999). Intervals are over states nested in seeds, and seeds are reported individually.

The framework is `m45c/strategic.py`. It is built on the M4.5b engine: the real resolver and chance stream, common
random numbers across candidates, and learned continuations through `NetPolicy`. A smoke run of all six families on
the pilot final (3 states × 4 rollouts × 50 turns, 8 s) ran end to end. That was a check of the framework, not
evidence.

| question | descriptive measures (records, ledger) | counterfactual family (plan vs the policy's own continuation) |
|---|---|---|
| **A. Wave management** | wave category by wave position, window strength, crash state, freeze phase; the realized effect 4 turns later | `A_wave`: Last Hit / Thin / Slow Push / Shove for 4 turns from states with fronts in contact |
| **B. Freezing and zoning** | freeze episodes (frequency, length, denial, gold and XP to the frozen-on side, end type), threat-only denial (uncollected gold and XP under a credible threat, without connected trades), the frozen-on reply | `B_freeze`: keep (Thin + Hold) / guard (Thin + Trade) / break; `B_frozen`: Shove / step back / Recall; initialized ×1.76 power-gap and scripted-freeze starts |
| **C. Trading and minion damage** | trade attempts and connections by enemy front size, level, HP, power and catchability; the minion share of trade damage (ledger) | `C_trade`: Trade / Hold / Safe for one turn while the enemy front outweighs ours by a wave and the enemy is catchable |
| **D. Vision** | Ward rate by exposure, own wave position and enemy presence; wards before forward play; interruptions; ganks on exposed turns with and without vision | `D_vision`: Ward / Last Hit + Hold with vision down and the own front at the lane cell or beyond |
| **E. Recall and investment** | Recall by HP, unspent gold, wave state and enemy absence; interrupted Recalls; power on return; the opponent's adjustment after a return | `E_recall`: Recall now / stay and Shove 3 turns / step back 3 turns at HP ≤ 50% |
| **F. General competence** | the fixed suite; the M4.5b restricted-game strategies; own snapshots (history, tournament); cross-seed play; the M4.4b finals through the adapter (**OOD**, labelled); exploitability lower bounds from warm-started best-response attackers (the former M4.4d, the M4.3b protocol) | — |

Additional items:
- Early-game behaviour (levels 1–10) is reported separately, because it is about 11% of training experience.
- Viewer recordings of representative games are exported for the debugger.
- The attackers (a training cost of about 1M transitions each) are part of the strategic battery's approval, not of
  this plan's.

---

## 10. Readiness decision

### **GO** — the pilot is stable, and the recommended full training plan (§8.2) is ready for approval

What supports it:
- Correctness is verified on the calibrated preset itself: masks, log-probabilities, seats, rewards, truncations and
  bootstraps (§1.4).
- Training is reproducible bit for bit.
- The pilot is healthy (§5) and improves against fixed and historical opponents (§4).
- Runtime and memory are measured, and the plan respects them (§7).

No open issue requires a fix before training. Known limitations and their expected impact:

| # | limitation | expected impact | handled by |
|---|---|---|---|
| 1 | one pilot seed | seed-to-seed spread unknown (M4.4b: SD 0.2–0.9) | 6 campaign seeds; seed-level reporting |
| 2 | not plateaued at 1.0M | the 8M budget is a margin, not a prediction | midpoint and plateau criteria (§8.4) |
| 3 | the fixed scripted suite is already beaten everywhere | it can no longer reveal weaknesses | cross-seed, historical and best-response evaluation as primary (§9 F) |
| 4 | natural power gaps almost absent; levels 1–10 are about 11% of training experience (55 turns per 512-turn game) | zoning by a stronger side and early-game play are rarely trained or observed naturally | initialized scenarios; separate early-game reporting; no environment change (the preset is frozen) |
| 5 | Ward rare (≈ 1% of decisions) | Vision may stay rare (M4.5b caveat 5) | counterfactual `D_vision`; rarity is not judged a failure without it |
| 6 | about 34–42% of Recall attempts interrupted (M4.4b: 22–38%) | possibly suboptimal timing, or the rational guessing game M4.4c found | `E_recall` counterfactuals |
| 7 | explained variance 0.41–0.44 (M4.4b ≈ 0.5) | noisier credit assignment for trades (M4.5b caveat 4) | tracked as a health quantity; no change |
| 8 | full lane caches use 1.5 GB per process | 11 concurrent runs would not fit at cache 20,000 | at most 6 concurrent at 20,000, or cache 5,000 (identical results, about 4% slower) |
| 9 | M4.5b's accepted caveats (minion share, Shove-centred scripted equilibria, conditional freezes, situational Vision) | as documented there | tracked in the evaluation |

**Requested decision:** approve the default (6 × 8,192,000), choose an alternative, or adjust it. Nothing further
will be trained until then.

---

## 11. Reproduction

```bash
python -m lanegame.experiments.rl.m45c audit smoke smoke_eval      # seconds
python -m lanegame.experiments.rl.m45c pilot prefix_check          # 4 min (one process)
python -m lanegame.experiments.rl.m45c eval_pilot behavior_pilot   # 10 min on 11 workers
python -m lanegame.experiments.rl.m45c bench                        # 4 min (profile, scaling, cache probe)
python -c "from lanegame.experiments.rl.m45c.strategic import smoke; smoke()"
python -m lanegame.experiments.rl.m45c report                       # results/m45c/report.md, figures, full_plan.json
python -m pytest tests/test_m45c_training.py
```

The pilot refuses to overwrite an existing run directory. `train_full` refuses to start without
`--confirm-m45c-full`.
