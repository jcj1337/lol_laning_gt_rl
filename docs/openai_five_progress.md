# OpenAI Five-inspired self-play: progress log

Branch `research/openai-five-selfplay` (from `m2-m3-autonomous` at 9444de9). Session of 2026-10-11.
Final report: `docs/openai_five_selfplay_report.md`; methodology notes: `docs/openai_five_adaptation.md`.

## Phase A: audit of M3 (done)

Read: `lanegame/rl/{network,agent,ppo,train,selfplay,evaluate}.py`, `lanegame/experiments/rl/*.py`,
`lanegame/env/{core,observation,agents}.py`, the M3 sections of `docs/m2_m3_implementation_report.md`,
`results/rl/m3b_selfplay.md`, `m3c_evaluation.md`, `m3c_br_warm.md`. Ran `tests/test_rl.py`,
`tests/test_rl_selfplay.py` (15 passed, 10 s).

PPO (`ppo.py`, `train.py`): no material bug found.
- Masking: illegal logits -> -1e9, probability exactly 0, never sampled, 0 entropy contribution; the mask is the
  engine's own legal set. Log-probabilities from `log_softmax` of the masked logits; ratio = exp(new - old) with
  old log-probs recorded at collection under the same parameters that sampled the action.
- Clipped surrogate, clipped value loss (±2), value target = GAE λ-return, entropy and policy loss averaged over
  steps with more than one legal action (forced absence steps carry no gradient), per-minibatch advantage
  normalization over the same steps, global gradient-norm clip 0.5, Adam with linear annealing.
- Termination vs truncation: GAE bootstraps 0 after the horizon, V(final obs) after a time-limit cut, never across
  episode boundaries (tested by hand in `test_rl.py`). Obs, mask, action, reward and value arrays are aligned
  per (t, env); the next-value of step t is the value at t+1 only when the episode continues.
- Seeding: torch init, sampling generator, opponent/seat generator and per-episode SeedSequence children all
  derive from `PPOConfig.seed`; bit-for-bit reproducible with one torch thread (tested).
- Minor, left unchanged to keep the baseline code path identical: `Collector._reset` with `skill_levels` rebuilds
  the environment each episode without passing `gank_reward` (only reachable with a legacy configuration,
  `expected_gank_reward=True` and `skill_levels`, a combination no experiment uses) and loses the lane cache
  (performance only). The new collector keeps one environment per skill pair.

Existing self-play (`selfplay.py`): verified that M3b is NOT current-policy self-play.
- The learner controls one seat; the opponent is a scripted agent (30%), the latest frozen snapshot (35%) or a
  uniformly drawn snapshot (35%). Snapshots are `NetAgent` deep copies with `requires_grad=False` and `eval()`;
  they are genuinely frozen (tested). The "latest" snapshot can be up to 25 updates old. Only the learner's
  transitions are trained on; learner steps = environment turns.
- Opponents are drawn per episode with the run's seeded generator (reproducible). Snapshot files carry name,
  step, run and PPO config; loading reproduces the network (round trip tested).
- `train_ppo` / `train_selfplay` overwrite `history.jsonl`, snapshots and `final.pt` silently when rerun into the
  same directory. Not changed (baseline code); the new trainer and the new experiment driver refuse to write into
  an existing run directory and never replace a checkpoint.
- Metrics: `mean_return` mixes scripted and snapshot opponents (`return_by_opponent` separates them). Accurate.

Existing evaluation (`evaluate.py`, `experiments/rl/evaluation.py`):
- Held-out seed roots (10,000 / 20,000 / 30,000) disjoint from training streams; both seats on the same seeds
  (common random numbers); frozen agents (no update); exact exploitability only on legacy configurations
  (`as_policy` refuses champion mode), champion numbers labeled as lower bounds. Correct.
- The 95% interval of `head_to_head` treats the two seat games of one seed as independent. Measured: they are
  negatively correlated (−0.45 vs hold_farmer, −0.17 vs aggressive_trader for a published policy), so the
  reported intervals are conservative (up to ~35% too wide). Kept (conservative); noted in the report.

Seat symmetry of the champion game: turn resolution and observations are mirror-equivariant under a permutation of
the chance vector (existing tests). Measured: always-Trade mirror games over 24,000 episodes give a seat-A payoff
of −0.008 ± 0.016; hold_farmer, uniform, aggressive_trader, opportunistic_trader mirrors are all within noise.

## Phase B/C: shared-policy self-play with a current / historical mixture (done)

- `lanegame/rl/shared.py` (new): `SharedConfig` (current_fraction 0.8, snapshot_every 25,600 agent
  transitions, pool_size 8, optional initial snapshot), `MatchupSampler` (per-episode: current-policy game with
  probability current_fraction or when the pool is empty; else a uniform snapshot from the pool and a uniform
  learner seat), `SharedCollector` (two trajectory slots per environment; one batched forward pass for all
  trainable seats; historical seats act through their frozen `NetAgent`), `rollout_batch` (GAE over all slots,
  PPO batch of trainable entries only), `train_shared` (budget and LR schedule in agent transitions; run.json,
  history.jsonl, snapshots with metadata, final.pt; refuses existing run directories).
- `lanegame/rl/agent.py`: `save_checkpoint(..., exist_ok=True)`; `exist_ok=False` refuses to overwrite.
- `tests/test_rl_shared.py` (new, 14 tests, ~10 s): both seats from one network with matching log-probs and
  values; own-perspective observations (seat B = mirrored seat A view, asymmetric skills); legal actions and
  masks; zero-sum rewards not negated twice (episode sums = ±payoff); historical games train only the learner
  seat and snapshots stay frozen; recorded log-probs = rollout policy and unchanged by the update; termination,
  truncation and bootstrap values; snapshot files reproduce action distributions and carry metadata; existing
  directories and checkpoints are not overwritten; mixture statistics (80/20 within 4.5 SE, seats and snapshot
  choice uniform, bootstrap); accounting (env turns, agent transitions, current pairs); reproducibility; seat-swap
  symmetry; legacy configuration compatibility.
- Mutation check: double-negating seat B's reward and recording the snapshot's seat as trainable each make the
  intended tests fail.
- Bug found and fixed during the smoke stage: per-rollout counts overwrote the cumulative `agent_transitions` /
  `env_turns` keys of the history records (training unaffected); per-rollout counts are now `rollout_*`.

## Phase D: experiments (in progress)

- Driver `lanegame/experiments/rl/openai_five.py` (training stages) and `openai_five_eval.py` (evaluation,
  exploitability, report, figures). Results in `results/rl/openai_five/`; checkpoints in
  `results/rl/checkpoints/of5/` (git-ignored).
- Protocol decisions: all conditions from scratch, same seeds (same initial network), skills drawn per episode and
  seat from {0.2, 0.5, 0.8} (so the skill analysis is meaningful for every condition), ~1,024 agent transitions per
  update in every condition (m3b 16 envs x 64 x 1 seat; shared 8 envs x 64 x 2 seats), snapshot every 25,600
  agent transitions, budget in agent transitions.
- Smoke (10,240 transitions, 1 seed): all three trainers run; ~5,000 agent transitions/s per process for the
  shared trainer.
- Pilot (51,200 transitions, seed 0): m3b 15 s, sp100 8 s, sp80 9 s. At 51k both shared conditions beat
  hold_farmer, aggressive_trader, punisher and opportunistic_trader (+0.3 to +1.2) without ever training against
  them; m3b +0.3 to +1.3 (it trains against them). sp80 and sp100 are bit-identical until the first snapshot (same
  seed, empty pool), as designed.
- Main budget chosen: 307,200 agent transitions per seed (the published M3b self-play budget; each run < 2 min).

- Main training (wall clock 2.3 min, 10 processes): 9 runs, 307,200 agent transitions each;
  environment turns 307,200 (m3b), 153,600 (sp100), ~169,500 (sp80); historical share of sp80 transitions 0.10.
- Evaluation (`evaluate`, 1,761 head-to-head jobs, 23 min with the test suite running alongside). Bug fixed in the
  report: sp80 snapshots fall a few hundred transitions apart across seeds, so learning curves are binned by the
  nominal snapshot index (multiples of 25,600).
- Full suite after the implementation: 476 passed (10.5 min, slowed by the concurrent evaluation).
- Commit 84112de: implementation, tests, driver, adaptation doc, progress log.

Preliminary findings (before exploitability):
- Fixed scripted opponents: all three conditions comparable; m3b best on its own training suite
  (+4.27 vs +3.47 / +4.00) and on held-out scripted agents (+5.00 vs +4.72 / +4.79), differences within the
  3-seed t intervals.
- Cross-play: family ordering sp80 > sp100 > m3b by small margins; sp80 beats m3b in all 9 seed pairings
  (+0.04..+0.35, mean +0.20); all new policies ~ parity with the published skill-randomized M3b policies (trained
  on 512k steps) and beat the published equal-skill M3b policies by 1.2-2.2.
- Snapshot matrices: no forgetting or cycles inside any run (no older snapshot beats a newer one by 2 SE).
- Behavior: all learned policies are Hold/Trade mixtures; Yield, Safe, Recall, Freeze and siege objectives
  essentially unused.
- Skill: shared self-play policies condition Trade on skills (5 of 6 runs; Trade share 0.20-0.55 across skill
  pairs), m3b policies do not (0 of 3; flat 0.54-0.57, as in M3c). When mechanically outclassed by a learned
  reference, shared policies lose 0.2-0.6 less per game than the seed-paired m3b policy (all 6 pairings).

Exploitability (`exploit`, 30 attackers of 102,400 steps, 3.9 min): lower bounds M3b +0.36 ± 0.34, 100/0
+0.35 ± 0.53, 80/20 +0.24 ± 0.70 (per seed +0.04..+0.56); from-scratch attackers never reach parity; calibration
against pub_sprand_s1 +0.17 / +0.12 (M3c: +0.11).

## Phase F/G (done unless noted)

- `behavior_detail` (recorded traces): shared self-play trades much less when behind in HP (0.25-0.30 vs 0.51 for
  m3b in mirror games) and late in the game; recall essentially unused; no forward positioning.
- `exploit_strong` (one warm attacker per target, 307,200 steps, 2.9 min): bounds rise for 5 of 9 targets (up to
  +1.15), Spearman +0.57 between budgets; condition means +0.53 / +0.70 / +0.64 (indistinguishable); the
  standard attacker is budget-limited, so standard-budget exploitability does not rank policies reliably.
- `legacy_exact` (1.6 min): exact exploitability on K = 16 rises in all methods (final 3.43 / 3.38 / 3.93; uniform
  3.74); 80/20 more exploitable than 100/0 in all 3 seeds from 61k to 143k transitions. The re-run M3b legacy
  runs reproduce the published series bit for bit.
- New figure `fig_skill.png`; Markdown header bug (literal `|`) in behavior_detail fixed.
- 50/50 ablation (`train_ablation`, 1.2 min): performance like 80/20 (cross-play +0.15), most exploitable condition
  under the 3x attacker (+0.85 ± 0.10).
- Equal environment turns (`train_envturn`, 1.8 min; sp100 614,400 and sp80 ~553,500 transitions at ~307k turns):
  clearly stronger than m3b (cross-play +0.31..+0.76 vs the m3b policies, 6/6 runs) and less exploitable (3x
  attacker +0.32 / +0.46).
- `evaluate_extra` (1,206 jobs, 15.7 min), `exploit_extra` (27 attackers, 3.6 min), `exploit_strong_extra` (9
  attackers of 307,200 steps, 2.8 min), `report_extra`.
- Total RL wall clock ≈ 60 min on ≤ 10 processes; no run longer than 2.5 min.

## Phase H: report (done)

`docs/openai_five_selfplay_report.md` written from the generated tables (`results/rl/openai_five/*.md`), every
number cross-checked against the JSON results. Behavior-detail and main report regenerated after the fixes.

Remaining issues / not done: resumption inside a run (optimizer and RNG state) is not implemented (the driver
resumes at run granularity); only 3 seeds per condition; attackers are budget-limited (documented).
