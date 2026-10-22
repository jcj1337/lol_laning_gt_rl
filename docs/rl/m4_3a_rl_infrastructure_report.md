# M4.3a — RL infrastructure integration and correctness audit

Branch `development/m4-3a-rl-infrastructure`, on top of `main` at `fbd69e1` (M4.2e-D). 2026-10-12.

**Purpose.** Before the M4.3 self-play experiments, establish that the environment interface, the PPO
implementation and the self-play method are correct, reproducible and well enough structured to carry the research
question of the README: *can an RL agent learn to take advantage of the interactions of the laning simulator without
being taught League strategies?* The milestone adds no mechanics and changes no game specification; every preset's
digest pin is unchanged.

**Deliverables.** (1) The PettingZoo Parallel API adapter brought up to the M4 environment and validated as the
project's standard multi-agent interface (§2). (2) A PPO and self-play audit with independent numerical references
(§3). (3) A bounded smoke test of shared-policy self-play on `GameConfig.m4_lasthit()` (§4). (4) README, roadmap
and this report (§6). (5) Unresolved concerns and recommendations for M4.3b (§7).

## Contents

1. Audit of the existing architecture: what is reused, what changed
2. PettingZoo: compatibility, validation, and the decision on its role in training
3. PPO and self-play correctness audit
4. The M4.3a smoke test on `m4_lasthit()`
5. What is verified and what is only empirically supported
6. Files, tests, reproduction
7. Unresolved concerns and recommendations before M4.3b

## 1. Audit of the existing architecture

| component | location | verdict |
|---|---|---|
| M4 environment, observation, masks | `lanegame/env/core.py`, `observation_m4.py` | reused unchanged. One `step` resolves a turn through `champions_m4.resolve_turn_m4`; `terminated` is never set, the `max_turns` time limit is a truncation; the 88-entry observation is lossless (decoder-proven) and in [0, 1]; the mask is the engine's legal set and `step` raises on an illegal action |
| PettingZoo adapter | `lanegame/env/pettingzoo_adapter.py` | **improved** (§2): M4 champion states and `StartState` starts, infos with the legal sets, an AEC conversion, documented seeding and continuing-time semantics |
| policy/value network, masked categorical | `lanegame/rl/network.py` | reused unchanged; its masking properties are now verified exactly (§3.3) |
| PPO (GAE, clipped update) | `lanegame/rl/ppo.py` | reused; **adds** the pre-clip gradient norm to the statistics. GAE and the loss/gradient are now checked against independent references (§3.1, §3.2) |
| experience collection, continuing time | `lanegame/rl/train.py` | reused; **adds** critic/return statistics and truncation counts to the history; `net.eval()` after the update (no behavioural change: the network has no dropout or normalization) |
| shared-policy self-play | `lanegame/rl/shared.py` | reused; **adds** per-seat, truncation, termination and single-choice counts to the rollout accounting and critic/return/advantage statistics to the history |
| snapshot pool, matchup sampler | `lanegame/rl/selfplay.py`, `shared.py` | reused unchanged; isolation re-verified on `m4_lasthit()` |
| checkpoints | `lanegame/rl/agent.py` | **improved**: format version 2 carries a SHA-256 digest of the weights (verified on load) and provenance (library versions, commit); version-1 files load unchanged; `checkpoint_info`, `weights_digest` |
| evaluation | `lanegame/env/discounted.py`, `rl/evaluate.py` | reused unchanged; the E1 discounted evaluator with the guaranteed tail bound is the M4 evaluator (Ω = 107.04 resources for `m4_lasthit()`, bound 1.05 at W = 460 and 0.005 at W = 1000) |
| CLI | `lanegame/__main__.py` | `rl-shared --model m4_lasthit` (and the other M4 presets) trains under `PPOConfig.continuing`; `experiments-rl m43a_smoke` |
| earlier M4 smoke | `lanegame/experiments/m4/smoke.py` | targets `m4()` and the `m2c` fixture; left as a historical driver. The M4.3 smoke is `lanegame/experiments/rl/m43a_smoke.py` |

Nothing was replaced. The M1–M3 design (a NumPy environment with typed `Observation`/`StepResult` records, a
CleanRL-style single-file PPO, a thin PettingZoo adapter) holds up for M4; the gaps were in validation (no reference
checks, no adapter equivalence test, no M4 trainer tests), observability (no gradient or critic statistics) and
checkpoint integrity, and those are what this milestone adds.

## 2. PettingZoo

### 2.1 Compatibility with `GameConfig.m4_lasthit()`

The adapter already wrapped any `GameConfig`, but three things were wrong or missing for M4:

- `reset(options={"champ": [...]})` read a list-form champion state as a legacy `ChampState`; under ruleset m4 it
  must be an `M4State` (the environment raises otherwise). Fixed through `starts.champ_from_list`, and a
  `StartState` (`lanegame.env.starts`) is accepted directly as `options={"start": st}`.
- No validation existed on any M4 preset, and no test compared the adapter's trajectories with the native
  environment's.
- Continuing time was undocumented at the adapter: without `max_turns` an episode never ends (the Parallel API has
  no notion of a game length), which is correct but has to be said.

Also added: `infos` at reset carry each seat's legal set; `aec()` returns the same environment through PettingZoo's
AEC API (`parallel_to_aec`); the chance-stream seeding and how to reproduce a `play_episode` game through the
adapter are documented in the module docstring.

### 2.2 Validation (`tests/test_pettingzoo_m4.py`, 11 tests)

- PettingZoo's official `parallel_api_test` passes on `m4()`, `m4_emergent()`, `m4_natural()`, `m4_lane()` and
  `m4_lasthit()`; the official AEC `api_test` passes through the conversion. (The two legacy adapter tests in
  `test_env.py` and `test_champions.py` still pass.) PettingZoo's advisory warnings (dict observation space, agent
  names not of the form `player_0`) are conventions, not failures; the dict observation with `action_mask` is the
  convention PettingZoo's own classic environments use.
- Spaces, masks, dtypes, infos at every step of a 120-turn game; truncation empties `agents`; stepping after the
  end raises.
- **Trajectory equivalence.** Under the same chance seed and the same 300-turn action sequence the adapter and a
  native `LaneEnv` produce identical observation vectors, masks, rewards, termination and truncation flags and
  turn records (`state_after`, `champ_after`, `reward_units`, `chance`, `events`): exact equality, not tolerance.
- A `play_episode` game of two scripted natural-wave agents is reproduced through the adapter from
  `episode_streams(seed)`: identical rewards and actions.
- Custom start states: a mid-game `StartState`, the `root`/`k0`/`champ` arguments, and their list forms all give
  the native environment's observations and the same next turn; a legacy list form is still read as a `ChampState`.
- `state()` is the lossless global state under ruleset m4: `decode_m4` reads the phase, lane, champion state and
  skills back at every step.

### 2.3 Decision: the adapter is the interface, `LaneEnv` stays the training substrate

PettingZoo is now the standardized interface for anything outside this repository: any Parallel-API trainer can
consume `LaneParallelEnv` and, because of the equivalence test, its results are directly comparable with ours. The
training loop in `lanegame.rl` keeps stepping `LaneEnv` directly, for these reasons, in order of weight:

1. **Correctness.** The collectors need, per environment, two trajectory slots (one per seat), a matchup pinned for
   the length of a game, a batched forward pass over every learner-controlled seat of every environment, and the
   pre-reset final observation at a game time limit for the bootstrap. The Parallel API gives the final observation
   but nothing else of this; training through it would wrap the same logic in dictionary plumbing and add no
   guarantee. Every guarantee the trainer relies on is now tested directly (§3).
2. **Maintainability.** The typed `Observation` (with `state`, `champ`, `legal`, `cfg`) and `StepResult.info` feed
   the traces, the discounted evaluator, the diagnostics and the viewer. The adapter's observation dict is a
   projection of that record.
3. **Interoperability** is served by the adapter, not by the trainer: the external-tool use case is "run another
   algorithm on our game", and that works today.
4. **Performance** was not decisive: the adapter's own cost is a dictionary and an `int8` cast per step, a few
   microseconds against the 250 µs environment step.

Recommendation for M4.3b (optional): run one external PPO implementation through the adapter on `m4_lasthit()` with
a matched budget as an independent implementation check. Our PPO is already checked against an independent
re-implementation of its loss (§3.2); an end-to-end external run would additionally check the training loop.

## 3. PPO and self-play correctness audit

The audit read `ppo.py`, `train.py`, `shared.py`, `selfplay.py`, `network.py`, `agent.py` and `discounted.py`
against the stated semantics (module docstrings, design §10.4) and against CleanRL's PPO as the reference
convention, then wrote the checks of `tests/test_rl_reference.py` (10 tests) and `tests/test_rl_m4_lasthit.py` (6
tests). **No defect was found in the algorithm.** The items below say what was checked and how.

### 3.1 Generalized advantage estimation

`compute_gae` (backward recursion with `terminated` dropping the successor value and `done` stopping the
λ-recursion) is compared with an independently written **forward-view** λ-return: for each step, the mixture
(1 − λ) Σ λ^(n−1) G^(n) of n-step returns within the episode, the last available n-step return taking the remaining
weight λ^(n−1), every n-step return bootstrapping the recorded successor value (zero after a termination). On
random rewards, values and random termination/truncation flags, for (γ, λ) = (0.99, 0.95), (1, 1), (0.9, 0),
(0.5, 0.7), the two agree to 1e-10 in both advantages and value targets. The special cases are pinned: γ = λ = 1 on
a terminated episode gives the Monte Carlo return, λ = 0 the one-step TD error. The earlier by-hand tests (three
steps; exact-critic zero advantages; the biased-critic truncation formula) remain.

### 3.2 The clipped objective, value loss, entropy and the gradient

`reference_loss` re-implements the PPO minibatch loss independently of `lanegame.rl.ppo`: logits masked with −∞ on
illegal actions, `torch.distributions.Categorical`, per-minibatch advantage normalization over choice steps, the
clipped surrogate as the maximum of the unclipped and clipped terms, the clipped value loss as the maximum of the
two squared errors, the entropy bonus. On one 192-sample minibatch with a deliberately perturbed network (ratios
inside and outside the clip range, seven single-choice steps), `ppo_update` with gradient clipping disabled reports
the same policy loss, value loss and entropy (1e-6) and leaves **the same parameter gradients** on every tensor
(atol 1e-6, rtol 1e-5); the reported `grad_norm` equals the norm of the reference gradient. The clip fraction and the
k3 KL estimator are finite and in range. The sign and clipping conventions of the update are therefore those of the
reference, not merely plausible.

### 3.3 Action masking and gradient behaviour

Exact statements, checked exactly: illegal actions have probability 0.0 (not merely small: `exp(−1e9 − max)`
underflows to 0 in float32); the gradient of the log-probability and of the entropy with respect to an illegal logit
is 0.0; the distribution over legal actions is invariant to the network's output for illegal actions (adding 50 to
an illegal logit changes nothing). The log-probability recorded at collection (`MaskedCategorical`, float32, −1e9
masking) equals the acting agent's distribution (`NetAgent.probabilities`, float64, −∞ masking) to 1e-6 on real
`m4_lasthit()` observations, so acting, collecting and updating use one policy. Steps with a single legal action (an
absent champion) carry no policy gradient and do not enter the policy statistics: replacing their advantages by 1e3
leaves every gradient and every reported statistic bit-identical. At the environment, `LaneEnv.step` raises on an
illegal action, so a completed run is itself the proof that only legal actions were played; the recorded masks are
checked against the engine's legal sets on the rollouts.

### 3.4 Shared-policy learning from both seats, reward signs, zero sum

On `m4_lasthit()` with 100% current-policy games every slot is trainable, the per-seat transition counts are equal
(exactly, 204,800 each in the smoke test), and seat B's reward is exactly −(seat A's) at every step. Seat B's
observation at every step decodes (`decode_m4`) to the mirror of seat A's game and equals `observe` of the mirrored
state from seat A's perspective: one network sees the same game from both chairs. Rewards are the environment's
`StepResult.rewards[seat]`, scaled once; nothing is negated twice (the M3 tests of this remain).

### 3.5 Bootstrapping at rollout boundaries and time limits

With segments of 30 turns and a game length of 12, truncations occur exactly at each environment's seeded first
length and then every 12 turns, and never a termination. Within a game `next_values[t]` is `values[t + 1]`. At a
truncation the recorded bootstrap equals the network's value of the **pre-reset final observation**, verified
independently: environment 0's first game is replayed on a fresh `LaneEnv` seeded from the collector's deterministic
episode stream with the recorded actions, its final observation evaluated, and the value matches for both seats. At
a segment end the bootstrap equals the value recorded for the first step of the next segment (same observation, same
network, no update in between). The truncated step's advantage is its own TD error (no λ-recursion across games).
These are the continuing-time semantics of design §10.4; what they do *not* guarantee (bootstrapping approximates
the continuation value with the critic's error) is unchanged and restated in §7.

### 3.6 Historical snapshots and opponent sampling

Snapshots are deep copies in `eval` mode with gradients disabled; after a PPO update on a rollout of historical
games on `m4_lasthit()` the snapshot's weights are bit-identical. Snapshots act through NumPy sampling from their own
per-episode generator, never through the torch generator, so the historical opponent never perturbs the learner's
random stream. The matchup sampler's statistics (current fraction, uniform seat, uniform pool draw, the empty-pool
bootstrap) are the M3 tests. In the smoke test the 80/20 run had 71 historical games in 436 (16%, the first 25,600
transitions being current-only by construction) and seat shares 49.8 / 50.2%.

### 3.7 On-policy collection, reproducibility, checkpoints

A rollout is collected with one fixed parameter vector and the recorded log-probabilities are that network's (M3
test). Two `train_shared` runs with the same configuration and seed, in the same process (test) or in separate
worker processes (smoke test), give bit-identical weights (equal SHA-256 digests) and identical histories; a
different seed differs. Checkpoints: `final.pt` reproduces the action distribution of the trained network;
`run.json` round-trips to the exact `GameConfig`; snapshot files carry the configuration; an altered checkpoint is
refused by its digest and version-1 files still load.

### 3.8 Things noted and left as they are

- `approx_kl` early stopping (`target_kl`, off by default) tests the last minibatch's KL rather than the epoch mean
  (CleanRL's convention). Documented here; unused.
- The learning rate anneals over the run's budget, so runs of different budgets are not prefixes of each other
  (already stated in the long self-play study).
- The critic is not constrained to be antisymmetric under the seat mirror, so V̂(s0) is not forced to 0. In the
  smoke test it is within ±0.6 resources of 0 at a return scale of ~1.1 (std of the λ-returns), which is the
  expected order. A symmetric critic (averaging V̂(o) and −V̂(mirror(o))) is a possible variance reduction, not a
  correctness fix.

## 4. The M4.3a smoke test on `m4_lasthit()`

Driver `lanegame.experiments.rl.m43a_smoke` (`python -m lanegame experiments-rl m43a_smoke`); results
`results/rl/m43a_smoke.md` and `.json`; checkpoints `results/rl/checkpoints/m43a/` (untracked). Shared-policy
self-play with `PPOConfig.continuing` (γ = 0.99, λ = 0.95, segments of 128 turns, games of 512 turns with seeded
staggered first games, reward scale 0.1, value clip 0.5, 16 environments, snapshots every 25,600 agent transitions,
pool 8, one torch thread per run, five runs trained in parallel processes). Budget: 409,600 agent transitions for
the two conditions, 102,400 for the three reproducibility runs; about 2.3M environment turns in all, 5 minutes.

| run | condition | agent transitions | env turns | wall-clock | transitions/s | env turns/s | collect share |
|---|---|---|---|---|---|---|---|
| sp100_s0 | 100% current | 409,600 | 204,800 | 76 s | 5,388 | 2,694 | 0.74 |
| sp80_s0 | 80% current / 20% historical | 410,001 | 223,232 | 87 s | 4,700 | 2,559 | 0.77 |
| repro_s0a, s0b, s1 | 100% current | 102,400 | 51,200 | 21 s | ~4,800 | ~2,400 | 0.76 |

**Numerical health.** Every loss, gradient norm, KL, clip fraction and critic statistic is finite in every update.
Pre-clip gradient norms average 0.4 (max 1.05 in sp100, one spike of 3.96 in sp80), clip fraction 0.03, mean
approximate KL 0.003 (max 0.011): a well-behaved PPO regime. Explained variance of the critic rises from −2 at the
first update to 0.70 (sp100) and 0.75 (sp80). The critic is on the return scale: mean |V̂| of the last rollouts 0.02,
largest |V̂| in the run 10.7 resources, the standard deviation of the λ-returns 1.2, mean |r| per turn 0.5–0.6; the
guaranteed bound is |V| ≤ Ω = 107. V̂(s0) is −0.25 (sp100) and +0.03 (sp80), against the symmetric value 0.

**Seats, masks, time.** sp100 trained on 204,800 transitions per seat (exact); sp80 on 204,280 / 205,721 with
36,463 historical transitions. 39,866 (10%) of the sp100 transitions had a single legal action (an absent champion)
and carry no policy gradient. Every finished game ended in exactly one truncation (400 and 436 games), never a
termination. Masks: the environment raises on an illegal action, and none was raised.

**Reproducibility.** `repro_s0a` and `repro_s0b` (seed 0, separate processes) have the same weights digest
`69069a2d…` and identical histories; `repro_s1` differs. Checkpoints record torch 2.14.1 and the commit.

**Throughput.** About 5,000 agent transitions/s per process under five-process load (2,500–2,700 environment
turns/s), 74–77% of the time in collection: the environment step (≈250 µs) dominates, the 2 × 128 MLP is cheap. A
single process runs about 20% faster (the CLI check gave 4,800 sps with 4 environments). At this rate 1M agent
transitions take 3–4 minutes per run and 10M about 35 minutes, so the M4.3b pilot fits a workstation; the lever for
more is processes, not vectorized inference.

**Evaluation readout** (protocol E1, W = 460, guaranteed truncation bound 1.05 resources, 16 paired episodes per
matchup, 95% paired CI). Both trained policies beat every scripted natural-wave reference by 10–19 resources
(`hold_farmer` +11.3 / +11.2, `slow_pusher` +10.4 / +10.4, `trading_pusher` +17.4 / +18.9) and the untrained initial
network by +11.6 / +12.0; the initial network loses to the farmers (−5 to −7). sp80 beats sp100 by 0.80 ± 0.47 at
this budget. In self-play both policies Clear on 82% of present turns, Hold on 3–8%, Thin on 10–15%; Hold-position
54–58%, Trade 27%, Yield 11–14%, Safe 4–5%, Recall 0%; Pressure 3–4%; about 90 connected trades and 17 deaths per
side per 460 turns. Entropy fell from 2.58 to 0.8 within the budget.

All of this is an integration readout. The budget is a tenth of a pilot, the references are the weak scripted suite,
and no strategic claim is made. What it does say: the trainer learns something on this environment (it moves far
from the initial policy in a consistent direction in both conditions), the evaluator and its tail bound work on the
new preset, and the behaviour it learns first (clear and trade) is the thing to watch in M4.3b (§7).

## 5. What is verified and what is only empirically supported

**Verified** (exact identities or independent references, by test):

- the adapter's trajectories equal the native environment's (exact), on random actions, on `play_episode` games and
  from custom start states; PettingZoo's official API tests pass on every M4 preset and on the AEC conversion;
  the global state is lossless;
- GAE equals the forward-view λ-return with random terminations and truncations; the PPO loss, entropy and
  parameter gradients equal an independent implementation; masking is exact in probability and gradient; collecting
  and acting use one policy; single-choice steps carry no gradient;
- zero-sum antisymmetric rewards, both seats' transitions, mirror observations; truncations exactly at the game
  times; bootstraps from the pre-reset observation (replayed independently) and across segment boundaries; no
  λ-leakage across games; snapshots frozen and isolated; bit-identical reproducibility across processes; checkpoint
  digests and round trips.

**Empirically supported** (one bounded run per condition): finite and well-scaled losses, gradients and critic;
explained variance; throughput; the evaluation readout; the behaviour profile. **Approximations by design**: the
bootstrapped truncation (design §10.4); the uniform historical sampling and the 80/20 fraction as a method choice.
**Not established**: that PPO self-play discovers the intended laning strategies on `m4_lasthit()`; that the PPO
hyperparameters are suitable for a long run.

## 6. Files, tests, reproduction

| file | change |
|---|---|
| `lanegame/env/pettingzoo_adapter.py` | M4 champion states, `StartState` starts, legal sets in infos, `aec()`, `cfg` / `max_turns` properties, documentation |
| `lanegame/rl/ppo.py` | `grad_norm`, `grad_norm_max` statistics |
| `lanegame/rl/shared.py` | per-seat, truncation, termination and single-choice counts; critic, return and advantage statistics |
| `lanegame/rl/train.py` | critic, return, truncation statistics; `net.eval()` after the update |
| `lanegame/rl/agent.py` | checkpoint format 2: weights digest (verified on load), provenance; `weights_digest`, `checkpoint_info` |
| `lanegame/rl/__init__.py` | exports |
| `lanegame/experiments/rl/m43a_smoke.py` | the smoke driver (train, repro, evaluate, report) |
| `lanegame/__main__.py` | `rl-shared --model`, `experiments-rl m43a_smoke` |
| `tests/test_pettingzoo_m4.py` (11), `tests/test_rl_reference.py` (10), `tests/test_rl_m4_lasthit.py` (6) | new |
| `README.md`, `docs/README.md`, `docs/rl_research_roadmap.md` §Z, this report | documentation |

Tests: the three new files run in about 8 s; `tests/test_rl.py`, `test_rl_shared.py`, `test_rl_selfplay.py`,
`test_m4_time.py` and the two legacy adapter tests pass unchanged (49 + 2). The full suite at this commit: **898
passed, 4 skipped (slow enumerations, `--runslow`), 2 expected failures (recorded design findings), in 7 min 33 s.**

Reproduction:

```bash
python -m pytest tests/test_pettingzoo_m4.py tests/test_rl_reference.py tests/test_rl_m4_lasthit.py
python -m lanegame experiments-rl m43a_smoke                     # ~5 min, 5 worker processes
python -m lanegame.experiments.rl.m43a_smoke train --workers 2   # stages: train | repro | evaluate | report
python -m lanegame rl-shared --model m4_lasthit --current-fraction 1.0 --steps 409600 --out results/rl/checkpoints/x
```

## 7. Unresolved concerns and recommendations before M4.3b

1. **Early entropy collapse and the clear-and-trade policy.** Entropy fell from 2.58 to 0.8 in 400k transitions
   and both conditions converged on clearing 82% of the time with 17 deaths per side per 460 turns. Three
   possibilities, not distinguished by this test: a genuine early optimum of the game (dying is cheap: the kill
   bounty is 0.5 resources and death costs 3 turns of absence, against 1 resource per 8-HP wave spawning every two
   turns, so a death is worth about 2 resources to the opponent while a cleared wave denies a similar amount every
   few turns), an exploration failure (entropy coefficient 0.01 with 46 actions), or a reward
   scale effect (λ-returns of std 1.2 in scaled units; advantages are normalized per minibatch, so this is
   unlikely). Recommendation: in the pilot, log the entropy schedule and run an entropy-coefficient control (0.01
   vs 0.03) at the same budget; audit death economics with the diagnostics tools (what a death costs in discounted
   farm and bounty) before changing any parameter; judge the policies against the sampled equilibria of the
   M4.2e-C report §6.4 (Clear + punish when ahead, slow pushes at equal power), not against the natural suite.
2. **Evaluation references.** The scripted natural-wave suite is weak (every trained policy beats it by 10+). The
   pilot needs the zoning-harness policies of `experiments/m4/zoning_natural.py` as references and an exploitability
   measurement (`approximate_best_response` is finite-horizon; a continuing-time attacker is the same trainer with an
   `OpponentSampler` of one frozen `NetAgent`, evaluated with `evaluate_discounted`; wire that up before the pilot).
3. **Headline window.** W = 460 is the development window (bound 1.05). Headline numbers use W = 1000 (bound
   0.005), at about twice the evaluation cost.
4. **Bootstrapping is an approximation** (design §10.4): with an imperfect critic the λ-return targets are biased
   by γ(V̂ − V) at every game boundary, propagated with weight (γλ)^j. The game length 512 ≫ 1/(1 − γ) keeps few
   transitions near a boundary. If a long run shows value drift, lengthen `game_length` before touching γ.
5. **Historical sampling.** Uniform over the last 8 snapshots at 20% is our own choice (OpenAI Five weighted by
   quality). The 80/20 vs 100/0 comparison is preserved in the driver and should be run at pilot scale with 3+
   seeds before either is adopted.
6. **Hyperparameters are M3's.** Learning rate 3e-4 annealed, 4 epochs, minibatch 256, clip 0.2, reward scale 0.1
   and value clip 0.5 (provisional in `PPOConfig.continuing`). The smoke test shows them stable, not tuned. A small
   grid (reward scale, entropy coefficient, learning rate) at pilot budget is the cheap insurance.
7. **Throughput.** 5k transitions/s per process; the environment step is the cost. Parallel processes (the drivers'
   pattern) scale it; a vectorized environment would need a rewrite of the mechanics and is not recommended.
8. **Optional external cross-check.** One external PPO through the PettingZoo adapter on `m4_lasthit()` at matched
   budget, as an end-to-end check of the training loop (§2.3).

None of these blocks the pilot. The infrastructure is correct as far as tests can establish it; the open questions
are about what the game and the method produce, which is the question M4.3b exists to answer.
