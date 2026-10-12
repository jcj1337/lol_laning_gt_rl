# Overnight implementation log: M2b–M2e, M3a–M3c

Running log of the autonomous session started 2026-10-11. Newest entries at the bottom of each section.
Work happens on the local branch `m2-m3-autonomous` (one commit per verified milestone, nothing pushed).

## Status

| Milestone | Status | Commit |
|---|---|---|
| M2b Recall and absence | completed and verified (39 champion + 65 env tests) | see git log |
| M2c Stochastic trading and skill | completed and verified (52 champion + 65 env tests) | see git log |
| M2d Ganks, death, respawn | completed and verified (67 champion + 65 env tests) | see git log |
| M2e Validation and benchmarking (gate) | completed and verified (full suite 444 passed, 5 min 20 s; 0 invariant violations in 120,960 turns) | see git log |
| M3a PPO vs scripted opponents | completed and verified (14 RL tests; beats best scripted responses) | see git log |
| M3b Self-play with frozen snapshots | completed and verified (6 tests; 3 seeds x 2 variants + legacy exact) | see git log |
| M3c Evaluation and approximate best responses | completed and verified (tests; held-out evaluation, scratch + warm BRs) | see git log |

## Session start

- Repository state verified: branch `main` clean at `08865df` ("phase m1 and m2a").
- `tests/test_champions.py`: 25 passed (0.7 s). Full-suite baseline: **395 passed in 5 min 33 s** (matches the M2a report).
- PyTorch was not installed; installed `torch 2.14.1+cpu` into `.venv` (needed for M3, optional extra only).
- Shell note: heredocs containing backticks fail in this Bash wrapper; patch scripts go to the scratchpad.

## Log

### M2b Recall and absence

Implemented in `lanegame/champions.py` (+ `config.py`, `env/core.py`, `env/agents.py`, `env/trace.py`).

- State: `absent` = number of upcoming turns the champion must sit out. Recall sets `recall_turns - 1`
  (the Recall turn itself is the first missed turn), each absent turn counts it down; the champion is at
  `-T` while away and gets full HP when the timer reaches 0. Default `recall_turns = 2`: Recall on t,
  forced ABSENT on t+1, back at the tower with full HP on t+2. Configurable (tested for 1, 2, 3).
- Legality: one Recall code (`champ_act(SP, B_RECALL)` = 12) when present and `pos <= recall_line (-1)`;
  `{ABSENT}` is the whole legal set while absent; ABSENT is never legal otherwise.
- **Deviation from the brief (documented in the module):** the brief suggested disallowing Recall under
  an immediate credible threat. With a state-based rule ("enemy champion in contact at turn start"),
  exploration showed a Safe champion at -1 facing a Holding opponent at the lane (0) could *never* recall
  (-1 and 0 are within trade range, and no behavior walks voluntarily to -2). Because actions are
  simultaneous, whether the threat materializes is only known at resolution, so instead: an enemy
  **Trade that reaches the recaller interrupts the recall** (it stays where it was, channeling: no farm, no
  wave contribution, no zoning, traded as a vulnerable defender, gankable with Safe's term). A dive can
  interrupt a recall at the tower. This is one conditional rule (like Yield's evasion) and creates a
  recall-versus-trade guessing game.
- Away sides: wave plays Slow Push with no champion reach, no objective, no farm, no zoning, no trade,
  gank probability 0; a present Trade whose opponent is absent behaves like Hold. Exposure of an away side
  is reported as -T.
- Economics (no recall penalty): from a constructed state where A's 8 HP crash sieges B's tower, a recall
  costs 0.00 resources over the next 6 turns versus staying; from a state where B's 8 HP wave walks into
  A's tower it costs 0.62 (test `test_well_timed_recall_costs_less_than_a_poorly_timed_one`).
- Tests: new `tests/test_champion_recall.py` (14 tests); three M2a tests updated (Recall is no longer
  reserved). 39 champion tests pass in 0.6 s. env-demo champion throughput 7,500 turns/s.
- Committed as `acabf74`.

### M2c Stochastic trading and mechanical skill

- Hit chance of a connected trade i -> j:
  `h = clip(0.55 + 0.4 (m_i - m_j) + beta[beh_j] + 0.05 (sigma_i - sigma_j) - 0.15 [j in own tower cell], 0.05, 0.95)`,
  `beta = (Hold 0, Yield -0.25, Trade 0, Safe -0.40, Recall/channel 0)`; damage 0 / 1 / 2 with
  `P = (1-h, h(1-q), hq)`, `q = clip(0.3 + 0.2 (m_i - m_j), 0, 1)`. Roadmap E.5 defaults, except that the
  2-HP chance uses the skill *difference* (the roadmap had `q2_skill * m_i`), so a skilled defender also
  mitigates. Every term is a `GameConfig` field (`trade_hit_*`, `trade_double_*`).
- Skill enters only these two probabilities (test: positions, modes, lane, farm and gank risk are identical
  under different skills).
- Tower shot: a connected trade ending in the enemy tower cell (a dive) takes 1 HP with p = 1.0, or 0.5 if
  the diver's crashed minions are under that tower (start of turn or crashing this turn).
- A Hold defender does not hit back; Trade vs Trade both roll from start-of-turn HP.
- All chance events of a champion-mode turn come from one fixed-length vector of 10 uniforms per turn
  (`champions.N_CHANCE`, layout `U_TRADE_HIT`, `U_TRADE_DOUBLE`, `U_TOWER`, `U_GANK`, `U_GANK_DAMAGE`), drawn
  whatever the actions: identical chance draws turn by turn for any two policies (common random numbers).
  `resolve_turn(..., u=None)` means no chance event. Legacy configurations keep their two gank draws.
- HP floored at 1 (temporary nonlethal clamp, removed in M2d). Ganks still the legacy resource loss.
- Matchups (300 episodes, K = 32): Trade vs Hold lands 17.2 of 31 connected trades, 22.3 HP dealt;
  Trade vs Trade symmetric (23.1 / 22.5 HP); Yield and Safe evade all 31 trade attempts.
- Tests: new `tests/test_champion_combat.py` (13 tests: formula defaults and clipping, skill monotonicity
  for offense and defense, behavior ordering, execution-only skill, opportunity cost, evasion, simultaneity,
  interrupted-recall and partial-escape terms, tower shot free/busy, 10,000-draw distribution check within
  4.5 SE, mirror equivariance with mirrored chance, no guaranteed damage, reproducibility and CRN).
  52 champion tests pass in 2.6 s.
- Committed as `e83618d`.

### M2d Ganks, death, respawn

- Champion mode has its own gank-occurrence parameters (`champ_gank_*`); the legacy `gank_*` fields and
  the legacy expected-loss model are untouched. Occurrence:
  `p = clip(0 + 0.12 (e+T)/2T + beh{H .01, Y .01, T .03, S 0} + 0.02 [Shove, non-Safe] - 0.01 dsigma, 0, 0.30) * (1 - 0.75 [vision])`.
  The legacy unconditional base 0.15 (harmless as an expected loss because it cancelled in the zero-sum
  payoff) would have produced ~7 HP-damaging ganks per player per game; with base 0 a Hold champion at the
  lane is ganked ~2.1 times per 32-turn game, a Safe one ~1.0, and a champion under its own tower almost
  never. Vision at -1 saves 0.0225 per turn: not mandatory when safely positioned.
- Damage given a gank: `Binomial(4, p_e)`, `p_e = clip(0.2 + 0.6 (e+T)/2T - 0.2 [Yield/Safe] - 0.2 (m - 0.5), 0, 1)`
  (0 to 4 HP; mean 0.8 at the own tower, 2.0 at the lane, 3.2 under the enemy tower). 4 < 6 HP, so a
  full-HP champion never dies to one gank. Skill enters gank damage (escape execution) with a small term:
  the roadmap had left this open; the brief asks skill to improve defensive execution. Vision changes only
  occurrence (tested).
- Death: all damage of the turn (trades, tower shot, gank) is summed; HP <= 0 -> dies at the end of the
  turn (keeps that turn's farm), `absent = death_turns = 3`, HP 0 while away, ABSENT on t+1..t+3, back at
  the own tower with full HP on t+4. Simultaneous deaths possible. Temporary M2c HP floor removed.
- Reward in champion mode: farm + plates + `kill_bounty` (default 0) differences. The legacy `gank_loss`
  term is not charged (no double counting); `info["gank_payoff"]` is always 0 in champion mode.
- Scripted matchups (400 episodes, K = 32, no recall logic in these agents): Hold vs Hold 0.05 +- 0.04,
  2.1 ganks, 0.33 deaths per player; Trade vs Hold -0.58 +- 0.06 for the trader (Hold dies 2.7 times but
  the trader halves its own farm on 31 connected turns); Trade vs Yield +12.0, Trade vs Safe +12.9;
  Hold vs Safe +9.0; Shove-Hold vs Hold -3.9. Payoff std fell from ~16 to ~1 (no 5-resource gank loss).
- Tests: new `tests/test_champion_ganks.py` (15 tests: occurrence defaults and ordering, exposure kept while
  retreating, damage distribution and inputs, no one-gank death from full HP and possible zero damage,
  vision only on occurrence, 10,000-draw binomial check, ganks under tower, exact death timeline,
  simultaneous deaths, damage sources adding up, recall vs death, episode end during death, rewards without
  gank loss and with bounties, reproducibility, death cost depending on wave state). Three earlier tests
  updated. 67 champion tests + 65 env tests pass.
- Committed as `bcb6ba6`.

### M2e Validation and benchmarking (gate)

New: `lanegame/experiments/champions/` (`invariants`, `tournament`, `skill`, `concerns`, `benchmark`;
`python -m lanegame experiments-champions all`, results in `results/champions/`), the scripted suite
`lanegame.env.agents.champion_suite()` (hold_farmer, yield_farmer, aggressive_trader, safe_player, shover,
conservative_recaller, opportunistic_trader, vision_player, punisher; `ChampionScriptAgent` with low-HP recall,
opportunistic trading, stacking and absence punishment rules) and `tests/test_champion_validation.py` (7 tests).

- **A. Correctness.** Turn-by-turn checker of every rule (bounds, masks = legal sets, absence timeline,
  movement, HP and damage accounting, zero-sum reward decomposition, no remote farming, no remote siege, away
  champions take no part, observation purity/injectivity/bounds, mirror equivariance with mirrored chance,
  replay from recorded chance and from seeds): **120,960 turns over three configurations, 0 violations**.
  Seat symmetry: hold_farmer self-play +0.006 +- 0.027 over 8,000 independent episodes.
- **B. Behavior.** 15-agent round robin (225 ordered pairs x 400 episodes, CRN). Final defaults: TRADE-type
  agents crush passive/evasive play (+8 to +12.5 vs Yield/Safe) but lose slightly to careful farmers
  (-0.2 to -1.1 vs hold_farmer / punisher / conservative_recaller); shoving and Safe are clearly bad;
  meta-Nash over the scripted agents: punisher 0.56, HOLD 0.41, TRADE 0.03.
- **C. Skill.** Hit rate 0.51 -> 0.63 -> 0.76 -> 0.92 for A's skill advantage 0 -> 0.3 -> 0.6 -> 1.0, damage
  per gank falls monotonically; payoff effect is strategy-dependent (a 0.35-skill trader still beats a
  0.65-skill Yield farmer by +7.9; skill barely matters there because trades never connect).
- **D/E. Concerns and degenerate incentives** (`results/champions/concerns.md`); three evidence-based
  changes to the champion defaults, nothing in the legacy models:
  1. `kill_bounty` 0 -> **0.5** (half a wave; real kill gold is ~2.4 early waves). With 0 a death cost only
     ~1-1.5 resources (about the same as a recall plus the walk back), Trade lost to every non-evasive agent,
     never recalling beat recalling at <= 2 HP, and the scripted meta-game was 100% naive Hold: HP damage
     had almost no strategic consequence (an explicitly listed degenerate incentive). With 0.5, trading is
     viable but exploitable and the meta-game mixes; the absence remains the main cost of a death.
  2. `retreat_farm_reach` (new) **2**: a threatened Yield collects kills within 2 cells at Safe efficiency.
     With 1 a Yield champion zoned by a persistent trader kept 1.2 of 15.3 available resources (-11.4 per
     game); the roadmap intended it to "farm at Safe efficiency". Now -8.2; no other matchup changes.
  3. Gank-damage defense now requires *defensive play* (mode safe or retreat) instead of the Yield/Safe
     label: an unthreatened Yield stands and farms exactly like Hold but got half of Hold's gank damage for
     free (YIELD beat HOLD by +0.41 with identical play). Now YIELD vs HOLD = 0.00 exactly.
  Kept after measurement: `trade_farm_factor 0.5` (1.0 makes Trade dominant: +1.9 vs a recaller, +3.9 vs
  naive hold; 0.25 makes it useless), `farm_reach 1`, `champion_speed 2` (1 lets traders pin holders:
  +2.4), tower shot (worth ~1.4 resources to a tower-hugging defender), gank model (removing ganks changes
  forward strategies by ~1 resource, so exposure matters), Safe zones nothing, Trade chase ignores minions
  (documented). Degenerate probes: recall spam -13.1, permanent Safe -8.7, Vision is used but rarely (sieges
  are rare when the opponent contests the wave), always-Shove -4.7.
- **F. Throughput.** 6,100-6,900 turns/s uncached, ~11,500 with the lane cache (scripted agents);
  raw `env.step` 7,200 / 9,200; legacy siege model through the env 11,500.
- Committed as `e311586`.

### M3a PPO against scripted opponents (implementation)

New package `lanegame/rl/` (PyTorch, optional extra `rl`; `lanegame.env` never imports it):

- `network.py`: `PolicyValueNet` (2 x 128 tanh shared trunk, policy head over 46 logits, value head;
  orthogonal init, policy gain 0.01); `MaskedCategorical` (illegal logits -> -1e9: probability exactly 0,
  never sampled, zero entropy contribution; a forced single action has log-prob 0 and zero gradient).
- `ppo.py`: `PPOConfig` (defaults: 16 envs x 64 steps, 4 epochs, minibatch 256, Adam 3e-4 annealed, gamma 1,
  lambda 0.95, clip 0.2, value clip 2.0, vf 0.5, entropy 0.01, grad norm 0.5, unscaled rewards, 1 torch
  thread); `compute_gae` (terminated: no bootstrap; truncated: bootstrap with V(final obs); advantages stop
  at episode boundaries); `ppo_update` (clipped surrogate and value loss, entropy bonus, grad clipping;
  forced-action steps excluded from policy/entropy averages, kept for the value loss).
- `train.py`: `Collector` (n_envs lock-step environments, one batched forward for the learner, random seat
  per episode, fresh opponent per episode from an `OpponentSampler`, per-episode SeedSequence children,
  optional per-episode skill randomization) and `train_ppo` (history.jsonl, final.pt, eval hook, update hook).
- `agent.py`: `NetAgent` (frozen copy; exact masked softmax as `distribution`, so `as_policy` gives exact
  values on legacy configs), checkpoints (`save_checkpoint` / `load_checkpoint` / `load_agent`).
- `evaluate.py` (M3c): `head_to_head` (both seats, same seeds, skill follows the player), `round_robin`,
  `best_scripted_response`, `approximate_best_response`.
- `selfplay.py` (M3b): `SnapshotPool` (bounded, frozen, keeps the initial anchor), `SelfPlaySampler`
  (p_scripted / p_latest / uniform over the pool), `train_selfplay` (snapshot every N updates, files with
  metadata).
- CLI: `experiments-rl scripted|legacy|selfplay|evaluation`, `rl-train`, `rl-selfplay`, `rl-eval`, `rl-br`.
- Tests: `tests/test_rl.py` (9) and `tests/test_rl_selfplay.py` (5), 14 tests in ~13 s.
- Smoke test: 10k steps vs yield_farmer: training return -4.3 -> -0.9 in 3.6 s; 100k steps: +8.6 +- 0.15 in
  both seats (best scripted response, always Trade: +8.2), ~6,600 learner steps/s.

### M3a results (`results/rl/m3a_scripted.md`, 4.6 min on 10 processes)

- Single opponent, 102,400 steps x 3 seeds, held-out evaluation in both seats (500 episodes per seat):
  vs yield_farmer +8.28 +- 0.17 (best scripted response, always Trade: +8.17); vs aggressive_trader
  +4.25 +- 0.99 (best scripted: conservative_recaller +0.65); vs hold_farmer +1.24 +- 0.88 (best scripted:
  naive hold +0.22). Initial (near-uniform) policy: -4.0, -2.1, -4.3. Every seed beats the best scripted
  response; ~30 s training per run.
- Mixture of 13 scripted agents, 204,800 steps x 3 seeds: average +4.32 +- 0.27 over the 13 opponents
  (best single scripted agent `trade` +3.56; uniform -2.16). Beats every careful farmer (hold_farmer +1.21,
  punisher +1.03, conservative_recaller +2.73) that no scripted agent beats by more than ~0.2 (except
  conservative_recaller by hold +0.98); does NOT beat the naive persistent `trade` (-0.12; the best scripted
  response to it is conservative_recaller +1.11). Behavior mix: Trade 70%, Yield 12%, Hold 10%.
- Status: M3a completed and verified (14 RL tests; learning demonstrated against every opponent tested).
- Committed as `6e0597b`.

### M3b results (`results/rl/m3b_selfplay.md`, 3.3 min on 9 processes)

- Champion game, equal skill, 307,200 steps x 3 seeds from the M3a mixture policies (30% scripted, 35% latest
  snapshot, 35% pool; pool 8; snapshot every 25 updates): steady improvement against every fixed reference
  (held-out seeds): vs hold_farmer +1.28 -> +2.13, vs aggressive_trader +1.73 -> +2.84, vs punisher
  +1.13 -> +1.98, vs the frozen initial (M3a) policy +1.32 -> +2.17 (mean of 3 seeds, CIs +-0.3 to 0.5).
- Skill-randomized variant (skills {0.2, 0.5, 0.8} per seat per episode): at 0.5/0.5 vs hold_farmer +1.55,
  vs the initial policy +2.35 after 307k steps.
- **Negative result (certified):** self-play from scratch on the legacy siege model (K = 16, 204,800 steps):
  the EXACT exploitability of the snapshots falls from ~3.7 (uniform) to 2.52 after 20k steps and then
  RISES to 3.43 (per seed 4.09 / 3.23 / 2.96); greedy (argmax) snapshots are at 4.97 throughout. The legacy
  equilibrium is mixed; the PPO policy sharpens (entropy falls) and learns to beat its pool, which is not
  the same as approaching the equilibrium. Expect the champion-game self-play policies to be exploitable.

### Legacy exact PPO benchmark (`results/rl/m3a_legacy_exact.md`, 0.8 min)

- PPO (102,400 steps x 3 seeds) against single legacy scripted opponents on the siege model K = 16; every
  value exact (network as a Markov policy, backward induction): share of the exact best-response gain over
  uniform play achieved by the stochastic PPO policy: passive_safe 23-25%, freeze_contest 31-33%,
  aggressive 67-75%. Exact best responses (+6.28 vs passive_safe) need long, specific siege sequences
  (crash small waves, establish vision, pressure); PPO with this budget does not find them. Greedy
  versions do better (+1.02 vs passive_safe) than the stochastic policies (+0.23). Learned policies have
  exact exploitability 2.3-3.9 (uniform 3.74). Calibration: PPO best responses are loose lower bounds.
- Variance reduction (new, optional, legacy configurations only): `LaneEnv(gank_reward="expected")` pays the
  gank term as its conditional expectation; in the legacy game a gank changes nothing but the payoff, so
  expected returns and policy gradients are unchanged (Rao-Blackwellization; every deterministic-pair episode
  then equals the exact value, test `tests/test_env_expected_gank.py`). Share of the exact BR gain with it:
  passive_safe 37-39% (realized 23-25%), freeze_contest 46-53% (31-33%), aggressive 94% (67-75%).
- Legacy self-play with the variance-reduced reward: exact exploitability rises to 4.84 (worse than uniform
  3.74) at entropy 0.01 (final entropy 0.16), ~3.8 at entropy 0.05. Noise is not the cause; frozen-snapshot
  self-play PPO does not approach the mixed equilibrium here (recommendation: PSRO meta-Nash / average policy).

### M3c results (`results/rl/m3c_evaluation.md` 6.9 min, `results/rl/m3c_br_warm.md` 2.5 min)

- vs 13 scripted agents (held-out seeds, both seats): M3a mixture +4.30 (worst -0.11), equal-skill self-play
  +4.81 (worst +1.19 +- 0.62 over seeds), skill-randomized self-play +4.78 (worst +0.77): self-play policies
  beat every scripted agent.
- PPO cross-play: sp_s1 and the three sprand policies form a strong cluster (~+1.4 average, ~0 among
  themselves); sp_s0 / sp_s2 lose ~2 to them; all self-play policies beat the M3a policies by ~2.
- Each self-play policy beats its own earlier snapshots by monotonically less (+2.0 vs its start, 0 vs
  itself): no cycling visible within the pool.
- Skill: payoff rises ~1.57 per 0.3 of skill advantage (against hold_farmer and in self-play), but the
  learned Trade share is flat (0.54-0.55) over all 9 skill pairs: no skill-dependent aggression was
  learned at this budget. Exposure differences track deaths (away turns count as -2), not positioning.
  A 0.2-skill equal-skill-trained PPO policy beats the 0.5-skill hold_farmer by +1.21 +- 0.19, but not
  reliably the 0.8-skill one (+0.28 +- 0.64).
- Approximate exploitability (max of from-scratch and warm-started PPO responses, 102,400 steps, held-out):
  M3a mix +2.05 / +2.67 / +2.45; sp +2.67 / +0.25 / +3.24; sprand +0.46 / +0.11 / +0.27. From-scratch
  responses could not reach parity against the robust targets (-0.5), the warm start (copy of the target)
  can; both are lower bounds, not certificates. Scripted calibration: hold_farmer +0.61 (best scripted
  +0.19), punisher +0.48 (best scripted +0.04).
- Figure: `results/rl/fig_learning_curves.png`.
- Total RL compute: ~22 min wall clock on 9-10 processes (~3 CPU-hours), well inside the budget.

## Final state

- Final full suite: **462 passed in 5 min 29 s** (395 pre-existing + 49 champion + 18 RL/environment).
- Report: `docs/rl/m2_m3_implementation_report.md`; roadmap section N points to it; README updated.
- Nothing pushed; work is on the local branch `m2-m3-autonomous`. Network checkpoints are in
  `results/rl/checkpoints/` (git-ignored, regenerable with `python -m lanegame experiments-rl ...`).
- Remaining work (not started, out of scope): M3d PSRO (recommended next), M4 economy, M5 opponent modelling,
  M6 dashboard.
