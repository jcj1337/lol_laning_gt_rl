# Strategic diagnostics and robust self-play: progress log

Branch `research/openai-five-diagnostics` (from `research/openai-five-selfplay` at e34252f). Session of 2026-10-11.
Final report: `docs/openai_five_diagnostics_report.md`. Results: `results/rl/diagnostics/`; checkpoints
(git-ignored) `results/rl/checkpoints/diag/`.

## Phase 0: inspection (done)

Read the roadmap, the M2/M3 report, the OpenAI Five report, adaptation notes and progress log; `champions.py`,
`config.py`, `mechanics.py`, `env/`, `rl/`, `experiments/rl/`. All checkpoints of the previous study are present
locally (`results/rl/checkpoints/of5/`, 86 MB). Baseline full suite: **476 passed** (7.2 min).

Mechanics facts that matter for the diagnosis (from the source, confirmed in traces):
- HP enters nothing but death: trade hit chances, farm, zoning, gank probability and wave contribution do not
  depend on HP. A champion at 1 HP farms and trades exactly like one at 6 HP until it dies.
- Death: absent 3 turns, then back at its tower with full HP; opponent gets the 0.5 kill bounty. Recall: 2 turns
  away (recall turn + 1), back with full HP, no bounty. Recall is legal only at cell <= -1.
- When both champions play Slow Push, the lane is a deterministic 2-cycle at the lane cell (x = 0): fronts of
  3 HP / 1 HP alternate, 0.5 resources available per side per turn. Hold's cell is its wave's cell, so a champion
  is past the lane only if its wave is (Shove) or it chases.

## Phase A: action audit (done; `results/rl/diagnostics/audit.md`, 1.5 min)

New: `lanegame/diagnostics/` (NumPy only): `audit.py` (`audit_games`: head-to-head protocol with the exact sampling
of `play_episode`, one row per focal turn with legality, the policy's full distribution, state, choice and
outcomes; summaries `legality_table`, `opportunity_table`, `state_table`, `outcome_table`, `economy_table`),
`branch.py` (`Snapshot`, `restore`, `rollout`, `branch`, plans, `ProbCache`). Driver
`lanegame/experiments/rl/diagnostics.py`. Checked: `audit_games` reproduces `head_to_head` payoffs exactly and
`rollout` without a plan reproduces `play_episode` exactly.

Findings (9 main policies + pub_sprand_s1, mirror games, 400 games each):
- Recall legal on 39-43% of present turns (champion at -1/-2), but at HP <= 2 and legal (2-4% of turns) the
  policies put 0.000 on it. 5-6.5 turns per game are spent at HP <= 2, mostly at the lane cell where Recall is
  illegal. Deaths 1.7-2.1 per game; recalls ~0.01.
- Freeze legal 0.02-0.22 turns per game; a siege objective legal 0-4 turns in 400 games; crashes ~0.00-0.01 per
  game. Wave management and siege are almost never available in the visited state distribution.
- Forward positioning: Hold's cell (the farm spot) is past the lane on 0.0-0.9% of turns; there is no action that
  moves a champion past its own wave except chasing a trader or a siege objective.
- Farm: 6.3-6.8 of 15.3 available per game; lost to trading mode (halved farm) 4.0-5.5, to absence 2.3-3.0.
  Enemy farm per turn 0.41 while the focal is away vs 0.15 while present. Payoff per kill difference +1.3 to +1.6.
- Opponent P(Trade) >= 0.5 on 45-69% of turns; Yield 0.4-1.5%, Safe 0.2%.

## Phase B: counterfactual evaluation (done; `counterfactual.md`, 13.7 min on 5 processes)

10 scenarios (6 sampled from mirror games, 2 constructed with a scripted Shove prefix, 2 at unequal skill), every
legal first action + short plans, 120 paired rollouts per candidate, 1,026 states. Bug found and fixed: the
walk-back-and-recall plan kept choosing Safe after its champion died on the way back (the engine's legality check
raised; ~15 min of compute lost); the stage now saves one file per finished job and resumes.
- Recall vs the policy at legal HP <= 2 states: -0.02 to -0.47; walk back and recall at the lane ~0 (-0.18 to +0.19).
- Yield / Safe vs Hold: -0.07 to -0.28. Trade vs Hold: -0.15 to +0.14, intervals include 0 (indifference).
- One-turn Freeze == Slow Push exactly in all 72 constructed Freeze-legal states.
- Siege (constructed; reachable in only 1-12 of 24 attempts): Pressure > Hold (+0.2 to +0.8), Pressure > Vision.
- Best single deviation significant (Bonferroni) in 21-75% of states, +0.08 to +0.33 on average, mostly a Shove,
  but Shove - Slow Push averaged over 963 states is +0.00 / +0.01 / +0.10 (M3b / 100/0 / 80/20).
- `calibration.md`: shared-policy critics match the Monte Carlo values (bias <= 0.18) except at Recall-legal
  low-HP states (+0.38); M3b's critic is optimistic by up to +1.3 in mirror games.

## Phase A.3: HP value (done; `hpvalue.md`, 1.8 min)

HP is worth ~0.25 resources per point, nearly linearly (HP 1 vs 6: -1.24 to -1.39), about the same as being dead
(-1.14 to -1.22, bounty excluded).

## Phase C (done; `overlay.md`, `overlay_late.md`, `attacker_behavior.md`, `snapshot_audit.md`)

- Rule overlays against the unmodified policy (1,000 episodes per seat): recall at HP <= 2 -0.14 to -0.47, at HP <= 1
  ~0; punish-shove -0.3 to -1.0; lane-phase shove -0.1 to -0.5; shove+pressure every turn -2.2 to -2.8; pressure /
  freeze rules ~0 (rarely fire). Added after the attacker audit: Shove (Pressure when legal) from turn 28 beats all 9
  policies (+0.11 to +0.60); from turn 24 ~0.
- The strongest attackers siege (1.3-2.6 plates/game), shove on ~30% of turns, mostly late.
- Snapshot audit: rare actions collapse within 25-50k transitions; sieges never visited afterwards.
- Decision gate: Recall, Yield, Safe, Trade/Hold mixing = reasonable behavior; sieges and wave timing = training
  issue; Freeze = mechanics limitation (design); forward positioning = structural. No mechanics change made.

## Phase D: stronger attackers (done; `robustness.md`)

21 attackers x 1,024,000 steps (34 min on 6 processes), curves evaluated on SeedSequence(50,000) (207 evaluations).
Lower bounds +0.71 to +2.63 (previously +0.11 to +1.15 at 307k); condition means M3b +1.96 +- 1.76, 100/0
+1.75 +- 1.86, 80/20 +1.01 +- 0.64; 11/21 curves unsaturated; Spearman with earlier budgets -0.03 / +0.17;
same-target attackers differ by up to 2.0.

## Phase E: long self-play (done; `long_selfplay.md`)

10 runs (100/0 and 80/20, seeds 0-4, 1,024,000 agent transitions) trained in 7.5 min (7 processes; the
`train_long.md` header says 10 processes: it prints the driver's default). Evaluation: 1,395 head-to-head jobs + 30
audits; 20 warm attackers x 1,024k (finals and ~307k snapshots, seed 920).
- Improvement against fixed learned opponents until ~800k (vs pub_sprand_s1 +0.5 to +0.6; final vs own 307k snapshot
  +0.34 to +0.42); cross-play vs the 9 main policies +0.50 to +0.61.
- No forgetting or cycling (0 of 450 older-wins pairs; monotone 0.98-1.00).
- 80/20 - 100/0: -0.12 +- 0.45 in cross-play (6/25 pairings), -0.33 +- 1.32 in exploitability: no benefit.
- Shove 8-24% of turns in 9/10 runs; still no sieges, no Recall. Skill-conditioned Trade in 9/10 runs.
- Exploitability (one attacker per target): finals below their 307k snapshots in 9/10 runs (80/20 -0.53 +- 0.32);
  outlier long_sp100_s3 (+2.41, siege exploit) is the one run that never learned to shove.
- Also added: `lanegame/experiments/rl/attacker_behavior.py` (where the attackers shove), the late-shove overlay rules
  (`overlay_late`), and fixed Markdown headers containing a literal `|` in three generated tables.

Full suite after all changes: **500 passed** (476 + 24 new; 9.3 min with experiments running alongside).

## Tests

`tests/test_diagnostics.py` (24 tests, ~12 s): replay of a restored snapshot with the recorded chance draws,
snapshot round trip / mirror / side edits, observation helper = environment and mirror-consistent, configuration
mismatch refused, rollout = play_episode, sampling rule, reward accounting and decomposition, illegal plans rejected,
DEFER semantics, branch reproducibility and seeding, fresh rollout seeds, opponent branch-turn action and observation
independent of the focal candidate (spy agent), common random numbers, frozen policies and cached probabilities,
advantage algebra, seat symmetry, audit = head_to_head (payoffs, action frequencies, present/absent attribution),
audit reward decomposition and summaries, player-assigned skills, rule overlay, seed-root separation, Phase D plan
and paired difference, scenario plans legal on absent turns (regression test for the bug above). Mutation check: making a plan skip its seat's uniform draw fails 2 tests.

Lessons: write results JSON before formatting Markdown (a formatting bug lost one audit run); do not mutate modules
on disk while background workers might import them.
