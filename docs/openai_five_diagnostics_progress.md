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

## Phase B: counterfactual evaluation (running)

10 scenarios (6 sampled from mirror games, 2 constructed with a scripted shove prefix, 2 at unequal skill),
every legal first action + short plans, 120 paired rollouts per candidate, both seats following the policy.
First single-state probes (20 rollouts, sp100_s0): one Shove at a 2-cycle phase +0.65 ± 0.13 over Slow Push;
Pressure in a constructed siege +1.0 over Hold; walk-back-and-recall at HP <= 2 on the lane +0.48 over Hold.

## Phase D: stronger attackers (running in the background)

`lanegame/experiments/rl/robustness.py`: 21 attackers x 1,024,000 steps (2 warm per main target, seeds 910/911;
scratch for m3b_s0, sp100_s0, sp80_s0, seed 912), checkpoints at 25.6k ... 1,024k, ~8 min per attacker.

## Tests

`tests/test_diagnostics.py` (23 tests, ~11 s): replay of a restored snapshot with the recorded chance draws,
snapshot round trip / mirror / side edits, observation helper = environment and mirror-consistent, configuration
mismatch refused, rollout = play_episode, sampling rule, reward accounting and decomposition, illegal plans rejected,
DEFER semantics, branch reproducibility and seeding, fresh rollout seeds, opponent branch-turn action and observation
independent of the focal candidate (spy agent), common random numbers, frozen policies and cached probabilities,
advantage algebra, seat symmetry, audit = head_to_head (payoffs, action frequencies, present/absent attribution),
audit reward decomposition and summaries, player-assigned skills, rule overlay, seed-root separation, Phase D plan
and paired difference. Mutation check: making a plan skip its seat's uniform draw fails 2 tests.

Lessons: write results JSON before formatting Markdown (a formatting bug lost one audit run); do not mutate modules
on disk while background workers might import them.
