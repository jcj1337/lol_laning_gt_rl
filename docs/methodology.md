# Methodology

How the project builds its games, solves or trains on them, and measures the result. The rules themselves are in
the [model](model.md); what the methods found is in [findings](findings.md). Section numbers of archived reports refer
to the documents in the [archive](archive/README.md).

## 1. The modelling standard

Every mechanic follows [design_philosophy.md](design_philosophy.md), whose standard is to simplify how League works
without changing why its laning strategies work. A rule is added only to create the incentive behind a real strategy, never to make an
agent behave in a recognizable way, and a fixed constant may not stand in for a consequence the game can produce.

The reward is fixed across the champion game. It is the zero-sum difference of earned gold (farm, tower plates and
kills), discounted at γ 0.99 in continuing time. There is no terminal state. Games end by truncation, and training
bootstraps the truncation. XP, HP, items and position are state, never reward. No strategy is scripted for the
learning agent; scripted strategies appear only as opponents and as alternative plans in evaluations.

## 2. Rulesets

One lane engine (`lanegame/engine/`) runs every ruleset, selected by a frozen `GameConfig`. A new rule enters as a
new field whose default reproduces the old behaviour exactly, is switched on in a new preset or a keyword override,
and is pinned by digest tests on every earlier preset (`tests/test_legacy_pins.py`,
`tests/test_m45a_mechanics.py::test_historical_presets_are_pinned`).

| preset | what it added | why (source) |
|---|---|---|
| `v1()` | the original exact wave game: three wave actions, expected gank cost | a solvable model of wave management (Phase 1) |
| `v2()` | farm in the payoff, Contest and Safe postures | strategy should be economic, not all gank exposure |
| `GameConfig()` (siege) | tower pressure and vision inside the action | a crash should be worth something to the crasher |
| `champions()` | M2 champions: HP, trades, Recall, ganks, death; 46 codes | the first learning game |
| `m4()` | the integrated economy, levels, items, plates, continuing time | the M3 game limited what could be learned |
| `m4_continuous()` | all unspent gold invested on Recall | items in two tiers were inert |
| `m4_emergent()` | a bounded investment curve, stand-off positioning, the pre-crash farm spot | unbounded power; trades against a retreat never landed |
| `m4_natural_prototype()`, `m4_natural()` | natural minion waves on 8-HP fronts, the entry crash, tower fire 3 | one set of minion rules whatever the actions; 4-HP fronts never stacked |
| `m4_lane()` | tower collection rates 0.9 and 0.6, durability | fixed collection rates did zoning's work with the wrong sign |
| `m4_lasthit()` | wave actions as shares of one capacity | Hold dealt no wave damage and Thin did not scale |
| `m4_four_actions()` | Last Hit, Thin, Slow Push, Shove; 61 codes | power should set how large a correction is, not how much damage is forced |
| `m4_four_actions_econ()` | a smaller crash tax, trade factor 0.8, kill gold 2, death timers 3/4/5 | Shove dominated because of the crash tax |
| `m4_four_actions_econ_aggro()` | no tower busy discount for divers | dives were underpriced |
| `m45_unified()` | the Vision node and Ward action, XP range 2, minion retaliation and per-tick aggro, the reach ball | one integrated revision of positional combat (M4.5a) |
| `m45_calibrated()` | the `last_front` farm-spot fix, retaliation on the exchange tick | M4.5b calibration; **the research preset** |

Candidate variants are keyword overrides, never presets; the C1 investment curve of M4.6c is
`m45_calibrated(invest_gain=3.24, invest_scale=64.0)`. The archived reports give the measurements behind each step,
and [the project history](archive/history.md) lists the rules that were tried and rejected.

## 3. Exact game theory (Phase 1)

`lanegame/exact/` solves the wave-only game exactly. The reachable state space is enumerated layer by layer
(`engine/game.py`), and backward induction (`exact/solver.py`) solves each stage's zero-sum matrix game
(`exact/matrix_game.py`): saddle points are detected exactly, otherwise a HiGHS linear program is solved and its
duality gap certified. `exact/best_response.py` evaluates any policy pair exactly and computes best responses and
exploitability. The finite horizon K is 32 in the published results (172,907 decision nodes in the siege model).

The exact game remained the project's oracle: M3 calibrated PPO against exact best responses on the K = 16 arena,
which is how frozen-snapshot self-play was shown to raise exploitability (findings §2).

## 4. The environment contract

`lanegame/env/` (`LaneEnv`) plays every ruleset as a simultaneous-move game.

- Each seat sees an own-perspective, Markov observation (88 entries in the four-action presets, 90 with the Vision
  node, 103 with the opt-in previous action, 116 with the opponent-modelling belief) and a legal-action mask.
- Chance enters only through ganks and trade or tower rolls. The environment draws a fixed number of uniforms per
  turn whatever the actions, so two policies can be compared on common random numbers. The same seed and the same
  actions give a bit-identical game. Tests pin this, and the lane-step cache is an exact memo.
- Champion rulesets run in continuing time. The end of the lane phase (about turn 94, level 14) is an evaluation
  checkpoint, never a terminal state.
- `env/pettingzoo_adapter.py` exposes the PettingZoo Parallel API for outside use; the trainer steps `LaneEnv`
  directly.

## 5. Training

Every M4 and M5 campaign uses shared-policy self-play (`lanegame/rl/shared.py`): one network plays both seats and
learns from both seats' transitions. The opponent is the current policy (OpenAI Five's 80/20 mixture with historical
snapshots bought nothing, findings §2); snapshots are kept for evaluation. The adopted settings (M4.4b onwards):

| setting | value |
|---|---|
| network | masked policy and value MLP, 2 × 128 tanh, shared trunk |
| PPO | clip 0.2, value clip 0.5, 4 epochs, minibatch 256, gradient norm 0.5 |
| optimizer | Adam, learning rate 3 · 10⁻⁴ annealed to 0 |
| returns | γ 0.99, GAE λ 0.95, reward scale 0.1, truncation bootstrapped |
| exploration | entropy coefficient 0.03 |
| collection | 16 environments × 128-turn segments × 2 seats; games of 512 turns |

Budgets count agent transitions (two per joint turn). Runs are seeded from disjoint seed blocks per study and
reproduce bit for bit; checkpoints carry a SHA-256 digest of their weights, verified on load. Opt-in hooks
(`SharedCollector.extra_obs`, `observe_step`, `train_shared(on_update=...)`, `SeparateCriticNet`, network widening)
let a study add inputs, monitoring or a separate critic while default training stays bit-identical, which the tests
pin. Stages that cost real compute refuse to start without a confirm flag that records the user's approval.

## 6. Evaluation instruments

Self-play return is zero by construction, so progress and strategy are measured against other policies and by
controlled comparisons.

- **Head to head and fixed suites.** Paired games with swapped seats against scripted references, earlier policies
  and other seeds' finals. Fixed suites saturate, so strength claims rest on head to head and cross-lineage games.
- **Paired counterfactuals** (`lanegame/diagnostics/branch.py`). A game state is restored and the agent's own play
  is compared with a fixed alternative plan, both continued by the same policies on the same random numbers. M4.4c
  introduced multi-turn plans with stop conditions and fallbacks.
- **The strategic battery** (`experiments/m45c/battery.py`, M4.5c). Families of natural and constructed states, each
  alternative scored as ΔQ₅₀, the 50-turn gold difference plus 0.99⁵⁰ times the critic's value difference. A 300-turn
  validation subset checks the short horizon; a family that fails validation is re-run at 150 turns (exception 3). A
  behaviour holds when no alternative beats the policy by more than 0.2.
- **Best-response attackers** (M4.5c, M5d). An attacker starts as a copy of its frozen target and trains with the plain
  zero-sum reward; its lead is a lower bound on the target's exploitability, never a certificate.
- **The lane-exit readout** (`lanegame/diagnostics/lane_exit.py`, M4.8). E_h is the earned-gold lead now (invested
  plus unspent) plus its expected change over h continuation turns by the policies that produced the state,
  undiscounted and without a critic tail. E_20 is the standard lane-phase readout; XP, HP, absence and the wave are
  reported beside it and never folded in.
- **A rule on the critic.** Critic values and bootstrapped tails are not evidence in strength-gap states, where the
  shared critic is known to be blind (M4.6d to M4.6f); rollouts and unbootstrapped G_h are used instead.

## 7. Statistics and pre-registration

From M4.6a on, each study froze a pre-registration document before any training or evaluation: arms, seeds,
budgets, endpoints, decision thresholds and an outcome table that maps results to a classification. Earlier studies
(M4.4b, M4.5b, M4.5c) fixed their criteria inside their plans before running.

- Seeds are the units. Results are means over seeds with t intervals (M5 used six pairs, t = 2.571).
- A gain counts only when its mean reaches the registered threshold and its interval excludes zero; an inconclusive
  result stays inconclusive (M5's outcome 6 and TOST equivalence tests were added for this).
- Secondary hypotheses form one fixed Holm family (M5: 15 hypotheses).
- A changed protocol is a dated amendment. M5's chain (`artifacts/m5/preregistration.json`, every version under
  `artifacts/m5/prereg_versions/`) is verified before every campaign stage.
- Exploratory analyses are labelled as such and never change a registered verdict.

## 8. Reproducibility and data

- `artifacts/` holds the curated records, tracked: every published study's summaries, the Phase 1 records, M5's
  complete records with its pre-registration chain, reproducibility manifest and SHA-256 inventory, and the 7
  networks the demo ships with ([artifacts/README.md](../artifacts/README.md)).
- `runs/` is git-ignored. It holds what runs write that is not curated, above all every checkpoint
  (`runs/checkpoints/<study>/`). A fresh clone has only the 7 shipped networks; `lanegame.paths.checkpoint` falls back
  to them.
- Records are never rewritten. Records written before the reorganization of 2026-11-08 name files by the old layout,
  and `lanegame.paths.relocated` maps those paths ([MOVES.md](archive/MOVES.md)).
- The code and the raw outputs of studies whose packages were removed are at commit
  [`f613f0a0`](https://github.com/jcj1337/lol_game_theory_laning/tree/f613f0a00a2d46fe717de7e8c79077fa9e0310c3),
  where each archived report's commands run as written.
