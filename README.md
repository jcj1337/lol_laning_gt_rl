# Laning as a Two-Player Game: Exact Equilibria and Self-Play Reinforcement Learning

A quantitative research project that models League of Legends laning as a simplified, sequential, two-player
zero-sum game and studies the strategies that arise in it, first with exact game theory and now with multi-agent
reinforcement learning.

## The research question

Two champions share one lane. Each turn both choose, simultaneously, how to handle the minion wave (last-hit, thin
it, clear it), where to stand (hold, yield, trade, play safe), and whether to go home. The simulator resolves
minion fights, tower fire, crashes and bounces, trades whose damage depends on level and items, ganks, recalls, XP,
gold, item purchases and tower plates. None of these is strategically interesting by itself. What makes laning a
game is how they interact: clearing a wave early makes it crash under the enemy tower, which opens a recall window
and denies the opponent farm, but exposes the pusher to a gank; a level lead turns a trade from a bluff into a
threat; a frozen wave starves the opponent only while it is held; plates reward sustained pressure. Those
interactions are what good League players exploit, and every one of them emerges from the rules rather than being
scripted.

**Can an RL agent learn to take advantage of those interactions without us explicitly teaching it League of Legends strategies?**

The agents see only the game's observations and legal actions, and their reward is the economic outcome: the
difference in gold earned, nothing else. No reward is given for freezing, slow pushing, recalling at the right time
or any other named strategy. Whether such behaviour appears is the thing being measured, through
quantitative evaluation of the learned policies: their discounted value against fixed references, their
exploitability (approximate best responses), and audits of what they actually do, compared with the sampled
equilibria of a scripted strategy space.

**Status, stated plainly.** The environment has been built, calibrated and validated (M1–M4.2), the RL
infrastructure audited (M4.3a) and a first self-play pilot run and analysed (M4.3b). That pilot learned strong,
state-dependent policies whose dominant wave action, clearing, counterfactual rollouts showed to be a best reply in
the calibrated game, so the game rather than the learner limited what could emerge. M4.3c–M4.3d therefore redesigned
wave management into four actions (Last Hit, Thin, Slow Push, Shove) and calibrated the economy (tower farming,
trades, death, tower retaliation on dives), producing two experimental candidate economies. **M4.4a** has made the
PPO pipeline and its strategic diagnostics ready for the four-action game (state-based freeze, zoning, vision and
plate-phase metrics; paired counterfactuals; smoke-tested on all three economies). The controlled comparison
(**M4.4b**: three economies × three seeds, plus a matched 2× extension) found that the calibrated economies
produce state-dependent wave management, short freezes that are preserved and broken but deny little, a targeted
response to tower retaliation and conditional defence, while the original economy stays Shove-dominated. **M4.4c** (in
review) tested every apparent gap with multi-turn paired counterfactuals: the learned policies match or beat every
hand-designed alternative against their own opponent, and the remaining imperfections are the simulator's incentives
(cheap-to-break freezes, cheap-to-answer threats, short sieges, three wave actions that coincide in most states) and
self-play's opponent distribution, not PPO failures. **M4.5a** (in review) answers that review with one integrated
mechanics revision, the EXPERIMENTAL candidate `m45_unified()`:

- a shared off-lane Vision node with a contestable Ward action;
- minion retaliation and positional aggro with per-tick exclusive targeting;
- level-scaled minion damage on champions;
- XP range 2;
- wave maintenance during trades;
- distance-limited wave damage.

Every earlier preset is unchanged. Integrated calibration (M4.5b) produced `m45_calibrated()`. Fresh PPO training on
it (M4.5c) is complete: six seeds × 8,192,000 and a matched 2× extension × 16,384,000 agent transitions, a
comprehensive evaluation, a 50-turn strategic counterfactual battery and learned best-response attackers
([research report](docs/m4/m4_5/m4_5c_research_report.md)). M4.4d's robustness work was done within M4.5c.

M4.6 then diagnosed the remaining gaps without changing the game. The investment-curve candidate C1 stays
provisional, and no freeze or concession mechanics change is justified. The critic is calibrated in natural states,
but the trained shared trunk discards strength, so it undervalues rare large power gaps
([M4.6e](docs/m4/m4_6/m4_6e_report.md)). A targeted pilot gave the critic its own trunk. It recovered about a third of
the strength value the shared critic misses, but within the tested budget it produced no material change in the
agents' strategic behaviour ([M4.6f](docs/m4/m4_6/m4_6f_report.md)), so the limitation is accepted and documented.

The [M4 closeout review](docs/m4/m4_closeout_report.md) assessed every laning goal against this evidence and
recommends **starting M5 with documented limitations**:
- Freezing pays only in a narrow early window.
- Leads fade late.
- Large strength gaps and unfamiliar opponent styles are not fully exploited.

It recommends the baseline `m45_calibrated()` over C1 and the shared network, a decision that awaits review.

Before the final campaign, the [M4.7 horizon study](docs/m4/m4_7/m4_7_horizon_report.md) asked whether the long
continuing objective puts too much weight beyond the laning phase (before about level 14, which arrives at turn 94).
- The early-lane conclusions do not depend on rewards after that phase.
- A matched pilot with a shorter discount (H 60) gave no reliable lane-phase gain and cost play later in the game and
  against fixed opponents.
- So γ 0.99 is kept.

The continuing objective stays as it is, and the game gains no ending at level 14 (decision of 2026-10-17). Who
leaves the lane better placed is measured instead at an evaluation checkpoint. The
[M4.8 lane-exit evaluation](docs/m4/m4_8/m4_8_lane_exit_report.md) validated such a readout and applied it to the
matched baseline and C1 policies. The readout is the earned-gold lead plus its expected change over a few repeated
continuation turns, with XP, HP and the wave reported beside it.
- The readout proved useful: exact, and calibrated against the games' own continuations.
- Raw gold at the lane exit misstates the economic position by at least half a wave in about two thirds of natural
  states, because of pending resets, HP and waves.
- The recommended readout is E_20, the expected lead 20 turns after the checkpoint.
- The baseline and C1 curves are indistinguishable at the lane exit.

After review (2026-10-17) the baseline `m45_calibrated()` was chosen for M5, and E_20 was adopted as the standard
lane-phase readout.

Before designing Bayesian opponent modelling, the
[OM1 feasibility study](docs/opponent_modelling/om1_feasibility_report.md) tested with existing policies whether it
could pay:
- The lane phase reveals the opponent's style within 10–20 observed turns.
- A belief-driven switcher between existing policies beats the baseline policy by about a wave at the lane exit
  (+0.90 ± 0.27), mostly against exploitable scripted styles.
- It loses against opponents outside its model (stochastic style mixtures).

So the design went ahead. The user chose its scope, robustness first: model both scripted styles and competent
learned opponents, and count a gain only if nothing is lost against competent or unfamiliar ones. The opponent's
previous action becomes an opt-in observation field. The [OM2 design](docs/opponent_modelling/om2_design.md) is
written and implemented behind opt-in switches. Its [matched pilot](docs/opponent_modelling/om2_pilot_report.md)
trained the same network with and without the belief:
- With the belief, the agent beats the equally trained agent without it by +2.03 ± 1.72 at the lane exit against
  recognizable styles, in every seed.
- It loses nothing against competent, mixed or style-switching opponents, or head to head (pre-registered outcome 1:
  modelling pays and is safe).
- Most exploitation comes from training against the styles at all. The belief adds about 40% more and protects
  competence, which the agent without it loses slightly.

Next is the user's decision on opponent modelling in M5, the final training and evaluation campaign, then M6, the
research presentation.

## How we study it

1. **An exact game as the foundation (Phase 1).** Wave management alone is a finite-horizon zero-sum stochastic
   game that we solve exactly: equilibrium values and mixed strategies, exact best responses and exploitability.
   This gives an unexploitable baseline and calibrates the RL environment, since a network policy on that game has
   an exactly computable value.
2. **A richer game that is not exactly solvable (M2 and M4).** Champions, HP, trading, recalls, then the integrated
   M4 economy: continuing time with a discounted objective (gamma = 0.99, no last turn to exploit), XP and levels,
   gold and items, durability plates, natural wave dynamics in which pushes, freezes, crashes and bounces follow
   from one damage law.
3. **Multi-agent reinforcement learning.** A simultaneous-move environment with own-perspective observations and
   legal-action masks, exposed through the PettingZoo Parallel API; a PyTorch policy/value network; a masked PPO
   implementation (CleanRL-style, audited against independent references); shared-policy self-play in which one
   network plays both seats and trains on both seats' experience, against the current policy or a mixture with
   frozen past versions; continuing-time rollouts with bootstrapped truncations; discounted evaluation with a
   guaranteed truncation bound.
4. **Quantitative evaluation.** Head-to-head and discounted values with confidence intervals and paired seeds,
   approximate best responses as exploitability lower bounds, action audits and counterfactual rollouts, and the
   exact solver where it applies.

Every mechanic is judged against one standard, [docs/design_philosophy.md](docs/design_philosophy.md): simplify how
League of Legends works without changing why its laning strategies work. The mechanics are deliberately stylized; no
parameter is claimed to be an empirical League statistic.

## Roadmap

The canonical roadmap is [docs/rl_research_roadmap.md](docs/rl_research_roadmap.md) (current-position summary at the
top). In short:

| Milestone | Content | Status |
|---|---|---|
| **Phase 1. Mathematical foundation** | Stochastic game, mechanics, exact equilibrium solver, best responses, exploitability | **complete** ([siege & vision study](docs/phase1/phase1_siege_vision_study.md); earlier: [revision](docs/phase1/phase1_revision_report.md), [original](docs/phase1/phase1_implementation_report.md)) |
| **M1. RL-ready game environment** | `lanegame/env/`: simultaneous moves, own-perspective observations, legal masks, zero-sum rewards, PettingZoo adapter | **complete** |
| **M2. Champion mechanics** | positions, Hold / Yield / Trade / Safe / Recall, trades with skill, ganks, death | **complete** ([M2/M3 report](docs/rl/m2_m3_implementation_report.md)) |
| **M3. Initial PPO / self-play research** | masked PPO, snapshot and shared-policy self-play, approximate best responses, diagnostics; revealed game-design limits that motivated M4 | **complete** ([self-play](docs/rl/openai_five_selfplay_report.md), [diagnostics](docs/rl/openai_five_diagnostics_report.md)) |
| **M4.1–M4.2. Integrated laning model** | continuing time, XP, levels, gold, items, plates; natural waves; zoning; wave actions (`m4_lasthit()`) | **complete** ([design](docs/m4/m4_1/m4_integrated_design.md), [implementation](docs/m4/m4_2/m4_implementation_report.md), [natural waves](docs/natural_waves/m4_2e_natural_waves_final_review.md), [wave actions](docs/natural_waves/m4_2e_wave_actions_report.md)) |
| **M4.3. PPO pilot, wave management and economic calibration** | RL audit, PPO pilot and validation, the [mathematical reference](docs/m4/laning_game_mathematical_reference.md), four-action redesign, economic calibration, tower pressure | **complete** ([pilot](docs/rl/m4_3b_ppo_pilot_report.md), [four actions](docs/m4/m4_3/m4_3c_four_action_wave_redesign.md), [economics](docs/m4/m4_3/m4_3c_strategic_economics_calibration.md), [tower pressure](docs/m4/m4_3/m4_3d_tower_pressure_calibration.md)) |
| **M4.4a. PPO readiness and strategic diagnostics** | three-environment training specification, checkpoint guard, state-based metrics, paired counterfactuals, smoke tests | **complete** ([report](docs/m4/m4_4/m4_4a_ppo_readiness.md), [the PPO pipeline explained](docs/rl/ppo_training_explained.md)) |
| **M4.4b. Controlled PPO self-play training** | 3 economies × 3 seeds × 2M transitions, a matched 2× extension; strength, convergence, strategic behaviour, counterfactual scorecard, representative games | **complete, in review** ([report](docs/m4/m4_4/m4_4b_ppo_training_report.md), [tables](results/m44b/report.md)) |
| **M4.4c. Strategic behaviour and economic incentives** | multi-turn paired counterfactuals on the M4.4b policies: interrupted Recall, freezes, zoning, Vision before Pressure, the four wave actions | **complete** ([report](docs/m4/m4_4/m4_4c_strategic_investigation.md), [tables](results/m44c/report.md)) |
| **M4.5a. Unified positional mechanics** | shared Vision node W and contestable Ward, per-tick minion retaliation and positional aggro, champions walking tick by tick with their wave (revision 2), minion damage on champions, XP range 2, capped wave budget while trading, reach ball (`m45_unified()`) | **complete** ([spec](docs/m4/m4_5/m4_5a_unified_mechanics_spec.md), [report](docs/m4/m4_5/m4_5a_unified_mechanics_report.md), [smoke tables](results/m45a/report.md)) |
| **M4.5b. Integrated calibration and strategic evaluation** | paired scripted simulations, restricted games, a joint factorial and a holdout; the farm-spot correction for mutual Shoves and retaliation from the exchange tick (`m45_calibrated()`); GO WITH CAVEATS for M4.5c | **complete, in review** ([report](docs/m4/m4_5/m4_5b_calibration_report.md), [tables](results/m45b/report.md)) |
| **M4.5c. PPO self-play on the calibrated model** | pilot: one seed × 1,024,000 agent transitions on `m45_calibrated()`, correctness verified, beats every fixed M4.5b strategy and still improving; recommended campaign 6 seeds × 8,192,000 | **pilot complete** ([pilot report](docs/m4/m4_5/m4_5c_pilot_report.md), [tables](results/m45c/report.md)) |
| **M4.5c. PPO training and evaluation on the revised model** | six seeds × 8,192,000 agent transitions (healthy, consistent, still improving head to head; extension criterion met); 50-turn strategic battery, attackers and the rest of the evaluation prepared and gated | **complete** ([research report](docs/m4/m4_5/m4_5c_research_report.md), [tables](results/m45c/final/report.md); [8.192M training report](docs/m4/m4_5/m4_5c_training_report.md)) |
| **M4.6. Design review and diagnosis** | zoning and freeze diagnosis, investment audit, C1-versus-baseline PPO pilot (C1 provisional, not adopted), favourable freezes and concession, the value diagnostic (observation, training exposure, value learnability), the separate-critic pilot | **M4.6a–d complete; M4.6e and M4.6f complete, in review** ([M4.6d](docs/m4/m4_6/m4_6d_report.md), [M4.6e](docs/m4/m4_6/m4_6e_report.md), [M4.6f](docs/m4/m4_6/m4_6f_report.md); index [docs/m4/README.md](docs/m4/README.md)) |
| One targeted learning pilot (M4.6f) | a separate critic trunk against an unchanged matched control, six seed pairs, separate 4.096M and 8.192M continuations from the same start | **complete, in review**: the critic learns about a third of the strength value; no material behavioural improvement demonstrated within the tested budget; investigation closed ([report](docs/m4/m4_6/m4_6f_report.md)) |
| M4 closeout | explicit C1-versus-baseline decision; final mechanics, objective and training setup; documented freeze conditions and limitations | **complete, reviewed**: ready for M5 with documented limitations; the baseline chosen for M5 on 2026-10-17, shared critic ([report](docs/m4/m4_closeout_report.md)) |
| M4.7 early-lane horizon study | what part of play occurs before about level 14; baseline and C1 there; whether delayed rewards drive the early conclusions; a matched two-seed pilot of a shorter discount | **complete, reviewed**: keep γ 0.99; the shorter objective (H 60) gave no reliable lane-phase gain and cost later play ([report](docs/m4/m4_7/m4_7_horizon_report.md)) |
| M4.8 lane-exit evaluation | an evaluation-only readout of who leaves the lane better placed (the earned-gold lead plus its expected change over h repeated continuation turns; XP, HP and the wave beside it), validated and applied to the matched baseline/C1 policies | **complete, reviewed**: useful (exact, calibrated against recorded continuations); E_20 adopted 2026-10-17 beside raw gold and the non-gold dimensions; raw gold misleads materially in about two thirds of natural lane-exit states; baseline and C1 indistinguishable there ([report](docs/m4/m4_8/m4_8_lane_exit_report.md)) |
| OM1 opponent-modelling feasibility | identifiability of opponent styles in the lane phase and the value of knowing them, with existing policies; evaluation only | **complete, in review**: feasible and worthwhile; identified within 10–20 turns; switcher +0.90 ± 0.27 over the baseline policy, mostly against scripted styles; misspecified opponents are the risk ([report](docs/opponent_modelling/om1_feasibility_report.md)) |
| OM2 Bayesian opponent modelling: design | beliefs about the opponent's style, updated during a game; scope C (robustness first); the opponent's previous action observed | **complete, in review** (opt-in; [design](docs/opponent_modelling/om2_design.md)) |
| OM2 matched opponent-modelling pilot | the same opponent distribution in both arms, with and without the belief; 3 seed pairs × 4.096M | **complete, in review**: outcome 1, modelling pays and is safe; +2.03 ± 1.72 against recognizable styles, no loss against competent, mixed or head-to-head opponents ([report](docs/opponent_modelling/om2_pilot_report.md)) |
| M5. Final training and evaluation campaign | multi-seed shared-policy PPO on the selected setup; learning progress, strategic counterfactuals, diverse opponents, cross-seed results, empirical exploitability; lane-phase outcomes read with the lane-exit readout | proposed, after the opponent-modelling design and pilot (scope proposal and first task M5a in the closeout report §8) |
| M6. Research presentation and reproducibility | consolidated report, reproducible commands, figures, representative trajectories, visual demo, README | proposed |

The documentation index is [docs/README.md](docs/README.md). **Every rule, equation and parameter of the current
research game is in one place: [docs/m4/laning_game_mathematical_reference.md](docs/m4/laning_game_mathematical_reference.md)**
(kept up to date with every rule change; the summaries below are older and coarser).

## The games

**The exact game (Phase 1, `GameConfig()`).** Each turn both players simultaneously choose a wave-management action
(Slow Push, Shove, Freeze) and a positioning action (Contest, Safe); a player whose crashed minions are under the
enemy tower also chooses a siege objective (Apply Pressure or Establish Vision). Minion waves are explicit packets
that spawn, walk, merge and fight with discrete Lanchester attrition; a crashed wave sits under the tower as HP mass
that tower fire removes. The economy is event-based: every HP of minions that dies is a farm opportunity for the
side that killed it, plates are claimed one per pressure turn, ganks are a per-turn risk reduced by vision, and the
payoff is relative resources, zero-sum. The solver is backward induction with exact stage games (saddle detection,
otherwise a HiGHS LP), certified duality gaps, exact best responses and exploitability. Main findings of the siege
study (K = 32): a big crash becomes an asset once it can be sieged (+6.1 instead of −5.8 to the crasher); the
equilibrium establishes vision first and applies pressure second; the symmetric equilibrium is an even, contested,
crash-free lane fight; predictable play is much more exploitable (passive_safe 21.5 instead of 9.4) through the
siege channel. Details and caveats in the [study](docs/phase1/phase1_siege_vision_study.md). The two earlier models
are available bit-for-bit as `GameConfig.v2()` (farming revision) and `GameConfig.v1()` (original).

**The M4 game (`GameConfig.m4()` and its presets).** A continuing game with no terminal state, discounted with
gamma = 0.99 and trained in bootstrapped segments; champions with persistent positions, centi-HP and attack power
scaled by level and items; XP within 3 cells and levels 1–18; an exact micro-resource gold ledger; purchases on
return to base; persistent tower damage with three plates; ganks with HP damage, death and respawn; a lossless,
decoder-proven 88-feature observation; 46 actions. The M4.2 milestones added opt-in presets on top of `m4()`:
`m4_emergent()` (positioning, zoning and investment readiness), `m4_natural()` (natural wave mechanics: Hold, Thin
and Clear are champion damage on the wave, and pushes, freezes, crashes and bounces follow from the damage law),
`m4_lane()` (danger as consequences instead of fixed collection rates; investment buys durability) and
`m4_lasthit()`, the M4.3 target: Hold, Thin and Clear as shares (1/8, 1/2, 1) of one wave-clear capacity, so that
normal last-hitting pushes a little and thinning grows with power, on a half-wave Lanchester law. Every legacy
configuration is bit-identical, guarded by `tests/test_legacy_pins.py`.

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate      # macOS/Linux: source .venv/bin/activate
pip install -e ".[test,rl]"                          # numpy, scipy, matplotlib, pytest; torch, pettingzoo, gymnasium
python -m pytest                                      # full suite (exact game, champion game, M4, RL): ~1,080 tests, ~8 min
python -m lanegame solve --horizon 32                 # exact siege model -> results/siege/solution_K32.json (~4 min, 3 GB)
python -m lanegame scenarios                          # 23 traced mechanics scenarios with PASS/FAIL
python -m lanegame exploit --horizon 24               # exploitability of the siege baselines
python -m lanegame env-demo --a aggressive --b passive_safe --episodes 200 --seed 42   # scripted agents through the RL environment
python -m lanegame env-demo --model champions --a opportunistic_trader --b hold_farmer   # the champion game (M2)
python -m lanegame rl-shared --model m4_lasthit --current-fraction 1.0 --steps 409600 --out results/rl/checkpoints/demo_m4
                                                      # shared-policy self-play PPO on the M4.3 target (continuing time)
python -m lanegame experiments-rl m43a_smoke          # the M4.3a smoke test (results/rl/m43a_smoke.md, ~5 min)
python -m lanegame.experiments.rl.m43b train curves headline exploit behavior counterfactual plans recall trajectories figures report
                                                      # the M4.3b pilot stage by stage (results/rl/m43b/, ~2 h on 9 processes)
python -m lanegame.experiments.rl.m43b validation     # the follow-up: val_wave_cf val_horizon val_games val_freeze val_report (~1.5 h)
python -m lanegame.experiments.rl.m43b manifest       # results/rl/m43b/MANIFEST.json and the checkpoint bundle (results/rl/m43b_artifacts/)
python -m lanegame.experiments.m43c all --workers 9   # M4.3c design diagnostics: examples, progression audit, gold + XP
                                                      # re-pricing, freeze and death harnesses (results/m43c/, ~35 min; no training)
python -m lanegame.experiments.rl.m44 audit smoke_train smoke_repro smoke_eval demo plan report
                                                      # M4.4a: four-action PPO readiness (results/m44a/, ~5 min on 9 processes)
python -m lanegame.experiments.rl.m44 train_main --confirm-m44b && python -m lanegame.experiments.rl.m44 eval_main
                                                      # M4.4b: 3 economies x 3 seeds x 2M transitions (~13 + 50 min on 9 processes)
python -m lanegame.experiments.rl.m44 repro_main extra_main traces_main recall_main report_main
                                                      # M4.4b supplementary analyses and results/m44b/report.md
python -m lanegame.experiments.rl.m44 train_ext --confirm-m44b && python -m lanegame.experiments.rl.m44 eval_ext
                                                      # M4.4b matched 2x-budget extension (4,096,000 transitions per run)
sh results/m44c/queue.sh                              # M4.4c: every multi-turn counterfactual stage, resumable (~5.3 h on
python -m lanegame.experiments.rl.m44c traces report  # 11 processes), then viewer pairs and results/m44c/report.md
python -m lanegame rl-shared --current-fraction 0.8 --skill-levels 0.2,0.5,0.8 --out results/rl/checkpoints/demo_shared
                                                      # the champion game, 80% current / 20% past versions
python -m lanegame rl-eval results/rl/checkpoints/demo_shared/final.pt   # champion game: both seats, held-out seeds, 95% CIs
python -m lanegame experiments-rl openai_five         # the OpenAI Five-inspired study (results/rl/openai_five/)
python -m lanegame experiments-siege all              # regenerate every siege-study result (results/siege/)
python -m lanegame.experiments.m45a all              # M4.5a: demonstration recordings + smoke study (results/m45a/, ~30 s)
python -m lanegame.experiments.m45a movement         # M4.5a correction: scenarios, revision 1 vs 2 (~30 s)
python -m lanegame.experiments.m45b <stage> ...      # M4.5b: calibration stages (results/m45b/; ~2 h for all on 11
                                                      # processes; the stage list: docs/m4/m4_5/m4_5b_calibration_report.md §15)
python -m lanegame.experiments.rl.m45c <stage> ...  # M4.5c: pilot stages, the campaign (train_full, needs
                                                      # --confirm-m45c-full), health_eval / health_report
                                                      # (results/m45c/full/), battery / attackers (gated)
streamlit run lanegame/viewer/app.py                  # the visual lane debugger (pip install -e ".[viewer]"); mode
                                                      # "Recorded trajectory" replays results/rl/m43b/trajectories/*.json
```

The exact-game commands accept `--model siege|revised|v1` (default `siege`).

Using the environment from Python, natively or through PettingZoo:

```python
from lanegame import GameConfig
from lanegame.env import LaneEnv
from lanegame.env.pettingzoo_adapter import LaneParallelEnv

env = LaneEnv(GameConfig.m4_lasthit(), max_turns=512)
obs_a, obs_b = env.reset(seed=0)                 # own-perspective float32 vectors, legal masks, the shared state
result = env.step(obs_a.legal[0], obs_b.legal[0])   # both actions at once; rewards (r, -r); info with every event

pz = LaneParallelEnv(GameConfig.m4_lasthit(), max_turns=512)
observations, infos = pz.reset(seed=0)           # {"A": {"observation", "action_mask"}, "B": ...}
```

## Repository layout

```
lanegame/                 engine (pure Python + NumPy/SciPy, no presentation code)
  config.py               all model parameters (GameConfig and its presets: v1, v2, siege, champions, m4 ... m4_lasthit)
  state.py                Markov lane state and canonical encoding
  mechanics.py            packet movement, merging, fights, thinning, crashes, tower fire, spawns, plates, vision timers
  game.py                 legal actions incl. siege objectives, farming, plates, gank model, reachable states
  matrix_game.py          zero-sum matrix games: saddle detection, HiGHS LP, certificates, support enumeration
  solver.py               backward induction -> values, mixed strategies, Q matrices
  best_response.py        exact policy evaluation, best responses, exploitability
  policy.py, baselines.py policy interface; scripted baselines
  analysis.py             exact forward occupancy and path statistics
  reachability.py         reachable lane and siege events with witnesses
  simulate.py             seeded Monte Carlo with realized gank outcomes
  scenarios.py, render.py named behavioral scenarios (23) and text traces of the lane
  io.py                   JSON export format for solutions (version 3)
  champions.py            champion mode (M2): positions, Hold/Yield/Trade/Safe/Recall, absence, trades, skill, ganks, death
  champions_m4.py         the M4 ruleset's state and resolver (Freeze maintenance, XP/levels, items, ledger, plates)
  progression.py          M4 integer tables: max HP, HP unit, attack power, damage, XP thresholds, purchases
  lane_enum.py            exhaustive reachable-lane enumeration and the enumerated bounds M4 relies on
  scenarios_m4.py         M4 parked-front harness and replayable scenario witnesses (data/m4_scenarios.json)
  env/                    the simultaneous-move RL environment: core.py (LaneEnv), observation.py / observation_m4.py
                          (own-view layouts, the lossless M4 decoder), agents.py / agents_m4.py (scripted agents),
                          trace.py (episode logs), discounted.py (E1/E2 evaluation with the guaranteed tail bound),
                          starts.py (witnessed start archives), pettingzoo_adapter.py (the PettingZoo Parallel API)
  rl/                     PyTorch: network.py (policy/value MLP, masked categorical), ppo.py (GAE, clipped update),
                          train.py (collector, PPO loop, continuing-time game times), selfplay.py (snapshot pool),
                          shared.py (shared-policy self-play, current / historical mixture), agent.py (NetAgent,
                          checkpoints with digests), evaluate.py (head-to-head, round robin, approximate best responses)
  diagnostics/            action audits and paired counterfactual rollouts from restorable states
  viewer/                 the Streamlit lane debugger (M4.2e-D)
  experiments/rl/         RL studies, the M4.3a smoke test, the M4.3b pilot package m43b/ with its follow-up
                          (intervene.py, validation.py, manifest.py), the M4.4 package m44/ (four-action training,
                          state-based strategic metrics, paired counterfactuals); results/rl/ (large JSON gzipped; see
                          results/rl/m43b/ARTIFACTS.md); checkpoints in results/rl/checkpoints/, untracked, bundled on request
  experiments/m4/         M4 validation and calibration drivers (results/m4*/)
  experiments/champions/  champion-mode validation (results/champions/)
  experiments/siege/      the siege study (results/siege/); experiments/, experiments/v1/: the earlier exact models
tests/                    pytest suite: exact models, champion game, M4 mechanics, environment, PettingZoo, PPO references
results/                  outputs of every study, one folder per study, with the commands that produced them
docs/                     design_philosophy.md, rl_research_roadmap.md and, by phase, phase1/, rl/, m4/, natural_waves/
legacy/                   the original Q-learning prototype, kept unchanged for reference
```

## The earlier models

- **Farming revision** (`GameConfig.v2()`): event-based farming, Contest/Safe positioning, mass-based tower fire, a
  post-crash choice between crash resources and two turns of 50% vision. Documented in
  [docs/phase1/phase1_revision_report.md](docs/phase1/phase1_revision_report.md), results in `results/revision/`.
- **Original Phase 1** (`GameConfig.v1()`): wave actions only, no farming, one-packet-per-turn tower consumption.
  Documented in [docs/phase1/phase1_implementation_report.md](docs/phase1/phase1_implementation_report.md), results
  in `results/`.
- **The first Q-learning prototype** is in [legacy/](legacy/), audited in section B of the original report.
