# Project history

How the project got from a small solvable game to the final campaign, what was tried and abandoned, and what it
taught. It condenses the archived reports for a reader who wants the path, not only the destination; the measured
results are in [findings](../findings.md) and the documents themselves are listed in the [archive index](README.md).

Citations name the archived report and its section. "Roadmap" is [rl_research_roadmap.md](rl_research_roadmap.md),
whose summary and sections L to BG record every milestone's outcome and the user's decisions in order. Statements
marked *(synthesis)* are this review's reading across reports, not a claim any single report makes.

## 1. Timeline

| dates | era | studies | in one line |
|---|---|---|---|
| 2025-12 to 2026-01 | prototype | the Q-learning notebook (`legacy/`) | a tabular learner on a fixed payoff matrix; audited in Phase 1 and replaced |
| 2026-10-10 to 10-11 | Phase 1 | v1, revision (v2), siege study | the exact game: passive, crash-free equilibria; siege objectives make crashes worth buying |
| 2026-10-11 | M1 to M3 | environment, M2 champions, M3 PPO and self-play, OpenAI Five study, diagnostics | PPO learns; snapshot self-play drifts; the game, more than the learner, limits play |
| 2026-10-11 to 10-12 | M4.1 to M4.2 | integrated design, implementation, calibration, continuous investment, wave readiness, parallel workstreams, natural waves, zoning, wave actions, debugger | an integrated continuing-time economy built and calibrated rule by rule |
| 2026-10-12 to 10-13 | M4.3 | RL infrastructure audit, first PPO pilot and validation, objective design, screens, four-action redesign, economic calibration, tower pressure | Clear/Shove dominance traced to the economy and fixed there |
| 2026-10-13 to 10-15 | M4.4 to M4.5 | PPO readiness, controlled training, strategic investigation, unified mechanics, calibration (`m45_calibrated()`), the M4.5c campaign | PPO discovers conditional strategy; the research preset is fixed |
| 2026-10-15 to 10-17 | M4.6 to M4.8 | design review, diagnosis, investment audit, C1 pilot, freeze study, value diagnostic, separate critic, closeout, horizon study, lane exit | the limits are training distribution and representation, not the rules; E_20 adopted |
| 2026-10-17 | OM1, OM2 | feasibility, belief design and matched pilot | opponent modelling is feasible and pays in a pilot |
| 2026-10-17 to 10-19 | M5 | design and readiness, training, battery and cells, attackers, closeout | the modelling package pays against styles at a competence cost |
| 2026-10-19 to 10-22 | M6 | plan, session core, web application, Sequences, research overview; repository audit | the demo, verified against the research recordings |
| 2026-10-29 | NC1 | network capacity (experimental) | 2 × 128 suffices; widening lost at M5's learning rate |
| 2026-11-08 | reorganization | this archive | the repository restructured around the final research |

## 2. How the project evolved

**The exact game (Phase 1).** The project began by replacing a Q-learning notebook whose Q-table was keyed on
floats, had no time index and was not zero-sum on 53.8% of steps
([v1 report](phase1/phase1_implementation_report.md) §B) with an exactly solvable stochastic game of wave
management. All three versions of that game had passive, crash-free symmetric equilibria. The farming revision made
strategy economic; the siege study let a crash buy tower pressure and vision, which turned a crash liability into an
asset (−5.79 to +6.05) but left the symmetric path unchanged. The exact game was kept as an oracle rather than
enlarged ([roadmap](rl_research_roadmap.md) §A).

**The first learning game (M1 to M3).** Champions made the game unsolvable, so it moved into an environment with an
exact bridge to the solver. PPO learned, but frozen-snapshot self-play became more exploitable with training, and of
the OpenAI Five recipe only the shared policy helped. The diagnostics study showed what the learner could not fix: a
one-turn Freeze equal to Slow Push in every reachable state, a late Shove rewarded by the 32-turn end, and sieges no
self-play run visited ([diagnostics](rl/openai_five_diagnostics_report.md) §C, §D). The planned PSRO stage was
dropped when the scope was narrowed to OpenAI Five-style self-play (roadmap §D).

**An integrated economy (M4.1 to M4.2).** The M2 game was replaced by one integrated ruleset in continuing time: XP
and levels, gold invested on Recall, tower plates and persistent tower damage
([M4 design](m4/m4_1/m4_integrated_design.md)). It was then calibrated one mechanism at a time with scripted play
and solved restricted games, each accepted change becoming an opt-in preset (the chain is in
[methodology](../methodology.md) §2). Four workstreams ran in parallel worktrees; positioning, zoning and investment
merged into `m4_emergent()`, and natural waves on 4-HP fronts did not
([parallel report](m4/m4_2/m4_emergent_parallel_report.md) §2). Minion waves were rebuilt as natural waves that follow
one set of rules whatever the champions do, and the wave actions were redefined as shares of a champion's damage
capacity. The standard for every mechanic, [design_philosophy.md](../design_philosophy.md), dates from this era.

**The first PPO pilot and the economy (M4.3).** On `m4_lasthit()` PPO cleared the enemy wave in 63% to 75% of turns
under every training variant. Wave-only interventions showed this was the game's best reply, not a learning failure
([validation](rl/m4_3b_validation_report.md) §8). Neither a four-action redesign, XP in the reward nor a terminal
plate changed it; only the economy did, above all the crash tax under the defender's own tower
([economic calibration](m4/m4_3/m4_3c_strategic_economics_calibration.md) §0).

**Learned strategy (M4.4 to M4.5).** On the calibrated four-action economies PPO played state-dependent wave
management. Multi-turn paired counterfactuals (M4.4c) found no profitable strategy PPO consistently failed to exploit
against its own opponent and reversed an earlier "no zoning" verdict. M4.5 then integrated positional combat, an
off-lane Vision node with a contestable Ward and per-tick minion aggro, calibrated them (finding and narrowly fixing
an inherited farm-spot defect) and selected `m45_calibrated()`. Six seeds trained on it to 16.384M transitions
discovered conditional wave management, freezing, selective trading, concession at initialized gaps and Recall
timing, avoided Wards and bullying against competent play, and stayed exploitable by attackers
([M4.5c research report](m4/m4_5/m4_5c_research_report.md)).

**Diagnosis (M4.6 to M4.8).** A design review proposed one root cause for cheap freeze breaks and weak zoning;
M4.6a's diagnosis rejected it for zoning, which the rules reward but self-play never meets (gaps occur on under 0.4%
of turns). Freezes break cheaply for structural reasons. The critic proved nearly blind to strength because the
shared trunk drops it, and a separate critic recovered about a third without a demonstrated behavioural gain. A
steeper investment curve (C1) stayed inconclusive. The closeout judged M4 ready for M5 with documented limitations.
The horizon study fixed the lane phase at turns 0 to 93 and kept γ 0.99, and the lane-exit readout E_20 replaced raw
gold as the lane-phase measure ([closeout](m4/m4_closeout_report.md); [M4.8](m4/m4_8/m4_8_lane_exit_report.md)).

**Opponent modelling and the final campaign (OM1, OM2, M5).** The user chose Bayesian opponent modelling as M5's
question. OM1 showed the opponent's type is identifiable within 10 to 20 turns and worth knowing; OM2 built an
opt-in previous-action observation and a belief over 11 library types plus "mixed" and "unknown", and its matched
pilot paid. M5 compared that whole package with continued pure self-play under a frozen, once-amended
pre-registration. The package paid against scripted styles and cost competence against competent opponents; both
arms stayed exploitable ([M5 final report](m5/m5e_closeout_report.md)).

**Presentation (M6) and after.** M6 built a web demo in which a visitor plays or watches the pair-0 finals of both
arms, with every number taken from the engine and the research recordings re-simulated at startup. A read-only audit
(2026-10-22) found no defect that changes a stored result and several wording and provenance issues
([audit](audit/repository_audit_2026-10-22.md)). NC1 then asked whether the network was too small.

## 3. Reward and objective decisions

| decision | when | reason given | source |
|---|---|---|---|
| exact game zero-sum, risk-neutral, undiscounted, terminal value 0; ganks as expected loss | Phase 1 | a tractable stochastic game | v1 report §C.8, §C.9 |
| a crash pays nothing in itself ("no duplicate crash rewards") | siege study | value should come from what the crash buys | siege study §C.8 |
| RL reward = realized farm + plates + kills difference; HP, XP, items, tower damage are state | roadmap plan | reward only the economy | roadmap §C.2, §E.9 |
| gank base 0.15 dropped from the champion game | M2 | as an HP event it would give about 7 damaging ganks per player per game | M2–M3 report §B.6 |
| `kill_bounty` 0.5 (planned 0) | M2e | with 0 the scripted meta-game was 100% naive Hold | M2–M3 report §C.6 |
| continuing time, γ 0.99, no terminal state | M4.1 | the 32-turn end rewarded a late Shove; average reward and purchase actions rejected | M4 design §1.2, §10, §16 |
| XP channel in the reward not adopted | M4.3c | changed no equilibrium at any weight 0 to ½ | economic calibration §6 |
| termination at the last plate not recommended | M4.3c | a terminal event one player controls invites prolongation | economic calibration §7 |
| `kill_bounty` 2.0 | M4.3c econ | with the crash tax, makes play state-dependent | economic calibration §12 |
| γ 1 − 1/60 tested and rejected | M4.7 | no reliable lane gain; a loss against fixed opponents | M4.7 §5, §6 |
| no terminal state at level 14; lane exit as an evaluation checkpoint | user, 2026-10-17 | keep the continuing objective | roadmap summary |
| attackers use the plain zero-sum reward | M5 | measure exploitability, not deception | M5 pre-registration §8 |

Apart from parameter values, the champion game's reward never changed after M4.1. It stayed the zero-sum
earned-gold difference with γ 0.99 and bootstrapped truncation.

## 4. Simulation rules tried and rejected

Each line is a rule that was built or screened and not adopted, with the reason its report gives.

- Phase 1: ceiling rounding of firepower (a 1-HP surplus doubled it), linear attrition, creeping Slow Push in v1; a
  queue rule and a "hold" Slow Push that made the v2 state space unbounded (1.95 million nodes at K = 128); engaging
  defenders, which removed bounces; a per-tower plate budget (683,425 nodes at K = 32)
  ([v1](phase1/phase1_implementation_report.md) §D.1; [revision](phase1/phase1_revision_report.md) §E;
  [siege](phase1/phase1_siege_vision_study.md) §J).
- M4.1: Freeze as a maintained action, later replaced when natural waves made freezing a lane state; two item tiers,
  which were inert (M4.2 §6).
- M4.2b: a lighter gank (0.40) and stronger items (1.35), recommended for review only and never adopted; pushing lost
  because of lane economics, not ganks ([calibration](m4/m4_2/m4_calibration_report.md)).
- M4.2c: continuous investment at exponent 0.9, which grew without bound; replaced by a saturating curve.
- M4.2d-A: wave-bounce patches A and B, "no parking past the lane", `siege_engage` and stronger tower fire; natural
  waves produced a bounce instead ([wave readiness](m4/m4_2/m4_wave_readiness_report.md) §4).
- M4.2d: natural waves on 4-HP fronts (fronts never stacked; Shove dominated by −4.18 per 40 turns); `"contact"`
  zoning (forced trading).
- M4.2e: tower fire τ = 1 (a 42-turn, 196-HP self-feeding siege) and τ = 2; AP scaling of Thin; an idle siege wait
  "written to manufacture a bounce" ([natural waves final](natural_waves/m4_2e_natural_waves_final_review.md) §2);
  two exchanges per trade (a trade-fest); the loser farm rule; removing the Safe base
  ([zoning](natural_waves/m4_2e_zoning_report.md)); the whole-wave damage law and a 16-HP wave
  ([wave actions](natural_waves/m4_2e_wave_actions_report.md) §2).
- M4.3c: XP ranges 1 and 2 on the five-cell lane, adaptive Thin, `strength_nofloor` (left open)
  ([xp range](m4/m4_3/m4_3c_xp_range_screen.md); [wave control](m4/m4_3/m4_3c_wave_control_screen.md);
  [four actions](m4/m4_3/m4_3c_four_action_wave_redesign.md) §7.2).
- M4.5a: turn-locked targeting, a Hold exemption, moving end cells after the lane step, a Safe penalty; revision 1's
  positional rule, which treated a symptom (kept as `m45_unified_r1()`)
  ([unified mechanics](m4/m4_5/m4_5a_unified_mechanics_report.md) §3, §10).
- M4.5b: minion α 80 with retaliation delay 1 (α also scales aggro); growth 1.12, "a post-hoc projection"
  ([calibration](m4/m4_5/m4_5b_calibration_report.md) §9.2, §11.2).
- M4.6: two trade exchanges (L1, "not supported"), reset pricing (L2), an Engage behaviour (L3); a freeze-durability
  mechanic, named but not recommended; the C1, C2 and C3 investment curves (C1 stays a provisional opt-in)
  ([M4.6a](m4/m4_6/m4_6a_report.md) §10; [M4.6d](m4/m4_6/m4_6d_report.md) §8; [closeout](m4/m4_closeout_report.md)
  §5.1).

## 5. Approaches that failed or surprised

- Frozen-snapshot self-play raised exact exploitability; more entropy and a variance-reduced reward did not stop it
  ([M2–M3](rl/m2_m3_implementation_report.md) §D.3). Historical opponents never reduced exploitability at any tested
  budget ([OpenAI Five](rl/openai_five_selfplay_report.md) §G.2; [diagnostics](rl/openai_five_diagnostics_report.md)
  §F).
- Every unconditional rule that used the unused mechanics lost (shove plus pressure −2.31 to −2.58; diagnostics
  §D.1).
- Small attackers gave unreliable bounds. Two attackers on one target differed by up to 2.0, and 1M-step attackers
  raised the 102k bounds several-fold (diagnostics §E).
- M4.3a's "no Recall" was a learning-rate schedule artefact, not a strategy
  ([PPO pilot](rl/m4_3b_ppo_pilot_report.md)).
- No tested rule produced the League stand-back concession at the lane (M4.2d-C, M4.2e-B).
- One-turn counterfactuals resolved too little (median ±0.43); M4.4b's zoning metric missed denial by absence and its
  Recall statistic conditioned on a guessing game's outcome ([M4.4c](m4/m4_4/m4_4c_strategic_investigation.md) §5.5,
  §8).
- Learning-rate annealing makes every run's end look stable; a matched fresh extension and an annealing test were
  needed to see the policies still improving (M4.5c research report §2.1).
- Hand-written Shove and trade hybrids lost to plain Shove; scripted equilibria describe the strategy set, not the
  game (M4.5b §7.2).
- The M4.6 design review's single root cause was wrong for zoning, and its predictions were wrong on three of four
  counts (M4.6a §8.6). Discovery-selected winners in M4.6d did not confirm on independent seeds (§3.2).
- Better value estimates did not change decision rankings or behaviour (M4.6e §5.3; M4.6f §5.2).
- The short horizon cost late-game play and play against scripted opponents ([M4.7](m4/m4_7/m4_7_horizon_report.md)
  §5.3).
- OM1's switcher lost most of the oracle's value against look-alike styles and up to −1.45 against mixtures
  ([OM1](opponent_modelling/om1_feasibility_report.md) §4, §5). In OM2 most exploitation came from training against
  the styles, not from the belief ([OM2](opponent_modelling/om2_pilot_report.md) §5).
- M5's package lost competence in every seed and did not improve over its start while pure self-play did
  ([M5 initial report](m5/m5b_initial_report.md) §4, §5).
- NC1's wider network lost to its control, and its protocol had no outcome row for an arm that loses
  ([NC1](capacity/nc1_report.md) §6.4).

## 6. Corrections made after the fact

Historical reports are never edited; later documents corrected them. The main corrections:

| corrected claim | correction | where |
|---|---|---|
| M3 "no skill-dependent aggression" | specific to frozen-snapshot training | OpenAI Five report §G.7 |
| M4.4b "zoning not learned" | withdrawn; zoning converts through kills and forced Recalls | M4.4c §5.5 |
| M4.4b "no simulator defect" | the farm-spot defect was present on `aggro` too | M4.5b §8.2 |
| M4.5c stronger side "+2 cells forward", walk-up −0.22, trading −0.34 | just behind the lane cell, −0.02 to −0.13, about −0.08 | M4.6b §1.3; roadmap §AN |
| M4.5c "bullying avoided" | holds only for self-play's gap-free distribution (a gap specialist +2.11) | M4.6a §6.1 |
| M4.6a and M4.6c critic bias readings | understated three- to fourfold by bootstrapped tails | M4.6d §6.1 |
| M4.6e cancellation explanation | a plausible partial explanation, not a demonstrated one | closeout §2 |
| M4.6f "12M transitions", "same data", "sharing is enough", "did not produce better behaviour" | two separate continuations; arms diverged after one update; sharing, capacity and optimizer scaling changed together; not demonstrated within the tested budget | closeout §2 |
| M4.7 "baseline leads shrink by a fifth", pilot lane gain +0.26 | −0.10 ± 0.10 and +0.17 ± 0.18 at the lane-exit readout | M4.8 §4.5 |
| M5b "om about +0.25, an interval including 0" | +0.24 with C_cross [+0.004, +0.55] excluding 0 | audit W6; research overview §9 |
| M5c G9 "missed"; "same strategic choices"; freezes "within 0.03" | open (suggestive); the same four registered findings; 0.014, 0.015, 0.034 | M5e §4; audit W6 |
| M5c concession failure in both arms | the arms fail on different evidence (om at 50 turns; sp only at 150 turns in two seeds) | audit R2 |
| M5e "+6.4 against held-out scripted variants" | averages four scripted variants (+3.7 to +14.4) with three conditional styles (−0.2 to −0.7) | audit W1; roadmap §BF |
| statements in M4.2e to M4.5b reports that differ from the code | DEPRECATED, OUTDATED, SUPERSEDED or DISCREPANCY rows | [model](../model.md) §12; [archived reference](m4/laning_game_mathematical_reference_m4_lasthit.md) §12 |

## 7. Contradictions between sources

Found while preparing this archive; none changes a registered verdict.

- The roadmap calls unvisited sieges a limit of the game; the diagnostics report classed them as a training issue
  (§D.5), because 1M-step attackers found a profitable siege plan.
- Skill-conditioned Trade appears in 5 of 6 runs in the OpenAI Five report and 4 of 6 in diagnostics §G, under a
  different measurement that §G acknowledges.
- The roadmap §W gives one-item leads of 2 to 3 per 32 turns; the M4.2e-B summary gives 1 to 3.
- The wave-actions report's per-turn bound of 10.36 is corrected to 10.25 in the archived reference §12.
- M4.4b §8 says no simulator defect was found; M4.5b §8.2 found the farm-spot defect on the same preset.
- The roadmap gives the shared critic's large-gap recovery as 0.03; the M4.6f report's 8.192M table gives 0.02
  (0.03 is its 4.096M value).
- The M4.6e pre-registration says the critic moves +0.07 for +8 invested; the report (§2.2) says +0.03.
- The roadmap says the user dropped "the ending experiment proposed after M4.7"; the M4.7 report recommends no ending
  mechanism (§6), and no archived document contains that proposal.
- M4.6a proposed a gap-aware pilot with a 0% control (M4.6c-2); what ran as M4.6c was C1 against the baseline with
  no 0% control, and no source says why.
- The OM2 report gives the belief's share of exploitation as "about 28%" (summary, of +7.2) and "about 40% more"
  (§6, on top of +5.1); the denominators differ.
- Fixed-opponent games were 36% to 37% of learner transitions in M5a readiness and 33% in M5b training.
- The roadmap §BG reports NC1's weak-side Recall gain (+0.46) without the report's "exploratory" label.
- The M2–M3 report gives about 3.5 CPU-hours for its campaign and the overnight log about 3.

## 8. Limitations and open questions

Questions each era left, with what later work did about them.

- Exact-game magnitudes depend on hand-set parameters, and reported frequencies depend on equilibrium selection
  (Phase 1). Not revisited.
- The champion games have only exploitability lower bounds (M3 onwards). Never resolved; M5's attackers are the
  strongest bounds.
- Shove dominance at equal power persisted through plate gold under the econ candidates (M4.3d); M4.5c found Shove
  modal in 22% of states after the integrated revision.
- Last Hit, Thin and Slow Push coincide in 68% to 77% of turns (M4.4c §7). Still open; the closeout lists Slow Push
  as weak.
- Natural power gaps are about 0.1% of experience, so zoning and early-game claims rest on constructed or initialized
  states (M4.5c §9, M4.6a). Gap training and archive starts were tried; the exposure stayed small (M4.6e §3).
- The critic is blind to level and to large gaps (M4.6f §6). A from-scratch separate critic is untested; NC1's
  strength features are a candidate.
- E_h depends on the continuation policies and values gold only; gold leaders behind in XP or HP lost more than
  predicted, cause not identified (M4.8 §3.5, §6).
- M5 could not attribute its competence cost to any one ingredient, pure self-play had not converged, the
  exploitability difference between arms is unresolved, and no human was ever evaluated (M5e §11).
- Provenance: 212 result files record a dirty working tree, the M5 code digest matches no commit (numbers still
  reproduce at HEAD), and the pre-registration chain is anchored only by git (audit R1, R3). A `reproduce` re-run on
  a clean commit is still to do.

## 9. Lessons for future research

From the reports:

- Keep an exactly solvable arena as an oracle. Only it showed self-play exploitability rising
  ([M2–M3](rl/m2_m3_implementation_report.md) §D.3).
- Test whether the game rewards a strategy before blaming the learner. The wave-only harness settled Clear dominance
  without training ([validation](rl/m4_3b_validation_report.md) §8).
- Fixed constants that stand in for consequences can reverse incentives
  ([zoning](natural_waves/m4_2e_zoning_report.md); [design philosophy](../design_philosophy.md) §3).
- Single levers are inert when equilibrium play never pays their price, so screen combinations
  ([economic calibration](m4/m4_3/m4_3c_strategic_economics_calibration.md) §4.1).
- A terminal event one player controls invites prolongation (same report §7).
- Judge strategy by paired counterfactuals, not action frequencies, and never condition on the outcome of a
  simultaneous guess ([M4.4c](m4/m4_4/m4_4c_strategic_investigation.md) §3.5, §5.5).
- More states beat longer rollouts; bootstrap with the critic and validate on a subset
  ([horizon study](m4/m4_5/m4_5c_horizon_study.md) §5).
- Use the three-way test: does any strategy convert, does the policy, can a learner learn it
  ([M4.6a](m4/m4_6/m4_6a_report.md) §8.3).
- Power a screen before applying its rule ([M4.6b](m4/m4_6/m4_6b_investment_report.md) §9).
- Never read critic values as evidence in rare states ([closeout](m4/m4_closeout_report.md) §5.2), and state a null
  as "not demonstrated within the tested budget".
- Settle pending consequences before reading a checkpoint lead ([M4.8](m4/m4_8/m4_8_lane_exit_report.md) §4.1).
- The opponent distribution, not the belief, drives exploitation ([OM2](opponent_modelling/om2_pilot_report.md) §6).
- Cheap supervised probes predicted every PPO outcome in NC1; under Adam a wider network takes larger steps, so
  compare architectures at a retuned learning rate ([NC1](capacity/nc1_report.md) §7, §8).
- Generate published results from a clean committed tree, and let a demonstration say only what the engine says
  ([audit](audit/repository_audit_2026-10-22.md) §5).

*(synthesis)*

- Audit legality and visitation before reading behaviour. The M3 Freeze, the M4.3a Recall and the M4.5b farm-spot
  defect were each found by looking at the game, not the learner.
- Test each ingredient of a borrowed recipe separately; the OpenAI Five mixture and M5's package both combined parts
  with different effects.
- A "safe" matched pilot means only "no worse than an equally trained control"; OM2's exploratory comparison with
  the common start foreshadowed M5's competence cost.
- Pre-registered outcome classes that keep inconclusive results inconclusive let M5's exploitability contrast stand
  as inconclusive instead of being read as equality.
- Every protocol needs an outcome row for the arm that loses.

## 10. What happened to the code

The reorganization of 2026-11-08 removed the experiment packages that nothing maintained depends on: the v1 and
revision experiments, the M2 champion experiments, the M3 and OpenAI Five scripts, most M4.2 to M4.3 study
packages, the M4.7 and M4.8 drivers, OM1's stages and every study's report generators and command-line entry points.
The parts that the final campaign, the demo and the remaining tests import are kept in `experiments/` under their
study names ([experiments/README.md](../../experiments/README.md)). Everything removed is at commit
[`f613f0a0`](https://github.com/jcj1337/lol_game_theory_laning/tree/f613f0a00a2d46fe717de7e8c79077fa9e0310c3), where
the commands in the archived reports run as written; [MOVES.md](MOVES.md) lists every moved and removed path.
