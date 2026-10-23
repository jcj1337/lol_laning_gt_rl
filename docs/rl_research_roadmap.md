# RL Research Roadmap: From Exact Stochastic Games to Multi-Agent Reinforcement Learning

> **This is the canonical project roadmap.** The summary below (updated 2026-10-22: M5 is complete, committed by the user in a07e3fdf. Its pre-registered outcome 2 is that the opponent-modelling package pays at a competence cost. Both arms retain four of five strategic findings. Dedicated attackers exploit both arms, and the difference between the arms is inconclusive. M6 is under way. M6a built and verified the session core, and the user approved the interface design. M6b built the maintained application, `web/` over the FastAPI layer, verified in the browser and awaiting the user's acceptance. M6c was narrowed on 2026-10-22 to a small collection of guided replays within Watch, shown as Sequences; the user approved the format and the first sequence and put further sequences on hold, §BE. M6d has started: the demo's misstatements found by the audit are fixed and the research overview is written, §BF)
> states where the project is and what comes next. Sections A–K are the original plan
> written before M1 (kept unchanged for the record; its milestone list was later revised, see the notes in section D).
> Sections L–BA record each milestone's outcome in the order it happened, with links to the full reports (M4.5b: §AJ;
> M4.5c pilot: §AK; M4.5c training: §AL; M4.5c completion: §AM; M4.6a–b: §AN; M4.6c: §AO; M4.6d: §AP; M4.6e: §AQ;
> M4.6f: §AR; M4 closeout: §AS; M4.7 horizon study: §AT; M4.8 lane-exit evaluation: §AU; OM1 opponent-modelling feasibility: §AV;
> OM2 opponent-modelling design and matched pilot: §AW; M5a campaign design, pre-registration and readiness: §AX; M5a
> correction pass, M5b training and the initial evaluations: §AY; M5c strategic battery and limitation cells: §AZ;
> M5d core attackers and the M5e closeout: §BA; the M6 plan: §BB; M6a, the session core and the prototypes: §BC;
> M6b, the playable application: §BD; M6c step 1, the guided-replay format and the first example: §BE; M6d part 1,
> the demo corrections and the research overview: §BF).
> The M5 plan (M5a–M5e) is under "M5 plan" below, and the M6 plan (M6a to M6e) under "M6 plan".
> The README and `docs/README.md` link here.

## Roadmap summary (current position, 2026-10-22)

**Where the project stands.** The RL-ready environment (M1), the champion mechanics (M2) and an initial PPO / self-play
programme (M3) are complete. M3's diagnostics showed that the game itself, not the learner, limited what could be
learned (an inert Freeze, horizon effects, sieges never visited), so M4 was deliberately expanded: an integrated laning
model (economy, progression, natural waves), a PPO pilot that exposed Clear / Shove dominance as a property of the
game's incentives, a four-action wave-management redesign and an economic calibration, all *before* committing to
substantial PPO research. **M4.4a** made the PPO pipeline and its strategic diagnostics ready for the four-action game.
**M4.4b** ran the controlled comparison: 3 economies × 3 seeds × 2,048,000 transitions plus a matched 2× extension.
The calibrated economies produce state-dependent wave management, short freezes that are preserved and broken but deny
little, a targeted response to the tower-aggro correction and conditional defence, while the original economy remains
Shove-dominated (§AG). **M4.4c** tested the apparent gaps with multi-turn paired
counterfactuals (5,982 states): the learned policies match or beat every hand-designed alternative against their own
opponent; interrupted Recalls are a rational guessing game, freezes are worth only what the opponent concedes, zoning
*is* learned (through kills and forced Recalls), Vision-first does not pay in short sieges, and three of the four wave
actions coincide in most states. The remaining imperfections are simulator incentives and self-play's opponent
distribution, not PPO failures (§AH). The review of M4.4c's design questions led to a focused mechanics revision,
**M4.5**, implemented as one unit because its parts interact: **M4.5a** (committed, revision 2) adds a shared off-lane
Vision node with a contestable Ward action, minion retaliation and positional aggro with per-tick exclusive targeting
and level-scaled minion damage, XP range 2, wave maintenance during trades and distance-limited wave damage, as the
EXPERIMENTAL candidate `m45_unified()`; every earlier preset is unchanged (§AI). **M4.5b** (completed, reviewed)
calibrated the integrated mechanics together with paired scripted simulations, restricted games and a holdout: it found
and narrowly corrected a pre-existing farm-spot defect of mutual Shoves, moved minion retaliation to the exchange tick,
retained every other provisional number, and selected the candidate `m45_calibrated()` (GO WITH CAVEATS; §AJ). The
review accepted its caveats, including the mid-to-late minion share (`minion_damage_growth` stays 1.08). **M4.5c**
trained fresh shared-policy PPO on it. Its pilot is complete: one seed, 1,024,000 agent transitions, with
correctness verified on the preset. The policy beats every fixed M4.5b strategy and the out-of-distribution M4.4b
network, and it was still improving at the end. The approved 6 seeds × 8,192,000 campaign is trained (§AL): all six runs are healthy, bit-for-bit reproducible, consistent across
seeds and still improving head to head (+0.96 against their midpoints, +0.91 against other runs' midpoints) while flat against
the fixed suite since about 4M. The pre-registered extension criterion was met. **M4.5c is complete (§AM):**
- the matched 2× extension (6 × 16,384,000) is stronger in all 36 pairings (+1.03), and annealing alone explains
  nothing (−0.07);
- the comprehensive evaluation, the 50-turn strategic battery (validated: 150-turn re-runs changed nothing) and six
  best-response attackers were run;
- PPO self-play discovered conditional wave management, conditional freezing, selective Trading, concession under
  threat and Recall timing, each shown by paired counterfactuals, and rationally avoided Wards, sustained freezes and
  bullying, which this model does not reward against competent play;
- the policies are exploitable by target-tuned aggression (+1.0, quarter-budget attackers) and leave value against
  conceding opponents.

**GO WITH CAVEATS** for the next phase. M4.4d's robustness and
exploitability work is postponed into M4.5c's evaluation rather than run on a simulator being revised.

**M4.6 (design review and diagnosis, §AN–§AR).** M4.6a–d diagnosed the remaining strategic gaps of the M4.5c / M4.6c
policies; M4.6e diagnosed the critic, and M4.6f ran the one targeted learning pilot.
- **M4.6b–c: the investment curve.** M4.6b screened alternative investment curves. M4.6c trained the C1 curve against
  the baseline: the primary result is inconclusive, and C1 (`m45_calibrated(invest_gain=3.24, invest_scale=64.0)`)
  stays provisional and is not adopted.
- **M4.6d: freezes, concession and specialists.** The freeze and concession choices exist and the policies make them
  correctly in natural states; no mechanics change is justified. The critic is calibrated in natural states but nearly
  blind to strength.
- **M4.6e (completed, reviewed): the value diagnostic.**
  - The observation is exact and lossless and carries strength at an ordinary scale.
  - The trained shared actor–critic trunk discards strength in the layer the value head reads.
  - A separately trained value network learns the strength premiums from the unchanged observation.
  - Large gaps are essentially absent from training: duel ≥ 3 never occurs.
  - Better values do not materially improve the ranking of the decisions M4.6d flagged.
- **M4.6f (completed, reviewed): the separate-critic pilot.** The user authorized the conditional pilot.
  - **Design.** Six matched seed pairs continued the M4.6c C1 policies: the shared network against the same actor
    with its own critic trunk.
    - Two budgets: 4.096M transitions, then the pre-registered extension to 8.192M.
    - These are two separate continuations from the same start, not one 12M run.
    - The arms were matched at the start and diverged after the first update.
  - **The separate critic learns part of the strength value; the shared one did not, in either continuation.**
    Plausible-edge recovery 0.45 vs 0.21, pure edges 0.31 vs 0.02, large gaps 0.32 vs 0.03, all 6/6 seeds and still
    rising, with no natural-state cost. That is about a third of the strength premium; it is still nearly blind to
    level.
  - **No material behavioural improvement was demonstrated within the tested budget.**
    - Strength conversion and defence, freezes, weak-side Recall windows and natural states show no difference beyond
      noise (resolution about ±0.2).
    - Head to head there is no material difference (+0.06 ± 0.23). The powered head to head (+0.13 ± 0.09) suggests a
      small edge, not established.
  - **Classification:** pre-registered outcome 4 (V1 +0.24, just short of +0.25). Substantively, better values without
    a demonstrated behavioural benefit.
  - **The limitation is accepted and the investigation closed.** The separate critic is a verified opt-in option, not
    an adopted default.
- **M4 closeout (completed, reviewed; §AS, [report](m4/m4_closeout_report.md)).**
  - **Recommendation: ready for M5 with documented limitations.** No strategic blocker; the evidence supports closing
    M4.
  - **A strategic acceptance table** classifies each goal. Used: wave management and Shove, Recall, contesting
    freezes, the weak side's choices, conversion of plausible edges against the tested alternatives. Missed: guarding
    in the narrow favourable-freeze window, large gaps, unfamiliar opponent styles. Weak: late leads, Slow Push in most
    states, Wards.
  - **Recommended setup (decided by the user on 2026-10-17; M5 continued `ext/baseline:s` instead of a fresh
    initialization):** baseline `m45_calibrated()` (C1 stays a provisional opt-in), the shared network, fresh
    initialization, M4.6c archive starts, 6 × 32.768M.
  - **Five overstatements in the M4.6e–f interpretations corrected.**
  - **M5's scope proposed;** first task M5a, the pre-registration and readiness check.
- **M4.7 early-lane horizon study** (completed, reviewed; §AT, [report](m4/m4_7/m4_7_horizon_report.md)).
  Does the long continuing objective put too much weight beyond the laning phase (before about level 14)?
  - **The lane phase is turns 0–93.** Level 14 arrives at turn 94, almost like a clock, in both curves.
    - It is 17% of training transitions.
    - It carries 61% of the γ 0.99 objective's weight from s0, and 36% of the weight of a decision taken inside it.
    - Kills, plates, Pressure and freezes are concentrated in it.
  - **Baseline and C1 are the same game there,** except that C1's early earned leads persist. C1's extra value is
    mostly paid after level 14.
  - **Delayed rewards do not drive the early conclusions.** The paired benefits of early actions and advantages arrive
    mostly before level 14 and keep their sign and order under H 80, 60 and 40.
  - **A matched pilot** (2 seeds × 4.096M continuations, γ 0.99 against γ 1 − 1/60) produced:
    - no reliable lane-phase gain (+0.26 ± 0.17 and −0.09 ± 0.17);
    - a loss against fixed opponents (−0.69 and −0.36 ± 0.21), mostly after the lane phase and against scripted
      opponents;
    - a few changed decisions (more freeze holding, fewer late Recalls).
  - **Outcome 5 (regression), pre-registered.** It would be 6 (inconclusive) under the analysis code's first, unpaired
    interval, corrected to the protocol after the results were seen.
  - **Recommendation: keep γ 0.99,** and analyse the lane phase in elapsed-turn windows. Nothing adopted.
  - *Correction (M4.8, §AU; the M4.7 report is unchanged).* Read with the lane-exit readout at both ends:
    - from turn 50 to 94 the baseline's leads change little (−0.10 ± 0.10, not "shrink by a fifth") and C1's grow
      (+0.15 ± 0.09); the C1 − baseline contrast (+0.25 ± 0.13) is unchanged;
    - the pilot's seed-0 lane-phase lead (+0.26) was partly consequences still pending at turn 94 (E_20:
      +0.17 ± 0.18).

**Decisions of 2026-10-17** (the user; approved direction):
- Keep the continuing training objective: the earned-gold reward, γ 0.99, no terminal state.
- Do not add an actual termination around level 14. The ending experiment proposed after M4.7 is dropped, and further
  discount experiments are not the immediate plan.
- A "lane exit" is an **evaluation checkpoint**: short repeated continuations account for pending farm and Recall
  consequences.
- XP, HP and unresolved positional advantages stay visible separately; the economic readout is not a complete
  "who is winning" score.
- Validate the evaluation method and apply it to the matched baseline/C1 policies before making configuration
  recommendations.
- Bayesian opponent modelling is desired future scope: design it next, then run a controlled pilot before the final
  M5 campaign.
- The critic investigation stays closed (M4.6e–f).

- **M4.8 lane-exit evaluation** (reviewed; §AU, [report](m4/m4_8/m4_8_lane_exit_report.md)).
  Evaluation only: no training, and no change to rules, reward, objective, defaults or checkpoints.
  - **The readout.** E_h = the earned-gold lead now (invested + unspent) plus its expected change over h repeated
    continuation turns by the policies that produced the state; undiscounted, no critic tail. XP, HP, absence and
    the wave are reported beside it, never converted.
  - **Validated (Stage 1).**
    - Accounting, orientation and restoration are exact.
    - Selected wave/Recall cases move as their mechanisms predict, with the recorded events to show it. A returner
      facing a due opponent gains +2.6 by h = 10.
    - Equal raw gold with a completed reset is worth +0.6 to +0.9.
    - Predictions match the games' recorded continuations (slope 0.99 at h = 10 and 20).
    - The pre-registered classification is **useful**, and the window rule selects **E_20** (one reset cycle; 95% of
      pending items resolved; flat to h = 40).
  - **Applied (Stage 2, turn 94).**
    - Raw gold misleads materially in about two thirds of natural lane-exit states (SD of the true adjustment 1.16
      against a mean |lead| of 3.5; the leader flips in 4%, 16% of close states). Reset status, HP and the wave
      explain it.
    - Baseline and C1 are indistinguishable on every lane-exit readout.
    - M4.7's curve contrast stands. Its 50 → 94 lead conversion is refined (the baseline's shrinkage halves, C1's
      leads grow), and its horizon pilot's one lane-phase gain disappears.
  - **Recommendation** (no default changed): report lane-phase outcomes as D0 → E_20 with the non-gold panel; the
    baseline stays the recommended configuration; next is the Bayesian opponent-modelling design.

**Decisions of 2026-10-17, after review of M4.6d–M4.8** (the user reviewed and committed that work, e051b427):
1. **M5 configuration: the baseline `m45_calibrated()`.** C1 stays a provisional opt-in. The shared critic, as the
   closeout recommended, is assumed unless the user changes it.
2. **The lane-exit readout is adopted** as the standard lane-phase economic readout for the opponent-modelling pilot and
   M5: D0 → E_20 ± 95% with the non-gold panel beside it.
3. **Next:** the opponent-modelling feasibility study OM1 (identifiability and value of information with existing
   policies, evaluation only), before the design itself.

- **OM1 opponent-modelling feasibility** (reviewed; §AV,
  [report](opponent_modelling/om1_feasibility_report.md)). Evaluation only, with existing policies; no training.
  - **Pre-registered outcome 1: feasible and worthwhile.**
  - **Identification is fast.** Within 10 observed turns the believed type already has the right best response in
    96% of games, and in 99% within 20.
  - **Knowing the type pays.** A Bayesian switcher built from existing policies beats π by +0.90 ± 0.27 at the
    lane-exit readout, 83% of an oracle.
  - **The value is concentrated against scripted styles:** +1.36 against them, +0.21 ± 0.11 against learned
    opponents.
  - **Opponents outside the library are the risk:** the switcher loses −0.8 to −1.5 against stochastic style
    mixtures.
  - **Design proposals:**
    - act on the response, not the type;
    - a "none of the above" fallback;
    - belief-conditioned trained responses;
    - an explicit decision on how the opponent's action is observed.

**Decisions of 2026-10-17, after OM1:** the design's scope is **C, "both, robustness first"** (archetypes and competent
learned opponents; train against mixed and switching opponents; a gain must come with no loss against competent or
unfamiliar opponents), and the opponent's **previous action becomes an opt-in observation field**.

- **OM2 opponent-modelling design and matched pilot** (reviewed: committed by the user, a6e9b63f; §AW,
  [report](opponent_modelling/om2_pilot_report.md), [design](opponent_modelling/om2_design.md),
  [pre-registration](opponent_modelling/om2_preregistration.md)).
  - **Design (reviewed, carried into M5 as arm A):**
    - a Bayesian belief over 11 library types (6 scripted archetypes, 5 learned policies) and two catch-alls,
      "mixed" and "unknown", with forgetting λ = 0.97;
    - the belief's 13 features are appended to the policy's input;
    - training mixes 50% self-play with 15 fixed opponents, including mixtures and style switchers.
  - **Implemented behind opt-in switches** (no default, preset, digest pin or checkpoint changed):
    - the observation field `obs_last_action` (103 entries with `m45_calibrated()`);
    - the reusable belief module `lanegame/opponent/`;
    - two collector hooks in the shared trainer (default training verified bit-identical);
    - zero-weight network widening;
    - the OM2 study package and 11 tests.
  - **Readiness:** 17 of 17 smoke checks pass. Both arms start as exactly `ext/baseline:s`, and a re-run is
    bit-identical. Throughput is about 1,580 transitions/s for the treatment and 2,650 for the control.
  - **The pilot** (compute approved by the user and run on 2026-10-17): control (belief held at the prior) against
    treatment (the belief), 3 seed pairs × 4.096M from `ext/baseline:s`; 72 minutes of training.
  - **Pre-registered outcome 1: modelling pays and is safe.** Treatment − control at the lane exit:
    - archetypes +2.03 ± 1.72, positive in every seed;
    - competent learned opponents +0.43, held-out mixtures and switchers +0.95, head to head +0.32 (2 of 3 seeds);
      no loss anywhere.
  - **The belief works as designed.** It identifies 5 of 6 archetypes within 10 turns, reads competent opponents as
    learned types, and follows style switches in 10–60 turns. The trained policy acts on it.
  - **Exploratory (not pre-registered), against the common start π_s:**
    - Training against the styles alone gives +5.1 against the archetypes; the belief adds the further +2.0.
    - Against competent opponents both arms end slightly below π_s: the control −0.59 (every seed), the treatment
      −0.16 ± 0.30.
  - **Recommendation:** carry the tested form into the M5 design, and make exploitability (dedicated attackers, an
    imitate-then-switch opponent) and competence against pure self-play M5 measurements.

**Decision of 2026-10-17, after OM2** (the user, in the M5a brief; approved direction): opponent modelling enters M5,
and M5 compares the whole package with continued pure self-play. The brief fixed:
- the baseline `m45_calibrated()`;
- the objective (gold difference, γ 0.99, continuing time, bootstrapped truncation, no termination at level 14);
- the shared actor–critic;
- D0 → E_20 lane-phase reporting with the non-gold panel;
- rollout evidence in strength-gap states.

It also gave the starting proposal: two arms, six seed pairs continuing `ext/baseline:s`, OM2's settings, a fixed
library, and 32.768M transitions per run with a matched 16.384M checkpoint.

- **M5a: the campaign's design, pre-registration and readiness** (completed; §AX;
  [readiness report](m5/m5a_readiness_report.md), [design](m5/m5_design.md),
  [pre-registration](m5/m5_preregistration.md) frozen, then amended once before training as version 1; the user
  authorized the campaign on 2026-10-17, and it ran).
  - **The arms.**
    - A, opponent modelling: OM2's treatment collector exactly, with the previous action, the 13-feature belief, 50%
      self-play and 50% the 15 fixed opponents.
    - B, pure self-play: the M4.6c continuation, verified to be 100% current-policy; its snapshot pool is never played.
    - Both start as `ext/baseline:s` (verified bit-identical outputs) and continue 32.768M transitions, with
      the M4.6c start mixture and PPO settings.
    - The contrast estimates the package effect (information and opponent distribution together), not either
      ingredient.
  - **The package** `lanegame/experiments/rl/m5/`:
    - training, manifests, restarts, health checks;
    - the M5b–M5d instruments for both input formats;
    - the attackers;
    - the pre-registered statistics;
    - gated stages.
    - An opt-in `observe_turn` hook in the four shared rollout loops lets the historical instruments run a
      history-dependent policy. Stored records reproduce exactly through it.
  - **Readiness: 11 of 11 parts pass** (about 20 minutes of experimental compute).
    - The lineage verified; a disjoint seed block; both smokes re-run bit-identical.
    - Identical game starts in both arms.
    - Every om training input rebuilt independently by the evaluation policy.
    - Fixed-opponent games are 51% of games but 36–37% of learner transitions.
    - Six workers is the efficient count.
  - **Measured budget:** training 9.4–11.8 h, evaluation 13.7 h, attackers 10.3 h (a quarter of the target's
    budget) or 20.6 h (half). About 34–36 h in all.
  - **Proposals:**
    - quarter-budget attackers;
    - an adaptation-exploitation test;
    - the full battery;
    - six workers, om runs first.
  - **A resolution caveat:** with six pairs, a near-zero competence effect would most likely be reported as
    inconclusive.

**Decisions of 2026-10-17, after M5a** (the user's correction-and-campaign brief):
- six pairs, 32.768M per run, six workers with the om runs first;
- a bounded correction pass first;
- then main training and the initial evaluations: monitoring, opponent families and head to head, including the
  midpoints and long games;
- the battery, the cells and every attacker stage not authorized;
- the om_table attackers reclassified as an optional follow-up, the core attacker stage being 12 quarter-budget
  attackers.

- **M5a corrections and readiness** (completed, §AY; [amendment 1](m5/m5_amendment_1.md)).
  - **The eight review findings, fixed before any training:**
    - an inconclusive gain now stays inconclusive (new headline outcome 6);
    - a fixed 15-hypothesis Holm family with one-sided tests and TOST for S1–S6;
    - S5 renamed (S5b is competence against historical opponents only, not P2a);
    - six-seed completeness;
    - the S8 analysis;
    - live training health monitoring with an evaluation gate;
    - core and optional attackers, with the live/table cross-evaluation implemented;
    - a verifiable amendment chain.
  - No threshold, arm, budget or family changed.
  - Checks: the full suite 1,534 passed, and both arms' monitored smokes are bit-identical.
- **M5b training and the initial M5d evaluations** (completed 2026-10-18, §AY;
  [report](m5/m5b_initial_report.md)).
  - **Training.** 12 runs, 138/138 health checks, no alarm, 10.9 h. The sp arm was still improving at the end; the om
    arm was not shown to be.
  - **Pre-registered outcome 2: the package pays at a competence cost.**
    - P1, the exploitation gain: +9.26 ± 0.30, demonstrated, against library-like scripted styles. There is no gain
      against conditional styles on a competent base.
    - P2a, against competent opponents: −0.63 ± 0.35, a material loss.
    - P2b, head to head: −0.44 ± 0.28, a material loss. Both losses hold in every seed.
  - **Secondary, all determined under Holm:** robustness to misspecified opponents preserved (+5.20); long-horizon
    competence, the long head to head, the cross-seed head to head and the midpoint competence all material losses;
    the midpoint gain demonstrated.
  - **Exploratory: pure self-play kept improving over the common start (+0.9); the package did not (+0.2, interval
    including 0).**
  - **Not established:** strategic replication, the limitation cells, exploitability.

**Decision of 2026-10-18** (the user): run the strategic battery and the limitation cells. The full design ran.

- **M5c strategic battery and limitation cells** (completed 2026-10-19, §AZ; [report](m5/m5c_strategic_report.md)).
  - **What ran.** The full M4.5c battery on all 12 finals, the registered 150-turn re-run of the 51 of 144 (final,
    family) pairs that failed horizon validation, and the seven M4 limitation families. 16.4 h in all.
  - **Replication.** Conditional wave management, conditional freezing and the reply to freezes, selective trading,
    and Recall timing replicate in both arms (6 of 6 seeds).
    - **Concession under threat does not replicate in either arm** (om 3 of 6, sp 4 of 6). The failure is in the
      constructed zoning cells only; the weaker-side lead cells hold. *Qualifier added 2026-10-22 (audit R2):* sp
      fails only in seeds 4 and 5, whose zoning family the registered rule re-read at 150 turns; at 50 turns sp
      holds in all six. The om failures are at 50 turns.
    - The sp freezing verdict is fragile: three seeds sit within 0.035 of the margin (0.014, 0.015 and 0.034;
      corrected 2026-10-22 from "within 0.03").
  - **Cross-arm limitation cells.** The om finals are worse than the sp finals in every family, from the same
    states: strength use −0.53, freezes −0.26, Recall windows −0.42, natural states −0.54, large gaps −0.39. Equal
    and lead states suffer alike, matching the M5d competence loss. One caveat: the om final starts these cells with
    its prior belief and no previous action.
  - **Unchanged from M4 in both arms:** the lead premium, the weaker side's choices, the cost of forced plans, and
    the missed guard in the favourable-freeze window (about twice as large in om).
  - **Horizon validation.** 51 of 144 pairs failed, mostly on bias. An unregistered diagnostic shows 150 turns agree
    with 300 turns in all 51.

**Decision of 2026-10-19** (the user, approved direction): run the 12 core attackers at the quarter budget with their
registered evaluation, then close M5 (M5e). Not authorized: the om_table ablation, extra seeds, attacker restarts for
poor performance, budget extensions, new arms, mechanics changes and experiments on the cause of the competence loss.

- **M5d core attackers** (completed 2026-10-19, §BA; [M5e report §5](m5/m5e_closeout_report.md)).
  - **What ran.** 12 attackers, six against each arm's finals, 8.192M transitions each, each starting as a copy of
    its target. All 12 are healthy (180 of 180 checks, no alarm). Training took 5.96 h and evaluation 0.52 h.
  - **Lower bounds.** Both arms are exploitable. At the lane exit the attackers lead the om finals by +2.96
    [+0.32, +5.61] and the sp finals by +1.15 [+0.34, +1.96], in all six seeds each.
  - **S6, the registered contrast** (om − sp +1.81 [−0.56, +4.19]), is **inconclusive**: no increase, decrease or
    equivalence is shown. One om seed (+7.91) drives much of the spread.
  - **No transfer.** Against the other arm's final and the common start every attacker does worse than its starting
    point. The attacks are target-specific.
- **M5e closeout** (completed 2026-10-19, committed by the user, a07e3fdf; [report](m5/m5e_closeout_report.md)).
  - **The secondary Holm family is final** (15 hypotheses). Every earlier claim stands, and S6 is inconclusive.
  - **Recommendation for M6.** Present both arms as a trade-off, with the modelling package for recognizable, exploitable
    opponent styles and pure self-play for competent play. Neither is uniformly better, and the style gain and the
    competence loss are never combined into one score.
  - **Reproducibility.** `results/m5/reproducibility.md` lists the commands, seeds, protocol version and a
    key-checkpoint table. The SHA-256 of every file is in `results/m5/inventory_files.json.gz`, and every network's
    identity in `results/m5/inventory.json`. It records commit 312d118d with a dirty working tree, from before the M5
    commit, and the M6 plan proposes a re-run on the committed tree (M6d). There are 15 viewer recordings in
    `results/m5/traces/`.
- **Corrections to earlier M5 summaries** (2026-10-19; the M5c report stays unchanged).
  - *Constructed zoning is open (suggestive), not missed.* The M5c goal table labelled the om arm "missed". The best
    alternative is +0.168, 95% interval [−0.006, +0.343], so it is not confirmed. Concession under threat still does
    not replicate under the registered rule.
  - *Same findings, not the same choices.* Both arms retain four registered strategic findings. They differ in what
    they achieve against competent opponents, and passing the same battery criteria does not show identical actions.
  - *The cold-start caveat stands.* A deficit that grows from 50 to 150 turns does not exclude an early cold-start
    effect. The independent support is the s0 evaluations, where both arms start naturally.
  - *A package contrast.* M5 does not isolate whether the competence cost comes from the belief, the opponent
    distribution or the reduced self-play exposure.

**Decision of 2026-10-19** (the user, approved direction): M6 builds a new web application, later deployable in
public, whose centre is a human playing one seat against a trained M5 policy, either pure self-play or the
opponent-modelling package. The Streamlit viewer is not reworked. It is archived as legacy during implementation.
Policy-versus-policy observation and guided replays of verified research games are supporting modes to consider.
Public deployment is an intended capability, not an authorization to deploy or to buy hosting.

- **The M6 plan** (written 2026-10-19; approved with one change and revised on 2026-10-21; §BB;
  [plan](m6/m6_plan.md)). M6a has started (below). The web application does not exist yet, and the viewer has not
  moved.
  - **Five milestones.** M6a defines the product and proves the backend's correctness rules on the real finals,
    then reviews two or three throwaway prototypes. M6b builds the playable application. M6c adds the verified
    replays and the findings. M6d covers documentation, the three reproduction bundles and the Streamlit archive.
    M6e prepares the release and ends at a deployment approval checkpoint. *Revised 2026-10-22 (the user):* M6c is
    a small collection of optional guided replays within Watch, in three steps (the format and one example; the
    user's review; a few more examples). The research question, method, aggregate results, comparisons and
    limitations move to M6d's research overview (§BE).
  - **The initial release.** Play against `sp:0` or `om:0` from a seed, with backend-validated legal actions,
    both actions revealed together, the belief hidden or shown, stopping without a winner and a downloadable record
    that replays bit for bit. It also has the 15 verified M5 recordings with provenance and the findings page. A
    continuation-based E_20 comes later. *Changed 2026-10-21:* live bot-versus-bot (Watch) moved into the initial
    release. *Changed 2026-10-22:* the release keeps the three verified seed-0 replays (M6b) and adds the guided
    replays; the other 12 recordings stay research artifacts, and there is no findings page in the application.
  - **The architecture (proposed).** React with TypeScript and Vite, and FastAPI over the unchanged simulator and
    policies, in one container. One process holds the sessions in memory, with one simulation thread. Replays are
    static files that work without the inference backend.
  - **Measured serving cost.** About 1.2 ms per pure self-play turn and 2.7 ms per modelling turn, 22 KiB per
    session, and about 250 MB per process. The modelling opponent also needs its 5 library networks, which are
    git-ignored. `results/m5/inventory.json` lists their SHA-256, but the key-checkpoint table of
    `reproducibility.md` does not.
  - **Nothing new in the game.** M6 adds no mechanics, rewards, objectives or training. Optional research stays
    separate.

**Decisions of 2026-10-21** (the user's M6a brief; approved direction):
- The initial release has two primary modes. In Play a human chooses discrete actions against `sp:0` or `om:0`. In
  Watch two bots play a bounded match that the visitor explores through a timeline. Live bot-versus-bot simulation is
  no longer a post-release feature.
- A Chess.com-style turn history, shared by Play and Watch, governs navigation. Browsing never changes the game,
  and no history data reveals an unresolved action, a future state or a hidden belief.
- The frontend has explicit clarity requirements (plan §5.2). The old Streamlit UI is not a visual template.
- React with TypeScript and Vite in `web/` in this repository, over FastAPI, unless a blocker appears. `docs/` is
  tracked again.
- Scripted opponents are a design decision. Their inclusion is not a prerequisite for M6a.

- **M6a: the session core and the interface design** (completed 2026-10-21, §BC;
  [technical report](m6/m6a_technical_report.md), [decision record](m6/m6a_decision_record.md)).
  - **The session core** (`lanegame/demo/`) is built and verified. It has a registry that checks all 7 checkpoint
    files against a tracked manifest before any is unpickled, sessions with human, model and scripted seats, a
    recording with per-turn state digests, deterministic replay, history frames, and an action menu whose reasons
    come from the engine's legality predicates.
  - **Verified on the real finals.** The three seed-0 M5 recordings re-simulate exactly (actions, `payoff_A` to the
    last bit, belief log posteriors identical). Every check of the brief has a passing test (29 tests). Nothing in
    the game, the networks or the results changed.
  - **Measured.** A full turn with frames and JSON costs 1.4 ms (`sp:0`) and 2.3 ms (`om:0`). A 150-turn history
    rebuild takes about 0.09 s (22 KB gzipped). A bounded 150-turn bot game takes 0.27 to 0.48 s. About 0.1 MB per
    session.
  - **Three prototypes** (a game-like arena, a schematic board, a narrated timeline) ran from `prototypes/m6a/`,
    staged against verified recordings. After trying them, the user chose C with A's action controls and a better
    gold-lead graph.
  - **The approved design** (decision record §7) came out of six review rounds. One frosted glass dock sits over a
    diagonal top-down lane, with minions drawn as balls (one per HP), the ward spot as a bush, stepped ticks, dark
    mode and a rail layout on a phone held sideways. Play is **live** through a local JSON service over the session
    core. The reference implementation was `prototypes/m6a/glass/`, archived at `legacy/m6a_glass_prototype/`
    once M6b implemented it.
  - **M6b, the playable application** (completed 2026-10-22; §BD; [report](m6/m6b_report.md)). The maintained
    application: `web/` (React, TypeScript, Vite) served by `python -m lanegame.demo` (FastAPI over the session
    core; Watch simulations in a separate worker process; sessions expire; a typed OpenAPI contract generates the
    frontend's types). Play, Watch, history review, reload recovery, the record download and the stale-response and
    unanswered-request rules were verified in a scripted browser journey with screenshots. The prototypes were
    removed. The user's acceptance of the build is pending.
  - **M6c, sequences within Watch** (first called guided replays; narrowed 2026-10-22; step 1 completed the same
    day, review pending; §BE; [report](m6/m6c_guided_replays.md)). A sequence is a readable JSON file
    (`lanegame/demo/guides/<id>.json`): a
    verified research recording, a start position and three to five events, each with a short note grounded
    in the recorded state and an optional highlight of a champion, tower, wave, crashed minions or the ward spot.
    The server validates every sequence against the verified replay at startup and serves it at `/api/guides`.
    Watch lists the sequences beside the recorded games and opens one paused at its start with a compact note
    panel; Next event and Play walk the turns to the next event and stop there, and arrows on the lane show how
    each champion and wave moved. The first one, revised after the user's reviews, "Stacking to Crashing", covers
    turns 75 to 84 of the self-play agent against the
    always-push bot: trades keep the bot from recalling, nothing clears the agent's wave, the wave stacks and
    crashes, and Pressure on the crash breaks two plates. Its five notes are pinned to the recording by a test.

**Open decisions for the user.** Acceptance of the M6b build. The review of M6c's format and first example (the
sequence, the start, the checkpoint positions, the wording, the highlights and the note panel's placement), after
which a few more examples follow. Whether scripted styles are offered as opponents, and
which ones ([decision record](m6/m6a_decision_record.md) §4); M6b offered the two trained agents only. Later, a license
before any bundle is published, and at the M6e checkpoint the provider, budget and deployment. Further mechanics or
training experiments are optional future research, not prerequisites ([M5e report §14](m5/m5e_closeout_report.md)).

**M5 status: complete (2026-10-19), committed by the user (a07e3fdf). The optional om_table ablation was not run.
M6 status: M6a complete (2026-10-21), M6b complete (2026-10-22, acceptance pending). M6c, narrowed on 2026-10-22 to
sequences within Watch (first called guided replays): step 1 (the format and the first example) complete and approved
by the user (§BE); further sequences are on hold at the user's request. M6d started on 2026-10-22: the demo's
misstatements are fixed and the research overview is written, awaiting the user's approval (§BF); the reproduction
guide, the bundles, the `reproduce` re-run and the Streamlit archive follow. M6e not started.**

**The agreed path to completion** (the sequence agreed with the user on 2026-10-16, extended on 2026-10-17; details
in "Path to completion" below):

| step | content | status |
|---|---|---|
| **focused value diagnostic (M4.6e)** | observation and measurement verification; actual training exposure; supervised value-representation study; no mechanics change, no PPO campaign | **completed, reviewed** (§AQ): no pilot recommended |
| **one targeted learning pilot (M4.6f)** | a separate critic trunk against an unchanged matched control; value learning, decision behaviour and ordinary-play regression; no bundled fixes | **completed, reviewed** (§AR; authorized by the user): the critic learns part of the strength value; no material behavioural improvement demonstrated within the tested budget; investigation closed |
| **M4 closeout** | an explicit C1-versus-baseline decision; final mechanics, objective and training setup (including whether M5 uses the separate critic); the narrow conditions where freezing pays and the remaining limitations documented; mechanics reopened only for a concrete blocking issue | **completed, reviewed** (§AS, [report](m4/m4_closeout_report.md)): ready for M5 with documented limitations; M4 closed; **setup decided 2026-10-17: the baseline** (C1 opt-in; shared critic assumed) |
| **early-lane horizon study (M4.7)** | requested by the user before the final M5 campaign: what part of play occurs before about level 14; baseline and C1 there; whether delayed rewards drive the early conclusions; one bounded matched pilot of a shorter discount (opt-in, no new termination, rewards or mechanics) | **completed, reviewed** (§AT, [report](m4/m4_7/m4_7_horizon_report.md)): keep γ 0.99; the pilot (H 60) gave no reliable lane-phase gain and cost play later and against fixed opponents (pre-registered outcome 5); nothing adopted. An actual-ending experiment proposed afterwards was dropped on 2026-10-17 in favour of the lane-exit evaluation (M4.8) |
| **1. validate the lane-exit evaluation (M4.8 Stage 1)** | the readout E_h (raw earned-gold lead plus its expected change over h continuation turns; undiscounted, no critic tail), exact accounting, restoration, paired randomness, orientation; selected wave/Recall cases, controlled variants, recorded-event accounting, calibration against recorded continuations | **completed, reviewed** (§AU, [report](m4/m4_8/m4_8_lane_exit_report.md)): exact accounting, orientation and restoration; the cases move as their mechanisms predict, with recorded-event accounting; predictions calibrated against the games' own continuations (slope 0.99); pre-registered classification **useful**, default window **E_20** |
| **2. apply it to the matched baseline/C1 policies and settle configuration recommendations (M4.8 Stage 2)** | natural lane-exit checkpoints at turn 94 (84, 104 and 50 as checks) of the M4.6c 8.192M policies; how often raw gold misleads, why, stability, baseline vs C1, earlier conclusions re-checked | **completed, reviewed** (§AU): raw turn-94 gold differs materially from E_20 in about two thirds of natural states (true SD 1.16; leader flips in 4%), explained by reset status, HP and the wave; robust to the continuation policy (correlation 0.96–0.98); baseline and C1 indistinguishable at the lane exit; M4.7's curve contrast stands and its pilot conclusion is reinforced; **decided 2026-10-17: the baseline for M5, E_20 adopted** |
| **3. design Bayesian opponent modelling** | beliefs about the opponent's style, updated during a game; type library, inference, the policy's use of the belief, evaluation design | **completed, reviewed** (committed by the user, a6e9b63f): the feasibility study OM1 is complete, reviewed (§AV, [report](opponent_modelling/om1_feasibility_report.md)): outcome 1, feasible and worthwhile; identification within 10–20 observed turns; switcher +0.90 ± 0.27 over π, mostly against scripted styles; misspecified (mixed) opponents are the risk. Scope C and the last-action observation decided 2026-10-17. **Design written and implemented behind opt-in switches** (§AW, [design](opponent_modelling/om2_design.md)), reviewed; carried into M5 as arm A |
| **4. matched opponent-modelling pilot** | a controlled pilot with the same opponent distribution in both arms (with and without the belief), evaluated with the existing instruments and the lane-exit readout | **completed, reviewed** (§AW, [report](opponent_modelling/om2_pilot_report.md); compute approved 2026-10-17): pre-registered outcome 1, modelling pays and is safe: archetypes +2.03 ± 1.72 (every seed), no loss against competent, misspecified or head-to-head opponents; exploratory: both arms slightly below their start against competent opponents (control −0.59, treatment −0.16) |
| **5. M5: final training and evaluation campaign** | two arms, opponent-modelling package against continued pure self-play, six matched pairs from `ext/baseline:s`; learning progress, strategic replication, the M4 limitations re-measured, opponent families, cross-seed and cross-arm results, dedicated attackers; lane phase read as D0 → E_20 | **M5a completed** (§AX). **M5a corrections completed** (§AY, amendment 1). **M5b training and the initial M5d evaluations completed** (§AY): outcome 2, pays at a competence cost. **M5c battery and cells completed** (§AZ): four of five verdicts replicate in both arms; the om finals are worse in every limitation family. **Core attackers and M5e completed** (§BA; committed by the user, a07e3fdf): both arms exploitable, S6 inconclusive, no transfer; recommendation: present both arms as a trade-off. **om_table ablation: optional, not run** ("M5 plan" below) |
| **6. M6: presentation, interactive demonstration and reproducibility** | a new web application where a human plays against either M5 arm, with verified research replays and the findings; a concise README and research overview; reproducible commands and three bundles (demo, evidence, training); the Streamlit viewer archived as legacy; demonstrated findings kept apart from remaining hypotheses | **approved direction** (the user, 2026-10-19: presentation, demonstration and reproducibility; a new web application with human versus policy at its centre; 2026-10-21: Play and Watch in the initial release). **M6a completed** (§BB, §BC, [plan](m6/m6_plan.md)): the session core is built and verified; the user approved the interface design on 2026-10-21. **M6b completed** (§BD, [report](m6/m6b_report.md)): the maintained application `web/` over the FastAPI layer, verified in the browser; acceptance pending. **M6c narrowed to guided replays within Watch (2026-10-22); its step 1 completed** (§BE, [report](m6/m6c_guided_replays.md)): the format and the first sequence, approved by the user; further sequences on hold. **M6d started** (§BF): the demo corrections and the [research overview](m6/research_overview.md), awaiting approval. M6e not started; the viewer not yet archived |

**Stopping principle.** A perfect critic, or the discovery of every recognizable laning strategy, is not required. A
remaining weakness that is small, confined to unusual states, or resistant to one justified targeted intervention is
documented and the project proceeds, unless it undermines the central research claim. The freeze abstraction is a
deliberate simplification (`docs/design_philosophy.md`), not a perfect reproduction of League.

### Status legend

| label | meaning |
|---|---|
| **completed (verified)** | implemented, tested, documented; its tests pass in the current regression suite |
| **completed (initial / exploratory)** | done as a bounded research milestone; its conclusions are recorded, not final |
| **in review** | implemented and tested, uncommitted, awaiting the user's review |
| **authorized** | the user approved the work (and its compute budget where relevant); not yet complete |
| **conditional** | would run only if a stated condition is met; not authorized until the user approves it |
| **planned** | specified (objective, tasks, deliverables, criteria), not started |
| **proposed** | a direction only; needs its own justification and specification before work starts |
| **approved direction** | the user decided that this comes next and in what order (2026-10-17; for M6, 2026-10-19); its details remain proposals until specified and reviewed |

### Milestones

| milestone | content | status | record |
|---|---|---|---|
| **M1** RL-ready game environment | `LaneEnv` (simultaneous moves, own-perspective observations, legal masks, zero-sum rewards, seeded chance streams), scripted agents, the exact-solver bridge, PettingZoo adapter, symmetry tests | completed (verified: `tests/test_env.py`, `tests/test_pettingzoo_m4.py` pass in the M4.4a regression run) | §J, §L |
| **M2** Champion mechanics | positions, Hold / Yield / Trade / Safe / Recall, absence and return, stochastic trades with skill, ganks with HP damage, death and respawn | completed (verified: `tests/test_champion*.py`) | §E, §M, §N; [M2/M3 report](rl/m2_m3_implementation_report.md) |
| **M3** Initial PPO / self-play research | masked PPO, frozen-snapshot and shared-policy (OpenAI Five-inspired) self-play, approximate best responses, strategic diagnostics | completed as an initial research milestone; it revealed weaknesses of the game design rather than meaningful strategy, which motivated M4 | §N, §O; [self-play](rl/openai_five_selfplay_report.md), [diagnostics](rl/openai_five_diagnostics_report.md) |
| **M4** Integrated laning model and strategic validation | M4.1 design → M4.2 mechanics → M4.3 wave management and economics → M4.4 PPO self-play → M4.5 unified positional mechanics, calibration and retraining → M4.6 design review and diagnosis → M4 closeout | M4.1–M4.4c completed; M4.5a, M4.5b completed; M4.5c completed (6 × 8.192M and the 2× extension 6 × 16.384M trained; comprehensive evaluation, strategic battery and attackers done; GO WITH CAVEATS); M4.4d done within M4.5c (attackers); M4.6a–d completed; M4.6e and M4.6f (the targeted learning pilot) completed, reviewed; **M4 closeout completed and reviewed: M4 closed (ready for M5 with documented limitations); setup decided 2026-10-17: the baseline**; M4.7 early-lane horizon study completed (keep γ 0.99); M4.8 lane-exit evaluation completed (E_20 adopted) | below; §AS, §AT, §AU |
| Bayesian opponent modelling (OM1 feasibility, OM2 design and matched pilot) | beliefs about the opponent's style, updated during a game; designed first, then a controlled pilot with the same opponent distribution in both arms | **completed, reviewed** (committed by the user, a6e9b63f): OM1 outcome 1, feasible and worthwhile (§AV); OM2 implemented behind opt-in switches and its matched pilot run, pre-registered outcome 1, modelling pays and is safe (§AW). Carried into M5 as the modelling arm (approved direction, 2026-10-17) | §AV, §AW |
| **M5** Main training and evaluation campaign | the opponent-modelling package against continued pure self-play: two arms, six matched pairs continuing `ext/baseline:s`, 32.768M each; M5b training, M5c strategic replication and the M4 limitations, M5d opponent families, cross-seed/cross-arm results and attackers, M5e closeout; lane phase read as D0 → E_20 | **M5a and its corrections completed (§AX, §AY)**; **M5b training and the initial M5d evaluations completed (§AY)**: outcome 2, the package pays at a competence cost. **M5c completed (§AZ)**. **The core attackers and M5e completed (§BA); M5 committed by the user (a07e3fdf)**: S6 inconclusive; both arms exploitable; present both arms as a trade-off. The optional om_table ablation was not run | "M5 plan" below; [design](m5/m5_design.md), reports [M5b](m5/m5b_initial_report.md), [M5c](m5/m5c_strategic_report.md), [M5e](m5/m5e_closeout_report.md) |
| **M6** Presentation, interactive demonstration and reproducibility | a new web application (human versus either M5 arm, verified replays, the findings), README and research overview, reproduction bundles, the Streamlit viewer archived; demonstrated findings separated from hypotheses | approved direction (2026-10-19, extended 2026-10-21, M6c narrowed 2026-10-22); **M6a completed** (the session core and the approved interface design, 2026-10-21); **M6b completed** (the maintained application, 2026-10-22, acceptance pending); **M6c step 1 completed** (the guided-replay format and the first example, 2026-10-22, approved; further sequences on hold); **M6d started** (the research overview, 2026-10-22, approval pending) | "M6 plan" below; §BB, §BC, §BD, §BE; [plan](m6/m6_plan.md) |

### M4 in detail (repository names, in order)

| phase | what was done | outcome | record |
|---|---|---|---|
| **M4.1** Integrated design | mathematical specification of continuing time, wave dynamics with Freeze maintenance (replaced by natural waves in M4.2e), gold, XP and levels, items and combat power, recalls and purchases, persistent tower damage and plates; implementation plan; design audit | completed | §O; [design](m4/m4_1/m4_integrated_design.md), [plan](m4/m4_2/m4_implementation_plan.md), [audit](m4/m4_1/m4_design_audit.md) |
| **M4.2** Mechanics implementation and calibration | WP0–WP10 and validation S0–S4 (`GameConfig.m4()`); M4.2b calibration (ganks, items, the Shove → Recall cycle); M4.2c continuous investment (experiment); M4.2d wave readiness, positioning, combat-driven zoning, investment readiness and their integration (`m4_emergent()`); M4.2e-A natural waves (`m4_natural()`); M4.2e-B zoning on natural waves (`m4_lane()`); M4.2e-C wave actions as shares of one capacity (`m4_lasthit()`, the base preset of the mathematical reference until its rewrite of 2026-10-20); M4.2e-D visual debugger | completed | §P–§Y |
| **M4.3** PPO pilot, wave management and economic calibration | M4.3a RL infrastructure audit; M4.3b PPO self-play pilot on `m4_lasthit()` and its follow-up validation (Clear dominance is the game's incentive); M4.3c design (gold + XP objective, laning termination: PROPOSED, not adopted; the living mathematical reference); M4.3c screens (xp_range, wave control); M4.3c four-action redesign (Last Hit / Thin / Slow Push / Shove, `m4_four_actions()`); M4.3c stabilization (Shove terminology, four-action debugger, mixed-action sequences, economic calibration `m4_four_actions_econ()`, exploratory PPO pilot); M4.3d tower farming, pressure and gank risk (`m4_four_actions_econ_aggro()`) | completed | §Z–§AE |
| **M4.4** PPO self-play and emergent strategy evaluation | M4.4a infrastructure and diagnostics; M4.4b controlled training (3 economies × 3 seeds × 2M transitions, matched 2× extension, initial strategic evaluation); M4.4c deep investigation of strategic behaviour and economic incentives (multi-turn counterfactuals); M4.4d robustness, validation and follow-up research | M4.4a–M4.4c completed (M4.4c: CONDITIONAL GO, reviewed; its design questions became M4.5); M4.4d postponed, its content folded into M4.5c's evaluation | §AF, §AG, §AH; [M4.4a report](m4/m4_4/m4_4a_ppo_readiness.md), [M4.4b report](m4/m4_4/m4_4b_ppo_training_report.md), [M4.4c report](m4/m4_4/m4_4c_strategic_investigation.md) |
| **M4.5** Unified positional mechanics, calibration and retraining | M4.5a unified mechanics implementation (`m45_unified()`); M4.5b integrated calibration and strategic evaluation (`m45_calibrated()`); M4.5c PPO training and evaluation on the revised model | M4.5a completed (revision 2 committed); M4.5b completed (GO WITH CAVEATS, caveats accepted); M4.5c completed: pilot (GO); six seeds × 8,192,000 and the matched 2× extension × 16,384,000; comprehensive evaluation, battery and attackers (GO WITH CAVEATS) | §AI, §AJ, §AK, §AL, §AM; [M4.5c research report](m4/m4_5/m4_5c_research_report.md); [M4.5c pilot report](m4/m4_5/m4_5c_pilot_report.md); [M4.5c training report](m4/m4_5/m4_5c_training_report.md); [M4.5a spec](m4/m4_5/m4_5a_unified_mechanics_spec.md), [M4.5a report](m4/m4_5/m4_5a_unified_mechanics_report.md), [M4.5b report](m4/m4_5/m4_5b_calibration_report.md), [archived reference §14–§15](m4/laning_game_mathematical_reference_m4_lasthit.md#15-experimental-the-m45b-calibrated-candidate-m45_calibrated) |
| **M4.6** Design review and diagnosis | the design review; M4.6a zoning and freeze diagnosis; M4.6b investment audit and curve screen; M4.6c C1-versus-baseline PPO pilot; M4.6d favourable freezes, concession and the specialist gap; M4.6e observation, training exposure and value learnability; M4.6f separate-critic continuation pilot | M4.6a–d completed; C1 provisional, not adopted; no mechanics change justified (M4.6d); M4.6e completed, reviewed (the critic's strength blindness diagnosed); M4.6f completed, reviewed (a separate critic learns part of the strength value; no material behavioural improvement demonstrated within the tested budget; the limitation accepted) | §AN–§AR; [design review](m4/m4_6/m4_6_design_review.md), [M4.6a](m4/m4_6/m4_6a_report.md), [M4.6b](m4/m4_6/m4_6b_investment_report.md), [M4.6c](m4/m4_6/m4_6c_report.md), [M4.6d](m4/m4_6/m4_6d_report.md), [M4.6e](m4/m4_6/m4_6e_report.md), [M4.6f](m4/m4_6/m4_6f_report.md) |
| **M4 closeout** | explicit C1-versus-baseline decision; final mechanics, objective and training setup; documented freeze conditions and limitations | completed, reviewed: strategic acceptance table, configuration recommendations (baseline, shared critic; the setup decided by the user on 2026-10-17), corrections, M5 scope proposal; ready for M5 with documented limitations | §AS; [closeout report](m4/m4_closeout_report.md) |
| **M4.7** Early-lane horizon study | the lane phase (before about level 14) in current play; baseline and C1 there; delayed rewards in the published early conclusions; a matched two-seed pilot of a shorter discount (γ 1 − 1/60) against γ 0.99 | completed, reviewed: delayed rewards do not drive the early conclusions; the shorter objective changed a few decisions, gave no reliable lane-phase gain and cost later and fixed-opponent play; keep γ 0.99 | §AT; [report](m4/m4_7/m4_7_horizon_report.md), [pre-registration](m4/m4_7/m4_7_preregistration.md) |
| **M4.8** Lane-exit evaluation | an evaluation-only readout of who leaves the lane better placed: E_h = the earned-gold lead plus its expected change over h continuation turns (undiscounted, no critic tail), with XP, HP, absence and the wave beside it; validated on selected wave/Recall cases, controlled variants and recorded continuations; applied to the matched M4.6c baseline and C1 policies at turn 94 | completed, reviewed: the readout is useful (exact, calibrated against recorded continuations, informative in about two thirds of natural lane-exit states); default window E_20 with the non-gold dimensions beside it; baseline and C1 indistinguishable at the lane exit; M4.7's 50 → 94 conversion levels refined, its pilot conclusion reinforced; the user adopted E_20 and D0 → E_20 reporting on 2026-10-17 | §AU; [report](m4/m4_8/m4_8_lane_exit_report.md), [protocol](m4/m4_8/m4_8_protocol.md) |

### What is what

| category | items |
|---|---|
| **implemented and tested** | the exact game and its solver (legacy configurations bit-identical, pinned); `LaneEnv` and the PettingZoo adapter; the M4 ruleset and every preset up to `m4_four_actions_econ_aggro()`; masked PPO, GAE, shared-policy self-play, checkpoints with verified digests; the discounted evaluator with its guaranteed tail bound; counterfactual branching and action audits; the M4.3b (three-action) and M4.4 (four-action) diagnostic packages; the visual debugger; the M4.4b trained checkpoints (3 economies × 3 seeds at 2,048,000 and 4,096,000 transitions, git-ignored) and their supplementary analysis stages; the M4.4c multi-turn counterfactual package (`experiments/rl/m44c`: strategic plans, paired multi-turn engine, opponent styles, resumable runs); the M4.5a unified positional mechanics (`positional.py`; the Ward, minion-targeting, trade-budget and reach rules; the 90-entry observation; the viewer's W node; `experiments/m45a` demonstrations and smoke study); the M4.5b calibration package (`experiments/m45b`: paired engine and ledger, exact one-turn enumeration, contingent strategies, freeze / trade / Vision / zoning / matrix / timing studies, sensitivity, holdout validation, learned reference, recordings) and the opt-in farm-spot rule `"last_front"`; the opt-in separate-critic network `SeparateCriticNet` (M4.6f: exact initialization from a shared network, checkpoint-compatible); the lane-exit readout `diagnostics/lane_exit.py` (M4.8: evaluation only; exact accounting, checkpoint descriptors, repeated CRN continuations with a per-seat event ledger, window extraction, resolution times) and the M4.8 study package `experiments/rl/m48`; the OM1 feasibility package `experiments/rl/om1`; the opponent-modelling building blocks of OM2 (all opt-in: the observation field `obs_last_action`, the belief module `lanegame/opponent/`, the trainer's collector hooks `augment` / `observe_step`, `widen_inputs`) and the OM2 study package `experiments/rl/om2`; the M5 package `experiments/rl/m5` (M5a: both arms' training drivers, the history-dependent evaluation policy `OMPolicy`, the M5b–M5d instruments for both input formats, attackers, the pre-registered statistics, readiness checks R0–R10; `tests/test_m5.py`) and the opt-in `observe_turn` hook of the four shared rollout loops (stored M4.5c / M4.6c / M4.6f / OM2 records reproduce exactly); the M5a corrections (amendment 1: the opt-in read-only trainer hook `train_shared(on_update=...)`, `m5/health.py` runtime monitoring, coordinator and evaluation gate, `m5/strategic.py` (S8), the amended analysis, the verifiable amendment chain; `tests/test_m5_corrections.py`); the M5c–M5e stages (`battery_extend`; the core attacker stage `m5/attackers.py` under the runtime monitor and the coordinator, with `attack_check` gating `attack_eval`; the exploratory `traces` and `attack_behaviour` stages in `m5/traces.py`; the `reproduce` inventory in `m5/reproduce.py`; `tests/test_m5_attackers.py`) |
| **experimental candidates** (opt-in, not defaults) | C1 = `m45_calibrated(invest_gain=3.24, invest_scale=64.0)` (a keyword override; provisional, not adopted; M4.6c, §AS); `m45_unified()` (M4.5a revision 2, the M4.5b baseline) and `m45_unified_r1()`; `m4_four_actions()`, `m4_four_actions_econ()`, `m4_four_actions_econ_aggro()` (the M4.4 comparison); opt-in fields `xp_reward_weight`, `death_turns_by_level`, `terminal_rule = "final_plate"` (not recommended); the earlier experiments `m4_continuous()`, `m4_emergent()`; the training option `SeparateCriticNet` (M4.6f: better critic values in strength states, no material behavioural improvement demonstrated within the tested budget; recommended at closeout as an opt-in, not an M5 factor); the objective override `m45_calibrated(gamma=1 - 1/60)` (M4.7's pilot arm: a keyword override of the existing `gamma` field, not a preset; not recommended, §AT) |
| **adopted research baselines** | objective: the discounted gold lead, γ = 0.99, continuing time, no terminal state (unchanged since M4.2); the living [mathematical reference](m4/laning_game_mathematical_reference.md) describes `m45_calibrated()` directly (rewritten 2026-10-20; the earlier presets are its §14 and the opt-in fields its §15; the earlier version, whose §1–§10 described `m4_lasthit()`, is archived unchanged as [laning_game_mathematical_reference_m4_lasthit.md](m4/laning_game_mathematical_reference_m4_lasthit.md)); **research preset in use since M4.5c: `m45_calibrated()`** (M4.5b; chosen by the user on 2026-10-17 as the M5 configuration; a preset, not the `GameConfig()` default), with the M4.5c PPO settings (entropy 0.03, 100% current-policy shared self-play, 16 × 128-turn segments, 512-turn bootstrapped games, lr 3·10⁻⁴ annealed); evaluation: paired seats, E1 from s0, W = 1000 headline; strategic counterfactuals by rollouts, with no-bootstrap returns in strength-gap states (M4.6d–e); lane-phase outcomes: the raw earned-gold lead at a lane-exit checkpoint read together with the lane-exit readout E_h and the non-gold dimensions beside it (M4.8, E_20 adopted by the user on 2026-10-17; evaluation only, the objective is unchanged). Earlier baselines, for the record: the M4.3b `ent03` condition on `m4_lasthit()`, and the M4.4 candidate `m4_four_actions_econ_aggro()` |
| **underway** | nothing. **M5 is complete, committed by the user in a07e3fdf** (§AX to §BA): outcome 2, the package pays at a competence cost; four of five strategic findings retained in both arms; both arms exploitable, S6 inconclusive. M6a is complete (§BC): the session core is built and verified, and the user approved the interface design. M6b is complete (§BD): the maintained application is built and verified in the browser, and the user's acceptance is pending. M6c (guided replays within Watch, the scope narrowed on 2026-10-22) has its format and first example built (§BE); the user's review is pending. Mechanics changes and further investment or discount experiments stay paused by the user's decisions; there is no lane termination (2026-10-17) |
| **conditional** | none (the one targeted learning pilot ran as M4.6f, §AR) |
| **approved direction** (2026-10-17, 2026-10-19) | the Bayesian opponent-modelling design and matched pilot (done: OM1, OM2); **M5 with the opponent-modelling package compared with continued pure self-play** (decided in the M5a brief; the baseline `m45_calibrated()`, the continuing γ 0.99 objective, the shared critic, D0 → E_20 reporting and rollout evidence fixed); then M6. **2026-10-19:** M6 builds a new web application, later deployable in public, with a human playing against either M5 arm at its centre; the Streamlit viewer is archived as legacy during implementation, not reworked; policy-versus-policy observation and guided replay are supporting modes to consider |
| **proposed** | the M6 plan's details, in review ([plan](m6/m6_plan.md)): the five milestones M6a to M6e, the initial release, the architecture (React with TypeScript and Vite, FastAPI, one container), the session and record design, the three bundles and the safeguards; M6c's steps 2 and 3 (the review of the first guided replay, then a few more examples in the accepted format). Optional future research, none required ([M5e report §14](m5/m5e_closeout_report.md)): isolating the cause of the competence cost (a matched arm with the fixed opponents but no belief, or the belief with equal self-play exposure), the om_table adaptation ablation, larger attacker budgets, and the zoning and favourable-freeze gaps both arms share |
| **unresolved questions** | Answered by M4.4c (§AH): the interrupted Recall is a rational guessing game (Recall is the best default; a small ≈ 0.1-resource conditional gap); freezes pay only against opponents that concede them; the stronger side does zone, through kills and forced Recalls; Vision-first does not pay in typical (short) sieges; Last Hit / Thin / Slow Push coincide in 68–77% of turns. Open: How exploitable are four-action policies (no four-action attacker yet)? *Answered later* by M4.5c's six attackers (+1.0, §AM) and M5d's twelve (§BA). How much value do they leave against conceding / passive opponents (+0.4 to +1.0 measured in probes)? Does Vision-first pay in long sieges (positive trend, unresolved)? **Design decisions after review**: Vision, combat-wave coupling, minion aggro and the XP range were revised together in M4.5a (§AI), to be calibrated in M4.5b; still open: whether to keep three non-Shove wave actions that coincide in most states, whether a champion may step into its own tower cell by choice. **Answered by M4.5b** (§AJ): Ward value is situational and contestable; early trades into big waves cost HP with retaliation from the exchange tick; freezes held with Thin + Trade pay conditionally (mid- and late game; punishing walk-ups when stronger); Safe costs 0.15–0.56 farm per turn and threat-only zoning exists near the weaker side's tower; scripted restricted games are Shove-centred, which learned policies beat. **Open after M4.5b**: accept the 21% mid-to-late minion share or adopt `minion_damage_growth = 1.12` after fresh validation; whether PPO on the candidate learns freezes, wards and selective trading (M4.5c). *Both answered later.* The review accepted the share (`minion_damage_growth` stays 1.08), and M4.5c's policies learned conditional freezing and selective Trading and rationally avoided Wards (§AM). Structural limits documented earlier remain: contact geometry at the tower, ganks independent of the opponent, a weak XP economy on a five-cell lane. The M4.3c objective / termination proposal remains undecided (*superseded 2026-10-17*: the continuing objective is kept and no termination is added; see the decisions in the summary). **Carried forward by the M4 closeout (§AS)**, as M5 measurements rather than M4 work: guarding in the narrow favourable-freeze window, large-gap conversion, the weak side's Safe at Recall windows, value left against conceding styles, exploitability at larger attacker budgets, the isolated effect of archive starts, and whether Last Hit / Thin / Slow Push still coincide on `m45_calibrated()`. Accepted as documented limitations: the narrow freeze window, late leads that fade, the shared critic's strength blindness. **Answered by M4.7 (§AT)**: the γ 0.99 objective does not make the early-lane conclusions depend on post-lane rewards, and a shorter objective (H 60) does not improve lane-phase play in a matched continuation; not tested: a from-scratch short-horizon campaign, H 80, a short-horizon C1. **Answered by M4.8 (§AU)**: raw earned gold at the lane exit misstates the economic position materially in about two thirds of natural turn-94 states, because of pending resets, HP and the wave; a 20-turn repeated-continuation readout resolves 95% of the pending items, is calibrated against recorded continuations and is robust to in-distribution continuation policies; at the lane exit baseline and C1 do not differ. Open: the readout's dependence on unfamiliar continuation styles; how to value XP, HP and position (deliberately not converted). **Answered by OM2 (§AW)**: a Bayesian belief over opponent types, trained with library opponents, mixtures and switchers, adds +2.0 at the lane exit against recognizable styles over an equally trained agent without it, and protects competence against competent opponents. Open: exploitability by a dedicated or deceptive attacker (the belief follows switches only in 10–60 turns), the effect of the last-action field alone, and what a fixed-opponent share costs pure self-play competence. **Open for M5 (§AX)**: the user's decisions on seed count, attacker budget, the adaptation test and the battery size (decided since as six pairs, quarter-budget core attackers, an optional om_table ablation that was not run, and the full battery); whether six pairs resolve a near-zero competence effect (most likely inconclusive under OM2's variance); exploitability of the modelling agent by a dedicated or deceptive learned attacker. **Answered by M5 (§AY to §BA)**: the competence effect was not near zero, and six pairs resolved it as a material loss; dedicated quarter-budget attackers exploit both arms target-specifically. **Open after M5**: whether the package is more exploitable than pure self-play (S6 inconclusive); what causes the competence cost (belief, opponent mix or self-play exposure); exploitation by a deceptive, stateful attacker (not tested). **Open for M6 (§BB, §BC)**: the plan's three decisions were settled on 2026-10-21 (Watch joins the initial release, `docs/` is tracked, the frontend is `web/`); open are the visual direction and whether scripted styles are offered as opponents; later, a license before any bundle is published, and at the M6e checkpoint the hosting provider, the budget and deployment itself. Not measured by M5 and not claimed by M6: how either policy plays against humans. |

### M4.4 plan

**M4.4a — PPO infrastructure and diagnostics** (completed; [report](m4/m4_4/m4_4a_ppo_readiness.md))

| | |
|---|---|
| objective | make the PPO pipeline and the evaluation instruments correct for the four-action game, and measure what training will cost, before spending the training budget |
| engineering tasks | verify shared self-play, masking, rewards and time semantics on the three environments; a checkpoint compatibility guard; four-action decoding; state-based strategic metrics (wave management, freezes, zoning, vision / dives, plate phases); component interventions and paired counterfactuals; smoke training and evaluation; checkpoint → viewer traces; the M4.4b specification; documentation |
| deliverables | `lanegame/experiments/rl/m44/`, `tests/test_m44a_readiness.py`, `results/m44a/`, the readiness report, [the PPO pipeline explained](rl/ppo_training_explained.md), this roadmap update |
| completion criteria | every smoke pipeline check passes; re-training is bit-identical; the regression suite passes; the M4.4b specification has a measured runtime; the review approves M4.4b |
| dependencies | M4.3a (audited trainer), M4.3b (diagnostic patterns), M4.3c (`m4_four_actions()`, `m4_four_actions_econ()`), M4.3d (`m4_four_actions_econ_aggro()`) |

**M4.4b — Controlled PPO self-play training** (completed, reviewed, committed by the user in f4bcb7da; [report](m4/m4_4/m4_4b_ppo_training_report.md), [specification](../results/m44a/m44b_plan.md), [tables](../results/m44b/report.md))

| | |
|---|---|
| objective | train shared-policy self-play on the three economies with identical methodology and independent seeds, so that differences between environments can be attributed to the economy |
| engineering tasks | 9 runs (3 environments × seeds 0, 1, 2 × 2,048,000 transitions; paired initialization across environments); evaluation: references at W = 1000, cross-play within each environment, history (own snapshots, initial network), learning curves, behaviour diagnostics, counterfactuals, viewer traces |
| deliverables | `results/m44b/` (training summary with health checks, evaluation JSON), git-ignored checkpoints, an M4.4b report |
| completion criteria | all runs pass the health checks; learning curves read for a plateau against fixed references and own snapshots (else the budget is extended equally for every environment); seed-to-seed spread within each environment reported before any between-environment comparison |
| dependencies | M4.4a approved |
| outcome | all criteria met: 9 / 9 runs at the exact budget, 81 / 81 checks, bit-identical full-budget re-runs; strength plateaued against fixed references with no cycling, but behaviour still drifted, so a matched 2× extension was run for all environments (4,096,000 transitions; reported separately); recommendation CONDITIONAL GO for M4.4c (§AG) |

**M4.4c — Deep investigation of strategic behaviour and economic incentives** (completed, reviewed, committed by the user in d9f92df5; [report](m4/m4_4/m4_4c_strategic_investigation.md), [tables](../results/m44c/report.md))

| | |
|---|---|
| objective | decide, for each strategic gap M4.4b found (interrupted Recall, weak freezes, no zoning, rare Vision before Pressure, indistinct Last Hit / Thin / Slow Push), whether it is a PPO learning failure, an evaluation weakness or a consequence of the simulator's incentives |
| engineering tasks | multi-turn strategic plans over existing legal actions (preconditions, stop conditions, fallbacks); paired multi-turn counterfactuals with episode ledgers and several horizons; opponent protocols and styles; natural and initialized states; the 4.10M finals as primary and the 2.05M finals as replication; closed-form mechanics; viewer traces of counterfactual pairs |
| deliverables | `lanegame/experiments/rl/m44c/`, `tests/test_m44c_strategic.py`, `results/m44c/` (report, summary, figures, traces, manifest; raw per-state files git-ignored), the M4.4c report |
| completion criteria | every question answered with paired, multi-turn evidence and uncertainty across states and across seeds; no claim from action frequencies alone; no rule change |
| dependencies | M4.4b |
| outcome | all five investigations completed (5,982 states, ≈ 5.3 h); no profitable strategy found that PPO misses against its own opponent; the gaps are simulator incentives (C) and self-play's opponent distribution (D); zoning verdict of M4.4b withdrawn; CONDITIONAL GO for M4.4d (§AH) |

**M4.4d — Robustness, validation and follow-up research** (postponed: folded into M4.5c's evaluation, so that robustness
and exploitability are measured on the revised simulator, not on one already being revised; the M4.4 checkpoints remain
available for historical comparisons)

| | |
|---|---|
| objective | assess how robust and exploitable the learned four-action policies are, and settle the follow-up questions M4.4c left open |
| engineering tasks | four-action approximate best-response attackers (warm-started, as in M4.3b) as exploitability lower bounds per economy; evaluation against conceding, passive and aggressive opponent styles and the scripted references (where M4.4c measured +0.4 to +1.0 of unexploited value); the long-siege Vision case with targeted initialized states and more rollouts; deliberate cross-environment evaluations; a convergence criterion only if M6 needs a quantitative profile |
| deliverables | a robustness and exploitability report; attacker checkpoints |
| completion criteria | exploitability lower bounds per economy with intervals; every statement qualified as "under the given continuation policies" |
| dependencies | M4.4c; review of the four design questions recorded in §AH (they decide which economy M4.4d evaluates) |

### M4.5 plan

**M4.5a — Unified mechanics implementation** (completed, reviewed, revision 2 committed by the user in 8dc5acf5; [spec](m4/m4_5/m4_5a_unified_mechanics_spec.md),
[report](m4/m4_5/m4_5a_unified_mechanics_report.md), [archived reference §14](m4/laning_game_mathematical_reference_m4_lasthit.md#14-experimental-the-m45a-unified-positional-candidate-m45_unified), [smoke tables](../results/m45a/report.md))

| | |
|---|---|
| objective | implement the complete positional Vision, minion aggro and retaliation, XP and combat / wave interaction redesign in one opt-in candidate preset, and establish its mathematical correctness and reproducibility |
| engineering tasks | the movement graph and the shared Vision node W; a contestable Ward action; per-tick exclusive minion targeting (retaliation after positive trade damage, positional aggro); level-scaled minion damage on champions; capped wave budgets for connected traders; the reach ball; XP range 2; the 90-entry observation; viewer support; tests; demonstrations; documentation |
| deliverables | `GameConfig.m45_unified()`, `lanegame/positional.py`, engine changes behind opt-in fields, `tests/test_m45a_mechanics.py`, `lanegame/experiments/m45a/`, `results/m45a/`, the spec, the report, reference §14 |
| completion criteria | every earlier preset bit-identical; deterministic replay; mirror equivariance; zero-sum rewards; masks, observation decoding and PettingZoo on the new preset; the checkpoint guard refuses old networks; regression suite green |
| dependencies | M4.4c and its review |
| outcome | all criteria met; GO WITH CAVEATS for M4.5b (§AI); corrected after review (revision 2: wave-following movement, positional aggro on actual positions; §AI, report §10) |

**M4.5b — Integrated calibration and strategic evaluation** (completed, reviewed, committed by the user in 45b7dfa7; [report](m4/m4_5/m4_5b_calibration_report.md),
[archived reference §15](m4/laning_game_mathematical_reference_m4_lasthit.md#15-experimental-the-m45b-calibrated-candidate-m45_calibrated), [tables](../results/m45b/report.md))

| | |
|---|---|
| objective | a substantial integrated experimental investigation of the completed simulator: calibrate Vision, minion damage, combat-wave coupling, Safe economics and freeze interactions together, studying individual subsystems only where the integrated results identify a concrete issue |
| engineering tasks | trade EV into waves by level and wave size; Ward value and contestability (paired multi-turn counterfactuals, the M4.4c engine); freezes with Thin + Trade (Scenarios A and B); Safe / Yield and threat-only zoning; restricted-game equilibria of the four-action strategies plus Ward strategies; a positional-aggro audit; joint sensitivity analysis of the provisional parameters (M4.5a report §9.2) |
| deliverables | a calibrated candidate preset (or a documented decision to keep `m45_unified()`), an M4.5b report, the reference updated |
| completion criteria | each provisional parameter either justified by integrated evidence or flagged; no single strategy dominating every state; scripted opponents, controlled scenarios, multi-turn counterfactuals and sensitivity analyses with paired seeds and intervals |
| dependencies | M4.5a reviewed |
| outcome | `m45_calibrated()` = revision 2 + the farm-spot correction `"last_front"` + retaliation delay 0; every other number retained; GO WITH CAVEATS for M4.5c; one holdout criterion failed (mid-to-late minion share; the review accepted it, so `minion_damage_growth` stays 1.08); §AJ |

**M4.5c — PPO training and evaluation on the revised model** (completed: [research report](m4/m4_5/m4_5c_research_report.md), [tables](../results/m45c/final/report.md); [pilot report](m4/m4_5/m4_5c_pilot_report.md); [8.192M training report](m4/m4_5/m4_5c_training_report.md))

| | |
|---|---|
| objective | train fresh shared-policy PPO agents (the OpenAI Five-inspired self-play of M3–M4.4; no new RL frameworks or algorithms) on the calibrated environment and evaluate whether freezing, zoning, combat, Vision timing, retreats and Recall / reset behaviour emerge |
| engineering tasks | the M4.4b protocol (seeds, budgets, health checks, learning curves) on the new 90-input / 61-action network; the M4.4 / M4.4c diagnostics and counterfactuals adapted to W and minion targeting; then M4.4d's robustness work on the new policies: best-response attackers (exploitability lower bounds), conceding / passive / aggressive opponent styles, cross-environment comparisons with the M4.4 finals |
| deliverables | checkpoints (git-ignored), an M4.5c report, viewer recordings |
| completion criteria | as M4.4b, plus the exploitability and robustness measures of the former M4.4d |
| dependencies | M4.5b reviewed (done; minion share accepted at 1.08); the M4.4 diagnostics adapted to W and minion targeting, with the M4.5b ledger quantities in the health checks (done in the pilot: `experiments/rl/m45c`); the M4.5b scripted strategies as fixed evaluation opponents (done) |
| progress | pilot complete (§AK): pipeline verified on the preset, one seed × 1,024,000 transitions, learning curve, behaviour, throughput; runtime audit and horizon study; **6 seeds × 8,192,000 trained** (§AL: 55 min concurrent, all checks pass, seed-0 re-run, limited health evaluation, framework prepared and gated); **completed (§AM)**: matched 2× extension 6 × 16,384,000; budget comparison; comprehensive standard evaluation; 50-turn strategic battery with validation and exception 3; six best-response attackers; GO WITH CAVEATS |

### M5 plan

The final campaign ([design](m5/m5_design.md), [pre-registration](m5/m5_preregistration.md),
[M5a readiness report](m5/m5a_readiness_report.md), §AX).

**Its three questions:**
1. Which strategic findings replicate on the final policies?
2. What does the opponent-modelling package add beyond continued pure self-play?
3. Does it preserve competence, and how exploitable are the agents?

**The arms.**
- **A, opponent modelling (`om`):** OM2's treatment, with the previous action, the 13-feature belief (λ 0.97) and a
  fixed library, 50% self-play and 50% the 15 fixed opponents.
- **B, pure self-play (`sp`):** 100% current-policy shared self-play, as M4.6c.
- **Both:** start from `ext/baseline:s` (24.576M prior transitions, the same outputs); s = 0…5; 32.768M additional
  transitions each, with a matched 16.384M checkpoint; the M4.6c start mixture and PPO settings; PPO seed 70,000 + s.
- **What the contrast estimates:** the package effect. Information and opponent distribution change together, the
  archive-start effect is not isolated, and the arms' trajectories diverge after the first update.

| configuration item | status |
|---|---|
| opponent modelling in M5, compared as a package with pure self-play | **approved direction** (2026-10-17) |
| `m45_calibrated()`, gold-difference reward, γ 0.99, continuing time, no termination at level 14, the shared critic, D0 → E_20 with the panel, rollout evidence | **approved (fixed by the brief)** |
| two arms, six pairs, continuation of `ext/baseline:s`, OM2's settings, a fixed library | approved as the starting proposal; kept by M5a |
| 32.768M per run, a 16.384M matched checkpoint; six pairs; six workers, om runs first | **approved and run** (the user, 2026-10-17) |
| the correction pass: headline outcome 6, the 15-hypothesis Holm family, S5a / S5b, completeness, S8, runtime health and the evaluation gate, the amendment chain | **completed** (amendment 1, before training) |
| core attackers: 12 at a quarter budget (8.192M) | **approved 2026-10-19 and run** (§BA) |
| the om_table adaptation ablation (6 attackers) | **optional follow-up, not run** (exploratory; a separate decision) |
| the full battery and the limitation cells | **approved 2026-10-18 and run** (the full design, §AZ) |
| the non-inferiority margin δ = 0.25 and the classifications | frozen in the pre-registration, amended once (version 1); any change is an amendment |

**M5a — design, pre-registration and readiness** (completed, §AX)

| | |
|---|---|
| objective | a concrete, reviewable campaign: the design, the frozen pre-registration, drivers for both arms and every instrument, and bounded readiness checks |
| deliverables | `docs/m5/` (design, pre-registration, readiness protocol, readiness report); `lanegame/experiments/rl/m5/`; `tests/test_m5.py`; `results/m5/` (`readiness.json`, `budget.json`, `preregistration.json`, `report.md`, benchmark records) |
| completion criteria | readiness R0–R10 passed and the regression suite green; the pre-registration frozen with its SHA-256; the budget measured |
| outcome | all met: 11 of 11 readiness parts; the pre-registration frozen (the user authorized the campaign on 2026-10-17, and the document was amended once before training, version 1); the budget about 34–36 h |

**M5a corrections — amendment 1** (completed 2026-10-17, §AY)

| | |
|---|---|
| objective | fix the review's findings in the analysis, the training monitoring, the gates and the amendment bookkeeping before training, without changing the experiment |
| outcome | all eight findings fixed and tested; the pre-registration amended (version 1); A1–A4 passed (monitored smokes bit-identical; full suite 1,534 passed, 4 skipped, 2 xfailed) |

**M5b — training** (completed 2026-10-18, §AY; gate `--confirm-m5-train`)

| | |
|---|---|
| objective | the 12 runs, healthy and reproducible, with their learning curves and convergence probe |
| tasks | training (6 workers, om runs first); the pre-registered health checks; restarts only for technical failures, with an exact-prefix re-run; monitoring: 16-point curves against a fixed panel, history against 25 / 50 / 75% snapshots and the start, the final-vs-75% convergence probe |
| deliverables | finals and snapshots (git-ignored); `results/m5/train_main.json`; run manifests; the monitoring records |
| completion criteria | every health check passed; no extension, since the 32.768M results are final |
| dependencies | the user's decisions and approval (given 2026-10-17) |
| estimated cost | 9.4–11.8 h of training; 0.5 h of monitoring |
| outcome | 12/12 runs, 138/138 checks, no alarm; 10.90 h of training, 0.56 h of monitoring; sp still improving at the end, om not shown to be; no cycling |

**M5c — strategic evaluation** (completed 2026-10-19, §AZ; gate `--confirm-m5-eval`)

| | |
|---|---|
| objective | which M4.5c findings replicate, and every M4 limitation re-measured, per arm and between arms |
| tasks | the full M4.5c battery with the M4.6b corrections on all 12 finals; the M4 limitation families: weaker side, lead premium, strength use, freeze and breaker cells, Recall windows, natural states, large gaps with a gap specialist; D0 → E_20 and the panel from every game |
| criteria | a verdict replicates when, in at least 5 of 6 seeds, no fixed alternative beats the policy by more than 0.2; every closeout class labelled used / missed / weak / open |
| dependencies | M5b's finals |
| estimated cost | 11.8 h at the measured rates (battery 10.2 h, cells 1.6 h), plus a contingency for 150-turn re-runs after a failed validation |
| outcome | measured 16.4 h (battery 8.73 h, cells 2.11 h, 150-turn re-runs of 51 failed pairs 5.58 h). Four of five verdicts replicate in both arms; concession under threat does not (om 3/6, sp 4/6); the om finals are worse than the sp finals in every limitation family ([report](m5/m5c_strategic_report.md)) |

**M5d — opponent modelling and robustness** (families and head to head completed 2026-10-18, §AY; core attackers completed 2026-10-19, §BA; gates `--confirm-m5-eval`, `--confirm-m5-attackers`)

| | |
|---|---|
| objective | the package's exploitation gain, its competence and its exploitability |
| tasks | **families from s0:** library and held-out styles (including conditional styles on a competent base), misspecified mixtures and switchers, imitate-then-switch, competent historical and final opponents, the head to head (same seed and cross-seed), the 500-turn objective, the midpoints. **Attackers:** dedicated attackers against both arms and against the om final without in-game adaptation, with parity starts, the om input format and a matched budget, evaluated on independent games; the belief mechanism |
| criteria | P1, exploitation gain (≥ +0.25, interval above 0); P2a and P2b, competence by non-inferiority with δ 0.25 (inconclusive is never reported as preserved); the headline outcomes 1–6 (amendment 1); the secondary endpoints in one fixed Holm family |
| dependencies | M5b's finals |
| estimated cost | families 1.7 h (measured 1.87 h); core attackers 6.6 h + 0.5 h (quarter) or 13.3 h + 0.5 h (half); the optional om_table ablation +4.1 h |
| outcome | **outcome 2, pays at a competence cost**: P1 +9.26 ± 0.30 demonstrated; P2a −0.63 ± 0.35 and P2b −0.44 ± 0.28 material losses; S1 preserved; S2, S3, S4, S5b material losses; S5a demonstrated ([report](m5/m5b_initial_report.md)). Core attackers (5.96 h + 0.52 h): lower bounds om +2.96, sp +1.15 at the lane exit, both positive in every seed; S6 om − sp +1.81 [−0.56, +4.19] inconclusive; no transfer ([M5e report](m5/m5e_closeout_report.md)) |

**M5e — closeout** (completed 2026-10-19, committed by the user, a07e3fdf; §BA)

| | |
|---|---|
| objective | strategic replication; learning progress and convergence limits; exploitation gains with competence and robustness trade-offs, side by side; exploitability lower bounds; the recommended final configuration for M6; reproducibility materials and representative viewer traces |
| dependencies | M5b–M5d |
| outcome | [final report](m5/m5e_closeout_report.md): the Holm family final; M5 complete; present both arms as a trade-off in M6; `results/m5/reproducibility.md` and 15 viewer recordings; corrections of three M5c interpretations in the living summaries |

### M6 plan

The full plan is [m6_plan.md](m6/m6_plan.md) (§BB, revised 2026-10-21 and 2026-10-22). M6a (§BC) and M6b (§BD)
are complete. M6c was narrowed on 2026-10-22 to guided replays within Watch, and its first step is complete (§BE).
M6d started on 2026-10-22 with the demo corrections and the research overview (§BF); M6e is **planned, not
started**. M6 adds no mechanics, rewards, objectives or training.

| item | status |
|---|---|
| a new web application with human versus trained policy at its centre, both M5 arms as opponents, later deployable in public | **approved direction** (the user, 2026-10-19) |
| Play and Watch (a bounded bot-versus-bot match, generated first and played back) as the two primary modes of the initial release | **approved direction** (the user, 2026-10-21) |
| a Chess.com-style turn history shared by Play and Watch; the frontend clarity requirements | **approved requirements** (the user, 2026-10-21; plan §5.1, §5.2) |
| the Streamlit viewer archived as legacy during implementation, not reworked | **approved direction**; planned for M6d, not done |
| sequences (first called guided replays) of verified research games within Watch, as M6c's whole content | **approved direction** (the user, 2026-10-22); the format and the first sequence are **completed** and approved (§BE); further sequences are on hold at the user's request |
| the research overview | **completed** (2026-10-22, §BF, [overview](m6/research_overview.md)); its numbers are checked against the M5 records; the user's approval of the text is pending |
| a separate findings page and research explanation inside the application | **dropped** (the user, 2026-10-22); the research overview is M6d documentation, linked from the application |
| React with TypeScript and Vite in `web/`, FastAPI over the unchanged Python simulator and policies, one container | **approved** (the user, 2026-10-21), unless a concrete blocker appears; M6a found none |
| the headless session core and the backend's correctness rules | **completed** (M6a, §BC) |
| the interface design | **approved** (the user, 2026-10-21, after six review rounds; decision record §7; the reference prototype is archived at `legacy/m6a_glass_prototype/`) |
| the playable application: the FastAPI layer, `web/`, the lifecycle and limits, the browser verification | **completed** (M6b, 2026-10-22, §BD); the user's acceptance of the build is pending |
| scripted styles as opponents | **unresolved** (the M6a decision record §4); M6b offers the two trained agents only |
| the bundles and the safeguards | proposed |
| a hosting provider, spending and public deployment | not decided; the M6e approval checkpoint |

| milestone | objective | effort | user decision |
|---|---|---|---|
| **M6a** product definition and technical readiness (**completed** 2026-10-21) | the headless session core with the correctness rules proved on the real seed-0 finals; journeys and the API, recording and frame contracts; two or three clickable prototypes of the lane view, the turn history and Play and Watch; after the user's choice, refinement (six review rounds in the event) | M (the session core alone S) | the prototype direction (given) |
| **M6b** the playable application (**completed** 2026-10-22, §BD) | FastAPI and the new frontend; a visitor plays a full game against either opponent, with legal actions explained, both actions revealed together, the belief hidden or shown, history navigation and a record that replays bit for bit; Watch generates a bounded bot match and plays it back | L (the frontend is the repository's first production JavaScript) | acceptance of the playable build (pending) |
| **M6c** sequences within Watch (first called guided replays; narrowed 2026-10-22; **step 1 completed**, §BE) | a lightweight format over the verified research replays (a JSON file per guide, validated against the replay at startup), a compact note panel in Watch with Previous and Next, and a small collection of examples: the format and the first example (step 1), the user's review (step 2), a few more examples (step 3) | S per step | acceptance of the format and the first example, then of each further example |
| **M6d** documentation, the research overview and reproducibility (**started** 2026-10-22, §BF: the research overview written) | a concise README, the research overview (the question, the method, the six-seed results, the comparisons and the limitations; moved here from M6c), the reproduction guide, three bundles with SHA-256 manifests, `reproduce` re-run on the committed tree, the Streamlit UI moved to `legacy/` after a tag | M | the research overview's text; a license; where the evidence and training bundles are published |
| **M6e** release readiness | clean-install checks, a load measurement, proportionate safeguards, a container image, deployment notes, the approval checkpoint | M | provider, budget, go or no-go |

The first implementation task, the headless session core of M6a (the plan's §11), is done (§BC).

### Path to completion (agreed sequence, 2026-10-16; extended 2026-10-17)

The user's decisions of 2026-10-16:
- pause mechanics changes and further investment experiments;
- diagnose encoding, actual training exposure and value learnability before any training change (done: M4.6e, §AQ);
- allow at most one targeted learning pilot, and only if justified;
- accept documented limitations where further work is unlikely to improve the project materially.

Status labels follow the legend above.

**1. Focused value diagnostic (M4.6e)** (completed, reviewed; [report](m4/m4_6/m4_6e_report.md),
[pre-registration](m4/m4_6/m4_6e_preregistration.md), [tables](../results/m46e/report.md))
- Observation and measurement verification; the actual training exposure of the problematic state families; a
  supervised value-representation study (the existing critic, a readout of its frozen trunk, a fresh value network,
  explicit strength features).
- No mechanics change and no PPO campaign.
- Outcome:
  - the observation is exact and carries strength;
  - the trained trunk discards it;
  - a separate value network learns it from the unchanged observation;
  - large gaps are essentially absent from training;
  - better values do not improve the flagged decisions.
- Recommendation: accept the limitation and proceed to M4 closeout.

**2. One targeted learning pilot (M4.6f)** (completed, reviewed; authorized by the user on 2026-10-16;
[report](m4/m4_6/m4_6f_report.md), [pre-registration](m4/m4_6/m4_6f_preregistration.md),
[tables](../results/m46f/report.md))
- **Design.** M4.6e's named candidate, a separate critic trunk:
  - the same actor and the same PPO, against an unchanged matched control;
  - six seed pairs, 4.096M transitions, then the pre-registered extension to 8.192M;
  - value learning against each network's own returns; behaviour by rollouts (head to head, strength conversion,
    freezes, weak-side Recall windows, natural states) and ordinary-play guards.
- **Outcome.**
  - The separate critic learns part of the strength value: plausible-edge recovery 0.45 vs 0.21, large gaps 0.32 vs
    0.03, 6/6 seeds, with no natural-state cost.
  - Neither continuation of the shared network (4.096M or 8.192M: separate runs from the same start) changed the
    shared critic's strength recovery.
  - No material change in the targeted behaviour was demonstrated within the tested budget (resolution about ±0.2).
    Head to head there is no material difference.
  - Pre-registered outcome 4 (V1 narrowly short). Substantively, better values without a demonstrated behavioural
    benefit.
- **Recommendation.** Accept the limitation and close the investigation; decide at M4 closeout whether M5 uses the
  separate critic (an opt-in option, not a fix).

**3. M4 closeout** (completed, reviewed; [report](m4/m4_closeout_report.md), §AS)
- **Outcome.** Ready for M5 with documented limitations; closing M4 is recommended. The user decided the setup on
  2026-10-17 (the baseline, C1 an opt-in, the shared critic assumed), and M5 continued `ext/baseline:s` instead of
  starting fresh. The recommendation was:
  - the baseline `m45_calibrated()` over C1, which stays a provisional opt-in;
  - the shared network, with the separate critic an opt-in that is not an M5 factor;
  - fresh initialization, M4.6c archive starts, 6 × 32.768M.
- The original scope of this step, kept for the record (two parentheticals corrected at closeout):
  - An explicit C1-versus-baseline decision: C1 is provisional and has not been adopted.
  - Finalize the mechanics, the objective and the training setup for M5, including whether M5 uses the separate
    critic (M4.6f: better values, no measured regression, 4% slower, no demonstrated behavioural benefit).
  - Document the narrow conditions where freezing pays: an early, fresh freeze with an HP edge (M4.6d). Document the
    remaining limitations:
    - the shared critic's strength blindness outside natural states (about a third of it removed by a separate
      critic, M4.6f);
    - the actor's conversion of strength advantages, which a substantially better critic did not measurably change
      within the tested budget (M4.6f);
    - the large-gap conversion gap;
    - the freezer's weak threat on a walk-up after the early levels;
    - an abstraction that is not a perfect reproduction of League.
  - Reopen mechanics only for a concrete blocking issue.

**3a. Early-lane horizon study (M4.7)** (completed, reviewed; [report](m4/m4_7/m4_7_horizon_report.md),
[pre-registration](m4/m4_7/m4_7_preregistration.md), [tables](../results/m47/report.md))
- **Requested by the user before the final M5 campaign.** Does the long continuing objective place too much emphasis
  beyond the laning interaction (before about level 14)?
- **Stage 1** (existing records plus new paired 300-turn rollouts):
  - the lane phase is turns 0–93 (level 14 at turn 94) and 17% of training;
  - baseline and C1 coincide there;
  - the early benefits arrive mostly before level 14 and keep their order under shorter horizons.
- **Stage 2** (user-approved, pre-registered; 2 seeds × 4.096M; γ 0.99 against γ 1 − 1/60): outcome 5. There was no
  reliable lane-phase gain; play later in the game and against fixed opponents suffered.
- **Recommendation:** keep γ 0.99; report lane-phase readouts in elapsed-turn windows. Training occupancy (83% after
  level 14) is the lever a discount cannot move: future work, not done.

**The user's decisions of 2026-10-17** (approved direction; recorded in the summary above):
- Keep the continuing objective (earned-gold reward, γ 0.99, no terminal state). There is no termination around
  level 14; the ending experiment proposed after M4.7 and further discount experiments are not the immediate plan.
- Treat the lane exit as an evaluation checkpoint with short repeated continuations. Keep XP, HP and positional
  advantages visible separately.
- Validate that method and apply it to the matched baseline/C1 policies before configuration recommendations
  (steps 1–2 below).
- Design Bayesian opponent modelling next and pilot it before the final M5 campaign (steps 3–4).
- The critic investigation stays closed.

The remaining sequence, numbered as agreed on 2026-10-17 (it continues the 2026-10-16 sequence above):

**3b. Steps 1–2: validate the lane-exit evaluation and apply it to baseline/C1 (M4.8)** (completed, reviewed;
[report](m4/m4_8/m4_8_lane_exit_report.md), [protocol](m4/m4_8/m4_8_protocol.md), [tables](../results/m48/report.md),
§AU)
- **Stage 1 (validation):**
  - the readout E_h (earned-gold lead plus its expected change over h repeated continuation turns; undiscounted, no
    critic tail);
  - exact accounting, orientation and restoration;
  - five checkpoint-selected wave/Recall case types and controlled "already reset" variants;
  - calibration against recorded continuations.
  - Outcome: pre-registered classification **useful**, default window **E_20**.
- **Stage 2 (application):**
  - 2,304 natural checkpoints of the matched M4.6c policies (turn 94, plus 84, 104 and 50);
  - raw gold misleads materially in about two thirds of states;
  - baseline and C1 are indistinguishable at the lane exit;
  - M4.7's conversion slope is refined and its pilot conclusion reinforced (§AU).
- **Recommendation (adopted by the user on 2026-10-17):**
  - lane-phase outcomes as D0 → E_20 with XP, HP, absence and the wave beside it;
  - the baseline stays the recommended configuration;
  - no default changed.

**3c. Step 3: design Bayesian opponent modelling** (completed, reviewed; §AV, §AW)
- OM1, the feasibility study (evaluation only): outcome 1, feasible and worthwhile
  ([report](opponent_modelling/om1_feasibility_report.md)).
- The design is written and implemented behind opt-in switches ([design](opponent_modelling/om2_design.md)):
  - the previous-action observation field;
  - the belief over 11 library types and two catch-alls;
  - the collector hooks;
  - zero-weight widening.
- *Correction (2026-10-17, M5a):* until this update the entry read "not designed or implemented", which had been
  stale since OM2.

**3d. Step 4: matched opponent-modelling pilot** (completed, reviewed; §AW,
[report](opponent_modelling/om2_pilot_report.md))
- Control (belief at the prior) against treatment (the belief), 3 seed pairs × 4.096M from `ext/baseline:s`.
- Pre-registered outcome 1: modelling pays and is safe.
- Exploratory, against the common start: the belief protects competence that the 50% fixed-opponent mix otherwise
  erodes slightly.

**4. Step 5: M5, the final training and evaluation campaign** (M5a and its corrections completed, §AX and §AY; M5b
training and the initial M5d evaluations completed, §AY; M5c completed, §AZ; the core attackers and the M5e closeout
completed, §BA; M5 committed by the user, a07e3fdf; the full plan is "M5 plan" above and [m5_design.md](m5/m5_design.md))
- *Superseded 2026-10-17.* M5 now compares the opponent-modelling package with continued pure self-play (the user's
  decision in the M5a brief). The proposal below, from the closeout, is kept for the record: its fresh initialization
  was replaced by continuations of `ext/baseline:s`, and its "adaptive opponents" by the modelling arm with M5d's
  opponent families and attackers.
- The commitments below are the agreed sequence of 2026-10-16. The closeout report §8 lists what they leave
  underspecified and proposes a concrete scope, labelled as a proposal:
  - an objective;
  - success evidence S1–S6;
  - the training setup;
  - the meaning of "adaptive opponents";
  - the first task, M5a, the pre-registration and readiness check.
- Multi-seed shared-policy PPO on the setup selected at closeout.
- Evaluation reuses the existing infrastructure:
  - learning progress against fixed references and own history;
  - strategic counterfactuals (the M4.5c battery);
  - diverse opponents (historical, scripted, adaptive), which absorbs the earlier adaptive-opponent proposal;
  - cross-seed results;
  - empirical exploitability with best-response attackers.
- Budget and initialization (fresh or continued) are chosen from evidence at closeout.
- Values in strength-gap states are measured with rollouts, not with the critic (M4.6e).
- Lane-phase outcomes are read at a lane-exit checkpoint with the raw lead, the lane-exit readout and the non-gold
  dimensions side by side (M4.8).
- It follows the opponent-modelling design and pilot (steps 3–4); whether modelling enters the campaign depends on
  the pilot.

**5. Step 6: M6, presentation, interactive demonstration and reproducibility** (approved direction, 2026-10-19,
extended 2026-10-21; M6a completed, §BC; M6b completed, §BD; the full plan is "M6 plan" above and
[m6_plan.md](m6/m6_plan.md))
- A new web application in which a human plays against either M5 arm (Play) and watches a bounded bot match
  (Watch), with the verified research replays and the findings. A concise README, a research overview, reproducible commands and three bundles. The Streamlit viewer is
  archived as legacy.
- Demonstrated findings are kept separate from remaining hypotheses.
- *Superseded 2026-10-19.* This step used to name "a visual demonstration built on the viewer". The user chose a new
  web application instead and does not want the Streamlit viewer reworked.

**Stopping principle.** A perfect critic, or the discovery of every recognizable laning strategy, is not required. If
the remaining weakness is small, confined to unusual states, or resistant to one justified targeted intervention,
document it and proceed, unless it undermines the central research claim.

*Earlier proposal, kept for the record (2026-10-14).* The original plan (section D) named M5 "Bayesian opponent
modelling and exploitation versus robustness" and M6 "Research dashboard". The scope decision of 2026-10-11 narrowed
the research to OpenAI Five-inspired self-play with PPO, after which they were re-proposed:
- M5 as adaptive-opponent and game-theoretic research (robustness and exploitation against historical, scripted and
  adaptive opponents, from the existing shared-PPO approach);
- M6 as research presentation and portfolio.

The agreed sequence above supersedes that proposal: adaptive and diverse opponents become part of M5's evaluation.

---

*Original status line of the first version (before M1, kept for the record):*

*Status: planning document, October 2026. Nothing in this document has been implemented. It was written after inspecting the engine (`lanegame/`), the three Phase 1 reports, the test suite (305 tests) and the siege-study results, and after a short throughput benchmark of the existing engine (section I). Every cost figure that is not marked "measured" is a derived expectation, not a promise. Terminology follows the siege study: the midpoint cell x = 0 is "the lane"; coordinates are own-perspective unless stated (own tower at −T = −2, the lane at 0, enemy tower at +T = +2).*

---

## A. Executive recommendation

> **Design standard (added 2026-10-12, M4.2e-B):** every mechanic proposed below and in later sections is judged against [design_philosophy.md](design_philosophy.md): simplify how League works without changing why its laning strategies work.

**Keep the exact game as the benchmark, add a champion layer beside it, and train against it with PPO self-play.** Concretely:

1. **Do not fork the simulator.** The deterministic lane dynamics (`mechanics.step_turn`), the legality rules, the farming economy, the gank-probability formula and the mirror symmetry are reusable as they stand. The RL environment should wrap them, not copy them. One optional argument to `step_turn` (a per-side champion reach cap) is the only change the existing mechanics need, and it defaults to current behaviour.
2. **Put the new mechanics in a separate champion layer** (`lanegame/champions.py`) that runs after the lane step inside one turn: persistent champion position, HP, the four behaviours (Farm-Hold, Farm-Yield, Trade, Safe), Recall, stochastic trades, ganks as events with HP consequences, death and absence. The layer is switched on by one configuration flag that defaults to off, so `GameConfig()`, `GameConfig.v2()` and `GameConfig.v1()` keep reproducing their reports bit for bit.
3. **Expose both games through one simultaneous-move environment** (`lanegame/env/`): `reset()`, `step(action_a, action_b)`, own-perspective observations that are pure functions of (turn, state), legal-action masks, zero-sum per-turn rewards, deterministic seeding with separate random streams for chance events and for agents. A PettingZoo `ParallelEnv` adapter is a thin optional wrapper over it, not the core.
4. **Make the exact solver the regression oracle for learning.** Any agent whose decision is a function of the observation is a Markov policy of the exact game, so on the legacy configurations (K ≤ 16, under six seconds to solve) its exploitability, its exact value against any scripted opponent, and the gap between what it learned and the exact best response can be computed to machine precision. That is how the trainer is validated before it is pointed at the champion game, where no exact answer exists.
5. **Train in stages:** PPO with action masking and one network for both seats (the mirror symmetry makes this free), first against fixed scripted opponents on the exact game, then in self-play with an opponent pool on the champion game, then with approximate best responses as the exploitability measure, and only then PSRO, for which the existing `solve_matrix_game` is already the meta-solver.
6. **Keep every milestone small and measured.** No new milestone performs a K = 32 exact solve (the existing `results/siege/solution_K32.json` is the frozen artifact for that), tests that need an exact solve use K ≤ 16, throughput is benchmarked before any training is scheduled, and research sweeps are separated from correctness tests.

**Why this makes sense.** The siege study showed where exact enumeration stops paying: the vision timer alone multiplied the state space twelvefold and one more persistent counter (the per-tower plate budget) multiplied it by a further 3.5 before anything was solved. Every mechanic the research question now needs (HP, position, absence timers, levels, items, plate durability) is exactly that kind of persistent counter, so exhaustive enumeration of the expanded game is not a realistic goal. At the same time, the question itself ("when to exploit, how much risk to accept when the model may be wrong") is a question about policies, which RL answers directly and which the exact game can still arbitrate on small instances. The repository is unusually well prepared for this: the state is already own-perspective and mirror-symmetric, the policies are already probability vectors over legal sets, the Monte Carlo simulator already samples with a seeded generator, and the engine already runs about 25,000 lane turns per second in pure Python (measured), which is enough for the first training experiments without any vectorisation work.

---

## B. Current architecture assessment

### B.1 What exists and what it is worth for the transition

| Module | What it does today | Verdict for the RL transition |
|---|---|---|
| `config.py` `GameConfig` | Frozen dataclass with every parameter, presets `v1()` / `v2()`, `with_()`, JSON round trip, `validate()` | **Reuse and extend.** New champion fields default to "off"; presets stay unchanged. Per-seat parameters (mechanical skill) are a new kind of field: the first asymmetric ones. |
| `state.py` `State` | Own-perspective per-side records, `mirror()`, `side(s)`, list round trip | **Reuse unchanged for the lane.** The champion layer adds its own per-side record; the environment state is a composite (lane `State` + two champion records + k). Keeping `State` untouched keeps every exact code path bit-identical. |
| `mechanics.py` `step_turn`, `legal_actions`, `TurnResult` | Deterministic tick loop, merges, fights, thinning, crashes, tower fire, spawns, plates, vision; kills keyed by position for farming; exposure | **Reuse as the shared simulator core.** Needs one optional argument (per-side champion reach cap, default = today's `posture_cap` table). `TurnResult.kills` is exactly what XP (M4) needs. `legal_actions` is the mask generator for the legacy action space. |
| `game.py` `LaneGame`, `gank_probability`, `collection_efficiency`, `turn_farm` | Cached transitions, expected gank payoff, farm collection, reachable layers | **Reuse the functions, leave the class alone.** `gank_probability` becomes the probability that a gank *occurs*; `collection_efficiency` is reused with a behaviour-to-posture mapping. `LaneGame` stays the exact game's adapter. Its unbounded transition cache and interning dictionary are right for enumeration and wrong for long RL rollouts (see B.3). |
| `matrix_game.py` | Exact zero-sum matrix games with certificates; support enumeration | **Reuse.** It is the PSRO meta-solver and the per-state check for exact exploitability. |
| `solver.py`, `best_response.py`, `analysis.py` | Backward induction, exact best responses and exploitability for any `Policy`, exact forward statistics | **Reuse as the benchmark.** They assume one lane successor per joint action (plus the player-chosen crash options). A chance-node extension is a modest, optional change (M2e) and must be guarded by the pinned v1 / v2 / siege values. |
| `policy.py` `Policy`, `TabularPolicy`, `ScriptedPolicy`, `own_view` | `move(k, state, side, legal)` returns a distribution; own-view rule for scripted policies | **Reuse.** The `Agent` of the RL environment and the `Policy` of the exact game are two views of the same object; the adapters in both directions are a few lines each. |
| `baselines.py` | Twelve scripted opponents (siege tendencies included), `prefer` / `resolve` helpers | **Reuse as the scripted-opponent suite** for M1 and as the first opponent *types* for M5. M2 adds champion-behaviour baselines. |
| `simulate.py` | Seeded Monte Carlo over `Policy` pairs, realized ganks, expected farm | **Superseded for RL by the environment, kept for the exact models.** The environment must agree with it statistically (test) but need not consume random numbers in the same order. |
| `io.py` | JSON solution schema version 3 | **Pattern to copy** for the episode-trace schema the dashboard will read. The 68 MB `solution_K32.json` is the frozen K = 32 equilibrium. |
| `reachability.py`, `scenarios.py`, `render.py` | Reachable events with replayable witnesses, 23 traced scenarios | **Reuse as the validation method** for every new rule: each champion mechanic gets scenarios and reachable-event witnesses before any training. |
| `experiments/`, `experiments/v1/`, `experiments/siege/` | The three studies | **Untouched.** New research code goes in new sub-packages. |
| `tests/conftest.py` | Session-scoped K = 16 solves of three models | **Reuse.** The fixtures are the exact oracle for the environment tests. The suite takes about five minutes because three K = 16 solves and many exhaustive transition sweeps run every time; a `slow` marker (M0) keeps new-module development fast. |
| `legacy/` | The first Q-learning prototype: an opponent pool with periodic greedy snapshots | Reference only. Its training loop is the ancestor of the pool-based self-play proposed in F. |

### B.2 Measured costs (this machine, single-threaded Python 3.13, 12 logical CPUs)

| Quantity | Measured value | Source |
|---|---|---|
| `step_turn` + `legal_actions`, uncached, siege model | about 25,000 turns/s (40 µs per turn) | benchmark run for this document |
| Full K = 32 episode through `simulate`, scripted policies | 475 to 750 episodes/s (15,000 to 24,000 turns/s) | same run |
| Exact solve, siege model, K = 8 / 12 / 16 | 0.2 s / 1.2 s / 5.6 s (189 / 1,039 / 4,951 nodes) | same run |
| Exact solve, siege model, K = 24 / 32 | about 30 s / 3.5 min, 616 MB / 3.05 GB | `results/siege/benchmark.md` |
| Exact solve, previous revision (`v2`), K = 32 | under 10 s, 116 MB | `results/revision/horizon_benchmark.md` |
| Full test suite | 305 tests, about 5 min | README, siege study section I |
| Regenerating the siege study | about 2 h 55 min | siege study section J |

The lane simulator is therefore not the bottleneck for early RL. One million environment turns cost roughly 40 to 70 s of simulator time at the measured rate; the champion layer will add work (two more random draws and a position update per side per turn) that must be measured in M2e, and policy-network inference will dominate in any case.

### B.3 Integration points

- **Turn structure.** `step_turn(state, k, a, b, cfg)` is a pure function with a complete event record. The champion layer can be inserted *around* it: resolve champion intentions before the lane step (to know how far forward each champion acts on its wave), run the lane step, then resolve trades and ganks from the positions the turn produced.
- **Actions as integers plus a legal set.** Every policy already works with "a probability vector aligned with the legal tuple". A mask over a fixed action space is the same information in a fixed shape.
- **Own-perspective state.** `State.mirror()` and `own_view` mean one policy can play either seat. For a neural policy this halves the data needed and removes seat-specific bugs. Mechanical skill breaks the *symmetry* but not the *equivariance*: the observation just has to include own and enemy skill.
- **Markov observations.** If the observation is a function of (k, state, side) only, any agent is a `Policy` of the exact game. That is the bridge to `evaluate`, `best_response_value`, `exploitability` and `path_statistics`.
- **Event accounting.** `TurnResult.kills[s]` (HP of enemy minions that died, keyed by own-perspective cell) is the natural input for proximity XP; `TurnResult.exposure`, `plates`, `thinned`, `vision_set` are the natural per-turn statistics for the dashboard.

### B.4 Limitations and risks for the transition

1. **The solver assumes a deterministic lane.** `build_q` reads one `pre_state` per joint action. Ganks are expected losses, not events. Exact solving of any configuration with stochastic HP damage needs chance nodes (an expectation over successors in `Q`, in `induct` and in `path_statistics`). Small change, but it touches the certified core; it must be guarded by the pinned values.
2. **Action encoding is coupled to posture.** `a = wave + 3·posture + 6·objective` with `posture ∈ {Contest, Safe}`. The four champion behaviours (plus Recall and an "absent" no-op) need a new encoding; the old one must remain valid for the old configurations.
3. **Exposure is derived from the wave.** The champion "is where its wave goes" (`reach`), capped by posture. Persistent position makes exposure a state variable instead. The gank formula is unaffected (it takes an exposure integer); its *input* changes.
4. **Caches grow without bound.** `LaneGame._cache` and `_intern` are right for enumeration (2.7 M entries, 3 GB at K = 32) and wrong for a long-running environment with stochastic successors. The environment should cache only the deterministic lane step, keyed by (lane state, phase, lane-relevant action components), with a size bound.
5. **One random stream.** `simulate.play_episode` draws actions, ganks and choices from one generator. For reproducible research the environment needs separate streams: chance events (trades, ganks) and agents must be seedable independently, so that two policies can be compared on identical chance draws (common random numbers).
6. **Expected gank penalty inside the payoff.** `Transition.payoff` includes `L·(p_B − p_A)`. In champion mode that term must disappear (the cost of a gank is HP and absence) without changing the legacy payoff.
7. **Symmetry assumptions in tests.** Several invariant tests (`test_mirror_equivariance_of_transitions`, `test_value_antisymmetry`) assume the two players are identical. With per-seat skill they hold only when the skills are equal; the tests must be parametrised accordingly.
8. **Full information, no history.** The exact game is perfect-information and Markov. Opponent modelling (M5) needs public action history as a *feature*; the environment must expose it in `info` without putting it into the Markov observation used for the exact bridge.
9. **State-space growth is immediate.** Per side, 5 positions × 7 HP values × 4 absence values is a factor of 140; squared, 19,600 on top of the lane states. Exact enumeration of champion configurations is feasible only at tiny horizons (K ≤ 8) with reduced ranges (section D, M2e). This is expected and is the reason for the transition.
10. **Test-suite time.** Five minutes per run is already a tax on every change; the new modules must not add solves to the default run.

---

## C. Proposed target architecture

### C.1 Layers

```
                       ┌──────────────────────────────────────────────────────┐
                       │  research utilities                                   │
                       │  experiments/rl/   opponent types, posteriors,        │
                       │  tournaments, frontier sweeps, trace export (M3–M6)   │
                       └───────────────▲──────────────────────────▲───────────┘
                                       │                          │
     ┌─────────────────────────────────┴───────┐     ┌────────────┴─────────────────────────┐
     │  exact benchmark  (unchanged modules)   │     │  learning  lanegame/rl/  (M3)         │
     │  solver.py  best_response.py            │     │  PPO (masked, own-view shared net),   │
     │  analysis.py  matrix_game.py            │     │  opponent pool, approximate BR, PSRO  │
     │  LaneGame (game.py)                     │     │  Agent protocol; Policy <-> Agent     │
     └───────────────▲───────────────▲─────────┘     └────────────▲──────────────────────────┘
                     │               │  Policy wrapper (Markov obs)  │
                     │               └───────────────┬──────────────┘
                     │                               │
     ┌───────────────┴───────────────────────────────┴───────────────────────────┐
     │  environment  lanegame/env/  (M1)                                          │
     │  LaneEnv: reset / step(a, b) / masks / obs / rewards / seeding / trace      │
     │  observation.py (own-view, Markov), agents.py (scripted), pettingzoo.py    │
     └───────────────▲───────────────────────────────▲───────────────────────────┘
                     │                               │
     ┌───────────────┴───────────────┐   ┌───────────┴──────────────────────────┐
     │  lane mechanics (existing)    │   │  champion layer  lanegame/champions.py │
     │  mechanics.step_turn          │◄──┤  position, HP, behaviours, trades,     │
     │  legal_actions, farming,      │   │  ganks as events, recall, death,       │
     │  gank_probability, plates,    │   │  absence; later XP, items, plate HP    │
     │  vision                       │   │  (M2, M4)                              │
     └───────────────────────────────┘   └───────────────────────────────────────┘
                     ▲                               ▲
                     └──────────── GameConfig ───────┘   (one config; champion fields default off)
```

- **Shared simulator** = `mechanics.step_turn` plus the champion layer. The exact benchmark uses the lane mechanics directly (as today) and, optionally, the champion layer on tiny configurations through a chance-node game class. The RL environment uses both.
- **Exact benchmark** = the existing modules, untouched except for the optional chance-node generalisation.
- **Environment** = the only thing agents and trainers talk to. It depends on NumPy only.
- **Learning** = PyTorch code that depends on the environment and never on the solver.
- **Research utilities** = experiment scripts that may use everything.

### C.2 Answers to the fifteen architectural questions

1. **Build on `LaneGame` or add a layer?** Add a layer. `LaneGame` stays the exact game. The environment calls `step_turn` and the champion layer directly, with its own bounded cache. Reason: `LaneGame`'s contract (deterministic successor, expected gank payoff, unbounded cache) is exactly what the exact solver wants and exactly what an RL environment does not.
2. **Exact solver without constraining RL mechanics?** Three mechanisms: (a) configuration flags that default to the legacy behaviour, pinned by the existing reproduction tests; (b) the Markov-observation bridge, so any learned agent is exactly evaluable on legacy configurations; (c) an optional chance-node game class for tiny champion configurations, gated by a reachable-node budget. The solver never has to handle the full champion game.
3. **PettingZoo Parallel API, custom interface, or both?** Both, in that order of importance: a custom simultaneous-move core (`LaneEnv.step(a, b)`), plus a thin `ParallelEnv` adapter behind an optional import. Trade-offs: PettingZoo's Parallel API is already simultaneous and gives tooling compatibility, but it imposes string agent ids, Gymnasium space objects, a dependency, and dictionary plumbing that the exact bridge and the tests do not need. The adapter is about a hundred lines and is tested with `pettingzoo.test.parallel_api_test` when the package is installed.
4. **Observation structure?** Own-perspective, fixed-size, Markov. Scalars (turn fraction, turns remaining, phase one-hot), a 9-cell lane grid with own and enemy minion HP per cell, per-side siege mass, vision timer, pending plates and siege eligibility; in champion mode also HP, position one-hot, absence timer, level, items, skill for both sides. Both a flat vector (for networks) and the structured state (for scripted agents and analysis) are returned. Normalised by configuration ranges.
5. **Action combinations and masks?** One flat `Discrete` index over the product of components, with a boolean legality mask from the engine. Legacy mode: the existing 18 codes. Champion mode: wave (3) × behaviour (5: Hold, Yield, Trade, Safe, Recall) × objective (3) = 45 codes plus one `ABSENT` no-op, 46 in total; typically 4 to 9 are legal. Masking by setting illegal logits to −∞. Component factorisation of the policy head is deferred unless the flat head learns poorly.
6. **Persistent champion position?** An own-perspective cell `c ∈ {−T, …, +T}` (five values), moved at most `champion_speed` cells per turn toward a behaviour-dependent target; exposure for ganks is the maximum of the start and end cells. Details in E.4. The three-level abstraction (Defensive / Wave / Forward) is the fallback if five cells prove too fine.
7. **Reproducible stochastic trading and ganks?** All chance in the champion layer is drawn from one `numpy.random.Generator` owned by the environment and seeded at `reset`, separate from any agent randomness; draws happen in a fixed order (A's trade, B's trade, A's gank, B's gank, damage). Same seed, same policies, same trajectory; same seed, different policies, same chance draws until the first divergent action (common random numbers).
8. **Recall, death, absence?** One per-side integer `absent` (turns still away). Recall sets it to `recall_turns` (2) and is legal only near one's own tower; death sets it to `death_turns` (3). While absent the only legal action is `ABSENT`, the wave follows a configurable default wave action, farm collection and zoning are zero, the champion returns at the own tower cell with full HP. Details in E.8.
9. **What should the reward optimise?** The per-turn zero-sum relative resource gain: (farm + plates + bounty) of A minus the same of B, realized (not expected), undiscounted, terminal value 0. No gank penalty term in champion mode; the cost of being ganked is HP and absence. Optional shaping knobs (death penalty, terminal HP value) exist in the configuration and default to zero.
10. **Zero-sum with HP, items, XP, tower damage?** By construction: a single scalar `r` per turn, A receives `r`, B receives `−r`, and HP, XP, levels, items and plate damage are *state*, never reward. Gold spent on items is not a loss (it was already counted when earned) and item power is not a gain (its value is what it earns later). Plate rewards are paid when thresholds break, once. Terminal valuation of unspent gold, items or HP is a configuration knob defaulting to zero; if ever switched on it must be applied symmetrically so the game stays zero-sum.
11. **Essential before RL training?** M1 (interface) and M2 (HP, position, behaviours, trading, skill, ganks as events, recall, death). These are the mechanics the research question is about; training on anything less answers a different question.
12. **Safe to add afterwards?** XP and levels, spendable gold and items, plate durability, distance-based farming, cannon waves, passive regeneration, partial observability. All are state additions that extend the observation and the mask without changing the step contract.
13. **Most practical first baseline?** PPO with action masking, one network for both seats via the own-view observation, Generalised Advantage Estimation, a small MLP, CPU training, implemented in the style of a single-file CleanRL script rather than through a multi-agent framework. Validate it against fixed scripted opponents on the exact game before any self-play.
14. **Validation against the exact benchmark?** Wrap the trained network as a `Policy` (deterministic in the observation), then: exact value against each scripted opponent (`evaluate`) versus the Monte Carlo estimate; exact best-response value (`best_response_value`) versus what PPO reached when trained against that opponent; exact exploitability of the self-play policy on K ≤ 16; convergence of exploitability over training. All on legacy configurations and, if M2e lands, on tiny champion configurations.
15. **Opponent modelling later without rewriting agents?** The agent sees an `Observation` built by a pluggable builder; belief features are an optional appended block (zeros until M5). Opponent models are `Policy` objects (the existing interface), so likelihoods are exact action probabilities at (k, state). The environment exposes the public joint-action history in `info` for posterior updates and never exposes the opponent's current action. Adaptive agents are ordinary agents with internal state.

---

## D. Implementation milestones

Each milestone is meant to be one bounded Claude Code task with its own tests and its own definition of done. Sizes are given as sessions (S: a single short session; M: one to two sessions; L: two to three sessions) rather than hours, because the uncertainty is in review and design iteration, not typing.

### M0. Test-speed hygiene (optional, S)

- **Scope.** Add a `slow` pytest marker to tests that need a K = 16 solve or an exhaustive transition sweep; add `python -m pytest -m "not slow"` as the development loop; document it in the README. No behaviour change.
- **Dependencies.** None.
- **Files.** `tests/conftest.py`, `pyproject.toml` (`markers`), README.
- **Definition of done.** The fast subset runs in well under a minute; the full suite is unchanged (305 pass).
- **Cost.** Development only.
- **Risk.** Mislabelling a fast test as slow; harmless.
- **Defer.** Any test restructuring beyond markers.

### M1. RL-ready environment around the existing game (M)

Specified in full in section J. Summary: `lanegame/env/` with `LaneEnv`, Markov own-view observations, masks over the 18 legacy codes, zero-sum realized rewards, seeded chance stream, scripted `Agent`s playing complete episodes, Policy/Agent adapters, an optional PettingZoo adapter, an `env-demo` CLI with a throughput print, and tests that tie the environment to the exact solver.

- **Dependencies.** None (M0 recommended).
- **Files.** New: `lanegame/env/{__init__,core,observation,agents,trace,pettingzoo_adapter}.py`, `tests/test_env.py`. Touched: `lanegame/__main__.py` (one subcommand), `pyproject.toml` (optional extra), README (layout). Not touched: `mechanics.py`, `game.py`, `solver.py`, `state.py`, `config.py`.
- **Definition of done.** Two scripted agents play seeded, reproducible K = 32 episodes; environment expectations match `evaluate` on the K = 16 fixtures; with ganks off every episode equals the exact value; an agent wrapped as a `Policy` has an exact exploitability; the PettingZoo API test passes when the package is installed; full suite still passes.
- **Cost.** Simulator: unchanged (measured 475 to 750 episodes/s through the old simulator; the environment should be within a factor of two). Tests: seconds beyond the existing fixtures.
- **Risks.** Observation that is accidentally not a function of (k, state) (breaks the exact bridge); mask misaligned with action codes; reward sum drifting from `simulate`.
- **Defer.** Champion mechanics, vectorised environments, Gymnasium spaces beyond the adapter, any learning.

### M2. Champion HP, trading and recall (L in total, delivered as five S/M tasks)

Specification in section E. The split below orders the work so that nothing stochastic is added before the deterministic skeleton is tested, and nothing can kill a champion before absence exists.

**M2a. Champion state, persistent position, behaviours (M).**
- Scope: `champions.py` with the per-side champion record (position, HP, absence, skill), the 46-code champion action space, legality, behaviour-to-target movement, exposure as max(start, end), behaviour-to-posture mapping for farming, the optional `reach_cap` argument to `step_turn`; `GameConfig` champion fields (all default off) and a `GameConfig.champions()` preset; the environment running the champion game; champion scenarios (`scenarios.py` set 4) and events (`reachability.py`). HP is present but nothing damages it yet.
- Dependencies: M1.
- Files: new `lanegame/champions.py`; touched `config.py`, `mechanics.py` (one optional parameter), `env/core.py`, `env/observation.py`, `scenarios.py`, `reachability.py`, `baselines.py` (champion-behaviour baselines), new `tests/test_champions.py`.
- Done: legacy configs reproduce (fixtures unchanged); movement limits, legality, exposure rule and farming mapping each have a test; mirror equivariance holds with equal skills; two scripted champion agents complete episodes.
- Risks: `step_turn` regression (guarded by the three pinned models); legality corner cases (Recall near the tower, objectives during absence).
- Defer: damage, ganks as events, death.

**M2b. Recall and absence (S).**
- Scope: Recall legality and effect, the absence timer, the absent default wave action, zero farming and zoning while absent, return with full HP at the own tower. Deterministic.
- Dependencies: M2a.
- Done: a recalling champion is away exactly `recall_turns`; the opponent's farm rises while it is away (scenario); mask is `{ABSENT}` while absent; Recall is illegal forward of the recall line.
- Defer: shopping (M4b).

**M2c. Stochastic trading and mechanical skill (M).**
- Scope: the trade resolution of E.5, the hit-chance formula with skill, the 0/1/2 damage distribution, Trade-versus-each-behaviour contingencies, the tower-shot rule, HP floor at zero *without* death (death comes in M2d; HP 0 is clamped to 1 behind a temporary flag, or deaths are simply disabled by configuration).
- Dependencies: M2b.
- Done: monotonicity tests (higher skill never lowers hit chance; Safe never raises the chance of being hit; out-of-range trades do nothing and fall back to farming); seed reproducibility; the distribution of damage over 10,000 draws matches the formula within binomial tolerance; champion scenarios for each of the five interactions of section 4.4 of the brief.
- Risks: a formula that is too strong or too weak makes Trade dominant or useless; keep every term configurable and report the default's numbers, do not tune for drama.
- Defer: level and item terms (present in the formula as zero-valued inputs).

**M2d. Ganks as events, death and respawn (M).**
- Scope: sample the gank from `gank_probability` (occurrence), then damage from the depth-and-behaviour distribution of E.7, apply HP loss, death when HP reaches zero, absence `death_turns`, return with full HP; remove the expected-loss term from the champion-mode reward; optional `kill_bounty` and `death_penalty` knobs defaulting to zero.
- Dependencies: M2c.
- Done: vision reduces only occurrence (test: damage distribution unchanged under vision); a full-HP champion cannot die to one gank at default parameters (test of the designed property); death leads to exactly `death_turns` of absence; the reward has no gank term; the zero-sum identity holds per turn.
- Defer: assists, bounties that scale, anything tower-related beyond the tower shot.

**M2e. Validation pass and exact micro-benchmark (M).**
- Scope: throughput benchmark of the champion environment; a full set of champion-mode scripted baselines (trade-happy, yield-farmer, safe-recaller, pressure-greedy); a tournament of scripted agents with confidence intervals (research sanity, not test); optional chance-node game class (`ChampionGame.outcomes`) plus solver / best-response / path-statistics generalisation, used only on a tiny configuration (K = 8, HP 3, three positions) under a reachable-node cap, with the three legacy models re-pinned.
- Dependencies: M2a to M2d.
- Done: benchmark numbers recorded in `results/env/benchmark.md`; every scripted pairing completes 1,000 seeded episodes; if the chance-node class is built, backward induction equals tree recursion on the tiny configuration and the legacy values are unchanged.
- Risks: the chance-node refactor is the only step that touches certified code; if the budget is tight, skip it and rely on the legacy-configuration bridge.
- Defer: any tuning sweep larger than a handful of settings.

### M3. Initial RL training (L, delivered as four tasks)

Specification in section F.

**M3a. PPO against fixed scripted opponents on the exact game (M).** `lanegame/rl/` with a masked PPO (own-view shared network), a batch of environments, checkpoints and seeds. Train against `passive_safe`, `aggressive` and `uniform` on the siege model at K = 16; compare the learned value to the exact best-response value and the exact value of the learned policy (via the wrapper). Done when PPO reaches a stated fraction (to be measured; aim high, but record what it is) of the exact best-response value against each fixed opponent within a budget recorded in the results. This is where the first real training-cost numbers come from.

**M3b. Self-play with an opponent pool on the champion game (M).** Same trainer; opponents sampled from recent checkpoints and the scripted suite; K = 32; symmetric skills first. Done when training is reproducible from seed, the learning curves and the metric suite (F.3) are written to `results/rl/`, and a frozen "self-play v0" checkpoint exists.

**M3c. Evaluation harness (M).** Tournament against the scripted suite with confidence intervals; approximate exploitability = value of a PPO best response trained against the frozen policy for a fixed budget; exact exploitability on legacy configurations; common-random-number pairing. Done when the harness runs from one command on a checkpoint and produces a Markdown and JSON report.

**M3d. PSRO (optional, M).** Oracle = M3a's trainer against a mixture; meta-solver = `solve_matrix_game` on the empirical payoff matrix; symmetric single population. Only after M3b and M3c are reproducible. Done when the meta-game exploitability trends down over iterations on the champion game at K = 16 or 24.

### M4. Economic progression (L, delivered as three tasks)

> **Superseded (2026-10-11).** M4 is now one integrated, separately versioned ruleset. Its specification is
> [m4_integrated_design.md](m4/m4_1/m4_integrated_design.md), with the plan in [m4_implementation_plan.md](m4/m4_2/m4_implementation_plan.md)
> and the review in [m4_design_audit.md](m4/m4_1/m4_design_audit.md); see section O. The three-task plan below and
> section G are kept for the record only.

Specification in section G.

**M4a. Experience and levels (S/M).** Proximity XP from `TurnResult.kills`, three levels, level term in the trade formula. **M4b. Spendable gold and items (M).** Gold ledger, two upgrade tracks of two tiers, purchase decisions on absence turns, no reward effect. **M4c. Plate durability (M).** Persistent tower damage counter, plate thresholds with increasing durability, plate rewards on breaks, late-game plate removal; `plate_mode="durability"` in champion configurations only.

### M5. Bayesian opponent modelling and exploitation versus robustness (L)

> **Revised (2026-10-11 / 2026-10-13).** Not pursued in this form: the 2026-10-11 scope decision narrowed the research
> to OpenAI Five-inspired self-play with PPO. M5 is re-proposed as "adaptive opponent and game-theoretic research" in
> the roadmap summary at the top. The text below is kept for the record.

Specification in section H. First on the exact game (K = 16, everything exact and cheap), then on the champion game with the M3 agents.

### M6. Research dashboard (L, not started until M3c exists)

> **Revised (2026-10-13).** Re-proposed as "research presentation and portfolio" in the roadmap summary at the top
> (the visual debugger of M4.2e-D is the closest existing piece). The text below is kept for the record.

Section D.6 below lists only what earlier milestones must leave behind for it: a versioned episode-trace schema (M1 `trace.py`, extended in M2 and M4), aggregate result JSON written by every experiment (as today), and a skill-grid export (policy behaviour as a function of both skills) produced by M3c.

---

## E. Champion mechanics specification

This section records the agreed design for the expanded game, proposes defaults where the brief left them open, and marks every remaining decision. Parameter names are proposals for `GameConfig`.

### E.1 State

Per side, in addition to the lane record (`fx, fh, rears, siege, v, plates`):

| Field | Meaning | Range (defaults) |
|---|---|---|
| `c` | champion cell, own coordinates | −T … +T (−2 … 2) |
| `hp` | champion health | 0 … `champion_hp` (6) |
| `absent` | turns still away (0 = present) | 0 … max(`recall_turns`, `death_turns`) (0 … 3) |
| `xp`, `level`, `gold`, `items` | M4 only | see G |

Per game (constant): `skill = (m_A, m_B)`, each in [0, 1].

Initial champion state: `c = −T`, `hp = champion_hp`, `absent = 0`. The environment state is `EnvState(lane: State, champ: (ChampSide, ChampSide), k: int)`; `mirror()` swaps the two champion records and the two skills.

### E.2 Actions and legality

`a = wave + 3·behaviour + 15·objective`, with `wave ∈ {SP, SH, F}`, `behaviour ∈ {HOLD, YIELD, TRADE, SAFE, RECALL}`, `objective ∈ {NONE, PRESSURE, VISION}`; plus `ABSENT = 45`. Legal set of a present champion:

- wave components as today (`wave_actions`: Freeze only when the enemy front is at or behind own −1);
- behaviours HOLD, YIELD, TRADE, SAFE always; RECALL only when `c ≤ recall_line` (−1) and only with `wave = SP` (one canonical code, no duplicates);
- siege objectives exactly as today, attached to HOLD and TRADE (the behaviours that stand with the wave); YIELD and SAFE during an eligible siege mean disengaging; HOLD or TRADE during an eligible siege *requires* an objective (as Contest does today).

Legal set of an absent champion: `{ABSENT}`. The environment steps every turn for both players, so the API never changes shape; an absent player's agent is called and must return the only legal action.

The legacy action space (18 codes) is kept for configurations with `champions=False`; `GameConfig.action_codes()` reports which one applies. Scripted baselines that only know wave and posture are mapped (Contest → HOLD, Safe → SAFE) so the existing suite plays the champion game unchanged.

### E.3 Resolution order of one turn

Both actions are read; nothing below reveals either action before both are committed.

1. **Champion intent.** From (behaviour, own state, enemy behaviour and state) compute each champion's target cell and its end cell `c'` (E.4). The Yield contingency ("retreat if the enemy Trades and is in reach") is evaluated here, from the enemy's *action*, which is legitimate because the environment resolves the joint action; the players never saw it.
2. **Lane step.** `step_turn` with the wave actions, the siege objectives, and `reach_cap = c'` per side (an absent side gets a cap below the lane so it is never present). Unchanged mechanics: movement, fights, thinning, crashes, tower fire, spawns, plates, vision.
3. **Farming.** `turn_farm` with the behaviour-to-posture mapping (E.4) and zero collection for absent sides.
4. **Trades.** For each side that chose TRADE and whose target is present and in range (E.5), draw the damage. Both draws use the pre-trade HP, so mutual death is possible.
5. **Ganks.** For each present side, draw occurrence from `gank_probability(state, side, action, cfg, exposure)` with `exposure = max(c, c')`, the vision timer as it stood at the start of the turn, and the action term of E.7; if it occurs, draw damage (E.7). Trade damage is applied before the gank draw, so a champion weakened by a trade is more likely to die to the gank that follows.
6. **Death and recall.** HP ≤ 0 → `hp = 0`, `absent = death_turns`, side leaves the lane (`c = −T` on return). RECALL chosen this turn → `absent = recall_turns`. A returning champion (timer reaching 0 at the end of this turn) stands at `c = −T` with `hp = champion_hp` at the start of the next turn.
7. **End of turn.** Timers (vision, absence) count down exactly as vision does today; XP, gold and purchases (M4).
8. **Reward.** `r = (farm_A + plates_A + bounty_A) − (farm_B + plates_B + bounty_B)`; `info` carries the full decomposition and every event.

Two consequences worth stating: farm collected in the turn of one's death is kept (the death is resolved after the lane step), and a champion that recalls still participates in nothing this turn and is not gankable (it channels at a cell it must already have retreated to, `c ≤ −1`).

### E.4 Persistent champion position

Target cell by behaviour (own coordinates, `w = own front cell` if a front exists, else the current cell):

| Behaviour | Target | Notes |
|---|---|---|
| HOLD | `clip(w, −T, T)`; `+T` if own crashed mass is under the enemy tower and an objective was chosen | today's Contest |
| TRADE | HOLD target + 1, clipped to `+T` | the aggressive step forward the brief asks for |
| YIELD | HOLD target; HOLD target − 1 if the enemy chose TRADE and its end cell is within trade range | contingent retreat |
| SAFE | `min(c, safe_line)` with `safe_line = −1` | never forward; today's Safe cap |
| RECALL | `−T`, then absent | legal only from `c ≤ −1` |

Movement: `c' = c + clip(target − c, −champion_speed, +champion_speed)` with `champion_speed = 2` (minions move two cells per turn; a champion keeps pace with its wave). Exposure for the gank formula: `e = max(c, c')`. Persistence therefore bites in two places: a champion that stands at `+2` and chooses SAFE is still exposed at `+2` this turn and reaches `0` at best; it cannot Recall until it has spent a turn getting to `−1`.

Farming posture: HOLD and TRADE → Contest efficiencies; YIELD → Contest unless it retreated this turn, then Safe; SAFE → Safe; absent → 0. Zoning uses the same mapping for the opponent. The existing `collection_efficiency` is unchanged in M2; a distance-based efficiency (kill cell versus champion cell) is a possible later refinement.

**Open decisions.** `champion_speed` 2 (default) versus 1 (stronger persistence; Hold could not follow a Shove). Whether Trade's "+1" should apply when the enemy champion is already adjacent (no need to step forward). Whether the three-level abstraction should replace the five cells if the position dimension slows learning.

### E.5 Stochastic trading

A side `i` that chose TRADE against a present target `j` *connects* if the global distance between their end cells is at most `trade_range = 1` (global positions: A at `c_A`, B at `−c_B`; distance `|c_A + c_B|`). If it does not connect (the target retreated, is far back or absent), the attacker falls back to farming at its forward position: no damage, Hold efficiencies, Trade exposure.

If it connects, the hit chance is

`h_i = clip(h0 + α·(m_i − m_j) + β[behaviour_j] + γ·(σ_i − σ_j) + λ·(level_i − level_j) + ι·(off_i − def_j), h_min, h_max)`

with `σ` the leading-wave stack count (as in the gank formula), `β[HOLD] = β[TRADE] = 0`, `β[YIELD] = −0.25`, `β[SAFE] = −0.40`, and the level and item terms zero until M4. Damage: `0` with probability `1 − h`, `1` with probability `h·(1 − q2)`, `2` with probability `h·q2`, `q2 = q2_base + q2_skill·m_i`.

Proposed defaults: `h0 = 0.55`, `α = 0.4`, `γ = 0.05`, `h_min = 0.05`, `h_max = 0.95`, `q2_base = 0.3`, `q2_skill = 0.2`. At equal skill and Hold defence a trade lands 1 HP about 38% of the time and 2 HP about 17%; against Safe about 10% and 5%. These numbers are a starting point to be reported, not calibrated.

Tower: a connecting attacker whose end cell is `+T` while the enemy tower is *not* occupied by its own crashed mass takes `tower_shot = 1` HP (a dive). With its mass under the tower the tower is busy, consistent with the thinning rule.

Trade versus Trade: both roll from pre-trade HP. Trade versus Hold: the defender takes the roll and farms at full efficiency. Trade versus Yield: the defender steps back (if in reach), takes the reduced roll, farms at Safe efficiency. Trade versus Safe: the roll is reduced further and often does not connect; the attacker is exposed at the forward cell either way. Farm versus Safe: no roll; the farmer collects at Contest efficiency without zoning. None of these is hard-coded as an outcome; they follow from positions, ranges and `β`.

**Open decisions.** Whether `h` should also depend on the attacker's own HP (desperation) or on the defender's HP (execute). Whether to add passive regeneration (`regen_at_tower = 1` HP per turn while SAFE at `−T`; default 0). Whether minions near the target should deal a flat point of damage to a diving attacker (default no).

### E.6 Mechanical skill

`skill: (m_A, m_B)` in `GameConfig`, default `(0.5, 0.5)`. Skill enters only execution probabilities: the hit chance (through `α`) and the two-damage chance (through `q2_skill`); the same difference term rewards defensive execution, because the defender's skill lowers the attacker's `h`. Skill does not enter the gank model in M2 (open: a `−κ·m_i` term on gank damage). Skill is observed by both agents (own and enemy) so one network can play both seats and so policies trained over a distribution of skills can be queried at any slider position later. Nothing in the environment makes a skilled champion aggressive; whether higher `m` leads a learned policy to trade more, stand further forward and get ganked more is the first empirical question of M3b.

### E.7 Ganks as events

- **Occurrence.** `P(gank_i) = gank_probability(...)` as today: base, exposure, action, wave-strength and vision terms. The action term becomes behaviour-and-wave based: HOLD +0.02, YIELD +0.02, TRADE +0.05, SAFE 0, plus +0.03 for Shove (today's SH-C 0.05 minus SP-C 0.02). Vision multiplies occurrence by `(1 − ρ)` and nothing else.
- **Damage given a gank.** `g = 1 + X`, `X ~ Binomial(2, p_e)`, `p_e = clip(p_g0 + p_g1·(e + T)/(2T) − p_def·[behaviour ∈ {YIELD, SAFE}], 0, 1)`, defaults `p_g0 = 0.2`, `p_g1 = 0.6`, `p_def = 0.2`. At the own tower a gank deals 1 / 2 / 3 with probabilities 0.64 / 0.32 / 0.04; under the enemy tower 0.04 / 0.32 / 0.64.
- **Death.** `hp − g ≤ 0`. With `champion_hp = 6` and a maximum of 3 gank damage, a full-HP champion never dies to a single gank: death requires prior trade damage or a second gank, which is how "probability of death depends on current HP" emerges without a separate death roll.
- **Consequences.** `absent = death_turns = 3`; the opponent's gain is the absence (free farm, free sieges, no zoning), plus `kill_bounty` resources if that knob is switched on (default 0). `gank_loss` (the expected-loss penalty L) is unused in champion mode and removed from the reward.

**Open decisions.** Whether a champion with its own crashed mass under the enemy tower should be gankable at a lower rate (tower aggro on minions). Whether a killed champion should drop the plates pending on its siege (default: yes, as the siege ends when its mass is cleared anyway).

### E.8 Recall, death, absence and return

| Event | Turns away | Where it leaves from | Return |
|---|---|---|---|
| Recall (chosen in turn k) | `recall_turns = 2` (turns k and k+1) | `c ≤ −1` required | start of turn k+2 at `−T`, full HP |
| Death (in turn k) | `death_turns = 3` (turns k+1, k+2, k+3) | anywhere | start of turn k+4 at `−T`, full HP |

While absent: legal set `{ABSENT}`; the lane uses `absent_wave_action = SP` with no champion presence (no shove damage, no freeze-break push, no objectives, no thinning); farm collection and zoning are zero; the side cannot be traded with or ganked; its vision timer keeps counting down; its pending plates lapse when its mass is cleared, as today. In M4, the absent turns are when purchases happen.

**Open decisions.** `absent_wave_action`: Slow Push (the stylised park-at-the-lane rule; default, no new code path) versus a "natural" free walk (a new movement mode). Whether Recall should take effect at the end of the turn instead (one more turn of exposure, more realistic, more complex).

### E.9 Reward and objective

`r_k = (F_A + P_A + B_A) − (F_B + P_B + B_B)`, realized, with `B` the kill bounty (default 0); zero-sum by construction; undiscounted; terminal value 0. Knobs defaulting to zero and intended for ablations only: `death_penalty`, `kill_bounty`, `terminal_hp_value`. The per-turn `info` dictionary reports farm, plates, bounty, trade damage dealt and taken, gank occurrence and damage, deaths, recalls, positions, exposure and the legal sets, so every metric in F.3 is a sum over `info`.

### E.10 Defaults at a glance

| Parameter | Default | Parameter | Default |
|---|---|---|---|
| `champions` | False (legacy); True in `GameConfig.champions()` | `champion_hp` | 6 |
| `champion_speed` | 2 | `safe_line`, `recall_line` | −1, −1 |
| `recall_turns`, `death_turns` | 2, 3 | `absent_wave_action` | SP |
| `trade_range` | 1 | `h0`, `α`, `γ`, `h_min`, `h_max` | 0.55, 0.4, 0.05, 0.05, 0.95 |
| `β` (HOLD, YIELD, TRADE, SAFE) | 0, −0.25, 0, −0.40 | `q2_base`, `q2_skill` | 0.3, 0.2 |
| `tower_shot` | 1 | `skill` | (0.5, 0.5) |
| gank action term (HOLD, YIELD, TRADE, SAFE, +SH) | 0.02, 0.02, 0.05, 0, +0.03 | `p_g0`, `p_g1`, `p_def` | 0.2, 0.6, 0.2 |
| `kill_bounty`, `death_penalty`, `terminal_hp_value` | 0, 0, 0 | `gank_loss` | unused in champion mode |

### E.11 Decisions required before M2 implementation

1. `champion_speed` 2 or 1 (E.4).
2. Whether siege objectives may attach to YIELD (E.2; proposed no).
3. Recall timing: immediate (proposed) or end of turn (E.8).
4. `absent_wave_action` (E.8; proposed SP).
5. Whether to include a kill bounty at all in the default (proposed 0) and whether to include passive regeneration at the own tower (proposed 0).
6. Whether skill enters gank damage (proposed no in M2).
7. Whether the exact chance-node micro-benchmark (M2e) is wanted at all, given that the legacy bridge already validates the trainer.

---

## F. RL algorithm and evaluation plan

### F.1 Why PPO with masking, and why not a framework

The game is two-player, zero-sum, simultaneous-move, finite-horizon (K = 32), perfect-information with chance, with 4 to 9 legal actions out of 46 and episodes of 32 steps. Policy-gradient methods with action masking handle variable legal sets cleanly; the horizon is short enough that bootstrapped value estimates are easy; the mirror symmetry lets one network play both seats, which halves sample needs and makes self-play a single-population problem. PPO is the most-used and best-understood option for this shape, is stable without much tuning, and its code is short.

Rejected or deferred: Stable-Baselines3 (single-agent, masking only through `sb3-contrib`, awkward for simultaneous two-player play); RLlib (heavy dependency, slow iteration for a game that steps in microseconds); Deep CFR and regret methods (overkill for perfect-information with chance, and the exact solver already provides equilibria where they are computable); NFSP (a reasonable second baseline if self-play PPO cycles; it needs a supervised average-policy network, which is deferred); tabular methods (the champion state is too large, and the legacy game is already solved exactly).

Implementation: PyTorch on CPU, a single-file PPO in the style of CleanRL adapted for (a) a batch of `LaneEnv`s stepped in a Python loop (vectorisation deferred until measured necessary), (b) masked categorical policies, (c) two seats per environment sharing one network through the own-view observation, (d) an opponent sampler. MLP of two hidden layers (128 units) with a policy head over 46 logits and a value head. Dependencies: `torch` as an optional extra (`rl`), never imported by `lanegame.env`.

### F.2 Staged approach

1. **M3a, supervised by the exact game.** Train against each fixed scripted opponent on the siege model at K = 16 (4,951 nodes, solved in six seconds). Oracle: `best_response_value` gives the exact optimum; the wrapped network gives its exact value. Pass criterion to be recorded, not assumed: report the fraction of the exact best-response gain achieved per opponent and per budget. Also report exact exploitability of the learned policy (expected to be high: a best response to one opponent is exploitable by others), which demonstrates the measurement before it is needed.
2. **M3b, self-play with a pool.** Opponents drawn from the latest checkpoint (probability 0.5), earlier checkpoints (0.3) and the scripted suite (0.2); the numbers are a starting point. Symmetric skills. Record learning curves, the metric suite and exact exploitability on the legacy configuration as a side check (train one run on the legacy K = 16 game in self-play; its exploitability must fall over training).
3. **M3c, evaluation harness.** Sections F.3 and F.4.
4. **Opponent populations proper.** If self-play cycles (visible as oscillating head-to-head results between checkpoints), move to a fixed-population scheme: keep every checkpoint, sample uniformly or by recent performance (a fictitious-self-play flavour).
5. **M3d, PSRO.** Oracle = the M3a trainer against a fixed mixture; meta-solver = `solve_matrix_game` on the empirical payoff matrix of the population (zero-sum, so the meta-strategy is a Nash of the restricted game with a certified gap). Start on the champion game at K = 16 or 24 so each oracle step is cheap.

### F.3 Metrics (all computed from `info`, all with confidence intervals)

- Relative resources per game (the objective), and win probability defined as `P(total > 0)`.
- Farm and plate income per side; farm missed (available minus collected).
- Trades attempted, connected, damage dealt and taken; HP at the end and HP time-series.
- Deaths, recalls, turns absent; ganks suffered; gank exposure (sum of exposure cells, time at `+T`).
- Position histogram; action frequencies by behaviour, wave and objective; vision turns; siege turns.
- Performance against each scripted opponent and against the frozen self-play policies of earlier runs.
- Approximate exploitability: the value a PPO best response trained for a fixed budget achieves against the frozen policy (report the budget).
- Exact exploitability and exact head-to-head values on legacy configurations (K ≤ 16), and on tiny champion configurations if M2e lands.
- Variance: standard errors from at least 2,000 evaluation episodes per cell; three training seeds per setting; common random numbers when comparing policies.

### F.4 What is and is not claimed

On the champion game no Nash guarantee is claimed. Claims take the form "policy X beats the scripted suite by Y ± Z, loses at most W ± Z to a best response trained with budget B, and on the exactly solvable sub-configuration its exploitability is E". Where an experiment can be run on the exact game (K = 16 legacy, tiny champion configurations) it should be, because that is where "mathematically defensible" is literally true.

---

## G. Economic progression plan (M4)

> **Superseded by section O and [m4_integrated_design.md](m4/m4_1/m4_integrated_design.md).** The three levels, the two item
> tracks with hit-chance bonuses, the per-turn plate removal (`plates_fall_turn = 24`) and the 32-turn calibration
> targets below are no longer the plan.

### G.1 Experience and levels (M4a)

- XP source: enemy minion HP that dies within `xp_range = 1` cell of a *present* champion, read from `TurnResult.kills` (already keyed by cell), times `xp_per_hp`. Last-hit gold is unchanged, so a zoned player can collect XP but not gold, and a player far away collects neither, as the brief asks.
- Levels: three (1, 2, 3) at cumulative thresholds (`xp_levels = (0, 8, 20)` in wave-HP units, to be set so level 3 is reachable around the middle of a 32-turn game under equal play). Effect: the `λ` term of the trade formula (`λ = 0.10` per level of difference) and nothing else by default. Open: `hp_max` by level (6 / 7 / 8), minion damage by level.
- State: `xp` (integer) and `level` (derived; store `xp` only). Observation adds both sides' level and XP fraction.

### G.2 Spendable gold and items (M4b)

- Ledger: `gold_i = earned_i − spent_i`, with `earned` the cumulative reward components (farm, plates, bounty). Spending never changes the reward (section C.2, question 10).
- Items: two tracks (offence, defence), two tiers each (`item_cost = (4, 7)` resources per tier). Offence adds `ι_off = 0.08` per tier to one's hit chance and, at tier 2, +1 thinning per Apply Pressure turn; defence subtracts `ι_def = 0.08` per tier from the enemy's hit chance and gives a `block = 0.15` per tier chance to reduce incoming trade or gank damage by one. Open: a single combined track instead (simpler, fewer strategic archetypes).
- When: on absence turns only (recall or death). The absent player's legal set becomes `{ABSENT, BUY_OFF, BUY_DEF}` filtered by affordability on the last absent turn before return; this reuses the turn-step API and keeps the decision simultaneous-safe (nothing the opponent does that turn depends on it).
- Objective: unchanged. The question the design asks is whether a recall that costs two turns of farm pays back through a stronger lane presence; that answer must be learned, not written into the reward. Terminal value of items and unspent gold is a knob defaulting to zero.

### G.3 Tower plate durability (M4c)

- State per side: `plate_damage` dealt to the enemy tower (integer, persistent).
- Plates: `plate_count = 3` per tower with durabilities `plate_durability = (2, 3, 4)` hits (increasing); Apply Pressure deals one hit per turn (two with offence tier 2); crossing a threshold pays `plate_value[i]` (`(1.0, 1.0, 1.5)` proposed) once. After `plates_fall_turn = 24` remaining plates are removed without payment, creating the timing pressure the brief describes.
- Replaces the per-crash opportunities in champion configurations (`plate_mode="durability"`); `plate_mode="crash"` stays for the exact models. Thinning is unchanged. The siege study's reason for rejecting a persistent plate counter (state explosion under exact enumeration) does not apply to the RL environment.
- Open: whether partial damage decays when the siege ends (default no: the persistence is the point); whether the defender can repair (default no).

### G.4 Reward accounting, once more

Earned resources are the only reward. Spending, item power, levels, HP and tower damage are state. Plate rewards are paid once at thresholds. A terminal valuation, if ever introduced, must be symmetric and documented as a change of objective. Under these rules the expected per-turn reward of the two players sums to zero exactly, which the environment tests check.

---

## H. Opponent modelling research (M5)

### H.1 Two arenas

Run every experiment first on the **exact arena** (siege model, K = 16, or the previous revision at K = 32 if a longer horizon matters): opponent types are `Policy` objects, the posterior is exact, best responses and exploitability are exact, and a belief-augmented best response is enumerable for a handful of types. Then repeat on the **champion arena** with the M3 agents, where the same quantities are estimated. The exact arena is cheap (seconds per solve) and is where the quantitative claims are tightest.

### H.2 Type library

Parameterised scripted families (each parameter a tendency the brief lists): trade propensity, Hold-versus-Yield preference, shove frequency, freeze preference, pressure greed, recall HP threshold, vision-first. Plus ε-mixtures of each with the robust policy, plus frozen learned checkpoints. Types are Markov policies, so the likelihood of an observed action at (k, state) is exact.

### H.3 Inference

Finite type set with a prior; Bayes update per turn from the observed opponent action; a forgetting factor for strategy shifts; posterior entropy as the confidence measure. The combat randomness never enters the likelihood (actions are observed exactly), which is the formal separation between uncertainty about the opponent (epistemic) and randomness of trades and ganks (aleatoric) that the brief asks for. In the champion arena the agent sees the public action history in `info`; the modelling code lives outside the agent's Markov observation.

### H.4 Policies to compare

1. Robust: the exact equilibrium (exact arena) or the self-play policy (champion arena).
2. Best response to a known type: exact (`best_response_value`) or PPO fine-tuned against the fixed type.
3. Bayesian adaptive, two versions: (a) switch to the best response of the maximum-a-posteriori type once its posterior exceeds a threshold τ, otherwise play robust; (b) a belief-conditioned policy trained against the type prior with the posterior appended to the observation.
4. Risk-constrained exploitation: in the exact arena, the restricted Nash response (the opponent plays the model with probability p and is free otherwise; a small solver extension that fixes a component of one player's mixture), sweeping p; in the champion arena, deviation only when the approximate exploitability of the exploiting policy stays below a budget ε.

### H.5 Experiments and metrics

- Exploitation gain versus confidence: expected value against the true type as a function of τ (or ε), with the posterior entropy at the deviation time.
- Severity of being wrong: value when the true type is not the inferred one, and when it is outside the library (misspecification), and when it switches at a fixed turn.
- Skill asymmetry: the same grid at `m_opp ∈ {0.2, 0.5, 0.8}`; does a mechanically stronger opponent make exploitation more or less attractive?
- Noise: scale trade and gank randomness through the configuration and measure how much of the outcome variance is aleatoric (common random numbers across policies isolate it) versus epistemic (variance across types).
- Metrics: mean and CVaR of relative resources; regret against the oracle best response; worst case over the type set; time to identify (turns until the posterior passes τ); frequency and cost of wrong deviations.

---

## I. Performance and development budget

### I.1 Measured baseline and derived expectations

| Item | Value | Status |
|---|---|---|
| Lane turn (uncached) | 40 µs | measured |
| K = 32 episode, scripted policies | 1.3 to 2.1 ms | measured |
| Exact solve K = 8 / 12 / 16 (siege) | 0.2 / 1.2 / 5.6 s | measured |
| One million environment turns, legacy game | about 40 to 70 s single-threaded | derived from the two lines above |
| Champion layer overhead | to be measured in M2e; expectation: under 2× | derived (two draws and a position update per side) |
| PPO wall-clock | to be measured in M3a; expectation: network forward passes dominate, CPU sufficient for a 2×128 MLP | derived; no number is claimed |
| Exact K = 32 siege solve | 3.5 min, 3 GB | measured (siege study); not to be rerun in any milestone |

### I.2 Rules that keep milestones short

- **No exact solve above K = 16 in tests or milestone work.** The frozen `results/siege/solution_K32.json` is the K = 32 equilibrium for any evaluation that needs it (`io.read_solution` reconstructs the `TabularPolicy`).
- **Correctness tests and research sweeps live in different places.** Tests prove rules on small configurations; `experiments/` scripts produce numbers into `results/` with a JSON and a Markdown file each, as today, and are never run by the test suite.
- **Fast loop.** After M0, development runs `pytest -m "not slow"` (seconds); the full suite runs once at the end of a milestone.
- **Benchmark before training.** M1 prints throughput in `env-demo`; M2e records it; M3a records training wall-clock per million steps before any self-play is scheduled.
- **Small first, then scale.** K = 16 and small opponent sets until a result is reproducible across three seeds; champion game at K = 32 only in M3b and later.
- **Cache reusable results.** Checkpoints, tournament matrices, posteriors and solution files are written once and read by later experiments; experiments declare their inputs.
- **Memory.** Nothing in the roadmap needs more than a few hundred MB except the (excluded) K = 32 solve.

### I.3 Four kinds of cost, kept apart

- **Development time** (sessions): M0 S, M1 M, M2 L as five S/M tasks, M3 L as four M tasks, M4 L as three S/M tasks, M5 L, M6 L.
- **Simulator runtime**: negligible for M1 and M2 (seconds per thousand episodes, measured); the M2e benchmark decides whether any vectorisation is ever worth doing.
- **Training compute**: unknown until M3a; the plan is designed so that the first number is produced by a small run on the exact game, not by a long self-play run.
- **Model and token usage** (Claude Code sessions): bounded by scope. Each milestone touches a listed set of files, has its own tests, and does not require reading or rerunning the three studies. Research write-ups are produced by experiment scripts, not by hand.

---

## J. Immediate next task: Milestone 1, RL-ready environment

This section is written as the brief for the next Claude Code session. It is deliberately limited to the existing game (no champion mechanics).

### J.1 Goal

Add `lanegame/env/`, a simultaneous-move environment around the existing siege / revised / v1 games, such that two independently controlled scripted agents play complete, seeded, reproducible episodes, and such that any agent whose action depends only on the observation can be evaluated exactly by the existing solver.

### J.2 Modules and public API

`lanegame/env/core.py`

```python
@dataclass(frozen=True)
class Observation:
    k: int                      # decision index
    side: int                   # 0 = A, 1 = B
    state: State                # the lane state (shared, perfect information)
    vector: np.ndarray          # own-perspective float32 features, fixed length
    mask: np.ndarray            # bool, length = env.n_actions, True = legal

@dataclass
class StepResult:
    obs: Tuple[Observation, Observation]
    rewards: Tuple[float, float]        # (r, -r), realized payoff to A this turn and its negative
    terminated: bool                    # k reached cfg.horizon
    truncated: bool                     # reserved (max_turns < horizon); False by default
    info: dict                          # decomposition and events (see J.4)

class LaneEnv:
    def __init__(self, cfg: GameConfig, max_turns: int | None = None): ...
    n_actions: int                      # 18 (positioning on), 3 (v1); = max(cfg.action_codes()) + 1
    obs_size: int
    def reset(self, seed: int | None = None, root: State | None = None, k0: int = 0) -> Tuple[Observation, Observation]: ...
    def step(self, action_a: int, action_b: int) -> StepResult: ...
    def legal(self, side: int) -> Tuple[int, ...]: ...
    @property
    def k(self) -> int: ...
    @property
    def state(self) -> State: ...
```

Rules: `step` raises `ValueError` on an illegal action or when called after termination; the chance generator is `np.random.default_rng(seed)` created in `reset` and used only for gank draws (and, later, trades); the environment holds no agent randomness. Rewards are the realized `farm_payoff + plate_payoff + gank_loss·(g_B − g_A) + crash choice payoff`, exactly the per-turn quantities `simulate.play_episode` accumulates. The crash-choice stage of the `v2` / `v1` models is handled inside `step` with the `greedy_resources` default unless the agents provide a chooser (keep `v2` support minimal: the greedy default plus an optional `choose` hook on the `Agent`).

`lanegame/env/observation.py`

```python
def observe(state: State, k: int, side: int, cfg: GameConfig) -> np.ndarray
def obs_size(cfg: GameConfig) -> int
def legal_mask(state: State, side: int, cfg: GameConfig) -> np.ndarray
```

Features (own perspective, in this order): `k / K`, `(K − k) / K`, one-hot of `k mod cfg.phase_period`; lane grid of `2·base + 1` cells with own minion HP per cell and enemy minion HP per cell (fronts and rears, divided by `cfg.hp_cap`); own and enemy: siege mass / `siege_cap`, vision timer / `vision_turns`, pending plates / `plates_cap`, siege-eligible flag, has-front flag. `observe` must satisfy `observe(s, k, 1, cfg) == observe(s.mirror(), k, 0, cfg)`.

`lanegame/env/agents.py`

```python
class Agent(Protocol):
    def act(self, obs: Observation, rng: np.random.Generator) -> int: ...

class ScriptedAgent:          # wraps a lanegame.policy.Policy; samples from policy.move(k, state, side, legal) with rng
class PolicyFromAgent(Policy): # wraps an Agent whose act is deterministic given obs -> exact evaluation
def play_episode(env, agent_a, agent_b, seed, record=False) -> EpisodeLog
def run_episodes(env, agent_a, agent_b, episodes, seed) -> EpisodeStats
```

Agent randomness comes from a generator the caller seeds (`np.random.SeedSequence(seed).spawn(3)` → environment, agent A, agent B), so swapping one agent does not change the chance draws of the other or of the environment.

`lanegame/env/trace.py`: `EpisodeLog` (per-turn records: k, state list, actions and labels, legal sets, rewards, decomposition, gank draws) and `write_trace(log, path)` with `{"schema": "lanegame.episode", "version": 1, "config": ..., "turns": [...]}`.

`lanegame/env/pettingzoo_adapter.py`: `class LaneParallelEnv(pettingzoo.ParallelEnv)` with agents `["A", "B"]`, `Discrete(n_actions)` action spaces, `Dict({"observation": Box, "action_mask": MultiBinary})` observation spaces, `reset(seed, options)` and `step(actions)`; imports `pettingzoo` and `gymnasium` lazily and raises a clear `ImportError` otherwise.

CLI: `python -m lanegame env-demo [--model siege|revised|v1] [--a aggressive] [--b passive_safe] [--episodes 200] [--seed 0] [--horizon 32]` prints `EpisodeStats` (mean ± standard error, ganks, farm, plates, action frequencies) and episodes per second.

### J.3 Packaging

`pyproject.toml`: add `lanegame.env` to `packages`; optional extras `rl = ["pettingzoo>=1.24", "gymnasium>=0.29"]` (PyTorch comes with M3). `requirements.txt` unchanged. README: one layout line and one quick-start line.

### J.4 `info` contents per step

`farm (A, B)`, `farm_available (A, B)`, `plates (A, B)`, `gank_p (A, B)`, `ganked (A, B)` booleans, `crash_hp`, `exposure`, `objectives`, `vision_set`, `thinned`, `legal (A, B)`, `actions (a, b)`, `action_labels`. Everything a later dashboard or metric needs for the legacy game.

### J.5 Tests (`tests/test_env.py`, using the existing session fixtures)

1. `test_mask_equals_legal_actions` on every reachable (k, state) of the three K = 16 fixtures.
2. `test_observation_is_markov_and_mirror_equivariant`: equal states give equal vectors; `observe(s, k, B) == observe(s.mirror(), k, A)`; vector length equals `obs_size(cfg)`; all features in [0, 1].
3. `test_env_matches_exact_expectation`: for five policy pairs per model (reuse `tests/test_simulation.PAIRS`), the mean of 3,000 episodes is within 4.5 standard errors of `evaluate(game, layers, pa, pb)`.
4. `test_env_without_randomness_equals_exact_value`: with every gank parameter zero, every episode total equals `evaluate` to 1e-12 and the farm and plate sums equal `path_statistics`.
5. `test_env_is_reproducible`: same seed, same trajectories (actions, rewards, gank draws); different seed differs; replacing agent B's seed leaves the environment's gank draws against identical action sequences unchanged.
6. `test_step_rejects_illegal_actions_and_after_termination`.
7. `test_rewards_are_zero_sum_and_sum_to_episode_payoff`.
8. `test_policy_from_agent_is_exactly_evaluable`: a deterministic agent (argmax of a fixed scoring of the observation) wrapped as a `Policy` has an exact value and exploitability from `evaluate` / `exploitability`, and 2,000 environment episodes agree with the exact value within 4.5 standard errors.
9. `test_simultaneity_by_construction`: the observation objects returned before `step` do not change when the other agent's action changes (compare `obs` for B across two different `a` choices from the same state: identical), and no public method accepts one action without the other.
10. `test_pettingzoo_api` (skipped when the package is missing): `pettingzoo.test.parallel_api_test(LaneParallelEnv(cfg), num_cycles=50)`.
11. `test_env_demo_cli_runs` with 5 episodes.

### J.6 Acceptance criteria

- All existing 305 tests pass unchanged; the new tests pass; nothing under `mechanics.py`, `game.py`, `solver.py`, `state.py`, `config.py`, `experiments/` or `results/` is modified.
- `python -m lanegame env-demo --a aggressive --b passive_safe --episodes 200 --seed 0` runs in seconds and prints throughput; the number is recorded in the session summary.
- A one-paragraph note in `docs/rl_research_roadmap.md` (this file, a new section "M1 outcome") records the measured throughput, the observation size, and any deviation from this specification.

### J.7 Out of scope for M1

Champion mechanics of any kind; changes to the action encoding; vectorised or batched environments; neural networks; Gymnasium spaces outside the adapter; the dashboard; any experiment script; any K > 16 exact solve.

---

## K. Summary

1. **Architecture.** Shared simulator = the existing deterministic lane mechanics plus a new champion layer; one `GameConfig` with champion fields defaulting to off; a NumPy-only simultaneous-move environment (`lanegame/env/`) over both; the exact modules untouched and used as the benchmark through a Markov-observation bridge; PyTorch training code (`lanegame/rl/`) that depends only on the environment; a thin PettingZoo adapter for tooling.
2. **Sequence.** M0 test markers (optional) → M1 environment → M2a position and behaviours → M2b recall and absence → M2c trading and skill → M2d ganks as events and death → M2e validation and optional exact micro-benchmark → M3a PPO versus scripted opponents on the exact game → M3b self-play with a pool on the champion game → M3c evaluation harness → M3d PSRO (optional) → M4a XP → M4b gold and items → M4c plate durability → M5 opponent modelling (exact arena first) → M6 dashboard.
3. **Reused as is.** `GameConfig`, `State` and its mirror, `step_turn` and `legal_actions`, `gank_probability` and `collection_efficiency`, the matrix-game solver, backward induction, best responses, exploitability, path statistics, the `Policy` interface and scripted baselines, the JSON conventions, reachability and scenarios, the K = 16 test fixtures, the frozen K = 32 solution.
4. **Unresolved design decisions** (E.11 and the open items of E.4 to E.8, G, H): champion speed; whether Yield may carry siege objectives; recall timing and legality line; the absent side's wave behaviour; kill bounty and regeneration (both proposed off); skill in gank damage; single versus two item tracks; plate thresholds and the late-game removal turn; whether to build the chance-node exact game at all; the equilibrium-selection rule for reporting (inherited from the earlier reports).
5. **Complexity and compute risks.** The champion state multiplies the lane state by about 10⁴, so exact enumeration is limited to tiny configurations by design. The simulator is fast enough (25,000 turns per second measured) that training cost will be set by network inference, to be measured in M3a before any long run. The one place where the certified core is touched (optional chance nodes) is guarded by the pinned values of all three models. The trade and gank formulas are the main modelling risk: they must be reported at their defaults, not tuned for interesting curves.
6. **Smallest useful next task.** Milestone 1 as specified in section J: the environment, scripted agents, the exact bridge and its tests, with no new mechanics.
7. **Location of this document.** `docs/rl_research_roadmap.md`.

---

## L. M1 outcome (implemented 2026-10-11)

**Status.** Milestone 1 is implemented and verified against the exact game. No file under `mechanics.py`, `game.py`, `solver.py`, `state.py`, `config.py`, `experiments/` or `results/` was touched; the existing 305 tests pass unchanged alongside 65 new ones (370 in total, 7 min 44 s on this machine).

### L.1 Files

| Added | Purpose |
|---|---|
| `lanegame/env/core.py` | `LaneEnv`, `Observation`, `StepResult`, `Chooser`, `greedy_chooser` |
| `lanegame/env/observation.py` | `observe`, `legal_mask`, `obs_size`, `n_actions`, `feature_names` |
| `lanegame/env/agents.py` | `Agent` protocol, `ScriptedAgent`, `GreedyObservationAgent`, `AgentPolicy`, `as_policy`, `play_episode`, `run_episodes`, `episode_streams`, `agent_chooser`, `sample` |
| `lanegame/env/trace.py` | `EpisodeLog`, `EpisodeStats`, `trace_dict`, `write_trace`, `read_trace` (schema `lanegame.episode`, version 1) |
| `lanegame/env/pettingzoo_adapter.py` | `LaneParallelEnv` (optional: pettingzoo, gymnasium) |
| `lanegame/env/__init__.py` | exports |
| `tests/test_env.py` | 65 tests (sections A to J of the brief, plus traces) |

| Changed | Change |
|---|---|
| `lanegame/__main__.py` | `env-demo` subcommand (`--model --a --b --episodes --seed --horizon --exact --trace --cache`) |
| `pyproject.toml` | package `lanegame.env`; optional extra `rl = [pettingzoo, gymnasium]` |
| `README.md` | roadmap pointer, quick-start line, layout entry |

### L.2 Public API

```python
env = LaneEnv(cfg, max_turns=None, choosers=None, cache_size=0)
obs_a, obs_b = env.reset(seed=42, root=None, k0=0)       # Observation(k, side, state, legal, vector, mask)
result = env.step(action_a, action_b)                    # StepResult(obs, rewards=(r, -r), terminated, truncated, info)
env.legal(side); env.masks(); env.observe(); env.k; env.state; env.done; env.n_actions; env.obs_size

observe(state, k, side, cfg) -> float32[obs_size]        # pure function; observe(s, k, 1) == observe(s.mirror(), k, 0)
legal_mask(state, side, cfg) -> bool[n_actions]          # True exactly on mechanics.legal_actions

ScriptedAgent(policy)            # any lanegame Policy; distribution() = policy.move, choose() = policy.choose
GreedyObservationAgent(W, b)     # deterministic argmax of a linear score; one-hot distribution
as_policy(agent, cfg) -> Policy  # exact bridge; TypeError for agents without an explicit distribution
play_episode(env, agent_a, agent_b, seed, record=False) -> EpisodeLog
run_episodes(env, agent_a, agent_b, episodes, seed) -> EpisodeStats   # mean, stderr, farm, plates, ganks, action frequencies, throughput
write_trace(log, cfg, path) / read_trace(path)
LaneParallelEnv(cfg)             # PettingZoo ParallelEnv; obs = {"observation", "action_mask"}
```

Reward per turn (payoff to A; B receives the negative): farm difference + plate difference + `gank_loss` × (B ganked − A ganked) + crash-reward difference, with the two gank checks realized from the environment's own generator. `info` carries every component, the gank probabilities and uniform draws, crash sizes, options and choices, exposure, objectives, vision, thinning, the engine's event counters, and the state before the turn, before the choice stage and after.

### L.3 Sizes and measured throughput (single thread, this machine)

| Configuration | obs_size | n_actions | legal per turn |
|---|---|---|---|
| siege (`GameConfig()`) | 34 | 18 | 4 to 6 |
| farming revision (`v2`) | 34 | 6 | 4 to 6 |
| original (`v1`) | 32 | 3 | 2 to 3 |

| Measurement | Value |
|---|---|
| `env-demo`, aggressive vs passive_safe, K = 32, 200 episodes | 245 episodes/s, 7,800 turns/s (no cache) |
| same with `--cache 20000` (bounded LRU of lane steps) | 368 episodes/s, 11,800 turns/s |
| uniform vs uniform, K = 32 | 230 episodes/s, 7,400 turns/s (8,200 with cache) |
| K = 16 scripted pairs, 2,000 episodes | about 490 episodes/s, 7,800 turns/s |
| raw `step_turn` + `legal_actions` (reference) | about 25,000 turns/s |
| old `simulate.play_episode` (reference) | 15,000 to 24,000 turns/s |

The wrapper costs about 90 µs per turn on top of the 40 µs lane step: two observations and masks, the legal sets, the uncached farm and gank computation (which `LaneGame.transition` cached), the `info` record and the log. Profiling found no single dominant item after two cheap fixes (inverse-CDF sampling instead of `Generator.choice`, list-built observations). The bounded cache recovers about half of the gap for repetitive scripted play; it is off by default so memory stays predictable. Further optimisation is deferred until M3a measures whether the environment or the network dominates.

### L.4 Tests (`tests/test_env.py`, 65 items, about 100 s including the three K = 16 fixture solves)

A. The mask equals the engine's legal set at every reachable node and through the environment; illegal actions, out-of-range codes and stepping outside an episode raise. B. Observations are float32, in [0, 1], deterministic, mirror-equivariant at every reachable node, encode the state (spot checks), and never depend on the current turn's actions. C. Same seed gives identical trajectories; the chance stream is independent of the actions played and of the agents' draws (common random numbers); the agents' streams are independent of each other. D. Rewards are zero-sum and decompose exactly; the deterministic parts of a step equal `LaneGame.transition` with and without the cache. E. The mean of 2,000 episodes agrees with the exact expected value within 4.5 standard errors for five policy pairs per model, equilibrium policies included. F. With ganks off and deterministic pairs every episode equals the exact value and records the exact path's farm, plates and crash resources. G. `ScriptedAgent` round-trips to its policy; a deterministic observation agent has a valid one-hot distribution at every node equal to its environment action, an exact value that the environment reproduces, a non-negative exact exploitability, and earns at most v* against the equilibrium; a sampling-only agent is refused by `as_policy` but plays; in the "choice" mode the agents' own crash choices are used (a vision-taking shover from a pre-crash root is reproduced exactly) and the greedy fallback is reported. H. Termination, truncation, reset, roots and k0, invalid `max_turns`. Traces round-trip through JSON with chained states and consistent accounting. I. `pettingzoo.test.parallel_api_test` passes for all three configurations (skipped if the package is missing). J. The CLI runs with and without `--exact`.

### L.5 Deviations from section J, and known limitations

- `Observation` carries the legal tuple next to the mask (agents need it; it is the engine's set, not a second rule).
- The crash-choice stage of the older configurations is resolved through `choosers` callbacks wired to the agents' `choose` methods rather than by a greedy default, as the M1 brief required; `info["choice_default_used"]` marks trajectories in which the default was used. Through the PettingZoo adapter the default is used unless `choosers` are passed (the Parallel API has no second stage). The siege model has no such stage.
- Agents sample with an inverse-CDF `sample()` rather than `Generator.choice`, so the environment does not consume random numbers in the same order as `simulate.play_episode`; agreement with it is statistical, agreement with the exact solver is what the tests check.
- `n_actions = max(cfg.action_codes()) + 1` (18 / 6 / 3), not a constant 18.
- The exact-value test uses 2,000 episodes per pair instead of 3,000.
- `seed=None` gives fresh entropy (non-reproducible by design); the exact bridge requires an explicit `distribution`; a grid cell is clipped at 1 when two packets of one side share a cell beyond `hp_cap` (possible near a base only).
- Not done, by scope: champion mechanics, vectorised environments, any learning.
- The optional packages `pettingzoo` and `gymnasium` were installed into the project's `.venv` to run the adapter test; the core environment does not import them.

### L.6 Reproduction

```bash
python -m lanegame env-demo --a aggressive --b passive_safe --episodes 200 --seed 42          # siege, K = 32
python -m lanegame env-demo --a stack_builder --b freeze_contest --horizon 16 --episodes 2000 --seed 42 --exact
python -m lanegame env-demo --model v1 --a always_shove --b defensive_freeze --horizon 16 --episodes 2000 --seed 7 --exact
python -m lanegame env-demo --episodes 500 --seed 42 --cache 20000 --trace results/env_trace_demo.json
python -m pytest tests/test_env.py -q            # the new tests (~100 s)
python -m pytest -q                              # everything (370 tests)
pip install "pettingzoo>=1.24" "gymnasium>=0.29" # optional, for the adapter test (skipped otherwise)
```

---

## M. M2a Outcome — Champion State, Positioning, and Behaviors (implemented 2026-10-11)

**Status.** Champion mode exists as an opt-in configuration, `GameConfig.champions()`, played through `lanegame.env`. HP, absence, Recall and skill effects are dormant as planned. `GameConfig()`, `GameConfig.v2()` and `GameConfig.v1()` are unchanged: no default changed, the lane step's new parameter defaults to the legacy rule, and the existing tests, including the reproduction tests of all three reports, pass. The exact solver was not touched.

### M.1 Files

| File | Change |
|---|---|
| `lanegame/champions.py` | **new**: champion state, legality, the turn resolution, farming and gank terms |
| `lanegame/config.py` | champion fields (all inert unless `champion_mode`), `GameConfig.champions()`, the champion action encoding and labels, validation |
| `lanegame/mechanics.py` | `step_turn(..., champion_reach=None)`: optional per-side champion contribution line; `None` is the legacy rule |
| `lanegame/env/core.py`, `observation.py`, `agents.py`, `trace.py`, `pettingzoo_adapter.py`, `__init__.py` | champion state in the environment, observations, masks, champion agents, trace fields, adapter option |
| `lanegame/__main__.py` | `env-demo --model champions` |
| `tests/test_champions.py` | **new**: 25 tests |

### M.2 State and actions

Per side: `pos` (own coordinates, −2…+2, start −2), `hp` (6, constant), `absent` (0). Per seat: `skill_a`, `skill_b` in the configuration (default 0.5, no effect yet). `ChampState.mirror()` swaps sides, and `swap_seats(cfg)` swaps the skills. The lane `State` is unchanged, and the environment holds both.

Action code `a = wave + 3·behavior + 15·objective`, with behaviors Hold 0, Yield 1, Trade 2, Safe 3, Recall 4 (reserved), and `ABSENT = 45`. That gives 46 slots. The legal sets are:

- The wave component follows the lane's own rule; Freeze is state-dependent as before.
- Hold, Yield, Trade and Safe are always legal without an objective.
- Pressure and Vision are legal only with Hold or Trade, and only when the side's crashed mass is under the enemy tower **and** its champion can reach that tower this turn (`pos + champion_speed ≥ 2`).
- Recall and ABSENT are never legal. Passing them to the resolver raises `NotImplementedError`.

There are 8 to 12 legal actions without an eligible siege and 16 with one.

### M.3 Turn resolution (`champions.resolve_turn`)

1. **Wave destinations.** The lane is probed once with full champion support. Each side's farm spot is its leading wave's end cell, clipped to ±2.
2. **Targets.** An objective targets +2. Hold and Yield target the farm spot. Safe targets min(farm spot, −1). A Trade without an objective then chases its opponent's intended cell (its start cell if the opponent also trades). It stays put if already in contact, otherwise goes one cell short, and dives into the tower cell if that is where the opponent will be. Movement is at most `champion_speed = 2` cells per turn.
3. **Evasion.** A Yield or Safe champion that would end in contact with an enemy trader retreats to the furthest-forward cell out of contact, never ahead of its own target. Hold never evades.
4. **Contact.** Champions are in contact within `trade_range = 1`, except that a champion in its own tower cell can be reached only from inside that cell. A Trade connects on contact. A connected trade suspends that trader's siege objective for the turn.
5. **Modes.**
   - *farm*: Hold, unthreatened Yield, unconnected Trade. Contributes to fights up to end + 1.
   - *trading*: a connected Trade. No contribution, and its collection is multiplied by `trade_farm_factor = 0.5`.
   - *retreat*: a threatened Yield. Contributes only at or behind min(end, −1).
   - *safe*: Safe. Likewise.
6. **Lane step.** `step_turn` runs with the effective objectives and those contribution lines.
7. **Farming.** The legacy accounting applies, but a champion collects only kills within `farm_reach = 1` cell of its end cell. The legacy efficiency is used, with Contest for farm and trading and Safe for retreat and safe. The opponent zones a kill only when it is in farm or trading mode and within one cell of that kill.
8. **Exposure.** Exposure is max(start, end). The gank probability keeps the legacy form with a behavior term: Hold and Yield 0.02, Trade 0.05, Safe 0, plus 0.03 for Shove with a non-Safe behavior. Hold and Safe reproduce the legacy Contest and Safe terms exactly. The gank is still a resource loss, realized by the environment.

Siege compatibility follows from steps 2, 4 and 6. Plates, thinning and vision happen only when the champion stands at the enemy tower and is not trading. A crashed wave with a distant champion yields nothing beyond the plate opportunities the lane already records.

### M.4 Observations and masks

`obs_size` is 50: the 34 M1 features plus 8 per champion. Those are a one-hot of its cell in the observer's coordinates (5 cells), HP/6, absent/3, and skill. The mask has 46 slots and comes from `champion_legal_actions`, which builds on `mechanics.wave_actions` and `siege_eligible`. Legacy vectors and masks are unchanged.

The M1 grid-clipping concern does not arise in practice. Over every reachable state of the siege model (K = 12), the revised model (K = 16) and v1 (K = 16), no two packets of a side share a cell and the observation vector is injective within each turn.

### M.5 Behavior matchups (scripted, 200 seeded episodes each, K = 32, payoff to A)

| A vs B | payoff | farm A / B | mean exposure A / B | trades connected (A) | B evaded |
|---|---|---|---|---|---|
| Hold vs Hold | +0.13 | 9.61 / 9.61 | −0.06 / −0.06 | 0 | 0 |
| Hold vs Yield | +0.12 | 9.61 / 9.61 | −0.06 / −0.06 | 0 | 0 |
| Hold vs Safe | +2.36 | 13.72 / 4.27 | −0.06 / −1.03 | 0 | 0 |
| Trade vs Hold | −9.95 | 4.80 / 9.61 | 0.00 / −0.06 | 31 | 0 |
| Trade vs Yield | +0.75 | 13.72 / 0.00 | 0.00 / −2.00 | 0 | 31 |
| Trade vs Safe | −2.35 | 13.72 / 0.00 | 0.00 / −2.00 | 0 | 31 |
| Trade vs Trade | +0.35 | 4.80 / 4.80 | 0.00 / 0.00 | 32 | 0 |

These rows confirm the design:

- An unthreatened Yield is exactly a Hold.
- Hold against Safe reproduces the legacy Contest-against-Safe economy to the cent (farm 13.725 / 4.27, payoff +2.36).
- A trade that connects halves the trader's farm.
- Yield and Safe escape a trader at the lane by giving up their farm.
- A trader whose target evades falls back to full farming and keeps its forward position.

Without HP damage (M2c), being traded costs nothing yet, so Trade is not attractive in these numbers. That is expected.

### M.6 Tests and throughput

- **Tests.** `tests/test_champions.py` has 25 tests and runs in about 1 s. It covers configuration and encoding, and the lane step reproducing the legacy step on every reachable K = 8 transition when given the legacy reach lines. It covers initial state, asymmetric skills, observation bounds and mirror equivariance, and mirror equivariance of the whole turn. It checks bounded and persistent movement and dormant HP, then movement and behavior: Hold following its wave, Trade approaching and connecting, Safe not teleporting, unthreatened Yield equal to Hold, threatened Yield retreating, Trade falling back to farming with its exposure kept, a connected Trade losing shove and farm, and tower protection with dives. Finally it covers farm reach and zoning, legacy-equivalent gank terms, masks equal to the legal sets with no reserved or redundant codes, the no-remote-siege rules including objective suspension, reserved mechanics raising, zero-sum and reproducible scripted episodes, the legacy-agent mapping, PettingZoo in champion mode, and the CLI. The 65 M1 tests pass unchanged, and the full suite passes: 395 tests in 6 min 33 s.
- **Throughput.** Champion mode runs 5,000 to 6,200 turns/s (158 to 193 K = 32 episodes/s) uncached, against 8,400 to 8,800 for M1 on the same machine. With `cache_size = 20000` it runs 7,500 to 11,600. The extra cost is the lane probe of step 1 plus the champion bookkeeping.

### M.7 Deviations from the M2a proposal, and decisions to revisit before M2b

1. **Hold, Yield and Safe follow the wave's end-of-turn cell.** Using the wave's start cell made champions lag one turn behind a walking wave and lose the first contact's farm and shove. The cost is one extra lane step per turn. The final wave can differ slightly from the probed one when a champion withdraws its contribution.
2. **Safe follows its wave up to the safe line instead of never moving forward.** The literal rule left a Safe champion at its tower for the whole game with no farm at all, unlike the legacy Safe it replaces.
3. **Safe also evades a credible trade.** The brief expects Trade against Safe to usually fail. Without evasion a Safe champion on the safe line was traded every turn by a trader at the lane.
4. **Tower-cell protection, with a dive as the only way in.** This is new; without it a tower-hugging defender was traded every turn from one cell outside its tower. Diving costs the trader full exposure and all lane farm. M2c's tower shot should apply exactly to divers.
5. **A trader chases its opponent's intended cell, not its start cell.** Chasing the start cell made champions pass each other and cycle every turn. Both behaviors are committed simultaneously, so this is resolution of committed actions, not observation.
6. **Hold and Trade without an objective stay legal during a siege.** They are no longer dominated by Pressure, because they keep the champion with its next wave instead of at the tower.
7. **Pressure or Vision with Trade is suspended when the trade connects.** This is the attention cost.

Open questions:

- Whether `trade_farm_factor = 0.5`, `farm_reach = 1`, the gank behavior terms and `champion_speed = 2` are the right defaults.
- Whether a threatened Yield should keep partial wave contribution.
- Whether Trade's chase should also be limited by minions (it currently ignores them).
- Whether Safe should drop its zoning effect on the opponent entirely. It currently zones nothing, as in the legacy model.

### M.8 Reproduction

```bash
python -m lanegame env-demo --model champions --a trade --b hold --episodes 200 --seed 42
python -m lanegame env-demo --model champions --a aggressive --b passive_safe    # legacy baselines mapped into champion actions
python -m pytest tests/test_champions.py -q      # 25 tests, ~1 s
python -m pytest tests/test_env.py -q            # M1 compatibility, ~100 s
python -m pytest -q                              # everything
```

---

## N. M2b–M3c outcome (implemented 2026-10-11)

M2b–M2e (Recall and absence, stochastic trading with mechanical skill, ganks with HP damage, death and respawn,
the validation gate) and M3a–M3c (masked PPO, frozen-snapshot self-play, evaluation with approximate best
responses) are implemented. The full write-up is [docs/rl/m2_m3_implementation_report.md](rl/m2_m3_implementation_report.md);
the running log is [docs/rl/overnight_implementation_log.md](rl/overnight_implementation_log.md). Decisions that differ
from sections E and F above:

- Recall is legal at `pos <= -1` with no state-based threat rule; an enemy Trade that reaches the recaller
  interrupts it (a state rule made Recall impossible whenever the opponent stood at the lane).
- Champion mode has its own gank-occurrence parameters with no unconditional base rate; gank damage is
  Binomial(4, p_e) (0-4 HP), its defensive reduction requires actually playing defensively; skill enters gank
  damage with a small term.
- The 2-HP trade chance uses the skill difference (defensive mitigation), not the attacker's skill alone.
- M2e changed three champion defaults on evidence: `kill_bounty` 0.5, `retreat_farm_reach` 2, defensive-play
  gank defense.
- The M3a calibration on the legacy game (K = 16, exact) shows PPO reaching 23-94% of the exact best-response
  gain (more with the optional, exact `gank_reward="expected"` variance reduction), and frozen-snapshot self-play
  that does NOT reduce exact exploitability: section F.2 step 2 should be followed by PSRO (step 5) before scaling.

---

## O. M4 amendment: integrated economy, wave dynamics and continuing time (design 2026-10-11)

The diagnostics study ([openai_five_diagnostics_report.md](rl/openai_five_diagnostics_report.md)) found limits in the
game design that PPO tuning cannot remove:

- Freeze is inert in every reachable state (§D.5).
- Recall restores HP only, at a high price.
- The 32-turn terminal horizon rewards a late Shove.
- Sieges are profitable but never visited in self-play.

M4 therefore replaces section G with **one integrated ruleset**, `GameConfig.m4()`. The legacy exact game and
`GameConfig.champions()` stay unchanged and pinned.

| Document | Content |
|---|---|
| [m4_integrated_design.md](m4/m4_1/m4_integrated_design.md) | the mathematical specification: state, resolution order, Freeze maintenance, gold and XP, levels 1–18, fixed-point HP and attack power, one item track with automatic purchase at base, persistent tower damage with three plates, continuing time with γ = 0.99, observation, parameters, invariants, ten worked scenarios |
| [m4_implementation_plan.md](m4/m4_2/m4_implementation_plan.md) | work packages WP0–WP10, the tests per package, integration tests, legacy pins, validation stages S0–S6, conditions before training |
| [m4_design_audit.md](m4/m4_1/m4_design_audit.md) | contradictions, exploits, redundancies, snowballing, parameters needing experiments, ranked alternatives |

Differences from section G:

- levels 1–18 instead of three;
- attack power and max HP scale with level and item tier, while hit probability depends on skill only;
- one item track instead of two;
- no plate removal at a fixed turn;
- no terminal turn at all.

**Next milestone:** M4.2 = WP0 + WP1 (ruleset scaffold, legacy pins, continuing time with segment PPO and discounted
evaluation), then WP2–WP7.

## P. M4.2 outcome: integrated implementation and pre-training validation (2026-10-11)

Branch `implementation/m4-integrated`. WP0–WP10 of the M4 plan are implemented as the separately versioned ruleset
`GameConfig.m4()`; every legacy configuration stays bit-identical (pins taken on the unchanged engine). Validation
S0–S4 is complete. Full account: [m4_implementation_report.md](m4/m4_2/m4_implementation_report.md); results:
`results/m4/`.

- S0 legacy pins and all 500 earlier tests pass; S1: 679 tests pass (2 strict expected failures record design
  findings), a 68 000-turn soak is clean, the observation is exactly decodable on the exhaustive reachable lane.
- S2: levels 2 / 6 / 11 / 18 at turns 4 / 19 / 58 / 144 (targets 4 / 20 / 59 / 147).
- S3: Freeze gate G1, G2 and the G3 rule pass on 163 reachable scenarios; a Shove + Hold breaker breaks a freeze later
  than the lane-only design table because M2's farm-spot rule walks it back when its probe predicts a crash.
- S4: no dominant scripted strategy, moderate snowballing; item tiers have almost no measurable value (item leads last
  16–22 of 400 turns), and siege pressure is strongly unprofitable for scripted agents.
- Throughput 6 444 turns/s uncached (8 818 cached), target 4 000.

**Next:** M4.3 (S5 PPO smoke and pilot, S6 research runs), after a review of design decision D4 (item prices and
tiers) in light of the S4 item finding.

## Q. M4.2b calibration: gank severity, item strength, the Shove → Recall cycle (2026-10-12)

Branch `calibration/m4-2b`. Exploratory staged calibration within a ~2M-turn budget; **M4 defaults unchanged**. Full
account: [m4_calibration_report.md](m4/m4_2/m4_calibration_report.md); results: `results/m4_calibration/`.

- **The shove penalty is lane farm, not ganks.** A presser loses −17 per 400 turns against hold_farmer, −16.2 of it in
  farm (crashed waves farmed by the defender at its tower; the parked configuration). Gank damage 0.60 → 0.40 halves
  forward deaths and is worth +1.26 [+1.12, +1.39] at W = 1,000; no sign changes.
- **Items:** zero value without trades; at 1.20 the first tier misses the level 6–11 breakpoint; in trading play leads
  last 7–11 turns. A higher `item_mult` raises combat value (and the aggressive trader's edge over passive play,
  +2.30 at 1.35) but does not make affordability matter to Recall.
- **The conditional Shove → Recall → Purchase agent** beats the presser by +15 but loses to passive and trading
  baselines; Shove before Recall was never significantly better than recalling directly.
- **The M2 farm-spot rule:** a keep-reach prototype (experiment only) reproduces the design's Freeze-break table in all
  64 cells and is worth +1.5 to pushing; proposed as a separate rule change.
- Fixes: lifecycle hooks (reset / notify) in the discounted evaluator; an explicit `allow_gamma_mismatch` opt-in.

**Recommended for review (not adopted):** `gank_damage_position = 0.40`, `item_mult = 1.35` (conservative: 0.40 /
1.20). The passive-equilibrium risk remains structural (crash farm, farm-spot rule, item persistence); decide these
before M4.3.

## R. M4.2c experiment: continuous gold investment (2026-10-12)

Branch `experiment/m4-continuous-investment`. An opt-in prototype, `GameConfig.m4_continuous(invest_alpha=…)`: every
return to base invests all unspent gold, AP(L, S) = AP(L, 0) [1 + α ((1 + S)^0.9 − 1)]. **`m4()` (two tiers) is
unchanged and remains the default.** Full account: [m4_continuous_investment_report.md](m4/m4_2/m4_continuous_investment_report.md);
results: `results/m4_continuous_investment/` (≈204k turns, 11 s wall).

- Mechanically sound: exact ledger without clamp, zero-sum rewards unaffected by spending, mirror symmetry, investment
  kept through death, strictly monotone fixed-point power (guarded at 10^6 resources), baseline throughput.
- Not yet strategically better than two tiers: no detectable Recall-timing value (scripted agents return every
  ~6–12 turns anyway); investment acts only through trades; duel odds stay breakpoint-quantized (knife edges, e.g.
  0.03 resources at L18). Leads make an `edge_trader` concede trades and die more, but pay the leader only against a
  passive opponent.
- Risk: unbounded growth (×4.3 at 120 resources; one 2-HP hit kills at S ≈ 68) within W = 1,000. The observation is
  exact only below 1,024 resources.

**Next (if pursued):** a curve-shape sweep (p 0.5–0.7 or log), the combat quantization review, and a gold-threshold
sweep of the recall agents before any PPO use.

## S. M4.2d-A readiness: wave bounce and Freeze breaking (2026-10-12)

Branch `readiness/m4-2d-wave`. Short deterministic diagnostic: [m4_wave_readiness_report.md](m4/m4_2/m4_wave_readiness_report.md);
results `results/m4_wave_readiness/` (≈26k turns, ≈12 s wall).

- **No bounce.** After a crash the crasher's follow-up waves park and stack in the defender's half (`stack_creep`)
  and re-crash every ~11 turns. Without a crash the contact point is neutrally stable at every cell. A push is
  permanent; only an opposing Shove moves the lane, and the shover then pays (≈0.11–0.13 farm per turn).
- Shove → Crash → Recall works: a Recall during a 9-HP crash is free, and a mistimed one costs ≈1 resource and
  8 XP. Bounce → Return does not exist.
- **Freeze-breaking artifact:** classified as a mechanical bug. It is fixed by the opt-in
  `GameConfig.m4(farm_spot_rule="pre_crash_front")`, which reproduces the 64-cell design table and meets the SHx
  gate. The default is unchanged.

**Recommendation: BLOCKED** until reviewers choose a restoring-force rule (report §4: proposal A, an unstacked Slow
Push front past the lane walks plus no parking during the own siege; or proposal B, the siege condition only) or
accept permanent pushes.


## T. M4.2d parallel emergent mechanics and integration (2026-10-12)

Four workstreams ran concurrently in separate worktrees from f504f5e. The results were integrated on
`integration/m4-emergent-ready` into one **opt-in** preset, `GameConfig.m4_emergent()`. **`m4()` and `m4_continuous()`
are unchanged.** Full account: [m4_emergent_parallel_report.md](m4/m4_2/m4_emergent_parallel_report.md); driver
`lanegame.experiments.m4.emergent_ready`; results `results/m4_emergent_ready/` (6,347 turns, 1.5 s wall).

| Workstream | Branch | Outcome |
|---|---|---|
| A natural waves | `prototype/m4-emergent-wave` | **Blocked, not integrated.** Waves walk freely and champions only add damage, but at wave HP 4 fronts never stack, so there are no crashes, plates or bounces and Shove is dominated. Unblocking needs a storage rule or a rescale (W = 8). |
| B positioning | `prototype/m4-positioning` | **Integrated** (`position_rule="standoff"`). Yield farms from a stand-off cell; a Yield that starts in contact with a Trade is pinned. Trades on a stand-off land 6 of 6 instead of 0. No action-space change. |
| C zoning | `prototype/m4-zoning` | **Integrated** (`zoning_rule="none"`). Removes the automatic proximity loss and adds no power term. Removing it alone does not create zoning. |
| D investment + observation | `prototype/m4-investment-readiness` | **Integrated.** Bounded saturating curve `1 + 1.5 (1 − sqrt(16/(16+S)))` (no one-shots, below ×2.5) and a compressed 88-entry observation, lossless through the training horizon. |

**Joint tests: mechanical feasibility.**

- **Waves (scripted):** stacking and crashing work, and the defender can counter them. There is no bounce.
- **Positioning:** a stronger champion can hold forward without attacking. The threat it poses scales with investment
  (549 → 967 expected centi-HP on a pinned stand-off).
- **Power matters:** A's Trade into a contesting B goes from −0.69 to +0.21 per 10 turns as the gap grows.
- **Hold beats Trade** once A has positional control.
- **Recall timing matters**, and returns invest all unspent gold.
- **Correctness passes:** mirror symmetry, masks, replay, gold/XP/minion accounting, invariants, lossless
  observations and zero-sum rewards. Tests: 230 passed, 3 skipped.

**Main caveat: threat-based concession is not rational at current payoffs.**

- Retreating costs the weaker champion more farm than being traded costs it HP.
- The deterministic flip needs A to hold ×3.4–4.0 B's attack power, but the safe investment curve stays below ×2.5.
- Other caveats: no bounce; a passive lane at equal power; weak late leads.

**Decision: GO WITH CAVEATS.** Freeze `m4_emergent()` and proceed to the M4.3 PPO smoke test. Before drawing strategic
conclusions from trained policies, reviewers decide whether to move trade damage or retreat farm retention. The
smallest check is a 2 × 2 rerun of the `zoning` and `hp` stages (about 10k turns).


## U. M4.2e-A natural wave mechanics (2026-10-12)

Branch `development/m4-natural-waves`, from `integration/m4-emergent-ready` 23e48be.

- **Preset:** **opt-in** `GameConfig.m4_natural()`. It is `m4_emergent()` plus `wave_dynamics="natural"` on an
  8-HP wave. `m4()`, `m4_continuous()` and `m4_emergent()` are unchanged, and the historical outputs are
  byte-identical.
- **Full account:** [m4_2e_natural_waves_report.md](natural_waves/m4_2e_natural_waves_report.md).
- **Driver:** `lanegame.experiments.m4.natural_waves`; results `results/m4_natural_waves/` (2.5k turns, 0.4 s).
- **Tests:** `tests/test_m4_natural_waves.py`.

**Rules**

- **One set of minion rules, whatever the actions.** Every front walks, engages on contact, merges and crashes. A
  defender's front engages the crashed mass under its tower; the idle siege wait (D7) is removed. Slow Push parking,
  creeping and the Freeze holds are removed too.
- **Wave slots become champion damage on enemy minions:** Hold 0, Thin 1, Clear 2 HP per tick (slots SP, F, SH). The
  damage needs presence and reach: a connected Trade, a Recall, absence or a Safe retreat deals none.
- **The 46-code action space and the 88-entry observation are unchanged.**

**Timescale fix**

- At 4 HP a contested front dies (2 ticks under Thin) before a reinforcement reaches it (3 ticks after spawning).
- At 8 HP it survives (4 ticks under Thin, 8 under Hold), so reinforcements merge and edges are stored.
- The per-HP economy is rescaled so a wave is still worth 1 resource and 8 XP: farm 0.125/HP, XP 1/HP, gold unit
  1e-7, crash cap 96.
- Geometry, spawn timing and tower damage are unchanged. Raising tower damage shrinks the defender's post-crash
  accumulation and halves plate hits.

**Results** (all from s0, deterministic, no seeded stacks)

- **Neutral:** symmetric perpetual contact at the lane.
- **Slow push:** one turn of Thin → a 24-HP stack → a 16-HP crash at k18.
- **Deliberate crash:** Clear on a built stack crashes at k11 with 19 HP, against k15 and 16 HP without Clear. Clear
  alone never crashes.
- **Bounce:** every crash is followed by the defender pushing into the attacker's half 14–17 turns later and crashing
  back with 16–25 HP. The push–bounce cycle lasts about 38–40 turns.
- **Freeze:** bounded Thin turns an enemy push into a wave held inside one's own tower cell. Holding keeps it for
  14–21 turns with a large farm edge (14.9 against 7.0). The opponent's Thin or Clear breaks it in one turn, with a
  crash.
- **Recall:** 0.46–1.35 resources depending on the enemy crash timing; Recalls after one's own crash are cheap.

**Budget:** ≈ 15.2k turns (target 10k, cap 20k). That includes four runs of the same deterministic driver and a 2k
timing benchmark; the unique experimental content is about 5.4k turns.

**Decision: PASS WITH CAVEATS.**

- The freeze is temporary.
- Clear needs a stack to crash.
- Pushing costs farm against a passive opponent: −2.0 to −4.7 per 40 turns.
- The evaluation bound is analytic (Ω = 107, against 68 enumerated for the scripted lane); an enumeration is
  deferred.
- One cell separates the lane from the tower.

**Next steps (not started):**

1. Verify combat-driven zoning on `m4_natural()`, in the freeze state and in the post-crash siege. Champion reach is
   now wave control, so zoning carries a non-gold cost.
2. Decide the push incentive (enemy-tower collection, plate value, ganks in the enemy half) alongside 1.
3. Enumerate, or accept, the natural lane's evaluation bound.
4. Integrate and freeze the game for the M4.3 PPO smoke test.

## V. M4.2e-A.1 finalizing natural wave mechanics (2026-10-12)

Branch `development/m4-natural-waves`, on top of M4.2e-A.

- **Full account:** [m4_2e_natural_waves_final_review.md](natural_waves/m4_2e_natural_waves_final_review.md).
- **Driver:** `lanegame.experiments.m4.natural_waves_final`; results `results/m4_natural_waves/natural_waves_final.json`
  (≈ 19k turns, 2 s).
- **Tests:** `tests/test_m4_natural_final.py`.
- **Presets:**
  - `GameConfig.m4_natural()` is now the final natural-wave preset.
  - The M4.2e-A prototype is kept as `GameConfig.m4_natural_prototype()`; section U and its report reproduce with it.
  - Every other preset is unchanged.

**Diagnosis**

- **The prototype's long sieges were an artefact.** A follow-up wave entering the enemy tower cell next to a defender
  did not crash; it fought there beside the crashed mass. So:
  - the defender fought the follow-ups, on 13 of 14 ticks;
  - the tower fired at both the mass and the follow-ups;
  - the mass died to tower fire alone (11 of 12 HP) and hit the tower for the whole siege.
- **The bounce is not caused by siege length.** Bounce timing stays fixed while siege length varies 3–6× with tower
  damage.
- **Clear could not crash because of reach, not damage.** At 16 HP per tick it stalled exactly like at 2: defenders
  behind the tower were out of reach.

**Final rules: one rule, one parameter, two companions**

| change | effect |
|---|---|
| Entering the enemy tower cell is the crash | follow-up waves join one siege; one tower target |
| Tower 3 HP per tick | 3 × 4 > 8: the tower alone out-kills the reinforcement stream (1 and 2 give self-sustaining sieges) |
| A champion in the enemy tower cell reaches one cell behind it | Apply Pressure / Vision only |
| Clear = round(4 × AP) HP per turn | Thin stays 2 per turn (controlled), Hold 0 |

W = 8, the `floor(h / 8)` law, the geometry, spawns, the economy and the 46-action / 88-entry interface are unchanged.

**Results** (deterministic, from s0)

- **Sieges** last 3–9 turns with 2–5 minion tower hits (prototype: 6–13 turns, 10–43 hits).
- **Stacked crashes bounce** after 20–22 turns with 24 HP. Small fast crashes start a longer cycle of about 40 turns.
- **Power progression:** from a neutral lane, Clear crashes in 5 turns at AP 1 and 1–2 at AP 4, and smaller (25 → 13–16
  HP). At AP ≥ 2, Clear shoves every fresh wave.
- **Freeze outside tower range** (enemy at −1, own front in its tower cell):
  - held 20+ turns under the freezer's Thin, with a 2.4× farm edge;
  - 13 turns uncontrolled;
  - broken by the opponent's Clear within 2 turns.
- **No action dominates a passive opponent:** a one-turn slow push +4.6 over 80 turns; constant Thin / Clear −10.8 /
  −18.8.
- **Recall:** free right after one's own large crash; 1.7–1.8 during the enemy's siege.
- **Performance** unchanged: lane step 38.7 µs, env step 206 µs.

**Decision: PASS WITH CAVEATS.**

- Push profitability hinges on `collect_contest_enemy_tower` (0.5); at 1.0 push + Pressure becomes profitable (+3 to
  +4.4).
- Plates now come mainly from Pressure.
- Large power gaps allow a perma-shove or perma-siege.
- Freezes need a precise control band and form mainly after an enemy crash.
- One cell separates the lane from the tower.

**Next steps (not started):**

1. Combat-driven zoning on `m4_natural()`: the freeze state and the Pressure siege.
2. Calibrate the push economics there (enemy-tower collection, plate gold, Pressure damage).
3. Enumerate or accept the evaluation bound (per-turn 10.36).
4. Integrate and freeze for PPO.

## W. M4.2e-B design philosophy and combat-driven zoning (2026-10-12)

Branch `development/m4-natural-waves`, on top of M4.2e-A.1.

- **Design philosophy:** [design_philosophy.md](design_philosophy.md), now the standard every mechanic is judged
  against: simplify how League works without changing why its laning strategies work (core principles, intended
  qualitative similarities, accepted simplifications, research and validation standards).
- **Full account:** [m4_2e_zoning_report.md](natural_waves/m4_2e_zoning_report.md).
- **Driver:** `lanegame.experiments.m4.zoning_natural` (reactive scripted policy-matrix games solved as zero-sum
  games: 13 shared policies, 4 states, 5 power gaps, 9 rule variants; 135M turns, 7 min); results
  `results/m4_zoning_natural/`.
- **Tests:** `tests/test_m4_zoning_natural.py`.
- **Candidate preset:** **opt-in** `GameConfig.m4_lane()` = `m4_natural()` plus `collect_contest_enemy_tower 0.9`,
  `collect_contest_own_tower 0.6`, `invest_durability True`. Every other preset is unchanged (random-play digests
  pinned). Also implemented, tested and left off: `trade_exchanges` (one exchange per tick; a trade-fest at equal
  power) and `trade_farm_rule "loser"` (no measurable effect without it).

**Findings**

- Under the A.1 rules power was almost worthless: +0.3 per 32 turns at a x1.76 lead in the neutral lane; pushing
  lost at every gap (-6 pushed, -3 siege); a freeze paid +4.7 at every gap. Two fixed collection rates did it: the
  0.5 rate under the enemy tower (a danger proxy) and the full 0.9 collection of a crashed wave under one's own
  tower (a missing inefficiency).
- The weaker champion never stands back at the lane in any of 180 solved policy games: on this geometry a passive
  concession is almost total (farm reach equals threat reach and the wave does not drift toward a champion that
  stops contesting), while contesting a x1.76 opponent cost 0-4 because trading was a symmetric war of attrition.
  Per-tick exchanges, a two-wave bounty and removing the posture base were tested and rejected (trade-fest at equal
  power; noise; free Safe).
- With `m4_lane()`: every state's value is monotone in power; a one-item lead is worth 2-3 per 32 turns and a large
  lead 4-7 (15-25% of the lane's income); the stronger champion earns it through wave control and attacks on under a
  third of its turns; the equal lane is even and active; pushing when ahead pays. The weaker champion's rational
  concessions are wave control (shove away, or farm the crash under its tower while the stronger champion takes
  plates), position (the stand-off keeps half its farm unhit) and tempo (recalls).

**Decision: PASS WITH CAVEATS.** The stand-at-XP-range concession of League is not produced by any rule tested; the
fix is a design decision (last-hitting as wave damage, or an all-in action) listed for review together with the two
calibrated parameters (own-tower rate 0.6, kill bounty 0.5). No PPO was run; the game is not declared final.

**Next steps (not started):** design review of the hypotheses above; then integrate and freeze for the M4.3 PPO
smoke test on `m4_lane()`.

## X. M4.2e-C wave action redesign and mechanical calibration (2026-10-12)

Branch `development/m4-natural-waves`, on top of M4.2e-B.

- **Full account:** [m4_2e_wave_actions_report.md](natural_waves/m4_2e_wave_actions_report.md).
- **Driver:** `lanegame.experiments.m4.wave_actions` (74k deterministic turns, 8 s; results
  `results/m4_wave_actions/`); sampled policy games through `lanegame.experiments.m4.zoning_natural lasthit` (2.5M
  turns, 1 min; `results/m4_zoning_natural/summary_lasthit.txt`, `trajectories_lasthit.txt`).
- **Tests:** `tests/test_m4_wave_actions.py` (16).
- **Candidate preset:** **opt-in** `GameConfig.m4_lasthit()` = `m4_lane()` plus `wave_clear_scaling "capacity"`
  (`wave_hold_fraction` 1/8, `wave_thin_fraction` 1/2), `damage_law "lanchester_half"` and `thin_damage 0`. Every
  other preset is unchanged (digests pinned).

**The model.** One wave-clear capacity C = round(4 AP) HP per turn; Hold uses an eighth of it (1 HP per turn until AP
2.9, then 2, then 3), Thin half (2 at level 1, 9 at level 18 with four items), Clear all. In the brief's notation
Thin = Hold + 3/7 (Clear - Hold). The minion damage law moves to half-wave resolution (firepower floor(2h / 8) per turn,
split over the ticks), because with last-hit damage on both sides the fronts live in the whole-wave law's flat 8-15 HP
band and a one-turn edge took 19 turns to crash instead of 13; under the finer law it takes 15. Apply Pressure's own
thinning is removed: the wave action is the champion's damage on the defenders (Hold + Pressure with the old thinning
equals Thin + Pressure without it).

**Findings** (deterministic unless stated)

- The neutral lane stays symmetric at every power; a champion three times stronger pushes it by last-hitting alone.
- A wave drifts to a champion that stands back: crash in 10-12 turns against a Safe or recalling opponent; standing
  back costs the Safe champion 3.8 instead of 8.9 per 40 turns.
- Thin scales: from a neutral lane it crashes in 9 / 5 / 3 / 2 turns at AP 1 / 2 / 3 / 4 (before: 7 at every power).
- Sieges 2-6 turns at tower 3 (self-sustaining at 1, as before); crashes 21-27 HP bounce 16-26 turns later with 23-29.
- The outside-tower freeze survives in a narrower band (enemy 12-14 HP against own 8-11): reached after absorbing an
  enemy crash, held 20-30 turns with a 1.7x farm edge under a threshold controller (Thin when the enemy front
  reaches 14), broken by Clear in 1-2 turns; a freezer at AP 3 pushes it out by last-hitting.
- Recall from a neutral lane costs about a plate (3.9) against an opponent that takes plates; after one's own crash
  it is free.
- Policy games (16 seeds): against a stronger champion that only threatens, standing back is now the weaker
  champion's best reply (farm 9.8 of 13, never hit; the bare threat is worth -0.2 to -1.5 to the zoner instead of
  +6.5 to +10.5); the threat converts through wave control (Clear + punish, +6 to +8), so the equilibria keep the
  stronger champion Clearing and the weaker one contesting. Values stay monotone in power; the equal lane is even
  (+0.04, both sides slow-push); the pushed, siege and freeze states are worth +1.4 / +1.9 / +2.1 at equal power
  (before 0.0 / +1.2 / 0.0).

**Decision: PASS WITH CAVEATS.** The freeze band is 3 HP wide and needs an absolute threshold; a champion far ahead
cannot freeze; a crash into one's own tower is still collected at 0.6 and the drift punishes passive opponents that
never take plates; the stand-back concession is a best reply to bare threats, not an equilibrium outcome; scripted
evidence only. Listed for review: the own-tower rate (harness variants `lasthit_ot5` / `lasthit_ot7`) and whether a
16-HP wave is worth its rescale to widen the freeze band.

**Next steps (not started):** review; then the M4.3 PPO smoke test on `m4_lasthit()`, judged against the sampled
equilibria of the report's section 6.4.

## Y. M4.2e-D visual simulation debugger (2026-10-12)

Branch `development/m4-natural-waves`, on top of M4.2e-C. A temporary validation tool, not a mechanics change.

- **Full account:** [m4_2e_debugger.md](natural_waves/m4_2e_debugger.md).
- **Launch:** `pip install -e .[viewer]`, then `streamlit run lanegame/viewer/app.py`.
- **Code:** `lanegame/viewer/` has four modules:
  - `session.py`: setups, recording and replay, global coordinates, summaries, events;
  - `policies.py`: the `zoning_natural` policies with explained fallbacks, and the example matchups;
  - `render.py`: the Plotly figures;
  - `app.py`: the Streamlit app.

  The only engine change is the opt-in `LaneEnv(record_ticks=True)`, which exposes the authoritative lane step's
  tick trace as `info["lane_trace"]`. It is off by default and every existing result is unchanged.
- **Tests:** `tests/test_viewer.py` (23). They cover replay against a plain `LaneEnv`, determinism and branching,
  that the trace is authoritative, B mirroring, the economy identity, legality, that the policies match
  `zoning_natural`, and that every drawn frame follows the states or the tick trace.
- **Delivered:** strategy-vs-strategy matchups (7 examples), manual play for A with branching and take-over, a
  compact lane picture (packets as HP-labelled rectangles, crashed masses, towers with plates, champions, contact and
  movement arrows) with turn and tick playback (Play / Pause, speed, tick-by-tick), player strip and summary line,
  champion and economy panels, timelines, the event log and a grouped `info` inspector. The layout follows the
  reference prototype of the M4.2e-D UI refinement brief.
- **Status:** implemented. M4.3 not started.

## Z. M4.3a RL infrastructure integration and correctness audit (2026-10-12)

Branch `development/m4-3a-rl-infrastructure`, on top of M4.2e-D. No mechanics change; every preset's digest pin is
unchanged.

- **Full account:** [m4_3a_rl_infrastructure_report.md](rl/m4_3a_rl_infrastructure_report.md).
- **Research question made explicit** (README): can an RL agent learn to take advantage of the simulator's strategic
  interactions without being taught League strategies? Rewards stay economic (relative gold); nothing is shaped.
- **PettingZoo.** The Parallel API adapter now takes M4 champion states and `StartState` starts, reports the legal
  sets, converts to AEC, and documents seeding and continuing time. PettingZoo's official API tests pass on every M4
  preset; the adapter's trajectories equal the native `LaneEnv`'s exactly (observations, masks, rewards, flags, turn
  records). Decision: the adapter is the standard external interface; the trainer keeps stepping `LaneEnv` for its
  per-seat bookkeeping, pinned matchups, batched forward passes and pre-reset bootstraps, all now tested directly.
- **PPO audit.** No defect found. New independent references: GAE equals the forward-view λ-return with random
  terminations and truncations; the PPO loss, entropy and parameter gradients equal a re-implementation on
  `torch.distributions.Categorical`; masking is exact in probability and gradient; single-choice steps carry no
  gradient; truncation bootstraps equal the value of the independently replayed pre-reset observation; segment
  bootstraps equal the next segment's first value; bit-identical reproducibility across processes; checkpoints
  carry a verified SHA-256 digest and provenance (format 2). Training histories now record gradient norms, critic and
  return scales, per-seat and truncation counts.
- **Smoke test on `m4_lasthit()`** (`lanegame.experiments.rl.m43a_smoke`, `results/rl/m43a_smoke.md`): 100/0 and
  80/20 shared self-play at 409,600 agent transitions each plus three 102,400-transition reproducibility runs, 5
  minutes in five processes. All 19 checks pass: finite losses and gradients (pre-clip norm ≈ 0.4, max 4), KL
  0.003, explained variance 0.70–0.75, critic on the return scale (|V̂| ≤ 10.7 against Ω = 107, V̂(s0) within ±0.3),
  equal seat counts, one truncation per game and no termination, identical digests for identical seeds. Throughput
  ≈ 5,000 agent transitions/s per process (the 250 µs environment step is 75% of the time). Readout, not a result:
  both policies beat the scripted natural-wave suite by 10–19 resources and the initial network by 12; sp80 beats
  sp100 by 0.8 ± 0.5; both clear 82% of turns and trade 27% with 17 deaths per side per 460 turns; entropy fell
  2.58 → 0.8.
- **Tests:** `tests/test_pettingzoo_m4.py` (11), `tests/test_rl_reference.py` (10), `tests/test_rl_m4_lasthit.py` (6);
  full suite 898 passed, 4 skipped, 2 expected failures in 7 min 33 s.
- **CLI:** `rl-shared --model m4_lasthit` (continuing-time preset), `experiments-rl m43a_smoke`.
- **Open before M4.3b** (report §7): the early entropy collapse and the clear-and-trade policy (entropy-coefficient
  control; audit death economics; judge against the M4.2e-C §6.4 equilibria); stronger evaluation references and a
  continuing-time exploitability attacker; headline window W = 1000; bootstrapping remains an approximation;
  the 80/20 vs 100/0 comparison at pilot scale with seeds; a small hyperparameter grid.

**Status:** complete. **Next:** M4.3b self-play pilot and research runs on `m4_lasthit()`.

## AA. M4.3b PPO self-play pilot and strategic learning diagnostics (2026-10-12)

Branch `development/m4-3b-ppo-pilot`, on top of M4.3a (`main` at 1ad8848). No mechanics, reward, observation or
hyperparameter of the game changed; every preset's digest pin is unchanged.

- **Full account:** [m4_3b_ppo_pilot_report.md](rl/m4_3b_ppo_pilot_report.md); results and figures in
  `results/rl/m43b/` (`report.md` has every table); checkpoints in `results/rl/checkpoints/m43b/` (untracked).
- **Driver:** `lanegame.experiments.rl.m43b` (stages train, curves, headline, exploit, behavior, counterfactual,
  plans, recall, archive, trajectories, figures, report; every evaluation stage resumable);
  `python -m lanegame experiments-rl m43b_pilot`.
- **Grid:** `base` (100% current, entropy 0.01), `ent03` (0.03), `hist80` (80/20, pool 8) and the follow-up `mixed`
  (half the games from an archive of 857 witnessed reachable states), 3 seeds each, 2,048,000 agent transitions per
  run (13 min for nine runs in nine processes). All PPO health checks pass; the baseline's entropy plateaus at
  0.5–0.75 nats from 400k on (no collapse).
- **References:** the reactive policies of the M4.2e-B/C policy games wrapped as environment agents (`farm`,
  `push_punish`, `clear_punish`, `bully_shove`, `freeze_punish`, `respect`, `crash_recall`, `trading_pusher`).
- **Is PPO learning (W = 1000, 32 paired episodes)?** Every final beats every reference by 9–20 resources and its
  untrained network by 12; finals beat their own snapshots at 100k / 400k / 1M by +4 / +1.5 / +0.5; the reference
  values plateau after 100k because the references are dominated. Attackers (warm-started, half the budget) gain
  0.75–2.0: `ent03` 0.88 ± 0.13, `base` 1.42 ± 0.55, `hist80` 1.37 ± 0.14. Cross-play: `ent03` beats `base` by
  0.50 ± 0.22 and is the most reproducible across seeds (seed SD 0.2 against 0.9); `hist80` and `mixed` are even
  with `base`.
- **Exploration.** The 0.03 coefficient doubles the entropy and moves the policy towards Thin (0.13 → 0.21) and
  Trade (0.27 → 0.36) with a small gain on every instrument; Clear stays the dominant wave action (63–75%) under
  both coefficients, with and without historical opponents, and from the archive starts.
- **State dependence.** The behaviour choice is strongly conditioned (Trade at the lane, Safe and Hold at the tower,
  Recall at low HP and after the own crash, Pressure on 76–90% of legal turns); the wave choice is Clear in 60–90%
  of every situation, modulated towards Thin while sieging and Hold at low HP. Single-move counterfactuals: the
  policy's move is within a few tenths of every alternative at every sampled state. Plan-level counterfactuals:
  a 14-turn slow push, pure last-hitting and the 20-turn freeze controller lose 1.1–2.8 resources against the
  policy for every policy; sustained Clear 0.2–0.9. Clear dominance is a best reply to Clear: a game property, not
  an exploration or coverage failure.
- **Recall and death.** Recall is legal on ~60% of turns and taken on 6–8% (12–16 per 460-turn game, 2–4× the
  deaths); 55–80% of investment arrives through Recalls. Counterfactually, Recall pays after the own crash (+0.5
  over staying), is even at low HP, costs 0.3–0.9 elsewhere, and does not depend on unspent gold; a death costs
  1.1–1.3 in full against 0.0–0.55 for a completed Recall. The M4.3a "no Recall" was a schedule artefact (its
  learning rate had annealed before the discovery, which happens between 100k and 400k).
- **Mixed start:** no change on any measure; retired unless a power-gap archive is built (policies see power gaps on
  0.3% of self-play turns).
- **Tests:** `tests/test_rl_m43b.py` (11); full suite 909 passed, 4 skipped, 2 expected failures. **Viewer:** mode
  "Recorded trajectory" replays the exported games.

**Decision:** M4.3b complete. PPO self-play learns strong, locally consistent, state-dependent policies whose
dominant line (Clear with trades and resets) the game's dynamics reward against any opponent that clears. **Next
(M4.3c):** entropy 0.03 as the default; population attackers; the 100/0 vs 80/20 comparison with a wide snapshot
pool; and a design investigation of Clear dominance (kill bounty, gank severity, own-tower collection rate) judged
by the plan-level counterfactual harness against the learned policies, not by scripted play.

## AB. M4.3b follow-up: strategic validation and research-artifact cleanup (2026-10-13)

Branch `development/m4-3b-validation`, on top of `development/m4-3b-ppo-pilot` (98a4341). No mechanics, reward,
hyperparameter or checkpoint changed; no training. Full account: [m4_3b_validation_report.md](rl/m4_3b_validation_report.md);
tables and figures in `results/rl/m43b/validation/`.

- **The confound.** M4.3b §7.2 forced complete actions (Thin + Hold position, Hold + Hold, the freeze controller with
  Hold position), so its 1.5–2.8 losses mixed the wave management with the loss of the policy's adaptive behaviour.
  New: `Plan.transform` (a distribution-level hook of the diagnostics rollout, backwards compatible) and
  `intervene.py`: the wave component is forced on the policy's probability vector, behaviour / objectives / Recall
  stay the policy's (two explicit rules, "marginal" and "conditional", both reported); legality holds without
  fallback because all three waves are legal on every present turn and the rest of the code does not depend on
  the wave (tested on every legal set met in play).
- **Result 1 (sampled states, W = 300, four opponents).** Wave-only Hold, slow push and freeze lose 1.7–2.6 against
  the policy (within 0.2 of the full-action values on the same states and streams), 0.6–1.6 against the pure
  last-hitter, 1.8–3.1 against the reactive slow pusher, 2.3–4.0 against the constant clearer; significant in 16–24
  of 24 states per cell. Wave-only Thin costs 0.2–1.0 (not significant in about half the states); wave-only Clear
  is within 0.1 of the policy. The confound was ≤ 0.2 against the policy and ≤ 0.8 against the scripted opponents,
  in both directions.
- **Result 2 (whole games, W = 1000).** A policy that never Clears while the enemy is present loses 8–13 resources
  per game against every opponent, the passive farmer and the scripted freezer included; forced Thin costs 1–2
  against Holders and 5–6 against clearers (the one opponent-dependent wave interaction); forced Clear ± 0.5.
- **Result 3 (horizon).** The same rollouts read at W = 300 / 460 / 1000: the paired tail beyond 300 is 0.00 ± 0.01
  (max 0.08 over 544 pairs, no sign change). The 300-turn window was sufficient; the guaranteed bound is a bound on
  levels, not on paired differences.
- **Result 4 (freeze states).** Of 88 freeze-like states, half are maintainable freezes against a Holder (held
  10–19 of 20 turns) and a quarter favourable; the freeze loses the same 1.5–4 in every class. Synthetic power gaps
  (constructed starts, labelled): Hold and freeze lose in all 72 cells and lose *more* when ahead against a contesting
  opponent; Thin is marginally positive (+0.2–0.6) only when far ahead at a frozen lane against scripted opponents.
- **Interpretation.** Outcome C of the brief: the tested strategy class is unprofitable under the game as calibrated
  (Clear costs the clearer nothing the reward counts), confirmed with adaptive behaviour, four opponents, the long
  horizon and maintainable freezes. Not an exploration failure; the one gap between what the game rewards and what
  the policies play is the Thin edge when far ahead, which self-play never visits. Outcome D is not excluded for
  strategies that coordinate behaviour differently (a wave-fixed best-response attacker would close it).
- **Artifacts.** JSON results ≥ 1 MB gzipped (readers accept both forms); `results/rl/m43b/MANIFEST.json` (files,
  SHA-256, commit / stage / command, checkpoint digests); a 5.2 MB checkpoint bundle in the git-ignored
  `results/rl/m43b_artifacts/`; `ARTIFACTS.md`. The M4.3b commit is 3.8 MB packed (under 3% of the pack), so **no
  history rewrite is recommended**; a no-force-push replacement-branch recipe is in the report §9 for the user's
  decision.
- **Tests:** `tests/test_rl_m43b_validation.py` (14); diagnostics tests unchanged.

**Decision:** M4.3b conclusions stand with stronger evidence; the plan-level attribution to wave management was
correct. **Next (M4.3c, to be decided together):** the design investigation of Clear dominance with `val_wave_cf`
against the existing finals as the screen for each lever (minutes, no training), training only for levers that move
the wave-only loss towards zero; plus one wave-fixed best-response attacker to close outcome D. No further self-play
campaign on the unchanged game.

## AC. M4.3c design: economic objective, laning termination, mathematical reference (2026-10-13)

Branch `development/m4-3b-validation` (working tree on 26f8c83). **Design only**: no game rule, preset, observation,
reward or checkpoint changed; no training. Full account:
[m4_3c_economic_objective_design.md](m4/m4_3/m4_3c_economic_objective_design.md); the new living reference of the current
game: [laning_game_mathematical_reference.md](m4/laning_game_mathematical_reference.md); tables in
`results/m43c/report.md`; diagnostics `lanegame/experiments/m43c/` (`python -m lanegame.experiments.m43c all`).

- **Reference (Part A).** One document for `m4_lasthit()`: state, observation, 46 actions and legality, natural wave
  physics with engine-generated examples, collection and XP, combat / ganks / death / Recall, investment and
  durability, towers and the exact reward (a constructed turn with farm, plate and bounty), the RL objective, three
  walkthroughs, glossary; a reconciliation table of ≈30 disagreements between historical documents and the code
  (deprecated rules, outdated numbers, a few genuine discrepancies such as Safe still damaging the wave and one-turn
  kills being possible by combination).
- **Gold + XP (Part B).** The discounted flow objective equals the expected lead at a geometric end of laning, so
  flows and stocks are one objective. Re-pricing the M4.3b wave-only counterfactuals (reproduced exactly) and 261 new
  laning-phase states under gold + λ·XP (λ ≤ 1/4) or level progress changes the sign of nothing: XP accrues at the
  spawn rate whenever a champion is present, so XP mostly prices absence.
- **Termination (Part C).** In self-play the first final plate falls at turn 109–143 (levels 15–18; both level 18 at
  143–146); against references at turn 29–55; two last-hitters never finish a tower. Pressure supplies 95–100% of
  tower damage, so the end is the attacker's choice. 79–81% of the M4.3b validation states lie after a tower's last
  plate; 40–68% of the headline margins over references are earned before the proposed end. Termination creates a
  "prolong when ahead" incentive (first sign: withholding wave damage turns slightly positive against a passive
  opponent once games can end).
- **Freeze and death (Part D).** Walking up to a held freeze costs nothing over 40 turns (a delay, paid in HP);
  stepping back costs 4.9 gold; breaking it pays (+4.8 for the learned policy). The causes are cheap HP (death and
  Recall refill it) and zoning only through trades, not wave mechanics or XP. A death costs 0.7 / 1.4 / 1.25 (early /
  mid / late), 1.2–1.5 above a Recall; each extra absent turn costs 0.4–0.6 because lane income does not grow with
  level.

**Decision (pending review):** proposed objective = incremental gold + (1/8)·credited XP, terminal at either tower's
last plate, γ = 0.99 kept as the laning hazard, no fallback horizon, death timers unchanged; game-length calibration
(thresholds and γ) to be decided with the screens before any training. Stopped before implementation and training.

## AD. M4.3c screens, four-action wave redesign and stabilization (2026-10-13)

*Added to this roadmap in M4.4a; the work itself was committed as 4bd116c and b2e05ad. Full accounts in the linked
reports; nothing here restates them beyond the decisions.*

- **Quick screens.** [xp_range](m4/m4_3/m4_3c_xp_range_screen.md): keep 3 (range 2 taxes unfollowed crashes, range 1 breaks
  neutral XP). [Wave control](m4/m4_3/m4_3c_wave_control_screen.md): adopt Hold capped at 1 HP on its own; adaptive Thin not
  yet (opt-in fields `wave_hold_cap`, `wave_thin_rule`).
- **Four-action redesign** ([report](m4/m4_3/m4_3c_four_action_wave_redesign.md)): EXPERIMENTAL `GameConfig.m4_four_actions()`
  = `m4_lasthit()` + Last Hit (1 HP), Thin and Slow Push with window-strength targets, Shove (full capacity); Slow Push
  codes appended as 46–60 (61-code space, every historical code unchanged). Mechanically ready for PPO; Clear / Shove
  dominance unchanged.
- **Stabilization and economic calibration** ([report](m4/m4_3/m4_3c_strategic_economics_calibration.md)): Shove
  terminology, the four-action debugger, mixed-action sequences; diagnosis that the own-tower "crash tax" (0.6) and
  break-even trades (0.5) made Shove every side's best reply; EXPERIMENTAL candidate `m4_four_actions_econ()` (own tower
  0.8, trade factor 0.8, bounty 2, death 3 / 4 / 5 turns by level); opt-in `xp_reward_weight`, `death_turns_by_level`,
  `terminal_rule`; an exploratory PPO pilot (2 seeds × 2M) in which the candidate's policy mixes all four actions and
  keeps freezes with Thin / Slow Push / Last Hit.

**Decision:** the four-action model is the experimental basis for the next PPO milestone; the M4.3b instruments must
first be generalized to four wave slots (done in M4.4a, §AF).

## AE. M4.3d tower farming, pressure and gank risk (2026-10-13)

*Added in M4.4a; committed as f931ae8. Full account: [m4_3d_tower_pressure_calibration.md](m4/m4_3/m4_3d_tower_pressure_calibration.md).*

- Keep the own-tower collection rate 0.8 (an uncontested crash is farm-neutral for the defender; 0.9 makes it a gift).
- Gank risk already prices forward pressure (3–4 resources per 100 turns at equal power; Vision halves it): no gank
  change.
- The tower's "busy" discount under-priced a strong champion's dive: EXPERIMENTAL candidate
  `m4_four_actions_econ_aggro()` = econ + `tower_shot_p_busy` 1.0, `tower_adjacent_p_busy` 0.5, identical to econ in
  every game without a busy dive.
- Structural limits documented, not tuned: contact geometry at the tower, ganks independent of the opponent.

**Decision:** PPO on `m4_four_actions_econ_aggro()` with `m4_four_actions()` as the control and
`m4_four_actions_econ()` as the dive control, after the instruments are generalized (M4.4).

## AF. M4.4a four-action PPO readiness, strategic diagnostics and roadmap update (2026-10-13)

Branch `main`, working tree on f931ae8, **uncommitted, in review**. No game rule, preset, reward, observation, action or
PPO hyperparameter changed; the M4.4b budget was not spent. Full account:
[m4_4a_ppo_readiness.md](m4/m4_4/m4_4a_ppo_readiness.md); the pipeline explained for a reader learning PPO:
[ppo_training_explained.md](rl/ppo_training_explained.md); tables `results/m44a/report.md`; M4.4b specification
`results/m44a/m44b_plan.md`.

- **Package** `lanegame/experiments/rl/m44/`: the three environments with one training specification and paired
  seeds; a checkpoint guard (`load_policy`: different shapes always refused, other presets only with an explicit
  cross-environment opt-in); 61-code decoding (34 codes can ever be legal); game records; state-based metrics (wave
  management by decision state and intent versus effect, freeze episodes, zoning under a credible threat, vision /
  pressure / dives / tower shots, plate phases from the tower state); component interventions (wave, behaviour,
  objective; legality enforced); paired counterfactuals with two opponent protocols and a separate critic readout;
  training, evaluation, viewer traces, the M4.4b plan, the report. `diagnostics.branch` gained a backwards-compatible
  `opponent_plan`; the viewer and the `rl-shared` CLI gained the two calibrated presets.
- **Verified**: shared self-play on all three environments (own-perspective mirror, both seats trained, antisymmetric
  rewards, legal sampling, truncation-only time semantics); smoke training of the full matrix at 204,800 transitions
  (81 / 81 pipeline checks; every final beats its initial network by +5.7 to +10.0); bit-identical re-training; the
  whole evaluation pipeline including viewer replay; 31 new tests; full suite 1073 passed, 4 skipped, 2 expected
  failures.
- **Demonstration** on the exploratory M4.3c pilot finals: the instruments reproduce the pilot's freeze contrast,
  show the four-action policy switching from Pressure to Vision once the plates are gone, show that Slow Push does not
  produce a strength lead more often than Last Hit, and find that only 16–24% of learned dives are the "busy" dives
  `aggro` changes.
- **Cost**: M4.4b training ≈ 15 min wall clock (9 processes, 2.0 CPU-hours for 18.4M transitions), evaluation ≈ 37 min.

**Recommendation:** GO for M4.4b as specified, after review. **Next:** M4.4b (`python -m lanegame.experiments.rl.m44
train_main --confirm-m44b`, then `eval_main`).


## AG. M4.4b controlled PPO self-play training and initial strategic evaluation (2026-10-13)

Branch `main`, working tree on f931ae8 with the M4.4a changes, **uncommitted, in review**. No game rule, preset, reward,
observation, action or PPO hyperparameter changed. Full account:
[m4_4b_ppo_training_report.md](m4/m4_4/m4_4b_ppo_training_report.md); tables and figures `results/m44b/report.md`,
`results/m44b/figures/`; checkpoints `results/rl/checkpoints/m44/{main,ext}/` (git-ignored).

- **Run**: 3 economies (`four`, `econ`, `aggro`) × seeds 0, 1, 2 × 2,048,000 transitions with identical settings and
  paired initial weights (13.3 min on 9 processes); the pre-registered evaluation (50 min); supplementary analyses
  (snapshot tournament, more opponents, snapshot behaviour, integrity, a labelled transfer test, representative viewer
  games, an interrupted-Recall follow-up); a matched 2× extension (4,096,000 transitions, all 9 runs). 81 / 81 pipeline
  checks at both budgets; full-budget re-training bit-identical; regression suite 1083 passed.
- **Strength and convergence**: every final beats its initial network (+11.8 to +13.0), all ten references and every
  earlier checkpoint (0 of 324 tournament pairs lost: no cycling); flat against fixed references from ≈ 1M. Behaviour was
  still drifting toward Shove at 2.05M; the 2× extension stopped the aggregate drift (Shove ≈ 35%) but is ≈ +1 resource
  stronger head-to-head and changed conditional details: qualitatively stable, quantitatively budget-dependent.
- **Economies**: `four` Shoves ≈ 70% in every state (outcome C of M4.3c, confirmed by the transfer test); `econ` /
  `aggro` Shove 36% conditionally (14–16% holding a freeze, 56–59% frozen on, 21–24% with an enemy crash under their
  tower). Freezes emerge (7–8 per 1000 champion-turns), are preserved and broken, but deny weakly (≈ 50% useful at 2.05M,
  61–64% at 4.10M; +0.5 resources per episode). `aggro` cuts busy dives by 60% (every seed) and changes nothing else
  measurable. The weaker side concedes in an initialized ×1.76 gap (and Safe beats Trade on every `aggro` seed); the
  stronger side does not zone; natural play has no power gaps. Vision is used by plate phase but rarely before
  sustained Pressure. 22–38% of chosen Recalls are interrupted by an enemy Trade, often fatally.
- **Counterfactuals** (pre-registered one-turn swaps, 8 states × 16 rollouts): null within ±0.3 in four of five
  situations; `aggro` Trade − Safe under threat −0.67 [−1.04, −0.30]. The critic's one-step sign agrees in 53% of states.

**Recommendation:** CONDITIONAL GO for M4.4c: keep `m4_four_actions_econ_aggro()` as the primary candidate (`econ` as
dive control, `four` as the Shove control), analyse the 4.10M finals with the 2.05M finals as replication, fix a
convergence criterion, and move the counterfactual questions to multi-turn interventions with more states.
**Next:** review; then M4.4c.


## AH. M4.4c deep investigation of strategic behaviour and economic incentives (2026-10-14)

Branch `main` on f4bcb7d, **uncommitted, in review**. No game rule, preset, reward, observation, action or PPO setting
changed; no network trained. Full account: [m4_4c_strategic_investigation.md](m4/m4_4/m4_4c_strategic_investigation.md);
tables `results/m44c/report.md`, figures `results/m44c/figures/`, viewer recordings `results/m44c/trajectories/`.

- **Framework** `lanegame/experiments/rl/m44c/`: multi-turn strategic plans over existing legal actions (preconditions,
  stop conditions, logged fallbacks) on the M4.4a rollout engine; paired common-random-number rollouts with episode
  ledgers and discounted returns over 5–300 turns; opponent protocols (policy, modal) and styles (the learned policy
  with one component forced in one situation); resumable stage runs; 16 tests. 5,982 starting states, ≈ 5.3 h on 11
  processes; weights unchanged in every job; a re-run is bit-identical.
- **Recall**: a simultaneous guessing game; Recall is the best or tied-best of five legal alternatives (Hold −0.6 to
  −0.8, Safe −0.14 to −0.38 resources over 300 turns); the policy is within ±0.07 of always recalling; Safe pays only
  above an opponent Trade probability of ≈ 0.69, where the policy still Recalls ≈ 65% (≈ 0.1 per affected decision).
- **Freezes**: preserving beats breaking by ≈ 1 resource against an opponent that steps back or keeps last-hitting, and
  by ≈ 0 against the learned opponent or one that breaks it (breaking is cheap: forced preservation adds ≈ 0.5 turns).
- **Zoning**: learned. The stronger side beats every scripted zoning plan and converts its edge through selective
  trades into kills and forced Recalls; threat-only denial is mechanically weak (a Safe defender pays ≈ 0.08 / turn).
  The M4.4b "not learned" verdict is withdrawn.
- **Vision**: Vision-first ≈ 0 in typical (2–4-turn) sieges; the policy's objective choice is unbeaten (it stops
  pressuring after the plates); long sieges unresolved.
- **Wave actions**: Last Hit, Thin and Slow Push leave the same lane in 68–77% of natural turns; 8-turn sequences differ
  by ≤ ≈ 0.2 resources (80–255 rollouts per state to resolve); essentially Shove vs not-Shove.
- **Proposed design decisions (recorded for review, not adopted, not implemented)**: (1) three non-Shove wave actions
  that coincide in most states; (2) the strength of threat-only zoning (Safe evasion, the Safe collection cost);
  (3) voluntary retreat into the own tower cell; (4) Vision's duration versus siege length. M4 mechanics development is
  not reopened by this milestone.

**Recommendation:** CONDITIONAL GO for M4.4d, scoped to robustness and exploitability of the `aggro` (primary) and
`econ` policies, after the review of the four design questions. No further training sweep. **Next:** review; then M4.4d.

## AI. M4.5a unified positional combat, Vision and laning mechanics (2026-10-14)

**Status:** completed, in review (uncommitted). [Spec](m4/m4_5/m4_5a_unified_mechanics_spec.md),
[report](m4/m4_5/m4_5a_unified_mechanics_report.md),
[reference §14](m4/laning_game_mathematical_reference.md#14-experimental-the-m45a-unified-positional-candidate-m45_unified),
[smoke tables](../results/m45a/report.md); recordings `results/m45a/trajectories/`.

**Context.** The review of M4.4c's design questions (Vision versus siege length, threat-only zoning, the cost of
Safe, combat and wave control) chose one integrated mechanics revision instead of separate calibration phases,
because the mechanics interact. The roadmap became M4.5a (implementation), M4.5b (integrated calibration) and M4.5c
(PPO on the revised model); M4.4d's robustness work moved into M4.5c.

**What was built.** `GameConfig.m45_unified()` = `m4_four_actions_econ_aggro()` + six opt-in rules:

- a shared off-lane Vision node W on a small movement graph, with a contestable Ward action (code 30) replacing the
  siege Vision objective;
- XP range 2 by graph distance;
- minion retaliation after positive trade damage;
- positional aggro on a stationary champion standing in the enemy wave ahead of its own;
- exclusive per-tick minion targeting, with damage $\max(1,\lfloor50m/1.08^{L-1}\rceil)$ centi-HP divided by $M(S)$;
- a connected trader's wave budget capped at $\lfloor C/2\rceil$, plus a one-cell reach ball for wave damage.

The action space keeps 61 codes; the observation has 90 entries. The user's mid-task request for per-tick targeting
was adopted:

- the trade exchange lands on tick 0 and minions retaliate from tick 1;
- positional aggro is re-checked at every tick;
- a champion that moved this turn is exempt for the turn (found necessary by the demonstrations).

**Validation.**

- 73 new tests (geometry, Ward, targeting, damage, trade budget, reach, mirror equivariance, soaks, replay, PettingZoo,
  checkpoint guard, pins, viewer, demonstrations).
- 12 earlier champion presets byte-identical with the committed engine.
- The contract and viewer tests extended for the new fields and matchups; the full suite green.
- 7 demonstration recordings, each replaying exactly.
- A 25-second smoke study:
  - the mechanics fire at plausible rates, with no positional false positives in passive farming;
  - minion damage per hit turn falls from 9% of the bar at level 1 to 1.2% at level 18;
  - throughput is 0.88× the aggro preset.

**Recommendation:** GO WITH CAVEATS for M4.5b. The caveats are about incentives:

- every Ward can be contested;
- whether a ward is worth a turn of farm;
- one-tick retaliation;
- the trading bully loses more;
- freezes held with Thin + Trade;
- Safe's price.

**Next:** review; then M4.5b.

**Correction after review (revision 2, 2026-10-14; [report §10](m4/m4_5/m4_5a_unified_mechanics_report.md)).** The
"moved this turn" exemption treated a symptom: a Hold champion stood at its wave's end-of-turn cell during both ticks
while the minions were still at their start-of-tick cells. Now champions walk to the same end cells tick by tick
(`champion_movement_rule = "tick"`): Hold, an unthreatened Yield and Safe follow their wave's front of each tick, and
positional aggro reads each tick's positions after its movement, moved or not. Revision 1 is kept as
`m45_unified_r1()` (byte-identical to the committed engine); recordings carry a rules revision, and the committed ones
replay as revision 1. Ordinary last-hitting draws no positional aggro; chasers, Ward walks through a held lane cell and
retreats through the enemy wave now do. 58 new tests. M4.5b and M4.5c are unchanged; M4.5b runs on revision 2.

## AJ. M4.5b integrated calibration and strategic evaluation (2026-10-14)

**Status:** completed, in review (uncommitted). [Report](m4/m4_5/m4_5b_calibration_report.md),
[reference §15](m4/laning_game_mathematical_reference.md#15-experimental-the-m45b-calibrated-candidate-m45_calibrated),
[tables](../results/m45b/report.md); code `lanegame/experiments/m45b/`; recordings `results/m45b/trajectories/`.

**What was done.** A paired, common-random-number study of about 60 scripted contingent strategies on
`m45_unified()` revision 2. Its parts:

- exact one-turn trade, zoning and Vision enumerations;
- freezer × response games on a freeze grid, with local action values and post-reset decisions;
- one-ward and Ward-rule counterfactuals;
- a Safe / Yield / threat decomposition;
- 24-strategy restricted games with seed bootstraps;
- timing diagnostics;
- an α × δ × φ factorial, with one-at-a-time checks;
- a holdout fixed before calibration (seeds 88,000–88,499, its own states);
- a learned reference (the M4.4b aggro finals, evaluation only).

About 220 million simulated turns, no training.

**Findings.**

- **A farm-spot defect, inherited from M4.2d.** In mutual Shoves the probe predicts both fronts dying, so both
  followers walk out of reach of the fight: 38–50% of Shove-mirror farm died out of reach. It is corrected by the
  opt-in `farm_spot_rule = "last_front"` (5–7%); it had inflated the apparent value of freezing against a shoving
  opponent.
- **Retaliation depended on the tick split.** A damaged 8-HP wave never retaliated, and early trades into big
  stacks were HP-positive; fixed by `minion_retaliation_delay = 0`.
- **Freezes are conditional.** They pay mid- and late game, and a strong freezer punishes a walk-up Shove; the weak
  side's replies vary, including resets.
- **Vision is situational.**
- **Safe is costly; threat-only zoning exists** near the weaker side's tower.
- **Restricted scripted games are Shove-centred** on this and the historical preset; learned policies beat the
  scripted Shove by about +3.
- **The approximations M4.5a flagged are negligible.**

**Decision.** `GameConfig.m45_calibrated()` = revision 2 + the two fields above; every other number retained (α 50,
ρ 1.08, φ 0.5, Vision 6 turns × 0.25, W exposure 0, XP range 2). Every earlier preset is byte-identical (14 checked).

**Holdout.** The strategic effects generalize. The mid-to-late minion share fails its criterion (21% of the trade
damage at level 14, threshold 15%), and was not retuned on the holdout.

**Recommendation:** GO WITH CAVEATS for M4.5c, after deciding the minion share (accept, or
`minion_damage_growth = 1.12` with a fresh validation). Tracked caveats: Shove-heavy convergence, rare freezes, noisy
single-trade values, Vision magnitude, the residual full-support probe.

## AK. M4.5c pilot: shared-policy PPO on the calibrated M4.5 environment (2026-10-14)

**Status:** pilot complete, in review (uncommitted). **The full multi-seed campaign has not started; it awaits explicit
approval of the budget.** [Pilot report](m4/m4_5/m4_5c_pilot_report.md), [tables](../results/m45c/report.md); code
`lanegame/experiments/rl/m45c/`; tests `tests/test_m45c_training.py` (16); checkpoints
`results/rl/checkpoints/m45c/` (git-ignored).

**What was done.**
- The M4.4 PPO infrastructure was reused unchanged on `m45_calibrated()`: 90 inputs, 61 codes, the network trained
  from scratch, M4.4 weights refused by the guard. Two opt-in arguments were added to `train_shared`, both bit-identical
  by default:
  - `stop_at`, a prefix of a longer learning-rate schedule;
  - `rollout_stats`, per-rollout action, mask and seat diagnostics.
- Correctness was verified on the preset itself: mirror observations, antisymmetric rewards, equal seats with gradient
  from both, masks (Ward offered, never-legal codes never), masked log-probabilities, truncation bootstraps,
  reproducibility.
- A smoke test and one pilot: seed 100, the 4,096,000-transition schedule stopped at 1,024,000 agent transitions
  (512,000 joint steps).
- Evaluation against nine M4.5b strategies and the out-of-distribution M4.4b aggro network, from both seats; behaviour
  diagnostics; throughput, concurrency and memory benchmarks; a counterfactual framework for the full evaluation.

**Findings.**
- **Learning.** The scripted mean went from −7.4 (untrained) to +2.5 resources. The final beats every fixed strategy
  (+1.0 to +6.1, W = 1000) and the OOD network (+0.67). It still improves at the end: +0.67 against its 0.77M
  snapshot, and better than its half-way snapshot against 9 of 10 opponents.
- **Stability and behaviour.** Training was stable (entropy 2.86 → 1.77, matching M4.4b aggro on the same schedule;
  KL ≈ 0.005). Behaviour is state-conditioned (Recall and Safe by HP, Pressure when legal, freeze-phase actions);
  Shove is at 32% and still drifting; Ward is rare.
- **Learns more slowly than M4.4.** M4.4b aggro plateaued against its references by ≈ 0.3M.
- **Cost.** 4,277 agent transitions/s for one process; 0.58× per process with 11 processes. Memory is 1.5 GB per
  process once the lane caches are full, so concurrency is limited; a smaller cache gives identical results about 4%
  slower.

**Recommendation.** GO. The default campaign is 6 seeds × 8,192,000 agent transitions (learning rate annealed per
run, 100% current-policy self-play, a midpoint at 4.096M, pre-registered plateau / extension / stop criteria), at an
estimated ≈ 46 min of training and ≈ 2.5 h in all. Alternatives: 3 × 4,096,000 and 10 × 16,384,000.

**Follow-ups before approval (2026-10-14, in review).**
- A [runtime and scope audit](m4/m4_5/m4_5c_runtime_audit.md) reconstructed M4.4's actual cost (about 9.7 h in three
  blocks). It found the pilot's 2.5 h estimate too low, because the counterfactual battery and the attackers were
  under-priced. It also corrected the memory plan (lane cache 5,000).
- A [horizon study](m4/m4_5/m4_5c_horizon_study.md) (7.6 M simulated turns) found that 50-turn paired counterfactuals with a
  critic-bootstrapped statistic agree with 300-turn references, with no significant contradiction, and are 8–9× more
  informative per unit of compute.
- Recommended: a 50-turn default with exceptions and a per-network 300-turn validation subset. The revised strategic
  battery takes about 1.8 h; the campaign about 6 h (4.5–8.5 h).
- Training and evaluation still await explicit approval.

## AL. M4.5c full campaign: six-seed PPO training and evaluation readiness (2026-10-15)

**Status:** trained and health-checked (uncommitted). **The comprehensive evaluation was not run**; it and the proposed
matched 2× extension await explicit approval. [Training report](m4/m4_5/m4_5c_training_report.md),
[tables](../results/m45c/full/report.md); code `lanegame/experiments/rl/m45c/` (new: `full`, `health`, `health_report`,
`battery`, `attackers`, `traces`); tests `tests/test_m45c_full.py` (17); checkpoints `results/rl/checkpoints/m45c/full/`
(git-ignored).

**What was done.**
- **Training.** Seeds 0–5 × 8,192,000 agent transitions on `m45_calibrated()`, unchanged, from scratch, with the pilot's
  PPO settings and `lane_cache` 5,000. The six ran concurrently in 55 min at 2,490 transitions/s per process; the
  machine was busy with other applications.
- **Reproducibility.** The seed-0 full-budget re-run ran separately: there was not enough free memory for a 7th
  concurrent process. It started early, overlapping the end of the evaluation, with the user's approval.
- **Limited health evaluation.** About 34 M simulated turns, on the reserved seeds 93,000+:
  - curves against the fixed suite;
  - final vs midpoint and final vs 6.144M (W = 1000);
  - finals vs their own past and adjacent checkpoints;
  - cross-play of the finals, plus finals vs the other runs' midpoints;
  - behaviour, including an initialized power gap.
- **Framework.** The revised 50-turn strategic battery, the attackers, viewer traces and the remaining standard
  evaluation were built, tested, smoke-run and gated.

**Findings.**
- **Health.** All 122 checks pass; no pre-registered alarm fired; the seeds' diagnostics are nearly identical.
- **Reproducibility.** The seed-0 re-run is **bit-identical** to the original: final weights, all 320 snapshots, and
  every non-wall-clock history field.
- **Strength against fixed opponents.** The scripted-suite mean went −7.0 → +3.6 at the midpoint and +3.4 at the
  final; the OOD M4.4b network −10.1 → +2.4. Strength against fixed opponents is flat in the second half.
- **Head to head.** The final beats its midpoint by +0.96 ± 0.18 (every seed significant) and the other runs'
  midpoints by +0.91 ± 0.10. The gain is general, not lineage-specific, and shrinks over the last quarter. **The
  pre-registered extension criterion is met** (5 of 6 runs not plateaued).
- **No cycling; the seeds are consistent.** Cross-play is within ±0.61.
- **Behaviour.**
  - Shove 28–36%.
  - Freezes are fewer than at the midpoint (2.6–4.4 per 1000 turns) but won by the freezer in 60–80% of episodes.
  - Trading is selective by HP and by wave position.
  - In an initialized ×1.76 gap the weaker side concedes, but the stronger side denies almost nothing.
  - Wards are rare and seed-dependent; Recall attempts are interrupted 34–40% of the time.
- **Updated estimate** of the comprehensive evaluation: ≈ 5.7 h (4.7–7.0). Of that, the battery is 2.2 h (47.6 M turns,
  enlarged by the early-game and power-gap scenarios) and the attackers 1.5 h.

**Recommendation.** GO WITH CAVEATS. Training has not converged head to head, and the pre-registered rule calls for
proposing a matched 2× extension (16,384,000, fresh runs). The report recommends approving it before the strategic
battery (≈ 1.8 h of training plus ≈ 1 h of health check); evaluating the 8.192M finals now is the defensible
alternative.

## AM. M4.5c completion: matched 2× extension, comprehensive evaluation, strategic battery, attackers (2026-10-15)

**Status:** completed (uncommitted). Authorized end to end without approval stops.
[Research report](m4/m4_5/m4_5c_research_report.md); tables [`results/m45c/final/report.md`](../results/m45c/final/report.md),
[`results/m45c/ext/report.md`](../results/m45c/ext/report.md),
[`results/m45c/battery/report_ext.md`](../results/m45c/battery/report_ext.md); code `lanegame/experiments/rl/m45c/`
(extended: `full`, `health`, `health_report`, `battery`, `attackers`; new `campaign_report`); tests
`tests/test_m45c_full.py` (18); checkpoints and raw battery jobs in `results/rl/checkpoints/m45c/{ext,battery,attack}/`
(git-ignored).

**What was done.**
- **Extension.** Six fresh runs × 16,384,000 on `m45_calibrated()`, settings unchanged except the budget: 93 min,
  102 / 102 checks.
- **Budget comparison and comprehensive standard evaluation.** About 2 h: 21-point curves, the W = 1000 headline,
  tournaments, cross-play, 16.4M vs 8.2M, the annealing test, historical references, behaviour, 39 verified viewer
  traces.
- **Strategic battery.** The approved 50-turn design, 1 h 37 min:
  - 12 families including early-game and power-gap scenarios;
  - opponent protocols;
  - replication on the 8.192M finals;
  - exact one-turn mechanics;
  - the 300-turn validation subset.

  Exception 3 re-ran 32 failed (policy, family) validations at 150 turns (58 min).
- **Attackers.** Six best-response attackers at a fixed 4,096,000 learner transitions each, a quarter of the
  target's budget (a stated limitation): 66 min.

**Findings.**
- **The extension helped.** It is stronger in all 36 pairings (+1.03 ± 0.11), and annealing alone explains nothing
  (−0.07 ± 0.25). Not converged (+0.36 against 12.3M); no cycling; seeds interchangeable.
- **Strategies discovered**, shown by paired counterfactuals:
  - conditional wave management: no fixed wave action beats the policy's; Shove is best only immediately;
  - conditional freezing: worth what the opponent concedes, +0.56 / +1.86 against back-off, −0.20 against learned
    play; and the right reply to freezes;
  - selective Trading;
  - concession under threat;
  - Recall timing.
- **Rationally avoided:** Wards (a forced Ward costs 0.3–0.6), sustained freezes and bullying against competent play.
- **Gaps:**
  - value left against conceding opponents (+0.56 to +1.86);
  - exploitability lower bound +1.02 ± 0.48 (6/6), target-specific, through aggression and freezing.
- **The 50-turn horizon default held.** 150-turn re-runs: mean change +0.006, 0 contradictions. The per-policy
  validation rule is too noisy at its subset size.

**Recommendation.** GO WITH CAVEATS. Next is a decision between:
- **M5:** opponent-adaptive training and robustness, the main research gap;
- **M4.6:** a design review of the Vision payoff, freeze durability and the zoning payoff, as questions about what the
  model should represent, not tuning toward realistic frequencies.

Also recommended: a convergence probe, half-budget attackers, and the battery's validation rule pooled over policies.

## AN. M4.6a diagnosis and M4.6b investment audit (2026-10-15)

Reports: [M4.6a](m4/m4_6/m4_6a_report.md), [M4.6b](m4/m4_6/m4_6b_investment_report.md) (pre-registrations beside them).
No mechanics, defaults or checkpoints changed; no general training campaign.

**M4.6a: freeze breaking and threat-based zoning.**
- Zoning is rewarded by the rules. PPO learns more of it when trained from gap states (+2.11 ± 0.62 for a specialist),
  but natural self-play rarely creates gaps.
- Freezes break cheaply for structural reasons, although a guard can hold one early.
- The reset is not subsidized, and a bar of HP is worth 1.2–1.5 resources.
- Recommendation: training changes (M4.6c) first, with the measurement fixes.

**M4.6b: investment audit, candidate-curve screen and measurement fixes.**
- **Measurement fixes, verified by reproduction.**
  - Positions are now lane-only, with Ward occupancy reported separately.
  - The economic channels now include farm lost while away, bounty and plates.
  - Each battery candidate has a behaviour-only twin, with the scripted plans kept as references.
  - Corrected readings: the stronger side stands just behind the lane cell, not +2 cells forward. Walking up to a
    freeze costs −0.02 to −0.13, not −0.22. The trading decision costs about −0.08, not −0.34.
- **The ×1.01 is a late-game statement.**
  - Natural leads are 4–10% of earned gold.
  - Saturation compresses them 6–11× (elasticity 0.17 falling to 0.09).
  - Before turn 50, leads sit in unspent gold.
  - M4.6a's lead bin had its median at turn 418.
- **Early investment already matters.** Twenty turns of income buy ×1.25 in a duel early, ×1.06 mid-game and ×1.02
  late.
- **Candidates screened.** Three strength-matched opt-in curves: C1 and C2 saturate more slowly, C3 is a linear
  reference.
  - They raise the mid and late leverage of leads, and natural leads persist.
  - One-turn incentives, freeze contests, Recall and the equal lane are essentially unchanged.
  - By the pre-registered rule **no candidate advances**: late wave capacity rises 3–7 HP (D4), the early premium
    doubles (D2), and the screen was too underpowered to establish the mid and late premium (D1).
  - An exploratory powered follow-up (K = 128, both orientations) shows the effect. The mid-game premium rises ×1.6
    (C1), ×2.2 (C2) and ×2.2 (C3), all significant. The late premium rises ×2–3 under C2 and C3.
- **Recommendation: retain the current curve for now, and settle D4 and D2 first.** Is +3 HP of late capacity
  acceptable, or should combat be decoupled from wave capacity and Pressure? D2 also needs a sharper
  early-decisiveness test. If the review accepts the costs, pilot C1, with the requirements in the report's §9.
  Awaiting review.

## AO. M4.6c C1-versus-baseline shared-policy PPO pilot (2026-10-15)

Report: [M4.6c](m4/m4_6/m4_6c_report.md); protocol [m4_6c_preregistration.md](m4/m4_6/m4_6c_preregistration.md)
(frozen before training, three dated amendments). Code `lanegame/experiments/rl/m46c/`, tests `tests/test_m46c.py`;
one opt-in library argument (`train_shared(collector_cls=...)`, default bit-identical). `GameConfig.m45_calibrated()`
unchanged; C1 stays an experimental opt-in.

**Design.** The six 16.384M M4.5c finals were continued in both arms (six matched pairs), and the objective was
verified unchanged.
- **Budgets:** 4,096,000 transitions, then the pre-registered matched extension to 8,192,000, which fired.
- **Self-play:** 100% current-policy shared self-play.
- **Starts:** 25% of transitions from a shared gold-defined archive (natural, purchased-lead, zoning and freeze states;
  both roles).
- **Evaluation:** held-out cells, 50-turn horizon.

**Findings.**
- **The pre-adaptation effect replicates.** M4.6b's mid-game ×1.5–1.9 premium is reproduced before adaptation.
- **Criterion A.**
  - At 4.096M adaptation had shrunk it: the registered mid +8 G50 difference was +0.06 ± 0.14, ×1.21, not met. The
    extension fired.
  - At 8.192M it is +0.09 ± 0.11, ×1.32, 5/6 seeds: inconclusive. Mid +16 is +0.34 ± 0.17 (×1.76, 6/6).
- **Where the effect is.**
  - It lies in the mechanics: C1-trained and baseline-trained policies play C1 equally well.
  - There is no immediate early effect and no late effect at 50 turns.
- **Natural play** is within ±10%, apart from notes on Ward use and late deaths.
  - Late wave capacity is +4.2 HP, which only makes late guarding costlier.
  - Natural leads stay small and are never decisive.
- **The weaker side** keeps its options: no forced response beats its policy, and it does not die more.
- **Freezes** are unchanged; they are not an investment-curve question.
- **Head to head**, the adapted policies are interchangeable across curves.
- **Learning:** both arms improve over their starts (+0.4 to +0.5 at 8.192M). Both critics still undervalue early leads
  by about 1 resource.

**Recommendation.** Option 3: confirm the primary with six more matched pairs at 8.192M, then choose C1 if criterion A
is met. Either curve is coherent if the user decides now. Awaiting review.

## AP. M4.6d favourable freezes, the weaker side's decision and the specialist gap (2026-10-16)

Report: [M4.6d](m4/m4_6/m4_6d_report.md); protocol [m4_6d_preregistration.md](m4/m4_6/m4_6d_preregistration.md)
(frozen after the descriptive stage, four dated amendments and one dated exploratory follow-up). Code
`lanegame/experiments/rl/m46d/`, tests `tests/test_m46d.py`. Diagnostic only: no training, no rule or default change;
C1 stays provisional.

**Design.** The 8.192M M4.6c policies (C1 principal, baseline reference) as both seats.
- **Descriptive:** natural freeze episodes, classified.
- **Plan × plan matrix** of adaptive freezer and breaker plans from constructed freeze cells (levels 3–8, plausible
  and extreme strength / HP conditions), from a breaker decision map and from natural early freeze states.
- **Confirmation** on independent seeds.
- **Specialists:** gap verification and a switch / substitution decomposition.
- **Critic:** long no-bootstrap returns and a one-turn ranking test.

**Findings.**
- **Freezes.**
  - A freeze can be defended only while the breaker's wave clear is below the freezer's 8-HP wave: through level 8
    under C1, level 6 under the baseline.
  - In the best case, guarding beats passive keeping (+0.25 confirmed). It beats pushing or the policy only with an HP
    edge (+0.3 to +0.5 confirmed), never with both champions healthy.
  - Natural freezes are mostly the breaker's own push arriving with a worn freezer wave. There no defence beats the
    policy.
- **Defence.**
  - Healthy breakers should contest, and the policy does; every concession costs.
  - At low HP, reset and contest are close (the discovery lead did not confirm).
  - The breaker's policy is nowhere materially inferior against its own opponent. It under-commits against a freezer
    that trades every walk-up (+0.6 to +0.8 confirmed).
- **Specialists.** The gap persists on large constructed gaps (+1.15 baseline, +1.73 C1 cross-environment). On plausible
  advantages the specialists are worse (−0.40). The gap is many small trade-timing gains, not Recall reading.
- **Critic.**
  - Calibrated in natural states (slope 0.985).
  - Nearly blind to strength: a ×1.25 edge is valued +0.02 against a realized +0.63, a ×1.76 gap about +0.3 against
    +5 to +7. Earlier bootstrapped readings understated this threefold.
  - It ranks the freezer's options at a freeze at chance.
  - In one rare state family it prefers Safe correctly where the actor plays Yield.

**Recommendation.** Accept the abstraction for freezing and concession; no mechanics change. The next step is a no-PPO
value-representation probe (observation, representation or value training). Awaiting review.

## AQ. M4.6e observation, training exposure and value learnability (2026-10-16)

Report: [M4.6e](m4/m4_6/m4_6e_report.md); protocol [m4_6e_preregistration.md](m4/m4_6/m4_6e_preregistration.md)
(frozen before any supervised fit; one dated post-hoc sensitivity amendment). Code `lanegame/experiments/rl/m46e/`,
tests `tests/test_m46e.py`, tables `results/m46e/report.md`. Diagnostic only: no PPO or actor training; no rule,
observation, reward or default change; C1 stays provisional.

**Design.** The six 8.192M C1 general policies, each playing itself.
- **Q1.** Controlled single-feature state pairs; decode, mirror and injectivity checks on 5,915 reachable states;
  checkpoint and scale compatibility; trunk probes; value-versus-policy gradient shares on training-like batches.
- **Q2.** The training runs' recorded start counts and the archive's composition (direct); training occupancy
  reconstructed from frozen snapshots (an estimate).
- **Q3.** Policy-specific no-bootstrap G300 labels: natural games and 679 paired variants of 200 training-archive
  states; an independent 800-turn tail check.
- **Q4.** The existing critic, a ridge readout of its frozen trunk, a fresh value network and the same with
  observation-derived strength features, under grouped held-out splits plus strength and family hold-outs; tested on
  the M4.6d states and on the M4.6d decision rankings.

**Findings.**
- **Observation.**
  - Exact, lossless, mirror-consistent, injective; strength is present at an ordinary scale (the duel ratio is an exact
    function of two entries).
  - No implementation defect.
  - The critic moves +0.03 for 8 invested resources, about 0 for a level, and ±0.3 to 0.5 for HP or wave changes.
- **Exposure.**
  - Per 8.192M run: plausible early edges common (≈ 137,000 joint turns); large gaps (duel ≥ 2) ≈ 1,300; extreme gaps
    (duel ≥ 3, the M4.6d ×1.76 cells) none, and the archive itself contains none.
  - Favourable freezes with a damaged breaker ≈ 1,400; the weak-side Safe / Yield window ≈ 18,000.
- **Labels.** G300 is within ±0.05 of G800 on the problematic constructed cells.
- **Learnability.**
  - A fresh network on the unchanged observation recovers 71–87% of the strength premiums (critic 1–2%) and 84% of the
    explainable variance at constructed states (critic 32%).
  - Explicit features help extrapolation beyond the trained strength range.
  - A linear readout of the trained trunk recovers 2–3%: its last layer keeps HP (R² 0.92) but has discarded strength
    (R² 0.18; first layer 0.75; observation 1.00).
  - The value term is under 1% of the trunk's gradient.
- **Decisions.** Better values do not materially improve the ranking of the M4.6d decisions (freezer's Spearman 0.09 →
  0.14–0.17), and the weak-side Safe choice is already ranked correctly by the critic.

**Recommendation.** No learning pilot and no fix. Treat the critic's strength blindness as a documented limitation
(evaluate strength-gap states with rollouts) and proceed to M4 closeout. The named candidate, if the question is ever
reopened, is a value function with its own representation. Awaiting review.

## AR. M4.6f separate-critic continuation pilot (2026-10-16)

Report: [M4.6f](m4/m4_6/m4_6f_report.md); protocol [m4_6f_preregistration.md](m4/m4_6/m4_6f_preregistration.md)
(frozen after the smoke test, before any pilot run; two dated amendments: a powered head to head and an exploratory
economy profile). Code `lanegame/experiments/rl/m46f/` and the opt-in `lanegame.rl.SeparateCriticNet`; tests
`tests/test_m46f.py`; tables `results/m46f/report.md`. No rule, observation, reward, start distribution, PPO setting or
default changed; C1 stays provisional. The user authorized this pilot after M4.6e had recommended none.

**Design.**
- **Arms.** Six matched seed pairs continuing the M4.6c 8.192M C1 policies:
  - control: the unchanged shared network;
  - intervention: the same actor, with the critic trunk a copy of the shared trunk.
- **Equal start.** Initial probabilities and values are identical to the start's, which is verified exactly.
- **Identical PPO.** The same PPO, start distribution and seeds; a fresh Adam with lr 3·10⁻⁴ annealed to 0 in both
  arms.
- **Budget.** 4.096M transitions, then the pre-registered extension to 8.192M, which fired. The extension consists
  of new runs from the same start with their own schedule, not a continuation of the 4.096M runs.
- **Evaluation.**
  - Value: each critic against its own policy's returns, on the M4.6d states and seeds.
  - Behaviour: rollouts only, as 2 × 2 round robins within a seed on common random numbers (head to head, strength
    conversion and defence, freezes, weak-side Recall windows, natural states) plus fixed-opponent guards.

**Findings (8.192M).**
- **Value.** The separate critic learns part of the strength value (about a third), and the shared critic does not
  within the tested continuations:
  - plausible-edge recovery 0.45 vs 0.21 (+0.24 ± 0.20, 6/6);
  - pure edges 0.31 vs 0.02;
  - large gaps 0.32 vs 0.03;
  - log duel decodable from the value head's input, R² 0.64 vs 0.17;
  - still rising from 4.096M; natural-state calibration as good or better.
  - It remains nearly blind to level. Neither continuation of the shared network (4.096M or 8.192M) changed the
    shared critic's strength recovery.
- **Behaviour.** No difference beyond noise in strength conversion, defence, freezes, weak-side Recall windows or
  natural states, at either budget.
  - Head to head: no material difference (+0.06 ± 0.23).
  - The powered head to head (+0.13 ± 0.09, 6/6) and the comparison against other seeds' starts (+0.34 ± 0.29)
    suggest a small general edge at 8.192M only, not established.
- **Cost and health.** −4% throughput; every health check passes; the global gradient clip does not couple the arms.
- **Classification.** Pre-registered outcome 4: V1 was narrowly short of +0.25 and the 0.5 floor. Substantively,
  better values without a behavioural benefit demonstrated within the tested budget.

**Recommendation.** Accept the limitation, close the investigation and proceed to M4 closeout. Record the separate
critic as a verified opt-in option for M5 (a setup decision for closeout, not a strategic fix). No further critic
pilot. Awaiting review.

*Wording of this section corrected at the M4 closeout (§AS; the
[closeout report](m4/m4_closeout_report.md) §2 lists every correction).*
- The two budgets are separate continuations, not one 12M run.
- The arms diverged after the first update.
- The intervention changes sharing, capacity and optimizer scaling together.
- "No difference" means no material improvement demonstrated within the tested budget.

*The results, thresholds and classification are unchanged.*

## AS. M4 closeout review (2026-10-16)

Report: [M4 closeout](m4/m4_closeout_report.md). Consolidation only: no training or new experiment, and no rule,
parameter, objective, default or checkpoint change.

**Checks.**
- Read-only checks of the M4.6c and M4.6f training lineage, divergence and scorecard records.
- The full regression suite on the current working tree: 1,444 passed, 4 skipped, 2 xfailed.

**Recommendation: ready for M5 with documented limitations.** The evidence supports closing M4. The status stays "in
review" until the user decides the setup.

**Strategic acceptance** (report §4; classes relative to the tested alternatives).
- **Used:**
  - state-conditioned wave management and Shove;
  - Recall timing;
  - contesting freezes when healthy;
  - the weak side's choices;
  - conversion of plausible early edges (no tested alternative beats the policy; whether more is available is open).
- **Missed:**
  - guarding in the narrow favourable-freeze window (+0.3 to +0.5, 0.03% of training turns);
  - committing against an unfamiliar guard;
  - Safe at weak-side Recall windows (0.4% of turns);
  - large gaps (a generalization diagnostic);
  - unfamiliar opponent styles.
- **Weak under the mechanics:**
  - freezing outside the early HP-edge window;
  - late leads;
  - Slow Push in most states;
  - the trade decision in equal states;
  - Wards.
- **No limitation blocks M5.** Two would, if the research intent required them: freezes that hold against competent
  breakers, and leads that matter late. Both are mechanics changes the user paused.

**Configuration (recommendations awaiting the user).**
- **Environment.** The baseline `m45_calibrated()` remains the supported configuration. C1 stays a provisional opt-in:
  neither pre-registered adoption test was met, its effect sits in constructed mid-game lead states, and it changes no
  limitation that matters to M5.
- **Network.** The shared `PolicyValueNet` remains the default. `SeparateCriticNet` is a verified opt-in and not an M5
  factor.
- **Critic values** are not evidence in strength-gap states under either network.
- **M5 training setup (proposal).**
  - Fresh initialization.
  - The M4.6c archive start mixture, paired by seed with the M4.5c s0-only finals.
  - The M4.5c PPO settings.
  - 6 × 32.768M.

**Corrections** (report §2): the five M4.6e–f interpretation overstatements, plus two summary phrasings ("fixed much of
the value error", "never does").

**M5** (report §8).
- The roadmap's commitments are kept.
- Underspecified points are named:
  - the objective;
  - the success criteria;
  - "adaptive opponents", which has no implementation;
  - the start distribution;
  - M4.5c's unassigned follow-ups.
- A concrete scope is proposed, with success evidence S1–S6.
- First task: **M5a**, the pre-registration and readiness check: assembled drivers, an instrument check on the M4.5c
  finals, smoke and throughput, a frozen protocol; about 1–2 h of compute.

**Decisions for the user.**
1. C1 or the baseline.
2. The meaning of "adaptive opponents".
3. Re-confirm the accepted freeze and progression abstractions.

## AT. M4.7 early-lane horizon study (2026-10-16)

Report: [M4.7 horizon study](m4/m4_7/m4_7_horizon_report.md); pre-registration:
[m4_7_preregistration.md](m4/m4_7/m4_7_preregistration.md); tables: `results/m47/report.md`; code:
`lanegame/experiments/rl/m47/`; tests: `tests/test_m47.py` (11).

**Question.** The user asked whether the conclusions and learned strategies change when the focus is the laning phase
before about level 14, and whether a shorter effective training horizon would fit that focus better. The user accepts
the documented simplifications; "before about level 14" is a modelling preference, not a claim about League timings.

**Stage 1: what happens before level 14** (the matched M4.6c 8.192M baseline and C1 policies).
- **The phase.**
  - Level 14 at turn 94 (p10–p90 90–99) in both curves and across the lineage, so the lane phase is turns 0–93.
  - It is 17% of training transitions (reconstructed occupancy).
  - It carries 61% of the γ 0.99 weight from s0; a lane-phase decision has 36% of its weight inside the phase.
  - Kills, plates, Pressure and freezes are 1.4–2 times as frequent there as later; income per turn is the same.
- **Baseline and C1** are indistinguishable in timeline, decisions, income, Recalls and freezes. The exception: C1's
  early earned leads persist from turn 50 to 94 (slope +0.04 against −0.20). Most of C1's extra value for an early
  lead is paid after level 14.
- **Delayed rewards.**
  - Early leads keep about 90% (level 4), 75% (level 8) and half (level 11) of their premium inside the lane phase.
  - No tested alternative to the policy in natural early states, and no freeze plan, changes sign or order under
    H 80, 60 or 40.
  - After-L14 outcomes are uncorrelated with lane-phase outcomes in natural play.

**Stage 2: the matched pilot.**
- **Design.** Pre-registered (SHA-256 recorded, one dated amendment); the user approved the budget.
  - 2 seeds × 4.096M continuations of `ext/baseline:s`: γ 0.99 against γ 1 − 1/60 (H 60), the only difference.
  - Checks: smoke 66/66, training 50/50.
- **Head to head,** the short arm's lane-phase lead: +0.26 ± 0.17 and −0.09 ± 0.17.
- **Against fixed opponents** (paired): −0.69 and −0.36 ± 0.21 under γ 0.99, mostly after the lane phase and against
  scripted opponents.
- **Decisions changed in both seeds:** more freeze holding (+1.6 points of seat-turns), less Hold, more conceding
  when frozen on, fewer and lower-HP Recalls after the lane phase.
- **Pre-registered outcome 5 (regression).** It would be 6 (inconclusive) under the first code's unpaired interval,
  corrected to the protocol's paired interval after the results were seen; both are reported.

**Recommendation.** Keep γ 0.99 for M5 and analyse the lane phase with elapsed-turn windows (turns 0–93). Do not
adopt a new discount or investment curve. Not tested: a from-scratch short-horizon campaign, H 80, a short-horizon C1.

**Future work (described only).**
- Training occupancy weighted toward the lane phase (start distribution or game lengths, keeping the objective).
- Other ending mechanisms (none recommended).
- Bayesian opponent modelling, pending.

**Decisions for the user.**
1. Keep γ 0.99 (recommended), or pursue a shorter horizon further.
2. Whether to test lane-weighted training occupancy before M5.
3. The scope and placement of Bayesian opponent modelling.

## AU. M4.8 lane-exit evaluation (2026-10-17)

Report: [M4.8 lane-exit evaluation](m4/m4_8/m4_8_lane_exit_report.md); protocol:
[m4_8_protocol.md](m4/m4_8/m4_8_protocol.md) (frozen before any evaluation run, one dated amendment); tables:
`results/m48/report.md`; code: `lanegame/diagnostics/lane_exit.py` (the reusable readout) and
`lanegame/experiments/rl/m48/`; tests: `tests/test_m48.py` (18).

**Decisions that led to it** (the user, 2026-10-17; summary above):
- keep the continuing objective with no termination around level 14;
- drop the ending experiment proposed after M4.7;
- measure lane outcomes at an evaluation checkpoint with short repeated continuations, keeping XP, HP and position
  visible separately;
- validate that method and apply it to the matched baseline/C1 policies before configuration recommendations.

Evaluation only: no training, and no rule, reward, objective, default or checkpoint change.

**The readout.** At a checkpoint, E_h = D0 + E[Σ_{t<h} r_t]:
- D0 is the earned-gold lead (invested + unspent; spending is not earning);
- the expectation is over K continuations by the policies that produced the state, on common random numbers;
- undiscounted, with no critic tail; distinct from the discounted G_h and from critic values.

**Stage 1: validation.**
- **Exact:** accounting in every continuation; mirrored continuations negate exactly (12,288); the 2,304 natural
  checkpoints replay their records bit for bit.
- **Cases** (469 checkpoint-selected states, K 128, folded to the focal seat), Δ10:
  - returner facing a due opponent: +2.55 ± 0.19, recorded +2.70;
  - dead with a wave at its tower: −0.20 at h 5, then +0.91;
  - frozen on: −1.08;
  - stable: ≈ 0;
  - gold leader trailing in XP or HP: −1.73;
  - every adjustment is accounted for by farm, absence, bounty and plates as the mechanism predicts.
- **Controlled:** with D0 identical, an already-completed reset is worth +0.87 ± 0.18 at h 10 and +0.62 ± 0.13 at h 20.
- **Calibration** against the games' own continuations (576 natural states): slopes 1.00, 0.99, 0.99 at h 5, 10, 20;
  residuals within ±0.15. E_20 predicts the recorded lead at turns 150 and 200 better than D0.
- **Caveat:** gold leaders trailing in XP or HP lost more in their recorded continuations than predicted (z ≈ 2.8 in the
  validation cases; the same direction, not significant, in the natural states). The cause is not identified; no
  overall bias.
- **Classification (protocol §7): useful. Default window (protocol §6): E_20.** E_5 and E_10 fail coverage, because
  due resets are 27% and 60% resolved by then; E_20 resolves 95%, has no excess drift and a median ±0.39.

**Stage 2: the matched M4.6c baseline/C1 policies at turn 94** (576 states; 84, 104 and 50 as checks).
- **How often:** about two thirds of states carry a material, certain adjustment at h 20 (64% baseline, 66% C1).
  - The SD of the true adjustment is 1.16, against a mean |D0| of 3.47.
  - The leader flips in 4.3% of states, 16% of those with |D0| < 0.5.
- **Why:**
  - a seat freshly back (23% of states, +1.16);
  - an asymmetric due reset (16%, +0.70);
  - a death (−1.24 at h 5, about 0 by h 20);
  - reset status, wave and HP together explain about 80% of the variance (cross-validated); D0 itself 2%.
- **Stable:**
  - flat from h 20 to 40;
  - another seed's policy or the other curve's package correlates 0.96–0.98;
  - the same picture at turns 84 and 104.
- **Baseline vs C1:** no difference on any readout:
  - distributions: |E_20| +0.31 ± 0.46;
  - conversion after turn 94: +0.00 ± 0.06;
  - matched states continued by both packages: the leader gains +0.02 ± 0.04 under C1;
  - the value of a completed reset: +0.06 ± 0.20.
- **Earlier conclusions re-checked:**
  - M4.7's |lead| comparison at 94 is unchanged.
  - M4.7's 50 → 94 conversion slopes keep their C1 − baseline contrast (+0.25 ± 0.13). Their levels shift: the
    baseline goes −0.20 → −0.10 ± 0.10, and C1 goes +0.04 → +0.15 ± 0.09. This corrects M4.7's "baseline leads
    shrink by a fifth; C1's persist" to "baseline leads change little; C1's grow somewhat"; the M4.7 report is not
    edited.
  - The M4.7 pilot's primary (supplementary re-check, protocol amendment) goes from +0.26 / −0.09 raw to
    +0.17 ± 0.18 / −0.14 ± 0.17 with E_20, so no seed shows a lane-phase gain. Outcome 5 rests on the fixed-opponent
    regression and is unchanged.

**Recommendations** (awaiting the user; nothing adopted):
- Present lane-phase outcomes as D0 → E_20 ± 95% with the non-gold panel (levels and XP, HP, absence, turns since
  return, stocks).
- Give an overall verdict only when gold, XP and HP agree. They disagree in 21% of natural turn-94 states.
- K 128 per state for state-level claims, K 8–16 for population means.
- The baseline stays the recommended configuration.
- γ 0.99 and the continuing objective are kept.

**Decisions for the user.**
1. Adopt the lane-exit presentation as the standard lane-phase economic readout (recommended).
2. Baseline or C1 for M5 (unchanged by M4.8; baseline recommended).
3. The scope of the Bayesian opponent-modelling design.

**Next:** step 3, the Bayesian opponent-modelling design (approved direction; not designed). Then the matched
opponent-modelling pilot (the same opponent distribution in both arms), the final M5 campaign and M6.

## AV. OM1 opponent-modelling feasibility (2026-10-17)

Report: [OM1 opponent-modelling feasibility](opponent_modelling/om1_feasibility_report.md); protocol:
[om1_protocol.md](opponent_modelling/om1_protocol.md) (frozen before any game, no amendment); tables:
`results/om1/report.md`; code: `lanegame/experiments/rl/om1/`; tests: `tests/test_om1.py` (7).

**Why.** Step 3 of the agreed sequence is to design Bayesian opponent modelling. Before designing it, the user agreed
to test, with existing policies only, whether the lane phase reveals the opponent's type early enough
(identifiability), and whether knowing it would change what to play (value of information). Evaluation only: no
training, and no network, observation, rule, reward, objective or default change.

**Design.**
- **Focal policy:** the adopted baseline's `ext/baseline:s`, s = 0..5.
- **Library:** 15 types with a uniform prior:
  - the nine M4.5c scripted references;
  - six learned policies (peer, attacker, freeze and gap specialists, short-horizon, early).
- **Belief:** exact Bayes from the opponent's observed action (an upper bound on information), with a coarse variant
  that hides the wave choice.
- **Responders:** for each type, the best existing responder r\*, selected on separate games.
- **Contrasts:** an oracle that knows the type, and a Bayesian switcher that plays π until its belief reaches 0.9,
  then plays r\*.
- **Value:** the lead at turn 114 (the lane-exit readout E_20 at turn 94), paired seats, common random numbers.
- **Robustness:** seven out-of-library opponents.

**Results.**
- **Identification:**
  - Decision-relevant identification (the believed type has the true type's best responder) is 96% after 10 observed
    turns, 99% after 20 and 100% after 40.
  - Hiding the wave choice delays it by 5–10 turns.
  - The Shove variants stay confused with each other, at no cost, since they share a responder.
- **Value:** the switcher beats π by **+0.90 ± 0.27** (six seeds, all positive), +1.52 ± 0.30 at turn 150; the oracle
  +1.09 ± 0.44. No fixed existing policy beats π on average.
- **Where:**
  - scripted styles: +1.36 ± 0.43;
  - learned opponents: +0.21 ± 0.11 (gains against the attacker and the freezer; a loss of −0.61 ± 0.36 against the
    peer, where the selected responder was a false positive).
- **Robustness:**
  - New instances of the learned families: no harm.
  - Per-turn mixtures of two scripted styles: **−0.83 and −1.45**. The belief settles on one of the two styles and the
    switcher plays the wrong specialist half the time.
  - Pooled: −0.23 ± 0.24, inside the −0.25 tolerance only narrowly.

**Classification (protocol §4): outcome 1, feasible and worthwhile.**

**Proposals for the design** (step 3; not adopted):
1. Act on the posterior mass of a response, not of a single type. The switcher lost most of the oracle's value
   against look-alike types.
2. Add a misspecification ("none of the above") fallback, and train or evaluate against mixed and style-switching
   opponents.
3. Use a belief-conditioned policy trained against the library, rather than switching between fixed policies.
4. Decide how the opponent's action is observed: inferred from state changes, or an opt-in observation field (new
   checkpoints for that variant).

**Limitations:**
- in-library optimism;
- an upper-bound signal;
- existing responders only (the oracle is a lower bound);
- a uniform prior that weights scripted styles heavily;
- six focal seeds.

**Decisions for the user:** the scope of the design (which opponent distribution matters for M5; the proposals
above).

## AW. OM2 opponent-modelling design, implementation and matched pilot (2026-10-17)

Report: [om2_pilot_report.md](opponent_modelling/om2_pilot_report.md); design:
[om2_design.md](opponent_modelling/om2_design.md) (proposed, in review); pre-registration:
[om2_preregistration.md](opponent_modelling/om2_preregistration.md) (frozen, SHA-256 in
`results/om2/preregistration.json`, no amendment); tables: `results/om2/report.md`; code: `lanegame/opponent/`,
`lanegame/experiments/rl/om2/`; tests: `tests/test_om2.py` (11).

**Decisions it implements (the user, 2026-10-17, after OM1):**
- **Scope C, "both, robustness first":** the agent models recognizable archetypes and competent learned opponents and
  trains against mixed and style-switching opponents. A gain counts only with no loss against competent or unfamiliar
  opponents.
- **Decision 2:** the opponent's previous action is an opt-in observation field.

**Design (proposed).**
- **Observation:** `GameConfig.obs_last_action` appends 13 entries for the opponent's previous action (103 with
  `m45_calibrated()`). No rule changes; mathematical reference §15.4.
- **Belief:** a posterior over 11 library types and two catch-alls, with forgetting λ = 0.97 per turn (about a
  33-turn memory) and turns where the opponent can only be Absent skipped.
  - Library types: the archetypes g_farm, g_safe, g_bully, g_shove, fz_guard_sh and g_shove_ward (ε = 0.05 tremble),
    and five learned policies with exact likelihoods.
  - Catch-alls: "mixed" (a different library style each turn) and "unknown" (uniform). The "mixed" catch-all was added
    during implementation, when the test showed that a uniform catch-all does not catch per-turn mixtures (design §3).
- **Policy:** the shared `PolicyValueNet` with 103 + 13 = 116 inputs, widened from `ext/baseline:s` with zero weights.
  Both arms start as exactly that policy.
- **Training:** the M4.6c continuation settings. Half the games are self-play; the other half face one of 15 fixed
  opponents: the 11 library types, 2 per-turn mixtures and 2 switchers. Only the learner trains in those games.

**Implementation** (every piece opt-in; no default, preset, objective, digest pin or checkpoint changed):
- the observation field, with `LaneEnv.last_actions` and `Snapshot.last_actions` (restorable, mirrored);
- the belief module: types, `Belief`, batched likelihoods;
- the collector hooks `augment` / `observe_step` in `SharedCollector`. Default training was checked bit-identical to
  the previous trainer, and a test pins the digest;
- `widen_inputs`;
- the OM2 collector, which batches belief updates across environments;
- the evaluation harness: paired seats, common random numbers, belief traces, belief-swap probes;
- the analysis with the pre-registered classification.

**Readiness:**
- 17 of 17 smoke checks pass:
  - both arms start from the same rollout;
  - the belief varies in the treatment and is exactly constant in the control;
  - library, mixture and switcher opponents all appear;
  - the re-run is bit-identical.
- Throughput with 3 concurrent runs: control about 2,650 transitions/s, treatment about 1,580.
- In smoke games the belief identifies g_bully by turn 40 and g_safe within 2 turns.
- The control's belief-swap probe distance is exactly 0 by construction: its 13 constant inputs keep identical weight
  columns.

**The frozen pilot:**
- **Arms:** control (belief held at the prior) against treatment (the belief); same network, data stream and seeds.
- **Size:** 3 seed pairs × 4,096,000 transitions, about 1.25–1.5 h of training plus 15–20 min of evaluation.
- **Value:** the lead at turn 114 (E_20 at the lane exit), each arm against the same opponents on the same games.
- **Families:**
  - E_in: the archetypes;
  - E_out: held-out scripted variants (descriptive);
  - R1: competent learned opponents, namely the other seeds' baseline policies and held-out instances;
  - R2: held-out mixtures and switchers;
  - R3: head to head.
- **Outcomes:**
  1. a gain against the archetypes (≥ +0.25, positive in every seed) and no loss: modelling pays and is safe;
  2. a gain with a loss (≤ −0.25, negative in every seed) in R1, R2 or R3: pays but costs;
  3. no gain and no loss: no material gain;
  4. no gain with a loss: harmful.
- No extension rule.

**The pilot ran** (compute approved by the user on 2026-10-17): 6 runs in 72 minutes, all complete; within each
seed both arms started from the same weights and processed the same transitions. Evaluation took 69 cells and 11
minutes.

**Results (treatment − control, lead at turn 114; seeds as units, t intervals with 2 df):**

| family | seeds 0 / 1 / 2 | mean ± 95% |
|---|---|---|
| E_in: the 6 archetypes | +2.57 / +1.25 / +2.26 | **+2.03 ± 1.72** |
| E_out: held-out scripted (descriptive) | +2.82 / +0.64 / +2.06 | +1.84 ± 2.74 |
| R1: competent learned | +0.05 / +0.58 / +0.65 | +0.43 ± 0.81 |
| R2: held-out mixtures and switchers | +1.34 / +0.36 / +1.15 | +0.95 ± 1.29 |
| R3: head to head | −0.33 / +0.79 / +0.50 | +0.32 ± 1.45 |

**Classification: outcome 1, modelling pays and is safe.**

- **Mechanism:**
  - 5 of 6 archetypes are identified within 10 turns in at least 98% of games; fz_guard_sh is shared with its
    look-alike g_shove, and the treatment still gains +2.68 against it.
  - Competent opponents are read as a spread over the learned types.
  - Switches are followed in 10–60 turns, sometimes through "mixed".
  - The treatment's critic explains 0.93–0.94 of the return variance, against 0.66–0.71 for the control.
  - Belief-swap probes move the trained policy by a total variation of 0.14–0.32.
- **The one negative opponent** is the bully/farm per-turn mixture (−0.52; the two styles agree on most turns).
- **Exploratory reference (not pre-registered):** the common start π_s on the same games.
  - Against the archetypes: control +5.1 over π_s, treatment +7.2. Training against the styles does most of the
    exploiting; the belief adds +2.0.
  - Against competent opponents: control −0.59 (every seed negative), treatment −0.16 ± 0.30.
  - Head to head against π_s: both about 0.
  - "Safe" therefore means no worse than the equally trained control and close to the start. The belief
    largely protects competence that the 50% fixed-opponent mix otherwise erodes.
- **Limitations:**
  - three seed pairs;
  - library-favourable tests;
  - no deceptive or dedicated attacker, the main untested risk given the slow switch tracking;
  - no arm without the last-action field;
  - evaluation from s0 only.

**Recommendation and decision for the user:**
- Carry Bayesian opponent modelling into the M5 design in the tested form: `obs_last_action`, the 13-feature
  belief, and self-play plus library opponents.
- Make dedicated attackers, an imitate-then-switch opponent and the competent-play comparison against pure
  self-play part of M5's evaluation.
- The fixed-opponent share, the library's growth and continuing versus training from scratch remain M5 design
  choices.

## AX. M5a campaign design, pre-registration and readiness (2026-10-17)

Report: [m5a_readiness_report.md](m5/m5a_readiness_report.md); design: [m5_design.md](m5/m5_design.md) (proposed,
awaiting decisions); pre-registration: [m5_preregistration.md](m5/m5_preregistration.md) (frozen, awaiting campaign
authorization; SHA-256 `d2424022bd07535f…af7304` in `results/m5/preregistration.json`); readiness protocol:
[m5a_readiness_protocol.md](m5/m5a_readiness_protocol.md) (recorded before the first check); code:
`lanegame/experiments/rl/m5/`; tests: `tests/test_m5.py` (20); records: `results/m5/`.

**The brief (the user, 2026-10-17).**
- M5 includes opponent modelling and compares the whole package with continued pure self-play.
- Kept fixed:
  - the baseline;
  - the continuing γ 0.99 objective with bootstrapped truncation and no level-14 termination;
  - the shared critic;
  - D0 → E_20 reporting;
  - rollout evidence;
  - no new mechanics, curves, discounts, critics or algorithms.
- The starting proposal: two arms, six pairs continuing `ext/baseline:s`, OM2's settings, a fixed library, 32.768M per
  run with a 16.384M matched checkpoint.

**Design (proposed).**
- **Arm A** is OM2's treatment collector, imported unchanged.
- **Arm B** is the M4.6c continuation. It is verified to be 100% current-policy shared self-play: the snapshot pool
  exists but is never played.
- **The primary contrast** A − B is the package effect: information and opponent distribution together.
- **M5b:**
  - health checks, curves, history and the final-vs-75% probe;
  - restart rules with an exact-prefix re-run;
  - no extension rule.
- **M5c:** the full M4.5c battery with the M4.6b corrections, and the seven M4 limitation families. Replication needs
  at least 5 of 6 seeds within 0.2.
- **M5d:**
  - style, misspecified, imitate-then-switch, competent-historical and competent-final families;
  - head to head, the 500-turn objective and the midpoints;
  - parity-start attackers in the om input format, and an adaptation-exploitation test against the om final with its
    belief frozen at typical self-play values.
- **Statistics:**
  - seeds as units, t intervals with the game-level and seed-level variance kept apart;
  - P1 (gain ≥ +0.25), and P2a / P2b by non-inferiority with δ = 0.25 (inconclusive never reported as preserved);
  - headline outcomes 1–5, style gains and competence never netted;
  - Holm for the secondary claims.

**Implementation.**
- The package covers:
  - training and manifests;
  - the history-dependent evaluation policy `OMPolicy`;
  - the M5b–M5d instruments for both formats;
  - attackers;
  - statistics;
  - readiness;
  - gated stages (`--confirm-m5-train` / `-eval` / `-attackers`).
- One opt-in library change: the `observe_turn` hook in the four shared rollout loops, with `inner_policies` in six plan
  wrappers and `value_at` in the critic bootstrap. No change for existing policies.

**Readiness (11 of 11 parts pass; about 20 minutes of experimental compute).**
- Lineage verified: M4.5c fresh 16.384M → M4.6c 8.192M → M5 32.768M, 57.344M cumulative over three schedules.
- The seed block 70,000–70,999 is disjoint from every other study's roots.
- The widened start equals `ext/baseline:s` bit for bit in batched evaluation.
- Pins pass, and stored M4.5c, M4.6c, M4.6f (C1) and OM2 records reproduce exactly through the hooked loops.
- Both smokes re-run bit-identical.
- The first 200 game starts are identical in both arms.
- One full om rollout's 2,863 training inputs and 14 truncation bootstraps were rebuilt independently by the evaluation
  policy: the state and previous action exact, the belief within 2.9·10⁻⁶. The first run's bit-exact belief comparison
  failed on float32 rounding and is reported.
- Fixed-opponent games are 51% of games and 36–37% of learner transitions.
- Six workers is the efficient count (om aggregate +3% at 8).
- Every instrument ran at reduced samples, with steady-state costs measured.

**Budget** (6 workers):
- training 9.4–11.8 h (om runs 5.8–8.3 h, sp 3.6 h);
- evaluation 13.7 h (the battery 9.3 h: om turns cost about 3× with the belief);
- attackers 10.3 h (18 × a quarter budget) or 20.6 h (half);
- about 34–36 h in all.

**Evidence-backed proposals** (design §11; none adopted):
1. quarter-budget attackers;
2. the adaptation test;
3. the full battery;
4. six workers with the om runs first;
5. the common start's natural states for the cross-arm cells.

Six pairs are kept, with an honest resolution statement: under OM2's variance a near-zero competence effect is
conclusive only 42% of the time (ten pairs 71%).

**Decisions for the user** (design §12):
1. Authorize training.
2. The seed count.
3. The attacker budget.
4. The adaptation test.
5. The battery size.
6. The protocol's rules.

**M5 campaign status: not launched, awaiting approval.**

## AY. M5a correction pass, M5b training and the initial M5d evaluations (2026-10-17 to 2026-10-18)

Report: [m5b_initial_report.md](m5/m5b_initial_report.md); amendment: [m5_amendment_1.md](m5/m5_amendment_1.md)
(pre-registration version 1, `21e35e07…`, recorded before training; the original `d2424022…` archived); design:
[m5_design.md](m5/m5_design.md); tables: `results/m5/report.md`; records: `results/m5/` (`train_main.json`,
`analysis_main.json`, `monitoring_main.json`, `corrections_readiness.json`, `progress_main.jsonl`, the stage manifests);
tests: `tests/test_m5_corrections.py` (24) and `tests/test_m5.py` (20).

**The brief (the user, 2026-10-17).**
- Authorized: a bounded correction pass; main training (12 runs × 32.768M); the training-health review; monitoring and
  history; the opponent-family and head-to-head evaluations with their midpoint and long-game comparisons; analysis
  and documentation.
- Not authorized: the battery, the cells, every attacker stage, extra seeds or extensions, and any change of
  mechanics, reward, discount or architecture.
- The om_table attackers became an optional follow-up; the core attacker stage is 12 quarter-budget attackers.

**The correction pass (amendment 1).** It fixed the review's eight findings before any training:
- the headline: an inconclusive P1 is no longer "no material gain" (new outcome 6), and outcome 4 needs a material gain
  ruled out;
- one fixed family of 15 one-sided hypotheses for S1–S6, with TOST for equivalence and an interim "secured" rule;
- S5a / S5b, the midpoint competence named as the historical-opponent comparison it is;
- six-seed completeness;
- the S8 analysis, with coverage and the horizon-validation path;
- runtime health monitoring through a read-only trainer hook (bit-identical, tested), the coordinator's failure rule
  and an evaluation gate with checkpoint identities;
- the core and optional attackers, with the cross-evaluation implemented;
- a verifiable amendment chain.

Nothing about the experiment changed. Checks A1–A4 passed: the monitored smokes are bit-identical to the stored M5a
runs, and the full suite gave 1,534 passed, 4 skipped (`--runslow`) and 2 xfailed.

**Training.** All 12 runs are complete and healthy: 138/138 checks, no runtime alarm, no technical failure, no
restart, 10.90 h (estimate 9.4–11.8 h). The om arm played 21.9M environment turns against sp's 16.4M and had 21.8M
self-play transitions against 32.8M.

**Monitoring.** Both arms first lost to their own start (the re-warm cost) and recovered.
- At the end, against the start (64 pairs × 500 turns): sp +0.94 ± 0.54, om +0.20 ± 0.68.
- Convergence probe: sp still improving (+0.59 ± 0.45, non-convergence reported as a limitation); om not shown to be
  (−0.08 ± 0.32).
- No cycling.

**M5d: outcome 2, pays at a competence cost.**
- **P1:** +9.26 ± 0.30, demonstrated, all six seeds +8.9 to +9.6. Against the library archetypes +12.2 and the held-out
  styles +6.4. Against conditional styles on a competent base there is no gain.
- **P2a:** −0.63 ± 0.35, a material loss. **P2b:** −0.44 ± 0.28, a material loss. Every seed is negative in both.
- **Secondary**, each claim determined under Holm while S6 is pending:
  - S1 misspecified opponents preserved (+5.20 ± 0.38);
  - material losses in S2, the long-horizon competence (−0.60), S3, the long head to head (−0.90), S4, the cross-seed
    head to head (−0.68), and S5b, the midpoint competence against historical opponents (−0.80);
  - S5a, the midpoint gain, demonstrated (+8.29).
- **The lane exit.** Against competent opponents the deficit is purely economic: D0 −0.48 → E_20 −0.65 (C_hist), with
  equal levels, XP, HP and absence. Against scripted styles the om arm leaves the lane ahead in gold (+9.7), level
  (+0.7), XP (+31) and kills.
- **Exploratory.** The competence gap is mostly improvement over the common start that pure self-play gained and the
  package did not (competent families: sp +0.87, om +0.25 with an interval including 0). After a scheduled switch to a
  competent policy the om arm loses about 0.4 in the first 20 turns (a possible short adaptation lag), and its belief
  moves to the catch-alls within 20 turns.

**Not established, and not implied by the headline:** strategic replication, the M4 limitation re-measurements and
exploitability.

**Next decision for the user:**
- the strategic battery and the limitation cells: the full design about 11.8 h, or with the reductions of design
  §10.1;
- the core attackers: 12 at a quarter budget, about 7.1 h;
- the optional om_table ablation.

## AZ. M5c strategic battery and limitation cells (2026-10-18 to 2026-10-19)

Report: [m5c_strategic_report.md](m5/m5c_strategic_report.md). Tables: `results/m5/report.md` (S8) and
`results/m5/battery/battery_om.md`, `battery_sp.md`. Records: `results/m5/strategic_main.json`,
`battery_extension.json`, `battery_validation_150_diagnostic.json`. Decision trail: `results/m5/decisions_m5c.tsv`.

**The decision (the user, 2026-10-18).** Run the strategic battery and the limitation cells, with no reduction named.
The full design ran.

**What ran.**
- A smoke on the real finals (B1, 14 of 14 checks) verified the never-executed 150-turn re-run path first.
- The full M4.5c battery on all 12 finals: 14,712 records, 8.73 h.
- The seven limitation families: 3,732 records, 2.11 h.
- The registered 150-turn re-run of the 51 (final, family) pairs that failed horizon validation: 5,296 records,
  5.58 h, run after a pre-set 12 h ceiling check.

**Results.**
- **Replication.** Four of five M4.5c verdicts replicate in both arms (6 of 6 seeds): conditional wave management,
  conditional freezing and the reply to freezes, selective trading, and Recall timing.
  - Concession under threat does not replicate (om 3 of 6, sp 4 of 6). The failure is in the constructed zoning
    cells: a one-turn Shove beats the om policy by +0.17 ± 0.17 in all six seeds. The weaker-side lead cells hold
    in both arms.
  - The sp freezing verdict is fragile: three seeds sit within 0.03 of the margin.
- **Cross-arm limitation cells, om − sp from the same states:** strength use −0.53, freezes −0.26, Recall windows
  −0.42, natural states −0.54, large gaps −0.39. All or nearly all seeds are negative, equal and lead states alike.
- **Per arm.** The lead premium is unchanged and equal between the arms. No forced weaker-side response beats either
  policy. The favourable-freeze guard still beats both policies (+0.53 om, +0.24 sp, 6 of 6). The gap specialist
  still out-converts the sp finals (+1.03 ± 0.58).
- **Horizon validation.** 51 of 144 pairs failed: 29 on bias, 15 on slope, 7 on both.
  - An unregistered diagnostic on the validation subset finds 150 turns agree with 300 turns in all 51 failed pairs.
    It supports the re-run horizon but does not check the re-run records.
  - In families that passed, Q50 and G50 margins differ by 0.03 on average, so the critic does not drive the
    verdicts.
- **Caveats.**
  - The om final starts the cross-arm cells with its prior belief.
  - Only three gap specialists exist.
  - Both sp verdicts near the threshold (freezing, concession) are one seed from changing.

**Next decision for the user:** the 12 core attackers (recommended, about 7.1 h). The optional om_table ablation is
not recommended. M5e follows.

## BA. M5d core attackers and the M5e closeout (2026-10-19)

Report: [m5e_closeout_report.md](m5/m5e_closeout_report.md). Tables: `results/m5/report.md`. Records:
`results/m5/attack_check_main.json`, `analysis_main.json`, `attack_behaviour.json`, `traces.json`,
`corrections_readiness.json` part C1. Reproducibility: `results/m5/reproducibility.md`. Decision trail:
`results/m5/decisions_m5e.tsv`.

**The decision (the user, 2026-10-19).** Run the 12 core attackers at the quarter budget, their registered evaluation
and the M5e closeout. The optional om_table ablation, extra seeds, attacker restarts for poor performance and
experiments on the cause of the competence loss were not authorized.

**Before the attackers.**
- An independent read-only audit found no defect that would bias S6 or the attacker training. It found five gaps in
  robustness and reporting, and all five were fixed. The attackers now run under the main training's §3 rules: a
  live health monitor, the coordinator, deterministic restarts with a prefix check, and a post-hoc check that gates
  the evaluation. The analysis gained the readouts §8 and §9 name: the discounted lead beside S6, each arm's lower
  bound, and transfer.
- Readiness part C1 ran the whole attacker path on the real finals at tiny sizes and passed 56 of 56. It covered the
  targets' SHA-256, initialization, the PPO settings, bit-identical re-runs with and without the monitor, an
  interrupted run refused, preserved and restarted with an identical prefix, and real attacker records read by S6
  and the report.
- The four `--runslow` tests passed, the first time they ran in M5.

**Results.**
- **Training.** 12 runs, 180 of 180 checks, no alarm or restart, 5.96 h (estimate 6.64 h). Evaluation took 0.52 h.
- **Lower bounds on exploitability** (the attacker's lead at turn 114 against its own target, six seeds):
  - against the om finals +2.96 [+0.32, +5.61];
  - against the sp finals +1.15 [+0.34, +1.96].
  Both are positive in every seed. The sp value matches M4.5c's +1.02.
- **S6** (registered, om − sp): +1.81 [−0.56, +4.19], inconclusive. The increase test has p = 0.053. The spread is
  between attackers (τ 2.22 against a game SE of 0.42), and one om seed contributes +6.18.
- **The Holm family is final** over 15 hypotheses. S1 preserved; S2, S3, S4, S5b material losses; S5a demonstrated;
  S6 inconclusive.
- **No transfer** (exploratory). Over its starting point, each attacker gains −0.95 (om attacker against sp:s), −0.48
  (sp attacker against om:s) and −0.92 (sp attacker against the start).
- **What changes** (descriptive). The om attackers yield and play Safe less, and hold and trade more. Their targets
  recall and die more often. The sp attackers shove less and trade more.

**Interpretation.** The attackers leave the exploitation-against-competence trade-off as M5d established it. They add
that the package does not make the agent robust to a dedicated attacker, since its lower bound is positive in every
seed.
Whether the package is easier to exploit than pure self-play remains unresolved at this budget.

**Recommendation for M6.** Present both arms as a trade-off, with the modelling package for recognizable, exploitable
styles and pure self-play for competent play. Use all six seeds for claims, and the seed-0 pair for demonstrations.

**Corrections to living summaries** (the M5c report stays unchanged):
- the om constructed-zoning gap is open (suggestive), not missed;
- both arms retain the same four findings, which is not the same choices;
- the cold-start caveat rests on the s0 evaluations.

**M5 is complete.** M6, presentation, demonstration and reproducibility, is next.

## BB. M6 plan: presentation, interactive demonstration and reproducibility (2026-10-19)

Plan: [m6_plan.md](m6/m6_plan.md). Status at the time: **planned, not started; in review.** Nothing was implemented,
the Streamlit viewer had not moved, and nothing was deployed. *Update 2026-10-21:* the user approved the plan with
Watch added to the initial release, and M6a has started (§BC).

**The direction (the user, 2026-10-19).**
- M6 builds a new web application that can later be deployed publicly. Its centre is a human who inspects the lane
  and picks an action each turn against a trained M5 policy, pure self-play or the opponent-modelling package.
- The opponent acts from the pre-turn state and history, and both actions resolve together.
- The Streamlit viewer is archived as legacy, not reworked.
- Python stays authoritative for mechanics, legality, inference and the belief.
- No new mechanics, rewards, objectives or training.
- Deployment is an intended capability, not an authorization.

**What the inspection found.**
- **Reusable as is.** `LaneEnv` (its own random generator, exactly 10 chance uniforms per turn, illegal codes
  refused before any change), the M5 policy loaders and `OMPolicy` (a pure `decide`, the belief updated only after
  both codes are fixed), and the 15 verified recordings. Also `lanegame/viewer/session.py`, which has no UI code and
  provides frames, summaries and event text.
- **Per-tick and per-turn facts.** The engine records packets, fights, crashes, thinning and minion fire per tick.
  Trades, tower shots on champions and ganks exist per turn only, so an animation must not place them at a tick.
- **Serve-time dependencies.** `om:0` needs the 5 library networks besides its own checkpoint. All 7 networks are
  git-ignored and about 1.1 MB together. `results/m5/inventory.json` lists the library networks' SHA-256, but the
  key-checkpoint table of `reproducibility.md` does not.
- **Loading needs care.** `torch.load` unpickles, so the server checks file SHA-256 values against a manifest before
  loading. The policies draw their sampling uniforms for a fixed horizon, so the session sets it to its turn cap.
- **Measured cost on a busy machine.** About 1.2 ms per pure self-play turn and 2.7 ms per modelling turn, 22 KiB
  per session, about 250 MB per process and a 5 to 6 s cold start. Memory per process sets the hosting cost.
- **Repository gaps.** No license, no Dockerfile, no CI and no JavaScript toolchain. `.gitignore` ignores `docs/`, so
  the M5c and M5e reports and the plan itself are untracked. `reproducibility.md` was generated on an uncommitted
  tree.
- **Viewer imports.** 15 experiment modules and 7 test files import `lanegame.viewer`. The archive therefore moves
  only `app.py` and `render.py`, after a tag, together with the two tests that import `render.py`.
- **The session loop already works.** A read-only probe drove `LaneEnv` with both M5 policies through
  `crn_policy`, with `observe_turn` after each step, and reproduced all 150 action pairs and the payoff of the
  recorded `om:0` against `sp:0` game.
- **A fairness detail.** The weights are public, so a visitor who knew the opponent's sampling seed could compute
  its choice offline. The plan keeps that seed secret on the server until the session stops.

**The plan.**
- Five milestones, M6a to M6e ("M6 plan" above), with twelve correctness rules that the backend proves before any UI
  exists.
- An initial release centred on human-versus-policy play, with the verified replays and the findings.
- Three bundles (demo, evidence, training).
- One container with in-memory sessions, and static replays that work without inference.
- The findings presented with all six seeds, the seed-0 pair as the demonstration pair by M5's convention, and no
  claim about play against humans.

**Decisions for the user.**
- Approve the milestones and the initial release.
- The `docs/` line in `.gitignore`.
- Where the frontend lives.
- Later, a license before any bundle is published, then the provider, budget and deployment itself at the M6e
  checkpoint.

**Next.** On approval, the first task is the headless session core (the plan's §11).

## BC. M6a: the session core, the contracts and the prototypes (2026-10-21)

Reports: [m6a_technical_report.md](m6/m6a_technical_report.md) (the backend) and
[m6a_decision_record.md](m6/m6a_decision_record.md) (requirements, choices, the prototypes and the final design).
Status: **completed (2026-10-21).** The backend is verified, and the user approved the interface design. No mechanic,
reward, objective, network or stored result changed, and no training ran.

**The brief (the user, 2026-10-21; approved direction).** Two primary modes in the initial release, Play and Watch,
with Watch moved from "later" into it. A Chess.com-style turn history shared by both. Explicit frontend clarity
requirements. React with TypeScript and Vite in `web/` over FastAPI. `docs/` tracked again. Two or three clickable
prototypes on verified recordings before any production frontend, then a pause for design feedback.

**What M6a built.**
- `lanegame/demo/`:
  - the registry and its tracked manifest (the 7 files and their SHA-256, matched against `results/m5/inventory.json`);
  - `GameSession` with human, model and scripted seats, M5's turn loop, idempotent submissions and secret seeds;
  - the recording format with per-turn state digests, and replay;
  - frames for the lane and the history;
  - the action menu, with the engine's reasons for every unavailable choice.
- `lanegame/experiments/m6a/` (stages `verify`, `measure`, `export`) and `tests/test_demo_session.py` (29 tests).
- `prototypes/m6a/`: three directions on a shared history controller, then the refined and glass interfaces, with
  screenshots in `results/m6/prototypes/`.
- `.gitignore` no longer ignores `docs/`. The 85 docs files on disk (the 81 untracked by `b94200f7` and four newer
  ones) plus the M6a documents are tracked again.

**What it found and fixed in the design.**
- The registry hashes each file's bytes and unpickles the same bytes, so `torch.load` never reads an unchecked file. It
  fills `OMPolicy`'s library caches with the checked networks.
- History navigation runs on the client's cache of resolved turns. The server rebuilds a history from the codes only
  after a reload (about 0.09 s for 150 turns).
- A drawn chance seed stays secret until the session stops, like the policy seed, because it foresees the chance rolls.
- The core supports the six scripted archetypes, needed to re-simulate the two seed-0 `g_shove` recordings.

**Verified.** The three seed-0 M5 recordings re-simulate exactly through the session core (actions, `payoff_A`
to the last bit, belief log posteriors identical). The 12 attacker recordings replay exactly from their actions.
Tests cover every check of the brief: the opponent's code fixed before the human acts, illegal submissions changing
nothing, isolation, idempotency, bit-for-bit replay, refusal of a tampered checkpoint before loading, and history that
neither changes the session nor reveals hidden information. The full suite passed on the final M6a tree: 1,580
passed, 4 skipped (the `--runslow` tests), 2 expected failures.

**Measured.** One turn with frames and JSON costs 1.4 ms (`sp:0`) and 2.3 ms (`om:0`), with a 2 KB gzipped response. A
bounded 150-turn bot game takes 0.27 to 0.48 s plus 0.1 s of frames and is at most 35 KB gzipped. A session retains
about 0.1 MB. The cold start took 2.4 s.

**Proposed.** The API, recording and frame contracts of the technical report §4: Watch as one bounded simulation
response played back by the client; history navigation on the client; the separate simulation worker of plan §8.2.

**Decisions for the user.** The visual direction (given, below). Whether scripted styles are offered as opponents
(still open).

**The refinement round (2026-10-21; the user's choice, approved direction).** Playing the prototypes showed that staged
Play looked like a game while it replayed recorded actions. The user chose prototype C as the foundation, with A's
bottom action controls and a better gold-lead graph, under explicit restraint rules: one system font family, no
branding or decoration, no duplicated values, details in one collapsed panel. They asked for live Play if the session
core could be connected with modest work. It could. `lanegame/demo/service.py` serves the sessions as JSON through the
local `prototypes/m6a/serve.py`, and connecting it exposed one defect, fixed: drawn seeds now stay below 2^53, so a
browser keeps them exact. The refined interface is `prototypes/m6a/refined/` ([decision record](m6/m6a_decision_record.md)
§6).

**The design rounds and the close (2026-10-21; the user's choices, approved).** Five more rounds followed the
refinement. The research ids left the visible interface for plain names ("Self-play agent", "Adaptive agent",
"Always-push bot"). The gold graph became LoL-style. Minions became balls, one per HP. A top-down diagonal lane fills
the window, and the panels merged into one frosted glass dock with an automatic rail layout on a phone held sideways
and a dark mode. The ward spot became a bush, a champion in it is drawn see-through, and each turn steps through its
two ticks with plain descriptions. A report of being stuck after warding was not a bug. Any lane action leaves the ward
spot, and the rules forbid Recall there. The demo's wording now says so. The user approved this design, and
`prototypes/m6a/glass/` is the reference for M6b (decision record §7). The overlap check passed in 240 of 240 cases
across two themes and five window sizes, and every live-play check resolved the submitted action. **M6a is complete.
M6b is next.**

## BD. M6b: the playable application (2026-10-22)

Report: [m6b_report.md](m6/m6b_report.md). Status: **completed (2026-10-22); the user's acceptance of the build is
pending, and nothing is committed (the user's instruction).** No mechanic, reward, objective, network or stored result
changed, and no training ran.

**The brief (the user, 2026-10-22).** Turn the approved M6a interface and the verified session core into a complete,
maintained application in the planned stack (React with TypeScript and Vite in `web/`, FastAPI over the Python
service), with a focused audit of the M6a work, a reviewed concurrency model, Play and Watch complete, reliable history
and session recovery, the cleanup of the prototypes, focused tests and a browser verification with screenshots.

**The audit.** The M6a claims held. Four integration risks were real and are closed: Watch simulations shared the
human turns' lock (now a separate worker process with its own registry and caches); a late response could act on a
replaced game (every request carries a generation); an unanswered submission left the screen guessing (the client
re-reads the server, appends what it missed, and only then offers a retry, which the server resolves once); no reload
recovery or expiry existed (the browser resumes its session, sessions expire after 30 idle minutes). One stale
statement in the M6a report (§6, "Play is staged") was corrected with a dated note. No engine issue was found.

**What M6b built.**
- `lanegame/demo/schemas.py` (the pydantic contract, unknown fields refused), `api.py` (the routes of the M6a
  contract plus the record download, simulations through the worker and the three verified research replays; the
  error statuses; a 16 KB body limit; per-address rate limits; the idle sweeper; the built frontend), `worker.py`,
  `__main__.py` (`python -m lanegame.demo`), the `demo` extra. `web/openapi.json` and the generated
  `web/src/api/schema.d.ts` are the typed boundary; a test fails on drift.
- `web/`: a faithful port of the approved glass design (the diagonal lane fitted around one glass surface, the dock,
  rail and console arrangements, light and dark, the stepped ticks and their words, the gold graph, Details, the
  dialogs), with a history controller that keeps progress and the viewed position apart, a store with the generation
  guard and the reconciliation rules (unit-tested against a fake server), and an error boundary.
- `python -m lanegame.experiments.m6b browser` (the scripted journey with screenshots) and the `web-demo` feature of
  `verify-lanegame` (`demo start|stop`, `browser open` readiness, `browser js`).

**Verified.** The API tests (16), the core tests (29), the frontend tests (10), the type check and the build pass;
the full suite passed (1596 passed, 4 skipped, 2 expected failures). The browser journey passes 71 of 71 checks: a live
game through real clicks whose positions and turns equal the server's re-simulated history; a trade outcome in
Details; review and return; a reload that resumes; stop, download and a bit-for-bit replay of the downloaded record in
Python; Watch with a generated match whose record replays; forced absence; duplicate and conflicting submissions; an
unanswered request and a lost session; hidden and shown beliefs; a stale response after a new game; keyboard play;
reduced motion; and the overlap check in 80 of 80 layout cases at five window sizes in both themes.

**Cleaned up.** `prototypes/` removed (37 tracked files, 2.5 MB, including 1.7 MB of generated data), the approved
glass prototype's code archived unchanged at `legacy/m6a_glass_prototype/`, the m6a `export` stage removed, the
screenshots of the design rounds kept in `results/m6/prototypes/`; CLAUDE.md, the README, `docs/README.md`, the plan
and this roadmap point at the application.

**Decisions for the user.** Acceptance of the playable build; whether scripted styles are offered as opponents
(M6b offers the two trained agents only); whether to commit. **Next: M6c**, the research replays as static files with
guided notes and the findings page. *Revised 2026-10-22: the user narrowed M6c to guided replays within Watch (§BE);
no static files, no findings page.*

## BE. M6c step 1: the guided-replay format and the first example (2026-10-22)

Report: [m6c_guided_replays.md](m6/m6c_guided_replays.md). Status: **step 1 completed (2026-10-22); the user's
review is pending, and nothing is committed.** No mechanic, reward, objective, network or stored result changed, and
no training ran.

**The decision (the user, 2026-10-22; approved direction).** The application stays a simple research demonstration
focused on Play and Watch. M6c is a small collection of optional guided replays within Watch, in three steps: the
format and one complete example, the user's review and refinement, then a few more examples. Dropped: a separate
findings page, extensive in-app research explanation, and a prominent gallery of all 15 recordings. Retained: Play
and Watch as they are, the clear label of a recorded research game against a generated simulation, deterministic
verification of every recording a guide uses, and a simple link to the research documentation. The research
question, the method, the aggregate results, the comparisons and the limitations move to M6d's research overview.
The 15 recordings stay as research artifacts; nothing was deleted.

**What step 1 built.**
- `lanegame/demo/guides.py` and `lanegame/demo/guides/<id>.json`: a guide names a verified research replay, a start
  position, an optional end and three to five checkpoints (id, position, title, a note of at most 400 characters,
  an optional highlight of a champion, tower, wave, crashed minions or the ward spot). Position t is the state after
  t turns. The loader refuses a file that breaks a rule with a message naming the file and the rule, and the service
  checks every position and highlight against the verified replay bundle at startup.
- `GET /api/guides` and `GET /api/guides/{id}` in the typed contract (`schemas.py`, `web/openapi.json`, the generated
  TypeScript types).
- Watch lists the guides beside the recorded games. A guide opens its recording paused at the start position with a
  compact note panel in the dock: the introduction, then each event with Previous and Next. Speed, the history,
  the graph and Details work as before. A move away from an event keeps the note labelled with its turn and offers
  "Return to this event"; notes can be hidden; closing the guide keeps the replay. The Play mode button
  returns to the game the browser remembers instead of starting a new one. Guides never appear in a live game or a
  generated match.
- The first example as first built: turns 12 to 25 of the seed-0 head to head, a wave that stacked, crashed and
  was followed by a recall. *Revised the same day after the user's first review* (the M6c report §8): in a guide,
  Play ran past every event to turn 150 (and, under reduced motion, a turn lasted 0.34 s at 1x, fourteen times
  faster than an animated turn); the guide's steps are now called events and positions units; and the example is
  now a chain where each event causes the next. "Stacking to Crashing" (first titled "Stack the wave, crash it,
  take a plate") covers turns 75 to 84 of
  `sp_s0_vs_g_shove` (the self-play agent against the always-push bot): the agent's trades interrupt the low bot's
  Recall twice, the bot's champion removes none of the agent's minions, the agent's wave grows to 22 HP and
  crashes into the bot's tower with 20 HP, and Pressure on the crash breaks two plates (+1.00 gold each) while the
  lead moves from the bot's 1.03 to the agent's 3.63. Every link is a rule of the engine or the bot's fixed rule.
  Next event and Play walk the turns to the next event and stop there, and plain Watch under reduced motion now
  holds a turn as long as its animated frames. *Second review, same day:* the lane now draws arrows for how each
  champion and leading wave moved in the turn shown (a walk, a recall to base, a return from base, a death, a wave
  crashing into the tower), from per-turn moves the server states and tests against every turn of the three
  replays; a third review made them short trails right behind each champion and wave. *Fourth review:* the
  feature is branded Sequences for visitors (each sequence is a chain of events in one game), and this one is
  titled "Stacking to Crashing" (id `stacking-to-crashing`, address `?sequence=stacking-to-crashing`).
- `tests/test_demo_guides.py` (the format's rules, the bundle checks, the API, and every number the notes quote
  pinned to the replay), `web/src/game/store.test.ts` additions, and `python -m lanegame.experiments.m6c browser`
  (the sequence driven in headless Edge with screenshots in `results/m6/m6c/`).

**Decisions for the user (step 2).** The chosen sequence, the start position, the checkpoint positions, the
wording, the highlights and the note panel's placement; then a few more examples in the accepted format (step 3).

## BF. M6d part 1: the demo corrections and the research overview (2026-10-22)

Overview: [research_overview.md](m6/research_overview.md). Status: **the demo corrections and the research overview
are completed (2026-10-22); the user's approval of the overview's text is pending, and nothing is committed.** No
mechanic, reward, objective, network or stored research result changed, and no training ran.

**The decision (the user, 2026-10-22).** The user approved the first sequence, "Stacking to Crashing", put further
sequences on hold, and asked for everything up to the M6d research overview, plus a README section on the sequence and
how it translates to real League.

**The demo corrections** (the audit's findings, [audit](audit/repository_audit_2026-10-22.md) §3).
- D1. A trade, tower shot, gank or minion fire now quotes its damage as damage, since durability from invested gold
  divides the sum. A new minor outcome, `hp_lost`, gives the HP a seat lost whenever the two differ. A test over the
  450 turns of the three research replays rebuilds the resolver's sum and its reduction for every seat.
- D2. A fight between two adjacent units is narrated as "between" them, no longer as "at a base".
- D3. Each tick frame carries `packets_lost`, the minion packets each seat lost in that tick, and the narration says a
  front wave "is cleared" instead of "walks" when it died. The count matches the engine's own death counter on 1,800
  ticks (the three replays and three random games).
- D6. A failed stop at the turn cap offers a retry instead of leaving the record and the belief unshown.
- D7. The API test that depended on a drawn chance seed is seeded and recalls at the first legal turn.
- W2 and W3. The "computes no game rule" claim now says what the browser narrates, and the About dialog states the
  competence cost as a loss in all six pairs, the attackers' margins as lower bounds and that pure self-play was still
  improving, with a link to the overview.

**The research overview.** A newcomer's account of M5: the question, the game and what one unit of gold means, the two
arms, the measurement, the results with all six pairs and their 95% intervals, the arms side by side, the
limitations, what the demo shows, and the corrections to earlier wording. `python -m lanegame.experiments.m6d
overview` checks every number against `analysis_main.json`, `strategic_main.json`, `monitoring_main.json` and the
battery summaries, and fails on any signed number it cannot trace.

**Corrections to earlier wording** (made in the living summaries, the plan's §9 and the overview; the reports stay as
written).
- The held-out family's +6.4 averages four scripted variants (+3.7 to +14.4, mean +11.4) and three conditional styles
  (−0.2 to −0.7); "+6.4 against held-out scripted variants" understated the gain against scripted styles (audit W1).
- Pure self-play fails concession under threat only in the two seeds re-read at 150 turns and holds in all six at 50
  turns (audit R2).
- The sp freezing margins are 0.014, 0.015 and 0.034, not all "within 0.03"; the C_cross interval of the om arm against
  its start, [+0.004, +0.55], excludes 0; the README's "about three times" is 3.6 times (audit W6).

**The README.** A section on "Stacking to Crashing" maps each event to League: damage cancels a recall, a slow push
stacks a wave, a crashed wave lets a champion hit the tower, plates pay gold until 14 minutes, and a late recall costs
plates. It names what the model simplifies and that one game is an example.

**Not done in M6d yet.** The reproduction guide, the three bundles, the `reproduce` re-run on a clean commit (audit
R1) and the Streamlit archive.
