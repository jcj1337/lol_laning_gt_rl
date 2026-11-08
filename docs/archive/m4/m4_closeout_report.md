# M4 closeout review

*2026-10-16. Status: review complete, awaiting the user's decision. This is consolidation only. No training or new
experiment ran, and no rule, parameter, objective, observation, default, preset or checkpoint changed. The configuration
choices in §5 are recommendations, and nothing in the code adopts them. Nothing is staged or committed.*

*Inputs:*
- *the [roadmap](../rl_research_roadmap.md), the [design philosophy](../design_philosophy.md) and the
  [mathematical reference](laning_game_mathematical_reference.md);*
- *the M4.5c [research report](m4_5/m4_5c_research_report.md) and the M4.6 [design review](m4_6/m4_6_design_review.md);*
- *the M4.6 reports [a](m4_6/m4_6a_report.md), [b](m4_6/m4_6b_investment_report.md), [c](m4_6/m4_6c_report.md),
  [d](m4_6/m4_6d_report.md), [e](m4_6/m4_6e_report.md) and [f](m4_6/m4_6f_report.md), with their pre-registrations and
  generated tables;*
- *the training and analysis records under `results/m46c/` and `results/m46f/`.*

*Earlier reports are historical records and are not edited ([docs/m4/README.md](README.md)). The corrections they need
are in §2, and the living summaries (the roadmap, the README files) were corrected to match.*

**Terms.**
- **Resources and returns.** A *resource* is the gold of one fresh wave; natural income is about 0.40 resources per
  turn. *G_h* is a seat's discounted (γ 0.99) earned-gold lead over h turns, without a critic bootstrap.
- **Duel ratio** = (1.05^ΔL · M_A / M_B)²: damage per exchange, stronger over weaker.
  - A *plausible edge* is one level plus 3 invested resources early (duel ×1.21–1.27), sometimes with the opponent at
    60% HP.
  - A *large gap* is an investment ratio of ×1.5–2.2 (×1.76 gives duel ×3.1).
- **Policies.**
  - *M4.5c finals*: the six fresh 16.384M runs on `m45_calibrated()`.
  - *M4.6c arms*: continuations of the M4.5c finals in each environment.
  - *M4.6f arms*: continuations of the M4.6c 8.192M C1 policies, with the shared or the separate critic.
- **Classes** (§3): *used*, *missed*, *weak*, *open*.
- Intervals are 95% t intervals over six seeds unless stated.

## Summary

**Recommendation: ready for M5, with documented limitations.**
- No strategic blocker was found, and the evidence supports closing M4.
- Before M5 starts, two things remain: a setup decision for the user (§5, §9) and M5's own first task, a
  pre-registration and readiness check (§8.4). Neither is further M4 research.

**What M4 established.**
- **The intended structure is in the game.** The mechanics reproduce the causal structure the design intends for most
  laning goals: wave tempo against exposure, HP as a priced resource, Recall timing, contest-or-concede, and strength
  turning into resources through kills, forced Recalls and absence (M4.5b, M4.6a).
- **Self-play learns conditional strategies without strategy rewards.**
  - Shared-policy PPO self-play learned state-conditioned wave management, with Shove used where it pays.
  - It learned to stay out of heavy waves while trading, to time Recalls, to contest freezes when healthy, and to
    concede under a real threat.
  - Against competent opponents, no tested fixed alternative beats these choices, in every seed (M4.5c, M4.6c, M4.6d).
- **Plausible early advantages are converted.** The general policies convert them at least as well as every tested
  alternative, including the large-gap specialists, which do worse there (−0.40 ± 0.12, M4.6d).

**What remains limited.** All of this is documented and none of it blocks M5 (§6).
- **Freezing pays only in a narrow window.** The freeze must be early, with a fresh freezer wave and a damaged
  breaker, and then it is worth about one turn of income (+0.30 to +0.51).
  - Outside the window the mechanics make breaking cheap.
  - Inside it the policy under-uses guarding, in a state family that is 0.03% of training turns.
- **Leads matter early and fade late.** Twenty turns of income buy ×1.25 in a duel early and ×1.02 late. Natural leads
  are small.
- **Large gaps, a generalization diagnostic.** Specialists still out-convert the general policies (+1.15 ± 0.28), in
  states that natural play almost never produces.
- **The shared critic is strength-blind outside natural states.** The opt-in separate critic recovers about a third of
  the strength premium. Within the tested 8.192M continuation, no material behavioural improvement was demonstrated.
- **Self-play's opponent distribution leaves gaps.** The policies leave value against conceding opponents (+0.56 to
  +1.86). Attackers at a quarter of the target's budget set an exploitability lower bound of +1.02 ± 0.48.
- **Other limits.**
  - Slow Push differs from Last Hit and Thin mainly against passive opponents.
  - Control of XP is weakly evidenced.
  - In equal states the trading decision itself is close to neutral.

**Configuration (recommended, awaiting the user's decision; §5).**
- **Environment.** `GameConfig.m45_calibrated()`, unchanged, as the supported baseline. C1
  (`m45_calibrated(invest_gain=3.24, invest_scale=64.0)`) stays a documented, provisional opt-in. It is not adopted, and
  no confirmation campaign is needed for either choice.
- **Network.** The shared `PolicyValueNet` stays the default. `SeparateCriticNet` is a verified opt-in, not an M5
  factor while the critic investigation is closed.
- **Evaluation rule, under either network.** Critic values are never evidence in strength-gap states; rollouts are.
- **Training setup for M5 (proposal, §5.3).**
  - Fresh initialization.
  - M4.6c's archive start mixture (about 25% of transitions), paired by seed with the M4.5c s0-only finals.
  - M4.5c's PPO settings.
  - 6 seeds × 32.768M transitions.

**Corrections (§2).**
- The M4.6f 4.096M and 8.192M runs are separate continuations from the same checkpoint, not 12M transitions
  experienced by one policy.
- The arms diverged after the first update, so they did not train on identical data.
- The intervention changes sharing, capacity and optimizer scaling together, so it does not identify one mechanism.
- Cancellation of a common value error is a plausible, untested and incomplete explanation.
- "No difference" means no material improvement was demonstrated within the tested budget, not zero benefit.

**M5 (§8).**
- The roadmap's commitments are verified and kept.
- Several parts were underspecified: the objective, the success evidence, the budget, the training start distribution
  and what "adaptive opponents" means. A concrete scope is proposed and labelled as a proposal.
- **First task: M5a, the campaign pre-registration and readiness check.** No campaign runs, and it needs about 1–2 h
  of compute.

**Decisions for the user (§9).**
1. C1 or the baseline curve for M5 (recommended: the baseline).
2. What "adaptive opponents" means in M5. Recommended: learned best responses and conditional scripted styles. The
   alternative, opponent modelling, would be new scope.
3. Re-confirm that the narrow freeze window and the late-flat progression are accepted abstractions. If either is not,
   it is a mechanics change that must come before M5.

---

## 1. Scope, method and checks

**What was done.**
- The goals of M4 were recovered from the repository (§3).
- Every goal was assessed against the strongest rollout and counterfactual evidence of M4.5–M4.6 (§4).
- The five interpretations the review brief questioned were checked against the training records, the analysis
  outputs and the pre-registration (§2).

The roadmap's M5 entry was compared with its earlier versions (§8).

**Checks performed** (read-only, on existing artifacts):

| check | source | result |
|---|---|---|
| M4.6f continuation lineage | `results/m46f/train_main.json`, `train_ext.json` | all 24 runs (2 phases × 2 arms × 6 seeds) start from `results/rl/checkpoints/m46c/train/ext/C1/s<seed>/final.pt`, with the same checkpoint SHA-256 per seed in both phases; every run's initial actor digest equals the start's; budgets 4,096,000 and 8,192,000 |
| M4.6c continuation lineage | `results/m46c/train_main.json`, `train_ext.json` | both phases start from the M4.5c 16.384M finals (`m45c/ext/s<seed>/final.pt`) |
| matched start, then divergence | `results/m46f/smoke.json` | "first rollout identical in both arms", "the arms diverge after the first update", "separate re-run: bit-identical weights": all true |
| how far the arms drifted | `results/m46f/analysis.json`, `value_natural/drift` | at 8.192M the arms' finals differ by a total-variation distance of 0.27–0.28 per decision; each moved KL 0.23–0.29 from the start |
| classification | `results/m46f/analysis.json`, `ext/scorecard` | V1 "small" (ratio 0.449 separate vs 0.207 shared, start 0.173); B1 "no material difference"; powered B1 "small"; B2 "no material difference"; outcome 4. The report quotes these correctly |
| protocol | [m4_6f_preregistration.md](m4_6/m4_6f_preregistration.md) | the extension was registered as "new runs from start:s with an 8.192M schedule". The protocol was right; only later interpretation overstated |
| regression suite | `python -m pytest` on the current working tree (including the uncommitted M4.6d–f library changes) | 1,444 passed, 4 skipped, 2 xfailed in 8 min 11 s: the same counts M4.6f reported, with the legacy digest pins passing |

**No targeted experiment was run.** Two open questions could change a setup decision, and both need PPO training:
- **The isolated effect of the archive start mixture.** M4.6c ran no 0% control.
  - It does not block M5. The proposed M5 design measures it as a seed-paired comparison with the M4.5c finals
    (§5.3).
- **C1's primary effect.** M4.6c's proposed confirmation was paused by the user.
  - It does not block M5. Both curves are coherent, and the choice is a design preference (§5.1).

## 2. Corrections to earlier interpretations

The reported numbers, the pre-registered thresholds, the outcome-4 classification, both M4.6f amendments and every
deviation table stand unchanged. These are the thresholds: V1 materiality +0.25 with the 0.5 ratio floor; B1 0.20 /
0.30; B2 0.15 / 0.25. The pre-registered classification is not revisited: V1 stays "small" and the outcome stays 4.
What changes is how the results are described.

| # | earlier statement (where) | what the artifacts show | corrected reading |
|---|---|---|---|
| 1 | "The shared critic does not move. **12M additional transitions in total (both phases)** changed its strength recovery by +0.00 to +0.03" (M4.6f §5.1) | the 4.096M and the 8.192M runs each start from the same M4.6c checkpoint, with their own learning-rate schedule (§1). No policy experienced 12.288M transitions in this pilot | **Neither continuation changed the shared critic's strength recovery (+0.00 to +0.03).** Both were separate runs from the same start, 4.096M and 8.192M. The largest budget any one M4.6f policy experienced is 8.192M. Its full lineage is 16.384M fresh (baseline curve, s0 starts), then 8.192M (C1, archive starts, learning rate re-warmed), then 8.192M (C1, re-warmed): 32.768M transitions over three schedules and two environments |
| 2 | "The same network, **data** and optimizer learn strength value once the critic has its own trunk" (M4.6f §6) | matched: the initial weights (verified exactly), the PPO seed, the hyperparameters, the start-state sequence and the first rollout. From the first update the policies differ (smoke check), so the states visited and the returns differ; the finals differ by a total-variation distance of 0.27–0.28 per decision | **Matched initialization, seeds and start distribution; after the first update each arm trained on its own policy's data.** Each critic was evaluated against its own policy's returns, as designed. The two arms did not see identical data |
| 3 | "**Sharing is enough to cause the blindness here.** … actor–critic sharing is sufficient to keep the critic strength-blind"; "sharing the trunk with the actor is enough to keep the critic strength-blind, and removing the sharing is enough to start fixing it" (M4.6f summary and §6) | the intervention changes three things at once: it removes the value gradient from the actor's trunk (and the policy gradient from the critic's), adds 28,160 parameters (36,158 → 64,318), and gives the critic trunk its own Adam normalization. Only the global gradient clip was ruled out as a coupling. The report's own §1.1 says "a positive result supports the practical change, not one of these mechanisms" | **The shared configuration stays strength-blind under continued training. The separate configuration learns part of the strength value, but it changes sharing, capacity and optimizer scaling together.** The pilot validates the practical intervention; it does not identify sharing as the cause |
| 4 | "a strength error that is the same for every candidate of a decision cancels in the advantage; **only actions that change strength (spending, experience) can be credited through it**" (M4.6f §6; "consistent with" in its summary; M4.6e §5.3) | the successor states of different actions differ in HP, wave and position. A critic that values strength also values how strength interacts with them: an HP lead or a crashed wave is worth more to the side that can convert it. Those interaction terms do not cancel, so a better critic can change the advantages of Trade, Safe, positioning and wave actions without those actions changing strength. M4.6e's ranking test is consistent with cancellation dominating in the flagged decisions (better supervised values moved the freezer's Spearman from 0.09 to 0.14–0.17), but it does not test the mechanism inside PPO. Untested alternatives: the actor following slowly under an annealed learning rate; advantage noise; the behavioural families' resolution (about ±0.2); changes in states the families do not sample | **Cancellation of a common, additive value error is a plausible partial explanation, not a demonstrated or complete one.** Better valuation of existing strength can, in principle, change trading, positioning and wave choices |
| 5 | "The better critic **did not produce** better strategic behaviour" (M4.6f summary); "Behaviour did not follow" (roadmap summary); "behaviour unchanged" (docs/README.md, project memory); "did not change the agents' strategic behaviour" (README) | at 8.192M: B1 +0.06 ± 0.23 and B2 +0.01 ± 0.11, both "no material difference" (intervals inside ±0.30 and ±0.25). The targeted families have intervals of ±0.2–0.25. The powered head to head (+0.13 ± 0.09, 6/6) suggests a small general edge, not established. Tested: one 8.192M continuation of an already strong policy. Not tested: from-scratch training and longer budgets | **No material behavioural improvement was demonstrated within the tested budget** (an 8.192M continuation). Effects below about ±0.2 resources, effects at larger budgets and effects in a from-scratch campaign are not excluded. This is not evidence of zero benefit |
| 6 | "A targeted pilot giving the critic its own trunk **fixed much of the value error**" (README) | the separate critic recovers about a third of the strength premium: pure edge 0.02 → 0.31, large + gap 0.02 → 0.32, V1 0.21 → 0.45. It still predicts 1.66 of a realized 5.7 for a ×1.76 gap and values a level at +0.06 | **It recovered about a third of the strength value the shared critic misses.** It still underestimates large gaps about threefold and is nearly blind to level |
| 7 | "The separate critic learns strength value; **the shared one never does**" (roadmap summary); "More training alone does not change the shared critic" / "leaves the shared critic unchanged" (roadmap path step 2, §AR) | the shared arm's V1 moved +0.03 ± 0.15 over its start, and its edge recovery stayed at 0.01–0.02, in both continuations | **Within the tested continuations (up to 8.192M, M4.6c settings), further training did not measurably change the shared critic's strength recovery.** "Never" is not established |

**Where the corrections were applied.**
- The roadmap: the summary, path-to-completion step 2, and §AR (a wording fix with a pointer here).
- The README's status paragraph and roadmap table.
- `docs/README.md`.
- The project memory.

The M4.6e and M4.6f reports are unchanged, by the repository's convention; this section is their correction.

## 3. The goals M4 was meant to meet

**Sources.**
- The README's research question: can an agent learn to exploit laning interactions with gold as the only reward?
- The design philosophy §2, the intended qualitative relationships:
  - combat strength comes from progression;
  - threats shape positions;
  - trading has opportunity costs;
  - Recall timing is a decision;
  - no action is unconditionally best.
- The two concerns of the M4.6 design review:
  - freezes break cheaply;
  - strength rarely becomes denial.
- The M4.5c research question: can PPO self-play discover economically meaningful strategies without shaping?
- The roadmap's stopping principle: a perfect critic, or every recognizable strategy, is not required. A weakness that
  is small, confined to unusual states or resistant to one justified intervention is documented, unless it undermines
  the central claim.

Together these give the goals G1–G9 below, with G10–G12 as diagnostics.

**Classes used in §4.** "Used" is relative to the alternatives that were tested: scripted plans, forced actions,
specialists and attackers. It is not a certificate of optimal play.

| class | meaning |
|---|---|
| **used** | a profitable opportunity exists, and the policy uses it: no tested alternative beats the policy against competent opponents |
| **missed** | an opportunity exists, and the policy misses part of it: a tested alternative beats the policy, confirmed |
| **weak** | the opportunity is weak or absent under the tested mechanics |
| **open** | the existing evidence cannot distinguish these cases |

Rarity alone is never read as a failure, and a demonstration in an extreme constructed state is never read as success.

## 4. Strategic acceptance assessment

### 4.1 The table

| goal and intended interaction | strongest evidence (rollouts and paired counterfactuals) | works when / fails when | evidence on | class | remaining uncertainty | blocks M5? |
|---|---|---|---|---|---|---|
| **G1 gold → combat strength.** Levels and investment raise damage, durability and wave clear through one saturating map; a lead must matter and keep mattering, without a snowball | exact mapping (M4.6b): 20 turns of income buy ×1.25 in a duel early, ×1.06 mid, ×1.02 late. Learned play (M4.6c, baseline, a +8 lead's G50 premium): early +0.63 ± 0.06, mid +0.29 ± 0.03, late +0.14 ± 0.09. Natural leads are 4–10% of earned gold; an attack ratio ≥ 1.25 occurs on 0.05% of turns | works early (levels up to about 10): a lead buys a real duel edge and keeps paying for 50 turns. Fades mid and late through saturation and level convergence | both | **used** early; **weak** late, by design | whether the late fade is the intended design (the C1 question: a mid +16 lead earns ×1.76 more under C1, late unchanged) | **no**: a deliberate property of the saturating curve ("no unbounded snowball"); investment work was paused by the user; C1 is documented as the option |
| **G2 strength → access to farm and XP.** A credible threat makes the weaker side give up position, farm or XP, or pay in HP, deaths or absence | M4.6a: a ×1.76 gap becomes +4.5 resources over 100 turns under the learned policy, about half through kills and the weaker side's absence and 9% through posture. M4.6c: forcing indiscriminate threats from a +16 lead is −0.34 ± 0.13 against the policy (baseline, 8.192M), because the leader converts by choosing its trades. M4.6d: the gap specialists are worse than the general policy on plausible edges (−0.40 ± 0.12 in C1, −0.38 ± 0.42 baseline). M4.6a E2: no scripted threat rule matches the policy. XP: a breaker in a favourable freeze loses 2–6 XP (M4.6d); a level's realized value is +0.27 ± 0.46 (M4.6e); XP is not rewarded | works through kills, interrupted Recalls and absence. Fails as posture-only denial and as indiscriminate bullying of a competent opponent | both | **used** for farm at plausible edges (relative to tested alternatives); **open** for XP, and for whether more value is available at plausible edges | no best response has been trained from plausible-edge starts; the critics are nearly blind to level | **no**: M5 measures it with rollouts |
| **G3 freezing.** Holding the wave next to one's own tower denies farm and XP, and is worth defending when a walk-up is dangerous for the breaker | M4.6d, confirmed on independent seeds: a freeze holds only while the breaker's wave clear is below the freezer's 8-HP wave and the freezer trades the walk-up (through level 6 baseline, level 8 C1). Best case: guard − passive +0.25 ± 0.16. With the breaker at 60% HP: guard − push +0.30 to +0.51, guard − policy +0.30 to +0.47. Both healthy, even at ×1.25: −0.06 to +0.01. The denial is real: farm lost, not delayed. M4.5c: a sustained freeze is −0.20 against learned play and +0.56 / +1.86 against a back-off opponent. Natural freezes: 87% are the breaker's own push arriving, 68% last one turn; no defence beats the policy in natural early freeze states | works early, with a fresh wave in the freezer's tower cell, a breaker that walked up damaged, or an opponent who concedes. Fails after the early levels, with both champions healthy, and in worn natural freezes | both: the window is mechanics, the under-use inside it is the policy | **missed** inside the window (+0.3 to +0.5 left, in about 1,400 joint turns per 8.192M run, 0.03%; the critic ranks the freezer's options at chance; a better critic changed nothing, B3 +0.05 ± 0.23); **weak** outside it | whether more exposure would teach guarding; whether the narrow window is the intended design | **no**: the abstraction was accepted (M4.6d, the 2026-10-16 decision to pause mechanics changes). It would block only if the research intent requires freezes that hold against competent breakers (§9) |
| **G4 contesting and breaking a freeze.** The frozen-on side contests while contesting is cheaper than conceding, and resets or concedes when it is not | M4.6d: the healthy breaker's policy is as good as committing to the break and better than every concession (a 3-turn concession costs 0.4–0.7). At 45% HP reset and contest are close (reset +0.1 to +0.25, not confirmed). The policy is nowhere materially inferior against its own opponent, but under-commits against a freezer that trades every walk-up (commit − policy +0.57 ± 0.33 and +0.82 ± 0.42, confirmed). M4.6a: the Shove is always the breaker's best one-turn reply; a guarded walk-up turn costs it 0.12–0.26 of a bar. M4.6b correction: with the policy's own wave, walking up costs −0.02 to −0.13 | contest when healthy (cheap: the walk-up cell farms at full rate). The gap appears only against a freezer style that self-play never produces | both | **used** against self-play opponents; **missed** against an out-of-distribution freezer | robustness to unfamiliar freezer styles | **no**: an opponent-diversity question for M5's evaluation |
| **G5 slow push.** Build a stack that crashes as a larger wave; it pays against a passive opponent and is answered by Thin or Shove | M4.4c (the four-action economies before M4.5): Last Hit, Thin and Slow Push leave identical lanes in 68–77% of turns. Slow Push beats a passive last-hitter (+2.65 over 16 turns in the clean lane), and any Thin or Shove reply erases it; Slow Push → Shove is +0.11 [0.00, +0.23] over Last Hit ×8. M4.5c (`m45_calibrated()`): one turn of Slow Push is −0.19 against the policy's choice; all six finals have games with three or more Slow Push turns followed by a crash of 12 HP or more; no fixed wave action beats the policy | works against passive (last-hitting) opponents; fails against any opponent that thins or shoves | mainly mechanics; for the policy, only that no fixed sequence beats it | **weak** in most states; **open** on `m45_calibrated()` (the identity shares were not re-measured after M4.5a) | the coincidence of the three non-Shove actions on the current preset | **no**: a design question open since M4.4c. M5 should claim conditional wave management (Shove or not), not a distinct slow-push strategy |
| **G6 shoving.** Shove for tempo (crashes, Recall windows, plates) when the later costs are low, and not when exposure dominates | M4.5c: Shove is 33% of the mass and state-conditioned (47–56% with the enemy away, 33–52% when frozen on, 18–26% at levels 1–5). It has the best immediate gold (+0.45) but is the modal action in only 22% of contact states. One turn of Shove is −0.11 and Shove ×4 is −0.35 against the policy. All six seeds. M4.3b's Clear dominance did not return | used where it pays; over-shoving loses | both | **used** | none material | **no** |
| **G7 trading HP.** Trade when HP, strength and the wave favour it; HP is a resource priced by what it costs in farm, deaths and resets | M4.6a: a bar of HP is worth 1.2–1.5 resources, and no single trading cost is load-bearing. M4.5c with the M4.6b correction: standing in a heavier wave costs about 0.3, and the policy rarely trades into one (the modal action in 15% of such states). The trading decision itself (Trade − Hold, behaviour only) is about −0.08, within noise; single trades are near neutral (±0.2). With a lead, the policy's selective trading beats forced threats (G2). M4.6d: the specialists' large-gap edge is many small trade-timing gains | works as positional restraint and as selective trading with an edge. In equal states the trade decision is near neutral | both | **used** with an edge and for positioning; **weak** for the trade decision in equal states; **missed** in large gaps (G10) | none beyond G10 | **no** |
| **G8 Recall and reset.** Timing depends on HP, gold, the wave and the opponent | M4.5c: staying at low HP costs −0.89 (−0.96 at 8.192M); stepping back instead of resetting costs −0.33; recalling now is as good as the policy's choice. The policy puts 28% of its mass on Recall below 35% HP and 1% at 70% HP or more. M4.6a: the reset is not subsidized (investing on return 0.00 ± 0.12; a full-HP Recall costs about 0.4). M4.6f: a better critic did not change the Recall rate | 41% of attempts are interrupted: a guessing game the counterfactuals do not show to be a mistake | both | **used** | the interrupted-Recall guessing game | **no** |
| **G9 weak-side alternatives.** The weaker side concedes position, farm or wave when staying costs more, and keeps contesting otherwise | M4.6c §5.1 (constructed +8 / +16 lead cells, 8.192M): no forced trailer response beats the policy. Hold-only is 0.12–0.30 below it; Safe-only, Trade-only and reset are 0.96 or more below; deaths are unchanged. §5.2 (natural lead states): no one-turn alternative beats the policy, and its mass leans the right way (correlation +0.10 to +0.35). M4.5c: concession under a ×1.76 threat. M4.6d: at a Recall window next to a stronger opponent, a damaged weak side should step back with Safe rather than Yield (+0.6 by rollouts). The actor mostly Yields; a better critic moved its mass slightly toward Safe (0.19 → 0.25) with no measurable change (B4 −0.03 ± 0.24) | works in natural and plausible states. The one gap is a rare family: about 18,000 joint turns per run, 0.4% | both | **used**; **missed** in the one rare family | none material beyond that family | **no** |
| **G10 large strength gaps** (a generalization diagnostic, not natural-play competence). A large gap should be converted decisively | M4.6a: ×1.76 is converted into +4.5 under the policy, and a specialist adds +2.11 ± 0.62. The specialists' edge over the general policy on the M4.6a gap cells: +1.48 (M4.5c finals), +1.34 (M4.6c 4.096M baseline arm), +1.15 ± 0.28 (8.192M baseline arm); +1.73 ± 0.15 in C1 (cross-environment); ×2.2 gives +2.08. Exposure: duel ≥ 2 in about 1,300 joint turns per 8.192M run (0.03%, all from archive starts); duel ≥ 3 never, not even in the archive. Critic: the ×1.76 premium is predicted at 0.14 (shared) and 1.66 (separate) against a realized 5.7 | converts, but leaves value, in states natural play does not produce | policy and training distribution | **missed** | whether gap exposure in training closes it: the archive mixture's defence gains of +0.22 / +0.28 are confounded with more training | **no**: outside the natural distribution; reported as a diagnostic |
| **G11 Vision (secondary).** A Ward before Pressure should pay in sieges | M4.5c: a forced Ward costs 0.3–0.6 against every opponent; Wards are 2% of decisions | rarely profitable on this model | mainly mechanics | **weak** | long sieges (positive trend, unresolved since M4.4c) | **no** |
| **G12 robustness to other opponents** | M4.5c: attackers at a quarter of the target's budget beat each final by +1.02 ± 0.48 (6/6), target-specifically; value left against conceding styles, +0.56 / +1.86 for a sustained freeze and +1.28 for bullying a wary opponent. M4.6a: a specialist's better play works against all 15 finals it never trained against | works against competent self-play opponents; gaps against styles self-play never shows | policy and self-play's opponent distribution | **missed** against unfamiliar styles (expected of self-play without opponent modelling) | the size of the gap at larger attacker budgets | **no**: this is M5's core evaluation topic |

### 4.2 The points that need particular care

**Plausible early advantages.**
- *Do they arise?* Commonly: about 137,000 joint turns per 8.192M run, 3.4% (M4.6e).
- *What do the mechanics give them?* A duel ratio of ×1.21–1.27. An early +8 lead earns +0.63 over 50 turns under
  learned play (M4.6c), and a ×1.25 edge realizes about +0.6 over 300 turns (M4.6d).
- *Does the policy use them?* Every tested alternative does worse: forced threats, scripted threat rules, and the
  large-gap specialists. No learned best response has been trained from plausible-edge starts, so whether more value
  is available there is **open**.
- *The critic.* It values these edges at about 0 (+0.02 against +0.63). The separate critic repairs a third of that
  without a measurable behavioural change.
- **Verdict: accept.** M5 measures conversion with rollouts, and the classification stays "used, relative to tested
  alternatives".

**Freezing against realistic opponents.**
- *Against learned breakers.* A sustained freeze loses (−0.20, M4.5c). Natural freezes are mostly transients: the
  breaker's own push arriving, with the freezer's wave already worn. There the policy's immediate push-out is right.
- *The one profitable window.* Early, with an HP edge: about one turn of income, confirmed. The policy under-uses it.
- *Why the window is narrow (M4.6d §7).* The freezer's walk-up threat is limited in time: wave clear outgrows the 8-HP
  wave by level 6–8. Its strength is not the limit. This is the one interaction weaker than League's.
- **Verdict: accept, as M4.6d recommended.** It is a design choice (for example, waves that strengthen over time), not a
  defect, and the user has paused mechanics changes.

**The cost of contesting or breaking a freeze.**
- *Breaking is cheap by construction* (design review, M4.6a):
  - the walk-up cell farms at full rate (collection 0.9, all XP);
  - a guard gets one exchange per turn, 0.12–0.26 of a bar;
  - the Shove is always the breaker's best one-turn reply;
  - with the policy's own wave, accepting the walk-up costs only −0.02 to −0.13.
- *Conceding costs.* A 3-turn concession costs 0.4–0.7.
- *Where the policy has a gap.* It contests correctly against its own opponent. Its one gap is under-committing against
  a freezer style that self-play does not produce.
- **Verdict: the mechanics make contesting correct when healthy, and the policy does it.** The out-of-distribution gap
  belongs to M5's robustness evaluation.

**Weak-side alternatives.**
- *The options exist and are priced sensibly* (M4.6c, M4.6d):
  - staying and farming is close to the policy;
  - retreating, fighting back and resetting for 10 turns each cost more than a resource;
  - at low HP, reset and contest are close.
- *The policy's choices.* It mixes correctly, and its deaths do not rise when it is behind.
- *The one miss.* Safe instead of Yield at a Recall window next to a stronger opponent, in a family that is 0.4% of
  turns.
- **Verdict: used, with one rare miss.**

**Extreme gaps** stay a generalization diagnostic (G10).
- Duel ≥ 3 never occurs in training, not even in the archive.
- So the specialists' +1.15 to +1.73 measures how far shared self-play generalizes, not how it plays the natural game.

## 5. Configuration and setup recommendations

### 5.1 Environment: the baseline `m45_calibrated()` or C1

| criterion | baseline `m45_calibrated()` | C1 `m45_calibrated(invest_gain=3.24, invest_scale=64.0)` | source |
|---|---|---|---|
| pre-registered adoption tests | not applicable (current preset) | M4.6b screen: no candidate advances (D4 late capacity, D2 early premium, D1 underpowered). M4.6c criterion A: inconclusive, +0.09 ± 0.11 (×1.32) | M4.6b, M4.6c |
| mid-game lead premium, +16 lead, G50 | +0.45 ± 0.05 | +0.78 ± 0.14 (+0.34 ± 0.17, ×1.76, 6/6) | M4.6c §4.1 |
| early (G20) and late premiums | | unchanged (×0.97 early; no late difference at 50 turns) | M4.6c |
| natural play | | within ±10%; Ward −32% and late deaths +19% (both not significant); natural leads stay small, never decisive | M4.6c §7 |
| late wave capacity | | +4.2 HP (guarding a late freeze costs more) | M4.6c |
| freeze defence window | through level 6 | through level 8 | M4.6d |
| head to head; the weaker side | interchangeable | interchangeable; the weaker side keeps its options | M4.6c |
| freezes, critic blindness, large gaps, opponent distribution | | unchanged: none of these depends on the curve | M4.6c §6, M4.6d §3.6, M4.6e |
| evidence base | the full from-scratch campaign (M4.5c: 6 × 16.384M, battery, attackers), M4.6a, the M4.6c baseline arm | the M4.6c C1 arm and every M4.6d–f diagnostic (continuations only) | |
| documentation | the reference's §15 | a keyword override; not in the reference | |

**Recommendation: keep the baseline as the supported configuration. C1 stays a documented, provisional opt-in.**
- **Neither pre-registered adoption test was met.** The project's standard is "evidence before adoption … in
  representative states" (design philosophy §4). C1's effect is in constructed mid-game lead states, while natural leads
  stay small and never decisive under either curve.
- **C1 addresses none of the limitations that matter to M5.** Freezes, critic blindness, large gaps and the opponent
  distribution are curve-independent.
- **The baseline has the complete from-scratch evidence base** against which M5 will be compared, and the reference
  documents it.
- **Adopting C1 also means accepting +4 HP of late wave capacity** (M4.6b's D4), a review question that was never
  settled.

**Tradeoff, stated plainly.**
- C1 is a safe, modest step toward "a lead must keep mattering" in the mid-game.
- The M4.6d–f diagnostics ran on C1, and they remain valid: the freeze results are curve-independent (M4.6d §3.6), and
  play is interchangeable across curves (M4.6c).
- If the user values the stronger mid-game lead, C1 is coherent for M5. M5 would then lose the seed-paired comparison
  with the M4.5c finals, and the reference would need a C1 section.
- No confirmation campaign is needed for either choice.

**What stays as it is.**
- C1 remains a keyword override, as now. No preset is added or changed.
- The C1 checkpoints (M4.6c C1 arm, M4.6f) remain valid for C1 studies: same observation and action space, and their
  checkpoint digests are unaffected.

### 5.2 The critic options

| | `PolicyValueNet` (shared trunk) | `SeparateCriticNet` (own critic trunk) |
|---|---|---|
| status | **default**; the network of every M3–M4 campaign | **verified opt-in** (M4.6f); exact initialization from a shared checkpoint; checkpoint-compatible |
| value in natural states | calibrated: slope about 1.0, R² 0.71–0.77 against long returns | as good or better: R² 0.82 |
| value in strength states | nearly blind: recovers 1–2% of pure-strength premiums, 21% with an HP component (V1) | recovers about a third (pure edge 0.31, large + gap 0.32, V1 0.45); still blind to level (+0.06 per level) |
| behaviour | the reference | no material difference demonstrated in an 8.192M continuation (B1, B2, every targeted family); a possible small general edge (+0.13 ± 0.09, an amendment) not established |
| cost | | −4% throughput, +28,160 parameters |
| tested as | from scratch and as continuations | continuations of strong C1 policies only; from-scratch behaviour untested |
| mechanism | | sharing, capacity and optimizer scaling changed together (§2 item 3) |

**How to describe the options.**
- **Shared.** The default training network. Its critic is calibrated where the game goes and nearly blind to strength
  outside natural states.
- **Separate.** A verified opt-in that improves value estimates in strength states. No behavioural benefit was
  demonstrated within the tested budget. It is not a strategic fix.

**How to use them.**
- **M5 uses the shared network.** This keeps continuity with M4.5c, the seed-paired comparison. The separate critic's
  behavioural effect is not demonstrated, and its from-scratch behaviour is untested.
- **The separate critic is not an M5 factor.** The user closed the critic investigation. If it is reopened, the one open
  question M4.6f names is a from-scratch matched pair.
- **Reach for the separate critic only for value readouts in strength states.** Use it when a future study needs critic
  values as estimates there.
- **Under either network:**
  - critic values are not evidence in strength-gap or constructed-lead states, where rollouts are used (M4.6e);
  - in those families, report the no-bootstrap G_h (G150 or longer) rather than the critic-bootstrapped 50-turn
    statistic, whose tail inherits the critic's blindness (M4.6d found earlier bootstrapped readings understated
    premiums three- to fourfold);
  - in natural-state families the bootstrapped 50-turn default stays valid: the M4.5c horizon study validated it, and
    150-turn re-runs changed nothing.

### 5.3 Training setup for M5 (proposal)

| item | recommendation | evidence | alternative |
|---|---|---|---|
| environment | `m45_calibrated()` | §5.1 | C1, if the user prefers |
| network | `PolicyValueNet`, 90 → 128 → 128 tanh | §5.2 | `SeparateCriticNet` (not recommended while the investigation is closed) |
| PPO | M4.5c unchanged: 100% current-policy shared self-play; entropy 0.03, clip 0.2, 4 epochs, minibatch 256, vf_coef 0.5, value clip 0.5, gradient norm 0.5, 16 environments × 128-turn segments, lr 3·10⁻⁴ annealed to 0, γ 0.99, λ 0.95, reward scale 0.1 | the audited, reproducible protocol of M4.5c and M4.6c–f | |
| initialization | **fresh** (`torch.manual_seed(seed)`, seeds 0–5, as M4.5c) | one clean schedule per seed, reproducible end to end for M6. It avoids the re-warm cost (about 0.5 resources over the first million transitions, M4.6c) and the multi-schedule, two-environment lineage of the latest checkpoints (§2 item 1) | continue the M4.6c baseline-arm finals: cheaper and already strong, but a three-schedule lineage |
| start distribution | **the M4.6c mixture**: archive start probability 0.532, about 25% of transitions; archive games 150 turns, s0 games 512 | the only source of strength-gap and favourable-freeze exposure in training (M4.6e). M4.6c recommended it for the main campaign; with it the specialists' edge shrank by about a fifth (confounded with more training). Its isolated effect was never measured, so the M5 runs are paired by seed with the M4.5c s0-only finals (same environment, network, PPO and initial weights) | s0 only, as M4.5c: M5 would then be a longer replication |
| budget | **6 seeds × 32,768,000 transitions**, snapshots every 25,600 | M4.5c was not converged at 16.384M (+0.36 ± 0.30 against its 12.3M checkpoint). Continued training kept improving: M4.6c's arms gained +0.4 to +0.5 over the M4.5c finals, and M4.6f's gained +0.44 / +0.58 over their M4.6c starts. The 16.384M midpoints compare with the M4.5c finals: M4.5c's annealing test found a midpoint as strong as an annealed final at the same data (−0.07 ± 0.25) | 16.384M with a pre-registered conditional 2× extension, as M4.5c did |
| wall clock | about 3.5–4 h for six concurrent runs at the measured 2,500–2,600 transitions/s per run with archive starts (M4.6f) | | |

**Dependencies (git-ignored, present locally).**
- The archive: `results/rl/checkpoints/m46c/archive/`. It is rebuilt by `python -m lanegame.experiments.rl.m46c
  archive` from the M4.5c finals' self-play.
- The M4.5c finals.
- The M4.6a specialists.
- The M4.6c finals.

## 6. Accepted limitations and blockers

| limitation | kind | why it can be accepted | what M5 does about it |
|---|---|---|---|
| freezing pays only in an early window with an HP edge (G3) | mechanics (deliberate abstraction) | the denial inside the window is real; breaking is correct outside it; natural freezes are transients; the user paused mechanics changes; M4.6d recommended acceptance | reports freezing as a conditional tool; measures the window with the M4.6d cells |
| late leads fade; natural leads are small (G1) | mechanics (saturating curve) | it is the design's guard against snowballing; investment work was paused; C1 changes only the mid-game | concentrates strength families on the early and mid game |
| large-gap conversion (G10) | training distribution and policy | states absent from natural play (duel ≥ 3 never occurs) | reports it as a generalization diagnostic |
| the shared critic is strength-blind (§5.2) | learning and representation | calibrated in natural states; the one justified intervention (M4.6f) improved values but showed no behavioural gain within its budget | rollouts in strength states, and no-bootstrap G_h in constructed families |
| rare missed responses: guarding a favourable freeze, committing against a guard, Safe at a Recall window (G3, G4, G9) | policy, in rare or out-of-distribution states (0.03–0.4% of turns) | each is worth about 0.3–0.8 in its own family, and none occurs often in natural play | re-measures each with the same instruments |
| value left against conceding styles; exploitability +1.02 at a quarter budget (G12) | self-play's opponent distribution | expected of self-play without opponent modelling | the core of M5's evaluation: diverse opponents and stronger attackers |
| Last Hit, Thin and Slow Push coincide in most states (G5) | mechanics (wave-action design) | Shove versus not-Shove is the economically real distinction; the policy's mix is not beaten | claims conditional wave management, not three-way distinctions |
| control of XP is weakly evidenced; a level's value is noisy and the critic blind to it (G2) | mechanics (five-cell lane, converging levels) and measurement | XP is not rewarded; its effect runs through levels, which converge | reports XP denial as a channel, without a separate claim |
| the early game is about 10% of experience; early conclusions are less certain (M4.5c) | training distribution | the early orderings matched the later ones and the 150-turn re-runs confirmed them | powered early-game families |
| Wards rarely pay (G11) | mechanics | rational restraint, shown by counterfactuals | none beyond reporting |
| the abstraction is not League | design philosophy | deliberate | M6 keeps findings and hypotheses apart |

**Blockers: none.**
- **The stopping principle.**
  - Each limitation is small in natural play, confined to rare states, resistant to the one justified targeted
    intervention, or a deliberate abstraction.
  - None undermines the central claim. That claim is that shared-policy PPO self-play, with gold as the only reward,
    learns the conditional laning strategies the model rewards, shown by rollouts and paired counterfactuals.
- **What would make one a blocker.** One would arise if the research intent required either of the following:
  - freezes that hold against competent breakers;
  - leads that keep mattering in the late game.

  Both are mechanics changes. Because M5's results are tied to the mechanics, either would have to come before M5.
  The user's decision of 2026-10-16 to pause mechanics changes and accept documented limitations covers both. §9 asks
  for a re-confirmation, not a new decision.

## 7. Readiness recommendation

**Ready for M5, with documented limitations.**
- **Why not "ready after a small closeout action".**
  - Nothing in M4 needs to run first.
  - The setup decision is a choice between coherent options and needs no experiment.
  - Writing M5's pre-registration is M5's own first task.
- **Why not "not ready".**
  - No limitation in §6 is a strategic blocker.
  - The two that could become one are deliberate abstractions, already accepted.
- **M4's status.**
  - The evidence supports closing M4.
  - The roadmap records the closeout as "completed, in review", with closure recommended, until the user confirms the
    configuration (§9).
  - M5 is not launched automatically. Its campaign stage stays behind the `--confirm-<milestone>-<stage>` gate.

## 8. M5: the main training and evaluation campaign

### 8.1 Established roadmap commitments (verified, 2026-10-16)

From the roadmap's milestone table and "Path to completion", step 4:
- multi-seed shared-policy PPO on the setup selected at M4 closeout;
- evaluation with the existing infrastructure:
  - learning progress against fixed references and own history;
  - strategic counterfactuals (the M4.5c battery);
  - diverse opponents (historical, scripted, adaptive). This absorbs the earlier adaptive-opponent proposal;
  - cross-seed results;
  - empirical exploitability with best-response attackers;
- budget and initialization (fresh or continued) chosen from evidence at closeout;
- values in strength-gap states measured with rollouts, not with the critic;
- not launched automatically; M6 follows.

### 8.2 What is underspecified or inconsistent

1. **The scope has no objective, deliverables or success criteria.** Beyond the evaluation list, the roadmap does not
   say what M5 must show, what counts as success, or what it adds to M4.5c, which already trained six seeds and ran
   the battery and attackers.
2. **"Adaptive opponents" has no definition and no implementation.**
   - The original M5 (§D, §H) meant Bayesian opponent modelling. The 2026-10-11 scope decision dropped it.
   - The code has scripted style wrappers (`experiments/rl/m44c/opponents.py`), learned attackers and specialists. It
     has no opponent that adapts during a game.
3. **The training start distribution is not mentioned.** M4.6c changed it (25% archive starts), and its effect was never
   isolated.
4. **M4.5c's own follow-ups are not assigned anywhere.** They are a convergence probe, attackers at half the target's
   budget, and the battery's validation rule pooled over policies.
5. **Stale status in the roadmap's "What is what" table** (corrected in this update).
   - It listed the adopted reference and the training candidate as `m4_lasthit()` and
     `m4_four_actions_econ_aggro()`.
   - Every campaign since M4.5c trains `m45_calibrated()`, and CLAUDE.md calls it the current research preset.
6. **The mathematical reference's header still names `m4_lasthit()` as CURRENT.** `m45_calibrated()` appears there only
   as an EXPERIMENTAL section (§15). This is not corrected here, because it follows the user's configuration decision
   (§8.4 step 1).

### 8.3 Proposed scope (PROPOSAL, not a roadmap commitment until approved)

**Objective.** Train the project's definitive policy set on the closeout setup, and produce the pre-registered evidence
M6 will present. Two questions:
1. Does the central claim of M4.5c replicate on the final setup at a larger budget?
2. How robust and how exploitable are the policies against opponents self-play does not produce?

**Design.** §5.3: six fresh seeds × 32.768M on the baseline with archive starts and the shared network, paired by seed
with the M4.5c finals.

**Evaluation** (existing code, assembled by import):
- **Health and reproducibility:** the M4.5c checks; one seed re-run.
- **Learning curves:** against the fixed suite and own history; a convergence probe (the final against its 75%
  snapshot).
- **Head to head:** against the M4.5c finals (at the matched 16.384M midpoint and at the final).
- **The M4.5c battery**, with the M4.6b measurement fixes:
  - the behaviour-only twins;
  - lane-only positions;
  - the economic channels.
- **The M4 limitation families,** each with its own instrument:
  - lead premium and the weaker side's responses (M4.6c);
  - freeze and contest cells (M4.6d);
  - strength use, Recall windows and natural states (M4.6f B2–B5);
  - large gaps, with the M4.6a specialists as fixed opponents.
- **Diverse opponents:**
  - historical: the M4.5c finals and the M4.6c baseline-arm finals;
  - scripted: the core suite and the style families, including conceding, back-off and a walk-up trader;
  - learned: the specialists and new attackers.
- **Attackers:** at half the target's budget (M4.5c's own recommendation), reported as exploitability lower bounds.
- **Viewer traces.**

**Deliverables.**
- A pre-registration.
- `lanegame/experiments/rl/m5/` (training and evaluation drivers reusing earlier milestones by import).
- `tests/test_m5.py`.
- Checkpoints (git-ignored).
- `results/m5/` (generated tables, analysis, traces).
- The M5 report.
- Roadmap and README updates.

**Success evidence** (pre-registered; a result in either direction is a finding, and nothing depends on an action
frequency):
- **S1 health.** Every run passes the M4.5c checks, and a re-run is bit-identical.
- **S2 progress.** Learning curves and the convergence probe are reported. No cycling.
- **S3 against M4.** The M5 finals' result against the M4.5c finals is reported with intervals. The seed-paired
  midpoint comparison estimates the archive mixture's effect.
- **S4 replication.** Each M4.5c "discovered" verdict holds in at least 5 of 6 seeds, with no fixed alternative beating
  the policy by more than a pre-registered materiality in natural-state families (0.2 proposed, M4.6f's B1 level).
  M4.5c's verdicts rested on the same kind of evidence. The verdicts are conditional wave
  management, conditional freezing and the reply to freezes, selective trading, concession under threat, and Recall
  timing.
- **S5 the limitations, re-measured.** Every §4 class is re-measured with the same instrument and reported as used,
  missed, weak or open. There are no target values: a change in either direction is a result.
- **S6 robustness.** Cross-seed matrix, results against the diverse-opponent suite, and attacker lower bounds with
  intervals.
- **Overall.** M5 succeeds if it delivers this package complete. The central claim is then either confirmed on the
  final setup, or a specific contradiction is reported.

**Dependencies on M4.**
- The closeout's setup decision (§9).
- `m45_calibrated()` with the regression suite and pins green (verified, §1).
- The evaluation packages of M4.5c and M4.6a–f.
- The git-ignored archive, finals and specialists (§5.3).

**M4 limitations that matter to M5** (§6, last column):
- the critic's blindness, which dictates rollout-based evaluation in strength states;
- the rarity of the gap, freeze and Recall-window families, which calls for constructed cells with seeds as the units;
- the opponent distribution, the reason for the robustness evaluation;
- non-convergence, the reason for the larger budget.

The narrow freeze window, the late fade, Slow Push and the Wards constrain what M5 may claim. They are not things M5 is
expected to change.

**Out of scope** (the roadmap and the user's decisions):
- mechanics or reward changes;
- investment-curve work;
- further critic variants;
- new RL algorithms (PSRO and similar);
- opponent modelling, unless the user decides otherwise (§9).

**Estimated compute.**
- Training: about 4 h.
- Evaluation: about 6–10 h, by M4.5c's measured costs (comprehensive evaluation 2 h, battery 1.6 h, a 150-turn
  exception 1 h, quarter-budget attackers 1.1 h). Half-budget attackers on a 32.768M target cost several times the
  latter.

### 8.4 First bounded task: M5a, pre-registration and readiness (no campaign)

1. **Record the decision.** Write the user's setup decision (§9) into the roadmap. If the baseline is confirmed as the
   supported preset, update the reference's header labels and add a §13 line. That is a labelling change, not a rule
   change.
2. **Assemble the drivers.** Build `experiments/rl/m5` by import:
   - training: `train_shared` with the M4.6c collector;
   - the evaluation suite of §8.3.

   No new mechanics.
3. **Check the instruments.** Run the assembled suite at reduced K on the existing M4.5c 16.384M finals. Confirm that it
   reproduces the published M4.5c and M4.6 readouts within their intervals before it judges new policies.
4. **Smoke and throughput.**
   - A 10-update smoke test of the exact setup (fresh initialization with archive starts) and a bit-identical re-run.
   - A six-concurrent throughput measurement, to fix the budget and the wall clock.
5. **Freeze the pre-registration.**
   - Contents: setup, budget, families, seeds, success criteria S1–S6 and any extension rule.
   - Record its SHA-256, as in M4.6c–f.
   - Gate the campaign behind `--confirm-m5-train`.

**Budget and exit.**
- About 1–2 h of compute and no training beyond the smoke test.
- Deliverables: the pre-registration, `results/m5/readiness.json` and a short M5a note.
- Exit: the user approves the pre-registration and the budget.

## 9. Decisions for the user

| decision | options | recommendation | consequence of the alternative |
|---|---|---|---|
| **1. C1 or the baseline curve for M5** | baseline `m45_calibrated()`; C1 | **baseline** (§5.1) | C1 is coherent and harmless as far as measured. It strengthens mid-game leads, but loses the seed-paired comparison with M4.5c and needs a reference section |
| **2. What "adaptive opponents" means in M5** | (a) learned best responses (attackers, specialists) plus conditional scripted styles; (b) also opponent modelling (beliefs about the opponent, updated during a game) | **(a)**: it fits the 2026-10-11 scope decision and the existing code | (b) is new scope, in effect the original §H plan, and would need its own design and budget |
| **3. Re-confirm the accepted abstractions** | accept the narrow freeze window and the late-flat progression as documented limitations; or redesign first | **accept** (as decided on 2026-10-16) | a redesign is a mechanics change that must precede M5, and the M4.5–M4.6 evidence would need re-establishing on it |

**Defaults, applied unless the user overrides them** (§5.2–5.3):
- the shared network;
- fresh initialization;
- archive starts;
- 32.768M per seed.

## 10. Files

**New:** this report.

**Modified** (documentation only):
- `docs/rl_research_roadmap.md`:
  - the summary;
  - the path to completion;
  - the milestone and "What is what" tables;
  - §AR wording;
  - the new §AS.
- `README.md`: the status paragraph and the roadmap table.
- `docs/README.md` and `docs/m4/README.md`: the index entries.

**Unchanged:**
- all code and tests;
- every preset, default and checkpoint;
- the mathematical reference (no rule, parameter or objective changed);
- the M4.5c–M4.6f reports and pre-registrations.

**Checks run:** the read-only artifact checks of §1 and the full regression suite.
