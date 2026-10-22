# M5c: strategic battery and limitation cells

*2026-10-19. Status: **M5c complete.** The strategic battery, its registered 150-turn re-runs and the seven M4
limitation families ran on all twelve M5 finals, and S8 is assembled with no partial verdict. The dedicated attackers
have not run. M5 is not complete.*

- *Protocol:* [m5_preregistration.md](m5_preregistration.md), version 1 (amendment 1).
- *Design:* [m5_design.md](m5_design.md) §6, §10.1.
- *Earlier stage:* [m5b_initial_report.md](m5b_initial_report.md) (training and the M5d endpoints, outcome 2).
- *Generated tables:* `results/m5/report.md` (S8 section) and the M4.5c-style battery tables
  `results/m5/battery/battery_om.md` and `battery_sp.md`.
- *Records:* `results/m5/strategic_main.json` (S8), `battery_extension.json`, `battery_validation_150_diagnostic.json`,
  `eval_battery.json` and `eval_cells.json` (checkpoint identities), `corrections_readiness.json` part B1 (the smoke).
- *Decision trail:* `results/m5/decisions_m5c.tsv`.

## Summary

- **Four of the five M4.5c strategic verdicts replicate in both arms.** Conditional wave management, conditional
  freezing and the reply to freezes, selective trading, and Recall timing hold in 6 of 6 seeds in each arm.
- **Concession under threat does not replicate in either arm** (om 3 of 6 seeds, sp 4 of 6, 5 needed).
  - The failure comes only from the constructed zoning cells (`H_zone_c`).
  - In the weaker-side lead cells no forced response beats either policy.
  - In `H_zone_c` a one-turn Shove beats the om policy by +0.17 ± 0.17 pooled: positive in all six seeds, but the
    interval reaches 0. The sp arm fails by one seed, on two seeds whose best alternatives differ (+0.21, +0.33).
- **The sp arm's freezing verdict is fragile.** It replicates, but three seeds sit within 0.03 of the 0.2 margin. On
  the pure-rollout G50, seeds 1 and 2 cross the margin (+0.22, +0.25), and the rule would fail.
- **The om finals are worse than the sp finals in every limitation family**, in all or nearly all seeds, with both
  finals placed in the same states:

  | family | om − sp |
  |---|---|
  | strength use | −0.53 ± 0.13 (150 turns) |
  | freezes | −0.26 ± 0.14 |
  | Recall windows | −0.42 ± 0.26 (100 turns) |
  | natural lead, trailer and freeze states | −0.54 ± 0.09 (150 turns) |
  | large gaps | −0.39 ± 0.25 (150 turns) |

  This matches the M5d competence loss in direction and size. One caveat: the om final starts each of these cells
  with its prior belief and no previous action (§5), which the sp final does not need.
- **Unchanged from M4 in both arms:**
  - the lead premium (early +8 G50: om +0.60, sp +0.59, M4.6c +0.63);
  - the weaker side's choices;
  - the cost of forced Wards and Shove plans;
  - the missed guard in the favourable-freeze window (guard beats the policy by +0.53 in om and +0.24 in sp, in all
    six seeds).
- **Horizon validation failed in 51 of 144 (final, family) pairs**, 22 in om and 29 in sp.
  - Each failed pair was re-run at 150 turns as registered.
  - An unregistered diagnostic on the validation subset finds that the 150-turn margins agree with the 300-turn
    margins in all 51 pairs. That supports 150 turns as the replacement horizon. It does not check the re-run records
    themselves.
- **Compute:** 16.4 h in all (battery 8.73 h, cells 2.11 h, re-runs 5.58 h).
  - Estimates: 9.33 h for the battery and 1.50 h for the cells, from M5a's plan.
  - The 6.63 h re-run estimate is mine, computed from the battery's verdicts before the re-run started. The M5b
    report had given only a per-family range.

**Recommended next step: the 12 core attackers** (about 7.1 h). Exploitability (S6) is the last registered endpoint.
It completes the secondary Holm family and answers the third campaign question. §9 lists the rest.

## 1. What ran

| stage | content | records | wall h (estimate) |
|---|---|---|---|
| B1 smoke | the battery and cell instruments on `om:0` and `sp:0` at tiny sizes, the 150-turn re-run path on real records, re-collection of natural states | scratch only | 0.04 |
| `battery` | the full M4.5c design on all 12 finals: 12 families, 48 natural or 24 early states or the full grid, K 12 or 24, every protocol, the 300-turn validation subset | 12 × 1,226 = 14,712 | 8.73 (9.33) |
| `cells` | weaker side and lead premium per final; strength use, freezes, Recall windows, natural states of the common start, and large gaps with a gap specialist, cross-arm per seed | 3,732 | 2.11 (1.50) |
| `battery_extend` | the 150-turn re-run of every failed (final, family) pair, all states and protocols | 5,296 | 5.58 (6.63) |

- **Gates.** The user approved the battery and the cells on 2026-10-18, with no reduction named. The full design ran.
- **Checks before each evaluation stage:**
  - the training check;
  - the SHA-256 of every evaluated checkpoint (192 files);
  - the active pre-registration version.
  Each stage wrote an identity manifest.
- **Before the long run,** the B1 smoke checked the never-executed 150-turn path on real finals:
  - 14 of 14 checks passed in 127 s;
  - the re-collected natural states matched the original collection exactly (seed, turn and seat);
  - the S8 analysis read the 150-turn records back.
- **Coverage is complete.**
  - Every family of every final has every expected (state, protocol) record.
  - Every natural-state collection reached its designed size.
  - Every cell family has its designed count: weaker side 432, premium 540, strength use 648, freezes 432, Recall 288,
    natural 960, gaps 432.

## 2. Horizon validation and the 150-turn re-run

The battery reads each alternative's margin over the policy at 50 turns (Q in natural families, G in constructed
ones). M4.5c's pre-registered check compares that 50-turn margin with the 300-turn margin on a fixed subset of states,
per final and family. It fails when:
- the bias exceeds max(0.15, 2 SE);
- the slope leaves [0.8, 1.25];
- or two significant readings contradict.
A failed family is re-run at 150 turns, and the 150-turn value replaces the 50-turn one.

| | om | sp |
|---|---|---|
| failed pairs (of 72 per arm) | 22 | 29 |
| most frequent | `B_frozen`, `C_trade`, `C_trade_early`, `G_trade_c` (3 each) | `C_trade` (5), `A_wave_early`, `B_frozen`, `C_trade_early` (4 each) |
| families that never failed | `B_freeze` | `A_wave` |

- **Bias drives most failures.** Of the 51, 29 fail on the bias alone, 15 on the slope alone and 7 on both. None
  fails on a contradiction. With K = 12 to 24 rollouts per state the standard errors are small, so modest systematic
  shifts fail.
- **150 turns agrees with 300 turns on the validation subset** (unregistered diagnostic, `strategic.validate_at`,
  `results/m5/battery_validation_150_diagnostic.json`). I added it after the user asked what happens if a re-run
  fails again. It reuses the same subset and the same pass rule, so it supports the choice of horizon. It is not a
  check of the re-run records, which use every state.
  - The 300-turn validation records also hold the turn-150 values, so the same pass rule can compare 150 with 300.
  - It passes in **51 of 51** failed pairs, with biases within ±0.12 and slopes 0.85 to 1.09.
  - At 100 turns, 9 of the 51 would still fail.
- **The critic does not drive the natural-family verdicts.**
  - Before the run, the om critic was a concern: it was trained on a 50/50 mix and predicts about +4 at s0 for both
    seats. Q50 adds 0.99^50 ≈ 0.6 of that critic.
  - In the 60 natural (final, family) pairs that passed validation, the Q50 and pure-rollout G50 margins differ by
    0.03 on average and at most 0.09.
  - The 0.2 rule gives a different answer in only 2 of the 60, both within 0.04 of the margin: om s4 `A_wave_early`
    and sp s1 `B_freeze`.

## 3. Strategic replication (S8)

The rule: a verdict holds in a seed when no fixed alternative beats the policy by more than 0.2, under protocol
"policy", at the primary statistic (Q150 or G150 after a re-run). A verdict replicates with at least 5 of 6 seeds.
Per-seed values are the largest margin of any alternative over the verdict's families.

| verdict | families | om | per-seed gap (om) | sp | per-seed gap (sp) |
|---|---|---|---|---|---|
| conditional wave management | `A_wave` | **replicates** 6/6 | −0.14 −0.14 −0.15 −0.03 +0.07 −0.03 | **replicates** 6/6 | −0.08 −0.11 −0.05 −0.27 −0.24 −0.13 |
| conditional freezing and the reply to freezes | `B_freeze`, `B_frozen` | **replicates** 6/6 | −0.15 +0.10 +0.00 +0.10 +0.12 −0.10 | **replicates** 6/6 (fragile) | +0.19 +0.18 +0.17 +0.10 −0.03 +0.08 |
| selective trading | `C_trade` | **replicates** 6/6 | −0.09 −0.02 −0.05 −0.10 +0.14 −0.06 | **replicates** 6/6 | −0.09 −0.13 −0.08 +0.00 −0.21 −0.13 |
| concession under threat | `H_zone_c`, weaker-side cells | **does not replicate** 3/6 | +0.28 +0.10 +0.14 +0.26 +0.42 +0.15 | **does not replicate** 4/6 | −0.09 +0.02 +0.02 +0.09 +0.21 +0.33 |
| Recall timing | `E_recall` | **replicates** 6/6 | −0.08 −0.16 −0.13 +0.01 −0.26 −0.27 | **replicates** 6/6 | −0.04 −0.09 −0.24 −0.21 −0.18 −0.08 |

**Concession under threat, by component.**
- **The weaker-side lead cells hold in all twelve finals.** The best forced weak-side response (hold, safe, trade or
  reset) minus the policy is −0.14 to −0.85.
- **`H_zone_c` (constructed zoning states, the stronger, equal or weaker side at levels 6 and 10) carries the failure.**
  - om exceeds the margin in seeds 0, 3 and 4 (+0.28, +0.26, +0.42).
  - Pooled over seeds, a one-turn Shove beats the om policy by +0.17 ± 0.17, positive in all six seeds.
  - sp exceeds it in seeds 4 and 5 (+0.21 and +0.33, both read at 150 turns after a re-run). Pooled, its best
    alternative is +0.03.
- **Some of each per-seed gap is selection noise** (an inference). The rule takes the largest of nine alternatives,
  and the best alternative changes from seed to seed. The pooled om Shove effect is positive in every seed, but its
  interval (+0.17 ± 0.17) reaches 0, so it is suggestive, not confirmed. The verdicts stand as registered.

**The sp freezing verdict is fragile.** It holds, but seeds 0 to 2 are within 0.03 of the margin.
- On the pure-rollout G50, seeds 1 and 2 cross the margin (+0.22 and +0.25), so the rule read on G50 would give 4 of 6
  and fail.
- Seed 1 passed validation, so its 0.22 against Q50's 0.19 is a clean disagreement. Seed 2's G50 reading includes
  `B_frozen` at 50 turns, the horizon that failed validation there.
- Pooled over seeds, the best freeze alternative in sp (`1t:guard`, −0.07 ± 0.18) is near the policy.

The sp concession verdict goes the other way: it fails on the registered statistic by one seed and would replicate on
the other statistic. Both sp verdicts sit one seed from the threshold.

## 4. The battery families per arm

The best fixed alternative's margin over the policy, pooled over the six finals (positive means the alternative beats
the policy):

| family | om best alternative | om margin | sp best alternative | sp margin |
|---|---|---|---|---|
| `A_wave` | Shove once | −0.11 ± 0.10 | Shove once | −0.20 ± 0.08 |
| `A_wave_early` | Slow Push once | −0.02 ± 0.09 | Shove once | −0.24 ± 0.31 |
| `B_freeze` | guard once | −0.10 ± 0.21 | guard once | −0.07 ± 0.18 |
| `B_frozen` | Shove once | −0.14 ± 0.17 | Shove once | −0.04 ± 0.17 |
| `C_trade` | Safe once | −0.08 ± 0.11 | Safe (behaviour) once | −0.13 ± 0.07 |
| `C_trade_early` | Hold (behaviour) once | −0.21 ± 0.14 | Hold (behaviour) once | −0.02 ± 0.12 |
| `D_ward` | farm once | −0.16 ± 0.14 | Ward once | −0.18 ± 0.10 |
| `E_recall` | Recall now | −0.15 ± 0.11 | Recall now | −0.16 ± 0.08 |
| `F_freeze_c` (constructed) | guard for 20 turns | **+0.53 ± 0.40** (6/6) | guard once | **+0.24 ± 0.13** (6/6) |
| `G_trade_c` (constructed) | Safe once | +0.12 ± 0.24 (4/6) | Trade once | +0.04 ± 0.23 |
| `H_zone_c` (constructed) | Shove once | **+0.17 ± 0.17** (6/6) | Hold (behaviour) once | +0.03 ± 0.19 |
| `I_ward_c` (constructed) | farm once | −0.28 ± 0.31 | farm once | −0.10 ± 0.15 |

- **Natural families.** No alternative beats either policy on average. Forced Shove plans (−0.39 om, −0.47 sp), Thin
  plans (−0.50, −0.59), Wards and staying at low HP cost what they cost in M4.5c.
- **Slow Push.** In om early states it coincides with the policy (−0.02), as M4.4c found for the wave actions.
- **The favourable-freeze window.** Guarding beats the policy in every final of both arms. This is the gap the M4
  closeout listed as missed (G3), and it is about twice as large in om.

## 5. Cross-arm limitation cells

Both finals play seat A from the same states and rollout seeds, against each arm's final in seat B. The table gives om
minus sp averaged over the two seat-B finals, with the seed count where om is ahead.

| family | cells | horizon | om − sp | om ahead in |
|---|---|---|---|---|
| strength use (M4.6f B2) | all | 150 | −0.53 ± 0.13 | 0 of 6 |
| | conversion (A+8, A+16, edge, plausible, HP) | 150 | −0.62 ± 0.22 | 0 of 6 |
| | defence (B+8, B+16, breaker) | 150 | −0.45 ± 0.23 | 0 of 6 |
| | equal | 150 | −0.52 ± 0.11 | 0 of 6 |
| freezes (M4.6f B3) | all | 50 | −0.26 ± 0.14 | 0 of 6 |
| | freezer, favourable | 50 | −0.32 ± 0.22 | 0 of 6 |
| | breaker, favourable | 50 | −0.31 ± 0.25 | 0 of 6 |
| Recall windows (M4.6f B4) | all | 100 | −0.42 ± 0.26 | 0 of 6 |
| | weak side against a stronger opponent | 100 | −0.42 ± 0.31 | 0 of 6 |
| natural states of `ext/baseline:s` (M4.6f B5) | leader, trailer | 150 | −0.54 ± 0.09 | 0 of 6 |
| | freezer, breaker | 50 | −0.28, −0.22 | 0 of 6 |
| large gaps (M4.6a E4) | gap cells | 150 | −0.30 ± 0.30 | 0 of 6 |
| | equal cells | 150 | −0.65 ± 0.44 | 0 of 6 |
| | ×2.2 gap | 50 | +0.17 ± 0.51 | 4 of 6 |

- **The deficit is as large in equal cells as in lead cells.** It is general play, not a failure to use a lead.
- **A cold-start caveat.** These cells use constructed states or the common start's natural states, which no om game
  produced. The om final therefore starts each one with its prior belief and no previous action.
  - That is the convention it trained with at archive starts (about a quarter of its training transitions). It is
    not its natural mid-game state, and the sp final needs no such input.
  - Two observations argue against a pure cold-start artefact. The deficit grows from 50 to 150 turns, after the belief
    has updated. And the M5d games from s0, where both arms start naturally, show the same competence loss.
  - The cells cannot rule it out, though. Read the size of the cross-arm deficit with this in mind.
- **The gap specialist out-converts the sp finals by +1.03 ± 0.58** in the gap cells, consistent with M4.6c's +1.15.
  Against the om finals its edge is +0.48 ± 1.57, too imprecise to read.
  - Only three specialists exist (seeds 0, 2 and 4). Seeds 1, 3 and 5 reuse the neighbour's, so the six seeds and the
    6-of-6 count overstate the independent evidence.

## 6. Per-arm cells

| quantity (each final against its own policy) | om | sp | om − sp |
|---|---|---|---|
| lead premium, early +8, G50 | +0.60 ± 0.16 | +0.59 ± 0.12 | +0.01 ± 0.22 |
| lead premium, mid +8, G50 | +0.31 ± 0.12 | +0.24 ± 0.13 | +0.08 ± 0.18 |
| lead premium, late +8, G50 | +0.15 ± 0.09 | +0.16 ± 0.10 | −0.02 ± 0.14 |
| weak side +8: best forced response − policy | −0.58 ± 0.24 | −0.32 ± 0.14 | −0.26 ± 0.25 |
| strong side +16: threaten − policy | −0.57 ± 0.28 | −0.69 ± 0.15 | +0.12 ± 0.30 |

- **A lead buys the same premium in both arms**, and it matches M4.6c (early +0.63, mid +0.29, late +0.14).
- **In both arms the policy beats every forced weaker-side response and forced threats from the stronger side.**

## 7. The M4 goals re-measured

The classes follow the closeout's definitions:
- **used:** no tested alternative beats the policy against competent opponents;
- **missed:** a tested alternative beats it, confirmed;
- **weak:** the opportunity is weak under these mechanics;
- **open:** the evidence cannot decide.

| goal | om | sp | evidence | change from M4 |
|---|---|---|---|---|
| G1 gold to strength | used early, weak late | used early, weak late | premium cells, §6 | none |
| G2 strength to farm and XP | used | used | forced threats lose to the policy; cross-arm conversion om − sp −0.62 | om converts less than sp |
| G3 freezing | used in natural states; **missed** in the favourable window | used in natural states; **missed** in the favourable window | `B_freeze`; `F_freeze_c` guard +0.53 (om), +0.24 (sp), 6/6 | none; larger in om |
| G4 contesting a freeze | used | used | `B_frozen` best −0.14 (om), −0.04 (sp) | none |
| G5 Slow Push | weak | weak | `A_wave` Slow Push −0.16, −0.21; coincides early in om | none |
| G6 Shove | used | used | Shove once −0.11, −0.20; Shove plans −0.39, −0.47 | none |
| G7 trading | used | used | `C_trade` best −0.08, −0.13; constructed trade benches +0.12 (om), +0.04 (sp) | none |
| G8 Recall | used | used | Recall now −0.15, −0.16; staying −0.49, −0.57 | none |
| G9 the weaker side | used in lead cells; **missed** in constructed zoning states | used in lead cells; zoning verdict fails on 2 seeds | weaker-side cells; `H_zone_c` Shove +0.17 (om, 6/6) | the M4.5c concession verdict no longer replicates |
| G10 large gaps | open | missed | specialist − sp +1.03 ± 0.58; specialist − om imprecise | none for sp |
| G11 Vision | weak | weak | forced Ward −0.18 natural, about −0.6 constructed | none |
| G12 robustness | used against scripted styles; competence lost against competent opponents; exploitability open | missed against unfamiliar styles; exploitability open | M5d: P1 +9.26, P2a −0.63; attackers not run | the om package moves G12 for scripted styles |

## 8. What M5c adds to the M5d result

- **M5d found that the modelling package pays at a competence cost.** M5c shows where the cost lives. In every
  limitation family the om final does worse than the sp final from the same states (with the cold-start caveat of §5).
  - The deficit is as large in equal states as in lead states, and in defence as in conversion.
  - It grows from 50 to 150 turns.
  - It is a general loss of play quality, not one missing skill.
- **The om policies still make the same strategic choices.**
  - Four of five verdicts replicate.
  - The premium, the weaker side's choices and the costs of forced plans match M4 and the sp arm.
  - The package's play differs in how much it gains from those choices against competent opponents, not in which
    choices it makes.
- **Two M4-era gaps remain in both arms:** the guard in the favourable-freeze window, and the specialist's edge in
  large gaps.
- **One M4.5c verdict, concession under threat in constructed zoning states, no longer replicates in either arm.**
  - In om it fails by two seeds, and one alternative is ahead in all six (with an interval reaching 0).
  - In sp it fails by one seed.

## 9. Failures, deviations and limitations

- **Failures.** None in the battery, the cells or the re-runs.
- **Readiness.** The B1 smoke recorded validation failures at its tiny sample size. That was expected, and it was
  only a check of the pipeline.
- **Deviations from the protocol.** None in the experiment. Added during this stage, without changing any registered
  rule or classification:
  - the `battery_extend` stage, which runs the registered exception 3 and collects states in parallel;
  - the G50 reading beside the registered statistic;
  - the 150-against-300 diagnostic;
  - the per-family summary and the generated report tables.
- **The re-run ceiling.** Before the run I set a ceiling of 12 h on the re-run, after which I would stop and report.
  - The estimate was 6.6 h, so the re-run went ahead automatically, under the battery approval.
  - The ceiling was my choice, not a registered rule. The user saw the estimate and asked to continue as scheduled.
- **Tests.** The M5 test files pass (44 tests). The full suite after this stage gives 1,535 passed, 4 skipped (the
  `--runslow` tests) and 2 expected failures (`results/m5/regression_suite_m5c.log`).
- **Limits of the reading.**
  - The replication rule takes the largest of several alternatives per family, which biases the per-seed gap upward.
  - Constructed families are states the policies rarely see.
  - Cross-arm cells use the common start's natural states, which neither arm produced.
  - The gap specialist exists for seeds 0, 2 and 4 only; seeds 1, 3 and 5 use the neighbour's.
- **The sp freezing verdict** replicates with three seeds near the margin. Treat it as fragile.
- **The generated report's "other statistic" column** reads G50 for natural families and Q50 for constructed ones,
  always at 50 turns. For a family re-run at 150 turns it therefore mixes a horizon change with the statistic change.
  §2 uses only families that passed validation.

## 10. Recommended next steps

1. **Run the 12 core attackers** (`--confirm-m5-attackers`, a quarter budget, about 7.1 h). Recommended.
   - S6 is the last registered secondary endpoint. Its three hypotheses are the only open members of the Holm family.
   - It also answers the campaign's third question, how exploitable each arm is.
   - Given the om arm's competence loss, whether it is also easier to exploit is the most informative open question.
2. **Skip the optional om_table ablation** unless the adaptation channel itself becomes a question.
   - Its reading is limited: it compares two lower bounds and changes the target's inputs out of distribution.
   - Neither M5d nor M5c isolates the belief channel. The ablation would not isolate it as a cause of the competence
     loss either, so the follow-up in item 4 is the better test.
3. **Then M5e, the closeout:**
   - consolidate M5b–M5d into the final report;
   - recommend the configuration for M6;
   - produce the reproducibility materials.
   The evidence so far supports presenting the two arms as a trade-off: the package for exploiting scripted, library-like
   play; pure self-play for competent play. It does not support either as uniformly better.
4. **Possible follow-up research, proposals only, needing a new design and approval:**
   - Isolate the cause of the competence cost. The package changes the belief, the opponent mix and the self-play
     exposure (−33% of self-play transitions) together. A matched arm with the fixed opponents but no belief, or with
     equal self-play transitions, would separate them.
   - Re-examine the constructed zoning states and the favourable-freeze window. Both are gaps the policies share and
     self-play does not close.
5. **Housekeeping:**
   - the four `--runslow` tests have not been run in this campaign;
   - all M5 work since commit `82568827` is uncommitted.
