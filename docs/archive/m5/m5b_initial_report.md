# M5: correction pass, main training and initial evaluations (first review package)

*2026-10-18. Status: **M5a correction pass complete (amendment 1). M5b training complete and healthy. M5b monitoring
and the M5d opponent-family and head-to-head evaluations complete and analysed. The strategic battery, the limitation
cells and the attackers have not run and await review and approval.** M5 is not complete.*

- *Protocol:* [m5_preregistration.md](m5_preregistration.md), version 1, amended before training by
  [m5_amendment_1.md](m5_amendment_1.md).
- *Design:* [m5_design.md](m5_design.md).
- *Generated tables:* `results/m5/report.md`.
- *Records:*
  - `results/m5/train_main.json` (health checks), `analysis_main.json`, `monitoring_main.json`;
  - `corrections_readiness.json` (the correction pass's checks);
  - `progress_main.jsonl` (the training log), `eval_monitor.json` and `eval_families.json` (the evaluation stages'
    checkpoint identities).

## Summary

**The pre-registered headline is outcome 2: the opponent-modelling package pays at a competence cost.**

| endpoint | om − sp (6 seeds, 95% t interval) | per seed | class |
|---|---|---|---|
| P1 exploitation gain (½ S_in + ½ S_out, lead at turn 114) | **+9.26 ± 0.30** | +8.94 +9.25 +9.32 +9.47 +9.64 +8.92 | demonstrated |
| P2a competence against competent opponents (½ C_hist + ½ C_cross) | **−0.63 ± 0.35** | −0.79 −0.70 −0.38 −0.46 −0.24 −1.17 | material loss |
| P2b competence head to head (om:s vs sp:s, 128 pairs) | **−0.44 ± 0.28** | −0.79 −0.08 −0.52 −0.27 −0.30 −0.67 | material loss |

- **The gain.** The modelling arm beats library-like scripted opponents far more often than pure self-play: +12.2
  against the library archetypes and +6.4 against the held-out styles. The gain is concentrated on scripted
  opponents, and against conditional styles built on a competent base it is zero (−0.2 to −0.7, see §5.2).
- **The cost.** Against competent opponents the modelling arm is worse, in all six seeds.
  - The loss is consistent across both kinds of competent opponent: −0.65 against the historical ones and −0.60 against
    the other finals.
  - It persists over the longer game: −0.60 for the competent families discounted over 500 turns (S2), −0.90 in the
    500-turn head to head (S3). These are discounted readouts, not on the same scale as the lead at turn 114.
  - It shows across seeds (S4, −0.68) and is already present at the 16.384M midpoint (S5b, −0.80).
  - Each of these secondary claims passes its interval rule and is secured under the amended Holm rule: no pending
    result (the attackers' S6) can change it.
- **Where the cost comes from (exploratory).** Against the common start `ext/baseline:s` on the competent families,
  the sp finals improved by about +0.87 (C_hist +0.86 ± 0.32, C_cross +0.88 ± 0.25). The om finals improved by about
  +0.25, an interval that includes 0 (+0.21 ± 0.52, +0.28 ± 0.27).
  - The history comparison agrees: sp final − start is +0.94 ± 0.54, om final − start is +0.20 ± 0.68.
  - So the competence gap is mostly improvement that pure self-play gained and the package did not. It is not shown to
    be a fall below the start.
  - The package trained on 21.8M self-play transitions against sp's 32.8M, and half its games were against fixed
    opponents. This design cannot separate those causes from the belief information.
- **Robustness to misspecified opponents (S1): preserved, +5.20 ± 0.38.**
  - Mixtures: +5.4 and +6.4. Held-out switchers: +10.8 and +8.0.
  - Imitate-then-switch: the om arm gains while its opponent plays an archetype (+0.9 to +4.4 before the switch). The
    20-turn windows after the switch to a competent policy show small losses (−0.1 to −0.5, three of the six intervals
    excluding 0), and the belief moves to its catch-alls within 20 turns.
- **Convergence.**
  - The sp arm was still improving at the end: final − 75% snapshot +0.59 ± 0.45, "still improving", a reported
    limitation.
  - The om arm was not shown to be: −0.08 ± 0.32.
  - No cycling in either arm.
- **Training health.** All 12 runs complete, 138/138 pre-registered checks passed, no runtime alarm, no technical
  failure, no restart. Wall time 10.90 h (estimate 9.4–11.8 h).
- **Not established here:** strategic replication (S8), the M4 limitation re-measurements, and exploitability (S6). Those
  stages have not run, and the headline says nothing about them.

**Recommendation for the next decision.** The evidence and the instruments support proceeding to the strategic
battery and the limitation cells: the full design for the om finals, about 10.2 h + 1.6 h at the measured rates
(§9). With a material competence cost established, the core attackers (12 at the quarter budget, about 7.1 h) are
also the natural next measurement. Both need the user's approval.

## 1. The correction pass (amendment 1)

The review of M5a (commit `82568827`) found eight defects. None had been fixed before this pass. Each is fixed and
tested. The protocol changes are recorded as amendment 1: the pre-registration moved from version 0 (`d2424022…`,
archived byte for byte) to version 1 (`21e35e07…`) before any training. No arm, seed, budget, family, readout or
threshold changed, and no campaign result existed when the rules were fixed.

| # | finding | fix | tests |
|---|---|---|---|
| A | `headline()` mapped an inconclusive P1 to "no material gain" | P1's "detected" class split by U; outcome 4 needs a material gain ruled out (U < 0.25); new outcome 6, "gain not demonstrated"; the competence status reported with every outcome; the headline states its scope | `test_headline_never_turns_an_inconclusive_gain_…`, `test_gain_classes_separate_…` |
| B | S6/S7 allowed three claims but gave Holm one one-sided p-value | 15 fixed hypotheses for S1–S6 (preserved / loss; gain / material gain ruled out; increase / decrease / TOST equivalence) in one Holm family. A claim needs its interval condition and Holm. While the family is incomplete a claim is "secured" when p ≤ 0.05/15; a declined stage enters with p = 1 | `test_interval_rules_and_one_sided_tests_agree` (300 random cases: each bound and its test agree exactly at 0.025), `test_holm_…`, `test_secondary_claims_…` |
| B | the midpoint endpoint was called "P2a at 16.384M" but uses C_hist only | S5a (midpoint gain); S5b (midpoint competence against historical opponents only, "not P2a"); the midpoint head to head descriptive; the former S7 exploratory and optional | end-to-end synthetic analysis |
| C | missing seeds were silently skipped | an endpoint is classified only with all six seeds, otherwise "partial (n/6)" with the missing records listed. The headline needs P1, P2a and P2b complete. Replication counts a missing seed as unknown | `test_missing_records_…`, `test_replication_never_counts_a_missing_seed` |
| D | S8 and the reporting path were unassembled | `m5/strategic.py`: battery replication (protocol "policy", Q50/G50, maximum over the fixed alternatives), the weaker-side gap, coverage (missing records, short state collections), the horizon-validation path (fail → pending until the 150-turn re-run, then Q150/G150), premium and cross-arm cells. Also `analysis.monitoring`, the D0 → E_20 panel, switch windows, belief readouts and the report tables | `test_battery_family_gap_coverage_and_the_validation_path`, `test_weak_side_and_cross_arm_cells_…`, `test_monitoring_…`, `test_switch_windows_…`, `test_report_tables_…` |
| E | the om_table attackers were planned as core; the promised live/table cross-evaluation had no jobs | the core is 12 quarter-budget attackers (om, sp). om_table is an optional follow-up behind `--include-om-table`, with the corrected reading. The cross-evaluation jobs are implemented (each target form is played by both attackers on the same games) | `test_attacker_core_and_optional_job_definitions`, `test_cli_refuses_the_optional_tables_…` |
| F | the stop conditions were checked only after training; a failed check did not block evaluation; the failure rule was not enforced | `train_shared(on_update=...)` (opt-in, read-only) with `health.RunMonitor`: defect alarms end a run, investigate alarms are recorded and the run continues. `health.run_campaign`: more than two failed pairs stop the campaign; a progress log. `train_check` adds checkpoint identities, restart-prefix checks and monitor/post-hoc agreement. Every evaluation stage needs a passed or resolved check and matching SHA-256s, and writes an identity manifest | `test_run_monitor_…`, `test_monitored_training_is_bit_identical_to_the_unmonitored_pin`, `test_coordinator_stops_after_more_than_two_failed_pairs`, `test_evaluation_gate_refuses_…`, `test_restart_prefix_checks_are_enforced` |
| G | `amend()` recorded a hash that `require_prereg()` never checked | a verifiable chain: the original record kept, every version archived, each amendment with its date, reason, replaced and new hashes, its rationale document's hash and its authorization. The gate validates the active version and rejects any undocumented change | `test_amendment_chain_…`, `test_the_real_preregistration_chain_verifies` |

- **Interrupted runs.** Exact mid-run resume does not exist (the trainer stores no optimizer or collector state).
  Recovery is a deterministic restart from the original start checkpoint (`--restart-incomplete`). It repeats the
  lost work, up to one run's wall time (7.7 h for an om run), and the health check enforces that the restart
  reproduces the preserved prefix. No restart was needed.
- **The future-stage plan** (design §7.1, §10, §10.1):
  - the core attacker estimate is separated from the optional addition;
  - the battery's horizons are broken down by purpose (§9 below);
  - nothing about those stages was run.

## 2. Tests and checks

| check | result |
|---|---|
| A1 monitored smoke: both arms' R4 smoke re-run with the runtime monitor attached, compared with the stored M5a runs | **passed** on the second run: final weights, snapshot weights and every non-wall-clock history field identical; live alarms equal the post-hoc scan. *First run failed:* it compared snapshot files by file SHA-256, and checkpoint files embed provenance (git state, time), so only that block differed. The criterion was corrected to weight digests (M4.5c's reproducibility criterion); both runs are kept in the record |
| A2 inputs of the main training | passed: the chain verifies (active version 1); the run manifest builds with every start, library network and archive identity; the 12 initial networks are the starts; no main run existed |
| A3 targeted tests and digest pins (`tests/test_m5.py`, `test_m5_corrections.py`, `test_om2.py`, `test_legacy_pins.py`, the M4.5a/M4.5b pins) | 88 passed, 1 skipped (the `--runslow` legacy pin) |
| A4 full regression suite | **1,534 passed, 4 skipped, 2 xfailed** in 8 min 20 s (`results/m5/regression_suite_amendment1.log`) |
| after A4 | the M5 test files again, 44 passed (one exploratory readout test and one labelling refinement were added after A4 collected) |

- **Skips.** The 4 skips are the slow tests that need `--runslow`: `test_legacy_pins.py:76`,
  `test_m4_observation.py:55` and `:65`, `test_m4_tower.py:81`. They were not run in this pass or in M5a. Every non-slow
  digest pin passed. The 2 xfails are expected failures from earlier milestones. The 3 warnings come from PettingZoo's
  API checker.
- **Not verified by any test or run here:**
  - `strategic.extension_jobs` (building the 150-turn re-run of a failed family) is defined but has never been
    executed; it should be smoke-tested at the start of the battery stage;
  - the battery and cells stages' identity manifests have not been exercised on real runs;
  - the coordinator's campaign-stop path is tested with stand-in jobs, not with failing training runs;
  - the defect-alarm stop is tested on synthetic history entries; no real run raised an alarm.

## 3. Training (M5b)

- **Health.** All 12 runs are complete, and the pre-registered check passes 138/138:
  - finite statistics, the budget reached within one update, no termination, one truncation per game;
  - |V| ≤ Ω, no illegal action, initial weights equal to the starts, archive share 0.248–0.254;
  - the matched snapshots present;
  - no stop condition, sp 100% current-policy with equal seats, om fixed-opponent game share 0.498–0.503;
  - the runtime monitor agrees with the post-hoc scan.
- **Runtime monitoring.** It raised no alarm during training. The coordinator recorded no technical failure and no
  stop, and no run was restarted.

| arm | updates | wall h per run | transitions/s | entropy min / last | max 10-update KL | max pre-clip gradient norm | max \|V\| (resources) | V(s0) |
|---|---|---|---|---|---|---|---|---|
| om (6 runs) | 10,651–10,687 | 7.67–7.74 | 1,177–1,187 | 1.31–1.38 / 1.64–1.74 | 0.0073–0.0076 | 17.6–23.2 | 25.1–26.7 | +3.7 to +4.3 |
| sp (6 runs) | 8,000 | 3.15–3.16 | 2,880–2,893 | 1.47–1.51 / 1.70–1.76 | 0.0072–0.0074 | 1.6–2.0 | 5.3–6.0 | −0.5 to +0.3 |

- **The om arm's larger critic and gradient scale** (|V|, the pre-clip gradient norm, V(s0) ≈ +4 for both seats) follows
  from its registered opponent mix:
  - half its games are against fixed opponents it beats by wide margins, whereas pure self-play is zero-sum by
    construction;
  - it is well inside the stop thresholds (the gradient alarm is 100, and the clip at 0.5 applies);
  - it is not a defect.
- **The package's transitions, as designed.** The om arm played 21.9M environment turns against sp's 16.4M (+33%).
  Fixed-opponent games were 50% of its games and 33% of its learner transitions, so it had 21.8M self-play transitions
  against sp's 32.8M.
- **Training-distribution diagnostic** (not an endpoint; `results/m5/om_training_fixed_opponent_returns.json`). The om
  learner's whole-game return against its fixed training opponents rose in every decile of every seed.
  - In the last decile it is large against the scripted archetypes, mixtures and switchers (about 400–680 resources
    per game).
  - It is small against the learned library types: peer 41, freezer 78, attacker 86, gapper 121, short 0.2.

**Runtime by stage** (measured; M5a estimate in brackets):

| stage | wall h | notes |
|---|---|---|
| correction-pass compute | ≈ 0.2 | A1 twice (27 s each), A3 49 s, A4 8 min 24 s, the M5 tests several times; well within the 1–2 h allowance |
| M5b training | **10.90** [9.4–11.8] | om runs 7.7 h, then the sp runs 3.15 h; the om arm ran at 0.97 of its R9 benchmark rate, sp at 1.03 |
| M5b monitoring (1,008 cells) | **0.56** [0.50] | worker time ×1.10 the R10-based estimate |
| M5d families (546 cells) | **1.87** [1.72] | worker time ×1.09 |
| analysis and report | < 0.01 | |

Memory was tight but did not limit training. The six training processes held about 650 MB each, with about 0.6 GiB
of physical memory free and Windows memory compression active, and the om arm still ran at 97% of its benchmark rate.

## 4. Learning curves and convergence (monitoring; never used to select a checkpoint)

**Learning curves** (the lead at turn 114 against the fixed panel, 16 pairs × 150 turns, seed mean ± 95%):

| | 2.048M | 8.192M | 16.384M | 24.576M | 32.768M |
|---|---|---|---|---|---|
| om, panel mean | +5.41 ± 0.28 | +6.83 ± 0.61 | +8.44 ± 0.74 | +8.97 ± 0.36 | +9.92 ± 0.47 |
| om vs `ext/baseline:s` (its own start) | −0.96 ± 0.56 | −0.81 ± 0.92 | −0.38 ± 0.81 | −0.07 ± 1.00 | +0.56 ± 0.92 |
| om vs `ext/baseline:o` | −1.33 ± 0.71 | −0.90 ± 0.79 | −0.39 ± 0.50 | −0.13 ± 0.44 | +0.55 ± 0.38 |
| sp, panel mean | +2.28 ± 0.57 | +2.34 ± 0.36 | +2.65 ± 0.40 | +3.11 ± 0.48 | +3.51 ± 0.46 |
| sp vs `ext/baseline:s` | −0.70 ± 0.77 | −0.61 ± 0.62 | +0.30 ± 0.67 | +0.25 ± 0.67 | +1.22 ± 1.23 |
| sp vs `ext/baseline:o` | −0.92 ± 0.58 | −0.42 ± 0.61 | −0.05 ± 1.15 | +0.29 ± 0.51 | +1.10 ± 0.28 |

- **Both arms first lose to their own start** (the learning-rate re-warm cost every continuation pays) and then
  recover.
- **The om arm's panel mean climbs mostly against the scripted opponents:** g_farm +5.7 → +15.1, g_shove +12.2 →
  +17.9, g_bully +11.5 → +15.5. The sp arm is flat against them.

**History** (each final against its own snapshots and its start, 64 pairs × 500 turns; lead at turn 114):

| | final − 25% | final − 50% | final − 75% (convergence probe) | final − start |
|---|---|---|---|---|
| om | +1.14 ± 0.55 | +0.95 ± 0.33 | **−0.08 ± 0.32** | **+0.20 ± 0.68** (seed 5 −1.02, significant) |
| sp | +1.35 ± 0.63 | +0.98 ± 0.57 | **+0.59 ± 0.45** | **+0.94 ± 0.54** |

- **Convergence probe** (registered rule: "still improving" when ≥ +0.25 with L > 0):
  - **sp: still improving.** The pure self-play continuation had not converged at 32.768M, a limitation; no extension
    is proposed (the protocol has none).
  - **om: not shown to be still improving.** The probe finds no improvement between 24.576M and the final. The
    16-pair curve against its own start is too noisy to say more (−0.07 ± 1.00 at 24.576M, +0.56 ± 0.92 at the
    final).
- **Cycling** (M4.5c's rule, a final losing significantly to an earlier snapshot on at least 2 seeds): none in either
  arm.

## 5. M5d: modelling minus self-play

Every family value below is the seed mean of om − sp (the lead at turn 114) with a 95% t interval over six seeds; all
families and endpoints are complete (six seeds).

### 5.1 Exploitation (P1)

| family | om − sp | om − start | sp − start |
|---|---|---|---|
| S_in (the 6 library archetypes) | +12.15 ± 0.39 | +11.94 ± 0.78 | −0.22 ± 0.62 |
| S_out (4 held-out scripted variants + 3 conditional styles on `ext/baseline:o`) | +6.36 ± 0.56 | +6.31 ± 0.48 | −0.05 ± 0.64 |
| **P1 = ½ S_in + ½ S_out** | **+9.26 ± 0.30** (τ 0.25, mean game SE 0.14) | | |

- **Against the held-out scripted variants:** +13.1 to +14.4. Against `g_slow_guard`: +3.7 ± 1.6.
- **Against the three conditional styles on a competent base:** no gain. respect −0.18 ± 0.35, passive −0.67 ± 0.66,
  stepback −0.20 ± 0.69.
- **Reading.** The exploitation gain is a gain against scripted, library-like play. It does not carry over to
  styles built on a competent policy.

### 5.2 Competence (P2a, P2b)

| family | om − sp | om − start | sp − start |
|---|---|---|---|
| C_hist (`ext/baseline:p` for p ≠ s, 5 held-out learned instances) | −0.65 ± 0.39 | +0.21 ± 0.52 | +0.86 ± 0.32 |
| C_cross (the other seeds' finals of both arms) | −0.60 ± 0.32 | +0.28 ± 0.27 | +0.88 ± 0.25 |
| **P2a = ½ C_hist + ½ C_cross** | **−0.63 ± 0.35** (τ 0.31, SE 0.13): material loss | | |
| **P2b head to head** (om:s vs sp:s, 128 pairs) | **−0.44 ± 0.28** (τ 0.02, SE 0.27): material loss | | |

- **Every one of the 20 competent opponent slots has a negative om − sp** (the slots are different policies in
  different seeds). The largest gaps are against `attacker_b`
  (−1.27), `base:1` (−1.01) and `c1_policy` (−0.97); the smallest against `freezer_b` (−0.01).
- **The C_cross gap is about the same whether the opponent is an om final (−0.50 to −0.98) or an sp final (−0.28 to
  −0.85).**

### 5.3 Robustness and the other secondary endpoints (amended Holm rule; all six claims determined)

| endpoint | om − sp | interval class | claim after Holm |
|---|---|---|---|
| S1 misspecified opponents (M) | +5.20 ± 0.38 | preserved | preserved (secured, p < 10⁻⁶) |
| S2 long-horizon competence (500 turns, discounted) | −0.60 ± 0.25 | material loss | material loss (secured, p = 0.0008) |
| S3 long-horizon head to head (discounted) | −0.90 ± 0.43 | material loss | material loss (secured, p = 0.0014) |
| S4 cross-seed head to head (om:s vs sp:p) | −0.68 ± 0.29 | material loss | material loss (secured, p = 0.0009) |
| S5a midpoint exploitation gain | +8.29 ± 0.81 | demonstrated | demonstrated (secured, p < 10⁻⁶) |
| S5b midpoint competence, historical opponents only | −0.80 ± 0.40 | material loss | material loss (secured, p = 0.0018) |
| midpoint head to head (descriptive) | −0.51 ± 0.88 | (inconclusive) | not a claim |
| S6 exploitability | — | not run | pending (its three hypotheses have no data) |

- **The Holm family is not final, because S6 has no data.** Each claim above is still determined:
  - its supporting hypothesis has p ≤ 0.05/15 (secured);
  - its opposite hypothesis has p ≥ 0.99, which Holm can never reject;
  - so no attacker result can change any of these six classifications.
- **The labelling "determined"** (rather than "interim") was added after the results were seen. It is a logical
  consequence of Holm, not a rule change, and it changed no classification.

**Misspecified opponents, by opponent and window** (exploratory; reward summed in the window, om − sp):

| opponent | before the switch | switch + 0–19 | switch + 20–59 | lead at 114 |
|---|---|---|---|---|
| `mix_bully_farm` | | | | +5.42 ± 1.25 |
| `mix_shove_safe` | | | | +6.39 ± 0.73 |
| `switch_safe_shove_30` | −0.29 ± 0.61 | +1.83 ± 0.48 | +5.32 ± 0.35 | +10.84 ± 1.05 |
| `switch_bully_farm_50` | +1.24 ± 0.40 | −0.08 ± 0.43 | +6.19 ± 0.27 | +7.96 ± 0.58 |
| `its:g_farm:40:o` | +3.11 ± 0.36 | −0.42 ± 0.39 | −0.49 ± 0.35 | +2.32 ± 0.34 |
| `its:g_safe:20:o` | +0.94 ± 0.54 | −0.43 ± 0.49 | −0.38 ± 0.52 | −0.25 ± 0.57 |
| `its:g_shove:40:o` | +4.37 ± 0.57 | −0.42 ± 0.25 | −0.13 ± 0.41 | +3.73 ± 0.49 |

- **The imitate-then-switch diagnostic.** After the switch to the competent `ext/baseline:o` the om arm loses about
  0.4 in the first 20 turns (two of the three intervals exclude 0) and −0.1 to −0.5 over the next 40 turns (one of
  three).
  - That first window is faster than its average competence gap against these bases (about −0.06 to −0.18 per 20
    turns at turn 114).
  - So a short adaptation lag may add to the gap. It is small and imprecise, and a scheduled switch is not a deceptive
    learned opponent.
- **The belief.** It moves to its catch-alls within 20 turns of the switch: catch-all mass 0.00 at turn 20 and 1.00 at
  turn 40 against `its:g_safe:20:o`. By turn 94 it has partly re-identified the competent policy (catch-all 0.28–0.51).
- **Mechanism, descriptive.** The om finals identify `g_farm`, `g_safe`, `g_bully` and `g_shove_ward` within 5 turns
  (≥ 99% of games). `g_shove` and `fz_guard_sh` share their mass (about 0.5 each early; `fz_guard_sh` is the most
  probable type in 52% of games by turn 94). The catch-all mass is high against `mix_shove_safe` (0.98 by turn 94) and
  `style:respect` (0.88), but low against `mix_bully_farm` (0.28) and `style:passive` (0.10).

### 5.4 Lane exit: D0 → E_20 with the non-gold panel (turn 94, focal orientation, om − sp)

| quantity | S_in | S_out | M | C_hist | C_cross | head to head (om's side) |
|---|---|---|---|---|---|---|
| D0, earned lead at 94 | +9.66 ± 0.44 | +5.31 ± 0.46 | +3.94 ± 0.31 | −0.48 ± 0.32 | −0.43 ± 0.28 | −0.32 ± 0.22 |
| E_20 (lead at 114) | +12.15 ± 0.39 | +6.36 ± 0.56 | +5.20 ± 0.38 | −0.65 ± 0.39 | −0.60 ± 0.32 | −0.44 ± 0.28 |
| E_20 − D0 | +2.49 ± 0.15 | +1.05 ± 0.15 | +1.27 ± 0.12 | −0.17 ± 0.08 | −0.17 ± 0.12 | −0.12 ± 0.13 |
| level difference | +0.72 ± 0.08 | +0.33 ± 0.05 | +0.40 ± 0.05 | +0.01 ± 0.03 | −0.01 ± 0.04 | +0.02 ± 0.06 |
| XP difference | +31.5 ± 4.3 | +14.7 ± 1.5 | +15.6 ± 1.6 | −0.3 ± 0.7 | −0.2 ± 1.2 | +0.1 ± 1.3 |
| HP difference (both present) | +0.04 ± 0.02 | +0.00 ± 0.04 | +0.06 ± 0.04 | +0.01 ± 0.03 | −0.00 ± 0.03 | −0.00 ± 0.05 |
| opponent away (share) | +0.10 ± 0.02 | +0.07 ± 0.01 | +0.08 ± 0.01 | +0.00 ± 0.01 | −0.00 ± 0.01 | (0.08 each side) |
| opponent dead (share) | +0.11 ± 0.02 | +0.07 ± 0.02 | +0.09 ± 0.01 | −0.00 ± 0.01 | −0.00 ± 0.01 | (0.03 each side) |
| invested gold, focal − opponent | +3.23 ± 0.98 | +3.86 ± 0.88 | +1.26 ± 0.72 | −0.40 ± 0.40 | −0.52 ± 0.51 | −0.47 ± 0.63 |

- **Accounting check.** D0 from the lane-exit panel equals the summed rewards to turn 94 in every family.
- **Against scripted and misspecified opponents,** the om arm leaves the lane ahead in gold, levels and XP. It has
  killed or forced out its opponent more often, and its lead keeps growing after turn 94.
- **Against competent opponents,** the deficit is purely economic. Levels, XP, HP and absence are equal, and about a
  quarter of the deficit (−0.17) accrues between turn 94 and turn 114.

## 6. Primary outcome and secondary claims, under the amended rules

- **P1 demonstrated; P2a and P2b material losses → outcome 2, "pays at a competence cost".**
  - The outcome rests only on the three primary endpoints, each meeting its own interval criterion with six complete
    seeds.
  - A material loss in either of P2a and P2b is two one-sided tests at 0.025 (≤ 0.05 together); both are losses here.
- **Secondary.** S1 preserved; S2, S3, S4 and S5b material losses; S5a demonstrated. All six claims are determined
  under Holm while S6 is pending.
- **Exploratory** (not tests): the arm-versus-start contrasts, per-opponent values, the panel, the switch windows and
  the belief readouts.

## 7. Failures, deviations and uncertainty

- **Failures.** None in training or evaluation. One readiness-check defect (A1's comparison criterion) and a few watcher
  false alarms (a tooling script outside the repository) were fixed.
- **Deviations from the protocol.** None in the experiment.
  - **Added after training launched, reporting and analysis only:**
    - the generated runtime section;
    - the "not rejectable" / "determined" labelling of Holm statuses (a logical consequence of Holm; it changed no
      class);
    - the training-distribution diagnostic.
  - **Added before training launched, and so not informed by any result:** the switch-window and belief readouts,
    registered as exploratory.
- **What the contrast is.** om − sp is the package effect. It changes the information (the belief and the previous
  action), the opponent distribution (half the games against fixed opponents) and the self-play exposure (−33% of
  self-play transitions, +33% environment turns) together, and it cannot attribute the competence cost to any one of
  them.
- **Statistical uncertainty.**
  - **The competence loss is established by the registered rule** (U < 0 and mean ≤ −0.25), but its size is uncertain:
    P2a's interval runs from −0.98 to −0.27, P2b's from −0.72 to −0.16.
  - **Precision.** The between-seed variation is mostly seed variation (τ about 0.3), not game sampling, except in the
    head to head (τ 0.02 against a game SE of 0.27).
- **Non-convergence.** The sp arm was still improving at the end; the om arm was not shown to be.
  - A longer schedule could change the size of the competence gap, in either direction.
  - No extension is proposed or authorized.
- **Scope.** The style families are library-favourable, and the competent opponents come from the same lineage.
  Nothing is measured beyond 500 turns.

## 8. Readiness for the next stages

- **The evidence supports proceeding.** Training is complete and healthy, the finals are fixed and identified (SHA-256
  in `train_main.json`, verified by every stage), and the primary and secondary M5d endpoints are complete and
  classified.
- **The battery and cell instruments are ready.** They ran end to end on stand-ins in M5a (R10), and their analysis
  (S8) is implemented and tested on synthetic records.
- **One piece is untested:** the 150-turn re-run path after a failed validation (`strategic.extension_jobs`). It should
  be smoke-tested at the start of the battery stage, at negligible cost.
- **What the next stages add.** The outcome makes them informative:
  - whether the om finals' strategic play (wave management, freezes, trading, concession, Recall) still replicates;
  - where the competence cost appears in the M4 limitation families;
  - how exploitable each arm is.

## 9. The proposed next stages (awaiting approval; nothing has run)

Estimates are the R10-based design sizes × 1.09, the ratio measured on the monitoring and families stages; six
workers.

| stage | content | estimate |
|---|---|---|
| **M5c strategic battery** | the full M4.5c design on all 12 finals | **10.2 h** |
| of which | standard 50-turn continuations | 4.9 h |
| | exception 1: the 20-turn freeze plans at 60 turns | 1.8 h |
| | the 300-turn validation subset (om 2.6 h, sp 0.9 h) | 3.5 h |
| | contingent: a failed family re-run at 150 turns | 0.3–4.2 h per family over all twelve finals |
| **M5c limitation cells** | weaker side, premium and freezes (H 50), Recall windows (H 100), strength use, large gaps and natural states (H 150) | **1.6 h** |
| **core attackers** | 12 × 8.192M (6 against om finals, 6 against sp finals), evaluation 36 cells × 128 pairs × 500 turns | **6.6 h + 0.5 h** |
| optional om_table ablation | 6 attackers, belief tables, 30 more cells including the 12 live/table cross-evaluation cells | +3.7 h + 0.03 h + 0.5 h |

**Horizon exceptions** (design §10.1):
- **Exception 1 (60 turns).** It comes from the plans, not the policies. Keep it.
- **The 300-turn validation subset.**
  - It is necessary for the om finals, a new history-dependent policy kind that no earlier validation covers.
  - For the sp finals it partly duplicates M4.5c's lineage evidence.
- **The 100- and 150-turn cells** keep comparability with the M4 limitation records. Keep them.
- **The 500-turn games of the families stage** are whole games, already run, and not strategic exceptions.

**Reductions available:**
- R1: drop the sp validation, saving about 0.9 h and losing the sp finals' horizon check;
- R2: validate at 150 turns, saving about 2.1 h and losing detection between turns 150 and 300;
- R3: a reduced battery, saving about 3.3 h and halving the natural states per family.

The recommendation is the full design; R1 is the only reduction whose loss is largely covered by established
evidence.

## 10. The next decision for the user

1. **Approve the strategic battery and the limitation cells** (`--confirm-m5-eval` for `battery` and `cells`): the
   full design (recommended, about 11.8 h plus a validation contingency), or with R1 / R2 / R3.
2. **Approve the core attacker stage** (`--confirm-m5-attackers`): 12 attackers at the quarter budget (recommended),
   or half (about 13.3 h training).
3. **Decide on the optional om_table ablation** (`--include-om-table`). It is exploratory, read with the limits in
   design §7.1.

Nothing else is pending. Training and the authorized evaluations are final, and M5e (closeout) follows the stages
above.
