# M5a: campaign design, pre-registration and readiness

*2026-10-17. Status: **complete; the campaign is not launched and awaits the user's decisions and approval.***
- *Design: [m5_design.md](m5_design.md).*
- *Pre-registration: [m5_preregistration.md](m5_preregistration.md), frozen and awaiting campaign authorization (SHA-256
  `d2424022bd07535f…af7304`, in `results/m5/preregistration.json`).*
- *Readiness protocol: [m5a_readiness_protocol.md](m5a_readiness_protocol.md), recorded before the first check.*
- *Records: `results/m5/readiness.json`, `results/m5/budget.json`, `results/m5/report.md`.*

## Summary

**What M5a delivers.**
- **A specified two-arm campaign, with M5b–M5e, endpoints, classifications and gates.**
  - A: opponent modelling (OM2's treatment collector).
  - B: pure self-play, the M4.6c continuation.
  - Six matched pairs from `ext/baseline:s`, 32.768M transitions each.
- **The package `lanegame/experiments/rl/m5/`.** Its training and evaluation drivers import the historical instruments
  and work for both input formats.
- **A bounded readiness check.** All 11 experimental parts pass. About 20 minutes of experimental compute plus the
  regression suite.

**What readiness verified** (protocol R0–R10, all passed):
- **R0 lineage.** Every start is the M4.6c 8.192M continuation (PPO seed 98,000 + s, archive share 25%) of the M4.5c
  fresh 16.384M final of the same seed (digest-checked). Prior training is 24.576M; after M5 it would be 57.344M over
  three schedules. Every dependency is present. M4.6a gap specialists exist for seeds 0, 2 and 4 only.
- **R1 seeds.** The block 70,000–70,999 contains no root constant of any other study.
- **R2 identical start.** The widened om network equals `ext/baseline:s` bit for bit in batched evaluation, with any
  belief and any previous action, for all six seeds. A single-row forward pass differs by at most 9.5·10⁻⁷.
- **R3 nothing earlier changed.**
  - The historical digest pins and OM2's default-trainer pin pass.
  - Stored records are reproduced exactly through the hooked shared loops:
    - M4.5c battery jobs (4: natural, scripted-protocol and constructed);
    - M4.6c weaker-side cells of `ext/baseline:0` (3);
    - M4.6f's cross-candidate freeze cells in the C1 environment (4);
    - OM2 evaluation games of both pilot arms (3 cells).
- **R4 smoke.** Each arm ran about 10 updates of the real schedule, and a re-run of each was bit-identical in weights
  and history.
  - sp: 100% current-policy transitions, equal seats.
  - om: library and self-play games, belief features that are valid posteriors and vary, and the previous-action field.
  - Both: no illegal action, KL < 0.05, archive starts.
- **R5 common random numbers.** The first 200 game starts are identical in both arms: kind (84 s0, 116 archive),
  archive state, mirroring, length and chance seed.
- **R6 inputs.**
  - Training inputs equal evaluation inputs: one full om rollout's 2,863 inputs and 14 truncation bootstraps were rebuilt
    independently by `OMPolicy` from the start-of-turn states and past codes alone.
  - The state and previous-action blocks match bit for bit; the belief block within 2.9·10⁻⁶ (float32 rounding).
  - The belief is the prior at every game start, s0 and archive alike.
  - The sp input is exactly the 90-entry observation.
- **R7 opponents.** 70,000 simulated draws cover all 15 fixed opponents within 10% of uniform; fixed-opponent games are
  50% of draws.
  - Measured in training: 51% of games but **36–37% of learner transitions**, 1.46 transitions per environment turn.
- **R8 equivalences.**
  - A widened start in the om path equals the 90-input start in a battery job: G and Q within 10⁻⁵, the same first
    actions.
  - `decide` is pure.
  - The opponent's code enters the input only after the turn.
- **R9 concurrency.**
  - Six workers is the efficient count: om aggregate 5,718 / 6,002 / 7,329 / 7,562 transitions/s at 4 / 5 / 6 / 8
    runs. Eight runs slow each om run by 23% for +3%.
  - The simulated campaign is shortest with six workers and the om runs first.
- **R10 instruments.** Every M5b–M5d instrument ran end to end at reduced samples on stand-ins, and its steady-state
  cost was measured. The instruments: s0 families including imitate-then-switch and conditional styles; long games;
  the battery with recorded om beliefs; the seven cell families; the lane-exit readout from recorded om states; the
  attackers' belief tables.

**Proposed changes and decisions** (design §11–§12):
1. **Attacker budget: a quarter** (8.192M), not half: 10.3 h against 20.6 h for 18 attackers.
2. **An adaptation-exploitation test:** dedicated attackers against the om final with its belief frozen at its typical
   self-play values.
3. **Six seed pairs kept.** With OM2's competent-opponent variance, a true null competence effect would be
   called preserved only 42% of the time (ten pairs: 71%), so the competence endpoint may end inconclusive.
4. **The full battery for both arms** (9.3 h); a reduced design saves about 3 h.
5. **Six workers, om runs first.**

**Measured budget** (6 workers):
- training 9.4–11.8 h;
- evaluation 13.7 h;
- attackers 10.3 h (quarter) or 20.6 h (half);
- total about 34–36 h with quarter-budget attackers.

## 1. What was built

| piece | where | notes |
|---|---|---|
| arms, seeds, budgets, specs, provenance | `m5/common.py` | seed block 70,000–70,999; `checkpoint_identity`, `code_digest` |
| training, manifests, health checks, restarts | `m5/train.py` | imports `om2.collector.om_collector_cls("treatment")` and `m46c.train.collector()`; the restart moves an incomplete run to `partial/` and its re-run must reproduce the prefix |
| history-dependent policies | `m5/players.py` | `OMPolicy`: belief, previous codes, pure `decide`, `observe_turn`, `value_at`; `crn_policy` for 90 / 103 / 116 inputs |
| s0 games | `m5/games.py` | reproduces OM2's evaluation exactly; imitate-then-switch, conditional styles; the lane-exit panel at turn 94 |
| battery, cells | `m5/battery.py`, `m5/cells.py` | M4.5c / M4.6 instruments for both formats; the belief conventions of design §4.1 |
| attackers | `m5/attackers.py` | `train_shared` with a per-game stateful target; belief tables for `om_table` |
| designs, stages | `m5/evaluate.py` | 546 family cells and 1,008 monitoring cells, unique seeds; gated stages |
| statistics | `m5/analysis.py` | interval classifications, the headline, Holm, the variance split, the non-inferiority resolution |
| readiness | `m5/readiness.py`, `m5/bench.py`, `m5/instruments.py` | R0–R10; the budget |
| CLI | `m5/__main__.py` | the readiness stages; `--confirm-m5-train`, `--confirm-m5-eval`, `--confirm-m5-attackers` |
| shared code, opt-in only | `diagnostics/lane_exit.py`, `m45b/engine.py`, `m45c/horizon.py`, `m46a/engine.py`, `m46a/policies.py`, `m46d/plans.py`, `m46d/specialists.py` | the `observe_turn` hook in four loops, `inner_policies` in six plan wrappers, `value_at` in the critic; no change for existing policies (R3) |
| tests | `tests/test_m5.py` (20) | below |

## 2. Readiness results in detail

**Stand-ins:** the OM2 treatment finals (om format) and the M4.7 control finals (sp format: pure self-play continuations
of `ext/baseline:s`). Their values say nothing about M5's outcomes.

| check | result |
|---|---|
| R0 lineage and artifacts | pass (25 items) |
| R1 seed block | pass |
| R2 initialization | pass (18 items; single-row ≤ 9.5·10⁻⁷) |
| R3 defaults and exact reproductions | pass (pins; 4 + 3 + 4 records and 3 cells exact) |
| R4 smoke and re-runs | pass (28 items) |
| R5 common random numbers | pass (200 / 200 starts identical) |
| R6 inputs | pass; **first run failed**, see below |
| R7 opponents and shares | pass |
| R8 equivalences | pass |
| R9 concurrency | pass (8 configurations, 12 min) |
| R10 instruments | pass; **first two runs failed**, see below |

**Failures and fixes** (all implementation defects in the readiness code, none in the campaign code):
- **R4, first attempt.** It was started from a stdin script, which Windows multiprocessing cannot spawn. It was re-run
  through the module's CLI.
- **R6, first run.** The reconstruction compared whole inputs bit for bit and failed on the belief block only, with a
  maximum difference of 2.9·10⁻⁶. Training batches the five library networks' forward passes across environments;
  the evaluator evaluates one item. That is float32 rounding of the same sums.
  - The check now demands bit-exact state and previous-action blocks and a belief within 10⁻⁵, the same tolerance R2
    and R8 use for the same effect. Both runs are recorded.
- **R10, first runs.**
  - The battery job tried to collect natural states for a constructed family, which has no predicate.
  - A check read a field that one job type does not have.
  - Both were fixed. Instrument jobs now return an error record instead of aborting the whole check.
- **R10, the first rates.** These came from jobs too small to amortize start-up (250–400 turns/s, against OM2's measured
  800 for full cells). A steady-state measurement (a warm-up, then a timed segment, six classes at once) was added. It
  supplies the budget's rates.

**Reduced-sample readouts** (weak evidence, reported as such):
- **Lane-exit readout from recorded om states** (8 states at turn 94, K 8). E_20 with the recorded beliefs is
  −0.35 ± 1.21 (SE) from the games' own lead at turn 114. With beliefs reset to the prior it is −0.70 ± 1.27.
  - Accounting is exact in every continuation.
  - Eight states cannot distinguish the two conventions. The pre-registered conventions follow training, not this
    check.
- **Stand-in leads at turn 114** (6 or 4 pairs):
  - the om stand-in: +15.0 against `g_shove` and −2.3 against `ext/baseline:3`;
  - the sp stand-in: +5.2 against `g_shove`.

  These are the direction OM2 reported, with intervals too wide to read anything into.

**Throughput and budget.** See design §10. The short-run benchmark under-reads the om arm against OM2's full runs
(1,200 against 1,416–1,645 at 5 workers) and over-reads the sp arm (2,805 against 2,450–2,620). The budget gives both
calibrations as a range. The om cost is dominated by the belief's per-turn likelihoods.

## 3. Tests

`tests/test_m5.py` (20 tests, 8 s):
- **The seed block and specifications:** disjoint seeds (a scan of every experiment package); the policy specifications
  and the two configurations being one game; the campaign gates refusing without their flags.
- **The hook:** reaches each stateful policy once, even inside plans; leaves stateless rollouts unchanged.
- **The om path:**
  - batched exactness of the widened start;
  - common game starts;
  - every training input rebuilt by `OMPolicy`;
  - M5 games equal OM2's;
  - a pure `decide`;
  - the belief modes;
  - an exact M4.6c cell reproduction.
- **The design:** unique family seeds and paths.
- **The statistics:** the classifications and the headline; Holm, the variance split and the resolution; the
  secondary endpoints' empty case and one-sided p-values; the replication rule; share accounting and the prefix
  check; the scheduler.

Tests needing git-ignored checkpoints skip when they are absent. Here all ran.

**Full regression suite** (`python -m pytest`, final code): **1,511 passed, 4 skipped, 2 xfailed** in 8 min 7 s
(log: `results/m5/regression_suite.log`). That is 1,491 before M5a plus the 20 new tests.
- The 4 skips are the slow tests that need `--runslow`: one in `test_legacy_pins.py` (line 76), two in
  `test_m4_observation.py`, one in `test_m4_tower.py`. **They were not run here and are not verified by this
  readiness.** Every non-slow digest pin passed.
- The 2 xfails are expected failures from earlier milestones.

## 4. What remains (the campaign)

Nothing in M5b–M5e has run. One piece of analysis code is still to be assembled: S8, the per-arm strategic
replication and the cross-arm M5c contrasts. It will reuse the M4.5c battery analysis and the M4.6c / M4.6f cell
analyses on the M5 records. The M5d analysis (P1, P2a, P2b and S1–S7) is implemented in `analysis.analyse` and
unit-tested on synthetic inputs. The campaign needs:
- the user's decisions (design §12): seed count, attacker budget, the adaptation test, the battery size;
- the approval flags `--confirm-m5-train`, `--confirm-m5-eval` and `--confirm-m5-attackers`.

Order: training (9.4–11.8 h), then the evaluation (13.7 h) and the attackers (10.3 h at a quarter budget). Their order
is free once the finals exist.

**Reproduce M5a:**

```bash
python -m lanegame.experiments.rl.m5 ready_protocol ready budget report --workers 6
python -m pytest tests/test_m5.py
```
