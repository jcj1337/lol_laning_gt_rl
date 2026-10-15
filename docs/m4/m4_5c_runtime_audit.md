# M4.5c — Runtime and experimental-scope audit before full training

*2026-10-14. Investigation only: no training, no change to the experiment, nothing staged or committed. It audits the
plan of the [M4.5c pilot report](m4_5c_pilot_report.md) §7–§9 (6 seeds × 8,192,000 agent transitions, "≈ 2.5 h").
Two new measurements were taken for this audit:
- a **sustained 6-process benchmark**: the exact per-update work of training (collect, GAE, PPO update) at learning
  rate 0 (weights verified unchanged, no checkpoints), with lane caches 20,000 and 5,000;
- a one-job **evaluation memory** check.

Their scripts and raw output are in the session scratchpad; the numbers are quoted here. Everything else comes from
repository artifacts: logs, JSON, file timestamps.*

---

## 0. Findings in brief

1. **M4.4 took about 9.7 h of wall clock in three blocks**: M4.4a ≈ 0.3 h, M4.4b 2.0 h, M4.4c 7.35 h. M4.4c includes
   1.8 h lost to a Recall run that crashed at its final save. The "7–8 hours" matches either M4.4c alone (7.35 h) or
   M4.4b plus M4.4c's successful stages (≈ 7.5 h). The records cannot say which grouping was meant.
2. **The 2.5 h estimate for M4.5c is too low, by about 4×.** The training part holds up: 43–50 min, now from a
   sustained measurement. The evaluation part does not.
   - The counterfactual battery was priced at 15.6 M simulated turns, against **231 M** in M4.4c: 15× less volume,
     with no opponent protocols, 100-turn rollouts instead of 300, and no replication.
   - The best-response attackers were excluded from the estimate altogether.
   - **Revised central estimate: ≈ 10 h** (plausible 7–14 h). Training is about 0.8 h of it.
3. **The six-process throughput figures are not an error, but the report presented them badly.**
   - 2,936 per process is the training loop's own rate.
   - 12,499 is the total transitions divided by the wall clock of the whole multi-process call. That wall clock
     includes ≈ 4.5 s of Windows process start-up on runs of only 11 s.
   - The sustained steady state at 6 processes is **3,200 transitions/s per process**, measured.
4. **Memory needs a correction.** The machine has 15.9 GiB in total; other applications typically use about 6 GiB.
   - With 6 full-cache processes at `lane_cache = 20,000`, available memory fell to 0.01 GB and Windows paged about
     4 GiB of other applications to disk.
   - At `lane_cache = 5,000`: 0.64 GB per process, at least 4.3 GiB free, and equal or slightly better throughput.
     Results are identical (tested).
5. **6 × 8,192,000 remains appropriate.** The evaluation plan needs to be brought up to M4.4c scale before the
   strategic campaign, and the execution plan needs six small corrections (§6).

---

## 1. What M4.4 actually did

### 1.1 Timeline (from log start / end stamps, file modification times and recorded durations)

| block | stage | start → end | duration | processes | notes |
|---|---|---|---|---|---|
| M4.4a (Oct 13) | audit, smoke_train (9 × 204,800), smoke_repro (3 runs), smoke_eval, demo, plan, regression, report | ≈ 19:35 → 19:52:37 | ≈ 0.3 h | 9 | smoke training 93 s, re-run 76 s; regression 7.4 min |
| M4.4b (Oct 13) | `train_main`: 9 runs × 2,048,000 | 21:11:29 → 21:24:47 | 797 s | 9 training processes | per-run 677–789 s |
| | `repro_main`: 3 runs × 2,048,000 (full-budget re-runs, bit-identical check) | 21:25:10 → 21:39:06 | 834 s | 3 | **concurrent with eval_main** |
| | `eval_main` (pre-registered evaluation) | 21:25:10 → 22:15:33 | 3,022 s | 9 workers | 1,044 matchups in 2,006 s, then behaviour, counterfactuals, traces; **concurrent with repro_main, then with extra_main** |
| | `extra_main` (tournament, extra references, snapshot behaviour, integrity, transfer test) | 21:39:12 → 22:21:04 | 2,510 s | 9 workers | 396 matchups in 2,162 s; overlapped eval_main for 36 min (18 processes on 12 CPUs) |
| | `traces_main`, `recall_main` | 22:15:46 → 22:20:41 | 32 s + 21 s | | |
| | `train_ext`: 9 runs × 4,096,000 (matched 2× extension) | 22:23:28 → 22:57:03 | 2,013 s | 9 | the regression suite (19 min) ran **concurrently** |
| | `eval_ext` | 22:57:17 → 23:11:36 | 857 s | 9 workers | 288 matchups in 747 s |
| M4.4c (Oct 13–14) | first `recall` attempt | ≈ 23:41 → 01:29:15 | 6,463 s | 11 workers | **crashed at the final save** (`TypeError` in `meta()`); results lost; resumable runner added afterwards |
| | `recall`, `recall_rep`, `freeze`, `freeze_init`, `freeze_rep`, `zoning`, `zoning_nat`, `zoning_mech`, `vision` (+ init), `waves_mech`, `waves`, `traces` | 01:33:46 → 06:53:35 | 19,194 s (5.33 h) | 11 workers, sequential queue (`queue.sh`) | 5,982 states, 231 M rollout turns (§1.3) |
| | regression, report | → 07:02:32 | ≈ 9 min | | regression 7.4 min |

Wall clock, gaps between blocks excluded: **M4.4a ≈ 0.3 h + M4.4b 2.00 h + M4.4c 7.35 h ≈ 9.7 h.**
- M4.4c alone was 7.35 h; without the crashed run, 5.6 h.
- M4.4b's stage durations add up to 2.8 h, but overlapping stages compressed that into 2.0 h of wall clock. The
  overlap also inflated each stage's own duration.

### 1.2 Answers to the specific questions

| question | answer (evidence) |
|---|---|
| independent policies trained | 9 primary (3 environments × seeds 0–2) at 2,048,000, and 9 more at 4,096,000 (the matched extension: fresh runs with the same seeds, not continuations) = **18 policies**; plus 3 bit-identical full-budget re-runs and 9 smoke runs |
| configurations and seeds | 3 environments (`four`, `econ`, `aggro`) × 3 seeds |
| transitions per run | 2,048,000 (main), 4,096,000 (extension), 204,800 (smoke) |
| total training transitions | 18.43 M (main) + 36.86 M (ext) + 6.14 M (re-runs) + 2.46 M (smoke and its re-runs) = **63.9 M** |
| sequential or concurrent | each stage trained its runs **concurrently**: 9 processes (main, ext, smoke), 3 (re-runs); stages followed one another, except the overlaps marked above |
| what `--workers` controlled | **both**, stage by stage: `m44 train(...)` passes it to `run_jobs` as the number of concurrent training processes (9); the evaluation stages pass it as the number of evaluation worker processes (9 in M4.4b, 11 in M4.4c). One flag, two meanings, never both at once within a stage |
| measured training time | 797 + 834 + 2,013 s (+ 169 s smoke) = **1.0 h** |
| measured evaluation time | M4.4b stages 3,022 + 2,510 + 32 + 21 + 857 s = 1.8 h of stage time (partly overlapped); M4.4c 5.33 h, plus the 1.8 h crashed run |
| other contributors | 3–4 regression runs (7–19 min each), report generation, the M4.4a demonstration on the M4.3c pilot finals |
| reruns and failures | the deliberate full-budget re-runs (834 s); **one failed run, 6,463 s, lost to a save bug**; no other failed attempts in the logs |

### 1.3 M4.4c's simulated volume (computed from the raw results)

| stage | states | rollouts (states × candidates × rollouts per candidate) | rollout turns (× 300) | wall | turns per worker-second |
|---|---|---|---|---|---|
| recall | 1,584 | 228,096 | 68.4 M | 81.9 min | 1,266 |
| recall_rep (2.05M finals) | 336 | 48,384 | 14.5 M | 18.5 min | 1,192 |
| freeze | 792 | 95,040 | 28.5 M | 36.8 min | 1,173 |
| freeze_init | 576 | 69,120 | 20.7 M | 27.9 min | 1,126 |
| freeze_rep | 144 | 17,280 | 5.2 M | 7.3 min | 1,078 |
| zoning | 528 | 76,032 | 22.8 M | 36.4 min | 950 |
| zoning_nat | 360 | 51,840 | 15.6 M | 28.0 min | 842 |
| zoning_mech | 126 | 12,096 | 3.6 M | 5.9 min | 940 |
| vision | 684 | 82,080 | 24.6 M | 32.5 min | 1,147 |
| vision_init | 60 | 14,400 | 4.3 M | 7.3 min | 893 |
| waves | 792 | 76,032 | 22.8 M | 36.3 min | 951 |
| **total** | **5,982** | | **231 M** | **5.31 h** | **1,099** |

Every M4.4c candidate was evaluated:
- with 24 paired rollouts of 300 turns (waves: 16);
- against 2 to 5 opponent protocols (the policy itself, its modal action, and style opponents: breaker, passive,
  stepback, respect, shover);
- on 9 primary policies, with replication of Recall and freeze on 6 of the 2.05M finals.

---

## 2. Side-by-side: M4.4 actual vs M4.5c

**M** = measured; **P** = projected from measured rates.

| dimension | M4.4 actual (M) | M4.5c as proposed in the pilot report (P) | M4.5c revised (P, §5) |
|---|---|---|---|
| configurations | 3 environments | 1 (`m45_calibrated()`) | 1 |
| independent seeds | 3 per environment (9 policies; 18 with the extension) | 6 | 6 (+ 1 full-budget re-run of seed 0) |
| transitions per policy | 2,048,000 and 4,096,000 | 8,192,000 | 8,192,000 |
| total training transitions | 63.9 M (incl. extension, re-runs, smoke) | 49.2 M | 57.3 M with the re-run |
| training concurrency | 9 processes (3 for re-runs) | 6 | 7 (6 + re-run) |
| training throughput per process | 2,747 (main), 2,136 (ext, with tests running alongside); aggregate 23,140 / 18,311 | 2,936 short-run; 12,499 "aggregate" (§3) | **3,200 sustained (M, lane cache 5,000)** |
| training duration | 1.0 h | 46 min | **45–55 min** |
| fixed-opponent evaluations | 90 headline matchups (10 references, 32 seeds, W = 1000) + 900 curve matchups (20 points × 5 references, 16 seeds, W = 460) + extra references | 1,260 curve matchups (21 points × 10 opponents, 32 seeds) + 120 headline (64 seeds, W = 1000) | same |
| historical checkpoints | 45 history + 324 tournament matchups; extension vs main finals | 54 history + 216 tournament | same |
| cross-seed | 9 pairs (3 per environment) + 18 labelled transfer matchups | 15 pairs (64 seeds, W = 1000) | same |
| counterfactual strategy | eval_main 5 × 8 states × 16 rollouts per final (≈ 7 M turns) + **M4.4c: 5,982 states, 231 M turns, 5.3 h** | 6 families × 48 states × 16 rollouts × 100 turns, policy opponent only: **15.6 M turns, 0.35 h** | M4.4c-scale battery adapted to M4.5: **≈ 212 M turns, ≈ 5.4 h** |
| best response / exploitability | **none** (M4.4d postponed into M4.5c) | mentioned, **excluded from the estimate** | 6 attackers: **≈ 1.0 h** |
| total evaluation duration | ≈ 7.1 h of stage time (M4.4b 1.8 h + M4.4c 5.3 h) | 1.8 h | **≈ 8.1 h** |
| total wall clock | ≈ 9.7 h (≈ 7.9 h without the crashed run) | 2.5 h | **≈ 10 h (7–14 h)** |

**Coverage.** The pilot report's evaluation plan was comparable to M4.4 in its strength measurements: curves,
headline, history, tournament and cross-play are equal or larger. It was **not** comparable to M4.4c in its
counterfactuals, which is where M4.4 spent 74% of its evaluation time:
- the volume was 15× smaller;
- there were no opponent protocols, so the plans were judged only against the policy itself, where M4.4c also used
  the modal action and style opponents;
- the horizon was 100 turns, against 300;
- there was no replication on a second checkpoint;
- no mechanistic enumerations were included;
- there was no resumability, the lesson of M4.4c's crashed run.

---

## 3. Reconciling the M4.5c throughput figures

**What each number is** (`m45c/train.py`):
- `per_process_mean` averages each run's `agent_transitions / res.elapsed`. `res.elapsed` is `train_shared`'s own
  timer: it starts after the network and the 16 environments are built, and ends after the last update.
- `aggregate_tps` is the sum of transitions divided by the wall clock of the whole `run_jobs` call. That wall clock
  includes `ProcessPoolExecutor` start-up on Windows (spawn, importing torch and the package in every process),
  building each run's environments, and the end-of-run digest and value computations.

**Reconstruction.** For six processes:

| quantity | value |
|---|---|
| total transitions | 6 × 32,768 = 196,608 |
| wall clock at 12,499/s | 15.7 s |
| each run's own loop time at 2,936/s | 11.2 s |
| fixed overhead | ≈ 4.5 s |

The same reconstruction gives about 3.7 s at 3 processes and 6.3 s at 11. At 1 process the jobs run in-process, the
overhead is 0.1 s, and the two figures agree (4,223 vs 4,173). So this is **not a calculation error**. It is a fixed
start-up cost that dominated an 8-update benchmark (about 30% of its wall clock), and the report did not explain it.
For a 45-minute run it is under 0.5%.

The short benchmark was **not** fully representative: 8 updates from the untrained network, with nearly empty caches.
It was replaced for this audit by a sustained measurement: the pilot's final network, 180 updates per process, caches
full after about 110 updates, learning rate 0.

| lane cache | per process, updates 2–30 | per process, updates 121–180 (caches full) | spread across the 6 | peak per process | minimum system RAM available |
|---|---|---|---|---|---|
| 20,000 | 3,090–3,152 | **3,140–3,216** | < 3% | 1.32–1.39 GB (working set trimmed by paging) | **0.01 GB**: Windows paged ≈ 4 GiB of other applications to disk |
| 5,000 | 3,075–3,143 | **3,196–3,224** | < 1% | 0.64 GB | **4.27 GiB** |

**Reading.**
- Sustained throughput does not fall as the caches fill. Full caches are slightly faster.
- The trained network is not slower than the untrained one.
- The 6-process steady state (3,200) is 7–9% above the short benchmark's 2,936.
- Cache growth changes memory, not speed: the pilot reached 1.60 GB alone at cache 20,000.
- At 20,000 six processes do **not** fit comfortably on this machine as it is normally used:
  - total 15.9 GiB;
  - Firefox, VS Code, memory compression and the system held about 6 GiB during the audit;
  - only 3.0 GiB was available before the first benchmark.
- At 5,000 they fit with margin, and the cache test shows bit-identical training at any cache size.

**Recalculated training time for six concurrent 8,192,000-transition runs:**
- 8,192,000 / 3,200 = 2,560 s = **42.7 min** per run.
- With process start-up, snapshot writes (18 ms each, 320 per run) and a straggler allowance (≤ 1% spread measured;
  M4.4b's runs spread 15%, but across different environments): **45 min central, 43–55 min range**.
- An optional 7th process for a full-budget re-run of seed 0 costs about 3–5% per process (interpolated from the
  scaling curve).

**Worker counts.**
- Training at 6–7 processes on 12 logical CPUs is safe at cache 5,000 (≈ 4.5 GB).
- Evaluation at 11 workers is safe: one headline-size evaluation job peaks at 338 MB, so 11 workers need ≈ 3.7 GB.
  An evaluation job runs at 2,392 turns/s alone and ≈ 1,150 per worker with 11 workers (measured in the pilot).

**Concurrency assumptions.** Nothing in the M4.5c estimate assumed two stages running at once. Training, standard
evaluation and the battery were summed sequentially. M4.4b did overlap stages, which compressed its wall clock but
slowed the stages themselves:
- the extension runs dropped to 1,500–1,600 transitions/s per process (from 2,400–2,600) exactly while the regression
  suite ran alongside them;
- eval_main and extra_main ran 18 processes on 12 CPUs.

The M4.5c campaign should run training alone on the machine.

---

## 4. The full M4.5c evaluation workload

| item | in the pilot report's 2.5 h? | core or optional | revised cost (P) |
|---|---|---|---|
| fixed-opponent learning curves (21 points × 10 opponents × 6 seeds) | yes (49 min) | core | 49 min |
| midpoint and final headline (64 seeds, W = 1000) | yes (20 min) | core | 20 min |
| history (final vs 8 snapshots + untrained) | yes (2 min) | core | 2 min |
| historical tournament (9 checkpoints, round robin, per seed) | yes (4 min) | core | 4 min |
| cross-seed play (15 pairs) | yes (3 min) | core | 3 min |
| state-conditioned behaviour (records + ledger, 5 checkpoints) | yes (7 min) | core | 7 min |
| integrity checks, representative-trace search, interrupted-Recall follow-up (M4.4b) | **no** | core (cheap) | ≈ 5 min |
| viewer trajectories | **no** | core (cheap) | ≈ 2 min |
| counterfactual families A–E at M4.4c scale (below) | **no**: 0.35 h for a 15× smaller design | core | **≈ 5.4 h** |
| mechanistic enumerations (waves, zoning; M4.4c `*_mech`) | **no** | core | ≈ 0.1 h (included above) |
| learned best-response attackers (one per final) | **no** | core (the former M4.4d) | **≈ 1.0 h** (or 0.3 h at the smaller budget, below) |
| attackers on the midpoints; a 3×-budget attacker check | no | optional | ≈ +1.0 h / +0.6 h |
| full-budget reproducibility re-run of seed 0 (M4.4b practice) | no | recommended | ≈ +3 min (a 7th concurrent process) |
| regression suite, pre-flight smoke, report and figures | no | core | ≈ 15 min |
| the matched 2× extension (if §8.4 of the pilot report triggers it) | no | contingent, separate approval | ≈ 1.5 h training + 0.5 h evaluation |

### 4.1 A counterfactual battery comparable to M4.4c

The sizes are M4.4c's own: 24 paired rollouts of 300 turns (waves: 16), the same opponent protocols, initialized
grids where natural states are rare. Turns are per final:

| family | states | candidates | protocols | turns |
|---|---|---|---|---|
| E Recall: contested / repeated / retreat (80) + safe (16) | 96 | 6 | 2 (safe 1) | 7.60 M |
| B freeze, natural | 24 | 5 | 5 | 4.32 M |
| B freeze, initialized grid | 24 | 5 | 4 | 3.46 M |
| B zoning, initialized grid | 16 | 6 | 5 | 3.46 M |
| B zoning, natural | 24 | 6 | 2 | 2.07 M |
| B zoning, behaviour matrix | 6 | 3 | 3 (32 rollouts) | 0.52 M |
| D Vision (Ward), natural + initialized | 60 + 4 (48 rollouts) | 5 | 2 | 4.90 M |
| A waves, sequences | 24 | 6 | 5 (16 rollouts) | 3.46 M |
| C trading into waves (new in M4.5): natural + initialized grid (level × wave size × power) | 24 + 16 | 4 | 2 | 2.30 M |
| **per final** | | | | **32.1 M** |

Totals:
- × 6 finals = 192.5 M turns.
- Plus a replication of Recall and freeze on the 6 midpoint snapshots, at M4.4c's replication scale: 19.7 M.
- **≈ 212 M turns**, about M4.4c's 231 M.

At M4.4c's measured 1,099 turns per worker-second, adjusted for M4.5's ≈ 10% costlier environment step, that is
**≈ 1,000 per worker-second, so ≈ 5.4 h on 11 workers**. The M4.5c framework's own smoke ran at ≈ 1,240. The range is
**3.6–7.1 h**: 900–1,200 turns per worker-second, and ±20% on the final family sizes.

**Development needed first** (not compute). `m45c/strategic.py` has the families and the paired engine. It still
needs:
- opponent protocols (modal and the style opponents);
- the initialized grids for M4.5;
- the midpoint replication;
- resumable stages (M4.4c's `run_resumable`).

This should be implemented and smoke-tested before the battery runs.

### 4.2 Best-response attackers

The protocol the pilot report cites (M4.3b, `m43b/exploit.py`):
- **one attacker per final**, warm-started from the target's weights;
- `train_ppo` against the frozen target, so only the learner's seat is trained: one transition per joint step;
- **budget: half the target's**;
- evaluated against the target (W = 460 and 1000) and against the other seeds' finals (transfer).

The pilot report also said "about 1M transitions each", which contradicts the protocol for 8.192M targets. It must
be resolved:

| variant | attackers | transitions each (= joint steps) | training (6 concurrent) | evaluation | total |
|---|---|---|---|---|---|
| **protocol (half the target's budget)** | 6 | 4,096,000 | ≈ 53 min | ≈ 4 min | **≈ 1.0 h** (0.8–1.3) |
| smaller budget | 6 | 1,024,000 | ≈ 13 min | ≈ 4 min | ≈ 0.3 h |

How the training time was derived:
- 6-process self-play runs at 1,600 joint steps/s.
- An attacker queries its frozen opponent one observation at a time. M4.3b measured attackers at about 0.8× the
  self-play joint-step rate under load (17 min per 1,024,000).
- So about 1,280 joint steps/s per attacker.

The protocol variant gives the stronger lower bound and is the M4.4d design. The smaller budget is cheaper, but a
null result from it would say less.

---

## 5. Revised estimate

| component | central | plausible range | basis |
|---|---|---|---|
| 1. PPO training: 6 × 8,192,000, plus seed 0 re-run concurrently | **0.8 h** | 0.7–0.9 h | sustained 6-process measurement (M); straggler allowance |
| 2. standard evaluation | **1.5 h** | 1.3–1.9 h | measured evaluation rates; M4.4a's evaluation estimate was 35% below M4.4b's actual (37 vs 50 min), though the actual ran under contention, so the upper end allows for that |
| 3. state-conditioned strategic evaluation (M4.4c scale) | **5.4 h** | 3.6–7.1 h | M4.4c's measured volume and rate (§4.1) |
| 4. best-response attackers (protocol budget) | **1.0 h** | 0.3–1.3 h | M4.3b protocol, measured rates (§4.2); the low end is the 1.0M variant |
| 5. regression, pre-flight smoke, reports | 0.3 h | 0.2–0.4 h | measured (regression 7.6 min in this session) |
| margin for reruns and failures | **1.3 h** (≈ 15%) | 0.5–2.5 h | M4.4c lost 27% of its stage time to one failure; the margin assumes resumable stages |
| **total** | **≈ 10 h** | **≈ 7–14 h** | |

### Why M4.5c can be faster or slower than M4.4

Its training is not larger in total: 49.2 M agent transitions (57.3 M with the re-run), against M4.4's 63.9 M. It is
2× longer per policy, but there is one environment, not three.

**Faster:**
- one environment;
- a higher per-process rate at 6 processes than at 9 (3,200 vs 2,136–2,747);
- no stages competing with training;
- no planned extension: it is contingent on the plateau criteria, whereas M4.4b ran one;
- resumable evaluation stages should avoid M4.4c's 1.8 h loss.

**Slower:**
- each M4.5 environment step costs about 10% more;
- the per-policy standard evaluation is heavier: 21 curve points against 10 opponents at 32 seeds; a 64-seed
  headline; 15 cross-play pairs;
- the counterfactual battery adds trading-into-waves and Ward families that M4.4c did not have;
- the attackers are new work (M4.4d was never run);
- 6 policies get the full battery, against M4.4c's 9 primary policies, so roughly a wash with the added families.

**Net:** about the same order as M4.4 (≈ 10 h vs ≈ 9.7 h). The training share is smaller and the evaluation share
larger.

---

## 6. Recommendation

- **Budget: 6 × 8,192,000 remains appropriate.** The audit confirms the training cost with a sustained measurement
  (about 45 min). The budget was chosen on learning evidence, and nothing here argues for changing it.
- **The evaluation plan is not yet comprehensive enough.** Its strength measurements match or exceed M4.4's. Its
  counterfactual battery must be brought to M4.4c scale before it runs (§4.1). Attackers must be included at the
  protocol budget (§4.2), and the cheap M4.4b items added: integrity checks, trace search, the interrupted-Recall
  follow-up, viewer recordings.
- **The 2.5 h estimate should be revised to ≈ 10 h (7–14 h).** About 0.8 h is training. About 9 h is evaluation, of
  which 5.4 h is the M4.4c-scale counterfactual battery and 1.0 h the attackers.
- **M4.4 work the current plan would have skipped:**
  - M4.4c-scale counterfactuals with opponent protocols, initialized grids, replication and mechanistic enumerations;
  - the full-budget reproducibility re-run;
  - the integrity checks;
  - the representative-trace search;
  - the interrupted-Recall follow-up.

  The best-response attackers were never run in M4.4. They are owed from M4.4d and were left out of the estimate.

### Corrections to the execution plan before approval

1. **`lane_cache = 5,000`** for the training runs. Results are identical (tested), throughput is equal or better under
   load, and at 20,000 the machine ran out of RAM.
2. **Training alone on the machine**: no tests or evaluations in parallel. M4.4b's extension lost up to 40% per update
   while the regression suite ran alongside.
3. **A full-budget re-run of seed 0** as a 7th concurrent process, for the bit-identity check M4.4b made (about 3 min
   more wall clock).
4. **Report corrections**:
   - throughput: sustained 3,200 per process at 6 processes; "aggregate" figures of short runs include start-up;
   - memory: 15.9 GiB total, about 9–10 GiB typically available;
   - attackers: the protocol budget is half the target's.
5. **Two approvals instead of one**, as the original brief separated them:
   - training, about 1 h including the pre-flight smoke and reproducibility check;
   - then the post-training evaluation, about 9 h, after the counterfactual battery has been brought to M4.4c scale,
     made resumable and smoke-tested (development time, not counted above).
6. **The attacker budget** decided explicitly: protocol, 4,096,000 each (≈ 1.0 h), recommended; or 1,024,000
   (≈ 0.3 h).

Nothing was trained or launched during this audit. The pilot report's §7–§9 numbers will be corrected once you have
decided on these points.
