# M4.5c — Full multi-seed PPO training and short-horizon evaluation readiness

*2026-10-14 / 15. Branch `main` on 45b7dfa with the uncommitted horizon study; nothing staged or committed. **Approved
campaign only:** six policies, plus the full-budget seed-0 re-run and a limited health evaluation. The comprehensive
evaluation (strategic battery, full standard evaluation, best-response attackers) was **not** run; its framework is
prepared and gated (§G). Environment `GameConfig.m45_calibrated()`, unchanged; no rule, reward, observation, action or
PPO setting changed.

- Generated tables: [`results/m45c/full/report.md`](../../../results/m45c/full/report.md). Every number below that is not
  computed in the text comes from there.
- Raw results: `results/m45c/full/*.json(.gz)`; figures: `results/m45c/full/figures/`.
- Checkpoints (git-ignored): `results/rl/checkpoints/m45c/full/s{0..5}/` and `s0_repeat/`.
- Code (new): `m45c/full.py`, `health.py`, `health_report.py`, `battery.py`, `attackers.py`, `traces.py`.
- Tests: `tests/test_m45c_full.py` (new).*

**Transition convention** (as in the pilot report): budgets are **agent transitions**. One joint step gives 2 agent
transitions in 100% current-policy self-play. One update is 4,096 agent transitions.

---

## Summary

**What was run.**
- **Training.** Six independent policies on `m45_calibrated()`, seeds 0–5, from scratch: 8,192,000 agent transitions
  each, 49.2 M in all, with the learning rate 3e-4 annealed to 0 over each run and 100% current-policy self-play. They
  ran concurrently in 55 min.
- **Reproducibility.** A full-budget re-run of seed 0.
- **A limited health evaluation**, about 34 M simulated turns:
  - learning curves against the fixed suite;
  - final vs midpoint and final vs 6.144M (W = 1000);
  - finals vs their own past and adjacent checkpoints;
  - cross-play of the finals;
  - finals vs the other runs' midpoints;
  - behaviour.
- **Framework.** The revised 50-turn strategic battery, the attackers, viewer traces and the rest of the standard
  evaluation were built, tested and smoke-run. None of them was run at scale.

**Findings.**

| question | answer |
|---|---|
| did all six seeds train? | **yes**: 102 / 102 pipeline checks, 18 / 18 learning checks, 2,000 updates each, midpoint and final present |
| reproducible? | **yes, exactly**: the seed-0 full-budget re-run equals the original in its final weights, all 320 snapshots and every non-wall-clock history field |
| PPO diagnostics healthy? | **yes**, and nearly identical across seeds: KL ≤ 0.009 (falling to 0 with the annealed rate), entropy 2.86 → 1.74 → 1.81 (a small, learned late rise), explained variance 0.42–0.45, gradient norms ≤ 1.95, no alarm, 0 illegal actions |
| improved against the scripted M4.5b opponents? | **yes, in every seed**: scripted-suite mean −7.0 → +3.4 (finals +2.8 to +3.9). They beat all nine strategies and the OOD M4.4b network (−10.1 → +2.4), from both seats |
| finals better than earlier snapshots? | **yes**: every final beats or ties every earlier checkpoint of its run, and beats its own midpoint by +0.75 to +1.22 (W = 1000, every seed significant) |
| still improving from 4.096M to 8.192M? | **head to head, yes**: +0.96 ± 0.18 against the own midpoint, and +0.91 ± 0.10 against the *other* runs' midpoints. The gain is general, not lineage-specific. It shrinks over the last quarter (+0.45 against 6.14M). **Against the fixed suite, no**: flat since about 4M (−0.24 ± 0.58). **The pre-registered extension criterion is met** (5 of 6 runs not plateaued) |
| seeds consistent? | **yes**: cross-play within ±0.61 (2 of 15 pairs significant), the same state-conditioned profile in all six; they differ mainly in Ward use and freeze frequency |
| self-play cycling? | **no sign**: 1 significant regression in 66 adjacent comparisons (about chance), and none against any final |

**Behaviour (preliminary, descriptive).**
- **Shove:** 28–36% of the mass (31% in training, drifting slowly).
- **Freezes:** 2.6–4.4 per 1000 turns, about 5 turns long, won by the freezer in 60–80% of episodes. The freezer Thins
  and Trades a walk-up; the frozen-on side Shoves (39–51%) or resets.
- **Selective Trading:** Trade at high HP (27–31%), Safe at low HP (76–82%), Safe under the enemy wave on the enemy half
  (38–44%).
- **Zoning, from an initialized ×1.76 gap:** the weaker side concedes (Yield, Safe under threat, stays at its tower).
  The stronger side stands forward but denies almost nothing.
- **Ward:** rare, 1–2% of the mass and 8–22 per 1000 turns depending on the seed.
- **Recall:** by HP (23–31% below 35% HP), mostly buying resets at about 61% HP; 34–40% of attempts are interrupted.
- **Early game:** less Shove and less Safe than later.

**Readiness.**
- The 50-turn methodology (ΔQ₅₀ primary, ΔG₅₀ beside it, ledgers at 1–20 turns) is implemented on the horizon study's
  engine and tested. So are its three exceptions and the per-family 300-turn validation subset with its pre-registered
  pass rule.
- The battery adds early-game families, power-gap constructed states, opponent protocols, midpoint replication and
  exact one-turn mechanics.
- The comprehensive evaluation is estimated at **≈ 5.7 h (4.7–7.0)**.

**Recommendation: GO WITH CAVEATS** (§I). The caveat that matters is that training has not converged head to head.
The pre-registered criterion calls for proposing a matched 2× extension, and I recommend approving it **before** the
strategic battery. Evaluating the 8.192M finals now is the defensible alternative. Nothing beyond the approved scope
was started.

---

## A. Training completion

| run | agent transitions | joint steps | PPO updates | snapshots | status | wall clock | transitions/s (per process) | peak memory | final digest |
|---|---|---|---|---|---|---|---|---|---|
| s0 | 8,192,000 | 4,096,000 | 2,000 | 320 | complete | 54.8 min | 2,494 | 0.64 GB | `9ad2c141…` |
| s1 | 8,192,000 | 4,096,000 | 2,000 | 320 | complete | 54.9 min | 2,493 | 0.64 GB | `329364ed…` |
| s2 | 8,192,000 | 4,096,000 | 2,000 | 320 | complete | 55.1 min | 2,484 | 0.64 GB | `f471dfbc…` |
| s3 | 8,192,000 | 4,096,000 | 2,000 | 320 | complete | 55.0 min | 2,486 | 0.64 GB | `644343ca…` |
| s4 | 8,192,000 | 4,096,000 | 2,000 | 320 | complete | 55.1 min | 2,482 | 0.64 GB | `83450c9e…` |
| s5 | 8,192,000 | 4,096,000 | 2,000 | 320 | complete | 55.0 min | 2,489 | 0.64 GB | `8425bda9…` |
| s0_repeat | 8,192,000 | 4,096,000 | 2,000 | 320 | complete | 45.3 min | 3,020 (shared the machine, then alone) | 0.65 GB | `9ad2c141…` (= s0) |

- **Totals.** 49,152,000 primary agent transitions (24,576,000 joint steps, 12,000 updates). 8,000 games finished per
  run, each truncated at 512 turns and bootstrapped, none terminated.
- **Concurrency.** The six runs ran concurrently in one call: 3,307 s (55.1 min), 22:57:45 → 23:52:52.
- **Checkpoints** (`results/rl/checkpoints/m45c/full/s<seed>/`, 319 MB in all, git-ignored):
  - `final.pt`, `history.jsonl`, `run.json` and the worker-written `summary.json`;
  - `snapshots/snap_<transitions>.pt` every 25,600 transitions;
  - the **midpoint is `snapshots/snap_004096000.pt`**, present in every run;
  - the last snapshot is bit-identical to `final.pt` (checked).
- **Checks.** All 102 pipeline checks of the six runs pass, and so do the 18 learning checks:
  - finite statistics, |V| ≤ Ω, gradient norms, KL;
  - equal seats, 2 transitions per joint step, the exact budget, 2,000 updates;
  - no illegal sample and no never-legal offer;
  - exact reward antisymmetry, V_A(s0) = V_B(s0);
  - midpoint and final present, lane cache 5,000.
- **Learning checks.** The symmetric start is valued between −0.27 and +0.04 resources; its true value is 0.
- The record is `results/m45c/full/full_train.json`, rebuilt by `train_check` from the run directories alone.

**Execution decisions.**
- **Memory.** When training was due to start, the machine had 2.25 GiB free of 15.9 GiB. Firefox, VS Code, Discord
  and a game were open.
  - Six processes at the approved `lane_cache = 5,000` peaked at 0.64 GB each (3.7 GB in all). Free memory fell to
    0.28 GiB during the cache fill and then held at about 1.0 GiB (median). Paging stayed small: swap grew by at most
    0.24 GiB.
  - A seventh process would have added another 0.64 GB on top of that. The resource measurements did **not** support
    running the re-run safely as a 7th process, so it was scheduled to run **alone after the evaluation**.
- **Throughput.** 2,490 transitions/s per process, against the runtime audit's 3,200 measured on a quieter machine.
  - The other applications took CPU: about 79% of the machine was busy during training.
  - So the six runs took 55 min instead of the central 45 min, at the top of the audit's 43–55 min range.
  - Per-update throughput rose to about 2,950 in the last tenth, when that load dropped.
- **Nothing ran alongside the six primary runs.** No evaluation, test or regression ran concurrently with them. The
  only commands were read-only status checks and code edits, so their throughput record is clean.
- **Order after training.** An automatic chain started when all six runs reported complete: check, health evaluation,
  behaviour, smoke runs, tests, re-run, each step waiting for the previous one.
- **One approved exception.** During the evaluation, applications were closed (4.25 GiB free, the game closed), and
  **with your explicit approval** the seed-0 re-run was started early, at 00:28. It overlapped the last health parts,
  behaviour, the smoke runs, the tests and the cross-lineage check, instead of waiting for them.
  - The exception covers the re-run only. Bit-identity does not depend on machine load (§B), and the evaluation
    results do not either: they use fixed seeds.
  - Only speed changed: the re-run ran at 1,400–2,400 transitions/s while it shared the machine, and about 4,100 alone.
  - The chain's own later attempt to start the re-run was refused by the never-overwrite rule, as designed.

## B. Reproducibility

The M4.4b protocol: one complete, independent re-training of seed 0 at the full budget, in a fresh directory
(`s0_repeat`), compared with the original by `train_check` (`full.reproducibility`):

| compared | result |
|---|---|
| initial weights (`torch.manual_seed(0)`) | identical |
| final weights | **identical**: digest `9ad2c141943f2284…` in both |
| final checkpoint digest | identical |
| snapshots | **all 320 identical** (every 25,600 transitions, the midpoint included); none differs |
| training history | **identical in every non-wall-clock field** of all 2,000 entries: losses, KL, entropy, explained variance, gradient norms, value statistics, learning rate, action and mask diagnostics, per-seat rewards and advantages, episode outcomes. Only `elapsed`, `sps`, `collect_s` and `update_s` differ |
| first differing history entry | none |

- **The match is exact, not near.** The re-run ran under different conditions from the original: started 36 minutes
  after the original finished, sharing the machine with the evaluation for part of its run and alone for the rest, at
  1,400–4,200 transitions/s against the original's steady 2,490.
- **Training is therefore a deterministic function of the seed and the configuration.** Concurrency, machine load and
  memory pressure do not affect it.
- **Every checkpoint of the campaign can be regenerated bit for bit.** All 20 checks of the re-run pass; with the six
  primaries, 122 of 122.

## C. Training stability

Seed means by tenth of the run, with the range over seeds at the start, middle and end (`results/m45c/full/report.md`
§2; figure `figures/training_diagnostics.png`):

| quantity | first tenth | middle (t5) | last tenth | reading |
|---|---|---|---|---|
| policy entropy (nats) | 2.03 (initial 2.86) | 1.74 | 1.81 [1.78–1.88] | falls to about 1.74 by 3M, then rises slowly over the last 40% (+0.07), in every seed |
| approximate KL | 0.0056 | 0.0044 | 0.0004 | falls linearly with the annealed learning rate; maximum of any update 0.0087; maximum 10-update mean 0.0066 (alarm 0.05) |
| clip fraction | 0.068 | 0.048 | 0.001 | likewise follows the learning rate |
| explained variance | 0.50 | 0.42 | 0.45 [0.43–0.48] | stable from 1M at 0.42–0.45; the pilot's level, below M4.4b's 0.5 (M4.5b caveat 4: trade values are noisy) |
| gradient norm, pre-clip | 0.55 | 0.70 | 0.76 | slow rise, so most minibatches are norm-clipped at 0.5, as in M4.4b and the pilot; maximum 1.95 (alarm 100) |
| value loss (scaled) | 0.0064 | 0.0065 | 0.0060 | flat |
| critic | \|V\| ≤ 18.9 resources ≤ Ω; V(s0) between −0.27 and +0.04 | | | bounded |

- **Alarms.** No pre-registered stop condition fired in any run: no non-finite value, no sustained KL, no gradient
  alarm, entropy never below 1.55 (floor 0.3).
- **Action legality.** 0 illegal samples and 0 never-legal offers in 49 M transitions. Rewards are exactly
  antisymmetric in every rollout.
- **Seeds.** They agree closely on every diagnostic (figure): the largest seed-to-seed spread is in explained
  variance and value loss, which are noisy update to update.
- **The late entropy rise is learned, not numerical.**
  - The entropy bonus and the policy gradient scale with the same learning rate, so annealing does not change their
    balance.
  - It coincides with a shift toward defensive behaviours: Safe rises from 17% to 20% of lane behaviours and Hold
    falls, while deaths keep falling.
  - It is small, smooth and present in every seed. It is noted here, not treated as instability.

**Self-play episode statistics** of the training rollouts (seed means, first → last tenth):

| quantity | first tenth | last tenth | trend |
|---|---|---|---|
| deaths per 1000 champion-turns | 10.1 | 6.2 | falling throughout, still falling in the last tenth (6.5 → 6.2) |
| completed Recalls per 1000 champion-turns | 38.3 | 43.4 | rising |
| Recall attempts interrupted | 31% | 39% (37–41% by seed) | as in the pilot (34–42%) |
| trade attempts per 1000 champion-turns | 215 | 217 | flat; connect rate 0.65 → 0.63 |
| farm collection | 0.744 | 0.763 | flat after 1M |
| plates per 1000 turns | 11.6 | 11.7 | flat |
| sampled Shove share | 25% | 31% (27–33% by seed) | 29% by 1.6M, then a slow drift to 31% |
| sampled Ward share | 0.6% | 1.3% (0.7–2.0% by seed) | rare; doubles over training |
| sampled Recall share | 5.8% | 7.2% | rising |

## D. Learning progress

### D.1 Against the fixed suite

The design (`results/m45c/full/report.md` §3):
- the untrained network and 12 checkpoints, all on the 409,600 grid of the planned 21-point curve;
- opponents: the nine M4.5b scripted strategies and the out-of-distribution M4.4b aggro network;
- 32 paired seeds, both seats, W = 460;
- the same evaluation seeds for every checkpoint and every training seed.

Values are seed means ± t interval (6 seeds):

| checkpoint | scripted-suite mean | seed range | M4.4b aggro (OOD) |
|---|---|---|---|
| untrained | −7.02 ± 0.04 | −7.08 to −6.98 | −10.09 |
| 0.41M | +1.57 ± 0.30 | +1.20 to +1.99 | −1.56 |
| 0.82M | +2.31 ± 0.56 | +1.55 to +3.05 | −0.45 |
| 1.64M | +2.72 ± 0.28 | +2.45 to +3.10 | +0.81 |
| 2.46M | +2.75 ± 0.54 | +2.25 to +3.44 | +1.33 |
| 3.28M | +3.26 ± 0.48 | +2.93 to +4.12 | +1.71 |
| **4.10M (midpoint)** | **+3.64 ± 0.32** | +3.36 to +4.16 | **+2.27** |
| 4.92M | +3.36 ± 0.43 | +2.83 to +3.87 | +2.18 |
| 5.73M | +3.20 ± 0.41 | +2.74 to +3.71 | +2.03 |
| 6.14M | +3.26 ± 0.32 | +2.80 to +3.68 | +2.60 |
| 6.55M | +3.23 ± 0.32 | +2.96 to +3.76 | +2.24 |
| 7.37M | +3.35 ± 0.52 | +2.87 to +4.30 | +2.27 |
| **8.19M (final)** | **+3.40 ± 0.41** | +2.80 to +3.92 | **+2.41** |

- **Every seed improved against the representative scripted opponents.** Untrained to midpoint is +10.7 ± 0.4
  resources; against the M4.4b network it is +12.4.
- **The finals beat every one of the ten opponents from both seats.** They beat the M4.4b network by +2.41; the pilot
  at 1.02M managed +0.33.
- **Two things changed relative to the pilot.**
  - Strength against the fixed opponents rose until about 4M, where the pilot had not plateaued at 1M.
  - The second half of the schedule then left it **flat**:
    - final − midpoint is −0.24 ± 0.58 on the scripted mean and +0.15 ± 0.32 against the M4.4b network;
    - by opponent, the finals gained against the aggressive trader (`g_bully` +1.00 ± 0.39) and lost a little
      against the Ward-oriented Shover (`g_shove_ward` −0.84 ± 0.45);
    - against the other seven, final − midpoint is within ±0.65 with intervals covering 0.

### D.2 Against their own past (the pre-registered second-half test)

Head-to-head at W = 1000 with 64 paired seeds (§4 of the tables), and the final against every curve checkpoint at
W = 460 (§5):

| final vs | s0 | s1 | s2 | s3 | s4 | s5 | seed mean |
|---|---|---|---|---|---|---|---|
| untrained | +11.00 | +10.63 | +11.05 | +10.99 | +11.28 | +10.29 | |
| 1.64M | +1.75 | +1.58 | +2.18 | +1.47 | +1.78 | +1.38 | |
| 3.28M | +1.22 | +1.22 | +0.71 | +1.11 | +0.67 | +0.70 | |
| **midpoint 4.10M (W = 1000)** | **+1.07 ± 0.41** | **+0.81 ± 0.41** | **+0.75 ± 0.44** | **+0.92 ± 0.48** | **+1.22 ± 0.44** | **+0.98 ± 0.36** | **+0.96 ± 0.18** |
| 4.92M | +0.46 | +0.24 | +0.29 | +0.17 | +0.84 | +0.41 | |
| 5.73M | +0.52 | −0.32 | −0.42 | +0.57 | +0.38 | +0.50 | |
| **6.14M (W = 1000)** | **+0.31 ± 0.43** | **+0.80 ± 0.39** | **+0.49 ± 0.51** | **+0.28 ± 0.43** | **+0.45 ± 0.40** | **+0.36 ± 0.43** | **+0.45 ± 0.20** |
| 6.55M | +0.48 | +0.54 | −0.18 | +0.51 | +0.52 | +0.64 | |
| 7.37M | −0.06 | −0.12 | −0.11 | +0.70 | +0.02 | +0.35 | |

- **The final beats its own midpoint head to head in every seed.** The gain is +0.75 to +1.22, every interval excludes
  0, seat A and seat B agree, and the seed mean is +0.96 ± 0.18. Against 6.14M the gain is +0.45 ± 0.20, significant
  in seeds 1 and 4.
- **The gain shrinks toward the end of the schedule, as an annealed learning rate predicts.** Against 7.37M it is
  about +0.13 on average and significant only for seed 3.
- **No final loses significantly to any earlier checkpoint** (60 comparisons).

**Pre-registered plateau and extension rules** (pilot report §8.4, evaluated mechanically, §4 of the tables):

| run | (a) curve: final − 6.144M, \|Δ\| < 0.3 | (b) final vs 6.144M within ±0.3, interval covering 0 | plateaued |
|---|---|---|---|
| s0 | −0.08 ✓ | +0.31 ± 0.43 ✗ (just above 0.3) | no |
| s1 | −0.35 ✗ | +0.80 ± 0.39 ✗ | no |
| s2 | +0.20 ✓ | +0.49 ± 0.51 ✗ | no |
| s3 | +0.24 ✓ | +0.28 ± 0.43 ✓ | **yes** |
| s4 | −0.03 ✓ | +0.45 ± 0.40 ✗ (interval excludes 0) | no |
| s5 | +0.84 ✗ | +0.36 ± 0.43 ✗ | no |

**The extension criterion is met.**
- 5 of 6 runs are not plateaued (at least 3 required).
- Final − midpoint is +0.96 ± 0.18 (more than +0.5, with an interval excluding 0, required).

By the pre-registered rule this triggers a *proposal* of a matched 2× extension (16,384,000 per seed). That needs its
own approval and was **not** started (§I).

**What the second-half gain is, and what it is not.**
- **It is a gain against strong learned play, not against the fixed opponents.** The same finals are flat against
  the scripted suite and the M4.4b network (D.1). In self-play this pattern means one of two things:
  - the policy keeps improving in ways the scripted opponents do not exercise;
  - or it adapts to its own lineage.

  §D.3 tests which.
- **It mixes more data with annealing.** The midpoint is a checkpoint of the 8.192M schedule at 50% of the learning
  rate, **not** a separately trained 4.096M policy, which would have annealed to 0. "Final − midpoint" therefore
  combines 4.1M more transitions with the learning rate falling from 1.5e-4 to 0. A policy taken mid-schedule still
  carries the update noise of a high learning rate; part of the head-to-head gain may come from annealing itself.
  M4.4b's matched extension (both runs annealed) gained +0.6 to +1.5 at 2× the budget.

### D.3 Does the gain transfer to other lineages? (added after D.2, not pre-registered)

**The test.** If the second-half gain were adaptation to the run's own lineage, it would shrink against the *other*
runs' midpoints. If it is general strength, it would not.

**Design.** Each final against the five other midpoints, W = 1000, 64 paired seeds (30 matchups, 3.8 M turns, seed
93,450; `health_eval --health-parts cross_mid`). The transfer gain of final *i* is
mean over *j* ≠ *i* of [v(final_i, midpoint_j) − v(final_i, final_j)]: lineage *j*'s second-half gain, measured by an
independent policy.

| final | vs own midpoint | vs the other midpoints (mean) | transfer gain | that lineage's gain, measured by the other five finals |
|---|---|---|---|---|
| s0 | +1.07 | +1.05 | +1.04 | +0.86 |
| s1 | +0.81 | +1.11 | +0.87 | +1.12 |
| s2 | +0.75 | +0.74 | +0.77 | +0.94 |
| s3 | +0.92 | +0.97 | +0.88 | +0.82 |
| s4 | +1.22 | +0.74 | +0.88 | +0.91 |
| s5 | +0.98 | +0.83 | +1.00 | +0.79 |
| **seed mean** | **+0.96 ± 0.18** | **+0.91 ± 0.17** | **+0.91 ± 0.10** | |

- **The gain transfers in full.** 29 of the 30 final-vs-other-midpoint pairs are significant, at +0.33 to +1.50. The
  second half produced policies that are stronger against *any* run's midpoint by about as much as against their own.
- **It is learning progress, not lineage-specific adaptation and not cycling.** (Annealing may still contribute;
  §D.2.)
- **The fixed suite stopped measuring it after about 4M.** The finals beat it by about +3.4 everywhere, and the gain
  is in play that those opponents do not exercise. The pilot report anticipated this: the scripted suite "can only show
  the policy's margin, not its weaknesses".

### D.4 Self-play cycling

- **Adjacent checkpoints** (each curve checkpoint against its predecessor, W = 460, 32 paired seeds; §5 of the
  tables): 66 comparisons over the six runs. Later checkpoints win or tie, with large gains until about 2.5M and small,
  noisy ones after.
- **One significant regression:** seed 5, 6.55M against 6.14M, −0.48 ± 0.45. That is about what chance gives in 66
  one-sided tests at 2.5% (1.6 expected).
- **The pre-registered stop criterion is not met:** it needs a significant regression on at least 2 seeds.
- **No preliminary sign of cycling.** Every final also beats or ties every earlier checkpoint of its run (D.2).

## E. Preliminary behaviour

These are descriptive readouts: self-play records of 16 games × 1000 turns per checkpoint (both seats focal), 64
early-game games of 120 turns, the final against two fixed opponents, the M4.5b ledger, and an initialized power gap
(§7 of the tables). They are not strategic conclusions; those are what the battery is for.

**Action distribution of the finals** (mean probability mass, self-play; the midpoint in brackets):

| | Last Hit | Thin | Slow Push | Shove | Ward | Recall | Hold | Yield | Trade | Safe | entropy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| range over the six finals | 0.18–0.20 | 0.17–0.23 | 0.19–0.21 | **0.28–0.36** | 0.009–0.023 | 0.068–0.085 | 0.24–0.28 | 0.23–0.25 | 0.23–0.26 | 0.24–0.28 | 1.78–1.89 |
| range over the six midpoints | 0.18–0.25 | 0.17–0.23 | 0.20–0.25 | 0.24–0.32 | 0.005–0.025 | 0.074–0.089 | 0.26–0.31 | 0.23–0.28 | 0.23–0.32 | 0.18–0.23 | 1.65–1.81 |

(Hold, Yield, Trade and Safe are shares of the lane mass.)

**State-conditioned behaviour of the finals** (the six finals' range, then decisions per final):

| question | state | what the finals do |
|---|---|---|
| **Shove** | all decisions | 28–36% of the mass (midpoints 24–32%); higher when the enemy is away (43–49%), when frozen on (39–51%), and at levels 6–10 than 1–5 (31–43% vs 25–35%) |
| **freeze phase** | holding a freeze (1,560–1,905) | Thin 14–26%, Trade 18–26%: the freezer keeps the wave and fights a walk-up |
| | frozen on (1,508–1,805) | Shove 39–51%, Recall 10–13%: break or reset |
| | freezes (≥ 4 held turns) | 2.6–4.4 per 1000 turns, about 5 turns long; the freezer out-earns the frozen-on side in **60–80%** of them by +0.7 to +1.4 resources (midpoints: 4.1–9.2 per 1000, 61–76%): fewer freezes than at the midpoint, at least as successful |
| **threat-based zoning** | natural power gaps | essentially absent: 9–59 "stronger" decisions (and as many "weaker") of about 30,000 per final |
| | initialized ×1.76 gap (self-play from the neutral start) | the **weaker** side concedes. It Yields 32–40% of the time (the stronger side 25–30%), Trades less (0.14–0.20 vs 0.17–0.29 attempts per turn), Shoves more (34–45%), stands at its tower 35–39% of the time (the stronger side 12–18%), and dies 3–8× as often |
| | weaker side under a credible threat (M4.4 definition) vs not threatened | Safe 14–24% vs 7–15%, Trade 10–16% vs 18–27%, Recall 2–4% vs 7–16% |
| | the stronger side's pressure | it stands further forward (mean position +1.0 to +2.0) but the weaker side's uncollected gold rises by at most 0.015 per turn when the stronger side is near. **Concession without active denial**, as in M4.4b |
| **selective Trading** | HP ≥ 70% vs HP < 35% | Trade 27–31% vs Safe 76–82% |
| | waves meeting on the own half vs the enemy half | Trade 16–21% on the own half; Safe 38–44% on the enemy half (under the enemy wave's fire) |
| | connect rate (training) | 0.63; in the power-gap games Trading costs the trader farm (collection 0.67–0.71 while trading vs 0.83–0.87 while holding) |
| **Ward** | Ward legal, own vision down (16,000–18,000) | Ward mass 1–4%; 8–22 Wards per 1000 turns (midpoints 5–24), 5–8% interrupted; it varies most by seed of any behaviour (s4 22 per 1000, s5 8) |
| **Recall** | HP < 35% vs HP ≥ 70% | Recall 23–31% vs 1–2% |
| | completed Recalls | 41–46 per 1000 turns: 13–15 at low HP, 26–32 at about 61% HP with about 7.5 unspent gold (a reset that buys in 99% of cases) |
| | interrupted | 34–40% of attempts (ledger; the pilot 34–42%, M4.4b 22–38%) |
| **early game** | levels 1–5 / 6–10 / 11–18 (early games) | Shove 23–35% / 32–42% / 36–43%; Safe 7–11% / 13–18% / 19–22%; Trade 15–26% throughout. Early play Shoves less and plays less Safe |
| **siege** | own crash under the enemy tower, Pressure legal | Pressure 54–61% |
| **minions** | ledger | minion fire is 4.6–6.2% of the damage a champion takes |

**Against fixed opponents.**
- **Against `g_bully`** (Trades whenever the enemy is catchable), the finals are frozen on 24–27% of the time,
  Recall 76–86 times per 1000 turns and Pressure 73–117 per 1000. They lead by +7.4 to +9.5 per 100 turns.
- **Against `fz_guard_sh`** (freezer), the finals almost never die (0.2–1.2 per 1000 turns) and lead by +4.3 to +5.1
  per 100 turns.

## F. Cross-seed consistency

| | finding |
|---|---|
| training diagnostics | nearly indistinguishable (figure); final entropy 1.78–1.90 |
| strength against the suite | finals +2.80 to +3.92 on the scripted mean, +2.10 to +2.71 against the M4.4b network |
| cross-play of the finals (W = 1000, 64 paired seeds) | largest \|value\| 0.61; 2 of 15 pairs significant (s1 over s5 +0.61 ± 0.48, s3 over s4 +0.61 ± 0.43); row means −0.18 to +0.25. Far from the pre-registered "seeds disagree" gap of 2 |
| second-half gain | +0.75 to +1.22 against the own midpoint, every seed |
| behaviour | the same state-conditioned profile in every seed |
| where seeds differ | Ward use (8–22 per 1000 turns), Shove share (28–36%), freeze frequency (2.6–4.4 per 1000) |

**The six seeds learned the same kind of policy.** No seed is an outlier in strength, and none forms a separate group
in cross-play. The differences are quantitative and concentrated in the rarest behaviours, Ward and freezes. They
should be read seed by seed in the strategic evaluation, not averaged away.

---

## G. Short-horizon evaluation readiness

The approved methodology (horizon study §6) is implemented, along with every other part of the comprehensive
evaluation. Each part refuses to run without its own confirmation flag, and each has been exercised end to end on a
trained final at tiny sizes (smoke) or by tests. Nothing in this section was run at scale.

| part | module / stage | status | gate |
|---|---|---|---|
| **strategic battery**: paired CRN rollouts, multi-checkpoint G_h / Q_h with the evaluated policy's own critic, M4.5b ledger, one-turn and sustained candidates reported separately, resumable per-job files | `battery.py`, stages `battery_plan`, `battery_smoke`, `battery` | implemented, tested, smoke-run on `final:full:0` (§G.1) | `--confirm-m45c-battery` |
| families A–E on natural states, plus **early-game** A and C (focal level ≤ 10, from 60-turn games) | `battery.FAMILIES` | implemented | |
| constructed families with **power gaps and early levels**: freeze grid F (×1.76 / equal × held 12 / 20 × level 10 / 15, plus ×1.76 at level 6); trade bench G (level 4 / 11 / 18 × front 8 / 24 × equal / ×1.76); zoning H (3 situations × A stronger / equal / weaker at level 10, plus stronger at level 6); Ward I (4 situations × level 10 / 6) | `battery.GRIDS` | implemented | |
| opponent protocols: policy, modal; breaker / back-off / reset (freeze); three freezer styles (frozen-on); wary (trade, zoning); chaser = `g_bully` (Ward) | `battery.FAMILIES` | implemented | |
| horizons: 50 by default; exception 1 (the 20-turn freeze plans run 60 turns); exception 2 (G₅₀ primary on constructed states); exception 3 (a family that fails validation is re-run at 150) | `family_horizon`, `primary_statistic`, `jobs(extended=...)` | implemented, tested | |
| **300-turn validation subset**: fixed-seed max(8, 15%) of each natural family + every constructed cell, protocol "policy"; pass rule \|bias\| ≤ max(0.15, 2 SE), slope in [0.8, 1.25], no contradiction | `validation_indices`, `validate` | implemented, tested (pass and fail cases) | |
| midpoint replication (B freeze, B frozen, E Recall, F; policy and modal) | `battery.REPLICATION`, `run` | implemented | |
| **mechanistic diagnostics**: the exact one-turn expectation (M4.5b enumeration, chance branches including ganks) of every candidate's first action against the opponent's modal reply, at every state | `battery.mechanics` | implemented, tested, in the smoke | runs with the battery |
| battery report: per family, protocol and candidate; pooled over the six finals with every final shown; validation verdicts | `battery.report` | implemented, tested | |
| **standard evaluation, rest**: the 8 missing points of the 21-point curve (same seeds as the health curve), the W = 1000 headline of final and midpoint against the suite, a 9-checkpoint round robin per run | `health.py` parts `curve_rest`, `headline`, `tournament` | implemented, tested (job design) | `--confirm-m45c-comprehensive` |
| **best-response attackers**: one per final, warm-started, the target's PPO settings, 4,096,000 learner transitions (half the target's budget), evaluated at W = 460 / 1000 and against the other finals | `attackers.py`, stages `attackers_smoke`, `attackers` | implemented, smoke-run (2 updates) | `--confirm-m45c-attackers` |
| **viewer trajectories**: M4.4b's state-defined search (freezes, Slow Push landing, Ward then Pressure, busy dives, the threatened weak side) with M4.5 labels and the `m45_calibrated` stamp, every export verified by replay in the viewer | `traces.py`, stages `traces_smoke`, `traces` | implemented, tested, smoke-run | `--confirm-m45c-comprehensive` |

### G.1 What the smoke runs and tests showed

| check | result |
|---|---|
| `battery_smoke` on `final:full:0`: every family, 2 natural states (or the first constructed cell), K = 2, every protocol, the validation subset forced to one state | 64 jobs, 63,460 simulated turns, 29 s on 11 workers. Natural states were found for all 8 natural families, including both early-game families, within 2 self-play games each. The mechanics stage, the validation statistics and the report ran end to end. *The verdicts of a 1-state, K = 2 smoke are meaningless by design; it tests the plumbing, not the method* |
| one-turn exact mechanics (smoke, `final:full:0`) | example readouts: Shove at a wave contact is worth +1.22 of immediate lead over the policy's modal action; keeping or guarding a freeze costs 0.45–0.56 on its first turn; Trading into a bigger enemy wave costs 0.47. These are the "immediate mechanical consequence" layer, the costs that arrive before any delayed payoff |
| `traces_smoke` | the state-defined search found freeze-denies, freeze-broken, Slow-Push-lands and weak-side-threatened games in 3 short games, exported them with M4.5 labels, and **every export replayed exactly** in the viewer's session code (actions and payoff) |
| `attackers_smoke` | 2 updates of a warm-started attacker against `final:full:0` trained and evaluated (982 learner transitions/s alone, on a busy machine) |
| tests | `tests/test_m45c_full.py` (17 new) plus the existing M4.5c training (16) and horizon (4) tests: **37 passed**. They cover the campaign gate and the never-overwrite rule, failure records, the pre-registered alarms, the health design (reserved seeds, checkpoints on the schedule, one seed per part), the plateau / extension rules, the cross-play matrix, the battery's exceptions, validation subset, job design, validation rule (pass, bias and contradiction failures), report pooling, gates, a real paired job and the exact mechanics on the pilot policy, and an exact viewer replay of exported traces. **Full regression suite: 1394 passed, 4 skipped, 2 xfailed** (8 min, run after all training) |

### G.2 Remaining before the comprehensive evaluation

Nothing blocks it. Three things are worth deciding or doing first:

1. **Whether to extend training first** (§D.2, §I). The battery should run on the policies the study will report.
2. **A battery-report figure set and per-seed tables** for the full run. The tables pool the finals with every final
   shown; figures (horizon curves per family) are not written yet. About 0.5 h of development.
3. **The interrupted-Recall follow-up** (M4.4b `recall_followup`). Not ported, because the battery's E family and the
   ledger now measure the same thing (34–40% interrupted). It can be added cheaply if wanted.

---

## H. Updated evaluation estimate

Rates were **measured in this execution**, on this machine as it was being used:
- evaluation: network vs scripted opponents **1,097** simulated turns per worker-second, network vs network **734**;
- training: **2,488** transitions/s per process with 6 concurrent;
- battery: the horizon study's at-scale **550** turns per worker-second (the tiny smoke ran at 747, which sets the
  optimistic end of the range).

All on 11 workers (`results/m45c/full/report.md` §9; `health_report.estimate`):

| component | content | hours |
|---|---|---|
| standard evaluation, rest | the 8 missing curve points (14.1 M turns), the W = 1000 headline of final and midpoint against the suite (15.4 M), the 9-checkpoint tournament per run (3.2 M), behaviour of 5 checkpoints, integrity checks, traces and viewer recordings. The health evaluation's parts are **reused**, not repeated: 33.8 M turns of curve, head-to-head, history, adjacent checkpoints, cross-play and cross-lineage, already on the comprehensive design's seeds | **1.0** |
| strategic battery | 47.6 M simulated turns = 6 finals × 6.54 M + 6 midpoints × 1.39 M (replication). It includes the 300-turn validation subset (39% of the turns: every constructed cell is validated), the two early-game families and the power-gap and level-6 constructed cells | **2.2** (1.6–2.7) |
| best-response attackers | 6 × 4,096,000 learner transitions (half the target's budget), 6 concurrent, at 0.8 × the measured self-play joint-step rate (M4.3b's ratio), plus evaluation against the target and transfer | **1.5** (1.1 on a quiet machine) |
| tests, regression, reports | | 0.3 |
| margin (15%) | | 0.7 |
| **total** | | **≈ 5.7 h (4.7–7.0)** |

**Against the plan.**
- The approved plan's ≈ 6.2 h has become ≈ 5.7 h, because training is done and the health evaluation's 33.8 M turns
  are reusable.
- The battery is **larger** than the horizon study's 36 M turns. The brief asked for early-game and power-gap
  scenarios: two early-game natural families, power-gap and level-6 constructed cells, and a weaker-side zoning cell.
  Opponent diversity, state coverage and replication were not reduced to compensate.

**If the extension is approved first** (§I): six fresh 16,384,000-transition runs would take ≈ 1.8 h at this
execution's throughput (≈ 1.4 h on a quiet machine). A health check of the new finals on the same design would take
≈ 1 h. The comprehensive evaluation would then run on the 16.4M finals at about the same 5.7 h, since its cost does
not depend on the training budget.

---

## I. Recommendation

### **GO WITH CAVEATS**

Training succeeded, and the policies and the evaluation framework are ready for the comprehensive evaluation.

**What supports GO:**
- All six seeds trained cleanly. All 102 pipeline checks and the 18 learning checks pass, no pre-registered alarm
  fired, and there were 0 illegal actions in 49 M transitions. Training is bit-for-bit reproducible (§B).
- **They learned.** The scripted-suite mean went from −7.0 to +3.4 resources and the M4.4b network from −10.1 to +2.4.
  The finals beat every earlier checkpoint of their run, and every other run's midpoint.
- **The seeds are consistent:** cross-play within ±0.61, the same behavioural profile in all six.
- **No sign of self-play cycling.**
- **Behaviour is state-conditioned in every seed:** Recall and Safe by HP, freeze keeping and breaking, Pressure,
  concession under threat. Freezes deny in 60–80% of episodes.
- **The 50-turn methodology is implemented and smoke-tested.** That includes the validation safeguards, the early-game
  and power-gap scenarios, opponent protocols, replication and mechanistic diagnostics. So are the attackers, the
  traces and the rest of the standard evaluation, each behind its own gate.

**The caveats:**
1. **Training has not converged head to head.**
   - The pre-registered extension criterion is **met**: 5 of 6 runs did not plateau, and the final beats its midpoint
     by +0.96 ± 0.18.
   - The gain is **genuine and general**: +0.91 ± 0.10 against independent lineages (§D.3).
   - Over the last quarter it is small (+0.45 against 6.14M, about +0.13 against 7.37M), as an annealed schedule
     makes it.
   - Against the fixed suite, strength has been flat since about 4M.
2. **The midpoint comparison mixes more data with annealing** (§D.2). How much of the +0.96 is annealing, and how much
   is the extra 4.1M transitions, cannot be separated with these runs.
3. **Natural power gaps are still almost absent** (about 0.1–0.2% of decisions), and levels 1–10 are about 10% of
   experience. Conclusions about zoning and the early game rest on the initialized scenarios, which the battery has.
4. **Rare behaviours vary by seed.** Ward use spans 8–22 per 1000 turns and freezes 2.6–4.4 per 1000. Report them per
   seed.
5. **About 34–40% of Recall attempts are interrupted**, as in the pilot. The battery's E family tests whether that is
   rational (M4.4c found it was a guessing game).

**Is more training justified before the strategic evaluation? Yes. I recommend approving the pre-registered matched
2× extension first** (six fresh runs of 16,384,000 transitions, same settings, the M4.4b protocol):
- **The criterion was set in advance for exactly this case.** It is met, and the cross-lineage test rules out the
  benign explanation, lineage-specific adaptation.
- **The battery measures what the policies do.** It is the most expensive and most interpretive part of the study
  (≈ 2.2 h of a 5.7 h evaluation). Running it on policies still gaining about 1 resource per 4M transitions against
  learned play would spend that on an intermediate stage.
- **The extension would answer what the battery cannot.** M4.4b's 2× extension left the qualitative profile unchanged
  but changed exactly the details this study cares about: freeze denial 50% → 61–64%, and more Vision. The extension
  shows whether Shove (31%), freeze success (60–80%) and Ward use (1–2%) are stable at 2× the budget.
- **It is affordable.** ≈ 1.8 h of training plus ≈ 1 h of health check, against re-doing 5.7 h of evaluation later if
  the extension were run after it.

**The alternative is defensible too.** Run the comprehensive evaluation on the 8.192M finals now, with caveat 1 stated
in every conclusion. M4.4b found its strategic profile stable under a 2× extension, so the qualitative answers may
well not change.

**Either way, nothing further was started.** No extension, battery, standard evaluation or attackers.

---

## Reproduction

```bash
python -m lanegame.experiments.rl.m45c train_preflight                                   # 7 updates, seconds
python -m lanegame.experiments.rl.m45c train_full --confirm-m45c-full --repeat separate  # six runs, 55 min here
python -m lanegame.experiments.rl.m45c train_full --confirm-m45c-full --repeat only      # seed-0 re-run
python -m lanegame.experiments.rl.m45c train_check health_training                        # records, alarms
python -m lanegame.experiments.rl.m45c health_eval health_behavior                        # about 50 min on 11 workers
python -m lanegame.experiments.rl.m45c health_eval --health-parts cross_mid               # about 7 min
python -m lanegame.experiments.rl.m45c battery_smoke traces_smoke attackers_smoke         # about 1 min
python -m lanegame.experiments.rl.m45c health_report                                      # tables, figures, estimate
python -m pytest tests/test_m45c_full.py tests/test_m45c_training.py tests/test_m45c_horizon.py
```

Gated, not run: `battery --confirm-m45c-battery`, `attackers --confirm-m45c-attackers`,
`health_eval --health-parts curve_rest,headline,tournament --confirm-m45c-comprehensive`,
`traces --confirm-m45c-comprehensive`.
