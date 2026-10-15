# M4.5c — Research report: shared-policy PPO self-play on the calibrated laning model

*2026-10-15. Branch `main` on 45b7dfa with the uncommitted M4.5c horizon study, campaign and evaluation code; nothing
staged or committed. Environment `GameConfig.m45_calibrated()`, unchanged throughout: no rule, reward, observation,
action, network or PPO setting was changed. This report completes M4.5c; it builds on
[the pilot](m4_5c_pilot_report.md), [the runtime audit](m4_5c_runtime_audit.md),
[the horizon study](m4_5c_horizon_study.md) and [the 8.192M campaign](m4_5c_training_report.md).*

- Generated tables:
  - [`results/m45c/final/report.md`](../../results/m45c/final/report.md): both budgets, battery, attackers, traces;
  - [`results/m45c/ext/report.md`](../../results/m45c/ext/report.md): the 16.384M runs, the comprehensive standard
    evaluation and the budget comparison;
  - [`results/m45c/full/report.md`](../../results/m45c/full/report.md): the 8.192M runs;
  - [`results/m45c/battery/report_ext.md`](../../results/m45c/battery/report_ext.md) and
    [`report_full_replication.md`](../../results/m45c/battery/report_full_replication.md): the strategic battery.
- Every number below that is not computed in the text comes from those tables.
- Figures: `results/m45c/{final,ext,full}/figures/`.
- Checkpoints, battery jobs and attackers (git-ignored): `results/rl/checkpoints/m45c/{full,ext,battery,attack}/`.

## Summary

**The question.** Can PPO self-play independently discover economically meaningful League-inspired laning
strategies, without explicit strategy shaping?

**Answer: yes, conditionally.** The trained policies discovered:
- state-conditioned wave management;
- conditional freezing and the correct reply to freezes;
- selective Trading under minion fire;
- concession under threat;
- well-timed Recalls.

Each is shown by paired counterfactuals, not by action frequencies. They did **not** adopt strategies that this
model's economics do not reward against competent opponents (frequent Wards, sustained freezes, bullying by a stronger
champion), and the counterfactuals show that restraint is correct.

What remains open:
- the policies leave value against conceding opponent styles they never meet in self-play;
- a specialist attacker can exploit each one by about one resource per game.

**What was run (all on `m45_calibrated()`, unchanged):**

| stage | content | wall clock |
|---|---|---|
| matched 2× extension | six fresh runs × 16,384,000 transitions; 102 / 102 checks; no alarm | 93 min |
| budget comparison and comprehensive standard evaluation | 21-point curves, W = 1000 headline, history, tournaments, cross-play, 16.4M vs 8.2M, the annealing test, historical references, behaviour, 39 verified viewer traces | 2 h 0 min |
| strategic battery | 12 families (A–I, plus early-game A and C) × 6 finals, replication on the 8.192M finals, exact one-turn mechanics, the 300-turn validation subset | 1 h 37 min |
| exception 3 | 32 (policy, family) pairs re-run at 150 turns | 58 min |
| attackers | six learned best responses at 4,096,000 learner transitions each, with evaluation | 1 h 6 min |

**Main results.**

| question | result |
|---|---|
| did the extension help? | **yes**: 16.384M finals beat 8.192M finals in **all 36 pairs** (+1.03 ± 0.11 cross-seed). Against the fixed suite +4.00 vs +3.40, against the M4.4b network +3.02 vs +2.41, with the same behaviour profile |
| annealing or data? | **data**: an annealed 8.192M final is no stronger than a still-learning 8.192M midpoint (−0.07 ± 0.25). The 8.192M campaign's second-half gain was real learning |
| converged? | **no, but slowing**: +0.36 ± 0.30 against the 12.3M checkpoint; within noise against 13–15M; no cycling; the six seeds are interchangeable |
| strength | beats all nine hand-designed strategies by +2.6 to +7.5 and the OOD M4.4b network by +3.2, from both seats |
| wave management | no fixed wave action beats the policy's choice (−0.11 to −0.25 one-turn, −0.35 / −0.61 for 4-turn plans). Shove (33% of decisions) is used where it pays: it has the best immediate gold, but the policies forgo it when later costs dominate |
| freezing | worth what the opponent concedes: +0.56 natural / +1.86 constructed against a back-off opponent, about 0 against breakers, −0.20 against the learned policies. Accepting a freeze costs 0.2–0.4; breaking once is right |
| Trading | trading into a heavier wave costs 0.2–0.3 and the policies avoid it; single trades are near neutral |
| zoning | the weaker side concedes. The stronger side's sustained bullying costs 0.2–1.0 against learned play and pays only against a conceding scripted opponent (+1.28) |
| Vision | a forced Ward costs 0.3–0.6: low Ward use is economically justified on this model |
| Recall | staying at low HP costs about 0.9; recalling equals the policy's choice; interrupted attempts are not shown to be mistakes |
| horizons | the 50-turn default held. 32 failed validations re-run at 150 turns changed nothing: mean change +0.006 ± 0.010, slope 1.04, 0 contradictions in 80 doubly significant comparisons |
| exploitability (lower bound, quarter-budget attackers) | +1.02 ± 0.48, 6/6 significant; target-specific (≈ 0 against other finals); won by target-tuned aggression and freezing |

**Assessment: GO WITH CAVEATS** for the next phase (§9).
- **Research:** opponent-style exploitation, convergence, and the battery's validation rule.
- **Simulator design (review questions, not tuning):** the Vision payoff, freeze durability and the zoning payoff.

---

## 1. The training extension

**What was run.**
- The pre-registered matched 2× extension, authorized after the 8.192M campaign met its extension criterion.
- Six **fresh** runs, seeds 0–5, of 16,384,000 agent transitions each (98,304,000 in all; 8,192,000 joint steps and
  4,000 PPO updates per run).
- The learning rate 3e-4 was annealed linearly to 0 over each run's own 16,384,000.
- Every other setting is the 8.192M campaign's: shared-policy PPO, 100% current-policy self-play, the pilot's
  hyperparameters, `lane_cache = 5,000`, snapshots every 25,600 transitions (640 per run).
- **No weights were transferred.** Each run starts from `torch.manual_seed(seed)`, so its initial weights equal
  those of the same seed's 8.192M run (digests checked). The two budgets are therefore paired by seed.
- The six runs ran concurrently in 93 min, 01:31 → 03:04, at about 2,935 transitions/s per process, peaking at
  0.64 GB per process. Checkpoints are in `results/rl/checkpoints/m45c/ext/s<seed>/` and the record is
  `results/m45c/ext/ext_train.json`.

**Correctness: all 102 checks pass (17 per run), and no pre-registered alarm fired.**

| invariant | evidence |
|---|---|
| valid action masking | 0 illegal samples and 0 never-legal offers in 98.3 M transitions |
| exact zero-sum accounting | Σ reward A + Σ reward B = 0 in every one of 24,000 rollouts |
| transition counts | 2 agent transitions per joint step, equal seats, exactly 16,384,000 and 4,000 updates per run |
| finite losses and gradients | every loss, gradient and critic statistic finite; gradient norm ≤ 1.82 (alarm 100) |
| stable KL and entropy | KL ≤ 0.0092 per update (maximum 10-update mean ≤ 0.007; alarm 0.05); entropy ≥ 1.53 (floor 0.3) |
| critic | V(s0) between −0.30 and +0.10 resources (true value 0), |V| ≤ Ω |
| deterministic seeds | the initial digests equal the 8.192M runs' digests seed for seed. Reproducibility of a full-budget run was shown on the 8.192M campaign (bit-identical seed-0 re-run); no further re-run was needed |
| checkpoints | 640 snapshots per run; the midpoint (8,192,000) present; the last snapshot bit-identical to `final.pt` |

**Training diagnostics compared with the 8.192M runs**, by tenth of each run's own schedule (seed means):

| quantity | 8.192M runs, last tenth | 16.384M runs, last tenth | over the schedule |
|---|---|---|---|
| entropy | 1.81 | 1.80 | both fall to about 1.74 by 30–40% of the schedule, then rise slowly |
| approximate KL, clip fraction | → 0 | → 0 | both follow the annealed learning rate |
| explained variance | 0.45 | 0.45 | flat |
| gradient norm (pre-clip) | 0.76 | 0.79 | slow rise in both |
| sampled Shove share | 31% | 31% | the extension reaches 31% by 40% of its schedule and stays |
| sampled Ward share | 1.3% | **1.7%** | higher in the extension from the fourth tenth on |
| sampled Recall share | 7.2% | 7.4% | |
| Safe share of lane behaviours | 20% | **22%** | keeps rising through the extension; Yield falls (26% → 24%) |
| deaths per 1000 champion-turns | 6.2 | 6.6 | |
| Recall attempts interrupted | 39% | 42% | |
| ganks per 1000 turns | 119 | 115 | slowly falling through the extension |

The extension trained as cleanly as the 8.192M campaign. The only drifts that continue past 8M transitions are small:
more Safe, less Yield, slightly more Ward, fewer ganks suffered.

## 2. Budget comparison: 16.384M against 8.192M

**Design.** Both sets of finals completed their own annealing schedule. All comparisons use the same fixed, paired
evaluation seeds and both seats, and every difference is paired by training seed. The comparisons were specified
before any extension result (`health.py`):

- **direct play**, every 16.384M final against every 8.192M final: 36 pairs, W = 1000, 64 paired seeds, seed 94,500;
- **the annealing test**, every 8.192M final (annealed to 0) against every one of the extension's 8.192M midpoints.
  Both have trained on exactly 8,192,000 transitions; the midpoints are still at half the learning rate. Seed 94,550;
- **curves against the fixed suite** at the same fractions of each schedule (seed 93,000);
- **historical references**, the pilot final (1.02M) and the OOD M4.4b network (seed 94,600);
- **behaviour readouts** on the same record seeds.

### 2.1 Strength

| comparison | result |
|---|---|
| **16.384M finals vs 8.192M finals**, same seed | **+1.09 ± 0.11** (6 of 6 significant) |
| 16.384M finals vs 8.192M finals, cross-seed | **+1.03 ± 0.11**. **All 36 pairs significant**, range +0.55 to +1.59 |
| **annealing test**: 8.192M finals vs the extension's 8.192M midpoints, same seed | **−0.07 ± 0.25** |
| annealing test, cross-seed | −0.11 ± 0.12 (2 of 36 pairs above 0, 5 below) |
| scripted-suite mean at the final (W = 460) | 8.192M **+3.40 ± 0.41** → 16.384M **+4.00 ± 0.20**; paired difference **+0.60 ± 0.49** (5 of 6 seeds up, s3 equal) |
| the OOD M4.4b network at the final (W = 460) | +2.41 → **+3.02 ± 0.21**; paired difference **+0.60 ± 0.42** (6 of 6 up) |
| the pilot final (W = 1000) | +2.00 ± 0.21 → +2.33 ± 0.25 |
| the OOD M4.4b network (W = 1000) | +2.54 ± 0.18 → +3.13 ± 0.15 |

- **The extension produced stronger policies, in every seed and against every reference.** The direct comparison is
  unambiguous: about one resource per game against an 8.192M final of any seed.
- **Annealing alone is worth about nothing.** A policy annealed to 0 at 8.192M transitions is no stronger than a
  policy that has seen the same 8.192M transitions while still learning at half the rate (−0.07 ± 0.25). The 8.192M
  campaign's open caveat was that "final − midpoint" mixed more data with annealing. It is settled: **its +0.96
  second-half gain came from the additional 4.1M transitions, not from annealing**.
- **Strength against the fixed suite, which had looked flat over the 8.192M campaign's second half, kept rising** over
  the extension's last quarter: +3.58 at 12.3M to +4.00 at 16.4M (figure `results/m45c/final/figures/budget_curves.png`).
  The 8.192M runs' flat second half was therefore not a ceiling of the fixed suite. It came from progress the suite
  measures only slowly.

### 2.2 Is the 16.384M policy converged? No, but it is slowing

| quantity (16.384M runs) | value |
|---|---|
| final vs own midpoint (8.192M, W = 1000) | +0.97 ± 0.51 (seeds +0.18 to +1.55) |
| final vs own 12.288M (W = 1000) | +0.36 ± 0.30 |
| final vs 13.1M / 14.7M (W = 460) | +0.11 to +0.47 / −0.07 to +0.31: no longer significant |
| pre-registered plateau rule | 2 of 6 runs plateaued (s2, s4); 4 not |
| adjacent checkpoints | **no significant regression in any of 66 comparisons** |
| cross-play of the 16.384M finals | largest \|value\| 0.43, **no pair significant**, row means −0.21 to +0.15 |

- **Still improving.** The pre-registered rule would again call for an extension: 4 of 6 runs did not plateau, and
  final − midpoint exceeds +0.5. **No further extension is authorized, and none was run.**
- **But the improvement is slowing.** The gain against the 12.3M checkpoint is a third of the gain against the
  midpoint, and against 13–15M it is within noise.
- **What this means for the strategic evaluation.** It describes policies that are well trained but not converged.
  Per-budget improvement is about +1.0 per doubling, at the doubling from 8M to 16M.
- **The six 16.384M finals are even more alike than the 8.192M finals.** Cross-play: no significant pair, against 2 of
  15 at 8.192M.

### 2.3 Behaviour: the profile is stable; a few details moved

Self-play readouts of the finals: seed means, with differences paired by seed (16.384M − 8.192M). Tables are in
`results/m45c/final/report.md` §2.

| readout | 8.192M | 16.384M | difference | |
|---|---|---|---|---|
| Shove (probability mass) | 0.323 | 0.329 | +0.005 ± 0.038 | unchanged |
| Ward mass | 0.016 | 0.022 | +0.006 ± 0.008 | n.s. (Wards per 1000 turns 14.8 → 20.1, n.s.) |
| Trade (of lane behaviours) | 0.243 | 0.264 | **+0.021 ± 0.016** | slightly more Trading |
| Safe at HP < 35% | 0.795 | 0.755 | **−0.039 ± 0.028** | slightly less retreating at low HP |
| Shove at levels 1–5 | 0.293 | 0.231 | **−0.062 ± 0.046** | less early Shove |
| freezes per 1000 turns | 3.3 | 2.7 | −0.6 ± 1.0 | n.s. |
| freeze denial share | 0.69 | 0.69 | 0.00 ± 0.10 | unchanged |
| Recall attempts interrupted | 0.356 | 0.411 | **+0.055 ± 0.037** | more interrupted |
| farm collection | 0.767 | 0.759 | −0.008 ± 0.016 | n.s. |
| power gap: the stronger side's position | +1.30 | +2.03 | | further forward |
| power gap: the weaker side's gold left uncollected when the stronger side is near | +0.008 | **+0.014 ± 0.005** | | now significantly above 0, but tiny |

**The extension made the policies about one resource per game stronger without changing what they do in any
qualitative way.** This replicates M4.4b's matched-extension result.

What did move:
- the stronger side presses further forward in a power gap;
- Shove is used less in the early game;
- Trading rises slightly;
- more Recall attempts are interrupted.

None of these is large. The strategic battery (§4–§5) asks whether they are good decisions.

## 3. Policy strength of the primary policy set (the six 16.384M finals)

All values are evaluate_discounted: paired seats on one chance stream, the discounted gold lead in resources, 95%
intervals over paired evaluation seeds, then seed means with t intervals over the six training seeds. Tables:
`results/m45c/ext/report.md` §3–§6 and §10.

**Full learning curve:** 21 points, every 819,200 transitions, against the nine scripted strategies and the OOD M4.4b
network (32 paired seeds, W = 460):
- untrained −7.02, 0.82M +2.46, 3.28M +3.09, 8.19M +3.24, 12.3M +3.58, **final +4.00 ± 0.20** (seeds +3.70 to +4.24);
- OOD: −10.09 → +3.02 ± 0.21.
- The curve rises quickly to about +3 by 3M and then slowly but steadily. The point-to-point noise of a single seed is
  about ±0.5, so the slow phase is visible only in the seed mean.

**Headline** (W = 1000, 64 paired seeds, both seats):

| opponent | role | 16.384M finals | 8.192M finals | difference (paired) |
|---|---|---|---|---|
| g_farm | passive farming | +2.62 ± 0.32 | +2.62 ± 0.21 | −0.01 ± 0.33 |
| g_shove | plain Shove | +3.27 ± 0.29 | +2.33 ± 0.39 | **+0.94 ± 0.64** |
| g_shove_sr | Shove with Recall | +3.25 ± 0.31 | +2.41 ± 0.43 | **+0.85 ± 0.66** |
| wk_shove_reset_sh | Shove with frozen-on reset | +3.28 ± 0.28 | +2.49 ± 0.44 | **+0.79 ± 0.64** |
| g_shove_edge | Shove, selective trades | +4.06 ± 0.31 | +3.08 ± 0.36 | **+0.99 ± 0.54** |
| g_bully | aggressive Trading | +7.46 ± 0.51 | +7.21 ± 0.57 | +0.25 ± 0.78 |
| g_safe | defensive Safe | +6.16 ± 0.27 | +6.30 ± 0.28 | −0.14 ± 0.36 |
| fz_guard_sh | freeze-oriented | +3.25 ± 0.29 | +2.43 ± 0.34 | **+0.82 ± 0.58** |
| g_shove_ward | Ward-oriented Shove | +2.66 ± 0.54 | +1.95 ± 0.47 | +0.71 ± 0.90 |
| M4.4b aggro (OOD) | learned, earlier model | +3.17 ± 0.26 | +2.42 ± 0.43 | **+0.75 ± 0.58** |
| **scripted mean** | | **+4.00 ± 0.24** | **+3.43 ± 0.29** | **+0.58 ± 0.47** |

- **Every opponent is beaten by every final**, from both seats.
- **There is a seat difference:** seat A +4.25 ± 0.27, seat B +3.76 ± 0.24 on the scripted mean. The 8.192M finals
  show a smaller gap in the same direction (+3.56 / +3.29).
  - The game is mirror-symmetric, and self-play trains one policy on both seats (V_A(s0) = V_B(s0) exactly).
  - The pilot showed the opposite direction against several opponents (for example `g_shove` +1.37 A, +1.83 B).
  - So the difference reflects how particular policy–opponent pairs play out from each seat on the paired chance
    streams. It was not investigated further, and every conclusion here uses the mean of the two seats.
- **The extension's gain is where the scripted opponents are strongest.** The Shove-centred strategies are the
  restricted-game equilibria of M4.5b §7; against them the finals gained +0.8 to +1.0. Against the extremes (passive
  farming, always-Safe, always-Trade), already beaten by +2.6 to +7.5, nothing changed.

**Historical PPO references** (W = 1000):
- against the M4.5c pilot final (1.02M transitions): +2.33 ± 0.25 (8.192M finals: +2.00 ± 0.21);
- against the OOD M4.4b aggro network: +3.13 ± 0.15 (+2.54 ± 0.18).

**Against their own past.**
- Every final beats its untrained network by +11.3 to +11.8.
- They beat every curve checkpoint up to 11.5M significantly.
- Against 13.1M and 14.7M the differences are within noise (§2.2).
- **The per-run tournament** (9 checkpoints from 3.3M to 16.4M, round robin, W = 460, 16 paired seeds) orders the
  checkpoints by training time in every run. Mean scores rise from about −1.2 at 3.3M to +0.6 to +1.1 at the final.
  Only 2 of 216 pairs show a later checkpoint losing significantly, both in seed 5 (9.8M over 11.5M and 13.1M); that
  is within chance. **No self-play cycling.**

**Across seeds:**
- The six 16.384M finals are interchangeable in cross-play: largest |value| 0.43, no pair significant.
- Every 16.384M final beats every 8.192M final (§2.1).

## 4. The strategic battery: design as run

The approved short-horizon methodology (horizon study §6) was run on the **six 16.384M finals** (the primary set).
The replication families ran on the **six 8.192M finals** (the comparison group). Code: `m45c/battery.py`. Raw jobs:
`results/rl/checkpoints/m45c/battery/`. Tables: `results/m45c/battery/report_ext.md`, `report_full_replication.md`,
and `results/m45c/final/report.md` §3–§5.

**Method.**
- **Comparisons.** Every comparison is a paired, common-random-number difference between a candidate and the policy's
  own choice from the same start state. Values in this report are that difference, "candidate − policy", in
  resources. Negative means the policy's own choice was better.
- **Statistic.** The primary statistic is ΔQ₅₀ (critic-bootstrapped) for natural states and ΔG₅₀ (observed return)
  for constructed states. The other statistic is always reported beside it.
- **Horizons.** Each rollout is recorded at 1, 2, 3, 5, 10, 15, 20, 30, 40 and 50 turns. The freeze families run 60
  turns (exception 1: 20-turn plans + 40).
- **Candidates.** One-turn candidates ("1t:") and sustained plans ("plan:") are reported separately.
- **Opponent protocols.** The policy itself (sampling), its modal action, and style opponents:
  - freeze holder's opponents: the breaker `wk_shove`, the back-off `wk_back`, the reset `wk_shove_reset`;
  - frozen-on side's opponents: the freezers `fz_passive_sh`, `fz_guard_sh`, `fz_bully_sh`;
  - Trading and zoning: the wary `g_wary`;
  - Ward: the punisher `g_bully`.

**Sizes, per final.**
- Natural families: 48 states (24 for the two early-game families), each from a different self-play game of that
  final, with K = 12 paired rollouts.
- Constructed families: K = 24 on:
  - F: 10 freeze cells;
  - G: 12 trade-bench cells (levels 4 / 11 / 18 × fronts × power equal / ×1.76);
  - H: 12 zoning cells (A stronger / equal / weaker at level 10, stronger at level 6);
  - I: 8 Ward cells (levels 10 and 6).
- Every state also gets the **exact one-turn expectation** of every candidate's first action (§5.7).

**Validation subset (§6.3).** A fixed-seed max(8, 15%) of each natural family's states, and every constructed cell,
ran to 300 turns under protocol "policy".

**Volume and time.** 6 × 1,226 jobs plus the replication (6 × 164 jobs). The battery itself took 1 h 37 min on 11
workers (05:03 → 06:40), well under the 2.2 h estimate. The 150-turn re-runs of exception 3 followed (§6).

## 5. Counterfactual results: which decisions pay, and when

Values are seed means of the six 16.384M finals (each a mean over that final's states), ± t intervals over finals.
"k/6" counts the finals whose own interval excludes 0 in that direction. Unless stated, the protocol is "policy" and
the horizon 50.

### 5.1 Wave management (family A: 48 natural states per final where the two fronts are in contact)

| candidate (vs the policy's own choice) | ΔQ₅ | ΔQ₂₀ | ΔQ₅₀ | ΔG₅₀ |
|---|---|---|---|---|
| 1t Last Hit + Hold | −0.23 ± 0.07 | −0.22 ± 0.10 | **−0.25 ± 0.08** | −0.26 |
| 1t Thin + Hold | −0.19 ± 0.05 | −0.18 ± 0.10 | **−0.23 ± 0.11** | −0.23 |
| 1t Slow Push + Hold | −0.18 ± 0.05 | −0.15 ± 0.08 | **−0.19 ± 0.12** | −0.19 |
| 1t Shove + Hold | −0.14 ± 0.03 | −0.09 ± 0.06 | **−0.11 ± 0.10** | −0.12 |
| plan Thin ×4 | −0.50 ± 0.14 | −0.53 ± 0.14 | **−0.61 ± 0.16** | −0.64 |
| plan Shove ×4 | −0.34 ± 0.08 | −0.29 ± 0.13 | **−0.35 ± 0.20** | −0.36 |

- **No fixed wave action beats the policy's state-conditioned choice.**
  - Per final, 23 of the 24 one-turn values are negative (the exception: Shove, +0.02 on seed 0), and all 12 plan
    values are.
  - Under the modal-opponent protocol, every mean is negative too (−0.18 to −0.74).
- **Shove is the best fixed alternative, but still worse than the policy.**
  - *Immediately*, Shove is worth **+0.45** more gold than the policy's modal action (exact one-turn mechanics, §5.7).
  - Yet the modal action is Shove + Hold in only 22% of these states.
  - Over 50 turns, forcing Shove costs −0.11 once and −0.35 over four turns.
  - **The policies give up immediate gold for later value. That is the decisive evidence that their Shove use (about
    33% of decisions) is conditional, not dominance.**
- **Sustained Thin is the worst plan** (−0.61): holding a wave back for four turns loses farm the policy would take.
- **Early game** (levels ≤ 10, 24 states per final): the same ordering, smaller and less certain. One-turn
  alternatives −0.13 to +0.02, plans −0.19 to −0.22, intervals ±0.2. Shove is again closest to the policy's choice
  (+0.02 ± 0.22).

### 5.2 Freezing

**Natural states where the focal side holds a freeze** (B freeze, 48 per final):

| candidate | vs the policy (learned) | vs `wk_back` (backs off) | vs `wk_shove` (breaks) | vs `wk_shove_reset` |
|---|---|---|---|---|
| 1t keep (Thin + Hold) | −0.02 ± 0.12 | +0.07 ± 0.08 | −0.00 ± 0.08 | −0.01 ± 0.10 |
| 1t guard (Thin + Trade) | −0.05 ± 0.14 | −0.09 ± 0.08 | −0.03 ± 0.08 | −0.03 ± 0.10 |
| 1t break (Shove + Hold) | +0.02 ± 0.09 | **−0.23 ± 0.11** | −0.09 ± 0.11 | −0.11 ± 0.14 |
| **plan keep** (Thin + Hold while held, ≤ 20 turns) | **−0.20 ± 0.07** | **+0.56 ± 0.09 (6/6)** | −0.06 ± 0.07 | −0.10 ± 0.12 |
| plan guard (the `fz_guard` rule) | −0.06 ± 0.12 | **+0.33 ± 0.17** | +0.03 ± 0.09 | +0.01 ± 0.09 |

**Constructed freeze grid** (F, 10 cells, ΔG₅₀; the freezer A at its tower holding B's wave):

| candidate | vs the policy | vs `wk_back` | vs `wk_shove` | vs `wk_shove_reset` |
|---|---|---|---|---|
| plan keep | −0.14 ± 0.30 | **+1.86 ± 0.22 (6/6)** | +0.03 ± 0.14 | +0.02 ± 0.17 |
| plan guard | −0.14 ± 0.16 | −0.05 ± 0.17 | +0.07 ± 0.12 | −0.11 ± 0.16 |
| 1t keep | −0.10 ± 0.24 | +0.23 ± 0.10 | +0.12 ± 0.12 | +0.13 ± 0.14 |

- **A freeze is worth what the opponent concedes.**
  - Sustaining it against an opponent who backs off denies **+0.56** (natural) to **+1.86** (constructed) resources
    over 50 turns, in every final.
  - Against an opponent who walks up and Shoves, or resets, it is worth about 0: the freeze is gone in a turn or two.
  - Against the learned policies themselves, sustaining it **costs** −0.20 ± 0.07: they break freezes (33–52% Shove
    when frozen on).
  - This replicates M4.4c's finding on the new model, with a larger and better-powered design.
- **The policies' freeze behaviour fits their opponents.**
  - Freezes are rare (1.9–3.8 per 1000 turns) but succeed when they are held (60–78% denial; mean 69%, §2.3).
  - In a held freeze, the policy's own choice is at least as good as keeping or guarding for one turn (−0.02, −0.05).
  - Against the policy as opponent, it beats committing to the freeze (−0.20).
  - **Rarity here is the right answer to opponents who break freezes, not a failure to discover them.**
- **Guarding a freeze** (Thin + Trade a walk-up) is worth less than keeping it passively against a back-off opponent
  (+0.33 against +0.56; −0.05 against +1.86 on the grid). Trading costs HP and invites the wave's retaliation, and
  against an opponent who has already backed off it adds nothing.
- **The timing of a freeze's value.** The keep plan costs about −0.5 on turn 1: Thinning gives up the turn's farm.
  It turns positive by turn 5–20, and its value is complete by turn 20. From the raw G: +1.87 at 20 turns, +1.86 at
  50. The horizon study's case study is confirmed on the trained policies.

**Natural states where the focal side is frozen on** (B frozen, 48 per final), against three freezer styles:

| candidate | vs the policy | vs `fz_passive_sh` | vs `fz_guard_sh` | vs `fz_bully_sh` |
|---|---|---|---|---|
| 1t Shove (break) | −0.02 ± 0.09 | −0.01 ± 0.08 | −0.03 ± 0.11 | +0.02 ± 0.06 |
| 1t walk up and farm (Last Hit + Hold) | **−0.22 ± 0.10** | **−0.27 ± 0.09** | **−0.29 ± 0.11** | **−0.37 ± 0.09** |
| 1t step back (Last Hit + Safe) | **−0.20 ± 0.11** | **−0.38 ± 0.14** | **−0.33 ± 0.11** | **−0.34 ± 0.11** |
| 1t Recall (where legal) | −0.08 ± 0.29 | −0.15 ± 0.40 | −0.22 ± 0.28 | −0.25 ± 0.15 |
| plan Shove ×3 | **−0.30 ± 0.06** | **−0.34 ± 0.13** | **−0.47 ± 0.20** | **−0.45 ± 0.13** |

- **The frozen-on response is right.** Accepting the freeze, whether by walking up to farm or stepping back, costs
  0.2–0.4 against every freezer style. Breaking it once with a Shove is as good as the policy's own choice. Shoving
  three turns in a row costs 0.3–0.5.
- **The learned response is therefore a single break, then back to state-conditioned play.**

### 5.3 Trading and positional pressure

**Natural states** (C trade, 48 per final): the enemy can be caught without a dive, and its front outweighs ours by a
full wave.

| candidate | vs the policy | vs modal | vs `g_wary` |
|---|---|---|---|
| 1t Trade | **−0.34 ± 0.16** | **−0.30 ± 0.16** | **−0.18 ± 0.06** |
| 1t Hold | **−0.34 ± 0.13** | **−0.40 ± 0.13** | −0.17 ± 0.10 |
| 1t Safe | **−0.11 ± 0.07** | −0.09 ± 0.06 | −0.10 ± 0.09 |

- **Trading into a heavier enemy wave is a mistake, and the policies avoid it.**
  - Forcing a Trade costs 0.2–0.3 against every opponent; standing in the wave (Hold) costs as much.
  - Here the policy's modal action is a Trade in only 15% of states.
  - Its choice beats even the Safe retreat (−0.11).
  - This is the minion-retaliation and positional-aggro mechanic of M4.5a at work: the immediate cost of a Trade
    (−0.38, exact one-turn) is not repaid later.
- **Early game** (levels ≤ 10): the same pattern, smaller (Trade −0.21 ± 0.13, Safe −0.03 ± 0.14).

**Constructed trade bench** (G, 12 cells; ΔG₅₀):
- A's 8-HP front against B's 8- or 24-HP front at levels 4, 11 and 18, equal power or A ×1.76.
- Trade, Hold and Safe are all within about ±0.2 of the policy's choice, pooled over cells. A single trade is near
  value-neutral (M4.5b caveat 4).
- Cell detail: `results/m45c/final/report.md` §3b.

**Zoning by a stronger champion** (H, constructed, ΔG₅₀): one-turn choices differ from the policy's by at most about
±0.6 in any cell, mostly within noise. The clear exception is a one-turn Shove at a neutral lane against the wary
opponent (−0.5 to −0.6). The decisive test is the sustained bully plan (Trade whenever catchable for 10 turns):

| situation | A ×1.76 stronger | equal | A weaker (B ×1.5) |
|---|---|---|---|
| **vs the policy**: neutral / A pushing / B pushing (level 10) | −0.23 / −0.99 / +0.15 | −0.39 / −1.36 / −1.74 | −1.05 / −1.76 / −1.64 |
| **vs `g_wary`** (retreats under threat): neutral / A pushing / B pushing | +0.26 / −0.43 / **+1.28 ± 0.32** | +0.20 / −0.22 / +0.52 | −0.64 / −1.15 / −0.39 |

- **Pressing a power advantage pays only against an opponent who concedes, in the right position.** Against `g_wary`,
  a ×1.76-stronger A that bullies while B pushes into A's half gains +1.28 ± 0.32.
- **Against the learned policies, sustained bullying never pays**, and it is very costly without a power advantage
  (−1.4 to −1.8). They trade back and punish over-extension.
- **The 16.384M policies' behaviour in a power gap is consistent with these economics.** The stronger side presses
  forward (position +2.0) but does not bully, and its denial is tiny (§2.3). The weaker side concedes position.
- **What the battery shows here:**
  - "concession without resource denial" is **not a failure to discover zoning**: zoning is not profitable against
    the policies' own responses;
  - it would be profitable against the passive concession a scripted wary opponent makes;
  - the policies never meet such opponents in self-play.

### 5.4 Vision

**Natural states** (D ward, 48 per final): Ward legal, the focal side's vision down, its front at the lane cell or
beyond.

| candidate | vs the policy | vs modal | vs `g_bully` |
|---|---|---|---|
| 1t Ward | **−0.43 ± 0.09** (6/6) | **−0.29 ± 0.11** | **−0.39 ± 0.11** |
| 1t Last Hit + Hold | **−0.25 ± 0.12** | **−0.27 ± 0.10** | **−0.39 ± 0.09** |

**Constructed Vision situations** (I: neutral / forward / pre-siege / defending at levels 10 and 6; ΔG₅₀): Ward
−0.61 ± 0.28 against the policy, −0.50 against modal, −0.18 against `g_bully`.

- **In the states where it is legal, a Ward costs 0.3–0.6 resources over 50 turns relative to the policy's choice**,
  against every opponent. That includes the punishing `g_bully`, against whom Vision should matter most.
- **The cost is the turn spent walking to W.** Exactly −0.72 of immediate lead (§5.7). The later benefit (fewer
  successful ganks on exposed turns) recovers only part of it by turn 50.
- **The policies' low Ward use (about 2% of decisions; 16–27 per 1000 turns) is economically justified on this model.**
  They Ward rarely and at their own chosen moments. When the battery forces a Ward where they did not choose one,
  it loses. This agrees with M4.5b (Ward value situational and contestable) and is a property of the calibrated
  Vision economy, not a learning failure.

### 5.5 Recall

**Natural states** (E recall, 48 per final): Recall legal, HP ≤ 50%.

| candidate | vs the policy | vs modal |
|---|---|---|
| 1t Recall now | −0.03 ± 0.19 | −0.28 ± 0.16 |
| plan stay (Shove + Hold ×3) | **−0.89 ± 0.18** (6/6) | **−1.23 ± 0.30** |
| plan step back (Last Hit + Safe ×3) | **−0.33 ± 0.16** | −0.17 ± 0.29 |

- **Recall timing is appropriate.**
  - Recalling now is as good as the policy's own choice; its modal action here is Recall in 46% of states.
  - Staying in lane at low HP costs about one resource, through deaths and lost farm.
  - Stepping back instead of resetting costs a third of that.
- **The interrupted Recalls** (41% of attempts at 16.384M) **are not shown to be a mistake.** At these states,
  committing to Recall is not worse than the policy's mixed choice against the learned opponent. Against the modal
  opponent, who interrupts deterministically, it is −0.28. This is M4.4c's "rational guessing game": the attempt is
  worth making even though it is often interrupted.

### 5.6 Replication on the 8.192M finals

The replication families (freeze, frozen-on, Recall, freeze grid; protocols policy and modal; 24 natural states per
final) ran on the six 8.192M finals: `results/m45c/battery/report_full_replication.md`.

| comparison (protocol "policy", ΔQ₅₀; F: ΔG₅₀) | 8.192M finals | 16.384M finals |
|---|---|---|
| holding a freeze: plan keep | −0.17 ± 0.21 | −0.20 ± 0.07 |
| holding a freeze: 1t keep / guard / break | −0.03 / −0.06 / −0.02 | −0.02 / −0.05 / +0.02 |
| frozen on: walk up and farm / step back | −0.19 ± 0.15 / −0.21 ± 0.27 | −0.22 ± 0.10 / −0.20 ± 0.11 |
| frozen on: 1t Shove / plan Shove ×3 | −0.08 ± 0.12 / −0.33 ± 0.19 | −0.02 ± 0.09 / −0.30 ± 0.06 |
| low HP: Recall now / stay ×3 / step back ×3 | +0.11 ± 0.19 / −0.96 ± 0.31 / −0.29 ± 0.18 | −0.03 ± 0.19 / −0.89 ± 0.18 / −0.33 ± 0.16 |
| freeze grid: plan keep / plan guard | −0.06 ± 0.19 / −0.03 ± 0.25 | −0.14 ± 0.30 / −0.14 ± 0.16 |

**Every replicated conclusion holds at half the budget**, with the same signs and similar sizes:
- committing to a freeze against learned play costs a little;
- accepting a freeze costs more than breaking it once;
- over-shoving after a break costs;
- staying in lane at low HP costs about a resource;
- recalling is at least as good as the policy's own choice.

**These are properties of PPO-trained policies on this model at both budgets,** not of one training stage. (The
replication used protocols policy and modal, so the style-opponent results — the freeze's value against a back-off
opponent — rest on the 16.384M finals alone.)

### 5.7 Exact one-turn mechanics

The exact one-turn expectation of each candidate's first action against the opponent's modal reply (chance branches
and ganks enumerated; `battery.mechanics`), compared with the policy's modal action, where the candidate's action is
legal:

| family | candidate | immediate lead difference | the policy's modal action already is the candidate |
|---|---|---|---|
| A wave | Shove + Hold | **+0.45** | 22% |
| A wave | Last Hit / Thin / Slow Push + Hold | −0.47 / −0.32 / −0.27 | 1–2% |
| B freeze | break (Shove) | +0.30 | 35% |
| B freeze | keep / guard | −0.25 / −0.21 | 3–4% |
| B frozen | Shove | +0.18 | 54% |
| B frozen | walk up / step back / Recall (legal in 36%) | −0.62 / −0.81 / −0.73 | 1% / 2% / 26% |
| C trade | Trade / Hold / Safe | −0.38 / −0.36 / −0.31 | 15% / 0% / 5% |
| D ward | Ward | **−0.72** | 2% |
| E recall | Recall | −0.30 | 46% |
| E recall | stay (Shove) | +0.62 | 2% |

**What the immediate values say.**
- In almost every family the action with the **best immediate gold** is a Shove (or staying to Shove at low HP).
- **The policies choose it far less often than an immediate-gold maximizer would**, and the 50-turn counterfactuals
  say they are right (§5.1, §5.5).
- Their decisions price delayed consequences: the next wave, the freeze it breaks, the death it risks, the reset it
  forgoes. Those are exactly what a myopic or Shove-dominant policy would miss.

## 6. Horizon validation

**The pre-registered check (horizon study §6.3).** For each policy and family, on its 300-turn validation units:
- the pooled bias of the primary statistic at 50 turns against its 300-turn value, |bias| ≤ max(0.15, 2 SE);
- the slope of the 300-turn value on the 50-turn value across (state, candidate) pairs, within [0.8, 1.25];
- no contradiction: no pair significant at both horizons with opposite signs.

Any failure re-runs that family's whole design for that policy at 150 turns (exception 3).

| family | per-policy verdicts: pass / fail (6 policies) | bias, mean over policies | slope, mean over policies | contradictions |
|---|---|---|---|---|
| A wave | 3 / 3 | −0.07 ± 0.11 | 0.92 ± 0.18 | 0 |
| A wave, early game | 3 / 3 | +0.02 ± 0.17 | 1.10 ± 0.13 | 0 |
| B freeze | 2 / 4 | +0.01 ± 0.27 | 0.96 ± 0.14 | 0 |
| B frozen | 4 / 2 | +0.05 ± 0.15 | 1.02 ± 0.17 | 0 |
| C trade | 5 / 1 | +0.01 ± 0.08 | 1.05 ± 0.16 | 0 |
| C trade, early game | 1 / 5 | +0.06 ± 0.32 | 1.09 ± 0.26 | 0 |
| D ward | 5 / 1 | +0.00 ± 0.14 | 0.99 ± 0.19 | 0 |
| E recall | 5 / 1 | −0.03 ± 0.16 | 0.97 ± 0.10 | 0 |
| F freeze grid | 4 / 2 | +0.09 ± 0.10 | 0.90 ± 0.09 | 0 |
| G trade bench | 3 / 3 | −0.04 ± 0.07 | 0.88 ± 0.26 | 0 |
| H zoning | 6 / 0 | −0.01 ± 0.12 | 0.93 ± 0.10 | 0 |
| I Ward situations | 3 / 3 | +0.02 ± 0.19 | 1.07 ± 0.22 | 0 |
| replication on the 8.192M finals (4 families) | 20 / 4 | −0.04 to −0.09 | 0.97 to 1.14 | 0 |

**What the verdicts show.**
- **Failures.** 28 of the 72 primary (policy, family) checks failed, and 4 of the 24 replication checks.
- **No real bias.** Pooled over policies, the bias is within ±0.09 in every family, with every interval covering 0.
  Mean slopes are 0.88 to 1.10.
- **No contradiction anywhere**, in any of the 96 checks.
- **The failures are per-policy fluctuations.** A slope from 16–50 pairs, or a bias whose 2 SE is about 0.2, often
  falls just outside the window: for example slope 0.77, or bias +0.32 with a 2 SE of 0.21.

**Exception 3 was applied as written.** All 32 failing (policy, family) pairs were re-run at 150 turns, for all their
states and protocols: 1 h, 06:39 → 07:37. The 150-turn rollouts extend the same trajectories as the 50-turn ones
(same states, same common random numbers), so the comparison is exact:

| 50 vs 150 turns on the re-run families | value |
|---|---|
| (policy, family, protocol, candidate) comparisons | 419 |
| mean change, 150 − 50 | **+0.006 ± 0.010** |
| slope of the 150-turn value on the 50-turn value | **1.04** |
| significant at both horizons | 80, **of which contradicting: 0** |
| significant only at 150 / only at 50 | 21 / 30 |

**Conclusion on horizons: the approved 50-turn default held on the trained policies.**
- Extending every family that failed validation to 150 turns changed no conclusion. No sign reversed, and the mean
  change is zero.
- Significance moved in 51 comparisons in both directions, about as noise would.
- The reported 50-turn tables are therefore final. The 150-turn values, where computed, are in the battery report's
  extra column.
- **The validation rule itself, at this subset size, is too sensitive to sampling noise.** It flagged 33% of checks
  where no bias exists. A future battery should apply it pooled over policies per family, or enlarge the per-policy
  subset (§10).

**How consequences unfold over the horizon:**
- **1–5 turns, mechanical.** One-turn values at turn 1 already have the sign of their 50-turn values in most rows.
  The exceptions are exactly the delayed-payoff plans:
  - a freeze keep plan, −0.55 at turn 1 against a back-off opponent and +1.86 at 50;
  - Thin ×4 and Shove ×4, whose costs grow from −0.15 / −0.09 at turn 1 to −0.61 / −0.35.
- **10–20 turns, local economics.** Most values are complete by 20 turns (§5 tables: ΔQ₂₀ ≈ ΔQ₅₀).
- **50 turns, strategic.**
  - Constructed freeze and bully plans resolve between 20 and 50 turns: bully against `g_wary` in B's push, −1.96
    at 5 turns and +1.28 at 50.
  - Nothing changes after 50 (exception 3).

## 7. Exploitability: learned best-response attackers

**Protocol** (`m45c/attackers.py`; M4.3b, the deferred M4.4d):
- one attacker per 16.384M final;
- warm-started from the target's weights, so it starts at parity;
- trained with PPO against the frozen target only, with the target's own PPO settings and the learner's seat drawn
  per game;
- evaluated with 64 paired seeds at W = 460 and W = 1000 against its target, and at W = 1000 against the other five
  finals and the same seed's 8.192M final.

**Budget: 4,096,000 learner transitions each.** That is **one quarter** of the target's 16,384,000, not the half the
M4.3b protocol prescribes. As the brief specified, the budget stayed fixed rather than doubling with the target.
**This is a limitation:** the lower bounds below are weaker than a half-budget protocol would give. The six ran
concurrently in 52–55 min at about 1,280 learner transitions/s each. Checkpoints are in
`results/rl/checkpoints/m45c/attack/ext_s<seed>/`, the record in `results/m45c/attackers/attackers_ext.json`.

| target | attacker vs target, W = 460 | W = 1000 (95% interval) | seat A / B | vs the other five 16.384M finals | vs the same seed's 8.192M final |
|---|---|---|---|---|---|
| s0 | +1.30 | **+1.32** [+0.84, +1.80] | +1.00 / +1.64 | −0.44 ± 0.29 | +0.46 |
| s1 | +0.70 | **+0.71** [+0.27, +1.14] | +0.75 / +0.67 | +0.04 ± 0.46 | +0.00 |
| s2 | +0.78 | **+0.79** [+0.40, +1.19] | +1.18 / +0.40 | +0.26 ± 0.25 | +1.56 |
| s3 | +0.77 | **+0.78** [+0.32, +1.23] | +1.02 / +0.53 | −0.06 ± 0.20 | +1.10 |
| s4 | +1.79 | **+1.83** [+1.36, +2.29] | +1.78 / +1.87 | +0.08 ± 0.19 | +1.25 |
| s5 | +0.69 | **+0.71** [+0.30, +1.12] | +1.02 / +0.40 | −0.39 ± 0.23 | +0.25 |
| **mean** | +1.01 ± 0.47 | **+1.02 ± 0.48**, significant on **6 of 6** | | ≈ 0 | +0.77 |

**What this measures.**
- **Every final is exploitable by a learned specialist.** The lower bound is about **+1.0 resource per game**
  (+0.71 to +1.83), at a quarter of the target's budget.
  - That is roughly the strength of one budget doubling: 16.384M over 8.192M was +1.05.
  - It is small next to the finals' margin over every hand-designed strategy (+2.6 to +7.5).
  - Scripted baselines and ordinary self-play opponents do not find these weaknesses. Within self-play, every
    cross-seed pair is within ±0.43.
- **The weaknesses are target-specific.**
  - The same attackers are about even (−0.44 to +0.26) against the other five finals.
  - Against the same seed's 8.192M final they keep only part of their target's own advantage over it (+0.77 against
    the target's +1.09).
  - Each attacker found a response tuned to one network. It did not find a flaw shared by the policy family.

**What the attackers exploit** (`attackers.behaviour`: 8 games × 500 turns per pair, both seats; tables in
`results/m45c/final/report.md` §6):

| | attacker | target, in those games | target, in self-play |
|---|---|---|---|
| Trade (of lane behaviours) | **0.31** | 0.24 | 0.26 |
| Safe (of lane behaviours) | 0.19 | **0.31** | 0.27 |
| Shove mass | 0.26 | 0.32 | 0.33 |
| Thin / Slow Push mass | 0.23 / 0.23 | 0.19 / 0.19 | 0.19 / 0.19 |
| trade attempts / connected, per 1000 turns | **272 / 184** | 205 / 129 | |
| freezes per 1000 turns / freezer's denial share | **4.6 / 0.78** | 3.2 / 0.65 | |
| deaths per 1000 turns | 6.9 | **9.3** | |
| lead per 100 turns | +1.32 | −1.32 | |

**The attackers win through target-tuned aggression and freezing.** These are the two strategies the self-play
policies use least, because their own self-play opponents trade back and break freezes (§5.2–§5.3).
- **The attacker** Trades about a third more and connects 40% more trades. It holds more freezes, guards them with
  Trades rather than Safe, and keeps fighting at low HP (Trade 17%, Safe 62%; the target 6% / 80%).
- **The target answers** with more Safe than it plays in self-play, and dies more.
- **The policies' concession under pressure** (§5.3, §2.3) is exactly what a specialist can farm. Self-play never
  presents that opponent.

**This confirms §9's main gap from a second, independent direction.** The policies are close to a best response to
their own training distribution, but not to a deliberately aggressive, adapted opponent.

All of these are **empirical lower bounds**, limited by the attackers' budget, network and optimizer. They are not
game-theoretic exploitability.

## 8. Representative trajectories

**What was exported.** 39 games of the six 16.384M finals, found by M4.4b's state-defined search (`m45c/traces.py`)
in self-play from s0 (24 games × 460 turns per final) and from the viewer's neutral start with a ×1.76 power gap
(12 × 300). For each final and situation, the best-scoring game was exported to `results/m45c/traces/`, with M4.5
action labels and the `m45_calibrated` stamp.

**Every export was verified** by replaying it through the viewer's own session code: equal actions and equal payoff
on every turn.

| situation (state-defined) | finals with an example | what to look for |
|---|---|---|
| freeze denies: a freeze held ≥ 4 turns in which the freezer out-earns the frozen-on side | 6 / 6 | the freezer Thins and holds; the frozen-on side loses farm and XP until it Shoves or resets |
| freeze fails: the freezer does not out-earn | 6 / 6 | the frozen-on side breaks quickly, or the freezer's own wave crashes |
| freeze broken by a Shove from the frozen-on side | 6 / 6 | the break the policies use in 33–52% of frozen-on turns |
| Slow Push lands: ≥ 3 Slow Push turns followed by a crash of ≥ 12 HP | 6 / 6 | the stacked crash and the plate / Pressure that follows |
| Ward, then siege Pressure within three turns | **3 / 6** | rare: the policies Ward, but seldom immediately before extended Pressure |
| busy tower dive | 6 / 6 | dives into a tower busy with a wave |
| weak side under a credible threat (×1.76 gap) | 6 / 6 | the weaker champion yields or steps back near its tower |

Open them in the debugger:
1. `streamlit run lanegame/viewer/app.py`;
2. choose the mode "Recorded trajectory";
3. set "Folder of recordings" to `results/m45c/traces`.

The labels name the turns of the situation.

---

## 9. Research conclusions

**The question:** can PPO self-play independently discover economically meaningful League-inspired laning
strategies, without explicit strategy shaping?

**Answer: yes, for the strategies this model's economics reward. The policies did not adopt the ones it does not
reward, and that restraint is itself correct.**

The reward was the zero-sum discounted gold lead and nothing else: no strategy-specific reward, no curriculum, no
demonstrations, no scripted opponents in training. Every conclusion below holds in all six independent seeds unless
stated, and the replicated ones hold at both budgets.

The evidence comes in four kinds:
- **observed behaviour (O):** what the networks do;
- **counterfactual evidence (C):** paired, controlled alternatives from the same states;
- **robustness (R):** across opponents, seeds and budgets;
- **limitations (L).**

| strategy | verdict | evidence |
|---|---|---|
| **conditional wave management** | **discovered** | O: Shove is 33% of the mass, conditioned on state (47–56% with the enemy away, 33–52% when frozen on, 18–26% at levels 1–5). C: in wave-contact states no fixed wave action matches the policy's choice. One turn of Last Hit / Thin / Slow Push / Shove costs −0.25 / −0.23 / −0.19 / −0.11; Thin ×4 costs −0.61, Shove ×4 −0.35. Shove has the best *immediate* gold (+0.45) but is the modal action in only 22% of these states. R: every seed, both protocols. **Shove is not dominant: it is used where it pays** |
| **freezing, and answering freezes** | **discovered as a conditional tool** | O: freezes are rare (1.9–3.8 per 1000 turns) but deny in 60–78% of episodes (mean 69%). C: a sustained freeze pays only against an opponent who backs off (+0.56 natural, +1.86 constructed, 6/6) and *costs* against the learned policies (−0.20), which break it. Frozen on, breaking once is optimal: accepting costs −0.2 to −0.4 against every freezer style, over-shoving −0.3 to −0.5. R: replicated at 8.192M. **Rarity is the right response to opponents who break freezes** |
| **selective Trading under minion fire** | **discovered** | O: Trade at high HP 31%, Safe at low HP 76%; the modal action Trades into a heavier wave in only 15% of such states. C: forcing a Trade there costs −0.34 (−0.18 to −0.30 against other opponents); the exact one-turn cost is −0.38 and is not repaid later. Constructed trade bench: single trades are near neutral (±0.2), as M4.5b found |
| **threat-based zoning** | **conceding is learned; active denial is not, and against learned play it does not pay** | O: in an initialized ×1.76 gap the weaker side yields, retreats and stays near its tower. The stronger side moves forward (position +2.0) but denies only 0.014 gold per turn. C: sustained bullying costs −0.2 to −1.0 against the learned policies even when ×1.76 stronger, and −1.4 to −1.8 without the advantage. It pays (+1.28) only against a scripted opponent who concedes when threatened. L: natural power gaps are about 0.1% of decisions |
| **Vision (Ward)** | **used rarely, correctly priced** | O: about 2% of decisions, 16–27 per 1000 turns, varying by seed. C: a forced Ward where the policy did not choose one costs −0.3 to −0.6 against every opponent, including the punishing `g_bully`. The immediate cost (−0.72: the walk to W) is only partly repaid by turn 50. Ward-then-Pressure play is rare (3 of 6 finals have an example). **Low Ward use is economically justified on this model** |
| **Recall timing** | **discovered** | O: Recall 28% at HP < 35%, 1% at HP ≥ 70%; most completed Recalls are buying resets at about 61% HP. C: staying in lane at low HP costs −0.89 (−1.23 against the modal opponent); stepping back instead of resetting −0.33; recalling now ≈ the policy's choice. R: replicated at 8.192M (−0.96 for staying). L: 41% of attempts are interrupted, a guessing game the counterfactuals do not show to be a mistake |
| **early-game play** (levels ≤ 10, about 10% of experience) | **sensible but less certain** | C: the same orderings as later (alternatives −0.03 to −0.22, intervals ±0.2). Early-game Shove fell with more training (−0.06). L: these states are under-represented in training, and their validation failed most often (C trade early 5/6). The 150-turn re-runs confirmed the 50-turn conclusions there too |

**Policy strength and robustness.**
- **Against fixed opponents.** The six 16.384M finals beat all nine hand-designed strategies by +2.6 to +7.5,
  including M4.5b's restricted-game equilibria (+3.3), and the OOD M4.4b network by +3.2.
- **Over training.** They beat their own past, the 8.192M finals (+1.05) and the pilot (+2.3), with no self-play
  cycling. The six seeds are interchangeable.
- **Not converged.** +0.36 against their 12.3M checkpoint.

- **Exploitability.** A learned specialist at a quarter of the target's budget beats each final by +0.7 to +1.8
  (+1.0 on average, 6/6 significant). It does so through target-tuned aggression and freezing, and its exploits do
  not transfer to the other finals (§7).

**What did not emerge, and why.**
- **No strategy is learned "for its own sake".**
  - The policies do not freeze, ward or bully more than their opponents make worthwhile.
  - Where a scripted opponent concedes (backs off a freeze, retreats from a threat), the policies leave value on the
    table: +0.56 / +1.86 for a sustained freeze, +1.28 for bullying.
  - That is expected of self-play without opponent modelling. A self-play policy best-responds to *its own* opponent
    distribution, which never concedes like that.
  - **This is the main exploitation gap of the current approach, and it is a research direction, not a defect
    (§10).**
- **The unusual findings named in the brief, and what each turned out to be:**

| unusual finding | conclusion |
|---|---|
| Shove continuing to dominate frequencies | conditional use; immediate-gold-maximizing Shove would be worse |
| freezes successful but rare | rational against freeze-breaking opponents |
| Ward rarely used | the forced Ward loses money on this model |
| the weaker side retreats without denial | the stronger side's denial does not pay against learned trading-back |
| high interrupted-Recall rate | a guessing game, not shown to be a mistake |
| regressions after more training | none |
| differences between seeds | quantitative, mostly in the rare behaviours (Ward, freeze frequency) |

**Exploratory versus pre-specified.**
- **Pre-specified:**
  - the training campaign, the extension criterion and the budget comparison design;
  - the battery's design, horizons, validation rule and exceptions;
  - the attackers' protocol and budget.
- **Exploratory:**
  - the cross-lineage midpoint test, added after the 8.192M head-to-head;
  - the per-cell breakdown of constructed families;
  - the mechanics' "modal = candidate" shares;
  - every interpretation of *why* a strategy does or does not pay.

## 10. Next steps (M4.6 and later)

### Research limitations: method and training, with the simulator unchanged

1. **Opponent-style exploitation (the main gap).** The policies leave +0.5 to +1.9 against conceding opponents.
   Candidates, within the project's OpenAI Five-style scope:
   - a training population mixing self-play with scripted conceding / passive / wary styles;
   - an opponent-conditioned policy, M5's adaptive-opponent work.

   Measure with this battery whether the freeze and bully plans' value against conceders is captured, and at what
   cost against learned play.
2. **Convergence.**
   - The 16.384M policies still improve (+0.36 against 12.3M, +1.05 per doubling from 8M to 16M).
   - A further doubling would show whether the gain saturates. It was not authorized here and needs its own approval.
   - A cheaper alternative: one seed at 32M as a probe.
3. **Exploitability at the protocol budget.** The attackers ran at a fixed 4,096,000, a quarter of the target's
   budget. An attacker at half (8,192,000) on two or three finals would test whether the lower bound is
   budget-limited.
4. **The battery's validation rule.** Apply §6.3 pooled over the six policies per family, or raise the per-policy
   subset to about 25% of states. At the current size it flagged 33% of checks with no bias present, and the 150-turn
   re-runs changed nothing.
5. **Training distribution.** Early levels (about 10%) and power gaps (about 0.1%) are rare in self-play. Seeding a
   share of training games from diverse start states (levels, investments) would train the states the battery had to
   construct. This is a training-design change, not a simulator change.
6. **The seat difference** (A ≈ +0.5 over B against the scripted suite, reversed in the pilot) is unexplained.
   Decompose it by opponent and chance stream.

### Simulator-design limitations: questions for design review, not tuning targets

1. **Vision economy.**
   - A Ward costs a turn of farm, and its benefit (gank protection, information the policy cannot otherwise use)
     rarely repays it within 50 turns.
   - If Vision is meant to be a meaningful laning skill, its payoff structure is the design question. Possible levers:
     longer-lived information, a reason to know the enemy's position, a gank model that responds to Vision more
     strongly.
   - Not a change to make results "look realistic". A question of what the game is meant to model.
2. **Freeze durability.**
   - A freeze is broken by a single Shove from the frozen-on side, at small cost. League freezes are durable because
     walking up to break one is dangerous.
   - In this model, positional aggro and trading back make the walk-up cheap.
   - Whether breaking a freeze should cost more is a design decision for M4.6.
3. **Zoning payoff.**
   - Combat advantage converts into resource denial only against conceding opponents.
   - The trade-back and minion-retaliation economics (M4.5a) make sustained bullying unprofitable against a
     competent opponent.
   - If threat-based zoning by a stronger champion is meant to emerge against competent play, the trade economics
     are the lever. Same caveat as above.
4. **Accepted M4.5b caveats** still apply, among them the mid-to-late minion share and Shove-centred scripted
   equilibria. Nothing in M4.5c reopens them.

### Assessment for the next research phase

**GO WITH CAVEATS.** M4.5c is complete, and its results are a sound basis for the next phase. That phase is either
M5's adaptive-opponent work or an M4.6 design review of the three simulator questions above.

**What supports GO:**
- All campaigns trained correctly and reproducibly.
- The evaluation followed the approved methodology, including its exceptions; its 50-turn default was validated on
  the trained policies.
- The strategic conclusions hold across six seeds, two budgets, several opponent protocols and both seats.

**The caveats, all documented above:**
1. The policies are not converged (+0.36 against their 12.3M checkpoint).
2. They are exploitable by target-tuned aggression (+1.0, at a quarter-budget attacker).
3. They leave value against conceding opponent styles they never meet in self-play.
4. Vision, freeze durability and zoning payoffs are weak on this model by design, which is a review question, not a
   tuning target.
5. The validation rule, at its subset size, is noisier than intended.

**Nothing in the results calls for changing the simulator to make behaviour look more realistic.** Any change to the
model must start from the design questions in §10, not from the learned frequencies.

---

## 11. Execution, corrections and reproduction

**Timeline (2026-10-15).**

| stage | time | notes |
|---|---|---|
| extension training (6 concurrent) | 01:31 → 03:04 | nothing else ran alongside it |
| checks, then the health and budget-comparison evaluation | 03:04 → 04:07 | 11 workers |
| behaviour, then the comprehensive standard evaluation, then traces | 04:07 → 05:03 | |
| strategic battery (primary + replication) | 05:03 → 06:40 | |
| exception 3 (150-turn re-runs) | 06:40 → 07:37 | |
| attackers (6 concurrent), with evaluation | 07:38 → 08:45 | nothing else ran alongside them |
| attacker behaviour, reports, M4.5c tests (38 passed), full regression (**1395 passed, 4 skipped, 2 xfailed**) | 08:45 → 08:54 | |

Every stage ran after the previous one ended (`results/m45c/ext/chain_*.log`), and no evaluation overlapped any
training.

**Corrections made during the campaign.** All are in analysis or bookkeeping code. None changed any simulation, seed,
design or result.

| issue | correction | effect on results |
|---|---|---|
| a 150-turn re-run (exception 3) would have overwritten a policy's exact-mechanics file with the re-run families only | the re-run skips the mechanics stage, which does not depend on the horizon | none; caught before the re-runs started |
| the mechanics summary counted states where a candidate's code was illegal (it falls back to the policy's action) as "the modal action equals the candidate", diluting the lead differences | restricted to states where the candidate's code is legal (Recall in frozen-on states: legal in 36%) | the summary only; the battery's rollouts were unaffected |
| the campaign report failed on constructed cells re-run at 150 turns (a missing column) | missing columns print n/a | report generation only |
| the battery figure titles named the 150-turn horizon while plotting 50-turn values | titles state the plotted statistic | figure labels only |
| the two battery reports wrote the same summary file name | summaries are named per report | bookkeeping |
| raw battery jobs (about 9,000 files, 388 MB) | stored under the git-ignored checkpoint tree; manifests and reports in `results/m45c/battery/` | storage only |

**Artifacts.**
- Under `results/m45c/` (small, generated):
  - `ext/`: the 16.384M records, health and comprehensive evaluation, `report.md`, figures (3.4 MB);
  - `final/`: the campaign report, `summary.json`, figures (1.5 MB);
  - `battery/`: reports, summaries, manifests (1.8 MB);
  - `attackers/`: the attacker records and behaviour (0.5 MB);
  - `traces/`: 39 viewer recordings (1.6 MB).
- Under `results/rl/checkpoints/m45c/` (git-ignored):
  - `ext/` (636 MB): six runs × 640 snapshots;
  - `battery/` (388 MB): raw per-job results and mechanics;
  - `attack/ext_s*` (10 MB): six attackers.
- The 8.192M campaign's checkpoints and records are untouched (`full/`). One part of the comparison, the 8.192M
  finals' W = 1000 headline, is stored with the extension's record (`results/m45c/ext/health_eval.json`,
  `headline_main`).

**Reproduction.**

```bash
P="python -m lanegame.experiments.rl.m45c"
$P train_ext --confirm-m45c-ext                                             # 6 x 16,384,000 (93 min here)
$P train_check health_training --phase ext
$P health_eval --phase ext --confirm-m45c-comprehensive \
   --health-parts curve,h2h,history,cycle,cross,vs_main,anneal,historical,headline,headline_main,curve_rest,tournament
$P health_behavior --phase ext
$P traces --phase ext --confirm-m45c-comprehensive
$P battery --phase ext --confirm-m45c-battery                              # resumable
$P battery_extend --phase ext --confirm-m45c-battery                       # exception 3
$P battery_report --phase ext
$P attackers --phase ext --confirm-m45c-attackers
$P attackers_behaviour --phase ext
$P health_report --phase ext
$P campaign_report
python -m pytest tests/test_m45c_full.py tests/test_m45c_training.py tests/test_m45c_horizon.py
```

All stages are deterministic given their seeds (training bit-identical on re-run; evaluation on fixed paired seeds).
The evaluation and battery stages resume from their saved results.
