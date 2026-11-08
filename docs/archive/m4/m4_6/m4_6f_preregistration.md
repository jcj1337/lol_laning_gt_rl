# M4.6f — pre-registration of the separate-critic continuation pilot

*2026-10-16. Frozen (SHA-256 in `results/m46f/preregistration.json`) after the smoke test and before any pilot
training run. Already seen at freezing time: the initial-equivalence checks for seed 0 (all pass: action probabilities
and values bit-identical on 532 observations, no shared storage, gradient routing as intended), the pipeline checks of
a 10-update smoke run of both arms (all pass), their throughput, and the gradient probe of their first update (the
critic-only parameters carry 0.04 of a 0.73 pre-clip gradient norm). No evaluation of any trained network had been
run.*

## Question

Does giving the critic its own representation improve value learning and strategic behaviour, compared with
continuing the existing shared actor–critic PPO for the same additional budget? The four outcomes to distinguish:
1. more training alone is sufficient;
2. separating the critic helps;
3. better value estimates do not produce a meaningful behavioural benefit;
4. the experiment is inconclusive, or the intervention does not address the problem.

## Scope

A bounded learning pilot. No rule, observation, reward, start distribution or PPO setting changes. C1
(`m45_calibrated(invest_gain=3.24, invest_scale=64.0)`) is the environment because M4.6d–e diagnosed the critics
there; it stays a provisional opt-in and is not adopted. The library gains an opt-in network class
(`SeparateCriticNet`); `PolicyValueNet` and every existing checkpoint are unchanged.

## Arms (six matched seed pairs)

| item | control `shared` | intervention `separate` |
|---|---|---|
| network | `PolicyValueNet`: one 90 → 128 → 128 tanh trunk, linear policy and value heads | `SeparateCriticNet`: the same actor (trunk and policy head); the value head reads its own 90 → 128 → 128 tanh critic trunk |
| initialization | the weights of `start:s` = M4.6c `ext/C1:s` (8.192M C1 final), s = 0..5 | the same actor; critic trunk = a copy of the start's trunk, value head = the start's (`from_shared`) |
| everything else | identical: M4.6c's PPO (100% current-policy shared self-play, one policy for both seats; γ 0.99, λ 0.95, 128-turn segments, reward scale 0.1, value clip 0.5, vf_coef 0.5, 4 epochs, minibatch 256, clip 0.2, entropy 0.03, one global gradient-norm clip 0.5 over all parameters, 16 environments); the M4.6c mixed start distribution (archive start probability 0.532, archive games 150 turns, s0 games 512); a fresh Adam over all parameters (checkpoints hold no Adam moments) with lr 3·10⁻⁴ annealed linearly to 0 over the run (the M4.6c protocol: the annealed start is re-warmed equally in both arms); PPO seed 89,000 + s in both arms | |

Initial equivalence is verified for every seed before training (stage `verify`): probabilities and values of both
arms equal the start's exactly on representative observations; no shared storage; the policy loss reaches no critic
parameter and the value loss no actor parameter in `separate`; the value loss reaches the trunk in `shared`.

What a difference can and cannot mean: the intervention adds 28,160 critic parameters, removes the value gradient
from the actor's trunk, and lets Adam normalize the critic trunk's updates by the value gradient alone. A positive
result supports the practical change, not one of these mechanisms.

## Budget

- 4,096,000 agent transitions per run (1,000 updates), 12 runs, 6 concurrent (the two arms of a seed side by side).
  Measured throughput (M4.6c, 6 concurrent: about 2,150 transitions/s per run; the smoke test: separate arm about
  4% slower) gives about 35 minutes per run, 70 minutes in all.
- Six seed pairs instead of two: at this cost six pairs give 5 degrees of freedom for every between-arm interval
  (two pairs would give 1).
- Snapshots every 25,600 transitions; intermediate readouts at 2,048,000 (and the head-to-head curve at 1.024M,
  2.048M, 3.072M).

## Evaluation (C1; resources; γ 0.99; intervals are 95% t intervals over the six seed pairs, 5 df)

Every value label is the evaluated network playing itself, from the labelled seat, without bootstrap: the return its
own critic is trained to predict. The starting critics' labels are M4.6d's (same states, same rollout seeds).

### Value

- **V-calib** (primary value set): the M4.6d calibration set. 42 cells: 12 freeze (levels 3/5/8 × equal, edge,
  plaus, ×1.76; held 12), 12 lane (neutral and push × levels 4/8 × equal, edge, ×1.76, full HP), 18 M4.6a gap cells
  (×1.5, ×1.76), K 32 × H 300; plus the 16 natural lead / even states of start:s's own self-play, K 32 × H 500.
  Rollout seeds are M4.6d's (99,500 root), so labels pair rollout for rollout with the starting critic's. Finals: the
  full set. 2.048M snapshots: the cells only, K 16.
- **V-nat**: 32 s0 self-play games of 600 turns per network (finals and starts), labels every 10 turns from 5 to 295
  (G300, both seats).
- **V-probe**: ridge probes from the representation the value head reads (critic trunk, or the shared trunk) to the
  log duel ratio and the own HP fraction, on the M4.6e labelled observations of seed s, held-out R² on M4.6e's main
  test groups; the critic's response to single-feature changes at the three M4.6e base states.
- **V-train**: explained variance and value loss over training; the gradient probe (norms of actor and critic parts,
  the clip's coupling, the value term on the actor trunk).

### Behaviour (actual rollouts; never critic scores)

Paired families use the M4.6a paired engine (CRN: the chance stream and each seat's sampling uniforms are common to
the candidates). In each cell, seat A is played by each candidate `shared:s` and `separate:s` (the 4.096M finals of the
same seed) against seat B = each opponent of {`shared:s`, `separate:s`}: a 2 × 2 round robin per seed; the readout
d = mean over the two opponents of G(separate vs O) − G(shared vs O).

- **B1 head to head** (E1 from s0, paired seats): `separate:s` vs `shared:s`, 64 episodes, W 1000. Also each final
  vs `start:s` (64, W 1000), vs the other five starts (16 each, W 460), vs the scripted core suite (`g_farm`,
  `g_shove`, `g_bully`, `fz_guard_sh`; 16 each, W 460), and `separate` vs `shared` at 1.024M, 2.048M, 3.072M (32,
  W 460).
- **B2 strength use** (cells K 16 × H 150):
  - the M4.6c lead cells, early and mid stages × {neutral, a_push, b_push} × {equal, A+8, A+16, B+8, B+16};
  - the M4.6d lane cells, {neutral, push} × levels {4, 8} × {equal, edge, plaus = edge with B at 0.6 HP, hp = equal
    with B at 0.6 HP, brk = B one level up and +3 invested};
  - a large-gap diagnostic: the same lane cells with A ×1.76 (reported separately).
  - Conversion cells (A advantaged): A+8, A+16, edge, plaus, hp (24 cells). Defence cells (A disadvantaged): B+8,
    B+16, brk (16 cells).
  - **B2 = the mean of d over the 40 conversion and defence cells at G150.** Reported with it: conversion and defence
    separately, G50, the specificity contrast d(cell) − d(matching equal cell), and denial (the opponent's uncollected
    farm and XP lost).
- **B3 freezes**: the M4.6d freeze cells (held 12; levels 3/5/8 × equal, edge, plaus, hp, brk, ×1.76), the freezer
  in seat A and, mirrored, the breaker in seat A; K 32 × H 50 (M4.6d's decision window). Readouts: G50; turns the
  freeze is held, the breaker's lost farm and XP; deaths.
- **B4 weak side**: Recall windows (neutral lane, the weak side at its −1 with HP 0.5 or 0.35, the strong side equal,
  edge or ×1.76, levels 5 and 8: 12 states), the weak side in seat A (mirrored) and the strong side in seat A; K 32 ×
  H 100. Readouts: G50, G100; the weak side's deaths, Recalls, Safe and Yield turns; the first-turn action
  probabilities of each actor.
- **B5 natural states**: start:s's natural leader and trailer states (M4.6c; invested lead ≥ 4; 24 + 24) at K 16 × H
  150, and its favourable natural early freezes (M4.6d; 12; freezer and, mirrored, breaker) at K 16 × H 50.

## Primary outcomes and practical criteria (at the final evaluated budget)

Each between-arm statistic is classified as:
- **benefit**: mean ≥ +m and the interval excludes 0;
- **regression**: mean ≤ −m and the interval excludes 0;
- **no material difference**: the interval lies inside (−e, +e);
- **small**: the interval excludes 0 but |mean| < m;
- **inconclusive**: otherwise.

| statistic | definition | m | e |
|---|---|---|---|
| **V1 (primary value)** | edge + plaus premium recovery ratio (Σ predicted / Σ realized over the 10 contrasts × 2 seats, M4.6e's P1), separate − shared | 0.25, and the separate arm's ratio ≥ 0.5 | 0.15 |
| V2 (guard) | MAE against G500 on the natural states, separate − shared | regression if ≥ +0.10 with the interval excluding 0 | |
| V3 (diagnostic) | large + gap ratio, separate − shared | reported | |
| **B1 (primary behaviour)** | `separate:s` vs `shared:s`, W 1000 | 0.20 | 0.30 |
| **B2 (primary behaviour)** | strength use, G150 | 0.15 | 0.25 |
| guards | each final vs start:s (separate − shared); the scripted suite (mean over references) | regression if ≤ −0.20 with the interval excluding 0 | |

**Outcome mapping.**
- **2, separating the critic helps:** V1 benefit, B1 or B2 benefit, and no regression (B1, B2, V2, guards).
- **3, better values without a behavioural benefit:** V1 benefit, and B1 and B2 each "no material difference" or
  "small".
- **1, more training alone is sufficient:** V1 not a benefit while the shared arm's own ratio rises materially over
  its start (shared − start ≥ +0.25, the interval excluding 0), and B1, B2 not a benefit.
- **4, inconclusive or not addressing the problem:** V1 not a benefit and the separate arm's ratio < 0.5 (separation
  does not make the critic learn the premium in this budget); or V1 benefit with B1 / B2 inconclusive.
- A regression of the separate arm on B1, B2, V2 or a guard is reported as harm whatever else holds.

Secondary families (B3–B5, V-nat, V-probe, the curve) describe where any difference arises; they do not change the
classification. Absence of significance is not evidence of no effect: inconclusive results are reported as such.

## Conditional extension (decided once, on the 4.096M results)

Extend both arms of all six seeds to 8,192,000 (new runs from start:s with an 8.192M schedule, M4.6c's extension
protocol; every family re-run on the extension finals, which then become primary) if and only if all hold:
- (a) the separate critic is learning: V1's interval excludes 0 on the positive side, or the separate arm's ratio
  rose from 2.048M to 4.096M with the paired interval excluding 0;
- (b) behaviour is unresolved: neither B1 nor B2 is "benefit", "regression" or "no material difference", and the
  point estimate of B1 or B2 is positive;
- (c) health: no pre-registered stop condition in any run.

There is no further extension.

## Health and stop conditions

M4.5c's: non-finite statistics, approximate KL above 0.05 sustained over 10 updates, gradient norm above 100,
entropy below 0.3 nats, an illegal action sampled. Plus: the separate critic's explained variance below 0 over the
last 100 updates of a run.

## Seeds and files

Seeds 89,000–89,999 (training 89,000 + s; natural labels 89,200; head to head 89,300; cells 89,400; natural states
89,500; gradient probe 89,600; equivalence 89,700); the calibration reuses M4.6d's 99,500 (paired with the starting
critic's labels). Raw records under `results/rl/checkpoints/m46f/` (git-ignored); summaries, the generated tables
and this record under `results/m46f/`; the report `docs/m4/m4_6/m4_6f_report.md`.
