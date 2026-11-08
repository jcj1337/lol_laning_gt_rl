# M4.6f — Separate-critic continuation pilot

*2026-10-16. Status: complete, awaiting review. A bounded learning pilot. No rule, observation, reward, start
distribution, PPO setting or default changed, and no existing checkpoint was modified. The library gained an opt-in
network class (`SeparateCriticNet`); `PolicyValueNet` and every earlier checkpoint load and behave exactly as before.
C1 (`m45_calibrated(invest_gain=3.24, invest_scale=64.0)`) is the environment because M4.6d–e diagnosed the critics
there; it stays a provisional opt-in and is not adopted.*

*Protocol: [m4_6f_preregistration.md](m4_6f_preregistration.md), frozen after the smoke test and before any pilot run
(SHA-256 in `results/m46f/preregistration.json`, with two dated amendments, §1.4). Generated tables:
`results/m46f/report.md`. Previous stages: [M4.6c](m4_6c_report.md), [M4.6d](m4_6d_report.md),
[M4.6e](m4_6e_report.md). Nothing is staged or committed.*

**Terms.**
- A *resource* is the gold of one fresh wave; natural income is about 0.40 resources per turn. *G_h* is a seat's
  discounted (γ 0.99) earned-gold lead over h turns, without bootstrap.
- *Recovery ratio* (M4.6e's P1) = Σ predicted premium / Σ realized premium over paired M4.6d cells: the critic's
  prediction against its own policy's K 32 × 300-turn returns.
  - *edge*: one level and +3 invested (duel ×1.21–1.27);
  - *plaus*: edge with the opponent at 60% HP;
  - *large*: an investment ratio of 1.76 (duel ×3.1);
  - *gap*: M4.6a's ×1.5 and ×1.76 cells.
- *shared*: the existing network, whose actor and critic share one trunk (the control). *separate*: the same actor
  with its own critic trunk, copied from the shared trunk at the start. Both arms continue the same M4.6c policy
  (`start:s`) for the same budget, and one policy still plays both seats.
- Intervals are 95% t intervals over the six seed pairs (5 df); "6+/0−" counts seeds by sign.

## Summary

**Main conclusion.**
- **The critic learns strength once it has its own trunk.** The shared setup does not learn strength value at all
  with the same additional training. The improvement grows with budget and costs nothing in natural states.
- **The better critic did not produce better strategic behaviour.** In rollouts, the separate-critic actor converts
  strength advantages, defends, freezes and plays weak-side Recall windows no better than the control.
- **Classification.** By the pre-registered rules this is **outcome 4, not resolved**. The critic's recovery stayed
  just below both registered thresholds: +0.24 against +0.25, and a ratio of 0.45 against 0.5. Substantively the
  result is closest to **outcome 3, better values without a meaningful behavioural benefit**.
- **Outcome 1 is ruled out for the critic.** More training of the shared network leaves it exactly as blind.

**Key numbers.** These are the 8.192M extension, which is primary; the pre-registered rule fired at 4.096M.

| | start | shared | separate | separate − shared |
|---|---|---|---|---|
| edge + plaus recovery (V1) | 0.17 | 0.21 | 0.45 | **+0.24 ± 0.20** (6+/0−): small |
| edge recovery (strength only) | 0.01 | 0.02 | 0.31 | **+0.29 ± 0.12** (6+/0−) |
| large + gap recovery (V3) | 0.02 | 0.02 | 0.32 | **+0.30 ± 0.03** (6+/0−) |
| predicted ×1.76 premium (realized 5.7) | 0.14 | 0.14 | 1.66 | **+1.52 ± 0.09** |
| log duel decodable from the value head's input (R²) | 0.17 | 0.17 | 0.64 | **+0.48 ± 0.04** |
| natural-state MAE against G500 (V2) | 0.42 | 0.52 | 0.37 | −0.14 ± 0.20: no regression |
| B1 head to head from s0 (W 1000) | | | | +0.06 ± 0.23: no material difference |
| B1 powered, 320 episodes per seed (amendment 1) | | | | **+0.13 ± 0.09** (6+/0−): small |
| B2 strength use (40 cells, G150) | | | | +0.01 ± 0.11: no material difference |

- **Value.**
  - The separate critic recovers about a third of the premium of pure strength edges and of large gaps; the shared
    critic recovers 2%. It was at 0.18 (edge) and 0.22 (large + gap) at 4.096M and was still rising.
  - It has learned investment: +32 invested is worth +1.2 resources to it, against 0.09 to the shared critic.
  - It stays nearly blind to level: +0.06 per level.
  - Its natural-state calibration is as good or better: R² 0.82 vs 0.71, and single-rollout R² +0.07 ± 0.05 (6/6).
- **Behaviour.** Nowhere is there a difference beyond noise:
  - strength conversion −0.01 ± 0.15, defence +0.03 ± 0.24;
  - the freezer in favourable freezes +0.05 ± 0.23;
  - the weak side at Recall windows −0.03 ± 0.24;
  - natural leader / trailer states +0.07 ± 0.09 / +0.01 ± 0.16.
- **A small general edge, not established as a benefit.**
  - At 8.192M the separate arm beats the control by +0.13 ± 0.09 in the powered head to head. That statistic is an
    amendment, and it is below the 0.20 materiality.
  - Against the other seeds' starts it scores +0.34 ± 0.29 more than the control.
  - Neither was present at 4.096M (+0.07 ± 0.15 and −0.03 ± 0.07). The registered B1 shows no material difference at
    either budget.
- **No regression and no coupling.**
  - Every health check passed.
  - The global gradient clip shrank the actor's step by only 0.02% more in the separate arm.
  - Throughput cost: −4%.

**What it narrows.**
- M4.6e left open which part of PPO value training failed. In this setup, sharing the trunk with the actor is enough
  to keep the critic strength-blind, and removing the sharing is enough to start fixing it.
- Why the better values do not reach the actor is consistent with M4.6e's ranking result. The candidates of one
  decision start from the same strength, so a strength-dependent value error is nearly the same for every candidate
  and cancels in the advantage.

**Recommendation: accept the limitation, close this investigation, and proceed to M4 closeout.**
- At closeout, record the separate critic as a verified, low-cost option for M5: better values, no measured
  regression.
- It is not a fix for strategic behaviour, and it should not be adopted on value accuracy alone.
- No further critic pilot.

---

## 1. What was run

### 1.1 The change and the matched arms

| item | control `shared` | intervention `separate` |
|---|---|---|
| network | `PolicyValueNet`: one 90 → 128 → 128 tanh trunk, linear policy (61) and value heads; 36,158 parameters | `SeparateCriticNet`: the same actor (trunk and policy head, same parameter names); the value head reads its own 90 → 128 → 128 tanh critic trunk; 64,318 parameters |
| initialization | the weights of `start:s` = M4.6c `ext/C1:s` (the 8.192M C1 final), s = 0..5 | `SeparateCriticNet.from_shared(start:s)`: the start's actor; critic trunk = a copy of the start's trunk; value head = the start's |
| shared by both | M4.6c's PPO unchanged: 100% current-policy shared self-play (one policy plays both seats), γ 0.99, λ 0.95, 128-turn segments, reward scale 0.1, value clip 0.5, vf_coef 0.5, 4 epochs, minibatch 256, clip 0.2, entropy 0.03, one global gradient-norm clip 0.5 over all parameters, 16 environments, snapshots every 25,600 transitions; M4.6c's mixed start distribution (archive games: 24.3–25.7% of transitions, identical in both arms of a seed); PPO seed 89,000 + s | |
| optimizer | a fresh Adam over all parameters and lr 3·10⁻⁴ annealed linearly to 0 over the run, in both arms. Checkpoints store no Adam moments; the start's schedule had annealed to 0, so continuing it would have meant no learning. This is the M4.6c protocol (it re-warms the learning rate equally in both arms) | |

**Library change** (opt-in, backward compatible):
- `lanegame/rl/network.py`: `SeparateCriticNet` (with `from_shared`, `actor_parameters`, `critic_parameters`),
  `build_network(spec)`, and `value_features` on both classes (the representation the value head reads).
- `lanegame/rl/agent.py`: `load_checkpoint` rebuilds the class from the stored spec; a spec without a `critic` entry
  is a `PolicyValueNet`, so every earlier checkpoint loads unchanged.
- `lanegame/rl/shared.py`: run.json records the network's class (the string is unchanged for `PolicyValueNet`).
- `ppo_update`, the collector and GAE are untouched.

**What a difference can and cannot mean.** The intervention removes the value gradient from the actor's trunk,
adds 28,160 critic parameters, and lets Adam scale the critic trunk's updates by the value gradient alone. In the
shared trunk the value term is under 1% of the gradient and Adam's per-parameter scaling is set by the policy term,
so in the control the critic's representation moves only as fast as the policy objective lets it. A positive result
supports the practical change, not one of these mechanisms.

### 1.2 Budget, throughput and runtime

| phase | runs | budget per run | throughput per run (6 concurrent) | wall clock |
|---|---|---|---|---|
| main (pre-registered) | 6 seeds × 2 arms | 4,096,000 agent transitions (1,000 updates) | shared 2,599, separate 2,485 transitions/s (−4.4% ± 0.8%) | 55 min (26–28 min per run, two waves) |
| extension (fired by the pre-registered rule, §5) | 6 × 2 | 8,192,000 (2,000 updates; new runs from start:s with an 8.192M schedule) | shared 2,586, separate 2,482 transitions/s (−4.0%) | 1 h 50 min (53–55 min per run, two waves) |

Six seed pairs instead of the suggested two: at the measured cost six pairs gave 5 degrees of freedom for every
between-arm interval (two would give 1) for about an hour of training. Peak memory per run: about 650 MB in both arms.

### 1.3 Evaluation

Every value label is the evaluated network playing itself without bootstrap (the return its own critic is trained to
predict), in resources, γ 0.99. Behaviour is measured only by rollouts. In the paired families each cell is a 2 × 2
round robin within a seed: seat A is played by each candidate (`shared:s`, `separate:s`) against seat B played by
each of them, on common random numbers; the readout is the mean over the two opponents of
G(separate vs O) − G(shared vs O). Intervals are 95% t intervals over the six seed pairs (5 df); seeds are the
independent units, and rollouts within a seed only make that seed's estimate precise.

| family | content | budget (4.096M) |
|---|---|---|
| V-calib | the M4.6d calibration set: 42 cells (freeze, lane, gap) × K 32 × H 300 and 16 natural lead / even states × K 32 × H 500; the M4.6d rollout seeds, so labels pair with the starting critic's M4.6d labels; 2.048M snapshots: cells, K 16 | 7.9M + 2.4M turns |
| V-nat | 32 s0 self-play games × 600 turns per network (finals and starts), G300 labels every 10 turns | 0.35M |
| V-probe | ridge probes of the value head's input on M4.6e's labelled observations; critic responses to single-feature changes | — |
| B1 | E1 head to head: separate vs shared (64 episodes, W 1000); each vs start:s (64, W 1000), vs the other five starts (16 each, W 460), vs four scripted references (16 each, W 460); curve at 1.024 / 2.048 / 3.072M | 4.9M |
| B2 | 54 cells (M4.6c lead cells early / mid; M4.6d lane cells; a ×1.76 diagnostic), K 16 × H 150 | 3.1M |
| B3 | 18 M4.6d freeze cells, freezer and breaker in seat A, K 32 × H 50 | 1.4M |
| B4 | 12 Recall windows, weak and strong side in seat A, K 32 × H 100; first-turn action probabilities | 1.8M |
| B5 | start:s's natural leader / trailer states (24 + 24) and favourable natural freezes (12, both roles), K 16 | 4.1M |

### 1.4 Deviations and amendments

| item | what happened |
|---|---|
| amendment 1 (post hoc, before any extension result) | B1's per-seed measurement error (stderr 0.23 from 64 episodes) is about its between-seed SD (0.26): B1 is limited by episodes, not seeds. A powered B1 adds 256 paired-seat episodes per seed (new root 89,310) for both finals; the registered 64-episode B1 stays the classified statistic |
| amendment 2 (exploratory, before any extension result) | the separate critic stopped penalizing unspent gold (+4 gold: −0.10 → +0.01), the one change that could alter Recall and spending behaviour; a natural self-play economy profile (16 games × 600 turns per network) was added, descriptive only |
| outcome label wording | the analysis code's label for the pre-registered outcome-4 branch "V1 not a benefit and the separate ratio < 0.5" first read "separation does not make the critic learn the premium". That overstated it (the critic learns part of it, §4); the label now reads "recovers the premium only partially". The rule is unchanged |
| classification precedence | the registered classes overlap when an interval excludes 0 inside the equivalence margin; such a result is classed "small" (stated in the code before any classification was read) |
| a test fixture | orthogonal initialization depends on torch's thread count (a pre-existing property); the new tests set one thread before building networks, as `train_shared` does. The pilot's networks are copied from checkpoints, so it is unaffected |

## 2. Verification

**Initial equivalence (stage `verify`, all six seeds, 60 checks, all pass).** On 532 representative observations per
seed (s0 self-play, 64 archive states, the 42 M4.6d cells, both seats):
- the action probabilities and values of both arms' initial networks equal the start's **exactly** (maximum
  absolute difference 0);
- the separate arm's actor parameters have the start's digest; actor and critic share no storage; perturbing the
  critic trunk leaves every action probability unchanged;
- gradient routing: in `separate` the policy loss reaches no critic parameter and the value loss no actor
  parameter; in `shared` the value loss reaches the trunk;
- a checkpoint round trip restores the class and the digest; the start checkpoints still load as `PolicyValueNet`.

**Pipeline (smoke, then every run).** Both arms' first rollouts are identical (same initial policy, same seeds);
the arms diverge after the first update; a re-run of the separate arm is bit-identical. All 156 training checks pass
(budget exact, no termination, critic bounded, seats trained equally, no illegal action, initial actor = start,
archive share 0.20–0.30, no stop condition, explained variance > 0 over the last 100 updates). Maximum 10-update
KL 0.007, minimum entropy 1.53 nats, maximum gradient norm 1.69.

**The global gradient clip does not couple the arms.** The clip acts on the norm over all parameters, and it is
active on most updates (pre-clip norm 0.75 against 0.5). The gradient probe (every 25 updates, four minibatches)
shows the critic-only parameters carry 0.2–0.3% of the squared norm in both arms; the factor by which they shrink the
actor's clipped step is 0.9988 (shared) and 0.9984 (separate).

**Tests.** `tests/test_m46f.py` (12): exact equivalence of `from_shared`, untouched global RNG, no shared storage,
gradient routing in both classes, checkpoint round trips and backward compatibility, a reproducible `train_shared` run
with the new class, that with vf_coef 0 the separate critic never moves while the shared network's values do, that
the gradient probe is read only (bit-identical training with and without it), the classification rules, the cell
builders, and the real-start equivalence. The full suite on the final code: 1,444 passed, 4 skipped, 2 xfailed
(7 min 46 s), including the legacy digest pins.

## 3. Training

| quantity (4.096M runs; mean over seeds) | shared | separate | separate − shared |
|---|---|---|---|
| explained variance, first 10 updates | 0.451 | 0.449 | −0.00 ± 0.04 |
| explained variance, last 100 updates | 0.446 | 0.498 | **+0.05 ± 0.05** (6+/0−) |
| value loss, last 100 updates (resources²) | 0.64 | 0.61 | −0.03 ± 0.12 |
| entropy, last 100 updates | 1.72 | 1.72 | +0.00 ± 0.05 |
| mean approximate KL / clip fraction | 0.0038 / 0.044 | 0.0038 / 0.044 | — |
| value term on the actor trunk (gradient norm) | 0.0047 | 0 | — |
| value term on the critic trunk | — | 0.0064 | — |

Explained variance (against the run's own λ-returns) over ten blocks of updates: shared 0.435, 0.420, 0.428, 0.430,
0.428, 0.430, 0.438, 0.430, 0.453, 0.446; separate 0.445, 0.460, 0.475, 0.470, 0.503, 0.477, 0.481, 0.481, 0.498, 0.498.
The separate critic improves steadily on its training targets; the shared one does not.

Both continuations moved their policies about equally far from the start: KL(start ‖ final) on the start's natural
decisions 0.23 (shared) and 0.26 (separate), and the two arms' finals differ from each other by a total-variation
distance of 0.29 per decision. The re-warmed learning rate moves the policy substantially in both arms.

## 4. Results at 4.096M (the pre-registered main phase)

### 4.1 Value

| readout (M4.6d set, own-policy labels) | start | shared | separate | separate − shared | shared − start |
|---|---|---|---|---|---|
| **V1: edge + plaus recovery ratio** | 0.17 | 0.21 | 0.33 | **+0.12 ± 0.07** (6+/0−) | +0.04 ± 0.08 |
| edge ratio (strength only) | 0.01 | 0.01 | 0.18 | **+0.17 ± 0.06** (6+/0−) | −0.00 ± 0.01 |
| plaus ratio (edge with an HP gap) | 0.38 | 0.72 | 0.65 | −0.06 ± 0.58 | +0.33 ± 0.43 |
| edge + plaus: predicted / realized premium (resources) | 0.15 / 0.96 | 0.19 / 0.89 | 0.30 / 0.90 | **+0.11 ± 0.04** (predicted) | **+0.04 ± 0.03** |
| **V3: large + gap ratio** | 0.02 | 0.03 | 0.22 | **+0.20 ± 0.05** (6+/0−) | +0.00 ± 0.01 |
| ×1.76: predicted / realized premium | 0.14 / 5.77 | 0.16 / 5.93 | 1.14 / 5.59 | **+0.98 ± 0.25** (predicted) | +0.01 ± 0.04 |
| seat-A bias: freeze / lane / gap cells | −1.99 / −1.66 / −4.64 | −2.04 / −1.71 / −4.70 | −1.51 / −1.28 / −3.58 | **+0.53**, **+0.43**, **+1.12** | ≈ 0 |
| **V2: natural MAE (G500 states)** | 0.42 | 0.50 | 0.49 | −0.01 ± 0.20 | +0.08 ± 0.13 |
| natural slope / R² | 1.01 / 0.77 | 1.02 / 0.73 | 1.02 / 0.76 | −0.01 / +0.03 | |

- **The separate critic learns strength value, partially, in every seed.** It recovers about a fifth of the premium
  of a pure strength edge and of large gaps, where the shared critic recovers 1–3%. The shared critic, with the same
  budget, does not move on strength at all.
- **It is short of the pre-registered threshold.** V1 = +0.12 against a materiality of +0.25, and the separate ratio
  is 0.33 against the 0.5 floor: "small".
- **No cost in natural states**: MAE, bias, slope and R² are unchanged (V2: no regression). On single-rollout natural
  labels (V-nat) the arms are also equal (MAE 2.20 vs 2.28, R² 0.14 vs 0.12; dominated by return noise).
- **Still rising at 4.096M.** From the 2.048M snapshots (K 16) to the finals the separate critic's edge ratio went
  0.12 → 0.19, large + gap 0.16 → 0.22, the predicted ×1.76 premium 0.81 → 1.14 resources (the shared critic flat).
  The K 16 edge + plaus ratio itself is unstable at 2.048M (one seed's realized denominator is near 0), so the trend
  is read on these components.

**Representation and responses.** A ridge probe of the value head's input recovers the log duel ratio with held-out
R² 0.47 in the separate critic against 0.17 in the shared trunk (+0.30 ± 0.06, 6/6); the actor's trunk stays at 0.17
in both arms; HP stays at R² 0.90–0.92 everywhere. A ×1.76 contrast moves the value representation 0.17 of its natural
spread (shared 0.11). The critic's response at neutral L8 (resources):

| change of seat A | start | shared | separate |
|---|---|---|---|
| +8 invested | +0.03 | +0.04 | +0.26 |
| +32 invested | +0.10 | +0.11 | +0.78 |
| +4 unspent gold | −0.10 | −0.10 | +0.01 |
| +1 level | +0.00 | +0.01 | +0.03 |
| own HP to 60% | −0.48 | −0.50 | −0.50 |
| enemy HP to 60% | +0.49 | +0.59 | +0.46 |

The separate critic has learned investment (and no longer penalizes unspent gold) but is still nearly blind to
level.

### 4.2 Behaviour

| statistic | separate − shared | class |
|---|---|---|
| **B1** head to head from s0 (64 episodes, W 1000) | +0.10 ± 0.27 (4+/2−) | inconclusive |
| B1 powered (amendment 1; 320 episodes pooled) | +0.07 ± 0.15 (4+/2−) | no material difference (not used for the outcome) |
| curve: 1.024M / 2.048M / 3.072M (W 460) | −0.06 ± 0.51 / +0.42 ± 0.60 / −0.12 ± 0.49 | |
| guard: vs start:s, separate − shared | −0.02 ± 0.44 | no regression |
| vs the other five starts, separate − shared | −0.03 ± 0.07 (3+/3−) | |
| guard: scripted core suite, separate − shared | +0.00 ± 0.59 | no regression |
| **B2** strength use, G150 (40 conversion and defence cells) | **+0.14 ± 0.09** (6+/0−) | small |
| conversion cells / defence cells / equal cells, G150 | +0.09 ± 0.20 / **+0.23 ± 0.20** / **+0.23 ± 0.21** | |
| conversion − matching equal cell, G50 / G150 | **−0.12 ± 0.06** (0+/6−) / −0.16 ± 0.19 (0+/6−) | |
| defence − matching equal cell, G150 | +0.02 ± 0.29 | |
| large ×1.76 cells (diagnostic), G150 | −0.38 ± 0.43 (1+/5−) | |
| denial in conversion cells (opponent's lost farm / XP, 150 turns) | −0.00 ± 0.91 / +0.6 ± 7.8 (levels 18.3 / 46.1) | |

- **No behavioural benefit is established.**
  - Head to head, the separate arm is +0.10 ± 0.27.
  - With five times the episodes (amendment 1) it is +0.07 ± 0.15. That interval, −0.08 to +0.23, includes 0 and only
    just includes +0.2.
  - Against fixed opponents (the start, the other seeds' starts, scripted strategies) the two arms are
    indistinguishable; the other-starts comparison is tight (−0.03 ± 0.07).
- **B2 is positive in all six seeds but is general, not strength-specific.** The separate arm gains as much in
  equal-strength cells (+0.23) as in defence cells (+0.23), and *less* in conversion cells: converting an advantage
  beyond what it gains in the matching equal cell is negative in every seed (−0.12 ± 0.06 at 50 turns). The critic
  that values strength more did not produce an actor that converts strength better.
- **Decision families (B3–B5): nothing outside noise.** Freezer in favourable freezes +0.06 ± 0.13 (G50), breaker
  +0.09 ± 0.17; weak side at Recall windows +0.05 ± 0.37, strong side −0.11 ± 0.13 (1+/5−); natural leader / trailer
  states −0.04 ± 0.07 / +0.01 ± 0.12 (G50). The weak side's first-turn mass shifts slightly from Yield to Safe
  (Yield 0.63 → 0.58 → 0.55, Safe 0.19 → 0.21 → 0.25 for start → shared → separate), the direction M4.6d's critic
  preferred, without a measurable change in its rollout value.

**Natural-play economy (amendment 2, exploratory).** Each final and each start playing itself, 16 games × 600
turns on common seeds. Nothing moved: Recalls 42.3 vs 43.3 per 1,000 turns (+1.0 ± 6.7), unspent gold at a Recall
7.81 vs 7.61 (−0.21 ± 0.67), mean unspent gold 5.12 vs 5.01, invested gold at turn 300 113.8 vs 114.0, collection
rate 0.752 vs 0.746, trades connected 144 vs 151 per 1,000 (+7 ± 19), deaths 6.8 vs 6.8 per 1,000 (shared vs
separate). The critic's changed valuation of unspent gold did not change how the actor Recalls and spends.

### 4.3 Scorecard and the extension rule

At 4.096M: V1 small, V2 no regression, B1 inconclusive, B2 small, no guard regression. By the pre-registered mapping
this is outcome 4 (V1 not a benefit, the separate ratio below 0.5). The extension rule fired: (a) V1's interval
excludes 0; (b) neither B1 nor B2 is resolved and both point estimates are positive; (c) every health check passes.

## 5. Results at 8.192M (the extension; primary)

**Protocol.** Twelve new runs from the same starts, each with an 8.192M schedule (M4.6c's extension protocol).
Every family was re-run on their finals.

**Health.**
- All 156 training checks pass: maximum 10-update KL 0.007, minimum entropy 1.51 nats, maximum gradient norm 1.96.
- The critic-only parameters carry 0.2–0.3% of the squared gradient norm in both arms (clip shrink 0.9988 vs 0.9985).

### 5.1 Value

| readout | start | shared | separate | separate − shared | shared − start |
|---|---|---|---|---|---|
| **V1: edge + plaus recovery** | 0.17 | 0.21 | 0.45 | **+0.24 ± 0.20** (6+/0−) | +0.03 ± 0.15 |
| edge / plaus recovery | 0.01 / 0.38 | 0.02 / 0.47 | 0.31 / 0.71 | **+0.29 ± 0.12** / **+0.23 ± 0.23** | +0.01 / +0.09 |
| edge + plaus: predicted / realized premium | 0.15 / 0.96 | 0.16 / 0.99 | 0.38 / 0.94 | **+0.22 ± 0.05** (predicted) | +0.01 ± 0.03 |
| **V3: large + gap recovery** | 0.02 | 0.02 | 0.32 | **+0.30 ± 0.03** (6+/0−) | −0.00 ± 0.01 |
| ×1.76: predicted / realized | 0.14 / 5.77 | 0.14 / 5.78 | 1.66 / 5.68 | **+1.52 ± 0.09** (predicted) | −0.00 ± 0.04 |
| seat-A bias: freeze / lane / gap | −1.99 / −1.66 / −4.64 | −1.95 / −1.67 / −4.70 | −1.36 / −1.02 / −3.08 | **+0.59**, **+0.66**, **+1.62** | ≈ 0 |
| **V2: natural MAE (G500)** | 0.42 | 0.52 | 0.37 | −0.14 ± 0.20 | +0.10 ± 0.17 |
| natural slope / R² | 1.01 / 0.77 | 0.95 / 0.71 | 0.97 / 0.82 | +0.02 / +0.10 ± 0.14 | |
| single-rollout natural labels: MAE / R² | 2.23 / 0.11 | 2.34 / 0.10 | 2.09 / 0.17 | −0.24 ± 0.42 / **+0.07 ± 0.05** | |
| explained variance on training targets, last 100 updates | — | 0.44 | 0.52 | **+0.08 ± 0.03** (6+/0−) | |
| log duel ← value head's input / ← actor trunk (R²) | 0.17 / 0.17 | 0.17 / 0.17 | 0.64 / 0.16 | **+0.48 ± 0.04** / −0.01 | |

Critic responses at neutral L8 (resources, shared → separate):

| change of seat A | shared | separate |
|---|---|---|
| +8 invested | 0.03 | 0.41 |
| +32 invested | 0.09 | 1.19 |
| +4 unspent gold | −0.09 | +0.09 |
| +1 level | 0.00 | 0.06 |
| own HP to 60% | −0.63 | −0.51 |
| enemy HP to 60% | +0.42 | +0.49 |

**Reading.**
- **V1 is "small" by the registered thresholds** (+0.25, and a 0.5 ratio), missed narrowly.
- **The rest of the value evidence points one way.**
  - The separate critic learns the investment premium and the large-gap values in every seed.
  - It keeps improving between 4.096M and 8.192M: edge 0.18 → 0.31, large + gap 0.22 → 0.32, predicted ×1.76 premium
    1.14 → 1.66 resources.
  - It is at least as accurate in natural states.
- **The shared critic does not move.** 12M additional transitions in total (both phases) changed its strength
  recovery by +0.00 to +0.03.
- **What the separate critic still misses.**
  - It recovers only a third of the large-gap premium, a family training never visits (M4.6e).
  - It values a level at almost nothing.

### 5.2 Behaviour

| statistic | separate − shared | class |
|---|---|---|
| **B1** head to head (64 episodes, W 1000) | +0.06 ± 0.23 (3+/3−) | no material difference |
| B1 powered (amendment 1; 320 episodes pooled) | **+0.13 ± 0.09** (6+/0−) | small (not used for the outcome) |
| each vs start:s (W 1000): shared / separate / difference | +0.44 ± 0.15 / +0.58 ± 0.24 / +0.14 ± 0.25 | no regression |
| vs the other five starts (W 460): shared / separate / difference | +0.08 ± 0.13 / +0.42 ± 0.21 / **+0.34 ± 0.29** (5+/1−) | |
| scripted core suite: difference | +0.09 ± 0.51 | no regression |
| **B2** strength use, G150 | +0.01 ± 0.11 (3+/3−) | no material difference |
| conversion / defence / equal cells, G150 | −0.01 ± 0.15 / +0.03 ± 0.24 / −0.03 ± 0.39 | |
| conversion − equal / defence − equal, G150 | −0.01 ± 0.25 / +0.10 ± 0.53 | |
| large ×1.76 cells (diagnostic), G150 | −0.25 ± 0.47 | |
| denial in conversion cells: the opponent's lost farm / XP | −0.09 ± 0.39 / −0.6 ± 2.4 | |
| B3 freezer in favourable freezes (edge, plaus, hp), G50 | +0.05 ± 0.23 | |
| B3 breaker against a favourable freeze, G50 | +0.05 ± 0.20 | |
| B4 weak side vs a stronger opponent, G50 / G100 | −0.03 ± 0.24 / +0.01 ± 0.23 | |
| B4 strong side, advantaged, G50 | −0.13 ± 0.30 | |
| B5 natural leader / trailer states, G150 | +0.08 ± 0.10 / +0.09 ± 0.20 | |
| B5 natural favourable freezes: freezer / breaker, G50 | −0.01 ± 0.23 / +0.02 ± 0.17 | |

- **The targeted behaviours did not change.**
  - Every strength-specific readout is within noise and centred near 0, at 4.096M and at 8.192M.
  - One cell kind has an interval excluding 0: mid-game B+8, −0.48 ± 0.31. It is one of 16, and it has no counterpart
    at the other lead size or the other budget.
- **Action tendencies shifted slightly; outcomes did not.**
  - The weak side's first-turn Safe mass is 0.25–0.26 in both arms (start 0.19).
  - The separate actor Recalls a little more in those windows (0.032 vs 0.011).
  - In favourable freezes the separate freezer trades a little more (Trade 0.17 vs 0.13).
  - None of this moves the rollout values.
- **A possible general edge, reported but not established.**
  - At 8.192M three readouts favour the separate arm: the powered head to head (+0.13 ± 0.09, 6/6), the other-starts
    comparison (+0.34 ± 0.29) and its own start (+0.14 ± 0.25).
  - But they are an amendment and secondary comparisons among many. They are below the registered materiality,
    absent at 4.096M, and not seen in the registered B1.
- **Economy (amendment 2).**
  - The separate actor carries slightly more unspent gold: +0.36 ± 0.06 at a Recall (6/6).
  - This is because the shared arm drifted from the start's profile (−0.48) while the separate arm stayed near it.
  - Recall rate, investment pace, collection, deaths and trades are unchanged.

### 5.3 Consistency across seeds

| seed | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| V1 (separate − shared), 4.096M | +0.23 | +0.11 | +0.07 | +0.16 | +0.11 | +0.03 |
| V1, 8.192M | +0.16 | +0.35 | +0.03 | +0.15 | +0.20 | +0.58 |
| edge recovery, separate / shared, 8.192M | 0.19 / 0.00 | 0.41 / 0.05 | 0.29 / 0.05 | 0.21 / 0.03 | 0.32 / 0.00 | 0.47 / 0.00 |
| V3 (large + gap), 8.192M | +0.27 | +0.31 | +0.32 | +0.30 | +0.34 | +0.25 |
| B1 registered, 8.192M | +0.15 | −0.08 | +0.16 | −0.10 | +0.40 | −0.18 |
| B1 powered, 8.192M | +0.26 | +0.03 | +0.16 | +0.04 | +0.14 | +0.15 |
| B2, 8.192M | +0.06 | −0.08 | +0.16 | −0.08 | +0.07 | −0.09 |

- **Value:** the improvement is consistent in every seed and at both budgets.
- **Behaviour:** the readouts change sign across seeds, except the powered head to head at 8.192M.
- **The registered B1's spread is mostly measurement noise:** its per-seed standard error (0.23 at 64 episodes) is
  about the same as its between-seed SD.

**Sensitivity of the behavioural comparison.**
- **Both policies drifted far.** The two arms' finals differ by a total-variation distance of 0.27 per decision, and
  each is about as far from the start (KL 0.26) as the other. The re-warmed learning rate moves both policies a long
  way, in different directions, along a plateau of nearly equal strength.
- **A critic-caused change in the actor has to stand out against that drift.**
- **Resolution.** With six seed pairs, the registered statistics resolve about ±0.2 resources (B1) and ±0.1 (B2).
  A smaller effect is not excluded, in particular a general one of the size the powered B1 suggests.

## 6. Interpretation

| question | answer | evidence |
|---|---|---|
| 1. Is more training alone sufficient? | **no, for the critic** | the shared critic's strength recovery changes by +0.00 to +0.03 over 4.096M and 8.192M of further training; its edge recovery stays at 0.01–0.02 |
| 2. Does separating the critic help? | **for value learning, yes, partially; for behaviour, not demonstrated** | V1 +0.24 (6/6, just short of the threshold), V3 +0.30, probes +0.48, no natural cost; B1, B2 and every targeted family show no material difference |
| 3. Better values without behavioural benefit? | **the closest description** | the value gain is large and consistent; the strength-specific behaviours are unchanged at both budgets |
| 4. Inconclusive, or not addressing the problem? | **formally yes (pre-registered outcome 4)** | V1 missed the registered materiality by 0.01 and the 0.5 floor by 0.05; the separate critic still values a level at almost nothing and recovers a third of the large-gap premium |

**What the experiment establishes.**
- **Sharing is enough to cause the blindness here.** In this setup, actor–critic sharing is sufficient to keep the
  critic strength-blind. The same network, data and optimizer learn strength value once the critic has its own trunk.
  M4.6e's supervised result carries over to PPO's own bootstrapped training.
- **Separation is cheap.** Removing the sharing costs 4% throughput and 28,160 parameters. It causes no measured
  regression in health, natural calibration, ordinary play or the fixed-opponent guards.
- **The actor does not follow, within 8.192M transitions.** A substantially better critic does not change how the
  actor converts strength, defends, freezes or plays weak-side Recall windows.

**What remains unresolved.**
- **Which mechanism.** The candidates are the removed gradient interference, Adam's per-parameter scaling of the
  critic trunk, and the added capacity. The gradient probe rules out one alternative: the global clip did not couple
  the arms.
- **Whether the actor would follow later.** The critic was still improving at 8.192M. The pre-registration allowed
  one extension, and this experiment does not extend further.
- **Whether the small general edge is real**: powered B1 +0.13 ± 0.09 and other starts +0.34 ± 0.29, at 8.192M only.
- **A from-scratch campaign.** A separate critic present from the start could shape different strategies; continuing
  a converged policy cannot show that.
- **Level and large gaps.** The critic remains nearly blind to level, and large gaps are absent from training
  (M4.6e). Their values are still underestimated threefold.
- **Why values do not reach the actor.** This is explained, not tested:
  - a strength error that is the same for every candidate of a decision cancels in the advantage (M4.6e §5.3);
  - only actions that change strength (spending, experience) can be credited through it;
  - the economy profile shows the actor's spending barely moved.

## 7. Recommendation

**Accept the limitation and close this investigation; proceed to M4 closeout.**
- **Why.** The one justified targeted intervention changed the critic, as intended, but not the strategic behaviour
  it was meant to improve. By the stopping principle, the remaining weakness is documented and the project proceeds.
  That weakness is the actor's conversion of plausible strength advantages, plus the critic's blindness to level and
  to large gaps.
- **At M4 closeout**, record `SeparateCriticNet` as a verified, opt-in training option for M5:
  - better values in strength states;
  - no measured regression;
  - a 4% throughput cost.
  - Choosing it for M5 is a setup decision, not a strategic fix. Do not adopt it on value accuracy alone; it is the
    better choice if M5 reads critic values anywhere.
  - Either way, strength-gap states keep being evaluated with rollouts.
- **Not recommended:** another critic pilot (a value-loss weight, explicit strength features, longer extensions).
  The value side is already answered. The behavioural side did not respond to a much better critic, so a slightly
  better one is unlikely to change it.

## 8. Reproduction, files and tests

**Commands** (from the repository root; every stage is resumable, with records under `results/rl/checkpoints/m46f/`):

```bash
P="python -m lanegame.experiments.rl.m46f"
$P verify smoke prereg                       # equivalence (60 checks), smoke, freeze the protocol
$P train --phase main --workers 6            # 12 x 4.096M (55 min)
bash results/m46f/run_queue_main.sh          # train_check, value (main, mid), behaviour, ext_rule, analyse, report
bash results/m46f/run_queue_ext.sh           # the fired extension: 12 x 8.192M (1 h 50 min) and its evaluation
$P behaviour --phase main --part h2h_power,economy --workers 2    # amendments 1 and 2 (also --phase ext)
$P analyse report                            # results/m46f/analysis.json and report.md
python -m pytest tests/test_m46f.py          # 12 tests
```

**Wall clock.**
- Training: 55 min (main) + 1 h 50 min (extension).
- Evaluation: about 1 h per phase with 8 workers (value 15 min, behaviour 35 min), plus the amendments.

**Seeds: the range 89,000–89,999**, except the calibration (last item):
- training 89,000 + s, the same in both arms;
- natural labels 89,200;
- head to head 89,300; powered head to head 89,310;
- cells 89,400–89,402;
- natural states 89,500; economy 89,550 + s;
- gradient probe 89,600; equivalence 89,700;
- the calibration reuses M4.6d's 99,500, so that every critic is labelled on the same rollout seeds.

**New files:**
- `lanegame/experiments/rl/m46f/`: `common`, `train` (with the gradient probe), `verify`, `value`, `behaviour`,
  `analysis`, `report`, the CLI;
- `tests/test_m46f.py`;
- `docs/m4/m4_6/m4_6f_preregistration.md` and this report;
- `results/m46f/`:
  - `preregistration.json` (with the amendments), `verify.json`, `smoke.json`;
  - `train_main.json`, `train_ext.json`, `extension_decision.json`, `analysis.json`;
  - the generated `report.md`, the logs and the queues.

**Modified:**
- `lanegame/rl/network.py`, `agent.py`, `shared.py`, `__init__.py` (§1.1; opt-in, backward compatible);
- `docs/rl_research_roadmap.md`, `docs/m4/README.md`, `docs/README.md`, `README.md`, `CLAUDE.md` (one line).

**Unchanged:**
- The checkpoints (git-ignored) are under `results/rl/checkpoints/m46f/train/{main,ext}/{shared,separate}/s<seed>/`.
  The M4.6c starting checkpoints were only read.
- The mathematical reference is unchanged, because no rule, parameter or objective changed.
