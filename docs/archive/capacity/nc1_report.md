# NC1 network capacity report

*2026-10-29. Status: complete, awaiting review. Uncommitted on the branch `experiment/nc1-capacity`. No rule,
observation, reward, default or stored checkpoint changed. The library gained two opt-in, function-preserving network
changes (`widen_hidden`, `StrengthFeatureNet`) and one default-preserving parameter (`m46e.fit.make_mlp(hidden=...)`).*

*Protocol: [nc1_preregistration.md](nc1_preregistration.md), frozen at 09:24 (SHA-256 75adc051… in
`results/nc1/preregistration.json`) after the smoke stage and before any fit, behaviour rollout or PPO update. No
amendment. Generated tables: `results/nc1/report.md`.*

**Terms.**
- **Resources.** A resource is the gold of one fresh wave. G_h is a seat's discounted (γ 0.99) earned-gold lead over h
  turns, with no critic bootstrap.
- **G and A.** G is an M4.5c final (`final:ext:t`, the generalist). A is its M4.6a gap attacker, a specialist trained
  from power-gap starts that converts gaps about +2.1 G100 better than G.
- **λ (lost specialization).** The share of the specialist's difference from the generalist that a student fails to
  copy, on held-out gap states. 0 keeps all of it and 1 plays like the generalist.
- **ρ (retention).** The share of the specialist's +G100 gain over G that a student keeps when it plays.
- **W, F and the control.** W is `ext/baseline:s` widened to 2 × 256. F is `ext/baseline:s` with six strength
  features as extra inputs. The control is M5's pure self-play snapshot of the same seed at 16,384,000 transitions.
- **Intervals.** 95% t intervals. Over six seeds, t₅ = 2.571.

---

## Summary

**The question.** No study had varied the policy network, a 2 × 128 tanh MLP chosen by convention in M3a. Does its
size or the form of its inputs limit the agent?

**On this evidence, size does not limit the agent, and the input form does not limit its play.** NC1 stops here by its
registered recommendation rule.

- **2 × 128 can represent better play than PPO finds (stage 1).** One fresh 2 × 128 network, fitted to copy the M4.5c
  generalist on natural states and M4.6a's gap specialist on power-gap states, kept 98% of the specialist's +2.1 G100
  gain in play. It lost no natural play. Doubling the width or adding depth improved the copy only marginally, and
  never significantly in play. The registered verdict is "2 × 128 suffices". The generalist's power-gap deficit comes
  from training, not from capacity.
- **Width does not help the value function either (stage 1).** A 2 × 256 value network recovers no more of the
  strength value the critic misses than 2 × 128 (−0.001 ± 0.029). Six observation-derived strength features recover
  much more (+0.73 ± 0.31), as M4.6e found.
- **The trained networks use their width, but their second layer saturates (stage 1).** No unit is dead, and the
  second layer spans 91–94 effective dimensions of 128. But 61–67% of its activations sit in tanh's flat region in the
  M5 finals, up from 33–36% in the M4.5c finals. The registered rule fires on that saturation and names normalization
  as the next network change to test. NC1 did not test it.
- **A wider network trained with M5's settings played worse (stage 2).** W, widened to 2 × 256 without changing its
  outputs, lost to its paired M5 control head to head (−0.67 ± 0.59, lead at turn 114). It was also worse in the
  conversion (−0.47 ± 0.37) and power-gap (−0.39 ± 0.35) families. At the same learning rate, its updates moved the
  policy about 50% further, so W tests width together with a larger effective step.
- **The strength features gave a more strength-aware critic but no measurable gain in play (stage 2).** F, the same
  network with the six features as zero-weight inputs, trained like the control.
  - Head to head it was level (−0.17 ± 0.42).
  - Its critic predicts +0.95 ± 0.25 of a realized +4.7 power-gap premium, against the control's +0.14 ± 0.09.
  - It was better at Recall windows (+0.17 ± 0.11), most of all as the weaker side facing a stronger opponent
    (+0.46 ± 0.27, an exploratory group), with no family worse.
  - None of this meets the registered bar for "improves play".
- **Stage 1 predicted stage 2.** Stage 1 found that size does not limit representation, and the wider network did not
  improve. It found that the features help the value function, and F's critic improved. It found that the features
  do not help represent the policy, and F's play did not change measurably.

**Recommendation.** Keep the 2 × 128 tanh network. Do not widen it at M5's learning rate. The strength features are a
cheap, safe opt-in for a strength-aware critic, worth reconsidering only if a later study needs critic values in power
gaps. The adoption decision is the user's.

## 1. What was run

| stage | content | compute | result |
|---|---|---|---|
| smoke | the control, W and F on seeds 0 and 1 for 25 updates, six at a time | 41 s | M5's history reproduced exactly; W and F sample M5's first rollout |
| Q1 | unit usage of 18 stored networks, then of the 12 pilot finals and their 6 controls | 6.5 min + about 5 min | no dead units; layer-2 saturation 0.61–0.67 in M5's finals |
| Q2 | 12 teacher data sets, 63 student fits, 15 students played on E4's streams and from s0 | 3.4 + 37 + 19 min | "2 × 128 suffices"; E4's 450 records reproduced exactly |
| Q3 | 54 value refits on M4.6e's labels | 2 min | width does not bind for the value |
| stage 2 | W and F × 6 seeds × 16,384,000 transitions | 3.5 h wall | all complete, no health alarm |
| stage 3 | E1 head to head (6 × 256 games per arm), E2 limitation cells (3,600 jobs), E3 | 42 + 30 min | W worse; F level, with a better critic |

The pre-registration froze at 09:24 and the last stage finished at 13:25.

## 2. Q1. Trained networks use their width, but the second layer saturates

Each network played itself for 16 games from s0 and 100 turns from each of 75 held-out gap starts. About 6,200 network
inputs per network entered the readouts.

| networks | layer 1 inactive | layer 1 saturation | layer 2 inactive | layer 2 saturation | layer 2 effective rank | layer 2 components for 99% |
|---|---|---|---|---|---|---|
| M5 sp finals (6) | 0 | 0.02 | 0 | 0.61–0.62 | 91–94 | 106–110 |
| M5 om finals (6) | 0 | 0.02–0.03 | 0–2 | 0.62–0.67 | 91–94 | 101–107 |
| M4.5c finals `ext:0, 2, 4` | 0 | 0.00 | 0 | 0.33–0.36 | 82–84 | 99–102 |
| M4.6a gap attackers | 0 | 0.00 | 0 | 0.38–0.41 | 85–86 | 103–105 |

- **No unit is dead.** At most 2 of 128 units in any layer stay at one sign or never move.
- **The second layer saturates, and saturation grows with training.** In the M5 finals 61–67% of second-layer
  activations sit at |h| ≥ 0.99, where the tanh derivative is at most 0.02. The M4.5c finals, with 16.4M transitions of
  training instead of 57.3M, sit at 33–36%. The attackers, which continued the M4.5c finals for 4.1M transitions, sit
  just above them.
- **The layer still spans most directions.** Its effective rank is 91–94 of 128, and 101–110 principal components carry
  99% of the variance. The saturated units flip sign across states, so they act as many binary features rather than as
  idle width.
- **The registered rule fires on saturation.** The median second-layer saturation share is 0.62, above the 0.5
  threshold, while the median count of inactive units is 0. By the rule, normalization (LayerNorm before each tanh) is
  the next network change to test. A LayerNorm network cannot start as exactly `ext/baseline:s`, so this pilot does not
  test it.

## 3. Q2. One 2 × 128 network represents the generalist and the specialist together

For each target, students were fitted to copy G on natural self-play states and A on the states of A's gap games (A in
seat A). About 87,000 training examples and 15,000 validation examples per student, three fit seeds each.

**What the students copy** (held-out KL, mean of three fit seeds):

| target | D = KL(A ‖ G) | λ U64 | λ U128 | λ U256 | λ U3×128 | λ U128F | κ U128 (nats) |
|---|---|---|---|---|---|---|---|
| 0 | 0.601 | 0.078 | 0.035 | 0.016 | 0.015 | 0.025 | +0.006 |
| 2 | 0.693 | 0.063 | 0.029 | 0.013 | 0.013 | 0.020 | +0.007 |
| 4 | 0.639 | 0.069 | 0.031 | 0.014 | 0.012 | 0.021 | +0.007 |

- The specialist differs from the generalist by 0.6–0.7 nats on its own states. A single 2 × 128 student copies all
  but about 3% of that difference (λ 0.029–0.035) and stays within 0.007 nats of its single-teacher floor on natural
  states.
- Capacity helps smoothly and slightly. Halving the width doubles λ, and doubling it halves λ. Depth (3 × 128) and
  width (2 × 256) do equally well. The six strength features (U128F) recover about a third of what doubling the width
  recovers.

**What the students play** (G100 on E4's held-out gap states, 75 per target, against G, on E4's own rollout seeds):

| target | attacker − G | U64 − G | U128 − G | U256 − G | U3×128 − G | U128F − G | U256 − U128 | U128F − U128 |
|---|---|---|---|---|---|---|---|---|
| 0 | +2.34 ± 0.40 | +2.11 ± 0.36 | +2.14 ± 0.34 | +2.32 ± 0.43 | +2.29 ± 0.39 | +2.41 ± 0.40 | +0.18 ± 0.35 | +0.27 ± 0.35 |
| 2 | +1.84 ± 0.43 | +1.63 ± 0.48 | +1.86 ± 0.44 | +1.95 ± 0.41 | +1.81 ± 0.41 | +1.77 ± 0.40 | +0.09 ± 0.34 | −0.09 ± 0.37 |
| 4 | +2.14 ± 0.35 | +2.00 ± 0.33 | +2.21 ± 0.36 | +2.37 ± 0.36 | +2.34 ± 0.34 | +2.56 ± 0.37 | +0.16 ± 0.33 | +0.35 ± 0.35 |

- The pooled retention is 0.98 for U128, 1.05 for U256, 1.02 for U3×128, 1.07 for U128F and 0.91 for U64.
- Against G from s0, the students' estimates range from −0.41 to +0.31 and every interval includes 0. None lies below
  −0.3, so the natural guard holds.
- The re-run of G and A reproduced all 450 of E4's stored rollout records exactly.

**Verdict (registered rule): "2 × 128 suffices".** λ(128) ≤ 0.10 on every target, 2 × 256 improves λ by less than 0.05,
and ρ(128) = 0.98. "Width binds" and "features help" both fail. Neither U256 − U128 nor U128F − U128 is significant on
any target. The generalist's power-gap deficit (M4.6a) is not a capacity limit. A 2 × 128 network can hold the better
gap play beside the generalist's natural play.

## 4. Q3. A wider value network recovers no more strength value

M4.6e's arm C refitted at three more shapes on the stored labels of the six `ext/C1:s` policies. A refit of the
2 × 128 shape reproduced M4.6e's stored predictions exactly (maximum difference 0.0).

| value network | P1 edge + plaus | P1 large + gap | natural MAE |
|---|---|---|---|
| 2 × 64 | 0.84 ± 0.26 | 0.91 ± 0.13 | 2.23 |
| 2 × 128 (M4.6e arm C) | 0.75 ± 0.26 | 0.84 ± 0.12 | 2.23 |
| 2 × 256 | 0.75 ± 0.25 | 0.80 ± 0.12 | 2.23 |
| 3 × 128 | 0.85 ± 0.32 | 0.88 ± 0.11 | 2.27 |
| 2 × 128 + strength features (M4.6e arm D) | 1.48 ± 0.54 | 1.13 ± 0.11 | 2.23 |

- Paired by policy, 2 × 256 minus 2 × 128 is −0.001 ± 0.029 (edge + plaus) and −0.04 ± 0.06 (large + gap). Width does
  not bind for the value (registered rule).
- The input form does. The six features add +0.73 ± 0.31 and +0.29 ± 0.10, as M4.6e found. The smaller 2 × 64 network
  recovers slightly more than 2 × 128 (+0.09 ± 0.04 and +0.07 ± 0.03).

## 5. Stage 2. The PPO pilot trained cleanly, and W took larger steps

**Pairing.** The smoke stage showed the design holds exactly. A re-run of the control configuration reproduced M5's
history for 25 updates on every deterministic field (seeds 0 and 1). W and F, which start as exactly
`ext/baseline:s`, sampled M5's first rollout (equal actions digests).

**Runs.** All twelve runs reached 16,384,000 transitions with no health alarm (no DEFECT, no INVESTIGATE). W took
1.83 h per run and F 1.66 h, six at a time. With stage 1 sharing the machine for the first hour, the pilot took 3.5 h of
wall clock.

| last 200 updates (mean over six seeds) | entropy | approx. KL per update | clip fraction | gradient norm |
|---|---|---|---|---|
| control (M5 sp, updates 3,801–4,000) | 1.667 | 0.0041 | 0.046 | 0.760 |
| F (2 × 128 + features) | 1.660 | 0.0041 | 0.047 | 0.768 |
| W (2 × 256) | 1.611 | 0.0061 | 0.080 | 0.884 |

- F trained like the control on every statistic.
- W did not. At the same learning rate its policy moved about 50% further per update (KL), its updates hit the clip
  bound 74% more often, and its entropy fell further. Adam scales each parameter's step to the learning rate, so a network with
  three times the parameters changes its outputs more per update. W therefore tests width together with a larger
  effective step size, not width alone.
- W uses its new units. On Q1's states its second layer has an effective rank of 177–181 of 256 (the control: 93–94),
  with no inactive unit and the same saturation share (0.60–0.65 against 0.60–0.61).

## 6. Stage 3. Neither arm improves play

Every readout is arm − paired control per seed, then the mean over six seeds with a 95% t interval. The control is
M5's pure self-play snapshot of the same seed at 16,384,000 transitions.

### 6.1 E1, head to head (primary)

256 paired-seat games of 150 turns per seed from s0, on NC1's seeds. W and F met the control on the same games. The
statistic is M5's lane-exit readout, the arm's lead at turn 114.

| arm | arm − control | per seed (s0 to s5) |
|---|---|---|
| W (2 × 256) | **−0.67 ± 0.59** | −1.20, −1.14, −0.68, −0.06, −1.02, +0.10 |
| F (2 × 128 + features) | −0.17 ± 0.42 | −0.78, −0.51, +0.17, +0.04, −0.16, +0.19 |

W lost to its control, with an interval that excludes 0. F's interval includes 0.

### 6.2 E2, the limitation families

The arm and the control played seat A on the same rollouts, each against both networks in seat B. The rows are M5c's
"all" groups at their horizons, then the groups that moved.

| family and group | W − control | F − control |
|---|---|---|
| conv, all, G150 | **−0.47 ± 0.37** | −0.05 ± 0.25 |
| conv, ×1.76 conversion, G50 | −0.14 ± 0.51 | **+0.38 ± 0.28** |
| freeze, all, G50 | **−0.27 ± 0.19** | +0.12 ± 0.16 |
| freeze, freezer, G50 | −0.28 ± 0.26 | **+0.11 ± 0.10** |
| recall, all, G100 | −0.31 ± 0.39 | **+0.17 ± 0.11** |
| recall, weak side facing a stronger opponent, G100 | −0.30 ± 0.54 | **+0.46 ± 0.27** |
| recall, strong and advantaged, G100 | −0.29 ± 0.48 | **+0.12 ± 0.07** |
| gaps, all, G150 | **−0.39 ± 0.35** | +0.24 ± 0.39 |
| gaps, gap cells, G150 | −0.44 ± 0.48 | +0.35 ± 0.57 |

Bold marks an interval that excludes 0.

- **W is worse everywhere.** Its mean is negative in all four families and significantly so in three. Conv and gaps
  meet the registered regression clause (a mean below −0.3 with an interval excluding 0).
- **F is never worse, and better in a few groups.** Its Recall-window gain is concentrated where the weaker side faces
  a stronger opponent. That is the M4.6d family where a damaged weak side should step back rather than Yield, and where
  M4.6f's separate critic moved behaviour slightly without a measurable gain. Its power-gap readouts point upward but
  are not significant, so F does not meet "improves gap play". The table reads about 40 groups per arm, and the
  pre-registration names only the "all" rows, so F's group-level gains are exploratory.
- **The gap specialist's margin.** M4.6a's gap specialist out-converts each network in that network's own game by
  +1.63 ± 0.36 (W), +0.92 ± 0.60 (control) and −0.10 ± 0.64 (F) at the gap cells (G150). These margins come from
  different games, so they are descriptive. They place F closest to the specialist and W furthest from it.

### 6.3 E3, descriptive readouts

| network | log duel R² from the last hidden layer | critic's predicted gap premium | realized gap premium (G150, self-play) |
|---|---|---|---|
| control | 0.161 ± 0.010 | +0.14 ± 0.09 | +5.37 ± 1.02 |
| W | 0.219 ± 0.016 | +0.23 ± 0.08 | +5.31 ± 0.50 |
| F | 0.253 ± 0.021 | +0.95 ± 0.25 | +4.73 ± 0.48 |

- The control's last hidden layer keeps little of the strength relation (R² 0.16), as M4.6e found for C1's policies
  (0.18). The features raise it by +0.092 ± 0.026 and the width by +0.058 ± 0.021.
- F's critic sees about a fifth of the realized power-gap premium, against 3–4% for the control and W. It is still
  far from calibrated in power gaps.
- Training health and unit usage are in §5.

### 6.4 The registered verdicts

| arm | E1 | regressions | improves gap play | registered verdict |
|---|---|---|---|---|
| W | interval below 0 | conv, gaps | no | **inconclusive** |
| F | interval includes 0 but reaches −0.59 | none | no | **inconclusive** |

The registered table has no row for an arm that loses. W falls into "inconclusive" only because of that gap in the
protocol. In plain terms W is worse than its control. F's interval extends below −0.5, so it misses "no material
effect" as well. Neither arm "improves play".

## 7. Did stage 1 predict stage 2?

| stage 1 finding | stage 2 observation | agreement |
|---|---|---|
| 2 × 128 represents the better gap play (Q2) | W, with more capacity, did not convert gaps better | yes |
| width does not help the value (Q3) | W's critic premium +0.23 against the control's +0.14 | yes |
| the features help the value (Q3) | F's critic premium +0.95, its trunk keeps more strength | yes |
| the features do not help represent the policy (Q2, U128F − U128 not significant) | F level head to head, gap play not significantly better | yes |

Stage 1's cheap supervised probes predicted every stage 2 outcome. W's loss is the one result stage 1 could not
predict, because it comes from optimization (a larger effective step) rather than representation.

## 8. Recommendation

By the registered rule, neither arm improves play and stage 1 found 2 × 128 sufficient. NC1 therefore documents that
the network's size and input form do not limit the agent on this evidence, and the investigation stops.

Proposed details for the user's review:
- **Keep the 2 × 128 tanh network** for any further training. Its capacity is not what holds the policy back. The
  M4.6a reading stands. Gap states are rare in training, and training on them teaches conversion.
- **Do not widen at M5's learning rate.** If width is ever revisited, its learning rate needs retuning first, for
  example to match the control's KL per update. NC1 gives no reason to do so.
- **The strength features are a candidate, not an adoption.** They are opt-in and function-preserving at the start,
  and they trained safely. They gave the critic about seven times more of the power-gap premium and a Recall-window
  gain for the weak side, with no loss anywhere. A study that needs strength-aware critic values could reuse them.
  Adoption would need the registered full-budget confirmation first, and the decision is the user's.
- **Saturation is the open network question.** Q1's rule fired. The second layer saturates more the longer the
  lineage trains, a pattern associated with lost plasticity in long training. Testing LayerNorm would need a design
  that cannot pair as exactly with M5 as NC1's arms did. NC1 does not recommend running it now.

## 9. Limitations

- **One budget and one learning rate.** The pilot stopped at 16,384,000 transitions, half of M5's schedule, and kept
  M5's learning rate. W's result mixes width with a larger effective step. A wider network with a retuned learning
  rate, or a longer run, is untested.
- **Continuation, not training from scratch.** Both arms continue a network with 24.6M transitions of training. A
  network trained wide from the start may learn differently.
- **The distillation probe uses the teachers' states.** Imitation errors compound in play. The behaviour readouts
  measured that directly and found retention near 1 at every width.
- **The fits did not converge.** Every Q2 student used all 150 epochs, so the λ values are upper bounds. The budget
  was the same for every student, so the comparison across shapes stays fair.
- **The specialist margins come from different games.** §6.2's specialist readouts compare networks in their own
  games, not on paired rollouts.
- **The Q1 saturation threshold is a proxy.** A saturated unit still carries information when it flips sign across
  states. The rule's firing names a direction to test, not a measured cost.

## 10. Reproduction, deviations and decisions

**Commands** (from the repository root, after the pre-registration is frozen):

```bash
python -m lanegame.experiments.rl.nc1 smoke prereg
python -m lanegame.experiments.rl.nc1 train --confirm-nc1-train
python -m lanegame.experiments.rl.nc1 q1 q2data q2fit q2behaviour q2analyse q3 --workers 3
python -m lanegame.experiments.rl.nc1 evaluate q1pilot report
python -m pytest tests/test_nc1.py
```

**Deviations from the pre-registration.** None in design or rules. Two implementation details:
- Q1 keeps every 5th network evaluation, which includes turns with a single legal action. The pre-registration says
  every 5th turn.
- Q2 records decision turns only (at least two legal actions) at even turn indices.

**Decisions made during the autonomous run** (the user granted autonomy on 2026-10-29):
- Both PPO arms ran regardless of stage 1, at 16,384,000 transitions. That was the recommended default, chosen from
  the smoke's measured throughput.
- `--confirm-nc1-train` was passed on the user's message approving up to 8 hours of compute.
- The work stays on the branch `experiment/nc1-capacity`, and nothing is committed. The user's staged M6d changes were
  not touched.
- `m46e.fit.make_mlp` gained a `hidden` parameter whose default reproduces M4.6e's stored fits exactly (checked).
- E2 uses NC1's rollout seeds (root 48,900) in place of M5's, as registered.

**Artifacts.**
- `results/nc1/` holds `smoke.json`, `preregistration.json`, `q1.json`, `q1_pilot.json`, `q2.json`, `q3.json`,
  `q3_repro.json`, `training.json`, `evaluation.json`, `report.md` and the stage logs.
- `results/rl/checkpoints/nc1/` (git-ignored) holds the twelve pilot runs with every snapshot, the Q2 data, fits and
  students, and every evaluation record.
