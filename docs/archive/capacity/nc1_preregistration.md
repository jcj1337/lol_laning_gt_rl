# NC1 pre-registration of the network capacity study

*2026-10-29. Frozen (SHA-256 in `results/nc1/preregistration.json`) after the smoke stage and before any fit,
behaviour rollout or PPO training. Before freezing, the author had seen M4.6a's E4 results, M4.6e's results (including
arm C at 2 × 128 and arm D, the baselines of Q3), a single-thread timing of the network at six sizes, and the smoke
stage (`results/nc1/smoke.json`). The smoke checked the pairing and measured throughput. It produced no evaluation
number.*

## Scope

NC1 asks whether the 2 × 128 tanh network limits the agent. It changes no rule, observation, reward, default or stored
checkpoint. The library gains two opt-in, function-preserving network changes, `widen_hidden` and
`StrengthFeatureNet`. The environment is `m45_calibrated()`, except in Q3, which reuses M4.6e's labels from C1
(`m45_calibrated(invest_gain=3.24, invest_scale=64.0)`).

**Why.** No study has varied the width. M3a chose 2 × 128 for a game with 50 inputs, following common PPO practice,
and every later campaign kept it so that runs stayed comparable (M4.3b lists it among its limitations).
- M4.6e fitted a fresh 2 × 128 value network to the critic's labels. It recovered 71–87% of the strength premiums the
  critic misses. With six observation-derived strength features (arm D) it recovered 95–167%, and the report found the
  features necessary to extrapolate beyond the trained strength range.
- M4.6a's gap attackers, with the same architecture, convert power gaps +2.11 G100 better than their targets.

NC1 has two stages. Stage 1 asks what a network can represent, by supervised fits. Stage 2 asks what a network learns
under PPO, by a pilot paired with M5's pure self-play arm. Stage 2 runs whatever stage 1 finds. Stage 1's verdict is
then a prediction that stage 2 checks.

## Stage 1. What the network can represent (no PPO)

### Q1. Unit usage (descriptive, one rule)

- **Networks.** 18 in all.
  - The M5 finals `sp:0–5` and `om:0–5`.
  - The M4.5c targets `final:ext:0`, `final:ext:2` and `final:ext:4`.
  - Their M4.6a gap attackers (`results/rl/checkpoints/m46a/e4/gap_s{0,2,4}/final.pt`).
- **States.** Each network plays itself, and states are taken every 5th turn from both seats.
  - 16 games of 500 turns from s0.
  - 100 turns from each held-out E4 gap start of target 0 (27 constructed cells, 48 semi-natural states).

  OM networks play with their belief (`m5.players`).
- **Inactive units.** A unit counts as inactive when it stays at |h| ≥ 0.99 with one sign on at least 99% of states,
  or when its SD across states is below 0.01.
- **Saturation share.** The share of (unit, state) pairs at |h| ≥ 0.99, where the tanh derivative is at most 0.02.
- **Effective rank.** The exponential of the entropy of the normalized singular values of the centred activations, and
  the number of principal components that reach 99% of the variance.
- **Rule.** Layer 2 "leaves width unused" when, over the 18 networks, the median count of inactive units is at least
  32 or the median saturation share is at least 0.5. The report then names normalization (LayerNorm before each tanh)
  as the next network change to test. A LayerNorm arm cannot start as exactly ext/baseline:s, so it is not part of this
  pilot. The sp and om networks are reported side by side and not tested against each other.
- The same readouts describe the 12 pilot finals after stage 2. They show whether W's new units came into use.

### Q2. The capacity probe

**Teachers.** For each target t ∈ {0, 2, 4} there are two teachers, both 2 × 128.
- **G** is `final:ext:t`, the M4.5c final (the generalist).
- **A** is the M4.6a gap attacker of t (the specialist). It was warm-started from G and trained for 4,096,000
  transitions from gap starts as the stronger side.

**Data.** All seeds are new. A group is one game or one archive start.

| set | source | teacher |
|---|---|---|
| N train | G self-play from s0, 48 games × 500 turns, every 2nd turn, both seats | G |
| P train | A in seat A (the stronger side) against G, from each distinct start of E4's gap training archive (27 cells, 576 semi-natural states), one 150-turn rollout each, every 2nd turn | A for seat A, G for seat B |
| N test | 16 further G self-play games | G |
| P test | the P-train protocol from E4's held-out gap starts (27 cells, 48 held-out semi-natural states) | as P train |

The fits hold out 15% of the training groups for validation. N and P carry equal total weight.

**Students.** Each student is a freshly initialized network (orthogonal init, value head unused). The loss is
KL(teacher ‖ student) over the legal actions. Training follows M4.6e's settings: Adam 10⁻³, minibatch 512, up to 150
epochs, and early stopping after 15 epochs without a validation gain. Every student has fit seeds 0, 1 and 2.

| student | trained on | network |
|---|---|---|
| U64, U128, U256, U3×128 | N ∪ P, each example with its own teacher | 2 × 64, 2 × 128, 2 × 256, 3 × 128 |
| U128F | N ∪ P, each example with its own teacher | 2 × 128 plus the six strength features (`StrengthFeatureNet`) |
| F_G | N and P seat B, teacher G | 2 × 128 |
| F_A | P seat A, teacher A | 2 × 128 |

A 2 × 128 student can represent each teacher exactly. The test KL of F_G and F_A is therefore the fitting floor, the
error that remains with no capacity limit.

**Primary readouts.** These are on held-out data, averaged over fit seeds, per target and pooled.
- **D** is the mean KL(A ‖ G) on P-test seat-A states, the distance between specialist and generalist on the states
  where the specialist plays.
- **Lost specialization** λ(w) = [KL(A ‖ U_w) − KL(A ‖ F_A)] / D on P-test seat-A states. At λ = 0 the student keeps
  all of the specialist's difference. At λ = 1 it plays like the generalist.
- **Natural excess** κ(w) = KL(G ‖ U_w) − KL(G ‖ F_G) on N test, in nats, reported beside D for scale.
- If D is below 0.05 nats for a target, λ does not apply to that target and the behavioural retention decides.

**Behaviour.** The behaviour readouts play fit seed 0 of each union student.
- **Gap.** Each student plays seat A against G (seat B) from E4's held-out gap sets.
  - The rollouts use E4's seeds (the `E4_EVAL_ROOT` streams, K 16 per cell, 8 per semi-natural state), a horizon of
    150 turns and G100 as the primary measure. The pooling of cells and semi-natural states follows E4's amendment.
  - G and A play again on the same streams. Their difference must reproduce E4's stored per-target values exactly.
  - Retention ρ(w) = (U_w − G) / (A − G).
- **Natural.** Each student plays G from s0 (`evaluate_discounted`, 64 paired seeds, W = 460).

**Decision rules** (fixed now)

| result | condition | reading |
|---|---|---|
| **2 × 128 suffices** | λ(128) ≤ 0.10 on all three targets, λ(256) ≥ λ(128) − 0.05, and pooled ρ(128) ≥ 0.5 | one 2 × 128 network represents both policies, so the generalist's gap deficit comes from training, not from size |
| **width binds** | λ(128) − λ(256) ≥ 0.10 on all three targets, and U256 − U128 on the gap sets is positive with a 95% interval excluding 0 on at least two targets | a 2 × 128 network cannot represent both policies, and a 2 × 256 one can |
| **features help** | λ(128) − λ(128F) ≥ 0.10 on all three targets, and U128F − U128 on the gap sets is positive with a 95% interval excluding 0 on at least two targets | the input form, not the size, limits the 2 × 128 network |
| **partial** | none of the above | reported |

"Width binds" and "features help" can both hold. The natural guard applies to every result. A student whose natural
result against G has a 95% interval entirely below −0.3 counts as a natural regression. U64 and U3×128 are
descriptive. They show the shape of the capacity curve and enter no rule.

**Known bias.** The KL readouts use states the teachers visit, not the states a student visits. Imitation errors
compound in play, so ρ can fall below what λ suggests. The bias applies equally at every width, so the comparison
across widths stays fair.

### Q3. Value capacity on M4.6e's labels

Q3 refits M4.6e's arm C at 2 × 64, 2 × 256 and 3 × 128. It uses the stored labels of the six `ext/C1:s` policies, the
main split, the same groups, the same fit seeds 0–2 and the same procedure. A refit of 2 × 128 at fit seed 0 for one
policy must reproduce M4.6e's stored predictions exactly. The stored 2 × 128 arm C and arm D results are the
baselines. The readouts are M4.6e's P1 ratios (edge + plaus, and large + gap) and the natural MAE.

Width "binds for the value" when 2 × 256 raises both P1 ratios by at least 0.10 over 2 × 128 (mean over the six
policies, 95% interval excluding 0) and keeps the natural MAE within 1.05 × that of 2 × 128.

## Stage 2. What the network learns (PPO pilot)

**Arms.** Each arm starts as exactly `ext/baseline:s`, the start of M5's arms.
- **W** widens the network to 2 × 256 (`widen_hidden`). The new units take seeded orthogonal incoming weights (torch
  seed 48,000 + s) and zero outgoing weights.
- **F** keeps 2 × 128 and adds the six strength features as inputs with zero weights (`with_strength_features`).

**Protocol.** Everything but the network is M5's sp run, imported from `m5.train`.
- PPO seed 70,000 + s, the M4.6c settings, the M4.6c start mixture, 100% current-policy self-play.
- A fresh Adam with the learning rate annealed linearly over M5's 32,768,000 transitions.
- `stop_at` 16,384,000 transitions, an exact prefix of that schedule.
- Six seeds per arm, run under M5's `health.RunMonitor`. A DEFECT alarm ends a run, and the report states it.

**Control.** M5's sp snapshot of the same seed at 16,384,000 (`snap_016384000.pt`), M5's registered midpoint. No
control run is retrained. The smoke stage verified the pairing on seeds 0 and 1.
- A re-run of the control reproduced M5's history exactly for 25 updates, on every deterministic field.
- W and F sampled exactly M5's first rollout (the actions digests match).

**Cost.** Measured at the pilot's load of six concurrent runs, the steady rates are 3,380 (W) and 3,500 (F)
transitions/s. A run then takes about 1.35 h, and the twelve runs on six workers take about 3 h.

## Stage 3. Evaluation of the pilot

Every readout compares an arm's final with its paired control, per seed. The summary is the mean over the six seeds
with a 95% t interval (t₅ = 2.571).

- **E1, primary: head to head.** The arm's final against the paired control from s0, with M5's design
  (`m5.games.cell_job`). There are 256 games of 150 turns per seed, each played in both seat orientations, on NC1's
  seeds (root 48,800). The statistic is M5's lane-exit readout, the arm's mean lead at turn 114.
- **E2: the limitation families.** These are M5c's cross-arm families conv, freeze, recall and gaps
  (`m5.cells`), at M5's sizes and horizons, on NC1's seeds (root 48,900).
  - The arm and the control both play seat A on the same rollouts. They face each of the two in seat B, and the
    readout averages over those two.
  - The readout is arm − control by cell group at M5's horizons: conv G50 and G150, freeze G50, recall G50 and G100,
    gaps G50 and G150.
  - The gaps family also carries the M4.6a gap specialist. Its margin over each final measures how much of the
    specialist's edge each network leaves.
- **E3, descriptive.**
  - Training health and learning curves.
  - Q1's unit usage on the 12 finals.
  - A ridge probe of the log duel ratio from the last hidden layer on held-out natural states (M4.6e §2.3), arm against
    control.
  - The critic's predicted strength premium at the E4 gap cells against the realized premium.

**Decision rules** (fixed now)

| result | condition |
|---|---|
| **improves play** | E1 is positive with a 95% interval excluding 0, and no E2 family's "all" group at its longest horizon is below −0.3 with an interval excluding 0 |
| **improves gap play** | E2 gaps "all" at G150 is positive with a 95% interval excluding 0 |
| **no material effect** | E1's interval includes 0 and lies within ±0.5 |
| **inconclusive** | anything else |

## Recommendation rule

- If an arm improves play, NC1 recommends a full-budget confirmation for it (32,768,000 transitions, paired with M5's
  sp finals, and attackers for exploitability) before any adoption. Adoption is the user's decision.
- If neither arm improves play and stage 1 finds 2 × 128 sufficient, NC1 documents that the network's size and input
  form do not limit the agent on this evidence, and the investigation stops there (stopping principle).
- Any other combination is reported as found, with the stage 1 / stage 2 agreement stated.

## Outside NC1's scope

- Training from scratch at other widths. It would not be comparable to a lineage with 57M transitions of training.
- Recurrent networks. M4.6e found the observation exactly decodable to the Markov state, and OM2's belief already
  carries the opponent's history.
- Activations other than tanh, and a factored action head.
- The OM arm in the PPO pilot. It trains about 2.3 times slower.
- The full strategic battery, and attackers on the pilot finals.

## Practical details

- **Code.**
  - The package is `lanegame/experiments/rl/nc1/`. It writes results to `results/nc1/` and data to
    `results/rl/checkpoints/nc1/` (git-ignored).
  - The library changes are in `lanegame/rl/network.py`, with tests in `tests/test_nc1.py`.
  - The package reuses `m5.train`, `m5.games`, `m5.cells`, `m5.strategic`, `m46a.e4` and `m46e.fit`.
- **Seeds.** NC1 draws from 48,000–48,999, a range no experiment uses.
  - W's new units use 48,000 + s.
  - Q1 states start at 48,100.
  - N train at 48,200, P train at 48,300, N test at 48,400 and P test at 48,500.
  - Fits and splits at 48,600, the natural behaviour check at 48,700.
  - E1 at 48,800 and E2 at 48,900.
- **Gate.** The `train` stage needs `--confirm-nc1-train`. The user approved the compute on 2026-10-29 in the words
  "one shot this … 8 hours is the ceiling, 4–6 hours is fine".
- **Branch.** The work stays uncommitted on `experiment/nc1-capacity`.
