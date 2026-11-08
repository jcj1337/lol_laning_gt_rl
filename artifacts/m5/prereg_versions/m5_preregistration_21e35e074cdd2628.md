# M5 pre-registration: the final training and evaluation campaign

*2026-10-17. Status: **frozen after the M5a readiness checks, amended once before any campaign training (amendment 1,
2026-10-17: [m5_amendment_1.md](m5_amendment_1.md), the M5a correction pass).** The chain of versions is in
`results/m5/preregistration.json`: the original record (SHA-256 `d2424022…af7304`, archived under
`results/m5/prereg_versions/`), each amendment's date, reason, replaced and new hashes, and its rationale document.
Every campaign stage validates the chain and runs only on its active version. Any further change is a dated amendment.
This document does not track the campaign's progress (see the design document and `results/m5/`).*

*Seen at freezing time: the readiness checks, the measured throughputs and costs, and pipeline outputs of stand-in
policies (OM2 and M4.7 finals) at reduced samples. They carry no information about M5's outcomes.*

*Design and rationale: [m5_design.md](m5_design.md). Code: `lanegame/experiments/rl/m5/`.*

## 1. Questions

1. Which M4.5c strategic findings replicate on the final policies of each arm?
2. What does the opponent-modelling package (arm A) add beyond continued pure self-play (arm B)?
3. Does arm A preserve competence against competent opponents, and how exploitable are both arms?

The primary contrast A − B estimates the **package effect**: the information (previous action and belief) and the
opponent distribution change together. It is neither ingredient alone, and neither arm isolates the archive-start
effect.

## 2. Arms and training

| item | A: `om` | B: `sp` |
|---|---|---|
| environment | `GameConfig.m45_calibrated(obs_last_action=True)` | `GameConfig.m45_calibrated()` |
| network | `PolicyValueNet` 116 → 128 → 128 tanh, 61 masked logits + value = `widen_inputs(ext/baseline:s, 116)` | `PolicyValueNet` 90 → 128 → 128 = a copy of `ext/baseline:s` |
| extra inputs | OM2's belief (11 library types, "mixed", "unknown"; λ 0.97, ε 0.05), reset at every game start; the library fixed | none |
| collector | `om2.collector.om_collector_cls("treatment")`: 50% current-policy self-play, 50% one of OM2's 15 fixed training opponents, uniform, the learner in a uniform seat, only the learner trains | `m46c.train.collector()`, 100% current-policy self-play (`current_fraction` 1.0; snapshots are saved but never played) |
| both | the M4.6c start mixture (β 0.532, archive games 150 turns, s0 games 512, mirroring ½); the M4.6c PPO settings; a fresh Adam at 3·10⁻⁴ annealed linearly to 0 over BUDGET; snapshots every 25,600 | |

- **Pairs.** s ∈ {0, 1, 2, 3, 4, 5}; both arms of pair s start from `ext/baseline:s`, the M4.6c 8.192M continuation of
  the M4.5c 16.384M final (24.576M prior transitions), and use PPO seed 70,000 + s.
- **Budget.** BUDGET = 32,768,000 additional agent transitions per run (12 runs, 393,216,000 in all).
- **Checkpoints.**
  - The **finals** (BUDGET) are the evaluated policies.
  - The **midpoints** are the first snapshot at or after 16,384,000.
  - The **75% snapshots** are the first at or after 24,576,000.

## 3. Health checks, failures and extensions

- **Health checks** (`m5.train.check`, all required):
  - finite statistics;
  - the budget reached within one update;
  - no termination, and one truncation per finished game;
  - |V| ≤ Ω;
  - no illegal action;
  - initial weights equal to the arm's start;
  - an archive transition share of 0.20–0.30;
  - the midpoint and 75% snapshots present;
  - none of M4.5c's stop conditions (KL > 0.05 sustained over 10 updates, a gradient norm > 100, entropy < 0.3 nats, an
    illegal or never-legal code);
  - sp: no fixed-opponent transition and equal seat counts;
  - om: a fixed-opponent game share of 0.45–0.55.
- **Runtime monitoring (amendment 1).** The stop conditions are watched during training, update by update
  (`m5.health.RunMonitor`, an opt-in trainer hook that only reads; a monitored run is the unmonitored run bit for bit).
  The first occurrence of each alarm is recorded with its evidence in the run's `health.jsonl`.
  - **Defect alarms** (a non-finite statistic, an illegal action sampled, a never-legal code offered) end the run at
    that update. Its history stays on disk, and the run is a technical failure until investigated.
  - **Investigate alarms** (sustained KL, gradient norm, entropy floor) are recorded, and the run continues to its
    budget. Such a run is kept as it is if the alarm turns out to be a property of learning, and it is restarted
    anyway if it is a defect.
  - The coordinator logs progress (`results/m5/progress_main.json` / `.jsonl`) and applies the failure rule below.
- **A crash or interruption.** The trainer stores no optimizer or collector state, so a run cannot resume mid-schedule.
  The run is restarted deterministically from its original start (`--restart-incomplete`), which repeats the lost work.
  It must reproduce the preserved history prefix exactly (`train.prefix_check`, enforced by the health check); if it
  does not, stop and investigate.
- **A failed health check or a stop condition.** Investigate.
  - An implementation defect is fixed and every affected run restarted: both runs of a pair when the defect touches
    both arms. It is reported.
  - A training outcome that is not a defect stays as it is and is reported. It is never replaced by another seed.
  - If more than two pairs fail for technical reasons, stop and report. The coordinator enforces this: the remaining
    runs are cancelled or terminated and the stop is recorded.
- **The evaluation gate (amendment 1).** No evaluation stage runs unless the training check passed, or every failed
  check has a recorded classification as a property of training (`train_resolve`), and every run is complete. Each
  stage also verifies that the checkpoints it evaluates are the files whose SHA-256 the check recorded, and records the
  identities of everything it reads.
- **No extension rule.** The 32.768M results are final. Non-convergence is reported as a limitation. An extension would
  need new approval and an amendment, would apply to both arms equally, and may not depend on the direction of a
  result.

## 4. Monitoring (not endpoints)

- **Learning curves:** each run's snapshots every 2.048M against `ext/baseline:s`, `ext/baseline:(s+3 mod 6)`,
  `g_farm`, `g_shove` and `g_bully`; 16 pairs × 150 turns; the lead at turn 114.
- **History:** each final against its 25%, 50% and 75% snapshots and against `ext/baseline:s`; 64 pairs × 500 turns;
  the lead at 114 and the discounted lead.
- **Convergence probe:** final − 75% snapshot (the final's lead over it). It is reported "still improving" when the
  seed mean is ≥ +0.25 with its interval above 0.
- Monitoring never selects a checkpoint and never changes an endpoint, a family or a threshold.

## 5. Value and games

- **Value:** a focal policy's undiscounted earned-gold lead at turn 114 in games from s0. This is the lane-exit readout
  E_20 at turn 94 in expectation, the readout the user adopted.
- **Reported beside it, never folded in:**
  - D0, the lead at turn 94;
  - the lane-exit panel at turn 94 (level and XP differences, HP, absence status, turns since return, unspent and
    invested gold, wave stocks).
- **The objective's horizon:** the long families add the discounted lead (γ 0.99) over 500 turns.
- **Games** (`m5.games.cell_job`):
  - Paired seats: every game is played with the focal in seat A and in seat B on the same chance stream.
  - Common random numbers across focals and arms.
  - Seeds: root · 10⁷ + cell · 10⁴ + game, with root 70,300 (families) or 70,200 (head to head).
  - Om policies sample with the uniform of default_rng([seed, 7, seat]) and update their belief after every turn, as in
    training. Fixed opponents act with default_rng([seed, 11, seat]).

## 6. M5d families (seed s; o = s + 3 mod 6; focals om:s and sp:s, plus `ext/baseline:s` as an exploratory reference)

| family | opponents | games |
|---|---|---|
| S_in | `g_farm`, `g_safe`, `g_bully`, `g_shove`, `fz_guard_sh`, `g_shove_ward` | 64 × 150 |
| S_out | `g_shove_sr`, `wk_shove_reset_sh`, `g_shove_edge`, `g_slow_guard`; `style:respect:o`, `style:passive:o`, `style:stepback:o` | 64 × 150 |
| M | `mix_bully_farm`, `mix_shove_safe`, `switch_safe_shove_30`, `switch_bully_farm_50`; `its:g_farm:40:o`, `its:g_safe:20:o`, `its:g_shove:40:o` | 64 × 150 |
| C_hist | `base:p` for the five p ≠ s; `attacker_b`, `freezer_b`, `gapper_b`, `short_b`, `c1_policy` | 64 × 150; 32 × 500 |
| C_cross | `om:p` and `sp:p` for the five p ≠ s | 64 × 150; 32 × 500 |
| H2H | om:s vs sp:s | 128 × 150; 32 × 500 |
| H2H cross-seed | om:s vs sp:p, p ≠ s | 32 × 150 each |
| mid | the midpoints of om:s and sp:s on S_in, S_out, C_hist; midpoint head to head | 32 × 150 |

- `style:x:o` is the M4.4c conditional style x on the competent policy `ext/baseline:o`.
- `its:X:t:o` plays archetype X while the turn index is below t, then `ext/baseline:o`.

**Family value of seed s:** the mean over the family's opponents (equal weights) of the per-opponent mean over game pairs
of om − sp. H2H is om's own lead.

## 7. M5c strategic evaluation

- **The M4.5c battery** (`m5.battery`): every family, candidate, protocol, horizon and exception of `m45c.battery`, on
  each final.
  - Natural states come from the final's own self-play (states game seeds 704,000 + …; rollouts 7,045,000 + r).
  - Om natural states carry their recorded beliefs; constructed cells start at the prior.
  - The primary statistic is M4.5c's: Q50 in natural families, G50 in constructed ones.
- **The M4 limitation families** (`m5.cells`), rollouts only, no critic:
  - weaker side (K 32, H 50) and the lead premium J1 (K 64, G20 / G50) on each final;
  - cross-arm conv (K 16, H 150), freeze (K 32, H 50), Recall windows (K 32, H 100), natural states of
    `ext/baseline:s` (K 16, H 150; freezes H 50) and large gaps with a gap specialist (K 16, H 150), per seed.
- **Replication rule (per arm):** a natural-family verdict holds in a seed when no fixed alternative beats the policy by
  more than 0.2 (the family mean of alternative − policy at the primary statistic). It replicates with ≥ 5 of 6
  seeds.
  - The verdicts: conditional wave management (A_wave), conditional freezing and the reply to freezes (B_freeze,
    B_frozen), selective trading (C_trade), concession under threat (H_zone_c, and the weaker-side cells), Recall
    timing (E_recall).
  - Every M4 closeout class is re-measured and labelled used / missed / weak / open by the closeout's definitions.
    There are no target values.
- **The statistic, fixed by amendment 1** (`m5.strategic`):
  - A battery family's gap in a seed is the largest family mean of (alternative − policy) over its fixed alternatives
    (one-turn candidates and sustained plans).
    - It is measured under protocol **"policy"**, the policy's own sampling play. The scripted protocols measure value
      against conceding styles and are reported only.
    - It uses the primary statistic at 50 turns: Q in natural families, G in the constructed H_zone_c.
  - The weaker-side gap is the larger, over the +8 and +16 leads, of the best forced weak-side response (hold, safe,
    trade, reset) minus the policy at G50 over every weak-focal cell (M4.6c §5.1).
  - A verdict holds in a seed when every one of its families holds. The verdicts are not in the Holm family.
- **Completeness and coverage.** A family's value needs every expected (state, protocol) record. A seed with missing
  records is unknown, never counted as holding; the verdict stays "partial" unless the missing seeds cannot change it.
  A natural family whose state collection found fewer states than designed is analysed on the states it has and
  flagged.
- **Horizon validation** (M4.5c's rule, applied to each policy and family on its 300-turn subset):
  - pass: the 50-turn statistic stands;
  - fail: the registered handling is M4.5c's exception 3, a re-run of that family at 150 turns for all its states and
    protocols. Until it exists the family, and every verdict using it, is "pending"; then the 150-turn statistic
    replaces the 50-turn one;
  - fewer than 3 validation pairs: used and flagged "unvalidated".

## 8. Attackers

*Amendment 1: the attacker stage still awaits its own approval. Its **core** is the 12 attackers against `om:s` and
`sp:s`. The 6 attackers against `om_table:s` are an **optional follow-up** needing a separate decision; they are
exploratory and outside the Holm family (§9).*

- **Targets and starts.** 12 core attackers, plus 6 optional:
  - core: 6 against `om:s` (belief live) and 6 against `sp:s`;
  - optional: 6 against `om_table:s` (the om final with its belief inputs replaced by the per-turn mean belief of its
    own self-play: 32 games × 512 turns, seeds 70,750 · 10⁷ + g);
  - each starts as a copy of its target in the 116-input om format (an sp target widened with zero weights);
  - each observes the state, the target's previous action and its own OM2 belief about the target.
- **Reading the optional ablation.**
  - om − om_table compares two approximate lower bounds, so a difference can come from what each attacker learned,
    not only from the targets.
  - Removing the live belief can change the target's competence, and it creates state-belief combinations the network
    never saw in training.
  - It does not isolate a causal adaptation vulnerability, and the attackers are stateless, so it is not evidence of
    deliberate deception.
  - If run, it includes the live/table cross-evaluation: each attacker against the other target form, on the same
    games as the other attacker's own target cell.
- **Training:**
  - s0 starts, 512-turn games, the M4.6c PPO settings;
  - PPO seed 70,600 + 10 · kind + s (kind 0 om, 1 sp, 2 om_table);
  - budget 8,192,000 transitions per attacker: a quarter of the target's M5 continuation budget of 32,768,000.
    The 24,576,000 prior transitions are shared by both arms' targets and inherited through the parity start, so they
    are not in the denominator. The alternative, half (16,384,000), is a user decision and would be recorded as an
    amendment before any attacker trains;
  - the plain zero-sum reward, with no deception term.
- **Evaluation:** each attacker against its target, the other arm's final and `ext/baseline:s`; 128 pairs × 500 turns;
  the lead at 114 and the discounted lead.
- **Exploitability** of a target = its attacker's value against it, a lower bound.

## 9. Endpoints and classification

**Intervals:** seeds are the units; 95% t intervals with df 5 over the six per-seed values; [L, U] is the interval.
Each seed's game-level SE and the split of the between-seed variance (the game part, and τ² = max(0, s² − mean SE²))
are reported for every family.

**Completeness (amendment 1).** An endpoint is classified only when the records of all six seeds exist. Otherwise it
is reported as "partial (n/6)" with its missing records listed and no class. A missing record never shrinks the
replication count or the multiplicity family. The headline needs P1, P2a and P2b complete.

**Primary endpoints** (finals, lead at 114):
- **P1, exploitation gain:** the seed values ½ S_in + ½ S_out. Demonstrated when mean ≥ +0.25 and L > 0. The other
  classes (amendment 1 splits the first one by U; thresholds unchanged):
  - detected, below materiality: L > 0 and U < 0.25 (a material gain ruled out);
  - detected, materiality unresolved: L > 0, mean < 0.25 ≤ U;
  - loss: U < 0 (a material gain ruled out);
  - no material gain: U < 0.25 and L ≤ 0 (a material gain ruled out);
  - inconclusive: L ≤ 0 and U ≥ 0.25 (a material gain neither shown nor ruled out).
- **P2a, competence against competent opponents:** ½ C_hist + ½ C_cross.
- **P2b, competence head to head:** H2H (om:s vs sp:s, 128 pairs).

P2a and P2b are classified by non-inferiority with margin δ = 0.25:
- preserved: L > −δ ("a small loss detected" when also U < 0);
- material loss: U < 0 and mean ≤ −δ;
- loss, materiality unresolved: U < 0;
- inconclusive: otherwise. Insufficient precision is reported as inconclusive, never as preserved.

**Competence status** (reported with every outcome): preserved (P2a and P2b preserved); material loss (either); loss
detected, materiality unresolved (either); otherwise unresolved.

**Headline** (amendment 1: an inconclusive P1 is never "no material gain"):

| outcome | condition |
|---|---|
| 1 pays and preserves competence | P1 demonstrated; competence preserved |
| 2 pays at a competence cost | P1 demonstrated; a material competence loss |
| 3 pays, competence unresolved | P1 demonstrated; otherwise |
| 4 no material gain | P1 rules out a material gain (no material gain, loss, or detected below materiality); no material competence loss |
| 5 harmful | P1 not demonstrated; a material competence loss |
| 6 gain not demonstrated | P1 inconclusive or detected with materiality unresolved; no material competence loss |

Style gains and competence results are always reported side by side, never netted. Outcome 1 needs no adjustment,
because each part must meet its own interval criterion. A material loss in either of P2a and P2b is two one-sided
tests at 0.025, within 0.05 together. The headline covers P1, P2a and P2b only. It does not imply anything about
strategic replication (S8), the limitation cells or exploitability (S6), which are reported with their own status.

**Secondary endpoints:**

| id | endpoint | classification |
|---|---|---|
| S1 | M (misspecified opponents, imitate-then-switch included), om − sp | non-inferiority, δ 0.25 |
| S2 | long-horizon competence: ½ C_hist + ½ C_cross at 500 turns, discounted | non-inferiority, δ 0.25 |
| S3 | long-horizon head to head, discounted | non-inferiority, δ 0.25 |
| S4 | cross-seed head to head, om:s vs sp:p (mean over p) | non-inferiority, δ 0.25 |
| S5a | midpoint (16.384M) exploitation gain: ½ S_in + ½ S_out of the midpoints, om − sp | as P1 |
| S5b | midpoint (16.384M) competence against the **historical** competent opponents only (C_hist), om − sp. Not P2a: the cross-arm finals are not played at the midpoint | non-inferiority, δ 0.25 |
| S6 | exploitability, om − sp (each core attacker's lead at turn 114 against its own target; the discounted 500-turn lead beside it) | increase demonstrated: L > 0 and mean ≥ 0.25; decrease demonstrated: U < 0 and mean ≤ −0.25; no material difference: L > −0.25 and U < 0.25; otherwise inconclusive |
| S8 | strategic replication per arm (§7) and om − sp in the cross-arm M5c families | §7; the cross-arm contrasts descriptive with intervals; not in the Holm family |

*Amendment 1:* the midpoint head to head is descriptive. The former S7 (the adaptation test, exploitability of `om`
minus that of `om_table`) is exploratory, because its attackers are an optional follow-up (§8).

**Tests and multiplicity (amendment 1).** Every permitted claim of S1–S6 is a one-sided seed-level t test (df 5),
fixed now, independent of the results:
- S1, S2, S3, S4, S5b: "preserved" (H0 μ ≤ −δ) and "loss" (H0 μ ≥ 0);
- S5a: "gain" (H0 μ ≤ 0) and "material gain ruled out" (H0 μ ≥ 0.25);
- S6: "increase" (H0 μ ≤ 0), "decrease" (H0 μ ≥ 0) and "equivalence" (two one-sided tests of H0 |μ| ≥ δ; p is the
  larger one-sided p).

These **15 hypotheses form one Holm family** at familywise 0.05. A secondary claim is made only when its interval
condition holds and Holm rejects its hypothesis; the unadjusted interval class is reported beside it. A class is
assembled from the supported claims only (for example, "material loss" needs "loss" supported and mean ≤ −δ).
- Holm is final only when every hypothesis has a p-value.
- Before that, a claim is "secured" when p ≤ 0.05 / 15, since Holm rejects it whatever the pending p-values are.
  Every other claim is "pending".
- If the user declines a stage, its hypotheses enter with p = 1, which keeps the error control.

**Exploratory, not pre-registered as tests:**
- each arm against `ext/baseline:s` (the common start);
- per-opponent values; the lane-exit panel;
- the post-switch window of the imitate-then-switch games;
- the belief mechanism (identification, catch-all mass, switch tracking, belief-swap probes);
- attacker transfer;
- the optional adaptation ablation, om − om_table (§8), with its live/table cross-evaluation;
- the midpoint head to head;
- the battery's ledger channels.

## 10. Resolution, stated in advance

With six seeds and a per-seed SD like OM2's competent-opponent contrast (0.28–0.33, about 40% of it game sampling),
the P2 rule calls:
- a true loss of 0.5 a material loss with probability about 0.93;
- a true +0.25 preserved with probability about 0.93;
- a true null preserved with probability about 0.42, otherwise mostly inconclusive.

An inconclusive competence result is therefore a likely outcome of a near-zero effect, and it will be reported as such.

## 11. Known limitations

- **The data.** Continuations, not fresh training: three schedules and three learning-rate re-warms in the lineage.
  One opponent library and one mix. Six pairs.
- **The tests' scope.**
  - Library-favourable style families. Competent opponents come from the same lineage.
  - Attackers are stateless and budget-limited: lower bounds only.
  - The scheduled imitate-then-switch opponent is not a deceptive learned attacker.
- **The contrast.** A − B is a package contrast at equal agent transitions. A plays about 36% more environment turns
  and gets about a third fewer self-play transitions.
- **Coverage.** Values beyond 500 turns are not measured.

## 12. Amendments

| # | date | document | summary |
|---|---|---|---|
| 1 | 2026-10-17, before any campaign training | [m5_amendment_1.md](m5_amendment_1.md) | the M5a correction pass: the headline no longer maps an inconclusive gain to "no material gain" (outcome 6), P1's classes split by U, fixed tests and one 15-hypothesis Holm family for S1–S6 with an interim rule, S5 renamed (S5a, S5b, descriptive midpoint head to head), six-seed completeness, the S8 statistic and validation path, runtime health monitoring with the evaluation gate, the core / optional attackers, the verifiable amendment chain. No threshold, arm, budget or family changed |
