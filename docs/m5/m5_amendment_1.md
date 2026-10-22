# M5 pre-registration, amendment 1: the M5a correction pass

*2026-10-17. Recorded before any M5 campaign training, as a dated amendment of
[m5_preregistration.md](m5_preregistration.md). The original frozen text (SHA-256 `d2424022…af7304`) is archived byte
for byte at `results/m5/prereg_versions/m5_preregistration_d2424022bd07535f.md`. The chain of versions, with this
document's hash, is in `results/m5/preregistration.json`. Every campaign stage validates the chain and refuses to run
on any other version of the document.*

## Why

A review of M5a (at commit `82568827`) found the following defects in the frozen protocol and its code. None of them
concerns the experiment itself.

1. The headline mapped an inconclusive exploitation result to "no material gain".
2. The secondary endpoints S6/S7 allowed claims of an increase, a decrease and equivalence, but supplied only a one-sided
   increase p-value to Holm.
3. The midpoint competence endpoint was called "P2a at 16.384M". The analysis compares the midpoints only with the
   historical opponents, while P2a also includes the cross-arm finals.
4. Missing records could silently shrink the replication count and the Holm family.
5. S8 (strategic replication and the cross-arm M5c contrasts) had no analysis code.
6. The training stop conditions were checked only after every run had finished. A failed check did not prevent the
   evaluation stages from running. The technical-failure policy was not enforced by the coordinator.
7. A recorded amendment was never checked: `require_prereg()` validated only the original hash.
8. The design promised a live/table cross-evaluation for the optional adaptation ablation, but no job generated it.

The user authorized this correction pass together with the main training and the opponent-family evaluations (the
brief of 2026-10-17, §1). Under that authorization the corrections must not change the experiment.

**Seen when this amendment was written:** the same material as at freezing, namely the readiness checks, the measured
costs and stand-in pipeline outputs, plus the review findings. No M5 campaign result existed: no M5 policy had been
trained or evaluated.

## What did not change

- The two arms, their environments, networks, collectors and starts. The six pairs and PPO seeds 70,000 + s.
- The budget (32.768M additional agent transitions per run), the midpoint, the 75% snapshot and the PPO settings.
- The mechanics, the reward, γ, continuing time, truncation and the architecture.
- The opponent families, the game counts, the seeds, the value readout (the lead at turn 114) and the panel.
- Every threshold: GAIN_MIN = 0.25 for P1 / S5a, δ = 0.25 for every non-inferiority and exploitability rule, the
  replication margin 0.2 with 5 of 6 seeds, and familywise α = 0.05.
- The strategic battery and the limitation cells: their design, sizes and horizons. Their stages still await approval,
  and the horizon breakdown below is input for that review, not a change.
- No result was seen, so no rule was chosen with knowledge of a result.

## What changed

| # | item | before | after | effect |
|---|---|---|---|---|
| 1 | P1 classes | "detected, below materiality" whenever L > 0 and the mean < 0.25 | split by U: **below materiality** (U < 0.25, a material gain ruled out) and **materiality unresolved** (U ≥ 0.25) | "demonstrated" is unchanged; the classes now say whether a material gain is ruled out |
| 2 | headline | outcome 4 "no material gain" whenever P1 was not demonstrated and no material loss occurred, including inconclusive P1 | outcome 4 needs a material gain **ruled out** (U < 0.25: classes no material gain, loss, below materiality). New **outcome 6, "gain not demonstrated"**, covers inconclusive P1 and materiality unresolved. The competence status (preserved / material loss / loss detected, materiality unresolved / unresolved) is reported in every outcome, and the headline states its scope (P1, P2a, P2b only) | an inconclusive result stays inconclusive. Outcomes 1, 2, 3 and 5 are unchanged |
| 3 | secondary tests | "one-sided seed-level t tests in the stated direction", Holm over the claims of "demonstrated"; only an increase p-value for S6/S7 | every permitted claim is a fixed one-sided t test (df 5): non-inferiority endpoints have "preserved" (H0 μ ≤ −δ) and "loss" (H0 μ ≥ 0); S5a has "gain" (H0 μ ≤ 0) and "material gain ruled out" (H0 μ ≥ 0.25); S6 has "increase" (H0 μ ≤ 0), "decrease" (H0 μ ≥ 0) and "equivalence" (two one-sided tests, H0 \|μ\| ≥ δ). **One Holm family of 15 hypotheses**, fixed now, at familywise 0.05. A claim needs its interval condition and Holm's rejection; the unadjusted interval class is reported beside it | no direction is chosen after the results. An interval bound and its one-sided test agree exactly at 0.025 (tested) |
| 4 | partial families | (unspecified) | Holm is final only when every hypothesis has a p-value. Before that, a claim is **secured** when p ≤ 0.05 / 15 (Holm rejects it whatever the pending p-values are) and pending otherwise. A stage the user declines enters with p = 1, which keeps the error control | S6 (attackers) is not authorized in this stage, so S1–S5 claims stay interim until the attacker decision is made |
| 5 | S5 | "the midpoints: P1 and P2a at 16.384M" | **S5a**: midpoint exploitation gain (½ S_in + ½ S_out, as P1). **S5b**: midpoint competence against the **historical** competent opponents only (C_hist), named as such: it is not P2a, because the cross-arm finals are not played at the midpoint. The midpoint head to head is reported as **descriptive** | the label now says what is computed |
| 6 | S7 | the adaptation test as a secondary endpoint (6 om_table attackers) | **exploratory and optional**, outside the family (see #10) | the family no longer depends on an optional stage |
| 7 | completeness | the analysis skipped a seed with missing records | an endpoint is classified only with all six seeds. Otherwise it is "partial (n/6)" with the missing records listed and no class. The headline needs P1, P2a and P2b complete. A replication verdict counts a missing seed as unknown, never as holding | no silent shrinking |
| 8 | S8 | "§7; the cross-arm contrasts descriptive with intervals" | the replication statistic is fixed: protocol **"policy"**, the primary statistic at 50 turns (Q natural, G constructed), max over the family's fixed alternatives. For "concession under threat" it also covers the weaker-side cells' best forced response at G50. A verdict holds in a seed when each of its families does. The handling of a failed horizon validation is fixed (M4.5c's exception 3: re-run at 150 turns, pending until then). Missing records give "partial"; a short state collection is flagged. S8 stays outside the Holm family | the S8 analysis exists and is tested on synthetic records |
| 9 | runtime health | the stop conditions checked after training | M4.5c's stop conditions are watched **during** training, update by update, by an opt-in trainer hook that reads only (a monitored run is bit-identical, tested). **Defect** alarms (a non-finite statistic, an illegal action, a never-legal code) end the run at once. **Investigate** alarms (sustained KL, gradient norm, entropy floor) are recorded, and the run continues to its budget, since such a run is kept "as it is" if the alarm is a property of learning. The coordinator stops the campaign when more than two pairs fail technically. Every evaluation stage refuses to run until the training check passes or each failed check has a recorded classification, and it verifies that the evaluated checkpoints are the ones the check recorded. A restart must reproduce its preserved prefix, and the check enforces it | failures can no longer slip through to evaluation. Healthy runs are unchanged |
| 10 | attackers (future stage, not authorized here) | 18 attackers: 6 om, 6 sp, 6 om_table | the **core** stage is **12 quarter-budget attackers** (6 against the om finals, 6 against the sp finals, 8.192M each), still awaiting its own approval. The **6 om_table attackers are an optional follow-up** needing a separate decision. Their reading is limited: om − om_table compares two approximate lower bounds, so a difference can reflect what each attacker learned. Removing the live belief can change the target's competence and creates state-belief combinations never seen in training. The ablation does not isolate a causal adaptation vulnerability and is not evidence of deliberate deception. The live/table cross-evaluation is now implemented and tested as job definitions | core estimate 6.6 h of training + 0.5 h of evaluation; the optional addition +3.7 h + 0.03 h (tables) + 0.4 h |
| 11 | the amendment mechanism | `amend()` recorded a note that no gate checked | a verifiable chain: the original record kept, every version archived, each amendment with its date, reason, the hash it replaces, the new hash, its own document's hash and its authorization. The gate validates the active version and rejects any undocumented change (tested) | — |

## Stage authorization (2026-10-17, the user's brief)

- Authorized: this correction pass; the main training (`--confirm-m5-train`); training health; monitoring and history
  (`monitor`); the opponent-family and head-to-head evaluations including the midpoint and long games (`families`);
  analysis, reporting and documentation for those stages.
- Not authorized: `battery`, `cells`, any attacker stage, belief tables or attacker evaluation, extra seeds, arms or
  extensions, and any change of mechanics, reward, discount, architecture or hyperparameters.

## The future battery: work by horizon (for the review after this stage, no change made)

Measured R10 rates, six workers, all twelve finals (`python -m lanegame.experiments.rl.m5 plan_future`):

| part | purpose | horizon | wall h |
|---|---|---|---|
| battery | standard continuations | 50 | 4.45 |
| battery | exception 1: sustained freeze plans (plan length + 40) | 60 | 1.65 |
| battery | validation subset extended from the family horizon to 300 turns | 300 | 3.23 |
| battery | contingent: exception 3 re-run of a family that fails validation | 150 | 0.3–3.9 per family on all twelve finals; 17.5 if every family failed everywhere |
| cells | weaker side, lead premium, freezes, natural freezes | 50 | 0.53 |
| cells | Recall windows | 100 | 0.16 |
| cells | strength use, large gaps, natural leader / trailer states | 150 | 0.81 |

The 300-turn validation subset is the largest exception: about a third of the battery's cost. The design document
(§10.1) sets out what each exception buys, what reducing it would lose, and which checks only duplicate M4.5c's
evidence. The decision is left to the review after this stage.
