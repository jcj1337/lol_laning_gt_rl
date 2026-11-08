# OM2 pre-registration: the matched Bayesian opponent-modelling pilot

*2026-10-17. Frozen (SHA-256 in `results/om2/preregistration.json`) after the readiness smoke and before any pilot
training run. Seen at freezing time:*
- *the readiness checks and throughputs (`results/om2/smoke.json`: 17 of 17 checks pass);*
- *a pipeline smoke of the evaluation and the probes on 6-update smoke networks (2 game pairs of 40 turns, one seed).
  Its numbers carry no information about the pilot.*

*Design: [om2_design.md](om2_design.md). Feasibility: [OM1](om1_feasibility_report.md).*

## 1. Question

Can the agent learn to adapt to who it is playing, exploiting recognizable styles, without becoming worse against
competent or unfamiliar opponents? The scope is the user's option C, robustness first. Specifically: does a belief about
the opponent, updated from its actions, add value beyond the opponent's last action, when both policies are trained on
the same data stream?

## 2. Arms

| item | control | treatment |
|---|---|---|
| environment | `m45_calibrated(obs_last_action=True)`: the adopted baseline plus the opponent's previous action (103 entries) | the same |
| network input | 103 + 13 constant entries (the belief's prior) | 103 + the belief's posterior over 11 library types, "mixed" and "unknown" |
| network | `PolicyValueNet` 116 → 128 → 128 tanh, 61 masked logits + value; initialized from `ext/baseline:s` with zero weights on the 26 new inputs, so both arms start as exactly that policy | the same |
| training | M4.6c PPO settings; M4.6c start mixture (512-turn games from s0, 150-turn games from archive states, a quarter of the transitions); fresh Adam at 3·10⁻⁴ annealed to 0; PPO seed 61,000 + s | the same |
| opponent mix | 50% current-policy self-play (both seats train); 50% a fixed opponent, uniform over 15: the 6 archetypes, the 5 learned library policies, 2 per-turn mixtures, 2 style switchers (only the learner trains) | the same draws, in the same order |
| belief | none (constant prior) | exact likelihoods, ε = 0.05 for scripted types, forgetting λ = 0.97, updated every turn from the opponent's action |

- **Seeds:** s ∈ {0, 1, 2}; 6 runs.
- **Budget:** 4,096,000 agent transitions each, with snapshots every 25,600.

**Measured readiness throughput** (3 concurrent runs): control about 2,650 transitions/s, treatment about 1,580/s. With
5 concurrent runs on this machine, expect roughly 60 minutes per treatment run and 36 per control run: **about
1.25–1.5 h of training** wall time. The evaluation adds about 15–20 minutes.

**No extension rule.** If the result is inconclusive, it is reported as such and nothing more is trained under this
protocol.

## 3. Evaluation (`om2/evaluate.py`)

**Games.**
- Each arm's final plays every opponent from s0.
- **Seats:** paired, each game played with the arm in seat A and in seat B on the same chance stream.
- **Pairs:** 64 per cell, 150 turns per game.
- **Common random numbers:** both arms meet each opponent on the same games and streams. Each arm samples with CRN
  uniforms and updates its belief exactly as in training.
- **Seeds:** 61,100·10⁷ + cell·10⁴ + game.

**Value:** the arm's earned-gold lead at turn 114, the adopted lane-exit readout E_20 at turn 94 in expectation. The
lane lead at turn 94 and the lead at turn 150 are descriptive.

| family | opponents |
|---|---|
| **E_in** (exploitation, in the library) | the 6 archetypes: g_farm, g_safe, g_bully, g_shove, fz_guard_sh, g_shove_ward |
| E_out (descriptive) | held-out scripted variants: g_shove_sr, wk_shove_reset_sh, g_shove_edge, g_slow_guard |
| **R1** (competent) | the adopted baseline policies of seeds 3, 4, 5 (`ext/baseline:o`); held-out learned instances: attacker ext_s0, freeze specialist s4, gap specialist s4, M4.7 short s1, C1 policy `ext/C1:2` |
| **R2** (misspecified) | held-out mixtures (bully/farm, shove/safe, per turn 50/50); held-out switchers (safe → shove at turn 30, bully → farm at turn 50) |
| **R3** (head to head) | the treatment against the control of the same seed |

**Contrast.** For each opponent and seed: treatment − control, the mean over the cell's pairs of the paired difference.
A family's value for a seed is the mean over its opponents. R3 is the treatment's own lead.

## 4. Criteria and outcomes

- **E (gain):** E_in's mean over the three seeds is ≥ +0.25, **and** E_in is positive in every seed.
- **R (loss)**, for each of R1, R2, R3: the family's mean is ≤ −0.25, **and** it is negative in every seed.

| outcome | condition | recommendation (nothing is adopted automatically) |
|---|---|---|
| **1** modelling pays and is safe | E, and no R | carry Bayesian opponent modelling into the M5 design (its exact form a user decision) |
| **2** pays, but costs | E, and at least one R | a trade-off for the user: the gain against styles against the cost named by R |
| **3** no material gain | not E, and no R | do not add opponent modelling to M5 as designed; the last-action observation alone is a separate, untested question |
| **4** harmful | not E, and at least one R | do not add it |

Descriptive only (cannot change the outcome):
- E_out; the pooled-game intervals; the lane lead and the lead at turn 150; per-opponent values;
- the treatment's identification of the archetypes (MAP after 10 / 20 / 40 / 94 turns);
- the catch-alls' posterior mass against R2;
- belief-swap probes: the total-variation distance between the trained policy with a one-type belief and with the
  prior, on the treatment's own observations. The control's distance is 0 by construction: its constant inputs give
  all 13 columns the same gradient.

## 5. Known limitations

- **Three seed pairs.** Seed-level intervals have 2 degrees of freedom, so the criteria require the seeds to agree.
- **Continuation, not fresh training.** Both arms continue a policy trained for 24.6M transitions without opponent
  information, with a fresh learning rate.
- **4.096M transitions is a pilot budget.** A null result bounds this budget, not the idea.
- **One library and one opponent mix.** The value depends on the type distribution: 9 of the 15 training opponents
  are scripted or built from scripts.
- **Opponent information.** The observation reveals the opponent's previous action. That is an assumption of
  decision 2, which the user approved.
- **Proxies.** Exploitability by a dedicated attacker trained against the treatment is not measured; R3 is the
  pilot's proxy.
