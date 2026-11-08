# OM2 design: Bayesian opponent modelling

*2026-10-17. Status: **proposed design, implemented behind opt-in switches, awaiting review.** The readiness checklist
(§8) is complete. The pilot, frozen in [om2_preregistration.md](om2_preregistration.md), ran on 2026-10-17:
[om2_pilot_report.md](om2_pilot_report.md) (outcome 1, modelling pays and is safe).*

**Approved by the user (2026-10-17):**
- the direction (design, then a matched pilot, before M5);
- scope C, "both, robustness first";
- the opponent's last action as an opt-in observation field.

Everything else here is a proposal. No default, preset, objective or existing checkpoint changes.

Inputs: [OM1 feasibility](om1_feasibility_report.md); the original plan's §H (roadmap); the adopted M5 setup (baseline
`m45_calibrated()`, γ 0.99, shared critic, lane-phase outcomes read with E_20).

## 1. Goal and the question the pilot answers

The project asks whether an RL agent learns to exploit laning's interactions without being taught League strategy.
Opponent modelling extends that to the opponent itself. **Can the agent learn to adapt to who it is playing,
exploiting recognizable habits, without becoming worse against competent or unfamiliar opponents?** This is the
exploitation-versus-robustness trade-off of the original M5 plan. OM1 showed it is real in this game:
- identification is fast;
- knowing the type is worth about a wave at the lane exit;
- acting on a wrong model costs up to 1.5.

Scope C (the user's choice):
- the agent models both recognizable archetypes and competent learned opponents;
- it trains against mixed and style-switching opponents;
- success needs a gain against archetypes **and** no loss against competent or unfamiliar opponents, reported
  separately.

What never changes:
- The reward is the gold difference only, and the objective stays γ 0.99 with continuing time.
- The agent's own strategy is never scripted; scripts appear only as opponents.
- The adopted baseline and E_20 stay as they are.

## 2. What the agent observes (decision 2: the opponent's last action)

**New opt-in `GameConfig` field `obs_last_action`** (default `False`: every preset, digest pin and checkpoint is
unchanged). When on, each seat's observation gains 13 entries after the existing 90, encoding the action the
**opponent** chose on the previous turn:

| entries | meaning |
|---|---|
| none | no previous turn yet (game start, or a restored state without a recorded action) |
| absent, ward, recall | the opponent was away, placed a ward, or recalled |
| hold, yield, trade, safe | the behaviour of a lane action |
| last hit, thin, slow push, shove | its wave slot |
| pressure | its objective was Pressure |

- **The format is 90 + 13 = 103 entries.** The first 90 are unchanged, so a 90-input network reads the prefix and plays
  exactly as before. The opponent library's learned types rely on this.
- **The environment remembers each seat's last action.** `reset` sets it to "none"; a restored state may pass it.
- **Nothing is hidden or revealed early.** It is the previous turn's action, which both players saw happen. The
  simultaneous move of the current turn stays unseen.
- **Consequence:** policies trained with the field have 103+ inputs and cannot be loaded into the 90-entry observation,
  or the reverse. That is why it is opt-in. The mathematical reference documents the variant (§2.4, §13).

## 3. The belief

A Bayesian posterior over a finite library of opponent types, updated after every turn from the opponent's observed
action. It is a generalization of OM1's, as a reusable module `lanegame/opponent/`:
- **Types and likelihoods:**
  - a learned type contributes its exact action probability;
  - a scripted type trembles: (1 − ε) on its action, ε = 0.05 spread over the legal codes;
  - **two catch-alls**, the misspecification fallback OM1 found missing:
    - **"mixed"**: a different library type each turn. Its likelihood is the mean of the library types' probabilities
      of the observed action. It explains a per-turn mixture of styles better than either style alone, while a
      consistent style beats it.
    - **"unknown"**: every legal action equally likely, for play unlike any library type.

  *Correction during implementation (2026-10-17).* The first version had only the uniform catch-all. Its test showed
  that a uniform catch-all does **not** catch per-turn mixtures. With about 30 legal actions, a scripted type that is
  right half the time is still far better than chance, so it beats "everything equally likely", which is why OM1's
  belief split between the two mixed styles. The "mixed" catch-all fixes this; the tests check that a mixture goes to
  "mixed", random play to "unknown", and a consistent style to itself.
- **Forgetting.** The log posterior decays toward the prior by a factor λ = 0.97 per turn: an effective memory of about
  33 turns, which lets the belief follow an opponent that switches style. λ = 1 is exact Bayes; OM1 identified most
  types within 10–20 turns, well inside this memory.
- **Uninformative turns.** A turn in which the opponent can only be Absent carries no information.
- **Interface:** `Belief.update(log-likelihoods of the observed action)` and `Belief.features()`. The features are the
  posterior over the 11 library types and the two catch-alls, appended to the policy's input.

**Library for the belief (11 types + "mixed" + "unknown" = 13 features)**, chosen from OM1:
- **Archetypes** (scripted, 6):
  - `g_farm`, passive last-hitting;
  - `g_safe`, defensive;
  - `g_bully`, aggressive trading;
  - `g_shove`, perma-shove;
  - `fz_guard_sh`, freeze-oriented;
  - `g_shove_ward`, vision-oriented shoving.

  The other Shove variants are left out: OM1 showed they are look-alikes with the same best response.
- **Learned** (5): peer (the M4.5c 8.192M final of seed 3), attacker (M4.5c attacker ext_s3), freezer (M4.6a freeze
  specialist s2), gapper (gap specialist s2) and short (M4.7 H 60 arm s0). The immature "early" policy is dropped.

## 4. The policy

- **Input:** 103 (state + last action) + 13 (belief features) = **116**. The architecture is unchanged otherwise: the
  shared `PolicyValueNet`, 128 → 128 tanh, 61 masked logits plus a value. There is no recurrence; the belief carries
  the history.
- **Initialization from the adopted baseline policy `ext/baseline:s`:** the 90 existing input columns are copied and
  the 26 new columns are zero. At the start the network computes exactly π_s's outputs, so both pilot arms begin as the
  same policy.
- **Acting on the response rather than the type** (OM1's first lesson) needs no special rule. A trained policy that sees
  the whole posterior learns which mixtures of types call for which play.

## 5. Training (both pilot arms identical except the belief input)

- **Trainer.** The shared-policy PPO of every M4 campaign, plus two opt-in hooks in `SharedCollector`:
  - `augment` (extra policy inputs for a learner seat);
  - `observe_step` (after each step).

  Both default to doing nothing, which is bit-identical training. The opponent-modelling collector uses the hooks to
  update one belief per learner seat and to append its features.
- **Settings.** The M4.6c PPO settings (entropy 0.03, 16 × 128-turn segments, λ 0.95, reward scale 0.1, lr 3·10⁻⁴
  annealed) and the M4.6c start mixture (archive starts, 512 / 150-turn games), as in the M4.6c and M4.7
  continuations.
- **Opponent mix (the same in both arms):**
  - 50% of games: current-policy self-play, both seats training. This keeps the competence that M1–M4 built.
  - 50% of games: a fixed opponent drawn uniformly from 15 training opponents:
    - the 11 library types;
    - two per-turn mixtures, bully/safe and shove/farm;
    - two style switchers: farm → bully at turn 40, and shove → safe at turn 60.

    Only the learner seat trains in these games, as with M4's historical opponents.
  - The mixtures and switchers are deliberately **not** belief types: they teach the policy what to do when the belief
    says "unknown" or moves.
- **Arms:**
  - **Control:** obs_last_action on; its 13 belief inputs are held at the prior. It has the same network and data
    stream but no belief information.
  - **Treatment:** obs_last_action on; the belief's features.

  The contrast isolates what the belief adds beyond the immediate last action. A third arm without opponent
  information would separate the observation field's own effect; it is optional and not budgeted.
- **Budget (proposal, measured in readiness):** 3 seed pairs (s = 0, 1, 2) × 4.096M agent transitions, the M4.7 pilot
  size, from `ext/baseline:s`. Common random numbers: PPO seeds and start streams are shared between the arms of a seed.

## 6. Evaluation (to be frozen in the pilot's pre-registration)

All games start from s0, with paired seats and common random numbers.
- **Value:** the lane-exit readout E_20 at turn 94, read as the lead at turn 114. The lane lead at turn 94 and the lead
  at turn 150 (the game length of the evaluation) are descriptive.
- **Comparison:** each arm's final against the same opponents.

| family | opponents | question |
|---|---|---|
| **E: exploitation** | the 6 archetypes (in the library); 4 held-out scripted variants (`g_shove_sr`, `wk_shove_reset_sh`, `g_shove_edge`, `g_slow_guard`) | does the treatment gain against recognizable styles, in and out of the library? |
| **R1: competent** | the other seeds' baseline finals `ext/baseline:o`, the C1 policy, held-out learned instances (attacker ext_s0, freeze s4, gap s4, short s1) | no loss against strong and unfamiliar learned opponents |
| **R2: misspecified** | held-out mixtures (bully/farm, shove/safe) and switchers (safe → shove at 30, bully → farm at 50) | no loss when the opponent fits no type and changes style |
| **R3: head to head** | treatment vs control, paired seats | does modelling make the agent exploitable by an equally trained non-modelling agent? |
| **M: mechanism** | the treatment's own belief in its games; belief-swap probes (the same state with the belief set to different types) | does the policy identify and use the belief, or ignore it? |

**Success:** E ≥ +0.25, and none of R1–R3 ≤ −0.25. Each is reported separately per seed. The exact thresholds,
seeds as units, and the outcome classes are fixed in the pre-registration before any training.

## 7. What this design does not settle

- **Exploitability by a dedicated attacker** of the treatment (a best response trained against it). That is M5-scale
  work; R3 is the pilot's proxy.
- **Opponents far outside both families** (human play).
- **Belief cost in training.** Each learner turn needs the 11 types' likelihoods on the opponent's state: one batched
  network forward per learned type, plus a scripted decision per scripted type. Readiness measures the throughput, and
  the pilot budget follows from it.
- **Inferring the opponent's action from state changes** (the alternative to decision 2). Not needed under decision 2.

## 8. Readiness checklist (before asking for the pilot's budget)

1. **Observation field:**
   - default off reproduces every digest pin;
   - on, it encodes the opponent's previous action;
   - it is mirror-equivariant, deterministic and restorable.
2. **Belief module:**
   - exact Bayes at λ = 1;
   - forgetting;
   - the catch-all wins against mixtures and loses against single styles;
   - absent turns are uninformative.
3. **Trainer hooks:** default training bit-identical; the hooks reach every network input (rollout, truncation
   bootstrap, final bootstrap).
4. **Initialization:** the expanded network reproduces π_s's outputs exactly at start.
5. **Smoke training of both arms:**
   - the belief features vary in the treatment and are constant in the control;
   - library, mixture and switcher opponents all appear;
   - both arms start from the same rollout;
   - a re-run is bit-identical.
6. **Throughput,** and the budget estimate.
7. **Evaluation harness:** a trained network plays from s0 with the last-action observation and the belief, on common
   random numbers.
8. **Pre-registration frozen; the budget submitted for approval.**
