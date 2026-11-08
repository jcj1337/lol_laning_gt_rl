# OM2: matched Bayesian opponent-modelling pilot

*2026-10-17. Status: complete, awaiting review. Pilot compute approved by the user on 2026-10-17. Nothing is adopted:
the opponent-modelling pieces stay opt-in, and no default, preset, objective or existing checkpoint changed.*

- Design: [om2_design.md](om2_design.md).
- Pre-registration: [om2_preregistration.md](om2_preregistration.md), frozen before any pilot training (SHA-256 in
  `results/om2/preregistration.json`, no amendment).
- Generated tables: `results/om2/report.md`. Code: `lanegame/opponent/`, `lanegame/experiments/rl/om2/`. Tests:
  `tests/test_om2.py` (11).

**The question.** Can the agent learn to adapt to who it is playing, exploiting recognizable styles, without becoming
worse against competent or unfamiliar opponents? Scope C (the user's choice) puts robustness first.

**The comparison.** Two arms continue the adopted baseline policy `ext/baseline:s` for 4,096,000 transitions each, with
3 seed pairs. Both are the same network and see the same data stream, including the opponent's previous action. Half
of their games are self-play; the other half face one of 15 fixed opponents (scripted archetypes, learned policies, style
mixtures and style switchers).
- The **treatment** also receives a Bayesian belief about which opponent type it faces, updated every turn.
- The **control** receives the belief's prior instead: the same inputs, held constant.

**The value.** Each arm's earned-gold lead at turn 114, from s0, in resources. This is the adopted lane-exit readout
E_20 at turn 94. Each arm meets the same opponents on the same paired-seat games (64 pairs per cell).

## Summary

**Pre-registered outcome 1: modelling pays and is safe.**

- **It pays against recognizable styles.** Against the six library archetypes the treatment leads the control by
  **+2.03 ± 1.72** (seeds +2.57, +1.25, +2.26). The criterion was ≥ +0.25 with every seed positive. Against four
  held-out scripted variants the lead is +1.84, positive in every seed.
- **No robustness family loses:**

  | family | treatment − control | seeds positive |
  |---|---|---|
  | R1: competent learned opponents | +0.43 ± 0.81 | 3 of 3 |
  | R2: held-out mixtures and switchers | +0.95 ± 1.29 | 3 of 3 |
  | R3: head to head | +0.32 ± 1.45 | 2 of 3 |

  None comes near the loss criterion (≤ −0.25 with every seed negative).
- **The policy uses the belief, and the belief is sensible:**
  - Five of the six archetypes are identified within 10 turns in at least 98% of games. The sixth, fz_guard_sh, is
    confused with its look-alike g_shove.
  - Competent learned opponents are read as a spread over the learned types, never as a scripted style.
  - The switchers are followed after each switch, but slowly: it takes 10 to 60 turns, sometimes passing through
    "mixed".
  - A shove/safe per-turn mixture goes to the "mixed" catch-all (94% by turn 40).
  - Setting the belief to one archetype changes the trained policy's action distribution by a total variation of
    0.14–0.32.

**Exploratory reference** (not pre-registered; §5). The common starting policy π_s was played on the same games:
- **Exploitation is mostly learned without a belief.** Against the archetypes, training against them already lifts
  the control +5.1 over π_s, with only the last action observed. The belief adds the further +2.0, about 28% of the
  treatment's +7.2.
- **"Safe" means "no worse than the equally trained control".** Against competent learned opponents, both arms end
  slightly below π_s: control −0.59 (every seed negative), treatment −0.16 ± 0.30. Spending half of training on
  fixed opponents costs some competence, and the belief largely protects against that cost.
- **Neither arm became stronger than π_s head to head:** control −0.12, treatment −0.08.

**Recommendation** (§8). Carry Bayesian opponent modelling into the M5 design, in the form the pilot tested. Make two
things part of M5 rather than assuming them:
- dedicated exploitability tests of the modelling agent, including an opponent that imitates a style and then
  switches;
- a check that competent play does not erode relative to pure self-play.

## 1. What ran

| item | value |
|---|---|
| environment | `m45_calibrated(obs_last_action=True)`: the adopted baseline plus the opponent's previous action (103 entries; mathematical reference §15.4) |
| arms | control: 103 + 13 inputs held at the prior; treatment: 103 + the belief's posterior over 11 library types, "mixed" and "unknown" (forgetting λ = 0.97, ε = 0.05 for scripted types) |
| start | `ext/baseline:s` widened with zero weights: both arms start as exactly that policy |
| training | M4.6c PPO settings and start mixture; 50% self-play, 50% one of 15 fixed opponents (uniform); PPO seed 61,000 + s; same draws in both arms |
| budget | 3 seeds × 2 arms × 4,096,000 transitions; 72 minutes of wall time on 5 workers |
| evaluation | 69 cells (each final against 22 opponents, plus head to head), 64 paired-seat pairs of 150 turns each; 11 minutes |

## 2. Training

All six runs completed.
- **Matched inputs:** within each seed, both arms started from the same weights (verified digest) and processed the
  same transitions.
- **Clean updates:** no illegal action was sampled, and KL stayed small.
- **The belief behaved as designed.** In the treatment its inputs varied (mean spread 0.18), and the uniform
  "unknown" catch-all kept a mean mass of only 0.3%. In the control they stayed exactly at the prior.
- **Critic.** Over the last 10% of updates the treatment's critic explained 0.93–0.94 of the return variance, against
  0.66–0.71 for the control. Knowing the opponent's type makes returns far more predictable when half the games are
  against fixed opponents.

| run | transitions/s | wall (min) | entropy (last 10%) | explained variance |
|---|---|---|---|---|
| control s0 / s1 / s2 | 2,188 / 2,209 / 2,228 | 31 / 31 / 31 | 1.74 / 1.69 / 1.67 | 0.71 / 0.71 / 0.66 |
| treatment s0 / s1 / s2 | 1,416 / 1,419 / 1,645 | 48 / 48 / 42 | 1.69 / 1.61 / 1.64 | 0.94 / 0.94 / 0.93 |

## 3. Pre-registered results

Treatment − control, the lead at turn 114. Seeds are the units; intervals are t intervals with 2 df.

| family | seed 0 | seed 1 | seed 2 | mean ± 95% | criterion |
|---|---|---|---|---|---|
| **E_in**: the 6 archetypes | +2.57 | +1.25 | +2.26 | **+2.03 ± 1.72** | gain met |
| E_out: 4 held-out scripted (descriptive) | +2.82 | +0.64 | +2.06 | +1.84 ± 2.74 | — |
| **R1**: 8 competent learned | +0.05 | +0.58 | +0.65 | +0.43 ± 0.81 | no loss |
| **R2**: 2 held-out mixtures, 2 switchers | +1.34 | +0.36 | +1.15 | +0.95 ± 1.29 | no loss |
| **R3**: treatment vs control | −0.33 | +0.79 | +0.50 | +0.32 ± 1.45 | no loss |

**Classification: E met, no R met → outcome 1, "modelling pays and is safe".**

The other readouts agree (descriptive; pooled over games, 1.96 SE):
- The lane lead at turn 94 shows the same pattern, slightly smaller: E_in +1.67 ± 0.20, R1 +0.32 ± 0.21.
- The lead at turn 150 shows it larger: E_in +2.97 ± 0.25, R2 +1.57 ± 0.32.

## 4. Where the gain comes from

**Per opponent** (treatment − control, mean over seeds):

| family | largest gains | smallest |
|---|---|---|
| E_in | g_shove +2.99, fz_guard_sh +2.68, g_farm +2.55 | g_bully +0.51 (seeds +0.60, −1.46, +2.40) |
| E_out | g_shove_sr +2.50, g_shove_edge +2.12, wk_shove_reset_sh +2.10 | g_slow_guard +0.64 |
| R1 | freezer_b +0.59, short_b +0.57, c1_policy +0.56 | gapper_b +0.19, base:3 +0.12 |
| R2 | mix_shove_safe +1.74, switch_bully_farm_50 +1.73 | **mix_bully_farm −0.52** (seeds +0.75, −2.02, −0.28) |

- **Against styles, the gain is broad.** It is largest against the Shove family and the freeze-guard style, which a
  belief tells apart from passive styles within a few turns.
- **The one negative case** is the bully/farm per-turn mixture. Bully and farm agree on most turns, so the belief
  stays split between g_bully, g_farm and "mixed" (about a third each at turn 94). It is the risk OM1 found
  (misspecified mixtures), much reduced but not gone.

**Identification** (the treatment's own belief in its evaluation games; MAP type correct):

| archetype | after 10 turns | 20 | 40 | 94 |
|---|---|---|---|---|
| g_farm, g_safe, g_shove | 100% | 100% | 100% | 100% |
| g_bully | 98% | 100% | 100% | 100% |
| g_shove_ward | 99% | 99% | 100% | 100% |
| fz_guard_sh | 1% | 26% | 44% | 49% |

fz_guard_sh is shared with g_shove (0.72 / 0.28 at turn 40), the look-alike pair OM1 found. The treatment still gains
+2.68 against it.

**Other opponents** (the belief's largest masses):
- Competent learned opponents are read as a mixture of the learned types. For the other seeds' baseline policies, for
  example: attacker 0.36–0.46, short 0.18–0.29 and peer 0.20–0.27 at turn 40. The held-out gap and freeze specialists
  are read as their own families (gapper 0.71; freezer 0.48).
- The switchers are tracked, but slowly.
  - Against bully → farm at turn 50, the belief is bully 1.00 at turn 40, still bully 0.91 at turn 60, and farm
    1.00 at turn 94. Bully and farm agree on most turns, so the evidence of the switch accumulates slowly.
  - Against safe → shove at turn 30, it moves to "mixed" (0.57 at turn 40, 0.99 at turn 60) before settling on
    shove (0.86 at turn 94).

**Does the policy use it?** The belief-swap probe sets the belief to one type and measures the change in the action
distribution, on the treatment's own states against the archetypes.
- Total variation: 0.14–0.32 for an archetype, 0.07–0.14 for a learned type, 0.06–0.14 for a catch-all.
- The control's distance is exactly 0, as it must be: its 13 constant inputs keep identical weight columns.

## 5. Exploratory reference: the common start π_s (NOT pre-registered)

Added after the classification, to read it. It cannot change the outcome. π_s = `ext/baseline:s` (no opponent
information) played the same games as the arms, with the same chance streams and CRN sampling, so arm − π_s is a
paired contrast. A test confirms that the reference player plays exactly as the untrained control. Each arm also played
π_s head to head (64 pairs).

| family | π_s | control − π_s | treatment − π_s |
|---|---|---|---|
| E_in | +5.01 ± 1.38 | +5.12 ± 1.45 | +7.15 ± 2.66 |
| E_out | +5.23 ± 1.75 | +6.68 ± 1.66 | +8.52 ± 4.00 |
| R1 | +0.63 ± 1.04 | −0.59 ± 0.97 (seeds −0.15, −0.88, −0.74) | −0.16 ± 0.30 (seeds −0.09, −0.30, −0.09) |
| R2 | +6.63 ± 1.22 | +3.42 ± 0.91 | +4.37 ± 1.81 |
| head to head vs π_s | — | −0.12 ± 1.51 | −0.08 ± 1.26 |

- **Training against the styles does most of the exploiting.** The control, which sees only the opponent's last
  action, gains +5.1 over π_s against the archetypes. Most of that is against the Shove family (+7.4 to +7.9). The
  belief adds the remaining +2.0.
- **Competence against competent play is not improved, and drifts down without the belief.** Against R1 the control
  is below π_s in every seed. Under the pre-registered loss rule, which was not pre-registered for this contrast, it
  would count as a loss (−0.59). The treatment stays within −0.16 ± 0.30.
  - A plausible mechanism, not tested: the control's critic cannot tell a scripted opponent from self-play (explained
    variance 0.66–0.71 against 0.93–0.94), so its advantage estimates are noisier in every game.
- **Mixtures and switchers:** both arms gain over π_s (+3.4, +4.4) from training against other mixtures and switchers.

## 6. What this means

- **The project's question gets a positive answer at pilot scale.** An RL agent given a belief about its opponent
  learns, without any scripted strategy, to:
  - exploit recognizable habits more than an equally trained agent without one;
  - keep its edge when the opponent mixes or changes style;
  - lose less against competent opponents than the agent without a belief.
- **The value is concentrated against exploitable styles.** That matches OM1. Against competent learned opponents the
  modelling agent is merely not worse than the baseline policy (−0.16 ± 0.30 against π_s). Against the agents of the
  M5 campaign, which will be competent, the gain should be expected to be small.
- **The belief is not the main source of exploitation; the opponent distribution is.** Training against styles is
  what makes a policy exploit them. The belief adds about 40% more (+2.0 on top of +5.1) and protects competence. A design without
  scripted opponents in training would not show these gains.

## 7. Limitations

- **Three seed pairs and one 4.096M budget.** Seed intervals have 2 df. E_in is consistent in sign (3 of 3), but its
  interval is wide (± 1.72).
- **Library-favourable tests.**
  - E_in uses the very styles the belief models.
  - Three of the four E_out variants are Shove look-alikes.
  - R1's competent opponents come from the same training lineage as the library's learned types.
- **No deceptive opponent.** No opponent was tested that imitates one style to be classified and then exploits the
  response. Forgetting (λ = 0.97) lets the belief follow a switch, but the switchers show it takes 10–60 turns. An
  opponent that switches deliberately at the right moment could profit from that lag. No such exploiter was trained. Exploitability by a dedicated attacker is the main untested risk.
- **The last-action field's own effect is untested.** Both arms observe the opponent's last action, and there is no
  arm without it.
- **The control's competence drift is exploratory.** It comes from the reference of §5, which was not pre-registered.
- **Evaluation from s0 only, read at the lane exit.** Values after turn 150 are not measured.
- **Training mix.** 9 of the 15 fixed training opponents are scripted or built from scripts, and half of all games
  are against fixed opponents. A different mix would change both the gains and the drift.

## 8. Recommendation and decisions for the user

The pre-registered recommendation for outcome 1 is to carry Bayesian opponent modelling into the M5 design, its exact
form a user decision. My recommendation:

1. **Adopt the tested form as M5's opponent-modelling configuration:**
   - the `obs_last_action` observation;
   - the 13-feature belief (11 library types plus "mixed" and "unknown", λ = 0.97);
   - a training mix of self-play and library opponents.

   Do not adopt the control's setup without the belief. It is the same training at the same cost, minus the
   protection of competence.
2. **Make robustness an M5 measurement, not an assumption.** M5's evaluation should include:
   - dedicated attackers trained against the modelling agent and against a pure self-play agent (exploitability);
   - an imitate-then-switch opponent;
   - the competent-play comparison against pure self-play (does a fixed-opponent share erode competence?).
3. **Open M5 design choices** (not settled by this pilot):
   - the fixed-opponent share (50% here);
   - whether the library grows with M5's own policies;
   - whether M5 trains from scratch or continues;
   - whether an arm without the last-action field is worth its cost.

**Decisions for the user:**
- whether to carry opponent modelling into M5, and in which form;
- whether M5's design should be drafted next (M5a pre-registration), with the items above as proposals.

## 9. Reproduction

```bash
python -m lanegame.experiments.rl.om2 smoke prereg            # readiness and the frozen pre-registration
python -m lanegame.experiments.rl.om2 train --confirm-om2-train --workers 5   # 6 runs, ~72 min (approval needed)
python -m lanegame.experiments.rl.om2 eval analyse explore report --workers 5
```

- **Seeds:** training 61,000 + s, smoke 61,050, evaluation 61,100, probes 61,900, exploratory head to head 61,200.
- **Raw files** (git-ignored): `results/rl/checkpoints/om2/`.
- **Summaries:** `results/om2/`, namely `smoke.json`, `train_main.json`, `analysis_main.json`, `explore.json` and
  `report.md`.
