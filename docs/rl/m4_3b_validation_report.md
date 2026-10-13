# M4.3b follow-up — Strategic validation of the wave-management conclusions and research-artifact cleanup

Branch `development/m4-3b-validation` (on top of `development/m4-3b-ppo-pilot`, commit 98a4341). No mechanics,
reward, observation, hyperparameter or checkpoint of M4.3b changed; no network was trained. Everything here is
evaluated on the M4.3b finals (`base` seeds 0, 1, 2 and `ent03` seed 0) and their attackers, which are locally
available and load with verified weight digests (manifest §9).

The question. M4.3b found that the learned policies Clear on 60–90% of their turns and that every multi-turn
alternative (a slow push, pure last-hitting, a held freeze) loses 1–3 discounted resources against the policy.
It attributed Clear dominance to the game's incentives (a best reply to Clear) rather than to a failure of
exploration. This follow-up asks whether that conclusion survives (i) an intervention that changes **only** the
wave management, (ii) opponents other than the policy itself, (iii) a longer evaluation window, and (iv) a check
of which "freeze-like" states are actually freezes, including synthetic states with power gaps. The results tables
are in `results/rl/m43b/validation/report.md`; the stage files and figures are in `results/rl/m43b/validation/`.

## Contents

1. What was confounded in the M4.3b plan-level counterfactuals
2. The wave-only intervention: definition, legality, two rules
3. Experimental design
4. Results: wave-only interventions at sampled states, four opponents
5. Results: whole-game value of a wave rule with adaptive behaviour
6. Results: the evaluation horizon
7. Results: which freeze-like states are freezes; synthetic power gaps
8. Interpretation: did the conclusions about Clear, Thin, Hold, slow pushing and freezing change?
9. Repository artifacts: what changed, what is proposed
10. Limitations and open questions
11. Recommendation for M4.3c
12. Files, tests, reproduction

## 1. What was confounded in the M4.3b plan-level counterfactuals

`plans.py` (M4.3b report §7.2) committed the focal champion to a *complete* action for the plan's duration:
`slow_push` = Thin + Hold position for 2 turns then Hold + Hold for 12; `hold` = Hold + Hold for 14; `freeze` = the
threshold controller with Hold position for 20 turns. Two strategic variables moved at once. The policy's champion
behaviour is strongly state-dependent (M4.3b §6: Trade at the lane, Safe when the enemy mass reaches the tower, Recall
at low HP and after the own crash, Pressure on 76–90% of the turns it is legal), and all of it was switched off for
14–20 turns. A measured loss of 1.5–2.8 resources could therefore be any mixture of (a) the wave management under
test and (b) the loss of trades, retreats, Recalls and Pressure. The report's attribution of the loss to the wave
management ("the player who withheld damage has paid a wave of farm for nothing") was plausible from the farm
decomposition but not isolated. A second gap: every plan was valued against the policy itself, so a strategy that
pays only against a different kind of opponent could not show. A third: the 300-turn window carries a guaranteed
truncation bound of 5.2 resources (γ^300 Ω), larger than every effect measured, and the pairing argument that
cancels the shared tail was asserted, not checked.

## 2. The wave-only intervention

`lanegame/experiments/rl/m43b/intervene.py`. The 46-code action space is a product: a = wave + 3 · behaviour + 15 ·
objective for a present champion (45 codes) plus ABSENT. Under natural waves (`mechanics.wave_actions`) all three
wave components (Hold, Thin, Clear) are legal on every turn a champion is present; the legality of the behaviour and
objective parts does not depend on the wave (Pressure / Vision are legal with Hold or Trade when the own crashed
mass is under the enemy tower and the champion can reach it; Recall is the single code `champ_act(W_HOLD, B_RECALL)`
at or behind the recall line; ABSENT is the only code of an absent champion). Hence replacing the wave of any legal
present-champion code other than Recall yields a legal code, with no fallback needed; the test
`test_wave_replacement_is_legal_on_every_legal_set_met_in_play` checks this invariant on every legal set met in a
reference game (with and without Pressure, with and without Recall, absent).

The intervention acts on the policy's **probability vector** p over `obs.legal` at each turn, not on a sampled action,
and the rollout samples from the reshaped vector with the same uniform draw it would have used (a new, backwards-
compatible hook `Plan.transform(obs, t, p)` in `diagnostics/branch.py`; a plan that does not implement it behaves
exactly as before, and the existing diagnostics tests pass unchanged). Two explicit rules, both reproducible and both
reported:

| rule | behaviour / objective of the forced action | Recall | absent |
|---|---|---|---|
| **marginal** (component replacement) | every action a ≠ Recall sends its mass to (w, behaviour(a), objective(a)): the policy's *marginal* over behaviour and objective, whatever wave it would have paired them with | kept at the policy's own probability | unchanged (ABSENT) |
| **conditional** (policy-conditional) | the mass of the actions whose wave is w is renormalised: the policy's *conditional* distribution of behaviour and objective given that it chooses w itself | kept at the policy's own probability | unchanged |

The rules coincide when the policy's behaviour is independent of its wave choice and differ when it is correlated
(Clear + Trade against Hold + Safe is the typical pairing); reporting both brackets what "the policy's own behaviour
under a different wave" can mean. Neither rule touches the opponent, the chance streams or the continuation after
the window, so every candidate stays paired with the policy's own play rollout for rollout. A `WavePlan` is DEFER on
every turn of its window (an intervention of N turns means "for the next N turns, whenever present"; absence does
not end it, unlike the M4.3b plans, which handed back when the champion was absent) and then hands control back for
good. The wave functions are: Hold, Thin, Clear (constant); the slow-push schedule of `plans.py` (Thin 2 turns then
Hold); the M4.2e-C threshold freeze controller (Thin while the enemy front stands at own −1 or deeper with ≥ 14 HP,
else Hold); for whole games the reactive slow push of `zoning_natural` (in contact, Thin while the own front is at
most 2 HP ahead) and shove-while-away. `InterventionAgent` is the same reshaping as an environment agent, for
whole-game evaluation. Tests: `tests/test_rl_m43b_validation.py` (14: legality on every legal set, both rules are
distributions that keep Recall and absence, the marginal rule keeps the behaviour / objective marginals, the
conditional rule renormalises the policy's own slice, the wave functions read the state from the observer's side,
the plan forces only the wave and hands back after its window, the hook keeps one draw per seat per turn and the
opponent's stream, an identity transform reproduces the policy rollout exactly, invalid transforms are rejected, the
agent matches the rule and plays legal games, the stage helpers, and the untrained network through the engine).

## 3. Experimental design

All stages reuse the M4.3b finals (`base` seeds 0–2, `ent03` seed 0) and attackers, loaded with verified digests; no
training. Evaluation seeds are `SeedSequence` roots 67,500–67,999, disjoint from every stream of the pilot.

| stage | what | states / episodes | window | candidates | opponents |
|---|---|---|---|---|---|
| **wave_cf** | paired counterfactual rollouts from sampled reachable states (`behavior.sample_states`: states of games of the policy *against the opponent of the cell*, turns 8–310, at most 2 per game) | 8 states per (policy, opponent, situation); 16 paired rollouts per candidate | 300 | policy; wave-only marginal: Hold, Thin, Clear (14 turns), slow push (14), freeze controller (20); wave-only conditional: Hold, slow push, freeze; the original full-action `hold`, `slow_push`, `freeze` of M4.3b on the same states and streams | the policy itself; `farm` (Hold the wave, Hold position: the Holder the M4.2e-C probes used); `push_punish` (reactive slow push + punish, the equal-power equilibrium line); `clear_punish` (Clear every turn + punish) |
| **games** | whole games from s0, the policy with its wave forced for the whole game by a rule (behaviour, objectives, Recall as the policy decides) against each opponent, minus the unmodified policy on the same episode seeds | 32 paired episodes (both seats) | 1000 | none; marginal: Hold, Thin, Clear, reactive slow push, freeze controller, shove-while-away; conditional: Hold, reactive slow push, freeze | the policy itself, `farm`, `push_punish`, `clear_punish`, `freeze_punish`, the policy's own attacker |
| **horizon** | the wave_cf states and rollout roots rolled out for 1000 turns; the discounted sum read at W = 300, 460 and 1000 from the same rollout | 8 states, 16 rollouts | 1000 | policy; m:Hold, m:Clear, m:slow push, m:freeze; f:freeze (self only) | self, `farm`, `clear_punish` × neutral, freeze-like; `base` seeds 0–2 |
| **freeze** (a) | every sampled freeze-like state classified by rolling the wave-only freeze controller 20 turns against `farm` and counting turns the enemy front stays at own −1 (maintainable ≥ 10 of 20; favourable = maintainable and power ≥ 1, HP ≥ 50%) | the wave_cf freeze-like states | 20 | — | `farm`, self |
| **freeze** (b), **SYNTHETIC** | the deterministic neutral (k = 9) and freeze (k = 51) starts of the M4.2e-B harness with assigned power gaps (`with_gap`: equal, moderate 16 vs 4 invested, large 50 vs 0, +3 levels, both), the focal champion ahead or behind | 1 constructed state per cell, 16 paired rollouts | 300 | policy; m:Hold, m:Thin, m:Clear, m:slow push, m:freeze | self, `farm`, `push_punish`, `clear_punish` |

Situations (the M4.3b masks): **neutral** = fronts fighting at the lane cell at equal power; **enemy front at own −1**
(the lane position a freeze needs, whatever the own front does); **freeze-like** = enemy front at own −1 and own front
in the own tower cell. Statistics: within a cell the paired advantage per state (16 rollouts) and its standard error;
across the 8 states the mean ± 1.96 SE with states as the unit, and the share of states with a significant gain /
loss; across the three `base` training seeds the mean of the per-seed cell means ± 1.96 SE with seeds as the unit
(n = 3, indicative). Cross-play pairings that reuse the same networks are never counted as replicates; `ent03`
seed 0 is reported as one more policy, not pooled.

## 4. Results: wave-only interventions at sampled states, four opponents

`results/rl/m43b/validation/wave_cf.json`, `report.md` §1, `fig_val_wave_cf.png`. Q(plan) − Q(policy's own play)
in discounted resources over 300 turns; mean over the three `base` seeds of the per-seed state means ± 1.96 SE over
seeds; in brackets the pooled number of states (out of 24 = 3 seeds × 8) with a significant loss. `ent03` s0 (one
policy) is in `report.md`; it agrees in sign and size everywhere.

| opponent | situation | m:Hold | m:Thin | m:Clear | m:slow push | m:freeze | c:Hold | c:freeze | f:Hold (M4.3b) | f:freeze (M4.3b) |
|---|---|---|---|---|---|---|---|---|---|---|
| self | neutral | −2.02 ± 0.07 (22/24) | −0.96 ± 0.21 (13/24) | −0.01 ± 0.13 (1/24) | −1.84 ± 0.17 (21/24) | −2.50 ± 0.12 (23/24) | −2.30 ± 0.44 | −2.91 ± 0.44 | −1.99 ± 0.25 | −2.50 ± 0.33 |
| self | enemy at own −1 | −1.95 ± 0.35 (24/24) | −0.90 ± 0.38 (13/24) | −0.06 ± 0.12 (2/24) | −1.68 ± 0.44 (20/24) | −2.53 ± 0.23 (24/24) | −2.08 ± 0.31 | −2.85 ± 0.13 | −1.87 ± 0.40 | −2.51 ± 0.38 |
| self | freeze-like | −2.12 ± 0.15 (22/24) | −1.12 ± 0.14 (16/24) | +0.03 ± 0.21 (1/24) | −1.95 ± 0.26 (22/24) | −2.64 ± 0.35 (24/24) | −2.29 ± 0.26 | −2.90 ± 0.30 | −1.95 ± 0.13 | −2.40 ± 0.39 |
| farm | neutral (20 states) | −1.24 ± 0.33 (16/20) | −0.33 ± 0.06 (10/20) | +0.04 ± 0.08 (0/20) | −1.33 ± 0.24 (18/20) | −1.63 ± 0.28 (19/20) | −1.43 ± 0.58 | −2.14 ± 0.65 | −1.33 ± 0.47 | −1.90 ± 0.37 |
| farm | enemy at own −1 (9 states) | −1.18 ± 0.31 (7/9) | −0.18 ± 0.43 (1/9) | +0.10 ± 0.02 (0/9) | −0.61 ± 0.31 (3/9) | −0.91 ± 0.26 (6/9) | −0.95 ± 0.45 | −1.35 ± 0.01 | −1.13 ± 0.14 | −1.23 ± 0.83 |
| farm | freeze-like (8 states, 1 seed) | −0.75 (4/8) | −0.16 (2/8) | +0.03 (0/8) | −0.76 (5/8) | −1.08 (5/8) | −0.86 | −1.31 | −1.00 | −1.62 |
| push_punish | neutral | −3.08 ± 0.28 (24/24) | −0.63 ± 0.33 (11/24) | +0.00 ± 0.02 (0/24) | −2.44 ± 0.13 (24/24) | −3.10 ± 0.27 (24/24) | −2.87 ± 0.50 | −3.40 ± 0.12 | −2.45 ± 0.42 | −2.51 ± 0.36 |
| push_punish | enemy at own −1 | −3.00 ± 0.26 (23/24) | −0.40 ± 0.25 (6/24) | −0.04 ± 0.01 (0/24) | −1.88 ± 0.34 (22/24) | −2.50 ± 0.13 (23/24) | −2.71 ± 0.35 | −2.82 ± 0.17 | −2.43 ± 0.15 | −2.35 ± 0.32 |
| push_punish | freeze-like | −3.06 ± 0.46 (24/24) | −0.18 ± 0.19 (4/24) | −0.11 ± 0.15 (0/24) | −1.79 ± 0.10 (21/24) | −2.54 ± 0.43 (23/24) | −2.29 ± 0.72 | −2.39 ± 0.37 | −2.41 ± 1.05 | −2.09 ± 0.75 |
| clear_punish | neutral | −2.67 ± 0.22 (24/24) | −0.85 ± 0.06 (7/24) | −0.10 ± 0.15 (1/24) | −2.34 ± 0.39 (23/24) | −3.41 ± 0.37 (24/24) | −2.63 ± 0.87 | −3.65 ± 1.04 | −3.00 ± 0.11 | −3.86 ± 0.24 |
| clear_punish | enemy at own −1 | −3.40 ± 0.51 (24/24) | −1.73 ± 0.59 (18/24) | −0.04 ± 0.29 (2/24) | −3.28 ± 0.55 (22/24) | −3.99 ± 0.34 (24/24) | −2.81 ± 1.12 | −3.82 ± 1.06 | −3.57 ± 0.34 | −4.47 ± 0.20 |
| clear_punish | freeze-like | −3.06 ± 0.37 (23/24) | −1.21 ± 0.35 (14/24) | +0.08 ± 0.08 (0/24) | −2.42 ± 0.42 (19/24) | −3.75 ± 0.39 (24/24) | −2.39 ± 0.69 | −3.40 ± 0.90 | −3.10 ± 0.10 | −3.95 ± 0.41 |

Against `farm` the policy's Clear pushes the lane permanently into the farmer's tower, so an enemy front at own −1
is rare (9 states in 3 × 60 games) and a freeze-like state rarer (8, from one seed); those rows are thin and
labelled.

**The confound was small.** Full-action minus wave-only on the same states and streams (mean over the `base` seeds
± 1.96 SE, the three situations in order; `report.md` §1.5 has every cell):

| opponent | f:Hold − m:Hold | f:slow push − m:slow push | f:freeze − m:freeze |
|---|---|---|---|
| self | +0.03 ± 0.20, +0.08 ± 0.11, +0.17 ± 0.25 | +0.02 ± 0.15, −0.18 ± 0.03, +0.02 ± 0.43 | +0.01 ± 0.21, +0.02 ± 0.15, +0.24 ± 0.64 |
| farm | −0.09 ± 0.53, +0.04 ± 0.17 | −0.19 ± 0.22, −0.24 ± 0.55 | −0.27 ± 0.31, −0.33 ± 0.57 |
| push_punish | +0.64 ± 0.37, +0.58 ± 0.14, +0.65 ± 0.63 | +0.76 ± 0.32, +0.43 ± 0.03, +0.55 ± 0.11 | +0.59 ± 0.24, +0.15 ± 0.39, +0.45 ± 0.75 |
| clear_punish | −0.34 ± 0.25, −0.16 ± 0.41, −0.04 ± 0.41 | −0.71 ± 0.54, −0.11 ± 0.19, −0.41 ± 0.44 | −0.45 ± 0.43, −0.48 ± 0.15, −0.20 ± 0.79 |

Against the policy itself the two interventions agree to within 0.2: the M4.3b losses were the wave management.
Against `clear_punish` keeping the adaptive behaviour recovers 0.1–0.7 (the policy's Safe and Recall responses are
worth something against an aggressive clearer); against `push_punish` the *full-action* plans are 0.4–0.8 **less**
bad than the wave-only ones: the policy's own behaviour against the punisher (Trade, and the 19–21 deaths per 300
turns it accepts; M4.3b §6.2) costs more than the Hold position the old plan imposed. Neither correction comes near
the 1.5–3.5 size of the losses, and the conditional rule, which gives the forced wave the behaviour the policy itself
pairs with it, is never better than the marginal rule (0.2–0.5 worse in most cells; the policy pairs Hold with Safe
and Yield, which withdraw from farm).

**Reading the table.**

- **Hold, the slow push and the freeze controller lose against every opponent, in every situation, for every
  policy, with the champion behaviour fully adaptive** (a significant loss in 16–24 of 24 states per cell against
  the three reactive opponents and the policy; 4–7 of 8–9 in the thin `farm` rows). The loss is farm: the plan's own
  farm minus the policy's over the window falls by 1.4–3.5 while deaths are unchanged (the full decomposition is in
  `report.md` §1); nothing is bought with it.
- **The opponent matters for the size, not the sign.** Against the pure last-hitter the losses halve (Hold −1.2,
  freeze −1.6 at the neutral lane) because the policy's alternative there, a free push into a passive farmer's
  tower, is worth less than against a contesting opponent; against the slow pusher and the clearer they grow to
  2.4–4.0. There is no opponent in the set against which withholding wave damage pays.
- **Thin is the mild case.** Wave-only Thin costs 0.2–1.0 against the policy, the farmer and the slow pusher (and is
  not significantly different from the policy's play in 10–18 of 24 states), 0.9–1.7 against the clearer. A Thin
  edge is the only wave-control move that keeps most of the income, and the policy's own 13–21% Thin (M4.3b §5.1)
  sits where this measurement says it is cheap.
- **Clear is the policy's own play:** wave-only Clear is within ±0.1 of the policy everywhere (the policy already
  Clears on 78–100% of these states, column `P(Clear)` in `report.md`), and the small positive entries against the
  farmer (+0.04 to +0.10) say the policy's residual Thin and Hold against a passive opponent cost it nothing it can
  measure.
- **Where the states come from.** The sampled states' attack-power ratio is 0.87–1.34 (mean 1.00–1.17) and the
  focal HP 53–92%: as M4.3b noted, play between these policies does not produce leads or deficits, which is why
  §7 adds synthetic gaps.

## 5. Results: whole-game value of a wave rule with adaptive behaviour

`games.json`, `report.md` §2, `fig_val_games.png`. The policy plays whole games from s0 with its wave component
forced by a rule for the entire game (behaviour, objectives and Recall as the policy decides, under the marginal or
the conditional rule); W = 1000, 32 paired episodes in both seats. Entry = value(rule) − value(unmodified policy)
against the same opponent on the same episode seeds; mean over the three `base` seeds ± 1.96 SE over seeds (the
per-policy paired intervals are ± 0.2–0.9; `report.md` §2). `none` = the unmodified policy's value, per seed.

| opponent | none (s0 / s1 / s2) | m:Hold | m:Thin | m:Clear | m:reactive slow push | m:freeze | m:shove while away | c:Hold | c:freeze |
|---|---|---|---|---|---|---|---|---|---|
| self | 0 | −11.8 ± 1.8 | −6.4 ± 0.5 | −0.2 ± 0.1 | −11.2 ± 1.7 | −11.7 ± 1.8 | −11.3 ± 1.4 | −13.4 ± 2.4 | −13.2 ± 2.5 |
| farm | +10.9 / +10.6 / +11.9 | −8.1 ± 1.3 | −1.7 ± 0.6 | +0.6 ± 0.7 | −6.0 ± 0.6 | −8.2 ± 1.4 | −8.0 ± 1.2 | −10.8 ± 2.9 | −11.2 ± 2.8 |
| push_punish | +15.1 / +12.4 / +15.1 | −11.9 ± 1.5 | −1.9 ± 1.2 | +0.2 ± 0.3 | −8.1 ± 0.7 | −11.0 ± 1.6 | −11.3 ± 1.4 | −14.5 ± 3.2 | −13.8 ± 2.8 |
| clear_punish | +8.6 / +9.1 / +9.2 | −12.9 ± 0.6 | −5.9 ± 0.9 | −0.1 ± 0.3 | −11.4 ± 0.9 | −12.5 ± 0.6 | −12.5 ± 1.6 | −16.4 ± 2.0 | −16.3 ± 2.4 |
| freeze_punish | +15.6 / +12.8 / +15.8 | −8.3 ± 1.9 | −1.3 ± 0.9 | +0.1 ± 0.4 | −7.2 ± 1.5 | −8.2 ± 1.5 | −8.4 ± 1.9 | −11.2 ± 4.2 | −11.4 ± 4.1 |
| its attacker | −1.8 / −1.3 / −1.1 | −11.1 ± 2.6 | −5.2 ± 1.0 | +0.2 ± 0.2 | −10.4 ± 2.3 | −10.9 ± 2.8 | −10.2 ± 2.3 | −12.7 ± 2.8 | −12.5 ± 2.8 |

(`ent03` s0 is within the same ranges in every cell.)

- **A policy that never Clears while the enemy is present loses 8–13 resources over a 1000-turn game**, against
  every opponent including the passive farmer and the scripted freezer: about a tenth of the lane's discounted
  income, and as much as or more than the 9–20 the learned policies are ahead of the references. Hold, the reactive
  slow push, the freeze controller and shove-only-while-away are indistinguishable from each other (they are the
  same rule where it matters: Hold while the enemy is on the lane); the conditional rule adds 1.5–4 of loss on top,
  because the behaviour the policy pairs with Hold (Safe, Yield) withdraws from the farm.
- **Thin is opponent-dependent.** Forcing Thin for the whole game costs 1.3–1.9 against the farmer, the slow pusher
  and the freezer (a sixth of the Hold loss), but 5–6 against the policy, the clearer and the attacker. A Thin edge
  survives against an opponent that does not remove it and is punished by one that does; this is the one place
  where "a strategy pays against some opponents and not others" shows, and the direction is the one M4.2e-C
  predicted (Thin against Holders, Clear against clearers), not a freeze or a slow push paying anywhere.
- **Clear is at the policy's own value** everywhere (−0.2 to +0.6): the policy's residual Thin / Hold is worth
  nothing measurable even against a farmer, where a wave-only Clear is nominally +0.6 ± 0.7.

## 6. Results: the evaluation horizon

`horizon.json`, `report.md` §3, `fig_val_horizon.png`. The wave_cf states and rollout roots of the `base` seeds (self,
farm and clear_punish × neutral and freeze-like) were rolled out for 1000 turns and the discounted sum read at
W = 300, 460 and 1000 from the same rollouts. The W = 300 column reproduces the §4 estimates exactly (same streams;
the stage's smoke test asserts it to 10⁻¹²). Across the 3 seeds:

| cell | candidate | W = 300 | W = 460 | W = 1000 | tail (1000 − 300) |
|---|---|---|---|---|---|
| self, neutral | m:Hold | −2.02 ± 0.07 | −2.02 ± 0.05 | −2.02 ± 0.05 | −0.00 ± 0.01 |
| self, neutral | m:freeze | −2.50 ± 0.12 | −2.52 ± 0.11 | −2.52 ± 0.11 | −0.01 ± 0.01 |
| self, freeze-like | m:freeze | −2.64 ± 0.35 | −2.64 ± 0.35 | −2.64 ± 0.35 | −0.00 ± 0.01 |
| self, freeze-like | f:freeze | −2.40 ± 0.39 | −2.40 ± 0.39 | −2.40 ± 0.39 | −0.00 ± 0.01 |
| farm, neutral | m:freeze | −1.63 ± 0.28 | −1.62 ± 0.28 | −1.62 ± 0.28 | +0.00 ± 0.00 |
| clear_punish, neutral | m:freeze | −3.41 ± 0.37 | −3.40 ± 0.36 | −3.40 ± 0.36 | +0.00 ± 0.01 |
| clear_punish, freeze-like | m:Hold | −3.06 ± 0.37 | −3.07 ± 0.37 | −3.07 ± 0.37 | −0.00 ± 0.01 |

Every candidate in every cell behaves the same (`report.md` §3): over 544 (state, candidate) pairs the tail
contribution between turn 300 and turn 1000 has mean −0.001, mean absolute value 0.017, maximum 0.08, and no
state's estimate changes sign (one flip, for a wave-only Clear whose estimate is 0.00 ± 0.01). The guaranteed bounds
(5.2 at W = 300, 1.05 at 460, 0.005 at 1000) are bounds on the *level* of a single return, not on a paired
difference; after a 14–20-turn deviation the two trajectories' reward streams are indistinguishable in distribution
by turn 300, and the pairing cancels the shared tail as M4.3b asserted. **The 300-turn window was sufficient for
every comparison in M4.3b §7 and here; the finite-window estimates can be read as estimates of the infinite-horizon
difference to within ± 0.02.** (The level estimates of §5 use W = 1000, where the bound is 0.005.)

## 7. Results: which freeze-like states are freezes; synthetic power gaps

### 7.1. Classification of the sampled freeze-like states (`freeze.json`, `report.md` §4)

Each of the 88 freeze-like states (8 per policy and opponent where they exist) was classified by rolling the
wave-only freeze controller for 20 turns against a Holding opponent (`farm`) and against the policy, counting the
turns the enemy front stands at own −1 with HP > 0:

| opponent of the cell | states | superficial (held < 10 of 20 vs farm) | maintainable (≥ 10) | favourable (maintainable, power ≥ 1, HP ≥ 50%) | held vs farm | held vs the policy |
|---|---|---|---|---|---|---|
| self | 24 | 9 | 13 | 2 | 0–18.4 | 0.5–8.2 |
| farm | 8 | 2 | 1 | 5 | 4.6–15.9 | 0.1–4.5 |
| push_punish | 24 | 12 | 4 | 8 | 0.2–19.0 | 0.4–7.6 |
| clear_punish | 24 | 12 | 1 | 11 | 0–17.1 | 0.2–10.8 |

Half of the mask's states are genuinely maintainable freezes against a Holder (the controller keeps the enemy front
at own −1 for 10–19 of 20 turns; the others crash or dissolve within a few turns), and a quarter are favourable by
the power and HP criterion. Against the policy itself the same controller holds the lane for 0–11 turns (typically
1–6): the opponent's own Clear breaks the freeze by removing the wave the freezer is trying to keep. The freeze
advantage by class (pooled states, mean ± 95%):

| opponent | class | states | m:freeze | c:freeze | f:freeze (M4.3b) | m:Hold |
|---|---|---|---|---|---|---|
| self | superficial | 9 | −2.54 ± 0.59 | −2.90 ± 0.58 | −2.31 ± 0.51 | −2.08 ± 0.66 |
| self | maintainable | 13 | −2.57 ± 0.51 | −2.78 ± 0.47 | −2.30 ± 0.45 | −2.01 ± 0.39 |
| self | favourable | 2 | −3.56 ± 0.59 | −3.73 ± 1.20 | −3.52 ± 0.37 | −3.00 ± 0.48 |
| farm | favourable | 5 | −1.49 ± 0.40 | −1.40 ± 0.29 | −1.70 ± 0.50 | −0.98 ± 0.51 |
| push_punish | favourable | 8 | −2.48 ± 0.82 | −2.43 ± 0.69 | −2.29 ± 0.94 | −3.19 ± 0.86 |
| push_punish | superficial | 12 | −2.38 ± 0.50 | −2.28 ± 0.45 | −1.81 ± 0.53 | −2.84 ± 0.56 |
| clear_punish | favourable | 11 | −4.06 ± 0.55 | −3.40 ± 0.63 | −4.11 ± 0.88 | −3.19 ± 0.57 |
| clear_punish | superficial | 12 | −3.51 ± 0.46 | −3.48 ± 0.52 | −3.85 ± 0.76 | −2.99 ± 0.61 |

**Maintainability and favourability do not rescue the freeze.** The loss is the same in every class (and if
anything larger where the freeze is favourable), against every opponent, including the Holder against whom the
freeze is physically held for most of the window. A held freeze in this game keeps the enemy wave alive next to the
own tower and gives up the Clear income of the turns it is held; it buys no position, no gank safety the policy
values, and no denial the opponent feels (the farmer keeps last-hitting the held wave from its side).

### 7.2. SYNTHETIC sensitivity: constructed starts with assigned power gaps (`report.md` §5, `fig_val_synthetic.png`)

**These states are constructed, not sampled from play**, and are reported separately for that reason: the neutral
(k = 9, both fronts at the lane) and freeze (k = 51, the start of the longest held freeze of the M4.2e-C setup) starts
of the M4.2e-B harness, both champions at the same XP, with `with_gap` applied (A at 16 vs 4 resources invested,
50 vs 0, +3 levels, or both: attack-power ratios 1.24, 1.76, 1.35, 2.38 and their inverses). The focal champion is
the one ahead (seat A) or the one behind (seat B). W = 300, 16 paired rollouts, three `base` seeds (mean ± 1.96 SE
over seeds; `report.md` §5 has all 72 cells):

| start, gap, focal | opponent | m:Hold | m:Thin | m:Clear | m:slow push | m:freeze |
|---|---|---|---|---|---|---|
| neutral, equal | self | −2.66 ± 0.88 | −1.60 ± 0.68 | −0.05 ± 0.09 | −2.67 ± 0.80 | −3.25 ± 1.18 |
| neutral, equal | farm | −0.71 ± 0.78 | −0.37 ± 0.31 | +0.01 ± 0.05 | −0.04 ± 0.50 | −1.03 ± 0.44 |
| neutral, large (×1.76), ahead | self | −3.98 ± 0.40 | −1.14 ± 0.63 | +0.01 ± 0.19 | −2.82 ± 0.83 | −4.92 ± 0.12 |
| neutral, large, ahead | clear_punish | −0.66 ± 0.08 | +0.33 ± 0.54 | +0.15 ± 0.15 | −0.15 ± 0.60 | −1.29 ± 0.11 |
| neutral, large, behind | self | −1.52 ± 0.39 | −0.95 ± 0.17 | 0.00 | −1.29 ± 0.32 | −1.58 ± 0.62 |
| neutral, large, behind | push_punish | −1.84 ± 0.72 | −0.59 ± 0.57 | +0.05 ± 0.07 | −2.03 ± 0.38 | −2.58 ± 0.72 |
| freeze, equal | farm | −0.11 ± 0.22 | −0.43 ± 0.24 | +0.14 ± 0.19 | −0.16 ± 0.37 | −0.44 ± 0.17 |
| freeze, equal | push_punish | −3.20 ± 0.44 | +0.12 ± 0.43 | +0.08 ± 0.25 | −2.37 ± 0.21 | −2.02 ± 0.45 |
| freeze, large, ahead | farm | −0.43 ± 0.39 | **+0.61 ± 0.41** | 0.00 | −0.15 ± 0.34 | −0.85 ± 0.29 |
| freeze, large, ahead | push_punish | −4.17 ± 0.63 | +0.20 ± 0.47 | −0.07 ± 0.08 | −0.34 ± 0.31 | −2.70 ± 0.65 |
| freeze, large, ahead | clear_punish | −3.35 ± 1.10 | +0.57 ± 1.68 | +0.01 ± 0.23 | −0.92 ± 1.11 | −4.15 ± 0.97 |
| freeze, large, behind | farm | −2.73 ± 0.83 | −0.32 ± 0.38 | +0.06 ± 0.12 | −1.04 ± 0.88 | −3.23 ± 0.49 |
| freeze, large, behind | push_punish | −3.83 ± 0.77 | −0.66 ± 0.30 | +0.07 ± 0.14 | −1.60 ± 0.11 | −5.00 ± 0.76 |
| freeze, ×2.38, ahead | clear_punish | −2.97 ± 0.41 | −0.55 ± 0.21 | +0.02 ± 0.13 | −1.71 ± 0.42 | −3.03 ± 0.60 |

- **Hold and the freeze controller lose in all 72 cells** (−0.1 to −5.0; significant for 2–3 of the 3 seeds in
  most). A power lead makes them *more* costly against a contesting opponent (a ×1.76 lead at the neutral lane turns
  Hold's −2.7 into −4.0 against the policy): the stronger champion's larger wave capacity makes every turn of
  withheld damage dearer. A deficit does not make them cheaper either (−1.1 to −5.0 when behind).
- **Thin is the only wave-control move that is ever positive**, and only when the focal champion is far ahead at
  an already-frozen lane against a scripted opponent: +0.61 ± 0.41 against the farmer and +0.57 ± 1.68 against the
  clearer at ×1.76 (2 of 3 seeds significant), +0.2 to +0.5 against the slow pusher at ×1.24–1.76 (0–1 of 3). The
  size is a few tenths of a resource, the opposite sign appears as soon as the opponent is the policy, and the
  states are synthetic. It is consistent with §4 and §5 (a Thin edge is cheap, and against a Holder nearly free)
  and is the strongest evidence this study found for any wave-control move beating the policy's Clear.
- The slow push is close to zero against the farmer and the clearer when ahead (−0.04 to −0.15) and negative
  otherwise.

## 8. Interpretation: did the conclusions change?

Against the brief's illustrative outcomes:

- **A (alternatives become profitable once champion behaviour is adaptive): no.** The full-action and the wave-only
  plans agree to within 0.2 against the policy and within 0.8 against every scripted opponent, on the same states
  and streams; the corrections go both ways (adaptive behaviour helps against the clearer, hurts against the
  punisher) and never approach the 1.5–4 size of the losses. The M4.3b attribution of the loss to the wave
  management was correct.
- **B (profitable against some opponents, not against aggressive clearers): only for Thin, and only in size.** No
  opponent in the set, including the pure last-hitter and the scripted freezer, makes Hold, the slow push or a held
  freeze pay, at a sampled state (§4), over a whole game (§5), or from a synthetic favourable freeze with a power lead
  (§7). What *is* opponent-dependent is the cost of a Thin edge: 1–2 per game against opponents that leave it alone,
  5–6 against ones that clear it, and marginally positive when far ahead against a Holder. The simulator does
  contain an opponent-dependent strategic interaction in the wave dimension, but it is Thin-against-Holders, not
  freezing or slow pushing.
- **C (alternatives remain unprofitable under improved conditions): yes, for the tested class.** Wave management
  that withholds damage loses against every opponent, with the policy's own adaptive behaviour, at the 1000-turn
  horizon, at reachable and at maintainable and at favourable freezes, with and without power gaps.
- **D (viable coordinated strategies exist but are undiscovered): not excluded, but not supported.** This study
  tested one strategy class, wave control with the policy's own behaviour. A strategy that coordinates the
  *behaviour* differently from the policy (e.g. Hold + Safe as a zoning stance, or Thin + Trade only when ahead) is
  outside it, and the only general instrument for that class is best-response training, whose M4.3b result (an
  attacker gains 0.75–2.0 over a final, and the attackers' play is Clear-based too) is the current upper bound on
  what is left undiscovered. The synthetic Thin result (+0.6 when far ahead at a freeze) is small enough that a
  policy that never sees power gaps would not be expected to learn it, and that is the one concrete gap between
  what the game rewards and what the policies play.

**What the quantitative model incentivises, stated plainly.** Under `m4_lasthit`, clearing the enemy wave every turn
costs the clearer nothing it could otherwise keep: Clear's capacity removes the whole spawn at AP 1, the farm of a
cleared wave is collected at the same rate as that of a held one, a crash into the enemy tower loses nothing (the
defender collects it at 0.6 and the crasher takes plates), and the only prices of pushing forward (the Shove
gank-exposure term, trades, a one-resource death) are small next to a wave of farm. A held freeze or a slow push
therefore gives up income every turn and receives nothing the reward counts; this is why the learned policies, four
training conditions, their attackers and the scripted equilibria of M4.2e-C all sit on the Clear line, and why
the wave-only interventions find the same losses the full-action ones did. This is a property of the game as
calibrated, not a defect in the learner, and no game parameter was changed here (brief §8).

**Possible explanations and design investigations, not undertaken.** For a freeze or a slow push to pay, holding the
wave must buy something the reward counts: (i) a last-hit income that depends on where the wave is (the own-tower
collection rate 0.6 is the only position dependence now, and it penalises the defender, not the pusher); (ii) a cost
to Clear itself (a collection penalty while clearing, or Clear exposing the champion for more than the Shove gank
term); (iii) a larger price for being pushed in (kill bounty 0.5 → 1–2, gank severity, as M4.3b §14 listed). The
wave-only harness is the right test for each: a lever is interesting if the wave-only loss of m:slow push or
m:freeze against the *learned* policy shrinks towards zero or changes sign at reachable states, before any retraining.

## 9. Repository artifacts: what changed, what is proposed

Inspected: the M4.3b commit 98a4341 adds 64 blobs, **3.8 MB packed** (Git's zlib compresses the JSON results about
ten-fold: `counterfactual.json` 3.1 MB → 0.59 MB, `recall.json` 1.6 MB → 0.32 MB, `archive_train.json` 7.0 MB →
under 0.25 MB; the eight figures are 1.9 MB of it). The whole repository pack is 143 MB, of which the M4.3b commit is
under 3%; the largest single blobs in history are `results/m4_zoning_natural/zoning_natural.json.gz` (4.3 MB) and
`results/siege/solution_K32.json` (68 MB in the working tree, 2.3 MB packed), both on `main` already. The working
tree cost of M4.3b was the real problem: 17 MB of results, 1.1 M lines of JSON in one commit, a 764k-line file.

**Done (non-destructive):**

- `results/rl/m43b/{archive_train,counterfactual,recall}.json` → `.json.gz` (7.0 / 3.2 / 1.7 MB → 0.19 / 0.43 / 0.22
  MB); `validation/wave_cf.json` likewise (2.5 → ~0.4 MB). Every reader accepts both forms (`common.load`,
  `common.read_json`, `validation.vload`, `env.starts.load_archive`; `save_archive` writes `.gz` by extension), so no
  stage, figure or report changed behaviour; `CONDITIONS["mixed"].start_archive` and `archive.ARCHIVE_PATH` point at
  the gzipped archive. Tests cover the gzipped archive through `load_archive`.
- `results/rl/m43b/MANIFEST.json` (`python -m lanegame.experiments.rl.m43b manifest`): every result file with size,
  SHA-256 and, from its own `meta`, the commit, timestamp, stage, seeds and command; every checkpoint (finals,
  attackers, run configs, histories) with size, SHA-256 and the loader-verified weights digest; the PPO and
  self-play configuration of every run.
- The checkpoint bundle `results/rl/m43b_artifacts/m43b_checkpoints_<commit>.tar.gz` (+ `.sha256`): 54 files,
  5.2 MB, everything an evaluation needs without retraining; `manifest.verify_bundle` checks it against the manifest.
  `results/rl/m43b_artifacts/` is git-ignored; the bundle is meant to be attached to a release or kept on a shared
  drive. The 150 MB of training snapshots stay local (listed in the manifest on request; `bundle_checkpoints(full=True)`
  packs them) — they are needed only for the learning curves and the self-play progress matchups.
- `results/rl/m43b/ARTIFACTS.md`: the artifact policy, what each file is, which commit and command produced it,
  how to reproduce with and without retraining.

**Proposed, needs your decision (history):** the raw JSON files remain in the branch's history (commit 98a4341,
already on `origin`). Rewriting would save about 1 MB of pack (the gzipped files pack no smaller than Git's own
compression of the raw ones) at the cost of a force-push or a replacement branch. On the numbers, **I recommend
merging the branch as it is** and keeping the gzip-at-1-MB rule for every later study. If you prefer a clean
history anyway, the safe route without a force-push is: `git checkout -b development/m4-3b-clean main`, copy the
tree of `development/m4-3b-validation` into it (`git checkout development/m4-3b-validation -- .`), commit once (or
twice, re-applying 98a4341 with the gzipped files and then the follow-up), push the new branch, merge it, and delete
the old remote branches only after the merge is verified; nothing is lost at any step. I have not done any of this.

**Not touched, flagged:** `results/siege/solution_K32.json` (68 MB in the tree, `python -m lanegame solve --horizon 32`
regenerates it in 4 min) and the other large tracked results of earlier studies would benefit from the same gzip
rule; that is a separate, equally non-destructive change.

## 10. Limitations and open questions

1. **One strategy class.** The interventions fix the wave and leave the behaviour to the policy; a strategy that
   changes both in a coordinated way (a zoning stance, trades timed to a Thin edge) is untested except through the
   M4.3b attackers. A best-response search against a frozen final with the wave fixed (train only the behaviour
   under m:freeze) would close that gap cheaply.
2. **Plan lengths.** 14 turns of Hold / Thin / slow push and 20 of the freeze controller at the state level; the
   whole-game rules at the other extreme. An intermediate (a 40–60-turn freeze then release) was not run; the
   whole-game rows bound it from below and the state rows from above, and both are negative.
3. **Opponents are four scripted lines plus the policies and their attackers.** A learned opponent that holds would
   be the stronger test of "Thin against Holders"; none exists because self-play does not produce one.
4. **Synthetic states** carry the M4.2e-B harness's construction (common XP, no gold, the k = 51 freeze after a
   scripted setup); they show sensitivity, not frequency.
5. **n = 3 training seeds**; the across-seed intervals are indicative. The within-cell state intervals (8 states) are
   ± 0.1–0.6 and the sign agreement across cells is what carries the conclusion.
6. **Thin positive cells** (synthetic, far ahead, scripted opponents) are at the edge of resolution (+0.2 to +0.6,
   2 of 3 seeds); they would need 3× the rollouts to be claimed at the usual confidence.

## 11. Recommendation for M4.3c

The evidence supports **not** spending M4.3c on more self-play training or on exploration devices for the wave
dimension: the game does not reward what the policies do not do, and three independent instruments agree to
within the noise. The useful next milestone is the **design investigation of Clear dominance** (M4.3b §14, item 4),
now with a sharper test and a shorter loop:

1. For each candidate lever (own-tower collection rate, a collection or exposure cost of Clear, kill bounty, gank
   severity, the Clear share of capacity), run `val_wave_cf` against the *existing* finals first (no training): a
   lever is worth a training run only if m:slow push or m:freeze at reachable states moves towards zero or changes
   sign. This is minutes per lever.
2. Only for levers that pass, train 3 seeds at 2M with entropy 0.03 and re-run the wave-only harness and the games
   stage; a lever is interesting if the *learned* policies move their wave mix.
3. Independently of the levers, one cheap check of outcome D: a best-response attacker with the wave fixed to the
   freeze controller or to Thin (behaviour learned), against a frozen final, from the archive of reachable states
   with assigned power gaps; if even that attacker cannot beat the policy's Clear, the "undiscovered strategy"
   explanation is closed for this game.
4. Keep the M4.3b defaults (entropy 0.03, population attackers, W = 1000) for any training that follows.

I have not started any of this.

## 12. Files, tests, reproduction

| file | content |
|---|---|
| `lanegame/diagnostics/branch.py` | `Plan.transform(obs, t, p)`: the distribution-level hook of the rollout (identity by default; one draw per seat per turn unchanged) |
| `lanegame/experiments/rl/m43b/intervene.py` | the two intervention rules, the wave functions, `WavePlan`, `InterventionAgent` |
| `validation.py` | the four stages (`wave_cf`, `games`, `horizon`, `freeze`), their seeds and summaries |
| `validation_report.py` | `results/rl/m43b/validation/report.md` and the four figures |
| `manifest.py` | the manifest, the checkpoint bundle and its verification, `compress_results` |
| `common.py`, `env/starts.py`, `archive.py` | gzip-transparent readers; the archive path |
| `__main__.py` | stages `val_wave_cf`, `val_games`, `val_horizon`, `val_freeze`, `val_report`, `manifest`; `validation` runs the five |
| `tests/test_rl_m43b_validation.py` | 14 tests (§2) |
| `results/rl/m43b/validation/` | `wave_cf.json.gz`, `games.json`, `horizon.json`, `freeze.json`, `report.md`, `fig_val_*.png`, the stage logs |
| `results/rl/m43b/MANIFEST.json`, `ARTIFACTS.md` | the manifest and the artifact policy |

Tests at this commit: `tests/test_rl_m43b_validation.py` (14) in 8 s; `tests/test_diagnostics.py`,
`tests/test_m4_diagnostics.py` (31) unchanged and passing; the full suite: 909 passed, 4 skipped, 2 expected failures in 7 min (`results/rl/m43b/validation/pytest_full.log`, run without the new file), plus the 14 new tests.

Reproduction (12 logical CPUs, 9 worker processes; the checkpoints of M4.3b in place):

```bash
python -m pytest tests/test_rl_m43b_validation.py
python -m lanegame.experiments.rl.m43b val_wave_cf --workers 9      # 23 min
python -m lanegame.experiments.rl.m43b val_horizon val_games val_freeze --workers 9   # 15 + 18 + 6 min
python -m lanegame.experiments.rl.m43b val_report                   # tables and figures, 3 s
python -m lanegame.experiments.rl.m43b manifest                     # MANIFEST.json and the checkpoint bundle
```
