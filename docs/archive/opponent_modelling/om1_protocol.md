# OM1 protocol: opponent-modelling feasibility

*2026-10-17. Frozen (SHA-256 in `results/om1/protocol.json`) after the smoke and throughput check, before any selection
or evaluation game. Seen at freezing time: the smoke checks and throughput only (`results/om1/smoke.json`).*

## 1. Question

Before Bayesian opponent modelling is designed (step 3 of the agreed sequence), is it feasible and worth it in this
game?
1. **Identifiability.** Does the lane phase reveal which kind of opponent one faces, early enough to act on?
2. **Value of information.** Would knowing the type change what to play, and by how much?

**Evaluation only.** No training. No network, observation, rule, reward, objective or default changes. The focal
policy is the adopted M5 baseline: `ext/baseline:s` (the M4.6c 8.192M continuation, `m45_calibrated()`), s = 0..5.

## 2. Types, responders and the belief

**Library** (the types a modeller holds; existing policies only, uniform prior):
- the nine M4.5c scripted references (deterministic): `g_farm`, `g_shove`, `g_shove_sr`, `wk_shove_reset_sh`,
  `g_shove_edge`, `g_bully`, `g_safe`, `fz_guard_sh`, `g_shove_ward`;
- six learned policies:
  - peer: the M4.5c 8.192M final, seed 3;
  - attacker: the M4.5c attacker ext_s3;
  - freezer: the M4.6a freeze specialist s2;
  - gapper: the M4.6a gap specialist s2;
  - short: the M4.7 H 60 arm, seed 0;
  - early: the M4.5c 1.024M pilot.

**Out-of-library opponents** (robustness only):
- other instances of the learned families: attacker ext_s0, freeze s4, gap s4, M4.7 short s1;
- a C1 policy, `ext/C1:2`;
- two stochastic mixtures that no type represents: each turn, `g_bully`/`g_safe` or `g_shove`/`g_farm`, 50/50.

**Belief** (the responder's view of the opponent, `om1/library.py`):
- Prior: uniform over the 15 types.
- Signal: after each turn the opponent's chosen action is observed. This is an upper bound on information, because the
  game's observation is the state, from which the action is only partly inferable.
- Likelihoods:
  - a learned type's exact action probability;
  - a scripted type trembles: (1 − ε) on its action, ε spread over the legal codes, ε = 0.05 (sensitivity 0.01 and
    0.2).
- Absent turns carry no information.
- A coarse variant observes only the behaviour class and Pressure, not the wave choice.

**Responders.** The menu is the focal policy π plus the 15 library types.
- **Oracle:** knows the opponent's type and plays its selected responder r*(type) throughout.
- **Best fixed responder r_fix:** the single menu responder with the best mean value over the library.
- **Bayesian switcher:** plays π_s until its full-signal belief in some type reaches θ = 0.9, then that type's r*; it
  re-reads the belief every turn.

## 3. Games and readouts

- **Games:** from s0, paired seats (each game played with the responder in seat A and in seat B on the same chance
  stream). Common random numbers: every responder of a cell meets the opponent on the same games and streams.
- **Value of a pair:** the responder's undiscounted earned-gold lead at turn 114, averaged over the two orientations.
  This is the adopted lane-exit readout E_20 at turn 94, in expectation (M4.8). The lead at turn 150 is descriptive.
- **Stage A, selection** (seeds 60,100·10⁷ + …):
  - every menu responder against every library type, 64 pairs, 120 turns; π's pairs rotate over the six seeds.
  - r*(type) = the best responder, if its selection value beats π's by at least M = 0.25, else π.
  - r_fix: the best mean responder, if it beats π's mean by at least M, else π.
- **Stage B, evaluation** (fresh games, seeds 60,200·10⁷ + …):
  - per seed s and library type: π_s, the oracle (if r* ≠ π), r_fix (if ≠ π and ≠ r*) and the switcher.
  - 96 pairs, 150 turns.
  - π_s's games track the belief through turn 94: posterior at turns 1–94, full, coarse and ε variants.
- **Stage C, robustness** (seeds 60,300·10⁷ + …): per seed and out-of-library opponent, π_s and the switcher, 64 pairs,
  150 turns.
- **Budget:** about 13M simulated turns, about 1.5 h on 5 workers (smoke throughput).

## 4. Criteria (pre-registered)

**Readouts.**
- **I (identifiability):** decision-relevant identification at turn 40 means r*(MAP type after 40 turns) = r*(true
  type), from π_s's tracked games. I is met when its mean over the 15 types is at least 2/3. The plain share with the
  true type's posterior ≥ 0.9 at 5–94 turns is reported beside it.
- **G_B:** switcher − π_s, value at turn 114, averaged over the 15 library types per seed. Seeds are the units (t
  interval, 5 df).
- **G_O:** oracle − π_s, the same way.
- **VOI:** oracle − r_fix, the same way.
- **R (robustness):** switcher − π_s against the out-of-library opponents, pooled per seed. R is violated when it is
  ≤ −0.25 with its interval excluding 0.

**Outcomes.**
1. **Feasible and worthwhile:** I met, and G_B ≥ +0.25 with its interval excluding 0. Next: the design document.
2. **Narrow:** not 1, but at least one type has an oracle gain ≥ +0.5 (pooled over seeds and games, interval
   excluding 0) and decision-relevant identification at turn 40 ≥ 2/3. Next: a design scoped to those types.
3. **Identifiable, little value:** I met, but G_O < +0.25 or its interval includes 0, and no type qualifies under 2.
   Next: modelling with existing responders adds little; pause, or one trained-responder probe (needs approval).
4. **Not identifiable:** I not met, and not 2.

A violated R is flagged on any outcome as a robustness risk the design would have to solve.

## 5. Known limitations

- **The menu is existing policies.** Trained best responses could gain more, so the oracle gain is a lower bound on
  the value of information.
- **The signal is the opponent's action.** That is more than the game's observation reveals; the coarse variant shows
  how much the wave choice matters.
- **The library is the modeller's assumed world.** In-library results are optimistic about identification; stage C
  tests misspecified opponents.
- **Lane phase only.** The value is read at turn 114; the lead at turn 150 is descriptive.
- **Six focal seeds.** Population claims rest on six training seeds.
