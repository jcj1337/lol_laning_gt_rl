# OM1 — Opponent-modelling feasibility

*2026-10-17. Status: complete, awaiting review. Evaluation only: no training, and no network, observation, rule,
reward, objective, default or checkpoint changed.*

- Protocol: [om1_protocol.md](om1_protocol.md), frozen after the smoke check and before any selection or evaluation
  game (SHA-256 in `results/om1/protocol.json`).
- Generated tables: `results/om1/report.md`. Code: `lanegame/experiments/rl/om1/`; tests: `tests/test_om1.py`.

**Why this study.** Step 3 of the agreed sequence is to design Bayesian opponent modelling: a belief about which kind of
opponent one faces, updated during the game, and a policy that uses it. Before designing it, OM1 asks two questions
with existing policies only:
1. **Identifiability.** Does the lane phase reveal the opponent's type early enough to act on?
2. **Value of information.** Does knowing the type change what one should play, and by how much?

**Terms.**
- **Focal policy π_s:** the adopted M5 baseline's matched policy `ext/baseline:s`, s = 0..5.
- **Library:** the 15 types a modeller holds:
  - the nine M4.5c scripted references;
  - six learned policies (peer, attacker, freezer, gapper, short, early; protocol §2).
- **Value:** a responder's earned-gold lead at turn 114, the adopted lane-exit readout E_20 at turn 94 in expectation.
  Each game is played in both seat orientations on the same chance stream.
- **r\*(type):** the existing policy that does best against that type (selected on separate games); π unless another
  policy beats it by at least 0.25.
- **Oracle:** knows the type from the start and plays r\*.
- **Switcher:** plays π until its Bayesian belief in some type reaches 0.9, then plays that type's r\*. It is a
  modelling policy built entirely from existing parts.
- **Signal:** after each turn the opponent's chosen action is observed. This is an upper bound on what the game's
  observation reveals; the coarse variant hides the wave choice.

## Summary

**Pre-registered outcome 1: feasible and worthwhile.** Opponent modelling can pay in this game, and the design (step 3)
should go ahead. Where the value comes from, and the two failure modes OM1 exposed, shape that design.

- **The lane phase reveals the opponent quickly** (in-library, the opponent's action observed after each turn):
  - **Decision-relevant identification** (the believed type has the same best responder as the true type): 96% of
    games after 10 observed turns, 99% after 20, 100% after 40.
  - **Exact identification** is lower (80% at 40), because the five Shove variants look alike. They share the same
    best responder, so the confusion costs nothing.
  - **Without the wave choice** (behaviour and Pressure only), decision-relevant identification is still 82% at 10
    turns, 93% at 20 and 99% at 40.
  - The scripted types' tremble allowance (ε 0.01 to 0.2) makes no difference.
- **Knowing the type is worth about a wave by the lane exit.** The Bayesian switcher, which plays π until its belief
  reaches 0.9 and then plays the type's best existing responder, beats π by **+0.90 ± 0.27**, measured as the
  E_20-based lane-exit value at turn 114.
  - The mean is taken over the 15 types and the six seeds are the units; all six are positive (+0.60 to +1.21).
  - That is 83% of what an oracle that knows the type gets (+1.09 ± 0.44).
  - The gain grows after the lane phase: +1.52 ± 0.30 at turn 150.
  - No single existing policy does better than π on average (the best fixed responder is π), so this value comes from
    adapting to the type, not from π being weak.
- **The value is concentrated against exploitable scripted styles** (descriptive split, not pre-registered):
  - scripted types: +1.36 ± 0.43;
  - learned types: +0.21 ± 0.11;
  - without the largest single case (`g_safe`, +4.1): +0.68 ± 0.29.
  - Against learned opponents it gains against the attacker (+1.10) and the freezer (+0.58), and **loses against the
    peer** (−0.61 ± 0.36). The peer is the type closest to normal self-play; its selected responder was a false
    positive of the selection stage.
- **Misspecified opponents are the main risk.**
  - Unseen instances of the learned families are handled without harm: −0.04 to +0.34, none significant.
  - The two stochastic style mixtures, which no library type represents, cost **−0.83 ± 0.38** (bully/safe) and
    **−1.45 ± 0.33** (shove/farm). The belief splits between the two styles, and the switcher plays a specialist
    responder that the other half punishes.
  - Pooled over the seven out-of-library opponents: −0.23 ± 0.24. That is just inside the pre-registered tolerance
    (−0.25), so the robustness flag is not raised, but only narrowly.

**What this means for the design** (proposals for step 3, for review):
1. **Act on the decision, not the type.** The switcher waited for one type to reach 0.9. Against the look-alike Shove
   variants it rarely did, and it captured +0.24 where the oracle got +0.96. A belief that acts when the posterior mass
   of one *response* is high would take most of that.
2. **Guard against misspecification.** Add an explicit "none of the above" check: compare the library's predictive
   likelihood with a generic model and fall back to π when nothing fits. Train or evaluate against mixed and
   style-switching opponents. The mixtures are where switching hurt.
3. **Learned responses rather than switching between fixed policies.** The value against competent learned opponents
   is small with existing responders (+0.21), and one selected responder was wrong. A policy that conditions on the
   belief and is trained against the library would remove the selection noise and could find better responses. The
   matched pilot (step 4) is where that is tested.
4. **Information channel.** The game's observation is the current state; it contains no action history. The design
   must either infer the opponent's action from state changes or add the opponent's last action to the observation.
   The latter would be an opt-in observation field, and it would invalidate existing checkpoints for that variant.
   OM1's coarse-signal result suggests that most of the information survives a partial view.

**Limitations.**
- In-library results are optimistic: the modeller's world contains the true opponent.
- The signal (the opponent's full action) is an upper bound.
- The responders are existing policies, so the oracle gain is a lower bound on the value of information.
- The type prior is uniform over a library with nine scripted styles, which weights the exploitable cases heavily.
- Six focal seeds.

**Next step:** the opponent-modelling design document (step 3). Its scope is a proposal until reviewed; then the
matched pilot (step 4).

---

## 1. Design (protocol §2–§3)

| stage | games | what it gives |
|---|---|---|
| A, selection | every existing policy in the menu (π and the 15 library types) against every library type; 64 paired-seat pairs × 120 turns per cell (240 cells) | r\*(type) per type and the best fixed responder |
| B, evaluation | fresh games, for each of the six focal seeds and each library type: π_s, the oracle, the switcher; 96 pairs × 150 turns; π_s's games track the belief through turn 94 | identifiability; value of information, paired on common random numbers |
| C, robustness | for each focal seed and seven out-of-library opponents: π_s and the switcher; 64 pairs × 150 turns | what the switcher does when the opponent is not in its library |

- **Belief:** a uniform prior over the 15 types, updated with the opponent's observed action. A learned type's
  likelihood is its exact action probability. A scripted type's likelihood trembles: (1 − ε) on its action, ε = 0.05
  spread over the legal codes; ε = 0.01 and 0.2 are sensitivity checks.
- **Out-of-library opponents:**
  - other instances of the learned families: an attacker trained against another seed, a freeze and a gap specialist
    of another seed, another short-horizon run;
  - a C1 policy;
  - two per-turn 50/50 mixtures of scripted styles: bully/safe and shove/farm.

## 2. Stage A: selection

Every menu responder (π, pooled over the six seeds, and the 15 library types) played every library type: 64 pairs ×
120 turns per cell.

- **π beats 14 of the 15 types.** The exception is the short-horizon policy (−0.27).
- **For 13 types another existing policy did better by at least 0.25 on these games:**
  - the freeze specialist against the Shove scripts;
  - the scripted `g_shove_ward` against `g_farm` and `g_bully`;
  - the scripted `g_farm` against `g_safe` (+5.5: plain farming exploits a Safe player);
  - the short-horizon policy against the attacker and the freezer;
  - the attacker against the peer.
- **The best fixed responder** over the whole library is π: the freeze specialist's mean is only +0.04 higher, below
  the threshold.

Selection values are maxima over 16 responders on 64 pairs, so they are optimistic; stage B re-measures them on fresh
games. One choice did not survive: the attacker against the peer (selection +0.74, evaluation −0.46 ± 0.39).

## 3. Identifiability

From π_s's own games against each type (6 seeds × 96 pairs × 2 orientations per type), the belief after t observed
turns, mean over the 15 types:

| turns observed | 3 | 5 | 10 | 20 | 40 | 94 |
|---|---|---|---|---|---|---|
| true type's posterior ≥ 0.9 | 25% | 35% | 52% | 69% | 80% | 86% |
| the most likely type is the true one | 54% | 62% | 71% | 79% | 86% | 91% |
| **decision-relevant** (the most likely type has the true type's responder) | 86% | 91% | **96%** | **99%** | **100%** | 100% |
| decision-relevant, coarse signal (no wave choice) | 48% | 54% | 82% | 93% | 99% | 100% |

- **Fast types** (posterior ≥ 0.9 within 5 turns in ≥ 98% of games): `g_farm`, `g_bully`, `g_safe` and
  `g_shove_ward`. Their behaviour differs from everything else from the first turns.
- **Learned types** reach ≥ 0.9 in 39–76% of games by turn 10 and 81–95% by turn 20; they are almost never confused
  with each other by turn 40.
- **The Shove family** (`g_shove`, `g_shove_sr`, `wk_shove_reset_sh`, `fz_guard_sh`, partly `g_shove_edge`) differs
  only in rare conditional choices: a Recall threshold, a reset when frozen on, a guarded trade. They stay confused
  for a long time (`g_shove`: 5% at turn 40), but all have the same best responder.
- **I (pre-registered) is met:** decision-relevant identification at turn 40 is 100% (threshold 2/3).
- **The signal assumption matters little for decisions.** With the wave choice hidden, decision-relevant
  identification lags by about 5–10 turns, then catches up. `g_farm` is the exception: it is recognized by its wave
  choice (Last Hit), and its coarse identification at turn 40 is 10%.

## 4. Value of information

Each contrast is paired on common random numbers within a cell: the same 96 game pairs for π_s, the oracle and the
switcher. Population values are means over the 15 types per seed, with the six seeds as units.

| contrast | lead at 114 (E_20 at 94) | lane lead at 94 | lead at 150 |
|---|---|---|---|
| **switcher − π (G_B)** | **+0.90 ± 0.27** | +0.71 ± 0.24 | +1.52 ± 0.30 |
| oracle − π (G_O) | +1.09 ± 0.44 | +0.77 ± 0.41 | +1.72 ± 0.43 |
| oracle − best fixed responder (= π) | +1.09 ± 0.44 | | |
| π's own lead against the library (reference) | +3.43 ± 0.41 | | |

Per type (pooled over seeds; 576 pairs each):

| type | best responder r* | π's lead | oracle − π | switcher − π | switcher plays r* (share of turns) | first switch (median turn) |
|---|---|---|---|---|---|---|
| g_farm | g_shove_ward | +2.65 | +1.18 ± 0.18 | +0.99 ± 0.19 | 98% | 3 |
| g_shove | freezer | +4.18 | +0.96 ± 0.37 | +0.24 ± 0.13 | 13% | never (median) |
| g_shove_sr | freezer | +3.79 | +1.73 ± 0.36 | +1.08 ± 0.28 | 62% | 40 |
| wk_shove_reset_sh | freezer | +4.28 | +1.07 ± 0.34 | +0.42 ± 0.21 | 31% | never (median) |
| g_shove_edge | freezer | +4.73 | +1.55 ± 0.36 | +1.74 ± 0.36 | 81% | 24 |
| g_bully | g_shove_ward | +9.18 | +1.31 ± 0.33 | +1.40 ± 0.31 | 98% | 3 |
| g_safe | g_farm | +6.92 | +4.65 ± 0.21 | +4.08 ± 0.21 | 99% | 2 |
| fz_guard_sh | freezer | +4.09 | +0.77 ± 0.36 | +0.82 ± 0.24 | 47% | 68 |
| g_shove_ward | freezer | +3.24 | +1.44 ± 0.36 | +1.51 ± 0.35 | 98% | 3 |
| peer | attacker | +1.79 | −0.46 ± 0.39 | **−0.61 ± 0.36** | 90% | 11 |
| attacker | short | +0.63 | +1.27 ± 0.36 | +1.10 ± 0.36 | 91% | 10 |
| freezer | short | +0.84 | +0.62 ± 0.41 | +0.58 ± 0.38 | 92% | 8 |
| gapper | π | +1.96 | 0 | −0.07 ± 0.06 | 0% | never |
| short | short | −0.25 | +0.25 ± 0.25 | +0.28 ± 0.34 | 91% | 10 |
| early | π | +3.39 | 0 | +0.00 ± 0.06 | 0% | never |

- **The switcher gets 83% of the oracle's value overall.** It loses most against the Shove look-alikes, where it waits
  for one type to reach 0.9 although the response is already clear. Against `g_shove` it switches in only 13% of
  turns.
- **Against competent learned opponents the value is small:** +0.21 ± 0.11 over the six learned types (descriptive),
  driven by the attacker and the freezer. The peer case is a loss caused by a wrong selection.
- **The value is not a lane-phase artefact.** It is +0.71 at turn 94, +0.90 at 114 and +1.52 at 150: the adapted
  responders keep gaining after the lane phase.

## 5. Robustness: opponents outside the library

Per opponent, pooled over the six seeds (384 pairs each):

| opponent | switcher − π (lead at 114) | believed to be after 40 turns | switcher's play |
|---|---|---|---|
| attacker of another seed | −0.04 ± 0.49 | short 39%, attacker 29%, peer 27% | short 47%, π 34% |
| freeze specialist of another seed | +0.06 ± 0.37 | freezer 51%, gapper 36% | π 58%, short 40% |
| gap specialist of another seed | −0.01 ± 0.26 | gapper 82%, freezer 17% | π 93% |
| another short-horizon run | +0.34 ± 0.45 | attacker 40%, short 30%, peer 19% | short 54%, π 38% |
| a C1 policy | +0.31 ± 0.36 | attacker 51%, short 20%, peer 20% | short 53%, π 36% |
| **50/50 bully/safe mixture** | **−0.83 ± 0.38** | g_bully 53%, g_safe 47% | g_shove_ward 46%, g_farm 45% |
| **50/50 shove/farm mixture** | **−1.45 ± 0.33** | g_shove 49%, g_farm 48% | π 55%, g_shove_ward 40% |
| pooled (seeds as units) | −0.23 ± 0.24 | | |

- **New instances of the learned families are absorbed.** The belief maps them to their nearest library type, and
  the switcher neither gains nor loses materially.
- **Mixtures are the failure mode.** A per-turn mixture of two scripted styles is confidently identified as one of
  them in about half the games, so the switcher plays the specialist responder for the wrong half. A modeller that can
  say "none of these fits" would stay with π. That check is cheap: under a mixture the library's predictive
  likelihood is poor, because neither style explains every action.
- **The pre-registered robustness flag is not raised** (pooled −0.23, tolerance −0.25), but the margin is small and
  rests on which out-of-library opponents were chosen.

## 6. Classification and recommendation

| criterion (protocol §4) | result | met |
|---|---|---|
| I: decision-relevant identification at turn 40 ≥ 2/3 | 100% | yes |
| G_B ≥ +0.25, interval excluding 0 | +0.90 ± 0.27 | yes |
| R violated (out-of-library ≤ −0.25, interval excluding 0) | −0.23 ± 0.24 | no (narrowly) |

**Outcome 1: feasible and worthwhile.** Proceed to the opponent-modelling design (step 3). The proposals for its scope
are in the summary. In short:
- act on the response, not the type;
- add a misspecification fallback and train against mixtures;
- learn belief-conditioned responses, rather than switching between fixed policies;
- settle how the opponent's action is observed.

## 7. Limitations

- **In-library optimism.** The modeller's world contains the true opponent in stages A and B. Stage C shows that the
  value does not carry over to misspecified opponents and can turn negative.
- **The signal is an upper bound.** OM1 lets the modeller see the opponent's chosen action. The game's observation is
  the state; with only the behaviour class, decisions are delayed by 5–10 turns.
- **Existing responders only.**
  - The oracle gain is a lower bound on what knowing the type is worth.
  - The selection is noisy (64 pairs per cell), and one choice (the peer's) was wrong on fresh games.
- **The type prior.** A uniform prior over a library with nine scripted styles weights exploitable opponents heavily.
  Against learned opponents only, the value is +0.21 ± 0.11. What M5's opponent distribution will be is a design
  question.
- **Six focal seeds; one library.** Population claims rest on six training seeds of one configuration.

## 8. Runtime, checks and files

**Runtime** (wall clock, 5 workers at below-normal priority on a shared 6-core desktop):

| stage | wall |
|---|---|
| smoke | 1 min |
| A: selection (240 cells × 64 pairs × 120 turns) | 574 s |
| B: evaluation (90 cells × 96 pairs × 150 turns, three responders, belief tracking) | 3,984 s |
| C: robustness (42 cells × 64 pairs × 150 turns) | 793 s |
| analysis and tables | 5 s |

That is about 1 h 30 min of compute in all.

**Checks.**
- **Smoke checks pass:**
  - deterministic re-runs;
  - a self-play pair's lead is exactly 0;
  - the posterior stays normalized;
  - the switcher equals π until its first switch.
- `tests/test_om1.py` (7) passes: the coarse signal, the scripted tremble likelihood, exact Bayes and absent turns,
  the selection rule, reproducible mixtures, the learned likelihood equal to the sampling distribution, and the
  switcher equal to π until it switches.
- Full regression suite: 1,480 passed, 4 skipped, 2 xfailed in 8 min (the 1,473 of M4.8 plus the 7 new tests).
- The protocol was frozen before any selection or evaluation game (SHA-256 e6972991…); no amendment.

**Files.**
- New code: `lanegame/experiments/rl/om1/` (`common`, `library`, `games`, `stages`, `analysis`, `report`, CLI);
  `tests/test_om1.py`.
- New documents: this report and [the protocol](om1_protocol.md), in the new folder `docs/opponent_modelling/`.
- Generated (`results/om1/`): `smoke.json`, `protocol.json`, `selection.json`, `analysis.json`, `report.md`, logs.
  Raw game records (git-ignored) are under `results/rl/checkpoints/om1/`.
- Updated: the roadmap (§AV, step 3, the decisions of 2026-10-17), `README.md`, `docs/README.md`, the mathematical
  reference's §9.3 status line.

**Unchanged:** every library module, preset, default, objective, observation and checkpoint. Nothing is staged or
committed.

**Reproduce:**

```
python -m lanegame.experiments.rl.om1 smoke protocol A select B C analyse report --workers 5
```
