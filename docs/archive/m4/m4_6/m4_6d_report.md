# M4.6d — Favourable freezes, the weaker side's decision, and the specialist gap

*2026-10-16. Status: diagnostic complete, awaiting review. No PPO training ran; no rule, reward, parameter or default
changed; C1 (`m45_calibrated(invest_gain=3.24, invest_scale=64.0)`) stays a provisional opt-in and is not adopted.
Protocol: [m4_6d_preregistration.md](m4_6d_preregistration.md), frozen after the descriptive stage and before any
counterfactual run (SHA-256 in `results/m46d/preregistration.json`, four dated amendments and one dated exploratory follow-up, §1.3). Generated tables:
`results/m46d/report.md`. Previous stages: [M4.6a](m4_6a_report.md), [M4.6b](m4_6b_investment_report.md),
[M4.6c](m4_6c_report.md). Nothing is staged or committed by this work.*

**Terms.**
- A *resource* is the gold of one fresh wave; natural income is about 0.40 resources per turn.
- *G_h* is the focal side's discounted earned-gold lead over h turns (the training objective truncated at h; γ = 0.99).
  G50 is the primary readout.
- The *freezer* (seat A) holds the enemy wave at its own −1 with its own small wave in its tower cell; the *breaker*
  (seat B) is frozen on. A *walk-up* is the breaker standing at its own +1, next to the freezer's tower.
- *Duel ratio* = (1.05^(ΔL) · M_A / M_B)²: damage per exchange in victim bars, A over B. *Capacity* C is a champion's
  wave clear per turn (`round(4 AP)`).
- *The policies* are the six 8.192M M4.6c continuations in each environment (`ext/C1:s`, `ext/baseline:s`), each
  playing both seats. *Specialists* are the M4.6a gap attackers g0, g2, g4 (baseline-trained).
- Unless stated, intervals are 95% t intervals over the six policies (5 df); **bold** in tables marks an interval
  excluding 0. "Exploratory" marks options or stages added by amendment after reading earlier results.

---

## Summary

**The question.** Do the mechanics let a stronger player profit from defending a favourable early freeze? When should
the weaker player contest, concede or reset? And why do specialists still beat shared self-play?

**Answer in one paragraph.** The intended choices exist, but in a narrow window, and the learned policies play them
correctly in natural states.
- Defending a freeze pays about one turn of income, and only when the freezer has an HP edge.
- Healthy breakers should contest, and they do; conceding never confirmed as better; resetting at low HP is roughly as
  good as contesting.
- The remaining deficits are learning-side and sit in rare strength-dependent states, alongside an established
  defect: the critics are calibrated in natural states but nearly blind to strength.
- **No mechanics change is justified. The next step is one value-function study.**

**1. Freezing.**
- **A freeze can be defended only early.** It holds against a walk-up Shove exactly while the breaker's wave clear is
  below the freezer's 8-HP wave and the freezer trades the walk-up: through level 8 under C1, level 6 under the
  baseline. A passive freeze never survives a walk-up Shove.
- **In the best case** (a fresh wave in the freezer's tower cell, the breaker walked up), confirmed on independent seeds:
  - guarding beats passive keeping by +0.25 ± 0.16;
  - with the breaker at 60% HP, guarding beats pushing by +0.30 to +0.51 (significant) and the policy by +0.30 to +0.47
    (significant for the guard that trades every walk-up). That is about one turn of income, and it grows to about +1 by
    150 turns;
  - with both champions healthy, even at a ×1.25 edge, defending is worth nothing (−0.06 to +0.01).
- **The value is real denial.** The breaker's farm is permanently lost, not delayed; it loses 2–6 XP and dies more.
  Holding the wave longer is easy, and it is not the same as profiting from it.
- **Breaking stays the breaker's best reply, and holding still has value.** A guard carries at least half the
  restricted-game weight in 19 of 26 cells.
- **Natural freezes are a different object.**
  - 87% are the breaker's own push arriving, and 68% last one turn.
  - The freezer's wave is usually already worn, so a guard holds against a walk-up only 50–63% of the time.
  - In natural early freeze states no defence beats the policy, which mostly pushes out at once, correctly.
- **The result does not depend on the investment curve.**

**2. Defence: contest, concede or reset.**
- **Healthy: contest.** The breaker's policy is as good as committing to the break and better than every concession.
  A 3-turn concession costs 0.4–0.7; standing back forfeits all farm and XP to save cheap HP.
- **At 45% HP** resetting led in discovery but did not confirm (+0.1 to +0.25): reset and contest are close.
- **Against its own opponent the breaker's policy is nowhere materially inferior.**
- **Against a freezer that trades every walk-up it under-commits** (commit +0.57 ± 0.33 and +0.82 ± 0.42, confirmed).
  That is a gap against a strategy its self-play opponent does not play.
- **The concerning pattern** (contest, die, gain little) was not found on confirmation.

**3. Specialists.**
- **On large constructed gaps the gap persists** against the 8.192M policies: +1.15 ± 0.28 (baseline environment) and
  +1.73 ± 0.15 (C1, cross-environment), growing with the gap.
- **On plausible early advantages the specialists are worse** than the general policies (−0.40 ± 0.12 in C1).
- **The gap is many small trade-timing gains**, spread evenly over 50 turns and mostly in ordinary trade windows. It is
  not Recall reading: hindsight-free Recall windows contribute +0.07 ± 0.11.

**4. Critic.**
- **Calibrated in natural states**: bias +0.12 / −0.06, slope 0.985 against 500-turn returns, residual tail
  +0.00 ± 0.01.
- **Nearly blind to strength.** It values a ×1.25 edge at +0.02 against a realized +0.63, and a ×1.76 gap at about +0.3
  against +5 to +7, in both environments. Earlier readings (M4.6a, M4.6c), which included the critic's own tail,
  understated this three- to fourfold.
- **It ranks the freezer's options at a freeze at chance**, but ranks Recall-window decisions sensibly.
- **In one rare state family it correctly prefers an action the actor rarely takes**: Safe instead of Yield for a
  damaged weak side next to a stronger opponent, +0.6.
- **Its causal role in the actor's gaps is a hypothesis, not established.**

**5. Mechanics versus learning.**
- **Structural:** the capacity threshold, the push-arrival origin of natural freezes, and the cost of conceding (the
  walk-up cell farms at full rate).
- **Learning:** the critic's strength blindness and the actor's under-use of three rare profitable responses (guarding a
  favourable freeze, committing against a guard, Safe in a Recall window).
- **The one interaction weaker than League's** is the freezer's threat on a walk-up after the early levels. It is a
  design choice (for example wave strength scaling with time), named but not recommended (§8).

**6. Recommendation.** Accept the current abstraction for freezing and concession. As the single next step, run a
no-PPO value-representation probe (§8): it tells whether the critic's strength blindness lies in the observation, in
the self-play representation or in value training, before any further training or mechanics change.
**Verdict: PASS WITH CAVEATS.**

---

## 1. What was run

### 1.1 Design

| stage | content | turns simulated | wall clock |
|---|---|---|---|
| records | s0 self-play of the 12 general policies (6 per environment), 48 × 400 turns each, full per-turn state, policy marginals and critic values | 0.23M | 1 min |
| freezes | descriptive: 13,646 freeze-state episodes classified and disaggregated; natural early strength differences; the M4.6c constructed-freeze records split by cell | none | 11 s |
| A | 26 constructed freeze cells (levels 3 / 5 / 8; eight strength / HP conditions; two small-wave cells) × 6 policies, freezer {policy, guard_sel, passive, push} × breaker {policy, commit, contest_sel, concede, reset}, K 32, H 50 | 5.0M | 18 min |
| AL | level 5 equal / plaus / ×1.76, 4 freezer × 3 breaker plans, H 150 | 1.0M | 3 min |
| A2 (exploratory, amendment 3) | the A cells with freezer `guard_walk`, same seeds (paired with A) | 1.2M | 4 min |
| B | the breaker's decision map: {freeze, neutral lane, pushed-on lane} × levels {4, 8} × {equal, edge, ×1.76} × breaker HP {1.0, 0.7, 0.45}; freezer {policy, guard_sel, guard_walk*} × breaker {policy, commit, contest_sel, concede, concede3*, reset}, K 24, H 50 | 7.0M | 23 min |
| N | natural early (level ≤ 9) freeze states of each C1 policy's own self-play, 12 favourable + 12 other per policy, full matrix (5 × 5), K 16, H 50 | 2.9M | 10 min |
| Ab | the A core cells in the baseline environment (reference) | 2.3M | 10 min |
| confirm | independent rollout seeds, K 48: every A cell's freezer plans against the breaker's policy, the 11 selected B contrasts (no N kind met the selection rule) | 2.4M | 13 min |
| C1 verify | specialists vs the general policy as the stronger side, 27 M4.6a gap cells + 14 plausible cells, both environments, K 16, H 50 | 1.6M | 12 min |
| C2 decomposition | baseline environment, pairs (g_t, ext/baseline:t), 22 cells, 22 switch / substitution candidates, K 32, H 60 | 2.8M | 17 min |
| critic | calibration: 528 state-policy jobs, K 32, H 300 (natural states H 500), no bootstrap; ranking: 240 decision states × 5–7 one-turn actions, K 16, H 150 | 5.7M + 3.5M (+0.6M exploratory) | 40 + 14 (+3) min |

Everything ran on one 6-core machine with 6–9 worker processes (about 270 MB each) and an exact lane memo; results do
not depend on the memo. Every comparison is paired: within a job all candidates share the chance stream and each
seat's sampling uniforms, rollout by rollout.

### 1.2 Plans

Every plan restricts the policy's own action distribution to a slice of components chosen by a start-of-turn rule and
samples it with the policy's own uniform. Outside its active turns (condition false, HP at or below its release level,
absent, or after n turns) the policy plays, so Recall, disengagement and conversion are always available.

| side | plan | rule while active | n, condition, release |
|---|---|---|---|
| freezer | guard_sel | wave Last Hit / Thin; Trade iff the breaker walked up, is catchable without a dive, our HP ≥ 50% and our HP-aware duel edge ≥ 0; else Hold | 20, while A holds a freeze, 40% |
| | guard_walk* | the same without the edge condition (trade any walk-up) | 20, while held, 40% |
| | passive | wave Last Hit / Thin; Hold | 20, while held, 40% |
| | push | wave Shove (the policy's behaviour) | 2 turns |
| breaker | commit | Shove + Hold | 10, while frozen on, 25% |
| | contest_sel | Shove + Hold iff our HP ≥ 60% and the freezer is away, below 50% or without a duel edge on us; else Last Hit + Safe | 10, while frozen on, 30% |
| | concede | Last Hit + Safe until the freezer is away or below 50% HP | 10, while frozen on, 30% |
| | concede3* | the same, at most 3 turns | 3, while frozen on, 30% |
| | reset | walk back and Recall when legal, then the policy | 10 |

### 1.3 Deviations and amendments (dated in `preregistration.json`, each before the affected stage)

| item | what happened |
|---|---|
| amendment 1 | critic budget: the baseline reference uses the gap and freeze cells only (natural states in C1 only); ranking K 16 |
| amendment 2 | `concede3` (a brief concession) added to B, because A showed the 10-turn concession strongly dominated and a brief concession is the fair test of "concede temporarily" |
| amendment 3 | `guard_walk` added (A2, B, N). A single-game trace showed that `guard_sel` does not trade the turn-0 walk-up at equal power or at the plausible edge with both champions healthy: in a freeze the defender's HP-aware duel edge is slightly negative (−0.02 equal, −0.002 edge) because of the minion hit modifier, while that turn decides whether the freeze survives (§3.1) |
| amendment 4 | the confirmation sample replicates every A cell's freezer plans against the breaker's policy (not only the pre-registered L3 / L5 edge and plaus cells), so class-level contrasts are confirmed on independent seeds |
| natural-state confirmation | planned on N's states with new rollout seeds; no N kind met the selection rule, so none ran |
| exploratory follow-up (dated) | after reading the critic ranking: the weak side's Yield codes at the Recall windows (§6.2), because the actor's dominant choice was not in the tested set |
| a confirmation bug | the first confirmation runs stopped on the B jobs against `guard_walk` (the plan dictionary lacked it); fixed and re-run for the missing jobs only (every job is keyed by its path, results are deterministic) |
| duplicate runners | the confirmation and the critic ranking each ran with two runners for speed (per-process temporary files); identical jobs give identical files |
| index | `docs/README.md` and the README do not list the M4.6 reports (M4.6c did not add them either); only `docs/m4/README.md` was updated |

## 2. Freezes in natural play (descriptive; accounting, not causal)

Freeze *state* (unchanged metric, `m44.records.holds_freeze`): both fronts alive and in contact, the enemy front at the
freezer's own −1 or deeper. An *episode* is a maximal run of turns that start in that state for the same freezer.
C1 self-play: 7,018 episodes in 115,200 turns (6.1 per 100 turns; 5.4% of turns); baseline 6,628 (5.8 per 100).

**Most freeze states are not intentional freezes.**

| C1 (baseline similar) | value |
|---|---|
| episodes that began right after the breaker's own Shove / Slow Push ("pushed-in") | 87% |
| single-turn episodes; share of freeze-state turns in them | 68%; 38% |
| episodes of ≥ 4 turns ("maintained": two waves arrived) | 10% (0.62 per 100 turns) |
| the freezer's own wave choice while in the state: Shove / Slow Push / Thin / Last Hit | 61% / 12% / 12% / 14% |
| episodes in which the freezer preserved the wave (Last Hit / Thin on ≥ 75% of its turns) | 16% |
| "favourable" episodes (freezer present, HP ≥ 60%, duel ≥ ×1.10, held wave ≥ 9 HP) | 2.6% (0.16 per 100 turns) |

The state mostly records the breaker's own push arriving at the freezer's tower, and the policy, as freezer, usually
releases it at once with its own Shove.

**What makes a freeze last: the breaker's HP, not the freezer's strength.**

| breaker's HP at the start | episodes | mean length | ≥ 4 turns | freezer − breaker earned (episode + 20 turns) | breaker deaths |
|---|---|---|---|---|---|
| < 45% | 795 | 3.4 | 42% | +1.69 | 0.42 |
| 45–70% | 1,355 | 2.0 | 15% | +0.99 | 0.19 |
| ≥ 70% | 4,550 | 1.4 | 3% | +0.50 | 0.10 |

By the freezer's level: L1–5 freezes hold larger enemy waves (13 HP against 6) and last longer (3.0 turns, 23% maintained;
52% end with the held wave crashing in), L14–18 freezes are short (1.6 turns). By strength: freezers ≥ ×1.25 last 2.4
turns (24% maintained) against 1.75 at even strength. The breaker's policy at a freeze: at ≥ 70% HP it Shoves 48–66% and
almost never steps back (2–3% Safe); below 45% it steps back (39–44%) or recalls (38–53%).

**Natural early strength differences** (both present, turns 5–100; full table in the generated report):

| lower level | duel ≥ ×1.10 | ≥ ×1.25 | ≥ ×1.5 | HP gap median / q90 | HP-adjusted ratio ≥ 2 |
|---|---|---|---|---|---|
| 1–5 | 36% | 4% | 0.0% | 0.18 / 0.54 | 14% |
| 6–9 | 57% | 19% | 1.6% | 0.25 / 0.63 | 25% |
| 10–13 | 48% | 9% | 0.2% | 0.22 / 0.61 | 21% |

So the plausible early edge is about ×1.2–1.3 in duel terms (one level plus about 3 invested resources: the "edge"
condition) together with an HP gap of about 0.4 (the "plaus" condition). ×1.76 in investment (duel ×3.1) is an extreme
diagnostic.

**The existing M4.6c constructed-freeze records, split by cell** (4.096M policies; G50 candidate − policy): the rigid
guard (Thin + Trade every held turn) holds the wave 9–12 of 20 turns at level 4 against 2.5–5 under the policy and
triples the breaker's deaths, yet is −1.1 to +0.4 in value; passive keeping is −0.0 to −1.4; the breaker's step-back is
−1.1 to −3.1 everywhere. A freeze *start* itself is worth +0.3 to +2.5 to the freezer under policy play, even though
the freeze rarely survives: the breaker's held wave dies on the freezer's side.

## 3. Workstream A — can defending a favourable early freeze pay?

### 3.1 One turn, deterministic: when a guard can hold

From each A cell, B plays the walk-up Shove; A plays Thin + Hold (keep), Thin + Trade (guard) or Last Hit + Hold; 200
chance draws each (the outcome turned out not to depend on chance).

| | passive freeze survives the turn | guarded freeze survives the turn |
|---|---|---|
| C1, levels 3–8, every strength condition (breaker capacity C_B ≤ 7) | 0% | **100%** |
| C1, level 10 (C_B 9) | 0% | 0% (100% at ×1.76, where C_B is 6) |
| baseline, levels 3–6 (C_B ≤ 7) | 0% | 100% |
| baseline, level 8 and above (C_B ≥ 8) | 0% | 0% (×1.76: 100%) |

The mechanism is M4.6a's E1: a guard steps into the held wave, which then fires at the guard instead of the freezer's
8-HP wave, so only the breaker's own wave clear hits that wave. **A freeze can be defended exactly while the breaker's
capacity is below the freezer's wave (C ≤ 7).** Under C1 that lasts to about level 8, two levels longer than under the
baseline, because C1 is weaker at small investment. A passive freeze never survives a walk-up Shove.

### 3.2 Against the learned breaker: freezer plans

Discovery seeds. G50, plan − policy, against the breaker's policy (C1; interval over the six policies). Held = A's freeze turns in the
first 20 (policy / guard_sel / guard_walk* / passive / push). The last column is the freezer's strategy in the solved
5 × 5 restricted game (policy-mean payoffs).

| cell | policy (abs) | guard_sel | guard_walk* | passive | push | held /20 | restricted equilibrium (freezer) |
|---|---|---|---|---|---|---|---|
| L3 equal | +1.89 | +0.25 ± 0.67 | +0.15 ± 0.50 | -0.24 ± 0.70 | -0.10 ± 0.59 | 5.9 / 8.0 / 12.0 / 6.7 / 6.4 | guard_sel 1.00 |
| L3 inv | +2.13 | +0.60 ± 0.74 | +0.26 ± 0.37 | +0.06 ± 0.57 | -0.07 ± 0.41 | 6.9 / 9.2 / 12.8 / 7.3 / 6.8 | guard_sel 0.09, guard_walk* 0.91 |
| L3 lvl | +2.35 | -0.15 ± 0.59 | +0.05 ± 0.55 | **-0.36 ± 0.22** | -0.19 ± 0.59 | 6.9 / 9.3 / 12.7 / 7.4 / 6.6 | guard_walk* 1.00 |
| L3 hp (B 60%) | +2.46 | **+0.43 ± 0.21** | **+0.40 ± 0.30** | +0.26 ± 0.32 | +0.27 ± 0.35 | 5.7 / 12.4 / 13.0 / 6.6 / 6.0 | guard_sel 1.00 |
| L3 edge | +2.50 | +0.05 ± 0.76 | +0.22 ± 0.77 | -0.14 ± 0.79 | -0.09 ± 0.26 | 7.0 / 9.5 / 12.5 / 7.2 / 6.6 | guard_walk* 1.00 |
| L3 plaus (B 60%) | +3.13 | +0.31 ± 0.41 | +0.34 ± 0.40 | -0.20 ± 0.74 | -0.10 ± 0.23 | 6.6 / 12.4 / 12.7 / 7.8 / 7.2 | guard_walk* 1.00 |
| L3 brk | +1.42 | -0.27 ± 0.48 | -0.05 ± 0.72 | -0.33 ± 0.53 | -0.09 ± 0.28 | 5.7 / 6.9 / 10.9 / 5.9 / 5.8 | policy 0.77, push 0.23 |
| L3 x1.76 | +5.56 | +0.42 ± 0.88 | +0.43 ± 0.91 | -0.34 ± 0.48 | -0.09 ± 0.43 | 3.7 / 12.2 / 12.4 / 4.8 / 1.9 | guard_sel 0.48, guard_walk* 0.52 |
| L5 equal | +1.40 | -0.19 ± 0.40 | +0.19 ± 0.50 | **-0.35 ± 0.34** | +0.18 ± 0.35 | 4.2 / 4.7 / 10.0 / 3.6 / 4.0 | policy 0.22, push 0.78 |
| L5 inv | +1.95 | -0.30 ± 0.40 | +0.10 ± 0.54 | -0.39 ± 0.54 | +0.11 ± 0.12 | 3.8 / 4.8 / 10.2 / 3.5 / 4.0 | policy 0.59, push 0.41 |
| L5 lvl | +1.60 | -0.06 ± 0.69 | +0.22 ± 0.73 | -0.22 ± 0.56 | +0.14 ± 0.23 | 4.1 / 4.4 / 10.9 / 3.3 / 4.1 | guard_sel 0.69, push 0.31 |
| L5 hp (B 60%) | +2.24 | +0.27 ± 0.55 | **+0.81 ± 0.51** | +0.00 ± 0.38 | -0.15 ± 0.77 | 5.5 / 9.5 / 12.0 / 5.2 / 5.0 | policy 0.31, guard_walk* 0.69 |
| L5 edge | +2.18 | -0.18 ± 0.77 | +0.48 ± 0.49 | -0.39 ± 0.76 | +0.02 ± 0.23 | 3.9 / 4.8 / 11.0 / 3.2 / 4.1 | guard_sel 0.72, guard_walk* 0.28 |
| L5 plaus (B 60%) | +2.77 | **+0.67 ± 0.49** | **+0.71 ± 0.60** | -0.20 ± 0.54 | -0.16 ± 0.33 | 5.3 / 11.2 / 11.9 / 5.4 / 4.9 | guard_sel 0.17, guard_walk* 0.83 |
| L5 brk | +0.76 | -0.09 ± 0.29 | +0.36 ± 0.39 | -0.18 ± 0.40 | -0.13 ± 0.37 | 2.8 / 3.3 / 9.1 / 2.7 / 2.8 | guard_walk* 1.00 |
| L5 x1.76 | +5.60 | +0.53 ± 1.09 | +0.69 ± 1.12 | -0.15 ± 0.51 | -0.04 ± 0.19 | 2.3 / 12.5 / 12.8 / 3.4 / 0.9 | guard_sel 1.00 |
| L8 equal | +1.52 | +0.07 ± 0.50 | +0.16 ± 0.26 | +0.04 ± 0.50 | +0.07 ± 0.31 | 2.3 / 2.7 / 6.9 / 2.1 / 2.3 | push 0.60, guard_walk* 0.40 |
| L8 inv | +1.89 | -0.09 ± 0.52 | -0.08 ± 0.46 | -0.25 ± 0.37 | +0.07 ± 0.29 | 2.4 / 3.0 / 6.8 / 2.3 / 2.3 | push 1.00 |
| L8 lvl | +1.74 | -0.03 ± 0.63 | +0.10 ± 0.60 | -0.12 ± 0.49 | **+0.20 ± 0.17** | 2.3 / 2.9 / 7.1 / 2.0 / 2.2 | guard_sel 0.03, push 0.70, guard_walk* 0.27 |
| L8 hp (B 60%) | +2.31 | +0.05 ± 0.84 | +0.11 ± 0.44 | -0.17 ± 0.44 | -0.26 ± 0.59 | 3.6 / 8.4 / 9.1 / 3.5 / 2.8 | guard_sel 1.00 |
| L8 edge | +1.96 | +0.13 ± 0.94 | +0.07 ± 0.82 | +0.14 ± 0.77 | +0.13 ± 0.45 | 1.7 / 2.7 / 6.9 / 1.9 / 1.5 | guard_sel 0.66, passive 0.34 |
| L8 plaus (B 60%) | +2.85 | **+0.48 ± 0.35** | +0.45 ± 0.53 | -0.35 ± 0.77 | -0.15 ± 0.22 | 2.6 / 8.3 / 8.8 / 3.6 / 1.1 | guard_sel 0.49, guard_walk* 0.51 |
| L8 brk | +0.90 | -0.02 ± 0.67 | +0.15 ± 0.72 | -0.22 ± 0.72 | +0.21 ± 0.37 | 2.3 / 2.5 / 5.3 / 2.1 / 2.1 | policy 0.41, push 0.59 |
| L8 x1.76 | +5.66 | **+0.66 ± 0.31** | **+0.65 ± 0.30** | -0.31 ± 0.31 | +0.01 ± 0.27 | 1.7 / 10.9 / 11.1 / 2.2 / 1.0 | guard_walk* 1.00 |
| held 6, L5 equal | +1.39 | -0.28 ± 0.84 | +0.03 ± 0.83 | **-0.73 ± 0.50** | -0.33 ± 0.55 | 4.3 / 9.6 / 11.2 / 5.4 / 2.6 | guard_walk* 1.00 |
| held 6, L5 plaus (B 60%) | +2.38 | +0.41 ± 0.66 | +0.49 ± 0.56 | -0.32 ± 0.64 | **-0.69 ± 0.54** | 5.0 / 10.8 / 11.3 / 6.0 / 1.7 | guard_sel 1.00 |

Classes, discovery seeds (contrasts averaged over a class's cells within each policy):

| class | guard_sel − policy | guard_walk* − policy | passive − policy | push − policy | guard_sel − passive | guard_sel − push |
|---|---|---|---|---|---|---|
| level 3 (8 cells) | +0.20 ± 0.24 | +0.22 ± 0.39 | -0.16 ± 0.21 | -0.06 ± 0.19 | **+0.37 ± 0.24** | **+0.26 ± 0.16** |
| level 5 | +0.08 ± 0.34 | **+0.45 ± 0.23** | **-0.24 ± 0.18** | -0.00 ± 0.21 | **+0.32 ± 0.20** | +0.09 ± 0.34 |
| level 8 | +0.16 ± 0.22 | **+0.20 ± 0.17** | -0.15 ± 0.17 | +0.04 ± 0.12 | **+0.31 ± 0.17** | +0.12 ± 0.30 |
| equal power | +0.04 ± 0.31 | **+0.17 ± 0.11** | -0.18 ± 0.33 | +0.05 ± 0.29 | **+0.23 ± 0.18** | -0.01 ± 0.37 |
| +3 invested | +0.07 ± 0.11 | +0.09 ± 0.36 | -0.19 ± 0.25 | +0.04 ± 0.09 | **+0.26 ± 0.17** | +0.03 ± 0.14 |
| +1 level | -0.08 ± 0.38 | +0.12 ± 0.27 | -0.23 ± 0.24 | +0.05 ± 0.18 | +0.15 ± 0.23 | -0.13 ± 0.30 |
| equal power, B at 60% | +0.25 ± 0.29 | **+0.44 ± 0.18** | +0.03 ± 0.23 | -0.05 ± 0.30 | **+0.22 ± 0.14** | **+0.29 ± 0.20** |
| edge (+1 level, +3 invested) | +0.00 ± 0.51 | **+0.26 ± 0.12** | -0.13 ± 0.50 | +0.02 ± 0.13 | +0.13 ± 0.16 | -0.02 ± 0.41 |
| plaus (edge, B at 60%) | **+0.49 ± 0.25** | **+0.50 ± 0.30** | -0.25 ± 0.50 | -0.14 ± 0.16 | **+0.74 ± 0.35** | **+0.62 ± 0.31** |
| breaker has the edge (control) | -0.13 ± 0.36 | +0.15 ± 0.57 | -0.25 ± 0.35 | -0.00 ± 0.28 | +0.12 ± 0.21 | -0.12 ± 0.26 |
| ×1.76 (extreme) | +0.54 ± 0.58 | +0.59 ± 0.60 | **-0.27 ± 0.20** | -0.04 ± 0.21 | **+0.80 ± 0.54** | **+0.58 ± 0.40** |
| edge and plaus at levels 3 and 5 | +0.21 ± 0.28 | **+0.44 ± 0.33** | **-0.23 ± 0.14** | -0.08 ± 0.09 | **+0.45 ± 0.20** | +0.30 ± 0.31 |
| held wave 6 (level 5) | +0.06 ± 0.61 | +0.26 ± 0.38 | **-0.52 ± 0.42** | **-0.51 ± 0.29** | **+0.59 ± 0.43** | **+0.58 ± 0.34** |

**Confirmation on independent seeds** (K 48, every A cell; these are the established estimates):

| class | guard_sel − policy | guard_walk* − policy | passive − policy | push − policy | guard_sel − passive | guard_sel − push | guard_walk* − push |
|---|---|---|---|---|---|---|---|
| all held-12 cells (24) | +0.05 ± 0.21 | +0.07 ± 0.19 | −0.20 ± 0.24 | −0.01 ± 0.09 | **+0.25 ± 0.16** | +0.06 ± 0.18 | +0.08 ± 0.15 |
| level 3 / 5 / 8 | +0.12 / +0.07 / −0.04 | +0.02 / +0.26 / −0.06 | −0.14 / −0.21 / −0.24 | +0.01 / +0.02 / −0.05 | **+0.26** / **+0.28** / +0.20 | +0.11 / +0.06 / +0.01 | +0.00 / +0.24 / −0.01 |
| equal power | −0.19 ± 0.34 | **−0.31 ± 0.28** | −0.27 ± 0.45 | +0.01 ± 0.21 | +0.07 ± 0.14 | −0.20 ± 0.27 | **−0.32 ± 0.14** |
| edge (+1 level, +3 invested) | −0.01 ± 0.31 | −0.06 ± 0.23 | −0.18 ± 0.35 | −0.02 ± 0.06 | +0.16 ± 0.19 | +0.01 ± 0.33 | −0.04 ± 0.18 |
| equal power, B at 60% | +0.30 ± 0.32 | **+0.47 ± 0.31** | −0.13 ± 0.16 | +0.00 ± 0.10 | **+0.44 ± 0.32** | **+0.30 ± 0.28** | **+0.46 ± 0.31** |
| plaus (edge, B at 60%) | +0.34 ± 0.39 | +0.41 ± 0.41 | −0.22 ± 0.32 | −0.09 ± 0.30 | **+0.56 ± 0.37** | **+0.43 ± 0.30** | **+0.51 ± 0.35** |
| edge and plaus at levels 3 and 5 | +0.31 ± 0.31 | **+0.41 ± 0.33** | −0.11 ± 0.38 | +0.02 ± 0.19 | **+0.42 ± 0.19** | +0.29 ± 0.31 | **+0.38 ± 0.21** |
| ×1.76 (extreme) | +0.56 ± 0.56 | **+0.59 ± 0.49** | −0.11 ± 0.32 | −0.01 ± 0.27 | **+0.67 ± 0.58** | **+0.57 ± 0.50** | **+0.60 ± 0.41** |
| breaker has the edge (control) | −0.17 ± 0.22 | −0.17 ± 0.28 | −0.15 ± 0.28 | +0.04 ± 0.06 | −0.02 ± 0.18 | −0.21 ± 0.19 | −0.21 ± 0.28 |

Per cell, the confirmed guard gains over the policy are level 3 plaus (+0.40 ± 0.36 guard_sel, +0.43 ± 0.38
guard_walk), level 5 plaus (guard_walk +0.80 ± 0.63; guard_sel +0.64 ± 0.61) and level 8 with the breaker at 60%
(guard_walk +0.33 ± 0.20); level 8 plaus is ≈ 0 (−0.02 ± 0.48).

Reading, by the pre-registered rule ("defending pays" = guard − max(passive, push) > 0 with the interval excluding 0,
and guard − policy ≥ 0), on the confirmation estimates:
- **Guarding always beats keeping the freeze passively** (+0.25 ± 0.16 over all cells; +0.26 to +0.28 at levels 3
  and 5). Passive keeping is never better than the policy (−0.11 to −0.27): an undefended freeze simply breaks, and
  keeping one's own wave small meanwhile costs farm.
- **Defending pays when the freezer has an exchange edge from HP.** With the breaker at 60%, guarding beats pushing by
  +0.30 to +0.51 and the policy by +0.30 to +0.47 (the trade-every-walk-up guard significant); the rule is met for the
  HP and plaus classes and for ×1.76. It is about one turn of income per favourable freeze start.
- **With both champions healthy, defending is worth nothing over the policy or a push**, even with the plausible
  ×1.25 edge (−0.06 to +0.01). The pre-registered selective guard does not trade the walk-up there (§1.3). The
  exploratory `guard_walk`, which does, holds the wave 7–13 of 20 turns, but its discovery gains (+0.17 equal, +0.26
  edge, +0.45 over level 5) did not replicate: −0.31 ± 0.28 at equal power, −0.06 ± 0.23 with the edge. This is the
  case the pre-registration warned about: the best of several noisy alternatives, selected on the discovery seeds.
- **Level.** Under C1 the guard still holds through level 8 (§3.1), but the wave is held for fewer turns there (5–9 of
  20) and no level-8 class gain is confirmed.
- **Small held wave (6 HP, level 5).** Guarding beats passive and push by about +0.6 in discovery; against the policy it
  is ±0.1 to +0.4.
- **The breaker-advantaged control** shows no defence value (−0.17).

### 3.3 Keeping the wave, denying resources, and the freezer's net

Channels over 50 turns, plan − policy, both against the breaker's policy (undiscounted, resources unless stated):

| cell | plan | held /20 | B farm | B available farm | B lost while away | B XP | B deaths | A HP lost (bars) | A bounty | G10 | G50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L5 plaus | guard_sel | 11.2 (5.3) | −0.27 | +0.01 | +0.22 | −2.6 | +0.20 | +0.04 | +0.40 | −0.05 | **+0.67** |
| L5 plaus | passive | 5.4 | +0.39 | +0.09 | −0.20 | +2.1 | −0.03 | −0.29 | −0.05 | −0.40 | −0.20 |
| L5 equal | guard_walk* | 10.0 (4.2) | −0.13 | +0.04 | +0.22 | −2.1 | +0.09 | +0.25 | +0.19 | −0.19 | +0.19 |
| L5 edge | guard_walk* | 11.0 (3.9) | −0.29 | −0.13 | +0.23 | −4.3 | +0.10 | +0.25 | +0.21 | −0.28 | +0.48 |
| L5 ×1.76 | guard_sel | 12.5 (2.3) | −0.49 | +0.01 | +0.37 | −6.2 | +0.28 | +0.14 | +0.56 | −0.24 | +0.53 |
| L8 equal | guard_walk* | 6.9 (2.3) | −0.17 | +0.06 | +0.31 | −2.7 | +0.17 | +0.06 | +0.33 | −0.58 | +0.16 |

(held: the plan, with the policy in brackets.)
- **Keeping the wave in place** is easy to buy: a guard doubles or triples the held turns.
- **Denial** comes through absence and deaths, not posture: the breaker's farm lost while present barely moves, its
  farm lost while away, its deaths and the freezer's bounty rise, and it loses 2–6 XP (XP is not rewarded but feeds
  levels).
- **The freezer pays first and collects later**: G10 is −0.05 to −0.58 for every guard (HP and its own farm while it
  trades), and the value appears by G50.
- **Delayed or permanent?** Over 150 turns (AL, level 5 plaus) guard_sel − policy grows from +0.67 (G50) to +0.96
  (G100) and +1.06 ± 0.92 (G150). The breaker's available farm is unchanged while its collected farm stays 0.4 lower
  at 100 turns: the denied gold is lost, not delayed. Passive keeping recovers to +0.18 by G150 (its early cost was
  a delay). At level 5 equal, pushing is the best freezer plan at every horizon (+0.33 ± 0.33 at G150).

### 3.4 Robustness: does holding have value if breaking is the best reply?

The restricted game is solved per cell on the discovery payoffs (policy means), so it is descriptive:
- **The breaker's best reply to both guards is to keep breaking** (commit or its policy) in 19 of the 26 cells; the
  others are low-HP cells where it resets.
- **Yet holding has value.** A guard plan carries at least half the freezer's equilibrium weight in 19 of 26 cells and
  has the best maximin in 19 (15 of 26 in the pre-registered 4 × 5 game without `guard_walk`). Breaking is the right
  reply *and* the freeze can be worth defending: a defended freeze costs the breaker HP, deaths and XP even when it
  plays well.
- **Where no guard carries weight:** level-5 / level-8 cells at equal power or +3 invested, where pushing or the
  policy is the equilibrium, and the breaker-advantaged controls. These agree with the confirmation: without an HP
  edge, defending buys nothing.

### 3.5 Natural early freeze states (N)

Each C1 policy's own s0 self-play supplied 12 favourable and 12 other early freeze states (start-of-turn freeze
state, levels ≤ 9; favourable = freezer present, HP ≥ 60%, duel ≥ ×1.10, held wave ≥ 9 HP). Favourable states:
median level 7, duel ×1.18 (q10–q90 ×1.12–1.36), breaker HP median 0.74 (0.25–1.0), held wave 12 (10–20). Full
5 × 5 matrix, K 16, G50:

| states | policy (abs) | guard_sel − policy | guard_walk* − policy | passive − policy | push − policy | guard_sel − passive | held /20 (policy / guard_sel / guard_walk*) | restricted equilibrium (freezer) |
|---|---|---|---|---|---|---|---|---|
| favourable (72) | +2.05 ± 0.27 | −0.14 ± 0.33 | −0.05 ± 0.39 | −0.23 ± 0.31 | −0.07 ± 0.14 | +0.09 ± 0.12 | 2.9 / 4.5 / 5.3 | push 0.92, policy 0.08 |
| other (72) | +0.90 ± 0.39 | −0.10 ± 0.21 | +0.06 ± 0.28 | **−0.28 ± 0.23** | −0.03 ± 0.12 | **+0.18 ± 0.16** | 3.2 / 5.3 / 6.3 | push 1.00 |

Breaker options against the freezer's policy (B's units), favourable / other: commit +0.07 ± 0.16 / +0.11 ± 0.18;
concede −0.82 / −0.61 and reset −0.38 / −0.47 (both significant). No natural kind met the confirmation selection rule.

- **In natural early freeze states defending does not beat the policy**, favourable or not. Guarding still beats
  passive keeping (+0.09 to +0.34), and pushing is the restricted-game choice.
- **Why the constructed result does not carry over.** The constructed cell is the best case for a freezer: a fresh
  8-HP wave in its tower cell, the breaker already walked up, a reinforcement due next turn. In the natural states the
  freezer's own wave is usually worn down (below a fresh 8 HP in 54–81% of them, 5 HP or less in about half) and the
  breaker has walked up in only 43–51%. From those exact states a guard keeps the freeze against a walk-up Shove 50–63% of the
  time (100% in every constructed cell up to level 8), and the guards hold the wave only about 5 of 20 turns.
- **The breaker's policy is right in natural freezes**: contest; every concession and the reset cost 0.4–0.8.

### 3.6 The baseline environment (Ab, reference)

The A core cells (equal / edge / plaus / ×1.76 at levels 3, 5, 8) with the baseline 8.192M policies in the baseline
environment, discovery seeds, guard_sel − policy (G50): equal −0.16 ± 0.30, edge +0.01 ± 0.29, **plaus +0.36 ± 0.17**,
×1.76 +0.82 ± 0.62; guard_sel − passive +0.12 to +0.75; held 4.6 / 5.5 / 9.0 / 11.1 of 20 turns. The same pattern as
C1: defending pays only with an HP or a large strength edge. The deterministic threshold differs (the baseline guard
holds the walk-up only through level 6, §3.1), and at level 8 equal power the baseline guard holds 2.5 of 20 turns
against 2.7 under C1 for guard_sel. **The freeze result does not depend on the investment curve**, as M4.6c found.

### 3.7 Answer to A

**Yes, in a narrow window; no, as a general strategy of the current game.**

- **Where defending pays.** A freezer that holds a suitable early freeze (a fresh own wave in its tower cell, the
  breaker walked up, levels 3–5 and up to 8 under C1) gains about +0.3 to +0.5 per 50 turns over pushing or its policy
  by trading the walk-up. This holds when the breaker is already damaged (60% HP), and at ×1.76. The gain is real
  denial: the breaker's farm is permanently lost, it loses XP and dies more. It grows to about +1 over 150 turns.
- **Where it does not pay.**
  - With both champions healthy, even at the natural 90th-percentile power edge (×1.25), defending is worth nothing over
    pushing, and passive keeping always costs about 0.2.
  - Once the breaker's wave clear reaches 8 HP (baseline level 8, C1 level 10) a freeze cannot be defended at all.
  - In natural early freeze states, which are mostly the breaker's own push arriving with the freezer's wave already
    worn, no defence beats the policy.
- **Why.**
  - The contest is decided in the first turn by capacity against the freezer's wave. A guard turns that turn only
    because the held wave fires at the guard.
  - The value of holding is what the breaker loses in HP, deaths and XP while it keeps contesting. That is enough only
    when the breaker is already damaged.
  - The policy's habit of shoving out of its freezes is right in natural freeze states. It leaves about +0.4 in the
    constructed favourable case, a state the natural game seldom presents (favourable freeze episodes: 0.16 per 100
    turns).
- **Keeping the wave in place, denying resources and the net are three different things.** Guards double or triple the
  held turns everywhere, deny a little farm and some XP everywhere, and improve the net only with an HP edge.

## 4. Workstream B — contest, concede or reset?

### 4.1 The decision map

B's units (−ΔG50, option − B's policy), mean over the six cells of each row group (2 levels × {equal, edge, ×1.76});
"best" counts cells where the option has the highest mean (most are within the interval of the policy, see the
generated table):

| freezer | lane | B's HP | best option (cells of 6) | commit | contest_sel | concede (10) | concede3* | reset |
|---|---|---|---|---|---|---|---|---|
| policy | frozen on (12-HP held wave) | 100% | policy 5, commit 1 | -0.16 | -0.39 | -0.92 | -0.71 | -0.94 |
| policy | frozen on (12-HP held wave) | 70% | commit 5, policy 1 | +0.08 | -0.65 | -0.90 | -0.49 | -0.49 |
| policy | frozen on (12-HP held wave) | 45% | commit 5, policy 1 | +0.22 | -0.74 | -0.71 | -0.39 | -0.30 |
| policy | neutral lane | 100% | policy 4, contest_sel 1, concede3* 1 | -0.24 | -0.27 | -0.91 | -0.31 | -0.40 |
| policy | neutral lane | 70% | reset 3, policy 2, concede3* 1 | -0.13 | -0.69 | -0.74 | -0.05 | +0.02 |
| policy | neutral lane | 45% | reset 4, policy 1, concede3* 1 | -0.49 | -0.62 | -0.64 | -0.05 | +0.27 |
| policy | A pushing | 100% | commit 4, contest_sel 1, policy 1 | +0.13 | -0.14 | -0.75 | -0.61 | -0.78 |
| policy | A pushing | 70% | commit 3, policy 3 | +0.03 | -1.08 | -1.25 | -0.68 | -0.70 |
| policy | A pushing | 45% | policy 5, commit 1 | -0.26 | -1.07 | -1.07 | -0.52 | -0.50 |
| guard_sel | frozen on (12-HP held wave) | 100% | commit 3, policy 2, concede3* 1 | -0.07 | -0.31 | -2.75 | -0.59 | -1.00 |
| guard_sel | frozen on (12-HP held wave) | 70% | commit 2, concede3* 2, reset 1, policy 1 | -0.08 | -0.82 | -1.79 | -0.05 | -0.02 |
| guard_sel | frozen on (12-HP held wave) | 45% | reset 6 | -0.31 | -1.26 | -1.39 | -0.34 | +0.47 |
| guard_sel | neutral lane | 100% | policy 3, commit 1, contest_sel 1, concede3* 1 | -0.21 | -0.26 | -0.90 | -0.27 | -0.35 |
| guard_sel | neutral lane | 70% | reset 3, policy 2, concede3* 1 | -0.24 | -0.67 | -0.72 | -0.05 | +0.01 |
| guard_sel | neutral lane | 45% | reset 5, policy 1 | -0.50 | -0.60 | -0.63 | -0.11 | +0.31 |
| guard_sel | A pushing | 100% | commit 6 | +0.16 | -0.22 | -0.80 | -0.63 | -0.89 |
| guard_sel | A pushing | 70% | commit 4, policy 2 | +0.02 | -1.08 | -1.25 | -0.69 | -0.75 |
| guard_sel | A pushing | 45% | policy 5, commit 1 | -0.28 | -1.05 | -1.05 | -0.53 | -0.53 |
| guard_walk* | frozen on (12-HP held wave) | 100% | commit 5, concede3* 1 | +0.14 | -0.31 | -1.96 | -0.18 | -0.73 |
| guard_walk* | frozen on (12-HP held wave) | 70% | commit 3, policy 1, reset 1, concede3* 1 | -0.00 | -0.96 | -1.68 | -0.01 | -0.04 |
| guard_walk* | frozen on (12-HP held wave) | 45% | reset 5, policy 1 | -0.32 | -1.24 | -1.39 | -0.39 | +0.34 |
| guard_walk* | neutral lane | 100% | policy 4, contest_sel 1, concede3* 1 | -0.28 | -0.32 | -0.93 | -0.31 | -0.37 |
| guard_walk* | neutral lane | 70% | reset 3, policy 2, concede3* 1 | -0.25 | -0.67 | -0.72 | -0.04 | +0.02 |
| guard_walk* | neutral lane | 45% | reset 5, policy 1 | -0.49 | -0.58 | -0.61 | -0.08 | +0.32 |
| guard_walk* | A pushing | 100% | commit 5, contest_sel 1 | +0.17 | -0.13 | -0.77 | -0.59 | -0.80 |
| guard_walk* | A pushing | 70% | commit 4, policy 2 | +0.00 | -1.08 | -1.24 | -0.70 | -0.70 |
| guard_walk* | A pushing | 45% | policy 5, commit 1 | -0.31 | -1.05 | -1.04 | -0.52 | -0.49 |

Reading:
- **Healthy and frozen on: contest.** Against the freezer's policy the breaker's own play and commit are equivalent and
  better than every concession. A brief (3-turn) concession costs 0.4–0.7, a 10-turn one 0.7–0.9, a reset 0.3–0.9.
- **Against a guard while healthy: still contest**, and commit slightly beats the policy (+0.14 against guard_walk).
- **At 45% HP against a guard, resetting is at least as good as contesting** in discovery (best in 11 of 12 guarded
  freeze cells, +0.34 to +0.47 on average), but the selected cells did not confirm it (§4.3): +0.10 to +0.25 on
  independent seeds, none significant. Read: reset and contest are close there.
- **At 45% HP in a neutral lane** the same: reset leads in discovery (+0.27 to +0.32) and is +0.10 to +0.24 (n.s.) on
  the confirmation seeds.
- **Being pushed on (A's push):** contest at every HP; resetting or conceding there costs about 0.5 even at 45%,
  because the arriving wave is lost.
- **Selective contest** (`contest_sel`) is dominated wherever its HP rule makes it concede, i.e. at 45–70% HP.

### 4.2 The sequence behind the options (B's channels over 50 turns, option − B's policy)

| cell | freezer | option | B earned | B farm | B XP | B deaths | B turns away | B HP lost (bars) | A earned |
|---|---|---|---|---|---|---|---|---|---|
| L4 equal, B at 100% | policy | concede3 | −0.92 | −0.68 | −6.8 | −0.15 | −0.6 | −0.21 | +0.16 |
| L4 equal, B at 100% | policy | reset | −0.89 | −0.82 | −6.2 | −0.09 | +1.0 | −0.17 | +0.17 |
| L4 equal, B at 45% | guard_walk* | policy (contest) | (reference) | | | | | | |
| L4 equal, B at 45% | guard_walk* | commit | −0.14 | −0.38 | −3.3 | +0.30 | +0.7 | +0.29 | +0.42 |
| L4 equal, B at 45% | guard_walk* | reset | −0.40 | −0.16 | −0.9 | −0.20 | +0.8 | +0.04 | −0.25 |
| L8 equal, B at 45% | guard_walk* | reset | −0.05 | +0.00 | +1.1 | −0.25 | +0.5 | +0.10 | −0.67 |
| neutral L8 equal, B at 45% | policy | reset | +0.22 | +0.12 | +1.5 | −0.25 | −0.3 | −0.10 | −0.61 |

- **Why conceding a freeze is expensive.** Standing back at its own −1 the breaker is out of reach of the freezer's dying
  minions (they die at its +2) and out of XP range, so it forfeits about 0.07 resources and 0.6 XP per turn, while the
  HP it saves (0.2 bars over 3 turns) is worth only about 0.25–0.3 resources at M4.6a's 1.2–1.5 per bar. The freeze
  persists while it stands back.
- **Why resetting is competitive at low HP against a guard** (discovery channels; the net gain did not confirm,
  §4.3). Contesting at 45% HP against a freezer that trades the walk-up costs deaths (+0.3) and the bounty. A reset
  removes them (deaths −0.2 to −0.25) and cuts the freezer's earnings by 0.25–0.67. The breaker gives up some farm while
  away, which roughly offsets the gain.
- **Resources missed versus delayed.** In every concession the breaker's *available* farm is unchanged (±0.1) while its
  collected farm falls: what it concedes is permanently missed, not delayed.

### 4.3 Is the policy materially inferior anywhere?

Pre-registered rule: best − policy ≥ 0.25 with the interval excluding 0 in discovery **and** on independent seeds
(K 48). Eleven B contrasts met the discovery selection rule (≥ 0.15, interval excluding 0):

| cell | freezer | option | discovery | confirmation | verdict |
|---|---|---|---|---|---|
| freeze L4 equal, B 100% | guard_walk* | commit | +0.36 ± 0.33 | **+0.82 ± 0.42** | policy materially inferior (against an exploratory opponent) |
| freeze L4 edge, B 100% | guard_walk* | commit | +0.37 ± 0.29 | **+0.57 ± 0.33** | policy materially inferior (against an exploratory opponent) |
| freeze L4 edge, B 45% | policy | commit | +0.65 ± 0.52 | +0.07 ± 0.35 | not confirmed |
| freeze L4 ×1.76, B 70% | guard_sel / guard_walk* | reset | +0.45 / +0.54 | +0.19 / +0.25 (n.s.) | not confirmed |
| freeze L8 edge, B 45% | guard_sel | reset | +0.47 ± 0.38 | +0.10 ± 0.38 | not confirmed |
| freeze L8 ×1.76, B 70% | guard_sel / guard_walk* | concede3* | +0.32 / +0.24 | −0.15 / −0.20 | not confirmed |
| neutral L8 equal, B 45% | policy / guard_sel / guard_walk* | reset | +0.72 / +0.77 / +0.80 | +0.24 / +0.10 / +0.18 (n.s.) | not confirmed |

- **Against its own self-play opponent (the freezer's policy) the breaker's policy is nowhere materially inferior.**
  The one discovery candidate (commit at 45% HP, level 4 edge) fell to +0.07.
- **Against a freezer that trades every walk-up, the breaker's policy under-commits** when healthy: walking up and
  Shoving every turn beats it by +0.6 to +0.8 at level 4. The policy is a best response to the freezer it trains
  against, which rarely guards; this is an exploitability gap against an unfamiliar strategy, not an error against its
  own opponent.
- **The concerning pattern (repeated contesting and dying for little) was not found.** Where contesting at low HP
  looked worse than resetting, the difference did not survive independent seeds. In natural freeze states the
  breaker's policy beats every alternative except commit, which ties it.

## 5. Workstream C — what do specialists do better?

### 5.1 The remaining gap, on comparable states

The specialist in seat A against the general policy in seat B, minus the general policy's own strong-side play from
the same state (G50; interval over the six general policies; the three specialists averaged):

| cells | baseline environment (specialists' own) | C1 (cross-environment for the specialists) |
|---|---|---|
| M4.6a gap cells, all 27 (×1.5 / ×1.76 / ×2.2, levels 6 / 10 / 14) | **+1.15 ± 0.28** | **+1.73 ± 0.15** |
| ×1.5 | **+0.44 ± 0.23** | **+0.53 ± 0.31** |
| ×1.76 | **+0.93 ± 0.25** | **+1.33 ± 0.17** |
| ×2.2 | **+2.08 ± 0.44** | **+3.33 ± 0.33** |
| plausible: edge + weaker side at 60% HP (neutral, pushing, freeze; 6 cells) | −0.38 ± 0.42 | **−0.40 ± 0.12** |
| plausible: edge, full HP (4 cells) | −0.27 ± 0.44 | **−0.52 ± 0.20** |
| plausible: equal power, weaker side at 60% (4 cells) | −0.37 ± 0.44 | **−0.56 ± 0.27** |

- **The gap persists for large constructed gaps** against the 8.192M general policies: +1.15 in the specialists' own
  environment (M4.6c measured +1.34 against the 4.096M baseline arm, M4.6a +1.48 against the M4.5c finals), and it
  grows steeply with the gap.
- **On plausible early advantages the specialists are worse than the general policies** (−0.4 to −0.6, significant
  under C1). The specialists' aggression is calibrated to large gaps; at the advantages the natural game produces, the
  general policies' restraint is the better play.
- **Part of the C1 number is the environment, not the specialists.** The same constructed gold buys a larger strength
  gap under C1 at mid and late investment, and the general policies' own strong-side value is also higher there
  (+4.93 against +3.92). The specialists were never trained in C1.

### 5.2 What they do differently (accounting, gap cells, 50 turns, specialist − general)

| channel | baseline | C1 |
|---|---|---|
| A's trades attempted / connected | +7.9 / +7.0 | +7.8 / +6.3 |
| B's Recalls interrupted | +1.94 | +1.58 |
| B's deaths / turns away | +0.90 / +2.8 | +0.98 / +3.0 |
| B's earned gold | −1.66 | −1.90 |
| A's earned gold | −0.28 | +0.22 |
| A's deaths | +0.27 | +0.17 |

The specialist earns about what the general earns itself; its whole edge is denial through trades, interrupted Recalls,
deaths and absence (as in M4.6a). On the plausible cells it makes the same extra trades (+6 to +7 connected) and kills
somewhat more (+0.3 to +0.4), but its own deaths rise (+0.3 to +0.4) and its own earnings fall (−1.0 to −1.3): there the
aggression does not pay.

### 5.3 Which decisions explain the gap (baseline environment, 3 specialist / general pairs × 22 cells)

Two paired designs on common streams, the weak side always the general policy (pre-registration §4):
- *spec-first switch*: the specialist for the first k turns, then the general. Its increments are the
  performance-difference decomposition of the gap under the specialist's own state distribution.
- *one-turn substitution*: the general throughout, except the specialist's choice at turn k. It measures the
  specialist's choices at states the general itself reaches.

States are classed from the start-of-turn state only (no hindsight).

| quantity | gap cells (54 units) | plausible cells (12 units) |
|---|---|---|
| total gap, G50 | **+1.17 ± 0.20** | +0.16 ± 0.27 |
| one-turn substitutions summed over the 10 sampled turns of 0–20 | +0.24 ± 0.45 | +0.11 ± 0.74 |

By when and where the gap accrues (spec-first increments over all 66 units, 54 gap and 12 plausible cells, whose total
is +0.99 ± 0.19; interval over units):

| turns | 0–4 | 4–8 | 8–14 | 14–30 | 30–50 |
|---|---|---|---|---|---|
| increment | +0.16 | +0.27 | +0.22 | +0.25 | +0.09 |

| state class at the block start (specialist's path) | summed increment |
|---|---|
| trade window (catchable, weak side above 35% / not in a Recall window) | **+0.65 ± 0.18** |
| out of reach (weak side not catchable without a dive) | **+0.31 ± 0.20** |
| Recall window (weak side may Recall, ≤ 60% HP, catchable) | +0.07 ± 0.11 |
| kill window (≤ 35% HP, catchable) | −0.02 ± 0.04 |
| weak side away | −0.02 ± 0.09 |

One-turn substitution values at the general's own states, by the action pair (per occurrence, interval over units):

| state, specialist / general choice | value | rollout-turns |
|---|---|---|
| trade window, specialist Trades / general does not | +0.08 ± 0.12 | 3,434 |
| Recall window, specialist Trades / general does not | +0.11 ± 0.40 | 383 |
| kill window, specialist Trades / general does not | +1.05 ± 1.87 | 18 |
| trade window, both Trade (wave component differs) | +0.04 ± 0.05 | 3,182 |

- **The gap is not one decision.** It accrues at a nearly constant rate over the whole 50 turns and mostly in ordinary
  trade windows and while the weak side stays out of reach: the specialist trades more often and keeps the weak side
  away, turn after turn.
- **Recall interruption is not the explanation** under hindsight-free classification: Recall windows contribute
  +0.07 ± 0.11. The specialist's extra interrupted Recalls (+1.9 per 50 turns, §5.2) come from trading in ordinary
  windows when the weak side happens to recall, not from reading Recall windows better.
- **Choices at the same state.** On the general's own path a single specialist choice is worth about +0.02 per turn on
  average (+0.24 over 10 of the first 21 turns; roughly +0.5 over all 21, against the spec-first +0.71 over the same
  turns). That is consistent with most of the gap being choices the general could make at states it reaches, but the
  interval is too wide to separate this from the state-distribution effect.

## 6. The critic

### 6.1 Calibration against long realized returns (no bootstrap)

Each general policy against itself (the pairing its critic predicts), K 32 rollouts per state, 300 turns (natural
states 500), returns **without** the critic's tail. Units: critic outputs divided by the 0.1 reward scale (checked in
every `run.json`).

**The units and the truncation tail are verified:**
- across the 96 natural states the slope of the realized return on V is 0.985 (correlation 0.89, intercept −0.03);
- extending natural-state returns from 300 to 500 turns changes them by +0.00 ± 0.01;
- the critic's own tail at turn 300 averages −0.01;
- V_A + V_B ≈ 0 (zero-sum consistency within ±0.4).

Bias = V(s) − E[G_H] for the stronger / freezing seat A (the weak seat mirrors it):

| states (per policy) | C1: V_A | C1: realized G | **C1: bias** | baseline: bias |
|---|---|---|---|---|
| natural, even (8) | −0.11 | −0.23 | +0.12 ± 0.27 | |
| natural, invested lead ≥ 4 (8) | +0.23 | +0.29 | −0.06 ± 0.19 | |
| freeze, equal power (3) | +1.24 | +1.52 | −0.27 ± 0.48 | −0.32 ± 0.30 |
| freeze, edge (+1 level, +3 invested) | +1.26 | +2.15 | **−0.89 ± 0.34** | **−0.96 ± 0.48** |
| freeze, plaus (edge, B at 60%) | +1.73 | +2.80 | **−1.08 ± 0.40** | **−0.76 ± 0.27** |
| freeze, ×1.76 | +1.39 | +7.09 | **−5.71 ± 0.63** | **−4.28 ± 0.29** |
| neutral / pushing lane, equal | −0.28 | −0.84 | +0.57 ± 0.45 | |
| neutral / pushing lane, edge | −0.26 | +0.13 | −0.38 ± 0.47 | |
| neutral / pushing lane, ×1.76 | −0.11 | +5.07 | **−5.17 ± 0.60** | |
| M4.6a gap cells ×1.5 (9) | +0.28 | +3.88 | **−3.60 ± 0.52** | **−2.84 ± 0.30** |
| M4.6a gap cells ×1.76 (9) | +0.31 | +6.00 | **−5.69 ± 0.69** | **−4.30 ± 0.19** |

- **The critics are calibrated where the natural game goes**: natural even and lead states within ±0.12, slope ≈ 1.
- **They are almost blind to strength.**
  - The plausible edge (×1.25 duel) is valued +0.02 above the equal freeze; realized play gives it +0.63 (baseline
    +0.00 against +0.65).
  - A 40% HP gap is partly seen: +0.49 of a realized +1.28 (38%).
  - A ×1.76 gap is valued at +0.0 to +0.3, against a realized +5 to +7.
- Both environments show this. The C1 critics, whose training had 25% of transitions from archive starts with
  purchased leads and constructed gaps (M4.6c), are as blind as the baseline ones.
- **Correction to earlier readings.** M4.6a's −1.3 to −1.6 at ×1.76 (and M4.6c's "about −1 early") used a 50–100-turn
  return plus the critic's own bootstrapped tail and averaged over later turns. The start-state bias without the tail
  is three to four times larger.
- Constructed equal-power cells also carry a smaller, sign-varying bias (−0.3 to +0.6). Part of every constructed-cell
  bias is "unusual state" rather than strength; the strength-specific part is the edge − equal difference (−0.6 to
  −1.0).

### 6.2 Ranking of one-turn decisions

At 40 decision states (the A freeze cells for each seat and eight constructed Recall windows: neutral lane, weak side
at its −1 with 50% or 35% HP, the stronger side at the edge or ×1.76, levels 5 and 8), each candidate one-turn action
then the policy (K 16, 150 turns, no bootstrap) is compared with the critic's one-step look-ahead r₀ + γ V(s₁) and
with the actor's probability:

| decision | Spearman (critic vs rollouts) | sign agreement | same best action | actor mass on the rollout-best action |
|---|---|---|---|---|
| freezer's first move at a freeze (12 states) | 0.09 | 51% | 18% | 0.10 |
| breaker's first move at a freeze (12) | 0.38 | 63% | 35% | 0.18 |
| stronger side at a Recall window (8) | 0.43 | 67% | 58% | 0.14 |
| weak side at a Recall window (8) | 0.44 | 69% | 77% | 0.04 |

- **At a freeze the critic barely separates the freezer's options.** Its look-ahead advantages are all within ±0.2,
  while the rollouts spread them by ±1. It always ranks Shove first (+0.03), and so does the actor (40–55% on Shove).
- **It does rank the breaker's bad options**: stepping back and trading into the held wave are −0.4 to −0.6 by the
  critic and −0.8 to −1.7 by the rollouts.
- **At Recall windows the critic ranks sensibly.** For the stronger side, Trade over Hold or Shove agrees with the
  rollouts, and the actor already puts about 55% on Trade.
- **One profitable decision the actor rarely takes, which the critic sees** (exploratory follow-up with the codes the
  actor actually uses):
  - The state: the weak side at its own −1 with 35–50% HP, next to a stronger opponent.
  - The actor plays Yield (52–74%), which does not evade a Trade from a champion it starts in contact with. Safe does
    evade it.
  - Safe beats the actor's policy by +0.6 on average over the eight states (L8 ×1.76 at 35%: +0.99 ± 0.32; individual
    states are noisy). Yield is ≈ the policy.
  - The critic's look-ahead gives Safe +0.3 to +1.0, the right sign and size.
  - Yet the actor plays Safe 5% at 50% HP and 31–36% at 35%.

  These are rare, constructed states, so this is a demonstrated actor–critic mismatch in rare states. It is not a
  measured cost in natural play.

### 6.3 What is established, and what is not

- **Established:**
  - the critics are calibrated in natural states;
  - they assign almost no value to investment or level advantages and little to HP advantages, at constructed states,
    in both environments;
  - they rank freezer decisions at a freeze no better than chance;
  - in at least one rare state family they identify a better action that the actor rarely takes.
- **Not established:** that the critic's strength blindness *causes* the actor's gaps (the under-guarded favourable
  freeze, the large-gap conversion deficit, the Yield/Safe miss). It is the most direct candidate. The actor's gaps
  are also all in states the self-play distribution rarely contains, which by itself limits the training signal.
- **Not tested:** shared-policy interference (one network for both roles). Nothing here isolates it.

## 7. Mechanics versus learning

| observation | mechanics (structural) | learning (policy / critic / distribution) | evidence |
|---|---|---|---|
| a freeze can be defended only while the breaker's wave clear is below the freezer's 8-HP wave | **yes**: a capacity threshold, deterministic (§3.1); ends at level 8 (baseline) / 10 (C1) | | exact one-turn checks, every cell |
| natural freezes are mostly the opponent's push arriving, with a worn freezer wave | **yes**: the freeze state is produced by pushes; a guard keeps it only 50–63% of the time from natural states | | §2, §3.5 |
| defending pays only with an HP edge | **yes**: the defence's value is the breaker's HP / deaths / XP while contesting; HP is cheap relative to farm unless already low | | confirmation §3.2; channels §3.3 |
| the policy rarely defends a favourable constructed freeze (+0.4 left) | | **yes**: rare state; the critic ranks the freezer's options at a freeze at chance | §3.2, §6.2 |
| conceding a freeze costs 0.4–0.9 even briefly | **yes**: standing back forfeits all farm and XP (out of reach and XP range) while the walk-up cell farms at full rate | | B, §4.2 |
| the breaker's policy contests correctly against its own freezer; under-commits against a guard | | **partly**: the gap exists only against a strategy its self-play opponent does not play | §4.3 |
| specialists convert large gaps better, general policies play plausible advantages better | the economy rarely creates large gaps (M4.6a/b) | **yes**: the large-gap skill is a training-distribution gap; the plausible-edge restraint is learned | §5 |
| critics blind to strength | | **yes, established**; its causal role in the actor's gaps is a hypothesis | §6 |

**Interpretation against the design philosophy** (§2: "the weaker champion gives up position, farm or wave control when
the expected cost of staying exceeds the cost of conceding, and keeps contesting when it does not"):
- The current game reproduces this relationship. The weaker side contests when healthy because contesting is cheap
  and conceding is expensive. It does not concede when damaged either: conceding a freeze stays expensive, while
  contesting and resetting are close.
- A League player, too, cannot afford to give up a frozen wave for long; but in League the threat of being killed
  under the freezer's minions makes walking up genuinely dangerous. Here a walk-up is dangerous only in the early
  window and only when the breaker is already damaged.
- That is the one intended interaction that is weaker than League's: **the freezer's threat on a walk-up.** It is
  bounded by one trade per turn and by wave clear that outgrows the 8-HP wave. Note that it is limited in *time*
  (capacity outgrows a fixed wave) more than in *strength*.

**Learning.** The established learning-side facts are:
1. the critics assign almost no value to strength advantages at constructed states;
2. the actor under-uses three profitable responses: guarding a favourable freeze, committing against a guard, and
   stepping back with Safe in a Recall window;
3. the specialists still out-convert large gaps.

All three are in states the self-play distribution seldom contains. The 25% archive-start mixture of M4.6c (which
already contained constructed freezes and strength gaps) did not repair the critic's blindness at 8.192M.

## 8. Recommendation

**Recommendation: accept the current abstraction for freezing and concession (no mechanics change), and make the next
step one focused value-function study on the training side.**

**Why no mechanics change.**
- The intended choices exist and are economically coherent:
  - a damaged breaker facing a stronger freezer cannot contest a guarded early freeze for free;
  - a healthy breaker should break freezes;
  - standing back is the worst reply;
  - pushing out of a transient freeze is right.
- The learned policies make these choices correctly in natural states: no confirmed error against their own opponent.
- The one weaker-than-League interaction is the freezer's threat on a walk-up after the early levels. Changing it is
  a design decision, not a defect. It would mean a mechanic that keeps freezes defensible as wave clear grows (for
  example wave strength that scales with time, as League minions do), so it is named here but **not recommended now**:
  - natural freezes are mostly worn-wave transients that a stronger defence would not reach;
  - the measured value of defence even in the best case is about one turn of income;
  - nothing in this study shows that the agents would use a stronger defence well, given the critic's blindness.
- No evidence here supports changing trading, Recall, investment or the reward.

**Why a value-function study next.** The clearest established defect is the critic.
- It is calibrated in natural states but assigns about 0 to a plausible ×1.25 edge and about 5% of a ×1.76 gap.
- It ranks the freezer's options at a freeze at chance.
- It survived M4.6c's archive mixture.

Every actor gap found here (favourable freeze defence, committing against a guard, Safe at a Recall window, large-gap
conversion) sits in strength-dependent states. Before another training change it is worth knowing which of three
explanations holds:
1. **the observation or representation cannot express strength value;**
2. **PPO's bootstrapped value targets and the rarity of these states keep it from being learned;**
3. **the value is learnable, but the actor does not follow it.**

**Proposed next step (for review; not run):** a no-PPO value-representation probe.
- **Labels:** reuse and extend M4.6d's no-bootstrap 300-turn returns (constructed strength / HP / freeze cells and
  natural states, both seats; roughly 3,000 labelled states, held out by cell).
- **Fits:**
  - (a) a linear probe on the frozen 8.192M networks' last hidden layer;
  - (b) a freshly initialized value MLP of the same shape on the same 90-entry observation;
  - (c) the same with explicit strength features (duel ratio, HP ratio) appended, as a diagnostic only.
- **Readouts:** held-out R², and the recovered premium of the edge, plaus and ×1.76 cells against their realized
  +0.6 / +1.3 / +5.7.
- **Decision rule:**
  - (a) succeeds → the representation has it and the value training is the bottleneck. A training-side follow-up
    would then target value learning (value-target horizon, λ, value-loss weighting, the strength mix of starts).
  - Only (b) succeeds → self-play features lack it.
  - Only (c) succeeds → the observation's strength encoding is the bottleneck, a feature-scaling question for review.
- **Cost:** about 1 hour of labelling at the M4.6d rates plus minutes of fitting.

**Not recommended:**
- more investment-curve work, or adopting C1 on this evidence (freezes do not depend on the curve: §3.6, M4.6c);
- a global change to trading or Recall;
- training more specialists.

**Milestone verdict: PASS WITH CAVEATS.** The questions are answered on confirmed estimates. The caveats are §10's
limitations, mainly the small plan library, constructed cells and K 16–48 per cell.

## 9. Explanatory trajectories (exploratory, single games; `results/m46d/traces.json`)

Single games of the C1 policy of seed 0 (both seats) from the level-5 plaus freeze cell (A one level and 3 invested
ahead, B at 60% HP), common seed. Notation: wave–behaviour (LH Last Hit, TN Thin, SL Slow Push, SH Shove; H Hold,
Y Yield, T Trade, S Safe); HP after the turn; held = A's freeze after the turn; A's cumulative lead.

**1. The guard holds, the breaker keeps contesting, then resets** (A guard_sel, B its policy):

| turn | A | B | HP A / B | held | event | lead |
|---|---|---|---|---|---|---|
| 0 | TN-T | SH-Y | 0.85 / 0.60 | yes | A trades the walk-up from inside the held wave; B's Shove does not break it | +0.56 |
| 1–3 | LH-H, LH-T, LH-H | SH-Y, SH-Y, SH-H | 0.59 / 0.60 | yes | B keeps shoving; the held wave's fire wears A down | +0.00 |
| 4–5 | TN-H, SH-S | SH-Y, SH-S | 0.61 / 0.62 | no | the freeze breaks on turn 4 | +0.55 |
| 6–7 | SH-Y, Recall | Recall, away | 1.0 / 1.0 | re-forms | both reset | +1.15 |
| 9–15 | guard again | contests again | 0.41 / 1.0 | yes | the cycle repeats | +1.40 |

**2. The same start, the breaker concedes** (A guard_sel, B `concede`): B stands back at its −1 (LH-S) for nine turns.
The freeze holds all nine, and every one of A's minions that dies is out of B's reach. A's lead climbs +0.45, +0.90 …
+5.74 by turn 9, when B recalls. A then over-extends with Shove and dies on turn 14 (+2.69 at turn 15). Conceding a
freeze is the worst reply, and a conversion error by the winner gives back half of it.

**3. The same start, both policies** (A policy, B policy): A releases at once (SL-H on turn 0) and the freeze breaks.
B, at 60%, steps back and recalls on turn 2 (+1.15). On turn 7 both trade: A falls to 6%, B to 48%. A tries to recall;
B dives into A's tower cell to interrupt it twice, and dies there on turn 11 (+2.74). The policy's way
of playing the favourable start (release, punish the reset) earns about as much as guarding in this game; across
rollouts, guarding leads by +0.4 (§3.2).

**4. Level 8, equal power, trade-every-walk-up guard** (A guard_walk*, B its policy; second seed): the guard trades
the walk-up for three turns (A 0.89 → 0.70, B 0.67 → 0.50) and the freeze holds two of them. B Shoves through on turn
2, recalls on turn 7, and the freeze re-forms while it is away (+1.6 at turn 8). Both then trade at the freeze until A
is at 18%, and the lead returns to +0.8. At equal power the defence's HP cost and its denial roughly cancel, which is
the confirmed −0.3 to +0.0.

## 10. Reproduction, files, tests and limitations

**Commands** (from the repository root; each stage resumable; per-job records under `results/rl/checkpoints/m46d/`):

```bash
P="python -m lanegame.experiments.rl.m46d"
$P records --workers 5               # full-state natural self-play, 12 policies x 48 x 400 turns (1 min)
$P freezes                           # descriptive tables, results/m46d/freezes.json (11 s)
$P prereg                            # freeze docs/m4/m4_6/m4_6d_preregistration.md (SHA-256)
$P matrix --part A,AL --workers 6    # 21 min
$P matrix --part A2,B,N,Ab --workers 6   # 47 min
$P specialists --part verify --workers 3 # 12 min
$P specialists --part decomp --workers 3 # 17 min
$P critic --part calib --workers 3   # 40 min
$P confirm --workers 3               # 13 min (with results/m46d/run_confirm_reverse.py as a second runner)
$P critic --part rank --workers 3    # 14 min
python results/m46d/run_rank_yield.py    # exploratory Yield follow-up (3 min)
$P traces analyse report
python -m pytest tests/test_m46d.py
```

**Runtime.** About 1 h 40 min of wall clock for all simulation on 6–9 workers, roughly 36M simulated turns. The
whole investigation (reading, design, runs, analysis, report) took about 3 hours of the 5–8 hour allowance; it stopped
early because the questions were answered and the remaining uncertainty is about learning, which this study's no-training
boundary cannot resolve.

**New files:**
- `lanegame/experiments/rl/m46d/`: `common`, `records`, `freezes`, `cells`, `plans`, `matrix`, `specialists`,
  `critic`, `analysis`, `analysis_c`, `traces`, `report`, CLI;
- `tests/test_m46d.py` (6 tests);
- `docs/m4/m4_6/m4_6d_preregistration.md` and this report;
- `results/m46d/`: `preregistration.json` (four amendments plus the dated exploratory Yield follow-up),
  `freezes.json`, `analysis.json`, `traces.json`, the generated `report.md`, stage logs, the queue scripts and the two
  helper runners.

**Modified:** `docs/m4/README.md` (index) and `docs/rl_research_roadmap.md` (§AP and the status row). No library code,
rule, preset, default, reward, test pin or checkpoint changed. The mathematical reference is unchanged, because no
rule or objective changed.

**Tests.** `tests/test_m46d.py`:
- the strength conditions give the stated duel ratios;
- every cell is a legal state with the intended layout and HP;
- the episode classifier on a synthetic record;
- plan restriction and release;
- the matrix job's policy pair equals the M4.6a engine and is bit-reproducible;
- the switch / substitution identity S1 ≡ U0.

Run on the final code with the M4.6a / b / c and shared-trainer suites: 42 passed.

**Seeds:** 99,000–99,999 (records 99,000; constructed draws 99,100; natural states 99,150; matrix discovery 99,200;
confirmation 99,300; specialists 99,400; critic 99,500; traces 99,600). **Checkpoint provenance:** every evaluated
policy's SHA-256 prefix is in `results/m46d/freezes.json`. The networks are the M4.6c 8.192M continuations, the M4.5c
finals and the M4.6a specialists, unchanged.

**Limitations.**
- **Plans are a small library.** A plan that failed to pay is not proof that no defence or concession pays. guard_sel
  missed the turn-0 walk-up at full HP (§1.3), and only the exploratory guard_walk tested that case.
- **Constructed cells are best cases or stylized starts.** The natural-state family (N) is the realism check, with 12
  states per kind and policy.
- **K 16–48 per cell** leaves single-cell intervals wide (often ±0.3–0.6). The classes and the confirmation carry the
  conclusions; cell-level discovery values are not established results.
- **The confirmation reuses the same constructed cells and N's states** with new rollout seeds. It controls the
  selection of noisy alternatives, not the choice of cells.
- **The decomposition's same-state part samples 10 of the first 21 turns.** Its sum is noisy; the time and class
  pattern is the robust result.
- **Critic calibration** uses 32 rollouts per state (standard error about 0.5 per state) and is pooled across states
  and policies. The ranking test uses K 16 and is noisy per state.
- **The specialists were trained in the baseline environment** against the M4.5c finals; their C1 numbers are
  cross-environment.
- **No result here speaks to a from-scratch campaign**, to shared-policy interference, or to C1's adoption.
- **The working tree was being staged by another process during this work** (new M4.6d files appeared in the index).
  Nothing was staged, committed or unstaged by this work.
