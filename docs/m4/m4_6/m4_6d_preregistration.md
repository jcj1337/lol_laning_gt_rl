# M4.6d — Pre-registration: favourable freezes, the weaker side's decision, and the specialist gap

*2026-10-16. Frozen (SHA-256 in `results/m46d/preregistration.json`) after the descriptive stage (`records`,
`freezes`: accounting only) and before any counterfactual rollout. Diagnostic only: no PPO training, no rule, reward,
parameter or default changes, C1 not adopted. Package `lanegame/experiments/rl/m46d/`; seeds 99,000–99,999.*

## 1. Environments, policies, objective

- **Principal environment:** C1 = `m45_calibrated(invest_gain=3.24, invest_scale=64.0)` (provisional, not adopted).
  **Reference:** the baseline `m45_calibrated()`.
- **General policies:** the six 8.192M M4.6c continuations of each arm (`ext/C1:s`, `ext/baseline:s`, s = 0…5), each
  playing both seats (the self-play pairing its critic predicts). Checkpoint digests are recorded in every summary.
- **Specialists:** the M4.6a gap attackers `gap_s0/2/4` (and freeze attackers `freeze_s0/2/4` where stated), trained in
  the baseline environment against the M4.5c finals 0/2/4. Use in C1 is labelled CROSS-ENVIRONMENT.
- **Objective (verified in M4.6c §1):** the zero-sum discounted earned-gold lead, γ = 0.99, gold = farm + plates +
  bounty, XP unrewarded, no terminal state; values in unscaled resources (critic outputs divided by the 0.1 reward
  scale, checked in each run's `run.json`).
- **Horizons:** G50 is the primary readout; G10/G20 for mechanism; G150 for the long subset; the critic diagnostic uses
  long returns (§5).

## 2. What the descriptive stage showed (motivates the design; not a test)

From 48 × 400-turn self-play games per general policy (`results/m46d/freezes.json`):
- About 90% of freeze-state episodes begin right after the frozen-on side's own Shove / Slow Push ("pushed-in"); about
  40% of freeze-state turns are single-turn episodes; about 10% of episodes last ≥ 4 turns.
- The policy, as freezer, Shoves out of its own freeze state 50–75% of the time; as breaker at ≥ 70% HP it Shoves
  about 50–65% and at < 45% HP it mostly steps back or recalls.
- "Favourable" episodes (freezer present, HP ≥ 60%, duel ≥ ×1.10, held wave ≥ 9) are about 2.6% of episodes.
- Early natural strength differences: at levels 6–9 the duel ratio is ≥ ×1.25 on about 20% of turns and ≥ ×1.5 on
  1–5%; the HP-adjusted ratio is ≥ 2 on about 25% of turns. So the plausible early edge is about ×1.2–1.3 in duel
  terms plus an HP gap of 0.4; ×1.76 investment (duel ×3.1) is an extreme diagnostic case.

## 3. Workstreams A and B: the plan × plan matrix

**Plans** (`plans.py`): each restricts the policy's own distribution to a component slice chosen by a start-of-turn
rule and samples it with the policy's own uniform; outside its active turns (condition false, HP at or below the
release level, absent, or after n turns) the policy plays, so Recall and disengagement stay available.

| side | plan | rule while active | n, condition, release |
|---|---|---|---|
| freezer | policy | the policy | |
| | guard_sel | wave Last Hit / Thin; Trade iff the breaker is present, walked up (at our −1 or deeper), catchable without a dive, our HP ≥ 50% and our duel edge ≥ 0; else Hold | 20, while we hold a freeze, 40% |
| | passive | wave Last Hit / Thin; Hold | 20, while held, 40% |
| | push | wave Shove (behaviour the policy's) | 2 turns |
| breaker | policy | the policy | |
| | commit | Shove + Hold | 10, while frozen on, 25% |
| | contest_sel | Shove + Hold iff our HP ≥ 60% and the freezer is away, below 50% HP, or its duel edge on us ≤ 0.05; else Last Hit + Safe | 10, while frozen on, 30% |
| | concede | Last Hit + Safe until the freezer is away or below 50% HP | 10, while frozen on, 30% |
| | reset | walk back and Recall when legal, then the policy | 10 |

In the neutral / pushing-wave controls of B the breaker plans drop the condition (commit 3 turns, the others 6).

**Cells** (`cells.py`; A = the stronger / freezing side; natural investment path S(L)):
- **A (`grid_a`):** freeze (A holds B's 12-HP wave at A's −1, A's 8-HP wave in its tower cell, B walked up), levels
  3 / 5 / 8 × {equal, inv (+3 invested), lvl (+1 level), hp (equal power, B at 60% HP), edge (+1 level and +3
  invested: duel ×1.21–1.27), plaus (edge with B at 60%), brk (B has the edge: control), x1.76 (EXTREME diagnostic)},
  plus held 6 at level 5 for equal and plaus. Full 4 × 5 matrix, K 32, H 50, C1, six policies.
- **AL:** level 5 equal / plaus / x1.76, freezer 4 plans × breaker {policy, commit, concede}, K 32, H 150.
- **Ab:** the A cells equal / edge / plaus / x1.76 at levels 3, 5, 8 in the baseline environment (reference).
- **B (`grid_b`):** wave {freeze 12, neutral, A pushing} × level {4, 8} × {equal, edge, x1.76} × B's HP {1.0, 0.7,
  0.45}; freezer {policy, guard_sel} × the five breaker options; K 24, H 50, C1.
- **N:** natural early (level ≤ 9) freeze states of each C1 policy's own self-play, 12 favourable + 12 other per
  policy (NATSTATE_ROOT seeds), full 4 × 5 matrix, K 16, H 50.

**Pairing and units.** Every combo of a job shares the rollout seeds (chance stream and both seats' sampling uniforms).
The unit of inference is the policy (six per arm): a contrast is averaged over rollouts and cells within a policy, and
the 95% t interval is over the six policies (5 df). Cell-level tables show the same over policies.

**Principal comparisons (A).** Per cell and per class (level; strength condition), against the breaker's policy:
1. guard_sel − passive, guard_sel − push, guard_sel − policy, passive − push (G50; G20 mechanism);
2. three readouts kept separate: (i) *keeping the wave*: A's freeze-held turns in the first 20; (ii) *denial*: B's
   farm collected, B's uncollected farm while present and while away, B's XP; (iii) *net*: G50, with A's own costs
   (HP lost, deaths, A's farm);
3. robustness: each freezer plan's worst case over the five breaker options, and the restricted 4 × 5 game's value
   and equilibrium (solved per cell on the policy-mean payoffs).

*Reading rule.* "Defending pays" in a class when guard_sel − max(passive, push) > 0 with the interval over policies
excluding 0, and guard_sel − policy ≥ 0. "Holding has value even though breaking is the best reply" when guard_sel or
passive carries weight in the restricted equilibrium or has the best maximin. A longer freeze is not counted as a
benefit unless (iii) improves.

**Principal comparisons (B).** Per B cell and freezer plan: each breaker option minus the breaker's policy, in B's
units (−ΔG50). Regions: the best option b*; *clear* if b* beats every other option with the interval excluding 0, else
the set of options within the interval of b* (marked *uncertain*). "The policy is materially inferior" requires
best − policy ≥ 0.25 resources (G50; ≈ 0.6 turns of income) with the interval excluding 0 in discovery **and** a
confirmation (below) with the interval excluding 0 in the same direction. Channels as in A (B's damage, deaths,
Recalls, turns away; farm and XP lost permanently vs delayed — avail vs collected at 50, and at 150 in AL; the
freezer's own costs).

**Confirmation (stage `confirm`).** Independent rollout seeds (CONF_ROOT), K 48, for (a) every B or N cell where some
option beats the policy by ≥ 0.15 with the interval excluding 0 in discovery (the selected option, the policy and
the next-best option, against the same freezer plan), and (b) regardless of discovery, the A contrasts of 1 in the
cells L3 / L5 edge and plaus. Discovery gains are reported as discovery estimates only; established gains are the
confirmation estimates.

## 4. Workstream C: the specialist gap

- **C1 gap verification:** strong side = the general policy itself or a specialist `gap_t` (t = 0, 2, 4), weak side =
  the general policy, on (i) the 27 M4.6a gap cells (×1.5 / ×1.76 / ×2.2, levels 6 / 10 / 14, three situations;
  baseline-curve gold), (ii) plausible cells (neutral and A-pushing at levels 4 and 8 with the edge condition and the
  weak side at 60% HP, plus the A freeze plaus cells), K 16, G50, in the baseline environment (the specialists' own)
  and in C1 (cross-environment). Advantage = specialist G50 − the general's own strong-side G50, interval over the six
  general policies.
- **C2 decomposition** (baseline environment; pairs (gap_t, ext/baseline:t), t = 0, 2, 4; the ×1.76 and ×1.5 gap
  cells, and the plausible neutral/push cells): paired candidates on common streams,
  - *spec-first switch* S_k: the specialist for turns < k, then the general (k ∈ {1, 2, 3, 4, 6, 8, 10, 14, 20, 30}
    and ∞); S_{k+1} − S_k = γ^k E_{d_spec}[A^gen(s_k, π_spec)] (the performance-difference decomposition under the
    specialist's state distribution);
  - *one-turn substitution* U_k: the general throughout except the specialist's choice at turn k (k ∈ {0, 1, 2, 3, 4,
    6, 8, 10, 14, 20}); U_k − gen = γ^k E_{d_gen}[A^gen(s_k, π_spec)] (the specialist's choices at the states the
    general itself reaches).
  Per rollout, the increments are classified by the START-OF-TURN state at k (no hindsight): Recall window (the weak
  side may Recall, HP ≤ 60%, catchable), kill window (weak HP ≤ 35%, catchable), trade window (catchable otherwise),
  out of reach, weak side away; and by the substituted action (Trade / Hold / Safe / Recall; Shove / other wave).
  The two designs separate "same-state choices" (U) from "states reached" (S − U, descriptive).
- *Reading rule:* a decision class "explains" part of the gap when its summed increments are positive with the interval
  over the three pairs × cells excluding 0; descriptive trajectories are labelled as such.

## 5. The critic diagnostic

- States: the A freeze cells (both seats), the B neutral/push cells at full HP, the M4.6a ×1.5 / ×1.76 gap cells
  (both seats), and natural states of each policy's self-play (lead ≥ 4 and even).
- For each general policy in its own environment (C1 principal, baseline reference): V(s) (critic, both seats) against
  realized self-play returns G_H without bootstrap, K 32, H 300, and H 500 for a subset (the tail bound: the measured
  change from 300 to 500, plus γ^500 times the largest |V| seen). Bias = V − E[G], by state class; zero-sum consistency
  V_A + V_B.
- Ranking: at selected decision states (freeze cells; gap Recall windows), each of a few one-turn actions then the
  policy (H 150, no bootstrap) against the critic's one-step look-ahead E[r + γ V(s')], and the actor's
  probabilities: sign agreement and rank correlation of the advantages.
- A biased critic is reported as a fact about calibration only; claims about its effect on learning are hypotheses.

## 6. Exploratory (labelled) and what is not done

Exploratory: trajectories, per-cell drill-downs, any extra cell or horizon added after reading results (dated in
`preregistration.json`). Not done: training, the C1 confirmation campaign, mechanics changes, new algorithms. Any
isolated diagnostic ablation, if needed, is labelled, uses an opt-in configuration field, and is not adopted.
