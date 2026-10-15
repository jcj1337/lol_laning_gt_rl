# M4.5c — How long should a strategic counterfactual rollout be?

*2026-10-14. A methodological study before the full M4.5c campaign. **No training, no change to the game, the reward,
the observation or PPO.** Environment `GameConfig.m45_calibrated()`; policy: the M4.5c pilot final (seed 100,
1,024,000 transitions) as continuation and opponent, plus M4.5b scripted opponents. Code:
`lanegame/experiments/rl/m45c/horizon.py` (experiment) and `horizon_report.py` (analysis). Generated tables:
[`results/m45c/horizon/report.md`](../../results/m45c/horizon/report.md) (every number below that is not computed in
the text comes from there); figures `results/m45c/horizon/figures/`; summary `results/m45c/horizon/summary.json.gz`;
raw per-job results `results/m45c/horizon/partial/` (24 MB, gzipped). Tests: `tests/test_m45c_horizon.py` (4).*

---

## 0. Answer

**Yes: adopt 50 turns as the default maximum for local strategic counterfactuals.** Three conditions come with it:

1. **Bootstrapped primary statistic.** The primary statistic is the critic-bootstrapped paired difference
   ΔQ₅₀ = ΔG₅₀ + 0.99⁵⁰ ΔV(s₅₀). The raw ΔG₅₀ and the local ledger at 5, 10, 20 and 50 turns are always reported next
   to it.
2. **Three exceptions** get longer horizons, chosen in advance (§6):
   - sustained plans longer than about 10 turns;
   - interventions on constructed states far from the policy's experience;
   - any family whose validation subset fails.
3. **A 300-turn validation subset in every family** (§6.3).

**Evidence** (7.6 million simulated turns, §1):

| finding | evidence |
|---|---|
| **No significant 50-turn conclusion contradicted a significant 300-turn one** | 0 of 26 state-level conclusions (raw G), 0 of 29 (bootstrapped Q) |
| the best candidate at 50 turns is the best at 300 | 15–16 of 18 family × opponent comparisons, median Kendall τ 0.93–0.95; the mismatches are candidates within noise of each other |
| in natural states, what changes between 50 and 300 turns is noise | its variance across (state, candidate) pairs (0.17) is no larger than rollout noise alone predicts (0.20) |
| pooled bias of ΔQ₅₀ against the 300-turn reference | within ±0.1 resources in every natural family, never significant |
| a 50-turn rollout is **8–9× more informative per unit of compute** than a 300-turn one | the SD of one paired difference grows from 1.1–1.3 (5 turns) to 2.6–2.8 (50) and 3.4 (300), while the effects do not grow |

**Where the hypothesis fails, and needs care:**

- **Very short raw returns are misleading.** At 5 turns, raw ΔG contradicts the 300-turn verdict as often as it
  agrees (14 contradictions, 13 agreements). The cost of an action (thinning instead of farming, a Ward turn, a trade
  under minion fire) arrives before its payoff.
- **Some payoffs arrive after 20 turns:**
  - a 10-turn bully plan against a wary opponent reads −1.42 at 10 turns, +0.46 at 20 and +0.44 at 300;
  - a forced trade on the constructed early-level trade bench reads −0.22 at 20 turns, +0.50 at 50 and +0.68 at 300.

  50 turns catches both; 20 does not.
- **The critic does not always anticipate.** The bootstrap gets the long-run sign right from 5 turns in natural states
  and in the freeze grid. It does not anticipate the bully plan's or the bench trade's delayed payoff.

**Freezing (§3).** Hypothesis confirmed: unsuccessful freezes are visible within 1–5 turns, and successful ones are
priced by 20 turns. Against a breaker the freeze is gone after one turn: the frozen side Shoves (P(held) ≈ 0). Against
an opponent who backs off, the sustained freeze reads +2.08 ± 0.25 resources at 20 turns, against +2.00 ± 0.32 at 300.
No freeze survived to turn 20 in any cell.

**Revised strategic battery (§7).** About 36 M simulated turns and about 1.8 h on 11 workers. The M4.4c-scale design
would need 212 M turns and about 9.7 h at the rate measured here. The new design keeps opponent protocols and
replication, doubles the states per family, and gives a narrower interval per comparison (±0.24 against ±0.32). The
whole saving comes from shorter rollouts; the 300-turn validation adds about a quarter back. **Revised M4.5c total:
≈ 6 h (≈ 4.5–8.5 h)**, against ≈ 10 h in the runtime audit.

---

## 1. Experimental design

**Paired common random numbers (CRN).** Every comparison is a candidate intervention against the policy's own choice,
from the same start state, using the same random numbers:
- The chance stream is `default_rng(seed)` per rollout, ten uniforms per turn whatever the actions.
- Each policy's sampling uniform at turn i is fixed by (seed, seat, i). Two branches that diverge in history therefore
  still share their randomness turn by turn; sequential draws would desynchronize after the first forced action.

**One trajectory per rollout.** Every horizon (1, 2, 3, 5, 10, 15, 20, 30, 40, 50; and 75, 100, 150, 200, 300 for the
long units) is a checkpoint of that one trajectory, so differences across horizons are paired too. Tests check three
things:
- a 50-turn rollout is an exact prefix of the 300-turn one;
- the recorder's G_h equals the M4.5b engine's;
- Q_h = G_h + 0.99^h V(s_h) on the state actually reached.

**Quantities recorded at each checkpoint h:**

| quantity | definition |
|---|---|
| G_h | the focal side's truncated discounted gold lead, Σ_{t<h} 0.99^t r_t |
| Q_h | G_h plus 0.99^h times the pilot critic's value of the reached state. Reported **separately**, never substituted for G_h |
| ledger | cumulative per-side gold, farm lost out of reach (denial), XP, XP lost, damage by source (trade, minions, tower, gank), deaths, Recalls, time away, wards placed and interrupted, freeze turns, plates, connected trades |
| state | HP, level, investment, absence, vision timers, freeze flags, crashed masses, wave position |

**One-turn candidates and sustained plans are separate.**
- A one-turn candidate ("1t:") forces one action at the start state and then hands back to the policy.
- A sustained plan ("plan:") repeats an action or a scripted rule for up to n turns while its condition holds.

| family | states | candidates (the baseline is always "policy") | opponents |
|---|---|---|---|
| A wave | natural: fronts in contact | 1t Last Hit / Thin / Slow Push / Shove (+ Hold); plan Thin ×4, Shove ×4 | policy, modal |
| B freeze | natural: the focal side holds a freeze | 1t keep (Thin + Hold) / guard (Thin + Trade) / break (Shove + Hold); plan keep, plan guard (the M4.5b `fz_guard` rule) while held, ≤ 20 | policy, modal |
| B frozen | natural: the focal side is frozen on | 1t Shove / walk-up farm / step back / Recall; plan Shove ×3 | policy, modal |
| C trade | natural: enemy catchable without a dive, its front ≥ 8 HP heavier | 1t Trade / Hold / Safe | policy, modal |
| D ward | natural: Ward legal, vision down, own front at the lane cell or beyond | 1t Ward / Last Hit + Hold | policy, modal |
| E recall | natural: Recall legal, HP ≤ 50% | 1t Recall; plan stay (Shove ×3) / step back (×3) | policy, modal |
| F freeze grid | constructed (M4.5b `freeze_state`): power ×1.76 or equal × held wave 12 or 20 HP × level 10 or 15 (8 cells) | as B freeze | policy, modal, `wk_shove` (walks up and Shoves), `wk_back` (steps back), `wk_shove_reset` (resets below 60% HP and contests) |
| G trade bench | constructed: level 4 / 11 / 18 × enemy front 8 / 24 HP (6 cells) | as C trade | policy, modal |
| H zoning | constructed, A ×1.76 stronger: neutral / A's push / B's push | 1t Trade / Hold / Shove; plan bully (Trade whenever catchable) ×10 | policy, modal, `g_wary` (steps back when threatened) |
| I ward | constructed Vision situations: neutral / forward / pre-siege / defending | as D ward | policy, modal |

**Sizes** (fixed before the run, in `horizon.py`):
- 40 natural states per family, each from a different self-play game: seeds 926,000+, at least 20 turns in, seat B
  states mirrored.
- The first 12 states of each natural family are the **long units**: 300 turns, protocol "policy". The other natural
  units run 50 turns.
- Every constructed cell runs 300 turns under every protocol.
- K = 24 paired rollouts per (state, protocol, candidate).
- Rollout seeds 9,270,000+, in the M4.5c pilot range; the 93,000–94,999 range is untouched.

**Volume and cost.**
- 549 jobs, 7,610,400 simulated turns, 3.75 worker-hours, 21 minutes on 11 workers.
- 530 turns per worker-second (50-turn jobs) and 583 (300-turn jobs). Both seats are networks, and the critic is
  evaluated at every checkpoint.
- Every forced code was legal in its state (all natural and constructed units), except Recall in the frozen-on
  family (legal in 32 of 80 state × protocol units). Illegal units are excluded and counted.

**Statistics.**
- States are the units; rollouts are nested in states.
- m_u(h) is the state mean of the paired differences, w_u(h) their variance.
- Family results are the mean over states with a t interval.
- σ_w² is the pooled within-state variance of one paired difference.
- σ_b² = var(m_u) − σ_w²/K is the between-state variance of the true effects.
- A state-level conclusion is "better" or "worse" when |m_u| > 2 SE_u, otherwise "unclear".

---

## 2. Horizon sensitivity

### 2.1 Advantages over horizons (natural states, protocol "policy")

Selected rows; the full table is in the generated report. Values are paired differences candidate − policy in
resources, with 95% intervals over states.

| family | candidate | G₅ | G₂₀ | G₅₀ | Q₅ | Q₂₀ | Q₅₀ | G₃₀₀ (long units) |
|---|---|---|---|---|---|---|---|---|
| A wave | 1t Shove | −0.06 ± 0.15 | −0.04 ± 0.20 | −0.06 ± 0.21 | −0.15 ± 0.11 | −0.10 ± 0.17 | −0.09 ± 0.20 | −0.04 ± 0.34 |
| A wave | plan Thin ×4 | −0.62 ± 0.21 | −0.52 ± 0.29 | −0.61 ± 0.27 | −0.49 ± 0.24 | −0.55 ± 0.26 | −0.62 ± 0.26 | −0.67 ± 0.50 |
| A wave | plan Shove ×4 | **+0.21 ± 0.21** | −0.44 ± 0.25 | −0.37 ± 0.27 | −0.40 ± 0.17 | −0.43 ± 0.22 | −0.42 ± 0.26 | −0.39 ± 0.52 |
| B freeze | 1t keep | −0.13 ± 0.13 | −0.04 ± 0.14 | −0.01 ± 0.16 | −0.04 ± 0.10 | −0.03 ± 0.12 | −0.02 ± 0.15 | +0.26 ± 0.34 |
| B frozen | 1t step back | −0.29 ± 0.09 | −0.26 ± 0.15 | −0.35 ± 0.22 | −0.18 ± 0.07 | −0.24 ± 0.12 | −0.33 ± 0.20 | −0.22 ± 0.62 |
| B frozen | plan Shove ×3 | **+0.05 ± 0.14** | −0.37 ± 0.19 | −0.38 ± 0.24 | −0.34 ± 0.14 | −0.38 ± 0.18 | −0.34 ± 0.23 | −0.61 ± 0.46 |
| C trade | 1t Trade | −0.25 ± 0.12 | −0.23 ± 0.19 | −0.20 ± 0.20 | −0.21 ± 0.13 | −0.21 ± 0.17 | −0.18 ± 0.19 | −0.31 ± 0.42 |
| D ward | 1t Ward | −0.41 ± 0.18 | −0.18 ± 0.20 | −0.15 ± 0.23 | −0.28 ± 0.16 | −0.16 ± 0.20 | −0.09 ± 0.22 | −0.12 ± 0.72 |
| E recall | plan stay (Shove ×3) | **−0.07 ± 0.18** | −0.86 ± 0.25 | −0.89 ± 0.25 | −0.89 ± 0.14 | −0.86 ± 0.22 | −0.86 ± 0.24 | −0.71 ± 0.62 |

In natural states, a one-turn deviation from the pilot's choice is worth little (|Δ| ≤ 0.5) and rarely helps. This is
the M4.4c result again: the policy's own choice is close to the best of the alternatives. The multi-turn Shove, Thin and
stay plans move value more (−0.4 to −0.9).

The bootstrapped Q is **flat across horizons**: Q₅, Q₂₀ and Q₅₀ agree within about ±0.2 in every row and match the
long-unit G₃₀₀ within its interval. The raw G at 5 turns has the wrong sign
for exactly the plans whose cost comes late (the bold cells):
- the 3–4-turn Shoves win gold first (+0.21, +0.05) and lose it later (−0.4);
- staying in lane at low HP looks free for 5 turns (−0.07) and then costs a death (−0.86).

Figures: [`advantage_G.png`](../../results/m45c/horizon/figures/advantage_G.png),
[`advantage_Q.png`](../../results/m45c/horizon/figures/advantage_Q.png).

### 2.2 Variance and uncertainty grow with the horizon; effects do not

| h | σ_w (SD of one paired rollout difference), G / Q | σ_b (SD of true state effects), G / Q | precision per unit compute vs G₃₀₀ (K = 8), G / Q |
|---|---|---|---|
| 5 | 1.33 / 1.12 | 0.42 / 0.39 | 243× / 304× |
| 10 | 1.89 / 1.50 | 0.51 / 0.41 | 68× / 108× |
| 20 | 2.21 / 1.88 | 0.46 / 0.42 | 30× / 39× |
| 50 | 2.82 / 2.61 | 0.43 / 0.41 | 8.3× / 9.5× |
| 150 | 3.32 / 3.29 | 0.41 / 0.42 | 2.1× / 2.1× |
| 300 | 3.40 / 3.40 | 0.41 / 0.41 | 1.0× / 1.0× |

Longer rollouts add variance (later trades, ganks, deaths and Recalls that have nothing to do with the decision) but
no signal: σ_b, the spread of the true effects between states, is the same at every horizon. Discounting caps the
growth. A turn-300 reward weighs 0.05, so σ_w flattens after about 150 turns, but by then six times the compute of a
50-turn rollout has bought no new information about the decision.

Confidence intervals are therefore *wider* at long horizons for the same number of rollouts. For example, B freeze 1t
keep on the long units: ±0.25 at 20 turns, ±0.31 at 50, ±0.34 at 300.

---

## 3. Freezing case study (constructed freeze grid)

### 3.1 Persistence: how long a freeze survives

From [`freeze_persistence.png`](../../results/m45c/horizon/figures/freeze_persistence.png). Each value is P(the freezer
still holds the freeze at turn t) for the sustained keep plan, over the 8 cells.

| opponent (frozen side) | t = 1 | t = 5 | t = 10 | t = 20 | reading |
|---|---|---|---|---|---|
| `wk_back` (steps back) | 1.0 | 1.0 | 0.0 (12-HP held wave) / 1.0 (20 HP) | 0.0 | the freeze holds until the held wave is gone, about 5–10 turns, then ends by itself |
| `wk_shove` (breaks) | 0.0 | ≤ 0.3 | 0.0 | 0.0 | **broken on the first turn** |
| `wk_shove_reset` | 0.0 | ≤ 0.3 | 0.0 | 0.0 | broken on the first turn |
| the pilot (policy / modal) | 0.33–0.58 / 0.0 | ≤ 0.1 | ≤ 0.2 | ≤ 0.2 | the pilot breaks most freezes at once |

The guard plan (Thin + Trade on a catchable walk-up) holds longer than keeping passively in the two level-10 cells
where the freezer is ×1.76 stronger: P(held) 0.96–1.0 at t = 2, against 0.29–0.33 for the pilot's own choice, and
0.46–0.67 at t = 5. At level 15, and against an equal-power opponent, it does not hold longer: at equal power it holds
*less* (≤ 0.08 at t = 2). **No freeze survived to turn 20 in any cell** (most ended within 10 turns). The full
consequence of a freeze decision therefore falls well inside 50 turns.

### 3.2 Value over horizons

Pooled over the 8 cells; ± is the 95% interval from the within-cell rollouts.

| opponent | candidate | ΔG₁ | ΔG₅ | ΔG₁₀ | ΔG₂₀ | ΔG₅₀ | ΔG₃₀₀ | ΔQ₅ | ΔQ₁₀ |
|---|---|---|---|---|---|---|---|---|---|
| `wk_back` | **plan keep** | −0.44 ± 0.05 | −0.34 ± 0.06 | +0.24 ± 0.16 | **+2.08 ± 0.25** | +1.81 ± 0.25 | **+2.00 ± 0.32** | **+0.60 ± 0.07** | +1.16 ± 0.13 |
| `wk_back` | 1t keep | −0.44 | −0.07 | +0.09 | +0.25 | +0.25 | +0.24 ± 0.29 | +0.20 | +0.21 |
| `wk_back` | plan guard | −0.48 | −0.35 | +0.29 | +0.15 | +0.17 | +0.20 ± 0.34 | −0.10 | +0.08 |
| `wk_back` | 1t guard | −0.52 | −0.07 | −0.20 | −0.43 | −0.45 | −0.32 ± 0.31 | −0.45 | −0.46 |
| `wk_shove` | plan keep | −0.41 | −0.13 | −0.20 | −0.04 | +0.02 | +0.23 ± 0.27 | +0.09 | −0.00 |
| `wk_shove` | plan guard | −0.40 | −0.17 | +0.03 | +0.14 | +0.15 | +0.34 ± 0.33 | +0.13 | +0.21 |
| the pilot | plan keep | −0.44 | −0.35 | −0.26 | −0.29 | −0.23 | −0.28 ± 0.43 | −0.09 | −0.16 |
| the pilot | 1t break | +0.22 | +0.07 | +0.05 | +0.00 | +0.08 | +0.09 ± 0.25 | +0.02 | −0.00 |

**What the freeze grid shows:**

- **A successful freeze is priced by 20 turns.** Against a side that backs off, the denial accumulates during the 5–10
  held turns, and its value is complete at turn 20 (+2.08) with no later change (+2.00 at 300). The mechanism, from the
  ledger: B loses farm and XP while A's farm is unchanged.
- **The first few turns give the wrong sign.** Thinning instead of farming costs −0.44 on turn 1 and −0.34 by turn 5.
  The raw first significant sign contradicts the 300-turn verdict in all 8 cells against `wk_back`. The bootstrapped
  ΔQ₅ already reads +0.60 and agrees in 7 of 8 cells.
- **An unsuccessful freeze is visible mechanically at once.** Against a breaker the freeze state is gone on turn 1, and
  every later difference is within ±0.4 of zero at every horizon. "The freeze broke on turn 1–2" is the decisive
  observation. No long rollout is needed to see that the freeze achieved nothing; whether the attempt cost anything is
  a 20–50-turn question, answered as about 0.
- **Guarding** (Thin + Trade) costs HP where it fights. In the ×1.76 cells the freezer has 0.74–0.87 HP at turn 5,
  against 0.91–0.99 for the keep plan; at equal power the guard rarely trades and costs little. The cost is visible by
  turn 5–10. Guarding is not worth more than keeping in this grid: +0.20 against +2.00 versus `wk_back`, and about 0
  versus the others.
- **Resets** (`wk_shove_reset`): the frozen side breaks the freeze before any reset is needed, so a "returns healthier"
  phase never arose in these cells. That interaction is not covered here (§9).
- **One turn against a plan.** One turn of keeping is worth +0.24 against `wk_back`; keeping for the whole freeze is
  worth +2.00. The two answer different questions, as the brief anticipated, and must not be pooled.

**Early-verdict test** (sustained keep plan, 40 cell × opponent units):

| estimator | significant at that horizon | of which also significant at 300 | agreements |
|---|---|---|---|
| ΔG₁₀ | 6 | 3 | 3 |
| ΔG₂₀ | 9 | 8 | 8 |
| ΔG₅₀ | 11 | 8 | 8 |
| ΔQ₅ | 24 | 10 | 10 |
| ΔQ₁₀ | 12 | 8 | 8 |
| ΔQ₂₀ | 10 | 8 | 8 |
| ΔQ₅₀ | 11 | 9 | 9 |

**Answer to the brief's hypothesis.** Unsuccessful freezes are identifiable well before 50 turns: mechanically on
turns 1–2, economically by 20. Successful ones are priced correctly by 20 turns, or by 5–10 turns with the bootstrap.

---

## 4. Short-versus-long agreement

**State level** (every long unit × candidate, 568 pairs; "significant" = |mean| > 2 SE over 24 rollouts):

| estimator at h | significant at h and at 300 | agree | contradict | correlation of state means with 300 | slope of 300 on h |
|---|---|---|---|---|---|
| G₅ | 27 | 13 | **14** | 0.21 | 0.35 |
| G₂₀ | 26 | 26 | 0 | 0.68 | 0.84 |
| G₅₀ | 26 | 26 | **0** | 0.82 | 0.92 |
| G₁₅₀ | 49 | 49 | 0 | 0.98 | 0.99 |
| Q₅ | 34 | 34 | 0 | 0.56 | 1.09 |
| Q₂₀ | 30 | 30 | 0 | 0.71 | 0.98 |
| Q₅₀ | 29 | 29 | **0** | 0.85 | 1.04 |

The low counts of significant pairs mean that most single states cannot be resolved with 24 rollouts at *any*
horizon: 504–512 of the 568 are unclear at both 50 and 300. Per-state verdicts need many more rollouts, which argues
again for short rollouts. The slopes are the cleaner population-level evidence:
- Q_h is unbiased at every h from 5 turns (slope 0.98–1.09);
- raw G_h at 20 and 50 turns understates long-run effects by 16% and 8% (slopes 0.84, 0.92).

**Family level** (top candidate and Kendall τ against 300 turns, 18 family × opponent comparisons):

| h | G: top = top at 300 | G: median τ | Q: top = top at 300 | Q: median τ |
|---|---|---|---|---|
| 5 | 6 / 18 | 0.07 | 8 / 18 | 0.50 |
| 20 | 11 / 18 | 0.60 | 12 / 18 | 0.60 |
| 50 | **15 / 18** | **0.93** | **16 / 18** | **0.95** |
| 150 | 17 / 18 | 1.00 | 15 / 18 | 1.00 |

At 50 turns, every disagreement involves candidates whose intervals overlap: for example, Shove against Hold on the
equal-power zoning cells.

**What changes after 50 turns, and why.**
- In natural states, (Q₃₀₀ − G₅₀) has variance 0.17 across 280 (state, candidate) pairs, and rollout noise alone
  predicts 0.20: **no detectable systematic delayed component**.
- The critic's prediction at turn 50 explains 9% of it (R²).
- No state descriptor at turn 50 predicts it (|r| ≤ 0.12): not HP, investment, level, absence, vision, freeze state or
  crashed waves.

The constructed cells show the genuine delayed consequences:
- **The bench trade.** The forced trade's value shifts from −0.22 at 20 turns to +0.50 at 50 and +0.68 at 300, with
  the opponent playing the pilot policy. The cells are early levels (4, 11, 18) far from the pilot's experience: 94% of
  its decisions are at level ≥ 11, and the bench layout is artificial.
- **The bully plan.** It reverses from −1.42 at 10 turns to +0.46 at 20 against `g_wary`. Its kills and XP denial
  accrue after the plan's own HP cost: against the pilot, +7.6 XP lost by B and +0.35 deaths of B by turn 20.

Both are resolved by 50 turns, and neither changes after 50 beyond noise.

---

## 5. Statistical efficiency

**Precision per unit of compute.** For a family-level effect, the variance of the estimate is
(σ_b² + σ_w²/K) / S for S states and K rollouts per state, and its cost is proportional to S · K · h. From the
measured components (§2.2):
- a 50-turn rollout gives **8–9×** the precision per simulated turn of a 300-turn one;
- a 20-turn rollout gives 30–39×.

**More states or more rollouts?** σ_b (0.4) is small next to σ_w (2.6–3.4), so the optimal allocation for a
family-level mean is many states with few rollouts each (K* = 1 for that estimand alone). Three reasons to keep K
moderate (8–12) anyway:
- the within-state variance must be estimated;
- state-conditioned analysis needs per-state means;
- states have a collection cost.

| question | answer |
|---|---|
| Longer rollouts or more samples? | **more samples**: at a fixed budget, 6× more paired rollouts at 50 turns beat 300-turn rollouts at every family, and spreading them over more states beats more rollouts per state |
| Does a shorter horizon bias the answer? | not in natural states at 50 turns (bias ≤ 0.1, slopes 0.92 and 1.04); yes at ≤ 20 turns for plans whose payoff comes late, and yes for raw G at ≤ 10 turns in general |

---

## 6. Recommended evaluation horizons

### 6.1 By what is being measured

| objective (brief §1) | horizon | statistic |
|---|---|---|
| immediate mechanical consequence: did the action do what it should (freeze held, wave moved, trade connected, ward placed, Recall completed) | **1–5 turns** | ledger and state differences at turns 1, 2, 5 (the freeze persistence curve) |
| local economic consequence: gold, XP, HP, denial, deaths, wave control | **10–20 turns** (≤ 50) | ledger differences at 10 and 20, reported with 50 |
| strategic value of the decision: the long-run discounted return | **50 turns** (default maximum) | **ΔQ₅₀** primary, ΔG₅₀ beside it; longer only by the exceptions below |
| general policy strength, history, tournaments, exploitability | full games (W = 460 / 1000) | unchanged: these evaluate policies, not decisions |

### 6.2 Exceptions, decided in advance

1. **Sustained plans.** The horizon is at least the plan's maximum length + 40 turns, capped at 100. The 10-turn bully
   plan needed 20 turns to show its payoff. Plans of up to 10 turns stay at 50.
2. **Constructed states far from the policy's experience** (levels 1–6, artificial layouts): ΔG₅₀ is primary and ΔQ₅₀
   secondary. The critic is not trusted there: it did not anticipate the bench trade's payoff at 20 turns. These cells
   always enter the validation subset.
3. **A family that fails validation** (§6.3) is re-run at 150 turns, for all its states.

### 6.3 Validation, per trained network and family

| step | rule |
|---|---|
| **subset** | a random 15% of the states (at least 8), drawn by a fixed seed before the run, plus every constructed cell, run to 300 turns under protocol "policy". Their first 50 turns are the 50-turn estimate itself: one trajectory, no extra cost for the short horizons |
| **check** | pooled bias ΔQ₅₀ − ΔQ₃₀₀ with its 95% interval, the slope of ΔQ₃₀₀ on ΔQ₅₀ across (state, candidate), and contradictions among significant pairs |
| **pass** | |bias| ≤ max(0.15, 2 SE), slope in [0.8, 1.25], no contradiction |
| **fail** | the family's full design is re-run at 150 turns, and both results are reported. This is the only adaptive extension, and it is decided by the rule, not by the result's direction |
| **uncertainty, not bias, is the problem** | when a pre-registered comparison's interval exceeds ±0.25, more *states* are added (fresh seeds, same design) before more rollouts per state. Never a longer horizon |

### 6.4 Why not 20 turns by default

20 turns gives 3–4× more precision per unit of compute than 50, and it is right in natural states. But it misses the
late-payoff cases of §4 (bully plan, bench trade), and its ranking agreement drops to 11–12 of 18. Use it as a
screening tier for local economics, not for verdicts.

---

## 7. Revised M4.5c strategic evaluation plan

Every methodological strength of M4.4c is kept, sized for 50-turn rollouts. Per trained final (6 finals):

| family | states | candidates (+ policy) | opponent protocols | K | turns per final |
|---|---|---|---|---|---|
| A wave | 48 natural | 6 + 1 | policy, modal | 12 | 0.40 M |
| B freeze | 48 natural | 5 + 1 | policy, modal, breaker, back-off, reset | 12 | 0.86 M |
| B frozen | 48 natural | 5 + 1 | policy, modal, three freezer styles | 12 | 0.86 M |
| C trade | 48 natural | 3 + 1 | policy, modal, wary | 12 | 0.35 M |
| D ward | 48 natural | 2 + 1 | policy, modal, chaser | 12 | 0.26 M |
| E recall | 48 natural | 3 + 1 | policy, modal | 12 | 0.23 M |
| F freeze grid | 8 constructed | 5 + 1 | 5 as above | 24 | 0.29 M |
| G trade grid | 12 constructed (level × wave × power) | 3 + 1 | policy, modal, wary | 24 | 0.17 M |
| H zoning | 6 constructed (3 situations × 2 gaps) | 4 + 1 | policy, modal, wary | 24 | 0.11 M |
| I ward | 4 constructed | 2 + 1 | policy, modal, chaser | 24 | 0.04 M |
| 300-turn validation subset (§6.3) | 15% of states + every constructed cell, protocol policy | | | 12 | ≈ 1.36 M |
| **per final** | | | | | **≈ 4.9 M** |

| item | volume |
|---|---|
| 6 finals | 29.6 M |
| replication on the 6 midpoint networks (B freeze, B frozen, E recall, F; policy and modal), as M4.4c replicated Recall and freeze | 6.2 M |
| mechanistic diagnostics (M4.5b exact one-turn enumeration, wave mechanics) | about 0.1 h |
| **total** | **≈ 36 M simulated turns** |

At the measured 550 turns per worker-second on 11 workers, that is **≈ 1.8 h, range 1.4–2.4 h** (rate 450–650 per
worker-second, design ±20%).

**Comparison with an M4.4c-scale design for one comparison** (one family, one candidate, one protocol, one final):

| | M4.4c-scale | revised |
|---|---|---|
| design | 24 states × 24 rollouts × 300 turns | 48 states × 12 rollouts × 50 turns, ΔQ₅₀ |
| rollout-turns per candidate | 172,800 | 28,800 (6× fewer) |
| 95% interval | ±0.32 | **±0.24** |

| where the volume goes | effect |
|---|---|
| horizon 300 → 50 | 6× fewer turns per rollout: **the only source of saving** |
| states 24 → 48, rollouts 24 → 12 | the same number of rollouts per candidate, spread over twice the states: better precision, more state-conditioned coverage |
| opponent protocols | kept (policy, modal, and style opponents where M4.4c had them) |
| replication | kept on the key families |
| added | the 300-turn validation subset (≈ 28% of the battery) and the new M4.5 families (trade into waves, Ward) |

### 7.1 Updated campaign estimate

| component | central | range | change from the runtime audit |
|---|---|---|---|
| PPO training (6 × 8,192,000, + seed-0 re-run) | 0.8 h | 0.7–0.9 h | unchanged |
| standard evaluation (curves, headline, history, tournament, cross-play, behaviour, traces) | 1.5 h | 1.3–1.9 h | unchanged |
| strategic evaluation (§7) | **1.8 h** | 1.4–2.4 h | was 5.4 h (M4.4c-scale, assumed 1,000 turns/s) |
| best-response attackers (6, protocol budget) | 1.0 h | 0.3–1.3 h | unchanged |
| regression, smoke, reports | 0.3 h | 0.2–0.4 h | unchanged |
| margin (≈ 15%) | 0.8 h | 0.4–1.5 h | |
| **total** | **≈ 6.2 h** | **≈ 4.5–8.5 h** | was ≈ 10 h |

**Correction to the runtime audit.** The audit priced an M4.4c-scale battery at about 1,000 turns per worker-second,
from M4.4c's engine. This study measured **550** for the M4.5c learned-policy engine: both seats are networks with
per-decision observations, and the critic is evaluated at checkpoints. An M4.4c-scale battery would therefore take
about 9.7 h, not 5.4. The revised design avoids the issue rather than depending on it. Batching inference across
rollouts would speed both designs up; that is engineering work, not needed for the estimate.

---

## 8. Framework changes made (reusable in the full campaign)

`m45c/horizon.py`:
- **`CRNNet`**: per-turn common random numbers for network policies;
- **`rollout`**: a multi-checkpoint recorder giving G_h, Q_h, ledger and state at any set of horizons from one
  trajectory;
- **one-turn and plan candidates** (`build_candidate`), labelled separately;
- **resumable** gzipped per-job results: an interrupted run continues where it stopped, the M4.4c lesson.

`horizon_report.py`: paired aggregation, variance components, agreement and ranking statistics, and the validation
statistics of §6.3. The strategic battery of §7 should be built on these, replacing the single-horizon, single-protocol
`strategic.py` families. Tests: `tests/test_m45c_horizon.py`.

---

## 9. Limitations

| # | limitation | consequence and handling |
|---|---|---|
| 1 | One pilot policy, still learning (1.02M transitions) | The trained finals' critics, and therefore Q₅₀, may be better or worse calibrated. The per-final validation subset (§6.3) is what makes the default safe; nothing here transfers automatically |
| 2 | The 300-turn reference is itself noisy | 24 rollouts give SE ≈ 0.7 per state, so "0 contradictions" partly reflects low power (most single states are unclear at 300). The population-level slopes and bias estimates (§4) are the stronger evidence |
| 3 | Early-game states are rare in natural play (levels 1–10 ≈ 6–11% of decisions) | Compounding advantages (an early investment or level lead) may need longer than 50 turns to pay; the bench trade at level 4 hinted at a delayed payoff. Exception 2 and the validation subset cover constructed early states; the full campaign should add natural early-game states (games stopped before level 10) to the validation subset |
| 4 | Scripted opponents | The critic predicts the pilot's self-play continuation, not play against a scripted opponent. The freeze grid still agreed, but against scripted protocols ΔG₅₀ should be read first |
| 5 | Resets returning healthier, freeze bounces and very long freezes did not occur in the freeze grid | A freeze held beyond about 10 turns, or a reset-and-return sequence, would need a targeted constructed state and, by exception 1, a horizon of the sequence length + 40 |
| 6 | Long-run objectives on purpose | Questions about cumulative effects over a whole laning phase (total gold over 300 turns under a freeze-heavy style) are not local decisions. They belong to the full-game evaluations (curves, headline, tournaments) at W = 460 / 1000, which this recommendation does not shorten |

---

## 10. Recommendation

**Adopt a 50-turn maximum default for M4.5c local strategic counterfactuals**, with:

- ΔQ₅₀ (critic-bootstrapped) as the primary statistic, ΔG₅₀ always beside it, and ledger diagnostics at 1–5 and
  10–20 turns for the mechanical and local economic questions;
- the three exceptions of §6.2;
- the per-final, per-family 300-turn validation subset with its pass rule (§6.3);
- more states, not longer rollouts, when uncertainty is the problem;
- no raw-return verdicts from horizons of 10 turns or less;
- one-turn choices and sustained plans evaluated and reported separately;
- full-game evaluations of strength, history, tournaments and exploitability left unchanged.

**Approval needed for:**
- the revised strategic plan (§7);
- the updated estimate (≈ 6 h in all, ≈ 1.8 h of it strategic);
- building the campaign's counterfactual battery on the `horizon.py` framework.

Nothing else was run: no PPO training, no strategic battery, no attackers.

---

## 11. Reproduction

```bash
python -m lanegame.experiments.rl.m45c horizon           # 549 resumable jobs, 7.6 M turns, 21 min on 11 workers
python -m lanegame.experiments.rl.m45c horizon_report    # summary.json.gz, report.md, figures
python -m pytest tests/test_m45c_horizon.py
```
