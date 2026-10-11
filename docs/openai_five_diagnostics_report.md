# Strategic Diagnostics and Robust Self-Play: Report

Branch `research/openai-five-diagnostics` (from `research/openai-five-selfplay`), session of 2026-10-11.
Work log: `docs/openai_five_diagnostics_progress.md`. Generated tables (every number below is taken from them):
`results/rl/diagnostics/{audit, hpvalue, counterfactual, calibration, snapshot_audit, overlay, overlay_late,
attacker_behavior, robustness, long_selfplay}.md`. Coordinates are own-perspective (own tower −2, the lane cell 0,
enemy tower +2); payoffs are resources per 32-turn game (1 resource ≈ one wave of farm), zero-sum.

The game rules were **not changed** in this session: every result is on the unchanged `GameConfig.champions()`
environment that the previous study's policies were trained on.

---

## A. Executive summary

**1. Why do the learned agents overwhelmingly choose Hold and Trade?** Because, in the states the policies
actually reach, Hold and Trade are the only actions with value, and they are worth the same at the margin.

- *The other mechanics are almost never available.* With both waves on Slow Push the lane is a deterministic
  2-cycle at the lane cell, so waves never crash: crashes 0.00–0.01 per game, a siege objective is legal on 0–4
  turns in 400 mirror games, Freeze is legal on 0.02–0.22 turns per game, and Hold's cell is past the lane on
  ≤ 0.9% of turns (a champion stands where its wave is; no action moves it forward except chasing or a siege
  objective). Recall is legal on ~40% of turns, but champions at HP ≤ 2 stand at the lane cell, where it is not.
- *Trade vs Hold is a near-indifference.* In every counterfactual scenario (Phase B), committing to Trade instead
  of Hold changes the return by −0.15 to +0.14 per state on average, every interval including 0 (except a −0.01
  effect while the enemy is away) — what an equilibrium mixture looks like.
  Trade halves the trader's farm (−0.16 per turn) and deals 0.66 HP per turn; a kill is worth +1.3 to +1.6
  to the payoff; the two balance.

**2. Are Recall, wave management, vision and sieges unprofitable, or is PPO missing them?** Both, depending on the
mechanic:

| Mechanic | Verdict | Key evidence |
|---|---|---|
| Recall | **Reasonable behavior** (unprofitable on average) | Recall vs the policy at legal HP ≤ 2 states: −0.02 to −0.47; "recall at HP ≤ 2" overlay −0.14 to −0.47 per game, "at HP ≤ 1" +0.03 ± 0.18 (break-even) |
| Yield, Safe | **Reasonable behavior** | vs Hold when the opponent trades: −0.07 to −0.25; Safe vs Hold in danger −0.02 to −0.28 |
| HP | **Valued** | ≈ 0.22–0.30 resources per HP point; 1 HP is worth about the same as being dead |
| Siege (crash → Apply Pressure), wave timing | **Training problem: profitable but not discovered** | 1M-step attackers exploit every target with Shove→crash→Pressure (+1.6 to +2.6, 1.3–2.6 plates per game); once a siege exists Pressure beats Hold (+0.2 to +0.8); the policies' rare-action probabilities collapse in the first 25–50k transitions and siege states are never visited again |
| Freeze | **Mechanics limitation** (not a bug) | in all 72 reachable Freeze-legal states a one-turn Freeze is *identical* to Slow Push (difference 0.000); a permanent Freeze rule changes nothing (±0.00) |
| Forward positioning | **Mechanics (structural)** | position follows the wave; forward cells require one's wave to be forward, i.e. shoving |
| Skill-conditioned Trade | **Persists, but worth little** | Trade rises with own skill in 4/6 shared runs (0/3 M3b); Trade − Hold ≈ 0 at both skill extremes |

The decisive distinction is between one-step and multi-step deviations. Unconditional rules that use the missing
mechanics all *lose* against the policies (shove+pressure every turn −2.3 to −2.6; punish-shove −0.3 to −1.0;
lane-phase shove −0.1 to −0.5), but *timed* ones win: shoving only in the last four turns beats all 9 policies
(+0.11 to +0.60, 8 of 9 above +0.4), and a PPO attacker trained for 1M steps against a frozen policy finds a *timed*
siege strategy (Shove on ~30% of turns, mostly late; Pressure on every legal turn) worth +2 per game. The siege
strategy is profitable, but it needs a multi-turn plan and state visitation that self-play never provides.

**3. How robust are the policies against stronger adversaries?** Much less than reported before. 1,024k-step
attackers raise the exploitability lower bounds of the nine main-study policies from +0.11–1.15 (307k) to
**+0.71–2.63** (condition means M3b +1.96 ± 1.76, 100/0 +1.75 ± 1.86, 80/20 +1.01 ± 0.64; 95% t over 3 seeds).
11 of 21 attacker curves are still rising at 1,024k steps, two attackers on the same target can differ by 2
resources (+0.62 vs +2.63), and the ranking of targets is unrelated to the earlier budgets (Spearman −0.03 vs 102k,
+0.17 vs 307k). The bounds are not saturated; earlier exploitability rankings were attacker-limited. No condition
is established as less exploitable.

**4. Does longer self-play help, and do historical opponents?** Longer training (1,024k transitions, 5 seeds per
condition) helps against fixed learned opponents (+0.5 to +0.6 against the published M3b policy, vs +0.2 to +0.3 at
307k) and against the main-study policies (+0.50 to +0.61), shows no forgetting or cycling in any run, learns more
Shove (8–24% of turns) and stronger skill-conditioned aggression, but still never sieges. Historical opponents give
no measurable benefit: 80/20 − 100/0 = −0.12 ± 0.45 in cross-play (80/20 wins 6 of 25 seed pairings).

---

## B. Strategic action audit (Phase A; `audit.md`)

Each of the 9 main-study policies (M3b, shared 100/0, shared 80/20; seeds 0–2) and the published `pub_sprand_s1`
plays 200 held-out episodes per seat against itself and 100 per seat against `hold_farmer` and
`aggressive_trader` (equal skill), with its full action distribution recorded on every turn
(`lanegame.diagnostics.audit_games`, which reproduces `head_to_head` payoffs exactly).

**Legality versus selection (mirror games, family means; legal share of present turns / policy probability when legal):**

| family | Hold | Yield | Trade | Safe | Recall | Shove | Freeze | Pressure / Vision |
|---|---|---|---|---|---|---|---|---|
| M3b | 1.00 / 0.38 | 1.00 / 0.018 | 1.00 / 0.60 | 1.00 / 0.002 | 0.43 / 0.001 | 1.00 / 0.012 | 0.00 / 0.46 | 0.00 / – |
| 100/0 | 1.00 / 0.52 | 1.00 / 0.007 | 1.00 / 0.47 | 1.00 / 0.002 | 0.39 / 0.001 | 1.00 / 0.061 | 0.01 / 0.09 | 0.00 / 0.28, 0.07 |
| 80/20 | 1.00 / 0.44 | 1.00 / 0.004 | 1.00 / 0.55 | 1.00 / 0.002 | 0.42 / 0.000 | 1.00 / 0.008 | 0.00 / 0.06 | 0.00 / – |

Low selection is not low availability for Yield, Safe, Shove and Recall (legal on 39–100% of turns), but it is for
Freeze and the siege objectives (legal on < 1% and ≈ 0% of turns).

**Relevant situations (mirror games):**

| situation | share of present turns | policy probability of the relevant action |
|---|---|---|
| Recall legal and HP ≤ 2 | 2.2–4.3% | Recall 0.000 (all families) |
| HP ≤ 2 at or past the lane (Recall illegal) | 16–21% | Safe 0.002–0.004, Yield 0.004–0.020, Trade 0.29–0.58, Hold 0.40–0.70 |
| enemy champion away (punish window) | 16–20% | Shove 0.005–0.085 |
| opponent P(Trade) ≥ 0.5 | 45–69% | Yield 0.004–0.015, Safe 0.002–0.004 |
| Freeze legal | 0.0–0.8% | Freeze 0.06–0.46 (n = 0–89 turns) |
| siege objective legal | 0–4 turns in 400 games | – |
| own wave past the lane | 0.0–0.9% | – |

**States.** Champions start a turn at HP ≤ 2 on 5–6.5 turns per game and die 1.7–2.1 times per game; 91% of deaths
happen from HP ≤ 2 and 27–29% involve gank damage. Recall legality falls with HP (HP 6: 64–75%; HP 1: 9–17%)
because damaged champions are at the lane cell, where the trades happen.

**Economy (per game, focal policy, mirror):** farm collected 6.3–6.8 of 15.3 available. Farm lost while trading
(a connected trade halves the trader's farm) 4.0–5.5, while away (dead or recalled) 2.3–3.0, while farming
(zoning, reach) 0.8–2.2; plates 0.00; kill bounty 0.83–0.97 each way. While a champion is away the opponent farms
0.41 per turn instead of 0.15. The per-game payoff rises by +1.3 to +1.6 per unit of kill difference (the
empirical value of a kill, bounty included). Against `aggressive_trader` the same policies win +2.9 to +3.2.

**Realized outcomes by action (mirror, descriptive):** a Hold turn yields +0.16 to +0.21 immediate reward (farm 0.32
vs 0.09–0.13 for the opponent), a Trade turn ≈ 0 (farm 0.17 vs 0.19–0.23) but deals 0.66 HP; Yield and Safe turns
yield −0.01 to −0.20 and take almost no damage.

### B.1 What HP is worth (`hpvalue.md`)

Paired rollouts (150 per variant) from 30 sampled mirror states per representative policy in which only the focal
champion's HP differs (or it has just died; every variant is reachable, HP being the outcome of chance), both seats
following the policy:

| policy | HP 1 | HP 2 | HP 3 | HP 4 | HP 5 | dead (bounty excluded) |
|---|---|---|---|---|---|---|
| m3b_s0 | −1.35 ± 0.13 | −1.09 ± 0.11 | −0.79 ± 0.09 | −0.52 ± 0.07 | −0.24 ± 0.04 | −1.19 ± 0.15 |
| sp100_s0 | −1.39 ± 0.10 | −1.15 ± 0.09 | −0.87 ± 0.07 | −0.55 ± 0.05 | −0.25 ± 0.04 | −1.22 ± 0.08 |
| sp80_s0 | −1.24 ± 0.14 | −0.93 ± 0.13 | −0.68 ± 0.11 | −0.44 ± 0.08 | −0.22 ± 0.04 | −1.14 ± 0.18 |

(Return to the end of the game relative to full HP; mean over states ± 95%.) HP has a substantial, nearly linear
value (≈ 0.25 per point) even though it changes nothing until death: lost HP brings the next death (≈ 1.2 plus the
0.5 bounty) closer. The value is larger late in the game (HP 1, turns 20–26: −1.49 to −1.70). **The environment
does give agents a reason to value HP**, and the shared policies act on it (they trade less when behind in HP; see
the previous report). What it does not give is a cheap way to restore HP: Recall costs two turns off the lane and
must start from cell −1 or behind.

---

## C. Counterfactual action evaluation (Phase B; `counterfactual.md`)

**Method** (`lanegame.diagnostics.branch`). A `Snapshot` (lane state, champion state, turn, skills) is restored
into the unchanged environment through its own `reset`. For each sampled state, *every legal first action* of the
focal player (plus short plans) is committed in 120 paired rollouts; afterwards both seats follow the same learned
policy, which is therefore both the continuation policy and the fixed opponent. Pairing: rollout m of every
candidate uses the same chance stream and the same sampling streams, each seat draws exactly one uniform per turn
whether or not a plan overrides it, and the opponent's branch-turn action is drawn from its own observation, which
does not contain the focal player's action (tested with a spy agent: identical observation and action for every
candidate). The advantage of action a is Q(a) − Σ_b π(b)Q(b), estimated per rollout (control variate). States are
the unit of replication. 10 scenarios × 3 representative policies × 24 states, plus 5 scenarios × the other 6 main
policies × 10 states (1,026 states, 13.7 min). These are Monte Carlo values of these continuation policies, not
game-theoretic values.

**Required comparisons** (mean over states ± 95%; representative policies m3b_s0 / sp100_s0 / sp80_s0; the other
seeds agree in sign except where noted, see the full tables):

| comparison (scenario) | m3b_s0 | sp100_s0 | sp80_s0 | reading |
|---|---|---|---|---|
| Recall − policy (HP ≤ 2, Recall legal) | −0.23 ± 0.17 | −0.27 ± 0.07 | −0.02 ± 0.20 | Recall not better; other seeds −0.15 to −0.47 |
| Recall − Hold (same) | +0.03 ± 0.12 | −0.19 ± 0.08 | +0.15 ± 0.13 | mixed sign |
| walk back & recall − Hold (HP ≤ 2 at the lane) | −0.10 ± 0.09 | −0.06 ± 0.10 | −0.10 ± 0.16 | other seeds −0.18 to +0.19: ≈ 0 |
| Safe − Hold (same) | −0.23 ± 0.10 | −0.21 ± 0.07 | −0.23 ± 0.10 | retreating loses |
| Shove − Hold (both present, HP ≥ 3) | −0.14 ± 0.12 | +0.15 ± 0.15 | +0.16 ± 0.16 | state-dependent: other seeds −0.33 to +0.18 |
| Shove while the enemy is away − Hold | −0.36 ± 0.48 | +0.10 ± 0.22 | +0.05 ± 0.17 | other seeds −0.45 to +0.23 |
| Yield − Hold (opponent P(Trade) ≥ 0.6) | −0.15 ± 0.08 | −0.12 ± 0.05 | −0.07 ± 0.11 | other seeds −0.16 to −0.24 |
| Safe − Yield (HP ≤ 3, opponent P(Trade) ≥ 0.4) | −0.03 ± 0.09 | −0.09 ± 0.05 | +0.03 ± 0.10 | ≈ 0; both lose to Hold (−0.02 to −0.28) |
| Freeze − Slow Push (Freeze legal; constructed) | 0.000 | 0.000 | 0.000 | identical in all 72 states |
| Freeze − Shove (same) | −0.13 ± 0.02 | −0.12 ± 0.02 | −0.12 ± 0.03 | |
| Pressure − Hold (siege; constructed) | +0.47 (2 states) | +0.22 ± 0.62 (12) | +0.78 (1 state) | Pressure > Vision by +0.18 to +0.71 |
| Trade − Hold, skill 0.2 vs 0.8 | +0.08 ± 0.07 | −0.09 ± 0.08 | −0.04 ± 0.13 | indifference |
| Trade − Hold, skill 0.8 vs 0.2 | +0.04 ± 0.07 | −0.04 ± 0.11 | +0.02 ± 0.08 | indifference |

Constructed scenarios are reachable by construction: from a mirror-game state the opponent (Freeze) or the focal
player (siege) plays Shove + Hold for up to 4 (6) turns and the first qualifying state is taken. Even so, a siege
was reached in only 1–12 of 24 attempts per policy.

**Does the policy miss profitable opportunities?** In 21–75% of states (by scenario and family) at least one legal
action beats the policy's own mixture significantly after a Bonferroni correction for the number of legal actions;
the mean best advantage is +0.08 to +0.33 per state (siege states +1.1 to +2.2). The improving action is mostly a
Shove (Hold), and the policy puts 0.0–0.3% on it. But averaged over all 963 states, Shove − Slow
Push is +0.00 ± 0.04 (M3b, 318 states), +0.01 ± 0.04 (100/0, 328), +0.10 ± 0.04 (80/20, 317): a Shove helps in some states and hurts in
about as many. The policies are locally improvable by *state-conditional* wave timing, not by more Shoves.

**Critic check** (`calibration.md`). At these states the shared-policy value heads agree with the Monte Carlo value
of the policy (bias −0.14 to +0.18, correlation 0.59–0.99) except in the rarely visited Recall-legal low-HP states
(optimistic by +0.38 to +0.39). M3b's critic is biased upward by +0.2 to +1.3 in mirror games (its training
opponents were weaker). The policies do not misjudge the states they visit; they do not visit the states that
would reveal sieges.

---

## D. Mechanics versus training (Phase C)

### D.1 Rule overlays (`overlay.md`, `overlay_late.md`)

Each policy plays itself unchanged except where one interpretable rule fires, head-to-head against the unmodified
policy (1,000 held-out episodes per seat, ± 0.05–0.08 per policy):

| rule (fires when) | M3b | 100/0 | 80/20 |
|---|---|---|---|
| recall_hp2: HP ≤ 2 → Recall, or walk back with Safe | −0.34 ± 0.44 | −0.25 ± 0.32 | −0.29 ± 0.42 |
| recall_hp1: HP ≤ 1 → same | −0.00 ± 0.22 | +0.02 ± 0.18 | +0.03 ± 0.18 |
| punish_shove: enemy away → Shove | −0.58 ± 0.63 | −0.64 ± 0.83 | −0.47 ± 0.11 |
| pressure: objective legal → Hold + Pressure | +0.00 ± 0.01 | +0.01 ± 0.03 | +0.00 ± 0.00 |
| freeze: Freeze legal → Freeze | −0.00 ± 0.01 | −0.01 ± 0.05 | +0.00 ± 0.00 |
| shove_phase: both fronts 3 HP at the lane → Shove | −0.28 ± 0.50 | −0.36 ± 0.26 | −0.36 ± 0.18 |
| shove_pressure: always Shove, Pressure when legal | −2.41 ± 0.52 | −2.58 ± 0.61 | −2.31 ± 0.17 |
| combo: recall_hp2, pressure, punish_shove | −0.24 ± 0.65 | −0.36 ± 1.44 | −0.08 ± 0.24 |
| late_shove24: from turn 24 on, Shove (Pressure when legal) | −0.03 ± 0.12 | −0.00 ± 0.19 | +0.05 ± 0.05 |
| **late_shove28: from turn 28 on, same** | **+0.45 ± 0.16** | **+0.42 ± 0.68** | **+0.54 ± 0.06** |

(95% t over 3 seeds.) Recall at HP ≤ 2 cuts deaths from ~1.8 to 0.3–0.6 per game but costs more in farm and turns
away than it saves; at HP ≤ 1 it breaks even. Every unconditional use of the missing wave mechanics loses.

### D.2 What the strong attackers do (`attacker_behavior.md`)

The four strongest 1,024k attackers (+1.6 to +2.6) are the only ones that siege: 0.8–1.5 crashes and 1.3–2.6 plates
per game, P(Pressure | legal) = 0.99–1.00, Shove on ~25–35% of turns, rising toward the end of the game (P(Shove)
0.03–0.13 before turn 8, 0.37–0.45 from turn 24), and Shove on 96–100% of turns while their own siege lasts. They accept more
deaths than their targets (1.95–2.43 vs 0.61–1.14). The other attackers (+0.4 to +1.5) win without sieges; most of
them also shove more late in the game (up to 0.56 from turn 24) than early. No attacker Recalls, Yields, Safes or Freezes measurably.

### D.3 Late-game shove (`overlay_late.md`)

Prompted by D.2, two rules were added after the first overlay run: Shove (Hold; Pressure when legal) from turn 24
(28) to the end. *Shoving in the last four turns beats the unmodified policy for all 9 policies* (+0.53, +0.42,
+0.41 M3b; +0.60, +0.56, +0.11 100/0; +0.54, +0.56, +0.51 80/20; each ± 0.06–0.07 over 2,000 games), with almost no plates
(≤ 0.03) and +0.0 to +0.11 deaths: it is own farm (+0.40 to +0.54 per game; +0.07 for sp100_s2, relative to the
policy alone). A Shove kills the enemy front faster and the pushed wave's
costs (a crash into the enemy tower, a bounce) fall after the 32-turn horizon. Starting at turn 24 gives ≈ 0: then
those costs arrive within the game. The policies observe the turns remaining, so a best response would learn this
horizon effect; none of the 9 policies did (P(Shove) 0.3–6% overall). This is the cleanest single example of a
profitable strategy that PPO self-play did not discover, and it accounts for part of what the attackers exploit
(they shove most from turn 24 on). It is also a property of the finite horizon (the wave state at turn 32 has no
value), worth keeping in mind when interpreting late-game behavior.

### D.4 Training-time view (`snapshot_audit.md`)

Every rare action is abandoned early: P(Shove) 0.03–0.10 at 25.6k transitions, < 0.01 by 51k; P(Yield) 0.02–0.31 →
< 0.02; policy entropy 2.2 (uniform) → 0.95–1.50 at 25.6k → 0.21–0.38 at 307k. From 51k transitions on, Freeze-legal
and siege states are not visited at all in mirror games. In self-play the opponent never crashes a wave either, so
neither seat ever experiences a siege, and no gradient ever reaches the Pressure and Vision logits.

### D.5 Classification (decision gate)

| issue | class | why |
|---|---|---|
| Recall unused | **Reasonable equilibrium behavior** | unprofitable at HP ≤ 2 (counterfactual and overlay), break-even at HP 1; HP is valued, recall is simply expensive (2 turns away, must start at cell ≤ −1) |
| Yield / Safe unused | **Reasonable equilibrium behavior** | retreating gives up more farm than the HP it saves is worth (≈ 0.25 per HP) |
| Siege objectives and wave timing unused | **Training issue** | a profitable multi-turn plan exists (+1.6 to +2.6 against every family) and PPO finds it when it trains against a fixed target for 1M steps; a one-line rule (Shove in the last 4 turns) beats all 9 policies by +0.11 to +0.60; self-play never visits sieges after the first ~50k transitions |
| Freeze unused | **Mechanics limitation** (design, not a defect) | Freeze becomes legal only when the enemy wave is already in contact with one's own, and then it is identical to Slow Push; it never mattered in 72 reachable states or as a game-long rule |
| No forward positioning | **Mechanics (structural)** | a champion's cell follows its wave; forward cells require one's wave to be forward, which is the wave-timing issue above |
| Hold/Trade mixture | **Reasonable equilibrium behavior** | Trade − Hold ≈ 0 in every scenario and skill matchup |

No simulator bug was found (transitions are replayed exactly from snapshots; reward decomposition holds turn by turn).
**No mechanics change was made.** Freeze's inertness and the "HP only matters through death / Recall only behind the
lane" design are recorded as recommendations (section J), not implemented: they are design choices, they would
change every result, and they need their own measured study.

---

## E. Robustness evaluation (Phase D; `robustness.md`)

**Protocol.** The existing PPO best-response trainer (`train_ppo` against the frozen target only, default
`PPOConfig`, identical for every attacker), 1,024,000 steps; two warm-started attackers (copies of the target,
payoff 0 at step 0; training seeds 910, 911) for each of the 9 main-study policies and one from-scratch attacker
(seed 912) for m3b_s0, sp100_s0, sp80_s0; 21 attackers, 34 min on 6 processes. Checkpoints at 25.6k … 1,024k
evaluated against the target on held-out seeds (SeedSequence(50,000), 200 episodes per seat, 500 at the end), both
seats, common random numbers, so checkpoint differences are paired. Learning rate annealed over the full budget (an
intermediate checkpoint is not a separately trained shorter attacker).

| target | previous warm, 102k | previous warm, 307k | warm, 1,024k (two seeds) | scratch, 1,024k | lower bound |
|---|---|---|---|---|---|
| m3b_s0 | +0.42 | +0.45 | +0.55, +0.63 | +2.04 | **+2.04** |
| m3b_s1 | +0.20 | +0.19 | +2.63, +0.62 | | **+2.63** |
| m3b_s2 | +0.45 | +0.96 | +1.13, +1.22 | | **+1.22** |
| sp100_s0 | +0.35 | +1.15 | +1.69, +0.85 | +2.47 | **+2.47** |
| sp100_s1 | +0.13 | +0.16 | +0.31, +1.80 | | **+1.80** |
| sp100_s2 | +0.56 | +0.81 | +0.96, +0.97 | | **+0.97** |
| sp80_s0 | +0.04 | +0.11 | +0.56, +0.67 | +0.71 | **+0.71** |
| sp80_s1 | +0.11 | +0.88 | +1.20, +1.18 | | **+1.20** |
| sp80_s2 | +0.56 | +0.93 | +1.11, +1.10 | | **+1.11** |
| condition mean (95% t, 3 seeds) | | | | | M3b +1.96 ± 1.76, 100/0 +1.75 ± 1.86, 80/20 +1.01 ± 0.64 |

(± ≈ 0.09 per final estimate.) Readings:

- **The bounds were attacker-limited and still are.** 11 of 21 curves fail the saturation criterion between 614k and
  1,024k steps (paired increment ≥ 0.10 or significantly positive); 10 have flattened, including both curves of only
  2 of 9 targets (sp100_s2, sp80_s2). From-scratch attackers, which never reached parity
  at 102k, end at +0.71 to +2.47.
- **Discovery is a rare event.** Two warm attackers on the same target end at +0.31 vs +1.80 (sp100_s1) and
  +2.63 vs +0.62 (m3b_s1): the large values are the attackers that found the siege plan (D.2), after 200–600k steps. A single attacker per target underestimates exploitability.
- **Rankings do not transfer across budgets** (Spearman −0.03 with 102k, +0.17 with 307k).
- **Methods.** 80/20 has the lowest mean bound and the smallest spread (+0.71 to +1.20), but its interval overlaps
  the others and its three targets happen to be the ones no attacker sieged; with 3 seeds and unsaturated attackers
  this is not an established difference.

The exact legacy calibration of the previous study (K = 16, exact exploitability) remains the only certified
robustness measurement and is not repeated here; it concerns a different game and is not mixed with these numbers.

---

## F. Extended training (Phase E; `long_selfplay.md`)

**Justification and protocol.** The diagnostics found no simulator defect and a training-side gap (strategies that
self-play does not discover), so longer training on the unchanged environment is a direct test of whether more
self-play closes the gap. Pure shared self-play (100/0) and the OpenAI Five-inspired mixture (80/20), 5 seeds each
(0–4), 1,024,000 agent transitions per run, otherwise exactly the main study's protocol (same trainer, network, PPO
settings, skill distribution, snapshot every 25,600 transitions). The learning rate anneals over the longer budget,
so these runs are not continuations of the 307k main runs. Cost: 7.5 min wall clock for all 10 runs (≈ 210–280 s per
run on one thread); environment turns 512,000 (100/0) vs ≈ 567,800 (80/20; 10.8% of its transitions come from
historical games). Evaluation on held-out roots 57,000–57,004; condition ± = 95% t over 5 seeds (t₄ = 2.78).

**Learning curves at equal agent transitions** (snapshots vs fixed opponents, 200 episodes per seat):

| transitions | vs `pub_sprand_s1`: 100/0 | 80/20 | vs `m3b_s1`: 100/0 | 80/20 | vs `hold_farmer`: 100/0 | 80/20 |
|---|---|---|---|---|---|---|
| 102,400 | −0.08 ± 0.10 | −0.01 ± 0.04 | −0.07 ± 0.04 | −0.01 ± 0.06 | +0.81 ± 0.18 | +0.89 ± 0.15 |
| 307,200 | +0.32 ± 0.19 | +0.22 ± 0.22 | +0.30 ± 0.20 | +0.22 ± 0.22 | +1.03 ± 0.40 | +1.30 ± 0.47 |
| 614,400 | +0.53 ± 0.23 | +0.44 ± 0.25 | +0.53 ± 0.23 | +0.43 ± 0.19 | +1.36 ± 0.41 | +1.32 ± 0.34 |
| 819,200 | +0.61 ± 0.28 | +0.45 ± 0.23 | +0.60 ± 0.24 | +0.42 ± 0.19 | +1.45 ± 0.31 | +1.49 ± 0.47 |
| 1,024,000 | +0.57 ± 0.24 | +0.50 ± 0.24 | +0.55 ± 0.25 | +0.49 ± 0.18 | +1.30 ± 0.27 | +1.36 ± 0.31 |

Both conditions keep improving against fixed learned opponents until ~600–800k transitions and then plateau. The
final policy beats its own ≈307k snapshot by +0.42 (100/0; per seed +0.39, +0.53, +0.56, +0.07, +0.54) and +0.34
(80/20; +0.29, +0.43, +0.33, +0.45, +0.19). Cross-play: the long policies beat the 9 main-study (307k) policies by
+0.61 ± 0.22 (100/0) and +0.50 ± 0.17 (80/20).

**Forgetting and cycling** (every fourth snapshot of each run against every earlier one, 150 episodes per seat):
none. 0 of 450 (newer, older) pairs won by the older snapshot by more than 2 SE; monotonicity 0.98–1.00 in all 10
runs; the worst final-vs-late-snapshot payoff is −0.08.

**100/0 versus 80/20.** No benefit of historical opponents. Against the pool of long policies 80/20 − 100/0 =
−0.12 ± 0.45 (per seed −0.34, −0.26, −0.31, +0.52, −0.22); in direct seed-paired games −0.09 ± 0.42; 80/20 wins 6 of
the 25 (80/20, 100/0) seed pairings. Against every fixed reference the two conditions are within their intervals,
with 100/0 nominally ahead. With five seeds the difference is still not established in either direction; it is not
more consistent than at 307k.

**New behavior.** Shove use rose from 0.3–6% (307k policies) to 8–24% of turns in 9 of 10 long runs (Shove while the
enemy is away 0.14–0.22), Yield to 2–10% in three 80/20 runs. Recall stayed at 0.000, crashes at 0.00–0.02 per game
and objective-legal turns at ≈ 0: longer self-play learned more wave pressure but still never reached a siege.

**Skill-conditioned aggression persists and strengthens:** Trade probability at skill 0.2 vs 0.8 / 0.8 vs 0.2 is
0.32–0.66 / 0.56–0.84, rising with own skill in 9 of 10 runs (equal in one).

**Exploitability.** (filled in from `long_exploit` below)

---

## G. Mechanical-skill behavior

- **Persistence.** In mirror games at skills 0.2 vs 0.8 and 0.8 vs 0.2, the Trade probability rises with own skill
  for 4 of 6 shared-self-play policies (slopes +0.09 to +0.37; sp100_s1 and sp80_s0 slightly negative) and for 0 of 3
  M3b policies (−0.04 to −0.18). Skill-conditioned aggression persists in shared self-play, less uniformly than in
  the previous report's `hold_farmer` measurement (5 of 6).
- **Value.** At both skill extremes, committing to Trade instead of Hold changes the return by −0.09 to +0.08 (all
  intervals include 0; Phase B). The conditioning moves the policy along a near-indifference curve: it is
  strategically coherent, but contributes little to the payoff. Skill itself dominates outcomes: the 0.8-skill side
  wins a mirror game by +2.5 to +3.1.

---

## H. Limitations

- **Monte Carlo, not equilibrium values.** Counterfactual values hold the continuation policy fixed (one-step
  deviations against the policy itself); a deviation can be profitable because the opponent responds poorly in
  states it never visits, which is exploitability, not proof that the action is intrinsically good. Multi-step plans
  were tested only in the forms described.
- **Sampling.** 24 (10) states per scenario and policy, 120 rollouts per action (SE ≈ 0.1 per state difference); the
  "best deviation" statistics are maxima over 8–25 actions and are reported with a Bonferroni correction.
- **Constructed scenarios.** Freeze-legal and siege states had to be constructed by scripted Shove prefixes; they are
  reachable, but their distribution is not the policies' own.
- **Attackers.** Lower bounds only; 11/21 curves unsaturated; one or two attackers per target; the LR schedule
  depends on the budget.
- **Seeds.** 3 training seeds for the main-study comparisons (t₂ = 4.30), 5 for Phase E.
- **Model.** HP matters only through death; Recall is legal only at cell ≤ −1; the lane's symmetric 2-cycle makes
  sieges unreachable without deliberate shoving. These are design choices of the current environment.

---

## I. Reproduction

```bash
pip install -r requirements.txt && pip install torch
python -m pytest tests/test_diagnostics.py -q          # the new tests (24, ~12 s)
python -m pytest -q                                    # full suite: 500 passed (9.3 min under load)

# Phase A-C (results/rl/diagnostics/; checkpoints of the previous study must exist:
#   python -m lanegame.experiments.rl.openai_five train  regenerates them bit for bit)
python -m lanegame.experiments.rl.diagnostics audit           # 1.5 min
python -m lanegame.experiments.rl.diagnostics hpvalue         # 2 min
python -m lanegame.experiments.rl.diagnostics counterfactual  # 14 min on 5 processes (resumable per job)
python -m lanegame.experiments.rl.diagnostics calibration
python -m lanegame.experiments.rl.diagnostics snapshot_audit
python -m lanegame.experiments.rl.diagnostics overlay         # 11 min
python -m lanegame.experiments.rl.diagnostics overlay_late
# Phase D (checkpoints under results/rl/checkpoints/diag/br/, git-ignored)
python -m lanegame.experiments.rl.robustness train            # 21 attackers x 1,024,000 steps, 34 min on 6 processes
python -m lanegame.experiments.rl.robustness evaluate
python -m lanegame.experiments.rl.robustness report
python -m lanegame.experiments.rl.attacker_behavior
# Phase E (runs under results/rl/checkpoints/of5/long_*, git-ignored)
python -m lanegame.experiments.rl.long_selfplay train         # 10 runs x 1,024,000 transitions, 7.5 min
python -m lanegame.experiments.rl.long_selfplay evaluate
python -m lanegame.experiments.rl.long_selfplay exploit
python -m lanegame.experiments.rl.long_selfplay report
```

Seeds: diagnostics roots 52,000–57,999 (audit 52,000–52,010, HP value 53,000, counterfactual states 54,000 + 100·scenario
+ policy, rollouts 55,000 + …, overlays 56,000, Phase E 57,000–57,004), Phase D evaluation 50,000, attacker training
seeds 910–912 and 920, Phase E training seeds 0–4. All disjoint from the training streams and the earlier
evaluation roots (tested).

---

## J. Recommended next steps

1. **Make the training distribution reach sieges before changing anything else.** The siege plan is profitable and
   discoverable, but self-play abandons Shove within ~50k transitions and never visits a siege again. Within the
   OpenAI Five recipe, test (a) starting a fraction of training episodes from diverse reachable states (e.g. states
   recorded from the Phase D attackers' games, which contain crashes and sieges) and (b) a slower entropy decay;
   judge both with the counterfactual harness (does Pressure acquire value?) and with the 1M-step attackers, not with
   scripted opponents.
2. **Adopt the stronger exploitability probe as the default.** At least two warm attackers and one from-scratch
   attacker per target, ≥ 1M steps, with the learning curve and the paired last increment reported; treat
   102k/307k-step bounds as unreliable, and report the maximum over attackers as the bound.
3. **Then decide the Freeze and Recall design questions with measurements.** Freeze is inert in every reachable
   state where it is legal, and Recall is break-even at best because HP is restored only by leaving the lane for two
   turns from behind the lane. If these mechanics are meant to matter, revise them as a separately versioned champion
   configuration and re-run Phases A–C on it; do not compare agents across the two rule sets.
