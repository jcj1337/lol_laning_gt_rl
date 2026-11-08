# M4.6c — Pre-registration of the C1-versus-baseline training pilot

*Written 2026-10-15 after the smoke test (pipeline checks only: no comparison of arms was computed) and before any
pilot training run. Frozen by stage `prereg` (SHA-256 in `results/m46c/preregistration.json`); later changes are dated
amendments in that file. `GameConfig.m45_calibrated()` is unchanged. C1 is the opt-in configuration
`m45_calibrated(invest_gain=3.24, invest_scale=64.0)` (M4.6b `curves.ARMS["C1"]`), experimental, not a default.*

## 1. Question and objective

**Question.** After shared-policy PPO has adapted to each investment curve under the same training procedure, does C1
make purchased and earned leads more consequential than the baseline does, without unacceptable side effects on the
weaker side's options, early-game decisiveness, wave management and the equal lane?

**The training objective, verified in code** (`champions_m4.resolve_turn_m4`, `env/core.py`, `rl/ppo.py`,
`rl/shared.py`, `config.py`), identical in both arms and not changed:
- Per turn, seat A receives r = (E_A − E_B), where E is the gold a champion earns that turn: farm + plate gold + kill
  bounty (2.0 in this preset), in exact ledger units divided by 10⁷. Seat B receives −r. The game is zero-sum.
- `xp_reward_weight = 0`: XP, levels, HP, investment, position and waves are not rewarded except through the gold they
  later produce. There is no shaping term of any kind.
- The objective is J = E[Σ_k 0.99^k r_k]: continuing time, γ = 0.99 is part of the game, there is no terminal state
  (`terminal_rule = "none"`).
- Training: rewards × 0.1 (scale only), GAE λ 0.95, games cut at 512 turns and bootstrapped with the critic (a
  truncation, never a termination); the critic is trained on the same discounted return.
- The documentation agrees: the mathematical reference §8.6 (reward), §8.7 (`xp_reward_weight` is an EXPERIMENTAL
  opt-in, off) and §9.1–9.2 (objective, training). The earlier "gold and XP" proposal (M4.3c) was never adopted. No
  conflict blocks the pilot.

## 2. Arms and training (fixed)

| item | both arms |
|---|---|
| environment | baseline: `m45_calibrated()`; C1: `m45_calibrated(invest_gain=3.24, invest_scale=64.0)`. Nothing else differs |
| starting weights | the six M4.5c 16.384M finals (`results/rl/checkpoints/m45c/ext/s<s>/final.pt`), seed s continued in both arms (6 matched pairs) |
| optimizer | a fresh Adam (checkpoints store no optimizer state); lr 3·10⁻⁴ annealed linearly to 0 over the run, as every earlier warm-started run (M4.5c and M4.6a attackers) |
| budget | 4,096,000 agent transitions per run (1,000 updates; 25% of the finals' own training); midpoint snapshot 2,048,000 |
| PPO | M4.5c settings unchanged: 100% current-policy shared self-play (both seats trained), entropy 0.03, clip 0.2, 4 epochs, minibatch 256, value clip 0.5, gradient norm 0.5, 16 environments × segments of 128 turns, 2 × 128 tanh MLP, snapshots every 25,600 |
| PPO seed | 98,000 + s, the same in both arms |
| starts | games start from the shared archive with probability 0.532 and last 150 turns; otherwise from s0 for 512 turns. About 25% of agent transitions are in archive games. Each archive draw is mirrored with probability ½ |

**Why continuation.** A fresh run would need about 16M transitions per arm and seed to reach the finals' level. A
continuation estimates the curve difference after 4.1M transitions of adaptation from a strong common starting
policy. The C1 arm starts from baseline-trained behaviour, which the matched baseline arm controls for: both arms get
the same extra training, starts and re-warmed learning rate. The pilot cannot say what a from-scratch C1 campaign
would converge to, and it does not identify the effect of diverse starts (no 0% control).

**The archive** (one file for both arms; `archive.py`; 4,800 states). Every state is defined by gold, levels, HP,
positions and waves, never by a combat ratio:
- natural (1,800): decision states of the six finals' s0 self-play in the baseline environment, turns 5–299;
- natural + purchased lead (1,500): the same kind of states with one side's invested gold raised by 2–32 resources
  (log-uniform; median 7.6);
- constructed zoning (900): the M4.5b situations neutral / A's push / B's push at levels 3–18, investment along the
  audit's natural path ×0.7–1.3, a lead of 0 (¼) or 2–32, HP full (½) or 0.35–1;
- constructed freezes (600): the M4.5b freeze builder at levels 3–15, held 5–20, the breaker walked up or at its lane
  cell, a 2–24 lead for the freezer, the breaker or neither.

Because the network plays both seats, every archive game trains the stronger and the weaker role at once.

**Stop conditions** (M4.5c, unchanged): a non-finite statistic, mean approximate KL > 0.05 over 10 updates, a gradient
norm > 100, entropy < 0.3, an illegal action. Any one stops the analysis of that run for investigation.

**Extension rule** (decided only by the data below, never by preference). A matched continuation to 8,192,000 (fresh
runs from the same finals, every setting the same, the learning rate annealed over 8,192,000) runs for both arms only
if both hold:
1. the primary decision (§5) at 4,096,000 is not "met";
2. the primary statistic moved by at least 0.05 resources between the 2,048,000 snapshots and the finals, in either
   arm (pooled over seeds): adaptation is still under way.

Otherwise the pilot stops at 4,096,000.

## 3. Policies, environments and held-out states

For each seed s: **start:s** (the M4.5c final), **baseline:s** and **C1:s** (the arm finals). Each policy is evaluated
in the environment it was trained on unless a table says CROSS-ENVIRONMENT (start:s and baseline:s under C1 are out of
distribution, as in M4.6b).

**Held-out evaluation states:**
- the M4.6b stage cells (early level 8 / S 9, mid 16 / 42, late 18 / 108; neutral, A's push, B's push; leads 0 and +8
  / +16 on either side; full HP, phase 2);
- the M4.6b S4 freeze cells (levels 4 / 10 / 15, held 5 / 12, investment equal / freezer +8 / breaker +8);
- natural states drawn from each evaluated policy's own self-play on new seeds.

None of these is in the archive (checked by state key: 0 collisions). The constructed cells belong to the archive's
families, so they test in-family generalization, not a new family.

**Seeds:** 98,000–98,999 (archive 98,100 / 98,800, evaluation states 98,200, rollouts 98,300, natural games 98,400,
head-to-head 98,500, curves 98,600, traces 98,700).

## 4. Evaluation families

Statistic: G_h, the focal seat's discounted gold lead over h turns, the objective truncated at h. The default horizon
is **50**. Paired rollouts share the chance stream and the per-seat sampling uniforms (M4.6a engine). Ledger channels
are accounting identities, not causal attributions.

| family | content | size |
|---|---|---|
| **premium** (primary) | lead cells × both orientations, policy against itself. J1: arm policy, own environment (critic recorded). J2: start:s in seat B with seat A = start:s or the arm policy, own environment (B0-relative conversion and defence). J3: baseline:s under C1 (CROSS-ENVIRONMENT) | 45 cells × 6 seeds; K 128 (J3: 64); h 50 |
| premium150 | continuation check: early / mid / late neutral (equal, ±16) and mid neutral ±8, arm policy, own environment | K 64, h 150 |
| premium_mid | the extension statistic: mid cells with the 2,048,000 snapshots | K 128, h 50 |
| battery | trailer focal at +8 / +16: policy, Hold only, Safe only, Trade only, reset (10-turn behaviour-only references, the wave left to the policy); leader focal: policy, threaten, Hold only | 18 + 18 cells; K 32; h 50 |
| natural_cf | natural lead states from the arm policy's own self-play (both present, invested lead ≥ 4, turn ≥ 20). Trailer focal: policy and one-turn Hold / Safe / Trade (behaviour only) and Recall where legal. Leader focal: policy and one-turn Trade / Hold / Safe. Records the policy's own probabilities | 24 + 24 states per seed; K 32; h 50 |
| freeze | S4 cells: freezer (policy, keep, guard), breaker (policy, break, walk up, step back, reset); natural freeze states from own self-play with one-turn alternatives | 36 cells + 16 + 16 states per seed; K 32; h 50 |
| specialists | the M4.6a gap specialists (targets s0, s2, s4) in seat A from M4.6a's 27 gap cells against π in seat B, versus π's own play. Baseline environment for start:s and baseline:s. Under C1 (specialists CROSS-ENVIRONMENT) for start:s and C1:s | K 16; h 50 |
| natural | s0 self-play, 48 games × 600 turns: start:s in both environments, baseline:s and C1:s in their own. Leads, ratios, persistence, lane metrics, freezes, Recall, deaths; behaviour records (8 games) for state-conditioned action tables | |
| h2h | learning curves (snapshots 0.5 / 1 / 2 / 3 / 4.1M against start:s, W 460, 32 seeds); final against start:s (W 1000, 64 seeds); against the other starts and within-arm round robin (W 460, 16 seeds); the nine M4.5c fixed references (W 460, 16 seeds); C1:s against baseline:s in both environments (W 460, 32 seeds) | |
| critic | from J1: V(s) at the cell against the realized G50 + 0.99⁵⁰ V(s₅₀), by lead size and side | |

**The premium** of a lead d at a stage, for one seed: averaged over the three situations,

  P = ½ [(G(A+d) − G(equal)) − (G(B+d) − G(equal))],

where G is seat A's mean G_h. Both orientations are included, so a seat effect and the equal cell cancel. In a
zero-sum game this is also the trailer's loss: a larger premium is necessarily a larger disadvantage for the trailer,
and that alone is not evidence of excessive snowballing.

**Conversion and defence learned** (J2): Δ(cell) = G(arm policy in A vs start:s in B) − G(start:s vs start:s).
- Conversion gain at +d: mean over situations of Δ(A+d) − Δ(equal).
- Defence gain at +d: Δ(B+d) − Δ(equal), where A trails.

Both are lead-specific: the equal cell removes general improvement.

## 5. Principal comparisons, uncertainty and decision criteria

**Uncertainty.** The unit is the training seed. Each per-seed value is a mean over K paired rollouts, with its rollout
interval reported. Arms are compared by the paired difference per seed (the same starting final), with the mean over
the 6 seeds, a t interval with 5 degrees of freedom (95%), and the count of seeds by sign. A secondary family claims
significance only after Holm correction within the family. A nonsignificant difference is not equivalence.

**Why the M4.6b criteria are revised.** D1 was underpowered and measured untrained policies. D2's early-premium test
could not separate "decisive early" from "stays meaningful into the mid-game". D4's fixed ±2 HP capacity bound judged
a trade-off by a number rather than by play. P6 flagged the trailer's larger loss, which is the mirror image of a
larger premium in a zero-sum game. The criteria below replace them.

| id | criterion | measure | outcome |
|---|---|---|---|
| **A** (primary) leads more consequential after adaptation | premium, mid stage, +8, G50: P(C1:s, C1) − P(baseline:s, baseline) | **met** if the difference's interval excludes 0, the ratio of means is ≥ 1.25, the difference is ≥ +0.08 resources and ≥ 5 of 6 seeds are positive. **Inconclusive** if the point ratio is ≥ 1.25 but the interval includes 0. **Not met** otherwise. Supporting cells (mid +16 G50, late +8 / +16 G50; Holm) are reported and must not contradict, meaning no significant negative difference |
| **B** no excessive early amplification | (i) early +8 premium at G20 (immediate effect, before the mid-game), ratio of means; (ii) natural play: slope of the signed earned lead at turn 150 on the lead at turn 50; (iii) share of present turns with an attack ratio ≥ 1.25; (iv) share of games with \|earned lead\| at turn 150 ≥ 15% of mean earned gold | **flag** if (i) ratio > 1.5 with the difference's interval excluding 0; or (ii) the C1 slope exceeds the baseline's by > 0.3 (interval excluding 0) and is > 1.3; or (iii) > 5%; or (iv) > 10% and > 2× the baseline's |
| **C** lane and wave side effects | natural play, C1 arm against baseline arm: crashed HP, plates by turns 150 and 300, deaths, Recalls, collection, connected trades, freeze-held share, absence, Ward rate; Shove share overall and late (turns 240–600); late crashed HP; late wave capacity with its consequences (late freezes held, one-turn breaks) | **issue** if a metric moves by more than ±25% (Shove share ±0.10) with the interval excluding 0; **note** beyond ±15%. Higher late capacity is not by itself an issue; it is judged by these consequences |
| **D** the weaker side keeps useful responses | battery and natural_cf, trailer focal, C1 arm: (i) whether a forced response beats the policy; (ii) "contest-and-die": the trailer's deaths over 50 turns from lead cells ≥ 1.5× the baseline arm's AND Safe-only or reset beats the policy (interval excluding 0) | **flag** if (i) a forced response beats the C1 policy by > 0.15 with the interval excluding 0 in a cell group where it does not beat the baseline policy, or (ii) holds |
| **E** health and no regression | stop conditions; baseline arm final against start:s in the baseline environment, W 1000 | **issue** if a stop condition fires, or if baseline:s − start:s < −0.2 (pooled), which would make the continuation procedure suspect |

**Recommendation rule.**
1. **C1 ready to become the candidate environment**: A met, no B flag, no C issue, no D flag, E passes.
2. **Retain the baseline**: A not met; or A met with issues in two or more of B, C and D.
3. **Resolve one specific issue first**: A met (or inconclusive under the extension rule) with exactly one flag or
   issue in B–D. The report names the issue and the focused test that would settle it.

Freeze contests are reported and interpreted, not a criterion: C1 is not expected to solve freezing. Contesting, taking
damage and occasionally dying can be rational, so a death count alone is never a flag.

## 6. Exploratory (labelled in the report)

- a strength-matched supplement: leads chosen so that C1's duel ratio at the cell equals the baseline's at +8,
  separating the curve's leverage from how each policy converts a given strength;
- the specialists under C1;
- representative trajectories;
- anything added after outcomes are seen, with a dated amendment before it runs.
