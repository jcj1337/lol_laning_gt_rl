# M4.6c — C1-versus-baseline training pilot

*2026-10-15. Status: complete, awaiting review. `GameConfig.m45_calibrated()` is unchanged. C1
(`m45_calibrated(invest_gain=3.24, invest_scale=64.0)`) remains an experimental opt-in configuration and is not a
default. The reward, combat, Recall, Vision and wave rules were not changed. Nothing is staged or committed.*

- Protocol: [m4_6c_preregistration.md](m4_6c_preregistration.md). It was frozen before any training run (SHA-256 in
  `results/m46c/preregistration.json`, with three dated amendments, §2.4).
- Generated tables: `results/m46c/report.md`.
- Previous stages: [M4.6a](m4_6a_report.md) and [M4.6b](m4_6b_investment_report.md).

**Terms.**
- A *resource* is the gold of one fresh wave; natural income is about 0.40 resources per turn.
- *G_h* is a seat's discounted gold lead over h turns: the training objective truncated at h.
- A *+8 lead* is 8 resources (20 turns of income) of extra invested gold on one side, at one of three stage-typical
  states: early (level 8, 9 invested), mid (16, 42) and late (18, 108).
- The *premium* of a lead is the leader's extra G_h over the equal state, policy against itself, averaged over three
  lane situations and both seat orientations (§4.1). In this zero-sum game it is also exactly the trailer's extra loss.
- *start:s* is the M4.5c 16.384M final of seed s.
- *baseline:s* and *C1:s* are its continuations in each arm. *ext/…* are the 8.192M continuations.
- Every interval is a 95% t interval over the six training seeds (5 degrees of freedom), unless stated. "4+/2−" counts
  seeds by sign.

---

## Summary

**The question.** After shared-policy PPO adapts to each investment curve under the same training, does C1 make
purchased leads more consequential than the baseline, without unacceptable side effects?

**Answer: probably yes, modestly and mainly in the mid-game, with no measurable side effects. The pre-registered
primary test is not conclusive.**
- **Before adaptation the M4.6b effect replicates.** On the same held-out states, the M4.5c finals convert a mid-game
  lead ×1.5–1.9 more under C1.
- **After 4.096M transitions of matched adaptation, most of it had gone.** The registered statistic, the premium of a
  mid-game +8 lead over 50 turns (C1 − baseline), was +0.06 ± 0.14 (×1.21): not met. It was moving in opposite
  directions in the two arms, so the pre-registered extension rule fired.
- **After 8.192M it is back.** The same statistic is +0.09 ± 0.11 (×1.32, 5 of 6 seeds): inconclusive, because the
  interval just includes 0. The supporting mid-game +16 cell is +0.34 ± 0.17 (×1.76, 6 of 6 seeds, significant after
  Holm).
- **The effect is in the mechanics.** At 8.192M, C1-trained and baseline-trained policies play the C1 game equally well
  in both roles.
- **It keeps paying beyond 50 turns** (exploratory, powered). At 150 turns the mid +16 difference grows to
  +0.54 ± 0.18 (×1.98, 6 of 6 seeds). Mid +8 stays at about +0.1 (×1.3), and late leads gain nothing at any horizon.
- **What C1 does not do.**
  - **No early decisiveness:** the immediate (20-turn) early premium is unchanged (×0.97).
  - **No late-game effect at 50 turns.**
  - **Natural play is almost unchanged.** Natural leads stay small and are not decisive (an attack ratio ≥ 1.25 on
    0.05% of turns). Every lane metric stays within ±10%, except Ward use, which is down but not significantly.
  - **The higher late wave capacity (+4.2 HP)** has no measurable consequence, except that guarding a late freeze
    costs more.
  - **The weaker side keeps its options:** no forced response beats its policy, it does not die more, and contesting
    freezes stays correct.
  - **Head to head:** a C1-trained policy is no stronger than a baseline-trained one in either environment.

**Pre-registered rule outcome (8.192M scorecard, as fixed by amendment 1):**

| A, primary | B, early amplification | C, side effects | D, weaker side | E, health |
|---|---|---|---|---|
| inconclusive | no flag | notes only (Ward −32%, late deaths +19%; both not significant) | no flag | pass |

This maps to **option 3: resolve one specific issue first.** That issue is confirming the primary effect, which is
plausible, consistent in direction and harmless, but not established at the registered cell with six seeds. A focused
confirmation is proposed in §12. If the user prefers to decide now: both environments are coherent, and nothing
measured argues against C1. Its benefit is real in constructed mid-game lead states and small in natural play.

**Freezes.** C1 neither helps nor harms them. Breaking stays cheap, as M4.6a explained, for structural reasons that
the investment curve does not touch.

---

## 1. The training objective, verified

Code read: `champions_m4.resolve_turn_m4` (the ledger), `env/core.py` (`LaneEnv._step_m4`), `rl/ppo.py`
(`PPOConfig.continuing`, `compute_gae`), `rl/shared.py` (`SharedCollector`), `config.py` (`m45_calibrated`).

| element | implemented | where |
|---|---|---|
| per-turn reward to seat A | r = E_A − E_B, where E is the gold earned that turn: farm + plate gold + kill bounty, in exact integer ledger units divided by 10⁷. Seat B gets −r | `resolve_turn_m4` (`reward_units = earned[A] − earned[B] + xp_reward_units`); `LaneEnv` returns (r, −r) |
| XP | not rewarded: `xp_reward_weight = 0.0`, so the XP term is identically 0 | `config.py` |
| shaping | none | |
| discount | γ = 0.99, part of the game (a training γ that differs is refused) | `check_time_semantics` |
| terminal handling | no terminal state (`terminal_rule = "none"`). A training game ends only by truncation, bootstrapped with the critic's value of the next observation (512 turns from s0; 150 turns from the archive here) | `GameTimes`, `compute_gae` |
| scale | rewards × 0.1 for learning only; every reported value is unscaled | `PPOConfig.reward_scale` |
| kill bounty | 2.0 resources | `m45_calibrated` (inherited from the M4.3c econ candidate) |

The documentation agrees with the code:
- the mathematical reference: §8.6 the reward, §8.7 `xp_reward_weight` as an EXPERIMENTAL opt-in that is off,
  §9.1–9.2 the objective and training;
- the README: "the difference in gold earned, nothing else".

The "gold and XP" objective was an M4.3c proposal. It was screened ("XP flips nothing") and never adopted.

**No conflict. The objective, the zero-sum discounted earned-gold lead, was preserved unchanged in both arms.** The
mathematical reference is unchanged, because no rule, parameter of the preset or objective changed.

## 2. Design, budgets and runtime

### 2.1 Arms and training

| item | both arms |
|---|---|
| environment | baseline `m45_calibrated()`; C1 `m45_calibrated(invest_gain=3.24, invest_scale=64.0)`. Checked against `m46b/curves.py`: the two configurations differ in exactly these two fields (test) |
| initialization | the six 16.384M M4.5c finals; seed s continued in both arms (six matched pairs) |
| optimizer | a fresh Adam, lr 3·10⁻⁴ annealed linearly to 0 over the run. Checkpoints store no Adam moments, and every earlier warm-started run (the M4.5c and M4.6a attackers) did the same |
| budget | 4,096,000 agent transitions per run (1,000 updates, 25% of the finals' training), then the pre-registered extension to 8,192,000 (§2.4) |
| PPO and self-play | M4.5c unchanged: 100% current-policy shared self-play (one network, both seats trained), entropy 0.03, clip 0.2, 4 epochs, minibatch 256, value clip 0.5, gradient norm 0.5, 16 environments × 128-turn segments, 2 × 128 tanh MLP, snapshots every 25,600 transitions |
| PPO seed | 98,000 + s, the same in both arms |
| start distribution | each game starts from the shared archive with probability 0.532 and lasts 150 turns, otherwise from s0 for 512 turns. Each archive draw is mirrored with probability ½. Realized share of agent transitions in archive games: 0.245–0.259 |

**Continuation, not fresh initialization.** A fresh run would need about 16M transitions per arm and seed to reach
the finals' strength. Continuing estimates the curve difference after a fixed amount of adaptation from a strong
common start.
- The C1 arm starts from behaviour learned under the baseline.
- The matched baseline arm controls for everything else the continuation does: the extra training, the archive starts
  and the re-warmed learning rate.
- The pilot cannot say what a from-scratch C1 campaign would converge to.
- Without a 0%-archive control it does not identify the effect of diverse starts.

**Exactly paired starts.** Training games end only at fixed lengths, so the sequence of start states depends only on
the seed. Both arms of a seed saw the same 6,500 games' start states in the same order.

### 2.2 The start archive

One file, 4,800 states, used identically in both arms (`archive.py`). States are defined by gold, levels, HP,
positions and waves, never by a combat ratio, so the same purchased lead is a larger strength gap under C1 at mid and
late investment: the effect under test.

| family | states | content | share of archive transitions |
|---|---|---|---|
| natural | 1,800 | decision states of the six finals' s0 self-play (baseline environment), turns 5–299 | 37% |
| natural + purchased lead | 1,500 | the same kind of states with one side's invested gold raised by 2–32 resources (log-uniform; median 7.6) | 32% |
| constructed zoning | 900 | the M4.5b situations neutral / A's push / B's push at levels 3–18, investment on the audit's natural path × 0.7–1.3, lead 0 (¼) or 2–32, HP full (½) or 0.35–1 | 19% |
| constructed freezes | 600 | the M4.5b freeze builder at levels 3–15, held wave 5–20, the breaker walked up or at its lane cell, a 2–24 lead for the freezer, the breaker or neither | 12% |

The network plays both seats, so every archive game trains the stronger and the weaker role together.

**Held out.** The evaluation cells are the M4.6b stage cells and S4 freeze cells, and none is in the archive (checked
by state key: 0 collisions). Natural evaluation states come from each evaluated policy's own self-play on new seeds.
The constructed cells belong to the archive's families, so they test in-family generalization, not a new family.

### 2.3 Runtime

| stage | wall clock (6–7 workers) |
|---|---|
| archive | 21 s |
| smoke test (both arms, 10 updates, bit-identical re-run) | 27 s |
| training, 12 × 4.096M (two waves of 6) | 47 min (about 2,950 transitions/s per run, peak 645 MB each) |
| evaluation of the 4.096M policies: premium 27 min, h2h 15 min, freeze 13 min, natural counterfactuals 11 min, battery 7 min, specialists 6 min, the rest a few minutes each | about 1 h 40 min |
| extension training, 12 × 8.192M (two waves of 6) | 1 h 57 min. Another application on the machine (a game client) took CPU throughout, so runs went at about 1,700–2,000 transitions/s instead of 2,950. Results are bit-reproducible, so only the wall clock changed |
| extension evaluation (premium 29 min, natural counterfactuals 11 min, battery 7 min, h2h 7 min, the two role swaps 12 min, natural 1 min) | about 1 h 10 min |
| exploratory powered 150-turn premium, both budgets | 34 min |
| total | about 7 hours of wall clock, of which about 2.7 h training |

### 2.4 Deviations and amendments (all dated in `preregistration.json`, each before its runs)

| item | what happened |
|---|---|
| seeds | the brief suggested about three paired seeds. Six were used, because 4.1M-transition runs take 23 minutes and six seeds tighten the across-seed interval about 2.4× (t(5) instead of t(2)). Written into the pre-registration before training |
| extension | the pre-registered extension rule fired (§4.5), and a matched 8.192M continuation ran for both arms. Amendment 1 fixed the re-evaluated families before any extension run, and fixed that the recommendation uses the 8.192M scorecard |
| exploratory role swap | added after the 4.096M premium decomposition (§4.2), before it ran (amendment 2) |
| exploratory powered 150-turn premium | added after the pre-registered 150-turn continuation check suggested that the arm difference grows with the horizon (§4.4), before it ran (amendment 3) |
| library change | `lanegame/rl/shared.py`: an opt-in `collector_cls` argument of `train_shared` (default unchanged, bit-identical, tested), mirroring M4.6a's argument of `train_ppo` |
| a reporting bug | during development the report generator briefly had a broken string literal. It was fixed before any stage used it; no result was affected |
| machine load | a game client ran during the extension and slowed it about 1.6×. All runs are seeded and deterministic, so only the wall clock changed |
| decision budget | amendment 1 made the 8.192M scorecard decisive, before any extension result existed |

## 3. Training health and learning

All 24 runs (12 at each budget) completed with every pipeline check passing (11 per run: 132 / 132 per budget) and
no pre-registered stop condition.

| quantity (mean over the six runs of an arm) | baseline 4.096M | C1 4.096M | baseline 8.192M | C1 8.192M |
|---|---|---|---|---|
| entropy, first → last 10% of updates | 1.78 → 1.76 | 1.75 → 1.76 | 1.77 → 1.75 | 1.76 → 1.77 |
| approximate KL, mean / largest | 0.0038 / 0.0094 | 0.0038 / 0.0097 | 0.0038 / 0.0093 | 0.0038 / 0.0097 |
| clip fraction | 0.044 | 0.044 | 0.044 | 0.043 |
| largest gradient norm | 1.69 | 1.66 | 1.87 | 1.84 |
| explained variance, first → last 10% | 0.44 → 0.44 | 0.41 → 0.43 | 0.44 → 0.46 | 0.42 → 0.45 |
| Shove share of training actions, first → last 10% | 0.30 → 0.30 | 0.30 → 0.32 | 0.30 → 0.33 | 0.30 → 0.32 |
| deaths per 100 training turns, first → last 10% | 1.50 → 1.38 | 1.61 → 1.37 | 1.48 → 1.17 | 1.58 → 1.25 |
| archive share of transitions | 0.245–0.259 | 0.245–0.259 | 0.249–0.255 | 0.249–0.255 |

**Learning curves.** Each snapshot is played against its own starting final from s0 in its own environment (W 460,
32 paired seeds). Under C1 the starting final is out of distribution.

| transitions | baseline arm | C1 arm |
|---|---|---|
| 0.512M | −0.67 ± 0.35 (0+/6−) | −0.50 ± 0.31 (0+/6−) |
| 1.024M | −0.56 ± 0.20 | −0.56 ± 0.60 |
| 2.048M | +0.00 ± 0.14 | −0.28 ± 0.58 |
| 3.072M | −0.18 ± 0.18 | −0.02 ± 0.27 |
| 4.096M | +0.04 ± 0.30 | +0.21 ± 0.38 |
| 4.096M, W 1000, 64 seeds | **+0.22 ± 0.26** (5+/1−) | **+0.27 ± 0.22** (5+/1−) |
| 8.192M, W 1000, 64 seeds | **+0.51 ± 0.26** (6+/0−) | **+0.41 ± 0.18** (6+/0−) |
| 8.192M against the same arm's 4.096M final (W 460) | +0.26 ± 0.36 | +0.12 ± 0.18 |

- **The re-warmed learning rate costs strength at first.** Both arms fall about half a resource below their starting
  policies in the first million transitions. They recover by 2–3M, while the learning rate anneals. A continuation
  campaign should use a lower peak learning rate or a warm-up. This affects both arms equally, so it does not bias the
  comparison.
- **The continuation does not regress** (criterion E). baseline:s − start:s is +0.22 ± 0.26 at 4.096M and
  +0.51 ± 0.26 at 8.192M, positive in all six seeds. Both arms are still improving: neither is converged.

## 4. Leads after adaptation (the primary question)

### 4.1 The premium, each arm's own policies in their own environment

Design: 45 held-out cells per seed. Each cell is a stage (early / mid / late), a situation (neutral, A's push, B's
push) and a lead (0, A+8, B+8, A+16, B+16). K = 128 paired rollouts per cell, horizon 50.

The premium is P = ½[(G(A+d) − G(equal)) − (G(B+d) − G(equal))], averaged over situations. Seat effects and the
equal cell cancel.

The 8.192M arms are listed first, because amendment 1 makes them decisive. The last column is the same comparison at
4.096M.

| stage | lead | h | baseline 8.192M | C1 8.192M | C1 − baseline, 8.192M | ratio | seeds | C1 − baseline, 4.096M |
|---|---|---|---|---|---|---|---|---|
| early | +8 | G20 | +0.37 ± 0.04 | +0.36 ± 0.07 | −0.01 ± 0.07 | ×0.97 | 3+/3− | −0.07 ± 0.11 (×0.85) |
| early | +8 | G50 | +0.63 ± 0.06 | +0.71 ± 0.08 | +0.08 ± 0.06 | ×1.13 | 5+/1− | +0.01 ± 0.20 (×1.02) |
| early | +16 | G20 | +0.68 ± 0.11 | +0.74 ± 0.13 | +0.06 ± 0.15 | ×1.09 | 4+/2− | −0.08 ± 0.12 (×0.90) |
| early | +16 | G50 | +1.22 ± 0.14 | +1.40 ± 0.18 | +0.18 ± 0.23 | ×1.15 | 5+/1− | +0.07 ± 0.27 (×1.06) |
| mid | +8 | G20 | +0.13 ± 0.04 | +0.20 ± 0.08 | +0.07 ± 0.08 | ×1.50 | 5+/1− | +0.01 ± 0.09 (×1.04) |
| **mid** | **+8** | **G50** | **+0.29 ± 0.03** | **+0.38 ± 0.12** | **+0.09 ± 0.11** | **×1.32** | **5+/1−** | **+0.06 ± 0.14 (×1.21)** |
| mid | +16 | G20 | +0.22 ± 0.05 | +0.36 ± 0.06 | +0.14 ± 0.07 | ×1.62 | 6+/0− | +0.10 ± 0.09 (×1.48) |
| mid | +16 | G50 | +0.45 ± 0.05 | +0.78 ± 0.14 | **+0.34 ± 0.17** | ×1.76 | 6+/0− | +0.16 ± 0.27 (×1.33) |
| late | +8 | G20 | +0.07 ± 0.04 | +0.08 ± 0.04 | +0.01 ± 0.03 | ×1.10 | 4+/2− | −0.01 ± 0.05 (×0.89) |
| late | +8 | G50 | +0.14 ± 0.09 | +0.19 ± 0.08 | +0.05 ± 0.17 | ×1.35 | 3+/3− | +0.04 ± 0.15 (×1.38) |
| late | +16 | G20 | +0.14 ± 0.01 | +0.13 ± 0.05 | −0.02 ± 0.07 | ×0.87 | 2+/4− | −0.04 ± 0.06 (×0.67) |
| late | +16 | G50 | +0.27 ± 0.11 | +0.24 ± 0.08 | −0.03 ± 0.16 | ×0.89 | 3+/3− | +0.04 ± 0.13 (×1.21) |

**Criterion A.**

| budget | registered cell (mid +8 G50) | per seed | outcome | supporting cells (Holm) |
|---|---|---|---|---|
| 4.096M | +0.06 ± 0.14, ×1.21, 4+/2− | +0.18, −0.11, −0.05, +0.24, +0.03, +0.08 | **not met** (ratio below 1.25, interval includes 0) | none significant |
| 8.192M | +0.09 ± 0.11, ×1.32, 5+/1−, p = 0.09 | −0.01, +0.06, +0.18, +0.27, +0.03, +0.03 | **inconclusive**: ratio ≥ 1.25, difference ≥ +0.08, 5 of 6 seeds positive, but the interval (−0.02, +0.20) includes 0 | mid +16 significant and positive (p = 0.004); late +8 and late +16 not significant; no contradiction |

At 8.192M:
- **Mid-game:** every cell has C1 ahead (×1.32–1.76, 5–6 of 6 seeds). The +16 lead's 50-turn premium rises from
  +0.45 to +0.78 resources.
- **Early:** the immediate effect (G20) is unchanged. The 50-turn premium is ×1.13–1.15, because a lead bought early
  keeps more of its strength into the mid-game, as the M4.6b mapping predicts.
- **Late, at 50 turns:** no difference. C1's late leverage is the smallest of the M4.6b candidates, and here it is
  not detectable.
- **Size.** The registered effect is +0.09 resources per 8-resource mid-game lead over 50 turns, about a quarter of a
  turn's income. It is a modest strengthening, in line with C1 being the modest candidate.

### 4.2 What adaptation did: the 4.096M dip and its reversal

Decomposition at G50, at 4.096M (the same cells and seeds):

| stage | lead | start:s, baseline env | start:s, C1 env (cross-env) | **before adaptation**: C1 − baseline | baseline:s in C1 env (cross-env) | **C1-trained − baseline-trained, C1 env** | baseline:s − start:s, baseline env |
|---|---|---|---|---|---|---|---|
| early | +8 | +0.64 | +0.77 | **+0.13 ± 0.10** | +0.86 | −0.12 ± 0.31 | +0.09 ± 0.09 |
| early | +16 | +1.13 | +1.40 | **+0.27 ± 0.03** | +1.51 | −0.20 ± 0.21 | +0.10 ± 0.18 |
| mid | +8 | +0.27 | +0.42 | **+0.14 ± 0.13** (×1.53) | +0.40 | −0.03 ± 0.16 | +0.03 ± 0.16 |
| mid | +16 | +0.42 | +0.78 | **+0.36 ± 0.13** (×1.85) | +0.74 | −0.11 ± 0.26 | +0.05 ± 0.20 |
| late | +8 | +0.15 | +0.13 | −0.02 ± 0.17 | +0.16 | −0.01 ± 0.13 | −0.04 ± 0.15 |
| late | +16 | +0.20 | +0.24 | +0.03 ± 0.15 | +0.27 | −0.02 ± 0.25 | +0.00 ± 0.16 |

- **Before adaptation the M4.6b result replicates.** The same untrained-for-C1 policies, on the same states, convert a
  mid-game lead ×1.5–1.9 more under C1 (M4.6b's powered follow-up: ×1.4–1.6). Early leads gain too, while the late
  gain is absent, as in M4.6b.
- **At 4.096M, training under C1 had removed most of it.** In the C1 environment, C1-trained policies had a *smaller*
  premium than baseline-continued policies trained with the identical procedure (−0.03 to −0.20, none significant).
- **The exploratory role swap shows who.** Under C1, with a baseline-trained opponent in seat B, the seat-A policy is
  either baseline-trained or C1-trained:
  - at 4.096M the C1-trained *leader* converted early leads worse: −0.24 ± 0.14 at early +16, 0 of 6 seeds;
  - mid-game, neither role differed significantly.

  The C1 arm's first half of training made its leaders more cautious early, a transient.
- **At 8.192M the shrinkage is gone.**
  - C1-trained against baseline-trained in the C1 environment is −0.09 to +0.06 at every cell.
  - The role swap finds no role-specific difference: conversion −0.06 to +0.12, defence +0.04 to +0.13, none
    significant.
  - The C1 arm's J2 gains over start:s are now balanced: conversion +0.27 and defence +0.27 at mid +16, against +0.05
    and −0.00 for the baseline arm.

  What remains is the mechanics' own effect: C1-trained policies in C1 (+0.78 at mid +16) match baseline-trained
  policies in C1 (+0.72) and the start policies in C1 (+0.78).
- **The baseline arm's premium barely moves** at either budget (−0.01 to +0.10 against start:s).

The registered statistic at 4.096M was therefore read during a transient of adaptation. The extension rule, which
fired on exactly that movement, did its job. The two budgets are separate runs (the 8.192M runs were not continued
from the 4.096M ones), so the reversal is also partly run-to-run variation, which is itself sizeable: the per-seed
primary values at the two budgets are only weakly correlated.

**Where the lead's value goes** (8.192M policies). These are undiscounted channels over 50 turns for a +16 lead:
lead cells minus the equal cell, both orientations. They are accounting, not causation.

| stage | arm | leader earned | of which farm / plates / bounty | leader turns away | leader connected trades | trailer earned | trailer turns away |
|---|---|---|---|---|---|---|---|
| early | baseline | +0.82 | +0.34 / +0.34 / +0.14 | −0.40 | +0.35 | −0.67 | +0.34 |
| early | C1 | +0.95 | +0.34 / +0.37 / +0.23 | −0.57 | +0.50 | −0.80 | +0.45 |
| mid | baseline | +0.28 | +0.12 / +0.08 / +0.08 | −0.13 | +0.20 | −0.27 | +0.11 |
| mid | C1 | **+0.58** | +0.21 / +0.23 / +0.14 | −0.20 | +0.24 | −0.40 | **+0.25** |
| late | baseline | +0.10 | +0.02 / +0.03 / +0.05 | −0.15 | +0.04 | −0.24 | +0.06 |
| late | C1 | +0.17 | +0.04 / +0.05 / +0.08 | −0.12 | +0.16 | −0.14 | +0.14 |

- **The mid-game leader converts a C1 lead twice as well,** through all three channels: its own farm, plates, and
  kills that send the trailer home for longer.
- **This is threat-based lane control, not passive farming.** The leader connects more trades. The trailer spends more
  turns away: absence, the channel M4.6a found for power gaps.
- **XP moves only early**, by about 2 XP, as in M4.6b.

### 4.3 The critics

The critic's value of the cell (seat A) against the realized Q50 = G50 + 0.99⁵⁰ V(s₅₀):

| cell | baseline arm | C1 arm |
|---|---|---|
| equal, mid | +0.19 ± 0.36 | +0.23 ± 0.15 |
| leader, early +16 | −0.98 ± 0.17 | −1.06 ± 0.40 |
| trailer, early +16 | +1.31 ± 0.26 | +1.36 ± 0.22 |
| leader, mid +16 | −0.28 ± 0.33 | −0.34 ± 0.36 |
| trailer, mid +16 | +0.61 ± 0.31 | +0.83 ± 0.24 |

At 8.192M the values are nearly the same:
- leader early +16: −0.97 (baseline), −1.04 (C1);
- trailer early +16: +1.27, +1.53;
- leader mid +16: −0.18, −0.45;
- trailer mid +16: +0.62, +0.95.

Even with a quarter of transitions from archive starts, both critics still undervalue an early 16-resource lead by
about one resource and overvalue the trailer's position by the same amount. M4.6a measured −1.3 to −1.6 for the much
larger ×1.76 gap.

The C1 critics are slightly more biased mid-game, where C1 leads are worth more. Power differences remain hard for the
value function in both arms. A plausible consequence (a hypothesis) is that conversion learning is slowed: the critic
under-rewards actions that create or exploit a lead.

### 4.4 Horizon: does the 50-turn readout miss a later difference?

Pre-registered continuation check: neutral cells only, K = 64, horizon 150.

| stage, lead | baseline G50 → G150 | C1 G50 → G150 | C1 − baseline at G50 / G100 / G150 |
|---|---|---|---|
| early +16 | +1.27 → +1.72 | +1.25 → +2.06 | −0.02 / +0.23 / +0.33 (± 0.6) |
| mid +8 | +0.45 → +0.55 | +0.29 → +0.53 | −0.15 / −0.02 / −0.02 (± 0.35) |
| mid +16 | +0.43 → +0.63 | +0.45 → +0.86 | +0.02 / +0.09 / +0.23 (± 0.5) |
| late +16 | +0.32 → +0.34 | +0.25 → +0.56 | −0.07 / +0.16 / +0.22 (± 0.2; 5+/1− at G150) |

Under the baseline a lead's premium is mostly collected within 50 turns. Under C1 it keeps accruing, as the M4.6b
mapping predicts: the same gap keeps more of its strength while both sides invest. The check is noisy (one situation,
K = 64). The powered exploratory follow-up is in §4.6.

### 4.5 The extension rule

The registered statistic at the 2.048M snapshots (the same 15 mid cells, K = 128) against the 4.096M finals:

| arm | 2.048M | 4.096M | change |
|---|---|---|---|
| baseline | +0.23 ± 0.05 | +0.30 ± 0.09 | +0.069 |
| C1 | +0.43 ± 0.22 | +0.36 ± 0.13 | −0.070 |

Both conditions held: A not met, and a move of at least 0.05 in either arm. So the matched 8.192M continuation ran.
The two arms moved toward each other: the C1 premium fell as C1 training went on, while the baseline premium rose.

### 4.6 Exploratory: the powered 150-turn premium (amendment 3)

The design: each arm's own policies, mid and late cells, all three situations, both orientations, K = 128, rolled to
150 turns. The G50 column reproduces §4.1 exactly, as it should with the same states and seeds. The table gives
C1 − baseline, with the ratio and the seed count:

| budget | cell | G50 | G100 | G150 |
|---|---|---|---|---|
| 4.096M | mid +8 | +0.06 ± 0.14 (×1.21) | +0.16 ± 0.17 (×1.48) | +0.17 ± 0.17 (×1.43, 6+/0−) |
| 4.096M | mid +16 | +0.16 ± 0.27 (×1.33) | +0.24 ± 0.32 (×1.38) | +0.29 ± 0.31 (×1.45, 5+/1−) |
| 4.096M | late +8 / +16 | +0.04 / +0.04 | −0.02 / +0.10 | −0.02 / +0.13 (n.s.) |
| 8.192M | mid +8 | +0.09 ± 0.11 (×1.32) | +0.10 ± 0.11 (×1.29) | +0.11 ± 0.12 (×1.29, 4+/2−) |
| 8.192M | **mid +16** | +0.34 ± 0.17 (×1.76) | +0.46 ± 0.18 (×1.86) | **+0.54 ± 0.18 (×1.98, 6+/0−)** |
| 8.192M | late +8 / +16 | +0.05 / −0.03 | +0.07 / +0.05 | +0.08 / +0.03 (n.s.) |

- **A mid-game C1 lead keeps paying beyond 50 turns.** The objective's weight extends there: 37% of it lies beyond
  turn 100.
- At 8.192M a 16-resource mid-game lead returns +1.08 resources over 150 turns under C1, against +0.55 under the
  baseline.
- **The 50-turn primary therefore understates C1's mid-game effect** somewhat, but not enough to change the registered
  outcome. Mid +8 stays at about +0.1 (×1.3) at every horizon.
- **Late leads do not gain under C1 at any horizon.** The neutral-only hint in §4.4 did not survive the powered design.
- **Exploratory, labelled:** it changes no pre-registered criterion.

## 5. The weaker side

### 5.1 Forced responses from constructed lead cells (10-turn behaviour-only references, the wave left to the policy)

The trailer is in seat A. The table gives G50 (candidate − policy), all three stages pooled:

| lead | arm | policy G50 | Hold only | Safe only | Trade only | reset | trailer deaths / 50 turns |
|---|---|---|---|---|---|---|---|
| +8 | baseline | −0.48 ± 0.15 | −0.17 ± 0.23 | −1.48 ± 0.20 | −1.53 ± 0.36 | −1.09 ± 0.22 | 0.46 |
| +8 | C1 | −0.46 ± 0.17 | −0.18 ± 0.19 | −1.48 ± 0.23 | −1.51 ± 0.17 | −1.38 ± 0.09 | 0.36 |
| +16 | baseline | −0.72 ± 0.13 | −0.25 ± 0.27 | −1.46 ± 0.10 | −1.58 ± 0.38 | −1.14 ± 0.29 | 0.51 |
| +16 | C1 | −0.79 ± 0.23 | −0.32 ± 0.23 | −1.37 ± 0.16 | −1.52 ± 0.39 | −1.27 ± 0.20 | 0.40 |

- No forced response beats the trailer's policy in any stage or arm. The best, staying and farming (Hold only), is
  0.0 to 0.4 below it.
- Retreating, fighting back or resetting for 10 turns each costs more than a resource.
- The C1 trailer dies 20–25% less often from the same cells. It loses about the same in G50: the trailer's loss is
  within ±0.07 of the baseline arm at every pooled lead size.
- No contest-and-die pattern: deaths fall, and no non-contesting response beats the policy.

**At 8.192M**, pooled over stages:

| lead | arm | policy G50 | Hold only | Safe only | Trade only | reset | trailer deaths / 50 |
|---|---|---|---|---|---|---|---|
| +8 | baseline | −0.60 ± 0.07 | −0.23 ± 0.14 | −1.41 ± 0.18 | −1.23 ± 0.20 | −0.99 ± 0.12 | 0.44 |
| +8 | C1 | −0.64 ± 0.18 | −0.24 ± 0.16 | −1.53 ± 0.33 | −1.45 ± 0.35 | −1.13 ± 0.31 | 0.44 |
| +16 | baseline | −0.90 ± 0.17 | −0.12 ± 0.17 | −1.45 ± 0.23 | −1.45 ± 0.21 | −0.96 ± 0.26 | 0.47 |
| +16 | C1 | −0.90 ± 0.20 | −0.30 ± 0.31 | −1.46 ± 0.26 | −1.59 ± 0.44 | −1.22 ± 0.28 | 0.47 |

- Again no forced response beats the policy in any stage or arm.
- Deaths are equal.
- The trailer's loss from constructed cells is the same in both arms. The larger C1 premium in self-play (§4.1)
  appears against an adapted C1 leader, whereas these cells pool the stages.

**Strong side** (leader in seat A). Forcing threaten (Trade whenever catchable) or Hold only is worse than the policy
in both arms and at both budgets. At +16, pooled, threaten − policy is:

| budget | baseline | C1 |
|---|---|---|
| 4.096M | −0.57 ± 0.29 | −0.39 ± 0.10 |
| 8.192M | −0.34 ± 0.13 | −0.52 ± 0.14 |

Indiscriminate bullying does not pay under either curve. The 8.192M C1 leader's own G50 at +16, pooled over stages,
is +0.71 against +0.45: it converts better by choosing its trades, not by trading more.

### 5.2 State-conditioned choices at natural lead states (one-turn counterfactuals)

24 trailer states and 24 leader states per seed come from each arm policy's own self-play: both champions present,
invested lead ≥ 4, turn ≥ 20. One-turn behaviour-only alternatives, K = 32, G50:

| focal | arm | Hold | Safe | Trade | Recall (where legal) | policy's mass on Trade / Safe | corr(P, value) Trade / Safe |
|---|---|---|---|---|---|---|---|
| trailer | baseline | −0.11 ± 0.13 | −0.13 ± 0.13 | −0.17 ± 0.16 | −0.19 ± 0.20 | 0.22 / 0.21 | +0.10 / +0.17 |
| trailer | C1 | −0.04 ± 0.09 | −0.13 ± 0.05 | −0.11 ± 0.19 | −0.25 ± 0.17 | 0.23 / 0.24 | +0.12 / +0.11 |
| leader | baseline | −0.03 ± 0.11 | −0.14 ± 0.10 | −0.04 ± 0.12 | | 0.34 / 0.14 | +0.28 / +0.17 |
| leader | C1 | −0.08 ± 0.13 | −0.09 ± 0.08 | −0.08 ± 0.13 | | 0.28 / 0.18 | +0.10 / +0.31 |

- **No one-turn alternative beats the policy on average**, for either role in either arm, at either budget.
  - At 8.192M, trailer: Hold −0.25 / −0.17, Safe −0.13 / −0.13, Trade −0.17 / −0.21, Recall −0.20 / −0.33
    (baseline / C1).
  - At 8.192M, leader: Trade −0.07 / −0.12.
- **The policy mixes, and its mixture leans the right way.** The probability it puts on Trade or Safe correlates
  weakly but positively with that option's counterfactual value, state by state (+0.10 to +0.35).

**What the policies do, by state** (self-play records, 8 games × 600 turns per policy; mean probability masses):

| state | arm | Trade | Safe | Recall | Shove |
|---|---|---|---|---|---|
| leading by ≥ 8 invested | baseline / C1 | 0.32 / 0.35 | 0.22 / 0.20 | 0.05 / 0.05 | 0.31 / 0.31 |
| even (\|lead\| < 3) | baseline / C1 | 0.28 / 0.27 | 0.24 / 0.24 | 0.07 / 0.07 | 0.30 / 0.32 |
| trailing by ≥ 8 | baseline / C1 | 0.24 / 0.23 | 0.34 / 0.35 | 0.10 / 0.11 | 0.31 / 0.33 |
| trailing ≥ 4 at HP < 50% | baseline / C1 | 0.13 / 0.12 | 0.68 / 0.69 | 0.26 / 0.25 | 0.23 / 0.26 |
| frozen on | baseline / C1 | 0.13 / 0.15 | 0.20 / 0.20 | 0.12 / 0.09 | 0.43 / 0.43 |

Both arms condition on the lead in the same way:
- the leader trades more and retreats less;
- the trailer retreats and recalls more, but still contests: when frozen on it Shoves to break 43% of the time.

C1 shifts none of this by more than a few points.

### 5.3 The M4.6a gap specialists as fixed opponents

The specialists sit in seat A from the 27 M4.6a gap cells (investment gaps of ×1.5–2.2 built under the baseline
curve), against π in seat B. The specialist's advantage is its G50 minus π's own strong-side G50:

| environment | π | advantage | defence gain over start:s |
|---|---|---|---|
| baseline | start:s | +1.69 ± 0.42 | |
| baseline | baseline:s | +1.34 ± 0.35 | +0.22 ± 0.26 (5+/1−) |
| C1 (specialists cross-env) | start:s | +2.29 ± 0.57 | |
| C1 (specialists cross-env) | C1:s | +2.44 ± 0.42 | +0.28 ± 0.34 (5+/1−) |

A quarter of the training games from archive starts made both arms somewhat harder to exploit from large gaps:
defence gains of +0.22 and +0.28, about a fifth of the specialists' edge. The specialists still convert much better
than either arm's policy. Shared self-play at this budget acquired only part of the gap skill that M4.6a's dedicated
specialists learned.

## 6. Wave contests and freezes

Constructed S4 cells (levels 4 / 10 / 15, held 5 / 12, investment equal / freezer +8 / breaker +8), G50:

| level | arm | breaker: break | walk up | step back | reset | freezer: keep | guard | freeze held / 20 turns under the policy | breaker deaths / 20 |
|---|---|---|---|---|---|---|---|---|---|
| all | baseline | +0.08 ± 0.08 | −0.22 ± 0.16 | −1.67 ± 0.12 | −0.67 ± 0.11 | −0.49 ± 0.14 | −0.19 ± 0.20 | 2.0 | 0.17 |
| all | C1 | −0.06 ± 0.11 | −0.31 ± 0.20 | −1.67 ± 0.18 | −0.74 ± 0.23 | −0.40 ± 0.13 | −0.49 ± 0.14 | 2.1 | 0.10 |
| 4 | baseline / C1 | +0.04 / −0.08 | −0.09 / −0.06 | −1.88 / −1.71 | −1.10 / −0.95 | −0.28 / −0.18 | −0.41 / −0.47 | 3.9 / 4.1 | 0.28 / 0.18 |
| 10 | baseline / C1 | +0.08 / −0.09 | −0.22 / −0.22 | −1.62 / −1.76 | −0.41 / −0.50 | −0.06 / −0.12 | +0.03 / −0.21 | 1.3 / 1.3 | 0.14 / 0.07 |
| 15 | baseline / C1 | +0.11 / −0.00 | −0.36 / −0.65 | −1.50 / −1.55 | −0.51 / −0.77 | −1.13 / −0.89 | −0.21 / −0.79 | 0.9 / 0.9 | 0.10 / 0.05 |

- **Contesting a freeze remains the breaker's right choice under both curves.**
  - An immediate break is as good as the policy.
  - Walking up costs a little.
  - Stepping back is the worst choice everywhere (−1.5 to −1.9).
  - Resetting costs about two-thirds of a resource.
- The C1 breaker dies about 40% less while contesting.
- **Freezes last as long under C1:** about 4 turns in 20 at level 4, 1.3 at level 10, 0.9 at level 15.
- **Guarding rarely pays.** A guard holds the freeze longer (6–7 of 20 turns at level 4) but costs value in both
  arms, except equal at level 10 under the baseline.
- **Late guarding is costlier under C1** (level 15: −0.79 against −0.21). This is the one place where C1's higher late
  wave capacity visibly removes a (weak) option.

At natural freeze states from each policy's self-play, no one-turn alternative (break, walk up, step back, keep,
guard) beats the policy in either arm.

## 7. Natural play and side effects

s0 self-play, 48 games × 600 turns per policy, on common seeds across arms. The table gives the C1 arm against the
baseline arm, paired by seed.

| metric | start:s (baseline env) | baseline arm | C1 arm | C1 − baseline | relative |
|---|---|---|---|---|---|
| crashed HP / 100 turns | 65.9 | 67.2 | 64.2 | −3.0 ± 7.4 | −4% |
| plates by turn 150 / by 300 | 1.88 / 2.72 | 1.94 / 2.79 | 1.78 / 2.69 | −0.16 ± 0.34 / −0.10 ± 0.12 | −8% / −4% |
| deaths / 100 turns | 0.62 | 0.64 | 0.63 | −0.00 ± 0.22 | −1% |
| Recalls / 100 turns | 4.22 | 4.36 | 4.28 | −0.07 ± 0.35 | −2% |
| farm collection | 0.762 | 0.760 | 0.760 | −0.001 ± 0.020 | 0% |
| connected trades / 100 turns | 13.5 | 14.1 | 14.0 | −0.1 ± 2.0 | −1% |
| freeze-held share (all / turns 241–600) | 0.047 / 0.043 | 0.048 / 0.044 | 0.050 / 0.048 | +0.002 / +0.004 | +4% / +9% |
| freeze episodes / 100 turns; mean length | 2.70; 1.73 | 2.84; 1.70 | 2.90; 1.73 | +0.06; +0.03 | +2% |
| Ward turns / 100 | 1.93 | 1.80 | 1.53 | −0.27 ± 0.42 | −15% (note) |
| Shove share (all / 101–240 / 241–600) | 0.373 / 0.405 / 0.369 | 0.357 / 0.388 / 0.351 | 0.374 / 0.421 / 0.362 | +0.017 ± 0.012 / +0.033 ± 0.018 / +0.011 ± 0.012 | |
| crashed HP / 100, turns 241–600 | 70.6 | 71.2 | 68.8 | −2.4 ± 7.7 | −3% |
| wave capacity C (1–100 / 101–240 / 241–600) | 8.3 / 16.1 / 18.7 | 8.3 / 16.0 / 18.7 | 7.9 / 17.1 / 22.9 | −0.3 / +1.1 / **+4.2** | −4% / +7% / +23% |

| leads and strength | start:s (baseline env) | baseline arm | C1 arm | C1 − baseline |
|---|---|---|---|---|
| \|earned lead\| median at turns 50 / 150 / 300 | 1.7 / 4.2 / 5.0 | 2.0 / 3.8 / 4.5 | 1.7 / 3.6 / 4.2 | −0.3 / −0.2 / −0.3 (n.s.) |
| \|earned lead\| at 150, q90 | 9.3 | 9.2 | 8.2 | −1.0 ± 1.8 |
| slope of the lead at 150 on the lead at 50 | 0.97 | 0.80 | 0.75 | −0.05 ± 0.34 |
| slope of the lead at 300 on the lead at 100 | 0.93 | 0.86 | 0.87 | +0.02 ± 0.19 |
| games with \|lead at 150\| ≥ 15% of earned gold | 12% | 11% | 8% | −2.4 ± 5.3 pts |
| attack ratio ≥ 1.25, share of present turns | 0.2% | 0.2% | 0.1% | |
| the gold leader's duel ratio, turns 20–50 / 90–150 / 240–320 | ×1.00 / ×1.04 / ×1.01 | ×1.02 / ×1.04 / ×1.01 | ×1.00 / ×1.06 / ×1.03 | |

**The same comparison at 8.192M** (C1 arm − baseline arm, paired by seed):

| metric | baseline arm | C1 arm | C1 − baseline | relative |
|---|---|---|---|---|
| crashed HP / 100 turns (all / late) | 68.6 / 73.6 | 67.6 / 72.5 | −1.0 ± 4.3 / −1.1 ± 2.5 | −1% / −1% |
| plates by 150 / by 300 | 1.95 / 2.72 | 1.82 / 2.73 | −0.13 ± 0.35 / +0.01 ± 0.19 | −7% / 0% |
| deaths / 100 turns (all / late) | 0.54 / 0.45 | 0.61 / 0.54 | +0.07 ± 0.09 / +0.09 ± 0.15 | +13% / +19% (note) |
| Recalls / 100 turns | 4.45 | 4.21 | −0.24 ± 0.28 | −5% |
| farm collection | 0.764 | 0.757 | −0.007 ± 0.013 | −1% |
| connected trades / 100 turns | 13.3 | 13.6 | +0.3 ± 1.2 | +2% |
| freeze-held share (all / late) | 0.047 / 0.043 | 0.052 / 0.049 | +0.004 / +0.006 | +10% / +14% |
| Ward turns / 100 | 1.93 | 1.31 | −0.62 ± 0.74 | −32% (note) |
| Shove share (all / 101–240 / 241–600) | 0.381 / 0.414 / 0.379 | 0.368 / 0.415 / 0.358 | −0.013 / +0.000 / −0.021 | |
| wave capacity C, turns 241–600 | 18.7 | 22.8 | +4.2 | +22% |
| \|earned lead\| at 150, median / q90 | 3.2 / 8.7 | 4.0 / 9.1 | +0.8 ± 1.1 / +0.5 ± 1.0 | |
| slope of the lead at 150 on the lead at 50 | 0.84 | 0.98 | +0.14 ± 0.33 | |
| games with \|lead at 150\| ≥ 15% of earned gold | 10.1% | 10.4% | +0.3 ± 5.8 pts | |
| attack ratio ≥ 1.25, share of present turns | 0.17% | 0.05% | | |
| the leader's duel ratio, turns 90–150 / 240–320 | ×1.04 / ×1.01 | ×1.06 / ×1.03 | | |

- **Natural play is almost unchanged at both budgets.**
  - At 4.096M every lane metric is within ±10%, except Ward use (−15%, not significant).
  - At 8.192M it is within ±10% except three notes, none significant: Ward use −32% (it is rare in both arms, about 1–2
    turns per 100), deaths +13% (late +19%) and late freeze-held share +14%.
  - The Shove share moves by at most 0.03, far inside the ±0.10 bound.
- **Late wave capacity is 4.2 HP higher, and nothing measurable follows from it.** Late crashes are 3% lower, late
  freezes are held slightly more (+9%, not significant), the late Shove share moves +0.01 and late deaths do not
  change.
- **Natural leads do not become decisive under C1.**
  - At 4.096M leads are slightly smaller and persistence is unchanged.
  - At 8.192M they are slightly larger and more persistent (slope 0.98 against 0.84, not significant). That is the
    direction the design intent asks for: a lead made early stays useful.
  - The decisive-game share and the strength ratios stay far from every bound.
  - The natural leader's mid-game duel ratio rises only from ×1.04 to ×1.06. Natural leads are small (about 4
    resources at turn 150, 6–7% of earned gold), so even C1's larger elasticity turns them into small strength
    differences, as M4.6b found.

## 8. Robustness

| measure (own environment, from s0) | baseline arm | C1 arm |
|---|---|---|
| final − start:s (W 1000, 64 seeds) | +0.22 ± 0.26 | +0.27 ± 0.22 |
| final against the other five starts (W 460) | +0.36 ± 0.24 | +0.12 ± 0.34 |
| against the nine M4.5c fixed references: final / start:s | +4.23 / +3.83 (+0.40 ± 0.37) | +3.66 / +3.90 (−0.24 ± 0.43) |
| within-arm round robin: mean \|value\| (pairs with an interval excluding 0) | 0.36 (1 of 15) | 0.46 (2 of 15) |
| C1:s − baseline:s, C1 environment / baseline environment (W 460) | | −0.04 ± 0.32 / +0.13 ± 0.56 |
| 8.192M: final − start:s (W 1000) | +0.51 ± 0.26 | +0.41 ± 0.18 |
| 8.192M: final against the other five starts (W 460) | +0.44 ± 0.20 | +0.54 ± 0.17 |
| 8.192M: ext C1:s − ext baseline:s, C1 environment / baseline environment | | −0.01 ± 0.36 / +0.09 ± 0.44 |

- **Training under C1 buys no head-to-head advantage under C1**, at either budget. A policy trained under the baseline
  plays the C1 game as well as one trained under C1, and the reverse. From s0 the two curves produce nearly the same
  game.
- Seeds are interchangeable in both arms (round robin).
- No conclusion rests on one run. The primary's per-seed values are listed in §4.1. The decomposition's signs hold in
  4–6 of 6 seeds.
- Against the fixed references the C1 finals did slightly worse than their starts, while the baseline finals did
  better (not significant). The references play C1 with fixed rules, so these numbers are not comparable across
  environments.

## 9. Representative trajectories

*Exploratory, illustrative only.* These are single games of the 8.192M seed-0 policies against themselves, on a
common seed (`results/m46c/traces.json`; 2 seeds × 4 scenarios × 2 arms). Notation: wave/behaviour (H Hold, Y Yield,
T Trade, S Safe, R Recall; +P Pressure); HP fractions after the turn; A's cumulative earned lead.

**Mid-game neutral lane, A leads by 16 invested, C1** (A's wave capacity C 16, B's 14):

| turn | A | B | HP A / B | event | A's lead |
|---|---|---|---|---|---|
| 3 | LH-S | SL-Y | 1.00 / 0.71 | B took minion and trade damage | 0.0 |
| 4 | SL-T | TN-H | 0.99 / dead | A's trade connects; B, already at 0.71, dies (5 turns away) | +2.0 |
| 5–9 | SH-H(+P) | absent | 0.99 / – | A shoves, crashes 5–8 HP a turn into B's tower and applies Pressure: a plate | +7.5 |
| 11–24 | mixed, trading | trading back | 0.84 → 0.12 / 0.99 | B returns and trades back; A keeps trading at falling HP | +6.5 … +7.8 |
| 26 | LH-R | TN-T | dead / 0.86 | A's Recall is interrupted and A dies: bounty to B | +4.8 |
| 27–30 | absent | SH-H(+P) | – / 0.87 | B crashes into A's tower | +1.2 |

**The same start, baseline** (C 15 against 14):
- turn 4: A's trade drops B to 8% HP, but B survives.
- turns 5–12: B plays Safe at its tower.
- turn 13: B recalls, and A shoves and pressures.
- A's lead peaks at +4.8 and settles near +2.5.

Both games show the conversion channel the tables measure: a lead turns a trade into a kill or a forced reset, and
the absence into farm, crashes and plates. Under C1 the same 16 resources made A's trade lethal (B at 0.71 rather than
full HP).

The C1 game also shows the risk. The leader kept trading at low HP, was caught recalling, and gave back most of the
lead. The trailer's patient trading back was a useful response, not a doomed one.

**Freeze, level 10, A holds B's 12-HP wave, B leads by 8** (both arms alike):
- B breaks at once (Shove from its walk-up cell), and B's wave crashes 9–10 HP into A's tower.
- The freeze is gone by turn 2.
- In the baseline game B then trades down to 39% HP, retreats and recalls. In the C1 game it is hit to 7% HP and
  retreats without dying.

That is §6's finding in miniature: contesting is right, the freeze does not survive, and neither curve changes it.

## 10. Scorecard (pre-registration §5)

| criterion | 4.096M | 8.192M (decisive, amendment 1) |
|---|---|---|
| **A** leads more consequential after adaptation: mid +8 G50, interval excluding 0, ratio ≥ 1.25, ≥ +0.08, ≥ 5 of 6 seeds | **not met**: +0.06 ± 0.14, ×1.21, 4+/2− | **inconclusive**: +0.09 ± 0.11, ×1.32, 5+/1−. Supporting mid +16 +0.34 ± 0.17 (×1.76, 6/6, Holm-significant); no contradiction |
| **B** no excessive early amplification | no flag: early +8 G20 ×0.85; natural slope (150 on 50) 0.75; attack ratio ≥ 1.25 on 0.1% of turns; decisive games 8% | no flag: G20 ×0.97; slope 0.98; 0.05%; decisive games 10.4% against 10.1% (the flag needs > 10% *and* > 2× the baseline's) |
| **C** lane and wave side effects | note: Ward −15% (not significant) | notes: Ward −32%, late deaths +19% (neither significant); no issue |
| **D** the weaker side keeps useful responses | no flag | no flag |
| **E** health, no regression | pass (+0.22 ± 0.26 over start) | pass (+0.51 ± 0.26) |
| **rule outcome** | 2: retain the baseline | **3: resolve one specific issue first** |

**A gap in the registered rule.** The rule lists:
- option 1 for A met with no flags;
- option 2 for A not met, or A met with two or more issues;
- option 3 for A met *or inconclusive* with exactly one issue.

The 8.192M outcome, A inconclusive with no flags, is not literally listed. The only option that allows an inconclusive
primary is 3, and the specific issue is the primary itself. The scorecard generator applies that reading.

## 11. What the pilot shows, and what it does not

**Demonstrated** (paired, pre-registered, six seeds):
1. Under matched shared-policy self-play, C1 makes a purchased mid-game lead worth more at 8.192M. All four mid-game
   cells are ×1.32–1.76 in 5–6 of 6 seeds. The +16 cell is significant after Holm correction; the registered +8 cell
   just misses its interval.
2. C1 does not make the immediate exchange of an early lead more decisive (G20 ×0.97). Its larger early 50-turn
   premium (×1.13) is a lead that stays useful into the mid-game.
3. At 50 turns, C1 does not change late-game leads.
4. Nothing measured in natural play moves beyond ±10% except rare Ward use and late deaths (notes, not significant).
   - Freezes, Shoves, crashes, plates, Recalls, collection and trades are all unchanged.
   - The higher late wave capacity (+4.2 HP) shows only as a costlier late guard (§6).
5. The weaker side's options survive.
   - No forced response beats its policy, from constructed or natural states, at either budget.
   - It does not die more.
   - Contesting a freeze stays right; stepping back stays worst.
6. The adapted policies are interchangeable across curves head to head. The curve changes the value of leads, not how
   good a policy must be to play the game.
7. Matched adaptation is not monotone. At 4.096M the C1 arm's leaders had become more cautious early and its premium
   had dipped; at 8.192M this had reversed.

**Hypotheses, not demonstrated:**
- That the 4.096M dip was a transient of adaptation rather than run-to-run variation. Both contribute. The two budgets
  are separate runs, and the per-seed primary values are weakly correlated across them.
- That the critics' persistent undervaluation of leads (about −1 resource early, §4.3) slows conversion learning.
- That a longer horizon would reveal a late-game C1 effect (§4.4, §4.6).
- Anything about a from-scratch C1 campaign, or about C2.

**About freezes.** The pilot confirms that freezes are not an investment-curve question.
- Under both curves a freeze holds about as long, a contest stays correct, and guarding pays only rarely.
- Under C1, late guarding pays even less.
- What limits freezes is the M4.6a mechanism: the breaker's wave clear against a fixed 8-HP wave, and a walk-up cell
  that farms at full rate.
- A stronger investment curve cannot be the answer, and the pilot does not suggest a mechanics change. Whether durable
  freezes are wanted remains a design choice. The model is coherent without them: freezing is used when it pays, and
  contesting it is rational.

**About what C1 buys.** C1 does what M4.6b designed it to do, and survives adaptation at 8.192M. But natural leads stay
small, about 4 resources at turn 150, so in natural self-play the change is nearly invisible: the leader's mid-game
duel ratio goes from ×1.04 to ×1.06. C1 matters in the states where leads are large, which the game rarely produces
and which the archive made common in training.

**What the pilot cannot claim:**
- It did not identify the effect of the diverse starts (no 0% control).
- The C1 arm started from baseline-trained behaviour.
- Six seeds bound the across-seed precision to about ±0.1 resources on the primary.

## 12. Recommendation

**Recommendation 3: resolve one specifically identified issue before choosing.**

The issue is the confirmation of the primary effect. C1's mid-game effect is consistent in direction, about the size
M4.6b predicted, and harmless by every pre-registered side-effect criterion. But with six seeds, the registered cell
(+0.09, ×1.32) is not distinguishable from zero. The per-seed spread comes mostly from training noise, which more
paired runs average out.

**Proposed focused step (not launched):**
- 6 more matched pairs at 8.192M: the same six finals, new PPO seeds 98,010 + s, every other setting unchanged
  (about 2 hours of training on an unloaded machine).
- Re-evaluate only:
  - the premium family (J1);
  - the natural lane metrics and the leads table.
- Analysis: by starting final, the two replicates averaged (t(5)), with the 12 pairs reported as a secondary.
- Decision: C1 becomes the candidate for the main campaign if criterion A is met on the pooled data and no B–D
  criterion flags. Otherwise the baseline is retained.
- Pre-register it before running, as here.

**If the user prefers to decide without it,** the evidence supports either choice. Neither choice risks the coherence
of the model.
- **C1:** a modestly stronger and more persistent mid-game lead, no measured cost.
- **Baseline:** nearly the same natural game, and no unconfirmed change to the environment.

My judgement on the substance: C1 is a safe, modest improvement toward the design intent. It is not a large one, and
it does not change the character of natural play.

**Not recommended:**
- Further investment-curve work (C2 or stronger) in this phase. The binding constraint on how often leads matter is
  how small natural leads are, as M4.6b showed; the curve's leverage is not it.
- A mechanics change for freezes on the strength of this pilot.

**For the main campaign, whichever curve:**
- Keep the archive start mixture. The critics still misvalue leads, and the M4.6a specialists still out-convert both
  arms by +1.3 to +2.4.
- Use a lower peak learning rate or a warm-up if continuing from existing finals. The re-warm cost about half a
  resource for the first million transitions.

## 13. Reproduction, files and tests

**Commands** (each stage resumable; per-job records are cached under `results/rl/checkpoints/m46c/`):

```bash
P="python -m lanegame.experiments.rl.m46c"
$P archive smoke --workers 6        # the shared archive (21 s); pipeline smoke test with a bit-identical re-run
$P prereg                           # freeze docs/m4/m4_6/m4_6c_preregistration.md (SHA-256)
$P train --workers 6                # 12 x 4,096,000 (47 min)
$P train_check
$P eval_premium eval_premium_mid eval_h2h eval_battery eval_natural eval_records \
   eval_natural_cf eval_freeze eval_specialists eval_premium150 --workers 7
$P train --phase ext --workers 6    # 12 x 8,192,000 (amendment 1)
$P train_check --phase ext
$P eval_premium_ext eval_h2h_ext eval_roleswap eval_roleswap_ext eval_natural_ext eval_battery_ext \
   eval_natural_cf_ext eval_premium150p eval_premium150p_ext --workers 7
$P traces analyse report
python -m pytest tests/test_m46c.py
```

The same sequence is recorded in `results/m46c/run_eval.sh`, `run_ext.sh` and `run_after_ext.sh`.

**New files:**
- `lanegame/experiments/rl/m46c/`: common, archive, collector, train, evaluate, natural, analysis, report, traces,
  CLI.
- `tests/test_m46c.py`: 8 tests.
- `docs/m4/m4_6/m4_6c_preregistration.md` and this report.
- `results/m46c/`: pre-registration record with the three amendments, archive description, training records
  (`train_main.json`, `train_ext.json`), `analysis.json`, `scorecard.json`, `traces.json`, the generated `report.md`,
  stage logs and resource logs.

**Modified:**
- `lanegame/rl/shared.py`: the opt-in `collector_cls` argument of `train_shared`, which records `describe_starts()` in
  final.pt when the collector has it. The default is unchanged and bit-identical (tested).
- `docs/m4/README.md`: the index.
- `docs/rl_research_roadmap.md`: §AO and the status row.

**Not changed:**
- `GameConfig.m45_calibrated()` and every other preset, rule, reward and default.
- The mathematical reference: no rule or objective changed.
- Every existing checkpoint and result. New outputs went to new roots.

**Tests.** `tests/test_m46c.py`:
- the arms differ in exactly the two investment fields, and the objective fields are as stated;
- policy specifications;
- the `collector_cls` default is bit-identical;
- the mixed collector at β = 0 is bit-identical to `SharedCollector`;
- the mixed collector's start draws, game lengths and mirroring;
- constructed archive states are valid in both arms and disjoint from the evaluation cells;
- the premium statistic cancels seat effects and the equal cell;
- the t-interval helper.

Run on the final code: `tests/test_m46c.py` 8 passed. The existing suites that exercise the trainer and the reused harness also pass: `test_rl_shared`, `test_m45c_training`, `test_m46a`, `test_m46b`, `test_m4_time`, `test_rl_m43b`, `test_m44a_readiness`, `test_rl_m4_lasthit` and `test_m43c_four_actions`, 128 passed.

**Seeds:** 98,000–98,999. **Checkpoints** (git-ignored):
- `results/rl/checkpoints/m46c/train/{smoke,main,ext}/<arm>/s<seed>/`, with run.json, history, snapshots every
  25,600 transitions, final.pt and summary.json;
- the archive and every evaluation record under `results/rl/checkpoints/m46c/`.

Nothing is staged or committed.
