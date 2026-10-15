# M4.6a — Diagnosis of freeze breaking and threat-based zoning

*2026-10-15. Status: diagnosis complete, awaiting review. Environment `GameConfig.m45_calibrated()`, unchanged: no
rule, parameter, observation, action, reward or network changed, no M4.6b lever was evaluated and no general self-play
training ran. Protocol: [m4_6a_preregistration.md](m4_6a_preregistration.md), frozen before any experiment (SHA-256 in
`results/m46a/preregistration.json`). Design: [m4_6_design_review.md](m4_6_design_review.md). Generated tables:
`results/m46a/report.md`. Nothing is staged or committed.*

**Terms used throughout.** One *resource* is the gold of one fresh wave; a side's income is about 45 resources per 100
turns. *G_h* is the focal side's discounted gold lead over h turns; *Q_h* adds the critic's value at turn h. A *paired*
difference compares two interventions from the same state on the same chance and sampling streams. *×1.76* is a
champion whose investment multiplies its attack power and its durability by 1.76 relative to the opponent. A *bar* is a
champion's maximum HP. *The policy* means the six 16.384M finals of M4.5c. "k+/k− of 6" counts finals whose own
interval excludes zero.

---

## Summary

**The two concerns have different causes.**

**Zoning: the game already rewards it, and PPO learns it when trained on the right states.** M4.5c concluded that
stronger champions deny almost nothing. That verdict rested on measurement artefacts and on a training distribution
that almost never contains a power gap.
- **The rules reward strength.** A ×1.76 champion's Trade beats Hold in one turn (E1). Under the trained policy it
  turns the gap into +4.5 resources (G100), half through kills and the weaker side's absence and only 9% through
  posture denial (E3).
- **A learned specialist converts 50% more.** Trained from gap states, it adds **+2.11 ± 0.62** (3 of 3 targets),
  entirely through denial: about twice the kills and caught Recalls (E4). This meets the pre-registered "distribution
  binding" criterion. Its better play works against all 15 finals it never trained against.
- **No scripted threat rule matches the policy** (E2). The finals already convert better than simple rules; they
  simply never learned the rest, because gaps of ×1.25 or more occur on 0.1–0.4% of their training turns. Their critics
  undervalue a ×1.76 lead by about 1.5 resources (E6).
- **The game rarely creates gaps.** Natural self-play almost never produces them: a 10–20 resource gold lead gives a
  power ratio of about ×1.01, because investment saturates and levels converge.

**Freezes: cheap to break for structural reasons, not learning ones.**
- **A passive freeze breaks in one turn.** The walk-up cell is a full-rate farming cell.
- **A guard can hold one**, unexpectedly. The held wave fires at the guard instead of the freezer's minions, so the
  freeze survives in 9 of 18 constructed cells (early and mid game, or a weaker breaker). Even then, the Shove is still
  the breaker's best one-turn reply.
- **A freeze specialist learned only general exploitation.** At equal power it found nothing that transfers (E4-P2:
  "weak evidence").

**Other results.**
- **The reset is not subsidized.** Investing on return is worth 0.00 ± 0.12; a full-HP Recall costs about 0.4.
- **HP is not cheap.** A bar is worth 1.2–1.5 resources under learned play.
- **No single trading cost is load-bearing** (E1), so the design review's trade-scale lever is not supported.
- **Four artefacts in M4.4c / M4.5c readouts** distorted the zoning verdict (§8.5):
  - a position metric that averaged in the Ward node;
  - a posture-only denial metric;
  - and, in both M4.5c's bully plan and M4.4c's Farm plan, a forced Last Hit wave.

**Recommendation: training changes (M4.6c) first, with the measurement fixes; no mechanics change now.**
- M4.6c-1 fixes the readouts.
- M4.6c-2 is a gap-aware self-play pilot (25% archive starts, with a 0% control) with pre-stated criteria (§10).
- Two mechanics questions remain design choices for later, not defects: durable freezes against competent play, and
  how often the economy should create strength gaps.

---

## 1. What was run

| experiment | content | simulated turns | wall clock |
|---|---|---|---|
| E1 exact one-turn ledger | 18 freeze cells × 30 action pairs, 12 zoning cells × 20 pairs, 5 counterfactual cost removals; every chance branch enumerated with ganks | none (exact) | 11 s |
| E3 power-gap self-play | 6 finals × 3 levels × 5 gaps × 2 orientations × 48 games × 300 turns; the M4.5c stage re-run; a position check on 12 policies | 2.7 M | 9 min |
| E5 price of a reset and of HP | 4 natural families × 48 states × 6 finals, K = 12, horizon 50; validation subset at 300; the failing family re-run at 150 | 9.3 M | 18 + 20 min |
| E2 selective threats | 3 families (24 constructed cells, 24 natural HP-edge states, 24 semi-natural gap states per final) × 8 candidates × 2 opponents, horizon 60; validation subsets at 300; the failing family re-run at 150 | 17.7 M | 27 + 48 min |
| E4 learned best responses | 3 targets × 2 start archives × 4,096,000 learner transitions; held-out evaluation against controls | see §6 | see §6 |
| E6 credit assignment | from E2, E5 and E3 records | none | 1 min |

The machine had about 2–3 GB of free memory throughout (other applications open), so stages ran on 3–6 workers with
exact lane-step caches of 3,000 entries. Results do not depend on the cache size.

---

## 2. E1: what the mechanics do in one turn

**Question.** In one turn, with every chance outcome enumerated, what does contesting a freeze cost the contester and
the defender, and what is a stronger champion's Trade worth against each reply? No learning is involved.

### 2.1 Freezes: a guard can hold a freeze, a passive freeze cannot

States: the M4.5b freeze builder. A holds B's wave of 5, 12 or 20 HP at A's −1 with its own 8-HP wave in its tower
cell; B has walked up to its +1. Levels 4, 10 and 15; equal power or A ×1.76.

| A's reply to B's walk-up Shove | cells where the freeze survives the turn | why |
|---|---|---|
| keep (Thin + Hold) | 2 of 18 (level 4, held wave 5) | B's Shove budget plus the held wave's own fire kill A's 8-HP front |
| guard (Thin + Trade) | **9 of 18**: every level-4 cell, and every level-10 cell at ×1.76 | the guard has to step into the held wave to reach B. That wave then fires at the guard instead of A's minions (positional aggro, and retaliation when the hit lands), so only B's own budget hits A's front. With B's budget below 8 HP (5 at level 4, 6 at level 10 with nothing invested) the front survives |
| guard, level 10 at equal power and every level-15 cell | 0 of 9 | B's own budget is 8 HP or more |

**This mechanism was not anticipated in the design review**, which assumed a walk-up Shove always breaks a freeze in
one turn. Minion aggro, introduced in M4.5a, gives the freezer a real defence early and mid game whenever the breaker's
own wave clear is smaller than the freezer's wave.

What a guarded turn costs each side:
- **B** loses 0.21–0.26 bars at ×1.76 and 0.12–0.17 at equal power, against 0.03–0.05 against a passive freeze.
  P(B dies this turn) is 0.008–0.011 at ×1.76.
- **A** loses 0.02–0.33 bars, mostly from the held wave's fire, more at low levels with large held waves.

Who should do what in that one turn (HP priced at E5's measured 1.48 resources per bar; 1.0 and 2.0 give the same
signs):
- **B's best one-turn reply is always the Shove.** Shove − walk-up against the guard is +0.43 to +0.78, and Shove −
  step back is +0.39 to +0.89. The walk-up cell is a full-rate farming cell (collection 0.9, all XP), and the guard's
  single exchange per turn costs B less than the Shove earns.
- **A's best one-turn reply is usually the guard.** Guard − keep against the Shove is −0.13 to +0.46, positive in every
  ×1.76 cell and negative only at equal power with large held waves (5 of 18 cells).

### 2.2 The one-turn threat game favours the stronger side's Trade

States: the M4.5b zoning situations neutral, A's push and B's push, at levels 6 and 10, equal or ×1.76.

| contrast (priced at 1.48 per bar) | range over the 12 cells | at ×1.76 |
|---|---|---|
| A: Trade − Hold, B holds its ground | +0.08 to +0.38 | +0.29 to +0.38 |
| A: Trade − Hold, B plays Safe | −0.10 to +0.14 | +0.07 to +0.14 (neutral, A's push), −0.06 (B's push) |
| B: Safe − Hold, A Trades | −0.25 to +0.11 | positive only in A's push (+0.08, +0.11) |

- A connected Trade at ×1.76 deals 0.19–0.23 bars and takes 0.04–0.12, and costs A 0.09–0.11 of farm (the 0.8
  trading factor).
- Against a Trade, **stepping back rarely pays the weaker side**: Safe saves less HP than the farm it gives up, except
  in A's push at ×1.76.
- So in one turn a stronger champion's threat is credible and profitable. Whatever makes sustained bullying unprofitable
  is not the immediate exchange.

### 2.3 No single trading cost is load-bearing

Each trading cost was removed alone through an existing configuration field (counterfactual one-turn rows, never
rolled out or trained).

| removed | guard − keep: sign changes | Trade − Hold: sign changes | effect on the mean |
|---|---|---|---|
| trader's farm factor (0.8 → 1.0) | 4 of 36 | 0 of 12 | guard +0.26 → +0.37; Trade +0.26 → +0.36 |
| capped wave budget | 0 | 0 | none |
| minion retaliation | 0 | 2 | Trade +0.26 → **+0.12** |
| positional aggro | 0 | 0 | guard +0.26 → **+0.19** |
| Trade gank term | 0 | 0 | +0.01 |

None reaches the pre-registered half of the cells (E1-P2). Two effects run against intuition, and both have the same
cause. Retaliation and positional aggro *help* the trader or the guard in the gold ledger, because the minions that turn
on a champion stop killing that champion's minions. The opponent then farms less. The HP the champion takes (0.04–0.09
bars) is worth less than that farm.

---

## 3. E3: how much a power gap is worth, and through which channels

**Question.** How large is the stronger side's absolute advantage under the trained policy, and where does it come
from? This is the baseline for priority 3: E2 and E4 ask whether anything converts the gap *better* than this.

Design: each final against itself from the neutral zoning start at levels 6, 10 and 14, with A's lead at ×1.24, ×1.5,
×1.76 or ×2.2 (investment only), both orientations, 48 games × 300 turns per cell. The *premium* is (stronger − weaker)
in the gap game minus the same quantity in the equal-power game with the same seeds.

### 3.1 The premium is large and grows with the gap

| gap | G20 | G50 | G100 | G300 | lead over 100 turns (undiscounted) |
|---|---|---|---|---|---|
| ×1.24 | +0.66 ± 0.18 | +1.27 ± 0.30 | +1.72 ± 0.32 | +1.78 ± 0.28 | +2.45 ± 0.46 |
| ×1.5 | +1.41 ± 0.28 | +2.62 ± 0.41 | +3.32 ± 0.49 | +3.42 ± 0.47 | +4.64 ± 0.71 |
| ×1.76 | +2.03 ± 0.27 | +3.64 ± 0.50 | **+4.54 ± 0.51** | +4.71 ± 0.45 | **+6.23 ± 0.67** |
| ×2.2 | +3.43 ± 0.51 | +5.79 ± 0.68 | +7.28 ± 0.70 | +7.96 ± 0.75 | +10.00 ± 0.89 |

At ×1.76 the stronger side gains about 14% of a side's income over 100 turns. Most of it is earned in the first 50
turns, while the weaker side's investment catches up.

### 3.2 Half of the premium is kills and absence; posture denial is a tenth

Decomposition of the undiscounted lead at 100 turns (resources; each column is the change relative to the equal game,
signed so that positive favours the stronger side; the columns sum to the lead):

| gap | lead | stronger: farm | stronger: kill bounty | stronger: plates | weaker: fewer minions | weaker: farm lost while present | weaker: farm lost while away | weaker: bounty it no longer gets | weaker: plates it no longer gets |
|---|---|---|---|---|---|---|---|---|---|
| ×1.5 | +4.64 | +0.62 | +0.97 | +0.78 | +0.06 | +0.34 | +0.94 | +0.36 | +0.56 |
| ×1.76 | +6.23 | +0.71 | +1.30 | +1.06 | +0.05 | +0.55 | +1.31 | +0.52 | +0.73 |
| ×2.2 | +10.00 | +1.34 | +2.27 | +1.48 | +0.12 | +0.59 | +2.14 | +1.01 | +1.05 |

At ×1.76:
- **kills and absence** are +3.13 (50%): the weaker side is away 1.79 more turns, dies 0.65 more times and recalls
  0.56 fewer times per 100 turns, so deaths replace resets;
- **plates** are +1.79 (29%);
- **farm and posture** are +1.26 (20%), of which posture denial (farm lost while present) is +0.55 (9%);
- the weaker side also loses 10 XP more per 100 turns.

The design review's H4 ("the stronger side's lead comes from absence, not farm") holds: (d) + (e) = 3.13 against
(a) + (c) = 1.26.

### 3.3 Two measurement artefacts in M4.5c's zoning readouts

1. **The "stronger side presses forward (+2.0)" readout includes the Ward node.** M4.4's position metric averages
   every cell, and the off-lane Ward node is stored as 99. A mean above +2 is impossible on lane cells. Recomputed on
   M4.5c's own stage with new seeds:

   | policies | stronger, M4.5c's metric | stronger, lane cells only | stronger at the Ward node | weaker, lane cells only |
   |---|---|---|---|---|
   | 16.384M finals | +0.87 to +2.49 | **−0.15 to −0.27** | 1.0–2.7% of turns | −0.91 to −1.07 |
   | 8.192M finals | +0.52 to +2.50 | −0.07 to −0.22 | 0.6–2.7% | −0.97 to −1.08 |

   The stronger side does not press forward; it stands slightly behind the lane cell. The weaker side stands one cell
   further back. M4.5c's "+1.30 → +2.03" between budgets measured more Warding, not more pressure.
2. **"Denial when near" counts only posture.** It measures the weaker side's uncollected gold while the stronger side is
   near, which E3 puts at 9% of the premium. In E3 the same quantity is even slightly negative (−0.04 per turn),
   because being catchable selects turns where the weaker side is in the fight and farming. It cannot see kills,
   absence or plates.

---

## 4. E5: the price of a reset and of HP

**Question.** Is a reset subsidized (does recalling to invest gold make HP nearly free), and what is one bar of HP worth
under learned play? Natural states of each final's self-play, K = 12 paired rollouts, Q50 primary.

| family (48 states per final) | contrast | ΔQ50 | at 150 turns |
|---|---|---|---|
| Recall legal, HP ≥ 90%, ≥ 2 resources unspent | Recall − stay | **−0.40 ± 0.17** (5− of 6) | −0.41 ± 0.16 |
| same | Recall − stay with the gold already invested | −0.40 ± 0.07 (6−) | −0.43 ± 0.10 |
| same | **shopping component** (the difference of the two) | **−0.00 ± 0.12** | |
| same | pre-invested state − original, both played by the policy | −0.01 ± 0.14 | |
| Recall legal, HP 40–60% | Recall − stay | **+0.19 ± 0.11** | |
| same | Recall − policy | −0.17 ± 0.16 | |

| family | HP lowered by 0.3 bar: ΔQ50 | **price per bar** | extra Recalls in 20 turns | extra turns away in 50 |
|---|---|---|---|---|
| natural, levels ≤ 10 | −0.37 ± 0.13 | **1.23 ± 0.44** | +0.28 | +0.70 |
| natural, levels ≥ 11 | −0.44 ± 0.14 | **1.48 ± 0.47** | +0.24 | +0.86 |

- **The reset is not subsidized.** Investing on return is worth nothing measurable over 50–150 turns: the marginal
  effect of a few resources of investment at typical investment levels is tiny. A full-HP Recall costs about 0.4
  resources, about one turn of income. That is "cheap" by the pre-registered −0.5 threshold but not "nearly free"
  (−0.2). The design review's reset-subsidy hypothesis is **not supported**.
- **HP is not cheap.** A bar is worth 1.2–1.5 resources under learned play, the range M4.5b measured with scripted play.
  It is paid mainly through an earlier Recall and the turns away it causes.
- At half HP a Recall beats staying (+0.19). The policy's own mixed choice is slightly better than forcing either.
- Validation: the full-HP family failed the pooled rule (bias +0.18). The pre-registered 150-turn re-run changed
  nothing (mean change −0.03, slope 0.96, the same conclusions). The other three families passed.

---

## 5. E2: does any selective threat convert a power gap better than the policy?

**Question.** With the wave left to the policy, does forcing a threat behaviour (Trade when the stronger side can catch
the weaker without a dive) earn more than the policy's own play from the same state? This is the decisive test of
"conversion over the trained policy" without learning.

Plans (20 turns, then the policy; horizon 60):
- the behaviour-only plans force Hold, the hunt rule, an edge rule, a no-dive bully, or the freeze controller with the
  hunt rule;
- the scripted plans are M4.5c's `g_bully` (Last Hit wave + no-dive bully) and M4.4c's Farm (Last Hit + Hold).

Every plan hands the turn to the policy at or below 40% HP, so none is penalized for being unable to Recall.

### 5.1 Against learned play no threat plan beats the policy

ΔG60, constructed zoning cells (6 cells per gap: three situations × levels 6 and 10), opponent = the policy:

| gap | policy's absolute G60 | threaten − policy | edge − policy | bully (no dive) − policy | freeze + threat − policy | Hold only − policy |
|---|---|---|---|---|---|---|
| equal | +0.14 | −0.85 ± 0.35 | −0.53 ± 0.18 | −0.94 ± 0.43 | −1.61 ± 0.25 | −0.57 ± 0.21 |
| ×1.5 | +3.22 | −0.57 ± 0.23 | −0.54 ± 0.16 | −0.58 ± 0.23 | −0.83 ± 0.23 | −0.92 ± 0.33 |
| **×1.76** | **+4.46** | **−0.40 ± 0.58** (1− of 6) | −0.29 ± 0.37 | −0.32 ± 0.57 | −0.56 ± 0.27 | −1.21 ± 0.52 (5−) |
| ×2.2 | +6.44 | +0.06 ± 0.33 | +0.01 ± 0.39 | +0.06 ± 0.31 | −0.00 ± 0.49 | −1.38 ± 0.51 (6−) |

At 150 turns (the pre-registered re-run after the constructed family failed validation; mean change +0.07, slope 0.94)
threaten − policy at ×1.76 is −0.38 ± 0.59, and at ×2.2 +0.14 ± 0.44.

| state family | threaten − policy | bully − policy | Hold only − policy | threaten − Hold only |
|---|---|---|---|---|
| semi-natural ×1.76 gap states (from E3's games; ΔG60) | −0.75 ± 0.20 (6−) | −0.74 ± 0.28 (4−) | −1.08 ± 0.17 (6−) | +0.33 ± 0.10 |
| natural HP-edge states (ΔQ60) | −0.91 ± 0.17 (6−) | −1.02 ± 0.23 (6−) | −0.73 ± 0.24 (5−) | −0.18 ± 0.31 |

- **E2-P1 (primary): no conversion gap.** At ×1.76 the selective threat is −0.40 ± 0.58 against the policy, far from
  the pre-registered +0.25. Across the 12 contrasts of the four selective plans in the three families, none is positive
  and significant (Holm).
- **E2-P3: the threat is valuable, and the policy already has it.** Threaten − Hold only is −0.28 (equal), +0.35
  (×1.5), **+0.80 ± 0.29** (×1.76) and **+1.45 ± 0.36** (×2.2, 6 of 6 finals): a clean gap gradient, +0.89 and +1.54
  at 150 turns. Forcing Hold costs the policy 0.9–1.4 at the gaps. The policy's own play already earns what the threat
  earns over passive play, and more.
- **How the policy's conversion works** (ledger at 50 turns, ×1.76, Hold only − policy): with Hold forced, the weaker
  side is away 1.59 fewer turns and dies 0.34 fewer times, and the stronger side connects 3.4 fewer trades. The policy's
  advantage over passive play is its trades, and their payoff is the weaker side's absence.
- **Forcing more trades than the policy chooses** (threaten − policy) gives the weaker side 1.3 more turns away, but
  costs the stronger side 0.18 bars, 0.13 deaths and 0.31 of its own gold. The net is −0.4, not significant.
- **Against the scripted conceder `g_wary`** every behaviour plan is within ±0.2 of the policy on the constructed cells
  and −0.4 to −0.7 on the natural and semi-natural states. The policy already extracts what the plans extract from a
  conceding opponent.

### 5.2 E2-P4: M4.5c's bully test and M4.4c's Farm baseline were confounded by their wave

| contrast at ×1.76 (constructed, opponent policy) | 60 turns | 150 turns | semi-natural (60) | natural (60) |
|---|---|---|---|---|
| scripted `g_bully` (Last Hit wave) − no-dive bully with the policy's wave | −0.50 ± 0.45 | −0.26 ± 0.47 | −0.78 ± 0.18 (6−) | −0.64 ± 0.27 |
| scripted Farm (Last Hit + Hold) − Hold only with the policy's wave | **−2.11 ± 0.24** (6−) | | −2.50 ± 0.17 | −0.43 ± 0.09 |

- M4.5c's "bully ×10 costs against learned play" bundled a forced Last Hit wave into the bully: 0.3–0.8 of its loss is
  the wave.
- M4.4c's "Threaten − Farm +1.13" compared a behaviour-only plan with a wave-and-behaviour plan. Most of it is the 2.1
  cost of Farm's forced Last Hit wave.
- **The one case where M4.5c found bullying profitable** (bully against `g_wary` in B's push, +1.28) is reproduced:
  +1.54 and +1.63 at levels 10 and 6. But the same Trade rule with the policy's own wave earns only −0.04 to +0.17
  there. The gain is the forced Last Hit wave holding B's wave next to A's tower against an opponent that steps back: a
  freeze effect, not a threat effect.

### 5.3 Validation

The natural and semi-natural families passed the pooled rule. The constructed family failed (bias +0.19, slope 0.97),
and its 150-turn re-run left every conclusion unchanged.

---

## 6. E4: learned best responses from gap and freeze starts

**Question.** Scripted plans (E2) can only test the rules someone thought of. Can PPO itself learn to convert a power
gap, or to hold a freeze, better than the trained policy, if it is trained on those states? And if it can, is that a
better strategy in general, or an exploit of one network?

**Protocol** (pre-registered, plus one dated amendment fixing the pooling before any E4 result was read):
- Targets: finals s0, s2 and s4. For each, two attackers are warm-started from the target and trained by PPO against
  the frozen target for 4,096,000 learner transitions: the M4.5c attacker budget and settings.
- The only change from M4.5c: every training game starts from a start archive and lasts 150 turns, and the learner
  always plays the archive's focal side.
- The gap archive holds 27 constructed zoning states at ×1.5 / ×1.76 / ×2.2 and levels 6 / 10 / 14 (repeated to
  half of the archive), plus 576 semi-natural states from the target's E3 training-split games.
- The freeze archive holds 18 constructed freeze states plus 240 natural freeze states of the target's self-play.
- Evaluation is on held-out starts (the constructed cells with new seeds, and states from held-out games), the target
  in seat B. Seat A is played by the target itself (the baseline), the attacker, or the same target's M4.5c s0-attacker
  (the control for general exploitability), all on the same chance and sampling streams.

**Training health**: 2,000 updates per run in 70 minutes at about 970 learner transitions/s per run (three
concurrent); entropy 1.7 → 1.6; maximum KL ≤ 0.010; finite losses; 27,322–27,325 archive starts per run, of which
about half were mirrored for seat B, and none from s0.

### 6.1 Power gaps: a learned specialist converts the gap 50% better

| G100 (resources, discounted) | pooled over targets | s0 | s2 | s4 |
|---|---|---|---|---|
| target's own play as the stronger side (absolute) | +3.75 to +4.42 | +3.75 | +4.42 | +4.22 |
| **gap attacker − target** | **+2.11 ± 0.62** (3/3) | +2.34 ± 0.40 | +1.84 ± 0.43 | +2.14 ± 0.35 |
| s0-attacker − target (control) | +0.65 ± 0.85 (2/3) | +0.80 ± 0.29 | +0.25 ± 0.37 | +0.88 ± 0.29 |
| **gap attacker − s0-attacker** | **+1.46 ± 0.44** (3/3) | +1.53 ± 0.40 | +1.59 ± 0.37 | +1.26 ± 0.38 |
| gap attacker − target, held-out semi-natural states only | +1.83 ± 0.34 (3/3) | +1.96 | +1.68 | +1.84 |
| gap attacker − target, the same cells at equal power | +0.77 ± 0.05 (1/3) | +0.77 | +0.75 | +0.79 |
| gap attacker − target, from s0 (W = 460, 64 paired seeds) | ≈ 0 | +0.11 ± 0.62 | +0.04 ± 0.65 | −0.06 ± 0.84 |

(± are 95% intervals; "3/3" = targets with an interval excluding 0.)

- **E4-P1 meets the pre-registered "distribution binding" criterion.** Both conditions hold: gap attacker − target is
  +2.11 ≥ +0.5 with 3 of 3 targets significant, and gap attacker − s0-attacker is +1.46 ≥ +0.3. The result holds at
  G20 / G50 / G150 (+0.81 / +1.48 / +2.44) and on held-out states from games the attackers never saw.
- **The gain is mostly gap-specific.** On the same cells at equal power the attacker gains +0.77, about a third, and
  matches the s0-attacker there (−0.06). From s0 it is no better than its target. It learned to play power gaps, not a
  general exploit.
- **It is not an exploit of one network** (exploratory transfer test, added after this result to tell the two apart).
  Each gap attacker played the five finals it never trained against, from the constructed gap cells, and was compared
  with each final's own strong-side play. It beat them by **+2.08 ± 0.58, +1.78 ± 0.76 and +2.34 ± 0.61**, significant
  against **15 of 15** opponents. Every M4.5c final shares the same conversion deficit.

**How it converts** (100 turns, attacker − target, pooled; the target's own level in brackets):

| channel | stronger side (attacker) | weaker side (opponent) |
|---|---|---|
| Trade turns | **+13.6** (25.2) | −1.4 (20.2) |
| Yield / Safe turns | −4.7 / −2.8 (19.3 / 16.0) | Safe +3.0 (18.2) |
| trade damage dealt (bars) | +2.24 (2.96) | |
| Recall attempts interrupted | | **+3.06** (3.56) |
| deaths | +0.37 (0.43) | **+1.31** (1.45) |
| turns away | +0.0 | **+4.4** (13.8) |
| farm | −2.64 (38.7) | farm lost +3.74, of which while away +2.67 |
| gold earned | **−0.07** (43.5) | **−3.19** (37.8) |

- The attacker earns the same gold as the target would; **its whole gain is denial**.
- It Trades about 50% more, retreats less, and above all catches the weaker side's Recall attempts, about twice as
  often as the target does. Those become kills: the weaker side dies about twice as often and spends 4.4 more turns
  away per 100.
- It pays for this with its own farm (−2.6) and a few deaths (+0.37), and recovers both in bounty.
- This is threat-based lane control in the League sense: the weaker side cannot both farm and safely reset. It emerges
  from the existing rules once PPO trains on power-gap states.
- E2's scripted "Trade whenever catchable" rules did not find it, because the value lies in *which* trades and in
  punishing Recall attempts. The trained finals do not find it because they almost never meet a power gap.

### 6.2 Freezes

| freeze attacker | G50 (primary) | G100 |
|---|---|---|
| target's own play as the freezer (absolute, pooled states) | +1.44 to +1.53 | |
| **freeze attacker − target** | **+0.69 ± 0.61** (3/3) | +1.07 ± 0.85 (3/3) |
| s0-attacker − target (control) | +0.50 ± 0.85 (2/3) | +0.86 ± 1.37 (2/3) |
| **freeze attacker − s0-attacker** | **+0.19 ± 0.25** (0/3) | +0.21 ± 0.54 (1/3) |
| on held-out natural freeze states: attacker − s0-attacker | +0.02 ± 0.05 | −0.03 ± 0.38 |
| on the constructed freeze cells: attacker − s0-attacker | +0.63 ± 1.01 (1/3) | +0.85 ± 1.06 (3/3) |
| from s0 against the target (W = 460) | +0.63 / +0.55 / +1.83 (s0-attackers: +0.79 / +0.50 / +1.82) | |

- **E4-P2 is "weak evidence" by the pre-registered rule.** The attacker beats its target (+0.69), but no more than the
  general s0-attacker does from the same starts (+0.19 ± 0.25, 0 of 3).
- **What the freeze attacker learned is general exploitation, not a better freeze.** From s0 it exploits its target
  exactly as much as an s0-attacker. It holds the freeze only 0.5 turns longer than the target (3.3 turns per 50), and
  the s0-attacker holds it 0.7 longer. On natural freeze states from held-out games it adds nothing over the
  s0-attacker.
- **What transfers is the gap skill again** (exploratory). Against the five finals it never trained against, on the
  constructed freeze cells, it gains +0.78 to +1.05 (G100; 9 of 15 opponents significant). By gap, that is **+1.76 to
  +1.85 in the ×1.76 cells and −0.30 to +0.32 at equal power**, none significant. Half of its archive held ×1.76 freeze
  states, and the transferable skill it took from them is power-gap conversion through the same channel as §6.1:
  connected trades +8.4, the weaker side's deaths +0.6 and turns away +2.6 per 100.
- This is "not found at this budget", not proof that no better freezer exists (the user's priority 2). It does agree
  with every other freeze result: E1's one-turn ledger, the M4.5c battery's "a freeze is worth what the opponent
  concedes", and M4.5b's freeze premium of about 0 at equal power.

---

## 7. E6: how visible these decisions are to the learner

**Signal and noise.** One forced threat turn changes the 60-turn lead by 0.03–0.07 resources, while one paired 60-turn
rollout against learned play has a standard deviation of about 3.5 (about 1.4 against the scripted conceder).

| family, opponent policy | threaten − policy | |effect| / SD of one paired rollout | paired rollouts one state needs for 2 SE |
|---|---|---|---|
| constructed cells | −0.44 | 0.12 | about 260 |
| semi-natural gap states | −0.75 | 0.20 | about 100 |
| natural HP-edge states | −0.92 | 0.27 | about 60 |

**The critic does not understand power gaps** (exploratory, not pre-registered). V(s) is compared with the realized
100-turn return plus the bootstrapped tail in E3's games:

| gap | bias for the stronger side | bias for the weaker side | explained variance |
|---|---|---|---|
| equal | +0.02 (either side) | | 0.07–0.13 |
| ×1.24 | −0.60 to −0.73 | +0.60 to +0.71 | 0.07–0.13 |
| ×1.76 | **−1.31 to −1.62** | +1.25 to +1.52 | 0.08–0.14 |
| ×2.2 | −2.55 to −3.08 | +2.37 to +2.87 | 0.09–0.14 |

- The bias grows in proportion to the gap and is nearly symmetric. The critic values a ×1.76 state much as it values an
  equal one.
- The investment entries are in the observation. The value function simply never learned their worth, because natural
  power gaps are about 0.1% of training decisions.
- At equal power the critic is unbiased. On Trade-connected turns it is slightly more pessimistic than elsewhere (−0.04
  against +0.02).

---

## 8. Synthesis: what the concerns turn out to be

### 8.1 Freezing and breaking freezes (the user's question 1)

| question | answer |
|---|---|
| Why is breaking a freeze cheap? | A *passive* freeze breaks in one turn: the breaker's Shove plus the held wave's fire kill the freezer's 8-HP wave (16 of 18 cells). The walk-up cell is a full-rate farming cell (0.9 collection, all XP), so contesting costs only HP. |
| Does the system already punish contesting? | **Yes, under conditions.** A guarding freezer must step into the held wave. That wave then fires at the guard instead of the freezer's minions, so the freeze survives a walk-up Shove whenever the breaker's own wave clear is below the freezer's wave: every level-4 cell, and level 10 at ×1.76 (9 of 18 cells). The breaker then pays 0.21–0.26 bars a turn at ×1.76. M4.5b measured guarding at +0.8 to +1.0 per 100 turns over passive freezing in those states. |
| Why is breaking still the right reply? | In one turn the Shove at the walk-up cell beats walking up by +0.43 to +0.78 and stepping back by +0.39 to +0.89, even against the guard: one exchange a turn costs less than the farm. From level 10 at equal power, and at level 15, the breaker's own wave clear (≥ 8 HP) breaks even a guarded freeze. |
| Could a learned freezer do better? | Not found. A freeze specialist trained on freeze states gains no more than a general exploiter, and at equal power nothing that transfers to other finals (§6.2). |
| Rules, parameters or learning? | **The rules and their scale.** The contest is decided by wave-clear capacity against a fixed 8-HP wave, plus the absence of any farming penalty at the walk-up cell. Capacity grows with level and investment; the freezer's wave does not. No single trading cost is load-bearing (E1-P2), and learning did not find a better freeze. |

### 8.2 Threat-based zoning and champion strength (question 2)

| question | answer |
|---|---|
| Does combat strength become lane control? | **Yes, more than M4.5c reported.** At ×1.76 the stronger side's lead is +4.5 G100 (+6.2 resources undiscounted per 100 turns, about 14% of income). Half of it is kills and the weaker side's absence, 29% plates, 9% posture denial (§3). |
| Is the threat of a winning trade represented? | Yes. In one turn, a ×1.76 champion's Trade beats Hold by +0.29 to +0.38 at the measured HP price, and stepping back rarely pays the weaker side (§2.2). Against learned play, the selective threat beats passive Hold by +0.80 (×1.76) and +1.45 (×2.2) (§5.1). |
| Do trading costs outweigh strength? | No. No cost is load-bearing (§2.3). HP is worth 1.2–1.5 resources a bar, and resets are not subsidized (§4). A learned specialist converts a gap 50% better than the policy, despite paying those costs, and gets its gain entirely from denial (§6.1). |
| Could the system produce zoning that the agents don't learn? | **Yes, that is the finding.** The trained finals capture part of the gap's value (they already beat every scripted threat rule, §5.1). A specialist trained from gap states captures +2.1 more, and that better play works against every final (15 of 15). |
| Do the two problems share a cause? | **No, contrary to the design review.** Zoning is held back by measurement and training distribution; freezes by the mechanics of the contest. The design review's shared cause ("contesting costs one turn of cheap exposure") explains freezes, not zoning: for zoning, the exposure is not cheap (1.2–1.5 per bar), and conversion through lethal pressure works. |

### 8.3 Training distribution versus simulator mechanics (question 3)

- **Power gaps are out of the training distribution.** In s0 self-play a gap of ×1.25 or more occurs on 0.07–0.37% of
  turns, almost only in the first 50, and never ×1.5.
- **The critics show it.** They are unbiased at equal power but undervalue a ×1.76 lead by 1.3–1.6 resources (§7).
- **Training on those states teaches conversion.** E4-P1 meets its criterion, and the better play transfers to every
  final.
- **Diverse starts would therefore change the policy without any mechanics change.** That is the direct answer to
  question 3 for zoning. For freezes, the corresponding test (E4-P2) found nothing beyond general exploitation.

**How the two failures were told apart** (the protocol of the pre-registration, as run):
- the exact and scripted screens (E1, E2) ask whether any strategy converts;
- the learned policies' own counterfactuals (E2, against the policy) ask whether the policy converts;
- learned best responses trained on the relevant starts, with an s0-trained control and a transfer test (E4), ask
  whether a policy can learn to convert.

Zoning passed all three; freezing passed none beyond the guard's one-turn value.

### 8.4 A structural property behind zoning's rarity (exploratory)

Natural self-play almost never produces the strength differences that zoning needs:
- **Gold leads do not become power.** A cumulative lead of 10–20 resources (25–45 turns of income) comes with a power
  ratio of about ×1.01. Investment saturates (half its gain by 48 resources), levels converge (XP is mostly shared and
  capped at 18 by about turn 145), and the trailing side keeps investing.
- **Gaps exist only early.** Ratios of ×1.25 or more appear almost only in the first 50 turns (1–2.5% of those turns)
  and never after turn 150.

So even a policy that converts gaps well would rarely have one to convert. This is a design property of the economy
(no "snowball" from leads into strength, and no champion matchups), not a defect. It bounds how often threat-based
zoning can appear in natural play, whatever the training.

### 8.5 Measurement limitations found in earlier milestones (question 4)

| earlier result | what was wrong | corrected reading |
|---|---|---|
| M4.5c: "the stronger side presses forward (position +2.0), +1.30 → +2.03 between budgets" | the M4.4 position metric averages the off-lane Ward node, stored as 99 | lane-only: the stronger side stands at −0.15 to −0.36, the weaker one cell back; the change between budgets was more Warding |
| M4.5c: "the stronger side denies only 0.014 gold per turn" | posture denial only (gold uncollected while near) | posture is 9% of the premium; kills, absence and plates are 79% |
| M4.5c: "sustained bullying costs −0.2 to −1.0 against learned play; +1.28 only against `g_wary`" | the bully plan also forced a Last Hit wave for 10 turns | the wave alone costs 0.3–0.8; the +1.28 is a freeze effect of that wave, not the threat |
| M4.4c: "Threaten − Farm +1.13" | Farm forced a Last Hit wave too | the forced wave costs 2.1; with the policy's wave the threat's value is +0.80 (×1.76) |
| the validation rule (M4.5c §10 and here) | the pooled bias criterion (0.15 floor) flagged two plan families (biases +0.18 and +0.19) | both 150-turn re-runs changed nothing (mean change −0.03 and +0.07, slopes 0.96 and 0.94): a criterion problem, not a horizon problem |

### 8.6 The design review's hypotheses, scored

| hypothesis | verdict |
|---|---|
| H1 scale: the contest resolves in one turn, combat is small, recovery cheap | **freezes: partly** (passive freezes and late game yes; a guard holds early and mid game). **Zoning: rejected** (the trade is profitable, HP costs 1.2–1.5 a bar, a learned specialist converts strongly at the current scale) |
| H2 the asymmetric cost of aggression sets a ceiling | **rejected**: no cost is load-bearing; retaliation even helps the trader in the gold ledger |
| H3 geometry | **partly**: posture denial is small and the walk-up cell is a full-rate farming cell |
| H4 the lead comes from absence, not farm | **confirmed** |
| H5 training distribution | **confirmed for zoning; not found for freezes** |
| H6 policies leave value against conceders | **not confirmed for zoning** (no plan beats the policy against `g_wary`); M4.5c's freeze result against a back-off opponent stands |
| H7 measurement hole in M4.5c's zoning verdict | **confirmed**, larger than expected (four artefacts) |
| the reset subsidy (design review §10) | **rejected**: investing on return is worth 0.00 ± 0.12 |

My written expectations in the design review were wrong on three of four counts. The specialist converted far more
than "somewhat", the reset is not close to free, and for zoning the binding constraint is the distribution, not the
scale.

### 8.7 Classification (the user's five categories)

| category | freezes | zoning |
|---|---|---|
| structural limitation of the mechanics | **primary**: the contest is decided by wave-clear capacity against a fixed 8-HP wave; the walk-up cell has no farming penalty | a structural property bounds its *frequency*: the economy compresses gold leads, so natural power gaps are rare (§8.4) |
| parameter scale / balance | secondary: the level at which the guard stops holding is a capacity-scale property | **not supported**: no trading cost is load-bearing; the HP price is substantial |
| training distribution / strategy discovery | not found (E4-P2 weak) | **primary**: gap states are 0.1–0.4% of training; training on them gives +2.1 that transfers to all finals |
| evaluation / measurement | small (the freeze conclusions of M4.5c stand) | **primary for the earlier verdict**: four artefacts hid the conversion |

**In one sentence:** the simulator already supports threat-based zoning, and PPO discovers it when trained on states
where it matters. M4.5c missed it through measurement and a training distribution that almost never contains such
states. Freezes are cheap to break for structural reasons that no learner here overcame.

## 9. What remains uncertain

- **Three targets, one budget.** The E4 attackers ran on s0, s2 and s4 at 4,096,000 transitions. The gap result is
  strong (3/3, transfer 15/15); the freeze null is "not found", not "impossible".
- **Constructed gaps are investment-only.** Natural early-game gaps come mostly from levels; the semi-natural states
  (E3's games) cover the dynamics after an investment gap, not a level gap.
- **The specialists play only the strong side.** Whether a self-play policy can learn both the conversion and the
  defence against it, without losing strength from s0, is untested: that is M4.6c's question.
- **Whether better gap play matters for natural play.** The gap specialists are no better than their targets from s0
  (≈ 0), because natural gaps are rare. Training on gaps improves play in states that the natural game seldom reaches.
- **E1 is one turn.** It measures the immediate exchange, not the continuation, which E2 and E4 measure.
- **E2's plan library is small.** That a learned specialist found +2.1 where no scripted rule found anything is a
  reminder that scripted screens bound value from below only.
- **The critic-by-gap analysis is exploratory.** Its bias measure uses a 100-turn bootstrapped target, which carries
  chance noise of about 3 resources per turn of data. The mean bias is well determined (hundreds of thousands of
  turns), but the explained variance is low everywhere.

## 10. Recommendation and proposed next steps (not implemented)

**Recommendation: training changes (M4.6c) first, with the measurement fixes; no mechanics change now.**
- The evidence does not support changing the trading economy. Lever L1 of the design review (two exchanges per turn) is
  **not** supported: trading scale is not what limits zoning.
- Two design questions for freezes and for the frequency of strength gaps are well posed, but are the user's choice,
  not defects (§10.3).

### 10.1 M4.6c-1: measurement fixes (no training; about 1 hour)

1. **Positions on lane cells only**, the Ward share separate, in the M4.4 / M4.5c behaviour metrics.
2. **Denial by channel** (farm lost while present and while away, bounty, plates, deaths) in the health readouts and the
   battery, as E3 and E4 report it.
3. **Behaviour-only plan families** in the battery (the wave left to the policy), with the scripted versions kept as
   labelled references.
4. **The validation rule.** Keep the pooled per-family check, but replace the absolute 0.15 bias floor by a criterion
   relative to the family's effect sizes, or apply exception 3 only when the 150-turn re-run itself changes a sign.

*Decision criterion:* M4.5c's zoning readouts, recomputed with 1–2, should show the conversion that E3 measured
directly. If they do not, the fix is wrong.

### 10.2 M4.6c-2: gap-aware self-play pilot (training; about 3–4 hours on this machine)

Design, to be pre-registered:
- **Treatment:** continue three 16.384M finals (s1, s3, s5: not used as E4 targets) with shared-policy self-play for
  4,096,000 transitions each, 25% of training games from a start archive. The archive holds the E4 gap states (both
  orientations, so that one network learns both the conversion and the defence) plus the freeze archive.
- **Control:** the same continuation with 0% archive, to separate gap starts from more training.
- **Evaluation:** on held-out gap starts and from s0 as before, plus the M4.6a gap specialists as fixed opponents.

Decision criteria (all on held-out starts, paired):

| criterion | threshold | why |
|---|---|---|
| strong-side conversion: new policy − its own M4.5c final, both against an M4.5c final in seat B | ≥ +1.0 G100 (half of the specialist's +2.1), 2 of 3 seeds | the general policy acquires the skill |
| defence: the M4.6a gap specialist's advantage against the new policy | ≤ +1.0 G100 (from +2.1) | the weak side learns to survive it |
| no regression: new policy against the control continuation, from s0, W = 1000 | ≥ −0.2 | gap starts must not cost natural play |
| critic: bias for the stronger side at ×1.76 | ≤ 0.5 (from 1.3–1.6) | the value function learns what a gap is worth |
| control: the 0%-archive continuation | must **not** meet the first criterion | otherwise the gain is just more training |

If all pass, a fresh campaign with a gap share in the start distribution becomes the candidate training design for
M5. If the conversion criterion fails while the specialists succeed, the limitation is shared-policy self-play itself
(the two roles interfere), which is an M5 question.

### 10.3 M4.6b design questions (deferred; only if the user wants these behaviours in natural play)

1. **Freeze contest.** Should contesting a guarded freeze carry a cost beyond one exchange per turn, or should
   wave-clear capacity grow less steeply relative to a wave late in the game? A durable freeze against competent play
   needs one of the two. A screen would use E1's ledger and E2's harness.
2. **Strength differences in natural play.** Should gold leads translate into power more strongly (the investment
   curve), or should a matchup asymmetry exist (a per-champion stat multiplier, not a kit)? Threat-based zoning works
   when a gap exists, but the economy almost never creates one (§8.4). A screen would use E3's natural-gap measurement
   with E2 and E4.

Neither should be screened before M4.6c-2: if gap-aware training already produces the conversion, question 2 becomes a
choice about how often the game should present it, not whether the agents can use it.

**Stop.** Nothing beyond M4.6a was implemented. The next experiments above await the user's decision.

---

## 11. Reproduction, deviations and artifacts

**Commands** (each stage resumable; `--workers` as memory allows):

```bash
P="python -m lanegame.experiments.rl.m46a"
$P prereg smoke                                     # freeze the pre-registration; pipeline checks
$P e1 e3 e5 e2 --workers 6                          # E1, E3, E5, E2
$P e5_extend e2_extend --workers 3                  # exception 3 re-runs
$P e3_positions e3_natural_gaps --workers 2         # exploratory measurement checks
$P e4_archives e4_smoke --workers 2
$P e4_train --kinds gap --workers 3 && $P e4_train --kinds freeze --workers 3
$P e4_eval --workers 3 && $P e4_reanalyse && $P e4_transfer --workers 6
$P e6 report
python -m pytest tests/test_m46a.py
```

**Deviations, corrections and additions** (in order):

| item | what happened | effect |
|---|---|---|
| E5-full predicate and the E1-P2 price | the pre-registration was amended before it was frozen: E5-full also needs ≥ 2 resources unspent (a Recall right after a return is degenerate), and E1-P2 is evaluated at 1.5 per bar | before any run |
| E4 pooling | a dated amendment, before any E4 result, fixed the pooling of constructed and held-out states | as recorded in `preregistration.json` |
| exception 3 | E5 recall_full and E2 constructed failed the pooled validation; both were re-run at 150 turns as pre-registered | conclusions unchanged |
| E3 position readout | my first E3 code also averaged the Ward node into positions (the same artefact as M4.5c's). It was corrected and E3 re-run: 8,640 of 8,640 games identical in everything else | only E3-P3's position column changed |
| E3 re-run interrupted | paused once to free memory for the transfer test, then resumed (per-cell files) | none |
| exploratory additions (labelled) | the position check, the natural-gap measurement, the critic by gap, both transfer tests, the freeze split by gap, the attacker profile | not pre-registered; reported as exploratory |
| smoke-test checks | two pipeline checks were mis-specified (a float tolerance, the attacker update count) and fixed | pipeline only |
| an interim message | I first counted 8 guarded freezes that survive; the correct count is 9 of 18 | report uses 9 |
| library change | `lanegame/rl/train.py`: opt-in `collector_cls` argument of `train_ppo`, default unchanged (bit-identical, tested) | none for any earlier run |

**Artifacts.**
- Code: `lanegame/experiments/rl/m46a/` (common, policies, states, engine, families, e1, e2, e3, e4, e5, e6, report,
  smoke, CLI) and `tests/test_m46a.py` (8 tests).
- Results (small, generated, in `results/m46a/`): `preregistration.json`, `e1_exact.json`, `e2_summary.json`,
  `e3_summary.json`, `e3_positions.json`, `e3_natural_gaps.json`, `e3_determinism.json`, `e4_archives.json`,
  `e4_training.json`, `e4_summary.json`, `e5_summary.json`, `e6_summary.json`, `report.md`, stage logs.
- Raw jobs, archives and the six attackers (git-ignored): `results/rl/checkpoints/m46a/`.
- Seeds: 96,000–96,999.
- The mathematical reference is unchanged: no rule, parameter or objective changed.
