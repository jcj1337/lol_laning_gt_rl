# M4.5b — Integrated calibration and strategic evaluation

*2026-10-14. Branch `main` on 8dc5acf (M4.5a revision 2); nothing staged or committed. **No PPO training; no change to
the reward, the action space, the observation, any earlier preset or any earlier result.** Code:
`lanegame/experiments/m45b/` (new), one opt-in rule value in `lanegame/champions_m4.py`, the candidate preset in
`lanegame/config.py`. Generated tables: [`results/m45b/report.md`](../../../results/m45b/report.md) (every number below
that is not computed in the text comes from there); raw results `results/m45b/*.json(.gz)`; viewer recordings
`results/m45b/trajectories/`. Tests: `tests/test_m45b_calibration.py`, `tests/test_m45b_candidate.py` (new).
Mathematical reference: [§15](../laning_game_mathematical_reference.md#15-experimental-the-m45b-calibrated-candidate-m45_calibrated).*

---

## 0. Summary

**Question.** M4.5a made `m45_unified()` correct and reproducible; are its combined strategic and economic incentives
sensible, and which provisional numbers should change?

**What was run.** A paired, common-random-number simulation study of scripted contingent strategies (≈ 60, from
passive farming to freeze guards, Shove-then-Recall, resets and Ward rules) through the real resolver:
- exact one-turn enumerations;
- multi-turn counterfactuals from constructed and natural states;
- restricted-game equilibria with seed bootstraps;
- a joint factorial over the provisional parameters;
- a holdout evaluation on seeds and states fixed before calibration.

About 220 million simulated turns; no PPO training.

**What we found.**

1. **A pre-existing mechanics defect, now corrected.** In mutual Shoves the farm-spot probe predicts both fronts
   dying, so both followers walk back to their next packets, out of reach of the fight: 38–50% of the farm of
   Shove-mirror play died out of reach (21.5% of all scripted play). The defect predates M4.5a (identical on the
   historical aggro preset). It distorted the results most where the scripted equilibria live, and it had inflated
   the apparent value of freezing against a shoving opponent. The narrow correction
   `farm_spot_rule = "last_front"` cuts it to 5–7% (§8).
2. **Retaliation depended on the tick split.** With delay 1, a damaged 8-HP wave never retaliated, and an early
   trade into a 24-HP stack was HP-positive. `minion_retaliation_delay = 0` passes all six pre-registered trade
   criteria (§4, §9).
3. **Freezing is a real but conditional option.** On the corrected mechanics it pays mid- and late game, at equal
   power with a large held wave and on the holdout at levels 13–16. A strong freezer punishes a walk-up Shove (+0.8 to
   +1.0), but not at equal power. Passive freezes are broken at once. The weak side's best reply varies (Shove with a
   reset, wary Shove, Yield, walk-up Last Hit), and the reset sequence of the brief appears. After a reset, keeping the
   guard beats recalling (§3).
4. **Vision is situational.** It pays for exposed shovers and against chasers, not for passive farmers or in Shove
   mirrors, at every tested duration; a bully interrupts half of the wards, a guard a fifth (§5).
5. **Safe is no longer cheap** (0.15–0.56 farm per turn; always-Safe is exploitable), and **threat-only zoning
   exists** where the weaker side defends near its tower (−2.1 resources per 50 turns with ≈ 3 connected trades; §6).
6. **Restricted scripted games are Shove-centred**, on this and the historical preset; the members of the
   equilibria differ by contingent rules (resets, Wards, freeze phases, selective trades). Learned M4.4b policies
   beat the scripted Shove by ≈ +3 on their own preset, so this is a limit of the scripted set (§7).
7. **The three approximations M4.5a flagged are negligible**: probe targeting 0.3% of champion-turns, crossings 0.1%,
   end-contact retaliation 4% of retaliation (§8.1).

**What changed.** `GameConfig.m45_calibrated()` = `m45_unified()` + `farm_spot_rule = "last_front"` (correction) +
`minion_retaliation_delay = 0` (calibration). Everything else is retained. α, ρ, φ, Vision and W exposure were
screened: the freeze, zoning and restricted-game indicators do not depend on them within range, while Vision's
magnitude does depend on its duration (§5, §10). Every earlier preset is
byte-identical (14 presets checked).

**Holdout.** The correction's effects generalize: freeze value later in the game, punishing walk-ups, more Vision
value, freeze-phase play in natural states, Safe's price and threat-only zoning. The early-trade fix generalizes.
**But the mid-to-late minion share fails its holdout criterion** (21% of the trade damage at level 14 against a 15%
threshold), and a power-gap criterion misses by 0.001 (§11). This was not retuned on the holdout; the reviewer chooses
between accepting it and `minion_damage_growth = 1.12` after fresh validation.

**Readiness: GO WITH CAVEATS** for M4.5c (§14).

---

## 1. Methodology

### 1.1 What was reused and what was built

The M4.4c engine evaluates *learned* policies through their observations; M4.5b has none on the M4.5 presets (no PPO
in this phase), so its counterfactuals use scripted strategies. The new package reuses the project's pieces wherever
they fit:

| piece | source | use in M4.5b |
|---|---|---|
| the turn resolver, lane-step cache, tick traces | `champions_m4.resolve_turn_m4`, `LaneEnv._lane_turn` | `engine.play`: the same resolver, the environment's chance stream (`default_rng(seed)`, ten uniforms per turn whatever the actions), so the viewer replays every game from its `Setup` (tested turn by turn) |
| four-action scripted strategies and pre-rules | `experiments/m43c/strategies.py` (`Strategy`, wave and behaviour rules, the low-HP Recall rule) | extended by `strategies.Contingent` (phase-dependent rules, below) |
| Ward strategies, M4.5a starts | `experiments/m45a/scenarios.py` | in the libraries |
| standard starts and power gaps | `zoning_natural.starts`, `viewer.session.start_state` | the freeze start of the exploration; viewer recordings |
| freeze predicate | `experiments/rl/m44/records.holds_freeze` | freeze durations and phases |
| matrix-game solver | `matrix_game.solve_matrix_game` | restricted games, best replies, supports |
| recording export and replay check | `experiments/m45a/demos.summarize`, `experiments/rl/m44/trace.verify` | `demos.py` |
| M4.4b checkpoints | `results/rl/checkpoints/m44/ext` (git-ignored) | a learned reference opponent (evaluation only, `reference.py`) |

New: `engine` (paired loop, a 65-key per-side ledger with snapshots at 10 / 20 / 50 turns, discounted leads G1–G300,
state collection, composite plans), `exact` (one-turn expectations by enumerating the trade, tower and, optionally,
gank dice), `states` (parametric starts and the holdout split), `strategies` (≈ 60 contingent strategies), one module
per Priority, `sensitivity`, `validate`, `reference`, `demos`, `report`.

**Contingent strategies.** A `Contingent` strategy has a wave and a behaviour rule per phase (*freezing*: we hold
the enemy front at our −1 or deeper; *frozen*: the enemy holds ours; *default*), plus optional pre-rules: the M4.3c
low-HP Recall (≤ 40%, or ≤ 70% while the enemy is away), *Shove-then-Recall* below a threshold, a *frozen-on reset*
(Recall below 60% while frozen on), and Ward rules (`favorable`, `forward`, `exposed`, `away`, `always`). New
behaviour rules: `guard` (Trade an enemy catchable this turn without a dive, at own HP ≥ 50% with a non-negative duel
edge), `nodive` (Trade whenever catchable without a dive), `wary` / `wary_yield` (Safe / Yield while a stronger enemy
can catch us). The freezer family (`fz_*`) varies the freeze-phase rule and what follows a broken freeze; the
responses (`wk_*`) vary the frozen-phase rule, each with a Last Hit or a Shove default (`_sh`).

### 1.2 Seeds, states and the holdout split

| set | values | used by |
|---|---|---|
| DEV seeds | 87,000–87,499 | every calibration and diagnostic stage |
| HOLDOUT seeds | 88,000–88,499 | `validate` only |
| TRACE seeds | 87,900–87,999 | viewer recordings |
| DEV states | freeze: levels 4 / 10, gaps equal / A ×1.76, held 5 / 12 / 20 HP, freezer HP 1.0 / 0.6; trade bench: levels 2 / 6 / 11 / 18, waves 8 / 16 / 24 HP; Vision: neutral, forward, presiege, defending, frozen_on; zoning: neutral, a_push (and b_push in the baseline zoning matrix) | development |
| HOLDOUT states | freeze: levels 3 / 7 / 13 / 16, gap A ×1.5, held 8 / 16, HP 0.8 / 0.45; trade bench: levels 3 / 8 / 14, waves 12 / 20; Vision: push_lane, deep_farm; zoning: b_push at levels 7 / 13, ×1.5; natural states at turns 60 / 150 | `validate` only |

The split was written to `results/m45b/holdout_split.json` before any calibration run. One amendment, also before any
calibration run and recorded in the file: the Vision situation `frozen_on`, assigned to the holdout, had been used by
the development Vision stage; it was reclassified as DEV and two new holdout situations were defined. The `b_push`
zoning situation appears in the baseline (DEV-seed) zoning matrix and, with different levels, gap and seeds, in the
holdout. All ranges are disjoint from every earlier study (10,000–81,999, 85,000–86,999, 90,000–91,999; tested).

### 1.3 Measurements and statistics

- **Objective.** G<sub>h</sub> = Σ<sub>t<h</sub> 0.99<sup>t</sup> r<sub>t</sub>, A's discounted gold lead (the RL
  objective truncated at h); the reward is unchanged (earned gold, zero-sum).
- **Ledger** per side: gold by source, farm available and uncollected, XP credited / available / lost, incoming
  damage by source in bars (trade, minion retaliation, minion positional, tower, gank), minion units and ticks by cause,
  trades attempted / connected / landed, evasions, deaths by cause, Recalls, turns away, wards attempted / placed /
  interrupted, protected and exposed protected turns, ganks, freeze turns (held, frozen on), crashes, tower damage,
  plates, capped trades, action counts, end-state HP / level / investment; snapshots at 10, 20 and 50 turns (XP reaches
  the level cap within 200 turns from level 10, so it is compared at short horizons).
- **Pairing.** Every comparison uses the same DEV seed list in every cell (common random numbers).
- **Intervals.** 95% t intervals of paired differences over seeds, or over states when states are the units
  (labelled). Restricted-game supports: a seed bootstrap (pairing kept).
- **Freeze premium.** The value of the freezer × response game minus the value of the same game restricted to the
  non-freezing rows (`g_shove`, `shove_recall`). The in-sample premium is biased upward (a maximum over noisy rows), so
  the report uses a split-sample premium: the equilibrium mixtures are fitted on a random half of the seeds and
  evaluated on the other half against B's best reply there, averaged over 100 splits (conservative).
- **Exact one-turn expectations** (`exact.py`) are checked against Monte Carlo in the tests.

### 1.4 Pre-registered criteria

Fixed after the baseline measurements and before any calibration run. *Trades* (exact, one turn; delta = the Trade's
HP exchange minus Hold's against the same B action, because without own cover the enemy wave fires on positional aggro
whether A trades or not):

| | criterion |
|---|---|
| B1 | L2, equal power, own cover, B Hold: trading into a 24-HP wave is not HP-positive (delta ≤ 0) |
| B2 | the same into an 8-HP wave: clearly HP-positive (delta ≥ +0.05 bar) |
| B3 | L2, A ×1.76, 24-HP wave: clearly HP-positive (delta ≥ +0.05) |
| B4 | L18, 128 invested, equal power, 24-HP wave: minion fire ≤ 15% of the trade damage dealt |
| B5 | the minion share of the trade damage falls strictly with level (2, 6, 11, 18) |
| B6 | at L2, delta falls strictly with the enemy wave (8 > 16 > 24 HP) |

*Freezes*: guarding should beat passive freezing against a walk-up Shove somewhere; freezing should not dominate
everywhere (the premium is not positive in every state); the weak side's best reply should vary with the state.
*Vision*: positive value in some exposed or aggressive context, negative in the neutral passive lane, no Ward rule
universally best. *Zoning*: stepping back must cost something; a stronger side must be able to extract value from a
cautious opponent without connecting many trades. *Integrated*: no strategy outside the Shove family should be the
pure equilibrium of every restricted game; any dominance must be explained.

---

## 2. Baseline: how the revision-2 simulator behaves

`GameConfig.m45_unified()` (revision 2) on the DEV battery, before any change. Each line points to its section.

**What works as intended.**

- Minion fire on traders falls with level (68% of the trade damage at level 2, 4.5% at level 18 into a 24-HP wave),
  power still decides trades, and chasing a retreating defender into its wave is expensive (§4.1).
- Vision is contestable (51% of wards interrupted by a bully, none by farmers) and conditionally valuable: it pays for
  exposed shovers and against chasers, not for passive farmers or in Shove mirrors (§5).
- Safe costs real resources now (0.15–0.56 farm per turn, 5–9.5 resources per 50 turns when used all the time), and a
  stronger side extracts value from a cautious opponent with very few connected trades (§6).
- Freezes produce responses and counter-responses: a passive freeze is broken at once by a walk-up Shove; a guarded
  one holds 3–10 turns and is answered by Shove with a Recall reset; freezing has a positive premium mid-game (§3).
- The three timing approximations named by M4.5a are negligible (probe targeting 0.26% of champion-turns, crossings
  0.1%, end-contact retaliation 4% of retaliation; §8.1).

**The largest limitations.**

1. **A farm-spot defect, inherited from M4.2d** (§8.2): in mutual Shoves both followers walk out of reach of the
   fight their probe predicts, and 38–50% of the farm of Shove-mirror play dies out of reach (21.5% of all scripted
   play). Material because Shove
   mirrors are where the scripted equilibria are.
2. **Retaliation depends on the tick split** (§4.1): with delay 1 a damaged 8-HP wave never retaliates, and trading
   into a 24-HP stack at level 2 with own cover is HP-positive (criterion B1).
3. **Shove-centred restricted games** (§7), on this preset and the historical one; learned conditional policies beat
   the scripted Shove, so this is a limit of the scripted set, carried as a caveat.
4. **Freezing pays only mid-game** (§3): at level 4 guarding a freeze costs up to 0.9 bars of positional fire per 20
   turns (0.2–0.9 by state) and the premium is never positive; a kill made while freezing can lose the wave state (the bounce,
   demonstration 10).
5. **Single trades are economically neutral and noisy** (§4.3): one tick of diverted minion fire moves the next 20
   turns by ±0.8 resources through the wave state.

Items 1 and 2 are corrected / calibrated (§9); items 3–5 are measured properties, reported as caveats (§14).

---

## 3. Priority A — Strong-side freezing, zoning and Recall

### 3.1 First look: why the strong freezer lost to a walk-up Shove

The M4.5a smoke pairing (`freeze_bully4` ×1.76 v `walk_up_shove4`, −4.62 over 200 turns) was decomposed first:

- **The scripted freezer dives.** It chases B into B's tower cell every turn: 10,200 cHP of tower damage taken (more
  than all its other damage together), 11 Recalls, 24 turns away, 49% of its available farm collected against B's 65%.
  A scripted-behaviour artefact, not a property of freezing.
- **A freezer that only trades a walk-up without diving (`guard`)** wins the confrontation: in the first 10 turns
  +3.6 resources more than a passive freezer, B dying or recalling. It then loses it all by turn 20: during B's
  absence A's larger wave pushes and crashes into B's tower, and B returns to farm the crash under its own tower
  while A's lane is empty (*the kill bounce*, demonstration 10). A kill made while freezing must be converted (Shove
  and take plates, or keep the wave), which a Last Hit continuation does not do.
- **The standard freeze start is not self-sustaining**: the held wave (5 HP) is smaller than the freezer's (8 HP), so
  the freeze ends within ~17 turns even when nobody contests it. The grid below varies the held wave.

### 3.2 Design

States: A holds B's front of 5 / 12 / 20 HP at A's −1, A's 8-HP front in its tower cell, A at its tower, B walked up
to its +1; levels 4 and 10; equal power or A ×1.76; A at full or 60% HP (24 DEV states). Rows: ten freezer strategies
(passive Thin + Hold; guard; guard only against a walk-up; trade whenever catchable without a dive; Shove after the
freeze; Shove-then-Recall below 55%; Shove while the enemy is away) and the non-freezing references g_shove and
shove_recall. Columns: 16 frozen-on responses (walk-up Shove, Shove + Safe, walk-up Last Hit, step back, Yield,
contest, Shove but Safe under a stronger threat, Shove with a Recall reset below 60%, each with a Last Hit or a Shove
default). 40 paired seeds × 100 turns per cell, payoff G100 (`freeze_grid.json`; candidate: `candidate_dev.json`).

### 3.3 Results

Selected states (G100; premium = split-sample freeze premium; "g − p" = guarding minus passive freezing against the
walk-up Shove, paired; full tables in `results/m45b/report.md`):

| state | premium: baseline → candidate | g − p v walk-up Shove: baseline → candidate | A's equilibrium (candidate) | B's best reply to the guard (candidate) |
|---|---|---|---|---|
| L4 equal, held 5 | −0.24 → **+0.44** | +1.02 ± 0.84 → +0.40 ± 0.61 | fz_bully_sh 1.00 | walk-up Shove |
| L4 equal, held 12 | −0.11 → −0.08 | **+1.38 ± 0.56** → −0.09 ± 0.14 | shove_recall 1.00 | Shove + reset |
| L4 ×1.76, held 5 | −0.12 → −0.03 | +0.61 ± 0.37 → **+0.80 ± 0.50** | g_shove 1.00 | Yield |
| L4 ×1.76, held 12, A 60% | −0.09 → −0.09 | −0.03 → −0.09 | fz_guard_line_sh 0.99 | Yield |
| L10 equal, held 12 | +0.06 → +0.26 | +0.04 → −0.03 | fz_passive_sh 1.00 | wary Shove |
| L10 equal, held 20 | **+1.11 → +0.89** | ≈ 0 | fz_bully_sh 0.91 | walk-up Last Hit |
| L10 equal, held 20, A 60% | +0.60 → +0.69 | ≈ 0 | fz_bully_sh 1.00 | walk-up Shove |
| L10 ×1.76, held 5 | **+1.21 → −0.15** | +0.57 ± 0.57 → **+0.98 ± 0.66** | g_shove 1.00 | Shove + reset |
| L10 ×1.76, held 12 | +0.89 → +0.18 | +0.05 → +0.44 ± 0.54 | g_shove 0.87, fz_guard_line_sh 0.13 | Shove + reset |
| L10 ×1.76, held 20 | −0.17 → −0.04 | ≈ 0 | fz_guard_sh 1.00 | walk-up Shove |

Across the 24 states, the split premium exceeds +0.2 in 8 baseline states (all at level 10) and in 4 candidate states
(L4 equal held 5; L10 equal held 12 and held 20 at both HP). Freezer strategies carry equilibrium weight (> 0.05) in
14 of the 24 candidate states.

**The baseline's large-gap freeze premium was an artefact of the farm-spot defect** (§8). Against a walk-up *Shove*
the shoving freezer and the shoving responder were in a Shove mirror, where both lost a third of their farm to the
defect; the freezing rows avoided the mirror. With the correction the premium at ×1.76 falls to ≈ 0. What remains
is mechanical: at equal power with B's large wave sitting at A's tower, the walk-up B stands deep at its +1 while the
wave crashes, and a freezer that trades it there (`fz_bully_sh`, then Shove) earns the premium; the wave is not held
for long in these states (0 held turns against a walk-up Shove), so the value is that of the freeze phase's trades,
not of a long freeze. At level 4 equal power with a small held wave the same trading freezer is the equilibrium. The factorial (§10) confirms
that the correction, not α or δ, moves the premium.

### 3.4 The sequence of the brief, step by step (candidate)

| step | measured |
|---|---|
| 1. a stronger champion establishes a freeze | the freeze holds against a walk-up Last Hit for ~20 turns (passive) and against a walk-up Shove for 2–10 turns only if guarded (0 if passive) |
| 2. the weaker side approaches to break it | walk-up Shove breaks a passive freeze on the first turn; Shove + Safe cannot (no reach), stepping back costs farm |
| 3. the stronger side threatens or trades | guarding beats passive freezing against the walk-up Shove when strong (+0.80 ± 0.50 at L4, +0.98 ± 0.66 at L10, held 5), not at equal power |
| 4. the weaker side chooses | its best reply varies by state: walk-up Shove, Shove with a reset, wary Shove, Yield, walk-up Last Hit, contest (six different replies across the 24 states) |
| 5. wave control while trading | the guard keeps Thinning while connected (the trade budget cap never binds for Thin); φ 0.25–1.0 changes nothing (§10) |
| 6. trading and retaliation wear the freezer down | minion fire on the guard 0.2–0.9 bars per 20 turns at L4, 0.0–0.3 at L10 (mostly positional: it steps into the held wave) |
| 7. the weaker side resets | Shove with a Recall reset below 60% is B's best reply to the guard in 7 of the 24 candidate states |
| 8. the freezer re-evaluates | 29 natural states where B has just returned at full HP and A (≤ 75% HP) still holds the freeze (baseline): keep guarding beats keeping passively (−0.34 ± 0.17), Shove-then-Recall (−0.64 ± 0.25) and Recall now (−0.44 ± 0.20) at G100 |

### 3.5 Local action values and Recall economics

From each freeze state, A's first action is forced and A then plays `fz_guard_sh` (`freeze_local.json`, 200 paired
seeds, baseline). The Trade at the freeze ranges from +1.4 (G5) at L10 equal with a 20-HP held wave to −1.2 at
L10 ×1.76 with a 12-HP held wave against the walk-up Shove; breaking the freeze with Shove pays later in some states
(+1.8 at G100, L10 equal, 5-HP wave) and loses in others (−2.1 against a stepping-back B). Freeze decisions are
genuinely state-dependent: no first action is best everywhere.

### 3.6 Causes, as asked

| possible cause of a freeze failure | finding |
|---|---|
| inadequate champion damage | no: guarding wins the HP exchange (4.5 bars dealt for 0.6 of minion fire per 200 turns at L10 ×1.76) |
| excessive minion retaliation | not at L10; at L4 the guard's cost is positional fire (stepping into the held wave), real and intended |
| insufficient resource denial | partly: a walk-up at the enemy's +1 farms the freezer's dying minions at the full enemy-tower rate (0.9) and keeps XP range; the denial is the gank exposure of that cell and the guard's threat |
| excessive wave-clear effectiveness | yes for passive freezes: a walk-up Shove kills the freezer's 8-HP front in one turn |
| movement or contact limits | the tower-cell contact rule: a freezer in its own tower cell cannot reach the walk-up without stepping into its wave |
| wave-breaking dynamics | the kill bounce (§3.1) |
| Recall and return economics | the reset is used and effective (step 7) |
| inappropriate scripted behaviour | the M4.5a smoke freezer dived; contingent freezers fix it |
| a mechanics defect | **the farm-spot defect (§8)** distorted the large-gap results; corrected |

No freeze parameter was changed: the evidence does not support one (`trade_wave_fraction` is inert, minion damage
does not move the freeze indicators).

---

## 4. Priority B — Minion damage, retaliation and wave dynamics

### 4.1 One-turn trade values (exact)

The bench (`states.trade_bench`): A trades B, who stands in its own wave at the lane cell. *cover*: A's 8-HP front in
contact at A's −1, A with it; *nocover*: no A front (B's wave walks into A); *ahead*: A stands in B's front's cell, its
own front behind. Baseline, B Hold, full HP, equal power (bars of each champion's max HP):

| level | wave | cover: delta | landed: dealt / minion fire | nocover: delta | ahead: delta |
|---|---|---|---|---|---|
| 2 | 8 | +0.119 | 0.217 / 0.000 | +0.119 | +0.119 |
| 2 | 16 | +0.072 | 0.217 / 0.073 | +0.108 | +0.108 |
| 2 | 24 | **+0.031** | 0.217 / 0.148 | +0.098 | +0.097 |
| 18 | 24 | +0.093 | 0.433 / 0.020 | +0.098 | +0.098 |

- **Retaliation lasts one tick and depends on tick parity.** With `minion_retaliation_delay = 1` the designated
  wave fires only on tick 1. Under the half-wave law a wave's turn firepower is split over the ticks earlier-first, and
  a damaged 8-HP wave has 1 unit split 1 + 0: it never retaliates. 16 HP fire 1 unit, 24 HP 2 units.
- **B1 fails narrowly**: at level 2 trading into a 24-HP stack with own cover is still HP-positive (+0.031 bar).
  B2–B6 pass: small waves +0.119, a ×1.76 trader +0.135, the minion share 68% → 32% → 13% → 4.5% of the damage dealt
  from level 2 to 18.
- **Without cover the cost is positional, not retaliation**: the enemy wave walking into A (or A standing in it) fires
  whether A trades or not (0.22 bars at L2 into 24 HP), so delta is about the trade's own expected damage. Chasing a
  defender who steps back with Safe into its wave is the expensive case: an unconnected chase at L2 into 24 HP takes
  0.37 bars of positional fire.

### 4.2 The minion-parameter screen

All 18 combinations of α ∈ {30, 50, 80}, ρ ∈ {1.05, 1.08, 1.12}, δ ∈ {0, 1} (exact, `trades.json`):

| setting | criteria passed | B1 (24 HP) | B2 (8 HP) | B3 (×1.76) | B4 (L18 share) | delta 8 / 16 / 24 HP |
|---|---|---|---|---|---|---|
| α 50, ρ 1.08, δ 1 (baseline) | 5 | +0.031 ✗ | +0.119 | +0.135 | 4.5% | +0.119 / +0.072 / +0.031 |
| α 50, ρ 1.08, **δ 0** | **6** | **−0.067** | +0.079 | +0.078 | 11.4% | +0.079 / −0.002 / −0.067 |
| α 30, ρ 1.08, δ 0 | 6 | −0.002 | +0.095 | +0.116 | 6.9% | +0.095 / +0.042 / −0.002 |
| α 80, ρ 1.08, δ 1 | 6 | −0.008 | +0.119 | +0.112 | 7.2% | +0.119 / +0.050 / −0.008 |
| α 80, ρ 1.08, δ 0 | 4 | −0.167 | +0.055 | +0.022 ✗ | 18.1% ✗ | +0.055 / −0.068 / −0.167 |

No setting with δ = 1 and α ≤ 50 passes B1. δ = 0 at α 50 passes everything, with a clean gradient by wave size, and
keeps power decisive. α 80 with δ 1 also passes, narrowly (B1 at −0.008), but it scales positional aggro by 1.6 as
well, a broader change than the one diagnosed. ρ shifts only the late-game share.

### 4.3 Economic value of a trade

A single trade is a small economic decision; its value is dominated by what it does to the wave.

- **The shadow price of HP.** Paired starts with A at full HP against A at 70%: 1–2 resources per bar under passive
  continuation (G20: +0.70 ± 0.63 at L2 to +2.16 ± 0.61 at L18; noisy under aggressive continuation), so one trade's
  ±0.1 bar is worth ±0.1–0.2.
- **Natural states** (360 decision states from four pairings, states as units, 32 paired rollouts each, `trades_natural`):
  Trade versus Hold on one turn, then the pairing's own strategies.

  | setting | G50, all states | G50, enemy blob ≥ 17 HP | HP lost @10, A / B |
  |---|---|---|---|
  | baseline | −0.02 ± 0.06 | +0.04 ± 0.22 | 0.06 / 0.07 |
  | δ 0 | −0.05 ± 0.06 | −0.07 ± 0.24 | 0.07 / 0.07 |
  | `last_front` | −0.00 ± 0.05 | −0.17 ± 0.19 | 0.03 / 0.08 |
  | candidate (both) | −0.03 ± 0.05 | −0.19 ± 0.20 | 0.04 / 0.07 |

  Trades are economically neutral on average in every setting, with a negative tendency into big waves; δ does not
  change the economics significantly.
- **Bench states** show why single-trade values are noisy: at L2–L18, one landed trade into a 16-HP wave with cover
  costs −0.6 to −0.75 at G5 and gains +0.6 to +0.8 at G20 *at every level and power gap*. The tick of retaliation
  diverts B's stack from A's 8-HP front, which survives a turn longer; B's crash and the bounce that would have
  favoured A come later, and B's stack grows. One tick of diverted fire moves the economy by ±0.8 through the wave state,
  far more than the trade's HP. This is the intended consequence of exclusive targeting (aggro preserves waves), but it
  makes the economic value of a trade state-dependent and noisy: a credit-assignment caveat for M4.5c.

### 4.4 Wave consequences of minion fire on champions

From every turn of 12 scripted pairings and uniform random play (`timing.json`, section D):

- minions fire at a champion on 5.5% of champion-turns (scripted; 12% random), 71% of them while their front is in a
  lane fight;
- the diverted fire kept an otherwise-dying enemy front alive on 2.8% of those ticks (0.15% of champion-turns);
- fire while the attacker's crashed mass stood under the defender's tower (a diverted defence): 0.1% of fire ticks.

Stacks did not grow beyond the packet cap and no rear-packet bound was exceeded in any run. The exploitable pattern
found is the bench effect of §4.3 (a trade that preserves the own wave), not an infinite loop; it is reported, not
corrected.

---

## 5. Priority C — Vision positioning and economics

### 5.1 One ward (local value)

From the Vision situations (neutral, forward, presiege, defending, frozen_on) at levels 4 and 10 and gaps equal /
A ×1.76 / B ×1.5, A wards on the first turn or plays its base strategy's action; both then continue (A: g_farm or
g_shove; B: g_farm, g_shove, g_guard, g_bully). 240 cells × 200 paired seeds × 100 turns (`vision.json`).

- **The ward turn costs** 0.23–0.45 resources for a farmer (the turn's farm and a wave without its last-hitter) and
  0.65–1.57 for a shover (a turn without the full wave capacity).
- **It buys little protection per ward**: over 100 turns gank damage falls by 0.1–0.3 bars and gank deaths by a few
  hundredths, because one ward covers 6 of the 100 turns.
- **Contestability works**: a first ward is placed every time against farmers and shovers, 69% of the time against a
  guard and 43% against a bully (whose trades interrupt it).
- One-ward G100: 22 cells significantly positive, 137 negative, 81 not distinguishable from 0. The positive cells are
  presiege / neutral positions of a farmer or shover against a bully (up to +3.7 at G100), where the chaser follows
  the warder toward W into A's wave: "ward baiting", a positional interaction rather than Vision value.

### 5.2 Ward rules as strategies

Ward rules (`favorable`, `forward`, `exposed`, `away`, `always`) on three base strategies against four opponents, from
s0 and from the neutral lane at three power gaps, 48 paired seeds × 200 turns (240 contrasts against the same base
without wards):

| base \ opponent | g_farm | g_shove | g_guard | g_bully |
|---|---|---|---|---|
| g_farm | −0.58 | −0.63 | −0.73 | −0.50 |
| g_shove | **+0.35** | −0.99 | −0.10 | **+0.57** |
| g_guard | −1.17 | −1.11 | −1.50 | **+0.91** |

(mean dG200 over rules and starts). 24 contrasts are significantly positive, 121 negative. A shover that wards when its
wave is forward halves its gank damage (12.0 → 6.6 bars per 200 turns) and gank deaths (0.98 → 0.52) and gains
+0.6 ± 0.5 against a farmer and +1.1 ± 0.9 against a bully, but loses −1.6 ± 0.6 in the Shove mirror, where the ward
turn concedes the wave war. Interruption rates: 51% of ward attempts against a bully, 19% against a guard, ≈ 0 against
farmers and shovers. "Ward whenever vision is down" is never the best rule.

**Answer: Vision is situationally valuable, not dominated and not universal**: it pays for the champion that will
stand exposed against an opponent that does not punish the lost wave turn, and against chasers, and it does not pay for
passive farmers or in Shove mirrors. The stronger champion does not ward more safely as such (interruption depends on
the opponent's trading, not on power); it gains because its exposed positions are worth protecting.

### 5.3 Vision parameters

| setting | one ward: mean dG100 (sig + / − of 32) | Ward rules: mean dG200 (sig + / − of 36) | contexts where a Ward rule beats none |
|---|---|---|---|
| 4 turns, ×0.25 | −0.015 (2 / 10) | −0.66 (2 / 20) | 3 / 12 |
| **6 turns, ×0.25 (baseline)** | +0.049 (4 / 6) | −0.28 (5 / 16) | 5 / 12 |
| 8 turns, ×0.25 | +0.069 (4 / 6) | −0.07 (12 / 13) | 7 / 12 |
| 6 turns, ×0.5 | +0.013 (3 / 9) | −0.45 (1 / 18) | 5 / 12 |
| W exposure −1 / +1 / +2 | +0.049 / +0.037 / +0.005 | −0.23 / −0.56 / −0.94 | 6 / 5 / 2 of 12 |

Vision's value rises monotonically with its duration and reduction, and **the contexts where it pays are the same at
every setting** (shovers against farmers and bullies; never in the mirror; never for farmers against shovers): at 8
turns the mirror is worse (−2.47 instead of −1.62) and the favourable contexts better (+1.31 instead of +0.60). The
pre-registered Vision criteria hold at 6 turns, so the baseline is retained; duration is the most magnitude-sensitive
retained parameter (§14 caveat 4). Walking through a held lane cell to W (positional fire on the route) is part of the
measured cost; it makes wards from the wrong side of the wave costlier, as intended.

### 5.4 The candidate

Under `m45_calibrated()` (`candidate_dev.json`) the same Ward-rule contrasts give 39 significantly positive and 120
negative of 240 (baseline 24 / 121); a ward pays most for a shover against a bully (+1.49 mean dG200) and for a guard
against a bully (+1.50), a little for a shover against a farmer (+0.25), and never for a farmer (−0.27 to −0.88) or
in the Shove mirror (−1.12). The contexts are those of the baseline; the corrected farm spot slightly raises the value
of protected forward play.

---

## 6. Priority D — Safe, Yield and threat-based zoning

### 6.1 One turn: the price of stepping back (exact)

From the neutral lane, A's push (the fight near B's tower) and B's push (near A's tower), levels 2 / 6 / 11, equal and
A ×1.76 (`zoning.json`): B's farm and XP lost and HP saved by Yield or Safe instead of Hold, for the same A action.

| situation | Safe v A Hold: farm / XP | Safe v A Trade: farm / XP / HP saved (equal, ×1.76) | Yield v A Trade |
|---|---|---|---|
| neutral | −0.30 / −1.0 | −0.39 / −0.4 / +0.12, +0.21 | identical to Hold (pinned: in contact at the start) |
| A's push | −0.15 / 0 | −0.16 / −0.4 / +0.13, +0.23 | −0.02 farm, +0.03 / +0.06 HP |
| B's push | −0.56 / −1.0 | −0.45 / −0.2 / +0.09, +0.15 | identical to Hold |

The numbers do not depend on the level (both the reach ball and the HP exchange scale with it). **One turn of Safe now
costs 0.15–0.56 resources of farm**, against ≈ 0.08 in M4.4c: the reach ball and XP range 2 make a step back expensive.
At the HP price of §4.3 (1–2 resources per bar), Safe pays against a single expected trade only for a clearly weaker
champion, and Yield's evasion is weak because a Yield in contact at the start is pinned.

### 6.2 Multi-turn decomposition

The stronger side's strategies (g_farm, g_bully, g_guard, g_shove, g_shove_guard) against the weaker side's (g_farm,
g_yield, g_safe, g_wary, g_wary_yield, g_thin, g_shove), three situations × levels 6 / 10 × gaps ×1.76 / equal,
64 paired seeds × 100 turns. B's earned resources over 50 turns, paired differences:

| situation (L6, ×1.76) | always Safe, no threat | always Yield | A Trades whenever catchable | A guards (threat + trades) | **threat-only**: A guards, B wary | A's connected trades / 100 turns (guard v wary) |
|---|---|---|---|---|---|---|
| neutral | −9.11 ± 0.34 | +0.81 ± 0.40 | −4.27 ± 0.63 | −2.14 ± 0.47 | **−1.32 ± 0.37** | 1.6 |
| A's push | −4.92 ± 0.33 | −0.38 ± 0.08 | −2.77 ± 0.58 | +0.65 ± 0.28 | +1.15 ± 0.31 | 1.5 |
| B's push | −9.49 ± 0.26 | −0.45 ± 0.18 | −3.85 ± 0.43 | −1.79 ± 0.40 | **−2.09 ± 0.29** | 2.5 |

(the same pattern at level 10 and at equal power, smaller; `results/m45b/report.md`).

- **Safe is not a cheap answer.** Always-Safe costs B 5–9.5 resources and up to 37 XP in 50 turns against an opponent
  who never threatens it, and the stronger side's best G100 against an always-Safe B is +2.0 to +6.9 higher than
  against a farmer (exploitable).
- **Threat-only zoning exists.** A cautious B that steps back only when a stronger A can catch it loses 1.3–2.1
  resources per 50 turns in the neutral lane and near its own tower although A connects only 1.4–2.5 trades per 100
  turns; B's XP falls by up to 16 in the B-push lane. In A's push lane the threat costs A more than B (+1.15), because
  A leaves its pushing wave to threaten.
- **B's best reply depends on the state**: Shove against farming, bullying and shoving, wary or wary-Yield against a
  guard at ×1.76, Thin against a shover at ×1.76.

### 6.3 The candidate

B's earned resources at 50 turns, baseline → candidate (level 6, A ×1.76; `candidate_dev.json`):

| situation | always Safe | A guards | threat-only (A guards, B wary) | exploit always-Safe (A's G100) |
|---|---|---|---|---|
| neutral | −9.11 → −7.18 ± 0.47 | −2.14 → −0.82 ± 0.53 | −1.32 → −0.43 ± 0.44 | +3.29 → +2.25 |
| A's push | −4.92 → −4.94 ± 0.31 | +0.65 → +0.24 ± 0.34 | +1.15 → +0.78 ± 0.31 | +2.31 → +1.75 |
| B's push | −9.49 → −7.30 ± 0.36 | −1.79 → −1.97 ± 0.37 | −2.09 → −2.10 ± 0.37 | +5.42 → +3.15 |

Safe stays expensive and exploitable. Threat-only zoning survives where it matters most, when the weaker side
defends near its own tower (−2.1, with ≈ 3 connected trades per 100 turns), and becomes marginal in the neutral lane
(−0.4 ± 0.4): part of the baseline's neutral-lane effect was the farm-spot defect hitting the stepping-back side.

---

## 7. Priority E — Integrated strategy balance

### 7.1 Restricted games

24 strategies (Last Hit, Thin, Slow Push, Shove and its Recall / grow / hunt variants, guard, bully, Safe, wary,
Yield, three Ward variants, four Shove + selective-trade hybrids, two freeze-phase and two frozen-phase strategies),
every pairing on the same 48 DEV seeds × 200 turns, payoff G200 (`matrix.json`). From s0 the game is symmetric, so the
antisymmetrized matrix (M − Mᵀ)/2 is solved as well.

| game | equilibrium (antisymmetrized when symmetric) | bootstrap support frequency | weakly dominated |
|---|---|---|---|
| r2, s0, equal | wk_shove_reset_sh (pure) | wk_shove_reset_sh 0.66, g_shove 0.23, g_shove_sr 0.15, wk_wary_shove_sh 0.09, fz_passive_sh 0.07 | g_slow, g_guard, g_bully, g_safe, g_wary, g_guard_ward, g_shove_guard, g_slow_guard |
| aggro, s0, equal | wk_shove_reset_sh (pure) | wk_shove_reset_sh 0.56, g_shove 0.48, fz_passive_sh 0.21, wk_wary_shove_sh 0.14, g_shove_sr 0.12 | g_slow, g_safe, g_shove_guard, wk_wary_shove_sh |
| r2, neutral, A ×1.76 | A: g_shove_ward 0.67, g_shove 0.29; B: grow_shove4 0.70, g_shove_sr 0.28 | — | g_guard, g_safe, g_guard_ward, g_slow_guard |
| r2, A's push, equal | A: wk_wary_shove_sh 0.55, wk_shove_reset_sh 0.45; B: g_shove_edge 0.52, wk_shove_reset_sh 0.48 | — | g_guard, g_bully, g_safe, g_wary, g_ward, g_guard_ward, g_slow_guard |

`wk_shove_reset_sh` is Shove with a Recall reset below 60% HP while frozen on; `fz_passive_sh` freezes while it can
and Shoves otherwise. Every equilibrium is inside the **Shove family**; what distinguishes its members is contingent
(resets, Wards, a freeze phase, a cautious step back), and the non-Shove strategies are dominated or far behind.

### 7.2 Is this the game or the strategy set?

The same holds on the historical aggro preset, so it is not caused by M4.5a; M4.3c and M4.3d found Shove the baseline
equilibrium of every restricted scripted game. To separate the game from the strategy set, the M4.4b aggro finals
(learned conditional policies, evaluation only) played the scripted strategies (`reference.json`):

| preset | network seeds | G200 for the network against g_shove | against g_shove_sr | against wk_shove_reset_sh | against grow_shove4 |
|---|---|---|---|---|---|
| aggro (native) | 0 / 1 / 2 | +3.42 / +3.42 / +2.78 (all ±0.7) | +3.10 / +3.39 / +2.78 | +3.80 / +3.51 / +3.03 | +3.78 / +2.09 / +3.16 |
| r2 (transfer, OOD) | 0 / 1 / 2 | +1.00 / +1.39 / +1.14 (all ±0.7) | +0.09 / +0.75 / +0.89 | +0.47 / +1.26 / +1.18 | +0.02 / +0.15 / −1.20 |
| candidate (transfer, OOD) | 0 / 1 / 2 | +0.70 / +0.87 / +0.37 (±0.7) | +0.57 / +0.94 / +0.96 | +0.65 / +0.82 / +0.64 | −0.35 / −1.20 / −1.31 |

(both seats, 32 DEV seeds × 200 turns each; intervals ≈ ±0.6–0.8). Natively each network beats **every** scripted
strategy by +2.1 to +7.0, plain Shove by ≈ +3, mostly through selective trades that kill the shover (B dies 4–5 times
per 200 turns against 0.7–1.0 for the network). Transferred out of distribution (it never saw minion targeting, the
reach ball or XP range 2; its siege-Vision code is now Ward, which it plays 5–16 times by accident), the same
networks still beat plain Shove on revision 2 on every seed, and keep a positive point estimate on the candidate (one
of three significant); they lose to `grow_shove4` in transfer. A policy trained on the M4.5 preset itself is
M4.5c's question.

Hand-written Shove + selective-trade hybrids do not reproduce this: `g_shove_guard`, `g_shove_edge` and
`g_slow_guard` all lose to plain Shove (−0.6 to −4.9) on both presets, because scripted trading pays the trade-farm
factor, chases into waves and gives up the wave war. **Shove's dominance is a property of the scripted strategy set,
not of the game**: conditional play that trades selectively beats it by ≈ +3 on its own preset, and the same
networks, out of distribution, still beat it on revision 2 and, in point estimate, on the candidate.

### 7.3 Answers

| question | answer |
|---|---|
| Is Shove universally dominant again? | within scripted sets yes (both presets); against learned conditional play no (§7.2) |
| Is aggressive Trading consistently exploitable? | indiscriminate trading (bully, guard on every catchable turn) is dominated; selective trading is what beats Shove |
| Is pure passive farming overly effective? | no: Last Hit / Thin are dominated by Shove-family strategies everywhere |
| Is Safe an excessively cheap answer to pressure? | no: always-Safe is exploitable (+2 to +7) and costs 5–9.5 resources per 50 turns (§6) |
| Is Ward situationally valuable or generally dominated? | situational: in the equilibrium at ×1.76 (A plays g_shove_ward 0.67), dominated in passive play (§5) |
| Do freezes produce responses and counter-responses? | yes: a guarded freeze is broken by Shove + reset, not by a plain walk-up; freezing enters equilibria mid-game (§3) |
| Do policies benefit from switching actions by state? | yes: every equilibrium member is contingent, and the learned policies beat every fixed script |
| Are the trade-offs sensitive to reasonable parameter changes? | the restricted games are not (§10) |

### 7.4 The candidate

| game | equilibrium | bootstrap support frequency (s0) |
|---|---|---|
| s0, equal | g_shove_sr (pure, antisymmetrized) | g_shove_sr 0.44, g_shove 0.25, fz_passive_sh 0.20, wk_shove_reset_sh 0.17, fz_guard_sh 0.07, g_shove_edge 0.01, wk_wary_shove_sh 0.01 |
| neutral, A ×1.76 | A: g_shove_edge 0.83, wk_wary_shove_sh 0.17; B: g_shove_edge 0.74, g_shove_wary 0.26 | — |
| A's push, equal | A: g_shove_edge 1.00; B: wk_wary_shove_sh 1.00 | — |

The restricted games remain Shove-centred, but their contingent members change: the Shove that trades only with a
clear duel edge (`g_shove_edge`), which lost to plain Shove under the baseline, carries most of the equilibrium mass
from the two asymmetric starts, and from s0 the bootstrap support spreads over five strategies (baseline: one strategy
in two thirds of the resamples). From s0 itself `g_shove_edge` still trails plain Shove (−0.46).

---

## 8. Priority F — Movement and timing diagnostics, and the farm-spot defect

Every turn of 12 scripted pairings (16 DEV seeds × 300 turns, 108k champion-turns) and of uniform random legal play
(8k champion-turns) was analysed from the resolver's own records (`timing.py`; `timing.json`).

### 8.1 The three approximations named by M4.5a

| | scripted play | random play | verdict |
|---|---|---|---|
| **A. probe ignores minion targeting**: per-tick fronts differ between the probe re-run *with the authoritative reach and budgets* and the authoritative lane step | 0.26% of champion-turns | 0.57% | negligible; unchanged |
| **B. crossing between ticks**: a champion and the oncoming enemy front swap cells within a tick, own front behind | 0.11% of champion-turns; the firepower a swept-path rule could have fired: 0.012 cHP per champion-turn | 0.45%; 0.033 cHP | rare and worth nothing; movers involved: Hold and Trade, no exploit pattern; unchanged |
| **C. trade contact at the end cells, exchange on tick 0**: connected trades whose contact exists only at the end of a full-speed chase | 6.5% of connected trades, drawing 4.2% of all retaliation (0.023 cHP per champion-turn) | 14% and 10% (0.09 cHP) | a contact-timed exchange would remove that retaliation; immaterial; unchanged |

### 8.2 The defect behind the probe error: a self-defeating farm spot

The probe re-run exactly as the resolver runs it differs from the authoritative lane step on 28.8% of scripted
champion-turns, almost none of it from minion targeting (8.1 A). The source is the probe's other approximation, which
predates M4.5a: it resolves the turn under *full champion support*, every playing champion damaging enemy minions with
the line reach `reach_cap` wherever they are, and a follower's farm spot is its front's cell at the end of the probe.

When the probe has the own front die, the spot jumps back to the next packet. In a mutual Shove both fronts are
predicted to die (each to the other champion's Shove), so **both champions walk back to their next packets**, where
the reach ball cannot touch the fight; the fronts survive, and neither champion collects. The example of the tests:
both 4-HP fronts and both champions at the lane cell, both Shove: the probe kills both fronts, both end at −2 with reach
−3…−1, both fronts survive, farm 0 / 0 (it would have been 0.45 each). Two fresh 8-HP fronts walking to meet at the
lane cell produce the same pattern.

Measured farm dying beyond the champion's reach, as a share of the farm available, over every turn of the timing
stage's pairings from s0 (16 DEV seeds × 300 turns; `timing.json`, `candidate_dev.json`):

| pairing | baseline (`pre_crash_front`) | candidate (`last_front`) | farm collected, baseline → candidate |
|---|---|---|---|
| g_shove v g_shove | **43.9%** | 6.6% | 0.50 → 0.83 |
| g_shove_guard v g_shove | 50.3% | 5.2% | 0.43 → 0.82 |
| g_shove_ward v g_shove | 38.0% | 6.3% | 0.55 → 0.84 |
| fz_guard_sh v wk_shove_sh (both shove from s0) | 45.0% | 6.6% | 0.49 → 0.83 |
| g_ward v g_bully | 21.6% | 18.9% | 0.65 → 0.67 |
| g_wary v g_bully | 19.6% | 17.7% | 0.68 → 0.69 |
| g_bully v g_farm | 15.3% | 13.8% | 0.70 → 0.71 |
| g_farm v g_farm | 11.4% | 10.4% | 0.76 → 0.77 |
| g_shove v g_farm, g_guard v g_shove, g_safe v g_shove, g_slow_guard v g_shove | 1.4–3.7% | 1.2–2.5% | 0.80–0.85 → 0.83–0.85 |
| **all scripted play** | **21.5%** | **7.5%** | **0.67 → 0.79** |
| uniform random play | 9.3% | 6.7% | 0.75 → 0.77 |

(The exploratory measurement on 200-turn games gave 27–32% in mirrors; longer games, where wave capacity grows with
level, lose more.) The residual 10–19% in non-Shove pairings is ordinary positioning (a trader or a warder away from
its wave), not the defect.

The defect is identical on `m4_four_actions_econ_aggro()` and `m45_unified_r1()` (Shove mirror: 76.5% of available
farm collected, end cells inconsistent with the realized fronts on 30% of turns), so it is inherited from M4.2d's
natural waves, not introduced by M4.5a. It matters because **Shove mirrors are where the scripted equilibria live**
(§7) and, with Shove at ~36% in the M4.4b policies, a large share of self-play. A two-pass resolution (feeding the
authoritative fronts back as the probe) is not a fixed point (it oscillates on 20% of turns), so it is not a
correction.

### 8.3 The correction (`farm_spot_rule = "last_front"`)

The narrowest consistent rule extends the existing M4.2d option `pre_crash_front` (a front that *crashes* in the probe
leaves the farm spot at its pre-turn cell) to a front that *dies*: if side s's pre-turn front stops leading during the
probe (crashed, gone, or the leading packet behind its previous cell), the farm spot is the cell the front held last
while it led (its pre-turn cell if lost on tick 0, else its cell after the previous tick); the per-tick goals of §14.5a
use the same rule tick by tick. The champion walks back to its next wave on the following turn. Equations: reference
§15.1. Properties:

- it changes nothing on turns where no front is lost in the probe (tested on 300+ random turns);
- action semantics, ticks, decisions, spaces and the reward are unchanged; mirror equivariance holds (tested);
- the side effect is real and small: a follower that stays where its front died can stand in the enemy wave's path
  (positional fire in the Last Hit mirror 0 → 0.03 ticks per champion-turn);
- it needs `champion_movement_rule = "tick"` (the probe's per-tick fronts; validated) and is opt-in: every earlier
  preset keeps `pre_crash_front`.

What it does *not* fix: the probe still assumes full support, so two-pass inconsistencies remain on ~22% of mirror
turns, now mostly benign (both cells within reach of the fight). The remaining out-of-reach farm in mirrors (5–7%) is
of the same order as in non-mirror play.

---

## 9. Calibration decisions

The candidate is `GameConfig.m45_calibrated()` = `m45_unified()` revision 2 with **one mechanical correction and one
numerical calibration**. Everything else is retained.

### 9.1 Mechanical correction: `farm_spot_rule = "last_front"`

1. **Problem.** In mutual Shoves, followers walk out of reach of the fight their probe predicted; 38–50% of the farm
   of Shove-mirror play dies out of reach (21.5% of all scripted play).
2. **Evidence.** §8.2: the per-pairing out-of-reach shares; the concrete two-cell examples (tests); identical on the
   historical aggro preset; the two-pass check.
3. **Change.** A front lost in the probe keeps its last cell as farm spot (reference §15.1); opt-in, earlier presets
   unchanged.
4. **Before / after.** Out-of-reach farm in mirrors 38–50% → 5–7%; collected 0.43–0.55 → 0.82–0.84; non-mirror play
   unchanged (all scripted play: 21.5% → 7.5%, collected 0.67 → 0.79). Strategic consequences (§3, §6, §7): the freeze premium at ×1.76 falls to ≈ 0 and stays at equal power with a large held wave
   (it had been inflated by the mirror losses of shoving rows, §3.3); guarding against a walk-up Shove pays for a strong
   freezer with a small held wave (+0.8 to +1.0); threat-only zoning in the neutral lane shrinks (−1.3 → −0.4) while
   near the weaker side's tower it is unchanged (−2.1, §6.3); a Shove that trades only with a clear duel edge enters the
   equilibria of asymmetric starts, and the equilibrium support from s0 spreads over five Shove-family strategies (§7.4).
   The factorial (§10) attributes these shifts to the correction, not to the delay.
5. **Regressions and limits.** Positional fire on followers in the Last Hit mirror 0 → 0.03 ticks per champion-turn;
   the full-support probe remains an approximation.

### 9.2 Numerical calibration: `minion_retaliation_delay = 0`

1. **Problem.** Retaliation fires only on the turn's last tick, where the half-wave law leaves a damaged 8-HP wave no
   firepower: whether minions retaliate depends on the tick split rather than on the wave (8 HP never, 16 HP 1 unit,
   24 HP 2 units), and at level 2 trading into a 24-HP stack with own cover is HP-positive (criterion B1 fails, +0.031
   bar). The M4.5a report had flagged this (§9.1 item 3: "retaliation lasts one tick; α = 50 compensates
   provisionally; delay 0, α and trades into 8 / 16 / 24-HP waves should be screened together").
2. **Evidence.** §4.2: of the 18 (α, ρ, δ) settings, every δ = 1 setting with α ≤ 50 fails B1; δ = 0 at α 50 passes
   all six criteria with the gradient +0.079 / −0.002 / −0.067 by wave size and +0.078 for a ×1.76 trader into the
   big wave. The same holds on the corrected mechanics (§10). The economic value of trades on natural states is
   unchanged within its intervals (§4.3), the freeze, zoning and matrix indicators do not move (§10).
3. **Change.** `minion_retaliation_delay` 1 → 0 (the wave turns on the trader from the exchange tick, so for the
   whole turn). α stays 50 (the M4.5a rationale for raising it from 30 was the one-tick delay; with δ = 0, α 30 passes
   B1 only at −0.002 and α 80 fails B3 and B4).
4. **Before / after.** B1 +0.031 → −0.067; late-game minion share 4.5% → 11.4% (criterion B4: ≤ 15%); single 8-HP
   waves now retaliate (1 unit).
5. **Regressions and limits.** None measured beyond the larger late-game share. α 80 with δ 1 is the alternative that
   also passes, narrowly; it was not chosen because α also scales positional aggro, which was not diagnosed as wrong.

### 9.3 Retained parameters

| parameter | baseline | investigated | decision | why |
|---|---|---|---|---|
| `minion_damage_alpha` | 50 | 30 / 50 / 80, jointly with δ and ρ; on both mechanics | retained | with δ 0 it passes B1–B6; freeze, zoning and matrix indicators flat in 30–80 |
| `minion_damage_growth` | 1.08 | 1.05 / 1.08 / 1.12 | retained | only the late-game share moves (B4 18% at α 50, δ 0, ρ 1.05) |
| `minion_range` | 1 | (no screen) | retained | no evidence against it |
| `trade_wave_fraction` | 0.5 | 0.25 / 0.5 / 1.0, in the freeze grid and the factorial | retained | no measurable effect on any indicator: freezers Thin within the cap, and no tested strategy shoves while trading often enough to matter |
| `vision_turns`, `vision_reduction` | 6, ×0.25 | 4 / 6 / 8 × 0.5 / 0.75 | retained, **sensitive** | Vision is already conditional (pays for exposed shovers, not for farmers or in Shove mirrors); 8 turns scales the magnitude in both directions without changing where it pays |
| `ward_exposure` | 0 | −1 / 0 / +1 / +2 | retained | −1 helps Vision slightly, +1 / +2 hurt; no defect |
| `xp_range` | 2 | — | fixed by design | not tuned (brief) |
| everything else | `m45_unified()` | — | unchanged | — |

---

## 10. Joint sensitivity

An indicator battery (`sensitivity.py`), each a reduced Priority experiment on DEV seeds: the six trade criteria and
two multi-turn trade values; the split-sample freeze premium and the guard − passive contrast in two freeze states; one
ward's value in two Vision situations; the passive cost of Safe and the threat-only zoning effect; an 8-strategy
symmetric restricted game from s0. Run on (i) the full factorial α {30, 50, 80} × δ {0, 1} × φ {0.25, 0.5, 1.0} on the
baseline mechanics (18 configurations), (ii) α × δ on the corrected mechanics (6), and (iii) one-at-a-time ρ 1.05 /
1.12, 8 Vision turns, W exposure −1 and φ 0.25 / 1.0 around the candidate (`sensitivity.json`,
`sensitivity_cal.json`).

| indicator | range over (i) | sensitive to | on the corrected mechanics (ii, iii) |
|---|---|---|---|
| trade criteria passed | 4–6 | δ, α | 6 at the candidate; 4 at α 80 δ 0 |
| B1 delta (L2, 24 HP) | −0.17 … +0.06 | δ (sign), α | −0.07 at the candidate |
| freeze premium, L10 ×1.76 h12 | +0.91 … +1.11 | — | **+0.11 … +0.17** (the correction, not α / δ, moves it) |
| freeze premium, L4 equal h12 | −0.16 … +0.11 | — | −0.02 everywhere |
| guard − passive v walk-up Shove, L4 equal | +1.28 … +1.55 | — | −0.09 everywhere |
| one ward, neutral L4 v bully (G50) | −0.71 … +0.01 | α, δ (noise ±0.35) | −0.02 … −0.51 |
| always-Safe cost (B, 50 turns) | 9.11 | none | 7.12–7.21 |
| threat-only zoning (B, 50 turns) | 1.32–1.33 | none | 0.41–0.45 |
| 8-strategy matrix | pure g_shove in all 18 | none | pure g_shove_sr in all 12 |

**Robust to the provisional numbers:** the freeze economics, the zoning economics and the Shove-centred restricted
games, within α 30–80, δ 0–1, ρ 1.05–1.12, φ 0.25–1.0, Vision 6–8 turns and W exposure −1 / 0. **Sensitive:** the
trade criteria (δ, α), the magnitude of Vision (duration, §5) and the single-trade economics (wave timing, §4.3). The
largest effect on strategic outcomes in the whole study is not a parameter but the farm-spot correction (§9.1), which
moves the freeze premium, the threat-only zoning effect and the guard − passive contrast.

---

## 11. Final evaluation on the holdout

Both configurations on the HOLDOUT seeds (88,000–88,499) and the HOLDOUT states, which no calibration stage used
(`validation.json`; §1.2). Same code as the development stages.

### 11.1 Results

| block | `m45_unified` (revision 2) | `m45_calibrated` | verdict |
|---|---|---|---|
| **HB1** L3, equal, 20-HP wave: no free trade (delta ≤ 0) | +0.039 ✗ | **−0.048** ✓ | the B1 problem and its fix both generalize |
| HB2 delta falls from a 12- to a 20-HP wave | +0.076 > +0.039 ✓ | +0.011 > −0.048 ✓ | |
| **HB3** L3, A ×1.5 into 20 HP: delta ≥ +0.05 | +0.107 ✓ | +0.049 ✗ (by 0.001) | marginal: positive, not "clearly" |
| **HB4** L14 (48 invested), 20 HP: minion share ≤ 15% | 8.6% ✓ | **21.4% ✗** | **regression** (below) |
| HB5 the share falls with level (3, 8, 14) | ✓ | ✓ | |
| multi-turn, Trade v Hold on bench states (G20, 12 / 20 HP) | +0.21 / +0.59 (L3) … | −1.08 / +0.33 (L3) … | bench wave timing (§4.3); sign flips with the wave |
| freeze states with split premium > +0.2 | 0 / 16 | **4 / 16** (L13 and L16, freezer at 80%) | freezing pays later in the game |
| freeze states with freezer strategies in A's equilibrium | 8 / 16 | **12 / 16** | |
| guard − passive v walk-up Shove, L16 ×1.5 held 8 / 16 | +0.12 / +0.01 | **+0.50 ± 0.45** / +0.39 ± 0.46 | punishing the walk-up when stronger |
| Ward rules: significantly positive / negative of 120 | 4 / 58 | **21 / 63** | more conditional value |
| one ward: positive / negative of 32 | 6 / 22 | 7 / 16 | |
| always-Safe cost, b_push L7 / L13 (B, 50 turns) | −8.75 / −9.27 | −6.52 / −7.03 | Safe stays expensive |
| threat-only zoning, b_push L7 / L13 | −1.12 / −1.74 | −1.21 / −2.02 | holds |
| restricted game from s0 (24 strategies) | g_shove (pure); bootstrap: g_shove 0.57, g_shove_edge 0.30, wk_shove_reset_sh 0.30 | g_shove (pure); bootstrap: g_shove 0.57, wk_shove_reset_sh 0.40, g_shove_edge 0.17, wk_wary_shove_sh 0.12, fz_guard_sh 0.10 | Shove-centred, both |
| 8-strategy games from natural states (turns 60 / 150 of g_shove_guard v g_guard) | A: fz_guard_sh in 1 of 5, Shove / Shove + Recall otherwise | A: **fz_guard_sh in 4 of 6** | freeze-phase play in natural states |

### 11.2 The regression: the mid-to-late minion share

`minion_retaliation_delay = 0` doubles the retaliation of a landed trade at every level, and the level falloff is set
by the resistance ρ<sup>L−1</sup> and the victim's durability M(S), which were not changed. The development criterion
B4 tested only level 18 with 128 invested (11.4% with δ 0, 4.5% with δ 1), and B5 only that the share falls; at
level 11 the development share was already 34% (13% with δ 1). The holdout's level 14 makes the mid-to-late cost
visible: 21% of the trade damage dealt goes back as minion fire.

**Not retuned here.** Changing the candidate after seeing the holdout would spend the holdout. For review:

- *accept*: 21% at level 14 is a real but moderate trade cost (the trade still nets +0.08 bar in HP against Hold there);
  or
- *adopt `minion_damage_growth = 1.12` with δ 0*, a variant that passed all six development criteria (B4 6.0% at
  level 18; §4.2). By the damage law the share scales by (1.08 / 1.12)<sup>L−1</sup>, so the level-14 share would be
  ≈ 13% (0.214 × 0.62) and the level-3 values change by ≈ 7%. **This is a post-hoc projection from the holdout
  failure, not a validated result**: it needs fresh holdout states and seeds (e.g. 89,000–89,999) before M4.5c.

### 11.3 What the holdout establishes

The farm-spot correction's strategic effects generalize: freezing becomes a real, conditional option later in the
game, the guard punishes walk-ups when stronger, Vision's conditional value grows, and the equilibria of natural
states include freeze-phase play. Safe's price and threat-only zoning hold. The trade criteria split: the early fix
generalizes, the mid-to-late cost is too high by the pre-registered threshold. Shove-centred restricted games remain
the property of the scripted set in both configurations.

---

## 12. Visual diagnostics

`streamlit run lanegame/viewer/app.py` → "Recorded trajectory", folder `results/m45b/trajectories`. Each
demonstration is recorded twice, under `m45_unified` (revision 2) and under `m45_calibrated`, as
`<name>__<config>.json`; each uses the first TRACE seed (87,900–87,999) that shows its event under that configuration
(a predicate fixed in the label), so the two recordings of a pair may use different seeds. Every file replays exactly
through the viewer's session code (`m44.trace.verify`). The starts are registered with the viewer
(`experiments/m45b/scenarios.py`, `m45b_*` scenarios; viewer gap "equal": each start carries its own levels and power).

| file | brief item | pairing (A v B), start | what to watch |
|---|---|---|---|
| `m45b_01_freeze_held` | 1. a successful freeze | fz_passive_sh v wk_lh_sh, L10 ×1.76, held 12 | the held wave at A's −1; B farming at its +1 (exposed to ganks); ≥ 12 held turns |
| `m45b_02_freeze_broken` | 2. a broken freeze | fz_passive_sh v wk_shove_sh, same start | the walk-up Shove kills A's front on the first turn |
| `m45b_03_punish_walkup` | 3. punishing a walk-up Shove | fz_guard_sh v wk_shove_sh, L10 ×1.76, held 5 | A steps into B's wave to trade (positional fire on A), B dies within 10 turns |
| `m45b_04_retreat_loses` | 4. a retreat that loses access | fz_guard_sh v wk_back_sh, L10 ×1.76, held 12 | B takes no trade but cannot reach A's dying minions |
| `m45b_05_freeze_then_recall` | 5. a freeze kept until Recall | fz_guard_sh_sr v wk_shove_reset_sh, L4 ×1.76, held 12 | minion fire and B's resets wear A down; A Shoves and Recalls |
| `m45b_06_ward_favorable` | 6. a favourable Ward | g_shove_ward v g_farm, Vision forward, L4 | the ward placed with A's wave forward; protected turns spent exposed |
| `m45b_07_ward_interrupted` | 7. an interrupted Ward | g_ward v g_bully, Vision neutral, L4 | B reaches the cell next to W and interrupts the ward |
| `m45b_08_trade_big_wave_L2` | 8. an early trade into a large wave | first-turn Trade then g_farm v g_farm, trade bench L2, 24 HP | retaliation on A after the landed hit: tick 1 under revision 2, ticks 0 and 1 under the candidate |
| `m45b_09_trade_big_wave_L18` | 9. a late trade | the same at L18, 128 invested | the same fire is ≈ 1% of A's bar |
| `m45b_10_kill_bounce` | 10. an unexpected interaction | fz_guard v wk_lh, L10 ×1.76, held 12 | B dies; A's larger wave pushes and crashes while B is away; B farms the crash under its tower |

| demonstration | revision 2: seed, payoff to A, deaths A / B, minion cHP on A | candidate: the same |
|---|---|---|
| `m45b_01_freeze_held` | 87900, +5.42, 0 / 0, 0 | 87900, +2.15, 0 / 0, 0 |
| `m45b_02_freeze_broken` | 87900, +2.98, 0 / 0, 0 | 87900, +2.35, 0 / 0, 0 |
| `m45b_03_punish_walkup` | 87901, +6.54, 0 / 1, 200 | 87901, +4.84, 0 / 1, 200 |
| `m45b_04_retreat_loses` | 87900, +4.92, 0 / 0, 200 | 87900, +4.21, 0 / 0, 200 |
| `m45b_05_freeze_then_recall` | 87900, +0.92, 0 / 0, 576 | 87900, +0.21, 0 / 0, 576 |
| `m45b_06_ward_favorable` | 87900, +2.34, 0 / 0, 0 | 87900, +2.00, 0 / 0, 0 |
| `m45b_07_ward_interrupted` | 87900, −1.31, 0 / 0, 0 | 87900, +5.43, 0 / 1, 119 |
| `m45b_08_trade_big_wave_L2` | 87900, +3.19, 0 / 0, **93** | 87900, +3.02, 0 / 0, **324** |
| `m45b_09_trade_big_wave_L18` | 87900, +3.19, 0 / 0, 27 | 87900, +3.02, 0 / 0, 95 |
| `m45b_10_kill_bounce` | 87936, +1.42, 0 / 1, 425 | 87936, +2.04, 0 / 1, 468 |

All 20 files replay exactly. Recordings 8 and 9 show the calibration directly: the same landed hit draws retaliation
on both ticks under the candidate (324 cHP over the 12 turns at level 2, a third of the bar; 95 cHP, 7% of the bar, at
level 18). Single recordings are illustrations, not estimates: the payoff differences between the two
configurations are one seed each.

---

## 13. Compatibility, tests and runtime

- **Earlier presets are untouched.** The committed engine (`HEAD` = 8dc5acf) and the modified one produce
  byte-identical turn records, tick traces and observations on random games for 14 presets, `m45_unified` (revision 2)
  and `m45_unified_r1` included (`results/m45b/compatibility.json`); two of the digests are pinned in
  `tests/test_m45b_candidate.py`. Every earlier preset keeps `farm_spot_rule = "pre_crash_front"` and delay 1.
- **Engine changes.** `champions_m4.resolve_turn_m4` step 1 and `_tick_goals` gain the `"last_front"` branch
  (`_last_front_cell`), active only under that value; `GameConfig` validates it (natural waves and tick movement
  required; the legacy rulesets reject a non-default `farm_spot_rule` through `M4_ECONOMY_FIELDS`). No other engine
  code changed.
- **New preset** `GameConfig.m45_calibrated()` (`M45_CALIBRATED_FIELDS`); spaces unchanged (61 actions, 90
  observation entries); the checkpoint guard recognises it (`preset_name`) and refuses 88-input networks on it like
  on `m45_unified()`; the viewer offers it as its first configuration and registers the M4.5b starts.
- **Tests.** 82 infrastructure tests (`tests/test_m45b_calibration.py`: engine versus viewer replay, exact versus Monte
  Carlo, every strategy legal, state builders, power ratios, the holdout split, the probe recomputation, the probe
  override, viewer registration) and 21 candidate tests (`tests/test_m45b_candidate.py`: the contract, earlier presets'
  values, validation, the corrected defect cases, the correction's scope, retaliation timing, mirror equivariance on 400
  random states, an environment soak with the decoder, deterministic replay, the viewer, the guard, two pinned
  digests). Full regression suite: **1355 passed, 4 skipped, 2 xfailed** (the same three PettingZoo warnings as
  M4.5a; 15 min under load; `results/m45b/regression.log`), collected before the two pin tests were added; the M4.5 and
  configuration test files with the pins: 370 passed.
- **Throughput** (uniform random play, `LaneEnv`, best of 3 × 3,000 turns): `m45_calibrated` 3,436 turns/s,
  `m45_unified` 3,544 (0.97×), `m4_four_actions_econ_aggro` 3,884 (0.88×).
- **No result or recording of an earlier phase was overwritten**; `results/m45b/` is new.

---

## 14. Readiness for M4.5c

### **GO WITH CAVEATS** for fresh shared-policy PPO on `GameConfig.m45_calibrated()`

The mechanics are correct, deterministic and mirror-equivariant, and every earlier preset is byte-identical. The one
mechanics defect found (§8) is corrected. The integrated incentives now give the strategic structure the brief asks for:
- conditional freezes, with value mid- and late game;
- a stronger side that can punish a walk-up;
- weak-side replies that vary with the state, including resets;
- Vision that pays in some positions against some opponents;
- a step back that costs real resources;
- threat-only zoning near the weaker side's tower;
- early trades into large waves that are no longer free.

None of the caveats blocks training; caveat 1 needs a decision first.

| # | caveat | severity | evidence | likely effect on PPO | what M4.5c should track |
|---|---|---|---|---|---|
| 1 | **Mid-to-late minion share.** The holdout criterion HB4 fails: 21% of the trade damage dealt returns as minion fire at level 14 (baseline 8.6%) | moderate | §11.2 | mid-game trades into waves somewhat less attractive | **decide before training**: accept, or adopt `minion_damage_growth = 1.12` after a fresh validation (new seeds and states); a ≈ 10-minute rerun of the trade bench and the battery |
| 2 | Restricted scripted games are Shove-centred, on this preset and the historical one | moderate | §7; learned policies beat Shove by ≈ +3 natively | convergence toward Shove-heavy play is possible; M4.4b learned conditional Shove (36%) under the same economy | Shove share by lane state, trade frequency, the scripted Shove family as fixed evaluation opponents |
| 3 | Freezes are conditional and therefore rare: premium > +0.2 in 4 of 24 DEV and 4 of 16 holdout states | moderate | §3, §11 | few and short freezes, mostly mid- and late game | freeze frequency and duration, the frozen-on side's farm and XP at the walk-up cell, guard trades |
| 4 | A single trade's economic value is dominated by wave timing (one tick of diverted fire moves G20 by ±0.8) | moderate (learning signal) | §4.3 | noisy credit assignment for trades; slower learning of trade timing | the critic's error on trade turns, trade frequency by wave size and level |
| 5 | Vision is net negative in half of the tested contexts; its magnitude depends on duration | low–moderate | §5 | PPO may ward rarely; ward-baiting a chaser is available | ward rate by position and opponent, interruptions, protected exposed turns; 8 turns is the pre-measured alternative |
| 6 | The probe still assumes full support (7.5% of scripted farm out of reach, mostly ordinary positioning) | low | §8 | small farm losses in mutual pushes | out-of-reach farm (the timing module measures it) |
| 7 | Threat-only zoning is marginal in the neutral lane (−0.4 ± 0.4); it holds near the weaker side's tower (−2.1) | low | §6.3, §11 | threat-based zoning mostly in defensive lanes | step-back frequency under threat, the stronger side's denial |
| 8 | The kill bounce: a kill during a freeze can lose the wave state | low | §3.1, recording 10 | an intended dynamic that PPO must learn to convert | post-kill wave state, Shove / Recall after kills |
| 9 | Every strategic conclusion is relative to scripted strategies (and one out-of-distribution learned reference) | methodological | §1, §7.2 | — | M4.5c's own counterfactuals on learned policies |

**Prerequisites for M4.5c.**

1. The decision on caveat 1.
2. The M4.4 / M4.4c diagnostics adapted to W and minion targeting (planned), plus the M4.5b ledger quantities in the
   training health checks: out-of-reach farm, freeze phases, wards and exposure, minion fire by cause.
3. Fixed evaluation opponents from M4.5b (g_shove_sr, wk_shove_reset_sh, fz_guard_sh, g_shove_edge, g_shove_ward)
   alongside self-play and historical snapshots.

Nothing in M4.5b requires a change to the PPO trainer, the observation or the action space.

---

## 15. Reproduction

```bash
python -m lanegame.experiments.m45b split                     # the holdout split (written once)
python -m lanegame.experiments.m45b trades freeze freeze_local freeze_sweep vision vision_sweep zoning timing matrix
python -m lanegame.experiments.m45b trades_natural sensitivity sensitivity_cal candidate
python -m lanegame.experiments.m45b validate demos reference report
python -m pytest tests/test_m45b_calibration.py tests/test_m45b_candidate.py
python -m pytest                                             # the full regression suite
streamlit run lanegame/viewer/app.py                         # Recorded trajectory: results/m45b/trajectories
```

Runtime on 11 worker processes: trades 1.2 min, freeze grid 7.4, freeze local / reset 4.3, freeze sweep 6.5, Vision
6.4, Vision sweep 5.6, zoning 1.5, timing 0.2, matrix 11.8, sensitivity 7.2, natural trades 3.1, corrected sensitivity
11.1, candidate battery 26.8, holdout validation 24.7, demonstrations and reference 1.3: about 2 hours of stages
(≈ 220 million simulated turns), plus exploratory runs. Every stage writes its own JSON; `report` regenerates
`results/m45b/report.md` from them in a second.
