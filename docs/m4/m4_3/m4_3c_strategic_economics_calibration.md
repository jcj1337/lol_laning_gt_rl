# M4.3c — Stabilization of the Four-Action Model, Visual Debugger and Economic Calibration

**Status.** Stage A (Shove terminology, debugger, mixed-action sequences) is IMPLEMENTED and validated. Stage B
(economics) is a DIAGNOSIS plus an EXPERIMENTAL CANDIDATE preset, `GameConfig.m4_four_actions_econ()`, offered for
review. Nothing was adopted as a default: `m4_lasthit()` (the research preset), `m4_four_actions()` (the four-action
baseline) and every other preset are unchanged, and every trained checkpoint keeps its environment. Rules are cited
from the [mathematical reference](../laning_game_mathematical_reference.md) ("Reference §n"); the four-action model is
Reference §4.10, the new opt-in rules Reference §8.7.

Branch `development/m4-3b-validation` (on top of `4bd116c`, uncommitted working tree). Every number below was produced
by the engine through the real environment (`LaneEnv`), with scripted strategies on paired seeds, unless it is labelled
PPO (§10, exploratory). Reproduction: §13.

---

## 0. Executive findings

**What prevented strategically meaningful freezing, zoning and alternative wave management?** Not deaths, not XP,
and not the four wave actions. Two fixed rates of the lane economy did, and they act together:

1. **The crash tax.** Kills in the defender's own tower cell are collected at `collect_contest_own_tower` = 0.6
   (0.9 elsewhere). Every crashed wave therefore destroys a third of its gold for the defender, while the pusher farms
   the defender's waves at 0.9 near the enemy tower. Against a perma-Shove the defender's collection efficiency falls
   from 0.80 to 0.65 of the available farm: about 3 resources per 40 turns, for as long as the shoving lasts. With
   plate gold (3 per tower, finite) on top, Shove beats Last Hit by +6.9 per 40 turns at equal power and is the best
   reply of **both** players in **every** tested state (nine scenarios, levels 3–10, power gaps from ×0.57 to ×2.04).
   Removing plate gold changes no equilibrium; raising the own-tower rate to 0.8 changes several.
2. **Break-even threats.** A connected trader collects at `trade_farm_factor` = 0.5. Chasing a walk-up at ×1.76
   attack power kills it about once per 40 turns and costs it 2.2 farm, but costs the chaser 2.6 farm: harassment
   pays nothing, so a power lead can be monetized only through Shove's larger capacity (an investment lead of +5 is
   worth +3.9 over 100 turns in a mutual-Shove lane and +0.1 for a trader).

**Hypotheses of the brief.**

| hypothesis | verdict | evidence |
|---|---|---|
| deaths are too inexpensive | **true in proportion, but not the bottleneck** | a kill pays 0.5 waves (League ≈ 2.4); yet a 2.0 bounty, a 5- or 7-turn or a level-scaled absence change **no** equilibrium alone (values move ±0.1): equilibrium play dies 0.05 times per 40 turns (§5) |
| XP denial has too little economic weight | **rejected as a lever** | an XP channel in the reward with any weight 0–½ changes no equilibrium; XP is lost almost only to absence, and the level cap makes the XP channel reward *gaining* levels, not *being ahead* (§6) |
| combat threats do not influence lane control | **true, for an economic reason** | walking up to a ×1.76 chaser **is** dangerous (1–2 deaths and 3–4 bars per 40 turns), but creating the danger costs the chaser as much gold as it costs the victim (§5) |
| the continuing horizon and tower plates distort Shove | **secondary** | plates are 1–5% of income over 300 turns; a terminal at the last plate changes no equilibrium and mainly cuts the leader's ongoing income (§7) |

**What fixes it, in the scripted games.** Only the *combination* of a smaller crash tax with cheaper threats
produces state-dependent play; neither does it alone. The selected candidate sets four fixed rates to
League-proportional values: own-tower collection 0.8, trade farm factor 0.8, kill bounty 2.0, and a death absence of
3 / 4 / 5 turns at levels 1–6 / 7–12 / 13–18. In the restricted games:

- **equal power:** both sides still contest (mutual Shove) in every state, early and mid game;
- **far behind (×1.76, ×2.04):** the weaker side's best single reply to a perma-shover is to walk up and **Thin**
  (defend the wave) instead of shoving back (52–61% of its equilibrium on two independent seed sets); playing safe
  (respect) is as good as breaking, not better, so "retreat" is not established (§8.1);
- **far ahead:** the stronger freezer keeps the freeze and chases while it holds (100% of the freeze-phase games, both
  seed sets), and the lane leader chases (27–31%);
- **power persists through trading:** an investment lead of +5 / +50 now buys +1.5 / +3.6 over 100 turns to a
  trader (≈ 0 before), a level lead +0.8 to +1.6;
- **Shove still breaks bad states:** a weak or an equal freezer is broken by Shove (100% of the full games; 41–42% of
  the weak-freezer phase games, where walking up is the alternative), and Shove remains the stronger side's default.

A bounded PPO pilot (EXPLORATORY, 2 seeds × 2M transitions per configuration, §10) agrees: on the baseline the
learned policy shoves 70–75% of the time, and 85–92% even while holding a freeze; on the candidate it uses all four
actions (Shove 31–34%), shoves only 13–18% while holding a freeze against 49–54% when frozen on, plays Safe twice as
often and dies half as often.

**Recommendation.** The four-action mechanics are ready (Stage A). The economy of the four-action **baseline** is not
ready for a strategic PPO experiment: its only rational strategy is Shove, and PPO learns exactly that. The candidate
is mechanically sound (identical wave physics, unchanged spaces), produces state-dependent play in scripted games and
in the pilot, and is offered for review as the economy of the next PPO experiment: `m4_four_actions_econ()` against
`m4_four_actions()`, 3 seeds, with the M4.3b instruments generalized to four wave slots (§12).

---

## 1. Scope and what changed

| item | change | where |
|---|---|---|
| A1 Shove terminology | slot SH of the four-action model is **Shove** (`config.W_SHOVE` = `W_CLEAR`, label `SH`); three-action presets keep Clear; codes, masks, budgets and checkpoints unchanged | `config.py` (`FOUR_WAVE_NAMES/SHORT/DISPLAY`, `wave_display_names`), the label test |
| A2 debugger | the four-action preset (default in the UI), derived scenarios, a B-ahead gap, four-action strategies and eight example matchups, a Wave management tab (budget calculated vs HP applied, per tick), wave damage on the lane picture | `lanegame/viewer/{session,policies,app,render}.py` |
| A3 sequences | a four-action strategy library (state rules, turn schedules, memoryless Recall rules, phase strategies), a paired-seed harness through `LaneEnv`, the sequence study | `experiments/m43c/{strategies,harness,sequences}.py` |
| B instruments | restricted matrix games, phase games, threat diagnosis, power / death / horizon probes, report tables, PPO pilot | `experiments/m43c/{economics,economics_probes,calibration_report,pilot}.py` |
| B engine opt-ins | `xp_reward_weight`, `death_turns_by_level`, `terminal_rule` (all off by default; M4 ruleset only) | `config.py`, `champions_m4.py`, `env/core.py`, `env/discounted.py` |
| B candidate | `GameConfig.m4_four_actions_econ()` | `config.py` |
| docs | this report; Reference §0 labels, §3.1, §4.10, §5.3, §6.5, §8.5–§8.7, glossary, §13; a terminology note in the four-action redesign report | `docs/m4/` |

---

## 2. Stage A — the finalized four-action model

### 2.1 Shove terminology (A1)

| slot | code range | three-action presets (`m4_lasthit()`, …) | four-action model (`m4_four_actions()`) | budget |
|---|---|---|---|---|
| 0 `SP` | 0–44 (w = 0) | Hold `HO` | **Last Hit** `LH` | $\min(1,\lfloor C/8\rceil)=1$ |
| 2 `F` | 0–44 (w = 2) | Thin `TN` | **Thin** `TN` | $\min(\lfloor C/2\rceil,1+\text{need}_0)$ |
| 3 | 46–60 | — | **Slow Push** `SL` | $\min(\lfloor C/2\rceil,1+\text{need}_1)$ |
| 1 `SH` | 0–44 (w = 1) | Clear `CL` | **Shove** `SH` | $C$ |

The rename is semantic only. `W_SHOVE` *is* `W_CLEAR` (the scripted game's original Shove slot), so no code, mask,
budget or checkpoint changes meaning; tests pin that Shove spends $C$ on every state and that the three-action labels
are byte-identical. Historical reports and result files keep the names they were written with (the four-action
redesign report says Clear; a note at its top says so).

### 2.2 The visual debugger (A2)

```bash
streamlit run lanegame/viewer/app.py
```

The existing Streamlit debugger (`lanegame/viewer`) was extended rather than replaced; every number it shows is still
read from the environment's `info`, its states or the engine's tables, and a recording is still a pure function of
(setup, actions).

| what | where in the UI |
|---|---|
| configuration `m4_four_actions` (first in the list; the first example matchup opens on it) | sidebar |
| scenarios: `freeze_early` (the freeze at level 4), `freeze21` (an oversized 21-HP wave held next to A's tower), `oversized` (B's 24-HP front at the lane cell); gap `b_large` (B ×1.76: A is the weak side) | sidebar |
| four-action strategies (names ending in 4, e.g. `grow_shove4`, `freeze_hunt4`, `pz_back`), with a note per turn naming the rule that fired | Policy A / B |
| example matchups 4A-1 … 4A-8 (below) | "Example matchup" |
| manual play with all four wave actions, each labelled with the HP it would spend this turn | Manual play |
| per side: capacity $C$, own / enemy blob, window strengths, Thin and Slow Push needs, the **calculated** budget, the HP **applied** per tick and in total, mode, reach line, why a budget stayed unspent | tab "Wave management" |
| level and this frame's wave damage above each champion (`L10 · Slow Push 2/2`) | lane picture |
| a removed-of-budget event per side and turn | event lists |
| turns and ticks, positions, packets and reinforcements, contact, crashes, tower damage and plates, gold / XP / HP / invested power, Recall and death absence | unchanged panels |

| matchup | what it shows |
|---|---|
| 4A-1 Slow Push → Last Hit → Shove | one 2-HP nudge, the lead compounding under Last Hit, a 19-HP crash, a plate, then a death to a gank while sieging |
| 4A-2 Thin an oversized wave, then hold | the 21-HP wave next to A's tower: equal power crashes anyway; with the gap `large` Thin removes 6, 6, 6, 5, 2, 1 HP and the freeze holds 9 of 12 turns |
| 4A-3 Slow Push against Thin | the stationary 8 ↔ 11 cycle; only HP loss and Recalls move the lane |
| 4A-4 Slow Push, then Shove, against Thin | the schedule switches at turn 10 and crashes at turn 14 |
| 4A-5 Freeze → Recall → Return | three Recalls while the freeze holds; the freeze survives every absence |
| 4A-6 Crash, then Recall | a Recall timed by the own crash |
| 4A-7 Strong freezer against a walk-up | a ×1.76 freezer in its tower cell never trades: the tower-cell contact rule (Reference §6.1) |
| 4A-8 Weak champion against a big wave | Thin capped at 2–3 HP lets a 24-HP wave crash |

Tests (`tests/test_viewer.py`: six new test functions, and the parametrized ones now cover the eight four-action
matchups too; 52 tests): every four-action matchup replays exactly from its actions, its tick traces
are the authoritative lane step with the recorded state-dependent budgets, the panel's budget is the engine's and its
applied damage equals the trace's and the lane events' (never above the budget), manual play accepts all four actions
(and the 46-code presets refuse Slow Push), the scheduled strategies replay, and the picture labels the wave damage.
The app was run headlessly (Streamlit `AppTest`) in every mode and on every four-action matchup without an exception.

### 2.3 Mixed-action sequences (A3)

`python -m lanegame.experiments.m43c sequences` (`results/m43c/sequences.{json,md}`): 31 cells, 64 paired seeds each
through the environment plus one deterministic run per cell. Capacities evolve with levels and investment as in play.

| sequence (A vs B, start) | measured (40 turns unless stated; deterministic run in brackets) |
|---|---|
| Slow Push → Last Hit → Shove vs Last Hit (neutral L3) | first crash at turn 15.6 (15), 4.7 crashes of 11.6 HP, 1.28 plates, A − B +3.25 gold. The Last Hit phase is free: once ahead, Slow Push already spends 1 HP, so Slow Push and Slow Push → Last Hit spend the same |
| the same vs Thin | 0.4 crashes, −0.01 gold: Thin removes every nudge |
| Slow Push vs Thin (neutral L6) | 0 / 0 crashes deterministically, contact stays at the lane; with chance 0.5 / 0.8, only after HP loss and Recalls |
| Slow Push 10 turns, then Shove, vs Thin | first crash at turn 15.5 (14), 10.8 crashes, +4.72 gold: the pusher adapts by shoving |
| Thin vs a 21-HP wave next to A's tower (L10, 20 turns) | B crashes 3 (equal) / 2 (A ×1.76) times; A's damage never exceeds 1 + need; the freeze holds 2.2 / 12.3 turns |
| Freeze → Recall → Return (freeze L10) | 4.2 Recalls while the freeze holds; held 27.7 turns (40 of 40 deterministically) against 16.1 without Recalls |
| weak vs strong (neutral L10; A ×0.57, equal, ×1.76) | Last Hit spends 1 HP and Slow Push crashes first at turn 16 deterministically at every gap (15.6–17.4 with chance); Shove's crashes and applied damage grow with power (15.9 / 16.0 / 18.4 crashes, 120 / 120 / 130 HP) |

```
4A-1, seed 3 (global cells; A / B: wave HP applied/budget)
turn -4         -3         -2         -1         +0         +1         +2         +3         +4
0    .          .          A8         .          A8 B8 a b  .          B8         .          .
1    A8         .          .          .          A12 B11 a b.          .          .          B8           SL2/2 / LH1/1
2    .          .          A8         .          A9 B7 a b  .          B8         .          .            LH1/1 / LH1/1
5    A8         .          .          .          A14 B11 a b.          .          .          B8           LH1/1 / LH1/1
9    A8         .          .          .          A17 B8 a b .          .          .          B8           LH1/1 / LH1/1
13   A8         .          .          .          A23 B3 a b .          .          .          B8           LH1/1 / LH1/1
14   .          .          A8         .          .          A21 a      B4 b       .          .            SH2/5 / LH0/1
15   A8         .          .          .          A8         a          [A19] b    .          B8           SH1/5 / LH1/1  A crash 19
```

Shove's budget is mostly unspent once the enemy wave is dead (113 of 217 HP applied over 40 turns): overkill is
lost, so the applied damage, not the budget, is what moves the lane.

**Two findings for Stage B came out of these runs.** (i) A defender gains from a crash it does not prevent: when B's
21-HP wave crashes into A's tower, A collects it under its tower and earns its XP, so Last Hit out-earns Thin there
(+6.1 against +3.7 gold over 20 turns, +53 XP). (ii) The inherited low-HP pre-rule of the scripted policies never
recalls in contact with an enemy at the adjacent cell, so a low-HP champion can stand at its own −1 for several turns
(the 4A-3 trace with seed 1); `Strategy(recall_in_contact=True)` removes this, the default keeps the historical rule.
No engine bug was found.

### 2.4 Stage A acceptance

| criterion | status |
|---|---|
| all four actions execute correctly | yes: the redesign's 16 tests plus the new budget and panel tests |
| Shove terminology integrated | yes: labels, viewer, traces, docs; historical results keep Clear |
| mixed-action sequences work through the environment | yes: §2.3, 11 tests in `tests/test_m43c_stabilization.py` |
| the debugger supports the new action space | yes: §2.2 |
| action codes and masks | unchanged; the redesign's encoding and legality tests pass |
| legacy presets and checkpoints in their original environments | yes: the legacy digests, the M4 digests and the 46-output refusal test pass |
| reproducible traces | yes: deterministic replays and seeded recordings are pinned |

Full suite after Stage A: **1014 passed, 4 skipped, 2 xfailed** (974 before; the same skips and expected failures).

---

## 3. Baseline economics of the four-action model (B1)

### 3.1 Instruments

**Restricted zero-sum matrix games** (`experiments/m43c/economics.py`). From a start state, every strategy of A's set
plays every strategy of B's set: 64 paired games of 100 turns per cell through the environment (common random
numbers: game $e$ of every cell uses the same seed), so a matrix holds 24–64 cells and 1,536–4,096 games. The matrix
of mean values to A is solved as a zero-sum game for three objectives: **J** = the discounted gold difference over
100 turns (the CURRENT reward, γ = 0.99), **Jx** = J plus 1/8 per XP of the discounted credited-XP difference (§6), and
**G40** = the undiscounted gold difference over the first 40 turns (the scenario-local economics). Reported: the
value, both equilibrium mixtures, best replies, the weight B puts on conceding strategies (step back, respect) and on
the conditional one (cautious), and the weight A puts on chasing strategies.

An equilibrium of such a game is an equilibrium **of the restricted game only**: a best reply inside a handful of
scripts is not a best response of the laning game, and a mixed equilibrium is a statement about these scripts.

**Scenarios** (starts of the M4.2e-B harness generated under `m4_lasthit()`, as in the four-action redesign study;
champions at full HP, no gold, the level and power gap set as stated):

| | start | levels | gap (A over B) |
|---|---|---|---|
| F4, F10 | the freeze: B's wave held at A's −1, A's small wave in A's tower cell | 4 / 10 | equal |
| F10m, F10g, F10gl | the freeze | 10 | ×1.24 (16 v 4 invested); ×1.76 (50 v 0; duel ×3.1); +3 levels and 50 (×2.04; duel ×4.2) |
| F10b | the freeze | 10 | the freezer is the weak side (B 50 invested) |
| N3, N10, N10g | the neutral lane: both 8-HP fronts and both champions at the lane cell | 3 / 10 / 10 | equal / equal / ×1.76 |
| P4 … P10b | the freeze, **phase games** (§3.5) | as F4 … F10b | as F4 … F10b |

**Strategies** (`experiments/m43c/strategies.py`; every one Recalls at 40% HP and resets while the enemy is away, the
M4.2e-B pre-rules):

| set | strategies |
|---|---|
| freezer A | `farm4` Last Hit · `freeze4` Thin (keeps a freeze) · `freeze_hunt4` Thin + chase when the duel edge is positive · `freeze_bully4` Thin + Trade every turn · `shove4` Shove + Pressure |
| frozen-on B | `walk_up4` Last Hit · `walk_up_thin4` Thin · `walk_up_shove4` Shove + Pressure (break) · `step_back4` Safe · `respect4` Safe when the enemy starts in reach, else Yield · `cautious4` Safe when a stronger enemy is in reach · `contest4` Thin + trade back · `hunt4` chase |
| neutral (both seats) | `farm4`, `thin4`, `slowpush4`, `build_shove4` (Slow Push, Shove at 20 HP), `shove4`, `hunt4`, `shove_hunt4`, `respect4` |

`hunt` is new: zoning_natural's `punish` and `edge` need contact at the **start** of the turn, which a champion in its
own tower cell never has with an enemy one cell ahead (Reference §6.1), so a freezer can only threaten a walk-up by
chasing.

### 3.2 The baseline equilibrium is Shove, everywhere

| scenario | value to A (J) | A's equilibrium | B's equilibrium |
|---|---|---|---|
| F4 freeze, L4, equal | +0.61 | Shove | break (walk up + Shove) |
| F10 freeze, L10, equal | +1.10 | Shove | break |
| F10m ×1.24 | +4.62 | Shove | break |
| F10g ×1.76 | +7.25 | Shove | break |
| F10gl ×2.04 | +7.34 | Shove | break |
| F10b the freezer is weak | −3.31 | Shove | break |
| N3, N10 equal | −0.19, +0.01 | Shove | Shove |
| N10g ×1.76 | +6.50 | Shove | Shove |

Every equilibrium is pure and every best reply of both sides to every strategy is Shove (or the Shove break). The
same holds for G40 and for Jx. Power changes the value, never the strategy: the weaker side's best reply to a ×2.04
opponent is to shove back.

### 3.3 Why: the crash tax and the plates

Shove against Last Hit at the neutral lane, level 10, equal power, 40 turns (64 games):

| | A (Shove) | B (Last Hit) |
|---|---|---|
| gold earned | 20.24 | 13.38 |
| farm / plates / bounty | 17.24 / 3.00 / 0.00 | 13.15 / 0.00 / 0.23 |
| collection efficiency (farm ÷ farm at full rate) | 0.82 | **0.65** (0.80 when both last-hit) |
| crashes (HP) | 15.8 (133) | 0 |
| deaths / Recalls / turns away | 0.47 / 1.64 / 4.55 | 0.00 / 0.28 / 0.56 |
| HP lost (bars) | 2.06 (ganks 2.42 before mitigation) | 0.18 |

The shover earns +6.86 in 40 turns although it is away 4.6 turns (11% of the time) and dies 0.47 times, because the
defender collects every crashed HP in its own tower cell at 0.6. Isolating the two channels (same seeds):

| variant (one change) | A − B gold (40 turns) | J (100 turns) | B's collection efficiency | equilibria |
|---|---|---|---|---|
| baseline | +6.86 | +9.25 | 0.65 | Shove everywhere |
| `pg0`: no plate gold | +3.83 | +6.44 | 0.65 | **unchanged** (Shove everywhere) |
| `ot80`: own-tower collection 0.8 | +3.65 | +4.11 | 0.81 | far behind, B defends with Thin instead of shoving back (F10g, F10gl, N10g) |
| `ot90`: own-tower collection 0.9 | +2.05 | +1.53 | 0.89 | mixed in every gap scenario; crashes become gifts to the defender (Thin, which prevents the enemy's crash, then loses 1.3 to Last Hit) |

So the persistent driver is the crash tax (about 0.15 resources per turn of shoving, 3 per 40 turns), and plates are a
finite one-off (3 resources per tower, all three gone within 35 turns against a passive opponent). The own-tower rate
is a knife edge: at 0.6 crashing taxes the defender a third of the wave, at 0.9 a crash is a free gift (full collection
plus the XP of the whole crashed wave), at 0.8 a crash still costs the defender a ninth.

### 3.4 What gold and power buy: transient versus persistent

The value of a power lead (`econ_probes`, neutral lane L10, both seats the same strategy, 100 turns, 64 paired games;
A's discounted gain over the equal start, ± 95%):

| lead of A | passive (`farm4`) | mutual Shove (`shove4`) | trader (`hunt4`) | `shove_hunt4` |
|---|---|---|---|---|
| +2 invested | +0.13 ± 0.22 | +1.97 ± 0.59 | −0.06 ± 0.43 | −0.46 ± 0.73 |
| +5 | +0.32 | +3.90 | +0.08 | −0.32 |
| +20 | +0.47 | +6.38 | −0.30 | +1.38 |
| +50 | +0.72 | +6.77 | +0.25 | +2.59 |
| +1 level | −0.02 | +2.05 | −0.19 | −1.37 |
| +3 levels | −0.00 | +3.74 | −0.63 | −2.09 |

Gold is only worth something if it is invested and the power is then *used*. A passive lane does not use power, and a
trading lane cannot profit from it (§5). Only Shove converts power into income: a larger capacity $C$ wins the
mutual-Shove race, and the loser's crashes pay the crash tax. An investment lead of 5 resources (eleven turns of lane
income) is worth 3.9 over 100 turns: a persistent, compounding advantage, but through one channel only.

### 3.5 Freeze-phase games

A strategy fixed for 100 turns ("freeze for ever", "always step back") is a whole-game commitment, not a response to a
freeze. The phase games therefore offer each side a freeze-specific rule **only while the freeze holds** and Shove (the
baseline equilibrium strategy) otherwise: A `pz_keep` / `pz_hunt` / `pz_bully` (keep the freeze, alone / chasing /
trading every turn) or `shove4`; B `pz_walk` / `pz_thin` / `pz_back` / `pz_cautious` / `pz_contest` or `shove4`
(break at once).

| scenario | baseline: A | baseline: B |
|---|---|---|
| P4, P10 equal | Shove | Shove (break) |
| P10g ×1.76 | **hunt during the freeze 0.79**, Shove 0.21 | walk up and Thin 0.58, Shove 0.42 |
| P10gl ×2.04 | Shove | Shove |
| P10b the freezer is weak | Shove 0.66, keep 0.34 | walk up 0.52, Shove 0.48 |

So a freeze has some value at baseline, for a stronger freezer that chases while the freeze holds. What the baseline
lacks is in the full games: the weaker side's answer to a stronger opponent is always to shove back, and power buys
income only through Shove.

---

## 4. Parameter experiments

Workflow of the brief: diagnose (§3), isolate one change at a time, then combine only the levers that moved something,
then check every scenario. 23 configurations (`economics.VARIANTS`: the baseline, 12 single changes of
`m4_four_actions()`, 10 combinations), each played on the same 14 scenarios (432 + 120 cells, 64 games of 100 turns
each, the same seeds): 3.5 million game turns per configuration. Values were chosen for a reason, not swept:

| variant | change | why this value |
|---|---|---|
| `k1`, `k2` | kill bounty 1.0, 2.0 | one and two waves; a League kill is about 2.4 early waves (0.5 now) |
| `d5`, `d7` | death absence 5, 7 turns | the brief's candidates (3 now) |
| `dS1` | absence $2+\lceil L/6\rceil$ = 3 / 4 / 5 | League's respawn grows from 10 s to 52 s, plus a walk back of about 30 s: ≈ 2.7 / 4 / 5 turns of 15 s |
| `tf80`, `tf100` | trade farm factor 0.8, 1.0 | a trade costs a fifth / none of the turn's farm (half now); 0.75 is not an exact ledger multiple |
| `ot80`, `ot90` | own-tower collection 0.8, 0.9 | DIAGNOSTIC of the crash tax (0.6 now; 0.9 = the lane rate) |
| `pg0` | no plate gold | DIAGNOSTIC of the plates |
| `rc3` | a Recall is 3 turns away | League's recall channel plus the walk back is about 2.5–3 turns (2 now) |
| `E1` | terminal at either tower's last plate | the PROPOSED end of laning of the economic design document (§7) |
| combinations | `ot80`/`ot90` with `tf80`/`tf100`, then with `k2`, `dS1`, `rc3` | the only isolated levers that moved an equilibrium, then the amplifiers |

### 4.1 Scorecard

Equilibrium weights of the restricted games (objective J; `results/m43c/calibration.md` has every table, Jx and
G40 too). "Wary" = B's weight on conceding (step back, respect) or conditional (cautious) strategies; "A freezes" =
A's weight on freeze strategies; the phase-game columns read the same weights in the freeze-phase games of §3.5.

| variant | B contests at equal power (F10; P10) | B wary vs ×1.76 / ×2.04 (full; phase) | A freezes when ahead ×1.76 / ×2.04 (full; phase) | value of a ×1.76 lead (freeze / lane) | distinct equilibria |
|---|---|---|---|---|---|
| `base` | 1.00; 1.00 | 0.00 / 0.00; 0.00 / 0.00 | 0.00 / 0.00; 0.79 / 0.00 | 6.15 / 6.49 | 4 |
| `k1`, `k2`, `d5`, `d7`, `dS1` | 1.00; 1.00 | 0.00 / 0.00; ≤ 0.03 | 0.00 / 0.00; 0.42–0.61 / 0.00 | 6.10–6.32 / 6.45–6.56 | 4 |
| `tf80`, `tf100` | 1.00; 1.00 | 0.00 / 0.00; ≤ 0.11 | 0.00 / 0.00; 0.36–0.37 / 0.00 | 6.15 / 6.49 | 4 |
| `rc3` | 1.00; 1.00 | 0.00 / 0.00 | 0.00 / 0.00; 0.46 / 0.00 | 5.65 / 5.93 | 4 |
| `pg0` | 1.00; 1.00 | 0.00 / 0.00 | 0.00 / 0.00; 0.46 / 0.00 | 3.98 / 4.35 | 4 |
| `ot80` | 1.00; 1.00 | 0.00 / 0.00 (Thin) | 0.00 / 0.00; 0.40 / 0.00 | 3.54 / 3.70 | 6 |
| `ot90` | 1.00; 1.00 | 0.00 / 0.00; 0.20 / 0.00 | 0.00 / 0.00; 0.40 / 0.00 | 2.01 / 1.69 | 7 |
| `E1` | 1.00; 1.00 | 0.00 / 0.00 (Thin) | 0.00 / 0.00; 0.38 / 0.00 | 4.88 / 5.00 | 6 |
| `ot80_tf80` | 1.00; 1.00 | 0.00 / 0.00; 0.09 / 0.00 | 0.03 / 0.01; 0.37 / 0.00 | 3.61 / 3.73 | 6 |
| `ot80_tf100` | 1.00; 1.00 | **0.44** / 0.00 | 0.03 / 0.01; 0.43 / **1.00** | 3.67 / 3.78 | 8 |
| `ot90_tf100` | 1.00; 1.00 | 0.38 / 0.00 | 0.12 / 0.11; 1.00 / 1.00 (also at equal power) | 2.23 / 2.14 | 9 |
| `ot80_tf100_k2` | 1.00; 1.00 | 0.29 / 0.00 | 0.06 / 0.04; 1.00 / 1.00 | 3.79 / 3.85 | 8 |
| `ot80_tf100_dS1` | 1.00; 1.00 | 0.45 / 0.00 | 0.03 / 0.02; 1.00 / 1.00 | 3.69 / 3.74 | 7 |
| `ot80_tf100_rc3` | 1.00; 1.00 | **0.78** / 0.00 | 0.08 / 0.00; 0.40 / 0.00 | 2.71 / 3.42 | 5 |
| `ot80_tf80_k2` | 1.00; 1.00 | 0.42 / 0.00 | 0.06 / 0.04; 1.00 / 1.00 | 3.70 / 3.76 | 7 |
| `ot80_tf80_dS1` | 1.00; 1.00 | 0.00 / 0.00; 0.06 / 0.00 | 0.03 / 0.02; 0.39 / 0.00 | 3.63 / 3.70 | 6 |
| **`ot80_tf80_k2_dS1` = candidate** | **1.00; 1.00** | **0.48** / 0.00 | 0.08 / 0.05; **1.00 / 1.00** | 3.68 / 3.74 | **8** |
| `ot80_tf100_k2_dS1` | 1.00; 1.00 | 0.27 / 0.00 | 0.08 / 0.06; 1.00 / 1.00 | 3.80 / 3.84 | 7 |

Readings:

1. **Every isolated death, bounty, trade-cost and Recall change is inert** (equilibria unchanged, values ±0.1):
   equilibrium play neither trades nor dies, so the price of a death or of a trade is never paid.
2. **The crash tax is the only isolated lever** (`ot80`, `ot90`): it decides whether the weaker side shoves back or
   defends. Removing it entirely (`ot90`) makes crashes a gift and the lead worth a third of its baseline value.
3. **Threats need both**: cheaper trades matter only once the crash tax no longer makes Shove the answer to
   everything (`ot80_tf100`: the weak side defends or plays safe, the far-ahead freezer harasses during the freeze).
4. **The death economy becomes an amplifier once threats are used**: with trades at 0.8, neither `k2` nor `dS1`
   alone restores the freeze-phase harassment that `tf100` produces, but together they do (1.00 at ×1.76 and ×2.04).
   The "wary" weights of 0.27–0.78 in this column are the respect strategy, which on fresh seeds turns out to be tied
   with breaking (§8.1): read them as "conceding is no longer worse than shoving back", not as a robust concession.
5. **Equal power keeps contesting** in every variant (F4, F10, P4, P10, N3, N10), and the weaker freezer is still
   broken by Shove.

---

## 5. Combat threat and death (B3)

### 5.1 Is walking up to a stronger champion dangerous?

The frozen-on champion B against three freezers, baseline, 40 turns (64 games; HP lost in bars of B's maximum, by
source before mitigation):

| scenario | A | B | B deaths | B Recalls | B HP lost | trades / ganks | A trades connected | A farm | B farm | A − B gold |
|---|---|---|---|---|---|---|---|---|---|---|
| F10 equal | freeze4 | walk up | 0.08 | 0.80 | 0.78 | 0.00 / 0.79 | 0 | 16.27 | 13.78 | +2.50 |
| F10 equal | freeze_hunt4 | walk up | 0.36 | 1.95 | 2.00 | 1.30 / 0.86 | 11.7 | 14.09 | 12.00 | +2.22 |
| F10 equal | freeze_bully4 | walk up | 0.59 | 2.25 | 2.66 | 2.25 / 0.61 | 24.1 | 5.89 | 8.85 | −3.02 |
| F10g ×1.76 | freeze4 | walk up | 0.08 | 0.69 | 0.78 | 0.00 / 0.79 | 0 | 16.46 | 13.60 | +2.89 |
| F10g ×1.76 | freeze_hunt4 | walk up | **1.02** | 1.75 | 2.94 | 2.47 / 0.66 | 12.7 | 13.86 | 11.40 | +2.96 |
| F10g ×1.76 | freeze_bully4 | walk up | **1.98** | 1.28 | 3.91 | 3.69 / 0.55 | 24.2 | 8.14 | 8.92 | +0.11 |
| F10g ×1.76 | freeze_hunt4 | break (Shove) | 1.72 | 2.86 | 4.59 | 3.71 / 1.71 | 20.7 | 9.73 | 14.33 | −6.32 |
| F10g ×1.76 | freeze_hunt4 | step back | 0.00 | 0.28 | 0.25 | 0.10 / 0.14 | 2.2 | 14.49 | 9.16 | +5.32 |

**Yes, the danger is real**: against a ×1.76 chaser the walk-up dies once or twice per 40 turns and loses 3–4 bars,
almost all of it to trades. Against a passive freezer only ganks hurt it (0.8 bars, 0.08 deaths), whatever the gap.
**But the danger does not pay its creator**: chasing costs the freezer 2.6 (hunt) to 8.3 (bully) farm, as much as the
victim loses, so A − B gold barely moves (+2.89 → +2.96). And breaking by Shove still pays B more than anything else
(A − B −6.32) although B then dies 1.7 times.

### 5.2 The price of a death

Paired games from the same state, B alive or B just killed (absent d turns, then a normal return); the bounty
included; J over 100 turns (`econ_probes`, 64 games, ± 0.3–0.7):

| state | d = 1 | d = 3 (current) | d = 5 | d = 7 | candidate, d = 3 |
|---|---|---|---|---|---|
| neutral lane L10, both Shove | 2.58 | 2.96 | 4.90 | 6.53 | 3.27 |
| freeze L10, A freezes, B walks up | 0.56 | 0.88 | 1.14 | 1.27 | 2.03 |
| freeze L10, A freezes, B breaks | 1.02 | 1.75 | 2.64 | 3.43 | 2.73 |
| freeze L10 ×1.76, A hunts, B walks up | 0.44 | 0.47 | 0.61 | 0.26 | 1.55 |

A death costs what the opponent can do with the absence: about one resource per extra turn in a shoving lane (the
opponent crashes and takes plates), almost nothing in a freeze (the frozen wave waits next to the freezer's tower).
Against a ×1.76 hunter, being away is barely worse than walking up into it. So the timer is not the bottleneck: the
death is rare in equilibrium (§4), and when it happens its price depends on the wave state far more than on its length.

### 5.3 What makes the threat credible

The chain is: power → profitable harassment → forced absences → the absence's price. At baseline the second link is
missing (trade farm factor 0.5); the third and fourth then never come into play. With trades at 0.8 and League-priced
deaths (bounty 2, absence 3 / 4 / 5) the chain closes: harassment by a stronger champion pays, the victim's absences
are worth something, and the weaker side stops shoving back into a stronger opponent (it defends the wave; playing
safe becomes as good as breaking). No zoning rule was added: every change of behaviour in §8 is a choice of the
scripted strategies between consequences that the engine computes (trades, tower shots, ganks, deaths, absences,
collection by position).

---

## 6. XP as an economic objective (B2)

Implemented as the EXPERIMENTAL opt-in `xp_reward_weight` λ (Reference §8.7): $r_A=(\Delta G_A-\Delta G_B)+\lambda(\Delta
\tilde X_A-\Delta\tilde X_B)$ with $\tilde X$ the credited XP (zero beyond the cap), exact in ledger units, zero-sum and
mirror-equivariant (tests), XP never converted into gold, investment untouched. Because the reward does not change the
dynamics of scripted agents, every weight is scored on the **same** games (the stored discounted XP channel).

| scenario | λ = 0 | λ = 1/16 | λ = 1/8 | λ = 1/4 | λ = 1/2 |
|---|---|---|---|---|---|
| F4 (freeze L4) | +0.61 | +1.37 | +2.13 | +3.65 | +6.68 |
| F10 | +1.10 | +1.53 | +1.96 | +2.81 | +4.53 |
| F10g ×1.76 | +7.25 | +7.77 | +8.29 | +9.33 | +11.42 |
| F10gl ×2.04 (+3 levels) | +7.34 | +4.51 | +1.69 | −3.97 | **−15.29** |
| N10 | +0.01 | +0.00 | −0.00 | −0.02 | −0.04 |
| N10g ×1.76 | +6.50 | +6.54 | +6.58 | +6.66 | +6.82 |

(values of the restricted games; **every equilibrium is the same pure Shove profile at every weight**.)

- **XP changes no strategy.** On a five-cell lane with a three-cell XP range, XP is lost almost only to absence, and
  the XP difference of a game follows the gold difference (it adds value to whoever already wins).
- **The level cap distorts it.** A champion three levels ahead reaches level 18 first and then credits nothing, while
  its rival still levels: over 100 turns the XP channel favours the *trailing* champion (F10gl flips sign at λ = 1/4).
  The incremental XP channel rewards *gaining* XP, not *having* it; with both champions at the cap by turn ≈ 142 in
  every matchup (§7), it vanishes for most of a long game. A level-progress weight (value per XP ∝ 1/n(L)) would
  soften the late-game overvaluation but not the cap.
- **A transient vs a persistent lead.** A level lead is worth +2.1 to +3.7 of gold in a mutual-Shove lane over 100
  turns (§3.4), more than its XP value at λ = 1/8: the gold channel already prices levels by what they win.

**Not adopted**: the candidate keeps the gold-only reward. The opt-in remains for a later objective study.

---

## 7. The laning horizon (B5)

From the standard start, 300 turns, 32 games per pair (`econ_probes` horizon; median turns over the games in which the
event happened, the level at the event in brackets; so a rare third plate can have a smaller median than the second):

| A v B | 1st plate | 2nd plate | 3rd plate | first tower's last plate | both level 18 | plate share of income |
|---|---|---|---|---|---|---|
| Last Hit v Last Hit | 185 (L17) | 273 (L18) | never | never | 144 | 0.6% |
| Shove v Shove | 45 (L10) | 88 (L14) | 118 (L16) | 142 (78% of games) | 142 | 2.5% |
| Shove v Last Hit | 14 (L5) | 27 (L7) | 35 (L8) | 35 | 142 | 1.3% |
| Slow Push v Thin | 48 (L10) | 226 (L16) | 224 (L18) | 224 (6%) | 142 | 0.2% |
| Shove + hunt v Shove + hunt | 37 (L8) | 53 (L10) | 72 (L12) | 100 | 148 | 4.8% |
| candidate, Shove v Shove | 48 | 84 | 123 | 132 (72%) | 141 | 2.5% |

1. **Plates are early or never.** A dominant pusher takes a tower's three plates by turn ≈ 35 at level 8; balanced
   pushers by turn ≈ 120–140 at level 16–18; passive lanes almost never. Levels are a clock: both champions reach 18 at
   turn 141–148 in every pair, so "late" laning is always a level-18 duel.
2. **Farm, not plates, is the reward.** Plates are 0.2–4.8% of income over 300 turns. Removing plate gold changes no
   equilibrium (§3.3). Tower mechanics do not make Shove dominant; the crash tax (a collection rule) does.
3. **The discount does not favour plates.** With γ = 0.99 the crash tax's 0.15 resources per turn is worth about 15
   discounted resources, five times a tower's plates. Truncation (training games of 512 turns, bootstrapped) and
   evaluation windows (W = 1000) are properties of runs, not of the game; with an exact critic they leave the
   objective unchanged (Reference §9.2).
4. **A natural terminal does not help.** `E1` (`terminal_rule = "final_plate"`, implemented with correct terminal
   semantics: no bootstrap, plate gold of the final turn paid) changes no equilibrium except that far-ahead games
   end early: Shove v Last Hit is worth +5.08 instead of +9.25, because the shover's last plate stops its own income
   advantage. That is the "rational prolongation" the economic design document predicted: under E1 a leader should
   *avoid* the last plate, and a trailer cannot end the game by taking the leader's. Passive lanes never end.

**Recommendation for the horizon**: keep continuing time with γ = 0.99 read as the exogenous end of laning. Termination
at the last plate is implemented as an opt-in but not recommended. If a shorter laning phase is wanted, the evidence
points at the level clock (levels 15–18 arrive at turns 85–140 whatever the play) rather than at tower damage; that is
a separate redesign (§11).

---

## 8. Contesting versus conceding (B4)

The equilibria of the baseline and of the candidate in every scenario (J = 100 turns discounted; G40 = the first 40
turns; strategy names without the suffix 4):

| scenario | baseline (J): A / B | candidate (J): A / B | candidate (G40): A / B |
|---|---|---|---|
| F4 freeze L4, equal | Shove / break | Shove / break | Shove / break |
| F10 freeze L10, equal | Shove / break | Shove / break | Shove / break |
| F10m ×1.24 | Shove / break | Shove / break | Shove / break |
| F10g ×1.76 | Shove / break | Shove 0.92, freeze + bully 0.05, freeze + hunt 0.03 / **walk up + Thin 0.52, respect 0.48** (fresh seeds: Thin 0.56, break 0.44) | Shove / break |
| F10gl ×2.04 | Shove / break | Shove 0.95, freeze + hunt 0.05 / walk up + Thin 0.61, break 0.39 | Shove / break |
| F10b the freezer is weak | Shove / break | Shove / break | Shove / break |
| N3, N10 equal | Shove / Shove | Shove / Shove | Shove / Shove |
| N10g ×1.76 | Shove / Shove | Shove 0.73, Shove + hunt 0.27 / build then Shove 0.79, Shove 0.21 | Shove + hunt / Shove |
| P4, P10 equal (phase) | Shove / Shove | **freeze + hunt (P4) / freeze + bully (P10)** / Shove | the same |
| P10g ×1.76 (phase) | freeze + hunt 0.79, Shove 0.21 / walk up + Thin 0.58, Shove 0.42 | **freeze + bully 1.00** / Shove | freeze + hunt / Shove |
| P10gl ×2.04 (phase) | Shove / Shove | **freeze + bully 1.00** / Shove | freeze + hunt / Shove |
| P10b weak freezer (phase) | Shove 0.66, keep 0.34 / walk up 0.52, Shove 0.48 | Shove 0.58, freeze + hunt 0.42 / walk up 0.59, Shove 0.41 | Shove 0.60, keep 0.40 / cautious 0.64, Shove 0.36 |

Against the intended properties (hypotheses to evaluate, not outcomes to manufacture):

| property | baseline | candidate |
|---|---|---|
| contesting is rational at equal power | yes (by shoving back) | yes, in every equal scenario, early (L3–4) and mid (L10) |
| contesting becomes dangerous against a much stronger opponent | dangerous (§5.1) but still the best reply | the weak side stops shoving back at ×1.76–×2.04 and defends the wave with Thin, over the 100-turn objective (robust, §8.1); within 40 turns breaking is still its best reply |
| retreating can become the rational response to a credible threat | never | **not established**: respect (Safe while the enemy is in reach) ties with breaking against a ×1.76 shover (48% on one seed set, 0% on the other) |
| a champion ahead uses its advantage for lane control | only through Shove | through Shove and through harassment: the stronger freezer keeps the freeze and harasses while it holds (100% in the phase games), the lane leader chases (27%) |
| Shove still breaks unfavourable wave states | always | yes: it is the frozen-on side's reply at equal power and against a weak freezer (41–100%), and every side's default |
| the strategy depends on the state | 4 distinct equilibrium profiles | 8 |
| a stronger champion has no guaranteed right to zone | — | the weak side's mixture keeps a third to half of its weight on contesting strategies (walk up + Thin, break) at every gap |

Two limits of this evidence. The weak side's responses are *full-game* strategies (Thin or respect for 100 turns), so
the reading is "the weak side defends against a stronger perma-shover", not "the weak side gives up a freeze". And the
phase games put the freeze-holder's harassment in equilibrium at equal power (P4 on both seed sets, P10 on one), but
the frozen-on side breaks at once there, so the freeze rarely lasts long enough for that harassment to happen.

### 8.1 Robustness

Thirteen scenarios replayed on fresh seeds (root 76,500) with 128 games per cell (`economics_{base,candidate}_r2`):

| finding | 64 games (root 76,000) | 128 fresh games (root 76,500) | robust? |
|---|---|---|---|
| baseline: pure Shove in every full game and P4, P10, P10gl | yes | yes | **yes** |
| baseline P10g: the ×1.76 freezer chases during the freeze | hunt 0.79 | bully 0.70 | **yes** (in kind) |
| candidate F10g/F10gl: the weak side defends with Thin instead of shoving back | 0.52 / 0.61 | 0.56 / 0.59 | **yes** |
| candidate F10g: the weak side plays safe (respect) | 0.48 | 0.00 (break 0.44 instead) | **no**: respect and break are tied (A − B 4.81–4.85 against 4.71–5.13, ± 0.15–0.3) |
| candidate P10g/P10gl: the stronger freezer keeps the freeze and harasses | bully 1.00 / 1.00 | hunt 1.00 / 1.00 | **yes** (in kind) |
| candidate P4: the equal freezer chases | hunt 1.00 | hunt 1.00 | yes |
| candidate P10: the equal freezer harasses | bully 1.00 | Shove 1.00 | **no** (values 1.07 against 0.95: noise) |
| candidate N10g: the lane leader chases | 0.27 | 0.31 | **yes** |
| candidate: equal power contests, a weak freezer is broken | yes | yes | **yes** |

So the robust picture under the candidate is: the weaker side **defends the wave** (walks up and Thins) rather than
shoving back, the stronger side **keeps a freeze and chases** while it holds and chases in the lane, and equal
champions contest. Retreat as a pure concession (respect) is *as good as* breaking for the weak side, not better.

---

## 9. Selected configuration: `GameConfig.m4_four_actions_econ()` (EXPERIMENTAL CANDIDATE)

`m4_four_actions()` plus:

| field | four-action baseline | candidate | physical reading | measured role |
|---|---|---|---|---|
| `collect_contest_own_tower` | 0.6 | **0.8** | last-hitting a crashed wave under one's tower loses a ninth of the lane rate instead of a third | removes the state-independent crash tax that made Shove every side's best reply (§3.3); keeps a crash costly for the defender, so pushing when ahead still pays (a ×1.76 lead is worth +3.7, against +2.0 at 0.9) |
| `trade_farm_factor` | 0.5 | **0.8** | a connected trader loses a fifth of the turn's farm (it already deals no wave damage and suspends its objective) | makes a stronger champion's harassment pay (§5.1); with 0.5 every chase was break-even |
| `kill_bounty` | 0.5 | **2.0** | a kill pays two waves (League ≈ 2.4 early waves) | inert alone (§4); prices the deaths that harassment now produces |
| `death_turns_by_level` | 3 at every level | **3 / 4 / 5** at levels 1–6 / 7–12 / 13–18 ($2+\lceil L/6\rceil$) | League's respawn grows with level | inert alone; with the bounty, closes the threat chain (§5.3) |
| `champion_absent_max` | 3 | 5 | the observation's absence scale | none (needed by the schedule) |

Formulas changed (Reference §5.3, §6.5, §8.7): the collection rate of a kill at the own tower cell is
$\min(b_\mu(x),\,0.8)$; a connected trader's rate is multiplied by 0.8; the opponent of a champion that dies receives
2.0; the dead champion is away $2+\lceil L/6\rceil$ turns, $L$ its level at the start of the turn of its death. Nothing
else changes: the wave mechanics and the four actions (a deterministic run is identical turn by turn to the baseline's;
test), the gold-only reward, XP, levels, investment, towers, plates, continuing time, the 61-code action space and the
88-entry observation (whose absence entry is now $\alpha/5$, so networks must be retrained).

Comparison with the baseline (equal-power neutral lane L10, the equilibrium cell, 40 turns): income 16.2 / 16.3
per side (15.7 / 15.8 baseline), deaths 0.08 / 0.05, Recalls 0.8 / 0.9, no trades. The candidate does not turn the
equal lane into a trade-fest: at equal power hunting a passive farmer is about break-even (+0.11 per 40 turns, −0.67
at baseline), mutual hunting lowers both incomes (14.2 / 14.4) and is not an equilibrium. A death costs 2.0–3.3
resources (about 0.9–3.0 at baseline). Plates fall at the same turns (§7).

The value of a power lead under the candidate (§3.4's probe): an investment lead of +2 / +5 / +20 / +50 is worth
+0.84 / +1.46 / +1.48 / +3.55 to a trader (`hunt4`; about 0 at baseline) and +0.79 / +1.92 / +3.73 / +3.90 in a
mutual-Shove lane (+1.97 / +3.90 / +6.38 / +6.77 at baseline); a level lead of +1 / +3 is worth +0.84 / +1.55 to a
trader. Power therefore converts into lane income through two channels instead of one, and less explosively through
Shove. In a passive lane a level lead is worth −0.1 (the leader's deaths last longer under the schedule).

---

## 10. PPO pilot (EXPLORATORY)

`python -m lanegame.experiments.m43c pilot` (`results/m43c/pilot.json`; checkpoints in the git-ignored
`results/rl/checkpoints/m43c_pilot/`). Shared-policy self-play exactly as M4.3b's best condition `ent03`
(`train_shared`, PPOConfig.continuing: γ 0.99, GAE 0.95, 16 environments × 128-turn segments, games of 512 turns
bootstrapped, entropy 0.03), **2 seeds × 2,048,000 transitions** per configuration, fresh 61-output networks.
Evaluated on 32 self-play games of 300 turns from s0 per run (the policy's own sampling) and 32 paired games (both
seats) against six scripted strategies. **A pilot of two seeds is not a convergence or equilibrium result**; it
shows whether learned play reacts to the economy at all.

| | baseline s0 / s1 | candidate s0 / s1 |
|---|---|---|
| training: entropy at the start → last 10 updates | 2.85 → 1.05 / 1.17 | 2.85 → 1.75 / 1.75 |
| explained variance (last 10 updates) | 0.61 / 0.60 | 0.46 / 0.54 |
| wave mix, all states: Last Hit / Thin / Slow Push / **Shove** | 11 / 8 / 6 / **75%** · 12 / 8 / 10 / **70%** | 23 / 20 / 26 / **31%** · 27 / 19 / 20 / **34%** |
| neutral (fronts in contact at the lane): Shove | 81% · 82% | 32% · 38% |
| **while holding a freeze** (we hold the enemy front at our −1): Shove | **85% · 92%** | **13% · 18%** (Thin 31 / 25%, Slow Push 36 / 28%, Last Hit 21 / 30%) |
| **while frozen on**: Shove | 64% · 80% | 49% · 54% |
| behaviours: Hold / Yield / Trade / Safe / Recall | 36 / 17 / 33 / 10 / 4% · 36 / 23 / 30 / 7 / 4% | 24 / 21 / 29 / 19 / 7% · 24 / 20 / 29 / 21 / 6% |
| Trade rate when the duel edge is positive / negative | 41 / 23% · 39 / 18% | 41 / 17% · 41 / 17% |
| deaths / Recalls per side per 300 turns | 6.1 / 9.5 · 4.6 / 9.8 | 2.7 / 13.2 · 2.6 / 12.6 |
| crashes per side; connected trades per side | 33 / 62 · 31 / 55 | 34 / 47 · 37 / 47 |
| plates 1 / 2 / 3 (median turn) | 47 / 66 / 85 · 36 / 62 / 88 | 31 / 50 / 72 · 32 / 61 / 77 |
| J against `shove4` / `farm4` / `freeze_hunt4` (300 turns, ± 95%) | +3.0 / +10.2 / +11.0 · +3.0 / +11.0 / +11.3 | +3.9 / +3.9 / +6.7 · +3.2 / +4.0 / +7.1 |

Readings (exploratory):

- **Baseline: learned play is Shove.** Three quarters of all wave actions, and the freezer itself shoves 85–92% of the
  time when it holds a freeze: the learned policy finds the only rational strategy of the baseline economy.
- **Candidate: a state-dependent mix.** The same training produces all four wave actions, and the freeze contrast
  reverses: while holding a freeze the policy shoves 13–18% of the time (it keeps the freeze with Thin, Slow Push and
  Last Hit), while frozen on it shoves 49–54% (it breaks). It plays Safe twice as often and dies about half as often
  (2.6–2.7 against 4.6–6.1 per 300 turns), as a League-priced death should make it.
- **Caveat: convergence.** The candidate's policy stayed less concentrated (entropy 1.75 against 1.1), which alone
  spreads a mix over actions; it cannot explain the reversal between freezing and frozen-on states, but longer runs
  and more seeds are needed before any strategic reading. Margins over the scripted references are smaller under the
  candidate (+3–7 against +3–13), so its economy is also less exploitable by a single pattern, or its policy is
  weaker; two seeds cannot separate these.

---

## 11. Remaining issues and limitations

1. **Restricted games.** Every equilibrium above is an equilibrium of small scripted strategy sets. The concessions
   found are full-game strategies (respect for 100 turns); freeze-specific responses were tested in phase games with
   Shove as the default. A learned policy can find strategies outside these sets.
2. **The horizon of the concession.** Under the candidate the weak side's caution against a ×1.76 opponent pays over
   the 100-turn discounted objective, not over the first 40 turns (where breaking by Shove still wins). The cost of
   contesting a stronger champion accumulates through absences and deaths; a policy has to value that horizon.
3. **Harassment from one's own tower at equal power.** In the candidate's phase games the freeze-holder harasses even
   at equal power (it fights next to its own tower, where the attacker cannot answer without tower shots). The
   frozen-on side breaks at once, so it is rarely played out; worth watching in PPO.
4. **Contact geometry.** A champion in its own tower cell is in contact only with a champion in the same cell
   (Reference §6.1), so a freezer threatens a walk-up only by chasing out of its tower. Not changed (it is a
   positioning rule, outside this study), but it is why `punish` never fires for a freezer.
5. **XP.** XP denial is negligible on a five-cell lane with a three-cell XP range (the frozen-on side at its −1 still
   earns all XP but behind-tower kills), and the level is a clock (level 18 at turn 141–148 in every pairing). An
   XP channel in the reward changes no strategy and is distorted by the cap (§6). A meaningful XP economy would need a
   geometry change (lane length or XP range), which the earlier XP-range screen rejected for other reasons.
6. **The crash-tax value.** 0.8 sits inside a knife edge: at 0.6 a crash taxes the defender a third, at 0.9 it is a
   gift. Only 0.8 and 0.9 were tested (0.75 and 0.85 are not exact ledger multiples with the zone factors); the choice
   is a calibration judgment (League's under-tower last-hitting loses some, not a third).
7. **Scripted pre-rules.** The inherited low-HP rule can leave a champion in contact at its −1 without recalling
   (§2.3); `recall_in_contact` exists, the studies keep the historical default.
8. **The jungle.** Ganks are a fixed hazard by exposure, independent of the opponent; most of the pusher's risk comes
   from them, not from the defender. The defender's own counterplay (trading from its tower) only pays once trades are
   cheap (§5).
9. **Termination.** `terminal_rule = "final_plate"` is implemented and tested but not recommended (§7): it ends
   dominant games early and gives a leader a reason to avoid the last plate.
10. **Instruments for PPO.** The M4.3b behaviour and wave-only intervention tools (`experiments/rl/m43b/behavior.py`,
    `intervene.py`) still assume three wave slots; the pilot uses its own four-slot counters (§10).
11. **Noise.** 64 paired games per cell give ± 0.3–0.7 on a cell's value; a mixed equilibrium's weights move with
    such noise. §8.1 reruns the key scenarios on fresh seeds with 128 games.

---

## 12. Recommendation

1. **Stage A is complete**: the four-action model with Shove terminology, the debugger and the sequence harness are
   ready for use.
2. **Do not run strategic PPO on the four-action baseline** expecting freezes, concessions or zoning: in its economy
   Shove is the best reply in every tested state for both sides, so the incentives contain nothing else to discover.
3. **Next step: a PPO experiment on `m4_four_actions_econ()`**, with `m4_four_actions()` as the control, M4.3b
   settings (entropy 0.03), 3 seeds × 2M transitions, after generalizing the M4.3b behaviour and intervention tools to
   four wave slots. Read: the wave mix by situation and by power gap, conditional trading, freeze holds, concession
   when behind, and the wave-only counterfactuals.
4. **Keep the objective gold-only and the game continuing** (no XP channel, no terminal state) for that experiment;
   both opt-ins stay available.
5. **Further mechanical work, if PPO confirms the scripted picture is too thin:** the contact geometry around the own
   tower (§11.4) and the XP geometry (§11.5) are the two remaining structural reasons why a freeze denies little.

---

## 13. Reproduction and files

```bash
python -m lanegame.experiments.m43c sequences --workers 10            # Stage A3, 10 s
python -m lanegame.experiments.m43c economics --workers 10            # every variant, ~3.5 min each (~80 min)
python -m lanegame.experiments.m43c econ_probes --variants base candidate ot80_tf100 --workers 4
python -m lanegame.experiments.m43c calibration_report                # results/m43c/calibration.md
python -m lanegame.experiments.m43c pilot --variants base candidate --seeds 0 1 --workers 4   # ~1 h
python -m pytest tests/test_m43c_stabilization.py tests/test_m43c_economics.py tests/test_viewer.py
```

Tests at completion: `python -m pytest tests` → **1027 passed, 4 skipped, 2 xfailed** (974 before this work; the same
skips and expected failures; 53 new tests).

Seeds: SeedSequence roots 75,000 (sequences), 76,000 (matrix games), 76,500 (robustness rerun), 77,100–77,102
(probes), 78,000–78,001 (pilot evaluation); pilot training seeds 0 and 1. Disjoint from every earlier study.

| file | content |
|---|---|
| `lanegame/config.py` | `W_SHOVE`, four-action names and display names, `wave_display_names`; fields `xp_reward_weight`, `death_turns_by_level`, `terminal_rule` with validation; preset `m4_four_actions_econ()` (`M4_FOUR_ECON_FIELDS`, `DEATH_TURNS_S1`) |
| `lanegame/champions_m4.py` | `M4Context.xp_units`, `death_turns`; the XP reward channel, the level schedule and the terminal flag (`M4Turn.xp_reward_units`, `terminal`) |
| `lanegame/env/core.py`, `env/discounted.py` | `terminated` from the terminal rule, `info["xp_reward_units"]`, `info["terminal"]`; the truncation and per-turn bounds count the XP channel and the death schedule |
| `lanegame/viewer/{session,policies,app,render}.py` | the four-action debugger (§2.2) |
| `lanegame/experiments/m43c/strategies.py` | the four-action strategy library and phase strategies |
| `lanegame/experiments/m43c/harness.py` | the paired-seed harness and per-turn ledger |
| `lanegame/experiments/m43c/sequences.py` | Stage A3 |
| `lanegame/experiments/m43c/economics.py`, `economics_probes.py`, `calibration_report.py`, `pilot.py` | Stage B |
| `lanegame/experiments/m43c/__main__.py`, `four_actions.py` | the new stages; a terminology note |
| `tests/test_m43c_stabilization.py`, `tests/test_m43c_economics.py`, `tests/test_viewer.py`, `tests/test_m43c_four_actions.py`, `tests/test_m4_config.py` | new and extended tests (the last: the three new fields in the legacy / m2c rejection table) |
| `results/m43c/sequences.*`, `economics_*.json.gz`, `econ_probes_*.json`, `calibration.md`, `pilot.json` | results |
| `docs/m4/laning_game_mathematical_reference.md`, `docs/m4/m4_3/m4_3c_four_action_wave_redesign.md`, `docs/README.md` | documentation |


