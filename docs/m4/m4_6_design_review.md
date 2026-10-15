# M4.6 — Design review: freezes, zoning and the price of contesting

*2026-10-15. Status: **PROPOSED**. Nothing in this document is implemented: no preset, parameter, test, training run,
result or roadmap entry changed, and the mathematical reference is untouched. It is an independent assessment written
from the code of `GameConfig.m45_calibrated()`, the mathematical reference
([laning_game_mathematical_reference.md](laning_game_mathematical_reference.md) §4–§8, §14–§15), the M4.5b calibration
report, the M4.4c strategic investigation, the M4.5c research report and its generated tables. It is written to be
read without the repository: every number names its source, and the terms are defined once below.*

**Terms.** One *resource* = the gold of one fresh 8-HP wave at full collection; a side's income is capped at 0.45–0.5 per
turn by the spawn rate. A *turn* is one simultaneous decision by both champions (two movement ticks; a wave every two
turns, so a turn is roughly 15 seconds of League). Wave actions: *Last Hit* (1 HP), *Thin* and *Slow Push* (state-dependent
partial clears), *Shove* (the whole capacity C = round(4·AP*) HP, at least 4 at level 1, 6 at level 10). Behaviours:
*Hold* (stand at the farm spot, never evade), *Yield* (stand off, evade unless already in contact), *Trade* (chase and
attack), *Safe* (fall back to the own half, evade, reduced collection), *Recall*. A *freeze* is a state: the enemy wave
held at the freezer's −1 by an even minion exchange, the freezer's small wave in its tower cell. *×1.76* is the attack-power
ratio of a champion with about 48 resources invested against one with none; because investment also divides damage
taken, its duel ratio is ×3.1. *A bar* is a champion's maximum HP.

---

## 0. Summary

**Diagnosis.** The two concerns in the brief, that a freeze is broken by one Shove at little cost and that combat
strength rarely becomes resource denial, have one root: **contesting a resource costs one turn of exposure, and one turn
of exposure is cheap.** Three rules of the current model make it so.

1. **A contest resolves in one turn.** C is at least 4 HP from level 1 and 6 at level 10, while a freeze is held by an
   own wave of 8–11 HP that the held enemy wave already hits for 3 HP per turn. A walk-up Shove therefore kills the
   freezer's wave on the turn it is played (M4.5b candidate freeze grid: a passive freeze is held 0 turns against a
   walk-up Shove in 20 of 24 cells, 2 turns in the other 4).
2. **One turn of exposure costs at most about a fifth of a bar.** A connected trade is one exchange of 0, 1 or 2 units
   at hit probability 0.55: 12% of a bar at equal power, 21% at ×1.76. A gank at the walk-up cell occurs with probability
   0.10–0.12 and deals 43% of a bar on average. At the measured price of HP (1–2 resources per bar, M4.5b §4.3) the
   break turn costs 0.2–0.5 resources and earns +0.18 of immediate gold plus the freed lane.
3. **Disengagement is always available.** Safe evades any Trade when an out-of-contact cell lies within its two-cell
   move, and is otherwise hit only with probability 0.15; Recall from the own half refills everything in two turns for
   about one resource and invests the gold on top. Threat converts into economy only when the victim *chooses* to stay (a kill), to step back (Safe rates)
   or to reset (absence), and the victim picks the cheapest of the three each turn.

The aggressor pays every turn it threatens: 80% farm while connected, half its Shove budget, the gank term of the chase
cell, the victim's wave's retaliation when its hit lands, positional fire when the chase puts it inside the held wave
(which the chase rule and the tower-cell contact rule force in the freeze geometry), and the tower if it dives. A
sustained threat is therefore a wash in resources unless it kills, and a kill needs the victim to stay below the lethal
line by choice.

This is mostly a property of the **rules and their scale**: the three clocks of a turn (a wave dies in one turn, a bar
takes 5–8 turns of connected trades to remove, a bar is refilled in two turns) are not mutually consistent, and the
action set has a poke but no commitment. It is partly a **parameter** property (the same rules at a higher per-turn
combat magnitude would behave differently). It is only secondarily a **learning** property: the learning-free evidence is
M4.5b's restricted games, in which no freezer and no bully is in the stronger side's equilibrium among 24 scripted
strategies at ×1.76; plain Shove and edge-trading Shove are. PPO found the same thing from the other direction.

**But the zoning evidence is incomplete.** M4.5c tested only the indiscriminate bully plan (Trade every turn, dives
included), which M4.4c had already shown to lose on the earlier model; it did not re-run M4.4c's selective *Threaten*
plan (+1.13 over farming at ×1.76 there). It measured denial as "gold left uncollected when the stronger side is near",
the posture channel that M4.4c §5.5 showed to be the wrong one (conversion ran through kills and forced Recalls, "denial
by absence"). The learned stronger side's absolute lead at ×1.76, its kills and the Recalls it forced were not recorded.
So "combat strength does not translate into lane control" is not yet established for selective threats on the current
model. The attackers point the same way: they beat every final by +1.02 by trading a third more and freezing more, so the
finals concede more than the game requires, in places specific to each network.

**Recommendation.** Do not train and do not change mechanics yet. Run a diagnosis stage (M4.6a, about six hours of
compute, all on existing infrastructure) built around one decisive test: learned best responses trained from power-gap
and freeze starts against the finals. If they convert ×1.76 into much more than the finals do, the binding gap is the
training distribution (M4.6c: diverse starts and a style population). If they do not, and the exact ledger confirms the
one-turn break and the sub-scale combat clock, screen two coherence levers on the scripted harness (M4.6b): the per-turn
combat magnitude (`trade_exchanges = 2`, already implemented and off) and the price of a reset. Only after that a fresh
PPO run (M4.6d) on the M4.5c pipeline, unchanged.

---

## 1. What the evidence says

All values are resources; "vs" means the named candidate minus the policy's own choice, or the named side's lead.

**Freezes**

| finding | value | source |
|---|---|---|
| keep plan (Thin + Hold while held) vs learned opponent / vs back-off opponent, natural states | −0.20 ± 0.07 / **+0.56 ± 0.09** (6/6 finals) | m4_5c_research_report §5.2 |
| keep plan vs back-off, constructed grid | **+1.86 ± 0.22** (6/6) | §5.2 |
| frozen on: one break vs the policy / accepting (walk up, step back) / three Shoves | −0.02 / −0.22, −0.20 / −0.30 | §5.2 |
| exact immediate value of the frozen-on Shove over the modal action | +0.18 | §5.7 |
| freezes per 1000 turns, denial share, length (16.384M finals) | 2.7, 0.69, 5.5 turns | results/m45c/final/report.md §2 |
| passive freeze held against a walk-up Shove, candidate grid | 0 turns in 20 of 24 cells, 2 in the other 4 | results/m45b/report.md, candidate freeze grid |
| guard − passive against a walk-up Shove, ×1.76, held 5 HP, L4 / L10 | +0.80 ± 0.50 / +0.98 ± 0.66 | m4_5b_calibration_report §3.3 |
| freeze premium (freezing vs the best non-freezing strategy), ×1.76, level 10 | 0.00 … +0.33 (split-sample −0.15 … +0.18); ≤ 0.2 in 4 of 6 cells | §3.3 and grid |
| stronger side's equilibrium strategy at ×1.76 | plain Shove carries the majority in 8 of 12 large-gap cells; a guarding or bullying freezer in 4 | candidate freeze grid |
| weaker side's best reply to a guarded freeze at ×1.76 | Shove, then Recall below 60% HP, in 6 of 12 cells (every small-held-wave cell); a plain or wary Shove in 4; Yield or Last Hit in 2 | §3.4 and grid |

**Zoning**

| finding | value | source |
|---|---|---|
| bully ×10 at ×1.76 vs learned: neutral / A pushing / B pushing | −0.23 / −0.99 / +0.15 | m4_5c_research_report §5.3 |
| bully ×10 at ×1.76 vs `g_wary` (steps back when catchable) | +0.26 / −0.43 / **+1.28 ± 0.32** | §5.3 |
| bully ×10 at equal power vs learned | −0.39 / −1.36 / −1.74 | §5.3 |
| learned stronger side at ×1.76: position, Trade rate, posture denial | +2.0 cells, 0.26, 0.014 per turn | results/m45c/final/report.md §2 |
| threat-only zoning (A guards, B wary), ×1.76, per 50 turns: neutral / A's push / B's push | −0.43 ± 0.44 / +0.78 ± 0.31 / **−2.10 ± 0.37** | m4_5b_calibration_report §6.3 |
| one turn of Safe instead of Hold | −0.15 … −0.56 of farm | §6.1 |
| always Safe, 50 turns | −7.2 … −8.1 | §6.3 |
| earlier model (M4.4c, aggro): Threaten − Farm at ×1.76; learned stronger side's 20-turn lead; weaker side died | +1.13; +2.19; 45% | m4_4c_strategic_investigation §5.4 |
| earlier model: indiscriminate Bully − Farm at equal power | −2.05 (the bully dies in 81% of rollouts) | §5.4 |

**Costs and prices**

| quantity | value | source |
|---|---|---|
| one connected trade on a Hold defender | E[D] = 0.715 units: 12% of a bar equal, 21% at ×1.76; a double hit 33% / 59% | reference §6.2, §7.3 |
| price of HP | 1–2 resources per bar under a farming continuation | m4_5b §4.3 |
| gank at exposure +1 | p = 0.10–0.12 per turn, 2.6 units ≈ 43% of a bar; never lethal from full HP alone | reference §6.4–§6.5 |
| positional fire from a 12–15 HP wave | 3 units per turn: 17% of a bar at L4, 8% at L10, divided by the victim's M(S) | reference §14.6 |
| Recall | 2 turns, ≈ 1 resource of income, full HP, all gold invested | reference §6.6 |
| death | bounty 2.0 to the killer + 3 / 4 / 5 turns away | reference §8.7 |
| trader's farm while connected / Shove budget | ×0.8 / halved | reference §8.7, §14.7 |
| attackers vs their targets | +1.02 ± 0.48 (6/6); Trade 0.31 vs 0.26; freezes 4.6 vs 3.2 per 1000; target-specific | m4_5c_research_report §7 |
| natural power gaps / early levels in the training data | ≈ 0.1% / ≈ 10% of decisions | §10 |

---

## 2. The mechanics behind the findings

### 2.1 Breaking a freeze: the one-turn ledger

Set-up (reference §10.1; M4.5b §3.2): A holds B's 12–15 HP wave at A's −1; A's 8–11 HP wave meets it in A's tower cell; A
stands in its tower cell; B stands inside its own held wave at B's +1. Level 10; B has nothing invested; A is equal or
×1.76.

What happens when B plays Shove + Hold:

- **B's reach** from +1 covers +2, A's tower cell, where A's wave is. B's budget C = 6 (level 10) and the held wave's
  firepower 3 remove up to 9 HP: an 8–9 HP front dies on this turn, a 10–11 HP front on the next. The held wave then
  walks into A's tower and crashes; A collects it at the own-tower rate 0.8 (a crash is farm-neutral by the M4.3d
  calibration), B collects A's dying minions at 0.9, and B's next waves meet A's bounce near the lane cell. The freeze is
  over and the lane is reset.
- **B's cost if A guards** (A Trades): a champion in its own tower cell is in contact only with a champion inside that
  cell (reference §6.1), so A must step out, and the chase rule puts it at A's −1, inside the held wave. A's hit lands
  with probability 0.55 for 1.3 units at A's power: 12% of B's bar at equal power, 21% at ×1.76. B's gank exposure at
  +1 adds about 5% in expectation. B pays about 17% (equal) or 26% (×1.76) of a bar for the break turn, 0.2–0.5
  resources at the measured price of HP.
- **B's gain**: +0.18 of immediate gold over its modal action (exact one-turn mechanics), the freed lane, and the
  option to reset at about one resource whenever its HP is low.
- **A's cost of guarding**: 80% farm while connected (−0.09 on that turn), the held wave's positional fire on both
  ticks (8% of A's bar at level 10, 17% at level 4, divided by A's investment multiplier), a higher gank term at −1,
  and nothing from its Thin (the trade cap does not bind). M4.5b measured guarding at +0.8 to +1.0 over passive
  freezing against the walk-up Shove at ×1.76 with a small held wave, and ≈ 0 at equal power.

So the system **does** punish a weaker champion that contests a strong freeze, and the punishment is worth about one
resource per 100 turns to the freezer. The weaker side's best reply is nevertheless to break, resetting when low
(Shove with a Recall reset in 6 of the 12 large-gap cells, a plain or wary Shove in 4 more), because:

- the break completes before a second punishment can land;
- the HP it costs is refilled by a reset that costs about one resource and also invests B's gold;
- A's alternative to freezing, plain Shove with Pressure, earns A as much (plain Shove carries most of A's equilibrium
  in 8 of 12 large-gap cells; the freeze premium is 0.0–0.3). The strong side's lead at ×1.76 in those cells is +2.8 to
  +5.8 per 100 turns whatever A does.

Nothing in this chain is a defect; each step is the rule working as written. The *scale* is what makes the chain cheap:
in the time one wave takes to die, a champion loses at most a fifth of a bar, and a bar is restored in two turns.

### 2.2 Why threat rarely becomes denial

Zoning can only work through the victim's choices (reference §5.3: a champion's presence never reduces collection). The
channels and what each is worth:

| channel | what the victim loses | what forces it |
|---|---|---|
| Safe posture | 0.15–0.56 of farm per turn (reach 1; rates 0.6 / 0.4 / 0.2); XP only once it stands more than 2 cells from the kills | a Trade it cannot afford to take |
| absence (Recall) | ≈ 0.45 per turn for 2–3 turns, minus the investment it gains on return | HP below its own threshold |
| death | bounty 2.0 + 3–5 turns | staying below the lethal line (a double hit: 33% of a bar equal, 59% at ×1.76) |
| XP geometry | nothing from the lane cell: with XP range 2, a champion at its own 0 still receives the XP of kills in the enemy tower cell | being pushed to its own −1 |

What the threatener pays to keep the threat live, every turn: it must be in contact (one cell), which in the neutral
lane means leaving its farm spot when the opponent steps back (with the reach ball that is lost farm), in the enemy's
half means a dive (shot with probability 1.0), and in the freeze geometry means standing inside the held wave
(positional fire). Connected, it farms at 0.8 and its Shove is halved; if its hit lands, the victim's wave retaliates on
both ticks.

The result is the geometry pattern M4.5b measured and the M4.5c battery reproduced: threat-only zoning is worth −2.1 per
50 turns to the weaker side near the stronger side's tower (the freeze geometry), ≈ 0 in the neutral lane, and it costs
the threatener in the enemy's half. The bully plan's +1.28 against a wary opponent in B's push and its losses elsewhere
are the same pattern. **The two concerns share the cause:** the only geometry in which a stronger champion can deny
without fighting is the freeze geometry, and there the contest costs one turn.

### 2.3 The clocks of one turn

| clock | in the model | rough League equivalent per 15 s | consistent? |
|---|---|---|---|
| minion income | 4 HP = 0.5 resources per side, hard cap | half a wave | by construction |
| a champion clearing a wave | C = 4 (L1), 6 (L10), 9 (L18 unitemised), at most 23: 0.5–1.2 waves per turn | 0.3–0.5 wave early, about a wave mid-game | generous early, fine later |
| champion damage on a champion | one exchange: 12% of a bar equal, 21% at ×1.76; no option to commit more | poke 10–25%; a committed exchange from a stronger laner 40–100% | **low by 2–4× for committed play, and there is no commit option** |
| HP recovery | Recall 2 turns ≈ 1 resource, full HP, gold invested | back and walk 25–45 s ≈ 1–1.5 waves | consistent |
| death | ≈ 3.4–4.3 resources | 300 g + timer + walk ≈ 2.5–3 waves plus the lane | consistent |
| XP range | 2 of 5 cells | ≈ 1400 of ≈ 3500 units | consistent |
| gank | 0.10–0.15 per forward turn, 43–53% of a bar, never lethal alone | rarer, lethal when caught | frequent but mild |

The outlier is champion-on-champion damage per turn: the game has a poke and no all-in. A poke cannot zone a determined
opponent in League either; what zones is the threat of an all-in the victim cannot walk out of. Here the victim can always
walk out on the next turn (Safe evades; Yield is pinned only if it started in contact), so a threat is capped at one
exchange per engagement, and the engagement is re-priced from zero every turn.

### 2.4 What the game already does right

- Breaking a freeze is correctly the best reply when the freezer cannot punish; accepting one costs 0.2–0.4 (M4.5c §5.2).
- A guard at ×1.76 beats passive freezing and the weaker side's reply is a reset (M4.5b §3.3–§3.4), which is League-like.
- Indiscriminate aggression loses (M4.4c §5.4, M4.5c §5.3), also League-like.
- The weaker side yields position under a credible threat (M4.5c §2.3).
- Freezes pay +0.6 to +1.9 against opponents that concede, and the policies' rare freezing is the right reply to
  opponents that break (M4.5c §5.2).

---

## 3. Answers to the questions in the brief

| question | answer |
|---|---|
| Why is breaking a freeze inexpensive? | The break completes in one turn (C ≥ the freezer's wave plus the held wave's own firepower), one turn of exposure costs at most 21% of a bar plus a 5% gank expectation, and that HP is refilled by a one-resource reset that also invests gold. See §2.1. |
| How do positioning, power, Trading, retaliation, wave actions and farming opportunity interact there? | Positioning: the freezer is unreachable in its tower cell and must step into the held wave to threaten, where it takes positional fire. Power: it sets the one-turn cost (12% → 21% of a bar at ×1.76) and the lethal line (59% for a double hit). Trading: it costs the trader 20% farm and invites retaliation, so the guard is worth +1 per 100 turns, not more. Wave actions: Shove removes the freezer's whole wave in a turn; Thin cannot hold it. Farming: the walk-up collects at the full 0.9 rate and keeps all XP, so the only price of contesting is HP. |
| Does the system already punish a weaker champion that contests? | Yes, under a guard and at a power gap: +0.8 to +1.0 per 100 turns over a passive freeze (M4.5b). The weaker side still breaks and resets, and the freezer earns no more than it would by shoving. |
| Rules, parameters or learning? | Mainly rules and scale (one-turn contest, poke-only combat, cheap reset); partly parameters (the same rules at a higher per-turn combat magnitude behave differently); learning only in the margin (the finals over-concede in network-specific states; power gaps are out of distribution for both sides). |
| Why doesn't combat strength become economic control? | Because denial needs the victim's choice (Safe, reset or a death), the threat must be re-issued every turn at the threatener's cost, the victim can always step out of contact, and the XP geometry gives it a free cell. Strength converts through kills and forced Recalls; M4.5c did not measure that channel. |
| Is the threat of a winning trade represented adequately? | The threat is represented: a ×1.76 champion wins every exchange 3:1. What is missing is *commitment*: the threat is one exchange per turn and never more, so it is a poke. |
| Do the costs of Trading outweigh the benefit of strength? | For sustained, indiscriminate trading, yes, on every tested model. For selective trading the answer on the current model is open: M4.4c's Threaten plan was not re-run after the M4.5a retaliation and reach rules. |
| Could the system already produce zoning that PPO fails to exploit? | Partly. The attackers exploit each final by trading and freezing more (+1.0), and power-gap play is out of distribution, so the finals likely leave some conversion on the table. But the restricted games (no learning) show no scripted strategy that turns ×1.76 into denial in the neutral lane, so the ceiling is low whatever PPO does. |
| Do the freeze and zoning problems share a cause? | Yes: both are "contesting costs one turn of poke", and the one geometry where threat denies is the freeze geometry. |
| Would diverse starting states help without mechanics changes? | They would train the states the battery had to construct and probably raise the stronger side's conversion and freeze use at the margin. They cannot make a freeze beat shoving if the incentives say otherwise. Test it before deciding (E4). |
| How to separate a discovery failure from an incentive failure? | Three-way: exact and scripted screens (does any strategy convert?), the learned policies' counterfactuals (do they convert?), and learned best responses from the relevant starts (can a policy learn to convert against them?). If the first says no, it is the incentives; if the first says yes and the third learns it, it is the distribution; if the first says yes and the third does not, it is learning. |

---

## 4. Competing hypotheses

| # | hypothesis | predicts | status on current evidence | discriminating test |
|---|---|---|---|---|
| H1 | **Scale**: a contest resolves in one turn while per-turn champion damage is small and HP recovery cheap | scripted screens find no strategy that makes contesting costly; the break's exact value is positive at equal power and ≈ 0 at ×1.76 | supported (M4.5b grid, §5.7 exact values, §2.1 ledger) | E1 (exact ledger), E4 (if learned attackers cannot do better, H1 is binding) |
| H2 | **Asymmetric cost of aggression**: the trader pays farm, budget, exposure and retaliation; the victim pays only HP | sustained bullying loses even at ×1.76 against any non-conceding reply | supported (M4.4c, M4.5c §5.3); but it is League-like for poke, and its weight depends on the price of HP (H1) | E1 rows with and without the trader's costs; a `trade_farm_rule = "loser"` row |
| H3 | **Geometry**: 5-cell lane, XP range 2, reach 1, tower-cell contact rule, chase into the held wave | XP cannot be denied from the lane cell; the guard pays positional fire; zoning works only near the own tower | supported (M4.5b §3.6, §6.3 by geometry) | E1 (XP and gold by cell), E2 by situation |
| H4 | **Wave economics**: income capped by spawn and a farm-neutral crash return a safe wave to the loser ("kill bounce") | a lead won by a kill during a freeze evaporates by turn 20; a stronger side's lead comes from absence, not farm | supported (M4.5b §3.1) | E2 outcome tables (lead decomposition) |
| H5 | **Training distribution**: power gaps are 0.1% of decisions, so both sides' play at ×1.76 is out of distribution | learned best responses from gap starts convert more than the finals; diverse-start training changes gap behaviour | plausible, untested | **E4** (decisive), then M4.6c |
| H6 | **Self-play equilibrium**: the finals best-respond to an opponent that never concedes, so they do not freeze or threaten where a conceding opponent would pay | attackers win by aggression and freezing; the finals leave value against conceders | supported in the margin (+1.0 attackers, +0.6 to +1.9 vs conceders); target-specific, so not a family-wide trait | M4.6c population training, measured by the existing battery |
| H7 | **Measurement hole**: M4.5c tested only the indiscriminate bully and the posture-denial metric; selective threats and denial by absence were not measured on this model | re-running M4.4c's plans and outcomes on the current model changes the zoning verdict | likely (M4.4c §5.5 showed the conversion channel; the M4.5c readouts omit it) | **E2, E3** |

Most plausible: H1 + H3 set the ceiling (what any policy can get), H7 means the ceiling on the current model is not yet
measured for selective play, and H5/H6 explain the margin the finals leave. H2 and H4 are consequences of H1's scale
rather than independent causes.

---

## 5. Proposed diagnosis stage: M4.6a (no mechanics change, no new preset, no self-play training)

Every experiment reuses existing code. Horizons follow the horizon study (50-turn ΔQ₅₀ with a 300-turn validation subset,
pooled per family as M4.5c §10 recommends). Seeds from the reserved ranges.

| id | question | method | infrastructure | cost | decision rule |
|---|---|---|---|---|---|
| **E1** exact ledger of contesting | what exactly does a walk-up break cost and earn, by level, power gap, held-wave size and guard; same for a guard turn | exact one-turn expectations (chance and ganks enumerated): B's HP loss in bars and in resources at the measured price, B's gold and XP by cell, P(freezer's front dies), A's farm, positional fire and exposure; rows with the trader's costs removed one at a time; a row with positional fire switched off for an adjacent engaged front (design question, §6) | `m45b.exact`, `m45b.states.freeze_state`, `m45c.battery.mechanics` | minutes | decomposes the +0.18 / −0.2…−0.5 of §2.1 into its terms; feeds the lever choice in §6 |
| **E2** selective threats on the current model | does a *selective* stronger-side plan convert ×1.76, and through which channel | M4.4c's plan family (Farm, Threaten, Bully, Freeze+threat, Shove+Pressure) and its outcome table (weaker side's uncollected gold, XP lost, HP lost, died, recalled; connected trades; stronger side died) from the H cells and the natural HP-edge states, against the 16.384M finals and the scripted replies (ignore, respect, wary, shover) | `rl.m44c.zoning` plans and outcome readouts, adapted to the `m45c.horizon` engine (the plan rules read only the observation and the state) | 1–2 h on 11 workers at 50 turns + validation subset | if Threaten − Farm at ×1.76 is ≈ +1 as in M4.4c, denial by absence survived M4.5a and the brief's zoning concern is a measurement artefact; if ≈ 0, M4.5a's retaliation and reach rules removed it (H1/H2) |
| **E3** power-gap self-play readouts | what the learned stronger side actually earns at ×1.76 | add lead per 100 turns, kills, forced Recalls, the weaker side's XP deficit and absence turns to the existing ×1.76 behaviour stage; also at ×1.24 and ×2.04 | `m45c.behavior` (the games are already played) | minutes | the baseline E4 must beat |
| **E4** learned best responses from gap and freeze starts (decisive) | can a policy learn to convert a power gap or a freeze against these finals, if trained on those states | the attacker protocol (warm start from the target, 4.096M transitions, PPO vs the frozen target) with the start distribution replaced: (a) the learner ×1.76 stronger from the neutral and freeze starts, (b) the learner as freezer from the F grid; two targets (s0, s4: the least and most exploitable); a control attacker from s0 | `m45c.attackers` with a start-state archive (M4.3b's `mixed`-start mechanism, to be wired into the m45c trainer: a small, testable addition) | ≈ 1 h for 4–5 concurrent runs + evaluation | if the gap-start attacker's lead at ×1.76 exceeds the finals' own (E3) by ≳ 1 per 100 turns through kills, Recalls or denial, the training distribution is binding (go to M4.6c first); if it matches the finals, the incentives are binding (go to M4.6b); report the attacker's behaviour either way |
| **E5** the price of a reset | is a mid-HP Recall free or a gain on this model, and how much of that is the investment rule | paired value of Recall at full HP vs at 50% HP from natural states (E-family states plus full-HP states), decomposed into income lost and investment gained (re-run with investment deferred to the next return as a counterfactual) | `m45c.horizon` paired engine | ≈ 30 min | if a full-HP Recall costs < 0.5 or a 50%-HP Recall is ≈ 0, the reset subsidises every contest (lever L2) |
| **E6** (optional) credit assignment | is the value of a selective threat below the reward noise PPO sees | the critic's error on trade turns vs other turns in the finals; the signal-to-noise of the E2 plan values per decision (M4.4c §7.3 method) | `m45c.health`, M4.4c §7.3 | ≈ 20 min | if the per-decision value is < 0.1 against a per-turn noise ≈ 1, PPO cannot be expected to find it; informs M4.6c vs M4.6b |

Total ≈ 5–6 h of compute, no approval-gated training beyond the four or five attacker runs (gated like M4.5c's).

**Pre-registered outcomes.** Before running: (i) the sign and size of Threaten − Farm at ×1.76 (E2), (ii) the threshold
for E4 ("exceeds the finals' lead by one resource per 100 turns"), (iii) the E5 thresholds. Record them in the M4.6a
plan file as M4.5c did.

---

## 6. Candidate levers for M4.6b (only if M4.6a says the incentives bind)

Each lever would be an opt-in field defaulting off, screened on M4.5b's scripted harness (exact bench, freeze grid,
zoning decomposition, restricted-game matrices; DEV seeds, then HOLDOUT), with the gates below, before any PPO.
`m45_calibrated()` and every M4.5c result stay the baseline.

| lever | what it changes | why it is a coherence fix, not a tuning target | gates |
|---|---|---|---|
| **L1 per-turn combat magnitude**: `trade_exchanges = 2` (implemented, off; one exchange per tick, like minion fire) | a connected trade becomes two independent exchanges: 24% of a bar equal, 42% at ×1.76; two double hits kill from full HP at ×1.76 with p ≈ 3%, and three or more units (p ≈ 15%) kill from 60% | it aligns the combat clock with the wave and recovery clocks (§2.3) and gives the threat a lethal line without a new action; retaliation already runs per tick | (a) the equal lane from s0 stays a farming duel: trade attempts and deaths per 1000 within ≈ 1.5× the current scripted equilibria, no always-Trade strategy in the s0 equilibrium; (b) the walk-up break against a guard at ×1.76 costs more than it earns without a reset; (c) threat-only zoning in the neutral lane at ×1.76 becomes clearly negative for the weaker side; (d) the equal-power freeze grid changes within noise |
| **L2 reset pricing** (only if E5 shows a reset is free) | candidates, cheapest first: `recall_turns` 3 (an explicit walk back); investment credited one return later; a minimum investment lot | in League a back with nothing to buy is a pure loss; here continuous investment makes every back partly a gain, which subsidises contesting | (a) Recall timing counterfactuals (E family) keep their signs; (b) deaths per 1000 do not rise; (c) the full-HP Recall's value becomes clearly negative |
| **L3 a commit behaviour** (larger; only if L1 fails its gates) | an *Engage* behaviour: two exchanges, the target cannot evade on that turn, no wave damage, farm ×0.5, full retaliation and the tower shot if adjacent to the enemy tower | the threat that zones in League is the all-in, which the action set lacks | as L1 plus: Engage must not dominate Trade at equal power |

**Not recommended, and why:** presence-based zoning factors (deprecated; denial without a decision), a change to
`xp_range` (fixed by design and League-consistent), the own-tower rate (0.8 is farm-neutral by construction), the Vision
economy (the brief excludes it), the kill bounty and death timers (already League-proportional), and any rule that reads
the opponent's action or the "freeze" label (behaviour exemptions, which the user ruled out in M4.5a).

**One design question for E1, not a change:** positional aggro fires at a champion standing in the enemy front's cell
whenever its own front is behind it, even when that own front is adjacent and engaged. In League, minions already
fighting minions do not switch to a champion that merely stands there. In the freeze geometry this is the guard's main
cost at low levels (M4.5b §3.6). E1 should report the guard's value with and without that fire so the review can decide
whether the cover condition should include an adjacent engaged front.

---

## 7. Training-side options: M4.6c (if M4.6a says the distribution binds, or in any case as a cheap complement)

- **Diverse starts.** A share of training games (25–50%) from an archive of constructed and witnessed states: power
  gaps ×1.24–×2.04 in both seats, freeze states, early-level states. The M4.3b `mixed` start mechanism exists; the M4.5c
  battery already defines the state families. Evaluate with the unchanged battery plus E3's readouts.
- **A style population.** A share of games against scripted conceding, wary and aggressive opponents (M4.5b's
  strategies) alongside past snapshots, OpenAI Five style. This targets the +0.6 to +1.9 the finals leave against
  conceders and the attackers' +1.0.
- **What these cannot do.** They train the states and opponents the finals never meet; they do not change what a freeze
  or a threat is worth against competent play. If E2 and E4 show the ceiling is low, M4.6c alone will not raise it.

---

## 8. Roadmap

| stage | content | gate to the next stage |
|---|---|---|
| **M4.6a** diagnosis (≈ 1 day; ≈ 6 h compute) | E1–E5 (E6 optional); a report with the hypothesis table of §4 filled in | E4's verdict and E2's Threaten − Farm decide the fork |
| **M4.6b** mechanics screen (≈ 1–2 days; M4.5b-scale scripted runs) | L1 first, L2 only on E5's evidence, L3 only if L1 fails; opt-in fields, DEV then HOLDOUT; reference §16 added only for an adopted candidate | all gates of §6 pass on HOLDOUT; a candidate preset `m46_...()` with every earlier preset byte-identical |
| **M4.6c** training-side changes (≈ 1 day) | diverse starts and/or a style population on the baseline or the candidate; pilot at 1.024M, then one full seed | the battery's gap and freeze families and E3's readouts move in the predicted direction without losing strength against the M4.5c finals |
| **M4.6d** fresh PPO (the M4.5c campaign design unchanged: 6 × 16.384M, health checks, battery, attackers) | only on an adopted candidate and/or training change | the M4.5c report's main table, side by side with the baseline |

M4.6b and M4.6c can run in either order or together, but neither before M4.6a. No PPO before M4.6d.

---

## 9. Trade-offs and recommendation

- **Leave the model unchanged.** Defensible: the model is coherent and its results are honest; freezes that break when
  the freezer cannot punish, and bullies that lose, are League-like. The cost is that the research question about
  threat-based play stays answered with "the game does not reward it", and §2.3 shows one clock clearly out of scale.
- **Change training only.** Cheap, reuses everything, and addresses the margin the finals leave. It cannot raise a
  ceiling set by the incentives, and it risks a long campaign that confirms M4.5c.
- **Refine mechanics.** The right move if M4.6a confirms the scale diagnosis, and only for the minimal lever (L1 is one
  existing field). The risk is the one the config note already records: at the old economy, two exchanges made the equal
  lane a trade-fest. The gates in §6 exist to catch that, and the M4.5a retaliation and positional rules now tax trades
  in HP, which did not exist when that note was written.

**Recommendation: the combination, in the order a → (b and/or c) → d.** M4.6a is cheap and decides the fork with a
test that directly answers the brief's question 3. My expectation, stated so it can be checked: E2 will find a modest
positive Threaten − Farm at ×1.76 (denial by absence, smaller than M4.4c's +1.1), E4's gap-start attackers will convert
somewhat more than the finals but not by a large margin, E5 will find a mid-HP reset close to free, and the conclusion
will be that the scale (H1) binds with a distribution margin (H5) on top. In that case: screen L1 (and L2 if E5 says so)
in M4.6b, add diverse starts in M4.6c, then one fresh campaign.

---

## 10. Other observations

1. **The M4.5c zoning metric.** "Gold left uncollected when the stronger side is near" measures posture denial only.
   M4.4c §5.5 showed the conversion runs through kills and forced Recalls. E3 adds those readouts; the research report's
   zoning row should be re-read after E2 and E3.
2. **The reset subsidy.** On the earlier model a Recall from a natural mid-HP state was worth −0.1 to +0.5 (a gain,
   reference §6.7), because the return invests all gold. If that holds here (E5), the price of HP is partly negative,
   which lowers the cost of every contest and every walk-up independently of combat scale.
3. **Credit assignment.** M4.5b caveat 4: one tick of diverted minion fire moves a 20-turn return by ±0.8, far above a
   trade's HP value. Small, real values of selective threats may sit below the noise PPO sees (E6).
4. **Early game.** Levels ≤ 10 are ≈ 10% of training and the early-game validations failed most often. The cheapest
   diverse-start archive is an early-level one, whatever the fork.
5. **The seat difference** (+0.5 for seat A against the scripted suite, reversed in the pilot) is unexplained and should be
   decomposed before it is carried into another campaign; it is cheap (existing evaluation records).
6. **Fog of war.** Zoning in League also works through uncertainty about the jungler. The model has none (both see
   everything; a gank is a hazard rate). Out of scope here, but it bounds how "League-like" threat-only zoning can be.

---

## Sources

- `docs/m4/laning_game_mathematical_reference.md` §4.5, §4.8, §5.3–§5.4, §6.1–§6.7, §7.3, §8.7–§8.8, §10.1, §14.5–§14.8, §15.
- `docs/m4/m4_5b_calibration_report.md` §3 (freezes), §4.3 (price of HP, trade value), §6 (Safe, Yield, threat-only zoning), §7 (restricted games), §14 (caveats); `results/m45b/report.md` (candidate freeze grid, zoning decomposition).
- `docs/m4/m4_4c_strategic_investigation.md` §4 (freezing), §5 (strong-side zoning, denial by absence), §10 (design questions).
- `docs/m4/m4_5c_research_report.md` §2.3, §5.1–§5.7, §7, §9, §10; `results/m45c/final/report.md` §2, §3b (H cells), §5, §6; `results/m45c/battery/report_ext.md` (G, H).
- Code: `lanegame/config.py` (`m4_four_actions_econ`, `m45_unified`, `m45_calibrated`, `trade_exchanges`, `trade_farm_rule`), `lanegame/champions.py` (`in_contact`, `_evasion_cell`, `trade_hit_probability`, `champion_gank_probability`), `lanegame/champions_m4.py` (`resolve_turn_m4` steps 1–9b, `contact_cell_target`), `lanegame/positional.py` (`in_contact`, `reach_interval`, `minion_damage_centi`), `lanegame/experiments/rl/m45c/horizon.py` (`FAMILY_CANDS`), `lanegame/experiments/rl/m44c/zoning.py` (Threaten, Bully plans).
