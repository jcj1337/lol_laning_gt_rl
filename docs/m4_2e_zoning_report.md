# M4.2e-B — Design Philosophy and Combat-Driven Zoning

Branch `development/m4-natural-waves`, on top of M4.2e-A.1 (9a282d1). Date: 2026-10-12.

| | |
|---|---|
| **Deliverable 1** | [design_philosophy.md](design_philosophy.md), linked from the README and the roadmap. |
| **Candidate preset** | `GameConfig.m4_lane()`: `m4_natural()` with three opt-in changes (§6). Every historical preset is unchanged (digest-pinned). |
| **Driver** | `python -m lanegame.experiments.m4.zoning_natural` (full run: 135M resolver turns, 7 min on 11 processes; `quick` runs the neutral state in 15 s). Results in `results/m4_zoning_natural/`. |
| **Tests** | `tests/test_m4_zoning_natural.py` (16 tests) plus the existing focused suite. |
| **Verdict** | **PASS WITH CAVEATS** (§10). The zoning question is answered differently from how it was asked: in this game a credible threat converts into wave control, not into the weaker champion standing at experience range. |

Coordinates are each side's OWN coordinates unless a trajectory says GLOBAL (own tower −2, the lane 0, enemy tower
+2). "Gold" means resources (1 resource = one fresh 8-HP wave of farm). One HP unit is a sixth of the bar.
"A" is the stronger champion in every asymmetric test and the row player in every matrix; values are to A.

## Summary

- **The question was whether a credible combat threat can make a weaker champion surrender position, farm, XP or
  wave control when that is rational.** The M4.2d study had found that it never did. This study reproduces that
  finding on natural waves, explains it, tests five competing corrections, and adopts the two kinds that work:
  the collection rates and durability.
- **Why the weaker champion never stands back (observed, §3).** On this five-cell lane a passive concession at the
  lane is almost total: a champion out of the enemy's reach is out of reach of the minions too, and the wave does
  not move toward a champion that stops contesting it. Standing back costs 6–12 gold per 32 turns. Contesting a
  ×1.76 opponent cost 0.2–4, because trading was a symmetric war of attrition: both traders paid half their farm,
  the weaker one could always trade back, and a recall at 40% HP capped its loss. Even three deaths in 32 turns did
  not make standing back rational. No combat change tested makes it so at the lane (§5).
- **Two fixed collection rates were doing the work zoning should do, with the wrong sign (observed, §2, §4).** The
  0.5 rate under the enemy tower stood in for danger and made pushing lose at every power gap (−6 to −7 per 32 turns
  from a pushed lane, −3 to −4 from a siege, regardless of a ×2.4 lead). The defender collected a crashed wave at
  the full 0.9 under its own tower, so every crash was a gift. Under those rates a freeze paid +4.7 at every gap and
  the neutral lane paid the stronger champion +0.3. Power barely mattered.
- **The adopted correction** (§6): collect at the ordinary 0.9 under the enemy tower (the danger is explicit: tower
  shots, ganks, the defender's home advantage), collect at 0.6 in one's own tower cell (the tower competes for last
  hits), and let invested gold buy durability as well as attack power. With these three changes every state's
  value is monotone in power, pushing when ahead pays, a lead of one item's worth is worth 1–3 gold per 32 turns
  and a large lead 3–7 (10–25% of the lane's income), and the stronger champion earns it mainly through wave
  control, trading on fewer than a third of its turns.
- **What the weaker champion does instead of standing back (observed, §7).** It shoves the wave away from the
  stronger champion and trades back only when caught; it farms its crashed waves under its tower while the stronger
  champion sieges for plates; it recalls at low health. These are concessions of wave control and position, and they
  are the League pattern for a lost lane. The stand-off cell gives a zoned champion half its farm without ever
  being hit.
- **Rejected or left off:** one exchange per tick (`trade_exchanges = 2`) turns the equal lane into a trade-fest
  and adds nothing once durability is in; charging a mutual trade's farm cost to its loser (`trade_farm_rule =
  "loser"`) is coherent but adds nothing measurable without per-tick exchanges; removing the Safe posture base makes
  standing one cell back free; a kill bounty of two waves is the only thing that produced a stand-back concession
  (in the freeze state, ×1.76, and only with per-tick exchanges), at the price of a noisier equal lane; it is left
  for review.
- **Caveats:** the equal-power lane now favours pushing over holding (both sides shove in equilibrium); a freeze is
  not worth holding at equal power because breaking it pays the breaker; the study is scripted-policy evidence, not
  learned play; an own-tower rate of 0.6 is a calibration choice (§8).

---

## 1. What was asked, and the tools

Three tools, none of them training:

1. **An inventory** (§2) of every rule that moves gold without a combat event: fixed collection-efficiency factors
   on one side, explicit danger mechanisms on the other.
2. **The one-turn concession inequality** (§3.1), the M4.2d-C analysis re-derived on natural waves from four start
   states, with chance events entered as expectations. It gives the *flip multiplier* m\*: the attack multiplier at
   which the weaker champion's best reply to a Trade becomes a concession. It cannot see absences, pushes or plates.
3. **Sampled policy games** (§3.2 onward): thirteen reactive scripted policies, the same set for both seats, played
   from each start state for 32 turns (16 waves) with the engine's own chance events on common random numbers
   (16 seeds; trades land, towers shoot, ganks happen, champions recall and die), under five power gaps. Each
   13 × 13 matrix of mean values is solved as a zero-sum game. This captures what the one-turn analysis cannot.

**States** (deterministic from s0; both champions then put at equal XP, full HP, no gold, no investment before the
gap is applied):

| state | k | GLOBAL lane | champions (own cells) | what it is |
|---|---|---|---|---|
| neutral | 9 | fronts 13/13 at 0 | A 0, B 0 | symmetric contact at the lane |
| pushed | 16 | A's 24-HP stack at +1 against B's 1-HP front | A 0, B 0 | A's slow push one turn before it crashes |
| siege | 20 | A's 24-HP mass under B's tower | A +2 (Hold + Pressure), B −2 | the turn after the crash |
| freeze | 36 | A 11 at −2, B 19 at −1 | A −2, B +1 | the outside-tower freeze of the A.1 review |

**Power gaps** (investment is the saturating curve; levels raise attack power and max HP):

| gap | A, B invested (resources) | A level lead | attack-power ratio | fight ratio with durability |
|---|---|---|---|---|
| equal | 0, 0 | 0 | 1.00 | 1.00 |
| moderate | 16, 4 | 0 | 1.24 | 1.54 |
| levels | 0, 0 | +3 | 1.35 | 1.35 |
| large | 50, 0 | 0 | 1.76 | 3.10 |
| large_levels | 50, 0 | +3 | 2.38 | 4.2 |

Sixteen resources invested is about one League item; fifty is a fed champion against an empty one.

**Policies** (shared by both seats; every policy recalls at or below 40% HP once out of the enemy's reach, walks back
first otherwise, and resets while the enemy is dead or recalled if below 70%):

| name | behaviour | wave | objective |
|---|---|---|---|
| farm | Hold | Hold | — |
| bully | Trade every turn the enemy is present | Hold | — |
| punish | Trade only when the enemy starts the turn within reach and outside its tower cell | Hold | — |
| edge | Trade when in reach and the duel edge is positive | Hold | — |
| standoff | Yield (the stand-off cell) | Hold | — |
| respect | Safe when the enemy starts within reach, else Yield | Hold | — |
| tower | Hold at the tower when the wave is there and nobody is in reach, else Safe | Hold | — |
| push | Hold | slow push (Thin while not more than 2 HP ahead) | Pressure |
| push_punish | punish | slow push | Pressure |
| clear_punish | punish | Clear | Pressure |
| freeze_punish | punish | freeze (Thin while the enemy front leads by 8) | — |
| bully_shove | bully | Clear while the enemy is away, else Hold | Pressure |
| punish_shove | punish | Clear while the enemy is away, else Hold | Pressure |

"standoff", "respect" and "tower" are the policies that give way; a cell counts as a concession only when the
weaker champion's best reply is one of them *and* it collects less than 80% of what plain contesting collects
against the same opponent.

**Value.** Gold earned by A minus gold earned by B over the window (farm, plates and bounties; exact ledger
units), plus 0.25 resources per HP unit of the terminal health difference (a full bar for a champion that is
away). Sensitivities at 0 and 0.4 per unit and with XP valued at 0.05 per point are in the JSON.

**Variants.** Every variant is `GameConfig.m4_natural(**fields)`; "base" is the M4.2e-A.1 preset.

## 2. Inventory: fixed collection factors and explicit dangers (base rules)

Gold per minion HP is `farm_per_hp · base(posture, cell) · (1 − zone) · t`, with the zone already off
(`zoning_rule = "none"` since M4.2d). The fixed factors:

| factor | value | where it applies | what it stands for |
|---|---|---|---|
| `collect_contest` | 0.9 | every contest kill | last-hit inefficiency; uniform, not zoning |
| `collect_contest_enemy_tower` | 0.5 | contest kills in the enemy tower cell | **danger next to the enemy tower** (the A.1 review's words: "stands in for the threat of farming next to the enemy tower") |
| Safe / retreat posture base | 0.8 − 0.2 (x + 2): 0.8 / 0.6 / 0.4 / 0.2 / 0 by cell | a Safe champion, or a Yield that backed off | **the price of respecting a threat**, set by fiat: 44% of the contest rate at the lane |
| `retreat_farm_reach` | 2 (vs `farm_reach` 1) | a Yield that backed off | **a partial concession by fiat**: the only way a zoned champion keeps anything at the lane (0.2 of 0.45 per turn) |
| `trade_farm_factor` | 0.5 | every connected trader, win or lose | the opportunity cost of fighting; symmetric |
| kills in the own tower cell | 0.9 (contest) | a crashed enemy mass dying to the tower and the defenders | nothing: towers do not compete for last hits |

The explicit dangers (no gold enters them directly):

| mechanism | magnitude, base rules |
|---|---|
| a connected trade on a Hold defender | 0.72 HP units expected per turn (hit 0.55; double 0.3), times the attacker's attack power |
| ganks | expected units per turn for a Hold champion at own cells −2 … +2: 0.01, 0.06, 0.14, 0.26, 0.42; +0.02 to +0.04 per turn for Trade or Clear |
| tower shot on a connected trader in the enemy tower cell | 1 unit with p = 1.0 (0.5 while its own crash is under that tower) |
| tower-adjacent shot on a connected trader next to the enemy tower | 1 unit with p = 0.5 (0.25 busy) |
| a champion in its own tower cell | can only be traded by a dive |
| the pin | a Yield that starts the turn within reach cannot evade |
| recall at 40% HP | 2 turns away plus the walk back; a death: 3 turns and a 0.5 bounty |

The enemy-tower rate and the posture base are proxies for consequences the dangers are supposed to produce. The
retreat reach is a proxy for a concession the dynamics are supposed to produce. The own-tower rate is the opposite:
a real mechanical inefficiency that was missing.

## 3. Why the weaker champion never stands back

### 3.1 The one-turn inequality on natural waves (deterministic; base rules)

Neutral state, both at Hold waves, one turn (expected damage in HP units):

| A – B | gold A | gold B | B's loss: reach / posture | damage to B | damage to A |
|---|---|---|---|---|---|
| Hold – Hold | 0.45 | 0.45 | 0 / 0 | 0.14 | 0.14 |
| Trade – Hold | 0.225 | 0.45 | 0 / 0 | 0.86 | 0.18 |
| Trade – Yield (pinned) | 0.225 | 0.45 | 0 / 0 | 0.86 | 0.18 |
| Trade – Safe (evades) | 0.45 | 0 | 0.45 / 0 | 0.07 | 0.18 |
| Hold – Safe (unthreatened) | 0.45 | 0.20 | 0 / 0.25 | 0.07 | 0.14 |

A stand-off that starts in reach is pinned, so the only concession is Safe, which costs B all 0.45 and refunds A's
0.225 trade cost. B concedes only when 0.25 · 0.72 · m exceeds 0.675: **m\* = 3.68** (3.54 over six turns; 2.3–2.4
at 0.4 per unit). The M4.2d numbers were 3.4–4.0. In the freeze state m\* is **1.56**: there the zoned champion's
income is already halved by the freeze and the breaker fights next to the enemy tower, so a threat is credible at
a moderate gap in the one-turn sense. In the pushed and siege states m\* = 1.0: the sieging champion has nothing
to gain by trading at all.

### 3.2 The policy games (sampled, base rules)

Equilibrium value to A, 16 seeds × 32 turns, 0.25 per HP unit:

| state | equal | moderate | levels | large | large_levels |
|---|---|---|---|---|---|
| neutral | −0.42 | −0.12 | −0.25 | +0.34 | +1.17 |
| pushed (A pushed) | −6.54 | −6.88 | −6.88 | −6.05 | −5.83 |
| siege (A sieges) | −4.17 | −3.52 | −4.18 | −3.23 | −3.47 |
| freeze (A freezes) | +4.69 | +4.70 | +4.69 | +4.69 | +4.69 |

Three things stand out. Power is almost worthless at the lane (+0.34 at ×1.76). Pushing loses at every gap. A
freeze is worth the same at every gap, and the freezer never trades (0.00 trades per turn in every freeze
equilibrium): the weaker champion's reply is to Clear the frozen wave away within two turns, which the freezer
cannot punish from inside its tower cell.

**The cost of giving way, neutral state, ×1.76** (cells of the matrix, value to A per 32 turns):

| A plays | B contests (farm) | B trades back (bully) | B stands off | B respects | B hides at its tower |
|---|---|---|---|---|---|
| punish | +0.32 (B dies 1.1×) | +1.32 (B dies 1.1×) | +0.18 (pinned 17×, dies 0.6×) | +5.99 (never hit; 78% of its farm) | +9.97 (35%) |
| bully | −0.38 (B dies 1.4×) | +0.25 (B dies 1.4×) | +4.29 (pinned 4×) | +6.50 (87%) | +12.74 (5%) |

A stand-off that starts in reach is pinned, so against a punisher it is just a Hold one cell back; "respect" is
never hit and keeps 78–87% of its farm because it steps in on every turn the punisher does not attack (the retreat
reach of 2 does the rest); "tower" keeps 5–35%. Against every aggressive policy, contesting or trading back costs
B 0 to 1.3 even though it dies once or more per 32 turns, and standing back costs 6 to 13. The gap is an order of
magnitude, so no plausible threat closes it. The reason is on A's side of the ledger as much as B's: a connected
trade halves A's farm as well as threatening B's health, so a bullying A at ×1.76 loses 0.4 per 32 turns against a
champion that just keeps farming, dies 1.4 times and recalls 1.3 times (trajectory in §7.1).

### 3.3 Competing explanations tested

| hypothesis | variant | neutral ×1.76 | neutral equal (A trades/turn) | pushed ×1.76 | freeze ×1.76 | stands back? |
|---|---|---|---|---|---|---|
| H0 base | — | +0.34 | −0.42 (0.00) | −6.05 | +4.69 | no |
| H1 damage: one exchange per tick | `trade_exchanges = 2` + loser | +1.39 | −0.65 (0.45) | −4.91 | +4.60 | no |
| H2 decisiveness: durability (+ loser) | `invest_durability`, `trade_farm_rule = "loser"`, with ×2 | +3.75 | −0.61 (0.32) | −3.64 | +6.61 | no |
| H3 deaths too cheap: bounty 2 | `kill_bounty = 2` on H1 + H2 + economy | +7.12 | −1.18 (0.33) | +3.32 | +9.64 | freeze only (B hides 0.70) |
| H4 the fixed rates | enemy tower 0.9, own tower 0.6 | +2.29 | +0.20 (0.35) | +0.78 | +2.16 | no |
| H4 + H2 with loser | `t_ot6_dur_loser` | +5.16 | −0.14 (0.10) | +2.87 | +5.45 | no |
| **H4 + durability (adopted)** | **`m4_lane()`** | see §6 | see §6 | see §6 | see §6 | no |
| H4 + H1 + H2 | + `trade_exchanges = 2` | +4.89 | −0.18 (0.38) | +2.73 | +4.74 | no |

(H1 alone: `x2_loser`; H2 with H1: `x2_dur_loser`; the full table of nine variants × four states × five gaps is in
`results/m4_zoning_natural/summary_full.txt`; the adopted preset's own full run is in
`results/m4_zoning_natural/zoning_natural_t_ot6_dur.json.gz` and `summary_m4_lane.txt`.) The earlier quick runs also tested removing the posture
base and the retreat reach ("reach"): standing one cell back then costs nothing when unthreatened, so "Safe" became a
free insurance policy and nothing else changed; and raising the bounty alone (`k2`, `t_k2`, `t_x2_k2`).

The reading: damage per exchange (H1) raises the stakes for both sides and makes the equal lane a trade-fest
(bully against bully, each side losing half its income); it adds nothing on top of H2 and H4 (+4.89 against +5.16).
Deaths (H3) are the only lever that produced a stand-back concession, and only in the freeze state with per-tick
exchanges. The fixed rates (H4) are what made power worthless outside the lane. Durability (H2) is what makes a lead
pay at the lane; the loser rule adds nothing measurable on top of it (§8). Both H2 and H4 are needed for a lead to
pay everywhere, and neither makes the weaker champion stand back.

## 4. The fixed rates, in detail

Under the base rules a crash is a gift: the attacker's stack dies to the defender's tower (3 HP per tick) and the
defender collects every HP at 0.9 from its tower cell, while the attacker collects the defender's minions at 0.5
in the tower cell and not at all behind it. The siege state is worth −3.2 to a ×1.76 sieger and the pushed state
−6.1. With the enemy-tower rate at 0.9 alone (`tower` variant, quick run): pushed −2.8, siege −2.8. With the own-tower
rate at 0.6 as well (`t_ot6`): pushed −0.04 at equal power rising to +1.5, siege +0.5 rising to +3.2.

The own-tower rate is a mechanical inefficiency, not a zoning proxy: last-hitting under a tower is harder, and a
crashing stack dies faster than it can be last-hit. The enemy-tower rate was a proxy; its consequences are now
explicit and responsive: a connected trader next to the enemy tower takes the adjacent shot, a champion standing in
the enemy tower cell takes 0.42 expected units of gank damage per turn against 0.14 at the lane, and the defender
trades from its tower cell without a dive while the attacker needs one.

The Safe posture base is kept. It is the only thing that prices a one-cell retreat on a lane where farm reach equals
the step, and removing it (the "reach" variants) made Safe a free insurance posture. The retreat reach of 2 is
kept with it; both are flagged as geometry proxies in the design philosophy (§3 there).

## 5. The combat changes, in detail

**Durability.** Investment now divides the damage a champion takes by its own multiplier, in exact integers
(`M4Context.mitigate`); the saturating curve is unchanged, so a hit never kills from full health. At 16 resources
(one item) the fight ratio is 1.54 instead of 1.24; at 50 it is 3.1 instead of 1.76. Levels already gave both attack
power and max HP; investment now does the same. One-turn trade damage at ×1.76: A deals 1.27 units, B deals 0.41
instead of 0.72. A champion at ×1.76 that punishes a contesting opponent for 32 turns takes 3 HP units, mostly
from ganks, instead of 5; the opponent takes 25.

**Loser pays** (implemented, tested, not in the preset). In a mutual connected trade the side that lost the smaller
fraction of its max HP keeps its farm; ties and unilateral trades are charged as before. The rolls are therefore
made before the gold step (resolver step 9a). On deterministic paths (no chance events) nothing changes. It is the
coherent version of the trade cost (the loser of a trade is the one knocked off its last hits), but the policy games
cannot measure its effect: with durability and one roll per turn it changes every state's value by less than one
standard error (§8), and with two rolls per turn by +0.46 at ×1.76. It stays an opt-in field.

**Exchanges per tick** (`trade_exchanges = 2`) is implemented and tested but not in the preset. One roll per turn
stays.

## 6. The candidate preset

```python
GameConfig.m4_lane() == GameConfig.m4_natural(collect_contest_enemy_tower=0.9, collect_contest_own_tower=0.6,
                                               invest_durability=True)
```

Everything else is M4.2e-A.1: natural waves, the entry crash, tower 3, AP-scaled Clear, the stand-off and pin,
no proximity zoning, the saturating investment curve, 46 actions, 88 observation entries.

**Equilibrium value to A, `m4_lane()`** (16 seeds × 32 turns; A's trades per turn in brackets):

| state | equal | moderate | levels | large | large_levels |
|---|---|---|---|---|---|
| neutral | −0.12 (0.12) | +2.14 (0.36) | +1.02 (0.28) | +4.73 (0.30) | +6.50 (0.22) |
| pushed | −0.02 (0.19) | +0.95 (0.28) | +0.40 (0.28) | +2.67 (0.27) | +3.79 (0.28) |
| siege | +1.19 (0.24) | +3.01 (0.23) | +1.97 (0.23) | +4.09 (0.01) | +3.99 (0.01) |
| freeze | +0.00 (0.14) | +2.12 (0.25) | +1.97 (0.14) | +5.64 (0.21) | +6.72 (0.18) |

(Standard errors of the cells are 0.3–0.8.) The lane's total income is about 27 per 32 turns when both farm, so a
large lead is worth 10–25% of it. Equilibria: at equal power A mixes slow push (0.71) and Clear (0.28) against B's
Clear (0.79) and slow push (0.21); at ×1.76 both play Clear with punishing; in the siege at ×1.76 A keeps sieging
and B pushes back out, and at ×2.38 B farms under its tower (the "tower" policy, 1.00); in the freeze both Clear at
every gap. A siege is worth +1.2 to the sieger even at equal power: a crashed stack under the enemy tower pays in
plates and in the defender's lost last hits.

**Against the brief's expectations:**

- *At equal strength both champions have meaningful opportunities to contest.* Yes: the value is within noise of
  zero in every state, every contest policy appears in some equilibrium, and A trades on 12–24% of its turns.
- *As combat advantages increase, unfavourable trades become more consequential.* Yes: a contesting B at ×1.76
  takes 1.3 units per connected turn and deals 0.4; against A's aggressive policies its deaths per 32 turns go
  from 0–0.3 at equal power to 0.1–1.6 and its recalls to 1–3; a punishing A takes 3 units in 32 turns, mostly
  from ganks, while the contesting B takes 25. B's best reply to aggression is to shove the wave (Clear or slow
  push) at every gap; trading back is a best reply only to an opponent that hides at its tower.
- *A disadvantaged champion sometimes rationally retreats, yields or recalls instead of repeatedly accepting
  unfavourable engagements.* Partly. It recalls (1–3 times per 32 turns), it gives up wave control, and it farms
  under its tower when sieged (the siege at ×2.38 is B's "tower" policy, with A trading once in a hundred turns).
  It does not stand at experience range while the stronger champion holds the wave at the lane (§7.2).
- *A stronger champion can establish positional control without attacking continuously.* Yes: in every ×1.76
  equilibrium A attacks on at most 30% of its turns; in the siege it attacks on 1%; its value comes from pushing,
  plates and the under-tower rate.
- *None of this is guaranteed by attack power alone.* The weaker champion's counterplay is in every equilibrium:
  Clear to move the wave, trade back when caught (a pinned stand-off is still hit), recall, farm the crash. A
  perma-shoving ×1.76 champion that stands in the enemy tower cell for twenty turns loses two thirds of its bar to
  ganks and the defender's free trades (§7.1).

## 7. Representative trajectories (`m4_lane()`, sampled, seed 5000)

The full traces are in `results/m4_zoning_natural/trajectories_candidate.txt` (GLOBAL lane strings).

### 7.1 Neutral lane, ×1.76

**A punishes, B farms.** A pokes every turn B starts in reach; B keeps farming at 0.45 and absorbs 1.3 units per
turn; after eight turns B is at 1.0 of 6 units, walks back, recalls, returns at full health, and the cycle repeats.
A's farm is 0.23 on every poking turn. Over 32 turns A is −2.1 on this seed: poking a champion that keeps farming
barely pays even at ×1.76, because the poke costs half of A's farm on every poking turn and B's recall costs B
only three turns. (Matrix cell over 16 seeds: +0.72 ± 0.58.)

**Both Clear and punish.** A's Clear budget is 7 HP per turn against B's 4. A kills every fresh B wave at the lane,
crashes 8 HP into B's tower every two turns, stands in B's tower cell with Pressure and takes three plates by turn
20. B, in its tower cell, trades A every turn from safety (A does not dive) and collects its crashes at 0.6. After
twenty turns of being ganked and traded in the enemy tower cell A is at 1.9 of 6 units; it walks out at turn 31,
recalls, and the window ends at +10.0. This is the perma-siege of the A.1 review, now with a real price.

**A punishes, B respects.** B is Safe on the turns it starts in reach and stands off otherwise. It is never hit; it
farms 0.45 on alternate turns and nothing on the others. A gets +7.2: the stand-off cat-and-mouse gives the zoned
champion half its farm for free, and the zoner half the lane for free.

### 7.2 Why standing back loses: the wave does not come

In every neutral-state trajectory where B stands back, the fronts stay at the lane: a Hold champion does no wave
damage, so B's absence from the wave does not push it toward B. B's minions die at the lane, two cells from its
tower, and B collects nothing. In League the wave drifts toward a champion that stops last-hitting; here it does
not, by the A.1 design (Hold is the zero-damage slot). The only way the wave reaches B is for A to Thin or Clear,
which A does when it pays (the own-tower rate makes it pay), and then B farms under its tower at 0.6 while A takes
plates. That is the siege state, and it is where "concede and farm under tower" is in B's equilibrium.

### 7.3 Freeze, ×1.76

A holds B's wave at its −1. B steps up and Clears it; A Clears too, at 7 HP per turn against B's 4, and wins the
minion fight outright: by turn 38 A's front is back to 11 HP and B's has been pushed to B's side. The two trades
on the way leave B at 1.4 units; B walks back and recalls at turn 40, A crashes 9 HP into B's tower on the same
turn and sieges with Pressure for twelve turns: +10.3 over the window. At equal power the same break succeeds:
B's 22-HP wave crashes into A's tower at turn 38, B takes plates, and a freezer that merely holds is −4.9 by the
end. A freeze is worth holding by the champion that out-clears the breaker, which is League.

### 7.4 Equal power: both push

A slow-pushes, B Clears and punishes; the waves crash alternately; ganks and trades drain both; A recalls at turn
15 and B ends the window at 0.9 of 6 units. Final −0.6 to A. Equal lanes are active and roughly even.

## 8. Calibration sensitivity (quick runs: 8 seeds, all states, equal and ×1.76)

Data: `results/m4_zoning_natural/zoning_natural_sensitivity.json.gz` and `summary_sensitivity.txt`.

**The own-tower rate** (enemy tower 0.9 and durability in every row; the loser rule on, which changes nothing
here; equilibrium value to A, equal / ×1.76):

| own-tower rate | neutral | pushed | siege | freeze |
|---|---|---|---|---|
| 0.5 | −0.21 / +5.75 | +0.85 / +3.72 | +2.21 / +5.44 | −0.45 / +5.45 |
| **0.6** | −0.17 / +5.30 | +0.76 / +3.24 | +1.70 / +4.24 | +0.00 / +5.16 |
| 0.7 | −0.01 / +4.85 | +0.84 / +2.18 | +1.25 / +2.48 | +0.51 / +4.87 |
| 0.9 (no inefficiency) | +0.08 / +3.40 | −2.14 / −0.68 | −1.54 / −0.79 | +1.07 / +3.59 |

The rate is necessary: at 0.9 the enemy-tower rate alone does not make pushing pay (the pusher loses the pushed and
siege states again at ×1.76). Between 0.5 and 0.7 the ordering and the equal-power values are the same; the rate
sets how much a siege is worth (+2.5 to +5.4 at ×1.76). 0.6 is the middle value.

**The kill bounty** (on the same rules): 0.5 / 1.0 / 2.0 give +5.30 / +5.47 / +5.61 in the neutral lane at ×1.76
and +0.00 / +0.27 / +0.80 in the freeze state at equal power, with no concession in any cell. Without per-tick
exchanges the bounty is not a lever; with them it was the only thing that produced a stand-back concession (§3.3).

**Single mechanisms** (equal / ×1.76; `t_ot6` from the 16-seed run, the rest 8 seeds):

| variant | neutral | pushed | siege | freeze |
|---|---|---|---|---|
| economy only (`t_ot6`) | +0.20 / +2.29 | −0.04 / +0.78 | +0.50 / +2.46 | +0.16 / +2.16 |
| economy + loser (`t_ot6_loser`) | −0.03 / +2.96 | +0.62 / +0.74 | +1.11 / +2.46 | +0.49 / +1.53 |
| economy + durability (`t_ot6_dur` = **`m4_lane()`**) | −0.18 / +5.13 | +1.05 / +3.18 | +1.96 / +4.24 | +0.33 / +5.43 |
| economy + durability + loser (`t_ot6_dur_loser`) | −0.17 / +5.30 | +0.76 / +3.24 | +1.70 / +4.24 | +0.00 / +5.16 |

Durability is what makes the lead decisive (+2.3 → +5.1 at the lane, +2.2 → +5.4 in the freeze); the loser rule
adds nothing measurable on top of it (every difference is within one standard error), so it is left out of the
preset by the smallest-change rule and kept as an opt-in field.

## 9. Correctness, performance, budget

- **Default preservation.** Random-legal-play digests of `m4()`, `m4_emergent()` and `m4_natural()` (300 turns,
  the full resolver output) are pinned and unchanged; the focused suite (the new file plus the natural-wave final
  and prototype pins, zoning, positioning, config, resolver, combat, economy, env, integration, legacy pins,
  invariants, calibration, investment, farm spot, freeze gate, scenarios, time, tower and diagnostics tests, 25
  files) passes: 356 passed, 4 skipped, 2 expected failures, in 82 s. The full suite with `--runslow` was not run.
  The new fields are M4-economy fields: the legacy and m2c rulesets reject them.
- **New tests** (`tests/test_m4_zoning_natural.py`, 16): default preservation and the three digest pins; field
  validation; the n-exchange distribution is the convolution and its draw is the inverse CDF with the no-chance
  convention; two exchanges double expected damage and the tables cover four base HP; the loser rule charges the
  loser only, ties and unilateral trades as before; durability divides all damage taken exactly; the own-tower rate
  caps every posture; the candidate preset's delta is exactly `M4_LANE_FIELDS`; mirror symmetry with a stronger seat
  and the invariants on 400 random turns; the 88-entry observation valid and lossless on 200 turns; an env soak
  (zero sum, exact accounting, minion conservation) and deterministic replay; the candidate's own digest.
- **Performance.** The resolver's step 9a (trade rolls before gold) adds no work; `mitigate` is one integer
  division per side per turn. The full driver ran 135,468,526 resolver turns in 417 s on 11 processes: about
  30k turns per second per process including the policy logic, so the env step stays at the A.1 figure (about
  200 µs). The quick runs are 2–3M turns in 20–50 s.
- **Budget.** Experimental turns: about 260M (the nine-variant full run 135M, the final preset's full run 2.5M, the
  sensitivity run 93M, development quick runs about 27M, trajectories 0.3k). Every run is deterministic from its
  seeds and reproduces from the driver. Results are kept as gzipped JSON (`zoning_natural.json.gz`, the ablation;
  `zoning_natural_t_ot6_dur.json.gz`, the final preset; `zoning_natural_sensitivity.json.gz`, §8) with text
  summaries (`summary_full.txt`, `summary_m4_lane.txt`, `summary_sensitivity.txt`, `trajectories_candidate.txt`).

## 10. Findings, hypotheses, recommendation

**Observed.**

1. Under the A.1 rules power is almost worthless: +0.34 at ×1.76 in the neutral lane, pushing loses at every gap,
   a freeze pays the same at every gap.
2. The weaker champion never stands back in 180 solved policy games across nine rule variants, four states and
   five gaps. Standing back costs 6–13 per 32 turns at the lane; contesting costs 0–4. Only a two-wave kill bounty
   together with per-tick exchanges produced a stand-back concession, in the freeze state.
3. The 0.5 enemy-tower rate and the full-rate collection of crashes under one's own tower are what made pushing lose
   and power worthless outside the lane. The posture base and the retreat reach are proxies too, but removing them
   makes a one-cell retreat free on this geometry.
4. With the three changes of `m4_lane()`, every state's value is monotone in power, a lead pays 1–7 per 32 turns,
   the stronger champion earns it through wave control and attacks on under a third of its turns, and the equal
   lane is even and active.
5. The weaker champion's rational concessions are wave control (it shoves the wave away or lets it crash and farms
   under its tower), position (the stand-off, which keeps half its farm), and tempo (recalls). The stand-at-XP-range
   concession of League does not occur because the wave does not drift toward a champion that stops contesting.

**Hypotheses.**

- Making last-hitting itself a small wave damage (Hold at 1 HP per tick) would make the wave drift toward a zoned
  champion and create the League partial concession dynamically; it would also change every A.1 wave result and
  needs its own study.
- A kill bounty near League's proportion (about two waves) makes deaths matter enough for stand-back concession in
  the freeze state; its effect on the equal lane's variance is the cost.
- An all-in action (a committed multi-exchange fight that a champion starting in reach cannot escape) is the
  League mechanism this game lacks; it would change the action space and is a design-review item, not a tweak.

**Recommendation.** Adopt `m4_lane()` as the preset for the M4.3 PPO smoke test, with the two parameters that were
set by calibration rather than by principle (the own-tower rate 0.6 and the kill bounty 0.5) listed for review.
Do not expect trained agents to stand at experience range against a stronger opponent; expect them to shove,
trade back when caught, recall, and farm under tower, and judge the learned play against §6, not against the
League pattern the geometry cannot express.

### Verdict: PASS WITH CAVEATS

**Why it passes.** The fixed rates that duplicated zoning's consequences are identified and replaced by consequences;
the adopted combat change is minimal and principled (items buy durability) and makes a lead decisive without
making the equal lane bloody; every correctness guarantee holds;
historical presets are untouched; the environment is unchanged in interface and speed.

**Caveats.**

1. Stand-back concession at the lane is not produced by any tested rule; the geometry (farm reach = threat reach,
   Hold = no wave damage) rules it out, and the fix is a design decision (§10 hypotheses).
2. At equal power the equilibrium pushes rather than holds; a passive opponent is punished by 4–5 per 32 turns.
3. A freeze is not worth holding at equal power; it is worth +2 to +6 with a lead.
4. All evidence is scripted-policy evidence over 32-turn windows with linear health valuation at the end; learned
   play is the real test.
5. The own-tower rate (0.6) is calibrated, not derived (§8); 0.5–0.7 give the same ordering.
