# M4.5a — Unified positional mechanics: design specification

*2026-10-14. Written before implementation; the implemented rules are recorded in the
[M4.5a report](m4_5a_unified_mechanics_report.md) and the [mathematical reference](laning_game_mathematical_reference.md)
§14. Everything here is the EXPERIMENTAL candidate preset `GameConfig.m45_unified()` =
`m4_four_actions_econ_aggro()` plus the opt-in fields below. No earlier preset changes.*

## 1. Scope and principles

Six mechanics are added together because they interact: a shared off-lane Vision node **W** with a contestable
Ward action, XP range 2, minion retaliation after successful champion damage, positional minion aggro, exclusive
per-tick minion targeting with level-scaled minion damage on champions, and capped wave maintenance during a
connected Trade. One supporting change makes the champion's wave damage respect distance (a **reach ball** instead of
the historical one-sided reach line), which the Safe / Shove and W requirements need.

Principles: (i) every new rule is a configuration field whose default reproduces every earlier preset exactly;
(ii) nothing is scripted by coordinate: outcomes follow from movement, wave state, combat and chance; (iii) both actions
are still chosen simultaneously at the start of the turn and the chance vector is still exactly ten uniforms per turn,
drawn whatever happens; (iv) all arithmetic is in exact integers or rationals (deterministic replay, mirror symmetry);
(v) the reward is unchanged (gold earned, zero-sum): no new term rewards or penalizes any action.

## 2. The movement graph and the node W

Champion positions, in each side's own coordinates: $P=\{-2,-1,0,+1,+2,W\}$. Lane cells form the path
$-2-(-1)-0-(+1)-(+2)$; **W is attached to the lane cell 0** (the midpoint, which is the same physical cell for both
sides), so W is mirror-invariant: in A's coordinates and in B's coordinates it is the same node.

$$
d(\ell,\ell')=|\ell-\ell'|,\qquad d(\ell,W)=1+|\ell|,\qquad d(W,W)=0 .
$$

Kill cells of minions may lie outside $[-2,2]$ (defenders behind a tower at $+3$); the same formula applies. The other
side's cell $y$ seen in my coordinates is $\mu(y)=-y$ for a lane cell and $\mu(W)=W$.

* **Movement.** A champion moves along a shortest path toward its target, at most `champion_speed` = 2 edges per turn.
  Entering or leaving W always passes through 0: from $\pm1$ or 0 a champion reaches W in one turn; from $\pm2$ it needs
  two. No teleportation.
* **Contact** (trades, Recall interruption, evasion, pins): $d(x,\mu(y))\le$ `trade_range` (1), keeping the historical
  own-tower rule (a champion in its own tower cell is reachable only from inside that cell). W is in contact with 0 and
  with W.
* **Chase target.** The node nearest to the chaser's start that is in contact with the opponent's reference cell; ties
  go to the smaller lane cell, W last (as before, among the lane cells).
* **Exposure** (gank probability and damage): W counts as the own cell `ward_exposure` = 0 (the lane midpoint): a
  champion in W is as exposed to ganks as one standing at the lane.
* **Legality.** Pressure needs $d(\text{pos},+2)\le2$ (unchanged on lane cells; never from W). Recall needs a lane cell
  $\le-1$ (never from W).

## 3. Ward and contestable placement

The historical siege objective *Establish Vision* is replaced (`vision_rule = "ward"`): the action code 30
(= Last Hit wave + Hold + Vision, label **WARD**) means *walk to W and place a ward*. Every other Vision code is never
legal. Ward is legal for a present champion with $d(\text{pos},W)\le2$, i.e. at $-1$, 0, $+1$ or W, with or without a
siege.

* The warder moves to W and ends there. Mode **ward**: no wave damage, no gold (it is busy placing); XP as any present
  champion by graph distance; gank exposure of W; trade-exposed like a Hold champion (hit modifier 0); it never evades.
* **Interruption.** The placement fails iff the enemy's connected Trade deals positive damage to the warder this turn
  (after mitigation), or the warder dies this turn. A Trade that does not connect, or connects and misses, does not
  interrupt. Ganks and minions do not interrupt (they are not the enemy champion's attack).
* **Effect** (unchanged numbers, provisional): a placed ward sets the side's vision timer to `vision_turns` = 6 at the
  end of the turn; for the next 6 turns (not the current one) the side's gank occurrence is multiplied by
  $1-0.75$. Timers are per side and independent; both sides may ward in the same turn.
* **Return.** The next turn the champion walks back (W → 0 → $\pm1$ in one turn). Minions keep moving, fighting and
  crashing meanwhile; kills during the Ward turn are not collected by the warder.

## 4. Resolution order of one turn

1. Both actions are committed simultaneously.
2. **Probe** (unchanged): the farm spots, from a lane step with every champion that plays the lane at full support
   (a warder, like a recaller, does not) and no minion targeting.
3. **Champion movement** (targets, chase, Ward path), Recall interruption, evasion: every champion's end cell. Champions
   stand at their end cells for both ticks of the turn.
4. **Trade contact** and the **trade rolls** (chance vector; start-of-turn hit probabilities, as before). The single
   exchange of a connected Trade resolves on **tick 0** of the turn.
5. **Modes, reach and wave budgets.** A connected trader keeps a capped wave budget (§6).
6. **Lane step, per tick $t=0,1$:** (a) **targeting** from the state at the *start* of tick $t$ (§5); (b) minion
   movement, merging, entry crash; (c) each wave whose designation holds and whose front is within `minion_range` = 1
   of the designated champion fires its whole tick firepower at that champion; (d) contact fight, siege fight,
   champions' wave damage, minion tower hits and tower fire, with the firing waves dealing **no** minion damage on that
   tick.
7. Gold and XP (graph distances, XP range 2), ganks, tower shots, the summed damage of the turn (trade + tower +
   gank + minions, divided by $M(S)$ by the existing durability rule), death, ward placement, plates, progression,
   ledger, absence, reward (unchanged).

## 5. Minion targeting (per tick)

Each side's wave (its leading blob: front packet plus packets queued within one cell) allocates its firepower on
tick $t$ to exactly **one** target category: enemy minions (lane front, or the crashed mass under its tower) **or**
the enemy champion. Damage dealt on an earlier tick is never reassigned.

**Designation on tick $t$** (start-of-tick state; side $s$'s wave, enemy champion $o$ present on the lane or in W):

1. **Retaliation** — $o$'s connected Trade dealt positive damage to $s$'s champion this turn, and
   $t\ge$ `minion_retaliation_delay` (= 1: the exchange lands on tick 0, the minions react from the next tick).
2. **Positional aggro** — $o$ did **not move** this turn (start cell = end cell), and at the start of tick $t$ it
   stands on a lane cell $c$ that holds $s$'s front, and $o$'s own minions cannot take the attacks: $o$ has no front,
   or its front is strictly behind $c$ (in $o$'s coordinates), and $o$ has no crashed mass in that cell (a champion
   pressuring with its siege is covered by it). *Why "did not move":* champions move once per turn, before the ticks,
   and a Hold champion moves to the cell its own wave will reach this turn; a winning wave's champion would otherwise
   be targeted at tick 0 by the very wave its own wave is about to push back (found in the first demonstrations). A
   moving champion is in transit; if it stays, its new cell's contact counts from the next turn — the original
   "starts a turn occupying the enemy wave's position" — while a stationary champion is re-checked every tick, so a
   wave walking into its cell during tick 0 targets it on tick 1.
3. Otherwise normal minion-versus-minion combat.

Both designations name the same champion (the only enemy champion); the record keeps the first that applies.
**Execution:** after the tick's movement, the designated wave fires iff its front cell $x$ is within
$d(x,\mu(\text{pos}_o))\le$ `minion_range`; otherwise it fights minions normally on that tick. A champion one cell away
from the enemy wave at the start of a tick is not targeted by positional aggro on that tick; a wave that walks into a
champion's cell during tick $t$ can target it from tick $t+1$. Crashed masses never target champions.

**Why per tick.** The tick loop already exists and every quantity the designation needs (packets, crashed masses,
fixed champion cells) is available at the start of each tick, so per-tick targeting adds one designation step and
one firing step to `mechanics._tick_natural` without restructuring the combat engine. It lets a wave fight minions on
tick 0 and turn on a champion on tick 1 after the trade lands (as requested), and it ends positional aggro as soon as
the geometry stops holding. A turn-locked rule would need a separate notion of "intended" versus "possible" targets
and would either waste firepower when the champion leaves range or double-count it.

**Damage on a champion.** A wave's tick firepower $f_t(h)$ (the half-wave Lanchester law, unchanged) is counted in
*minion damage units*; the units $m$ a champion takes in a turn become

$$
D_{\text{minion}}(m,L)=\max\Bigl(1,\ \Bigl\lfloor\frac{\alpha\,m}{\rho^{\,L-1}}\Bigr\rceil\Bigr)\ \text{centi-HP}\quad(m>0),
\qquad\alpha=50,\ \rho=1.08,
$$

($L$ = the victim's start-of-turn level), added to the turn's other damage and divided by the victim's $M(S)$
(`invest_durability`), so $R(L,S)=\rho^{L-1}M(S)$. Relative to $H(L)$: an 8-HP wave firing for a whole turn (2 units)
costs 17% of a level-1 bar and about 1% at level 18 with 128 resources invested; never exactly 0.

## 6. Wave maintenance during a connected Trade

`trade_wave_rule = "capped"`: a connected trader keeps its farm-mode reach and spends
$B'=\min\bigl(B_w,\ \lfloor\varphi C\rceil\bigr)$, $\varphi=0.5$, so Last Hit, Thin and Slow Push are unchanged
(their own cap is $\lfloor C/2\rceil$) and Shove is halved. It still collects gold at `trade_farm_factor` (0.8).

## 7. Reach ball

`wave_reach_rule = "ball"`: a champion's wave damage reaches enemy minions at cells $x$ with $d(x,\text{end})\le$
`farm_reach` (1), capped at the reach cap ($+3$ from the enemy tower cell); Safe and retreating Yield additionally only
$x\le\min(\text{end},-1)$. From W: only the lane cell 0. Historically the reach was a one-sided line with no lower
bound, so a forward champion damaged minions (and the crashed mass under its own tower) arbitrarily far behind it.

## 8. Spaces

Actions: unchanged 61 codes; code 30 is Ward, codes 31–32, 36–38 and the Slow Push Vision codes are never legal.
Observation: the champion cell one-hot gains a W entry per champion: **90** entries (88 before). Networks trained on
earlier presets are refused by the M4.4 compatibility guard (shape mismatch).

## 9. Parameters (all provisional, M4.5b calibrates)

| field | value | role |
|---|---|---|
| `vision_rule` | `"ward"` | shared node W, Ward action, no siege Vision |
| `ward_exposure` | 0 | gank exposure cell of W |
| `vision_turns`, `vision_reduction` | 6, 0.75 | unchanged baselines |
| `xp_range` | 2 | graph distance |
| `minion_retaliation`, `minion_positional_aggro` | True, True | targeting triggers |
| `minion_range` | 1 | graph distance at which a wave's front hits a champion |
| `minion_retaliation_delay` | 1 | ticks between the trade exchange (tick 0) and retaliation |
| `minion_damage_alpha` | 50 | centi-HP per minion damage unit at level 1 |
| `minion_damage_growth` | 1.08 | per-level resistance growth $\rho$ |
| `trade_wave_rule`, `trade_wave_fraction` | `"capped"`, 0.5 | wave budget of a connected trader |
| `wave_reach_rule` | `"ball"` | distance-limited wave damage |
