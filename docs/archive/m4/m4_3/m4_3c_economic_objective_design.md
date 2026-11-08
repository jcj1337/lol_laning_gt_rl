# M4.3c — Economic Objective Redesign, Laning Termination, Freeze Incentives and Death Timers

**Status: DESIGN (PROPOSED). Nothing in the game, its presets or the trained agents was changed.** Every number below
was measured on the unchanged research preset `GameConfig.m4_lasthit()` with the M4.3b finals and reference
policies, by the isolated diagnostics in `lanegame/experiments/m43c/` (results in `results/m43c/`). No PPO training
was run. The current rules are described in the companion
[laning_game_mathematical_reference.md](../laning_game_mathematical_reference.md) (cited below as "Reference §n").

Branch: `development/m4-3b-validation` (working tree on top of `26f8c83`).

---

## 0. Review gate

**1. Proposed formulation** (§5). A zero-sum, incremental economic reward that counts gold and credited XP,

$$
r_{A,k}=(\Delta G_{A,k}-\Delta G_{B,k})+\lambda\,(\Delta\tilde X_{A,k}-\Delta\tilde X_{B,k}),\qquad \lambda=\tfrac18\ \text{resources per XP (one wave's XP = one wave's gold)},
$$

with $\Delta G$ = farm + plate gold + bounty (unchanged) and $\Delta\tilde X$ the XP credited after the level cap; a
**terminal state** at the end of the turn in which either tower's cumulative damage reaches its last plate
threshold (no bonus, plate gold paid, terminal value 0); and the discount $\gamma=0.99$ **kept and reinterpreted** as
the exogenous hazard that the laning phase ends for reasons outside the lane. No fixed fallback horizon in the game.
Death timers unchanged. Both changes opt-in, behind new configuration fields; `m4_lasthit()` stays bit-identical.

**2. Problems addressed.** XP becomes part of the economic outcome (the brief's issue 2); the game gains a genuine,
Markov, rule-based end (issue 3); the incremental and terminal views of the objective become one (§2.1, §3.4).
**Not** addressed by these changes, and the evidence says so: Clear dominance and the weakness of freezes (issue 1)
— neither XP nor termination makes withholding wave damage profitable against the learned policy (§2.5, §4.1); the
fixed death cost (issue 4) — kept for a separate, evidence-driven decision (§4.2). Issue 5 is addressed by the
reference document.

**3. New exploits and edge cases** (§3.5). Termination is controlled by the attacker (Pressure supplies 95–100% of
tower damage), so a player with a positive income flow gains by **not** taking the last plate: the first measurable
sign of this appears already (withholding wave damage turns slightly positive against a passive opponent once games
can end). Passive games never terminate (a scripted last-hitter pair broke one plate in 1000 turns), so the hazard
must stay. Simultaneous final plates, plates broken while away and the final plate's gold need explicit but simple
rules. XP stops at level 18, so the XP term vanishes in late play.

**4. Evidence** (§2.5, §3.2, §4).
- The proposed end comes late: in self-play of the M4.3b finals the first tower's final plate falls at a median of
  turn 109–143, when the champions are level 15–18 (both reach level 18 at turn 143–146); 24–35% of the discounted
  objective's weight lies beyond it. Against the scripted references the learned policy ends the game by turn 29–55.
- Most of the M4.3b headline margins are earned **after** the proposed end: against `push_punish` +15.2 over 1000
  turns, of which +6.5 before the reference's tower is exhausted. 81% of the states behind the M4.3b validation's
  wave-only conclusions were sampled after a tower's last plate had fallen.
- Adding XP to the objective changes the sign of no wave-only intervention in 120 cells (60 re-scored M4.3b cells,
  60 new laning-phase cells; $\lambda\le\tfrac14$ or level-progress weights up to 2; one cell moves from −0.0 to
  +0.01 ± 0.2): XP is earned at almost the same rate by both players whatever the wave management.
- With termination, withholding wave damage loses *more* against the learned policy (the race to the last plate) and
  turns slightly positive only against the passive last-hitter (+0.3 gold, +0.9 to +1.1 with XP; 8 states, ±1).
- A held freeze costs the frozen-on champion nothing over 40 turns if it walks up (the deficit after 20 turns is a
  delay), costs it 4.9 gold if it steps back, and is profitably broken by the learned policy (+4.8, three plates).
- A death costs 0.7 / 1.4 / 1.25 resources (early / mid / late, bounty included), 1.2–1.5 more than a Recall from the
  same state, and each extra absent turn about 0.4–0.6, because lane income does not grow with level.

**5. Implementation complexity** (§6): small. Two opt-in configuration fields, about 60 lines in the resolver and the
environment, no observation or action change, the existing PPO, evaluation and diagnostics code already handles
terminated episodes. One day including tests and the reference update.

**6. Recommended next steps** (§8). (1) Implement the two opt-in fields and their tests; (2) re-run this study's
screens on the opt-in configuration with the existing finals; (3) decide the game-length calibration (thresholds vs
hazard) **with** the screens (§3.6); (4) only then a 3-seed × 2M PPO run (entropy 0.03), evaluated with the M4.3b
instruments plus the freeze and death harnesses of this study. Freeze and death-timer mechanics are separate
investigations to be screened first with the same harnesses (§4.1–§4.2, §8).

**Stopped here.** No large PPO campaign was launched and the established objective was not replaced.

---

## Contents

1. [Starting point](#1-starting-point)
2. [Part B — gold and XP as the economic objective](#2-part-b--gold-and-xp-as-the-economic-objective)
3. [Part C — ending the laning phase at the final plate](#3-part-c--ending-the-laning-phase-at-the-final-plate)
4. [Part D — freeze incentives and death timers](#4-part-d--freeze-incentives-and-death-timers)
5. [The proposed formulation](#5-the-proposed-formulation)
6. [Engineering changes](#6-engineering-changes)
7. [Research hypotheses](#7-research-hypotheses)
8. [Roadmap](#8-roadmap)
9. [Experiments, files and reproduction](#9-experiments-files-and-reproduction)

---

## 1. Starting point

| issue (brief) | what the current game does (Reference) | evidence |
|---|---|---|
| Clear dominates; freezes and slow pushes do not pay | Clear's capacity ($\ge4$ HP per turn) removes at least the whole enemy spawn (§4.9); crashing is cheap; death is a cheap reset (§6.7) | M4.3b, M4.3b validation; §4.1 here |
| XP is not in the reward | the reward is farm + plate gold + bounty (§8.6); XP only acts through levels | — |
| the game never ends; levels reach 18 | no terminal state (§8.5); XP accrues at ≈4 per present turn, level 18 at 544 XP (§5.4) | §3.2 here: both level 18 at a median turn of 142–158 in 26 of 27 matchups |
| fixed death timer | 3 absent turns at every level (§6.6) | §4.2 here |
| mechanics hard to see as a whole | — | the reference document |

---

## 2. Part B — gold and XP as the economic objective

### 2.1 What the current objective already is

Let $G_s(k)$ be side $s$'s cumulative earned gold after $k$ turns and $D(k)=G_A(k)-G_B(k)$ the gold lead, $D(0)=0$.
The current reward is $r_k=D(k+1)-D(k)$, and summation by parts gives an identity that settles several of the
brief's concerns at once:

$$
J=\sum_{k\ge0}\gamma^k\bigl(D(k+1)-D(k)\bigr)=(1-\gamma)\sum_{k\ge0}\gamma^k D(k+1)=\mathbb E\bigl[D(T)\bigr],
\qquad \Pr(T=k+1)=(1-\gamma)\gamma^k .
$$

**The discounted sum of per-turn gold differences is exactly the expected gold lead at a random end of laning** $T$,
geometric with mean $1/(1-\gamma)=100$ turns. Three consequences:

- Rewarding the per-turn *flow* and rewarding the *stock* (the lead) are the same objective: the flow form does not
  "repeatedly reward accumulated wealth" more or less than the stock form; it evaluates the lead once, at a random
  time. Any economic quantity added as a flow inherits this reading.
- The objective is the familiar laning statistic: the gold difference "at the end of laning" (GD@14 in League
  analytics), with the end drawn at random.
- A real termination time $\tau$ (Part C) composes cleanly: $J=\mathbb E[D(\min(T,\tau))]$ (§3.4).

### 2.2 The design principles as mathematical requirements

| principle (brief) | requirement on the reward $r_k=\Phi(\text{turn events})$ | consequence |
|---|---|---|
| 1. do not reward item purchases | investing must not change $r$ | already true: the reward counts gold *earned*; investing moves gold between $g$ and $S$ (Reference §7.2) |
| 2. do not reward absolute combat power every turn | no term in $AP^\ast$, $H$, $L$ levels themselves per turn | only *flows* of resources; power acts through its consequences |
| 3. do not repeatedly reward accumulated wealth | no term in the stock $E_s$ per turn | flows only; by §2.1 flows ≡ one evaluation of the stock at a random end |
| 4. preserve XP-denial incentives | XP lost by one side must not be neutral | $\lambda>0$ on the *difference* of credited XP |
| 5. preserve the gold / XP distinction | separate accounting, separate observation | two channels; XP is never converted into gold in the state |
| 6. do not reward named strategies | no term in wave actions, positions, crashes, freezes | only economic channels |
| 7. zero-sum, clearly defined | $r_B=-r_A$; every term a difference of per-side quantities | all candidates below |

### 2.3 Candidate formulations

All candidates keep the gold channel $\Delta G$ unchanged and differ in how XP enters.

**B0 — gold only (CURRENT).** $r=\Delta G_A-\Delta G_B$. XP matters only through what levels later earn.

**B1 — constant value per credited XP.** $r=(\Delta G_A-\Delta G_B)+\lambda(\Delta\tilde X_A-\Delta\tilde X_B)$, with
$\Delta\tilde X$ the XP actually credited (zero beyond level 18). Simple, transparent, exact in ledger units
($\lambda=1/8$ is 1,250,000 units per XP). By §2.1, $J=\mathbb E[\text{GD}_T+\lambda\,\text{XPD}_T]$: the classic
"gold and XP difference at the end of laning".

**B2 — level progress.** $r=(\Delta G_A-\Delta G_B)+\mu(\Delta\Lambda_A-\Delta\Lambda_B)$ with
$\Lambda(X)=L(X)+\dfrac{X-\mathcal X(L)}{n(L)}$, the level as a continuous quantity. Each level is worth $\mu$; one
XP at level $L$ is worth $\mu/n(L)$: $\mu/8$ at level 1, $\mu/38$ at level 11, $\mu/56$ at level 17. Because every
level multiplies HP and attack power by the same 1.05, $\Lambda$ is proportional to the log-power bought with XP;
B2 values XP by the power it buys, with diminishing value per XP built in. The cap is automatic.

**B3 — power-equivalent exchange rate.** Value XP at the gold that would buy the same power: a level multiplies $AP$
and HP by 1.05, an investment multiplies $AP$ and divides damage taken by $M(S)$, so a level is worth the $\Delta S$
with $M(S+\Delta S)=1.05\,M(S)$:

| invested $S$ | $\Delta S$ per level | value per XP at level 2 / 5 / 8 / 11 / 14 / 17 |
|---|---|---|
| 0 | 1.1 | 0.10 / 0.06 / 0.04 / 0.03 / 0.02 / 0.02 |
| 16 | 4.8 | 0.44 / 0.24 / 0.17 / 0.13 / 0.10 / 0.09 |
| 32 | 10.5 | 0.96 / 0.53 / 0.36 / 0.28 / 0.22 / 0.19 |
| 64 | 27.3 | 2.48 / 1.36 / 0.94 / 0.72 / 0.58 / 0.49 |

Along actual play ($S$ grows with $L$: about $S\approx5$ at level 5, 25 at level 11, 50 at level 17) the rate is
≈0.1–0.4 per XP and **rises** over the game, because gold's power saturates faster than XP's. As a reward this is
state-dependent and depends on the gold stock, so it rewards relative power by the back door (principle 2); it is
useful only as a calibration reference for $\lambda$.

**B4 — terminal evaluation only.** Pay nothing per turn and $\text{GD}+\lambda\,\text{XPD}$ once at the end. With
$\gamma=1$ and a certain end it equals B1 exactly (telescoping); with $\gamma<1$ it does not and has perverse
incentives (§3.4).

**Comparison.**

| | zero-sum | exact ledger | cap handled | diminishing value per XP | depends on the stock | parameters |
|---|---|---|---|---|---|---|
| B0 | yes | yes | — | — | no | — |
| B1 | yes | yes ($\lambda=k\cdot10^{-7}$) | yes (credited XP) | no | no | $\lambda$ |
| B2 | yes | rational, needs rounding to the ledger | yes | yes ($\propto1/n(L)$) | through the level only | $\mu$ |
| B3 | yes | no | yes | rises instead | **yes (gold stock)** | — |
| B4 | yes | yes | yes | no | final stock | $\lambda$; sparse signal |

### 2.4 Choosing the value of XP; double counting; distortions

**Anchors for λ.** (a) *Per-wave parity*, $\lambda=1/8$: a fresh wave is 1 resource of gold at full collection and
8 XP, so a wave's XP counts as much as a wave's gold. (b) *Power equivalence*: 0.1–0.4 along play (B3); $\lambda=1/8$
sits at its low end. (c) The M4.2e-B zoning study used 0.05 per XP as a sensitivity value. Recommended: $\lambda=1/8$
as the reference point, with $\lambda\in\{0,1/16,1/4\}$ as sensitivity.

**Simple examples** ($\lambda=1/8$, $\mu=1$; costs to the acting champion; Reference §5.4 and §6.7; "measured" =
paired rollouts of §4.1–§4.2):

| event | cost under B0 (gold) | XP the champion misses at the time | naive extra cost (missed XP × λ; B2: × μ / n(L)) | measured extra cost under B1 |
|---|---|---|---|---|
| a 3-turn death at turn ≈40 (level ≈9) | 0.70 (bounty included) | up to 12 | 1.5 (B2: 0.38) | **0.23** |
| the same death at level 18 | 1.25 | 0 (capped) | 0 | 0 |
| a Recall at turn ≈40 | −0.51 (a gain at mid HP) | up to 8 | 1.0 (B2: 0.25) | **≈0** (−0.07 early, +0.01 mid) |
| stepping back from a freeze for 40 turns | 4.89 | 2.5 | 0.31 (B2: 0.07) | 0.31 |
| walking up to a freeze, first 20 turns | 3.88 | 33 | 4.1 (B2: 0.94) | 4.1 after 20 turns, ≈0 after 40 |

The measured costs are far below the naive ones because **XP missed during an absence is mostly only delayed**: the
enemy wave that pushes in while the champion is away dies at its tower after it returns, inside its XP range. Only the
minions that die during the absence itself are lost, and the opponent's resulting XP lead closes when it reaches the
cap.

**Double counting.** In the accounting sense there is none: each XP and each resource is credited once, when it is
earned. In the economic sense the objective counts resources *and* their fruits (XP buys levels, levels win trades
and waves, which earn gold), but so does the current gold-only reward (gold buys investment, which earns more gold).
That is what "economic lead at the end of laning" means, and it is not a distortion as long as nothing is counted
twice at the moment it is earned.

**Distortions to watch.**

- *XP mostly measures presence.* Both champions earn XP from the opponent's minions at the spawn rate whenever they
  are present within 3 cells (Reference §5.4); the XP difference comes almost only from absence. In the progression
  audit the XP lead at the end correlates 0.80 with the gold lead in self-play, and its median is 8 XP (1 resource at
  $\lambda=1/8$) against a median gold lead of 3.5. B1 therefore mainly raises the price of being away, and only
  modestly (+0.23 per early death, about 0 per Recall, measured).
- *The cap.* After level 18 a champion earns no credited XP, while a trailing champion still does: the XP part of a
  lead closes by itself late in the game. With termination before the cap this matters little; without it, the XP
  term vanishes for most of a 1000-turn evaluation.
- *Reward scale.* $\lambda=1/8$ adds up to 0.5 per turn to a per-turn flow of 0.45: rewards roughly double in scale;
  PPO's `reward_scale` and value clip should be checked.
- *Constant λ overvalues late XP relative to its power* (a late level costs 7× the XP of an early one for the same
  ×1.05). B2 corrects this; B1 accepts it for simplicity. The measured effect of the choice is small (§2.5).

### 2.5 Does adding XP change the incentives? (experiments)

**Wave-only interventions re-priced** (`python -m lanegame.experiments.m43c xp_cf`, `results/m43c/xp_cf.json.gz`). The
M4.3b validation's wave-only counterfactuals (the policy's own play with only its wave component forced to Hold,
Thin, a slow-push schedule or the freeze controller for 14–20 turns; the M4.3b full-action freeze plan; 16 paired
rollouts of 300 turns per state) scored on the same rollouts under every objective. Two stages:

- *rescore*: the 253 stored states of the `base` seeds (the gold channel reproduces the stored M4.3b numbers exactly:
  largest deviation 0.0); 79% of them lie after one tower's last plate, 50% with both champions at level 18;
- *early*: 261 new states of the same situations, restricted to the laning phase the proposal keeps (no tower
  exhausted, neither champion at level 18; median turn 35), from games of `base` seeds 0–2 and `ent03` seed 0 against
  the policy itself, `farm`, `push_punish` and `clear_punish`. Against `farm` and `push_punish` such states are rare
  because the policy ends their towers by turn ≈30.

Advantage of the intervention over the policy's own play (resources; laning-phase states; mean over the policies of the per-policy state means):

| opponent, situation | candidate | gold (B0) | B1 λ=1/16 | B1 λ=1/8 | B1 λ=1/4 | B2 μ=1 | B2 μ=2 |
|---|---|---|---|---|---|---|---|
| self, neutral | Hold (14) | −2.29 | −2.40 | −2.51 | −2.74 | −2.34 | −2.38 |
| self, neutral | Thin (14) | −1.17 | −1.21 | −1.24 | −1.30 | −1.19 | −1.20 |
| self, neutral | freeze controller (20) | −2.93 | −3.06 | −3.19 | −3.45 | −2.98 | −3.04 |
| self, freeze-like | freeze controller | −3.42 | −3.54 | −3.67 | −3.93 | −3.47 | −3.52 |
| farm, neutral | freeze controller | −1.32 | −1.44 | −1.56 | −1.80 | −1.39 | −1.46 |
| push_punish, neutral | slow push (14) | −1.72 | −1.86 | −2.00 | −2.29 | −1.79 | −1.86 |
| clear_punish, freeze-like | freeze controller | −3.10 | −3.23 | −3.37 | −3.64 | −3.16 | −3.22 |
| clear_punish, neutral | Thin | −0.62 | −0.66 | −0.71 | −0.79 | −0.64 | −0.66 |

(full tables: `results/m43c/report.md`.) Over all 120 cells (2 stages × 4 opponents × 3 situations × 5 candidates),
in the 300-turn window, no cell is positive under the gold objective, under level progress or at $\lambda\le1/16$;
at $\lambda\ge1/8$ one re-scored cell becomes marginally positive (Thin against `farm` in enemy-pushing states,
+0.01 and +0.04, ±0.2). $\lambda=1/8$ shifts the advantages by −0.06 (re-scored) and −0.15 (laning phase) on average,
between −0.52 and +0.17: counting XP makes withholding wave damage slightly *worse* on average (the withholder's
discounted XP lead falls by 0.4–4 XP).

**Freeze harness** (§4.1): XP only adds a transient to the walk-up response (−33 XP after 20 turns, recovered by
turn 40) and does not change the ranking of the frozen-on champion's responses.

**Death harness** (§4.2): with $\lambda=1/8$ a 3-turn death costs 0.23 more at turn ≈40, 0.27 more at turn ≈100 and
nothing at level 18.

**Conclusion for Part B.** Counting XP is a sound and cheap improvement of the *definition* of the objective (it
makes the reward the familiar gold-plus-XP lead and prices absence more fully), but it is not a mechanism: it does not
change what the current game rewards in the wave dimension, at any tested weight. The brief's hope that XP would
change freezing incentives is not supported by the measurements: on a five-cell lane with a three-cell XP range,
XP is lost almost only to absence (a champion at its own $-1$ misses only kills behind the enemy tower), and even XP
missed while away is mostly only delayed.

---

## 3. Part C — ending the laning phase at the final plate

### 3.1 Now

The game never terminates (Reference §8.5, §9.1); the discount is equivalent to a geometric end of laning with mean
100 turns (§2.1). Training cuts games at 512 turns and bootstraps; evaluation reads $W=1000$ turns.

### 3.2 Progression audit

`python -m lanegame.experiments.m43c progression` (`results/m43c/progression.json.gz`): whole games from $s_0$ on the
unchanged game, 1000 turns, 32 games per matchup (27 matchups). "Turn of plate $j$" pools both towers (the attacker
breaking it). "End" is the proposed termination: the first turn at which either tower's last plate falls. Medians
with the 10th–90th percentiles in brackets.

**Self-play of the M4.3b finals** (8 runs: `base` s0–s2, `ent03` s0–s2, `hist80` s0, `mixed` s0):

| | first plate | second plate | third plate | end (either tower) | both level 18 |
|---|---|---|---|---|---|
| turn | 50–73 (27–111) | 93–114 (52–155) | 132–162 (91–277) | **109–143** (60–203) | **143–146** (141–149) |
| levels at that turn | 10–12 | 14–16 | 17–18 | **15–18** | 18 |
| earned gold per side | 20–29 | 38–46 | 54–67 | 44–57 | ≈55 |

Every self-play game ends within 1000 turns, 62–88% of them by turn 150; the two final plates never fall on the same
turn. The weight $\gamma^{\text{end}}$ of the objective that lies after the end is 0.24–0.35; after both champions
reach level 18, 0.23. Of a 512-turn training game, about 70–75% of the turns lie after the proposed end.

**The learned policy against the references** (`base` s0 and `ent03` s0 against 8 and 4 references): the end comes
at turn 29–55 (levels 7–10), the policy finishing the reference's tower in every game; it dies 11–78 times per 1000
turns. Its discounted advantage over 1000 turns shrinks to 40–68% of itself if play stops at the end:

| opponent | value over 1000 turns | value up to the end | end (median turn) |
|---|---|---|---|
| farm | +10.75 | +5.10 | 36 |
| push_punish | +15.20 | +6.50 | 31.5 |
| clear_punish | +7.99 | +5.43 | 52.5 |
| bully_shove | +15.75 | +6.93 | 33 |
| freeze_punish | +15.75 | +6.34 | 30 |
| attacker of base_s0 (attacker's value) | +1.92 | +1.57 | 167 |

**Reference pairings**: two last-hitters (`farm` vs `farm`, no Pressure) break one plate at a median turn of 415 by
minion hits alone and never a second; the game would never end. Other scripted pairings end at turn 36–126.

**Tower damage comes from Pressure.** In every learned matchup 95–100% of the tower damage is Apply Pressure; minion
hits are negligible. The end of the game is therefore always a deliberate choice of the attacker.

**If the tower ended earlier** (same play, read from the same games): ending at the *first* plate threshold would
come at turn 38–55 in self-play (levels 8–10.5), at the *second* at turn 76–100 (levels 12.5–14).

### 3.3 Why the end comes so late, and how that compares with League

With one turn ≈ 15 s (a wave every 30 s, two turns per wave), League's plates fall off at 14:00 ≈ turn 52, when solo
laners are about level 9–11; the first outer tower typically falls between turns 55 and 80. The simulator's level
curve matches this (level 11 at turn ≈58 at full presence), but its towers do not: the final plate falls at turn
109–143.

Mechanical causes, in order:

1. **XP is a fixed, presence-only flow.** 4 XP per present turn whatever happens (Reference §5.4); both champions
   reach level 18 at turn ≈145 in every matchup. Levels are a clock, not an outcome.
2. **Plates fall only to Pressure.** Pressure needs an own crash, the champion in the enemy tower cell during the
   siege, and no connected trade. One Pressure turn adds $\lfloor400AP^\ast\rceil$: 15 turns per tower at level 1,
   about 6 at mid-game power ($AP^\ast\approx2.5$). In self-play each side pressures only 12–21 turns per 1000-turn
   game, because a Pressure turn needs the own wave to have crashed, exposes the champion (gank 0.13–0.17 per turn)
   and competes with defending the opponent's crashes.
3. **The thresholds were calibrated for 4-HP crashes** (DEPRECATED design: "a 12-HP crash breaks the first plate"),
   before natural waves, the half-wave law and the tower's 3 HP per tick made minion hits negligible.

The end therefore arrives when the champions are level 15–18, which is the situation the brief flagged.

### 3.4 Ways to end the laning phase

| | rule | Markov / stationary? | what the policy optimizes | notes |
|---|---|---|---|---|
| E0 (CURRENT) | no termination; $\gamma=0.99$ | yes / yes | $\mathbb E[\text{lead at }T]$, $T\sim$ Geometric(0.01) | towers can be exhausted with 70% of training play left |
| **E1 (proposed)** | terminal when either $\tau_s\ge6000$; incremental reward; $\gamma$ kept | yes / yes ($\tau$ is in the state and the observation) | $\mathbb E[\text{lead at }\min(T,\tau_{\text{end}})]$ | the attacker controls the end |
| E2 | E1 with a terminal payment of the lead instead of incremental rewards | yes / yes | with $\gamma<1$: $\mathbb E[\gamma^{\tau}\,\text{lead}_\tau]$ | **perverse**: punishes prolonging even when the lead grows (below) |
| E3 | E1 plus a fixed fallback horizon $K$, $\gamma=1$ | only with a countdown in the observation; non-stationary | the lead at $\min(\tau,K)$ | end-of-game effects (free deaths, refusal to invest before $K$) |
| E4 | E1 with a calibrated hazard ($\gamma=0.985$: mean 67 turns; 0.98: 50) | yes / yes | the lead at a random end with a League-like mean | changes patience: the value of investment and of future income |
| E5 | an earlier endogenous end (lower thresholds) | yes / yes | E1 with a shorter laning phase | strengthens pushing; must be screened (§3.6) |

**Incremental versus terminal.** With $\gamma=1$ and a certain end, the two are identical
($\sum_k r_k=\text{lead}_\tau$). With $\gamma<1$ they differ. The incremental reward evaluates the lead at the
*random* end $\min(T,\tau)$, which is the hazard reading of §2.1 and has no preference about when the lead is
realized. The terminal-only payment evaluates $\gamma^{\tau}\,\text{lead}_\tau$: a lead of 5 at turn 100 is worth
$5\times0.366=1.83$, a lead of 6 at turn 120 only $6\times0.299=1.79$. The leader is pushed to end early and the
trailer to stall, for no economic reason. **E2 is not recommended.**

**Why not a fixed fallback horizon.** A horizon $K$ makes the policy depend on time, so the observation must carry a
countdown (the legacy game did). It also creates end effects: in the last turns a death or an unspent purse costs
nothing. Passive play still needs *some* end, and the geometric hazard ($\gamma$) is the only memoryless one, so keep
it. Training and evaluation keep their *run-level* truncations: bootstrapped at 512 turns for training, $W=1000$ for
evaluation. Those are not rules of the game.

### 3.5 What termination changes strategically

**The end becomes an action.** The attacker who takes the last plate receives one resource and stops the flow of
both sides for good. If the expected continuation flow to the attacker is $d$ per turn, finishing is worth
$1-\dfrac{d}{1-\gamma}$ relative to farming on, ignoring the opponent's options. A leader with a positive flow
($d>0.01$ per turn) prefers to *prolong*. The trailer would like to end the game but can do so only by finishing the
*leader's* tower. This is the "rational prolongation" the brief anticipated. Here it is mechanically possible because
plates fall only to Pressure: the leader prolongs simply by not pressuring and by not crashing waves into a nearly
exhausted tower. The latter requires withholding wave damage near the end, i.e. the first non-arbitrary reason in this
game to Hold or freeze.

**First evidence.** In the laning-phase counterfactuals with returns truncated at the proposed end (§2.5 rollouts,
`G_T`), withholding wave damage against the passive last-hitter (`farm`, enemy-pushing states) turns slightly
positive: +0.33 (Hold), +0.35 (freeze), +0.86 / +1.09 with $\lambda=1/8$. The sample is 8 states and the interval is
about ±1, so this is not significant. Against the learned policy, termination makes withholding *worse*: Hold −3.24,
freeze −3.73 at the neutral lane, against −2.29 and −2.93 without termination. When the opponent pushes, the race
to the other's last plate punishes the side that withholds. These are first-order screens with policies that were
not trained for termination.

**Edge cases and rules.**

| case | rule (proposed) | why |
|---|---|---|
| both last plates fall on the same turn | terminal; both receive their plate gold in that turn's reward | symmetric; no tie-break needed; never observed in 864 self-play and reference games |
| the last plate falls while its attacker is away | terminal; plate gold paid (as now) | keeps `plate_gold_requires_presence = False` |
| last-plate gold | paid (1.0) | no bonus beyond an ordinary plate; removing it would create a disincentive to finish |
| neither tower falls | the game continues; it ends by the hazard ($\gamma$) | a passive pair never ends otherwise |
| plate gold in the objective | kept (3 resources per tower) | tower income is economic; it is small |
| the observation | unchanged | $\tau$ and plate progress of both towers are already observed: the end is predictable from the state |
| terminal value | 0 for both sides | zero-sum, no bonus |

**Training and evaluation.** PPO: `terminated=True` on the terminal turn, so GAE drops the bootstrap there (already
supported by `compute_gae`); the 512-turn time limit stays as a bootstrapped truncation for games that do not end;
episodes become ~130 turns in self-play. Segments of 128 turns span episode boundaries as they already do. Evaluation:
`evaluate_discounted` already stops a game when it is `done`, the paired design and the truncation bound remain valid
(termination only shortens the tail), and headline values change meaning (§3.2: most of the current margins are
earned after the end). Counterfactual rollouts stop at the end (`branch.rollout` already breaks on `done`).

### 3.6 Game-length calibration

The brief asks whether durability, thresholds, XP and economic scaling make games too long, and for mechanically
justified alternatives rather than an arbitrary earlier end. The measurements give:

| lever | effect on the end (self-play) | incentive effects to evaluate |
|---|---|---|
| none (thresholds 1,600 / 3,600 / 6,000) | turn 109–143, levels 15–18 | the laning phase includes the level-cap period; XP stops mattering at its end |
| scale the thresholds (e.g. ×0.6 → end near today's second plate) | turn ≈76–100, levels 12.5–14 | fewer Pressure turns per plate: pushing and crashing become more decisive, plausibly strengthening Clear |
| scale ×0.27 (end near today's first plate) | turn ≈38–55, levels 8–10.5 (League's 14:00) | all three plates fall within the damage of one of today's plates: each Pressure turn becomes about four times as decisive |
| slower XP (steeper thresholds) | levels at the end fall; the end itself does not move | the level curve currently matches League's; slowing it breaks that anchor |
| hazard $\gamma$ (E4) | the *expected* end moves to 67 (0.985) or 50 (0.98) turns | less patience: investment and long plans are worth less; deaths cost less in discounted terms |
| Pressure damage or Pressure legality | as threshold scaling, through a different channel | changes the exposure / reward balance of standing at the enemy tower |

**Recommendation.** Do not shorten the game in this milestone. Adopt E1 + hazard as the objective and treat the
threshold scale and $\gamma$ as **one** calibration decision, taken with the screens of this study:

1. the progression audit (termination turn and level distributions);
2. the wave-only harness with termination (does withholding near the end become profitable for the leader, and is
   that the "rational prolongation" or a stall?);
3. the freeze and death harnesses.

Run them on the opt-in configuration for two or three candidate pairs: thresholds ×1 / ×0.6, $\gamma$ 0.99 / 0.985.
Choose before any training. If the intended laning phase is League's, the end should come at turn ≈55–80 at levels
≈10–13; the current game reaches that point at its first or second plate.

---

## 4. Part D — freeze incentives and death timers

### 4.1 Does an early freeze create the intended dilemma?

`python -m lanegame.experiments.m43c freeze` (`results/m43c/freeze.json`, Reference §5.6): from the deterministic
freeze start (turn 51, level 10), A holds B's wave at its own $-1$ with the threshold controller; B answers with one
fixed response; 64 paired seeds; gaps equal, A ahead (×1.76 attack power, ×3.1 duel ratio) and A behind.

| B's response (equal power) | 20 turns: B gold − A gold, XP diff, HP lost | 40 turns: gold diff, XP diff, HP lost, deaths |
|---|---|---|
| walk up, last-hit (Hold) | −3.88, −33, 0.69 | **+0.20**, +0.4, 0.88, 0.09 |
| step back (Safe) | −2.86, −11, 0.06 | **−4.89**, −2.5, 0.08, 0 |
| trade every turn | −4.95, −19, 1.55 | −6.46, −13, 2.97, 0.77 |
| break: Clear + Hold | −0.06, −7, 0.81 | +0.72, −7, 1.44, 0.12 |
| break: Clear + punish + Pressure | +1.83, −7, 1.15 | **+4.70**, −3, 2.22, 0.61 |
| the M4.3b policy | +3.38, −6, 1.21 | **+4.79**, −11, 2.06, 1.09 |
| (A does not freeze; B walks up) | −2.44, −18, 0.12 | −2.93, −15, 0.18, 0 |

With A ahead, walking up costs B 1.4 gold over 40 turns and the learned policy's break still pays +4.4. With A
behind, breaking pays +6.7. A freezer that also punishes the walker (Trade when B starts in reach) does worse: a
connected trade halves the trader's farm, and against the learned policy that freezer ends 8.0 behind over 40 turns.

**Why the dilemma is only partly present**, by the brief's five candidate causes:

| cause | verdict | evidence |
|---|---|---|
| wave mechanics | **not the cause**: the freeze is maintainable | held 15–17 of the first 20 turns against a walker; a freeze band exists (Reference §4.8) |
| farming reach | **partly**: stepping back costs gold, walking up does not | B at its $+1$ collects the held wave's kills at 0.9; B at $-1$ collects nothing beyond one cell |
| zoning | **a cause**: there is no passive zoning; zoning works only through trades, and trades cost the trader half its farm | `zoning_rule = "none"`; the punishing freezer loses |
| combat threat | **the main cause**: walking up is paid in HP only, and HP is refilled cheaply | 1.8 ganks per 20 turns at exposure $+1$, ≈0.9 bars per 40 turns, 0.09 deaths; the tower never shoots a farming champion; a death costs ≈1.3 |
| reward valuation | **not a cause**: XP changes no ranking | §2.5; on a five-cell lane with a three-cell XP range XP is lost almost only to absence |

**Mechanic hypotheses**, to be screened with this freeze harness and the wave-only harness before any training; none
is implemented:

- (D1) make HP last: a higher death cost (§4.2) raises the price of walking up into gank range;
- (D2) aggro: the tower, or the freezer's minions, threaten a champion that trades near them;
- (D3) XP geometry: an XP range of 2 on the five-cell lane would make stepping back cost XP as well as gold;
- (D4) danger under the enemy tower as a consequence rather than a rate (the 0.9 enemy-tower collection rate
  replaced the old 0.5 danger proxy in M4.2e-B; the danger it stood for is now only gank exposure).

### 4.2 Death timers

`python -m lanegame.experiments.m43c death` (`results/m43c/death.json`): from self-play states of the `base` finals
in three phases, 24 states each, 16 paired rollouts of 300 turns. The focal champion is replaced by "just recalled"
or "just died with an absence of $d$ turns" ($d>3$ by an absence-extension device outside the engine; the engine is
unchanged). Cost = value lost relative to the unchanged state; the 0.5 bounty is included for deaths.

| phase (turn, level) | Recall | death $d$=1 | $d$=3 (current) | $d$=5 | $d$=8 | $d$=12 | cost per extra turn |
|---|---|---|---|---|---|---|---|
| early (40, 8.6), gold | −0.51 | −0.03 | **0.70** | 2.14 | 3.85 | 5.87 | 0.57 |
| early, gold + XP (λ=1/8) | −0.58 | −0.09 | 0.93 | 3.02 | 5.70 | 8.90 | 0.89 |
| mid (101, 14.5), gold | −0.11 | 0.39 | **1.40** | 2.71 | 4.14 | 5.65 | 0.47 |
| mid, gold + XP | −0.10 | 0.40 | 1.67 | 3.35 | 5.26 | 7.39 | 0.64 |
| late (245, 18), gold (= gold + XP) | −0.31 | 0.21 | **1.25** | 2.01 | 3.36 | 4.93 | 0.41 |

(± 0.3 for every entry; negative = a gain.) Readings:

- **Income per turn does not grow with level**, so the cost of an absent turn does not either: 0.4–0.6 resources,
  the lane income. A fixed timer makes a death cost the same at every level; a level-dependent timer would make it
  costlier only by adding turns.
- **A death is a Recall plus one turn plus 0.5.** $d=1$ costs what a Recall does plus the bounty. Both refill HP and
  invest all gold, so there is no special "death-as-shopping" advantage. A death replaces a Recall whenever walking
  back would cost more than about 1.2–1.5, which is often true at the front.
- **XP raises the early cost, not the late one**: +0.23 / +0.27 / 0 per current death at λ=1/8, more for long timers.
- **Power differences** do not change the per-turn cost; a lead persists through the absence because levels and
  investment are kept.

**Candidate bounded schedules** (EXPERIMENTAL, not implemented), with predicted costs from the measured slopes (gold
objective):

| schedule | levels 1–5 | 6–11 | 12–15 | 16–18 | predicted cost of a death early / mid / late |
|---|---|---|---|---|---|
| current: 3 | 3 | 3 | 3 | 3 | 0.7 / 1.4 / 1.25 |
| S1: $2+\lceil L/6\rceil$ (3–5) | 3 | 3–4 | 4–5 | 5 | 1.3 / 2.7 / 2.0 |
| S2: $\operatorname{clip}(1+\lfloor L/3\rfloor,1,7)$ (League-shaped) | 1–2 | 3–4 | 5–6 | 6–7 | 0.7–1.3 / 2.7–3.2 / 2.9 |

S2 makes deaths before level 6 almost free ($d=1$–2 cost −0.03 to 0.7 at level ≈9; League's early death cost comes
from kill gold, about 2.4 waves, not from the timer). In this simulator the **bounty** is the lever for early death
cost and the timer the lever for late death cost. Recommendation: no change now. If the freeze dilemma (§4.1, D1) is to rest on HP risk, screen S1 and a two-wave
bounty with the freeze and death harnesses first. The M4.2e-B study found a two-wave bounty to be the one change that
produced a stand-back concession.

---

## 5. The proposed formulation

$$
r_{A,k}=\bigl(\Delta G_{A,k}-\Delta G_{B,k}\bigr)+\lambda\bigl(\Delta\tilde X_{A,k}-\Delta\tilde X_{B,k}\bigr),
\qquad
J=\mathbb E\Biggl[\sum_{k=0}^{\tau_{\text{end}}-1}\gamma^k r_{A,k}\Biggr],
$$

$$
\tau_{\text{end}}=\min\{k:\ \max(\tau_A(k),\tau_B(k))\ge6000\}\ (\infty\text{ if never}),\qquad \gamma=0.99,\qquad \lambda=\tfrac18 .
$$

Equivalently, $J=\mathbb E\bigl[\text{GD}(T')+\lambda\,\text{XPD}(T')\bigr]$ with $T'=\min(T,\tau_{\text{end}})$ and
$T$ geometric with mean 100 turns, independent of play: **the expected gold-and-XP lead at the end of the laning
phase**, which ends when a tower's last plate falls or, failing that, at a random time.

| component | value | status |
|---|---|---|
| gold channel | farm + plate gold + bounty, as now | CURRENT |
| XP channel | credited XP (zero beyond 544), $\lambda=1/8$ (sensitivity 0, 1/16, 1/4) | PROPOSED |
| terminal state | end of the turn in which either $\tau$ reaches 6,000; simultaneous allowed; no bonus | PROPOSED |
| discount | 0.99, read as the laning hazard; calibration with the thresholds (§3.6) | CURRENT value, PROPOSED reading |
| fallback horizon | none in the game; run-level truncation only | CURRENT |
| death timer, bounty, freeze mechanics | unchanged; separate screens | CURRENT |

Level progress (B2, $\mu=1$) is the alternative if the reviewer prefers diminishing value per XP; the measurements
cannot distinguish them (§2.5).

---

## 6. Engineering changes

| item | change | size |
|---|---|---|
| `config.py` | `xp_reward_weight: float = 0.0` (λ; validated as an exact ledger multiple per XP; ruleset `m4` only); `terminal_rule: str = "none"` (`"final_plate"`); preset `GameConfig.m4_lasthit_xp_end(**kw)` = `m4_lasthit()` + both fields; added to `M4_ECONOMY_FIELDS` so legacy rulesets reject them | ~30 lines |
| `champions_m4.resolve_turn_m4` step 15 | `reward_units += λ_units (xp_gain_A − xp_gain_B)`; `M4Turn.terminal` | ~10 lines |
| `env/core.LaneEnv._step_m4` | `terminated = cfg.terminal_rule == "final_plate" and max(tau) ≥ θ_last`; `info["reward_components"]` (gold, XP) | ~10 lines |
| PPO, shared self-play, GAE | none (termination already supported); check `reward_scale` and `vf_clip` at λ=1/8 | 0 |
| `env/discounted.py` | none functionally; the truncation constant Ω gains the XP channel's analogue of its farm terms (the same bound on minion HP dying per turn, times $\lambda\xi$ instead of the farm value) | ~5 lines |
| diagnostics, M4.3b instruments | none (`branch.rollout` stops at `done`) | 0 |
| tests | pins: `m4_lasthit()` digests unchanged; reward identity with λ; termination at exactly the crossing turn, simultaneous case, no termination when off; GAE with terminal; Ω bound | ~120 lines |
| docs | reference §8.6, §9, §0 tables and an example; README / roadmap | — |

---

## 7. Research hypotheses

| | hypothesis | test | instrument |
|---|---|---|---|
| H1 | XP in the objective does not change the wave-management equilibrium | trained policies' wave mix and wave-only advantages under λ=1/8 vs 0 | PPO 3 seeds; `val_wave_cf`, `m43c xp_cf` |
| H2 | termination at the final plate creates a measurable "prolong when ahead" behaviour near exhausted towers | Pressure and Hold/Thin rates of the leader as a function of $\tau_{\text{enemy}}$ | conditional tables (M4.3b `behavior`) |
| H3 | with termination, early Recall/death decisions matter more (the lead is evaluated sooner) | death and Recall costs by phase | `m43c death` |
| H4 | the freeze dilemma depends on the price of HP, not on XP | freeze harness under D1 levers (bounty, timer schedule) | `m43c freeze` |
| H5 | a calibrated end (thresholds and γ) yields end levels of ≈10–13 without strengthening Clear dominance | progression audit + wave-only harness per calibration | `m43c progression`, `xp_cf` |
| H6 | the post-termination farming duel was where much of M4.3b's margin over references came from | headline values to the end vs to W | `m43c progression` (already: 40–68% before the end) |

---

## 8. Roadmap

1. **Review** this document and the reference (now).
2. **Implement** `xp_reward_weight` and `terminal_rule` as opt-in fields with tests; update the reference in the same
   change (§13 log). No training.
3. **Screen** the opt-in configuration with the existing finals: progression audit (termination distribution),
   wave-only harness with termination, freeze and death harnesses. Minutes each.
4. **Calibrate** the end (thresholds × {1, 0.6}, γ ∈ {0.99, 0.985}) with step 3's screens; record the decision.
5. **Train** 3 seeds × 2,048,000 transitions, entropy 0.03, on the chosen configuration; evaluate with the M4.3b
   instruments (references, cross-play, attackers at W = 1000, conditional behaviour, wave-only counterfactuals) and
   this study's harnesses.
6. **Separately**, screen the freeze levers D1–D4 and the death schedules S1/S2 + a two-wave bounty with the freeze and
   death harnesses; train only what moves the screens.

Stop before step 5 for review.

---

## 9. Experiments, files and reproduction

| stage | what | runtime (9 workers) | output |
|---|---|---|---|
| `examples` | the reference's worked examples (wave actions, a constructed reward turn, tables) | 10 s | `results/m43c/examples.{md,json}` |
| `progression` | 27 matchups × 32 games × 1000 turns, plate / level / wealth events | 70 s | `progression.json.gz` |
| `xp_cf` | re-scored M4.3b wave-only states (reproduces the stored gold advantages exactly) and new laning-phase states, all objectives on the same rollouts, with and without termination | 23 min | `xp_cf.json.gz` |
| `freeze` | Scenario A / B windows, 64 paired seeds, 3 gaps | 35 s (3 workers) | `freeze.json` |
| `death` | Recall and deaths of 1–12 turns from 72 states in three phases | 10 min (3 workers) | `death.json` |
| `report` | the tables of this document | 2 s | `report.md` |

```bash
python -m lanegame.experiments.m43c examples progression xp_cf freeze death report --workers 9
python -m pytest tests/test_m43c_diagnostics.py
```

Seeds: SeedSequence roots 71,000–74,999 (new, disjoint from every earlier study); `xp_cf` re-scoring uses the M4.3b
validation's own roots. Code: `lanegame/experiments/m43c/` (`common.py` objective channels, `rollouts.py` channel
rollouts and the absence-extension device, `progression.py`, `xp_cf.py`, `freeze.py`, `death.py`, `examples.py`,
`report.py`). The engine, presets, observation, PPO and the trained checkpoints are unchanged.

**Limitations.** Every objective comparison re-prices play by policies trained on the gold objective without
termination (first-order screens, not the value of re-trained policies). Laning-phase states against `farm` and
`push_punish` are few (8–18 per cell) because the learned policy ends those games by turn ≈30. The freeze scenario is
one constructed start with scripted responses. Death-timer costs for $d>3$ use an external device and the policies'
observation of absence stays within its trained range by construction; the opponent never sees an absence longer than
3 at once.
