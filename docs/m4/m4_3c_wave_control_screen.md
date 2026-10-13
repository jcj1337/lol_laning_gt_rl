# M4.3c — Quick Wave-Control Screen (Hold1 / Adaptive Thin)

**Status: SCREEN. The research preset `m4_lasthit()`, every other preset and the trained agents are unchanged.** Two
opt-in configuration fields were added and are off by default (EXPERIMENTAL): `wave_hold_cap` and
`wave_thin_rule = "adaptive"`. Every number comes from deterministic scripted play through the M4 resolver with no
chance events. The runs reuse the M4.2e-B `neutral` and `freeze` starts and the M4.2e-C threshold freezer, and one
fixed-policy check uses the M4.3b final (`lanegame/experiments/m43c/wave_control.py`, results in
`results/m43c/wave_control.*`). No PPO was trained. Rules are cited from the
[mathematical reference](laning_game_mathematical_reference.md) ("Reference §n").

Branch: `development/m4-3b-validation` (working tree on top of `26f8c83`).

---

## 0. Answer

**Recommendation: adopt Hold1 on its own. Do not adopt Adaptive Thin yet** (§8).

Hold1 does what was intended and cost nothing measurable in this screen. Adaptive Thin works as a controller: it never
removes more than the need, a weak champion corrects only part of an imbalance, and it stops at the target. But it
does not fix the case that matters most for freezing. When the opponent steps back or retreats, the freezer's own wave
still grows and the freeze breaks after 7–12 turns. The cause is that the opponent's minions alone kill fewer HP per
turn than the freezer's wave receives. No rule on champion damage can change that, because the least a champion can
deal is 0.

| question | answer |
|---|---|
| Does Hold1 keep ordinary farming and progression? | **Yes.** Baseline Hold is already 1 HP below capacity C = 12, which covers every level-3/6/10 tier, even at ×1.76 attack power. So nothing changes there. Equal-power Hold/Hold lanes stay at the lane cell for 60 of 60 turns with no crash, in every configuration. In learned self-play the policy chooses Hold on 4.75 of 160 turns per side, and gold, XP and level timing are unchanged. |
| Does Hold1 stop a strong champion from pushing by last-hitting? | **Yes.** A stronger last-hitter against a last-hitter crashes 5 times in 60 turns at L13 A+ (Hold 2) and 11 times at L18 A++ (Hold 3). With Hold1 it crashes 0 times, and the contact stays at the lane cell. |
| How long does a freeze last (threshold freezer)? | Against B walking up / stepping back / retreating / clearing, the longest hold is 18 / 6 / 6 / 2–4 turns, and the freeze first breaks at turn 19 / 7 / 7 / 2–4. This holds at levels 3–10, at equal power and with A at ×1.76, and is identical in all three configurations. |
| Does a stronger freezer break its own freeze under the baseline? | **Yes, through Hold.** At C ≥ 12 the baseline Hold is 2–3 HP. Against a walk-up, the longest hold drops to 11 turns (L13 A+), 8 (L18 +64) and 2 (L18 A++). With Hold1 it is 18 at every power. A freezer that Thins every turn breaks the freeze under the baseline at every power (longest hold 0–6), because baseline Thin is a slow push. |
| Can the opponent still break a freeze? | **Yes, in all configurations.** B's Clear breaks it within 1–4 turns at every power. Against the threshold and adaptive freezers, B's wave reaches A's tower 10–21 times in 40 turns. |
| Adaptive Thin: does a weak champion correct only partly, a strong one without overshoot, and does it stop at the target? | **Yes.** From 8 against 24 HP at the lane (need 13), a Thin budget of 2–3 (levels 3–10) removes 2–3 HP per turn, and B's stack crashes into A's tower 3–4 times in 12 turns. A budget of 11 removes 11, 7, 3, 1 and holds the contact at the lane for all 12 turns. Baseline Thin with the same budget removes 36 HP more than needed, and A crashes 5 times. Adaptive Thin never exceeded the need (a test enforces this). |
| Is Thin distinct from Hold and Clear? | **In an imbalance, yes.** Hold lets every stack of 12 HP or more through, Clear pushes A's wave into B's tower, and adaptive Thin holds the lane when its budget is large enough. **In a balanced lane, no.** Adaptive Thin vs Hold is stationary, just like Hold vs Hold, from level 6 on. Thin no longer builds a slow push, so only Clear can. |
| When B retreats, does the friendly wave still grow and push out? Why? | **Yes, in every configuration** (§4). The adaptive freezer deals 0 on 36 of 40 turns, yet its front grows from 6 to 23 HP by turn 11, the freeze breaks at turn 12, and A crashes 5 times. A's front gains 4 HP per turn and loses only the 1–3 HP that B's minions deal, because B's champion no longer kills A's minions. |
| Can a conceding opponent recover? | Against the threshold freezer B has little to lose: the freeze collapses toward B within 7 turns anyway. Against the adaptive freezer, conceding for 20 turns costs B 2.7–6.3 gold and up to 62 XP. After 40 turns of walking up, the gold gap shrinks to 0.7–2.2 at levels 3 and 10 but stays at 2.7 when A is at ×1.76. XP lost while retreating stays 8–37 below (§6). |
| Economic advantage (gold; gold + XP/8)? | When B picks its best response, freezing is not more profitable than Clearing. On gold + XP, Clear beats the adaptive freezer at all nine tiers, by 0.85–7.0. On gold alone the adaptive freezer is ahead at two tiers, by at most 0.15. At L10, Clear guarantees A +0.45 gold and +1.82 gold + XP. The adaptive freezer guarantees +0.11 and +0.49, because B Clears. The threshold freezer guarantees −1.76 and −3.14, because B walks up. The adaptive freezer does punish conceding hard: B loses 8.1 gold and 13.6–16.5 gold + XP, against 4.2–4.7 and 2.6–5.3 with the threshold freezer. |

---

## 1. Configurations and the mathematics of the Thin action

| configuration | Hold | Thin | Clear |
|---|---|---|---|
| A baseline, `m4_lasthit()` | $\mathrm{round}(C/8)$ | $\mathrm{round}(C/2)$ | $C$ |
| B hold1, `+ wave_hold_cap=1` | $\min(1,\mathrm{round}(C/8)) = 1$ | $\mathrm{round}(C/2)$ | $C$ |
| C hold1 + adaptive, `+ wave_thin_rule="adaptive"` | 1 | $\min(\mathrm{round}(C/2),\ \text{need})$ | $C$ |

$C=\mathrm{round}(4\,AP)$ HP per turn (Reference §4.5), split over the turn's two ticks with the earlier tick rounding up.
Clear is unchanged. Power tiers are fixed for each window; "A+" means 50 resources invested (×1.76 attack power):

| tier | A (level, invested) | B | $C_A$ / $C_B$ | A's Hold / Thin / Clear, baseline | Hold under B, C |
|---|---|---|---|---|---|
| L3, L3 A+ | 3, 0 / 3, 50 | 3, 0 | 4 / 4, 8 / 4 | 1/2/4, 1/4/8 | 1 |
| L6, L6 A+ | 6, 0 / 6, 50 | 6, 0 | 5 / 5, 9 / 5 | 1/3/5, 1/5/9 | 1 |
| L10, L10 A+ | 10, 0 / 10, 50 | 10, 0 | 6 / 6, 11 / 6 | 1/3/6, 1/6/11 | 1 |
| L13 A+ | 13, 50 | 13, 0 | 13 / 7 | **2**/7/13 | 1 |
| L18 A++ | 18, 10 000 (near the attack-power supremum) | 18, 0 | 22 / 9 | **3**/11/22 | 1 |
| L18 +64 | 18, 64 | 18, 64 | 17 / 17 | **2**/9/17 | 1 |

### 1.1 Adaptive Thin (`mechanics.thin_need`)

$$
D_{\text{Thin}}=\min\bigl(\mathrm{round}(C/2),\ \text{need}\bigr),\qquad
\text{need}=\max\bigl(0,\ h_{\text{enemy}}-h^\ast\bigr),\qquad
h^\ast=\max\{h:\ f(h)\le f(h_{\text{own}})\}=\Bigl\lfloor\frac{W\,(f(h_{\text{own}})+1)-1}{2}\Bigr\rfloor
$$

Here $f(h)=\max(1,\lfloor 2h/W\rfloor)$ is the per-turn firepower of the half-wave law (Reference §4.3), $W=8$, and
$f=0$, $h^\ast=0$ when the side has no packet at all. $h_{\text{own}}$ and $h_{\text{enemy}}$ are the two **blobs at
the start of the turn**: the leading packet plus the packets queued within one cell behind it, which is the HP that
fights now (Reference §4.2).

| own blob (HP) | 1–7 | 8–11 | 12–15 | 16–19 | 20–23 | 24 |
|---|---|---|---|---|---|---|
| $f$ (HP per turn) | 1 | 2 | 3 | 4 | 5 | 6 |
| largest enemy blob kept, $h^\ast$ | 7 | 11 | 15 | 19 | 23 | 27 |

**Why this target.** Each side's front receives $R=4$ HP per turn on average (Reference §4.1). In contact, A's front
loses $f(h_B)+c_B$ and B's front loses $f(h_A)+c_A$, where $c$ is champion damage. The contact stands still only when
both fronts lose $R$. With equal champion damage that means $f(h_A)=f(h_B)$: the two blobs are in the same firepower
class. That is not equal HP. 8 against 11 is balanced; 11 against 12 is not. Thin removes the enemy HP above that
class, up to its budget. The budget is spent like any other (Reference §4.5): only on reachable minions, with overkill
lost.

**What it uses:** only the current lane state, meaning packet positions and HP. It does not predict the opponent's
action, forecast later turns or aim for a position. **There is no floor:** in a balanced lane Thin deals 0 while Hold
still deals 1. This is the literal formula, and it makes Thin the only action that does not push.

**What it ignores (the limitations):**
- Reinforcements still walking to the front. A spawn two cells behind the front joins it on the next tick, so next to
  the champion's own tower the target trims too early (§3.3).
- Both champions' damage.
- Tower fire.
- The HP the two blobs exchange during the turn.

A sensitivity variant that counts the packets joining each front during the turn is reported in §3.3. It is
harness-only and not an engine rule.

---

## 2. Scenario A: neutral last-hitting

The `neutral` start: turn 9, both 8-HP fronts and both champions at the lane cell. B Holds for 60 turns. Each cell
reads: A crashes / B crashes · turns in contact (mean contact cell, global) · A's champion damage per turn · B − A gold.

| A's action | tier | A baseline | B hold1 | C hold1 + adaptive |
|---|---|---|---|---|
| Hold | L3 … L10 A+ (6 tiers) | 0/0 · 60 (0.00) · 1.00 · 0.00 | same | same |
| Hold | L13 A+ (Hold 2) | **5/3** · 43 (+0.26) · 1.62 · +0.68 | 0/0 · 60 (0.00) · 1.00 · 0.00 | same as B |
| Hold | L18 A++ (Hold 3) | **11/0** · 35 (+0.91) · 2.02 · −1.31 | 0/0 · 60 (0.00) · 1.00 · 0.00 | same as B |
| Hold | L18 +64 (both Hold 2) | 0/0 · 60 (0.00) · 2.00 · 0.00 | 0/0 · 60 (0.00) · 1.00 · 0.00 | same as B |
| Thin | L3 | 5/3 · 43 (+0.26) · 1.62 · +0.68 | same | 4/1 · 47 (**−1.02**) · 0.77 · +1.46 |
| Thin | L3 A+, L6 … L18 (8 tiers) | 11–30/0 · 16–35 (+0.5 to +0.9) · 2.0–2.8 · −1.3 to −4.6 | same | **0/0 · 60 (0.00) · 1.47 · +0.23** |
| Clear | all | 14–30/0 · 0–30 · 2.3–3.1 · −1.5 to −5.3 | same | same |

With Hold/Hold the lane runs the 8 ↔ 12 HP cycle of Reference §4.9 (example 1): 27.00 gold and 240 XP per side in 60
turns. At equal high power (L18 +64) Hold1 only makes the blobs larger (a mean of 10.0 HP instead of 7.6) and moves the
gold by 0.23 per side. Hold1 changes the neutral lane only where the baseline Hold exceeds 1, and there it removes the
strong last-hitter's drift completely.

---

## 3. Scenarios B and D: established freeze

The `freeze` start: turn 51. B's 5-HP front is held at A's −1 with a fresh B wave behind it and 6 HP of B's crashed
under A's tower. A's 8-HP front is in A's tower cell, A stands at −2 and B at its +1. A plays 40 turns as:
- the threshold freezer: Thin while B's front is at A's −1 or deeper with ≥ 14 HP, else Hold;
- Thin every turn;
- or Clear every turn (no freeze, the reference).

B answers for the whole window by walking up (Hold, Hold), stepping back (Safe), retreating (Safe with
`safe_line = −2`) or clearing (Clear, Hold). A turn is *held* when B's front stands at A's −1 or deeper and both fronts
are alive.

### 3.1 Freeze duration

Each cell reads: longest continuous hold / first turn not held (of 40).

**Threshold freezer.** Configuration C equals B here, because the controller Thins at most 3 times.

| B's response | L3 … L10 A+, all configurations | L13 A+: baseline → Hold1 | L18 +64: baseline → Hold1 | L18 A++: baseline → Hold1 |
|---|---|---|---|---|
| walk up | 18 / 19 | **11 / 7** → 18 / 19 | **8 / 9** → 18 / 19 | **2 / 3** → 18 / 19 |
| step back | 6 / 7 | 4 / 5 → 6 / 7 | 4 / 5 → 6 / 7 | 2 / 3 → 6 / 7 |
| retreat | 6 / 7 | 4 / 5 → 6 / 7 | 4 / 5 → 6 / 7 | 2 / 3 → 6 / 7 |
| Clear | 2–4 / 2–4 | 1 / 2 → 1 / 2 | 0 / 1 → 0 / 1 | 1 / 2 → 1 / 2 |

**Thin every turn.** The baseline and Hold1 columns are identical.

| B's response | baseline: L3 / L6, L10 / A+ tiers and L18 | adaptive: L3 / L6, L10 / Thin budget ≥ 4 (A+ tiers, L18) |
|---|---|---|
| walk up | 6 / 2 / 0–2 | 19 / **25** / 12–14 |
| step back | 4 / 2 / 0–2 | 11 / 11 / 7–10 |
| retreat | 4 / 2 / 0–2 | 11 / 11 / 7–10 |
| Clear | 4 / 2 / 0–2 | 3 / 2 / 0–3 |

Champion damage over 40 turns (A on B's minions):
- threshold freezer: 31–40 HP under Hold1, 50–88 HP under the baseline at C ≥ 12;
- Thin every turn: 50–124 HP under the baseline, 10–44 HP when adaptive. The adaptive freezer deals 0 on 20–38 of its
  40 Thin turns.

### 3.2 Economics at L10 (40 turns, B − A; gold / gold + XP/8)

| A plays | configuration | B walks up | B steps back | B retreats | B Clears | A's margin against B's best |
|---|---|---|---|---|---|---|
| threshold freezer | all | +1.76 / +3.14 | −4.20 / −2.58 | −4.65 / −5.28 | +0.34 / −0.16 | −1.76 / −3.14 (B walks up) |
| Thin every turn | baseline, Hold1 | −3.38 / −6.38 | −6.17 / −9.42 | −5.29 / −8.54 | +1.57 / +1.20 | −1.57 / −1.20 (B Clears) |
| Thin every turn | adaptive | −2.44 / −4.31 | **−8.09 / −13.59** | **−8.14 / −16.51** | −0.11 / −0.49 | +0.11 / +0.49 (B Clears) |
| Clear (no freeze) | all | −2.59 / −4.71 | −4.46 / −7.21 | −4.46 / −7.21 | −0.45 / −1.82 | **+0.45 / +1.82** (B Clears) |

Across all nine tiers, A's guaranteed margin with the adaptive freezer is 0.85–7.0 below Clear's on gold + XP. On gold
alone it is ahead only at L6 (−0.15 against −0.30) and L10 A+ (+0.11 against +0.04). The threshold freezer is below
Clear at every tier. The adaptive freezer is A's best play only against a B who concedes. Against a B who contests by
clearing, Clear stays better.

### 3.3 Why a strong adaptive freezer breaks its own freeze sooner

Look at L13 A+ against a walk-up, turn 2. A's front holds 5 HP ($f=1$, so $h^\ast=7$) and B's blob 12, so Thin removes
5 HP. But A's fresh 8-HP wave is two cells behind and merges on the next tick. B's front drops to 4 HP, A's front grows
from then on, and the freeze breaks at turn 13 (trace below). A freezer with Thin budget 3 removes only 3 HP on that turn,
and its freeze lasts 25 turns. This is the latency limitation of §1.1. It reverses the power dependence: a larger
budget gives a shorter freeze.

```
L13 A+, configuration C, A Thins every turn, B walks up (selected turns)
turn -4         -3         -2         -1         +0         +1         +2         +3         +4           A/B damage
0    .          .          A8 [B6] a  B5 b       .          .          B8         .          .
1    A8         .          A5 a       B12 b      .          .          .          .          B8           T0/H1
2    .          .          A10 a      B4 b       .          .          B8         .          .            T5/H1
4    .          .          A13 a      B4 b       .          .          B8         .          .            T3/H1
8    .          .          A19 a      B5 b       .          .          B8         .          .            T0/H1
10   .          .          A22 a      B4 b       .          .          B8         .          .            T0/H1
12   .          A8         A17 a      B1 b       .          .          B8         .          .            T0/H1
13   A8         .          .          A23 a      B5 b       .          .          .          B8           T0/H0
16   .          .          A8         .          A8         A18 a      B5 b       .          .            T0/H0
```

**Sensitivity check.** The variant applies the same parity on the HP that fights during the turn: the front plus
every packet within 3 cells behind it. With it, the strong-freezer hold rises from 12 to 17 turns, and L3/L10 move to
21 (from 19/25). Step back and retreat give 10 turns at every power, where the engine estimate gave 7–11. Neutral
Thin vs Hold at L3 goes from 4/1 crashes to 1/1. The power dependence shrinks, and none of the conclusions above
change.

### 3.4 Wave HP over time

A's blob / B's blob at the end of turns 1, 5, 10, 15, 20, 30 and 40.

| L10 unless stated | blobs |
|---|---|
| threshold freezer, B walks up (all configurations) | 5/11 · 8/13 · 14/9 · 13/10 · 27/8 · 4/12 · 6/8 |
| threshold freezer, B retreats (all configurations) | 6/11 · 14/10 · 23/5 · 8/9 · 7/14 · 7/19 · 2/8 |
| adaptive Thin, B walks up | 5/12 · 7/13 · 13/8 · 12/12 · 18/7 · 7/7 · 10/14 |
| adaptive Thin, B retreats | 6/12 · 12/13 · **25/4 · 35/8** · 8/4 · 14/13 · 8/2 |
| L13 A+, threshold freezer, B walks up, baseline | 5/10 · 11/7 · **18/5** · 8/13 · 4/16 · 6/15 · 13/4 |
| L13 A+, threshold freezer, B walks up, Hold1 | 5/11 · 8/13 · 14/9 · 13/10 · 27/8 · 4/12 · 6/8 |

Crash sizes: the freezer's waves that break out reach B's tower with 13–20 HP each (A crashes 1–5 times, 20–90 HP in
total). B's crashes under Clear are 5–12 HP each.

---

## 4. The retreat case: why the friendly wave still grows

The flows on A's front (the freezer's) in the held state, per turn:

$$
\Delta h_A = \underbrace{R}_{4\ \text{HP: one 8-HP wave every 2 turns}} - \underbrace{f(h_B)}_{\text{B's minions}} - \underbrace{c_B}_{\text{B's champion}}
$$

A's own damage does not appear in this equation. When B walks up, $c_B=1$ and B's front of 12–15 HP gives $f=3$, so
$\Delta h_A=0$ and the freeze is stationary. When B steps back or retreats, $c_B=0$ and B's front of 4–13 HP gives
$f=1$–$3$. A's front then gains 1–3 HP per turn whatever A does. At 20 HP or more ($f\ge5$) A's front kills B's front
faster than B's reinforcement arrives (it has to walk from B's base at +4), and A's wave walks out. The
threshold freezer, which also last-hits for 1, loses the freeze at turn 7. The adaptive freezer, which deals 0, loses
it at turn 12:

```
L10, configuration C, A Thins every turn, B retreats (selected turns)
turn -4         -3         -2         -1         +0         +1         +2         +3         +4           A/B damage
0    .          .          A8 [B6] a  B5 b       .          .          B8         .          .
2    .          .          A11 a      B6         .          .          B8 b       .          .            T3/H0
4    .          .          A14 a      B8         .          .          B8 b       .          .            T1/H0
6    .          .          A17 a      B8         .          .          B8 b       .          .            T0/H0
8    .          .          A20 a      B7         .          .          B8 b       .          .            T0/H0
11   A8         .          A23 a      B6         .          .          b          .          B8           T0/H0
12   .          A8         A22 a      .          .          .          B8 b       .          .            T0/H0
13   A8         .          .          A8         A21 B5 a   .          b          .          B8           T0/H0
16   .          .          A8         .          .          a          [A30] b    B2         .            T0/H1
```

So in this game **a freeze is maintained by the frozen-on champion's last hits, not by the freezer**. The freezer can
slow its own wave only by dealing less damage, and Hold1 plus adaptive Thin already reach the floor of 0. That is why
the adaptive freeze lasts up to 5 turns longer against a retreat, and why no change to Thin alone can hold it much
longer.
Holding a freeze against a conceding opponent would need a rule that drains the freezer's wave without the opponent's
help. That is the freeze-stability question of the XP-range screen (§6), and it is outside this screen.

---

## 5. Scenario C: correcting an imbalance

The neutral lane with B's front set larger: A's front of 8 (or 4) HP against B's 12–24 HP at the lane cell. A Thins
(Hold position) for 12 turns and B Holds with C = 6. Each cell reads:
- A's damage in turns 1–4;
- HP removed above the need over 12 turns;
- the contact after turn 12 (lane = 0; −2 = at A's tower; none = a wave is walking or has crashed);
- A crashes / B crashes.

| start | need | Thin budget (tier) | baseline Thin | adaptive Thin |
|---|---|---|---|---|
| 8 v 12 | 1 | 2 (L3) | 2,2,2,2 · 3 · −2 · 0/1 | 1,2,2,2 · 0 · −2 · 0/1 |
| | | 3 (L10) | 3,3,3,3 · 21 · none · 1/0 | 1,3,3,3 · 0 · lane · 0/0 |
| | | 6 (L10 A+) | 6,6,6,3 · 35 · none · 4/0 | 1,5,3,1 · 0 · lane · 0/0 |
| | | 11 (L18 A++) | 11,10,1,0 · 44 · +1 · 5/0 | 1,5,3,1 · 0 · lane · 0/0 |
| 8 v 16 | 5 | 2 | 2,2,2,2 · 6 · −2 · 0/2 | 2,2,2,2 · 0 · none · 0/3 |
| | | 3 | 3,3,3,3 · 0 · lane · 0/0 | 3,3,3,3 · 0 · lane · 0/0 |
| | | 6 | 6,6,6,5 · 32 · none · 3/0 | 5,5,3,1 · 0 · lane · 0/0 |
| | | 11 | 11,8,6,0 · 39 · none · 5/0 | 5,5,3,1 · 0 · lane · 0/0 |
| 8 v 24 | 13 | 2 | 2,2,2,1 · 9 · −2 · 0/3 | 2,2,2,1 · 0 · none · 0/4 |
| | | 3 | 3,3,3,1 · 15 · −1 · 0/2 | 3,3,3,1 · 0 · −2 · 0/3 |
| | | 6 | 6,6,6,0 · 21 · +1 · 1/1 | 6,6,6,0 · 0 · −2 · 0/1 |
| | | 11 | 11,11,11,0 · 36 · none · 5/0 | **11,7,3,1 · 0 · lane · 0/0** |

The 4 v 20 start gives the same pattern (`results/m43c/wave_control.md`). For comparison, under configuration C a Hold
(1 HP) lets every one of these stacks through: B crashes 1–4 times. Clear pushes A's wave into B's tower (A crashes
1–6 times), except with budget 2 against stacks of 20 HP or more.

- **A weak champion corrects only part.** With a Thin budget of 2, even an excess of 1 HP is not corrected, because
  the need grows as B's reinforcements merge (1 → 4 → 5 → 4) faster than 2 HP per turn removes it. Baseline Thin
  fails in the same way.
- **A strong champion corrects without overshoot.** It removes 13 HP in four turns and never more than the need.
- **Thin stops at the target.** Once the blobs are at parity it falls to 1–3 HP per turn, which is B's last-hit edge
  plus the merges.

8 v 24 at the lane. Baseline Thin, budget 11: B's stack is gone in two turns, and A's wave walks into B's tower.

```
turn -4         -3         -2         -1         +0         +1         +2         +3         +4           A/B damage
0    .          .          A8         .          A8 B24 a b .          B8         .          .
1    A8         .          .          .          A8 B18 a b .          .          .          B8           T11/H1
2    .          .          A8         .          A4 B6 a b  .          B8         .          .            T11/H1
3    A8         .          .          .          A10 a      .          b          .          B8           T11/H0
4    .          .          A8         .          a          .          B7 [A8] b  .          .            T0/H1
5    A8         .          .          .          A8 a       .          B6 b       .          B8           T0/H1
6    .          .          A8         .          a          .          B8 [A5] b  .          .            T5/H1
```

Adaptive Thin, budget 11: 11, 7, 3, 1 HP, then a stable 3 / 1 cycle with the contact at the lane.

```
turn -4         -3         -2         -1         +0         +1         +2         +3         +4           A/B damage
0    .          .          A8         .          A8 B24 a b .          B8         .          .
1    A8         .          .          .          A8 B18 a b .          .          .          B8           T11/H1
2    .          .          A8         .          A4 B10 a b .          B8         .          .            T7/H1
3    A8         .          .          .          A8 B12 a b .          .          .          B8           T3/H1
4    .          .          A8         .          A4 B10 a b .          B8         .          .            T1/H1
5    A8         .          .          .          A8 B12 a b .          .          .          B8           T3/H1
6    .          .          A8         .          A4 B10 a b .          B8         .          .            T1/H1
```

Adaptive Thin, budget 2 (level 3): a partial correction. B's stack reaches A's tower at turn 6.

```
turn -4         -3         -2         -1         +0         +1         +2         +3         +4           A/B damage
0    .          .          A8         .          A8 B24 a b .          B8         .          .
1    A8         .          .          .          A8 B19 a b B8         .          .          B8           T2/H1
2    .          .          A8         .          A1 B16 a b B8         B8         .          .            T2/H1
3    A8         .          .          .          A1 B21 a b B8         .          .          B8           T2/H1
4    .          .          A5 a       B18 b      B8         .          B8         .          .            T1/H0
5    A8         .          a          B23 b      B8         .          .          .          B8           T2/H1
6    .          A2         [B25] a    b          .          .          B8         .          .            T2/H0
7    A8         .          [B17] a    .          B8 b       .          .          .          B8           T1/H0
8    .          A4         [B18] a    .          b          .          B8         .          .            T0/H0
```

---

## 6. Recovery after conceding

B concedes (steps back or retreats) for 20 turns and then walks up for 40. Each cell gives the conceding line minus
the walk-up-throughout line, B − A gold and XP, at turn 20 → turn 60.

| tier | freezer (configuration) | step back: gold, XP | retreat: gold, XP |
|---|---|---|---|
| L3 | adaptive Thin (C) | −6.21 → −2.16, −39 → −14 | −6.26 → −2.21, −62 → −37 |
| L10 | adaptive Thin (C) | −3.96 → −0.66, −16 → +3 | −4.01 → −0.71, −39 → −20 |
| L10 A+ | adaptive Thin (C) | −2.72 → −2.73, +7 → +2 | −2.77 → −2.78, −3 → −8 |
| L10 | threshold (all) | +0.39 → −0.21, +18 → +18 | +0.34 → −0.26, +8 → +8 |

Against the adaptive freezer, B recovers the gold it conceded at levels 3 and 10. The freezer's piled-up wave
eventually crashes into B's tower, where B farms it. At ×1.76 the gold is not recovered. XP missed while out of range
during a retreat is never recovered, as already shown in the XP-range screen (§3). Against the threshold freezer,
conceding costs nothing by turn 20 because the freeze collapses toward B within 7 turns. These are single deterministic
trajectories: the 40 turns after the concession can diverge by ±2–6 resources depending on the phase of the crash
cycle (the L3 and L10 A+ threshold rows in `results/m43c/wave_control.md` swing by that much). Only effects larger than
this are read here.

---

## 7. Learned self-play under real progression (fixed policy)

The M4.3b final `base s0` plays itself from the initial state with its own levels and investment: 16 seeds × 160
turns, per side, mean ± 95% half-width. It was trained under configuration A, so this is a screen of a fixed policy,
not an equilibrium.

| quantity | A baseline | B hold1 | C hold1 + adaptive |
|---|---|---|---|
| gold | 60.48 ± 0.80 | 60.47 ± 0.80 | 60.20 ± 0.70 |
| XP by turn 100 | 383.6 ± 2.7 | 383.6 ± 2.7 | 383.8 ± 2.5 |
| turn of level 16 | 113.7 ± 0.6 | 113.7 ± 0.6 | 113.4 ± 0.7 |
| crashes | 17.1 ± 1.6 | 17.1 ± 1.6 | 14.4 ± 1.4 |
| Hold / Thin / Clear turns | 4.8 / 10.9 / 129.5 | same | 3.7 / 9.2 / 132.1 |
| HP removed on Hold turns (Hold ≥ 2 under the baseline on 3.7 of them) | 0.88 | 0.44 | 0.34 |
| HP removed on Thin turns | 18.9 | 18.9 | **0.00** |
| turns with the contact off the lane cell | 8.9 ± 3.7 | 8.9 ± 3.7 | 15.1 ± 3.9 |

Hold1 changes nothing measurable, because the policy Clears on 81% of turns and rarely Holds. Under adaptive Thin, the
policy's Thin turns, which it used as a push, deal nothing at all.

---

## 8. Unintended consequences and recommendation

**Unintended consequences**
1. **Thin loses its slow-push role** under the adaptive rule. Thin vs Hold is stationary from level 6 on, and only
   Clear can build an edge. With one fewer intermediate option this may strengthen Clear dominance; not tested.
2. **At low capacity, adaptive Thin holds the lane worse than Hold.** At L3 (budget 2), Thin vs Hold holds the lane for
   15 turns. Then B's wave crashes into A's tower and the contact stays there for 25 turns, where Hold vs Hold never
   leaves the lane cell. Thin deals 0 at parity, then cannot catch up once B's 1-HP-per-turn edge crosses a firepower
   class.
3. **The power dependence is reversed at the freeze line** (§3.3). A larger Thin budget over-corrects, so the freeze
   lasts 12 turns instead of 25. The turn-window target reduces this to 17 against 21.
4. **Conceding to an adaptive freezer is punished hard**: B loses 13.6–16.5 resources (gold + XP) over 40 turns at
   L10, against 2.6–5.3 with the threshold freezer. This comes from the longer hold, not from a new denial rule, and
   B's Clear still neutralizes it (−0.49).
5. **Learned Thin becomes a no-op** (18.9 → 0 HP per game). A policy trained under configuration C would have to
   learn a new meaning for that action.
6. Hold1 at equal high power only makes the blobs larger (10.0 against 7.6 HP) and moves gold by 0.23 per 60 turns.

**Recommendation: separately, not together.**
- **Adopt Hold1** as the next candidate change of the research preset (not done here). It removes the automatic
  last-hit push of strong champions (§2). It makes a freeze's length independent of the freezer's power (§3.1), keeps
  the counterplay, and changes nothing below C = 12 or in learned play. It must be pinned in the mathematical
  reference when it is adopted.
- **Do not adopt Adaptive Thin yet.** As a controller it does what was asked (§5), but:
  1. it cannot fix the retreat case, which is a minion-flow property (§4);
  2. a design decision is needed on whether Thin should still slow-push (consequence 1);
  3. its target should at least count the reinforcements that arrive during the turn (§3.3).

  Revisit it after the freeze-stability question (§4) is decided, and pair it with Hold1 if it is adopted. Without
  Hold1, a high-power Hold (2–3 HP) would push harder than an adaptive Thin; that combination was not tested.

Freezing was not made more profitable by tuning. The target is the plain parity of the Lanchester law, and Clear
remains A's best play against a best-responding B on gold + XP at every tier (§3.2).

---

## 9. Reproduction, files, limitations

```bash
python -m lanegame.experiments.m43c wave_control --workers 3     # 10 s; writes results/m43c/wave_control.{json.gz,md}
python -m pytest tests/test_m43c_wave_control.py                 # 19 tests
```

| item | content |
|---|---|
| `lanegame/config.py` | `wave_hold_cap` (0 = off) and `wave_thin_rule` (`"share"` default, `"adaptive"`): opt-in, M4-only, `"capacity"` scaling only; adaptive needs the half-wave law |
| `lanegame/champions_m4.py` | `budget_from_capacity`: the Hold cap |
| `lanegame/mechanics.py` | `thin_need`; `step_turn` caps a Thin side's budget at it under `"adaptive"` |
| `lanegame/experiments/m43c/wave_control.py` | the screen: fixed-power driver, stages neutral / freeze / recovery / imbalance / sensitivity / games, traces, tables |
| `results/m43c/wave_control.json.gz` | every run (window statistics and per-turn series of blobs, stocks, fronts, budgets, need, damage), traces, games |
| `results/m43c/wave_control.md` | all tables, including every tier, the no-freeze reference, the 4 v 20 start, the full recovery table and all traces |
| `results/m43c/wave_control.log` | run log |
| `tests/test_m43c_wave_control.py` | defaults and budgets, validation, worked examples of `thin_need`, the adaptive cap in `step_turn`, HP conservation and mirror symmetry under the adaptive rule, pins of §2, §3, §5 |

**Preservation.** `m4_lasthit()` and every other preset keep their defaults (`wave_hold_cap = 0`,
`wave_thin_rule = "share"`). The behavior digests of the historical presets and of `m4_lane()`/`m4_lasthit()`
(`test_m4_wave_actions.py`, `test_m4_zoning_natural.py`, `test_legacy_pins.py`) pass unchanged. The mathematical
reference lists the two fields as EXPERIMENTAL in §4.5 and in its §13 log.

**Limitations.**
- Deterministic scripted play with no chance events: no trades, no HP loss, no recalls. Positions follow the scripted
  behaviors.
- Single trajectories, so effects below the ±2–6 resource phase noise of §6 are not read.
- Attack power is held fixed over each window, and there is a single freeze start (turn 51).
- B's responses are fixed for the whole window, and "B's best" means the best of four scripted responses.
- The learned policy was trained under configuration A.
- The adaptive target ignores walking reinforcements, champion damage and tower fire (§1.1).
