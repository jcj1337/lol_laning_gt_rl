# M4 Design Audit

*The critical review that accompanies [m4_integrated_design.md](m4_integrated_design.md) ("design §n") and
[m4_implementation_plan.md](m4_implementation_plan.md) ("plan §n"), 2026-10-11. It lists what is weak, uncertain
or possibly wrong in the proposal. Nothing here is hidden from the specification; this document collects it in one
place. "Measured" means the scripts in `results/m4_design/scripts/`, run on the unchanged code: scripted agents, exhaustive lane enumeration,
analytical duel models, no training.*

---

## 1. Contradictions and tensions in the proposed mechanics

### 1.1 "Freeze primarily denies gold" is only partly a property of the Freeze action

**Finding.** The audit found that a parked configuration, with the enemy front at the defender's −1 and the
defender's waves dying inside its own tower cell, is **stable under mutual Slow Push for 3–5 HP fronts**. That
holds without any Freeze (design §1.1, §6.4).

**What follows.**

- The gold displacement (the opponent must farm next to the defender's tower at efficiency 0.5) is a property of
  where the waves are. It does not depend on the wave action.
- Freeze's distinct effect is **crash denial**: it stops 5–9 HP creeping fronts and small shoved fronts from
  entering the tower.
- Through crash denial Freeze *prolongs* the gold displacement and *denies persistent tower damage* (§11).

**The tension with the brief.**

- The brief describes Freeze as primarily a gold-denial tool.
- In this model Freeze is primarily a **crash- and tower-denial tool that keeps a gold-denying configuration
  alive**.
- Making Freeze itself cause the gold displacement would need a new rule that pulls the contact point toward the
  freezer, e.g. the freezer's waves stop one cell earlier. That would duplicate what the lane already does.

**Recommendation.** Accept the reformulation. Validate with the plan's test 1 (strictly later crashes under
Freeze) and S3 (the fraction of Freeze-legal situations that differ, now with maintenance).

### 1.2 XP range 3 makes XP nearly "presence-based"

**What was measured.** With every kill at the lane cell in passive play, XP ranges 1, 2 and 3 are identical
(1.88 HP per turn). XP denial needs the hard geometry: the opponent pushed back to its own tower cell, with kills
inside the far tower cell.

**Consequences.**

- Level differences will come mostly from **absence** (deaths: 3 turns ≈ 11 XP; recalls: 2 turns ≈ 7.5 XP). Zoning
  will contribute little.
- That is consistent with the brief: XP denial is conditional and requires combat advantage. It also means the
  "lose gold, keep XP" pattern will be common, and "lose both" rare.
- If S4 finds XP denial never occurs, the alternative is R_xp = 2 plus an XP-keeping behavior (design §16 A3).
  That adds or changes an action.

### 1.3 Power acts on duels in steps (breakpoints)

**The cause.** A kill takes 3–6 hits.

**Measured from full HP:**

- +1, +2 and +3 levels give the *same* duel probabilities (0.583 / 0.092 / 0.325);
- +1 item tier at level 6 changes the duel edge by only +0.01, because 766 / 153 = 5.002 hits rounds up to 6.

**Averaged over mid-game HP (40–100% of max) the effect is monotone:**

| Lead | +1 level | +2 levels | +3 levels | +1 tier | +2 tiers |
|---|---|---|---|---|---|
| Duel edge | +0.12 | +0.18 | +0.29 | +0.19 | +0.34 |

**Risks.** Level-ups and purchases have *state-dependent*, discontinuous values. PPO may learn them poorly. The
first tier may look worthless in some states and decisive in others.

**Fix if needed** (design §16 A8). Two half-strength rolls per trade: same mean, half the variance, twice the hits
to kill. The edge becomes smoother: +0.18, +0.51, +0.51 from full HP, and +0.14, +0.25, +0.36 averaged. It is not
adopted by default because it changes M2's trade variance.

### 1.4 Equal-level scaling cancels

AP and max HP both grow by 1.05 per level. At equal levels and items every damage-to-HP ratio equals M2's (design
§8.3). Levels therefore matter **only through differences**: in mirror self-play the absolute level is irrelevant
to combat. This is intended. It means the ruleset's novelty lies entirely in asymmetries (absence, zoning, timing of
purchases), which self-play produces only if one side errs or takes risks.

### 1.5 Max-HP growth does not protect against ganks or towers

Gank and tower damage scale with the *victim's* max HP (design §8.3), so leveling protects only against trades. The
reason is that ganks would otherwise become steadily less lethal over a continuing game, at a rate unrelated to the
opponent. It is a modelling choice, not an emergent property, and it weakens the defensive value of levels. The
alternative (scale with the attacker's level) is design §16 A7.

### 1.6 "Recall allows meaningful spending decisions" vs automatic purchase

There is no purchase choice. The tiers are sequential and gold has no other use, so buying when at base is weakly
dominant (design §8.5). The only spending decision is *when to go to base*. If reviewers want a visible shop
decision, it must add an option with a real trade-off: a second track, or a consumable. That contradicts the brief's
"extremely simple items".

### 1.7 The continuing game has no laning-phase end

**What League has and M4 does not.** In League, laning ends around 14 minutes, when plates fall and towers start to
go. M4 has no end of laning:

- the asymptotic game is the capped lane (design §10.6);
- plates are finite;
- towers never fall.

That is what the brief asked for. It does mean the late game (turns ≳ 200) is a stationary M2-like lane with bigger
numbers, and it holds about 13% of the discounted weight at γ = 0.99.

**Two caveats.**

- The objective weights the first ~150 turns most. That makes γ a decision with strategic content. It is design
  decision D1, and its sensitivity must be reported.
- The training start state s₀ is always level 1, so the late game is learned only by playing through to it.

### 1.8 Freeze maintenance rarely matters for the most common parked states

For 3–4 HP parked fronts nothing in the lane distinguishes Freeze from Slow Push, maintained or not. Maintenance
matters exactly where Freeze matters: medium creeping fronts and small shoved fronts. It is therefore a correct but
narrow rule. Ablation is planned (plan §7 S6).

---

## 2. Degenerate incentives and possible exploits

### 2.1 A passive equilibrium may persist

**Measured.** Mutual Slow Push is a crash-free 2-cycle at the lane cell. A plain Shove against Slow Push never
crashes either: the shoved wave dies in the defender's tower cell. Crashes need deliberate asymmetry (stacking,
timing against the enemy's absence).

**What M4 adds and does not add.** M4 adds reasons to crash: persistent tower damage, a continuing game, no horizon
to waste. It adds no new way to crash.

**The risk.** If M2's crash rarity carries over, tower durability, plates, bounces and Freeze could all stay
unvisited, as Freeze and sieges were in M2 (diagnostics D.4: rare states abandoned within 50k transitions).

**This is the largest risk of the whole design.** Mitigation: the start-state archive and diverse resets (plan WP1,
§6), and the counterfactual checks that Pressure, Freeze and Recall-to-buy have value where reached (plan S4).

### 2.2 Death as a shopping trip

Deliberate death is dominated by Recall whenever Recall is legal and uninterrupted (design §8.5). The margin
depends on `kill_bounty`:

- at 0.5 a death costs ≈ 0.5 more than walk-back + Recall;
- at 0 the two are close to indifferent when the walk-back turn is dangerous.

**Do not set `kill_bounty = 0`** in M4 without re-checking this.

### 2.3 Recall-to-buy could become dominant, or never used

A +20% damage tier is worth about a two-level lead in duels (§1.3). With purchases the value of Recall rises.
Neither outcome is acceptable as a design default:

- if the gain exceeds two turns of lane presence in most states, "Recall whenever affordable" dominates;
- if not, Recall stays unused, and items arrive only at deaths.

S4 must find positive and negative states; otherwise the design is revisited (prices or tiers, design D4) *before*
training. **Do not tune prices to equalize Recall frequency.** Tune them only if the sign pattern is degenerate,
and report why.

### 2.4 Crash-then-recall

Minion tower damage and plate gold accrue while the attacker is away (design §11.3). So "crash a stacked wave, then
Recall" earns plates without exposure. This is League-like and not degenerate in itself, but it is strong:

- a 12-HP crash alone breaks the first plate;
- the defender cannot stop it once the crash has happened, except by thinning the bounce.

The alternative that requires presence for plate gold is design §16 A9. Measure its share of plate gold in S4.

### 2.5 The tower-adjacent shot may make all forward trading unprofitable

p_adj = 0.5 penalizes any trade from the cell next to the enemy tower. It is intended as the freezer's home
advantage, but it also taxes:

- legitimate forward pressure;
- the shover breaking a freeze, if it trades.

It could make the defender's side of the lane favorable for every engagement and strengthen §2.1. Its ablation is
part of S6 (design D3).

### 2.6 Training-time artifacts

- **Truncation** at `game_length` is unobserved and bootstrapped, so it cannot be exploited by the policy.
- **A miscalibrated critic** at long horizons can still bias the policy near the start (γ = 0.99, returns of
  ±10–30). The smoke stage checks value scale and explained variance.
- **Lock-step games** would correlate batches; random first-game lengths are required (plan WP1).

### 2.7 Things checked and found not to be exploits

- **Automatic purchase timing.** It happens only on the last turn away, so it cannot be duplicated, triggered while
  present or reversed.
- **The gold clamp.** It removes only gold that can never be spent.
- **XP for an interrupted (channeling) Recall.** The champion is present and exposed, and earns no gold.
- **Pressure after all plates.** It still thins; τ clamps; nothing is paid twice.
- **Leveling to escape death.** Death is resolved first (design §5).

---

## 3. Mechanics that may be redundant or observationally indistinguishable

| Pair | When indistinguishable | Consequence |
|---|---|---|
| Freeze vs Slow Push | in contact (97% of Freeze-legal contact situations) and in 3–4 HP parked states | intended; the mask still offers both. PPO sees identical outcomes and splits arbitrarily, which inflates entropy but is harmless |
| Yield vs Hold | unthreatened (M2) | unchanged from M2; diagnostics found Yield unused |
| Safe at −1 vs threatened Yield | same end cell when a trader threatens | the differences are only gold efficiency and gank damage |
| XP range 1, 2, 3 | passive play (all kills at the lane cell) | XP is effectively presence; see §1.2 |
| Level vs item tier | both raise damage; only levels raise HP | distinguishable in duels: an item ≈ 3.7 levels of damage, no HP |
| Recall vs death as a way to shop | when the walk-back is costly and `kill_bounty` is small | see §2.2 |
| Vision | unchanged from M2 | not touched by M4 |

---

## 4. Risks of irreversible snowballing

**The loop.** An economic lead wins more trades, which means more opponent deaths (3 turns away each, with no XP
and no gold). During those turns the leader farms ≈ 0.41 per turn instead of 0.15 (M2 measurement), so it buys
items sooner and its lead grows.

**Built-in brakes:**

- **Rising XP per level** (+3 per level). A leader needs more XP per level, so level gaps tend to close at equal
  absorption.
- **Caps.** L = 18 and N = 2 bound the gap. Two capped champions are equal again in economy.
- **Finite plates.** Siege income is bounded.
- **Gank and tower damage scaled by the victim.** A behind champion is not *more* vulnerable to ganks than a leader
  is.
- **Skill unaffected.** A mechanically stronger player keeps its hit advantage.

**Missing brakes** (deliberately, per the brief): no comeback XP, no shutdown bounty, no defensive item, no death
timer scaling.

**Irreversibility within the discounted horizon.** Two early deaths cost ≈ 22 XP (≈ 1 level at levels 4–6) and
≈ 1–2 resources (the first tier delayed by ~5–8 turns). That gap shrinks only as the leader approaches the caps,
≈ 150 turns, by which time γ^150 ≈ 0.22.

**How to measure it** (plan S2/S4 and later research):

- P(final discounted margin > 0 | margin after 30 turns) under scripted and learned play;
- the distribution of level gaps over time.

If early leads decide > 90% of games in learned play, consider design §16 A5 (a defensive tier) or A7 (attacker-level
ganks), as measured ablations.

---

## 5. Risks of excessive state- or action-space growth

**Actions.** No growth: still 46 codes, 4–9 legal.

**State.** Per side, M4 adds xp (545 values), items (3), gold (continuous, bounded by 10), HP in centi units (≤ 1,376
values instead of 7) and tower damage (≤ 6,001 values).

- Exact enumeration was already impossible in champion mode, so nothing is lost there.
- For learning, the critic must generalize across levels and items. Normalized features (design §12) make level an
  ordinal input, not a one-hot.

**Observation.** 50 → 66 entries. Negligible for the MLP.

**Episodes.** Games of 512 turns and evaluation windows of 460–1,000 turns instead of 32.

- Sample cost per *game* is ~15–30× larger.
- Sample cost per *transition* is unchanged.
- The real risk is coverage: late-game states are visited only after ~150 turns of play.

**Throughput.** The M2 champion environment runs ~6k turns per second uncached. M4 adds a few integer table
look-ups per turn; the target is ≥ 4k turns per second (plan §6).

**Evaluation cost** is the binding compute constraint (plan §6). A 2,000-game head-to-head at W = 1,000 is ~2M turns,
≈ 8 min single-threaded.

---

## 6. Parameters that cannot be justified without experiments

| Parameter | Default | Why it needs experiments | Validated by |
|---|---|---|---|
| γ | 0.99 | changes the game's equilibria; trades early vs late | sensitivity 0.98 / 0.995 (research stage) |
| `xp_per_hp` | 2 | anchored to League per wave; the turn timing depends on learned absorption | S2 (±25% band) |
| HP / AP growth per level | 1.05 / 1.05 | brief (provisional); the size of level effects | S4 counterfactuals, duel tables |
| `item_mult`, `item_prices` | 1.20; (4, 6) | brief (provisional); the Recall sign pattern | S4 (§2.3) |
| `plate_thresholds`, `pressure_tower_damage` | (16, 36, 60) units; 4 per turn | plate pace; the siege incentive | S3 / S4 |
| `tower_adjacent_p` | 0.5 / 0.25 | home advantage vs passivity | ablation in S6 |
| `xp_range` | 3 | whether XP denial ever occurs | S4; A3 if never |
| `kill_bounty` | 0.5 | death-shopping margin, value of kills | S4 |
| `game_length`, `segment_length` | 512, 128 | training only (no effect on the objective); value-learning stability | S5 smoke |

**Parameters that can be justified analytically:**

- the XP threshold *shape*;
- the integer tables;
- the purchase rule;
- the ledger;
- the continuing-time semantics;
- the contact-cell chase rule;
- Freeze maintenance.

---

## 7. Recommended alternatives and actions, ranked by significance

1. **Make rare states trainable from the start.** Reachability, not mechanics, decided the fate of Freeze and
   sieges in M2 (§2.1). Implement the start-state archive and diverse resets as configurable options of WP1, and
   report every M4 result with and without them. This matters more than any parameter.
2. **Keep the continuing discounted game. Do not adopt a terminal laning event** (design §16 A1). A terminal
   valuation reintroduces end-of-episode incentives of the kind M4 removes. If γ = 0.99 proves hard to train, lower
   γ to 0.98 (a stated change of objective) before considering a terminal state.
3. **Reformulate Freeze's purpose in the documentation** as crash and tower denial plus prolonging the parked
   configuration (§1.1), and test it that way. Do not add a rule that makes Freeze itself move the contact point
   unless S3 shows Freeze is still inert under M4 incentives.
4. **Adopt two half-strength rolls per trade** (design A8) if S4 shows level or tier values dominated by
   breakpoints (§1.3).
5. **Measure the tower-adjacent shot by ablation** before treating it as settled (§2.5).
6. **Keep the legacy siege-wait rule (D7) for now**, but measure its effect on bounce timing. Fixing it to
   "wait at −3 only" is a candidate if bounces prove too slow to make Freeze reachable.
7. **XP range 2 with an XP-keeping behavior** (design A3) only if XP denial never occurs in S4.
8. **A defensive tier or attacker-level ganks** only if snowballing exceeds the §4 threshold.
9. **Plate gold for present champions only** (design A9) if crash-then-recall dominates plate income.

## 8. Lessons carried over from the diagnostics

- **Exploitability.** A weak best-response attacker underestimates exploitable behavior. In M4 use ≥ 2 warm-started
  attackers and 1 from scratch per target, ≥ 1M steps each. Report the budget, the attacker count and the learning
  curves. The maximum over attackers is a lower bound, not a certified Nash exploitability.
- **One-step counterfactuals are not enough.** A one-turn deviation misses multi-turn plans (sieges, freezes,
  recall timing). M4 counterfactuals must include short plans.
- **Comparing rulesets.** Never compare M2 and M4 numbers as the same game. The M2 findings motivate M4; they are
  not results about it.
