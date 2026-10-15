# M4.6a — Pre-registration of the diagnosis stage

*Written 2026-10-15, before any M4.6a experiment was run. Implements the diagnosis stage proposed in
[m4_6_design_review.md](m4_6_design_review.md) §5, approved by the user with five priorities (E1, E2, E3, E5 before
interpreting E4; E4 with controls; absolute advantage kept apart from improvement over the trained policy;
reproducibility, paired comparisons and the short-horizon methodology; unexpected results explained, not forced).
Environment `GameConfig.m45_calibrated()`, unchanged. No mechanics change, no M4.6b lever, no general self-play
training. The machine-readable copy is `results/m46a/preregistration.json` (stage `prereg`, with the SHA-256 of this
file); every experiment stage refuses to run without it.*

## 1. Policies, seeds, statistics

- **Evaluated policies**: the six 16.384M finals `final:ext:0..5` (the M4.5c primary set).
- **Controls already trained**: the six M4.5c best-response attackers (`results/rl/checkpoints/m45c/attack/ext_s*`,
  4,096,000 learner transitions from s0 against each final).
- **Seeds**: the new range 96,000–96,999, disjoint from every earlier study (10,000–81,999, 85,000–88,499,
  90,000–94,999).

| use | root |
|---|---|
| E3 gap self-play games (the training split: games 0–31 of a cell; held-out split: games 32–47) | 96,000 |
| E2 natural state collection | 96,100 |
| E2 / E5 paired rollouts (rollout r uses seed root × 100 + r) | 96,200 |
| E4 attacker training (seed + target) | 96,300 |
| E4 evaluation (rollout r of state i: root × 1,000 + 100 i + r) | 96,400 |
| E5 natural state collection | 96,500 |
| validation-subset selection (fixed before results) | 96,700 |

- **Pairing**: every comparison is a paired, common-random-number difference from the same start state: the same
  chance stream and the same per-turn sampling uniforms for both seats (M4.5c `horizon.CRNNet`), differing only by the
  intervention.
- **Statistic**: G_h (observed discounted gold lead of the focal seat A) for constructed and semi-natural states, Q_h =
  G_h + 0.99^h V(s_h) (the evaluated policy's critic) for natural states, as in the M4.5c battery (exception 2). The
  other statistic is always reported.
- **Horizons** (horizon study rule): plans of ≤ 10 turns 50; longer plans plan length + 40 (20-turn plans: 60).
  E4 policy replacements are whole-policy comparisons: G100 primary for the gap attackers, G50 for the freeze attackers
  (freezes resolve by 20 turns), every other horizon reported.
- **Validation**: a fixed-seed max(4, 15%) of each natural and semi-natural family, and 6 constructed cells, run to 300
  turns under protocol "policy", checked **pooled over the six policies per family** (the M4.5c recommendation): bias
  |mean(Δ300 − Δh)| ≤ max(0.15, 2 SE), slope of Δ300 on Δh in [0.8, 1.25], no contradiction (a pair significant at
  both horizons with opposite signs). A family that fails is re-run at 150 turns.
- **Intervals**: per policy, the mean over its states of the per-state paired mean; across the six policies a t
  interval (95%, df 5). "k/6" counts policies whose own interval (states as units) excludes 0. E4 per target:
  intervals with evaluation states as clusters; pooled over targets: the mean, a t interval (df 2) and the count of
  targets significant.
- **Absolute versus improvement** (the user's priority 3): every table states the stronger side's absolute lead under
  the trained policy beside the improvement of an alternative over that policy. A "conversion gain" is always an
  improvement over the policy from the same state, never the absolute lead.

## 2. Experiments and primary comparisons

### E1 — exact one-turn ledger of contesting (no learning, no rollouts)

States: the M4.5b freeze builder (A holds B's wave at A's −1, B walked up to its +1) at levels 4 / 10 / 15 × equal
power / A ×1.76 × held wave 5 / 12 / 20 HP (18 cells); the M4.5b zoning situations (neutral, A's push, B's push) at
levels 6 / 10 × equal / ×1.76 (12 cells). Every pair of the listed actions, exact expectation over the turn's chance
branches with ganks: earned gold, farm, XP, HP lost and dealt (bars), minion fire split into retaliation and positional
aggro, death probability, P(A's front dies), P(the freeze survives the turn).

- Freezer A: keep (Thin + Hold), guard (Thin + Trade), Last Hit + Hold, Last Hit + Trade, break (Shove + Hold).
- Frozen-on B: break (Shove + Hold), walk up (Last Hit + Hold), step back (Last Hit + Safe), Shove + Safe, Yield
  (Last Hit + Yield), contest (Shove + Trade).
- Zoning: A ∈ {Last Hit, Shove} × {Hold, Trade}; B ∈ {Last Hit + Hold, Yield, Safe, Trade, Shove + Hold}.

**Decomposition (counterfactual one-turn rows, labelled; not candidates, never trained or rolled out)**: the guard and
the threat re-evaluated with one trading cost removed at a time through existing configuration fields:
`trade_farm_factor` 1.0, `trade_wave_fraction` 1.0, `minion_retaliation` off, `minion_positional_aggro` off, the
Trade gank term equal to Hold's. `trade_exchanges` (lever L1) is **not** evaluated (M4.6b).

**E1-P1** (descriptive): P(the freeze survives B's walk-up break) per cell, and B's expected HP-priced cost of the
break against the guard, at prices of 1, 1.5 and 2 resources per bar and at E5's measured price.
**E1-P2**: a cost is called *load-bearing* if removing it alone changes the sign of the guard's (vs the break) or the
threat's (Trade − Hold vs a Hold defender) HP-priced one-turn value in at least half of the cells, at a price of
1.5 resources per bar (E5's measured price as a sensitivity).

### E3 — power-gap self-play (absolute advantage and its channels)

Each final in self-play from the M4.5b zoning situation "neutral" at levels 6 / 10 / 14 with A's attack-power lead
equal / ×1.24 / ×1.5 / ×1.76 / ×2.2 (investment only), both orientations (A stronger, and the mirror with B stronger,
signs folded to "stronger minus weaker"), 48 games × 300 turns per cell and orientation (games 0–31 the training split
used by E4's archives, 32–47 held out); plus the M4.5c start (viewer neutral, level 3, ×1.76) for continuity.

**E3-P1**: the power premium, (stronger − weaker) lead at a gap minus the same at equal power, at G20 / G50 / G100 /
G300 and undiscounted at 100 / 300.
**E3-P2**: the decomposition of the premium into (a) the stronger side's own farm gain, (b) the weaker side's lost
minion flow (fewer enemy minions dying on its ledger), (c) posture denial (farm lost out of reach while present),
(d) absence (farm lost while away), (e) bounty and plates. The design review's H4 ("the stronger side's lead comes
from absence, not farm") predicts (d) + (e) > (a) + (c).
**E3-P3**: the M4.5c readouts (weaker side's Yield share and time at its tower, stronger side's Trade rate, position,
"denial when near") reproduced, to check that E3 measures the same thing M4.5c did.

### E5 — the price of a reset and of HP

Natural decision states from each final's self-play (48 per family per final, K = 12 paired rollouts, Q50 primary):

- **E5-full**: Recall legal, own HP ≥ 90% and at least 2 resources unspent (a Recall right after a return, with
  nothing to invest, is degenerate); **E5-mid**: Recall legal, HP 40–60%. Candidates: the policy, 1t Recall, 1t
  stay (Last Hit + Hold), 1t Shove + Hold.
- **E5-shop**: the same states with the focal side's unspent gold moved into its investment before the turn (a state
  variant, no rule change); Recall − stay in the original minus in the variant = the part of a Recall's relative value
  that comes from investing on return.
- **E5-hp**: natural states with both present and focal HP ≥ 70% (early game, levels ≤ 10, and later): the state as is
  against the same state with the focal HP lowered by 0.3 bar, both continued by the policy: the shadow price of one
  bar under learned play = ΔQ50 / 0.3.

**E5-P1**: the reset is called *cheap* if Recall − stay at full HP is above −0.5 (ΔQ50), *nearly free* above −0.2.
**E5-P2**: the shopping share. **E5-P3**: the price of HP per bar, early and late.

### E2 — selective threats on the current model (scripted plans on the trained policy)

Plans for the focal stronger side A, 20 turns, then the policy; horizon 60. "beh" plans force only the behaviour and
leave the wave and objective to the policy's own conditional choice (the policy's distribution restricted to the
slice and renormalized, the same sampling uniform as the baseline). Every plan hands the turn to the policy while A
is at or below 40% HP (so no plan is penalized for being unable to Recall); Ward and Recall are never forced.

| plan | behaviour | wave |
|---|---|---|
| policy | the policy | the policy |
| farm_beh | Hold | the policy |
| threaten_beh | Trade if the enemy is catchable without a dive and the duel edge is positive (the M4.3c / M4.4c "hunt" rule), else Hold | the policy |
| edge_beh | Trade if catchable without a dive, own HP ≥ 60% and duel edge ≥ 0.05, else Hold | the policy |
| bully_beh | Trade whenever catchable without a dive, else Hold | the policy |
| freeze_threat | the hunt rule | the freeze controller (Thin when the fronts touch at our −1 or deeper and the enemy front has ≥ 14 HP, else Last Hit) |
| g_bully_script | M4.5b `g_bully` (Last Hit + the no-dive bully), M4.5c's plan | Last Hit |
| g_farm_script | M4.5b `g_farm` (Last Hit + Hold) | Last Hit |

States: (i) **constructed** M4.5b zoning situations neutral / A's push / B's push × levels 6 / 10 × gaps equal / ×1.5
/ ×1.76 / ×2.2 (24 cells, K = 24); (ii) **natural HP edge**: each final's self-play states where the focal side's
HP-aware duel edge is ≥ 0.15, both present, the enemy outside its tower cell and catchable without a dive (24 per
final, K = 12); (iii) **semi-natural gap**: states from E3's ×1.76 self-play (training split), turns 3–80, the stronger
side focal, both present, power ratio still ≥ 1.4 (24 per final, K = 12). Opponent protocols: the policy, and the
scripted conceder `g_wary`.

**E2-P1 (primary)**: threaten_beh − policy at ×1.76, constructed cells pooled, protocol policy, ΔG60. A *conversion
gap* if ≥ +0.25 with the t interval over the six policies excluding 0.
**E2-P2**: each of threaten_beh, edge_beh, bully_beh, freeze_threat − policy at ×1.76 (Holm-adjusted over the four),
and the same contrasts on the semi-natural and natural families.
**E2-P3**: threaten_beh − farm_beh at ×1.76 (the value of the selective threat over passive play, both with the
policy's waves), and its gap gradient (equal → ×2.2).
**E2-P4 (measurement)**: g_bully_script − bully_beh and g_farm_script − farm_beh at ×1.76: the cost of forcing the
Last Hit wave in M4.5c's bully plan and M4.4c's Farm plan.
Channels at 20 and 50 turns for every contrast: the weaker side's farm lost while present, turns away, deaths and XP
lost; the stronger side's deaths, damage taken and connected trades.

### E4 — learned best responses from gap and freeze starts

Targets fixed now: finals **s0, s2, s4** (even seeds; M4.5c attacker gains +1.32 / +0.79 / +1.83). For each target
two attackers, warm-started from the target, PPO against the frozen target only, **4,096,000 learner transitions**
(the M4.5c attacker budget and settings), with one change: training games start from a start archive (`start_mix`
1.0) and last 150 turns (bootstrapped truncation), and the learner always plays the archive's focal side (the archive
state is mirrored when the learner draws seat B):

- **gap attacker**: archive = the constructed zoning situations × levels 6 / 10 / 14 × gaps ×1.5 / ×1.76 / ×2.2, plus
  semi-natural gap states from the target's E3 training-split games; the learner is the stronger side;
- **freeze attacker**: archive = the M4.5b freeze builder (equal / ×1.76 × held 5 / 12 / 20 × levels 4 / 10 / 15),
  plus natural freeze states of the target's self-play (training-split seeds); the learner is the freezer.

Evaluation on held-out starts (the constructed cells with evaluation seeds, semi-natural / natural states from
held-out games), the target always the opponent, the focal seat played by (i) the target itself, (ii) the attacker,
(iii) the M4.5c s0-attacker of the same target (control: general exploitability), same chance and sampling streams.
Also: each attacker evaluated from s0 (`evaluate_discounted`, 64 paired seeds, W = 460), and the gap attacker on the
equal-power versions of the constructed cells (is any gain gap-specific?).

**E4-P1 (gap)**: gap attacker − target at G100 on held-out gap starts. *Distribution binding* if the pooled mean is
≥ +0.5 with at least 2 of 3 targets significant **and** gap attacker − s0-attacker ≥ +0.3 pooled; *weak evidence* if
only the first holds; *not shown* otherwise. **E4-P2 (freeze)**: the same with the freeze attacker at G50.
**E4-P3**: the channels of any gain (as E2). A null result is reported as "not found at this budget", never as proof of
a ceiling (the user's priority 2).

### E6 — credit assignment (light, from E2 and E3 data)

The signal-to-noise ratio of the E2 plan contrasts per paired rollout (mean / SD of the per-rollout difference), and
the evaluated critics' error on Trade turns against other turns in E3's games (V(s) against the realized discounted
return over 100 turns plus the bootstrapped tail).

## 3. Decision mapping (fixed before results)

| finding | reading |
|---|---|
| E2-P4 negative and significant, or E3 shows large conversion through absence that M4.5c's metric missed | **evaluation / measurement** limitation in M4.5c's zoning verdict |
| E2-P1 or E2-P2 positive (a scripted plan beats the policy at a power gap against learned play) | the policy does not exploit a value the mechanics offer: **discovery or distribution** |
| … and E4-P1 meets "distribution binding" | **training distribution**: targeted starts teach it |
| … and E4-P1 not shown | **discovery** under PPO at this budget, or the plan's value is opponent-specific (checked with the channels) |
| E2-P1/P2 null and E2-P3 null (no threat beats passive farming at any gap against learned play), E4-P1 not shown | **incentives** bind: structural or scale, separated by E1 |
| E1-P2 finds a single load-bearing cost | **parameter / balance** (that cost's size) |
| E1 finds the break completes in one turn and no single cost is load-bearing | **structural / time-scale** (the contest resolves faster than any combat cost can accrue) |
| E5-P1 "nearly free" | the reset subsidy contributes, whatever else holds |

Smoke tests at tiny sizes check the pipelines only (pass / fail and timing); their numbers are not reported or used.
