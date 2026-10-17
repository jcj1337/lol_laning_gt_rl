# M5 design: the final training and evaluation campaign

*2026-10-17. Status: **proposed design, awaiting the user's decisions and campaign authorization.** M5a (this design,
the pre-registration, the drivers and the readiness check) is complete; no part of the campaign has run. Readiness:
[m5a_readiness_report.md](m5a_readiness_report.md). The frozen protocol:
[m5_preregistration.md](m5_preregistration.md).*

**Approved direction** (the user, 2026-10-17, in the M5a brief): M5 includes opponent modelling and compares the whole
package with continued pure self-play. Kept fixed by that brief:
- the baseline `m45_calibrated()`;
- the gold-difference reward, γ = 0.99, continuing time, bootstrapped truncation, no termination at level 14 or turn 94;
- no new mechanics, investment curves, discount experiments, critic variants or RL algorithms; the shared actor–critic;
- lane-phase reporting as D0 at turn 94 → E_20, with uncertainty and the XP / HP / absence / wave panel beside it;
- rollout evidence, never critic values, in strength-gap states; a standard strategic horizon of 50 turns.

Everything else below is a proposal. §12 lists what needs the user's decision before the campaign.

## 1. The questions

1. **Which strategic findings replicate on the final policies?** (M5c)
2. **What does the opponent-modelling package add beyond continued pure self-play?** (M5d, the primary contrast)
3. **Does it preserve competence, and how exploitable are the resulting agents?** (M5d)

OM2 already measured the belief's own value under mixed-opponent training (outcome 1; §AW of the roadmap). M5 does not
repeat that comparison. Its primary contrast is the overall package against what the project would otherwise train.

M5 succeeds by delivering credible evidence on all three, including negative findings. It does not need the modelling arm
to win.

## 2. The two arms

| | **A: opponent modelling (`om`)** | **B: pure self-play (`sp`)** |
|---|---|---|
| environment | `m45_calibrated(obs_last_action=True)`: the same game, the observation extended by the opponent's previous action (103 entries; reference §15.4) | `m45_calibrated()` (90 entries) |
| network | `PolicyValueNet` 116 → 128 → 128 tanh, 61 masked logits + value: `widen_inputs(ext/baseline:s, 116)`, zero weights on the 26 new inputs | `PolicyValueNet` 90 → 128 → 128: a copy of `ext/baseline:s` |
| extra inputs | OM2's belief: posterior over 11 library types + "mixed" + "unknown" (13); λ = 0.97, ε = 0.05; reset at every game start; the library fixed for the campaign | none |
| opponents in training | 50% of games current-policy self-play (both seats train); 50% one of OM2's 15 fixed opponents (6 archetypes, 5 learned library policies, 2 per-turn mixtures, 2 style switchers), the learner in a random seat, only the learner trains | 100% current-policy self-play |
| collector | `om2.collector.om_collector_cls("treatment")`, imported unchanged | `m46c.train.collector()`, imported unchanged |
| shared by both | the M4.6c start mixture (β = 0.532, archive games 150 turns, s0 games 512, mirroring ½; about 25% of transitions); the M4.6c PPO settings (entropy 0.03, clip 0.2, 4 epochs, minibatch 256, vf 0.5, value clip 0.5, gradient norm 0.5, 16 environments × 128-turn segments, γ 0.99, λ 0.95, reward scale 0.1); a fresh Adam at 3·10⁻⁴ annealed linearly to 0 over the budget; PPO seed 70,000 + s in both arms of pair s; snapshots every 25,600 transitions | |

**What "pure self-play" means here (verified, R4–R5).** It is exactly the continuation every M4 campaign since M4.5c
ran: `SharedConfig(current_fraction=1.0)`, so every training game is a current-policy game. A snapshot pool exists
(eight snapshots, one every 25,600 transitions), but with a current fraction of 1 no historical game is ever drawn. The
smoke confirms 100% current-policy transitions and equal seat counts. The "current/historical mechanism" the brief
asks to preserve is therefore the current-only setting; OpenAI Five's 80/20 mix was tested in M4.3b and bought nothing.

**Identical start (verified, R2).** Within each pair both arms start as `ext/baseline:s`: the widened network computes
the same actor and value outputs on any belief and any previous action. The equality is bit-exact in batched evaluation;
a single-row forward pass differs by at most 9.5·10⁻⁷ in a logit (a different BLAS kernel for a wider inner dimension).

**What is matched and what is not.**
- Matched: initial weights and outputs; the PPO seed; the hyperparameters; the agent-transition budget; and the
  sequence of game starts. R5 drew the first 200 starts of both collectors: the same kind (s0 or archive), the same
  archive state and mirroring, the same game length and the same chance seed in every game.
- Not matched: everything after the first update. The policies, the states they visit and their returns diverge.
  Common random numbers make the arms comparable; they do not make the training trajectories identical, and no
  claim of identical data is made.
- **Per transition.** A arm's fixed-opponent games are half the games but only about a third of its learner
  transitions (36% in the smoke): only the learner's seat is a sample there. At equal transitions, A therefore plays
  about 1.47 transitions per environment turn against B's 2.0, and so about 36% more turns. A also gets about 0.64 × 32.768M ≈ 21M self-play
  transitions against B's 32.768M. The equal-transition budget is the brief's matching rule and is kept; this
  asymmetry is part of the package.

**What the contrast estimates.** A − B changes two things at once: the information (previous action and belief) and
the opponent distribution (half the games against fixed opponents). It estimates the **package effect**, not either
ingredient. OM2 bounds the belief's part under the mixed distribution (+2.0 against archetypes, safe). Neither arm
isolates the archive-start effect, since both use the same start mixture. Neither arm is fresh training: both continue a
24.576M-transition policy (§3).

## 3. Lineage and budget

**Lineage of the start** (verified from checkpoint metadata, R0; identical for s = 0…5):

| stage | schedule | starts | PPO seed | transitions |
|---|---|---|---|---|
| M4.5c `ext` final | fresh, 16.384M, lr 3·10⁻⁴ → 0 | s0 only, 100% current-policy | s | 16,384,000 |
| M4.6c `ext/baseline:s` | continuation, 8.192M, fresh Adam, lr re-warmed | M4.6c mixture (25.0–25.5% archive transitions) | 98,000 + s | 8,192,000 |
| **M5 (proposed)** | continuation, 32.768M, fresh Adam, lr re-warmed | M4.6c mixture | 70,000 + s | 32,768,000 |

Prior training is therefore 24.576M transitions over two schedules; after M5 it would be 57.344M over three, with
three learning-rate re-warms. Each continuation pays a re-warm cost (about 0.5 resources over the first million
transitions in M4.6c). The closeout's original proposal of fresh training was replaced in the brief by this
continuation, so that M5 inherits the competence OM2 built on.

**Budget (proposal).** 32,768,000 additional agent transitions per run; the matched checkpoint is the first snapshot
at or after 16,384,000. Sp updates hold exactly 4,096 transitions, so its snapshot is exactly at 16,384,000. Om updates
vary (about 3,070), so its snapshot falls within one update after. There are 12 runs and 393,216,000 additional
transitions in total. This is the brief's proposal. M5a found no evidence for a different size, and the measured costs
are in §10.

## 4. The M5 package (`lanegame/experiments/rl/m5/`)

| module | content |
|---|---|
| `common.py` | arms, configurations, budgets, the seed block 70,000–70,999, policy specifications, checkpoint identities, the code digest, pre-registration freezing |
| `train.py` | `train_shared` with each arm's imported collector; run manifests; health checks; restart handling; status |
| `players.py` | `OMPolicy` (the om network with its belief and previous codes, for every engine loop) and `crn_policy` (CRNNet for 90 inputs, OMPolicy for 103 or 116) |
| `games.py` | games from s0 (M5b, M5d): network seats and the OM2 opponents plus imitate-then-switch, conditional styles and network agents; the lane-exit panel |
| `battery.py` | the M4.5c battery for both formats |
| `cells.py` | the M4.6 limitation families for both formats |
| `attackers.py` | attacker training against frozen targets (live or tabled belief) and their evaluation |
| `evaluate.py` | the M5b–M5d designs as job lists, the gated stage drivers, `plan()` |
| `analysis.py` | the pre-registered statistics and classifications |
| `readiness.py`, `bench.py`, `instruments.py` | the M5a checks R0–R10, the concurrency benchmark and the budget |
| `report.py`, `__main__.py` | generated tables; the CLI and its gates |

**Shared code changed, opt-in only.** The shared rollout loops never told a policy what its opponent played. Four
loops now call an `observe_turn` hook after each turn: `m45b.engine.play`, `m45c.horizon.rollout`,
`m46a.engine.rollout` and `diagnostics.lane_exit.continuation`. The plan wrappers (`Then`, `Fixed`, `BehaviorPlan`,
`ResetPlan`, the M4.6d `Plan` and `TwoNet`) list their inner policies, so the hook reaches a stateful policy inside a
plan. `m45c.horizon.critic_value` asks a stateful policy for its own value. Policies without these attributes are
untouched. R3 reproduced stored M4.5c battery, M4.6c weaker-side, M4.6f (C1) cross-candidate and OM2 evaluation
records exactly through the hooked loops, and the digest pins pass.

### 4.1 Belief and previous-action handling (identical to training, verified in R6 and R8)

| situation | belief | previous-action field |
|---|---|---|
| a game from s0 | uniform prior | "none" |
| an archive start in training | uniform prior (reset at every game start) | "none" |
| a constructed evaluation cell, or a natural state of the common start's games | uniform prior, the archive-start convention | "none" |
| a continuation branched from a recorded om game (battery natural states, lane-exit states) | the recorded log posterior, update count and previous codes of that turn; the same for every candidate | the recorded codes |
| a mirrored state (focal seat B → A) | the two seats' records swapped | swapped |
| after every turn | updated with the opponent's code on the opponent's start-of-turn observation; Absent-only turns change nothing | the turn's codes |
| a truncation bootstrap | the belief after the last update (the input of the next turn) | the last codes |
| an episode boundary | discarded with any unapplied update | reset |

Only public information enters: the start-of-turn state and the code just played. The current turn's codes are never
part of the current input. R6 rebuilt every om training input of one full rollout (2,863 inputs, s0 and archive
games, every game start) and 14 truncation bootstraps independently with `OMPolicy`. The state and previous-action
blocks match bit for bit, and the belief block within 2.9·10⁻⁶. That residual is float32 rounding: training batches the
likelihood forward passes across environments, while the evaluator runs them one at a time.

### 4.2 Seeds, manifests, gates and restarts

- **Seed block 70,000–70,999** (R1: no other study's root constant lies in it):
  - training 70,000 + s;
  - benchmark 70,050;
  - curves 70,100, history 70,150;
  - head to head 70,200, families 70,300;
  - battery 70,400 / 70,450, cells 70,500, lane exit 70,550;
  - attackers 70,600 / 70,700;
  - readiness 70,800, traces 70,900.
- **Run manifest** (`train.run_manifest`, written before training as `results/m5/manifest_main_<time>.json`):
  - the configuration and the code revision (git commit, dirty flag and a SHA-256 over every source file);
  - the start checkpoints' file SHA-256s and weight digests, the library networks and the archive file;
  - the seeds and the pre-registration hash.
- **Gates.** `train` needs `--confirm-m5-train`; the evaluation stages (`monitor`, `battery`, `cells`, `families`)
  need `--confirm-m5-eval`; the attacker stages need `--confirm-m5-attackers`. Every campaign stage refuses to start
  without the frozen pre-registration.
- **Resuming.** The trainer stores no optimizer or collector state, so a run cannot resume mid-schedule.
  - Complete runs are skipped.
  - An incomplete run directory is refused, unless `--restart-incomplete` is passed. That flag moves it intact to
    `train/main/partial/<run>_<time>/` and restarts the run from its start.
  - By the determinism contract (R4: re-runs bit-identical), the restart must reproduce the preserved history prefix
    exactly, and `train.prefix_check` verifies this.
  - Evaluation cells are resumable file by file.

## 5. M5b: training

- **Runs.** The 12 runs of §2–§3. The longer om runs are submitted first.
- **Health checks** (`train.check`, pre-registered):
  - finite statistics;
  - the budget reached within one update;
  - no termination, and one truncation per finished game;
  - |V| ≤ Ω;
  - no illegal action;
  - initial weights equal to the arm's start;
  - an archive transition share of 0.20–0.30;
  - the matched checkpoints present;
  - none of M4.5c's stop conditions (sustained KL > 0.05, a gradient norm > 100, entropy < 0.3 nats, an illegal or
    never-legal code);
  - sp: 100% current-policy and equal seats;
  - om: a fixed-opponent game share of 0.45–0.55.
- **Learning curves** (monitoring): every run's 16 snapshots (every 2.048M) against a fixed panel: its own start
  `ext/baseline:s`, another seed's start, `g_farm`, `g_shove` and `g_bully`. 16 pairs × 150 turns, read at turn 114.
- **History.** Each final against its own 25%, 50% and 75% snapshots and against `ext/baseline:s`, 64 pairs × 500
  turns.
  - **Convergence probe:** the final against its 75% snapshot (24.576M). It is reported as "still improving" when the
    lead is ≥ 0.25 with its interval above 0.
- **Matched checkpoints.** The 16.384M midpoints play the primary families at 32 pairs (budget sensitivity).
- **Monitoring is separate from evaluation.**
  - Curves and history never select a checkpoint and never change an endpoint.
  - The finals at 32.768M are the evaluated policies whatever the curves show.
- **Technical failure and restart.**
  - **A crash or interruption:** restart from scratch with `--restart-incomplete`; the prefix must reproduce.
  - **A health-check failure or stop condition:** investigate.
    - An implementation defect: fix it, then restart the affected runs. If the defect touches both arms, restart both
      runs of every affected pair. Report it.
    - A property of training (for example entropy collapse) that is not a defect: the run stays as it is and is
      reported. It is never re-run with another seed.
  - **More than two pairs failing technically:** stop and report.
- **No extension rule.** The results at 32.768M are final. Continued improvement is reported as non-convergence, a
  limitation. Any extension would need a new approval and a dated amendment, apply to both arms equally, and could not
  be triggered by the direction of a result.

## 6. M5c: strategic evaluation

Each instrument is imported and runs on both arms' finals. The rollouts never use a critic bootstrap in strength-gap
or constructed states.

| finding / M4 limitation (closeout §4) | instrument (M5 module) | design |
|---|---|---|
| conditional wave management, Shove (G5, G6); whether Last Hit / Thin / Slow Push still coincide | the M4.5c battery `A_wave`, `A_wave_early` (`battery`) | 48 natural states (24 early), K 12, H 50, the M4.5c validation subset at 300 turns |
| freeze preservation and breaking (G3, G4) | battery `B_freeze`, `B_frozen`, `F_freeze_c`; M4.6d freeze cells, freezer and breaker (`cells.freeze`, M4.6f B3) | battery as M4.5c; cells K 32 × H 50 |
| trading, concession and wave risk (G7, G9) | battery `C_trade`, `C_trade_early`, `G_trade_c`, `H_zone_c`; M4.6c weaker-side cells (`cells.weak`) | battery as M4.5c; cells K 32 × H 50 |
| Recall timing (G8), Safe at weak-side Recall windows (G9 miss) | battery `E_recall`; Recall windows (`cells.recall`, M4.6f B4) | K 32 × H 100 |
| conversion of plausible early advantages (G1, G2) | lead premium J1 (`cells.premium`); strength use (`cells.conv`, M4.6f B2) | premium K 64 × G20/G50; conv K 16 × H 150 |
| natural strength and freeze states | natural leader / trailer / freeze states of the common start (`cells.natural`, M4.6f B5) | K 16 × H 150 (freezes H 50) |
| large gaps (G10, a generalization diagnostic) | M4.6a E4 gap cells with a gap specialist as a third candidate (`cells.gaps`) | K 16 × H 150 |
| Wards (G11) | battery `D_ward`, `I_ward_c` | as M4.5c |
| lane phase | D0 at turn 94, the lead at 114 (= E_20 at turn 94 in expectation) and the panel, from every M5d game | — |

- **Per-arm families.** The battery, the weaker-side cells and the lead premium are run on each final against its own
  policy: the policy's behaviour in its own game.
- **Cross-arm families** (conv, freeze, recall, natural, gaps; M4.6f's design):
  - both arms' finals are paired candidates in seat A, against each arm's final in seat B, on the same rollout seeds;
  - om − sp is the M5 contrast, and each candidate against its own arm is the within-arm reading.
- **Beliefs.** The om arm's natural battery states come from its own games with their recorded beliefs. Constructed
  cells, and the common start's natural states, start at the prior (§4.1).
- **Replication (S4, pre-registered).** A M4.5c "discovered" verdict holds in a seed when no fixed alternative beats the
  policy by more than 0.2 in that natural-state family. It replicates with at least 5 of 6 seeds, reported per arm.
  - The verdicts: conditional wave management, conditional freezing and the reply to freezes, selective trading,
    concession under threat, and Recall timing.
  - Every §4 class of the closeout (used / missed / weak / open) is re-measured with the instrument above. There are
    no target values: a change either way is a result.

## 7. M5d: opponent modelling and robustness

**Games from s0** (`games`; 64 paired-seat pairs × 150 turns unless stated). The focals are om:s, sp:s and, as an
exploratory reference only, the common start `ext/baseline:s`, all on the same games. Here o = (s + 3) mod 6.

| family | opponents | role |
|---|---|---|
| **S_in** recognizable, in the library | the 6 OM2 archetypes | exploitation (P1) |
| **S_out** recognizable, held out | the 4 held-out scripted variants of OM2; 3 conditional styles (respect, passive, stepback) on the competent base `ext/baseline:o` | exploitation (P1) |
| **M** misspecified | 2 held-out per-turn mixtures, 2 held-out switchers; 3 imitate-then-switch opponents (`g_farm` until turn 40, `g_safe` until 20, `g_shove` until 40, then `ext/baseline:o`) | robustness (secondary) |
| **C_hist** competent, historical | `ext/baseline:o` for the five other seeds; the 5 held-out learned instances (attacker, freeze and gap specialists, short-horizon, C1) | competence (P2a) |
| **C_cross** competent, final | the other seeds' finals of both arms (10) | competence (P2a) |
| **H2H** | om:s vs sp:s (128 pairs × 150, and 32 × 500); om:s vs sp:o for the five o ≠ s (32 × 150) | competence (P2b); cross-seed |
| **long** | C_hist, C_cross and the same-seed head to head at 500 turns (the discounted objective) | competence beyond the lane phase |
| **mid** | the 16.384M midpoints on S_in, S_out, C_hist and the head to head (32 pairs) | budget sensitivity |

- **The imitate-then-switch diagnostic** is a scheduled switch:
  - The opponent plays a library archetype that the belief identifies within a few turns, then a competent policy.
  - The post-switch window measures the adaptation lag OM2 saw (10–60 turns).
  - A scheduled switch is not evidence about a deliberately deceptive learned opponent; that is the attacker's
    question.
- **Mechanism (descriptive).** The om finals' belief traces at turns 1–94:
  - identification of the archetypes;
  - catch-all mass against mixtures;
  - tracking of switches;
  - belief-swap probes.

### 7.1 Attackers

| item | proposal |
|---|---|
| targets | each arm's final (12); for the adaptation test, the om final with its belief inputs replaced by the mean belief of its own self-play at that turn (`om_table`, 6): the same network without in-game adaptation |
| access | black box: the attacker plays the target. The om target's belief is live, updated from the attacker's actions exactly as in deployment; no weights, belief or gradients are read |
| observation | the om input format for every attacker: the state, the target's previous action, and the attacker's own belief over the OM2 library (116). Attackers of both arms have the same information channel as arm A |
| start | a copy of the target in that format: an om target's own weights; an sp target widened with zero weights. This is the parity start of M4.5c, so a gain measures exploitability, not catching up |
| objective | the training objective unchanged: zero-sum gold difference, γ 0.99, continuing time, the learner in a random seat, only the learner trains. **No reward for deception or for moving the target's belief.** Whether exploiting the adaptation emerges is measured |
| games | s0 starts, 512-turn bootstrapped games (the M4.5c attacker protocol); the M4.6c PPO settings; fresh Adam, lr annealed over the attacker's budget |
| budget | **recommended: quarter = 8,192,000 transitions**, a quarter of the target's M5 continuation budget (32.768M), M4.5c's ratio. The 24.576M of prior training is shared by both arms' targets and inherited by every attacker through its parity start, so it is not in the denominator. **Alternative: half = 16,384,000**, as the closeout and M4.5c proposed; half of the cumulative 57.344M would be 28.672M. The quarter budget is recommended for cost (§10–§11): 10.3 h against 20.6 h for the 18 attackers |
| evaluation | independent games from s0, 128 pairs × 500 turns: the lead at 114 and the discounted lead. Transfer: against the other arm's final of the seed and against the common start. The om target's belief trace against its attacker |
| reading | exploitability lower bounds, limited by budget, network and optimizer. A − B in exploitability, paired by seed |

- **Adaptation-exploitation test.** Exploitability of `om` (live belief) minus exploitability of `om_table`, each
  against its own dedicated attacker of the same budget.
  - Positive: the live adaptation is a channel a learned attacker exploits.
  - Negative: the adaptation protects the agent.
  - Cross-evaluation (each attacker against the other target) separates target-specific tuning from the channel.
  - The attacker's policy is stateless: it can condition on the state, the target's last action and its own belief,
    but it has no clock. A deliberately timed deception is covered only by the scheduled imitate-then-switch.

## 8. M5e: closeout

- **The report:**
  - strategic replication per arm;
  - learning progress, the convergence probe and the limits of the evidence;
  - exploitation gains (style families) and competence and robustness trade-offs, side by side, never netted;
  - exploitability lower bounds;
  - the recommended final configuration for M6, and every remaining limitation.
- **Reproducibility materials:**
  - the frozen pre-registration and manifests;
  - checkpoint digests;
  - one command per stage;
  - representative viewer traces from the finals (seeds 70,900+).
- **Labels.** Findings are labelled demonstrated, not demonstrated, or inconclusive, as the classifications of the
  pre-registration give them.

## 9. Statistics (summary; the binding text is the pre-registration)

- **Unit and readouts.**
  - Units: training seeds (pairs), paired om − sp; 95% t intervals over six seeds.
  - Game-level uncertainty is kept apart: each seed's value carries its game SE, and the between-seed variance is split
    into a game part and a seed part.
  - Value: the lead at turn 114 (E_20 at the lane exit); D0 and the non-gold panel beside it; the discounted lead over
    500 turns in the long families.
- **Primary endpoints.**
  - **P1, exploitation gain:** om − sp against the style families, ½ S_in + ½ S_out, equal weights within each.
    Demonstrated when the mean is ≥ +0.25 and the interval lies above 0.
  - **P2a, competence:** om − sp against ½ C_hist + ½ C_cross.
  - **P2b, competence:** om's lead over sp in the same-seed head to head.
- **Competence uses non-inferiority, not "no detected loss".** The margin is δ = 0.25 resources.
  - Preserved when the lower bound is > −δ.
  - Material loss when the upper bound is < 0 and the mean is ≤ −δ.
  - Otherwise inconclusive (insufficient precision), never reported as preserved.
- **Why δ = 0.25.**
  - It is OM2's and M4.7's pre-registered loss threshold, and within M4.6f's materiality range (0.20–0.30).
  - It is about 0.6 turns of income and 7% of the mean |lead| at the lane exit.
  - It is a quarter of OM1's value of information and a quarter of M4.5c's attacker gain.
  - A margin of 0.5 would let half of the modelling gain OM1 measured disappear unremarked.
- **Resolution with six seeds** (Monte Carlo of the rule, `analysis.ni_resolution`). OM2's competent-opponent contrast
  had a between-seed SD of 0.33, of which 41% was game sampling (3 seeds).
  - Assumed per-seed SD 0.28: a true loss of 0.5 is called a material loss 93% of the time, and a true gain of +0.25
    is called preserved 93% of the time.
  - A true null is called preserved 42% of the time and inconclusive 55%.
  - Ten pairs would raise the null case to 71%.
  - More games per cell help little: the seed part dominates.
  - **Six pairs therefore resolve large effects but probably not a near-zero competence effect.** This is a decision
    for the user (§12).
- **Headline outcomes:**
  1. pays and preserves competence;
  2. pays at a competence cost;
  3. pays, competence unresolved;
  4. no material gain;
  5. harmful.
- **Secondary endpoints.**
  - The members: robustness family M; the long-horizon competence; cross-seed head to head; the midpoints;
    exploitability A − B; the adaptation test.
  - Claimed only when Holm-adjusted (familywise 0.05).
  - Robustness families use the same non-inferiority rule.
- **Exploratory.** Each arm against the common start; per-opponent tables; the belief mechanism.

## 10. Measured costs and the campaign budget

All figures measured in M5a on this machine (i5-11600K, 6 cores / 12 threads, 16 GB), see the readiness report.

**Training throughput** (R9: short runs of the campaign's code, all runs of a configuration measured in their common
window):

| concurrent runs | om per run | sp per run | aggregate |
|---|---|---|---|
| 4 om | 1,430 | | 5,718 |
| 5 om | 1,200 | | 6,002 |
| 6 om | 1,221 | | 7,329 |
| 8 om | 945 | | 7,562 |
| 6 sp | | 2,805 | 16,832 |
| 8 sp | | 2,443 | 19,547 |
| 3 om + 3 sp | 1,144 | 2,859 | 12,010 |
| 3 attackers vs om + 3 vs sp | 686 | 849 | 4,606 |

Short runs differ from full-length ones in opposite directions for the two arms:
- sp: 2,805 against the identical code path's full-length 2,450–2,620 at 6 concurrent (M4.6f);
- om: 1,200 at 5 against OM2's full-length treatment runs, 1,416–1,645 at 5 workers.

The budget therefore gives a range:
- **conservative:** both arms scaled by the sp factor 0.90;
- **full-run evidence:** each arm scaled by its own factor.

Peak memory is about 640 MB per process.

**Worker count: 6, the om runs first.** The campaign was simulated event by event, each run progressing at the rate of
the measured configuration it is in:

| workers | om first | interleaved |
|---|---|---|
| 4 | 13.4–16.1 h | 13.7–17.9 h |
| 5 | 12.6–16.4 h | 12.9–16.9 h |
| **6** | **9.4–11.8 h** | 12.0–16.3 h |
| 8 | 11.1–12.9 h | 10.9–13.6 h |

- Eight processes add only 3% of om throughput (6 physical cores) and slow every om run by 23%.
- Mixing sp runs among the om runs does not speed them up (1,144 beside three sp runs).

**Evaluation costs** (R10: steady-state turns per worker-second with six classes running at once):

| class | rate |
|---|---|
| s0 games, no om seat / one / two | 1,360 / 560 / 353 |
| battery, sp / om | 990 / 331 |
| cells, cross-arm / om / sp | 520 / 395 / 1,091 |
| lane-exit continuations (om) | 325 |

An om turn costs about three times an sp turn. The belief's per-turn likelihoods (five network forwards and six
scripted decisions) dominate. An exact saving of the repeated configuration copies is already in the evaluator.

**The campaign budget** (6 workers; `results/m5/budget.json`):

| part | content | wall clock |
|---|---|---|
| M5b training | 12 runs × 32.768M (om runs 5.8–8.3 h each, sp runs 3.6 h) | **9.4–11.8 h** |
| M5b monitoring | 1,008 curve and history cells, 7.7M turns | 0.5 h |
| M5c battery | the full M4.5c design with M4.6b's twins (8.33M turns per final; M4.5c ran 6.54M before the twins) × 12 finals | 9.3 h |
| M5c cells | the seven limitation families, 17.3M turns | 1.5 h |
| M5d families | 546 cells, 28.8M turns | 1.7 h |
| M5d attackers, quarter (recommended) | 18 × 8.192M (6 without the adaptation test: 12 × 8.192M) | 10.3 h (6.6 h) |
| M5d attackers, half (alternative) | 18 × 16.384M (12 × 16.384M) | 20.6 h (13.3 h) |
| attacker evaluation and belief tables | 54 cells; 6 tables | 0.7 h |
| **total** | with quarter attackers / with half | **≈ 34–36 h / ≈ 44–46 h** |

Training must finish before the evaluation and the attackers. The finals are their inputs; the om finals' self-play
tables are inputs of the adaptation test.

## 11. Deviations from the brief and evidence-backed changes

Each item gives the brief's proposal or the earlier plan, the evidence, the change, its effect on interpretation and
its status. Nothing proposed here is adopted until the user decides.

| # | item | original | evidence | proposed / done | effect on interpretation | status |
|---|---|---|---|---|---|---|
| 1 | attacker budget | half the target's budget (closeout §8.3; M4.5c's recommendation) | R9/R10: 620–770 transitions/s per attacker; 18 attackers cost 20.6 h at half, 10.3 h at a quarter | **quarter (8.192M)**, M4.5c's ratio; half documented as the alternative | weaker lower bounds than at half; M4.5c's quarter-budget attackers still found +1.0 | **proposed, needs a decision** |
| 2 | adaptation-exploitation test | "a specified test" (brief) | none needed | dedicated attackers against `om_table` (the om final without in-game adaptation), matched budget; 6 extra attackers | separates exploitation of the adaptation from general exploitability | **proposed, needs a decision** |
| 3 | attacker observation and start | to be defined (brief) | parity start is M4.5c's protocol; one input format for both arms keeps the attackers matched | the om input format for every attacker; a copy of its target | equal information channels across arms; the attacker has no clock | proposed |
| 4 | seed count | six pairs (brief) | `analysis.ni_resolution` with OM2's competent-opponent SD: a near-zero competence effect is called preserved only 42% of the time | **kept at six**; ten pairs documented (null case 71%) | a small competence effect is likely to be reported as inconclusive | **kept; the alternative needs a decision** |
| 5 | workers and run order | "five, six or fewer" to be investigated (brief) | R9 and the simulation (§10) | 6 workers, om runs first | none | proposed |
| 6 | fixed-opponent share | 50% of games (OM2) | measured: 51% of games are 36–37% of learner transitions; arm A plays about 1.47 transitions per turn | unchanged (equal transitions is the brief's matching rule); reported | arm A has about 21M self-play transitions against B's 32.8M; part of the package contrast | reported, no change |
| 7 | shared rollout loops | none planned | the instruments could not run a history-dependent policy (four loops never passed the opponent's code; guards refused 116 inputs) | opt-in `observe_turn` hook, `inner_policies`, `value_at`; verified bit-identical on stored M4.5c / M4.6c / M4.6f / OM2 records | none for earlier results | done (library change, opt-in) |
| 8 | large-gap specialists | the seed's M4.6a gap specialist | M4.6a trained specialists for seeds 0, 2 and 4 only (R0) | seeds 1, 3 and 5 use the specialist of seed s − 1 (M4.6a: specialists transfer, 15/15) | a cross-seed specialist for half the seeds; it is a diagnostic | done (data limitation) |
| 9 | the battery's size | M4.5c's battery, 6.54M turns per final | the M4.6b twins added to five families (brief: "with the M4.6b corrections") make 8.33M; om turns cost about 3× | the full design, both arms | about 9.3 h; a reduced design (24 natural / 12 early states) would save about 3 h and halve the natural families' precision | proposed (a reduction is a user option) |
| 10 | natural states of the cross-arm cells | each policy's own natural states (M4.6f used the start's) | neither arm's own states are neutral | the common start's natural states (no recorded belief: the archive-start convention) | the cross-arm contrast is measured on states neither arm produced | proposed |
| 11 | lane-exit reporting | D0 → E_20 with the panel (approved) | OM2's estimand: the lead at turn 114 in s0 games is E_20 at turn 94 in expectation; R10 checked the per-state readout on recorded om states | population D0 / E_20 / panel from every M5d game; the per-state continuation readout available, not budgeted | none | proposed |
| 12 | R6, first run | — | the first reconstruction demanded the whole input bit for bit and failed on the belief block only (max 2.9·10⁻⁶: float32 rounding of batched against single likelihoods) | state and previous action bit-exact, belief within 10⁻⁵, the same tolerance as R2 and R8; both runs reported | none | done |
| 13 | extension | — | — | none: the 32.768M results are final | non-convergence is reported as a limitation | proposed |

## 12. Decisions and approvals needed before the campaign

**Approved by the user** (the M5a brief, 2026-10-17):
- the direction: opponent modelling enters M5 as a package compared with continued pure self-play;
- the fixed items listed at the top of this document;
- the starting proposal: two arms, six seed pairs, continuation of `ext/baseline:s`, OM2's settings, the fixed
  library, and 32.768M with a 16.384M matched checkpoint as a proposal.

**Decisions needed before the campaign:**

| # | decision | options | recommendation | consequence |
|---|---|---|---|---|
| 1 | **authorize M5b training** | 12 runs × 32.768M on 6 workers | yes, as pre-registered | 9.4–11.8 h; `--confirm-m5-train` |
| 2 | **seed count** | six pairs; ten pairs | six (as proposed) | ten: +8 runs (about +7 h of training, +67% of the per-seed evaluation), and a near-zero competence effect is resolved 71% instead of 42% of the time |
| 3 | **attacker budget** | quarter (8.192M); half (16.384M) | quarter | half: +10.3 h, tighter lower bounds |
| 4 | **the adaptation test** | include (6 more attackers); omit | include | omit: −3.7 h at a quarter budget; S7 is not measured |
| 5 | **evaluation budget** | the full battery; a reduced battery | full | reduced: about −3 h, half the natural states per family |
| 6 | **the protocol's rules** | δ = 0.25, the classes, the families, the replication rule, Holm for secondary claims | as frozen | any change is a dated amendment before training |

**Approvals needed, by flag:**
- training: `--confirm-m5-train`;
- evaluation (M5b monitoring, M5c, M5d families): `--confirm-m5-eval`;
- attackers, their tables and their evaluation: `--confirm-m5-attackers`.

**Campaign status: not launched, awaiting approval.**
