# How the PPO self-play pipeline works

*Written for M4.4a (2026-10-13). It describes the code as it is, for the three four-action environments of M4.4
(`GameConfig.m4_four_actions()`, `m4_four_actions_econ()`, `m4_four_actions_econ_aggro()`); the earlier presets differ
only where noted. Every number in the worked example comes from a real network of the M4.4a smoke run
(`results/rl/checkpoints/m44/smoke/aggro_s0/final.pt`, 204,800 transitions) playing the `aggro` environment.*

**Where the code is.** `lanegame/env/observation_m4.py` (the observation), `lanegame/config.py` (`champ_act`,
`champ_wave`, ... the action encoding), `lanegame/rl/network.py` (the network and the masked distribution),
`lanegame/rl/shared.py` (shared-policy self-play: collection and the training loop), `lanegame/rl/ppo.py` (GAE and
the PPO update), `lanegame/rl/train.py` (`GameTimes`: game lengths and truncation), `lanegame/rl/agent.py` (frozen
networks as agents, checkpoints), `lanegame/experiments/rl/m44/common.py` (the M4.4 settings).

---

## 0. The picture in one paragraph

Two champions play the same lane. Every turn each one receives a list of 88 numbers describing the game **from its
own side** and a list of which of the 61 action codes are legal. One neural network turns those 88 numbers into a
score for every action and an estimate of how much gold its side will gain from here on. Illegal actions are
removed, the scores become probabilities, and each champion draws its action. Both actions are revealed at once and
the simulator resolves the turn. The reward is the gold gap the turn created (+r for one side, −r for the other).
After 16 parallel games have each advanced 128 turns, the 4,096 recorded decisions (both seats) are used to nudge
the network: actions that turned out better than the network expected become more likely, and the value estimate is
moved towards what actually happened. Repeat 500 times.

---

## 1. The input: the 88-feature observation

`observe_m4(state, k, side, cfg, champ)` returns a float32 vector with every entry in [0, 1]. It is a function of
the lane, the champion state, the spawn phase `k mod 4` and the observing side only: no turn counter, no history.
The full list of names is `m4_feature_names(cfg)`; the groups are:

| block | entries | feature names | normalization |
|---|---|---|---|
| phase | 4 | `phase_0` … `phase_3` | one-hot of k mod 4 (waves spawn every 2 turns) |
| own lane | 19 | `own_front_x-4` … `own_front_x+4`, `own_front_hp`, `own_rears_x-4` … `own_rears_x+4` | front cell one-hot (all zero: no front); front HP / 24; rear packets per cell / 4 |
| enemy lane | 19 | `enemy_front_x…`, `enemy_front_hp`, `enemy_rears_x…` | the same, **mirrored into own coordinates** |
| own side | 3 | `own_siege_mass`, `own_vision`, `own_siege_eligible` | my crashed minions under the enemy tower / 96; my vision timer / 6; crash present (0/1) |
| enemy side | 3 | `enemy_siege_mass`, `enemy_vision`, `enemy_siege_eligible` | the same for the opponent |
| own tower progress | 4 | `own_tower_dmg`, `own_plates_broken`, `own_plate_progress`, `own_plates_left` | damage **I** dealt to the enemy tower / 6000; plates broken / 3; progress to the next plate; plates left / 3 |
| enemy tower progress | 4 | `enemy_tower_dmg`, … | the same for the damage dealt to **my** tower |
| own champion | 16 | `own_champ_x-2` … `own_champ_x+2`, `own_champ_hp_frac`, `own_champ_hp_max`, `own_champ_absent`, `own_champ_skill`, `own_champ_level`, `own_champ_xp_progress`, `own_champ_invested_whole`, `own_champ_gold_whole`, `own_champ_invested_frac`, `own_champ_gold_frac`, `own_champ_attack_power` | cell one-hot; HP / max HP of the level; max HP / max HP at level 18; turns away / **3 (four) or 5 (econ, aggro)**; skill (0.5); (level − 1) / 17; XP progress within the level; whole invested resources n/(n + 64); whole unspent resources n/(n + 8); the sub-resource remainders; attack power AP/(AP + AP(18, 0)) |
| enemy champion | 16 | `enemy_champ_…` | the same, the cell mirrored |

**Own perspective.** Each side stores its data in its own coordinates (own tower −2, the lane cell 0, enemy tower
+2), so mirroring the game is just swapping the two records. Seat A's vector lists A's data in the "own" blocks and
B's in the "enemy" blocks; seat B's vector is built the same way from B's side. At the symmetric start s0 the two
vectors are identical. The encoding is exactly invertible (`decode_m4`), which the tests use to check that seat B's
vector is seat A's view of the mirrored game.

**Example** (the worked turn of §5; nonzero entries only). Both fronts fight at the lane cell; A's front has 7 HP,
B's 13 HP; both champions stand at their own −1; A has full HP, B 70%:

| feature | seat A | seat B |
|---|---|---|
| `own_front_hp` / `enemy_front_hp` | 0.292 (7/24) / 0.542 (13/24) | 0.542 / 0.292 |
| `own_champ_hp_frac` / `enemy_champ_hp_frac` | 1.000 / 0.698 | 0.698 / 1.000 |
| `own_champ_level` / `enemy_champ_level` | 0.118 (level 3) / 0.176 (level 4) | 0.176 / 0.118 |
| `own_champ_x-1`, `enemy_champ_x+1` | 1, 1 | 1, 1 |
| `own_champ_gold_whole` | 0 (0.49 unspent) | 0.273 (3 whole resources unspent) |

Everything in A's "own" columns appears in B's "enemy" columns and vice versa: that is the only difference between
the two inputs.

**Economy-specific scale.** `econ` and `aggro` allow deaths of up to 5 turns (`champion_absent_max` 5) instead of 3,
so the absence entries are divided by 5. A network trained on one preset therefore reads the other preset's
observations slightly wrongly; `experiments/rl/m44/common.load_policy` refuses such a cross-preset load unless it is
explicitly requested.

---

## 2. The output: 61 action codes

An action is a triple (wave management, behaviour, siege objective) packed into one integer:

| codes | meaning |
|---|---|
| 0 … 44 | wave ∈ {Last Hit = slot 0, Shove = slot 1, Thin = slot 2} × behaviour ∈ {Hold, Yield, Trade, Safe, Recall} × objective ∈ {none, Pressure, Vision}: `a = wave + 3·behaviour + 15·objective` |
| 45 | ABSENT (the only legal code while the champion is dead or recalling) |
| 46 … 60 | Slow Push (slot 3) × behaviour × objective: `a = 46 + behaviour + 5·objective` (appended so that codes 0–45 keep their M4.3b meaning) |

Only **34 of the 61 codes can ever be legal**: 16 lane codes (4 waves × Hold / Yield / Trade / Safe), 16 siege codes
(4 waves × Hold / Trade × Pressure / Vision, only while my crash is under the enemy tower and I can reach it), the one
Recall code 12 (at or behind my −1 cell) and ABSENT. The other 27 codes exist only because the encoding is a product
(Safe with Pressure, Recall with Shove, ...). A typical lane turn has 17 legal codes.

**From scores to probabilities** (`network.MaskedCategorical`). The network outputs 61 raw scores (logits). Every
illegal code's logit is replaced by −10⁹, so after the softmax

$$\pi(a\mid o)=\frac{e^{z_a}}{\sum_{b\ \text{legal}}e^{z_b}}\quad\text{for legal }a,\qquad \pi(a\mid o)=0\ \text{exactly for illegal }a .$$

An illegal code can never be sampled, contributes nothing to the entropy and receives no gradient. In the worked
turn the network gave code 13 (Recall with Shove, never legal) a raw logit of −0.013, harmless because it is masked.

**Sampling.** During training the collector draws with `torch.multinomial` from these probabilities (seeded
generator). During evaluation a `NetAgent` draws with one uniform from the caller's NumPy generator (inverse CDF), so
two evaluations on the same seed take the same actions. Seat A's top six probabilities in the worked turn:

| code | action | probability |
|---|---|---|
| 6 | Last Hit + Trade | 0.253 |
| 48 | Slow Push + Trade | 0.239 |
| 49 | Slow Push + Safe | 0.110 |
| 8 | Thin + Trade | 0.093 ← sampled |
| 3 | Last Hit + Yield | 0.085 |
| 9 | Last Hit + Safe | 0.064 |

---

## 3. Shared-policy self-play: one network, two players

**Why one network can play both seats.** The game is symmetric: the rules do not care which side is A, and both
seats have the same skill. Because every observation is written from the observer's own side, "the situation I am in"
looks the same to the network whichever seat it sits in. The reward is zero-sum: seat A receives r, seat B receives
−r for the same turn. So "a good move for A in situation x" and "a good move for B in the mirrored situation x" are the
same decision, and one set of parameters θ can learn it from both seats' experience (`rl/shared.py`).

**Why the two players still act differently.** The network is a function of its input. The two seats receive
different inputs whenever the game is not symmetric (different fronts, HP, levels, gold, positions), so they get
different probability vectors. In the worked turn A (full HP, weaker front) put 25% on Last Hit + Trade and 9% on
Thin + Trade; B (70% HP, stronger front) put 18%, 15% and 14% on Last Hit + Trade, Slow Push + Safe and Slow Push +
Trade. Even with identical inputs they would usually differ, because each draws its own random action.

**What is trained.** In M4.4 every training game is a *current-policy* game (`SharedConfig(current_fraction=1.0)`):
the current network plays both seats and **both seats' decisions are training samples**, so one game turn produces two
transitions. Snapshots of the network are saved every 25,600 transitions; they are used to evaluate progress (§6),
not as training opponents. (The trainer also supports OpenAI Five's 80/20 mixture with past versions; M4.3b found no
benefit, so M4.4 does not use it.)

---

## 4. Actor and critic

The network (`PolicyValueNet`) is a small multilayer perceptron with a shared trunk and two heads:

$$o\in[0,1]^{88}\ \xrightarrow{\ \tanh(W_1o+b_1)\ }\ h_1\in\mathbb R^{128}\ \xrightarrow{\ \tanh(W_2h_1+b_2)\ }\ h_2\in\mathbb R^{128},\qquad z=W_\pi h_2+b_\pi\in\mathbb R^{61},\quad V=w_v^\top h_2+b_v .$$

- The **actor** (policy head) gives the logits z, hence π_θ(a | o): what to do. Its initial weights are tiny
  (orthogonal init with gain 0.01), so an untrained network plays almost uniformly over the legal codes (initial
  entropy 2.85 nats ≈ ln 17).
- The **critic** (value head) gives V_θ(o): the expected discounted sum of this side's future rewards from here,
  "how far ahead am I, in gold, from now on". It is trained in scaled units (rewards × 0.1); divide by 0.1 to read
  resources. In the worked turn V(A) = +1.91 and V(B) = −1.13 resources. For an exact critic of a zero-sum game these
  would be equal and opposite; the gap is the critic's error after 50 updates.

**The objective** of the game is the discounted gold lead (reference §9.1):

$$J=\mathbb E\Bigl[\sum_{k\ge0}\gamma^k r_k\Bigr],\qquad \gamma=0.99 .$$

γ belongs to the game: it says the laning phase ends after a random number of turns with mean 100. Training with any
other γ is refused (`check_time_semantics`).

**Advantage: was the action better than expected?** The temporal-difference error compares what happened in one
turn with what the critic predicted:

$$\delta_t=r_t+\gamma V(o_{t+1})-V(o_t).$$

Positive δ: the turn went better than the critic expected. Generalized advantage estimation (GAE, λ = 0.95) averages
these errors over the following turns with decaying weights, trading a little bias for much less noise:

$$A_t=\delta_t+\gamma\lambda\,\delta_{t+1}+(\gamma\lambda)^2\delta_{t+2}+\dots,\qquad R_t=A_t+V(o_t)\ \text{(the critic's target)}.$$

The sum stops at the end of a game; at a cut (§5) the next value is still bootstrapped (`compute_gae`).

**The PPO update** (`ppo.ppo_update`). With ρ_t = π_θ(a_t | o_t) / π_old(a_t | o_t), the probability ratio between
the network being trained and the one that collected the data:

$$L=\underbrace{-\,\mathbb E\bigl[\min(\rho_tA_t,\ \operatorname{clip}(\rho_t,0.8,1.2)A_t)\bigr]}_{\text{actor: make good actions likelier, but not too fast}}
+0.5\,\underbrace{\mathbb E\bigl[\max\bigl((V-R_t)^2,(V^{\text{clip}}-R_t)^2\bigr)\bigr]/2}_{\text{critic: predict what happened}}
-0.03\,\underbrace{\mathbb E[H(\pi(\cdot\mid o_t))]}_{\text{keep exploring}} .$$

In words: if an action had a positive advantage, raise its probability; negative, lower it. The clip stops any one
batch from changing a probability by more than about 20%, so the network does not over-react to noisy estimates. The
critic is pulled towards the realized returns (also with a clip, ±0.5 in scaled units, around its old prediction).
The entropy bonus (coefficient 0.03, the M4.3b `ent03` setting) rewards keeping the distribution spread, so the
policy keeps trying alternatives. Advantages are normalized per minibatch; decisions with a single legal code (an
absent champion) carry no policy gradient but still train the critic. Gradients are clipped to norm 0.5; Adam with
learning rate 3·10⁻⁴, decreased linearly to 0 over the run.

---

## 5. The learning loop, step by step

**Many games at once.** `SharedCollector` runs 16 independent environments in lock-step. On each step it stacks the
observations of all 32 seats, runs **one** batched forward pass of the network, samples 32 actions under their masks
and steps the 16 games. Every environment has its own random streams (chance events, seeded per game), so a run is
bit-for-bit reproducible from its seed (verified in M4.4a for all three environments).

**Games and segments.** The game never ends by itself. A training game is cut after 512 turns (a *truncation*: the
critic's value of the final observation stands in for the rest, it is not treated as the end of the world), and the
first game of each environment gets a random length between 1 and 512 so that the 16 games are at different ages. Data
is collected in segments of 128 turns per environment; a segment that cuts a game bootstraps from the current value.
One update therefore uses 16 × 128 × 2 = **4,096 transitions**, and the M4.4b budget of 2,048,000 transitions is
**500 updates**.

**One turn of one game (the worked example, `aggro`, turn 13).**

1. *Observations.* Fronts fight at the lane cell (A 7 HP, B 13 HP, plus a fresh wave each at its own −2); both
   champions at their own −1; A at full HP and level 3, B at 70% and level 4. A's and B's 88-vectors are the two
   columns of the table in §1.
2. *Network.* One forward pass gives each seat 61 logits and a value. After masking, A's distribution is the table in
   §2 and B's puts 0.18 on Last Hit + Trade, 0.15 on Slow Push + Safe, 0.14 on Slow Push + Trade, ...
3. *Simultaneous choice.* A draws code 8, **Thin + Trade** (probability 0.093, log-probability −2.376); B draws code
   49, **Slow Push + Safe**. Neither sees the other's choice.
4. *Resolution* (`champions_m4.resolve_turn_m4`). A's Thin budget is 3 HP (B's wave is the stronger, so Thin removes
   the excess); B's Slow Push budget is 1 HP. A walks forward to trade; B's Safe keeps it out of reach (no trade
   connects). A's champion ends at the lane cell, B at its tower. A collects 0.675 resources of farm, B none.
5. *Reward.* r = +0.675 for A, −0.675 for B (zero-sum, exactly the gold-ledger difference). Training uses
   r × 0.1 = +0.0675 / −0.0675.
6. *Stored experience.* Two rows, one per seat: (observation, legal mask, action, log-probability under the
   collecting network, value, scaled reward, terminated = False, truncated = False). Seat A's row is
   (o_A, mask_A, 8, −2.376, 0.191, +0.0675, False, False).
7. *Advantages.* When the segment is complete, GAE runs backwards over every seat's row sequence. For seat A over
   turns 13–16 (scaled rewards +0.0675, +0.04125, −0.05625, −0.0225; values 0.191, 0.113, 0.083, 0.156; value after
   turn 16: 0.154):

   | turn | δ_t = r + 0.99 V(next) − V | A_t (GAE) | R_t = A_t + V |
   |---|---|---|---|
   | 13 | 0.0675 + 0.99·0.113 − 0.191 = −0.011 | −0.010 | 0.181 |
   | 14 | +0.010 | +0.002 | 0.115 |
   | 15 | +0.015 | −0.009 | 0.074 |
   | 16 | −0.026 | −0.026 | 0.130 |

   (A_t here is truncated after turn 16 for illustration; in training the sum runs to the end of the segment.)
   Turn 13's farm was slightly *less* than the critic had already priced in, so Thin + Trade there gets a small
   negative advantage: after the update it becomes a little less likely in situations like this one.
8. *Update.* After all 16 environments have advanced 128 turns, the 4,096 rows are shuffled and used for 4 epochs of
   minibatches of 256 (64 gradient steps). Then collection resumes with the new parameters.

Every 25,600 transitions (about every 6 updates) the current network is saved as a snapshot; the final network and
every snapshot carry a SHA-256 digest of their weights that is checked when they are loaded.

---

## 6. Training versus evaluation

| what | what it measures | where |
|---|---|---|
| training reward | nothing about progress in self-play (see below) | `history.jsonl` |
| PPO health | finite losses, entropy, KL, explained variance, gradient norms, critic scale | `history.jsonl`, `train_*.json` checks |
| value against scripted references | strength against fixed, interpretable strategies (`ref:farm4`, `ref:shove4`, ...) | `evaluate.py` (paired seats, window 1000, ± 95%) |
| value against earlier snapshots and the initial network | improvement over its own past (historical policies) | `evaluate.jobs_history`, learning curves |
| cross-play between seeds | whether independent runs found compatible strategies | `evaluate.jobs_crossplay` |
| state-dependent behaviour | what the policy does in which situation, and what that produces (freezes, crashes, denial) | `records.py`, `metrics.py` |
| counterfactual advantage | whether a different choice at a reached state would have paid, under stated continuation policies | `counterfactual.py` |

**Why the self-play reward is near zero even when both players improve.** In a current-policy game both seats are
the same network, s0 is symmetric and the reward is zero-sum. Whatever the network has learned, seat A's expected
return equals seat B's, and they sum to zero, so both are zero. If the network gets better at trading, it also gets
better at defending against trades; the gold gap between two copies of itself does not grow. Progress has to be
measured against something that does not improve with it: fixed references, frozen snapshots, the initial network.
(In the M4.4a smoke run, the 204,800-transition finals beat their own initial networks by +5.7 to +10.0 resources per
game while their self-play return stayed at 0.)

**Why strength is not strategy.** Beating a scripted reference shows the policy exploits that reference. It does not
show the policy freezes, zones or slow-pushes: those are read from the behaviour diagnostics (state-based definitions:
a freeze is a lane state that persisted, not a chosen action) and tested with counterfactuals.

---

## 7. Known limitations

- **Observation.** The game has perfect information (both champions' gold, HP, positions and the whole lane are
  visible), so no belief or memory is needed; but the network has no history either: it cannot condition on what the
  opponent did a few turns ago (no opponent modelling, no "they always shove after recalling"). The spawn phase is
  the only clock.
- **Policy.** A 2 × 128 MLP trained for 2M transitions. It is stochastic by design and its entropy stays above 1 nat
  in the four-action presets at the M4.4 budget, so some action mixing is just residual exploration. Rare situations
  (large power gaps occur on under 1% of self-play turns) are seen too seldom to be learned well; the initialized
  scenarios of M4.4 measure behaviour there but do not train on them.
- **Horizon.** γ = 0.99 weighs the next ~100 turns. Training games are cut at 512 turns and bootstrapped; with an
  imperfect critic the bootstrapped targets are biased (`ppo.py` docstring). In self-play the plates are gone by turn
  ~140–150 and levels cap at 18 around turn 145, so most training data comes from the late game.
- **Reward.** Gold only (farm, plates, kill bounty). XP is not rewarded directly (it buys power, which buys gold).
  Anything the reward does not count (e.g. pressure for its own sake) is learned only if it eventually pays in gold.
- **Self-play.** Self-play can cycle and has no exploitability guarantee: a policy can beat its own past and every
  reference and still lose to a strategy it never met. Three seeds per environment measure run-to-run variation,
  not convergence.
- **What PPO cannot establish.** A learned policy is a local optimum of a training procedure, not an equilibrium. A
  counterfactual contrast is the value of a deviation *against the given continuation policies*, not a best response.
  None of the numbers in M4.4 is a game-theoretic solution; exact equilibria exist only for the Phase 1 game.
