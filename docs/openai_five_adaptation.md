# Adapting OpenAI Five's self-play to the laning game

Source: Berner et al., *Dota 2 with Large Scale Deep Reinforcement Learning*, arXiv:1912.06680 (2019); section
references (§, App.) are to that paper. "Documented" means stated in the paper; "ours" means a choice made for
this project. The OpenAI blog post (2018) describes an earlier system and is cited only where marked.

## 1. Relevant ideas from OpenAI Five (documented)

| Topic | What the paper documents |
|---|---|
| Algorithm | PPO with GAE, λ = 0.95, clip 0.2, entropy 0.01 → 0.001 (§3.2, Table 2). Policy and value share a network (App. H). Rewards are normalized by a running estimate of their standard deviation (Table 2, note c). The paper does not mention advantage normalization, value clipping or a number of epochs; data reuse is controlled as "sample reuse" (§4.4). |
| Zero-sum rewards | "We ensure that all our rewards are zero-sum, by subtracting from each hero's reward the average of the enemies' rewards" (App. G). Team spirit τ mixes a hero's reward with the team mean (App. G, Eq. 12). |
| Self-play mixture | "play the latest policy against itself for 80% of games, and play against older policies for 20% of games" (§3.2; App. N). |
| Past-opponent sampling | An opponent manager keeps past versions with quality scores q_i, samples p_i ∝ exp(q_i); when the current agent beats version i, q_i ← q_i − η/(N p_i), η = 0.01; every 10 iterations the current agent joins the pool with the maximum existing quality (App. N, Eq. 14). |
| Why past opponents | "to obtain more robust strategies and avoid strategy collapse in which the agent forgets how to play against a wide variety of opponents because it only requires a narrow set of strategies to defeat its immediate past version" (App. N). Described as similar in spirit to fictitious play, but with the latest policy rather than an average policy (§5). |
| Versions / checkpoints | Every 32 gradient steps a new parameter version is published; a controller stores the metadata needed to stop and restart (§3.2). All past versions are kept; after model surgery they must be converted too (App. B). |
| Evaluation | TrueSkill against a fixed pool of 83 reference agents with frozen ratings, matches only within ~15–85% expected win rate (§4.1, App. J). A hand-scripted bot rates about 100–105. The paper measures skill by win rates against other (fixed) models (App. B, Fig. 6). |
| Long horizons | Single-layer 4096-unit LSTM (§3.1); discount expressed as a horizon and annealed (OpenAI Five 60 → 840 s; §4.5, Table 2); potential-style shaped rewards (App. G). |
| Stability / reproducibility | "Rerun": a fresh run with the final code beat the final OpenAI Five > 98% of games with ~20% of the compute (§4.2). Identical small runs vary by ~2 TrueSkill (App. M). Staleness of ~8 versions and sample reuse of 2–3 slow learning markedly (§4.4). No claim of optimal hyperparameters (App. C). |

Not stated in the paper (so not attributable to it): whether both teams' experience is used for training in
latest-vs-latest games, and whether the past opponent's side is discarded in past-opponent games.

## 2. What this project already implemented (before this study)

- Masked PPO with GAE (λ = 0.95), clipped surrogate and value loss, entropy 0.01, one own-perspective MLP for both
  seats (`lanegame/rl/`). Zero-sum rewards by construction of the environment (A receives r, B receives −r).
- M3b frozen-snapshot self-play: ONE learner seat per game against 30% scripted agents, 35% the latest snapshot,
  35% a uniform snapshot (pool of 8, the first kept as an anchor). Only the learner trains.
- Evaluation against fixed scripted references, cross-play, snapshots, and PPO approximate best responses
  (lower bounds on exploitability; exact exploitability exists only for the legacy configurations).

## 3. What was missing

- Current-policy self-play: no game in which the current network plays both sides.
- Using both players' experience from a game.
- An OpenAI Five-style current/historical mixture, accounting in trainable samples, and safe checkpoint handling.

## 4. What this study reproduces (ours, adapted)

| OpenAI Five | This study |
|---|---|
| 80% latest-vs-latest, 20% vs past versions | `current_fraction = 0.8` per training episode (configurable; 1.0 and 0.5 also run) |
| Past versions in an opponent manager | Bounded FIFO pool of the 8 most recent frozen snapshots, one every 25,600 agent transitions |
| Quality-score sampling | **Uniform** sampling over the pool (simpler, documented rule; the quality-score rule is deliberately not used) |
| Both teams' experience in mirror games? (not stated) | **Ours:** both seats' transitions are trained on in current-policy games (the game is symmetric and the network is shared, so each seat's own-perspective trajectory is an on-policy sample) |
| Past-opponent side (not stated) | **Ours:** only the current network's seat is trained on; the snapshot acts through its saved weights |
| Zero-sum shaping by subtracting enemy reward | Not needed: the environment reward is already zero-sum (r, −r); each seat optimizes its own reward |
| PPO on fresh data, low staleness | Each rollout is collected with one fixed parameter vector; no staleness, no reuse beyond PPO's 4 epochs |
| Evaluation against fixed references | Fixed scripted references (a training suite and a held-out set), a fixed learned reference, cross-play matrices, snapshot matrices, PPO best responses |
| Versioned checkpoints with metadata | Snapshot and final files with step counts (agent transitions and environment turns), seed, configurations, model spec, snapshot identity; run directories are never overwritten |

Bootstrap (ours): until the first snapshot exists every game is a current-policy game.

## 5. Deliberately excluded

LSTM and large networks (the observation is Markov and complete: 50 features, 32 turns); discount/horizon
annealing (γ = 1 on a 32-turn game with the turns remaining observed); reward shaping, reward normalization and
team spirit (one player per side, rewards already in resources); quality-score opponent sampling; model surgery;
distributed/asynchronous rollouts; TrueSkill (replaced by direct head-to-head payoffs with confidence intervals
against fixed references, which this two-player game makes cheap); league training, PSRO, search.

## 6. Differences between the games that matter

- **Scale and horizon.** Dota: ~20,000 decisions per game, partial observability, 5 v 5. Here: 32 simultaneous
  turns, full observability, 1 v 1, 46 actions. Credit assignment is not the bottleneck; strategic coverage is.
- **Stochasticity.** Trades, tower shots and ganks are random; reward noise is large relative to strategic edges
  (M3a: PPO is a loose best responder on the exact legacy game).
- **Mixed equilibria.** A simultaneous-move zero-sum game generally needs randomized play; on the legacy game,
  frozen-snapshot self-play did not approach the mixed equilibrium (M3b). Self-play against the latest policy is
  iterated best response in spirit and can cycle; OpenAI Five's past-opponent games are the mitigation under study.
- **Measurement.** In current-policy self-play the two seats' returns cancel exactly; win rate against the current
  self is uninformative, so progress is measured only against fixed opponents and by approximate best responses.
- **Mechanical skill.** Execution skill is an observed parameter here (trade hit chances), with no Dota analogue
  in the paper's setting.
