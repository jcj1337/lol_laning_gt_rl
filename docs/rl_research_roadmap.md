# RL Research Roadmap: From Exact Stochastic Games to Multi-Agent Reinforcement Learning

*Status: planning document, October 2026. Nothing in this document has been implemented. It was written after inspecting the engine (`lanegame/`), the three Phase 1 reports, the test suite (305 tests) and the siege-study results, and after a short throughput benchmark of the existing engine (section I). Every cost figure that is not marked "measured" is a derived expectation, not a promise. Terminology follows the siege study: the midpoint cell x = 0 is "the lane"; coordinates are own-perspective unless stated (own tower at −T = −2, the lane at 0, enemy tower at +T = +2).*

---

## A. Executive recommendation

> **Design standard (added 2026-10-12, M4.2e-B):** every mechanic proposed below and in later sections is judged against [design_philosophy.md](design_philosophy.md): simplify how League works without changing why its laning strategies work.

**Keep the exact game as the benchmark, add a champion layer beside it, and train against it with PPO self-play.** Concretely:

1. **Do not fork the simulator.** The deterministic lane dynamics (`mechanics.step_turn`), the legality rules, the farming economy, the gank-probability formula and the mirror symmetry are reusable as they stand. The RL environment should wrap them, not copy them. One optional argument to `step_turn` (a per-side champion reach cap) is the only change the existing mechanics need, and it defaults to current behaviour.
2. **Put the new mechanics in a separate champion layer** (`lanegame/champions.py`) that runs after the lane step inside one turn: persistent champion position, HP, the four behaviours (Farm-Hold, Farm-Yield, Trade, Safe), Recall, stochastic trades, ganks as events with HP consequences, death and absence. The layer is switched on by one configuration flag that defaults to off, so `GameConfig()`, `GameConfig.v2()` and `GameConfig.v1()` keep reproducing their reports bit for bit.
3. **Expose both games through one simultaneous-move environment** (`lanegame/env/`): `reset()`, `step(action_a, action_b)`, own-perspective observations that are pure functions of (turn, state), legal-action masks, zero-sum per-turn rewards, deterministic seeding with separate random streams for chance events and for agents. A PettingZoo `ParallelEnv` adapter is a thin optional wrapper over it, not the core.
4. **Make the exact solver the regression oracle for learning.** Any agent whose decision is a function of the observation is a Markov policy of the exact game, so on the legacy configurations (K ≤ 16, under six seconds to solve) its exploitability, its exact value against any scripted opponent, and the gap between what it learned and the exact best response can be computed to machine precision. That is how the trainer is validated before it is pointed at the champion game, where no exact answer exists.
5. **Train in stages:** PPO with action masking and one network for both seats (the mirror symmetry makes this free), first against fixed scripted opponents on the exact game, then in self-play with an opponent pool on the champion game, then with approximate best responses as the exploitability measure, and only then PSRO, for which the existing `solve_matrix_game` is already the meta-solver.
6. **Keep every milestone small and measured.** No new milestone performs a K = 32 exact solve (the existing `results/siege/solution_K32.json` is the frozen artifact for that), tests that need an exact solve use K ≤ 16, throughput is benchmarked before any training is scheduled, and research sweeps are separated from correctness tests.

**Why this makes sense.** The siege study showed where exact enumeration stops paying: the vision timer alone multiplied the state space twelvefold and one more persistent counter (the per-tower plate budget) multiplied it by a further 3.5 before anything was solved. Every mechanic the research question now needs (HP, position, absence timers, levels, items, plate durability) is exactly that kind of persistent counter, so exhaustive enumeration of the expanded game is not a realistic goal. At the same time, the question itself ("when to exploit, how much risk to accept when the model may be wrong") is a question about policies, which RL answers directly and which the exact game can still arbitrate on small instances. The repository is unusually well prepared for this: the state is already own-perspective and mirror-symmetric, the policies are already probability vectors over legal sets, the Monte Carlo simulator already samples with a seeded generator, and the engine already runs about 25,000 lane turns per second in pure Python (measured), which is enough for the first training experiments without any vectorisation work.

---

## B. Current architecture assessment

### B.1 What exists and what it is worth for the transition

| Module | What it does today | Verdict for the RL transition |
|---|---|---|
| `config.py` `GameConfig` | Frozen dataclass with every parameter, presets `v1()` / `v2()`, `with_()`, JSON round trip, `validate()` | **Reuse and extend.** New champion fields default to "off"; presets stay unchanged. Per-seat parameters (mechanical skill) are a new kind of field: the first asymmetric ones. |
| `state.py` `State` | Own-perspective per-side records, `mirror()`, `side(s)`, list round trip | **Reuse unchanged for the lane.** The champion layer adds its own per-side record; the environment state is a composite (lane `State` + two champion records + k). Keeping `State` untouched keeps every exact code path bit-identical. |
| `mechanics.py` `step_turn`, `legal_actions`, `TurnResult` | Deterministic tick loop, merges, fights, thinning, crashes, tower fire, spawns, plates, vision; kills keyed by position for farming; exposure | **Reuse as the shared simulator core.** Needs one optional argument (per-side champion reach cap, default = today's `posture_cap` table). `TurnResult.kills` is exactly what XP (M4) needs. `legal_actions` is the mask generator for the legacy action space. |
| `game.py` `LaneGame`, `gank_probability`, `collection_efficiency`, `turn_farm` | Cached transitions, expected gank payoff, farm collection, reachable layers | **Reuse the functions, leave the class alone.** `gank_probability` becomes the probability that a gank *occurs*; `collection_efficiency` is reused with a behaviour-to-posture mapping. `LaneGame` stays the exact game's adapter. Its unbounded transition cache and interning dictionary are right for enumeration and wrong for long RL rollouts (see B.3). |
| `matrix_game.py` | Exact zero-sum matrix games with certificates; support enumeration | **Reuse.** It is the PSRO meta-solver and the per-state check for exact exploitability. |
| `solver.py`, `best_response.py`, `analysis.py` | Backward induction, exact best responses and exploitability for any `Policy`, exact forward statistics | **Reuse as the benchmark.** They assume one lane successor per joint action (plus the player-chosen crash options). A chance-node extension is a modest, optional change (M2e) and must be guarded by the pinned v1 / v2 / siege values. |
| `policy.py` `Policy`, `TabularPolicy`, `ScriptedPolicy`, `own_view` | `move(k, state, side, legal)` returns a distribution; own-view rule for scripted policies | **Reuse.** The `Agent` of the RL environment and the `Policy` of the exact game are two views of the same object; the adapters in both directions are a few lines each. |
| `baselines.py` | Twelve scripted opponents (siege tendencies included), `prefer` / `resolve` helpers | **Reuse as the scripted-opponent suite** for M1 and as the first opponent *types* for M5. M2 adds champion-behaviour baselines. |
| `simulate.py` | Seeded Monte Carlo over `Policy` pairs, realized ganks, expected farm | **Superseded for RL by the environment, kept for the exact models.** The environment must agree with it statistically (test) but need not consume random numbers in the same order. |
| `io.py` | JSON solution schema version 3 | **Pattern to copy** for the episode-trace schema the dashboard will read. The 68 MB `solution_K32.json` is the frozen K = 32 equilibrium. |
| `reachability.py`, `scenarios.py`, `render.py` | Reachable events with replayable witnesses, 23 traced scenarios | **Reuse as the validation method** for every new rule: each champion mechanic gets scenarios and reachable-event witnesses before any training. |
| `experiments/`, `experiments/v1/`, `experiments/siege/` | The three studies | **Untouched.** New research code goes in new sub-packages. |
| `tests/conftest.py` | Session-scoped K = 16 solves of three models | **Reuse.** The fixtures are the exact oracle for the environment tests. The suite takes about five minutes because three K = 16 solves and many exhaustive transition sweeps run every time; a `slow` marker (M0) keeps new-module development fast. |
| `legacy/` | The first Q-learning prototype: an opponent pool with periodic greedy snapshots | Reference only. Its training loop is the ancestor of the pool-based self-play proposed in F. |

### B.2 Measured costs (this machine, single-threaded Python 3.13, 12 logical CPUs)

| Quantity | Measured value | Source |
|---|---|---|
| `step_turn` + `legal_actions`, uncached, siege model | about 25,000 turns/s (40 µs per turn) | benchmark run for this document |
| Full K = 32 episode through `simulate`, scripted policies | 475 to 750 episodes/s (15,000 to 24,000 turns/s) | same run |
| Exact solve, siege model, K = 8 / 12 / 16 | 0.2 s / 1.2 s / 5.6 s (189 / 1,039 / 4,951 nodes) | same run |
| Exact solve, siege model, K = 24 / 32 | about 30 s / 3.5 min, 616 MB / 3.05 GB | `results/siege/benchmark.md` |
| Exact solve, previous revision (`v2`), K = 32 | under 10 s, 116 MB | `results/revision/horizon_benchmark.md` |
| Full test suite | 305 tests, about 5 min | README, siege study section I |
| Regenerating the siege study | about 2 h 55 min | siege study section J |

The lane simulator is therefore not the bottleneck for early RL. One million environment turns cost roughly 40 to 70 s of simulator time at the measured rate; the champion layer will add work (two more random draws and a position update per side per turn) that must be measured in M2e, and policy-network inference will dominate in any case.

### B.3 Integration points

- **Turn structure.** `step_turn(state, k, a, b, cfg)` is a pure function with a complete event record. The champion layer can be inserted *around* it: resolve champion intentions before the lane step (to know how far forward each champion acts on its wave), run the lane step, then resolve trades and ganks from the positions the turn produced.
- **Actions as integers plus a legal set.** Every policy already works with "a probability vector aligned with the legal tuple". A mask over a fixed action space is the same information in a fixed shape.
- **Own-perspective state.** `State.mirror()` and `own_view` mean one policy can play either seat. For a neural policy this halves the data needed and removes seat-specific bugs. Mechanical skill breaks the *symmetry* but not the *equivariance*: the observation just has to include own and enemy skill.
- **Markov observations.** If the observation is a function of (k, state, side) only, any agent is a `Policy` of the exact game. That is the bridge to `evaluate`, `best_response_value`, `exploitability` and `path_statistics`.
- **Event accounting.** `TurnResult.kills[s]` (HP of enemy minions that died, keyed by own-perspective cell) is the natural input for proximity XP; `TurnResult.exposure`, `plates`, `thinned`, `vision_set` are the natural per-turn statistics for the dashboard.

### B.4 Limitations and risks for the transition

1. **The solver assumes a deterministic lane.** `build_q` reads one `pre_state` per joint action. Ganks are expected losses, not events. Exact solving of any configuration with stochastic HP damage needs chance nodes (an expectation over successors in `Q`, in `induct` and in `path_statistics`). Small change, but it touches the certified core; it must be guarded by the pinned values.
2. **Action encoding is coupled to posture.** `a = wave + 3·posture + 6·objective` with `posture ∈ {Contest, Safe}`. The four champion behaviours (plus Recall and an "absent" no-op) need a new encoding; the old one must remain valid for the old configurations.
3. **Exposure is derived from the wave.** The champion "is where its wave goes" (`reach`), capped by posture. Persistent position makes exposure a state variable instead. The gank formula is unaffected (it takes an exposure integer); its *input* changes.
4. **Caches grow without bound.** `LaneGame._cache` and `_intern` are right for enumeration (2.7 M entries, 3 GB at K = 32) and wrong for a long-running environment with stochastic successors. The environment should cache only the deterministic lane step, keyed by (lane state, phase, lane-relevant action components), with a size bound.
5. **One random stream.** `simulate.play_episode` draws actions, ganks and choices from one generator. For reproducible research the environment needs separate streams: chance events (trades, ganks) and agents must be seedable independently, so that two policies can be compared on identical chance draws (common random numbers).
6. **Expected gank penalty inside the payoff.** `Transition.payoff` includes `L·(p_B − p_A)`. In champion mode that term must disappear (the cost of a gank is HP and absence) without changing the legacy payoff.
7. **Symmetry assumptions in tests.** Several invariant tests (`test_mirror_equivariance_of_transitions`, `test_value_antisymmetry`) assume the two players are identical. With per-seat skill they hold only when the skills are equal; the tests must be parametrised accordingly.
8. **Full information, no history.** The exact game is perfect-information and Markov. Opponent modelling (M5) needs public action history as a *feature*; the environment must expose it in `info` without putting it into the Markov observation used for the exact bridge.
9. **State-space growth is immediate.** Per side, 5 positions × 7 HP values × 4 absence values is a factor of 140; squared, 19,600 on top of the lane states. Exact enumeration of champion configurations is feasible only at tiny horizons (K ≤ 8) with reduced ranges (section D, M2e). This is expected and is the reason for the transition.
10. **Test-suite time.** Five minutes per run is already a tax on every change; the new modules must not add solves to the default run.

---

## C. Proposed target architecture

### C.1 Layers

```
                       ┌──────────────────────────────────────────────────────┐
                       │  research utilities                                   │
                       │  experiments/rl/   opponent types, posteriors,        │
                       │  tournaments, frontier sweeps, trace export (M3–M6)   │
                       └───────────────▲──────────────────────────▲───────────┘
                                       │                          │
     ┌─────────────────────────────────┴───────┐     ┌────────────┴─────────────────────────┐
     │  exact benchmark  (unchanged modules)   │     │  learning  lanegame/rl/  (M3)         │
     │  solver.py  best_response.py            │     │  PPO (masked, own-view shared net),   │
     │  analysis.py  matrix_game.py            │     │  opponent pool, approximate BR, PSRO  │
     │  LaneGame (game.py)                     │     │  Agent protocol; Policy <-> Agent     │
     └───────────────▲───────────────▲─────────┘     └────────────▲──────────────────────────┘
                     │               │  Policy wrapper (Markov obs)  │
                     │               └───────────────┬──────────────┘
                     │                               │
     ┌───────────────┴───────────────────────────────┴───────────────────────────┐
     │  environment  lanegame/env/  (M1)                                          │
     │  LaneEnv: reset / step(a, b) / masks / obs / rewards / seeding / trace      │
     │  observation.py (own-view, Markov), agents.py (scripted), pettingzoo.py    │
     └───────────────▲───────────────────────────────▲───────────────────────────┘
                     │                               │
     ┌───────────────┴───────────────┐   ┌───────────┴──────────────────────────┐
     │  lane mechanics (existing)    │   │  champion layer  lanegame/champions.py │
     │  mechanics.step_turn          │◄──┤  position, HP, behaviours, trades,     │
     │  legal_actions, farming,      │   │  ganks as events, recall, death,       │
     │  gank_probability, plates,    │   │  absence; later XP, items, plate HP    │
     │  vision                       │   │  (M2, M4)                              │
     └───────────────────────────────┘   └───────────────────────────────────────┘
                     ▲                               ▲
                     └──────────── GameConfig ───────┘   (one config; champion fields default off)
```

- **Shared simulator** = `mechanics.step_turn` plus the champion layer. The exact benchmark uses the lane mechanics directly (as today) and, optionally, the champion layer on tiny configurations through a chance-node game class. The RL environment uses both.
- **Exact benchmark** = the existing modules, untouched except for the optional chance-node generalisation.
- **Environment** = the only thing agents and trainers talk to. It depends on NumPy only.
- **Learning** = PyTorch code that depends on the environment and never on the solver.
- **Research utilities** = experiment scripts that may use everything.

### C.2 Answers to the fifteen architectural questions

1. **Build on `LaneGame` or add a layer?** Add a layer. `LaneGame` stays the exact game. The environment calls `step_turn` and the champion layer directly, with its own bounded cache. Reason: `LaneGame`'s contract (deterministic successor, expected gank payoff, unbounded cache) is exactly what the exact solver wants and exactly what an RL environment does not.
2. **Exact solver without constraining RL mechanics?** Three mechanisms: (a) configuration flags that default to the legacy behaviour, pinned by the existing reproduction tests; (b) the Markov-observation bridge, so any learned agent is exactly evaluable on legacy configurations; (c) an optional chance-node game class for tiny champion configurations, gated by a reachable-node budget. The solver never has to handle the full champion game.
3. **PettingZoo Parallel API, custom interface, or both?** Both, in that order of importance: a custom simultaneous-move core (`LaneEnv.step(a, b)`), plus a thin `ParallelEnv` adapter behind an optional import. Trade-offs: PettingZoo's Parallel API is already simultaneous and gives tooling compatibility, but it imposes string agent ids, Gymnasium space objects, a dependency, and dictionary plumbing that the exact bridge and the tests do not need. The adapter is about a hundred lines and is tested with `pettingzoo.test.parallel_api_test` when the package is installed.
4. **Observation structure?** Own-perspective, fixed-size, Markov. Scalars (turn fraction, turns remaining, phase one-hot), a 9-cell lane grid with own and enemy minion HP per cell, per-side siege mass, vision timer, pending plates and siege eligibility; in champion mode also HP, position one-hot, absence timer, level, items, skill for both sides. Both a flat vector (for networks) and the structured state (for scripted agents and analysis) are returned. Normalised by configuration ranges.
5. **Action combinations and masks?** One flat `Discrete` index over the product of components, with a boolean legality mask from the engine. Legacy mode: the existing 18 codes. Champion mode: wave (3) × behaviour (5: Hold, Yield, Trade, Safe, Recall) × objective (3) = 45 codes plus one `ABSENT` no-op, 46 in total; typically 4 to 9 are legal. Masking by setting illegal logits to −∞. Component factorisation of the policy head is deferred unless the flat head learns poorly.
6. **Persistent champion position?** An own-perspective cell `c ∈ {−T, …, +T}` (five values), moved at most `champion_speed` cells per turn toward a behaviour-dependent target; exposure for ganks is the maximum of the start and end cells. Details in E.4. The three-level abstraction (Defensive / Wave / Forward) is the fallback if five cells prove too fine.
7. **Reproducible stochastic trading and ganks?** All chance in the champion layer is drawn from one `numpy.random.Generator` owned by the environment and seeded at `reset`, separate from any agent randomness; draws happen in a fixed order (A's trade, B's trade, A's gank, B's gank, damage). Same seed, same policies, same trajectory; same seed, different policies, same chance draws until the first divergent action (common random numbers).
8. **Recall, death, absence?** One per-side integer `absent` (turns still away). Recall sets it to `recall_turns` (2) and is legal only near one's own tower; death sets it to `death_turns` (3). While absent the only legal action is `ABSENT`, the wave follows a configurable default wave action, farm collection and zoning are zero, the champion returns at the own tower cell with full HP. Details in E.8.
9. **What should the reward optimise?** The per-turn zero-sum relative resource gain: (farm + plates + bounty) of A minus the same of B, realized (not expected), undiscounted, terminal value 0. No gank penalty term in champion mode; the cost of being ganked is HP and absence. Optional shaping knobs (death penalty, terminal HP value) exist in the configuration and default to zero.
10. **Zero-sum with HP, items, XP, tower damage?** By construction: a single scalar `r` per turn, A receives `r`, B receives `−r`, and HP, XP, levels, items and plate damage are *state*, never reward. Gold spent on items is not a loss (it was already counted when earned) and item power is not a gain (its value is what it earns later). Plate rewards are paid when thresholds break, once. Terminal valuation of unspent gold, items or HP is a configuration knob defaulting to zero; if ever switched on it must be applied symmetrically so the game stays zero-sum.
11. **Essential before RL training?** M1 (interface) and M2 (HP, position, behaviours, trading, skill, ganks as events, recall, death). These are the mechanics the research question is about; training on anything less answers a different question.
12. **Safe to add afterwards?** XP and levels, spendable gold and items, plate durability, distance-based farming, cannon waves, passive regeneration, partial observability. All are state additions that extend the observation and the mask without changing the step contract.
13. **Most practical first baseline?** PPO with action masking, one network for both seats via the own-view observation, Generalised Advantage Estimation, a small MLP, CPU training, implemented in the style of a single-file CleanRL script rather than through a multi-agent framework. Validate it against fixed scripted opponents on the exact game before any self-play.
14. **Validation against the exact benchmark?** Wrap the trained network as a `Policy` (deterministic in the observation), then: exact value against each scripted opponent (`evaluate`) versus the Monte Carlo estimate; exact best-response value (`best_response_value`) versus what PPO reached when trained against that opponent; exact exploitability of the self-play policy on K ≤ 16; convergence of exploitability over training. All on legacy configurations and, if M2e lands, on tiny champion configurations.
15. **Opponent modelling later without rewriting agents?** The agent sees an `Observation` built by a pluggable builder; belief features are an optional appended block (zeros until M5). Opponent models are `Policy` objects (the existing interface), so likelihoods are exact action probabilities at (k, state). The environment exposes the public joint-action history in `info` for posterior updates and never exposes the opponent's current action. Adaptive agents are ordinary agents with internal state.

---

## D. Implementation milestones

Each milestone is meant to be one bounded Claude Code task with its own tests and its own definition of done. Sizes are given as sessions (S: a single short session; M: one to two sessions; L: two to three sessions) rather than hours, because the uncertainty is in review and design iteration, not typing.

### M0. Test-speed hygiene (optional, S)

- **Scope.** Add a `slow` pytest marker to tests that need a K = 16 solve or an exhaustive transition sweep; add `python -m pytest -m "not slow"` as the development loop; document it in the README. No behaviour change.
- **Dependencies.** None.
- **Files.** `tests/conftest.py`, `pyproject.toml` (`markers`), README.
- **Definition of done.** The fast subset runs in well under a minute; the full suite is unchanged (305 pass).
- **Cost.** Development only.
- **Risk.** Mislabelling a fast test as slow; harmless.
- **Defer.** Any test restructuring beyond markers.

### M1. RL-ready environment around the existing game (M)

Specified in full in section J. Summary: `lanegame/env/` with `LaneEnv`, Markov own-view observations, masks over the 18 legacy codes, zero-sum realized rewards, seeded chance stream, scripted `Agent`s playing complete episodes, Policy/Agent adapters, an optional PettingZoo adapter, an `env-demo` CLI with a throughput print, and tests that tie the environment to the exact solver.

- **Dependencies.** None (M0 recommended).
- **Files.** New: `lanegame/env/{__init__,core,observation,agents,trace,pettingzoo_adapter}.py`, `tests/test_env.py`. Touched: `lanegame/__main__.py` (one subcommand), `pyproject.toml` (optional extra), README (layout). Not touched: `mechanics.py`, `game.py`, `solver.py`, `state.py`, `config.py`.
- **Definition of done.** Two scripted agents play seeded, reproducible K = 32 episodes; environment expectations match `evaluate` on the K = 16 fixtures; with ganks off every episode equals the exact value; an agent wrapped as a `Policy` has an exact exploitability; the PettingZoo API test passes when the package is installed; full suite still passes.
- **Cost.** Simulator: unchanged (measured 475 to 750 episodes/s through the old simulator; the environment should be within a factor of two). Tests: seconds beyond the existing fixtures.
- **Risks.** Observation that is accidentally not a function of (k, state) (breaks the exact bridge); mask misaligned with action codes; reward sum drifting from `simulate`.
- **Defer.** Champion mechanics, vectorised environments, Gymnasium spaces beyond the adapter, any learning.

### M2. Champion HP, trading and recall (L in total, delivered as five S/M tasks)

Specification in section E. The split below orders the work so that nothing stochastic is added before the deterministic skeleton is tested, and nothing can kill a champion before absence exists.

**M2a. Champion state, persistent position, behaviours (M).**
- Scope: `champions.py` with the per-side champion record (position, HP, absence, skill), the 46-code champion action space, legality, behaviour-to-target movement, exposure as max(start, end), behaviour-to-posture mapping for farming, the optional `reach_cap` argument to `step_turn`; `GameConfig` champion fields (all default off) and a `GameConfig.champions()` preset; the environment running the champion game; champion scenarios (`scenarios.py` set 4) and events (`reachability.py`). HP is present but nothing damages it yet.
- Dependencies: M1.
- Files: new `lanegame/champions.py`; touched `config.py`, `mechanics.py` (one optional parameter), `env/core.py`, `env/observation.py`, `scenarios.py`, `reachability.py`, `baselines.py` (champion-behaviour baselines), new `tests/test_champions.py`.
- Done: legacy configs reproduce (fixtures unchanged); movement limits, legality, exposure rule and farming mapping each have a test; mirror equivariance holds with equal skills; two scripted champion agents complete episodes.
- Risks: `step_turn` regression (guarded by the three pinned models); legality corner cases (Recall near the tower, objectives during absence).
- Defer: damage, ganks as events, death.

**M2b. Recall and absence (S).**
- Scope: Recall legality and effect, the absence timer, the absent default wave action, zero farming and zoning while absent, return with full HP at the own tower. Deterministic.
- Dependencies: M2a.
- Done: a recalling champion is away exactly `recall_turns`; the opponent's farm rises while it is away (scenario); mask is `{ABSENT}` while absent; Recall is illegal forward of the recall line.
- Defer: shopping (M4b).

**M2c. Stochastic trading and mechanical skill (M).**
- Scope: the trade resolution of E.5, the hit-chance formula with skill, the 0/1/2 damage distribution, Trade-versus-each-behaviour contingencies, the tower-shot rule, HP floor at zero *without* death (death comes in M2d; HP 0 is clamped to 1 behind a temporary flag, or deaths are simply disabled by configuration).
- Dependencies: M2b.
- Done: monotonicity tests (higher skill never lowers hit chance; Safe never raises the chance of being hit; out-of-range trades do nothing and fall back to farming); seed reproducibility; the distribution of damage over 10,000 draws matches the formula within binomial tolerance; champion scenarios for each of the five interactions of section 4.4 of the brief.
- Risks: a formula that is too strong or too weak makes Trade dominant or useless; keep every term configurable and report the default's numbers, do not tune for drama.
- Defer: level and item terms (present in the formula as zero-valued inputs).

**M2d. Ganks as events, death and respawn (M).**
- Scope: sample the gank from `gank_probability` (occurrence), then damage from the depth-and-behaviour distribution of E.7, apply HP loss, death when HP reaches zero, absence `death_turns`, return with full HP; remove the expected-loss term from the champion-mode reward; optional `kill_bounty` and `death_penalty` knobs defaulting to zero.
- Dependencies: M2c.
- Done: vision reduces only occurrence (test: damage distribution unchanged under vision); a full-HP champion cannot die to one gank at default parameters (test of the designed property); death leads to exactly `death_turns` of absence; the reward has no gank term; the zero-sum identity holds per turn.
- Defer: assists, bounties that scale, anything tower-related beyond the tower shot.

**M2e. Validation pass and exact micro-benchmark (M).**
- Scope: throughput benchmark of the champion environment; a full set of champion-mode scripted baselines (trade-happy, yield-farmer, safe-recaller, pressure-greedy); a tournament of scripted agents with confidence intervals (research sanity, not test); optional chance-node game class (`ChampionGame.outcomes`) plus solver / best-response / path-statistics generalisation, used only on a tiny configuration (K = 8, HP 3, three positions) under a reachable-node cap, with the three legacy models re-pinned.
- Dependencies: M2a to M2d.
- Done: benchmark numbers recorded in `results/env/benchmark.md`; every scripted pairing completes 1,000 seeded episodes; if the chance-node class is built, backward induction equals tree recursion on the tiny configuration and the legacy values are unchanged.
- Risks: the chance-node refactor is the only step that touches certified code; if the budget is tight, skip it and rely on the legacy-configuration bridge.
- Defer: any tuning sweep larger than a handful of settings.

### M3. Initial RL training (L, delivered as four tasks)

Specification in section F.

**M3a. PPO against fixed scripted opponents on the exact game (M).** `lanegame/rl/` with a masked PPO (own-view shared network), a batch of environments, checkpoints and seeds. Train against `passive_safe`, `aggressive` and `uniform` on the siege model at K = 16; compare the learned value to the exact best-response value and the exact value of the learned policy (via the wrapper). Done when PPO reaches a stated fraction (to be measured; aim high, but record what it is) of the exact best-response value against each fixed opponent within a budget recorded in the results. This is where the first real training-cost numbers come from.

**M3b. Self-play with an opponent pool on the champion game (M).** Same trainer; opponents sampled from recent checkpoints and the scripted suite; K = 32; symmetric skills first. Done when training is reproducible from seed, the learning curves and the metric suite (F.3) are written to `results/rl/`, and a frozen "self-play v0" checkpoint exists.

**M3c. Evaluation harness (M).** Tournament against the scripted suite with confidence intervals; approximate exploitability = value of a PPO best response trained against the frozen policy for a fixed budget; exact exploitability on legacy configurations; common-random-number pairing. Done when the harness runs from one command on a checkpoint and produces a Markdown and JSON report.

**M3d. PSRO (optional, M).** Oracle = M3a's trainer against a mixture; meta-solver = `solve_matrix_game` on the empirical payoff matrix; symmetric single population. Only after M3b and M3c are reproducible. Done when the meta-game exploitability trends down over iterations on the champion game at K = 16 or 24.

### M4. Economic progression (L, delivered as three tasks)

> **Superseded (2026-10-11).** M4 is now one integrated, separately versioned ruleset. Its specification is
> [m4_integrated_design.md](m4/m4_integrated_design.md), with the plan in [m4_implementation_plan.md](m4/m4_implementation_plan.md)
> and the review in [m4_design_audit.md](m4/m4_design_audit.md); see section O. The three-task plan below and
> section G are kept for the record only.

Specification in section G.

**M4a. Experience and levels (S/M).** Proximity XP from `TurnResult.kills`, three levels, level term in the trade formula. **M4b. Spendable gold and items (M).** Gold ledger, two upgrade tracks of two tiers, purchase decisions on absence turns, no reward effect. **M4c. Plate durability (M).** Persistent tower damage counter, plate thresholds with increasing durability, plate rewards on breaks, late-game plate removal; `plate_mode="durability"` in champion configurations only.

### M5. Bayesian opponent modelling and exploitation versus robustness (L)

Specification in section H. First on the exact game (K = 16, everything exact and cheap), then on the champion game with the M3 agents.

### M6. Research dashboard (L, not started until M3c exists)

Section D.6 below lists only what earlier milestones must leave behind for it: a versioned episode-trace schema (M1 `trace.py`, extended in M2 and M4), aggregate result JSON written by every experiment (as today), and a skill-grid export (policy behaviour as a function of both skills) produced by M3c.

---

## E. Champion mechanics specification

This section records the agreed design for the expanded game, proposes defaults where the brief left them open, and marks every remaining decision. Parameter names are proposals for `GameConfig`.

### E.1 State

Per side, in addition to the lane record (`fx, fh, rears, siege, v, plates`):

| Field | Meaning | Range (defaults) |
|---|---|---|
| `c` | champion cell, own coordinates | −T … +T (−2 … 2) |
| `hp` | champion health | 0 … `champion_hp` (6) |
| `absent` | turns still away (0 = present) | 0 … max(`recall_turns`, `death_turns`) (0 … 3) |
| `xp`, `level`, `gold`, `items` | M4 only | see G |

Per game (constant): `skill = (m_A, m_B)`, each in [0, 1].

Initial champion state: `c = −T`, `hp = champion_hp`, `absent = 0`. The environment state is `EnvState(lane: State, champ: (ChampSide, ChampSide), k: int)`; `mirror()` swaps the two champion records and the two skills.

### E.2 Actions and legality

`a = wave + 3·behaviour + 15·objective`, with `wave ∈ {SP, SH, F}`, `behaviour ∈ {HOLD, YIELD, TRADE, SAFE, RECALL}`, `objective ∈ {NONE, PRESSURE, VISION}`; plus `ABSENT = 45`. Legal set of a present champion:

- wave components as today (`wave_actions`: Freeze only when the enemy front is at or behind own −1);
- behaviours HOLD, YIELD, TRADE, SAFE always; RECALL only when `c ≤ recall_line` (−1) and only with `wave = SP` (one canonical code, no duplicates);
- siege objectives exactly as today, attached to HOLD and TRADE (the behaviours that stand with the wave); YIELD and SAFE during an eligible siege mean disengaging; HOLD or TRADE during an eligible siege *requires* an objective (as Contest does today).

Legal set of an absent champion: `{ABSENT}`. The environment steps every turn for both players, so the API never changes shape; an absent player's agent is called and must return the only legal action.

The legacy action space (18 codes) is kept for configurations with `champions=False`; `GameConfig.action_codes()` reports which one applies. Scripted baselines that only know wave and posture are mapped (Contest → HOLD, Safe → SAFE) so the existing suite plays the champion game unchanged.

### E.3 Resolution order of one turn

Both actions are read; nothing below reveals either action before both are committed.

1. **Champion intent.** From (behaviour, own state, enemy behaviour and state) compute each champion's target cell and its end cell `c'` (E.4). The Yield contingency ("retreat if the enemy Trades and is in reach") is evaluated here, from the enemy's *action*, which is legitimate because the environment resolves the joint action; the players never saw it.
2. **Lane step.** `step_turn` with the wave actions, the siege objectives, and `reach_cap = c'` per side (an absent side gets a cap below the lane so it is never present). Unchanged mechanics: movement, fights, thinning, crashes, tower fire, spawns, plates, vision.
3. **Farming.** `turn_farm` with the behaviour-to-posture mapping (E.4) and zero collection for absent sides.
4. **Trades.** For each side that chose TRADE and whose target is present and in range (E.5), draw the damage. Both draws use the pre-trade HP, so mutual death is possible.
5. **Ganks.** For each present side, draw occurrence from `gank_probability(state, side, action, cfg, exposure)` with `exposure = max(c, c')`, the vision timer as it stood at the start of the turn, and the action term of E.7; if it occurs, draw damage (E.7). Trade damage is applied before the gank draw, so a champion weakened by a trade is more likely to die to the gank that follows.
6. **Death and recall.** HP ≤ 0 → `hp = 0`, `absent = death_turns`, side leaves the lane (`c = −T` on return). RECALL chosen this turn → `absent = recall_turns`. A returning champion (timer reaching 0 at the end of this turn) stands at `c = −T` with `hp = champion_hp` at the start of the next turn.
7. **End of turn.** Timers (vision, absence) count down exactly as vision does today; XP, gold and purchases (M4).
8. **Reward.** `r = (farm_A + plates_A + bounty_A) − (farm_B + plates_B + bounty_B)`; `info` carries the full decomposition and every event.

Two consequences worth stating: farm collected in the turn of one's death is kept (the death is resolved after the lane step), and a champion that recalls still participates in nothing this turn and is not gankable (it channels at a cell it must already have retreated to, `c ≤ −1`).

### E.4 Persistent champion position

Target cell by behaviour (own coordinates, `w = own front cell` if a front exists, else the current cell):

| Behaviour | Target | Notes |
|---|---|---|
| HOLD | `clip(w, −T, T)`; `+T` if own crashed mass is under the enemy tower and an objective was chosen | today's Contest |
| TRADE | HOLD target + 1, clipped to `+T` | the aggressive step forward the brief asks for |
| YIELD | HOLD target; HOLD target − 1 if the enemy chose TRADE and its end cell is within trade range | contingent retreat |
| SAFE | `min(c, safe_line)` with `safe_line = −1` | never forward; today's Safe cap |
| RECALL | `−T`, then absent | legal only from `c ≤ −1` |

Movement: `c' = c + clip(target − c, −champion_speed, +champion_speed)` with `champion_speed = 2` (minions move two cells per turn; a champion keeps pace with its wave). Exposure for the gank formula: `e = max(c, c')`. Persistence therefore bites in two places: a champion that stands at `+2` and chooses SAFE is still exposed at `+2` this turn and reaches `0` at best; it cannot Recall until it has spent a turn getting to `−1`.

Farming posture: HOLD and TRADE → Contest efficiencies; YIELD → Contest unless it retreated this turn, then Safe; SAFE → Safe; absent → 0. Zoning uses the same mapping for the opponent. The existing `collection_efficiency` is unchanged in M2; a distance-based efficiency (kill cell versus champion cell) is a possible later refinement.

**Open decisions.** `champion_speed` 2 (default) versus 1 (stronger persistence; Hold could not follow a Shove). Whether Trade's "+1" should apply when the enemy champion is already adjacent (no need to step forward). Whether the three-level abstraction should replace the five cells if the position dimension slows learning.

### E.5 Stochastic trading

A side `i` that chose TRADE against a present target `j` *connects* if the global distance between their end cells is at most `trade_range = 1` (global positions: A at `c_A`, B at `−c_B`; distance `|c_A + c_B|`). If it does not connect (the target retreated, is far back or absent), the attacker falls back to farming at its forward position: no damage, Hold efficiencies, Trade exposure.

If it connects, the hit chance is

`h_i = clip(h0 + α·(m_i − m_j) + β[behaviour_j] + γ·(σ_i − σ_j) + λ·(level_i − level_j) + ι·(off_i − def_j), h_min, h_max)`

with `σ` the leading-wave stack count (as in the gank formula), `β[HOLD] = β[TRADE] = 0`, `β[YIELD] = −0.25`, `β[SAFE] = −0.40`, and the level and item terms zero until M4. Damage: `0` with probability `1 − h`, `1` with probability `h·(1 − q2)`, `2` with probability `h·q2`, `q2 = q2_base + q2_skill·m_i`.

Proposed defaults: `h0 = 0.55`, `α = 0.4`, `γ = 0.05`, `h_min = 0.05`, `h_max = 0.95`, `q2_base = 0.3`, `q2_skill = 0.2`. At equal skill and Hold defence a trade lands 1 HP about 38% of the time and 2 HP about 17%; against Safe about 10% and 5%. These numbers are a starting point to be reported, not calibrated.

Tower: a connecting attacker whose end cell is `+T` while the enemy tower is *not* occupied by its own crashed mass takes `tower_shot = 1` HP (a dive). With its mass under the tower the tower is busy, consistent with the thinning rule.

Trade versus Trade: both roll from pre-trade HP. Trade versus Hold: the defender takes the roll and farms at full efficiency. Trade versus Yield: the defender steps back (if in reach), takes the reduced roll, farms at Safe efficiency. Trade versus Safe: the roll is reduced further and often does not connect; the attacker is exposed at the forward cell either way. Farm versus Safe: no roll; the farmer collects at Contest efficiency without zoning. None of these is hard-coded as an outcome; they follow from positions, ranges and `β`.

**Open decisions.** Whether `h` should also depend on the attacker's own HP (desperation) or on the defender's HP (execute). Whether to add passive regeneration (`regen_at_tower = 1` HP per turn while SAFE at `−T`; default 0). Whether minions near the target should deal a flat point of damage to a diving attacker (default no).

### E.6 Mechanical skill

`skill: (m_A, m_B)` in `GameConfig`, default `(0.5, 0.5)`. Skill enters only execution probabilities: the hit chance (through `α`) and the two-damage chance (through `q2_skill`); the same difference term rewards defensive execution, because the defender's skill lowers the attacker's `h`. Skill does not enter the gank model in M2 (open: a `−κ·m_i` term on gank damage). Skill is observed by both agents (own and enemy) so one network can play both seats and so policies trained over a distribution of skills can be queried at any slider position later. Nothing in the environment makes a skilled champion aggressive; whether higher `m` leads a learned policy to trade more, stand further forward and get ganked more is the first empirical question of M3b.

### E.7 Ganks as events

- **Occurrence.** `P(gank_i) = gank_probability(...)` as today: base, exposure, action, wave-strength and vision terms. The action term becomes behaviour-and-wave based: HOLD +0.02, YIELD +0.02, TRADE +0.05, SAFE 0, plus +0.03 for Shove (today's SH-C 0.05 minus SP-C 0.02). Vision multiplies occurrence by `(1 − ρ)` and nothing else.
- **Damage given a gank.** `g = 1 + X`, `X ~ Binomial(2, p_e)`, `p_e = clip(p_g0 + p_g1·(e + T)/(2T) − p_def·[behaviour ∈ {YIELD, SAFE}], 0, 1)`, defaults `p_g0 = 0.2`, `p_g1 = 0.6`, `p_def = 0.2`. At the own tower a gank deals 1 / 2 / 3 with probabilities 0.64 / 0.32 / 0.04; under the enemy tower 0.04 / 0.32 / 0.64.
- **Death.** `hp − g ≤ 0`. With `champion_hp = 6` and a maximum of 3 gank damage, a full-HP champion never dies to a single gank: death requires prior trade damage or a second gank, which is how "probability of death depends on current HP" emerges without a separate death roll.
- **Consequences.** `absent = death_turns = 3`; the opponent's gain is the absence (free farm, free sieges, no zoning), plus `kill_bounty` resources if that knob is switched on (default 0). `gank_loss` (the expected-loss penalty L) is unused in champion mode and removed from the reward.

**Open decisions.** Whether a champion with its own crashed mass under the enemy tower should be gankable at a lower rate (tower aggro on minions). Whether a killed champion should drop the plates pending on its siege (default: yes, as the siege ends when its mass is cleared anyway).

### E.8 Recall, death, absence and return

| Event | Turns away | Where it leaves from | Return |
|---|---|---|---|
| Recall (chosen in turn k) | `recall_turns = 2` (turns k and k+1) | `c ≤ −1` required | start of turn k+2 at `−T`, full HP |
| Death (in turn k) | `death_turns = 3` (turns k+1, k+2, k+3) | anywhere | start of turn k+4 at `−T`, full HP |

While absent: legal set `{ABSENT}`; the lane uses `absent_wave_action = SP` with no champion presence (no shove damage, no freeze-break push, no objectives, no thinning); farm collection and zoning are zero; the side cannot be traded with or ganked; its vision timer keeps counting down; its pending plates lapse when its mass is cleared, as today. In M4, the absent turns are when purchases happen.

**Open decisions.** `absent_wave_action`: Slow Push (the stylised park-at-the-lane rule; default, no new code path) versus a "natural" free walk (a new movement mode). Whether Recall should take effect at the end of the turn instead (one more turn of exposure, more realistic, more complex).

### E.9 Reward and objective

`r_k = (F_A + P_A + B_A) − (F_B + P_B + B_B)`, realized, with `B` the kill bounty (default 0); zero-sum by construction; undiscounted; terminal value 0. Knobs defaulting to zero and intended for ablations only: `death_penalty`, `kill_bounty`, `terminal_hp_value`. The per-turn `info` dictionary reports farm, plates, bounty, trade damage dealt and taken, gank occurrence and damage, deaths, recalls, positions, exposure and the legal sets, so every metric in F.3 is a sum over `info`.

### E.10 Defaults at a glance

| Parameter | Default | Parameter | Default |
|---|---|---|---|
| `champions` | False (legacy); True in `GameConfig.champions()` | `champion_hp` | 6 |
| `champion_speed` | 2 | `safe_line`, `recall_line` | −1, −1 |
| `recall_turns`, `death_turns` | 2, 3 | `absent_wave_action` | SP |
| `trade_range` | 1 | `h0`, `α`, `γ`, `h_min`, `h_max` | 0.55, 0.4, 0.05, 0.05, 0.95 |
| `β` (HOLD, YIELD, TRADE, SAFE) | 0, −0.25, 0, −0.40 | `q2_base`, `q2_skill` | 0.3, 0.2 |
| `tower_shot` | 1 | `skill` | (0.5, 0.5) |
| gank action term (HOLD, YIELD, TRADE, SAFE, +SH) | 0.02, 0.02, 0.05, 0, +0.03 | `p_g0`, `p_g1`, `p_def` | 0.2, 0.6, 0.2 |
| `kill_bounty`, `death_penalty`, `terminal_hp_value` | 0, 0, 0 | `gank_loss` | unused in champion mode |

### E.11 Decisions required before M2 implementation

1. `champion_speed` 2 or 1 (E.4).
2. Whether siege objectives may attach to YIELD (E.2; proposed no).
3. Recall timing: immediate (proposed) or end of turn (E.8).
4. `absent_wave_action` (E.8; proposed SP).
5. Whether to include a kill bounty at all in the default (proposed 0) and whether to include passive regeneration at the own tower (proposed 0).
6. Whether skill enters gank damage (proposed no in M2).
7. Whether the exact chance-node micro-benchmark (M2e) is wanted at all, given that the legacy bridge already validates the trainer.

---

## F. RL algorithm and evaluation plan

### F.1 Why PPO with masking, and why not a framework

The game is two-player, zero-sum, simultaneous-move, finite-horizon (K = 32), perfect-information with chance, with 4 to 9 legal actions out of 46 and episodes of 32 steps. Policy-gradient methods with action masking handle variable legal sets cleanly; the horizon is short enough that bootstrapped value estimates are easy; the mirror symmetry lets one network play both seats, which halves sample needs and makes self-play a single-population problem. PPO is the most-used and best-understood option for this shape, is stable without much tuning, and its code is short.

Rejected or deferred: Stable-Baselines3 (single-agent, masking only through `sb3-contrib`, awkward for simultaneous two-player play); RLlib (heavy dependency, slow iteration for a game that steps in microseconds); Deep CFR and regret methods (overkill for perfect-information with chance, and the exact solver already provides equilibria where they are computable); NFSP (a reasonable second baseline if self-play PPO cycles; it needs a supervised average-policy network, which is deferred); tabular methods (the champion state is too large, and the legacy game is already solved exactly).

Implementation: PyTorch on CPU, a single-file PPO in the style of CleanRL adapted for (a) a batch of `LaneEnv`s stepped in a Python loop (vectorisation deferred until measured necessary), (b) masked categorical policies, (c) two seats per environment sharing one network through the own-view observation, (d) an opponent sampler. MLP of two hidden layers (128 units) with a policy head over 46 logits and a value head. Dependencies: `torch` as an optional extra (`rl`), never imported by `lanegame.env`.

### F.2 Staged approach

1. **M3a, supervised by the exact game.** Train against each fixed scripted opponent on the siege model at K = 16 (4,951 nodes, solved in six seconds). Oracle: `best_response_value` gives the exact optimum; the wrapped network gives its exact value. Pass criterion to be recorded, not assumed: report the fraction of the exact best-response gain achieved per opponent and per budget. Also report exact exploitability of the learned policy (expected to be high: a best response to one opponent is exploitable by others), which demonstrates the measurement before it is needed.
2. **M3b, self-play with a pool.** Opponents drawn from the latest checkpoint (probability 0.5), earlier checkpoints (0.3) and the scripted suite (0.2); the numbers are a starting point. Symmetric skills. Record learning curves, the metric suite and exact exploitability on the legacy configuration as a side check (train one run on the legacy K = 16 game in self-play; its exploitability must fall over training).
3. **M3c, evaluation harness.** Sections F.3 and F.4.
4. **Opponent populations proper.** If self-play cycles (visible as oscillating head-to-head results between checkpoints), move to a fixed-population scheme: keep every checkpoint, sample uniformly or by recent performance (a fictitious-self-play flavour).
5. **M3d, PSRO.** Oracle = the M3a trainer against a fixed mixture; meta-solver = `solve_matrix_game` on the empirical payoff matrix of the population (zero-sum, so the meta-strategy is a Nash of the restricted game with a certified gap). Start on the champion game at K = 16 or 24 so each oracle step is cheap.

### F.3 Metrics (all computed from `info`, all with confidence intervals)

- Relative resources per game (the objective), and win probability defined as `P(total > 0)`.
- Farm and plate income per side; farm missed (available minus collected).
- Trades attempted, connected, damage dealt and taken; HP at the end and HP time-series.
- Deaths, recalls, turns absent; ganks suffered; gank exposure (sum of exposure cells, time at `+T`).
- Position histogram; action frequencies by behaviour, wave and objective; vision turns; siege turns.
- Performance against each scripted opponent and against the frozen self-play policies of earlier runs.
- Approximate exploitability: the value a PPO best response trained for a fixed budget achieves against the frozen policy (report the budget).
- Exact exploitability and exact head-to-head values on legacy configurations (K ≤ 16), and on tiny champion configurations if M2e lands.
- Variance: standard errors from at least 2,000 evaluation episodes per cell; three training seeds per setting; common random numbers when comparing policies.

### F.4 What is and is not claimed

On the champion game no Nash guarantee is claimed. Claims take the form "policy X beats the scripted suite by Y ± Z, loses at most W ± Z to a best response trained with budget B, and on the exactly solvable sub-configuration its exploitability is E". Where an experiment can be run on the exact game (K = 16 legacy, tiny champion configurations) it should be, because that is where "mathematically defensible" is literally true.

---

## G. Economic progression plan (M4)

> **Superseded by section O and [m4_integrated_design.md](m4/m4_integrated_design.md).** The three levels, the two item
> tracks with hit-chance bonuses, the per-turn plate removal (`plates_fall_turn = 24`) and the 32-turn calibration
> targets below are no longer the plan.

### G.1 Experience and levels (M4a)

- XP source: enemy minion HP that dies within `xp_range = 1` cell of a *present* champion, read from `TurnResult.kills` (already keyed by cell), times `xp_per_hp`. Last-hit gold is unchanged, so a zoned player can collect XP but not gold, and a player far away collects neither, as the brief asks.
- Levels: three (1, 2, 3) at cumulative thresholds (`xp_levels = (0, 8, 20)` in wave-HP units, to be set so level 3 is reachable around the middle of a 32-turn game under equal play). Effect: the `λ` term of the trade formula (`λ = 0.10` per level of difference) and nothing else by default. Open: `hp_max` by level (6 / 7 / 8), minion damage by level.
- State: `xp` (integer) and `level` (derived; store `xp` only). Observation adds both sides' level and XP fraction.

### G.2 Spendable gold and items (M4b)

- Ledger: `gold_i = earned_i − spent_i`, with `earned` the cumulative reward components (farm, plates, bounty). Spending never changes the reward (section C.2, question 10).
- Items: two tracks (offence, defence), two tiers each (`item_cost = (4, 7)` resources per tier). Offence adds `ι_off = 0.08` per tier to one's hit chance and, at tier 2, +1 thinning per Apply Pressure turn; defence subtracts `ι_def = 0.08` per tier from the enemy's hit chance and gives a `block = 0.15` per tier chance to reduce incoming trade or gank damage by one. Open: a single combined track instead (simpler, fewer strategic archetypes).
- When: on absence turns only (recall or death). The absent player's legal set becomes `{ABSENT, BUY_OFF, BUY_DEF}` filtered by affordability on the last absent turn before return; this reuses the turn-step API and keeps the decision simultaneous-safe (nothing the opponent does that turn depends on it).
- Objective: unchanged. The question the design asks is whether a recall that costs two turns of farm pays back through a stronger lane presence; that answer must be learned, not written into the reward. Terminal value of items and unspent gold is a knob defaulting to zero.

### G.3 Tower plate durability (M4c)

- State per side: `plate_damage` dealt to the enemy tower (integer, persistent).
- Plates: `plate_count = 3` per tower with durabilities `plate_durability = (2, 3, 4)` hits (increasing); Apply Pressure deals one hit per turn (two with offence tier 2); crossing a threshold pays `plate_value[i]` (`(1.0, 1.0, 1.5)` proposed) once. After `plates_fall_turn = 24` remaining plates are removed without payment, creating the timing pressure the brief describes.
- Replaces the per-crash opportunities in champion configurations (`plate_mode="durability"`); `plate_mode="crash"` stays for the exact models. Thinning is unchanged. The siege study's reason for rejecting a persistent plate counter (state explosion under exact enumeration) does not apply to the RL environment.
- Open: whether partial damage decays when the siege ends (default no: the persistence is the point); whether the defender can repair (default no).

### G.4 Reward accounting, once more

Earned resources are the only reward. Spending, item power, levels, HP and tower damage are state. Plate rewards are paid once at thresholds. A terminal valuation, if ever introduced, must be symmetric and documented as a change of objective. Under these rules the expected per-turn reward of the two players sums to zero exactly, which the environment tests check.

---

## H. Opponent modelling research (M5)

### H.1 Two arenas

Run every experiment first on the **exact arena** (siege model, K = 16, or the previous revision at K = 32 if a longer horizon matters): opponent types are `Policy` objects, the posterior is exact, best responses and exploitability are exact, and a belief-augmented best response is enumerable for a handful of types. Then repeat on the **champion arena** with the M3 agents, where the same quantities are estimated. The exact arena is cheap (seconds per solve) and is where the quantitative claims are tightest.

### H.2 Type library

Parameterised scripted families (each parameter a tendency the brief lists): trade propensity, Hold-versus-Yield preference, shove frequency, freeze preference, pressure greed, recall HP threshold, vision-first. Plus ε-mixtures of each with the robust policy, plus frozen learned checkpoints. Types are Markov policies, so the likelihood of an observed action at (k, state) is exact.

### H.3 Inference

Finite type set with a prior; Bayes update per turn from the observed opponent action; a forgetting factor for strategy shifts; posterior entropy as the confidence measure. The combat randomness never enters the likelihood (actions are observed exactly), which is the formal separation between uncertainty about the opponent (epistemic) and randomness of trades and ganks (aleatoric) that the brief asks for. In the champion arena the agent sees the public action history in `info`; the modelling code lives outside the agent's Markov observation.

### H.4 Policies to compare

1. Robust: the exact equilibrium (exact arena) or the self-play policy (champion arena).
2. Best response to a known type: exact (`best_response_value`) or PPO fine-tuned against the fixed type.
3. Bayesian adaptive, two versions: (a) switch to the best response of the maximum-a-posteriori type once its posterior exceeds a threshold τ, otherwise play robust; (b) a belief-conditioned policy trained against the type prior with the posterior appended to the observation.
4. Risk-constrained exploitation: in the exact arena, the restricted Nash response (the opponent plays the model with probability p and is free otherwise; a small solver extension that fixes a component of one player's mixture), sweeping p; in the champion arena, deviation only when the approximate exploitability of the exploiting policy stays below a budget ε.

### H.5 Experiments and metrics

- Exploitation gain versus confidence: expected value against the true type as a function of τ (or ε), with the posterior entropy at the deviation time.
- Severity of being wrong: value when the true type is not the inferred one, and when it is outside the library (misspecification), and when it switches at a fixed turn.
- Skill asymmetry: the same grid at `m_opp ∈ {0.2, 0.5, 0.8}`; does a mechanically stronger opponent make exploitation more or less attractive?
- Noise: scale trade and gank randomness through the configuration and measure how much of the outcome variance is aleatoric (common random numbers across policies isolate it) versus epistemic (variance across types).
- Metrics: mean and CVaR of relative resources; regret against the oracle best response; worst case over the type set; time to identify (turns until the posterior passes τ); frequency and cost of wrong deviations.

---

## I. Performance and development budget

### I.1 Measured baseline and derived expectations

| Item | Value | Status |
|---|---|---|
| Lane turn (uncached) | 40 µs | measured |
| K = 32 episode, scripted policies | 1.3 to 2.1 ms | measured |
| Exact solve K = 8 / 12 / 16 (siege) | 0.2 / 1.2 / 5.6 s | measured |
| One million environment turns, legacy game | about 40 to 70 s single-threaded | derived from the two lines above |
| Champion layer overhead | to be measured in M2e; expectation: under 2× | derived (two draws and a position update per side) |
| PPO wall-clock | to be measured in M3a; expectation: network forward passes dominate, CPU sufficient for a 2×128 MLP | derived; no number is claimed |
| Exact K = 32 siege solve | 3.5 min, 3 GB | measured (siege study); not to be rerun in any milestone |

### I.2 Rules that keep milestones short

- **No exact solve above K = 16 in tests or milestone work.** The frozen `results/siege/solution_K32.json` is the K = 32 equilibrium for any evaluation that needs it (`io.read_solution` reconstructs the `TabularPolicy`).
- **Correctness tests and research sweeps live in different places.** Tests prove rules on small configurations; `experiments/` scripts produce numbers into `results/` with a JSON and a Markdown file each, as today, and are never run by the test suite.
- **Fast loop.** After M0, development runs `pytest -m "not slow"` (seconds); the full suite runs once at the end of a milestone.
- **Benchmark before training.** M1 prints throughput in `env-demo`; M2e records it; M3a records training wall-clock per million steps before any self-play is scheduled.
- **Small first, then scale.** K = 16 and small opponent sets until a result is reproducible across three seeds; champion game at K = 32 only in M3b and later.
- **Cache reusable results.** Checkpoints, tournament matrices, posteriors and solution files are written once and read by later experiments; experiments declare their inputs.
- **Memory.** Nothing in the roadmap needs more than a few hundred MB except the (excluded) K = 32 solve.

### I.3 Four kinds of cost, kept apart

- **Development time** (sessions): M0 S, M1 M, M2 L as five S/M tasks, M3 L as four M tasks, M4 L as three S/M tasks, M5 L, M6 L.
- **Simulator runtime**: negligible for M1 and M2 (seconds per thousand episodes, measured); the M2e benchmark decides whether any vectorisation is ever worth doing.
- **Training compute**: unknown until M3a; the plan is designed so that the first number is produced by a small run on the exact game, not by a long self-play run.
- **Model and token usage** (Claude Code sessions): bounded by scope. Each milestone touches a listed set of files, has its own tests, and does not require reading or rerunning the three studies. Research write-ups are produced by experiment scripts, not by hand.

---

## J. Immediate next task: Milestone 1, RL-ready environment

This section is written as the brief for the next Claude Code session. It is deliberately limited to the existing game (no champion mechanics).

### J.1 Goal

Add `lanegame/env/`, a simultaneous-move environment around the existing siege / revised / v1 games, such that two independently controlled scripted agents play complete, seeded, reproducible episodes, and such that any agent whose action depends only on the observation can be evaluated exactly by the existing solver.

### J.2 Modules and public API

`lanegame/env/core.py`

```python
@dataclass(frozen=True)
class Observation:
    k: int                      # decision index
    side: int                   # 0 = A, 1 = B
    state: State                # the lane state (shared, perfect information)
    vector: np.ndarray          # own-perspective float32 features, fixed length
    mask: np.ndarray            # bool, length = env.n_actions, True = legal

@dataclass
class StepResult:
    obs: Tuple[Observation, Observation]
    rewards: Tuple[float, float]        # (r, -r), realized payoff to A this turn and its negative
    terminated: bool                    # k reached cfg.horizon
    truncated: bool                     # reserved (max_turns < horizon); False by default
    info: dict                          # decomposition and events (see J.4)

class LaneEnv:
    def __init__(self, cfg: GameConfig, max_turns: int | None = None): ...
    n_actions: int                      # 18 (positioning on), 3 (v1); = max(cfg.action_codes()) + 1
    obs_size: int
    def reset(self, seed: int | None = None, root: State | None = None, k0: int = 0) -> Tuple[Observation, Observation]: ...
    def step(self, action_a: int, action_b: int) -> StepResult: ...
    def legal(self, side: int) -> Tuple[int, ...]: ...
    @property
    def k(self) -> int: ...
    @property
    def state(self) -> State: ...
```

Rules: `step` raises `ValueError` on an illegal action or when called after termination; the chance generator is `np.random.default_rng(seed)` created in `reset` and used only for gank draws (and, later, trades); the environment holds no agent randomness. Rewards are the realized `farm_payoff + plate_payoff + gank_loss·(g_B − g_A) + crash choice payoff`, exactly the per-turn quantities `simulate.play_episode` accumulates. The crash-choice stage of the `v2` / `v1` models is handled inside `step` with the `greedy_resources` default unless the agents provide a chooser (keep `v2` support minimal: the greedy default plus an optional `choose` hook on the `Agent`).

`lanegame/env/observation.py`

```python
def observe(state: State, k: int, side: int, cfg: GameConfig) -> np.ndarray
def obs_size(cfg: GameConfig) -> int
def legal_mask(state: State, side: int, cfg: GameConfig) -> np.ndarray
```

Features (own perspective, in this order): `k / K`, `(K − k) / K`, one-hot of `k mod cfg.phase_period`; lane grid of `2·base + 1` cells with own minion HP per cell and enemy minion HP per cell (fronts and rears, divided by `cfg.hp_cap`); own and enemy: siege mass / `siege_cap`, vision timer / `vision_turns`, pending plates / `plates_cap`, siege-eligible flag, has-front flag. `observe` must satisfy `observe(s, k, 1, cfg) == observe(s.mirror(), k, 0, cfg)`.

`lanegame/env/agents.py`

```python
class Agent(Protocol):
    def act(self, obs: Observation, rng: np.random.Generator) -> int: ...

class ScriptedAgent:          # wraps a lanegame.policy.Policy; samples from policy.move(k, state, side, legal) with rng
class PolicyFromAgent(Policy): # wraps an Agent whose act is deterministic given obs -> exact evaluation
def play_episode(env, agent_a, agent_b, seed, record=False) -> EpisodeLog
def run_episodes(env, agent_a, agent_b, episodes, seed) -> EpisodeStats
```

Agent randomness comes from a generator the caller seeds (`np.random.SeedSequence(seed).spawn(3)` → environment, agent A, agent B), so swapping one agent does not change the chance draws of the other or of the environment.

`lanegame/env/trace.py`: `EpisodeLog` (per-turn records: k, state list, actions and labels, legal sets, rewards, decomposition, gank draws) and `write_trace(log, path)` with `{"schema": "lanegame.episode", "version": 1, "config": ..., "turns": [...]}`.

`lanegame/env/pettingzoo_adapter.py`: `class LaneParallelEnv(pettingzoo.ParallelEnv)` with agents `["A", "B"]`, `Discrete(n_actions)` action spaces, `Dict({"observation": Box, "action_mask": MultiBinary})` observation spaces, `reset(seed, options)` and `step(actions)`; imports `pettingzoo` and `gymnasium` lazily and raises a clear `ImportError` otherwise.

CLI: `python -m lanegame env-demo [--model siege|revised|v1] [--a aggressive] [--b passive_safe] [--episodes 200] [--seed 0] [--horizon 32]` prints `EpisodeStats` (mean ± standard error, ganks, farm, plates, action frequencies) and episodes per second.

### J.3 Packaging

`pyproject.toml`: add `lanegame.env` to `packages`; optional extras `rl = ["pettingzoo>=1.24", "gymnasium>=0.29"]` (PyTorch comes with M3). `requirements.txt` unchanged. README: one layout line and one quick-start line.

### J.4 `info` contents per step

`farm (A, B)`, `farm_available (A, B)`, `plates (A, B)`, `gank_p (A, B)`, `ganked (A, B)` booleans, `crash_hp`, `exposure`, `objectives`, `vision_set`, `thinned`, `legal (A, B)`, `actions (a, b)`, `action_labels`. Everything a later dashboard or metric needs for the legacy game.

### J.5 Tests (`tests/test_env.py`, using the existing session fixtures)

1. `test_mask_equals_legal_actions` on every reachable (k, state) of the three K = 16 fixtures.
2. `test_observation_is_markov_and_mirror_equivariant`: equal states give equal vectors; `observe(s, k, B) == observe(s.mirror(), k, A)`; vector length equals `obs_size(cfg)`; all features in [0, 1].
3. `test_env_matches_exact_expectation`: for five policy pairs per model (reuse `tests/test_simulation.PAIRS`), the mean of 3,000 episodes is within 4.5 standard errors of `evaluate(game, layers, pa, pb)`.
4. `test_env_without_randomness_equals_exact_value`: with every gank parameter zero, every episode total equals `evaluate` to 1e-12 and the farm and plate sums equal `path_statistics`.
5. `test_env_is_reproducible`: same seed, same trajectories (actions, rewards, gank draws); different seed differs; replacing agent B's seed leaves the environment's gank draws against identical action sequences unchanged.
6. `test_step_rejects_illegal_actions_and_after_termination`.
7. `test_rewards_are_zero_sum_and_sum_to_episode_payoff`.
8. `test_policy_from_agent_is_exactly_evaluable`: a deterministic agent (argmax of a fixed scoring of the observation) wrapped as a `Policy` has an exact value and exploitability from `evaluate` / `exploitability`, and 2,000 environment episodes agree with the exact value within 4.5 standard errors.
9. `test_simultaneity_by_construction`: the observation objects returned before `step` do not change when the other agent's action changes (compare `obs` for B across two different `a` choices from the same state: identical), and no public method accepts one action without the other.
10. `test_pettingzoo_api` (skipped when the package is missing): `pettingzoo.test.parallel_api_test(LaneParallelEnv(cfg), num_cycles=50)`.
11. `test_env_demo_cli_runs` with 5 episodes.

### J.6 Acceptance criteria

- All existing 305 tests pass unchanged; the new tests pass; nothing under `mechanics.py`, `game.py`, `solver.py`, `state.py`, `config.py`, `experiments/` or `results/` is modified.
- `python -m lanegame env-demo --a aggressive --b passive_safe --episodes 200 --seed 0` runs in seconds and prints throughput; the number is recorded in the session summary.
- A one-paragraph note in `docs/rl_research_roadmap.md` (this file, a new section "M1 outcome") records the measured throughput, the observation size, and any deviation from this specification.

### J.7 Out of scope for M1

Champion mechanics of any kind; changes to the action encoding; vectorised or batched environments; neural networks; Gymnasium spaces outside the adapter; the dashboard; any experiment script; any K > 16 exact solve.

---

## K. Summary

1. **Architecture.** Shared simulator = the existing deterministic lane mechanics plus a new champion layer; one `GameConfig` with champion fields defaulting to off; a NumPy-only simultaneous-move environment (`lanegame/env/`) over both; the exact modules untouched and used as the benchmark through a Markov-observation bridge; PyTorch training code (`lanegame/rl/`) that depends only on the environment; a thin PettingZoo adapter for tooling.
2. **Sequence.** M0 test markers (optional) → M1 environment → M2a position and behaviours → M2b recall and absence → M2c trading and skill → M2d ganks as events and death → M2e validation and optional exact micro-benchmark → M3a PPO versus scripted opponents on the exact game → M3b self-play with a pool on the champion game → M3c evaluation harness → M3d PSRO (optional) → M4a XP → M4b gold and items → M4c plate durability → M5 opponent modelling (exact arena first) → M6 dashboard.
3. **Reused as is.** `GameConfig`, `State` and its mirror, `step_turn` and `legal_actions`, `gank_probability` and `collection_efficiency`, the matrix-game solver, backward induction, best responses, exploitability, path statistics, the `Policy` interface and scripted baselines, the JSON conventions, reachability and scenarios, the K = 16 test fixtures, the frozen K = 32 solution.
4. **Unresolved design decisions** (E.11 and the open items of E.4 to E.8, G, H): champion speed; whether Yield may carry siege objectives; recall timing and legality line; the absent side's wave behaviour; kill bounty and regeneration (both proposed off); skill in gank damage; single versus two item tracks; plate thresholds and the late-game removal turn; whether to build the chance-node exact game at all; the equilibrium-selection rule for reporting (inherited from the earlier reports).
5. **Complexity and compute risks.** The champion state multiplies the lane state by about 10⁴, so exact enumeration is limited to tiny configurations by design. The simulator is fast enough (25,000 turns per second measured) that training cost will be set by network inference, to be measured in M3a before any long run. The one place where the certified core is touched (optional chance nodes) is guarded by the pinned values of all three models. The trade and gank formulas are the main modelling risk: they must be reported at their defaults, not tuned for interesting curves.
6. **Smallest useful next task.** Milestone 1 as specified in section J: the environment, scripted agents, the exact bridge and its tests, with no new mechanics.
7. **Location of this document.** `docs/rl_research_roadmap.md`.

---

## L. M1 outcome (implemented 2026-10-11)

**Status.** Milestone 1 is implemented and verified against the exact game. No file under `mechanics.py`, `game.py`, `solver.py`, `state.py`, `config.py`, `experiments/` or `results/` was touched; the existing 305 tests pass unchanged alongside 65 new ones (370 in total, 7 min 44 s on this machine).

### L.1 Files

| Added | Purpose |
|---|---|
| `lanegame/env/core.py` | `LaneEnv`, `Observation`, `StepResult`, `Chooser`, `greedy_chooser` |
| `lanegame/env/observation.py` | `observe`, `legal_mask`, `obs_size`, `n_actions`, `feature_names` |
| `lanegame/env/agents.py` | `Agent` protocol, `ScriptedAgent`, `GreedyObservationAgent`, `AgentPolicy`, `as_policy`, `play_episode`, `run_episodes`, `episode_streams`, `agent_chooser`, `sample` |
| `lanegame/env/trace.py` | `EpisodeLog`, `EpisodeStats`, `trace_dict`, `write_trace`, `read_trace` (schema `lanegame.episode`, version 1) |
| `lanegame/env/pettingzoo_adapter.py` | `LaneParallelEnv` (optional: pettingzoo, gymnasium) |
| `lanegame/env/__init__.py` | exports |
| `tests/test_env.py` | 65 tests (sections A to J of the brief, plus traces) |

| Changed | Change |
|---|---|
| `lanegame/__main__.py` | `env-demo` subcommand (`--model --a --b --episodes --seed --horizon --exact --trace --cache`) |
| `pyproject.toml` | package `lanegame.env`; optional extra `rl = [pettingzoo, gymnasium]` |
| `README.md` | roadmap pointer, quick-start line, layout entry |

### L.2 Public API

```python
env = LaneEnv(cfg, max_turns=None, choosers=None, cache_size=0)
obs_a, obs_b = env.reset(seed=42, root=None, k0=0)       # Observation(k, side, state, legal, vector, mask)
result = env.step(action_a, action_b)                    # StepResult(obs, rewards=(r, -r), terminated, truncated, info)
env.legal(side); env.masks(); env.observe(); env.k; env.state; env.done; env.n_actions; env.obs_size

observe(state, k, side, cfg) -> float32[obs_size]        # pure function; observe(s, k, 1) == observe(s.mirror(), k, 0)
legal_mask(state, side, cfg) -> bool[n_actions]          # True exactly on mechanics.legal_actions

ScriptedAgent(policy)            # any lanegame Policy; distribution() = policy.move, choose() = policy.choose
GreedyObservationAgent(W, b)     # deterministic argmax of a linear score; one-hot distribution
as_policy(agent, cfg) -> Policy  # exact bridge; TypeError for agents without an explicit distribution
play_episode(env, agent_a, agent_b, seed, record=False) -> EpisodeLog
run_episodes(env, agent_a, agent_b, episodes, seed) -> EpisodeStats   # mean, stderr, farm, plates, ganks, action frequencies, throughput
write_trace(log, cfg, path) / read_trace(path)
LaneParallelEnv(cfg)             # PettingZoo ParallelEnv; obs = {"observation", "action_mask"}
```

Reward per turn (payoff to A; B receives the negative): farm difference + plate difference + `gank_loss` × (B ganked − A ganked) + crash-reward difference, with the two gank checks realized from the environment's own generator. `info` carries every component, the gank probabilities and uniform draws, crash sizes, options and choices, exposure, objectives, vision, thinning, the engine's event counters, and the state before the turn, before the choice stage and after.

### L.3 Sizes and measured throughput (single thread, this machine)

| Configuration | obs_size | n_actions | legal per turn |
|---|---|---|---|
| siege (`GameConfig()`) | 34 | 18 | 4 to 6 |
| farming revision (`v2`) | 34 | 6 | 4 to 6 |
| original (`v1`) | 32 | 3 | 2 to 3 |

| Measurement | Value |
|---|---|
| `env-demo`, aggressive vs passive_safe, K = 32, 200 episodes | 245 episodes/s, 7,800 turns/s (no cache) |
| same with `--cache 20000` (bounded LRU of lane steps) | 368 episodes/s, 11,800 turns/s |
| uniform vs uniform, K = 32 | 230 episodes/s, 7,400 turns/s (8,200 with cache) |
| K = 16 scripted pairs, 2,000 episodes | about 490 episodes/s, 7,800 turns/s |
| raw `step_turn` + `legal_actions` (reference) | about 25,000 turns/s |
| old `simulate.play_episode` (reference) | 15,000 to 24,000 turns/s |

The wrapper costs about 90 µs per turn on top of the 40 µs lane step: two observations and masks, the legal sets, the uncached farm and gank computation (which `LaneGame.transition` cached), the `info` record and the log. Profiling found no single dominant item after two cheap fixes (inverse-CDF sampling instead of `Generator.choice`, list-built observations). The bounded cache recovers about half of the gap for repetitive scripted play; it is off by default so memory stays predictable. Further optimisation is deferred until M3a measures whether the environment or the network dominates.

### L.4 Tests (`tests/test_env.py`, 65 items, about 100 s including the three K = 16 fixture solves)

A. The mask equals the engine's legal set at every reachable node and through the environment; illegal actions, out-of-range codes and stepping outside an episode raise. B. Observations are float32, in [0, 1], deterministic, mirror-equivariant at every reachable node, encode the state (spot checks), and never depend on the current turn's actions. C. Same seed gives identical trajectories; the chance stream is independent of the actions played and of the agents' draws (common random numbers); the agents' streams are independent of each other. D. Rewards are zero-sum and decompose exactly; the deterministic parts of a step equal `LaneGame.transition` with and without the cache. E. The mean of 2,000 episodes agrees with the exact expected value within 4.5 standard errors for five policy pairs per model, equilibrium policies included. F. With ganks off and deterministic pairs every episode equals the exact value and records the exact path's farm, plates and crash resources. G. `ScriptedAgent` round-trips to its policy; a deterministic observation agent has a valid one-hot distribution at every node equal to its environment action, an exact value that the environment reproduces, a non-negative exact exploitability, and earns at most v* against the equilibrium; a sampling-only agent is refused by `as_policy` but plays; in the "choice" mode the agents' own crash choices are used (a vision-taking shover from a pre-crash root is reproduced exactly) and the greedy fallback is reported. H. Termination, truncation, reset, roots and k0, invalid `max_turns`. Traces round-trip through JSON with chained states and consistent accounting. I. `pettingzoo.test.parallel_api_test` passes for all three configurations (skipped if the package is missing). J. The CLI runs with and without `--exact`.

### L.5 Deviations from section J, and known limitations

- `Observation` carries the legal tuple next to the mask (agents need it; it is the engine's set, not a second rule).
- The crash-choice stage of the older configurations is resolved through `choosers` callbacks wired to the agents' `choose` methods rather than by a greedy default, as the M1 brief required; `info["choice_default_used"]` marks trajectories in which the default was used. Through the PettingZoo adapter the default is used unless `choosers` are passed (the Parallel API has no second stage). The siege model has no such stage.
- Agents sample with an inverse-CDF `sample()` rather than `Generator.choice`, so the environment does not consume random numbers in the same order as `simulate.play_episode`; agreement with it is statistical, agreement with the exact solver is what the tests check.
- `n_actions = max(cfg.action_codes()) + 1` (18 / 6 / 3), not a constant 18.
- The exact-value test uses 2,000 episodes per pair instead of 3,000.
- `seed=None` gives fresh entropy (non-reproducible by design); the exact bridge requires an explicit `distribution`; a grid cell is clipped at 1 when two packets of one side share a cell beyond `hp_cap` (possible near a base only).
- Not done, by scope: champion mechanics, vectorised environments, any learning.
- The optional packages `pettingzoo` and `gymnasium` were installed into the project's `.venv` to run the adapter test; the core environment does not import them.

### L.6 Reproduction

```bash
python -m lanegame env-demo --a aggressive --b passive_safe --episodes 200 --seed 42          # siege, K = 32
python -m lanegame env-demo --a stack_builder --b freeze_contest --horizon 16 --episodes 2000 --seed 42 --exact
python -m lanegame env-demo --model v1 --a always_shove --b defensive_freeze --horizon 16 --episodes 2000 --seed 7 --exact
python -m lanegame env-demo --episodes 500 --seed 42 --cache 20000 --trace results/env_trace_demo.json
python -m pytest tests/test_env.py -q            # the new tests (~100 s)
python -m pytest -q                              # everything (370 tests)
pip install "pettingzoo>=1.24" "gymnasium>=0.29" # optional, for the adapter test (skipped otherwise)
```

---

## M. M2a Outcome — Champion State, Positioning, and Behaviors (implemented 2026-10-11)

**Status.** Champion mode exists as an opt-in configuration, `GameConfig.champions()`, played through `lanegame.env`. HP, absence, Recall and skill effects are dormant as planned. `GameConfig()`, `GameConfig.v2()` and `GameConfig.v1()` are unchanged: no default changed, the lane step's new parameter defaults to the legacy rule, and the existing tests, including the reproduction tests of all three reports, pass. The exact solver was not touched.

### M.1 Files

| File | Change |
|---|---|
| `lanegame/champions.py` | **new**: champion state, legality, the turn resolution, farming and gank terms |
| `lanegame/config.py` | champion fields (all inert unless `champion_mode`), `GameConfig.champions()`, the champion action encoding and labels, validation |
| `lanegame/mechanics.py` | `step_turn(..., champion_reach=None)`: optional per-side champion contribution line; `None` is the legacy rule |
| `lanegame/env/core.py`, `observation.py`, `agents.py`, `trace.py`, `pettingzoo_adapter.py`, `__init__.py` | champion state in the environment, observations, masks, champion agents, trace fields, adapter option |
| `lanegame/__main__.py` | `env-demo --model champions` |
| `tests/test_champions.py` | **new**: 25 tests |

### M.2 State and actions

Per side: `pos` (own coordinates, −2…+2, start −2), `hp` (6, constant), `absent` (0). Per seat: `skill_a`, `skill_b` in the configuration (default 0.5, no effect yet). `ChampState.mirror()` swaps sides, and `swap_seats(cfg)` swaps the skills. The lane `State` is unchanged, and the environment holds both.

Action code `a = wave + 3·behavior + 15·objective`, with behaviors Hold 0, Yield 1, Trade 2, Safe 3, Recall 4 (reserved), and `ABSENT = 45`. That gives 46 slots. The legal sets are:

- The wave component follows the lane's own rule; Freeze is state-dependent as before.
- Hold, Yield, Trade and Safe are always legal without an objective.
- Pressure and Vision are legal only with Hold or Trade, and only when the side's crashed mass is under the enemy tower **and** its champion can reach that tower this turn (`pos + champion_speed ≥ 2`).
- Recall and ABSENT are never legal. Passing them to the resolver raises `NotImplementedError`.

There are 8 to 12 legal actions without an eligible siege and 16 with one.

### M.3 Turn resolution (`champions.resolve_turn`)

1. **Wave destinations.** The lane is probed once with full champion support. Each side's farm spot is its leading wave's end cell, clipped to ±2.
2. **Targets.** An objective targets +2. Hold and Yield target the farm spot. Safe targets min(farm spot, −1). A Trade without an objective then chases its opponent's intended cell (its start cell if the opponent also trades). It stays put if already in contact, otherwise goes one cell short, and dives into the tower cell if that is where the opponent will be. Movement is at most `champion_speed = 2` cells per turn.
3. **Evasion.** A Yield or Safe champion that would end in contact with an enemy trader retreats to the furthest-forward cell out of contact, never ahead of its own target. Hold never evades.
4. **Contact.** Champions are in contact within `trade_range = 1`, except that a champion in its own tower cell can be reached only from inside that cell. A Trade connects on contact. A connected trade suspends that trader's siege objective for the turn.
5. **Modes.**
   - *farm*: Hold, unthreatened Yield, unconnected Trade. Contributes to fights up to end + 1.
   - *trading*: a connected Trade. No contribution, and its collection is multiplied by `trade_farm_factor = 0.5`.
   - *retreat*: a threatened Yield. Contributes only at or behind min(end, −1).
   - *safe*: Safe. Likewise.
6. **Lane step.** `step_turn` runs with the effective objectives and those contribution lines.
7. **Farming.** The legacy accounting applies, but a champion collects only kills within `farm_reach = 1` cell of its end cell. The legacy efficiency is used, with Contest for farm and trading and Safe for retreat and safe. The opponent zones a kill only when it is in farm or trading mode and within one cell of that kill.
8. **Exposure.** Exposure is max(start, end). The gank probability keeps the legacy form with a behavior term: Hold and Yield 0.02, Trade 0.05, Safe 0, plus 0.03 for Shove with a non-Safe behavior. Hold and Safe reproduce the legacy Contest and Safe terms exactly. The gank is still a resource loss, realized by the environment.

Siege compatibility follows from steps 2, 4 and 6. Plates, thinning and vision happen only when the champion stands at the enemy tower and is not trading. A crashed wave with a distant champion yields nothing beyond the plate opportunities the lane already records.

### M.4 Observations and masks

`obs_size` is 50: the 34 M1 features plus 8 per champion. Those are a one-hot of its cell in the observer's coordinates (5 cells), HP/6, absent/3, and skill. The mask has 46 slots and comes from `champion_legal_actions`, which builds on `mechanics.wave_actions` and `siege_eligible`. Legacy vectors and masks are unchanged.

The M1 grid-clipping concern does not arise in practice. Over every reachable state of the siege model (K = 12), the revised model (K = 16) and v1 (K = 16), no two packets of a side share a cell and the observation vector is injective within each turn.

### M.5 Behavior matchups (scripted, 200 seeded episodes each, K = 32, payoff to A)

| A vs B | payoff | farm A / B | mean exposure A / B | trades connected (A) | B evaded |
|---|---|---|---|---|---|
| Hold vs Hold | +0.13 | 9.61 / 9.61 | −0.06 / −0.06 | 0 | 0 |
| Hold vs Yield | +0.12 | 9.61 / 9.61 | −0.06 / −0.06 | 0 | 0 |
| Hold vs Safe | +2.36 | 13.72 / 4.27 | −0.06 / −1.03 | 0 | 0 |
| Trade vs Hold | −9.95 | 4.80 / 9.61 | 0.00 / −0.06 | 31 | 0 |
| Trade vs Yield | +0.75 | 13.72 / 0.00 | 0.00 / −2.00 | 0 | 31 |
| Trade vs Safe | −2.35 | 13.72 / 0.00 | 0.00 / −2.00 | 0 | 31 |
| Trade vs Trade | +0.35 | 4.80 / 4.80 | 0.00 / 0.00 | 32 | 0 |

These rows confirm the design:

- An unthreatened Yield is exactly a Hold.
- Hold against Safe reproduces the legacy Contest-against-Safe economy to the cent (farm 13.725 / 4.27, payoff +2.36).
- A trade that connects halves the trader's farm.
- Yield and Safe escape a trader at the lane by giving up their farm.
- A trader whose target evades falls back to full farming and keeps its forward position.

Without HP damage (M2c), being traded costs nothing yet, so Trade is not attractive in these numbers. That is expected.

### M.6 Tests and throughput

- **Tests.** `tests/test_champions.py` has 25 tests and runs in about 1 s. It covers configuration and encoding, and the lane step reproducing the legacy step on every reachable K = 8 transition when given the legacy reach lines. It covers initial state, asymmetric skills, observation bounds and mirror equivariance, and mirror equivariance of the whole turn. It checks bounded and persistent movement and dormant HP, then movement and behavior: Hold following its wave, Trade approaching and connecting, Safe not teleporting, unthreatened Yield equal to Hold, threatened Yield retreating, Trade falling back to farming with its exposure kept, a connected Trade losing shove and farm, and tower protection with dives. Finally it covers farm reach and zoning, legacy-equivalent gank terms, masks equal to the legal sets with no reserved or redundant codes, the no-remote-siege rules including objective suspension, reserved mechanics raising, zero-sum and reproducible scripted episodes, the legacy-agent mapping, PettingZoo in champion mode, and the CLI. The 65 M1 tests pass unchanged, and the full suite passes: 395 tests in 6 min 33 s.
- **Throughput.** Champion mode runs 5,000 to 6,200 turns/s (158 to 193 K = 32 episodes/s) uncached, against 8,400 to 8,800 for M1 on the same machine. With `cache_size = 20000` it runs 7,500 to 11,600. The extra cost is the lane probe of step 1 plus the champion bookkeeping.

### M.7 Deviations from the M2a proposal, and decisions to revisit before M2b

1. **Hold, Yield and Safe follow the wave's end-of-turn cell.** Using the wave's start cell made champions lag one turn behind a walking wave and lose the first contact's farm and shove. The cost is one extra lane step per turn. The final wave can differ slightly from the probed one when a champion withdraws its contribution.
2. **Safe follows its wave up to the safe line instead of never moving forward.** The literal rule left a Safe champion at its tower for the whole game with no farm at all, unlike the legacy Safe it replaces.
3. **Safe also evades a credible trade.** The brief expects Trade against Safe to usually fail. Without evasion a Safe champion on the safe line was traded every turn by a trader at the lane.
4. **Tower-cell protection, with a dive as the only way in.** This is new; without it a tower-hugging defender was traded every turn from one cell outside its tower. Diving costs the trader full exposure and all lane farm. M2c's tower shot should apply exactly to divers.
5. **A trader chases its opponent's intended cell, not its start cell.** Chasing the start cell made champions pass each other and cycle every turn. Both behaviors are committed simultaneously, so this is resolution of committed actions, not observation.
6. **Hold and Trade without an objective stay legal during a siege.** They are no longer dominated by Pressure, because they keep the champion with its next wave instead of at the tower.
7. **Pressure or Vision with Trade is suspended when the trade connects.** This is the attention cost.

Open questions:

- Whether `trade_farm_factor = 0.5`, `farm_reach = 1`, the gank behavior terms and `champion_speed = 2` are the right defaults.
- Whether a threatened Yield should keep partial wave contribution.
- Whether Trade's chase should also be limited by minions (it currently ignores them).
- Whether Safe should drop its zoning effect on the opponent entirely. It currently zones nothing, as in the legacy model.

### M.8 Reproduction

```bash
python -m lanegame env-demo --model champions --a trade --b hold --episodes 200 --seed 42
python -m lanegame env-demo --model champions --a aggressive --b passive_safe    # legacy baselines mapped into champion actions
python -m pytest tests/test_champions.py -q      # 25 tests, ~1 s
python -m pytest tests/test_env.py -q            # M1 compatibility, ~100 s
python -m pytest -q                              # everything
```

---

## N. M2b–M3c outcome (implemented 2026-10-11)

M2b–M2e (Recall and absence, stochastic trading with mechanical skill, ganks with HP damage, death and respawn,
the validation gate) and M3a–M3c (masked PPO, frozen-snapshot self-play, evaluation with approximate best
responses) are implemented. The full write-up is [docs/rl/m2_m3_implementation_report.md](rl/m2_m3_implementation_report.md);
the running log is [docs/rl/overnight_implementation_log.md](rl/overnight_implementation_log.md). Decisions that differ
from sections E and F above:

- Recall is legal at `pos <= -1` with no state-based threat rule; an enemy Trade that reaches the recaller
  interrupts it (a state rule made Recall impossible whenever the opponent stood at the lane).
- Champion mode has its own gank-occurrence parameters with no unconditional base rate; gank damage is
  Binomial(4, p_e) (0-4 HP), its defensive reduction requires actually playing defensively; skill enters gank
  damage with a small term.
- The 2-HP trade chance uses the skill difference (defensive mitigation), not the attacker's skill alone.
- M2e changed three champion defaults on evidence: `kill_bounty` 0.5, `retreat_farm_reach` 2, defensive-play
  gank defense.
- The M3a calibration on the legacy game (K = 16, exact) shows PPO reaching 23-94% of the exact best-response
  gain (more with the optional, exact `gank_reward="expected"` variance reduction), and frozen-snapshot self-play
  that does NOT reduce exact exploitability: section F.2 step 2 should be followed by PSRO (step 5) before scaling.

---

## O. M4 amendment: integrated economy, wave dynamics and continuing time (design 2026-10-11)

The diagnostics study ([openai_five_diagnostics_report.md](rl/openai_five_diagnostics_report.md)) found limits in the
game design that PPO tuning cannot remove:

- Freeze is inert in every reachable state (§D.5).
- Recall restores HP only, at a high price.
- The 32-turn terminal horizon rewards a late Shove.
- Sieges are profitable but never visited in self-play.

M4 therefore replaces section G with **one integrated ruleset**, `GameConfig.m4()`. The legacy exact game and
`GameConfig.champions()` stay unchanged and pinned.

| Document | Content |
|---|---|
| [m4_integrated_design.md](m4/m4_integrated_design.md) | the mathematical specification: state, resolution order, Freeze maintenance, gold and XP, levels 1–18, fixed-point HP and attack power, one item track with automatic purchase at base, persistent tower damage with three plates, continuing time with γ = 0.99, observation, parameters, invariants, ten worked scenarios |
| [m4_implementation_plan.md](m4/m4_implementation_plan.md) | work packages WP0–WP10, the tests per package, integration tests, legacy pins, validation stages S0–S6, conditions before training |
| [m4_design_audit.md](m4/m4_design_audit.md) | contradictions, exploits, redundancies, snowballing, parameters needing experiments, ranked alternatives |

Differences from section G:

- levels 1–18 instead of three;
- attack power and max HP scale with level and item tier, while hit probability depends on skill only;
- one item track instead of two;
- no plate removal at a fixed turn;
- no terminal turn at all.

**Next milestone:** M4.2 = WP0 + WP1 (ruleset scaffold, legacy pins, continuing time with segment PPO and discounted
evaluation), then WP2–WP7.

## P. M4.2 outcome: integrated implementation and pre-training validation (2026-10-11)

Branch `implementation/m4-integrated`. WP0–WP10 of the M4 plan are implemented as the separately versioned ruleset
`GameConfig.m4()`; every legacy configuration stays bit-identical (pins taken on the unchanged engine). Validation
S0–S4 is complete. Full account: [m4_implementation_report.md](m4/m4_implementation_report.md); results:
`results/m4/`.

- S0 legacy pins and all 500 earlier tests pass; S1: 679 tests pass (2 strict expected failures record design
  findings), a 68 000-turn soak is clean, the observation is exactly decodable on the exhaustive reachable lane.
- S2: levels 2 / 6 / 11 / 18 at turns 4 / 19 / 58 / 144 (targets 4 / 20 / 59 / 147).
- S3: Freeze gate G1, G2 and the G3 rule pass on 163 reachable scenarios; a Shove + Hold breaker breaks a freeze later
  than the lane-only design table because M2's farm-spot rule walks it back when its probe predicts a crash.
- S4: no dominant scripted strategy, moderate snowballing; item tiers have almost no measurable value (item leads last
  16–22 of 400 turns), and siege pressure is strongly unprofitable for scripted agents.
- Throughput 6 444 turns/s uncached (8 818 cached), target 4 000.

**Next:** M4.3 (S5 PPO smoke and pilot, S6 research runs), after a review of design decision D4 (item prices and
tiers) in light of the S4 item finding.

## Q. M4.2b calibration: gank severity, item strength, the Shove → Recall cycle (2026-10-12)

Branch `calibration/m4-2b`. Exploratory staged calibration within a ~2M-turn budget; **M4 defaults unchanged**. Full
account: [m4_calibration_report.md](m4/m4_calibration_report.md); results: `results/m4_calibration/`.

- **The shove penalty is lane farm, not ganks.** A presser loses −17 per 400 turns against hold_farmer, −16.2 of it in
  farm (crashed waves farmed by the defender at its tower; the parked configuration). Gank damage 0.60 → 0.40 halves
  forward deaths and is worth +1.26 [+1.12, +1.39] at W = 1,000; no sign changes.
- **Items:** zero value without trades; at 1.20 the first tier misses the level 6–11 breakpoint; in trading play leads
  last 7–11 turns. A higher `item_mult` raises combat value (and the aggressive trader's edge over passive play,
  +2.30 at 1.35) but does not make affordability matter to Recall.
- **The conditional Shove → Recall → Purchase agent** beats the presser by +15 but loses to passive and trading
  baselines; Shove before Recall was never significantly better than recalling directly.
- **The M2 farm-spot rule:** a keep-reach prototype (experiment only) reproduces the design's Freeze-break table in all
  64 cells and is worth +1.5 to pushing; proposed as a separate rule change.
- Fixes: lifecycle hooks (reset / notify) in the discounted evaluator; an explicit `allow_gamma_mismatch` opt-in.

**Recommended for review (not adopted):** `gank_damage_position = 0.40`, `item_mult = 1.35` (conservative: 0.40 /
1.20). The passive-equilibrium risk remains structural (crash farm, farm-spot rule, item persistence); decide these
before M4.3.

## R. M4.2c experiment: continuous gold investment (2026-10-12)

Branch `experiment/m4-continuous-investment`. An opt-in prototype, `GameConfig.m4_continuous(invest_alpha=…)`: every
return to base invests all unspent gold, AP(L, S) = AP(L, 0) [1 + α ((1 + S)^0.9 − 1)]. **`m4()` (two tiers) is
unchanged and remains the default.** Full account: [m4_continuous_investment_report.md](m4/m4_continuous_investment_report.md);
results: `results/m4_continuous_investment/` (≈204k turns, 11 s wall).

- Mechanically sound: exact ledger without clamp, zero-sum rewards unaffected by spending, mirror symmetry, investment
  kept through death, strictly monotone fixed-point power (guarded at 10^6 resources), baseline throughput.
- Not yet strategically better than two tiers: no detectable Recall-timing value (scripted agents return every
  ~6–12 turns anyway); investment acts only through trades; duel odds stay breakpoint-quantized (knife edges, e.g.
  0.03 resources at L18). Leads make an `edge_trader` concede trades and die more, but pay the leader only against a
  passive opponent.
- Risk: unbounded growth (×4.3 at 120 resources; one 2-HP hit kills at S ≈ 68) within W = 1,000. The observation is
  exact only below 1,024 resources.

**Next (if pursued):** a curve-shape sweep (p 0.5–0.7 or log), the combat quantization review, and a gold-threshold
sweep of the recall agents before any PPO use.

## S. M4.2d-A readiness: wave bounce and Freeze breaking (2026-10-12)

Branch `readiness/m4-2d-wave`. Short deterministic diagnostic: [m4_wave_readiness_report.md](m4/m4_wave_readiness_report.md);
results `results/m4_wave_readiness/` (≈26k turns, ≈12 s wall).

- **No bounce.** After a crash the crasher's follow-up waves park and stack in the defender's half (`stack_creep`)
  and re-crash every ~11 turns. Without a crash the contact point is neutrally stable at every cell. A push is
  permanent; only an opposing Shove moves the lane, and the shover then pays (≈0.11–0.13 farm per turn).
- Shove → Crash → Recall works: a Recall during a 9-HP crash is free, and a mistimed one costs ≈1 resource and
  8 XP. Bounce → Return does not exist.
- **Freeze-breaking artifact:** classified as a mechanical bug. It is fixed by the opt-in
  `GameConfig.m4(farm_spot_rule="pre_crash_front")`, which reproduces the 64-cell design table and meets the SHx
  gate. The default is unchanged.

**Recommendation: BLOCKED** until reviewers choose a restoring-force rule (report §4: proposal A, an unstacked Slow
Push front past the lane walks plus no parking during the own siege; or proposal B, the siege condition only) or
accept permanent pushes.


## T. M4.2d parallel emergent mechanics and integration (2026-10-12)

Four workstreams ran concurrently in separate worktrees from f504f5e. The results were integrated on
`integration/m4-emergent-ready` into one **opt-in** preset, `GameConfig.m4_emergent()`. **`m4()` and `m4_continuous()`
are unchanged.** Full account: [m4_emergent_parallel_report.md](m4/m4_emergent_parallel_report.md); driver
`lanegame.experiments.m4.emergent_ready`; results `results/m4_emergent_ready/` (6,347 turns, 1.5 s wall).

| Workstream | Branch | Outcome |
|---|---|---|
| A natural waves | `prototype/m4-emergent-wave` | **Blocked, not integrated.** Waves walk freely and champions only add damage, but at wave HP 4 fronts never stack, so there are no crashes, plates or bounces and Shove is dominated. Unblocking needs a storage rule or a rescale (W = 8). |
| B positioning | `prototype/m4-positioning` | **Integrated** (`position_rule="standoff"`). Yield farms from a stand-off cell; a Yield that starts in contact with a Trade is pinned. Trades on a stand-off land 6 of 6 instead of 0. No action-space change. |
| C zoning | `prototype/m4-zoning` | **Integrated** (`zoning_rule="none"`). Removes the automatic proximity loss and adds no power term. Removing it alone does not create zoning. |
| D investment + observation | `prototype/m4-investment-readiness` | **Integrated.** Bounded saturating curve `1 + 1.5 (1 − sqrt(16/(16+S)))` (no one-shots, below ×2.5) and a compressed 88-entry observation, lossless through the training horizon. |

**Joint tests: mechanical feasibility.**

- **Waves (scripted):** stacking and crashing work, and the defender can counter them. There is no bounce.
- **Positioning:** a stronger champion can hold forward without attacking. The threat it poses scales with investment
  (549 → 967 expected centi-HP on a pinned stand-off).
- **Power matters:** A's Trade into a contesting B goes from −0.69 to +0.21 per 10 turns as the gap grows.
- **Hold beats Trade** once A has positional control.
- **Recall timing matters**, and returns invest all unspent gold.
- **Correctness passes:** mirror symmetry, masks, replay, gold/XP/minion accounting, invariants, lossless
  observations and zero-sum rewards. Tests: 230 passed, 3 skipped.

**Main caveat: threat-based concession is not rational at current payoffs.**

- Retreating costs the weaker champion more farm than being traded costs it HP.
- The deterministic flip needs A to hold ×3.4–4.0 B's attack power, but the safe investment curve stays below ×2.5.
- Other caveats: no bounce; a passive lane at equal power; weak late leads.

**Decision: GO WITH CAVEATS.** Freeze `m4_emergent()` and proceed to the M4.3 PPO smoke test. Before drawing strategic
conclusions from trained policies, reviewers decide whether to move trade damage or retreat farm retention. The
smallest check is a 2 × 2 rerun of the `zoning` and `hp` stages (about 10k turns).


## U. M4.2e-A natural wave mechanics (2026-10-12)

Branch `development/m4-natural-waves`, from `integration/m4-emergent-ready` 23e48be.

- **Preset:** **opt-in** `GameConfig.m4_natural()`. It is `m4_emergent()` plus `wave_dynamics="natural"` on an
  8-HP wave. `m4()`, `m4_continuous()` and `m4_emergent()` are unchanged, and the historical outputs are
  byte-identical.
- **Full account:** [m4_2e_natural_waves_report.md](natural_waves/m4_2e_natural_waves_report.md).
- **Driver:** `lanegame.experiments.m4.natural_waves`; results `results/m4_natural_waves/` (2.5k turns, 0.4 s).
- **Tests:** `tests/test_m4_natural_waves.py`.

**Rules**

- **One set of minion rules, whatever the actions.** Every front walks, engages on contact, merges and crashes. A
  defender's front engages the crashed mass under its tower; the idle siege wait (D7) is removed. Slow Push parking,
  creeping and the Freeze holds are removed too.
- **Wave slots become champion damage on enemy minions:** Hold 0, Thin 1, Clear 2 HP per tick (slots SP, F, SH). The
  damage needs presence and reach: a connected Trade, a Recall, absence or a Safe retreat deals none.
- **The 46-code action space and the 88-entry observation are unchanged.**

**Timescale fix**

- At 4 HP a contested front dies (2 ticks under Thin) before a reinforcement reaches it (3 ticks after spawning).
- At 8 HP it survives (4 ticks under Thin, 8 under Hold), so reinforcements merge and edges are stored.
- The per-HP economy is rescaled so a wave is still worth 1 resource and 8 XP: farm 0.125/HP, XP 1/HP, gold unit
  1e-7, crash cap 96.
- Geometry, spawn timing and tower damage are unchanged. Raising tower damage shrinks the defender's post-crash
  accumulation and halves plate hits.

**Results** (all from s0, deterministic, no seeded stacks)

- **Neutral:** symmetric perpetual contact at the lane.
- **Slow push:** one turn of Thin → a 24-HP stack → a 16-HP crash at k18.
- **Deliberate crash:** Clear on a built stack crashes at k11 with 19 HP, against k15 and 16 HP without Clear. Clear
  alone never crashes.
- **Bounce:** every crash is followed by the defender pushing into the attacker's half 14–17 turns later and crashing
  back with 16–25 HP. The push–bounce cycle lasts about 38–40 turns.
- **Freeze:** bounded Thin turns an enemy push into a wave held inside one's own tower cell. Holding keeps it for
  14–21 turns with a large farm edge (14.9 against 7.0). The opponent's Thin or Clear breaks it in one turn, with a
  crash.
- **Recall:** 0.46–1.35 resources depending on the enemy crash timing; Recalls after one's own crash are cheap.

**Budget:** ≈ 15.2k turns (target 10k, cap 20k). That includes four runs of the same deterministic driver and a 2k
timing benchmark; the unique experimental content is about 5.4k turns.

**Decision: PASS WITH CAVEATS.**

- The freeze is temporary.
- Clear needs a stack to crash.
- Pushing costs farm against a passive opponent: −2.0 to −4.7 per 40 turns.
- The evaluation bound is analytic (Ω = 107, against 68 enumerated for the scripted lane); an enumeration is
  deferred.
- One cell separates the lane from the tower.

**Next steps (not started):**

1. Verify combat-driven zoning on `m4_natural()`, in the freeze state and in the post-crash siege. Champion reach is
   now wave control, so zoning carries a non-gold cost.
2. Decide the push incentive (enemy-tower collection, plate value, ganks in the enemy half) alongside 1.
3. Enumerate, or accept, the natural lane's evaluation bound.
4. Integrate and freeze the game for the M4.3 PPO smoke test.

## V. M4.2e-A.1 finalizing natural wave mechanics (2026-10-12)

Branch `development/m4-natural-waves`, on top of M4.2e-A.

- **Full account:** [m4_2e_natural_waves_final_review.md](natural_waves/m4_2e_natural_waves_final_review.md).
- **Driver:** `lanegame.experiments.m4.natural_waves_final`; results `results/m4_natural_waves/natural_waves_final.json`
  (≈ 19k turns, 2 s).
- **Tests:** `tests/test_m4_natural_final.py`.
- **Presets:**
  - `GameConfig.m4_natural()` is now the final natural-wave preset.
  - The M4.2e-A prototype is kept as `GameConfig.m4_natural_prototype()`; section U and its report reproduce with it.
  - Every other preset is unchanged.

**Diagnosis**

- **The prototype's long sieges were an artefact.** A follow-up wave entering the enemy tower cell next to a defender
  did not crash; it fought there beside the crashed mass. So:
  - the defender fought the follow-ups, on 13 of 14 ticks;
  - the tower fired at both the mass and the follow-ups;
  - the mass died to tower fire alone (11 of 12 HP) and hit the tower for the whole siege.
- **The bounce is not caused by siege length.** Bounce timing stays fixed while siege length varies 3–6× with tower
  damage.
- **Clear could not crash because of reach, not damage.** At 16 HP per tick it stalled exactly like at 2: defenders
  behind the tower were out of reach.

**Final rules: one rule, one parameter, two companions**

| change | effect |
|---|---|
| Entering the enemy tower cell is the crash | follow-up waves join one siege; one tower target |
| Tower 3 HP per tick | 3 × 4 > 8: the tower alone out-kills the reinforcement stream (1 and 2 give self-sustaining sieges) |
| A champion in the enemy tower cell reaches one cell behind it | Apply Pressure / Vision only |
| Clear = round(4 × AP) HP per turn | Thin stays 2 per turn (controlled), Hold 0 |

W = 8, the `floor(h / 8)` law, the geometry, spawns, the economy and the 46-action / 88-entry interface are unchanged.

**Results** (deterministic, from s0)

- **Sieges** last 3–9 turns with 2–5 minion tower hits (prototype: 6–13 turns, 10–43 hits).
- **Stacked crashes bounce** after 20–22 turns with 24 HP. Small fast crashes start a longer cycle of about 40 turns.
- **Power progression:** from a neutral lane, Clear crashes in 5 turns at AP 1 and 1–2 at AP 4, and smaller (25 → 13–16
  HP). At AP ≥ 2, Clear shoves every fresh wave.
- **Freeze outside tower range** (enemy at −1, own front in its tower cell):
  - held 20+ turns under the freezer's Thin, with a 2.4× farm edge;
  - 13 turns uncontrolled;
  - broken by the opponent's Clear within 2 turns.
- **No action dominates a passive opponent:** a one-turn slow push +4.6 over 80 turns; constant Thin / Clear −10.8 /
  −18.8.
- **Recall:** free right after one's own large crash; 1.7–1.8 during the enemy's siege.
- **Performance** unchanged: lane step 38.7 µs, env step 206 µs.

**Decision: PASS WITH CAVEATS.**

- Push profitability hinges on `collect_contest_enemy_tower` (0.5); at 1.0 push + Pressure becomes profitable (+3 to
  +4.4).
- Plates now come mainly from Pressure.
- Large power gaps allow a perma-shove or perma-siege.
- Freezes need a precise control band and form mainly after an enemy crash.
- One cell separates the lane from the tower.

**Next steps (not started):**

1. Combat-driven zoning on `m4_natural()`: the freeze state and the Pressure siege.
2. Calibrate the push economics there (enemy-tower collection, plate gold, Pressure damage).
3. Enumerate or accept the evaluation bound (per-turn 10.36).
4. Integrate and freeze for PPO.

## W. M4.2e-B design philosophy and combat-driven zoning (2026-10-12)

Branch `development/m4-natural-waves`, on top of M4.2e-A.1.

- **Design philosophy:** [design_philosophy.md](design_philosophy.md), now the standard every mechanic is judged
  against: simplify how League works without changing why its laning strategies work (core principles, intended
  qualitative similarities, accepted simplifications, research and validation standards).
- **Full account:** [m4_2e_zoning_report.md](natural_waves/m4_2e_zoning_report.md).
- **Driver:** `lanegame.experiments.m4.zoning_natural` (reactive scripted policy-matrix games solved as zero-sum
  games: 13 shared policies, 4 states, 5 power gaps, 9 rule variants; 135M turns, 7 min); results
  `results/m4_zoning_natural/`.
- **Tests:** `tests/test_m4_zoning_natural.py`.
- **Candidate preset:** **opt-in** `GameConfig.m4_lane()` = `m4_natural()` plus `collect_contest_enemy_tower 0.9`,
  `collect_contest_own_tower 0.6`, `invest_durability True`. Every other preset is unchanged (random-play digests
  pinned). Also implemented, tested and left off: `trade_exchanges` (one exchange per tick; a trade-fest at equal
  power) and `trade_farm_rule "loser"` (no measurable effect without it).

**Findings**

- Under the A.1 rules power was almost worthless: +0.3 per 32 turns at a x1.76 lead in the neutral lane; pushing
  lost at every gap (-6 pushed, -3 siege); a freeze paid +4.7 at every gap. Two fixed collection rates did it: the
  0.5 rate under the enemy tower (a danger proxy) and the full 0.9 collection of a crashed wave under one's own
  tower (a missing inefficiency).
- The weaker champion never stands back at the lane in any of 180 solved policy games: on this geometry a passive
  concession is almost total (farm reach equals threat reach and the wave does not drift toward a champion that
  stops contesting), while contesting a x1.76 opponent cost 0-4 because trading was a symmetric war of attrition.
  Per-tick exchanges, a two-wave bounty and removing the posture base were tested and rejected (trade-fest at equal
  power; noise; free Safe).
- With `m4_lane()`: every state's value is monotone in power; a one-item lead is worth 2-3 per 32 turns and a large
  lead 4-7 (15-25% of the lane's income); the stronger champion earns it through wave control and attacks on under a
  third of its turns; the equal lane is even and active; pushing when ahead pays. The weaker champion's rational
  concessions are wave control (shove away, or farm the crash under its tower while the stronger champion takes
  plates), position (the stand-off keeps half its farm unhit) and tempo (recalls).

**Decision: PASS WITH CAVEATS.** The stand-at-XP-range concession of League is not produced by any rule tested; the
fix is a design decision (last-hitting as wave damage, or an all-in action) listed for review together with the two
calibrated parameters (own-tower rate 0.6, kill bounty 0.5). No PPO was run; the game is not declared final.

**Next steps (not started):** design review of the hypotheses above; then integrate and freeze for the M4.3 PPO
smoke test on `m4_lane()`.

## X. M4.2e-C wave action redesign and mechanical calibration (2026-10-12)

Branch `development/m4-natural-waves`, on top of M4.2e-B.

- **Full account:** [m4_2e_wave_actions_report.md](natural_waves/m4_2e_wave_actions_report.md).
- **Driver:** `lanegame.experiments.m4.wave_actions` (74k deterministic turns, 8 s; results
  `results/m4_wave_actions/`); sampled policy games through `lanegame.experiments.m4.zoning_natural lasthit` (2.5M
  turns, 1 min; `results/m4_zoning_natural/summary_lasthit.txt`, `trajectories_lasthit.txt`).
- **Tests:** `tests/test_m4_wave_actions.py` (16).
- **Candidate preset:** **opt-in** `GameConfig.m4_lasthit()` = `m4_lane()` plus `wave_clear_scaling "capacity"`
  (`wave_hold_fraction` 1/8, `wave_thin_fraction` 1/2), `damage_law "lanchester_half"` and `thin_damage 0`. Every
  other preset is unchanged (digests pinned).

**The model.** One wave-clear capacity C = round(4 AP) HP per turn; Hold uses an eighth of it (1 HP per turn until AP
2.9, then 2, then 3), Thin half (2 at level 1, 9 at level 18 with four items), Clear all. In the brief's notation
Thin = Hold + 3/7 (Clear - Hold). The minion damage law moves to half-wave resolution (firepower floor(2h / 8) per turn,
split over the ticks), because with last-hit damage on both sides the fronts live in the whole-wave law's flat 8-15 HP
band and a one-turn edge took 19 turns to crash instead of 13; under the finer law it takes 15. Apply Pressure's own
thinning is removed: the wave action is the champion's damage on the defenders (Hold + Pressure with the old thinning
equals Thin + Pressure without it).

**Findings** (deterministic unless stated)

- The neutral lane stays symmetric at every power; a champion three times stronger pushes it by last-hitting alone.
- A wave drifts to a champion that stands back: crash in 10-12 turns against a Safe or recalling opponent; standing
  back costs the Safe champion 3.8 instead of 8.9 per 40 turns.
- Thin scales: from a neutral lane it crashes in 9 / 5 / 3 / 2 turns at AP 1 / 2 / 3 / 4 (before: 7 at every power).
- Sieges 2-6 turns at tower 3 (self-sustaining at 1, as before); crashes 21-27 HP bounce 16-26 turns later with 23-29.
- The outside-tower freeze survives in a narrower band (enemy 12-14 HP against own 8-11): reached after absorbing an
  enemy crash, held 20-30 turns with a 1.7x farm edge under a threshold controller (Thin when the enemy front
  reaches 14), broken by Clear in 1-2 turns; a freezer at AP 3 pushes it out by last-hitting.
- Recall from a neutral lane costs about a plate (3.9) against an opponent that takes plates; after one's own crash
  it is free.
- Policy games (16 seeds): against a stronger champion that only threatens, standing back is now the weaker
  champion's best reply (farm 9.8 of 13, never hit; the bare threat is worth -0.2 to -1.5 to the zoner instead of
  +6.5 to +10.5); the threat converts through wave control (Clear + punish, +6 to +8), so the equilibria keep the
  stronger champion Clearing and the weaker one contesting. Values stay monotone in power; the equal lane is even
  (+0.04, both sides slow-push); the pushed, siege and freeze states are worth +1.4 / +1.9 / +2.1 at equal power
  (before 0.0 / +1.2 / 0.0).

**Decision: PASS WITH CAVEATS.** The freeze band is 3 HP wide and needs an absolute threshold; a champion far ahead
cannot freeze; a crash into one's own tower is still collected at 0.6 and the drift punishes passive opponents that
never take plates; the stand-back concession is a best reply to bare threats, not an equilibrium outcome; scripted
evidence only. Listed for review: the own-tower rate (harness variants `lasthit_ot5` / `lasthit_ot7`) and whether a
16-HP wave is worth its rescale to widen the freeze band.

**Next steps (not started):** review; then the M4.3 PPO smoke test on `m4_lasthit()`, judged against the sampled
equilibria of the report's section 6.4.

## Y. M4.2e-D visual simulation debugger (2026-10-12)

Branch `development/m4-natural-waves`, on top of M4.2e-C. A temporary validation tool, not a mechanics change.

- **Full account:** [m4_2e_debugger.md](natural_waves/m4_2e_debugger.md).
- **Launch:** `pip install -e .[viewer]`, then `streamlit run lanegame/viewer/app.py`.
- **Code:** `lanegame/viewer/` has four modules:
  - `session.py`: setups, recording and replay, global coordinates, summaries, events;
  - `policies.py`: the `zoning_natural` policies with explained fallbacks, and the example matchups;
  - `render.py`: the Plotly figures;
  - `app.py`: the Streamlit app.

  The only engine change is the opt-in `LaneEnv(record_ticks=True)`, which exposes the authoritative lane step's
  tick trace as `info["lane_trace"]`. It is off by default and every existing result is unchanged.
- **Tests:** `tests/test_viewer.py` (23). They cover replay against a plain `LaneEnv`, determinism and branching,
  that the trace is authoritative, B mirroring, the economy identity, legality, that the policies match
  `zoning_natural`, and that every drawn frame follows the states or the tick trace.
- **Delivered:** strategy-vs-strategy matchups (7 examples), manual play for A with branching and take-over, a
  compact lane picture (packets as HP-labelled rectangles, crashed masses, towers with plates, champions, contact and
  movement arrows) with turn and tick playback (Play / Pause, speed, tick-by-tick), player strip and summary line,
  champion and economy panels, timelines, the event log and a grouped `info` inspector. The layout follows the
  reference prototype of the M4.2e-D UI refinement brief.
- **Status:** implemented. M4.3 not started.

## Z. M4.3a RL infrastructure integration and correctness audit (2026-10-12)

Branch `development/m4-3a-rl-infrastructure`, on top of M4.2e-D. No mechanics change; every preset's digest pin is
unchanged.

- **Full account:** [m4_3a_rl_infrastructure_report.md](rl/m4_3a_rl_infrastructure_report.md).
- **Research question made explicit** (README): can an RL agent learn to take advantage of the simulator's strategic
  interactions without being taught League strategies? Rewards stay economic (relative gold); nothing is shaped.
- **PettingZoo.** The Parallel API adapter now takes M4 champion states and `StartState` starts, reports the legal
  sets, converts to AEC, and documents seeding and continuing time. PettingZoo's official API tests pass on every M4
  preset; the adapter's trajectories equal the native `LaneEnv`'s exactly (observations, masks, rewards, flags, turn
  records). Decision: the adapter is the standard external interface; the trainer keeps stepping `LaneEnv` for its
  per-seat bookkeeping, pinned matchups, batched forward passes and pre-reset bootstraps, all now tested directly.
- **PPO audit.** No defect found. New independent references: GAE equals the forward-view λ-return with random
  terminations and truncations; the PPO loss, entropy and parameter gradients equal a re-implementation on
  `torch.distributions.Categorical`; masking is exact in probability and gradient; single-choice steps carry no
  gradient; truncation bootstraps equal the value of the independently replayed pre-reset observation; segment
  bootstraps equal the next segment's first value; bit-identical reproducibility across processes; checkpoints
  carry a verified SHA-256 digest and provenance (format 2). Training histories now record gradient norms, critic and
  return scales, per-seat and truncation counts.
- **Smoke test on `m4_lasthit()`** (`lanegame.experiments.rl.m43a_smoke`, `results/rl/m43a_smoke.md`): 100/0 and
  80/20 shared self-play at 409,600 agent transitions each plus three 102,400-transition reproducibility runs, 5
  minutes in five processes. All 19 checks pass: finite losses and gradients (pre-clip norm ≈ 0.4, max 4), KL
  0.003, explained variance 0.70–0.75, critic on the return scale (|V̂| ≤ 10.7 against Ω = 107, V̂(s0) within ±0.3),
  equal seat counts, one truncation per game and no termination, identical digests for identical seeds. Throughput
  ≈ 5,000 agent transitions/s per process (the 250 µs environment step is 75% of the time). Readout, not a result:
  both policies beat the scripted natural-wave suite by 10–19 resources and the initial network by 12; sp80 beats
  sp100 by 0.8 ± 0.5; both clear 82% of turns and trade 27% with 17 deaths per side per 460 turns; entropy fell
  2.58 → 0.8.
- **Tests:** `tests/test_pettingzoo_m4.py` (11), `tests/test_rl_reference.py` (10), `tests/test_rl_m4_lasthit.py` (6);
  full suite 898 passed, 4 skipped, 2 expected failures in 7 min 33 s.
- **CLI:** `rl-shared --model m4_lasthit` (continuing-time preset), `experiments-rl m43a_smoke`.
- **Open before M4.3b** (report §7): the early entropy collapse and the clear-and-trade policy (entropy-coefficient
  control; audit death economics; judge against the M4.2e-C §6.4 equilibria); stronger evaluation references and a
  continuing-time exploitability attacker; headline window W = 1000; bootstrapping remains an approximation;
  the 80/20 vs 100/0 comparison at pilot scale with seeds; a small hyperparameter grid.

**Status:** complete. **Next:** M4.3b self-play pilot and research runs on `m4_lasthit()`.

## AA. M4.3b PPO self-play pilot and strategic learning diagnostics (2026-10-12)

Branch `development/m4-3b-ppo-pilot`, on top of M4.3a (`main` at 1ad8848). No mechanics, reward, observation or
hyperparameter of the game changed; every preset's digest pin is unchanged.

- **Full account:** [m4_3b_ppo_pilot_report.md](rl/m4_3b_ppo_pilot_report.md); results and figures in
  `results/rl/m43b/` (`report.md` has every table); checkpoints in `results/rl/checkpoints/m43b/` (untracked).
- **Driver:** `lanegame.experiments.rl.m43b` (stages train, curves, headline, exploit, behavior, counterfactual,
  plans, recall, archive, trajectories, figures, report; every evaluation stage resumable);
  `python -m lanegame experiments-rl m43b_pilot`.
- **Grid:** `base` (100% current, entropy 0.01), `ent03` (0.03), `hist80` (80/20, pool 8) and the follow-up `mixed`
  (half the games from an archive of 857 witnessed reachable states), 3 seeds each, 2,048,000 agent transitions per
  run (13 min for nine runs in nine processes). All PPO health checks pass; the baseline's entropy plateaus at
  0.5–0.75 nats from 400k on (no collapse).
- **References:** the reactive policies of the M4.2e-B/C policy games wrapped as environment agents (`farm`,
  `push_punish`, `clear_punish`, `bully_shove`, `freeze_punish`, `respect`, `crash_recall`, `trading_pusher`).
- **Is PPO learning (W = 1000, 32 paired episodes)?** Every final beats every reference by 9–20 resources and its
  untrained network by 12; finals beat their own snapshots at 100k / 400k / 1M by +4 / +1.5 / +0.5; the reference
  values plateau after 100k because the references are dominated. Attackers (warm-started, half the budget) gain
  0.75–2.0: `ent03` 0.88 ± 0.13, `base` 1.42 ± 0.55, `hist80` 1.37 ± 0.14. Cross-play: `ent03` beats `base` by
  0.50 ± 0.22 and is the most reproducible across seeds (seed SD 0.2 against 0.9); `hist80` and `mixed` are even
  with `base`.
- **Exploration.** The 0.03 coefficient doubles the entropy and moves the policy towards Thin (0.13 → 0.21) and
  Trade (0.27 → 0.36) with a small gain on every instrument; Clear stays the dominant wave action (63–75%) under
  both coefficients, with and without historical opponents, and from the archive starts.
- **State dependence.** The behaviour choice is strongly conditioned (Trade at the lane, Safe and Hold at the tower,
  Recall at low HP and after the own crash, Pressure on 76–90% of legal turns); the wave choice is Clear in 60–90%
  of every situation, modulated towards Thin while sieging and Hold at low HP. Single-move counterfactuals: the
  policy's move is within a few tenths of every alternative at every sampled state. Plan-level counterfactuals:
  a 14-turn slow push, pure last-hitting and the 20-turn freeze controller lose 1.1–2.8 resources against the
  policy for every policy; sustained Clear 0.2–0.9. Clear dominance is a best reply to Clear: a game property, not
  an exploration or coverage failure.
- **Recall and death.** Recall is legal on ~60% of turns and taken on 6–8% (12–16 per 460-turn game, 2–4× the
  deaths); 55–80% of investment arrives through Recalls. Counterfactually, Recall pays after the own crash (+0.5
  over staying), is even at low HP, costs 0.3–0.9 elsewhere, and does not depend on unspent gold; a death costs
  1.1–1.3 in full against 0.0–0.55 for a completed Recall. The M4.3a "no Recall" was a schedule artefact (its
  learning rate had annealed before the discovery, which happens between 100k and 400k).
- **Mixed start:** no change on any measure; retired unless a power-gap archive is built (policies see power gaps on
  0.3% of self-play turns).
- **Tests:** `tests/test_rl_m43b.py` (11); full suite 909 passed, 4 skipped, 2 expected failures. **Viewer:** mode
  "Recorded trajectory" replays the exported games.

**Decision:** M4.3b complete. PPO self-play learns strong, locally consistent, state-dependent policies whose
dominant line (Clear with trades and resets) the game's dynamics reward against any opponent that clears. **Next
(M4.3c):** entropy 0.03 as the default; population attackers; the 100/0 vs 80/20 comparison with a wide snapshot
pool; and a design investigation of Clear dominance (kill bounty, gank severity, own-tower collection rate) judged
by the plan-level counterfactual harness against the learned policies, not by scripted play.

## AB. M4.3b follow-up: strategic validation and research-artifact cleanup (2026-10-13)

Branch `development/m4-3b-validation`, on top of `development/m4-3b-ppo-pilot` (98a4341). No mechanics, reward,
hyperparameter or checkpoint changed; no training. Full account: [m4_3b_validation_report.md](rl/m4_3b_validation_report.md);
tables and figures in `results/rl/m43b/validation/`.

- **The confound.** M4.3b §7.2 forced complete actions (Thin + Hold position, Hold + Hold, the freeze controller with
  Hold position), so its 1.5–2.8 losses mixed the wave management with the loss of the policy's adaptive behaviour.
  New: `Plan.transform` (a distribution-level hook of the diagnostics rollout, backwards compatible) and
  `intervene.py`: the wave component is forced on the policy's probability vector, behaviour / objectives / Recall
  stay the policy's (two explicit rules, "marginal" and "conditional", both reported); legality holds without
  fallback because all three waves are legal on every present turn and the rest of the code does not depend on
  the wave (tested on every legal set met in play).
- **Result 1 (sampled states, W = 300, four opponents).** Wave-only Hold, slow push and freeze lose 1.7–2.6 against
  the policy (within 0.2 of the full-action values on the same states and streams), 0.6–1.6 against the pure
  last-hitter, 1.8–3.1 against the reactive slow pusher, 2.3–4.0 against the constant clearer; significant in 16–24
  of 24 states per cell. Wave-only Thin costs 0.2–1.0 (not significant in about half the states); wave-only Clear
  is within 0.1 of the policy. The confound was ≤ 0.2 against the policy and ≤ 0.8 against the scripted opponents,
  in both directions.
- **Result 2 (whole games, W = 1000).** A policy that never Clears while the enemy is present loses 8–13 resources
  per game against every opponent, the passive farmer and the scripted freezer included; forced Thin costs 1–2
  against Holders and 5–6 against clearers (the one opponent-dependent wave interaction); forced Clear ± 0.5.
- **Result 3 (horizon).** The same rollouts read at W = 300 / 460 / 1000: the paired tail beyond 300 is 0.00 ± 0.01
  (max 0.08 over 544 pairs, no sign change). The 300-turn window was sufficient; the guaranteed bound is a bound on
  levels, not on paired differences.
- **Result 4 (freeze states).** Of 88 freeze-like states, half are maintainable freezes against a Holder (held
  10–19 of 20 turns) and a quarter favourable; the freeze loses the same 1.5–4 in every class. Synthetic power gaps
  (constructed starts, labelled): Hold and freeze lose in all 72 cells and lose *more* when ahead against a contesting
  opponent; Thin is marginally positive (+0.2–0.6) only when far ahead at a frozen lane against scripted opponents.
- **Interpretation.** Outcome C of the brief: the tested strategy class is unprofitable under the game as calibrated
  (Clear costs the clearer nothing the reward counts), confirmed with adaptive behaviour, four opponents, the long
  horizon and maintainable freezes. Not an exploration failure; the one gap between what the game rewards and what
  the policies play is the Thin edge when far ahead, which self-play never visits. Outcome D is not excluded for
  strategies that coordinate behaviour differently (a wave-fixed best-response attacker would close it).
- **Artifacts.** JSON results ≥ 1 MB gzipped (readers accept both forms); `results/rl/m43b/MANIFEST.json` (files,
  SHA-256, commit / stage / command, checkpoint digests); a 5.2 MB checkpoint bundle in the git-ignored
  `results/rl/m43b_artifacts/`; `ARTIFACTS.md`. The M4.3b commit is 3.8 MB packed (under 3% of the pack), so **no
  history rewrite is recommended**; a no-force-push replacement-branch recipe is in the report §9 for the user's
  decision.
- **Tests:** `tests/test_rl_m43b_validation.py` (14); diagnostics tests unchanged.

**Decision:** M4.3b conclusions stand with stronger evidence; the plan-level attribution to wave management was
correct. **Next (M4.3c, to be decided together):** the design investigation of Clear dominance with `val_wave_cf`
against the existing finals as the screen for each lever (minutes, no training), training only for levers that move
the wave-only loss towards zero; plus one wave-fixed best-response attacker to close outcome D. No further self-play
campaign on the unchanged game.

## AC. M4.3c design: economic objective, laning termination, mathematical reference (2026-10-13)

Branch `development/m4-3b-validation` (working tree on 26f8c83). **Design only**: no game rule, preset, observation,
reward or checkpoint changed; no training. Full account:
[m4_3c_economic_objective_design.md](m4/m4_3c_economic_objective_design.md); the new living reference of the current
game: [laning_game_mathematical_reference.md](m4/laning_game_mathematical_reference.md); tables in
`results/m43c/report.md`; diagnostics `lanegame/experiments/m43c/` (`python -m lanegame.experiments.m43c all`).

- **Reference (Part A).** One document for `m4_lasthit()`: state, observation, 46 actions and legality, natural wave
  physics with engine-generated examples, collection and XP, combat / ganks / death / Recall, investment and
  durability, towers and the exact reward (a constructed turn with farm, plate and bounty), the RL objective, three
  walkthroughs, glossary; a reconciliation table of ≈30 disagreements between historical documents and the code
  (deprecated rules, outdated numbers, a few genuine discrepancies such as Safe still damaging the wave and one-turn
  kills being possible by combination).
- **Gold + XP (Part B).** The discounted flow objective equals the expected lead at a geometric end of laning, so
  flows and stocks are one objective. Re-pricing the M4.3b wave-only counterfactuals (reproduced exactly) and 261 new
  laning-phase states under gold + λ·XP (λ ≤ 1/4) or level progress changes the sign of nothing: XP accrues at the
  spawn rate whenever a champion is present, so XP mostly prices absence.
- **Termination (Part C).** In self-play the first final plate falls at turn 109–143 (levels 15–18; both level 18 at
  143–146); against references at turn 29–55; two last-hitters never finish a tower. Pressure supplies 95–100% of
  tower damage, so the end is the attacker's choice. 79–81% of the M4.3b validation states lie after a tower's last
  plate; 40–68% of the headline margins over references are earned before the proposed end. Termination creates a
  "prolong when ahead" incentive (first sign: withholding wave damage turns slightly positive against a passive
  opponent once games can end).
- **Freeze and death (Part D).** Walking up to a held freeze costs nothing over 40 turns (a delay, paid in HP);
  stepping back costs 4.9 gold; breaking it pays (+4.8 for the learned policy). The causes are cheap HP (death and
  Recall refill it) and zoning only through trades, not wave mechanics or XP. A death costs 0.7 / 1.4 / 1.25 (early /
  mid / late), 1.2–1.5 above a Recall; each extra absent turn costs 0.4–0.6 because lane income does not grow with
  level.

**Decision (pending review):** proposed objective = incremental gold + (1/8)·credited XP, terminal at either tower's
last plate, γ = 0.99 kept as the laning hazard, no fallback horizon, death timers unchanged; game-length calibration
(thresholds and γ) to be decided with the screens before any training. Stopped before implementation and training.
