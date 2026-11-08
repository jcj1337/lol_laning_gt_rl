# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research codebase that models League of Legends laning as a two-player zero-sum stochastic game. Phase 1 solves a
wave-management game exactly (equilibria, best responses, exploitability). From M2 on, champions and an integrated
economy make the game unsolvable, and shared-policy PPO self-play trains on it; the final campaign (M5) compared
Bayesian opponent modelling with pure self-play. The reward is gold difference only; no strategy is scripted. M6 is a
web demo of the M5 agents. Read `README.md` for the framing, `docs/findings.md` for the results and
`docs/archive/history.md` for how the project got here.

The repository was reorganized on 2026-11-08 (`docs/archive/MOVES.md`). The pre-reorganization tree, with every
removed study package and raw result, is commit `f613f0a0`.

## Commands

The venv is `.venv/` (Python 3.13, CPU torch), with the project installed editable (it provides both `lanegame` and
`experiments`). Run commands from the repository root: paths to `artifacts/`, `runs/` and `docs/` are relative.

```bash
pip install -e ".[test,rl,demo]"       # demo = the web demo's API
python -m pytest                       # full suite, ~1,500 tests, ~9-12 min
python -m pytest tests/test_m4_env.py                  # one file
python -m pytest tests/test_m4_env.py::test_name       # one test (or -k pattern)
python -m pytest --runslow             # also run @pytest.mark.slow (exhaustive enumerations, the K=32 re-solve)
cd web && npm install && npm run build && npm test     # the frontend (Node 22.12 or later)
```

There is no linter or formatter configured. Tests needing torch, pettingzoo or fastapi skip themselves when those
packages are missing. Tests that need git-ignored checkpoints (`runs/checkpoints/`) skip when the files are absent;
the 7 demo networks are tracked in `artifacts/checkpoints/` and always available.

- Library CLI: `python -m lanegame <command>` (the module docstring of `lanegame/__main__.py` lists them): `solve`,
  `scenarios`, `exploit`, `simulate`, `reach`, `strategy` (exact game, `--model siege|revised|v1`), `env-demo`
  (scripted agents through the env), `rl-train`, `rl-selfplay`, `rl-shared --model <preset>`, `rl-eval`, `rl-br`.
- Studies: `python -m experiments.<study> <stage> [<stage> ...] [--workers N]`, for example
  `python -m experiments.m5 report`. `experiments/README.md` lists the runnable studies (`m5`, `nc1`, `siege`,
  `reference`, `m6a` to `m6d`) and their stages.
- Expensive stages (training campaigns, batteries, attackers) refuse to start without a confirm flag such as
  `--confirm-m5-train` or `--confirm-nc1-train`. That flag means the user approved the compute budget, so never pass it
  on your own initiative. Long stages are resumable and write partial results under `runs/`.
- The demo: `python -m lanegame.demo` serves the API and the built frontend on 127.0.0.1:8765.

## Layout

```text
lanegame/          the library
  engine/          GameConfig and presets, state, mechanics, the exact game layer, champion resolvers, progression
  exact/           solver, matrix games, best responses, policies, baselines, analysis, simulation, reachability
  env/             LaneEnv, observations, scripted agents, starts, traces, the PettingZoo adapter
  rl/              network, PPO, the collector and continuing-time trainer, shared-policy self-play, checkpoints
  opponent/        OM2's opponent agents and the Bayesian belief
  diagnostics/     action audits, paired counterfactual branches, the lane-exit readout
  viewer/          the viewer recording format (session.py); the Streamlit UI was retired
  demo/            the web demo's backend: registry, sessions, records, frames, actions, guides, service, API, worker
  paths.py         artifacts/ and runs/, the shipped-checkpoint fallback, the mapping of pre-2026-11-08 paths
  io.py            JSON helpers and the solved-game format
experiments/       one package per study (experiments/README.md)
tests/             the test suite; m4_scenarios.py and data/ are test support
web/               the demo's frontend (React, TypeScript, Vite)
docs/              findings, model, methodology, research overview, design philosophy; archive/ for every original report
artifacts/         curated, tracked records of every study and the 7 demo networks (artifacts/README.md)
runs/              git-ignored: everything else runs write, above all every checkpoint (runs/checkpoints/<study>/)
legacy/            the original Q-learning prototype, kept unchanged for reference
```

## Architecture

**One lane engine, several game layers.** Every ruleset is selected by a `GameConfig` (frozen dataclass,
`lanegame/engine/config.py`):

- `engine/mechanics.py` is the deterministic wave step (`step_turn`: packet movement, merging, fights, crashes, tower
  fire, spawns, plates). The only randomness in the game is ganks and trade or tower rolls.
- Exact game (Phase 1): `engine/game.py` (`LaneGame`) adds legality, farm, expected gank cost and reachable layers on
  top of `mechanics`. `exact/solver.py` runs backward induction with `exact/matrix_game.py` (saddle detection,
  otherwise a HiGHS LP), and `exact/best_response.py` computes exact best responses and exploitability. Presets are
  `GameConfig()` (siege, the default), `.v2()` and `.v1()`.
- Champion games are played only through the env, never solved. `engine/champions.py` is the M2 resolver
  (`GameConfig.champions()`) and holds definitions the later resolver shares. `engine/champions_m4.py`
  (`resolve_turn_m4`) is a separately versioned resolver for every `ruleset="m4"` preset; its module docstring gives
  the numbered order of turn resolution. `engine/progression.py` holds the integer level and item tables.
  `engine/positional.py` is the M4.5 movement graph (lane cells plus the shared Vision node `WARD`).
- `env/core.py` (`LaneEnv`) is the simultaneous-move environment. It dispatches on `cfg.is_m4` and
  `cfg.champion_mode`, with matching observation modules (`observation.py`, `observation_m4.py`) and scripted agents
  (`agents.py`, `agents_m4.py`). `pettingzoo_adapter.py` wraps it in the Parallel API. M4 rulesets run in continuing
  time: there is no terminal state, episodes end only by truncation, and training bootstraps the truncation. The end
  of the lane phase (about level 14, turn 94) is an evaluation checkpoint, never a terminal state (decision of
  2026-10-17).
- `rl/`: `network.py` (masked policy/value MLP; the opt-in `SeparateCriticNet`; `widen_inputs`, `widen_hidden`,
  `StrengthFeatureNet`), `ppo.py` (GAE plus the clipped update), `train.py` (collector, continuing-time `GameTimes`),
  `shared.py` (shared-policy self-play, the trainer of every M4 and M5 campaign), `selfplay.py` (the snapshot pool).
  `SharedCollector` has opt-in hooks (`extra_obs` / `augment`, `observe_step`) for policy inputs computed outside the
  game, such as OM2's belief, and `train_shared(on_update=...)` passes each update's history entry to a read-only
  callback (M5's health monitor). With the defaults, training is bit-identical (pinned in `tests/test_om2.py` and
  `tests/test_m5_corrections.py`). `agent.py` has `NetAgent` and checkpoints, which carry a SHA-256 weights digest
  verified on load; `load_checkpoint` resolves a missing `runs/checkpoints/` path to its shipped copy.
- `opponent/` (OM2): opponent agents (scripted, network, per-turn mixture, style switcher) and the Bayesian belief over
  a library of opponent types (`Belief` with forgetting, batched `loglik`, the catch-alls "mixed" and "unknown").
- `diagnostics/`: action audits (`audit.py`), paired counterfactual rollouts from restorable states (`branch.py`) and
  the lane-exit readout (`lane_exit.py`). The readout is evaluation only: E_h = the earned-gold lead now (invested +
  unspent) plus its expected change over h continuation turns, undiscounted, with no critic tail. XP, HP, absence and
  the wave are reported beside it, never folded into it.
- Library and experiments. `lanegame` does not import `experiments`, with three exceptions: `demo/` and
  `viewer/session.py` (they play M5's networks and its research recordings) and the lazy import of scripted
  strategies in `opponent/types.py`.

**Experiments.** `experiments/m5/` is the final campaign. Arm A (opponent modelling, OM2's treatment collector)
against arm B (pure self-play); both continue `ext/baseline:s`, seeds 70,000–70,999.

- Its design, pre-registration and amendment are in `docs/archive/m5/`. The pre-registration has a verifiable
  amendment chain (`common.amend`, `artifacts/m5/preregistration.json`, every version archived under
  `artifacts/m5/prereg_versions/`). Every campaign stage runs only when the document is the chain's active version.
  The record names documents and checkpoints by their 2026-10 paths; `common.resolve` translates them
  (`lanegame.paths.relocated`, then the shipped-copy fallback). Never edit the pre-registration documents: their
  SHA-256 is the chain.
- The campaign stages need `--confirm-m5-train`, `--confirm-m5-eval` or `--confirm-m5-attackers`. Every evaluation
  stage refuses to start unless `train_check` passed or each failed check has a `train_resolve` record, and it
  verifies the checkpoints' SHA-256. `analysis.py` classifies an endpoint only with all six seeds; `strategic.py` is
  S8; `battery_extend` re-runs failed horizon validations at 150 turns; `reproduce` writes `reproducibility.md` with
  the SHA-256 of every network and record.
- History-dependent policies in the instruments: the shared rollout loops (`m45b.engine.play`,
  `m45c.horizon.rollout`, `m46a.engine.rollout`, `lanegame.diagnostics.lane_exit.continuation`) call an opt-in hook
  after every turn, `observe_turn(cfg, lane, m4, k, codes)`. A stateful policy's `decide` must be pure, since the
  instruments call it outside the loop. `experiments/m5/players.py` has `OMPolicy` and `crn_policy`.
- The other packages named after earlier studies (`m4`, `m43b` to `m46f`, `om1`, `om2`) are only the parts the
  final campaign, the demo or the tests import, reused by import across studies. Their CLIs, report generators and
  remaining stages are at `f613f0a0`. Don't copy code between them; import it.
- `experiments/nc1/` is the network capacity study (2026-10-29, experimental, the user's review open).

**The demo (M6).** `lanegame/demo/` is the session core and `web/` the frontend ([archive/m6/](docs/archive/m6/)
has the plan and reports). M6 adds no mechanics, rewards or training.

- `registry.py` checks the 7 checkpoint files in `artifacts/checkpoints/` against `manifest.json` before any is
  unpickled. Never load a demo checkpoint another way. `service.py` is the JSON layer: one lock around all simulation
  in a process, sessions that expire after 30 idle minutes, the three seed-0 research replays of
  `artifacts/m5/traces/` verified at startup. `worker.py` runs Watch simulations in a separate spawned process; never
  simulate on a second thread of the serving process (the M5 modules' caches are unlocked).
- `session.py`: bot seats decide before the human's code is accepted; submissions are idempotent by turn number; a
  drawn chance seed and the policy seed stay secret until the session stops. `record.py` has state digests and
  replay. `frames.py` and `actions.py` give the frames and the menu, with reasons from the legality predicates. Every
  turn frame carries `moves` (`frames.moves`, checked by `tests/test_demo_moves.py`); a tick's `packets_lost` is
  rebuilt from the trace (`frames.packets_lost`, checked by `tests/test_demo_narration.py`).
- A Sequence (the code says guide) is one JSON file in `lanegame/demo/guides/`: a verified research replay, a start
  position and three to five events (`checkpoints`). `guides.py` validates it and the service checks it against the
  replay at startup. A new guide gets a test like `tests/test_demo_guides.py`.
- `web/openapi.json` and `web/src/api/schema.d.ts` are generated from `lanegame/demo/schemas.py`
  (`python -m lanegame.demo --openapi web/openapi.json`, then `npm run types`); `tests/test_demo_api.py` fails when the
  committed schema drifts. The browser computes no game rule; its words restate facts the frames carry
  (`web/src/game/words.test.ts`).
- Verification: `python -m experiments.m6a verify measure`, `python -m experiments.m6b browser` (the visitor journey
  in headless Edge), `python -m experiments.m6c browser`, `python -m experiments.m6d overview` (every number in
  `docs/research_overview.md`; a number added there needs a check in `m6d/overview_check.py`).
- Open M6 items (user decisions or not started): the reproduction guide and bundles, a `reproduce` re-run on a clean
  commit (audit R1), scripted opponents in the demo, a license, M6e deployment.

**The preset chain.** M4 presets build on each other, each as the previous preset plus opt-in fields:
`m4 → m4_continuous → m4_emergent → m4_natural_prototype → m4_natural → m4_lane → m4_lasthit → m4_four_actions →
m4_four_actions_econ → m4_four_actions_econ_aggro → m45_unified → m45_calibrated`. The research preset is
`m45_calibrated()`. Candidate variants are keyword overrides rather than new presets; M4.6's "C1", for example, is
`m45_calibrated(invest_gain=3.24, invest_scale=64.0)`. Rules for changing mechanics:

- A new rule goes in as a new `GameConfig` field whose default reproduces the old behaviour exactly. Turn it on in a
  new preset or an override, record it in that preset's `*_FIELDS` tuple, and validate it in `_validate_ruleset` /
  `_validate_m45`. Never change an existing default or an existing preset.
- Earlier presets are guarded by digest pins: `tests/test_legacy_pins.py` (v1, v2, siege, champions and solver values),
  `test_historical_presets_are_pinned` in `test_m45a_mechanics.py` and the pins in `test_m45b_candidate.py`. A failing
  pin is a regression. Never update the digest to make it pass.
- The action space has 61 codes from `m4_four_actions()` on (Slow Push is 46–60, Ward is 30; earlier M4 presets have
  46). The observation has 88 entries, or 90 with the M4.5 Vision node; the opt-in `obs_last_action` appends 13 (103
  with `m45_calibrated()`, used by OM2). A change to the observation or its scales invalidates trained checkpoints.

**Determinism is a tested contract.** The env draws a fixed number of chance uniforms per turn, whatever the actions,
so that policies are compared on common random numbers. Same seed plus same actions gives a bit-identical rollout. The
lane-step cache is an exact memo. Training runs reproduce bit for bit, and tests check all of this. Don't introduce
action-dependent RNG consumption or unseeded randomness.

**Data.** A study writes curated summaries to `artifacts/<study>/` and everything else to `runs/` (`CKPT` is
`runs/checkpoints/<study>/`). Records are never rewritten by hand; re-running a deterministic stage reproduces its
file byte for byte. Records written before 2026-11-08 name files as `results/...` and `docs/<milestone>/...`;
`lanegame.paths.relocated` maps them. A new study writes to `runs/` and promotes only what it publishes.

## Documentation conventions

- `docs/model.md` is the living description of the game for `m45_calibrated()`. Any change to a rule, a parameter or
  the objective updates the matching equation, table or example there in the same piece of work and adds a line to
  its §13 change log. Its engine-computed examples come from `python -m experiments.reference`
  (`artifacts/reference/`), and `tests/test_reference_examples.py` pins the deterministic ones. The version that
  described `m4_lasthit()` is `docs/archive/m4/laning_game_mathematical_reference_m4_lasthit.md`; reports written
  before 2026-10-20 cite its section numbers.
- `docs/findings.md`, `docs/methodology.md`, `docs/research_overview.md` and `docs/archive/history.md` are the living
  summaries. Keep evidence, interpretation and decisions apart, and label status (approved direction, proposed,
  completed, unresolved decision).
- Everything in `docs/archive/` is a historical record, byte-identical to the original. Don't edit it, not even its
  links. Corrections go into a later document, `docs/archive/history.md` §6 and the model's §12.
- A study produces a pre-registration (written before training or evaluation), a report and generated tables in
  `artifacts/<study>/`. Add the report to `docs/archive/README.md` and its findings to `docs/findings.md`.
- `README.md` is for general visitors: short, plain language, no status log.
- `docs/design_philosophy.md` is the standard for any mechanic: simplify how League works without changing why its
  laning strategies work.
- Terminology: the midpoint cell x = 0 is "the lane" (never "river"). The full-capacity wave slot is "Shove" in the
  four-action presets and "Clear" in `m4_lasthit()` and earlier; it is the same code. The reward term paid for a
  champion's death is "kills" (never "bounty"); the config field `kill_bounty` and the `info` keys `bounty` /
  `bounty_units` keep the old word. In visitor-facing demo text: Sequences and events, units (not cells), "back to
  base" (never "home").
- `legacy/` is the original Q-learning prototype, kept unchanged for reference.
