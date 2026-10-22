# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research codebase that models League of Legends laning as a two-player zero-sum stochastic game. Phase 1 solves a
wave-management game exactly (equilibria, best responses, exploitability). M2 onward adds champions and an integrated
economy that cannot be solved exactly, and trains shared-policy PPO self-play on it. The reward is gold difference
only; no strategy is scripted. Read `README.md` for the research framing and `docs/rl_research_roadmap.md` (summary
at the top) for where the project currently stands.

## Commands

The venv is `.venv/` (Python 3.13, CPU torch), with `lanegame` installed editable. Packages are auto-discovered
(`lanegame*` with an `__init__.py`), so a new milestone package needs no `pyproject.toml` entry. Run commands from the
repo root anyway: experiment drivers write to relative `results/` paths.

```bash
pip install -e ".[test,rl,viewer,demo]"   # viewer = the Streamlit debugger, demo = the M6 web demo's API
python -m pytest                       # full suite, ~1,600 tests, ~9-12 min
python -m pytest tests/test_m4_env.py                  # one file
python -m pytest tests/test_m4_env.py::test_name       # one test (or -k pattern)
python -m pytest --runslow             # also run @pytest.mark.slow (exhaustive enumerations, K=32 re-solve)
```

There is no linter or formatter configured. Tests needing torch, pettingzoo or plotly skip themselves when those
packages are missing. Tests that need git-ignored checkpoints (`results/rl/checkpoints/`, `*.pt`) skip when the files
are absent.

Main CLI: `python -m lanegame <command>`. The module docstring of `lanegame/__main__.py` lists every command. The most
common are `solve` / `scenarios` / `exploit` (exact game, `--model siege|revised|v1`), `env-demo` (scripted agents
through the env) and `rl-shared --model <preset>` (shared-policy PPO; it accepts presets up to
`m4_four_actions_econ_aggro()`, and the M4.5 and later campaigns train through their milestone drivers).

Each milestone's experiments run as `python -m lanegame.experiments[.rl].<milestone> <stage> [<stage> ...] [--workers N]`,
for example `python -m lanegame.experiments.m45b validate`. The stage list is in that package's `__main__.py`
docstring. Most expensive stages (training campaigns, batteries, attackers) refuse to start without a confirm flag
such as `--confirm-m44b`, `--confirm-m45c-full` or `--confirm-m5-train`. That flag means the user approved the compute
budget, so never pass it on your own initiative. A few earlier training stages (`m46a e4_train`, `m46c train`,
`m46f train`) have no flag and are gated only by their pre-registration; ask before running those too. Long stages
are resumable and write partial results under `results/<milestone>/.../partial/`.

Viewer: `streamlit run lanegame/viewer/app.py` (replays recorded trajectories from actions).

## Architecture

**One lane engine, several game layers.** Every ruleset is selected by a `GameConfig` (frozen dataclass,
`lanegame/config.py`):

- `mechanics.py` is the deterministic wave step (`step_turn`: packet movement, merging, fights, crashes, tower fire,
  spawns, plates). The only randomness in the game is ganks and trade or tower rolls.
- Exact game (Phase 1): `game.LaneGame` adds legality, farm and expected gank cost, and reachable layers on top of
  `mechanics`. `solver.py` runs backward induction with `matrix_game.py` (saddle detection, otherwise a HiGHS LP),
  and `best_response.py` computes exact best responses and exploitability. Presets are `GameConfig()` (siege, the
  default), `.v2()` and `.v1()`.
- Champion games are played only through the env, never solved. `champions.py` is the M2 resolver
  (`GameConfig.champions()`). `champions_m4.py` (`resolve_turn_m4`) is a separately versioned resolver for every
  `ruleset="m4"` preset; its module docstring gives the numbered order of turn resolution. `progression.py` holds the
  integer level and item tables. `positional.py` is the M4.5 movement graph (lane cells plus the shared Vision node
  `WARD`).
- `env/core.py` (`LaneEnv`) is the simultaneous-move environment. It dispatches on `cfg.is_m4` and
  `cfg.champion_mode`, with matching observation modules (`observation.py`, `observation_m4.py`) and scripted agents
  (`agents.py`, `agents_m4.py`). `pettingzoo_adapter.py` wraps it in the Parallel API. M4 rulesets run in continuing
  time: there is no terminal state, episodes end only by truncation, and training bootstraps the truncation. The end of
  the lane phase (about level 14, turn 94) is an evaluation checkpoint, never a terminal state (decision of 2026-10-17).
- `rl/`: `network.py` (masked policy/value MLP; the opt-in `SeparateCriticNet` gives the critic its own trunk, M4.6f;
  `widen_inputs` adds zero-weight inputs), `ppo.py` (GAE plus the clipped update), `train.py` (collector,
  continuing-time `GameTimes`) and `shared.py` (shared-policy self-play: one network plays both seats against a
  current/historical mixture; this is the trainer every M4 campaign uses). `SharedCollector` has opt-in hooks
  (`extra_obs` / `augment`, `observe_step`) for policy inputs computed outside the game, such as OM2's belief; with the
  defaults, training is bit-identical (pinned in `tests/test_om2.py`). `train_shared(on_update=...)` (M5) passes each
  update's history entry to a read-only callback (M5's runtime health monitor); a monitored run is bit-identical (pinned
  in `tests/test_m5_corrections.py`). `agent.py` has `NetAgent` and checkpoints,
  which carry a SHA-256 weights digest verified on load. `evaluate.py` covers head-to-head and approximate best
  responses.
- `opponent/` (OM2): opponent agents (scripted, network, per-turn mixture, style switcher) and the Bayesian belief over
  a library of opponent types (`Belief` with forgetting, batched `loglik`, the catch-alls "mixed" and "unknown").
- `diagnostics/` holds action audits (`audit.py`), paired counterfactual rollouts from restorable states
  (`branch.py`) and the lane-exit readout (`lane_exit.py`, M4.8). The readout is evaluation only: E_h = the
  earned-gold lead now (invested + unspent) plus its expected change over h continuation turns, undiscounted, with no
  critic tail. XP, HP, absence and the wave are reported beside it, never folded into it.
- History-dependent policies in the instruments (M5). The shared rollout loops call an opt-in hook after every turn:
  `observe_turn(cfg, lane, m4, k, codes)` (canonical in `diagnostics/lane_exit.py`). The loops are
  `m45b.engine.play`, `m45c.horizon.rollout`, `m46a.engine.rollout` and `lane_exit.continuation`.
  - Plan wrappers forward the hook through `inner_policies()`. `horizon.critic_value` asks such a policy for
    `value_at`.
  - Stateless policies are untouched: stored M4.5c, M4.6c and M4.6f records reproduce exactly (`tests/test_m5.py`).
  - A stateful policy's `decide` must be pure, since the instruments call it outside the loop.
  - `experiments/rl/m5/players.py` has `OMPolicy`, an OM2 network with its belief and the opponent's previous codes,
    and `crn_policy` for any input width (90, 103 or 116).
- `experiments/` has one package per milestone, writing to `results/<milestone>/`. Shared code is reused across
  milestones by import (for example, M4.6c builds on m45c and m46b), not by copying.
- `experiments/rl/m5/` is the final campaign: arm A (opponent modelling, the OM2 treatment collector) against arm B
  (pure self-play). Both continue `ext/baseline:s`; seeds are 70,000–70,999.
  - Its design, pre-registration and amendments are in `docs/m5/`.
    - The pre-registration has a verifiable amendment chain (`common.amend`, `results/m5/preregistration.json`, every
      version archived under `results/m5/prereg_versions/`).
    - Every campaign stage runs only when the document is the chain's active version.
    - To change the protocol: `prereg_archive`, then edit, then `amend --reason ... --amendment-doc ...`.
  - M5a readiness is `python -m lanegame.experiments.rl.m5 ready`; the correction pass's checks are `corrections`.
  - The campaign stages need `--confirm-m5-train`, `--confirm-m5-eval` or `--confirm-m5-attackers` (the optional
    om_table ablation also `--include-om-table`).
  - Training runs under `health.py`:
    - live stop conditions: defect alarms end a run, investigate alarms are recorded and block evaluation;
    - the coordinator's rule: more than two failed pairs stop the campaign;
    - a progress log in `results/m5/progress_main.json`.
  - Every evaluation stage refuses to start unless `train_check` passed or each failed check has a `train_resolve`
    record, and it verifies the checkpoints' SHA-256.
  - `analysis.py` classifies an endpoint only with all six seeds. `strategic.py` is S8 (battery and cell replication).
  - The `battery_extend` stage runs M4.5c's exception 3 after `battery`: every (final, family) that fails horizon
    validation is re-run at 150 turns, and S8 then reads the 150-turn statistic for it.
  - The `attackers` stage follows the training's rules (the runtime monitor, the coordinator, `--restart-incomplete`
    with a prefix check) and ends with `attack_check`. `attack_eval` refuses to start until that check passed or each
    failure has a resolution.
  - `traces` exports viewer recordings of selected evaluation games, `attack_behaviour` summarizes the attackers'
    actions, and `reproduce` writes `results/m5/reproducibility.md` with the SHA-256 of every network and record.
  - M5 is complete (2026-10-19, [final report](docs/m5/m5e_closeout_report.md)). M6, presentation, demonstration
    and reproducibility, follows.
- M6 ([plan](docs/m6/m6_plan.md)) is a new web application with two modes: Play (a human against `sp:0` or `om:0`)
  and Watch (a bounded bot match explored through a timeline). M6 adds no mechanics, rewards or training. The
  Streamlit viewer stays where it is until M6d archives its UI, and it is not a visual template.
  - M6a (complete, 2026-10-21) built the session core, and the user approved the glass interface design
    ([decision record](docs/m6/m6a_decision_record.md) §7). M6b (complete, 2026-10-22,
    [report](docs/m6/m6b_report.md)) built the maintained application from it. The prototypes are gone: the approved
    prototype's code is archived unchanged at `legacy/m6a_glass_prototype/` (it does not run) and its screenshots
    stay in `results/m6/prototypes/`. M6c was narrowed on 2026-10-22 to a small collection of optional guided
    replays within Watch, shown to visitors as Sequences ([report](docs/m6/m6c_guided_replays.md)); there is no findings page, and the research
    explanation belongs in M6d's documentation. Its step 1 (the format and the first example) is done and awaits the
    user's review; steps 2 (review) and 3 (a few more examples) follow.
  - A sequence is one JSON file in `lanegame/demo/guides/` (the code calls it a guide; the first,
    `stacking-to-crashing.json`, is the template): a
    verified research replay, a start position and three to five events (the JSON field is `checkpoints`) whose
    notes form a chain, each grounded in a rule of the engine, with an optional highlight. In a guide, Next event and
    Play walk the turns to the next event without tick frames and stop there. `guides.py` validates the file at the boundary and the service checks every position and highlight
    against the verified replay at startup, so a bad guide stops the server with a message naming the file and the
    rule. Position t is the state after t turns. `tests/test_demo_guides.py` pins the first guide's numbers to the
    recording; a new guide gets the same kind of test. The guide state lives in the Watch store
    (`web/src/game/store.ts`, panel `web/src/ui/Guide.tsx`); it never appears in a live game.
  - The application is `web/` (React, TypeScript, Vite; `cd web && npm install && npm run build`), served by
    `python -m lanegame.demo` on 127.0.0.1:8765 (`lanegame/demo/api.py`, FastAPI, the `demo` extra). `npm run dev`
    in `web/` proxies `/api` to it. `web/openapi.json` and `web/src/api/schema.d.ts` are generated from
    `lanegame/demo/schemas.py` (`python -m lanegame.demo --openapi web/openapi.json`, then `npm run types`);
    `tests/test_demo_api.py` fails when the committed schema drifts. The browser computes no game rule.
  - `lanegame/demo/` is the headless session core ([technical report](docs/m6/m6a_technical_report.md)). `registry.py`
    checks the 7 checkpoint files against `manifest.json` before any is unpickled. Never load a demo checkpoint
    another way. `service.py` is the JSON layer: one lock around all simulation in a process, sessions that expire
    after 30 idle minutes, the three seed-0 research replays verified at startup. `worker.py` runs Watch simulations in
    a separate spawned process with its own registry and caches; never simulate on a second thread of the serving
    process (the M5 modules' caches are unlocked).
  - `session.py`: bot seats decide before the human's code is accepted; submissions are idempotent by turn number; a
    drawn chance seed and the policy seed stay secret until the session stops. `record.py` has state digests and
    replay. `frames.py` and `actions.py` give the frames and the menu, with reasons from the legality predicates.
    Every turn frame carries `moves`, each seat's champion move and leading-wave move, which the lane draws as
    arrows; `frames.moves` holds the rule (a wave move only while the turn's leading packet survives) and
    `tests/test_demo_moves.py` checks it against every turn of the three replays.
  - The frontend keeps the resolved turns and the viewed position apart (`web/src/game/controller.ts`), tags every
    request with a generation so a late response from a replaced game is dropped, and re-reads the server after an
    unanswered submission before it offers a retry (`web/src/game/store.ts`).
  - `python -m lanegame.experiments.m6a verify measure` re-checks the research recordings and measures the costs.
    `python -m lanegame.experiments.m6b browser` drives the built app through the whole visitor journey in headless
    Edge (evidence in `results/m6/m6b_browser.json` and `results/m6/m6b/`), and `python -m lanegame.experiments.m6c
    browser` does the same for the sequences (`results/m6/m6c_browser.json`, `results/m6/m6c/`).
    `tests/test_demo_session.py` and `tests/test_demo_api.py` cover the correctness rules and the API; `npm test`
    in `web/` covers the history, stale-response and guide rules.

**The preset chain.** M4 presets build on each other, each as the previous preset plus opt-in fields:
`m4 → m4_continuous → m4_emergent → m4_natural_prototype → m4_natural → m4_lane → m4_lasthit → m4_four_actions →
m4_four_actions_econ → m4_four_actions_econ_aggro → m45_unified → m45_calibrated`. The current research preset is
`m45_calibrated()`. Candidate variants are passed as keyword overrides rather than new presets; M4.6's "C1", for
example, is `m45_calibrated(invest_gain=3.24, invest_scale=64.0)`. Rules for changing mechanics:

- A new rule goes in as a new `GameConfig` field whose default reproduces the old behaviour exactly. Turn it on in a
  new preset or an override, record it in that preset's `*_FIELDS` tuple, and validate it in `_validate_ruleset` /
  `_validate_m45`. Never change an existing default or an existing preset.
- Earlier presets are guarded by digest pins: `tests/test_legacy_pins.py` (v1, v2, siege, champions and solver values),
  plus `test_historical_presets_are_pinned` in `test_m45a_mechanics.py` and the pins in `test_m45b_candidate.py`. A
  failing pin is a regression. Never update the digest to make it pass.
- The action space has 61 codes from `m4_four_actions()` on (Slow Push is 46–60, Ward is 30; earlier M4 presets have
  46). The observation has 88 entries, or 90 with the M4.5 Vision node; the opt-in field `obs_last_action` appends 13 entries for the opponent's previous action (103 with
  `m45_calibrated()`, used by OM2). A change to the observation or to its scales invalidates trained checkpoints.

**Determinism is a tested contract.** The env draws a fixed number of chance uniforms per turn, whatever the actions,
so that policies are compared on common random numbers. Same seed plus same actions gives a bit-identical rollout. The
lane-step cache is an exact memo. Training runs reproduce bit for bit, and tests check all of this. Don't introduce
action-dependent RNG consumption or unseeded randomness.

## Documentation conventions

- `docs/m4/laning_game_mathematical_reference.md` is the living description of the game, written for the research
  preset `m45_calibrated()`. Any change to a rule, a parameter or the objective must update the matching equation,
  table or example there in the same piece of work and add a line to its §13 change log. Its engine-computed
  examples and measurements come from `python -m lanegame.experiments.reference` (results/reference/), and
  `tests/test_reference_examples.py` pins the deterministic ones. The version before the 2026-10-20 rewrite, which
  described `m4_lasthit()`, is archived as `docs/m4/laning_game_mathematical_reference_m4_lasthit.md`; earlier
  reports cite its section numbers.
- Milestone reports (`docs/phase1/`, `docs/rl/`, `docs/m4/m4_X/`, `docs/natural_waves/`, `docs/opponent_modelling/`,
  `docs/m5/`) are historical records. Don't edit them after the fact; corrections go in later reports and in the
  reference's §12.
- A milestone usually produces a pre-registration (written before training or evaluation), a report under `docs/`, and
  generated tables at `results/<milestone>/report.md`. It also updates the roadmap summary,
  `docs/README.md` / `docs/m4/README.md` and the README's roadmap table.
- `docs/design_philosophy.md` is the standard for any mechanic: simplify how League works without changing why its
  laning strategies work.
- Terminology: the midpoint cell x = 0 is "the lane" (never "river"). The full-capacity wave slot is "Shove" in the
  four-action presets and "Clear" in `m4_lasthit()` and earlier; it is the same code. The reward term paid for a
  champion's death is "kills" (never "bounty"). The config field `kill_bounty` and the `info` keys `bounty` /
  `bounty_units` keep the old word, as do reports written before 2026-10-20.
- `legacy/` is the original Q-learning prototype, kept unchanged for reference.
