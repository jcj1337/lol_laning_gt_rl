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
pip install -e ".[test,rl,viewer]"     # viewer = the Streamlit debugger
python -m pytest                       # full suite, ~1,510 tests, ~8-11 min
python -m pytest tests/test_m4_env.py                  # one file
python -m pytest tests/test_m4_env.py::test_name       # one test (or -k pattern)
python -m pytest --runslow             # also run @pytest.mark.slow (exhaustive enumerations, K=32 re-solve)
```

There is no linter or formatter configured. Tests needing torch, pettingzoo or plotly skip themselves when those
packages are missing. Tests that need git-ignored checkpoints (`results/rl/checkpoints/`, `*.pt`) skip when the files
are absent.

Main CLI: `python -m lanegame <command>`. The module docstring of `lanegame/__main__.py` lists every command. The most
common are `solve` / `scenarios` / `exploit` (exact game, `--model siege|revised|v1`), `env-demo` (scripted agents
through the env) and `rl-shared --model <preset>` (shared-policy PPO).

Each milestone's experiments run as `python -m lanegame.experiments[.rl].<milestone> <stage> [<stage> ...] [--workers N]`,
for example `python -m lanegame.experiments.rl.m46c train`. The stage list is in that package's `__main__.py`
docstring. Expensive stages (training campaigns, batteries, attackers) refuse to start without a
`--confirm-<milestone>-<stage>` flag. That flag means the user approved the compute budget, so never pass it on your own
initiative. Long stages are resumable and write partial results under `results/<milestone>/.../partial/`.

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
    and reproducibility, is next.
- M6 is planned, not started ([plan](docs/m6/m6_plan.md), in review). It is a new web application in which a human
  plays against either M5 policy (`sp:0`, `om:0`). The Streamlit viewer stays where it is until M6d archives its UI.
  M6 adds no mechanics, rewards or training.

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
- The action space has 61 codes (Slow Push is 46–60, Ward is 30). The observation has 88 entries, or 90 with the M4.5
  Vision node; the opt-in field `obs_last_action` appends 13 entries for the opponent's previous action (103 with
  `m45_calibrated()`, used by OM2). A change to the observation or to its scales invalidates trained checkpoints.

**Determinism is a tested contract.** The env draws a fixed number of chance uniforms per turn, whatever the actions,
so that policies are compared on common random numbers. Same seed plus same actions gives a bit-identical rollout. The
lane-step cache is an exact memo. Training runs reproduce bit for bit, and tests check all of this. Don't introduce
action-dependent RNG consumption or unseeded randomness.

## Documentation conventions

- `docs/m4/laning_game_mathematical_reference.md` is the living description of the game. Any change to a rule, a
  parameter or the objective must update the matching equation, table or example there in the same piece of work and
  add a line to its §13 change log.
- Milestone reports (`docs/phase1/`, `docs/rl/`, `docs/m4/m4_X/`, `docs/natural_waves/`) are historical records. Don't
  edit them after the fact; corrections go in later reports and in the reference's §12.
- A milestone usually produces a pre-registration (written before training or evaluation), a report under `docs/`, and
  generated tables at `results/<milestone>/report.md`. It also updates the roadmap summary,
  `docs/README.md` / `docs/m4/README.md` and the README's roadmap table.
- `docs/design_philosophy.md` is the standard for any mechanic: simplify how League works without changing why its
  laning strategies work.
- Terminology: the midpoint cell x = 0 is "the lane" (never "river"). The full-capacity wave slot is "Shove" in the
  four-action presets and "Clear" in `m4_lasthit()` and earlier; it is the same code.
- `legacy/` is the original Q-learning prototype, kept unchanged for reference.
