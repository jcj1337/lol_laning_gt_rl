# M6a technical report: the session core, its contracts and its costs

*2026-10-21. Status: **complete.** The backend is verified, and the user approved the interface design
([decision record](m6a_decision_record.md) §7), so M6a as a whole is complete. Nothing here changes a mechanic, a
reward, a training objective, a trained network or a stored research result. No training ran.*

M6a has two outcomes, a trustworthy game-session backend and a reviewed frontend direction. This report covers the
first. The session core is the Python package `lanegame/demo/`. The plan's correctness rules 1 to 5 and 7 pass on the
real seed-0 finals, and the costs below inform the API and playback contracts that M6b implements. The prototypes are
described in the decision record and `prototypes/m6a/README.md`.

- *Plan:* [m6_plan.md](m6_plan.md) (revised 2026-10-21). *Evidence for M5:* [m5e_closeout_report.md](../m5/m5e_closeout_report.md).
- *Records:* `results/m6/m6a_verify.json`, `results/m6/m6a_measurements.json`, `results/m6/m6a_export.json`,
  `results/m6/regression_suite_m6a.log`.
- *Commands:* `python -m lanegame.experiments.m6a verify measure export` (about 35 s; needs the 7 checkpoint files).
  Tests: `python -m pytest tests/test_demo_session.py` (29 tests, about 11 s).

## Summary

- **Built.** A registry that checks all 7 checkpoint files against a tracked manifest before any is unpickled. A
  `GameSession` with human, model and scripted seats and the turn loop M5 used. A versioned recording with per-turn
  state digests and deterministic replay. Frames for the lane view and history navigation, plus an action menu whose
  reasons come from the engine's own legality predicates.
- **Verified.** The three seed-0 M5 recordings re-simulate exactly through the session core, with every action,
  `payoff_A` to the last bit, and the belief's log posteriors at M5's check turns with a difference of 0.0. The 12
  attacker recordings replay exactly from their actions. Every check the brief listed has a test, and all pass (27
  at first; 29 with the local service added in the refinement round, §7).
- **Measured.** One turn with its frames and JSON takes 1.4 ms against `sp:0` and 2.3 ms against `om:0` (median).
  A full 150-turn history takes about 0.09 s to re-simulate and is 22 KB gzipped. A bounded 150-turn bot game takes
  0.27 to 0.48 s plus 0.1 s for its frames, and is 23 to 35 KB gzipped. A session retains about 0.1 MB after 150
  turns.
- **Blockers.** None for M6b. §6 lists the open items.

## 1. What the inspection changed

The plan was written on 2026-10-19. The repository on 2026-10-21 differed from it in these respects.

| finding | consequence |
|---|---|
| Commit `b94200f7` untracked all 81 docs files, and `.gitignore` still ignored `docs/` | the line is removed. The 85 files on disk (the 81 plus the M5c and M5e reports, the M6 plan and the archived M4 reference) are tracked again. The 81 differ from their last tracked versions by line endings only |
| `rl.load_checkpoint` calls `torch.load(weights_only=False)` before it checks the weights digest | the registry reads each file once, hashes those bytes against the manifest, and unpickles the same bytes from memory. Nothing can change between the check and the load |
| `OMPolicy` loads its 5 library networks lazily on first use | the registry puts the checked networks into the M5 loaders' caches (`players.install`, `om2.common._NETS`) and builds the belief types from them, so the lazy path never reads a file |
| two of the three seed-0 research recordings have the scripted `g_shove` in one seat | the core supports the six scripted library archetypes as seats, so those recordings re-simulate. Offering scripted opponents to visitors stays a design decision |
| the seed-0 head to head has no death | the prototypes use `om_s0_vs_g_shove` (6 deaths of the scripted seat) for deaths and forced absent turns, and say so |
| history navigation needs every resolved turn's frames again, while the plan sent each turn's frames once and dropped them | the client keeps the frames it receives. The server re-simulates a history from the recorded codes when a client reloads or retries an older turn (§4.1) |
| a drawn chance seed lets anyone who knows it foresee the trade, gank and tower rolls | a drawn chance seed stays secret until the session stops, like the policy seed. The plan only said this of the policy seed |
| Python 3.12 and later sums floats with compensation in `sum()` | the session adds rewards turn by turn, as M5's recordings did, so `payoff_A` reproduces exactly |
| the Streamlit viewer draws a returning champion at its tower during the last turn of its absence | the demo's frames follow the engine's `info["away"]` and show the champion away until the turn ends. The test records this as the one deliberate difference from `frame_view` |

## 2. The session core

| module | content |
|---|---|
| `lanegame/demo/manifest.json` | the 7 files and their SHA-256, copied from `results/m5/inventory.json`. A test checks every entry against the inventory |
| `registry.py` | `ModelCard` for `sp:0` and `om:0` (arm, seed, configuration, input width, file SHA-256, weights digest, dependencies, training budget, evidence links). `Registry.load()` checks every file before the first unpickle and refuses all of them on one mismatch. `policy()` builds the seat with M5's own `players.crn_policy` |
| `session.py` | `GameSession`, `Seat`, `from_research`, `simulate`, `replay` |
| `record.py` | the recording format, `state_digest`, `replay_frames` (codes alone) and `replay_actions` (v1 viewer recordings) |
| `frames.py` | position and turn frames (§4.3), built from the viewer's tested functions |
| `actions.py` | the action menu and `explain`, the reason for any refused code |
| `service.py` | (added in the refinement round, §7) a JSON service over the sessions: create, submit, view, history, stop, simulate; one lock serializes all simulation; refusals carry an HTTP status |
| `lanegame/experiments/m6a/` | the stages `verify`, `measure` and `export` |

The core imports no web framework. M6b's API is a thin layer over it, and `service.py` is already that layer for the
local demo server.

**The turn loop** is M5's (`experiments/rl/m5/games.play`).
1. When a turn begins, every bot seat decides from the pre-turn state, and the session stores its code.
2. A human submission names a turn number and a code. The session checks the code against the legal set of the
   pre-turn state. A refused code changes nothing.
3. The environment resolves the turn with both codes. Each bot seat's `observe_turn` then runs with the pre-turn
   state, so the belief updates only after both codes are fixed.
4. The next turn begins, unless the turn cap (150) is reached.

**Turn numbers.** Position p is the state after p turns, and position 0 is the initial position. Turn t moves position
t − 1 to position t. The lane-exit marker is position 94 (`games.LANE_END`). The observed lead is the earned-gold lead
at position 114 (`games.READ_TURN`), which M5 averaged over many games to realize E_20 at the lane exit.

**Seeds.**
- *Research reproduction.* The policy seed equals the chance seed, M5's convention.
- *Human game.* The server draws a separate policy seed. It also draws the chance seed unless the visitor chooses
  one, and the record marks a chosen seed. A drawn chance seed and the policy seed stay secret until the session
  stops.
- *New simulation.* The server draws both seeds unless the caller gives them. Both appear in the stopped record.

Bot seats sample as M5 evaluated them. A model seat uses the inverse CDF with the uniforms of
`default_rng([policy_seed, 7, seat])`, and a scripted seat uses the generator `default_rng([policy_seed, 11, seat])`.
No seat plays greedily, and a test confirms that sampled actions differ from the most probable ones.

**States and refusals.** A session is `awaiting_action` or `stopped` (reason `visitor`, `turn_cap` or `expired`).
Every refusal is a `SessionError` that leaves the session unchanged:

| refusal | when |
|---|---|
| `IllegalAction` | the code is not legal at the pre-turn state, or is not an integer. The message is the menu's reason |
| `TurnConflict` | a resolved turn is resubmitted with a different code |
| `TurnOutOfOrder` | the turn number is later than the current turn, or is not an integer |
| `SessionStopped` | the session has stopped |
| `HiddenInformation` | a history reaches past the resolved positions, or a replay is requested while the seeds are withheld |

A resubmission of a resolved turn with the same code returns the stored result. A per-session lock serializes
submissions. Forced turns are resolved one at a time. While the visitor's champion is away, Absent (45) is the only
legal code, `forced()` reports it, and the client can continue through the absence with one control that submits 45
once per turn.

## 3. Verification

| check (brief §5) | test or stage | result |
|---|---|---|
| the M5 reference match reproduces exactly under the research seed convention | `test_the_seed0_research_recordings_reproduce_exactly_through_the_session_core` (3 recordings), stage `verify` | all 150 actions, `payoff_A` bit for bit, belief log posteriors at the 8 check turns identical (0.0). The two `g_shove` games reproduce through the scripted seat |
| sessions against both trained opponents replay correctly | `test_a_full_human_session_replays_bit_for_bit_and_a_changed_record_does_not` (`sp:0`, `om:0`), `test_new_simulations_are_labelled_and_replay` | 150-turn human sessions replay with every policy code recomputed and every digest equal. A changed code or digest is refused |
| different legal human actions from the same pre-turn state do not change the opponent's fixed action | `test_the_opponent_code_is_fixed_before_the_human_acts_and_equals_a_fresh_replay` (both opponents) | for every legal human code at a turn where the opponent had a real choice, a fresh replay computed the same stored code, and the turn resolved with it |
| the belief updates after both actions, from the pre-turn state | `test_the_belief_updates_after_both_codes_with_the_pre_turn_state` | deciding leaves the belief unchanged, and the update equals M5's `OMPolicy.observe_turn` on the pre-turn state |
| illegal submissions leave state and random streams unchanged | `test_illegal_or_malformed_submissions_change_nothing` (both opponents) | every illegal code, out-of-range codes, booleans, strings, floats, `None` and bad turn numbers: state, digest, turn index, chance generator, bot uniforms, belief and stored code unchanged |
| interleaved sessions match sequential execution | `test_interleaved_sessions_equal_sequential_ones` | four sessions (human against `sp:0` and `om:0`, human in seat B, `om:0` against `g_bully`) give identical records |
| duplicate submissions resolve once | `test_a_duplicate_submission_resolves_once_and_a_conflicting_one_is_refused`, `test_concurrent_duplicate_submissions_resolve_the_turn_once` | a retry returns the stored result, a conflict is refused, and 8 simultaneous threads resolve the turn once |
| tampered checkpoint files are rejected before loading | `test_an_altered_or_missing_checkpoint_is_refused_before_anything_is_unpickled` (no checkpoint needed), `test_a_tampered_real_checkpoint_copy_is_refused` | one flipped bit in a library network, or a missing file, stops the registry before `torch.load` is called once |
| historical views cannot change the session or reveal hidden information | `test_history_cannot_change_the_session_or_reach_past_the_resolved_turns`, `test_views_never_carry_the_stored_code_or_a_secret`, `test_belief_modes` | history requests leave the session unchanged and refuse unresolved positions. Replacing the stored opponent code changes no byte of the view. Seeds and hidden beliefs are absent while the session is active, and the belief history appears after the stop in either mode |
| frames follow the engine, and the menu follows the legality predicates | `test_frames_equal_the_viewers_and_hold_only_per_tick_facts_in_ticks`, `test_the_menu_marks_exactly_the_engines_legal_codes_and_explains_the_rest` | packets, crashed masses, contacts and champion cells equal `frame_view` tick by tick (§1, last row). Turn-level events never appear in ticks. At every position of a recording, both seats' menus mark exactly the engine's legal codes, and every other code has a reason |
| stopping declares no winner; the marks | `test_stopping_declares_no_winner_and_reveals_the_seeds`, `test_the_lane_exit_marks` | the summary describes the state. Position 94 carries the lane-exit panel and position 114 the lead, equal to the reward sum and to `lane_exit.lead` |
| the 12 attacker recordings | stage `verify` | each replays exactly from its actions. Their attackers are not in the registry, so their codes are not recomputed (§6) |

All checkpoint-dependent tests ran on the real files. None skipped. The full regression suite result is in
`results/m6/regression_suite_m6a.log` and in §3.1.

### 3.1 Regression suite

`python -m pytest` on the final M6a tree: **1,580 passed, 4 skipped, 2 expected failures, 3 warnings, in 10 minutes
31 s** (`results/m6/regression_suite_m6a.log`). The 4 skips are the `--runslow` tests and the 2 expected failures come
from earlier milestones, the same as in the M5e run (1,541 passed then). The 3 warnings come from PettingZoo's API
checker. The 39 additional passing tests are this milestone's 29 and the reference generator's, added after M5e. The
run covers every later edit to `lanegame/demo`, including the service and the ward-spot wording of the design rounds.
An earlier run, before the service existed, passed 1,578.

## 4. Contracts (proposed for M6b)

### 4.1 API

The API is a projection of the session core. Every request resolves at most one turn, except a bounded simulation.

| request | response | notes |
|---|---|---|
| `GET /api/models` | the public model cards | scripted styles only if the user decides to offer them |
| `POST /api/sessions` `{opponent, human_seat, belief_mode, chance_seed?}` | the view | the server draws the seeds it is not given and never accepts a policy seed |
| `GET /api/sessions/{id}` | the view | the current decision point |
| `POST /api/sessions/{id}/turns` `{turn, code}` | `{turn, position, belief?, view}` | 422 with the menu's reason for an illegal code; 409 for a conflict or a turn out of order; 410 after the stop. A duplicate returns the stored `turn` and `position` |
| `GET /api/sessions/{id}/history?from=&to=` | `{positions, turns, beliefs?}` | resolved positions only, re-simulated from the codes. The client calls it after a reload |
| `POST /api/sessions/{id}/stop` | the summary | no winner |
| `GET /api/sessions/{id}/record` | the recording | seeds and hidden beliefs withheld while active |
| `POST /api/simulations` `{a, b, chance_seed?}` | `{recording, positions, turns, beliefs}` | Watch. One bounded game (at most 150 turns), generated in full, then played back by the client |
| static `replays/<name>.json` | the same shape as a simulation | research recordings, verified when the bundle is built. They work with the API stopped |

**The view** has `state`, `position`, `next_turn`, `turn_cap`, the seats (model id, label, file SHA-256, weights digest),
`belief_mode`, the seeds (withheld while active unless chosen), `current` (the position frame), `marks` (lane exit,
lead at 114), and for a human seat `opponent_ready` (always true, never what) and `menu`. It carries `belief` only in
shown mode or after the stop.

**History navigation is a client concern.** The client keeps every position and turn it has received and navigates
them without a request. Navigation never changes `progress` (the resolved turns) and never submits. The server is
consulted only to rebuild the cache after a reload. A historical view disables submission and shows a "Return to
current turn" control. No endpoint returns an unresolved position, the stored opponent code, a probability, a logit or
a seed of an active session.

### 4.2 Recording (`lanegame.demo_recording`, version 1)

| field | content |
|---|---|
| `provenance` | `kind` (`research_record`, `new_simulation`, `human_game`), `note` (an example, not evidence, for the last two), `source` for research records (file, SHA-256, M5 cell and selection rule) |
| `config` | the name (`m45_calibrated`) and the SHA-256 of `GameConfig.to_dict()` |
| `seeds` | `chance`, `chance_chosen`, `policy`, `policy_rule` (`chance`: M5's convention; `drawn`; `given`) |
| `seats` | per seat: `human`, or a model (id, label, arm, seed, file SHA-256, weights digest, sampling), or a scripted archetype and its rule |
| `belief_mode`, `turn_cap`, `status` | the mode chosen at the start; the cap; state, stop reason and turns |
| `turns` | per turn `[code A, code B, state digest]`, the digest being the first 16 hex digits of SHA-256 over the canonical JSON of (lane state, champion state, k) |
| `lead_A`, `marks` | seat A's total reward (added turn by turn); the lane-exit panel and the lead at 114 |
| `beliefs` | the opponent-modelling seat's posteriors at positions 0 to n, where the session may show them |

`replay(recording)` re-simulates with the policies, recomputes every bot code and checks every digest. The 15 M5
recordings stay in their v1 viewer format. `from_research` converts the three seed-0 games, and M6c's bundle build
converts the rest.

### 4.3 Frames (`lanegame.demo_frames`, version 1)

| shape | content | source |
|---|---|---|
| position p | `lane` (packets `[x, hp, front]` per seat, crashed mass `[x, hp]`, contact), `champions` (level, HP and max HP in centi-HP, XP and its level bounds, gold earned, unspent and invested, attack power, status present / recall / dead, returns in, cell or null at W, vision timer, capacity and wave budgets), `towers` (damage, plates lost), `lead_A`, `wave_status` | the viewer's `summary`, `lane_objects`, `contact_of`, `wave_status`; `lane_exit.lead` |
| turn t | `codes`, `names`, `parts` (wave, behaviour, objective or Recall / Ward / Absent), `modes`, `ticks`, `outcomes`, `delta` (gold, farm, XP, HP change, HP taken), `reward_A` | `info` of the authoritative step |
| tick | packets, crashed masses, contact, each champion's cell after the tick, the wave HP each champion removed, the minion fire each took, HP crashed | `info["lane_trace"]`, `info["champion_path"]` |
| outcome | `kind`, `seat`, `major`, `text` with `{A}` / `{B}` placeholders for the client's names, `data` | `info`, in the resolver's order (lane, Recall interruption, trades, tower shots, ganks, minion fire, deaths and kills, tower damage and plates, level-ups, wards, Recalls, returns, investment) |

Coordinates are global (seat A's own). A frame never holds the chance uniforms, a probability, a logit or a policy
seed. Turn-level events are outcomes only. The trade outcomes are "no contact", "miss" (connected, no damage) and
"hit" with its damage.

### 4.4 The action menu

`actions.menu` groups the codes as the reference's §3.3 does: four wave choices, four behaviours, Pressure (with Hold or
Trade only), Recall and Ward as single actions, and Absent as the forced code while away. Every unavailable choice has
a reason derived from the predicate that failed (`siege_eligible` and the reach for Pressure, `recall_legal`,
`ward_legal`, the absence counter). The menu raises if the codes it marks legal ever differ from
`champion_legal_actions`.

## 5. Measurements

Measured on this machine (12 logical CPUs, one torch thread, 10% CPU load at the start of the stage), from
`results/m6/m6a_measurements.json`.

| quantity | `sp:0` | `om:0` |
|---|---|---|
| one turn: submission, step, belief update, the next decision, frames, view with menu, `json.dumps` (median / 95th percentile, 600 turns) | 1.44 / 1.69 ms | 2.29 / 3.04 ms |
| of which JSON serialization | 0.08 ms | 0.09 ms |
| turn response | 6.1 KB, 2.1 KB gzipped | 6.0 KB, 2.0 KB gzipped |
| full 150-turn history re-simulated with frames (median of 5) | 88 ms | 90 ms |
| its size | 328 KB, 22 KB gzipped | 337 KB, 23 KB gzipped |
| recording of 150 turns | 6.1 KB, 3.1 KB gzipped | 14.8 KB with beliefs (6.1 KB without), 3.6 KB gzipped |
| memory retained by one session after 150 turns | 86 KB | 106 KB |

| bounded bot game (150 turns) | simulation | frames and JSON | bundle gzipped |
|---|---|---|---|
| `om:0` against `sp:0` | 0.35 s | 0.09 s | 32 KB |
| `sp:0` against `sp:0` | 0.27 s | 0.09 s | 25 KB |
| `om:0` against `om:0` | 0.48 s | 0.10 s | 35 KB |
| `om:0` against `g_shove` | 0.32 s | 0.09 s | 23 KB |

A cold start in a fresh interpreter took 1.8 to 1.9 s to import torch and 0.6 s to import lanegame and check and load
the 7 networks. The plan's 5 to 6 s was measured under 83 to 93% CPU load.

**What the numbers imply for the design.**
- Request and response per turn is enough. A turn costs about 2 ms on the server, so one simulation thread handles
  several hundred turn requests per second, and a human takes seconds per turn.
- History navigation should run on the client's cache. A rebuild from the codes costs about 0.1 s and 22 KB
  gzipped, which suits a page reload but would make every history click slow.
- Watch can generate first and play back afterwards. A bounded game costs at most about 0.5 s plus 0.1 s of frames and
  is about 35 KB gzipped, so one response carries it. On the single simulation thread it would delay human turns by up
  to half a second, so M6b runs simulations in a separate worker with its own concurrency limit (plan §8.2).
- Memory per session is small. A thousand finished sessions hold about 0.1 GB, against about 0.25 GB for the process
  itself, so the session cap guards CPU and abuse rather than memory.
- Static replays are cheap. A verified research game with frames is 22 to 35 KB gzipped.

## 6. Open items

None blocks M6b.
- **The production API layer** (FastAPI and uvicorn as a `demo` extra) and rules 6, 8 (for app records) and 10 to 12 at the API
  level belong to M6b. The core already implements them.
- **The attacker recordings** replay from their actions only. Re-simulating their codes would need the 12 attacker
  networks as registry entries, 12 more manifest files. M6c decides whether its bundle build needs that or the action
  replay suffices.
- **The prototypes' data files** contain whole recorded games, because Play is staged against them. The production
  client never receives an unresolved turn.
- **The demo frames and the Streamlit viewer** differ in one deliberate place (§1, last row).
- **Scripted opponents** are supported by the core and their inclusion is undecided ([decision record](m6a_decision_record.md)).

## 7. The refinement round: live Play through a local service (2026-10-21)

After trying the prototypes, the user found that staged Play looked like a game while it replayed recorded actions,
and asked for live Play if the session core could be connected with modest work. It could.

- **`lanegame/demo/service.py`** wraps the session core in JSON requests: `status`, `create` (a human in seat A
  against `sp:0` or `om:0`, the belief hidden or shown), `submit`, `view`, `history`, `stop` and `simulate` (Watch). It
  maps every refusal to an HTTP status: 422 for an illegal action (with the menu's reason), 409 for a conflict or a
  turn out of order, 410 after the stop, 403 for an unresolved position, 404 for an unknown session, 503 when the
  engine is not loaded. One lock serializes all simulation. It is the subset of §4.1 that the demo needs, with the
  shapes §4.1 proposes.
- **`prototypes/m6a/serve.py`** serves it with the Python standard library on 127.0.0.1, beside the pages. Without the
  checkpoints the pages still load, and Play says that the engine is unavailable.
- **A fix found while connecting it.** Drawn seeds were 62-bit integers, which a browser's JSON numbers cannot hold
  exactly, so a record downloaded through a browser would have failed to replay. Drawn seeds are now below 2^53. A
  test checks the range.
- **Tests.** `test_the_service_plays_a_live_game_and_maps_refusals_to_statuses` (a live game, every refusal and its
  status, hidden beliefs absent until the stop, the record replaying after a JSON round trip, a bounded simulation) and
  `test_drawn_seeds_stay_exact_as_browser_json_numbers`.

The refined interface (`prototypes/m6a/refined/`) and every later one, including the approved `prototypes/m6a/glass/`,
use this service, so their Play is live. The three earlier directions keep their staged walkthroughs and are marked
as such.
