# M6b report: the playable application

*2026-10-22. Status: **complete, awaiting the user's acceptance of the build.** Nothing here changes a mechanic, a
reward, a training objective, a trained network or a stored research result. No training ran. Nothing is committed:
the user asked for the work to stay in the working tree for review.*

M6b turned the approved M6a interface and the verified session core into one maintained application: the frontend
`web/` (React, TypeScript, Vite) served by `python -m lanegame.demo`, a FastAPI layer over `lanegame/demo/`. A visitor
plays a full game against either trained M5 agent, watches a generated match between them, browses the history of
either, reloads without losing the game, stops, and downloads a record that replays bit for bit in Python. The M6a
prototypes are gone, and the documentation points at the application.

- *Plan:* [m6_plan.md](m6_plan.md) §6 M6b. *Design:* [m6a_decision_record.md](m6a_decision_record.md) §7. *Core:*
  [m6a_technical_report.md](m6a_technical_report.md).
- *Launch:* `pip install -e ".[demo]"`, `cd web && npm install && npm run build`, then `python -m lanegame.demo`
  (http://127.0.0.1:8765/). Development: `npm run dev` in `web/` with the API running. Details in
  [web/README.md](../../web/README.md).
- *Evidence:* `results/m6/m6b_browser.json` and the screenshots in `results/m6/m6b/` (the browser journey),
  `results/m6/regression_suite_m6b.log` (the full test suite).
- *Tests:* `tests/test_demo_api.py` (16), `tests/test_demo_session.py` (29, unchanged), `web/src/game/*.test.ts`
  (10, `npm test`).

## Summary

- **Audit.** The M6a core was sound and is reused unchanged in its rules. Four integration risks were real: Watch
  simulations shared the human turns' lock, late responses could act on a replaced game, an unanswered submission
  left the screen guessing, and there was no reload recovery or expiry. All four are closed (§2, §3).
- **Built.** The typed API (`lanegame/demo/api.py`, `schemas.py`), the simulation worker process (`worker.py`), the
  launcher, session expiry and limits, the record download and the three verified research replays; the frontend, a
  faithful port of the approved glass design; the verification tooling (§4, §5).
- **Verified.** 71 of 71 browser checks pass across the whole journey the brief listed, including the
  overlap check in 80 of 80 layout cases; every API test, frontend test and the regression suite pass (§6).
- **Cleaned up.** The prototypes, their server, their generated data and the stage that wrote it are removed; the
  approved prototype's code is archived; the documents point at the application (§7).
- **Open.** Scripted opponents (the user's decision), static replays and the findings (M6c), hosting-grade
  safeguards (M6e), and a few inherited interface limits (§8).

## 1. The audit of the M6a work

The repository matched the M6a report in every claim checked: the session core, its 29 tests, the three seed-0
reproductions, the service with its HTTP statuses, and the approved prototype under `prototypes/m6a/glass/` with
live Play through `serve.py`. The findings that shaped M6b:

| finding | consequence |
|---|---|
| `DemoService.simulate` ran a 0.3 to 0.5 s bot game under the same lock as human turns. A second thread would share the M5 loaders' caches (`players._NETS`, `_TYPES`, `om2.common._NETS`), the `m45b` lane caches and torch's single-thread setting | Watch simulations run in a separate spawned process with its own registry (§3) |
| the prototype's `submit` had no generation guard: a turn result or a stop answered after a new game had started acted on the new game's state | every request carries the generation it was made in; a late answer is dropped (tested in `store.test.ts` and in the browser) |
| a failed or timed-out submission showed an error and nothing else; the visitor could not tell whether the turn had resolved | the client re-reads the server's view, appends any turn it missed from the history, and only then offers a retry of the same action, which the server resolves at most once |
| no reload recovery, no session expiry; the service dropped the oldest session beyond 64 silently | the browser remembers its session and resumes it; sessions expire after 30 idle minutes; the cap is 256 with the least recently active dropped first; a lost session is reported with a way out |
| the service's error responses were a convention of `serve.py`, and the frames had no schema | pydantic models for every shape (`schemas.py`, `extra="forbid"`); the OpenAPI schema generates the frontend's types; a test validates the core's real output against the models and the committed schema against the app |
| the M6a report's §6 said the prototypes' data files existed "because Play is staged against them", which the refinement round had made false | corrected with a dated note in that report; the files and the `export` stage that wrote them are removed |
| `lanegame/demo/service.py`, CLAUDE.md, the README, the plan and the roadmap named `prototypes/m6a/serve.py` as the launch path | all point at `python -m lanegame.demo` and `web/` |
| the engine | nothing found. Every value the application shows is a field of the core's frames; the browser computes no rule |

Duplicated code: the five prototype folders each carried their own copy of the lane, the controls and the history,
over one shared controller. The application has one of each. The prototypes' `serve.py` route table duplicated the
service's method map; the API replaces both.

## 2. What was built

### 2.1 The server (`lanegame/demo/`)

| module | content |
|---|---|
| `schemas.py` | the pydantic contract: model cards, status, frames (position, turn, tick, outcome, delta), the menu, the view, history, the turn result, the stop response, the recording, the game bundle, requests and the error body. Unknown fields are refused, so a field added on one side fails on the other |
| `api.py` | `create_app`: the routes of the M6a contract (§4.1 there) plus `GET /api/sessions/{id}/record`, `POST /api/simulations` through the worker, `GET /api/replays[/{name}]`; the error mapping (400, 403, 404, 409, 410, 413, 422, 429, 503); a 16 KB body limit; per-address token buckets for session creation, turns and simulations; the idle sweeper; the built frontend with the single-page fallback (or a page that says how to build it) |
| `worker.py` | the simulation process: `init_worker` loads its own registry; `simulate` runs `DemoService.simulate` there. `api.SimulationPool` owns one such process, a bounded queue of 4, a 120 s timeout, and reports `starting`, `ready` or `unavailable` in `/api/status` |
| `service.py` | gained `record`, `replays` and `replay` (the three seed-0 recordings re-simulated through `from_research` at startup, so a recording that does not reproduce is a startup failure), `expire`, the 256-session cap, and `load=False` for a server without an engine |
| `session.py` | gained `bundle()`: a whole game for playback (positions, turns, the beliefs the session may show, the recording) |
| `__main__.py` | `python -m lanegame.demo [--host] [--port] [--no-worker] [--static] [--idle-minutes] [--max-sessions]` and `--openapi PATH` |
| `pyproject.toml` | the `demo` extra: FastAPI, uvicorn, httpx (for the tests) |

Startup loads the registry (about 2.5 s), verifies the three research replays (about 1.5 s) and starts the worker,
which is ready about 3 s later; Watch's "New game" waits for it and says so.

### 2.2 The frontend (`web/`)

A port of `prototypes/m6a/glass/` into typed modules, with the imperative scene kept imperative:

| module | content |
|---|---|
| `src/api/client.ts`, `src/api/schema.d.ts` | the typed fetch client (a 15 s timeout, `ApiError` with the server's status and reason, status 0 for an unanswered request) and the generated types |
| `src/game/types.ts`, `frames.ts`, `words.ts` | the aliases, the sub-frames of a turn (start, one per recorded tick, end) and every word on screen, carried over from the prototype |
| `src/game/controller.ts` | the history controller (`Controller` for a whole bot game, `Live` for a growing human game): progress and the viewed position kept apart; navigation never submits; stepped frames; playback; `appendHistory` for a reload or a catch-up |
| `src/game/store.ts` | the state and every action: start-up with session resume, new game, Watch, simulate, submit with reconciliation, forced turns, stop, download, selection, panels, layout, theme, keys. The API is injectable, so the stale-response and reconciliation rules are unit-tested without a browser |
| `src/scene/lane.ts` | the diagonal lane, fitted around the glass surface, drawn into one `<svg>`; colours from the theme's CSS variables |
| `src/ui/*.tsx` | the top bar, the surface and its zones (status, turn line, history with navigation, controls, graph, the large graph), the controls (actions, the away panel, the step panel, playback, the stop summary, the problem block), Details, the dialogs, and an error boundary that keeps a rendering error from leaving a blank page |
| `src/styles.css` | the glass stylesheet, with the additions M6b needed (problem and loading blocks, the seed field, the download control) |

The three arrangements (dock, rail, console), the automatic layout by window shape, the themes, the stepped ticks and
their words, the gold graph and the Details drawer are the approved design's. Two additions: a "More options" seed
field in the New game dialog (plan §7, a chosen chance seed is recorded as chosen) and the download control after the
stop. Scripted opponents are not offered (plan §10 item 5, still the user's decision).

### 2.3 Tooling

- `python -m lanegame.experiments.m6b browser`: the scripted journey (§6).
- `lanegame/experiments/m6b/browser.py`: the server and headless-Edge helpers it uses (reusable for later checks).
- `.claude/skills/verify-lanegame`: `demo start|stop` (the server on a free port in 8700 to 8799), `browser open`
  waiting for the page's ready flag, `browser js` for the app's hooks, and `features/web-demo.md`.

## 3. The concurrency model

Plan §8.2 asked for a separate worker for bounded work. The serving process keeps the one simulation lock for human
turns, histories and the startup replays; nothing else simulates in it. Watch simulations run in one child process
started with the "spawn" context, which loads its own registry (the 7 files checked again) and keeps its own copies of
every module-level cache. The parent sends two model ids and a turn count and receives the finished game bundle;
nothing mutable crosses the boundary. At most four simulations wait for the worker, and further requests are refused
with 429 and a message. If the worker cannot start or fails, `/api/status` reports `unavailable` and Watch's dialog says
so; human play is unaffected. `--no-worker` runs simulations in the serving process under the lock (used by the tests
and useful on a machine short of memory; it costs about 250 MB less and stalls human turns by up to half a second).

A test starts the worker, generates a game through the API, replays its recording in the test process, and
re-simulates it there from the revealed seeds: the same turns and lead, with no session created in the serving
process.

## 4. Play

Every behaviour the brief listed, as the browser journey exercises it (§6):

- Actions chosen in the interface resolve in the simulator: the server's re-simulated history equals the client's
  positions and turns after five turns chosen by clicking.
- The menu marks the legal choices and explains the rest with the engine's reason (Ward's distance, Recall's cell,
  Pressure's missing crashed minions, Pressure with Hold or Trade only). A code the client did not offer is refused
  with the same reason (422).
- The opponent's action is fixed before the submission (the pending row says "Chosen"), and both are revealed
  together at the turn's start frame.
- A resolved turn steps through its start, each recorded tick and its end, with per-tick words from the tick's
  facts only; turn-level outcomes appear at the end. Trades are shown as no contact, miss or hit with the damage,
  from the authoritative outcome fields.
- Recall leads to a forced Absent turn with one Continue control, resolved one engine turn at a time; death and
  return follow the same path (seen in the generated Watch matches and in the core's tests).
- Stop declares no winner, shows the state, the lane-exit mark and the turn-114 lead where reached, reveals the
  seeds, and offers the record download. The 150-turn cap ends the game the same way.
- Hidden and shown belief modes behave as in the core: no belief data on the client in hidden mode until the stop,
  all 13 types with their probabilities in shown mode. Model identity, seeds, marks and the assessment wording are
  the server's.

## 5. Watch, history and recovery

- Watch opens a verified research recording (labelled "Recorded research game") or generates a match between the
  two agents in either seat ("Generated game"); Play, Pause, speed (0.5× to 4×), First, Previous, Next, Last, the
  history rows and the gold graph all navigate it; "Play from the start" restarts it.
- The same controller serves Play's history: browsing never changes progress, submission is disabled while browsing
  and "Return to current turn" (or Escape) restores the decision point. The lane, the status chips, the summary, the
  graph and Details all follow the viewed position; the server's position does not move.
- A reload resumes the session the browser remembers while the server holds it, rebuilding the history from the
  server's re-simulation; `?fresh` starts a new game instead and is then removed from the address, so a later reload
  resumes.
- Duplicate submissions resolve once (the server returns the stored result); a conflicting one is refused with 409,
  and the client catches up from the server before reporting it.
- An unanswered request (no answer, or a timeout) leaves the game where it was and offers "Reconnect and retry"; a
  lost answer whose turn did resolve is recovered from the history without a duplicate; a session the server no
  longer holds (expired, or a restart) is reported with a "New game" way out; a server without an engine says so in
  Play and Watch.
- A turn result or a stop that arrives after a new game started is dropped (the generation guard).
- Hidden information: the client's data never holds an unresolved opponent action, a future position, a hidden
  belief, or the seeds of an active session; the API tests check the view, the history and the record for these.

## 6. Verification

### 6.1 Tests

| suite | result |
|---|---|
| `tests/test_demo_api.py` (new, 16 tests) | the committed OpenAPI schema equals the app's; the models accept the core's real output and refuse extras; a live game with every status (200, 400, 403, 404, 409, 410, 422); history after a reload equals the turn results and hides the future; forced Absent; hidden and shown beliefs; eight simultaneous identical submissions resolve once; a simulation bundle replays; the research replays are verified bundles; expiry and the 404 recovery; the cap; rate and size limits; a server without an engine; the static frontend; the worker process |
| `tests/test_demo_session.py` (29, unchanged) | the M6a core rules |
| `web/src/game/controller.test.ts`, `store.test.ts` (10) | browsing never changes progress; the belief's visibility; catch-up appends only continuing turns; a late game or turn answer is dropped; an unanswered submission re-reads the server (appends, or offers a retry); an illegal action changes nothing; a lost session is lost; the selection rules |
| `npm run typecheck`, `npm run build` | clean |
| `python -m pytest` (the full suite) | 1596 passed, 4 skipped, 2 expected failures in 12 minutes (`results/m6/regression_suite_m6b.log`) |

### 6.2 The browser journey

`python -m lanegame.experiments.m6b browser` starts the real server and the built frontend, drives headless Edge
through real clicks and key presses, reads the app's own state hook, and compares with the server where the engine is
the authority. **71 of 71 checks pass** (264.0 s; `results/m6/m6b_browser.json`).

| part | checked | screenshot |
|---|---|---|
| 1. a live game | starts against the self-play agent; the opponent shown as having chosen; no secret seed or belief on the client | `01_play_start` |
| 2. actions | the first turn stepped by hand (start, tick 1, tick 2, end); five turns chosen by clicking resolve with the engine's codes; the client's positions and turns equal the server's history | `02a` to `02d` |
| 3. a trade | trades until the engine records one; the outcome in Details with its result (hit, miss seen) | `03_trade_details` |
| 4. review | turn 2 shown, progress unchanged, submission replaced by Return, the banner, the graph's mark, the server's position unmoved; Return restores the decision point | `04_review_turn_2` |
| 5. reload | the same session at the same turn, the rebuilt history equal to the client's | `05_after_reload` |
| 6. stop and download | no winner, seeds revealed, the record downloaded | `06_stopped` |
| 7. replay | the downloaded record replays bit for bit in Python (40 turns, seat B's codes recomputed) | `downloaded_record.json` |
| 8. Watch | a research recording; the worker ready; a generated 150-turn match whose record replays; Play, Pause, speed, Last, First, Next, Previous, a history row, the graph, restart | `07_watch_research`, `08a` to `08c` |
| 9. absence | Recall, the Absent-only menu with the reasons, Continue resolves the away turn | `09_away_turn` |
| 10. duplicates | two simultaneous identical submissions resolve once; a conflict is refused; the interface catches up | |
| 11. unanswered | a dropped request leaves the game where it was and retries once; a lost answer is recovered from the history; a lost session offers a way out | `10_unanswered_request`, `11_lost_session` |
| 13. beliefs | hidden: no data until the stop, then the history; shown: 13 types with probabilities | `12_belief_shown` |
| 14. stale | a turn answered after a new game started is dropped | |
| 15. keyboard | arrows, Home, End, Escape, Space on a choice, Enter to play, the turn buttons | |
| 16. reduced motion | a turn shows its end state at once | |
| 17. layouts | five window sizes (1440×900, 1024×700, 844×390, 915×412, 390×844) in both themes, a busy research position and a live game, with the history, the large graph and Details open: no panel covers a champion, wave, tower or the ward spot in 80 of 80 cases | `13_*` |

Each part also checks that the page reported no error.

## 7. Cleanup

| path | action | why |
|---|---|---|
| `prototypes/m6a/arena/`, `board/`, `story/` (12 files) | removed | the three first directions, staged walkthroughs, superseded in M6a |
| `prototypes/m6a/refined/`, `hud/` (9 files) | removed | the superseded live-Play rounds |
| `prototypes/m6a/glass/{app.js, index.html, style.css}`, `shared/core.js` | archived as `legacy/m6a_glass_prototype/` with a README | the approved design's reference code; not runnable, since its server and data are gone |
| `prototypes/m6a/glass/shots*.json`, `*/shots.json` | removed | screenshot specifications of the prototype tooling |
| `prototypes/m6a/data/*.json` (1.7 MB) | removed | generated by the `export` stage; the application serves the same recordings through its API |
| `prototypes/m6a/index.html`, `serve.py`, `tools/shoot.py`, `README.md` | removed | the launcher, the server and the screenshot tool, replaced by `python -m lanegame.demo` and `lanegame/experiments/m6b/` |
| `lanegame/experiments/m6a/export.py` and the `export` stage | removed | it only wrote the prototypes' data; `verify` and `measure` remain |
| `results/m6/prototypes/` (129 screenshots, 22 MB) | kept | the record of the six design rounds the decision record cites |
| `results/m6/m6a_export.json` | kept | the verification record of the export stage's run, cited by the M6a report |
| the Streamlit viewer | untouched | M6d's archive |

Every removed file was tracked and written in the M6a session; nothing untracked, no checkpoint, no recording and no
experiment result was touched. The documents were updated: CLAUDE.md, the README, `docs/README.md`, the plan's status
and layout, the roadmap (summary, the M6 plan, §BD), the M6a report (two dated notes) and the decision record (one).

## 8. Remaining limitations

Correctness: none known. Everything listed is scope or an inherited interface limit.

- **Scripted opponents** are not offered; the core supports them (the user's open decision).
- **Research replays** come from the API, re-simulated at startup; static files that work without the server, the
  other twelve recordings and the findings page are M6c.
- **Safeguards** are proportionate to a local demo: per-process rate limits by client address, a body limit, expiry
  and a cap. Hosting-grade measures, a load test and a container are M6e.
- **The worker process** costs about 250 MB of memory beyond the serving process and is ready about 3 s after the
  server; the first generated match waits for it.
- **Phone layouts** were checked in Edge's device emulation only; the lane stays small on phones, as in the
  prototype.
- **Manual stepping** costs four clicks per turn; "Play ticks automatically" removes them.
- **Reduced motion** skips the stepped frames, as the approved design did; "Show the ticks" still steps a turn.
- **The graph** is pointer-only; the keys and the turn buttons cover the same navigation.
- **The record** downloads only after the stop, because the seeds stay secret while the session is active.
- **`vl.py click --text`** needs `--no-settle` on this page (the harness's settle logic is Streamlit's).

## 9. What the user decides

1. Acceptance of the playable build (the plan's M6b decision).
2. Whether scripted styles are offered as opponents, and which ones.
3. Whether to commit this work as it stands.
