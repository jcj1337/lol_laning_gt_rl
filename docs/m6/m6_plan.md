# M6 plan: presentation, interactive demonstration and reproducibility

*Written 2026-10-19, revised 2026-10-21. Status: **M6a in progress.** The headless session core is built and
verified ([M6a technical report](m6a_technical_report.md)). After trying three clickable prototypes, the user chose
prototype C with prototype A's action controls and a better gold-lead graph ([M6a decision record](m6a_decision_record.md)
§6). One refined interface, with live Play through a local service over the session core, awaits the user's visual
review, so M6a is still pending. The production frontend (M6b) has not started, and `web/` does not exist yet. The Streamlit viewer has not been archived and works as
before. No hosting has been chosen, paid for or deployed.*

*The 2026-10-21 revision records the user's M6a brief. Live bot-versus-bot simulation (Watch) is now part of the
initial release. A Chess.com-style turn history governs navigation in Play and Watch. The frontend has explicit
clarity requirements. The three decisions of the first version are settled (§10). Sections that the brief or the M6a
work changed say so.*

M5 is complete ([final report](../m5/m5e_closeout_report.md)). M6 makes its findings understandable, playable and
reproducible. Its centre is a new web application with two primary modes. In **Play**, a visitor chooses discrete
actions for one seat against a trained M5 policy, either pure self-play or the opponent-modelling package. In
**Watch**, two bots play a bounded simulated match that the visitor explores through a timeline. The same application
replays verified research games and explains what M5 found and what it did not. M6 adds no mechanics, rewards,
training objectives or training runs. Further research stays optional and is never a prerequisite for finishing M6.

The work runs in five milestones: M6a (product definition and technical readiness), M6b (the playable application),
M6c (research demonstration and explanation), M6d (documentation and reproducibility) and M6e (release readiness).
The initial release is complete at the end of M6e, and public deployment waits for the user's approval there.

## 1. Status of each part

Status labels follow the roadmap's legend.

| item | status |
|---|---|
| M6 is presentation, demonstration and reproducibility | approved direction (the user, 2026-10-19) |
| a new web application that can later be deployed publicly, with human versus trained policy at its centre and both M5 arms offered as opponents | approved direction (the user, 2026-10-19) |
| two primary modes in the initial release: Play (a human against a trained agent) and Watch (two bots in a bounded simulated match, generated first and then played back) | approved direction (the user, 2026-10-21); live bot-versus-bot is no longer a post-release feature |
| the two seed-0 M5 agents (`sp:0`, `om:0`) offered for either bot seat | approved direction (the user, 2026-10-21) |
| scripted strategies as additional opponents | a design decision, assessed in the [decision record](m6a_decision_record.md); the session core supports them |
| a Chess.com-style turn history shared by Play and Watch, and the frontend clarity requirements (§5.1, §5.2) | approved requirements (the user, 2026-10-21) |
| the Streamlit viewer archived as legacy during implementation, not reworked | approved direction; planned for M6d, not done. The old UI is not a visual template |
| guided replay of verified research games as a supporting mode | approved direction to consider it; its scope in §7 is proposed |
| React with TypeScript and Vite in `web/` in this repository, FastAPI over the Python simulator, one container | approved by the user on 2026-10-21, unless a concrete blocker appears; M6a found none |
| `docs/` tracked in git | approved (the user, 2026-10-21) and done: the blanket ignore rule is removed |
| the headless session core and the rules it proves (M6a) | completed and verified ([technical report](m6a_technical_report.md)) |
| the visual direction | unresolved; the user chooses among the M6a prototypes |
| the milestones, the bundles and the safeguards | proposed (this plan) |
| a hosting provider, spending and public deployment | not decided; a checkpoint in M6e |

## 2. What exists and what M6 reuses

The repository was inspected on 2026-10-19, in the working tree before M5 was committed. The user then committed
M5's code and results in `a07e3fdf`. Each row names the reusable part and the gap M6 has to close. M6a closed several
gaps on 2026-10-21. The rows say where, and the [technical report](m6a_technical_report.md) §1 lists what the second
inspection changed.

| component | where | reuse in M6 | gap |
|---|---|---|---|
| the simulator | `LaneEnv` (`lanegame/env/core.py`), `resolve_turn_m4` (`champions_m4.py`) | unchanged; Python stays authoritative for every rule | none. Each env owns its generator (`np.random.default_rng(seed)`) and draws exactly 10 chance uniforms per turn whatever the actions |
| legality | `champion_legal_actions` (`champions.py`), `Observation.legal` and `.mask` | the backend's validation; an illegal code raises `ValueError` before any state change or random draw | nothing said *why* an action is unavailable. **Closed in M6a:** `lanegame/demo/actions.py` derives every reason from the same predicates (`objective_reachable`, `ward_legal`, `recall_legal`, the absence counter) and checks its legal set against the engine's |
| action names | `GameConfig.label`, `viewer/session.action_name`, `BEHAVIOR_NAMES`, `OBJECTIVE_NAMES`, `wave_display_names` (`config.py`) | the labels for the 61 codes. Most codes combine a wave choice, a behaviour and an objective. Ward (30), Recall (12) and Absent (45) are single codes, and Absent is the only legal code while a champion is away | plain-language descriptions for newcomers |
| the two opponents | `crn_policy` and `OMPolicy` (`experiments/rl/m5/players.py`); `policy_path("sp:0")`, `policy_path("om:0")` (`m5/common.py`) | the seed-0 finals exactly as M5 evaluated them: `sp:0` (90 inputs) and `om:0` (116 inputs, the 103-entry observation plus 13 belief features) | `decide(cfg, lane, m4, s, i)` samples from uniforms drawn in advance for `horizon` turns, so the server sets `horizon` to the session's turn cap. Both seats take the 90-entry `m45_calibrated()` configuration, and `OMPolicy` builds its own 116-entry input. The server imports these loaders and does not copy them |
| the belief | `Belief` (`lanegame/opponent/belief.py`); the library in `experiments/rl/om2/common.py` and `om1/common.py` | `OMPolicy.decide` is pure, and `observe_turn` updates the belief only after both codes are fixed, from the pre-turn state | none in the mechanism. The library is a serve-time dependency (next row) |
| the belief's library | 6 scripted archetypes (`g_farm`, `g_safe`, `g_bully`, `g_shove`, `fz_guard_sh`, `g_shove_ward`, code in `experiments/m45b/strategies.py`); 5 networks; the catch-alls "mixed" and "unknown" | required to serve `om:0` | the 5 networks (`m45c/full/s3`, `m45c/attack/ext_s3`, `m46a/e4/freeze_s2`, `m46a/e4/gap_s2`, `m47/train/main/short/s0`, each `final.pt` under `results/rl/checkpoints/`) are git-ignored. `results/m5/inventory.json` lists them with their file SHA-256, but the key-checkpoint table of `reproducibility.md` does not. **M6a:** `lanegame/demo/manifest.json` lists all 7 files with the inventory's SHA-256 (a test checks them against it) |
| checkpoints | `load_checkpoint` (`lanegame/rl/agent.py`) | loading with the weights digest verified | `torch.load` runs with `weights_only=False`, which unpickles the file before the digest check. **Closed in M6a:** `lanegame/demo/registry.py` reads each file once, checks its SHA-256 against the manifest, refuses every file on one mismatch, and unpickles the checked bytes from memory. It fills `OMPolicy`'s library caches with the checked networks, so the lazy load never reads a file |
| recordings | `lanegame.viewer_recording` JSON; producers in `experiments/rl/m43b/trajectories.py` and `m5/traces.py`; 15 M5 recordings in `results/m5/traces/` (452 KB) | the research replays. All 15 replay exactly (M5e report §13, re-checked in M6a) | no provenance kind, no file SHA-256 per seat, no policy seed and no per-turn state digest. **M6a:** the app format `lanegame.demo_recording` has all of them (`lanegame/demo/record.py`). The 12 attacker recordings still name the attacker by a Windows path |
| replay and frames | `lanegame/viewer/session.py` (630 lines, no UI dependency): `play`, `load_recording`, `frame_view`, `summary`, `timeline`, `wave_panel`, `turn_events`, `wave_status` | the source of every frame and every text description the frontend shows | 11 experiment modules and 7 test files import `lanegame.viewer`, so `session.py` and `policies.py` stay importable where they are. **M6a:** the session core has its own loop with `observe_turn`, and `lanegame/demo/frames.py` builds the frames from the viewer's functions (tested equal to `frame_view`) |
| per-tick events | `LaneEnv(record_ticks=True)` puts the authoritative step's trace in `info["lane_trace"]` (`mechanics.step_turn`) | packets, fights, crashes, thinning, champion damage on the wave, minion fire and champion paths, tick by tick | trades, tower shots on champions, ganks, wards, Recalls, deaths, plates and level-ups exist only per turn. Placing them at a tick would invent timing |
| the Streamlit UI | `lanegame/viewer/app.py` (464 lines), `render.py` (223 lines, plotly) | not reused | archived in M6d |
| reproducibility | `results/m5/reproducibility.md`, `inventory.json` (429 networks hashed), the `reproduce` stage | the evidence for the findings page and the reproduction guide | generated at `312d118d` with an uncommitted tree, before the M5 commit `a07e3fdf`, so it needs a re-run on the committed tree. No bundle or archive command exists |
| packaging | `pyproject.toml` (extras `test`, `rl`, `viewer`) | the Python install | no Dockerfile, no CI, no LICENSE. `requirements.txt` has no torch and pins nothing. No JavaScript toolchain in the repo (Node 24 and Docker are installed on this machine) |
| repository hygiene | `.gitignore` ignored `docs/`, and commit `b94200f7` untracked all 81 docs files | | **Closed in M6a:** the blanket rule is removed, and the 85 files on disk plus the M6a documents are tracked again |
| tests and live checks | 1,541 passed in 8.5 minutes in the M5e regression run; `.claude/skills/verify-lanegame` drives the Streamlit viewer in headless Edge over DevTools | the regression suite; the same browser approach for the new app | the skill covers only the old viewer |

### 2.1 Measured serving cost

A probe on 2026-10-19 loaded the two seed-0 finals through M5's own loaders (`C.setup`, one torch thread) and played
seeded games. Other programs kept the machine at 83 to 93% total CPU, so the numbers are pessimistic. The file
SHA-256 of both finals matched `reproducibility.md`. The belief updated as intended, and its entropy fell from 2.56
to between 0.23 and 0.38 nats over 150 turns.

| item | measured |
|---|---|
| one `sp:0` turn (decision 0.6 ms median, plus the engine step) | about 1.2 ms |
| one `om:0` turn (decision 0.4 ms, belief update 1.8 ms, plus the step) | about 2.7 ms |
| a 150-turn game, `om:0` against `sp:0` | 0.6 s |
| a continuation-based E_20 from turn 94 with K = 24 rollouts (M5's `lane_exit_om` setting) | 0.7 s (`sp`), 2.0 s (`om`) |
| per-session state (env plus `OMPolicy`) | about 22 KiB |
| one process after loading both opponents and the library | about 240 to 260 MB resident, almost all of it torch |
| cold start (import of torch and lanegame) | 5 to 6 s |
| the 7 networks the demo needs | 154 to 170 KB each, about 1.1 MB in all |

The belief update dominates the `om` turn. It runs 5 library forward passes and 6 scripted decisions. Six torch
threads made every step slower, so one thread is the right setting. The measurement excludes HTTP and the
serialization of frames, which M6a measures. The inference from these numbers is that one process can serve
hundreds of concurrent human sessions, because a human takes seconds per turn. Memory per process, not per session,
sets the hosting cost.

M6a measured the full turn on a lightly loaded machine ([technical report](m6a_technical_report.md) §5). With its
frames, the view, the action menu and JSON, one turn takes 1.4 ms against `sp:0` and 2.3 ms against `om:0` (median),
and its response is about 2 KB gzipped. A 150-turn history re-simulates in about 0.09 s. A 150-turn bot game takes 0.27
to 0.48 s plus 0.1 s of frames. A session retains about 0.1 MB after 150 turns, more than the 22 KiB above because it
keeps the record and the belief history. The cold start took 2.4 s.

### 2.2 The session loop already works

The plan's reviewer ran the loop that the session core will use, read-only, on 2026-10-19. A
`LaneEnv(m45_calibrated(), record_ticks=True)` was reset with the recording's seed, `crn_policy("sp:0", ...)` drove
seat A and `crn_policy("om:0", ...)` seat B, with `decide` before each step and `observe_turn` after it on the
pre-turn state. It reproduced all 150 action pairs and the payoff of `h2h_om_s0_vs_sp_s0.json`, at one and at six
torch threads.

## 3. Correctness requirements

Every requirement below is a rule of the design and an acceptance check. M6a proves rules 1 to 5 and 7 on the real
finals before any UI exists. **It did so on 2026-10-21** ([technical report](m6a_technical_report.md) §3). It also
implements rules 10 to 13 in the session core, where tests cover them. M6b proves rules 6, 10, 11, 12 and 13 through
the API and the browser, and rule 8 for the records the app writes. M6c proves rule 8 for research records, and rule 9.
Rule 13 was added on 2026-10-21. Rule 5 now also keeps a drawn chance seed secret until the session stops, because a
known chance seed foresees the trade, gank and tower rolls.

| # | requirement | design rule | acceptance check |
|---|---|---|---|
| 1 | simultaneous decisions | When turn t begins, the server computes the opponent's code from the pre-turn state and the history up to t − 1, and stores it in the session before it sends the turn-t view. The visitor's submission completes the turn against the stored code. `observe_turn` runs after both codes are fixed, with the pre-turn state. The server draws the opponent's sampling seed and keeps it secret until the session stops, because the weights are public and a known seed would let a visitor compute the opponent's choice offline. No response before resolution carries the code, its probabilities or its logits | From a copy of the session taken before turn t begins, play turn t once for every legal visitor action. The opponent's code is the same in every branch, and it equals a fresh replay of the history up to t − 1. A schema test shows that no response before resolution carries the code, probabilities or logits |
| 2 | legal actions | The backend checks every submitted code against the legal set of the pre-turn state. The client's menu is a convenience | An illegal or malformed code returns an error and leaves the turn index, the state and the random stream unchanged |
| 3 | session isolation | Each session owns its env, its random streams, its policy instance (sampling stream, belief, previous codes) and its record. Only immutable objects are shared: network weights in evaluation mode, configurations, the stateless scripted types. All simulation runs on one worker thread per process, which also serializes the module-level caches (`m45b` lane cache, `lru_cache` tables) | Interleaving two sessions turn by turn gives the same records as playing them one after the other. Records from a concurrent load test equal their sequential replays |
| 4 | duplicate requests | A submission names its turn index. The same index with the same action returns the stored result. A different action for a resolved turn, or a future index, is refused. A per-session lock serializes submissions | A double submit and a retried submit resolve the turn once, and the record holds one entry per turn |
| 5 | deterministic replay | The server draws the chance seed unless the visitor chooses one, for example to play the same seed against the other opponent. The record marks a chosen seed, because it lets anyone foresee the chance rolls. A record holds the configuration and its digest, the chance seed, the policy seeds (revealed when the session stops), each seat's model id and file SHA-256 (or "visitor"), the belief mode and every turn's two codes with a state digest. Replay re-simulates and recomputes every policy seat's codes | Every record the app writes replays with equal state digests and equal policy codes. The session core reproduces `h2h_om_s0_vs_sp_s0.json` exactly when its policies drive both seats |
| 6 | faithful animation | The frontend animates only per-tick facts that the engine records and Python serializes into frames. Turn-level events appear as turn outcomes, never at an invented tick. The frontend computes no game rule | A schema test traces every frame field to an engine field. The frontend's frames for one recording equal `frame_view` |
| 7 | model identity | A registry describes each opponent: arm, seed, training budget, configuration, input width, file SHA-256, the library networks it needs and a link to the M5 evidence. The server checks every file's SHA-256 before loading and refuses to start on a mismatch. The play screen always names the opponent | A corrupted file stops the server at start-up. Each record's model fields match the registry |
| 8 | replay provenance | Every record carries one of three kinds: *research record* (an M5 recording, its SHA-256 in the bundle manifest, re-verified when the bundle is built), *new simulation* (both seats are policies, generated by the app) and *human game*. The kind is shown wherever a record is shown. The initial release opens only bundled research records and records the app itself generated; importing files is out of scope | The bundle build refuses a research record whose replay differs from its stored payoff. Each record view shows its kind |
| 9 | research claims | The app never claims performance against humans. M5 measured the policies against scripted styles, other policies and learned attackers only. The wording rules are in §9 | A copy review against §9 in M6c |
| 10 | session ending | A visitor can stop at any turn. Stopping is not a terminal state, carries no reward and declares no winner. The turn cap (proposed: 150) is a demo limit, stated as such. The stop summary describes the state (earned-gold lead, XP, levels, HP, absence, the wave), not a result | No win or loss wording anywhere. The summary names the stop turn and the reason |
| 11 | lane-exit assessment | Turn 94 is marked as the lane exit. A visitor who plays to turn 114 sees the undiscounted lead after 114 turns. M5 used that lead to realize E_20 at the turn-94 lane exit (`games.READ_TURN`). The app labels it as one game, not an expectation. No policy plays the visitor's seat for it. A continuation-based E_20 is not in the initial release. If it is added later, the screen names the continuation policy that takes over the visitor's seat and labels the number as that policy's expected result from the position | The panel's values equal the lead computed from the record. No E_20 appears without its continuation policy |
| 12 | belief visibility | The visitor chooses at the start whether the modelling opponent's belief is hidden (default) or shown. Showing it changes what the visitor knows, so the record stores the mode. After the session stops, the belief history is available in either mode. The belief view states that the library holds 6 scripted styles, 5 learned policies, a mixture and "unknown", and no human. In shown mode a visitor can try to steer the belief on purpose. M5 never tested deliberate deception, so the app makes no claim about it | In hidden mode the API returns no belief data during play. The record carries the mode |
| 13 | history navigation | The session's progress (the resolved turns) and the position on screen are separate. Selecting turn t shows the state after turn t, and position 0 is the initial position. Every displayed value follows the selected position. Browsing history never changes, advances or resubmits the game, and submission is disabled while a historical position is on screen. A history request returns resolved positions only, without the stored opponent code, a probability, a seed of an active session or a hidden belief | A history request leaves the session unchanged and refuses unresolved positions (M6a). In the browser, browsing then returning to the current turn restores the actionable position, and submission is disabled while browsing (M6b) |

## 4. Architecture

The recommendation keeps the user's starting proposal and adds four constraints: one process, one simulation
thread, static replay files and a generated type contract.

```
browser: React + TypeScript, built by Vite
   │  JSON over HTTPS, same origin
   ▼
one container: Python 3.13, CPU torch, lanegame
   FastAPI ── sessions (in memory, expiring) ── one simulation thread
            └─ model registry (loaded once at start-up, SHA-256 checked)
   static files: the built frontend and the replay bundle (frames as JSON)
```

### 4.1 Data shapes

The design rests on four shapes. Names are provisional.

- **`ModelCard`.** An immutable registry entry: `id` (`sp:0`, `om:0`), arm, seed, a description, the configuration
  and its digest, the input width, the file path and SHA-256, the dependencies with their SHA-256, and the evidence
  links.
- **`GameSession`.** A state machine with the states `awaiting_action(t)`, `resolving(t)` and
  `stopped(reason)`, where the reason is the visitor, the turn cap or expiry. It owns the id, the model card, the
  seeds, the seat assignment, the belief mode, the env, the policy, the stored opponent code for turn t and the
  record.
- **`TurnRecord`.** The turn index, both codes, the turn's `info` events, the reward, the state digest after the turn
  and, for `om`, the belief after the update. The server sends each turn's per-tick frames with its result and does
  not keep them, so a session stays near its measured size.
- **`Recording`.** The versioned app format: provenance kind and source, configuration digest, seeds, seats with
  model ids and SHA-256 or "visitor", the belief mode, every turn's codes and state digest, and optional frames.
  The 15 M5 recordings stay in their v1 format; a build step converts copies.

The API is a projection of these shapes. A first sketch is `GET /api/models`, `POST /api/sessions` (opponent, optional
seed, belief mode), `POST /api/sessions/{id}/turns` (turn index and code), `POST /api/sessions/{id}/stop` and
`GET /api/sessions/{id}/record`. A turn view carries the state summary, every action with its name and, when
unavailable, the reason, the last turn's frames and events, and the belief when it is shown. Research replays are
static files, not API calls.

M6a implemented these shapes in `lanegame/demo/` and proposes the full contract in the [technical report](m6a_technical_report.md)
§4. The contract adds `GET /api/sessions/{id}/history` (resolved positions only, rebuilt from the codes after a reload),
`POST /api/simulations` (Watch: one bounded bot game, generated in full and played back by the client), and frame,
recording and menu formats. History navigation runs on the client's cache of the turns it has received.

### 4.2 Why this shape

- **Python stays authoritative.** The browser renders what the server sends. It holds no copy of the rules, so the
  determinism contract and the legacy pins stay the only definition of the game.
- **Request and response are enough.** Turns are discrete. The opponent's code is fixed before the turn view is
  sent, so the visitor's submission completes the turn. No WebSocket or server push is needed. Watch returns a bounded
  game in one response (about 35 KB gzipped), and the client plays it back at any speed.
- **One process with one simulation thread.** Per-turn work is a few milliseconds (§2.1), and the process, at about
  250 MB, is the cost. One thread removes the races on the unlocked module caches and matches the one-thread torch
  setting M5 used.
- **Static replay.** Research recordings become frame files at build time, verified by re-simulation. The frontend
  reads them as static assets, so replay and the findings stay available when the inference backend is down or not
  deployed.
- **A typed boundary.** FastAPI's pydantic models produce an OpenAPI schema, and the frontend's TypeScript types are
  generated from it, so a field renamed on one side fails the build on the other.

### 4.3 Alternatives assessed

Only alternatives that bear on a concrete requirement are listed.

- **Rework Streamlit.** The user excluded it. Streamlit also re-runs the whole script on every interaction, which makes
  the per-session turn state and the animation timeline awkward.
- **Server-rendered pages (for example htmx).** These need a smaller toolchain, but the replay timeline and per-tick
  animation are client-side state, which a single-page application handles directly.
- **A full-stack JavaScript framework (for example Next.js).** It adds a second server runtime and meets no
  requirement that FastAPI with static files does not.
- **The simulator in the browser** (a TypeScript port, or Pyodide with policies exported to ONNX). It would remove
  the backend, but it duplicates the authoritative engine and its determinism contract, and torch does not run in
  Pyodide. Static replay already covers the case without a backend.
- **Renderer.** This stays open until the M6a prototypes. The lane has few objects (five cells, the Vision node, two
  towers, a handful of packets, two champions), and accessibility needs a text equivalent with any renderer. SVG gives
  DOM elements that can carry labels. Canvas suits many moving objects, which this lane does not have. The prototypes
  start with SVG unless they show a need for something else.

### 4.4 Repository layout

The proposed layout keeps everything in this repository:

- `lanegame/demo/` holds the Python registry, session core, record format, frames, action menu, bundle build and API.
  The registry, session, record, frames and menu exist since M6a. The API module alone will import FastAPI, so the
  session core is testable with pytest only. A new optional extra, `demo` (FastAPI and uvicorn), installs the server
  in M6b.
- `web/` holds the frontend with its own `package.json` (approved by the user on 2026-10-21; created in M6b).
- `prototypes/m6a/` holds the throwaway M6a prototypes, isolated from `web/`. They need no build step.
- `deploy/` holds the Dockerfile and the deployment notes.

## 5. Experience requirements

The primary audience is a visitor new to the project, such as someone curious about game AI or League of Legends,
or a technically literate reviewer. Researchers are a secondary audience who follow links to the evidence. These are
requirements, not a design. The M6a prototypes decide layout, theme, components, renderer, animation style and the
pattern of action selection.

- **The lane situation reads at a glance.** Both champions' positions, the waves and the direction they are pushing,
  towers and plates, HP, levels, gold and Recall or absence state are visible and labelled in plain words.
- **Legal choices are discoverable and explained.** Most actions combine a wave choice (Last Hit, Thin, Slow Push
  or Shove), a behaviour and an objective. Ward, Recall and Absent are single codes. The 61 codes are never a flat
  list. An unavailable choice says why.
- **Forced turns are quick.** While the visitor's champion is away after a Recall or a death, Absent is the only
  legal code. Continuing through those turns takes one control, and the server still resolves them one at a time
  with the opponent's code fixed first.
- **Simultaneity is visible.** The screen shows that the opponent has already chosen, without revealing what, and
  reveals both actions together.
- **The result of each turn is understandable.** Events appear in engine order where the engine records an order
  (ticks) and as turn outcomes otherwise, with the change in gold, XP and HP.
- **Diagnostics stay out of the way.** The wave panel, per-tick details, the belief and the lane-exit panel open on
  request.
- **The belief is inspectable when that is useful.** It is visible in "shown" mode and after a game.
- **One application.** Play, Watch, replay and the findings share the lane view, the turn history, the event
  presentation and the recording infrastructure.
- **Accessible and responsive.** Every action is reachable by keyboard. The lane state and the turn events have a
  text equivalent. No information is carried by colour alone. Motion respects the reduced-motion setting. The design
  is desktop-first, with an explicit responsive approach down to phone width.

The user journeys are:

1. **Play.** Choose an opponent, optionally a seed and the belief mode. Play turns, browsing the history at any time.
   Stop, or reach the cap. Read the summary and download the record.
2. **Watch a new simulation.** Choose a bot for each seat. The server generates a bounded match, and the visitor
   explores it through the timeline with autoplay and speed control. It is labelled a new simulation.
3. **Watch a research game.** Pick a verified recording, see its provenance and what it shows, and step or play
   through it.
4. **Learn the findings.** Read the findings page, then jump into replays at specific turns and into the reports.
5. **Reproduce.** Run the demo from the README and its bundle, or regenerate the analysis from the evidence bundle.

### 5.1 The turn history (approved requirement, 2026-10-21)

The interaction borrows Chess.com's move history, not its branding. The starting layout is a prominent lane view with
a compact turn history beside it and navigation controls nearby; the prototypes vary the arrangement.
- One history row is one complete simultaneous turn and shows both players' actions.
- Selecting turn t shows the state after turn t. Turn zero is the initial position.
- First, previous, next and latest controls, plus clicking a row. Autoplay, pause and playback speed in Watch and in
  the review of a completed game.
- Browsing the history during play never changes or advances the game. A historical view is clearly distinct from
  the current decision point and has a prominent "Return to current turn" control. Submission is disabled while
  browsing. Returning restores the actionable position.
- Everything on screen follows the selected turn: the lane, HP, gold, XP, levels, absence, events and a permitted
  belief.
- No history endpoint or client data exposes an unresolved opponent action, a future state or a hidden belief.
- No undo, branching, "best move", "blunder", win probability or engine-style grading. The screen shows the
  simulator's factual outcomes.

### 5.2 Clarity (approved requirements, 2026-10-21)

The main screen puts, in order: the lane (champions, waves, towers, the immediate situation); essential status (HP,
levels, the resource difference, the current turn, Recall or absence); the primary interaction (action selection in
Play, playback in Watch); a compact, navigable turn history; and optional detail (beliefs, wave statistics, tick traces,
the full event log).
- The lane takes the largest share of the screen. No permanent panel for every simulator variable. Concise
  information sits near the object it describes where that reads better.
- Controls and the lane stay where they are as turns advance. Expanding logs and changing text do not shift the
  layout.
- Important outcomes are summarized, with full details on demand. Advanced diagnostics start collapsed. Research
  explanations live in a separate view or a deliberate expansion.
- The 61 codes never appear as a flat list. Selection is compact: a wave action, a behaviour and an objective, plus
  Ward and Recall as single actions. Unavailable choices are explained with the authoritative legality predicates.
  Selectable choices, the selected action and the final submission control look different from each other.
- Forced Absent turns take one control and still resolve one turn at a time.
- Human control means choosing actions in the discrete simulator: no free movement and no new spatial resolution.
- Animation shows recorded per-tick facts only. Events known only per turn appear as turn outcomes. Nothing implies
  individual units, positions or timing the engine does not model.

## 6. Milestones

Effort uses the roadmap's sizes: S is about one working session, M two to three, L four or more. The figures are
rough, and the main uncertainty is named for each milestone.

### M6a. Product definition and technical readiness

*Status (2026-10-21): the backend outcome is complete, and the design outcome awaits the user's choice of a
direction.*

- **Objective and outcome.** Two outcomes: a trustworthy game-session backend, and a frontend direction the user has
  reviewed. The backend meets the correctness rules before any UI is built. The user sees two or three clickable
  prototypes driven by verified recorded games, chooses a direction, and M6a then refines it into a representative
  Play and Watch experience and records the final decisions.
- **Scope.**
  - The headless session core (§11): done.
  - Frame exports of verified recordings for the prototypes: done (the three seed-0 games and one new simulation).
  - The user journeys and the API, recording and frame contracts: done, proposed ([technical report](m6a_technical_report.md) §4).
  - Three throwaway prototypes of the lane view, the turn history and the turn loop in Play and Watch: built,
    awaiting review.
  - A short decision record ([m6a_decision_record.md](m6a_decision_record.md)), including the deployment structure.
  - After the user's choice: one refinement of the chosen direction, and the final decisions recorded.
- **Exclusions.** The production frontend, deployment itself, hosting, publishing bundles and the Streamlit archive.
- **Dependencies.** The user's review of this plan (given in the 2026-10-21 brief).
- **Deliverables.**
  - `lanegame/demo/` (registry, manifest, session, record, frames, action menu), `lanegame/experiments/m6a/`
    (stages `verify`, `measure`, `export`) and `tests/test_demo_session.py`: done.
  - `docs/m6/m6a_technical_report.md` and `docs/m6/m6a_decision_record.md` (the plan's "product definition"
    split in two): done, the decision record pending the design choice.
  - `prototypes/m6a/` (code, isolated from `web/`, no build step) and screenshots under `results/m6/prototypes/`.
  - The measured cost of a full turn view with frames, of a history and of a bounded bot game: done.
- **Acceptance checks.**
  - Rules 1 to 5 and 7 of §3 pass in pytest on the real seed-0 finals: passed (27 tests).
  - The session core reproduces `h2h_om_s0_vs_sp_s0.json` when its policies drive both seats: passed, exactly, and
    so do the two seed-0 `g_shove` games.
  - The user chooses a prototype direction from the running prototypes and their screenshots: pending.
- **User decision.** The prototype direction, then acceptance of its refinement.
- **Effort.** S for the session core (as estimated). M in all. The design iteration remains the uncertainty.

### M6b. The playable application

- **Objective and outcome.** A visitor plays a full game against either opponent, and watches a bounded bot match,
  on a local install. This is the first playable milestone. In Play the visitor chooses an opponent, starts a seeded
  game, inspects the lane, submits a legal action, sees both actions and their consequences, browses the history, and
  continues, restarts or stops. In Watch the visitor picks a bot for each seat, and the app generates the match and
  plays it back.
- **Scope.**
  - The FastAPI app over the session core, with the legality explanations, the turn view, the history endpoint and
    the bounded simulation endpoint (run in a separate worker, §8.2).
  - The lane view, the turn history and the playback controls in the chosen direction, shared by Play and Watch.
  - The belief view in both modes.
  - The stop summary and the record download.
  - The lane-exit panel (the turn-94 marker and the turn-114 lead).
  - End-to-end browser checks, and a new feature file in `verify-lanegame` for the web app.
- **Exclusions.** Research replays and the findings (M6c), release hardening (M6e), continuation-based E_20, record
  import.
- **Dependencies.** M6a, including the chosen direction.
- **Deliverables.** `lanegame/demo/api.py`, `web/`, their tests, and the `verify-lanegame` feature.
- **Acceptance checks.**
  - Rules 6, 10, 11, 12 and 13 pass their checks, and so does rule 8 for the records the app writes. Rules 1 to 5 and
    7 still pass through the API.
  - A browser run plays 150 turns against each opponent, and the downloaded record replays bit for bit.
  - A browser run generates a Watch match, steps and autoplays it, and its record replays bit for bit.
  - Browsing the history during Play changes nothing in the session, and "Return to current turn" restores the
    decision point.
  - Keyboard-only play works.
  - Screenshots at phone and desktop widths are reviewed.
  - The user reviews the build in chat with screenshots and a short video.
- **User decision.** Acceptance of the playable build.
- **Effort.** L. The frontend is the first production JavaScript in the repository, and the lane view, the history
  and the action selection may need several iterations. The backend part is S to M, since the core exists.

### M6c. Research demonstration and explanation

- **Objective and outcome.** A visitor learns what M5 found and sees it in games. The outcome is a findings page and
  replays of the 15 verified recordings with provenance and short guided notes.
- **Scope.**
  - A bundle build that converts the M5 recordings to the app format, verifies each by re-simulation and writes
    frame files.
  - Guided notes for the three seed-0 demonstration games (`h2h_om_s0_vs_sp_s0`, `om_s0_vs_g_shove`,
    `sp_s0_vs_g_shove`) and an overview of the 12 attacker games. The two `g_shove` games have different seeds and
    seats. Each was selected separately as a median game, so the notes never present them as a paired comparison.
  - The findings page (§9), with the six-seed numbers, what one unit of lead means (from the mathematical reference)
    and links to the evidence.
  - Figures only from existing records.
- **Exclusions.** New evaluations, new statistics, new training, and choosing recordings by their performance. The
  recordings were selected by M5's fixed upper-median rule, and M6 keeps that selection.
- **Dependencies.** M6b's shared lane view and timeline; M6a's frame format.
- **Deliverables.** The bundle build in `lanegame/demo/`, the findings and replay views in `web/`, and the reviewed
  text.
- **Acceptance checks.**
  - The build refuses any recording whose replay differs from its record.
  - A script checks every number on the findings page against `results/m5/analysis_main.json` and
    `strategic_main.json`.
  - Replays and the findings work with the API stopped.
  - The text passes the review against §9. Rule 8 for research records and rule 9 pass their checks.
- **User decision.** Approval of the findings text.
- **Effort.** M. Writing explanations a newcomer can follow is the main work. How much explanation is enough is
  the uncertainty, and a reading by someone new to the project settles it.

### M6d. Documentation and reproducibility

- **Objective and outcome.** A reader can run the demo, regenerate the M5 analysis or rerun the campaign from the
  documentation alone. The outcome is a concise README, an accessible research overview and a reproduction guide with
  three bundles.
- **Scope.**
  - A shorter README that links to the overview, the roadmap and the reports.
  - `docs/m6/research_overview.md`, written for a newcomer.
  - The reproduction guide with three paths: run the demo, regenerate the analysis, rerun the campaign.
  - A packing command that writes each bundle and its SHA-256 manifest (§8).
  - A re-run of `reproduce` on the committed tree, with the 5 library networks added to the key-checkpoint table.
  - The Streamlit archive (below).
- **The Streamlit archive.**
  - Tag the last commit that has the viewer in place.
  - Move `lanegame/viewer/app.py` and `render.py` to `legacy/streamlit_viewer/`, with a README that names the tag and
    the command that runs the old viewer at it.
  - Keep `lanegame/viewer/session.py` and `policies.py` where they are, because 11 experiment modules and 7 test
    files import `lanegame.viewer`.
  - Move the two tests in `tests/test_viewer.py` that import `lanegame.viewer.render` with the archive.
  - Point the generated viewing instructions in `m5/traces.py` and `m43b/trajectories.py` at the new app or the
    tag. Leave the stored results files as they are.
  - Update the live indexes (README, `docs/README.md`, `CLAUDE.md`, the `viewer` extra, `verify-lanegame`). Leave the
    historical reports unchanged, since they describe their own time.
- **Exclusions.** Retraining, new analysis and edits to historical reports or frozen pre-registrations.
- **Dependencies.** M6b (the commands to document), the user's commit of M5 (done in `a07e3fdf`, apart from the
  git-ignored reports under `docs/`), and the license decision before any bundle is published (§10).
- **Deliverables.** README, `docs/m6/research_overview.md`, `docs/m6/reproduction.md`, the packing command, the
  regenerated `results/m5/reproducibility.md` and `legacy/streamlit_viewer/`.
- **Acceptance checks.**
  - In a fresh environment, the evidence bundle with `python -m lanegame.experiments.rl.m5 prereg_check train_check
    attack_check analyse report` reproduces `results/m5/report.md`.
  - The tag runs the old viewer, and the full suite passes after the move.
  - Every manifest's SHA-256 matches its files.
- **User decisions.** A license, and where the evidence and training bundles are published.
- **Effort.** M. Uncertainty is low. The fresh-environment run may expose path assumptions, such as the Windows paths
  inside the v1 recordings.

### M6e. Release readiness

- **Objective and outcome.** The initial release is ready to deploy, and the user has what the deployment decision
  needs. The outcome is a container image that runs the app from the demo bundle, measured limits, deployment
  instructions and an approval checkpoint.
- **Scope.**
  - Clean-install checks: a fresh clone, a fresh virtual environment and a container build.
  - A load measurement: concurrent sessions, memory, latency per turn view.
  - The safeguards of §8.
  - The Dockerfile, the deployment notes for a generic container host, and a release checklist.
- **Exclusions.** Buying hosting, deploying, a domain, accounts, and analytics beyond aggregate counters.
- **Dependencies.** M6b, M6c and M6d.
- **Deliverables.** `deploy/Dockerfile`, `docs/m6/deployment.md`, `results/m6/load_test.json` and the checklist.
- **Acceptance checks.**
  - The image starts from itself and the bundle alone, and refuses a tampered bundle.
  - The load test meets budgets set from M6a's measurements: latency per turn view at the 95th percentile, and memory
    at the session cap.
  - The test suite and the frontend checks pass inside the image.
  - The deployment checkpoint is presented to the user.
- **User decisions.** Provider, budget, domain, and go or no-go.
- **Effort.** M. The constraints of a provider (memory limits, cold starts, idle shutdown) are unknown until one is
  chosen.

## 7. The initial release

The initial release is the smallest version that demonstrates the project well. Human versus policy is its centre,
and Watch is its second primary mode (the user, 2026-10-21).

| feature | initial release | later | out of scope |
|---|---|---|---|
| play against `sp:0` or `om:0` from a chosen or random seed | ✓ | | |
| Watch: a bounded bot-versus-bot match between `sp:0` and `om:0` in either seat, generated first and played back through the timeline, labelled *new simulation* | ✓ (moved from "later" on 2026-10-21) | | |
| the Chess.com-style turn history in Play, Watch and replays, with autoplay and speed in Watch and review (§5.1) | ✓ | | |
| scripted styles as Watch opponents or Play opponents | a design decision (the core supports the six library archetypes) | | |
| legal actions with names and reasons, both actions revealed together, the turn's events | ✓ | | |
| belief hidden or shown (chosen at the start), the full belief history after the game | ✓ | | |
| stop at any turn, a 150-turn cap, a summary without a winner | ✓ | | |
| the lane-exit marker at turn 94 and the turn-114 lead | ✓ | | |
| download one's own record, replayable bit for bit | ✓ | | |
| replays of the 15 verified M5 recordings with provenance | ✓ | | |
| guided notes for the three seed-0 games | ✓ | more recordings with notes | |
| the findings page with the six-seed numbers and links | ✓ | | |
| continuation-based E_20 with a named continuation policy | | a user decision | |
| importing a record file | | possible | |
| a visible commitment to the opponent's action (a hash revealed after the turn) | | possible | |
| other seeds' finals as opponents | | possible, never chosen by their results | |
| accounts, multiplayer, matchmaking, leaderboards, payments, a scenario editor | | | ✓ |
| new mechanics, rewards, objectives or training | | | ✓ |

Watch was planned as the first follow-up and is now in the initial release. A request plays one bounded game (at
most 150 turns, 0.27 to 0.48 s of simulation plus 0.1 s of frames, M6a) between registry policies, labelled *new
simulation*: an example, never evidence for M5's findings. The client receives the whole game and plays it back, so
the simulation does not run at animation speed. It adds a provenance class, a rate limit and a separate worker
(§8.2).

## 8. Bundles and hosting

### 8.1 Three bundles

| bundle | contents | size | who needs it |
|---|---|---|---|
| demo | `sp:0`, `om:0` and the 5 library networks with a SHA-256 manifest; the 15 recordings converted, with frames; the built frontend | a few MB (the networks are about 1.1 MB, the v1 recordings 452 KB; frames add more) | every visitor's server, and anyone running the demo locally |
| evidence | `results/m5/` (7.7 MB, all of it tracked since `a07e3fdf`) plus the git-ignored records under `results/rl/checkpoints/m5/`: evaluation, battery, cells, states, attacker evaluation, run summaries, and the finals the gates verify; no snapshots | about 1.1 GiB, estimated from the record sizes in `reproducibility.md`; measured when packed | a reader regenerating the M5 tables |
| training | the whole `results/rl/checkpoints/m5/` tree (4.3 GiB, of which 2.6 GiB are training runs and snapshots, 0.8 GiB battery records and 0.75 GiB attacker runs), the M4.6c starts and archive, the 5 library networks | about 4.3 GiB or more | a reader rerunning the campaign (about 36 h on six workers) |

Visitors never download the evidence or training bundles. The app needs only the demo bundle.

### 8.2 Initial topology

- **One container, one process.** The process loads the model registry once at start-up, checks every SHA-256 and
  refuses to start on a mismatch.
- **Sessions in memory.** Each session expires after an idle period (proposed: 30 minutes). The server caps the
  number of live sessions. M6a measured about 0.1 MB retained per session after 150 turns, record and belief history
  included. A thousand sessions therefore stay near 0.1 GB, so the cap guards CPU and abuse rather than memory.
- **Bounded work.** A session stops at the turn cap. One request resolves one turn. A Watch request plays at most 150
  turns, and a future continuation readout would use a fixed K. Each takes up to about 0.5 s (Watch, M6a) or 2 s (a
  continuation, plan §2.1) on the measured machine, which would stall every session on the one simulation thread, so
  they run in a separate worker with its own limit.
- **Request limits.** Session creation and turn submission are rate-limited per client address in the process.
  Request bodies are small and size-limited. No request names a file path, and no file is uploaded.
- **Privacy.** No accounts and no personal data. Records stay in memory until the session expires, and a visitor
  downloads their own. The server logs only aggregate counters and errors.
- **Same origin.** The process serves the API and the static frontend, so no CORS configuration is needed.
- **Replay without inference.** Replays and the findings are static files. They can also be hosted on a static host
  without the API, in which case play is shown as unavailable.

### 8.3 Scaling later

Scale only after measuring real usage. If more than one server process is ever needed, sessions held in memory need
either sticky routing by session id, or a shared store of each session's seeds and codes. The determinism contract
allows the second, because any process can rebuild a session by replaying its codes. A rebuild costs about 1 to 3
ms per turn (§2.1). Neither is part of the initial release.

## 9. Presenting the research accurately

Quantitative claims use all six seeds. Each value is a mean over the six seed pairs with a 95% t interval. The
seed-0 pair (`om:0`, `sp:0`) is the demonstration pair by M5's convention, chosen by index and not by performance. M6
does not pick another seed after inspecting results. Each lead is the undiscounted earned-gold lead after 114
turns, which realizes E_20 at the turn-94 lane exit.

| claim | evidence (M5e report) |
|---|---|
| Both arms retain four registered strategic findings, in 6 of 6 seeds each: conditional wave management, conditional freezing and the reply to freezes, selective trading, and Recall timing. Concession under threat does not replicate in either arm | §4.1; `results/m5/strategic_main.json` |
| The modelling package gains substantially against library-like scripted styles, by +9.26 [+8.95, +9.56]. The gain is largest against the library archetypes (+12.2), smaller against held-out scripted variants (+6.4) and absent against conditional styles built on a competent policy (−0.2 to −0.7) | §3.1 |
| Pure self-play is better against competent opponents. The package is −0.63 [−0.98, −0.27] against competent finals and −0.44 [−0.72, −0.16] head to head, in every seed | §3.1 |
| Both arms are exploitable by dedicated attackers. The attackers' lower bounds are +2.96 [+0.32, +5.61] against the modelling finals and +1.15 [+0.34, +1.96] against the pure self-play finals, positive in every seed | §5.2 |
| The difference between the two lower bounds (+1.81 [−0.56, +4.19]) is inconclusive. Neither "more exploitable" nor "equally exploitable" is shown | §3.2, §5.2 |
| M5 compares whole packages. It does not show whether the competence cost comes from the belief inputs, the fixed-opponent games or the reduced self-play exposure | §6, §11 |

Wording rules for every page, note and caption:

- Show the gain and the cost side by side. Never combine them into one score, and never call either arm better
  overall.
- Call neither arm unexploitable, and call neither robust to a dedicated attacker. The modelling package's
  registered robustness covers mixtures and switchers only (S1).
- Say that M5 never evaluated play against humans, so a human game shows how the policies respond to one person and
  measures nothing about humans in general.
- Mark a single game, whether a replay, a new simulation or a visitor's game, as an example, not evidence. Two
  separately selected games are not a paired comparison. A Watch match between the two agents shows how they play,
  not which is better.
- Declare no winner. A stopped or capped game shows its state, and the lead it shows is the earned-gold lead, not a
  score or a win chance.
- In shown-belief mode, say that steering the belief on purpose is possible and that M5 never tested deliberate
  deception.
- Name the limitations from M5e §11 where they bear on a claim: pure self-play had not converged, the style families
  favour the library, and the attackers are approximate lower bounds that do not test deception.
- No further training is needed for M6 because pure self-play had not converged or because S6 is inconclusive. Those
  questions belong to the optional research in M5e §14.

## 10. Decisions

The three decisions the first version asked for were settled in the user's M6a brief of 2026-10-21:

1. **The milestones and the initial release: approved with one change.** Watch, live bot-versus-bot simulation, moves
   into the initial release (§7). A continuation-based E_20 stays out of it.
2. **The `docs/` line in `.gitignore`: removed.** The reports and this plan are tracked.
3. **Where the frontend lives: `web/` in this repository**, React with TypeScript and Vite over FastAPI.

Open now:

4. **The visual direction**, chosen from the M6a prototypes. *Decided on 2026-10-21:* C as the foundation, with A's
   action controls and a better gold-lead graph ([decision record](m6a_decision_record.md) §6). Open: the user's review
   of the refined interface, which M6a waits for.
5. **Scripted opponents** in Play or Watch, and which ones (the decision record assesses a small selection).

Later:

6. **A license and public distribution, before M6d publishes any bundle.** The repository has no license. Publishing
   needs one for the code, and a decision on whether the checkpoints and recordings are distributed. The public app
   also needs original visuals with no Riot Games art, logos or champion names. Check Riot's published policy for
   fan projects before release.
7. **Provider, budget and go or no-go, at the M6e checkpoint.**

Defaults applied in this plan, each reversible on request:

- the belief hidden by default and chosen at the start;
- a 150-turn cap;
- the visitor in seat A;
- a chance seed drawn by the server unless the visitor picks one, a separate policy seed, and both kept secret until
  the session stops (a chosen chance seed is marked in the record);
- the opponent sampling its actions as M5 evaluated it, not playing greedily;
- no server-side storage of human games;
- the Streamlit archive moving only the UI files.

## 11. The first implementation task (done, 2026-10-21)

The headless session core is built as planned, with no web framework and no UI, and with the additions the second
inspection called for. The [technical report](m6a_technical_report.md) describes it, lists every check and its result
(§3), and gives the measurements (§5).

- **Files.** `lanegame/demo/__init__.py`, `manifest.json`, `registry.py`, `session.py`, `record.py`, `frames.py`,
  `actions.py`; `lanegame/experiments/m6a/`; `tests/test_demo_session.py`.
- **Additions to the planned task.** The frames and the action menu (planned for M6b), scripted seats (needed to
  re-simulate the two seed-0 `g_shove` recordings), the history endpoint's core (`GameSession.history`), bounded bot
  games (`simulate`), and the drawn chance seed kept secret while a session is active.
- **Checks.** Rules 1 to 5 and 7 pass on the real seed-0 finals. The session core reproduces
  `h2h_om_s0_vs_sp_s0.json` with the same actions, `payoff_A` exactly and the belief's log posteriors at the check
  turns identical. A modified checkpoint file is refused before `torch.load`. The full suite result is in
  `results/m6/regression_suite_m6a.log`.
- **Changed nothing** in any existing module, mechanic, preset, network or recorded result.

## Appendix A. Pointers

- The M5 findings and their limits are in [m5e_closeout_report.md](../m5/m5e_closeout_report.md) §1 to §7, §11 and
  §14.
- Commands, seeds and digests are in `results/m5/reproducibility.md` and `results/m5/inventory.json`.
- The demonstration recordings are in `results/m5/traces/`, indexed by `results/m5/traces.json`.
- The opponents and their inputs are built in `lanegame/experiments/rl/m5/players.py`, `m5/common.py`,
  `experiments/rl/om2/common.py`, `om1/common.py` and `lanegame/opponent/belief.py`.
- The lane-exit readout is `lanegame/diagnostics/lane_exit.py`. M5's turn-114 lead is `READ_TURN` in `m5/games.py`.
- Frames and event text come from `lanegame/viewer/session.py`, and the per-tick trace from
  `mechanics.step_turn(record=True)`. The demo's frames, recordings, menu and session are in `lanegame/demo/`, and
  their contracts in the [M6a technical report](m6a_technical_report.md) §4.
- The M6a prototypes are in `prototypes/m6a/` (launch instructions in its README), compared in the
  [decision record](m6a_decision_record.md).
- The rules of the game are in [laning_game_mathematical_reference.md](../m4/laning_game_mathematical_reference.md).
- The design standard is [design_philosophy.md](../design_philosophy.md).
