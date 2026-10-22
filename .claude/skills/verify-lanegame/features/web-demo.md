# Web demo (Play and Watch)

The M6 web application: a visitor plays one seat against a trained M5 agent (Play) or watches a bounded match between
two agents (Watch), with a turn history that never changes the game, stepped ticks, a gold-lead graph, Details and a
downloadable record. The Python server is authoritative; the page computes no rule.

## Sub-features

- WD-1 Start a game against either agent (self-play `sp:0`, adaptive `om:0`), belief hidden or shown.
- WD-2 Submit an action; the opponent's action is fixed before yours and both are revealed; the turn steps through
  start, tick 1, tick 2 and the end.
- WD-3 Unavailable choices stay visible and explain why (the engine's reason).
- WD-4 Browse the history (rows, the turn buttons, the graph, ← → Home End) without changing the game; Return.
- WD-5 Reload resumes the same session; Stop declares no winner and reveals the seeds; the record downloads and
  replays bit for bit in Python.
- WD-6 Forced away turns (after a Recall or a death) take one Continue control.
- WD-7 Watch: a verified research recording or a generated match; Play, Pause, speed, First/Previous/Next/Last.
- WD-8 The same app at phone sizes, in dark mode, with history and the large graph open, keeps the lane clear.
- WD-9 Sequence (M6c): a recorded research game opens paused at the sequence's start with a note panel; Next
  event and Play walk the turns to the next event and stop there; Previous event jumps back; a manual move away
  from an event keeps the note labelled with its turn and offers "Return to this event"; notes can be hidden;
  closing the sequence keeps the replay.

## How to get to it (user POV)

- `python -m lanegame.demo` from the repo root (needs `pip install -e ".[demo]"`, the 7 checkpoints of
  `lanegame/demo/manifest.json`, and `cd web && npm install && npm run build` once), then http://127.0.0.1:8765/.
- The page opens in Play with a fresh game (or the game the browser remembers). Watch is the second mode button.
- Query parameters set up states for screenshots: `?fresh` (ignore the remembered game), `?opp=om:0&belief=shown`,
  `?seed=N` (a chosen chance seed), `?mode=watch&game=h2h_om_s0_vs_sp_s0&view=30`, `?layout=rail|dock|console`,
  `?theme=dark`, `?ticks=auto|manual`, `?still` (no animation), `?history=open`, `?graph=expanded`, `?detail=events`,
  `?mode=watch&sequence=stacking-to-crashing` (a sequence, M6c).

## Driving it with vl.py

Preconditions: `$VL demo start --run "$RUN"` printed `engine ready`, `$VL browser start --run "$RUN"` succeeded, and
`$VL browser open --run "$RUN" --url "<demo url>?fresh&ticks=manual"` returned `ready`. The app's state hook is
`window.__demo.state()` (progress, view, menu, turns, live, stopped, problem, …); read it with `browser js`.

- **WD-1 start.** `$VL browser js --run "$RUN" --expr "window.__demo.state()" --label wd1-start`: `live` true,
  `progress` 0, `canSubmit` true, `menu.legal` non-empty, `seeds.chance` null (secret while active).
  `$VL browser screenshot --run "$RUN" --label wd1-start` shows the dock with "Turn 1 of 150 · your move".
- **WD-2 submit.** `$VL browser click --run "$RUN" --text "Slow push" --no-settle`, then `--text "Hold"`, then
  `--text "Play turn 1: Slow push, Hold"`. The step panel shows "Both actions revealed"; `--text "Next tick"` twice,
  `--text "Show the result"`, `--text "Continue to turn 2"`. Then `state()` has `progress` 1 and `turns[0][0]` 46.
  Proof against the engine: `curl <demo url>api/sessions/<sid>/history?from=0&to=1` (sid from `state().sid`) lists
  the same codes.
- **WD-3 reasons.** At the start, `state().menu.singles[1]` (Ward) has `legal` false and a `reason` naming the steps
  to the ward spot; the Ward button shows it; Pressure's row says why it is unavailable.
- **WD-4 history.** `$VL browser js --expr "window.__demo.goto(0)"`, then `state()`: `view` 0, `progress`
  unchanged, `canSubmit` false; the banner reads "Viewing the start · the game is at turn N". `--text "Return to
  current turn"` restores `view` = `progress`.
- **WD-5 reload, stop, download.** `$VL browser open --run "$RUN" --url "<demo url>"` (no `?fresh`): `state().sid`
  and `progress` unchanged. `--text "Stop game"`: `stopped` true, `seeds.chance` a number, the summary says
  "declares no winner". `python -m lanegame.experiments.m6b browser` performs the download and the Python replay;
  by hand, "Download the record" saves `laning-game-<id>.json`, and
  `python -c "import json; from lanegame.demo import registry, session; print(session.replay(registry.Registry().load(), json.load(open('laning-game-<id>.json'))))"`
  reports `turns` equal to the game's length.
- **WD-6 away.** In a fresh game click `Recall`, then the submit button: the next panel offers only `Continue`
  (`state().menu.forced` 45); `--text "Continue"` resolves the away turn as code 45.
- **WD-7 watch.** `--text "Watch"`: `state().kind` `research_record`, `progress` 150. `--text "New game"` (the top
  bar) then `--text "Simulate"`: `kind` `new_simulation` within about 10 s. `--text "Play"` then `--text "Pause"`;
  `--text "4×"`; the turn buttons by their aria labels are `btnFirst`/`btnPrev`/`btnNext`/`btnLast`
  (`js --expr "document.getElementById('btnLast').click()"`).
- **WD-8 layouts.** `$VL browser open --url "<demo url>?mode=watch&game=h2h_om_s0_vs_sp_s0&view=147&still&theme=dark"`
  at `browser start --width 844 --height 390` for the rail; `js --expr "window.__demo.boxes()"` lists the scene's
  object boxes for an overlap check against `#surface`, `#topbar`, `#graphBig`.

- **WD-9 sequence.** `$VL browser open --url "<demo url>?mode=watch&sequence=stacking-to-crashing"`:
  `state().guide` has `id` "stacking-to-crashing", `at` null, `notes` true, `atEvent` true; `state().view` 74,
  `state().kind` `research_record`, `playing` false. `js --expr "window.__demo.guideNext()"`: `guide.at` 0 at once,
  `guide.walking` true while `view` climbs 75, 76, 77 (0.75 s a turn at 1x), then `playing` false at 77 and the
  panel shows "Trades keep the bot from going back to base" and "Next: Nobody clears the wave, so it stacks".
  `js --expr "window.__demo.goto(78)"`: `guide.atEvent` false and the panel offers "Return to this event";
  `--text "Return to this event"` restores `view` 77. `--text "Play"` walks to 80 and stops.
  `js --expr "window.__demo.guideNotes(false)"` collapses the panel to one line; `window.__demo.guideClose()` keeps
  the replay and clears `state().guide`. In a live game `state().guide` is always null.

The complete scripted journey, with screenshots and the overlap check at five window sizes in both themes, is
`python -m lanegame.experiments.m6b browser` (about 4 minutes; evidence in `results/m6/m6b_browser.json` and
`results/m6/m6b/`). The sequences' own check is `python -m lanegame.experiments.m6c browser` (evidence in
`results/m6/m6c_browser.json` and `results/m6/m6c/`).

## Gotchas

- `browser click --text` needs `--no-settle` here: the settle logic waits for Streamlit's indicators, which this
  page does not have. Prefer `browser js` with the hooks for state, and `click` for the real control.
- A submitted turn is "busy" until its frames have been shown; with `?ticks=manual` the step panel waits for you.
  Use `?ticks=auto` or `window.__demo.quick(code)` to resolve and show a turn at once.
- The remembered game lives in the browser's localStorage per origin: a new run's throwaway profile starts clean,
  but the same profile reopened without `?fresh` resumes the last game if the server still holds it.
- Watch's "New game" needs the simulation worker (`state().engine.simulations` `ready`); it starts about 3 s
  after the server.
- Never run the journey against a server you did not start in this run: it stops sessions and starts new ones.
