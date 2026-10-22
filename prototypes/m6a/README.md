# M6a demo and prototypes (throwaway)

`glass/` is the M6a design the user approved on 2026-10-21, and the reference for the production frontend (`web/`,
M6b). A top-down lane fills the window, and every control sits on one frosted glass surface over it. **Play in it is
live.** Your action goes to the tested session core and resolves. The other folders are history:

- `hud/`, the game view with separate panels (saved by the user before the glass round);
- `refined/`, the stacked layout of the first refinement;
- `arena/`, `board/` and `story/`, the three first directions. Their Play screens are staged walkthroughs that replay a
  recording.

None of this is the production frontend. There is no build step and no npm.

## Launch

From the repository root:

```bash
.venv/Scripts/python.exe prototypes/m6a/serve.py
```

It loads the engine (about 3 s), then prints `M6a demo: http://127.0.0.1:8765/`. Open that address. The server
listens on this machine only. Stop it with Ctrl+C. The engine needs the 7 git-ignored checkpoint files of
`lanegame/demo/manifest.json`. Without them, or with `--no-engine`, the pages still load, Watch shows the recorded
games, and Play says that the engine is unavailable.

## What to try in `glass/`

- **Play.** Choose the opponent and whether its belief is shown, then play. Pick a wave action and a behaviour (and
  Pressure when it is available), or Recall or Ward, and submit. Unavailable choices show the engine's reason.
- **Ticks.** A resolved turn pauses at the start, tick 1, tick 2 and the end, each with a plain description. Press
  Space, the right arrow or the step button to go on, or "Skip to the end of the turn". "Play ticks automatically"
  (beside the step button and in About) runs the steps by itself.
- **History.** Click a row or use ← → Home End to look at earlier turns. Submission is disabled while you do.
  **Return to current turn** (or Esc) brings back the decision.
- **Gold lead.** Blue above the line when the left player (you) has earned more gold, red below when the right player
  has. Hover to read a turn's lead, and click to jump there. The graph opens large on request.
- **Watch.** A recorded research game, or a new game between two agents ("New game between agents"). Autoplay,
  pause (Space) and speed (0.5× to 4×).
- **Details.** The full event log, wave numbers and ticks, champion numbers, the belief when it may be shown, and the
  game record.
- **About.** Light, dark or the system's theme, the layout, and the tick setting.
- **Layouts.** The dock is the default. A window wider than it is high and at most 520 px high (a phone held
  sideways) gets the rail. A window under 700 px wide (a phone held upright) gets a compact full-width dock. The
  layout follows every resize and rotation. `?layout=dock|rail|console` forces one.
- **The lane.** Each ball is 1 HP of minions. The engine tracks each wave's HP, not single minions, and a new wave is
  8. The ward spot is the bush beside the middle of the lane. A champion in the bush is drawn see-through, though the
  simulator hides no one.

## Limits

- **Every game is an example.** Your game and a new simulation are not evidence for M5's findings. The three seed-0
  recordings are verified M5 games, which the session core re-simulates exactly. `sim_om_vs_sp` is a new simulation
  made for the prototypes (chance seed 110,000, policy seed 110,500).
- **The recordings' data files hold whole games**, which suits Watch and review. A running game only ever receives
  resolved turns from the server.
- **Local only.** Sessions live in the server's memory and end when it stops. There is no record download yet.

## Files

| path | content |
|---|---|
| `index.html`, `serve.py` | the launcher, and the local server with the JSON API over `lanegame/demo/service.py` |
| `glass/` | the approved design (the reference for M6b) |
| `hud/`, `refined/` | earlier versions with live Play |
| `shared/core.js` | data loading, the history controller (progress kept apart from the viewed position), playback, keys |
| `arena/`, `board/`, `story/` | the three first directions (staged walkthroughs) |
| `data/` | the recorded games in the demo's frame contract, written by `python -m lanegame.experiments.m6a export` |
| `tools/shoot.py` | screenshots in headless Edge: `.venv/Scripts/python.exe prototypes/m6a/tools/shoot.py prototypes/m6a/<folder>/shots.json results/m6/prototypes/<folder>` |

Screenshots are in `results/m6/prototypes/<folder>/`. The decisions and the final design are in
`docs/m6/m6a_decision_record.md`.
