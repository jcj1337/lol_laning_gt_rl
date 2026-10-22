# M6a demo and prototypes (throwaway)

`hud/` is the M6a demo interface: a top-down lane that fills the window, with the controls in light panels over it.
It grew from `refined/` (prototype C's look with prototype A's action controls, as the user chose on 2026-10-21),
which stays as the previous layout. **Play in it is live**: your action goes to the tested session core and resolves. The earlier directions
**A · Arena** (`arena/`), **B · Board** (`board/`) and **C · Story** (`story/`) stay for reference. Their Play screens
are staged walkthroughs that replay a recording. None of this is the production frontend (`web/`, M6b): no build step,
no npm.

## Launch

From the repository root:

```bash
.venv/Scripts/python.exe prototypes/m6a/serve.py
```

It loads the engine (about 3 s), then prints `M6a demo: http://127.0.0.1:8765/`. Open that address. The server
listens on this machine only; stop it with Ctrl+C. The engine needs the 7 git-ignored checkpoint files of
`lanegame/demo/manifest.json`. Without them, or with `--no-engine`, the pages still load, Watch shows the recorded
games, and Play says that the engine is unavailable.

## What to try in the refined interface

- **Play.** Choose the opponent and whether its belief is shown, then play. Pick a wave action and a behaviour (and
  Pressure when it is available), or Recall or Ward, and submit. Unavailable choices show the engine's reason.
- **History.** Click a row or use ← → Home End to look at earlier turns. Submission is disabled while you do;
  **Return to current turn** (or Esc) brings back the decision.
- **Gold lead.** Blue above the line when the left player (you) has earned more gold, red below when the right player
  has. Hover to read a turn's lead, and click to jump there. "Show events" adds the event markers and the research
  checkpoints.
- **Watch.** A recorded research game, or a new game between two agents ("New game between agents"). Autoplay,
  pause (Space) and speed (0.5× to 4×).
- **Details.** The full event log, wave numbers and ticks, champion numbers, the belief when it may be shown, and the
  game record.
- **Minions.** Each ball is 1 HP of minions (the engine tracks each wave's HP, not single minions; a new wave is 8).
  Add `?minions=blocks` to the address to compare with the earlier HP blocks.

## Limits

- **Every game is an example.** Your game and a new simulation are not evidence for M5's findings. The three seed-0
  recordings are verified M5 games (re-simulated exactly by the session core); `sim_om_vs_sp` is a new simulation made
  for the prototypes (chance seed 110,000, policy seed 110,500).
- **The recordings' data files hold whole games**, which suits Watch and review. A running game only ever receives
  resolved turns from the server.
- **Local only.** Sessions live in the server's memory and end when it stops; there is no record download yet.

## Files

| path | content |
|---|---|
| `index.html`, `serve.py` | the launcher, and the local server with the JSON API over `lanegame/demo/service.py` |
| `hud/` | the demo interface (game view with overlaid panels) |
| `refined/` | the previous layout (stacked sections) |
| `shared/core.js` | data loading, the history controller (progress kept apart from the viewed position), playback, keys |
| `arena/`, `board/`, `story/` | the three earlier directions (staged walkthroughs) |
| `data/` | the recorded games in the demo's frame contract, written by `python -m lanegame.experiments.m6a export` |
| `tools/shoot.py` | screenshots in headless Edge: `.venv/Scripts/python.exe prototypes/m6a/tools/shoot.py prototypes/m6a/<folder>/shots.json results/m6/prototypes/<folder>` |

Screenshots are in `results/m6/prototypes/<folder>/`. The decisions are in `docs/m6/m6a_decision_record.md`.
