# Viewer replay

The lane debugger replays games so a user can watch them turn by turn. In Strategy matchup mode the user picks an
example matchup (or sets the configuration, start scenario, power gap, seed and both policies) and the app plays it.
In Recorded trajectory mode the app replays a saved recording. The user steps, scrubs or plays through turns, opens the
engine ticks within a turn, and reads the champions, wave budgets, timelines, event log and raw turn details in tabs.

## Sub-features

- `replay-matchup` selects an example matchup, which fills the setup fields.
- `replay-navigate` steps with Next, Back, End and Start.
- `replay-frames` shows the start, tick 1, tick 2 and end frames of a turn.
- `replay-play` auto-advances with Play and stops with Pause.
- `replay-seed` re-plays the matchup with another chance seed.
- `replay-tabs` shows the detail tabs, for example the legal actions in Turn details.
- `replay-recorded` replays a saved recording from `results/rl/m43b/trajectories/`.

## How to get to it (user POV)

- Run `streamlit run lanegame/viewer/app.py` and open the page. The sidebar title is `Lane debugger`.
- Choose `Strategy matchup` or `Recorded trajectory` in the sidebar's `Mode` radio.
- Use the playback row under the lane: `⏮ Start`, `◀ Back`, `▶ Play` / `⏸ Pause`, `Next ▶`, `End ⏭`, the turn slider
  and the speed selectbox, plus the `start / tick 1 / tick 2 / end` radio below them.
- Open the tabs at the bottom: Champions, Wave management, Timelines, Event log, Turn details, How to read this.

## Driving it with vl.py

Preconditions:

- `$VL viewer start`, `$VL browser start` and `$VL browser open` succeeded in this run, and `$VL doctor --run "$RUN"`
  exits 0.
- The page is fresh from `browser open`: Strategy matchup, `M4.5-1. Ward from a neutral lane`, seed 0.

- **Pick a matchup.** Choose `4A-6. Crash, then Recall`. Run `$VL browser select --run "$RUN" --key matchup --option "4A-6. Crash, then Recall"`,
  then `$VL browser widgets --run "$RUN" --label replay-matchup`. The widgets show `config` `m4_four_actions`,
  `scenario` `neutral`, `gap` `equal: equal power`, `pa` `grow_crash_recall4`, `pb` `farm4` and `turns` `40`. In
  `$VL browser text --run "$RUN" --scope main`, line 1 is `4A-6. Crash, then Recall` and line 2 is
  `Turn 0 of 40 · start · k = 1 · equal power · seed 0`.
- **Step forward and back.** Press Next three times, then Back once. Run `$VL browser click --run "$RUN" --text "Next ▶"`
  three times. The state line reads `Turn 3 of 40 · start · k = 4`. Then run `$VL browser click --run "$RUN" --text "◀ Back"`.
  The state line reads `Turn 2 of 40 · start · k = 3`.
- **Jump to the ends.** Run `$VL browser click --run "$RUN" --text "End ⏭"`. The state line reads
  `Turn 40 of 40 · final state · k = 41` and `Next ▶` is disabled. Run `$VL browser click --run "$RUN" --text "⏮ Start"`.
  The state line reads `Turn 0 of 40 · start · k = 1`.
- **Ticks within a turn.** Press Next once, then choose the tick frame. Run `$VL browser click --run "$RUN" --text "Next ▶"`
  and `$VL browser click --run "$RUN" --key frame --text "tick 1"`. The state line reads `Turn 1 of 40 · tick 1 · k = 2`.
  Run `$VL browser click --run "$RUN" --key frame --text "end"`. It reads `Turn 1 of 40 · end · k = 2`.
- **Play and pause.** Run `$VL browser click --run "$RUN" --text "▶ Play" --no-settle`, then
  `$VL browser wait --run "$RUN" --text "Turn 3 of 40" --no-settle --timeout 20`, then
  `$VL browser click --run "$RUN" --text "⏸ Pause" --no-settle`, then `$VL browser wait --run "$RUN" --text "▶ Play"`.
  The turn advanced on its own and then stopped. Two state-line reads a second apart are equal.
- **Change the seed.** Run `$VL browser fill --run "$RUN" --key seed --value 7`. The state line ends `seed 7`, and
  `browser widgets` shows `seed` `7`.
- **Turn details.** Run `$VL browser click --run "$RUN" --text "Turn details"`, then
  `$VL browser text --run "$RUN" --scope main --label replay-turn-details`. The text contains
  `Legal actions — A: LH-H, SH-H, TN-H, …` and a `B:` line.
- **Recorded trajectory.** Switch modes and rewind. Run `$VL browser click --run "$RUN" --key mode --text "Recorded trajectory"`.
  `browser widgets` shows `rec_folder` `results\rl\m43b\trajectories` and `rec_file` `attacker_vs_base_s0.json`, and
  line 1 reads `Recorded: the approximate best response (A) against the baseline final it was trained on (B)`. The
  turn carries over from the matchup, so run `$VL browser click --run "$RUN" --text "⏮ Start"`. Line 2 then reads
  `Turn 0 of 460 · start · k = 0 · equal power · seed 67000`.
- **Another recording.** Run `$VL browser select --run "$RUN" --key rec_file --option "base_s0_vs_crash_recall.json"`,
  then `$VL browser click --run "$RUN" --text "End ⏭"`. Line 1 reads
  `Recorded: baseline final (A) against the crash-and-Recall reference (B)` and line 2 reads
  `Turn 460 of 460 · final state · k = 460 · equal power · seed 67000`.
- **Proof.** Run `$VL browser screenshot --run "$RUN" --label replay-recorded-end`. The PNG shows the lane at turn 460
  and the sidebar on Recorded trajectory.

## Gotchas

- Selecting a matchup sets the configuration, scenario, gap, policies and turns, but **not** the seed. A seed typed
  earlier stays. Fill `seed` explicitly when a journey depends on it.
- The turn index survives mode switches and is clamped to the new game's length. After switching to Recorded
  trajectory and back, the matchup sits at its last turn and `Next ▶` is disabled. Press `⏮ Start` first.
- While playing, the page reruns on a timer and never settles. Use `--no-settle` on `click` and `wait` until Pause.
- The Event log and Wave management tables are canvas-drawn data grids, so their cells are not in `browser text`.
  Prove them with a screenshot (`--here` after scrolling to them, or `--full`).
- Recordings are read relative to the viewer's working directory. `vl.py` starts the viewer at the repo root, where
  the ten tracked recordings live.
- The selectbox for the power gap shows formatted labels (`equal: equal power`, `large: A x1.76 attack power (50 gold invested)`).
  Pass the label exactly as shown.
- Reference values were observed on 2026-10-18 at commit 82568827.
