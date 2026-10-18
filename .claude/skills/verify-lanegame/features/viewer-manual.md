# Viewer manual play

Manual play lets a user play side A by hand against a scripted side B. Each turn the user picks a wave action, a
champion behaviour and an objective from the legal ones and presses Play. Both actions resolve together, and the
lane, champions and events update. The user can undo the last turn, let a policy play A for several turns, go back and
branch from an earlier turn, or take over A from any turn of a Strategy matchup.

## Sub-features

- `manual-setup` enters Manual play and sets the configuration, scenario, gap, seed and B's policy.
- `manual-turn` plays one hand-picked action for A.
- `manual-undo` removes the last turn.
- `manual-autoplay` lets a named policy play A for N turns.
- `manual-branch` discards the turns after an earlier one and continues from there.
- `manual-takeover` switches from a Strategy matchup at turn i into Manual play, keeping turns 0 to i-1.

## How to get to it (user POV)

- Choose `Manual play` in the sidebar's `Mode` radio.
- Press `Take over A manually from turn <i>` under the playback row in Strategy matchup.
- Use the "Your move (A)" box: the `Wave management`, `Champion behavior` and `Objective` radios, the `Play: <action>`
  button, the `Let a policy play A for` selectbox with its `turns` input and `Auto-play`, and `Undo last`.
- When viewing an earlier turn, use `Branch here: discard turns i to n-1 and continue from turn i`.

## Driving it with vl.py

Preconditions:

- `$VL viewer start`, `$VL browser start` and `$VL browser open` succeeded in this run, and `$VL doctor --run "$RUN"`
  exits 0.

- **Enter Manual play.** Run `$VL browser click --run "$RUN" --key mode --text "Manual play"`. Line 1 of
  `$VL browser text --run "$RUN" --scope main` is `Manual play: you are A (blue)`, line 2 starts `Turn 0 of 0 · final state`,
  and the text contains `Your move (A), turn 0.`.
- **Set up the game.** Run `$VL browser select --run "$RUN" --key config --option m4_four_actions`,
  `$VL browser select --run "$RUN" --key scenario --option neutral`,
  `$VL browser select --run "$RUN" --key gap --option "equal: equal power"`,
  `$VL browser fill --run "$RUN" --key seed --value 7`
  and `$VL browser select --run "$RUN" --key pb --option farm4`. `browser widgets` shows exactly those values, and the
  `m_wave` options read `Last Hit (1 HP)`, `Thin (1 HP)`, `Slow Push (2 HP)` and `Shove (4 HP)`.
- **Play one turn.** Choose Shove and Trade, then Play. Run `$VL browser click --run "$RUN" --key m_wave --text "Shove (4 HP)"`,
  `$VL browser click --run "$RUN" --key m_beh --text "Trade"` and `$VL browser click --run "$RUN" --text "Play: Shove + Trade"`.
  The state line reads `Turn 1 of 1 · final state · k = 2 · equal power · seed 7`. The events text starts
  `A Shove: removed 0 of 4 HP; A's trade connected: dealt 0 cHP; A was ganked (p=0.11): 330 cHP;`. The gold advantage
  reads `+0.3 B`, and the Champions tab shows Champion A at HP `332/662`. Capture it with
  `$VL browser screenshot --run "$RUN" --label manual-turn`.
- **Undo.** Run `$VL browser click --run "$RUN" --text "Undo last"`. The state line reads `Turn 0 of 0`. Replay the
  turn with the three commands above before continuing.
- **Auto-play.** Let farm4 play A for four turns. Run `$VL browser select --run "$RUN" --key m_auto --option farm4`,
  `$VL browser fill --run "$RUN" --key m_auto_n --value 4` and `$VL browser click --run "$RUN" --text "Auto-play"`.
  The state line reads `Turn 5 of 5 · final state · k = 6`.
- **Branch.** Go back two turns and branch. Run `$VL browser click --run "$RUN" --text "◀ Back"` twice. The state line
  reads `Turn 3 of 5 · start · k = 4`, and the text contains `Viewing turn 3 of 5; actions are chosen at the latest turn.`
  Run `$VL browser click --run "$RUN" --text "Branch here" --contains`. The state line reads
  `Turn 3 of 3 · final state · k = 4`, and the text contains `Your move (A), turn 3.`.
- **Take over from a matchup.** Run `$VL browser click --run "$RUN" --key mode --text "Strategy matchup"`,
  `$VL browser select --run "$RUN" --key matchup --option "4A-6. Crash, then Recall"`,
  `$VL browser click --run "$RUN" --text "⏮ Start"` and then `$VL browser click --run "$RUN" --text "Next ▶"` four
  times, until the state line reads `Turn 4 of 40`. Then run
  `$VL browser click --run "$RUN" --text "Take over A manually from turn 4"`. Expected: `browser widgets` shows `mode`
  `Manual play`, line 1 is `Manual play: you are A (blue)`, line 2 starts `Turn 4 of 4`, and the text contains
  `Your move (A), turn 4.`.
- **Proof.** The `actions.log` entries and the text or widgets captures after each step, plus `manual-turn.png` and a
  final `$VL browser screenshot --run "$RUN" --label manual-end`.

## Gotchas

- **Take over fails as of 2026-10-18 (commit 82568827, Streamlit 1.65.0).** After the click the page stays in
  Strategy matchup at the same turn, and `mode` still reads `Strategy matchup`. It stays that way after a full rerun
  forced through the sidebar (changing `turns`), so the takeover is lost, not just delayed. A DOM `.click()` behaves
  the same, so the harness is not the cause. Report this journey as FAIL with the observed state until the viewer is
  fixed. Do not mark it verified by entering Manual play another way.
- Entering Manual play inherits the setup fields from whatever the session showed last (on a fresh page, the first
  matchup: `m45_unified`, seed 0). Set every field explicitly before asserting outcomes.
- Changing any setup field (configuration, scenario, gap, seed) clears the manual game back to turn 0.
- The wave radio labels include the HP budget, which depends on the lane, for example `Shove (4 HP)`. Use the exact
  label for the state you are in, or `--contains` with the action name.
- A `Take over A manually from turn <i>` text exists twice in the page (the button and its tooltip). The harness
  clicks the visible button, so leave `--nth` alone.
- Outcomes depend on the seed through gank and trade rolls (the `A was ganked (p=0.11)` above is a seed-7 roll). The
  same setup and actions reproduce exactly; a different seed may not.
- Reference values were observed on 2026-10-18 at commit 82568827.
