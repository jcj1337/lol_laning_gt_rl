# M4.2e-D — Visual simulation debugger

A temporary local tool for watching the M4 simulator (`GameConfig.m4_lasthit()` by default) turn by turn before the
M4.3 PPO smoke test. It runs the real engine (`LaneEnv` and `champions_m4.resolve_turn_m4`) and only draws what the
engine reports. It is a debugging aid, not a product.

## Install and launch

```
pip install -e .[viewer]            # or: pip install streamlit plotly
streamlit run lanegame/viewer/app.py
```

Streamlit opens `http://localhost:8501` in the browser.

## Modes

- **Strategy matchup.** Pick an example matchup or set the fields yourself: configuration, start scenario, power gap,
  seed, a policy for each side and the number of turns. The whole game is simulated at once. You then step through
  it.
- **Manual play.** You control A (blue) and B follows the selected policy. Choose a wave action (Hold / Thin /
  Clear), a behavior (Hold / Yield / Trade / Safe, or Recall when it is legal) and an objective (Pressure / Vision,
  only offered when legal). The Play button appears only when the composed code is in the engine's legal set. B
  decides from the same pre-turn state and the turn resolves both actions together.
  - "Let a policy play A" advances several turns with a policy.
  - "Undo last" removes the last turn.
  - Viewing an earlier turn offers "Branch here". It discards the later turns and continues from that turn with
    the same seed.
- **Take over.** "Take over A manually from turn i" (in the matchup sidebar) copies turns 0..i-1 of the current
  matchup into manual mode, with B keeping its policy.

## Screen

The page is organized around the lane picture, top to bottom:

1. **Matchup and turn.** The matchup (or "Manual play"), the turn, the frame within it, the engine's turn index k,
   the power gap and the seed.
2. **Players.** For each side: its policy, the action it chose this turn in words ("Thin + Hold + Pressure"), HP,
   level, attack power, gold earned and spendable, and the policy rule that chose the action (or the fallback it
   played).
3. **The lane.** Global coordinates: A's tower at −2, the lane's midpoint at 0, B's tower at +2, the bases at ±4.
   B's own coordinates are mirrored for display.
   - A (blue) is drawn on the upper row of the lane bar, B (red) on the lower row. A's minions walk right, B's left.
   - A bold rectangle is a side's leading packet; its label is the packet's minion HP (one fresh wave = 8 HP, so a
     label of 12 means one and a half waves, not twelve minions). Its width grows with HP.
   - Light rectangles are the reinforcements behind the front, always one fresh wave each ("2×8" = two in one cell).
   - A dark "crash N" rectangle at a tower is that side's crashed mass under the enemy tower.
   - Circles are the champions (faded at their own end of the lane while away). The dashed line marks fronts in
     contact (they fight). A short arrow shows the leading packet's movement since the previous frame; it is not
     drawn when that side lost a packet (a death or a crash) in between, since the "new front" is another packet.
   - Each tower shows the damage it has taken and its plates. Hovering any object gives its details.
4. **Playback.** Start / Back / Play / Next / End, a turn slider, a speed selector, and a frame selector within the
   turn: "start" (the state both players chose from), each authoritative engine tick, and "end" (after the spawn
   and merge at the end of the turn). "Play tick by tick" makes Play step through the ticks as well.
5. **Summary.** The gold advantage, one line on the waves (fighting where, separated, crashed under which tower) and
   the events of the turn just shown (merges, deaths, crashes, bounces, trades, ganks, plates, levels, recalls).
6. **Details** (tabs): the full champion panels (income / spendable / invested gold kept apart, wave budgets),
   timelines (income and A − B, HP, level, contact position), the event log, the turn's `info` grouped by topic
   with the integer ledger and the raw tick trace, and a "How to read this" explanation for newcomers.

## Policies, matchups, scenarios

- **Policies.** The policies are the reactive M4.2e-B/C policies of `experiments/m4/zoning_natural.py`, unchanged:
  farm, push, punish, respect, freeze_punish and the others. The viewer adds a few compositions of their existing
  rules (clear, build_clear, freeze, safe) and one crash-then-Recall policy. Each policy decides from the current
  state only.
- **Example matchups.** The sidebar offers seven:
  1. Passive farming
  2. Slow push
  3. Wave clear
  4. Freeze vs. breaker
  5. Trading vs. respect
  6. Crash and Recall
  7. Power advantage

  Their descriptions say what each policy attempts, not what will happen.
- **Start scenarios.**
  - `s0` is the standard initial state.
  - `neutral`, `pushed`, `siege` and `freeze` are the deterministic start states of M4.2e-B/C
    (`zoning_natural.starts`): played on the resolver from s0, then normalized as in those experiments (both
    champions at the larger XP, full HP, no gold, no investment).
  - The power gap is `zoning_natural.with_gap`: A's investment and/or level lead.

## How fidelity is kept

- **No stored live simulation.** A recording is a pure function of (setup, seed, actions). Every interaction replays
  it from the start, which takes well under a millisecond per turn, so stepping back, scrubbing and branching cannot
  consume randomness. The chance generator draws the same uniforms per turn whatever the actions, so a branch sees
  the same later chance draws.
- **Tick traces.** These come from one opt-in flag, `LaneEnv(record_ticks=True)`. It passes `record=True` to
  `mechanics.step_turn` (which changes no result) and stores the trace of the resolver's authoritative lane step,
  not the positioning probe, as `info["lane_trace"]`. The flag is off by default.
- **No re-derived values.** Every panel value comes from `info`, the states, or the progression tables (`context`,
  `plates_broken`).
- **Tests.** `tests/test_viewer.py` (23 tests) checks that:
  - every matchup's recording equals a plain uninstrumented `LaneEnv` fed the same actions;
  - replay is deterministic and branches preserve the prefix and the chance draws;
  - instrumentation changes nothing;
  - the trace equals a re-run of `step_turn` with the authoritative arguments (on 110 of 300 sampled turns the
    probe's trace differs, so the test can tell them apart) and agrees with the end state;
  - B is mirrored correctly;
  - the summaries match the state and the income identity holds (income = spendable + invested + clamped);
  - actions are legal and an illegal manual action is rejected;
  - the policies equal `zoning_natural.Policy.act` and are state-dependent;
  - every frame's packets, crashed masses, contact and arrows follow the states or the authoritative tick trace;
  - the figure builds for every frame, with one hover point per drawn object and champions at their global cell.

## Limitations

- During the tick frames the champions are drawn at their end-of-turn cells: the engine moves champions once per
  turn (before the lane ticks), not per tick.
- Spawns and merges happen after the ticks, so they first appear in the "end" frame.
- Manual mode controls A only. Policies have no memory.
- Tick traces exist only for ruleset `m4`. The viewer supports the natural-wave presets `m4_lasthit`, `m4_lane` and
  `m4_natural`.
- Play reruns the page on a timer (Streamlit fragment), so playback speed is bounded by the page rerun (about
  0.3 s per step on a laptop); "fast" may not keep up on long games.
- There is no state editor. Scenarios are the reproducible constructions listed above.
