# M6c: sequences within Watch (step 1, the format and the first sequence)

*2026-10-22. Status: **step 1 approved by the user on 2026-10-22, after four review rounds (§8); further sequences
are on hold at the user's request.** M6c was
narrowed on 2026-10-22 to a small
collection of optional guided replays within Watch (§1), which visitors see as **Sequences** since the fourth review
(§8): each sequence is a chain of events in one recorded game. The code and the file format keep the word guide. This document is the format's reference, the authoring
guide and the record of the first example. Nothing here changes a mechanic, a reward, a training objective, a
trained network or a stored research result. No training ran. Nothing is committed: the user reviews the work in
the working tree, as with M6b.*

- *Plan:* [m6_plan.md](m6_plan.md) §6 M6c. *Application:* [M6b report](m6b_report.md). *Core:*
  [M6a technical report](m6a_technical_report.md).
- *Launch:* `pip install -e ".[demo]"`, `cd web && npm install && npm run build`, then `python -m lanegame.demo`
  (http://127.0.0.1:8765/). Switch to **Watch** and pick "Stacking to Crashing" under **Sequences** in the game selector, or
  open http://127.0.0.1:8765/?mode=watch&sequence=stacking-to-crashing.
- *Evidence:* `results/m6/m6c_browser.json` and the screenshots in `results/m6/m6c/`.
- *Tests:* `tests/test_demo_guides.py`, the additions to `web/src/game/store.test.ts`, and the existing suites.

## 1. The revised scope

The user narrowed M6c on 2026-10-22. The application stays a simple research demonstration with Play and Watch.
The detailed research explanation belongs in the documentation.

M6c is now a small collection of optional guided replays within Watch, built in three steps:

1. the guide format and one complete example, prepared for review (this document);
2. the user's review and refinement of the chosen sequence, the start position, the event positions, the
   wording, the highlights and the note panel's placement (a first round is recorded in §8);
3. a small number of further examples in the accepted format.

Dropped: the separate findings page, extensive in-app research explanation, and a prominent gallery of all 15
research recordings. Retained: the Play and Watch experience as M6b built it, the label that tells a recorded
research game from a generated simulation, deterministic verification of every recording a guide uses, and a simple
link to the research documentation. The research question, the method, the aggregate results, the comparisons and
the limitations move to M6d's research overview. The 15 recordings stay under `results/m5/traces/` as research
artifacts; none was deleted.

A sequence is a curated recorded demonstration. It does not coach a live game.

## 2. The first example and why it was chosen

**The sequence.** `stacking-to-crashing`, "Stacking to Crashing" (first titled "Stack the wave, crash it, take a
plate"): turns 75 to 84 of the recorded game
`sp_s0_vs_g_shove`, with the self-play agent (`sp:0`) in seat A and the always-push bot (`g_shove`, a fixed rule
with no learning) in seat B. The recording is one of the three the server re-simulates and checks at every start
(M6b), selected by M5's fixed upper-median rule, not by this guide; its provenance (file, SHA-256, the selection
rule) travels with the replay and is shown under **Details**. The original file is untouched.

**What it shows.** Five events that form a chain, each one the cause of the next:

| event | position | what the recording shows | the link to the next event | highlight |
|---|---|---|---|---|
| Trades keep the bot from going back to base | 77 | the bot, at 274 of 1026 HP, chose Recall in turns 75 and 77; both times the agent's Trade reached it, which interrupts a Recall, and an interrupted champion does nothing to the waves; the turn-77 hit left the bot at 105 HP and cost the agent 125 HP under the bot's tower | the bot's champion is not clearing the agent's wave | the bot |
| Nobody clears the wave, so it stacks | 80 | turns 78 to 80: the bot played Last hit + Safe or tried to Recall again, and its champion removed none of the agent's minions; the agent chose Slow push twice, then Thin + Trade, taking 1 HP of the bot's minions a turn; its wave, grown to 22 HP as new waves joined, beat two small bot waves and reached the lane with 20 HP and nothing in front of it | nothing stands between the wave and the bot's tower | the agent's wave |
| The stacked wave crashes | 81 | the 20 HP wave crashed into the bot's tower | Pressure is allowed only while a champion's own minions are crashed under the enemy tower | the crashed minions |
| Pressure on the crash breaks a plate | 82 | the agent chose Thin + Hold + Pressure and damaged the tower by 1602, past the first plate's threshold of 1600: +1.00 gold; the bot's Recall went through; the agent earned 1.34 gold in the turn, the bot nothing; the lead went from 1.64 to 2.98 | the bot is away, the crash is still there | the bot's tower |
| A second plate while the bot is away | 84 | the agent kept up Pressure (1402, then 1302 tower damage) and passed the second plate's threshold of 3600: +1.00 gold; since turn 75 the lead moved from the bot's 1.03 to the agent's 3.63 | the end of the chain | the bot's tower |

**How it was chosen.** A script listed every crash of 9 HP or more in the three verified recordings with the
pushing side's actions and front-wave HP over the six turns before and every gain on either side over the six
turns after. Two crashes match the chain the user described at the first review ("the agent stacks the wave by
slow pushing while trading, then the wave crashes, and the agent gets resources"), both against the always-push
bot and both ending in plate gold for the side that stacked the wave: this one, and the adaptive agent's crash in
turn 54 of `om_s0_vs_g_shove`. This one was taken because the bot is present and visibly unable to leave, so every
link is on screen, while in the other the bot is away for most of the sequence.

**Why every link holds.** Each "because" in the notes is a rule of the engine, not a reading of anyone's mind.
A Recall is interrupted when an enemy Trade ends in contact with the recaller, and the interrupted champion
"stays and channels, with no gold and no wave damage" (reference §6.1, step 2b). The bot's low-HP behaviour is its
fixed rule: at 40% HP or less it Recalls when out of contact and otherwise walks back with Safe
(`experiments/m45b/strategies.py`); stating a scripted rule is stating code. Pressure needs crashed minions under
the enemy tower (the action menu's legality rule), and plates break at 1600, 3600 and 6000 tower damage (the
preset). The agent's own reasons are never stated: the notes say what it chose and what the engine did.
`tests/test_demo_guides.py` pins every number the notes quote to the recording, and checks that each appears in its
note as the browser prints it.

**The first draft and why it was replaced.** The first version of this guide, reviewed on 2026-10-22, walked turns
12 to 25 of the seed-0 head to head (`h2h_om_s0_vs_sp_s0`): the adaptive agent's wave stacked and crashed into the
self-play agent's tower, and the self-play agent later recalled. The user found that it lacked clear guidance. The
chain was muddled: the side that stacked the wave gained nothing from the crash (the defender earned the 0.90 gold
under its tower), and the recall followed a dive and a gank that the notes could only name. The user asked for a
chain where each event plainly leads to the next and ends in a gain for the agent that caused it.

## 3. The guide format (reference)

A guide is one JSON file, `lanegame/demo/guides/<id>.json`, loaded by `lanegame/demo/guides.py`. The first guide is
the template. The service validates every file at startup against the verified replay it refers to and serves the
result at `GET /api/guides` (the index) and `GET /api/guides/{id}`.

| field | type | meaning |
|---|---|---|
| `schema`, `version` | `"lanegame.demo_guide"`, `1` | the format |
| `id` | string, `^[a-z0-9][a-z0-9-]{1,63}$` | the stable identifier; equals the file name without `.json` |
| `title` | string, at most 80 characters | the plain title shown in the game selector and the panel |
| `intro` | string, at most 600 characters | what to watch for; shown at the start position |
| `replay` | one of the server's research replays (`service.RESEARCH_REPLAYS`) | the recording the guide annotates |
| `start` | integer | the position the guide opens at, paused; at most the first checkpoint's position |
| `end` | integer, optional | the last position the guide covers, at least the last checkpoint's; the panel names the covered turns |
| `checkpoints` | list of 1 to 12 | the ordered events (the field keeps its first name) |

Each checkpoint has:

| field | type | meaning |
|---|---|---|
| `id` | string, same pattern as the guide id, unique within the guide | the stable identifier of the event |
| `position` | integer | the position to show; positions increase strictly from one checkpoint to the next |
| `title` | string, at most 60 characters | the event's title; the panel also shows it as the previous event's "Next:" link |
| `text` | string, at most 400 characters | the note; short enough for the panel |
| `highlight` | optional object `{kind, seat}` | a game object to mark on the lane |

Position t is the state after t turns, and position 0 is the initial position (the application's convention). A note
that asks the viewer to anticipate a later event belongs at the earlier position.

Highlight kinds and what they require:

| kind | seat | what is marked | required at that position |
|---|---|---|---|
| `champion` | `A` or `B` | the champion's token | the champion is present (not recalling or dead) |
| `tower` | `A` or `B` | that seat's tower | always |
| `wave` | `A` or `B` | that seat's leading wave | the seat has a front wave on the lane |
| `crashed` | `A` or `B` | that seat's minions crashed under the enemy tower | crashed HP above zero |
| `ward_spot` | none | the bush beside the lane | always |

The browser handles the absence of an entity at a frame between positions (a champion that walks away during a
stepped turn) by drawing no ring; the server refuses a guide whose highlight points at nothing at its own position.

Rules the loader enforces, each with a message naming the file and the rule: the schema and version; the exact key
set (an unknown key is refused, so typos never pass); the id pattern and the file name; the length limits; the
replay name; the start, end and positions; non-empty, strictly increasing, unique checkpoints; the highlight's shape.
The service adds the checks that need the replay: every position is at most the recording's length, and every
highlight's entity exists at its position. A guide that breaks a rule stops the server with that message. A guide
whose recording is not on this machine is skipped, and `GET /api/guides/{id}` answers 404 with a message that names
the recording.

## 4. How to add or edit a guide

1. Pick the recording and the sequence. Dump the replay to read it: the bundle `DemoService().replay(name)` has
   `positions` (0 to n) and `turns` (1 to n), each turn with its outcomes and ticks. Choose a start that gives
   enough context and three to five positions that carry the sequence.
2. Copy `lanegame/demo/guides/stacking-to-crashing.json` to `lanegame/demo/guides/<new-id>.json`, set `id` to the file
   name, and write the title, the introduction, `replay`, `start`, `end` and the checkpoints. Quote only values the
   replay holds (HP, gold, units, wave HP, outcomes). Keep each note under 400 characters. One trap: the damage a
   trade, tower shot, gank or minion-fire outcome names (`data.damage`) is the amount dealt before the victim's
   durability; the HP the victim lost is the turn's `delta.hp_taken`. The two agree only while the victim has
   invested nothing. The first guide quotes HP lost and tower damage only. After an investment, quote the HP lost, or the
   HP bar will contradict the note. The repository audit of 2026-10-22 lists this wording as a defect to fix in
   `frames.py`.
3. Build a chain: each event's note says what was chosen and what the engine did, and its last sentence names what
   that makes possible next. Ground every "because" in a rule of the engine or a scripted bot's fixed rule. Do not
   call an action optimal, do not state what a trained agent intended, and do not offer one game as evidence for an
   aggregate finding (plan §9). Quote a lead as the browser prints it (`toFixed(2)`, which rounds an exact tie up).
4. Run `python -m pytest tests/test_demo_guides.py`. The format tests run without the engine; the bundle checks need
   the 7 checkpoints of `lanegame/demo/manifest.json`. Add a fact test for the new guide, in the style of the first
   guide's, so the numbers stay pinned to the recording.
5. Start the server (`python -m lanegame.demo`) and open `http://127.0.0.1:8765/?mode=watch&sequence=<new-id>`. A guide
   that breaks a rule stops the server with the rule's message. Read every note at its position.
6. Optional: run `python -m lanegame.experiments.m6c browser` for the scripted check and the screenshots.

No frontend change is needed for a new guide: the game selector lists every guide the server serves.

## 5. The Watch integration

What the viewer sees. In Watch, the game selector in the top bar gains a group **Sequences**. Choosing one loads
its verified recording (labelled "Recorded research game" as before), shows position `start` paused, and opens a
compact panel at the top of the dock's action area with the guide's title, its introduction and **Next event**.
**Next event** plays the turns that lead to the next event, one position at a time, and stops there; the panel then
shows "Event k of n", the event's title, "after turn p", the note, **Previous event**, **Next event** and, after the
buttons, "Next:" with the following event's title, the chain's forward link. A ring on the lane marks the event's
object while the viewed position is the event's. The playback row, the speed control, the turn history, the turn
buttons, the keys, the gold graph and Details work as before.

Decisions, each reversible at the review:

- **Next event and Play walk the chain.** In a guide, **Next event**, the **Play** button and the Space key all play
  forward to the next event and pause there, so a viewer watches the turns between a cause and its effect and then
  reads the note. The walk shows each resolved position for 0.75 s at 1x (0.19 s at 4x) without the tick frames,
  long enough for the lane's movement to finish. During it the panel shows the next event's title and "Playing to
  turn p: now at turn q" instead of a note, so no note is shown before the replay reaches the position it
  describes. Past the last event, Play walks on to the end of the recording the same way. **Previous event** jumps
  back without a walk. "Show the ticks" still steps any single turn frame by frame.
- **Movement arrows.** For the turn that produced the position on screen, the lane shows how each champion and
  each side's leading wave moved. Each arrow is a short trail, about a third of a unit long, that ends just behind
  the thing that moved and points the way it went: behind a champion's token for a walk, a dashed one for a Recall
  or a death (behind the token at its base) and for a return (behind the token back at its tower); behind the
  leading wave's cluster for a walk, and behind the crashed minions for a crash. Every kind but a walk carries a
  short label, and the legend explains the arrows. The server states these moves (`frames.moves`, the turn frame's `moves`) from the
  engine's states, so the browser draws facts and infers nothing. A wave arrow follows the leading wave the turn
  started with. The engine guarantees that only when the side lost no packet (packets die or crash only as the
  leading one), so a cleared wave gets no arrow rather than a wrong one. The arrows show in Play, Watch and the
  guides, on resolved positions and at the end of a stepped turn, and not during the ticks, which animate the
  motion themselves.
- **Placement.** The panel sits inside the dock, above the playback row, so the approved one-surface layout holds.
  The dock grows while notes are shown (from 194 px to 272 px at desktop widths, 300 px below 1150 px, where the
  narrower column wraps the introduction to seven lines), and the lane is fitted again around it, as it is when the
  history or the large graph opens. Nothing floats over the lane. While notes are shown the turn card keeps its
  actions line and drops its outcome list; the outcomes stay in the history rows and in Details. On a phone the
  panel's "after turn p" is left out, because the turn label just above already says it, and its text scrolls
  within 84 px.
- **Away from an event.** If the viewer moves to another position (a turn button, a history row, the graph or the
  keys), the note stays with its own label ("This note describes the position after turn 80") and a **Return to this
  event** control. The note is never presented as describing the current position. At the introduction that line
  takes the place of the "Covers turns" line and the documentation link, so the row stays one line in the fixed dock.
- **Hidden notes.** **Hide notes** collapses the panel to one line with the guide's title, **Show notes** and
  **Close sequence**; the dock returns to its usual height, the ring disappears, and the viewer watches normally.
- **Closing a guide** keeps the recording on screen. Choosing another recording, generating a match or starting a
  game clears the guide.
- **A live game is never stopped.** Opening a guide from Play switches the screen to Watch, as the Watch button
  always did; the game stays on the server and in the browser's memory. The **Play** mode button now returns to that
  game (M6b opened the New game dialog instead), and offers a new game only when none is remembered or the server
  no longer holds it.
- **Words.** The guide's steps are called events. Positions along the lane are units in everything a visitor reads:
  the notes, the menu's help and reasons, Details and the lane's text for screen readers. The API's field names
  (`cell`, `cells`, `checkpoints`) keep their names.
- **Research documentation.** The guide panel's introduction and the About dialog link to the repository's
  documentation index.
- **Not in live games.** The guide state belongs to Watch; a live game or a generated match never shows a note.

## 6. Verification

Every number here comes from a run on 2026-10-22 on the working tree of the revised step.

| check | result |
|---|---|
| `tests/test_demo_guides.py` (5 tests: the format's rules with 14 refused variants, the bundle rules against the real replays, the API and the status count, and every number the notes quote read from the recording and found in its note as the browser prints it) | 5 passed |
| `tests/test_demo_api.py` (16), `tests/test_demo_session.py` (29) and `tests/test_demo_moves.py` (3: the arrows' rule on hand-made frames without the engine, every turn of the three replays, and the guide's turns) with the guides' tests in one run | 53 passed. One API test is chance-dependent and fails about one run in three (the repository audit's D7); it passed in this run and predates this work |
| `npm run typecheck`, `npm test` (30: the 10 of M6b, 11 for the guides, the walk and the reduced-motion pace, 5 for which arrows a frame shows and their words, 4 for an arrow's place behind a token or a cluster), `npm run build` | clean, 30 passed, built. The new tests were checked against the old code: the pace test fails at the old 0.34 s turn, the Next-event test fails when Next jumps, and the end-of-recording test fails when Play does nothing there |
| `python -m lanegame.experiments.m6c browser` (`results/m6/m6c_browser.json`, 21 screenshots in `results/m6/m6c/`), with animation on | 81 of 81 checks. The address opens the guide paused at position 74 with the introduction. Next event walks 74, 75, 76, 77; 77, 78, 79, 80; 80, 81; 81, 82; 82, 83, 84, one position at a time, with the note panel on screen at every sample (sampled in the page every 150 ms) and the walk line in place of the note until arrival; at each event the title, the ring and "Next:" with the following title appear, and the screen shows the values the notes quote (the bot's 105 HP; "+2.98" after turn 82; "+3.63" and two plates lost after turn 84). Previous event jumps back. A turn button off an event keeps the note with its label, and "Return to this event" restores it. Play walks from event 1 to event 2 and stops, then to event 3; at the last event it walks on toward the end with the guide open. Hide notes, Show notes and Close sequence behave as before. A live game carries no guide, a guide opened over it leaves it on the server, and the Play mode button returns to it. A generated match carries no guide. An unknown guide shows the server's message. No panel covers a champion, wave, tower or the ward spot and the note panel fits at 1440x900, 1024x700 and 390x844 in both themes. With reduced motion, plain Watch shows one turn in the first two seconds at 1x. No page errors. Arrows: after turn 81 exactly three (the agent's wave crashing from the lane into the bot's tower with its label, the agent stepping back from the tower, the bot's new wave walking out of its base); after 82 the bot's dashed Recall arrow to its base; after 83 its return; after 74 the agent's walk into the bush; none at the start of the game; none on a turn's start frame and the three of turn 81 on its end frame; the screen-reader sentences; a live Recall drawn from the tower to the base; the guide's crash event with its ring and arrow; every arrow label clear of the drawing and the panels; the legend's second line clear of the top bar and the dock at three sizes in both themes; and, for every arrow inspected, a length between 20 and 40.5 px and a tip just behind the right token (outside its drawn shape, within 6 px of it) or within 6 px of the right wave or crashed cluster, outside it |
| "Play from the start" at the end of the recording, in the browser with and without reduced motion | returns to the introduction at position 74 and walks to event 1 (77), with no page errors |
| the M6b journey (`python -m lanegame.experiments.m6b browser`, output redirected so the committed evidence stays) | 71 of 71 checks, as in M6b: plain Watch playback with animation, the live game, history, reload and every other part are unchanged |

The screenshots were read: `12_arrows_crash`, `13_arrows_recall`, `14_arrows_return` and `15_live_recall` (the arrows point the right way, sit under the balls, chips and tokens, and their labels are legible), `11_phone_portrait_light` (crowded at that size, still legible), `02_event1` (the ring on the bot at 105 HP and "Next: Nobody clears the wave, so it
stacks"), `07_walking` (position 78 on the way to event 2, "Playing to turn 80: now at turn 78." in place of the
note), `04_event3` (the ring around the 20 HP crash under the bot's tower) and `05_event4` (the ring on the bot's
tower with its first plate gone and the lead at +2.98).

## 7. Limitations and what the review decides

- One example exists. The format carries what it needs for a few more; it is not a tutorial engine, and there is no
  authoring interface beyond editing a JSON file and running the tests.
- The notes quote the engine's numbers at their positions and say nothing about the trained agent's reasons; the
  bot's reasons are its fixed rule. Whether that level of detail reads well to a newcomer is the review's question.
- With animation on, a turn button (one position forward) still shows its turn frame by frame, and the step panel
  takes the note's place until the turn's frames are done; only Next event and Play walk without the frames.
- On a phone held upright the turn card's line of both actions is cut off at the right ("Push bot Las…"). It
  predates this step and does not hide the note.
- The highlight marks one object per event. An event that needs two marks would need a format change.
- "Show the ticks" uses the tick narration the repository audit found wrong in two cases (a fight between
  two units narrated as "at a base", a cleared wave narrated as walking; audit D2 and D3). The guide's walk does
  not show that narration, but a viewer who steps a turn of this guide's range can meet it. The fix is item 1 of
  the audit's remediation plan.
- The research documentation link points at the repository on GitHub; the application does not serve the documents.
- The research overview (the question, the method, the six-seed results, the comparisons, the limitations) is M6d
  work and does not exist yet.

The review decides the chosen chain, the start position, the event positions, the wording, the highlights, the
walk's pace and the note panel's placement. After that refinement, a few more examples follow in the accepted format.

## 8. The reviews (2026-10-22) and what changed

In the first round, the user tried the first version and reported four things.

1. **"Clicking Play breaks the demo: it goes all the way to turn 150 really quick and breaks the UI."** Reproduced in
   headless Edge. In a guide, Play was ordinary playback: it ran past every event to turn 150 while the note on
   screen still described the guide's start, and with animation on the tick step panel replaced the note panel on
   every turn and the note flashed back between turns. Under the system's reduced-motion setting it was worse: a
   turn without tick frames lasted 0.34 s at 1x, fourteen times faster than an animated turn at the same speed
   setting, so 150 turns went by in under a minute (13 s at 4x). Fixed at the cause: in a guide, Play walks to the
   next event and stops (§5), and in plain Watch a turn under reduced motion now lasts as long as its animated
   frames would (4.4 s at 1x), so the speed setting means the same in both modes.
2. **"Replace cell with unit in the explanations."** Done in every visitor-facing text (§5, Words).
3. **"Call them events instead of moments."** Done: "Event k of n", **Previous event**, **Next event**, **Return
   to this event**.
4. **"There still lacks a clear form of guidance; set up a butterfly effect: the agent stacks the wave by slow
   pushing while trading, the next event shows the wave crash and the agent get resources."** The example was
   replaced by a chain of that shape (§2), each note now ends by naming what it makes possible, the panel shows the
   next event's title as a forward link, and Next event plays the turns between the two events.

In the second round, the user found the simulator's layout still hard to read and asked for an arrow showing
which way each champion and wave moved: from base to tower when a champion walks back to lane, from tower to base
when it recalls. The lane now draws those arrows for every resolved turn (§5, Movement arrows). Their facts come
from the server and are tested against every turn of the three verified replays (`tests/test_demo_moves.py`):
waves only walk forward and at most three units, every crash arrow sits on a recorded crash, and every recall,
return and death arrow matches the recorded states. A test with hand-made frames checks the rule without the
engine, so it runs in a fresh clone.

In the third round, the user asked for shorter arrows placed directly behind each entity, "for example directly
behind the A circle", instead of lines from the start place to the end place. Each arrow is now a short trail
touching the back of the token or cluster that moved (§5, Movement arrows). The browser check reads every
arrow's tip and tail and confirms the length and that the tip sits against the right token or cluster.

In the fourth round, the user branded the feature for visitors: "Sequences" instead of guided replays, since each
is a sequence in a game made of events, and titled this one "Stacking to Crashing". Everything a visitor reads
says sequence: the Watch menu's group, the panel's kicker, **Close sequence**, the loading text and the server's
messages. The address takes `?sequence=<id>`, and this sequence's id is `stacking-to-crashing`. The code keeps
the word guide (`guides.py`, `/api/guides`, the folder `lanegame/demo/guides/`, the JSON schema
`lanegame.demo_guide`), so a reader of the code meets one internal name; renaming it would change nothing a
visitor sees.
