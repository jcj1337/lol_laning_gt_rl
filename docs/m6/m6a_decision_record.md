# M6a decision record

*2026-10-21. Status: **pending the user's visual review of the refined interface.** The backend is complete
([technical report](m6a_technical_report.md)). The user chose a direction after trying the three prototypes (§6),
and one refined interface with live Play now awaits review. M6a is declared complete only after that review, with the
final decisions recorded here.*

The record separates what the user decided, what I propose and what remains open. Plan: [m6_plan.md](m6_plan.md).

## 1. Approved requirements (the user's M6a brief, 2026-10-21)

- **Scope.** M6 presents the existing research. It changes no mechanic, reward, training objective, trained network or
  historical result, and needs no training.
- **Two primary modes.** *Play*: a human chooses discrete actions against `sp:0` or `om:0`. *Watch*: two bots play a
  bounded simulated match, generated first and then played back through a timeline. Watch is in the initial release.
- **Bots.** The two seed-0 M5 agents, for either bot seat, by M5's seed-selection convention. Scripted strategies are
  a design decision (§4).
- **Shared infrastructure.** Play and Watch share the lane view, the turn history, the event presentation and the
  recordings.
- **Stack and location.** React with TypeScript and Vite in `web/` in this repository, FastAPI with Python
  authoritative for every rule, unless a concrete blocker appears. M6a found none.
- **Docs tracked.** The blanket `docs/` ignore rule is removed (done).
- **Turn history** (plan §5.1). Chess.com's interaction pattern without its branding. One row per simultaneous turn
  with both actions; turn t shows the state after turn t, and turn 0 the initial position; first, previous, next,
  latest and row clicks; autoplay, pause and speed in Watch and in review. Browsing during play changes nothing,
  disables submission and offers a prominent "Return to current turn". Everything on screen follows the selected turn.
  No undo, branching, grading or win probability.
- **Hidden information.** No unresolved opponent action, future state or hidden belief in any history endpoint or
  client data. The belief is hidden by default, shown only if chosen at the start, and available in full after the
  session stops.
- **Clarity** (plan §5.2). The lane first, then essential status, the primary interaction, a compact history and
  optional detail. No permanent panel per variable. A stable layout. Summaries first, details on demand. Diagnostics
  collapsed. Research explanations kept apart. No flat list of 61 codes. Reasons for unavailable choices from the
  legality predicates. Forced Absent turns handled with one control. Desktop-first with a responsive approach,
  readable labels, keyboard access, signals beyond colour, reduced motion.
- **Faithfulness.** Animate only recorded per-tick facts. Present turn-level events as turn outcomes. Never imply
  individual units, positions or timing the engine does not model. No new spatial resolution or free movement.
- **Research meaning.** A human game or a new simulation is an example, not evidence for M5's findings. No claim that
  either agent is better overall or validated against humans. No winner declarations. The turn-94 lane-exit marker,
  the turn-114 observed lead and the 150-turn cap stay. No continuation-based evaluation in M6a.
- **The review gate.** No hosting, public deployment, published bundles or production M6b frontend in M6a.

## 2. Proposed choices (mine, awaiting review)

| choice | proposal | why |
|---|---|---|
| history navigation | the client keeps every resolved turn it receives and navigates them locally. The server rebuilds a history from the codes after a reload | navigation stays instant and can never touch the session. A rebuild costs about 0.09 s and 22 KB gzipped for 150 turns |
| Watch | one bounded simulation response (at most 150 turns, about 35 KB gzipped), played back by the client at 0.5×, 1×, 2× or 4×; simulations in a separate worker | a bot game costs 0.27 to 0.48 s, which would delay human turns on the single simulation thread |
| seeds | a drawn chance seed is secret until the session stops, as the policy seed is. A chosen seed is marked | a known chance seed foresees every trade, gank and tower roll |
| forced turns | the client offers one "continue" control and submits Absent once per away turn | one authoritative resolution per turn, as the plan requires, with one click |
| prototype location | `prototypes/m6a/`, isolated from `web/`, no build step, served by `prototypes/m6a/serve.py` | they run without the production toolchain, and nothing in `web/` depends on them. They can move to a throwaway branch before M6b |
| staged Play in the prototypes | any legal selection is accepted, the recorded action resolves, and the screen says whether the two matched | the prototypes judge layout and interaction. Only the backend resolves arbitrary choices |
| visitor seat | seat A by default; seat B possible (the core supports it) | as the plan's default |
| deployment structure | one container (Python 3.13, CPU torch, lanegame, the built `web/`); a demo bundle of the manifest, the 7 networks, the converted replays with frames and the built frontend; replays and findings also servable from a static host, with Play and Watch shown as unavailable there | the plan's §8, now with measured sizes (replays 22 to 35 KB gzipped each) |

## 3. Unresolved decisions (the user's)

1. **The visual direction** (§5). *Decided 2026-10-21* (§6): C as the foundation, with A's action bar and a better
   gold-lead graph. Open now: the user's review of the refined interface.
2. **Scripted opponents** (§4). None, a small selection in Watch only, or also in Play.
3. Later, as the plan says, a license before any bundle is published, and the provider, budget and go or no-go at the
   M6e checkpoint.

## 4. Scripted opponents (assessment)

The session core already seats the six scripted archetypes of OM2's library, because two of the three seed-0 research
recordings have `g_shove` in one seat. Offering them to visitors is a separate question.

- **For.** Their rules are short and legible. A visitor can predict them and so see what an agent does against a known
  style. They need no checkpoint file. Two verified seed-0 games already exist against `g_shove`.
- **Against.** All six are in `om:0`'s belief library, and `om:0` trained against them. A Watch match of `om:0` against
  one shows a matchup the package trained for, the kind where M5 measured its largest gain (+9.26 against library-like
  styles). A large lead there could easily be read as "the modelling agent is stronger", which M5 does not show. Pure
  self-play is better against competent opponents.
- **Proposal.** Offer at most three contrasting styles in Watch only, each named by its rule and marked as a scripted
  reference style that `om:0` trained against:
  - `g_shove`, "Shove + Hold (+ Pressure)";
  - `g_farm`, "Last Hit + Hold everywhere";
  - `g_bully`, "Last Hit; Trade whenever the enemy is catchable without a dive".
  Keep Play opponents to `sp:0` and `om:0`. Showing the same scripted style against both agents in one view would
  invite a comparison, so the Watch screen should present one match at a time with the example note.
- **Not a prerequisite for M6a.** The prototypes use the scripted recordings only as verified research records.

## 5. The prototypes

Three directions run from `prototypes/m6a/`. Launch them with `.venv/Scripts/python.exe prototypes/m6a/serve.py`
and open <http://127.0.0.1:8765/>. The launcher links to each direction's Play and Watch screens and to every
situation below; [the prototypes' README](../../prototypes/m6a/README.md) says what to try. Screenshots are in
`results/m6/prototypes/<direction>/`.

All three share one history controller (`prototypes/m6a/shared/core.js`). It keeps the game's progress apart from the
position on screen. Browsing never submits, a historical view disables submission, and "Return to current turn"
(Esc) restores the decision point. All three show the 61 codes as wave, behaviour and Pressure plus Recall and Ward,
with the engine's reason beside every unavailable choice. All three draw waves as HP groups, never as minions, and
animate only the two recorded ticks, with turn-level outcomes at the end of the turn. Play is staged. The recorded
action resolves, and the screen says whether the visitor's choice matched it.

### 5.1 How they differ

| | A · Arena | B · Board | C · Story |
|---|---|---|---|
| lane | an illustrated dark lane, about 63% of a 1440 × 900 screen; stone towers with plate pips, a brush patch for W; waves as HP-sized shapes; champion tokens with HP rings | a schematic SVG board of the 9 cells with labels; waves as HP bars in 8-HP segments, A above and B below; champions as letter-and-shape markers in their cells | a warm side-view strip; waves as HP-sized blocks; plain-language captions beside the objects ("Your wave: 10 HP, fighting at the lane · 1 more behind") |
| hierarchy | lane, then corner status cards, then a command bar; history in a slim right column | board and move list side by side, then a fixed status strip, then the action builder | lane, then a timeline scrubber with the earned-gold lead and event markers, then a narrated turn card beside the action composer; history as a left column with one-line summaries |
| density | the lowest; numbers live in the drawer | the highest; every value visible and aligned | medium; words instead of numbers where possible |
| history | rows with outcome icons, a "choosing… / ready ✓" row | the closest to Chess.com: numbered rows, outcome tags, marker rows at 94 and 114 | rows with a plain summary; the timeline doubles as the history at a glance |
| action choice | a command card with damage meters and a gold "Lock in" button | a segmented three-row builder with keyboard letters and a summary before "Submit" | radio cards with each wave's real HP budget and a preview sentence before "Play turn N" |
| motion | eased movement between ticks, per-tick floaters, end-of-turn toasts | short, precise steps | gentle easing, captions fading |
| detail | a drawer over the history column | tabs over the builder (close to act) | a sheet over the lower area |
| narrow screens | history below at 1100 px; at phone width the lane is a 156 px strip | the board keeps its geometry and scrolls sideways | stacked; the lane text gets small |
| cost to build in M6b | the highest: illustrated art to keep consistent | the lowest: SVG geometry and plain components | moderate: caption logic and the timeline |

**Tradeoffs.**
- *A* is the most engaging at a glance and the most game-like, but the least informative without opening the drawer.
  Its art costs the most to keep consistent, and a phone needs its own lane composition.
- *B* is the easiest to audit and the most faithful to the history requirement. It looks like a tool rather than a
  game. Its lane gets a smaller share than A's, and its detail tabs cover the builder.
- *C* is the easiest to follow for a newcomer and the best at showing a whole game's arc. Its bottom row is tight at
  900 px tall, and its captions need careful wording to stay factual.

### 5.2 Recommendation

**Use B as the base and take two elements from C.** B meets the approved requirements most directly. Its move list
is the Chess.com pattern, every value on screen traces to the frame contract, and it is the cheapest to build and test
in M6b. From C, take (1) the plain-language captions placed beside the lane objects, which help the newcomer audience
without a permanent panel, and (2) the timeline with event markers and the 94 and 114 marks as the Watch scrubber. In
the refinement, give B's lane more height by folding the status strip into labels at the lane, and open detail beside
the builder rather than over it. If the user prefers a game-like feel, A is the alternative, with the higher art cost
named above.

### 5.3 Situations and the recordings used

| situation | game (data file) | turns |
|---|---|---|
| quiet farming | `h2h_om_s0_vs_sp_s0` (verified M5 recording, `sp:0` in seat A against `om:0`) | runs from turn 3 and turn 126 |
| stacked waves and a crash | the same | turns 19 and 31 (a 10-HP wave crashes; 7 HP stay under the tower) |
| a trade with several outcomes | the same | a hit (turn 13), a miss (turn 30), no contact (turn 7); turn 147 has a miss, a hit and a tower shot together |
| Recall and the forced absent turn | the same | Recall at turn 24, away at turn 25 |
| a death and forced absent turns | `om_s0_vs_g_shove` (verified M5 recording, the scripted `g_shove` in seat A) | seat A dies at turn 11 and is away for turns 12 to 14 |
| history during play, Watch playback, detail open | the head to head, and `sim_om_vs_sp` (a new simulation, chance seed 110,000, policy seed 110,500) | any turn; each direction's screenshots 05 to 07 |

The head-to-head game has no death, so deaths come from the `g_shove` game. Every situation the brief listed is
shown with a verified recording. None was fabricated.

### 5.4 What the prototypes do not show

- The real request loop. Play is staged, and the data files hold whole games, which the production client never
  receives.
- Record download, a seed picker and the opponent picker of a new session.
- A full phone design. Each direction has a rough responsive approach only.
- Final art or copy. Captions and labels are drafts for review.

## 6. The direction chosen and the refinement (2026-10-21)

**The user's decision (approved direction).** After playing the prototypes, the user chose prototype C as the visual
foundation, with prototype A's bottom action controls and an improved gold-lead graph, refined into one interface.
This differs from my recommendation in §5.2 (B as the base), which the user did not take. No new set of directions.

**Approved requirements of the refinement** (the user's brief of 2026-10-21, second round):
- **Purpose.** A research demonstration: readable, functional and carefully designed, without the branding,
  spectacle, onboarding or product polish of a commercial game. No invented product names, logos, marketing copy,
  decorative badges or dashboards of implementation details. A plain descriptive title at most.
- **Less on screen.** The lane stays prominent. No value appears twice (HP in particular), the opponents' identities
  are not repeated across panels, and the long Watch explanation, repeated "Farming" rows, redundant wave descriptions
  and persistent explanatory paragraphs go. Provenance stays as one small record-type label, explained under "About
  this game". Exact wave numbers, XP progress, investment, full events, tick traces, beliefs and model metadata move
  into one collapsed Details panel with a few sections.
- **Typography and treatment.** One neutral sans-serif family (the system font stack, no external fonts), normal
  width, sentence case, a small consistent type scale, tabular numerals for changing numbers. Neutral surfaces,
  restrained borders, minimal shadows. No serif headings, display fonts, heavy uppercase, glows, gradients, textures or
  unnecessary cards, pills, icons and badges. Consistent team colours, clear selection states, good contrast.
- **Action controls.** A's structure in a stable area below the lane: a wave row, a behaviour row, a compact objective,
  Recall and Ward set apart, one submit button naming the selection, contextual help and reasons for unavailable
  choices.
- **Gold-lead graph.** Taller, titled "Gold lead" with the sign convention, a zero baseline, team-coloured shading
  above and below zero, a selected-turn marker with its value, hover and click synchronized with the lane and history.
  Event markers and research checkpoints hidden by default. No future turns during a running human game. Not a win
  probability or position evaluation.
- **The shared timeline** of §1 is kept, with readable action names instead of abbreviations.
- **Staged play.** Play must execute the selected action if the session core can be connected with modest work;
  otherwise the screen must present an interaction walkthrough, never a submission that silently substitutes a
  recorded action. No combat mechanics are diagnosed or changed from staged outcomes.

**What was done.**
- Play is **live**. A JSON service over the session core (`lanegame/demo/service.py`) runs in the local server
  (`prototypes/m6a/serve.py`), and the refined interface submits the visitor's own action to it
  ([technical report](m6a_technical_report.md) §7).
- The refined interface is `prototypes/m6a/refined/`, with screenshots in `results/m6/prototypes/refined/`.
- The three earlier directions stay for reference, marked in the launcher as superseded staged walkthroughs.

**The refined interface, against C and A.**
- *Removed.* The product names and badges, the serif headings and display fonts, the HP pills over the champions
  (HP appears once, in the status line), the wave captions, the long Watch explanation, the "Farming" rows and the
  persistent explanations. A plain "Laning simulator" title and one small record-type label remain.
- *History.* One row per turn with both actions in words ("Slow push, Trade", "Recall", "Away"), and a second line
  only when the turn had a major outcome. A viewed earlier turn shows "Viewing turn X · the game is at turn Y" with
  "Return to current turn", and submission is disabled.
- *Action controls.* A's rows, restyled: a wave row, a behaviour row, a Pressure toggle with its reason, Recall and
  Ward set apart with theirs, and one button naming the choice ("Play turn 6: Slow push, Trade"). A fixed-height
  help line holds hover help, reasons and engine errors, so nothing shifts. In Watch the same area holds playback and
  the selected turn's summary.
- *Gold lead.* A graph about 136 px tall with the sign convention and "Not a win probability", a zero baseline,
  team-coloured shading, y-ticks, a selected-turn marker with its value, hover inspection that highlights the history
  row, and click to select. Events and the 94 and 114 checkpoints sit behind "Show events". A running game shows no
  future turns.
- *Details.* One right-hand drawer over the screen with five sections: turn events, waves and ticks, champions,
  belief (when it may be shown) and the game record.

**Verified.**
- *Live Play.* In headless Edge against the engine, 36 Trade submissions in a game against `om:0` all resolved as
  submitted, with no substitution: 9 hits for 2,173 HP, 19 misses and 8 out of reach. The client's turn count matched
  the server's, browsing the history left the game where it was, and the page reported no errors.
- *The agent's checks.* Its scripted check passed 13 of 13: an illegal code refused with the engine's reason, a
  disabled choice that cannot be selected, the stop ending the server session, the busy turn 147 shown correctly, and
  the graph hover highlighting its row.
- *Screenshots.* The 12 screenshots in `results/m6/prototypes/refined/` cover the quiet lane, the busy turn, the action
  controls, history review, Watch, Details, the graph, a forced absence, a stopped game, phone width and 1024 px.

**Limitations.**
- Play needs the local server and the 7 checkpoint files.
- There is no record download yet.
- The phone layout is rough.
- Hovering the graph highlights a history row without scrolling to it.
- The opponent's sampling seed is drawn per game, so two games from the same chance seed differ.

**Second review round (the user, 2026-10-21).** The user asked for four changes, all made:
- *Plain names.* The research ids (`om:0`, `sp:0`; the 0 is M5's seed index) and "scripted g_shove" left the visible
  interface. The players are the "Self-play agent", the "Adaptive agent" (M5's opponent-modelling agent) and the
  "Always-push bot". The game list reads "Self-play agent vs. adaptive agent" and similar, grouped as recorded research
  games or games generated for the demo. The ids and the bots' rules appear only in Details, under "Game record".
- *A gold graph like the game's own.* Blue above zero while the left player (you, in Play) has earned more gold, red
  below while the right player has, with matching shading. The axis numbers carry no sign, captions say who is ahead,
  and a tie stays grey. The y-axis is symmetric and at least ±2 gold. In a running game the x-axis spans the turns
  played (at least 20) instead of all 150. The team colours across the interface became blue and red to match.
- *Absence shown once.* "Recalling, back in 1 turn" appears in the status line only. The action area greys its
  choices and offers one "Continue" button.
- *Plain wording elsewhere.* The record-type label reads "Recorded research game", "Generated game" or "Your game,
  live". "About this game" and the belief section describe the agents, the bots and the research's findings without
  jargon.

Re-checked after the changes: the live check (35 Trades, all executed as chosen; absences continued one turn at a
time), no page errors, and no ids or "scripted" in the visible text.

**Minions as balls (the user's request, 2026-10-21).** The square HP blocks with a number became clusters of balls.
The engine does not model single minions: a wave is a pool of HP (8 for a new wave, up to 24 when waves stack), and
every HP point that dies pays the gold of one HP. So each ball is one HP of minions, and a legend on the lane says so.
- A cluster fills column by column from the back, so HP removed by a champion or by the other wave leaves from the
  front edge. The removed balls fade out at the tick the engine records it.
- Waves behind the leading one are drawn lighter. Each cluster keeps a small HP count beside it.
- Crashed minions sit beside the base of the tower they attack.
- The earlier blocks remain at `?minions=blocks` for comparison.

Checked: after animated turns each cluster's ball count equals its HP, balls fade as HP is lost, and the page reports
no errors. Screenshots are in `results/m6/prototypes/refined_balls/`.

**A game view with overlaid panels (the user's brief, 2026-10-21).** The stacked sections became one scene: a top-down
lane fills the window and the controls float over it in light translucent panels (`prototypes/m6a/hud/`; `refined/`
stays as the previous layout).
- *Scene.* The simulator's nine cells on one road with faint cell dividers, bases at the ends, towers seen from above
  with their three plates, and the Vision node W north of cell 0 on a short path. Nothing suggests free movement or
  other areas. Your minions walk the upper half of the road and the opponent's the lower half; champions stand on the
  road's edges at their cells. The lane is drawn in the space the panels leave free and scales with it.
- *Panels.* Champion status in the top corners (the only place HP, level and absence appear); Play/Watch and game
  controls at the top centre, with the turn and record type below; a collapsible turn history on the left; the action
  strip (A's arrangement) at the bottom; a compact gold graph at the bottom right that expands on request; Details on
  demand. History review tints the scene and turns the submit button into "Return to current turn".
- *Checked.* An overlap test at 1440 × 900, 1280 × 720 and 1024 × 700, history open and closed, at the start, a busy
  turn, a crash, a ward, a death and a stacked wave, and in live games: no champion, minion cluster, tower or the W node
  under any panel, and clicks on empty overlay space reach the scene (42 of 42 cases). Live play: 35 of 35 actions
  executed as chosen, history review leaves the game unchanged, no page errors.
- *The reference image* (a top-down crop of a mid lane, sent after the first version) set the composition: the road
  now runs diagonally from the left player's base at the lower left to the right player's at the upper right, the
  towers stand beside the road at cells −2 and +2, and the Vision node is drawn as a "Ward spot" just off the road
  beside the middle cell, where the reference has its brushes. No river, water, jungle or brush is drawn: the
  simulator has none, and nothing may suggest more walkable ground. Labels, balls and champion tokens stay upright.
  The overlap check (42 of 42 clear) and the live check (34 of 34 actions as chosen) were run again on this version.

**Open.** The user's visual review of the game view. Scripted opponents (§4) remain undecided.
