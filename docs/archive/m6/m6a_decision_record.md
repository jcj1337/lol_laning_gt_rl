# M6a decision record

*2026-10-21. Status: **M6a complete.** The backend is built and verified ([technical report](m6a_technical_report.md)),
and the user approved the final interface design on 2026-10-21 after six review rounds. §7 states that design, which
M6b builds from. M6b has not started.*

The record separates what the user decided, what I proposed and what remains open. Plan: [m6_plan.md](m6_plan.md).

*2026-10-22 (M6b): the production application `web/` implements the design of §7 ([M6b report](m6b_report.md)). The
prototype folders were removed. The approved glass prototype's code is archived unchanged at
`legacy/m6a_glass_prototype/` (it no longer runs: its server and data files went with the prototypes), and the
screenshots of every round stay in `results/m6/prototypes/`. Paths under `prototypes/m6a/` below describe that time.*

## 1. Approved requirements (the user's M6a brief, 2026-10-21)

- **Scope.** M6 presents the existing research. It changes no mechanic, reward, training objective, trained network or
  historical result, and needs no training.
- **Two primary modes.** In Play, a human chooses discrete actions against `sp:0` or `om:0`. In Watch, two bots play a
  bounded simulated match, which the app generates first and then plays back through a timeline. Watch is in the
  initial release.
- **Bots.** The two seed-0 M5 agents, for either bot seat, by M5's seed-selection convention. Scripted strategies are
  a design decision (§4).
- **Shared infrastructure.** Play and Watch share the lane view, the turn history, the event presentation and the
  recordings.
- **Stack and location.** React with TypeScript and Vite in `web/` in this repository, and FastAPI with Python
  authoritative for every rule, unless a concrete blocker appears. M6a found none.
- **Docs tracked.** The blanket `docs/` ignore rule is gone.
- **Turn history** (plan §5.1). Chess.com's interaction pattern without its branding. Each row is one simultaneous turn
  with both actions. Turn t shows the state after turn t, and turn 0 is the initial position. The controls are first,
  previous, next, latest and row clicks, plus autoplay, pause and speed in Watch and in review. Browsing during play
  changes nothing, disables submission and offers a prominent "Return to current turn". Everything on screen follows
  the selected turn. No undo, branching, grading or win probability.
- **Hidden information.** No history endpoint or client data carries an unresolved opponent action, a future state or
  a hidden belief. The belief is hidden by default, shown only if the visitor chooses that at the start, and available
  in full after the session stops.
- **Clarity** (plan §5.2). The lane comes first, then essential status, the primary interaction, a compact history and
  optional detail. No permanent panel per variable, a stable layout, and summaries before details. Diagnostics start
  collapsed, and research explanations stay apart. The 61 codes never appear as a flat list, and every unavailable
  choice gives the reason from the legality predicates. One control handles forced Absent turns. The design is
  desktop-first with a responsive approach, readable labels, keyboard access, signals beyond colour and reduced motion.
- **Faithfulness.** The app animates only recorded per-tick facts and shows turn-level events as turn outcomes. It
  never implies individual units, positions or timing the engine does not model, and adds no spatial resolution or
  free movement.
- **Research meaning.** A human game or a new simulation is an example, not evidence for M5's findings. The app claims
  neither agent is better overall or validated against humans, and declares no winner. The turn-94 lane-exit marker,
  the turn-114 observed lead and the 150-turn cap stay. M6a adds no continuation-based evaluation.
- **The review gate.** M6a buys no hosting, deploys nothing, publishes no bundle and builds no production frontend.

## 2. Proposed technical choices (mine, carried into M6b as proposals)

The prototypes exercised each choice below. The user approved the resulting interface, not each choice by name, so
M6b treats them as proposals it may revise.

| choice | proposal | why |
|---|---|---|
| history navigation | the client keeps every resolved turn it receives and navigates them locally. The server rebuilds a history from the codes after a reload | navigation stays instant and can never touch the session. A rebuild costs about 0.09 s and 22 KB gzipped for 150 turns |
| Watch | one bounded simulation response (at most 150 turns, about 35 KB gzipped) that the client plays back at 0.5×, 1×, 2× or 4×, with simulations on a separate worker | a bot game costs 0.27 to 0.48 s, which would delay human turns on the single simulation thread |
| seeds | a drawn chance seed stays secret until the session stops, as the policy seed does. The record marks a chosen seed | a known chance seed foresees every trade, gank and tower roll |
| forced turns | one "Continue" control that submits Absent once per away turn | one authoritative resolution per turn, as the plan requires, with one click |
| prototype location | `prototypes/m6a/`, isolated from `web/`, with no build step, served by `prototypes/m6a/serve.py` | the prototypes run without the production toolchain, and nothing in `web/` depends on them |
| Play in the prototypes | *superseded.* The first prototypes staged Play against recordings. Since the refinement round every Play submission goes to the session core through `lanegame/demo/service.py` | the user mistook staged play for the game (§6) |
| visitor seat | seat A by default. The core also supports seat B | the plan's default |
| deployment structure | one container (Python 3.13, CPU torch, lanegame, the built `web/`); a demo bundle of the manifest, the 7 networks, the converted replays with frames and the built frontend; replays and findings also servable from a static host, where Play and Watch show as unavailable | the plan's §8, now with measured sizes (replays are 22 to 35 KB gzipped each) |

## 3. Unresolved decisions (the user's)

1. **Scripted opponents** (§4). None, a small selection in Watch only, or also in Play.
2. **A license** before M6d publishes any bundle.
3. **The provider, budget and go or no-go** at the M6e checkpoint.

The visual direction is no longer open. The user settled it on 2026-10-21 (§6, §7).

## 4. Scripted opponents (assessment)

The session core already seats the six scripted archetypes of OM2's library, because two of the three seed-0 research
recordings have `g_shove` in one seat. Offering them to visitors is a separate question.

- **For.** Their rules are short and legible. A visitor can predict them and so see what an agent does against a known
  style. They need no checkpoint file, and two verified seed-0 games already exist against `g_shove`.
- **Against.** All six are in `om:0`'s belief library, and `om:0` trained against them. A Watch match of `om:0` against
  one shows a matchup the package trained for, the kind where M5 measured its largest gain (+9.26 against library-like
  styles). A large lead there could easily read as "the modelling agent is stronger", which M5 does not show. Pure
  self-play is better against competent opponents.
- **Proposal.** Offer at most three contrasting styles in Watch only. Name each by its rule and mark it as a scripted
  reference style that `om:0` trained against:
  - `g_shove`, "Shove + Hold (+ Pressure)";
  - `g_farm`, "Last Hit + Hold everywhere";
  - `g_bully`, "Last Hit; Trade whenever the enemy is catchable without a dive".

  Keep Play opponents to `sp:0` and `om:0`. One view showing the same scripted style against both agents would invite
  a comparison, so Watch shows one match at a time with the example note.

## 5. The first prototypes (superseded)

Three directions ran first from `prototypes/m6a/`: A · Arena (`arena/`), a game-like illustrated lane; B · Board
(`board/`), a schematic board beside a Chess.com-style move list; and C · Story (`story/`), a narrated lane with a
lead timeline. All three staged Play against verified recordings. Screenshots are in
`results/m6/prototypes/<direction>/`.

I recommended B as the base with two elements from C. The user chose C instead, with A's action controls (§6).

The recordings behind every situation the brief listed are verified M5 games, and none was fabricated:

| situation | game (data file) | turns |
|---|---|---|
| quiet farming | `h2h_om_s0_vs_sp_s0` (verified M5 recording, `sp:0` in seat A against `om:0`) | runs from turn 3 and turn 126 |
| stacked waves and a crash | the same | turns 19 and 31 (a 10-HP wave crashes; 7 HP stay under the tower) |
| a trade with several outcomes | the same | a hit (turn 13), a miss (turn 30), no contact (turn 7); turn 147 has a miss, a hit and a tower shot together |
| Recall and the forced absent turn | the same | Recall at turn 24, away at turn 25 |
| a death and forced absent turns | `om_s0_vs_g_shove` (verified M5 recording, the scripted `g_shove` in seat A) | seat A dies at turn 11 and is away for turns 12 to 14 |
| Watch playback | `sim_om_vs_sp` (a new simulation, chance seed 110,000, policy seed 110,500) | any turn |

The head-to-head game has no death, so deaths come from the `g_shove` game.

## 6. The review rounds (2026-10-21)

Each round answered the user's feedback on the previous one. Every round reran the live-play check (each submitted
action resolved as chosen) and checked that the page reported no errors.

1. **Refined C, live Play** (`refined/`). The user chose C as the foundation with A's action controls and a better
   gold-lead graph. They also mistook staged Play for the game, so the session core now serves Play through
   `lanegame/demo/service.py` ([technical report](m6a_technical_report.md) §7). The refinement removed product names,
   badges, serif and display fonts, repeated values and persistent explanations. It moved exact numbers into one
   Details drawer.
2. **Plain names and a game-style gold graph.** The players became the "Self-play agent", the "Adaptive agent" and the
   "Always-push bot". The research ids (`sp:0`, `om:0`; the 0 is M5's seed index) appear only under Details. The gold
   graph turns blue above zero while the left player has earned more gold and red below, with unsigned axis numbers
   and captions saying who is ahead. The team colours became blue and red to match. Absence shows in one place only.
3. **Minions as balls.** Each ball is one HP of minions. The engine models a wave as a pool of HP, not single minions,
   and a legend on the lane says so. Lost HP fades from the front of the cluster at the tick the engine records it.
4. **A game view with panels over it** (`hud/`). A top-down lane fills the window, and the controls float over it. The
   user's reference image, a top-down crop of a mid lane, set the diagonal composition. The road runs from the left
   player's base at the lower left to the right player's at the upper right, with the towers beside it and the ward
   spot beside the middle cell. The app draws no river or terrain, because the simulator has none.
5. **One glass surface** (`glass/`). Three layouts merged the panels into one frosted surface: a dock, a rail and a
   console. The user chose the dock, then asked for an automatic layout, flat turn buttons, dark mode and a visible
   ward spot.
6. **The bush and stepped ticks.** The ward spot became a bush in the bases' style. A champion in the bush is drawn
   see-through, and the token's tooltip says this simulator hides no one. Each turn now pauses at both ticks and the
   end, each with a plain description of what the engine records at that point. The user reported being stuck after
   warding. The engine showed that any lane action leaves the ward spot and that Recall is not legal there. The
   interface behaved correctly, but its Recall reason said "Vision node W" and gave no way out. The demo's wording now
   says "ward spot" and "bush", and the help line tells the player how to leave.

## 7. The final design (approved 2026-10-21; the reference for M6b)

The reference implementation is `prototypes/m6a/glass/`. Its screenshots are in `results/m6/prototypes/glass/`. The
other prototype folders (`arena/`, `board/`, `story/`, `refined/`, `hud/`) are history.

- **The scene.** A top-down lane fills the window. The road runs diagonally from the left player's base at the lower
  left to the right player's base at the upper right. It shows the simulator's nine cells with faint dividers, and the
  towers stand beside it at cells −2 and +2 with their three plates. The ward spot is a bush beside the middle cell,
  joined to the road by a dashed path, with a translucent green fill, a thin outline and spikes on top. The app draws
  no terrain beyond that.
- **Minions.** Each ball is one HP of minions. Waves behind the leading one are lighter, and a small count sits beside
  each cluster. Crashed minions sit beside the tower they attack.
- **Champions.** Letter-and-shape tokens in team colours (A blue circle, B red diamond) stand on the road's two edges
  at their cells. A small chip beside each token gives the name, level and HP. A champion away at base is faded and
  dashed, and a champion in the bush is see-through.
- **One glass surface.** A frosted translucent dock along the bottom holds the recent turns with a flat segmented
  navigation control, the turn line, the action rows (wave, behaviour, Pressure, Recall and Ward, and a gold submit
  button that names the choice) and a compact gold-lead graph. The full history and a large graph open on request.
  Details and About open on demand. The top controls sit in one small floating pill.
- **Layout by device.** The dock is the default. A phone held sideways gets a rail down the left, and a phone held
  upright gets a compact full-width dock. The layout follows rotation without losing the game.
- **Themes.** Light and dark follow the system setting, with Auto, Light and Dark in About.
- **Turns.** A resolved turn pauses at the start, tick 1, tick 2 and the end. Each step has a plain description, and
  turn-level events appear only at the end. "Play ticks automatically" runs the steps at about 1.1 s each at 1×. Watch
  autoplay and forced away turns always run them automatically.
- **The gold graph.** Blue above zero while the left player has earned more gold and red below, with a grey tie, an
  unsigned axis and a selected-turn marker. In a running game the x-axis spans only the turns played.
- **Words.** Plain names for the players, plain reasons for every unavailable action, and no research ids outside
  Details.

**Verified on the final design.**
- **Overlap.** No panel covers a champion, minion cluster, tower or the ward spot in 240 of 240 cases (light and dark
  at 1440 × 900, 1024 × 700, 844 × 390, 915 × 412 and 390 × 844, every expanded state).
- **Live play.** Every stepped turn resolved with the submitted action, and submission waits until the turn has
  finished stepping.
- **Tests.** The 29 demo tests pass.

**Known limits that M6b inherits.**
- The lane is small on phones, about half its desktop size sideways and a third upright.
- Manual stepping costs about four clicks per turn.
- The tick descriptions do not explain minion-versus-minion losses or why minions fired, because the per-tick record
  does not say.
- Phones were tested in Edge's device emulation only.
