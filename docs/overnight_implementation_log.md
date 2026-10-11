# Overnight implementation log: M2b–M2e, M3a–M3c

Running log of the autonomous session started 2026-10-11. Newest entries at the bottom of each section.
Work happens on the local branch `m2-m3-autonomous` (one commit per verified milestone, nothing pushed).

## Status

| Milestone | Status | Commit |
|---|---|---|
| M2b Recall and absence | completed and verified (39 champion + 65 env tests) | see git log |
| M2c Stochastic trading and skill | not started | |
| M2d Ganks, death, respawn | not started | |
| M2e Validation and benchmarking (gate) | not started | |
| M3a PPO vs scripted opponents | not started | |
| M3b Self-play with frozen snapshots | not started | |
| M3c Evaluation and approximate best responses | not started | |

## Session start

- Repository state verified: branch `main` clean at `08865df` ("phase m1 and m2a").
- `tests/test_champions.py`: 25 passed (0.7 s). Full-suite baseline: **395 passed in 5 min 33 s** (matches the M2a report).
- PyTorch was not installed; installed `torch 2.14.1+cpu` into `.venv` (needed for M3, optional extra only).
- Shell note: heredocs containing backticks fail in this Bash wrapper; patch scripts go to the scratchpad.

## Log

### M2b Recall and absence

Implemented in `lanegame/champions.py` (+ `config.py`, `env/core.py`, `env/agents.py`, `env/trace.py`).

- State: `absent` = number of upcoming turns the champion must sit out. Recall sets `recall_turns - 1`
  (the Recall turn itself is the first missed turn), each absent turn counts it down; the champion is at
  `-T` while away and gets full HP when the timer reaches 0. Default `recall_turns = 2`: Recall on t,
  forced ABSENT on t+1, back at the tower with full HP on t+2. Configurable (tested for 1, 2, 3).
- Legality: one Recall code (`champ_act(SP, B_RECALL)` = 12) when present and `pos <= recall_line (-1)`;
  `{ABSENT}` is the whole legal set while absent; ABSENT is never legal otherwise.
- **Deviation from the brief (documented in the module):** the brief suggested disallowing Recall under
  an immediate credible threat. With a state-based rule ("enemy champion in contact at turn start"),
  exploration showed a Safe champion at -1 facing a Holding opponent at the lane (0) could *never* recall
  (-1 and 0 are within trade range, and no behavior walks voluntarily to -2). Because actions are
  simultaneous, whether the threat materializes is only known at resolution, so instead: an enemy
  **Trade that reaches the recaller interrupts the recall** (it stays where it was, channeling: no farm, no
  wave contribution, no zoning, traded as a vulnerable defender, gankable with Safe's term). A dive can
  interrupt a recall at the tower. This is one conditional rule (like Yield's evasion) and creates a
  recall-versus-trade guessing game.
- Away sides: wave plays Slow Push with no champion reach, no objective, no farm, no zoning, no trade,
  gank probability 0; a present Trade whose opponent is absent behaves like Hold. Exposure of an away side
  is reported as -T.
- Economics (no recall penalty): from a constructed state where A's 8 HP crash sieges B's tower, a recall
  costs 0.00 resources over the next 6 turns versus staying; from a state where B's 8 HP wave walks into
  A's tower it costs 0.62 (test `test_well_timed_recall_costs_less_than_a_poorly_timed_one`).
- Tests: new `tests/test_champion_recall.py` (14 tests); three M2a tests updated (Recall is no longer
  reserved). 39 champion tests pass in 0.6 s. env-demo champion throughput 7,500 turns/s.
