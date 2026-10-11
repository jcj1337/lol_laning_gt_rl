# Overnight implementation log: M2b–M2e, M3a–M3c

Running log of the autonomous session started 2026-10-11. Newest entries at the bottom of each section.
Work happens on the local branch `m2-m3-autonomous` (one commit per verified milestone, nothing pushed).

## Status

| Milestone | Status | Commit |
|---|---|---|
| M2b Recall and absence | completed and verified (39 champion + 65 env tests) | see git log |
| M2c Stochastic trading and skill | completed and verified (52 champion + 65 env tests) | see git log |
| M2d Ganks, death, respawn | completed and verified (67 champion + 65 env tests) | see git log |
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
- Committed as `acabf74`.

### M2c Stochastic trading and mechanical skill

- Hit chance of a connected trade i -> j:
  `h = clip(0.55 + 0.4 (m_i - m_j) + beta[beh_j] + 0.05 (sigma_i - sigma_j) - 0.15 [j in own tower cell], 0.05, 0.95)`,
  `beta = (Hold 0, Yield -0.25, Trade 0, Safe -0.40, Recall/channel 0)`; damage 0 / 1 / 2 with
  `P = (1-h, h(1-q), hq)`, `q = clip(0.3 + 0.2 (m_i - m_j), 0, 1)`. Roadmap E.5 defaults, except that the
  2-HP chance uses the skill *difference* (the roadmap had `q2_skill * m_i`), so a skilled defender also
  mitigates. Every term is a `GameConfig` field (`trade_hit_*`, `trade_double_*`).
- Skill enters only these two probabilities (test: positions, modes, lane, farm and gank risk are identical
  under different skills).
- Tower shot: a connected trade ending in the enemy tower cell (a dive) takes 1 HP with p = 1.0, or 0.5 if
  the diver's crashed minions are under that tower (start of turn or crashing this turn).
- A Hold defender does not hit back; Trade vs Trade both roll from start-of-turn HP.
- All chance events of a champion-mode turn come from one fixed-length vector of 10 uniforms per turn
  (`champions.N_CHANCE`, layout `U_TRADE_HIT`, `U_TRADE_DOUBLE`, `U_TOWER`, `U_GANK`, `U_GANK_DAMAGE`), drawn
  whatever the actions: identical chance draws turn by turn for any two policies (common random numbers).
  `resolve_turn(..., u=None)` means no chance event. Legacy configurations keep their two gank draws.
- HP floored at 1 (temporary nonlethal clamp, removed in M2d). Ganks still the legacy resource loss.
- Matchups (300 episodes, K = 32): Trade vs Hold lands 17.2 of 31 connected trades, 22.3 HP dealt;
  Trade vs Trade symmetric (23.1 / 22.5 HP); Yield and Safe evade all 31 trade attempts.
- Tests: new `tests/test_champion_combat.py` (13 tests: formula defaults and clipping, skill monotonicity
  for offense and defense, behavior ordering, execution-only skill, opportunity cost, evasion, simultaneity,
  interrupted-recall and partial-escape terms, tower shot free/busy, 10,000-draw distribution check within
  4.5 SE, mirror equivariance with mirrored chance, no guaranteed damage, reproducibility and CRN).
  52 champion tests pass in 2.6 s.
- Committed as `e83618d`.

### M2d Ganks, death, respawn

- Champion mode has its own gank-occurrence parameters (`champ_gank_*`); the legacy `gank_*` fields and
  the legacy expected-loss model are untouched. Occurrence:
  `p = clip(0 + 0.12 (e+T)/2T + beh{H .01, Y .01, T .03, S 0} + 0.02 [Shove, non-Safe] - 0.01 dsigma, 0, 0.30) * (1 - 0.75 [vision])`.
  The legacy unconditional base 0.15 (harmless as an expected loss because it cancelled in the zero-sum
  payoff) would have produced ~7 HP-damaging ganks per player per game; with base 0 a Hold champion at the
  lane is ganked ~2.1 times per 32-turn game, a Safe one ~1.0, and a champion under its own tower almost
  never. Vision at -1 saves 0.0225 per turn: not mandatory when safely positioned.
- Damage given a gank: `Binomial(4, p_e)`, `p_e = clip(0.2 + 0.6 (e+T)/2T - 0.2 [Yield/Safe] - 0.2 (m - 0.5), 0, 1)`
  (0 to 4 HP; mean 0.8 at the own tower, 2.0 at the lane, 3.2 under the enemy tower). 4 < 6 HP, so a
  full-HP champion never dies to one gank. Skill enters gank damage (escape execution) with a small term:
  the roadmap had left this open; the brief asks skill to improve defensive execution. Vision changes only
  occurrence (tested).
- Death: all damage of the turn (trades, tower shot, gank) is summed; HP <= 0 -> dies at the end of the
  turn (keeps that turn's farm), `absent = death_turns = 3`, HP 0 while away, ABSENT on t+1..t+3, back at
  the own tower with full HP on t+4. Simultaneous deaths possible. Temporary M2c HP floor removed.
- Reward in champion mode: farm + plates + `kill_bounty` (default 0) differences. The legacy `gank_loss`
  term is not charged (no double counting); `info["gank_payoff"]` is always 0 in champion mode.
- Scripted matchups (400 episodes, K = 32, no recall logic in these agents): Hold vs Hold 0.05 +- 0.04,
  2.1 ganks, 0.33 deaths per player; Trade vs Hold -0.58 +- 0.06 for the trader (Hold dies 2.7 times but
  the trader halves its own farm on 31 connected turns); Trade vs Yield +12.0, Trade vs Safe +12.9;
  Hold vs Safe +9.0; Shove-Hold vs Hold -3.9. Payoff std fell from ~16 to ~1 (no 5-resource gank loss).
- Tests: new `tests/test_champion_ganks.py` (15 tests: occurrence defaults and ordering, exposure kept while
  retreating, damage distribution and inputs, no one-gank death from full HP and possible zero damage,
  vision only on occurrence, 10,000-draw binomial check, ganks under tower, exact death timeline,
  simultaneous deaths, damage sources adding up, recall vs death, episode end during death, rewards without
  gank loss and with bounties, reproducibility, death cost depending on wave state). Three earlier tests
  updated. 67 champion tests + 65 env tests pass.
