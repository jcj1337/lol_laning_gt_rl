# Design Philosophy

*Established 2026-10-12 (milestone M4.2e-B). This document is the standard against which mechanics are evaluated. It
is deliberately short on numbers: parameters belong in `lanegame/config.py` and in the milestone reports.*

## The central idea

**Simplify how League of Legends works without fundamentally changing why its laning strategies work.**

The simulator is an abstraction of one lane. It does not reproduce League's numbers, champions, abilities or minion
AI, and it is not meant to. What it must preserve is the *causal structure* that makes League's laning strategies
correct: wave management, trading, zoning, recalls and sieges should be good or bad here for the same reasons they
are good or bad in League. If a strategy works in the simulator for a reason that does not exist in League, or fails
for a reason League does not have, the mechanics are wrong even if the numbers look plausible.

The environment exists to train and study agents with multi-agent reinforcement learning (PPO self-play) and to
compare them with exact game-theoretic baselines. It therefore has to stay compact, fast, mathematically coherent and
observable, and its equilibria have to be the product of its rules, not of its reward shaping.

## 1. Core design principles

1. **Consequences, not labels.** A champion's income, safety and options follow from where it stands, what the
   minions do, what the opponent can do to it and what it did last turn. No rule may pay or penalise an action because
   of what the action is called or because of who the opponent is. In particular, there are no farming penalties that
   activate merely because a stronger champion is nearby: zoning has to come from a credible threat and its
   consequences.
2. **One set of physical rules.** Minions walk, fight, merge, crash and die by the same rules whatever the players
   choose. Champion actions change the inputs to those rules (damage on minions, position, presence), never the rules
   themselves. A strategy such as freezing or slow pushing is a pattern of inputs that the physics rewards, not a mode
   the engine switches into.
3. **Resources are conserved and accounted for exactly.** Every minion HP that dies is a farm opportunity for exactly
   one side; gold, XP, plates and kills are integer ledgers; the reward is the exact zero-sum difference of the gold
   the two sides earn (farm, plate gold and kills). Nothing is created by a formula and nothing silently disappears.
4. **Power is earned and bounded.** Levels and investment make a champion stronger in combat and in wave control,
   through one monotone, saturating map. A lead must matter and must keep mattering, but no state may allow a
   one-turn kill from full health or an unbounded snowball.
5. **Positions matter because reach is limited.** Farming, wave damage, trading and experience each have a reach: one
   cell for the first three and two cells for experience in the research preset. Standing somewhere is a commitment: a champion in reach of the enemy at the start of a turn
   cannot undo that by reacting to the enemy's action.
6. **Symmetry and determinism.** The game is mirror-symmetric between the seats; the only randomness is an explicit
   per-turn chance vector that both policies see identically; every trajectory is replayable from (state, actions,
   chance).
7. **Markov and lossless observation.** Each player observes everything it needs to act optimally from the current
   state, and nothing else. Hidden state is forbidden; the observation is decodable back to the state.
8. **Small enough to be understood.** Five lane cells and one Vision node, four wave actions, five behaviours, one
   siege objective and one Ward.
   Every mechanic added must earn its place by changing which strategies are rational, and must be removable by a
   configuration switch that reproduces the previous game bit for bit.

## 2. Intended qualitative similarities to League

These are the relationships we want to recognise in the simulator's equilibria. They are intentions, not guarantees:
the point is that *when* they appear, they appear for League's reasons.

- **Wave advantages accumulate.** A small edge in a minion fight grows through reinforcements into a stack; a stack
  crashes; a crash under a tower comes back as a bounce. Slow pushing, fast pushing, freezing, crashing and bouncing
  are all produced by minion interactions plus champion damage, never by scripted wave movement.
- **Combat strength comes from progression.** Levels and investment raise damage and wave clear. A stronger champion
  clears faster, wins trades and can commit to fights the weaker champion must avoid.
- **Threats shape positions.** A champion that can win a fight controls where its opponent can safely stand. The
  weaker champion gives up position, farm or wave control when the expected cost of staying exceeds the cost of
  conceding, and keeps contesting when it does not. Standing forward is a credible threat only when the threatened
  champion cannot always escape it.
- **Trading has opportunity costs on both sides.** Fighting costs farm and wave control for the time it takes; losing
  a fight costs health, forces a recall or a death, and surrenders the wave in the meantime.
- **Recall timing is a decision.** Leaving the lane costs what dies in one's absence, so the right moment depends on
  the wave's position, one's health and gold, and the opponent's state.
- **Towers protect and punish.** Under one's own tower a champion is hard to reach; next to the enemy tower it is
  exposed. Sieges pay in plates and in the opponent's denied income, and bounce back if not followed up.
- **Outcomes depend on the state and the opponent.** No action is unconditionally best. The same action is right in
  one wave state and wrong in another, and right against one opponent and wrong against another.

## 3. Accepted simplifications and departures from League

The following are deliberate. They are not bugs to fix, though any of them may be revisited if a milestone shows it
is what breaks a relationship in section 2.

- **Geometry.** One lane of five champion cells (own tower, one cell, the lane, one cell, enemy tower) and nine minion
  cells, plus one off-lane Vision node attached to the lane cell, where a champion places a ward. "Next to the tower"
  and "just outside tower range" are the same cell.
- **Time.** Two-tick decision turns, waves every four ticks, a continuing game with discounting instead of a match
  clock. The recall time is a constant; the death timer grows with level in three steps (3, 4 and 5 turns).
- **Minions.** One HP pool per packet, an integer Lanchester law, caps on merged packets, no minion types, no
  cannon waves. Minions attack a champion only when it trades their champion or stands uncovered in their cell, and
  that damage falls with the champion's level.
- **Champions.** Identical champions distinguished only by a mechanical skill constant. No abilities, cooldowns,
  mana, ranges or crowd control: a trade is a probabilistic exchange whose damage scales with attack power; combat
  decisiveness is a matter of exchanges per turn and of the gap in power, not of a kit.
- **Progression.** One experience curve, one saturating investment curve bought automatically at base, no item
  choices, no runes. Investment is an abstract power stat, not a shop.
- **Economy.** Gold and XP per minion HP; plates and champion kills as the only other income; no jungle, no
  objectives, no roaming, no tower kills.
- **Information.** Perfect information about the state and no fog. Vision is one contestable ward at the shared
  Vision node, which lowers the gank rate for six turns.
- **Ganks.** An abstract threat whose probability depends on exposure and wave state, resolved as damage events.
- **Towers.** One target rule (crashed mass), a tower shot on divers and on traders next to the tower, plates by
  cumulative damage; no tower aggro mechanics.

What is *not* accepted is a hidden proxy: a constant that stands in for a consequence the rules are supposed to
produce. Where such constants exist for historical reasons (collection rates by posture or by tower cell are the
examples found in M4.2e-B), they are to be replaced by explicit consequences or justified as mechanical
inefficiencies in their own right.

## 4. Research and validation standards

- **Opt-in and reproducible.** Every mechanic is a `GameConfig` field whose default reproduces the previous preset bit
  for bit; historical presets (`GameConfig()`, `.v1()`, `.v2()`, `.champions()`, `.m4()`, `.m4_emergent()`,
  `.m4_natural()`, ...) are pinned by tests and are never edited to look as if they used newer rules.
- **Evidence before adoption.** A mechanic is adopted when experiments show that it changes which strategies are
  rational in the intended direction, in representative states, against opponents that respond. Competing
  explanations are tested before a change is made; a change that fixes a hand-picked scenario and nothing else is
  not adopted.
- **Deterministic first, sampled second, learned last.** Mechanical claims are checked on deterministic trajectories
  (no chance events); strategic claims on sampled windows with common random numbers and on solved policy games;
  learned play is the final test and is never used to tune mechanics.
- **Correctness guarantees are preserved by every change:** mirror symmetry, deterministic replay, exact zero-sum
  accounting (gold, XP, plates, kills), state invariants, legal-action masks that equal the engine's legal sets,
  a lossless bounded observation, and a bounded per-turn reward with a known evaluation tail.
- **Performance budget.** A resolver turn must stay in the hundreds of microseconds; the lane step in the tens.
- **Reports distinguish observed from hypothesised.** Observed results carry their method and budget; hypotheses and
  recommendations are labelled as such; every milestone ends with PASS, PASS WITH CAVEATS or BLOCKED.

## 5. How to use this document

When proposing a mechanic, answer four questions: which relationship in section 2 does it serve, which principle in
section 1 could it violate, which simplification in section 3 does it touch, and what experiment in the spirit of
section 4 would show it works. A mechanic that serves no relationship in section 2 is decoration; one that violates
principle 1 is a proxy; one that cannot be tested is a guess.

This document is not a specification. It does not forbid new actions, new state or new geometry, and it does not fix
any parameter. It fixes what the game is *for*.
