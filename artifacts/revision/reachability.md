# Experiment D: reachability from the standard start (K = 32)

Revised model: 14849 reachable decision nodes, 2480 distinct states. Original model: 905 nodes, 71 distinct states.

## Events (number of reachable transitions on which the event occurs)

| event | revised | original (v1) | first turn (revised) | witness replays |
|---|---|---|---|---|
| crash_small | 9372 | 578 | 6 | True |
| crash_medium | 5592 | 400 | 11 | True |
| crash_large | 1840 | 60 | 15 | True |
| crash_onto_siege | 3576 | 0 | 17 | True |
| freeze_held | 3392 | 394 | 6 | True |
| freeze_broken | 2632 | 276 | 6 | True |
| bounce | 8176 | 0 | 13 | True |
| siege_persists | 75416 | 0 | 8 | True |
| safe_disengages | 157754 | 0 | 1 | True |
| tower_farm | 125236 | 0 | 7 | True |
| farm_contest_vs_safe | 304876 | 3575 | 1 | True |
| minions_deleted | 0 | 126 | None | None |

## Witness action sequences from s_0 (revised model; A/B joint actions per turn, last turn triggers the event; crash choices: option indices taken after each turn, 0 = resources)

- **crash_small** (turn 6): `SP-C/SP-C SH-C/SH-C SP-C/SH-C SP-C/SH-C SP-C/SP-C SP-C/SH-C SP-C/SH-C`
  - state at the start of the event turn: `    a4      .      .     b5      .      .      .      .     b4`
- **crash_medium** (turn 11): `SP-C/SP-C SP-C/SH-C SP-C/SH-C SP-C/SH-C SP-C/SP-C SP-C/SP-C SP-C/SP-C SP-C/SP-C SP-C/SP-C SP-C/SP-C SH-C/SP-C SP-C/SH-C`
  - state at the start of the event turn: `     .      .     a2     b6      .      .     b4      .      .`
- **crash_large** (turn 15): `SP-C/SP-C SP-C/SH-C SP-C/SP-C SP-C/SP-C SP-C/SH-C SP-C/SH-C SP-C/SP-C SP-C/SP-C SP-C/SP-C SP-C/SP-C SP-C/SP-C SP-C/SP-C SP-C/SP-C SP-C/SP-C SP-C/SP-C SP-C/SH-C`
  - state at the start of the event turn: `     .      .     a2    b10      .      .     b4      .      .`
- **crash_onto_siege** (turn 17): `SP-C/SP-C SP-C/SH-C SP-C/SP-C SP-C/SP-C SP-C/SH-C SP-C/SH-C SP-C/SP-C SP-C/SP-C SP-C/SP-C SP-C/SP-C SP-C/SP-C SP-C/SP-C SP-C/SP-C SP-C/SP-C SP-C/SP-C SP-C/SP-C F-C/SH-C SP-C/SH-C`
  - state at the start of the event turn: `     .      .    B*6     b4      .      .     b4      .      .`
- **freeze_held** (turn 6): `SP-C/SP-C SH-C/SH-C SP-C/SH-C SP-C/SH-C SP-C/SP-C SH-C/SH-C F-C/SH-C`
  - state at the start of the event turn: `    a4      .      .     b3      .      .      .      .     b4`
- **freeze_broken** (turn 6): `SP-C/SP-C SP-C/SH-C SP-C/SH-C SP-C/SH-C SP-C/SP-C SP-C/SH-C F-C/SH-C`
  - state at the start of the event turn: `    a4      .      .     b6      .      .      .      .     b4`
- **bounce** (turn 13): `SP-C/SP-C SH-C/SH-C SP-C/SH-C SH-C/SH-C SP-C/SP-C SP-C/SH-C F-C/SH-C SP-C/SH-C SP-C/SP-C SP-C/SP-C SP-C/SH-C SP-C/SH-C SP-C/SP-C SP-C/SP-C`
  - state at the start of the event turn: `     .     a4    B*2      .     b4      .     b4      .      .`
- **siege_persists** (turn 8): `SP-C/SP-C SH-C/SH-C SP-C/SH-C SP-C/SH-C SP-C/SP-C SP-C/SH-C F-C/SH-C SP-C/SH-C SP-C/SP-C`
  - state at the start of the event turn: `    a4      .    B*3      .     b4      .      .      .     b4`
- **safe_disengages** (turn 1): `SP-C/SP-C SP-C/SP-S`
  - state at the start of the event turn: `     .      .     a4      .      .      .     b4      .      .`
- **tower_farm** (turn 7): `SP-C/SP-C SP-C/SH-C SP-C/SH-C SP-C/SH-C SP-C/SP-C SP-C/SH-S SP-C/SH-C SP-C/SH-C`
  - state at the start of the event turn: `     .      .   a2b3      .      .      .     b4      .      .`
- **farm_contest_vs_safe** (turn 1): `SP-C/SP-C SP-C/SP-C`
  - state at the start of the event turn: `     .      .     a4      .      .      .     b4      .      .`
- **minions_deleted**: not reachable

## Behavioural scenarios: constructed vs. reachable starts

A constructed start is a hand-built state used to isolate a mechanic; the event it illustrates is checked separately for reachability from s_0.

| scenario | start state | illustrated event | event reachable from s_0 |
|---|---|---|---|
| even_river_fight | reachable | - | - |
| shove_wins_river | reachable | - | - |
| slowpush_stacks | reachable | - | - |
| shove_contests_slowpush | reachable | - | - |
| unstacked_wave_waits | constructed | - | - |
| stacked_wave_creeps | constructed | - | - |
| freeze_holds | constructed | freeze_held | True |
| freeze_broken | constructed | freeze_broken | True |
| large_crash_persists | constructed | siege_persists | True |
| small_crash_clears_quickly | constructed | crash_small | True |
| tower_farm_contested | constructed | tower_farm | True |
| tower_farm_safe | constructed | tower_farm | True |
| bounce | constructed | bounce | True |
| safe_lets_wave_go | constructed | safe_disengages | True |
| freeze_denial | constructed | farm_contest_vs_safe | True |

## Realized collection efficiencies (revised; count of reachable (turn, side, location) kill records)

| efficiency | count |
|---|---|
| 0.0 | 101284 |
| 0.14 | 20636 |
| 0.2 | 20636 |
| 0.28 | 17718 |
| 0.35 | 51768 |
| 0.4 | 18172 |
| 0.42 | 22228 |
| 0.5 | 51826 |
| 0.6 | 21814 |
| 0.63 | 60070 |
| 0.68 | 45496 |
| 0.765 | 45496 |
| 0.8 | 44880 |
| 0.9 | 105020 |

## Timing-dependent strategies

Solved at K = 128 and compared at least 48 turns before the horizon. Values there still drift with the number of turns remaining by at most 0.125 per turn (measured on the same state in the same phase at k and k + 4; this is the persistent lane-priority gain discussed in the report), so a value difference between two phases larger than a few tenths reflects timing, not the horizon. Lane states reachable in more than one dynamics phase (k mod 4): 1972; of these, 243 have a different equilibrium support for A in different phases (same lane, different spawn/creep timing).

```
      a4      .      .    b11      .      .      .      .     b4
```
- phase 0 (k = 16): V* = 14.923, A plays {'SP-C': 1.0}
- phase 2 (k = 14): V* = -8.631, A plays {'SP-S': 1.0}

```
      a4      .      .      .      .    a11      .      .     b4
```
- phase 0 (k = 16): V* = -14.923, A plays {'SP-C': 1.0}
- phase 2 (k = 14): V* = 8.631, A plays {'SP-S': 1.0}

```
      a4      .      .      .      .    a10      .      .     b4
```
- phase 0 (k = 16): V* = -14.668, A plays {'SP-C': 1.0}
- phase 2 (k = 14): V* = 8.670, A plays {'SP-S': 1.0}

```
      a4      .      .    b10      .      .      .      .     b4
```
- phase 0 (k = 16): V* = 14.668, A plays {'SP-C': 1.0}
- phase 2 (k = 14): V* = -8.670, A plays {'SH-S': 1.0}

```
       .     a8    B*2    b12      .      .      .      .     b4
```
- phase 0 (k = 24): V* = 14.984, A plays {'SP-C': 1.0}
- phase 2 (k = 26): V* = -6.040, A plays {'SP-S': 1.0}

```
      a4      .      .      .      .    a12    A*2     b8      .
```
- phase 0 (k = 24): V* = -14.984, A plays {'SH-C': 1.0}
- phase 2 (k = 26): V* = 6.040, A plays {'SP-S': 1.0}

## Resource vs. vision choices

A has 401 crash-choice nodes; vision has positive probability at 45. Median |value(resources) - value(vision)| against B's equilibrium: 0.168; share of nodes where the two options are within 0.25 resources: 0.53.

- k = 13: `    a4      .      .      .     a4      .    A*4      .     b4` options [[1.0, False], [0.0, True]] -> P(vision) = 1.00, option values [-3.172, -3.078]
- k = 16: `     .      .     a4      .      .      .    A*9      .      .` options [[3.0, False], [2.0, True]] -> P(vision) = 1.00, option values [-1.899, -1.876]
- k = 16: `     .      .     a4      .      .      .   A*10      .      .` options [[3.0, False], [2.0, True]] -> P(vision) = 1.00, option values [-2.088, -1.9]
- k = 17: `    a4      .      .      .     a4      .    A*8      .     b4` options [[1.0, False], [0.0, True]] -> P(vision) = 1.00, option values [-3.563, -3.063]
- k = 17: `    a4      .      .      .     a4      .    A*4      .     b4` options [[1.0, False], [0.0, True]] -> P(vision) = 1.00, option values [-2.681, -2.578]
- k = 17: `    a4      .      .      .     a4      .    A*9      .     b4` options [[3.0, False], [2.0, True]] -> P(vision) = 1.00, option values [-1.76, -1.26]
- k = 19: `    a4      .      .      .     a4      .    A*8      .     b4` options [[1.0, False], [0.0, True]] -> P(vision) = 1.00, option values [-4.282, -4.178]
- k = 19: `    a4      .      .      .     a4      .   A*10      .     b4` options [[3.0, False], [2.0, True]] -> P(vision) = 1.00, option values [-2.648, -2.399]
