# Experiment E: timing and reachability (siege model)

## Reachable events from the standard start (K = 24; 36891 decision nodes, 13130 distinct states; 67 s)

| event | reachable transitions | first turn | witness replays |
|---|---|---|---|
| crash_small | 34016 | 6 | True |
| crash_medium | 11668 | 11 | True |
| crash_large | 2264 | 15 | True |
| crash_onto_siege | 14032 | 9 | True |
| freeze_held | 8280 | 6 | True |
| freeze_broken | 5820 | 6 | True |
| bounce | 18380 | 13 | True |
| siege_persists | 174248 | 8 | True |
| safe_disengages | 390216 | 1 | True |
| tower_farm | 403744 | 7 | True |
| farm_contest_vs_safe | 882868 | 1 | True |
| minions_deleted | 0 | None | None |
| pressure_applied | 129216 | 7 | True |
| plate_claimed | 113040 | 7 | True |
| pressure_no_plates | 16176 | 9 | True |
| thinning | 61840 | 8 | True |
| pressure_no_target | 67376 | 7 | True |
| vision_established | 129216 | 7 | True |
| vision_refreshed | 93920 | 9 | True |
| vision_expired | 96016 | 13 | True |
| pressure_under_vision | 93920 | 9 | True |
| siege_disengaged | 129216 | 7 | True |
| crash_under_vision | 32540 | 10 | True |

Witnesses (A/B joint actions per turn; the last turn triggers the event):

- **pressure_applied** (turn 7): `SP-C/SP-C SH-C/SH-C SH-C/SP-C SH-C/SP-C SP-C/SP-C SH-C/SP-C SH-C/SP-C SP-C+P/SP-C`
  - lane at the start of the event turn: `     .      .     a4      .      .      .    A*1      .      .`
- **plate_claimed** (turn 7): `SP-C/SP-C SH-C/SH-C SH-C/SP-C SH-C/SP-C SP-C/SP-C SH-C/SP-C SH-C/SP-C SP-C+P/SP-C`
  - lane at the start of the event turn: `     .      .     a4      .      .      .    A*1      .      .`
- **pressure_no_plates** (turn 9): `SP-C/SP-C SH-C/SH-C SP-C/SH-C SP-C/SH-C SP-C/SP-C SP-C/SH-C F-C/SH-C SP-C/SH-C SH-C/SH-C+P SP-C/SP-C+P`
  - lane at the start of the event turn: `     .     a1  b1B*1      .      .      .     b4      .      .`
- **thinning** (turn 8): `SP-C/SP-C SH-C/SH-C SP-C/SH-C SP-C/SH-C SP-C/SP-C SH-C/SH-C F-C/SH-C SP-C/SH-C SP-C/SP-C+P`
  - lane at the start of the event turn: `    a4      .    B*1      .     b4      .      .      .     b4`
- **pressure_no_target** (turn 7): `SP-C/SP-C SH-C/SH-C SH-C/SP-C SH-C/SP-C SP-C/SP-C SH-C/SP-C SH-C/SP-C SP-C+P/SP-C`
  - lane at the start of the event turn: `     .      .     a4      .      .      .    A*1      .      .`
- **vision_established** (turn 7): `SP-C/SP-C SH-C/SH-C SH-C/SP-C SH-C/SP-C SP-C/SP-C SH-C/SP-C SH-C/SP-C SP-C+V/SP-C`
  - lane at the start of the event turn: `     .      .     a4      .      .      .    A*1      .      .`
- **vision_refreshed** (turn 9): `SP-C/SP-C SH-C/SH-C SP-C/SH-C SP-C/SH-C SP-C/SP-C SP-C/SH-C F-C/SH-C SP-C/SH-C SH-C/SH-C+V SP-C/SP-C+V`
  - lane at the start of the event turn: `     .     a2  b1B*1      .      .      .     b4      .      .`
- **vision_expired** (turn 13): `SP-C/SP-C SH-C/SH-C SP-C/SH-C SP-C/SH-C SP-C/SP-C SP-C/SH-C SP-C/SH-C SP-C/SP-C+V SP-C/SP-C SH-C/SH-C F-C/SH-C SP-C/SH-C SH-C/SH-C+P SP-C/SP-C`
  - lane at the start of the event turn: `     .     a1     b1      .      .      .     b4      .      .`
- **pressure_under_vision** (turn 9): `SP-C/SP-C SH-C/SH-C SP-C/SH-C SP-C/SH-C SP-C/SP-C SP-C/SH-C F-C/SH-C SP-C/SH-C SH-C/SH-C+V SP-C/SP-C+P`
  - lane at the start of the event turn: `     .     a2  b1B*1      .      .      .     b4      .      .`
- **siege_disengaged** (turn 7): `SP-C/SP-C SH-C/SH-C SH-C/SP-C SH-C/SP-C SP-C/SP-C SH-C/SP-C SH-C/SP-C SP-S/SP-C`
  - lane at the start of the event turn: `     .      .     a4      .      .      .    A*1      .      .`
- **crash_under_vision** (turn 10): `SP-C/SP-C SH-C/SH-C SP-C/SH-C SP-C/SH-C SP-C/SP-C SP-C/SH-C SP-C/SH-C SP-C/SP-C+V SP-C/SP-C SP-C/SH-C SP-C/SH-C`
  - lane at the start of the event turn: `    a4      .      .     b6      .      .      .      .     b4`

## Exhaustive property checks (K = 24, 894272 reachable transitions)

| property | violations |
|---|---|
| vision_retroactive | 0 |
| vision_timer_wrong | 0 |
| refresh_stacked | 0 |
| plates_over_pending | 0 |
| plates_without_pressure | 0 |
| thinning_without_target | 0 |
| thinning_over_limit | 0 |
| thinning_without_pressure | 0 |
| hp_not_conserved | 0 |
| pressure_without_siege | 0 |
| plates_pending_without_siege | 0 |

Counts: vision_set 129216, refreshes 93920, expiries 96016, pressure_turns 129216, plates 113040, thinned 61840, pressure_under_vision 93920

## The Vision -> Shove -> Crash -> Pressure strategy

Breadth-first search from s_0 over (turn, state, history flag) at K = 24. 'crash >= h' requires the crash after the vision to have at least h HP (1 = any remnant, 4 = a full wave, 8 = medium, 12 = large). 'turns after vision' = turn of the protected pressure minus the turn the vision was established; the strategy works only if this is <= the vision duration.

| vision | tower HP/tick | crash >= | found | vision turn | crash turn | pressure turn | turns after vision | search nodes | s |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 1 | 1 | True | 8 | 10 | 11 | 3 | 1039 | 0.5 |
| 4 | 1 | 4 | True | 8 | 11 | 12 | 4 | 1459 | 0.9 |
| 4 | 1 | 8 | True | 14 | 17 | 18 | 4 | 8606 | 6.1 |
| 4 | 1 | 12 | False | -1 | -1 | -1 | -1 | 27549 | 21.3 |
| 4 | 2 | 1 | True | 8 | 10 | 11 | 3 | 847 | 0.5 |
| 4 | 2 | 4 | True | 8 | 11 | 12 | 4 | 1111 | 0.6 |
| 4 | 2 | 8 | False | -1 | -1 | -1 | -1 | 9081 | 7.2 |
| 4 | 2 | 12 | False | -1 | -1 | -1 | -1 | 9081 | 6.6 |
| 4 | 3 | 1 | True | 8 | 10 | 11 | 3 | 765 | 0.5 |
| 4 | 3 | 4 | True | 12 | 15 | 16 | 4 | 2155 | 1.3 |
| 4 | 3 | 8 | False | -1 | -1 | -1 | -1 | 5593 | 3.7 |
| 4 | 3 | 12 | False | -1 | -1 | -1 | -1 | 5593 | 3.7 |
| 6 | 1 | 1 | True | 8 | 10 | 11 | 3 | 1039 | 0.5 |
| 6 | 1 | 4 | True | 8 | 11 | 12 | 4 | 1543 | 0.8 |
| 6 | 1 | 8 | True | 14 | 17 | 18 | 4 | 11522 | 7.9 |
| 6 | 1 | 12 | False | -1 | -1 | -1 | -1 | 36891 | 27.4 |
| 6 | 2 | 1 | True | 8 | 10 | 11 | 3 | 847 | 0.5 |
| 6 | 2 | 4 | True | 8 | 11 | 12 | 4 | 1195 | 0.8 |
| 6 | 2 | 8 | False | -1 | -1 | -1 | -1 | 13425 | 8.7 |
| 6 | 2 | 12 | False | -1 | -1 | -1 | -1 | 13425 | 8.6 |
| 6 | 3 | 1 | True | 8 | 10 | 11 | 3 | 765 | 0.4 |
| 6 | 3 | 4 | True | 12 | 15 | 16 | 4 | 3105 | 1.9 |
| 6 | 3 | 8 | False | -1 | -1 | -1 | -1 | 8713 | 5.5 |
| 6 | 3 | 12 | False | -1 | -1 | -1 | -1 | 8713 | 6 |
| 8 | 1 | 1 | True | 8 | 10 | 11 | 3 | 1039 | 0.5 |
| 8 | 1 | 4 | True | 8 | 11 | 12 | 4 | 1543 | 0.8 |
| 8 | 1 | 8 | True | 14 | 17 | 18 | 4 | 14328 | 10.7 |
| 8 | 1 | 12 | True | 14 | 20 | 21 | 7 | 31733 | 22.5 |
| 8 | 2 | 1 | True | 8 | 10 | 11 | 3 | 847 | 0.4 |
| 8 | 2 | 4 | True | 8 | 11 | 12 | 4 | 1195 | 0.7 |
| 8 | 2 | 8 | False | -1 | -1 | -1 | -1 | 17947 | 11.4 |
| 8 | 2 | 12 | False | -1 | -1 | -1 | -1 | 17947 | 11.4 |
| 8 | 3 | 1 | True | 8 | 10 | 11 | 3 | 765 | 0.4 |
| 8 | 3 | 4 | True | 8 | 14 | 15 | 7 | 3015 | 1.7 |
| 8 | 3 | 8 | False | -1 | -1 | -1 | -1 | 12083 | 7.5 |
| 8 | 3 | 12 | False | -1 | -1 | -1 | -1 | 12083 | 7.9 |

### Trace: vision 6, tower 1, crash >= 1

plan: `SP-C/SP-C SP-C/SP-C SP-C/SP-C SH-C/SP-C SP-C/SP-C SP-C/SP-C SH-C/SP-C SH-C/SP-C SP-C+V/SP-C SH-C/SP-C SH-C/SP-C SP-C+P/SP-C`

| k | A | B | A siege mass | objective | vision | plates (avail -> after, +claimed) | B front before -> after | thinned | A crash HP | A exposure | p gank (unprotected) | lane after |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | SP-C | SP-C | 0->0 | (no siege) | 0->0 | 0->0 (+0) | (-4, 4)->(-2, 4) | 0 | 0 | -2 | 0.170 (0.170) | `     .      .     a4      .      .      .     b4      .      .` |
| 1 | SP-C | SP-C | 0->0 | (no siege) | 0->0 | 0->0 (+0) | (-2, 4)->(0, 3) | 0 | 0 | +0 | 0.220 (0.220) | `    a4      .      .      .   a3b3      .      .      .     b4` |
| 2 | SP-C | SP-C | 0->0 | (no siege) | 0->0 | 0->0 (+0) | (0, 3)->(0, 1) | 0 | 0 | +0 | 0.220 (0.220) | `     .      .     a4      .   a1b1      .     b4      .      .` |
| 3 | SH-C | SP-C | 0->0 | (no siege) | 0->0 | 0->0 (+0) | (0, 1)->(0, 1) | 0 | 0 | +0 | 0.250 (0.250) | `    a4      .      .      .   a3b1      .      .      .     b4` |
| 4 | SP-C | SP-C | 0->0 | (no siege) | 0->0 | 0->0 (+0) | (0, 1)->(-2, 4) | 0 | 0 | +0 | 0.220 (0.220) | `     .      .     a4      .     a2      .     b4      .      .` |
| 5 | SP-C | SP-C | 0->0 | (no siege) | 0->0 | 0->0 (+0) | (-2, 4)->(-1, 2) | 0 | 0 | +0 | 0.220 (0.220) | `    a4      .      .      .     a4     b2      .      .     b4` |
| 6 | SH-C | SP-C | 0->0 | (no siege) | 0->0 | 0->0 (+0) | (-1, 2)->(-2, 2) | 0 | 0 | +1 | 0.275 (0.275) | `     .      .     a4      .      .     a2     b2      .      .` |
| 7 | SH-C | SP-C | 0->1 | (no siege) | 0->0 | 0->1 (+0) | (-2, 2)->(-4, 4) | 0 | 1 | +2 | 0.300 (0.300) | `    a4      .      .      .     a4      .    A*1      .     b4` |
| 8 | SP-C+V | SP-C | 1->0 | Vision | 0->6 | 1->0 (+0) | (-4, 4)->(-2, 4) | 0 | 0 | +2 | 0.270 (0.270) | `     .      .     a4      .     a4      .     b4      .      .` |
| 9 | SH-C | SP-C | 0->0 | (no siege) | 6->5 | 0->0 (+0) | (-2, 4)->(-4, 4) | 0 | 0 | +1 | 0.069 (0.275) | `    a4      .      .      .      .     a6      .      .     b4` |
| 10 | SH-C | SP-C | 0->2 | (no siege) | 5->4 | 0->1 (+0) | (-4, 4)->(0, 0) | 0 | 2 | +2 | 0.070 (0.280) | `     .      .     a4      .      .      .    A*2      .      .` |
| 11 | SP-C+P | SP-C | 2->0 | Pressure | 4->3 | 1->0 (+1) | (0, 0)->(-4, 4) | 0 | 0 | +2 | 0.068 (0.270) | `    a4      .      .      .     a4      .      .      .     b4` |

### Trace: vision 6, tower 1, crash >= 4

plan: `SP-C/SP-C SP-C/SP-C SP-C/SP-C SH-C/SP-C SP-C/SP-C SH-C/SP-C SH-C/F-C SH-C/SP-C SP-C+V/SP-C SP-C/SP-C SH-C/SP-C SH-C/SP-C SP-C+P/SP-C`

| k | A | B | A siege mass | objective | vision | plates (avail -> after, +claimed) | B front before -> after | thinned | A crash HP | A exposure | p gank (unprotected) | lane after |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | SP-C | SP-C | 0->0 | (no siege) | 0->0 | 0->0 (+0) | (-4, 4)->(-2, 4) | 0 | 0 | -2 | 0.170 (0.170) | `     .      .     a4      .      .      .     b4      .      .` |
| 1 | SP-C | SP-C | 0->0 | (no siege) | 0->0 | 0->0 (+0) | (-2, 4)->(0, 3) | 0 | 0 | +0 | 0.220 (0.220) | `    a4      .      .      .   a3b3      .      .      .     b4` |
| 2 | SP-C | SP-C | 0->0 | (no siege) | 0->0 | 0->0 (+0) | (0, 3)->(0, 1) | 0 | 0 | +0 | 0.220 (0.220) | `     .      .     a4      .   a1b1      .     b4      .      .` |
| 3 | SH-C | SP-C | 0->0 | (no siege) | 0->0 | 0->0 (+0) | (0, 1)->(0, 1) | 0 | 0 | +0 | 0.250 (0.250) | `    a4      .      .      .   a3b1      .      .      .     b4` |
| 4 | SP-C | SP-C | 0->0 | (no siege) | 0->0 | 0->0 (+0) | (0, 1)->(-2, 4) | 0 | 0 | +0 | 0.220 (0.220) | `     .      .     a4      .     a2      .     b4      .      .` |
| 5 | SH-C | SP-C | 0->0 | (no siege) | 0->0 | 0->0 (+0) | (-2, 4)->(-4, 4) | 0 | 0 | +1 | 0.275 (0.275) | `    a4      .      .      .      .     a4      .      .     b4` |
| 6 | SH-C | F-C | 0->0 | (no siege) | 0->0 | 0->0 (+0) | (-4, 4)->(-2, 2) | 0 | 0 | +1 | 0.275 (0.275) | `     .      .     a4      .      .     a3     b2      .      .` |
| 7 | SH-C | SP-C | 0->2 | (no siege) | 0->0 | 0->1 (+0) | (-2, 2)->(-4, 4) | 0 | 2 | +2 | 0.300 (0.300) | `    a4      .      .      .     a4      .    A*2      .     b4` |
| 8 | SP-C+V | SP-C | 2->0 | Vision | 0->6 | 1->0 (+0) | (-4, 4)->(-3, 4) | 0 | 0 | +2 | 0.270 (0.270) | `     .      .     a4      .     a4      .      .     b4      .` |
| 9 | SP-C | SP-C | 0->0 | (no siege) | 6->5 | 0->0 (+0) | (-3, 4)->(-1, 2) | 0 | 0 | +0 | 0.055 (0.220) | `    a4      .      .      .     a7     b2      .      .     b4` |
| 10 | SH-C | SP-C | 0->0 | (no siege) | 5->4 | 0->0 (+0) | (-1, 2)->(-2, 2) | 0 | 0 | +1 | 0.064 (0.255) | `     .      .     a4      .      .     a5     b2      .      .` |
| 11 | SH-C | SP-C | 0->4 | (no siege) | 4->3 | 0->1 (+0) | (-2, 2)->(-4, 4) | 0 | 4 | +2 | 0.070 (0.280) | `    a4      .      .      .     a4      .    A*4      .     b4` |
| 12 | SP-C+P | SP-C | 4->2 | Pressure | 3->2 | 1->0 (+1) | (-4, 4)->(-3, 3) | 1 | 0 | +2 | 0.068 (0.270) | `     .      .     a4      .     a4      .    A*2     b3      .` |

### Trace: vision 6, tower 1, crash >= 8

plan: `SP-C/SP-C SP-C/SP-C SH-C/SP-C SH-C/SP-C SP-C/SP-C SH-C/SP-C SH-C/F-C SH-C/F-C SP-C/F-C SP-C/SP-C SP-C/SP-C SH-C/SP-C SP-S/SP-C SP-C+P/SP-C SP-C+V/SP-C SH-C/SP-C SH-C/SP-C SH-C/SP-C SP-C+P/SP-C`

| k | A | B | A siege mass | objective | vision | plates (avail -> after, +claimed) | B front before -> after | thinned | A crash HP | A exposure | p gank (unprotected) | lane after |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | SP-C | SP-C | 0->0 | (no siege) | 0->0 | 0->0 (+0) | (-4, 4)->(-2, 4) | 0 | 0 | -2 | 0.170 (0.170) | `     .      .     a4      .      .      .     b4      .      .` |
| 1 | SP-C | SP-C | 0->0 | (no siege) | 0->0 | 0->0 (+0) | (-2, 4)->(0, 3) | 0 | 0 | +0 | 0.220 (0.220) | `    a4      .      .      .   a3b3      .      .      .     b4` |
| 2 | SH-C | SP-C | 0->0 | (no siege) | 0->0 | 0->0 (+0) | (0, 3)->(-2, 4) | 0 | 0 | +0 | 0.250 (0.250) | `     .      .     a4      .     a1      .     b4      .      .` |
| 3 | SH-C | SP-C | 0->0 | (no siege) | 0->0 | 0->0 (+0) | (-2, 4)->(-4, 4) | 0 | 0 | +1 | 0.275 (0.275) | `    a4      .      .      .     a3      .      .      .     b4` |
| 4 | SP-C | SP-C | 0->0 | (no siege) | 0->0 | 0->0 (+0) | (-4, 4)->(-2, 4) | 0 | 0 | +0 | 0.220 (0.220) | `     .      .     a4      .     a3      .     b4      .      .` |
| 5 | SH-C | SP-C | 0->0 | (no siege) | 0->0 | 0->0 (+0) | (-2, 4)->(-4, 4) | 0 | 0 | +1 | 0.275 (0.275) | `    a4      .      .      .      .     a5      .      .     b4` |
| 6 | SH-C | F-C | 0->0 | (no siege) | 0->0 | 0->0 (+0) | (-4, 4)->(-2, 2) | 0 | 0 | +1 | 0.255 (0.255) | `     .      .     a4      .      .     a4     b2      .      .` |
| 7 | SH-C | F-C | 0->0 | (no siege) | 0->0 | 0->0 (+0) | (-2, 2)->(-4, 4) | 0 | 0 | +1 | 0.275 (0.275) | `    a4      .      .      .      .     a7      .      .     b4` |
| 8 | SP-C | F-C | 0->0 | (no siege) | 0->0 | 0->0 (+0) | (-4, 4)->(-2, 3) | 0 | 0 | +1 | 0.225 (0.225) | `     .      .     a4      .      .     a6     b3      .      .` |
| 9 | SP-C | SP-C | 0->0 | (no siege) | 0->0 | 0->0 (+0) | (-2, 3)->(-4, 4) | 0 | 0 | +1 | 0.225 (0.225) | `    a4      .      .      .      .     a8      .      .     b4` |
| 10 | SP-C | SP-C | 0->0 | (no siege) | 0->0 | 0->0 (+0) | (-4, 4)->(-2, 2) | 0 | 0 | +1 | 0.225 (0.225) | `     .      .     a4      .      .     a7     b2      .      .` |
| 11 | SH-C | SP-C | 0->6 | (no siege) | 0->0 | 0->2 (+0) | (-2, 2)->(-4, 4) | 0 | 6 | +2 | 0.280 (0.280) | `    a4      .      .      .     a4      .    A*6      .     b4` |
| 12 | SP-S | SP-C | 6->4 | - | 0->0 | 2->2 (+0) | (-4, 4)->(-3, 4) | 0 | 0 | -1 | 0.175 (0.175) | `     .      .     a4      .     a4      .    A*4     b4      .` |
| 13 | SP-C+P | SP-C | 4->2 | Pressure | 0->0 | 2->1 (+1) | (-3, 4)->(-3, 7) | 1 | 0 | +2 | 0.270 (0.270) | `    a4      .      .      .     a8      .    A*2     b7      .` |
| 14 | SP-C+V | SP-C | 2->0 | Vision | 0->6 | 1->0 (+0) | (-3, 7)->(-3, 7) | 0 | 0 | +2 | 0.270 (0.270) | `     .      .     a4      .     a8      .      .     b7      .` |
| 15 | SH-C | SP-C | 0->0 | (no siege) | 6->5 | 0->0 (+0) | (-3, 7)->(-2, 1) | 0 | 0 | +1 | 0.069 (0.275) | `    a4      .      .      .      .    a10     b1      .     b4` |
| 16 | SH-C | SP-C | 0->0 | (no siege) | 5->4 | 0->0 (+0) | (-2, 1)->(0, 0) | 0 | 0 | +1 | 0.059 (0.235) | `     .      .     a4      .      .     a8      .      .      .` |
| 17 | SH-C | SP-C | 0->7 | (no siege) | 4->3 | 0->2 (+0) | (0, 0)->(-4, 4) | 0 | 8 | +2 | 0.070 (0.280) | `    a4      .      .      .     a4      .    A*7      .     b4` |
| 18 | SP-C+P | SP-C | 7->5 | Pressure | 3->2 | 2->1 (+1) | (-4, 4)->(-3, 3) | 1 | 0 | +2 | 0.068 (0.270) | `     .      .     a4      .     a4      .    A*5     b3      .` |

## Bounce after a 12-HP crash (constructed 'bounce' start; B plays Slow Push + Contest)

released HP = HP of B's leading wave on the turn the crashed mass is gone; total = that wave plus any reinforcement queued directly behind it (a capped 12-HP wave can have a 4-HP wave queued). With Slow Push A's own waves wait at the lane, so only thinning acts on B's wave; with Shove A's arriving 4-HP waves also fight it under the tower (and die there).

| tower HP/tick | thin_damage | A's waves | A during the siege | siege turns | thinned | plates | released HP | released total HP |
|---|---|---|---|---|---|---|---|---|
| 1 | 0 | Slow Push (A's waves wait) | disengage (Safe) | 6 | 0 | 0 | 12 | 16 |
| 1 | 0 | Slow Push (A's waves wait) | vision every turn | 6 | 0 | 0 | 12 | 16 |
| 1 | 0 | Slow Push (A's waves wait) | pressure every turn | 6 | 0 | 3 | 12 | 16 |
| 1 | 0 | Shove (A's waves walk in) | disengage (Safe) | 6 | 0 | 0 | 12 | 12 |
| 1 | 0 | Shove (A's waves walk in) | vision every turn | 6 | 0 | 0 | 8 | 8 |
| 1 | 0 | Shove (A's waves walk in) | pressure every turn | 6 | 0 | 3 | 8 | 8 |
| 1 | 1 | Slow Push (A's waves wait) | disengage (Safe) | 6 | 0 | 0 | 12 | 16 |
| 1 | 1 | Slow Push (A's waves wait) | vision every turn | 6 | 0 | 0 | 12 | 16 |
| 1 | 1 | Slow Push (A's waves wait) | pressure every turn | 6 | 6 | 3 | 10 | 10 |
| 1 | 1 | Shove (A's waves walk in) | disengage (Safe) | 6 | 0 | 0 | 12 | 12 |
| 1 | 1 | Shove (A's waves walk in) | vision every turn | 6 | 0 | 0 | 8 | 8 |
| 1 | 1 | Shove (A's waves walk in) | pressure every turn | 6 | 5 | 3 | 4 | 4 |
| 1 | 2 | Slow Push (A's waves wait) | disengage (Safe) | 6 | 0 | 0 | 12 | 16 |
| 1 | 2 | Slow Push (A's waves wait) | vision every turn | 6 | 0 | 0 | 12 | 16 |
| 1 | 2 | Slow Push (A's waves wait) | pressure every turn | 6 | 12 | 3 | 4 | 4 |
| 1 | 2 | Shove (A's waves walk in) | disengage (Safe) | 6 | 0 | 0 | 12 | 12 |
| 1 | 2 | Shove (A's waves walk in) | vision every turn | 6 | 0 | 0 | 8 | 8 |
| 1 | 2 | Shove (A's waves walk in) | pressure every turn | 10 | 12 | 7 | 4 | 4 |
| 2 | 0 | Slow Push (A's waves wait) | disengage (Safe) | 3 | 0 | 0 | 8 | 8 |
| 2 | 0 | Slow Push (A's waves wait) | vision every turn | 3 | 0 | 0 | 8 | 8 |
| 2 | 0 | Slow Push (A's waves wait) | pressure every turn | 3 | 0 | 3 | 8 | 8 |
| 2 | 0 | Shove (A's waves walk in) | disengage (Safe) | 3 | 0 | 0 | 7 | 7 |
| 2 | 0 | Shove (A's waves walk in) | vision every turn | 3 | 0 | 0 | 6 | 6 |
| 2 | 0 | Shove (A's waves walk in) | pressure every turn | 3 | 0 | 3 | 6 | 6 |
| 2 | 1 | Slow Push (A's waves wait) | disengage (Safe) | 3 | 0 | 0 | 8 | 8 |
| 2 | 1 | Slow Push (A's waves wait) | vision every turn | 3 | 0 | 0 | 8 | 8 |
| 2 | 1 | Slow Push (A's waves wait) | pressure every turn | 3 | 3 | 3 | 5 | 5 |
| 2 | 1 | Shove (A's waves walk in) | disengage (Safe) | 3 | 0 | 0 | 7 | 7 |
| 2 | 1 | Shove (A's waves walk in) | vision every turn | 3 | 0 | 0 | 6 | 6 |
| 2 | 1 | Shove (A's waves walk in) | pressure every turn | 3 | 3 | 3 | 3 | 3 |
| 2 | 2 | Slow Push (A's waves wait) | disengage (Safe) | 3 | 0 | 0 | 8 | 8 |
| 2 | 2 | Slow Push (A's waves wait) | vision every turn | 3 | 0 | 0 | 8 | 8 |
| 2 | 2 | Slow Push (A's waves wait) | pressure every turn | 3 | 6 | 3 | 2 | 2 |
| 2 | 2 | Shove (A's waves walk in) | disengage (Safe) | 3 | 0 | 0 | 7 | 7 |
| 2 | 2 | Shove (A's waves walk in) | vision every turn | 3 | 0 | 0 | 6 | 6 |
| 2 | 2 | Shove (A's waves walk in) | pressure every turn | 4 | 6 | 4 | 4 | 4 |
| 3 | 0 | Slow Push (A's waves wait) | disengage (Safe) | 2 | 0 | 0 | 8 | 8 |
| 3 | 0 | Slow Push (A's waves wait) | vision every turn | 2 | 0 | 0 | 8 | 8 |
| 3 | 0 | Slow Push (A's waves wait) | pressure every turn | 2 | 0 | 2 | 8 | 8 |
| 3 | 0 | Shove (A's waves walk in) | disengage (Safe) | 2 | 0 | 0 | 8 | 8 |
| 3 | 0 | Shove (A's waves walk in) | vision every turn | 2 | 0 | 0 | 8 | 8 |
| 3 | 0 | Shove (A's waves walk in) | pressure every turn | 2 | 0 | 2 | 8 | 8 |
| 3 | 1 | Slow Push (A's waves wait) | disengage (Safe) | 2 | 0 | 0 | 8 | 8 |
| 3 | 1 | Slow Push (A's waves wait) | vision every turn | 2 | 0 | 0 | 8 | 8 |
| 3 | 1 | Slow Push (A's waves wait) | pressure every turn | 2 | 2 | 2 | 6 | 6 |
| 3 | 1 | Shove (A's waves walk in) | disengage (Safe) | 2 | 0 | 0 | 8 | 8 |
| 3 | 1 | Shove (A's waves walk in) | vision every turn | 2 | 0 | 0 | 8 | 8 |
| 3 | 1 | Shove (A's waves walk in) | pressure every turn | 2 | 2 | 2 | 6 | 6 |
| 3 | 2 | Slow Push (A's waves wait) | disengage (Safe) | 2 | 0 | 0 | 8 | 8 |
| 3 | 2 | Slow Push (A's waves wait) | vision every turn | 2 | 0 | 0 | 8 | 8 |
| 3 | 2 | Slow Push (A's waves wait) | pressure every turn | 2 | 4 | 2 | 4 | 4 |
| 3 | 2 | Shove (A's waves walk in) | disengage (Safe) | 2 | 0 | 0 | 8 | 8 |
| 3 | 2 | Shove (A's waves walk in) | vision every turn | 2 | 0 | 0 | 8 | 8 |
| 3 | 2 | Shove (A's waves walk in) | pressure every turn | 2 | 4 | 2 | 4 | 4 |
