# M4.5a movement correction: demonstrations and paired comparison (generated)

`python -m lanegame.experiments.m45a movement`; design and discussion: `docs/m4/m4_5a_unified_mechanics_report.md` §10. r1 = `m45_unified_r1()` (champions at their end cells for the whole turn, positional aggro only on a champion that did not move); r2 = `m45_unified()` (champion_movement_rule "tick"); "end cell" = r1 without its stationary exemption (the original false positive).

## Scenarios (one turn, level 2, k = 10; fire per tick for champion A / B: pos = positional, ret = retaliation, · = none)

| | scenario | actions | end cell | r1 | r2 | r2 path A ; B | mirror |
|---|---|---|---|---|---|---|---|
| A1 | Hold follows its winning wave (the brief's example): A and its 16-HP front at 0, B's 2-HP front at A's +1, which dies on tick 0 | LH-H v LH-H | A pos / ·; B · / · | A · / ·; B · / · | A · / ·; B · / · | [0, 1] ; [-2, -2] | ok |
| A2 | Hold follows its winning wave from behind (A and its 16-HP front at -1, B's 2-HP front at 0) | LH-H v LH-H | A pos / ·; B · / · | A · / ·; B · / · | A · / ·; B · / · | [-1, 0] ; [-2, -2] | ok |
| A3 | Safe follows its winning wave (A and its 16-HP front at -2, B's 2-HP front at A's -1) | LH-S v LH-H | A pos / ·; B · / · | A · / ·; B · / · | A · / ·; B · / · | [-2, -1] ; [-2, -2] | ok |
| A4 | Yield (stand-off) in the same lane: it stays one cell behind B's starting front | LH-Y v LH-H | A · / ·; B · / · | A · / ·; B · / · | A · / ·; B · / · | [-2, -2] ; [-2, -2] | ok |
| B1 | Deliberate advance: A at 0 (its front at -1) Pressures through B's front at A's +1, which waits while A's crashed mass is under B's tower | LH-H+P v LH-H | A · / ·; B · / · | A · / ·; B · / · | A pos / ·; B · / · | [1, 2] ; [-1, -1] | ok |
| B2 | The brief's B with walking waves: A at 0 (its front at -1) Trades toward B at its tower while B's front walks from A's +1 to 0: they swap cells within tick 0 (no shared cell at a tick boundary) | LH-T v LH-T | A · / ·; B · / · | A · / ·; B · / · | A · / ·; B · / · | [1, 2] ; [-1, -1] | ok |
| C1 | Trade chase into B's 16-HP wave (B stands behind it), connected, missed | LH-T v LH-H | A pos / pos; B · / · | A · / ·; B · / · | A pos / pos; B · / · | [-1, -1] ; [-1, 0] | ok |
| C2 | The same chase, landed: positional aggro, then retaliation | LH-T v LH-H (hit A) | A pos / ret; B · / · | A · / ret; B · / · | A pos / ret; B · / · | [-1, -1] ; [-1, 0] | ok |
| C3 | Trade chase into B's wave; B evades out of contact (the Trade does not connect) | LH-T v LH-S | A pos / ·; B · / · | A · / ·; B · / · | A pos / ·; B · / · | [0, 0] ; [-2, -2] | ok |
| D1 | A Holds at B's tower cell far ahead of its wave and walks back: in B's oncoming front's cell after tick 0, covered by its own wave at the lane cell after tick 1 | LH-H v LH-H | A · / ·; B · / · | A · / ·; B · / · | A pos / ·; B · / · | [1, 0] ; [-1, 0] | ok |
| D2 | Cover on the same tick: A stands at 0 and both fronts walk into its cell on tick 0 | LH-T v LH-S | A · / ·; B · / · | A · / ·; B · / · | A · / ·; B · / · | [0, 0] ; [-2, -2] | ok |
| D3 | No cover: as D2 with A's front far behind (at -3); B's front walks into A's cell on tick 0 | LH-T v LH-S | A · / ·; B · / · | A · / ·; B · / · | A pos / ·; B · / · | [0, 0] ; [-2, -2] | ok |
| W1 | Ward from +1 walks through B's front at the lane cell (A's front at -1) | WARD v LH-H | A · / ·; B · / · | A · / ·; B · / · | A pos / ·; B · / · | [0, 99] ; [-1, 0] | ok |
| S1 | Standing still in B's wave ahead of the own front (unchanged by the correction) | LH-T v LH-S | A pos / ·; B · / · | A pos / ·; B · / · | A pos / ·; B · / · | [0, 0] ; [-2, -2] | ok |
| N1 | Adjacent to B's waiting front, never in its cell | LH-T v LH-T | A · / ·; B · / · | A · / ·; B · / · | A · / ·; B · / · | [0, 0] ; [-1, -1] | ok |
| R1 | B dives A's Recall through A's 16-HP front at A's -1 and lands: positional aggro on tick 0, retaliation from the end cell on tick 1 | LH-R v LH-T (hit B) | A · / ·; B · / ret | A · / ·; B · / ret | A · / ·; B pos / ret | [-2, -2] ; [1, 2] | ok |

## Uniform random legal play (40 games x 300 turns, r2 trajectories)

Per champion-turn of each victim movement class (paired: the same state, actions and chance under each rule).

| class | champion-turns | moved | positional: end cell / r1 / r2 | retaliation: r1 / r2 | minion cHP: r1 / r2 | r2 positional ticks on a moved champion | r2 follower hits: genuine / probe mismatch |
|---|---|---|---|---|---|---|---|
| follow | 13772 | 0.55 | 0.0239 / 0.0025 / 0.0302 | 0.0000 / 0.0000 | 0.05 / 0.70 | 378 of 416 | 416 / 0 |
| chase | 4276 | 0.63 | 0.2432 / 0.0475 / 0.2823 | 0.1635 / 0.1658 | 5.08 / 11.21 | 988 of 1207 | — |
| evade | 1383 | 0.60 | 0.0058 / 0.0000 / 0.0174 | 0.0000 / 0.0000 | 0.00 / 0.48 | 24 of 24 | — |
| ward | 756 | 0.94 | 0.0000 / 0.0000 / 0.1138 | 0.0000 / 0.0000 | 0.00 / 2.87 | 86 of 86 | — |
| objective | 347 | 0.72 | 0.0346 / 0.0144 / 0.0288 | 0.0115 / 0.0115 | 0.50 / 0.71 | 6 of 10 | — |
| recall | 138 | 0.00 | 0.0580 / 0.0580 / 0.0725 | 0.0000 / 0.0000 | 1.51 / 1.83 | 0 of 10 | — |

All classes: positional ticks per champion-turn 0.0582 (end cell) / 0.0104 (r1) / 0.0730 (r2); retaliation 0.0293 / 0.0297; deaths per champion-turn 0.0189 (r1) / 0.0193 (r2).

## Scripted pairings (8 seeds x 200 turns, r2 trajectories)

| pairing | positional ticks / champion-turn: end cell / r1 / r2 | retaliation r1 / r2 | minion cHP r1 / r2 | deaths r1 / r2 | r2 follower hits: genuine / probe mismatch |
|---|---|---|---|---|---|
| farm4 v farm4 [s0, equal] | 0.0088 / 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.00 / 0.00 | 0.0006 / 0.0006 | 0 / 0 |
| ward_favorable4 v farm4 [s0, equal] | 0.0156 / 0.0000 / 0.0128 | 0.0000 / 0.0000 | 0.00 / 0.28 | 0.0003 / 0.0003 | 20 / 0 |
| ward_always4 v bully4 [neutral, equal] | 0.1041 / 0.0666 / 0.0925 | 0.0712 / 0.0712 | 4.97 / 5.78 | 0.0078 / 0.0081 | 36 / 2 |
| bully4 v farm4 [s0, equal] | 0.1322 / 0.0934 / 0.1241 | 0.0669 / 0.0669 | 6.11 / 7.00 | 0.0078 / 0.0088 | 28 / 2 |
| bully4 v step_back4 [neutral, equal] | 0.0856 / 0.0694 / 0.0612 | 0.0031 / 0.0031 | 2.14 / 1.70 | 0.0063 / 0.0059 | 8 / 0 |
| freeze_bully4 v walk_up_shove4 [freeze, large] | 0.0838 / 0.0178 / 0.1028 | 0.0416 / 0.0416 | 1.03 / 2.42 | 0.0169 / 0.0169 | 80 / 0 |
| ward_freeze4 v walk_up_shove4 [freeze, large] | 0.0200 / 0.0000 / 0.0472 | 0.0000 / 0.0000 | 0.00 / 1.10 | 0.0016 / 0.0016 | 75 / 0 |

## Throughput (uniform play, one process, best of 3)

| preset | turns / s |
|---|---|
| `m45_unified_r1` | 3,649 |
| `m45_unified` | 3,541 |

r2 / r1 = 0.971

Elapsed: 25.0 s.
