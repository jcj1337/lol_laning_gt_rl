# Training runs `envturn`

Wall clock 1.8 min (10 processes, one torch thread each).

| run | agent transitions | environment turns | updates | train time (s) | historical share of transitions | final entropy | final explained variance | max approx KL | snapshots |
|---|---|---|---|---|---|---|---|---|---|
| sp100_s0 | 614,400 | 307,200 | 600 | 94 | 0.000 | 0.34 | 0.68 | 0.0068 | 24 |
| sp100_s1 | 614,400 | 307,200 | 600 | 95 | 0.000 | 0.53 | 0.74 | 0.0042 | 24 |
| sp100_s2 | 614,400 | 307,200 | 600 | 95 | 0.000 | 0.42 | 0.74 | 0.0047 | 24 |
| sp80_s0 | 553,376 | 305,664 | 597 | 101 | 0.105 | 0.41 | 0.73 | 0.0061 | 21 |
| sp80_s1 | 553,760 | 306,176 | 598 | 105 | 0.106 | 0.61 | 0.69 | 0.0048 | 21 |
| sp80_s2 | 553,440 | 306,688 | 599 | 104 | 0.108 | 0.34 | 0.68 | 0.0041 | 21 |
