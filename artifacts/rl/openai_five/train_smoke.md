# Training runs `smoke`

Wall clock 0.1 min (10 processes, one torch thread each).

| run | agent transitions | environment turns | updates | train time (s) | historical share of transitions | final entropy | final explained variance | max approx KL | snapshots |
|---|---|---|---|---|---|---|---|---|---|
| m3b_s0 | 10,240 | 10,240 | 10 | 3 | - | 2.14 | 0.39 | 0.0006 | 0 |
| sp100_s0 | 10,240 | 5,120 | 10 | 2 | 0.000 | 2.16 | 0.46 | 0.0008 | 0 |
| sp80_s0 | 10,240 | 5,120 | 10 | 2 | 0.000 | 2.16 | 0.46 | 0.0008 | 0 |
