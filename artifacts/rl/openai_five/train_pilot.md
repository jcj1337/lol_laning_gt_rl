# Training runs `pilot`

Wall clock 0.3 min (10 processes, one torch thread each).

| run | agent transitions | environment turns | updates | train time (s) | historical share of transitions | final entropy | final explained variance | max approx KL | snapshots |
|---|---|---|---|---|---|---|---|---|---|
| m3b_s0 | 51,200 | 51,200 | 50 | 15 | - | 1.23 | 0.40 | 0.0025 | 2 |
| sp100_s0 | 51,200 | 25,600 | 50 | 8 | 0.000 | 0.92 | 0.50 | 0.0050 | 2 |
| sp80_s0 | 51,520 | 27,136 | 53 | 9 | 0.053 | 0.97 | 0.66 | 0.0050 | 2 |
