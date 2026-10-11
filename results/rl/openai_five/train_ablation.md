# Training runs `ablation`

Wall clock 1.2 min (10 processes, one torch thread each).

| run | agent transitions | environment turns | updates | train time (s) | historical share of transitions | final entropy | final explained variance | max approx KL | snapshots |
|---|---|---|---|---|---|---|---|---|---|
| sp50_s0 | 307,680 | 200,192 | 391 | 65 | 0.301 | 0.47 | 0.74 | 0.0067 | 12 |
| sp50_s1 | 307,584 | 199,680 | 390 | 64 | 0.298 | 0.26 | 0.79 | 0.0059 | 12 |
| sp50_s2 | 307,808 | 201,216 | 393 | 66 | 0.307 | 0.56 | 0.66 | 0.0063 | 12 |
