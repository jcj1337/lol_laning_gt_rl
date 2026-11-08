# Throughput and memory (M4.2)

Commit `1a40cf5069`; single thread; uniformly random legal actions; 40000 turns per measurement. Command: `python.exe C:\Users\light\OneDrive\Desktop\projects\lol_gt_laning\lanegame\experiments\m4\benchmark.py --turns 40000`.

| configuration | env turns/s (no lane cache) | env turns/s (LRU 20 000) | resolver turns/s | observe() µs |
|---|---|---|---|---|
| M2 champions() | 6,933 | 10,283 | 11,732 | 5.9 |
| M2 continuing fixture (m2c) | 6,998 | 10,449 | 11,880 | 5.7 |
| M4 m4() | 6,444 | 8,818 | 11,336 | 11.3 |

Target ≥ 4 000 M4 environment turns/s (plan §6): **met**.

Memory along one 100 000-turn M4 trajectory (tracemalloc; LRU lane cache of 20 000 entries):

| turns | current MB | peak MB | lane cache entries |
|---|---|---|---|
| 10000 | 19.13 | 19.14 | 6832 |
| 25000 | 28.95 | 28.96 | 10360 |
| 50000 | 39.38 | 39.39 | 13990 |
| 75000 | 45.18 | 45.19 | 16066 |
| 100000 | 50.05 | 50.06 | 17827 |

Without the lane cache (no growth: the memory above is the bounded LRU cache filling, about 2.6 KB per entry, at most 20 000 entries):

| turns | current MB | peak MB |
|---|---|---|
| 10000 | 0.02 | 0.03 |
| 25000 | 0.02 | 0.03 |
| 50000 | 0.02 | 0.03 |
| 75000 | 0.02 | 0.03 |
| 100000 | 0.02 | 0.03 |

Profile of the M4 environment step (cProfile, top functions by own time):

```
6533657 function calls in 4.399 seconds

   Ordered by: internal time
   List reduced from 158 to 14 due to restriction <14>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    20000    0.691    0.000    2.275    0.000 C:\Users\light\OneDrive\Desktop\projects\lol_gt_laning\lanegame\champions_m4.py:259(resolve_turn_m4)
    20000    0.454    0.000    4.388    0.000 C:\Users\light\OneDrive\Desktop\projects\lol_gt_laning\lanegame\env\core.py:385(_step_m4)
    40002    0.388    0.000    0.923    0.000 C:\Users\light\OneDrive\Desktop\projects\lol_gt_laning\lanegame\env\observation_m4.py:109(observe_m4)
    19064    0.218    0.000    0.420    0.000 C:\Users\light\OneDrive\Desktop\projects\lol_gt_laning\lanegame\mechanics.py:224(_tick)
        1    0.183    0.183    4.617    4.617 C:\Users\light\OneDrive\Desktop\projects\lol_gt_laning\lanegame\experiments\m4\benchmark.py:33(env_rate)
     9532    0.157    0.000    0.704    0.000 C:\Users\light\OneDrive\Desktop\projects\lol_gt_laning\lanegame\mechanics.py:408(step_turn)
    20001    0.148    0.000    1.535    0.000 C:\Users\light\OneDrive\Desktop\projects\lol_gt_laning\lanegame\env\core.py:445(_observe)
    40002    0.129    0.000    0.253    0.000 C:\Users\light\OneDrive\Desktop\projects\lol_gt_laning\lanegame\champions.py:235(champion_legal_actions)
    80004    0.101    0.000    0.113    0.000 C:\Users\light\OneDrive\Desktop\projects\lol_gt_laning\lanegame\env\observation_m4.py:72(_lane_block)
    40002    0.097    0.000    0.097    0.000 {built-in method numpy.asarray}
    40002    0.070    0.000    1.022    0.000 C:\Users\light\OneDrive\Desktop\projects\lol_gt_laning\lanegame\env\observation.py:109(observe)
    32874    0.063    0.000    0.104    0.000 C:\Users\light\OneDrive\Desktop\projects\lol_gt_laning\lanegame\champions.py:260(champion_gank_probability)
    40002    0.056    0.000    0.333    0.000 C:\Users\light\OneDrive\Desktop\projects\lol_gt_laning\lanegame\env\observation.py:164(legal_set)
    80004    0.055    0.000    0.061    0.000 C:\Users\light\OneDrive\Desktop\projects\lol_gt_laning\lanegame\env\observation_m4.py:93(_tower_block)
```
