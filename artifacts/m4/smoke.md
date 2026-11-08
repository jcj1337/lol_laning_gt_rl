# Training smoke checks (M4.2) — integration only, NOT research results

Commit `1a40cf5069`. Shared self-play PPO (current-policy fraction 0.8), `PPOConfig.continuing` (gamma 0.99, lambda 0.95, segments of 128, games of 512 with seeded staggered first games, reward scale 0.1). Command: `python.exe C:\Users\light\OneDrive\Desktop\projects\lol_gt_laning\lanegame\experiments\m4\smoke.py --m2c-steps 300000 --m4-steps 50000`.

| run | transitions | env turns | transitions/s | finite losses | explained variance (last 10) | entropy first → last | V_hat(s0) (resources) |
|---|---|---|---|---|---|---|---|
| m2c | 303317 | 163840 | 7,922 | True | 0.51 | 2.19 → 0.27 | +0.30 |
| m4 | 50176 | 26624 | 6,299 | True | 0.69 | 2.19 → 1.44 | +0.41 |

Shove rate of the trained policy (self-play, present turns) per 64-turn window of 512-turn games:

| run | rates by window | chi-square p | stationary (p > 0.01) |
|---|---|---|---|
| m2c | 0.000, 0.000, 0.000, 0.001, 0.000, 0.000, 0.000, 0.001 | 0.332 | True |
| m4 | 0.047, 0.039, 0.059, 0.039, 0.044, 0.045, 0.058, 0.042 | 0.271 | True |

By symmetry the self-play value of s0 is 0; V_hat(s0) far from 0 relative to the return scale (order 1 / (1 - gamma) x 0.1-0.3 = 10-30 resources) would indicate a mis-scaled critic.
