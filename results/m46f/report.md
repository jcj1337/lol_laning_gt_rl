# M4.6f generated tables

Intervals: 95% t over the six seed pairs (5 df); **bold** excludes 0; (k+/k−) counts seeds by sign. Resources throughout. 'separate − shared' is the paired difference within a seed.


# Phase ext

## Scorecard (ext)

| statistic | separate − shared | class |
|---|---|---|
| V1 edge+plaus recovery ratio | **+0.24 ± 0.20** (6+/0−) | small (separate 0.45, shared 0.21, start 0.17) |
| V2 natural MAE (G500 states) | -0.14 ± 0.20 (2+/4−) | no regression |
| V3 large+gap ratio | **+0.30 ± 0.03** (6+/0−) | diagnostic |
| B1 head to head (W 1000) | +0.06 ± 0.23 (3+/3−) | no material difference |
| B1 powered, 320 episodes (amendment 1; not used for the outcome) | **+0.13 ± 0.09** (6+/0−) | small |
| B2 strength use (G150) | +0.01 ± 0.11 (3+/3−) | no material difference |
| vs start: separate - shared | +0.14 ± 0.25 (5+/1−) | no regression |
| vs suite: separate - shared | +0.09 ± 0.51 (3+/3−) | no regression |
| shared − start, edge+plaus ratio | +0.03 ± 0.15 (5+/1−) | not material |

**Outcome: 4: not resolved in this budget: V1 is not a benefit and the separate critic recovers the premium only partially (ratio < 0.5)**

## Training

| quantity | shared | separate | separate − shared |
|---|---|---|---|
| transitions_per_s | 2586.1879 | 2482.2003 | **-103.9876 ± 25.0868** (0+/6−) |
| elapsed_s | 3167.7204 | 3300.6776 | **+132.9572 ± 33.1591** (6+/0−) |
| ev_first10 | 0.4705 | 0.4302 | **-0.0403 ± 0.0387** (1+/5−) |
| ev_last100 | 0.4425 | 0.5192 | **+0.0768 ± 0.0251** (6+/0−) |
| value_loss_first10_res2 | 0.5665 | 0.6365 | +0.0700 ± 0.0839 (5+/1−) |
| value_loss_last100_res2 | 0.6415 | 0.5544 | -0.0871 ± 0.0976 (1+/5−) |
| entropy_last100 | 1.7307 | 1.7416 | +0.0108 ± 0.0330 (3+/3−) |
| kl_mean | 0.0037 | 0.0037 | +0.0000 ± 0.0000 (3+/3−) |
| clipfrac_mean | 0.0433 | 0.0435 | +0.0001 ± 0.0003 (4+/2−) |
| grad_norm_mean | 0.7567 | 0.7566 | -0.0001 ± 0.0057 (2+/4−) |
| return_std_last100 | 1.5120 | 1.5108 | -0.0012 ± 0.0945 (2+/4−) |
| gp_total_norm_mean | 0.8044 | 0.8118 | +0.0074 ± 0.0089 (5+/1−) |
| gp_actor_norm_mean | 0.8035 | 0.8106 | +0.0071 ± 0.0088 (5+/1−) |
| gp_critic_norm_mean | 0.0365 | 0.0390 | **+0.0025 ± 0.0023** (6+/0−) |
| gp_trunk_pol_mean | 0.6000 | 0.6077 | +0.0077 ± 0.0078 (6+/0−) |
| gp_trunk_val_mean | 0.0042 | 0.0000 | **-0.0042 ± 0.0005** (0+/6−) |
| gp_trunk_cos_mean | 0.0009 | 0.0000 | -0.0009 ± 0.0045 (3+/3−) |
| gp_ctrunk_val_mean | 0.0000 | 0.0066 | **+0.0066 ± 0.0008** (6+/0−) |
| gp_clip_shrink_mean | 0.9988 | 0.9985 | **-0.0003 ± 0.0002** (0+/6−) |
| gp_critic_share_of_total_sq | 0.0022 | 0.0026 | **+0.0004 ± 0.0003** (6+/0−) |

Explained variance in ten blocks of updates (mean over seeds):

| arm | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| shared | 0.432 | 0.436 | 0.425 | 0.416 | 0.425 | 0.433 | 0.422 | 0.434 | 0.437 | 0.435 |
| separate | 0.455 | 0.492 | 0.496 | 0.483 | 0.472 | 0.479 | 0.491 | 0.485 | 0.502 | 0.515 |

## Value: M4.6d calibration set (own-policy labels, K 32, H 300 / 500)

| readout | start | shared | separate | separate − shared | shared − start | separate − start |
|---|---|---|---|---|---|---|
| edge+plaus ratio | 0.17 | 0.21 | 0.45 | **+0.24 ± 0.20** (6+/0−) | +0.03 ± 0.15 (5+/1−) | **+0.28 ± 0.24** (5+/1−) |
| edge ratio | 0.01 | 0.02 | 0.31 | **+0.29 ± 0.12** (6+/0−) | +0.01 ± 0.02 (5+/1−) | **+0.30 ± 0.12** (6+/0−) |
| plaus ratio | 0.38 | 0.47 | 0.71 | **+0.23 ± 0.23** (5+/1−) | +0.09 ± 0.22 (5+/1−) | +0.32 ± 0.35 (5+/1−) |
| large ratio | 0.02 | 0.02 | 0.30 | **+0.27 ± 0.04** (6+/0−) | -0.00 ± 0.01 (3+/3−) | **+0.27 ± 0.04** (6+/0−) |
| gap ratio | 0.02 | 0.03 | 0.33 | **+0.31 ± 0.03** (6+/0−) | +0.00 ± 0.01 (4+/2−) | **+0.31 ± 0.03** (6+/0−) |
| large+gap ratio | 0.02 | 0.02 | 0.32 | **+0.30 ± 0.03** (6+/0−) | -0.00 ± 0.01 (4+/2−) | **+0.30 ± 0.04** (6+/0−) |
| edge+plaus predicted | 0.15 | 0.16 | 0.38 | **+0.22 ± 0.05** (6+/0−) | +0.01 ± 0.03 (4+/2−) | **+0.23 ± 0.05** (6+/0−) |
| edge+plaus realized | 0.96 | 0.99 | 0.94 | -0.05 ± 0.49 (3+/3−) | +0.03 ± 0.65 (2+/4−) | -0.02 ± 0.62 (3+/3−) |
| x1.76 predicted | 0.14 | 0.14 | 1.66 | **+1.52 ± 0.09** (6+/0−) | -0.00 ± 0.04 (3+/3−) | **+1.52 ± 0.08** (6+/0−) |
| x1.76 realized | 5.77 | 5.78 | 5.68 | -0.11 ± 0.20 (1+/5−) | +0.02 ± 0.44 (2+/4−) | -0.09 ± 0.43 (1+/5−) |
| bias freeze (A) | -1.99 | -1.95 | -1.36 | **+0.59 ± 0.37** (6+/0−) | +0.04 ± 0.22 (3+/3−) | **+0.63 ± 0.36** (5+/1−) |
| bias lane (A) | -1.66 | -1.67 | -1.02 | **+0.66 ± 0.28** (6+/0−) | -0.01 ± 0.29 (3+/3−) | **+0.65 ± 0.46** (6+/0−) |
| bias gap (A) | -4.64 | -4.70 | -3.08 | **+1.62 ± 0.43** (6+/0−) | -0.05 ± 0.41 (3+/3−) | **+1.57 ± 0.66** (6+/0−) |
| natural MAE | 0.42 | 0.52 | 0.37 | -0.14 ± 0.20 (2+/4−) | +0.10 ± 0.17 (5+/1−) | -0.04 ± 0.13 (2+/4−) |
| natural bias | 0.01 | 0.02 | 0.00 | -0.02 ± 0.03 (3+/3−) | +0.01 ± 0.03 (4+/2−) | -0.00 ± 0.04 (2+/4−) |
| natural slope | 1.01 | 0.95 | 0.97 | +0.02 ± 0.10 (4+/2−) | -0.05 ± 0.14 (2+/4−) | -0.03 ± 0.17 (2+/4−) |
| natural R2 | 0.77 | 0.71 | 0.82 | +0.10 ± 0.14 (4+/2−) | -0.05 ± 0.14 (2+/4−) | +0.05 ± 0.15 (4+/2−) |

## Value: natural self-play labels (G300, single rollouts, both seats)

| readout | start | shared | separate | separate − shared |
|---|---|---|---|---|
| mae | 2.225 | 2.338 | 2.094 | -0.244 ± 0.424 (1+/5−) |
| bias | 0.007 | 0.012 | -0.001 | -0.013 ± 0.026 (2+/4−) |
| slope | 0.960 | 0.936 | 1.044 | +0.109 ± 0.115 (6+/0−) |
| r2 | 0.107 | 0.097 | 0.166 | **+0.069 ± 0.047** (6+/0−) |
| corr | 0.325 | 0.310 | 0.407 | **+0.097 ± 0.066** (6+/0−) |

Policy drift on the start's natural states (decisions with a choice):

| quantity | value |
|---|---|
| kl_start_shared | **+0.2633 ± 0.0209** (6+/0−) |
| kl_start_separate | **+0.2597 ± 0.0210** (6+/0−) |
| tv_shared_separate | **+0.2724 ± 0.0059** (6+/0−) |
| kl separate - shared | -0.0036 ± 0.0242 (3+/3−) |

## Value representation: ridge probes (held-out R², M4.6e main split)

| probe | start | shared | separate | separate − shared |
|---|---|---|---|---|
| log_duel<-value_features | 0.175 | 0.168 | 0.644 | **+0.475 ± 0.040** (6+/0−) |
| log_duel<-actor_trunk | 0.175 | 0.168 | 0.161 | -0.007 ± 0.012 (2+/4−) |
| hp_own<-value_features | 0.917 | 0.897 | 0.920 | **+0.023 ± 0.021** (6+/0−) |
| hp_own<-actor_trunk | 0.917 | 0.897 | 0.912 | +0.015 ± 0.015 (6+/0−) |
| value_features_move_x1.76_over_natural_spread | 0.103 | 0.117 | 0.232 | **+0.116 ± 0.023** (6+/0−) |

Critic response to single-feature changes of seat A (resources):

| change | start | shared | separate |
|---|---|---|---|
| freeze L5|invested +8 | 0.03 | 0.03 | 0.44 |
| freeze L5|invested +32 | 0.09 | 0.08 | 1.17 |
| freeze L5|unspent gold +4 | -0.05 | -0.05 | 0.09 |
| freeze L5|level +1 | 0.01 | 0.00 | 0.05 |
| freeze L5|HP 60% | -0.29 | -0.29 | -0.26 |
| freeze L5|enemy HP 60% | 0.47 | 0.48 | 0.44 |
| neutral L8|invested +8 | 0.03 | 0.03 | 0.41 |
| neutral L8|invested +32 | 0.10 | 0.09 | 1.19 |
| neutral L8|unspent gold +4 | -0.10 | -0.09 | 0.09 |
| neutral L8|level +1 | 0.00 | 0.00 | 0.06 |
| neutral L8|HP 60% | -0.48 | -0.63 | -0.51 |
| neutral L8|enemy HP 60% | 0.49 | 0.42 | 0.49 |
| neutral L15|invested +8 | 0.02 | 0.01 | 0.22 |
| neutral L15|invested +32 | 0.05 | 0.04 | 0.70 |
| neutral L15|unspent gold +4 | -0.09 | -0.08 | 0.09 |
| neutral L15|level +1 | 0.00 | 0.00 | 0.07 |
| neutral L15|HP 60% | -0.49 | -0.62 | -0.60 |
| neutral L15|enemy HP 60% | 0.46 | 0.40 | 0.62 |

## Behaviour: head to head from s0

| comparison | estimate |
|---|---|
| B1 separate vs shared (W 1000) | +0.06 ± 0.23 (3+/3−) |
| vs start: shared | **+0.44 ± 0.15** (6+/0−) |
| vs start: separate | **+0.58 ± 0.24** (6+/0−) |
| vs start: separate - shared | +0.14 ± 0.25 (5+/1−) |
| vs other starts: shared | +0.08 ± 0.13 (4+/2−) |
| vs other starts: separate | **+0.42 ± 0.21** (6+/0−) |
| vs other starts: separate - shared | **+0.34 ± 0.29** (5+/1−) |
| vs suite: shared | **+4.78 ± 0.25** (6+/0−) |
| vs suite: separate | **+4.88 ± 0.39** (6+/0−) |
| vs suite: separate - shared | +0.09 ± 0.51 (3+/3−) |
| B1 powered: 256 new episodes (amendment 1) | **+0.15 ± 0.10** (6+/0−) |
| B1 powered: pooled 320 episodes (amendment 1) | **+0.13 ± 0.09** (6+/0−) |

Scripted core suite by reference (W 460):

| reference | shared | separate | separate − shared |
|---|---|---|---|
| g_farm | 2.79 | 2.85 | +0.06 ± 0.70 (2+/4−) |
| g_shove | 3.96 | 4.03 | +0.08 ± 0.52 (3+/3−) |
| g_bully | 8.55 | 8.66 | +0.10 ± 1.14 (3+/3−) |
| fz_guard_sh | 3.83 | 3.96 | +0.13 ± 0.79 (3+/3−) |

## Behaviour: strength use (B2 cells; separate − shared, mean over both opponents)

| readout | G50 | G150 |
|---|---|---|
| B2 | -0.01 ± 0.09 (3+/3−) | +0.01 ± 0.11 (3+/3−) |
| conversion | -0.01 ± 0.14 (2+/4−) | -0.01 ± 0.15 (2+/4−) |
| defence | -0.01 ± 0.20 (3+/3−) | +0.03 ± 0.24 (3+/3−) |
| equal cells | -0.10 ± 0.26 (1+/5−) | -0.03 ± 0.39 (3+/3−) |
| conversion - equal | +0.06 ± 0.18 (4+/2−) | -0.01 ± 0.25 (3+/3−) |
| defence - equal | +0.13 ± 0.41 (4+/2−) | +0.10 ± 0.53 (4+/2−) |
| large x1.76 (diagnostic) | -0.08 ± 0.43 (3+/3−) | -0.25 ± 0.47 (2+/4−) |

By cell kind (G150):

| kind | separate − shared |
|---|---|
| lane brk | +0.15 ± 0.39 (5+/1−) |
| lane edge | -0.01 ± 0.26 (2+/4−) |
| lane equal | +0.13 ± 0.36 (4+/2−) |
| lane hp | -0.03 ± 0.30 (3+/3−) |
| lane plaus | -0.05 ± 0.38 (3+/3−) |
| lane x1.76 | -0.25 ± 0.47 (2+/4−) |
| lead early A+16 | -0.12 ± 0.44 (2+/4−) |
| lead early A+8 | +0.14 ± 0.55 (4+/2−) |
| lead early B+16 | +0.19 ± 0.45 (5+/1−) |
| lead early B+8 | +0.09 ± 0.61 (2+/4−) |
| lead early equal | -0.05 ± 0.54 (3+/3−) |
| lead mid A+16 | +0.15 ± 0.21 (5+/1−) |
| lead mid A+8 | -0.11 ± 0.61 (2+/4−) |
| lead mid B+16 | +0.17 ± 0.38 (3+/3−) |
| lead mid B+8 | **-0.48 ± 0.31** (0+/6−) |
| lead mid equal | -0.22 ± 0.70 (2+/4−) |

Denial and exposure (ledger at 150 turns):

| quantity | shared level | separate − shared |
|---|---|---|
| conversion: opponent farm lost (unc_B) | 18.27 | -0.09 ± 0.39 (3+/3−) |
| conversion: opponent XP lost (xp_lost_B) | 46.68 | -0.59 ± 2.44 (3+/3−) |
| conversion: own earned (earned_A) | 62.02 | +0.09 ± 0.23 (4+/2−) |
| conversion: opponent deaths | 1.16 | -0.03 ± 0.09 (2+/4−) |
| defence: own farm lost (unc_A) | 18.26 | -0.16 ± 0.53 (3+/3−) |
| defence: own XP lost (xp_lost_A) | 45.58 | -0.61 ± 2.45 (2+/4−) |
| defence: own deaths | 1.19 | -0.02 ± 0.16 (2+/4−) |

## Behaviour: freezes (B3; G50)


freezer:

| readout | separate − shared | shared level |
|---|---|---|
| equal G50 | +0.03 ± 0.21 (3+/3−) |  |
| edge G50 | +0.09 ± 0.23 (4+/2−) |  |
| plaus G50 | +0.17 ± 0.38 (3+/3−) |  |
| hp G50 | -0.09 ± 0.34 (2+/4−) |  |
| brk G50 | -0.11 ± 0.16 (1+/5−) |  |
| x1.76 G50 | -0.18 ± 0.28 (1+/5−) |  |
| favourable (edge, plaus, hp) G50 | +0.05 ± 0.23 (2+/4−) |  |
| favourable: held_A | +0.24 ± 0.77 (4+/2−) | 6.39 |
| favourable: unc_B | -0.00 ± 0.10 (3+/3−) | 6.29 |
| favourable: xp_lost_B | -0.42 ± 0.71 (2+/4−) | 18.22 |
| favourable: deaths_B | -0.05 ± 0.07 (2+/4−) | 0.56 |
| favourable: deaths_A | +0.01 ± 0.07 (4+/2−) | 0.42 |

breaker:

| readout | separate − shared | shared level |
|---|---|---|
| equal G50 | +0.03 ± 0.22 (2+/4−) |  |
| edge G50 | +0.07 ± 0.26 (2+/4−) |  |
| plaus G50 | +0.09 ± 0.29 (4+/2−) |  |
| hp G50 | -0.00 ± 0.15 (2+/4−) |  |
| brk G50 | -0.08 ± 0.14 (2+/4−) |  |
| x1.76 G50 | +0.03 ± 0.29 (3+/3−) |  |
| favourable (edge, plaus, hp) G50 | +0.05 ± 0.20 (3+/3−) |  |
| favourable (freezer advantaged): unc_A | **-0.11 ± 0.07** (0+/6−) | 6.55 |
| favourable (freezer advantaged): xp_lost_A | -0.41 ± 0.59 (1+/5−) | 19.35 |
| favourable (freezer advantaged): deaths_A | +0.03 ± 0.05 (4+/2−) | 0.57 |
| favourable (freezer advantaged): frozen_A | -0.53 ± 0.62 (1+/5−) | 6.61 |

## Behaviour: weak side at Recall windows (B4)


weak:

| readout | separate − shared | shared level |
|---|---|---|
| equal G50 | +0.05 ± 0.18 (4+/2−) |  |
| edge G50 | +0.01 ± 0.18 (3+/3−) |  |
| x1.76 G50 | -0.06 ± 0.36 (3+/3−) |  |
| stronger opponent (edge, x1.76) G50 | -0.03 ± 0.24 (3+/3−) |  |
| equal G100 | -0.02 ± 0.17 (4+/2−) |  |
| edge G100 | +0.04 ± 0.21 (3+/3−) |  |
| x1.76 G100 | -0.02 ± 0.39 (3+/3−) |  |
| stronger opponent (edge, x1.76) G100 | +0.01 ± 0.23 (2+/4−) |  |
| deaths_A (100 turns) | +0.00 ± 0.23 (4+/2−) | 1.28 |
| recalls_A (100 turns) | -0.19 ± 0.33 (2+/4−) | 4.68 |
| n_safe_A (100 turns) | +0.88 ± 1.19 (5+/1−) | 15.80 |
| n_yield_A (100 turns) | -0.36 ± 1.84 (3+/3−) | 24.47 |
| away_A (100 turns) | -0.37 ± 0.52 (2+/4−) | 14.56 |
| deaths_B (100 turns) | -0.02 ± 0.05 (2+/4−) | 0.46 |
| earned_A (100 turns) | +0.14 ± 0.38 (3+/3−) | 37.96 |

strong:

| readout | separate − shared | shared level |
|---|---|---|
| equal G50 | +0.05 ± 0.10 (3+/3−) |  |
| edge G50 | -0.03 ± 0.25 (3+/3−) |  |
| x1.76 G50 | -0.23 ± 0.45 (2+/4−) |  |
| advantaged (edge, x1.76) G50 | -0.13 ± 0.30 (1+/5−) |  |
| equal G100 | **+0.10 ± 0.09** (5+/1−) |  |
| edge G100 | -0.07 ± 0.39 (3+/3−) |  |
| x1.76 G100 | -0.33 ± 0.47 (2+/4−) |  |
| advantaged (edge, x1.76) G100 | -0.20 ± 0.31 (2+/4−) |  |
| deaths_A (100 turns) | +0.01 ± 0.12 (4+/2−) | 0.48 |
| recalls_A (100 turns) | -0.23 ± 0.34 (1+/5−) | 4.75 |
| n_safe_A (100 turns) | **+1.31 ± 0.86** (5+/1−) | 14.23 |
| n_yield_A (100 turns) | -0.22 ± 1.59 (2+/4−) | 20.13 |
| away_A (100 turns) | -0.42 ± 0.43 (1+/5−) | 11.52 |
| deaths_B (100 turns) | -0.12 ± 0.16 (1+/5−) | 1.45 |
| earned_A (100 turns) | -0.05 ± 0.37 (3+/3−) | 43.55 |

first-turn probabilities, weak side vs a stronger opponent:

| actor | recall | Hold | Yield | Trade | Safe | LH | TN | SL | SH |
|---|---|---|---|---|---|---|---|---|---|
| start | 0.016 | 0.130 | 0.633 | 0.028 | 0.194 | 0.073 | 0.107 | 0.152 | 0.652 |
| shared | 0.011 | 0.103 | 0.609 | 0.027 | 0.250 | 0.072 | 0.133 | 0.123 | 0.661 |
| separate | 0.032 | 0.104 | 0.566 | 0.041 | 0.257 | 0.119 | 0.139 | 0.167 | 0.544 |

first-turn probabilities, freezer in favourable freeze cells:

| actor | recall | Hold | Yield | Trade | Safe | LH | TN | SL | SH |
|---|---|---|---|---|---|---|---|---|---|
| start | 0.000 | 0.531 | 0.326 | 0.137 | 0.006 | 0.108 | 0.122 | 0.126 | 0.644 |
| shared | 0.000 | 0.560 | 0.304 | 0.129 | 0.007 | 0.068 | 0.083 | 0.109 | 0.740 |
| separate | 0.000 | 0.501 | 0.319 | 0.174 | 0.006 | 0.097 | 0.094 | 0.092 | 0.717 |

## Behaviour: natural states of start:s (B5)

| readout | separate − shared |
|---|---|
| leader G50 | +0.07 ± 0.09 (5+/1−) |
| leader G150 | +0.08 ± 0.10 (5+/1−) |
| trailer G50 | +0.01 ± 0.16 (2+/4−) |
| trailer G150 | +0.09 ± 0.20 (3+/3−) |
| freezer G50 | -0.01 ± 0.23 (4+/2−) |
| breaker G50 | +0.02 ± 0.17 (3+/3−) |

## Exploratory (amendment 2): natural self-play economy, 16 games × 600 turns per network

| quantity | start | shared | separate | separate − shared | shared − start | separate − start |
|---|---|---|---|---|---|---|
| away_share | 0.112 | 0.118 | 0.113 | **-0.006 ± 0.005** (0+/6−) | +0.006 ± 0.012 (4+/2−) | +0.000 ± 0.010 (2+/4−) |
| collection_rate | 0.758 | 0.748 | 0.759 | +0.011 ± 0.023 (4+/2−) | -0.010 ± 0.017 (2+/4−) | +0.001 ± 0.013 (3+/3−) |
| crashes_per_100_turns | 9.477 | 9.739 | 10.014 | +0.275 ± 0.597 (4+/2−) | +0.261 ± 0.529 (5+/1−) | +0.536 ± 0.642 (5+/1−) |
| deaths_per_1000 | 6.085 | 6.432 | 5.842 | -0.590 ± 2.431 (2+/4−) | +0.347 ± 2.439 (3+/3−) | -0.243 ± 1.618 (3+/3−) |
| earned_per_100_turns | 39.498 | 39.088 | 39.517 | +0.430 ± 0.660 (4+/2−) | **-0.410 ± 0.394** (1+/5−) | +0.019 ± 0.441 (4+/2−) |
| freeze_turns_share | 0.053 | 0.055 | 0.053 | -0.002 ± 0.007 (3+/3−) | +0.002 ± 0.009 (3+/3−) | +0.000 ± 0.007 (3+/3−) |
| frozen_on_share | 0.053 | 0.055 | 0.053 | -0.002 ± 0.007 (3+/3−) | +0.002 ± 0.009 (3+/3−) | +0.000 ± 0.007 (3+/3−) |
| gold_at_recall | 8.058 | 7.578 | 7.933 | **+0.356 ± 0.062** (6+/0−) | -0.480 ± 0.562 (2+/4−) | -0.124 ± 0.556 (2+/4−) |
| gold_present_mean | 5.275 | 4.939 | 5.238 | **+0.300 ± 0.131** (6+/0−) | -0.336 ± 0.378 (1+/5−) | -0.036 ± 0.311 (3+/3−) |
| invested_150 | 53.905 | 54.285 | 54.773 | +0.488 ± 1.385 (4+/2−) | +0.381 ± 1.149 (4+/2−) | +0.868 ± 1.678 (4+/2−) |
| invested_300 | 114.129 | 113.306 | 114.552 | +1.246 ± 1.957 (5+/1−) | **-0.823 ± 0.638** (0+/6−) | +0.423 ± 1.801 (2+/4−) |
| invested_600 | 230.750 | 229.042 | 231.287 | +2.245 ± 4.297 (4+/2−) | -1.709 ± 2.452 (2+/4−) | +0.536 ± 2.492 (3+/3−) |
| lead_per_100_turns | 0.000 | 0.000 | 0.000 | +0.000 ± 0.000 (0+/0−) | +0.000 ± 0.000 (0+/0−) | +0.000 ± 0.000 (0+/0−) |
| level_end | 18.000 | 18.000 | 18.000 | +0.000 ± 0.000 (0+/0−) | +0.000 ± 0.000 (0+/0−) | +0.000 ± 0.000 (0+/0−) |
| pressure_turns_per_1000 | 20.122 | 19.748 | 21.311 | +1.563 ± 2.355 (4+/2−) | -0.373 ± 3.537 (2+/4−) | +1.189 ± 4.183 (4+/2−) |
| recalls_gold_lt2_share | 0.030 | 0.027 | 0.033 | +0.006 ± 0.013 (4+/2−) | -0.003 ± 0.007 (1+/5−) | +0.004 ± 0.013 (4+/2−) |
| recalls_per_1000 | 41.875 | 43.941 | 42.639 | -1.302 ± 3.901 (3+/3−) | +2.066 ± 2.366 (6+/0−) | +0.764 ± 2.726 (5+/1−) |
| trade_attempts_per_1000 | 227.786 | 242.622 | 229.757 | -12.865 ± 22.913 (2+/4−) | +14.835 ± 16.725 (6+/0−) | +1.970 ± 15.796 (4+/2−) |
| trades_connected_per_1000 | 134.653 | 150.877 | 139.210 | -11.667 ± 17.948 (2+/4−) | +16.224 ± 19.456 (4+/2−) | +4.557 ± 14.884 (4+/2−) |
| wards_per_1000 | 13.637 | 13.767 | 15.608 | +1.840 ± 6.750 (4+/2−) | +0.130 ± 4.314 (3+/3−) | +1.970 ± 6.071 (3+/3−) |
| wards_placed_per_1000 | 13.099 | 13.099 | 15.165 | +2.066 ± 6.461 (4+/2−) | +0.000 ± 4.203 (3+/3−) | +2.066 ± 5.911 (4+/2−) |
| xp_missed_per_100_turns | 29.182 | 31.666 | 29.173 | -2.493 ± 5.513 (2+/4−) | +2.484 ± 6.033 (4+/2−) | -0.010 ± 4.769 (3+/3−) |

# Phase main

## Scorecard (main)

| statistic | separate − shared | class |
|---|---|---|
| V1 edge+plaus recovery ratio | **+0.12 ± 0.07** (6+/0−) | small (separate 0.33, shared 0.21, start 0.17) |
| V2 natural MAE (G500 states) | -0.01 ± 0.20 (2+/4−) | no regression |
| V3 large+gap ratio | **+0.20 ± 0.05** (6+/0−) | diagnostic |
| B1 head to head (W 1000) | +0.10 ± 0.27 (4+/2−) | inconclusive |
| B1 powered, 320 episodes (amendment 1; not used for the outcome) | +0.07 ± 0.15 (4+/2−) | no material difference |
| B2 strength use (G150) | **+0.14 ± 0.09** (6+/0−) | small |
| vs start: separate - shared | -0.02 ± 0.44 (2+/4−) | no regression |
| vs suite: separate - shared | +0.00 ± 0.59 (4+/2−) | no regression |
| shared − start, edge+plaus ratio | +0.04 ± 0.08 (4+/2−) | not material |

**Outcome: 4: not resolved in this budget: V1 is not a benefit and the separate critic recovers the premium only partially (ratio < 0.5)**

## Training

| quantity | shared | separate | separate − shared |
|---|---|---|---|
| transitions_per_s | 2599.1852 | 2485.1056 | **-114.0796 ± 20.8731** (0+/6−) |
| elapsed_s | 1580.3706 | 1652.3478 | **+71.9773 ± 8.2563** (6+/0−) |
| ev_first10 | 0.4505 | 0.4494 | -0.0011 ± 0.0432 (4+/2−) |
| ev_last100 | 0.4459 | 0.4984 | **+0.0525 ± 0.0478** (6+/0−) |
| value_loss_first10_res2 | 0.6102 | 0.5964 | -0.0138 ± 0.1029 (1+/5−) |
| value_loss_last100_res2 | 0.6377 | 0.6089 | -0.0287 ± 0.1197 (2+/4−) |
| entropy_last100 | 1.7186 | 1.7205 | +0.0019 ± 0.0498 (3+/3−) |
| kl_mean | 0.0038 | 0.0038 | **+0.0000 ± 0.0000** (6+/0−) |
| clipfrac_mean | 0.0435 | 0.0441 | **+0.0006 ± 0.0004** (6+/0−) |
| grad_norm_mean | 0.7572 | 0.7530 | **-0.0042 ± 0.0031** (0+/6−) |
| return_std_last100 | 1.5069 | 1.5506 | +0.0436 ± 0.0841 (5+/1−) |
| gp_total_norm_mean | 0.8081 | 0.8030 | -0.0051 ± 0.0105 (2+/4−) |
| gp_actor_norm_mean | 0.8071 | 0.8017 | -0.0054 ± 0.0107 (2+/4−) |
| gp_critic_norm_mean | 0.0371 | 0.0395 | +0.0025 ± 0.0031 (5+/1−) |
| gp_trunk_pol_mean | 0.6037 | 0.5990 | -0.0047 ± 0.0092 (2+/4−) |
| gp_trunk_val_mean | 0.0047 | 0.0000 | **-0.0047 ± 0.0008** (0+/6−) |
| gp_trunk_cos_mean | 0.0031 | 0.0000 | -0.0031 ± 0.0080 (2+/4−) |
| gp_ctrunk_val_mean | 0.0000 | 0.0064 | **+0.0064 ± 0.0008** (6+/0−) |
| gp_clip_shrink_mean | 0.9988 | 0.9984 | **-0.0004 ± 0.0003** (1+/5−) |
| gp_critic_share_of_total_sq | 0.0022 | 0.0028 | **+0.0005 ± 0.0005** (5+/1−) |

Explained variance in ten blocks of updates (mean over seeds):

| arm | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| shared | 0.435 | 0.420 | 0.428 | 0.430 | 0.428 | 0.430 | 0.438 | 0.430 | 0.453 | 0.446 |
| separate | 0.445 | 0.460 | 0.475 | 0.470 | 0.503 | 0.477 | 0.481 | 0.481 | 0.498 | 0.498 |

## Value: M4.6d calibration set (own-policy labels, K 32, H 300 / 500)

| readout | start | shared | separate | separate − shared | shared − start | separate − start |
|---|---|---|---|---|---|---|
| edge+plaus ratio | 0.17 | 0.21 | 0.33 | **+0.12 ± 0.07** (6+/0−) | +0.04 ± 0.08 (4+/2−) | **+0.16 ± 0.06** (6+/0−) |
| edge ratio | 0.01 | 0.01 | 0.18 | **+0.17 ± 0.06** (6+/0−) | -0.00 ± 0.01 (5+/1−) | **+0.17 ± 0.06** (6+/0−) |
| plaus ratio | 0.38 | 0.72 | 0.65 | -0.06 ± 0.58 (3+/3−) | +0.33 ± 0.43 (4+/2−) | +0.27 ± 0.29 (6+/0−) |
| large ratio | 0.02 | 0.03 | 0.21 | **+0.18 ± 0.05** (6+/0−) | +0.00 ± 0.01 (4+/2−) | **+0.18 ± 0.05** (6+/0−) |
| gap ratio | 0.02 | 0.03 | 0.23 | **+0.20 ± 0.05** (6+/0−) | +0.00 ± 0.01 (3+/3−) | **+0.20 ± 0.05** (6+/0−) |
| large+gap ratio | 0.02 | 0.03 | 0.22 | **+0.20 ± 0.05** (6+/0−) | +0.00 ± 0.01 (3+/3−) | **+0.20 ± 0.05** (6+/0−) |
| edge+plaus predicted | 0.15 | 0.19 | 0.30 | **+0.11 ± 0.04** (6+/0−) | **+0.04 ± 0.03** (6+/0−) | **+0.15 ± 0.04** (6+/0−) |
| edge+plaus realized | 0.96 | 0.89 | 0.90 | +0.01 ± 0.15 (4+/2−) | -0.07 ± 0.33 (2+/4−) | -0.06 ± 0.22 (2+/4−) |
| x1.76 predicted | 0.14 | 0.16 | 1.14 | **+0.98 ± 0.25** (6+/0−) | +0.01 ± 0.04 (4+/2−) | **+1.00 ± 0.22** (6+/0−) |
| x1.76 realized | 5.77 | 5.93 | 5.59 | -0.34 ± 0.72 (2+/4−) | +0.16 ± 0.32 (3+/3−) | -0.18 ± 0.70 (3+/3−) |
| bias freeze (A) | -1.99 | -2.04 | -1.51 | **+0.53 ± 0.24** (6+/0−) | -0.05 ± 0.30 (2+/4−) | **+0.48 ± 0.25** (6+/0−) |
| bias lane (A) | -1.66 | -1.71 | -1.28 | **+0.43 ± 0.40** (5+/1−) | -0.05 ± 0.50 (3+/3−) | **+0.39 ± 0.35** (6+/0−) |
| bias gap (A) | -4.64 | -4.70 | -3.58 | **+1.12 ± 0.93** (6+/0−) | -0.06 ± 0.90 (4+/2−) | **+1.06 ± 0.66** (6+/0−) |
| natural MAE | 0.42 | 0.50 | 0.49 | -0.01 ± 0.20 (2+/4−) | +0.08 ± 0.13 (5+/1−) | +0.08 ± 0.09 (4+/2−) |
| natural bias | 0.01 | -0.02 | 0.01 | +0.03 ± 0.04 (4+/2−) | -0.02 ± 0.06 (1+/5−) | +0.01 ± 0.05 (3+/3−) |
| natural slope | 1.01 | 1.02 | 1.02 | -0.01 ± 0.20 (3+/3−) | +0.02 ± 0.18 (3+/3−) | +0.01 ± 0.26 (3+/3−) |
| natural R2 | 0.77 | 0.73 | 0.76 | +0.03 ± 0.14 (3+/3−) | -0.04 ± 0.09 (3+/3−) | -0.01 ± 0.15 (2+/4−) |

## Value: natural self-play labels (G300, single rollouts, both seats)

| readout | start | shared | separate | separate − shared |
|---|---|---|---|---|
| mae | 2.225 | 2.203 | 2.279 | +0.075 ± 0.335 (4+/2−) |
| bias | 0.007 | 0.012 | 0.005 | -0.006 ± 0.016 (3+/3−) |
| slope | 0.960 | 1.115 | 0.989 | -0.126 ± 0.156 (2+/4−) |
| r2 | 0.107 | 0.141 | 0.119 | -0.022 ± 0.048 (3+/3−) |
| corr | 0.325 | 0.376 | 0.346 | -0.031 ± 0.068 (3+/3−) |

Policy drift on the start's natural states (decisions with a choice):

| quantity | value |
|---|---|
| kl_start_shared | **+0.2301 ± 0.0225** (6+/0−) |
| kl_start_separate | **+0.2592 ± 0.0440** (6+/0−) |
| tv_shared_separate | **+0.2853 ± 0.0136** (6+/0−) |
| kl separate - shared | +0.0292 ± 0.0340 (4+/2−) |

## Value representation: ridge probes (held-out R², M4.6e main split)

| probe | start | shared | separate | separate − shared |
|---|---|---|---|---|
| log_duel<-value_features | 0.175 | 0.171 | 0.474 | **+0.303 ± 0.063** (6+/0−) |
| log_duel<-actor_trunk | 0.175 | 0.171 | 0.168 | -0.003 ± 0.014 (3+/3−) |
| hp_own<-value_features | 0.917 | 0.901 | 0.916 | +0.014 ± 0.020 (5+/1−) |
| hp_own<-actor_trunk | 0.917 | 0.901 | 0.913 | +0.012 ± 0.012 (5+/1−) |
| value_features_move_x1.76_over_natural_spread | 0.103 | 0.106 | 0.165 | **+0.058 ± 0.024** (6+/0−) |

Critic response to single-feature changes of seat A (resources):

| change | start | shared | separate |
|---|---|---|---|
| freeze L5|invested +8 | 0.03 | 0.04 | 0.28 |
| freeze L5|invested +32 | 0.09 | 0.12 | 0.78 |
| freeze L5|unspent gold +4 | -0.05 | -0.04 | 0.03 |
| freeze L5|level +1 | 0.01 | 0.00 | 0.03 |
| freeze L5|HP 60% | -0.29 | -0.31 | -0.27 |
| freeze L5|enemy HP 60% | 0.47 | 0.58 | 0.48 |
| neutral L8|invested +8 | 0.03 | 0.03 | 0.26 |
| neutral L8|invested +32 | 0.10 | 0.11 | 0.78 |
| neutral L8|unspent gold +4 | -0.10 | -0.10 | 0.01 |
| neutral L8|level +1 | 0.00 | 0.01 | 0.03 |
| neutral L8|HP 60% | -0.48 | -0.50 | -0.50 |
| neutral L8|enemy HP 60% | 0.49 | 0.59 | 0.46 |
| neutral L15|invested +8 | 0.02 | 0.02 | 0.14 |
| neutral L15|invested +32 | 0.05 | 0.06 | 0.44 |
| neutral L15|unspent gold +4 | -0.09 | -0.10 | 0.01 |
| neutral L15|level +1 | 0.00 | 0.00 | 0.03 |
| neutral L15|HP 60% | -0.49 | -0.56 | -0.56 |
| neutral L15|enemy HP 60% | 0.46 | 0.55 | 0.54 |

## Behaviour: head to head from s0

| comparison | estimate |
|---|---|
| B1 separate vs shared (W 1000) | +0.10 ± 0.27 (4+/2−) |
| vs start: shared | +0.18 ± 0.27 (4+/2−) |
| vs start: separate | +0.16 ± 0.43 (4+/2−) |
| vs start: separate - shared | -0.02 ± 0.44 (2+/4−) |
| vs other starts: shared | +0.16 ± 0.19 (5+/1−) |
| vs other starts: separate | +0.13 ± 0.17 (4+/2−) |
| vs other starts: separate - shared | -0.03 ± 0.07 (3+/3−) |
| vs suite: shared | **+4.62 ± 0.50** (6+/0−) |
| vs suite: separate | **+4.62 ± 0.59** (6+/0−) |
| vs suite: separate - shared | +0.00 ± 0.59 (4+/2−) |
| B1 powered: 256 new episodes (amendment 1) | +0.07 ± 0.16 (4+/2−) |
| B1 powered: pooled 320 episodes (amendment 1) | +0.07 ± 0.15 (4+/2−) |

separate vs shared over training (W 460):

| transitions | separate vs shared |
|---|---|
| 1,024,000 | -0.06 ± 0.51 (3+/3−) |
| 2,048,000 | +0.42 ± 0.60 (5+/1−) |
| 3,072,000 | -0.12 ± 0.49 (2+/4−) |

Scripted core suite by reference (W 460):

| reference | shared | separate | separate − shared |
|---|---|---|---|
| g_farm | 2.63 | 2.45 | -0.19 ± 1.14 (3+/3−) |
| g_shove | 3.81 | 3.82 | +0.01 ± 1.09 (3+/3−) |
| g_bully | 8.11 | 8.48 | +0.37 ± 1.37 (5+/1−) |
| fz_guard_sh | 3.92 | 3.73 | -0.19 ± 1.05 (3+/3−) |

## Behaviour: strength use (B2 cells; separate − shared, mean over both opponents)

| readout | G50 | G150 |
|---|---|---|
| B2 | +0.07 ± 0.12 (5+/1−) | **+0.14 ± 0.09** (6+/0−) |
| conversion | +0.01 ± 0.22 (3+/3−) | +0.09 ± 0.20 (4+/2−) |
| defence | +0.17 ± 0.18 (5+/1−) | **+0.23 ± 0.20** (6+/0−) |
| equal cells | +0.11 ± 0.18 (5+/1−) | **+0.23 ± 0.20** (5+/1−) |
| conversion - equal | **-0.12 ± 0.06** (0+/6−) | -0.16 ± 0.19 (0+/6−) |
| defence - equal | +0.08 ± 0.32 (4+/2−) | +0.02 ± 0.29 (4+/2−) |
| large x1.76 (diagnostic) | -0.31 ± 0.39 (2+/4−) | -0.38 ± 0.43 (1+/5−) |

By cell kind (G150):

| kind | separate − shared |
|---|---|
| lane brk | +0.25 ± 0.48 (4+/2−) |
| lane edge | +0.14 ± 0.52 (4+/2−) |
| lane equal | **+0.31 ± 0.29** (6+/0−) |
| lane hp | +0.18 ± 0.47 (4+/2−) |
| lane plaus | +0.13 ± 0.37 (3+/3−) |
| lane x1.76 | -0.38 ± 0.43 (1+/5−) |
| lead early A+16 | +0.26 ± 0.43 (4+/2−) |
| lead early A+8 | -0.19 ± 0.44 (2+/4−) |
| lead early B+16 | +0.50 ± 0.52 (5+/1−) |
| lead early B+8 | +0.02 ± 0.73 (3+/3−) |
| lead early equal | **+0.44 ± 0.40** (6+/0−) |
| lead mid A+16 | -0.10 ± 0.40 (3+/3−) |
| lead mid A+8 | +0.14 ± 0.34 (4+/2−) |
| lead mid B+16 | +0.13 ± 0.38 (2+/4−) |
| lead mid B+8 | +0.23 ± 0.40 (4+/2−) |
| lead mid equal | -0.08 ± 0.55 (3+/3−) |

Denial and exposure (ledger at 150 turns):

| quantity | shared level | separate − shared |
|---|---|---|
| conversion: opponent farm lost (unc_B) | 18.25 | -0.00 ± 0.91 (2+/4−) |
| conversion: opponent XP lost (xp_lost_B) | 46.12 | +0.62 ± 7.77 (2+/4−) |
| conversion: own earned (earned_A) | 62.08 | -0.01 ± 0.72 (3+/3−) |
| conversion: opponent deaths | 1.21 | -0.04 ± 0.31 (1+/5−) |
| defence: own farm lost (unc_A) | 18.20 | -0.14 ± 1.33 (3+/3−) |
| defence: own XP lost (xp_lost_A) | 45.25 | -0.91 ± 5.51 (4+/2−) |
| defence: own deaths | 1.26 | -0.10 ± 0.34 (3+/3−) |

## Behaviour: freezes (B3; G50)


freezer:

| readout | separate − shared | shared level |
|---|---|---|
| equal G50 | -0.04 ± 0.15 (2+/4−) |  |
| edge G50 | +0.07 ± 0.25 (4+/2−) |  |
| plaus G50 | -0.05 ± 0.14 (2+/4−) |  |
| hp G50 | +0.16 ± 0.35 (3+/3−) |  |
| brk G50 | +0.01 ± 0.36 (4+/2−) |  |
| x1.76 G50 | -0.24 ± 0.32 (2+/4−) |  |
| favourable (edge, plaus, hp) G50 | +0.06 ± 0.13 (4+/2−) |  |
| favourable: held_A | -0.10 ± 1.35 (3+/3−) | 6.44 |
| favourable: unc_B | -0.01 ± 0.18 (3+/3−) | 6.29 |
| favourable: xp_lost_B | -0.16 ± 1.55 (2+/4−) | 18.06 |
| favourable: deaths_B | -0.01 ± 0.11 (2+/4−) | 0.56 |
| favourable: deaths_A | -0.03 ± 0.13 (2+/4−) | 0.46 |

breaker:

| readout | separate − shared | shared level |
|---|---|---|
| equal G50 | +0.02 ± 0.24 (3+/3−) |  |
| edge G50 | +0.15 ± 0.29 (5+/1−) |  |
| plaus G50 | +0.04 ± 0.11 (4+/2−) |  |
| hp G50 | +0.08 ± 0.23 (4+/2−) |  |
| brk G50 | +0.00 ± 0.33 (4+/2−) |  |
| x1.76 G50 | +0.06 ± 0.30 (4+/2−) |  |
| favourable (edge, plaus, hp) G50 | +0.09 ± 0.17 (5+/1−) |  |
| favourable (freezer advantaged): unc_A | -0.10 ± 0.31 (2+/4−) | 6.59 |
| favourable (freezer advantaged): xp_lost_A | -0.23 ± 1.51 (4+/2−) | 19.41 |
| favourable (freezer advantaged): deaths_A | -0.07 ± 0.13 (2+/4−) | 0.68 |
| favourable (freezer advantaged): frozen_A | -0.10 ± 0.47 (3+/3−) | 6.40 |

## Behaviour: weak side at Recall windows (B4)


weak:

| readout | separate − shared | shared level |
|---|---|---|
| equal G50 | -0.01 ± 0.35 (2+/4−) |  |
| edge G50 | +0.07 ± 0.38 (3+/3−) |  |
| x1.76 G50 | +0.03 ± 0.46 (3+/3−) |  |
| stronger opponent (edge, x1.76) G50 | +0.05 ± 0.37 (3+/3−) |  |
| equal G100 | -0.02 ± 0.44 (4+/2−) |  |
| edge G100 | +0.06 ± 0.46 (3+/3−) |  |
| x1.76 G100 | +0.05 ± 0.69 (3+/3−) |  |
| stronger opponent (edge, x1.76) G100 | +0.05 ± 0.53 (3+/3−) |  |
| deaths_A (100 turns) | -0.04 ± 0.31 (4+/2−) | 1.34 |
| recalls_A (100 turns) | -0.02 ± 0.51 (2+/4−) | 4.46 |
| n_safe_A (100 turns) | +1.63 ± 2.29 (4+/2−) | 16.21 |
| n_yield_A (100 turns) | +0.37 ± 1.96 (4+/2−) | 23.13 |
| away_A (100 turns) | -0.19 ± 0.39 (3+/3−) | 14.33 |
| deaths_B (100 turns) | +0.00 ± 0.11 (3+/3−) | 0.49 |
| earned_A (100 turns) | -0.03 ± 0.77 (3+/3−) | 38.05 |

strong:

| readout | separate − shared | shared level |
|---|---|---|
| equal G50 | +0.12 ± 0.30 (5+/1−) |  |
| edge G50 | -0.05 ± 0.27 (2+/4−) |  |
| x1.76 G50 | -0.16 ± 0.26 (2+/4−) |  |
| advantaged (edge, x1.76) G50 | -0.11 ± 0.13 (1+/5−) |  |
| equal G100 | +0.09 ± 0.37 (4+/2−) |  |
| edge G100 | -0.02 ± 0.40 (4+/2−) |  |
| x1.76 G100 | -0.12 ± 0.35 (2+/4−) |  |
| advantaged (edge, x1.76) G100 | -0.07 ± 0.20 (3+/3−) |  |
| deaths_A (100 turns) | -0.07 ± 0.13 (2+/4−) | 0.56 |
| recalls_A (100 turns) | +0.08 ± 0.30 (3+/3−) | 4.37 |
| n_safe_A (100 turns) | +1.08 ± 1.90 (5+/1−) | 15.08 |
| n_yield_A (100 turns) | +0.54 ± 2.29 (4+/2−) | 19.27 |
| away_A (100 turns) | -0.16 ± 0.22 (1+/5−) | 11.07 |
| deaths_B (100 turns) | -0.08 ± 0.23 (1+/5−) | 1.47 |
| earned_A (100 turns) | -0.03 ± 0.35 (2+/4−) | 43.58 |

first-turn probabilities, weak side vs a stronger opponent:

| actor | recall | Hold | Yield | Trade | Safe | LH | TN | SL | SH |
|---|---|---|---|---|---|---|---|---|---|
| start | 0.016 | 0.130 | 0.633 | 0.028 | 0.194 | 0.073 | 0.107 | 0.152 | 0.652 |
| shared | 0.019 | 0.146 | 0.575 | 0.048 | 0.212 | 0.107 | 0.110 | 0.133 | 0.631 |
| separate | 0.010 | 0.158 | 0.554 | 0.026 | 0.252 | 0.081 | 0.085 | 0.166 | 0.658 |

first-turn probabilities, freezer in favourable freeze cells:

| actor | recall | Hold | Yield | Trade | Safe | LH | TN | SL | SH |
|---|---|---|---|---|---|---|---|---|---|
| start | 0.000 | 0.531 | 0.326 | 0.137 | 0.006 | 0.108 | 0.122 | 0.126 | 0.644 |
| shared | 0.000 | 0.543 | 0.291 | 0.157 | 0.009 | 0.072 | 0.101 | 0.112 | 0.715 |
| separate | 0.000 | 0.443 | 0.362 | 0.187 | 0.007 | 0.096 | 0.126 | 0.134 | 0.644 |

## Behaviour: natural states of start:s (B5)

| readout | separate − shared |
|---|---|
| leader G50 | -0.04 ± 0.07 (2+/4−) |
| leader G150 | +0.01 ± 0.14 (3+/3−) |
| trailer G50 | +0.01 ± 0.12 (4+/2−) |
| trailer G150 | +0.00 ± 0.20 (3+/3−) |
| freezer G50 | -0.18 ± 0.21 (2+/4−) |
| breaker G50 | -0.12 ± 0.23 (3+/3−) |

## Exploratory (amendment 2): natural self-play economy, 16 games × 600 turns per network

| quantity | start | shared | separate | separate − shared | shared − start | separate − start |
|---|---|---|---|---|---|---|
| away_share | 0.112 | 0.117 | 0.119 | +0.002 ± 0.009 (4+/2−) | +0.004 ± 0.008 (5+/1−) | +0.006 ± 0.012 (5+/1−) |
| collection_rate | 0.758 | 0.752 | 0.746 | -0.006 ± 0.028 (3+/3−) | -0.006 ± 0.018 (1+/5−) | -0.012 ± 0.025 (2+/4−) |
| crashes_per_100_turns | 9.477 | 9.590 | 9.770 | +0.180 ± 1.190 (4+/2−) | +0.113 ± 0.865 (3+/3−) | +0.293 ± 0.521 (4+/2−) |
| deaths_per_1000 | 6.085 | 6.840 | 6.762 | -0.078 ± 3.127 (3+/3−) | +0.755 ± 2.610 (3+/2−) | +0.677 ± 2.057 (4+/2−) |
| earned_per_100_turns | 39.498 | 39.371 | 39.060 | -0.311 ± 0.856 (3+/3−) | -0.127 ± 0.454 (1+/5−) | -0.438 ± 0.885 (3+/3−) |
| freeze_turns_share | 0.053 | 0.053 | 0.053 | +0.001 ± 0.012 (3+/3−) | -0.001 ± 0.008 (3+/3−) | -0.000 ± 0.007 (3+/3−) |
| frozen_on_share | 0.053 | 0.053 | 0.053 | +0.001 ± 0.012 (3+/3−) | -0.001 ± 0.008 (3+/3−) | -0.000 ± 0.007 (3+/3−) |
| gold_at_recall | 8.058 | 7.813 | 7.607 | -0.206 ± 0.670 (1+/5−) | -0.244 ± 0.262 (1+/5−) | -0.450 ± 0.848 (1+/5−) |
| gold_present_mean | 5.275 | 5.120 | 5.012 | -0.108 ± 0.417 (1+/5−) | -0.154 ± 0.188 (1+/5−) | -0.263 ± 0.410 (1+/5−) |
| invested_150 | 53.905 | 53.957 | 54.108 | +0.151 ± 1.560 (2+/4−) | +0.053 ± 1.503 (4+/2−) | +0.204 ± 1.424 (3+/3−) |
| invested_300 | 114.129 | 113.822 | 114.000 | +0.178 ± 3.560 (4+/2−) | -0.307 ± 2.415 (2+/4−) | -0.129 ± 2.767 (4+/2−) |
| invested_600 | 230.750 | 230.808 | 228.773 | -2.035 ± 5.766 (3+/3−) | +0.057 ± 2.835 (2+/4−) | -1.978 ± 6.055 (3+/3−) |
| lead_per_100_turns | 0.000 | 0.000 | 0.000 | +0.000 ± 0.000 (0+/0−) | +0.000 ± 0.000 (0+/0−) | +0.000 ± 0.000 (0+/0−) |
| level_end | 18.000 | 18.000 | 18.000 | +0.000 ± 0.000 (0+/0−) | +0.000 ± 0.000 (0+/0−) | +0.000 ± 0.000 (0+/0−) |
| pressure_turns_per_1000 | 20.122 | 20.842 | 18.993 | -1.849 ± 6.708 (4+/2−) | +0.720 ± 6.687 (2+/4−) | -1.128 ± 5.227 (2+/4−) |
| recalls_gold_lt2_share | 0.030 | 0.030 | 0.035 | +0.005 ± 0.011 (5+/1−) | +0.000 ± 0.008 (4+/2−) | +0.006 ± 0.013 (5+/1−) |
| recalls_per_1000 | 41.875 | 42.309 | 43.299 | +0.990 ± 6.697 (4+/2−) | +0.434 ± 2.728 (4+/2−) | +1.424 ± 4.753 (4+/2−) |
| trade_attempts_per_1000 | 227.786 | 235.278 | 242.535 | +7.257 ± 32.287 (3+/3−) | +7.491 ± 25.187 (4+/2−) | +14.748 ± 32.951 (4+/2−) |
| trades_connected_per_1000 | 134.653 | 143.611 | 150.851 | +7.240 ± 18.991 (4+/2−) | +8.958 ± 14.610 (4+/1−) | +16.198 ± 22.874 (4+/2−) |
| wards_per_1000 | 13.637 | 13.785 | 13.576 | -0.208 ± 5.597 (4+/2−) | +0.148 ± 5.482 (2+/4−) | -0.061 ± 7.396 (3+/3−) |
| wards_placed_per_1000 | 13.099 | 13.307 | 13.047 | -0.260 ± 5.421 (4+/2−) | +0.208 ± 5.460 (2+/4−) | -0.052 ± 7.362 (3+/3−) |
| xp_missed_per_100_turns | 29.182 | 31.085 | 32.337 | +1.252 ± 7.183 (3+/3−) | +1.903 ± 5.942 (5+/1−) | +3.155 ± 6.190 (4+/2−) |

# 2.048M snapshots: value (cells, K 16)

| readout | start | shared | separate | separate − shared |
|---|---|---|---|---|
| edge+plaus ratio | 0.15 | -1.88 | 0.27 | +2.15 ± 5.28 (6+/0−) |
| edge ratio | 0.01 | 0.01 | 0.12 | **+0.11 ± 0.04** (6+/0−) |
| plaus ratio | 0.32 | 1.25 | 0.51 | -0.74 ± 2.12 (2+/4−) |
| large ratio | 0.03 | 0.03 | 0.14 | **+0.11 ± 0.03** (6+/0−) |
| gap ratio | 0.02 | 0.03 | 0.16 | **+0.14 ± 0.03** (6+/0−) |
| large+gap ratio | 0.02 | 0.03 | 0.15 | **+0.13 ± 0.03** (6+/0−) |
| edge+plaus predicted | 0.15 | 0.17 | 0.25 | **+0.08 ± 0.04** (6+/0−) |
| edge+plaus realized | 1.06 | 0.84 | 0.99 | +0.15 ± 0.69 (4+/2−) |
| x1.76 predicted | 0.14 | 0.14 | 0.81 | **+0.67 ± 0.18** (6+/0−) |
| x1.76 realized | 5.81 | 5.63 | 5.81 | +0.18 ± 0.94 (5+/1−) |
| bias freeze (A) | -1.90 | -1.95 | -1.45 | +0.50 ± 0.55 (5+/1−) |
| bias lane (A) | -1.61 | -1.35 | -1.53 | -0.18 ± 0.73 (3+/3−) |
| bias gap (A) | -4.72 | -4.39 | -3.77 | +0.62 ± 0.80 (4+/2−) |

# Extension rule

fire: True (a True, b True, c True)

