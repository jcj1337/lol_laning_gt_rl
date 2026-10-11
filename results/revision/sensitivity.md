# Experiment E: sensitivity and robustness (K = 32)

Every row re-solves the game exactly. Baseline = GameConfig() (the documented default). `contest_risk` is the flat Contest gank term (Shove + Contest adds 0.03 on top); `safe_line` is the furthest own position a Safe champion goes; `crash_reward_scale` multiplies the crash reward table.

Qualitative checks (Y = holds):
- C1 the symmetric equilibrium path never crashes
- C2 Contest is the majority posture on the equilibrium path
- C3 aggressive is the most exploitable baseline
- C4 against passive_safe the best response's gain is mostly economic (farm + crash), not ganks
- C5 a 3-wave crash under the enemy tower has negative continuation value for the crasher
- C6 at own -1 with Freeze legal, Freeze + Contest is optimal more often than Freeze + Safe

Across all 61 settings: C1 holds in 61, C2 holds in 47, C3 holds in 61, C4 holds in 61, C5 holds in 61, C6 holds in 58

## One parameter at a time

| setting | nodes | mixed | eq crashes | eq P(C) | node P(C) | P(F) legal | vision S/M/L | expl uniform | expl aggressive | expl freeze_contest | expl passive_safe | expl stack_builder | expl release_safe | expl contest_slowpush | V(crash under B) | V(freeze probe) | C1-C6 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 14849 | 1030 | 0 | 0.91 | 0.57 | 0.069 | 0.10/0.00/0.30 | 10.1 | 24.9 | 9.4 | 9.4 | 5.8 | 7.7 | 6.6 | -5.79 | 0.95 | YYYYYY |
| farm_per_hp=0.125 | 14849 | 1001 | 0 | 0.03 | 0.25 | 0.064 | 0.05/0.00/0.18 | 7.6 | 23.5 | 8.1 | 7.5 | 7.3 | 1.9 | 8.3 | -4.45 | 0.55 | YnYYYY |
| farm_per_hp=0.5 | 14849 | 762 | 0 | 0.97 | 0.82 | 0.065 | 0.14/0.00/0.33 | 16.7 | 30.3 | 16.2 | 17 | 5.4 | 19.5 | 6.4 | -8.37 | 1.31 | YYYYYY |
| collect_contest=0.8 | 14849 | 1590 | 0 | 0.88 | 0.48 | 0.069 | 0.09/0.00/0.27 | 9.4 | 24.4 | 8.5 | 8.4 | 6.1 | 6.2 | 7 | -5.66 | 0.85 | YYYYYY |
| collect_contest=1.0 | 14849 | 803 | 0 | 0.91 | 0.7 | 0.067 | 0.09/0.00/0.28 | 10.9 | 26 | 10.4 | 10.5 | 5.5 | 9.2 | 6.3 | -5.89 | 1.05 | YYYYYY |
| collect_safe_slope=0.1 | 14849 | 829 | 0 | 0.88 | 0.52 | 0.071 | 0.07/0.00/0.27 | 8.6 | 24.9 | 9.6 | 9.8 | 7 | 4.3 | 7.9 | -5.65 | 0.67 | YYYYYY |
| collect_safe_slope=0.3 | 14849 | 1042 | 0 | 0.91 | 0.59 | 0.069 | 0.10/0.00/0.30 | 10.5 | 24.9 | 10.5 | 10.6 | 5.7 | 9 | 6.6 | -5.75 | 0.95 | YYYYYY |
| zone_contest=0.15 | 14849 | 778 | 0 | 0.88 | 0.57 | 0.069 | 0.10/0.00/0.34 | 10 | 24.3 | 9 | 9 | 6.4 | 7.3 | 7.3 | -5.86 | 0.95 | YYYYYY |
| zone_contest=0.45 | 14849 | 1308 | 0 | 0.94 | 0.57 | 0.07 | 0.07/0.00/0.22 | 10.2 | 26 | 9.9 | 9.9 | 5.3 | 8.1 | 6 | -5.65 | 0.95 | YYYYYY |
| gank_loss=3.5 | 14849 | 1208 | 0 | 0.91 | 0.7 | 0.063 | 0.03/0.00/0.09 | 9 | 19 | 9.9 | 10.3 | 5.6 | 8.9 | 6.4 | -5.01 | 0.81 | YYYYYY |
| gank_loss=7.0 | 14849 | 1039 | 0 | 0.04 | 0.42 | 0.068 | 0.80/0.02/0.85 | 11.8 | 33.4 | 9.4 | 9.1 | 7.1 | 6.1 | 8 | -6.86 | 0.97 | YnYYYY |
| gank_position=0.05 | 14849 | 1091 | 0 | 0.91 | 0.69 | 0.06 | 0.10/0.00/0.27 | 7.8 | 17.6 | 11.9 | 12.3 | 8.1 | 8.1 | 9.2 | -5.8 | 0.66 | YYYYYY |
| gank_position=0.15 | 14849 | 845 | 0 | 0.94 | 0.5 | 0.076 | 0.24/0.00/0.29 | 12.5 | 32.2 | 7.7 | 7.6 | 6.9 | 7.3 | 5.2 | -5.6 | 1.08 | YYYYYY |
| gank_base=0.1 | 14849 | 1066 | 0 | 0.91 | 0.56 | 0.069 | 0.03/0.00/0.09 | 10.1 | 24.9 | 9.4 | 9.4 | 5.8 | 7.7 | 6.6 | -5.79 | 0.95 | YYYYYY |
| gank_base=0.2 | 14849 | 1094 | 0 | 0.91 | 0.57 | 0.069 | 0.77/0.00/0.84 | 10.1 | 24.9 | 9.4 | 9.5 | 6 | 7.7 | 6.8 | -5.85 | 0.95 | YYYYYY |
| vision_reduction=0.25 | 14849 | 945 | 0 | 0.91 | 0.55 | 0.069 | 0.00/0.00/0.00 | 10.1 | 24.9 | 9.4 | 9.4 | 5.8 | 7.7 | 6.6 | -5.79 | 0.95 | YYYYYY |
| vision_reduction=0.75 | 14849 | 977 | 0 | 0.91 | 0.59 | 0.07 | 0.82/0.05/0.85 | 10 | 24.9 | 9.4 | 10.2 | 6.5 | 7.7 | 7.4 | -5.91 | 0.95 | YYYYYY |
| contest_risk=0.0 | 14849 | 784 | 0 | 1 | 0.8 | 0.069 | 0.08/0.00/0.27 | 11.3 | 24.8 | 10.7 | 11.4 | 4.1 | 10.6 | 5.2 | -6.07 | 1.03 | YYYYYY |
| contest_risk=0.04 | 14849 | 1356 | 0 | 0.03 | 0.38 | 0.072 | 0.27/0.00/0.29 | 9.7 | 27.4 | 8.6 | 8 | 7.7 | 4.9 | 8.5 | -5.78 | 0.85 | YnYYYY |
| safe_line=-2 | 14851 | 1029 | 0 | 0.03 | 0.47 | 0.064 | 0.03/0.00/0.18 | 8.4 | 25 | 9.6 | 9.5 | 8 | 4.5 | 8.8 | -5.38 | 0.7 | YnYYYY |
| safe_line=0 | 14857 | 1051 | 0 | 0.97 | 0.67 | 0.069 | 0.59/0.00/0.54 | 13.3 | 24.9 | 8.7 | 8.9 | 4.2 | 15.4 | 5 | -6.15 | 1.03 | YYYYYY |
| shove_damage=0 | 32 | 0 | 0 | 0.94 | 0.94 | nan | nan/nan/nan | 2.6 | 5 | 2.5 | 2.5 | 0.2 | 2.5 | 0.2 | -9.61 | 0.95 | YYYYYn |
| shove_damage=2 | 16221 | 1974 | 0 | 0.94 | 0.57 | 0.11 | 0.09/0.00/0.20 | 10.2 | 25.1 | 11.3 | 11.7 | 6.8 | 9.1 | 8.8 | -5.86 | 0.95 | YYYYYY |
| freeze_break_threshold=8 | 14525 | 1009 | 0 | 0.91 | 0.57 | 0.058 | 0.09/0.00/0.32 | 9.9 | 24.9 | 9.4 | 9.4 | 5.8 | 7.7 | 6.6 | -5.79 | 0.95 | YYYYYY |
| freeze_break_threshold=12 | 16859 | 1080 | 0 | 0.91 | 0.57 | 0.082 | 0.11/0.00/0.31 | 10.1 | 24.9 | 8.8 | 9.4 | 5.8 | 7.7 | 6.6 | -5.64 | 0.95 | YYYYYY |
| tower_siege_damage=2 | 7275 | 542 | 0 | 0.91 | 0.58 | 0.117 | 0.03/0.00/0.05 | 10.1 | 24.9 | 10.6 | 10.9 | 7.2 | 7.7 | 7.9 | -3.74 | 0.95 | YYYYYY |
| crash_reward_scale=0.5 | 14849 | 1005 | 0 | 0.91 | 0.57 | 0.074 | 0.82/0.19/0.85 | 10.1 | 24.9 | 6.9 | 7 | 4.6 | 7.7 | 4 | -5.67 | 0.95 | YYYYYY |
| crash_reward_scale=2.0 | 14849 | 1090 | 0 | 0.91 | 0.57 | 0.065 | 0.00/0.00/0.00 | 10 | 24.9 | 16.5 | 17 | 12.7 | 7.7 | 14.5 | -5.91 | 0.95 | YYYYYY |

## Joint sweeps

| setting | nodes | mixed | eq crashes | eq P(C) | node P(C) | P(F) legal | vision S/M/L | expl uniform | expl aggressive | expl freeze_contest | expl passive_safe | expl stack_builder | expl release_safe | expl contest_slowpush | V(crash under B) | V(freeze probe) | C1-C6 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| farm_per_hp=0.125, gank_loss=3.5 | 14849 | 1146 | 0 | 0.04 | 0.42 | 0.064 | 0.02/0.00/0.06 | 5.9 | 16.7 | 8.1 | 8 | 6.5 | 3.1 | 7.6 | -3.49 | 0.49 | YnYYYY |
| farm_per_hp=0.125, gank_loss=5.0 | 14849 | 1001 | 0 | 0.03 | 0.25 | 0.064 | 0.05/0.00/0.18 | 7.6 | 23.5 | 8.1 | 7.5 | 7.3 | 1.9 | 8.3 | -4.45 | 0.55 | YnYYYY |
| farm_per_hp=0.125, gank_loss=7.0 | 14849 | 835 | 0 | 0 | 0.16 | 0.071 | 0.79/0.00/0.85 | 10.3 | 32.6 | 8.5 | 7.4 | 9.2 | 1.3 | 9.8 | -5.44 | 0.65 | YnYYYn |
| farm_per_hp=0.25, gank_loss=3.5 | 14849 | 1208 | 0 | 0.91 | 0.7 | 0.063 | 0.03/0.00/0.09 | 9 | 19 | 9.9 | 10.3 | 5.6 | 8.9 | 6.4 | -5.01 | 0.81 | YYYYYY |
| farm_per_hp=0.25, gank_loss=5.0 | 14849 | 1030 | 0 | 0.91 | 0.57 | 0.069 | 0.10/0.00/0.30 | 10.1 | 24.9 | 9.4 | 9.4 | 5.8 | 7.7 | 6.6 | -5.79 | 0.95 | YYYYYY |
| farm_per_hp=0.25, gank_loss=7.0 | 14849 | 1039 | 0 | 0.04 | 0.42 | 0.068 | 0.80/0.02/0.85 | 11.8 | 33.4 | 9.4 | 9.1 | 7.1 | 6.1 | 8 | -6.86 | 0.97 | YnYYYY |
| farm_per_hp=0.5, gank_loss=3.5 | 14849 | 816 | 0 | 0.97 | 0.86 | 0.059 | 0.05/0.00/0.10 | 15.8 | 24.5 | 17.9 | 18.5 | 5.5 | 20.8 | 6.5 | -7.53 | 1.09 | YYYYYY |
| farm_per_hp=0.5, gank_loss=5.0 | 14849 | 762 | 0 | 0.97 | 0.82 | 0.065 | 0.14/0.00/0.33 | 16.7 | 30.3 | 16.2 | 17 | 5.4 | 19.5 | 6.4 | -8.37 | 1.31 | YYYYYY |
| farm_per_hp=0.5, gank_loss=7.0 | 14849 | 1127 | 0 | 0.91 | 0.7 | 0.068 | 0.81/0.05/0.85 | 18 | 38.1 | 14.9 | 15.6 | 6.4 | 17.8 | 6.4 | -9.49 | 1.61 | YYYYYY |
| collect_safe_slope=0.1, zone_contest=0.15 | 14849 | 774 | 0 | 0.03 | 0.44 | 0.067 | 0.08/0.00/0.30 | 8.4 | 24.4 | 9.3 | 9.3 | 7.9 | 3.2 | 8.9 | -5.7 | 0.61 | YnYYYY |
| collect_safe_slope=0.1, zone_contest=0.3 | 14849 | 829 | 0 | 0.88 | 0.52 | 0.071 | 0.07/0.00/0.27 | 8.6 | 24.9 | 9.6 | 9.8 | 7 | 4.3 | 7.9 | -5.65 | 0.67 | YYYYYY |
| collect_safe_slope=0.1, zone_contest=0.45 | 14849 | 1035 | 0 | 0.94 | 0.55 | 0.072 | 0.05/0.00/0.18 | 9 | 26 | 10.1 | 10.5 | 6.3 | 5.4 | 7 | -5.62 | 0.73 | YYYYYY |
| collect_safe_slope=0.2, zone_contest=0.15 | 14849 | 778 | 0 | 0.88 | 0.57 | 0.069 | 0.10/0.00/0.34 | 10 | 24.3 | 9 | 9 | 6.4 | 7.3 | 7.3 | -5.86 | 0.95 | YYYYYY |
| collect_safe_slope=0.2, zone_contest=0.3 | 14849 | 1030 | 0 | 0.91 | 0.57 | 0.069 | 0.10/0.00/0.30 | 10.1 | 24.9 | 9.4 | 9.4 | 5.8 | 7.7 | 6.6 | -5.79 | 0.95 | YYYYYY |
| collect_safe_slope=0.2, zone_contest=0.45 | 14849 | 1308 | 0 | 0.94 | 0.57 | 0.07 | 0.07/0.00/0.22 | 10.2 | 26 | 9.9 | 9.9 | 5.3 | 8.1 | 6 | -5.65 | 0.95 | YYYYYY |
| collect_safe_slope=0.3, zone_contest=0.15 | 14849 | 767 | 0 | 0.91 | 0.59 | 0.068 | 0.10/0.00/0.34 | 10.4 | 24.3 | 10.3 | 10.3 | 6.3 | 8.9 | 7.2 | -5.84 | 0.95 | YYYYYY |
| collect_safe_slope=0.3, zone_contest=0.3 | 14849 | 1042 | 0 | 0.91 | 0.59 | 0.069 | 0.10/0.00/0.30 | 10.5 | 24.9 | 10.5 | 10.6 | 5.7 | 9 | 6.6 | -5.75 | 0.95 | YYYYYY |
| collect_safe_slope=0.3, zone_contest=0.45 | 14849 | 1261 | 0 | 0.94 | 0.59 | 0.069 | 0.07/0.00/0.22 | 10.6 | 26 | 10.7 | 11 | 5.2 | 9.2 | 5.9 | -5.62 | 0.95 | YYYYYY |
| crash_reward_scale=0.5, farm_per_hp=0.125 | 14849 | 991 | 0 | 0.03 | 0.25 | 0.067 | 0.80/0.14/0.85 | 7.6 | 23.5 | 4.9 | 4.6 | 5.2 | 1.9 | 5.4 | -4.45 | 0.55 | YnYYYY |
| crash_reward_scale=0.5, farm_per_hp=0.25 | 14849 | 1005 | 0 | 0.91 | 0.57 | 0.074 | 0.82/0.19/0.85 | 10.1 | 24.9 | 6.9 | 7 | 4.6 | 7.7 | 4 | -5.67 | 0.95 | YYYYYY |
| crash_reward_scale=0.5, farm_per_hp=0.5 | 14849 | 760 | 0 | 0.97 | 0.82 | 0.066 | 0.83/0.28/0.85 | 16.8 | 30.3 | 16.2 | 16.2 | 4.8 | 19.5 | 4.1 | -8.35 | 1.31 | YYYYYY |
| crash_reward_scale=1.0, farm_per_hp=0.125 | 14849 | 1001 | 0 | 0.03 | 0.25 | 0.064 | 0.05/0.00/0.18 | 7.6 | 23.5 | 8.1 | 7.5 | 7.3 | 1.9 | 8.3 | -4.45 | 0.55 | YnYYYY |
| crash_reward_scale=1.0, farm_per_hp=0.25 | 14849 | 1030 | 0 | 0.91 | 0.57 | 0.069 | 0.10/0.00/0.30 | 10.1 | 24.9 | 9.4 | 9.4 | 5.8 | 7.7 | 6.6 | -5.79 | 0.95 | YYYYYY |
| crash_reward_scale=1.0, farm_per_hp=0.5 | 14849 | 762 | 0 | 0.97 | 0.82 | 0.065 | 0.14/0.00/0.33 | 16.7 | 30.3 | 16.2 | 17 | 5.4 | 19.5 | 6.4 | -8.37 | 1.31 | YYYYYY |
| crash_reward_scale=2.0, farm_per_hp=0.125 | 14849 | 1057 | 0 | 0.03 | 0.26 | 0.061 | 0.00/0.00/0.00 | 7.5 | 23.5 | 16.1 | 15.5 | 13.8 | 1.9 | 16.3 | -4.45 | 0.55 | YnYYYY |
| crash_reward_scale=2.0, farm_per_hp=0.25 | 14849 | 1090 | 0 | 0.91 | 0.57 | 0.065 | 0.00/0.00/0.00 | 10 | 24.9 | 16.5 | 17 | 12.7 | 7.7 | 14.5 | -5.91 | 0.95 | YYYYYY |
| crash_reward_scale=2.0, farm_per_hp=0.5 | 14849 | 830 | 0 | 0.97 | 0.82 | 0.058 | 0.00/0.00/0.00 | 16.7 | 30.3 | 21 | 22.6 | 11.7 | 19.5 | 13.2 | -8.89 | 1.31 | YYYYYY |
| tower_siege_damage=1, farm_per_hp=0.125 | 14849 | 1001 | 0 | 0.03 | 0.25 | 0.064 | 0.05/0.00/0.18 | 7.6 | 23.5 | 8.1 | 7.5 | 7.3 | 1.9 | 8.3 | -4.45 | 0.55 | YnYYYY |
| tower_siege_damage=1, farm_per_hp=0.25 | 14849 | 1030 | 0 | 0.91 | 0.57 | 0.069 | 0.10/0.00/0.30 | 10.1 | 24.9 | 9.4 | 9.4 | 5.8 | 7.7 | 6.6 | -5.79 | 0.95 | YYYYYY |
| tower_siege_damage=1, farm_per_hp=0.5 | 14849 | 762 | 0 | 0.97 | 0.82 | 0.065 | 0.14/0.00/0.33 | 16.7 | 30.3 | 16.2 | 17 | 5.4 | 19.5 | 6.4 | -8.37 | 1.31 | YYYYYY |
| tower_siege_damage=2, farm_per_hp=0.125 | 7275 | 536 | 0 | 0.03 | 0.22 | 0.106 | 0.01/0.00/0.05 | 7.6 | 23.5 | 8.4 | 8.6 | 8.7 | 1.9 | 9.3 | -2.39 | 0.55 | YnYYYn |
| tower_siege_damage=2, farm_per_hp=0.25 | 7275 | 542 | 0 | 0.91 | 0.58 | 0.117 | 0.03/0.00/0.05 | 10.1 | 24.9 | 10.6 | 10.9 | 7.2 | 7.7 | 7.9 | -3.74 | 0.95 | YYYYYY |
| tower_siege_damage=2, farm_per_hp=0.5 | 7275 | 416 | 0 | 0.97 | 0.84 | 0.103 | 0.12/0.00/0.12 | 16.7 | 30.3 | 17 | 17.9 | 6.2 | 19.5 | 7.4 | -6.53 | 1.31 | YYYYYY |

## Ranking of baselines by exploitability (most exploitable first)

| setting | order |
|---|---|
| baseline | aggressive > uniform > freeze_contest > passive_safe > release_safe > contest_slowpush > stack_builder |
| farm_per_hp=0.125 | aggressive > contest_slowpush > freeze_contest > uniform > passive_safe > stack_builder > release_safe |
| farm_per_hp=0.5 | aggressive > release_safe > passive_safe > uniform > freeze_contest > contest_slowpush > stack_builder |
| collect_contest=0.8 | aggressive > uniform > freeze_contest > passive_safe > contest_slowpush > release_safe > stack_builder |
| collect_contest=1.0 | aggressive > uniform > passive_safe > freeze_contest > release_safe > contest_slowpush > stack_builder |
| collect_safe_slope=0.1 | aggressive > passive_safe > freeze_contest > uniform > contest_slowpush > stack_builder > release_safe |
| collect_safe_slope=0.3 | aggressive > passive_safe > freeze_contest > uniform > release_safe > contest_slowpush > stack_builder |
| zone_contest=0.15 | aggressive > uniform > freeze_contest > passive_safe > release_safe > contest_slowpush > stack_builder |
| zone_contest=0.45 | aggressive > uniform > passive_safe > freeze_contest > release_safe > contest_slowpush > stack_builder |
| gank_loss=3.5 | aggressive > passive_safe > freeze_contest > uniform > release_safe > contest_slowpush > stack_builder |
| gank_loss=7.0 | aggressive > uniform > freeze_contest > passive_safe > contest_slowpush > stack_builder > release_safe |
| gank_position=0.05 | aggressive > passive_safe > freeze_contest > contest_slowpush > stack_builder > release_safe > uniform |
| gank_position=0.15 | aggressive > uniform > freeze_contest > passive_safe > release_safe > stack_builder > contest_slowpush |
| gank_base=0.1 | aggressive > uniform > freeze_contest > passive_safe > release_safe > contest_slowpush > stack_builder |
| gank_base=0.2 | aggressive > uniform > passive_safe > freeze_contest > release_safe > contest_slowpush > stack_builder |
| vision_reduction=0.25 | aggressive > uniform > freeze_contest > passive_safe > release_safe > contest_slowpush > stack_builder |
| vision_reduction=0.75 | aggressive > passive_safe > uniform > freeze_contest > release_safe > contest_slowpush > stack_builder |
| contest_risk=0.0 | aggressive > passive_safe > uniform > freeze_contest > release_safe > contest_slowpush > stack_builder |
| contest_risk=0.04 | aggressive > uniform > freeze_contest > contest_slowpush > passive_safe > stack_builder > release_safe |
| safe_line=-2 | aggressive > freeze_contest > passive_safe > contest_slowpush > uniform > stack_builder > release_safe |
| safe_line=0 | aggressive > release_safe > uniform > passive_safe > freeze_contest > contest_slowpush > stack_builder |
| shove_damage=0 | aggressive > uniform > freeze_contest > passive_safe > release_safe > stack_builder > contest_slowpush |
| shove_damage=2 | aggressive > passive_safe > freeze_contest > uniform > release_safe > contest_slowpush > stack_builder |
| freeze_break_threshold=8 | aggressive > uniform > freeze_contest > passive_safe > release_safe > contest_slowpush > stack_builder |
| freeze_break_threshold=12 | aggressive > uniform > passive_safe > freeze_contest > release_safe > contest_slowpush > stack_builder |
| tower_siege_damage=2 | aggressive > passive_safe > freeze_contest > uniform > contest_slowpush > release_safe > stack_builder |
| crash_reward_scale=0.5 | aggressive > uniform > release_safe > passive_safe > freeze_contest > stack_builder > contest_slowpush |
| crash_reward_scale=2.0 | aggressive > passive_safe > freeze_contest > contest_slowpush > stack_builder > uniform > release_safe |
| farm_per_hp=0.125, gank_loss=3.5 | aggressive > freeze_contest > passive_safe > contest_slowpush > stack_builder > uniform > release_safe |
| farm_per_hp=0.125, gank_loss=5.0 | aggressive > contest_slowpush > freeze_contest > uniform > passive_safe > stack_builder > release_safe |
| farm_per_hp=0.125, gank_loss=7.0 | aggressive > uniform > contest_slowpush > stack_builder > freeze_contest > passive_safe > release_safe |
| farm_per_hp=0.25, gank_loss=3.5 | aggressive > passive_safe > freeze_contest > uniform > release_safe > contest_slowpush > stack_builder |
| farm_per_hp=0.25, gank_loss=5.0 | aggressive > uniform > freeze_contest > passive_safe > release_safe > contest_slowpush > stack_builder |
| farm_per_hp=0.25, gank_loss=7.0 | aggressive > uniform > freeze_contest > passive_safe > contest_slowpush > stack_builder > release_safe |
| farm_per_hp=0.5, gank_loss=3.5 | aggressive > release_safe > passive_safe > freeze_contest > uniform > contest_slowpush > stack_builder |
| farm_per_hp=0.5, gank_loss=5.0 | aggressive > release_safe > passive_safe > uniform > freeze_contest > contest_slowpush > stack_builder |
| farm_per_hp=0.5, gank_loss=7.0 | aggressive > uniform > release_safe > passive_safe > freeze_contest > stack_builder > contest_slowpush |
| collect_safe_slope=0.1, zone_contest=0.15 | aggressive > freeze_contest > passive_safe > contest_slowpush > uniform > stack_builder > release_safe |
| collect_safe_slope=0.1, zone_contest=0.3 | aggressive > passive_safe > freeze_contest > uniform > contest_slowpush > stack_builder > release_safe |
| collect_safe_slope=0.1, zone_contest=0.45 | aggressive > passive_safe > freeze_contest > uniform > contest_slowpush > stack_builder > release_safe |
| collect_safe_slope=0.2, zone_contest=0.15 | aggressive > uniform > freeze_contest > passive_safe > release_safe > contest_slowpush > stack_builder |
| collect_safe_slope=0.2, zone_contest=0.3 | aggressive > uniform > freeze_contest > passive_safe > release_safe > contest_slowpush > stack_builder |
| collect_safe_slope=0.2, zone_contest=0.45 | aggressive > uniform > passive_safe > freeze_contest > release_safe > contest_slowpush > stack_builder |
| collect_safe_slope=0.3, zone_contest=0.15 | aggressive > uniform > passive_safe > freeze_contest > release_safe > contest_slowpush > stack_builder |
| collect_safe_slope=0.3, zone_contest=0.3 | aggressive > passive_safe > freeze_contest > uniform > release_safe > contest_slowpush > stack_builder |
| collect_safe_slope=0.3, zone_contest=0.45 | aggressive > passive_safe > freeze_contest > uniform > release_safe > contest_slowpush > stack_builder |
| crash_reward_scale=0.5, farm_per_hp=0.125 | aggressive > uniform > contest_slowpush > stack_builder > freeze_contest > passive_safe > release_safe |
| crash_reward_scale=0.5, farm_per_hp=0.25 | aggressive > uniform > release_safe > passive_safe > freeze_contest > stack_builder > contest_slowpush |
| crash_reward_scale=0.5, farm_per_hp=0.5 | aggressive > release_safe > uniform > freeze_contest > passive_safe > stack_builder > contest_slowpush |
| crash_reward_scale=1.0, farm_per_hp=0.125 | aggressive > contest_slowpush > freeze_contest > uniform > passive_safe > stack_builder > release_safe |
| crash_reward_scale=1.0, farm_per_hp=0.25 | aggressive > uniform > freeze_contest > passive_safe > release_safe > contest_slowpush > stack_builder |
| crash_reward_scale=1.0, farm_per_hp=0.5 | aggressive > release_safe > passive_safe > uniform > freeze_contest > contest_slowpush > stack_builder |
| crash_reward_scale=2.0, farm_per_hp=0.125 | aggressive > contest_slowpush > freeze_contest > passive_safe > stack_builder > uniform > release_safe |
| crash_reward_scale=2.0, farm_per_hp=0.25 | aggressive > passive_safe > freeze_contest > contest_slowpush > stack_builder > uniform > release_safe |
| crash_reward_scale=2.0, farm_per_hp=0.5 | aggressive > passive_safe > freeze_contest > release_safe > uniform > contest_slowpush > stack_builder |
| tower_siege_damage=1, farm_per_hp=0.125 | aggressive > contest_slowpush > freeze_contest > uniform > passive_safe > stack_builder > release_safe |
| tower_siege_damage=1, farm_per_hp=0.25 | aggressive > uniform > freeze_contest > passive_safe > release_safe > contest_slowpush > stack_builder |
| tower_siege_damage=1, farm_per_hp=0.5 | aggressive > release_safe > passive_safe > uniform > freeze_contest > contest_slowpush > stack_builder |
| tower_siege_damage=2, farm_per_hp=0.125 | aggressive > contest_slowpush > stack_builder > passive_safe > freeze_contest > uniform > release_safe |
| tower_siege_damage=2, farm_per_hp=0.25 | aggressive > passive_safe > freeze_contest > uniform > contest_slowpush > release_safe > stack_builder |
| tower_siege_damage=2, farm_per_hp=0.5 | aggressive > release_safe > passive_safe > freeze_contest > uniform > contest_slowpush > stack_builder |
