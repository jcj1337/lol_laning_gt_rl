# Candidate collision models (horizon 32)

| variant | scenarios | states | mixed | eq crashes | eq path actions (A) | A surplus hp1 at river | A one wave hp2 at river | A 2-stack hp6 at river | B wave held at A freeze line |
|---|---|---|---|---|---|---|---|---|---|
| default (lanchester floor, SP hold) | 11/11 | 905 | 52 | 0 | F 0.47, SP 0.53 | -3.55 | -2.83 | 2.82 | 0.47 |
| lanchester_ceil | 11/11 | 879 | 28 | 0 | F 0.47, SP 0.53 | -3.18 | -2.83 | 2.82 | 0.47 |
| linear attrition | 9/11 | 653 | 26 | 0 | F 0.47, SP 0.53 | -6.32 | -6.32 | -3.6 | 0.47 |
| pressure comparison | 8/11 | 1326 | 368 | 0 | F 0.09, SH 0.72, SP 0.19 | 0.15 | 4.5 | 6.37 | 0.38 |
| SP half speed | 9/11 | 503 | 122 | 0 | F 0.47, SP 0.53 | -5.22 | -9.66 | -6.63 | 0.47 |
| SP creep | 10/11 | 503 | 122 | 0 | F 0.47, SP 0.53 | -5.22 | -9.66 | -6.63 | 0.47 |
| no tower damage | 10/11 | 937 | 6 | 0 | F 0.47, SP 0.53 | -3.55 | -2.83 | 2.89 | 0.47 |
| posture caps (SP river, F -1) | 11/11 | 905 | 40 | 0 | F 0.47, SP 0.53 | -3.55 | -2.83 | 2.82 | 0.35 |

## Failed scenario checks

- **default (lanchester floor, SP hold)**: none
- **lanchester_ceil**: none
- **linear attrition**: `reference` (after one turn: A front @+1 hp2 (stacks 0); packets destroyed A/B = [0, 1]); `slowpush_stacks` (A front hp by turn: [2, 2, 2, 2, 2, 2, 2, 2])
- **pressure comparison**: `reference` (after one turn: A front @+1 hp6 (stacks 2); packets destroyed A/B = [0, 1]); `small_wave_pushed_back` (packets destroyed A/B = [1, 0]; B front now at global +2 hp2); `shove_contests_slowpush` (A front hp by turn: [2, 4, 4, 6, 6, 6, 6, 6])
- **SP half speed**: `reference` (after one turn: A front @-4 hp2 (stacks 0); packets destroyed A/B = [0, 1]); `slowpush_stacks` (A front hp by turn: [1, 2, 1, 2, 1, 2, 1, 2])
- **SP creep**: `slowpush_stacks` (A front hp by turn: [1, 2, 1, 2, 1, 2, 1, 2])
- **no tower damage**: `small_wave_intercepted` (A crash hp: [1, 0])
- **posture caps (SP river, F -1)**: none

## Baseline exploitability (as player A)

| variant | uniform | always_shove | always_slowpush | defensive_freeze | stack_then_shove |
|---|---|---|---|---|---|
| default (lanchester floor, SP hold) | 11.86 | 19.45 | 4.85 | 1.85 | 4.85 |
| lanchester_ceil | 11.28 | 19.45 | 5.23 | 2.22 | 5.23 |
| linear attrition | 12.74 | 19.45 | 1.5 | 0 | 1.5 |
| pressure comparison | 10.45 | 23.4 | 9.85 | 6.85 | 9.85 |
| SP half speed | 14.98 | 19.45 | 1.5 | 0 | 1.5 |
| SP creep | 14.98 | 19.45 | 1.5 | 0 | 1.5 |
| no tower damage | 10.84 | 16.2 | 4.85 | 1.85 | 4.85 |
| posture caps (SP river, F -1) | 11.34 | 19.45 | 4.85 | 1.85 | 4.85 |

## States where one of A's actions weakly dominates all others

| variant | dominant-action states | eq support counts |
|---|---|---|
| default (lanchester floor, SP hold) | {'SP': 416, 'F': 227, 'SH': 20} | {'SP': 486, 'F': 301, 'SH': 170} |
| lanchester_ceil | {'SP': 437, 'F': 214, 'SH': 42} | {'SP': 475, 'F': 265, 'SH': 167} |
| linear attrition | {'SP': 298, 'F': 199} | {'SP': 401, 'F': 228, 'SH': 50} |
| pressure comparison | {'SP': 413, 'SH': 109, 'F': 327} | {'SP': 682, 'SH': 507, 'F': 505} |
| SP half speed | {'SP': 242, 'F': 52, 'SH': 12} | {'SP': 378, 'F': 150, 'SH': 97} |
| SP creep | {'SP': 230, 'F': 55, 'SH': 12} | {'SP': 378, 'F': 150, 'SH': 97} |
| no tower damage | {'SP': 402, 'F': 256, 'SH': 50} | {'SP': 491, 'F': 291, 'SH': 161} |
| posture caps (SP river, F -1) | {'SP': 434, 'F': 204, 'SH': 19} | {'SP': 492, 'F': 286, 'SH': 167} |
