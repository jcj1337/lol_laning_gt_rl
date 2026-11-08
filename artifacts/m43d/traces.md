# M4.3d representative lane traces

Global cells (A's tower at -2, B's at +2). `A12` an A packet of 12 HP, `[A14]` A's crashed mass under B's tower, `a` / `b` the champions. HP in bars after the turn. B farm: what B collected this turn of the ideal (0.1125 per enemy minion HP that died on B's ledger) and the part lost to the unconditional own-tower cap. A risk: A's gank probability this turn and the events that hit A. Seeds: SeedSequence(79,900, (e,)).

## Shove and leave: B farms the crash alone (L10, equal)

`econ`, start S10, gap equal, A `sh_leave`, B `lh`, episode 0

```
turn  A action  B action         -4        -3        -2        -1        +0        +1        +2        +3        +4  HP a/b     B farm (got/ideal, tower cap)  A risk
start                            A8         .         .         .        A8         . [A14],a,b         .        B8
0     SH-S      LH-H              .         .        A8         .         a         .   [A14],b        B4         .  1.00/1.00  0.80/0.90 cap 0.10            p=0.12 
1     SH-S      LH-H             A8         .         .         a        A8         .    [A6],b        B9         .  1.00/1.00  0.80/0.90 cap 0.10            p=0.06 
2     SH-S      LH-H              .         .        A8         a         .         .    [A7],b        B7         .  1.00/1.00  0.70/0.79 cap 0.09            p=0.04 
3     SH-S      LH-H             A8         .         .         a        A8         .         b       B14         .  1.00/1.00  0.70/0.79 cap 0.09            p=0.03 
4     SH-H      LH-H              .         .        A8         .         .      A4,a      B7,b         .         .  1.00/1.00  0.45/0.45 cap 0.00            p=0.13 
5     SH-H      LH-H             A8         .         .         .         .     A10,a         b         .        B8  1.00/1.00  0.23/0.23 cap 0.00            p=0.12 
6     SH-H      LH-H              .         .        A8         .         .         a    [A5],b        B6         .  1.00/1.00  0.50/0.56 cap 0.06            p=0.11 
7     SH-S      LH-H             A8         .         .         a        A8         .      B5,b         .        B8  1.00/1.00  0.50/0.56 cap 0.06            p=0.09 
8     SH-H      LH-H              .         .        A8         .         . A5,B6,a,b         .         .         .  1.00/1.00  0.34/0.34 cap 0.00            p=0.12 
9     SH-H      LH-H             A8         .         .         .         .     A11,a         b         .        B8  1.00/1.00  0.23/0.23 cap 0.00            p=0.12 
10    SH-H      LH-H              .         .        A8         .         .         a    [A6],b        B6         .  1.00/1.00  0.50/0.56 cap 0.06            p=0.11 
11    SH-S      LH-H             A8         .         .         a        A8         .         b       B13         .  1.00/1.00  0.60/0.68 cap 0.07            p=0.09 
```

## Shove and leave at own-tower rate 0.9 (the same seed)

`ot90`, start S10, gap equal, A `sh_leave`, B `lh`, episode 0

```
turn  A action  B action         -4        -3        -2        -1        +0        +1        +2        +3        +4  HP a/b     B farm (got/ideal, tower cap)  A risk
start                            A8         .         .         .        A8         . [A14],a,b         .        B8
0     SH-S      LH-H              .         .        A8         .         a         .   [A14],b        B4         .  1.00/1.00  0.90/0.90 cap 0.00            p=0.12 
1     SH-S      LH-H             A8         .         .         a        A8         .    [A6],b        B9         .  1.00/1.00  0.90/0.90 cap 0.00            p=0.06 
2     SH-S      LH-H              .         .        A8         a         .         .    [A7],b        B7         .  1.00/1.00  0.79/0.79 cap 0.00            p=0.04 
3     SH-S      LH-H             A8         .         .         a        A8         .         b       B14         .  1.00/1.00  0.79/0.79 cap 0.00            p=0.03 
4     SH-H      LH-H              .         .        A8         .         .      A4,a      B7,b         .         .  1.00/1.00  0.45/0.45 cap 0.00            p=0.13 
5     SH-H      LH-H             A8         .         .         .         .     A10,a         b         .        B8  1.00/1.00  0.23/0.23 cap 0.00            p=0.12 
6     SH-H      LH-H              .         .        A8         .         .         a    [A5],b        B6         .  1.00/1.00  0.56/0.56 cap 0.00            p=0.11 
7     SH-S      LH-H             A8         .         .         a        A8         .      B5,b         .        B8  1.00/1.00  0.56/0.56 cap 0.00            p=0.09 
8     SH-H      LH-H              .         .        A8         .         . A5,B6,a,b         .         .         .  1.00/1.00  0.34/0.34 cap 0.00            p=0.12 
9     SH-H      LH-H             A8         .         .         .         .     A11,a         b         .        B8  1.00/1.00  0.23/0.23 cap 0.00            p=0.12 
10    SH-H      LH-H              .         .        A8         .         .         a    [A6],b        B6         .  1.00/1.00  0.56/0.56 cap 0.00            p=0.11 
11    SH-S      LH-H             A8         .         .         a        A8         .         b       B13         .  1.00/1.00  0.68/0.68 cap 0.00            p=0.09 
```

## Shove + Pressure: A stands in B's tower cell (L10, equal)

`econ`, start S10, gap equal, A `sh_press`, B `lh`, episode 3

```
turn  A action  B action         -4        -3        -2        -1        +0        +1        +2        +3        +4  HP a/b     B farm (got/ideal, tower cap)  A risk
start                            A8         .         .         .        A8         . [A14],a,b         .        B8
0     SH-H+P    LH-H              .         .        A8         .         .         . [A14],a,b         .         .  1.00/1.00  0.80/0.90 cap 0.10            p=0.15 
1     SH-H+P    LH-H             A8         .         .         .        A8         .  [A7],a,b         .        B8  1.00/1.00  0.70/0.79 cap 0.09            p=0.15 plate
2     SH-H+P    LH-H              .         .        A8         .         .         .  [A8],a,b         .         .  0.33/1.00  0.70/0.79 cap 0.09            p=0.15 ganked 0.67
3     LH-S      LH-H             A8         .         .         .        A8         .    [A1],b         .        B8  0.00/1.00  0.70/0.79 cap 0.09            p=0.12 ganked 0.50; A DIES
4     ABSENT    LH-H              .         .        A8         .         .         . B6,[A8],b         .         .  0.00/1.00  0.10/0.11 cap 0.01            p=0.00 
5     ABSENT    LH-H             A8         .         .         .        A8         .      B5,b         .        B8  0.00/1.00  0.80/0.90 cap 0.10            p=0.00 
6     ABSENT    LH-H              .         .        A8         .         .  A5,B12,b         .         .         .  0.00/1.00  0.34/0.34 cap 0.00            p=0.00 
7     ABSENT    LH-H             A8         .         a         .         .  A9,B10,b         .         .        B8  1.00/1.00  0.45/0.45 cap 0.00            p=0.00 
8     SH-H      LH-H              .         .        A8         .         a  A6,B10,b         .         .         .  1.00/1.00  0.34/0.34 cap 0.00            p=0.09 
9     SH-H      LH-H             A8         .         .         .         . A12,B1,a,b         .         .        B8  1.00/1.00  0.23/0.23 cap 0.00            p=0.13 
10    SH-H      LH-H              .         .        A8         .         .         a B4,[A9],b         .         .  1.00/1.00  0.33/0.34 cap 0.01            p=0.11 
11    SH-H+P    LH-H             A8         .         .         .        A8         .  [A1],a,b         .        B8  1.00/1.00  0.80/0.90 cap 0.10            p=0.15 plate
12    SH-H+P    LH-H              .         .        A8         .         .         .  [A8],a,b         .         .  1.00/1.00  0.10/0.11 cap 0.01            p=0.15 
13    SH-H+P    LH-H             A8         .         .         .        A8         .  [A1],a,b         .        B8  1.00/1.00  0.70/0.79 cap 0.09            p=0.15 
```

## Shove and dive against a last-hitter (L10, equal)

`econ`, start S10, gap equal, A `sh_dive`, B `lh`, episode 1

```
turn  A action  B action         -4        -3        -2        -1        +0        +1        +2        +3        +4  HP a/b     B farm (got/ideal, tower cap)  A risk
start                            A8         .         .         .        A8         . [A14],a,b         .        B8
0     SH-T+P    LH-H              .         .        A8         .         .         . [A14],a,b        B4         .  0.83/1.00  0.80/0.90 cap 0.10            p=0.17 tower shot; trade dealt 0.00
1     SH-T+P    LH-H             A8         .         .         .        A8         .  [A6],a,b        B9         .  0.17/0.83  0.80/0.90 cap 0.10            p=0.17 ganked 0.67; trade dealt 0.17
2     LH-S      LH-H              .         .        A8         .         .         .    [A7],b        B7         .  0.00/0.83  0.70/0.79 cap 0.09            p=0.13 ganked 0.67; A DIES
3     ABSENT    LH-H             A8         .         .         .        A8         .         b       B14         .  0.00/0.83  0.70/0.79 cap 0.09            p=0.00 
4     ABSENT    LH-H              .         .        A8         .         .        A4     B13,b         .         .  0.00/0.83  0.45/0.45 cap 0.00            p=0.00 
5     ABSENT    LH-H             A8         .         .         .         .        A8     B11,b         .        B8  0.00/0.84  0.45/0.45 cap 0.00            p=0.00 
6     ABSENT    LH-H              .         .      A8,a         .         .        A3     B18,b         .         .  1.00/0.84  0.56/0.56 cap 0.00            p=0.00 
7     SH-H      LH-H             A8         .         .         .      A6,a     B13,b         .         .        B8  1.00/0.84  0.56/0.56 cap 0.00            p=0.11 
8     SH-H      LH-H              .         .        A8         .      A1,a     B14,b         .         .         .  1.00/0.84  0.56/0.56 cap 0.00            p=0.10 
9     SH-H      LH-H             A8         .         .         .      A5,a      B7,b         .         .        B8  1.00/0.84  0.45/0.45 cap 0.00            p=0.10 
10    SH-H      LH-H              .         .        A8         .      A2,a      B8,b         .         .         .  1.00/0.84  0.34/0.34 cap 0.00            p=0.09 
11    SH-H      LH-H             A8         .         .         .      A9,a         .         b         .        B8  1.00/0.84  0.00/0.11 cap 0.00            p=0.09 
```

## Shove and dive against Safe (L10, A x1.76)

`econ`, start S10, gap large, A `sh_dive`, B `safe`, episode 1

```
turn  A action  B action         -4        -3        -2        -1        +0        +1        +2        +3        +4  HP a/b     B farm (got/ideal, tower cap)  A risk
start                            A8         .         .         .        A8         . [A14],a,b         .        B8
0     SH-T+P    LH-S              .         .        A8         .         .         . [A14],a,b        B4         .  0.91/1.00  0.80/0.90 cap 0.10            p=0.17 tower shot; trade dealt 0.00
1     SH-T+P    LH-S             A8         .         .         .        A8         .  [A6],a,b        B9         .  0.53/1.00  0.80/0.90 cap 0.10            p=0.17 ganked 0.67; trade dealt 0.00
2     SH-T+P    LH-S              .         .        A8         .         .         .  [A7],a,b        B7         .  0.06/1.00  0.70/0.79 cap 0.09            p=0.18 ganked 0.67; tower shot; trade dealt 0.00
3     LH-S      LH-S             A8         .         .         .      A8,a         .         b       B14         .  0.06/1.00  0.70/0.79 cap 0.09            p=0.12 
4     LH-S      LH-S              .         .        A8         a         .        A5     B13,b         .         .  0.06/1.00  0.22/0.34 cap 0.00            p=0.07 
5     LH-R      LH-S             A8         .         .         .         .       A10     B11,b         .        B8  0.06/1.00  0.22/0.34 cap 0.00            p=0.00 
6     ABSENT    LH-S              .         .      A8,a         .         .        A6     B17,b         .         .  1.00/1.00  0.30/0.45 cap 0.00            p=0.00 
7     SH-H      LH-S             A8         .         .         .         a       A10     B15,b         .        B8  1.00/1.00  0.30/0.45 cap 0.00            p=0.11 
8     SH-H      LH-S              .         .        A8         .         .      A5,a     B11,b         .         .  1.00/1.00  0.38/0.56 cap 0.00            p=0.12 
9     SH-H      LH-S             A8         .         .         .         .     A12,a         b         .        B8  1.00/1.00  0.07/0.11 cap 0.00            p=0.13 
10    SH-H      LH-S              .         .        A8         .         .         a    [A7],b        B5         .  1.00/1.00  0.50/0.56 cap 0.06            p=0.11 
11    SH-T+P    LH-S             A8         .         .         .        A8         .       a,b       B12         .  0.91/1.00  0.70/0.79 cap 0.09            p=0.17 tower shot; trade dealt 0.00
```

## B punishes a sieger from its tower cell (L10, equal)

`econ`, start S10, gap equal, A `sh_press`, B `punish`, episode 3

```
turn  A action  B action         -4        -3        -2        -1        +0        +1        +2        +3        +4  HP a/b     B farm (got/ideal, tower cap)  A risk
start                            A8         .         .         .        A8         . [A14],a,b         .        B8
0     SH-H+P    LH-T              .         .        A8         .         .         . [A15],a,b         .         .  0.83/1.00  0.56/0.79 cap 0.09            p=0.15 B's trade dealt 0.17
1     SH-H+P    LH-T             A8         .         .         .        A8         .  [A9],a,b         .        B8  0.83/1.00  0.48/0.68 cap 0.07            p=0.15 B's trade dealt 0.00; plate
2     SH-H+P    LH-T              .         .        A8         .         .         . [A10],a,b         .         .  0.00/1.00  0.56/0.79 cap 0.09            p=0.15 ganked 0.67; B's trade dealt 0.17
3     LH-S      LH-T             A8         .         .         .        A8         b      [A4]         .        B8  0.00/1.00  0.48/0.68 cap 0.07            p=0.12 ganked 0.50; B's trade dealt 0.00; A DIES
4     ABSENT    LH-H              .         .        A8         .         .         . B6,[A8],b         .         .  0.00/1.00  0.40/0.45 cap 0.05            p=0.00 
5     ABSENT    LH-H             A8         .         .         .        A8         .      B5,b         .        B8  0.00/1.00  0.80/0.90 cap 0.10            p=0.00 
6     ABSENT    LH-H              .         .        A8         .         .  A5,B12,b         .         .         .  0.00/1.00  0.34/0.34 cap 0.00            p=0.00 
7     ABSENT    LH-H             A8         .         a         .         .  A9,B10,b         .         .        B8  1.00/1.00  0.45/0.45 cap 0.00            p=0.00 
8     SH-H      LH-H              .         .        A8         .         a  A6,B10,b         .         .         .  1.00/1.00  0.34/0.34 cap 0.00            p=0.09 
9     SH-H      LH-T             A8         .         .         .         . A13,B1,a,b         .         .        B8  1.00/1.00  0.09/0.11 cap 0.00            p=0.13 B's trade dealt 0.00
10    SH-H      LH-T              .         .        A8         .         .       a,b  B4,[A11]         .         .  1.00/1.00  0.17/0.23 cap 0.01            p=0.11 B's trade dealt 0.00
11    SH-H+P    LH-T             A8         .         .         .        A8         b    [A4],a         .        B8  1.00/1.00  0.56/0.79 cap 0.09            p=0.15 B's trade dealt 0.00; plate
```

## Vision, then dive (L10, A x1.76)

`econ`, start S10, gap large, A `sh_vis_dive`, B `lh`, episode 2

```
turn  A action  B action         -4        -3        -2        -1        +0        +1        +2        +3        +4  HP a/b     B farm (got/ideal, tower cap)  A risk
start                            A8         .         .         .        A8         . [A14],a,b         .        B8
0     SH-H+V    LH-H              .         .        A8         .         .         . [A14],a,b         .         .  1.00/1.00  0.80/0.90 cap 0.10            p=0.15 
1     SH-T+P    LH-H             A8         .         .         .        A8         .  [A7],a,b         .        B8  1.00/0.71  0.70/0.79 cap 0.09            p=0.04 trade dealt 0.29
2     SH-T+P    LH-H              .         .        A8         .         .         .  [A8],a,b        B6         .  1.00/0.12  0.70/0.79 cap 0.09            p=0.04 trade dealt 0.59
3     SH-T+P    LH-S             A8         .         .         .        A8         .       a,b       B13         .  1.00/0.12  0.80/0.90 cap 0.10            p=0.04 trade dealt 0.00
4     SH-H      LH-S              .         .        A8         .         .      A6,a      B1,b         .         .  1.00/0.12  0.15/0.23 cap 0.00            p=0.04 
5     SH-H      LH-R             A8         .         .         .        A8         a      [A5]         .        B8  1.00/0.12  0.00/0.11 cap 0.00            p=0.03 
6     SH-H+P    ABSENT            .         .        A8         .         .         .  [A8],a,b         .         .  1.00/1.00  0.00/0.56 cap 0.00            p=0.04 plate
7     SH-H+V    LH-H             A8         .         .         .        A8         .  [A1],a,b         .        B8  1.00/1.00  0.70/0.79 cap 0.09            p=0.15 
8     SH-T+P    LH-H              .         .        A8         .         .         . B6,[A8],a,b         .         .  1.00/0.75  0.10/0.11 cap 0.01            p=0.04 trade dealt 0.28
9     SH-T+P    LH-H             A8         .         .         .        A8         .    B5,a,b         .        B8  0.55/0.51  0.80/0.90 cap 0.10            p=0.04 ganked 0.67; tower shot; trade dealt 0.28
10    SH-H      LH-H              .         .        A8         .         a         . B8,[A5],b         .         .  0.55/0.51  0.33/0.34 cap 0.01            p=0.04 
11    SH-T+P    LH-H             A8         .         .         .        A8        B6       a,b         .        B8  0.55/0.51  0.50/0.56 cap 0.06            p=0.04 trade dealt 0.00
12    SH-H      LH-H              .         .        A8         .         .      A6,a      B3,b         .         .  0.55/0.51  0.11/0.23 cap 0.00            p=0.04 
13    SH-H      LH-H             A8         .         .         .        A8         a    [A4],b         .        B8  0.55/0.51  0.23/0.23 cap 0.00            p=0.03 
```
