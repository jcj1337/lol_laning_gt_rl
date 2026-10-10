# Equilibrium structure (horizon 32)

Reachable decision nodes: 905; nodes with a mixed equilibrium: 52.

## Average equilibrium probability of each action over reachable nodes

| player | SP | SH | F |
|---|---|---|---|
| A | 0.518 | 0.174 | 0.308 |
| B | 0.518 | 0.174 | 0.308 |

## Probability of choosing vision after a crash, by size and turns remaining

| size | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| large | 0.00 |  | 1.00 |  | 1.00 |  | 1.00 |  | 1.00 |  | 1.00 |  | 1.00 |  | 1.00 |  | 1.00 |  | 1.00 |  |  |  |  |  |  |  |  |
| medium | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |  |  |  |  |
| small | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

## Equilibrium in constructed situations (lane row: global coordinates -4..+4)

**A 2-stack at river, B wave just spawned** (k = 10, V* = 2.825)

```
      a2      .      .      .     a6      .      .      .     b2
```
A plays SP 1.00; B plays F 1.00. Q (rows A ['SP', 'SH'], cols B ['SP', 'SH', 'F']): `[[2.925, 3.175, 2.825], [-1.45, -1.2, -1.55]]`

**A 2-stack at river, B wave two cells from its tower** (k = 11, V* = 2.775)

```
       .      .     a2      .     a6      .     b2      .      .
```
A plays SP 1.00; B plays SH 1.00. Q (rows A ['SP', 'SH'], cols B ['SP', 'SH', 'F']): `[[2.925, 2.775, 2.825], [0.05, -0.825, -0.05]]`

**A 1-stack at river, B wave just spawned** (k = 10, V* = 2.425)

```
      a2      .      .      .     a4      .      .      .     b2
```
A plays SP 1.00; B plays F 1.00. Q (rows A ['SP', 'SH'], cols B ['SP', 'SH', 'F']): `[[2.525, 2.775, 2.425], [-2.525, -2.275, -2.625]]`

**A 1-stack at river, B wave two cells from its tower** (k = 11, V* = 2.575)

```
       .      .     a2      .     a4      .     b2      .      .
```
A plays SP 1.00; B plays SH 1.00. Q (rows A ['SP', 'SH'], cols B ['SP', 'SH', 'F']): `[[2.725, 2.575, 2.625], [-1.275, -1.9, -1.375]]`

**A 1-stack at +1, B wave two cells from its tower** (k = 11, V* = -2.004)

```
       .      .     a2      .      .     a4     b2      .      .
```
A plays SP 1.00; B plays SH 1.00. Q (rows A ['SP', 'SH'], cols B ['SP', 'SH', 'F']): `[[0.125, -2.004, 0.025], [-1.4, -2.025, -1.5]]`

**A 1-stack at +1, B wave just spawned** (k = 10, V* = -2.303)

```
      a2      .      .      .      .     a4      .      .     b2
```
A plays SP 0.14, SH 0.86; B plays SH 0.92, F 0.08. Q (rows A ['SP', 'SH'], cols B ['SP', 'SH', 'F']): `[[-0.15, -2.479, -0.25], [-2.525, -2.275, -2.625]]`

**B wave held at A's freeze line, A wave arriving** (k = 11, V* = 0.475)

```
       .      .     a2     b2      .      .      .      .      .
```
A plays F 1.00; B plays SP 1.00. Q (rows A ['SP', 'SH', 'F'], cols B ['SP', 'SH']): `[[0.375, 0.475], [-3.225, 0.375], [0.475, 4.1]]`

**B 1-stack at A's freeze line, A wave arriving** (k = 11, V* = 0.025)

```
       .      .     a2     b4      .      .      .      .      .
```
A plays SH 1.00; B plays SH 1.00. Q (rows A ['SP', 'SH', 'F'], cols B ['SP', 'SH']): `[[3.03, -0.6], [3.25, 0.025], [3.13, -0.5]]`

**A wave crashed (siege 1), lanes reset** (k = 10, V* = -0.850)

```
      a2      .      .      .      .      .    A*1     b2      .
```
A plays SP 1.00; B plays SP 1.00. Q (rows A ['SP', 'SH'], cols B ['SP', 'SH']): `[[-0.85, -0.6], [-1.1, -0.85]]`

**Symmetric start** (k = 0, V* = 0.000)

```
      a2      .      .      .      .      .      .      .     b2
```
A plays SP 1.00; B plays SP 1.00. Q (rows A ['SP', 'SH'], cols B ['SP', 'SH']): `[[0.0, 0.25], [-0.25, 0.0]]`

## Mixed-equilibrium nodes at k = 12

```
      a2      .      .     b3      .      .      .      .     b2   vision A0 B0
```
V* = 2.855; A: SH 0.87, F 0.13; B: SP 0.46, SH 0.54; Q (rows ['SP', 'SH', 'F'], cols ['SP', 'SH']) = `[[3.28, 2.3], [2.775, 2.925], [3.38, 2.4]]`

```
      a2      .      .     b4      .      .      .      .     b2   vision A0 B0
```
V* = 2.079; A: SH 0.92, F 0.08; B: SP 0.14, SH 0.86; Q (rows ['SP', 'SH', 'F'], cols ['SP', 'SH']) = `[[0.0, 2.3], [2.255, 2.05], [0.1, 2.4]]`

```
      a2      .      .      .      .     a3      .      .     b2   vision A0 B0
```
V* = -2.855; A: SP 0.46, SH 0.54; B: SH 0.87, F 0.13; Q (rows ['SP', 'SH'], cols ['SP', 'SH', 'F']) = `[[-3.28, -2.775, -3.38], [-2.3, -2.925, -2.4]]`

```
      a2      .      .      .      .     a4      .      .     b2   vision A0 B0
```
V* = -2.079; A: SP 0.14, SH 0.86; B: SH 0.92, F 0.08; Q (rows ['SP', 'SH'], cols ['SP', 'SH', 'F']) = `[[-0.0, -2.255, -0.1], [-2.3, -2.05, -2.4]]`
