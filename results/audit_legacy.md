# Legacy environment audit

| Measurement | Value |
|---|---|
| Distinct observation keys (continuous gold in key) | 13291 |
| Distinct float gold values observed | 420 |
| Steps where r_A + r_B != 0 (not zero-sum) | 53.8% |
| Mean |r_A + r_B| per step | 0.885 |
| Steps paying the crash bonus | 2211 |
| ...of which immediately after another crash bonus (no new event) | 532 |
| Freeze 'deny' bonus collected per episode (random play) | 2.19 |
| Observation keys revisited at a different time step t | 61282 |
| Wave position after Freeze from the enemy tower (w=+2) | 0 |
| QLearningAgent.update signature (no terminal flag) | `(self, s: Tuple[int, int, int, int, int, int], a: int, r: float, s_next: Tuple[int, int, int, int, int, int]) -> None` |
