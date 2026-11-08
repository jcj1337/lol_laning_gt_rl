# Per-turn behavior of the learned policies

Recorded episodes (150 per seat, SeedSequence(43000), equal skill 0.5). Shares are over the policy's present turns; `Trade if HP ...` is the Trade share on turns where both champions are present, by the policy's HP minus the enemy's; positions are the champion's cell at the start of the turn in own coordinates (lane cell = 0, own tower = −2).

## Against itself (mirror)

| policy family | pos behind lane | pos at lane | pos forward | Trade if HP behind ≥2 | Trade if HP even | Trade if HP ahead ≥2 | Trade early (k<11) | Trade mid | Trade late (k≥22) | games with a recall | first recall turn (per seed) | HP at recall (per seed) | Shove | Freeze |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M3b frozen-snapshot (existing) | 0.43 | 0.56 | 0.01 | 0.51 | 0.66 | 0.93 | 0.64 | 0.60 | 0.57 | 0.00 | - 7.0 13.0 | - 6.0 3.5 | 0.01 | 0.00 |
| shared self-play 100/0 | 0.40 | 0.60 | 0.01 | 0.25 | 0.49 | 0.89 | 0.55 | 0.49 | 0.34 | 0.01 | 14.0 0.0 7.5 | 6.0 6.0 6.0 | 0.06 | 0.00 |
| shared self-play 80/20 (OpenAI Five-inspired) | 0.43 | 0.57 | 0.00 | 0.30 | 0.62 | 0.95 | 0.64 | 0.55 | 0.46 | 0.00 | - - 0.0 | - - 6.0 | 0.01 | 0.00 |
| pub_sprand_s1 (published) | 0.43 | 0.57 | 0.00 | 0.67 | 0.76 | 0.98 | 0.70 | 0.72 | 0.68 | 0.01 | 0.8 | 6.0 | 0.01 | 0.00 |
| pub_sp_s0 (published) | 0.50 | 0.43 | 0.07 | 0.60 | 0.58 | 0.76 | 0.51 | 0.68 | 0.71 | 0.00 | - | - | 0.05 | 0.00 |

## Against hold_farmer

| policy family | pos behind lane | pos at lane | pos forward | Trade if HP behind ≥2 | Trade if HP even | Trade if HP ahead ≥2 | Trade early (k<11) | Trade mid | Trade late (k≥22) | games with a recall | first recall turn (per seed) | HP at recall (per seed) | Shove | Freeze |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M3b frozen-snapshot (existing) | 0.19 | 0.80 | 0.01 | 0.52 | 0.62 | 0.73 | 0.61 | 0.58 | 0.53 | 0.00 | - 0.0 - | - 6.0 - | 0.01 | 0.00 |
| shared self-play 100/0 | 0.16 | 0.82 | 0.03 | 0.23 | 0.42 | 0.66 | 0.53 | 0.42 | 0.23 | 0.00 | - 0.0 1.0 | - 6.0 6.0 | 0.05 | 0.00 |
| shared self-play 80/20 (OpenAI Five-inspired) | 0.24 | 0.76 | 0.00 | 0.31 | 0.53 | 0.71 | 0.62 | 0.48 | 0.34 | 0.00 | - - 0.0 | - - 6.0 | 0.01 | 0.00 |
| pub_sprand_s1 (published) | 0.23 | 0.77 | 0.01 | 0.67 | 0.71 | 0.76 | 0.65 | 0.67 | 0.63 | 0.01 | 0.8 | 6.0 | 0.00 | 0.00 |
| pub_sp_s0 (published) | 0.13 | 0.86 | 0.01 | 0.76 | 0.65 | 0.75 | 0.55 | 0.79 | 0.85 | 0.00 | - | - | 0.04 | 0.00 |

## Against aggressive_trader

| policy family | pos behind lane | pos at lane | pos forward | Trade if HP behind ≥2 | Trade if HP even | Trade if HP ahead ≥2 | Trade early (k<11) | Trade mid | Trade late (k≥22) | games with a recall | first recall turn (per seed) | HP at recall (per seed) | Shove | Freeze |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M3b frozen-snapshot (existing) | 0.39 | 0.60 | 0.00 | 0.43 | 0.63 | 0.84 | 0.53 | 0.52 | 0.49 | 0.00 | 14.0 0.0 14.0 | 6.0 6.0 6.0 | 0.01 | 0.00 |
| shared self-play 100/0 | 0.39 | 0.59 | 0.02 | 0.19 | 0.53 | 0.79 | 0.48 | 0.42 | 0.26 | 0.00 | 14.0 0.0 14.0 | 6.0 6.0 6.0 | 0.06 | 0.00 |
| shared self-play 80/20 (OpenAI Five-inspired) | 0.40 | 0.60 | 0.00 | 0.21 | 0.59 | 0.84 | 0.51 | 0.46 | 0.36 | 0.00 | 14.0 - 0.0 | 6.0 - 6.0 | 0.01 | 0.00 |
| pub_sprand_s1 (published) | 0.40 | 0.60 | 0.00 | 0.64 | 0.72 | 0.87 | 0.61 | 0.66 | 0.65 | 0.01 | 4.7 | 6.0 | 0.00 | 0.00 |
| pub_sp_s0 (published) | 0.51 | 0.44 | 0.04 | 0.63 | 0.70 | 0.84 | 0.64 | 0.75 | 0.78 | 0.00 | - | - | 0.03 | 0.00 |

## Against pub_sprand_s1

| policy family | pos behind lane | pos at lane | pos forward | Trade if HP behind ≥2 | Trade if HP even | Trade if HP ahead ≥2 | Trade early (k<11) | Trade mid | Trade late (k≥22) | games with a recall | first recall turn (per seed) | HP at recall (per seed) | Shove | Freeze |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M3b frozen-snapshot (existing) | 0.41 | 0.58 | 0.01 | 0.47 | 0.66 | 0.92 | 0.60 | 0.58 | 0.56 | 0.00 | - 7.0 21.5 | - 6.0 4.0 | 0.01 | 0.00 |
| shared self-play 100/0 | 0.39 | 0.59 | 0.03 | 0.26 | 0.55 | 0.89 | 0.55 | 0.50 | 0.33 | 0.00 | - 0.0 1.0 | - 6.0 6.0 | 0.06 | 0.00 |
| shared self-play 80/20 (OpenAI Five-inspired) | 0.42 | 0.58 | 0.00 | 0.29 | 0.66 | 0.96 | 0.64 | 0.53 | 0.45 | 0.00 | - - 0.0 | - - 6.0 | 0.01 | 0.00 |
| pub_sprand_s1 (published) | 0.43 | 0.57 | 0.00 | 0.67 | 0.76 | 0.98 | 0.70 | 0.72 | 0.68 | 0.01 | 0.8 | 6.0 | 0.01 | 0.00 |
| pub_sp_s0 (published) | 0.54 | 0.42 | 0.04 | 0.64 | 0.57 | 0.62 | 0.49 | 0.67 | 0.71 | 0.00 | - | - | 0.06 | 0.00 |
