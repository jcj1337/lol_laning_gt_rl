# M4 design checks (M4.1)

Inexpensive analytical and simulation checks behind the numbers in `docs/m4_integrated_design.md` and
`docs/m4_design_audit.md`. They read the unchanged engine and change no repository code. Run them from the
repository root with `PYTHONPATH=.`.

| Script | What it computes | Output |
|---|---|---|
| `m4_tables.py` | fixed-point HP / attack-power tables, XP thresholds vs League waves, duel-to-death odds | `m4_tables.txt` |
| `m4_quant.py` | duel edge for level / item / skill leads: full vs random mid-game HP, one vs two rolls per trade | `m4_quant.txt` |
| `m4_hold.py` | turns until an enemy front parked at the freezer's −1 crashes, Freeze vs Slow Push (lane only) | `m4_hold.txt` |
| `freeze_enum.py [legacy\|champ]` | exhaustive lane enumeration: Freeze-legal situations where Freeze ≠ Slow Push (about 10 min) | `freeze_enum_champ_out.txt` |
| `nodiff.py`, `flows.py`, `flows2.py`, `timing.py` | non-contact equal cases, mutual SP / SH cycles, Freeze-binding frequency, fight and travel times | stdout |
| `econ_measure.py [n32] [n200]`, `econ_analyze.py` | scripted-agent economy: kills by distance, gold, trades, deaths, XP rates, horizon 32 vs 200 | `econ_results.npz` (git-ignored), stdout |
