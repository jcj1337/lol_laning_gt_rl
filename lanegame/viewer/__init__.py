"""The viewer recording format (`session`): a research game stored as (setup, actions) and replayed through the engine.

The M6 demo reads these recordings (`lanegame.demo.frames`; the research replays in artifacts/m5/traces/), and the
experiments export them (`experiments.m5.traces`, `experiments.m44.trace`). The schema name
`lanegame.viewer_recording` comes from the Streamlit debugger of M4.2e-D that first replayed them; that UI was
retired on 2026-11-08 and is at commit f613f0a0 (`lanegame/viewer/app.py`).
"""
