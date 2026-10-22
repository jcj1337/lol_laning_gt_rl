"""The M6 demo (docs/m6/m6_plan.md, docs/m6/m6a_technical_report.md, docs/m6/m6b_report.md).

The headless core: the model registry with its checked checkpoint manifest (`registry`), game sessions with human,
model and scripted seats (`session`), the recording format and replay (`record`), frames for the lane view and the
turn history (`frames`), the action menu with the engine's reasons (`actions`) and the JSON service over the sessions
(`service`). None of these imports a web framework.

The server (M6b): `api` (FastAPI over the service, the typed contract of `schemas`, the built frontend `web/dist`),
`worker` (Watch simulations in a separate process) and `python -m lanegame.demo` (`__main__`)."""
