"""The M6 demo's headless core (docs/m6/m6_plan.md, docs/m6/m6a_technical_report.md): the model registry with its
checked checkpoint manifest (`registry`), game sessions with human, model and scripted seats (`session`), the
recording format and replay (`record`), frames for the lane view and the turn history (`frames`), and the action menu
with the engine's reasons (`actions`). No web framework is imported here; the M6b API is a layer over it."""
