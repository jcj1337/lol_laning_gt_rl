"""M4.6d: favourable freezes, the weaker side's contest / concede / reset decision, and the specialist gap
(docs/m4/m4_6/m4_6d_preregistration.md). Diagnostic only: no training, no rule or default changes.

Modules: common (definitions, seeds, statistics), records (full-state natural self-play), freezes (episode
classification and disaggregation), cells (constructed starts), plans (adaptive freezer and breaker plans around a
learned policy), matrix (the paired plan x plan rollout job), specialists (workstream C), critic (the long-return
critic diagnostic), report, and the CLI (`python -m lanegame.experiments.rl.m46d <stage> ...`).
"""
