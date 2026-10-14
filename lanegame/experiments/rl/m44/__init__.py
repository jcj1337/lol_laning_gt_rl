"""M4.4: PPO self-play on the four-action laning game and the evaluation of emergent strategy
(docs/m4/m4_4a_ppo_readiness.md; plan in docs/rl_research_roadmap.md, section AF).

M4.4a (this package's current scope) is infrastructure only: environment selection, checkpoint compatibility,
four-action decoding, per-turn game records, state-based strategic metrics (wave management, freezes, zoning, vision
and tower pressure, plate phases), component interventions and paired counterfactuals, bounded smoke training and the
M4.4b experiment specification. M4.4b (the 3 environments x 3 seeds x 2,048,000-transition comparison) is prepared
here and NOT run.

    python -m lanegame.experiments.rl.m44 [stage ...]       stages: see __main__.STAGES

Modules
    common          the three environments, the training settings (identical across environments), run paths, seeds,
                    agent specifications and the checkpoint compatibility guard
    actions         the 61-code action space: decoding, the codes that can ever be legal, probability masses
    records         one game -> per-turn, per-side records (decision state, policy masses, action, outcomes)
    metrics         the strategic metrics of docs/m4/m4_4a_ppo_readiness.md §4, computed from records only
    references      the four-action scripted reference suite (M4.3c / M4.3d strategies) as environment agents
    interventions   component interventions (wave / objective / behaviour) on a policy's distribution
    counterfactual  paired counterfactual evaluation from saved states (CRN, two opponent protocols, critic readout)
    train           shared-policy self-play runs (smoke and M4.4b) and their health summaries
    evaluate        evaluation against references, cross-play and historical snapshots
    trace           checkpoint -> viewer recording (lanegame/viewer "Recorded trajectory" mode)
    plan            the M4.4b experiment matrix, commands and runtime estimate
    report          the Markdown tables of results/m44a/
"""
