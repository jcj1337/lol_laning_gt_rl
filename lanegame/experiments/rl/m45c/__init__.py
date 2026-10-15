"""M4.5c: shared-policy PPO self-play on the calibrated M4.5 environment (`GameConfig.m45_calibrated()`).

This execution covers the pilot only: the interface audit, a correctness smoke test, one pilot run (seed 100, the
learning-rate schedule of a 4,096,000-transition run stopped at 1,024,000 agent transitions), its evaluation against
fixed opponents, behaviour diagnostics, throughput measurements and the recommended full-campaign plan. The full
multi-seed campaign is gated behind an explicit confirmation flag (see __main__). Modules:

    common       environment, training settings, transition accounting, paths, seeds, agent specifications
    actions      the M4.5 action space (Ward as its own category)
    references   the fixed evaluation opponents (M4.5b contingent strategies) and the OOD M4.4 network
    train        smoke, pilot, prefix check, rollout diagnostics, throughput benchmarks
    evaluate     learning curves, headline and history against fixed opponents
    behavior     state-conditioned action tables, freezes, recalls, the M4.5b ledger
    stages       audit, smoke evaluation, pilot evaluation and behaviour stages
    plan, report the full-campaign plan and results/m45c/report.md
"""
