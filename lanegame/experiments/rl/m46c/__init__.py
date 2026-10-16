"""M4.6c: a matched shared-policy PPO pilot comparing the C1 investment curve with the calibrated baseline
(docs/m4/m4_6/m4_6c_report.md; protocol docs/m4/m4_6/m4_6c_preregistration.md). `GameConfig.m45_calibrated()` is
unchanged; C1 is the opt-in configuration `m45_calibrated(invest_gain=3.24, invest_scale=64.0)` of M4.6b and never
becomes a default.

Modules
    common      arms, seeds, paths, policy specifications, network loading, statistics, I/O, the pre-registration
    archive     the shared training start archive (gold-defined) and the held-out evaluation cells
    collector   the mixed start distribution (`train_shared(collector_cls=...)`)
    train       smoke test, the matched continuation runs, health checks
    evaluate    the pre-registered evaluation families (paired rollouts, head-to-head, natural play)
    natural     natural self-play records and their metrics
    report      generated tables and the scorecard
CLI: `python -m lanegame.experiments.rl.m46c <stage> [--workers N]`.
"""
