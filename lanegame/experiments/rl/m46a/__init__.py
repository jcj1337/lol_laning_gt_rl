"""M4.6a diagnosis stage (docs/m4/m4_6/m4_6a_preregistration.md): why freezes break cheaply and why combat strength rarely
becomes lane control, separated into mechanics, scale, training distribution, learning and measurement. No rule,
preset, observation, action space or reward changes; the environment is `GameConfig.m45_calibrated()`.

Modules
    common     seeds, paths, the evaluated policies, worker set-up (exact lane caches), statistics, I/O, the
               pre-registration guard
    policies   behaviour-only plans on a trained policy (restricted and renormalized distribution, CRN sampling),
               scripted plans, state variants
    states     constructed states with arbitrary power gaps, natural and semi-natural state collection
    engine     the paired common-random-number rollout job (resumable), the extended ledger
    e1         exact one-turn ledger of contesting (no learning)
    e3         power-gap self-play: absolute advantage and its channels (+ the E6 critic aggregates)
    e5         the price of a reset and of HP
    e2         selective threats as plans on the trained policy
    e4         best responses trained from gap and freeze starts (seat-aware start archive)
    e6         credit-assignment summaries
    report     tables and the generated report
CLI: `python -m lanegame.experiments.rl.m46a <stage> [--workers N]`.
"""
