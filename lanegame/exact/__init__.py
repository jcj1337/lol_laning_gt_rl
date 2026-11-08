"""Exact game theory on the Phase 1 wave-management game (`GameConfig()`, `.v2()`, `.v1()`).

    solver         backward induction over the reachable layers (`solve_equilibrium`)
    matrix_game    one stage's zero-sum matrix game: saddle-point shortcuts, otherwise a HiGHS linear program
    best_response  exact policy evaluation, best responses and exploitability
    policy         time-dependent Markov policies
    baselines      simple interpretable policies for either side
    analysis       exact forward analysis of a policy pair (state occupancy, path statistics)
    simulate       Monte Carlo play with sampled gank outcomes
    reachability   which lane events can happen from the standard start, with witness action sequences
    scenarios      named lane scenarios with qualitative expectations

The champion rulesets are too large to solve; they are played through `lanegame.env` and studied with RL.
"""
