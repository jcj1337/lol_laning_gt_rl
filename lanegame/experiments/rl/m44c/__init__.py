"""M4.4c: strategic behaviour investigation and multi-turn counterfactual analysis of the M4.4b PPO policies
(docs/m4/m4_4c_strategic_investigation.md).

Question: are the missing or imperfect strategies of the trained policies failures of PPO learning, weaknesses of the
evaluation, or consequences of the simulator's actual incentives? Nothing here trains a network or changes the game:
the M4.4b checkpoints (4,096,000-transition finals as primary, 2,048,000 as replication) are evaluated through paired,
multi-turn counterfactual rollouts from saved states, with the M4.4a engine (`diagnostics.branch.rollout`, common random
numbers, the two opponent protocols) underneath.

    python -m lanegame.experiments.rl.m44c [stage ...]       stages: see __main__.STAGES

Modules
    common      checkpoints, evaluation seed roots (81,000-81,999), statistics, I/O
    plans       multi-turn strategic plans: per-turn rules over existing legal actions, stop conditions, fallbacks,
                logged per rollout; complete-action helpers (Recall, Safe / Hold keeping the policy's other components)
    engine      paired multi-turn branching from a saved state; episode ledgers and discounted returns at several
                horizons; per-state and across-state / across-seed aggregation
    states      state collection from natural self-play (decision-state predicates only) and initialized states
    opponents   opponent styles: the learned policy with one component forced in a stated situation, and the scripted
                references
    recall      Part 2: interrupted Recalls (Recall vs Safe vs Hold vs retreat-then-Recall)
    freeze      Part 3: the multi-turn value of a freeze (preserve vs break vs conditional)
    zoning      Part 4: strong-side zoning (stronger-side strategies x weaker-side responses; mechanics)
    vision      Part 5: Vision before sustained Pressure
    waves       Part 6: Last Hit / Thin / Slow Push / Shove: budgets, one-turn transitions, multi-turn sequences
    traces      counterfactual pairs from one state, exported for the Streamlit viewer and verified by replay
    report      results/m44c/report.md and figures
"""
