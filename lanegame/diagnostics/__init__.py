"""Strategic diagnostics of learned champion-mode policies (docs/rl/openai_five_diagnostics_report.md).

    branch   restorable snapshots of reachable states and paired (common-random-number) counterfactual rollouts:
             commit one action (or a short scripted plan) at a state, then follow fixed policies to the end
    audit    per-turn legality, policy probabilities, choices, states and outcomes of a policy's decisions
    lane_exit  (M4, evaluation only; import as `lanegame.diagnostics.lane_exit`) the lane-exit readout: the
             earned-gold lead at a checkpoint plus its expected change over h repeated continuation turns, the exact
             accounting, the checkpoint's non-gold dimensions and the per-seat event ledger of each continuation

NumPy only: agents are queried through `distribution(obs)`; a `lanegame.rl.NetAgent` works as any other agent.
"""

from .audit import (BEHAVIORS, MODES, OBJECTIVES, WAVES, Turns, audit_games, category_probabilities, economy_table,
                    legality_table, opportunity_table, outcome_table, state_table, summarize)
from .branch import (DEFER, BranchResult, Commit, FollowPolicy, FunctionPlan, Plan, ProbCache, Rollout, Snapshot, branch,
                     observation, restore, rollout, rollout_seed, sample_index)

__all__ = ["Snapshot", "observation", "DEFER", "restore", "rollout_seed", "Plan", "Commit", "FollowPolicy", "FunctionPlan", "ProbCache",
           "sample_index", "Rollout", "rollout", "BranchResult", "branch", "Turns", "audit_games",
           "category_probabilities", "legality_table", "opportunity_table", "state_table", "outcome_table",
           "economy_table", "summarize", "BEHAVIORS", "WAVES", "OBJECTIVES", "MODES"]
