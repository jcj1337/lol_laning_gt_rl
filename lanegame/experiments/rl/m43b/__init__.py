"""M4.3b: the PPO self-play pilot on GameConfig.m4_lasthit() (docs/rl/m4_3b_ppo_pilot_report.md).

    python -m lanegame.experiments.rl.m43b [stage ...]      stages: benchmark train curves headline exploit behavior
                                                            counterfactual recall trajectories figures report | all
    python -m lanegame experiments-rl m43b_pilot            the same, every stage

Modules: common (conditions, budgets, paths, seeds), references (the fixed reference opponents), train (the
conditions, in parallel processes), curves (snapshot evaluation against fixed references: learning curves and
behaviour over training), evaluate (the headline W = 1000 evaluation: references, cross-play, initial networks,
attackers), exploit (continuing-time approximate best responses), behavior (conditional action probabilities by
situation, paired counterfactual rollouts), recall (the Recall / death economics study), trajectories (recordings
for the viewer), figures, report.
"""
