# Documentation

| document | read it for |
|---|---|
| [findings.md](findings.md) | what the project found, from the exact game to the final campaign, with evidence and interpretation kept apart |
| [research_overview.md](research_overview.md) | the final campaign (M5) explained for newcomers; every number is checked against the records by `python -m experiments.m6d overview` |
| [model.md](model.md) | the game the agents play: every state variable, action, rule, reward term and the objective, for the research preset `m45_calibrated()` |
| [methodology.md](methodology.md) | how the games are built, solved, trained on and measured: rulesets, exact solution, environment contract, PPO settings, evaluation instruments, statistics, reproducibility |
| [design_philosophy.md](design_philosophy.md) | the standard every mechanic follows: simplify how League works without changing why its laning strategies work |
| [archive/history.md](archive/history.md) | how the project evolved: decisions on reward and rules, approaches that failed, corrections made later, open questions, lessons |
| [archive/README.md](archive/README.md) | every original report, plan, pre-registration and log, by era, unchanged |

Elsewhere in the repository:

- [README.md](../README.md): the project in brief, the demo and how to reproduce the results.
- [experiments/README.md](../experiments/README.md): the study packages and their stages.
- [artifacts/README.md](../artifacts/README.md): the curated research records and the demo's networks.
- [web/README.md](../web/README.md): the demo's frontend.

## Keeping the documents current

- `model.md` describes only the current research preset. A change to a rule, a parameter or the objective updates
  its equations, tables and examples in the same piece of work and adds a line to its §13 change log. Its
  engine-computed examples come from `python -m experiments.reference` (`artifacts/reference/`), and
  `tests/test_reference_examples.py` pins the deterministic ones.
- A number added to `research_overview.md` needs a check in `experiments/m6d/overview_check.py`.
- Archived documents are never edited. Corrections go into a later document and into
  [archive/history.md](archive/history.md) §6.
