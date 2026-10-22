# The M6a glass prototype (archived, not runnable)

The interface design the user approved on 2026-10-21 after six review rounds
([decision record](../../docs/m6/m6a_decision_record.md) §7), kept as the written reference for the production
application in `web/`, which implements it (M6b, 2026-10-22, [report](../../docs/m6/m6b_report.md)).

The four files are unchanged from `prototypes/m6a/glass/` and `prototypes/m6a/shared/core.js` at commit `1ee8c081`,
apart from the import path of `core.js`. They do not run on their own: the prototype's local server
(`prototypes/m6a/serve.py`) and the recorded data files it served were removed with the other prototypes once the
application worked. The screenshots of every design round are in `results/m6/prototypes/`.

Nothing in `web/` imports this folder.
