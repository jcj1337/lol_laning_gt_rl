"""M6b stages (docs/m6/m6b_report.md). Results go to results/m6/.

    python -m lanegame.experiments.m6b <stage> [<stage> ...]

    browser   the complete visitor journey in headless Edge against the real server and the built frontend
              (web/dist): a live game played through real clicks and checked against the engine, a trade outcome,
              history review and return, a reload that resumes the session, stop and the record download (replayed in
              Python), Watch with a generated match, forced absence, duplicate and conflicting submissions, hidden and
              shown beliefs, an unanswered request and a lost session, a stale response after a new game, keyboard
              play, reduced motion, and the layouts at five window sizes in both themes with the overlap check.
              Screenshots go to results/m6/m6b/, the checks to results/m6/m6b_browser.json (about 3 minutes)

The stage needs the 7 checkpoint files of lanegame/demo/manifest.json (git-ignored), the built frontend
(cd web && npm install && npm run build) and Microsoft Edge or Chrome. It trains nothing and changes no recording.
"""

from __future__ import annotations

import sys
import time

STAGES = {}


def stage(name):
    def deco(fn):
        STAGES[name] = fn
        return fn
    return deco


@stage("browser")
def _browser():
    from . import journey as J
    ok = J.run()
    if not ok:
        raise SystemExit("the browser journey had failing checks (see results/m6/m6b_browser.json)")


def main(argv) -> None:
    if not argv:
        print(__doc__)
        return
    for name in argv:
        if name not in STAGES:
            raise SystemExit(f"unknown stage {name!r}; stages: {', '.join(STAGES)}")
    for name in argv:
        t0 = time.time()
        print(f"== stage {name}", flush=True)
        STAGES[name]()
        print(f"== stage {name} done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
