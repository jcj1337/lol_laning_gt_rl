"""M6c stages (docs/m6/m6c_guided_replays.md). Results go to results/m6/.

    python -m lanegame.experiments.m6c <stage> [<stage> ...]

    browser   a sequence driven in headless Edge against the real server and the built frontend (web/dist), with
              animation on: the sequence opened from the address and from the Watch select (listed under Sequences,
              its kicker reading Sequence at the introduction); Next event walking to every event one position at a
              time (sampled every 150 ms: the note panel stays on screen and the tick step panel never appears), each
              event's note, highlight ring, next event's title and the values its note quotes; Previous event as a
              jump; the away line after a turn shown past an event and the way back; Play walking to the next event
              and stopping there, and past the last event on toward the end; hidden notes, Close sequence, a live
              game kept through a sequence and returned to, a generated match without a sequence, an unknown
              sequence's message; the layouts at three window sizes in both themes (a still screen) with the overlap
              check, the panel's measured heights and the lane legend's two lines kept clear of the panels; under
              the system's reduced-motion setting, plain Watch playback at the pace of the animated turns; and the
              lane's arrows of the last turn's moves: the replay after turns 81, 82, 83 and 74 (each arrow's move, the
              labels Crash, Recall and Back from base in the clear), none at the start or on a stepped turn's start
              frame, the three of turn 81 on its end frame, the screen-reader sentences, a live Recall drawn from -2
              to base, and the crash arrow beside the sequence's ring; wherever arrows are shown, each one's length
              (20 to 40.5 px) and its tip just behind the token or wave that moved. Screenshots go to
              results/m6/m6c/, the checks and the walks' recorded positions to results/m6/m6c_browser.json (under a
              minute)

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
    from . import guide_check as G
    ok = G.run()
    if not ok:
        raise SystemExit("the sequence check had failing checks (see results/m6/m6c_browser.json)")


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
