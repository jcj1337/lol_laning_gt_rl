"""Run the demo server: the API over the session core and the built frontend (docs/m6/m6b_report.md).

    python -m lanegame.demo [--host 127.0.0.1] [--port 8765] [--no-worker] [--static web/dist]
                            [--idle-minutes 30] [--max-sessions 256]
    python -m lanegame.demo --openapi web/openapi.json          write the API schema (no engine needed)

It runs from the repository root (the registry reads the checkpoints relative to it), loads the engine (about 3 s;
the 7 checkpoint files of lanegame/demo/manifest.json), verifies the three seed-0 research replays, starts
the simulation worker process, and serves http://127.0.0.1:8765/. Without the checkpoints the frontend still loads and
says that the engine is unavailable. Only this machine can connect unless --host says otherwise. Ctrl+C stops.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from .api import REPO, WEB_DIST, create_app


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m lanegame.demo", description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-worker", action="store_true", help="generate bot games in this process (slower turns)")
    ap.add_argument("--static", default=WEB_DIST, help="the built frontend to serve (default web/dist)")
    ap.add_argument("--idle-minutes", type=float, default=30.0, help="a session idle this long expires")
    ap.add_argument("--max-sessions", type=int, default=256)
    ap.add_argument("--openapi", metavar="PATH", help="write the OpenAPI schema to PATH and exit")
    a = ap.parse_args(argv)
    if os.path.isfile(os.path.join(REPO, "pyproject.toml")):
        os.chdir(REPO)
    if a.openapi:
        schema = create_app(static_dir=None).openapi()
        with open(a.openapi, "w", encoding="utf-8", newline="\n") as f:
            json.dump(schema, f, indent=1, sort_keys=True)
            f.write("\n")
        print(f"wrote {a.openapi}")
        return
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    app = create_app(sim_workers=0 if a.no_worker else 1, static_dir=a.static, idle_seconds=a.idle_minutes * 60,
                     max_sessions=a.max_sessions)
    print(f"Laning simulator demo: http://{a.host}:{a.port}/  (Ctrl+C to stop)", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main(sys.argv[1:])
