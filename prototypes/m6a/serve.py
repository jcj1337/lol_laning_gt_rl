"""Serve the M6a demo locally: the pages, and the game engine behind them (throwaway tooling).

    .venv/Scripts/python.exe prototypes/m6a/serve.py [--port 8765] [--no-engine]

Serves prototypes/m6a/ at http://127.0.0.1:<port>/, the screenshots in results/m6/prototypes/ under /shots/, and a
small JSON API under /api/ over the tested session core (lanegame/demo/service.py), so Play resolves the visitor's own
actions. Loading the engine needs the 7 git-ignored checkpoint files; without them (or with --no-engine) the pages
still serve and Play reports that the engine is unavailable. Only this machine can connect (127.0.0.1). Ctrl+C stops.

    GET  /api/status
    POST /api/sessions                  {"opponent": "sp:0" | "om:0", "belief": "hidden" | "shown", "seed"?: int}
    GET  /api/sessions/<id>
    POST /api/sessions/<id>/turns       {"turn": int, "code": int}
    GET  /api/sessions/<id>/history?from=&to=
    POST /api/sessions/<id>/stop
    POST /api/simulations               {"a": model, "b": model, "turns"?: int}
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SHOTS = REPO / "results" / "m6" / "prototypes"
MAX_BODY = 16_384
SERVICE = None


def _api(method: str, path: str, query: dict, body: dict):
    from lanegame.demo.service import ServiceError
    svc = SERVICE
    if svc is None:
        raise ServiceError(503, "engine_unavailable", "the server was started without the engine (--no-engine)")
    q = {k: v[-1] for k, v in query.items()}
    if method == "GET" and path == "/api/status":
        return svc.status()
    if method == "POST" and path == "/api/sessions":
        return svc.create(body.get("opponent", "sp:0"), body.get("belief", "hidden"), body.get("seed"))
    if method == "POST" and path == "/api/simulations":
        return svc.simulate(body.get("a", "om:0"), body.get("b", "sp:0"), body.get("turns", 150))
    m = re.fullmatch(r"/api/sessions/([0-9a-f]{32})(/turns|/history|/stop)?", path)
    if m:
        sid, sub = m.group(1), m.group(2)
        if method == "GET" and sub is None:
            return svc.view(sid)
        if method == "POST" and sub == "/turns":
            return svc.submit(sid, body.get("turn"), body.get("code"))
        if method == "GET" and sub == "/history":
            to = q.get("to")
            return svc.history(sid, int(q.get("from", 0)), None if to in (None, "") else int(to))
        if method == "POST" and sub == "/stop":
            return svc.stop(sid)
    raise ServiceError(404, "not_found", f"no API route {method} {path}")


class Handler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path: str) -> str:
        clean = path.split("?", 1)[0].split("#", 1)[0]
        if clean.startswith("/shots/"):
            p = (SHOTS / clean[len("/shots/"):]).resolve()
            return str(p) if SHOTS.resolve() in p.parents else str(SHOTS)
        return super().translate_path(path)

    def _json(self, status: int, data) -> None:
        b = json.dumps(data, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def _handle_api(self, method: str) -> None:
        from lanegame.demo.service import ServiceError
        parts = urlsplit(self.path)
        body = {}
        if method == "POST":
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                return self._json(413, {"error": "too_large", "message": "request body too large"})
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError
            except ValueError:
                return self._json(400, {"error": "bad_request", "message": "the body must be a JSON object"})
        try:
            self._json(200, _api(method, parts.path, parse_qs(parts.query), body))
        except ServiceError as e:
            self._json(e.status, e.public())
        except (TypeError, ValueError) as e:
            self._json(400, {"error": "bad_request", "message": str(e)})

    def do_GET(self):
        if self.path.startswith("/api/"):
            return self._handle_api("GET")
        return super().do_GET()

    def do_POST(self):
        if self.path.startswith("/api/"):
            return self._handle_api("POST")
        self.send_error(405)

    def end_headers(self):
        if not self.path.startswith("/api/"):
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass


def main() -> None:
    global SERVICE
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-engine", action="store_true", help="serve the pages only")
    a = ap.parse_args()
    if not a.no_engine:
        sys.path.insert(0, str(REPO))
        os.chdir(REPO)                                    # the registry reads checkpoints relative to the repo root
        from lanegame.demo.service import DemoService
        SERVICE = DemoService()
        st = SERVICE.status()
        print("engine: ready" if st["ready"] else f"engine unavailable ({st['error']}); pages only", flush=True)
    handler = functools.partial(Handler, directory=str(HERE))
    with http.server.ThreadingHTTPServer(("127.0.0.1", a.port), handler) as srv:
        print(f"M6a demo: http://127.0.0.1:{a.port}/  (Ctrl+C to stop)", flush=True)
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
