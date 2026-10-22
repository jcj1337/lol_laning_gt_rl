"""Screenshots of the M6a prototypes in headless Edge (throwaway tooling).

    .venv/Scripts/python.exe prototypes/m6a/tools/shoot.py <shots.json> [<out dir>]

<shots.json> is a list of {"name", "path" (relative to prototypes/m6a, with query), "width", "height",
"steps" (optional: JavaScript expressions run in order after the page is ready, each awaited), "wait_ms",
"clip_js" (optional: an expression returning the {x, y, width, height} to crop to), "clip_scale"}.
A page signals readiness by setting document.body.dataset.ready = "1". The tool starts the launcher's server
(serve.py) on a free local port, drives a throwaway headless Edge profile over DevTools, and writes
<out dir>/<name>.png (default results/m6/prototypes). Nothing else is written.
"""

from __future__ import annotations

import base64
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
EDGE = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"]


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class CDP:
    def __init__(self, port: int):
        from websockets.sync.client import connect
        pages = [t for t in json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list").read())
                 if t["type"] == "page"]
        self.ws = connect(pages[0]["webSocketDebuggerUrl"], max_size=None, open_timeout=10)
        self.n = 0

    def call(self, method: str, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv(timeout=60))
            if msg.get("id") == self.n:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    def js(self, expr: str):
        r = self.call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        if "exceptionDetails" in r:
            raise RuntimeError(f"page script failed: {r['exceptionDetails']}")
        return r["result"].get("value")


def main(spec: str, out: str) -> None:
    shots = json.loads(Path(spec).read_text(encoding="utf-8"))
    os.makedirs(out, exist_ok=True)
    port = free_port()
    server = subprocess.Popen([sys.executable, str(HERE / "serve.py"), "--port", str(port)], cwd=HERE,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    while True:                                            # the server binds after the engine has loaded
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=2).read()
            break
        except Exception:
            if time.time() - t0 > 60 or server.poll() is not None:
                raise SystemExit("the demo server did not start")
            time.sleep(0.3)
    profile = Path(tempfile.mkdtemp(prefix="m6a-shoot-"))
    exe = next(x for x in EDGE if Path(x).is_file())
    browser = subprocess.Popen([exe, "--headless=new", "--remote-debugging-port=0", f"--user-data-dir={profile}",
                                "--no-first-run", "--no-default-browser-check", "--disable-extensions",
                                "--force-device-scale-factor=1", "--hide-scrollbars", "about:blank"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        t0 = time.time()
        while not (profile / "DevToolsActivePort").is_file():
            if time.time() - t0 > 30:
                raise SystemExit("Edge did not open its DevTools port")
            time.sleep(0.2)
        dport = int((profile / "DevToolsActivePort").read_text().split()[0])
        time.sleep(0.5)
        c = CDP(dport)
        for s in shots:
            w, h = int(s.get("width", 1440)), int(s.get("height", 900))
            c.call("Emulation.setDeviceMetricsOverride", width=w, height=h, deviceScaleFactor=1, mobile=w < 600)
            feats = [{"name": "prefers-reduced-motion", "value": s.get("reduced_motion", "no-preference")},
                     {"name": "prefers-color-scheme", "value": s.get("color_scheme", "light")}]
            c.call("Emulation.setEmulatedMedia", features=feats)
            c.call("Page.navigate", url=f"http://127.0.0.1:{port}/{s['path']}")
            t0 = time.time()
            while c.js("document.body && document.body.dataset.ready") != "1":
                if time.time() - t0 > 20:
                    raise SystemExit(f"{s['name']}: the page did not become ready")
                time.sleep(0.2)
            for step in s.get("steps", []):
                c.js(step)
                time.sleep(0.25)
            time.sleep(s.get("wait_ms", 600) / 1000)
            shot = {"format": "png", "captureBeyondViewport": False}
            if s.get("clip_js"):                       # optional: crop to a box the page reports {x, y, width, height}
                shot["clip"] = {**c.js(s["clip_js"]), "scale": s.get("clip_scale", 1)}
            png = c.call("Page.captureScreenshot", **shot)["data"]
            p = Path(out) / f"{s['name']}.png"
            p.write_bytes(base64.b64decode(png))
            errs = c.js("window.__errors ? window.__errors.join(' | ') : ''")
            print(f"wrote {p.relative_to(REPO) if p.is_relative_to(REPO) else p}" + (f"  (page errors: {errs})" if errs else ""))
    finally:
        browser.kill()
        server.kill()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else str(REPO / "results" / "m6" / "prototypes"))
