"""Drive the demo the way a visitor does: the real server (`python -m lanegame.demo`) and the built frontend in a
headless Chromium (Edge) over the DevTools protocol. Used by the M6b verification stages and by anyone who wants a
scripted look at the app; nothing here touches the game's rules.

    with Server() as srv, Browser() as b:
        page = b.page(srv.url + "/?fresh", width=1440, height=900)
        page.ready()                               # the app set document.body.dataset.ready = "1"
        page.click("#btnSubmit")                   # a real click at the element's centre
        page.js("window.__demo.state()")           # the app's own state hook
        page.shot("results/m6/m6b/01.png")
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
from typing import Optional

REPO = Path(__file__).resolve().parents[3]
EDGE = [os.environ.get("LANEGAME_VERIFY_BROWSER", ""),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "/usr/bin/google-chrome", "/usr/bin/chromium"]


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def browser_path() -> Optional[str]:
    return next((x for x in EDGE if x and Path(x).is_file()), None)


class Server:
    """`python -m lanegame.demo` on a free port, from the repository root; waits until /api/status answers."""

    def __init__(self, port: Optional[int] = None, static: Optional[str] = None, extra: tuple = ()):
        self.port = port or free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.static, self.extra, self.proc = static, tuple(extra), None

    def __enter__(self) -> "Server":
        cmd = [sys.executable, "-m", "lanegame.demo", "--port", str(self.port)] + (["--static", self.static] if self.static else []) + list(self.extra)
        self.log = open(os.path.join(tempfile.gettempdir(), f"lanegame-demo-{self.port}.log"), "w", encoding="utf-8")
        self.proc = subprocess.Popen(cmd, cwd=REPO, stdout=self.log, stderr=subprocess.STDOUT)
        t0 = time.time()
        while True:
            try:
                self.status = json.loads(urllib.request.urlopen(f"{self.url}/api/status", timeout=2).read())
                break
            except Exception:
                if time.time() - t0 > 120 or self.proc.poll() is not None:
                    raise SystemExit(f"the demo server did not start (log: {self.log.name})")
                time.sleep(0.3)
        return self

    def api(self, method: str, path: str, body=None) -> dict:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data=data, method=method, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return {"status": r.status, "body": json.loads(r.read())}
        except urllib.error.HTTPError as e:
            return {"status": e.code, "body": json.loads(e.read() or b"{}")}

    def wait_simulations(self, timeout: float = 120) -> str:
        t0 = time.time()
        while True:
            st = self.api("GET", "/api/status")["body"]
            if st.get("simulations") != "starting" or time.time() - t0 > timeout:
                return st.get("simulations", "unavailable")
            time.sleep(0.3)

    def __exit__(self, *exc) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=10)
        self.log.close()


class CDP:
    def __init__(self, ws_url: str):
        from websockets.sync.client import connect
        self.ws = connect(ws_url, max_size=None, open_timeout=10)
        self.n = 0
        self.events: list = []

    def call(self, method: str, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv(timeout=120))
            if msg.get("id") == self.n:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})
            self.events.append(msg)

    def close(self) -> None:
        try:
            self.ws.close()
        except Exception:
            pass


class Page:
    """One tab, with the app's hooks."""

    def __init__(self, cdp: CDP, width: int, height: int):
        self.c = cdp
        self.size(width, height)

    def size(self, width: int, height: int) -> None:
        self.width, self.height = width, height
        self.c.call("Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=1, mobile=width < 600)

    def media(self, color_scheme: str = "light", reduced_motion: str = "no-preference") -> None:
        self.c.call("Emulation.setEmulatedMedia", features=[{"name": "prefers-reduced-motion", "value": reduced_motion},
                                                            {"name": "prefers-color-scheme", "value": color_scheme}])

    def goto(self, url: str) -> None:
        self.c.call("Page.navigate", url=url)
        self.ready()

    def reload(self) -> None:
        self.c.call("Page.reload")
        time.sleep(0.3)
        self.ready()

    def js(self, expr: str, timeout: float = 60):
        """Evaluate an expression (a promise is awaited) and return its JSON value."""
        r = self.c.call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        if "exceptionDetails" in r:
            ex = r["exceptionDetails"]
            raise RuntimeError(f"page script failed: {ex.get('exception', {}).get('description') or ex.get('text')}")
        return r["result"].get("value")

    def ready(self, timeout: float = 30) -> None:
        t0 = time.time()
        while True:
            try:
                if self.js("document.body && document.body.dataset.ready") == "1":
                    return
            except RuntimeError:
                pass
            if time.time() - t0 > timeout:
                raise TimeoutError("the page did not become ready")
            time.sleep(0.1)

    def wait(self, expr: str, timeout: float = 30, what: str = "") -> None:
        """Wait until a JavaScript expression is truthy."""
        t0 = time.time()
        while not self.js(expr):
            if time.time() - t0 > timeout:
                raise TimeoutError(f"timed out waiting for {what or expr}")
            time.sleep(0.08)

    def state(self) -> dict:
        return self.js("window.__demo.state()")

    def errors(self) -> list:
        return self.js("window.__errors || []") or []

    def rect(self, selector: str) -> Optional[dict]:
        return self.js(f"(() => {{ const e = document.querySelector({json.dumps(selector)}); if (!e) return null; "
                       f"const r = e.getBoundingClientRect(); return {{x: r.left, y: r.top, w: r.width, h: r.height}}; }})()")

    def click(self, selector: str, wait_ms: int = 120) -> None:
        """A real click at the centre of the element (scrolled into view first)."""
        r = self.js(f"(() => {{ const e = document.querySelector({json.dumps(selector)}); if (!e) return null; "
                    f"e.scrollIntoView({{block: 'nearest', inline: 'nearest'}}); const r = e.getBoundingClientRect(); "
                    f"return {{x: r.left + r.width / 2, y: r.top + r.height / 2, disabled: !!e.disabled}}; }})()")
        if r is None:
            raise RuntimeError(f"no element {selector}")
        if r["disabled"]:
            raise RuntimeError(f"{selector} is disabled")
        for kind in ("mousePressed", "mouseReleased"):
            self.c.call("Input.dispatchMouseEvent", type=kind, x=r["x"], y=r["y"], button="left", clickCount=1)
        time.sleep(wait_ms / 1000)

    def click_text(self, text: str, within: str = "body", exact: bool = True) -> None:
        sel = self.js(f"""(() => {{
            const root = document.querySelector({json.dumps(within)}); if (!root) return null;
            const els = [...root.querySelectorAll('button, a, [role=button], summary, label, option')];
            const t = {json.dumps(text)};
            const e = els.find((x) => ({'true' if exact else 'false'} ? x.textContent.trim() === t : x.textContent.includes(t)) && x.getClientRects().length);
            if (!e) return null; if (!e.id) e.id = 'auto_' + Math.random().toString(36).slice(2); return '#' + CSS.escape(e.id); }})()""")
        if sel is None:
            raise RuntimeError(f"no clickable element with text {text!r}")
        self.click(sel)

    def key(self, key: str, code: Optional[str] = None, text: Optional[str] = None, wait_ms: int = 80) -> None:
        """A key press (Input.dispatchKeyEvent), by DOM key name."""
        vk = {"ArrowLeft": 37, "ArrowRight": 39, "Home": 36, "End": 35, "Escape": 27, "Enter": 13, " ": 32, "Tab": 9}.get(key, 0)
        p = {"key": key, "code": code or ({" ": "Space", "Enter": "Enter"}.get(key, key)), "windowsVirtualKeyCode": vk, "nativeVirtualKeyCode": vk}
        if text is not None or key in (" ", "Enter"):
            p["text"] = text if text is not None else ("\r" if key == "Enter" else " ")
        self.c.call("Input.dispatchKeyEvent", type="keyDown", **p)
        self.c.call("Input.dispatchKeyEvent", type="keyUp", **{k: v for k, v in p.items() if k != "text"})
        time.sleep(wait_ms / 1000)

    def shot(self, path: str, clip: Optional[dict] = None, scale: float = 1.0) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        args = {"format": "png", "captureBeyondViewport": False}
        if clip:
            args["clip"] = {**clip, "scale": scale}
        png = self.c.call("Page.captureScreenshot", **args)["data"]
        Path(path).write_bytes(base64.b64decode(png))
        return path

    def downloads(self, folder: str) -> None:
        os.makedirs(folder, exist_ok=True)
        self.c.call("Browser.setDownloadBehavior", behavior="allow", downloadPath=str(Path(folder).resolve()), eventsEnabled=True)


class Browser:
    """A throwaway headless Edge profile with DevTools."""

    def __init__(self):
        self.exe = browser_path()
        if not self.exe:
            raise SystemExit("no Chromium browser found (set LANEGAME_VERIFY_BROWSER)")
        self.profile = Path(tempfile.mkdtemp(prefix="m6b-browser-"))
        self.proc = None
        self.cdp: Optional[CDP] = None

    def __enter__(self) -> "Browser":
        self.proc = subprocess.Popen([self.exe, "--headless=new", "--remote-debugging-port=0", f"--user-data-dir={self.profile}",
                                      "--no-first-run", "--no-default-browser-check", "--disable-extensions", "--force-device-scale-factor=1",
                                      "--hide-scrollbars", "--window-size=1440,900", "about:blank"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t0 = time.time()
        while not (self.profile / "DevToolsActivePort").is_file():
            if time.time() - t0 > 30:
                raise SystemExit("the browser did not open its DevTools port")
            time.sleep(0.2)
        self.port = int((self.profile / "DevToolsActivePort").read_text().split()[0])
        time.sleep(0.4)
        return self

    def page(self, url: str, width: int = 1440, height: int = 900, color_scheme: str = "light", reduced_motion: str = "no-preference") -> Page:
        if self.cdp is None:
            pages = [t for t in json.loads(urllib.request.urlopen(f"http://127.0.0.1:{self.port}/json/list").read()) if t["type"] == "page"]
            self.cdp = CDP(pages[0]["webSocketDebuggerUrl"])
            self.cdp.call("Page.enable")
            self.cdp.call("Runtime.enable")
        p = Page(self.cdp, width, height)
        p.media(color_scheme, reduced_motion)
        p.goto(url)
        return p

    def __exit__(self, *exc) -> None:
        if self.cdp:
            self.cdp.close()
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=10)


__all__ = ["Server", "Browser", "Page", "CDP", "REPO", "free_port", "browser_path"]
