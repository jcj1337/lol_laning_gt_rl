#!/usr/bin/env python3
"""verify-lanegame harness: drive lanegame's CLI and Streamlit viewer the way a user does, and keep the evidence.

Run it with the repo's venv, from the repo root:

    .venv/Scripts/python.exe .claude/skills/verify-lanegame/scripts/vl.py <command> ...

    new-run                                 create a run folder (prints RUN); every other command takes --run RUN
    doctor   [--run RUN]                    read-only health check: venv, editable install, git, deps, our instances
    cli      --run RUN --label L -- ARGS    `python -m lanegame ARGS` in RUN/work/cli, transcript + created files kept
    viewer   start|stop --run RUN [--port N]
    demo     start|stop --run RUN [--port N]  the M6 web demo (python -m lanegame.demo: the API and the built frontend)
    browser  start|stop --run RUN
    browser  open       --run RUN [--url U] [--wait-text T | --ready-js EXPR]
    browser  js         --run RUN --expr EXPR [--label L]   evaluate JavaScript in the page (the demo's window.__demo hooks)
    browser  click      --run RUN --text T [--key K | --scope sidebar|main] [--contains] [--nth N] [--no-settle]
    browser  select     --run RUN --key K --option T
    browser  fill       --run RUN --key K --value V
    browser  wait       --run RUN --text T [--scope ...] [--timeout S] [--no-settle]
    browser  text       --run RUN [--scope ...] [--label L]
    browser  widgets    --run RUN [--label L]          every keyed widget's value (selectbox values are not in text)
    browser  screenshot --run RUN --label L [--full | --here]   L.png + L.txt (page text) + L.widgets.json
    cleanup  --run RUN                      stop what this run started, delete RUN/work and RUN/state, keep RUN/evidence

Evidence lives in RUN/evidence (RUN = $LANEGAME_VERIFY_ROOT/<id>, default <system temp>/lanegame-verify/<id>) and
survives cleanup. Every command appends to RUN/evidence/actions.log.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
WIN = os.name == "nt"
VENV_PY = REPO / ".venv" / ("Scripts/python.exe" if WIN else "bin/python")
ROOT = Path(os.environ.get("LANEGAME_VERIFY_ROOT") or Path(tempfile.gettempdir()) / "lanegame-verify")
VIEWER_PORTS = range(8600, 8700)              # never 8501, the port a user's own `streamlit run` takes
DEMO_PORTS = range(8700, 8800)                # never 8765, the port a user's own `python -m lanegame.demo` takes
COPY_LIMIT = 2_000_000                        # created files up to this size are copied into the evidence
BROWSERS = [os.environ.get("LANEGAME_VERIFY_BROWSER", ""),
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "/usr/bin/google-chrome", "/usr/bin/chromium"]


# ---------------------------------------------------------------------------------------------------- run folders
def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def run_dir(arg: str) -> Path:
    p = Path(arg)
    p = p if p.is_absolute() else ROOT / arg
    if not (p / "run.json").is_file():
        sys.exit(f"no run at {p}; create one with `vl.py new-run`")
    return p


def log(run: Path, line: str) -> None:
    with open(run / "evidence" / "actions.log", "a", encoding="utf-8") as f:
        f.write(f"{now()}  {line}\n")


def state(run: Path, name: str):
    f = run / "state" / f"{name}.json"
    return json.loads(f.read_text()) if f.is_file() else None


def set_state(run: Path, name: str, value) -> None:
    f = run / "state" / f"{name}.json"
    if value is None:
        f.unlink(missing_ok=True)
    else:
        f.write_text(json.dumps(value, indent=1))


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, encoding="utf-8").stdout


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def repo_snapshot() -> dict:
    """Every non-ignored change git sees (with content hashes, so a second edit of a modified file shows), plus the
    git-ignored checkpoint folder's entries. A verification run must leave both exactly as it found them."""
    snap = {}
    for line in git("status", "--porcelain=v1", "--untracked-files=all").splitlines():
        rel = line[3:].split(" -> ")[-1].strip('"')
        p = REPO / rel
        snap[rel] = f"{line[:2]} {sha256(p) if p.is_file() else '-'}"
    ck = REPO / "results" / "rl" / "checkpoints"
    if ck.is_dir():
        for e in os.scandir(ck):
            snap[f"[ignored] results/rl/checkpoints/{e.name}"] = str(e.stat().st_mtime_ns)
    return snap


def repo_guard(run: Path) -> list:
    base = state(run, "repo_baseline") or {}
    cur = repo_snapshot()
    return sorted(k for k in set(base) | set(cur) if base.get(k) != cur.get(k))


def cmd_new_run(a) -> None:
    rid = dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + os.urandom(2).hex()
    run = ROOT / rid
    for sub in ("evidence", "work/cli", "state"):
        (run / sub).mkdir(parents=True)
    dirty = git("status", "--porcelain").splitlines()
    meta = {"id": rid, "created": now(), "repo": str(REPO), "git_head": git("rev-parse", "HEAD").strip(),
            "git_dirty_entries": len(dirty), "python": str(VENV_PY)}
    (run / "run.json").write_text(json.dumps(meta, indent=1))
    shutil.copy(run / "run.json", run / "evidence" / "run.json")
    set_state(run, "repo_baseline", repo_snapshot())
    log(run, f"new-run {rid} at git {meta['git_head'][:10]} ({len(dirty)} dirty entries)")
    print(f"RUN={run}")
    print(f"id {rid}; evidence goes to {run / 'evidence'}")


# ---------------------------------------------------------------------------------------------------- processes
def pid_alive(pid: int) -> bool:
    import psutil
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def kill_tree(pid: int) -> None:
    """Kill the process this run started and its descendants (never anything found by name)."""
    import psutil
    try:
        root = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    procs = root.children(recursive=True) + [root]
    for p in procs:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(procs, timeout=10)


def spawn(cmd: list, cwd: Path, log_path: Path) -> int:
    out = open(log_path, "ab")
    kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
          | subprocess.CREATE_NO_WINDOW} if WIN else {"start_new_session": True}
    p = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, **kw)
    return p.pid


def port_free(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def port_owner_ok(pid: int, port: int) -> bool:
    import psutil
    try:
        procs = [psutil.Process(pid)] + psutil.Process(pid).children(recursive=True)
    except psutil.NoSuchProcess:
        return False
    for p in procs:
        try:
            if any(c.laddr and c.laddr.port == port and c.status == psutil.CONN_LISTEN for c in p.net_connections()):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return False


def http_get(url: str, timeout: float = 3.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


# ---------------------------------------------------------------------------------------------------- doctor
def cmd_doctor(a) -> None:
    rows = []

    def check(name, ok, detail, fatal=True):
        rows.append((("PASS" if ok else ("FAIL" if fatal else "WARN")), name, detail))

    check("venv python", VENV_PY.is_file(), str(VENV_PY))
    check("running under the venv", Path(sys.executable).resolve() == VENV_PY.resolve(), sys.executable, fatal=False)
    r = subprocess.run([str(VENV_PY), "-c", "import lanegame, sys; print(lanegame.__file__); print(sys.version.split()[0])"],
                       capture_output=True, text=True, cwd=tempfile.gettempdir())
    lines = r.stdout.split()
    check("lanegame import (from outside the repo)", r.returncode == 0 and bool(lines)
          and Path(lines[0]).resolve().is_relative_to(REPO), " ".join(lines) or r.stderr.strip()[-200:])
    head = git("rev-parse", "--short", "HEAD").strip()
    check("git", bool(head), f"HEAD {head}, {len(git('status', '--porcelain').splitlines())} dirty entries", fatal=False)
    for mod in ("torch", "streamlit", "plotly", "psutil", "websockets"):
        r = subprocess.run([str(VENV_PY), "-c", f"import {mod}; print(getattr({mod}, '__version__', 'ok'))"],
                           capture_output=True, text=True)
        need = mod in ("psutil", "websockets")
        check(f"module {mod}", r.returncode == 0, r.stdout.strip() or "missing", fatal=need)
    br = next((b for b in BROWSERS if b and Path(b).is_file()), None)
    check("browser for the viewer", br is not None, br or "no Edge/Chrome found; set LANEGAME_VERIFY_BROWSER", fatal=False)
    if a.run:
        run = run_dir(a.run)
        check("run folder", True, str(run))
        v = state(run, "viewer")
        if v:
            alive = pid_alive(v["pid"])
            check("viewer process (ours)", alive, f"pid {v['pid']}")
            check("viewer port owned by our process", alive and port_owner_ok(v["pid"], v["port"]), f"port {v['port']}")
            try:
                h = http_get(f"{v['url']}_stcore/health")
            except Exception as e:                                   # noqa: BLE001
                h = f"error: {e}"
            check("viewer health endpoint", h.strip() == "ok", h.strip()[:80])
        d = state(run, "demo")
        if d:
            alive = pid_alive(d["pid"])
            check("demo process (ours)", alive, f"pid {d['pid']}")
            try:
                st = json.loads(http_get(f"{d['url']}api/status"))
                h = "ready" if st.get("ready") else f"engine unavailable: {st.get('error')}"
            except Exception as e:                                   # noqa: BLE001
                h = f"error: {e}"
            check("demo status endpoint", alive and not h.startswith("error"), h[:80], fatal=False)
        b = state(run, "browser")
        if b:
            alive = pid_alive(b["pid"])
            check("browser process (ours)", alive, f"pid {b['pid']}")
            try:
                ver = json.loads(http_get(f"http://127.0.0.1:{b['port']}/json/version")).get("Browser", "?")
            except Exception as e:                                   # noqa: BLE001
                ver = f"error: {e}"
            check("browser DevTools endpoint", alive and not ver.startswith("error"), ver)
        changed = repo_guard(run)
        check("repo untouched since new-run", not changed, ", ".join(changed[:5]) or "no change")
    w = max(len(r[1]) for r in rows)
    for status, name, detail in rows:
        print(f"{status}  {name:<{w}}  {detail}")
    sys.exit(1 if any(r[0] == "FAIL" for r in rows) else 0)


# ---------------------------------------------------------------------------------------------------- CLI journeys
def tree(root: Path) -> dict:
    out = {}
    for dp, _, files in os.walk(root):
        for fn in files:
            p = Path(dp) / fn
            st_ = p.stat()
            out[p.relative_to(root).as_posix()] = (st_.st_size, st_.st_mtime_ns)
    return out


def cmd_cli(a) -> None:
    run = run_dir(a.run)
    args = a.args[1:] if a.args and a.args[0] == "--" else a.args
    if not args:
        sys.exit("give the lanegame arguments after --, e.g. `cli --run RUN --label solve -- solve --horizon 8`")
    ev = run / "evidence"
    if (ev / f"{a.label}.json").exists():
        sys.exit(f"label {a.label!r} already used in this run; evidence is never overwritten")
    work = run / "work" / "cli"
    before = tree(work)
    cmd = [str(VENV_PY), "-m", "lanegame", *args]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    t0 = time.perf_counter()
    try:
        r = subprocess.run(cmd, cwd=work, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           env=env, timeout=a.timeout)
        code, out, err = r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired as e:
        code, out, err = "timeout", (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""), \
            f"killed after {a.timeout}s"
    secs = round(time.perf_counter() - t0, 2)
    after = tree(work)
    created = []
    for rel, (size, _) in sorted(after.items()):
        if before.get(rel) != after[rel]:
            p = work / rel
            rec = {"path": rel, "size": size, "sha256": sha256(p), "copied": size <= COPY_LIMIT}
            if rec["copied"]:
                dst = ev / f"{a.label}.files" / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dst)
            created.append(rec)
    changed = repo_guard(run)
    shown = "python -m lanegame " + subprocess.list2cmdline(args)
    (ev / f"{a.label}.cmd.txt").write_text(f"{shown}\ncwd {work}\ngit {git('rev-parse', 'HEAD').strip()}\n"
                                          f"started {now()}\n", encoding="utf-8")
    (ev / f"{a.label}.stdout.txt").write_text(out, encoding="utf-8")
    (ev / f"{a.label}.stderr.txt").write_text(err, encoding="utf-8")
    (ev / f"{a.label}.json").write_text(json.dumps(
        {"label": a.label, "command": shown, "cwd": str(work), "exit_code": code, "seconds": secs,
         "created_or_changed": created, "repo_changed": changed}, indent=1), encoding="utf-8")
    log(run, f"cli {a.label}: {shown} -> exit {code} in {secs}s, {len(created)} file(s) written"
             + (f"; REPO CHANGED: {changed}" if changed else ""))
    tail = out.rstrip().splitlines()[-a.tail:] if a.tail > 0 else []
    print(f"$ {shown}\nexit {code} in {secs}s (cwd {work})")
    print("\n".join(tail))
    if err.strip():
        print("--- stderr (tail)\n" + "\n".join(err.rstrip().splitlines()[-5:]))
    for c in created:
        print(f"wrote {c['path']} ({c['size']} bytes, sha256 {c['sha256'][:16]}...)")
    print(f"evidence: {ev / a.label}.*")
    if changed:
        print(f"REPO CHANGED during this command: {changed}")
        sys.exit(3)
    sys.exit(code if isinstance(code, int) else 124)


# ---------------------------------------------------------------------------------------------------- viewer
def cmd_viewer(a) -> None:
    run = run_dir(a.run)
    if a.action == "stop":
        v = state(run, "viewer")
        if v:
            kill_tree(v["pid"])
            set_state(run, "viewer", None)
            log(run, f"viewer stop pid {v['pid']}")
            print(f"stopped viewer pid {v['pid']}")
        else:
            print("no viewer recorded for this run")
        return
    v = state(run, "viewer")
    if v and pid_alive(v["pid"]):
        sys.exit(f"this run already has a viewer at {v['url']} (pid {v['pid']})")
    port = a.port or next((p for p in VIEWER_PORTS if port_free(p)), None)
    if port is None or not port_free(port):
        sys.exit(f"port {port} is busy; pick another with --port")
    cmd = [str(VENV_PY), "-m", "streamlit", "run", "lanegame/viewer/app.py", "--server.headless", "true",
           "--server.address", "127.0.0.1", "--server.port", str(port), "--browser.gatherUsageStats", "false",
           "--server.fileWatcherType", "none", "--server.runOnSave", "false"]
    pid = spawn(cmd, REPO, run / "evidence" / "viewer.log")
    url = f"http://127.0.0.1:{port}/"
    set_state(run, "viewer", {"pid": pid, "port": port, "url": url, "started": now()})
    t0 = time.time()
    while time.time() - t0 < a.timeout:
        if not pid_alive(pid):
            sys.exit(f"viewer exited during start; see {run / 'evidence' / 'viewer.log'}")
        try:
            if http_get(f"{url}_stcore/health", 2).strip() == "ok" and port_owner_ok(pid, port):
                break
        except Exception:                                            # noqa: BLE001
            pass
        time.sleep(0.5)
    else:
        sys.exit(f"viewer not healthy after {a.timeout}s; see {run / 'evidence' / 'viewer.log'}")
    log(run, f"viewer start pid {pid} {url}")
    print(f"viewer up at {url} (pid {pid}); server healthy. Open it with `browser open` to confirm the page renders.")


def cmd_demo(a) -> None:
    """The M6 web demo: `python -m lanegame.demo` on a free port in 8700-8799, from the repo root (it reads the
    checkpoints relative to it and serves web/dist); ready when /api/status answers (ready or not: without the
    checkpoints Play says the engine is unavailable)."""
    run = run_dir(a.run)
    if a.action == "stop":
        v = state(run, "demo")
        if v:
            kill_tree(v["pid"])
            set_state(run, "demo", None)
            log(run, f"demo stop pid {v['pid']}")
            print(f"stopped demo pid {v['pid']}")
        else:
            print("no demo recorded for this run")
        return
    v = state(run, "demo")
    if v and pid_alive(v["pid"]):
        sys.exit(f"this run already has a demo at {v['url']} (pid {v['pid']})")
    port = a.port or next((p for p in DEMO_PORTS if port_free(p)), None)
    if port is None or not port_free(port):
        sys.exit(f"port {port} is busy; pick another with --port")
    if not (REPO / "web" / "dist" / "index.html").is_file():
        sys.exit("the frontend is not built: cd web && npm install && npm run build")
    cmd = [str(VENV_PY), "-m", "lanegame.demo", "--port", str(port)]
    pid = spawn(cmd, REPO, run / "evidence" / "demo.log")
    url = f"http://127.0.0.1:{port}/"
    set_state(run, "demo", {"pid": pid, "port": port, "url": url, "started": now()})
    t0 = time.time()
    st = {}
    while time.time() - t0 < a.timeout:
        if not pid_alive(pid):
            sys.exit(f"demo exited during start; see {run / 'evidence' / 'demo.log'}")
        try:
            st = json.loads(http_get(f"{url}api/status", 2))
            if port_owner_ok(pid, port):
                break
        except Exception:                                            # noqa: BLE001
            pass
        time.sleep(0.5)
    else:
        sys.exit(f"demo not answering after {a.timeout}s; see {run / 'evidence' / 'demo.log'}")
    engine = "ready" if st.get("ready") else f"unavailable: {st.get('error')}"
    log(run, f"demo start pid {pid} {url} engine {engine}")
    print(f"demo up at {url} (pid {pid}); engine {engine}. Open it with `browser open --run RUN` "
          "(it waits for the page's ready flag).")


# ---------------------------------------------------------------------------------------------------- browser (CDP)
class CDP:
    """A DevTools session on the browser's first page; use it as `with CDP(port) as c:`."""

    def __init__(self, port: int):
        from websockets.sync.client import connect
        pages = [t for t in json.loads(http_get(f"http://127.0.0.1:{port}/json/list")) if t["type"] == "page"]
        if not pages:
            raise SystemExit("the browser has no page target")
        self.conn = connect(pages[0]["webSocketDebuggerUrl"], max_size=None, open_timeout=10)
        self.n = 0

    def __enter__(self):
        self.ws = self.conn.__enter__()
        return self

    def __exit__(self, *exc):
        return self.conn.__exit__(*exc)

    def call(self, method: str, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv(timeout=60))
            if msg.get("id") == self.n:
                if "error" in msg:
                    raise SystemExit(f"{method}: {msg['error']}")
                return msg.get("result", {})

    def js(self, expr: str):
        r = self.call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        if "exceptionDetails" in r:
            raise SystemExit(f"page script failed: {r['exceptionDetails'].get('exception', {}).get('description', r)}")
        return r["result"].get("value")

    def mouse_click(self, x: float, y: float) -> None:
        self.call("Input.dispatchMouseEvent", type="mouseMoved", x=x, y=y)
        self.call("Input.dispatchMouseEvent", type="mousePressed", x=x, y=y, button="left", clickCount=1)
        self.call("Input.dispatchMouseEvent", type="mouseReleased", x=x, y=y, button="left", clickCount=1)

    def key(self, key: str, code: str, vk: int, modifiers: int = 0, text: str = "") -> None:
        down = {"type": "keyDown", "key": key, "code": code, "windowsVirtualKeyCode": vk, "modifiers": modifiers}
        if text:
            down["text"] = text
        self.call("Input.dispatchKeyEvent", **down)
        self.call("Input.dispatchKeyEvent", type="keyUp", key=key, code=code, windowsVirtualKeyCode=vk,
                  modifiers=modifiers)


# Streamlit handles: widgets created with key=K sit in a container with class st-key-K; the sidebar and main area
# have data-testid stSidebar / stMain; elements awaiting a rerun carry data-stale="true".
JS_SCOPE = """
function lvScope(scope, key) {
  if (key) return document.querySelector('.st-key-' + CSS.escape(key));
  if (scope === 'sidebar') return document.querySelector('[data-testid="stSidebar"]');
  if (scope === 'main') return document.querySelector('[data-testid="stMain"]');
  return document.body;
}
function lvVisible(el) { const r = el.getBoundingClientRect(); return r.width > 0 && r.height > 0; }
"""
JS_STATE = JS_SCOPE + """
(() => {
  const status = document.querySelector('[data-testid="stStatusWidget"]');
  const running = !!(status && /running/i.test(status.innerText || status.getAttribute('aria-label') || ''));
  const stale = document.querySelectorAll('[data-stale="true"]').length;
  const exc = [...document.querySelectorAll('[data-testid="stException"]')].map(e => e.innerText.slice(0, 300));
  return {running, stale, exc, text: document.body.innerText.length + ':' + document.body.innerText.slice(-2000)};
})()
"""


def settle(c: CDP, timeout: float = 30.0) -> float:
    """Wait until Streamlit has finished the rerun: no running indicator, no stale elements, and the page text unchanged
    over two consecutive polls."""
    t0, last, quiet = time.time(), None, 0
    time.sleep(0.25)
    while time.time() - t0 < timeout:
        s = c.js(JS_STATE)
        if s["exc"]:
            raise SystemExit("the app raised an exception:\n" + "\n".join(s["exc"]))
        if not s["running"] and not s["stale"] and s["text"] == last:
            quiet += 1
            if quiet >= 2:
                return round(time.time() - t0, 2)
        else:
            quiet = 0
        last = s["text"]
        time.sleep(0.25)
    raise SystemExit(f"the page did not settle within {timeout}s")


JS_WIDGETS = """
(() => {
  const norm = s => (s || '').replace(/\\s+/g, ' ').trim(), out = {};
  for (const el of document.querySelectorAll('[class*="st-key-"]')) {
    const key = [...el.classList].find(x => x.startsWith('st-key-')).slice(7);
    let v = null;
    if (el.querySelector('[data-testid="stRadio"]')) {
      const s = el.querySelector('[data-testid="stRadioOption"][data-selected="true"]'); v = s ? norm(s.innerText) : null;
    } else if (el.querySelector('input[type=checkbox]')) v = el.querySelector('input[type=checkbox]').checked;
    else if (el.querySelector('input[type=range]')) v = [...el.querySelectorAll('input[type=range]')].map(i => i.value).join(',');
    else { const i = el.querySelector('input, textarea'); v = i ? i.value : null; }
    out[key] = v;
  }
  return out;
})()
"""


def widgets(c: CDP) -> dict:
    """The current value of every keyed widget (selectbox and number values are not in the page's innerText)."""
    return c.js(JS_WIDGETS)


def page_text(c: CDP, scope: str | None, key: str | None = None) -> str:
    return c.js(JS_SCOPE + f"(() => {{ const s = lvScope({json.dumps(scope)}, {json.dumps(key)}); "
                           "return s ? s.innerText : null; })()")


def find_and_center(c: CDP, text: str, scope: str | None, key: str | None, contains: bool, nth: int) -> dict:
    expr = JS_SCOPE + f"""
(() => {{
  const root = lvScope({json.dumps(scope)}, {json.dumps(key)});
  if (!root) return {{error: 'scope not found'}};
  const want = {json.dumps(text)}, contains = {json.dumps(contains)};
  const sel = 'button, [role="tab"], [role="option"], [role="radio"], label, a, summary, [data-testid="stExpander"] summary';
  const norm = s => (s || '').replace(/\\s+/g, ' ').trim();
  const hits = [...root.querySelectorAll(sel)].filter(el => lvVisible(el) &&
     (contains ? norm(el.innerText).includes(want) : norm(el.innerText) === want));
  // prefer the innermost match (a label inside a button, a tab's own element)
  const inner = hits.filter(el => !hits.some(o => o !== el && el.contains(o)));
  if (!inner.length) return {{error: 'no visible element with that text', candidates:
     [...root.querySelectorAll('button, [role="tab"], label')].filter(lvVisible).map(e => norm(e.innerText)).filter(Boolean).slice(0, 40)}};
  const el = inner[Math.min({nth}, inner.length - 1)];
  el.scrollIntoView({{block: 'center', inline: 'center'}});
  const r = el.getBoundingClientRect();
  const btn = el.closest('button');
  return {{x: r.left + r.width / 2, y: r.top + r.height / 2, tag: el.tagName.toLowerCase(),
           role: el.getAttribute('role'), disabled: !!(btn && btn.disabled), count: inner.length}};
}})()"""
    return c.js(expr)


def browser(run: Path) -> CDP:
    b = state(run, "browser")
    if not b or not pid_alive(b["pid"]):
        sys.exit("no browser running for this run; `vl.py browser start --run RUN` first")
    return CDP(b["port"])


def cmd_browser(a) -> None:
    run = run_dir(a.run)
    ev = run / "evidence"
    if a.action == "start":
        b = state(run, "browser")
        if b and pid_alive(b["pid"]):
            sys.exit(f"this run already has a browser (pid {b['pid']})")
        exe = next((x for x in BROWSERS if x and Path(x).is_file()), None)
        if not exe:
            sys.exit("no Edge or Chrome found; set LANEGAME_VERIFY_BROWSER to a Chromium browser executable")
        profile = run / "work" / "browser-profile"
        profile.mkdir(parents=True, exist_ok=True)
        (profile / "DevToolsActivePort").unlink(missing_ok=True)
        cmd = [exe, "--headless=new", "--remote-debugging-port=0", f"--user-data-dir={profile}", "--no-first-run",
               "--no-default-browser-check", "--disable-extensions", "--disable-sync", "--disable-features=Translate",
               f"--window-size={a.width},{a.height}", "--force-device-scale-factor=1", "about:blank"]
        pid = spawn(cmd, run / "work", ev / "browser.log")
        t0 = time.time()
        while not (profile / "DevToolsActivePort").is_file():
            if time.time() - t0 > 30 or not pid_alive(pid):
                kill_tree(pid)
                sys.exit(f"browser did not open its DevTools port; see {ev / 'browser.log'}")
            time.sleep(0.2)
        port = int((profile / "DevToolsActivePort").read_text().split()[0])
        set_state(run, "browser", {"pid": pid, "port": port, "exe": exe, "started": now()})
        log(run, f"browser start pid {pid} devtools {port} ({Path(exe).name}, headless {a.width}x{a.height})")
        print(f"headless browser up (pid {pid}, DevTools port {port})")
        return
    if a.action == "stop":
        b = state(run, "browser")
        if b:
            kill_tree(b["pid"])
            set_state(run, "browser", None)
            log(run, f"browser stop pid {b['pid']}")
            print(f"stopped browser pid {b['pid']}")
        else:
            print("no browser recorded for this run")
        return

    with browser(run) as c:
        page_action(a, run, ev, c)


def page_action(a, run: Path, ev: Path, c: CDP) -> None:
    if a.action == "js":
        v = c.js(a.expr)
        if a.label:
            (ev / f"{a.label}.json").write_text(json.dumps(v, indent=1, ensure_ascii=False), encoding="utf-8")
            log(run, f"browser js {a.expr[:80]!r} -> {a.label}.json")
        else:
            log(run, f"browser js {a.expr[:80]!r}")
        print(json.dumps(v, indent=1, ensure_ascii=False))
        return
    if a.action == "open":
        v = state(run, "viewer")
        d = state(run, "demo")
        url = a.url or (v and v["url"]) or (d and d["url"])
        if not url:
            sys.exit("no --url and no viewer or demo started in this run")
        if not a.url and v and not port_owner_ok(v["pid"], v["port"]):
            sys.exit("the recorded viewer port is not owned by this run's process; run doctor")
        c.call("Page.navigate", url=url)
        t0 = time.time()
        demo_page = d and url.startswith(d["url"])
        ready_js = a.ready_js or ("document.body && document.body.dataset.ready === '1'" if demo_page else None)
        if ready_js:
            while not c.js(ready_js):
                if time.time() - t0 > a.timeout:
                    sys.exit(f"page not ready within {a.timeout}s ({ready_js})")
                time.sleep(0.2)
            log(run, f"browser open {url} (ready: {ready_js})")
            print(f"opened {url}; ready after {round(time.time() - t0, 1)}s")
            return
        while a.wait_text not in (page_text(c, None) or ""):
            if time.time() - t0 > a.timeout:
                sys.exit(f"{a.wait_text!r} did not appear within {a.timeout}s")
            time.sleep(0.3)
        s = settle(c, a.timeout)
        log(run, f"browser open {url} (saw {a.wait_text!r}, settled {s}s)")
        print(f"opened {url}; {a.wait_text!r} rendered, settled in {s}s")
    elif a.action == "click":
        hit = find_and_center(c, a.text, a.scope, a.key, a.contains, a.nth)
        if "error" in hit:
            sys.exit(f"click {a.text!r}: {hit['error']}" + (f"; visible: {hit['candidates']}" if hit.get("candidates")
                                                            else ""))
        if hit["disabled"]:
            sys.exit(f"click {a.text!r}: the button is disabled")
        c.mouse_click(hit["x"], hit["y"])
        s = "skipped" if a.no_settle else settle(c, a.timeout)
        where = f"key {a.key}" if a.key else (a.scope or "page")
        log(run, f"browser click {a.text!r} in {where} ({hit['tag']}{'/' + hit['role'] if hit['role'] else ''}, "
                 f"{hit['count']} match(es), took #{a.nth}); settled {s}s")
        print(f"clicked {a.text!r} ({hit['tag']}, {hit['count']} match(es)); settled in {s}s")
    elif a.action == "select":
        box = c.js(JS_SCOPE + f"""(() => {{
          const root = lvScope(null, {json.dumps(a.key)});
          const el = root && (root.querySelector('[data-baseweb="select"]') || root.querySelector('input'));
          if (!el) return null; el.scrollIntoView({{block: 'center'}});
          const r = el.getBoundingClientRect(); return {{x: r.left + r.width / 2, y: r.top + r.height / 2}};
        }})()""")
        if not box:
            sys.exit(f"select: no selectbox with key {a.key!r}")
        c.mouse_click(box["x"], box["y"])
        c.key("a", "KeyA", 65, modifiers=8 if sys.platform == "darwin" else 2)
        c.key("Backspace", "Backspace", 8)
        c.call("Input.insertText", text=a.option)               # Streamlit selectboxes filter as you type
        time.sleep(0.4)
        opt = c.js(f"""(() => {{
          const norm = s => (s || '').replace(/\\s+/g, ' ').trim();
          const opts = [...document.querySelectorAll('[role="option"]')];
          const el = opts.find(o => norm(o.innerText) === {json.dumps(a.option)});
          if (!el) return {{error: opts.map(o => norm(o.innerText)).slice(0, 30)}};
          el.scrollIntoView({{block: 'nearest'}});
          const r = el.getBoundingClientRect(); return {{x: r.left + r.width / 2, y: r.top + r.height / 2}};
        }})()""")
        if "error" in opt:
            c.key("Escape", "Escape", 27)
            sys.exit(f"select {a.key}: no option exactly {a.option!r}; options after filtering: {opt['error']}")
        c.mouse_click(opt["x"], opt["y"])
        s = settle(c, a.timeout)
        got = widgets(c).get(a.key)
        log(run, f"browser select key {a.key} -> {a.option!r} (widget now {got!r}); settled {s}s")
        print(f"selected {a.option!r} in {a.key} (widget now {got!r}); settled in {s}s")
        if got != a.option:
            sys.exit(f"select {a.key}: the widget shows {got!r}, not {a.option!r}")
    elif a.action == "fill":
        box = c.js(JS_SCOPE + f"""(() => {{
          const root = lvScope(null, {json.dumps(a.key)});
          const el = root && root.querySelector('input, textarea');
          if (!el) return null; el.scrollIntoView({{block: 'center'}});
          const r = el.getBoundingClientRect(); return {{x: r.left + r.width / 2, y: r.top + r.height / 2}};
        }})()""")
        if not box:
            sys.exit(f"fill: no input with key {a.key!r}")
        c.mouse_click(box["x"], box["y"])
        c.key("a", "KeyA", 65, modifiers=8 if sys.platform == "darwin" else 2)     # select all
        c.key("Backspace", "Backspace", 8)
        c.call("Input.insertText", text=a.value)
        c.key("Enter", "Enter", 13, text="\r")
        s = settle(c, a.timeout)
        got = c.js(JS_SCOPE + f"(() => {{ const r = lvScope(null, {json.dumps(a.key)}); "
                              "const el = r && r.querySelector('input, textarea'); return el ? el.value : null; })()")
        log(run, f"browser fill key {a.key} <- {a.value!r} (input now {got!r}); settled {s}s")
        print(f"filled {a.key} with {a.value!r} (input now {got!r}); settled in {s}s")
    elif a.action == "wait":
        t0 = time.time()
        while a.text not in (page_text(c, a.scope, a.key) or ""):
            if time.time() - t0 > a.timeout:
                sys.exit(f"{a.text!r} did not appear within {a.timeout}s")
            time.sleep(0.3)
        s = "skipped" if a.no_settle else settle(c, a.timeout)
        log(run, f"browser wait {a.text!r}: present after {round(time.time() - t0, 2)}s")
        print(f"{a.text!r} present; settled in {s}s")
    elif a.action == "text":
        t = page_text(c, a.scope, a.key)
        if t is None:
            sys.exit("scope not found")
        if a.label:
            (ev / f"{a.label}.txt").write_text(t, encoding="utf-8")
            log(run, f"browser text ({a.scope or a.key or 'page'}) -> {a.label}.txt")
        print(t)
    elif a.action == "widgets":
        w = widgets(c)
        if a.label:
            (ev / f"{a.label}.widgets.json").write_text(json.dumps(w, indent=1, ensure_ascii=False), encoding="utf-8")
            log(run, f"browser widgets -> {a.label}.widgets.json")
        print(json.dumps(w, indent=1, ensure_ascii=False))
    elif a.action == "screenshot":
        import base64
        settle(c, a.timeout)
        main = "document.querySelector('[data-testid=\"stMain\"]')"       # the scrolling pane (the page itself does not)
        if not a.here:
            c.js(f"(() => {{ const m = {main}; if (m) m.scrollTop = 0; }})()")
        if a.full:
            h = c.js(f"(() => {{ const m = {main}; return Math.max(innerHeight, m ? m.scrollHeight : 0); }})()")
            c.call("Emulation.setDeviceMetricsOverride", width=c.js("innerWidth"), height=int(h) + 40,
                   deviceScaleFactor=1, mobile=False)
            time.sleep(1.0)                                       # let the charts re-lay out at the new height
        try:
            png = base64.b64decode(c.call("Page.captureScreenshot", format="png")["data"])
        finally:
            if a.full:
                c.call("Emulation.clearDeviceMetricsOverride")
        (ev / f"{a.label}.png").write_bytes(png)
        (ev / f"{a.label}.txt").write_text(page_text(c, None) or "", encoding="utf-8")
        (ev / f"{a.label}.widgets.json").write_text(json.dumps(widgets(c), indent=1, ensure_ascii=False),
                                                   encoding="utf-8")
        log(run, f"browser screenshot -> {a.label}.png + .txt + .widgets.json")
        print(f"saved {ev / a.label}.png, .txt and .widgets.json")


# ---------------------------------------------------------------------------------------------------- cleanup
def cmd_cleanup(a) -> None:
    run = run_dir(a.run)
    for name in ("browser", "viewer", "demo"):
        s = state(run, name)
        if s:
            kill_tree(s["pid"])
            if pid_alive(s["pid"]):
                sys.exit(f"{name} pid {s['pid']} survived the kill; stop it by pid and rerun cleanup")
            log(run, f"cleanup: stopped {name} pid {s['pid']}")
    changed = repo_guard(run)
    (run / "evidence" / "repo_guard.json").write_text(json.dumps({"checked": now(), "changed": changed}, indent=1))
    log(run, f"cleanup: repo guard {'CHANGED ' + str(changed) if changed else 'clean'}; removing work/ and state/")
    for sub in ("work", "state"):
        shutil.rmtree(run / sub, ignore_errors=True)
    files = sorted(p.relative_to(run / "evidence").as_posix() for p in (run / "evidence").rglob("*") if p.is_file())
    print(f"evidence kept at {run / 'evidence'} ({len(files)} files):")
    for f in files:
        print("  " + f)
    if changed:
        print(f"REPO CHANGED during this run: {changed}")
        sys.exit(3)


def main() -> None:
    ap = argparse.ArgumentParser(prog="vl.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("new-run").set_defaults(fn=cmd_new_run)
    p = sub.add_parser("doctor")
    p.add_argument("--run")
    p.set_defaults(fn=cmd_doctor)
    p = sub.add_parser("cli")
    p.add_argument("--run", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--timeout", type=float, default=900)
    p.add_argument("--tail", type=int, default=15)
    p.add_argument("args", nargs=argparse.REMAINDER)
    p.set_defaults(fn=cmd_cli)
    p = sub.add_parser("viewer")
    p.add_argument("action", choices=["start", "stop"])
    p.add_argument("--run", required=True)
    p.add_argument("--port", type=int)
    p.add_argument("--timeout", type=float, default=60)
    p.set_defaults(fn=cmd_viewer)
    p = sub.add_parser("demo")
    p.add_argument("action", choices=["start", "stop"])
    p.add_argument("--run", required=True)
    p.add_argument("--port", type=int)
    p.add_argument("--timeout", type=float, default=120)
    p.set_defaults(fn=cmd_demo)
    p = sub.add_parser("browser")
    p.add_argument("action", choices=["start", "stop", "open", "click", "select", "fill", "wait", "text", "widgets",
                                      "screenshot", "js"])
    p.add_argument("--run", required=True)
    p.add_argument("--url")
    p.add_argument("--wait-text", default="Lane debugger")
    p.add_argument("--ready-js", help="open: wait until this JavaScript expression is truthy (the demo's default)")
    p.add_argument("--expr", help="js: the expression to evaluate (a promise is awaited)")
    p.add_argument("--text")
    p.add_argument("--key")
    p.add_argument("--scope", choices=["sidebar", "main"])
    p.add_argument("--contains", action="store_true")
    p.add_argument("--nth", type=int, default=0)
    p.add_argument("--option")
    p.add_argument("--value")
    p.add_argument("--label")
    p.add_argument("--full", action="store_true", help="screenshot: the whole page, not one screen")
    p.add_argument("--here", action="store_true", help="screenshot: keep the current scroll position (default: top)")
    p.add_argument("--no-settle", action="store_true",
                   help="click/wait: do not wait for the page to stop changing (needed while the viewer is playing)")
    p.add_argument("--timeout", type=float, default=30)
    p.add_argument("--width", type=int, default=1600)
    p.add_argument("--height", type=int, default=1100)
    p.set_defaults(fn=cmd_browser)
    p = sub.add_parser("cleanup")
    p.add_argument("--run", required=True)
    p.set_defaults(fn=cmd_cleanup)
    a = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):          # the viewer's text has ⏮ ▶ ·; a cp1252 console cannot print them
        stream.reconfigure(encoding="utf-8", errors="replace")
    a.fn(a)


if __name__ == "__main__":
    main()
