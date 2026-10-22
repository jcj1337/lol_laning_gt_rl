"""The complete visitor journey of the M6b brief, driven in a headless browser against the real server (browser.py).

Every check pairs an action a visitor would take (a real click, a key press, a reload, a download) with the state the
app reports (window.__demo.state(), the app's own hook) and, where the engine is the authority, with what the server
says (its history, re-simulated from the codes). The checks are written to results/m6/m6b_browser.json and the
screenshots to results/m6/m6b/. Nothing here changes a recording or a result of an earlier milestone.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from typing import List, Optional

from ...config import champ_act
from ..m45b.common import git_commit, git_dirty
from .browser import REPO, Browser, Page, Server

OUT = os.path.join("results", "m6")
SHOTS = os.path.join(OUT, "m6b")
SIZES = [(1440, 900, "desktop"), (1024, 700, "small"), (844, 390, "phone_landscape"), (915, 412, "phone_landscape_tall"), (390, 844, "phone_portrait")]


class Journey:
    def __init__(self, srv: Server, b: Browser):
        self.srv, self.b = srv, b
        self.checks: List[dict] = []
        self.shots: List[str] = []
        self.page: Optional[Page] = None

    # ---- bookkeeping
    def check(self, name: str, ok: bool, detail="") -> bool:
        self.checks.append({"check": name, "ok": bool(ok), "detail": detail})
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not ok else ""), flush=True)
        return bool(ok)

    def shot(self, name: str, **kw) -> None:
        p = self.page.shot(os.path.join(SHOTS, f"{name}.png"), **kw)
        self.shots.append(p.replace("\\", "/"))

    def no_errors(self, where: str) -> None:
        errs = self.page.errors()
        self.check(f"{where}: no page errors", not errs, "; ".join(errs))

    def server_history(self, sid: str, a: int, b: Optional[int] = None) -> dict:
        q = f"?from={a}" + (f"&to={b}" if b is not None else "")
        return self.srv.api("GET", f"/api/sessions/{sid}/history{q}")["body"]

    def client_game(self) -> dict:
        return self.page.js("JSON.parse(JSON.stringify({positions: window.__demo.store.game.positions, turns: window.__demo.store.game.turns}))")

    # ---- the live game, through the real controls
    def play_turn(self, keys: List[str], expect_progress: int, step_manually: bool = False) -> dict:
        """Choose the actions by clicking their buttons, press the submit button, and wait until the turn is shown."""
        p = self.page
        for k in keys:
            p.click(f'[data-k="{k}"]')
        p.click("#btnSubmit")
        if step_manually:
            self.shot("02a_both_actions_revealed")
            heads = [p.state()["step"]["head"]]
            p.click("#btnTick"); heads.append(p.state()["step"]["head"])
            self.shot("02b_tick_1")
            p.click("#btnTick"); heads.append(p.state()["step"]["head"])
            p.click("#btnTick"); heads.append(p.state()["step"]["head"])
            self.shot("02c_end_of_turn")
            p.click("#btnTick")
            self.check("a turn steps through start, tick 1, tick 2 and the end", heads == ["Both actions revealed", "Tick 1 of 2", "Tick 2 of 2", f"End of turn {expect_progress}"], str(heads))
        p.wait(f"(() => {{ const s = window.__demo.state(); return s.progress === {expect_progress} && !s.step && !s.busy; }})()", 60, f"turn {expect_progress}")
        return p.state()

    def part_play(self) -> str:
        srv, b = self.srv, self.b
        self.page = p = b.page(f"{srv.url}/?fresh&ticks=manual", 1440, 900)
        st = p.state()
        sid = st["sid"]
        self.check("1. a live game starts against the self-play agent", st["live"] and st["mode"] == "play" and st["progress"] == 0 and st["canSubmit"] and st["kind"] == "human_game", json.dumps({k: st[k] for k in ("live", "mode", "progress", "canSubmit", "kind")}))
        self.check("1. the opponent is shown as having chosen, without what", p.js("document.querySelector('.h-list .pending .b')?.textContent") == "Chosen" and st["menu"] is not None)
        self.check("1. the view carries no secret seed and no belief while active", st["seeds"]["chance"] is None and st["seeds"]["policy"] is None and not st["beliefs"])
        self.shot("01_play_start")
        self.no_errors("start")
        # 2. several legal actions through the interface; the first stepped by hand
        st = self.play_turn(["w3", "b0"], 1, step_manually=True)
        self.check("2. the first submitted action resolved as chosen (Slow push, Hold = 46)", st["turns"] == [[46, st["turns"][0][1]]] and st["turns"][0][0] == 46, str(st["turns"]))
        p.js("window.__demo.ticks('auto')")
        codes = [46]
        for keys, (w, bh), t in ((["w1", "b2"], (1, 2), 2), (["w0", "b3"], (0, 3), 3), (["w2", "b1"], (2, 1), 4), (["w1", "b0"], (1, 0), 5)):
            code = champ_act(w, bh, 0)                      # the engine's own code for that wave and behaviour
            codes.append(code)
            st = self.play_turn(keys, t)
            self.check(f"2. turn {t} resolved with the chosen code {code}", st["turns"][t - 1][0] == code, str(st["turns"]))
        h = self.server_history(sid, 0, 5)
        cg = self.client_game()
        self.check("2. the client's positions and turns equal the server's re-simulated history", cg["positions"] == h["positions"] and cg["turns"] == h["turns"])
        self.check("2. the server resolved exactly the turns the interface submitted", [t["codes"][0] for t in h["turns"]] == codes, str([t["codes"] for t in h["turns"]]))
        self.shot("02d_after_five_turns")
        # 3. a trade outcome: keep trading until the engine records one, then inspect it in Details
        seen, t = set(), 5
        while t < 40 and len(seen) < 3:
            m = p.state()["menu"]
            if m["forced"] is not None:
                p.click("#btnAway")
                p.wait("(() => { const s = window.__demo.state(); return s.menu && s.menu.forced === null && !s.busy; })()", 60, "away turns")
                t = p.state()["progress"]
                continue
            st = self.play_turn(["w1", "b2"], t + 1)
            t += 1
            tr = cg = None
            for e in p.js(f"window.__demo.store.game.turns[{t - 1}].outcomes"):
                if e["kind"] == "trade":
                    seen.add(e["data"]["result"])
                    tr = e
            if tr and "03_trade_details" not in " ".join(self.shots):
                p.js("window.__demo.details('events')")
                p.wait("!!document.querySelector('#details .ev')", 10)
                self.shot("03_trade_details")
                text = p.js("document.querySelector('#details .ev').textContent")
                self.check("3. a Trade outcome is shown with its result in Details", any(w in text for w in ("hit", "missed", "could not reach")), text[:200])
                p.js("window.__demo.store.closeDetails()")
        self.check("3. trade results observed in this game (no contact, miss and hit are the three the engine distinguishes)", len(seen) >= 1, f"seen {sorted(seen)} by turn {t}")
        self.trade_results = sorted(seen)
        # 4. review an earlier turn and return
        prog = p.state()["progress"]
        p.click('.h-list li[data-p="2"]')
        st = p.state()
        self.check("4. clicking turn 2 shows position 2 and changes nothing", st["view"] == 2 and st["progress"] == prog and not st["canSubmit"], json.dumps({k: st[k] for k in ("view", "progress", "canSubmit")}))
        banner = p.js("document.getElementById('bannerText').textContent")
        self.check("4. the banner names the viewed turn and the game's turn", banner == f"Viewing turn 2 · the game is at turn {prog + 1}", banner)
        self.check("4. the submit control is replaced by Return while browsing", p.js("!!document.getElementById('btnBack') && !document.getElementById('btnSubmit')"))
        lead_shown = p.js("(document.querySelector('#graph .g-sel text') || document.getElementById('gVal') || {}).textContent")
        self.check("4. the gold graph marks the viewed position", (lead_shown or "").startswith("Turn 2:"), lead_shown or "")
        self.check("4. the status chips follow the viewed position (the lane shows position 2)", p.js("window.__demo.store.ctrl.frame.p") == 2)
        self.shot("04_review_turn_2")
        p.click("#btnReturn")
        st = p.state()
        self.check("4. Return to current turn restores the decision point", st["view"] == prog and st["canSubmit"], json.dumps({k: st[k] for k in ("view", "progress", "canSubmit")}))
        self.check("4. the server's position did not move while browsing", self.srv.api("GET", f"/api/sessions/{sid}")["body"]["position"] == prog)
        # 5. reload and recover
        before = self.client_game()
        p.reload()                                            # a plain reload: the app dropped `fresh` from the address
        p.js("window.__demo.ticks('auto')")                   # the address still says ticks=manual; the journey plays on
        st = p.state()
        self.check("5. a reload resumes the same session at the same turn", st["sid"] == sid and st["progress"] == prog and st["recovered"] == "ok", json.dumps({k: st[k] for k in ("sid", "progress", "recovered")}))
        self.check("5. the rebuilt history equals what the client had before the reload", self.client_game() == before)
        self.shot("05_after_reload")
        self.no_errors("after reload")
        # 6. stop and download
        dl = tempfile.mkdtemp(prefix="m6b-download-")
        p.downloads(dl)
        p.click("#btnStop")
        p.wait("window.__demo.state().stopped === true", 30, "stop")
        st = p.state()
        summary = p.js("document.getElementById('stopSummary').textContent")
        self.check("6. stopping declares no winner and reveals the seeds", "declares no winner" in summary and st["seeds"]["chance"] is not None and st["seeds"]["policy"] is not None, summary[:120])
        self.check("6. the belief history is absent for a self-play opponent", not st["beliefs"])
        self.shot("06_stopped")
        p.click("#btnDownload")
        t0 = time.time()
        files = []
        while time.time() - t0 < 15 and not files:
            files = [f for f in os.listdir(dl) if f.endswith(".json")]
            time.sleep(0.2)
        self.check("6. the record downloads as a JSON file", bool(files), str(os.listdir(dl)))
        self.downloaded = os.path.join(dl, files[0]) if files else None
        self.live_sid, self.live_progress = sid, prog
        return sid

    def part_replay_download(self) -> None:
        if not self.downloaded:
            self.check("7. the downloaded record replays bit for bit in Python", False, "no file")
            return
        from ...demo import session as S
        from ...demo.registry import Registry
        with open(self.downloaded, encoding="utf-8") as f:
            rec = json.load(f)
        reg = Registry().load()
        r = S.replay(reg, rec)
        self.check("7. the downloaded record replays bit for bit in Python", r["turns"] == self.live_progress and r["digests_checked"] == self.live_progress and r["codes_recomputed"] == ["B"], json.dumps(r))
        self.check("7. the record is a human game with its seats, mode and marks", rec["provenance"]["kind"] == "human_game" and rec["seats"][0]["kind"] == "human" and rec["seats"][1]["id"] == "sp:0" and rec["belief_mode"] == "hidden")
        shutil.copy(self.downloaded, os.path.join(SHOTS, "downloaded_record.json"))

    def part_watch(self) -> None:
        p = self.page
        p.click("#modeWatch")
        p.wait("window.__demo.state().kind === 'research_record'", 30, "a research replay")
        st = p.state()
        self.check("8. Watch opens a verified research recording, labelled as such", st["mode"] == "watch" and st["name"] == "h2h_om_s0_vs_sp_s0" and p.js("document.getElementById('recType').textContent") == "Recorded research game" and st["progress"] == 150)
        self.shot("07_watch_research")
        sims = self.srv.wait_simulations()
        self.check("8. the simulation worker process is ready", sims == "ready", sims)
        p.click("#btnSim")
        p.wait("document.getElementById('newSim').open", 5)
        p.click_text("Simulate", within="#newSim")
        p.wait("window.__demo.state().kind === 'new_simulation'", 150, "a generated match")
        st = p.state()
        self.check("8. a generated match is labelled as such and holds 150 turns", st["kind"] == "new_simulation" and st["progress"] == 150 and st["view"] == 0 and p.js("document.getElementById('recType').textContent") == "Generated game")
        rec = p.js("JSON.parse(JSON.stringify(window.__demo.store.game.recording))")
        from ...demo import session as S
        from ...demo.registry import Registry
        r = S.replay(Registry().load(), rec)
        self.check("8. the generated match's record replays bit for bit in Python", r["turns"] == 150 and r["codes_recomputed"] == ["A", "B"], json.dumps(r))
        p.click("#btnPlay")
        time.sleep(2.6)
        st = p.state()
        self.check("8. Play advances the match", st["playing"] and st["view"] >= 1, json.dumps({k: st[k] for k in ("playing", "view")}))
        p.click("#btnPlay")
        st = p.state()
        self.check("8. Pause stops it", not st["playing"])
        p.click('[data-s="4"]')
        self.check("8. the playback speed changes", p.state()["speed"] == 4)
        p.click("#btnLast")
        self.check("8. Last shows the final position", p.state()["view"] == 150)
        p.click("#btnFirst")
        self.check("8. First shows the start", p.state()["view"] == 0)
        p.click("#btnNext")
        time.sleep(1.2)
        self.check("8. Next shows the next turn", p.state()["view"] == 1)
        p.click("#btnPrev")
        self.check("8. Previous steps back", p.state()["view"] == 0)
        p.js("window.__demo.history(true)")
        p.wait("document.body.classList.contains('hist-open')", 5)
        p.click('.h-list li[data-p="40"]')
        self.check("8. clicking a history row shows that turn", p.state()["view"] == 40)
        self.shot("08a_watch_history_open")
        p.js("window.__demo.history(false)")
        r = p.rect("#graph")
        p.c.call("Input.dispatchMouseEvent", type="mouseMoved", x=r["x"] + r["w"] * 0.6, y=r["y"] + r["h"] / 2)
        time.sleep(0.2)
        p.c.call("Input.dispatchMouseEvent", type="mousePressed", x=r["x"] + r["w"] * 0.6, y=r["y"] + r["h"] / 2, button="left", clickCount=1)
        p.c.call("Input.dispatchMouseEvent", type="mouseReleased", x=r["x"] + r["w"] * 0.6, y=r["y"] + r["h"] / 2, button="left", clickCount=1)
        time.sleep(0.2)
        v = p.state()["view"]
        self.check("8. clicking the gold graph jumps to a turn", 60 <= v <= 120, str(v))
        self.shot("08b_watch_generated")
        p.js("window.__demo.graph(true)")
        time.sleep(0.4)
        self.shot("08c_watch_graph_expanded")
        p.js("window.__demo.graph(false)")
        p.click("#btnPlay")
        time.sleep(1.0)
        self.check("8. Play restarts playback from the viewed position", p.state()["playing"])
        p.click("#btnPlay")
        self.no_errors("watch")

    def part_absence(self) -> None:
        p = self.page
        p.js("window.__demo.ticks('auto'); window.__demo.newGame('sp:0', 'hidden')")
        p.wait("(() => { const s = window.__demo.state(); return s.live && s.progress === 0 && s.canSubmit; })()", 30, "a new game")
        sid = p.state()["sid"]
        p.click('[data-k="recall"]')
        self.check("9. Recall is offered at the start and selected on click", p.js("document.getElementById('btnSubmit').textContent") == "Play turn 1: Recall")
        p.click("#btnSubmit")
        p.wait("(() => { const s = window.__demo.state(); return s.progress === 1 && !s.step && !s.busy; })()", 60, "the Recall turn")
        m = p.state()["menu"]
        self.check("9. after a Recall the only action is Absent, and the choices say why", m["forced"] == 45 and m["away"]["status"] == "recall" and p.js("!!document.getElementById('btnAway')") and p.js("document.querySelectorAll('#controls .opt[aria-disabled=\"true\"]').length") == 9)
        self.shot("09_away_turn")
        p.click("#btnAway")
        p.wait("(() => { const s = window.__demo.state(); return s.progress >= 2 && s.menu && s.menu.forced === null && !s.busy; })()", 60, "Continue")
        st = p.state()
        self.check("9. Continue plays the away turn as Absent and the champion returns", st["progress"] == 2 and st["turns"][1][0] == 45 and st["canSubmit"], str(st["turns"]))
        self.absence_sid = sid

    def part_duplicates(self) -> None:
        p, sid = self.page, self.absence_sid
        st = p.state()
        t, code = st["progress"] + 1, st["menu"]["legal"][0]
        res = p.js(f"""(async () => {{
            const body = JSON.stringify({{turn: {t}, code: {code}}});
            const h = {{method: 'POST', headers: {{'Content-Type': 'application/json'}}, body}};
            const [a, b] = await Promise.all([fetch('/api/sessions/{sid}/turns', h), fetch('/api/sessions/{sid}/turns', h)]);
            const [ja, jb] = await Promise.all([a.json(), b.json()]);
            const c = await fetch('/api/sessions/{sid}/turns', {{...h, body: JSON.stringify({{turn: {t}, code: {code} === 2 ? 3 : 2}})}});
            const v = await (await fetch('/api/sessions/{sid}')).json();
            return {{a: a.status, b: b.status, same: JSON.stringify(ja.turn) === JSON.stringify(jb.turn), pa: ja.view.position, pb: jb.view.position, conflict: c.status, position: v.position}};
        }})()""")
        self.check("10. two simultaneous identical submissions resolve the turn once", res["a"] == 200 and res["b"] == 200 and res["same"] and res["pa"] == res["pb"] == t, json.dumps(res))
        self.check("10. a conflicting resubmission is refused with 409", res["conflict"] == 409 and res["position"] == t, json.dumps(res))
        # the interface had not seen that turn: its next submission (another action) conflicts, and it catches up
        p.click('[data-k="w1"]' if code != champ_act(1, 0, 0) else '[data-k="w0"]'); p.click('[data-k="b0"]'); p.click("#btnSubmit")
        p.wait(f"(() => {{ const s = window.__demo.state(); return s.progress === {t} && !s.busy; }})()", 30, "catch-up")
        st = p.state()
        self.check("10. a submission the server had already resolved differently is reported, and the client catches up", st["progress"] == t and st["turns"][t - 1][0] == code and "already resolved" in (st["error"] or ""), json.dumps({k: st[k] for k in ("progress", "error")}))

    def part_unanswered(self) -> None:
        p = self.page
        p.js("window.__realFetch = window.__realFetch || window.fetch")
        st = p.state()
        t = st["progress"] + 1
        # the request never reaches the server: nothing changes, and a retry is offered
        p.js("window.fetch = (u, o) => (String(u).includes('/turns') && !window.__dropped) ? (window.__dropped = true, Promise.reject(new TypeError('Failed to fetch'))) : window.__realFetch(u, o)")
        p.click('[data-k="w1"]'); p.click('[data-k="b0"]'); p.click("#btnSubmit")
        p.wait("!!window.__demo.state().problem", 30, "the problem block")
        st = p.state()
        self.check("11. an unanswered submission leaves the game where it was and says so", st["problem"]["kind"] == "unanswered" and st["progress"] == t - 1 and not st["busy"], json.dumps(st["problem"]))
        self.shot("10_unanswered_request")
        p.click_text("Reconnect and retry", within="#controls")
        p.wait(f"(() => {{ const s = window.__demo.state(); return s.progress === {t} && !s.busy && !s.problem; }})()", 60, "the retry")
        self.check("11. the retry resolves the same action once", p.state()["turns"][t - 1][0] == 1)
        # the request reached the server, the answer was lost: the client re-reads the server and shows the turn
        p.js("""window.__lost = false; window.fetch = async (u, o) => { const r = await window.__realFetch(u, o); if (String(u).includes('/turns') && !window.__lost) { window.__lost = true; throw new TypeError('Failed to fetch'); } return r; }""")
        p.click('[data-k="w0"]'); p.click('[data-k="b3"]'); p.click("#btnSubmit")
        p.wait(f"(() => {{ const s = window.__demo.state(); return s.progress === {t + 1} && !s.busy; }})()", 60, "the catch-up")
        st = p.state()
        self.check("11. a lost answer is recovered from the server's history without a duplicate turn", st["progress"] == t + 1 and st["turns"][t][0] == champ_act(0, 3, 0) and not st["problem"], json.dumps({k: st[k] for k in ("progress", "problem")}))
        # a session the server no longer holds
        p.js("""window.fetch = async (u, o) => (String(u).includes('/turns') || String(u).match(/\\/api\\/sessions\\/[0-9a-f]{32}$/)) ? new Response(JSON.stringify({error: 'unknown_session', message: 'no such session: it expired, the server restarted, or the id is wrong'}), {status: 404, headers: {'Content-Type': 'application/json'}}) : window.__realFetch(u, o)""")
        p.click('[data-k="w1"]'); p.click('[data-k="b0"]'); p.click("#btnSubmit")
        p.wait("!!window.__demo.state().problem", 30, "the lost block")
        st = p.state()
        self.check("12. a lost session is reported with a way out", st["problem"]["kind"] == "lost" and p.js("!!document.querySelector('#controls .problem')"), json.dumps(st["problem"]))
        self.shot("11_lost_session")
        p.js("window.fetch = window.__realFetch")
        p.click_text("New game", within="#controls")
        p.wait("document.getElementById('newGame').open", 5)
        self.check("12. the way out opens the New game dialog", True)
        p.click_text("Cancel", within="#newGame")

    def part_beliefs(self) -> None:
        p = self.page
        p.js("window.__demo.newGame('om:0', 'hidden')")
        p.wait("(() => { const s = window.__demo.state(); return s.live && s.progress === 0 && s.canSubmit; })()", 30, "a hidden-belief game")
        sid = p.state()["sid"]
        v = self.srv.api("GET", f"/api/sessions/{sid}")["body"]
        self.check("13. hidden mode: no belief data in the view or the client", v.get("belief") is None and not p.state()["beliefs"])
        p.js("window.__demo.details('belief')")
        p.wait("!!document.querySelector('#details [data-s=\"belief\"]')", 10)
        txt = p.js("document.querySelector('#details [data-s=\"belief\"]').textContent")
        self.check("13. hidden mode: Details says the belief is hidden until the end", "Hidden in this game" in txt, txt[:120])
        p.js("window.__demo.store.closeDetails()")
        for code in (2, 7):
            p.js(f"window.__demo.quick({code})")
            p.wait("(() => { const s = window.__demo.state(); return !s.busy && !s.step; })()", 60)
        p.js("window.__demo.stop()")
        p.wait("window.__demo.state().stopped === true", 30)
        self.check("13. hidden mode: the belief history appears after the stop", p.state()["beliefs"] and len(self.srv.api("GET", f"/api/sessions/{sid}/record")["body"]["beliefs"]["posteriors"]["B"]) == 3)
        p.js("window.__demo.newGame('om:0', 'shown')")
        p.wait("(() => { const s = window.__demo.state(); return s.live && s.progress === 0 && s.canSubmit; })()", 30, "a shown-belief game")
        p.js("window.__demo.quick(7)")
        p.wait("(() => { const s = window.__demo.state(); return s.progress === 1 && !s.busy && !s.step; })()", 60)
        p.js("window.__demo.details('belief')")
        p.wait("!!document.querySelector('#details .belief')", 10)
        rows = p.js("document.querySelectorAll('#details .belief .num').length")
        self.check("13. shown mode: the belief's 13 types are listed with their probabilities during play", rows == 13 and p.state()["beliefs"], str(rows))
        self.shot("12_belief_shown")
        p.js("window.__demo.store.closeDetails()")

    def part_stale(self) -> None:
        p = self.page
        res = p.js("""(async () => {
            const s = window.__demo.store;
            await s.newGame('sp:0', 'hidden');
            const a = s.live.sid;
            const code = s.ctrl.menu().legal[0];
            const submit = s.submit(code);                    // in flight for game A
            const fresh = s.newGame('sp:0', 'hidden');       // replaces it at once
            await Promise.all([submit, fresh]);
            const v = await (await fetch('/api/sessions/' + a)).json();
            return {a, b: s.live.sid, progress: s.ctrl.progress, turns: s.game.turns.length, busy: s.busy, serverA: v.position};
        })()""")
        self.check("14. a turn result that arrives after a new game started is dropped", res["a"] != res["b"] and res["progress"] == 0 and res["turns"] == 0 and not res["busy"], json.dumps(res))

    def part_keyboard(self) -> None:
        p = self.page
        p.wait("(() => { const s = window.__demo.state(); return s.live && s.progress === 0 && s.canSubmit; })()", 30)
        for code in (2, 7, 3):
            p.js(f"window.__demo.quick({code})")
            p.wait("(() => { const s = window.__demo.state(); return !s.busy && !s.step; })()", 60)
        prog = p.state()["progress"]
        p.js("document.activeElement && document.activeElement.blur()")
        p.key("ArrowLeft"); p.key("ArrowLeft")
        self.check("15. the arrow keys browse the history", p.state()["view"] == prog - 2)
        p.key("Home")
        self.check("15. Home shows the start", p.state()["view"] == 0)
        p.key("End")
        self.check("15. End shows the latest position", p.state()["view"] == prog)
        p.key("ArrowLeft"); p.key("Escape")
        self.check("15. Escape returns to the current turn", p.state()["view"] == prog and p.state()["canSubmit"])
        p.js("document.querySelector('[data-k=\"w1\"]').focus()"); p.key(" ")
        p.js("document.querySelector('[data-k=\"b0\"]').focus()"); p.key(" ")
        self.check("15. Space on a focused choice selects it", p.js("document.getElementById('btnSubmit').textContent") == f"Play turn {prog + 1}: Shove, Hold")
        p.js("document.activeElement.blur()")
        p.key("Enter")
        p.wait(f"(() => {{ const s = window.__demo.state(); return s.progress === {prog + 1} && !s.busy && !s.step; }})()", 60, "Enter submits")
        self.check("15. Enter plays the chosen action", p.state()["turns"][prog][0] == 1)
        p.js("document.getElementById('btnFirst').focus()"); p.key("Enter")
        self.check("15. the turn buttons work from the keyboard", p.state()["view"] == 0)
        p.key("End")
        self.no_errors("keyboard")

    def part_reduced_motion(self) -> None:
        p = self.page
        p.media("light", "reduce")
        p.reload()
        st = p.state()
        prog = st["progress"]
        p.click('[data-k="w2"]'); p.click('[data-k="b1"]'); p.click("#btnSubmit")
        p.wait(f"(() => {{ const s = window.__demo.state(); return s.progress === {prog + 1} && !s.busy; }})()", 60)
        self.check("16. with reduced motion a turn shows its end state at once, without stepping", p.state()["step"] is None and p.js("document.body.classList.contains('reduced')"))
        p.media("light", "no-preference")

    # ---- the layouts and the overlap check
    def overlaps(self) -> dict:
        return self.page.js("""(() => {
            const m = 0;
            const obs = ['topbar', 'surface', 'zHist', 'graphBig', 'details'].map((id) => document.getElementById(id))
              .filter((e) => e && !e.hidden && e.offsetParent !== null && (e.id !== 'zHist' || e.classList.contains('pop')))
              .map((e) => { const r = e.getBoundingClientRect(); return {id: e.id, x0: r.left - m, y0: r.top - m, x1: r.right + m, y1: r.bottom + m}; });
            const boxes = window.__demo.boxes();
            const hits = [];
            for (const [k, b] of Object.entries(boxes)) for (const o of obs) {
              const w = Math.min(b.x1, o.x1) - Math.max(b.x0, o.x0), h = Math.min(b.y1, o.y1) - Math.max(b.y0, o.y0);
              if (w > 1 && h > 1) hits.push(`${k}~${o.id}`);
            }
            return {objects: Object.keys(boxes).length, hits};
        })()""")

    def part_layouts(self) -> None:
        total, bad, cases = 0, [], 0
        # one live game for every size and theme: the first load starts it and plays five turns, the later ones
        # resume it (the browser remembers the session)
        p = self.page
        p.size(1440, 900)
        p.goto(self.srv.url + "/?fresh&still&opp=om:0&ticks=auto")
        p.js("(async () => { for (let i = 0; i < 5; i++) { const m = window.__demo.state().menu; await window.__demo.quick(m.forced !== null ? 45 : (i === 2 ? 7 : 2)); } })()")
        p.wait("(() => { const s = window.__demo.state(); return s.progress === 5 && !s.busy; })()", 60)
        for w, h, label in SIZES:
            for scheme in ("light", "dark"):
                for name, url, steps in (
                    ("busy", "/?mode=watch&game=h2h_om_s0_vs_sp_s0&view=147&still", []),
                    ("play", "/?still&ticks=auto", ["window.__demo.select({kind: 'lane', wave: 3, beh: 2})"]),
                ):
                    p.size(w, h)
                    p.media(scheme)
                    p.goto(self.srv.url + url)
                    p.wait("(() => { const s = window.__demo.state(); return !s.busy && !s.loading && s.progress !== null; })()", 60)
                    if name == "play" and p.state()["progress"] != 5:
                        self.check(f"17. {label} {scheme}: the live game resumed", False, json.dumps({k: p.state()[k] for k in ("progress", "problem", "recovered")}))
                    for s in steps:
                        p.js(s)
                    time.sleep(0.4)
                    for state in ("default", "history", "graph", "details"):
                        p.js({"default": "window.__demo.history(false); window.__demo.graph(false); window.__demo.store.closeDetails()",
                              "history": "window.__demo.history(true)", "graph": "window.__demo.history(false); window.__demo.graph(true)",
                              "details": "window.__demo.graph(false); window.__demo.details('events')"}[state])
                        time.sleep(0.45)
                        r = self.overlaps()
                        total += 1
                        cases += 1
                        hits = [x for x in r["hits"] if not x.startswith("chip_")]
                        if hits:
                            bad.append(f"{label} {scheme} {name} {state}: {hits}")
                        if (scheme == "dark" and name == "busy" and state == "default") or (scheme == "light" and name == "play" and state in ("default", "history")) or (name == "busy" and state == "graph" and scheme == "light"):
                            self.shot(f"13_{label}_{scheme}_{name}_{state}")
                    p.js("window.__demo.history(false); window.__demo.graph(false); window.__demo.store.closeDetails()")
        self.check(f"17. no panel covers a champion, wave, tower or the ward spot ({total - len(bad)} of {total} cases)", not bad, "; ".join(bad[:6]))
        self.page.size(1440, 900)
        self.page.media("light")

    # ---- the whole journey
    def run(self) -> bool:
        os.makedirs(SHOTS, exist_ok=True)
        for f in os.listdir(SHOTS):
            os.remove(os.path.join(SHOTS, f))
        self.downloaded = None
        self.part_play()
        self.part_replay_download()
        self.part_watch()
        self.part_absence()
        self.part_duplicates()
        self.part_unanswered()
        self.part_beliefs()
        self.part_stale()
        self.part_keyboard()
        self.part_reduced_motion()
        self.part_layouts()
        self.no_errors("end of the journey")
        return all(c["ok"] for c in self.checks)


def run() -> bool:
    dist = os.path.join(REPO, "web", "dist", "index.html")
    if not os.path.isfile(dist):
        raise SystemExit("the frontend is not built: cd web && npm install && npm run build")
    t0 = time.time()
    with Server() as srv, Browser() as b:
        if not srv.status.get("ready"):
            raise SystemExit(f"the engine is unavailable: {srv.status.get('error')}")
        j = Journey(srv, b)
        try:
            ok = j.run()
        except Exception as e:                                  # record how far it got, and what the page said
            errs = []
            try:
                errs = j.page.errors() if j.page else []
                st = j.page.state() if j.page else {}
                errs.append("state: " + json.dumps({k: st.get(k) for k in ("progress", "busy", "error", "problem", "loading")}))
            except Exception:                                   # noqa: BLE001
                pass
            j.check(f"journey aborted: {type(e).__name__}: {e}", False, "; ".join(errs))
            ok = False
    out = {"meta": {"study": "M6b", "commit": git_commit(), "working_tree_dirty": git_dirty(), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "elapsed_s": round(time.time() - t0, 1), "browser": b.exe, "frontend": "web/dist"},
           "passed": sum(c["ok"] for c in j.checks), "failed": sum(not c["ok"] for c in j.checks), "checks": j.checks,
           "trade_results_seen": getattr(j, "trade_results", []), "screenshots": j.shots}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "m6b_browser.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    print(f"{out['passed']} passed, {out['failed']} failed; {len(j.shots)} screenshots in {SHOTS}", flush=True)
    return ok


__all__ = ["run", "Journey"]
