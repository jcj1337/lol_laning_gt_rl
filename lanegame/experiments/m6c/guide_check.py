"""A sequence (M6c) driven in a headless browser against the real server and the built frontend (m6b/browser.py).

Every check pairs an action a visitor would take (a real click, a change of the Watch select, a page load) with the
state the app reports (window.__demo.state(), the app's own hook) and, where the server is the authority, with what it
says (the sequence it serves, a live session's position). The checks run with animation on, as a visitor sees the app,
except the layout check (a still screen) and the pace check (the system's reduced-motion setting). They are written to
results/m6/m6c_browser.json and the screenshots to results/m6/m6c/. Nothing here changes a sequence, a recording or a
result of an earlier milestone.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import List, Optional

from ..m45b.common import git_commit, git_dirty
from ..m6b.browser import REPO, Browser, Page, Server

OUT = os.path.join("results", "m6")
SHOTS = os.path.join(OUT, "m6c")
GUIDE = "stacking-to-crashing"
REPLAY = "sp_s0_vs_g_shove"
START = 74
POSITIONS = [77, 80, 81, 82, 84]                          # the sequence's events, as its file places them
SIZES = [(1440, 900, "desktop"), (1024, 700, "small"), (390, 844, "phone_portrait")]
FIT_JS = """(() => {
  const g = document.getElementById('guide'), s = document.getElementById('surface').getBoundingClientRect(), c = document.getElementById('controls');
  const r = g ? g.getBoundingClientRect() : null, a = document.querySelector('#turnCard .tc-acts'), cr = c.getBoundingClientRect();
  return {panel: r ? Math.round(r.height) : null, over: r ? Math.round(r.bottom - s.bottom) : null, surface: Math.round(s.height),
          scroll: c.scrollHeight, client: c.clientHeight, acts: a ? Math.round(a.getBoundingClientRect().bottom - cr.bottom) : null};
})()"""
# Samples the screen at once and then every 150 ms until playback stops (or the timeout), in the page, so a screenshot
# taken meanwhile leaves no gap: the view, and whether the note panel, the walk line and the tick step panel are shown.
SAMPLE_JS = """(window.__walk = new Promise((done) => {
  const out = [], t0 = Date.now();
  const look = () => { const s = window.__demo.state(); out.push({view: s.view, playing: s.playing, panel: !!document.getElementById('guide'),
    walk: !!document.getElementById('gdWalk'), stepper: !!document.querySelector('.stepper')}); return s.playing; };
  look();
  const id = setInterval(() => { if (!look() || Date.now() - t0 > %d) { clearInterval(id); done(out); } }, 150);
}), 0)"""
# The lane's arrows on screen: each one's move (its data attributes), its label, its title, its screen box, the names
# of whatever covers its label (a scene object of window.__demo.boxes(): champions, waves, chips, towers, the ward spot;
# or a panel of the surface), its tip and tail as the scene drew them, and the screen box of what moved: the seat's
# token, its leading cluster (the front packet of the frame on screen) or, for a crash, its crashed minions (null when
# none are drawn).
ARROWS_JS = """(() => {
  const box = (r) => ({x0: r.left, y0: r.top, x1: r.right, y1: r.bottom});
  const hit = (a, b) => Math.min(a.x1, b.x1) - Math.max(a.x0, b.x0) > 1 && Math.min(a.y1, b.y1) - Math.max(a.y0, b.y0) > 1;
  const panels = ['topbar', 'surface'].map((id) => document.getElementById(id)).filter((e) => e && !e.hidden && e.offsetParent !== null)
    .map((e) => [e.id, box(e.getBoundingClientRect())]);
  const boxes = window.__demo.boxes(), objects = Object.entries(boxes), f = window.__demo.store.ctrl.frame;
  const mover = (s, what, kind) => what === 'champion' ? `champion_${s}` : kind === 'crash' ? `wave_S${s}` : `wave_${s}${f.lane.packets[s].findIndex((q) => q[2])}`;
  return [...document.querySelectorAll('#lane .mv-arrow')].map((g) => {
    const t = g.querySelector('text'), lb = t ? box(t.getBoundingClientRect()) : null, r = box(g.getBoundingClientRect());
    const d = g.dataset, k = mover(d.seat, d.what, d.kind), mb = boxes[k];
    return {move: [d.seat, d.what, d.kind, d.from, d.to], label: t ? t.textContent : null,
            title: g.querySelector('title').textContent, box: [r.x0, r.y0, r.x1, r.y1].map(Math.round),
            under_panel: panels.filter(([, b]) => hit(r, b)).map(([id]) => id),
            label_covered_by: lb ? [...objects, ...panels].filter(([, b]) => hit(lb, b)).map(([k]) => k) : null,
            tip: [+d.tipX, +d.tipY], tail: [+d.tailX, +d.tailY], mover: k, mover_box: mb ? [mb.x0, mb.y0, mb.x1, mb.y1] : null};
  });
})()"""
LEGEND_JS = """(() => {
  const l = document.querySelector('#lane .lane-legend');
  if (!l) return null;
  const r = l.getBoundingClientRect();
  const hits = ['topbar', 'surface'].map((id) => document.getElementById(id)).filter((e) => e && !e.hidden && e.offsetParent !== null)
    .filter((e) => { const o = e.getBoundingClientRect(); return Math.min(r.right, o.right) > Math.max(r.left, o.left) && Math.min(r.bottom, o.bottom) > Math.max(r.top, o.top); })
    .map((e) => e.id);
  return {lines: [...l.querySelectorAll('text')].map((t) => t.textContent), box: [r.left, r.top, r.right, r.bottom].map(Math.round), hits};
})()"""


def overflows(fit: dict, strict: bool) -> bool:
    """The panel runs past the surface, the controls scroll (desktop and small), or the turn card's actions line is cut."""
    return fit["over"] > 0 or (strict and fit["scroll"] > fit["client"] + 1) or (fit["acts"] is not None and fit["acts"] > 0)


def trail_faults(a: dict) -> List[str]:
    """What is wrong with an arrow's shape on screen: a length outside 20 to 40.5 px, or a tip that is not just behind
    what moved. A champion's tip lies outside its token and within 6 px of it (the scene's tokHalf + 5): farther from
    the token's centre than its box's half-size, which is the circle's radius for A and the corner radius of B's diamond,
    and at most 6 px farther. The box itself reaches past a round token at its corners, so a tip just behind a token
    heading on a diagonal lies inside it. A leading wave's tip lies within 6 px of its cluster's box and outside it; a
    crash's lies within 6 px of the crashed minions' box when they are drawn."""
    (tx, ty), (sx, sy) = a["tip"], a["tail"]
    out = []
    n = math.hypot(tx - sx, ty - sy)
    if not 20 <= n <= 40.5:
        out.append(f"length {n:.1f} px")
    b = a["mover_box"]
    what, kind = a["move"][1], a["move"][2]
    if b is None:
        if kind != "crash":
            out.append(f"{a['mover']} is not drawn")
        return out
    x0, y0, x1, y1 = b
    inside = x0 < tx < x1 and y0 < ty < y1
    gap = math.hypot(max(x0 - tx, 0, tx - x1), max(y0 - ty, 0, ty - y1))
    if what == "champion":
        d, half = math.hypot(tx - (x0 + x1) / 2, ty - (y0 + y1) / 2), max(x1 - x0, y1 - y0) / 2
        if not half < d <= half + 6:
            out.append(f"tip {d:.1f} px from {a['mover']}'s centre (half-size {half:.1f})")
    elif gap > 6 or (kind == "walk" and inside):
        out.append(f"tip {gap:.1f} px from {a['mover']}'s box, inside it: {inside}")
    return out


def views(samples: List[dict]) -> List[int]:
    """The positions shown, in order, each once."""
    out: List[int] = []
    for s in samples:
        if not out or s["view"] != out[-1]:
            out.append(s["view"])
    return out


class GuideCheck:
    def __init__(self, srv: Server, b: Browser):
        self.srv, self.b = srv, b
        self.checks: List[dict] = []
        self.shots: List[str] = []
        self.walks: dict = {}
        self.page: Optional[Page] = None
        self.guide: dict = {}

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

    def text(self, selector: str) -> Optional[str]:
        return self.page.js(f"(document.querySelector({json.dumps(selector)}) || {{}}).textContent")

    def exists(self, selector: str) -> bool:
        return bool(self.page.js(f"!!document.querySelector({json.dumps(selector)})"))

    def wait_guide(self, gid: str = GUIDE, timeout: float = 30) -> None:
        self.page.wait(f"(() => {{ const s = window.__demo.state(); return !s.loading && !!s.guide && s.guide.id === {json.dumps(gid)}; }})()", timeout, f"the sequence {gid}")

    def guide_state(self) -> dict:
        st = self.page.state()
        return {k: st[k] for k in ("view", "guide", "playing", "walkingTo", "kind")}

    def chip_hp(self, seat: int) -> Optional[str]:
        return self.page.js(f"(() => {{ const e = document.querySelectorAll('#lane .tlab .hp')[{seat}]; return e ? e.textContent : null; }})()")

    def graph_label(self) -> str:
        return self.page.js("(document.querySelector('#graph .g-sel text') || document.getElementById('gVal') || {}).textContent") or ""

    # ---- the sequence from the address, and its events
    def part_open(self) -> None:
        self.page = p = self.b.page(f"{self.srv.url}/?mode=watch&sequence={GUIDE}", 1440, 900)
        self.wait_guide()
        st = p.state()
        self.check("1. the address opens the sequence on its research replay, paused at the sequence's start",
                   st["mode"] == "watch" and st["kind"] == "research_record" and st["name"] == REPLAY and st["view"] == START and not st["playing"],
                   json.dumps({k: st[k] for k in ("mode", "kind", "name", "view", "playing")}))
        self.check("1. the sequence starts at its introduction with the notes shown",
                   st["guide"] == {"id": GUIDE, "at": None, "notes": True, "atEvent": True, "walking": False, "position": START}, json.dumps(st["guide"]))
        rec, pick = self.text("#recType"), p.js("document.getElementById('gamePick').value")
        self.check("1. the top bar names the record and selects the sequence", rec == "Recorded research game" and pick == f"guide:{GUIDE}", f"{rec!r} {pick!r}")
        group = p.js("(() => { const g = [...document.querySelectorAll('#gamePick optgroup')].find((o) => o.label === 'Sequences'); "
                     "return g ? [...g.querySelectorAll('option')].map((o) => o.textContent) : null; })()")
        self.check("1. the Watch select lists Stacking to Crashing under Sequences", group is not None and "Stacking to Crashing" in group, json.dumps(group))
        title, kicker = self.text("#guide b"), self.text("#guide .gd-kicker")
        self.check("1. the panel shows the sequence's title", title == self.guide["title"], f"{title!r}")
        self.check("1. the panel's kicker at the introduction reads Sequence", kicker == "Sequence", f"{kicker!r}")
        self.shot("01_intro")
        self.no_errors("open")

    def walk(self, what: str, start: int, target: int, at: int, shot_mid: Optional[str] = None) -> List[int]:
        """After a click that starts a walk: the walk shows each position from `start` to `target` once, in order, with
        the note panel on screen and no tick step panel, and stops at `target`."""
        p = self.page
        p.js(SAMPLE_JS % 15000)
        st = p.state()
        line = self.text("#gdWalk") or ""
        if shot_mid:
            time.sleep(1.2)
            self.shot(shot_mid)
        samples = p.js("window.__walk")
        seq = views(samples)
        self.walks[what] = seq
        print(f"  walk to {target}: {seq}", flush=True)
        end = p.state()
        self.check(f"{what}: the walk starts at once with the walk line in place of the note",
                   st["playing"] and st["guide"]["walking"] and st["guide"]["at"] == at and samples[0]["walk"] and line.startswith(f"Playing to turn {target}:"),
                   json.dumps({"guide": st["guide"], "playing": st["playing"], "line": line}))
        self.check(f"{what}: the view rises one position at a time from {start} to {target} and stops there",
                   seq == list(range(start, target + 1)) and not end["playing"] and end["view"] == target and end["walkingTo"] is None, json.dumps({"views": seq, "end_view": end["view"]}))
        self.check(f"{what}: the note panel stays on screen at every sample and the tick step panel never appears",
                   all(s["panel"] for s in samples) and not any(s["stepper"] for s in samples), json.dumps([s for s in samples if not s["panel"] or s["stepper"]][:3]))
        return seq

    def part_events(self) -> None:
        p = self.page
        cps = self.guide["checkpoints"]
        self.check("2. the server's sequence has the five events at positions 77, 80, 81, 82 and 84", [c["position"] for c in cps] == POSITIONS, str([c["position"] for c in cps]))
        prev = START
        for i, c in enumerate(cps):
            k = i + 1
            p.click("#gdNext")
            self.walk(f"2. Next event to event {k}", prev, c["position"], i, shot_mid="07_walking" if k == 2 else None)
            prev = c["position"]
            time.sleep(0.5)                                     # the lane's transitions finish before the evidence is read
            st = p.state()
            title, ring, nxt = self.text("#guide b"), self.exists("#lane .hl-ring"), self.text("#gdNextTitle")
            want_next = f"Next: {cps[k]['title']}" if k < len(cps) else None
            self.check(f"2. event {k} at position {c['position']}: its note, title and ring, and the next event's title",
                       st["view"] == c["position"] and st["guide"]["at"] == i and st["guide"]["atEvent"] and not st["guide"]["walking"]
                       and title == c["title"] and ring and nxt == want_next and self.text("#guide .gd-text") == c["text"],
                       json.dumps({"view": st["view"], "guide": st["guide"], "title": title, "ring": ring, "next": nxt}))
            if k == 1:
                hp = self.chip_hp(1)
                self.check("2. event 1: the bot's lane chip shows the 105 HP the note quotes", hp == "105", f"{hp!r}")
            if k == 4:
                lab = self.graph_label()
                self.check("2. event 4: the graph marks turn 82 with the agent's lead of 2.98 the note quotes", lab.startswith("Turn 82:") and "+2.98" in lab, lab)
            if k == 5:
                lab = self.graph_label()
                plates = p.js("window.__demo.store.ctrl.frame.towers.B.plates_lost")
                tower = p.js("(() => { const n = window.__demo.store.names.B + \"'s tower:\"; return [...document.querySelectorAll('#lane title')].map((t) => t.textContent).find((s) => s.startsWith(n)) || ''; })()")
                self.check("2. event 5: the graph shows the lead of 3.63 and the bot's tower has lost its second plate",
                           "+3.63" in lab and plates == 2 and "1 of 3 plates left" in tower, json.dumps({"label": lab, "plates_lost": plates, "tower": tower}))
            self.shot(f"{k + 1:02d}_event{k}")
        self.check("2. at the last event Next is disabled and no next title is shown",
                   p.js("document.getElementById('gdNext').disabled") and not self.exists("#gdNextTitle"))
        p.click("#gdPrev")
        st = p.state()
        self.check("3. Previous event jumps back to event 4 without a walk", st["view"] == 82 and st["guide"]["at"] == 3 and not st["playing"] and st["guide"]["atEvent"],
                   json.dumps(self.guide_state()))

    def part_away(self) -> None:
        p = self.page
        p.click("#btnNext")
        st = p.state()
        stepped = st["step"] is not None and st["step"]["t"] == 83
        p.click("#btnSkip")
        p.click("#btnTick")                                      # Continue: the turn's frames are done
        st = p.state()
        away = self.text("#gdAway") or ""
        self.check("4. a turn shown after the event keeps its note and says which position the note describes",
                   stepped and st["view"] == 83 and st["step"] is None and st["guide"]["at"] == 3 and not st["guide"]["atEvent"]
                   and "after turn 82" in away and "Return to this event" in away and not self.exists("#lane .hl-ring"),
                   json.dumps({"stepped": stepped, "view": st["view"], "guide": st["guide"], "away": away}))
        self.shot("08_away_from_event")
        p.click("#gdReturn")
        st = p.state()
        self.check("4. Return to this event restores the position and the ring", st["view"] == 82 and st["guide"]["atEvent"] and self.exists("#lane .hl-ring"), json.dumps(self.guide_state()))

    def part_play(self) -> None:
        p = self.page
        p.js("window.__demo.guideSelect(0)")
        p.click("#btnPlay")
        self.walk("5. Play from event 1", 77, 80, 1)
        st = p.state()
        self.check("5. Play stops at event 2 with its note", st["guide"]["at"] == 1 and st["guide"]["atEvent"] and self.text("#guide b") == self.guide["checkpoints"][1]["title"], json.dumps(self.guide_state()))
        p.click("#btnPlay")
        self.walk("5. Play from event 2", 80, 81, 2)
        p.js("window.__demo.guideSelect(4)")
        p.click("#btnPlay")
        st = p.state()
        self.check("5. Play at the last event walks on toward the end of the recording", st["playing"] and st["walkingTo"] == 150 and not st["guide"]["walking"], json.dumps(self.guide_state()))
        time.sleep(3.0)
        st = p.state()
        self.check("5. three seconds later the view is past the last event, the sequence still open on it, no tick step panel",
                   st["playing"] and st["view"] > 84 and st["guide"] is not None and st["guide"]["at"] == 4 and self.exists("#guide") and not self.exists(".stepper"),
                   json.dumps(self.guide_state()))
        p.click("#btnPlay")
        st = p.state()
        self.check("5. Pause stops it", not st["playing"] and st["walkingTo"] is None, json.dumps(self.guide_state()))

    def part_notes(self) -> None:
        p = self.page
        if not p.state()["guide"]["atEvent"]:
            p.click("#gdReturn")
        shown = p.rect("#surface")["h"]
        p.click("#gdHide")
        st = p.state()
        hidden = p.rect("#surface")["h"]
        self.check("6. Hide notes collapses the panel to one line, drops the ring and shrinks the dock",
                   not st["guide"]["notes"] and self.exists("#guide.collapsed") and not self.exists("#lane .hl-ring") and hidden < shown,
                   json.dumps({"guide": st["guide"], "surface_shown": shown, "surface_hidden": hidden}))
        self.shot("09_notes_hidden")
        p.click("#gdShow")
        st = p.state()
        self.check("6. Show notes restores the panel, the ring and the dock", st["guide"]["notes"] and self.exists("#lane .hl-ring") and p.rect("#surface")["h"] == shown, json.dumps(st["guide"]))

    def part_close(self) -> None:
        p = self.page
        v, label = p.state()["view"], self.text("#gdClose")
        p.click("#gdClose")
        st = p.state()
        self.check("7. Close sequence leaves the replay on screen where it was",
                   label == "Close sequence" and st["guide"] is None and st["kind"] == "research_record" and st["view"] == v and not self.exists("#guide"),
                   json.dumps({"label": label, **self.guide_state()}))

    # ---- a live game kept through a sequence
    def part_live(self) -> None:
        p = self.page
        p.js("window.__demo.newGame('sp:0', 'hidden')")
        p.wait("(() => { const s = window.__demo.state(); return s.live && s.progress === 0 && s.canSubmit; })()", 30, "a new game")
        st = p.state()
        sid = st["sid"]
        self.check("8. a new live game carries no sequence", st["guide"] is None and not self.exists("#guide"))
        p.click("#modeWatch")
        p.wait("(() => { const s = window.__demo.state(); return s.mode === 'watch' && !s.loading && s.kind === 'research_record'; })()", 30, "Watch")
        p.js(f"(() => {{ const e = document.getElementById('gamePick'); e.value = 'guide:{GUIDE}'; e.dispatchEvent(new Event('change', {{bubbles: true}})); }})()")
        self.wait_guide()
        pick = p.js("document.getElementById('gamePick').value")
        self.check("8. the Watch select opens the sequence", p.state()["guide"]["id"] == GUIDE and pick == f"guide:{GUIDE}", f"{pick!r}")
        r = self.srv.api("GET", f"/api/sessions/{sid}")
        self.check("8. the live game is still on the server, at its first turn", r["status"] == 200 and r["body"].get("position") == 0, json.dumps({"status": r["status"], "position": r["body"].get("position")}))
        p.click("#modePlay")
        p.wait("(() => { const s = window.__demo.state(); return s.mode === 'play' && !s.loading && s.canSubmit; })()", 30, "the return to the live game")
        st = p.state()
        self.check("8. Play returns to the same game, ready for its first turn", st["sid"] == sid and st["canSubmit"] and st["guide"] is None and st["dialog"] is None,
                   json.dumps({k: st[k] for k in ("sid", "mode", "canSubmit", "guide", "dialog")}))
        self.no_errors("live game")

    def part_simulate(self) -> None:
        p = self.page
        p.js(f"window.__demo.guide({json.dumps(GUIDE)})")
        self.wait_guide()
        sims = self.srv.wait_simulations()
        self.check("9. the simulation worker is ready", sims == "ready", sims)
        p.js("window.__demo.simulate('om:0', 'sp:0')")
        p.wait("window.__demo.state().kind === 'new_simulation'", 150, "a generated match")
        st = p.state()
        self.check("9. a generated match replaces the sequence and carries none", st["kind"] == "new_simulation" and st["guide"] is None and not self.exists("#guide"), json.dumps(self.guide_state()))

    def part_unknown(self) -> None:
        p = self.page
        p.goto(f"{self.srv.url}/?mode=watch&sequence=nope")
        st = p.state()
        txt = self.text("#controls .problem") or ""
        self.check("10. an unknown sequence is reported as an error in the server's words",
                   st["problem"] is not None and st["problem"]["kind"] == "error" and st["problem"]["message"] == "no sequence named 'nope'" and "nope" in txt,
                   json.dumps({"problem": st["problem"], "text": txt}))
        self.shot("10_unknown_sequence")

    # ---- the layouts, the overlap check and the panel's height
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
        p = self.page
        bad, heights, legends = [], {}, []
        for w, h, label in SIZES:
            strict = label != "phone_portrait"                # the controls' own scroll is asserted at desktop and small
            for scheme in ("light", "dark"):
                p.size(w, h)
                p.media(scheme)
                p.goto(f"{self.srv.url}/?mode=watch&sequence={GUIDE}&still&theme={scheme}")
                self.wait_guide()
                p.js("window.__demo.guideSelect(2)")
                time.sleep(0.5)
                hits = [x for x in self.overlaps()["hits"] if not x.startswith("chip_")]
                if hits:
                    bad.append(f"{label} {scheme}: {hits}")
                leg = p.js(LEGEND_JS)
                if not leg or len(leg["lines"]) != 2 or not leg["lines"][1].startswith("Arrows:") or leg["hits"]:
                    legends.append(f"{label} {scheme}: {json.dumps(leg)}")
                fit = p.js(FIT_JS)
                if overflows(fit, strict):
                    bad.append(f"{label} {scheme} at the crash: the panel overflows {json.dumps(fit)}")
                self.shot(f"11_{label}_{scheme}")
                if scheme == "light":
                    hs = {}
                    for i in (None, 0, 1, 2, 3, 4):
                        p.js(f"window.__demo.guideSelect({json.dumps(i)})")
                        time.sleep(0.15)
                        fit = p.js(FIT_JS)
                        hs["intro" if i is None else f"event{i + 1}"] = fit["panel"]
                        if overflows(fit, strict):
                            bad.append(f"{label} at {'the intro' if i is None else f'event {i + 1}'}: the panel overflows {json.dumps(fit)}")
                    heights[label] = {"panel": hs, "surface": fit["surface"], "controls": fit["client"]}
        self.check(f"11. no panel covers a champion, wave, tower or the ward spot, and the note panel fits, at {len(SIZES)} sizes in both themes", not bad, "; ".join(bad[:8]))
        self.check("11. the note panel fits at every note (the measured heights are the detail)", not any("overflows" in b for b in bad), json.dumps(heights))
        self.check(f"11. the lane's legend shows its two lines (the balls, the arrows) clear of the top bar and the surface, at {len(SIZES)} sizes in both themes",
                   not legends, "; ".join(legends[:4]))
        print(f"  panel heights (px): {json.dumps(heights)}", flush=True)
        p.size(1440, 900)
        p.media("light")

    # ---- playback without motion keeps the animated pace (plain Watch)
    def part_pace(self) -> None:
        p = self.page
        p.media("light", "reduce")
        p.goto(f"{self.srv.url}/?mode=watch&game=h2h_om_s0_vs_sp_s0")
        p.wait("(() => { const s = window.__demo.state(); return !s.loading && s.name === 'h2h_om_s0_vs_sp_s0'; })()", 30, "the research game")
        p.click("#btnPlay")
        time.sleep(2.0)
        st = p.state()
        v1 = st["view"]
        self.check("12. with reduced motion, Play at 1x shows one turn in the first two seconds (a turn takes 4.4 s, as animated)",
                   st["playing"] and v1 is not None and v1 <= 1 and st["step"] is None, json.dumps({"view": v1, "playing": st["playing"]}))
        p.click('[data-s="4"]')
        time.sleep(2.6)
        st = p.state()
        self.check("12. at 4x the next 2.6 s add one to three turns", st["playing"] and 1 <= st["view"] - v1 <= 3, json.dumps({"before": v1, "after": st["view"]}))
        p.click("#btnPlay")
        self.check("12. Pause stops it", not p.state()["playing"])
        p.media("light", "no-preference")

    # ---- the arrows of the last turn's moves (plain Watch, a stepped turn, a live game, the sequence)
    def moves_at(self, where: str) -> List[dict]:
        """The arrows on screen; when there are any, also the check that each is a short trail just behind what moved."""
        out = self.page.js(ARROWS_JS)
        print(f"  arrows {where}: {json.dumps([a['move'] + [a['label']] for a in out])}", flush=True)
        if out:
            faults = {" ".join(a["move"]): f for a in out if (f := trail_faults(a))}
            self.check(f"13. {where}: each arrow is 20 to 40.5 px long and its tip sits just behind the token or wave that moved", not faults,
                       json.dumps({"faults": faults, "arrows": [{k: a[k] for k in ("move", "tip", "tail", "mover", "mover_box")} for a in out]}))
        return out

    def arrow_check(self, what: str, arrows: List[dict], want: List[list], label: Optional[str] = None) -> None:
        """Exactly the moves `want`, in any order, each drawn on the open lane; with `label`, the labelled arrow's label
        reads it and nothing covers it."""
        moves = sorted(a["move"] for a in arrows)
        drawn = all(a["box"][2] > a["box"][0] and a["box"][3] > a["box"][1] and not a["under_panel"] for a in arrows)
        labelled = [a for a in arrows if a["label"]]
        lab_ok = label is None or (len(labelled) == 1 and labelled[0]["label"] == label and labelled[0]["label_covered_by"] == [])
        self.check(what, moves == sorted(want) and drawn and lab_ok, json.dumps(arrows))

    def part_arrows(self) -> None:
        p = self.page
        p.size(1440, 900)
        p.media("light")
        p.goto(f"{self.srv.url}/?mode=watch&game={REPLAY}")
        p.wait(f"(() => {{ const s = window.__demo.state(); return !s.loading && s.name === {json.dumps(REPLAY)}; }})()", 30, "the research game")

        def at(pos: int) -> List[dict]:
            p.js(f"window.__demo.goto({pos})")
            time.sleep(0.6)
            return self.moves_at(f"after turn {pos}")

        crash81 = [["A", "wave", "crash", "0", "2"], ["A", "champion", "walk", "2", "1"], ["B", "wave", "walk", "4", "2"]]
        self.arrow_check("13. after turn 81: exactly three arrows, A's wave crashing 0 -> 2 (labelled Crash), A walking 2 -> 1 and B's wave walking 4 -> 2",
                         at(81), crash81, "Crash")
        self.shot("12_arrows_crash")
        self.arrow_check("13. after turn 82: B recalls 2 -> base with its label Recall in the clear, and A walks 1 -> 2",
                         at(82), [["B", "champion", "recall", "2", "base"], ["A", "champion", "walk", "1", "2"]], "Recall")
        self.shot("13_arrows_recall")
        self.arrow_check("13. after turn 83: B comes back base -> 2 with its label Back from base in the clear",
                         at(83), [["B", "champion", "return", "base", "2"]], "Back from base")
        self.shot("14_arrows_return")
        a74 = at(74)
        self.check("13. after turn 74: A walks 0 -> the ward spot W, and A's wave walks -2 -> -1",
                   sorted(a["move"] for a in a74) == sorted([["A", "champion", "walk", "0", "W"], ["A", "wave", "walk", "-2", "-1"]]), json.dumps(a74))
        a0 = at(0)
        self.check("13. the initial position has no arrows", a0 == [], json.dumps(a0))
        p.js("window.__demo.goto(80)")
        p.click("#btnNext")
        p.wait("(() => { const s = window.__demo.state(); return s.step && s.step.t === 81; })()", 10, "turn 81 stepped")
        st, start = p.state(), self.moves_at("at the start frame of turn 81")
        self.check("13. a stepped turn's start frame has no arrows (its frames animate the motion)", st["step"]["sub"] == "start" and start == [],
                   json.dumps({"step": st["step"], "arrows": start}))
        p.js("window.__demo.skip()")
        time.sleep(0.4)
        st, end = p.state(), self.moves_at("at the end frame of turn 81")
        self.arrow_check("13. its end frame has the three arrows of turn 81", end if st["step"] and st["step"]["sub"] == "end" else [], crash81)
        p.click("#btnTick")
        self.sr_check()
        self.no_errors("arrows in Watch")

        p.js("window.__demo.newGame('sp:0', 'hidden')")
        p.wait("(() => { const s = window.__demo.state(); return s.live && s.progress === 0 && s.canSubmit; })()", 30, "a new game")
        p.click('[data-k="recall"]')
        p.click("#btnSubmit")
        p.wait("(() => { const s = window.__demo.state(); return s.step && s.step.t === 1; })()", 60, "the Recall turn's frames")
        heads = []
        for _ in range(6):
            heads.append(p.state()["step"]["head"])
            if heads[-1] == "End of turn 1":
                break
            p.click("#btnTick")
        time.sleep(0.6)                                         # the lane's transitions finish before the arrows are measured
        live = self.moves_at("at the end of a live Recall turn")
        mine = [a for a in live if a["move"][:3] == ["A", "champion", "recall"]]
        self.check("13. a live game: at the end of turn 1 your Recall is drawn from -2 to base, titled in the second person",
                   heads[-1] == "End of turn 1" and len(mine) == 1 and mine[0]["move"] == ["A", "champion", "recall", "-2", "base"]
                   and mine[0]["title"] == "You recalled to base" and mine[0]["label"] == "Recall" and not mine[0]["under_panel"],
                   json.dumps({"heads": heads, "arrows": live}))
        self.shot("15_live_recall")
        p.click("#btnTick")
        self.no_errors("arrows in a live game")

        p.goto(f"{self.srv.url}/?mode=watch&sequence={GUIDE}")
        self.wait_guide()
        p.js("window.__demo.guideSelect(2)")
        time.sleep(0.6)
        st, g = p.state(), self.moves_at("at the sequence's event 3")
        crash = [a for a in g if a["move"] == ["A", "wave", "crash", "0", "2"]]
        ring = p.js("(() => { const r = document.querySelector('#lane .hl-ring'); if (!r) return null; const b = r.getBoundingClientRect(); return [b.left, b.top, b.right, b.bottom].map(Math.round); })()")
        self.check("13. the sequence's event 3 (the crash) shows the crash arrow on the open lane together with the sequence's ring",
                   st["view"] == 81 and st["guide"]["at"] == 2 and ring is not None and len(crash) == 1 and not crash[0]["under_panel"]
                   and crash[0]["label_covered_by"] == [], json.dumps({"guide": st["guide"], "ring": ring, "arrows": g}))
        self.shot("16_sequence_crash_arrows")
        self.no_errors("arrows in the sequence")

    def sr_check(self) -> None:
        """The screen-reader text of a shown position ends with one sentence per arrow, in the arrows' own words."""
        p = self.page
        p.js("window.__demo.goto(82)")
        time.sleep(0.3)
        txt = p.js("document.getElementById('laneText').textContent") or ""
        names = p.js("window.__demo.store.names")
        want = f"{names['A']} moved this turn. {names['B']} recalled to base."
        self.check("13. the lane's screen-reader text after turn 82 ends with a sentence per arrow", txt.endswith(want), json.dumps({"text": txt, "want": want}))

    # ---- the whole check
    def run(self) -> bool:
        os.makedirs(SHOTS, exist_ok=True)
        for f in os.listdir(SHOTS):
            os.remove(os.path.join(SHOTS, f))
        self.guide = self.srv.api("GET", f"/api/guides/{GUIDE}")["body"]
        self.part_open()
        self.part_events()
        self.part_away()
        self.part_play()
        self.part_notes()
        self.part_close()
        self.part_live()
        self.part_simulate()
        self.part_unknown()
        self.part_layouts()
        self.part_pace()
        self.part_arrows()
        self.no_errors("end of the check")
        return all(c["ok"] for c in self.checks)


def run() -> bool:
    dist = os.path.join(REPO, "web", "dist", "index.html")
    if not os.path.isfile(dist):
        raise SystemExit("the frontend is not built: cd web && npm install && npm run build")
    t0 = time.time()
    with Server() as srv, Browser() as b:
        if not srv.status.get("ready"):
            raise SystemExit(f"the engine is unavailable: {srv.status.get('error')}")
        j = GuideCheck(srv, b)
        try:
            ok = j.run()
        except Exception as e:                                  # record how far it got, and what the page said
            errs = []
            try:
                errs = j.page.errors() if j.page else []
                st = j.page.state() if j.page else {}
                errs.append("state: " + json.dumps({k: st.get(k) for k in ("view", "guide", "busy", "error", "problem", "loading")}))
            except Exception:                                   # noqa: BLE001
                pass
            j.check(f"check aborted: {type(e).__name__}: {e}", False, "; ".join(errs))
            ok = False
    out = {"meta": {"study": "M6c", "commit": git_commit(), "working_tree_dirty": git_dirty(), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "elapsed_s": round(time.time() - t0, 1), "browser": b.exe, "frontend": "web/dist"},
           "passed": sum(c["ok"] for c in j.checks), "failed": sum(not c["ok"] for c in j.checks), "checks": j.checks, "walks": j.walks,
           "screenshots": j.shots}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "m6c_browser.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    print(f"{out['passed']} passed, {out['failed']} failed; {len(j.shots)} screenshots in {SHOTS}", flush=True)
    return ok


__all__ = ["run", "GuideCheck"]
