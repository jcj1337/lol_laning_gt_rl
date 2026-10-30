"""Screenshots of the sequence for the project's web page (the case study on the author's site): the whole app at one
event, and the lane alone at the introduction and at each event, in the dark theme with animation off, at twice the
window's pixel density. Every lane picture is the same crop of the window (the bot's half of the lane, where all five
events happen), so the pictures line up from one event to the next. results/m6/site/site_shots.json keeps the crop
and, for every picture, each scene object's box inside it as fractions of the picture's width and height, which is
where a label on the page can point. Nothing here changes a sequence or a recording.
"""

from __future__ import annotations

import json
import os
import time

from ..m45b.common import git_commit, git_dirty
from ..m6b.browser import REPO, Browser, Server
from .guide_check import GUIDE

OUT = os.path.join("results", "m6", "site")
WIDTH, HEIGHT, SCALE = 1440, 900, 2
CROP = {"x": 640, "y": 66, "width": 780, "height": 324}     # CSS px of the window: below the top bar, the bot's half
HERO = 3                                                     # the whole app at event 4: the plate, the Recall, the lead


def in_crop(boxes: dict) -> dict:
    """Each object's box as fractions of the crop, for the objects that show in it."""
    x0, y0, w, h = CROP["x"], CROP["y"], CROP["width"], CROP["height"]
    out = {}
    for k, b in boxes.items():
        if b["x1"] > x0 and b["x0"] < x0 + w and b["y1"] > y0 and b["y0"] < y0 + h:
            out[k] = [round((b["x0"] - x0) / w, 4), round((b["y0"] - y0) / h, 4), round((b["x1"] - x0) / w, 4), round((b["y1"] - y0) / h, 4)]
    return out


def run() -> None:
    if not os.path.isfile(os.path.join(REPO, "web", "dist", "index.html")):
        raise SystemExit("the frontend is not built: cd web && npm install && npm run build")
    os.makedirs(OUT, exist_ok=True)
    pictures = []
    with Server() as srv, Browser() as b:
        if not srv.status.get("ready"):
            raise SystemExit(f"the engine is unavailable: {srv.status.get('error')}")
        guide = srv.api("GET", f"/api/guides/{GUIDE}")["body"]
        p = b.page(f"{srv.url}/?mode=watch&sequence={GUIDE}&still&theme=dark", WIDTH, HEIGHT, color_scheme="dark")
        p.wait(f"(() => {{ const s = window.__demo.state(); return !s.loading && !!s.guide && s.guide.id === {json.dumps(GUIDE)}; }})()", 30, "the sequence")
        p.c.call("Emulation.setDeviceMetricsOverride", width=WIDTH, height=HEIGHT, deviceScaleFactor=SCALE, mobile=False)
        stops = [(None, "intro", guide["start"], guide["title"])] + [(i, f"event{i + 1}", c["position"], c["title"]) for i, c in enumerate(guide["checkpoints"])]
        for i, name, position, title in stops:
            p.js(f"window.__demo.guideSelect({json.dumps(i)})")
            time.sleep(0.5)
            st = p.state()
            if st["view"] != position or st["guide"]["at"] != i:
                raise SystemExit(f"{name}: the replay shows position {st['view']} (event {st['guide']['at']}), not {position}")
            p.shot(os.path.join(OUT, f"{name}.png"), clip=CROP)
            pictures.append({"file": f"{name}.png", "view": position, "title": title, "boxes": in_crop(p.js("window.__demo.boxes()"))})
            if i == HERO:
                p.shot(os.path.join(OUT, "hero.png"))
                pictures.append({"file": "hero.png", "view": position, "title": title, "boxes": None})
        errs = p.errors()
        if errs:
            raise SystemExit("page errors: " + "; ".join(errs))
    out = {"meta": {"study": "M6c site pictures", "commit": git_commit(), "working_tree_dirty": git_dirty(),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "browser": b.exe, "frontend": "web/dist",
                    "window": [WIDTH, HEIGHT], "scale": SCALE, "theme": "dark", "crop": CROP, "sequence": GUIDE},
           "pictures": pictures}
    with open(os.path.join(OUT, "site_shots.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    print(f"{len(pictures)} pictures in {OUT}", flush=True)


__all__ = ["run", "CROP"]
