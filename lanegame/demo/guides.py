"""Guided replays of the M6 demo (M6c): a guide is a short, ordered set of notes over one verified research recording,
which the visitor walks through moment by moment.

A guide is one JSON file in `lanegame/demo/guides/`, named `<id>.json`:

    schema, version   "lanegame.demo_guide", 1
    id                the file's stem: lower-case letters, digits and hyphens
    title, intro      the guide's name and the paragraph the visitor reads first
    replay            the research recording it annotates, a key of `service.RESEARCH_REPLAYS`
    start, end        the positions the walk spans (`end` is optional: the walk then ends with the recording)
    checkpoints       1 to 12 notes, each at a position with a title, a text and an optional highlight

Positions follow the application's convention: position p is the state after p turns, position 0 is the initial
state, and a checkpoint at position p shows that state. A highlight names an entity of the lane view to draw the eye to
at that position: a seat's champion, tower, front wave or crashed minions, or the shared ward spot (no seat).

The files are data at a system boundary, so every rule is checked here and nowhere else. `read_guides` parses a folder
and checks the structural rules; `check_guide` checks a guide against the game bundle it annotates. The service and the
API trust the `Guide` objects. A broken rule is a `GuideError` whose message names the file and the rule:

    1  schema and version are the ones above
    2  the keys are exactly {schema, version, id, title, replay, start, intro, checkpoints} plus an optional `end`;
       a checkpoint's are exactly {id, position, title, text} plus an optional `highlight`
    3  the id matches ^[a-z0-9][a-z0-9-]{1,63}$ and equals the file's stem
    4  the title has at most 80 characters, the intro 600, a checkpoint's title 60 and its text 400, none empty (the
       notes must stay short for the panel)
    5  the replay is a research recording the service knows
    6  start >= 0, end >= start, start <= the first checkpoint's position, the last checkpoint's position <= end;
       against the bundle, start, end and every position are at most the recording's length
    7  1 to 12 checkpoints, their ids matching rule 3 and unique within the guide, their positions strictly increasing
    8  a highlight has exactly {kind, seat}: kind one of HIGHLIGHT_KINDS, seat "A" or "B" except for ward_spot,
       which takes none; against the bundle the champion is present, the wave has a front packet and the crashed
       mass has HP at the checkpoint's position (a tower and the ward spot always exist)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

GUIDE_SCHEMA = "lanegame.demo_guide"
GUIDE_VERSION = 1
GUIDES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "guides")
SEATS = ("A", "B")
ID = re.compile(r"[a-z0-9][a-z0-9-]{1,63}")
MAX_CHECKPOINTS = 12
LIMITS = {"title": 80, "intro": 600}
CHECKPOINT_LIMITS = {"title": 60, "text": 400}
KEYS = {"schema", "version", "id", "title", "replay", "start", "intro", "checkpoints"}
CHECKPOINT_KEYS = {"id", "position", "title", "text"}
_PRESENT = {                                   # what "the highlighted entity exists at position p" means, per kind
    "champion": lambda pos, seat: pos["champions"][seat]["status"] == "present",
    "tower": lambda pos, seat: True,
    "wave": lambda pos, seat: any(front for _, _, front in pos["lane"]["packets"][seat]),
    "crashed": lambda pos, seat: pos["lane"]["siege"][seat][1] > 0,
    "ward_spot": lambda pos, seat: True,
}
HIGHLIGHT_KINDS = tuple(_PRESENT)


class GuideError(ValueError):
    """A guide file breaks a rule of the module docstring; the message names the file and the rule."""


@dataclass(frozen=True)
class Highlight:
    kind: str
    seat: Optional[str]


@dataclass(frozen=True)
class Checkpoint:
    id: str
    position: int
    title: str
    text: str
    highlight: Optional[Highlight]


@dataclass(frozen=True)
class Guide:
    id: str
    title: str
    intro: str
    replay: str
    start: int
    end: Optional[int]
    checkpoints: Tuple[Checkpoint, ...]

    def public(self) -> dict:
        """The guide as the API serves it (schemas.Guide): these fields, the checkpoints as a list."""
        d = asdict(self)
        d["checkpoints"] = list(d["checkpoints"])
        return d


def _rule(file: str, n: int, name: str, what: str) -> GuideError:
    return GuideError(f"{file}: rule {n} ({name}): {what}")


def _keys(d, required: set, optional: set, file: str, n: int, where: str) -> None:
    if not isinstance(d, dict):
        raise _rule(file, n, "keys", f"{where} is not a JSON object")
    unknown, missing = set(d) - required - optional, required - set(d)
    if unknown:
        raise _rule(file, n, "keys", f"{where} has the unknown key(s) {sorted(unknown)}")
    if missing:
        raise _rule(file, n, "keys", f"{where} lacks {sorted(missing)}")


def _id(v, file: str, n: int, name: str, what: str) -> str:
    if not isinstance(v, str) or not ID.fullmatch(v):
        raise _rule(file, n, name, f"{what} {v!r} must match ^{ID.pattern}$")
    return v


def _int(v, file: str, what: str) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise _rule(file, 6, "positions", f"{what} must be an integer")
    return v


def _text(d: dict, key: str, limit: int, file: str, where: str) -> str:
    v = d[key]
    if not isinstance(v, str) or not v.strip() or len(v) > limit:
        raise _rule(file, 4, "text length", f"{where} {key} must be a non-empty string of at most {limit} characters")
    return v


def _highlight(h, file: str, where: str) -> Optional[Highlight]:
    if h is None:
        return None
    _keys(h, {"kind"}, {"seat"}, file, 8, f"{where} highlight")
    kind, seat = h["kind"], h.get("seat")
    if kind not in HIGHLIGHT_KINDS:
        raise _rule(file, 8, "highlight", f"{where} highlight kind {kind!r} is not one of {list(HIGHLIGHT_KINDS)}")
    if kind == "ward_spot" and seat is not None:
        raise _rule(file, 8, "highlight", f"{where} highlight of the ward spot takes no seat")
    if kind != "ward_spot" and seat not in SEATS:
        raise _rule(file, 8, "highlight", f"{where} highlight of a {kind} needs a seat, \"A\" or \"B\"")
    return Highlight(kind, seat)


def _checkpoint(c, file: str, i: int) -> Checkpoint:
    where = f"checkpoint {i + 1}"
    _keys(c, CHECKPOINT_KEYS, {"highlight"}, file, 2, where)
    cid = _id(c["id"], file, 7, "checkpoints", f"{where} id")
    position = _int(c["position"], file, f"{where} position")
    title = _text(c, "title", CHECKPOINT_LIMITS["title"], file, where)
    text = _text(c, "text", CHECKPOINT_LIMITS["text"], file, where)
    return Checkpoint(cid, position, title, text, _highlight(c.get("highlight"), file, where))


def _parse(d, file: str) -> Guide:
    if not isinstance(d, dict):
        raise _rule(file, 2, "keys", "the file is not a JSON object")
    if d.get("schema") != GUIDE_SCHEMA or d.get("version") != GUIDE_VERSION:
        raise _rule(file, 1, "schema", f"not a {GUIDE_SCHEMA} version {GUIDE_VERSION} guide")
    _keys(d, KEYS, {"end"}, file, 2, "the guide")
    stem = os.path.splitext(os.path.basename(file))[0]
    gid = _id(d["id"], file, 3, "id", "the id")
    if gid != stem:
        raise _rule(file, 3, "id", f"the id {gid!r} differs from the file's stem {stem!r}")
    title = _text(d, "title", LIMITS["title"], file, "the guide's")
    intro = _text(d, "intro", LIMITS["intro"], file, "the guide's")
    from .service import RESEARCH_REPLAYS            # the one catalog of recordings; service.py imports this module
    replay = d["replay"]
    if replay not in RESEARCH_REPLAYS:
        raise _rule(file, 5, "replay", f"{replay!r} is not a research recording; the recordings are "
                                       f"{list(RESEARCH_REPLAYS)}")
    start = _int(d["start"], file, "start")
    end = None if d.get("end") is None else _int(d["end"], file, "end")
    if start < 0:
        raise _rule(file, 6, "positions", f"start {start} is before position 0")
    if end is not None and end < start:
        raise _rule(file, 6, "positions", f"end {end} is before start {start}")
    cps = d["checkpoints"]
    if not isinstance(cps, list) or not 1 <= len(cps) <= MAX_CHECKPOINTS:
        raise _rule(file, 7, "checkpoints", f"a guide has 1 to {MAX_CHECKPOINTS} checkpoints in a list, not "
                                            f"{len(cps) if isinstance(cps, list) else cps!r}")
    checkpoints = tuple(_checkpoint(c, file, i) for i, c in enumerate(cps))
    ids = [c.id for c in checkpoints]
    if len(set(ids)) != len(ids):
        raise _rule(file, 7, "checkpoints", f"the checkpoint id(s) {sorted({i for i in ids if ids.count(i) > 1})} "
                                            "repeat")
    positions = [c.position for c in checkpoints]
    if any(b <= a for a, b in zip(positions, positions[1:])):
        raise _rule(file, 7, "checkpoints", f"the positions {positions} must strictly increase")
    if start > positions[0]:
        raise _rule(file, 6, "positions", f"start {start} is after the first checkpoint at {positions[0]}")
    if end is not None and positions[-1] > end:
        raise _rule(file, 6, "positions", f"the last checkpoint at {positions[-1]} is after end {end}")
    return Guide(gid, title, intro, replay, start, end, checkpoints)


def read_guides(folder: str = GUIDES_DIR) -> List[Guide]:
    """Every `*.json` file in `folder`, in file-name order, parsed and checked against the structural rules (the
    module docstring); the first bad file raises GuideError."""
    out = []
    for name in sorted(f for f in os.listdir(folder) if f.endswith(".json")):
        file = os.path.join(folder, name)
        with open(file, encoding="utf-8") as f:
            try:
                d = json.load(f)
            except ValueError as e:
                raise GuideError(f"{file}: not valid JSON ({e})") from e
        out.append(_parse(d, file))
    return out


def check_guide(guide: Guide, bundle: dict) -> None:
    """The rules that need the game (6 and 8 against the bundle, the module docstring). `bundle` is
    `session.GameSession.bundle()` of the guide's replay: positions 0..n and turns 1..n."""
    file, n = f"{guide.id}.json", len(bundle["turns"])

    def within(p: int, what: str) -> None:
        if p > n:
            raise _rule(file, 6, "positions", f"{what} {p} is beyond the recording's {n} turns")

    within(guide.start, "start")
    if guide.end is not None:
        within(guide.end, "end")
    for c in guide.checkpoints:
        within(c.position, f"checkpoint {c.id!r} at position")
        h = c.highlight
        if h is not None and not _PRESENT[h.kind](bundle["positions"][c.position], h.seat):
            raise _rule(file, 8, "highlight", f"checkpoint {c.id!r} highlights seat {h.seat}'s {h.kind}, which does "
                                              f"not exist at position {c.position}")


__all__ = ["GUIDE_SCHEMA", "GUIDE_VERSION", "GUIDES_DIR", "HIGHLIGHT_KINDS", "Guide", "Checkpoint", "Highlight",
           "GuideError", "read_guides", "check_guide"]
