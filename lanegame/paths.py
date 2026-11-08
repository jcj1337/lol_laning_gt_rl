"""Where the repository keeps its data.

    artifacts/   tracked and curated: the records of the published studies (`artifacts/<study>/`) and the 7 networks
                 the demo ships with (`artifacts/checkpoints/`, listed in lanegame/demo/manifest.json)
    runs/        git-ignored: what runs write that is not curated, above all every checkpoint
                 (`runs/checkpoints/<study>/...`) and resumable partial results

Paths are relative to the repository root, so commands run from there.

Records written before the reorganization of 2026-11-08 name files by the layout of that time (`results/<study>/`,
`results/rl/checkpoints/`, `docs/archive/m5/`). Records are never rewritten; `relocated` maps such a path to where the file
is now (docs/archive/MOVES.md lists every move).
"""

from __future__ import annotations

import os

ARTIFACTS = "artifacts"
RUNS = "runs"
CHECKPOINTS = os.path.join(RUNS, "checkpoints")
SHIPPED = os.path.join(ARTIFACTS, "checkpoints")

# old prefix -> new prefix; the first match wins, so the specific entries come first
_MOVED = (
    ("results/rl/checkpoints/", "runs/checkpoints/"),
    ("results/", "artifacts/"),
    ("docs/model.md", "docs/model.md"),
    ("docs/research_overview.md", "docs/research_overview.md"),
    ("docs/design_philosophy.md", "docs/design_philosophy.md"),
    ("docs/", "docs/archive/"),
)


def relocated(path: str) -> str:
    """The current location of a file that a record written before 2026-11-08 names."""
    p = path.replace("\\", "/")
    for old, new in _MOVED:
        if p == old or (old.endswith("/") and p.startswith(old)):
            return new + p[len(old):]
    return p


def checkpoint(path: str) -> str:
    """`path` itself, or, when it is missing, the shipped copy of a checkpoint under CHECKPOINTS (a fresh clone has
    only the 7 shipped networks, at the same relative path under SHIPPED)."""
    if os.path.exists(path):
        return path
    rel = os.path.relpath(path, CHECKPOINTS)
    if not rel.startswith(".."):
        shipped = os.path.join(SHIPPED, rel)
        if os.path.exists(shipped):
            return shipped
    return path


def shipped(path: str) -> str:
    """Where the repository ships a checkpoint that runs keep at `path` (under CHECKPOINTS): the same relative path
    under SHIPPED, with forward slashes."""
    rel = os.path.relpath(path, CHECKPOINTS)
    if rel.startswith(".."):
        raise ValueError(f"{path} is not under {CHECKPOINTS}")
    return f"{SHIPPED}/{rel}".replace("\\", "/")
