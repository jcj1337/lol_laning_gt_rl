"""Shared helpers for experiment scripts: output paths and Markdown tables."""

from __future__ import annotations

import os
import platform
import sys
import time
from typing import Iterable, List, Sequence

import numpy as np
import scipy

from ..io import dump_json

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")


def out_path(name: str) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, name)


def environment() -> dict:
    return {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
            "platform": platform.platform(), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}


def md_table(header: Sequence[str], rows: Iterable[Sequence]) -> str:
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4g}" if abs(v) >= 1e-3 or v == 0 else f"{v:.2e}"
        return str(v)
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(fmt(v) for v in r) + " |")
    return "\n".join(lines)


def save(name: str, data: dict, markdown: str) -> None:
    data = dict(data)
    data["environment"] = environment()
    dump_json(data, out_path(name + ".json"))
    with open(out_path(name + ".md"), "w", encoding="utf-8") as f:
        f.write(markdown.rstrip() + "\n")
    print(markdown)
