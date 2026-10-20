"""M5 reproducibility materials (M5e): the inventory of the git-ignored artifacts the M5 results depend on, with their
SHA-256, and the commands of every stage, generated into results/m5/inventory.json, results/m5/inventory_files.json.gz
and results/m5/reproducibility.md (`python -m lanegame.experiments.rl.m5 reproduce`).

Two levels of reproduction:
    from saved records   `analyse` and `report` re-derive every table from the evaluation records and the run
                         directories' summaries (minutes, no training, no game); the inventory lists the files they
                         read and their digests.
    the campaign         the stages in order, each behind its approval flag; with the same code, seeds and machine
                         class a stage reproduces bit for bit (the determinism contract: fixed chance draws per turn,
                         seeded samplers, tested re-runs); the measured wall times are in the runtime table.
"""

from __future__ import annotations

import gzip
import json
import os
import time
from typing import Dict, List

from . import common as C

RECORD_DIRS = ("eval", "battery", "cells", "states", os.path.join("attack", "eval", "main"),
               os.path.join("train", "main"), os.path.join("attack", "main"))

# (stage, command, approval flag or "", what it produces)
STAGES = (
    ("prereg_check", "prereg_check", "", "verifies the amendment chain; the document must be the active version"),
    ("M5a readiness", "ready", "", "R0-R10 (results/m5/readiness.json)"),
    ("correction pass and later readiness", "corrections", "", "A1-A4, B1 (battery smoke), C1 (attacker smoke) "
     "(results/m5/corrections_readiness.json)"),
    ("M5b training", "train", "--confirm-m5-train", "12 runs x 32,768,000 (train/main/<arm>/s<s>/)"),
    ("training check", "train_check", "", "results/m5/train_main.json (health, identities)"),
    ("M5b monitoring", "monitor", "--confirm-m5-eval", "learning curves and history games (eval/curve, eval/history)"),
    ("M5d families", "families", "--confirm-m5-eval", "opponent families, head to head, midpoints (eval/)"),
    ("M5c battery", "battery", "--confirm-m5-eval", "the M4.5c battery on 12 finals (battery/)"),
    ("M5c 150-turn re-runs", "battery_extend", "--confirm-m5-eval", "exception 3 for every failed validation"),
    ("M5c cells", "cells", "--confirm-m5-eval", "the M4 limitation families (cells/)"),
    ("M5d core attackers", "attackers", "--confirm-m5-attackers", "12 x 8,192,000 (attack/main/<kind>_s<s>/), then "
     "attack_check"),
    ("attacker check", "attack_check", "", "results/m5/attack_check_main.json"),
    ("M5d attacker evaluation", "attack_eval", "--confirm-m5-attackers", "36 cells x 128 pairs x 500 turns "
     "(attack/eval/main/)"),
    ("analysis", "analyse", "", "results/m5/analysis_main.json, monitoring_main.json, strategic_main.json"),
    ("traces (exploratory)", "traces", "", "15 viewer recordings (results/m5/traces/)"),
    ("attacker behaviour (exploratory)", "attack_behaviour", "", "results/m5/attack_behaviour.json"),
    ("report", "report", "", "results/m5/report.md"),
    ("reproducibility", "reproduce", "", "this inventory"),
)


def _ident(path: str) -> dict:
    p = path.replace("\\", "/")
    return {"path": p, "file_sha256": C.file_sha256(path) if os.path.exists(path) else None}


def networks() -> Dict[str, dict]:
    """Every network an M5 result read, by role, from the identities each stage recorded."""
    out: Dict[str, dict] = {}
    tc = C.load("train_main") or {}
    for run, ids in (tc.get("identities") or {}).items():
        for k, rec in ids.items():
            out[f"train:{run}:{k}"] = {"path": rec["path"], "file_sha256": rec["file_sha256"], "role": "M5b run"}
    ac = C.load("attack_check_main") or {}
    for run, rec in (ac.get("identities") or {}).items():
        out[f"attacker:{run}"] = {**rec, "role": "M5d core attacker final"}
    for st in ("monitor", "families", "battery", "cells", "attackers", "attack_eval"):
        for spec, rec in ((C.load(f"eval_{st}") or {}).get("identities") or {}).items():
            out.setdefault(f"policy:{spec}", {**rec, "role": f"read by {st}"})
    man = C.load("manifest_main_20261017-204945") or C.load("manifest_preview") or {}
    for s, rec in (man.get("start_checkpoints") or {}).items():
        out[f"start:{s}"] = {"path": rec["path"], "file_sha256": rec["file_sha256"], "role": "ext/baseline:s"}
    for n, rec in (man.get("library_networks") or {}).items():
        out[f"library:{n}"] = {"path": rec["path"], "file_sha256": rec["file_sha256"], "role": "OM2 library"}
    if man.get("archive"):
        out["archive"] = {**man["archive"], "role": "M4.6c start archive"}
    return out


def records() -> Dict[str, dict]:
    """Per record directory: file count, bytes and a digest over the sorted (relative path, SHA-256) list; the full
    list goes to inventory_files.json.gz."""
    base = os.path.join(C.CKPT)
    out, files = {}, {}
    for d in RECORD_DIRS:
        root = os.path.join(base, d)
        rows = []
        for dp, _, fs in os.walk(root):
            if "ready" in os.path.relpath(dp, root).split(os.sep)[0]:
                continue
            for f in fs:
                if f.endswith(".pt"):                      # networks are listed by identity (`networks`)
                    continue
                p = os.path.join(dp, f)
                rows.append((os.path.relpath(p, base).replace("\\", "/"), C.file_sha256(p), os.path.getsize(p)))
        rows.sort()
        import hashlib
        h = hashlib.sha256("\n".join(f"{r} {s}" for r, s, _ in rows).encode()).hexdigest()
        out[d.replace("\\", "/")] = {"files": len(rows), "bytes": int(sum(b for *_, b in rows)), "digest": h}
        files[d.replace("\\", "/")] = [[r, s] for r, s, _ in rows]
    return out, files


def inventory() -> dict:
    t0 = time.perf_counter()
    rec, files = records()
    pre = C.load("preregistration") or {}
    data = {"meta": C.meta(stage="reproduce"), "preregistration_active": C.active_version(pre) if pre else None,
            "code": C.code_digest(), "seeds": {"pairs": list(C.SEEDS), **C.ROOTS},
            "networks": networks(), "records": rec, "stages": [list(s) for s in STAGES],
            "wall_s": time.perf_counter() - t0}
    C.save_json("inventory", data)
    with gzip.open(os.path.join(C.OUT, "inventory_files.json.gz"), "wt", encoding="utf-8") as f:
        json.dump(files, f)
    write_markdown(data)
    return data


def write_markdown(d: dict) -> str:
    from .report import _t, runtime_actual
    act = d.get("preregistration_active") or {}
    rows = [[s, f"`python -m lanegame.experiments.rl.m5 {c}{' ' + fl if fl else ''} --workers 6`", w]
            for s, c, fl, w in d["stages"]]
    rt = runtime_actual()
    rrows = [[k, f"{v['wall_h']:.2f}"] for k, v in rt.items() if isinstance(v, dict) and "wall_h" in v]
    total = sum(v["wall_h"] for v in rt.values() if isinstance(v, dict) and "wall_h" in v)
    rrows.append(["all campaign stages", f"{total:.2f}"])
    nets = d["networks"]
    key = [k for k in nets if k.startswith(("train:", "attacker:", "start:")) and (k.endswith(":final") or
                                                                                  not k.startswith("train:"))]
    nrows = [[k, nets[k]["path"], f"`{(nets[k]['file_sha256'] or 'missing')[:16]}…`"] for k in sorted(key)]
    recs = [[k, v["files"], f"{v['bytes'] / 2 ** 20:,.1f}", f"`{v['digest'][:16]}…`"] for k, v in d["records"].items()]
    text = "\n".join([
        "# M5 reproducibility", "",
        "*Generated by `python -m lanegame.experiments.rl.m5 reproduce`. The full file list with SHA-256 is "
        "`results/m5/inventory_files.json.gz`. Every network identity is in `results/m5/inventory.json`.*", "",
        f"- The protocol is `docs/m5/m5_preregistration.md`, version {act.get('version')} (SHA-256 "
        f"`{act.get('sha256')}`). Its amendment chain is in `results/m5/preregistration.json`.",
        f"- The code at generation has sources SHA-256 `{d['code']['sources_sha256']}` over "
        f"{d['code']['python_files']} files, at commit `{d['code']['git_commit']}` with a "
        f"{'dirty' if d['code']['working_tree_dirty'] else 'clean'} working tree.",
        "- The sp arm uses `GameConfig.m45_calibrated()`. The om arm and every attacker use "
        "`m45_calibrated(obs_last_action=True)`.",
        "- The seed roots are " + ", ".join(f"{k} {v}" for k, v in d["seeds"].items()) + ".",
        "- `TRACE_ROOT` 70,900 is reserved and unused, because the traces replay games of the evaluation cells.", "",
        "## Two levels of reproduction", "",
        "1. **From the saved records.** `prereg_check`, `train_check`, `attack_check`, `analyse` and `report` "
        "re-derive every table in minutes, with no training and no game. They need the git-ignored records and run "
        "summaries listed below.",
        "2. **The campaign.** Run the stages below in order, each behind its approval flag. The campaign stages took "
        f"{total:.1f} h on six workers, plus the readiness checks. With the same code and seeds each stage reproduces "
        "bit for bit. Readiness parts R4, A1 and C1 test that for training and the attackers.", "",
        "## Stages", "", _t(["stage", "command", "produces"], rows), "",
        "## Measured wall time", "", _t(["stage", "wall h"], rrows), "",
        "## Git-ignored records (results/rl/checkpoints/m5/)", "",
        _t(["directory", "files", "MiB", "digest of (path, SHA-256) list"], recs), "",
        "## Key checkpoints", "", _t(["role", "path", "SHA-256"], nrows), ""])
    C.write_md("reproducibility", text)
    return text


__all__ = ["inventory", "networks", "records", "write_markdown", "STAGES", "RECORD_DIRS"]
