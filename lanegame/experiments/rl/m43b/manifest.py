"""Artifact manifest and checkpoint bundle of the M4.3b pilot and its follow-up (brief §7).

`write_manifest()` writes results/rl/m43b/MANIFEST.json: every result file of the study (tracked in Git) with its
size, SHA-256 and the commit / stage / command that produced it (from the file's own `meta` block where it has one),
and the inventory of the UNTRACKED checkpoints (results/rl/checkpoints/m43b/: size, SHA-256 of the file, the
weights digest the loader verifies, the run configuration), so that a reader can tell which checkpoint a number
came from and whether a copy is intact.

`bundle_checkpoints()` packs the checkpoints an evaluation needs without retraining (every run's final.pt, run.json
and history.jsonl, and the attackers) into one tar.gz under results/rl/m43b_artifacts/ (git-ignored), named by the
commit; with `full=True` the 80 training snapshots per run are included too (they are only needed to regenerate the
learning curves and the self-play progress matchups). `verify_bundle()` checks a bundle against the manifest.

`compress_results()` gzips the large JSON results (>= LARGE_BYTES) in place; every reader of this package
(`common.load`, `env.starts.load_archive`) accepts either form.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import tarfile
import time
from typing import Dict, List, Optional

from .common import CKPT, OUT, git_commit, read_json

MANIFEST = os.path.join(OUT, "MANIFEST.json")
ARTIFACT_DIR = os.path.join("results", "rl", "m43b_artifacts")
LARGE_BYTES = 1_000_000
STAGE_COMMANDS = {
    "train": "python -m lanegame.experiments.rl.m43b train --workers 9",
    "curves": "python -m lanegame.experiments.rl.m43b curves --workers 9",
    "headline": "python -m lanegame.experiments.rl.m43b headline --workers 9",
    "exploit": "python -m lanegame.experiments.rl.m43b exploit --workers 9",
    "behavior": "python -m lanegame.experiments.rl.m43b behavior --workers 9",
    "counterfactual": "python -m lanegame.experiments.rl.m43b counterfactual --workers 9",
    "plans": "python -m lanegame.experiments.rl.m43b plans --workers 4",
    "recall": "python -m lanegame.experiments.rl.m43b recall --workers 9",
    "trajectories": "python -m lanegame.experiments.rl.m43b trajectories",
    "archive": "python -m lanegame.experiments.rl.m43b archive",
    "validation.wave_cf": "python -m lanegame.experiments.rl.m43b val_wave_cf --workers 9",
    "validation.games": "python -m lanegame.experiments.rl.m43b val_games --workers 9",
    "validation.horizon": "python -m lanegame.experiments.rl.m43b val_horizon --workers 9",
    "validation.freeze": "python -m lanegame.experiments.rl.m43b val_freeze --workers 9",
}


def sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _open_json(path: str) -> Optional[dict]:
    try:
        if path.endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return json.load(f)
        if path.endswith(".json"):
            return read_json(path)
    except (OSError, ValueError):
        return None
    return None


def _meta_of(path: str) -> dict:
    d = _open_json(path) if path.endswith((".json", ".json.gz")) else None
    if not isinstance(d, dict):
        return {}
    m = d.get("meta") or d.get("generator") or {}
    out = {}
    for k in ("commit", "timestamp", "stage", "window", "seed", "states", "rollouts", "episodes"):
        if k in m:
            out[k] = m[k]
    if "module" in m:                                   # the start archive's generator block
        out["stage"] = "archive"
        out.update({k: m[k] for k in ("seed", "states", "games_played") if k in m})
    if "stage" in out and out["stage"] in STAGE_COMMANDS:
        out["command"] = STAGE_COMMANDS[out["stage"]]
    return out


def result_files() -> List[str]:
    out = []
    for root, _, files in os.walk(OUT):
        for f in files:
            p = os.path.join(root, f)
            if f == os.path.basename(MANIFEST):
                continue
            out.append(p.replace(os.sep, "/"))
    return sorted(out)


def checkpoint_inventory() -> Dict[str, dict]:
    """Every checkpoint under results/rl/checkpoints/m43b/ with its SHA-256 and the loader's weights digest."""
    inv = {}
    for root, _, files in os.walk(CKPT):
        for f in files:
            p = os.path.join(root, f).replace(os.sep, "/")
            rec = {"bytes": os.path.getsize(p), "sha256": sha256(p)}
            if f.endswith(".pt"):
                try:
                    import torch
                    ck = torch.load(p, map_location="cpu", weights_only=False)
                    rec["weights_digest"] = ck.get("digest") or ck.get("weights_digest")
                    rec["kind"] = "snapshot" if "/snapshots/" in p else ("attacker" if "/attackers/" in p else "final")
                except Exception as e:                    # pragma: no cover - a corrupt file is reported, not fatal
                    rec["error"] = f"{type(e).__name__}: {e}"
            inv[p] = rec
    return inv


def write_manifest(include_snapshots: bool = False) -> dict:
    t0 = time.perf_counter()
    files = {}
    for p in result_files():
        files[p] = {"bytes": os.path.getsize(p), "sha256": sha256(p), **_meta_of(p)}
    inv = checkpoint_inventory()
    ck = {p: r for p, r in inv.items() if include_snapshots or "/snapshots/" not in p}
    n_snap = sum(1 for p in inv if "/snapshots/" in p)
    runs = {}
    for p in inv:
        if p.endswith("run.json"):
            d = _open_json(p) or {}
            runs[os.path.basename(os.path.dirname(p))] = {k: d.get(k) for k in ("condition", "seed", "ppo", "shared")
                                                          if k in d}
    m = {"schema": "lanegame.m43b_manifest", "version": 1, "commit": git_commit(),
         "written": time.strftime("%Y-%m-%d %H:%M:%S"),
         "notes": ["results/rl/m43b/** is tracked in Git (large JSON files gzipped: readers accept both forms)",
                   "results/rl/checkpoints/m43b/** is NOT tracked (.gitignore); preserve it through the bundle",
                   "the weights digest is what lanegame.rl.load_agent verifies; equal digests <=> identical weights",
                   f"{n_snap} training snapshots exist locally; they are listed only with include_snapshots"],
         "results": files, "checkpoints": ck, "runs": runs,
         "bundle": {"command": "python -m lanegame.experiments.rl.m43b manifest   # also writes the bundle",
                    "directory": ARTIFACT_DIR},
         "elapsed_s": time.perf_counter() - t0}
    os.makedirs(OUT, exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=1)
    print(f"wrote {MANIFEST}: {len(files)} result files, {len(ck)} checkpoint files listed", flush=True)
    return m


def bundle_checkpoints(full: bool = False, out_dir: str = ARTIFACT_DIR) -> str:
    """results/rl/m43b_artifacts/m43b_checkpoints[_full]_<commit7>.tar.gz; returns its path. Not tracked by Git."""
    os.makedirs(out_dir, exist_ok=True)
    tag = git_commit()[:7]
    path = os.path.join(out_dir, f"m43b_checkpoints{'_full' if full else ''}_{tag}.tar.gz")
    n = 0
    with tarfile.open(path, "w:gz") as tar:
        for root, _, files in os.walk(CKPT):
            if not full and os.path.basename(root) == "snapshots":
                continue
            for f in files:
                p = os.path.join(root, f)
                tar.add(p, arcname=os.path.relpath(p, os.path.dirname(CKPT)).replace(os.sep, "/"))
                n += 1
    with open(path + ".sha256", "w", encoding="utf-8") as f:
        f.write(f"{sha256(path)}  {os.path.basename(path)}\n")
    print(f"wrote {path} ({n} files, {os.path.getsize(path) / 1e6:.1f} MB)", flush=True)
    return path


def verify_bundle(path: str, manifest: Optional[dict] = None) -> dict:
    """Extract nothing; compare the SHA-256 of each member with the manifest's checkpoint inventory."""
    manifest = manifest or read_json(MANIFEST)
    ck = manifest["checkpoints"]
    ok, missing, bad = 0, [], []
    with tarfile.open(path, "r:gz") as tar:
        for mem in tar.getmembers():
            if not mem.isfile():
                continue
            key = f"{os.path.dirname(CKPT)}/{mem.name}".replace(os.sep, "/")
            rec = ck.get(key)
            if rec is None:
                missing.append(mem.name)
                continue
            h = hashlib.sha256()
            fobj = tar.extractfile(mem)
            for b in iter(lambda: fobj.read(1 << 20), b""):
                h.update(b)
            if h.hexdigest() == rec["sha256"]:
                ok += 1
            else:
                bad.append(mem.name)
    return {"verified": ok, "not_in_manifest": missing, "mismatched": bad}


def compress_results(threshold: int = LARGE_BYTES, dry_run: bool = False) -> List[str]:
    """Gzip every .json result of the study at or above `threshold` bytes (the raw file is removed)."""
    done = []
    for p in result_files():
        if p.endswith(".json") and os.path.getsize(p) >= threshold:
            gz = p + ".gz"
            if not dry_run:
                with open(p, "rb") as fi, gzip.open(gz, "wb", compresslevel=6) as fo:
                    shutil.copyfileobj(fi, fo)
                os.remove(p)
            done.append(gz)
    return done


__all__ = ["write_manifest", "bundle_checkpoints", "verify_bundle", "compress_results", "checkpoint_inventory",
           "sha256", "MANIFEST", "ARTIFACT_DIR", "LARGE_BYTES"]
