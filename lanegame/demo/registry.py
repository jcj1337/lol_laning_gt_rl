"""The model registry of the M6 demo (docs/m6/m6_plan.md §3 rule 7, §4.1 `ModelCard`).

A registry entry is a `ModelCard`: one trained M5 policy that a session may seat, with its arm, seed, configuration,
input width, checkpoint file and the files it needs at serve time. `om:0` needs the 5 learned networks of its belief's
library (`om2.common.LEARNED`) besides its own checkpoint, so the demo depends on 7 files in all.

Trust. `torch.load` unpickles the files, so a file must be checked before it is read as a
checkpoint. `manifest.json` (tracked, beside this module) holds each file's SHA-256, copied from
`results/m5/inventory.json`. `Registry.load` reads every file's bytes once, hashes those bytes, refuses the whole
registry on any mismatch or missing file, and only then unpickles the same bytes (no second read, so nothing can change
between the check and the load). The weights digest stored in each checkpoint is verified as well
(`rl.load_checkpoint`).

The verified networks replace the entries of the M5 loaders' caches: `players.install` for the two finals and
`om2.common._NETS` for the library, so `OMPolicy`'s lazy library load (`players.belief_types`) reads only checked
objects. Sessions then build their policies with M5's own `players.crn_policy`, exactly as M5 evaluated them
(sampled, not greedy). Torch runs on one thread, as M5 set it.

Scripted seats are the six library archetypes of OM2 (`om2.common.ARCHETYPES`); they need no file. They exist in the
core so that the two seed-0 research recordings against `g_shove` can be re-simulated. Offering them to visitors is an
open design decision (docs/m6/m6a_decision_record.md).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifest.json")
MANIFEST_SCHEMA = "lanegame.demo_manifest"

_CK = "results/rl/checkpoints"
LIBRARY_FILES = {                     # om2.common.LEARNED, in the belief's type order
    "peer": f"{_CK}/m45c/full/s3/final.pt",
    "attacker": f"{_CK}/m45c/attack/ext_s3/final.pt",
    "freezer": f"{_CK}/m46a/e4/freeze_s2/final.pt",
    "gapper": f"{_CK}/m46a/e4/gap_s2/final.pt",
    "short": f"{_CK}/m47/train/main/short/s0/final.pt",
}
SCRIPTED_STYLES = ("g_farm", "g_safe", "g_bully", "g_shove", "fz_guard_sh", "g_shove_ward")   # om2.common.ARCHETYPES


class ChecksumError(RuntimeError):
    """A checkpoint file is missing or differs from the trusted manifest; nothing was loaded."""


@dataclass(frozen=True)
class ModelCard:
    id: str                          # the M5 policy specification, e.g. "sp:0"
    arm: str                         # "sp" (pure self-play, M5 arm B) | "om" (opponent modelling, M5 arm A)
    seed: int
    label: str
    description: str
    config: str                      # the GameConfig the policy observes
    inputs: int                      # network input width (90 or 116)
    path: str                        # the checkpoint, relative to the registry root
    dependencies: Tuple[str, ...] = ()
    training: str = ""
    evidence: Tuple[str, ...] = ()
    file_sha256: str = ""            # filled in from the manifest
    weights_digest: str = ""         # filled in by Registry.load (the checkpoint's own digest, verified)

    def public(self) -> dict:
        return {"id": self.id, "arm": self.arm, "seed": self.seed, "label": self.label,
                "description": self.description, "config": self.config, "inputs": self.inputs,
                "file_sha256": self.file_sha256, "weights_digest": self.weights_digest,
                "dependencies": list(self.dependencies), "training": self.training, "evidence": list(self.evidence)}


_TRAINING = ("continues ext/baseline:0 (24,576,000 earlier agent transitions) by 32,768,000 agent transitions in M5; "
             "the seed-0 final, chosen by index as M5's demonstration pair, not by performance")
CARDS = (
    ModelCard("sp:0", "sp", 0, "Pure self-play",
              "M5 arm B: trained only against itself. Stronger against competent opponents in M5's evaluation.",
              "m45_calibrated", 90, f"{_CK}/m5/train/main/sp/s0/final.pt", training=_TRAINING,
              evidence=("docs/m5/m5e_closeout_report.md#7-configuration-for-m6", "results/m5/analysis_main.json")),
    ModelCard("om:0", "om", 0, "Opponent modelling",
              "M5 arm A: keeps a Bayesian belief about its opponent's style and sees the opponent's previous action; "
              "half of its training games were against 15 fixed opponents. Gains most against recognizable scripted "
              "styles in M5's evaluation.",
              "m45_calibrated(obs_last_action=True)", 116, f"{_CK}/m5/train/main/om/s0/final.pt",
              dependencies=tuple(LIBRARY_FILES.values()), training=_TRAINING,
              evidence=("docs/m5/m5e_closeout_report.md#7-configuration-for-m6", "results/m5/analysis_main.json")),
)


def read_manifest(path: str = MANIFEST) -> Dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if d.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"{path} is not a {MANIFEST_SCHEMA} file")
    return d["files"]


def verified_bytes(root: str, rel: str, sha256: str) -> bytes:
    """The file's bytes, read once, if their SHA-256 equals the manifest's; ChecksumError otherwise."""
    p = os.path.join(root, *rel.split("/"))
    if not os.path.isfile(p):
        raise ChecksumError(f"{rel}: missing (the demo needs every file of lanegame/demo/manifest.json)")
    with open(p, "rb") as f:
        b = f.read()
    got = hashlib.sha256(b).hexdigest()
    if got != sha256:
        raise ChecksumError(f"{rel}: SHA-256 {got[:16]}... differs from the manifest's {sha256[:16]}...; refusing "
                            "to load any checkpoint")
    return b


@dataclass
class Registry:
    """The demo's policies. `load()` verifies all 7 files before any is unpickled (module docstring)."""
    root: str = "."
    manifest: str = MANIFEST
    cards: Dict[str, ModelCard] = field(default_factory=dict)
    loaded: bool = False

    def __post_init__(self):
        files = read_manifest(self.manifest)
        for c in CARDS:
            for rel in (c.path,) + c.dependencies:
                if rel not in files:
                    raise ValueError(f"{rel} ({c.id}) is not in the manifest")
            self.cards[c.id] = ModelCard(**{**c.__dict__, "file_sha256": files[c.path]["sha256"]})
        self._files = files
        self._nets: Dict[str, object] = {}

    def available(self) -> bool:
        """Every manifest file exists (not a verification)."""
        return all(os.path.isfile(os.path.join(self.root, *rel.split("/"))) for rel in self._files)

    def verify(self) -> Dict[str, bytes]:
        """Every manifest file's bytes, checked (ChecksumError on the first missing or altered file)."""
        return {rel: verified_bytes(self.root, rel, d["sha256"]) for rel, d in self._files.items()}

    def load(self) -> "Registry":
        if self.loaded:
            return self
        import torch
        from ..experiments.rl.m5 import common as C
        from ..experiments.rl.m5 import players as PL
        from ..experiments.rl.om2 import common as O2
        from ..rl import load_checkpoint, weights_digest
        torch.set_num_threads(1)
        blobs = self.verify()                                   # all 7 files checked before the first unpickle
        nets = {}
        for rel, b in blobs.items():
            net, _, _ = load_checkpoint(io.BytesIO(b))          # also verifies the stored weights digest
            net.eval()
            nets[rel] = (net, weights_digest(net.state_dict()))
        for c in self.cards.values():
            net, digest = nets[c.path]
            if net.obs_size != c.inputs or net.n_actions != 61:
                raise ValueError(f"{c.id}: {net.obs_size} inputs / {net.n_actions} outputs, expected {c.inputs} / 61")
            self.cards[c.id] = ModelCard(**{**c.__dict__, "weights_digest": digest})
            self._nets[c.id] = net
            PL.install(c.id, net)
        if dict(O2.LEARNED).keys() != LIBRARY_FILES.keys():
            raise ValueError("the OM2 library changed; update lanegame/demo/registry.py and its manifest")
        for name, key in O2.LEARNED.items():
            if key.replace("\\", "/") != LIBRARY_FILES[name]:
                raise ValueError(f"library type {name}: {key} is not {LIBRARY_FILES[name]}")
            net, _ = nets[LIBRARY_FILES[name]]
            if net.obs_size != 90:
                raise ValueError(f"library type {name}: {net.obs_size} inputs")
            O2._NETS[key] = net                                 # om2.common.net90's cache, keyed by its own path
        PL._TYPES["t"] = O2.belief_types()                      # built from the checked networks only
        if len(PL._TYPES["t"]) != C.K:
            raise ValueError("the belief library does not have K types")
        self.loaded = True
        return self

    def card(self, model_id: str) -> ModelCard:
        if model_id not in self.cards:
            raise KeyError(f"unknown model {model_id!r}; the registry has {sorted(self.cards)}")
        return self.cards[model_id]

    def policy(self, model_id: str, seat: int, policy_seed: int, horizon: int):
        """M5's engine policy for a seat (players.crn_policy: CRNNet for sp, OMPolicy with a live belief for om),
        sampling with the uniforms of default_rng([policy_seed, 7, seat]) for `horizon` turns."""
        from ..experiments.rl.m5 import players as PL
        if not self.loaded:
            raise RuntimeError("Registry.load() first")
        self.card(model_id)
        if PL.net_of(model_id) is not self._nets[model_id]:
            raise RuntimeError(f"{model_id}: the M5 loader's cache does not hold the verified network")
        return PL.crn_policy(model_id, 0, int(policy_seed), int(seat), int(horizon))

    def public(self) -> list:
        return [c.public() for c in self.cards.values()]


def scripted_agent(name: str):
    """A scripted library archetype as an agent (om2.common.agent), the only scripted seats the core accepts."""
    if name not in SCRIPTED_STYLES:
        raise KeyError(f"unknown scripted style {name!r}; choose from {list(SCRIPTED_STYLES)}")
    from ..experiments.rl.om2 import common as O2
    return O2.agent(name)


def scripted_description(name: str) -> str:
    """The archetype's own rule, as its strategy states it (experiments/m45b/strategies.py)."""
    from ..experiments.m45b.strategies import get
    return get(name).description


__all__ = ["ModelCard", "CARDS", "Registry", "ChecksumError", "MANIFEST", "LIBRARY_FILES", "SCRIPTED_STYLES",
           "read_manifest", "verified_bytes", "scripted_agent", "scripted_description"]
