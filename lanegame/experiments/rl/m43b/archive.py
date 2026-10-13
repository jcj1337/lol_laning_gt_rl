"""The training start-state archive of the mixed-start follow-up (M4.3b §7 of the brief; design §10.4 rho_train).

Self-play from s0 visits the states the current policy pair produces and nothing else. The follow-up condition
`mixed` starts half of its training games (PPOConfig.start_mix = 0.5) from a uniformly drawn state of an archive of
WITNESSED reachable states (lanegame.env.starts: every state carries the environment seed and the action prefix that
reproduce it from s0, and `verify_witness` replays it), the other half from s0. The evaluation objective is unchanged
(protocol E1 from s0; the archive is a training device only).

Composition. States are sampled from games of three kinds of generator, at turns 12 .. 400 of 460-turn games:
reference-policy pairs (the suite of `references`, which produce slow pushes, Clears, freezes, sieges, crash-and-
Recall cycles and stand-offs), self-play of the untrained initial network of seed 0 (near-uniform play: wide
coverage), and self-play of the baseline seed-0 final (the states the learned policy itself visits). Sampling is
STRATIFIED by situation (`behavior.situations`: a neutral lane, own / enemy wave pushing, a stacked front, an imminent
crash, defending at the tower, sieging, a freeze-like lane, the enemy absent, low HP, and "other") so that the rarer
strategic situations are represented, then closed under the seat mirror (the mirrored witness is the swapped action
prefix; the configuration is seat-symmetric). The archive reports its composition by situation and generator.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from ....config import CHAMP_ABSENT
from ....env import LaneEnv
from ....env.starts import StartState, close_under_mirror, mirror_closed, save_archive, verify_witness
from .behavior import turn_features, situations, FEATURE_KEYS
from .common import ARCHIVE_SEED, OUT, PRIMARY, agent_from_spec, cfg_game, meta, save_json
from .references import HEAD_REFS

ARCHIVE_PATH = os.path.join(OUT, "archive_train.json.gz")       # gzipped since the follow-up (load_archive reads both)
TARGETS = {"neutral": 60, "own_pushing": 60, "enemy_pushing": 60, "stacked": 30, "crash_imminent": 30, "defending": 50,
           "sieging": 40, "freeze_like": 40, "enemy_absent": 40, "low_hp": 40, "other": 50}
CLASS_OF = {"neutral": "neutral lane (fronts fighting at the lane cell, equal power)",
            "own_pushing": "own wave pushing (own front in the enemy half)",
            "enemy_pushing": "enemy wave pushing (enemy front in the own half)",
            "stacked": "own front stacked (>= 16 HP) in contact",
            "crash_imminent": "own crash imminent (own front next to the enemy tower)",
            "defending": "defending (enemy mass under the own tower)",
            "sieging": "sieging (own mass under the enemy tower, champion there)",
            "freeze_like": "freeze-like (enemy front at own -1, own front in the own tower cell)",
            "enemy_absent": "enemy champion absent",
            "low_hp": "low HP (<= 40%), enemy present"}
TURN_LO, TURN_HI = 12, 400


def generators(cfg) -> List[tuple]:
    """(label, agent A, agent B) of the games the archive samples from."""
    refs = {r: agent_from_spec(f"ref:{r}", cfg) for r in HEAD_REFS}
    out = []
    rng = np.random.default_rng(ARCHIVE_SEED)
    names = list(HEAD_REFS)
    for _ in range(24):
        a, b = names[int(rng.integers(len(names)))], names[int(rng.integers(len(names)))]
        out.append((f"ref:{a}|ref:{b}", refs[a], refs[b]))
    init = agent_from_spec("initial:0", cfg)
    for _ in range(8):
        out.append(("initial:0|initial:0", init, init))
    try:
        fin = agent_from_spec(f"final:{PRIMARY}:0", cfg)
        for _ in range(8):
            out.append((f"final:{PRIMARY}:0|final:{PRIMARY}:0", fin, fin))
    except (FileNotFoundError, KeyError, OSError):                   # the archive can be built before training
        pass
    return out


def _classify(f: dict) -> str:
    """The first matching class of TARGETS for a present champion's features (one row -> one class)."""
    T = 2
    checks = [("freeze_like", f["en_fh"] > 0 and f["en_fx"] == -1 and f["own_fh"] > 0 and f["own_fx"] == -T),
              ("sieging", f["own_siege"] > 0 and f["at_enemy_tower"]),
              ("defending", f["en_siege"] > 0),
              ("crash_imminent", f["own_fh"] > 0 and f["own_fx"] == T - 1),
              ("stacked", f["own_fh"] >= 16 and f["contact"]),
              ("enemy_absent", f["enemy_absent"]),
              ("low_hp", f["hp_frac"] <= 0.4 and not f["enemy_absent"]),
              ("own_pushing", f["own_fh"] > 0 and f["own_fx"] >= 1),
              ("enemy_pushing", f["en_fh"] > 0 and f["en_fx"] <= -1),
              ("neutral", f["contact"] and f["contact_cell"] == 0 and abs(f["power"] - 1) <= 0.1)]
    for name, ok in checks:
        if ok:
            return name
    return "other"


def build_archive(cfg=None, turns: int = 460, seed: int = ARCHIVE_SEED, targets: Optional[Dict[str, int]] = None,
                  path: str = ARCHIVE_PATH, verify: bool = True) -> dict:
    cfg = cfg if cfg is not None else cfg_game()
    targets = dict(targets or TARGETS)
    rng = np.random.default_rng(seed)
    env = LaneEnv(cfg, max_turns=turns, cache_size=20_000)
    found: Dict[str, List[StartState]] = {k: [] for k in targets}
    by_gen: Dict[str, int] = {}
    t0 = time.perf_counter()
    gens = generators(cfg)
    game = 0
    while any(len(found[k]) < targets[k] for k in targets) and game < 400:
        label, x, y = gens[game % len(gens)]
        env_seed = int(rng.integers(2 ** 31))
        rng_a, rng_b = (np.random.default_rng(s) for s in np.random.SeedSequence(env_seed).spawn(2))
        for ag in (x, y):
            if callable(getattr(ag, "reset", None)):
                ag.reset()
        obs = env.reset(seed=env_seed)
        actions: List[List[int]] = []
        cands: List[tuple] = []
        for t in range(turns):
            if TURN_LO <= t <= TURN_HI:
                for s in (0, 1):
                    if obs[s].legal != (CHAMP_ABSENT,):
                        f = turn_features(obs[s], cfg)
                        cands.append((t, s, _classify(f), env.state, env.champ, env.k % cfg.phase_period, list(actions)))
            a, b = x.act(obs[0], rng_a), y.act(obs[1], rng_b)
            r = env.step(a, b)
            for ag, s in ((x, 0), (y, 1)):
                nf = getattr(ag, "notify", None)
                if callable(nf):
                    nf(r.info, s)
            actions.append([int(a), int(b)])
            if r.done:
                break
            obs = r.obs
        game += 1
        order = rng.permutation(len(cands))
        took: Dict[str, int] = {}
        for i in order:
            t, s, cls, lane, champ, phase, prefix = cands[i]
            if cls not in targets or len(found[cls]) >= targets[cls] or took.get(cls, 0) >= 3:
                continue
            st = StartState(lane, champ, phase, {"seed": env_seed, "actions": prefix}, f"{cls}|{label}|k{t}|seat{s}")
            found[cls].append(st)
            took[cls] = took.get(cls, 0) + 1
            by_gen[label.split("|")[0].split(":")[0]] = by_gen.get(label.split("|")[0].split(":")[0], 0) + 1
    states = [s for k in targets for s in found[k]]
    # distinct decision situations only
    seen, uniq = set(), []
    for s in states:
        if s.key() not in seen:
            seen.add(s.key())
            uniq.append(s)
    closed = close_under_mirror(uniq)
    assert mirror_closed(closed)
    if verify:
        for s in closed:
            assert verify_witness(cfg, s), s.label
    composition = {k: len(found[k]) for k in targets}
    gen_counts = {}
    for s in closed:
        g = s.label.split("|")[1].split(":")[0]
        gen_counts[g] = gen_counts.get(g, 0) + 1
    info = {"states": len(closed), "distinct_before_mirror": len(uniq), "games_played": game,
            "composition_by_class": composition, "targets": targets, "by_generator": gen_counts,
            "turn_range": [TURN_LO, TURN_HI], "mirror_closed": True, "witnesses_verified": verify,
            "elapsed_s": time.perf_counter() - t0, "seed": seed}
    save_archive(path, closed, cfg, generator={"module": __name__, **info})
    print(f"archive: {len(closed)} states ({len(uniq)} before mirroring) from {game} games in {info['elapsed_s']:.0f} s; "
          f"{composition}", flush=True)
    return info


def archive_stage() -> dict:
    info = build_archive()
    save_json("archive", {"meta": meta(stage="archive"), "path": ARCHIVE_PATH, **info})
    return info


__all__ = ["build_archive", "archive_stage", "ARCHIVE_PATH", "TARGETS"]
