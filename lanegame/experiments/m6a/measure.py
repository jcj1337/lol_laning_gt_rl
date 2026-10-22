"""M6a verification and cost measurements (docs/m6/m6a_technical_report.md §3 and §5).

verify    The three seed-0 M5 recordings re-simulated through the session core with M5's research convention: actions,
          `payoff_A` (exactly) and the belief's log posteriors at M5's check turns; each then replays from its app
          recording. The 12 attacker recordings name their attacker by a checkpoint path that is not in the demo
          registry, so they are replayed from their actions only (the viewer's check): seat A's total must equal
          `payoff_A`.

measure   Wall-clock costs on this machine, one torch thread (other programs share the CPU; see `load`):
            cold start      a fresh interpreter importing lanegame and loading the registry (7 files checked)
            turn            one human submission end to end: validation, the environment step, the opponent's
                            belief update, the next turn's bot decision, the turn and position frames, the view
                            with the action menu, and json.dumps of the whole response
            history         a full 150-turn history re-simulated from the record with frames, and its JSON size
            recording       the app recording of a 150-turn game (hidden belief, and after the stop with beliefs)
            simulate        a bounded bot-versus-bot game of 150 turns, then its history with frames and JSON
            memory          tracemalloc's peak and retained bytes for one 150-turn session
          Human choices are a fixed stand-in (`chooser`); the costs do not depend on what is chosen.
"""

from __future__ import annotations

import gzip
import json
import os
import statistics as st
import subprocess
import sys
import time
import tracemalloc
from typing import Dict, List

from . import common as Cm


def chooser(t: int, legal) -> int:
    legal = list(legal)
    return legal[(t * 7 + 3) % len(legal)]


def _q(xs: List[float]) -> dict:
    xs = sorted(xs)
    return {"n": len(xs), "median_ms": round(1000 * st.median(xs), 3),
            "p95_ms": round(1000 * xs[min(len(xs) - 1, int(0.95 * len(xs)))], 3), "max_ms": round(1000 * xs[-1], 3),
            "mean_ms": round(1000 * st.mean(xs), 3)}


def _size(obj) -> dict:
    b = json.dumps(obj, separators=(",", ":")).encode()
    return {"json_bytes": len(b), "gzip_bytes": len(gzip.compress(b, 6))}


def verify() -> dict:
    from ...demo import record as R
    from ...demo import session as S
    reg = Cm.registry()
    out = {"research": {}, "attackers": {}}
    for name in Cm.RESEARCH_GAMES:
        path = os.path.join(Cm.TRACES, f"{name}.json")
        t0 = time.perf_counter()
        s, cmp = S.from_research(reg, path)
        rep = S.replay(reg, s.recording())
        out["research"][name] = {**cmp, "replay": rep, "file_sha256": R.file_sha256(path),
                                 "wall_s": round(time.perf_counter() - t0, 3)}
    cfg = R.CONFIGS["m45_calibrated"]()
    for f in sorted(os.listdir(Cm.TRACES)):
        if not f.startswith("attacker_"):
            continue
        d = json.load(open(os.path.join(Cm.TRACES, f), encoding="utf-8"))
        turns, total = R.replay_actions(cfg, d["seed"], d["actions"])
        if total != d["payoff_A"]:
            raise R.ReplayMismatch(f"{f}: {total!r} != {d['payoff_A']!r}")
        out["attackers"][f[:-5]] = {"turns": len(turns), "payoff_A": total, "exact": True,
                                    "policies_recomputed": False}
    out["summary"] = {"research_exact": len(out["research"]), "attackers_exact": len(out["attackers"])}
    return out


def _cold_start() -> dict:
    code = ("import time; t0 = time.perf_counter(); import torch; t1 = time.perf_counter(); "
            "from lanegame.demo.registry import Registry; r = Registry().load(); t2 = time.perf_counter(); "
            "print(round(t1 - t0, 3), round(t2 - t1, 3))")
    runs = []
    for _ in range(3):
        o = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        a, b = (float(x) for x in o.stdout.split())
        runs.append({"import_torch_s": a, "lanegame_and_registry_s": b})
    return {"runs": runs}


def _load() -> dict:
    try:
        ps = "(Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average"
        o = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                           capture_output=True, text=True, timeout=30)
        return {"cpu_load_percent_at_start": float(o.stdout.strip()), "logical_cpus": os.cpu_count()}
    except Exception:                                          # pragma: no cover
        return {"logical_cpus": os.cpu_count()}


def measure() -> dict:
    from ...demo import session as S
    reg = Cm.registry()
    res: Dict[str, object] = {"load": _load(), "cold_start": _cold_start()}
    # ---- one turn, end to end, against each opponent
    turn, sizes, hist, recs, mem = {}, {}, {}, {}, {}
    for opp in ("sp:0", "om:0"):
        times, core_t, ser_t, nbytes, gz = [], [], [], [], []
        last = None
        for j in range(4):
            s = S.GameSession(reg, (S.Seat.human(), S.Seat.model(opp)), chance_seed=Cm.MEASURE_ROOT + j,
                              policy_seed=Cm.MEASURE_ROOT + 100 + j)
            while s.state != S.STOPPED:
                t = s.position + 1
                code = chooser(t, s.env.legal(0))
                t0 = time.perf_counter()
                r = s.submit(t, code)
                t1 = time.perf_counter()
                b = json.dumps(r, separators=(",", ":")).encode()
                t2 = time.perf_counter()
                times.append(t2 - t0), core_t.append(t1 - t0), ser_t.append(t2 - t1)
                nbytes.append(len(b)), gz.append(len(gzip.compress(b, 6)))
            last = s
        turn[opp] = {"submit_and_serialize": _q(times), "submit": _q(core_t), "serialize": _q(ser_t)}
        sizes[opp] = {"response_json_bytes_median": int(st.median(nbytes)),
                      "response_gzip_bytes_median": int(st.median(gz)), "response_json_bytes_max": max(nbytes)}
        # history of the whole game, re-simulated with frames
        ht = []
        for _ in range(5):
            t0 = time.perf_counter()
            h = last.history()
            json.dumps(h, separators=(",", ":"))
            ht.append(time.perf_counter() - t0)
        hist[opp] = {"full_150": _q(ht), **_size(h)}
        t0 = time.perf_counter()
        last.history(140, 150)
        hist[opp]["last_10_ms"] = round(1000 * (time.perf_counter() - t0), 3)
        rec = last.recording()                                # stopped at the cap: seeds and beliefs included
        recs[opp] = {"with_beliefs": _size(rec), "codes_and_digests_only": _size({k: v for k, v in rec.items()
                                                                                  if k != "beliefs"})}
        # memory of one session
        tracemalloc.start()
        a0 = tracemalloc.take_snapshot()
        s = S.GameSession(reg, (S.Seat.human(), S.Seat.model(opp)), chance_seed=Cm.MEASURE_ROOT + 9,
                          policy_seed=Cm.MEASURE_ROOT + 109)
        while s.state != S.STOPPED:
            t = s.position + 1
            s.submit(t, chooser(t, s.env.legal(0)))
        cur, peak = tracemalloc.get_traced_memory()
        a1 = tracemalloc.take_snapshot()
        retained = sum(x.size_diff for x in a1.compare_to(a0, "filename"))
        tracemalloc.stop()
        mem[opp] = {"retained_bytes_after_150_turns": int(retained), "peak_traced_bytes": int(peak)}
        del s
    res.update(turn=turn, response=sizes, history=hist, recording=recs, memory=mem)
    # ---- bounded bot-versus-bot generation (Watch)
    sim = {}
    pairs = {"om:0 vs sp:0": (S.Seat.model("om:0"), S.Seat.model("sp:0")),
             "sp:0 vs sp:0": (S.Seat.model("sp:0"), S.Seat.model("sp:0")),
             "om:0 vs om:0": (S.Seat.model("om:0"), S.Seat.model("om:0")),
             "om:0 vs g_shove": (S.Seat.model("om:0"), S.Seat.scripted("g_shove"))}
    for label, (a, b) in pairs.items():
        g, f = [], []
        for j in range(3):
            t0 = time.perf_counter()
            s = S.simulate(reg, a, b, turns=150, chance_seed=Cm.SIM_ROOT + 100 + j, policy_seed=Cm.SIM_ROOT + 600 + j)
            t1 = time.perf_counter()
            h = s.history()
            bundle = {"recording": s.recording(), **h}
            blob = json.dumps(bundle, separators=(",", ":")).encode()
            t2 = time.perf_counter()
            g.append(t1 - t0), f.append(t2 - t1)
        sim[label] = {"simulate_150": _q(g), "frames_and_json": _q(f), "bundle_json_bytes": len(blob),
                      "bundle_gzip_bytes": len(gzip.compress(blob, 6))}
    res["simulate"] = sim
    return res


__all__ = ["verify", "measure", "chooser"]
