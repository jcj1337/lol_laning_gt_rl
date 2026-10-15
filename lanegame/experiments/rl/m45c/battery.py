"""The revised M4.5c strategic counterfactual battery (horizon study §6-§7, approved methodology). PREPARED AND
SMOKE-TESTED ONLY: `run()` refuses to start without its confirmation flag, which belongs to the comprehensive-evaluation
approval.

Built on the horizon study's engine (horizon.py), generalized from the pilot to any trained policy:
    paired CRN rollouts         `horizon.CRNNet` (per-turn sampling uniforms fixed by (seed, seat, turn)), one chance
                                stream per rollout seed shared by every candidate
    multi-checkpoint recording  `horizon.rollout`: G_h, Q_h = G_h + 0.99^h V(s_h) (the EVALUATED policy's own critic),
                                the M4.5b ledger and state descriptors at every checkpoint of ONE trajectory
    candidates                  one-turn ("1t:") and sustained plans ("plan:") kept apart (`horizon.FAMILY_CANDS`)
    resumable per-job results   results/rl/checkpoints/m45c/battery/<tag>/partial/<family>__<state>__<protocol>.json.gz
                                (git-ignored); manifests, summaries and reports in results/m45c/battery/

Horizons (horizon study §6): DEFAULT_H = 50 turns; exception 1 (sustained plans longer than 10 turns: plan length +
40, capped at 100 -> the 20-turn freeze plans run 60); exception 2 (constructed states far from the policy's
experience: G50 primary, Q50 secondary; analysis rule, `primary_statistic`); exception 3 (a family failing validation
is re-run at EXTENDED_H = 150 turns for all its states, `jobs(..., extended=...)`). The validation subset (§6.3): a
fixed-seed random max(8, 15%) of every natural family's states plus every constructed cell, under protocol "policy",
run to VALIDATION_H = 300 turns (their first 50 turns are the 50-turn estimate itself); `validate()` applies the
pre-registered pass rule.

Families (focal = seat A; natural states from the evaluated policy's own self-play, seat B mirrored; constructed
states from the M4.5b builders, labelled):
    A_wave, B_freeze, B_frozen, C_trade, D_ward, E_recall    natural, as in the horizon study
    A_wave_early, C_trade_early                              natural EARLY-GAME states: games of EARLY_TURNS turns,
                                                             focal level <= 10 (the pilot found levels 1-10 are ~6-11%
                                                             of decisions, so they get their own families)
    F_freeze_c   freeze grid: power x1.76 / equal x held 12 / 20 x level 10 / 15, plus the early x1.76 cells at level 6
    G_trade_c    trade bench: level 4 / 11 / 18 x enemy front 8 / 24 HP x power equal / x1.76
    H_zone_c     zoning: neutral / A's push / B's push x A stronger (x1.76) / equal / weaker (B x1.5) at level 10,
                 and A stronger at level 6
    I_ward_c     Vision situations neutral / forward / presiege / defending at levels 10 and 6
Opponent protocols: "policy" (the evaluated policy, sampling), "modal" (its most probable legal action), and M4.5b
style strategies: freeze -> breaker wk_shove, back-off wk_back, reset wk_shove_reset; frozen -> freezer styles
fz_passive_sh / fz_guard_sh / fz_bully_sh; trade and zoning -> wary g_wary; Ward -> chaser g_bully (Trades whenever the
enemy is catchable without a dive: the M4.5b Vision study's punishing opponent).

Replication (`REPLICATION`): the freeze, frozen-on, Recall and freeze-grid families, protocols policy and modal, on
the comparison group: the six 8.192M finals when the 16.384M extension is the primary set (`run`).

Seeds (reserved campaign range): state games 937,000+ (STATE_SEED_B x 10 + family offset + game), rollouts
9,380,000+ (ROLL_SEED_B x 100 + rollout index, the same K rollout seeds for every state and candidate), validation
selection default_rng(VAL_SEED_B + family index).
"""

from __future__ import annotations

import gzip
import json
import math
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import horizon as HZ
from .common import CKPT, OUT, env_cfg, meta, run_jobs, save_json

BOUT = os.path.join(CKPT, "battery")          # raw per-job results and manifests (git-ignored; ~10^4 small files)
BREPORT = os.path.join(OUT, "battery")        # manifests (copies), summaries and reports
DEFAULT_H, VALIDATION_H, EXTENDED_H, PLAN_MARGIN, PLAN_CAP = 50, 300, 150, 40, 100
N_NATURAL, N_EARLY, N_REPLICATION = 48, 24, 24
K_NATURAL, K_CONSTRUCTED = 12, 24
VAL_SHARE, VAL_MIN = 0.15, 8
EARLY_TURNS, EARLY_LEVEL = 60, 10
STATE_SEED_B, ROLL_SEED_B, VAL_SEED_B = 93_700, 93_800, 93_900
CONFIRM_FLAG = "--confirm-m45c-battery"

FREEZE_OPP = ("policy", "modal", "wk_shove", "wk_back", "wk_shove_reset")
FROZEN_OPP = ("policy", "modal", "fz_passive_sh", "fz_guard_sh", "fz_bully_sh")
TRADE_OPP = ("policy", "modal", "g_wary")
WARD_OPP = ("policy", "modal", "g_bully")
BASE_OPP = ("policy", "modal")
# family: (kind, predicate family in horizon.predicates() or None, candidate table, protocols)
FAMILIES: Dict[str, Tuple[str, Optional[str], str, Tuple[str, ...]]] = {
    "A_wave": ("natural", "A_wave", "A_wave", BASE_OPP),
    "A_wave_early": ("early", "A_wave", "A_wave", BASE_OPP),
    "B_freeze": ("natural", "B_freeze", "B_freeze", FREEZE_OPP),
    "B_frozen": ("natural", "B_frozen", "B_frozen", FROZEN_OPP),
    "C_trade": ("natural", "C_trade", "C_trade", TRADE_OPP),
    "C_trade_early": ("early", "C_trade", "C_trade", TRADE_OPP),
    "D_ward": ("natural", "D_ward", "D_ward", WARD_OPP),
    "E_recall": ("natural", "E_recall", "E_recall", BASE_OPP),
    "F_freeze_c": ("constructed", None, "F_freeze_c", FREEZE_OPP),
    "G_trade_c": ("constructed", None, "G_trade_c", TRADE_OPP),
    "H_zone_c": ("constructed", None, "H_zone_c", TRADE_OPP),
    "I_ward_c": ("constructed", None, "I_ward_c", WARD_OPP),
}
GRIDS = {
    "F_freeze_c": [(gap, held, lvl) for gap in ("large", "equal") for held in (12, 20) for lvl in (10, 15)]
                  + [("large", held, 6) for held in (12, 20)],
    "G_trade_c": [(lvl, w, gap) for lvl in (4, 11, 18) for w in (8, 24) for gap in ("equal", "large")],
    "H_zone_c": [(sit, gap, 10) for sit in ("neutral", "a_push", "b_push") for gap in ("large", "equal", "weak")]
                + [(sit, "large", 6) for sit in ("neutral", "a_push", "b_push")],
    "I_ward_c": [(sit, lvl) for lvl in (10, 6) for sit in ("neutral", "forward", "presiege", "defending")],
}
REPLICATION = {"families": ("B_freeze", "B_frozen", "E_recall", "F_freeze_c"), "protocols": BASE_OPP,
               "n_natural": N_REPLICATION}


def candidates(fam: str) -> Dict[str, tuple]:
    return HZ.FAMILY_CANDS[FAMILIES[fam][2]]


def family_horizon(fam: str) -> int:
    """Exception 1: max plan length + PLAN_MARGIN, capped at PLAN_CAP, for plans longer than 10 turns; else 50."""
    longest = max(spec[2] for spec in candidates(fam).values())
    return DEFAULT_H if longest <= 10 else min(longest + PLAN_MARGIN, PLAN_CAP)


def primary_statistic(fam: str) -> str:
    """Exception 2: G for constructed families (the critic is not trusted far from experience), Q otherwise."""
    return "G" if FAMILIES[fam][0] == "constructed" else "Q"


def checkpoints(horizon: int) -> Tuple[int, ...]:
    cps = tuple(c for c in HZ.LONG_CP if c <= horizon)
    return cps if horizon in cps else cps + (horizon,)


def constructed_state(cfg, fam: str, cell):
    from ...m45b import states as S
    if fam == "F_freeze_c":
        gap, held, lvl = cell
        return S.freeze_state(cfg, level=lvl, gap=gap, held=held)
    if fam == "G_trade_c":
        lvl, w, gap = cell
        return S.trade_bench(cfg, lvl, w, "cover", gap=gap)
    if fam == "H_zone_c":
        sit, gap, lvl = cell
        return S.zoning_state(cfg, sit, level=lvl, gap=gap)
    if fam == "I_ward_c":
        sit, lvl = cell
        return S.vision_state(cfg, sit, level=lvl)
    raise ValueError(fam)


def validation_indices(fam: str, n: int) -> List[int]:
    """The pre-selected validation subset of a natural family with n states (fixed seed, independent of results)."""
    if n <= 0:
        return []
    m = min(n, max(VAL_MIN, math.ceil(VAL_SHARE * n)))
    rng = np.random.default_rng(VAL_SEED_B + list(FAMILIES).index(fam))
    return sorted(int(i) for i in rng.choice(n, size=m, replace=False))


# ---------------------------------------------------------------------------------------------- policies
_NETS: Dict[str, object] = {}


def _net(spec: str):
    if spec not in _NETS:
        from .common import agent_from_spec
        _NETS[spec] = agent_from_spec(spec, env_cfg()).net
    return _NETS[spec]


def _level(cfg, m4, s: int) -> int:
    from ....champions_m4 import context
    tb = context(cfg).t
    return int(tb.level_by_xp[min(m4.side(s).xp, tb.xp_cap)])


def _early(pred):
    def p(cfg, lane, m4, s):
        return _level(cfg, m4, s) <= EARLY_LEVEL and pred(cfg, lane, m4, s)
    return p


def collect_job(args) -> tuple:
    """(spec, family, n): natural states of a family from the policy's own self-play (both seats sampling)."""
    spec, fam, n = args
    import torch
    torch.set_num_threads(1)
    from ...m45b import engine as E
    from ...m45b import states as S
    from ...m45b.reference import NetPolicy
    from .common import agent_from_spec
    cfg = env_cfg()
    agent = agent_from_spec(spec, cfg)
    kind, pfam = FAMILIES[fam][:2]
    pred = HZ.predicates()[pfam]
    turns, min_turn = (EARLY_TURNS, 3) if kind == "early" else (300, 20)
    if kind == "early":
        pred = _early(pred)
    off = list(FAMILIES).index(fam) * 1000
    out, games = [], 0
    for g in range(400):
        sd = STATE_SEED_B * 10 + off + g
        out += E.collect(cfg, S.s0(cfg), NetPolicy(agent, 0, sd, 0), NetPolicy(agent, 0, sd, 1), [sd], turns, pred,
                         per_game=1, min_turn=min_turn, rng_seed=sd)
        games += 1
        if len(out) >= n:
            break
    return (spec, fam), {"states": out[:n], "games": games}


def _opponent(proto: str, net, k0: int, seed: int, horizon: int):
    from ...m45b.strategies import get
    if proto == "policy":
        return HZ.CRNNet(net, k0, seed, 1, horizon)
    if proto == "modal":
        return HZ.CRNNet(net, k0, seed, 1, horizon, greedy=True)
    return get(proto)


def job_path(tag: str, fam: str, idx: int, proto: str, horizon: int, root: str = BOUT) -> str:
    sfx = "" if horizon in (DEFAULT_H, VALIDATION_H, family_horizon(fam)) else f"__h{horizon}"
    return os.path.join(root, tag, "partial", f"{fam}__{idx}__{proto}{sfx}.json.gz")


def state_job(args) -> tuple:
    """(spec, tag, family, index, start, protocol, horizon, label, K, root): every candidate x K paired rollouts from
    one start, the policy's own choice as the baseline (horizon.state_job's record format, plus spec and tag)."""
    spec, tag, fam, idx, start, proto, horizon, label, K, root = args
    path = job_path(tag, fam, idx, proto, horizon, root)
    key = f"{tag}|{fam}|{idx}|{proto}|{horizon}"
    if os.path.exists(path):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return key, json.load(f)
    import torch
    torch.set_num_threads(1)
    from ...m45b import engine as E
    cfg = env_cfg()
    net = _net(spec)
    k0 = start[2]
    cps = checkpoints(horizon)
    lcps = [c for c in HZ.LEDGER_CP if c <= horizon]
    keys = [f"{k}_{x}" for k in HZ.LEDGER_KEYS for x in ("A", "B")] + list(HZ.STATE_KEYS) + ["V"]
    cands = {"policy": None, **candidates(fam)}
    series = FAMILIES[fam][2] in ("B_freeze", "F_freeze_c")
    t0 = time.perf_counter()
    GQ, L, SR, first = {}, {}, {}, {}
    for name, cspec in cands.items():
        G = {c: [] for c in cps}
        Q = {c: [] for c in cps}
        led = {c: np.full((K, len(keys)), np.nan) for c in lcps}
        ser = {kk: np.zeros(HZ.SERIES_T) for kk in HZ.SERIES_KEYS} if series else None
        codes = []
        for r in range(K):
            sd = ROLL_SEED_B * 100 + r
            own = HZ.CRNNet(net, k0, sd, 0, horizon)
            pa = own if cspec is None else HZ.build_candidate(cspec, own)
            pb = _opponent(proto, net, k0, sd, horizon)
            codes.append(int(E.decide(pa, cfg, start[0], start[1], 0, 0)))
            out = HZ.rollout(cfg, start, pa, pb, sd, horizon, cps, critic=net, series=series)
            for c, row in out["cp"].items():
                G[c].append(row["G"])
                Q[c].append(row["Q"])
                if c in led:
                    led[c][r] = [row.get(kk, np.nan) for kk in keys]
            if ser is not None:
                for kk in HZ.SERIES_KEYS:
                    ser[kk] += np.asarray(out["series"][kk]) / K
        GQ[name] = {"G": {str(c): v for c, v in G.items()}, "Q": {str(c): v for c, v in Q.items()}}
        L[name] = led
        SR[name] = {kk: v.tolist() for kk, v in ser.items()} if ser is not None else None
        first[name] = codes
    ledger = {name: {str(c): HZ._summ(L[name][c], L["policy"][c]) for c in lcps} for name in cands}
    data = {"spec": spec, "tag": tag, "family": fam, "index": idx, "protocol": proto, "horizon": horizon,
            "label": label, "k0": int(k0), "K": K, "keys": keys, "kind": FAMILIES[fam][0],
            "candidates": {n: (list(s) if s else None) for n, s in cands.items()}, "GQ": GQ, "ledger": ledger,
            "series": SR if series else None, "first_codes": first, "elapsed_s": time.perf_counter() - t0,
            "turns": K * len(cands) * horizon}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)
    return key, data


# ---------------------------------------------------------------------------------------------- design
def jobs(spec: str, tag: str, natural: Dict[str, list], families: Sequence[str] = tuple(FAMILIES),
         protocols: Optional[Sequence[str]] = None, k_nat: int = K_NATURAL, k_con: int = K_CONSTRUCTED,
         extended: Sequence[str] = (), root: str = BOUT, grids: Optional[Dict[str, list]] = None) -> list:
    """The jobs of one evaluated policy. Natural families: every state x protocol at the family horizon, the
    validation subset's "policy" jobs at VALIDATION_H; constructed: every cell x protocol, "policy" at VALIDATION_H.
    `extended` families (exception 3) instead run every state x protocol at EXTENDED_H."""
    cfg = env_cfg()
    grids = grids or GRIDS
    out = []
    for fam in families:
        kind = FAMILIES[fam][0]
        protos = [p for p in FAMILIES[fam][3] if protocols is None or p in protocols]
        h = family_horizon(fam)
        if kind == "constructed":
            for i, cell in enumerate(grids[fam]):
                st = constructed_state(cfg, fam, cell)
                for p in protos:
                    hh = EXTENDED_H if fam in extended else (VALIDATION_H if p == "policy" else h)
                    out.append((spec, tag, fam, i, st, p, hh, f"constructed {cell}", k_con, root))
            continue
        states = natural.get(fam, [])
        val = set(validation_indices(fam, len(states)))
        for i, st in enumerate(states):
            for p in protos:
                hh = EXTENDED_H if fam in extended else (VALIDATION_H if (i in val and p == "policy") else h)
                out.append((spec, tag, fam, i, st["start"], p, hh,
                            f"{kind} (game seed {st['seed']}, turn {st['turn']}, seat {'AB'[st['seat']]})", k_nat, root))
    return out


def job_turns(j) -> int:
    return int(j[8] * (1 + len(candidates(j[2]))) * j[6])


def volume(n_natural: int = N_NATURAL, n_early: int = N_EARLY, finals: int = 6, midpoints: int = 6,
           rate: float = 550.0, workers: int = 11) -> dict:
    """Simulated turns of the battery (no simulation: the job list with nominal state counts) and hours at `rate`
    turns per worker-second (the horizon study measured 530-583 for this engine)."""
    per_fam = {}
    for fam, (kind, *_rest) in FAMILIES.items():
        n = n_early if kind == "early" else n_natural
        fake = {fam: [{"start": None, "seed": 0, "turn": 0, "seat": 0}] * n} if kind != "constructed" else {}
        js = jobs("x", "x", fake, (fam,)) if kind != "constructed" else _constructed_jobs_nominal(fam)
        v = [j for j in js if j[6] == VALIDATION_H]
        per_fam[fam] = {"jobs": len(js), "turns": sum(job_turns(j) for j in js),
                        "validation_turns": sum(job_turns(j) for j in v), "horizon": family_horizon(fam),
                        "states": n if kind != "constructed" else len(GRIDS[fam])}
    per_final = sum(v["turns"] for v in per_fam.values())
    rep = 0
    for fam in REPLICATION["families"]:
        kind = FAMILIES[fam][0]
        if kind == "constructed":
            js = [j for j in _constructed_jobs_nominal(fam) if j[5] in REPLICATION["protocols"]]
        else:
            fake = {fam: [{"start": None, "seed": 0, "turn": 0, "seat": 0}] * REPLICATION["n_natural"]}
            js = jobs("x", "x", fake, (fam,), REPLICATION["protocols"])
        rep += sum(job_turns(j) for j in js)
    total = finals * per_final + midpoints * rep
    hours = total / (rate * workers) / 3600
    return {"per_family": per_fam, "per_final": per_final, "replication_per_midpoint": rep, "total_turns": total,
            "rate_turns_per_worker_s": rate, "workers": workers, "hours": hours,
            "hours_range": [total / (650 * workers) / 3600, total / (450 * workers) / 3600]}


def _constructed_jobs_nominal(fam: str) -> list:
    h = family_horizon(fam)
    return [("x", "x", fam, i, None, p, VALIDATION_H if p == "policy" else h, "", K_CONSTRUCTED, BOUT)
            for i, _ in enumerate(GRIDS[fam]) for p in FAMILIES[fam][3]]


# ---------------------------------------------------------------------------------------------- mechanics
MECH_KEYS = ("earned", "farm", "xp", "taken_bar", "dealt_bar", "minion_bar", "died", "landed", "ward_placed",
             "crash_hp", "front_hp")


def mechanics_job(args) -> tuple:
    """(spec, family, [(index, start)]): the EXACT one-turn expectation (m45b.exact, chance branches enumerated,
    ganks on) of every candidate's first action and of the policy's modal action, against the opponent policy's modal
    reply, at every state: the zero-variance "immediate mechanical consequence" layer of the battery."""
    spec, fam, states = args
    import torch
    torch.set_num_threads(1)
    from ....champions import champion_legal_actions
    from ...m45b import engine as E
    from ...m45b.exact import expect
    cfg = env_cfg()
    net = _net(spec)
    rows = []
    for idx, (lane, m4, k0) in states:
        own = HZ.CRNNet(net, k0, 0, 0, 1, greedy=True)
        b = HZ.CRNNet(net, k0, 0, 1, 1, greedy=True).decide(cfg, lane, m4, 1, 0)[0]
        legal = set(champion_legal_actions(lane, m4, 0, cfg))
        firsts = {"policy (modal)": own.decide(cfg, lane, m4, 0, 0)[0]}
        for name, cspec in candidates(fam).items():
            firsts[name] = int(E.decide(HZ.build_candidate(cspec, own), cfg, lane, m4, 0, 0))
        row = {"index": idx, "b_modal": int(b), "codes": firsts, "legal": {n: c in legal for n, c in firsts.items()},
               "mean": {}}
        for name, a in firsts.items():
            if a not in legal:
                continue
            e = expect(cfg, lane, m4, k0, int(a), int(b), ganks="on")["mean"]
            row["mean"][name] = {k: e[k] for k in MECH_KEYS if k in e}
        rows.append(row)
    agg = {}
    for name in ["policy (modal)", *candidates(fam)]:
        diffs = [{k: r["mean"][name][k][0] - r["mean"]["policy (modal)"][k][0] for k in MECH_KEYS}
                 for r in rows if name in r["mean"] and "policy (modal)" in r["mean"]]
        lead = [(r["mean"][name]["earned"][0] - r["mean"][name]["earned"][1])
                - (r["mean"]["policy (modal)"]["earned"][0] - r["mean"]["policy (modal)"]["earned"][1])
                for r in rows if name in r["mean"] and "policy (modal)" in r["mean"]]
        agg[name] = {"states": len(diffs), "same_code_as_policy": float(np.mean(
                         [r["codes"][name] == r["codes"]["policy (modal)"] for r in rows])) if rows else float("nan"),
                     "immediate_lead_diff": float(np.mean(lead)) if lead else float("nan"),
                     "own_diff": {k: float(np.mean([d[k] for d in diffs])) for k in MECH_KEYS} if diffs else {}}
    return (spec, fam), {"family": fam, "states": rows, "aggregate": agg}


def mechanics(spec: str, tag: str, starts: Dict[str, list], workers: int, root: str = BOUT) -> dict:
    """mechanics_job over every family's starts ({family: [(index, start)]}); <root>/<tag>/mechanics.json."""
    js = [(spec, f, v) for f, v in starts.items() if v]
    res = {k[1]: v for k, v in run_jobs(mechanics_job, js, workers, f"battery mechanics {tag}")}
    save_json("mechanics", {"meta": meta(stage="battery_mechanics", spec=spec, tag=tag), "families": res},
              os.path.join(root, tag))
    return res


# ---------------------------------------------------------------------------------------------- execution
def collect(spec: str, families: Sequence[str], n_natural: int, n_early: int, workers: int) -> Dict[str, list]:
    js = [(spec, f, n_early if FAMILIES[f][0] == "early" else n_natural) for f in families
          if FAMILIES[f][0] != "constructed"]
    res = run_jobs(collect_job, js, workers, f"battery states {spec}")
    return {k[1]: v["states"] for k, v in res}, {k[1]: v["games"] for k, v in res}


def run_policy(spec: str, tag: str, workers: int, families: Sequence[str] = tuple(FAMILIES),
               protocols: Optional[Sequence[str]] = None, n_natural: int = N_NATURAL, n_early: int = N_EARLY,
               k_nat: int = K_NATURAL, k_con: int = K_CONSTRUCTED, root: str = BOUT,
               grids: Optional[Dict[str, list]] = None, extended: Sequence[str] = ()) -> dict:
    """Collect the policy's natural states, then run (resumably) every job of the design."""
    t0 = time.perf_counter()
    nat, games = collect(spec, families, n_natural, n_early, workers)
    t_col = time.perf_counter() - t0
    js = jobs(spec, tag, nat, families, protocols, k_nat, k_con, extended, root, grids)
    starts: Dict[str, list] = {}
    for j in js:
        if j[3] not in [i for i, _ in starts.get(j[2], [])]:
            starts.setdefault(j[2], []).append((j[3], j[4]))
    t_m = time.perf_counter()
    if not extended:                                  # one-turn mechanics do not depend on the horizon: an exception-3
        mechanics(spec, tag, starts, workers, root)   # re-run keeps the original file (all families) untouched
    t_mech = time.perf_counter() - t_m
    t1 = time.perf_counter()
    res = dict(run_jobs(state_job, js, workers, f"battery {tag}"))
    man = {"meta": meta(stage="battery", spec=spec, tag=tag, families=list(families), k_natural=k_nat,
                        k_constructed=k_con, n_natural=n_natural, n_early=n_early, extended=list(extended)),
           "states": {f: {"n": len(v), "games": games[f], "validation": validation_indices(f, len(v)),
                          "states": [{"seed": s["seed"], "turn": s["turn"], "seat": s["seat"], "k": int(s["start"][2])}
                                     for s in v]} for f, v in nat.items()},
           "timing": {"collect_s": t_col, "mechanics_s": t_mech, "rollouts_s": time.perf_counter() - t1,
                      "workers": workers,
                      "turns": int(sum(v["turns"] for v in res.values())),
                      "job_seconds": float(sum(v["elapsed_s"] for v in res.values()))},
           "jobs": sorted(res)}
    mname = "manifest_extended" if extended else "manifest"
    save_json(mname, man, os.path.join(root, tag))
    if root == BOUT:
        save_json(f"{mname}_{tag}", man, BREPORT)
    return man


def run(confirm: bool, workers: int, phase: str = "ext") -> None:
    """The comprehensive battery: the full design on the six finals of the primary phase (the 16.384M extension,
    predesignated 2026-10-15), and the replication families (freeze, frozen-on, Recall, freeze grid; policy and modal)
    on the six 8.192M finals, the comparison group: independent runs of the same seeds at half the budget, each
    annealed to 0 (decided before any extension result; it replaces the planned replication on mid-schedule
    snapshots). Resumable: finished jobs are read back from their files."""
    if not confirm:
        raise SystemExit(f"the strategic battery is part of the comprehensive evaluation; it needs its own approval "
                         f"and then {CONFIRM_FLAG}. Nothing was started.")
    from .full import FULL_SEEDS
    for s in FULL_SEEDS:
        run_policy(f"final:{phase}:{s}", f"s{s}_{phase}", workers)
    if phase != "full":
        for s in FULL_SEEDS:
            run_policy(f"final:full:{s}", f"s{s}_full", workers, REPLICATION["families"], REPLICATION["protocols"],
                       n_natural=REPLICATION["n_natural"])


SMOKE_ROOT = os.path.join(OUT, "battery_smoke")


def smoke(spec: str, workers: int) -> dict:
    """Every family end to end at tiny sizes on one policy (2 natural states, K = 2, the first constructed cell,
    every protocol), with the validation subset forced to the first state; then the analysis on its output."""
    grids = {f: g[:1] for f, g in GRIDS.items()}
    global VAL_MIN
    old = VAL_MIN
    VAL_MIN = 1
    try:
        man = run_policy(spec, "smoke", workers, n_natural=2, n_early=2, k_nat=2, k_con=2, root=SMOKE_ROOT,
                         grids=grids)
    finally:
        VAL_MIN = old
    jobs_ = load("smoke", SMOKE_ROOT)
    ana = analyze(jobs_)
    report(["smoke"], SMOKE_ROOT)
    t = man["timing"]
    out = {"manifest": man, "analysis_families": sorted(ana["families"]), "validation": ana["validation"],
           "jobs": len(jobs_), "turns": t["turns"],
           "turns_per_job_second": t["turns"] / max(t["job_seconds"], 1e-9)}
    save_json("smoke_summary", out, SMOKE_ROOT)
    return out


# ---------------------------------------------------------------------------------------------- analysis
def load(tag: str, root: str = BOUT) -> List[dict]:
    import glob
    out = []
    for p in sorted(glob.glob(os.path.join(root, tag, "partial", "*.json.gz"))):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def units(jobs_: List[dict], fam: str, proto: str, cand: str, X: str, min_h: int = 0):
    """[(state, {cp: paired diffs (K,)})] for one comparison; states whose forced first code was never legal are
    excluded (the plan fell back to the policy)."""
    from .horizon_report import intended
    out = []
    for j in jobs_:
        if j["family"] != fam or j["protocol"] != proto or cand not in j["GQ"] or j["horizon"] < min_h:
            continue
        code = intended(j["candidates"][cand])
        if code is not None and not any(c == code for c in j["first_codes"][cand]):
            continue
        gc, gp = j["GQ"][cand][X], j["GQ"]["policy"][X]
        out.append((j["index"], {int(c): np.asarray(gc[c]) - np.asarray(gp[c]) for c in gc}))
    return out


def aggregate(us, h: int) -> Optional[dict]:
    """States are the units: mean of state means with a t interval; within-state (sigma_w^2) and between-state
    (sigma_b^2) variance components with the data's own K."""
    from .horizon_report import tcrit
    rows = [u[1][h] for u in us if h in u[1]]
    if len(rows) < 2:
        return None
    K = len(rows[0])
    m = np.array([r.mean() for r in rows])
    w = np.array([r.var(ddof=1) if len(r) > 1 else np.nan for r in rows])
    n = len(m)
    se = float(m.std(ddof=1) / math.sqrt(n))
    sw2 = float(np.nanmean(w))
    return {"n": n, "K": K, "mean": float(m.mean()), "se": se, "ci": tcrit(n) * se, "sigma_w2": sw2,
            "sigma_b2": max(float(m.var(ddof=1) - sw2 / K), 0.0)}


def validate(jobs_: List[dict], fam: str) -> dict:
    """Horizon study §6.3 on the family's validation units (protocol "policy", horizon 300): pooled bias of the
    primary statistic at 50 against its 300-turn value, the slope of X300 on X50 across (state, candidate), and
    contradictions among pairs significant at both horizons. Pass: |bias| <= max(0.15, 2 SE), slope in [0.8, 1.25], no
    contradiction."""
    X = primary_statistic(fam)
    pairs = []
    for j in jobs_:
        if j["family"] != fam or j["protocol"] != "policy" or j["horizon"] < VALIDATION_H:
            continue
        for cand in j["GQ"]:
            if cand == "policy":
                continue
            us = units([j], fam, "policy", cand, X)
            if us:
                d = us[0][1]
                pairs.append((j["index"], cand, d[DEFAULT_H], d[VALIDATION_H]))
    if len(pairs) < 3:
        return {"statistic": X, "pairs": len(pairs), "verdict": "insufficient"}
    m50 = np.array([p[2].mean() for p in pairs])
    m300 = np.array([p[3].mean() for p in pairs])
    diff = m50 - m300
    se = float(diff.std(ddof=1) / math.sqrt(len(diff)))
    slope = float(np.polyfit(m50, m300, 1)[0]) if np.ptp(m50) > 0 else float("nan")

    def sig(x):
        s = x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else np.inf
        return 0 if abs(x.mean()) <= 2 * s else int(np.sign(x.mean()))
    contra = [(i, c) for i, c, a, b in pairs if sig(a) and sig(b) and sig(a) != sig(b)]
    bias = float(diff.mean())
    ok_bias = abs(bias) <= max(0.15, 2 * se)
    ok_slope = bool(np.isfinite(slope) and 0.8 <= slope <= 1.25)
    return {"statistic": X, "pairs": len(pairs), "states": len({p[0] for p in pairs}), "bias": bias, "bias_se": se,
            "slope": slope, "contradictions": contra, "bias_ok": ok_bias, "slope_ok": ok_slope,
            "verdict": "pass" if ok_bias and ok_slope and not contra else "fail -> re-run the family at 150 turns"}


def analyze(jobs_: List[dict], hs: Sequence[int] = (1, 5, 10, 20, 50)) -> dict:
    """Per family, protocol and candidate: G_h and Q_h at the reporting horizons (paired differences against the
    policy's own choice), the primary statistic marked; and the validation verdict of every family (always on the
    300-turn validation units). A family re-run at EXTENDED_H after failing validation (exception 3) is tabulated
    from its EXTENDED_H jobs only (every state and protocol), with the EXTENDED_H column added."""
    fams = sorted({j["family"] for j in jobs_})
    out = {"families": {}, "validation": {}}
    for fam in fams:
        fj = [j for j in jobs_ if j["family"] == fam]
        extended = any(j["horizon"] == EXTENDED_H for j in fj)
        tj = [j for j in fj if j["horizon"] == EXTENDED_H] if extended else fj
        fhs = tuple(hs) + ((EXTENDED_H,) if extended else ())
        protos = sorted({j["protocol"] for j in tj})
        cands = [c for c in candidates(fam)]
        tab = {}
        for p in protos:
            for c in cands:
                row = {}
                for X in ("G", "Q"):
                    us = units(tj, fam, p, c, X)
                    row[X] = {h: aggregate(us, h) for h in fhs}
                tab.setdefault(p, {})[c] = row
        out["families"][fam] = {"primary": primary_statistic(fam), "extended": extended,
                                "horizon": EXTENDED_H if extended else family_horizon(fam),
                                "states": len({j["index"] for j in tj}), "table": tab}
        out["validation"][fam] = validate(jobs_, fam)
    return out


def extend(confirm: bool, workers: int, phase: str = "ext") -> dict:
    """Exception 3 of the horizon study: every (policy, family) whose validation fails is re-run at EXTENDED_H for all
    its states and protocols (the 300-turn validation units are kept). Decided by the pre-registered rule alone."""
    if not confirm:
        raise SystemExit(f"the battery extension is part of the battery; it needs {CONFIRM_FLAG}.")
    from .full import FULL_SEEDS
    plan = {}
    for s in FULL_SEEDS:
        sets = [(f"s{s}_{phase}", f"final:{phase}:{s}", tuple(FAMILIES), None, N_NATURAL)]
        if phase != "full":
            sets.append((f"s{s}_full", f"final:full:{s}", REPLICATION["families"], REPLICATION["protocols"],
                         REPLICATION["n_natural"]))
        for tag, spec, fams, protos, n_nat in sets:
            js = load(tag)
            have = {j["family"] for j in js}
            verdicts = {f: validate(js, f) for f in fams if f in have}
            failed = [f for f, v in verdicts.items() if v["verdict"].startswith("fail")]
            plan[tag] = {"failed": failed, "verdicts": {f: v["verdict"] for f, v in verdicts.items()}}
            print(f"[battery extend] {tag}: failed validation {failed or 'none'}", flush=True)
            if failed:
                run_policy(spec, tag, workers, failed, protos, n_natural=n_nat, extended=failed)
    save_json("extension_plan", {"meta": meta(stage="battery_extend", phase=phase), "plan": plan}, BREPORT)
    return plan


def pooled_over_policies(analyses: Dict[str, dict], fam: str, proto: str, cand: str, X: str, h: int) -> dict:
    """Training seeds as replicates: the family means of every evaluated policy, their mean and t interval."""
    from .horizon_report import tcrit
    v = [a["families"][fam]["table"][proto][cand][X][h]["mean"] for a in analyses.values()
         if fam in a["families"] and a["families"][fam]["table"].get(proto, {}).get(cand, {}).get(X, {}).get(h)]
    if len(v) < 2:
        return {"n": len(v), "mean": v[0] if v else float("nan"), "ci": float("nan"), "per_policy": v}
    v = np.asarray(v)
    return {"n": len(v), "mean": float(v.mean()), "ci": tcrit(len(v)) * float(v.std(ddof=1)) / math.sqrt(len(v)),
            "per_policy": v.tolist()}


def report(tags: Sequence[str], root: str = BOUT, name: str = "report", hs: Sequence[int] = (1, 5, 20, 50),
           out: Optional[str] = None) -> str:
    """<root>/<name>.md and summary.json: per family, protocol and candidate the primary statistic at the reporting
    horizons (one-turn candidates and sustained plans in separate tables), pooled over the evaluated policies (training
    seeds as replicates, t interval) with every policy's own value, the other statistic at 50 beside it, and each
    family's validation verdict per policy (horizon study §6.3)."""
    from .common import write_md
    analyses = {t: analyze(load(t, root), hs) for t in tags}
    lines = [f"# M4.5c strategic battery ({', '.join(tags)})\n",
             "Paired differences candidate - policy's own choice (resources). Primary statistic per family: Q "
             "(critic-bootstrapped) for natural families, G (observed discounted return) for constructed ones "
             "(exception 2); the other one at 50 turns is shown beside it. Seed mean +- t interval over the evaluated "
             "policies, then each policy.\n"]
    fams = [f for f in FAMILIES if any(f in a["families"] for a in analyses.values())]
    for fam in fams:
        X = primary_statistic(fam)
        Y = "G" if X == "Q" else "Q"
        ext = any(a["families"].get(fam, {}).get("extended") for a in analyses.values())
        fh = tuple(hs) + ((EXTENDED_H,) if ext else ())
        lines.append(f"## {fam} ({FAMILIES[fam][0]}; horizon {family_horizon(fam)}"
                     f"{f', re-run at {EXTENDED_H} after failed validation on ' + ', '.join(t for t, a in analyses.items() if a['families'].get(fam, {}).get('extended')) if ext else ''}"
                     f"; primary {X})\n")
        protos = sorted({p for a in analyses.values() for p in a["families"].get(fam, {}).get("table", {})})
        for kind, lab in (("1t:", "one-turn candidates"), ("plan:", "sustained plans")):
            cands = [c for c in candidates(fam) if c.startswith(kind)]
            if not cands:
                continue
            rows = []
            for p in protos:
                for c in cands:
                    cells = []
                    for h in fh:
                        r = pooled_over_policies(analyses, fam, p, c, X, h)
                        cells.append(f"{r['mean']:+.2f}" + (f" ± {r['ci']:.2f}" if r["ci"] == r["ci"] else ""))
                    ry = pooled_over_policies(analyses, fam, p, c, Y, DEFAULT_H)
                    r50 = pooled_over_policies(analyses, fam, p, c, X, DEFAULT_H)
                    rows.append([p, c, *cells, f"{ry['mean']:+.2f}", " ".join(f"{v:+.2f}" for v in r50["per_policy"])])
            hdr = ["protocol", "candidate", *[f"{X}{h}" for h in fh], f"{Y}{DEFAULT_H}", f"{X}{DEFAULT_H} per policy"]
            lines.append(f"{lab}:\n")
            lines.append("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr))
            lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
            lines.append("")
        v = {t: a["validation"].get(fam, {}) for t, a in analyses.items()}
        lines.append("Validation (§6.3): " + "; ".join(
            f"{t}: {x.get('verdict')} (bias {x.get('bias', float('nan')):+.2f} ± {2 * x.get('bias_se', float('nan')):.2f}, "
            f"slope {x.get('slope', float('nan')):.2f}, contradictions {len(x.get('contradictions', []))}, "
            f"pairs {x.get('pairs')})" for t, x in v.items()) + "\n")
    out = out or (BREPORT if root == BOUT else root)
    save_json(f"summary_{name}", {"tags": list(tags), "analyses": analyses}, out)
    text = "\n".join(lines)
    write_md(name, text, out)
    return text


__all__ = ["FAMILIES", "GRIDS", "REPLICATION", "candidates", "family_horizon", "primary_statistic", "checkpoints",
           "validation_indices", "collect_job", "state_job", "jobs", "volume", "run_policy", "run", "smoke", "load",
           "units", "aggregate", "validate", "analyze", "pooled_over_policies", "DEFAULT_H", "VALIDATION_H",
           "EXTENDED_H", "CONFIRM_FLAG", "report", "mechanics", "mechanics_job", "extend"]
