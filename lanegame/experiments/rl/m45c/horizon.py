"""Horizon sensitivity of local strategic counterfactuals (M4.5c methodological study; no training, no rule change).

Question: how long must a paired counterfactual rollout be to evaluate one laning decision? Every comparison below is
a PAIRED, common-random-number (CRN) comparison of a candidate intervention against the policy's own choice from the
same start state; every horizon is read from the SAME trajectories (checkpoints of one rollout), so differences across
horizons are paired too.

Rollouts (`rollout`): the M4.5b resolver (`champions_m4.resolve_turn_m4`, the environment's chance stream
`default_rng(seed)`, N_CHANCE uniforms per turn whatever the actions) and the M4.5b ledger, snapshotted at CHECKPOINTS.
At each checkpoint h:
    G_h        sum_{t<h} 0.99^t r_t, the focal side's (seat A's) truncated discounted gold lead (resources)
    Q_h        G_h + 0.99^h V(s_h): G_h bootstrapped with the pilot critic's value of A's observation at s_h (resources;
               V / reward_scale). Reported SEPARATELY from G_h, never substituted for it
    ledger     cumulative per-side ledger quantities (gold earned, farm lost out of reach = denial, XP, XP lost,
               damage by source, deaths, recalls, time away, wards, freeze turns, plates, trades ...)
    state      descriptors of s_h (HP, level, investment, absence, vision timers, freeze flags, crashed masses, wave
               position) - the "unresolved consequences" a short horizon leaves behind

Policies. The continuation of both seats is the M4.5c pilot final (`final:pilot:100`) through `CRNNet`: its sampling
uniform at turn index i is the i-th draw of default_rng([seed, 7, seat]) whatever happened before, so two branches
that differ in history still share their randomness turn by turn (sequential draws would desynchronize after a forced
action). Opponent protocols: "policy" (the pilot, sampling), "modal" (the pilot's most probable legal action), and
M4.5b scripted strategies in the constructed families.

Candidates. "policy" (the network's own choice) is the baseline of every paired difference. A ONE-TURN candidate
("1t:") forces one action at the start state and hands back to the policy; a SUSTAINED plan ("plan:") repeats an
action or a scripted rule for up to n turns while its condition holds, then hands back. They answer different
questions and are labelled separately.

Families (focal = seat A; natural states are collected from pilot self-play and mirrored to seat A; constructed states
are M4.5b builders, labelled "constructed"):
    A_wave      natural: fronts in contact                  1t LH / TN / SL / SH (+ Hold); plan TN x4, SH x4
    B_freeze    natural: the focal side holds a freeze      1t keep (TN+Hold) / guard (TN+Trade) / break (SH+Hold);
                                                            plan keep / guard (fz_guard) while held, <= 20
    B_frozen    natural: the focal side is frozen on        1t shove / walk-up farm (LH+Hold) / step back (LH+Safe) /
                                                            recall; plan shove x3
    C_trade     natural: enemy catchable without a dive and its front outweighs ours by >= 8 HP
                                                            1t trade (LH+Trade) / hold / safe
    D_ward      natural: Ward legal, vision down, own front at the lane cell or beyond
                                                            1t ward / LH+Hold
    E_recall    natural: Recall legal, HP <= 50%            1t recall; plan stay (SH+Hold x3) / step back (LH+Safe x3)
    F_freeze_c  constructed freeze grid (M4.5b freeze_state): power large / equal x held 12 / 20 x level 10 / 15
                                                            B_freeze candidates; opponents policy / modal / wk_shove
                                                            (walk up and Shove) / wk_back (step back) / wk_shove_reset
                                                            (reset below 60% HP and contest)
    G_trade_c   constructed trade bench: level 4 / 11 / 18 x enemy front 8 / 24 HP   C_trade candidates
    H_zone_c    constructed zoning, A x1.76 stronger: neutral / A's push / B's push   1t trade / hold / shove;
                                                            plan bully (nodive) x10; opponents policy / modal / g_wary
    I_ward_c    constructed Vision situations: neutral / forward / presiege / defending   D_ward candidates

Sizes (pre-registered): 40 natural states per family, the first 12 of them the LONG subset; K = 24 paired rollouts per
(state, protocol, candidate); horizon 50 (checkpoints 1-50), 300 for the long subset and every constructed cell
(checkpoints up to 300). Seeds: state collection 92,600+, rollouts 92,700+ (pilot range; 93,000-94,999 untouched).
Jobs are resumable: each (family, state, protocol) result is written to results/m45c/horizon/partial/ as it finishes.
"""

from __future__ import annotations

import json
import os
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ....champions import N_CHANCE, champion_legal_actions
from ....config import (B_HOLD, B_RECALL, B_SAFE, B_TRADE, CHAMP_ABSENT, CHAMP_WARD, W_HOLD, W_SHOVE, W_SLOWPUSH,
                        W_THIN, champ_act)
from .common import OUT, PILOT_SEED, env_cfg, meta, run_jobs, save_json

HOUT = os.path.join(OUT, "horizon")
PARTIAL = os.path.join(HOUT, "partial")
GAMMA = 0.99
SHORT_CP = (1, 2, 3, 5, 10, 15, 20, 30, 40, 50)
LONG_CP = SHORT_CP + (75, 100, 150, 200, 300)
LEDGER_CP = (1, 5, 10, 20, 30, 50, 100, 150, 300)
SERIES_T = 50                                   # per-turn series (freeze families) for t = 1..50
N_NATURAL, N_LONG, K = 40, 12, 24
STATE_SEED, ROLL_SEED = 92_600, 92_700
SPEC = f"final:pilot:{PILOT_SEED}"
LEDGER_KEYS = ("earned", "farm", "unc", "xp", "xp_lost", "d_total", "d_trade", "d_min", "d_tower", "d_gank", "deaths",
               "recalls", "rec_intr", "away", "wards", "ward_placed", "ward_intr", "ganks", "held", "frozen",
               "crash_hp", "plates", "trade_conn", "trade_dealt",
               # M4.6b measurement fix: the channels of an economic outcome (accounting, not causal attribution):
               # earned = farm + bounty + plate; farm = farm_avail - unc; unc = (unc - unc_away) present + unc_away
               "farm_avail", "unc_away", "bounty", "plate", "trade_att")
STATE_KEYS = ("hpA", "hpB", "lvlA", "lvlB", "invA", "invB", "absA", "absB", "visA", "visB", "heldA", "frozenA",
              "siegeA", "siegeB", "wavepos")
SERIES_KEYS = ("heldA", "unc_B", "xp_lost_B", "hpA", "hpB", "deaths_A", "deaths_B", "lead")

LH, TN, SL, SH = W_HOLD, W_THIN, W_SLOWPUSH, W_SHOVE
RECALL = champ_act(W_HOLD, B_RECALL)


# ---------------------------------------------------------------------------------------------- policies
class CRNNet:
    """A network policy whose sampling uniform at turn index i is fixed by (seed, seat, i) (module docstring);
    `greedy` plays the most probable legal action (the "modal" protocol)."""

    def __init__(self, net, k0: int, seed: int, seat: int, horizon: int, greedy: bool = False, name: str = "net"):
        self.net, self.k0, self.greedy, self.name = net, int(k0), bool(greedy), name
        self.u = np.random.default_rng([int(seed), 7, int(seat)]).random(int(horizon) + 1)

    def probs(self, cfg, lane, m4, s: int, i: int):
        import torch
        from ....diagnostics.branch import Snapshot, observation
        obs = observation(cfg, Snapshot(lane, m4, self.k0 + i, (cfg.skill_a, cfg.skill_b)), s)
        with torch.no_grad():
            logits, _ = self.net(torch.as_tensor(obs.vector, dtype=torch.float32).unsqueeze(0))
        lg = logits[0].double().numpy()[list(obs.legal)]
        p = np.exp(lg - lg.max())
        return obs.legal, p / p.sum()

    def decide(self, cfg, lane, m4, s: int, i: int):
        legal, p = self.probs(cfg, lane, m4, s, i)
        if len(legal) == 1:
            return legal[0], "only"
        if self.greedy:
            return legal[int(np.argmax(p))], "modal"
        j = min(int(np.searchsorted(np.cumsum(p), self.u[i], side="right")), len(p) - 1)
        return legal[j], "sample"


def critic_value(net, cfg, lane, m4, k: int, s: int = 0) -> float:
    """The pilot critic's value of seat s's observation, in resources (V / reward_scale). Opt-in (M5): a
    history-dependent policy object with `value_at(cfg, lane, m4, k, s)` supplies its own value (its input includes
    what it has observed); a network is evaluated as before."""
    if hasattr(net, "value_at"):
        return float(net.value_at(cfg, lane, m4, k, s))
    import torch
    from ....diagnostics.branch import Snapshot, observation
    obs = observation(cfg, Snapshot(lane, m4, k, (cfg.skill_a, cfg.skill_b)), s)
    with torch.no_grad():
        v = float(net.value(torch.as_tensor(obs.vector, dtype=torch.float32).unsqueeze(0))[0])
    return v / 0.1


# ---------------------------------------------------------------------------------------------- one rollout
def _state_desc(cfg, lane, m4, tb) -> Dict[str, float]:
    from ..m44.records import holds_freeze, wave_position
    from ...m45b.engine import hp_frac
    sc = cfg.gold_scale
    out = {}
    for s, key in ((0, "A"), (1, "B")):
        me = m4.side(s)
        out[f"hp{key}"] = hp_frac(m4, s, tb)
        out[f"lvl{key}"] = float(tb.level_by_xp[min(me.xp, tb.xp_cap)])
        out[f"inv{key}"] = me.items / sc if cfg.continuous_items else float(me.items)
        out[f"abs{key}"] = float(me.absent > 0)
        out[f"vis{key}"] = float(lane.vision_of(s))
        out[f"siege{key}"] = float(lane.siege_of(s))
    out["heldA"] = float(holds_freeze(lane, 0))
    out["frozenA"] = float(holds_freeze(lane, 1))
    wp = wave_position(lane, 0)
    out["wavepos"] = float(wp) if np.isfinite(wp) else float("nan")
    return out


def rollout(cfg, start, pa, pb, seed: int, horizon: int, checkpoints: Sequence[int], critic=None,
            series: bool = False) -> dict:
    """One game of `horizon` turns from `start` (module docstring); returns {cp: {G, Q, V, ledger..., state...}} and,
    with `series`, per-turn series for t = 1..SERIES_T."""
    from ....champions_m4 import context, resolve_turn_m4
    from ..m44.records import holds_freeze
    from ...m45b import engine as E
    lane, m4, k0 = start
    ctx = context(cfg)
    tb = ctx.t
    ls = E.lane_cache(cfg)
    rng = np.random.default_rng(int(seed))
    led = E.Ledger(cfg, start)
    cps = set(c for c in checkpoints if c <= horizon)
    out: Dict[int, dict] = {}
    ser = {k: [] for k in SERIES_KEYS} if series else None
    G = 0.0
    for i in range(horizon):
        u = rng.random(N_CHANCE)
        a = E.decide(pa, cfg, lane, m4, 0, i)
        b = E.decide(pb, cfg, lane, m4, 1, i)
        t = resolve_turn_m4(lane, m4, k0 + i, a, b, cfg, lane_step=ls, u=u, ctx=ctx)
        E.observe_turn(pa, cfg, lane, m4, k0 + i, (a, b))
        E.observe_turn(pb, cfg, lane, m4, k0 + i, (a, b))
        led.add(t, lane)
        G += GAMMA ** i * t.reward
        lane, m4 = t.lane.state, t.champ
        n = i + 1
        if ser is not None and n <= SERIES_T:
            v = led.v
            ser["heldA"].append(float(holds_freeze(lane, 0)))
            ser["unc_B"].append(v["unc"][1])
            ser["xp_lost_B"].append(v["xp_lost"][1])
            ser["hpA"].append(E.hp_frac(m4, 0, tb))
            ser["hpB"].append(E.hp_frac(m4, 1, tb))
            ser["deaths_A"].append(v["deaths"][0])
            ser["deaths_B"].append(v["deaths"][1])
            ser["lead"].append(v["earned"][0] - v["earned"][1])
        if n in cps:
            row = {"G": G}
            if critic is not None:
                V = critic_value(critic, cfg, lane, m4, k0 + n, 0)
                row["V"], row["Q"] = V, G + GAMMA ** n * V
            if n in LEDGER_CP:
                for kk in LEDGER_KEYS:
                    row[f"{kk}_A"], row[f"{kk}_B"] = led.v[kk][0], led.v[kk][1]
                row.update(_state_desc(cfg, lane, m4, tb))
            out[n] = row
    res = {"cp": out}
    if ser is not None:
        res["series"] = ser
    return res


# ---------------------------------------------------------------------------------------------- candidates
def _fixed(code):
    return ("fixed", code, 1, None)


def _plan(code, n, cond=None):
    return ("fixed", code, n, cond)


def _rule(name, n, cond=None):
    return ("rule", name, n, cond)


def _beh(rule, n, cond=None):
    """M4.6b measurement fix: a BEHAVIOUR-ONLY candidate (m46a.policies.BehaviorPlan): the behaviour is forced, the wave
    and objective stay the policy's own conditional choice. One-turn candidates always force (no release); longer
    plans hand the turn to the policy at or below 40% HP (as M4.6a)."""
    return ("beh", rule, n, cond)


def cond_held(cfg, lane, m4, s):
    from ..m44.records import holds_freeze
    return holds_freeze(lane, s)


def cond_frozen(cfg, lane, m4, s):
    from ..m44.records import holds_freeze
    return holds_freeze(lane, 1 - s)


CONDS = {None: None, "held": cond_held, "frozen": cond_frozen}

FREEZE_CANDS = {"1t:keep": _fixed(champ_act(TN, B_HOLD)), "1t:guard": _fixed(champ_act(TN, B_TRADE)),
                "1t:break": _fixed(champ_act(SH, B_HOLD)),
                "plan:keep": _plan(champ_act(TN, B_HOLD), 20, "held"), "plan:guard": _rule("fz_guard", 20, "held")}
FAMILY_CANDS: Dict[str, Dict[str, tuple]] = {
    "A_wave": {"1t:LH": _fixed(champ_act(LH, B_HOLD)), "1t:TN": _fixed(champ_act(TN, B_HOLD)),
               "1t:SL": _fixed(champ_act(SL, B_HOLD)), "1t:SH": _fixed(champ_act(SH, B_HOLD)),
               "plan:TNx4": _plan(champ_act(TN, B_HOLD), 4), "plan:SHx4": _plan(champ_act(SH, B_HOLD), 4)},
    "B_freeze": FREEZE_CANDS,
    "B_frozen": {"1t:shove": _fixed(champ_act(SH, B_HOLD)), "1t:farm": _fixed(champ_act(LH, B_HOLD)),
                 "1t:step_back": _fixed(champ_act(LH, B_SAFE)), "1t:recall": _fixed(RECALL),
                 "plan:shovex3": _plan(champ_act(SH, B_HOLD), 3),
                 # M4.6b: behaviour-only twins of the two Last-Hit-wave references above
                 "1t:walk_up_beh": _beh("hold", 1), "1t:step_back_beh": _beh("safe", 1)},
    "C_trade": {"1t:trade": _fixed(champ_act(LH, B_TRADE)), "1t:hold": _fixed(champ_act(LH, B_HOLD)),
                "1t:safe": _fixed(champ_act(LH, B_SAFE)),
                "1t:trade_beh": _beh("trade", 1), "1t:hold_beh": _beh("hold", 1), "1t:safe_beh": _beh("safe", 1)},
    "D_ward": {"1t:ward": _fixed(CHAMP_WARD), "1t:farm": _fixed(champ_act(LH, B_HOLD))},
    "E_recall": {"1t:recall": _fixed(RECALL), "plan:stayx3": _plan(champ_act(SH, B_HOLD), 3),
                 "plan:step_backx3": _plan(champ_act(LH, B_SAFE), 3)},
    "F_freeze_c": FREEZE_CANDS,
    "G_trade_c": {"1t:trade": _fixed(champ_act(LH, B_TRADE)), "1t:hold": _fixed(champ_act(LH, B_HOLD)),
                  "1t:safe": _fixed(champ_act(LH, B_SAFE)),
                  "1t:trade_beh": _beh("trade", 1), "1t:hold_beh": _beh("hold", 1), "1t:safe_beh": _beh("safe", 1)},
    "H_zone_c": {"1t:trade": _fixed(champ_act(LH, B_TRADE)), "1t:hold": _fixed(champ_act(LH, B_HOLD)),
                 "1t:shove": _fixed(champ_act(SH, B_HOLD)),
                 "plan:bullyx10": _rule("g_bully", 10),          # scripted reference: forces the Last Hit wave too
                 "1t:trade_beh": _beh("trade", 1), "1t:hold_beh": _beh("hold", 1),
                 "plan:bully_behx10": _beh("bully", 10), "plan:threaten_behx10": _beh("threaten", 10),
                 "plan:farm_behx10": _beh("farm", 10)},
    "I_ward_c": {"1t:ward": _fixed(CHAMP_WARD), "1t:farm": _fixed(champ_act(LH, B_HOLD))},
}


def build_candidate(spec: tuple, own):
    from ...m45b import engine as E
    from ...m45b.strategies import get
    kind, x, n, cond = spec
    if kind == "beh":                                   # M4.6b: behaviour only, the wave left to the policy
        from ..m46a.policies import RULES, BehaviorPlan
        plan = BehaviorPlan(str(x), own, RULES[x], int(n), 0.4 if int(n) > 1 else 0.0)
        return plan if cond is None else E.Then(f"beh:{x}", plan, own, int(n), CONDS[cond])
    first = E.Fixed(str(x), int(x), own) if kind == "fixed" else get(x)
    return E.Then(f"{kind}:{x}", first, own, int(n), CONDS[cond])


# ---------------------------------------------------------------------------------------------- states
def predicates() -> Dict[str, Callable]:
    from . import strategic as SG
    return {"A_wave": SG.p_wave, "B_freeze": SG.p_freeze, "B_frozen": SG.p_frozen, "C_trade": SG.p_trade,
            "D_ward": SG.p_vision, "E_recall": SG.p_recall}


NATURAL = ("A_wave", "B_freeze", "B_frozen", "C_trade", "D_ward", "E_recall")
CONSTRUCTED = ("F_freeze_c", "G_trade_c", "H_zone_c", "I_ward_c")
F_GRID = [(gap, held, lvl) for gap in ("large", "equal") for held in (12, 20) for lvl in (10, 15)]
G_GRID = [(lvl, w) for lvl in (4, 11, 18) for w in (8, 24)]
H_GRID = ["neutral", "a_push", "b_push"]
I_GRID = ["neutral", "forward", "presiege", "defending"]
PROTOCOLS = {"natural": ("policy", "modal"), "F_freeze_c": ("policy", "modal", "wk_shove", "wk_back", "wk_shove_reset"),
             "G_trade_c": ("policy", "modal"), "H_zone_c": ("policy", "modal", "g_wary"),
             "I_ward_c": ("policy", "modal")}


def constructed_state(cfg, fam: str, cell):
    from ...m45b import states as S
    if fam == "F_freeze_c":
        gap, held, lvl = cell
        return S.freeze_state(cfg, level=lvl, gap=gap, held=held)
    if fam == "G_trade_c":
        lvl, w = cell
        return S.trade_bench(cfg, lvl, w, "cover")
    if fam == "H_zone_c":
        return S.zoning_state(cfg, cell, level=10, gap="large")
    if fam == "I_ward_c":
        return S.vision_state(cfg, cell, level=10)
    raise ValueError(fam)


def collect_job(args) -> tuple:
    """(family, n): natural states of a family from pilot self-play (both seats the pilot, sampling)."""
    fam, n = args
    import torch
    torch.set_num_threads(1)
    from ...m45b import engine as E
    from ...m45b import states as S
    from ...m45b.reference import NetPolicy
    from .common import agent_from_spec
    cfg = env_cfg()
    agent = agent_from_spec(SPEC, cfg)
    pred = predicates()[fam]
    off = NATURAL.index(fam) * 1000
    out, games = [], 0
    for g in range(400):
        sd = STATE_SEED * 10 + off + g
        out += E.collect(cfg, S.s0(cfg), NetPolicy(agent, 0, sd, 0), NetPolicy(agent, 0, sd, 1), [sd], 300, pred,
                         per_game=1, min_turn=20, rng_seed=sd)
        games += 1
        if len(out) >= n:
            break
    return fam, {"states": out[:n], "games": games}


# ---------------------------------------------------------------------------------------------- jobs
def _opponent(proto: str, net, k0: int, seed: int, horizon: int):
    from ...m45b.strategies import get
    if proto == "policy":
        return CRNNet(net, k0, seed, 1, horizon)
    if proto == "modal":
        return CRNNet(net, k0, seed, 1, horizon, greedy=True)
    return get(proto)


def _summ(rows_c: np.ndarray, rows_p: np.ndarray) -> dict:
    """Candidate mean, and mean / SD over rollouts of the paired difference candidate - policy (columns = keys)."""
    d = rows_c - rows_p
    return {"mean": np.nanmean(rows_c, 0).tolist(), "diff": np.nanmean(d, 0).tolist(),
            "diff_sd": np.nanstd(d, 0, ddof=1).tolist()}


def state_job(args) -> tuple:
    """(family, index, start, protocol, horizon, label): every candidate x K rollouts from one start. Stored per
    candidate: G_h and Q_h of every rollout (every checkpoint); for the ledger / state keys at LEDGER_CP the candidate
    mean and the paired difference to "policy" (mean and SD over the K paired rollouts); per-turn series (freeze
    families, means over rollouts); the first code played at the start state (to check that a forced code was legal)."""
    import gzip
    fam, idx, start, proto, horizon, label = args
    key = f"{fam}|{idx}|{proto}"
    path = os.path.join(PARTIAL, key.replace("|", "__") + ".json.gz")
    if os.path.exists(path):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return key, json.load(f)
    import torch
    torch.set_num_threads(1)
    from ...m45b import engine as E
    from .common import agent_from_spec
    cfg = env_cfg()
    net = agent_from_spec(SPEC, cfg).net
    k0 = start[2]
    cps = LONG_CP if horizon > 50 else SHORT_CP
    lcps = [c for c in LEDGER_CP if c <= horizon]
    keys = [f"{k}_{x}" for k in LEDGER_KEYS for x in ("A", "B")] + list(STATE_KEYS) + ["V"]
    cands = {"policy": None, **FAMILY_CANDS[fam]}
    series = fam in ("B_freeze", "F_freeze_c")
    t0 = time.perf_counter()
    GQ: Dict[str, dict] = {}
    L: Dict[str, Dict[int, np.ndarray]] = {}
    S: Dict[str, dict] = {}
    first_codes: Dict[str, List[int]] = {}
    for name, spec in cands.items():
        G = {c: [] for c in cps}
        Q = {c: [] for c in cps}
        led = {c: np.full((K, len(keys)), np.nan) for c in lcps}
        ser = {kk: np.zeros(SERIES_T) for kk in SERIES_KEYS} if series else None
        codes = []
        for r in range(K):
            sd = ROLL_SEED * 100 + r
            own = CRNNet(net, k0, sd, 0, horizon)
            pa = own if spec is None else build_candidate(spec, own)
            pb = _opponent(proto, net, k0, sd, horizon)
            codes.append(int(E.decide(pa, cfg, start[0], start[1], 0, 0)))
            out = rollout(cfg, start, pa, pb, sd, horizon, cps, critic=net, series=series)
            for c, row in out["cp"].items():
                G[c].append(row["G"])
                Q[c].append(row["Q"])
                if c in led:
                    led[c][r] = [row.get(kk, np.nan) for kk in keys]
            if ser is not None:
                for kk in SERIES_KEYS:
                    ser[kk] += np.asarray(out["series"][kk]) / K
        GQ[name] = {"G": {str(c): v for c, v in G.items()}, "Q": {str(c): v for c, v in Q.items()}}
        L[name] = led
        S[name] = {kk: v.tolist() for kk, v in ser.items()} if ser is not None else None
        first_codes[name] = codes
    ledger = {name: {str(c): _summ(L[name][c], L["policy"][c]) for c in lcps} for name in cands}
    data = {"family": fam, "index": idx, "protocol": proto, "horizon": horizon, "label": label, "k0": int(k0),
            "K": K, "keys": keys, "candidates": {n: (list(s) if s else None) for n, s in cands.items()},
            "GQ": GQ, "ledger": ledger, "series": S if series else None, "first_codes": first_codes,
            "elapsed_s": time.perf_counter() - t0, "turns": K * len(cands) * horizon}
    os.makedirs(PARTIAL, exist_ok=True)
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)
    return key, data


def jobs(natural_states: Dict[str, list]) -> list:
    cfg = env_cfg()
    out = []
    for fam in NATURAL:
        for i, st in enumerate(natural_states.get(fam, [])):
            for proto in PROTOCOLS["natural"]:
                long = i < N_LONG and proto == "policy"
                out.append((fam, i, st["start"], proto, 300 if long else 50,
                            f"natural (game seed {st['seed']}, turn {st['turn']}, seat {'AB'[st['seat']]})"))
    for fam, grid in (("F_freeze_c", F_GRID), ("G_trade_c", G_GRID), ("H_zone_c", H_GRID), ("I_ward_c", I_GRID)):
        for i, cell in enumerate(grid):
            st = constructed_state(cfg, fam, cell)
            for proto in PROTOCOLS[fam]:
                out.append((fam, i, st, proto, 300, f"constructed {cell}"))
    return out


def run(workers: int = 11) -> dict:
    """Collect natural states, then run every (family, state, protocol) job (resumable). Writes
    results/m45c/horizon/states.json (state descriptions) and raw.json.gz (every job)."""
    t0 = time.perf_counter()
    col = dict(run_jobs(collect_job, [(f, N_NATURAL) for f in NATURAL], workers, "horizon states"))
    t_col = time.perf_counter() - t0
    nat = {f: v["states"] for f, v in col.items()}
    js = jobs(nat)
    t1 = time.perf_counter()
    res = dict(run_jobs(state_job, js, workers, "horizon rollouts"))
    t_roll = time.perf_counter() - t1
    desc = {f: {"games": col[f]["games"], "n": len(nat[f]),
                "states": [{"seed": s["seed"], "turn": s["turn"], "seat": s["seat"], "k": int(s["start"][2])}
                           for s in nat[f]]} for f in NATURAL}
    data = {"meta": meta(stage="horizon", spec=SPEC, K=K, n_natural=N_NATURAL, n_long=N_LONG,
                         checkpoints_short=list(SHORT_CP), checkpoints_long=list(LONG_CP), ledger_cp=list(LEDGER_CP),
                         state_seed=STATE_SEED, roll_seed=ROLL_SEED, workers=workers),
            "states": desc, "timing": {"collect_s": t_col, "rollouts_s": t_roll, "workers": workers,
                                       "turns": int(sum(v["turns"] for v in res.values())),
                                       "job_seconds": float(sum(v["elapsed_s"] for v in res.values()))},
            "jobs": sorted(res), "raw": "per-job files in results/m45c/horizon/partial/ (gzipped JSON)"}
    save_json("manifest", data, HOUT)
    return data


__all__ = ["CRNNet", "critic_value", "rollout", "FAMILY_CANDS", "NATURAL", "CONSTRUCTED", "PROTOCOLS", "run", "jobs",
           "state_job", "collect_job", "build_candidate", "SHORT_CP", "LONG_CP", "LEDGER_CP", "K", "N_NATURAL",
           "N_LONG"]
