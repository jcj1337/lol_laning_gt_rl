"""Games from s0 for M5b (learning curves, history, head to head) and M5d (opponent families), design §6-§8.

A game of a cell starts from s0 with chance stream default_rng(sd) and is played twice, the focal policy in seat A and
in seat B (paired seats); every focal policy of a cell meets the opponent on the same games and streams (common random
numbers), so focal - focal differences are paired. Seeds: sd = root * 10^7 + cell * 10^4 + game.

Seat specifications:
    ("net", <M5 spec>[, mode])   a network: 90 inputs -> m45c.horizon.CRNNet, 103 / 116 -> players.OMPolicy (belief
                                 mode "live" | "prior" | ("table", T)); sampling uniform u[i] of default_rng([sd, 7, seat])
    ("opp", <name>)              a fixed opponent through the agent interface, generator default_rng([sd, 11, seat]):
                                 every OM2 name (`om2.common.agent`: archetypes, held-out scripted variants, the library's
                                 learned types, held-out learned instances, mixtures, switchers, "base:<s>"), plus
                                   its:<X>:<t>:<o>     imitate-then-switch: scripted archetype X while the turn index k < t,
                                                       then the competent policy ext/baseline:o
                                   style:<name>:<o>    an M4.4c conditional style (respect, passive, stepback, ignore,
                                                       shover, breaker) on the competent base ext/baseline:o
                                   pol:<M5 spec>       a 90-input M5 network as an agent (sampling with the seat generator)
With only networks and OM2 opponents this loop plays exactly the games of `om2.evaluate.play` (tested). The engine
runs `m45_calibrated()`; om-arm networks build their obs_last_action input themselves (players module docstring).

Recorded per game: r [T] (seat A's reward per turn), the om seats' belief log posteriors at CHECK_TURNS, and the
lane-exit panel at turn LANE_END (`diagnostics.lane_exit.describe`: levels and XP, HP, absence, turns since return,
gold, wave stocks; seat A's orientation).
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import common as C
from . import players as PL

T_EVAL = 150                    # games of the M5d families (the lead at READ_TURN is the primary value)
T_LONG = 500                    # the objective's horizon readout: the discounted lead over 500 turns (gamma 0.99)
READ_TURN = 114                 # the lane-exit readout: the lead at turn 114 = E_20 at turn 94 in expectation
LANE_END = 94
GAMMA = 0.99
CHECK_TURNS = (1, 3, 5, 10, 20, 40, 60, 94)
PANEL_KEYS = ("D0", "level_diff", "xp_diff", "hp_diff", "stock_diff", "status_A", "status_B", "returns_in_A",
              "returns_in_B", "since_return_A", "since_return_B", "unspent_A", "unspent_B", "invested_A", "invested_B")


# ---------------------------------------------------------------------------------------------- opponents
_AGENTS: Dict[str, object] = {}


def opponent_agent(name: str):
    """A fixed opponent by name (module docstring); cached per process (every agent here is memoryless or reads only
    the observation's turn index, so one instance serves every game)."""
    if name in _AGENTS:
        return _AGENTS[name]
    from ....opponent import NetAgent, ScriptedAgent, SwitchAgent
    from ...m45b.strategies import get
    from ..om2 import common as O2
    if name.startswith("its:"):
        _, x, t, o = name.split(":")
        a = SwitchAgent(ScriptedAgent(get(x), x), NetAgent(O2.net90(O2.start_path(int(o))), f"base:{o}"), int(t), name)
    elif name.startswith("style:"):
        from ..m44c.opponents import style
        _, st, o = name.split(":")
        a = style(NetAgent(O2.net90(O2.start_path(int(o))), f"base:{o}"), st)
        a.name = name
    elif name.startswith("pol:"):
        net = PL.net_of(name[4:])
        if net.obs_size != C.OBS_SP:
            raise ValueError(f"{name}: only 90-input networks play as agents; use a ('net', spec) seat")
        a = NetAgent(net, name)
    else:
        a = O2.agent(name)
    _AGENTS[name] = a
    return a


# ---------------------------------------------------------------------------------------------- one game
def _seat(spec: tuple, sd: int, seat: int, T: int):
    if spec[0] == "net":
        mode = spec[2] if len(spec) > 2 else "live"
        return ("net", PL.crn_policy(spec[1], 0, sd, seat, T, mode=mode))
    if spec[0] == "opp":
        return ("opp", opponent_agent(spec[1]), np.random.default_rng([int(sd), 11, int(seat)]))
    raise ValueError(spec)


def play(a_spec: tuple, b_spec: tuple, sd: int, T: int, panel: bool = True, lane_step=None, record: bool = False) -> dict:
    """One game from s0 (module docstring): {"r": [T], "trace": {seat: [len(CHECK_TURNS), K]}, "panel": {...}}; with
    `record`, also "codes" [T, 2] (both seats' actions) and "lp" {om seat: [T, K]} (its belief after every turn)."""
    from ....champions import N_CHANCE
    from ....champions_m4 import context, resolve_turn_m4
    from ....diagnostics import lane_exit as LX
    from ...m45b import engine as E
    from ...m45b import states as S
    cfg = C.arm_cfg("sp")
    cfg_om = C.arm_cfg("om")
    ls = lane_step or E.lane_cache(cfg)
    lane, m4, k0 = S.s0(cfg)
    ctx = context(cfg)
    seats = [_seat(a_spec, sd, 0, T), _seat(b_spec, sd, 1, T)]
    om = {s: p[1] for s, p in enumerate(seats) if p[0] == "net" and isinstance(p[1], PL.OMPolicy)
          and p[1].has_belief}
    need_obs = any(p[0] == "opp" for p in seats)
    rng = np.random.default_rng(int(sd))
    last = (-1, -1)
    r = np.zeros(T, dtype=np.float32)
    trace = {s: np.zeros((len(CHECK_TURNS), C.K), dtype=np.float32) for s in om}
    checks = {t: j for j, t in enumerate(CHECK_TURNS)}
    absent = [[], []]
    pan = None
    codes_all = np.zeros((T, 2), dtype=np.int16) if record else None
    lp_all = {s: np.zeros((T, C.K), dtype=np.float32) for s in om} if record else None
    for i in range(T):
        u = rng.random(N_CHANCE)
        if panel:
            for s in (0, 1):
                absent[s].append(int(m4.side(s).absent))
            if i == LANE_END:
                hist = {f"since_return_{x}": LX.since_return(absent[s], i) for s, x in ((0, "A"), (1, "B"))}
                d = LX.describe(cfg, (lane, m4, k0 + i), hist)
                pan = {k: float(d[k]) for k in PANEL_KEYS}
        obs = None
        if need_obs:
            obs = (PL.observation_of(cfg_om, lane, m4, k0 + i, last, 0), PL.observation_of(cfg_om, lane, m4, k0 + i, last, 1))
        codes = []
        for s, p in enumerate(seats):
            if p[0] == "net":
                codes.append(E.decide(p[1], cfg, lane, m4, s, i))
            else:
                codes.append(int(p[1].act(obs[s], p[2])))
        t = resolve_turn_m4(lane, m4, k0 + i, codes[0], codes[1], cfg, lane_step=ls, u=u, ctx=ctx)
        for s, p in enumerate(seats):
            if p[0] == "net":
                E.observe_turn(p[1], cfg, lane, m4, k0 + i, (codes[0], codes[1]))
        if record:
            codes_all[i] = codes
            for s, pol in om.items():
                lp_all[s][i] = pol.belief.lp
        if (i + 1) in checks:
            for s, pol in om.items():
                trace[s][checks[i + 1]] = pol.belief.lp
        r[i] = t.reward
        lane, m4 = t.lane.state, t.champ
        last = (codes[0], codes[1])
    out = {"r": r, "trace": trace, "panel": pan}
    if record:
        out.update({"codes": codes_all, "lp": lp_all})
    return out


# ---------------------------------------------------------------------------------------------- cells
def game_seed(root: int, cell: int, m: int) -> int:
    return int(root) * 10 ** 7 + int(cell) * 10 ** 4 + int(m)


def cell_job(j: dict) -> tuple:
    """`j`: key, path, focals (list of seat specs), opponent (seat spec), N, T, root, cell. Every focal plays the
    opponent on the same N paired-seat games; stores R [N, 2, T] (the focal's reward, orientation 0 = focal in seat A),
    the focal's belief traces, the opponent's belief traces when the opponent has a belief (f<i>__opp_trace: an om
    target's belief about the focal) and the lane-exit panels (focal orientation). Resumable: an existing file is
    returned."""
    path = j["path"]
    if os.path.exists(path):
        return j["key"], {"path": path, "cached": True}
    C.setup("sp")
    N, T = int(j["N"]), int(j["T"])
    out: Dict[str, np.ndarray] = {}
    t0 = time.perf_counter()
    opp = tuple(j["opponent"])
    for f, focal in enumerate(j["focals"]):
        focal = tuple(focal)
        R = np.zeros((N, 2, T), dtype=np.float32)
        tr = np.zeros((N, 2, len(CHECK_TURNS), C.K), dtype=np.float32)
        otr = np.zeros((N, 2, len(CHECK_TURNS), C.K), dtype=np.float32)
        opp_belief = False
        pan = np.full((N, 2, len(PANEL_KEYS)), np.nan, dtype=np.float32)
        for m in range(N):
            sd = game_seed(j["root"], j["cell"], m)
            for o, (sa, sb) in enumerate(((focal, opp), (opp, focal))):
                g = play(sa, sb, sd, T, panel=T > LANE_END)
                R[m, o] = g["r"] if o == 0 else -g["r"]
                if o in g["trace"]:
                    tr[m, o] = g["trace"][o]
                if 1 - o in g["trace"]:
                    otr[m, o], opp_belief = g["trace"][1 - o], True
                if g["panel"] is not None:
                    p = g["panel"]
                    pan[m, o] = [p[k] if o == 0 else _mirror_panel(k, p) for k in PANEL_KEYS]
        out[f"f{f}__r"], out[f"f{f}__trace"], out[f"f{f}__panel"] = R, tr, pan
        if opp_belief:
            out[f"f{f}__opp_trace"] = otr
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, **out, focals=np.asarray([repr(tuple(x)) for x in j["focals"]]),
                        opponent=np.asarray(repr(opp)), elapsed=np.asarray(time.perf_counter() - t0))
    os.replace(tmp, path)
    return j["key"], {"path": path, "elapsed": time.perf_counter() - t0}


def _mirror_panel(k: str, p: dict) -> float:
    """A panel entry in the other seat's orientation (differences negate, per-seat entries swap)."""
    if k in ("D0", "level_diff", "xp_diff", "hp_diff", "stock_diff"):
        return -p[k]
    if k.endswith("_A"):
        return p[k[:-2] + "_B"]
    if k.endswith("_B"):
        return p[k[:-2] + "_A"]
    return p[k]


def load(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def lead_at(R: np.ndarray, t: int = READ_TURN) -> np.ndarray:
    """Per pair: the focal's undiscounted lead after t turns, averaged over the two seat orientations."""
    return R[:, :, :t].sum(-1).mean(-1)


def discounted(R: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """Per pair: the focal's discounted lead over the whole game (the training objective's readout)."""
    w = gamma ** np.arange(R.shape[-1])
    return (R * w).sum(-1).mean(-1)


__all__ = ["T_EVAL", "T_LONG", "READ_TURN", "LANE_END", "CHECK_TURNS", "PANEL_KEYS", "opponent_agent", "play",
           "game_seed", "cell_job", "load", "lead_at", "discounted"]
