"""Lane-exit evaluation: the economic readout of a checkpoint state from short repeated continuations (M4.8).

EVALUATION ONLY. Nothing here changes the game, the reward, the objective (the discounted earned-gold lead, gamma
0.99, continuing time) or any training procedure. A "lane exit" is an evaluation checkpoint in a continuing game, not a
terminal state: the game goes on after it and the readout looks a few turns past it.

Why. The earned-gold lead at an instant can misstate who leaves the lane better placed when farm and resets are
pending: a side that has already recalled and is walking back to a stacked wave is slightly behind now and about to
collect, while a side that just collected still has to pay for its Recall. Short continuations from the checkpoint
resolve such pending consequences under the continuation policies.

Definitions (seat A's orientation; resources, 1 = one fresh wave; reference §7.2):

    earned_s   = S_s + g_s                       invested + unspent gold: every resource side s has received (farm +
                                                 plate gold + bounty). Investing on a return moves g into S and leaves
                                                 earned_s unchanged, so spending never counts as earning again.
    D0(s)      = earned_A - earned_B              the raw earned-gold lead at the checkpoint
    Delta_h(s) = E[ sum_{t<h} r_t ]               the expected change of that lead over the next h turns under the
                                                 continuation policies (r_t = the reward to A of turn t, exact ledger
                                                 increments; undiscounted)
    E_h(s)     = D0(s) + Delta_h(s)               the h-turn LANE-EXIT ECONOMIC READOUT: the expected earned-gold lead h
                                                 turns after the checkpoint

E_h is an inventory-style quantity: undiscounted, no critic tail, no bootstrap. It is NOT the discounted G_h of earlier
studies (sum_{t<h} 0.99^t r_t, a change only) and NOT a critic value. It is an expectation under the continuation
policies (by default the policies that produced the state), not a game-theoretic value, and it is a gold readout only:
XP, levels, HP, absence and the wave are reported beside it (`describe`, `window_state`) and are never folded into it.

A continuation is one game of T turns from the checkpoint. Turn i draws its chance vector as the i-th draw of N_CHANCE
uniforms from default_rng(seed) whatever the actions (common random numbers), exactly as every M4 rollout engine does;
policies are any objects with `decide(cfg, lane, m4, s, i) -> (code, why)` (`experiments.m45b.engine` interface). With
`mirror=True` the chance vector of each turn is mirrored (`champions.mirror_chance`): the continuation of the mirrored
state with mirrored chance and swapped seat policies is then the exact mirror image of the original (the resolver is
mirror-equivariant), which the tests use to check orientation.

The per-seat event ledger of a continuation (`continuation`), arrays [T, 2] over turns i = 0 .. T-1 (turn i of the
continuation) unless noted:
    earned, farm, plate, bounty       resources received on the turn (earned = farm + plate + bounty, exactly)
    avail                             farm available on the seat's kill ledger (farm_per_hp per enemy minion HP that
                                      died, collected or not); lost = avail - farm
    away, died, recalled, returned    absent during the turn / died / left by a Recall / absence ended (plays next turn)
    xpg                               XP credited
    hp, absent, xp, level             the state AFTER the turn (hp as a fraction of max HP, 1.0 while absent)
    r [T]                             the reward to seat A (resources)
    acct_units (scalar)               lead_units(final) - lead_units(start) - sum of the turns' reward_units: 0 exactly
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from ..champions import N_CHANCE, mirror_chance
from ..champions_m4 import M4State, context, resolve_turn_m4

A, B = 0, 1
SIDE_KEYS = ("earned", "farm", "plate", "bounty", "avail", "away", "died", "recalled", "returned", "xpg", "hp",
             "absent", "xp", "level")
_FLOAT = ("earned", "farm", "plate", "bounty", "avail", "hp")
_BOOL = ("away", "died", "recalled", "returned")
_INT = ("xpg", "absent", "xp", "level")


# ---------------------------------------------------------------------------------------------- accounting
def _check_continuous(cfg) -> None:
    if not cfg.continuous_items:
        raise ValueError("lane-exit accounting needs continuous investment (earned = invested + unspent); the tier "
                         "item models clamp unspent gold")


def earned_units(m4: M4State, s: int, cfg) -> int:
    """Seat s's cumulative earned gold in ledger units: invested + unspent (reference §7.2)."""
    _check_continuous(cfg)
    me = m4.side(s)
    return int(me.items) + int(me.gold)


def lead_units(m4: M4State, cfg) -> int:
    """Seat A's earned-gold lead in ledger units."""
    return earned_units(m4, A, cfg) - earned_units(m4, B, cfg)


def lead(m4: M4State, cfg) -> float:
    """Seat A's earned-gold lead D0 in resources."""
    return lead_units(m4, cfg) / cfg.gold_scale


def mirror_state(start):
    """The same situation with the seats swapped (each side's record is in its own coordinates)."""
    lane, m4, k = start
    return lane.mirror(), m4.mirror(), int(k)


# ---------------------------------------------------------------------------------------------- the checkpoint
def seat_status(m4: M4State, s: int) -> str:
    """"present", "recall" (away after a Recall, HP kept) or "dead" (away after a death, HP 0)."""
    me = m4.side(s)
    if me.absent <= 0:
        return "present"
    return "dead" if me.hp <= 0 else "recall"


def wave_side(lane, s: int, cfg) -> Dict[str, float]:
    """Seat s's wave situation in its own coordinates (own tower -T, the lane 0, enemy tower +T).

    stock     HP of ENEMY minions alive now (front + rear packets + the enemy's crashed mass under s's tower): every
              HP of it will die on s's kill ledger, collected by s or lost
    near      the part of `stock` at s's own cell -1 or deeper (including the crashed mass)
    siege     HP of s's own crashed minions under the enemy tower (dying on the enemy's ledger)
    stock_res / near_res   the same in resources at full collection (farm_per_hp per HP)
    """
    T = cfg.tower
    ofx, ofh, orear, osg, _, _ = lane.side(s)
    efx, efh, erear, esg, _, _ = lane.side(1 - s)
    stock = (efh if efh > 0 else 0) + cfg.wave_hp * len(erear) + esg
    near = (efh if efh > 0 and -efx <= -1 else 0) + cfg.wave_hp * sum(1 for r in erear if -r <= -1) + esg
    contact = ofh > 0 and efh > 0 and ofx + efx >= -1
    return {"own_front": float(ofx) if ofh > 0 else float("nan"), "own_hp": float(ofh),
            "enemy_front": float(-efx) if efh > 0 else float("nan"), "enemy_hp": float(efh),
            "own_rears": float(len(orear)), "enemy_rears": float(len(erear)),
            "stock": float(stock), "near": float(near), "siege": float(osg), "enemy_siege": float(esg),
            "stock_res": stock * cfg.farm_per_hp, "near_res": near * cfg.farm_per_hp,
            "contact": float(contact), "holds_freeze": float(contact and -efx <= -1),
            "own_tower": float(-T)}


def describe(cfg, start, history: Optional[Dict[str, Sequence[float]]] = None) -> Dict[str, float]:
    """Everything the readout reports beside E_h at a checkpoint (seat A's orientation for differences).

    Per seat (suffix _A / _B): status code (0 present, 1 recall, 2 dead), returns_in (turns still away), hp (fraction;
    1.0 while away), level, xp, invested, unspent, earned, pos (own coordinates; 99 = the Vision node W) and the
    `wave_side` quantities. Differences: D0, xp_diff, level_diff, hp_diff (both present only, else nan), stock_diff.
    `history` (optional) adds since_return_A / _B: turns since the seat last came back from an absence (or since the
    game's start)."""
    _check_continuous(cfg)
    lane, m4, k = start
    tb = context(cfg).t
    sc = cfg.gold_scale
    out: Dict[str, float] = {"k": float(k)}
    code = {"present": 0.0, "recall": 1.0, "dead": 2.0}
    for s, sx in ((A, "A"), (B, "B")):
        me = m4.side(s)
        L = tb.level_by_xp[min(me.xp, tb.xp_cap)]
        out[f"status_{sx}"] = code[seat_status(m4, s)]
        out[f"returns_in_{sx}"] = float(me.absent)
        out[f"hp_{sx}"] = 1.0 if me.absent else me.hp / tb.hp_max[L]
        out[f"level_{sx}"] = float(L)
        out[f"xp_{sx}"] = float(me.xp)
        out[f"invested_{sx}"] = me.items / sc
        out[f"unspent_{sx}"] = me.gold / sc
        out[f"earned_{sx}"] = (me.items + me.gold) / sc
        out[f"pos_{sx}"] = float(me.pos)
        for key, v in wave_side(lane, s, cfg).items():
            out[f"{key}_{sx}"] = v
    out["D0"] = lead(m4, cfg)
    out["xp_diff"] = out["xp_A"] - out["xp_B"]
    out["level_diff"] = out["level_A"] - out["level_B"]
    both = out["status_A"] == 0 and out["status_B"] == 0
    out["hp_diff"] = out["hp_A"] - out["hp_B"] if both else float("nan")
    out["stock_diff"] = out["stock_A"] - out["stock_B"]
    if history is not None:
        for sx in ("A", "B"):
            out[f"since_return_{sx}"] = float(history[f"since_return_{sx}"])
    return out


def since_return(absent: Sequence[int], i: int) -> int:
    """Turns since the seat last came back from an absence, at the start of turn i, from its start-of-turn absence
    counters absent[0 .. i] (a return completes at the end of turn j when absent[j] > 0 and absent[j + 1] == 0); i if it
    never left (the game's start counts as a fresh start)."""
    for j in range(i - 1, -1, -1):
        if absent[j] > 0 and absent[j + 1] == 0:
            return i - (j + 1)
    return i


# ---------------------------------------------------------------------------------------------- one continuation
def continuation(cfg, start, pa, pb, seed: int, T: int, lane_step=None, mirror: bool = False,
                 ctx=None, keep_states: bool = False) -> Dict[str, np.ndarray]:
    """One continuation of T turns from `start` with seat policies pa, pb and chance stream default_rng(seed) (mirrored
    turn by turn when `mirror`); returns the event ledger of the module docstring, plus "states" (the (lane, m4, k) at
    the start of every turn and the final one, T + 1 entries) with `keep_states`."""
    _check_continuous(cfg)
    lane, m4, k0 = start
    ctx = ctx or context(cfg)
    tb = ctx.t
    sc = cfg.gold_scale
    rng = np.random.default_rng(int(seed))
    out = {k: np.zeros((T, 2), dtype=np.float32 if k in _FLOAT else (bool if k in _BOOL else np.int16))
           for k in SIDE_KEYS}
    r = np.zeros(T, dtype=np.float64)
    units = 0
    lead0 = lead_units(m4, cfg)
    states = [(lane, m4, int(k0))] if keep_states else None
    for i in range(T):
        u = rng.random(N_CHANCE)
        if mirror:
            u = mirror_chance(u)
        a = _decide(pa, cfg, lane, m4, A, i)
        b = _decide(pb, cfg, lane, m4, B, i)
        t = resolve_turn_m4(lane, m4, k0 + i, a, b, cfg, lane_step=lane_step, u=u, ctx=ctx)
        units += t.reward_units
        r[i] = t.reward
        lane, m4 = t.lane.state, t.champ
        if states is not None:
            states.append((lane, m4, int(k0) + i + 1))
        for s in (A, B):
            me = m4.side(s)
            L = tb.level_by_xp[min(me.xp, tb.xp_cap)]
            out["earned"][i, s] = t.earned_units[s] / sc
            out["farm"][i, s] = t.farm_units[s] / sc
            out["plate"][i, s] = t.plate_units[s] / sc
            out["bounty"][i, s] = t.bounty_units[s] / sc
            out["avail"][i, s] = t.farm_available[s]
            out["away"][i, s] = t.away[s]
            out["died"][i, s] = t.died[s]
            out["recalled"][i, s] = t.recalled[s]
            out["returned"][i, s] = t.returned[s]
            out["xpg"][i, s] = t.xp_gain[s]
            out["hp"][i, s] = 1.0 if me.absent else me.hp / tb.hp_max[L]
            out["absent"][i, s] = me.absent
            out["xp"][i, s] = me.xp
            out["level"][i, s] = L
    out["r"] = r
    out["acct_units"] = np.int64(lead_units(m4, cfg) - lead0 - units)
    if states is not None:
        out["states"] = states
    return out


def _decide(pol, cfg, lane, m4, s: int, i: int) -> int:
    if hasattr(pol, "decide"):
        return int(pol.decide(cfg, lane, m4, s, i)[0])
    return int(pol(cfg, lane, m4, s, i))


def continuations(cfg, start, make_policies: Callable[[int], Tuple[object, object]], seeds: Sequence[int], T: int,
                  lane_step=None, mirror: bool = False) -> Dict[str, np.ndarray]:
    """K continuations (one per seed; `make_policies(seed) -> (pa, pb)`), stacked: arrays [K, T, 2], r [K, T],
    acct_units [K]. Raises if any continuation's accounting is not exact."""
    ctx = context(cfg)
    runs = []
    for sd in seeds:
        pa, pb = make_policies(int(sd))
        runs.append(continuation(cfg, start, pa, pb, int(sd), T, lane_step=lane_step, mirror=mirror, ctx=ctx))
    out = {k: np.stack([x[k] for x in runs]) for k in runs[0]}
    if np.any(out["acct_units"] != 0):
        raise AssertionError(f"lane-exit accounting is not exact: {out['acct_units'].tolist()}")
    return out


# ---------------------------------------------------------------------------------------------- readouts
def deltas(r: np.ndarray, h: int) -> np.ndarray:
    """r [..., T] -> the lead change over the first h turns of each continuation (prefix sums of the SAME streams)."""
    if h > r.shape[-1]:
        raise ValueError(f"window {h} exceeds the continuation length {r.shape[-1]}")
    return r[..., :h].sum(-1)


def mean_ci(x: np.ndarray, z: float = 1.96) -> Dict[str, float]:
    """Mean, standard error, standard deviation and the normal 95% half-width over the last axis (repetitions)."""
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    m = x.mean(-1)
    sd = x.std(-1, ddof=1) if n > 1 else np.full_like(m, np.nan)
    se = sd / np.sqrt(n)
    return {"mean": m, "se": se, "sd": sd, "ci": z * se, "n": n}


def readout(D0: float, r: np.ndarray, windows: Iterable[int]) -> Dict[int, Dict[str, float]]:
    """E_h = D0 + Delta_h for every window h from one set of continuations r [K, T] (fixed windows, no stopping
    rule): {h: {E, delta, se, sd, ci, lo, hi, K}}; lo / hi bound E_h's 95% interval."""
    out = {}
    for h in windows:
        st = mean_ci(deltas(r, h))
        d = float(st["mean"])
        out[int(h)] = {"E": D0 + d, "delta": d, "se": float(st["se"]), "sd": float(st["sd"]), "ci": float(st["ci"]),
                       "lo": D0 + d - float(st["ci"]), "hi": D0 + d + float(st["ci"]), "K": int(st["n"])}
    return out


def paired(r1: np.ndarray, r0: np.ndarray, h: int, D1: float = 0.0, D0: float = 0.0) -> Dict[str, float]:
    """E_h of one alternative minus another on common random numbers (rollout k of both used the same streams):
    (D1 - D0) + mean_k [Delta_h^1 - Delta_h^0] with its standard error."""
    st = mean_ci(deltas(r1, h) - deltas(r0, h))
    return {"diff": (D1 - D0) + float(st["mean"]), "se": float(st["se"]), "ci": float(st["ci"]), "K": int(st["n"])}


def components(led: Dict[str, np.ndarray], h: int) -> Dict[str, float]:
    """Mean per-continuation event accounting over the first h turns, per seat (_A / _B) and as A - B:
    farm, plate, bounty, earned, avail, lost (= avail - farm), lost_away (lost on turns the seat was away), deaths,
    recalls (successful Recall turns), returns, away turns, xp. The identity earned = farm + plate + bounty holds per
    seat and turn, and sum_t r_t = earned_A - earned_B per continuation."""
    w = slice(0, h)
    out: Dict[str, float] = {}
    lost = led["avail"][:, w] - led["farm"][:, w]
    per = {"farm": led["farm"][:, w], "plate": led["plate"][:, w], "bounty": led["bounty"][:, w],
           "earned": led["earned"][:, w], "avail": led["avail"][:, w], "lost": lost,
           "lost_away": lost * led["away"][:, w], "deaths": led["died"][:, w], "recalls": led["recalled"][:, w],
           "returns": led["returned"][:, w], "away": led["away"][:, w], "xp": led["xpg"][:, w]}
    for k, v in per.items():
        tot = v.astype(float).sum(1)                       # [K, 2]
        out[f"{k}_A"] = float(tot[:, A].mean())
        out[f"{k}_B"] = float(tot[:, B].mean())
        out[f"{k}_diff"] = float((tot[:, A] - tot[:, B]).mean())
    return out


def window_state(led: Dict[str, np.ndarray], h: int) -> Dict[str, float]:
    """The non-gold dimensions h turns after the checkpoint (means over continuations): XP difference, levels, P(away)
    per seat, HP of each seat while present, and the HP difference over continuations in which both are present."""
    i = h - 1
    xp = led["xp"][:, i].astype(float)
    lv = led["level"][:, i].astype(float)
    ab = led["absent"][:, i] > 0
    hp = led["hp"][:, i].astype(float)
    both = ~ab[:, A] & ~ab[:, B]
    out = {"xp_diff": float((xp[:, A] - xp[:, B]).mean()), "level_A": float(lv[:, A].mean()),
           "level_B": float(lv[:, B].mean()), "away_A": float(ab[:, A].mean()), "away_B": float(ab[:, B].mean()),
           "hp_diff_present": float((hp[both, A] - hp[both, B]).mean()) if both.any() else float("nan")}
    for s, sx in ((A, "A"), (B, "B")):
        pres = ~ab[:, s]
        out[f"hp_{sx}"] = float(hp[pres, s].mean()) if pres.any() else float("nan")
    return out


# ---------------------------------------------------------------------------------------------- pending consequences
def first_turn(mask: np.ndarray) -> np.ndarray:
    """mask [K, T] -> per continuation the number of turns after which the event first happened (1-based: the event
    happened during turn index t-1), or inf if it never did within the window."""
    hit = mask.any(-1)
    return np.where(hit, mask.argmax(-1) + 1.0, np.inf)


def return_times(led: Dict[str, np.ndarray]) -> np.ndarray:
    """[K, 2] turns until each seat next comes back from an absence (its Recall or death, then the absence; a seat
    already away comes back after its remaining absence), inf if not within the window."""
    return np.stack([first_turn(led["returned"][:, :, s]) for s in (A, B)], axis=-1)


def stock_times(led: Dict[str, np.ndarray], stock_hp: Tuple[float, float], farm_per_hp: float) -> np.ndarray:
    """[K, 2] turns until the enemy minion HP alive at the checkpoint (`wave_side` stock, per seat) has died on the
    seat's kill ledger, counting first-in first-out by HP (minion HP that dies is the oldest HP; packets merge, so HP is
    the unit); 0 if there was none, inf if not within the window."""
    out = np.zeros(led["avail"].shape[:1] + (2,))
    for s in (A, B):
        died = np.cumsum(led["avail"][:, :, s].astype(float), axis=-1) / farm_per_hp
        need = float(stock_hp[s])
        if need <= 0:
            continue
        out[:, s] = first_turn(died >= need - 1e-6)
    return out


def stock_collected(led: Dict[str, np.ndarray], stock_hp: Tuple[float, float], farm_per_hp: float) -> np.ndarray:
    """[K, 2] the share of each seat's checkpoint stock that it collected (gold), FIFO by HP: of the first `stock` HP
    that died on its ledger, the collected fraction (each turn's collection rate applied to the part of that turn's
    deaths that belong to the stock); nan if no stock."""
    K, T = led["avail"].shape[:2]
    out = np.full((K, 2), np.nan)
    for s in (A, B):
        need = float(stock_hp[s])
        if need <= 0:
            continue
        av = led["avail"][:, :, s].astype(float) / farm_per_hp
        fm = led["farm"][:, :, s].astype(float) / farm_per_hp
        for kk in range(K):
            left, got, seen = need, 0.0, 0.0
            for i in range(T):
                if left <= 1e-9:
                    break
                if av[kk, i] <= 0:
                    continue
                take = min(left, av[kk, i])
                got += take * fm[kk, i] / av[kk, i]
                seen += take
                left -= take
            out[kk, s] = got / seen if seen > 0 else np.nan
    return out


__all__ = ["SIDE_KEYS", "earned_units", "lead_units", "lead", "mirror_state", "seat_status", "wave_side", "describe",
           "since_return", "continuation", "continuations", "deltas", "mean_ci", "readout", "paired", "components",
           "window_state", "first_turn", "return_times", "stock_times", "stock_collected"]
