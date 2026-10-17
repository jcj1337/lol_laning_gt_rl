"""Start states of M4.6d (seat A = the stronger / freezing side, seat B = the weaker / frozen-on side, both in their
own coordinates).

Constructed cells (INITIALIZED, labelled; grounded in the natural distribution measured by the `freezes` stage):
    wave    freeze  A holds B's `held`-HP front at A's -1 with its own 8-HP front in its tower cell, A at its tower,
                    B walked up to its +1, B's next wave queued at its -2 (the M4.5b freeze builder; k 51)
            neutral both 8-HP fronts and both champions at the lane cell (M4.5b zoning "neutral"; k 2)
            push    A's 12-HP front at its +1 against B's 6-HP front: A pushing toward B's tower (zoning "a_push")
    level   both at the first XP of level L; invested gold on the natural path S(L) of M4.6c's archive
            (S(3) 0, S(5) 3.3, S(8) 11.6, S(4) 1.6, S(10) 18.4)
    strength (A over B)
            equal   same level and investment
            inv     A +3 invested (duel x1.10-1.15)
            lvl     A one level higher (duel x1.10)
            edge    A one level higher and +3 invested (duel x1.21-1.27): about the natural 90th percentile of the
                    early duel ratio (freezes stage)
            brk     the breaker (B) one level higher and +3 invested (control)
            x1.76   EXTREME DIAGNOSTIC: investment-only multiplier ratio 1.76 (duel x3.1, M4.6a's convention)
    HP      A full; B 1.0, 0.7, 0.6 or 0.45 of its max HP (the natural 75th-90th percentile of early HP gaps is
            0.43-0.63)
Natural states: freeze states (A holds a freeze) of each general policy's own s0 self-play at levels <= 9, split into
"favourable" (A present, HP >= 60%, duel >= x1.10, held wave >= 9 HP) and "other", collected on NATSTATE_ROOT seeds.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ....state import State
from . import common as C

LEVEL_S_PATH = ((3, 0.0), (6, 4.9), (9, 15.0), (12, 25.0), (14, 35.0), (17, 50.0), (18, 56.0))


def s_of_level(L: int) -> float:
    xs, ys = zip(*LEVEL_S_PATH)
    return round(float(np.interp(L, xs, ys)), 2)


def strength(cfg, cond: str, L: int) -> Tuple[int, int, float, float]:
    """(L_A, L_B, S_A, S_B) of a strength condition at level L."""
    S = s_of_level(L)
    if cond == "equal":
        return L, L, S, S
    if cond == "inv":
        return L, L, S + 3, S
    if cond == "lvl":
        return L + 1, L, S, S
    if cond == "edge":
        return L + 1, L, S + 3, S
    if cond == "brk":
        return L, L + 1, S, S + 3
    if cond == "x1.76":
        from ..m46a.states import invest_pair_mult
        lead, trail = invest_pair_mult(cfg, L, 1.76)
        return L, L, lead, trail
    raise ValueError(cond)


def build(cfg, wave: str, cond: str, L: int, hp_b: float = 1.0, held: int = 12, hp_a: float = 1.0):
    """(lane, champions, k) of a constructed cell (module docstring)."""
    from ....champions_m4 import context, initial_m4
    from ...m45b import states as S
    if wave == "freeze":
        lane, pos, k = State(-2, 8, (), 0, 1, held, (-2,), 0, 0, 0), (-2, 1), 51
    elif wave == "neutral":
        lane, pa, pb, _ = S.ZONING_SITUATIONS["neutral"]
        pos, k = (pa, pb), 2
    elif wave == "push":
        lane, pa, pb, _ = S.ZONING_SITUATIONS["a_push"]
        pos, k = (pa, pb), 2
    else:
        raise ValueError(wave)
    La, Lb, Sa, Sb = strength(cfg, cond, L)
    t = context(cfg).t
    m = initial_m4(cfg)
    sides = []
    for s, (side, Ls, Ss, hs) in enumerate(((m.a, La, Sa, hp_a), (m.b, Lb, Sb, hp_b))):
        sides.append(side._replace(pos=pos[s], xp=t.xp_cum[Ls], hp=max(1, int(round(hs * t.hp_max[Ls]))),
                                   items=int(round(Ss * cfg.gold_scale)), gold=0, tower_dmg=0))
    from ....champions_m4 import M4State
    return lane, M4State(*sides), k


def label(wave, cond, L, hp_b, held=12) -> str:
    return f"{wave}{held if wave == 'freeze' else ''} L{L} {cond} hpB{hp_b:g}"


# ---------------------------------------------------------------------------------------------- the grids
A_LEVELS = (3, 5, 8)
A_CONDS = ("equal", "inv", "lvl", "hp", "edge", "plaus", "brk", "x1.76")


def grid_a() -> List[dict]:
    """Workstream A: the freeze cells (full freezer x breaker matrix). 'hp' = equal power with B at 0.6 HP; 'plaus' =
    'edge' with B at 0.6 HP. Plus two small-held-wave variants at level 5."""
    out = []
    for L in A_LEVELS:
        for cond in A_CONDS:
            power = {"hp": "equal", "plaus": "edge"}.get(cond, cond)
            hpb = 0.6 if cond in ("hp", "plaus") else 1.0
            out.append({"wave": "freeze", "cond": power, "L": L, "hp_b": hpb, "held": 12, "name": cond})
    for cond in ("equal", "plaus"):
        power = {"plaus": "edge"}.get(cond, cond)
        out.append({"wave": "freeze", "cond": power, "L": 5, "hp_b": 0.6 if cond == "plaus" else 1.0, "held": 6,
                    "name": cond})
    return out


B_LEVELS = (4, 8)
B_CONDS = ("equal", "edge", "x1.76")
B_HP = (1.0, 0.7, 0.45)
B_WAVES = ("freeze", "neutral", "push")


def grid_b() -> List[dict]:
    """Workstream B: the breaker's decision map (freezer: policy or guard_sel; every breaker plan)."""
    return [{"wave": w, "cond": c, "L": L, "hp_b": h, "held": 12, "name": c}
            for w in B_WAVES for L in B_LEVELS for c in B_CONDS for h in B_HP]


def cell_start(cfg, cell: dict):
    return build(cfg, cell["wave"], cell["cond"], cell["L"], cell["hp_b"], cell.get("held", 12))


def cell_label(cell: dict) -> str:
    return f"{cell['wave']}{cell['held'] if cell['wave'] == 'freeze' else ''} L{cell['L']} {cell['name']} hpB{cell['hp_b']:g}"


# ---------------------------------------------------------------------------------------------- natural states
def natural_job(args) -> tuple:
    """(arm, spec, s, n_each, max_games): early natural freeze states of the policy's own self-play, favourable and
    other (module docstring), at most 2 per game, seat B states mirrored to A."""
    arm, spec, s, n_each, max_games = args
    cfg = C.setup(arm)
    from ...m45b import engine as E
    from ...m45b import states as S
    from ..m45c import horizon as HZ
    from ..m44.records import holds_freeze
    from .freezes import Strength
    from .plans import hp
    from ..m46a.states import level_of
    net = C.net_of(spec)
    st = Strength(cfg)
    sc = cfg.gold_scale

    def info(lane, m4, s_):
        o = 1 - s_
        a, b = m4.side(s_), m4.side(o)
        La, Lb = level_of(cfg, m4, s_), level_of(cfg, m4, o)
        duel, _, _ = st.parts(La, a.items / sc, Lb, b.items / sc)
        return La, duel, hp(cfg, m4, s_), lane.side(o)[1]

    def p_early(cfg_, lane, m4, s_):
        return m4.side(s_).absent == 0 and holds_freeze(lane, s_) and info(lane, m4, s_)[0] <= 9

    fav, oth, games = [], [], 0
    for g in range(max_games):
        sd = C.NATSTATE_ROOT * 1000 + 100 * s + g
        got = E.collect(cfg, S.s0(cfg), HZ.CRNNet(net, 0, sd, 0, 150), HZ.CRNNet(net, 0, sd, 1, 150), [sd], 150,
                        p_early, per_game=2, min_turn=3, rng_seed=sd)
        games += 1
        for x in got:
            lane, m4, k = x["start"]
            La, duel, hpa, held_hp = info(lane, m4, 0)
            f = hpa >= 0.6 and duel >= 1.10 and held_hp >= 9
            rec = {"start": C.state_to_list(x["start"]), "turn": x["turn"], "seed": sd, "L": La, "duel": duel,
                   "hp_a": hpa, "hp_b": hp(cfg, m4, 1), "held": held_hp, "favourable": bool(f)}
            (fav if f else oth).append(rec)
        if len(fav) >= n_each and len(oth) >= n_each:
            break
    return (arm, spec), {"favourable": fav[:n_each], "other": oth[:n_each], "games": games,
                         "found": {"favourable": len(fav), "other": len(oth)}}


__all__ = ["build", "grid_a", "grid_b", "cell_start", "cell_label", "strength", "s_of_level", "natural_job"]
