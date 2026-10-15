"""Start states of M4.6a (pre-registration §2). Every state is in seat A's coordinates with A the focal side.

Constructed (INITIALIZED, labelled): the M4.5b zoning situations and freeze builder with an arbitrary attack-power
multiplier m = M(S_A) / M(S_B) at a level (`invest_pair_mult`: both at the level's typical investment lowered on the
trailing side, or the trailing side at 0 and the leading side at M = m when that is impossible; investment only, so
the duel ratio is m^2).

Natural: decision states of a final's own self-play satisfying a predicate of the START-OF-TURN state only (never an
outcome), collected with `m45b.engine.collect` (seat B states mirrored to A). Predicates:
    hp_edge     both present, focal HP-aware duel edge >= 0.15, the enemy outside its own tower cell, catchable
                without a dive (E2 natural family; M4.4c's "edge" states)
    freeze      the focal side holds a freeze (M4.5c p_freeze; E4's freeze archive)
    recall_full Recall legal, HP >= 90%, >= 2 resources unspent (E5)
    recall_mid  Recall legal, 40% <= HP <= 60% (E5)
    hp_ok       both present, focal HP >= 70% (E5 price of HP; early = levels <= 10 in 60-turn games)
Semi-natural gap states come from E3's gap self-play records (e3.py).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from ....config import B_RECALL, W_HOLD, champ_act
from ....champions import champion_legal_actions

GAP_MULTS = {"equal": 1.0, "x1.24": 1.24, "x1.5": 1.5, "x1.76": 1.76, "x2.2": 2.2}


def invest_pair_mult(cfg, level: int, m: float) -> Tuple[float, float]:
    from ...m45b import states as S
    base = S.typical_invest(level)
    if m == 1.0:
        return base, base
    trail = S.invest_for(cfg, S.multiplier(cfg, base) / m)
    lead = base
    if trail is None:
        trail, lead = 0.0, S.invest_for(cfg, m)
    return lead, trail


def zoning_gap_state(cfg, situation: str, level: int, gap: str, hp=(1.0, 1.0), k: int = 2):
    """The M4.5b zoning situation (neutral / a_push / b_push) with A's power lead `gap` (GAP_MULTS)."""
    from ...m45b import states as S
    lane, pa, pb, _ = S.ZONING_SITUATIONS[situation]
    inv = invest_pair_mult(cfg, level, GAP_MULTS[gap])
    return lane, S.champs(cfg, (level, level), inv, (pa, pb), hp), k


def freeze_gap_state(cfg, level: int, gap: str, held: int, own: int = 8, b_pos: int = 1):
    """The M4.5b freeze builder (A holds B's `held`-HP wave at A's -1) with A's power lead `gap`."""
    from ...m45b import states as S
    st = S.freeze_state(cfg, level=level, gap="equal", held=held, own=own, b_pos=b_pos)
    lane, m4, k = st
    inv = invest_pair_mult(cfg, level, GAP_MULTS[gap])
    a, b = m4.a, m4.b
    m4 = m4._replace(a=a._replace(items=int(round(inv[0] * cfg.gold_scale))),
                     b=b._replace(items=int(round(inv[1] * cfg.gold_scale))))
    return lane, m4, k


def mirror(start):
    lane, m4, k = start
    return lane.mirror(), m4.mirror(), k


# ---------------------------------------------------------------------------------------------- predicates
def _present(m4, s) -> bool:
    return m4.side(s).absent == 0


def p_hp_edge(cfg, lane, m4, s) -> bool:
    from ...m4.zoning_natural import duel_edge
    from ...m45b.strategies import catchable
    if not (_present(m4, s) and _present(m4, 1 - s)):
        return False
    if m4.side(1 - s).pos == -cfg.tower:
        return False
    return duel_edge(cfg, lane, m4, s) >= 0.15 and catchable(cfg, m4, s)


def p_freeze(cfg, lane, m4, s) -> bool:
    from ..m45c.strategic import p_freeze as pf
    return pf(cfg, lane, m4, s)


def _recall_legal(cfg, lane, m4, s) -> bool:
    return champ_act(W_HOLD, B_RECALL) in champion_legal_actions(lane, m4, s, cfg)


def p_recall_full(cfg, lane, m4, s) -> bool:
    from ...m45b.strategies import hp_frac
    return (_recall_legal(cfg, lane, m4, s) and hp_frac(cfg, m4, s) >= 0.9
            and m4.side(s).gold >= 2 * cfg.gold_scale)


def p_recall_mid(cfg, lane, m4, s) -> bool:
    from ...m45b.strategies import hp_frac
    return _recall_legal(cfg, lane, m4, s) and 0.4 <= hp_frac(cfg, m4, s) <= 0.6


def p_hp_ok(cfg, lane, m4, s) -> bool:
    from ...m45b.strategies import hp_frac
    return _present(m4, s) and _present(m4, 1 - s) and hp_frac(cfg, m4, s) >= 0.7


def level_of(cfg, m4, s) -> int:
    from ....champions_m4 import context
    t = context(cfg).t
    return int(t.level_by_xp[min(m4.side(s).xp, t.xp_cap)])


def early(pred: Callable, max_level: int = 10) -> Callable:
    def p(cfg, lane, m4, s):
        return level_of(cfg, m4, s) <= max_level and pred(cfg, lane, m4, s)
    return p


def late(pred: Callable, min_level: int = 11) -> Callable:
    def p(cfg, lane, m4, s):
        return level_of(cfg, m4, s) >= min_level and pred(cfg, lane, m4, s)
    return p


PREDICATES: Dict[str, Tuple[Callable, int, int]] = {
    # name: (predicate, game turns, minimum turn)
    "hp_edge": (p_hp_edge, 300, 20),
    "freeze": (p_freeze, 300, 20),
    "recall_full": (p_recall_full, 300, 20),
    "recall_mid": (p_recall_mid, 300, 20),
    "hp_early": (early(p_hp_ok), 60, 3),
    "hp_late": (late(p_hp_ok), 300, 20),
}


def collect_job(args) -> tuple:
    """(final seed, family, n, root, offset, per_game): n natural states of a family from the final's self-play; game g
    uses seed root * 10 + offset + g (both seats the final, sampling)."""
    seed, fam, n, root, offset, per_game = args
    from .common import setup_worker, spec
    cfg = setup_worker()
    from ...m45b import engine as E
    from ...m45b import states as S
    from ...m45b.reference import NetPolicy
    from ..m45c.common import agent_from_spec
    agent = agent_from_spec(spec(seed), cfg)
    pred, turns, min_turn = PREDICATES[fam]
    out, games = [], 0
    for g in range(600):
        sd = root * 10 + offset + g
        out += E.collect(cfg, S.s0(cfg), NetPolicy(agent, 0, sd, 0), NetPolicy(agent, 0, sd, 1), [sd], turns, pred,
                         per_game=per_game, min_turn=min_turn, rng_seed=sd)
        games += 1
        if len(out) >= n:
            break
    return (seed, fam), {"states": out[:n], "games": games}


__all__ = ["GAP_MULTS", "invest_pair_mult", "zoning_gap_state", "freeze_gap_state", "mirror", "PREDICATES",
           "collect_job", "level_of", "p_hp_edge", "p_freeze"]
