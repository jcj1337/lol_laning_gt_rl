"""The M4 observation (docs/m4/m4_integrated_design.md §12; plan WP7): lossless, own-perspective, fixed length, every
entry in [0, 1], a function of (k mod P, lane, M4 champion state, side) only. No absolute time, no countdown.

Layout (88 entries with the default geometry):

  phase                 P = 4    one-hot of k mod P
  own lane              19       front cell one-hot over cells -base .. base (all zero = no front); front HP / hp_cap;
                                 rear-packet count per cell / rear_count_max, cells -base .. base
  enemy lane            19       the same for the enemy, mirrored into own coordinates
  own side              3        siege mass / siege_cap; vision / vision_turns; siege eligible
  enemy side            3        the same for the opponent
  own tower progress    4        tau / theta_last; plates broken / n; progress to the next plate (1 if all broken);
                                 plates left / n           (tau = damage WE dealt to the enemy tower)
  enemy tower progress  4        the same for the opponent's tau (damage dealt to OUR tower)
  own champion          16       cell one-hot (2T + 1); hp / H(L); H(L) / H(cap); absent / absent_max; skill;
                                 (L - 1) / (cap - 1); XP progress (x - X(L)) / n(L) (0 at the cap); items / tiers;
                                 gold / G_rem(0) (exact ledger units); can buy the next tier; can buy the next two
                                 tiers; AP(L, N) / AP(cap, tiers)
  enemy champion        16       the same, the cell mirrored

`decode_m4(vector, cfg)` is the constructive inverse (design §12.2): it reads (phase, lane, M4 state, skills) back
from the float32 vector, in the observer's own perspective (the observer as side A). Exactness rests on two
enumerated bounds of the default lane (rear packets per cell <= rear_count_max = 4, siege mass <= siege_cap = 48);
`observe_m4` raises instead of clipping if either is exceeded.

M4.2c continuous investment (item_model = "continuous", an experiment): the five item entries of each champion block
are replaced (same length) by

    invested_whole  min(floor(S / g), R_S) / R_S      S = invested gold (ledger units), g = gold_scale, R_S =
                                                     invest_obs_ref resources
    gold_whole      min(floor(G / g), R_G) / R_G      G = unspent gold, R_G = invest_obs_gold_ref resources
    invested_frac   (S mod g) / g                     the sub-resource part, exact in float32 for g <= 2^24
    gold_frac       (G mod g) / g
    attack_power    AP*(L, S) / (AP*(L, S) + AP(cap, 0))   bounded in (0, 1) for every S; not used by the decoder

Both quantities are unbounded in the game, so the encoding is lossless only while S < R_S and G < R_G resources
(the whole-resource entry is then < 1); at or beyond a reference the entry saturates at 1.0, the observation is no
longer injective and `decode_m4` raises. `observe_m4` does NOT raise there (the game itself has no cap).

M4.2d opt-in (invest_obs_encoding = "compressed", docs/m4/m4_investment_readiness_report.md): the two whole-resource
entries become n / (n + R) (n = floor(S / g) or floor(G / g); R = invest_obs_ref or invest_obs_gold_ref, now the
half-point), the other three entries are unchanged and the length stays 88. The entry is monotone, < 1 for every
reachable n (no saturation), well resolved near 0 (one resource of unspent gold moves it by about 1 / R), and
exactly decodable in float32 for n <= compressed_exact_max(R) = isqrt(R 2^23) - R (8,184 resources at R = 8; 23,106
at R = 64); beyond that bound the observation stays valid and `decode_m4` raises.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from ..config import GameConfig
from ..progression import tables
from ..state import State

CHAMP_TAIL = ("hp_frac", "hp_max", "absent", "skill", "level", "xp_progress", "items", "gold", "can_buy",
              "can_buy_two", "attack_power")
CHAMP_TAIL_CONTINUOUS = ("hp_frac", "hp_max", "absent", "skill", "level", "xp_progress", "invested_whole",
                         "gold_whole", "invested_frac", "gold_frac", "attack_power")
TOWER_NAMES = ("tower_dmg", "plates_broken", "plate_progress", "plates_left")
SIDE_NAMES = ("siege_mass", "vision", "siege_eligible")


def compressed_exact_max(ref: int) -> int:
    """Largest whole-resource count n whose "compressed" entry n / (n + ref) decodes exactly from float32: the
    float32 rounding error (<= 2^-25) times dn/dx = (n + ref)^2 / ref stays <= 1/4 for n + ref <= sqrt(ref 2^23)."""
    return math.isqrt(ref << 23) - ref


def _sizes(cfg: GameConfig):
    cells = 2 * cfg.base + 1
    lane = cells + 1 + cells
    champ = (2 * cfg.tower + 1) + len(CHAMP_TAIL)
    return cells, lane, champ


def m4_obs_size(cfg: GameConfig) -> int:
    cells, lane, champ = _sizes(cfg)
    return cfg.phase_period + 2 * lane + 2 * len(SIDE_NAMES) + 2 * len(TOWER_NAMES) + 2 * champ


def m4_feature_names(cfg: GameConfig) -> List[str]:
    names = [f"phase_{p}" for p in range(cfg.phase_period)]
    for who in ("own", "enemy"):
        names += [f"{who}_front_x{x:+d}" for x in range(-cfg.base, cfg.base + 1)]
        names += [f"{who}_front_hp"]
        names += [f"{who}_rears_x{x:+d}" for x in range(-cfg.base, cfg.base + 1)]
    for who in ("own", "enemy"):
        names += [f"{who}_{n}" for n in SIDE_NAMES]
    for who in ("own", "enemy"):
        names += [f"{who}_{n}" for n in TOWER_NAMES]
    for who in ("own", "enemy"):
        names += [f"{who}_champ_x{x:+d}" for x in range(-cfg.tower, cfg.tower + 1)]
        names += [f"{who}_champ_{n}" for n in (CHAMP_TAIL_CONTINUOUS if cfg.continuous_items else CHAMP_TAIL)]
    assert len(names) == m4_obs_size(cfg)
    return names


def _lane_block(fx, fh, rears, sgn, cfg: GameConfig, v: list, o: int) -> None:
    base, cells = cfg.base, 2 * cfg.base + 1
    if fh > 0:
        x = sgn * fx
        if not -base <= x <= base:
            raise ValueError(f"front at {fx} outside the lane")
        v[o + x + base] = 1.0
        v[o + cells] = fh / cfg.hp_cap
    r0 = o + cells + 1
    rmax = cfg.rear_count_max
    for x in rears:
        v[r0 + sgn * x + base] += 1.0
    for i in range(r0, r0 + cells):
        n = v[i]
        if n:
            if n > rmax:
                raise ValueError(f"{int(n)} rear packets in one cell exceed rear_count_max = {rmax}: the lossless "
                                 "encoding's enumerated bound is violated (re-run the lane enumeration)")
            v[i] = n / rmax


def _tower_block(tau: int, cfg: GameConfig, v: list, o: int) -> None:
    th = cfg.plate_thresholds
    n = len(th)
    pi = 0
    while pi < n and tau >= th[pi]:
        pi += 1
    v[o] = tau / th[-1]
    v[o + 1] = pi / n
    if pi < n:
        lo = th[pi - 1] if pi > 0 else 0
        v[o + 2] = (tau - lo) / (th[pi] - lo)
    else:
        v[o + 2] = 1.0
    v[o + 3] = (n - pi) / n


def observe_m4(state: State, k: int, side: int, cfg: GameConfig, champ) -> np.ndarray:
    """Own-perspective M4 feature vector of `side` (float32, entries in [0, 1]); `champ` is the M4State."""
    if champ is None:
        raise ValueError("ruleset m4: observe() needs the M4 champion state")
    t = tables(cfg)
    P = cfg.phase_period
    cells, lane_n, champ_n = _sizes(cfg)
    v = [0.0] * m4_obs_size(cfg)
    v[k % P] = 1.0
    o = P
    for who, sgn in ((side, 1), (1 - side, -1)):
        fx, fh, rears = state.side(who)[:3]
        _lane_block(fx, fh, rears, sgn, cfg, v, o)
        o += lane_n
    for who in (side, 1 - side):
        siege = state.siege_of(who)
        if siege > cfg.siege_cap:
            raise ValueError(f"siege mass {siege} exceeds siege_cap {cfg.siege_cap}")
        v[o] = siege / cfg.siege_cap
        v[o + 1] = state.vision_of(who) / cfg.vision_turns if cfg.vision_turns > 0 else 0.0
        v[o + 2] = 1.0 if siege > 0 else 0.0
        o += len(SIDE_NAMES)
    for who in (side, 1 - side):
        _tower_block(champ.side(who).tower_dmg, cfg, v, o)
        o += len(TOWER_NAMES)
    T = cfg.tower
    cap = t.level_cap
    tiers = t.item_tiers
    gscale = t.gold_cap[0]
    inv = None
    if cfg.continuous_items:
        from ..progression import investment
        inv = investment(cfg)
        g1, ref_s, ref_g = cfg.gold_scale, cfg.invest_obs_ref, cfg.invest_obs_gold_ref
        ap_ref = t.ap[cap][0] * inv.one
        compressed = cfg.invest_obs_encoding == "compressed"
    for who, sgn in ((side, 1), (1 - side, -1)):
        cs = champ.side(who)
        v[o + sgn * cs.pos + T] = 1.0
        o += 2 * T + 1
        x = cs.xp if cs.xp < t.xp_cap else t.xp_cap
        L = t.level_by_xp[x]
        H = t.hp_max[L]
        N = cs.items
        g = cs.gold
        v[o] = cs.hp / H
        v[o + 1] = H / t.hp_max_cap
        v[o + 2] = cs.absent / cfg.champion_absent_max
        v[o + 3] = cfg.skill_a if who == 0 else cfg.skill_b
        v[o + 4] = (L - 1) / (cap - 1) if cap > 1 else 0.0
        v[o + 5] = (x - t.xp_cum[L]) / t.xp_next[L] if L < cap else 0.0
        if inv is not None:
            if compressed:
                v[o + 6] = (N // g1) / (N // g1 + ref_s)
                v[o + 7] = (g // g1) / (g // g1 + ref_g)
            else:
                v[o + 6] = min(N // g1, ref_s) / ref_s
                v[o + 7] = min(g // g1, ref_g) / ref_g
            v[o + 8] = (N % g1) / g1
            v[o + 9] = (g % g1) / g1
            ap = inv.attack_power(L, N)
            v[o + 10] = ap / (ap + ap_ref)
        else:
            v[o + 6] = N / tiers if tiers else 0.0
            v[o + 7] = g / gscale if gscale else 0.0
            v[o + 8] = 1.0 if N < tiers and g >= t.prices[N] else 0.0
            v[o + 9] = 1.0 if N + 1 < tiers and g >= t.prices[N] + t.prices[N + 1] else 0.0
            v[o + 10] = t.ap[L][N] / t.ap_max
        o += len(CHAMP_TAIL)
    return np.asarray(v, dtype=np.float32)


# ---------------------------------------------------------------------- the constructive decoder (tests)
def decode_m4(vec, cfg: GameConfig) -> Tuple[int, State, object, Tuple[float, float]]:
    """(phase, lane State, M4State, (own skill, enemy skill)) read back from an M4 observation, in the observer's
    own perspective: the observer is side A. Exact on every reachable state (design §12.2). Under item_model
    "continuous" it is exact while invested and unspent gold are below their observation references and raises
    ValueError on a saturated entry (the value is then not recoverable)."""
    from ..champions_m4 import M4Side, M4State
    t = tables(cfg)
    vec = np.asarray(vec, dtype=np.float32)
    P = cfg.phase_period
    cells, lane_n, champ_n = _sizes(cfg)
    base, T = cfg.base, cfg.tower
    phase = int(np.argmax(vec[:P]))
    assert vec[phase] == 1.0 and vec[:P].sum() == 1.0
    o = P
    lanes = []
    for sgn in (1, -1):
        blk = vec[o:o + lane_n]
        onehot = blk[:cells]
        front = None
        if onehot.any():
            i = int(np.argmax(onehot))
            front = (sgn * (i - base), int(round(float(blk[cells]) * cfg.hp_cap)))
        rears = []
        for i in range(cells):
            rears += [sgn * (i - base)] * int(round(float(blk[cells + 1 + i]) * cfg.rear_count_max))
        lanes.append((front, rears))           # in that side's OWN coordinates
        o += lane_n
    sides = []
    for _ in range(2):
        sides.append((int(round(float(vec[o]) * cfg.siege_cap)), int(round(float(vec[o + 1]) * cfg.vision_turns))))
        o += len(SIDE_NAMES)
    taus = []
    for _ in range(2):
        taus.append(int(round(float(vec[o]) * cfg.plate_thresholds[-1])))
        o += len(TOWER_NAMES)
    champs, skills = [], []
    gscale = t.gold_cap[0]
    for sgn in (1, -1):
        cell = int(np.argmax(vec[o:o + 2 * T + 1])) - T
        o += 2 * T + 1
        L = 1 + int(round(float(vec[o + 4]) * (t.level_cap - 1)))
        hp = int(round(float(vec[o]) * t.hp_max[L]))
        absent = int(round(float(vec[o + 2]) * cfg.champion_absent_max))
        xp = t.xp_cum[L] + (int(round(float(vec[o + 5]) * t.xp_next[L])) if L < t.level_cap else 0)
        if cfg.continuous_items:
            g1 = cfg.gold_scale
            if cfg.invest_obs_encoding == "compressed":
                whole = []
                for x, ref in ((float(vec[o + 6]), cfg.invest_obs_ref), (float(vec[o + 7]), cfg.invest_obs_gold_ref)):
                    n = int(round(ref * x / (1.0 - x))) if x < 1.0 else -1
                    if not 0 <= n <= compressed_exact_max(ref):
                        raise ValueError("continuous investment: invested or unspent gold beyond the exact range "
                                         "of the compressed observation entry; not decodable")
                    whole.append(n)
                whole_s, whole_g = whole
            else:
                whole_s = int(round(float(vec[o + 6]) * cfg.invest_obs_ref))
                whole_g = int(round(float(vec[o + 7]) * cfg.invest_obs_gold_ref))
                if whole_s >= cfg.invest_obs_ref or whole_g >= cfg.invest_obs_gold_ref:
                    raise ValueError("continuous investment: invested or unspent gold at/above its observation "
                                     "reference; the observation is saturated and not decodable")
            items = whole_s * g1 + int(round(float(vec[o + 8]) * g1))
            gold = whole_g * g1 + int(round(float(vec[o + 9]) * g1))
        else:
            items = int(round(float(vec[o + 6]) * t.item_tiers)) if t.item_tiers else 0
            gold = int(round(float(vec[o + 7]) * gscale))
        champs.append((sgn * cell, hp, absent, xp, items, gold))
        skills.append(float(vec[o + 3]))
        o += len(CHAMP_TAIL)
    own = _side_record(*lanes[0], sides[0][0])
    enemy = _side_record(*lanes[1], sides[1][0])
    lane = State(*own, *enemy, sides[0][1], sides[1][1], 0, 0)
    m4 = M4State(M4Side(*champs[0], taus[0]), M4Side(*champs[1], taus[1]))
    return phase, lane, m4, (skills[0], skills[1])


def _side_record(front, rears, siege):
    """State side record (fx, fh, rears most advanced first, siege). A side without a front has no packets."""
    if front is None:
        if rears:
            raise ValueError("rear packets without a front cannot occur in a canonical State")
        return 0, 0, (), siege
    return front[0], front[1], tuple(sorted(rears, key=lambda x: -x)), siege
