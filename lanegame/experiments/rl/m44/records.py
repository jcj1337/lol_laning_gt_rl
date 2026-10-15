"""One game -> per-turn, per-side records: the decision state, the policy's probability masses, the action played and
what followed. Every strategic metric of M4.4 (metrics.py) is computed from these records and nothing else.

A game is played through the real environment (`LaneEnv`, the M4 resolver, its own chance stream) from s0 or from a
given start (an INITIALIZED scenario, labelled as such), with the stream split of `env.agents.episode_streams`
(environment chance, seat A sampling, seat B sampling) and the agent lifecycle of `env.discounted.game_agents`, so a
record of two agents equals `evaluate_discounted`'s game on the same seed turn by turn.

Fields (arrays of shape (turns, 2); column s is side s, every position in s's OWN coordinates: own tower -T = -2,
the lane cell 0, enemy tower +T = +2):

  decision state (start of the turn)
    present, en_present          the champion / the enemy champion is in lane (not absent)
    pos, epos, dist              own cell; the enemy's cell in own coordinates; |pos + enemy pos| (contact: <= 1, a
                                 champion in its own tower cell only from the same cell)
    hp_frac, ehp_frac            HP / max HP of the level
    level, elevel, invested, einvested, gold     level; invested and unspent gold (resources)
    power                        my expected hit in the enemy's HP units / the enemy's in mine (behavior.turn_features)
    edge                         zoning_natural.duel_edge (expected damage fraction exchanged; HP-aware; 0 if the enemy
                                 is away)
    own_fx, own_fh, en_fx, en_fh the fronts (cell, HP); en_fx in own coordinates; cell nan when there is no front
    contact                      both fronts alive and fighting (own_fx + enemy own_fx >= -1)
    wave_pos                     where the waves meet, own coordinates: the contact cell (own_fx - enemy own_fx) / 2;
                                 without contact the midpoint of the two fronts, the single front's cell, or nan
    s_own, s_en, wave_adv        window strengths (mechanics.window_strength, the quantity Thin / Slow Push target) and
                                 their difference (> 0: my wave is the stronger)
    own_siege, en_siege          my crashed mass under the enemy tower; the enemy's under mine (HP)
    held, held_on                FREEZE STATE (metrics §freeze): both fronts alive and in contact with the enemy front at
                                 my own -1 or deeper (I hold the enemy wave next to my tower) / the mirror image
    vision, evision              vision timers (turns left)
    plates_left, own_plates_left plates I can still take on the enemy tower / the enemy can still take on mine
    recall_legal, obj_legal, n_legal
  action
    a, wave (WAVE_ORDER index; -1 for Recall / absent), beh (B_*; -1 absent), obj (0 none / 1 Pressure / 2 Vision;
    -1 for Recall / absent), recall
  policy (when the agent exposes a distribution; nan otherwise)
    pw0..pw3 (wave masses, WAVE_ORDER, Recall excluded), pb0..pb4 (behaviour masses), po0..po2 (objective masses),
    prec (Recall), entropy (nats, over the legal codes)
  outcome of the turn
    reward, earned, farm, farm_avail, plate_gold, bounty (resources); xp, xp_avail, xp_missed (XP lost for good)
    died, recalled, away, returned, ganked, threatened (evaded an enemy Trade), pinned
    gank_p, exp_gank, gank_dmg, tower_p, exp_tower, tower_dmg, trade_taken, damage_taken (bars of max HP; trade, gank
    and tower damage before the victim's investment mitigation, damage_taken after it)
    trade_att, trade_conn, trade_dealt (bars of the victim)
    end_pos, dive (connected trade ending in the enemy tower cell), adj_trade (... ending next to it), busy (my crash
    is under the enemy tower or crashes this turn: the tower's "busy" condition of champions_m4 step 9b)
    crash (HP of my wave that crashed into the enemy tower this turn), en_crash (the enemy's into mine)
    plates (plates I broke), tower_dealt (tower damage dealt, centi-units)
    vision_set, pressure_eff, vision_eff (the objective took effect), budget, applied (wave damage HP)
    held_after, held_on_after    the freeze state of the post-turn lane
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ....champions import in_contact, objective_reachable, recall_legal
from ....champions_m4 import context
from ....config import CHAMP_ABSENT, PRESSURE, VISION, GameConfig
from ....env import LaneEnv
from ....env.agents import episode_streams, sample
from ....diagnostics.branch import Snapshot
from ....env.discounted import game_agents
from ....mechanics import start_packets, window_strength
from ....positional import dist as graph_dist, mirror_cell
from ....progression import plates_broken
from ....state import State
from ...m4.zoning_natural import duel_edge
from .actions import WAVE_INDEX, decode, full_distribution, masses

A, B = 0, 1

STATE_FIELDS = ("present", "en_present", "pos", "epos", "dist", "hp_frac", "ehp_frac", "level", "elevel", "invested",
                "einvested", "gold", "power", "edge", "own_fx", "own_fh", "en_fx", "en_fh", "contact", "wave_pos",
                "s_own", "s_en", "wave_adv", "own_siege", "en_siege", "held", "held_on", "vision", "evision",
                "plates_left", "own_plates_left", "recall_legal", "obj_legal", "n_legal")
ACTION_FIELDS = ("a", "wave", "beh", "obj", "recall")
PROB_FIELDS = ("pw0", "pw1", "pw2", "pw3", "pb0", "pb1", "pb2", "pb3", "pb4", "po0", "po1", "po2", "prec", "entropy")
OUTCOME_FIELDS = ("reward", "earned", "farm", "farm_avail", "plate_gold", "bounty", "xp", "xp_avail", "xp_missed",
                  "died", "recalled", "away", "returned", "ganked", "threatened", "pinned", "gank_p", "exp_gank",
                  "gank_dmg", "tower_p", "exp_tower", "tower_dmg", "trade_taken", "damage_taken", "trade_att",
                  "trade_conn", "trade_dealt", "end_pos", "dive", "adj_trade", "busy", "crash", "en_crash", "plates",
                  "tower_dealt", "vision_set", "pressure_eff", "vision_eff", "budget", "applied", "held_after",
                  "held_on_after")
FIELDS = STATE_FIELDS + ACTION_FIELDS + PROB_FIELDS + OUTCOME_FIELDS


@dataclass
class GameRecord:
    env: str                              # environment key (or preset name)
    names: Tuple[str, str]                # agents in seats A, B
    seed: object                          # SeedSequence entropy / spawn key or int
    start: str                            # "s0" or the scenario label
    initialized: bool                     # True when the game started from a constructed (non-s0) state
    data: Dict[str, np.ndarray] = field(default_factory=dict)
    snapshots: Optional[list] = None      # keep_snapshots: diagnostics.Snapshot at the start of every turn (not saved)

    @property
    def turns(self) -> int:
        return len(self.data["present"])

    def to_dict(self) -> dict:
        return {"env": self.env, "names": list(self.names), "seed": str(self.seed), "start": self.start,
                "initialized": self.initialized, "data": {k: v.tolist() for k, v in self.data.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "GameRecord":
        return cls(d["env"], tuple(d["names"]), d["seed"], d["start"], d["initialized"],
                   {k: np.asarray(v) for k, v in d["data"].items()})


# ---------------------------------------------------------------------------------------------- state features
def fronts(lane, s: int) -> Tuple[float, int, float, int]:
    """(own front cell, own HP, enemy front cell in s's own coordinates, enemy HP); cell nan without a front."""
    ofx, ofh = lane.side(s)[:2]
    efx, efh = lane.side(1 - s)[:2]
    return (float(ofx) if ofh > 0 else float("nan"), int(ofh), float(-efx) if efh > 0 else float("nan"), int(efh))


def in_wave_contact(lane, s: int) -> bool:
    ofx, ofh = lane.side(s)[:2]
    efx, efh = lane.side(1 - s)[:2]
    return ofh > 0 and efh > 0 and ofx + efx >= -1


def holds_freeze(lane, s: int) -> bool:
    """FREEZE STATE of side s (the freezer): both fronts alive and fighting, the enemy front at s's own -1 or deeper,
    i.e. s holds the enemy wave next to its own tower (reference §4.8; M4.3c harness `held_by` plus contact)."""
    efx, efh = lane.side(1 - s)[:2]
    return in_wave_contact(lane, s) and -efx <= -1


def wave_position(lane, s: int) -> float:
    ox, oh, ex, eh = fronts(lane, s)
    if oh > 0 and eh > 0:
        return (ox + ex) / 2.0                     # the contact cell when in contact, else the gap's midpoint
    if oh > 0:
        return ox
    if eh > 0:
        return ex
    return float("nan")


def power_ratio(ctx, m4, s: int) -> float:
    t = ctx.t
    me, en = m4.side(s), m4.side(1 - s)
    L = t.level_by_xp[min(me.xp, t.xp_cap)]
    eL = t.level_by_xp[min(en.xp, t.xp_cap)]
    da = ctx.hit(1, L, me.items) / t.hp_unit[eL]
    db = ctx.hit(1, eL, en.items) / t.hp_unit[L]
    return da / db if db > 0 else 1.0


def decision_state(cfg: GameConfig, ctx, lane, m4, k: int, s: int, legal: Sequence[int]) -> Dict[str, float]:
    t = ctx.t
    T, g = cfg.tower, cfg.gold_scale
    me, en = m4.side(s), m4.side(1 - s)
    L = t.level_by_xp[min(me.xp, t.xp_cap)]
    eL = t.level_by_xp[min(en.xp, t.xp_cap)]
    present, en_present = me.absent == 0, en.absent == 0
    P = start_packets(lane, cfg)
    s_own, s_en = float(window_strength(P[s], k, cfg)), float(window_strength(P[1 - s], k, cfg))
    ox, oh, ex, eh = fronts(lane, s)
    nplates = len(cfg.plate_thresholds)
    return {
        "present": present, "en_present": en_present, "pos": me.pos,
        # M4.6b measurement fix: the Ward node W (positional.WARD = 99) is not a lane cell, so the enemy cell is mirrored
        # with W kept as W and the distance is the movement-graph distance (|pos + enemy pos| on lane cells, as before)
        "epos": float(mirror_cell(en.pos)) if en_present else float("nan"),
        "dist": float(graph_dist(me.pos, mirror_cell(en.pos))) if (present and en_present) else float("nan"),
        "hp_frac": me.hp / t.hp_max[L], "ehp_frac": en.hp / t.hp_max[eL], "level": L, "elevel": eL,
        "invested": me.items / g, "einvested": en.items / g, "gold": me.gold / g,
        "power": power_ratio(ctx, m4, s), "edge": duel_edge(cfg, lane, m4, s) if present else 0.0,
        "own_fx": ox, "own_fh": oh, "en_fx": ex, "en_fh": eh, "contact": in_wave_contact(lane, s),
        "wave_pos": wave_position(lane, s), "s_own": s_own, "s_en": s_en, "wave_adv": s_own - s_en,
        "own_siege": lane.siege_of(s), "en_siege": lane.siege_of(1 - s),
        "held": holds_freeze(lane, s), "held_on": holds_freeze(lane, 1 - s),
        "vision": lane.vision_of(s), "evision": lane.vision_of(1 - s),
        "plates_left": nplates - plates_broken(me.tower_dmg, t), "own_plates_left": nplates - plates_broken(en.tower_dmg, t),
        "recall_legal": present and recall_legal(m4, s, cfg), "obj_legal": present and objective_reachable(lane, m4, s, cfg),
        "n_legal": len(legal),
    }


def action_fields(a: int) -> Dict[str, float]:
    d = decode(a)
    present_wave = not d.absent and not d.recall
    return {"a": a, "wave": d.wave_index if present_wave else -1, "beh": d.behavior if not d.absent else -1,
            "obj": d.objective if present_wave else -1, "recall": d.recall}


def prob_fields(p_full: Optional[np.ndarray]) -> Dict[str, float]:
    if p_full is None:
        return {k: float("nan") for k in PROB_FIELDS}
    m = masses(p_full)
    q = p_full[p_full > 0]
    out = {f"pw{i}": float(m["wave"][i]) for i in range(4)}
    out.update({f"pb{i}": float(m["beh"][i]) for i in range(5)})
    out.update({f"po{i}": float(m["obj"][i]) for i in range(3)})
    out["prec"] = float(m["recall"])
    out["entropy"] = float(-(q * np.log(q)).sum())
    return out


def outcome(cfg: GameConfig, ctx, info: dict, lane_after, s: int) -> Dict[str, float]:
    t = ctx.t
    T, sc = cfg.tower, cfg.gold_scale
    o = 1 - s
    lv, elv = info["levels_before"][s], info["levels_before"][o]
    H, u, eH = t.hp_max[lv], t.hp_unit[lv], t.hp_max[elv]
    away = bool(info["away"][s])
    end = int(info["champ_end"][s])
    conn = bool(info["trade_connected"][s])
    lane_before_siege = State.from_list(info["state_before"]).siege_of(s)
    eff = info["effective_objectives"][s] if info.get("effective_objectives") is not None else 0
    xp_avail = float(cfg.xp_per_hp * sum(info["kills"][s].values()))
    return {
        "reward": info["reward"] if s == A else -info["reward"], "earned": info["earned_units"][s] / sc,
        "farm": info["farm_units"][s] / sc, "farm_avail": float(info["farm_available"][s]),
        "plate_gold": info["plate_units"][s] / sc, "bounty": info["bounty_units"][s] / sc,
        "xp": float(info["xp_gain"][s]), "xp_avail": xp_avail,
        "xp_missed": xp_avail - float(info["xp_gain"][s]) - float(info["xp_discarded"][s]),
        "died": bool(info["died"][s]), "recalled": bool(info["recalled"][s]), "away": away,
        "returned": bool(info["returned"][s]), "ganked": bool(info["ganked"][s]),
        "threatened": bool(info["threatened"][s]), "pinned": bool(info["pinned"][s]) if info.get("pinned") else False,
        "gank_p": float(info["gank_p"][s]),
        "exp_gank": float(info["gank_p"][s]) * cfg.gank_damage_trials * float(info["gank_damage_p"][s]) * u / H,
        "gank_dmg": info["gank_damage"][s] / H, "tower_p": float(info["tower_p"][s]),
        "exp_tower": float(info["tower_p"][s]) * cfg.tower_shot_damage * u / H, "tower_dmg": info["tower_damage"][s] / H,
        "trade_taken": info["trade_damage"][o] / H, "damage_taken": info["damage_taken"][s] / H,
        "trade_att": bool(info["trade_attempted"][s]), "trade_conn": conn, "trade_dealt": info["trade_damage"][s] / eH,
        "end_pos": float(end) if not away else float("nan"), "dive": conn and end == T,
        "adj_trade": conn and end == T - 1, "busy": lane_before_siege > 0 or info["crash_hp"][s] > 0,
        "crash": int(info["crash_hp"][s]), "en_crash": int(info["crash_hp"][o]), "plates": int(info["plates_broken"][s]),
        "tower_dealt": int(info["tower_damage_dealt"][s]), "vision_set": bool(info["vision_set"][s]),
        "pressure_eff": eff == PRESSURE, "vision_eff": eff == VISION,
        "budget": int(info["wave_budget"][s]) if info.get("wave_budget") is not None else 0,
        "applied": int(info["events"]["champion_hp"][o]) if "champion_hp" in info["events"] else 0,
        "held_after": holds_freeze(lane_after, s), "held_on_after": holds_freeze(lane_after, o),
    }


# ---------------------------------------------------------------------------------------------- one game
def _choose(agent, obs, rng, probs: bool):
    """(action, full distribution or None). A NetAgent's draw is made from the distribution computed once, with the
    same inverse-CDF rule and the same single uniform as NetAgent.act (identical actions); other agents act through
    their own `act` and report `distribution` when they have one."""
    from ....rl import NetAgent
    if obs.legal == (CHAMP_ABSENT,):
        a = agent.act(obs, rng)
        return a, (None if not probs else _absent_full(obs))
    if isinstance(agent, NetAgent) and not agent.greedy:
        p_full = agent.probabilities(obs)
        p = p_full[list(obs.legal)]
        p = p / p.sum()
        return obs.legal[sample(rng, p)], p_full
    a = agent.act(obs, rng)
    if not probs or not callable(getattr(agent, "distribution", None)):
        return a, None
    return a, full_distribution(agent, obs)


def _absent_full(obs) -> np.ndarray:
    p = np.zeros(len(obs.mask))
    p[CHAMP_ABSENT] = 1.0
    return p


def play_record(cfg: GameConfig, agents: Sequence, seed, turns: int, start=None, start_label: str = "s0",
                env_label: Optional[str] = None, probs: bool = True, cache_size: int = 20_000,
                keep_snapshots: bool = False, viewer_seed: bool = False) -> GameRecord:
    """Play one game of `turns` turns (from s0, or from start = (lane, M4State, k0)) and record it (module docstring).
    `seed` is split as in `play_episode` / `evaluate_discounted`; agents get reset() / notify() as there. With
    `keep_snapshots` the restorable start-of-turn state of every turn is kept (`GameRecord.snapshots`), for the
    counterfactual state sampling. With `viewer_seed` the integer `seed` follows the viewer's recording convention
    instead (environment chance seed = `seed`, seat streams = SeedSequence(seed).spawn(2); the convention of
    `m43b.trajectories.record_game`), so the recorded game replays exactly in the debugger."""
    from .common import env_key
    ctx = context(cfg)
    env = LaneEnv(cfg, max_turns=turns, cache_size=cache_size)
    if viewer_seed:
        env_ss = int(seed)
        rng_a, rng_b = (np.random.default_rng(x) for x in np.random.SeedSequence(int(seed)).spawn(2))
    else:
        _, env_ss, rng_a, rng_b = episode_streams(seed)
    rngs = (rng_a, rng_b)
    ag = game_agents(tuple(agents))
    if start is None:
        obs = env.reset(seed=env_ss)
    else:
        lane, m4, k0 = start
        obs = env.reset(seed=env_ss, root=lane, k0=k0, champ=m4)
    cols: Dict[str, List[list]] = {k: [] for k in FIELDS}
    snaps = [] if keep_snapshots else None
    skills = (cfg.skill_a, cfg.skill_b)
    while True:
        row = [{}, {}]
        acts = [0, 0]
        for s in (A, B):
            o = obs[s]
            row[s].update(decision_state(cfg, ctx, o.state, o.champ, o.k, s, o.legal))
            a, p = _choose(ag[s], o, rngs[s], probs)
            acts[s] = int(a)
            row[s].update(action_fields(a))
            row[s].update(prob_fields(p if (p is not None and o.legal != (CHAMP_ABSENT,)) else None))
        r = env.step(acts[0], acts[1])
        if snaps is not None:
            snaps.append(Snapshot.from_info(r.info, skills))
        for s in (A, B):
            nf = getattr(ag[s], "notify", None)
            if callable(nf):
                nf(r.info, s)
        for s in (A, B):
            row[s].update(outcome(cfg, ctx, r.info, env.state, s))
        for k in FIELDS:
            cols[k].append([row[0][k], row[1][k]])
        if r.done:
            break
        obs = r.obs
    data = {k: np.asarray(v, dtype=float) for k, v in cols.items()}
    names = tuple(getattr(x, "name", "agent") for x in agents)
    return GameRecord(env_label or env_key(cfg) or "custom", names, getattr(seed, "spawn_key", seed), start_label,
                      start is not None, data, snaps)


def play_records(cfg: GameConfig, x, y, games: int, turns: int, seed: int, start=None, start_label: str = "s0",
                 both_seats: bool = True, env_label: Optional[str] = None) -> List[GameRecord]:
    """`games` seeds; x in seat A against y, and (both_seats) x in seat B on the same seeds (paired)."""
    out = []
    children = np.random.SeedSequence(seed).spawn(games)
    for ss in children:
        out.append(play_record(cfg, (x, y), np.random.SeedSequence(ss.entropy, spawn_key=ss.spawn_key + (0,)), turns,
                               start, start_label, env_label))
        if both_seats and x is not y:
            out.append(play_record(cfg, (y, x), np.random.SeedSequence(ss.entropy, spawn_key=ss.spawn_key + (0,)),
                                   turns, start, start_label, env_label))
    return out


def focal_sides(rec: GameRecord, focal: str) -> List[int]:
    """The columns of a record that belong to the agent named `focal` (both in self-play)."""
    return [s for s in (A, B) if rec.names[s] == focal]


__all__ = ["GameRecord", "FIELDS", "STATE_FIELDS", "ACTION_FIELDS", "PROB_FIELDS", "OUTCOME_FIELDS", "fronts",
           "in_wave_contact", "holds_freeze", "wave_position", "power_ratio", "decision_state", "action_fields",
           "prob_fields", "outcome", "play_record", "play_records", "focal_sides"]
