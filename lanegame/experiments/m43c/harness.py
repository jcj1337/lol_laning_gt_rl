"""Paired-seed simulation harness of the M4.3c stabilization study (docs/m4/m4_3/m4_3c_strategic_economics_calibration.md).

Every run goes through the real environment (`LaneEnv`, the M4 resolver, its own chance stream), from a named start
state, with scripted agents (`strategies.StrategyAgent`). Nothing here changes the game.

Starts (`start`): the deterministic states of the M4.2e-B harness generated under `m4_lasthit()` exactly as in the
four-action redesign study (`zoning_natural.starts`; the lane physics is the same in every natural preset), optionally
with B's front HP replaced, both champions put at the first XP of a level, and a power gap (`viewer.session.gapped`:
A's investment / level lead, or B's for "b_large"). The base states:

    neutral   turn 9: both 8-HP fronts and both champions at the lane cell
    freeze    turn 51: B's 5-HP front held at A's -1 (a fresh B wave behind it, 6 HP of B's crashed under A's tower),
              A's 8-HP front in A's tower cell, A at -2, B at its +1

Chance (`chance`): "env" draws the environment's ten uniforms per turn from the seed (trades, tower shots, ganks);
"none" replaces the environment's generator by one that returns 1.0 (no chance event ever happens: a deterministic
mechanical run, the u = None convention of the earlier screens, applied inside the real environment).

Seeds: episode e of a cell uses SeedSequence(root, spawn_key=(e,)), the same for every cell of a comparison (common
random numbers: two cells that differ only by one side's strategy see the same chance stream turn by turn until their
states diverge).

Per-turn ledger (`turn_row`) and window statistics (`window`), per side s (0 = A, 1 = B):

    gold       earned resources (farm + plate gold + bounty) and its parts; farm_avail = the farm at full collection
               (0.125 per enemy minion HP that died on s's ledger, whatever s's position)
    XP         credited XP; xp_avail = XP of every enemy minion HP that died this turn (what a present champion within
               range of every cell would get); xp_missed = xp_avail - credited - discarded: XP LOST FOR GOOD (the
               minions died while s was away or out of range); xp_pending_change = the change of the enemy minion HP
               still alive (lane + crashed under s's tower) over the window: XP DELAYED, not lost (a freeze withholds
               it; it is credited when those minions die)
    combat     HP lost (bars of the champion's max HP), trade attempts / connections / damage dealt (centi-HP before the
               victim's mitigation), ganks, tower shots taken, deaths, Recalls, turns away
    power      level, invested resources, attack power AP* and wave capacity C at the end; invested during the window
    waves      crashes (count, HP, first turn), tower damage dealt, plates, wave budget vs damage applied
    freeze     held_by[s] = turns the enemy front stood at s's own -1 or deeper with both fronts alive
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from ...champions_m4 import M4State, context
from ...config import GameConfig
from ...env import LaneEnv
from .common import mean_ci

A, B = 0, 1
LAMBDA_XP = (0.0, 1 / 16, 1 / 8, 1 / 4)
GAMMA = 0.99


class NoChance:
    """A stand-in for the environment's chance generator: every uniform is 1.0, so no chance event happens."""

    def random(self, n=None):
        return 1.0 if n is None else np.ones(n)


# ---------------------------------------------------------------------------------------------- starts
@lru_cache(maxsize=2)
def _base_starts() -> dict:
    from ..m4.zoning_natural import starts
    return starts(GameConfig.m4_lasthit())


def start(name: str, cfg: GameConfig, gap: str = "equal", level: Optional[int] = None, hb: Optional[int] = None,
          ha: Optional[int] = None) -> Tuple[object, M4State, int]:
    """(lane, champions, k) of a base start (module docstring) with B's / A's front HP set to hb / ha, both champions
    at `level`, and the power gap applied last."""
    from ...viewer.session import at_level, gapped
    lane, m4, k = _base_starts()[name]
    if hb is not None:
        lane = lane._replace(bh=hb)
    if ha is not None:
        lane = lane._replace(ah=ha)
    if level is not None:
        m4 = at_level(m4, cfg, level)
    return lane, gapped(m4, cfg, gap), k


# ---------------------------------------------------------------------------------------------- one game
def _alive_enemy(lane, s: int, cfg: GameConfig) -> int:
    """HP of side s's ENEMY minions still alive: the enemy's lane packets and its crashed mass under s's tower."""
    e = 1 - s
    fx, fh, rears, sg = lane.side(e)[:4]
    return max(fh, 0) + cfg.wave_hp * len(rears) + sg


def turn_row(info: dict, lane_after, cfg: GameConfig, ctx) -> dict:
    """The per-turn ledger of the module docstring from one StepResult.info."""
    sc, t = cfg.gold_scale, ctx.t
    after = info["champ_after"]
    row = {}
    for key in ("earned_units", "farm_units", "plate_units", "bounty_units"):
        row[key[:-6]] = [info[key][s] / sc for s in (A, B)]
    row["farm_avail"] = list(info["farm_available"])
    row["xp"] = list(info["xp_gain"])
    row["xp_disc"] = list(info["xp_discarded"])
    row["xp_avail"] = [cfg.xp_per_hp * sum(info["kills"][s].values()) for s in (A, B)]
    lv0 = info["levels_before"]
    row["hp_lost"] = [info["damage_taken"][s] / t.hp_max[lv0[s]] for s in (A, B)]
    # incoming damage by source, in bars of the victim's max HP, BEFORE the victim's durability mitigation (the
    # mitigation divides the turn's sum once, so per-source shares are exact only up to that common factor)
    row["dmg_trade"] = [info["trade_damage"][1 - s] / t.hp_max[lv0[s]] for s in (A, B)]
    row["dmg_gank"] = [info["gank_damage"][s] / t.hp_max[lv0[s]] for s in (A, B)]
    row["dmg_tower"] = [info["tower_damage"][s] / t.hp_max[lv0[s]] for s in (A, B)]
    for key, src in (("trade_att", "trade_attempted"), ("trade_conn", "trade_connected"), ("ganked", "ganked"),
                     ("died", "died"), ("recalled", "recalled"), ("away", "away"), ("returned", "returned")):
        row[key] = [int(bool(info[src][s])) for s in (A, B)]
    row["trade_dealt"] = list(info["trade_damage"])
    row["shot"] = [int(info["tower_damage"][s] > 0) for s in (A, B)]
    row["invested"] = [info["invested"][s] / sc for s in (A, B)]
    row["crash"] = list(info["crash_hp"])
    row["spawned_hp"] = list(info["events"]["spawned_hp"])          # side s's own minions (conservation checks)
    row["overflow_hp"] = list(info["events"]["overflow_hp"])
    row["tower"] = list(info["tower_damage_dealt"])
    row["plates_n"] = list(info["plates_broken"])
    row["wave"] = list(info["waves"])
    row["budget"] = list(info["wave_budget"]) if info.get("wave_budget") is not None else [0, 0]
    row["applied"] = [info["events"]["champion_hp"][1 - s] for s in (A, B)] if "champion_hp" in info["events"] \
        else [0, 0]
    row["level"] = list(info["levels"])
    row["hp"] = [after[s][1] / t.hp_max[info["levels"][s]] for s in (A, B)]
    l2 = lane_after
    row["held_by"] = [int(l2.bh > 0 and l2.ah > 0 and l2.bx >= 1), int(l2.ah > 0 and l2.bh > 0 and l2.ax >= 1)]
    row["contact"] = (l2.ax if (l2.ah > 0 and l2.bh > 0 and l2.ax + l2.bx >= -1) else None)
    row["pos"] = [info["champ_end"][0], -info["champ_end"][1]]          # global cells
    return row


def play(cfg: GameConfig, st, agents: Sequence, seed, turns: int, chance: str = "env",
         on_turn: Optional[Callable] = None) -> dict:
    """One game of `turns` turns from start st = (lane, m4, k) with two environment agents (each may define
    reset() / notify(info, side)); returns {"rows": per-turn ledgers, "start": ..., "end": ...}."""
    lane, m4, k = st
    ctx = context(cfg)
    env = LaneEnv(cfg, max_turns=turns, cache_size=5_000)
    ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    obs = env.reset(seed=ss, root=lane, k0=k, champ=m4)
    if chance == "none":
        env._rng = NoChance()
    elif chance != "env":
        raise ValueError(chance)
    for ag in agents:
        if callable(getattr(ag, "reset", None)):
            ag.reset()
    rows = []
    alive0 = [_alive_enemy(env.state, s, cfg) for s in (A, B)]
    while True:
        a, b = agents[0].act(obs[0]), agents[1].act(obs[1])
        r = env.step(a, b)
        for side, ag in enumerate(agents):
            if callable(getattr(ag, "notify", None)):
                ag.notify(r.info, side)
        rows.append(turn_row(r.info, env.state, cfg, ctx))
        if on_turn is not None:
            on_turn(len(rows) - 1, r, env)
        if r.done:
            break
        obs = r.obs
    alive1 = [_alive_enemy(env.state, s, cfg) for s in (A, B)]
    return {"rows": rows, "alive0": alive0, "alive1": alive1, "end_champ": env.champ, "end_lane": env.state,
            "k0": k}


# ---------------------------------------------------------------------------------------------- window statistics
def _longest(flags) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def window(game: dict, cfg: GameConfig, n: Optional[int] = None) -> dict:
    """Window statistics of the first n turns of one game (all of it by default). The end-of-game power numbers
    (invested_end, ap_end, duel_ratio_end, ...) belong to the whole-game window (n None) only, so that every
    sub-window has the same keys whether or not a game terminated (terminal_rule) before n turns."""
    whole = n is None
    rows = game["rows"][:n] if n else game["rows"]
    n = len(rows)
    ctx = context(cfg)
    S = lambda key, s: float(sum(r[key][s] for r in rows))
    out = {"turns": n}
    for key in ("earned", "farm", "plate", "bounty", "farm_avail", "xp", "xp_disc", "xp_avail", "hp_lost", "dmg_trade",
                "dmg_gank", "dmg_tower", "trade_att", "trade_conn", "trade_dealt", "ganked", "shot", "died",
                "recalled", "away", "invested", "tower", "plates_n", "budget", "applied", "held_by"):
        out[key] = [S(key, s) for s in (A, B)]
    out["xp_missed"] = [out["xp_avail"][s] - out["xp"][s] - out["xp_disc"][s] for s in (A, B)]
    out["crashes"] = [sum(1 for r in rows if r["crash"][s] > 0) for s in (A, B)]
    out["crash_hp"] = [S("crash", s) for s in (A, B)]
    out["first_crash"] = [next((i + 1 for i, r in enumerate(rows) if r["crash"][s] > 0), None) for s in (A, B)]
    out["longest_hold"] = [_longest(r["held_by"][s] for r in rows) for s in (A, B)]
    contacts = [r["contact"] for r in rows if r["contact"] is not None]
    out["contact_mean"] = float(np.mean(contacts)) if contacts else None
    out["level_end"] = list(rows[-1]["level"])
    out["hp_end"] = list(rows[-1]["hp"])
    # discounted gold and XP channels (the game's gamma; turn 0 of the window undiscounted)
    w = GAMMA ** np.arange(n)
    g = np.array([r["earned"][A] - r["earned"][B] for r in rows])
    x = np.array([r["xp"][A] - r["xp"][B] for r in rows])
    out["G"], out["X"] = float(w @ g), float(w @ x)
    if whole:
        m4 = game["end_champ"]
        t = ctx.t
        lv = [t.level_by_xp[min(m4.side(s).xp, t.xp_cap)] for s in (A, B)]
        items = [m4.side(s).items for s in (A, B)]
        out["invested_end"] = [items[s] / cfg.gold_scale for s in (A, B)]
        out["unspent_end"] = [m4.side(s).gold / cfg.gold_scale for s in (A, B)]
        ap = [ctx.inv.attack_power(lv[s], items[s]) / (10_000 * ctx.inv.one) for s in (A, B)]
        out["ap_end"] = ap
        out["capacity_end"] = [ctx.capacity(lv[s], items[s]) for s in (A, B)]
        hit = [ctx.hit(1, lv[s], items[s]) for s in (A, B)]
        mit = [ctx.mitigate(10 ** 6, items[s]) / 10 ** 6 for s in (A, B)]
        # A's expected hit on B in B's bars over B's on A in A's (the duel ratio, durability included)
        out["duel_ratio_end"] = (hit[A] * mit[B] / t.hp_max[lv[B]]) / (hit[B] * mit[A] / t.hp_max[lv[A]])
        out["xp_pending_change"] = [game["alive1"][s] * cfg.xp_per_hp - game["alive0"][s] * cfg.xp_per_hp
                                    for s in (A, B)]
        out["absent_end"] = [m4.side(s).absent for s in (A, B)]
    return out


def diff(w: dict, key: str, s: int = A):
    """Side s's value minus the other side's (key a per-side list)."""
    return w[key][s] - w[key][1 - s]


# ---------------------------------------------------------------------------------------------- paired cells
def seeds(root: int, n: int) -> List[np.random.SeedSequence]:
    return [np.random.SeedSequence(root, spawn_key=(e,)) for e in range(n)]


def cell(cfg: GameConfig, st, make_agents: Callable, root: int, n: int, turns: int, chance: str = "env",
         windows: Sequence[int] = ()) -> List[dict]:
    """n paired games of one cell; per game the window statistics over `turns` (and over each of `windows`)."""
    out = []
    for ss in seeds(root, n):
        g = play(cfg, st, make_agents(), ss, turns, chance)
        rec = {"all": window(g, cfg)}
        for wn in windows:
            rec[str(wn)] = window(g, cfg, wn)
        out.append(rec)
    return out


def agg(games: List[dict], key: str = "all") -> dict:
    """Mean and 95% interval over games of every numeric statistic (per side for lists) plus the key differences
    (A minus B)."""
    ws = [g[key] for g in games]
    out = {}
    for k, v in ws[0].items():
        if isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) or x is None for x in v):
            for s, side in ((A, "a"), (B, "b")):
                vals = [w[k][s] for w in ws if w[k][s] is not None]
                if vals:
                    out[f"{k}_{side}"] = mean_ci(vals)
            vals = [w[k][A] - w[k][B] for w in ws if w[k][A] is not None and w[k][B] is not None]
            if vals:
                out[f"{k}_diff"] = mean_ci(vals)
        elif isinstance(v, (int, float)):
            vals = [w[k] for w in ws if w[k] is not None]
            if vals:
                out[k] = mean_ci(vals)
    for lam in LAMBDA_XP:
        out[f"value_l{lam:g}"] = mean_ci([w["earned"][A] - w["earned"][B] + lam * (w["xp"][A] - w["xp"][B])
                                          for w in ws])
        out[f"J_l{lam:g}"] = mean_ci([w["G"] + lam * w["X"] for w in ws])
    out["n"] = len(ws)
    return out


def paired_diff(ga: List[dict], gb: List[dict], f: Callable, key: str = "all") -> Tuple[float, float]:
    """Mean and 95% interval of f(window of game e in cell a) - f(the same in cell b) over the paired games."""
    return mean_ci([f(x[key]) - f(y[key]) for x, y in zip(ga, gb)])


__all__ = ["NoChance", "start", "play", "turn_row", "window", "diff", "seeds", "cell", "agg", "paired_diff",
           "LAMBDA_XP", "GAMMA"]
