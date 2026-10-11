"""Strategic action audit (diagnostics milestone, Phase A): for every decision of a policy, what was legal, what
the policy's distribution put on each action category, what it chose, in which state, and what followed.

`audit_games(cfg, x, y, episodes, seed)` plays the head-to-head protocol of `rl.evaluate.head_to_head` (x in seat A
on `episodes` seeds, then x in seat B on the same seeds) with the exact sampling of `play_episode` (each agent's
`act` with its own stream), and records x's side of every turn as one row of a column table (`Turns`). Present
turns carry x's full action distribution (`distribution(obs)`, queried without touching the sampling streams) and
the opponent's probability of Trade when the opponent exposes a distribution.

Categories (champion mode): behaviors Hold / Yield / Trade / Safe / Recall, waves Slow Push / Shove / Freeze,
siege objectives Pressure / Vision. Hold, Yield, Trade and Safe are legal on every present turn; Recall when the
champion stands at or behind the recall line; Freeze when the enemy front is at or behind own -1; an objective
when the side's crashed mass is under the enemy tower and its champion can reach that tower this turn.

All realized quantities are from the focal player's side: reward = its zero-sum reward (farm + plates + bounty
differences), `ret_to_go` = its return from this turn to the end of the game.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..champions import objective_reachable, recall_legal
from ..config import (B_HOLD, B_RECALL, B_SAFE, B_TRADE, B_YIELD, CHAMP_ABSENT, F, PRESSURE, SH, SP, VISION,
                      GameConfig, champ_behavior, champ_objective, champ_wave)
from ..env import LaneEnv
from ..env.agents import episode_streams
from ..mechanics import wave_actions

BEHAVIORS = ("Hold", "Yield", "Trade", "Safe", "Recall")
WAVES = ("SP", "SH", "F")
OBJECTIVES = ("none", "Pressure", "Vision")
MODES = ("farm", "trading", "retreat", "safe", "channel", "away")


def category_probabilities(p_full: np.ndarray) -> Dict[str, np.ndarray]:
    """Probability mass per behavior, wave and objective from a full-length (46) distribution."""
    beh = np.zeros(5)
    wave = np.zeros(3)
    obj = np.zeros(3)
    for a in np.flatnonzero(p_full > 0):
        if a == CHAMP_ABSENT:
            continue
        beh[champ_behavior(a)] += p_full[a]
        wave[champ_wave(a)] += p_full[a]
        obj[champ_objective(a)] += p_full[a]
    return {"beh": beh, "wave": wave, "obj": obj}


def full_distribution(agent, obs, n_actions: int) -> np.ndarray:
    p = np.zeros(n_actions)
    p[list(obs.legal)] = np.asarray(agent.distribution(obs), dtype=float)
    return p


class Turns:
    """Column table of audited turns (dict of equal-length numpy arrays)."""

    def __init__(self, cols: Dict[str, np.ndarray]):
        self.cols = cols

    def __len__(self) -> int:
        return len(next(iter(self.cols.values()))) if self.cols else 0

    def __getitem__(self, k: str) -> np.ndarray:
        return self.cols[k]

    def where(self, mask: np.ndarray) -> "Turns":
        return Turns({k: v[mask] for k, v in self.cols.items()})

    @classmethod
    def concat(cls, parts: List["Turns"]) -> "Turns":
        parts = [p for p in parts if len(p)]
        return cls({k: np.concatenate([p.cols[k] for p in parts]) for k in parts[0].cols})


def _row(env: LaneEnv, ob, opp_ob, x, y, seat: int, ep: int, n_act: int) -> dict:
    cfg = env.cfg
    lane, champ, k = ob.state, ob.champ, ob.k
    me, en = champ.side(seat), champ.side(1 - seat)
    own, enemy = lane.side(seat), lane.side(1 - seat)
    present = me.absent == 0
    row = {"ep": ep, "seat": seat, "k": k, "present": present, "hp": me.hp, "pos": me.pos, "absent": me.absent,
           "ehp": en.hp, "epos": en.pos, "eabsent": en.absent,
           "front_x": own[0] if own[1] > 0 else -99, "front_hp": own[1],
           "efront_x": enemy[0] if enemy[1] > 0 else -99, "efront_hp": enemy[1],
           "siege": own[3], "esiege": enemy[3], "vision": own[4], "evision": enemy[4], "plates_pending": own[5],
           "skill": cfg.skill_a if seat == 0 else cfg.skill_b, "eskill": cfg.skill_b if seat == 0 else cfg.skill_a,
           "n_legal": len(ob.legal),
           "legal_recall": present and recall_legal(champ, seat, cfg),
           "legal_freeze": present and F in wave_actions(lane, seat, cfg),
           "legal_objective": present and objective_reachable(lane, champ, seat, cfg)}
    if present:
        p = full_distribution(x, ob, n_act)
        cp = category_probabilities(p)
        for i, b in enumerate(BEHAVIORS):
            row[f"p_{b}"] = cp["beh"][i]
        for i, w in enumerate(WAVES):
            row[f"p_{w}"] = cp["wave"][i]
        for i, o in enumerate(OBJECTIVES):
            row[f"p_obj_{o}"] = cp["obj"][i]
        row["entropy"] = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    else:
        for b in BEHAVIORS:
            row[f"p_{b}"] = 0.0
        for w in WAVES:
            row[f"p_{w}"] = 0.0
        for o in OBJECTIVES:
            row[f"p_obj_{o}"] = 0.0
        row["entropy"] = 0.0
    if en.absent == 0 and callable(getattr(y, "distribution", None)):
        q = full_distribution(y, opp_ob, n_act)
        row["opp_p_trade"] = float(category_probabilities(q)["beh"][B_TRADE])
    else:
        row["opp_p_trade"] = np.nan
    return row


def audit_games(cfg: GameConfig, x, y, episodes: int, seed: int, skill: Optional[tuple] = None) -> Turns:
    """Audit x's turns against y over both seats (`head_to_head` protocol: the same `episodes` seeds per seat).
    `skill = (m_x, m_y)` assigns skills to the players (not the seats)."""
    rows: List[dict] = []
    for seat in (0, 1):
        c = cfg
        if skill is not None:
            c = cfg.with_(skill_a=skill[0], skill_b=skill[1]) if seat == 0 else \
                cfg.with_(skill_a=skill[1], skill_b=skill[0])
        env = LaneEnv(c, cache_size=20_000)
        n_act = env.n_actions
        children = np.random.SeedSequence(seed).spawn(episodes)
        for ep, ss in enumerate(children):
            _, env_ss, rng_a, rng_b = episode_streams(ss)
            agents = (x, y) if seat == 0 else (y, x)
            rngs = (rng_a, rng_b)
            obs = env.reset(seed=env_ss)
            ep_rows = []
            while True:
                row = _row(env, obs[seat], obs[1 - seat], x, y, seat, ep, n_act)
                acts = (agents[0].act(obs[0], rngs[0]), agents[1].act(obs[1], rngs[1]))
                r = env.step(*acts)
                info = r.info
                a = acts[seat]
                o = 1 - seat
                row.update({
                    "a": a, "beh": int(champ_behavior(a)) if a != CHAMP_ABSENT else 5,
                    "wave": int(champ_wave(a)) if a != CHAMP_ABSENT else -1,
                    "obj": int(champ_objective(a)) if a != CHAMP_ABSENT else 0,
                    "opp_beh": info["behaviors"][o],
                    "reward": r.rewards[seat], "farm": info["farm"][seat], "efarm": info["farm"][o],
                    "avail": info["farm_available"][seat], "eavail": info["farm_available"][o],
                    "plates": info["plates"][seat], "eplates": info["plates"][o],
                    "bounty": info["bounty"][seat], "ebounty": info["bounty"][o],
                    "dmg_taken": info["damage_taken"][seat], "dmg_dealt": info["trade_damage"][seat],
                    "gank_dmg": info["gank_damage"][seat], "ganked": bool(info["ganked"][seat]),
                    "gank_p": info["gank_p"][seat], "tower_dmg": info["tower_damage"][seat],
                    "died": bool(info["died"][seat]), "edied": bool(info["died"][o]),
                    "recalled": bool(info["recalled"][seat]), "interrupted": bool(info["recall_interrupted"][seat]),
                    "connected": bool(info["trade_connected"][seat]), "econnected": bool(info["trade_connected"][o]),
                    "threatened": bool(info["threatened"][seat]), "mode": MODES.index(info["modes"][seat]),
                    "end_pos": info["champ_end"][seat], "farm_spot": info["farm_spot"][seat],
                    "exposure": info["exposure"][seat], "crash": info["crash_hp"][seat] > 0,
                    "ecrash": info["crash_hp"][o] > 0})
                ep_rows.append(row)
                if r.done:
                    break
                obs = r.obs
            g = 0.0
            for rw in reversed(ep_rows):                       # return to go and short-horizon outcomes
                g += rw["reward"]
                rw["ret_to_go"] = g
            n = len(ep_rows)
            for i, rw in enumerate(ep_rows):
                nxt = ep_rows[i:i + 4]
                rw["died_next4"] = any(z["died"] for z in nxt)
                rw["reward_next4"] = float(sum(z["reward"] for z in nxt))
                rw["turns_left"] = n - i
            rows.extend(ep_rows)
    cols = {k: np.array([rw[k] for rw in rows]) for k in rows[0]}
    return Turns(cols)


# ---------------------------------------------------------------------------------------------- summaries
def legality_table(t: Turns) -> Dict[str, dict]:
    """Per action category: share of present turns on which it is legal, mean policy probability when legal,
    realized selection frequency when legal (Phase A.1, items 1, 3, 4)."""
    pres = t.where(t["present"])
    out = {}
    always = np.ones(len(pres), dtype=bool)
    legal = {"Hold": always, "Yield": always, "Trade": always, "Safe": always, "Recall": pres["legal_recall"],
             "SP": always, "SH": always, "F": pres["legal_freeze"], "Pressure": pres["legal_objective"],
             "Vision": pres["legal_objective"]}
    chosen = {"Hold": pres["beh"] == B_HOLD, "Yield": pres["beh"] == B_YIELD, "Trade": pres["beh"] == B_TRADE,
              "Safe": pres["beh"] == B_SAFE, "Recall": pres["beh"] == B_RECALL, "SP": pres["wave"] == SP,
              "SH": pres["wave"] == SH, "F": pres["wave"] == F, "Pressure": pres["obj"] == PRESSURE,
              "Vision": pres["obj"] == VISION}
    prob = {"Hold": pres["p_Hold"], "Yield": pres["p_Yield"], "Trade": pres["p_Trade"], "Safe": pres["p_Safe"],
            "Recall": pres["p_Recall"], "SP": pres["p_SP"], "SH": pres["p_SH"], "F": pres["p_F"],
            "Pressure": pres["p_obj_Pressure"], "Vision": pres["p_obj_Vision"]}
    n = max(len(pres), 1)
    for c in legal:
        lg = legal[c]
        nl = int(lg.sum())
        out[c] = {"legal_share": nl / n, "n_legal": nl, "selected_share": float(chosen[c].mean()) if n else 0.0,
                  "p_when_legal": float(prob[c][lg].mean()) if nl else float("nan"),
                  "selected_when_legal": float(chosen[c][lg].mean()) if nl else float("nan")}
    return out


def _share(mask: np.ndarray) -> float:
    return float(mask.mean()) if len(mask) else float("nan")


def _mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if len(x) else float("nan")


def opportunity_table(t: Turns, hp_low: int = 2) -> Dict[str, dict]:
    """Strategically relevant situations (Phase A.1, item 2) and what the policy puts on the matching actions."""
    p = t.where(t["present"])
    n = max(len(p), 1)
    enemy_here = p["eabsent"] == 0
    opp_trade = np.nan_to_num(p["opp_p_trade"])
    situations = {
        "recall legal and HP <= 2": (p["legal_recall"] & (p["hp"] <= hp_low), ["Recall"]),
        "recall legal and HP <= 3": (p["legal_recall"] & (p["hp"] <= hp_low + 1), ["Recall"]),
        "HP <= 2 at or past the lane (Recall illegal)": ((p["hp"] <= hp_low) & ~p["legal_recall"],
                                                         ["Safe", "Yield", "Trade", "Hold"]),
        "HP <= 2, enemy present": ((p["hp"] <= hp_low) & enemy_here, ["Safe", "Yield", "Trade", "Hold", "Recall"]),
        "Freeze legal": (p["legal_freeze"], ["F", "SH"]),
        "siege objective legal": (p["legal_objective"], ["Pressure", "Vision"]),
        "enemy champion away (punish window)": (p["eabsent"] > 0, ["SH", "Hold", "Trade"]),
        "own front stacked (front HP > one wave)": (p["front_hp"] > 4, ["SH"]),
        "opponent P(Trade) >= 0.5": (opp_trade >= 0.5, ["Yield", "Safe", "Trade", "Hold"]),
        "HP <= 2 and opponent P(Trade) >= 0.5": ((p["hp"] <= hp_low) & (opp_trade >= 0.5),
                                                  ["Yield", "Safe", "Trade", "Hold"]),
        "own wave past the lane (farm spot > 0)": (p["farm_spot"] > 0, ["Hold", "Trade"]),
    }
    col = {"Hold": "p_Hold", "Yield": "p_Yield", "Trade": "p_Trade", "Safe": "p_Safe", "Recall": "p_Recall",
           "SP": "p_SP", "SH": "p_SH", "F": "p_F", "Pressure": "p_obj_Pressure", "Vision": "p_obj_Vision"}
    out = {}
    for name, (m, acts) in situations.items():
        out[name] = {"n": int(m.sum()), "share_of_present_turns": float(m.sum() / n),
                     "p": {a: _mean(p[col[a]][m]) for a in acts}}
    return out


def state_table(t: Turns) -> dict:
    """Where the focal champion is, at what HP, and how often each situation arises (Phase A.1, item 5)."""
    p = t.where(t["present"])
    games = len(np.unique(t["seat"] * 100_000 + t["ep"]))
    g = max(games, 1)
    death = t.where(t["died"])
    return {"games": games, "present_turns_per_game": len(p) / g,
            "hp_hist": {int(h): _share(p["hp"] == h) for h in range(1, 7)},
            "pos_hist": {int(x): _share(p["pos"] == x) for x in range(-2, 3)},
            "recall_legal_by_hp": {int(h): _share(p["legal_recall"][p["hp"] == h]) for h in range(1, 7)},
            "end_pos_hist": {int(x): _share(p["end_pos"] == x) for x in range(-2, 3)},
            "farm_spot_hist": {int(x): _share(p["farm_spot"] == x) for x in range(-2, 3)},
            "turns_hp_le2_per_game": float((p["hp"] <= 2).sum() / g),
            "deaths_per_game": float(t["died"].sum() / g),
            "hp_at_start_of_death_turn": {int(h): _share(death["hp"] == h) for h in range(1, 7)},
            "death_turn_damage_sources": {"trade": _mean(death["dmg_taken"] - death["gank_dmg"] - death["tower_dmg"]),
                                          "gank": _mean(death["gank_dmg"]), "tower": _mean(death["tower_dmg"])},
            "deaths_with_gank_damage": _share(death["gank_dmg"] > 0),
            "death_k_mean": _mean(death["k"]),
            "freeze_legal_per_game": float(p["legal_freeze"].sum() / g),
            "objective_legal_per_game": float(p["legal_objective"].sum() / g),
            "crashes_per_game": float(t["crash"].sum() / g),
            "enemy_away_turns_per_game": float((p["eabsent"] > 0).sum() / g)}


def outcome_table(t: Turns) -> Dict[str, dict]:
    """Realized outcomes after each chosen behavior and wave action (descriptive, confounded by the state in which
    the policy chooses them; Phase B gives the causal comparisons). Phase A.1, item 6."""
    p = t.where(t["present"])
    out = {}
    groups = {b: p["beh"] == i for i, b in enumerate(BEHAVIORS)}
    groups.update({"Shove": p["wave"] == SH, "Freeze": p["wave"] == F, "Pressure": p["obj"] == PRESSURE,
                   "Vision": p["obj"] == VISION})
    for name, m in groups.items():
        q = p.where(m)
        out[name] = {"n": int(m.sum()), "reward": _mean(q["reward"]), "reward_next4": _mean(q["reward_next4"]),
                     "farm": _mean(q["farm"]), "efarm": _mean(q["efarm"]), "dmg_taken": _mean(q["dmg_taken"]),
                     "dmg_dealt": _mean(q["dmg_dealt"]), "died_next4": _mean(q["died_next4"].astype(float)),
                     "connected": _mean(q["connected"].astype(float)), "hp": _mean(q["hp"]), "k": _mean(q["k"]),
                     "ret_to_go": _mean(q["ret_to_go"])}
    return out


def economy_table(t: Turns) -> dict:
    """Per-game reward decomposition and where farm is lost (Phase A.3)."""
    gid = t["seat"] * 100_000 + t["ep"]
    games = np.unique(gid)
    g = len(games)
    per = {k: float(t[k].sum() / g) for k in ("reward", "farm", "efarm", "avail", "eavail", "plates", "eplates",
                                               "bounty", "ebounty")}
    lost_by_mode = {m: float((t["avail"] - t["farm"])[t["mode"] == i].sum() / g) for i, m in enumerate(MODES)}
    turns_by_mode = {m: float((t["mode"] == i).sum() / g) for i, m in enumerate(MODES)}
    away = t["mode"] == MODES.index("away")
    pay = np.array([t["reward"][gid == x].sum() for x in games])
    kd = np.array([t["edied"][gid == x].sum() - t["died"][gid == x].sum() for x in games], dtype=float)
    slope = float(np.polyfit(kd, pay, 1)[0]) if kd.std() > 0 else float("nan")
    return {"per_game": per, "farm_lost_by_mode": lost_by_mode, "turns_by_mode": turns_by_mode,
            "opp_farm_per_turn_while_away": _mean(t["efarm"][away]),
            "opp_farm_per_turn_while_present": _mean(t["efarm"][~away]),
            "own_farm_per_present_turn": _mean(t["farm"][~away]),
            "payoff_per_kill_difference": slope, "payoff_sd": float(pay.std(ddof=1)) if g > 1 else float("nan"),
            "kill_difference_sd": float(kd.std(ddof=1)) if g > 1 else float("nan"),
            "payoff_mean": float(pay.mean())}


def summarize(t: Turns) -> dict:
    return {"legality": legality_table(t), "opportunities": opportunity_table(t), "states": state_table(t),
            "outcomes": outcome_table(t), "economy": economy_table(t)}
