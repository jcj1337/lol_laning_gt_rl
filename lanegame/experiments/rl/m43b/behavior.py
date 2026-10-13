"""What a policy does, and whether it depends on the state (M4.3b §4 of the brief).

Three instruments, all NumPy except the network forward pass:

  1. `play_rows`: games of a focal policy x against an opponent y (x in both seats on the same seeds, or both seats
     of a self-play game), one row per PRESENT turn of x: the state seen from x's side (fronts, contact, siege masses,
     positions, HP fractions, levels, attack-power ratio, absence, legality of Recall and Pressure, unspent gold),
     the policy's full action distribution collapsed to categories (wave Hold / Thin / Clear, behaviour Hold / Yield
     / Trade / Safe / Recall, Pressure), its entropy and the entropy normalized by log |legal|, the action actually
     played, and what followed (reward, farm, death, recall, the discounted return to go over the window).
  2. `situations`: named boolean masks over those rows (a neutral lane, own or enemy wave pushing, a stacked front, an
     imminent crash, defending at the tower, sieging, a freeze-like lane, the enemy absent, a power lead or deficit,
     low HP, Recall legal, cash ready). `conditional_tables` gives, per situation, the mean category probabilities and
     the realized choices: a policy that always Clears has the same row everywhere; a state-dependent one does not.
     `state_dependence` summarizes it as the mutual information between the chosen wave action and a coarse state
     partition, and as the spread of P(Clear) across situations.
  3. `counterfactual`: paired counterfactual rollouts (`lanegame.diagnostics.branch`) from sampled reachable states of
     each situation: commit to each legal first move (Hold / Thin / Clear with Hold position, Trade, Safe, Yield,
     Pressure where legal, Recall where legal) then follow the policy, against the policy, on common random numbers
     over a window of CF_WINDOW turns, discounted. This asks whether the move the policy prefers is the move its own
     continuation values most, i.e. whether its preference is state-appropriate or a habit.

`profile` is the compact readout of 1. used on the learning curves (action shares, mean probabilities, entropies,
per-game counters including returns after death against after Recall).
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ....champions_m4 import context
from ....config import (B_HOLD, B_RECALL, B_SAFE, B_TRADE, B_YIELD, CHAMP_ABSENT, GameConfig, PRESSURE, W_CLEAR,
                        W_HOLD, W_THIN, champ_act, champ_behavior, champ_objective, champ_wave)
from ....env import LaneEnv
from ....env.agents import episode_streams
from .common import (BEHAVIOR_SEED, CF_ROLL_SEED, CF_STATE_SEED, CONDITIONS, PRIMARY, SEEDS, WINDOW_DEV, agent_from_spec,
                     cfg_game, load, meta, run_jobs, save_json)

WAVES = ("Hold", "Thin", "Clear")
WAVE_INDEX = {W_HOLD: 0, W_THIN: 1, W_CLEAR: 2}              # slot code -> index in WAVES
BEHS = ("Hold", "Yield", "Trade", "Safe", "Recall")
RECALL = champ_act(W_HOLD, B_RECALL)
CF_WINDOW = 300                                              # turns of a counterfactual rollout (gamma^300 = 0.049)
CF_STATES, CF_ROLLOUTS = 8, 16
BEHAVIOR_GAMES, BEHAVIOR_GAMES_REF = 24, 12
BEHAVIOR_REFS = ("push_punish", "clear_punish")


# ---------------------------------------------------------------------------------------------- probabilities
def full_distribution(agent, obs) -> np.ndarray:
    """Full-length probability vector (46) of `agent` at `obs` (0 on illegal actions)."""
    f = getattr(agent, "probabilities", None)
    if callable(f):
        return np.asarray(f(obs), dtype=float)
    p = np.zeros(len(obs.mask))
    p[list(obs.legal)] = np.asarray(agent.distribution(obs), dtype=float)
    return p


def category_masses(p: np.ndarray) -> dict:
    """Wave (Hold, Thin, Clear), behaviour (Hold, Yield, Trade, Safe, Recall) and Pressure masses of a full vector."""
    wave, beh, press = np.zeros(3), np.zeros(5), 0.0
    for a in np.flatnonzero(p > 0):
        if a == CHAMP_ABSENT:
            continue
        wave[WAVE_INDEX[champ_wave(a)]] += p[a]
        beh[champ_behavior(a)] += p[a]
        if champ_objective(a) == PRESSURE:
            press += p[a]
    q = p[p > 0]
    return {"wave": wave, "beh": beh, "pressure": press, "entropy": float(-(q * np.log(q)).sum())}


# ---------------------------------------------------------------------------------------------- state features
def turn_features(obs, cfg: GameConfig, ctx=None) -> dict:
    """The decision situation of a PRESENT champion from its own side (own coordinates: own tower -T, enemy tower T)."""
    ctx = ctx if ctx is not None else context(cfg)
    t = ctx.t
    lane, m4, s = obs.state, obs.champ, obs.side
    me, en = m4.side(s), m4.side(1 - s)
    own_fx, own_fh = lane.side(s)[:2]
    en_fx, en_fh = lane.side(1 - s)[:2]
    L = t.level_by_xp[min(me.xp, t.xp_cap)]
    eL = t.level_by_xp[min(en.xp, t.xp_cap)]
    da = ctx.hit(1, L, me.items) / t.hp_unit[eL]
    db = ctx.hit(1, eL, en.items) / t.hp_unit[L]
    contact = own_fh > 0 and en_fh > 0 and own_fx + en_fx >= -1
    T = cfg.tower
    g = cfg.gold_scale
    return {
        "k": obs.k, "phase": obs.k % cfg.phase_period, "pos": me.pos, "epos": -en.pos if en.absent == 0 else -99,
        "dist": abs(me.pos + en.pos) if en.absent == 0 else 99,
        "hp_frac": me.hp / t.hp_max[L], "ehp_frac": (en.hp / t.hp_max[eL]) if en.absent == 0 else 1.0,
        "level": L, "elevel": eL, "power": da / db if db > 0 else 1.0,
        "own_fx": own_fx if own_fh > 0 else -99, "own_fh": own_fh, "en_fx": -en_fx if en_fh > 0 else 99, "en_fh": en_fh,
        "contact": contact, "contact_cell": (own_fx - en_fx) / 2 if contact else -99.0,
        "own_siege": lane.siege_of(s), "en_siege": lane.siege_of(1 - s),
        "enemy_absent": en.absent > 0, "recall_legal": RECALL in obs.legal,
        "pressure_legal": any(champ_objective(a) == PRESSURE for a in obs.legal),
        "gold": me.gold / g, "invested": me.items / g, "egold": en.gold / g, "einvested": en.items / g,
        "at_own_tower": me.pos == -T, "at_enemy_tower": me.pos == T,
        "tower_dmg": me.tower_dmg, "etower_dmg": en.tower_dmg, "n_legal": len(obs.legal),
    }


FEATURE_KEYS = ("k", "phase", "pos", "epos", "dist", "hp_frac", "ehp_frac", "level", "elevel", "power", "own_fx",
                "own_fh", "en_fx", "en_fh", "contact", "contact_cell", "own_siege", "en_siege", "enemy_absent",
                "recall_legal", "pressure_legal", "gold", "invested", "egold", "einvested", "at_own_tower",
                "at_enemy_tower", "tower_dmg", "etower_dmg", "n_legal")
PROB_KEYS = ("p_Hold", "p_Thin", "p_Clear", "p_bH", "p_bY", "p_bT", "p_bS", "p_bR", "p_P", "entropy", "norm_entropy")
OUTCOME_KEYS = ("wave", "beh", "press", "reward", "farm", "efarm", "died", "edied", "recalled", "ret_to_go", "ep",
                "seat", "turn")


def _probs(agent, obs) -> dict:
    p = full_distribution(agent, obs)
    c = category_masses(p)
    n = max(len(obs.legal), 1)
    return {"p_Hold": c["wave"][0], "p_Thin": c["wave"][1], "p_Clear": c["wave"][2],
            "p_bH": c["beh"][B_HOLD], "p_bY": c["beh"][B_YIELD], "p_bT": c["beh"][B_TRADE], "p_bS": c["beh"][B_SAFE],
            "p_bR": c["beh"][B_RECALL], "p_P": c["pressure"], "entropy": c["entropy"],
            "norm_entropy": c["entropy"] / np.log(n) if n > 1 else 0.0}


def play_rows(cfg: GameConfig, x, y, episodes: int, window: int, seed: int, focal_both: bool = False,
              infos_out: Optional[list] = None) -> Dict[str, np.ndarray]:
    """Rows of x's present turns. x in seat A on `episodes` seeds, then in seat B on the same seeds (paired); with
    `focal_both` (self-play: y is x) both seats of each game are focal and each seed is played once. `infos_out`
    collects (seat, ep, [info per turn]) for the state sampling of the counterfactuals."""
    ctx = context(cfg)
    gamma = cfg.gamma
    env = LaneEnv(cfg, max_turns=window, cache_size=20_000)
    rows: List[dict] = []
    counters = []
    seats = (0,) if focal_both else (0, 1)
    children = np.random.SeedSequence(seed).spawn(episodes)
    for seat in seats:
        for ep, ss in enumerate(children):
            _, env_ss, rng_a, rng_b = episode_streams(ss)
            agents = (x, y) if seat == 0 else (y, x)
            rngs = (rng_a, rng_b)
            for ag in agents:
                if callable(getattr(ag, "reset", None)):
                    ag.reset()
            obs = env.reset(seed=env_ss)
            focal = (0, 1) if focal_both else (seat,)
            ep_rows = {s: [] for s in focal}
            infos = []
            cnt = {s: {"deaths": 0, "recalls": 0, "trades": 0, "connected": 0, "farm": 0.0, "plates": 0.0, "crashes": 0,
                       "returns_after_death": 0, "returns_after_recall": 0, "invested_after_death": 0.0,
                       "invested_after_recall": 0.0, "gold_at_death": [], "gold_at_recall": [], "present": 0,
                       "pressure_turns": 0} for s in focal}
            cause = {0: None, 1: None}
            for turn in range(window):
                feats = {}
                for s in focal:
                    o = obs[s]
                    if o.legal != (CHAMP_ABSENT,):
                        f = turn_features(o, cfg, ctx)
                        f.update(_probs(agents[s], o))
                        feats[s] = f
                acts = (agents[0].act(obs[0], rngs[0]), agents[1].act(obs[1], rngs[1]))
                r = env.step(*acts)
                info = r.info
                for ag_s in (0, 1):
                    nf = getattr(agents[ag_s], "notify", None)
                    if callable(nf):
                        nf(info, ag_s)
                if infos_out is not None:
                    infos.append(info)
                for s in focal:
                    o_ = 1 - s
                    if info["died"][s]:
                        cause[s] = "death"
                        cnt[s]["gold_at_death"].append(info["champ_before"][s][5] / cfg.gold_scale)
                    elif info["recalled"][s]:
                        cause[s] = "recall"
                        cnt[s]["gold_at_recall"].append(info["champ_before"][s][5] / cfg.gold_scale)
                    if info["returned"][s]:
                        if cause[s] == "death":
                            cnt[s]["returns_after_death"] += 1
                            cnt[s]["invested_after_death"] += info["invested"][s] / cfg.gold_scale
                        elif cause[s] == "recall":
                            cnt[s]["returns_after_recall"] += 1
                            cnt[s]["invested_after_recall"] += info["invested"][s] / cfg.gold_scale
                    cnt[s]["deaths"] += int(info["died"][s])
                    cnt[s]["recalls"] += int(info["recalled"][s])
                    cnt[s]["trades"] += int(info["trade_attempted"][s])
                    cnt[s]["connected"] += int(info["trade_connected"][s])
                    cnt[s]["farm"] += info["farm"][s]
                    cnt[s]["plates"] += info["plates"][s]
                    cnt[s]["crashes"] += int(info["crash_hp"][s] > 0)
                    if s in feats:
                        a = acts[s]
                        f = feats[s]
                        cnt[s]["present"] += 1
                        cnt[s]["pressure_turns"] += int(champ_objective(a) == PRESSURE)
                        f.update({"wave": WAVE_INDEX[champ_wave(a)], "beh": champ_behavior(a),
                                  "press": int(champ_objective(a) == PRESSURE), "reward": r.rewards[s],
                                  "farm": info["farm"][s], "efarm": info["farm"][o_], "died": int(info["died"][s]),
                                  "edied": int(info["died"][o_]), "recalled": int(info["recalled"][s]),
                                  "ep": ep, "seat": s, "turn": turn, "_r": r.rewards[s]})
                        ep_rows[s].append(f)
                    else:
                        ep_rows[s].append({"_absent": True, "_r": r.rewards[s]})
                if r.done:
                    break
                obs = r.obs
            for s in focal:
                g = 0.0
                for rw in reversed(ep_rows[s]):               # discounted return to go over the rest of the window
                    g = rw["_r"] + gamma * g
                    rw["ret_to_go"] = g
                rows.extend(rw for rw in ep_rows[s] if "_absent" not in rw)
                counters.append({"seat": s, "ep": ep, **cnt[s]})
            if infos_out is not None:
                infos_out.append((seat, ep, infos))
    keys = FEATURE_KEYS + PROB_KEYS + OUTCOME_KEYS
    out = {k: np.array([rw[k] for rw in rows]) for k in keys}
    out["_counters"] = counters
    return out


# ---------------------------------------------------------------------------------------------- situations
def situations(R: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Named masks over rows (present turns of the focal champion). Overlapping by design."""
    pres = np.ones(len(R["k"]), dtype=bool)
    en_here = ~R["enemy_absent"]
    T = 2
    s = {
        "all present turns": pres,
        "neutral lane (fronts fighting at the lane cell, equal power)": (R["contact"] & (R["contact_cell"] == 0)
                                                                       & en_here & (np.abs(R["power"] - 1) <= 0.1)),
        # on the 5-cell lane the only enemy-half cell is own +1, next to the enemy tower: a front there crashes next
        "own wave pushing (own front at own +1, next to the enemy tower)": (R["own_fh"] > 0) & (R["own_fx"] >= 1),
        "enemy wave pushing (enemy front at own -1, next to the own tower)": (R["en_fh"] > 0) & (R["en_fx"] <= -1),
        "own front stacked (>= 16 HP) in contact": (R["own_fh"] >= 16) & R["contact"],
        "defending (enemy mass under the own tower)": R["en_siege"] > 0,
        "sieging (own mass under the enemy tower, champion there)": (R["own_siege"] > 0) & R["at_enemy_tower"],
        "freeze-like (enemy front at own -1, own front in the own tower cell)": ((R["en_fh"] > 0) & (R["en_fx"] == -1)
                                                                               & (R["own_fh"] > 0) & (R["own_fx"] == -T)),
        "enemy champion absent": R["enemy_absent"],
        "stronger (attack-power ratio >= 1.25), enemy present": (R["power"] >= 1.25) & en_here,
        "weaker (attack-power ratio <= 0.8), enemy present": (R["power"] <= 0.8) & en_here,
        "low HP (<= 40%), enemy present": (R["hp_frac"] <= 0.4) & en_here,
        "low HP (<= 40%) and Recall legal": (R["hp_frac"] <= 0.4) & R["recall_legal"],
        "Recall legal": R["recall_legal"],
        "Recall legal, HP >= 80%": R["recall_legal"] & (R["hp_frac"] >= 0.8),
        "Recall legal after own crash (own mass under the enemy tower)": R["recall_legal"] & (R["own_siege"] > 0),
        "Recall legal with >= 4 resources unspent": R["recall_legal"] & (R["gold"] >= 4),
        "Pressure legal": R["pressure_legal"],
        "enemy within trade reach (distance <= 1), enemy present": (R["dist"] <= 1) & en_here,
    }
    return s


CF_SITUATIONS = {                                           # short key -> situation name (counterfactual sampling)
    "neutral": "neutral lane (fronts fighting at the lane cell, equal power)",
    "own_pushing": "own wave pushing (own front at own +1, next to the enemy tower)",
    "enemy_pushing": "enemy wave pushing (enemy front at own -1, next to the own tower)",
    "stacked": "own front stacked (>= 16 HP) in contact",
    "defending": "defending (enemy mass under the own tower)",
    "freeze_like": "freeze-like (enemy front at own -1, own front in the own tower cell)",
    "enemy_absent": "enemy champion absent",
    "stronger": "stronger (attack-power ratio >= 1.25), enemy present",
    "weaker": "weaker (attack-power ratio <= 0.8), enemy present",
    "low_hp": "low HP (<= 40%), enemy present",
}


BOOL_KEYS = ("contact", "enemy_absent", "recall_legal", "pressure_legal", "at_own_tower", "at_enemy_tower")


def situation_names() -> List[str]:
    """The situation labels, in order (from an empty, correctly typed row table)."""
    R = {k: np.zeros(0, dtype=bool if k in BOOL_KEYS else float) for k in FEATURE_KEYS}
    return list(situations(R))


def _mean(x) -> float:
    return float(np.mean(x)) if len(x) else float("nan")


def conditional_tables(R: Dict[str, np.ndarray]) -> Dict[str, dict]:
    S = situations(R)
    n_all = max(len(R["k"]), 1)
    out = {}
    for name, m in S.items():
        n = int(m.sum())
        row = {"n": n, "share": n / n_all}
        for key in PROB_KEYS:
            row[key] = _mean(R[key][m])
        for i, w in enumerate(WAVES):
            row[f"chose_{w}"] = _mean(R["wave"][m] == i)
        for i, b in enumerate(BEHS):
            row[f"chose_b{b[0]}"] = _mean(R["beh"][m] == i)
        row["chose_P"] = _mean(R["press"][m])
        row["reward"] = _mean(R["reward"][m])
        row["ret_to_go"] = _mean(R["ret_to_go"][m])
        row["died"] = _mean(R["died"][m])
        row["mean_n_legal"] = _mean(R["n_legal"][m])
        out[name] = row
    return out


def _entropy(counts: np.ndarray) -> float:
    p = counts / max(counts.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def state_dependence(R: Dict[str, np.ndarray]) -> dict:
    """How much the realized wave / behaviour choice depends on a coarse state partition: I(action; partition) in
    nats, as a share of H(action), with the partition = (enemy present, contact cell class, power class, HP class,
    own siege, enemy siege). Also the spread of P(Clear) and P(Trade) across the named situations (n >= 200)."""
    cc = np.where(R["contact"], np.clip(np.round(R["contact_cell"]), -2, 2), 3).astype(int)
    pw = np.digitize(R["power"], [0.8, 1.25])
    hp = np.digitize(R["hp_frac"], [0.4, 0.8])
    part = ((R["enemy_absent"].astype(int) * 6 + (cc + 2)) * 3 + pw) * 3 + hp
    part = part * 2 + (R["own_siege"] > 0).astype(int)
    part = part * 2 + (R["en_siege"] > 0).astype(int)
    out = {}
    for label, act, k in (("wave", R["wave"], 3), ("behavior", R["beh"], 5)):
        H = _entropy(np.bincount(act, minlength=k).astype(float))
        Hc = 0.0
        for v in np.unique(part):
            m = part == v
            Hc += m.mean() * _entropy(np.bincount(act[m], minlength=k).astype(float))
        out[label] = {"H_action": H, "H_action_given_state": Hc, "mutual_information": H - Hc,
                      "mi_share": (H - Hc) / H if H > 0 else 0.0}
    tab = conditional_tables(R)
    big = {n: r for n, r in tab.items() if r["n"] >= 200}
    for key in ("p_Clear", "p_Thin", "p_Hold", "p_bT", "p_bS", "p_bR"):
        vals = {n: r[key] for n, r in big.items()}
        if vals:
            lo, hi = min(vals, key=vals.get), max(vals, key=vals.get)
            out[f"spread_{key}"] = {"min": vals[lo], "min_in": lo, "max": vals[hi], "max_in": hi,
                                    "marginal": tab["all present turns"][key]}
    out["partition_classes"] = int(len(np.unique(part)))
    return out


# ---------------------------------------------------------------------------------------------- profile
def profile(cfg: GameConfig, agent, games: int, window: int, seed: int, opponent=None) -> dict:
    """Compact readout of a policy: category shares and mean probabilities over present turns, entropies, counters
    per game and side. Self-play (both seats focal) unless an opponent is given."""
    R = play_rows(cfg, agent, opponent if opponent is not None else agent, games, window, seed,
                  focal_both=opponent is None)
    cnt = R.pop("_counters")
    n = max(len(R["k"]), 1)
    out = {"games": games, "window": window, "present_turns": int(n), "self_play": opponent is None,
           "waves": {w: _mean(R["wave"] == i) for i, w in enumerate(WAVES)},
           "behaviors": {b: _mean(R["beh"] == i) for i, b in enumerate(BEHS)},
           "pressure": _mean(R["press"]),
           "p_waves": {w: _mean(R[f"p_{w}"]) for w in WAVES},
           "p_behaviors": {b: _mean(R[f"p_b{b[0]}"]) for b in BEHS},
           "p_pressure": _mean(R["p_P"]), "entropy": _mean(R["entropy"]), "norm_entropy": _mean(R["norm_entropy"]),
           "wave_marginal_entropy": _entropy(np.bincount(R["wave"], minlength=3).astype(float)),
           "behavior_marginal_entropy": _entropy(np.bincount(R["beh"], minlength=5).astype(float)),
           "mean_n_legal": _mean(R["n_legal"]),
           "recall_legal_share": _mean(R["recall_legal"]),
           "p_recall_when_legal": _mean(R["p_bR"][R["recall_legal"]]),
           "chose_recall_when_legal": _mean(R["beh"][R["recall_legal"]] == B_RECALL)}
    g = max(len(cnt), 1)
    for key in ("deaths", "recalls", "trades", "connected", "farm", "plates", "crashes", "returns_after_death",
                "returns_after_recall", "invested_after_death", "invested_after_recall", "present", "pressure_turns"):
        out[f"per_game_{key}"] = float(sum(c[key] for c in cnt) / g)
    gd = [v for c in cnt for v in c["gold_at_death"]]
    gr = [v for c in cnt for v in c["gold_at_recall"]]
    out["gold_at_death_mean"], out["gold_at_recall_mean"] = _mean(gd), _mean(gr)
    out["state_dependence"] = {k: v for k, v in state_dependence(R).items() if k in ("wave", "behavior")}
    return out


# ---------------------------------------------------------------------------------------------- stage: behavior
def _behavior_job(args) -> tuple:
    spec, opp, games, window, seed = args
    import torch
    torch.set_num_threads(1)
    cfg = cfg_game()
    x = agent_from_spec(spec, cfg)
    y = x if opp == "self" else agent_from_spec(f"ref:{opp}", cfg)
    t0 = time.perf_counter()
    R = play_rows(cfg, x, y, games, window, seed, focal_both=(opp == "self"))
    cnt = R.pop("_counters")
    out = {"spec": spec, "opponent": opp, "games": games, "window": window, "rows": int(len(R["k"])),
           "tables": conditional_tables(R), "state_dependence": state_dependence(R),
           "per_game": {k: float(np.mean([c[k] for c in cnt])) for k in
                        ("deaths", "recalls", "connected", "farm", "plates", "crashes", "returns_after_death",
                         "returns_after_recall", "invested_after_death", "invested_after_recall")},
           "elapsed_s": time.perf_counter() - t0}
    return (spec, opp), out


def behavior(conds: Sequence[str] = tuple(CONDITIONS), seeds: Sequence[int] = SEEDS, workers: int = 9,
             force: bool = False) -> dict:
    """Results already in behavior.json are reused unless `force`."""
    t0 = time.perf_counter()
    jobs = []
    for c in conds:
        for s in seeds:
            jobs.append((f"final:{c}:{s}", "self", BEHAVIOR_GAMES, WINDOW_DEV, BEHAVIOR_SEED))
            for ref in BEHAVIOR_REFS:
                jobs.append((f"final:{c}:{s}", ref, BEHAVIOR_GAMES_REF, WINDOW_DEV, BEHAVIOR_SEED + 1))
    for s in seeds:
        jobs.append((f"initial:{s}", "self", BEHAVIOR_GAMES_REF, WINDOW_DEV, BEHAVIOR_SEED))
    for ref in BEHAVIOR_REFS:                                   # the references' own conditional behaviour
        jobs.append((f"ref:{ref}", "self", BEHAVIOR_GAMES_REF, WINDOW_DEV, BEHAVIOR_SEED))
    prev = (load("behavior") or {}) if not force else {}
    results = dict(prev.get("results", {}))
    todo = [j for j in jobs if f"{j[0]}|{j[1]}" not in results]
    print(f"behavior: {len(jobs)} jobs, {len(todo)} to compute", flush=True)
    res = run_jobs(_behavior_job, todo, workers, label="behavior")
    results.update({f"{k[0]}|{k[1]}": v for k, v in res})
    data = {"meta": meta(stage="behavior", window=WINDOW_DEV, seed=BEHAVIOR_SEED, games=BEHAVIOR_GAMES),
            "elapsed_s": time.perf_counter() - t0, "situations": situation_names(), "results": results}
    save_json("behavior", data)
    return data


# ---------------------------------------------------------------------------------------------- counterfactuals
def candidate_set(cfg: GameConfig, legal) -> dict:
    """Named first moves: wave x Hold position, Trade, Safe, Yield, the Pressure variants and Recall where legal,
    plus the policy's own sampled move."""
    from ....diagnostics import Commit, FollowPolicy
    want = [("Hold", champ_act(W_HOLD, B_HOLD)), ("Thin", champ_act(W_THIN, B_HOLD)), ("Clear", champ_act(W_CLEAR, B_HOLD)),
            ("Hold+Trade", champ_act(W_HOLD, B_TRADE)), ("Clear+Trade", champ_act(W_CLEAR, B_TRADE)),
            ("Hold+Safe", champ_act(W_HOLD, B_SAFE)), ("Clear+Safe", champ_act(W_CLEAR, B_SAFE)),
            ("Hold+Yield", champ_act(W_HOLD, B_YIELD)),
            ("Hold+P", champ_act(W_HOLD, B_HOLD, PRESSURE)), ("Clear+P", champ_act(W_CLEAR, B_HOLD, PRESSURE)),
            ("Recall", RECALL)]
    cands = {n: Commit(a, n) for n, a in want if a in legal}
    cands["policy"] = FollowPolicy()
    return cands


def situation_masks(keys: Sequence[str]) -> dict:
    """Mask functions (rows, situations) -> mask for the named counterfactual situations."""
    return {k: (lambda R, S, name=CF_SITUATIONS[k]: S[name]) for k in keys}


def sample_states(cfg: GameConfig, agent, masks: dict, n_states: int, seed: int, window: int = WINDOW_DEV,
                  max_games: int = 60, per_game: int = 2, opponent=None) -> Dict[str, list]:
    """Up to n_states (Snapshot, focal seat, features) per mask from self-play games of `agent` (or from games of
    `agent` against `opponent`, in both seats), at most `per_game` per game and mask, from turns 8 .. window -
    CF_WINDOW / 2 (so a rollout window follows). `masks` maps a key to a function (rows, situations) -> boolean mask
    over the rows."""
    from ....diagnostics import Snapshot
    rng = np.random.default_rng(seed)
    which = list(masks)
    found = {k: [] for k in which}
    skills = (cfg.skill_a, cfg.skill_b)
    self_play = opponent is None
    for g in range(max_games):
        if all(len(found[k]) >= n_states for k in which):
            break
        infos_out: list = []
        R = play_rows(cfg, agent, agent if self_play else opponent, 1, window, seed + 1000 * g, focal_both=self_play,
                      infos_out=infos_out)
        S = situations(R)
        by_game = {(st, ep): infos for st, ep, infos in infos_out}
        for key in which:
            if len(found[key]) >= n_states:
                continue
            m = masks[key](R, S) & (R["turn"] >= 8) & (R["turn"] <= window - CF_WINDOW // 2)
            idx = np.flatnonzero(m)
            rng.shuffle(idx)
            took = 0
            for i in idx:
                turn, seat = int(R["turn"][i]), int(R["seat"][i])
                infos = by_game[(0 if self_play else seat, int(R["ep"][i]))]
                snap = Snapshot.from_info(infos[turn], skills)
                feats = {k: (float(R[k][i]) if R[k].dtype != bool else bool(R[k][i])) for k in FEATURE_KEYS}
                found[key].append((snap, seat, feats))
                took += 1
                if took >= per_game or len(found[key]) >= n_states:
                    break
    return found


def _cf_state(cfg: GameConfig, agent, snap, seat: int, feats: dict, n_roll: int, roll_seed: int, key: tuple,
              cache) -> dict:
    from ....diagnostics import branch, observation
    ob = observation(cfg, snap, seat)
    cands = candidate_set(cfg, ob.legal)
    res = branch(cfg, snap, seat, cands, (agent, agent), n_roll, roll_seed, key=key, cache=cache,
                 max_turns=CF_WINDOW)
    p = full_distribution(agent, ob)
    pi = {n: float(p[c.action]) for n, c in cands.items() if hasattr(c, "action")}
    q = {c: res.mean(c) for c in res.candidates}
    adv = {c: res.diff(c, "policy") for c in res.candidates if c != "policy"}
    legal_c = [c for c in res.candidates if c != "policy"]
    best = max(legal_c, key=lambda c: q[c][0])
    modal = max(pi, key=pi.get)
    # the policy's wave preference (Hold / Thin / Clear with Hold position) against the rollout values
    waves = {w: q[w][0] for w in WAVES if w in q}
    best_wave = max(waves, key=waves.get) if waves else None
    pi_wave = {w: feats[f"p_{w}"] if f"p_{w}" in feats else None for w in WAVES}
    dec = {c: {"farm_diff": float((res.farm[c][:, 0] - res.farm[c][:, 1]).mean()),
               "bounty_diff": float((res.bounty[c][:, 0] - res.bounty[c][:, 1]).mean()),
               "plates_diff": float((res.plates[c][:, 0] - res.plates[c][:, 1]).mean()),
               "deaths": res.deaths[c].mean(0).tolist(), "recalls": res.recalls[c].mean(0).tolist()}
           for c in res.candidates}
    return {"seat": seat, "k": snap.k, "features": feats, "snapshot": snap.to_dict(), "pi": pi, "q": q, "adv": adv,
            "best": best, "best_adv": adv[best], "modal": modal, "pi_modal": pi[modal], "q_modal": q[modal],
            "modal_is_best": modal == best, "modal_minus_best": float(q[modal][0] - q[best][0]),
            "modal_minus_best_se": float(res.diff(modal, best)[1]), "best_wave": best_wave,
            "wave_values": waves, "decomposition": dec}


def _cf_job(args) -> tuple:
    spec, key, n_states, n_roll, state_seed, roll_seed = args
    import torch
    from ....diagnostics import ProbCache
    torch.set_num_threads(1)
    cfg = cfg_game()
    agent = agent_from_spec(spec, cfg)
    t0 = time.perf_counter()
    states = sample_states(cfg, agent, situation_masks([key]), n_states, state_seed)[key]
    cache = ProbCache()
    out = [_cf_state(cfg, agent, snap, seat, feats, n_roll, roll_seed, (i,), cache)
           for i, (snap, seat, feats) in enumerate(states)]
    return (spec, key), {"spec": spec, "situation": key, "states": out, "n_found": len(states),
                         "elapsed_s": time.perf_counter() - t0}


def aggregate_cf(states: List[dict]) -> dict:
    """Across the states of one (policy, situation): per candidate the mean paired advantage over the policy's own
    play (states are the unit), the share of states in which the policy's modal action is the best candidate, and
    the mean regret of the modal action against the best (resources, discounted over the rollout window)."""
    if not states:
        return {}
    cands = sorted({c for st in states for c in st["adv"]})
    per = {}
    for c in cands:
        d = np.array([st["adv"][c][0] for st in states if c in st["adv"]])
        se = np.array([st["adv"][c][1] for st in states if c in st["adv"]])
        h = 1.96 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        per[c] = {"n": int(len(d)), "mean_adv": float(d.mean()), "ci": float(h),
                  "share_pos": float(np.mean(d > 1.96 * se)), "share_neg": float(np.mean(d < -1.96 * se))}
    regret = np.array([st["modal_minus_best"] for st in states])
    best_waves = [st["best_wave"] for st in states if st["best_wave"]]
    return {"n_states": len(states), "candidates": per,
            "modal_is_best_share": float(np.mean([st["modal_is_best"] for st in states])),
            "modal_regret_mean": float(-regret.mean()),
            "modal_regret_ci": float(1.96 * regret.std(ddof=1) / np.sqrt(len(regret))) if len(regret) > 1 else float("nan"),
            "modal_actions": {m: sum(1 for st in states if st["modal"] == m) for m in {st["modal"] for st in states}},
            "best_actions": {m: sum(1 for st in states if st["best"] == m) for m in {st["best"] for st in states}},
            "best_wave_counts": {w: best_waves.count(w) for w in WAVES},
            "mean_pi_modal": float(np.mean([st["pi_modal"] for st in states]))}


def counterfactual(conds: Sequence[str] = tuple(CONDITIONS), seeds: Sequence[int] = SEEDS, workers: int = 9,
                   specs: Optional[Sequence[str]] = None, n_states: int = CF_STATES, n_roll: int = CF_ROLLOUTS,
                   keys: Optional[Sequence[str]] = None, save: bool = True, force: bool = False) -> dict:
    """Counterfactual first moves for the seed-0 final of every condition (and more seeds of the primary condition).
    Results already in counterfactual.json are reused unless `force`."""
    t0 = time.perf_counter()
    if specs is None:
        specs = [f"final:{c}:{seeds[0]}" for c in conds] + [f"final:{PRIMARY}:{s}" for s in seeds[1:] if PRIMARY in conds]
    keys = list(keys) if keys is not None else list(CF_SITUATIONS)
    jobs = [(spec, key, n_states, n_roll, CF_STATE_SEED + 10 * i + j, CF_ROLL_SEED + 10 * i + j)
            for i, spec in enumerate(specs) for j, key in enumerate(keys)]
    prev = (load("counterfactual") or {}) if (save and not force) else {}
    results = dict(prev.get("results", {}))
    todo = [j for j in jobs if f"{j[0]}|{j[1]}" not in results]
    print(f"counterfactual: {len(jobs)} jobs, {len(todo)} to compute", flush=True)
    res = run_jobs(_cf_job, todo, workers, label="counterfactual")
    results.update({f"{k[0]}|{k[1]}": v for k, v in res})
    for v in results.values():
        v["summary"] = aggregate_cf(v["states"])
    data = {"meta": meta(stage="counterfactual", window=CF_WINDOW, states=n_states, rollouts=n_roll,
                         state_seed=CF_STATE_SEED, roll_seed=CF_ROLL_SEED),
            "situations": CF_SITUATIONS, "elapsed_s": time.perf_counter() - t0, "results": results}
    if save:
        save_json("counterfactual", data)
    return data


__all__ = ["WAVES", "BEHS", "full_distribution", "category_masses", "turn_features", "play_rows", "situations",
           "conditional_tables", "state_dependence", "situation_names", "profile", "behavior", "candidate_set", "sample_states",
           "situation_masks", "aggregate_cf", "counterfactual", "CF_SITUATIONS", "CF_WINDOW"]
