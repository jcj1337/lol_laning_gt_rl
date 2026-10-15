"""M4.3c quick XP-range screen (docs/m4/m4_3/m4_3c_xp_range_screen.md): does a smaller XP collection range (xp_range 2 or 1
instead of 3) make XP denial during a freeze meaningful, without disrupting neutral laning?

Configuration overrides only: `GameConfig.m4_lasthit(xp_range=r)` for r in RANGES. The engine, every preset, the
observation and the M4.3b checkpoints are unchanged. The checkpoints were trained at xp_range 3, so their play under 2
or 1 is a screen of a fixed policy, not an equilibrium of the modified game.

Stage `freeze`  the harness of freeze.py: the `freeze` start of zoning_natural.starts (turn 51, level 10: B's wave held
                at A's -1, A's front in its tower cell, A at -2, B at its +1), built ONCE under the baseline config so
                that every range starts from the identical state; gaps equal and A ahead (x1.76 attack power); the same
                64 paired seeds per cell as freeze.py, so that the xp_range 3 cells reproduce results/m43c/freeze.json
                exactly. A freezes with the threshold controller ("freezer"), freezes and punishes a B that starts in
                reach ("freezer_punish"), or does not freeze ("no_freeze": Clear + Hold). B answers for 40 turns with
                one fixed response:
                  approach_farm   walk up and last-hit (Hold, Hold)
                  step_back       Safe: never ahead of its own -1 (safe_line -1, the preset)
                  retreat         Safe with safe_line -2 (a configuration override of this response's cells only):
                                  stay in its own tower cell
                  break_clear     Clear + Hold
                  break_punish    Clear + punish + Pressure
                  learned         the M4.3b final base s0 (learned_ent03: ent03 s0)
                then BOTH sides last-hit (Hold, Hold) for EXT more turns (the extension). XP still owed to B by the
                freezer's minions alive at turn 40 is collected there; XP missed before turn 40 is not.
                The `neutral` start (turn 9, both at the lane) with Hold/Hold, Clear/Clear and Clear vs Hold is run in
                the same stage as the scripted neutral control.
Stage `games`   whole games from s0 (self-play of three finals, base s0 against two references, two reference
                pairings), GAME_TURNS turns, GAME_EPISODES paired seeds per matchup and range (new root XP_RANGE_SEED).

The kill ledger. Every HP of side s's enemy minions that dies, whoever killed it, is recorded once at side s's own cell x
(`info["kills"][s]`); a champion that is not away and stands at end_s gains xp_per_hp per HP with |x - end_s| <= xp_range
(champions_m4 step 7b). Per side and window:
    avail          xp_per_hp x the HP on the side's ledger
    missed_range   avail on turns the champion was present minus the XP it gained (before the level-cap clamp):
                   out of range, LOST for good
    missed_away    avail on turns the champion was away: LOST for good
    pending        xp_per_hp x the OPPONENT's minion HP alive at the end of the window (front + one fresh wave per rear
                   packet + crashed mass): XP not yet available, collected later only if the side is present and in
                   range when those minions die (DELAYED)
and `hist`, the ledger HP of present turns over (champion cell, kill cell), which prices any range on the same play
(HP with |x - end| > r is missed at range r).
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ...config import B_TRADE, GameConfig, W_CLEAR
from .common import FREEZE_SEED, OUT, load, mean_ci, meta, save_json, write_md

RANGES = (3, 2, 1)
WINDOWS = (20, 40)                 # the response windows; the extension follows the last one
EXT = 20
LAMBDA = 1 / 8
FREEZE_EPISODES = 64               # the seeds of freeze.py (reproduction of its xp_range 3 cells)
GAME_EPISODES = 32
GAME_TURNS = 160
GAME_MARKS = (50, 100, 160)
XP_RANGE_SEED = 75_000             # new root, disjoint from every earlier study (common.py)
OFF = 4                            # histogram index offset: own cells -4..+4
GAPS = ("equal", "a_ahead")
FREEZERS = ("freezer", "freezer_punish")
RESPONSES = ("approach_farm", "step_back", "retreat", "break_clear", "break_punish", "learned", "learned_ent03")
NO_FREEZE_RESPONSES = ("approach_farm", "step_back", "learned")
NEUTRAL = (("hold_hold", "hold_hold"), ("clear_clear", "clear_clear"), ("clear_clear", "hold_hold"))
LEARNED = {"learned": "final:base:0", "learned_ent03": "final:ent03:0"}
GAMES = [("base_s0 self", "final:base:0", "self"), ("base_s1 self", "final:base:1", "self"),
         ("ent03_s0 self", "final:ent03:0", "self"), ("base_s0 vs farm", "final:base:0", "ref:farm"),
         ("base_s0 vs clear_punish", "final:base:0", "ref:clear_punish"), ("farm vs farm", "ref:farm", "ref:farm"),
         ("clear_punish vs clear_punish", "ref:clear_punish", "ref:clear_punish")]
EXAMPLE_CELLS = (("freeze", "freezer", "step_back", "equal"), ("freeze", "freezer", "approach_farm", "equal"),
                 ("freeze", "freezer", "retreat", "equal"))
EXAMPLE_TURNS = 12


def cfg_range(r: int, retreat: bool = False) -> GameConfig:
    """The research preset with xp_range r (and safe_line -2 for the retreat response)."""
    return GameConfig.m4_lasthit(xp_range=r, **({"safe_line": -2} if retreat else {}))


def stock(state, s: int, wave_hp: int) -> int:
    """HP of side s's minions alive: front + one fresh wave per rear packet + crashed mass."""
    _, fh, rears, siege = state.side(s)[:4]
    return max(fh, 0) + wave_hp * len(rears) + siege


# ---------------------------------------------------------------------------------------------- recording
def run_turns(cfg, env, obs, plan: Callable[[int], tuple], rngs, turns: int, trace: int = 0) -> tuple:
    """Play up to `turns` turns, seat agents from plan(t); per-turn records (module docstring) and the first `trace`
    turns in detail."""
    from ...champions_m4 import context
    t_tab = context(cfg).t
    xi = cfg.xp_per_hp
    rows, detail = [], []
    for t in range(turns):
        agents = plan(t)
        a = agents[0].act(obs[0], rngs[0])
        b = agents[1].act(obs[1], rngs[1])
        r = env.step(a, b)
        i = r.info
        for s in (0, 1):
            n = getattr(agents[s], "notify", None)
            if callable(n):
                n(i, s)
        st = r.obs[0].state
        kills = [{int(x): int(h) for x, h in i["kills"][s].items() if h} for s in (0, 1)]
        rows.append({
            "earned": [x / cfg.gold_scale for x in i["earned_units"]],
            "xp": list(i["xp_gain"]), "raw": [i["xp_gain"][s] + i["xp_discarded"][s] for s in (0, 1)],
            "avail": [xi * sum(kills[s].values()) for s in (0, 1)], "kills": kills,
            "end": list(i["champ_end"]), "away": [bool(x) for x in i["away"]],
            "dmg": [i["damage_taken"][s] / t_tab.hp_max[i["levels_before"][s]] for s in (0, 1)],
            "died": [int(x) for x in i["died"]], "ganked": [int(x) for x in i["ganked"]],
            "gank_p": [float(x) for x in i["gank_p"]], "conn": [int(x) for x in i["trade_connected"]],
            "recalled": [int(x) for x in i["recalled"]], "levels": list(i["levels"]),
            "stock": [stock(st, s, cfg.wave_hp) for s in (0, 1)],
            "held": int(st.bh > 0 and st.bx >= 1 and st.ah > 0 and st.ax <= -1),
            "wave_b": i["waves"][1], "beh_b": i["behaviors"][1]})
        if t < trace:
            detail.append({"k": i["k"], "lane_after": list(st), "end": list(i["champ_end"]), "kills": kills,
                           "xp": list(i["xp_gain"]), "away": [bool(x) for x in i["away"]],
                           "actions": list(i["action_labels"]), "levels": list(i["levels"])})
        if r.done:
            break
        obs = r.obs
    return rows, detail


def window_stats(rows: List[dict], n: int) -> dict:
    """Window totals of both sides (suffix _a / _b) over the first n turns, plus the (champion cell, kill cell)
    histograms of present turns."""
    rows = rows[:n]
    out: Dict[str, object] = {}
    for s, tag in ((0, "a"), (1, "b")):
        o = 1 - s
        pres = [r for r in rows if not r["away"][s]]
        out[f"gold_{tag}"] = float(sum(r["earned"][s] for r in rows))
        out[f"xp_{tag}"] = float(sum(r["xp"][s] for r in rows))
        out[f"avail_{tag}"] = float(sum(r["avail"][s] for r in rows))
        out[f"missed_range_{tag}"] = float(sum(r["avail"][s] - r["raw"][s] for r in pres))
        out[f"missed_away_{tag}"] = float(sum(r["avail"][s] for r in rows if r["away"][s]))
        out[f"pending_{tag}"] = float(rows[-1]["stock"][o])      # xp_per_hp = 1 in every preset used here
        out[f"dmg_{tag}"] = float(sum(r["dmg"][s] for r in rows))
        out[f"deaths_{tag}"] = float(sum(r["died"][s] for r in rows))
        out[f"ganks_{tag}"] = float(sum(r["ganked"][s] for r in rows))
        out[f"gank_p_{tag}"] = float(np.mean([r["gank_p"][s] for r in pres])) if pres else float("nan")
        out[f"trades_{tag}"] = float(sum(r["conn"][s] for r in rows))
        out[f"recalls_{tag}"] = float(sum(r["recalled"][s] for r in rows))
        out[f"away_{tag}"] = float(len(rows) - len(pres))
        out[f"pos_{tag}"] = float(np.mean([r["end"][s] for r in pres])) if pres else float("nan")
        out[f"level_{tag}"] = float(rows[-1]["levels"][s])
        h = np.zeros((2 * OFF + 1, 2 * OFF + 1), dtype=np.int64)
        for r in pres:
            for x, hp in r["kills"][s].items():
                h[r["end"][s] + OFF, x + OFF] += hp
        out[f"hist_{tag}"] = h
    pres_b = [r for r in rows if not r["away"][1]]
    out["held"] = float(sum(r["held"] for r in rows))
    out["held_end"] = float(rows[-1]["held"])
    out["b_clear_share"] = float(np.mean([r["wave_b"] == W_CLEAR for r in pres_b])) if pres_b else float("nan")
    out["b_trade_share"] = float(np.mean([r["beh_b"] == B_TRADE for r in pres_b])) if pres_b else float("nan")
    return out


def first_turn(rows: List[dict], s: int, level: int) -> float:
    return next((float(t + 1) for t, r in enumerate(rows) if r["levels"][s] >= level), float("nan"))


def summarize(games: List[dict], n: int) -> dict:
    """mean_ci of every scalar over games, summed histograms, and B's value under both objectives."""
    w = [g[str(n)] for g in games]
    out = {}
    for k, v in w[0].items():
        if k.startswith("hist_"):
            out[k] = np.sum([x[k] for x in w], axis=0).tolist()
        else:
            out[k] = mean_ci([x[k] for x in w])
    out["gold_diff_b"] = mean_ci([x["gold_b"] - x["gold_a"] for x in w])
    out["xp_diff_b"] = mean_ci([x["xp_b"] - x["xp_a"] for x in w])
    out["value_b_gold_xp"] = mean_ci([x["gold_b"] - x["gold_a"] + LAMBDA * (x["xp_b"] - x["xp_a"]) for x in w])
    return out


def _paired(games: List[dict]) -> Dict[str, Dict[str, List[float]]]:
    """Per-game window totals kept for paired comparisons across ranges (same seeds)."""
    keys = ("gold_a", "gold_b", "xp_a", "xp_b", "missed_range_b", "missed_away_b", "dmg_b", "deaths_b")
    return {str(n): {k: [g[str(n)][k] for g in games] for k in keys} for n in games[0]}


# ---------------------------------------------------------------------------------------------- stage freeze
def _freeze_job(args) -> tuple:
    scenario, a_name, b_name, gap, r, episodes = args
    import torch
    from ...env import LaneEnv
    from ...env.agents import episode_streams
    from ..m4.zoning_natural import Policy, starts
    from ..rl.m43b.common import agent_from_spec
    from ..rl.m43b.references import ReactiveAgent
    from . import freeze as fz
    torch.set_num_threads(1)
    base = GameConfig.m4_lasthit()
    start = fz.gapped(starts(base)[scenario], base, gap)          # identical start for every range
    cfg = cfg_range(r, retreat=(b_name == "retreat"))
    pols = fz._policies()
    pols["retreat"] = Policy("retreat", "safe")
    xa = ReactiveAgent(pols[a_name], a_name)
    yb = agent_from_spec(LEARNED[b_name], cfg) if b_name in LEARNED else ReactiveAgent(pols[b_name], b_name)
    farm = ReactiveAgent(Policy("farm"), "farm")
    gi = fz.GAPS.index(gap)                                       # freeze.py's seed layout
    turns = max(WINDOWS) + EXT
    plan = lambda t: (xa, yb) if t < max(WINDOWS) else (farm, farm)
    lane, m4, k = start
    t0 = time.perf_counter()
    games, example = [], None
    for ep in range(episodes):
        _, env_ss, rng_a, rng_b = episode_streams(np.random.SeedSequence(FREEZE_SEED, spawn_key=(gi, ep)))
        env = LaneEnv(cfg, max_turns=turns, cache_size=5_000)
        obs = env.reset(seed=env_ss, root=lane, k0=k, champ=m4)
        rows, detail = run_turns(cfg, env, obs, plan, (rng_a, rng_b), turns, EXAMPLE_TURNS if ep == 0 else 0)
        games.append({str(n): window_stats(rows, n) for n in WINDOWS + (turns,)})
        if ep == 0:
            example = {"start": {"lane": list(lane), "champ": [list(m4.a), list(m4.b)], "k": k}, "turns": detail}
    key = f"{scenario}|{a_name}|{b_name}|{gap}|r{r}"
    return key, {"games": games, "example": example, "elapsed_s": time.perf_counter() - t0}


def freeze_jobs(episodes: int = FREEZE_EPISODES) -> list:
    jobs = []
    for r in RANGES:
        for a, b in NEUTRAL:
            jobs.append(("neutral", a, b, "equal", r, episodes))
        for gap in GAPS:
            for a in FREEZERS:
                for b in RESPONSES:
                    jobs.append(("freeze", a, b, gap, r, episodes))
            for b in NO_FREEZE_RESPONSES:
                jobs.append(("freeze", "no_freeze", b, gap, r, episodes))
    return jobs


# ---------------------------------------------------------------------------------------------- stage games
def _game_job(args) -> tuple:
    name, spec_a, spec_b, mi, r, episodes, turns = args
    import torch
    from ...env import LaneEnv
    from ...env.agents import episode_streams
    from ...env.discounted import game_agents
    from ..rl.m43b.common import agent_from_spec
    torch.set_num_threads(1)
    cfg = cfg_range(r)
    xa = agent_from_spec(spec_a, cfg)
    yb = xa if spec_b == "self" else agent_from_spec(spec_b, cfg)
    t0 = time.perf_counter()
    games = []
    for ep in range(episodes):
        _, env_ss, rng_a, rng_b = episode_streams(np.random.SeedSequence(XP_RANGE_SEED, spawn_key=(mi, ep)))
        agents = game_agents((xa, yb))
        env = LaneEnv(cfg, max_turns=turns, cache_size=20_000)
        obs = env.reset(seed=env_ss)
        rows, _ = run_turns(cfg, env, obs, lambda t: agents, (rng_a, rng_b), turns)
        g = {str(n): window_stats(rows, n) for n in GAME_MARKS if n <= len(rows)}
        g["first"] = {f"L{L}_{tag}": first_turn(rows, s, L) for L in (6, 11, 16, 18) for s, tag in ((0, "a"), (1, "b"))}
        games.append(g)
    return f"{name}|r{r}", {"spec_a": spec_a, "spec_b": spec_b, "games": games, "elapsed_s": time.perf_counter() - t0}


def game_jobs(episodes: int = GAME_EPISODES, turns: int = GAME_TURNS) -> list:
    return [(name, a, b, mi, r, episodes, turns) for r in RANGES for mi, (name, a, b) in enumerate(GAMES)]


def summarize_games(games: List[dict]) -> dict:
    out = {str(n): summarize(games, n) for n in GAME_MARKS}
    out["first"] = {k: mean_ci([g["first"][k] for g in games]) for k in games[0]["first"]}
    for n in GAME_MARKS:
        out[str(n)]["abs_xp_diff"] = mean_ci([abs(g[str(n)]["xp_b"] - g[str(n)]["xp_a"]) for g in games])
    return out


# ---------------------------------------------------------------------------------------------- run
def run(workers: int = 9, stages: Sequence[str] = ("freeze", "games")) -> dict:
    from ..rl.m43b.common import run_jobs
    out = load("xp_range") or {}
    out["meta"] = meta(stage="xp_range", ranges=list(RANGES), windows=list(WINDOWS), ext=EXT, lam=LAMBDA,
                       freeze_episodes=FREEZE_EPISODES, freeze_seed_root=FREEZE_SEED, game_episodes=GAME_EPISODES,
                       game_turns=GAME_TURNS, game_seed_root=XP_RANGE_SEED,
                       preset="GameConfig.m4_lasthit(xp_range=r) (+ safe_line=-2 for the retreat response)")
    t0 = time.perf_counter()
    if "freeze" in stages:
        res = run_jobs(_freeze_job, freeze_jobs(), workers, "xp_range freeze")
        out["freeze"] = {k: {"windows": {str(n): summarize(v["games"], n) for n in WINDOWS + (max(WINDOWS) + EXT,)},
                             "paired": _paired(v["games"]), "example": v["example"]} for k, v in res}
    if "games" in stages:
        res = run_jobs(_game_job, game_jobs(), workers, "xp_range games")
        out["games"] = {k: {"spec_a": v["spec_a"], "spec_b": v["spec_b"], "summary": summarize_games(v["games"]),
                            "paired": _paired([{str(n): g[str(n)] for n in GAME_MARKS} for g in v["games"]])}
                        for k, v in res}
    out["meta"]["elapsed_s"] = time.perf_counter() - t0
    save_json("xp_range", out)
    return out


# ---------------------------------------------------------------------------------------------- report tables
def _f(x, d=2):
    return "—" if x is None or not np.isfinite(x) else f"{x:+.{d}f}" if d >= 0 else f"{x:.{-d}f}"


def _mc(v, d=2):
    m, c = v
    return f"{_f(m, d)} ± {abs(c):.{abs(d)}f}" if np.isfinite(c) else _f(m, d)


def distance_shares(hist) -> List[float]:
    """Shares of present-turn ledger HP at |kill cell - champion cell| = 0, 1, 2, 3, >= 4."""
    h = np.asarray(hist, dtype=float)
    tot = h.sum()
    d = np.zeros(5)
    n = h.shape[0]
    for e in range(n):
        for x in range(n):
            d[min(abs(x - e), 4)] += h[e, x]
    return (d / tot).tolist() if tot else [float("nan")] * 5


def paired_diff(a: Dict[str, List[float]], b: Dict[str, List[float]], lam: float = 0.0) -> tuple:
    """Paired mean and 95% half-width of B's value in cell a minus cell b (same seeds)."""
    va = np.array(a["gold_b"]) - np.array(a["gold_a"]) + lam * (np.array(a["xp_b"]) - np.array(a["xp_a"]))
    vb = np.array(b["gold_b"]) - np.array(b["gold_a"]) + lam * (np.array(b["xp_b"]) - np.array(b["xp_a"]))
    return mean_ci(va - vb)


def reproduction_check(out: dict) -> Optional[float]:
    """Largest deviation of the xp_range 3 freeze cells from results/m43c/freeze.json (same seeds and starts)."""
    old = load("freeze")
    if not old or "freeze" not in out:
        return None
    dev = 0.0
    for key, cell in out["freeze"].items():
        sc, a, b, gap, r = key.split("|")
        okey = f"{sc}|{a}|{b}|{gap}"
        if r != "r3" or okey not in old["cells"]:
            continue
        for n in WINDOWS:
            o, c = old["cells"][okey][str(n)], cell["windows"][str(n)]
            for k in ("gold_a", "gold_b", "xp_a", "xp_b", "dmg_b", "deaths_b", "held"):
                dev = max(dev, abs(o[k][0] - c[k][0]))
    return dev


def report(out: Optional[dict] = None) -> str:
    out = out or load("xp_range")
    L = ["# M4.3c XP-range screen: tables", "",
         "Generated by `python -m lanegame.experiments.m43c xp_range` (see docs/m4/m4_3/m4_3c_xp_range_screen.md). "
         "Means ± 95% half-widths; B = the frozen-on champion; values from B's side; lambda = 1/8.", ""]
    dev = reproduction_check(out)
    if dev is not None:
        L += [f"Reproduction: the xp_range 3 cells deviate from results/m43c/freeze.json by at most {dev:.2e} "
              "(gold, XP, HP lost, deaths, held turns; windows 20 and 40).", ""]
    fr = out.get("freeze", {})
    n40, nx = str(max(WINDOWS)), str(max(WINDOWS) + EXT)
    for gap in GAPS:
        for a in FREEZERS + ("no_freeze",):
            resp = RESPONSES if a != "no_freeze" else NO_FREEZE_RESPONSES
            L += [f"## Freeze: A = {a}, gap {gap}", "",
                  "| B response | r | gold B−A (40) | XP B−A (20) | XP B−A (40) | XP B−A (60, after Hold/Hold ext.) "
                  "| B missed: range / away (40) | B pending (40) | A missed range (40) | B cell | B HP lost | B deaths "
                  "| B ganks | B trades | held /40 | value gold (40) | value gold+XP (40) | value gold+XP (60) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for b in resp:
                for r in RANGES:
                    c = fr.get(f"freeze|{a}|{b}|{gap}|r{r}")
                    if not c:
                        continue
                    w20, w40, wx = c["windows"]["20"], c["windows"][n40], c["windows"][nx]
                    L.append(f"| {b} | {r} | {_mc(w40['gold_diff_b'])} | {_f(w20['xp_diff_b'][0], 1)} "
                             f"| {_mc(w40['xp_diff_b'], 1)} | {_mc(wx['xp_diff_b'], 1)} "
                             f"| {_f(w40['missed_range_b'][0], -1)} / {_f(w40['missed_away_b'][0], -1)} "
                             f"| {_f(w40['pending_b'][0], -1)} | {_f(w40['missed_range_a'][0], -1)} "
                             f"| {_f(w40['pos_b'][0], 1)} | {_f(w40['dmg_b'][0], -2)} | {_f(w40['deaths_b'][0], -2)} "
                             f"| {_f(w40['ganks_b'][0], -1)} | {_f(w40['trades_b'][0], -1)} | {_f(w40['held'][0], -1)} "
                             f"| {_mc(w40['gold_diff_b'])} | {_mc(w40['value_b_gold_xp'])} | {_mc(wx['value_b_gold_xp'])} |")
            L.append("")
        # paired change of B's value relative to xp_range 3 (same seeds)
        L += [f"## Freeze, gap {gap}: change of B's 40-turn value relative to xp_range 3 (paired)", "",
              "| A | B response | Δ gold (r2) | Δ gold+XP (r2) | Δ gold (r1) | Δ gold+XP (r1) |", "|---|---|---|---|---|---|"]
        for a in FREEZERS + ("no_freeze",):
            for b in (RESPONSES if a != "no_freeze" else NO_FREEZE_RESPONSES):
                base = fr.get(f"freeze|{a}|{b}|{gap}|r3")
                if not base:
                    continue
                cells = []
                for r in (2, 1):
                    c = fr.get(f"freeze|{a}|{b}|{gap}|r{r}")
                    cells += [_mc(paired_diff(c["paired"][n40], base["paired"][n40])),
                              _mc(paired_diff(c["paired"][n40], base["paired"][n40], LAMBDA))]
                L.append(f"| {a} | {b} | " + " | ".join(cells) + " |")
        L.append("")
    # geometry: where B's ledger HP dies relative to B, by response (range-3 play), and the static re-pricing
    L += ["## Geometry: B's ledger HP by distance from B (present turns, 40-turn window)", "",
          "Share of the HP of A's minions that died while B was present, by |kill cell − B's cell|; the last columns "
          "re-price the SAME play at ranges 2 and 1 (the XP B would have lost) next to the measured loss.", "",
          "| A | gap | B response | r (play) | d=0 | d=1 | d=2 | d=3 | d≥4 | ledger HP | static loss r2 | static loss r1 "
          "| measured missed (r) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for gap in GAPS:
        for a in FREEZERS:
            for b in RESPONSES:
                for r in RANGES:
                    c = fr.get(f"freeze|{a}|{b}|{gap}|r{r}")
                    if not c:
                        continue
                    w = c["windows"][n40]
                    h = np.asarray(w["hist_b"], dtype=float) / FREEZE_EPISODES
                    sh = distance_shares(h)
                    tot = h.sum()
                    L.append(f"| {a} | {gap} | {b} | {r} | " + " | ".join(f"{x:.2f}" for x in sh)
                             + f" | {tot:.1f} | {tot * sum(sh[3:]):.1f} | {tot * sum(sh[2:]):.1f} "
                             f"| {w['missed_range_b'][0]:.1f} |")
    L.append("")
    L += ["## Joint histogram, B's ledger: (B cell, kill cell), A = freezer, gap equal, xp_range 3 play, 40 turns", ""]
    for b in RESPONSES:
        c = fr.get(f"freeze|freezer|{b}|equal|r3")
        if not c:
            continue
        h = np.asarray(c["windows"][n40]["hist_b"], dtype=float) / FREEZE_EPISODES
        rows_ = [e for e in range(h.shape[0]) if h[e].sum() > 0]
        cols = [x for x in range(h.shape[1]) if h[:, x].sum() > 0]
        L += [f"**{b}** (HP per game; rows B's cell, columns kill cell, B's own coordinates)", "",
              "| B cell \\ kill cell | " + " | ".join(f"{x - OFF:+d}" for x in cols) + " |",
              "|---|" + "---|" * len(cols)]
        for e in rows_:
            L.append(f"| {e - OFF:+d} | " + " | ".join(f"{h[e, x]:.1f}" for x in cols) + " |")
        L.append("")
    # neutral scripted
    L += ["## Neutral start (turn 9), scripted, 40 turns (per side: A / B)", "",
          "| A / B | r | XP | missed range | missed away | gold | XP B−A | level at 40 |", "|---|---|---|---|---|---|---|---|"]
    for a, b in NEUTRAL:
        for r in RANGES:
            c = fr.get(f"neutral|{a}|{b}|equal|r{r}")
            if not c:
                continue
            w = c["windows"][n40]
            L.append(f"| {a} / {b} | {r} | {w['xp_a'][0]:.1f} / {w['xp_b'][0]:.1f} "
                     f"| {w['missed_range_a'][0]:.1f} / {w['missed_range_b'][0]:.1f} "
                     f"| {w['missed_away_a'][0]:.1f} / {w['missed_away_b'][0]:.1f} "
                     f"| {w['gold_a'][0]:.2f} / {w['gold_b'][0]:.2f} | {_mc(w['xp_diff_b'], 1)} "
                     f"| {w['level_a'][0]:.1f} / {w['level_b'][0]:.1f} |")
    L.append("")
    gm = out.get("games", {})
    if gm:
        L += ["## Whole games from s0 (160 turns; per side pooled over seats)", "",
              "| matchup | r | XP/side @100 | missed range/side @100 (% of ledger) | missed away/side @100 | |XP diff| @100 "
              "| gold/side @100 | gold A−B @160 | level @50 | level @100 | turn L11 | turn L16 | d≥2 / d≥3 share (present) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, _, _ in GAMES:
            for r in RANGES:
                c = gm.get(f"{name}|r{r}")
                if not c:
                    continue
                s100, s50, s160 = c["summary"]["100"], c["summary"]["50"], c["summary"]["160"]
                fi = c["summary"]["first"]
                pool = lambda w, k: (w[f"{k}_a"][0] + w[f"{k}_b"][0]) / 2
                ledger = pool(s100, "avail")
                miss = pool(s100, "missed_range")
                sh = distance_shares(np.asarray(s100["hist_a"]) + np.asarray(s100["hist_b"]))
                L.append(f"| {name} | {r} | {pool(s100, 'xp'):.1f} | {miss:.1f} ({100 * miss / ledger:.1f}%) "
                         f"| {pool(s100, 'missed_away'):.1f} | {_mc(s100['abs_xp_diff'], 1)} | {pool(s100, 'gold'):.2f} "
                         f"| {_mc((s160['gold_a'][0] - s160['gold_b'][0], s160['gold_diff_b'][1]))} "
                         f"| {pool(s50, 'level'):.1f} | {pool(s100, 'level'):.1f} "
                         f"| {(fi['L11_a'][0] + fi['L11_b'][0]) / 2:.0f} | {(fi['L16_a'][0] + fi['L16_b'][0]) / 2:.0f} "
                         f"| {sum(sh[2:]):.3f} / {sum(sh[3:]):.3f} |")
        L.append("")
        L += ["## Whole games: joint histogram of missed-prone kills (both sides pooled, xp_range 3 play, 100 turns)", "",
              "HP per game at |kill cell − champion cell| ≥ 2, by (champion cell, kill cell), own coordinates.", ""]
        for name in ("base_s0 self", "base_s0 vs clear_punish", "farm vs farm"):
            c = gm.get(f"{name}|r3")
            if not c:
                continue
            h = (np.asarray(c["summary"]["100"]["hist_a"], dtype=float)
                 + np.asarray(c["summary"]["100"]["hist_b"], dtype=float)) / GAME_EPISODES
            items = sorted(((h[e, x], e - OFF, x - OFF) for e in range(h.shape[0]) for x in range(h.shape[1])
                            if abs(x - e) >= 2 and h[e, x] > 0), reverse=True)[:8]
            L.append(f"- **{name}**: " + "; ".join(f"cell {e:+d} / kill {x:+d}: {v:.1f}" for v, e, x in items))
        L.append("")
    text = "\n".join(L)
    write_md("xp_range", text)
    return text


__all__ = ["run", "report", "cfg_range", "run_turns", "window_stats", "distance_shares", "RANGES"]
