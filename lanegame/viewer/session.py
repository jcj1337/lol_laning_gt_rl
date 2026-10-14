"""Simulation orchestration and trajectory storage for the viewer (M4.2e-D). No UI code.

A recording is a pure function of (Setup, action list): `play` resets a `LaneEnv` at the setup's start state with the
setup's seed and steps it; actions come from the given prefix first and then from the policies. The viewer never keeps
a live simulation: stepping back, scrubbing or branching only re-reads or re-plays, so it cannot consume randomness or
diverge from the engine. Every number shown is read from the environment's `info`, its states or the engine's
progression tables (`summary`, `turn_events`); nothing is recomputed.

Coordinates: lane and champion records are stored in each side's OWN coordinates; A's own coordinates are global and
B's are mirrored (global x = -own x). `lane_objects` and `champion_x` convert to global for rendering.

M4.3c stabilization: the EXPERIMENTAL four-action preset `m4_four_actions()` (Last Hit / Thin / Slow Push / Shove) is
the first configuration. Its start states are the `m4_lasthit()` states of the four-action redesign study (the lane is
the same physics; only the wave budgets differ), its wave budgets depend on the lane (`summary`, `wave_panel`), and
`wave_panel` shows each champion's calculated budget against the HP it actually removed, tick by tick.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

from ..champions_m4 import M4State, context, initial_m4
from ..config import (BEHAVIOR_NAMES, CHAMP_ABSENT, OBJECTIVE_NAMES, GameConfig, champ_behavior, champ_objective,
                      champ_wave, wave_display_names)
from ..env import LaneEnv
from ..experiments.m4 import natural_waves as nw
from ..experiments.m4 import zoning_natural as zn
from ..mechanics import _blob_hp, initial_state, start_packets, strength_need, window_strength
from ..progression import plates_broken
from ..state import State

A, B = 0, 1
SIDES = ("A", "B")

CONFIGS = {"m4_four_actions": GameConfig.m4_four_actions,
           # M4.4a: the two EXPERIMENTAL calibrated economies of the PPO comparison (recordings of their checkpoints
           # name them; lanegame/experiments/rl/m44/trace.py)
           "m4_four_actions_econ": GameConfig.m4_four_actions_econ,
           "m4_four_actions_econ_aggro": GameConfig.m4_four_actions_econ_aggro,
           "m4_lasthit": GameConfig.m4_lasthit, "m4_lane": GameConfig.m4_lane, "m4_natural": GameConfig.m4_natural}
# the configuration whose deterministic start states a configuration uses (the four-action redesign study generated
# its starts under m4_lasthit(); the lane physics is the same in every four-action preset)
START_SOURCE = {"m4_four_actions": "m4_lasthit", "m4_four_actions_econ": "m4_lasthit",
                "m4_four_actions_econ_aggro": "m4_lasthit"}
SCENARIOS = {
    "s0": "Standard initial state (turn 0: a fresh wave at each base, both champions at their tower).",
    "neutral": "Neutral lane, fronts in contact at the lane cell (k9 of Hold/Hold).",
    "pushed": "Established slow push: A's stack next to B's tower (A Thinned at k1, then Held).",
    "siege": "Active siege: A's crashed mass under B's tower, A in B's tower cell.",
    "freeze": "Outside-tower freeze: B's wave held at A's -1 cell by A's threshold freezer (both level 10).",
    "freeze_early": "The same freeze with both champions at level 4 (an early-laning freeze).",
    "freeze21": "The freeze with B's held front raised to 21 HP: an oversized wave at the freezer's tower.",
    "oversized": "The neutral lane with B's front raised to 24 HP: an oversized enemy wave at the lane cell.",
}
# scenario -> (base start, B front HP or None, both champions' level or None)
DERIVED = {"freeze_early": ("freeze", None, 4), "freeze21": ("freeze", 21, None), "oversized": ("neutral", 24, None)}
GAPS = {"equal": "equal power", "moderate": "A x1.24 attack power (16 vs 4 gold invested)",
        "large": "A x1.76 attack power (50 gold invested)", "levels": "A +3 levels",
        "large_levels": "A +3 levels and 50 gold invested (x2.04)",
        "b_large": "B x1.76 attack power (B 50 gold invested): A is the weak side"}


@dataclass(frozen=True)
class Setup:
    config: str = "m4_lasthit"
    scenario: str = "s0"
    gap: str = "equal"
    seed: int = 0


@lru_cache(maxsize=8)
def config(name: str) -> GameConfig:
    return CONFIGS[name]()


@lru_cache(maxsize=8)
def _starts(name: str) -> dict:
    return zn.starts(config(START_SOURCE.get(name, name)))


def at_level(m4: M4State, cfg: GameConfig, level: int) -> M4State:
    """Both champions at the first XP of `level`, with the full HP of that level."""
    t = context(cfg).t
    xp = t.xp_cum[level]
    return M4State(m4.a._replace(xp=xp, hp=t.hp_max[level]), m4.b._replace(xp=xp, hp=t.hp_max[level]))


def gapped(m4: M4State, cfg: GameConfig, gap: str) -> M4State:
    """`zoning_natural.with_gap` (A's lead), or its mirror for "b_large" (B 50 resources invested, A none)."""
    if gap == "equal":
        return m4
    if gap == "b_large":
        m = zn.with_gap(m4.mirror(), cfg, "large")
        return m.mirror()
    return zn.with_gap(m4, cfg, gap)


def start_state(setup: Setup) -> Tuple[State, M4State, int]:
    """(lane, champions, k0) of the setup. Non-s0 scenarios are the deterministic M4.2e-B/C start states
    (`zoning_natural.starts`, generated under START_SOURCE) or a DERIVED variant of one; the power gap is `gapped`
    (A's investment / level lead, or B's for "b_large")."""
    cfg = config(setup.config)
    if setup.scenario == "s0":
        lane, m4, k0 = initial_state(cfg), initial_m4(cfg), 0
    elif setup.scenario in DERIVED:
        base, hb, level = DERIVED[setup.scenario]
        lane, m4, k0 = _starts(setup.config)[base]
        if hb is not None:
            lane = lane._replace(bh=hb)
        if level is not None:
            m4 = at_level(m4, cfg, level)
    else:
        lane, m4, k0 = _starts(setup.config)[setup.scenario]
    return lane, gapped(m4, cfg, setup.gap), k0


@dataclass
class Turn:
    info: dict                    # LaneEnv's StepResult.info (with "lane_trace": the authoritative tick trace)
    notes: Tuple[str, str]        # why each side played its action (policy rule, fallback, "manual", "recorded")


@dataclass
class Recording:
    setup: Setup
    cfg: GameConfig
    lanes: List[State]            # lanes[i]: state at the START of turn i; lanes[-1] is the final state
    champs: List[M4State]
    legal: List[Tuple[tuple, tuple]]   # legal action codes of (A, B) at each state
    turns: List[Turn] = field(default_factory=list)
    k0: int = 0

    @property
    def n(self) -> int:
        return len(self.turns)

    @property
    def actions(self) -> List[Tuple[int, int]]:
        return [tuple(t.info["actions"]) for t in self.turns]

    def earned(self, i: int) -> Tuple[float, float]:
        """Cumulative gold income (farm + plates + bounties) of each side over turns 0 .. i-1, in resources."""
        sc = self.cfg.gold_scale
        return tuple(sum(t.info["earned_units"][s] for t in self.turns[:i]) / sc for s in (A, B))


def play(setup: Setup, prefix: Sequence[Tuple[int, int]] = (), policies=(None, None), turns: int = 0,
         prefix_note: str = "manual") -> Recording:
    """Replay `prefix` (pairs of action codes) from the setup's start and seed, then play `turns` more turns with the
    policies. A policy decides from the same pre-turn state for both sides (simultaneous moves) and the turn index i
    of the recording (turn schedules of lanegame.experiments.m43c.strategies; every other policy ignores it). Illegal
    actions raise (the environment's own check). `prefix_note` labels prefix actions of a side without a policy
    ("manual", or "recorded" for an exported trajectory)."""
    cfg = config(setup.config)
    lane, m4, k0 = start_state(setup)
    env = LaneEnv(cfg, record_ticks=True)
    env.reset(seed=setup.seed, root=lane, champ=m4, k0=k0)
    rec = Recording(setup, cfg, [env.state], [env.champ], [(env.legal(A), env.legal(B))], k0=env.k)
    for i in range(len(prefix) + turns):
        pair, notes = [0, 0], ["", ""]
        for s in (A, B):
            pol = policies[s]
            if i < len(prefix):
                pair[s] = int(prefix[i][s])
                if pol is None:
                    notes[s] = prefix_note
                else:
                    code, why = pol.decide(cfg, env.state, env.champ, s, i)
                    notes[s] = why if code == pair[s] else "recorded"
            else:
                pair[s], notes[s] = policies[s].decide(cfg, env.state, env.champ, s, i)
        r = env.step(*pair)
        rec.turns.append(Turn(r.info, tuple(notes)))
        rec.lanes.append(env.state)
        rec.champs.append(env.champ)
        rec.legal.append((env.legal(A), env.legal(B)))
        if r.terminated:                     # an EXPERIMENTAL terminal_rule ended the game: nothing follows
            break
    return rec


RECORDING_SCHEMA = "lanegame.viewer_recording"


def load_recording(path: str) -> Tuple[Setup, List[Tuple[int, int]], dict]:
    """An exported trajectory (lanegame.experiments.rl.m43b.trajectories): (Setup, actions, metadata). Replaying it
    with `play(setup, prefix=actions, prefix_note="recorded")` reproduces the recorded game exactly, because the
    recording was played from the same start with the same integer chance seed."""
    import json
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if d.get("schema") != RECORDING_SCHEMA:
        raise ValueError(f"{path} is not a {RECORDING_SCHEMA} file")
    setup = Setup(d.get("config", "m4_lasthit"), d.get("scenario", "s0"), d.get("gap", "equal"), int(d["seed"]))
    actions = [(int(a), int(b)) for a, b in d["actions"]]
    meta = {k: v for k, v in d.items() if k != "actions"}
    return setup, actions, meta


# ---------------------------------------------------------------------- global coordinates (rendering)
def packets(lane: State, s: int) -> List[Tuple[int, int, bool]]:
    """(own x, HP, is_front) of side s's walking packets."""
    fx, fh, rears = lane.side(s)[:3]
    out = [(fx, fh, True)] if fh > 0 else []
    return out + [(x, -1, False) for x in rears]          # rear HP filled in by lane_objects (always one wave)


def lane_objects(lane: State, cfg: GameConfig) -> dict:
    """Global-coordinate drawing data of a lane state: per side, packets [(x, hp, front)], and the crashed mass
    under the ENEMY tower as (x, hp)."""
    out = {}
    for s, sign in ((A, 1), (B, -1)):
        pk = [(sign * x, h if h > 0 else cfg.wave_hp, f) for x, h, f in packets(lane, s)]
        out[SIDES[s]] = {"packets": pk, "siege": (sign * cfg.tower, lane.siege_of(s))}
    return out


def trace_objects(tick: dict, cfg: GameConfig) -> dict:
    """The same drawing data for one entry of the authoritative tick trace (packets in own coordinates; the front is
    the most advanced packet, as in state.make_side)."""
    out = {}
    for s, sign in ((A, 1), (B, -1)):
        pk = sorted(tick[SIDES[s]], key=lambda p: -p[0])
        pk = [(sign * x, h, j == 0) for j, (x, h) in enumerate(pk)]
        out[SIDES[s]] = {"packets": pk, "siege": (sign * cfg.tower, tick["siege"][s])}
    return out


def champion_x(m4: M4State, s: int) -> int:
    return m4.side(s).pos if s == A else -m4.side(s).pos


def contact_x(lane: State) -> Optional[float]:
    """Global position of the contact between the two fronts (midway between them), None when not in contact."""
    if nw.contact_cell(lane) is None:
        return None
    return (lane.ax - lane.bx) / 2


# ---------------------------------------------------------------------- champion and economy summaries
def summary(rec: Recording, i: int, s: int) -> dict:
    """Champion s at the start of turn i (i = rec.n: the final state), from the state and the progression tables."""
    cfg = rec.cfg
    ctx = context(cfg)
    t, sc = ctx.t, cfg.gold_scale
    me = rec.champs[i].side(s)
    L = t.level_by_xp[min(me.xp, t.xp_cap)]
    if cfg.continuous_items:
        invested = me.items / sc
        ap = ctx.inv.attack_power(L, me.items) / (10_000 * ctx.inv.one)
    else:
        invested = sum(t.prices[:me.items]) / sc
        ap = t.ap[L][me.items] / 10_000
    xp_next = t.xp_cum[L + 1] - me.xp if L < t.level_cap else 0
    d = {"level": L, "hp": me.hp, "hp_max": t.hp_max[L], "xp": me.xp, "xp_to_next": xp_next,
         "gold_spendable": me.gold / sc, "gold_earned": rec.earned(i)[s], "invested": invested,
         "attack_power": ap, "hit_damage": ctx.hit(1, L, me.items), "absent": me.absent, "pos_own": me.pos,
         "pos_global": champion_x(rec.champs[i], s),
         "tower_damage_dealt": me.tower_dmg, "tower_damage_max": cfg.plate_thresholds[-1],
         "plates_broken": plates_broken(me.tower_dmg, t), "plates_total": len(cfg.plate_thresholds)}
    names = wave_display_names(cfg)
    if cfg.natural_waves and cfg.wave_clear_scaling == "capacity":
        # what each wave action would spend from THIS state (the four-action Thin and Slow Push depend on the lane)
        C = ctx.capacity(L, me.items)
        d["capacity"] = C
        d["wave_budgets"] = {names[slot]: ctx.state_budget(slot, C, rec.lanes[i], rec.k0 + i, s) for slot in names}
    elif cfg.natural_waves and cfg.wave_clear_scaling != "none":
        d["wave_budgets"] = {names[slot]: ctx.wave_budget(slot, L, me.items) for slot in names}
    return d


def timeline(rec: Recording) -> Dict[str, list]:
    """Per state i = 0 .. n: turn, cumulative income of both sides and their difference, HP fraction, level, XP,
    spendable gold, invested gold, contact position."""
    out = {key: [] for key in ("i", "k", "earned_A", "earned_B", "advantage", "hp_A", "hp_B", "level_A", "level_B",
                               "xp_A", "xp_B", "gold_A", "gold_B", "invested_A", "invested_B", "contact")}
    for i in range(rec.n + 1):
        out["i"].append(i)
        out["k"].append(rec.k0 + i)
        for s in (A, B):
            d = summary(rec, i, s)
            n = SIDES[s]
            out[f"earned_{n}"].append(d["gold_earned"])
            out[f"hp_{n}"].append(d["hp"] / d["hp_max"])
            out[f"level_{n}"].append(d["level"])
            out[f"xp_{n}"].append(d["xp"])
            out[f"gold_{n}"].append(d["gold_spendable"])
            out[f"invested_{n}"].append(d["invested"])
        out["advantage"].append(out["earned_A"][-1] - out["earned_B"][-1])
        out["contact"].append(contact_x(rec.lanes[i]))
    return out


# ---------------------------------------------------------------------- names
def action_name(cfg: GameConfig, code: int) -> str:
    """Readable action name: 'Thin + Hold + Pressure', 'Slow Push + Trade', 'Shove + Hold' (four-action model),
    'Recall', 'Absent' (the engine's own labels are 'TN-H+P', 'SL-T', 'SH-H')."""
    if code == CHAMP_ABSENT:
        return "Absent"
    beh, obj = champ_behavior(code), champ_objective(code)
    if BEHAVIOR_NAMES[beh] == "Recall":
        return "Recall"
    wave = wave_display_names(cfg).get(champ_wave(code), cfg.label(code).split("-")[0])
    return f"{wave} + {BEHAVIOR_NAMES[beh]}" + (f" + {OBJECTIVE_NAMES[obj]}" if obj else "")


def wave_name(cfg: GameConfig, slot: int) -> str:
    return wave_display_names(cfg).get(slot, str(slot))


# ---------------------------------------------------------------------- wave management of a turn (what each champion
# was allowed to spend on enemy minions, and what it actually removed)
def wave_panel(rec: Recording, i: int) -> Optional[Dict[str, dict]]:
    """Per side, for turn i (None at the final state): the chosen wave action, the capacity C and the start-of-turn
    lane numbers its budget is computed from (blobs; window strengths and the Thin / Slow Push needs of the
    four-action model), the budget the engine gave the lane step (`info["wave_budget"]`, the CALCULATED damage), the
    HP of enemy minions the champion actually removed, per tick and in total (the authoritative tick trace and
    `events["champion_hp"]`: the APPLIED damage), and why any budget stayed unspent."""
    if i >= rec.n:
        return None
    cfg, info = rec.cfg, rec.turns[i].info
    lane, k, m4 = rec.lanes[i], rec.k0 + i, rec.champs[i]
    ctx = context(cfg)
    P = start_packets(lane, cfg)
    four = cfg.natural_waves and cfg.wave_thin_rule == "strength"
    S = [window_strength(P[s], k, cfg) for s in (A, B)] if four else None
    trace = info.get("lane_trace") or []
    out = {}
    for s in (A, B):
        n = SIDES[s]
        budget = info["wave_budget"][s] if info.get("wave_budget") is not None else None
        dealt = info["events"]["champion_hp"][1 - s] if "champion_hp" in info["events"] else None
        ticks = [tk["champion"][s] for tk in trace] if trace else []
        mode = info["modes"][s]
        if info["away"][s]:
            why = "away (no wave damage)"
        elif mode in ("trading", "channel"):
            why = f"{'connected trade' if mode == 'trading' else 'interrupted Recall'}: no wave damage this turn"
        elif budget is not None and dealt is not None and dealt < budget:
            why = (f"{budget - dealt} HP unspent: no reachable enemy minion (reach line own "
                   f"{info['champion_reach'][s]:+d}) or the target died first (overkill is lost)")
        else:
            why = "spent in full" if budget else ""
        L = info["levels_before"][s]
        d = {"wave": wave_name(cfg, info["waves"][s]) if not info["away"][s] else "—",
             "capacity": ctx.capacity(L, m4.side(s).items) if cfg.wave_clear_scaling == "capacity" else None,
             "own_blob": _blob_hp(P[s], cfg), "enemy_blob": _blob_hp(P[1 - s], cfg),
             "budget": budget, "dealt": dealt, "dealt_per_tick": ticks, "reach": info["champion_reach"][s],
             "mode": mode, "note": why}
        if four:
            d["strength_own"] = float(S[s])
            d["strength_enemy"] = float(S[1 - s])
            d["need_thin"] = strength_need(P, s, k, 0, cfg)
            d["need_slowpush"] = strength_need(P, s, k, 1, cfg)
        out[n] = d
    return out


# ---------------------------------------------------------------------- frames of a turn (what the picture shows)
def n_frames(rec: Recording, i: int) -> int:
    """Frames of turn i: the start state, one per authoritative tick, the end state (1 for the final state)."""
    return 1 if i >= rec.n else rec.cfg.ticks_per_turn + 2


def frame_name(rec: Recording, i: int, f: int) -> str:
    if i >= rec.n:
        return "final state"
    return "start" if f == 0 else ("end" if f == rec.cfg.ticks_per_turn + 1 else f"tick {f}")


def fronts(objs: dict) -> Dict[str, Optional[int]]:
    """Global x of each side's leading packet (None without one)."""
    return {n: next((x for x, _, f in objs[n]["packets"] if f), None) for n in SIDES}


def contact_of(objs: dict) -> Optional[float]:
    """Global contact position: the two fronts fight when B's is at most one cell beyond A's (mechanics'
    `own_fx + en_fx >= -1`); the marker sits midway between them."""
    fa, fb = fronts(objs)["A"], fronts(objs)["B"]
    if fa is None or fb is None or fb - fa > 1:
        return None
    return (fa + fb) / 2


def _lost(info: dict) -> Dict[str, bool]:
    """Per side: the turn's lane events report one of its packets dying or crashing."""
    return {n: bool(info["events"]["died"][s] or info["crash_hp"][s]) for s, n in enumerate(SIDES)}


def frame_view(rec: Recording, i: int, f: int) -> dict:
    """Everything the lane picture draws for frame f of turn i, in global coordinates, read from the recording:
    packets and crashed masses (lane state, or the authoritative tick trace), champions and towers (the champion
    state at the start of the turn for the start frame, at its end otherwise: the resolver moves the champions
    once, before the lane ticks), the contact, and each front's displacement since the previous frame (the start
    frame: since the previous turn's start; the end frame: over the whole turn)."""
    cfg, T = rec.cfg, rec.cfg.ticks_per_turn
    lost = None                                 # per side: a packet died or crashed between prev and objs
    if i >= rec.n or f == 0:
        objs, j = lane_objects(rec.lanes[i], cfg), i
        prev = lane_objects(rec.lanes[i - 1], cfg) if i > 0 else None
        if i > 0:
            lost = _lost(rec.turns[i - 1].info)
    else:
        tr = rec.turns[i].info["lane_trace"]
        j = i + 1
        if f <= T:
            objs = trace_objects(tr[f - 1], cfg)
            prev = trace_objects(tr[f - 2], cfg) if f >= 2 else lane_objects(rec.lanes[i], cfg)
            lost = {n: len(objs[n]["packets"]) < len(prev[n]["packets"]) for n in SIDES}   # no spawn within a tick
        else:
            objs, prev = lane_objects(rec.lanes[i + 1], cfg), lane_objects(rec.lanes[i], cfg)
            lost = _lost(rec.turns[i].info)
    moves = {}
    for n in SIDES:
        now, before = fronts(objs)[n], (fronts(prev)[n] if prev is not None else None)
        # an arrow only while the leading packet is the same one: nothing of that side died or crashed meanwhile
        moves[n] = (before, now) if (prev is not None and not lost[n] and now is not None and before is not None
                                     and now != before) else None
    champs = {n: summary(rec, j, s) for s, n in enumerate(SIDES)}
    towers = {"A": champs["B"], "B": champs["A"]}          # A's tower carries the damage B dealt, and vice versa
    actions = tuple(rec.turns[i].info["actions"]) if i < rec.n else None
    # the champions' wave damage shown with the frame: the turn's budget (start frame), the HP removed in that tick
    # (tick frames) or over the whole turn (end frame)
    wave = None
    panel = wave_panel(rec, i)
    if panel is not None:
        wave = {}
        for s, n in enumerate(SIDES):
            p = panel[n]
            if f == 0:
                shown = None
            elif f <= T:
                shown = p["dealt_per_tick"][f - 1] if p["dealt_per_tick"] else None
            else:
                shown = p["dealt"]
            wave[n] = {"wave": p["wave"], "budget": p["budget"], "dealt": shown, "frame": "budget" if f == 0 else
                       ("tick" if f <= T else "turn")}
    return {"turn": i, "k": rec.k0 + i, "frame": f, "frame_name": frame_name(rec, i, f), "state_index": j,
            "packets": {n: objs[n]["packets"] for n in SIDES}, "siege": {n: objs[n]["siege"] for n in SIDES},
            "contact": contact_of(objs), "moves": moves, "champions": champs,
            "towers": {n: {"damage": towers[n]["tower_damage_dealt"], "damage_max": towers[n]["tower_damage_max"],
                           "plates": towers[n]["plates_broken"], "plates_total": towers[n]["plates_total"]}
                       for n in SIDES},
            "actions": actions,
            "action_names": tuple(action_name(cfg, a) for a in actions) if actions else ("", ""),
            "wave": wave}


def wave_status(view: dict) -> str:
    """One line on the waves: where the fronts are, whether they fight, what is under a tower."""
    fa, fb = fronts({n: {"packets": view["packets"][n]} for n in SIDES}).values()
    if fa is None and fb is None:
        s = "no wave in the lane"
    elif fa is None:
        s = f"only B's wave (front at {fb:+d})"
    elif fb is None:
        s = f"only A's wave (front at {fa:+d})"
    elif view["contact"] is not None:
        c = view["contact"]
        where = "A's side" if c < 0 else ("B's side" if c > 0 else "the midpoint")
        s = f"fighting at {c:+g} ({where})"
    else:
        s = f"separated: A's front at {fa:+d}, B's at {fb:+d}"
    for n, o in (("A", "B"), ("B", "A")):
        hp = view["siege"][n][1]
        if hp:
            s += f"; {n}'s crash under {o}'s tower ({hp} HP)"
    return s


# ---------------------------------------------------------------------- events of a turn (read from info)
def turn_events(info: dict, cfg: GameConfig) -> List[str]:
    """Human-readable events of one turn, in resolution order, from the environment's info only."""
    sc = cfg.gold_scale
    ev: List[str] = []
    for s in (A, B):
        n, o = SIDES[s], SIDES[1 - s]
        wb = info.get("wave_budget")
        if wb is not None and not info["away"][s] and "champion_hp" in info["events"]:
            dealt = info["events"]["champion_hp"][1 - s]
            ev.append(f"{n} {wave_name(cfg, info['waves'][s])}: removed {dealt} of {wb[s]} HP")
        if info["recalled"][s]:
            ev.append(f"{n} recalled")
        if info["recall_interrupted"][s]:
            ev.append(f"{n}'s recall was interrupted")
        if info["trade_attempted"][s]:
            ev.append(f"{n}'s trade connected: dealt {info['trade_damage'][s]} cHP" if info["trade_connected"][s]
                      else f"{n} tried to trade: no contact")
        if info["tower_damage"][s]:
            ev.append(f"{o}'s tower shot {n} for {info['tower_damage'][s]} cHP")
        if info["ganked"][s]:
            ev.append(f"{n} was ganked (p={info['gank_p'][s]:.2f}): {info['gank_damage'][s]} cHP")
        if info["died"][s]:
            ev.append(f"{n} DIED ({o} bounty {info['bounty'][1 - s]:.2f})")
        if info["events"]["merged"][s]:
            ev.append(f"{n}'s reinforcement merged into its front")
        if info["events"]["died"][s]:
            ev.append(f"{n} lost {info['events']['died'][s]} packet(s) ({info['events']['killed_hp'][s]} HP)")
        if info["crash_hp"][s]:
            ev.append(f"{n}'s wave crashed into {o}'s tower ({info['crash_hp'][s]} HP)")
        if info["state_before"][3 + 4 * s] > 0 and info["state_after"][3 + 4 * s] == 0:
            ev.append(f"{n}'s siege under {o}'s tower ended (the lane bounces back)")
        if info["tower_damage_dealt"][s]:
            ev.append(f"{n} damaged {o}'s tower: +{info['tower_damage_dealt'][s]}")
        if info["plates_broken"][s]:
            ev.append(f"{n} broke {info['plates_broken'][s]} plate(s): +{info['plate_units'][s] / sc:.2f} gold")
        if info["level_up"][s]:
            ev.append(f"{n} reached level {info['levels'][s]}")
        if info["invested"][s] or info["purchases"][s]:
            ev.append(f"{n} invested {info['invested'][s] / sc:.2f} gold at base")
        if info["returned"][s]:
            ev.append(f"{n} returns next turn")
    return ev


def turn_row(rec: Recording, j: int) -> dict:
    """One row of the event log for turn j."""
    info, cfg = rec.turns[j].info, rec.cfg
    sc = cfg.gold_scale
    return {"turn": j, "k": info["k"], "A action": action_name(cfg, info["actions"][0]),
            "B action": action_name(cfg, info["actions"][1]),
            "A gold": round(info["earned_units"][0] / sc, 3), "B gold": round(info["earned_units"][1] / sc, 3),
            "A xp": info["xp_gain"][0], "B xp": info["xp_gain"][1], "events": "; ".join(turn_events(info, cfg))}
