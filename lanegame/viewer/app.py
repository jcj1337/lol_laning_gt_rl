"""M4 lane debugger (M4.2e-D): a local Streamlit app over the real engine.

    streamlit run lanegame/viewer/app.py

Every rerun rebuilds the recording from (setup, actions) with session.play, which is cheap (well under a millisecond
per turn), so the page never holds a live simulation that could drift from the engine. Playback is a Streamlit
fragment that reruns on a timer while "playing" and advances the displayed turn (or tick) each time.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st  # noqa: E402

from lanegame.champions import CHAMP_ABSENT  # noqa: E402
from lanegame.config import (B_HOLD, B_RECALL, B_SAFE, B_TRADE, B_YIELD, BEHAVIOR_NAMES,  # noqa: E402
                             NONE, OBJECTIVE_NAMES, PRESSURE, VISION, W_HOLD, champ_act, champ_behavior,
                             champ_objective, wave_display_names)
from lanegame.viewer import policies as pl  # noqa: E402
from lanegame.viewer import render  # noqa: E402
from lanegame.viewer import session as ss  # noqa: E402

st.set_page_config(page_title="Lane debugger", layout="wide")
S = st.session_state
SPEEDS = {"slow": 1.6, "normal": 0.8, "fast": 0.35}        # seconds per playback step
for key, val in (("turn", 0), ("frame", 0), ("playing", False), ("speed", "normal"), ("animate_ticks", False),
                 ("_last_step", 0.0), ("manual", {"setup": None, "actions": []})):
    S.setdefault(key, val)
# Streamlit drops the state of widgets that are not rendered in a run; keep the fields across mode switches
for _k in ("matchup", "config", "scenario", "gap", "seed", "pa", "pb", "turns"):
    if _k in S:
        S[_k] = S[_k]

st.markdown("""<style>
.block-container {padding-top: 1.2rem; padding-bottom: 1rem;}
.lv-label {font-size: 0.75rem; color: #777; text-transform: uppercase; letter-spacing: 0.04em;}
.lv-value {font-size: 1.05rem; font-weight: 600; color: #222;}
.lv-big {font-size: 1.35rem; font-weight: 650; color: #222; line-height: 1.3;}
.lv-small {font-size: 0.85rem; color: #444;}
.lv-note {font-size: 0.78rem; color: #888;}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------- sidebar: setup
def apply_matchup():
    m = pl.MATCHUPS[S.matchup]
    S.config, S.scenario, S.gap, S.pa, S.pb, S.turns, S.turn, S.frame = (m.config, m.scenario, m.gap, m.pa, m.pb,
                                                                        m.turns, 0, 0)


def setup_widgets() -> ss.Setup:
    sb = st.sidebar
    cfg = sb.selectbox("Configuration", list(ss.CONFIGS), key="config")
    sc = sb.selectbox("Start scenario", list(ss.SCENARIOS), key="scenario")
    sb.caption(ss.SCENARIOS[sc])
    gap = sb.selectbox("Power gap (A over B)", list(ss.GAPS), key="gap", format_func=lambda g: f"{g}: {ss.GAPS[g]}")
    seed = int(sb.number_input("Seed (chance events: trades, ganks)", 0, 2 ** 31 - 1, key="seed", step=1))
    return ss.Setup(cfg, sc, gap, seed)


def policy_widget(label: str, key: str):
    name = st.sidebar.selectbox(label, list(pl.POLICIES), key=key)
    st.sidebar.caption(pl.POLICIES[name].description)
    return pl.POLICIES[name]


st.sidebar.title("Lane debugger")
mode = st.sidebar.radio("Mode", ["Strategy matchup", "Manual play", "Recorded trajectory"], key="mode")
recorded = None
if mode == "Recorded trajectory":
    folder = st.sidebar.text_input("Folder of recordings", os.path.join("results", "rl", "m43b", "trajectories"),
                                   key="rec_folder")
    files = sorted(f for f in os.listdir(folder) if f.endswith(".json")) if os.path.isdir(folder) else []
    if not files:
        st.sidebar.warning("No recordings found (export them with `python -m lanegame.experiments.rl.m43b trajectories`).")
        st.stop()
    fname = st.sidebar.selectbox("Recording", files, key="rec_file")
    setup, rec_actions, rec_meta = ss.load_recording(os.path.join(folder, fname))
    st.sidebar.caption(rec_meta.get("label", ""))
    st.sidebar.caption(f"A: {rec_meta['names'][0]} · B: {rec_meta['names'][1]} · seed {setup.seed} · "
                       f"{len(rec_actions)} turns")
    recorded = (rec_actions, rec_meta)
    pol_a = pol_b = None
    turns = 0
    title = f"Recorded: {rec_meta.get('label', fname)}"
elif mode == "Strategy matchup":
    if "matchup" not in S:
        S.matchup = next(iter(pl.MATCHUPS))
        apply_matchup()
    st.sidebar.selectbox("Example matchup (sets the fields below)", list(pl.MATCHUPS), key="matchup",
                         on_change=apply_matchup)
    st.sidebar.caption(pl.MATCHUPS[S.matchup].description)
    setup = setup_widgets()
    pol_a = policy_widget("Policy A (blue)", "pa")
    pol_b = policy_widget("Policy B (red)", "pb")
    turns = int(st.sidebar.number_input("Turns", 1, 500, key="turns"))
    title = S.matchup
else:
    S.setdefault("pb", "farm")
    setup = setup_widgets()
    pol_a, turns = None, 0
    pol_b = policy_widget("Opponent policy B (red)", "pb")
    man = S.manual
    if man["setup"] != setup:
        man["setup"], man["actions"], S.turn, S.frame = setup, [], 0, 0
    title = "Manual play: you are A (blue)"


def build() -> ss.Recording:
    if mode == "Recorded trajectory":
        return ss.play(setup, prefix=recorded[0], policies=(None, None), prefix_note="recorded")
    if mode == "Strategy matchup":
        return ss.play(setup, policies=(pol_a, pol_b), turns=turns)
    return ss.play(setup, prefix=S.manual["actions"], policies=(None, pol_b))


# ---------------------------------------------------------------------- navigation callbacks
def stop():
    S.playing = False


def go(i, f=0):
    stop()
    S.turn, S.frame = i, f


def toggle_play(n):
    if not S.playing and S.turn >= n:          # Play at the end restarts from the first turn
        S.turn, S.frame = 0, 0
    S.playing = not S.playing
    S._last_step = 0.0


def advance(rec: ss.Recording) -> None:
    """One playback step: the next tick (with the end of the turn folded into the next turn's start), or the next turn."""
    nf = ss.n_frames(rec, S.turn)
    if S.animate_ticks and S.frame < nf - 2:
        S.frame += 1
    elif S.turn < rec.n:
        S.turn, S.frame = S.turn + 1, 0
    if S.turn >= rec.n:
        S.playing = False


def kv(col, label, value, small=False):
    cls = "lv-small" if small else "lv-value"
    col.markdown(f"<div class='lv-label'>{label}</div><div class='{cls}'>{value}</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------- the page body (a fragment: reruns on a
# timer while playing; everything it shows is rebuilt from the recording on every run)
def body():
    rec = build()
    if S.playing and time.time() - S._last_step >= 0.8 * SPEEDS[S.speed]:
        advance(rec)
        S._last_step = time.time()
    S.turn = max(0, min(rec.n, S.turn))
    S.frame = max(0, min(ss.n_frames(rec, S.turn) - 1, S.frame))
    i, f = S.turn, S.frame
    view = ss.frame_view(rec, i, f)
    j = view["state_index"]                     # the champion / economy state shown
    info = rec.turns[i].info if i < rec.n else None
    cfg = rec.cfg

    # 1. matchup and turn
    st.markdown(f"<div class='lv-big'>{title}</div>"
                f"<div class='lv-small'>Turn {i} of {rec.n} · {view['frame_name']} · k = {view['k']} · "
                f"{ss.GAPS[setup.gap]} · seed {setup.seed}</div>", unsafe_allow_html=True)

    # 2. players
    if mode == "Recorded trajectory":
        pa_name, pb_name = recorded[1]["names"]
    else:
        pa_name, pb_name = (S.pa, S.pb) if mode == "Strategy matchup" else ("you", S.pb)
    tl = ss.timeline(rec)
    for col, s, name, pname in zip(st.columns(2), (0, 1), ("A", "B"), (pa_name, pb_name)):
        d = view["champions"][name]
        act = view["action_names"][s] if info is not None else "—"
        note = rec.turns[i].notes[s] if info is not None else ""
        away = f" · away for {d['absent']} more turn(s)" if d["absent"] else ""
        wv = (view.get("wave") or {}).get(name)
        dmg = ""
        if wv is not None and wv["budget"] is not None and not info["away"][s]:
            dmg = (f" · wave damage: budget {wv['budget']} HP" if wv["dealt"] is None else
                   f" · wave damage this {wv['frame']}: {wv['dealt']} HP (budget {wv['budget']})")
        col.markdown(
            f"<div class='lv-label'>Player {name} ({'blue' if s == 0 else 'red'}) · policy {pname}</div>"
            f"<div class='lv-big'>{act}</div>"
            f"<div class='lv-small'>HP {100 * d['hp'] / d['hp_max']:.0f}% · Level {d['level']} · "
            f"Attack power {d['attack_power']:.2f} · Gold {d['gold_earned']:.1f} earned, {d['gold_spendable']:.1f} "
            f"spendable, {d['invested']:.1f} invested{away}{dmg}</div><div class='lv-note'>{note}</div>",
            unsafe_allow_html=True)

    # 3. the lane
    st.plotly_chart(render.lane_figure(view, cfg), width="stretch", config={"displayModeBar": False})

    # 4. playback controls
    c = st.columns([0.8, 0.9, 1.1, 0.9, 0.8, 5.5, 1.4])
    c[0].button("⏮ Start", on_click=go, args=(0,), width="stretch")
    c[1].button("◀ Back", on_click=go, args=(i - 1,), width="stretch", disabled=i == 0)
    c[2].button("⏸ Pause" if S.playing else "▶ Play", on_click=toggle_play, args=(rec.n,), width="stretch",
                type="primary")
    c[3].button("Next ▶", on_click=go, args=(i + 1,), width="stretch", disabled=i >= rec.n)
    c[4].button("End ⏭", on_click=go, args=(rec.n,), width="stretch")
    if rec.n > 0:
        c[5].slider("Turn", 0, rec.n, key="turn", on_change=stop, label_visibility="collapsed")
    c[6].selectbox("Speed", list(SPEEDS), key="speed", label_visibility="collapsed")
    c2 = st.columns([3.2, 1.6, 4.2])
    nf = ss.n_frames(rec, i)
    if nf > 1:
        c2[0].radio("Within the turn", list(range(nf)), key="frame", horizontal=True, on_change=stop,
                    format_func=lambda x: ss.frame_name(rec, i, x), label_visibility="collapsed")
    c2[1].checkbox("Play tick by tick", key="animate_ticks")
    if mode == "Strategy matchup":
        def take_over():
            S.manual = {"setup": setup, "actions": list(rec.actions[:i])}
            S.mode, S.turn, S.frame, S.playing = "Manual play", i, 0, False
        c2[2].button(f"Take over A manually from turn {i}", on_click=take_over,
                     help="Keeps turns 0 to i-1 with the same seed; B keeps its policy.")

    # 5. summary
    adv = tl["advantage"][j]
    adv_s = "even" if abs(adv) < 0.005 else (f"+{adv:.1f} A" if adv > 0 else f"+{-adv:.1f} B")
    ev_turn = i - 1 if (f == 0 and i < rec.n) else min(i, rec.n - 1)
    events = ss.turn_events(rec.turns[ev_turn].info, cfg) if ev_turn >= 0 else []
    sc = st.columns([1.2, 2.6, 3.2])
    kv(sc[0], "Gold advantage", adv_s)
    kv(sc[1], "Waves", ss.wave_status(view), small=True)
    kv(sc[2], f"Events of turn {ev_turn}" if ev_turn >= 0 else "Events",
       "; ".join(events) if events else "nothing notable", small=True)
    st.caption("Bold rectangle: a side's leading minion packet (label = minion HP; one fresh wave = "
               f"{cfg.wave_hp} HP, so 12 is one and a half waves, not twelve minions). Light rectangle: reinforcements "
               "behind it. Dark 'crash': minions under the enemy tower. Circles: champions. Dashed line: the fronts "
               "are fighting. Arrow: the leading packet's movement since the previous frame. A walks right, B left.")

    # manual action selection
    if mode == "Manual play":
        manual_box(rec, i)

    # 6. details
    t1, tw, t2, t3, t4, t5 = st.tabs(["Champions", "Wave management", "Timelines", "Event log", "Turn details",
                                      "How to read this"])
    with t1:
        for col, s in zip(st.columns(2), (0, 1)):
            champion_panel(col, rec, j, s, info)
    with tw:
        wave_tab(rec, i)
    with t2:
        st.plotly_chart(render.timeline_figure(tl, i), width="stretch")
    with t3:
        st.dataframe([ss.turn_row(rec, x) for x in range(rec.n)], width="stretch", hide_index=True, height=420)
    with t4:
        turn_details(rec, i, info)
    with t5:
        st.markdown(HOW_TO_READ.format(W=cfg.wave_hp, T=cfg.tower))
    if S.playing != S._applied_playing:      # the play state changed in this fragment run: re-arm (or stop) the timer
        st.rerun()


def manual_box(rec: ss.Recording, i: int):
    def submit(code: int):
        b, _ = pol_b.decide(rec.cfg, rec.lanes[-1], rec.champs[-1], 1)     # B decides from the same pre-turn state
        S.manual["actions"] = list(rec.actions) + [(code, b)]
        S.turn, S.frame, S.playing = rec.n + 1, 0, False

    def branch():
        S.manual["actions"] = list(rec.actions[:i])

    def autoplay(name: str, n: int):
        r = ss.play(setup, prefix=rec.actions, policies=(pl.POLICIES[name], pol_b), turns=n)
        S.manual["actions"] = list(r.actions)
        S.turn, S.frame = r.n, 0

    def undo():
        S.manual["actions"] = list(rec.actions[:-1])
        S.turn, S.frame = max(0, rec.n - 1), 0

    box = st.container(border=True)
    if i < rec.n:
        box.info(f"Viewing turn {i} of {rec.n}; actions are chosen at the latest turn.")
        box.button(f"Branch here: discard turns {i} to {rec.n - 1} and continue from turn {i}", on_click=branch)
        return
    legal = rec.legal[-1][0]
    box.markdown(f"**Your move (A), turn {rec.n}.** B chooses from the same state; both actions resolve together.")
    if legal == (CHAMP_ABSENT,):
        box.button("Continue (A is away)", on_click=submit, args=(CHAMP_ABSENT,), type="primary")
    else:
        col = box.columns(3)
        names = wave_display_names(rec.cfg)                 # four-action model: Last Hit / Thin / Slow Push / Shove
        waves = [w for w in names if any(champ_act(w, b) in legal for b in (B_HOLD, B_YIELD, B_TRADE, B_SAFE))]
        budgets = ss.summary(rec, rec.n, 0).get("wave_budgets", {})
        wave = col[0].radio("Wave management", waves, key="m_wave",
                            format_func=lambda w: names[w] + (f" ({budgets[names[w]]} HP)" if names[w] in budgets
                                                              else ""))
        behs = [b for b in (B_HOLD, B_YIELD, B_TRADE, B_SAFE, B_RECALL) if any(champ_behavior(a) == b for a in legal)]
        beh = col[1].radio("Champion behavior", behs, format_func=BEHAVIOR_NAMES.get, key="m_beh")
        objs = [o for o in (NONE, PRESSURE, VISION) if any(champ_objective(a) == o for a in legal)]
        def obj_name(o):                                # M4.5a: Vision is the Ward at W (Last Hit + Hold only)
            if o == VISION and rec.cfg.vision_rule == "ward":
                return "Ward at W (Last Hit + Hold)"
            return OBJECTIVE_NAMES[o] or "none"
        obj = col[2].radio("Objective (Pressure needs a siege in reach)", objs, format_func=obj_name, key="m_obj")
        code = champ_act(W_HOLD, B_RECALL) if beh == B_RECALL else champ_act(wave, beh, obj)
        if code in legal:
            box.button(f"Play: {ss.action_name(rec.cfg, code)}", on_click=submit, args=(code,), type="primary")
        else:
            box.warning(f"{ss.action_name(rec.cfg, code)} is not legal here (Recall ignores wave and objective).")
        box.caption("Legal now: " + ", ".join(ss.action_name(rec.cfg, a) for a in legal))
    a1, a2, a3, a4 = box.columns([2, 1, 1, 1])
    ap = a1.selectbox("Let a policy play A for", list(pl.POLICIES), key="m_auto")
    nt = int(a2.number_input("turns", 1, 200, 5, key="m_auto_n"))
    a3.button("Auto-play", on_click=autoplay, args=(ap, nt))
    a4.button("Undo last", on_click=undo, disabled=rec.n == 0)


def champion_panel(col, rec: ss.Recording, j: int, s: int, info):
    n = ss.SIDES[s]
    d = ss.summary(rec, j, s)
    col.markdown(f"**Champion {n}**" + (f" — away ({d['absent']} turns)" if d["absent"] else ""))
    m = col.columns(4)
    m[0].metric("HP", f"{d['hp']}/{d['hp_max']}")
    m[1].metric("Level", d["level"], help=f"XP {d['xp']}, {d['xp_to_next']} to the next level")
    m[2].metric("Attack power", f"{d['attack_power']:.2f}", help=f"one trade hit deals {d['hit_damage']} centi-HP")
    m[3].metric("Tower damage dealt", f"{d['tower_damage_dealt']}", help=f"of {d['tower_damage_max']}")
    m = col.columns(4)
    m[0].metric("Income (cumulative)", f"{d['gold_earned']:.2f}", help="farm + plate gold + bounties since the start")
    m[1].metric("Spendable gold", f"{d['gold_spendable']:.2f}", help="unspent; invested at the next visit to base")
    m[2].metric("Invested", f"{d['invested']:.2f}", help="gold invested (includes a scenario's power gap)")
    m[3].metric("Plates broken", f"{d['plates_broken']}/{d['plates_total']}")
    if "wave_budgets" in d:
        cap = f" (capacity C = {d['capacity']})" if "capacity" in d else ""
        col.caption(f"Minion HP each wave action would remove this turn{cap}: "
                    + ", ".join(f"{k} {v}" for k, v in d["wave_budgets"].items()))
    vis = f" · vision {d['vision']} turn(s)" if d.get("vision") else ""
    col.caption(f"XP {d['xp']} ({d['xp_to_next']} to next) · own cell {ss.cell_text(d['pos_own'])} "
                f"(global {ss.cell_text(d['pos_global'])}){vis}")
    if info is not None:
        eff = (f"{ss.wave_name(rec.cfg, info['effective_waves'][s])} + {BEHAVIOR_NAMES[info['behaviors'][s]]}"
               + (f" + {OBJECTIVE_NAMES[info['effective_objectives'][s]]}" if info["effective_objectives"][s] else ""))
        col.markdown(f"Selected `{info['action_labels'][s]}` ({ss.action_name(rec.cfg, info['actions'][s])}) — "
                     f"effective {eff} · mode {info['modes'][s]} · own cell {ss.cell_text(info['champ_start'][s])} → "
                     f"{ss.cell_text(info['champ_end'][s])}")


def wave_tab(rec: ss.Recording, i: int):
    """Calculated against applied wave damage of turn i, with the lane numbers the budget comes from."""
    panel = ss.wave_panel(rec, i)
    if panel is None:
        st.write("Final state: no turn played from here.")
        return
    four = "need_thin" in panel["A"]
    rows = []
    for n in ss.SIDES:
        p = panel[n]
        row = {"side": n, "wave action": p["wave"], "capacity C": p["capacity"], "own blob": p["own_blob"],
               "enemy blob": p["enemy_blob"]}
        if four:
            row.update({"window strength own / enemy": f"{p['strength_own']:g} / {p['strength_enemy']:g}",
                        "Thin need": p["need_thin"], "Slow Push need": p["need_slowpush"]})
        row.update({"budget (calculated)": p["budget"], "removed per tick": ", ".join(map(str, p["dealt_per_tick"])),
                    "removed (applied)": p["dealt"], "mode": p["mode"], "reach (own)": p["reach"],
                    "note": p["note"]})
        rows.append(row)
    st.dataframe(rows, width="stretch", hide_index=True)
    st.caption(WAVE_HELP if four else "Budgets: Hold round(C/8), Thin round(C/2), Clear C (m4_lasthit shares); "
               "the lane step spends half of the budget per tick on the first reachable enemy minions.")


WAVE_HELP = (
    "Four-action budgets (computed once, from the lane at the start of the turn): Last Hit 1; Thin min(round(C/2), "
    "1 + Thin need); Slow Push min(round(C/2), 1 + Slow Push need); Shove C. The need is how much enemy window "
    "strength must go for parity (Thin) or a 1-HP lead (Slow Push); window strength counts the HP that will fight at a "
    "front over the next four ticks, reinforcements weighted by when they arrive. The lane step spends half of the "
    "budget per tick (the first tick rounds up) on the first reachable target: the enemy front in a fight with ours, "
    "else a crashed enemy mass under our tower, else the enemy's leading packet. Overkill is lost, so 'removed' can be "
    "smaller than the budget.")


def turn_details(rec: ss.Recording, i: int, info):
    if info is None:
        st.write("Final state: no turn played from here.")
        st.code(rec.lanes[i].describe() + "\n" + repr(rec.champs[i]))
        return
    sc = rec.cfg.gold_scale
    st.markdown("**Legal actions** — A: " + ", ".join(rec.cfg.label(a) for a in info["legal"][0])
                + "  \nB: " + ", ".join(rec.cfg.label(a) for a in info["legal"][1]))
    st.markdown("**Lane before / after** (own coordinates per side)")
    st.code(rec.lanes[i].describe() + "\n" + rec.lanes[i + 1].describe())
    st.markdown("**Champions before / after**")
    st.code(repr(rec.champs[i]) + "\n" + repr(rec.champs[i + 1]))
    ledger = {key: info[key] for key in ("farm_units", "plate_units", "bounty_units", "earned_units",
                                         "reward_units", "gold_unspendable", "invested")}
    ledger["gold_scale (units per resource)"] = sc
    groups = {
        "Actions": ["actions", "action_labels", "waves", "effective_waves", "behaviors", "objectives",
                    "effective_objectives", "modes", "lane_codes", "champion_reach", "farm_spot", "champ_start",
                    "champ_end", "pinned"],
        "Wave and minions": ["wave_budget", "events", "kills", "crash_hp", "thinned", "exposure", "tower_hits"],
        "Farm and XP": ["farm_available", "farm", "xp_gain", "xp_discarded", "levels_before", "levels", "level_up"],
        "Combat": ["trade_attempted", "trade_connected", "hit_p", "double_p", "trade_damage", "tower_p",
                   "adjacent_shot", "tower_damage", "damage_taken", "threatened"],
        "Ganks and deaths": ["gank_p", "gank_draws", "ganked", "gank_damage_p", "gank_damage", "died", "bounty",
                             "chance"],
        "Tower, recall, absence": ["tower_damage_dealt", "plates_broken", "plate_gold", "away", "recalled",
                                   "recall_interrupted", "returned", "purchases"],
        "M4.5a: Ward, minion targeting, trade budget": ["ward", "ward_placed", "ward_interrupted", "retaliation",
                                                        "minion_fire", "minion_hits", "minion_damage",
                                                        "trade_wave_capped"],
    }
    for g, keys in groups.items():
        with st.expander(g, expanded=g == "Actions"):
            st.json({k: info[k] for k in keys if k in info})
    with st.expander("Reward and economic ledger (integer units)"):
        st.json(ledger)
    with st.expander("Authoritative lane tick trace"):
        st.json(info.get("lane_trace"))
    with st.expander("Full info (raw)"):
        st.json({k: v for k, v in info.items() if k != "lane_trace"})


HOW_TO_READ = """
**The lane** is a line of cells. A's base is at −4 and its tower at −{T}; B's tower is at +{T} and its base at +4;
the cell 0 is the lane's midpoint. A's minions walk right, B's walk left. (Internally each side uses its own
mirrored coordinates; the picture always shows global ones.)

**Minions** come in packets. A packet is a group of minions measured by its total HP, not a count: one fresh wave
has {W} HP. The bold rectangle is a side's **leading packet** (its front), which does the fighting; the light
rectangles behind it are **reinforcements** (fresh waves walking up; they merge into the front when they reach it).
When the two fronts are within a cell of each other they fight (dashed line): each turn the bigger one deals more
damage, so a small edge grows (**slow push**). A front that reaches the enemy tower cell becomes a **crashed mass**
(dark rectangle) that the tower fires on while it damages the tower; when it is gone, the lane **bounces** back.

**Champions** (circles) choose each turn a wave action, a behaviour (Hold position, Yield, Trade, Safe, or Recall
near their tower) and, while sieging, an objective. Both choose at the same time from the same state. Wave actions in
the four-action model (`m4_four_actions`): **Last Hit** (1 HP of enemy minions per turn), **Thin** (last-hit, plus
only what removes the enemy's surplus), **Slow Push** (the least damage that puts our wave 1 HP ahead), **Shove**
(everything the champion can: its capacity C). In the three-action presets: Hold, Thin, Clear. The Wave management
tab shows, per side, what the action was allowed to spend (calculated) and what it actually removed (applied), tick by
tick. Levels, attack power, gold and investment are in the Champions tab; the Turn details tab shows every number the
engine reports for the turn.

**M4.5a (`m45_unified`)**: the green box above the lane cell 0 is **W**, the one off-lane Vision location both
champions share. **Ward** (from the own −1, 0, +1 or W) walks there and places a ward at the end of the turn (vision
for the next 6 turns, shown as "V6" on the champion) unless the enemy's Trade damages the warder that turn; a warder
deals no wave damage and collects no gold. A wave turns its fire on an enemy champion (red "minions" label on the
champion) on a tick after that champion's Trade damaged our champion (**retaliation**), or while that champion stands
in our front's cell ahead of its own wave (**positional aggro**); that tick the wave does not damage minions. A
champion's wave damage reaches only one cell around it; a connected trader still spends up to half its capacity on
minions.

**Playback**: a turn is two engine ticks. "Start" is the state both players saw when choosing; the ticks are the
engine's own intermediate lane states (champions are shown where they end the turn, since the engine moves them once
per turn, before the ticks); "end" is the state after the turn, including any new wave spawned at the bases.
"""


S._applied_playing = S.playing
run_body = st.fragment(run_every=SPEEDS[S.speed] if S.playing else None)(body)
run_body()
