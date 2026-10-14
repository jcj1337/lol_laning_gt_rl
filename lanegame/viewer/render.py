"""Plotly figures for the viewer (M4.2e-D). Rendering only: every value comes from session.frame_view / timeline.

The lane picture is a compact horizontal lane in GLOBAL coordinates (A's tower at -T, the lane cell at 0, B's tower
at +T; A walks right, B walks left). Blue is A, red is B. On the lane bar, A's packets sit on the upper row and B's
on the lower row: a bold rectangle is a side's leading packet, labelled with its minion HP (a packet is a group of
minions measured by HP, one fresh wave = cfg.wave_hp; it is not a minion count), lighter rectangles are the
reinforcements behind it (always one fresh wave each; "2x8" = two of them in one cell), and a dark rectangle at a
tower is that side's crashed mass under the enemy tower. Circles above the lane are the champions. A dashed line
marks the contact between the fronts, and a short arrow the leading packet's displacement since the previous frame.

M4.5a (vision_rule "ward"): the shared Vision node W is a green box above the lane cell 0, joined to it by a dashed
edge; a champion standing there is drawn inside it. A champion's label carries its vision timer ("V5") while its vision
is active, "ward placed" / "ward interrupted" on its Ward turn, and the minion fire it takes in the frame (cause and
minion damage units; the end frame adds the turn's centi-HP).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLOR = {"A": "#2f6fd6", "B": "#d64545"}
LIGHT = {"A": "#b7cdef", "B": "#f0b9b5"}
DARK = {"A": "#1b4a9e", "B": "#a12b26"}
TEXT_ON_LIGHT = {"A": "#1c3f80", "B": "#7a211c"}
TOWER = {"A": "#4d88dc", "B": "#dc5a55"}
SIGN = {"A": 1, "B": -1}
ROW = 0.29                 # |y| of a side's packet row (A above the axis, B below)
BAR = 0.6                  # half-height of the lane bar
XR, YR = (-4.9, 4.9), (-1.35, 2.05)
HEIGHT = 330
YR_WARD, HEIGHT_WARD = (-1.35, 3.25), 410      # M4.5a: room for the Vision node W above the lane
Y_LANE_CHAMP, Y_WARD = 1.5, 2.7
WARD_FILL, WARD_LINE, MINION = "#e3f1dc", "#6aa84f", "#8b1a1a"


def _rrect(x0, y0, x1, y1, rx=0.07, ry=0.11) -> str:
    rx, ry = min(rx, (x1 - x0) / 2), min(ry, (y1 - y0) / 2)
    return (f"M{x0 + rx},{y0} L{x1 - rx},{y0} Q{x1},{y0} {x1},{y0 + ry} L{x1},{y1 - ry} Q{x1},{y1} {x1 - rx},{y1} "
            f"L{x0 + rx},{y1} Q{x0},{y1} {x0},{y1 - ry} L{x0},{y0 + ry} Q{x0},{y0} {x0 + rx},{y0} Z")


def _shape(x0, y0, x1, y1, fill, line="white", width=1.5, layer="above", **kw) -> dict:
    return dict(type="path", path=_rrect(x0, y0, x1, y1, **kw), fillcolor=fill, line=dict(color=line, width=width),
                layer=layer)


def _text(x, y, text, size=12, color="#222", bold=False, **kw) -> dict:
    d = dict(x=x, y=y, text=f"<b>{text}</b>" if bold else text, showarrow=False, font=dict(size=size, color=color),
             xanchor="center", yanchor="middle")
    d.update(kw)
    return d


def _packet_items(view: dict, name: str, wave_hp: int) -> List[dict]:
    """Rectangles of one side: the front, reinforcement groups per cell, the crashed mass; stacked outward when two
    share a cell (a reinforcement queued in the front's cell sits just outside the bar)."""
    items = []
    for x, hp, front in view["packets"][name]:
        if front:
            w = max(0.5, min(0.98, 0.34 + 0.027 * hp))
            items.append(dict(x=x, w=w, h=0.46, fill=COLOR[name], text=str(hp), color="white", size=14, bold=True,
                              hover=f"{name}'s leading packet: {hp} HP ({hp / wave_hp:.2g} fresh waves) at global {x:+d}"))
    rears: Dict[int, int] = {}
    for x, hp, front in view["packets"][name]:
        if not front:
            rears[x] = rears.get(x, 0) + 1
    for x, n in rears.items():
        items.append(dict(x=x, w=0.44, h=0.32, fill=LIGHT[name], text=f"{n}×{wave_hp}" if n > 1 else str(wave_hp),
                          color=TEXT_ON_LIGHT[name], size=11, bold=False,
                          hover=f"{name}'s reinforcement{'s' if n > 1 else ''}: {n} fresh wave{'s' if n > 1 else ''} "
                                f"of {wave_hp} HP at global {x:+d}"))
    x, hp = view["siege"][name]
    if hp > 0:
        items.append(dict(x=x, w=max(0.62, min(1.25, 0.4 + 0.016 * hp)), h=0.46, fill=DARK[name], text=f"crash {hp}",
                          color="white", size=11, bold=True,
                          hover=f"{name}'s crashed minions under the enemy tower: {hp} HP (the tower fires on them)"))
    # stack items that share a cell outward from the lane axis
    seen: Dict[int, int] = {}
    for it in items:
        k = seen.get(it["x"], 0)
        seen[it["x"]] = k + 1
        it["y"] = SIGN[name] * (ROW + k * 0.42)
    return items


def lane_figure(view: dict, cfg) -> go.Figure:
    T, base, W = cfg.tower, cfg.base, cfg.wave_hp
    shapes, ann, hover_x, hover_y, hover_t = [], [], [], [], []
    # the lane bar and its cells
    shapes.append(_shape(-base - 0.55, -BAR, base + 0.55, BAR, "#dedede", line="#dedede", width=0, layer="below",
                         rx=0.12, ry=0.18))
    for x in range(-base, base + 1):
        shapes.append(dict(type="line", x0=x, x1=x, y0=-BAR, y1=-BAR + 0.1, line=dict(color="#bdbdbd", width=1),
                           layer="below"))
        shapes.append(dict(type="line", x0=x, x1=x, y0=BAR - 0.1, y1=BAR, line=dict(color="#bdbdbd", width=1),
                           layer="below"))
        ann.append(_text(x, -0.95, f"{x:+d}" if x else "0", size=13, color="#444"))
    for x, label in ((-base, "A base"), (-T, "A tower"), (0, "lane"), (T, "B tower"), (base, "B base")):
        ann.append(_text(x, -1.2, label, size=10, color="#8a8a8a"))
    # towers (A's at -T, B's at +T) with the damage they have taken
    for name, x in (("A", -T), ("B", T)):
        tw = view["towers"][name]
        shapes.append(_shape(x - 0.3, -0.72, x + 0.3, 0.72, TOWER[name], line="white", width=2, layer="below",
                             rx=0.08, ry=0.14))
        ann.append(_text(x, 0.86, f"<b>{name} tower</b><br><span style='font-size:10px;color:#777'>damage "
                                 f"{tw['damage']}/{tw['damage_max']} · plates {tw['plates']}/{tw['plates_total']}"
                                 f"</span>", size=12, color="#222", yanchor="bottom"))
        hover_x.append(x)
        hover_y.append(0.0)
        hover_t.append(f"{name}'s tower: {tw['damage']} of {tw['damage_max']} damage taken, {tw['plates']} of "
                       f"{tw['plates_total']} plates broken")
    # contact between the fronts
    if view["contact"] is not None:
        shapes.append(dict(type="line", x0=view["contact"], x1=view["contact"], y0=-BAR, y1=BAR,
                           line=dict(color="#555", width=1.5, dash="dash"), layer="below"))
    # packets
    for name in ("A", "B"):
        for it in _packet_items(view, name, W):
            shapes.append(_shape(it["x"] - it["w"] / 2, it["y"] - it["h"] / 2, it["x"] + it["w"] / 2,
                                 it["y"] + it["h"] / 2, it["fill"]))
            ann.append(_text(it["x"], it["y"], it["text"], size=it["size"], color=it["color"], bold=it["bold"]))
            hover_x.append(it["x"])
            hover_y.append(it["y"])
            hover_t.append(it["hover"])
        mv = view["moves"][name]
        if mv is not None:
            y = SIGN[name] * (BAR + 0.17)
            ann.append(dict(x=mv[1], y=y, ax=mv[0], ay=y, xref="x", yref="y", axref="x", ayref="y", text="",
                            showarrow=True, arrowhead=3, arrowsize=1.1, arrowwidth=2.2, arrowcolor=COLOR[name]))
    # M4.5a: the shared Vision node W above the lane cell 0
    ward_node = bool(view.get("ward_node"))
    if ward_node:
        shapes.append(dict(type="line", x0=0, x1=0, y0=BAR, y1=Y_WARD - 0.3, line=dict(color=WARD_LINE, width=1.5,
                                                                                       dash="dot"), layer="below"))
        shapes.append(_shape(-0.62, Y_WARD - 0.3, 0.62, Y_WARD + 0.3, WARD_FILL, line=WARD_LINE, width=1.5,
                             layer="below", rx=0.1, ry=0.1))
        ann.append(_text(0.9, Y_WARD, "W (Vision)", size=10, color=WARD_LINE, xanchor="left"))
    # champions
    cx, cy, ctext, ccol, cop, chover = [], [], [], [], [], []
    pos, ypos = {}, {}
    for name in ("A", "B"):
        d = view["champions"][name]
        away = d["absent"] > 0
        at_w = d.get("at_ward", False)
        x = (-base - 0.55 if name == "A" else base + 0.55) if away else (0 if at_w else d["pos_global"])
        pos[name], ypos[name] = x, (Y_WARD if at_w and not away else Y_LANE_CHAMP)
    if (not any(view["champions"][n]["absent"] for n in ("A", "B")) and pos["A"] == pos["B"]
            and ypos["A"] == ypos["B"]):
        pos["A"] -= 0.24
        pos["B"] += 0.24
    wave = view.get("wave") or {}
    fire = view.get("minion_fire") or {}
    wardev = view.get("ward") or {}
    for name in ("A", "B"):
        d = view["champions"][name]
        away = d["absent"] > 0
        cx.append(pos[name])
        cy.append(ypos[name])
        ctext.append(name)
        ccol.append(COLOR[name])
        cop.append(0.4 if away else 1.0)
        cell = "W" if d.get("at_ward") else f"{d['pos_own']:+d}"
        chover.append(f"Champion {name}: level {d['level']}, HP {d['hp']}/{d['hp_max']}, own cell {cell}"
                      + (f", away for {d['absent']} more turn(s)" if away else "")
                      + (f", vision {d['vision']} turn(s)" if d.get("vision") else ""))
        y = ypos[name]
        if away:
            ann.append(_text(pos[name], y + 0.4, f"away ({d['absent']})", size=10, color="#777"))
        else:
            # level and this frame's wave damage: the turn's budget (start), the HP removed in the tick / turn
            w = wave.get(name)
            txt = f"L{d['level']}"
            if w is not None and w["budget"] is not None and w["wave"] != "—" and not d.get("at_ward"):
                txt += (f" · {w['wave']} {w['budget']}" if w["dealt"] is None else
                        f" · {w['wave']} {w['dealt']}/{w['budget']}")
            if d.get("vision"):
                txt += f" · V{d['vision']}"
            if wardev.get(name):
                txt += f" · ward {wardev[name]}"
            ann.append(_text(pos[name], y + 0.42, txt, size=10, color=DARK[name]))
        mf = fire.get(name)
        if mf and mf["units"]:
            txt = f"minions {mf['units']}" + (f" = {mf['centi']} cHP" if mf.get("centi") else "") +                 (f" ({mf['cause']})" if mf["cause"] else "")
            ann.append(_text(pos[name], y - 0.38, txt, size=10, color=MINION, bold=True))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cx, y=cy, mode="markers", marker=dict(size=46, color=ccol, opacity=0.18),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=cx, y=cy, mode="markers+text", text=ctext, textfont=dict(color="white", size=15),
                             marker=dict(size=32, color=ccol, opacity=cop, line=dict(width=2, color="white")),
                             hovertext=chover, hoverinfo="text", name="champions", showlegend=False))
    fig.add_trace(go.Scatter(x=hover_x, y=hover_y, mode="markers", marker=dict(size=22, opacity=0),
                             hovertext=hover_t, hoverinfo="text", name="packets", showlegend=False))
    fig.update_layout(shapes=shapes, annotations=ann, height=HEIGHT_WARD if ward_node else HEIGHT,
                      margin=dict(l=4, r=4, t=4, b=4),
                      paper_bgcolor="white", plot_bgcolor="white", font=dict(family="Segoe UI, Helvetica, Arial"),
                      xaxis=dict(range=list(XR), visible=False, fixedrange=True),
                      yaxis=dict(range=list(YR_WARD if ward_node else YR), visible=False, fixedrange=True),
                      hovermode="closest",
                      dragmode=False)
    return fig


def timeline_figure(tl: dict, i: int) -> go.Figure:
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=("Cumulative gold income (dotted: A minus B)", "HP (fraction of maximum)",
                                        "Level", "Contact position (global; A's tower at -2, B's at +2)"))
    x = tl["i"]
    for n in ("A", "B"):
        fig.add_trace(go.Scatter(x=x, y=tl[f"earned_{n}"], name=f"{n} income", line=dict(color=COLOR[n])), 1, 1)
        fig.add_trace(go.Scatter(x=x, y=tl[f"hp_{n}"], name=f"{n} HP", line=dict(color=COLOR[n]),
                                 showlegend=False), 2, 1)
        fig.add_trace(go.Scatter(x=x, y=tl[f"level_{n}"], name=f"{n} level", line=dict(color=COLOR[n], shape="hv"),
                                 showlegend=False), 3, 1)
    fig.add_trace(go.Scatter(x=x, y=tl["advantage"], name="A − B", line=dict(color="black", dash="dot")), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=tl["contact"], name="contact", mode="lines+markers", connectgaps=False,
                             line=dict(color="gray"), showlegend=False), 4, 1)
    fig.add_vline(x=i, line=dict(color="green", width=1))
    fig.update_layout(height=640, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.08))
    fig.update_xaxes(title_text="turn", row=4, col=1)
    return fig
