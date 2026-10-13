"""Scripted policies for the viewer (M4.2e-D; four-action strategies added in the M4.3c stabilization).

The three-action policies ARE the M4.2e-B/C reactive policies of `lanegame.experiments.m4.zoning_natural`
(`Policy.act` picks the action; nothing here re-implements a rule), plus a few compositions of their existing behaviour
and wave rules and one crash-then-Recall policy. The four-action policies (names ending in "4") ARE the strategies of
`lanegame.experiments.m43c.strategies` (Last Hit / Thin / Slow Push / Shove wave rules, turn schedules, Recall rules).
`decide` returns the action together with a short note that names the rule that fired and, when the intended (wave,
behaviour, objective) code was not legal, the deterministic fallback that was played.

Every policy is a function of (cfg, lane, m4, side) and the recording's turn index i only (no memory; only the
scheduled strategies read i), so a recording replays from its actions.
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Optional, Tuple

from ..champions import CHAMP_ABSENT, champion_legal_actions
from ..champions_m4 import context
from ..config import (B_HOLD, B_RECALL, B_SAFE, B_TRADE, BEHAVIOR_NAMES, NATURAL_WAVE_NAMES, NONE, PRESSURE,
                      W_HOLD, champ_act)
from ..experiments.m43c import strategies as st
from ..experiments.m4 import zoning_natural as zn

RECALL = champ_act(W_HOLD, B_RECALL)
WALK_BACK = champ_act(W_HOLD, B_SAFE)


class ViewerPolicy(zn.Policy):
    """`zoning_natural.Policy` with a description and an explained decision."""

    def __init__(self, name: str, description: str, behavior: str = "hold", wave: str = "hold",
                 pressure: bool = False, **kw):
        super().__init__(name, behavior, wave, pressure, **kw)
        self.description = description

    def decide(self, cfg, lane, m4, s: int, i: Optional[int] = None) -> Tuple[int, str]:
        code = self.act(cfg, lane, m4, s)
        return code, self.explain(cfg, lane, m4, s, code)

    def explain(self, cfg, lane, m4, s: int, code: int) -> str:
        """Which of Policy.act's rules produced `code` (same order and tests as Policy.act)."""
        if code == CHAMP_ABSENT:
            return "absent"
        me, en, t = m4.side(s), m4.side(1 - s), context(cfg).t
        H = t.hp_max[t.level_by_xp[min(me.xp, t.xp_cap)]]
        reset = en.absent > 0 and me.hp <= zn.RESET_FRAC * H and lane.siege_of(1 - s) == 0
        if me.hp <= self.recall_frac * H or reset:
            why = "low HP" if me.hp <= self.recall_frac * H else "free reset (enemy away)"
            return f"{why}: Recall" if code == RECALL else f"{why}: Recall not legal here -> walk back (Safe)"
        beh = zn.behavior_rule(self.behavior, cfg, lane, m4, s)
        wave = zn.wave_rule(self.wave, cfg, lane, m4, s)
        intent = f"{NATURAL_WAVE_NAMES[wave]} + {BEHAVIOR_NAMES[beh]}"
        want = champ_act(wave, beh, PRESSURE if self.pressure and beh in (B_HOLD, B_TRADE) else NONE)
        if code == want:
            return intent + (" + Pressure" if want != champ_act(wave, beh) else "")
        if code == champ_act(wave, beh):
            return intent                      # Pressure is requested only while it is legal (a siege in reach)
        return f"{intent} not legal -> fallback {cfg.label(code)}"


class CrashRecallPolicy(ViewerPolicy):
    """Build and crash a wave (wave rule build_clear), then, while the own crashed minions are under the enemy tower,
    walk back and Recall (as agents_m4's crash_recall rule: not from the own tower cell, not while an enemy front
    threatens the own half)."""

    def act(self, cfg, lane, m4, s: int) -> int:
        code = self._crash_recall(cfg, lane, m4, s)
        return super().act(cfg, lane, m4, s) if code is None else code

    def _crash_recall(self, cfg, lane, m4, s: int):
        legal = champion_legal_actions(lane, m4, s, cfg)
        if legal == (CHAMP_ABSENT,):
            return None
        ex, eh = lane.side(1 - s)[:2]
        threat = eh > 0 and -ex <= -1
        if lane.siege_of(s) > 0 and m4.side(s).pos > -cfg.tower and not threat:
            return RECALL if (RECALL in legal and not zn.contact_at_start(cfg, m4, s)) else WALK_BACK
        return None

    def explain(self, cfg, lane, m4, s: int, code: int) -> str:
        if self._crash_recall(cfg, lane, m4, s) is not None:
            return "own wave crashed: Recall" if code == RECALL else "own wave crashed: walk back to Recall (Safe)"
        return super().explain(cfg, lane, m4, s, code)


def _p(name, desc, **kw) -> ViewerPolicy:
    return ViewerPolicy(name, desc, **kw)


POLICIES: Dict[str, ViewerPolicy] = {p.name: p for p in [
    _p("farm", "Hold the wave and Hold position: last-hit only."),
    _p("push", "Slow push: Thin while the own front is at most 2 HP ahead of the enemy front, else Hold; "
               "Pressure the tower while sieging.", wave="slow_push", pressure=True),
    _p("clear", "Clear the wave every turn (full wave-clear capacity); Pressure while sieging.", wave="clear",
       pressure=True),
    _p("build_clear", "Freeze breaker: Thin to build a stack, Clear once the own front holds 16 HP; Pressure while "
                      "sieging.", wave="build_clear", pressure=True),
    _p("freeze", "Freeze controller: once the contact stands at the own -1 cell, Thin while the enemy front holds "
                 ">= 14 HP, else Hold (keeps the enemy wave in the freeze band).", wave="freeze"),
    _p("bully", "Trade every turn the enemy is on the lane; farm with Hold.", behavior="trade"),
    _p("punish", "Trade only an enemy that starts the turn within reach (no tower dives).", behavior="punish"),
    _p("edge", "Trade when the duel edge is positive and the enemy is within reach.", behavior="edge"),
    _p("standoff", "Yield: farm from the stand-off distance.", behavior="standoff"),
    _p("respect", "Safe while the enemy starts within reach, else Yield.", behavior="respect"),
    _p("safe", "Always Safe: farm from behind the safe line.", behavior="safe"),
    _p("tower", "Concede the lane: farm only when the wave is at the own side, else Safe.", behavior="tower"),
    _p("push_punish", "Slow push + punish.", behavior="punish", wave="slow_push", pressure=True),
    _p("clear_punish", "Clear + punish.", behavior="punish", wave="clear", pressure=True),
    _p("freeze_punish", "Freeze controller + punish.", behavior="punish", wave="freeze"),
    _p("bully_shove", "Trade every turn; Clear while the enemy is away.", behavior="trade", wave="shove_away",
       pressure=True),
    CrashRecallPolicy("crash_recall", "Build a stack (Thin), Clear it into the tower once it holds 16 HP, then walk "
                                      "back and Recall while the crash occupies the enemy.", wave="build_clear",
                      pressure=True),
]}
# the four-action strategies (m4_four_actions(); under a three-action preset Slow Push is not legal and the strategy
# falls back to Last Hit / Hold, which its note says)
POLICIES.update(st.LIBRARY)


class Matchup(NamedTuple):
    scenario: str
    gap: str
    pa: str
    pb: str
    turns: int
    description: str
    config: str = "m4_lasthit"


FOUR = "m4_four_actions"
# The demonstration matchups. 4A-*: the four-action model (listed first: the viewer opens on 4A-1); 1-7: the
# three-action preset m4_lasthit() (M4.2e-D).
MATCHUPS: Dict[str, Matchup] = {
    "4A-1. Slow Push -> Last Hit -> Shove": Matchup(
        "neutral", "equal", "grow_shove4", "farm4", 40,
        "A nudges once (Slow Push), lets the 1-HP lead compound while it last-hits, and Shoves the stack into B's "
        "tower at 20 HP; Pressure while sieging. B only last-hits.", FOUR),
    "4A-2. Thin an oversized wave, then hold": Matchup(
        "freeze21", "equal", "freeze4", "walk_up4", 40,
        "B's 21-HP wave stands next to A's tower. A Thins every turn: it spends the surplus up to round(C/2) and then "
        "last-hits. At equal power (C 6, at most 3 HP) the wave still crashes; with the gap 'large' (C 11) Thin "
        "removes 6, 6, 6, 5, 2, 1 HP and the freeze holds 9 of 12 turns; a full correction needs C >= 15.", FOUR),
    "4A-3. Slow Push against Thin": Matchup(
        "neutral", "equal", "slowpush4", "thin4", 40,
        "Each turn A nudges 1 HP and B removes it: the lane stays at the midpoint. Only HP loss and Recalls move it.",
        FOUR),
    "4A-4. Slow Push, then Shove, against Thin": Matchup(
        "neutral", "equal", "sp_then_shove4", "thin4", 40,
        "A's slow push is neutralized by Thin, so after 10 turns A switches to Shove and breaks through.", FOUR),
    "4A-5. Freeze -> Recall -> Return": Matchup(
        "freeze", "equal", "freeze_recall4", "walk_up4", 40,
        "A keeps the freeze and Recalls once it holds 2 gold, while the enemy wave waits next to its tower; watch what "
        "the wave does during the two turns away.", FOUR),
    "4A-6. Crash, then Recall": Matchup(
        "neutral", "equal", "grow_crash_recall4", "farm4", 40,
        "A builds a slow push, Shoves it into B's tower and Recalls while B's tower and minions deal with the crash.",
        FOUR),
    "4A-7. Strong freezer against a walk-up": Matchup(
        "freeze", "large", "freeze_punish4", "walk_up4", 40,
        "A (x1.76 attack power, x3.1 in a duel) keeps the freeze and would punish B whenever B starts in reach; B "
        "walks up to farm anyway. It never trades: from inside its own tower cell A is in contact only with a champion "
        "in the same cell, so a walk-up next to the tower is never 'in reach' at the start of a turn.", FOUR),
    "4A-8. Weak champion against a big wave": Matchup(
        "oversized", "b_large", "thin4", "farm4", 30,
        "A is the weak side (B x1.76): its Thin is capped at round(C/2), so it corrects a 24-HP wave only partly.",
        FOUR),    "1. Passive farming": Matchup("s0", "equal", "farm", "farm", 40,
                                  "Both only last-hit (Hold/Hold). Watch the symmetric neutral lane."),
    "2. Slow push": Matchup("s0", "equal", "push", "farm", 40,
                            "A Thins while its front is small, then Holds; the Lanchester law grows the edge into a "
                            "stack and a crash. B only last-hits."),
    "3. Wave clear": Matchup("neutral", "equal", "clear", "push", 40,
                             "A Clears every turn against a slow pusher: fast crashes, bounces back."),
    "4. Freeze": Matchup("freeze", "equal", "freeze", "build_clear", 50,
                         "A holds the enemy wave outside its tower in the freeze band; B builds a stack and Clears to "
                         "break it."),
    "5. Trading and zoning": Matchup("neutral", "moderate", "bully", "respect", 40,
                                     "A (x1.24 power) Trades every turn; B backs off when threatened."),
    "6. Crash and Recall": Matchup("neutral", "equal", "crash_recall", "farm", 50,
                                   "A builds and crashes a wave, then Recalls while B deals with the crash."),
    "7. Power advantage": Matchup("neutral", "large", "push_punish", "push_punish", 40,
                                  "Same policy both sides; A has x1.76 attack power."),
}
