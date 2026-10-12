"""Scripted policies for the viewer (M4.2e-D).

The policies ARE the M4.2e-B/C reactive policies of `lanegame.experiments.m4.zoning_natural` (`Policy.act` picks the
action; nothing here re-implements a rule), plus a few compositions of their existing behaviour and wave rules and one
crash-then-Recall policy. `decide` returns the action together with a short note that names the rule that fired and,
when the intended (wave, behaviour, objective) code was not legal, the deterministic fallback that was played.

Every policy is a function of (cfg, lane, m4, side) only (no memory), so a recording replays from its actions.
"""

from __future__ import annotations

from typing import Dict, Tuple

from ..champions import CHAMP_ABSENT, champion_legal_actions
from ..champions_m4 import context
from ..config import (B_HOLD, B_RECALL, B_SAFE, B_TRADE, BEHAVIOR_NAMES, NATURAL_WAVE_NAMES, NONE, PRESSURE,
                      W_HOLD, champ_act)
from ..experiments.m4 import zoning_natural as zn

RECALL = champ_act(W_HOLD, B_RECALL)
WALK_BACK = champ_act(W_HOLD, B_SAFE)


class ViewerPolicy(zn.Policy):
    """`zoning_natural.Policy` with a description and an explained decision."""

    def __init__(self, name: str, description: str, behavior: str = "hold", wave: str = "hold",
                 pressure: bool = False, **kw):
        super().__init__(name, behavior, wave, pressure, **kw)
        self.description = description

    def decide(self, cfg, lane, m4, s: int) -> Tuple[int, str]:
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


# The demonstration matchups: (scenario, power gap, policy A, policy B, turns, what A attempts).
MATCHUPS = {
    "1. Passive farming": ("s0", "equal", "farm", "farm", 40,
                           "Both only last-hit (Hold/Hold). Watch the symmetric neutral lane."),
    "2. Slow push": ("s0", "equal", "push", "farm", 40,
                     "A Thins while its front is small, then Holds; the Lanchester law grows the edge into a stack "
                     "and a crash. B only last-hits."),
    "3. Wave clear": ("neutral", "equal", "clear", "push", 40,
                      "A Clears every turn against a slow pusher: fast crashes, bounces back."),
    "4. Freeze": ("freeze", "equal", "freeze", "build_clear", 50,
                  "A holds the enemy wave outside its tower in the freeze band; B builds a stack and Clears to break "
                  "it."),
    "5. Trading and zoning": ("neutral", "moderate", "bully", "respect", 40,
                              "A (x1.24 power) Trades every turn; B backs off when threatened."),
    "6. Crash and Recall": ("neutral", "equal", "crash_recall", "farm", 50,
                            "A builds and crashes a wave, then Recalls while B deals with the crash."),
    "7. Power advantage": ("neutral", "large", "push_punish", "push_punish", 40,
                           "Same policy both sides; A has x1.76 attack power."),
}
