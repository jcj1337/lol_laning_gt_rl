"""How long does a held Freeze last? Lane only, legacy mechanics (siege default), freezer A maintains the freeze.

Start: A has no front (its wave just died) and a fresh spawn queued per phase; B's front stands at A's -1
(B own +1) with e HP. A plays Freeze while legal, else Slow Push; B plays Slow Push or Shove (with reach).
Reports, per (e, B action, start phase): turns until B crashes into A's tower (or the freeze dissolves = Freeze
illegal), Freeze-legal turns, and kill locations (B's farm opportunities at its +2 vs elsewhere), over 24 turns.
Compared with A playing Slow Push throughout.
"""
from lanegame.config import GameConfig, SP, SH, F
from lanegame.mechanics import step_turn, wave_actions
from lanegame.state import State

cfg = GameConfig()
T = cfg.tower


def run(e, b_wave, k0, a_policy, turns=24):
    s = State(0, 0, (), 0, 1, e, (), 0, 0, 0, 0, 0)
    first_crash, legal_turns, b_kills_at_2, b_kills, a_kills = None, 0, 0, 0, 0
    for k in range(k0, k0 + turns):
        legal = F in wave_actions(s, 0, cfg)
        legal_turns += legal
        a = F if (a_policy == "F" and legal) else SP
        res = step_turn(s, k, a, b_wave, cfg, champion_reach=(T, T))
        if res.crash_hp[1] > 0 and first_crash is None:
            first_crash = k - k0
        for x, hp in res.kills[1].items():
            b_kills += hp
            if x == 2:
                b_kills_at_2 += hp
        a_kills += sum(res.kills[0].values())
        s = res.state
    return first_crash, legal_turns, b_kills_at_2, b_kills, a_kills


print("e  B   phase | A=Freeze: crash@ legal kB@+2/kB kA | A=SlowPush: crash@ kB@+2/kB kA")
for b_wave, bn in ((SP, "SP"), (SH, "SH")):
    for e in (2, 3, 4, 5, 6, 7, 8, 9):
        for k0 in (0, 1, 2, 3):
            f = run(e, b_wave, k0, "F")
            p = run(e, b_wave, k0, "SP")
            print(f"{e}  {bn}  {k0}     | {str(f[0]):>6} {f[1]:5d} {f[2]:4d}/{f[3]:<3d} {f[4]:3d} |"
                  f" {str(p[0]):>6} {p[2]:4d}/{p[3]:<3d} {p[4]:3d}")
