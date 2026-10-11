"""Timing facts and Freeze persistence rollouts."""
from lanegame.config import GameConfig, SP, SH, F
from lanegame.mechanics import step_turn, wave_actions
from lanegame.state import State

cfg = GameConfig()
NR = -10 ** 6


def fight(ha, hb, wa, wb, reach, label, xa=0, xb=-1):
    """Isolated fight A at own xa vs B at own xb (in contact), no spawns (even k only), count ticks."""
    st = State(xa, ha, (), 0, xb, hb, (), 0, 0, 0)
    ticks = 0
    for k in (0, 2, 4, 6, 8, 10):
        # remove spawns: use even k (next tick (k+1)*2 not a multiple of 4)
        r = step_turn(st, k, wa, wb, cfg, record=True, champion_reach=reach)
        for d in r.trace:
            ticks += 1
            if not d['A'] or not d['B']:
                print(f"  {label}: {ha}v{hb} ends after {ticks} ticks; survivors A{d['A']} B{d['B']}")
                return
        st = r.state


fight(4, 4, SP, SP, (NR, NR), "no champions")
fight(4, 4, SH, SH, (2, 2), "both shove (+1 each)")
fight(4, 4, SH, SP, (2, NR), "A shoves alone")
fight(8, 4, SP, SP, (NR, NR), "8 vs 4")
fight(12, 4, SP, SP, (NR, NR), "12 vs 4")
fight(12, 12, SP, SP, (NR, NR), "12 vs 12")
fight(4, 4, SP, SP, (NR, NR), "4v4, B in A's tower cell (tower +1 on B)", xa=-3, xb=2)

print("travel: a lone Shove wave from its base")
st = State(-4, 4, (), 0, 4, 0, (), 0, 0, 0)
tick = 0
for k in (0, 2, 4):
    r = step_turn(st, k, SH, SP, cfg, record=True)
    for d in r.trace:
        tick += 1
        print(f"  tick {tick}: A {d['A']} crashed {d['crashed']}")
    st = r.state._replace(bh=0, br=())
    if r.crash_hp[0]:
        break
print("travel: a lone Slow Push wave (stack_creep): 4 HP parks at the lane")
st = State(-4, 4, (), 0, 4, 0, (), 0, 0, 0)
for k in range(0, 8):
    r = step_turn(st, k, SP, SP, cfg)
    st = r.state._replace(bh=0, br=())
    print(f"  after turn {k}: A front {st.ax}:{st.ah} rears {st.ar} crash {r.crash_hp[0]}")

print("Freeze persistence: B front at A's -1 (B own +1), A front none / at -4; A plays F-when-legal vs SP;"
      " B plays SH with or without champion reach")
for a_front in ((-4, 4), (-4, 0)):
    for bw, breach in ((SH, 2), (SH, NR), (SP, 2)):
        for bh in (3, 5, 7, 9):
            outs = {}
            for apol in ("F", "SP"):
                st = State(a_front[0], a_front[1], (), 0, 1, bh, (), 0, 0, 0)
                log = []
                crashB = killsA = killsB = 0
                for k in range(0, 8):
                    a = F if (apol == "F" and F in wave_actions(st, 0, cfg)) else SP
                    r = step_turn(st, k, a, bw, cfg, champion_reach=(2, breach))
                    crashB += r.crash_hp[1]
                    killsA += sum(r.kills[0].values()); killsB += sum(r.kills[1].values())
                    log.append(("F" if a == F else "-") + ("H" if r.events["freeze_held"][1] else ""))
                    st = r.state
                outs[apol] = (crashB, killsA, killsB, "".join(log))
            if outs["F"] != outs["SP"]:
                print(f"  A{a_front} B h{bh} {'SH' if bw == SH else 'SP'} reach={'T' if breach == 2 else 'none'}: "
                      f"F-policy crashB/killsA/killsB/turns {outs['F']}  vs SP {outs['SP']}")
