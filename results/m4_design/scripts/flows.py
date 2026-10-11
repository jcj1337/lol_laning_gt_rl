"""Lane flows under scripted policies: mutual Slow Push cycle, mutual Shove, timing and minion death rate.
Legacy codes: wave + 3*posture (Contest = 0 -> reach T)."""
import collections
from lanegame.config import GameConfig, SP, SH, F
from lanegame.mechanics import step_turn, initial_state, wave_actions
from lanegame.state import State

cfg = GameConfig()
T = cfg.tower


def fmt(st):
    a = f"A f{st.ax}:{st.ah} r{list(st.ar)} sg{st.asg}" if st.ah else f"A - r{list(st.ar)} sg{st.asg}"
    b = f"B f{-st.bx}:{st.bh} r{[-x for x in st.br]} sg{st.bsg}" if st.bh else f"B - r{[-x for x in st.br]} sg{st.bsg}"
    return a + " | " + b + " (global)"


def play(pa, pb, K=64, show=0, st=None, k0=0):
    st = st or initial_state(cfg)
    tot = collections.Counter()
    hist = []
    for k in range(k0, k0 + K):
        a, b = pa(st, k, 0), pb(st, k, 1)
        r = step_turn(st, k, a, b, cfg, record=True)
        ka, kb = sum(r.kills[0].values()), sum(r.kills[1].values())   # enemy HP killed, credited to A / B
        tot["killsA"] += ka; tot["killsB"] += kb
        tot["crashA"] += r.crash_hp[0]; tot["crashB"] += r.crash_hp[1]
        tot["towerA"] += r.events["tower_hp"][0]; tot["towerB"] += r.events["tower_hp"][1]
        tot["laneA"] += r.events["killed_hp"][0] - r.events["tower_hp"][0]
        if k - k0 < show:
            print(f"k={k:2d} {fmt(st)}  -> kills A{ka} B{kb} crash {r.crash_hp}  ticks:",
                  [(d['A'], [[-p[0], p[1]] for p in d['B']], d['fight'] and d['fight'][4:]) for d in r.trace])
        hist.append((k % cfg.phase_period, st))
        st = r.state._replace(va=0, vb=0, ap=0, bp=0)
    return tot, hist, st


const = lambda c: (lambda st, k, s: c)
print("=== mutual Slow Push (SP-Contest) ===")
tot, hist, _ = play(const(SP), const(SP), 64, show=14)
print({k: v / 64 for k, v in tot.items()}, "per turn")
# cycle detection
first = {}
for i, h in enumerate(hist):
    if h in first:
        print(f"cycle: (state, phase) at k={i} repeats k={first[h]} -> period {i - first[h]} turns"); break
    first[h] = i
print("=== mutual Shove (SH-Contest) ===")
tot, hist, _ = play(const(SH), const(SH), 64, show=10)
print({k: v / 64 for k, v in tot.items()}, "per turn")
first = {}
for i, h in enumerate(hist):
    if h in first:
        print(f"cycle at k={i} repeats k={first[h]} -> period {i - first[h]}"); break
    first[h] = i
print("=== Shove (A) vs Slow Push (B) ===")
tot, hist, _ = play(const(SH), const(SP), 64, show=12)
print({k: v / 64 for k, v in tot.items()}, "per turn")
print("=== Shove (A) vs Freeze-when-legal else SP (B) ===")
fz = lambda st, k, s: F if F in wave_actions(st, s, cfg) else SP
tot, hist, _ = play(const(SH), fz, 64, show=12)
print({k: v / 64 for k, v in tot.items()}, "per turn")

print("=== travel time: A Shove alone (B has no packets; B spawns removed) ===")
st = State(-4, 4, (), 0, 4, 0, (), 0, 0, 0)   # B 'front' hp 0 = none
for t in range(4):
    r = step_turn(st, 1, SH, SP, cfg, record=True)   # k=1: no spawn at end of turn 1 (next tick 4 -> spawn!)
    print("turn", t, [(d['A'], d['crashed']) for d in r.trace], "crash", r.crash_hp)
    st = r.state._replace(bx=4, bh=0, br=(), ar=())
    if st.ah == 0:
        break
print("=== isolated 4v4 fight at the lane (no champions: reach NO_REACH) and with shove ===")
for wa, wb, reach in ((SP, SP, (-10**6, -10**6)), (SH, SH, (2, 2)), (SH, SP, (2, -10**6))):
    st = State(0, 4, (), 0, -1, 4, (), 0, 0, 0)      # A at 0, B at global +1: in contact
    ticks = 0
    for k in (1, 3, 5):                               # odd k: no spawn at end of turn
        r = step_turn(st, k, wa, wb, cfg, record=True, champion_reach=reach)
        for d in r.trace:
            ticks += 1
            print("  ", wa, wb, "tick", ticks, d['A'], d['B'], d['fight'] and d['fight'][4:])
        st = r.state
        if st.ah == 0 or st.bh == 0:
            break
