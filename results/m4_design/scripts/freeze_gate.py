"""Prototype of the M4 Freeze evaluation gate (design review, item 3). Design-only: no repository code changes.

Uses the UNCHANGED M2 champion resolver (`champions.resolve_turn`) for positions, farm spots, evasion, the lane
step and gold, and adds the M4 accounting on top, outside the engine:
  * Freeze maintenance (design §6.3): a Freeze whose champion is away / channeling / connected-trading or ends more
    than 1 cell from the freeze line is re-resolved as Slow Push;
  * XP: 2 per enemy minion HP dying within 3 cells of a present champion (design §7.3);
  * tower hits from the per-tick siege mass (design §11.2) plus Apply Pressure P(L, N) (L from XP, N = 0);
  * plates at 1600 / 3600 / 6000 centi-units, 1.0 resource each (design §11.3).
No chance events are drawn (u = None): no trade damage, no ganks, no deaths. Gank EXPOSURE is reported as the
expected occurrence and expected HP loss per turn from the M2 formulas, separately from the return. Trades and
the tower-adjacent shot are not modelled here; they are part of the full gate after implementation (plan S3).

Scenarios are reachable by construction: scripted prefixes from the initial state (B shoves for j turns while A
farms; or A recalls at turn r while B farms), keeping every distinct state in which Freeze is legal for A.
From each scenario A (defender) plays one of three policies and B (attacker) one of five, for H turns:
  A: freeze   - Freeze when legal (maintained), else Slow Push; Hold
     slowpush - Slow Push; Hold
     cshove   - Shove while own leading wave HP >= enemy leading wave HP (or no enemy front), else Slow Push; Hold
  B: park_hold  - Slow Push + Hold (farms at its wave)          park_safe - Slow Push + Safe (stays at -1)
     breaker    - Shove + Hold                                 presser   - Shove + Hold, Apply Pressure when legal
     stacker    - Slow Push until its front has >= 8 HP, then Shove; Hold; Pressure when legal
Reported per (A, B): first crash of B into A's tower, B's tower damage and plates on A's tower, gold, XP,
exposure, and the discounted relative return to A (gold + plate gold, gamma = 0.99) over H turns.
Run:  PYTHONPATH=. python results/m4_design/scripts/freeze_gate.py
"""
import collections
import statistics

from lanegame.champions import (AWAY, CHANNEL, TRADING, ChampState, champion_legal_actions, initial_champions,
                                objective_reachable, resolve_turn)
from lanegame.config import (B_HOLD, B_RECALL, B_SAFE, F, GameConfig, NONE, PRESSURE, SH, SP, champ_act)
from lanegame.mechanics import initial_state, minion_damage, step_turn

cfg = GameConfig.champions()
T = cfg.tower
GAMMA, H = 0.99, 120
THETA = (1600, 3600, 6000)
XP_CUM = [0, 0] + [8 * (L - 1) + 3 * (L - 1) * (L - 2) // 2 for L in range(2, 19)]


def level(x):
    return max(L for L in range(1, 19) if XP_CUM[L] <= x)


def pressure_damage(L, N=0):
    ap = int(10000 * 1.05 ** (L - 1) * 1.2 ** N + 0.5)
    return (400 * ap + 5000) // 10000


def lane_step(s, k, a, b, reach):
    return step_turn(s, k, a, b, cfg, champion_reach=reach, record=True)


def tower_hits(lane0, res, side):
    """Minion hits on the enemy tower by side's crashed mass (start-of-tick mass convention)."""
    m = lane0.siege_of(side)
    hits = 0
    for tr in res.trace:
        if m > 0:
            hits += minion_damage(m, cfg)
        m = tr["siege"][side]
    return hits


def pick(lane, champ, s, wave, beh, obj=NONE):
    legal = champion_legal_actions(lane, champ, s, cfg)
    if len(legal) == 1:
        return legal[0]                                   # ABSENT
    for w in (wave, SP):
        for o in ((obj, NONE) if obj != NONE else (NONE,)):
            a = champ_act(w, beh, o)
            if a in legal:
                return a
    raise RuntimeError("no legal action")


def front_hp(lane, s):
    return lane.ah if s == 0 else lane.bh


def a_policy(name, lane, champ):
    if name == "freeze":
        return pick(lane, champ, 0, F, B_HOLD)
    if name == "slowpush":
        return pick(lane, champ, 0, SP, B_HOLD)
    own, enemy = front_hp(lane, 0), front_hp(lane, 1)
    return pick(lane, champ, 0, SH if (enemy == 0 or own >= enemy) else SP, B_HOLD)


def b_policy(name, lane, champ):
    if name == "park_hold":
        return pick(lane, champ, 1, SP, B_HOLD)
    if name == "park_safe":
        return pick(lane, champ, 1, SP, B_SAFE)
    if name == "breaker":
        return pick(lane, champ, 1, SH, B_HOLD)
    obj = PRESSURE if objective_reachable(lane, champ, 1, cfg) else NONE
    if name == "presser":
        return pick(lane, champ, 1, SH, B_HOLD, obj)
    return pick(lane, champ, 1, SH if front_hp(lane, 1) >= 8 else SP, B_HOLD, obj)      # stacker


def turn(lane, champ, k, a, b):
    ct = resolve_turn(lane, champ, k, a, b, cfg, lane_step=lane_step, u=None)
    if a % 3 == F:                                         # Freeze maintenance (design §6.3)
        maintained = ct.modes[0] not in (AWAY, CHANNEL, TRADING) and abs(ct.end[0] - (-1)) <= cfg.farm_reach
        if not maintained:
            ct = resolve_turn(lane, champ, k, a - F + SP, b, cfg, lane_step=lane_step, u=None)
    return ct


def play(lane, champ, k0, an, bn):
    tau = [0, 0]
    xp = [0, 0]
    disc = 0.0
    out = collections.Counter()
    first_crash = None
    for t in range(H):
        k = k0 + t
        a, b = a_policy(an, lane, champ), b_policy(bn, lane, champ)
        ct = turn(lane, champ, k, a, b)
        res = ct.lane
        plate_gold = [0.0, 0.0]
        for s in (0, 1):
            before = sum(tau[s] >= th for th in THETA)
            add = 100 * tower_hits(lane, res, s)
            if ct.effective_objectives[s] == PRESSURE:
                add += pressure_damage(level(xp[s]))
            tau[s] = min(THETA[-1], tau[s] + add)
            plate_gold[s] = sum(tau[s] >= th for th in THETA) - before
            for x, hp in res.kills[s].items():
                if not ct.away[s] and abs(x - ct.end[s]) <= 3:
                    xp[s] += 2 * hp
                    out[f"xp{s}"] += 2 * hp
            out[f"gold{s}"] += ct.farm[s]
            out[f"plates{s}"] += plate_gold[s]
            out[f"gank_p{s}"] += ct.gank_p[s]
            out[f"gank_hp{s}"] += ct.gank_p[s] * cfg.gank_damage_trials * ct.gank_damage_p[s]
            out[f"freeze_turns{s}"] += (a % 3 == F) if s == 0 else 0
        if res.crash_hp[1] > 0:
            out["crashes_b"] += 1
            if first_crash is None:
                first_crash = t
        r = (ct.farm[0] + plate_gold[0]) - (ct.farm[1] + plate_gold[1])
        disc += GAMMA ** t * r
        lane, champ = res.state, ct.champ
    out["tau_b"] = tau[1]
    out["tau_a"] = tau[0]
    return disc, first_crash, out


# ---- reachable scenarios: Freeze legal for A
def scenarios():
    found = {}
    for j in range(1, 13):                                 # B shoves for j turns, A farms
        lane, champ = initial_state(cfg), initial_champions(cfg)
        for k in range(j + 6):
            a = pick(lane, champ, 0, SP, B_HOLD)
            b = pick(lane, champ, 1, SH if k < j else SP, B_HOLD)
            ct = resolve_turn(lane, champ, k, a, b, cfg, u=None)
            lane, champ = ct.lane.state, ct.champ
            if F in [x % 3 for x in champion_legal_actions(lane, champ, 0, cfg)] and champ.a.absent == 0:
                found.setdefault((lane, champ, (k + 1) % 4), ("B shoves", j, k + 1))
    for r in range(0, 9):                                  # A recalls at turn r, B farms (or shoves)
        for bw in (SP, SH):
            lane, champ = initial_state(cfg), initial_champions(cfg)
            for k in range(r + 12):
                if k == r and champion_legal_actions(lane, champ, 0, cfg).count(champ_act(SP, B_RECALL)):
                    a = champ_act(SP, B_RECALL)
                else:
                    a = pick(lane, champ, 0, SP, B_HOLD)
                b = pick(lane, champ, 1, bw, B_HOLD)
                ct = resolve_turn(lane, champ, k, a, b, cfg, u=None)
                lane, champ = ct.lane.state, ct.champ
                if F in [x % 3 for x in champion_legal_actions(lane, champ, 0, cfg)] and champ.a.absent == 0:
                    found.setdefault((lane, champ, (k + 1) % 4), ("A recalls, B " + ("SP" if bw == SP else "SH"), r, k + 1))
    import random                                          # seeded random legal prefixes (no chance events)
    rng = random.Random(20261011)
    for g in range(400):
        lane, champ = initial_state(cfg), initial_champions(cfg)
        pa, pb = rng.random(), rng.random()                # per-game propensity to shove
        for k in range(rng.randint(4, 40)):
            acts = []
            for s, p_sh in ((0, pa), (1, pb)):
                legal = champion_legal_actions(lane, champ, s, cfg)
                if len(legal) == 1:
                    acts.append(legal[0])
                    continue
                if champ_act(SP, B_RECALL) in legal and rng.random() < 0.05:
                    acts.append(champ_act(SP, B_RECALL))
                    continue
                w = SH if rng.random() < p_sh else (F if rng.random() < 0.3 else SP)
                beh = B_SAFE if rng.random() < 0.1 else B_HOLD
                obj = PRESSURE if objective_reachable(lane, champ, s, cfg) and rng.random() < 0.7 else NONE
                acts.append(pick(lane, champ, s, w, beh, obj))
            ct = resolve_turn(lane, champ, k, acts[0], acts[1], cfg, u=None)
            lane, champ = ct.lane.state, ct.champ
            if F in [x % 3 for x in champion_legal_actions(lane, champ, 0, cfg)] and champ.a.absent == 0:
                found.setdefault((lane, champ, (k + 1) % 4), ("random prefix", g, k + 1))
    return found


sc = scenarios()
print(f"reachable Freeze-legal scenarios (distinct lane+champion+phase): {len(sc)}")
A_POL = ("freeze", "slowpush", "cshove")
B_POL = ("park_hold", "park_safe", "breaker", "presser", "stacker")
rows = collections.defaultdict(list)
per_scen = []
for (lane, champ, ph), origin in sc.items():
    res = {}
    for bn in B_POL:
        for an in A_POL:
            res[(an, bn)] = play(lane, champ, ph, an, bn)
            rows[(an, bn)].append(res[(an, bn)])
    per_scen.append((origin, lane, champ, ph, res))

print(f"\nMeans over scenarios, H = {H}, gamma = {GAMMA}. disc = discounted relative return to A (gold + plates);")
print("crash@ = median first crash of B into A's tower (None = never, counted as H); tauB = B's damage on A's tower")
hdr = f"{'B policy':10s} {'A policy':9s} {'disc':>6s} {'crash@':>6s} {'crashes':>7s} {'tauB':>6s} {'platesB':>7s} " \
      f"{'goldA':>6s} {'goldB':>6s} {'xpA':>5s} {'xpB':>5s} {'gankHP_A':>8s} {'gankHP_B':>8s} {'Fturns':>6s}"
print(hdr)
for bn in B_POL:
    for an in A_POL:
        rr = rows[(an, bn)]
        n = len(rr)
        m = lambda key: sum(o[key] for _, _, o in rr) / n
        crash = statistics.median([H if c is None else c for _, c, _ in rr])
        print(f"{bn:10s} {an:9s} {sum(d for d, _, _ in rr) / n:6.2f} {crash:6.0f} {m('crashes_b'):7.2f} {m('tau_b'):6.0f} "
              f"{m('plates1'):7.2f} {m('gold0'):6.2f} {m('gold1'):6.2f} {m('xp0'):5.0f} {m('xp1'):5.0f} "
              f"{m('gank_hp0'):8.2f} {m('gank_hp1'):8.2f} {m('freeze_turns0'):6.1f}")

print("\nPer B policy: in how many scenarios is Freeze better / equal / worse than the best of Slow Push and conditional"
      " Shove (discounted relative return, |diff| > 0.05)?")
for bn in B_POL:
    better = equal = worse = 0
    diffs = []
    for origin, lane, champ, ph, res in per_scen:
        f = res[("freeze", bn)][0]
        alt = max(res[("slowpush", bn)][0], res[("cshove", bn)][0])
        d = f - alt
        diffs.append(d)
        if d > 0.05:
            better += 1
        elif d < -0.05:
            worse += 1
        else:
            equal += 1
    print(f"  {bn:10s} better {better:3d}  equal {equal:3d}  worse {worse:3d}   mean diff {sum(diffs) / len(diffs):+.2f}"
          f"  max {max(diffs):+.2f}  min {min(diffs):+.2f}")

print("\nFreeze minus Slow Push only (isolates the Freeze action):")
for bn in B_POL:
    ds = [res[("freeze", bn)][0] - res[("slowpush", bn)][0] for *_, res in per_scen]
    nz = sum(abs(d) > 0.05 for d in ds)
    print(f"  {bn:10s} mean {sum(ds) / len(ds):+.2f}  differs in {nz}/{len(ds)}  max {max(ds):+.2f}  min {min(ds):+.2f}")

print("\nScenario origins:", collections.Counter(o[0] for o, *_ in per_scen))

print("\nScenarios where Freeze differs from Slow Push (|diff| > 0.05) - state at the start, A's view:")
for origin, lane, champ, ph, res in per_scen:
    ds = {bn: res[("freeze", bn)][0] - res[("slowpush", bn)][0] for bn in B_POL}
    if any(abs(d) > 0.05 for d in ds.values()):
        own_c = lane.ah > 0 and lane.ax + lane.bx >= -1
        print(f"  A front {lane.ax:+d}:{lane.ah} rears {lane.ar} | B front at A's {-lane.bx:+d}:{lane.bh} rears {lane.br} "
              f"| sieges {lane.asg}/{lane.bsg} contact={own_c} ph={ph} | " + " ".join(f"{b}:{d:+.2f}" for b, d in ds.items()))
