"""Exhaustive Freeze-vs-Slow-Push audit on the lane (read-only w.r.t. the repo).

Reachability: fixed point over (lane state, k mod phase_period) from the initial state.
  mode "legacy":  legacy action set (wave x posture x objective) -> reach T (Contest) / -1 (Safe)
  mode "champ":   superset of champion-mode lane steps: wave x objective x champion_reach in R per side
Comparison: for every reachable (state, phase) and side s where F is legal, for every own objective/reach
and every enemy (wave, objective, reach), compare step_turn with own wave F vs SP.
"""
import sys, time, collections
from lanegame.config import GameConfig, SP, SH, F, PRESSURE, VISION, NONE
from lanegame.mechanics import step_turn, wave_actions, initial_state, _fronts_in_contact

mode = sys.argv[1] if len(sys.argv) > 1 else "legacy"
cfg = GameConfig()
T = cfg.tower
NO_REACH = -10 ** 6
R = (T, -1) if mode == "legacy" else (NO_REACH, -2, -1, 0, 1, 2)
PER = cfg.phase_period
cfg_nohold = cfg.with_(freeze_break_threshold=-100)   # Freeze without the hold on the enemy front


def side_opts(state, s):
    """(wave, objective, reach) options of side s."""
    sg = state.asg if s == 0 else state.bsg
    objs = (NONE, PRESSURE) if sg > 0 else (NONE,)   # VISION's lane step == NONE's
    out = []
    for w in wave_actions(state, s, cfg):
        for o in objs:
            for r in R:
                if mode == "legacy" and o != NONE and r != T:
                    continue            # legacy: objectives only with Contest
                out.append((w, o, r))
    return out


def run(state, ph, oa, ob, c=cfg):
    reach = None if mode == "legacy" and False else (oa[2], ob[2])
    return step_turn(state, ph, oa[0] + 6 * oa[1], ob[0] + 6 * ob[1], c, champion_reach=reach)


def sig(res):
    return (res.state, res.crash_hp, tuple(sorted(res.kills[0].items())), tuple(sorted(res.kills[1].items())),
            res.exposure, res.plates, res.thinned, res.vision_set)


def proj(st):
    return st._replace(va=0, vb=0, ap=0, bp=0)   # vision timers / plate counters never affect the lane


KMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 9
t0 = time.time()
root = (initial_state(cfg), 0)
seen = {root: 0}
frontier = [root]
depth = 0
while frontier and depth < KMAX - 1:
    depth += 1
    nxt = []
    for st, ph in frontier:
        for oa in side_opts(st, 0):
            for ob in side_opts(st, 1):
                s2 = (proj(run(st, ph, oa, ob).state), (ph + 1) % PER)
                if s2 not in seen:
                    seen[s2] = depth
                    nxt.append(s2)
    frontier = nxt
    print(f"  depth {depth}: new {len(nxt)}, total {len(seen)}, {time.time() - t0:.0f}s", flush=True)
print(f"[{mode}] reachable (state, phase) pairs: {len(seen)}  ({time.time() - t0:.0f}s)")

n_legal = 0
n_legal24 = n_diff24 = 0
sitc = collections.Counter()
n_diff_sit = 0
n_pairs = n_pairs_diff = 0
chan = collections.Counter()
feat = collections.Counter()
examples = []
ownfeat = collections.Counter()
for st, ph in sorted(seen, key=lambda z: (z[1], z[0])):
    for s in (0, 1):
        if F not in wave_actions(st, s, cfg):
            continue
        n_legal += 1
        e = 1 - s
        fx, fh, fr, fsg = st.side(s)[:4]
        ex, eh, er, esg = st.side(e)[:4]
        contact = fh > 0 and fx + ex >= -1
        any_diff = False
        for o in side_opts(st, s):
            if o[0] != F:
                continue
            o_sp = (SP,) + o[1:]
            for oe in side_opts(st, e):
                pair = (o, oe) if s == 0 else (oe, o)
                pair_sp = (o_sp, oe) if s == 0 else (oe, o_sp)
                rf = run(st, ph, *pair)
                rs = run(st, ph, *pair_sp)
                n_pairs += 1
                if sig(rf) == sig(rs):
                    continue
                n_pairs_diff += 1
                any_diff = True
                rnh = run(st, ph, *pair, c=cfg_nohold)
                own = sig(rnh) != sig(rs)
                held = rf.events["freeze_held"][e] > 0
                c = ("own-move" if own else "") + ("+" if own and held else "") + ("hold" if held else "")
                chan[c or "other"] += 1
                from lanegame.mechanics import _blob_hp
                push = cfg.freeze_break_push[oe[0]] if ex <= oe[2] else 0
                ebl = eh + sum(4 for r in er if r >= ex - 1)
                key = (c, ("no own front" if fh == 0 else f"own x={fx}"), f"enemy@own{-ex}", f"contact={contact}",
                       f"enemy wave={'SP SH F'.split()[oe[0]]}", f"eblob+push={ebl + push}",
                       f"esiege>0={esg > 0}", f"ph={ph}")
                if own:
                    ownfeat[(f"own x={fx} h={fh}", f"own rears={fr}", f"enemy@own{-ex} h={eh}", f"enemy rears={er}",
                             f"contact={contact}", f"esiege={esg}", f"ph={ph}", f"enemy wave={'SP SH F'.split()[oe[0]]}",
                             f"own reach={o[2]}", f"enemy reach={oe[2]}")] += 1
                feat[key] += 1
                if len(examples) < 3 or (own and len(examples) < 6):
                    tr_f = step_turn(st, ph, pair[0][0] + 6 * pair[0][1], pair[1][0] + 6 * pair[1][1], cfg,
                                     record=True, champion_reach=(pair[0][2], pair[1][2]))
                    tr_s = step_turn(st, ph, pair_sp[0][0] + 6 * pair_sp[0][1], pair_sp[1][0] + 6 * pair_sp[1][1],
                                     cfg, record=True, champion_reach=(pair_sp[0][2], pair_sp[1][2]))
                    examples.append((st.describe(), ph, s, pair, [(d["A"], d["B"], d["crashed"]) for d in tr_f.trace],
                                     [(d["A"], d["B"], d["crashed"]) for d in tr_s.trace], tr_f.crash_hp, tr_s.crash_hp))
        n_diff_sit += any_diff
        sitc[(contact, any_diff)] += 1
        if seen[(st, ph)] <= 23:
            n_legal24 += 1
            n_diff24 += any_diff
print(f"Freeze-legal (state, phase, side) situations: {n_legal}; with any F != SP outcome: {n_diff_sit}")
print(f"(own option, enemy option) pairs compared: {n_pairs}; differing: {n_pairs_diff}")
print("channels:", dict(chan))
print("situations by (contact at turn start, any difference):", dict(sitc))
print(f"first reached at k <= 23: legal situations {n_legal24}, with any difference {n_diff24}")
# contact breakdown of legal situations
cnt = collections.Counter()
for st, ph in seen:
    for s in (0, 1):
        if F in wave_actions(st, s, cfg):
            e = 1 - s
            fx, fh = st.side(s)[:2]
            ex, eh = st.side(e)[:2]
            cnt[("contact" if fh > 0 and fx + ex >= -1 else ("no own front" if fh == 0 else f"own front at {fx}, enemy at {-ex}"),
                 f"esiege>0={st.side(e)[3] > 0}", f"own siege>0={st.side(s)[3] > 0}")] += 1
print("legal-situation geometry:")
for k_, v in sorted(cnt.items(), key=lambda z: -z[1]):
    print("  ", v, k_)
print("differing-pair features (top 40):")
for k_, v in feat.most_common(40):
    print("  ", v, k_)
print("own-move differing pairs (top 30):")
for k_, v in ownfeat.most_common(30):
    print("  ", v, k_)
for ex_ in examples:
    print("EXAMPLE", ex_)
