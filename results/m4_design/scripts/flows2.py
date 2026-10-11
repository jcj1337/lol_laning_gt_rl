"""How often can Freeze bind in lane-only play? Random / biased policies over legacy codes.
Per side-turn: F legal; F legal with own front NOT in contact (hold can bind); F != SP for the realized enemy
action (same own posture). Also: what happened on the previous turn to the freezer."""
import collections, random, sys
from lanegame.config import GameConfig, SP, SH, F
from lanegame.mechanics import step_turn, initial_state, legal_actions, wave_of


def sig(r):
    return (r.state, r.crash_hp, tuple(sorted(r.kills[0].items())), tuple(sorted(r.kills[1].items())), r.exposure)


cfg = GameConfig()
G, K = 1500, 64
profiles = {
    "uniform legal (wave x posture x objective)": lambda st, s, rng, me: rng.choice(legal_actions(st, s, cfg)),
    "Shove-heavy 70% vs uniform": lambda st, s, rng, me: (SH if (s == 0 and rng.random() < .7) else rng.choice(legal_actions(st, s, cfg))),
    "SP 80% (Hold-like) both": lambda st, s, rng, me: (SP if rng.random() < .8 else rng.choice(legal_actions(st, s, cfg))),
}
for name, pol in profiles.items():
    rng = random.Random(1)
    c = collections.Counter()
    geo = collections.Counter()
    for g in range(G):
        st = initial_state(cfg)
        prev = None
        for k in range(K):
            acts = [pol(st, 0, rng, None), pol(st, 1, rng, None)]
            for s in (0, 1):
                e = 1 - s
                c["side-turns"] += 1
                if F not in [wave_of(a) for a in legal_actions(st, s, cfg)]:
                    continue
                c["F legal"] += 1
                fx, fh = st.side(s)[:2]
                ex = st.side(e)[0]
                contact = fh > 0 and fx + ex >= -1
                if not contact:
                    c["F legal, own front not in contact"] += 1
                    why = "no own front" if fh == 0 else f"own front at {fx}"
                    if prev is not None:
                        pfx, pfh = prev[0].side(s)[:2]
                        hist = ("own crash last turn" if prev[1].crash_hp[s] else
                                "own front killed last turn" if pfh > 0 and prev[1].events["died"][s] else "other")
                    else:
                        hist = "start"
                    geo[(why, hist, f"enemy siege>0={st.side(e)[3] > 0}")] += 1
                # F vs SP for the realized enemy action, same own posture/objective
                a_own = acts[s]
                base = a_own - wave_of(a_own)
                pf = [None, None]; pp = [None, None]
                pf[s], pp[s] = base + F, base + SP
                pf[e] = pp[e] = acts[e]
                if sig(step_turn(st, k, pf[0], pf[1], cfg)) != sig(step_turn(st, k, pp[0], pp[1], cfg)):
                    c["F != SP (realized enemy action)"] += 1
                    c[f"  ... with own front not in contact" if not contact else "  ... in contact"] += 1
            r = step_turn(st, k, acts[0], acts[1], cfg)
            prev = (st, r)
            st = r.state
    n = c["side-turns"]
    print(f"== {name}: {G} games x {K} turns, {n} side-turns")
    for key in ("F legal", "F legal, own front not in contact", "F != SP (realized enemy action)",
                "  ... in contact", "  ... with own front not in contact"):
        print(f"   {key:45s} {c[key]:7d}  ({100 * c[key] / n:.2f}% of side-turns, {c[key] / (2 * G):.2f} per side-game)")
    for kk, v in geo.most_common(8):
        print("     ", v, kk)
