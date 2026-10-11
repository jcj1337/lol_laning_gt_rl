"""Why do some non-contact Freeze-legal situations show no F/SP difference? Sample them (legacy reach set)."""
import collections, runpy, sys
sys.argv = ["x", "legacy"]
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    g = runpy.run_path(sys.path[0] + "/freeze_enum.py")
seen, cfg, wave_actions, F, run, sig, side_opts, SP = (g[k] for k in ("seen", "cfg", "wave_actions", "F", "run", "sig", "side_opts", "SP"))
why = collections.Counter()
for st, ph in seen:
    for s in (0, 1):
        if F not in wave_actions(st, s, cfg):
            continue
        e = 1 - s
        fx, fh = st.side(s)[:2]
        ex, eh, er, esg = st.side(e)[:4]
        if fh > 0 and fx + ex >= -1:
            continue
        diff = False
        for o in side_opts(st, s):
            if o[0] != F: continue
            for oe in side_opts(st, e):
                pr = (o, oe) if s == 0 else (oe, o)
                ps = (((SP,) + o[1:]), oe) if s == 0 else (oe, ((SP,) + o[1:]))
                if sig(run(st, ph, *pr)) != sig(run(st, ph, *ps)):
                    diff = True
        if not diff:
            blob = eh + sum(4 for r in er if r >= ex - 1)
            why[(f"enemy blob={'>=10' if blob >= 10 else ('6-9' if blob >= 6 else '<=5')}", f"phase={ph}",
                 f"enemy h>4={eh > 4}", f"esiege>0={esg > 0}")] += 1
for k, v in why.most_common(20):
    print(v, k)
