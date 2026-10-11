"""Command-line interface.

  python -m lanegame solve [--model revised|v1] [--horizon 32] [--out results/revision/solution_K32.json]
  python -m lanegame scenarios [--model revised|v1]
  python -m lanegame exploit [--model revised|v1] [--horizon 32]
  python -m lanegame simulate --a equilibrium --b uniform [--model ...] [--episodes 2000 --seed 0]
  python -m lanegame reach [--model revised|v1] [--horizon 32]
  python -m lanegame experiments all | before_after model_comparison ...   (revised model, results/revision/)
  python -m lanegame experiments-v1 all | audit_legacy model_comparison ... (original model, results/)
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time

from . import GameConfig, LaneGame
from .baselines import all_baselines, v1_baselines
from .best_response import evaluate, exploitability

EXPERIMENTS = ["before_after", "model_comparison", "reachability", "farm_economy", "contest_safe",
               "baselines", "sensitivity", "horizon_benchmark", "verification", "figures"]
EXPERIMENTS_V1 = ["audit_legacy", "model_comparison", "horizon_benchmark", "baselines",
                  "equilibrium_structure", "sensitivity", "verification", "figures"]


def _cfg(args, **kw) -> GameConfig:
    return GameConfig.v1(**kw) if args.model == "v1" else GameConfig(**kw)


def _baselines(args):
    return v1_baselines() if args.model == "v1" else all_baselines()


def _policy(name, sol, side, args):
    if name == "equilibrium":
        return sol.policy(side)
    for b in _baselines(args):
        if b.name == name:
            return b
    raise SystemExit(f"unknown policy {name!r}; choose equilibrium or one of "
                     f"{[b.name for b in _baselines(args)]}")


def cmd_solve(args):
    from .io import write_solution
    from .solver import solve_equilibrium
    cfg = _cfg(args, horizon=args.horizon)
    g = LaneGame(cfg)
    t = time.perf_counter()
    sol = solve_equilibrium(g)
    s0 = g.initial_state()
    la, lb = sol.legal[0][s0]
    print(f"[{args.model}] horizon {args.horizon}: V*(s0) = {sol.value:.6g}; {sol.stats.states} decision nodes, "
          f"{sol.stats.mixed_states} mixed; max certified gap {sol.stats.max_gap:.1e}; "
          f"{time.perf_counter() - t:.2f}s")
    print("equilibrium at s0: A", {cfg.label(a): round(float(p), 4) for a, p in zip(la, sol.x[0][s0])},
          "B", {cfg.label(b): round(float(p), 4) for b, p in zip(lb, sol.y[0][s0])})
    out = args.out or (f"results/solution_K{args.horizon}.json" if args.model == "v1"
                       else f"results/revision/solution_K{args.horizon}.json")
    write_solution(sol, out, include_q=not args.no_q)
    print("wrote", out)


def cmd_scenarios(args):
    from .scenarios import all_scenarios, run_scenario
    cfg = _cfg(args)
    bad = 0
    for sc in all_scenarios(cfg):
        ok, detail, lines, _ = run_scenario(sc, cfg)
        bad += not ok
        origin = "reachable start" if sc.start_reachable else "constructed start"
        print(f"\n[{'PASS' if ok else 'FAIL'}] {sc.name} ({origin}): {sc.description}\n  {detail}")
        print("\n".join(lines))
    sys.exit(1 if bad else 0)


def cmd_exploit(args):
    from .solver import solve_equilibrium
    g = LaneGame(_cfg(args, horizon=args.horizon))
    sol = solve_equilibrium(g)
    print(f"[{args.model}] V*(s0) = {sol.value:.6g}")
    for b in _baselines(args):
        e = exploitability(g, sol.layers, b, 0, sol.value)
        print(f"  {b.name:18s} exploitability {e.exploitability:8.3f}   vs equilibrium "
              f"{evaluate(g, sol.layers, b, sol.policy(1)):8.3f}")


def cmd_simulate(args):
    from .simulate import simulate
    from .solver import solve_equilibrium
    g = LaneGame(_cfg(args, horizon=args.horizon))
    sol = solve_equilibrium(g)
    pa, pb = _policy(args.a, sol, 0, args), _policy(args.b, sol, 1, args)
    exact = evaluate(g, sol.layers, pa, pb)
    r = simulate(g, pa, pb, args.episodes, args.seed)
    print(f"exact {exact:.4f}; Monte Carlo {r.mean:.4f} +- {r.stderr:.4f}")
    print(r.summary())


def cmd_reach(args):
    from .reachability import EVENTS, explore_events
    cfg = _cfg(args, horizon=args.horizon)
    rep = explore_events(LaneGame(cfg))
    print(f"[{args.model}] {rep.states} reachable decision nodes, {rep.distinct_states} distinct states")
    for e in EVENTS:
        w = rep.witness.get(e)
        plan = " ".join(f"{cfg.label(a)}/{cfg.label(b)}" for a, b in w.plan) if w else "-"
        print(f"  {e:22s} transitions {rep.counts[e]:8d}  first turn {rep.first_k.get(e, '-')!s:>3}  witness: {plan}")


def _run_experiments(names, available, package):
    names = available if "all" in names else names
    for n in names:
        if n not in available:
            raise SystemExit(f"unknown experiment {n}; choose from {available}")
        print(f"\n===== {n} =====")
        t = time.perf_counter()
        importlib.import_module(f"{package}.{n}").run()
        print(f"[{n}: {time.perf_counter() - t:.1f}s]")


def cmd_experiments(args):
    _run_experiments(args.names, EXPERIMENTS, "lanegame.experiments")


def cmd_experiments_v1(args):
    _run_experiments(args.names, EXPERIMENTS_V1, "lanegame.experiments.v1")


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m lanegame")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, horizon=True):
        s = sub.add_parser(name)
        s.add_argument("--model", choices=["revised", "v1"], default="revised")
        if horizon:
            s.add_argument("--horizon", type=int, default=32)
        s.set_defaults(fn=fn)
        return s

    s = add("solve", cmd_solve)
    s.add_argument("--out")
    s.add_argument("--no-q", action="store_true", help="omit action-value matrices from the JSON")
    add("scenarios", cmd_scenarios, horizon=False)
    add("exploit", cmd_exploit)
    s = add("simulate", cmd_simulate)
    s.add_argument("--a", default="equilibrium")
    s.add_argument("--b", default="equilibrium")
    s.add_argument("--episodes", type=int, default=2000)
    s.add_argument("--seed", type=int, default=0)
    add("reach", cmd_reach)
    s = sub.add_parser("experiments")
    s.add_argument("names", nargs="+")
    s.set_defaults(fn=cmd_experiments)
    s = sub.add_parser("experiments-v1")
    s.add_argument("names", nargs="+")
    s.set_defaults(fn=cmd_experiments_v1)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
