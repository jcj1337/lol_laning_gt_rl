"""Command-line interface.

  python -m lanegame solve [--horizon 32] [--out results/solution_K32.json]
  python -m lanegame scenarios
  python -m lanegame exploit [--horizon 32]
  python -m lanegame simulate --a equilibrium --b uniform [--episodes 2000 --seed 0]
  python -m lanegame experiments all | audit_legacy model_comparison horizon_benchmark ...
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time

from . import GameConfig, LaneGame
from .baselines import all_baselines
from .best_response import evaluate, exploitability
from .config import ACTION_SHORT

EXPERIMENTS = ["audit_legacy", "model_comparison", "horizon_benchmark", "baselines",
               "equilibrium_structure", "sensitivity", "verification", "figures"]


def _policy(name, sol, side):
    if name == "equilibrium":
        return sol.policy(side)
    for b in all_baselines():
        if b.name == name:
            return b
    raise SystemExit(f"unknown policy {name!r}; choose equilibrium or one of "
                     f"{[b.name for b in all_baselines()]}")


def cmd_solve(args):
    from .io import write_solution
    from .solver import solve_equilibrium
    g = LaneGame(GameConfig(horizon=args.horizon))
    t = time.perf_counter()
    sol = solve_equilibrium(g)
    s0 = g.initial_state()
    la, lb = sol.legal[0][s0]
    print(f"horizon {args.horizon}: V*(s0) = {sol.value:.6g}; {sol.stats.states} decision nodes, "
          f"{sol.stats.mixed_states} mixed; max certified gap {sol.stats.max_gap:.1e}; "
          f"{time.perf_counter() - t:.2f}s")
    print("equilibrium at s0: A", {ACTION_SHORT[a]: round(float(p), 4) for a, p in zip(la, sol.x[0][s0])},
          "B", {ACTION_SHORT[b]: round(float(p), 4) for b, p in zip(lb, sol.y[0][s0])})
    out = args.out or f"results/solution_K{args.horizon}.json"
    write_solution(sol, out, include_q=not args.no_q)
    print("wrote", out)


def cmd_scenarios(args):
    from .scenarios import all_scenarios, run_scenario
    cfg = GameConfig()
    bad = 0
    for sc in all_scenarios(cfg):
        ok, detail, lines, _ = run_scenario(sc, cfg)
        bad += not ok
        print(f"\n[{'PASS' if ok else 'FAIL'}] {sc.name}: {sc.description}\n  {detail}")
        print("\n".join(lines))
    sys.exit(1 if bad else 0)


def cmd_exploit(args):
    from .solver import solve_equilibrium
    g = LaneGame(GameConfig(horizon=args.horizon))
    sol = solve_equilibrium(g)
    print(f"V*(s0) = {sol.value:.6g}")
    for b in all_baselines():
        e = exploitability(g, sol.layers, b, 0, sol.value)
        print(f"  {b.name:18s} exploitability {e.exploitability:8.3f}   vs equilibrium "
              f"{evaluate(g, sol.layers, b, sol.policy(1)):8.3f}")


def cmd_simulate(args):
    from .simulate import simulate
    from .solver import solve_equilibrium
    g = LaneGame(GameConfig(horizon=args.horizon))
    sol = solve_equilibrium(g)
    pa, pb = _policy(args.a, sol, 0), _policy(args.b, sol, 1)
    exact = evaluate(g, sol.layers, pa, pb)
    r = simulate(g, pa, pb, args.episodes, args.seed)
    print(f"exact {exact:.4f}; Monte Carlo {r.mean:.4f} +- {r.stderr:.4f}")
    print(r.summary())


def cmd_experiments(args):
    names = EXPERIMENTS if "all" in args.names else args.names
    for n in names:
        if n not in EXPERIMENTS:
            raise SystemExit(f"unknown experiment {n}; choose from {EXPERIMENTS}")
        print(f"\n===== {n} =====")
        t = time.perf_counter()
        importlib.import_module(f"lanegame.experiments.{n}").run()
        print(f"[{n}: {time.perf_counter() - t:.1f}s]")


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m lanegame")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("solve")
    s.add_argument("--horizon", type=int, default=32)
    s.add_argument("--out")
    s.add_argument("--no-q", action="store_true", help="omit action-value matrices from the JSON")
    s.set_defaults(fn=cmd_solve)
    s = sub.add_parser("scenarios")
    s.set_defaults(fn=cmd_scenarios)
    s = sub.add_parser("exploit")
    s.add_argument("--horizon", type=int, default=32)
    s.set_defaults(fn=cmd_exploit)
    s = sub.add_parser("simulate")
    s.add_argument("--a", default="equilibrium")
    s.add_argument("--b", default="equilibrium")
    s.add_argument("--horizon", type=int, default=32)
    s.add_argument("--episodes", type=int, default=2000)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_simulate)
    s = sub.add_parser("experiments")
    s.add_argument("names", nargs="+")
    s.set_defaults(fn=cmd_experiments)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
