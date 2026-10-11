"""Command-line interface.

  python -m lanegame solve [--model siege|revised|v1] [--horizon 32] [--out results/siege/solution_K32.json]
  python -m lanegame scenarios [--model siege|revised|v1]
  python -m lanegame exploit [--model siege|revised|v1] [--horizon 32]
  python -m lanegame simulate --a equilibrium --b uniform [--model ...] [--episodes 2000 --seed 0]
  python -m lanegame reach [--model siege|revised|v1] [--horizon 32]
  python -m lanegame strategy [--horizon 24] [--vision-turns 6]   (Vision -> Shove -> Crash -> Pressure witness)
  python -m lanegame env-demo [--model siege|revised|v1|champions] --a aggressive --b passive_safe [--episodes 200 --seed 42 --exact --trace t.json]
                                             (two scripted agents through the RL environment, lanegame.env)
  python -m lanegame experiments-siege all | controlled siege_options ... (siege model, results/siege/)
  python -m lanegame experiments-champions all | invariants tournament skill concerns benchmark   (champion mode, results/champions/)
  python -m lanegame experiments all | before_after model_comparison ...   (previous revision, results/revision/)
  python -m lanegame experiments-v1 all | audit_legacy model_comparison ... (original model, results/)

The default model ("siege") is GameConfig(); "revised" is GameConfig.v2() and "v1" is GameConfig.v1().
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time

from . import GameConfig, LaneGame
from .baselines import all_baselines, siege_baselines, v1_baselines
from .best_response import evaluate, exploitability

EXPERIMENTS = ["before_after", "model_comparison", "reachability", "farm_economy", "contest_safe",
               "baselines", "sensitivity", "horizon_benchmark", "verification", "figures"]
EXPERIMENTS_V1 = ["audit_legacy", "model_comparison", "horizon_benchmark", "baselines",
                  "equilibrium_structure", "sensitivity", "verification", "figures"]
EXPERIMENTS_SIEGE = ["benchmark", "timing", "controlled", "siege_options", "crash_value", "vision_sweep",
                     "ablations", "sensitivity", "equilibrium", "verification", "figures"]
EXPERIMENTS_CHAMPIONS = ["invariants", "tournament", "skill", "concerns", "benchmark"]


def _cfg(args, **kw) -> GameConfig:
    if args.model == "v1":
        return GameConfig.v1(**kw)
    if args.model == "revised":
        return GameConfig.v2(**kw)
    if args.model == "champions":
        return GameConfig.champions(**kw)
    return GameConfig(**kw)


def _baselines(args):
    if args.model == "v1":
        return v1_baselines()
    return all_baselines() if args.model == "revised" else siege_baselines()


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
    out = args.out or {"v1": f"results/solution_K{args.horizon}.json",
                       "revised": f"results/revision/solution_K{args.horizon}.json",
                       "siege": f"results/siege/solution_K{args.horizon}.json"}[args.model]
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
    from .reachability import events_for, explore_events
    cfg = _cfg(args, horizon=args.horizon)
    rep = explore_events(LaneGame(cfg))
    print(f"[{args.model}] {rep.states} reachable decision nodes, {rep.distinct_states} distinct states")
    for e in events_for(cfg):
        w = rep.witness.get(e)
        plan = " ".join(f"{cfg.label(a)}/{cfg.label(b)}" for a, b in w.plan) if w else "-"
        print(f"  {e:22s} transitions {rep.counts[e]:8d}  first turn {rep.first_k.get(e, '-')!s:>3}  witness: {plan}")


def cmd_strategy(args):
    """Search the game tree from s_0 for the Vision -> Shove -> Crash -> Pressure sequence and print its trace."""
    from .reachability import find_vision_strategy
    cfg = GameConfig(horizon=args.horizon, vision_turns=args.vision_turns, tower_siege_damage=args.tower_damage)
    g = LaneGame(cfg)
    t = time.perf_counter()
    w = find_vision_strategy(g, horizon=args.horizon, require_plate=not args.no_plate, min_crash_hp=args.min_crash_hp)
    print(f"vision {cfg.vision_turns} turns, tower {cfg.tower_siege_damage} HP/tick, horizon {args.horizon}, "
          f"second crash >= {args.min_crash_hp} HP: "
          f"{'FOUND' if w.found else 'not found'} ({w.states_searched} search nodes, {time.perf_counter() - t:.1f}s)")
    if w.found:
        print(f"vision established on turn {w.vision_turn}, crash on turn {w.crash_turn}, pressure under that vision on "
              f"turn {w.k} ({w.turns_after_vision} turns after the vision)")
        print("plan:", " ".join(f"{cfg.label(a)}/{cfg.label(b)}" for a, b in w.plan))
        for row in w.trace:
            print(f"k={row['k']:2d} {row['A']:8s}/{row['B']:8s} siege {row['siege_mass_before']:2d}->{row['siege_mass_after']:2d} "
                  f"obj {row['objective']:10s} vision {row['vision_before']}->{row['vision_after']} plates "
                  f"{row['plates_available']}->{row['plates_after']} (+{row['plates_claimed']}) thinned {row['thinned']} "
                  f"crash {row['crash_hp']:2d} exposure {row['exposure']:+d} p_gank {row['gank_p']:.3f} "
                  f"(unprotected {row['gank_p_without_vision']:.3f})  | {row['lane_after']}")


def cmd_env_demo(args):
    """Two scripted agents play seeded episodes through the simultaneous-move environment (lanegame.env)."""
    from .env import LaneEnv, ScriptedAgent, champion_agents, run_episodes, write_trace
    cfg = _cfg(args, horizon=args.horizon)
    env = LaneEnv(cfg, cache_size=args.cache)
    if cfg.champion_mode:
        agents = champion_agents(cfg)
        pick = lambda n: agents.get(n) or agents.get(f"legacy:{n}") or (_ for _ in ()).throw(
            SystemExit(f"unknown champion-mode agent {n!r}; choose from {sorted(agents)}"))
        stats = run_episodes(env, pick(args.a), pick(args.b), args.episodes, args.seed, record_first=bool(args.trace))
        print(f"[champions] K = {args.horizon}, seed {args.seed}, realized ganks, zero-sum rewards (payoff to A); "
              f"no exact solver in champion mode")
        print(stats.format())
        if args.trace:
            write_trace(stats.first_log, cfg, args.trace)
            print("wrote", args.trace)
        return
    sol = None
    if "equilibrium" in (args.a, args.b):
        from .solver import solve_equilibrium
        print(f"solving the {args.model} game at K = {args.horizon} for the equilibrium policy "
              f"(minutes and gigabytes at K = 32 for the siege model) ...")
        sol = solve_equilibrium(LaneGame(cfg))
    pa, pb = _policy(args.a, sol, 0, args), _policy(args.b, sol, 1, args)
    stats = run_episodes(env, ScriptedAgent(pa), ScriptedAgent(pb), args.episodes, args.seed, record_first=bool(args.trace))
    print(f"[{args.model}] K = {args.horizon}, seed {args.seed}, realized ganks, zero-sum rewards (payoff to A)")
    print(stats.format())
    if args.exact:
        from .best_response import evaluate
        g = LaneGame(cfg)
        layers = sol.layers if sol is not None else g.reachable()
        exact = evaluate(g, layers, pa, pb)
        z = (stats.mean - exact) / stats.stderr if stats.stderr > 0 else float("nan")
        print(f"  exact expected payoff {exact:+.4f}; Monte Carlo {stats.mean:+.4f} +- {stats.stderr:.4f} (z = {z:+.2f})")
    if args.trace:
        write_trace(stats.first_log, cfg, args.trace)
        print("wrote", args.trace)


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


def cmd_experiments_siege(args):
    _run_experiments(args.names, EXPERIMENTS_SIEGE, "lanegame.experiments.siege")


def cmd_experiments_champions(args):
    _run_experiments(args.names, EXPERIMENTS_CHAMPIONS, "lanegame.experiments.champions")


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m lanegame")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, horizon=True):
        s = sub.add_parser(name)
        s.add_argument("--model", choices=["siege", "revised", "v1"], default="siege")
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
    s = sub.add_parser("env-demo")
    s.add_argument("--model", choices=["siege", "revised", "v1", "champions"], default="siege")
    s.add_argument("--horizon", type=int, default=32)
    s.set_defaults(fn=cmd_env_demo)
    s.add_argument("--a", default="aggressive", help="policy of player A: a baseline name or 'equilibrium' "
                   "(champion mode: hold, yield, trade, safe, shove_hold, shove_trade, hold_vision, freeze_hold, "
                   "uniform, or a legacy baseline name)")
    s.add_argument("--b", default="passive_safe")
    s.add_argument("--episodes", type=int, default=200)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--exact", action="store_true",
                   help="also compute the exact expected payoff of the pair (enumerates the reachable game; K <= 16 is cheap)")
    s.add_argument("--trace", help="write the first episode's trace (lanegame.episode JSON) to this path")
    s.add_argument("--cache", type=int, default=0, help="bounded LRU cache of lane steps, in entries (0 = off)")
    s = sub.add_parser("strategy")
    s.add_argument("--horizon", type=int, default=24)
    s.add_argument("--vision-turns", type=int, default=6)
    s.add_argument("--tower-damage", type=int, default=1)
    s.add_argument("--no-plate", action="store_true", help="do not require a plate to be claimed on the pressure turn")
    s.add_argument("--min-crash-hp", type=int, default=1, help="minimum HP of the crash after the vision (4 = a full wave)")
    s.set_defaults(fn=cmd_strategy)
    s = sub.add_parser("experiments-siege")
    s.add_argument("names", nargs="+")
    s.set_defaults(fn=cmd_experiments_siege)
    s = sub.add_parser("experiments-champions")
    s.add_argument("names", nargs="+")
    s.set_defaults(fn=cmd_experiments_champions)
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
