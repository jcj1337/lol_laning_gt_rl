"""Quantitative audit of the original (legacy) environment in legacy/src.

Runs the unmodified legacy LaneEnv with uniformly random actions and measures
the defects identified by code review: float state keys, missing time index,
non-zero-sum rewards, rewards collected repeatedly without a new event, and
freeze applicability anywhere on the lane.
"""

from __future__ import annotations

import importlib
import os
import random
import sys

from .common import md_table, save

LEGACY = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "legacy")


def _load():
    sys.path.insert(0, LEGACY)
    try:
        env = importlib.import_module("src.env")
        agents = importlib.import_module("src.agents")
    finally:
        sys.path.remove(LEGACY)
    return env, agents


def run(episodes: int = 2000, seed: int = 0) -> dict:
    env_mod, agents_mod = _load()
    random.seed(seed)
    SP, SH, F = env_mod.SP, env_mod.SH, env_mod.F
    env = env_mod.LaneEnv(env_mod.LaneParams())
    keys, g_values = set(), set()
    steps = nonzero_sum = 0
    sum_abs = 0.0
    crash_bonus_steps = repeat_crash_bonus = 0
    freeze_bonus = 0.0
    same_key_diff_time = {}
    collisions = 0
    for _ in range(episodes):
        (oy, oo) = env.reset()
        prev_crash = False
        done = False
        while not done:
            a, b = random.choice([SP, SH, F]), random.choice([SP, SH, F])
            w, m = env.w, env.m_self
            # crash bonus term evaluated exactly as legacy reward() does
            raw = env.w + env.delta_w(a) - env.delta_w(b)
            w_next = max(-2, min(2, raw))
            crash = (w_next == 2 and m == 2)
            if crash:
                crash_bonus_steps += 1
                repeat_crash_bonus += prev_crash
            prev_crash = crash
            if a == F and w <= -1:
                freeze_bonus += env.p.b_deny
            key = oy
            t = env.t
            if key in same_key_diff_time and same_key_diff_time[key] != t:
                collisions += 1
            same_key_diff_time.setdefault(key, t)
            (oy, oo), (ry, ro), done = env.step(a, b)
            keys.add(oy)
            g_values.add(env.g)
            steps += 1
            if abs(ry + ro) > 1e-12:
                nonzero_sum += 1
            sum_abs += abs(ry + ro)
    # Freeze legality: legacy allows F with the wave at the enemy tower (w = +2) and moves it back.
    env.reset()
    env.w = 2
    env.step(F, SP)
    freeze_at_enemy_tower_moves_wave = env.w
    # Terminal update: QLearningAgent.update has no `done` argument and always bootstraps.
    import inspect
    upd = inspect.signature(agents_mod.QLearningAgent.update)
    data = {
        "episodes": episodes, "steps": steps,
        "distinct_state_keys": len(keys),
        "distinct_float_gold_values": len(g_values),
        "fraction_steps_not_zero_sum": nonzero_sum / steps,
        "mean_abs_reward_sum": sum_abs / steps,
        "crash_bonus_steps": crash_bonus_steps,
        "crash_bonus_on_consecutive_steps": repeat_crash_bonus,
        "freeze_deny_bonus_per_episode": freeze_bonus / episodes,
        "state_keys_seen_at_different_times": collisions,
        "freeze_from_enemy_tower_moves_wave_to": freeze_at_enemy_tower_moves_wave,
        "q_update_signature": str(upd),
    }
    rows = [
        ("Distinct observation keys (continuous gold in key)", len(keys)),
        ("Distinct float gold values observed", len(g_values)),
        ("Steps where r_A + r_B != 0 (not zero-sum)", f"{100 * nonzero_sum / steps:.1f}%"),
        ("Mean |r_A + r_B| per step", f"{sum_abs / steps:.3f}"),
        ("Steps paying the crash bonus", crash_bonus_steps),
        ("...of which immediately after another crash bonus (no new event)", repeat_crash_bonus),
        ("Freeze 'deny' bonus collected per episode (random play)", f"{freeze_bonus / episodes:.2f}"),
        ("Observation keys revisited at a different time step t", collisions),
        ("Wave position after Freeze from the enemy tower (w=+2)", freeze_at_enemy_tower_moves_wave),
        ("QLearningAgent.update signature (no terminal flag)", f"`{upd}`"),
    ]
    md = "# Legacy environment audit\n\n" + md_table(["Measurement", "Value"], rows)
    save("audit_legacy", data, md)
    return data


if __name__ == "__main__":
    run()
