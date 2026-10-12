"""S3: wave mechanics and the pre-training Freeze gate on the implemented M4 environment (design §6.6; plan §7 S3).

    python -m lanegame.experiments.m4.freeze_gate [--rollouts 50] [--turns 120] [--procs N]

S3a (mechanics): the design §6.4 parked-front table on the implemented resolver; break windows of a Shove + Hold
breaker; the exhaustive lane comparison of Freeze and Slow Push in every Freeze-legal reachable situation (a
maintained Freeze is M2's Freeze in the lane step), with the share of those situations in which a Hold freezer can
maintain it; Freeze statistics of scripted play (Freeze-legal turns, held turns, breaks per 100 turns).

S3b (gate): >= 100 distinct reachable Freeze-legal states of the defender A (lane, M4 champion state, phase), each
with a witness (environment seed + action prefix from s0): scripted prefixes as in the design prototype (B shoves j
turns while A farms; A recalls at turn r; seeded random legal prefixes), now with chance events. From each state the
defender plays freeze / slowpush / cshove (identical behavior rules: Hold, low-HP recall, Trade on a positive duel
edge) against each attacker (park_hold, park_safe, breaker, presser, stacker, adaptive_breaker; plus `trader`,
reported separately), for H turns, with 50 paired rollouts per scenario: rollout m of every policy pair uses the same
environment seed, so all pairs see identical chance vectors turn by turn (common random numbers). Reported per pair:
B's first crash, crashes, B's tower damage and plates, gold and XP of both, exposure (gank occurrence, expected and
realized gank HP, trade HP), deaths, Freeze turns chosen / maintained, and the discounted relative return to A
(gamma = the game's). Gates G1-G3 (design §6.6):

  G1  |F - SP| > 0.05 in >= 20% of the scenarios, against >= 2 attacker policies;
  G2  some attacker with F - max(SP, cshove) >= +0.2 in some scenario AND some attacker with <= -0.2 in some scenario;
  G3  a reaching Shove with blob + 4 >= 10 is never held (tick-exact: recorded holds = holds the rule allows on the
      replayed tick states, over every gate turn), and the measured break
      windows of a Shove breaker against fronts of >= 3 HP (reported against the §6.4-6.5 windows).

Scenario-set means are E2-style diagnostics (the set is not mirror-closed and is not an evaluation archive).
"""

from __future__ import annotations

import argparse
import collections
import random
from typing import Dict, List

import numpy as np

from ...champions import champion_legal_actions, objective_reachable
from ...config import B_HOLD, B_RECALL, B_SAFE, F, NONE, PRESSURE, SH, SP, GameConfig, champ_act
from ...env import LaneEnv
from ...env.agents_m4 import gate_attackers, gate_defenders, m4_suite
from ...mechanics import _blob_hp, wave_actions
from ...scenarios_m4 import DESIGN_HOLD_TABLE, hold_run
from .common import Timer, fmt_ci, md_table, mean_se, meta, pool_map, save

REQUIRED_ATTACKERS = ("park_hold", "park_safe", "breaker", "presser", "stacker", "adaptive_breaker")
DEFENDERS = ("freeze", "slowpush", "cshove")
RECALL = champ_act(SP, B_RECALL)


# ---------------------------------------------------------------------- reachable Freeze-legal scenarios
def _pick(legal, wave, beh, obj=NONE):
    if len(legal) == 1:
        return legal[0]
    for w in (wave, SP):
        for o in ((obj, NONE) if obj != NONE else (NONE,)):
            a = champ_act(w, beh, o)
            if a in legal:
                return a
    return champ_act(SP, B_HOLD)


def _freeze_legal_for_a(env) -> bool:
    return env.champ.a.absent == 0 and F in wave_actions(env.state, 0, env.cfg)


def find_scenarios(cfg: GameConfig, n_random: int = 400, seed: int = 20261011) -> List[dict]:
    """Distinct reachable Freeze-legal states for A, each with a witness (seed, actions)."""
    found: Dict[tuple, dict] = {}
    env = LaneEnv(cfg, max_turns=60, cache_size=20_000)

    def record(origin, env_seed, actions):
        # distinct as in the design prototype: lane, champion cells and absences, phase (the M4 record's XP, gold
        # and HP would otherwise make nearly every prefix state distinct)
        c = env.champ
        key = (env.state, c.a.pos, c.b.pos, c.a.absent, c.b.absent, env.k % cfg.phase_period)
        if key not in found:
            found[key] = {"lane": env.state.to_list(), "champ": env.champ.to_list(), "phase": env.k % cfg.phase_period,
                          "k": env.k, "witness": {"seed": env_seed, "actions": [list(a) for a in actions]},
                          "origin": origin}

    def run(origin, env_seed, turns, policy):
        env.reset(seed=env_seed)
        actions = []
        for k in range(turns):
            la, lb = env.legal(0), env.legal(1)
            a, b = policy(k, la, lb)
            r = env.step(a, b)
            actions.append((a, b))
            if _freeze_legal_for_a(env):
                record(origin, env_seed, actions)
            if r.done:
                break
    for j in range(1, 13):                                       # B shoves for j turns while A farms
        for s in range(3):
            run(f"B shoves {j}", 1000 * j + s, j + 6,
                lambda k, la, lb, j=j: (_pick(la, SP, B_HOLD), _pick(lb, SH if k < j else SP, B_HOLD)))
    for r0 in range(0, 9):                                       # A recalls at turn r0 while B farms or shoves
        for bw in (SP, SH):
            for s in range(3):
                run(f"A recalls {r0}, B {'SH' if bw == SH else 'SP'}", 50_000 + 100 * r0 + 10 * bw + s, r0 + 12,
                    lambda k, la, lb, r0=r0, bw=bw: (RECALL if (k == r0 and RECALL in la) else _pick(la, SP, B_HOLD),
                                                     _pick(lb, bw, B_HOLD)))
    rng = random.Random(seed)
    for g in range(n_random):                                    # seeded random legal prefixes
        pa, pb = rng.random(), rng.random()
        turns = rng.randint(4, 40)

        def pol(k, la, lb, pa=pa, pb=pb):
            out = []
            for legal, p_sh, s in ((la, pa, 0), (lb, pb, 1)):
                if len(legal) == 1:
                    out.append(legal[0])
                    continue
                if RECALL in legal and rng.random() < 0.05:
                    out.append(RECALL)
                    continue
                w = SH if rng.random() < p_sh else (F if rng.random() < 0.3 else SP)
                beh = B_SAFE if rng.random() < 0.1 else B_HOLD
                obj = PRESSURE if rng.random() < 0.7 else NONE
                out.append(_pick(legal, w, beh, obj))
            return out[0], out[1]
        run(f"random {g}", 900_000 + g, turns, pol)
    return list(found.values())


# ---------------------------------------------------------------------- rollouts
def _play(env, scn, defender, attacker, seed, turns, gamma):
    from ...env.starts import StartState
    from ...state import State
    from ...champions_m4 import M4State
    st = StartState(State.from_list(scn["lane"]), M4State.from_list(scn["champ"]), scn["phase"])
    env.max_turns = turns
    obs = env.reset(seed=seed, root=st.lane, k0=st.phase, champ=st.champ)
    for ag in (defender, attacker):
        if hasattr(ag, "reset"):
            ag.reset()
    cfg = env.cfg
    out = collections.Counter()
    disc, w = 0.0, 1.0
    first_crash = None
    for t in range(turns):
        a, b = defender.act(obs[0]), attacker.act(obs[1])
        lane0 = env.state
        r = env.step(a, b)
        i = r.info
        if hasattr(attacker, "notify"):
            attacker.notify(i, 1)
        disc += w * r.rewards[0]
        w *= gamma
        if i["crash_hp"][1] > 0:
            out["crashes_b"] += 1
            if first_crash is None:
                first_crash = t
        for s, tag in ((0, "a"), (1, "b")):
            out[f"gold_{tag}"] += i["farm"][s]
            out[f"plate_gold_{tag}"] += i["plate_gold"][s]
            out[f"xp_{tag}"] += i["xp_gain"][s] + i["xp_discarded"][s]
            out[f"gank_p_{tag}"] += i["gank_p"][s]
            out[f"gank_hp_{tag}"] += i["gank_damage"][s] / 100.0
            out[f"trade_hp_taken_{tag}"] += (i["trade_damage"][1 - s]) / 100.0
            out[f"tower_shot_hp_{tag}"] += i["tower_damage"][s] / 100.0
            out[f"deaths_{tag}"] += int(i["died"][s])
            out[f"tower_dmg_{tag}"] += i["tower_damage_dealt"][s]
            out[f"plates_{tag}"] += i["plates_broken"][s]
        out["freeze_chosen"] += int(i["waves"][0] == F)
        out["freeze_maintained"] += int(i["freeze_maintained"][0])
        # G3 rule check, tick-exact: a reaching Shove whose blob + push reaches the threshold is never held
        if i["effective_waves"][0] == F and i["waves"][1] == SH:
            allowed, must_break = hold_ticks(lane0, i["k"], i["lane_codes"], i["champion_reach"], cfg)
            out["g3_checked"] += must_break
            out["g3_violations"] += int(i["events"]["freeze_held"][1] != allowed)
        if r.done:
            break
        obs = r.obs
    out["disc"] = disc
    out["first_crash"] = turns if first_crash is None else first_crash
    out["crashed"] = int(first_crash is not None)
    return dict(out)


def hold_ticks(lane0, k, codes, reach, cfg):
    """Replays the turn's authoritative lane step tick by tick (record=True) and returns, for B's front against A's
    effective Freeze, (holds the rule allows, ticks on which a reaching Shove with blob + push >= threshold wants to
    enter A's tower cell and must therefore break the hold). Tick 1 starts from the turn's state; later ticks start
    from the previous tick's trace."""
    from ...mechanics import _fronts_in_contact, step_turn
    res = step_turn(lane0, k, codes[0], codes[1], cfg, champion_reach=tuple(reach), record=True)
    P = [[], []]
    for s in (0, 1):
        fx, fh, rears, sg = lane0.side(s)[:4]
        if fh > 0:
            P[s].append([fx, fh])
        P[s] += [[x, cfg.wave_hp] for x in rears]
    starts = [(P, [lane0.asg, lane0.bsg])] + [((tr["A"], tr["B"]), tr["siege"]) for tr in res.trace[:-1]]
    T = cfg.tower
    allowed = must_break = 0
    for (pa, pb), siege in starts:
        if not pb or _fronts_in_contact([pa, pb]):
            continue
        x = pb[0][0]
        if siege[0] > 0 and x >= -T - 1:
            continue                                          # waiting behind its tower: no movement intent
        if not x + 1 > -cfg.freeze_line:
            continue                                          # not about to enter the freezer's tower cell
        push = cfg.freeze_break_push[SH] if x <= reach[1] else 0
        if _blob_hp(pb, cfg) + push < cfg.freeze_break_threshold:
            allowed += 1
        elif push:
            must_break += 1
    return allowed, must_break


def run_scenario(args):
    idx, scn, rollouts, turns, root, cfg_dict = args
    cfg = GameConfig.from_dict(cfg_dict)
    env = LaneEnv(cfg, cache_size=50_000)
    defs, atts = gate_defenders(), gate_attackers()
    res = {}
    for an, att in atts.items():
        for dn, dfn in defs.items():
            rows = [_play(env, scn, dfn, att, np.random.SeedSequence(root, spawn_key=(idx, m)), turns, cfg.gamma)
                    for m in range(rollouts)]
            res[(dn, an)] = rows
    return idx, {f"{d}|{a}": v for (d, a), v in res.items()}


# ---------------------------------------------------------------------- S3a
def s3a(cfg: GameConfig) -> dict:
    hold = {}
    for (e, w), (fz, sp) in DESIGN_HOLD_TABLE.items():
        key = f"{e}_{'SP' if w == SP else 'SH'}"
        hold[key] = {"design_freeze": fz, "design_slowpush": sp,
                     "freeze": [hold_run(e, w, ph, F, cfg) for ph in range(4)],
                     "slowpush": [hold_run(e, w, ph, SP, cfg)["first_crash"] for ph in range(4)]}
    return {"hold_table": hold}


def s3a_enumeration(cfg: GameConfig) -> dict:
    """Exhaustive: in every reachable (lane, phase) and side where Freeze is legal, does Freeze differ from Slow Push
    in the lane step for some own objective / reach and enemy option (the maintained case), and could a Hold freezer
    maintain it (its probed farm spot within farm_reach of the freeze line)?"""
    from ...lane_enum import reachable_lane, side_options
    from ...mechanics import step_turn
    seen = reachable_lane(cfg)
    legal_sit = differ = maintainable = differ_maintainable = 0
    for st, ph in seen:
        for s in (0, 1):
            if F not in wave_actions(st, s, cfg):
                continue
            legal_sit += 1
            own_opts = [o for o in side_options(st, s, cfg) if o[0] == F]
            enemy = side_options(st, 1 - s, cfg)
            d = False
            for (w, o, r) in own_opts:
                for (we, oe, re) in enemy:
                    acts_f = [None, None]
                    acts_p = [None, None]
                    acts_f[s], acts_f[1 - s] = F + 6 * o, we + 6 * oe
                    acts_p[s], acts_p[1 - s] = SP + 6 * o, we + 6 * oe
                    reach = [None, None]
                    reach[s], reach[1 - s] = r, re
                    rf = step_turn(st, ph, acts_f[0], acts_f[1], cfg, champion_reach=tuple(reach))
                    rp = step_turn(st, ph, acts_p[0], acts_p[1], cfg, champion_reach=tuple(reach))
                    if (rf.state, rf.crash_hp, rf.kills) != (rp.state, rp.crash_hp, rp.kills):
                        d = True
                        break
                if d:
                    break
            differ += d
            # Hold maintains when its farm spot (own front after the full-support probe) is within reach of the line
            probe_acts = [SP, SP]
            probe_acts[s] = F
            pr = step_turn(st, ph, probe_acts[0], probe_acts[1], cfg, champion_reach=(cfg.tower, cfg.tower))
            fx, fh = pr.state.side(s)[:2]
            m = fh == 0 or abs(max(-cfg.tower, min(cfg.tower, fx)) - cfg.freeze_line) <= cfg.farm_reach
            maintainable += m
            differ_maintainable += d and m
    return {"reachable_pairs": len(seen), "freeze_legal_situations": legal_sit, "freeze_differs": differ,
            "share_differs": differ / max(legal_sit, 1), "hold_maintainable": maintainable,
            "differs_and_hold_maintainable": differ_maintainable,
            "design_m2_reference": {"freeze_legal": 15458, "differ": 1126, "share": 0.073}}


def scripted_freeze_stats(cfg: GameConfig, games: int = 30, turns: int = 400) -> dict:
    """Freeze statistics of scripted play: Freeze-legal turns, chosen, maintained, held and broken per 100 turns."""
    suite = {**m4_suite(), **{"g_" + k: v for k, v in gate_attackers().items()}}
    pairs = [("freezer", "hold_farmer"), ("freezer", "g_breaker"), ("freezer", "g_presser"), ("freezer", "g_stacker"),
             ("freezer", "aggressive_trader"), ("hold_farmer", "hold_farmer")]
    env = LaneEnv(cfg, max_turns=turns, cache_size=50_000)
    out = {}
    from ...env import play_episode
    for x, y in pairs:
        c = collections.Counter()
        for g in range(games):
            log = play_episode(env, suite[x], suite[y], 7_000 + g, record=True)
            for i in log.turns:
                c["turns"] += 1
                c["freeze_legal"] += int(any(a % 3 == F for a in i["legal"][0] if a != 45))
                c["freeze_chosen"] += int(i["waves"][0] == F)
                c["freeze_maintained"] += int(i["freeze_maintained"][0])
                c["held"] += int(i["events"]["freeze_held"][1] > 0)
                c["broken"] += int(i["events"]["freeze_broken"][1] > 0)
                c["crashes_b"] += int(i["crash_hp"][1] > 0)
        n = c["turns"] / 100
        out[f"{x} vs {y}"] = {k: c[k] / n for k in c if k != "turns"}
    return out


# ---------------------------------------------------------------------- gates and report
def summarize(scns, results, turns, rollouts) -> dict:
    atts = list(gate_attackers())
    per_att = {}
    for an in atts:
        d_sp, d_best, se_sp = [], [], []
        rows = {d: [] for d in DEFENDERS}
        for idx, res in results:
            means = {}
            for dn in DEFENDERS:
                vals = res[f"{dn}|{an}"]
                ks = {k for v in vals for k in v}
                means[dn] = {k: float(np.mean([v.get(k, 0) for v in vals])) for k in ks}
                rows[dn].append(means[dn])
            f = np.array([v["disc"] for v in res[f"freeze|{an}"]])
            s = np.array([v["disc"] for v in res[f"slowpush|{an}"]])
            c = np.array([v["disc"] for v in res[f"cshove|{an}"]])
            d_sp.append(float((f - s).mean()))
            se_sp.append(float((f - s).std(ddof=1) / np.sqrt(len(f))) if len(f) > 1 else 0.0)
            d_best.append(float(f.mean() - max(s.mean(), c.mean())))
        d_sp, d_best, se_sp = np.array(d_sp), np.array(d_best), np.array(se_sp)
        keys = sorted({k for dn in DEFENDERS for r in rows[dn] for k in r})
        agg = {dn: {k: float(np.mean([r.get(k, 0.0) for r in rows[dn]])) for k in keys} for dn in DEFENDERS}
        per_att[an] = {
            "freeze_minus_slowpush": {"mean": float(d_sp.mean()), "min": float(d_sp.min()), "max": float(d_sp.max()),
                                      "share_differs": float(np.mean(np.abs(d_sp) > 0.05)),
                                      "share_significant": float(np.mean(np.abs(d_sp) > 2 * np.maximum(se_sp, 1e-12)))},
            "freeze_vs_best": {"better": int(np.sum(d_best > 0.05)), "equal": int(np.sum(np.abs(d_best) <= 0.05)),
                               "worse": int(np.sum(d_best < -0.05)), "max": float(d_best.max()),
                               "min": float(d_best.min()), "count_ge_0.2": int(np.sum(d_best >= 0.2)),
                               "count_le_-0.2": int(np.sum(d_best <= -0.2))},
            "means": agg}
    req = [a for a in REQUIRED_ATTACKERS]
    g1_atts = [a for a in req if per_att[a]["freeze_minus_slowpush"]["share_differs"] >= 0.20]
    g2_pos = [a for a in req if per_att[a]["freeze_vs_best"]["count_ge_0.2"] > 0]
    g2_neg = [a for a in req if per_att[a]["freeze_vs_best"]["count_le_-0.2"] > 0]
    g3_checked = sum(v.get("g3_checked", 0) for _, res in results for rows in res.values() for v in rows)
    g3_viol = sum(v.get("g3_violations", 0) for _, res in results for rows in res.values() for v in rows)
    # break windows: scenarios with B's front >= 3 HP at A's -1 at the start, Shove attackers, Freeze defender
    windows = {}
    for an in ("breaker", "presser", "adaptive_breaker"):
        fc = []
        for (idx, res) in results:
            lane = scns[idx]["lane"]
            if lane[5] >= 3 and -lane[4] == -1:
                fc += [v["first_crash"] for v in res[f"freeze|{an}"]]
        fc = np.array(fc)
        windows[an] = {"n": int(len(fc)), "crashed_within_6": float(np.mean(fc <= 6)) if len(fc) else None,
                       "crashed_within_12": float(np.mean(fc <= 12)) if len(fc) else None,
                       "crashed_within_H": float(np.mean(fc < turns)) if len(fc) else None,
                       "median_first_crash": float(np.median(fc)) if len(fc) else None}
    gates = {"G1": {"pass": len(g1_atts) >= 2, "attackers": g1_atts},
             "G2": {"pass": bool(g2_pos) and bool(g2_neg), "attackers_freeze_better": g2_pos,
                    "attackers_freeze_worse": g2_neg},
             "G3": {"rule_checked_turns": int(g3_checked), "rule_violations": int(g3_viol),
                    "rule_pass": g3_viol == 0 and g3_checked > 0, "break_windows": windows}}
    return {"per_attacker": per_att, "gates": gates}


def markdown(data) -> str:
    g = data["gates"]
    L = [f"# S3 — Wave mechanics and the Freeze gate (M4.2)", "",
         f"Commit `{data['meta']['commit'][:10]}`; {data['n_scenarios']} reachable Freeze-legal scenarios; "
         f"{data['rollouts']} paired rollouts (common random numbers) of {data['turns']} turns per scenario and policy "
         f"pair; gamma = {data['gamma']}. Command: `{data['meta']['command']}`.", "",
         "## Gates", "",
         f"- **G1 (distinct)**: {'PASS' if g['G1']['pass'] else 'FAIL'} — |F − SP| > 0.05 in ≥ 20% of the scenarios "
         f"against: {', '.join(g['G1']['attackers']) or 'none'}.",
         f"- **G2 (non-trivial sign)**: {'PASS' if g['G2']['pass'] else 'FAIL'} — Freeze ≥ +0.2 better than the best "
         f"alternative somewhere against: {', '.join(g['G2']['attackers_freeze_better']) or 'none'}; ≥ 0.2 worse "
         f"somewhere against: {', '.join(g['G2']['attackers_freeze_worse']) or 'none'}.",
         f"- **G3 (breakable)**: rule {'PASS' if g['G3']['rule_pass'] else 'FAIL'} — {g['G3']['rule_checked_turns']} "
         f"gate turns with a reaching Shove and blob + 4 ≥ 10, {g['G3']['rule_violations']} held. Break windows below.",
         "", "## Freeze against the alternatives, per attacker", ""]
    rows = []
    for an, v in data["summary"]["per_attacker"].items():
        fs, fb, m = v["freeze_minus_slowpush"], v["freeze_vs_best"], v["means"]
        rows.append([an + ("" if an in REQUIRED_ATTACKERS else " (extra)"), f"{m['freeze']['disc']:.2f}",
                     f"{m['slowpush']['disc']:.2f}", f"{m['cshove']['disc']:.2f}",
                     f"{fs['mean']:+.2f} ({fs['min']:+.2f}, {fs['max']:+.2f})", f"{fs['share_differs']:.0%}",
                     f"{fb['better']}/{fb['equal']}/{fb['worse']}",
                     f"{m['freeze']['crashes_b']:.2f} / {m['slowpush']['crashes_b']:.2f}",
                     f"{m['freeze']['plates_b']:.2f} / {m['slowpush']['plates_b']:.2f}",
                     f"{m['freeze']['gold_b']:.1f} / {m['slowpush']['gold_b']:.1f}",
                     f"{m['freeze']['xp_b']:.0f} / {m['slowpush']['xp_b']:.0f}"])
    L.append(md_table(["B policy", "Freeze", "Slow Push", "cond. Shove", "F − SP mean (min, max)", "F ≠ SP",
                       "F vs best: better/equal/worse", "B crashes (F / SP)", "B plates (F / SP)",
                       "B gold (F / SP)", "B XP (F / SP)"], rows))
    L += ["", "Exposure and combat (means per rollout, Freeze defender / Slow Push defender):", ""]
    rows = []
    for an, v in data["summary"]["per_attacker"].items():
        m = v["means"]
        rows.append([an, f"{m['freeze']['gank_p_a']:.2f} / {m['slowpush']['gank_p_a']:.2f}",
                     f"{m['freeze']['gank_p_b']:.2f} / {m['slowpush']['gank_p_b']:.2f}",
                     f"{m['freeze']['gank_hp_b']:.2f} / {m['slowpush']['gank_hp_b']:.2f}",
                     f"{m['freeze']['trade_hp_taken_b']:.2f} / {m['slowpush']['trade_hp_taken_b']:.2f}",
                     f"{m['freeze']['tower_shot_hp_b']:.2f} / {m['slowpush']['tower_shot_hp_b']:.2f}",
                     f"{m['freeze']['deaths_a']:.2f} / {m['slowpush']['deaths_a']:.2f}",
                     f"{m['freeze']['deaths_b']:.2f} / {m['slowpush']['deaths_b']:.2f}",
                     f"{m['freeze']['tower_dmg_b']:.0f} / {m['slowpush']['tower_dmg_b']:.0f}",
                     f"{m['freeze']['freeze_chosen']:.1f} / {m['freeze']['freeze_maintained']:.1f}",
                     f"{m['freeze']['first_crash']:.1f} / {m['slowpush']['first_crash']:.1f}"])
    L.append(md_table(["B policy", "A gank occurrences", "B gank occurrences", "B gank HP", "B trade HP taken",
                       "B tower-shot HP", "A deaths", "B deaths", "B tower damage (centi)",
                       "Freeze chosen / maintained (turns)", "B first crash (H = never)"], rows))
    L += ["", "## G3 break windows (scenarios with B's front ≥ 3 HP at A's −1)", ""]
    rows = [[a, w["n"], w["crashed_within_6"], w["crashed_within_12"], w["crashed_within_H"], w["median_first_crash"]]
            for a, w in g["G3"]["break_windows"].items()]
    L.append(md_table(["attacker", "rollouts", "broken ≤ 6 turns", "≤ 12", "within H", "median first crash"], rows))
    L += ["", "## S3a — the parked-front table on the implemented resolver (first crash; None = 24 turns parked)", ""]
    rows = []
    for key, v in data["s3a"]["hold_table"].items():
        rows.append([key, v["design_freeze"], [r["first_crash"] for r in v["freeze"]], v["design_slowpush"],
                     v["slowpush"]])
    L.append(md_table(["e, B wave", "design: Freeze", "implemented: Freeze", "design: Slow Push",
                       "implemented: Slow Push"], rows))
    if "enumeration" in data:
        e = data["enumeration"]
        L += ["", "## S3a — exhaustive lane comparison", "",
              f"{e['freeze_legal_situations']} Freeze-legal (state, phase, side) situations in {e['reachable_pairs']} "
              f"reachable pairs; Freeze differs from Slow Push (maintained case) in {e['freeze_differs']} "
              f"({e['share_differs']:.1%}; M2 reference 1126 of 15458, 7.3%). A Hold freezer can maintain the Freeze "
              f"in {e['hold_maintainable']} of them ({e['differs_and_hold_maintainable']} that also differ)."]
    L += ["", "## Freeze statistics of scripted play (per 100 turns)", ""]
    keys = ["freeze_legal", "freeze_chosen", "freeze_maintained", "held", "broken", "crashes_b"]
    rows = [[p] + [f"{v.get(k, 0):.2f}" for k in keys] for p, v in data["scripted"].items()]
    L.append(md_table(["pairing"] + keys, rows))
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", type=int, default=50)
    ap.add_argument("--turns", type=int, default=120)
    ap.add_argument("--procs", type=int, default=0)
    ap.add_argument("--max-scenarios", type=int, default=0)
    ap.add_argument("--no-enumeration", action="store_true")
    ap.add_argument("--root", type=int, default=46_002)
    a = ap.parse_args(argv)
    cfg = GameConfig.m4()
    timer = Timer()
    scns = find_scenarios(cfg)
    if a.max_scenarios:
        scns = scns[:a.max_scenarios]
    print(f"{len(scns)} scenarios ({timer():.0f} s)", flush=True)
    jobs = [(i, s, a.rollouts, a.turns, a.root, cfg.to_dict()) for i, s in enumerate(scns)]
    results = sorted(pool_map(run_scenario, jobs, a.procs), key=lambda x: x[0])
    print(f"rollouts done ({timer():.0f} s)", flush=True)
    data = {"meta": meta(cfg, rollouts=a.rollouts, turns=a.turns, root=a.root), "n_scenarios": len(scns),
            "rollouts": a.rollouts, "turns": a.turns, "gamma": cfg.gamma,
            "scenarios": scns, "origins": dict(collections.Counter(s["origin"].split(" ")[0] for s in scns))}
    data["summary"] = summarize(scns, results, a.turns, a.rollouts)
    data["gates"] = data["summary"]["gates"]
    data["s3a"] = s3a(cfg)
    if not a.no_enumeration:
        data["enumeration"] = s3a_enumeration(cfg)
        print(f"enumeration done ({timer():.0f} s)", flush=True)
    data["scripted"] = scripted_freeze_stats(cfg)
    data["per_scenario"] = {str(i): {k: {"disc_mean": float(np.mean([v["disc"] for v in rows])),
                                         "first_crash_mean": float(np.mean([v["first_crash"] for v in rows]))}
                                     for k, rows in res.items()} for i, res in results}
    data["elapsed_s"] = timer()
    save("freeze_gate", data, markdown(data))
    g = data["gates"]
    print(f"G1 {'PASS' if g['G1']['pass'] else 'FAIL'} {g['G1']['attackers']}; G2 {'PASS' if g['G2']['pass'] else 'FAIL'} "
          f"+{g['G2']['attackers_freeze_better']} -{g['G2']['attackers_freeze_worse']}; G3 rule "
          f"{g['G3']['rule_violations']}/{g['G3']['rule_checked_turns']} violations ({timer():.0f} s)")
    return data


if __name__ == "__main__":
    main()
