"""M6d's overview check: every result number in docs/m6/research_overview.md against the M5 records.

Three tables drive it:

- CLAIMS: one entry per claim, grouped by overview section. Each holds the text the overview prints and a function of
  the M5 records that produces that text with the overview's rounding (2 decimals for leads and intervals, 3 for the
  concession gaps, the freezing distances and the C_cross lower bound).
- VERDICTS: the record classes the overview's wording relies on, with the sentence that relies on each.
- CONSTANTS: numbers from the mathematical reference or the pre-registration, each with its source, a quote checked
  in that source's section and, where a record holds the value, a record check.

The checks:

- forward: each claim's produced text equals the printed text and is found in its section;
- reverse: every signed decimal in the overview, and every unsigned decimal in §5 and §6, lies inside a claim or a
  constant found in its section;
- verdicts: each record class equals the class the overview relies on, and the relying sentence is in its section.

The records are results/m5/analysis_main.json (A), strategic_main.json (S), monitoring_main.json (M) and
battery/summary_battery_{om,sp}.json.gz (BS). Intervals are read as stored, never recomputed (M5 used t = 2.571).
"""

from __future__ import annotations

import bisect
import gzip
import json
import os
import re
import statistics
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

OVERVIEW = "docs/m6/research_overview.md"
OUT = "results/m6/m6d_overview_check.json"
RECORDS = "results/m5"
REFERENCE = "docs/m4/laning_game_mathematical_reference.md"
PREREG = "docs/m5/m5_preregistration.md"

LEAD = "E_20 (lead at 114)"
ARMS = ("om", "sp")
WORDS = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
REPLICATES = {"replicates": True, "does not replicate": False}
CONVERGENCE = {"still improving (non-convergence: a limitation)": "yes", "not shown to be still improving": "not shown"}
FREEZING = "conditional freezing and the reply to freezes"
CONCESSION = "concession under threat"
BEHAVIOURS = {
    "conditional wave management": "wave management that depends on the situation",
    FREEZING: "freezing when it pays, and breaking the opponent's freeze",
    "selective trading": "choosing when to trade",
    "Recall timing": "Recall timing",
    CONCESSION: "giving ground when weaker and threatened",
}


class Records(NamedTuple):
    A: dict
    S: dict
    M: dict
    BS: dict


def load_records(root: str = RECORDS) -> Records:
    def js(name):
        with open(os.path.join(root, name), encoding="utf-8") as f:
            return json.load(f)

    def gz(arm):
        with gzip.open(os.path.join(root, "battery", f"summary_battery_{arm}.json.gz"), "rt", encoding="utf-8") as f:
            return json.load(f)

    return Records(js("analysis_main.json"), js("strategic_main.json"), js("monitoring_main.json"),
                   {arm: gz(arm) for arm in ARMS})


def load_sources() -> Dict[str, str]:
    out = {}
    for path in (REFERENCE, PREREG):
        with open(path, encoding="utf-8") as f:
            out[path] = f.read()
    return out


Read = Callable[[Records], Any]
Show = Callable[[Any], str]


@dataclass(frozen=True)
class Claim:
    id: str
    section: str
    text: str
    value: Read
    show: Show


@dataclass(frozen=True)
class Verdict:
    id: str
    section: str
    expected: str
    value: Callable[[Records], str]
    relies: str


@dataclass(frozen=True)
class Constant:
    id: str
    section: str
    text: str
    source: str
    source_section: str
    quote: str
    record: Optional[Tuple[Read, Show]] = None


def lead(v: float) -> str:
    return f"{v:+.2f}"


def gap(v: float) -> str:
    return f"{v:+.3f}"


def ci(r: dict) -> str:
    return f"{r['mean']:+.2f} [{r['L']:+.2f}, {r['U']:+.2f}]"


def row(*cells: str) -> str:
    return " | ".join(cells)


def listed(items: List[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def every_pair(n: int, of: int) -> str:
    return "in every pair" if n == of else f"in {n} of {of} pairs"


def all_pairs(n: int, of: int) -> str:
    return f"all {WORDS[of]} pairs" if n == of else f"{WORDS[n]} of the {WORDS[of]} pairs"


def turns(stats: List[str]) -> str:
    """The horizon of G50 / G150 readings; a mixed set prints both, which no claim matches."""
    return "/".join(sorted({s.lstrip("GQ") for s in stats}))


def mlu(r: dict) -> dict:
    return {"mean": r["mean"], "L": r["L"], "U": r["U"]}


def family(R: Records, fam: str, arm: str) -> dict:
    return mlu(R.A["panel"][f"fam:{fam}"]["quantities"][LEAD][arm])


def held_out(R: Records, conditional: bool) -> List[float]:
    """S_out's per-opponent om - sp means in slot order: the four scripted variants, or the three conditional styles
    (the `style:` opponents)."""
    slots = sorted(R.A["per_opponent"]["S_out"].items(), key=lambda kv: int(kv[0].split(":")[0]))
    return [v["om - sp"]["mean"] for k, v in slots if ("style_" in k) == conditional]


def replication(R: Records, arm: str, verdict: str) -> dict:
    return R.S["replication"]["replication"][arm][verdict]


def zoning_50(R: Records, arm: str, s: int) -> float:
    """The 50-turn H_zone_c reading, kept only in the battery summary: the best alternative's G50 over the policy."""
    table = R.BS[arm]["analyses"][f"{arm}_s{s}"]["families"]["H_zone_c"]["table"]["policy"]
    return max(v["G"]["50"]["mean"] for v in table.values())


def concession_failures(R: Records, arm: str) -> dict:
    rep = replication(R, arm, CONCESSION)
    pairs = sorted(int(s) for s, ok in rep["replication"]["holds"].items() if not ok)
    detail = R.S["replication"]["detail"]
    return {"pairs": pairs,
            "statistics": [detail[f"{arm}_s{s}"]["battery"]["H_zone_c"]["statistic"] for s in pairs],
            "gaps": [rep["per_seed_gap"][str(s)] for s in pairs],
            "gaps_50": [zoning_50(R, arm, s) for s in pairs]}


def holds_at_50(R: Records, arm: str) -> int:
    """Concession's pairs holding when H_zone_c is read at 50 turns (the weaker-side gap is read at 50 already)."""
    margin = R.S["rule"]["margin"]
    seeds = replication(R, arm, CONCESSION)["replication"]["holds"]
    return sum(max(zoning_50(R, arm, int(s)), R.S["replication"]["detail"][f"{arm}_s{s}"]["weak"]["gap"]) <= margin
               for s in seeds)


def closest_freezes(R: Records) -> List[Tuple[int, float]]:
    """sp's freezing verdict: the three pairs closest to the margin, as (pair, distance below it). The overview names
    the three closest."""
    margin = R.S["rule"]["margin"]
    gaps = replication(R, "sp", FREEZING)["per_seed_gap"]
    return sorted(((int(s), margin - g) for s, g in gaps.items()), key=lambda p: p[1])[:3]


def om_vs_start_competent(R: Records) -> float:
    vs = R.A["versus_start"]
    hist, cross = vs["om - start | C_hist"]["per_seed"], vs["om - start | C_cross"]["per_seed"]
    return statistics.mean((hist[s] + cross[s]) / 2 for s in hist)


def interval_row(label: str, name: str) -> Tuple[Read, Show]:
    return lambda R: mlu(R.A["endpoints"][name]), lambda v: row(label, ci(v))


def family_row(label: str, fam: str) -> Tuple[Read, Show]:
    return (lambda R: {arm: family(R, fam, arm) for arm in ("om", "sp", "om - sp")},
            lambda v: row(label, lead(v["om"]["mean"]), lead(v["sp"]["mean"]), ci(v["om - sp"])))


def below_zero_row(label: str, name: str) -> Tuple[Read, Show]:
    def value(R):
        e = R.A["endpoints"][name]
        return {**mlu(e), "neg": e["neg"], "n_seeds": e["n_seeds"]}

    return value, lambda v: row(label, ci(v), f"{v['neg']} of {v['n_seeds']}")


def behaviour_row(key: str) -> Tuple[Read, Show]:
    def value(R):
        return {arm: {k: replication(R, arm, key)["replication"][k] for k in ("n_holds", "n_seeds", "verdict")}
                for arm in ARMS}

    def show(v):
        reps = [arm for arm in ARMS if REPLICATES[v[arm]["verdict"]]]
        cell = {2: "replicates in both", 0: "replicates in neither"}.get(len(reps)) or f"replicates in {reps[0]} only"
        return row(BEHAVIOURS[key], *(f"{v[arm]['n_holds']} of {v[arm]['n_seeds']}" for arm in ARMS), cell)

    return value, show


CLAIMS: Tuple[Claim, ...] = (
    Claim("P1", "5.1", "+9.26 [+8.95, +9.56]", lambda R: mlu(R.A["endpoints"]["P1"]), ci),
    Claim("P1_pairs", "5.1", "positive in all six pairs",
          lambda R: [R.A["endpoints"]["P1"][k] for k in ("pos", "n_seeds")], lambda v: f"positive in {all_pairs(*v)}"),
    Claim("S_in_row", "5.1", "library styles | +16.87 | +4.72 | +12.15 [+11.77, +12.54]",
          *family_row("library styles", "S_in")),
    Claim("S_out_row", "5.1", "held-out styles | +10.62 | +4.26 | +6.36 [+5.80, +6.92]",
          *family_row("held-out styles", "S_out")),
    Claim("M_row", "5.1", "misspecified | +9.59 | +4.39 | +5.20 [+4.82, +5.59]", *family_row("misspecified", "M")),
    Claim("variants_mean", "5.1", "averaged +11.39", lambda R: statistics.mean(held_out(R, False)),
          lambda v: f"averaged {lead(v)}"),
    Claim("variants_each", "5.1", "+14.33, +14.42 and +13.12, and the fourth +3.70", lambda R: held_out(R, False),
          lambda v: f"{listed([lead(x) for x in v[:3]])}, and the fourth {lead(v[3])}"),
    Claim("conditional_each", "5.1", "the differences were -0.18, -0.67 and -0.20", lambda R: held_out(R, True),
          lambda v: f"the differences were {listed([lead(x) for x in v])}"),
    Claim("P2a_every_pair", "5.2", "did worse than the sp agents, in every pair",
          lambda R: [R.A["endpoints"]["P2a"][k] for k in ("neg", "n_seeds")],
          lambda v: f"did worse than the sp agents, {every_pair(*v)}"),
    Claim("P2a_row", "5.2", "competent opponents (historical and finals) | -0.63 [-0.98, -0.27] | 6 of 6",
          *below_zero_row("competent opponents (historical and finals)", "P2a")),
    Claim("P2b_row", "5.2", "head to head, om against sp of the same pair | -0.44 [-0.72, -0.16] | 6 of 6",
          *below_zero_row("head to head, om against sp of the same pair", "P2b")),
    Claim("S2_row", "5.2", "competent opponents, 500-turn games | -0.60 [-0.85, -0.35] | 6 of 6",
          *below_zero_row("competent opponents, 500-turn games", "S2")),
    Claim("S3_row", "5.2", "head to head, 500-turn games | -0.90 [-1.33, -0.48] | 6 of 6",
          *below_zero_row("head to head, 500-turn games", "S3")),
    Claim("S4_row", "5.2", "head to head, om against the other pairs' sp | -0.68 [-0.96, -0.39] | 6 of 6",
          *below_zero_row("head to head, om against the other pairs' sp", "S4")),
    Claim("C_hist_leads", "5.2", "om agents' lead was +0.70 and the sp agents' +1.35",
          lambda R: [family(R, "C_hist", arm)["mean"] for arm in ARMS],
          lambda v: f"om agents' lead was {lead(v[0])} and the sp agents' {lead(v[1])}"),
    Claim("C_cross_leads", "5.2", "om agents' lead was -0.31 and the sp agents' +0.29",
          lambda R: [family(R, "C_cross", arm)["mean"] for arm in ARMS],
          lambda v: f"om agents' lead was {lead(v[0])} and the sp agents' {lead(v[1])}"),
    Claim("headline", "5.2", 'outcome 2, "pays at a competence cost"',
          lambda R: [R.A["headline"]["outcome"], R.A["headline"]["name"]], lambda v: f'outcome {v[0]}, "{v[1]}"'),
    Claim("attackers_every_pair", "5.3", "Every attacker beat its target, in every pair",
          lambda R: [min(R.A["endpoints"][x]["pos"] for x in ("X_om", "X_sp")), R.A["endpoints"]["X_om"]["n_seeds"]],
          lambda v: f"Every attacker beat its target, {every_pair(*v)}"),
    Claim("X_om_row", "5.3", "om final agents | +2.96 [+0.32, +5.61]", *interval_row("om final agents", "X_om")),
    Claim("X_sp_row", "5.3", "sp final agents | +1.15 [+0.34, +1.96]", *interval_row("sp final agents", "X_sp")),
    Claim("S6", "5.3", "+1.81 [-0.56, +4.19]", lambda R: mlu(R.A["endpoints"]["S6"]), ci),
    Claim("X_om_largest", "5.3", "pair 3, whose attacker led by +7.91",
          lambda R: max(((int(s), v) for s, v in R.A["endpoints"]["X_om"]["per_seed"].items()), key=lambda p: p[1]),
          lambda v: f"pair {v[0]}, whose attacker led by {lead(v[1])}"),
    Claim("wave_row", "5.4", "wave management that depends on the situation | 6 of 6 | 6 of 6 | replicates in both",
          *behaviour_row("conditional wave management")),
    Claim("freezing_row", "5.4",
          "freezing when it pays, and breaking the opponent's freeze | 6 of 6 | 6 of 6 | replicates in both",
          *behaviour_row(FREEZING)),
    Claim("trading_row", "5.4", "choosing when to trade | 6 of 6 | 6 of 6 | replicates in both",
          *behaviour_row("selective trading")),
    Claim("recall_row", "5.4", "Recall timing | 6 of 6 | 6 of 6 | replicates in both", *behaviour_row("Recall timing")),
    Claim("concession_row", "5.4", "giving ground when weaker and threatened | 3 of 6 | 4 of 6 | replicates in neither",
          *behaviour_row(CONCESSION)),
    Claim("om_concession_fails", "5.4", "fail in pairs 0, 3 and 4 on the registered 50-turn measure",
          lambda R: concession_failures(R, "om"),
          lambda v: f"fail in pairs {listed([str(s) for s in v['pairs']])} on the registered "
                    f"{turns(v['statistics'])}-turn measure"),
    Claim("sp_concession_fails", "5.4", "fail only in pairs 4 and 5, and only on a 150-turn re-run",
          lambda R: concession_failures(R, "sp"),
          lambda v: f"fail only in pairs {listed([str(s) for s in v['pairs']])}, and only on a "
                    f"{turns(v['statistics'])}-turn re-run"),
    Claim("sp_concession_150", "5.4", "beat the agent's play by +0.212 and +0.329",
          lambda R: concession_failures(R, "sp")["gaps"], lambda v: f"beat the agent's play by {listed([gap(g) for g in v])}"),
    Claim("sp_concession_50", "5.4", "the margins were only +0.058 and +0.045",
          lambda R: concession_failures(R, "sp")["gaps_50"], lambda v: f"the margins were only {listed([gap(g) for g in v])}"),
    Claim("sp_concession_50_holds", "5.4", "pure self-play holds in all six pairs", lambda R: holds_at_50(R, "sp"),
          lambda n: f"pure self-play holds in {all_pairs(n, 6)}"),
    Claim("sp_freezing_narrow", "5.4", "In sp pairs 0, 1 and 2 the best alternative comes within 0.014, 0.015 and 0.034",
          closest_freezes,
          lambda v: f"In sp pairs {listed([str(s) for s, _ in v])} the best alternative comes within "
                    f"{listed([f'{d:.3f}' for _, d in v])}"),
    Claim("side_P1", "6", "gain against scripted styles, om - sp | +9.26 [+8.95, +9.56]",
          *interval_row("gain against scripted styles, om - sp", "P1")),
    Claim("side_P2a", "6", "against competent opponents, om - sp | -0.63 [-0.98, -0.27]",
          *interval_row("against competent opponents, om - sp", "P2a")),
    Claim("side_P2b", "6", "head to head, om's lead | -0.44 [-0.72, -0.16]", *interval_row("head to head, om's lead", "P2b")),
    Claim("side_attackers", "6", "attacker's lead against it | +2.96 [+0.32, +5.61] | +1.15 [+0.34, +1.96]",
          lambda R: [mlu(R.A["endpoints"][x]) for x in ("X_om", "X_sp")],
          lambda v: row("attacker's lead against it", ci(v[0]), ci(v[1]))),
    Claim("side_replicate", "6", "strategic behaviours that replicate | four of five | four of five",
          lambda R: {arm: [REPLICATES[replication(R, arm, key)["replication"]["verdict"]] for key in BEHAVIOURS]
                     for arm in ARMS},
          lambda v: row("strategic behaviours that replicate",
                        *(f"{WORDS[sum(v[arm])]} of {WORDS[len(v[arm])]}" for arm in ARMS))),
    Claim("side_convergence", "6", "still improving when training stopped | not shown | yes",
          lambda R: [R.M["convergence"][arm]["verdict"] for arm in ARMS],
          lambda v: row("still improving when training stopped", CONVERGENCE[v[0]], CONVERGENCE[v[1]])),
    Claim("convergence_sp", "6", "75% checkpoints by +0.59 [+0.14, +1.05]", lambda R: mlu(R.M["convergence"]["sp"]),
          lambda v: f"75% checkpoints by {ci(v)}"),
    Claim("convergence_om", "6", "om agents' difference was -0.08 [-0.40, +0.24]",
          lambda R: mlu(R.M["convergence"]["om"]), lambda v: f"om agents' difference was {ci(v)}"),
    Claim("quote_held_out", "9", "+6.4 against held-out scripted variants",
          lambda R: family(R, "S_out", "om - sp")["mean"], lambda v: f"{v:+.1f} against held-out scripted variants"),
    Claim("variants_mean_9", "9", "average +11.39", lambda R: statistics.mean(held_out(R, False)),
          lambda v: f"average {lead(v)}"),
    Claim("sp_concession_150_pairs", "9", "150-turn re-run of two pairs", lambda R: concession_failures(R, "sp"),
          lambda v: f"{turns(v['statistics'])}-turn re-run of {WORDS[len(v['pairs'])]} pairs"),
    Claim("sp_freezing_farthest", "9", "one of them is 0.034 away", lambda R: max(d for _, d in closest_freezes(R)),
          lambda v: f"one of them is {v:.3f} away"),
    # A quote of the closeout, which rounded this value to the nearest 0.05 ("about"); the next claim states it exactly.
    Claim("quote_om_vs_start", "9", "om about +0.25 against the start", om_vs_start_competent,
          lambda v: f"om about {round(v * 20) / 20:+.2f} against the start"),
    Claim("om_vs_start", "9", "The value is +0.24", om_vs_start_competent, lambda v: f"The value is {lead(v)}"),
    Claim("om_vs_start_C_cross", "9", "[+0.004, +0.55]", lambda R: mlu(R.A["versus_start"]["om - start | C_cross"]),
          lambda v: f"[{v['L']:+.3f}, {v['U']:+.2f}]"),
)


def endpoint_class(name: str) -> Callable[[Records], str]:
    return lambda R: R.A["endpoints"][name]["class"]


def replication_verdict(arm: str, key: str) -> Callable[[Records], str]:
    return lambda R: replication(R, arm, key)["replication"]["verdict"]


MATERIAL_LOSS = "Each of these meets the registered rule for a material loss"

VERDICTS: Tuple[Verdict, ...] = (
    Verdict("P1", "5.1", "demonstrated", endpoint_class("P1"), "Against scripted styles, the om agents built much larger leads"),
    Verdict("S1", "5.1", "preserved", endpoint_class("S1"), "the registered test that the gain was preserved passed"),
    Verdict("P2a", "5.2", "material loss", endpoint_class("P2a"), MATERIAL_LOSS),
    Verdict("P2b", "5.2", "material loss", endpoint_class("P2b"), MATERIAL_LOSS),
    Verdict("S2", "5.2", "material loss", endpoint_class("S2"), MATERIAL_LOSS),
    Verdict("S3", "5.2", "material loss", endpoint_class("S3"), MATERIAL_LOSS),
    Verdict("S4", "5.2", "material loss", endpoint_class("S4"), MATERIAL_LOSS),
    Verdict("S6", "5.3", "inconclusive", endpoint_class("S6"), "That is inconclusive"),
    *(Verdict(f"{arm} {key}", "5.4", "does not replicate" if key == CONCESSION else "replicates",
              replication_verdict(arm, key), BEHAVIOURS[key])
      for key in BEHAVIOURS for arm in ARMS),
)


def description_number(name: str, rx: str) -> Read:
    return lambda R: int(re.search(rx, R.A["endpoints"][name]["description"]).group(1))


def quantity_number(rx: str) -> Read:
    """The records hold these turn counts only in the panel's quantity labels, such as "D0 (earned lead at 94)"."""
    return lambda R: next(int(m.group(1)) for k in R.A["panel"]["fam:S_in"]["quantities"] if (m := re.fullmatch(rx, k)))


CONSTANTS: Tuple[Constant, ...] = (
    Constant("lane_exit_turn", "2", "around turn 94", REFERENCE, "5.4", "A side reaches level 14 at a median turn of 94",
             (quantity_number(r"D0 \(earned lead at (\d+)\)"), lambda n: f"around turn {n}")),
    Constant("lead_turn", "2", "after 114 turns", PREREG, "5", "earned-gold lead at turn 114",
             (description_number("P1", r"lead at (\d+)"), lambda n: f"after {n} turns")),
    Constant("readout_horizon", "2", "the next 20 turns", REFERENCE, "9.3", "$h=20$ is about one reset cycle",
             (quantity_number(r"E_(\d+) \(lead at \d+\)"), lambda n: f"the next {n} turns")),
    Constant("income", "2", "about 0.39 a turn", REFERENCE, "5.1", "a side earns 0.39 resources per turn"),
    Constant("lane_exit_gold", "2", "about 37 by the end of the laning phase", REFERENCE, "7.4",
             "| earned (resources) | 19.5 | 37.4 |"),
    Constant("death_cost", "2", "about 3.5 to 4", REFERENCE, "6.7", "a death costs about 3.5–4 resources"),
    Constant("budget", "3", "32,768,000 agent decisions", PREREG, "2", "BUDGET = 32,768,000 additional agent transitions",
             (lambda R: max(int(k.split("|")[1]) for k in R.M["curve_summary"]), lambda n: f"{n:,} agent decisions")),
    Constant("attacker_budget", "4", "8,192,000 decisions", PREREG, "8", "budget 8,192,000 transitions per attacker"),
    Constant("family_pairs", "4", "64 pairs of games", PREREG, "6", "`g_shove_ward` | 64 × 150"),
    Constant("h2h_pairs", "4", "128 such pairs", PREREG, "6", "| H2H | om:s vs sp:s | 128 × 150",
             (description_number("P2b", r"\((\d+) pairs\)"), lambda n: f"{n} such pairs")),
    Constant("gain_rule", "4", "at least +0.25", PREREG, "9", "Demonstrated when mean ≥ +0.25 and L > 0"),
    Constant("loss_rule", "4", "the mean is -0.25 or worse", PREREG, "9", "non-inferiority with margin δ = 0.25"),
    Constant("margin", "4", "more than 0.2 gold", PREREG, "7", "beats the policy by more than 0.2",
             (lambda R: R.S["rule"]["margin"], lambda m: f"more than {m:g} gold")),
    Constant("min_pairs", "4", "at least five of the six pairs", PREREG, "7", "It replicates with ≥ 5 of 6",
             (lambda R: R.S["rule"]["min_seeds"], lambda n: f"at least {WORDS[n]} of the six pairs")),
    Constant("long_games", "5.2", "500-turn games", PREREG, "6", "32 × 500",
             (description_number("S2", r"(\d+) turns"), lambda n: f"{n}-turn games")),
    Constant("threshold", "5.4", "the 0.2 threshold", PREREG, "7", "beats the policy by more than 0.2",
             (lambda R: R.S["rule"]["margin"], lambda m: f"the {m:g} threshold")),
)


HEADING = re.compile(r"^#{2,4} (\d+(?:\.\d+)*)\.? ", re.M)
SIGNED = re.compile(r"(?<![\w.§])[+-]\d+\.\d+(?!\d)")
UNSIGNED = re.compile(r"(?<![\w.§+-])\d+\.\d+(?!\d)")


def normalize(text: str) -> str:
    return text.replace("−", "-")


def pattern(text: str) -> re.Pattern:
    """The text as printed, across line breaks and indentation, not as part of a longer word or number."""
    body = r"\s+".join(re.escape(w) for w in normalize(text).split())
    return re.compile(r"(?<![\w.])" + body + r"(?!\w|\.\d)")


class Doc:
    """A markdown document split into its numbered sections (§5.1 runs from its heading to the next heading)."""

    def __init__(self, text: str):
        self.text = normalize(text)
        heads = [(m.group(1), m.start()) for m in HEADING.finditer(self.text)]
        ends = [s for _, s in heads[1:]] + [len(self.text)]
        self.sections = {sid: (start, end) for (sid, start), end in zip(heads, ends)}
        self._newlines = [i for i, c in enumerate(self.text) if c == "\n"]

    def line(self, pos: int) -> int:
        return bisect.bisect_left(self._newlines, pos) + 1

    def section_of(self, pos: int) -> str:
        return next((sid for sid, (a, b) in self.sections.items() if a <= pos < b), "")

    def find(self, section: str, text: str) -> List[Tuple[int, int]]:
        if section not in self.sections:
            return []
        a, b = self.sections[section]
        return [m.span() for m in pattern(text).finditer(self.text, a, b)]

    def tokens(self) -> List[Tuple[str, int, int]]:
        """Every signed decimal, and every unsigned decimal in §5 and §6, outside headings."""
        out = []
        for rx, scope in ((SIGNED, None), (UNSIGNED, ("5", "6"))):
            for m in rx.finditer(self.text):
                if self.text.startswith("#", self.text.rfind("\n", 0, m.start()) + 1):
                    continue
                if scope and self.section_of(m.start()).split(".")[0] not in scope:
                    continue
                out.append((m.group(), m.start(), m.end()))
        return sorted(out, key=lambda t: t[1])


def _jsonable(v: Any) -> Any:
    return json.loads(json.dumps(v, default=str))


def _evaluate(R: Records, value: Read, show: Show) -> Tuple[Any, Optional[str], Optional[str]]:
    try:
        v = value(R)
        return v, show(v), None
    except Exception as e:  # a record that lost a key or gained an unknown class fails its own check, not the run
        return None, None, f"{type(e).__name__}: {e}"


def check(overview_text: str, R: Records, sources: Dict[str, str]) -> Dict[str, Any]:
    doc = Doc(overview_text)
    source_docs = {path: Doc(text) for path, text in sources.items()}
    covered: Dict[str, List[Tuple[int, int, str]]] = {}

    def locate(kind: str, item) -> List[int]:
        found = doc.find(item.section, item.text)
        covered.setdefault(item.section, []).extend((a, b, f"{kind} {item.id}") for a, b in found)
        return [doc.line(a) for a, _ in found]

    claims = []
    for c in CLAIMS:
        v, produced, error = _evaluate(R, c.value, c.show)
        lines = locate("claim", c)
        claims.append({"id": c.id, "section": c.section, "expected": normalize(c.text), "produced": produced,
                       "found": bool(lines), "lines": lines, "value": _jsonable(v), "error": error,
                       "pass": produced == normalize(c.text) and bool(lines)})

    constants = []
    for k in CONSTANTS:
        lines = locate("constant", k)
        quoted = bool(source_docs[k.source].find(k.source_section, k.quote))
        entry = {"id": k.id, "section": k.section, "expected": normalize(k.text), "found": bool(lines), "lines": lines,
                 "source": f"{k.source} §{k.source_section}", "quote": k.quote, "quote_found": quoted}
        ok = bool(lines) and quoted
        if k.record:
            v, produced, error = _evaluate(R, *k.record)
            entry.update(produced=produced, value=_jsonable(v), error=error)
            ok = ok and produced == normalize(k.text)
        constants.append({**entry, "pass": ok})

    verdicts = []
    for d in VERDICTS:
        v, _, error = _evaluate(R, d.value, str)
        lines = [doc.line(a) for a, _ in doc.find(d.section, d.relies)]
        verdicts.append({"id": d.id, "section": d.section, "expected": d.expected, "value": v, "relies": d.relies,
                         "found": bool(lines), "lines": lines, "error": error,
                         "pass": v == d.expected and bool(lines)})

    tokens = []
    for tok, a, b in doc.tokens():
        sec = doc.section_of(a)
        by = [who for s, e, who in covered.get(sec, []) if s <= a and b <= e]
        tokens.append({"token": tok, "line": doc.line(a), "section": sec, "accounted_by": by})
    unaccounted = [t for t in tokens if not t["accounted_by"]]

    ok = all(x["pass"] for x in claims + constants + verdicts) and not unaccounted
    return {"ok": ok,
            "summary": {"claims": len(claims), "claims_failed": sum(not c["pass"] for c in claims),
                        "constants": len(constants), "constants_failed": sum(not c["pass"] for c in constants),
                        "verdicts": len(verdicts), "verdicts_failed": sum(not v["pass"] for v in verdicts),
                        "tokens": len(tokens), "unaccounted": len(unaccounted)},
            "claims": claims, "constants": constants, "verdicts": verdicts,
            "reverse": {"tokens": tokens, "unaccounted": unaccounted}}


def failures(report: Dict[str, Any]) -> List[str]:
    out = []
    for kind in ("claims", "constants", "verdicts"):
        for x in report[kind]:
            if not x["pass"]:
                got = x.get("produced", x.get("value"))
                out.append(f"{kind[:-1]} {x['id']} (§{x['section']}): expected {x['expected']!r}, got {got!r}, "
                           f"found in the overview {x['found']}"
                           + (f", quote found in {x['source']} {x['quote_found']}" if "quote_found" in x else "")
                           + (f", error {x['error']}" if x.get("error") else ""))
    out += [f"unaccounted {t['token']} at line {t['line']} (§{t['section']})" for t in report["reverse"]["unaccounted"]]
    return out


def summary_line(report: Dict[str, Any]) -> str:
    s = report["summary"]
    return (f"overview check {'PASS' if report['ok'] else 'FAIL'}: "
            f"{s['claims'] - s['claims_failed']}/{s['claims']} claims, "
            f"{s['constants'] - s['constants_failed']}/{s['constants']} constants, "
            f"{s['verdicts'] - s['verdicts_failed']}/{s['verdicts']} verdicts, "
            f"{s['tokens'] - s['unaccounted']}/{s['tokens']} decimal tokens accounted for")


def run(overview: str = OVERVIEW, out: str = OUT) -> bool:
    with open(overview, encoding="utf-8") as f:
        report = {"overview": overview, **check(f.read(), load_records(), load_sources())}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    for line in failures(report):
        print(line)
    print(summary_line(report), flush=True)
    return report["ok"]
