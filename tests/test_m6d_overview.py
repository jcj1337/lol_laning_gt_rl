"""M6d's overview check (lanegame/experiments/m6d/overview_check.py): every result number in
docs/m6/research_overview.md against the M5 records, and proof that the check reports a changed number, a number no
check accounts for, a changed record value, a changed class and a citation its source no longer supports."""

import copy
import json

import pytest

from lanegame.experiments.m6d import overview_check as O


@pytest.fixture(scope="module")
def inputs():
    with open(O.OVERVIEW, encoding="utf-8") as f:
        text = f.read()
    return text, O.load_records(), O.load_sources()


def failed(report):
    return {(kind, x["id"]) for kind in ("claims", "constants", "verdicts") for x in report[kind] if not x["pass"]}


def test_every_overview_number_matches_the_records(inputs):
    report = O.check(*inputs)
    assert report["ok"], "\n".join(O.failures(report))
    assert report["reverse"]["tokens"], "the overview's decimals were scanned"


def test_a_changed_number_and_an_unaccounted_number_are_reported(inputs, tmp_path):
    text, R, sources = inputs
    changed = text.replace("+12.15 [+11.77, +12.54]", "+12.15 [+11.78, +12.54]")
    changed = changed.replace("That is inconclusive.", "That is inconclusive, with a median of +3.14.")
    assert changed.count("+11.78") == 1 and changed.count("+3.14") == 1

    report = O.check(changed, R, sources)

    def line(token):
        return changed[:changed.index(token)].count("\n") + 1

    assert not report["ok"]
    assert failed(report) == {("claims", "S_in_row")}
    unaccounted = {(t["token"], t["line"]) for t in report["reverse"]["unaccounted"]}
    assert ("+11.78", line("+11.78")) in unaccounted
    assert ("+3.14", line("+3.14")) in unaccounted

    overview, out = tmp_path / "overview.md", tmp_path / "check.json"
    overview.write_text(changed, encoding="utf-8")
    assert O.run(overview=str(overview), out=str(out)) is False
    assert json.loads(out.read_text(encoding="utf-8"))["ok"] is False


def test_a_changed_record_class_or_source_is_reported(inputs):
    text, R, sources = inputs
    A = copy.deepcopy(R.A)
    A["endpoints"]["P1"]["U"] = 9.57
    A["endpoints"]["S6"]["class"] = "no material difference"
    edited = {**sources, O.REFERENCE: sources[O.REFERENCE].replace("earns 0.39 resources per turn",
                                                                   "earns 0.38 resources per turn")}

    report = O.check(text, R._replace(A=A), edited)

    assert failed(report) == {("claims", "P1"), ("claims", "side_P1"), ("verdicts", "S6"), ("constants", "income")}
    assert not report["reverse"]["unaccounted"]
