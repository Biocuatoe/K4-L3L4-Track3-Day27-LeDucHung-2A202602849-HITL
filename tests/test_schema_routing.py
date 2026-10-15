import pytest
from pydantic import ValidationError

from common.review import (
    classify_outcome,
    decide,
    ensure_analysis,
    questions_for,
    render_comment_body,
    validate_answers,
    validate_choice,
)
from common.schemas import PRAnalysis, risk_level_for
from exercises import exercise_1_confidence as ex1
from tests.conftest import make_analysis


def test_analysis_valid_and_invalid():
    ok = PRAnalysis(summary="s", confidence=0.5, confidence_reasoning="r")
    assert ok.comments == [] and ok.escalation_questions == []
    for bad in (-0.1, 1.01):
        with pytest.raises(ValidationError):
            PRAnalysis(summary="s", confidence=bad, confidence_reasoning="r")
    with pytest.raises(ValidationError):
        PRAnalysis(summary="s", confidence=0.5)  # missing reasoning


@pytest.mark.parametrize("c,expected", [
    (0.0, "escalate"), (0.5799, "escalate"), (0.58, "human_approval"),
    (0.65, "human_approval"), (0.7299, "human_approval"), (0.73, "auto_approve"), (1.0, "auto_approve"),
])
def test_decide_boundaries(c, expected):
    assert decide(c) == expected


@pytest.mark.parametrize("c,risk", [(0.57, "high"), (0.58, "med"), (0.72, "med"), (0.73, "low"), (0.0, "high")])
def test_risk_level_boundaries(c, risk):
    assert risk_level_for(c) == risk


def test_ensure_analysis():
    a = make_analysis(0.6)
    assert ensure_analysis(a) is a
    assert ensure_analysis(a.model_dump()).confidence == 0.6
    from common.review import LLMOutputError
    with pytest.raises(LLMOutputError):
        ensure_analysis(None)
    with pytest.raises(LLMOutputError):
        ensure_analysis({"summary": "x"})


def test_questions_for_bounds():
    assert len(questions_for(make_analysis(0.4))) == 2  # fallbacks fill to 2
    many = make_analysis(0.4, [f"q{i}" for i in range(9)])
    assert questions_for(many) == ["q0", "q1", "q2", "q3"]
    assert questions_for(make_analysis(0.4, ["a", "a", " "]))[0] == "a"


def test_validate_choice_and_answers():
    assert validate_choice({"choice": " Approve "})["choice"] == "approve"
    with pytest.raises(ValueError):
        validate_choice({"choice": "maybe"})
    with pytest.raises(ValueError):
        validate_choice({"choice": "edit", "feedback": "  "})
    with pytest.raises(ValueError):
        validate_choice({"choice": "edit"}, ("approve", "reject"))
    with pytest.raises(ValueError):
        validate_choice("approve")
    assert validate_answers(["q1", "q2"], {"q1": " a ", "q2": "b"}) == {"q1": "a", "q2": "b"}
    for bad in ({"q1": "a"}, {"q1": "a", "q2": " "}, {"q1": "a", "q2": "b", "zz": "c"}, ["a"]):
        with pytest.raises(ValueError):
            validate_answers(["q1", "q2"], bad)


def test_classify_outcome_is_exact_not_prefix():
    assert classify_outcome("auto_approved") == "posted"
    assert classify_outcome("committed") == "posted"
    assert classify_outcome("auto_commit_failed") == "failed"  # starts with "auto" but is a failure
    assert classify_outcome("commit_failed") == "failed"
    assert classify_outcome("rejected") == "rejected"
    assert classify_outcome("pending_escalation") == "unknown"
    assert classify_outcome(None) == "unknown"


def test_comment_body_includes_reviewer_note_only_when_approved():
    st = {"analysis": make_analysis(0.7), "human_feedback": "ship it", "human_choice": "approve"}
    assert "Reviewer note: ship it" in render_comment_body(st)
    st["human_choice"] = "reject"
    assert "Reviewer note" not in render_comment_body(st)


@pytest.mark.parametrize("conf,branch", [(0.9, "auto_approve"), (0.65, "human_approval"), (0.3, "escalate")])
def test_exercise1_graph_routes_all_branches(install, conf, branch):
    install(ex1, [make_analysis(conf)])
    final = ex1.build_graph().invoke({"pr_url": "https://github.com/acme/demo/pull/7"})
    assert final["decision"] == branch
    assert final["final_action"] == {
        "auto_approve": "auto_approved", "human_approval": "pending_human_approval",
        "escalate": "pending_escalation"}[branch]
