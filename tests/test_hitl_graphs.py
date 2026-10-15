"""Exercises 2 and 3 (in-memory checkpointer): interrupt / resume behaviour."""

import pytest
from langgraph.types import Command

from common.github import GitHubError
from exercises import exercise_2_hitl as ex2
from exercises import exercise_3_escalation as ex3
from tests.conftest import PR_URL, make_analysis


def start(module, thread="t1"):
    app = module.build_graph()
    cfg = {"configurable": {"thread_id": thread}}
    return app, cfg, app.invoke({"pr_url": PR_URL, "thread_id": thread}, cfg)


def test_ex2_approve_posts_once(install):
    rec, _ = install(ex2, [make_analysis(0.65)])
    app, cfg, r = start(ex2)
    p = r["__interrupt__"][0].value
    assert p["kind"] == "approval_request" and p["confidence"] == 0.65
    assert {"confidence_reasoning", "summary", "comments", "diff_preview"} <= p.keys()
    assert rec.posts == []  # nothing posted merely by pausing
    r = app.invoke(Command(resume={"choice": "approve", "feedback": ""}), cfg)
    assert r["final_action"] == "committed" and len(rec.posts) == 1
    assert r["comment_url"].endswith("#issuecomment-99")


def test_ex2_reject_never_posts(install):
    rec, _ = install(ex2, [make_analysis(0.65)])
    app, cfg, _ = start(ex2)
    r = app.invoke(Command(resume={"choice": "reject", "feedback": "no"}), cfg)
    assert r["final_action"] == "rejected" and rec.posts == []


def test_ex2_edit_revises_then_requires_confirmation(install):
    rec, llm = install(ex2, [make_analysis(0.65), make_analysis(0.7, summary="Revised summary")])
    app, cfg, _ = start(ex2)
    r = app.invoke(Command(resume={"choice": "edit", "feedback": "mention the migration"}), cfg)
    p = r["__interrupt__"][0].value
    assert p["kind"] == "final_confirmation" and "Revised summary" in p["comment_preview"]
    assert "mention the migration" in llm.calls[-1][1]["content"]
    assert rec.posts == []
    r = app.invoke(Command(resume={"choice": "approve"}), cfg)
    assert r["final_action"] == "committed" and len(rec.posts) == 1
    assert "Revised summary" in rec.posts[0][1]


def test_ex2_edit_then_reject_posts_nothing(install):
    rec, _ = install(ex2, [make_analysis(0.65), make_analysis(0.7)])
    app, cfg, _ = start(ex2)
    app.invoke(Command(resume={"choice": "edit", "feedback": "x"}), cfg)
    r = app.invoke(Command(resume={"choice": "reject"}), cfg)
    assert r["final_action"] == "rejected" and rec.posts == []


def test_ex2_auto_approve_and_failure_is_not_success(install):
    rec, _ = install(ex2, [make_analysis(0.9)])
    _, _, r = start(ex2)
    assert r["final_action"] == "auto_approved" and len(rec.posts) == 1

    rec, _ = install(ex2, [make_analysis(0.9)])
    rec.fail_with = GitHubError("GitHub comment post failed: HTTP 403")
    _, _, r = start(ex2, "t2")
    assert r["final_action"] == "auto_commit_failed"
    assert "403" in r["post_error"] and r["comment_url"] is None


def test_ex2_invalid_resume_payload_rejected(install):
    install(ex2, [make_analysis(0.65)])
    app, cfg, _ = start(ex2)
    with pytest.raises(ValueError):
        app.invoke(Command(resume={"choice": "yolo"}), cfg)


def test_ex3_escalation_flow(install):
    qs = ["Why MD5 in storage.py?", "Is SYNC_URL https in prod?", "Why plaintext token?"]
    rec, llm = install(ex3, [make_analysis(0.4, qs), make_analysis(0.8, summary="Refined")])
    app, cfg, r = start(ex3)
    p = r["__interrupt__"][0].value
    assert p["kind"] == "escalation" and p["questions"] == qs and p["risk_factors"]

    answers = dict(zip(qs, ["legacy", "yes", "temporary"]))
    r = app.invoke(Command(resume=answers), cfg)
    p = r["__interrupt__"][0].value
    assert p["kind"] == "final_confirmation" and p["source"] == "escalation"
    assert p["confidence"] == 0.8 and "Refined" in p["comment_preview"]
    refine_prompt = llm.calls[-1][1]["content"]
    assert all(a in refine_prompt for a in answers.values()) and "+print('hi')" in refine_prompt
    assert rec.posts == []  # refined review is confirmed before posting

    r = app.invoke(Command(resume={"choice": "approve"}), cfg)
    assert r["final_action"] == "committed_after_escalation" and len(rec.posts) == 1
    assert r["analysis"].confidence == 0.8


def test_ex3_escalation_missing_answers_rejected(install):
    install(ex3, [make_analysis(0.4, ["q1", "q2"])])
    app, cfg, _ = start(ex3)
    with pytest.raises(ValueError):
        app.invoke(Command(resume={"q1": "only one"}), cfg)


def test_ex3_no_llm_questions_falls_back(install):
    install(ex3, [make_analysis(0.4)])
    _, _, r = start(ex3)
    assert len(r["__interrupt__"][0].value["questions"]) >= 2


def test_ex3_escalation_reject_posts_nothing(install):
    rec, _ = install(ex3, [make_analysis(0.4, ["q1", "q2"]), make_analysis(0.7)])
    app, cfg, _ = start(ex3)
    app.invoke(Command(resume={"q1": "a", "q2": "b"}), cfg)
    r = app.invoke(Command(resume={"choice": "reject"}), cfg)
    assert r["final_action"] == "rejected" and rec.posts == []
