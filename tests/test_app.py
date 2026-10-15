"""Streamlit app driven headlessly with streamlit.testing (no browser, mocked I/O)."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from exercises import exercise_4_audit as ex4
from tests.conftest import PR_URL, make_analysis

APP = Path(__file__).resolve().parent.parent / "app.py"


def new_app() -> AppTest:
    return AppTest.from_file(str(APP), default_timeout=30)


def field(at: AppTest, label_part: str):
    return next(t for t in at.text_input if label_part in t.label)


def click(at: AppTest, label_part: str) -> None:
    btn = next(b for b in at.button if label_part in b.label)
    btn.click().run()


def test_renders_clean_and_rejects_bad_url():
    at = new_app().run()
    assert not at.exception
    field(at, "PR URL").set_value("not a pr")
    click(at, "Run review")
    assert any("Not a GitHub PR URL" in e.value for e in at.error)


def test_approval_flow_resumes_same_thread(install):
    rec, _ = install(ex4, [make_analysis(0.65)])
    at = new_app().run()
    field(at, "PR URL").set_value(PR_URL)
    click(at, "Run review")
    assert not at.exception
    tid = at.session_state["thread_id"]
    assert tid and any("Approval requested" in s.value for s in at.subheader)
    assert rec.posts == []  # rendering the card posted nothing

    field(at, "Feedback").set_value("looks good")
    click(at, "Approve & post")
    assert not at.exception
    assert at.session_state["thread_id"] == tid
    assert len(rec.posts) == 1
    assert any("Posted" in s.value for s in at.success)
    assert any("issuecomment-99" in m.value for m in at.markdown)


def test_escalation_flow_requires_all_answers_and_confirmation(install):
    qs = ["Why MD5 in storage.py?", "Is SYNC_URL https?"]
    rec, _ = install(ex4, [make_analysis(0.4, qs), make_analysis(0.8, summary="Refined")])
    at = new_app().run()
    field(at, "PR URL").set_value(PR_URL)
    click(at, "Run review")
    assert any("Strong escalation" in s.value for s in at.subheader)
    assert at.error  # risk factors highlighted

    field(at, "MD5").set_value("only first")
    click(at, "Submit answers")
    assert any("answer every question" in e.value for e in at.error)
    assert rec.posts == []

    field(at, "MD5").set_value("a1")
    field(at, "SYNC_URL").set_value("a2")
    click(at, "Submit answers")
    assert any("Review refined" in s.value for s in at.subheader)
    assert rec.posts == []  # refined review shown, not yet posted
    click(at, "Confirm & post")
    assert len(rec.posts) == 1 and any("Posted" in s.value for s in at.success)


def test_failed_post_shows_error_not_success(install):
    from common.github import GitHubError
    rec, _ = install(ex4, [make_analysis(0.9)])
    rec.fail_with = GitHubError("GitHub comment post failed: HTTP 403")
    at = new_app().run()
    field(at, "PR URL").set_value(PR_URL)
    click(at, "Run review")
    assert not any("Posted" in s.value for s in at.success)
    assert any("failed" in e.value for e in at.error)


def test_reload_restores_pending_interrupt_without_new_review(install):
    rec, llm = install(ex4, [make_analysis(0.65)])
    at = new_app().run()
    field(at, "PR URL").set_value(PR_URL)
    click(at, "Run review")
    tid = at.session_state["thread_id"]

    at2 = new_app()
    at2.query_params["thread"] = tid
    at2.run()
    assert at2.session_state["thread_id"] == tid
    assert any("Approval requested" in s.value for s in at2.subheader)
    assert len(llm.calls) == 1  # no second analysis was started
    assert any(tid[:8] in b.label for b in at2.sidebar.button)  # shown in recent sessions
