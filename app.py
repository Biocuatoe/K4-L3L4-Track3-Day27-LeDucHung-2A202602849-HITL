"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py

The UI drives the Exercise 4 graph (audit trail + durable SQLite checkpointer)
through `review_service`. The experience adapts to the confidence bucket
(thresholds in common/schemas.py):

    >= 73%       auto_approve     success card; reviewer does nothing
    58 – 72%     human_approval   Approve / Reject / Edit buttons
    <  58%       escalate         question form → refined review → final confirm

Nothing is posted to GitHub by rendering a card; only an explicit submit sends a
resume payload to the graph, always on the same thread_id.
"""

from __future__ import annotations

import asyncio
import os

import streamlit as st
from dotenv import load_dotenv

# Must be the first Streamlit call in the script.
st.set_page_config(page_title="HITL PR Review", layout="wide")

from common.db import list_sessions  # noqa: E402
from common.github import parse_pr_url  # noqa: E402
from common.llm import redact  # noqa: E402
from common.review import validate_answers  # noqa: E402
from review_service import (  # noqa: E402
    SessionView,
    StaleResumeError,
    continue_session,
    load_session,
    new_thread_id,
    resume_review,
    start_review,
)

load_dotenv()

RISK_ICON = {"low": "🟢", "med": "🟡", "high": "🔴"}
SEVERITY_ICON = {"nit": "💬", "suggestion": "💡", "issue": "⚠️", "blocker": "⛔"}


def run(coro):
    return asyncio.run(coro)


# ─── Session state ─────────────────────────────────────────────────────────
for key, default in (("thread_id", None), ("view", None), ("pr_url", "")):
    st.session_state.setdefault(key, default)


def set_view(view: SessionView) -> None:
    st.session_state.thread_id = view.thread_id
    st.session_state.view = view
    st.session_state.pr_url = view.values.get("pr_url", st.session_state.pr_url)
    st.query_params["thread"] = view.thread_id  # survives a browser reload


def open_session(thread_id: str) -> None:
    """Restore a persisted session (and any pending interrupt) — never starts a new review."""
    try:
        view = run(load_session(thread_id))
    except Exception as e:
        st.error(f"Could not load session: {redact(e)}")
        return
    if view.status == "empty":
        st.warning(f"No saved graph state for thread {thread_id}.")
        return
    set_view(view)


# Restore from ?thread=<id> after a page reload.
if st.session_state.view is None and st.query_params.get("thread"):
    open_session(st.query_params["thread"])


def call_service(coro_fn, *args, **kwargs) -> bool:
    """Run a service call, store the new view, surface errors without crashing the app."""
    try:
        set_view(run(coro_fn(*args, **kwargs)))
        return True
    except StaleResumeError as e:
        st.warning(f"{e}. Reloading the session's current state.")
        if st.session_state.thread_id:
            open_session(st.session_state.thread_id)
    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"{type(e).__name__}: {redact(e)}")
    return False


# ─── Page ──────────────────────────────────────────────────────────────────
st.title("HITL PR Review Agent")

with st.sidebar:
    st.header("Reviewer")
    reviewer = st.text_input(
        "GitHub username", value=os.environ.get("GITHUB_USER", ""),
        help="Recorded as reviewer_id on human audit events (applies to new reviews).",
    )
    st.header("Recent sessions")
    try:
        sessions = run(list_sessions(15))
    except Exception as e:
        sessions = []
        st.error(f"Could not read audit trail: {redact(e)}")
    if not sessions:
        st.caption("No sessions yet.")
    for s in sessions:
        pr = s["pr_url"].replace("https://github.com/", "")
        label = (f"{RISK_ICON.get(s['worst_risk'], '⚪')} {pr}\n"
                 f"{s['last_action']}:{s['last_decision']} · {s['events']} events · "
                 f"{s['thread_id'][:8]}")
        if st.button(label, key=f"open_{s['thread_id']}", use_container_width=True):
            open_session(s["thread_id"])
            st.rerun()

with st.form("start"):
    pr_url = st.text_input(
        "PR URL", value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    submitted = st.form_submit_button("Run review")

if submitted:
    try:
        parse_pr_url(pr_url)
    except ValueError as e:
        st.error(str(e))
    else:
        st.session_state.view = None
        st.session_state.pr_url = pr_url.strip()
        # New thread per review; every later resume reuses this id.
        thread_id = new_thread_id()
        st.session_state.thread_id = thread_id
        with st.spinner("Fetching PR + asking the LLM..."):
            ok = call_service(start_review, pr_url, thread_id, reviewer.strip() or None)
        if ok:
            st.rerun()


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def show_comments(comments: list[dict]) -> None:
    for c in comments:
        icon = SEVERITY_ICON.get(c["severity"], "")
        st.markdown(f"- {icon} **[{c['severity']}]** `{c['file']}:{c.get('line') or '?'}` — {c['body']}")


def render_approval_card(payload: dict) -> dict | None:
    """58–72% bucket: show the LLM review + 3 buttons. Returns a resume dict or None."""
    st.subheader(f"Approval requested — confidence {payload['confidence']:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])
    show_comments(payload.get("comments", []))
    with st.expander("Diff preview"):
        st.code(payload.get("diff_preview", ""), language="diff")

    with st.form("approval"):
        feedback = st.text_input("Feedback (required for Edit, optional otherwise)")
        col1, col2, col3 = st.columns(3)
        clicked = {
            "approve": col1.form_submit_button("Approve & post", type="primary"),
            "reject": col2.form_submit_button("Reject"),
            "edit": col3.form_submit_button("Edit (revise with feedback)"),
        }
    for choice, hit in clicked.items():
        if hit:
            return {"choice": choice, "feedback": feedback.strip()}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    """< 58% bucket: risk factors + question form. Returns {question: answer} or None."""
    st.subheader(f"Strong escalation — confidence {payload['confidence']:.0%}")
    st.caption(payload["confidence_reasoning"])
    if payload.get("risk_factors"):
        st.error("**Risks**\n\n" + "\n".join(f"- {r}" for r in payload["risk_factors"]))
    st.markdown(payload["summary"])

    with st.form("escalation"):
        answers = {
            q: st.text_input(q, key=f"q_{i}_{st.session_state.thread_id}")
            for i, q in enumerate(payload["questions"])
        }
        sent = st.form_submit_button("Submit answers", type="primary")
    if sent:
        try:
            return validate_answers(payload["questions"], answers)
        except ValueError as e:
            st.error(f"Please answer every question. {e}")
    return None


def render_confirmation_card(payload: dict) -> dict | None:
    """Refined / revised review: nothing is posted until the reviewer confirms."""
    origin = "after your answers" if payload.get("source") == "escalation" else "after your feedback"
    st.subheader(f"Review refined {origin} — confidence {payload['confidence']:.0%}")
    st.caption(payload["confidence_reasoning"])
    if payload.get("risk_factors"):
        st.warning("**Risks**\n\n" + "\n".join(f"- {r}" for r in payload["risk_factors"]))
    st.markdown("**Comment that will be posted:**")
    st.markdown(payload["comment_preview"])
    with st.form("confirm"):
        col1, col2 = st.columns(2)
        post = col1.form_submit_button("Confirm & post", type="primary")
        skip = col2.form_submit_button("Don't post")
    if post:
        return {"choice": "approve"}
    if skip:
        return {"choice": "reject"}
    return None


CARDS = {
    "approval_request": render_approval_card,
    "escalation": render_escalation_card,
    "final_confirmation": render_confirmation_card,
}


def render_final(view: SessionView) -> None:
    v = view.values
    action = v.get("final_action")
    a = v.get("analysis")
    outcome = view.outcome
    if outcome == "posted":
        kind = "Auto-approved and posted" if action == "auto_approved" else "Posted"
        st.success(f"✓ {kind} ({action})")
        if v.get("comment_url"):
            st.markdown(f"[View comment on GitHub]({v['comment_url']})")
    elif outcome == "rejected":
        st.warning("Rejected — no comment was posted.")
    elif outcome == "failed":
        st.error(f"✗ Posting to GitHub failed ({action}) — nothing was posted.")
        st.code(v.get("post_error") or "unknown error")
    else:
        st.info(f"Finished with final_action = {action}")
    if a is not None:
        st.caption(f"Confidence {a.confidence:.0%} — {a.confidence_reasoning}")
        st.markdown(a.summary)


# ─── Current session ───────────────────────────────────────────────────────
view: SessionView | None = st.session_state.view
if view is not None:
    st.divider()
    st.caption(
        f"PR: {view.values.get('pr_url', st.session_state.pr_url)} · thread_id = `{view.thread_id}` · "
        f"replay: `uv run python -m audit.replay --thread {view.thread_id}`"
    )
    if view.status == "interrupted" and view.payload:
        kind = view.payload.get("kind")
        card = CARDS.get(kind)
        if card is None:
            st.error(f"Unknown pending request: {kind!r}")
        else:
            answer = card(view.payload)
            if answer is not None:
                with st.spinner("Resuming..."):
                    ok = call_service(resume_review, view.thread_id, answer, kind)
                if ok:
                    st.rerun()
    elif view.status == "completed":
        render_final(view)
    elif view.status == "stalled":
        st.warning("This session stopped before finishing (e.g. a transient LLM or network error).")
        if st.button("Continue from last checkpoint"):
            with st.spinner("Continuing..."):
                ok = call_service(continue_session, view.thread_id)
            if ok:
                st.rerun()
    else:
        st.info("No saved state for this session.")
