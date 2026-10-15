"""Exercise 4 — Structured SQLite audit trail + durable checkpointer.

Zero setup — SQLite stores everything in a single file (`./hitl_audit.db`).
The audit_events schema is created (and migrated) automatically on first connection.

What this module does:

1. Uses AsyncSqliteSaver so the graph can resume after a crash / restart.
2. Emits an `AuditEntry` (common/schemas.py) for every meaningful step, so the
   full session can be replayed:  uv run python -m audit.replay --thread <id>
3. Is the graph that the Streamlit app (app.py) drives.

Graph:

    fetch_pr → analyze → route ─┬ auto_approve ───────────────────────────→ END
                                ├ human_approval ─┬ commit → END
                                │                 └ revise → confirm → commit
                                └ escalate → synthesize → confirm → commit

Resume correctness: LangGraph re-executes an interrupted node from its first
line when it resumes. Every audit row therefore carries a deterministic
`event_key` (= its action name) and the table has UNIQUE(thread_id, event_key),
so the "before interrupt" row of a node is written exactly once no matter how
many times the node re-runs. Side effects (posting to GitHub) live only in
`commit` / `auto_approve`, never before an `interrupt()`.

Audit rows never contain the diff or any secret — only counts, confidences,
decisions and short reasons.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
import uuid

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from rich.console import Console
from rich.panel import Panel

from common.checkpoint import sqlite_saver
from common.db import db_path, write_audit_event
from common.github import authenticated_login, fetch_pr, post_review_comment
from common.llm import get_llm, redact
from common.review import (
    DIFF_PREVIEW_CHARS,
    analysis_messages,
    decide,
    ensure_analysis,
    pending_interrupt,
    questions_for,
    refine_messages,
    render_comment_body,
    revise_messages,
    validate_answers,
    validate_choice,
)
from common.schemas import (
    AuditEntry,
    PRAnalysis,
    ReviewState,
    risk_level_for,
)

console = Console()
AGENT_ID = "pr-review-agent@v0.1"

# route decision → AuditEntry.decision vocabulary
ROUTE_DECISION = {"auto_approve": "auto", "human_approval": "pending", "escalate": "escalate"}


def _ms(t0: float) -> int:
    return int((time.monotonic() - t0) * 1000)


def default_reviewer() -> str:
    """GITHUB_USER, else the login that owns GITHUB_TOKEN, else 'unknown'."""
    return os.environ.get("GITHUB_USER") or authenticated_login() or "unknown"


def _reviewer(state, override: str | None = None) -> str:
    return override or state.get("reviewer_id") or default_reviewer()


async def audit(state, entry: AuditEntry, event_key: str | None = None) -> None:
    """Write one structured AuditEntry row to the `audit_events` table.

    `thread_id` and `pr_url` are taken from `state` so callers only build
    the entry itself. `event_key` defaults to the action name: each logical
    event is stored once per thread even if its node re-executes on resume.
    """
    await write_audit_event(
        thread_id=state["thread_id"], pr_url=state["pr_url"],
        entry=entry, event_key=event_key or entry.action,
    )


def _entry(action: str, a: PRAnalysis | None, decision: str, t0: float, reason: str | None,
           reviewer_id: str | None = None) -> AuditEntry:
    conf = a.confidence if a else 0.0
    return AuditEntry(
        agent_id=AGENT_ID, action=action, confidence=conf,
        risk_level=risk_level_for(conf) if a else "med",
        reviewer_id=reviewer_id, decision=decision, reason=reason,
        execution_time_ms=_ms(t0),
    )


# ─── Reference example ─────────────────────────────────────────────────────
async def node_fetch_pr(state):
    console.print("[cyan]→ fetch_pr[/cyan]")
    t0 = time.monotonic()
    with console.status("[dim]Fetching PR from GitHub...[/dim]"):
        pr = fetch_pr(state["pr_url"])
    console.print(f"  [green]✓[/green] {len(pr.files_changed)} files, head {pr.head_sha[:7]}")
    # Only the diff was fetched: confidence unknown → 0.0, neutral risk, decision "pending".
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="fetch_pr",
        confidence=0.0,
        risk_level="med",
        decision="pending",
        reason=f"Fetched {len(pr.files_changed)} files, head={pr.head_sha[:7]}",
        execution_time_ms=_ms(t0),
    ))
    return {
        "pr_title": pr.title,
        "pr_diff": pr.diff,
        "pr_files": pr.files_changed,
        "pr_head_sha": pr.head_sha,
    }
# ───────────────────────────────────────────────────────────────────────────


async def node_analyze(state):
    console.print("[cyan]→ analyze[/cyan]")
    t0 = time.monotonic()
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM reviewing the diff...[/dim]"):
        a = ensure_analysis(await llm.ainvoke(analysis_messages(state)))
    console.print(f"  [green]✓[/green] confidence={a.confidence:.0%}, {len(a.comments)} comment(s)")
    await audit(state, _entry(
        "analyze", a, "pending", t0,
        f"{a.confidence_reasoning} [{len(a.comments)} comments, {len(a.risk_factors)} risks]",
    ))
    return {"analysis": a}


async def node_route(state):
    console.print("[cyan]→ route[/cyan]")
    t0 = time.monotonic()
    a = state["analysis"]
    decision = decide(a.confidence)
    console.print(f"  [green]✓[/green] decision=[bold]{decision}[/bold] (confidence={a.confidence:.0%})")
    await audit(state, _entry(
        "route", a, ROUTE_DECISION[decision], t0,
        f"branch={decision} at confidence {a.confidence:.2f}",
    ))
    return {"decision": decision}


async def node_human_approval(state):
    t0 = time.monotonic()
    a = state["analysis"]

    # Written once even though this node re-runs on resume (unique event_key).
    await audit(state, _entry(
        "human_approval", a, "pending", t0,
        f"Approval requested at confidence {a.confidence:.2f}",
    ))

    resp = interrupt({
        "kind": "approval_request",
        "pr_url": state["pr_url"],
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "comments": [c.model_dump() for c in a.comments],
        "diff_preview": state["pr_diff"][:DIFF_PREVIEW_CHARS],
    })

    r = validate_choice(resp)
    reviewer = _reviewer(state, r.get("reviewer"))
    await audit(state, _entry(
        "human_approval_response", a, r["choice"], t0,
        r["feedback"] or f"Reviewer chose {r['choice']}", reviewer,
    ))
    return {"human_choice": r["choice"], "human_feedback": r["feedback"], "reviewer_id": reviewer}


async def node_revise(state):
    """Edit: the LLM rewrites the review from the reviewer's feedback."""
    console.print("[cyan]→ revise[/cyan]  [dim]applying reviewer feedback[/dim]")
    t0 = time.monotonic()
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM revising the review...[/dim]"):
        revised = ensure_analysis(await llm.ainvoke(revise_messages(state)))
    await audit(state, _entry(
        "revise", revised, "edit", t0,
        f"Revised from feedback; confidence {state['analysis'].confidence:.2f} → {revised.confidence:.2f}",
        state.get("reviewer_id"),
    ))
    return {"analysis": revised}


async def node_confirm(state):
    """Reviewer sees the refined / revised review and must confirm before it is posted."""
    t0 = time.monotonic()
    a = state["analysis"]
    await audit(state, _entry(
        "confirm", a, "pending", t0, "Final confirmation requested", state.get("reviewer_id"),
    ))
    resp = interrupt({
        "kind": "final_confirmation",
        "pr_url": state["pr_url"],
        "source": "escalation" if state.get("escalation_answers") else "edit",
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "risk_factors": a.risk_factors,
        "comments": [c.model_dump() for c in a.comments],
        "comment_preview": render_comment_body(state),
    })
    r = validate_choice(resp, ("approve", "reject"))
    reviewer = _reviewer(state, r.get("reviewer"))
    await audit(state, _entry(
        "confirm_response", a, r["choice"], t0,
        r["feedback"] or f"Reviewer chose {r['choice']}", reviewer,
    ))
    return {"human_choice": r["choice"], "human_feedback": r["feedback"], "reviewer_id": reviewer}


def _post(state, ok_label: str, fail_label: str) -> dict:
    """Post to GitHub. Failure is returned as `fail_label` + post_error, never as success."""
    try:
        posted = post_review_comment(state["pr_url"], render_comment_body(state))
        console.print(f"  [green]✓[/green] posted comment: {posted.html_url}")
        return {"final_action": ok_label, "comment_url": posted.html_url, "post_error": None}
    except Exception as e:
        console.print(f"  [red]✗[/red] post failed: {redact(e)}")
        return {"final_action": fail_label, "comment_url": None, "post_error": redact(e)}


async def node_commit(state):
    console.print("[cyan]→ commit[/cyan]")
    t0 = time.monotonic()
    a = state["analysis"]
    reviewer = state.get("reviewer_id")
    # Every path into commit has had an explicit human approve / reject.
    if state.get("human_choice") == "approve":
        update = _post(state, "committed", "commit_failed")
        if update["post_error"]:
            await audit(state, _entry("commit_failed", a, "approve", t0,
                                      f"Post failed: {update['post_error']}", reviewer))
        else:
            await audit(state, _entry("commit", a, "approve", t0,
                                      f"Posted {update['comment_url']}", reviewer))
        return update
    console.print(f"  [yellow]·[/yellow] skipping comment (choice={state.get('human_choice')})")
    await audit(state, _entry("commit", a, "reject", t0, "Rejected — no comment posted", reviewer))
    return {"final_action": "rejected"}


async def node_auto_approve(state):
    console.print("[cyan]→ auto_approve[/cyan]  [dim]high confidence — posting directly[/dim]")
    t0 = time.monotonic()
    a = state["analysis"]
    update = _post(state, "auto_approved", "auto_commit_failed")
    if update["post_error"]:
        await audit(state, _entry("auto_approve_failed", a, "auto", t0,
                                  f"Post failed: {update['post_error']}"))
    else:
        await audit(state, _entry("auto_approve", a, "auto", t0, f"Posted {update['comment_url']}"))
    return update


async def node_escalate(state):
    t0 = time.monotonic()
    a = state["analysis"]
    questions = questions_for(a)

    await audit(state, _entry(
        "escalate", a, "escalate", t0, f"Asked reviewer {len(questions)} question(s)",
    ))

    answers = interrupt({
        "kind": "escalation",
        "pr_url": state["pr_url"],
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "risk_factors": a.risk_factors,
        "questions": questions,
    })

    answers = validate_answers(questions, answers)
    reviewer = _reviewer(state)
    await audit(state, _entry(
        "escalate_answers", a, "escalate", t0, f"Reviewer answered {len(answers)} question(s)", reviewer,
    ))
    return {"escalation_answers": answers, "reviewer_id": reviewer}


async def node_synthesize(state):
    console.print("[cyan]→ synthesize[/cyan]")
    t0 = time.monotonic()
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM refining review with reviewer answers...[/dim]"):
        refined = ensure_analysis(await llm.ainvoke(refine_messages(state)))
    console.print(f"  [green]✓[/green] refined confidence={refined.confidence:.0%}")
    await audit(state, _entry(
        "synthesize", refined, "pending", t0,
        f"Refined confidence {state['analysis'].confidence:.2f} → {refined.confidence:.2f}: "
        f"{refined.confidence_reasoning}",
    ))
    return {"analysis": refined}


def after_approval(state) -> str:
    return "revise" if state.get("human_choice") == "edit" else "commit"


def build_graph(checkpointer):
    g = StateGraph(ReviewState)
    for name, fn in [
        ("fetch_pr", node_fetch_pr), ("analyze", node_analyze), ("route", node_route),
        ("auto_approve", node_auto_approve), ("human_approval", node_human_approval),
        ("revise", node_revise), ("confirm", node_confirm),
        ("commit", node_commit), ("escalate", node_escalate), ("synthesize", node_synthesize),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "fetch_pr")
    g.add_edge("fetch_pr", "analyze")
    g.add_edge("analyze", "route")
    g.add_conditional_edges(
        "route", lambda s: s["decision"],
        {"auto_approve": "auto_approve", "human_approval": "human_approval", "escalate": "escalate"},
    )
    g.add_edge("auto_approve", END)
    g.add_conditional_edges("human_approval", after_approval, {"revise": "revise", "commit": "commit"})
    g.add_edge("revise", "confirm")
    g.add_edge("escalate", "synthesize")
    g.add_edge("synthesize", "confirm")
    g.add_edge("confirm", "commit")
    g.add_edge("commit", END)
    return g.compile(checkpointer=checkpointer)


def _ask_choice(allowed: tuple[str, ...]) -> dict:
    while True:
        choice = console.input(f"[bold]{'/'.join(allowed)}?[/bold] ").strip().lower()
        feedback = ""
        if choice == "edit":
            feedback = console.input("What should change? ").strip()
        elif choice in allowed and choice != "approve":
            feedback = console.input("Feedback (optional): ").strip()
        try:
            return validate_choice({"choice": choice, "feedback": feedback}, allowed)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")


def handle_interrupt(payload):
    kind = payload["kind"]
    if kind == "approval_request":
        console.print(Panel.fit(
            payload["summary"], title=f"conf={payload['confidence']:.0%}", border_style="green",
        ))
        return _ask_choice(("approve", "reject", "edit"))
    if kind == "final_confirmation":
        console.print(Panel.fit(
            payload["comment_preview"], title=f"Post this review? conf={payload['confidence']:.0%}",
            border_style="yellow",
        ))
        return _ask_choice(("approve", "reject"))
    if kind == "escalation":
        console.print(Panel.fit(
            payload["summary"], title=f"Escalation conf={payload['confidence']:.0%}",
            border_style="yellow",
        ))
        if payload.get("risk_factors"):
            console.print("[red]Risks:[/red] " + "; ".join(payload["risk_factors"]))
        answers: dict[str, str] = {}
        for q in payload["questions"]:
            while not (answers.get(q) or "").strip():
                answers[q] = console.input(f"Q: {q}\nA: ").strip()
        return answers
    raise ValueError(f"Unknown interrupt kind: {kind}")


async def run(pr_url: str, thread_id: str | None):
    resuming = thread_id is not None
    thread_id = thread_id or str(uuid.uuid4())
    console.rule("[bold]Exercise 4 — SQLite audit trail[/bold]")
    console.print(f"[dim]PR: {pr_url}[/dim]")
    console.print(f"[dim]thread_id = {thread_id}[/dim]\n")

    async with sqlite_saver(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}

        snap = await app.aget_state(cfg) if resuming else None
        if snap is not None and snap.values:
            # Existing thread: continue it — never start a second review under this id.
            if not snap.next:
                console.print(f"[yellow]Thread already finished:[/yellow] {snap.values.get('final_action')}")
                return
            payload = pending_interrupt(snap)
            if payload is None:
                result = await app.ainvoke(None, cfg)
            else:
                result = await app.ainvoke(Command(resume=handle_interrupt(payload)), cfg)
        else:
            result = await app.ainvoke(
                {"pr_url": pr_url, "thread_id": thread_id, "reviewer_id": default_reviewer()}, cfg,
            )
        while "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            result = await app.ainvoke(Command(resume=handle_interrupt(payload)), cfg)

        console.rule("Final")
        console.print(f"final_action = {result.get('final_action')}")
        if result.get("comment_url"):
            console.print(f"comment: {result['comment_url']}")
        if result.get("post_error"):
            console.print(f"[red]post error: {result['post_error']}[/red]")
        console.print(f"\n[dim]Replay:[/dim] uv run python -m audit.replay --thread {thread_id}")


def main():
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--pr", required=True)
    p.add_argument("--thread", help="Resume an existing thread")
    args = p.parse_args()
    try:
        asyncio.run(run(args.pr, args.thread))
    except Exception as e:
        console.print(f"[red]✗ {type(e).__name__}:[/red] {redact(e)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
