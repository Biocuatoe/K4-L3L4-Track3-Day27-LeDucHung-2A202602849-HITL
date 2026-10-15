"""Exercise 2 — HITL with interrupt() + Command(resume=...).

Starting from exercise 1, turn the `human_approval` node into real HITL: call
interrupt() with a payload containing diff + reasoning, then resume the graph
with Command(resume=<user choice>).

Reviewer choices:
  approve — post the review as-is.
  reject  — post nothing.
  edit    — the reviewer's feedback is sent back to the LLM, which rewrites the
            review; the revised review is shown for a final approve/reject
            confirmation (second interrupt) before anything is posted.

Posting happens only in `commit` (never before an interrupt, so a node re-run
on resume cannot post twice). The escalate branch is still a placeholder here —
exercise 3 implements it.
"""

from __future__ import annotations

import argparse
import uuid

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from rich.console import Console
from rich.panel import Panel

from common.github import fetch_pr, post_review_comment
from common.llm import get_llm, redact
from common.review import (
    DIFF_PREVIEW_CHARS,
    analysis_messages,
    decide,
    ensure_analysis,
    render_comment_body,
    revise_messages,
    validate_choice,
)
from common.schemas import PRAnalysis, ReviewState

console = Console()


def node_fetch_pr(state: ReviewState) -> dict:
    console.print("[cyan]→ fetch_pr[/cyan]")
    with console.status("[dim]Fetching PR from GitHub...[/dim]"):
        pr = fetch_pr(state["pr_url"])
    console.print(f"  [green]✓[/green] {len(pr.files_changed)} files, head {pr.head_sha[:7]}")
    return {"pr_title": pr.title, "pr_diff": pr.diff, "pr_files": pr.files_changed, "pr_head_sha": pr.head_sha}


def node_analyze(state: ReviewState) -> dict:
    console.print("[cyan]→ analyze[/cyan]")
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM reviewing the diff...[/dim]"):
        analysis = ensure_analysis(llm.invoke(analysis_messages(state)))
    console.print(f"  [green]✓[/green] confidence={analysis.confidence:.0%}, {len(analysis.comments)} comment(s)")
    return {"analysis": analysis}


def node_route(state: ReviewState) -> dict:
    console.print("[cyan]→ route[/cyan]")
    c = state["analysis"].confidence
    decision = decide(c)
    console.print(f"  [green]✓[/green] decision=[bold]{decision}[/bold] (confidence={c:.0%})")
    return {"decision": decision}


def node_human_approval(state: ReviewState) -> dict:
    """Pause and ask the human. No side effects here: the node re-runs on resume."""
    a = state["analysis"]
    response = interrupt({
        "kind": "approval_request",
        "pr_url": state["pr_url"],
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "comments": [c.model_dump() for c in a.comments],
        "diff_preview": state["pr_diff"][:DIFF_PREVIEW_CHARS],
    })
    r = validate_choice(response)
    return {"human_choice": r["choice"], "human_feedback": r["feedback"]}


def node_revise(state: ReviewState) -> dict:
    """Edit: rewrite the review using the reviewer's feedback."""
    console.print("[cyan]→ revise[/cyan]  [dim]applying reviewer feedback[/dim]")
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM revising the review...[/dim]"):
        revised = ensure_analysis(llm.invoke(revise_messages(state)))
    return {"analysis": revised}


def node_confirm(state: ReviewState) -> dict:
    """Final confirmation of the revised review. Only 'approve' leads to posting."""
    a = state["analysis"]
    response = interrupt({
        "kind": "final_confirmation",
        "pr_url": state["pr_url"],
        "source": "edit",
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "risk_factors": a.risk_factors,
        "comments": [c.model_dump() for c in a.comments],
        "comment_preview": render_comment_body(state),
    })
    r = validate_choice(response, ("approve", "reject"))
    return {"human_choice": r["choice"], "human_feedback": r["feedback"]}


def _post(state: ReviewState, label: str) -> dict:
    """Post the review comment. Returns the state update — failure is never reported as success."""
    try:
        posted = post_review_comment(state["pr_url"], render_comment_body(state))
        console.print(f"  [green]✓[/green] posted comment: {posted.html_url}")
        return {"final_action": label, "comment_url": posted.html_url, "post_error": None}
    except Exception as e:
        console.print(f"  [red]✗[/red] post failed: {redact(e)}")
        return {"final_action": "commit_failed", "comment_url": None, "post_error": redact(e)}


def node_commit(state: ReviewState) -> dict:
    console.print("[cyan]→ commit[/cyan]")
    if state.get("human_choice") == "approve":
        return _post(state, "committed")
    console.print(f"  [yellow]·[/yellow] skipping comment (choice={state.get('human_choice')})")
    return {"final_action": "rejected"}


def node_auto_approve(state: ReviewState) -> dict:
    console.print("[cyan]→ auto_approve[/cyan]  [dim]high confidence — posting directly[/dim]")
    update = _post(state, "auto_approved")
    if update["final_action"] == "commit_failed":
        update["final_action"] = "auto_commit_failed"
    return update


def node_escalate(state: ReviewState) -> dict:
    return {"final_action": "pending_escalation"}


def after_approval(state: ReviewState) -> str:
    return "revise" if state.get("human_choice") == "edit" else "commit"


def build_graph():
    g = StateGraph(ReviewState)
    for name, fn in [
        ("fetch_pr", node_fetch_pr), ("analyze", node_analyze), ("route", node_route),
        ("auto_approve", node_auto_approve), ("human_approval", node_human_approval),
        ("revise", node_revise), ("confirm", node_confirm),
        ("escalate", node_escalate), ("commit", node_commit),
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
    g.add_edge("confirm", "commit")
    g.add_edge("commit", END)
    g.add_edge("escalate", END)
    return g.compile(checkpointer=MemorySaver())


def prompt_human(payload: dict) -> dict:
    kind = payload["kind"]
    if kind == "final_confirmation":
        console.print(Panel.fit(
            f"[bold]Revised review (confidence {payload['confidence']:.0%})[/bold]\n\n"
            f"{payload['comment_preview']}",
            title="Final confirmation", border_style="yellow",
        ))
        allowed = ("approve", "reject")
    else:
        console.print(Panel.fit(
            f"[bold]Confidence:[/bold] {payload['confidence']:.0%}\n"
            f"[dim]{payload['confidence_reasoning']}[/dim]\n\n"
            f"[bold]Summary:[/bold] {payload['summary']}",
            title="Approval request",
            border_style="green",
        ))
        for c in payload.get("comments", []):
            console.print(f"  [{c['severity']}] {c['file']}:{c.get('line') or '?'} — {c['body']}")
        if payload.get("diff_preview"):
            console.print("\n[dim]--- diff preview ---[/dim]")
            console.print(payload["diff_preview"], markup=False)
        allowed = ("approve", "reject", "edit")

    while True:
        choice = console.input(f"\n[bold]Choice ({'/'.join(allowed)})?[/bold] ").strip().lower()
        feedback = ""
        if choice == "edit":
            feedback = console.input("What should change? ").strip()
        elif choice in allowed and choice != "approve":
            feedback = console.input("Feedback (optional): ").strip()
        try:
            return validate_choice({"choice": choice, "feedback": feedback}, allowed)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", required=True)
    args = parser.parse_args()

    console.rule("[bold]Exercise 2 — HITL with interrupt()[/bold]")
    console.print(f"[dim]PR: {args.pr}[/dim]\n")

    app = build_graph()
    thread_id = str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": thread_id}}  # same thread_id for every resume
    console.print(f"[dim]thread_id = {thread_id}[/dim]\n")

    try:
        result = app.invoke({"pr_url": args.pr, "thread_id": thread_id}, cfg)
        while "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            answer = prompt_human(payload)
            result = app.invoke(Command(resume=answer), cfg)
    except Exception as e:
        console.print(f"[red]✗ {type(e).__name__}:[/red] {redact(e)}")
        raise SystemExit(1)

    console.rule("Done")
    console.print(result.get("final_action"))
    if result.get("comment_url"):
        console.print(f"comment: {result['comment_url']}")
    if result.get("post_error"):
        console.print(f"[red]post error: {result['post_error']}[/red]")


if __name__ == "__main__":
    main()
