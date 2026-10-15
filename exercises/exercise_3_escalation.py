"""Exercise 3 — Escalation branch with reviewer Q&A.

When confidence < 58%, the agent doesn't ask approve/reject — it asks specific
clarifying questions, re-runs the analysis with the answers (`synthesize`), and
shows the refined review for a final human confirmation before posting:

    escalate (interrupt: questions) → synthesize (LLM) → confirm (interrupt) → commit

The confirm step keeps a human in control of what gets posted; the 58–72%
branch keeps approve / reject / edit (edit = LLM rewrites the review from the
reviewer's feedback, then the same confirm step).
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
    questions_for,
    refine_messages,
    render_comment_body,
    revise_messages,
    validate_answers,
    validate_choice,
)
from common.schemas import PRAnalysis, ReviewState

console = Console()


def node_fetch_pr(state):
    console.print("[cyan]→ fetch_pr[/cyan]")
    with console.status("[dim]Fetching PR from GitHub...[/dim]"):
        pr = fetch_pr(state["pr_url"])
    console.print(f"  [green]✓[/green] {len(pr.files_changed)} files, head {pr.head_sha[:7]}")
    return {"pr_title": pr.title, "pr_diff": pr.diff, "pr_files": pr.files_changed, "pr_head_sha": pr.head_sha}


def node_analyze(state):
    console.print("[cyan]→ analyze[/cyan]")
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM reviewing the diff...[/dim]"):
        # The shared system prompt tells the model: if confidence < ESCALATE_THRESHOLD,
        # populate escalation_questions with 2–4 specific questions naming files / risks.
        analysis = ensure_analysis(llm.invoke(analysis_messages(state)))
    console.print(f"  [green]✓[/green] confidence={analysis.confidence:.0%}, {len(analysis.escalation_questions)} question(s)")
    return {"analysis": analysis}


def node_route(state):
    console.print("[cyan]→ route[/cyan]")
    c = state["analysis"].confidence
    decision = decide(c)
    console.print(f"  [green]✓[/green] decision=[bold]{decision}[/bold] (confidence={c:.0%})")
    return {"decision": decision}


def node_escalate(state: ReviewState) -> dict:
    """Ask the reviewer specific questions; return their answers in state."""
    a = state["analysis"]
    questions = questions_for(a)  # LLM questions, or generic fallbacks if it gave none
    answers = interrupt({
        "kind": "escalation",
        "pr_url": state["pr_url"],
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "risk_factors": a.risk_factors,
        "questions": questions,
    })
    return {"escalation_answers": validate_answers(questions, answers)}


def node_synthesize(state: ReviewState) -> dict:
    """Re-prompt LLM with the reviewer's answers and produce a refined review."""
    console.print("[cyan]→ synthesize[/cyan]")
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM refining review with reviewer answers...[/dim]"):
        refined = ensure_analysis(llm.invoke(refine_messages(state)))
    console.print(f"  [green]✓[/green] refined confidence={refined.confidence:.0%}")
    return {"analysis": refined}


def node_human_approval(state):
    a = state["analysis"]
    response = interrupt({
        "kind": "approval_request", "pr_url": state["pr_url"],
        "confidence": a.confidence, "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "comments": [c.model_dump() for c in a.comments],
        "diff_preview": state["pr_diff"][:DIFF_PREVIEW_CHARS],
    })
    r = validate_choice(response)
    return {"human_choice": r["choice"], "human_feedback": r["feedback"]}


def node_revise(state):
    console.print("[cyan]→ revise[/cyan]  [dim]applying reviewer feedback[/dim]")
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM revising the review...[/dim]"):
        return {"analysis": ensure_analysis(llm.invoke(revise_messages(state)))}


def node_confirm(state):
    """Show the refined/revised review; only an explicit 'approve' lets commit post it."""
    a = state["analysis"]
    response = interrupt({
        "kind": "final_confirmation", "pr_url": state["pr_url"],
        "source": "escalation" if state.get("escalation_answers") else "edit",
        "confidence": a.confidence, "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary, "risk_factors": a.risk_factors,
        "comments": [c.model_dump() for c in a.comments],
        "comment_preview": render_comment_body(state),
    })
    r = validate_choice(response, ("approve", "reject"))
    return {"human_choice": r["choice"], "human_feedback": r["feedback"]}


def _post(state, label: str) -> dict:
    try:
        posted = post_review_comment(state["pr_url"], render_comment_body(state))
        console.print(f"  [green]✓[/green] posted comment: {posted.html_url}")
        return {"final_action": label, "comment_url": posted.html_url, "post_error": None}
    except Exception as e:
        console.print(f"  [red]✗[/red] post failed: {redact(e)}")
        return {"final_action": "commit_failed", "comment_url": None, "post_error": redact(e)}


def node_commit(state):
    console.print("[cyan]→ commit[/cyan]")
    # Every path into commit has had an explicit human 'approve' / 'reject'.
    if state.get("human_choice") == "approve":
        label = "committed_after_escalation" if state.get("escalation_answers") else "committed"
        return _post(state, label)
    console.print(f"  [yellow]·[/yellow] skipping comment (choice={state.get('human_choice')})")
    return {"final_action": "rejected"}


def node_auto_approve(state):
    console.print("[cyan]→ auto_approve[/cyan]  [dim]high confidence — posting directly[/dim]")
    update = _post(state, "auto_approved")
    if update["final_action"] == "commit_failed":
        update["final_action"] = "auto_commit_failed"
    return update


def after_approval(state) -> str:
    return "revise" if state.get("human_choice") == "edit" else "commit"


def build_graph():
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
    return g.compile(checkpointer=MemorySaver())


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
            payload["summary"],
            title=f"Approve? conf={payload['confidence']:.0%}",
            border_style="green",
        ))
        return _ask_choice(("approve", "reject", "edit"))
    if kind == "escalation":
        console.print(Panel.fit(
            payload["summary"],
            title=f"Escalation conf={payload['confidence']:.0%}",
            border_style="yellow",
        ))
        if payload.get("risk_factors"):
            console.print("[red]Risks:[/red] " + "; ".join(payload["risk_factors"]))
        answers: dict[str, str] = {}
        for q in payload["questions"]:
            while not (answers.get(q) or "").strip():
                answers[q] = console.input(f"Q: {q}\nA: ").strip()
        return answers
    if kind == "final_confirmation":
        console.print(Panel.fit(
            payload["comment_preview"],
            title=f"Post this review? conf={payload['confidence']:.0%}",
            border_style="yellow",
        ))
        return _ask_choice(("approve", "reject"))
    raise ValueError(f"Unknown interrupt kind: {kind}")


def main():
    load_dotenv()
    p = argparse.ArgumentParser(); p.add_argument("--pr", required=True)
    args = p.parse_args()

    console.rule("[bold]Exercise 3 — escalation with reviewer Q&A[/bold]")
    console.print(f"[dim]PR: {args.pr}[/dim]\n")

    app = build_graph()
    thread_id = str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": thread_id}}
    console.print(f"[dim]thread_id = {thread_id}[/dim]\n")

    try:
        result = app.invoke({"pr_url": args.pr, "thread_id": thread_id}, cfg)
        while "__interrupt__" in result:
            result = app.invoke(Command(resume=handle_interrupt(result["__interrupt__"][0].value)), cfg)
    except Exception as e:
        console.print(f"[red]✗ {type(e).__name__}:[/red] {redact(e)}")
        raise SystemExit(1)

    console.rule("Final")
    console.print(f"final_action = {result.get('final_action')}")
    if "analysis" in result:
        console.print(f"final confidence = {result['analysis'].confidence:.0%}")
    if result.get("comment_url"):
        console.print(f"comment: {result['comment_url']}")
    if result.get("post_error"):
        console.print(f"[red]post error: {result['post_error']}[/red]")


if __name__ == "__main__":
    main()
