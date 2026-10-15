"""Shared fakes. No test touches the network, a real LLM, or a real PR."""

from __future__ import annotations

import pytest

from common.github import GitHubError, PostedComment, PullRequest
from common.schemas import PRAnalysis, ReviewComment

PR_URL = "https://github.com/acme/demo/pull/7"


def make_analysis(conf: float, questions: list[str] | None = None, summary: str = "Adds a thing") -> PRAnalysis:
    return PRAnalysis(
        summary=summary,
        risk_factors=["md5 hashing in storage.py"] if conf < 0.58 else [],
        comments=[ReviewComment(file="a.py", line=3, severity="issue", body="Check this")],
        confidence=conf,
        confidence_reasoning=f"because {conf}",
        escalation_questions=questions or [],
    )


class FakeLLM:
    """Stands in for get_llm(): hands out queued PRAnalysis objects and records prompts."""

    def __init__(self, analyses):
        self.queue = list(analyses)
        self.calls: list[list[dict]] = []

    def with_structured_output(self, schema):
        assert schema is PRAnalysis
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        return self.queue.pop(0)

    async def ainvoke(self, messages):
        return self.invoke(messages)


class Recorder:
    def __init__(self):
        self.posts: list[tuple[str, str]] = []
        self.fail_with: Exception | None = None

    def post(self, pr_url, body):
        if self.fail_with:
            raise self.fail_with
        self.posts.append((pr_url, body))
        return PostedComment(id=99, html_url=f"{pr_url}#issuecomment-99")


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HITL_DB_PATH", str(tmp_path / "audit.db"))
    monkeypatch.setenv("GITHUB_USER", "tester")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_fake_token_for_tests_0000")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fake-key-for-tests-0000")


@pytest.fixture
def install(monkeypatch):
    """install(module, [analyses]) → (Recorder, FakeLLM); patches the module's I/O seams."""

    def _install(module, analyses):
        rec, llm = Recorder(), FakeLLM(analyses)
        pr = PullRequest(
            url=PR_URL, owner="acme", repo="demo", number=7, title="Demo", author="bob",
            base_ref="main", head_ref="x", head_sha="abcdef1234", diff="+print('hi')\n",
            files_changed=["a.py"],
        )
        monkeypatch.setattr(module, "fetch_pr", lambda url: pr)
        monkeypatch.setattr(module, "get_llm", lambda *a, **k: llm)
        monkeypatch.setattr(module, "post_review_comment", rec.post, raising=False)
        return rec, llm

    return _install


__all__ = ["PR_URL", "make_analysis", "GitHubError"]
