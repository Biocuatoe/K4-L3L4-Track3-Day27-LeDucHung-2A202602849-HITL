import httpx
import pytest

from common import github
from common.github import GitHubError, fetch_pr, parse_pr_url, post_review_comment


def patch_http(monkeypatch, handler):
    real = httpx.Client
    monkeypatch.setattr(github.httpx, "Client",
                        lambda **kw: real(transport=httpx.MockTransport(handler), **kw))


def test_parse_valid_urls():
    assert parse_pr_url("https://github.com/VinUni-AI20k/PR-Demo/pull/1") == ("VinUni-AI20k", "PR-Demo", 1)
    assert parse_pr_url("  http://www.github.com/a/b.c/pull/42/files?x=1 ") == ("a", "b.c", 42)


@pytest.mark.parametrize("bad", [
    "", "not a url", "https://github.com/a/b/issues/3", "https://gitlab.com/a/b/pull/3",
    "https://github.com/a/b/pull/x", "see https://github.com/a/b/pull/3", "https://github.com/a/pull/3",
])
def test_parse_rejects_malformed(bad):
    with pytest.raises(ValueError):
        parse_pr_url(bad)


def _api(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path.endswith("/files"):
        return httpx.Response(200, json=[{"filename": "a.py"}, {"filename": "b.py"}])
    if "diff" in request.headers["accept"]:
        return httpx.Response(200, text="diff --git a/a.py b/a.py\n+x\n")
    return httpx.Response(200, json={
        "title": "T", "user": {"login": "bob"}, "base": {"ref": "main"},
        "head": {"ref": "f", "sha": "0123456789"}})


def test_fetch_pr_success(monkeypatch):
    patch_http(monkeypatch, _api)
    pr = fetch_pr("https://github.com/a/b/pull/5")
    assert (pr.title, pr.author, pr.number, pr.files_changed) == ("T", "bob", 5, ["a.py", "b.py"])
    assert pr.diff.startswith("diff --git")


@pytest.mark.parametrize("status,needle", [(404, "not found"), (401, "GITHUB_TOKEN"), (403, "scope")])
def test_fetch_pr_http_errors(monkeypatch, status, needle):
    patch_http(monkeypatch, lambda r: httpx.Response(status, json={"message": "nope"}))
    with pytest.raises(GitHubError) as e:
        fetch_pr("https://github.com/a/b/pull/5")
    assert str(status) in str(e.value) and needle in str(e.value)
    assert "ghp_fake" not in str(e.value)


def test_fetch_pr_network_error(monkeypatch):
    def boom(request):
        raise httpx.ConnectError("down")
    patch_http(monkeypatch, boom)
    with pytest.raises(GitHubError, match="ConnectError"):
        fetch_pr("https://github.com/a/b/pull/5")


def test_missing_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN")
    with pytest.raises(RuntimeError, match="GITHUB_TOKEN"):
        fetch_pr("https://github.com/a/b/pull/5")


def test_post_success_returns_link(monkeypatch):
    seen = {}

    def handler(request):
        seen["url"], seen["auth"] = str(request.url), request.headers["authorization"]
        return httpx.Response(201, json={"id": 123, "html_url": "https://github.com/a/b/pull/5#issuecomment-123"})

    patch_http(monkeypatch, handler)
    c = post_review_comment("https://github.com/a/b/pull/5", "hi")
    assert c.id == 123 and c.html_url.endswith("#issuecomment-123")
    assert seen["url"].endswith("/repos/a/b/issues/5/comments")


def test_post_builds_link_when_missing(monkeypatch):
    patch_http(monkeypatch, lambda r: httpx.Response(201, json={"id": 8}))
    assert post_review_comment("https://github.com/a/b/pull/5", "x").html_url == \
        "https://github.com/a/b/pull/5#issuecomment-8"


@pytest.mark.parametrize("status", [401, 403, 404, 422, 500])
def test_post_failure_raises(monkeypatch, status):
    patch_http(monkeypatch, lambda r: httpx.Response(status, json={"message": "bad"}))
    with pytest.raises(GitHubError, match=str(status)):
        post_review_comment("https://github.com/a/b/pull/5", "x")


def test_redact_hides_secrets():
    from common.llm import redact
    out = redact("boom ghp_fake_token_for_tests_0000 and sk-or-fake-key-for-tests-0000")
    assert "ghp_fake" not in out and "sk-or-fake" not in out
