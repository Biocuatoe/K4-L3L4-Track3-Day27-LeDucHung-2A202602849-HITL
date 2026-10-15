"""Thin wrapper around the GitHub REST API.

We talk to GitHub directly via `httpx` instead of shelling out to `gh`, so
students don't need a `gh` binary — just `GITHUB_TOKEN` in `.env`.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import httpx

API = "https://api.github.com"
PR_URL_RE = re.compile(
    r"^https?://(?:www\.)?github\.com/([\w.-]+)/([\w.-]+)/pull/(\d+)/?(?:[/?#].*)?$"
)


class GitHubError(RuntimeError):
    """Any GitHub failure (bad URL, HTTP error, network error). Never contains the token."""


@dataclass
class PostedComment:
    id: int
    html_url: str


@dataclass
class PullRequest:
    url: str
    owner: str
    repo: str
    number: int
    title: str
    author: str
    base_ref: str
    head_ref: str
    head_sha: str
    diff: str
    files_changed: list[str]


def _token() -> str:
    tok = os.environ.get("GITHUB_TOKEN")
    if not tok:
        raise RuntimeError(
            "GITHUB_TOKEN is not set. Copy .env.example to .env and paste a Personal "
            "Access Token (https://github.com/settings/tokens/new) with `public_repo` scope."
        )
    return tok


def _headers(accept: str = "application/vnd.github+json") -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_token()}",
        "Accept": accept,
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "Day27-HITL-Lab",
    }


def parse_pr_url(pr_url: str) -> tuple[str, str, int]:
    m = PR_URL_RE.match((pr_url or "").strip())
    if not m:
        raise ValueError(
            f"Not a GitHub PR URL: {pr_url!r} "
            "(expected https://github.com/<owner>/<repo>/pull/<number>)"
        )
    return m.group(1), m.group(2), int(m.group(3))


def _check(resp: httpx.Response, what: str) -> None:
    """Raise GitHubError with an actionable message on any non-2xx response."""
    if resp.is_success:
        return
    try:
        detail = resp.json().get("message", "")
    except Exception:
        detail = ""
    hint = {
        401: "token missing/invalid — check GITHUB_TOKEN",
        403: "forbidden — token lacks scope (`public_repo`) or rate limit hit",
        404: "not found — wrong PR URL, or token cannot see this repo",
        422: "GitHub rejected the request",
    }.get(resp.status_code, "")
    parts = [f"GitHub {what} failed: HTTP {resp.status_code}", detail, hint]
    raise GitHubError(" — ".join(p for p in parts if p))


def authenticated_login() -> str | None:
    """Login of the user that owns GITHUB_TOKEN, or None if it cannot be determined."""
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{API}/user", headers=_headers())
        return resp.json().get("login") if resp.is_success else None
    except Exception:
        return None


def fetch_pr(pr_url: str) -> PullRequest:
    """Fetch PR metadata + unified diff via the GitHub REST API."""
    owner, repo, number = parse_pr_url(pr_url)
    base = f"{API}/repos/{owner}/{repo}/pulls/{number}"

    try:
        with httpx.Client(timeout=30.0) as client:
            meta_resp = client.get(base, headers=_headers())
            _check(meta_resp, "PR lookup")
            meta = meta_resp.json()

            diff_resp = client.get(
                base, headers=_headers(accept="application/vnd.github.v3.diff")
            )
            _check(diff_resp, "diff download")
            diff = diff_resp.text

            files_resp = client.get(
                f"{base}/files", headers=_headers(), params={"per_page": 100}
            )
            _check(files_resp, "file list")
            files = [f["filename"] for f in files_resp.json()]
    except httpx.HTTPError as e:
        raise GitHubError(f"GitHub request failed: {type(e).__name__}") from e

    return PullRequest(
        url=pr_url,
        owner=owner,
        repo=repo,
        number=number,
        title=meta["title"],
        author=meta["user"]["login"],
        base_ref=meta["base"]["ref"],
        head_ref=meta["head"]["ref"],
        head_sha=meta["head"]["sha"],
        diff=diff,
        files_changed=files,
    )


def post_review_comment(pr_url: str, body: str) -> PostedComment:
    """Post a top-level discussion comment back to the PR.

    Uses the Issues endpoint (PRs are issues under the hood for top-level
    comments). For formal Approve/Request-changes use the Reviews endpoint
    instead — which requires collaborator status on the target repo.

    Returns the created comment (with its permalink). Raises GitHubError on any
    non-2xx response or network failure — never returns on failure.
    """
    owner, repo, number = parse_pr_url(pr_url)
    url = f"{API}/repos/{owner}/{repo}/issues/{number}/comments"
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, headers=_headers(), json={"body": body})
    except httpx.HTTPError as e:
        raise GitHubError(f"GitHub request failed: {type(e).__name__}") from e
    _check(resp, "comment post")
    data = resp.json()
    cid = int(data["id"])
    return PostedComment(
        id=cid,
        html_url=data.get("html_url") or f"{pr_url.strip().split('#')[0]}#issuecomment-{cid}",
    )
