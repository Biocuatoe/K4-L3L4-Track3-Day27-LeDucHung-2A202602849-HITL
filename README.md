# Day27 — Track 3: HITL PR Review Agent

A 2-hour lab that builds a human-in-the-loop pull-request review agent in **LangGraph**, end-to-end.

> 1. Agent reads a PR, analyzes code changes, proposes review comments.
> 2. Confidence ~72% → show diff + reasoning → user approves → agent commits the review.
> 3. Confidence ~58% → escalate: show context + specific questions for the reviewer.
> 4. Every interaction is written to a SQLite audit trail; full sessions can be replayed.

The four exercises (`exercises/`) plus the Streamlit app (`app.py`) together build the full system. All five are implemented.

## Background — what problem does this solve?

Code review is the bottleneck of most engineering teams: a senior engineer's time is finite, but PRs keep coming. PRs fall into roughly three buckets:

- **Mechanical** (typo fixes, dependency bumps, one-line refactors) — safe, but still need someone to click "Approve". Spending senior time here is pure waste.
- **Medium risk** (small features, schema additions) — usually fine, but a pair of human eyes catches edge cases. Reviewer time here pays off.
- **High risk** (auth, migrations, anything security-touching) — definitely need a human, often with *specific questions* about intent and threat model.

If you apply the same human-driven review process to all three, you burn senior bandwidth on trivial PRs **and** you also fail to give risky PRs the careful attention they deserve.

### The HITL solution

This lab builds an agent that auto-triages every PR:

1. The LLM reads the diff and produces a structured review with a **self-reported confidence score**.
2. **Confidence ≥ 73%** → the agent posts the review comment itself; no human in the loop.
3. **58–72%** → the agent pauses (`interrupt()`) and shows the reviewer the diff + LLM reasoning + a one-click **approve / reject / edit** panel. On approve, the comment posts.
4. **< 58%** → the agent **escalates strongly**. It shows the reviewer *specific questions the LLM is uncertain about* ("Why MD5? Is `SYNC_URL` meant to be HTTPS in production?") and waits for answers. Then it re-asks the LLM with the answers in context, shows the **refined** review, and posts it only after the reviewer confirms.

Every step writes a row to a structured audit table — every routing decision, every human interaction, every LLM call — so the team can replay any session for compliance or debugging.

### Flow diagram

```mermaid
flowchart TD
    Fetch[fetch_pr] --> Analyze[analyze]
    Analyze --> Route{confidence}
    Route -- ">= 73%" --> Auto[auto_approve]
    Route -- "58–72%" --> Human["human_approval<br/>(interrupt)"]
    Route -- "< 58%" --> Escalate["escalate<br/>(interrupt: questions)"]
    Human -- approve / reject --> Commit[commit]
    Human -- edit --> Revise["revise (LLM rewrites)"]
    Escalate --> Synth["synthesize (LLM re-analysis)"]
    Revise --> Confirm["confirm<br/>(interrupt: final approve / reject)"]
    Synth --> Confirm
    Confirm --> Commit
    Auto --> End([END])
    Commit --> End
```

Every node also writes one row to `audit_events`. The graph state is persisted by `AsyncSqliteSaver`, so an interrupted session can resume from exactly where it stopped — even after the process is killed.

## Lab assignment

**Goal:** Build a HITL agent with LangGraph `interrupt()` **+ a Streamlit approval UI**.

**Deliverables:**
- HITL agent
- Approval UI (Streamlit)
- Confidence-based routing
- SQLite audit trail (the original brief said PostgreSQL)

**What this scaffolding already covers.** Exercises 1–4 deliver the HITL agent, confidence-based routing, and the structured audit trail. Two implementation notes:
- The lab uses **SQLite** (`./hitl_audit.db`) instead of PostgreSQL for zero-setup. The schema is row-oriented with first-class columns — the same `AuditEntry` queries transfer to Postgres in production by swapping the checkpointer and connection string.
- The approval UI is currently a terminal panel (Rich). The Streamlit UI is the fifth exercise — see [Exercise 5](#exercise-5--streamlit-approval-ui-apppy) below.

## Confidence routing

| Confidence  | Branch           | Has HITL?                          | Demo PR |
|-------------|------------------|------------------------------------|---------|
| ≥ 73%       | `auto_approve`   | No — agent commits directly        | (use any tiny PR) |
| 58–72%      | `human_approval` | Yes — approve / reject / edit      | PR-Demo #1 |
| < 58%       | `escalate`       | Yes — answer specific questions    | PR-Demo #2 |

Thresholds live in `common/schemas.py` (`AUTO_APPROVE_THRESHOLD = 0.73`, `ESCALATE_THRESHOLD = 0.58`): `>= 0.73` auto, `0.58 <= c < 0.73` human approval, `< 0.58` escalate. Confidence is **model-generated** and varies between runs — routing is deterministic given the number, but the number is not.

## Layout

```
.
├── README.md                # this file — the only documentation
├── pyproject.toml
├── .env.example
│
├── common/                  # shared utilities
│   ├── llm.py               # ChatOpenAI factory (OpenRouter)
│   ├── github.py            # httpx REST API client: fetch_pr / post_review_comment
│   ├── db.py                # aiosqlite connection + write_audit_event
│   ├── review.py            # prompts, routing, payload validation, comment rendering
│   └── schemas.py           # ReviewState (TypedDict) + PRAnalysis + AuditEntry (Pydantic)
│
├── app.py + review_service.py   # Exercise 5: Streamlit UI + graph service layer
├── tests/                   # pytest suite (mocked I/O)
├── exercises/               # the four exercises
│   ├── exercise_1_confidence.py
│   ├── exercise_2_hitl.py
│   ├── exercise_3_escalation.py
│   └── exercise_4_audit.py
│
└── audit/                   # provided
    ├── schema.sql           # CREATE TABLE audit_events
    └── replay.py            # `python -m audit.replay --thread <id>` / --list
```

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- An LLM API key: [Groq](https://console.groq.com/keys) (free tier, default) or [OpenRouter](https://openrouter.ai/keys). Select with `LLM_PROVIDER=groq|openrouter`.
- A GitHub Personal Access Token — see next section

No Docker, no Postgres install, no `gh` CLI. The agent talks to GitHub directly via the REST API. Audit + checkpointer state both live in a single SQLite file (`./hitl_audit.db`) created automatically on first run.

## Get a GitHub Personal Access Token (PAT)

The agent calls the GitHub REST API directly to **read PR diffs** and **post review comments**. A PAT is required.

1. Open <https://github.com/settings/tokens/new> (classic — simplest).
2. Fill in:
   - **Note:** `Day27 HITL Lab`
   - **Expiration:** 30 days
   - **Scopes:** tick **`public_repo`** (enough for `VinUni-AI20k/PR-Demo` since it's a public repo).
3. Click **Generate token** → copy the `ghp_...` value immediately (shown only once).
4. Paste into `.env`:
   ```bash
   echo "GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" >> .env
   ```
5. Verify:
   ```bash
   curl -sH "Authorization: Bearer $(grep ^GITHUB_TOKEN .env | cut -d= -f2)" \
        https://api.github.com/repos/VinUni-AI20k/PR-Demo | grep '"full_name"'
   ```
   If you see `"full_name": "VinUni-AI20k/PR-Demo"` → OK. `401` → wrong scope or typo.

**Security:** `.env` is git-ignored; never commit a token. Use a classic PAT with only `public_repo` (or a fine-grained token limited to Issues read/write and Pull requests read on the target repo), a short expiry, and revoke it after the lab. Comments are posted under the token owner's account. Errors shown in the CLI/UI pass through a redactor that strips configured secret values.

## Setup (once)

```bash
git clone <lab-repo-url> && cd Day27-Track3-HITL
uv sync                                     # install Python deps
cp .env.example .env && $EDITOR .env        # set OPENROUTER_API_KEY and GITHUB_TOKEN (GITHUB_USER optional)
```

The SQLite file `hitl_audit.db` is created on demand when exercise 4 runs.

## Demo PRs

Two PRs are kept at the public repo **<https://github.com/VinUni-AI20k/PR-Demo>** for testing:

| PR | URL                                                                  | Expected branch          |
|----|----------------------------------------------------------------------|--------------------------|
| #1 — Add task priority field        | `https://github.com/VinUni-AI20k/PR-Demo/pull/1` | `human_approval` (~65%) |
| #2 — Add user login + cloud sync    | `https://github.com/VinUni-AI20k/PR-Demo/pull/2` | `escalate` (<58%)       |

PR #1 is mechanical (~40 lines) with one open question (schema migration). PR #2 has multiple red flags (MD5 password hashing, plaintext token storage, SQL injection in `Storage.add`, hard-coded user_id, no tests for new code) — the LLM should drop into the escalate branch and ask the reviewer specific questions.

## The 5 exercises

Do them in order — each builds on the previous. Each section names what is implemented; run the command shown.

### Exercise 1 — Confidence routing (`exercise_1_confidence.py`)

Build a small LangGraph that fetches a PR, analyzes it, then routes to one of three terminal nodes by confidence (`auto_approve` / `human_approval` / `escalate`). No HITL yet — terminals just print which branch fired.

**You implement:** `node_analyze` (call LLM with `with_structured_output(PRAnalysis)`), `node_route` (return decision string), and the graph wiring (`add_node`, `add_edge`, `add_conditional_edges`).

```bash
uv run python exercises/exercise_1_confidence.py --pr https://github.com/VinUni-AI20k/PR-Demo/pull/1
uv run python exercises/exercise_1_confidence.py --pr https://github.com/VinUni-AI20k/PR-Demo/pull/2
```

Success = the two PRs print **different** branches.

### Exercise 2 — HITL with `interrupt()` (`exercise_2_hitl.py`)

Turn the `human_approval` node from a placeholder into a real pause-and-ask. Use `interrupt(payload)` inside the node, then resume the graph from `main()` with `Command(resume=<user_answer>)`.

**You implement:** `node_human_approval` (call `interrupt()` with diff + reasoning), graph compile with `MemorySaver` checkpointer, and the resume loop in `main()`:
```python
while "__interrupt__" in result:
    payload = result["__interrupt__"][0].value
    answer = prompt_human(payload)
    result = app.invoke(Command(resume=answer), cfg)
```

```bash
uv run python exercises/exercise_2_hitl.py --pr https://github.com/VinUni-AI20k/PR-Demo/pull/1
# → terminal pauses, shows a green panel, asks approve / reject / edit
```

### Exercise 3 — Escalation with reviewer Q&A (`exercise_3_escalation.py`)

When confidence is low, the agent shouldn't ask "approve/reject" — it should ask **specific questions** to gather context, then re-run the analysis with the answers.

**You implement:** prompt the LLM to populate `escalation_questions` when low-confidence, `node_escalate` (calls `interrupt(kind="escalation")` with the questions), and `node_synthesize` (re-prompts the LLM using the reviewer's answers to produce a refined review).

```bash
uv run python exercises/exercise_3_escalation.py --pr https://github.com/VinUni-AI20k/PR-Demo/pull/2
# → terminal shows a yellow panel; type an answer for each question
```

### Exercise 4 — Structured SQLite audit trail (`exercise_4_audit.py`)

Use `AsyncSqliteSaver` so the graph can resume after a crash, **and** at every meaningful node event, write one structured row to `audit_events` using the `AuditEntry` Pydantic model defined in `common/schemas.py`:

```python
class AuditEntry(BaseModel):
    timestamp: datetime          # auto: datetime.now(timezone.utc)
    agent_id: str                # e.g. "pr-review-agent@v0.1"
    action: str                  # "fetch_pr" | "analyze" | "route" | "human_approval" | ...
    confidence: float            # 0.0 – 1.0
    risk_level: str              # "low" | "med" | "high"  (use risk_level_for(confidence))
    reviewer_id: str | None      # GitHub username on HITL steps, None otherwise
    decision: str                # "auto" | "approve" | "reject" | "edit" | "escalate" | "pending"
    reason: str | None           # LLM reasoning OR human feedback
    execution_time_ms: int       # measure with time.monotonic() per node
```

Each field maps to a first-class SQL column, so auditors can query directly:
```sql
SELECT AVG(confidence) FROM audit_events WHERE decision = 'approve';
SELECT * FROM audit_events WHERE risk_level = 'high' AND decision = 'auto';   -- shouldn't happen!
```

**You implement:** the `audit(...)` helper (calls `write_audit_event(thread_id=..., pr_url=..., entry=AuditEntry(...))`) and construct an `AuditEntry` in every node. The `AsyncSqliteSaver` and resume loop are already wired for you in `run()`.

```bash
uv run python exercises/exercise_4_audit.py --pr https://github.com/VinUni-AI20k/PR-Demo/pull/1
# → after the run, replay the full session:
uv run python -m audit.replay --thread <thread_id-printed-above>
uv run python -m audit.replay --list   # see recent threads
```

#### Why two storage mechanisms?

| Mechanism                            | Purpose                              | Schema             | Audience          |
|--------------------------------------|---------------------------------------|--------------------|-------------------|
| LangGraph **checkpointer** (SQLite)  | Resume graph after crash, time-travel | Binary blob        | LangGraph runtime |
| Table **`audit_events`**             | Structured decision log               | First-class columns (queryable) | Auditors, humans  |

Both live in the same `./hitl_audit.db` file. The assignment line "ghi vào audit trail" is fulfilled by `audit_events`. The checkpointer is a free bonus from using `AsyncSqliteSaver`.

### Exercise 5 — Streamlit approval UI (`app.py`)

The final assembly: wrap the LangGraph from exercises 1–4 into a **Streamlit** web UI so a reviewer can drive the whole flow from a browser instead of a terminal.

The same three confidence buckets apply, but the *human experience* changes per bucket:

| Confidence | Branch | What the reviewer sees in the UI |
|---|---|---|
| **≥ 73%** | `auto_approve` | A success card: confidence + LLM summary + a "View comment on GitHub" link. Reviewer does **nothing** — agent already posted. |
| **58–72%** | `human_approval` | The normal approval card: diff, LLM reasoning, list of comments + three buttons **Approve / Reject / Edit**. One click and the comment posts. |
| **< 58%** | `escalate` | The strong-escalation card: risk factors highlighted + a form with the LLM's specific questions. Reviewer fills answers, agent re-synthesizes, then shows the refined review for a final confirm. |

**Implemented in** `app.py` (UI) + `review_service.py` (graph I/O, testable without a browser):

- A PR URL input form → trigger the graph.
- An `interrupt()` handler that renders the right card depending on `payload["kind"]` (`approval_request` vs `escalation`).
- A resume call: `app.ainvoke(Command(resume=user_choice_or_answers), cfg)`.
- `thread_id` persisted in `st.session_state` so reloading the page or jumping back to it keeps the session.
- A small "recent sessions" sidebar driven by `audit.replay.list_threads`.

Run with:

```bash
uv run streamlit run app.py
```

### What the implementation does

- **Start:** URL form with validation → new `thread_id` per review → real Exercise 4 graph on `AsyncSqliteSaver`.
- **Approval card:** confidence + reasoning, summary, comments with severity, diff preview, **Approve & post / Reject / Edit**. Edit requires feedback; the LLM revises the review and the reviewer confirms before anything is posted.
- **Escalation card:** highlighted risk factors, one input per question (all required) → refined review → **Confirm & post / Don't post**.
- **Result:** distinguishes *posted* (with a link to the GitHub comment), *rejected*, and *posting failed* (shows the error; never a false success). Outcomes are classified by exact `final_action` value, not by prefix.
- **Recent sessions (sidebar):** from `audit_events` (`common.db.list_sessions`): PR, latest event, worst risk, event count. Clicking one restores the persisted graph state and re-shows any pending interrupt on the **same** `thread_id`. The id is also kept in `?thread=<id>` so a browser reload resumes the same session.
- **Safety:** resume payloads are validated against the pending interrupt before they reach the graph, and a resume for a thread that is no longer waiting (double click, stale tab) is refused, so a decision can not be applied — or a comment posted — twice.

## Final-action values

| `final_action` | Meaning |
|---|---|
| `auto_approved` | ≥ 73%: comment posted automatically |
| `committed` / `committed_after_escalation` | reviewer approved; comment posted |
| `rejected` | reviewer declined; nothing posted |
| `auto_commit_failed` / `commit_failed` | **posting failed** (`post_error` holds the reason) |

## Tests

```bash
uv run pytest -q          # 76 tests; all external I/O (GitHub, LLM) is mocked, temp SQLite files only
uv run ruff check .
```

The suite covers schema/routing boundaries, GitHub URL parsing and HTTP failures, approve/reject/edit/escalation flows (Ex 2–4), audit contents and "exactly once across pause/resume", worst-risk ranking, schema migration, checkpoint restoration, and the Streamlit app via `streamlit.testing` (no browser). It never posts to a real PR.

## Audit trail details

- Replay: `uv run python -m audit.replay --thread <id>`; list: `uv run python -m audit.replay --list` (shows latest event and **ranked** worst risk: high > med > low).
- **Resume correctness.** LangGraph re-runs an interrupted node from its first line on resume. Every audit row has a deterministic `event_key` (its action name) with `UNIQUE(thread_id, event_key)` + `INSERT OR IGNORE`, so the "before interrupt" row is written once. Older databases are migrated in place (`ALTER TABLE ... ADD COLUMN event_key`). Posting to GitHub happens only in `commit` / `auto_approve`, never before an `interrupt()`.
- Event `action`s: `fetch_pr, analyze, route, human_approval, human_approval_response, revise, confirm, confirm_response, escalate, escalate_answers, synthesize, commit | commit_failed, auto_approve | auto_approve_failed`. `decision` is one of `pending, approve, reject, edit, escalate, auto`.
- Rows hold counts, confidences and short reasons only — never the diff or any secret. `reviewer_id` is `GITHUB_USER`, else the login owning `GITHUB_TOKEN`, else `unknown`.
- Known limitation: because the key is the action name, a retried *identical* failure within one thread is not logged a second time.

## How the assignment maps to files

| Assignment line                                       | Where you implement it                              |
|-------------------------------------------------------|----------------------------------------------------|
| Agent reads PR, analyzes, proposes comments           | `exercise_1_confidence.py` (analyze + routing)     |
| Confidence-based routing (72% vs 58% etc.)            | `common/schemas.py` thresholds + `exercise_1`      |
| 72% → diff + reasoning → user approves → commit       | `exercise_2_hitl.py`                                |
| 58% → escalate with specific questions                | `exercise_3_escalation.py`                          |
| Audit trail + replay full session                     | `exercise_4_audit.py`, `audit/schema.sql`, `audit/replay.py` |
| Streamlit approval UI (final assembly)                | `app.py`                                            |

## Bonus challenges

1. **Time-travel.** Use `app.aget_state_history(config)` to list every checkpoint. Resume from an earlier checkpoint with a different reviewer answer and compare outcomes.
2. **Confidence calibration.** After N sessions, query `audit_events` to compute `AVG(confidence)` vs `count(human_choice = 'approve')`. Is the model over- or under-confident?
3. **Multi-reviewer fan-out.** Use the `Send` API to escalate the same questions to two reviewers (two threads) and wait for both.
4. **Auto-edit.** When the human chooses `edit`, call the LLM again to rewrite the review using `human_feedback`, then commit.

## FAQ / gotchas

**`interrupt()` raises `GraphInterrupt` instead of pausing.**
You forgot `checkpointer=...` in `.compile()`. Checkpointer is required for `interrupt()` to work.

**Graph restarts from scratch on resume.**
`thread_id` mismatch between the original `invoke` and the resuming `Command` call. Always print `thread_id` and use the same one when resuming.

**Duplicate review comments posted after resume.**
Side-effects placed *before* `interrupt()` in the same node — the node re-runs from the top after resume. Move side-effects to a downstream node (e.g. `commit`). Audit rows written before `interrupt()` are protected by the unique `event_key`.

**`post_review_comment` raises `GitHubError` (403 / 404).**
Token lacks `public_repo` scope, or repo is private and you only have `public_repo`. Re-create the PAT with the right scope.

**LLM stays overconfident on PR #2 and never escalates.**
Confidence is model output and is never forced. Exercise the branch deterministically with the mocked tests; for a live demo you may temporarily raise `ESCALATE_THRESHOLD` in `common/schemas.py`.

**`audit_events` schema is stale after a code change.**
The schema is created idempotently on first connection. To reset state completely, just delete the file:
```bash
rm hitl_audit.db
```

**Environment variables** (`.env`): `LLM_PROVIDER` (`groq` | `openrouter`), `GROQ_API_KEY` or `OPENROUTER_API_KEY` (the one matching the provider), `LLM_MODEL` (defaults: `openai/gpt-oss-120b` on Groq, `openai/gpt-4o-mini` on OpenRouter), `LLM_BASE_URL` (defaults to the provider endpoint; a host that does not match the provider is refused so a key is never sent to the wrong service), `GITHUB_TOKEN` (required), `GITHUB_USER` (optional reviewer id), `HITL_DB_PATH` (optional DB file).

**Open the SQLite file to inspect manually.**
```bash
sqlite3 hitl_audit.db "SELECT action, confidence, decision, reviewer_id FROM audit_events ORDER BY id;"
```

## Validation results

Run on 2026-10-15 (Windows, `uv`):

| Check | Result |
|---|---|
| `uv run pytest -q` | **83 passed** |
| `compileall`, `uv lock --check`, `uv run ruff check .` | clean |
| `uv run python -m audit.replay --list` / `--thread` | works; replayed a real session (below) |
| `uv run streamlit run app.py` | server starts, `/_stcore/health` returns 200; UI flows covered headlessly by `tests/test_app.py` |
| **Live GitHub** (`fetch_pr` on PR-Demo #1 and #2 with the real PAT) | works (3 and 4 files fetched) |
| Live LLM on OpenRouter (initial run) | superseded by the Groq verification below. Originally blocked: OpenRouter returned `402 Insufficient credits` for the configured key; the `:free` slugs I tried returned 404. No live confidence values were observed, so the "~65%" / "<58%" expectations for PR #1/#2 are **unverified**. |
| Real GitHub fetch + stub LLM + real SQLite, reject path (nothing posted) | passed: `fetch_pr, analyze, route, human_approval, human_approval_response, commit`, each once across the interrupt/resume; reviewer = `GITHUB_USER` |
| Live comment posting to a PR | **not performed** (needs a working LLM key). Posting, including HTTP failure → `commit_failed`, is covered by mocked tests. |

Once the OpenRouter account has credit, the live checks are the commands in each exercise section, then `uv run streamlit run app.py`.

### Groq live verification (2026-10-15)

Provider `groq`, model **`openai/gpt-oss-120b`**, `https://api.groq.com/openai/v1`. Real Groq calls and real GitHub **GET** requests; **every GitHub comment POST was stubbed in-process and a guard raised on any real `httpx` POST. No real comment was posted.** Smoke harnesses used temporary SQLite files.

| Check | Result |
|---|---|
| `get_llm().with_structured_output(PRAnalysis)` minimal request | succeeded, validated as `PRAnalysis` (function-calling mode works on this model) |
| Exercise 1, PR-Demo #1 (real command) | fetch OK, 3 files; confidence **66%** → `human_approval` |
| Exercise 1, PR-Demo #2 (real command) | fetch OK, 4 files; confidence **55%** → `escalate` |
| Exercise 3 on PR #2 (posting stubbed) | confidence 53% → escalation interrupt with 4 file-specific questions → answers → re-analysis (55%) → `final_confirmation` interrupt shown before posting → approve → `committed_after_escalation` (stubbed post) |
| Exercise 2 on PR #1 (posting stubbed) | this run scored ≥73% → `auto_approved` (stubbed post), so the approval/edit/reject path was **not** exercised live (covered by mocked tests only) |
| Exercise 4 via `review_service`, PR #1 (stubbed post) | 68% → approval interrupt → approve → `committed`; audit events `fetch_pr, analyze, route, human_approval, human_approval_response, commit`, no duplicates, same `thread_id`, reviewer `Biocuatoe`, session restored from a fresh connection |
| Exercise 4 via `review_service`, PR #2 (stubbed post) | scored ≥73% → `auto_approved` (stubbed); events `fetch_pr, analyze, route, auto_approve`, decision `auto`, no reviewer |
| `audit.replay --list` / `--thread` on the smoke DB | works |
| Streamlit headless server | starts; `/_stcore/health` → 200 |
| Streamlit script driven by `AppTest` with live Groq (stubbed post) | PR #1: approval card (65%) → "Approve & post" → success state, same thread; PR #2: auto-approve success card; sidebar listed both sessions. The **escalation card and edit flow were not exercised live** |

Confidence is model-generated and **varied between runs** on the same PR (PR #2: 55% in Exercise 1, 53% in Exercise 3, ≥73% in Exercise 4 and the UI run), so the demo PRs do not reliably land in a given branch; the expected branches above are not guaranteed. No thresholds or scores were altered.

Remaining: no real GitHub comment post was tested (intentionally). LangGraph prints a deprecation-style warning about deserializing `common.schemas.PRAnalysis` from checkpoints ("will be blocked in a future version"); it is harmless today.
