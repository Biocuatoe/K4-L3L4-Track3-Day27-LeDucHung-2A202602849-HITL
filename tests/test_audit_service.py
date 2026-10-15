"""Exercise 4 graph + service layer against a temporary SQLite file."""

import sqlite3

import pytest

import review_service as svc
from common import db
from common.github import GitHubError
from common.schemas import AuditEntry
from exercises import exercise_4_audit as ex4
from tests.conftest import PR_URL, make_analysis

QS = ["Why MD5 in storage.py?", "Is SYNC_URL https?"]


async def events(thread_id):
    return await db.replay_events(thread_id)


def actions(evs):
    return [e["action"] for e in evs]


async def test_auto_approve_audited_and_posted(install):
    rec, _ = install(ex4, [make_analysis(0.9)])
    tid = svc.new_thread_id()
    v = await svc.start_review(PR_URL, tid)
    assert v.status == "completed" and v.outcome == "posted" and len(rec.posts) == 1
    evs = await events(tid)
    assert actions(evs) == ["fetch_pr", "analyze", "route", "auto_approve"]
    auto = evs[-1]
    assert (auto["decision"], auto["reviewer_id"], auto["risk_level"]) == ("auto", None, "low")
    analyze = evs[1]
    assert analyze["confidence"] == 0.9 and "because 0.9" in analyze["reason"]
    assert evs[2]["decision"] == "auto" and "auto_approve" in evs[2]["reason"]
    assert all(e["execution_time_ms"] >= 0 for e in evs)


async def test_approve_flow_no_duplicate_events_and_stable_thread(install):
    rec, _ = install(ex4, [make_analysis(0.65)])
    tid = svc.new_thread_id()
    v = await svc.start_review(PR_URL, tid, reviewer="alice")
    assert v.status == "interrupted" and v.payload["kind"] == "approval_request"
    assert rec.posts == []
    assert actions(await events(tid)) == ["fetch_pr", "analyze", "route", "human_approval"]

    v = await svc.resume_review(tid, {"choice": "approve", "feedback": "lgtm"}, "approval_request")
    assert v.thread_id == tid and v.outcome == "posted"
    evs = await events(tid)
    # human_approval's node body ran twice, but its "pending" row exists once.
    assert actions(evs) == ["fetch_pr", "analyze", "route", "human_approval",
                            "human_approval_response", "commit"]
    pending, resp, commit = evs[3], evs[4], evs[5]
    assert pending["decision"] == "pending" and pending["reviewer_id"] is None
    assert (resp["decision"], resp["reviewer_id"], resp["reason"]) == ("approve", "alice", "lgtm")
    assert (commit["decision"], commit["reviewer_id"], commit["risk_level"]) == ("approve", "alice", "med")
    assert len(rec.posts) == 1 and v.values["comment_url"].endswith("#issuecomment-99")


async def test_reject_posts_nothing(install):
    rec, _ = install(ex4, [make_analysis(0.65)])
    tid = svc.new_thread_id()
    await svc.start_review(PR_URL, tid)
    v = await svc.resume_review(tid, {"choice": "reject", "feedback": "nah"})
    assert v.outcome == "rejected" and rec.posts == []
    evs = await events(tid)
    assert evs[-1]["action"] == "commit" and evs[-1]["decision"] == "reject"
    assert evs[-1]["reviewer_id"] == "tester"  # falls back to GITHUB_USER


async def test_edit_flow(install):
    rec, llm = install(ex4, [make_analysis(0.65), make_analysis(0.71, summary="Better")])
    tid = svc.new_thread_id()
    await svc.start_review(PR_URL, tid)
    v = await svc.resume_review(tid, {"choice": "edit", "feedback": "be nicer"})
    assert v.status == "interrupted" and v.payload["kind"] == "final_confirmation"
    assert rec.posts == [] and "be nicer" in llm.calls[-1][1]["content"]
    v = await svc.resume_review(tid, {"choice": "approve"}, "final_confirmation")
    assert v.outcome == "posted" and "Better" in rec.posts[0][1]
    acts = actions(await events(tid))
    assert acts.count("human_approval") == 1 and acts.count("confirm") == 1
    assert acts[-5:] == ["human_approval_response", "revise", "confirm", "confirm_response", "commit"]
    edit = [e for e in await events(tid) if e["action"] == "human_approval_response"][0]
    assert edit["decision"] == "edit"


async def test_escalation_flow(install):
    rec, llm = install(ex4, [make_analysis(0.4, QS), make_analysis(0.8)])
    tid = svc.new_thread_id()
    v = await svc.start_review(PR_URL, tid)
    assert v.payload["kind"] == "escalation" and v.payload["questions"] == QS

    with pytest.raises(ValueError):
        await svc.resume_review(tid, {QS[0]: "only one"})  # invalid → nothing consumed
    assert (await svc.load_session(tid)).status == "interrupted"

    v = await svc.resume_review(tid, {QS[0]: "legacy", QS[1]: "yes"}, "escalation")
    assert v.payload["kind"] == "final_confirmation" and rec.posts == []
    assert "legacy" in llm.calls[-1][1]["content"]
    v = await svc.resume_review(tid, {"choice": "approve"})
    assert v.outcome == "posted" and len(rec.posts) == 1

    evs = await events(tid)
    assert actions(evs) == ["fetch_pr", "analyze", "route", "escalate", "escalate_answers",
                            "synthesize", "confirm", "confirm_response", "commit"]
    assert evs[2]["decision"] == "escalate" and evs[2]["risk_level"] == "high"
    assert evs[3]["decision"] == "escalate" and evs[3]["reviewer_id"] is None
    assert evs[4]["reviewer_id"] == "tester"
    assert evs[5]["confidence"] == 0.8 and evs[5]["risk_level"] == "low"  # NEW confidence


async def test_post_failure_is_failure_not_success(install):
    rec, _ = install(ex4, [make_analysis(0.65)])
    rec.fail_with = GitHubError("GitHub comment post failed: HTTP 403 — forbidden")
    tid = svc.new_thread_id()
    await svc.start_review(PR_URL, tid)
    v = await svc.resume_review(tid, {"choice": "approve"})
    assert v.values["final_action"] == "commit_failed" and v.outcome == "failed"
    assert "403" in v.values["post_error"] and not v.values.get("comment_url")
    assert (await events(tid))[-1]["action"] == "commit_failed"


async def test_auto_post_failure_not_classified_as_posted(install):
    rec, _ = install(ex4, [make_analysis(0.95)])
    rec.fail_with = GitHubError("HTTP 500")
    v = await svc.start_review(PR_URL, svc.new_thread_id())
    assert v.values["final_action"] == "auto_commit_failed" and v.outcome == "failed"


async def test_stale_resume_cannot_double_post(install):
    rec, _ = install(ex4, [make_analysis(0.65)])
    tid = svc.new_thread_id()
    await svc.start_review(PR_URL, tid)
    await svc.resume_review(tid, {"choice": "approve"})
    with pytest.raises(svc.StaleResumeError):
        await svc.resume_review(tid, {"choice": "approve"})
    assert len(rec.posts) == 1


async def test_wrong_kind_resume_refused(install):
    install(ex4, [make_analysis(0.65)])
    tid = svc.new_thread_id()
    await svc.start_review(PR_URL, tid)
    with pytest.raises(svc.StaleResumeError):
        await svc.resume_review(tid, {"a": "b"}, expected_kind="escalation")


async def test_persisted_interrupt_restores_from_fresh_connection(install):
    install(ex4, [make_analysis(0.65)])
    tid = svc.new_thread_id()
    await svc.start_review(PR_URL, tid)
    v = await svc.load_session(tid)  # new AsyncSqliteSaver, as after a restart
    assert v.status == "interrupted" and v.payload["kind"] == "approval_request"
    assert v.values["pr_url"] == PR_URL
    assert (await svc.load_session("nope")).status == "empty"


async def test_malformed_url_and_duplicate_thread(install):
    install(ex4, [make_analysis(0.65)])
    with pytest.raises(ValueError):
        await svc.start_review("nonsense", svc.new_thread_id())
    tid = svc.new_thread_id()
    await svc.start_review(PR_URL, tid)
    with pytest.raises(svc.StaleResumeError):
        await svc.start_review(PR_URL, tid)


async def test_llm_failure_leaves_resumable_stalled_session(install, monkeypatch):
    _, real_llm = install(ex4, [make_analysis(0.9)])
    calls = {"n": 0}

    class Flaky:
        def with_structured_output(self, s):
            return self

        async def ainvoke(self, m):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("provider down")
            return await real_llm.ainvoke(m)

    monkeypatch.setattr(ex4, "get_llm", lambda *a, **k: Flaky())
    tid = svc.new_thread_id()
    with pytest.raises(RuntimeError):
        await svc.start_review(PR_URL, tid)
    assert (await svc.load_session(tid)).status == "stalled"
    v = await svc.continue_session(tid)
    assert v.status == "completed"


# ─── db layer ──────────────────────────────────────────────────────────────
def _entry(action="x", conf=0.5, risk="med"):
    return AuditEntry(agent_id="a", action=action, confidence=conf, risk_level=risk,
                      decision="pending", execution_time_ms=1)


async def test_event_key_dedupes_but_null_key_appends():
    kw = dict(thread_id="t", pr_url=PR_URL)
    assert await db.write_audit_event(entry=_entry(), event_key="k", **kw) is True
    assert await db.write_audit_event(entry=_entry(), event_key="k", **kw) is False
    assert await db.write_audit_event(entry=_entry(), **kw) is True
    assert await db.write_audit_event(entry=_entry(), **kw) is True
    assert len(await db.replay_events("t")) == 3


async def test_worst_risk_is_ranked_not_lexical():
    assert db.worst_risk(["low", "med"]) == "med"      # lexically "med" > "low" too
    assert db.worst_risk(["high", "med", "low"]) == "high"  # lexically "med" would win
    assert db.worst_risk([]) is None
    for tid, risks in (("a", ["low", "high", "med"]), ("b", ["low", "med"]), ("c", ["low"])):
        for r in risks:
            await db.write_audit_event(thread_id=tid, pr_url=PR_URL, entry=_entry(risk=r))
    sess = {s["thread_id"]: s for s in await db.list_sessions()}
    assert sess["a"]["worst_risk"] == "high" and sess["a"]["events"] == 3
    assert sess["b"]["worst_risk"] == "med" and sess["c"]["worst_risk"] == "low"
    assert sess["a"]["last_action"] == "x"


async def test_replay_is_chronological_and_list_newest_first():
    for i, a in enumerate(["one", "two", "three"]):
        await db.write_audit_event(thread_id="t1", pr_url=PR_URL, entry=_entry(a))
    await db.write_audit_event(thread_id="t2", pr_url=PR_URL, entry=_entry("other"))
    assert actions(await db.replay_events("t1")) == ["one", "two", "three"]
    assert [s["thread_id"] for s in await db.list_sessions()] == ["t2", "t1"]


async def test_old_schema_is_migrated(tmp_path, monkeypatch):
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.executescript("""CREATE TABLE audit_events (id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL, thread_id TEXT NOT NULL, pr_url TEXT NOT NULL, agent_id TEXT NOT NULL,
        action TEXT NOT NULL, confidence REAL NOT NULL, risk_level TEXT NOT NULL, reviewer_id TEXT,
        decision TEXT NOT NULL, reason TEXT, execution_time_ms INTEGER NOT NULL);
        INSERT INTO audit_events (timestamp,thread_id,pr_url,agent_id,action,confidence,risk_level,decision,execution_time_ms)
        VALUES ('2025-01-01','old','u','a','fetch_pr',0,'med','pending',1);""")
    con.commit(); con.close()
    monkeypatch.setenv("HITL_DB_PATH", str(path))
    assert len(await db.replay_events("old")) == 1
    assert await db.write_audit_event(thread_id="old", pr_url="u", entry=_entry(), event_key="k")
    assert not await db.write_audit_event(thread_id="old", pr_url="u", entry=_entry(), event_key="k")
    # idempotent re-init
    assert len(await db.replay_events("old")) == 2


async def test_no_secrets_or_diff_in_audit(install):
    install(ex4, [make_analysis(0.65)])
    tid = svc.new_thread_id()
    await svc.start_review(PR_URL, tid)
    await svc.resume_review(tid, {"choice": "approve"})
    blob = repr(await events(tid))
    assert "ghp_fake" not in blob and "sk-or-fake" not in blob and "print('hi')" not in blob


async def test_replay_cli_functions_run(install, capsys):
    install(ex4, [make_analysis(0.9)])
    tid = svc.new_thread_id()
    await svc.start_review(PR_URL, tid)
    from audit import replay
    await replay.list_threads()
    await replay.replay(tid)
    out = capsys.readouterr().out
    assert tid[:8] in out and "auto_approve" in out


async def test_checkpoint_roundtrip_has_no_deserialization_warning(install, caplog):
    install(ex4, [make_analysis(0.65)])
    tid = svc.new_thread_id()
    await svc.start_review(PR_URL, tid)
    with caplog.at_level("WARNING"):
        v = await svc.load_session(tid)  # reads PRAnalysis back from the SQLite checkpoint
    assert v.values["analysis"].confidence == 0.65
    assert not [r for r in caplog.records if "unregistered type" in r.getMessage()]
