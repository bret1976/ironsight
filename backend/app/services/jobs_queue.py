"""Durable multi-tenant job queue with leasing (scale workers)."""
from __future__ import annotations

import json
import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from app.services import audit, db, sla

log = logging.getLogger("ironsight.jobs")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _uid() -> str:
    return secrets.token_hex(12)


def enqueue(
    user_id: str,
    session_id: str,
    kind: str = "pipeline",
    *,
    org_id: Optional[str] = None,
    payload: Optional[dict] = None,
    priority: int = 0,
) -> dict[str, Any]:
    db.init_all()
    # Ensure priority column exists
    try:
        from app.services.range_ops import init_range_tables

        init_range_tables()
    except Exception:
        pass
    jid = _uid()
    now = _now()
    conn = db.connect()
    try:
        conn.execute(
            "INSERT INTO jobs (id, user_id, org_id, session_id, kind, status, progress, message, "
            "payload, attempts, max_attempts, priority, created_at, updated_at) "
            "VALUES (?,?,?,?,?,'queued',0,'Queued',?,0,3,?,?,?)",
            (
                jid,
                user_id,
                org_id,
                session_id,
                kind,
                json.dumps(payload or {}),
                int(priority),
                now,
                now,
            ),
        )
    except Exception:
        conn.execute(
            "INSERT INTO jobs (id, user_id, org_id, session_id, kind, status, progress, message, "
            "payload, attempts, max_attempts, created_at, updated_at) "
            "VALUES (?,?,?,?,?,'queued',0,'Queued',?,0,3,?,?)",
            (
                jid,
                user_id,
                org_id,
                session_id,
                kind,
                json.dumps(payload or {}),
                now,
                now,
            ),
        )
    conn.commit()
    conn.close()
    audit.log(
        "job.enqueue",
        actor_id=user_id,
        resource_type="job",
        resource_id=jid,
        detail={"session_id": session_id, "kind": kind, "priority": priority},
    )
    return get_job(jid) or {}


def get_job(job_id: str) -> Optional[dict[str, Any]]:
    db.init_all()
    conn = db.connect()
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_job(job_id: str, **fields: Any) -> None:
    allowed = {
        "status",
        "progress",
        "message",
        "error",
        "attempts",
        "lease_owner",
        "lease_until",
        "started_at",
        "finished_at",
    }
    sets = []
    vals = []
    for k, v in fields.items():
        if k in allowed:
            sets.append(f"{k} = ?")
            vals.append(v)
    if not sets:
        return
    sets.append("updated_at = ?")
    vals.append(_now())
    vals.append(job_id)
    conn = db.connect()
    conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?", vals)
    conn.commit()
    conn.close()


def list_jobs(user_id: str, limit: int = 50) -> list[dict[str, Any]]:
    db.init_all()
    conn = db.connect()
    rows = conn.execute(
        "SELECT * FROM jobs WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def lease_next(worker_id: str, *, kinds: Optional[list[str]] = None, lease_s: int = 600) -> Optional[dict[str, Any]]:
    """Atomically claim one queued job for a worker."""
    db.init_all()
    now = datetime.now(timezone.utc)
    now_s = now.isoformat().replace("+00:00", "Z")
    lease_until = (now + timedelta(seconds=lease_s)).isoformat().replace("+00:00", "Z")
    conn = db.connect()
    # reclaim expired leases
    conn.execute(
        "UPDATE jobs SET status = 'queued', lease_owner = NULL, lease_until = NULL, updated_at = ? "
        "WHERE status = 'running' AND lease_until IS NOT NULL AND lease_until < ?",
        (now_s, now_s),
    )
    # Higher priority first (Team plan), then oldest
    if kinds:
        placeholders = ",".join("?" * len(kinds))
        try:
            row = conn.execute(
                f"SELECT * FROM jobs WHERE status = 'queued' AND kind IN ({placeholders}) "
                "ORDER BY COALESCE(priority,0) DESC, created_at ASC LIMIT 1",
                kinds,
            ).fetchone()
        except Exception:
            row = conn.execute(
                f"SELECT * FROM jobs WHERE status = 'queued' AND kind IN ({placeholders}) "
                "ORDER BY created_at ASC LIMIT 1",
                kinds,
            ).fetchone()
    else:
        try:
            row = conn.execute(
                "SELECT * FROM jobs WHERE status = 'queued' "
                "ORDER BY COALESCE(priority,0) DESC, created_at ASC LIMIT 1"
            ).fetchone()
        except Exception:
            row = conn.execute(
                "SELECT * FROM jobs WHERE status = 'queued' ORDER BY created_at ASC LIMIT 1"
            ).fetchone()
    if not row:
        conn.commit()
        conn.close()
        return None
    job = dict(row)
    if int(job.get("attempts") or 0) >= int(job.get("max_attempts") or 3):
        conn.execute(
            "UPDATE jobs SET status = 'failed', error = ?, updated_at = ?, finished_at = ? WHERE id = ?",
            ("Max attempts exceeded", now_s, now_s, job["id"]),
        )
        conn.commit()
        conn.close()
        return None
    conn.execute(
        "UPDATE jobs SET status = 'running', lease_owner = ?, lease_until = ?, attempts = attempts + 1, "
        "started_at = COALESCE(started_at, ?), updated_at = ?, message = 'Running' WHERE id = ? AND status = 'queued'",
        (worker_id, lease_until, now_s, now_s, job["id"]),
    )
    if conn.total_changes == 0:
        conn.commit()
        conn.close()
        return None
    conn.commit()
    conn.close()
    return get_job(job["id"])


def complete(job_id: str, *, ok: bool, message: str = "", error: Optional[str] = None) -> None:
    update_job(
        job_id,
        status="done" if ok else "failed",
        progress=1.0 if ok else 0.0,
        message=message or ("Done" if ok else "Failed"),
        error=error,
        finished_at=_now(),
        lease_owner=None,
        lease_until=None,
    )
    job = get_job(job_id)
    if job:
        sla.record(
            "pipeline_done" if ok else "pipeline_failed",
            ok=ok,
            session_id=job.get("session_id"),
            user_id=job.get("user_id"),
            detail=error or message,
        )


def queue_stats() -> dict[str, Any]:
    db.init_all()
    conn = db.connect()
    rows = conn.execute(
        "SELECT status, COUNT(*) as n FROM jobs GROUP BY status"
    ).fetchall()
    conn.close()
    return {r["status"]: r["n"] for r in rows}
