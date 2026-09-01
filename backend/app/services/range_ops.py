"""Team range ops — dashboard, assignments, lanes, SLA contract.

Fixes Team thin spots: main value = range ops (not multi-mapper lab).
"""
from __future__ import annotations

import json
import secrets
from datetime import datetime, timezone
from typing import Any, Optional

from app.services import db, store


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _uid() -> str:
    return secrets.token_hex(10)


def init_range_tables() -> None:
    db.init_all()
    conn = db.connect()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS session_assignments (
            session_id TEXT NOT NULL,
            org_id TEXT NOT NULL,
            assignee_id TEXT NOT NULL,
            role_hint TEXT,
            assigned_by TEXT,
            created_at TEXT NOT NULL,
            PRIMARY KEY (session_id, assignee_id)
        );
        CREATE TABLE IF NOT EXISTS lanes (
            id TEXT PRIMARY KEY,
            org_id TEXT NOT NULL,
            name TEXT NOT NULL,
            notes TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS lane_bookings (
            id TEXT PRIMARY KEY,
            org_id TEXT NOT NULL,
            lane_id TEXT NOT NULL,
            title TEXT NOT NULL,
            starts_at TEXT NOT NULL,
            ends_at TEXT NOT NULL,
            user_id TEXT,
            session_id TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sla_acceptances (
            org_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            version TEXT NOT NULL,
            accepted_at TEXT NOT NULL,
            PRIMARY KEY (org_id, user_id, version)
        );
        CREATE TABLE IF NOT EXISTS magic_links (
            token TEXT PRIMARY KEY,
            email TEXT NOT NULL,
            user_id TEXT,
            expires_at TEXT NOT NULL,
            used INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );
        """
    )
    # job priority column
    cols = {r[1] for r in conn.execute("PRAGMA table_info(jobs)").fetchall()}
    if "priority" not in cols:
        try:
            conn.execute("ALTER TABLE jobs ADD COLUMN priority INTEGER NOT NULL DEFAULT 0")
        except Exception:
            pass
    conn.commit()
    conn.close()


def range_dashboard(org_id: str) -> dict[str, Any]:
    """Main Team value story: what needs coaching attention across the range."""
    init_range_tables()
    from app.core.config import get_settings

    sdir = get_settings().session_dir
    sessions = []
    needs_review_total = 0
    coach_ready = 0
    processing = 0
    for d in sdir.iterdir() if sdir.exists() else []:
        sp = d / "session.json"
        if not sp.exists():
            continue
        try:
            doc = json.loads(sp.read_text())
        except Exception:
            continue
        if doc.get("org_id") != org_id:
            continue
        stats = doc.get("stats") or {}
        shots = doc.get("shots") or []
        nr = sum(
            1
            for s in shots
            if s.get("needs_review")
            or s.get("classification") == "UNKNOWN"
            or (
                s.get("classification") in ("HIT", "MISS")
                and float(s.get("confidence") or 0) < 0.45
                and not s.get("human_corrected")
            )
        )
        needs_review_total += nr
        if doc.get("coach_locked") or doc.get("coach_ready"):
            coach_ready += 1
        if doc.get("status") == "processing":
            processing += 1
        sessions.append(
            {
                "id": doc.get("id"),
                "name": doc.get("name"),
                "status": doc.get("status"),
                "owner_id": doc.get("owner_id"),
                "assigned_to": doc.get("assigned_to"),
                "needs_review": nr,
                "coach_locked": bool(doc.get("coach_locked")),
                "shots": (stats.get("total_shots") or len(shots)),
                "updated_at": doc.get("updated_at"),
            }
        )
    sessions.sort(key=lambda x: x.get("updated_at") or "", reverse=True)

    conn = db.connect()
    assigns = conn.execute(
        "SELECT * FROM session_assignments WHERE org_id = ? ORDER BY created_at DESC LIMIT 100",
        (org_id,),
    ).fetchall()
    lanes = conn.execute("SELECT * FROM lanes WHERE org_id = ?", (org_id,)).fetchall()
    bookings = conn.execute(
        "SELECT * FROM lane_bookings WHERE org_id = ? ORDER BY starts_at DESC LIMIT 50",
        (org_id,),
    ).fetchall()
    conn.close()

    return {
        "org_id": org_id,
        "summary": {
            "sessions": len(sessions),
            "needs_review_shots": needs_review_total,
            "coach_ready_sessions": coach_ready,
            "processing": processing,
            "lanes": len(lanes),
            "upcoming_bookings": len(bookings),
        },
        "sessions": sessions[:50],
        "assignments": [dict(a) for a in assigns],
        "lanes": [dict(l) for l in lanes],
        "bookings": [dict(b) for b in bookings],
        "value_story": "Review queue + shared library + lanes — multi-mapper is optional lab only.",
    }


def assign_session(
    org_id: str,
    session_id: str,
    assignee_id: str,
    assigned_by: str,
    role_hint: str = "coach",
) -> dict[str, Any]:
    init_range_tables()
    doc = store.load_session(session_id)
    doc["assigned_to"] = assignee_id
    doc["org_id"] = org_id
    store.save_session(doc)
    conn = db.connect()
    conn.execute(
        "INSERT OR REPLACE INTO session_assignments "
        "(session_id, org_id, assignee_id, role_hint, assigned_by, created_at) "
        "VALUES (?,?,?,?,?,?)",
        (session_id, org_id, assignee_id, role_hint, assigned_by, _now()),
    )
    conn.commit()
    conn.close()
    return {"ok": True, "session_id": session_id, "assignee_id": assignee_id}


def create_lane(org_id: str, name: str, notes: str = "") -> dict[str, Any]:
    init_range_tables()
    lid = _uid()
    conn = db.connect()
    conn.execute(
        "INSERT INTO lanes (id, org_id, name, notes, created_at) VALUES (?,?,?,?,?)",
        (lid, org_id, name.strip() or "Lane", notes, _now()),
    )
    conn.commit()
    conn.close()
    return {"id": lid, "org_id": org_id, "name": name, "notes": notes}


def book_lane(
    org_id: str,
    lane_id: str,
    title: str,
    starts_at: str,
    ends_at: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    init_range_tables()
    bid = _uid()
    conn = db.connect()
    conn.execute(
        "INSERT INTO lane_bookings "
        "(id, org_id, lane_id, title, starts_at, ends_at, user_id, session_id, created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (bid, org_id, lane_id, title, starts_at, ends_at, user_id, session_id, _now()),
    )
    conn.commit()
    conn.close()
    return {"id": bid, "lane_id": lane_id, "title": title, "starts_at": starts_at, "ends_at": ends_at}


def accept_sla(org_id: str, user_id: str, version: str = "2026-07") -> dict[str, Any]:
    init_range_tables()
    conn = db.connect()
    conn.execute(
        "INSERT OR REPLACE INTO sla_acceptances (org_id, user_id, version, accepted_at) VALUES (?,?,?,?)",
        (org_id, user_id, version, _now()),
    )
    conn.commit()
    conn.close()
    return {"ok": True, "version": version, "accepted_at": _now()}


def sla_status(org_id: str) -> dict[str, Any]:
    init_range_tables()
    conn = db.connect()
    rows = conn.execute(
        "SELECT * FROM sla_acceptances WHERE org_id = ?", (org_id,)
    ).fetchall()
    conn.close()
    from app.services import sla as sla_metrics

    metrics = sla_metrics.summary(hours=24 * 7)
    return {
        "contract_version": "2026-07",
        "contract_url": "/api/legal/sla",
        "acceptances": [dict(r) for r in rows],
        "live_metrics": metrics,
        "targets": {
            "pipeline_success_rate_pct": 95,
            "support_response_hours": 48,
            "uptime_target_pct": 99.0,
        },
    }


def create_magic_link(email: str) -> dict[str, Any]:
    """Passwordless login (enterprise-lite SSO alternative)."""
    from app.services import mailer, users

    init_range_tables()
    email = email.strip().lower()
    u = users.get_user_by_email(email)
    token = secrets.token_urlsafe(32)
    from datetime import timedelta

    exp = (datetime.now(timezone.utc) + timedelta(minutes=20)).isoformat().replace("+00:00", "Z")
    conn = db.connect()
    conn.execute(
        "INSERT INTO magic_links (token, email, user_id, expires_at, used, created_at) VALUES (?,?,?,?,0,?)",
        (token, email, u["id"] if u else None, exp, _now()),
    )
    conn.commit()
    conn.close()
    from app.core.config import get_settings

    s = get_settings()
    link = f"{s.public_url.rstrip('/')}/?magic={token}"
    mailer.send_email(
        email,
        "Your IronSight sign-in link",
        f"Sign in (valid 20 minutes):\n\n{link}\n\nIf you didn't request this, ignore.",
    )
    out = {"ok": True, "sent": True}
    if s.environment != "production":
        out["token"] = token
        out["link"] = link
    return out


def consume_magic_link(token: str) -> dict[str, Any]:
    from app.services import users

    init_range_tables()
    conn = db.connect()
    row = conn.execute("SELECT * FROM magic_links WHERE token = ?", (token,)).fetchone()
    if not row or row["used"]:
        conn.close()
        raise ValueError("Invalid or used link")
    if row["expires_at"] < _now():
        conn.close()
        raise ValueError("Link expired")
    email = row["email"]
    uid = row["user_id"]
    conn.execute("UPDATE magic_links SET used = 1 WHERE token = ?", (token,))
    conn.commit()
    conn.close()
    u = users.get_user(uid) if uid else users.get_user_by_email(email)
    if not u:
        # auto-register free user for magic link
        try:
            u = users.register(email, secrets.token_urlsafe(16), email.split("@")[0])
        except ValueError:
            u = users.get_user_by_email(email)
    if not u:
        raise ValueError("Could not resolve user")
    token_jwt = users.create_token(u["id"])
    return {"token": token_jwt, "user": users.public_user(u)}
