"""Audit log for multi-tenant scale + support forensics."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from app.services import db


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def log(
    action: str,
    *,
    actor_id: Optional[str] = None,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    detail: Optional[dict[str, Any]] = None,
) -> None:
    db.init_all()
    conn = db.connect()
    conn.execute(
        "INSERT INTO audit_log (actor_id, action, resource_type, resource_id, detail, created_at) "
        "VALUES (?,?,?,?,?,?)",
        (
            actor_id,
            action,
            resource_type,
            resource_id,
            json.dumps(detail) if detail else None,
            _now(),
        ),
    )
    conn.commit()
    conn.close()


def list_recent(limit: int = 100) -> list[dict[str, Any]]:
    db.init_all()
    conn = db.connect()
    rows = conn.execute(
        "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
