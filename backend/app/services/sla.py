"""SLA / reliability metrics for scale readiness."""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from app.services import db


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def record(
    kind: str,
    *,
    ok: bool = True,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    duration_ms: Optional[float] = None,
    detail: Optional[str] = None,
) -> None:
    db.init_all()
    conn = db.connect()
    conn.execute(
        "INSERT INTO sla_events (kind, session_id, user_id, ok, duration_ms, detail, created_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (kind, session_id, user_id, 1 if ok else 0, duration_ms, detail, _now()),
    )
    conn.commit()
    conn.close()


def summary(hours: int = 24) -> dict[str, Any]:
    db.init_all()
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat().replace("+00:00", "Z")
    conn = db.connect()
    rows = conn.execute(
        "SELECT kind, ok, duration_ms FROM sla_events WHERE created_at >= ?",
        (since,),
    ).fetchall()
    conn.close()
    by_kind: dict[str, dict[str, Any]] = {}
    for r in rows:
        k = r["kind"]
        bucket = by_kind.setdefault(k, {"count": 0, "ok": 0, "fail": 0, "durations": []})
        bucket["count"] += 1
        if r["ok"]:
            bucket["ok"] += 1
        else:
            bucket["fail"] += 1
        if r["duration_ms"] is not None:
            bucket["durations"].append(float(r["duration_ms"]))

    out = {}
    for k, b in by_kind.items():
        durs = sorted(b["durations"])
        p95 = durs[int(len(durs) * 0.95) - 1] if durs else None
        out[k] = {
            "count": b["count"],
            "ok": b["ok"],
            "fail": b["fail"],
            "success_rate": round(b["ok"] / b["count"] * 100, 2) if b["count"] else None,
            "p95_ms": round(p95, 1) if p95 is not None else None,
        }

    pipe = out.get("pipeline_done") or out.get("pipeline")
    health = {
        "window_hours": hours,
        "events": out,
        "pipeline_success_rate": (pipe or {}).get("success_rate"),
        "pipeline_p95_ms": (pipe or {}).get("p95_ms"),
        "status": "healthy",
        "sample_size": sum(b["count"] for b in by_kind.values()) if by_kind else 0,
    }
    if health["sample_size"] == 0:
        health["status"] = "no_data"
        health["note"] = "No pipeline events in window yet — status is unknown, not green."
    rate = health["pipeline_success_rate"]
    if rate is not None and rate < 90:
        health["status"] = "degraded"
    if rate is not None and rate < 70:
        health["status"] = "critical"
    return health


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()

    def ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000.0
