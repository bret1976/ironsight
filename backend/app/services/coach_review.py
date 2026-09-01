"""Coach review workflow — make Fix score the product path, not optional.

Weakness fixed: hit/miss is assistive until a coach locks the session.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from app.services import store
from app.services.scoring import recompute_session_stats


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def review_queue(doc: dict[str, Any]) -> dict[str, Any]:
    """Shots that still need human attention, ordered for coach workflow."""
    shots = doc.get("shots") or []
    items = []
    for s in shots:
        cls = s.get("classification") or "UNKNOWN"
        conf = float(s.get("confidence") or 0)
        needs = bool(
            s.get("needs_review")
            or cls == "UNKNOWN"
            or (cls in ("HIT", "MISS") and conf < 0.45 and not s.get("human_corrected"))
        )
        if not needs and s.get("human_corrected"):
            continue
        if not needs and cls in ("HIT", "MISS", "NOT_A_SHOT") and conf >= 0.45:
            continue
        if needs or cls == "UNKNOWN":
            items.append(
                {
                    "shot_id": s.get("id"),
                    "t": s.get("timestamp_s"),
                    "classification": cls,
                    "confidence": conf,
                    "target": s.get("target_label") or s.get("target_id"),
                    "reasoning": (s.get("reasoning") or "")[:200],
                    "human_corrected": bool(s.get("human_corrected")),
                    "priority": 0 if cls == "UNKNOWN" else 1,
                }
            )
    items.sort(key=lambda x: (x["priority"], x.get("t") or 0))
    total = len([s for s in shots if s.get("classification") in ("HIT", "MISS", "UNKNOWN")])
    locked = bool(doc.get("coach_locked"))
    return {
        "session_id": doc.get("id"),
        "items": items,
        "needs_review_count": len(items),
        "total_shots": total,
        "review_complete": len(items) == 0 and total > 0,
        "coach_locked": locked,
        "coach_locked_at": doc.get("coach_locked_at"),
        "coach_locked_by": doc.get("coach_locked_by"),
        "coach_ready": locked or (len(items) == 0 and total > 0),
        "next_shot_id": items[0]["shot_id"] if items else None,
        "progress": {
            "reviewed": total - len(items),
            "remaining": len(items),
            "pct": round((total - len(items)) / total * 100, 1) if total else 0,
        },
    }


def mark_shot_reviewed(
    session_id: str,
    shot_id: int,
    *,
    classification: Optional[str] = None,
    note: Optional[str] = None,
    user_id: Optional[str] = None,
) -> dict[str, Any]:
    """Coach confirms or changes a shot — clears needs_review."""
    doc = store.load_session(session_id)
    for sh in doc.get("shots") or []:
        if int(sh.get("id", -1)) != int(shot_id):
            continue
        if classification:
            sh["classification"] = classification
        sh["human_corrected"] = True
        sh["needs_review"] = False
        sh["confidence"] = max(float(sh.get("confidence") or 0), 0.95)
        sh["reviewed_at"] = _now()
        sh["reviewed_by"] = user_id
        if note:
            sh["review_note"] = note
        break
    recompute_session_stats(doc)
    store.save_session(doc)
    return review_queue(doc)


def coach_lock(session_id: str, user_id: str, *, force: bool = False) -> dict[str, Any]:
    """Mark session as coach-verified for debrief / export badge."""
    doc = store.load_session(session_id)
    q = review_queue(doc)
    if q["needs_review_count"] > 0 and not force:
        return {
            "ok": False,
            "error": f"{q['needs_review_count']} shots still need review",
            "queue": q,
        }
    doc["coach_locked"] = True
    doc["coach_locked_at"] = _now()
    doc["coach_locked_by"] = user_id
    doc["coach_ready"] = True
    store.save_session(doc)
    q = review_queue(doc)
    return {"ok": True, "queue": q, "session_id": session_id}


def coach_unlock(session_id: str) -> dict[str, Any]:
    doc = store.load_session(session_id)
    doc["coach_locked"] = False
    doc["coach_ready"] = False
    store.save_session(doc)
    return {"ok": True, "queue": review_queue(doc)}
