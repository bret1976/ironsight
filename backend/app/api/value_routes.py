"""Routes that close Pro/Team value gaps: coach review, 3D quality, range ops."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import assert_session_owner, require_user
from app.services import coach_review, media_quality, orgs, range_ops, store, users

router = APIRouter(tags=["value"])


class ReviewShotBody(BaseModel):
    classification: Optional[str] = None
    note: Optional[str] = None


class LockBody(BaseModel):
    force: bool = False


class AssignBody(BaseModel):
    assignee_id: str
    role_hint: str = "coach"


class LaneBody(BaseModel):
    name: str
    notes: str = ""


class BookingBody(BaseModel):
    lane_id: str
    title: str
    starts_at: str
    ends_at: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class MagicBody(BaseModel):
    email: str


class MagicConsume(BaseModel):
    token: str


# ── Coach review (Pro weakness: Fix score must be the path) ───────────

@router.get("/sessions/{session_id}/review-queue")
async def get_review_queue(session_id: str, user: dict = Depends(require_user)):
    try:
        doc = store.load_session(session_id)
    except FileNotFoundError:
        raise HTTPException(404, "Session not found")
    assert_session_owner(doc, user)
    return coach_review.review_queue(doc)


@router.post("/sessions/{session_id}/review/{shot_id}")
async def review_shot(
    session_id: str,
    shot_id: int,
    body: ReviewShotBody,
    user: dict = Depends(require_user),
):
    try:
        doc = store.load_session(session_id)
    except FileNotFoundError:
        raise HTTPException(404, "Session not found")
    assert_session_owner(doc, user, write=True)
    if doc.get("public_demo"):
        raise HTTPException(400, "Demo is read-only")
    q = coach_review.mark_shot_reviewed(
        session_id,
        shot_id,
        classification=body.classification,
        note=body.note,
        user_id=user.get("id"),
    )
    return {"ok": True, "queue": q}


@router.post("/sessions/{session_id}/coach-lock")
async def lock_session(session_id: str, body: LockBody = LockBody(), user: dict = Depends(require_user)):
    try:
        doc = store.load_session(session_id)
    except FileNotFoundError:
        raise HTTPException(404, "Session not found")
    assert_session_owner(doc, user, write=True)
    result = coach_review.coach_lock(session_id, user.get("id") or "user", force=body.force)
    if not result.get("ok"):
        raise HTTPException(400, result.get("error") or "Cannot lock")
    return result


@router.post("/sessions/{session_id}/coach-unlock")
async def unlock_session(session_id: str, user: dict = Depends(require_user)):
    try:
        doc = store.load_session(session_id)
    except FileNotFoundError:
        raise HTTPException(404, "Session not found")
    assert_session_owner(doc, user, write=True)
    return coach_review.coach_unlock(session_id)


# ── 3D readiness (Pro weakness: quality depends on video) ─────────────

@router.get("/sessions/{session_id}/quality")
async def session_quality(session_id: str, user: dict = Depends(require_user)):
    try:
        doc = store.load_session(session_id)
    except FileNotFoundError:
        raise HTTPException(404, "Session not found")
    assert_session_owner(doc, user)
    report = media_quality.assess_session(doc)
    # persist for UI
    doc["media_quality"] = {
        "grade": report.get("grade"),
        "score_0_100": report.get("score_0_100"),
        "ready_for_3d": report.get("ready_for_3d"),
        "summary": report.get("summary"),
        "tips": report.get("tips"),
        "warnings": report.get("warnings"),
    }
    try:
        store.save_session(doc)
    except Exception:
        pass
    return report


# ── Team range ops ────────────────────────────────────────────────────

@router.get("/orgs/{org_id}/dashboard")
async def org_dashboard(org_id: str, user: dict = Depends(require_user)):
    try:
        orgs.assert_member(org_id, user["id"])
    except PermissionError as e:
        raise HTTPException(403, str(e))
    u = users.get_user(user["id"]) or user
    plan = users.PLANS.get(u.get("plan") or "free", users.PLANS["free"])
    if not plan.get("range_dashboard") and not u.get("is_admin"):
        # Pro orgs still get a light dashboard
        pass
    return range_ops.range_dashboard(org_id)


@router.post("/orgs/{org_id}/sessions/{session_id}/assign")
async def assign(org_id: str, session_id: str, body: AssignBody, user: dict = Depends(require_user)):
    try:
        orgs.assert_member(org_id, user["id"], roles={"owner", "admin", "coach", "member"})
    except PermissionError as e:
        raise HTTPException(403, str(e))
    try:
        return range_ops.assign_session(
            org_id, session_id, body.assignee_id, user["id"], body.role_hint
        )
    except FileNotFoundError:
        raise HTTPException(404, "Session not found")


@router.post("/orgs/{org_id}/lanes")
async def add_lane(org_id: str, body: LaneBody, user: dict = Depends(require_user)):
    try:
        orgs.assert_member(org_id, user["id"], roles={"owner", "admin"})
    except PermissionError as e:
        raise HTTPException(403, str(e))
    u = users.get_user(user["id"]) or {}
    plan = users.PLANS.get(u.get("plan") or "free", {})
    if not plan.get("lanes") and not u.get("is_admin"):
        raise HTTPException(402, "Lane scheduling requires Team plan")
    return range_ops.create_lane(org_id, body.name, body.notes)


@router.post("/orgs/{org_id}/bookings")
async def add_booking(org_id: str, body: BookingBody, user: dict = Depends(require_user)):
    try:
        orgs.assert_member(org_id, user["id"])
    except PermissionError as e:
        raise HTTPException(403, str(e))
    u = users.get_user(user["id"]) or {}
    plan = users.PLANS.get(u.get("plan") or "free", {})
    if not plan.get("lanes") and not u.get("is_admin"):
        raise HTTPException(402, "Lane scheduling requires Team plan")
    return range_ops.book_lane(
        org_id,
        body.lane_id,
        body.title,
        body.starts_at,
        body.ends_at,
        user_id=body.user_id or user["id"],
        session_id=body.session_id,
    )


@router.get("/orgs/{org_id}/sla")
async def org_sla(org_id: str, user: dict = Depends(require_user)):
    try:
        orgs.assert_member(org_id, user["id"])
    except PermissionError as e:
        raise HTTPException(403, str(e))
    return range_ops.sla_status(org_id)


@router.post("/orgs/{org_id}/sla/accept")
async def accept_sla(org_id: str, user: dict = Depends(require_user)):
    try:
        orgs.assert_member(org_id, user["id"], roles={"owner", "admin"})
    except PermissionError as e:
        raise HTTPException(403, str(e))
    return range_ops.accept_sla(org_id, user["id"])


# ── Magic link (enterprise-lite SSO alternative) ──────────────────────

@router.post("/auth/magic-link")
async def magic_link(body: MagicBody):
    return range_ops.create_magic_link(body.email)


@router.post("/auth/magic-link/consume")
async def magic_consume(body: MagicConsume):
    try:
        return range_ops.consume_magic_link(body.token)
    except ValueError as e:
        raise HTTPException(400, str(e))
