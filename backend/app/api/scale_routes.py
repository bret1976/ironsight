"""Scale APIs: orgs/seats, SLA, admin, support, onboarding, webhooks."""
from __future__ import annotations

import secrets
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from app.api.deps import require_user
from app.core.config import ROOT, get_settings
from app.services import audit, jobs_queue, orgs, sla, users
from app.services import store

router = APIRouter(tags=["scale"])


def _require_admin(user: dict) -> dict:
    if user.get("_anonymous"):
        raise HTTPException(401, "Admin required")
    u = users.get_user(user["id"])
    s = get_settings()
    if u and u.get("is_admin"):
        return u
    # bootstrap via ADMIN_API_KEY header handled elsewhere; allow first admin email
    raise HTTPException(403, "Admin only")


class OrgCreate(BaseModel):
    name: str = "My Range Team"


class InviteBody(BaseModel):
    email: str
    role: str = "member"


class AcceptInvite(BaseModel):
    token: str


class SupportBody(BaseModel):
    email: str = ""
    subject: str = Field(min_length=3)
    body: str = Field(min_length=10)


class WebhookBody(BaseModel):
    url: str
    events: list[str] = ["session.ready", "job.failed"]


class AdminSetPlan(BaseModel):
    email: str
    plan: str = "pro"


# ── Orgs / seats ──────────────────────────────────────────────────────

@router.post("/orgs")
async def create_org(body: OrgCreate, user: dict = Depends(require_user)):
    if user.get("_anonymous"):
        raise HTTPException(401, "Register to create a team")
    u = users.get_user(user["id"])
    plan = (u or {}).get("plan") or "free"
    # Pro = 3 seats (coach collab); Team = 10 seats (range)
    if plan not in ("pro", "team") and not (u or {}).get("is_admin"):
        raise HTTPException(
            402,
            "Organizations need Pro (3 seats) or Team (10 seats). Upgrade to collaborate.",
        )
    try:
        org = orgs.create_org(user["id"], body.name, plan=plan if plan in ("pro", "team") else "team")
        return {"org": org}
    except Exception as e:
        raise HTTPException(400, str(e))


@router.get("/orgs")
async def my_orgs(user: dict = Depends(require_user)):
    if user.get("_anonymous"):
        return {"orgs": []}
    return {"orgs": orgs.user_orgs(user["id"])}


@router.get("/orgs/{org_id}")
async def get_org(org_id: str, user: dict = Depends(require_user)):
    try:
        orgs.assert_member(org_id, user["id"])
    except PermissionError as e:
        raise HTTPException(403, str(e))
    o = orgs.get_org(org_id)
    if not o:
        raise HTTPException(404, "Org not found")
    return o


@router.post("/orgs/{org_id}/invites")
async def invite_member(org_id: str, body: InviteBody, user: dict = Depends(require_user)):
    try:
        inv = orgs.invite(org_id, body.email, user["id"], role=body.role)
    except PermissionError as e:
        raise HTTPException(403, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    s = get_settings()
    link = f"{s.public_url.rstrip('/')}/?invite={inv['token']}"
    from app.services import mailer

    mailer.send_email(
        body.email,
        "You're invited to IronSight Team",
        f"Join the team:\n\n{link}\n\nCreate an account with this email first if needed.",
    )
    return {"invite": inv, "link": link}


@router.post("/orgs/accept-invite")
async def accept_invite(body: AcceptInvite, user: dict = Depends(require_user)):
    if user.get("_anonymous"):
        raise HTTPException(401, "Sign in to accept invite")
    try:
        org = orgs.accept_invite(body.token, user["id"])
        return {"org": org}
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.delete("/orgs/{org_id}/members/{member_id}")
async def remove_member(org_id: str, member_id: str, user: dict = Depends(require_user)):
    try:
        orgs.remove_member(org_id, member_id, user["id"])
        return {"ok": True}
    except (PermissionError, ValueError) as e:
        raise HTTPException(400, str(e))


# ── Onboarding demo ───────────────────────────────────────────────────

@router.post("/onboarding/demo-session")
async def clone_demo(user: dict = Depends(require_user)):
    """Attach the public demo session for this user (read-only shared library).

    If the host has no baked demo session yet, create a private empty session
    for the user so onboarding never hard-fails in production.
    """
    s = get_settings()
    demo_id = s.demo_session_id
    try:
        doc = store.load_session(demo_id)
    except FileNotFoundError:
        # No shared demo library on this deploy — start a clean personal session.
        created = store.create_session(
            name="My first range session",
            owner_id=None if user.get("_anonymous") else user.get("id"),
        )
        audit.log(
            "onboarding.fresh_session",
            actor_id=user.get("id"),
            resource_type="session",
            resource_id=created.get("id"),
        )
        return {
            "session_id": created.get("id"),
            "name": created.get("name"),
            "status": created.get("status"),
            "stats": created.get("stats") or {},
            "note": (
                "No shared demo library on this host yet. "
                "Created a private session — upload dual-cam range video and Run Pipeline."
            ),
            "fallback": True,
        }
    # Ensure public flag
    if not doc.get("public_demo"):
        doc["public_demo"] = True
        store.save_session(doc)
    audit.log("onboarding.demo", actor_id=user.get("id"), resource_type="session", resource_id=demo_id)
    return {
        "session_id": demo_id,
        "name": doc.get("name"),
        "status": doc.get("status"),
        "stats": doc.get("stats"),
        "note": "Shared demo — open in Command Center. Upload your own dual-cam for a private session.",
        "fallback": False,
    }


# ── Support ───────────────────────────────────────────────────────────

@router.post("/support")
async def support_ticket(body: SupportBody, user: dict = Depends(require_user)):
    from app.services import db

    db.init_all()
    tid = secrets.token_hex(8)
    email = body.email or user.get("email") or "unknown"
    now = users._now() if hasattr(users, "_now") else __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
    conn = db.connect()
    conn.execute(
        "INSERT INTO support_tickets (id, user_id, email, subject, body, status, created_at) "
        "VALUES (?,?,?,?,?,'open',?)",
        (tid, None if user.get("_anonymous") else user.get("id"), email, body.subject, body.body, now),
    )
    conn.commit()
    conn.close()
    from app.services import mailer

    s = get_settings()
    mailer.send_email(
        s.support_email or "support@ironsight.local",
        f"[IronSight support] {body.subject}",
        f"From: {email}\nUser: {user.get('id')}\n\n{body.body}",
    )
    audit.log("support.ticket", actor_id=user.get("id"), resource_type="ticket", resource_id=tid)
    return {"ok": True, "ticket_id": tid}


# ── SLA ───────────────────────────────────────────────────────────────

@router.get("/sla")
async def sla_status(hours: int = 24):
    return sla.summary(hours=min(hours, 168))


@router.get("/queue/stats")
async def queue_stats(user: dict = Depends(require_user)):
    return {"queue": jobs_queue.queue_stats()}


# ── Admin ─────────────────────────────────────────────────────────────

@router.get("/admin/users")
async def admin_users(request: Request, user: dict = Depends(require_user)):
    _admin_gate(request, user)
    return {"users": users.list_users(200)}


@router.post("/admin/set-plan")
async def admin_set_plan(body: AdminSetPlan, request: Request, user: dict = Depends(require_user)):
    _admin_gate(request, user)
    u = users.get_user_by_email(body.email)
    if not u:
        raise HTTPException(404, "User not found")
    if body.plan not in users.PLANS:
        raise HTTPException(400, "Unknown plan")
    updated = users.update_user(u["id"], plan=body.plan)
    audit.log(
        "admin.set_plan",
        actor_id=user.get("id"),
        resource_type="user",
        resource_id=u["id"],
        detail={"plan": body.plan},
    )
    return {"user": users.public_user(updated)}


@router.get("/admin/audit")
async def admin_audit(request: Request, user: dict = Depends(require_user), limit: int = 100):
    _admin_gate(request, user)
    return {"events": audit.list_recent(limit)}


@router.get("/admin/tickets")
async def admin_tickets(request: Request, user: dict = Depends(require_user)):
    _admin_gate(request, user)
    from app.services import db

    db.init_all()
    conn = db.connect()
    rows = conn.execute(
        "SELECT * FROM support_tickets ORDER BY created_at DESC LIMIT 100"
    ).fetchall()
    conn.close()
    return {"tickets": [dict(r) for r in rows]}


def _admin_gate(request: Request, user: dict) -> None:
    s = get_settings()
    key = request.headers.get("x-admin-key") or request.query_params.get("admin_key")
    if s.admin_api_key and key == s.admin_api_key:
        return
    u = users.get_user(user["id"]) if not user.get("_anonymous") else None
    if u and u.get("is_admin"):
        return
    raise HTTPException(403, "Admin only — set is_admin or X-Admin-Key")


# ── Legal (static) ────────────────────────────────────────────────────

@router.get("/legal/{doc}")
async def legal_doc(doc: str, format: str = "html"):
    """Pretty HTML by default; ?format=md for raw markdown."""
    import html as html_mod
    import re

    from fastapi.responses import HTMLResponse

    allowed = {"terms", "privacy", "refund", "sla"}
    if doc not in allowed:
        raise HTTPException(404, "Unknown document")
    path = ROOT / "legal" / f"{doc}.md"
    if not path.exists():
        raise HTTPException(404, "Document missing")
    raw = path.read_text()
    if (format or "html").lower() in ("md", "markdown", "text"):
        return PlainTextResponse(raw, media_type="text/markdown; charset=utf-8")
    # Minimal markdown → HTML for readable legal pages
    body = html_mod.escape(raw)
    body = re.sub(r"^# (.+)$", r"<h1>\1</h1>", body, flags=re.M)
    body = re.sub(r"^## (.+)$", r"<h2>\1</h2>", body, flags=re.M)
    body = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", body)
    body = re.sub(r"\n\n+", "</p><p>", body)
    page = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"/>
<title>IronSight · {html_mod.escape(doc.title())}</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:720px;margin:40px auto;padding:0 20px;
background:#0b0f14;color:#e8eefc;line-height:1.55}}
h1,h2{{color:#5eead4}} a{{color:#a78bfa}}
</style></head><body><p>{body}</p>
<p><a href="/">← Back to IronSight</a></p></body></html>"""
    return HTMLResponse(page)
