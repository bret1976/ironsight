"""Auth + billing + plans for production SaaS."""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, Field

from app.api.deps import optional_user, require_user
from app.core.config import get_settings
from app.services import billing, users

router = APIRouter(tags=["auth"])


class RegisterBody(BaseModel):
    email: str
    password: str = Field(min_length=8)
    name: str = ""


class LoginBody(BaseModel):
    email: str
    password: str


class CheckoutBody(BaseModel):
    plan: str = "pro"


class ForgotBody(BaseModel):
    email: str


class ResetBody(BaseModel):
    token: str
    password: str = Field(min_length=8)


class ChangePasswordBody(BaseModel):
    old_password: str
    new_password: str = Field(min_length=8)


@router.get("/plans")
async def list_plans():
    s = get_settings()
    return {
        "plans": list(users.PLANS.values()),
        "stripe_configured": billing.stripe_configured(),
        "publishable_key": s.stripe_publishable_key or None,
    }


@router.post("/auth/register")
async def register(body: RegisterBody):
    users.init_db()
    try:
        u = users.register(body.email, body.password, body.name)
    except ValueError as e:
        raise HTTPException(400, str(e))
    token = users.create_token(u["id"])
    return {"token": token, "user": users.public_user(u)}


@router.post("/auth/login")
async def login(body: LoginBody):
    users.init_db()
    u = users.authenticate(body.email, body.password)
    if not u:
        raise HTTPException(401, "Invalid email or password")
    token = users.create_token(u["id"])
    return {"token": token, "user": users.public_user(u)}


@router.get("/auth/me")
async def me(user: dict = Depends(require_user)):
    if user.get("_anonymous"):
        return {"user": users.public_user({**user, "plan": "team"}), "anonymous": True}
    u = users.get_user(user["id"])
    if not u:
        raise HTTPException(401, "User not found")
    return {"user": users.public_user(u), "anonymous": False}


@router.post("/auth/forgot-password")
async def forgot_password(body: ForgotBody):
    users.init_db()
    result = users.create_password_reset(body.email)
    # Always 200 to avoid email enumeration
    return {
        "ok": True,
        "message": "If that email is registered, a reset link was sent (or written to mail outbox).",
        **{k: v for k, v in (result or {}).items() if k in ("reset_token", "link") and v},
    }


@router.post("/auth/reset-password")
async def reset_password(body: ResetBody):
    try:
        users.reset_password(body.token, body.password)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True, "message": "Password updated. Sign in."}


@router.post("/auth/change-password")
async def change_password(body: ChangePasswordBody, user: dict = Depends(require_user)):
    if user.get("_anonymous"):
        raise HTTPException(401, "Sign in required")
    try:
        users.change_password(user["id"], body.old_password, body.new_password)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


@router.delete("/auth/account")
async def delete_account(user: dict = Depends(require_user)):
    if user.get("_anonymous"):
        raise HTTPException(401, "Sign in required")
    from app.services import audit

    audit.log("user.delete", actor_id=user["id"], resource_type="user", resource_id=user["id"])
    users.delete_user(user["id"])
    return {"ok": True}


@router.post("/billing/checkout")
async def checkout(body: CheckoutBody, user: dict = Depends(require_user)):
    if user.get("_anonymous"):
        raise HTTPException(401, "Create an account to subscribe")
    s = get_settings()
    try:
        result = billing.create_checkout_session(
            user,
            body.plan,
            success_url=f"{s.public_url.rstrip('/')}/?billing=success",
            cancel_url=f"{s.public_url.rstrip('/')}/?billing=cancel",
        )
        return result
    except Exception as e:
        raise HTTPException(400, str(e))


@router.post("/billing/portal")
async def portal(user: dict = Depends(require_user)):
    if user.get("_anonymous"):
        raise HTTPException(401, "Create an account to manage billing")
    s = get_settings()
    try:
        return billing.create_portal_session(
            user, return_url=f"{s.public_url.rstrip('/')}/"
        )
    except Exception as e:
        raise HTTPException(400, str(e))


@router.post("/billing/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")
    try:
        result = billing.handle_webhook(payload, sig)
        return result
    except Exception as e:
        raise HTTPException(400, str(e))


class SetPlanBody(BaseModel):
    plan: str = "pro"


@router.post("/billing/dev-set-plan")
async def dev_set_plan(body: SetPlanBody, user: dict = Depends(require_user)):
    """
    Set plan without Stripe — development / offline sales only.
    Disabled when ENVIRONMENT=production unless STRIPE is not configured and
    DEV_BILLING_OVERRIDE=true is set (for operator-managed invoicing).
    """
    import os

    s = get_settings()
    # Allow when Stripe is not live-configured (test/offline sales) or explicit override
    stripe_live = bool(
        (s.stripe_secret_key or "").startswith("sk_live_")
        and s.stripe_price_pro
    )
    allow = (
        s.environment != "production"
        or not stripe_live
        or os.environ.get("DEV_BILLING_OVERRIDE", "").lower() in ("1", "true", "yes")
    )
    if not allow:
        raise HTTPException(403, "Manual plan changes disabled with live Stripe. Use Checkout.")
    if user.get("_anonymous"):
        raise HTTPException(401, "Create an account first")
    plan = (body.plan or "free").strip().lower()
    if plan not in users.PLANS:
        raise HTTPException(400, f"Unknown plan: {plan}")
    u = users.update_user(user["id"], plan=plan)
    return {"user": users.public_user(u), "note": "Plan set without Stripe (operator override)"}


@router.get("/jobs")
async def my_jobs(user: dict = Depends(require_user)):
    if user.get("_anonymous"):
        return {"jobs": []}
    return {"jobs": users.list_jobs(user["id"])}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, user: dict = Depends(require_user)):
    job = users.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if not user.get("_anonymous") and job["user_id"] != user["id"]:
        raise HTTPException(403, "Not your job")
    return job
