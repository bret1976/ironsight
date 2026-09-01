"""Auth dependencies for FastAPI routes."""
from __future__ import annotations

from typing import Any, Optional

from fastapi import Depends, Header, HTTPException, Request

from app.core.config import get_settings
from app.services import users


async def optional_user(
    request: Request,
    authorization: Optional[str] = Header(None),
) -> Optional[dict[str, Any]]:
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
    if not token:
        token = request.cookies.get("ironsight_token")
    if not token:
        token = request.query_params.get("token")
    if not token:
        return None
    uid = users.decode_token(token)
    if not uid:
        return None
    return users.get_user(uid)


async def require_user(user: Optional[dict] = Depends(optional_user)) -> dict[str, Any]:
    s = get_settings()
    if user:
        return user
    if not s.require_auth and s.environment != "production":
        # Dev convenience: anonymous local operator
        return {
            "id": "local",
            "email": "local@ironsight.dev",
            "name": "Local Operator",
            "plan": "team",
            "sessions_used_month": 0,
            "usage_month": None,
            "stripe_customer_id": None,
            "stripe_subscription_id": None,
            "_anonymous": True,
        }
    raise HTTPException(401, "Authentication required. Sign in or register.")


def assert_session_owner(doc: dict, user: dict, *, write: bool = False) -> None:
    if user.get("_anonymous") or user.get("id") == "local":
        return
    # Public demo: always read-only for everyone; never claim as private owner
    if doc.get("public_demo"):
        if write:
            raise HTTPException(
                403,
                "This is a sample practice — create your own with New practice to upload and analyze.",
            )
        return

    owner = doc.get("owner_id")
    org_id = doc.get("org_id")
    if owner and owner == user["id"]:
        return
    # Org-shared session
    if org_id:
        try:
            from app.services import orgs

            orgs.assert_member(org_id, user["id"])
            return
        except Exception:
            pass
    if owner and owner != user["id"]:
        raise HTTPException(403, "Not your session")
    # Legacy unowned (not demo): claim for first authenticated user
    if not owner and not user.get("_anonymous"):
        try:
            from app.services import store

            doc["owner_id"] = user["id"]
            store.save_session(doc)
        except Exception:
            pass
