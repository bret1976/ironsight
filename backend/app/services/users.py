"""User accounts + JWT auth for multi-tenant production."""
from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.core.config import get_settings

# Plans sellable today — Free = trial scrub; Pro = coach product; Team = volume + labs
PLANS = {
    "free": {
        "id": "free",
        "name": "Free",
        "price_usd": 0,
        "sessions_per_month": 3,
        "max_video_minutes": 5,
        "max_cameras": 2,
        "gsplat": False,
        "pose": False,
        "mappers": False,
        "recon": False,  # skip COLMAP on free (speed + cost)
        "coaching": False,  # full LLM debrief is Pro
        "wrap_export": False,  # share/export card is Pro
        "seats": 1,
        "priority": False,
        "lanes": False,
        "range_dashboard": False,
        "description": "Try dual-cam review — 3 short sessions, shot log + hit/miss assist",
    },
    "pro": {
        "id": "pro",
        "name": "Pro",
        "price_usd": 49,
        "sessions_per_month": 40,
        "max_video_minutes": 30,
        "max_cameras": 2,
        "gsplat": True,
        "pose": True,
        "mappers": False,
        "recon": True,
        "coaching": True,
        "wrap_export": True,
        "seats": 3,  # coach + assistant + shooter
        "priority": False,
        "lanes": False,
        "range_dashboard": False,
        "description": "Coach product — fixable scores, debrief export, 3D, 3 seats",
    },
    "team": {
        "id": "team",
        "name": "Team",
        "price_usd": 149,
        "sessions_per_month": 200,
        "max_video_minutes": 90,
        "max_cameras": 2,
        "gsplat": True,
        "pose": True,
        "mappers": True,
        "recon": True,
        "coaching": True,
        "wrap_export": True,
        "seats": 10,
        "priority": True,  # real priority job queue
        "lanes": True,
        "range_dashboard": True,
        "description": "Range ops — 10 seats, priority queue, lanes, shared library",
    },
}


def _db() -> sqlite3.Connection:
    from app.services import db as dbmod

    dbmod.init_all()
    return dbmod.connect()


def init_db() -> None:
    from app.services import db as dbmod

    dbmod.init_all()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _uid() -> str:
    return secrets.token_hex(12)


def _hash_password(password: str, salt: Optional[str] = None) -> str:
    salt = salt or secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 200_000)
    return f"pbkdf2_sha256${salt}${dk.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        algo, salt, hx = stored.split("$", 2)
        if algo != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 200_000)
        return secrets.compare_digest(dk.hex(), hx)
    except Exception:
        return False


def register(email: str, password: str, name: str = "") -> dict[str, Any]:
    email = email.strip().lower()
    if "@" not in email or len(password) < 8:
        raise ValueError("Valid email and password (8+ chars) required")
    init_db()
    conn = _db()
    uid = _uid()
    now = _now()
    try:
        conn.execute(
            "INSERT INTO users (id, email, password_hash, name, plan, usage_month, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, 'free', ?, ?, ?)",
            (uid, email, _hash_password(password), name.strip() or email.split("@")[0], now[:7], now, now),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        raise ValueError("Email already registered")
    conn.close()
    return get_user(uid)


def authenticate(email: str, password: str) -> Optional[dict[str, Any]]:
    init_db()
    conn = _db()
    row = conn.execute(
        "SELECT * FROM users WHERE email = ?", (email.strip().lower(),)
    ).fetchone()
    conn.close()
    if not row or not _verify_password(password, row["password_hash"]):
        return None
    return dict(row)


def get_user(user_id: str) -> Optional[dict[str, Any]]:
    init_db()
    conn = _db()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_email(email: str) -> Optional[dict[str, Any]]:
    init_db()
    conn = _db()
    row = conn.execute(
        "SELECT * FROM users WHERE email = ?", (email.strip().lower(),)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_stripe_customer(customer_id: str) -> Optional[dict[str, Any]]:
    init_db()
    conn = _db()
    row = conn.execute(
        "SELECT * FROM users WHERE stripe_customer_id = ?", (customer_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def update_user(user_id: str, **fields: Any) -> dict[str, Any]:
    allowed = {
        "name",
        "plan",
        "stripe_customer_id",
        "stripe_subscription_id",
        "sessions_used_month",
        "usage_month",
        "org_id",
        "is_admin",
        "password_hash",
    }
    sets = []
    vals = []
    for k, v in fields.items():
        if k in allowed:
            sets.append(f"{k} = ?")
            vals.append(v)
    if not sets:
        return get_user(user_id) or {}
    sets.append("updated_at = ?")
    vals.append(_now())
    vals.append(user_id)
    conn = _db()
    conn.execute(f"UPDATE users SET {', '.join(sets)} WHERE id = ?", vals)
    conn.commit()
    conn.close()
    return get_user(user_id) or {}


def public_user(u: dict[str, Any]) -> dict[str, Any]:
    plan_id = u.get("plan") or "free"
    plan = PLANS.get(plan_id, PLANS["free"])
    return {
        "id": u["id"],
        "email": u["email"],
        "name": u.get("name"),
        "plan": plan_id,
        "plan_details": plan,
        "sessions_used_month": u.get("sessions_used_month") or 0,
        "usage_month": u.get("usage_month"),
        "stripe_customer_id": u.get("stripe_customer_id"),
        "has_subscription": bool(u.get("stripe_subscription_id")),
        "org_id": u.get("org_id"),
        "is_admin": bool(u.get("is_admin")),
        "seats": plan.get("seats") or 1,
    }


def create_password_reset(email: str) -> Optional[dict[str, Any]]:
    """Create reset token. Always returns ok shape (no email enumeration)."""
    init_db()
    u = get_user_by_email(email)
    if not u:
        return {"ok": True, "sent": False}
    token = secrets.token_urlsafe(32)
    exp = datetime.now(timezone.utc).timestamp() + 3600
    exp_s = datetime.fromtimestamp(exp, timezone.utc).isoformat().replace("+00:00", "Z")
    conn = _db()
    conn.execute(
        "INSERT INTO password_resets (token, user_id, expires_at, used, created_at) VALUES (?,?,?,0,?)",
        (token, u["id"], exp_s, _now()),
    )
    conn.commit()
    conn.close()
    from app.core.config import get_settings
    from app.services import mailer

    s = get_settings()
    link = f"{s.public_url.rstrip('/')}/?reset={token}"
    mailer.send_email(
        u["email"],
        "IronSight password reset",
        f"Reset your password (valid 1 hour):\n\n{link}\n\nIf you did not request this, ignore.",
    )
    return {"ok": True, "sent": True, "reset_token": token if s.environment != "production" else None, "link": link if s.environment != "production" else None}


def reset_password(token: str, new_password: str) -> dict[str, Any]:
    if len(new_password) < 8:
        raise ValueError("Password must be 8+ characters")
    init_db()
    conn = _db()
    row = conn.execute("SELECT * FROM password_resets WHERE token = ?", (token,)).fetchone()
    if not row or row["used"]:
        conn.close()
        raise ValueError("Invalid or used reset token")
    if row["expires_at"] < _now():
        conn.close()
        raise ValueError("Reset token expired")
    uid = row["user_id"]
    conn.execute(
        "UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?",
        (_hash_password(new_password), _now(), uid),
    )
    conn.execute("UPDATE password_resets SET used = 1 WHERE token = ?", (token,))
    conn.commit()
    conn.close()
    return {"ok": True, "user_id": uid}


def change_password(user_id: str, old_password: str, new_password: str) -> None:
    u = get_user(user_id)
    if not u or not _verify_password(old_password, u["password_hash"]):
        raise ValueError("Current password incorrect")
    if len(new_password) < 8:
        raise ValueError("New password must be 8+ characters")
    update_user(user_id, password_hash=_hash_password(new_password))


def delete_user(user_id: str) -> None:
    init_db()
    conn = _db()
    conn.execute("DELETE FROM org_members WHERE user_id = ?", (user_id,))
    conn.execute("DELETE FROM password_resets WHERE user_id = ?", (user_id,))
    conn.execute("DELETE FROM jobs WHERE user_id = ?", (user_id,))
    conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()


def list_users(limit: int = 100) -> list[dict[str, Any]]:
    init_db()
    conn = _db()
    rows = conn.execute(
        "SELECT id, email, name, plan, org_id, is_admin, sessions_used_month, usage_month, created_at "
        "FROM users ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def ensure_admin_bootstrap() -> None:
    """If no admin exists and ADMIN_EMAIL matches a user, promote them."""
    import os

    email = (os.environ.get("ADMIN_EMAIL") or "").strip().lower()
    if not email:
        return
    u = get_user_by_email(email)
    if u and not u.get("is_admin"):
        update_user(u["id"], is_admin=1)


# ── JWT (HS256, no external dep beyond stdlib hmac) ─────────────────────

def _jwt_secret() -> str:
    s = get_settings()
    secret = (s.jwt_secret or s.api_key or "").strip()
    if not secret:
        # Dev fallback — production must set JWT_SECRET
        secret = "ironsight-dev-only-change-me"
    return secret


def create_token(user_id: str, *, hours: int = 72) -> str:
    import base64
    import hmac

    header = base64.urlsafe_b64encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode()).rstrip(b"=")
    exp = int(time.time()) + hours * 3600
    payload = base64.urlsafe_b64encode(
        json.dumps({"sub": user_id, "exp": exp}).encode()
    ).rstrip(b"=")
    msg = header + b"." + payload
    sig = hmac.new(_jwt_secret().encode(), msg, hashlib.sha256).digest()
    return (msg + b"." + base64.urlsafe_b64encode(sig).rstrip(b"=")).decode()


def decode_token(token: str) -> Optional[str]:
    import base64
    import hmac

    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header_b, payload_b, sig_b = parts

        def pad(s: str) -> bytes:
            return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))

        msg = f"{header_b}.{payload_b}".encode()
        expected = hmac.new(_jwt_secret().encode(), msg, hashlib.sha256).digest()
        if not secrets.compare_digest(base64.urlsafe_b64encode(expected).rstrip(b"=").decode(), sig_b):
            return None
        payload = json.loads(pad(payload_b))
        if int(payload.get("exp", 0)) < time.time():
            return None
        return str(payload.get("sub") or "")
    except Exception:
        return None


def check_can_create_session(user: dict[str, Any]) -> tuple[bool, str]:
    """Enforce plan quotas. Resets monthly counter."""
    plan = PLANS.get(user.get("plan") or "free", PLANS["free"])
    month = _now()[:7]
    used = int(user.get("sessions_used_month") or 0)
    um = user.get("usage_month") or ""
    if um != month:
        update_user(user["id"], sessions_used_month=0, usage_month=month)
        used = 0
    limit = int(plan["sessions_per_month"])
    if used >= limit:
        return False, f"Plan limit reached ({limit} sessions/{month}). Upgrade to continue."
    return True, ""


def record_session_created(user_id: str) -> None:
    u = get_user(user_id)
    if not u:
        return
    month = _now()[:7]
    used = int(u.get("sessions_used_month") or 0)
    if (u.get("usage_month") or "") != month:
        update_user(user_id, sessions_used_month=1, usage_month=month)
    else:
        update_user(user_id, sessions_used_month=used + 1, usage_month=month)


def plan_allows(user: dict[str, Any], feature: str) -> bool:
    plan = PLANS.get(user.get("plan") or "free", PLANS["free"])
    return bool(plan.get(feature))


# ── Jobs (delegates to durable queue) ──────────────────────────────────

def create_job(
    user_id: str,
    session_id: str,
    kind: str = "pipeline",
    org_id: Optional[str] = None,
    priority: int = 0,
) -> dict[str, Any]:
    from app.services import jobs_queue

    return jobs_queue.enqueue(user_id, session_id, kind, org_id=org_id, priority=priority)


def get_job(job_id: str) -> Optional[dict[str, Any]]:
    from app.services import jobs_queue

    return jobs_queue.get_job(job_id)


def update_job(job_id: str, **fields: Any) -> None:
    from app.services import jobs_queue

    jobs_queue.update_job(job_id, **fields)


def list_jobs(user_id: str, limit: int = 20) -> list[dict[str, Any]]:
    from app.services import jobs_queue

    return jobs_queue.list_jobs(user_id, limit=limit)
