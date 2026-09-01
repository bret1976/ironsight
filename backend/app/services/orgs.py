"""Multi-tenant organizations + seats (Scale tier)."""
from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from app.services import audit, db, users

# Seat limits by plan (Pro includes small collab team)
SEAT_LIMITS = {"free": 1, "pro": 3, "team": 10, "enterprise": 50}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _uid() -> str:
    return secrets.token_hex(12)


def create_org(owner_id: str, name: str, *, plan: str = "team") -> dict[str, Any]:
    db.init_all()
    oid = _uid()
    now = _now()
    # Prefer seats from user plan definition (Pro=3, Team=10)
    plan_def = users.PLANS.get(plan) or {}
    seats = int(plan_def.get("seats") or SEAT_LIMITS.get(plan, 10))
    conn = db.connect()
    conn.execute(
        "INSERT INTO orgs (id, name, owner_id, plan, seat_limit, created_at, updated_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (oid, name.strip() or "Team", owner_id, plan, seats, now, now),
    )
    conn.execute(
        "INSERT INTO org_members (org_id, user_id, role, created_at) VALUES (?,?,?,?)",
        (oid, owner_id, "owner", now),
    )
    conn.execute("UPDATE users SET org_id = ?, updated_at = ? WHERE id = ?", (oid, now, owner_id))
    conn.commit()
    conn.close()
    audit.log("org.create", actor_id=owner_id, resource_type="org", resource_id=oid, detail={"name": name})
    return get_org(oid) or {}


def get_org(org_id: str) -> Optional[dict[str, Any]]:
    db.init_all()
    conn = db.connect()
    row = conn.execute("SELECT * FROM orgs WHERE id = ?", (org_id,)).fetchone()
    if not row:
        conn.close()
        return None
    members = conn.execute(
        "SELECT m.user_id, m.role, m.created_at, u.email, u.name, u.plan "
        "FROM org_members m LEFT JOIN users u ON u.id = m.user_id WHERE m.org_id = ?",
        (org_id,),
    ).fetchall()
    conn.close()
    o = dict(row)
    o["members"] = [dict(m) for m in members]
    o["seat_used"] = len(o["members"])
    o["seats_available"] = max(0, int(o["seat_limit"]) - len(o["members"]))
    return o


def user_orgs(user_id: str) -> list[dict[str, Any]]:
    db.init_all()
    conn = db.connect()
    rows = conn.execute(
        "SELECT o.* FROM orgs o JOIN org_members m ON m.org_id = o.id WHERE m.user_id = ?",
        (user_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def member_role(org_id: str, user_id: str) -> Optional[str]:
    db.init_all()
    conn = db.connect()
    row = conn.execute(
        "SELECT role FROM org_members WHERE org_id = ? AND user_id = ?",
        (org_id, user_id),
    ).fetchone()
    conn.close()
    return row["role"] if row else None


def assert_member(org_id: str, user_id: str, *, roles: Optional[set[str]] = None) -> str:
    role = member_role(org_id, user_id)
    if not role:
        raise PermissionError("Not a member of this organization")
    if roles and role not in roles:
        raise PermissionError("Insufficient org role")
    return role


def invite(org_id: str, email: str, invited_by: str, role: str = "member") -> dict[str, Any]:
    assert_member(org_id, invited_by, roles={"owner", "admin"})
    org = get_org(org_id)
    if not org:
        raise ValueError("Org not found")
    if org["seats_available"] <= 0:
        raise ValueError(f"Seat limit reached ({org['seat_limit']}). Upgrade plan for more seats.")
    email = email.strip().lower()
    token = secrets.token_urlsafe(24)
    exp = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat().replace("+00:00", "Z")
    conn = db.connect()
    conn.execute(
        "INSERT INTO org_invites (token, org_id, email, role, invited_by, expires_at, created_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (token, org_id, email, role, invited_by, exp, _now()),
    )
    conn.commit()
    conn.close()
    audit.log(
        "org.invite",
        actor_id=invited_by,
        resource_type="org",
        resource_id=org_id,
        detail={"email": email, "role": role},
    )
    return {"token": token, "email": email, "expires_at": exp, "org_id": org_id, "role": role}


def accept_invite(token: str, user_id: str) -> dict[str, Any]:
    db.init_all()
    conn = db.connect()
    row = conn.execute("SELECT * FROM org_invites WHERE token = ?", (token,)).fetchone()
    if not row:
        conn.close()
        raise ValueError("Invalid invite")
    inv = dict(row)
    if inv["accepted"]:
        conn.close()
        raise ValueError("Invite already used")
    if inv["expires_at"] < _now():
        conn.close()
        raise ValueError("Invite expired")
    user = users.get_user(user_id)
    if not user or user["email"].lower() != inv["email"].lower():
        conn.close()
        raise ValueError("Invite email does not match your account")
    org = get_org(inv["org_id"])
    if not org:
        conn.close()
        raise ValueError("Org missing")
    if org["seats_available"] <= 0 and not member_role(inv["org_id"], user_id):
        conn.close()
        raise ValueError("No seats available")
    now = _now()
    conn.execute(
        "INSERT OR IGNORE INTO org_members (org_id, user_id, role, created_at) VALUES (?,?,?,?)",
        (inv["org_id"], user_id, inv["role"], now),
    )
    conn.execute("UPDATE org_invites SET accepted = 1 WHERE token = ?", (token,))
    conn.execute("UPDATE users SET org_id = ?, updated_at = ? WHERE id = ?", (inv["org_id"], now, user_id))
    conn.commit()
    conn.close()
    audit.log("org.accept_invite", actor_id=user_id, resource_type="org", resource_id=inv["org_id"])
    return get_org(inv["org_id"]) or {}


def remove_member(org_id: str, target_user_id: str, actor_id: str) -> None:
    assert_member(org_id, actor_id, roles={"owner", "admin"})
    org = get_org(org_id)
    if org and org["owner_id"] == target_user_id:
        raise ValueError("Cannot remove org owner")
    conn = db.connect()
    conn.execute("DELETE FROM org_members WHERE org_id = ? AND user_id = ?", (org_id, target_user_id))
    conn.execute(
        "UPDATE users SET org_id = NULL, updated_at = ? WHERE id = ? AND org_id = ?",
        (_now(), target_user_id, org_id),
    )
    conn.commit()
    conn.close()
    audit.log(
        "org.remove_member",
        actor_id=actor_id,
        resource_type="org",
        resource_id=org_id,
        detail={"user_id": target_user_id},
    )
