"""Shared SQLite connection + schema migrations for multi-tenant scale."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from app.core.config import get_settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    name TEXT,
    plan TEXT NOT NULL DEFAULT 'free',
    stripe_customer_id TEXT,
    stripe_subscription_id TEXT,
    sessions_used_month INTEGER NOT NULL DEFAULT 0,
    usage_month TEXT,
    org_id TEXT,
    is_admin INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS password_resets (
    token TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    used INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY(user_id) REFERENCES users(id)
);
CREATE TABLE IF NOT EXISTS orgs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    plan TEXT NOT NULL DEFAULT 'team',
    seat_limit INTEGER NOT NULL DEFAULT 10,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS org_members (
    org_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'member',
    created_at TEXT NOT NULL,
    PRIMARY KEY (org_id, user_id),
    FOREIGN KEY(org_id) REFERENCES orgs(id),
    FOREIGN KEY(user_id) REFERENCES users(id)
);
CREATE TABLE IF NOT EXISTS org_invites (
    token TEXT PRIMARY KEY,
    org_id TEXT NOT NULL,
    email TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'member',
    invited_by TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    accepted INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY(org_id) REFERENCES orgs(id)
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    org_id TEXT,
    session_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    progress REAL DEFAULT 0,
    message TEXT,
    error TEXT,
    payload TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    lease_owner TEXT,
    lease_until TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    FOREIGN KEY(user_id) REFERENCES users(id)
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_user ON jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_jobs_session ON jobs(session_id);
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    actor_id TEXT,
    action TEXT NOT NULL,
    resource_type TEXT,
    resource_id TEXT,
    detail TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_log(created_at);
CREATE TABLE IF NOT EXISTS sla_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    session_id TEXT,
    user_id TEXT,
    ok INTEGER NOT NULL DEFAULT 1,
    duration_ms REAL,
    detail TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sla_kind ON sla_events(kind, created_at);
CREATE TABLE IF NOT EXISTS outbound_webhooks (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    org_id TEXT,
    url TEXT NOT NULL,
    secret TEXT,
    events TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS support_tickets (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    email TEXT NOT NULL,
    subject TEXT NOT NULL,
    body TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    created_at TEXT NOT NULL
);
"""


def connect() -> sqlite3.Connection:
    s = get_settings()
    s.ensure_dirs()
    conn = sqlite3.connect(str(s.db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_all() -> None:
    conn = connect()
    conn.executescript(_SCHEMA)
    # Additive migrations for older DBs
    cols = {r[1] for r in conn.execute("PRAGMA table_info(users)").fetchall()}
    if "org_id" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN org_id TEXT")
    if "is_admin" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER NOT NULL DEFAULT 0")
    job_cols = {r[1] for r in conn.execute("PRAGMA table_info(jobs)").fetchall()}
    for col, decl in [
        ("org_id", "TEXT"),
        ("payload", "TEXT"),
        ("attempts", "INTEGER NOT NULL DEFAULT 0"),
        ("max_attempts", "INTEGER NOT NULL DEFAULT 3"),
        ("lease_owner", "TEXT"),
        ("lease_until", "TEXT"),
        ("started_at", "TEXT"),
        ("finished_at", "TEXT"),
    ]:
        if col not in job_cols:
            try:
                conn.execute(f"ALTER TABLE jobs ADD COLUMN {col} {decl}")
            except sqlite3.OperationalError:
                pass
    conn.commit()
    conn.close()


def row_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row else None
