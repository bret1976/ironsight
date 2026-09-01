"""Session persistence — JSON + SQLite index. All data is from real pipelines."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.core.config import get_settings


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def new_session_id() -> str:
    return uuid.uuid4().hex[:12]


def session_dir(session_id: str) -> Path:
    d = get_settings().session_dir / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def session_path(session_id: str) -> Path:
    return session_dir(session_id) / "session.json"


def create_session(
    name: str,
    owner_id: Optional[str] = None,
    *,
    org_id: Optional[str] = None,
) -> dict[str, Any]:
    sid = new_session_id()
    sdir = session_dir(sid)
    (sdir / "frames").mkdir(exist_ok=True)
    (sdir / "audio").mkdir(exist_ok=True)
    (sdir / "recon").mkdir(exist_ok=True)
    (sdir / "previews").mkdir(exist_ok=True)
    doc = {
        "id": sid,
        "name": name,
        "owner_id": owner_id,
        "org_id": org_id,
        "public_demo": False,
        "status": "created",
        "stage": "queued",
        "created_at": _now(),
        "updated_at": _now(),
        "cameras": [],
        "shots": [],
        "targets": [],
        "stats": {
            "total_shots": 0,
            "hits": 0,
            "misses": 0,
            "accuracy": 0.0,
            "duration_s": 0.0,
            "best_split_s": None,
            "avg_split_s": None,
        },
        "sync": None,
        "pointcloud": None,
        "coaching": None,
        "pipeline_log": [],
        "error": None,
    }
    save_session(doc)
    return doc


def load_session(session_id: str) -> dict[str, Any]:
    p = session_path(session_id)
    if not p.exists():
        raise FileNotFoundError(f"Session not found: {session_id}")
    return json.loads(p.read_text())


def save_session(doc: dict[str, Any]) -> dict[str, Any]:
    doc["updated_at"] = _now()
    p = session_path(doc["id"])
    p.write_text(json.dumps(doc, indent=2))
    return doc


def list_sessions(
    owner_id: Optional[str] = None,
    *,
    org_id: Optional[str] = None,
    include_public_demo: bool = True,
) -> list[dict[str, Any]]:
    root = get_settings().session_dir
    out = []
    if not root.exists():
        return out
    for d in sorted(root.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        sp = d / "session.json"
        if sp.exists():
            try:
                doc = json.loads(sp.read_text())
                if owner_id and owner_id not in ("local",):
                    oid = doc.get("owner_id")
                    doc_org = doc.get("org_id")
                    is_public = bool(doc.get("public_demo"))
                    allowed = (
                        oid == owner_id
                        or (org_id and doc_org and doc_org == org_id)
                        or (include_public_demo and is_public)
                    )
                    if not allowed:
                        continue
                gs = doc.get("gaussian_splat")
                pc = doc.get("pointcloud")
                out.append(
                    {
                        "id": doc["id"],
                        "name": doc.get("name"),
                        "owner_id": doc.get("owner_id"),
                        "org_id": doc.get("org_id"),
                        "public_demo": bool(doc.get("public_demo")),
                        "status": doc.get("status"),
                        "stage": doc.get("stage"),
                        "created_at": doc.get("created_at"),
                        "updated_at": doc.get("updated_at"),
                        "stats": doc.get("stats"),
                        "cameras": [
                            {"role": c.get("role"), "filename": c.get("filename")}
                            for c in (doc.get("cameras") or [])
                        ],
                        "cam_count": len(doc.get("cameras") or []),
                        "gaussian_splat": (
                            {
                                "gaussian_count": gs.get("gaussian_count"),
                                "method": gs.get("method"),
                                "num_iters": gs.get("num_iters"),
                            }
                            if gs
                            else None
                        ),
                        "pointcloud": (
                            {
                                "point_count": pc.get("point_count"),
                                "method": pc.get("method"),
                            }
                            if pc
                            else None
                        ),
                        "tracks": doc.get("tracks"),
                        "error": doc.get("error"),
                    }
                )
            except Exception:
                continue
    return out


def append_log(doc: dict[str, Any], stage: str, progress: float, message: str, data: Optional[dict] = None):
    doc.setdefault("pipeline_log", []).append(
        {
            "session_id": doc["id"],
            "stage": stage,
            "progress": progress,
            "message": message,
            "data": data or {},
            "ts": _now(),
        }
    )
    doc["stage"] = stage
    # keep log bounded
    if len(doc["pipeline_log"]) > 400:
        doc["pipeline_log"] = doc["pipeline_log"][-400:]
