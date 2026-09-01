from __future__ import annotations

import asyncio
import mimetypes
import shutil
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from app.api.deps import assert_session_owner, require_user
from app.core.config import get_settings
from app.models.schemas import CreateSessionRequest
from app.services import pipeline, store, users

router = APIRouter()

# session_id -> set of websockets
_ws_clients: dict[str, set[WebSocket]] = {}
_ws_global: set[WebSocket] = set()


async def broadcast(session_id: str, payload: dict[str, Any]):
    dead = []
    for ws in list(_ws_clients.get(session_id, set())):
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.get(session_id, set()).discard(ws)
    dead_g = []
    for ws in list(_ws_global):
        try:
            await ws.send_json(payload)
        except Exception:
            dead_g.append(ws)
    for ws in dead_g:
        _ws_global.discard(ws)


def _legal_docs_ready() -> bool:
    from app.core.config import ROOT
    allowed = ("terms", "privacy", "refund", "sla")
    return all((ROOT / "legal" / f"{d}.md").exists() for d in allowed)


def _legal_docs_status() -> dict:
    from app.core.config import ROOT
    out = {}
    for d in ("terms", "privacy", "refund", "sla"):
        out[d] = (ROOT / "legal" / f"{d}.md").exists()
    return out


def _demo_session_ready(demo_id: str) -> bool:
    try:
        from app.services import store
        store.load_session(demo_id)
        return True
    except Exception:
        return False


@router.get("/health")
async def health():
    from app.services import gaussian_splat, gaussian_train, recon_3d, yolo_plates

    s = get_settings()
    os_rt = gaussian_splat.opensplat_runtime()
    torch_dev = gaussian_train.device_str() if gaussian_train.torch_available() else "none"
    metal_xcode = gaussian_splat.metal_toolchain_available()
    # Active Metal training capability = OpenSplat MPS kernels OR PyTorch MPS
    metal_training = os_rt == "mps" or torch_dev == "mps"
    yolo_ok = yolo_plates.weights_available()
    return {
        "ok": True,
        "app": s.app_name,
        "version": s.app_version,
        "vision": bool(s.openrouter_api_key or s.xai_api_key),
        "vision_provider": "openrouter" if s.openrouter_api_key else ("xai" if s.xai_api_key else "none"),
        "vision_model": s.vision_model if s.openrouter_api_key else "grok-vision",
        "colmap": recon_3d.colmap_available(),
        "opensplat": gaussian_splat.opensplat_available(),
        "opensplat_runtime": os_rt,
        "opensplat_bin": gaussian_splat.opensplat_binary(),
        "metal_toolchain": metal_xcode,
        "metal_training": metal_training,
        "metal_path": (
            "opensplat_mps"
            if os_rt == "mps"
            else ("pytorch_mps" if torch_dev == "mps" else "cpu_only")
        ),
        "pytorch_3dgs": gaussian_train.torch_available(),
        "torch_device": torch_dev,
        "yolo_plates": yolo_ok,
        "yolo_weights": str(yolo_plates.WEIGHTS_PATH) if yolo_ok else None,
        "gsplat_enabled": s.gsplat_enabled,
        "gsplat_pytorch_iters": s.gsplat_pytorch_iters,
        "gsplat_opensplat_iters": s.gsplat_opensplat_iters,
        "recon_prefer_colmap": s.recon_prefer_colmap,
        "shot_profile": s.shot_profile,
        "recon_max_frames": s.recon_max_frames,
        "recon_point_budget": s.recon_point_budget,
        "saas": {
            "require_auth": s.require_auth,
            "environment": s.environment,
            "stripe_configured": bool(s.stripe_secret_key and s.stripe_price_pro),
            "stripe_mode": (
                "live"
                if (s.stripe_secret_key or "").startswith("sk_live_")
                else ("test" if (s.stripe_secret_key or "").startswith("sk_test_") else "none")
            ),
            "jwt_configured": bool(s.jwt_secret),
            "plans": list(users.PLANS.keys()),
            "scale": {
                "orgs": True,
                "seats": True,
                "job_queue": True,
                "sla": True,
                "legal": _legal_docs_ready(),
                "legal_docs": _legal_docs_status(),
                "rate_limit": True,
                "demo_session": s.demo_session_id,
                "demo_session_ready": _demo_session_ready(s.demo_session_id),
            },
        },
        "godseye": True,
        "xcode_install_hint": (
            None
            if metal_xcode
            else "Install Xcode from App Store, then: bash scripts/setup_metal_opensplat.sh"
        ),
    }


@router.post("/models/yolo/train")
async def train_yolo_plates(epochs: int = 30, imgsz: int = 640):
    """Fine-tune plate YOLO on harvested session labels (CV/VLM weak labels)."""
    from app.services import yolo_plates

    try:
        meta = await asyncio.to_thread(
            yolo_plates.train_plate_yolo, epochs=epochs, imgsz=imgsz
        )
        return {"ok": True, **meta}
    except Exception as e:
        raise HTTPException(500, f"YOLO train failed: {e}")


@router.get("/sessions/{session_id}/radar")
async def get_radar(session_id: str):
    """Through-wall radar occupancy volume JSON."""
    try:
        doc = store.load_session(session_id)
    except FileNotFoundError:
        raise HTTPException(404, "Session not found")
    radar = doc.get("radar") or {}
    path = radar.get("path") or str(store.session_dir(session_id) / "recon" / "radar_volume.json")
    if not Path(path).exists():
        # build on demand
        from app.services import radar_ghost

        pc = doc.get("pointcloud") or {}
        if not pc.get("path") or not Path(pc["path"]).exists():
            raise HTTPException(404, "Point cloud not ready for radar")
        try:
            built = await asyncio.to_thread(
                radar_ghost.build_radar_volume,
                Path(pc["path"]),
                doc.get("targets") or [],
                out_path=store.session_dir(session_id) / "recon" / "radar_volume.json",
            )
            doc["radar"] = {
                "path": built.get("path"),
                "voxel_count": built.get("voxel_count"),
                "grid": built.get("grid"),
                "method": built.get("method"),
                "returns": len(built.get("returns") or []),
            }
            store.save_session(doc)
            path = built["path"]
        except Exception as e:
            raise HTTPException(500, f"Radar build failed: {e}")
    return FileResponse(path, media_type="application/json")


@router.post("/sessions/{session_id}/mappers/run")
async def run_multi_mapper(
    session_id: str,
    max_frames: int = 40,
    include_high: bool = False,
    user: dict = Depends(require_user),
):
    """R&D multi-mapper experiment board (COLMAP tiers + OpenCV, Sim3 compare)."""
    from app.services import multi_mapper

    try:
        doc = store.load_session(session_id)
    except FileNotFoundError:
        raise HTTPException(404, "Session not found")
    assert_session_owner(doc, user)
    if not users.plan_allows(user, "mappers"):
        raise HTTPException(402, "Multi-mapper board requires Team plan. Upgrade to unlock.")
    cams = doc.get("cameras") or []
    if not cams:
        raise HTTPException(400, "Upload cameras first")
    video = Path(cams[0]["path"])
    sdir = store.session_dir(session_id)
    methods = ["colmap_low", "colmap_medium", "opencv_sift"]
    if include_high:
        methods.insert(2, "colmap_high")

    async def _run():
        return await asyncio.to_thread(
            multi_mapper.run_experiment_board,
            video,
            sdir / "mappers",
            max_frames=max_frames,
            methods=methods,
        )

    try:
        board = await _run()
        doc["mapper_board"] = {
            "path": board.get("path"),
            "primary": board.get("primary"),
            "fallback": board.get("fallback"),
            "reference": board.get("reference"),
            "summary": board.get("summary"),
            "comparisons": board.get("comparisons"),
        }
        store.save_session(doc)
        return board
    except Exception as e:
        raise HTTPException(500, f"Multi-mapper failed: {e}")


@router.get("/sessions/{session_id}/mappers")
async def get_mapper_board(session_id: str):
    try:
        doc = store.load_session(session_id)
    except FileNotFoundError:
        raise HTTPException(404, "Session not found")
    path = (doc.get("mapper_board") or {}).get("path") or str(
        store.session_dir(session_id) / "mappers" / "board.json"
    )
    if not Path(path).exists():
        raise HTTPException(404, "Mapper board not run yet — POST .../mappers/run")
    return FileResponse(path, media_type="application/json")


@router.post("/sessions/{session_id}/annotate")
async def annotate_shot(
    session_id: str,
    payload: dict[str, Any],
    user: dict = Depends(require_user),
):
    """Human review — single path through coach_review (keeps needs_review clean)."""
    from app.services import annotations, coach_review

    try:
        doc = store.load_session(session_id)
    except FileNotFoundError:
        raise HTTPException(404, "Session not found")
    assert_session_owner(doc, user, write=True)
    if doc.get("public_demo"):
        raise HTTPException(403, "Sample practice is read-only. Start a New practice.")
    shot_id = payload.get("shot_id")
    if shot_id is None:
        raise HTTPException(400, "shot_id required")
    try:
        # Apply full annotation fields first (bbox, target, etc.)
        result = await asyncio.to_thread(
            annotations.save_correction,
            session_id,
            shot_id=int(shot_id),
            classification=payload.get("classification"),
            target_id=payload.get("target_id"),
            target_label=payload.get("target_label"),
            target_color=payload.get("target_color"),
            bbox=payload.get("bbox"),
            note=payload.get("note"),
        )
        # Always clear needs_review via coach path
        if payload.get("classification"):
            q = await asyncio.to_thread(
                coach_review.mark_shot_reviewed,
                session_id,
                int(shot_id),
                classification=payload.get("classification"),
                note=payload.get("note"),
                user_id=user.get("id"),
            )
            result["queue"] = q
        return result
    except Exception as e:
        raise HTTPException(500, f"annotate failed: {e}")


@router.get("/sessions/{session_id}/wrap")
async def session_wrap(session_id: str, user: dict = Depends(require_user)):
    from app.services import annotations

    try:
        doc = store.load_session(session_id)
    except FileNotFoundError:
        raise HTTPException(404, "Session not found")
    assert_session_owner(doc, user)
    # Fresh user record (plan may have changed via Stripe / operator)
    u = users.get_user(user["id"]) if not user.get("_anonymous") else user
    plan = users.PLANS.get((u or user).get("plan") or "free", users.PLANS["free"])
    export = bool(plan.get("wrap_export"))
    coaching_ok = bool(plan.get("coaching"))
    sealed = bool(doc.get("coach_locked") or doc.get("coach_ready"))
    try:
        if not coaching_ok:
            doc["coaching_locked"] = True
        wrap = await asyncio.to_thread(
            annotations.build_wrap_card,
            session_id,
            export=export,
            watermark=not export,
        )
        wrap["export_allowed"] = export
        wrap["coaching_locked"] = not coaching_ok
        wrap["coach_locked"] = sealed
        wrap["coach_ready"] = sealed
        # Pro/Team: soft-block full export until coach seal (demo exempt)
        if export and not sealed and not doc.get("public_demo"):
            wrap["export_allowed"] = False
            wrap["export_blocked_reason"] = (
                "Seal as coach-ready first (fix unsure shots, then Seal). "
                "This keeps shared reports trustworthy."
            )
            wrap["needs_coach_seal"] = True
        if not coaching_ok:
            wrap["coaching"] = None
        return wrap
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/sessions/{session_id}/wrap/export")
async def session_wrap_export(
    session_id: str,
    fmt: str = "html",
    user: dict = Depends(require_user),
):
    """Download coach wrap as HTML or Markdown. Pro+ for full export."""
    from app.services import annotations

    try:
        doc = store.load_session(session_id)
    except FileNotFoundError:
        raise HTTPException(404, "Session not found")
    assert_session_owner(doc, user)
    u = users.get_user(user["id"]) if not user.get("_anonymous") else user
    plan = users.PLANS.get((u or user).get("plan") or "free", users.PLANS["free"])
    export = bool(plan.get("wrap_export"))
    if not export:
        raise HTTPException(
            402,
            "Full wrap export requires Pro or Team. Upgrade to download coach debriefs.",
        )
    if not doc.get("public_demo") and not (
        doc.get("coach_locked") or doc.get("coach_ready")
    ):
        raise HTTPException(
            400,
            "Seal this practice as coach-ready before downloading the report "
            "(fix unsure shots → Seal as coach-ready).",
        )
    wrap = await asyncio.to_thread(
        annotations.build_wrap_card, session_id, export=True, watermark=False
    )
    fmt = (fmt or "html").lower()
    sdir = store.session_dir(session_id)
    if fmt == "md" or fmt == "markdown":
        path = sdir / "wrap_export.md"
        media = "text/markdown"
        name = f"ironsight_{session_id}_wrap.md"
    else:
        path = sdir / "wrap_export.html"
        media = "text/html"
        name = f"ironsight_{session_id}_wrap.html"
    if not path.exists():
        raise HTTPException(500, "Export file missing")
    return FileResponse(path, media_type=media, filename=name)


@router.post("/sessions/{session_id}/stats/recompute")
async def recompute_stats(session_id: str, user: dict = Depends(require_user)):
    """Recompute honest accuracy after bulk edits."""
    from app.services.scoring import recompute_session_stats

    try:
        doc = store.load_session(session_id)
    except FileNotFoundError:
        raise HTTPException(404, "Session not found")
    assert_session_owner(doc, user)
    stats = recompute_session_stats(doc)
    store.save_session(doc)
    return {"stats": stats}


@router.get("/sessions/{session_id}/pose")
async def get_pose(session_id: str):
    try:
        doc = store.load_session(session_id)
    except FileNotFoundError:
        raise HTTPException(404, "Session not found")
    pose = doc.get("pose") or {}
    path = pose.get("path") or str(store.session_dir(session_id) / "pose" / "pose_track.json")
    if not Path(path).exists():
        # build on demand
        from app.services import pose_track

        cams = doc.get("cameras") or []
        if not cams:
            raise HTTPException(404, "No cameras for pose")
        # prefer observer full-body
        vpath = Path(cams[-1]["path"] if len(cams) > 1 else cams[0]["path"])
        try:
            primary = Path(cams[0]["path"])
            observer = Path(cams[1]["path"]) if len(cams) > 1 else None
            built = await asyncio.to_thread(
                pose_track.build_session_pose,
                video_path=primary,
                pointcloud=doc.get("pointcloud"),
                duration_s=float((doc.get("stats") or {}).get("duration_s") or 30),
                out_path=store.session_dir(session_id) / "pose" / "pose_track.json",
                observer_path=observer,
                sync_offset_s=float((doc.get("sync") or {}).get("offset_s") or 0.0),
            )
            doc["pose"] = {
                "path": built.get("path"),
                "method": built.get("method"),
                "sample_count": built.get("sample_count"),
                "multiview_frames": built.get("multiview_frames"),
                "dual_cam_calibrated": built.get("dual_cam_calibrated"),
            }
            store.save_session(doc)
            path = built["path"]
        except Exception as e:
            raise HTTPException(500, f"Pose build failed: {e}")
    return FileResponse(path, media_type="application/json")


@router.post("/sessions/{session_id}/dualcam/clean")
async def clean_dualcam(session_id: str, user: dict = Depends(require_user)):
    """Re-run SxS split / HUD crop / streamer mask on session uploads."""
    from app.services import dualcam_clean

    try:
        doc = store.load_session(session_id)
    except FileNotFoundError:
        raise HTTPException(404, "Session not found")
    assert_session_owner(doc, user)
    sdir = store.session_dir(session_id)
    try:
        result = await asyncio.to_thread(dualcam_clean.clean_session_uploads, sdir)
        if not result.get("ok"):
            raise HTTPException(400, result.get("reason") or "clean failed")
        # update cameras if produced
        cams = []
        if result.get("shooter") and Path(result["shooter"]).exists():
            cams.append(
                {
                    "role": "shooter_fpv",
                    "filename": Path(result["shooter"]).name,
                    "path": result["shooter"],
                    "duration_s": 0.0,
                    "fps": 30.0,
                    "width": 0,
                    "height": 0,
                    "has_audio": True,
                    "cleaned": True,
                }
            )
        if result.get("observer") and Path(result["observer"]).exists():
            cams.append(
                {
                    "role": "observer_3p",
                    "filename": Path(result["observer"]).name,
                    "path": result["observer"],
                    "duration_s": 0.0,
                    "fps": 30.0,
                    "width": 0,
                    "height": 0,
                    "has_audio": True,
                    "cleaned": True,
                }
            )
        if cams:
            doc["cameras"] = cams
        doc["dualcam_clean"] = {
            k: result.get(k)
            for k in ("ok", "mode", "is_sxs", "streamer_masked", "crop", "report_path")
        }
        store.save_session(doc)
        return {"ok": True, "cameras": doc.get("cameras"), "dualcam_clean": doc["dualcam_clean"]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"dualcam clean failed: {e}")


@router.get("/sessions")
async def list_sessions(user: dict = Depends(require_user)):
    owner = None if user.get("_anonymous") or user.get("id") == "local" else user["id"]
    org_id = None if user.get("_anonymous") else (user.get("org_id") or (users.get_user(user["id"]) or {}).get("org_id"))
    return {
        "sessions": store.list_sessions(owner_id=owner, org_id=org_id, include_public_demo=True)
    }


@router.post("/sessions")
async def create_session(body: CreateSessionRequest, user: dict = Depends(require_user)):
    ok, msg = users.check_can_create_session(user)
    if not ok:
        raise HTTPException(402, msg)  # Payment Required — upgrade plan
    owner = None if user.get("_anonymous") else user["id"]
    org_id = None
    if not user.get("_anonymous"):
        u = users.get_user(user["id"]) or user
        org_id = u.get("org_id")
    doc = store.create_session(body.name, owner_id=owner, org_id=org_id)
    if not user.get("_anonymous"):
        users.record_session_created(user["id"])
    return doc


@router.get("/sessions/{session_id}")
async def get_session(session_id: str, user: dict = Depends(require_user)):
    try:
        doc = store.load_session(session_id)
    except FileNotFoundError:
        raise HTTPException(404, "Session not found")
    assert_session_owner(doc, user)
    return doc


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, user: dict = Depends(require_user)):
    try:
        doc = store.load_session(session_id)
    except FileNotFoundError:
        raise HTTPException(404, "Session not found")
    assert_session_owner(doc, user)
    d = store.session_dir(session_id)
    if not d.exists():
        raise HTTPException(404, "Session not found")
    shutil.rmtree(d, ignore_errors=True)
    return {"ok": True}


@router.post("/sessions/{session_id}/upload")
async def upload_cameras(
    session_id: str,
    shooter: Optional[UploadFile] = File(None),
    observer: Optional[UploadFile] = File(None),
    primary: Optional[UploadFile] = File(None),
    user: dict = Depends(require_user),
):
    """Upload 1–2 camera videos. `shooter`/`primary` is FPV; `observer` is 3P."""
    from app.services import frames

    try:
        doc = store.load_session(session_id)
    except FileNotFoundError:
        raise HTTPException(404, "Session not found")
    assert_session_owner(doc, user)

    plan = users.PLANS.get(user.get("plan") or "free", users.PLANS["free"])
    max_cams = int(plan.get("max_cameras") or 2)
    max_min = float(plan.get("max_video_minutes") or 90)

    sdir = store.session_dir(session_id)
    up = sdir / "uploads"
    up.mkdir(exist_ok=True)
    cameras = []

    async def _save(f: UploadFile, role: str) -> dict:
        suffix = Path(f.filename or "video.mp4").suffix or ".mp4"
        dest = up / f"{role}{suffix}"
        with dest.open("wb") as out:
            while True:
                chunk = await f.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        meta = frames.probe_video(dest)
        dur = float(meta.get("duration_s") or 0.0)
        if dur > max_min * 60 + 5:
            dest.unlink(missing_ok=True)
            raise HTTPException(
                402,
                f"Video is {dur/60:.1f} min; your plan allows {max_min} min. Upgrade to process longer video.",
            )
        return {
            "role": role,
            "filename": f.filename or dest.name,
            "path": str(dest),
            "duration_s": dur,
            "fps": float(meta.get("fps") or 30.0),
            "width": int(meta.get("width") or 0),
            "height": int(meta.get("height") or 0),
            "has_audio": True,
        }

    main = shooter or primary
    if not main and not observer:
        raise HTTPException(400, "Provide at least one video file (shooter/primary or observer)")

    if main and observer and max_cams < 2:
        raise HTTPException(
            402,
            "Dual-cam upload requires Pro or Team. Free plan allows 1 camera — upgrade to unlock dual-cam.",
        )

    if main:
        cameras.append(await _save(main, "shooter_fpv"))
    if observer:
        cameras.append(await _save(observer, "observer_3p"))

    # If only observer uploaded, promote it to primary
    if not main and observer:
        cameras[0]["role"] = "shooter_fpv"

    doc["cameras"] = cameras
    doc["status"] = "uploading"
    store.save_session(doc)
    return doc


@router.post("/sessions/{session_id}/process")
async def process_session(session_id: str, user: dict = Depends(require_user)):
    import os

    try:
        doc = store.load_session(session_id)
    except FileNotFoundError:
        raise HTTPException(404, "Session not found")
    assert_session_owner(doc, user, write=True)
    if doc.get("public_demo"):
        raise HTTPException(400, "Demo session is read-only. Create a new session to process your video.")
    if not doc.get("cameras"):
        raise HTTPException(400, "Upload cameras first")
    if pipeline.is_running(session_id):
        return {"ok": True, "message": "Pipeline already running", "session_id": session_id}

    # Fresh plan from DB
    u = users.get_user(user["id"]) if not user.get("_anonymous") else user
    plan = users.PLANS.get((u or user).get("plan") or "free", users.PLANS["free"])
    doc["plan_features"] = {
        "gsplat": bool(plan.get("gsplat")),
        "pose": bool(plan.get("pose")),
        "mappers": bool(plan.get("mappers")),
        "recon": bool(plan.get("recon", True)),
        "coaching": bool(plan.get("coaching", True)),
        "wrap_export": bool(plan.get("wrap_export", True)),
        "max_video_minutes": plan.get("max_video_minutes"),
    }
    # Duration gate
    max_min = float(plan.get("max_video_minutes") or 90)
    for cam in doc.get("cameras") or []:
        dur = float(cam.get("duration_s") or 0)
        if dur > max_min * 60 + 5:
            raise HTTPException(
                402,
                f"Video is {dur/60:.1f} min; your plan allows {max_min} min. Upgrade to process longer video.",
            )
    store.save_session(doc)

    env_q = os.environ.get("USE_JOB_QUEUE", "").lower()
    use_queue = env_q in ("1", "true", "yes") or (
        env_q not in ("0", "false", "no") and bool(plan.get("priority"))
    )
    job = None
    if not user.get("_anonymous"):
        # Team plan: real priority queue (processed before free/pro when workers used)
        pri = 10 if plan.get("priority") else 0
        job = users.create_job(
            user["id"],
            session_id,
            "pipeline",
            org_id=(u or {}).get("org_id"),
            priority=pri,
        )

    # Team / USE_JOB_QUEUE: durable queue (worker must be running)
    if use_queue and job:
        return {
            "ok": True,
            "message": "Pipeline queued for worker",
            "session_id": session_id,
            "job_id": job["id"],
            "queued": True,
            "plan_features": doc["plan_features"],
        }

    async def on_event(payload: dict):
        await broadcast(session_id, {"type": "pipeline", **payload})
        if job:
            users.update_job(
                job["id"],
                status="running" if payload.get("stage") != "done" else "done",
                progress=float(payload.get("progress") or 0),
                message=payload.get("message") or "",
                error=payload.get("data", {}).get("error") if payload.get("stage") == "error" else None,
            )
            if payload.get("stage") == "error":
                users.update_job(job["id"], status="failed")
            if payload.get("stage") == "done":
                from app.services import jobs_queue, sla

                jobs_queue.complete(job["id"], ok=True, message="Done")
                sla.record("pipeline_done", ok=True, session_id=session_id, user_id=user.get("id"))

    await pipeline.start_pipeline(session_id, on_event=on_event)
    return {
        "ok": True,
        "message": "Pipeline started",
        "session_id": session_id,
        "job_id": job["id"] if job else None,
        "queued": False,
        "plan_features": doc["plan_features"],
    }


@router.post("/sessions/{session_id}/coach")
async def recoach(
    session_id: str,
    focus: Optional[str] = Form(None),
    user: dict = Depends(require_user),
):
    from app.services import hit_miss

    try:
        doc = store.load_session(session_id)
    except FileNotFoundError:
        raise HTTPException(404, "Session not found")
    assert_session_owner(doc, user)
    summary = [
        {
            "id": s["id"],
            "t": s["timestamp_s"],
            "result": s["classification"],
            "target": s.get("target_label"),
        }
        for s in doc.get("shots") or []
    ]
    try:
        text = await asyncio.to_thread(hit_miss.coach_session, summary, doc.get("stats") or {}, focus)
        doc["coaching"] = text
        store.save_session(doc)
        return {"coaching": text}
    except Exception as e:
        raise HTTPException(502, f"Coaching failed: {e}")


@router.get("/sessions/{session_id}/media/{kind}/{name:path}")
async def get_media(session_id: str, kind: str, name: str):
    """Serve session media (previews, frames, pointcloud json)."""
    sdir = store.session_dir(session_id)
    # kind: previews | frames | recon | audio | uploads
    if kind not in {"previews", "frames", "recon", "audio", "uploads", "gsplat", "tracks"}:
        raise HTTPException(400, "Invalid media kind")
    path = (sdir / kind / name).resolve()
    if not str(path).startswith(str(sdir.resolve())):
        raise HTTPException(403, "Path escape blocked")
    if not path.exists():
        raise HTTPException(404, "File not found")
    mime, _ = mimetypes.guess_type(str(path))
    return FileResponse(path, media_type=mime or "application/octet-stream")


@router.get("/sessions/{session_id}/tracks")
async def get_tracks(session_id: str, t: Optional[float] = None, camera: str = "primary"):
    """Multi-view target tracks. Optional ?t= seconds returns boxes at that time."""
    try:
        doc = store.load_session(session_id)
    except FileNotFoundError:
        raise HTTPException(404, "Session not found")

    tracks_meta = doc.get("tracks") or {}
    path = tracks_meta.get("path")
    full = None
    if path and Path(path).exists():
        import json as _json

        full = _json.loads(Path(path).read_text())
    elif doc.get("track_timeline"):
        full = {
            "timeline": doc.get("track_timeline"),
            "tracks": {},
            "keyframes": doc.get("track_keyframes") or [],
            "cameras": tracks_meta.get("cameras"),
            "method": tracks_meta.get("method"),
        }

    if full is None:
        # Fallback: shot-level bboxes only
        shot_boxes = []
        for s in doc.get("shots") or []:
            if s.get("bbox"):
                shot_boxes.append(
                    {
                        "t": s["timestamp_s"],
                        "targets": [
                            {
                                "id": s.get("target_id"),
                                "label": s.get("target_label"),
                                "color": s.get("target_color") or "steel",
                                "bbox": s["bbox"],
                                "conf": s.get("confidence") or 0.7,
                                "source": "shot",
                                "classification": s.get("classification"),
                            }
                        ]
                        + list(s.get("visible_targets") or []),
                    }
                )
        if t is not None and shot_boxes:
            best = min(shot_boxes, key=lambda x: abs(x["t"] - t))
            return {
                "t": t,
                "camera": camera,
                "targets": best["targets"] if abs(best["t"] - t) < 1.5 else [],
                "method": "shot_bbox_fallback",
            }
        return {
            "timeline": shot_boxes,
            "method": "shot_bbox_fallback",
            "cameras": tracks_meta.get("cameras"),
        }

    if t is not None:
        from app.services import target_track as tt

        boxes = tt.boxes_at_time(full, float(t), camera=camera)
        # Also blend shot-level engaged target at nearby shots
        for s in doc.get("shots") or []:
            if abs(s.get("timestamp_s", -999) - t) < 0.35 and s.get("bbox"):
                boxes = [
                    {
                        "id": s.get("target_id"),
                        "label": s.get("target_label"),
                        "color": s.get("target_color") or "steel",
                        "bbox": s["bbox"],
                        "conf": s.get("confidence") or 0.85,
                        "source": "shot",
                        "classification": s.get("classification"),
                        "active": True,
                    }
                ] + [b for b in boxes if b.get("id") != s.get("target_id")]
                break
        return {"t": t, "camera": camera, "targets": boxes, "method": full.get("method")}

    return {
        "path": path,
        "method": full.get("method"),
        "cameras": full.get("cameras"),
        "tracks": full.get("tracks"),
        "timeline": full.get("timeline"),
        "keyframes": full.get("keyframes"),
        "summary": tracks_meta,
    }


@router.post("/sessions/{session_id}/tracks/rebuild")
async def rebuild_tracks(session_id: str, user: dict = Depends(require_user)):
    """Re-run multi-view tracking without full pipeline."""
    from app.services import target_track

    try:
        doc = store.load_session(session_id)
    except FileNotFoundError:
        raise HTTPException(404, "Session not found")
    assert_session_owner(doc, user)
    cams = doc.get("cameras") or []
    if not cams:
        raise HTTPException(400, "No cameras")
    primary = Path(cams[0]["path"])
    observer = Path(cams[1]["path"]) if len(cams) > 1 else None
    sdir = store.session_dir(session_id)
    try:
        meta = await asyncio.to_thread(
            target_track.build_session_tracks,
            primary_path=primary,
            observer_path=observer if observer and observer.exists() else None,
            shots=doc.get("shots") or [],
            targets=doc.get("targets") or [],
            out_dir=sdir / "tracks",
            sync_offset_s=float((doc.get("sync") or {}).get("offset_s") or 0.0),
            pointcloud=doc.get("pointcloud"),
            use_vlm=bool(get_settings().openrouter_api_key or get_settings().xai_api_key),
            context=doc.get("name") or "",
        )
        doc["tracks"] = {
            "path": meta.get("path"),
            "method": meta.get("method"),
            "cameras": meta.get("cameras"),
            "track_count_primary": len((meta.get("tracks") or {}).get("primary") or []),
            "track_count_observer": len((meta.get("tracks") or {}).get("observer") or []),
            "keyframe_count": len(meta.get("keyframes") or []),
        }
        doc["track_timeline"] = meta.get("timeline")
        doc["track_keyframes"] = [
            {"t": k["t"], "camera": k["camera"], "targets": k.get("targets") or []}
            for k in (meta.get("keyframes") or [])
        ]
        store.save_session(doc)
        return doc["tracks"]
    except Exception as e:
        raise HTTPException(500, f"Track rebuild failed: {e}")


@router.get("/sessions/{session_id}/pointcloud")
async def get_pointcloud(session_id: str):
    try:
        doc = store.load_session(session_id)
    except FileNotFoundError:
        raise HTTPException(404, "Session not found")
    pc = doc.get("pointcloud")
    if not pc or not pc.get("path"):
        raise HTTPException(404, "Point cloud not ready")
    path = Path(pc["path"])
    if not path.exists():
        raise HTTPException(404, "Point cloud file missing")
    return FileResponse(path, media_type="application/json")


@router.get("/sessions/{session_id}/splat")
async def get_splat(session_id: str):
    """Serve 3D Gaussian Splat PLY / .splat for the God's Eye viewer."""
    try:
        doc = store.load_session(session_id)
    except FileNotFoundError:
        raise HTTPException(404, "Session not found")
    gs = doc.get("gaussian_splat") or {}
    candidates = [
        gs.get("splat_path"),
        gs.get("ply_path"),
        gs.get("path"),
        str(store.session_dir(session_id) / "gsplat" / "splat.ply"),
        str(store.session_dir(session_id) / "gsplat" / "scene.splat"),
    ]
    path = None
    for c in candidates:
        if c and Path(c).exists() and Path(c).stat().st_size > 100:
            path = Path(c)
            break
    if path is None:
        raise HTTPException(404, "Gaussian splat not ready")
    media = "application/octet-stream"
    if path.suffix.lower() == ".ply":
        media = "application/x-ply"
    return FileResponse(path, media_type=media, filename=path.name)


@router.post("/sessions/{session_id}/gsplat")
async def retrain_gsplat(
    session_id: str,
    iters: Optional[int] = None,
    quality: Optional[str] = None,
    user: dict = Depends(require_user),
):
    """Re-run 3DGS training on an existing session's COLMAP recon.

    Query params:
      - iters: override OpenSplat iteration count
      - quality: short|default|high → 600 / config / 7000 iters
    """
    try:
        doc = store.load_session(session_id)
    except FileNotFoundError:
        raise HTTPException(404, "Session not found")
    assert_session_owner(doc, user)
    if not users.plan_allows(user, "gsplat"):
        raise HTTPException(402, "3DGS God's Eye requires Pro or Team. Upgrade to unlock.")
    sdir = store.session_dir(session_id)
    recon_root = sdir / "recon"
    if not recon_root.exists():
        raise HTTPException(400, "Run reconstruction first")
    from app.core.config import get_settings
    from app.services import gaussian_splat

    s = get_settings()
    open_iters = int(s.gsplat_opensplat_iters)
    if quality:
        q = quality.strip().lower()
        if q in {"short", "fast"}:
            open_iters = 600
        elif q in {"high", "long", "max"}:
            open_iters = max(open_iters, 7000)
        elif q in {"default", "medium"}:
            open_iters = int(s.gsplat_opensplat_iters)
    if iters is not None and iters > 0:
        open_iters = int(iters)

    # Mark session as processing gsplat so UI can poll
    doc["status"] = "processing"
    doc["stage"] = "gaussian_splat"
    store.append_log(
        doc,
        "gaussian_splat",
        0.1,
        f"Retrain 3DGS started ({open_iters} iters, prefer OpenSplat Metal)",
        {"iters": open_iters},
    )
    store.save_session(doc)

    try:
        meta = await asyncio.to_thread(
            gaussian_splat.run_gaussian_splatting,
            colmap_workspace=recon_root / "colmap_ws" if (recon_root / "colmap_ws").exists() else recon_root,
            images_dir=recon_root / "colmap_images" if (recon_root / "colmap_images").exists() else recon_root / "frames",
            out_dir=sdir / "gsplat",
            num_iters=open_iters,
            allow_init_fallback=s.gsplat_allow_init_fallback,
            pytorch_iters=s.gsplat_pytorch_iters,
        )
        doc = store.load_session(session_id)
        doc["gaussian_splat"] = {
            "path": meta.get("path"),
            "ply_path": meta.get("ply_path"),
            "splat_path": meta.get("splat_path"),
            "gaussian_count": meta.get("gaussian_count"),
            "method": meta.get("method"),
            "num_iters": meta.get("num_iters"),
            "device": meta.get("device"),
            "opensplat_runtime": meta.get("opensplat_runtime"),
            "metal": meta.get("metal"),
            "final_loss": meta.get("final_loss"),
        }
        doc["status"] = "ready"
        doc["stage"] = "done"
        store.append_log(
            doc,
            "gaussian_splat",
            1.0,
            f"3DGS retrain complete ({meta.get('gaussian_count')} gaussians, {meta.get('method')})",
            {"gaussian_count": meta.get("gaussian_count"), "method": meta.get("method")},
        )
        store.save_session(doc)
        return doc["gaussian_splat"]
    except Exception as e:
        doc = store.load_session(session_id)
        doc["status"] = "ready"  # keep prior session usable
        doc["stage"] = "error"
        store.append_log(doc, "error", 1.0, f"3DGS retrain failed: {e}", {"error": str(e)})
        store.save_session(doc)
        raise HTTPException(500, str(e))


@router.websocket("/ws/{session_id}")
async def ws_session(websocket: WebSocket, session_id: str):
    await websocket.accept()
    _ws_clients.setdefault(session_id, set()).add(websocket)
    try:
        # send current snapshot
        try:
            doc = store.load_session(session_id)
            await websocket.send_json({"type": "snapshot", "session": doc})
        except FileNotFoundError:
            await websocket.send_json({"type": "error", "message": "Session not found"})
        while True:
            # keep alive; client may send pings
            msg = await websocket.receive_text()
            if msg == "ping":
                await websocket.send_json({"type": "pong"})
            elif msg == "refresh":
                try:
                    doc = store.load_session(session_id)
                    await websocket.send_json({"type": "snapshot", "session": doc})
                except FileNotFoundError:
                    pass
    except WebSocketDisconnect:
        pass
    finally:
        _ws_clients.get(session_id, set()).discard(websocket)


@router.websocket("/ws")
async def ws_global(websocket: WebSocket):
    await websocket.accept()
    _ws_global.add(websocket)
    try:
        while True:
            msg = await websocket.receive_text()
            if msg == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    finally:
        _ws_global.discard(websocket)
