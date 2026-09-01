"""IronSight end-to-end pipeline — every stage operates on real media."""
from __future__ import annotations

import asyncio
import traceback
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from app.core.config import get_settings
from app.services import (
    annotations,
    audio_shots,
    dualcam_clean,
    frames,
    gaussian_splat,
    hit_miss,
    pose_track,
    radar_ghost,
    recon_3d,
    store,
    target_track,
)

ProgressCB = Callable[[dict[str, Any]], Awaitable[None] | None]


class Pipeline:
    def __init__(self, session_id: str, on_event: Optional[ProgressCB] = None):
        self.session_id = session_id
        self.on_event = on_event
        self.settings = get_settings()

    async def _emit(self, doc: dict, stage: str, progress: float, message: str, data: Optional[dict] = None):
        store.append_log(doc, stage, progress, message, data)
        store.save_session(doc)
        payload = {
            "session_id": self.session_id,
            "stage": stage,
            "progress": progress,
            "message": message,
            "data": data or {},
            "status": doc.get("status"),
            "stats": doc.get("stats"),
        }
        if self.on_event:
            res = self.on_event(payload)
            if asyncio.iscoroutine(res):
                await res

    async def run(self) -> dict[str, Any]:
        doc = store.load_session(self.session_id)
        doc["status"] = "processing"
        doc["error"] = None
        store.save_session(doc)
        sdir = store.session_dir(self.session_id)

        try:
            await self._emit(doc, "ingest", 0.02, "Validating uploaded cameras")
            cameras = doc.get("cameras") or []
            if not cameras:
                raise RuntimeError("No cameras uploaded. Upload at least one range video.")

            # Clean SxS / strip YouTube HUD + streamer overlay before analysis
            await self._emit(
                doc,
                "dualcam_clean",
                0.03,
                "Cleaning dual-cam feeds (SxS split, HUD crop, streamer mask)",
            )
            try:
                cleaned = await asyncio.to_thread(
                    dualcam_clean.clean_session_uploads, sdir
                )
                doc["dualcam_clean"] = {
                    k: cleaned.get(k)
                    for k in (
                        "ok",
                        "mode",
                        "is_sxs",
                        "streamer_masked",
                        "crop",
                        "report_path",
                    )
                    if k in cleaned or cleaned.get(k) is not None
                }
                if cleaned.get("ok"):
                    new_cams = []
                    sh = cleaned.get("shooter")
                    ob = cleaned.get("observer")
                    if sh and Path(sh).exists():
                        new_cams.append(
                            {
                                "role": "shooter_fpv",
                                "filename": Path(sh).name,
                                "path": str(sh),
                                "duration_s": 0.0,
                                "fps": 30.0,
                                "width": 0,
                                "height": 0,
                                "has_audio": True,
                                "cleaned": True,
                            }
                        )
                    if ob and Path(ob).exists():
                        new_cams.append(
                            {
                                "role": "observer_3p",
                                "filename": Path(ob).name,
                                "path": str(ob),
                                "duration_s": 0.0,
                                "fps": 30.0,
                                "width": 0,
                                "height": 0,
                                "has_audio": True,
                                "cleaned": True,
                            }
                        )
                    if new_cams:
                        cameras = new_cams
                        doc["cameras"] = cameras
                        await self._emit(
                            doc,
                            "dualcam_clean",
                            0.04,
                            f"Clean feeds ready ({cleaned.get('mode')}"
                            f"{', streamer masked' if cleaned.get('streamer_masked') else ''})",
                            doc["dualcam_clean"],
                        )
            except Exception as e:
                doc["dualcam_clean"] = {"ok": False, "error": str(e)}
                await self._emit(
                    doc,
                    "dualcam_clean",
                    0.04,
                    f"Clean skipped: {e}",
                    {"error": str(e)},
                )

            primary = cameras[0]
            primary_path = Path(primary["path"])
            if not primary_path.exists():
                raise RuntimeError(f"Primary video missing: {primary_path}")

            # Probe + preview
            meta = frames.probe_video(primary_path)
            primary.update(
                {
                    "duration_s": meta["duration_s"],
                    "fps": meta["fps"],
                    "width": meta["width"],
                    "height": meta["height"],
                }
            )
            preview = sdir / "previews" / f"{primary['role']}_preview.mp4"
            try:
                frames.transcode_preview(primary_path, preview)
                primary["preview"] = str(preview)
            except Exception as e:
                primary["preview_error"] = str(e)

            # Second camera
            observer = cameras[1] if len(cameras) > 1 else None
            if observer:
                om = frames.probe_video(observer["path"])
                observer.update(
                    {
                        "duration_s": om["duration_s"],
                        "fps": om["fps"],
                        "width": om["width"],
                        "height": om["height"],
                    }
                )
                op = sdir / "previews" / f"{observer['role']}_preview.mp4"
                try:
                    frames.transcode_preview(observer["path"], op)
                    observer["preview"] = str(op)
                except Exception as e:
                    observer["preview_error"] = str(e)

            doc["cameras"] = cameras
            doc["stats"]["duration_s"] = float(meta["duration_s"])
            await self._emit(
                doc,
                "audio_extract",
                0.08,
                "Step 1/5 · Listening for bangs (audio)",
                {"step": 1, "step_label": "Listening", "steps_total": 5},
            )

            audio_primary = sdir / "audio" / "primary.wav"
            await asyncio.to_thread(
                audio_shots.extract_audio_from_video, primary_path, audio_primary
            )

            audio_observer = None
            if observer:
                audio_observer = sdir / "audio" / "observer.wav"
                await asyncio.to_thread(
                    audio_shots.extract_audio_from_video, observer["path"], audio_observer
                )

            await self._emit(
                doc,
                "shot_detection",
                0.18,
                "Step 1/5 · Finding shot times",
                {"step": 1, "step_label": "Finding shots", "steps_total": 5},
            )
            profile = getattr(self.settings, "shot_profile", "auto") or "auto"
            detected = await asyncio.to_thread(
                audio_shots.detect_shots,
                audio_primary,
                max_shots=self.settings.shot_max_per_session,
                profile=profile,  # type: ignore[arg-type]
            )
            await self._emit(
                doc,
                "shot_detection",
                0.28,
                f"Detected {len(detected)} shot events (profile={profile})",
                {
                    "count": len(detected),
                    "profile": profile,
                    "scores": [round(s.score, 2) for s in detected[:20]],
                },
            )

            # Dual-cam sync
            if audio_observer is not None:
                await self._emit(
                    doc,
                    "dual_cam_sync",
                    0.32,
                    "Step 2/5 · Locking both cameras in time",
                    {"step": 2, "step_label": "Sync cameras", "steps_total": 5},
                )
                sync = await asyncio.to_thread(
                    audio_shots.dual_cam_audio_sync, audio_primary, audio_observer
                )
                doc["sync"] = sync
                await self._emit(
                    doc,
                    "dual_cam_sync",
                    0.38,
                    f"Sync offset {sync['offset_s']:.4f}s (ratio {sync['peak_ratio']:.2f})",
                    sync,
                )
            else:
                doc["sync"] = {
                    "offset_s": 0.0,
                    "offset_frames": 0.0,
                    "peak_ratio": 1.0,
                    "passed": True,
                    "method": "single_camera",
                    "details": {},
                }

            # Hit/miss classification
            shots: list[dict[str, Any]] = []
            targets_map: dict[str, dict[str, Any]] = {}
            n = max(1, len(detected))
            await self._emit(
                doc,
                "hit_miss",
                0.40,
                f"Step 3/5 · Scoring {len(detected)} shots (hit / miss)",
                {"step": 3, "step_label": "Scoring shots", "steps_total": 5},
            )
            for i, det in enumerate(detected):
                prog = 0.40 + 0.35 * (i / n)
                await self._emit(
                    doc,
                    "hit_miss",
                    prog,
                    f"Step 3/5 · Shot {i + 1}/{len(detected)} — checking hit or miss",
                    {"step": 3, "shot": i + 1, "shots_total": len(detected)},
                )
                frame_dir = sdir / "frames" / f"shot_{i+1:03d}"
                frame_paths = await asyncio.to_thread(
                    frames.extract_shot_frames,
                    primary_path,
                    det.timestamp_s,
                    frame_dir,
                    prefix=f"s{i+1}",
                    pre_s=self.settings.hit_miss_pre_s,
                    post_s=self.settings.hit_miss_post_s,
                    count=self.settings.hit_miss_frame_count,
                )
                # Observer burst (more 3P frames → better HIT/MISS evidence)
                if observer:
                    obs_t = det.timestamp_s + float(doc["sync"].get("offset_s") or 0.0)
                    for j, dt in enumerate((-0.08, 0.0, 0.12, 0.28)):
                        try:
                            op = await asyncio.to_thread(
                                frames.extract_frame_at,
                                observer["path"],
                                max(0.0, obs_t + dt),
                                frame_dir / f"obs_{i+1:03d}_{j:02d}.jpg",
                            )
                            frame_paths.append(op)
                        except Exception:
                            pass

                try:
                    adj = await asyncio.to_thread(
                        hit_miss.classify_shot,
                        frame_paths,
                        shot_index=i + 1,
                        timestamp_s=det.timestamp_s,
                        context=doc.get("name"),
                        retry_on_unknown=True,
                    )
                except Exception as e:
                    adj = {
                        "classification": "UNKNOWN",
                        "confidence": 0.0,
                        "target_id": None,
                        "target_label": None,
                        "target_color": None,
                        "reasoning": f"VLM error: {e}",
                        "model": None,
                        "needs_review": True,
                    }

                cls = adj["classification"]
                raw_tid = adj.get("target_id") or adj.get("target_label")
                tlabel = adj.get("target_label") or (str(raw_tid) if raw_tid else None)
                tcolor = (adj.get("target_color") or "steel") or "steel"
                tid = str(raw_tid) if raw_tid else None

                # Only register targets for adjudicated HIT/MISS with an identity
                # NOT_A_SHOT = audio false positive (shown in log, excluded from accuracy)
                if cls in ("HIT", "MISS"):
                    if not tid:
                        tid = f"T{i+1}"
                        tlabel = tlabel or tid
                    if tid not in targets_map:
                        targets_map[tid] = {
                            "id": str(tid),
                            "label": str(tlabel or tid),
                            "color": str(tcolor),
                            "kind": "plate",
                            "hits": 0,
                            "misses": 0,
                            "last_result": None,
                            "position_3d": None,
                        }
                    if tlabel:
                        targets_map[tid]["label"] = str(tlabel)
                    if cls == "HIT":
                        targets_map[tid]["hits"] += 1
                        targets_map[tid]["last_result"] = "HIT"
                    else:
                        targets_map[tid]["misses"] += 1
                        targets_map[tid]["last_result"] = "MISS"

                shots.append(
                    {
                        "id": i + 1,
                        "timestamp_s": det.timestamp_s,
                        "duration_s": det.duration_s,
                        "energy": det.energy,
                        "peak_db": det.peak_db,
                        "classification": cls,
                        "confidence": adj.get("confidence") or 0.0,
                        "needs_review": bool(adj.get("needs_review")),
                        "target_id": tid,
                        "target_label": tlabel,
                        "target_color": str(tcolor) if tid else None,
                        "bbox": adj.get("bbox"),  # engaged target, normalized xywh
                        "bbox_source": adj.get("bbox_source"),
                        "visible_targets": adj.get("targets") or [],
                        "reasoning": adj.get("reasoning"),
                        "frame_paths": [str(p) for p in frame_paths],
                        "position_3d": None,
                        "camera": "primary",
                    }
                )
                # Incremental save so UI can stream results live
                doc["shots"] = shots
                doc["targets"] = list(targets_map.values())
                doc["stats"] = self._compute_stats(shots, meta["duration_s"])
                store.save_session(doc)

            doc["shots"] = shots
            doc["targets"] = list(targets_map.values())
            doc["stats"] = self._compute_stats(shots, meta["duration_s"])
            await self._emit(doc, "target_map", 0.78, f"Mapped {len(targets_map)} targets")

            # 3D reconstruction (Pro+ — free skips expensive COLMAP)
            plan_feats = doc.get("plan_features") or {}
            allow_recon = plan_feats.get("recon", True)
            if allow_recon is False:
                await self._emit(
                    doc,
                    "reconstruction_3d",
                    0.92,
                    "3D recon skipped (upgrade to Pro for COLMAP + God's Eye)",
                )
            prefer_colmap = bool(getattr(self.settings, "recon_prefer_colmap", True))
            colmap_on = recon_3d.colmap_available()
            if allow_recon is not False:
                await self._emit(
                    doc,
                    "reconstruction_3d",
                    0.80,
                    f"Sampling frames for recon (COLMAP={'yes' if colmap_on and prefer_colmap else 'fallback OpenCV dense'})",
                )
                recon_frames_dir = sdir / "recon" / "frames"
                recon_img_paths = await asyncio.to_thread(
                    frames.sample_frames_for_recon,
                    primary_path,
                    recon_frames_dir,
                    max_frames=self.settings.recon_max_frames,
                    sample_fps=self.settings.recon_sample_fps,
                )
                await self._emit(
                    doc,
                    "reconstruction_3d",
                    0.85,
                    f"Reconstructing scene from {len(recon_img_paths)} frames",
                )
                try:
                    pc_meta = await asyncio.to_thread(
                        recon_3d.reconstruct_from_images,
                        recon_img_paths,
                        sdir / "recon",
                        max_features=self.settings.recon_max_features,
                        point_budget=self.settings.recon_point_budget,
                        prefer_colmap=prefer_colmap,
                        colmap_dense=bool(getattr(self.settings, "recon_colmap_dense", True)),
                    )
                    placed = recon_3d.place_targets_in_scene(pc_meta, doc["targets"])
                    doc["targets"] = placed
                    # Assign shot 3D positions near their targets
                    tpos = {t["id"]: t.get("position_3d") for t in placed}
                    for sh in doc["shots"]:
                        sh["position_3d"] = tpos.get(sh.get("target_id"))
                    doc["pointcloud"] = {
                        "path": pc_meta["path"],
                        "point_count": pc_meta["point_count"],
                        "bounds_min": pc_meta["bounds_min"],
                        "bounds_max": pc_meta["bounds_max"],
                        "camera_poses": pc_meta.get("camera_poses") or [],
                        "method": pc_meta.get("method"),
                    }
                    await self._emit(
                        doc,
                        "reconstruction_3d",
                        0.92,
                        f"Point cloud ready ({pc_meta['point_count']} points, {pc_meta.get('method')})",
                        {"point_count": pc_meta["point_count"], "method": pc_meta.get("method")},
                    )
                except Exception as e:
                    await self._emit(
                        doc,
                        "reconstruction_3d",
                        0.92,
                        f"Reconstruction warning: {e}",
                        {"error": str(e)},
                    )

            # 3D Gaussian Splatting (God's Eye View photoreal layer)
            allow_gsplat = bool(getattr(self.settings, "gsplat_enabled", True))
            if plan_feats and plan_feats.get("gsplat") is False:
                allow_gsplat = False
                await self._emit(
                    doc,
                    "gaussian_splat",
                    0.93,
                    "3DGS skipped (upgrade to Pro for God's Eye Gaussian Splats)",
                )
            if allow_gsplat and allow_recon is not False:
                await self._emit(
                    doc,
                    "gaussian_splat",
                    0.93,
                    "Training 3D Gaussian Splatting (OpenSplat / PyTorch MPS)",
                )
                try:
                    recon_root = sdir / "recon"
                    colmap_ws = recon_root / "colmap_ws"
                    images_dir = recon_root / "colmap_images"
                    if not images_dir.exists():
                        images_dir = recon_root / "frames"
                    # Load points from pointcloud json if present
                    pts = cols = None
                    pc_path = (doc.get("pointcloud") or {}).get("path")
                    if pc_path and Path(pc_path).exists():
                        import json as _json

                        pc = _json.loads(Path(pc_path).read_text())
                        pos = pc.get("positions") or []
                        clr = pc.get("colors") or []
                        if pos:
                            import numpy as _np

                            pts = _np.array(pos, dtype=_np.float64).reshape(-1, 3)
                            if clr:
                                cols = (_np.array(clr, dtype=_np.float64).reshape(-1, 3) * 255.0)
                            else:
                                cols = _np.full_like(pts, 180.0)

                    gs_meta = await asyncio.to_thread(
                        gaussian_splat.run_gaussian_splatting,
                        colmap_workspace=colmap_ws if colmap_ws.exists() else recon_root,
                        images_dir=images_dir,
                        out_dir=sdir / "gsplat",
                        points_xyz=pts,
                        colors_rgb=cols,
                        num_iters=int(getattr(self.settings, "gsplat_opensplat_iters", 3000)),
                        allow_init_fallback=bool(
                            getattr(self.settings, "gsplat_allow_init_fallback", True)
                        ),
                        pytorch_iters=int(getattr(self.settings, "gsplat_pytorch_iters", 300)),
                    )
                    doc["gaussian_splat"] = {
                        "path": gs_meta.get("path"),
                        "ply_path": gs_meta.get("ply_path"),
                        "splat_path": gs_meta.get("splat_path"),
                        "gaussian_count": gs_meta.get("gaussian_count"),
                        "method": gs_meta.get("method"),
                        "num_iters": gs_meta.get("num_iters"),
                        "device": gs_meta.get("device"),
                        "final_loss": gs_meta.get("final_loss"),
                    }
                    await self._emit(
                        doc,
                        "gaussian_splat",
                        0.96,
                        f"3DGS ready ({gs_meta.get('gaussian_count')} gaussians, {gs_meta.get('method')})",
                        {
                            "gaussian_count": gs_meta.get("gaussian_count"),
                            "method": gs_meta.get("method"),
                        },
                    )
                except Exception as e:
                    doc["gaussian_splat"] = None
                    await self._emit(
                        doc,
                        "gaussian_splat",
                        0.96,
                        f"3DGS warning: {e}",
                        {"error": str(e)},
                    )

            # Through-wall radar occupancy volume (for Ghost-X radar mode)
            if doc.get("pointcloud") and (doc.get("pointcloud") or {}).get("path"):
                await self._emit(
                    doc,
                    "radar_ghost",
                    0.962,
                    "Building through-wall radar occupancy volume",
                )
                try:
                    radar = await asyncio.to_thread(
                        radar_ghost.build_radar_volume,
                        Path(doc["pointcloud"]["path"]),
                        doc.get("targets") or [],
                        out_path=sdir / "recon" / "radar_volume.json",
                    )
                    doc["radar"] = {
                        "path": radar.get("path"),
                        "voxel_count": radar.get("voxel_count"),
                        "grid": radar.get("grid"),
                        "method": radar.get("method"),
                        "returns": len(radar.get("returns") or []),
                    }
                    await self._emit(
                        doc,
                        "radar_ghost",
                        0.964,
                        f"Radar volume ready ({radar.get('voxel_count')} voxels, "
                        f"{doc['radar']['returns']} returns)",
                        doc["radar"],
                    )
                except Exception as e:
                    doc["radar"] = {"error": str(e)}
                    await self._emit(
                        doc, "radar_ghost", 0.964, f"Radar warning: {e}", {"error": str(e)}
                    )

            # Multi-view target tracks (YOLO + CV + VLM + LK + 3D projection)
            await self._emit(
                doc,
                "target_track",
                0.965,
                "Building multi-view target tracks (YOLO + CV + VLM + LK)",
            )
            try:
                tracks_meta = await asyncio.to_thread(
                    target_track.build_session_tracks,
                    primary_path=primary_path,
                    observer_path=Path(observer["path"]) if observer else None,
                    shots=doc["shots"],
                    targets=doc.get("targets") or [],
                    out_dir=sdir / "tracks",
                    sync_offset_s=float((doc.get("sync") or {}).get("offset_s") or 0.0),
                    pointcloud=doc.get("pointcloud"),
                    use_vlm=bool(self.settings.openrouter_api_key or self.settings.xai_api_key),
                    context=doc.get("name") or "",
                )
                doc["tracks"] = {
                    "path": tracks_meta.get("path"),
                    "method": tracks_meta.get("method"),
                    "cameras": tracks_meta.get("cameras"),
                    "track_count_primary": len((tracks_meta.get("tracks") or {}).get("primary") or []),
                    "track_count_observer": len((tracks_meta.get("tracks") or {}).get("observer") or []),
                    "keyframe_count": len(tracks_meta.get("keyframes") or []),
                }
                # Keep full timeline in session for UI (may be large but needed for HUD)
                doc["track_timeline"] = tracks_meta.get("timeline")
                doc["track_keyframes"] = [
                    {
                        "t": k["t"],
                        "camera": k["camera"],
                        "targets": k.get("targets") or [],
                    }
                    for k in (tracks_meta.get("keyframes") or [])
                ]
                await self._emit(
                    doc,
                    "target_track",
                    0.968,
                    f"Tracks ready · primary={doc['tracks']['track_count_primary']} "
                    f"observer={doc['tracks']['track_count_observer']} "
                    f"keyframes={doc['tracks']['keyframe_count']}",
                    doc["tracks"],
                )
            except Exception as e:
                doc["tracks"] = {"error": str(e)}
                await self._emit(
                    doc,
                    "target_track",
                    0.968,
                    f"Track warning: {e}",
                    {"error": str(e)},
                )

            # Pose / stick-figure track for 4D God's Eye
            plan_feats = doc.get("plan_features") or {}
            if plan_feats.get("pose") is False:
                await self._emit(
                    doc, "pose_track", 0.97, "Pose skipped (upgrade to Pro for stick-figure 4D)"
                )
            else:
                await self._emit(doc, "pose_track", 0.969, "Extracting multi-view pose track")
                try:
                    pose_meta = await asyncio.to_thread(
                        pose_track.build_session_pose,
                        video_path=primary_path,
                        pointcloud=doc.get("pointcloud"),
                        duration_s=float(doc.get("stats", {}).get("duration_s") or meta["duration_s"]),
                        out_path=sdir / "pose" / "pose_track.json",
                        sample_fps=3.0,
                        observer_path=Path(observer["path"]) if observer else None,
                        sync_offset_s=float((doc.get("sync") or {}).get("offset_s") or 0.0),
                    )
                    doc["pose"] = {
                        "path": pose_meta.get("path"),
                        "method": pose_meta.get("method"),
                        "sample_count": pose_meta.get("sample_count"),
                        "multiview_frames": pose_meta.get("multiview_frames"),
                        "dual_cam_calibrated": pose_meta.get("dual_cam_calibrated"),
                    }
                    await self._emit(
                        doc,
                        "pose_track",
                        0.97,
                        f"Pose track ready ({pose_meta.get('sample_count')} frames, {pose_meta.get('method')})",
                        doc["pose"],
                    )
                except Exception as e:
                    doc["pose"] = {"error": str(e)}
                    await self._emit(doc, "pose_track", 0.97, f"Pose warning: {e}", {"error": str(e)})

            # Coaching (Pro+)
            allow_coach = plan_feats.get("coaching", True)
            if allow_coach is False:
                doc["coaching"] = None
                doc["coaching_locked"] = True
                await self._emit(
                    doc,
                    "coaching",
                    0.972,
                    "Full coaching debrief is Pro — upgrade to unlock coach notes",
                )
            else:
                await self._emit(doc, "coaching", 0.972, "Generating post-run coaching debrief")
                try:
                    summary = [
                        {
                            "id": s["id"],
                            "t": s["timestamp_s"],
                            "result": s["classification"],
                            "target": s.get("target_label"),
                            "confidence": s.get("confidence"),
                            "needs_review": s.get("needs_review"),
                        }
                        for s in doc["shots"]
                    ]
                    coaching = await asyncio.to_thread(
                        hit_miss.coach_session, summary, doc["stats"]
                    )
                    doc["coaching"] = coaching
                    doc["coaching_locked"] = False
                except Exception as e:
                    doc["coaching"] = f"Coaching unavailable: {e}"

            # Session wrap card (Spotify-wrap style)
            try:
                wrap = await asyncio.to_thread(
                    annotations.build_wrap_card,
                    self.session_id,
                    export=bool(plan_feats.get("wrap_export", True)),
                )
                doc["wrap"] = {
                    "path": wrap.get("path"),
                    "headline": wrap.get("headline"),
                    "share_line": wrap.get("share_line"),
                    "export_allowed": wrap.get("export_allowed"),
                }
            except Exception as e:
                doc["wrap"] = {"error": str(e)}

            await self._emit(
                doc,
                "finalize",
                0.98,
                "Step 5/5 · Wrapping up — next: fix unsure shots then Seal",
                {"step": 5, "step_label": "Done", "steps_total": 5, "open_review": True},
            )
            doc["status"] = "ready"
            doc["stage"] = "done"
            doc["needs_coach_review"] = True
            store.save_session(doc)
            await self._emit(
                doc,
                "done",
                1.0,
                "Ready — fix unsure shots, then Seal as coach-ready",
                {
                    "stats": doc["stats"],
                    "open_review": True,
                    "step": 5,
                    "steps_total": 5,
                },
            )
            return doc

        except Exception as e:
            doc = store.load_session(self.session_id)
            doc["status"] = "failed"
            doc["stage"] = "error"
            doc["error"] = f"{e}\n{traceback.format_exc()[-1500:]}"
            store.save_session(doc)
            await self._emit(doc, "error", 1.0, str(e), {"error": doc["error"]})
            raise

    @staticmethod
    def _compute_stats(shots: list[dict], duration_s: float) -> dict[str, Any]:
        from app.services.scoring import compute_stats

        return compute_stats(shots, duration_s)


# In-process task registry
_running: dict[str, asyncio.Task] = {}


def is_running(session_id: str) -> bool:
    t = _running.get(session_id)
    return t is not None and not t.done()


async def start_pipeline(session_id: str, on_event: Optional[ProgressCB] = None) -> asyncio.Task:
    if is_running(session_id):
        return _running[session_id]

    pipe = Pipeline(session_id, on_event=on_event)

    async def _runner():
        try:
            await pipe.run()
        finally:
            _running.pop(session_id, None)

    task = asyncio.create_task(_runner())
    _running[session_id] = task
    return task
