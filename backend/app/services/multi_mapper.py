"""Multi-mapper experiment board (video: overnight COLMAP vs Pi3 vs lingbot-style R&D).

Runs several reconstruction pipelines, sim3-aligns trajectories to a reference
(COLMAP medium when available), scores them, picks primary + fallback.

No API keys. Methods:
- colmap_low / colmap_medium / colmap_high (quality flags)
- opencv_sift (dense pair SfM fallback)

Pi3 / lingbot-map are not open/installable here; we document them as external
slots and score whatever local engines we have.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np

from app.services import frames, recon_3d


ProgressCB = Optional[Callable[[str, float, str], None]]


def _emit(cb: ProgressCB, stage: str, p: float, msg: str):
    if cb:
        cb(stage, p, msg)


def run_experiment_board(
    video_path: Path,
    out_root: Path,
    *,
    max_frames: int = 48,
    sample_fps: float = 2.5,
    on_progress: ProgressCB = None,
    methods: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Run multi-mapper experiments on sampled frames from video."""
    out_root.mkdir(parents=True, exist_ok=True)
    img_dir = out_root / "images"
    if img_dir.exists():
        shutil.rmtree(img_dir)
    img_dir.mkdir(parents=True)

    _emit(on_progress, "sample", 0.05, "Sampling frames for multi-mapper board")
    paths = frames.sample_frames_for_recon(
        video_path, img_dir, max_frames=max_frames, sample_fps=sample_fps
    )
    if len(paths) < 4:
        raise RuntimeError(f"Need ≥4 frames, got {len(paths)}")

    catalog = methods or [
        "colmap_low",
        "colmap_medium",
        "opencv_sift",
    ]
    # only schedule colmap_high if colmap exists and user has time — optional
    if recon_3d.colmap_available() and "colmap_high" in (methods or []):
        pass

    results: list[dict[str, Any]] = []
    n = len(catalog)
    for i, mid in enumerate(catalog):
        p0 = 0.1 + 0.7 * (i / max(1, n))
        _emit(on_progress, mid, p0, f"Running mapper {mid}")
        t0 = time.time()
        exp_dir = out_root / mid
        exp_dir.mkdir(exist_ok=True)
        try:
            meta = _run_one(mid, paths, exp_dir)
            meta["elapsed_s"] = round(time.time() - t0, 2)
            meta["status"] = "ok" if meta.get("point_count", 0) > 20 else "weak"
            results.append(meta)
            _emit(
                on_progress,
                mid,
                p0 + 0.05,
                f"{mid}: {meta.get('point_count')} pts · {meta.get('elapsed_s')}s",
            )
        except Exception as e:
            results.append(
                {
                    "id": mid,
                    "status": "fail",
                    "error": str(e)[:500],
                    "elapsed_s": round(time.time() - t0, 2),
                    "trajectory": [],
                    "point_count": 0,
                }
            )
            _emit(on_progress, mid, p0 + 0.05, f"{mid} failed: {e}")

    # External placeholders (not run — documented for board)
    results.append(
        {
            "id": "pi3",
            "status": "external",
            "note": "Pi3 feed-forward mapper — research; not bundled. Slot for comparison when binary available.",
            "trajectory": [],
            "point_count": 0,
        }
    )
    results.append(
        {
            "id": "lingbot_map",
            "status": "external",
            "note": "lingbot-map real-time MLX mapper — research; not bundled. Slot for comparison.",
            "trajectory": [],
            "point_count": 0,
        }
    )

    # Pick reference = first successful colmap_* or best point count
    ref = None
    for r in results:
        if r.get("status") in ("ok", "weak") and r["id"].startswith("colmap") and r.get("trajectory"):
            ref = r
            break
    if ref is None:
        cands = [r for r in results if r.get("trajectory") and r.get("status") in ("ok", "weak")]
        if cands:
            ref = max(cands, key=lambda r: r.get("point_count") or 0)

    # Sim3-align trajectories to ref and score
    comparisons = []
    if ref and ref.get("trajectory"):
        T_ref = np.array(ref["trajectory"], dtype=np.float64)
        for r in results:
            if r.get("id") == ref["id"]:
                r["aligned_trajectory"] = r["trajectory"]
                r["rmse"] = 0.0
                r["drift"] = 0.0
                r["gate"] = "PASS (reference)"
                comparisons.append(
                    {
                        "id": r["id"],
                        "rmse": 0.0,
                        "drift": 0.0,
                        "gate": "PASS (reference)",
                        "point_count": r.get("point_count"),
                    }
                )
                continue
            if not r.get("trajectory") or len(r["trajectory"]) < 3:
                r["gate"] = "FAIL (no traj)"
                comparisons.append(
                    {
                        "id": r["id"],
                        "rmse": None,
                        "drift": None,
                        "gate": r["gate"],
                        "point_count": r.get("point_count"),
                        "status": r.get("status"),
                        "note": r.get("note") or r.get("error"),
                    }
                )
                continue
            T = np.array(r["trajectory"], dtype=np.float64)
            aligned, rmse, drift = sim3_align_trajectories(T_ref, T)
            r["aligned_trajectory"] = aligned.tolist()
            r["rmse"] = float(rmse)
            r["drift"] = float(drift)
            # gates like the video board
            if r.get("status") == "fail":
                gate = "FAIL"
            elif rmse < 0.5 and drift < 0.15:
                gate = "PASS"
            elif rmse < 1.5 and drift < 0.35:
                gate = "PASS (throughput)"
            else:
                gate = "FAIL (drift)"
            r["gate"] = gate
            comparisons.append(
                {
                    "id": r["id"],
                    "rmse": float(rmse),
                    "drift": float(drift),
                    "gate": gate,
                    "point_count": r.get("point_count"),
                    "elapsed_s": r.get("elapsed_s"),
                }
            )

    # Primary = best PASS with most points among local engines
    primary = ref["id"] if ref else None
    fallback = None
    locals_ok = [
        r
        for r in results
        if r.get("gate", "").startswith("PASS") and r.get("id") not in ("pi3", "lingbot_map")
    ]
    if locals_ok:
        locals_ok.sort(key=lambda r: (-(r.get("point_count") or 0), r.get("rmse") or 99))
        primary = locals_ok[0]["id"]
        if len(locals_ok) > 1:
            fallback = locals_ok[1]["id"]

    board = {
        "methods": results,
        "comparisons": comparisons,
        "reference": ref["id"] if ref else None,
        "primary": primary,
        "fallback": fallback,
        "image_count": len(paths),
        "topdown": _topdown_payload(results, ref["id"] if ref else None),
        "summary": (
            f"Primary={primary}, fallback={fallback}. "
            f"Local engines compared; Pi3/lingbot-map are external research slots."
        ),
    }
    (out_root / "board.json").write_text(json.dumps(board, indent=2))
    board["path"] = str(out_root / "board.json")
    _emit(on_progress, "done", 1.0, board["summary"])
    return board


def _run_one(mid: str, image_paths: list[Path], exp_dir: Path) -> dict[str, Any]:
    if mid.startswith("colmap"):
        if not recon_3d.colmap_available():
            raise RuntimeError("COLMAP not installed")
        quality = mid.split("_", 1)[-1]  # low|medium|high
        if quality not in ("low", "medium", "high"):
            quality = "medium"
        return _run_colmap(image_paths, exp_dir, quality=quality)
    if mid == "opencv_sift":
        meta = recon_3d.reconstruct_from_images(
            image_paths,
            exp_dir,
            max_features=5000,
            point_budget=50_000,
            prefer_colmap=False,
        )
        traj = []
        for p in meta.get("camera_poses") or []:
            pos = p.get("position")
            if not pos:
                continue
            if isinstance(pos, dict):
                traj.append([pos["x"], pos["y"], pos["z"]])
            else:
                traj.append([float(pos[0]), float(pos[1]), float(pos[2])])
        return {
            "id": mid,
            "method": meta.get("method"),
            "point_count": meta.get("point_count"),
            "trajectory": traj,
            "path": meta.get("path"),
        }
    raise RuntimeError(f"Unknown mapper {mid}")


def _run_colmap(image_paths: list[Path], exp_dir: Path, quality: str = "medium") -> dict[str, Any]:
    """Run COLMAP automatic reconstructor at a quality tier, then export cloud+traj."""
    img_dir = exp_dir / "images"
    img_dir.mkdir(exist_ok=True)
    staged = []
    for i, p in enumerate(image_paths):
        dst = img_dir / f"{i:05d}{Path(p).suffix or '.jpg'}"
        if not dst.exists():
            shutil.copy2(p, dst)
        staged.append(dst)

    ws = exp_dir / "colmap_ws"
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir()
    cmd = [
        "colmap",
        "automatic_reconstructor",
        "--image_path",
        str(img_dir),
        "--workspace_path",
        str(ws),
        "--quality",
        quality,
        "--dense",
        "0",
        "--use_gpu",
        "0",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    if r.returncode != 0:
        # Fall back to OpenCV inside this slot rather than hard-fail the board
        meta = recon_3d.reconstruct_from_images(
            staged, exp_dir / "export", prefer_colmap=False, point_budget=40_000
        )
        traj = _traj_from_meta(meta)
        return {
            "id": f"colmap_{quality}",
            "method": f"colmap_{quality}_fallback_{meta.get('method')}",
            "point_count": meta.get("point_count"),
            "trajectory": traj,
            "path": meta.get("path"),
            "warning": (r.stderr or r.stdout)[-300:],
        }

    # Export using recon_3d COLMAP path on the same images (reuses installer helpers)
    meta = recon_3d.reconstruct_from_images(
        staged,
        exp_dir / "export",
        prefer_colmap=True,
        colmap_dense=False,
        point_budget=80_000,
    )
    traj = _traj_from_meta(meta)
    return {
        "id": f"colmap_{quality}",
        "method": f"colmap_{quality}",
        "point_count": meta.get("point_count"),
        "trajectory": traj,
        "path": meta.get("path"),
        "workspace": str(ws),
    }


def _traj_from_meta(meta: dict) -> list[list[float]]:
    traj = []
    for p in meta.get("camera_poses") or []:
        pos = p.get("position")
        if not pos:
            continue
        if isinstance(pos, dict):
            traj.append([float(pos["x"]), float(pos["y"]), float(pos["z"])])
        else:
            traj.append([float(pos[0]), float(pos[1]), float(pos[2])])
    return traj


def sim3_align_trajectories(
    ref: np.ndarray, src: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """Umeyama Sim3 align src→ref. Returns aligned src, RMSE, drift fraction."""
    # Resample to same length
    n = min(len(ref), len(src), 200)
    if n < 3:
        return src, 99.0, 1.0
    R = _resample(ref, n)
    S = _resample(src, n)
    # Umeyama
    mu_R = R.mean(axis=0)
    mu_S = S.mean(axis=0)
    R0 = R - mu_R
    S0 = S - mu_S
    cov = (S0.T @ R0) / n
    U, D, Vt = np.linalg.svd(cov)
    Svd = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        Svd[-1, -1] = -1
    rot = U @ Svd @ Vt
    var_S = (S0**2).sum() / n
    scale = float(np.trace(np.diag(D) @ Svd) / (var_S + 1e-12))
    trans = mu_R - scale * rot @ mu_S
    aligned = (scale * (rot @ S.T)).T + trans
    err = np.linalg.norm(aligned - R, axis=1)
    rmse = float(np.sqrt(np.mean(err**2)))
    extent = float(np.linalg.norm(R.max(axis=0) - R.min(axis=0)) + 1e-6)
    drift = float(err[-1] / extent)
    return aligned, rmse, drift


def _resample(T: np.ndarray, n: int) -> np.ndarray:
    if len(T) == n:
        return T.astype(np.float64)
    idx = np.linspace(0, len(T) - 1, n)
    out = []
    for i in idx:
        lo = int(np.floor(i))
        hi = min(len(T) - 1, lo + 1)
        a = i - lo
        out.append((1 - a) * T[lo] + a * T[hi])
    return np.array(out, dtype=np.float64)


def _topdown_payload(results: list[dict], ref_id: Optional[str]) -> dict[str, Any]:
    """2D XY polylines for frontend top-down plot."""
    series = []
    for r in results:
        traj = r.get("aligned_trajectory") or r.get("trajectory") or []
        if len(traj) < 2:
            continue
        series.append(
            {
                "id": r["id"],
                "gate": r.get("gate"),
                "points": [[float(p[0]), float(p[2] if len(p) > 2 else p[1])] for p in traj],
                "is_ref": r["id"] == ref_id,
            }
        )
    return {"series": series, "axes": "x-z top-down"}
