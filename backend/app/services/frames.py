"""Video I/O helpers — real frames only, via OpenCV / ffmpeg."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def probe_video(path: str | Path) -> dict:
    path = str(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration = n / fps if fps > 0 else 0.0
    return {
        "path": path,
        "fps": fps,
        "frame_count": n,
        "width": w,
        "height": h,
        "duration_s": duration,
    }


def extract_frame_at(path: str | Path, t_s: float, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_idx = max(0, int(round(t_s * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        # fallback seek by msec
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_s * 1000.0))
        ok, frame = cap.read()
        cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame at t={t_s:.3f}s from {path}")
    cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return out_path


def extract_shot_frames(
    video_path: str | Path,
    timestamp_s: float,
    out_dir: str | Path,
    *,
    prefix: str = "shot",
    pre_s: float = 0.15,
    post_s: float = 0.55,
    count: int = 4,
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    times = np.linspace(timestamp_s - pre_s, timestamp_s + post_s, count)
    times = [max(0.0, float(t)) for t in times]
    paths: list[Path] = []
    for i, t in enumerate(times):
        p = out_dir / f"{prefix}_{timestamp_s:.3f}_{i:02d}.jpg"
        try:
            extract_frame_at(video_path, t, p)
            paths.append(p)
        except RuntimeError:
            continue
    return paths


def sample_frames_for_recon(
    video_path: str | Path,
    out_dir: str | Path,
    *,
    max_frames: int = 48,
    sample_fps: float = 2.0,
) -> list[Path]:
    """Sample evenly-spaced frames for structure-from-motion."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = probe_video(video_path)
    duration = meta["duration_s"]
    if duration <= 0:
        return []
    n = min(max_frames, max(8, int(duration * sample_fps)))
    times = np.linspace(0.05 * duration, 0.95 * duration, n)
    paths: list[Path] = []
    for i, t in enumerate(times):
        p = out_dir / f"recon_{i:04d}.jpg"
        try:
            extract_frame_at(video_path, float(t), p)
            paths.append(p)
        except RuntimeError:
            continue
    return paths


def transcode_preview(video_path: str | Path, out_path: str | Path, max_w: int = 1280) -> Path:
    """H.264 progressive preview for browser playback."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"scale='min({max_w},iw)':-2",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"preview transcode failed: {proc.stderr[-600:]}")
    return out_path
