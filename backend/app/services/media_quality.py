"""3D / dual-cam readiness checks — fix weak '3D depends on video' gap with guidance."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


def _motion_score(video_path: Path, samples: int = 12) -> float:
    """0–1 relative motion energy (higher = more camera/scene motion — good for SfM)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n < 4:
        cap.release()
        return 0.0
    idxs = np.linspace(0, max(0, n - 1), min(samples, n), dtype=int)
    prev = None
    diffs = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (160, 90))
        if prev is not None:
            d = float(np.mean(cv2.absdiff(gray, prev))) / 255.0
            diffs.append(d)
        prev = gray
    cap.release()
    if not diffs:
        return 0.0
    # Map typical 0.02–0.15 range into 0–1
    m = float(np.median(diffs))
    return float(min(1.0, max(0.0, (m - 0.01) / 0.12)))


def _brightness(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    mid = max(0, n // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray) / 255.0)


def assess_session(doc: dict[str, Any]) -> dict[str, Any]:
    """Return readiness scores + plain-language tips for dual-cam and 3D."""
    cams = doc.get("cameras") or []
    tips: list[str] = []
    warnings: list[str] = []
    scores: dict[str, Any] = {
        "camera_count": len(cams),
        "dual_cam": len(cams) >= 2,
        "duration_s": 0.0,
        "motion": None,
        "brightness": None,
        "sync_ok": bool((doc.get("sync") or {}).get("passed")),
    }

    paths = []
    for c in cams:
        p = Path(c.get("path") or "")
        if p.exists():
            paths.append(p)
        scores["duration_s"] = max(scores["duration_s"], float(c.get("duration_s") or 0))

    if not paths:
        return {
            "ready_for_3d": False,
            "ready_for_dual_review": False,
            "grade": "F",
            "scores": scores,
            "tips": ["Upload at least one range video first."],
            "warnings": ["No video files found on disk."],
        }

    primary = paths[0]
    try:
        scores["motion"] = round(_motion_score(primary), 3)
        scores["brightness"] = round(_brightness(primary), 3)
    except Exception as e:
        warnings.append(f"Could not sample video quality: {e}")

    # Tips
    if len(cams) < 2:
        tips.append("Add a side/observer camera — dual-cam improves hit/miss and 3D a lot.")
    else:
        tips.append("Dual-cam detected — keep both lenses seeing overlapping bay structure.")

    motion = scores.get("motion")
    if motion is not None:
        if motion < 0.15:
            warnings.append("Very little camera motion — walk or pan slowly for better 3D.")
            tips.append("For 3D: move the camera around the bay (orbit), don’t hold still.")
        elif motion > 0.85:
            warnings.append("Very shaky / fast motion — slow the pan for cleaner recon.")
        else:
            tips.append("Motion looks usable for 3D reconstruction.")

    bright = scores.get("brightness")
    if bright is not None:
        if bright < 0.18:
            warnings.append("Video is dark — add light or avoid pure night IR if possible.")
        elif bright > 0.92:
            warnings.append("Video is blown out / very bright — reduce exposure if 3D fails.")

    if scores["duration_s"] < 8:
        warnings.append("Clip is very short — 3D needs more viewpoints (aim 20–60s of movement).")
    if scores["duration_s"] > 0 and scores["duration_s"] < 20 and len(cams) < 2:
        tips.append("Short single-cam clips often score OK but 3D will be thin.")

    # Grade for 3D
    pts = 0
    if len(cams) >= 2:
        pts += 35
    if motion is not None and 0.2 <= motion <= 0.8:
        pts += 30
    elif motion is not None and motion > 0.1:
        pts += 15
    if bright is not None and 0.2 <= bright <= 0.85:
        pts += 20
    if scores["duration_s"] >= 15:
        pts += 15
    elif scores["duration_s"] >= 8:
        pts += 8

    if pts >= 75:
        grade = "A"
    elif pts >= 55:
        grade = "B"
    elif pts >= 40:
        grade = "C"
    elif pts >= 25:
        grade = "D"
    else:
        grade = "F"

    ready_3d = grade in ("A", "B", "C") and len(paths) >= 1
    ready_dual = len(cams) >= 1

    return {
        "ready_for_3d": ready_3d,
        "ready_for_dual_review": ready_dual,
        "grade": grade,
        "score_0_100": pts,
        "scores": scores,
        "tips": tips,
        "warnings": warnings,
        "summary": (
            f"3D readiness {grade} ({pts}/100). "
            + ("Dual-cam: yes. " if scores["dual_cam"] else "Dual-cam: no. ")
            + (f"Motion {motion:.2f}. " if motion is not None else "")
            + (f"Brightness {bright:.2f}." if bright is not None else "")
        ),
    }
