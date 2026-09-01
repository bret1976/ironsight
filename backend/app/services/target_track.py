"""Multi-view target detection + tracking for IronSight dual-cam HUD.

Pipeline (all real signals — no mock boxes):
1. Color/shape CV proposals for steel plates / colored poppers (OpenCV)
2. VLM keyframe refinement at shot times (pixel bboxes + labels)
3. CSRT/KCF OpenCV trackers to interpolate boxes between keyframes
4. Optional COLMAP 3D→2D projection when poses exist
5. Dual-cam: independent tracks on shooter + observer, label-matched

Output timeline (normalized xywh 0–1):
{
  "cameras": {
    "primary": {"width": W, "height": H, "fps": F},
    "observer": {...} | null
  },
  "keyframes": [
    {"t": 1.23, "camera": "primary", "targets": [
       {"id":"B3","label":"RED PLATE","color":"red",
        "bbox":[x,y,w,h], "confidence":0.9, "source":"vlm|cv|track|proj3d"}
    ]}
  ],
  "tracks": {
    "primary": [{"id":"B3","label":"...","color":"red","samples":[{"t":..,"bbox":[..],"conf":..}]}],
    "observer": [...]
  }
}
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from app.services import hit_miss

try:
    from app.services import yolo_plates
except Exception:  # pragma: no cover
    yolo_plates = None  # type: ignore


# ── helpers ──────────────────────────────────────────────────────────────

def _clamp01(v: float) -> float:
    return float(max(0.0, min(1.0, v)))


def xyxy_to_norm_xywh(x1, y1, x2, y2, w, h) -> list[float]:
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    return [
        _clamp01(x1 / w),
        _clamp01(y1 / h),
        _clamp01(bw / w),
        _clamp01(bh / h),
    ]


def norm_xywh_to_xyxy(box: list[float], w: int, h: int) -> tuple[int, int, int, int]:
    x, y, bw, bh = box
    x1 = int(x * w)
    y1 = int(y * h)
    x2 = int((x + bw) * w)
    y2 = int((y + bh) * h)
    return x1, y1, x2, y2


def iou(a: list[float], b: list[float]) -> float:
    """IoU of two normalized xywh boxes."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


# ── OpenCV color plate proposals ─────────────────────────────────────────

# HSV ranges for common range plate colors (outdoor-tolerant)
_HSV_RANGES: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    "red": [
        ((0, 70, 50), (12, 255, 255)),
        ((165, 70, 50), (180, 255, 255)),
    ],
    "blue": [((95, 60, 40), (135, 255, 255))],
    "white": [((0, 0, 160), (180, 55, 255))],
    "green": [((35, 50, 40), (90, 255, 255))],
    "yellow": [((18, 80, 80), (38, 255, 255))],
}


def detect_color_plates(
    frame_bgr: np.ndarray,
    *,
    min_area_frac: float = 0.0004,
    max_area_frac: float = 0.12,
) -> list[dict[str, Any]]:
    """Classical CV proposals for colored steel/popper plates."""
    h, w = frame_bgr.shape[:2]
    area = float(h * w)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    out: list[dict[str, Any]] = []

    for color, ranges in _HSV_RANGES.items():
        mask = np.zeros((h, w), dtype=np.uint8)
        for lo, hi in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lo), np.array(hi)))
        # clean
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            a = cv2.contourArea(c)
            if a < min_area_frac * area or a > max_area_frac * area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            # Prefer roughly plate-like aspect (not long strips)
            ar = bw / max(1, bh)
            if ar < 0.25 or ar > 4.0:
                continue
            # Compactness filter
            peri = cv2.arcLength(c, True)
            if peri <= 0:
                continue
            circularity = 4 * math.pi * a / (peri * peri)
            if circularity < 0.12:
                continue
            box = xyxy_to_norm_xywh(x, y, x + bw, y + bh, w, h)
            out.append(
                {
                    "id": f"{color[0].upper()}{len(out)+1}",
                    "label": f"{color.upper()} PLATE",
                    "color": color if color != "yellow" else "amber",
                    "bbox": box,
                    "confidence": float(min(0.85, 0.35 + circularity + a / area * 40)),
                    "source": "cv",
                }
            )
    # NMS-ish by IoU
    out.sort(key=lambda d: d["confidence"], reverse=True)
    kept: list[dict[str, Any]] = []
    for d in out:
        if any(iou(d["bbox"], k["bbox"]) > 0.45 for k in kept):
            continue
        kept.append(d)
    return kept[:16]


def detect_motion_blobs(
    prev_gray: Optional[np.ndarray],
    frame_bgr: np.ndarray,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Frame-diff blobs (muzzle flash / plate kick candidates)."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    h, w = gray.shape
    if prev_gray is None or prev_gray.shape != gray.shape:
        return [], gray
    diff = cv2.absdiff(prev_gray, gray)
    _, th = cv2.threshold(diff, 28, 255, cv2.THRESH_BINARY)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    area = float(h * w)
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 0.00015 * area or a > 0.08 * area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        blobs.append(
            {
                "id": f"M{len(blobs)+1}",
                "label": "MOTION",
                "color": "amber",
                "bbox": xyxy_to_norm_xywh(x, y, x + bw, y + bh, w, h),
                "confidence": 0.4,
                "source": "motion",
            }
        )
    return blobs[:8], gray


# ── VLM keyframe detection (pixel boxes) ─────────────────────────────────

def detect_targets_vlm(image_path: Path, *, context: str = "") -> list[dict[str, Any]]:
    """Ask vision model for all visible range targets with normalized bboxes."""
    try:
        data = hit_miss.detect_targets_in_frame(image_path, context=context)
    except Exception as e:
        return [{"error": str(e), "source": "vlm_error"}]
    targets = []
    for i, t in enumerate(data.get("targets") or []):
        bbox = t.get("bbox") or t.get("box")
        if not bbox or len(bbox) < 4:
            continue
        # Accept xywh normalized or xyxy normalized
        b = [float(x) for x in bbox[:4]]
        if max(b) > 1.5:  # pixel coords — normalize if image available
            img = cv2.imread(str(image_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            if b[2] > b[0] and b[3] > b[1]:  # xyxy
                b = xyxy_to_norm_xywh(b[0], b[1], b[2], b[3], w, h)
            else:
                b = [_clamp01(b[0] / w), _clamp01(b[1] / h), _clamp01(b[2] / w), _clamp01(b[3] / h)]
        else:
            # If xyxy form (x2>x1 style with all <1)
            if b[2] > b[0] and b[3] > b[1] and b[2] <= 1.05 and (b[2] - b[0]) < 0.95:
                # likely xyxy normalized
                b = [_clamp01(b[0]), _clamp01(b[1]), _clamp01(b[2] - b[0]), _clamp01(b[3] - b[1])]
            else:
                b = [_clamp01(b[0]), _clamp01(b[1]), _clamp01(b[2]), _clamp01(b[3])]
        tid = t.get("id") or t.get("target_id") or f"T{i+1}"
        label = t.get("label") or t.get("target_label") or str(tid)
        color = (t.get("color") or t.get("target_color") or "steel") or "steel"
        targets.append(
            {
                "id": str(tid),
                "label": str(label),
                "color": str(color).lower(),
                "bbox": b,
                "confidence": float(t.get("confidence") or 0.75),
                "source": "vlm",
            }
        )
    return targets


# ── Multi-object tracking (Lucas-Kanade optical flow — no opencv-contrib) ─

@dataclass
class TrackState:
    id: str
    label: str
    color: str
    samples: list[dict[str, Any]] = field(default_factory=list)
    # pixel xywh
    _x: float = 0.0
    _y: float = 0.0
    _bw: float = 0.0
    _bh: float = 0.0
    _pts: Any = None  # Nx1x2 float32 feature points
    _alive: bool = True
    _tmpl: Any = None  # grayscale template for NCC re-lock


def _good_features_in_box(gray: np.ndarray, x: float, y: float, bw: float, bh: float):
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2, y2 = min(gray.shape[1], int(x + bw)), min(gray.shape[0], int(y + bh))
    if x2 - x1 < 6 or y2 - y1 < 6:
        return None
    roi = gray[y1:y2, x1:x2]
    pts = cv2.goodFeaturesToTrack(
        roi, maxCorners=40, qualityLevel=0.01, minDistance=3, blockSize=5
    )
    if pts is None or len(pts) < 3:
        # fallback: grid of points inside box
        xs = np.linspace(x1 + 2, x2 - 2, 4)
        ys = np.linspace(y1 + 2, y2 - 2, 4)
        grid = np.array([[[xx, yy]] for yy in ys for xx in xs], dtype=np.float32)
        return grid
    pts = pts + np.array([[[x1, y1]]], dtype=np.float32)
    return pts


def track_video_segment(
    video_path: Path,
    seed_targets: list[dict[str, Any]],
    *,
    t0: float,
    t1: float,
    sample_hz: float = 6.0,
    camera: str = "primary",
) -> list[dict[str, Any]]:
    """Track seed boxes t0→t1 with LK optical flow + template re-lock."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    start_f = max(0, int(t0 * fps))
    end_f = int(min(total - 1, t1 * fps)) if total else int(t1 * fps)
    step = max(1, int(round(fps / sample_hz)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    ok, frame = cap.read()
    if not ok:
        cap.release()
        return []
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    )

    tracks: list[TrackState] = []
    for t in seed_targets:
        box = t.get("bbox")
        if not box:
            continue
        x1, y1, x2, y2 = norm_xywh_to_xyxy(box, w, h)
        bw, bh = float(max(6, x2 - x1)), float(max(6, y2 - y1))
        x, y = float(x1), float(y1)
        pts = _good_features_in_box(prev_gray, x, y, bw, bh)
        if pts is None:
            continue
        # template for re-lock
        xi, yi = int(x), int(y)
        tmpl = prev_gray[yi : yi + int(bh), xi : xi + int(bw)].copy()
        if tmpl.size < 25:
            continue
        st = TrackState(
            id=str(t.get("id") or t.get("label") or f"T{len(tracks)+1}"),
            label=str(t.get("label") or t.get("id") or "TARGET"),
            color=str(t.get("color") or "steel"),
            samples=[
                {
                    "t": start_f / fps,
                    "bbox": box,
                    "conf": float(t.get("confidence") or 0.7),
                    "source": t.get("source") or "seed",
                }
            ],
            _x=x,
            _y=y,
            _bw=bw,
            _bh=bh,
            _pts=pts,
            _tmpl=tmpl,
        )
        tracks.append(st)

    fidx = start_f
    while fidx < end_f:
        fidx += step
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tsec = fidx / fps
        for st in tracks:
            if not st._alive or st._pts is None:
                continue
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, st._pts, None, **lk_params
            )
            if nxt is None or status is None:
                st._alive = False
                continue
            good_new = nxt[status.flatten() == 1]
            good_old = st._pts[status.flatten() == 1]
            if len(good_new) < 3:
                # template re-lock near last box
                search = 40
                x1 = max(0, int(st._x - search))
                y1 = max(0, int(st._y - search))
                x2 = min(w, int(st._x + st._bw + search))
                y2 = min(h, int(st._y + st._bh + search))
                region = gray[y1:y2, x1:x2]
                if (
                    st._tmpl is not None
                    and region.shape[0] > st._tmpl.shape[0]
                    and region.shape[1] > st._tmpl.shape[1]
                ):
                    res = cv2.matchTemplate(region, st._tmpl, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)
                    if max_val > 0.45:
                        st._x = float(x1 + max_loc[0])
                        st._y = float(y1 + max_loc[1])
                        st._pts = _good_features_in_box(
                            gray, st._x, st._y, st._bw, st._bh
                        )
                        conf = float(0.4 + 0.4 * max_val)
                    else:
                        st._alive = False
                        continue
                else:
                    st._alive = False
                    continue
            else:
                # median displacement of tracked points
                d = good_new.reshape(-1, 2) - good_old.reshape(-1, 2)
                dx = float(np.median(d[:, 0]))
                dy = float(np.median(d[:, 1]))
                st._x = float(st._x + dx)
                st._y = float(st._y + dy)
                st._pts = good_new.reshape(-1, 1, 2).astype(np.float32)
                conf = 0.6

            # clamp
            st._x = float(max(-st._bw * 0.3, min(w - st._bw * 0.3, st._x)))
            st._y = float(max(-st._bh * 0.3, min(h - st._bh * 0.3, st._y)))
            if st._bw < 4 or st._bh < 4:
                st._alive = False
                continue
            box = xyxy_to_norm_xywh(
                st._x, st._y, st._x + st._bw, st._y + st._bh, w, h
            )
            st.samples.append(
                {"t": tsec, "bbox": box, "conf": conf, "source": "lk_track"}
            )
        prev_gray = gray

    cap.release()
    return [
        {
            "id": st.id,
            "label": st.label,
            "color": st.color,
            "camera": camera,
            "samples": st.samples,
        }
        for st in tracks
        if st.samples
    ]


# ── 3D projection ───────────────────────────────────────────────────────

def project_targets_3d(
    targets: list[dict[str, Any]],
    camera_pose: dict[str, Any],
    *,
    width: int,
    height: int,
    fx: Optional[float] = None,
    fy: Optional[float] = None,
) -> list[dict[str, Any]]:
    """Project target position_3d into normalized image bboxes via COLMAP pose."""
    R = np.array(camera_pose.get("R") or np.eye(3), dtype=np.float64)
    tvec = np.array(camera_pose.get("t") or [0, 0, 0], dtype=np.float64).reshape(3)
    if R.shape != (3, 3):
        return []
    # Prefer explicit camera position if present (world coords)
    if camera_pose.get("position"):
        C = np.array(
            [
                camera_pose["position"][0]
                if isinstance(camera_pose["position"], (list, tuple))
                else camera_pose["position"].get("x", 0),
                camera_pose["position"][1]
                if isinstance(camera_pose["position"], (list, tuple))
                else camera_pose["position"].get("y", 0),
                camera_pose["position"][2]
                if isinstance(camera_pose["position"], (list, tuple))
                else camera_pose["position"].get("z", 0),
            ],
            dtype=np.float64,
        )
    else:
        C = -R.T @ tvec

    fx = fx or float(width) * 0.9
    fy = fy or float(height) * 0.9
    cx, cy = width / 2.0, height / 2.0
    out = []
    for tgt in targets:
        p = tgt.get("position_3d")
        if not p:
            continue
        X = np.array([p["x"], p["y"], p["z"]], dtype=np.float64)
        # world → camera
        Xc = R @ (X - C)
        if Xc[2] <= 0.05:
            continue
        u = fx * (Xc[0] / Xc[2]) + cx
        v = fy * (Xc[1] / Xc[2]) + cy
        if u < -width * 0.2 or u > width * 1.2 or v < -height * 0.2 or v > height * 1.2:
            continue
        plate_m = 0.35
        s = (fx * plate_m / Xc[2]) / width
        s = float(max(0.02, min(0.18, s)))
        box = [
            _clamp01(u / width - s / 2),
            _clamp01(v / height - s * 0.7),
            s,
            s * 1.3,
        ]
        out.append(
            {
                "id": tgt.get("id"),
                "label": tgt.get("label") or tgt.get("id"),
                "color": tgt.get("color") or "steel",
                "bbox": box,
                "confidence": 0.5,
                "source": "proj3d",
                "depth_m": float(Xc[2]),
            }
        )
    return out


def nearest_pose(poses: list[dict], t_frac: float) -> Optional[dict]:
    if not poses:
        return None
    idx = int(round(t_frac * (len(poses) - 1)))
    idx = max(0, min(len(poses) - 1, idx))
    return poses[idx]


# ── Full session track build ─────────────────────────────────────────────

def build_session_tracks(
    *,
    primary_path: Path,
    observer_path: Optional[Path],
    shots: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    out_dir: Path,
    sync_offset_s: float = 0.0,
    pointcloud: Optional[dict[str, Any]] = None,
    use_vlm: bool = True,
    track_pad_s: float = 1.2,
    sample_hz: float = 5.0,
    context: str = "",
) -> dict[str, Any]:
    """Build multi-view track timeline for a processed session."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(primary_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {primary_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1)
    dur = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / max(fps, 1e-6)
    cap.release()

    yolo_on = bool(yolo_plates and yolo_plates.weights_available())
    result: dict[str, Any] = {
        "cameras": {
            "primary": {"width": w, "height": h, "fps": fps, "duration_s": dur},
            "observer": None,
        },
        "keyframes": [],
        "tracks": {"primary": [], "observer": []},
        "method": ("yolo+" if yolo_on else "") + "cv+vlm+lk+proj3d",
        "yolo": yolo_on,
    }

    obs_meta = None
    if observer_path and Path(observer_path).exists():
        oc = cv2.VideoCapture(str(observer_path))
        if oc.isOpened():
            obs_meta = {
                "width": int(oc.get(cv2.CAP_PROP_FRAME_WIDTH) or 1),
                "height": int(oc.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1),
                "fps": float(oc.get(cv2.CAP_PROP_FPS) or 30.0),
            }
            result["cameras"]["observer"] = obs_meta
        oc.release()

    poses = (pointcloud or {}).get("camera_poses") or []
    primary_tracks: dict[str, dict[str, Any]] = {}
    observer_tracks: dict[str, dict[str, Any]] = {}

    # Keyframe at each shot (+ a few mid-session map frames)
    key_times = sorted({float(s["timestamp_s"]) for s in shots if s.get("timestamp_s") is not None})
    if dur > 0 and len(key_times) < 3:
        key_times = sorted(set(key_times + [dur * 0.2, dur * 0.5, dur * 0.8]))

    for t in key_times:
        # --- primary ---
        frame_path = out_dir / f"kf_primary_{t:.3f}.jpg".replace(":", "_")
        _extract_frame(primary_path, t, frame_path)
        seeds: list[dict[str, Any]] = []
        if frame_path.exists():
            img = cv2.imread(str(frame_path))
            # 1) Trained YOLO plates (preferred when weights exist)
            if img is not None and yolo_plates and yolo_plates.weights_available():
                try:
                    ydet = yolo_plates.detect_plates_yolo(img, conf=0.12)
                    seeds = _merge_detections(ydet, seeds)
                except Exception:
                    pass
            # 2) Classical color CV
            if img is not None:
                seeds = _merge_detections(seeds, detect_color_plates(img))
            # 3) VLM labels (refine / name plates)
            if use_vlm:
                vlm = detect_targets_vlm(frame_path, context=context or "range dual-cam")
                vlm = [v for v in vlm if "bbox" in v]
                seeds = _merge_detections(vlm, seeds)
            # 4) Project 3D if available
            if poses and targets:
                pose = nearest_pose(poses, t / max(dur, 1e-6))
                if pose:
                    proj = project_targets_3d(targets, pose, width=w, height=h)
                    seeds = _merge_detections(seeds, proj)

        # Inject shot-level engaged bbox + visible_targets as high-confidence seeds
        shot_here = min(shots, key=lambda s: abs(s["timestamp_s"] - t), default=None)
        if shot_here and abs(shot_here["timestamp_s"] - t) < 0.5:
            extra = []
            if shot_here.get("bbox"):
                extra.append(
                    {
                        "id": shot_here.get("target_id") or f"S{shot_here['id']}",
                        "label": shot_here.get("target_label") or "TARGET",
                        "color": shot_here.get("target_color") or "steel",
                        "bbox": shot_here["bbox"],
                        "confidence": float(shot_here.get("confidence") or 0.85),
                        "source": "shot",
                        "classification": shot_here.get("classification"),
                        "shot_id": shot_here["id"],
                    }
                )
            for v in shot_here.get("visible_targets") or []:
                if v.get("bbox"):
                    extra.append(
                        {
                            "id": v.get("id") or v.get("label"),
                            "label": v.get("label") or v.get("id"),
                            "color": v.get("color") or "steel",
                            "bbox": v["bbox"],
                            "confidence": float(v.get("confidence") or 0.75),
                            "source": "shot_visible",
                        }
                    )
            seeds = _merge_detections(extra, seeds)
            if shot_here.get("target_id"):
                for s in seeds:
                    if s.get("id") == shot_here.get("target_id") or s.get("label") == shot_here.get(
                        "target_label"
                    ):
                        s["shot_id"] = shot_here["id"]
                        s["classification"] = shot_here.get("classification")

        result["keyframes"].append(
            {"t": t, "camera": "primary", "targets": seeds, "frame": str(frame_path)}
        )

        # Track forward/back a window around this keyframe
        if seeds:
            seg = track_video_segment(
                primary_path,
                seeds,
                t0=max(0.0, t - track_pad_s),
                t1=min(dur, t + track_pad_s),
                sample_hz=sample_hz,
                camera="primary",
            )
            for tr in seg:
                _upsert_track(primary_tracks, tr)

        # --- observer ---
        if observer_path and obs_meta:
            ot = max(0.0, t + sync_offset_s)
            oframe = out_dir / f"kf_observer_{ot:.3f}.jpg".replace(":", "_")
            _extract_frame(observer_path, ot, oframe)
            oseeds: list[dict[str, Any]] = []
            if oframe.exists():
                oimg = cv2.imread(str(oframe))
                if oimg is not None and yolo_plates and yolo_plates.weights_available():
                    try:
                        oseeds = _merge_detections(
                            yolo_plates.detect_plates_yolo(oimg, conf=0.12), oseeds
                        )
                    except Exception:
                        pass
                if oimg is not None:
                    oseeds = _merge_detections(oseeds, detect_color_plates(oimg))
                if use_vlm:
                    ovlm = detect_targets_vlm(oframe, context=context or "observer 3P range")
                    ovlm = [v for v in ovlm if "bbox" in v]
                    oseeds = _merge_detections(ovlm, oseeds)
            result["keyframes"].append(
                {"t": ot, "camera": "observer", "targets": oseeds, "frame": str(oframe)}
            )
            if oseeds:
                oseg = track_video_segment(
                    observer_path,
                    oseeds,
                    t0=max(0.0, ot - track_pad_s),
                    t1=ot + track_pad_s,
                    sample_hz=sample_hz,
                    camera="observer",
                )
                for tr in oseg:
                    _upsert_track(observer_tracks, tr)

    result["tracks"]["primary"] = list(primary_tracks.values())
    result["tracks"]["observer"] = list(observer_tracks.values())

    # Dense lookup table for frontend (samples every ~0.2s)
    result["timeline"] = {
        "primary": _build_timeline(result["tracks"]["primary"], dur, step=0.2),
        "observer": _build_timeline(result["tracks"]["observer"], dur, step=0.2)
        if observer_path
        else [],
    }

    out_path = out_dir / "tracks.json"
    out_path.write_text(json.dumps(result, indent=2))
    result["path"] = str(out_path)
    return result


def _extract_frame(video: Path, t: float, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(t * fps)))
    ok, frame = cap.read()
    cap.release()
    if ok:
        cv2.imwrite(str(out), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return out


def _merge_detections(
    primary: list[dict[str, Any]], secondary: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Merge two detection lists; primary wins on IoU conflict."""
    kept = list(primary)
    for s in secondary:
        if "bbox" not in s:
            continue
        if any(iou(s["bbox"], k["bbox"]) > 0.4 for k in kept if "bbox" in k):
            # enrich label if primary is CV-only
            for k in kept:
                if "bbox" in k and iou(s["bbox"], k["bbox"]) > 0.4:
                    if k.get("source") == "cv" and s.get("source") == "vlm":
                        k["id"] = s.get("id") or k["id"]
                        k["label"] = s.get("label") or k["label"]
                        k["color"] = s.get("color") or k["color"]
                        k["source"] = "vlm+cv"
                        k["confidence"] = max(float(k.get("confidence") or 0), float(s.get("confidence") or 0))
                    break
            continue
        kept.append(s)
    return kept


def _upsert_track(store: dict[str, dict], tr: dict[str, Any]) -> None:
    tid = str(tr["id"])
    if tid not in store:
        store[tid] = {
            "id": tid,
            "label": tr.get("label"),
            "color": tr.get("color"),
            "camera": tr.get("camera"),
            "samples": list(tr.get("samples") or []),
        }
        return
    # merge samples sorted by t
    existing = {round(s["t"], 3): s for s in store[tid]["samples"]}
    for s in tr.get("samples") or []:
        existing[round(s["t"], 3)] = s
    store[tid]["samples"] = [existing[k] for k in sorted(existing.keys())]
    if tr.get("label"):
        store[tid]["label"] = tr["label"]


def _build_timeline(
    tracks: list[dict[str, Any]], duration_s: float, step: float = 0.2
) -> list[dict[str, Any]]:
    """Dense time → active boxes for cheap frontend lookup."""
    if duration_s <= 0:
        return []
    times = np.arange(0.0, duration_s + step, step)
    timeline = []
    for t in times:
        boxes = []
        for tr in tracks:
            samples = tr.get("samples") or []
            if not samples:
                continue
            # nearest sample within 0.6s
            best = min(samples, key=lambda s: abs(s["t"] - t))
            if abs(best["t"] - t) > 0.6:
                continue
            boxes.append(
                {
                    "id": tr["id"],
                    "label": tr.get("label"),
                    "color": tr.get("color"),
                    "bbox": best["bbox"],
                    "conf": best.get("conf", 0.5),
                    "source": best.get("source"),
                }
            )
        timeline.append({"t": float(t), "targets": boxes})
    return timeline


def boxes_at_time(tracks_doc: dict[str, Any], t: float, camera: str = "primary") -> list[dict]:
    """Query helper: interpolate boxes at time t."""
    tl = (tracks_doc.get("timeline") or {}).get(camera) or []
    if not tl:
        # fall back to tracks samples
        out = []
        for tr in (tracks_doc.get("tracks") or {}).get(camera) or []:
            samples = tr.get("samples") or []
            if not samples:
                continue
            best = min(samples, key=lambda s: abs(s["t"] - t))
            if abs(best["t"] - t) <= 0.75:
                out.append(
                    {
                        "id": tr["id"],
                        "label": tr.get("label"),
                        "color": tr.get("color"),
                        "bbox": best["bbox"],
                        "conf": best.get("conf", 0.5),
                        "source": best.get("source"),
                    }
                )
        return out
    best = min(tl, key=lambda x: abs(x["t"] - t))
    if abs(best["t"] - t) > 0.75:
        return []
    return list(best.get("targets") or [])
