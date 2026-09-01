"""Multi-view body pose for IronSight God's Eye stick-figure.

Pipeline (no API keys):
1. MediaPipe Tasks Pose Landmarker (2D) on shooter + observer (synced)
2. Dual-cam relative pose from SIFT essential matrix (once)
3. Triangulate matching landmarks into 3D
4. Temporal smooth + optional COLMAP-path anchor for scale/drift

Falls back to HOG synthetic skeleton + COLMAP path if pose model missing.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from app.core.config import DATA_DIR

STICK_EDGES = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
]

MODEL_DIR = DATA_DIR / "models" / "mediapipe"
POSE_MODEL = MODEL_DIR / "pose_landmarker_lite.task"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)


def ensure_pose_model() -> Path:
    """Download MediaPipe pose landmarker weights (free, no API key)."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if POSE_MODEL.exists() and POSE_MODEL.stat().st_size > 100_000:
        return POSE_MODEL
    tmp = POSE_MODEL.with_suffix(".task.part")
    urllib.request.urlretrieve(POSE_MODEL_URL, tmp)
    tmp.replace(POSE_MODEL)
    return POSE_MODEL


def _make_landmarker():
    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        from mediapipe import Image as MpImage
        from mediapipe import ImageFormat

        model = ensure_pose_model()
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.4,
            min_pose_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        landmarker = vision.PoseLandmarker.create_from_options(options)
        return landmarker, MpImage, ImageFormat
    except Exception:
        return None, None, None


def detect_pose_2d_bgr(frame_bgr: np.ndarray, landmarker, MpImage, ImageFormat) -> Optional[dict]:
    """Return {landmarks:[{x,y,z,vis}], bbox} normalized, or None."""
    if landmarker is None:
        return None
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = MpImage(image_format=ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    if not result.pose_landmarks:
        return None
    lms_raw = result.pose_landmarks[0]
    lms = []
    xs, ys = [], []
    for lm in lms_raw:
        vis = float(getattr(lm, "visibility", getattr(lm, "presence", 0.8)) or 0.8)
        lms.append({"x": float(lm.x), "y": float(lm.y), "z": float(lm.z), "vis": vis})
        if vis > 0.25:
            xs.append(lm.x)
            ys.append(lm.y)
    bbox = None
    if xs and ys:
        bbox = [
            max(0.0, min(xs) - 0.02),
            max(0.0, min(ys) - 0.02),
            min(1.0, max(xs) - min(xs) + 0.04),
            min(1.0, max(ys) - min(ys) + 0.04),
        ]
    return {"landmarks": lms, "bbox": bbox, "source": "mediapipe_tasks"}


def _hog_pose(frame_bgr: np.ndarray) -> Optional[dict]:
    h, w = frame_bgr.shape[:2]
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    rects, _ = hog.detectMultiScale(frame_bgr, winStride=(8, 8), padding=(8, 8), scale=1.05)
    if not len(rects):
        return None
    x, y, bw, bh = max(rects, key=lambda r: r[2] * r[3])
    bbox = [x / w, y / h, bw / w, bh / h]
    return {
        "landmarks": _synthetic_skeleton(bbox),
        "bbox": bbox,
        "source": "hog_synthetic",
    }


def _synthetic_skeleton(bbox: list[float]) -> list[dict[str, Any]]:
    x, y, bw, bh = bbox
    cx = x + bw / 2

    def p(nx, ny, vis=0.8):
        return {"x": float(nx), "y": float(ny), "z": 0.0, "vis": vis}

    lms = [{"x": 0.0, "y": 0.0, "z": 0.0, "vis": 0.0} for _ in range(33)]
    lms[0] = p(cx, y + bh * 0.08)
    lms[11] = p(cx - bw * 0.22, y + bh * 0.22)
    lms[12] = p(cx + bw * 0.22, y + bh * 0.22)
    lms[13] = p(cx - bw * 0.32, y + bh * 0.38)
    lms[14] = p(cx + bw * 0.32, y + bh * 0.38)
    lms[15] = p(cx - bw * 0.28, y + bh * 0.52)
    lms[16] = p(cx + bw * 0.28, y + bh * 0.52)
    lms[23] = p(cx - bw * 0.15, y + bh * 0.52)
    lms[24] = p(cx + bw * 0.15, y + bh * 0.52)
    lms[25] = p(cx - bw * 0.14, y + bh * 0.72)
    lms[26] = p(cx + bw * 0.14, y + bh * 0.72)
    lms[27] = p(cx - bw * 0.12, y + bh * 0.95)
    lms[28] = p(cx + bw * 0.12, y + bh * 0.95)
    return lms


def estimate_dual_cam_pose(
    path_a: Path,
    path_b: Path,
    *,
    t: float = 2.0,
    offset_s: float = 0.0,
) -> Optional[dict[str, Any]]:
    """Estimate relative pose B←A via SIFT + essential matrix (pixels only)."""
    fa = _read_frame(path_a, t)
    fb = _read_frame(path_b, max(0.0, t + offset_s))
    if fa is None or fb is None:
        return None
    ha, wa = fa.shape[:2]
    hb, wb = fb.shape[:2]
    gray1 = cv2.cvtColor(fa, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(fb, cv2.COLOR_BGR2GRAY)
    if hasattr(cv2, "SIFT_create"):
        det = cv2.SIFT_create(nfeatures=4000)
    else:
        det = cv2.ORB_create(nfeatures=4000)
    k1, d1 = det.detectAndCompute(gray1, None)
    k2, d2 = det.detectAndCompute(gray2, None)
    if d1 is None or d2 is None or len(k1) < 30 or len(k2) < 30:
        return None
    if d1.dtype == np.float32:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        knn = matcher.knnMatch(d1, d2, k=2)
        good = [m for m, n in knn if m.distance < 0.75 * n.distance] if knn else []
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        good = list(matcher.match(d1, d2))
    if len(good) < 40:
        return None
    pts1 = np.float64([k1[m.queryIdx].pt for m in good])
    pts2 = np.float64([k2[m.trainIdx].pt for m in good])
    Ka = _K(wa, ha)
    Kb = _K(wb, hb)
    E, mask = cv2.findEssentialMat(pts1, pts2, Ka, method=cv2.RANSAC, prob=0.999, threshold=1.5)
    if E is None:
        return None
    _, R, tvec, mask2 = cv2.recoverPose(E, pts1, pts2, Ka)
    return {
        "R": R.tolist(),
        "t": tvec.reshape(3).tolist(),
        "K_a": Ka.tolist(),
        "K_b": Kb.tolist(),
        "inliers": int(mask2.sum()) if mask2 is not None else 0,
        "size_a": [wa, ha],
        "size_b": [wb, hb],
    }


def _K(w: int, h: int) -> np.ndarray:
    f = 0.9 * max(w, h)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)


def _read_frame(path: Path, t: float) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(t * fps)))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def triangulate_landmarks(
    lms_a: list[dict],
    lms_b: list[dict],
    dual: dict[str, Any],
) -> list[dict[str, Any]]:
    """Triangulate dual-view landmarks into camera-A frame."""
    Ka = np.array(dual["K_a"], dtype=np.float64)
    Kb = np.array(dual["K_b"], dtype=np.float64)
    R = np.array(dual["R"], dtype=np.float64)
    t = np.array(dual["t"], dtype=np.float64).reshape(3, 1)
    Pa = Ka @ np.hstack([np.eye(3), np.zeros((3, 1))])
    Pb = Kb @ np.hstack([R, t])
    wa, ha = dual["size_a"]
    wb, hb = dual["size_b"]
    out = []
    n = min(len(lms_a), len(lms_b), 33)
    for i in range(n):
        a, b = lms_a[i], lms_b[i]
        va, vb = float(a.get("vis") or 0), float(b.get("vis") or 0)
        if va < 0.25 or vb < 0.25:
            out.append({"x": 0.0, "y": 0.0, "z": 0.0, "vis": 0.0, "method": "skip"})
            continue
        pt1 = np.array([[a["x"] * wa], [a["y"] * ha]], dtype=np.float64)
        pt2 = np.array([[b["x"] * wb], [b["y"] * hb]], dtype=np.float64)
        Xh = cv2.triangulatePoints(Pa, Pb, pt1, pt2)
        X = (Xh[:3] / (Xh[3] + 1e-9)).reshape(3)
        # reject behind cameras / absurd depth
        if X[2] <= 0.05 or abs(X[0]) > 50 or abs(X[1]) > 50 or X[2] > 80:
            out.append({"x": 0.0, "y": 0.0, "z": 0.0, "vis": 0.0, "method": "reject"})
            continue
        out.append(
            {
                "x": float(X[0]),
                "y": float(X[1]),
                "z": float(X[2]),
                "vis": float(min(va, vb)),
                "method": "triangulate",
            }
        )
    return out


def extract_multiview_pose_sequence(
    primary_path: Path,
    observer_path: Optional[Path],
    *,
    sync_offset_s: float = 0.0,
    sample_fps: float = 3.0,
    max_frames: int = 160,
    colmap_poses: Optional[list[dict]] = None,
) -> dict[str, Any]:
    """Full multi-view pose track with triangulation when dual-cam present."""
    landmarker, MpImage, ImageFormat = _make_landmarker()
    dual = None
    if observer_path and Path(observer_path).exists():
        dual = estimate_dual_cam_pose(primary_path, observer_path, t=2.0, offset_s=sync_offset_s)

    cap = cv2.VideoCapture(str(primary_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {primary_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps / sample_fps)))
    duration_s = total / max(fps, 1e-6)

    seq = []
    fidx = 0
    while fidx < total and len(seq) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok:
            break
        t = fidx / fps
        pose_a = None
        if landmarker is not None:
            pose_a = detect_pose_2d_bgr(frame, landmarker, MpImage, ImageFormat)
        if pose_a is None:
            pose_a = _hog_pose(frame)

        pose_b = None
        if observer_path and Path(observer_path).exists():
            fb = _read_frame(observer_path, max(0.0, t + sync_offset_s))
            if fb is not None:
                if landmarker is not None:
                    pose_b = detect_pose_2d_bgr(fb, landmarker, MpImage, ImageFormat)
                if pose_b is None:
                    pose_b = _hog_pose(fb)

        landmarks_3d = None
        method = "mono"
        if pose_a and pose_b and dual and dual.get("inliers", 0) >= 30:
            landmarks_3d = triangulate_landmarks(pose_a["landmarks"], pose_b["landmarks"], dual)
            good = sum(1 for p in landmarks_3d if p.get("vis", 0) > 0.25)
            if good >= 6:
                method = "multiview_triangulate"
            else:
                landmarks_3d = None

        if landmarks_3d is None and pose_a:
            # Mono lift: place on COLMAP path with metric-ish scale
            landmarks_3d = _lift_mono(pose_a["landmarks"], t, duration_s, colmap_poses)
            method = f"mono_{pose_a.get('source') or 'pose'}"

        if landmarks_3d:
            root = _root_from_lms(landmarks_3d)
            seq.append(
                {
                    "t": t,
                    "root": root,
                    "landmarks": landmarks_3d,
                    "edges": STICK_EDGES,
                    "method": method,
                    "bbox_2d": pose_a.get("bbox") if pose_a else None,
                    "dual_view": bool(pose_b),
                }
            )
        fidx += step

    cap.release()
    if landmarker is not None:
        try:
            landmarker.close()
        except Exception:
            pass

    # Temporal smooth roots + landmarks
    seq = _smooth_sequence(seq)
    return {
        "sequence": seq,
        "dual_pose": dual,
        "multiview": bool(dual and dual.get("inliers", 0) >= 30),
        "model": "mediapipe_tasks" if landmarker else "hog",
    }


def _root_from_lms(lms: list[dict]) -> dict:
    # mid-hip 23/24 if visible else mean of good points
    pts = []
    for i in (23, 24, 11, 12):
        if i < len(lms) and lms[i].get("vis", 0) > 0.2:
            pts.append([lms[i]["x"], lms[i]["y"], lms[i]["z"]])
    if not pts:
        pts = [[p["x"], p["y"], p["z"]] for p in lms if p.get("vis", 0) > 0.2]
    if not pts:
        return {"x": 0.0, "y": 0.0, "z": 0.0}
    m = np.mean(pts, axis=0)
    return {"x": float(m[0]), "y": float(m[1]), "z": float(m[2])}


def _lift_mono(
    lms: list[dict],
    t: float,
    duration_s: float,
    colmap_poses: Optional[list[dict]],
) -> list[dict]:
    path = []
    for p in colmap_poses or []:
        pos = p.get("position")
        if not pos:
            continue
        if isinstance(pos, dict):
            path.append([pos["x"], pos["y"], pos["z"]])
        else:
            path.append([float(pos[0]), float(pos[1]), float(pos[2])])
    if not path:
        path = [[0, 0, float(i) * 0.3] for i in range(20)]
    path = np.array(path, dtype=np.float64)
    frac = min(1.0, max(0.0, t / max(duration_s, 1e-3)))
    base = path[int(frac * (len(path) - 1))]
    scale = 1.6
    out = []
    for lm in lms:
        lx = (lm["x"] - 0.5) * scale
        ly = -(lm["y"] - 0.55) * scale
        lz = float(lm.get("z") or 0) * scale * 0.25
        out.append(
            {
                "x": float(base[0] + lx),
                "y": float(base[1] + ly),
                "z": float(base[2] + lz),
                "vis": float(lm.get("vis") or 0),
                "method": "mono_lift",
            }
        )
    return out


def _smooth_sequence(seq: list[dict], win: int = 3) -> list[dict]:
    if len(seq) < 3:
        return seq
    roots = np.array([[s["root"]["x"], s["root"]["y"], s["root"]["z"]] for s in seq])
    ker = np.ones(win) / win
    for c in range(3):
        roots[:, c] = np.convolve(roots[:, c], ker, mode="same")
    # re-anchor landmarks by root delta
    for i, s in enumerate(seq):
        old = np.array([s["root"]["x"], s["root"]["y"], s["root"]["z"]])
        new = roots[i]
        d = new - old
        s["root"] = {"x": float(new[0]), "y": float(new[1]), "z": float(new[2])}
        for lm in s["landmarks"]:
            if lm.get("vis", 0) > 0.15:
                lm["x"] = float(lm["x"] + d[0])
                lm["y"] = float(lm["y"] + d[1])
                lm["z"] = float(lm["z"] + d[2])
    return seq


def build_session_pose(
    *,
    video_path: Path,
    pointcloud: Optional[dict[str, Any]],
    duration_s: float,
    out_path: Path,
    sample_fps: float = 3.0,
    observer_path: Optional[Path] = None,
    sync_offset_s: float = 0.0,
) -> dict[str, Any]:
    colmap_poses = (pointcloud or {}).get("camera_poses") or []
    mv = extract_multiview_pose_sequence(
        video_path,
        observer_path,
        sync_offset_s=sync_offset_s,
        sample_fps=sample_fps,
        colmap_poses=colmap_poses,
    )
    method = "multiview_triangulate" if mv.get("multiview") else mv.get("model", "mono")
    n_tri = sum(
        1
        for s in mv["sequence"]
        if s.get("method") == "multiview_triangulate"
    )
    doc = {
        "method": method + "+bundle_temporal",
        "sample_count": len(mv["sequence"]),
        "frames_2d": len(mv["sequence"]),
        "multiview_frames": n_tri,
        "dual_pose_inliers": (mv.get("dual_pose") or {}).get("inliers"),
        "edges": STICK_EDGES,
        "sequence": mv["sequence"],
        "path": str(out_path),
        "model": mv.get("model"),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # dual_pose R/t is large but useful — store summary only in file for dual
    if mv.get("dual_pose"):
        doc["dual_cam_calibrated"] = True
        doc["dual_pose"] = {
            "inliers": mv["dual_pose"].get("inliers"),
            "R": mv["dual_pose"].get("R"),
            "t": mv["dual_pose"].get("t"),
        }
    out_path.write_text(json.dumps(doc))
    return doc


def pose_at_time(pose_doc: dict[str, Any], t: float) -> Optional[dict[str, Any]]:
    seq = pose_doc.get("sequence") or []
    if not seq:
        return None
    best = min(seq, key=lambda s: abs(s["t"] - t))
    if abs(best["t"] - t) > 1.0:
        return None
    return best
