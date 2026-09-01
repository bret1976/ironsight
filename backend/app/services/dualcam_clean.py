"""Clean dual-cam ingest: split SxS, strip HUD chrome, mask streamer/webcam overlays.

Used when user uploads a single side-by-side recording (or dirty YouTube-style export)
so IronSight gets raw-looking shooter/observer feeds without UI chrome.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


def probe(path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    raw = subprocess.check_output(cmd, text=True)
    data = json.loads(raw)
    st = (data.get("streams") or [{}])[0]
    fmt = data.get("format") or {}
    w = int(st.get("width") or 0)
    h = int(st.get("height") or 0)
    dur = float(st.get("duration") or fmt.get("duration") or 0)
    return {"width": w, "height": h, "duration_s": dur}


def _sample_frame(path: Path, t: float = 1.0) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(0, t * fps)))
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def detect_sxs_layout(frame: np.ndarray) -> dict[str, Any]:
    """Detect vertical split (left/right dual cam) via center seam + symmetry."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # vertical edge energy along x
    sob = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    col_energy = np.mean(np.abs(sob), axis=0)
    # smooth
    k = max(5, w // 80) | 1
    kernel = np.ones(k) / k
    col_energy = np.convolve(col_energy, kernel, mode="same")
    # search mid 30–70%
    lo, hi = int(w * 0.35), int(w * 0.65)
    seam = int(lo + np.argmax(col_energy[lo:hi]))
    # score: high seam vs neighbors
    seam_score = float(col_energy[seam] / (np.mean(col_energy) + 1e-6))
    is_sxs = seam_score > 1.35 and 0.4 * w < seam < 0.6 * w
    # also accept exact half if landscape 16:9 content looks dual
    if not is_sxs and w >= h and abs(seam - w // 2) < w * 0.08:
        # check left/right mean color difference (different cameras)
        L = gray[:, : w // 2]
        R = gray[:, w // 2 :]
        if abs(float(L.mean()) - float(R.mean())) > 3.0:
            is_sxs = True
            seam = w // 2
    return {
        "is_sxs": bool(is_sxs),
        "seam_x": seam,
        "seam_score": seam_score,
        "width": w,
        "height": h,
    }


def detect_hud_margins(frame: np.ndarray) -> dict[str, int]:
    """Detect dark letterbox / green HUD bars (top/bottom/side chrome)."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # rows/cols mean brightness
    row_m = gray.mean(axis=1)
    col_m = gray.mean(axis=0)
    thr = max(18.0, float(np.percentile(row_m, 15)))

    def dark_run(arr, from_start=True):
        if from_start:
            n = 0
            for v in arr:
                if v < thr:
                    n += 1
                else:
                    break
            return n
        n = 0
        for v in arr[::-1]:
            if v < thr:
                n += 1
            else:
                break
        return n

    top = min(h // 5, dark_run(row_m, True))
    bot = min(h // 5, dark_run(row_m, False))
    left = min(w // 8, dark_run(col_m, True))
    right = min(w // 8, dark_run(col_m, False))
    # IronSight-style: thin top bar ~36px, bottom timeline ~80px on 1080p
    if h >= 720:
        top = max(top, int(h * 0.03))
        bot = max(bot, int(h * 0.08))
    return {"top": top, "bottom": bot, "left": left, "right": right}


def detect_streamer_roi(frame: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    """Find lower-right webcam/streamer face blob (common in YouTube dual demos)."""
    h, w = frame.shape[:2]
    # Search lower-right quadrant
    x0, y0 = int(w * 0.55), int(h * 0.45)
    roi = frame[y0:h, x0:w]
    if roi.size == 0:
        return None
    # Skin-ish + high-contrast person blob via HOG/cascade if available
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    face = None
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if Path(cascade_path).exists():
        det = cv2.CascadeClassifier(cascade_path)
        faces = det.detectMultiScale(gray, 1.1, 4, minSize=(40, 40))
        if len(faces):
            # largest face
            fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            # expand to upper-body card
            pad = int(max(fw, fh) * 0.85)
            x1 = x0 + max(0, fx - pad)
            y1 = y0 + max(0, fy - pad)
            x2 = min(w, x0 + fx + fw + pad)
            y2 = min(h, y0 + fy + fh + int(pad * 1.8))
            face = (x1, y1, x2 - x1, y2 - y1)

    if face is None:
        # Fallback: saturated “floating head” region — high local variance bottom-right
        var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if var > 80:
            # assume standard PIP ~22% width in corner
            pw, ph = int(w * 0.22), int(h * 0.35)
            face = (w - pw - 8, h - ph - 8, pw, ph)
        else:
            return None
    return face  # x,y,w,h absolute


def analyze_video(path: Path) -> dict[str, Any]:
    frame = _sample_frame(path, 1.5)
    if frame is None:
        frame = _sample_frame(path, 0.5)
    if frame is None:
        raise RuntimeError(f"Cannot read frames from {path}")
    meta = probe(path)
    sxs = detect_sxs_layout(frame)
    margins = detect_hud_margins(frame)
    streamer = detect_streamer_roi(frame)
    return {
        **meta,
        "sxs": sxs,
        "margins": margins,
        "streamer_roi": (
            {"x": streamer[0], "y": streamer[1], "w": streamer[2], "h": streamer[3]}
            if streamer
            else None
        ),
        "path": str(path),
    }


def _ffmpeg_run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr[-800:]}")


def clean_and_split_sxs(
    input_path: Path,
    out_dir: Path,
    *,
    name: str = "clean",
    force_sxs: Optional[bool] = None,
    mask_streamer: bool = True,
    crop_hud: bool = True,
) -> dict[str, Any]:
    """Produce clean shooter/observer MP4s from dirty SxS or dual feed.

    Returns paths + analysis.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    analysis = analyze_video(input_path)
    is_sxs = force_sxs if force_sxs is not None else analysis["sxs"]["is_sxs"]
    w, h = analysis["width"], analysis["height"]
    margins = analysis["margins"] if crop_hud else {"top": 0, "bottom": 0, "left": 0, "right": 0}
    top, bot, left, right = margins["top"], margins["bottom"], margins["left"], margins["right"]

    # Content rectangle after HUD crop
    cw = w - left - right
    ch = h - top - bot
    if cw < 64 or ch < 64:
        top = bot = left = right = 0
        cw, ch = w, h

    shooter_out = out_dir / f"{name}_shooter.mp4"
    observer_out = out_dir / f"{name}_observer.mp4"
    report = {
        "analysis": analysis,
        "is_sxs": is_sxs,
        "crop": {"top": top, "bottom": bot, "left": left, "right": right},
        "shooter": str(shooter_out),
        "observer": str(observer_out) if is_sxs else None,
        "streamer_masked": False,
    }

    if is_sxs:
        seam = analysis["sxs"]["seam_x"]
        # seam relative to full frame; after left crop:
        seam_c = seam - left
        left_w = max(32, min(cw - 32, seam_c))
        right_w = cw - left_w
        # Base crop to content then split
        # left pane: crop=left_w:ch:left:top
        # right pane: crop=right_w:ch:left+left_w:top
        vf_left = f"crop={left_w}:{ch}:{left}:{top},setsar=1"
        vf_right = f"crop={right_w}:{ch}:{left + left_w}:{top},setsar=1"

        # Streamer usually on observer (right) lower corner
        if mask_streamer and analysis.get("streamer_roi"):
            sr = analysis["streamer_roi"]
            # map to right-pane coords
            rx = sr["x"] - (left + left_w)
            ry = sr["y"] - top
            if rx + sr["w"] > 0 and ry + sr["h"] > 0 and rx < right_w and ry < ch:
                # clamp
                rx = max(0, rx)
                ry = max(0, ry)
                rw = min(sr["w"], right_w - rx)
                rh = min(sr["h"], ch - ry)
                if rw > 8 and rh > 8:
                    # solid mask + slight blur edge (delogo is picky on odd sizes)
                    rw -= rw % 2
                    rh -= rh % 2
                    if rw >= 8 and rh >= 8:
                        vf_right += (
                            f",drawbox=x={rx}:y={ry}:w={rw}:h={rh}:color=black@1:t=fill"
                        )
                        report["streamer_masked"] = True

        _ffmpeg_run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(input_path),
                "-vf",
                vf_left,
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                str(shooter_out),
            ]
        )
        _ffmpeg_run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(input_path),
                "-vf",
                vf_right,
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                str(observer_out),
            ]
        )
    else:
        # Single cam: crop HUD only
        vf = f"crop={cw}:{ch}:{left}:{top},setsar=1"
        if mask_streamer and analysis.get("streamer_roi"):
            sr = analysis["streamer_roi"]
            rx = max(0, sr["x"] - left)
            ry = max(0, sr["y"] - top)
            rw = min(sr["w"], cw - rx)
            rh = min(sr["h"], ch - ry)
            rw -= rw % 2
            rh -= rh % 2
            if rw >= 8 and rh >= 8:
                vf += f",drawbox=x={rx}:y={ry}:w={rw}:h={rh}:color=black@1:t=fill"
                report["streamer_masked"] = True
        _ffmpeg_run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(input_path),
                "-vf",
                vf,
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                str(shooter_out),
            ]
        )
        report["observer"] = None

    report_path = out_dir / f"{name}_clean_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    report["report_path"] = str(report_path)
    return report


def clean_session_uploads(session_dir: Path) -> dict[str, Any]:
    """If session has a single SxS upload, split+clean into shooter/observer."""
    up = session_dir / "uploads"
    if not up.exists():
        return {"ok": False, "reason": "no uploads"}
    videos = sorted(
        [p for p in up.iterdir() if p.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"}]
    )
    if not videos:
        return {"ok": False, "reason": "no videos"}

    # If already two roles, still clean each for HUD/streamer
    clean_dir = session_dir / "uploads_clean"
    clean_dir.mkdir(exist_ok=True)
    results = []

    if len(videos) == 1:
        r = clean_and_split_sxs(videos[0], clean_dir, name="session")
        results.append(r)
        return {"ok": True, "mode": "sxs_split" if r["is_sxs"] else "single_clean", **r}

    # Two+ files: clean each independently (no force split)
    cleaned = []
    for i, v in enumerate(videos[:2]):
        role = "shooter" if i == 0 else "observer"
        r = clean_and_split_sxs(v, clean_dir, name=role, force_sxs=False)
        cleaned.append(r)
    return {
        "ok": True,
        "mode": "pair_clean",
        "shooter": cleaned[0]["shooter"],
        "observer": cleaned[1]["shooter"] if len(cleaned) > 1 else None,
        "results": cleaned,
    }
