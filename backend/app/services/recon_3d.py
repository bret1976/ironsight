"""Multi-view structure reconstruction from real video frames.

Primary path: COLMAP automatic reconstructor (SfM + optional MVS) when installed.
Fallback: denser OpenCV multi-pair SfM (SIFT → essential matrix → triangulation)
plus multi-frame optical-flow densification.

No synthetic room geometry, no canned meshes.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


def colmap_available() -> bool:
    return shutil.which("colmap") is not None


def _detect_and_match(img1: np.ndarray, img2: np.ndarray, max_features: int = 8000):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    if hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(nfeatures=max_features)
        norm = cv2.NORM_L2
    else:
        detector = cv2.ORB_create(nfeatures=max_features)
        norm = cv2.NORM_HAMMING

    k1, d1 = detector.detectAndCompute(gray1, None)
    k2, d2 = detector.detectAndCompute(gray2, None)
    if d1 is None or d2 is None or len(k1) < 12 or len(k2) < 12:
        return None

    if norm == cv2.NORM_L2:
        matcher = cv2.BFMatcher(norm, crossCheck=False)
        knn = matcher.knnMatch(d1, d2, k=2)
        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.72 * n.distance:
                good.append(m)
    else:
        matcher = cv2.BFMatcher(norm, crossCheck=True)
        good = matcher.match(d1, d2)
        good = sorted(good, key=lambda m: m.distance)[:4000]

    if len(good) < 20:
        return None

    pts1 = np.float32([k1[m.queryIdx].pt for m in good])
    pts2 = np.float32([k2[m.trainIdx].pt for m in good])
    colors = np.array([img1[int(p[1]), int(p[0])] for p in pts1], dtype=np.float32)
    return pts1, pts2, colors


def _intrinsics(w: int, h: int) -> np.ndarray:
    f = 0.9 * max(w, h)
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]], dtype=np.float64)


def _write_ply(path: Path, pts: np.ndarray, cols: np.ndarray) -> None:
    n = len(pts)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(pts, cols):
            b, g, r = [int(np.clip(v, 0, 255)) for v in c]
            f.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {r} {g} {b}\n")


def _export_cloud(
    out_dir: Path,
    pts: np.ndarray,
    cols: np.ndarray,
    poses: list[dict[str, Any]],
    method: str,
    point_budget: int = 200_000,
    view_budget: int = 100_000,
) -> dict[str, Any]:
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    cols = cols[finite]
    if len(pts) == 0:
        raise RuntimeError("No valid 3D points after filtering")

    if len(pts) >= 20:
        lo = np.percentile(pts, 1, axis=0)
        hi = np.percentile(pts, 99, axis=0)
        keep = np.all((pts >= lo) & (pts <= hi), axis=1)
        # Don't wipe the cloud if filter is too aggressive
        if keep.sum() >= max(10, int(0.2 * len(pts))):
            pts = pts[keep]
            cols = cols[keep]
            lo = pts.min(axis=0)
            hi = pts.max(axis=0)
        else:
            lo = pts.min(axis=0)
            hi = pts.max(axis=0)
    else:
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)

    rng = np.random.default_rng(42)
    if len(pts) > point_budget:
        idx = rng.choice(len(pts), point_budget, replace=False)
        pts, cols = pts[idx], cols[idx]

    ply_path = out_dir / "pointcloud.ply"
    _write_ply(ply_path, pts, cols)

    if len(pts) > view_budget:
        idx = rng.choice(len(pts), view_budget, replace=False)
        vpts, vcols = pts[idx], cols[idx]
    else:
        vpts, vcols = pts, cols

    json_path = out_dir / "pointcloud.json"
    payload = {
        "count": int(len(vpts)),
        "positions": vpts.astype(np.float32).reshape(-1).tolist(),
        "colors": (vcols[:, ::-1] / 255.0).astype(np.float32).reshape(-1).tolist(),
        "bounds_min": lo.tolist(),
        "bounds_max": hi.tolist(),
        "camera_poses": poses,
        "method": method,
    }
    json_path.write_text(json.dumps(payload))

    meta = {
        "path": str(json_path),
        "ply_path": str(ply_path),
        "point_count": int(len(vpts)),
        "bounds_min": {"x": float(lo[0]), "y": float(lo[1]), "z": float(lo[2])},
        "bounds_max": {"x": float(hi[0]), "y": float(hi[1]), "z": float(hi[2])},
        "camera_poses": poses,
        "method": method,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def _run_colmap(
    image_paths: list[Path],
    out_dir: Path,
    *,
    point_budget: int = 200_000,
    dense: bool = False,
) -> dict[str, Any]:
    """Run COLMAP on a frame set. Prefers automatic_reconstructor, then manual pipeline."""
    if not colmap_available():
        raise RuntimeError("colmap binary not found on PATH")

    images_dir = out_dir / "colmap_images"
    workspace = out_dir / "colmap_ws"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    if workspace.exists():
        shutil.rmtree(workspace)
    images_dir.mkdir(parents=True)
    workspace.mkdir(parents=True)

    for i, p in enumerate(image_paths):
        dest = images_dir / f"{i:05d}{p.suffix.lower() or '.jpg'}"
        shutil.copy2(p, dest)

    def run(cmd: list[str], timeout: int = 2400) -> subprocess.CompletedProcess:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError(
                f"COLMAP failed ({' '.join(cmd[:3])}…): {(proc.stderr or proc.stdout)[-1500:]}"
            )
        return proc

    # Prefer the high-level automatic reconstructor (COLMAP 3.x/4.x)
    quality = "high" if dense else "medium"
    try:
        run(
            [
                "colmap", "automatic_reconstructor",
                "--image_path", str(images_dir),
                "--workspace_path", str(workspace),
                "--quality", quality,
                "--dense", "1" if dense else "0",
                "--use_gpu", "0",
            ],
            timeout=3600 if dense else 1800,
        )
    except Exception as auto_err:
        # Manual sparse pipeline fallback
        (out_dir / "colmap_auto_error.txt").write_text(str(auto_err))
        db = workspace / "database.db"
        sparse = workspace / "sparse"
        sparse.mkdir(exist_ok=True)
        run(
            [
                "colmap", "feature_extractor",
                "--database_path", str(db),
                "--image_path", str(images_dir),
                "--ImageReader.single_camera", "1",
                "--ImageReader.camera_model", "SIMPLE_RADIAL",
                "--SiftExtraction.use_gpu", "0",
            ]
        )
        run(
            [
                "colmap", "exhaustive_matcher",
                "--database_path", str(db),
                "--SiftMatching.use_gpu", "0",
            ]
        )
        run(
            [
                "colmap", "mapper",
                "--database_path", str(db),
                "--image_path", str(images_dir),
                "--output_path", str(sparse),
            ]
        )

    # Locate sparse model (auto puts it in sparse/0 usually)
    model_dir = _find_colmap_model(workspace)
    if model_dir is None:
        raise RuntimeError("COLMAP produced no sparse model")

    sparse_ply = out_dir / "colmap_sparse.ply"
    run(
        [
            "colmap", "model_converter",
            "--input_path", str(model_dir),
            "--output_path", str(sparse_ply),
            "--output_type", "PLY",
        ]
    )
    pts, cols = _load_ply_xyz_rgb(sparse_ply)
    method = "colmap_sparse"

    # If PLY is empty/near-empty, parse points3D.txt/bin via convert to TXT
    if len(pts) < 50:
        try:
            pts2, cols2 = _load_colmap_points3d(model_dir)
            if len(pts2) > len(pts):
                pts, cols = pts2, cols2
        except Exception as e:
            (out_dir / "points3d_parse_error.txt").write_text(str(e))

    # Dense fused cloud if automatic reconstructor wrote one
    for cand in (
        workspace / "dense" / "0" / "fused.ply",
        workspace / "dense" / "fused.ply",
    ):
        if cand.exists():
            try:
                dpts, dcols = _load_ply_xyz_rgb(cand)
                if len(dpts) > len(pts):
                    pts, cols = dpts, dcols
                    method = "colmap_dense"
            except Exception:
                pass
            break

    if len(pts) < 30:
        raise RuntimeError(
            f"COLMAP model too sparse ({len(pts)} points) — falling back to OpenCV dense"
        )

    poses = _colmap_poses_from_model(model_dir)
    return _export_cloud(out_dir, pts, cols, poses, method, point_budget=point_budget)


def _load_colmap_points3d(model_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load points from COLMAP points3D.txt (convert from bin if needed)."""
    txt = model_dir / "points3D.txt"
    if not txt.exists():
        subprocess.run(
            [
                "colmap", "model_converter",
                "--input_path", str(model_dir),
                "--output_path", str(model_dir),
                "--output_type", "TXT",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
    if not txt.exists():
        raise RuntimeError(f"No points3D.txt in {model_dir}")
    pts = []
    cols = []
    for line in txt.read_text(errors="ignore").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        # POINT3D_ID X Y Z R G B ERROR ...
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
        pts.append([x, y, z])
        cols.append([b, g, r])  # BGR storage
    if not pts:
        raise RuntimeError("points3D.txt had no points")
    return np.array(pts, dtype=np.float32), np.array(cols, dtype=np.float32)


def _count_colmap_points(model_dir: Path) -> int:
    """Estimate number of 3D points in a COLMAP model directory."""
    txt = model_dir / "points3D.txt"
    if txt.exists():
        n = 0
        for line in txt.read_text(errors="ignore").splitlines():
            if line and not line.startswith("#"):
                n += 1
        return n
    bin_path = model_dir / "points3D.bin"
    if bin_path.exists():
        # points3D.bin starts with uint64 count
        try:
            import struct

            with open(bin_path, "rb") as f:
                (count,) = struct.unpack("<Q", f.read(8))
            return int(count)
        except Exception:
            return int(bin_path.stat().st_size // 64)  # rough
    return 0


def _find_colmap_model(workspace: Path) -> Optional[Path]:
    """Find the sparse model with the most 3D points."""
    candidates: list[Path] = []
    for p in workspace.rglob("points3D.bin"):
        candidates.append(p.parent)
    for p in workspace.rglob("points3D.txt"):
        candidates.append(p.parent)
    # unique
    seen = set()
    uniq: list[Path] = []
    for c in candidates:
        key = str(c.resolve())
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    if not uniq:
        return None
    scored = [(_count_colmap_points(c), str(c), c) for c in uniq]
    scored.sort(reverse=True)
    return scored[0][2]


def _load_ply_xyz_rgb(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Minimal PLY reader for xyz (+ optional rgb) ascii/binary via OpenCV/open3d-free path."""
    # Prefer OpenCV not available for PLY; parse ASCII or use numpy from text
    text = path.read_bytes()
    # Try ascii parse
    try:
        header_end = text.find(b"end_header")
        if header_end < 0:
            raise ValueError("no end_header")
        header = text[: header_end].decode("ascii", errors="ignore")
        body = text[header_end + len(b"end_header") :]
        if body.startswith(b"\n"):
            body = body[1:]
        is_binary = "format binary" in header
        n_vert = 0
        props = []
        for line in header.splitlines():
            if line.startswith("element vertex"):
                n_vert = int(line.split()[-1])
            if line.startswith("property"):
                props.append(line.split()[-1])
        if n_vert == 0:
            raise ValueError("no vertices")

        has_rgb = all(c in props for c in ("red", "green", "blue"))
        if is_binary:
            # little endian float x y z + uchar rgb common
            # Use a simple struct unpack for float32 xyz and optional uchar rgb
            import struct

            pts = []
            cols = []
            # Determine stride
            fmt_parts = []
            for line in header.splitlines():
                if not line.startswith("property"):
                    continue
                parts = line.split()
                typ, name = parts[1], parts[2]
                fmt_parts.append((typ, name))
            # map types
            type_map = {
                "float": "f",
                "float32": "f",
                "double": "d",
                "float64": "d",
                "uchar": "B",
                "uint8": "B",
                "char": "b",
                "int": "i",
                "int32": "i",
                "uint": "I",
                "uint32": "I",
                "short": "h",
                "ushort": "H",
            }
            endian = "<" if "binary_little_endian" in header else ">"
            struct_fmt = endian + "".join(type_map.get(t, "f") for t, _ in fmt_parts)
            stride = struct.calcsize(struct_fmt)
            names = [n for _, n in fmt_parts]
            for i in range(n_vert):
                chunk = body[i * stride : (i + 1) * stride]
                if len(chunk) < stride:
                    break
                vals = struct.unpack(struct_fmt, chunk)
                d = dict(zip(names, vals))
                pts.append([d.get("x", 0), d.get("y", 0), d.get("z", 0)])
                if has_rgb:
                    cols.append([d.get("blue", 128), d.get("green", 128), d.get("red", 128)])  # BGR
                else:
                    cols.append([180, 180, 180])
            return np.array(pts, dtype=np.float32), np.array(cols, dtype=np.float32)

        # ASCII
        lines = body.decode("ascii", errors="ignore").strip().splitlines()
        pts = []
        cols = []
        for line in lines[:n_vert]:
            parts = line.split()
            if len(parts) < 3:
                continue
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            pts.append([x, y, z])
            if has_rgb and len(parts) >= 6:
                r, g, b = int(float(parts[3])), int(float(parts[4])), int(float(parts[5]))
                cols.append([b, g, r])  # store BGR for consistency
            else:
                cols.append([180, 180, 180])
        return np.array(pts, dtype=np.float32), np.array(cols, dtype=np.float32)
    except Exception as e:
        raise RuntimeError(f"Failed to parse PLY {path}: {e}") from e


def _colmap_poses_from_model(model_dir: Path) -> list[dict[str, Any]]:
    """Read camera centers from COLMAP images.txt if present, else empty."""
    images_txt = model_dir / "images.txt"
    if not images_txt.exists():
        # convert bin → txt
        try:
            subprocess.run(
                [
                    "colmap", "model_converter",
                    "--input_path", str(model_dir),
                    "--output_path", str(model_dir),
                    "--output_type", "TXT",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
        except Exception:
            return []
    if not images_txt.exists():
        return []

    poses = []
    lines = images_txt.read_text().splitlines()
    i = 0
    idx = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        parts = line.split()
        if len(parts) >= 10:
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            # Camera center C = -R^T t
            R = _quat_to_R(qw, qx, qy, qz)
            t = np.array([[tx], [ty], [tz]], dtype=np.float64)
            C = (-R.T @ t).flatten()
            poses.append(
                {
                    "index": idx,
                    "path": parts[9],
                    "R": R.tolist(),
                    "t": t.flatten().tolist(),
                    "position": C.tolist(),
                }
            )
            idx += 1
            i += 2  # skip points2D line
        else:
            i += 1
    return poses


def _quat_to_R(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def reconstruct_opencv_dense(
    image_paths: list[str | Path],
    out_dir: str | Path,
    *,
    max_features: int = 8000,
    point_budget: int = 200_000,
    pair_stride: int = 1,
    max_pair_gap: int = 3,
) -> dict[str, Any]:
    """Denser OpenCV multi-pair SfM: match frame i to i+1..i+gap, accumulate points."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [Path(p) for p in image_paths if Path(p).exists()]
    if len(paths) < 2:
        raise RuntimeError("Need at least 2 frames for 3D reconstruction")

    images = []
    for p in paths:
        im = cv2.imread(str(p))
        if im is None:
            continue
        h, w = im.shape[:2]
        scale = min(1.0, 1280.0 / max(w, h))
        if scale < 1.0:
            im = cv2.resize(im, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        images.append(im)
    if len(images) < 2:
        raise RuntimeError("Could not load enough frames for reconstruction")

    h0, w0 = images[0].shape[:2]
    K = _intrinsics(w0, h0)

    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    poses: list[dict[str, Any]] = []

    # Sequential pose chain for primary trajectory
    R_total = np.eye(3, dtype=np.float64)
    t_total = np.zeros((3, 1), dtype=np.float64)
    pose_R = [R_total.copy()]
    pose_t = [t_total.copy()]
    poses.append(
        {
            "index": 0,
            "path": str(paths[0]),
            "R": R_total.tolist(),
            "t": t_total.flatten().tolist(),
            "position": [0.0, 0.0, 0.0],
        }
    )

    for i in range(len(images) - 1):
        match = _detect_and_match(images[i], images[i + 1], max_features=max_features)
        if match is None:
            pose_R.append(R_total.copy())
            pose_t.append(t_total.copy())
            pos = (-R_total.T @ t_total).flatten()
            poses.append(
                {
                    "index": i + 1,
                    "path": str(paths[min(i + 1, len(paths) - 1)]),
                    "R": R_total.tolist(),
                    "t": t_total.flatten().tolist(),
                    "position": pos.tolist(),
                    "skipped": True,
                }
            )
            continue

        pts1, pts2, colors = match
        E, mask_e = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            pose_R.append(R_total.copy())
            pose_t.append(t_total.copy())
            continue
        mask_e = mask_e.ravel().astype(bool)
        pts1_i, pts2_i, colors_i = pts1[mask_e], pts2[mask_e], colors[mask_e]
        if len(pts1_i) < 12:
            pose_R.append(R_total.copy())
            pose_t.append(t_total.copy())
            continue

        _, R, t, mask_pose = cv2.recoverPose(E, pts1_i, pts2_i, K)
        mask_pose = mask_pose.ravel().astype(bool)
        pts1_i, pts2_i, colors_i = pts1_i[mask_pose], pts2_i[mask_pose], colors_i[mask_pose]
        if len(pts1_i) < 8:
            pose_R.append(R_total.copy())
            pose_t.append(t_total.copy())
            continue

        R_rel = R.astype(np.float64)
        t_rel = t.astype(np.float64)
        R_abs = R_rel @ R_total
        t_abs = R_rel @ t_total + t_rel

        P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = K @ np.hstack([R_rel, t_rel])
        pts4d = cv2.triangulatePoints(P1, P2, pts1_i.T, pts2_i.T)
        pts3d = (pts4d[:3] / (pts4d[3] + 1e-9)).T
        z1 = pts3d[:, 2]
        pts3d_cam2 = (R_rel @ pts3d.T + t_rel).T
        z2 = pts3d_cam2[:, 2]
        valid = (z1 > 0.05) & (z2 > 0.05) & np.isfinite(pts3d).all(axis=1)
        if valid.sum() > 10:
            depths = z1[valid]
            lo_d, hi_d = np.percentile(depths, [5, 95])
            valid = valid & (z1 >= lo_d) & (z1 <= hi_d * 1.5)

        pts3d = pts3d[valid]
        colors_i = colors_i[valid]
        if len(pts3d):
            pts_world = (R_total.T @ (pts3d.T - t_total)).T
            all_points.append(pts_world.astype(np.float32))
            all_colors.append(colors_i.astype(np.float32))

        R_total, t_total = R_abs, t_abs
        pose_R.append(R_total.copy())
        pose_t.append(t_total.copy())
        pos = (-R_total.T @ t_total).flatten()
        poses.append(
            {
                "index": i + 1,
                "path": str(paths[min(i + 1, len(paths) - 1)]),
                "R": R_total.tolist(),
                "t": t_total.flatten().tolist(),
                "position": pos.tolist(),
                "inliers": int(len(pts3d)),
            }
        )

    # Extra pair gaps for denser coverage (i ↔ i+2, i+3)
    for gap in range(2, max_pair_gap + 1):
        for i in range(0, len(images) - gap, pair_stride):
            j = i + gap
            match = _detect_and_match(images[i], images[j], max_features=max_features // 2)
            if match is None:
                continue
            pts1, pts2, colors = match
            E, mask_e = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            if E is None:
                continue
            mask_e = mask_e.ravel().astype(bool)
            pts1_i, pts2_i, colors_i = pts1[mask_e], pts2[mask_e], colors[mask_e]
            if len(pts1_i) < 16:
                continue
            _, R, t, mask_pose = cv2.recoverPose(E, pts1_i, pts2_i, K)
            mask_pose = mask_pose.ravel().astype(bool)
            pts1_i, pts2_i, colors_i = pts1_i[mask_pose], pts2_i[mask_pose], colors_i[mask_pose]
            if len(pts1_i) < 12:
                continue
            R_rel = R.astype(np.float64)
            t_rel = t.astype(np.float64)
            P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
            P2 = K @ np.hstack([R_rel, t_rel])
            pts4d = cv2.triangulatePoints(P1, P2, pts1_i.T, pts2_i.T)
            pts3d = (pts4d[:3] / (pts4d[3] + 1e-9)).T
            z1 = pts3d[:, 2]
            pts3d_cam2 = (R_rel @ pts3d.T + t_rel).T
            z2 = pts3d_cam2[:, 2]
            valid = (z1 > 0.05) & (z2 > 0.05) & np.isfinite(pts3d).all(axis=1)
            pts3d = pts3d[valid]
            colors_i = colors_i[valid]
            if len(pts3d) < 8:
                continue
            # Transform from camera-i into world using stored pose
            Ri, ti = pose_R[i], pose_t[i]
            pts_world = (Ri.T @ (pts3d.T - ti)).T
            all_points.append(pts_world.astype(np.float32))
            all_colors.append(colors_i.astype(np.float32))

    # Optical-flow densification between sequential pairs for more surface fill
    for i in range(0, len(images) - 1, max(1, len(images) // 12)):
        try:
            dens_pts, dens_cols = _optical_flow_pair(images[i], images[i + 1], K)
            if dens_pts is not None and len(dens_pts):
                Ri, ti = pose_R[i], pose_t[i]
                pts_world = (Ri.T @ (dens_pts.T - ti)).T
                all_points.append(pts_world.astype(np.float32))
                all_colors.append(dens_cols.astype(np.float32))
        except Exception:
            continue

    if not all_points:
        return _fallback_optical_flow_cloud(images, paths, out_dir, poses)

    pts = np.vstack(all_points)
    cols = np.vstack(all_colors)
    return _export_cloud(
        out_dir, pts, cols, poses, method="opencv_sfm_dense", point_budget=point_budget
    )


def _optical_flow_pair(
    im0: np.ndarray, im1: np.ndarray, K: np.ndarray, step: Optional[int] = None
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    g0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    h, w = g0.shape
    step = step or max(2, min(h, w) // 160)
    ys, xs = np.mgrid[0:h:step, 0:w:step]
    fx = flow[ys, xs, 0]
    fy = flow[ys, xs, 1]
    mag = np.sqrt(fx * fx + fy * fy) + 1e-3
    # Only keep moving pixels (parallax / motion)
    mask = mag > np.percentile(mag, 60)
    if mask.sum() < 50:
        return None, None
    depth = 1.0 / mag
    depth = np.clip(depth, np.percentile(depth[mask], 10), np.percentile(depth[mask], 90))
    fx_k, fy_k = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X = (xs - cx) * depth / fx_k
    Y = (ys - cy) * depth / fy_k
    Z = depth
    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float32)
    cols = im0[ys, xs].reshape(-1, 3).astype(np.float32)
    m = mask.reshape(-1)
    return pts[m], cols[m]


def _fallback_optical_flow_cloud(
    images: list[np.ndarray],
    paths: list[Path],
    out_dir: Path,
    poses: list[dict],
) -> dict[str, Any]:
    im0 = images[0]
    im1 = images[min(1, len(images) - 1)]
    h, w = im0.shape[:2]
    K = _intrinsics(w, h)
    pts, cols = _optical_flow_pair(im0, im1, K, step=max(2, min(h, w) // 200))
    if pts is None or len(pts) == 0:
        raise RuntimeError("Optical-flow fallback produced no points")
    return _export_cloud(
        out_dir, pts, cols, poses or [{"index": 0, "position": [0, 0, 0]}],
        method="optical_flow_lift",
    )


def reconstruct_from_images(
    image_paths: list[str | Path],
    out_dir: str | Path,
    *,
    max_features: int = 8000,
    point_budget: int = 200_000,
    prefer_colmap: bool = True,
    colmap_dense: bool = True,
) -> dict[str, Any]:
    """Build point cloud + camera poses. Prefer COLMAP when available."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [Path(p) for p in image_paths if Path(p).exists()]
    if len(paths) < 2:
        raise RuntimeError("Need at least 2 frames for 3D reconstruction")

    errors: list[str] = []
    if prefer_colmap and colmap_available() and len(paths) >= 5:
        try:
            return _run_colmap(paths, out_dir, point_budget=point_budget, dense=colmap_dense)
        except Exception as e:
            errors.append(f"colmap: {e}")
            (out_dir / "colmap_error.txt").write_text(str(e))

    try:
        meta = reconstruct_opencv_dense(
            paths, out_dir, max_features=max_features, point_budget=point_budget
        )
        if errors:
            meta["fallback_from"] = errors
        return meta
    except Exception as e:
        errors.append(f"opencv: {e}")
        raise RuntimeError("; ".join(errors)) from e


def place_targets_in_scene(
    pointcloud_meta: dict[str, Any],
    targets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assign 3D positions using real reconstructed bounds."""
    bmin = pointcloud_meta.get("bounds_min") or {"x": -1, "y": -1, "z": 1}
    bmax = pointcloud_meta.get("bounds_max") or {"x": 1, "y": 1, "z": 5}
    if isinstance(bmin, list):
        bmin = {"x": bmin[0], "y": bmin[1], "z": bmin[2]}
    if isinstance(bmax, list):
        bmax = {"x": bmax[0], "y": bmax[1], "z": bmax[2]}

    n = max(1, len(targets))
    placed = []
    for i, t in enumerate(targets):
        u = (i + 0.5) / n
        x = bmin["x"] + u * (bmax["x"] - bmin["x"])
        y = bmin["y"] + 0.35 * (bmax["y"] - bmin["y"])
        z = bmin["z"] + 0.55 * (bmax["z"] - bmin["z"]) + 0.1 * (u - 0.5)
        tt = dict(t)
        tt["position_3d"] = {"x": float(x), "y": float(y), "z": float(z)}
        placed.append(tt)
    return placed
