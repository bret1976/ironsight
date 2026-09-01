"""3D Gaussian Splatting training + export for IronSight God's Eye View.

Primary trainer: OpenSplat (Metal MPS on Apple Silicon / CPU fallback)
  https://github.com/pierotofy/OpenSplat

Input: COLMAP project (images + sparse model with points)
Output: splat.ply (INRIA 3DGS format) + optional compressed .splat

No mocks — if OpenSplat is missing or training fails, raises with a real error
unless allow_init_fallback=True, which still writes a real Gaussian PLY
initialized from COLMAP points (valid 3DGS schema, unoptimized).
"""
from __future__ import annotations

import json
import math
import os
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Any, Optional

import numpy as np


def opensplat_binary() -> Optional[str]:
    """Locate opensplat executable."""
    try:
        from app.core.config import get_settings

        cfg = (get_settings().opensplat_bin or "").strip()
        if cfg and Path(cfg).exists():
            return cfg
    except Exception:
        pass
    env = os.environ.get("OPENSPLAT_BIN", "").strip()
    if env and Path(env).exists():
        return env
    which = shutil.which("opensplat")
    if which:
        return which
    root = Path(__file__).resolve().parents[3]
    for cand in (
        root / "tools" / "OpenSplat" / "build" / "opensplat",
        root / "third_party" / "OpenSplat" / "build" / "opensplat",
        Path.home() / "OpenSplat" / "build" / "opensplat",
        Path("/opt/homebrew/bin/opensplat"),
        Path("/usr/local/bin/opensplat"),
    ):
        if cand.exists() and os.access(cand, os.X_OK):
            return str(cand)
    return None


def opensplat_available() -> bool:
    return opensplat_binary() is not None


def opensplat_runtime() -> str:
    """Return 'mps' | 'cpu' | 'none' for the installed OpenSplat binary."""
    try:
        from app.core.config import get_settings

        cfg = (get_settings().opensplat_runtime or "").strip().lower()
        if cfg in {"mps", "cpu", "metal"}:
            return "mps" if cfg == "metal" else cfg
    except Exception:
        pass
    env = os.environ.get("OPENSPLAT_RUNTIME", "").strip().lower()
    if env in {"mps", "cpu", "metal"}:
        return "mps" if env == "metal" else env
    binary = opensplat_binary()
    if not binary:
        return "none"
    stamp = Path(binary).parent / "opensplat.runtime"
    if stamp.exists():
        val = stamp.read_text().strip().upper()
        if val == "MPS":
            return "mps"
        if val == "CPU":
            return "cpu"
    # Heuristic: metal framework linked?
    try:
        out = subprocess.run(
            ["otool", "-L", binary], capture_output=True, text=True, timeout=10
        )
        if "Metal" in (out.stdout or "") or "metal" in (out.stdout or "").lower():
            return "mps"
    except Exception:
        pass
    return "cpu"


def metal_toolchain_available() -> bool:
    """True when xcrun can find metal + metallib (full Xcode)."""
    try:
        a = subprocess.run(
            ["xcrun", "--find", "metal"], capture_output=True, text=True, timeout=5
        )
        b = subprocess.run(
            ["xcrun", "--find", "metallib"], capture_output=True, text=True, timeout=5
        )
        return a.returncode == 0 and b.returncode == 0
    except Exception:
        return False


def prepare_colmap_project_for_opensplat(
    colmap_workspace: Path,
    images_dir: Path,
    project_dir: Path,
) -> Path:
    """
    OpenSplat expects a COLMAP project folder containing:
      images/  (or image files referenced by sparse)
      sparse/0/  cameras.bin images.bin points3D.bin

    We assemble a clean project_dir from our recon artifacts.
    """
    project_dir = Path(project_dir)
    if project_dir.exists():
        shutil.rmtree(project_dir)
    project_dir.mkdir(parents=True)

    # Images
    img_out = project_dir / "images"
    img_out.mkdir()
    src_images = Path(images_dir)
    if not src_images.exists():
        raise RuntimeError(f"COLMAP images missing: {src_images}")
    for p in sorted(src_images.iterdir()):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            shutil.copy2(p, img_out / p.name)

    # Sparse model — prefer richest model
    ws = Path(colmap_workspace)
    sparse_src = _best_sparse(ws)
    if sparse_src is None:
        raise RuntimeError(f"No COLMAP sparse model in {ws}")

    sparse_dst = project_dir / "sparse" / "0"
    sparse_dst.mkdir(parents=True)
    for name in (
        "cameras.bin",
        "images.bin",
        "points3D.bin",
        "cameras.txt",
        "images.txt",
        "points3D.txt",
        "frames.bin",
        "rigs.bin",
    ):
        src = sparse_src / name
        if src.exists():
            shutil.copy2(src, sparse_dst / name)

    # Ensure binary model exists (OpenSplat prefers bin)
    if not (sparse_dst / "points3D.bin").exists() and (sparse_dst / "points3D.txt").exists():
        subprocess.run(
            [
                "colmap",
                "model_converter",
                "--input_path",
                str(sparse_dst),
                "--output_path",
                str(sparse_dst),
                "--output_type",
                "BIN",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )

    if not (sparse_dst / "points3D.bin").exists() and not (sparse_dst / "points3D.txt").exists():
        raise RuntimeError("COLMAP project has no points3D — cannot init Gaussians")

    return project_dir


def _best_sparse(workspace: Path) -> Optional[Path]:
    candidates = []
    for p in workspace.rglob("points3D.bin"):
        candidates.append(p.parent)
    for p in workspace.rglob("points3D.txt"):
        if p.parent not in candidates:
            candidates.append(p.parent)
    if not candidates:
        return None

    def score(d: Path) -> int:
        bin_p = d / "points3D.bin"
        txt_p = d / "points3D.txt"
        if bin_p.exists():
            try:
                with open(bin_p, "rb") as f:
                    return int(struct.unpack("<Q", f.read(8))[0])
            except Exception:
                return bin_p.stat().st_size
        if txt_p.exists():
            return sum(1 for line in txt_p.read_text(errors="ignore").splitlines() if line and not line.startswith("#"))
        return 0

    candidates.sort(key=score, reverse=True)
    return candidates[0]


def train_opensplat(
    colmap_project: Path,
    out_dir: Path,
    *,
    num_iters: int = 7000,
    output_name: str = "splat.ply",
    max_gaussians: Optional[int] = None,
    resume: Optional[Path] = None,
) -> dict[str, Any]:
    """Run OpenSplat training. Produces splat.ply in out_dir."""
    binary = opensplat_binary()
    if not binary:
        raise RuntimeError(
            "OpenSplat not found. Build it (tools/OpenSplat) or set OPENSPLAT_BIN."
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_ply = out_dir / output_name
    out_splat = out_dir / "scene.splat"

    # Usage: opensplat /path/to/colmap_project -n N -o splat.ply -d 2
    runtime = opensplat_runtime()
    cmd = [
        binary,
        str(Path(colmap_project).resolve()),
        "-n",
        str(int(num_iters)),
        "-o",
        str(out_ply.resolve()),
        # Downscale more for long runs so quality trains finish on Metal
        "-d",
        "4" if int(num_iters) >= 1500 else "2",
        "--num-downscales",
        "1",
        "--sh-degree",
        "1",
    ]
    # Only force --cpu when binary is a CPU build. Metal/MPS builds must NOT get --cpu.
    if runtime != "mps":
        cmd.append("--cpu")
    if resume and Path(resume).exists():
        cmd.extend(["--resume", str(Path(resume).resolve())])

    log_path = out_dir / "opensplat.log"
    env = os.environ.copy()
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    # Ensure OpenCV dylibs resolve
    ocv_lib = "/opt/homebrew/opt/opencv/lib"
    env["DYLD_LIBRARY_PATH"] = ocv_lib + ":" + env.get("DYLD_LIBRARY_PATH", "")
    env["DYLD_FALLBACK_LIBRARY_PATH"] = ocv_lib + ":" + env.get("DYLD_FALLBACK_LIBRARY_PATH", "")

    # Generous timeout: Metal can vary; long runs need hours of headroom
    # Empirically ~0.2–1s/step with -d 4; use 8s/step budget + 30min base
    timeout_s = max(1800, int(num_iters) * 8 + 1800)

    # Stream stdout/stderr to log (capture_output can fill the OS pipe and hang OpenSplat)
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("CMD: " + " ".join(cmd) + "\n")
        log.write(f"runtime={runtime} timeout_s={timeout_s}\n\n")
        log.flush()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(out_dir),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as e:
            log.write(f"\nTIMEOUT after {timeout_s}s\n")
            raise RuntimeError(f"OpenSplat timed out after {timeout_s}s") from e
        log.write(f"\nexit={proc.returncode}\n")

    # OpenSplat may write splat.ply in cwd regardless of -o
    candidates = [
        out_ply,
        out_dir / "splat.ply",
        out_dir / "point_cloud.ply",
        out_dir / "output.ply",
        out_splat,
    ]
    # Also search for any new .ply/.splat
    candidates.extend(sorted(out_dir.glob("*.ply")))
    candidates.extend(sorted(out_dir.glob("*.splat")))

    result_path: Optional[Path] = None
    for c in candidates:
        if c.exists() and c.stat().st_size > 500:
            # Prefer files that look like gaussian PLYs (have scale/opacity props)
            if c.suffix.lower() == ".ply":
                head = c.read_bytes()[:800].decode("latin-1", errors="ignore")
                if "scale_0" in head or "opacity" in head or "f_dc_0" in head or "rot_0" in head:
                    result_path = c
                    break
            elif c.suffix.lower() == ".splat":
                result_path = c
                break
    if result_path is None:
        for c in candidates:
            if c.exists() and c.stat().st_size > 500:
                result_path = c
                break

    if result_path is None:
        raise RuntimeError(
            f"OpenSplat failed (exit {proc.returncode}). See {log_path}. "
            f"stderr: {(proc.stderr or '')[-800:]}"
        )

    # Normalize names
    final_ply = out_dir / "splat.ply"
    final_splat = out_dir / "scene.splat"
    if result_path.suffix.lower() == ".ply" and result_path.resolve() != final_ply.resolve():
        shutil.copy2(result_path, final_ply)
    elif result_path.suffix.lower() == ".splat":
        shutil.copy2(result_path, final_splat)
        # Keep splat as primary if no ply
        if not final_ply.exists():
            final_ply = final_splat

    # Also produce compressed .splat if we only have ply and opensplat supports it via re-run
    # Count gaussians from ply header
    count = _count_gaussians_in_file(final_ply if final_ply.exists() else result_path)

    meta = {
        "path": str(final_ply if final_ply.exists() else result_path),
        "splat_path": str(final_splat) if final_splat.exists() else None,
        "ply_path": str(final_ply) if final_ply.exists() else str(result_path),
        "gaussian_count": count,
        "method": "opensplat_3dgs",
        "num_iters": num_iters,
        "log": str(log_path),
        "trainer": binary,
        "device": runtime,
        "opensplat_runtime": runtime,
        "metal": runtime == "mps",
    }
    (out_dir / "splat_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def _count_gaussians_in_file(path: Path) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    if path.suffix.lower() == ".splat":
        # antimatter15 .splat: 32 bytes per gaussian
        return path.stat().st_size // 32
    data = path.read_bytes()
    header_end = data.find(b"end_header")
    if header_end < 0:
        return 0
    header = data[:header_end].decode("ascii", errors="ignore")
    for line in header.splitlines():
        if line.startswith("element vertex"):
            try:
                return int(line.split()[-1])
            except ValueError:
                return 0
    return 0


def init_gaussians_from_pointcloud(
    points_xyz: np.ndarray,
    colors_rgb: np.ndarray,
    out_ply: Path,
    *,
    scale_percentile: float = 50.0,
) -> dict[str, Any]:
    """
    Write a valid INRIA-style 3DGS PLY initialized from a point cloud.

    Each point becomes a Gaussian with:
      - mean = xyz
      - scale from kNN distance
      - identity rotation quaternion
      - opacity ~0.5 (logit space stored as raw opacity in PLY viewers often expect activated)
      - SH DC from RGB

    This is a real Gaussian scene file (viewable in GaussianSplats3D), used as
    OpenSplat fallback / resume init — not a mock mesh.
    """
    pts = np.asarray(points_xyz, dtype=np.float64)
    cols = np.asarray(colors_rgb, dtype=np.float64)
    if cols.max() > 1.5:
        cols = cols / 255.0
    n = len(pts)
    if n < 8:
        raise RuntimeError("Need at least 8 points to init Gaussians")

    # kNN scale estimate (k=3 mean distance)
    # For large n, subsample for distance estimation
    rng = np.random.default_rng(0)
    if n > 8000:
        sample_idx = rng.choice(n, 8000, replace=False)
        sample = pts[sample_idx]
    else:
        sample = pts
        sample_idx = np.arange(n)

    # Pairwise approx via chunked distances to 32 random anchors
    anchors = sample[rng.choice(len(sample), min(64, len(sample)), replace=False)]
    dmin = np.full(n, np.inf)
    for a in anchors:
        d = np.linalg.norm(pts - a, axis=1)
        # exclude self-ish
        d = np.where(d < 1e-8, np.inf, d)
        dmin = np.minimum(dmin, d)
    # Global median scale
    finite = dmin[np.isfinite(dmin)]
    base_scale = float(np.percentile(finite, scale_percentile)) if len(finite) else 0.02
    base_scale = max(base_scale * 0.35, 1e-4)
    scales = np.full((n, 3), base_scale, dtype=np.float32)
    # slight anisotropy noise for visual depth
    scales *= (1.0 + 0.15 * rng.standard_normal((n, 3)).astype(np.float32))
    scales = np.clip(scales, 1e-4, base_scale * 8)

    # Quaternion wxyz identity
    rots = np.zeros((n, 4), dtype=np.float32)
    rots[:, 0] = 1.0

    # Opacity (pre-sigmoid value ~0 so sigmoid~0.5; many viewers expect activated opacity in [0,1])
    opacity = np.full((n, 1), 0.7, dtype=np.float32)

    # SH DC: C0 = 0.5 / sqrt(pi) ≈ 0.2820947918; sh = (rgb - 0.5) / C0
    C0 = 0.28209479177387814
    sh_dc = ((cols - 0.5) / C0).astype(np.float32)

    out_ply = Path(out_ply)
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    _write_3dgs_ply(out_ply, pts.astype(np.float32), scales, rots, opacity, sh_dc)

    meta = {
        "path": str(out_ply),
        "ply_path": str(out_ply),
        "splat_path": None,
        "gaussian_count": n,
        "method": "colmap_init_3dgs",
        "num_iters": 0,
        "note": "Initialized Gaussians from COLMAP points (unoptimized). Train with OpenSplat for full quality.",
    }
    (out_ply.parent / "splat_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def _write_3dgs_ply(
    path: Path,
    means: np.ndarray,
    scales: np.ndarray,
    rots: np.ndarray,
    opacity: np.ndarray,
    sh_dc: np.ndarray,
) -> None:
    """Write INRIA 3DGS PLY (ascii) readable by GaussianSplats3D / SuperSplat."""
    n = len(means)
    # Zero higher-order SH (degree 0 only is fine for init / many viewers)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        for prop in (
            "x", "y", "z",
            "nx", "ny", "nz",
            "f_dc_0", "f_dc_1", "f_dc_2",
            "opacity",
            "scale_0", "scale_1", "scale_2",
            "rot_0", "rot_1", "rot_2", "rot_3",
        ):
            f.write(f"property float {prop}\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = means[i]
            # log-scale as stored by official 3DGS
            s0, s1, s2 = np.log(np.maximum(scales[i], 1e-8))
            r0, r1, r2, r3 = rots[i]
            # opacity as inverse-sigmoid of [0,1] value
            o = float(opacity[i, 0])
            o = min(max(o, 1e-4), 1 - 1e-4)
            o_logit = math.log(o / (1 - o))
            dc0, dc1, dc2 = sh_dc[i]
            f.write(
                f"{x:.6f} {y:.6f} {z:.6f} 0 0 0 "
                f"{dc0:.6f} {dc1:.6f} {dc2:.6f} "
                f"{o_logit:.6f} "
                f"{s0:.6f} {s1:.6f} {s2:.6f} "
                f"{r0:.6f} {r1:.6f} {r2:.6f} {r3:.6f}\n"
            )


def run_gaussian_splatting(
    *,
    colmap_workspace: Path,
    images_dir: Path,
    out_dir: Path,
    points_xyz: Optional[np.ndarray] = None,
    colors_rgb: Optional[np.ndarray] = None,
    num_iters: int = 7000,
    allow_init_fallback: bool = True,
    pytorch_iters: int = 400,
) -> dict[str, Any]:
    """
    Full 3DGS stage (priority order):
      1. OpenSplat (Metal/CPU) if binary available
      2. Pure PyTorch photometric 3DGS trainer (MPS/CPU)
      3. COLMAP-initialized Gaussian PLY (valid 3DGS schema, unoptimized)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    project = out_dir / "gs_colmap_project"
    errors: list[str] = []

    try:
        prepare_colmap_project_for_opensplat(colmap_workspace, images_dir, project)
    except Exception as e:
        errors.append(f"colmap_project: {e}")
        project = None  # type: ignore

    from app.core.config import get_settings
    from app.services import gaussian_train

    s = get_settings()
    runtime = opensplat_runtime()
    prefer_opensplat = bool(getattr(s, "gsplat_prefer_opensplat", True))
    prefer_pytorch_mps = bool(getattr(s, "gsplat_prefer_pytorch_mps", True))
    torch_ok = gaussian_train.torch_available()
    torch_dev = gaussian_train.device_str() if torch_ok else "none"

    def _run_opensplat() -> dict[str, Any]:
        os_iters = max(500, int(num_iters))
        meta = train_opensplat(project, out_dir, num_iters=os_iters)
        meta["device"] = runtime if runtime != "none" else "cpu"
        meta["opensplat_runtime"] = runtime
        return meta

    def _run_pytorch() -> dict[str, Any]:
        model_dir = _best_sparse(Path(colmap_workspace))
        if model_dir is None:
            raise RuntimeError("No COLMAP sparse model for PyTorch 3DGS")
        meta = gaussian_train.train_3dgs_pytorch(
            model_dir,
            images_dir,
            out_dir,
            num_iters=max(pytorch_iters, int(getattr(s, "gsplat_pytorch_iters", 500))),
            image_scale=float(getattr(s, "gsplat_image_scale", 0.30)),
            max_points=int(getattr(s, "gsplat_max_points", 8000)),
            max_train_views=int(getattr(s, "gsplat_max_views", 32)),
        )
        meta["fallback_errors"] = errors
        meta["metal"] = torch_dev == "mps"
        return meta

    # Training order:
    #  1) OpenSplat Metal (true MPS kernels) if runtime=mps
    #  2) PyTorch MPS (Metal GPU) when preferred or OpenSplat is CPU-only
    #  3) OpenSplat CPU
    #  4) PyTorch CPU
    tried_opensplat = False
    tried_pytorch = False

    if opensplat_available() and project is not None and runtime == "mps" and prefer_opensplat:
        tried_opensplat = True
        try:
            return _run_opensplat()
        except Exception as e:
            errors.append(f"opensplat_mps: {e}")
            (out_dir / "opensplat_error.txt").write_text(str(e))

    if torch_ok and torch_dev == "mps" and prefer_pytorch_mps:
        tried_pytorch = True
        try:
            return _run_pytorch()
        except Exception as e:
            errors.append(f"pytorch_mps: {e}")
            (out_dir / "pytorch_3dgs_error.txt").write_text(str(e))

    if opensplat_available() and project is not None and not tried_opensplat:
        tried_opensplat = True
        try:
            return _run_opensplat()
        except Exception as e:
            errors.append(f"opensplat: {e}")
            (out_dir / "opensplat_error.txt").write_text(str(e))

    if torch_ok and not tried_pytorch:
        tried_pytorch = True
        try:
            return _run_pytorch()
        except Exception as e:
            errors.append(f"pytorch_3dgs: {e}")
            (out_dir / "pytorch_3dgs_error.txt").write_text(str(e))

    if allow_init_fallback and points_xyz is not None and colors_rgb is not None:
        meta = init_gaussians_from_pointcloud(
            points_xyz, colors_rgb, out_dir / "splat.ply"
        )
        meta["fallback_errors"] = errors
        meta["method"] = "colmap_init_3dgs"
        return meta

    if allow_init_fallback:
        try:
            pts, cols = _load_points_from_colmap_workspace(colmap_workspace)
            meta = init_gaussians_from_pointcloud(pts, cols, out_dir / "splat.ply")
            meta["fallback_errors"] = errors
            return meta
        except Exception as e:
            errors.append(f"init_from_colmap: {e}")

    raise RuntimeError("3DGS failed: " + "; ".join(errors))


def _load_points_from_colmap_workspace(workspace: Path) -> tuple[np.ndarray, np.ndarray]:
    model = _best_sparse(Path(workspace))
    if model is None:
        raise RuntimeError("no sparse model")
    txt = model / "points3D.txt"
    if not txt.exists():
        subprocess.run(
            [
                "colmap", "model_converter",
                "--input_path", str(model),
                "--output_path", str(model),
                "--output_type", "TXT",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
    pts = []
    cols = []
    for line in txt.read_text(errors="ignore").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        cols.append([int(parts[4]), int(parts[5]), int(parts[6])])
    if not pts:
        raise RuntimeError("empty points3D")
    return np.array(pts, dtype=np.float64), np.array(cols, dtype=np.float64)
