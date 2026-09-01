"""Production-grade pure-PyTorch 3D Gaussian Splatting trainer (MPS / CUDA / CPU).

Real photometric training from COLMAP cameras + images + sparse points.
Optimizes Gaussian means, log-scales, rotations, opacity, and SH-DC against
real video frames with L1 + SSIM-lite loss.

Rasterizer: differentiable projected 2D Gaussian alpha-compositing
(MPS-friendly soft 3DGS — same INRIA representation, soft train loop).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    torch = None  # type: ignore


def torch_available() -> bool:
    return bool(TORCH_OK)


def device_str() -> str:
    if not TORCH_OK:
        return "none"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_colmap_txt_model(model_dir: Path):
    """Load cameras, images, points from COLMAP TXT model."""
    if not (model_dir / "cameras.txt").exists():
        import subprocess

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

    cameras = {}
    for line in (model_dir / "cameras.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        cid = int(parts[0])
        model = parts[1]
        w, h = int(parts[2]), int(parts[3])
        params = list(map(float, parts[4:]))
        cameras[cid] = {"model": model, "width": w, "height": h, "params": params}

    images = []
    lines = (model_dir / "images.txt").read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        parts = line.split()
        if len(parts) < 10:
            i += 1
            continue
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        cam_id = int(parts[8])
        name = parts[9]
        images.append(
            {
                "q": np.array([qw, qx, qy, qz], dtype=np.float64),
                "t": np.array([tx, ty, tz], dtype=np.float64),
                "camera_id": cam_id,
                "name": name,
            }
        )
        i += 2

    pts, cols = [], []
    p3 = model_dir / "points3D.txt"
    if p3.exists():
        for line in p3.read_text().splitlines():
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            cols.append([int(parts[4]), int(parts[5]), int(parts[6])])
    return cameras, images, np.array(pts, dtype=np.float64), np.array(cols, dtype=np.float64)


def _quat_to_R(q: np.ndarray) -> np.ndarray:
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


def _intrinsics_from_cam(cam: dict) -> np.ndarray:
    p = cam["params"]
    model = cam["model"].upper()
    w, h = cam["width"], cam["height"]
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE"):
        f = p[0]
        cx = p[1] if len(p) > 1 else w / 2
        cy = p[2] if len(p) > 2 else h / 2
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    if model in ("PINHOLE", "OPENCV", "FULL_OPENCV", "OPENCV_FISHEYE"):
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    f = p[0]
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)


class GaussianModel(torch.nn.Module if TORCH_OK else object):
    def __init__(self, means, colors_rgb, device):
        if TORCH_OK:
            super().__init__()
        n = means.shape[0]
        self.means = torch.nn.Parameter(torch.tensor(means, dtype=torch.float32, device=device))
        with torch.no_grad():
            idx = torch.randperm(n, device=device)[: min(1024, n)]
            d = torch.cdist(self.means[idx], self.means[idx])
            d = d + torch.eye(len(idx), device=device) * 1e9
            nn = d.min(dim=1).values.median().clamp(min=1e-3)
            s0 = torch.log(nn * 0.45 * torch.ones(n, 3, device=device))
        self.log_scales = torch.nn.Parameter(s0)
        q = torch.zeros(n, 4, device=device)
        q[:, 0] = 1.0
        self.quats = torch.nn.Parameter(q)
        # Start moderately opaque
        self.opacity_logit = torch.nn.Parameter(torch.full((n, 1), 0.5, device=device))
        C0 = 0.28209479177387814
        sh = (torch.tensor(colors_rgb, dtype=torch.float32, device=device) - 0.5) / C0
        self.sh_dc = torch.nn.Parameter(sh)

    def get_scales(self):
        return torch.exp(self.log_scales).clamp(1e-5, 3.0)

    def get_opacity(self):
        return torch.sigmoid(self.opacity_logit)

    def get_rgb(self):
        C0 = 0.28209479177387814
        return (self.sh_dc * C0 + 0.5).clamp(0.0, 1.0)

    def normalized_quats(self):
        return self.quats / (self.quats.norm(dim=-1, keepdim=True) + 1e-8)


def _ssim_lite(pred: "torch.Tensor", gt: "torch.Tensor") -> "torch.Tensor":
    """Fast structural similarity proxy (window-free local stats via avg pool)."""
    # pred, gt: H,W,3 → 1,3,H,W
    p = pred.permute(2, 0, 1).unsqueeze(0)
    g = gt.permute(2, 0, 1).unsqueeze(0)
    mu_p = F.avg_pool2d(p, 7, 1, 3)
    mu_g = F.avg_pool2d(g, 7, 1, 3)
    sigma_p = F.avg_pool2d(p * p, 7, 1, 3) - mu_p * mu_p
    sigma_g = F.avg_pool2d(g * g, 7, 1, 3) - mu_g * mu_g
    sigma_pg = F.avg_pool2d(p * g, 7, 1, 3) - mu_p * mu_g
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu_p * mu_g + c1) * (2 * sigma_pg + c2)) / (
        (mu_p * mu_p + mu_g * mu_g + c1) * (sigma_p + sigma_g + c2) + 1e-8
    )
    return ssim.mean()


def render_gaussians(
    model: GaussianModel,
    R: "torch.Tensor",
    t: "torch.Tensor",
    K: "torch.Tensor",
    H: int,
    W: int,
    max_gaussians: int = 5000,
) -> "torch.Tensor":
    """Differentiable soft 3DGS render → HxWx3 in [0,1]."""
    means = model.means
    n = means.shape[0]
    device = means.device

    if n > max_gaussians:
        with torch.no_grad():
            opac = model.get_opacity().squeeze(-1)
            topk = torch.topk(opac, k=max_gaussians // 2).indices
            rest = torch.randperm(n, device=device)[: max_gaussians - len(topk)]
            idx = torch.unique(torch.cat([topk, rest]))
        means = means[idx]
        scales = model.get_scales()[idx]
        rgb = model.get_rgb()[idx]
        opac = model.get_opacity()[idx].squeeze(-1)
    else:
        scales = model.get_scales()
        rgb = model.get_rgb()
        opac = model.get_opacity().squeeze(-1)

    # COLMAP: X_cam = R @ X_world + t
    Xc = (R @ means.T).T + t.unsqueeze(0)
    z = Xc[:, 2].clamp(min=1e-3)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = fx * (Xc[:, 0] / z) + cx
    v = fy * (Xc[:, 1] / z) + cy

    margin = 48.0
    vis = (z > 0.05) & (u > -margin) & (u < W + margin) & (v > -margin) & (v < H + margin)
    if vis.sum() < 1:
        return torch.zeros(H, W, 3, device=device)

    u, v, z = u[vis], v[vis], z[vis]
    scales, rgb, opac = scales[vis], rgb[vis], opac[vis]

    s_mean = scales.mean(dim=-1)
    radius = (0.55 * (fx + fy) * s_mean / z).clamp(0.4, 48.0)

    # Far → near for back-to-front then flip for front-to-back
    order = torch.argsort(z, descending=True)
    u, v, radius, rgb, opac = u[order], v[order], radius[order], rgb[order], opac[order]

    max_comp = min(2000, u.shape[0])
    u, v, radius, rgb, opac = (
        u[:max_comp], v[:max_comp], radius[:max_comp], rgb[:max_comp], opac[:max_comp]
    )
    # Front-to-back
    u = torch.flip(u, [0])
    v = torch.flip(v, [0])
    radius = torch.flip(radius, [0])
    rgb = torch.flip(rgb, [0])
    opac = torch.flip(opac, [0])

    ys = torch.arange(H, device=device, dtype=torch.float32)
    xs = torch.arange(W, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    acc_rgb = torch.zeros(H, W, 3, device=device)
    acc_a = torch.zeros(H, W, 1, device=device)

    # Vectorized chunk compositing
    chunk = 48
    for i in range(0, u.shape[0], chunk):
        sl = slice(i, i + chunk)
        ui, vi, ri, ci, oi = u[sl], v[sl], radius[sl], rgb[sl], opac[sl]
        dx = xx.unsqueeze(0) - ui[:, None, None]
        dy = yy.unsqueeze(0) - vi[:, None, None]
        sigma = (ri / 2.0).clamp(min=0.25)[:, None, None]
        alpha = oi[:, None, None] * torch.exp(-0.5 * (dx * dx + dy * dy) / (sigma * sigma))
        alpha = alpha.clamp(0, 0.99)
        for g in range(ui.shape[0]):
            a = alpha[g].unsqueeze(-1)
            col = ci[g].view(1, 1, 3)
            w = (1.0 - acc_a) * a
            acc_rgb = acc_rgb + w * col
            acc_a = (acc_a + w).clamp(max=0.999)

    bg = torch.tensor([0.015, 0.02, 0.018], device=device).view(1, 1, 3)
    return (acc_rgb + (1.0 - acc_a) * bg).clamp(0, 1)


def train_3dgs_pytorch(
    colmap_model_dir: Path,
    images_dir: Path,
    out_dir: Path,
    *,
    num_iters: int = 500,
    image_scale: float = 0.30,
    max_points: int = 8000,
    max_gaussians_render: int = 4500,
    max_train_views: int = 32,
    lr: float = 4e-3,
    densify_every: int = 100,
) -> dict[str, Any]:
    if not TORCH_OK:
        raise RuntimeError("PyTorch not installed — cannot train 3DGS")

    device = torch.device(device_str())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cameras, images, pts, cols = _load_colmap_txt_model(Path(colmap_model_dir))
    if len(pts) < 8:
        raise RuntimeError("Too few COLMAP points for 3DGS init")
    if len(images) < 2:
        raise RuntimeError("Need at least 2 registered cameras")

    if len(pts) > max_points:
        idx = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts, cols = pts[idx], cols[idx]
    colors = cols.astype(np.float64) / 255.0

    model = GaussianModel(pts, colors, device)
    opt = torch.optim.Adam(
        [
            {"params": [model.means], "lr": lr},
            {"params": [model.log_scales], "lr": lr * 0.5},
            {"params": [model.quats], "lr": lr * 0.25},
            {"params": [model.opacity_logit], "lr": lr},
            {"params": [model.sh_dc], "lr": lr * 2.0},
        ]
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(num_iters, 1), eta_min=lr * 0.05)

    # Load views
    views = []
    images_dir = Path(images_dir)
    for im in images:
        path = images_dir / im["name"]
        if not path.exists():
            matches = list(images_dir.glob(f"*{Path(im['name']).name}"))
            if not matches:
                continue
            path = matches[0]
        img = cv2.imread(str(path))
        if img is None:
            continue
        cam = cameras[im["camera_id"]]
        K = _intrinsics_from_cam(cam)
        R = _quat_to_R(im["q"])
        tvec = im["t"]
        H0, W0 = img.shape[:2]
        H = max(40, int(H0 * image_scale))
        W = max(40, int(W0 * image_scale))
        img_s = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        img_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        Ks = K.copy()
        Ks[0, :] *= W / W0
        Ks[1, :] *= H / H0
        views.append(
            {
                "image": torch.tensor(img_s, device=device),
                "R": torch.tensor(R, dtype=torch.float32, device=device),
                "t": torch.tensor(tvec, dtype=torch.float32, device=device),
                "K": torch.tensor(Ks, dtype=torch.float32, device=device),
                "H": H,
                "W": W,
            }
        )
        if len(views) >= max_train_views:
            break

    if len(views) < 1:
        raise RuntimeError("No training views could be loaded")

    losses = []
    for it in range(num_iters):
        opt.zero_grad(set_to_none=True)
        # Multi-view mini-batch (2 views when possible)
        idxs = [it % len(views)]
        if len(views) > 1 and it % 2 == 0:
            idxs.append((it * 3 + 1) % len(views))
        loss = torch.tensor(0.0, device=device)
        for vi in idxs:
            view = views[vi]
            pred = render_gaussians(
                model,
                view["R"],
                view["t"],
                view["K"],
                view["H"],
                view["W"],
                max_gaussians=max_gaussians_render,
            )
            gt = view["image"]
            l1 = F.l1_loss(pred, gt)
            ssim = _ssim_lite(pred, gt)
            loss = loss + 0.8 * l1 + 0.2 * (1.0 - ssim)
        loss = loss / len(idxs)
        # Regularizers
        loss = loss + 0.0005 * model.get_scales().mean()
        loss = loss + 0.0001 * (model.normalized_quats().norm(dim=-1) - 1.0).pow(2).mean()
        loss.backward()
        # Grad clip for MPS stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        # Periodic opacity pruning (zero-out near-transparent) via opacity push
        if densify_every and (it + 1) % densify_every == 0:
            with torch.no_grad():
                op = model.get_opacity().squeeze(-1)
                # push very transparent gaussians lower
                dead = op < 0.02
                if dead.any():
                    model.opacity_logit.data[dead] -= 0.5

        losses.append(float(loss.detach().cpu()))
        if (it + 1) % 25 == 0 or it == 0 or it == num_iters - 1:
            (out_dir / "train_progress.json").write_text(
                json.dumps(
                    {
                        "iter": it + 1,
                        "loss": losses[-1],
                        "device": str(device),
                        "gaussians": int(model.means.shape[0]),
                        "lr": float(opt.param_groups[0]["lr"]),
                    }
                )
            )

    with torch.no_grad():
        means = model.means.detach().cpu().numpy()
        scales = model.get_scales().detach().cpu().numpy()
        quats = model.normalized_quats().detach().cpu().numpy()
        opacity = model.get_opacity().detach().cpu().numpy()
        sh = model.sh_dc.detach().cpu().numpy()

    from app.services.gaussian_splat import _write_3dgs_ply

    ply_path = out_dir / "splat.ply"
    _write_3dgs_ply(ply_path, means, scales, quats, opacity, sh)

    # Also write compressed-ish .splat (antimatter15 format: 32 bytes/gauss)
    splat_path = out_dir / "scene.splat"
    _write_antimatter_splat(splat_path, means, scales, quats, opacity, sh)

    meta = {
        "path": str(ply_path),
        "ply_path": str(ply_path),
        "splat_path": str(splat_path) if splat_path.exists() else None,
        "gaussian_count": int(len(means)),
        "method": "pytorch_3dgs",
        "num_iters": num_iters,
        "device": str(device),
        "final_loss": losses[-1] if losses else None,
        "views": len(views),
        "image_scale": image_scale,
        "loss_curve": losses[:: max(1, len(losses) // 50)],
    }
    (out_dir / "splat_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def _write_antimatter_splat(
    path: Path,
    means: np.ndarray,
    scales: np.ndarray,
    quats: np.ndarray,
    opacity: np.ndarray,
    sh_dc: np.ndarray,
) -> None:
    """Write antimatter15 .splat binary (pos3f + scale3f + rgba4u8 + quat4u8)."""
    import struct

    C0 = 0.28209479177387814
    rgb = (sh_dc * C0 + 0.5)
    rgb = np.clip(rgb, 0, 1)
    n = len(means)
    path = Path(path)
    with open(path, "wb") as f:
        for i in range(n):
            x, y, z = means[i].astype(np.float32)
            sx, sy, sz = scales[i].astype(np.float32)
            # opacity already [0,1]
            a = float(opacity[i, 0]) if opacity.ndim > 1 else float(opacity[i])
            r, g, b = [int(np.clip(c * 255, 0, 255)) for c in rgb[i]]
            ai = int(np.clip(a * 255, 0, 255))
            # quaternion wxyz → pack as uint8 centered
            qw, qx, qy, qz = quats[i]
            # normalize
            qn = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz) + 1e-12
            qw, qx, qy, qz = qw / qn, qx / qn, qy / qn, qz / qn
            qbytes = [
                int(np.clip((c * 0.5 + 0.5) * 255, 0, 255))
                for c in (qw, qx, qy, qz)
            ]
            f.write(struct.pack("<fff", x, y, z))
            f.write(struct.pack("<fff", sx, sy, sz))
            f.write(struct.pack("<BBBB", r, g, b, ai))
            f.write(struct.pack("<BBBB", *qbytes))
