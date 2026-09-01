"""Multi-view free-space carving for software 'through-wall' sensing.

Not RF hardware. Implements the video's capability: multi-view geometry reveals
structure/occupancy that a single 2D frame cannot — including targets behind
mesh/walls that appear only after 3D fusion.

Algorithm:
1. Voxelize scene bounds from point cloud
2. For each COLMAP camera, cast rays through free space until occupied voxels
3. Voxels never hit by free-space rays but near points = structure (walls)
4. Occupied voxels visible only from secondary views = "behind wall" returns
5. Export radar-style volume + occluded target list for Ghost-X RF mode
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np


def build_through_wall_volume(
    pointcloud_path: Path,
    targets: list[dict[str, Any]],
    *,
    grid: int = 56,
    out_path: Optional[Path] = None,
) -> dict[str, Any]:
    pc = json.loads(Path(pointcloud_path).read_text())
    pos = np.array(pc.get("positions") or [], dtype=np.float64).reshape(-1, 3)
    if len(pos) < 10:
        raise RuntimeError("Point cloud too sparse for through-wall carving")

    poses = pc.get("camera_poses") or []
    bmin = pc.get("bounds_min") or {}
    bmax = pc.get("bounds_max") or {}
    if isinstance(bmin, dict):
        mn = np.array([bmin.get("x", 0), bmin.get("y", 0), bmin.get("z", 0)], dtype=np.float64)
        mx = np.array([bmax.get("x", 1), bmax.get("y", 1), bmax.get("z", 1)], dtype=np.float64)
    else:
        mn = np.array(bmin, dtype=np.float64)
        mx = np.array(bmax, dtype=np.float64)

    span = np.maximum(mx - mn, 1e-3)
    mn = mn - span * 0.08
    mx = mx + span * 0.08
    span = mx - mn

    # Occupancy from points
    occ = np.zeros((grid, grid, grid), dtype=np.float32)
    idx = np.clip(((pos - mn) / span * (grid - 1e-6)).astype(np.int32), 0, grid - 1)
    for i, j, k in idx:
        occ[i, j, k] += 1.0
    if occ.max() > 0:
        occ = occ / occ.max()

    # Free-space carving: rays from cameras mark free voxels
    free = np.zeros_like(occ)
    cam_centers = []
    for p in poses[:40]:
        c = p.get("position")
        if not c:
            continue
        if isinstance(c, dict):
            C = np.array([c["x"], c["y"], c["z"]], dtype=np.float64)
        else:
            C = np.array(c, dtype=np.float64)
        cam_centers.append(C)
        # sample rays toward scene center + random occupied voxels
        center = (mn + mx) * 0.5
        dirs = [center - C]
        # also sample toward some point cloud samples
        if len(pos) > 20:
            sel = pos[np.linspace(0, len(pos) - 1, 24).astype(int)]
            for s in sel:
                dirs.append(s - C)
        for d in dirs:
            dn = np.linalg.norm(d)
            if dn < 1e-6:
                continue
            d = d / dn
            # step along ray
            for step in np.linspace(0.05, dn * 0.98, 40):
                X = C + d * step
                gi = np.clip(((X - mn) / span * (grid - 1e-6)).astype(int), 0, grid - 1)
                # stop when we hit occupancy
                if occ[gi[0], gi[1], gi[2]] > 0.12:
                    break
                free[gi[0], gi[1], gi[2]] = 1.0

    # Structure = occupied and not fully free
    thr = float(np.percentile(occ[occ > 0], 35)) if np.any(occ > 0) else 0.05
    structure = (occ > thr).astype(np.float32)
    # "Behind wall" returns: occupied voxels with free space between them and some camera
    occluded = np.zeros_like(occ)
    for i, j, k in np.argwhere(structure > 0.5):
        X = mn + (np.array([i, j, k], dtype=np.float64) + 0.5) * (span / grid)
        for C in cam_centers[:12]:
            d = X - C
            dn = np.linalg.norm(d)
            if dn < 1e-6:
                continue
            d = d / dn
            blocked = False
            free_before = False
            for step in np.linspace(0.05, dn * 0.92, 28):
                Y = C + d * step
                gi = np.clip(((Y - mn) / span * (grid - 1e-6)).astype(int), 0, grid - 1)
                if free[gi[0], gi[1], gi[2]] > 0.5:
                    free_before = True
                if structure[gi[0], gi[1], gi[2]] > 0.5 and free_before:
                    # hit another structure before target voxel
                    if tuple(gi) != (i, j, k):
                        blocked = True
                        break
            if blocked:
                occluded[i, j, k] = max(occluded[i, j, k], float(occ[i, j, k]))

    # Export voxels
    nz = np.argwhere((occ > 0) | (free > 0))
    step = max(1, len(nz) // 10000)
    voxels = []
    for i, j, k in nz[::step]:
        voxels.append(
            {
                "i": int(i),
                "j": int(j),
                "k": int(k),
                "d": float(occ[i, j, k]),
                "structure": bool(structure[i, j, k] > 0.5),
                "free": bool(free[i, j, k] > 0.5),
                "occluded": bool(occluded[i, j, k] > 0.05),
            }
        )

    # Target returns with occluded flag via ray test
    returns = []
    for t in targets or []:
        p = t.get("position_3d")
        if not p:
            continue
        X = np.array([p["x"], p["y"], p["z"]], dtype=np.float64)
        occl = False
        for C in cam_centers[:8]:
            d = X - C
            dn = np.linalg.norm(d)
            if dn < 1e-6:
                continue
            d = d / dn
            hit_wall = False
            for step in np.linspace(0.1, dn * 0.9, 20):
                Y = C + d * step
                gi = np.clip(((Y - mn) / span * (grid - 1e-6)).astype(int), 0, grid - 1)
                if structure[gi[0], gi[1], gi[2]] > 0.5:
                    hit_wall = True
                    break
            if hit_wall:
                occl = True
                break
        returns.append(
            {
                "id": t.get("id"),
                "label": t.get("label") or t.get("id"),
                "color": t.get("color") or "steel",
                "last_result": t.get("last_result"),
                "position": {"x": float(X[0]), "y": float(X[1]), "z": float(X[2])},
                "behind_wall": occl,
                "hits": t.get("hits"),
                "misses": t.get("misses"),
            }
        )

    origin = {"x": float(cam_centers[0][0]), "y": float(cam_centers[0][1]), "z": float(cam_centers[0][2])} if cam_centers else {
        "x": float(mn[0]),
        "y": float(mn[1]),
        "z": float(mn[2]),
    }

    doc = {
        "grid": grid,
        "bounds_min": {"x": float(mn[0]), "y": float(mn[1]), "z": float(mn[2])},
        "bounds_max": {"x": float(mx[0]), "y": float(mx[1]), "z": float(mx[2])},
        "origin": origin,
        "voxels": voxels,
        "returns": returns,
        "point_count": int(len(pos)),
        "voxel_count": len(voxels),
        "occluded_voxels": int((occluded > 0.05).sum()),
        "cameras_used": len(cam_centers),
        "method": "multiview_freespace_carving",
        "note": "Software through-wall sensing via multi-view free-space carving — not RF hardware",
    }
    if out_path:
        Path(out_path).write_text(json.dumps(doc))
        doc["path"] = str(out_path)
    return doc
