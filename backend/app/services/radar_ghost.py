"""Through-wall radar volume — delegates to multi-view free-space carving."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from app.services.through_wall import build_through_wall_volume


def build_radar_volume(
    pointcloud_path: Path,
    targets: list[dict[str, Any]],
    *,
    grid: int = 56,
    out_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Build multi-view free-space carving volume (RF-style Ghost-X)."""
    return build_through_wall_volume(
        pointcloud_path, targets, grid=grid, out_path=out_path
    )
