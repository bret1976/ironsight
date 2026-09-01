from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class SessionStatus(str, Enum):
    created = "created"
    uploading = "uploading"
    processing = "processing"
    ready = "ready"
    failed = "failed"


class PipelineStage(str, Enum):
    queued = "queued"
    ingest = "ingest"
    audio_extract = "audio_extract"
    shot_detection = "shot_detection"
    dual_cam_sync = "dual_cam_sync"
    hit_miss = "hit_miss"
    target_map = "target_map"
    reconstruction_3d = "reconstruction_3d"
    coaching = "coaching"
    finalize = "finalize"
    done = "done"
    error = "error"


class Vec3(BaseModel):
    x: float
    y: float
    z: float


class ShotClassification(str, Enum):
    hit = "HIT"
    miss = "MISS"
    unknown = "UNKNOWN"


class ShotEvent(BaseModel):
    id: int
    timestamp_s: float
    duration_s: float = 0.05
    energy: float = 0.0
    peak_db: float = 0.0
    classification: ShotClassification = ShotClassification.unknown
    confidence: float = 0.0
    target_id: Optional[str] = None
    target_label: Optional[str] = None
    target_color: Optional[str] = None
    reasoning: Optional[str] = None
    frame_paths: list[str] = Field(default_factory=list)
    position_3d: Optional[Vec3] = None
    camera: str = "primary"  # primary | observer | synced


class TargetMarker(BaseModel):
    id: str
    label: str
    color: str = "white"
    kind: str = "plate"  # plate | silhouette | steel
    position_3d: Optional[Vec3] = None
    hits: int = 0
    misses: int = 0
    last_result: Optional[ShotClassification] = None


class CameraTrack(BaseModel):
    role: str  # shooter_fpv | observer_3p
    filename: str
    path: str
    duration_s: float = 0.0
    fps: float = 30.0
    width: int = 0
    height: int = 0
    has_audio: bool = True


class SyncResult(BaseModel):
    offset_s: float
    offset_frames: float
    peak_ratio: float
    method: str = "audio_cross_correlation"
    passed: bool = True
    details: dict[str, Any] = Field(default_factory=dict)


class PointCloudMeta(BaseModel):
    path: str
    point_count: int
    bounds_min: Vec3
    bounds_max: Vec3
    camera_poses: list[dict[str, Any]] = Field(default_factory=list)
    method: str = "opencv_sfm"


class SessionStats(BaseModel):
    total_shots: int = 0
    hits: int = 0
    misses: int = 0
    accuracy: float = 0.0
    duration_s: float = 0.0
    best_split_s: Optional[float] = None
    avg_split_s: Optional[float] = None


class PipelineEvent(BaseModel):
    session_id: str
    stage: PipelineStage
    progress: float = 0.0  # 0-1
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    ts: datetime = Field(default_factory=datetime.utcnow)


class SessionSummary(BaseModel):
    id: str
    name: str
    status: SessionStatus
    stage: PipelineStage = PipelineStage.queued
    created_at: datetime
    updated_at: datetime
    cameras: list[CameraTrack] = Field(default_factory=list)
    stats: SessionStats = Field(default_factory=SessionStats)
    error: Optional[str] = None


class SessionDetail(SessionSummary):
    shots: list[ShotEvent] = Field(default_factory=list)
    targets: list[TargetMarker] = Field(default_factory=list)
    sync: Optional[SyncResult] = None
    pointcloud: Optional[PointCloudMeta] = None
    coaching: Optional[str] = None
    timeline_path: Optional[str] = None
    pipeline_log: list[PipelineEvent] = Field(default_factory=list)


class CreateSessionRequest(BaseModel):
    name: str = "Range Session"


class CoachingRequest(BaseModel):
    focus: Optional[str] = None
