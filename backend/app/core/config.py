from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
SESSION_DIR = DATA_DIR / "sessions"
RECON_DIR = DATA_DIR / "reconstructions"
DB_PATH = DATA_DIR / "ironsight.db"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "IronSight"
    app_version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8741
    # Comma-separated. Never use * with credentials in production.
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8741"
    # development | production
    environment: str = "development"
    # Optional shared secret for mutating API routes (X-IronSight-Key header)
    api_key: str = ""
    # JWT secret for user sessions (REQUIRED in production)
    jwt_secret: str = ""
    # When true, all /api routes except health/auth/billing webhook need JWT
    require_auth: bool = False
    # Public site URL for Stripe redirects
    public_url: str = "http://127.0.0.1:5173"
    api_public_url: str = "http://127.0.0.1:8741"
    # Max single upload size in bytes (default 2 GiB)
    max_upload_bytes: int = 2_147_483_648
    # Request log level
    log_level: str = "INFO"

    # Stripe (chargeable product)
    stripe_secret_key: str = ""
    stripe_publishable_key: str = ""
    stripe_webhook_secret: str = ""
    stripe_price_pro: str = ""
    stripe_price_team: str = ""

    # SMTP (password reset / invites) — optional; falls back to data/mail_outbox/
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = "noreply@ironsight.local"
    # Support + admin
    support_email: str = "support@ironsight.local"
    admin_api_key: str = ""
    # Demo session id exposed to all tenants for onboarding
    demo_session_id: str = "8b27bffb48a6"
    # Session retention days (0 = keep forever)
    session_retention_days: int = 90
    # Rate limit
    rate_limit_per_minute: int = 120

    # OpenRouter (Gemini / vision models) — real inference, never mocked
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    vision_model: str = "google/gemini-2.5-flash"
    coaching_model: str = "google/gemini-2.5-flash"

    # Optional xAI fallback
    xai_api_key: str = ""
    xai_base_url: str = "https://api.x.ai/v1"

    # Shot detection (impulsive gunshot gates — rejects speech/music)
    # profile: auto | range | mixed — range is strict pure live-fire model
    shot_profile: str = "auto"
    shot_min_interval_s: float = 0.28
    shot_onset_delta: float = 0.40
    shot_energy_percentile: float = 99.3
    shot_bandpass_low_hz: float = 80.0
    shot_bandpass_high_hz: float = 8000.0
    shot_max_per_session: int = 40

    # Frame extraction around each shot for VLM
    # Wider window + more frames → fewer UNKNOWNs on dual-cam
    hit_miss_pre_s: float = 0.25
    hit_miss_post_s: float = 0.75
    hit_miss_frame_count: int = 6
    # When true, Team analyze jobs prefer durable queue (start worker)
    use_job_queue: bool = False

    # 3D reconstruction (denser defaults; COLMAP preferred when installed)
    recon_max_frames: int = 72
    recon_sample_fps: float = 3.0
    recon_max_features: int = 8000
    recon_point_budget: int = 200_000
    recon_prefer_colmap: bool = True
    # Dense MVS is expensive on CPU; sparse COLMAP is default, dense optional
    recon_colmap_dense: bool = False

    # 3D Gaussian Splatting (full production path)
    gsplat_enabled: bool = True
    # Longer Metal OpenSplat quality default (UI Retrain uses this)
    gsplat_opensplat_iters: int = 3000
    gsplat_pytorch_iters: int = 800
    gsplat_image_scale: float = 0.30
    gsplat_max_points: int = 8000
    gsplat_max_views: int = 32
    gsplat_allow_init_fallback: bool = True
    # Optional GodsEye photoreal 3D tiles (client-side; never required)
    cesium_ion_token: str = ""
    google_maps_api_key: str = ""
    # Optional keyed AIS (ships). Leave blank → keyless Open Waters snapshot.
    aisstream_api_key: str = ""
    aishub_username: str = ""
    marinetraffic_api_key: str = ""

    # Absolute path to opensplat binary (auto-detected if empty)
    opensplat_bin: str = ""
    # cpu | mps — stamped by build_opensplat.sh / setup_metal_opensplat.sh
    opensplat_runtime: str = "cpu"
    # Training preference
    gsplat_prefer_opensplat: bool = True
    # When true and torch MPS available, run PyTorch Metal path before OpenSplat CPU
    gsplat_prefer_pytorch_mps: bool = True

    data_dir: Path = DATA_DIR
    upload_dir: Path = UPLOAD_DIR
    session_dir: Path = SESSION_DIR
    recon_dir: Path = RECON_DIR
    db_path: Path = DB_PATH

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.upload_dir, self.session_dir, self.recon_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    s = Settings(
        openrouter_api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        xai_api_key=os.environ.get("XAI_API_KEY", ""),
        jwt_secret=os.environ.get("JWT_SECRET", os.environ.get("IRONSIGHT_JWT_SECRET", "")),
        stripe_secret_key=os.environ.get("STRIPE_SECRET_KEY", ""),
        stripe_publishable_key=os.environ.get("STRIPE_PUBLISHABLE_KEY", ""),
        stripe_webhook_secret=os.environ.get("STRIPE_WEBHOOK_SECRET", ""),
        stripe_price_pro=os.environ.get("STRIPE_PRICE_PRO", ""),
        stripe_price_team=os.environ.get("STRIPE_PRICE_TEAM", ""),
        public_url=os.environ.get("PUBLIC_URL", "http://127.0.0.1:5173"),
        api_public_url=os.environ.get("API_PUBLIC_URL", "http://127.0.0.1:8741"),
        require_auth=os.environ.get("REQUIRE_AUTH", "").lower() in ("1", "true", "yes"),
        environment=os.environ.get("ENVIRONMENT", "development"),
        cesium_ion_token=os.environ.get("CESIUM_ION_TOKEN")
        or os.environ.get("VITE_CESIUM_ION_TOKEN", ""),
        google_maps_api_key=os.environ.get("GOOGLE_MAPS_API_KEY")
        or os.environ.get("VITE_GOOGLE_MAPS_API_KEY", ""),
        aisstream_api_key=os.environ.get("AISSTREAM_API_KEY", ""),
        aishub_username=os.environ.get("AISHUB_USERNAME", ""),
        marinetraffic_api_key=os.environ.get("MARINETRAFFIC_API_KEY", ""),
    )
    s.ensure_dirs()
    return s
