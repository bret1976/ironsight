from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.auth_routes import router as auth_router
from app.api.godseye_routes import router as godseye_router
from app.api.routes import router
from app.api.scale_routes import router as scale_router
from app.api.value_routes import router as value_router
from app.core.config import ROOT, get_settings
from app.middleware.rate_limit import RateLimitMiddleware
from app.services import users as users_svc

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, (settings.log_level or "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("ironsight")


class UploadSizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject oversized request bodies early (upload safety)."""

    async def dispatch(self, request: Request, call_next):
        cl = request.headers.get("content-length")
        if cl and cl.isdigit():
            if int(cl) > settings.max_upload_bytes:
                return JSONResponse(
                    {"detail": f"Upload exceeds limit of {settings.max_upload_bytes} bytes"},
                    status_code=413,
                )
        return await call_next(request)


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Optional API key gate for mutating routes when API_KEY is configured."""

    MUTATING = {"POST", "PUT", "PATCH", "DELETE"}

    async def dispatch(self, request: Request, call_next):
        key = (settings.api_key or os.environ.get("IRONSIGHT_API_KEY") or "").strip()
        if not key:
            return await call_next(request)
        path = request.url.path
        # Health + SPA always open; GET media open
        # Public: health, auth, plans, stripe webhook
        public_prefixes = (
            "/api/health",
            "/api/auth/",
            "/api/plans",
            "/api/billing/webhook",
            "/api/sla",
            "/api/legal/",
            "/api/auth/magic-link",
            "/api/godseye",
        )
        if not path.startswith("/api") or any(
            path == p or path.startswith(p) for p in public_prefixes
        ):
            return await call_next(request)
        if request.method in self.MUTATING or path.startswith("/api/ws"):
            provided = request.headers.get("x-ironsight-key") or request.query_params.get("api_key")
            if provided != key:
                return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_dirs()
    users_svc.init_db()
    users_svc.ensure_admin_bootstrap()
    if not settings.openrouter_api_key:
        key = os.environ.get("OPENROUTER_API_KEY", "")
        if key:
            object.__setattr__(settings, "openrouter_api_key", key)
    if not settings.xai_api_key:
        key = os.environ.get("XAI_API_KEY", "")
        if key:
            object.__setattr__(settings, "xai_api_key", key)
    # Mark demo session public
    try:
        from app.services import store

        demo = settings.demo_session_id
        if demo:
            doc = store.load_session(demo)
            if not doc.get("public_demo"):
                doc["public_demo"] = True
                store.save_session(doc)
    except Exception:
        pass
    log.info(
        "IronSight starting env=%s port=%s vision=%s opensplat=%s ais=%s",
        settings.environment,
        settings.port,
        bool(settings.openrouter_api_key or settings.xai_api_key),
        bool(settings.opensplat_bin or os.environ.get("OPENSPLAT_BIN")),
        bool(settings.aisstream_api_key or settings.aishub_username or settings.marinetraffic_api_key),
    )
    try:
        from app.services import godseye_ais

        if settings.aisstream_api_key:
            godseye_ais.ensure_stream()
    except Exception:
        pass
    try:
        import asyncio

        from app.services import godseye_intel

        asyncio.create_task(godseye_intel.warm_cctv())
    except Exception:
        pass
    yield
    try:
        from app.services import godseye_ais

        await godseye_ais.stop_stream()
    except Exception:
        pass
    log.info("IronSight shutting down")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="IronSight — 2D range video → 4D tactical reconstruction. Real pipelines only.",
    lifespan=lifespan,
    docs_url="/api/docs" if settings.environment != "production" else None,
    redoc_url=None,
    openapi_url="/api/openapi.json" if settings.environment != "production" else "/api/openapi.json",
)

origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
# Production: never append wildcard with credentials
if settings.environment == "production":
    allow_origins = origins or ["http://127.0.0.1:8741"]
else:
    # Dev convenience: configured origins only (no * with credentials)
    allow_origins = origins or [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8741",
        "http://127.0.0.1:8741",
    ]

app.add_middleware(UploadSizeLimitMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(ApiKeyMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/api")
app.include_router(godseye_router, prefix="/api")
app.include_router(scale_router, prefix="/api")
app.include_router(value_router, prefix="/api")
app.include_router(router, prefix="/api")

FRONTEND_DIST = ROOT / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def spa(full_path: str):
        if full_path.startswith("api/"):
            return JSONResponse({"detail": "Not found"}, status_code=404)
        index = FRONTEND_DIST / "index.html"
        candidate = FRONTEND_DIST / full_path
        if full_path and candidate.exists() and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(index)


def run():
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        app_dir=str(BACKEND_ROOT),
        log_level=(settings.log_level or "info").lower(),
    )


if __name__ == "__main__":
    run()
