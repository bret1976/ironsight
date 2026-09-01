"""Public GodsEye intel routes — no auth required (keyless public feeds)."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.services import godseye_intel as intel

router = APIRouter(prefix="/godseye", tags=["godseye"])


class CommandBody(BaseModel):
    text: str = Field("", max_length=400)
    lat: float | None = None
    lon: float | None = None
    alt: float | None = None


@router.get("/config")
async def godseye_config():
    return intel.tile_config(get_settings())


@router.get("/flights")
async def godseye_flights(
    lat: float = Query(35.0, ge=-90, le=90),
    lon: float = Query(-30.0, ge=-180, le=180),
    radius_nm: float = Query(450, ge=25, le=1500),
    include_mil: bool = True,
):
    return await intel.fetch_flights(lat, lon, radius_nm, include_mil)


@router.get("/tle")
async def godseye_tle(
    groups: str = Query("stations,visual,weather"),
):
    wanted = [g.strip().lower() for g in groups.split(",") if g.strip()]
    return await intel.fetch_tles(wanted)


@router.get("/omm")
async def godseye_omm(
    groups: str = Query("stations,visual,weather"),
):
    """Same sat catalog as /tle, labeled as CelesTrak OMM JSON."""
    wanted = [g.strip().lower() for g in groups.split(",") if g.strip()]
    return await intel.fetch_tles(wanted)


@router.get("/quakes")
async def godseye_quakes():
    return await intel.fetch_quakes()


@router.get("/ships")
async def godseye_ships(
    lat: float = Query(35.0, ge=-90, le=90),
    lon: float = Query(-30.0, ge=-180, le=180),
    radius_nm: float = Query(450, ge=25, le=1500),
    bbox: Optional[str] = Query(None, description="minLat,minLon,maxLat,maxLon"),
):
    from app.services import godseye_ais as ais

    return await ais.fetch_ships(lat, lon, radius_nm, bbox=bbox)


@router.get("/geocode")
async def godseye_geocode(q: str = Query("", min_length=0, max_length=120)):
    return await intel.geocode(q)


@router.get("/fires")
async def godseye_fires():
    return await intel.fetch_fires()


@router.get("/storms")
async def godseye_storms():
    return await intel.fetch_storms()


@router.get("/hazards")
async def godseye_hazards():
    """Alias for /storms — EONET + NWS hazard contacts."""
    return await intel.fetch_storms()


@router.get("/sites")
async def godseye_sites(
    lat: float = Query(35.0, ge=-90, le=90),
    lon: float = Query(-30.0, ge=-180, le=180),
    radius_km: float = Query(80, ge=10, le=200),
):
    return await intel.fetch_sites(lat, lon, radius_km)


@router.get("/radio")
async def godseye_radio(
    lat: float = Query(35.0, ge=-90, le=90),
    lon: float = Query(-30.0, ge=-180, le=180),
):
    return await intel.fetch_radio(lat, lon)


@router.get("/missions")
async def godseye_missions():
    return await intel.fetch_missions()


@router.get("/cctv")
async def godseye_cctv():
    return await intel.fetch_cctv()


@router.get("/cctv/media")
async def godseye_cctv_media(u: str = Query("", max_length=500)):
    """Proxy an allowlisted city still. HLS and off-list hosts are rejected."""
    return await intel.proxy_cctv_media(u)


@router.get("/satstill")
async def godseye_satstill(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    z: int = Query(15, ge=4, le=19),
):
    """Photoreal satellite still for the lock card (Google, then ESRI)."""
    return await intel.proxy_sat_still(lat, lon, z, get_settings())


@router.post("/command")
async def godseye_command(body: CommandBody):
    from app.services.godseye_command import interpret_command

    return await interpret_command(
        body.text,
        {"lat": body.lat, "lon": body.lon, "alt": body.alt},
    )
