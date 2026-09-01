"""AIS ships for GodsEye.

Primary: keyless Open Waters snapshot (https://ais.openwaters.io/v1/vessels).
Optional secondary when a key is already configured:
  1. AISSTREAM_API_KEY  — live WebSocket (https://aisstream.io/apikeys)
  2. AISHUB_USERNAME    — REST snapshot (https://www.aishub.net/api)
  3. MARINETRAFFIC_API_KEY — REST export (paid)

Never invent positions. Keys stay on the server; the browser never sees them.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from typing import Any, Optional

import httpx

from app.core.config import get_settings

log = logging.getLogger("ironsight.godseye.ais")

AISSTREAM_URL = "wss://stream.aisstream.io/v0/stream"
AISHUB_URL = "https://data.aishub.net/ws.php"
MARINETRAFFIC_URL = "https://services.marinetraffic.com/api/exportvessels/v:8/{key}/timespan:10/protocol:jsono"
OPENWATERS_VESSELS = "https://ais.openwaters.io/v1/vessels"
# Anonymous Open Waters caps HTTP /v1/vessels at 120 req/min. Keep boxes small.
MAX_BBOX_SPAN_DEG = 8.0

# Busy shipping boxes used when the camera is looking at the whole Earth.
_HUBS = (
    (51.4, 1.5, 160),
    (1.26, 103.8, 140),
    (36.0, -5.5, 110),
    (29.95, 32.55, 140),
    (31.3, 121.7, 160),
    (29.7, -93.9, 160),
    (51.95, 4.3, 110),
    (22.3, 114.2, 110),
    (35.0, 139.8, 110),
    (40.65, -74.05, 140),
    (33.73, -118.26, 140),
    (25.25, 55.27, 140),
)

_CACHE: dict[str, dict[str, Any]] = {}
_TASK: Optional[asyncio.Task] = None
_STOP = asyncio.Event()
_BOXES: list[list[list[float]]] = []
_STREAM_OK = False
_STREAM_ERR = ""
_LAST_MSG_MS = 0

UA = "IronSightGodsEye/1.0 (+https://github.com/VegasCryptoAgent/ironsight)"


def _now_ms() -> int:
    return int(time.time() * 1000)


def _credentials() -> dict[str, str]:
    s = get_settings()
    return {
        "aisstream": (getattr(s, "aisstream_api_key", None) or "").strip(),
        "aishub": (getattr(s, "aishub_username", None) or "").strip(),
        "marinetraffic": (getattr(s, "marinetraffic_api_key", None) or "").strip(),
    }


def keyed_ships_provider() -> str:
    c = _credentials()
    if c["aisstream"]:
        return "aisstream"
    if c["aishub"]:
        return "aishub"
    if c["marinetraffic"]:
        return "marinetraffic"
    return ""


def ships_provider() -> str:
    keyed = keyed_ships_provider()
    if keyed:
        return f"openwaters+{keyed}"
    return "openwaters"


def ships_stub() -> dict[str, Any]:
    return {
        "kind": "ships",
        "status": "unavailable",
        "source": "openwaters",
        "provider": "openwaters",
        "freshness_ms": None,
        "label": "UNAVAILABLE",
        "simulated": False,
        "count": 0,
        "contacts": [],
        "errors": [],
        "note": "Open Waters AIS unreachable. Optional AISSTREAM_API_KEY remains unused.",
    }


def normalize_aisstream(msg: dict) -> Optional[dict[str, Any]]:
    if not isinstance(msg, dict):
        return None
    meta = msg.get("MetaData") or {}
    mtype = str(msg.get("MessageType") or "")
    inner = msg.get("Message") or {}
    body = inner.get(mtype) or inner.get("PositionReport") or {}
    lat = meta.get("latitude", meta.get("Latitude", body.get("Latitude")))
    lon = meta.get("longitude", meta.get("Longitude", body.get("Longitude")))
    if lat is None or lon is None:
        return None
    try:
        lat_f, lon_f = float(lat), float(lon)
    except (TypeError, ValueError):
        return None
    if not (-90 <= lat_f <= 90 and -180 <= lon_f <= 180):
        return None
    mmsi = str(meta.get("MMSI") or meta.get("MMSI_String") or body.get("UserID") or "").strip()
    if not mmsi:
        return None
    name = str(meta.get("ShipName") or meta.get("shipname") or "").strip() or f"MMSI {mmsi}"
    sog = body.get("Sog", body.get("SOG", 0)) or 0
    cog = body.get("Cog", body.get("TrueHeading", body.get("COG", 0))) or 0
    try:
        sog_f = float(sog)
    except (TypeError, ValueError):
        sog_f = 0.0
    try:
        cog_f = float(cog)
    except (TypeError, ValueError):
        cog_f = 0.0
    return {
        "id": f"mmsi-{mmsi}",
        "kind": "ship",
        "callsign": name,
        "mmsi": mmsi,
        "lat": lat_f,
        "lon": lon_f,
        "alt_m": 0.0,
        "heading": cog_f,
        "gs_kts": sog_f,
        "source": "aisstream",
    }


def normalize_aishub(payload: Any) -> list[dict[str, Any]]:
    rows: list[Any] = []
    if isinstance(payload, list):
        if len(payload) >= 2 and isinstance(payload[1], list):
            rows = payload[1]
        else:
            rows = [
                r
                for r in payload
                if isinstance(r, dict) and ("LATITUDE" in r or "LAT" in r)
            ]
    elif isinstance(payload, dict):
        if payload.get("ERROR"):
            return []
        rows = payload.get("DATA") or payload.get("vessels") or []
    out: list[dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        lat = r.get("LATITUDE", r.get("LAT", r.get("lat")))
        lon = r.get("LONGITUDE", r.get("LON", r.get("lon")))
        mmsi = r.get("MMSI", r.get("mmsi"))
        if lat is None or lon is None or not mmsi:
            continue
        try:
            lat_f, lon_f = float(lat), float(lon)
        except (TypeError, ValueError):
            continue
        name = str(r.get("NAME") or r.get("SHIPNAME") or r.get("name") or "").strip() or f"MMSI {mmsi}"
        try:
            sog = float(r.get("SOG") or r.get("SPEED") or 0)
        except (TypeError, ValueError):
            sog = 0.0
        try:
            hdg = float(r.get("HEADING") or r.get("COG") or 0)
        except (TypeError, ValueError):
            hdg = 0.0
        out.append(
            {
                "id": f"mmsi-{mmsi}",
                "kind": "ship",
                "callsign": name,
                "mmsi": str(mmsi),
                "lat": lat_f,
                "lon": lon_f,
                "alt_m": 0.0,
                "heading": hdg,
                "gs_kts": sog,
                "source": "aishub",
            }
        )
    return out


def normalize_marinetraffic(payload: Any) -> list[dict[str, Any]]:
    rows: list[Any]
    if isinstance(payload, dict):
        if payload.get("errors") or payload.get("ERROR"):
            return []
        rows = payload.get("DATA") or payload.get("vessels") or payload.get("data") or []
    elif isinstance(payload, list):
        rows = payload
    else:
        return []
    out: list[dict[str, Any]] = []
    for r in rows:
        if isinstance(r, list) and len(r) >= 5:
            mmsi, lat, lon, speed, heading = r[0], r[1], r[2], r[3], r[4]
            name = r[8] if len(r) > 8 else f"MMSI {mmsi}"
            rec = {"MMSI": mmsi, "LAT": lat, "LON": lon, "SPEED": speed, "HEADING": heading, "SHIPNAME": name}
        elif isinstance(r, dict):
            rec = r
        else:
            continue
        lat = rec.get("LAT", rec.get("LATITUDE"))
        lon = rec.get("LON", rec.get("LONGITUDE"))
        mmsi = rec.get("MMSI")
        if lat is None or lon is None or not mmsi:
            continue
        try:
            lat_f, lon_f = float(lat), float(lon)
            # MarineTraffic SPEED is often 0.1 kn units
            raw_spd = float(rec.get("SPEED") or rec.get("SOG") or 0)
            sog = raw_spd / 10.0 if raw_spd > 80 else raw_spd
            hdg = float(rec.get("HEADING") or rec.get("COURSE") or 0)
        except (TypeError, ValueError):
            continue
        name = str(rec.get("SHIPNAME") or rec.get("SHIP_NAME") or "").strip() or f"MMSI {mmsi}"
        out.append(
            {
                "id": f"mmsi-{mmsi}",
                "kind": "ship",
                "callsign": name,
                "mmsi": str(mmsi),
                "lat": lat_f,
                "lon": lon_f,
                "alt_m": 0.0,
                "heading": hdg,
                "gs_kts": sog,
                "source": "marinetraffic",
            }
        )
    return out


def clamp_bbox(
    min_lat: float, min_lon: float, max_lat: float, max_lon: float
) -> tuple[float, float, float, float]:
    """Clamp a viewport box to the globe and a sane max span."""
    min_lat = max(-90.0, min(90.0, float(min_lat)))
    max_lat = max(-90.0, min(90.0, float(max_lat)))
    min_lon = max(-180.0, min(180.0, float(min_lon)))
    max_lon = max(-180.0, min(180.0, float(max_lon)))
    if max_lat < min_lat:
        min_lat, max_lat = max_lat, min_lat
    if max_lon < min_lon:
        min_lon, max_lon = max_lon, min_lon
    if max_lat - min_lat > MAX_BBOX_SPAN_DEG:
        mid = (min_lat + max_lat) / 2.0
        half = MAX_BBOX_SPAN_DEG / 2.0
        min_lat, max_lat = max(-90.0, mid - half), min(90.0, mid + half)
    if max_lon - min_lon > MAX_BBOX_SPAN_DEG:
        mid = (min_lon + max_lon) / 2.0
        half = MAX_BBOX_SPAN_DEG / 2.0
        min_lon, max_lon = max(-180.0, mid - half), min(180.0, mid + half)
    return (min_lat, min_lon, max_lat, max_lon)


def parse_bbox(bbox: Optional[str]) -> Optional[tuple[float, float, float, float]]:
    if not bbox:
        return None
    parts = [p.strip() for p in str(bbox).split(",")]
    if len(parts) != 4:
        return None
    try:
        min_lat, min_lon, max_lat, max_lon = (float(p) for p in parts)
    except ValueError:
        return None
    return clamp_bbox(min_lat, min_lon, max_lat, max_lon)


def bbox_from_view(lat: float, lon: float, radius_nm: float) -> tuple[float, float, float, float]:
    box = _box(lat, lon, radius_nm)
    return clamp_bbox(box[0][0], box[0][1], box[1][0], box[1][1])


def normalize_openwaters(payload: Any) -> list[dict[str, Any]]:
    """Open Waters GeoJSON FeatureCollection → ship contacts. Pure — used by tests."""
    if isinstance(payload, dict):
        feats = payload.get("features") or []
    elif isinstance(payload, list):
        feats = payload
    else:
        return []
    out: list[dict[str, Any]] = []
    for feat in feats:
        if not isinstance(feat, dict):
            continue
        props = feat.get("properties") if isinstance(feat.get("properties"), dict) else {}
        geom = feat.get("geometry") if isinstance(feat.get("geometry"), dict) else {}
        coords = geom.get("coordinates") or []
        if len(coords) < 2:
            continue
        try:
            lon_f, lat_f = float(coords[0]), float(coords[1])
        except (TypeError, ValueError):
            continue
        if not (-90 <= lat_f <= 90 and -180 <= lon_f <= 180):
            continue
        mmsi = props.get("mmsi", feat.get("id"))
        if mmsi is None or mmsi == "":
            continue
        mmsi_s = str(mmsi).strip()
        name = str(props.get("name") or "").strip() or f"MMSI {mmsi_s}"
        try:
            sog = float(props["sog"]) if props.get("sog") is not None else 0.0
        except (TypeError, ValueError):
            sog = 0.0
        heading_raw = props.get("heading")
        if heading_raw is None:
            heading_raw = props.get("cog")
        try:
            hdg = float(heading_raw) if heading_raw is not None else 0.0
        except (TypeError, ValueError):
            hdg = 0.0
        try:
            cog = float(props["cog"]) if props.get("cog") is not None else hdg
        except (TypeError, ValueError):
            cog = hdg
        feed = str(props.get("source") or "").strip()
        out.append(
            {
                "id": f"mmsi-{mmsi_s}",
                "kind": "ship",
                "callsign": name,
                "mmsi": mmsi_s,
                "lat": lat_f,
                "lon": lon_f,
                "alt_m": 0.0,
                "heading": hdg,
                "gs_kts": sog,
                "sog": sog,
                "cog": cog,
                "source": "openwaters",
                "feed": feed,
                "station": str(props.get("station") or "").strip(),
                "msg_type": str(props.get("msg_type") or ""),
                "ais_kind": str(props.get("kind") or "vessel"),
                "seen": str(props.get("seen") or ""),
            }
        )
        if len(out) >= 400:
            break
    return out


def _dedupe_ships(rows: list[dict[str, Any]], cap: int = 400) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        i = row.get("id")
        if not i or i in seen:
            continue
        seen.add(i)
        out.append(row)
        if len(out) >= cap:
            break
    return out


def _box(lat: float, lon: float, radius_nm: float) -> list[list[float]]:
    dlat = max(0.4, radius_nm / 60.0)
    clat = max(0.2, math.cos(math.radians(lat)))
    dlon = max(0.4, radius_nm / (60.0 * clat))
    return [
        [max(-90.0, lat - dlat), max(-180.0, lon - dlon)],
        [min(90.0, lat + dlat), min(180.0, lon + dlon)],
    ]


def boxes_for_view(lat: float, lon: float, radius_nm: float) -> list[list[list[float]]]:
    if radius_nm >= 700:
        return [_box(hlat, hlon, hr) for hlat, hlon, hr in _HUBS]
    return [_box(lat, lon, radius_nm)]


def _remember(contact: dict[str, Any]) -> None:
    contact["_ts"] = _now_ms()
    _CACHE[contact["id"]] = contact
    if len(_CACHE) > 800:
        oldest = sorted(_CACHE.values(), key=lambda c: c.get("_ts") or 0)
        for row in oldest[: len(_CACHE) - 600]:
            _CACHE.pop(row["id"], None)


def snapshot(lat: float, lon: float, radius_nm: float, source: str) -> list[dict[str, Any]]:
    now = _now_ms()
    dlat = max(0.4, radius_nm / 60.0)
    clat = max(0.2, math.cos(math.radians(lat)))
    dlon = max(0.4, radius_nm / (60.0 * clat))
    out: list[dict[str, Any]] = []
    for c in list(_CACHE.values()):
        if now - (c.get("_ts") or 0) > 180_000:
            _CACHE.pop(c["id"], None)
            continue
        if radius_nm < 700:
            if abs(c["lat"] - lat) > dlat or abs(((c["lon"] - lon + 180) % 360) - 180) > dlon:
                continue
        row = {k: v for k, v in c.items() if k != "_ts"}
        row["source"] = source
        out.append(row)
        if len(out) >= 400:
            break
    return out


def set_view(lat: float, lon: float, radius_nm: float) -> None:
    global _BOXES
    _BOXES = boxes_for_view(lat, lon, radius_nm)


async def _aisstream_loop() -> None:
    global _STREAM_OK, _STREAM_ERR, _LAST_MSG_MS
    key = _credentials()["aisstream"]
    if not key:
        return
    try:
        import websockets
    except ImportError:
        _STREAM_ERR = "websockets package missing"
        return

    backoff = 2.0
    while not _STOP.is_set():
        try:
            boxes = _BOXES or boxes_for_view(35.0, -30.0, 900)
            async with websockets.connect(
                AISSTREAM_URL,
                max_size=2**22,
                ping_interval=20,
                ping_timeout=20,
                compression="deflate",
            ) as ws:
                await ws.send(
                    json.dumps(
                        {
                            "APIKey": key,
                            "BoundingBoxes": boxes,
                            "FilterMessageTypes": ["PositionReport"],
                        }
                    )
                )
                _STREAM_OK = True
                _STREAM_ERR = ""
                backoff = 2.0
                log.info("AISStream connected boxes=%s", len(boxes))
                async for raw in ws:
                    if _STOP.is_set():
                        break
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        continue
                    if msg.get("error") or msg.get("Error"):
                        _STREAM_ERR = str(msg.get("error") or msg.get("Error"))
                        continue
                    contact = normalize_aisstream(msg)
                    if not contact:
                        continue
                    _remember(contact)
                    _LAST_MSG_MS = _now_ms()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _STREAM_OK = False
            _STREAM_ERR = str(e)
            log.warning("AISStream reconnect in %.0fs: %s", backoff, e)
            try:
                await asyncio.wait_for(_STOP.wait(), timeout=backoff)
            except asyncio.TimeoutError:
                pass
            backoff = min(60.0, backoff * 1.7)


def ensure_stream() -> None:
    global _TASK
    if keyed_ships_provider() != "aisstream":
        return
    if _TASK and not _TASK.done():
        return
    _STOP.clear()
    _TASK = asyncio.create_task(_aisstream_loop(), name="godseye-aisstream")


async def stop_stream() -> None:
    _STOP.set()
    if _TASK:
        _TASK.cancel()
        try:
            await _TASK
        except (asyncio.CancelledError, Exception):
            pass


async def fetch_aishub(lat: float, lon: float, radius_nm: float) -> list[dict[str, Any]]:
    user = _credentials()["aishub"]
    if not user:
        return []
    box = _box(lat, lon, radius_nm)
    params = {
        "username": user,
        "format": "1",
        "output": "json",
        "compress": "0",
        "latmin": f"{box[0][0]:.3f}",
        "lonmin": f"{box[0][1]:.3f}",
        "latmax": f"{box[1][0]:.3f}",
        "lonmax": f"{box[1][1]:.3f}",
    }
    async with httpx.AsyncClient(follow_redirects=True, timeout=14.0) as client:
        r = await client.get(AISHUB_URL, params=params, headers={"User-Agent": UA})
        r.raise_for_status()
        data = r.json()
    contacts = normalize_aishub(data)
    for c in contacts:
        _remember(c)
    return contacts


async def fetch_marinetraffic() -> list[dict[str, Any]]:
    key = _credentials()["marinetraffic"]
    if not key:
        return []
    url = MARINETRAFFIC_URL.format(key=key)
    async with httpx.AsyncClient(follow_redirects=True, timeout=16.0) as client:
        r = await client.get(url, headers={"User-Agent": UA})
        r.raise_for_status()
        data = r.json()
    contacts = normalize_marinetraffic(data)
    for c in contacts:
        _remember(c)
    return contacts


async def fetch_openwaters(
    bbox: tuple[float, float, float, float],
    client: Optional[httpx.AsyncClient] = None,
) -> list[dict[str, Any]]:
    min_lat, min_lon, max_lat, max_lon = bbox
    params = {"bbox": f"{min_lat:.4f},{min_lon:.4f},{max_lat:.4f},{max_lon:.4f}"}
    own = client is None
    client = client or httpx.AsyncClient(follow_redirects=True)
    try:
        r = await client.get(
            OPENWATERS_VESSELS,
            params=params,
            headers={"User-Agent": UA, "Accept": "application/json"},
            timeout=12.0,
        )
        r.raise_for_status()
        return normalize_openwaters(r.json())
    finally:
        if own:
            await client.aclose()


async def fetch_ships(
    lat: float = 35.0,
    lon: float = -30.0,
    radius_nm: float = 450,
    bbox: Optional[str] = None,
) -> dict[str, Any]:
    lat = max(-89.9, min(89.9, float(lat)))
    lon = max(-179.9, min(179.9, float(lon)))
    radius_nm = max(25.0, min(1500.0, float(radius_nm)))
    view = parse_bbox(bbox) or bbox_from_view(lat, lon, radius_nm)
    set_view((view[0] + view[2]) / 2.0, (view[1] + view[3]) / 2.0, radius_nm)

    provider = ships_provider()
    keyed = keyed_ships_provider()
    errors: list[str] = []
    ow_contacts: list[dict[str, Any]] = []
    keyed_contacts: list[dict[str, Any]] = []
    ow_ok = False

    try:
        ow_contacts = await fetch_openwaters(view)
        ow_ok = True
    except Exception as e:
        errors.append(f"openwaters {e}")

    if keyed == "aisstream":
        ensure_stream()
        keyed_contacts = snapshot(lat, lon, radius_nm, "aisstream")
        if _STREAM_ERR:
            errors.append(_STREAM_ERR)
    elif keyed == "aishub":
        try:
            keyed_contacts = await fetch_aishub(lat, lon, radius_nm)
        except Exception as e:
            errors.append(str(e))
            keyed_contacts = snapshot(lat, lon, radius_nm, "aishub")
    elif keyed == "marinetraffic":
        try:
            keyed_contacts = await fetch_marinetraffic()
            keyed_contacts = snapshot(lat, lon, radius_nm, "marinetraffic") or keyed_contacts
        except Exception as e:
            errors.append(str(e))
            keyed_contacts = snapshot(lat, lon, radius_nm, "marinetraffic")

    # Open Waters is primary; keyed AIS fills gaps / adds extra MMSIs.
    contacts = _dedupe_ships(ow_contacts + keyed_contacts)
    connecting = keyed == "aisstream" and not keyed_contacts and bool(_STREAM_OK or _TASK)

    if contacts:
        label, status = "LIVE", "live"
    elif connecting and not ow_contacts:
        label, status = "CONNECTING", "live"
    elif ow_ok:
        label, status = "EMPTY", "empty"
    else:
        label, status = "UNAVAILABLE", "unavailable"

    if keyed:
        note = (
            "Keyless Open Waters AIS snapshot, plus optional "
            f"{keyed}. Honest empty only when both sources have no contacts."
        )
    else:
        note = (
            "Keyless Open Waters AIS snapshot (ais.openwaters.io). "
            "AISSTREAM_API_KEY stays optional."
        )

    return {
        "kind": "ships",
        "status": status,
        "source": provider,
        "provider": provider,
        "shipsProvider": provider,
        "bbox": [view[0], view[1], view[2], view[3]],
        "freshness_ms": _LAST_MSG_MS or (_now_ms() if contacts else None),
        "label": label,
        "simulated": False,
        "count": len(contacts),
        "contacts": contacts,
        "errors": errors[:4],
        "note": note,
    }
