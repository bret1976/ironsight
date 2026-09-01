"""GodsEye public spatial intel — server-side fetch + normalize.

Proxies keyless public feeds so the browser is not blocked by CORS, and
stamps source + freshness on every payload. No commercial cable/AIS dumps.
"""

from __future__ import annotations

import asyncio
import math
import time
from typing import Any, Optional
from urllib.parse import urlparse

import httpx

UA = "IronSightGodsEye/1.0 (+https://github.com/VegasCryptoAgent/ironsight)"
ADSB_BASE = "https://api.adsb.lol"
OPENSKY_BASE = "https://opensky-network.org/api/states/all"
CELESTRAK_GP = "https://celestrak.org/NORAD/elements/gp.php"
CELESTRAK_ISS = "https://celestrak.org/NORAD/elements/gp.php?CATNR=25544&FORMAT=tle"
CELESTRAK_ISS_OMM = "https://celestrak.org/NORAD/elements/gp.php?CATNR=25544&FORMAT=json"
ARISS_ISS = "https://live.ariss.org/iss.txt"
USGS_QUAKES = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_day.geojson"
NOMINATIM = "https://nominatim.openstreetmap.org/search"
FIRMS_CSV = (
    "https://firms.modaps.eosdis.nasa.gov/data/active_fire/"
    "noaa-20-viirs-c2/csv/J1_VIIRS_C2_Global_24h.csv"
)
EONET_EVENTS = "https://eonet.gsfc.nasa.gov/api/v3/events"
NWS_ALERTS = "https://api.weather.gov/alerts/active"
EONET_KIND = {
    "severestorms": "storm",
    "volcanoes": "volcano",
    "volcano": "volcano",
    "floods": "flood",
    "flood": "flood",
    "dusthaze": "dust",
    "dustandhaze": "dust",
}
EONET_SKIP = frozenset({"wildfires", "wildfire"})
NWS_KEEP = frozenset(
    {
        "tornado warning",
        "tornado watch",
        "severe thunderstorm warning",
        "hurricane warning",
        "hurricane watch",
        "tropical storm warning",
        "tropical storm watch",
        "flood warning",
        "flash flood warning",
        "winter storm warning",
        "blizzard warning",
        "storm warning",
        "special marine warning",
    }
)
STORMS_TTL_S = 4 * 60
OVERPASS = "https://overpass-api.de/api/interpreter"
OVERPASS_MIRRORS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
)
RADIO_BROWSER = "https://de1.api.radio-browser.info/json/stations/search"
RADIO_MIRRORS = (
    "https://de1.api.radio-browser.info/json/stations/search",
    "https://nl1.api.radio-browser.info/json/stations/search",
    "https://at1.api.radio-browser.info/json/stations/search",
)
GOOGLE_STATIC_MAP = "https://maps.googleapis.com/maps/api/staticmap"
ESRI_IMAGERY = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
LAUNCH_LIBRARY = "https://ll.thespacedevs.com/2.2.0/launch"
TFL_JAMCAM = "https://api.tfl.gov.uk/Place/Type/JamCam"
AUSTIN_CAMERAS = "https://data.austintexas.gov/resource/b4k4-adkb.json"
CALTRANS_CCTV = (
    "https://cwwp2.dot.ca.gov/data/d7/cctv/cctvStatusD07.json",
    "https://cwwp2.dot.ca.gov/data/d4/cctv/cctvStatusD04.json",
    "https://cwwp2.dot.ca.gov/data/d12/cctv/cctvStatusD12.json",
)
# Same-origin still proxy. HLS and random hosts stay out.
CCTV_MEDIA_HOSTS = {
    "s3-eu-west-1.amazonaws.com",
    "jamcams.tfl.gov.uk",
    "cctv.austinmobility.io",
    "cwwp2.dot.ca.gov",
}
BROWSER_VIDEO_SUFFIXES = (".mp4", ".webm", ".ogg")
# Austin lists TURNED_ON cameras whose still is a 12 KB "no signal" JPEG.
AUSTIN_DEAD_STILL_BYTES = 12805
AUSTIN_DEAD_STILL_ETAG = "cff5cb73f8b8a9faf64ea9d2cc5be8e9"

# In-memory TTL cache (process-local). Fine for a single Railway worker.
_CACHE: dict[str, tuple[float, Any]] = {}

# Named CelesTrak GP groups only. Never GROUP=active — that catalog 403s.
TLE_GROUPS = ("stations", "visual", "weather")
# Classic TLE catalog numbers are 5 digits; SATCAT crossed 100000 on 2026-07-11.
TLE_NORAD_MAX = 99999
OMM_KEYS = (
    "OBJECT_NAME",
    "OBJECT_ID",
    "EPOCH",
    "MEAN_MOTION",
    "ECCENTRICITY",
    "INCLINATION",
    "RA_OF_ASC_NODE",
    "ARG_OF_PERICENTER",
    "MEAN_ANOMALY",
    "EPHEMERIS_TYPE",
    "CLASSIFICATION_TYPE",
    "NORAD_CAT_ID",
    "ELEMENT_SET_NO",
    "REV_AT_EPOCH",
    "BSTAR",
    "MEAN_MOTION_DOT",
    "MEAN_MOTION_DDOT",
)
OMM_REQUIRED = (
    "EPOCH",
    "MEAN_MOTION",
    "ECCENTRICITY",
    "INCLINATION",
    "RA_OF_ASC_NODE",
    "ARG_OF_PERICENTER",
    "MEAN_ANOMALY",
    "NORAD_CAT_ID",
    "BSTAR",
)


def cache_get(key: str) -> Any | None:
    val, fresh = cache_peek(key)
    return val if fresh else None


def cache_peek(key: str) -> tuple[Any | None, bool]:
    """Return (value, fresh). Expired rows stay readable so CCTV does not go to 0 on redeploy."""
    row = _CACHE.get(key)
    if not row:
        return None, False
    exp, val = row
    return val, exp >= time.time()


def cache_set(key: str, val: Any, ttl: float) -> None:
    _CACHE[key] = (time.time() + ttl, val)


def _headers() -> dict[str, str]:
    return {"User-Agent": UA, "Accept": "application/json, text/plain, */*"}


def _nws_headers() -> dict[str, str]:
    return {"User-Agent": UA, "Accept": "application/geo+json"}


async def _get_text(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout: float = 12.0,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
) -> tuple[int, str]:
    r = await client.get(url, params=params, headers=headers or _headers(), timeout=timeout)
    return r.status_code, r.text


def _now_ms() -> int:
    return int(time.time() * 1000)


def _ft_to_m(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, str):
        if v.lower() in ("ground", "none", ""):
            return 0.0
        try:
            v = float(v)
        except ValueError:
            return None
    try:
        return float(v) * 0.3048
    except (TypeError, ValueError):
        return None


def normalize_adsb(payload: dict, *, fetched_at: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ac in payload.get("ac") or []:
        lat, lon = ac.get("lat"), ac.get("lon")
        if lat is None or lon is None:
            continue
        alt = ac.get("alt_geom")
        alt_m = _ft_to_m(alt if alt not in (None, "ground") else ac.get("alt_baro"))
        if alt_m is None:
            alt_m = 0.0
        hex_id = str(ac.get("hex") or "").strip().lower()
        if not hex_id:
            continue
        flight = str(ac.get("flight") or "").strip() or hex_id.upper()
        out.append(
            {
                "id": f"icao-{hex_id}",
                "kind": "flight",
                "callsign": flight,
                "hex": hex_id,
                "lat": float(lat),
                "lon": float(lon),
                "alt_m": float(alt_m),
                "heading": float(ac.get("track") or 0),
                "gs_kts": float(ac.get("gs") or 0),
                "type": str(ac.get("t") or ac.get("type") or ""),
                "reg": str(ac.get("r") or ""),
                "on_ground": str(ac.get("alt_baro") or "").lower() == "ground",
                "military": bool(ac.get("dbFlags") in (1, 2, 3) or ac.get("mil")),
                "source": "adsb.lol",
            }
        )
    return out


def normalize_opensky(payload: dict, *, fetched_at: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for st in payload.get("states") or []:
        if not st or len(st) < 8:
            continue
        icao, callsign, _cc, _tp, _lc, lon, lat, baro = st[:8]
        if lat is None or lon is None:
            continue
        heading = st[10] if len(st) > 10 else 0
        vel = st[9] if len(st) > 9 else 0
        on_gnd = bool(st[8]) if len(st) > 8 else False
        hex_id = str(icao or "").strip().lower()
        if not hex_id:
            continue
        out.append(
            {
                "id": f"icao-{hex_id}",
                "kind": "flight",
                "callsign": (str(callsign or "").strip() or hex_id.upper()),
                "hex": hex_id,
                "lat": float(lat),
                "lon": float(lon),
                "alt_m": float(baro or 0),
                "heading": float(heading or 0),
                "gs_kts": float(vel or 0) * 1.94384,
                "type": "",
                "reg": "",
                "on_ground": on_gnd,
                "military": False,
                "source": "opensky-anonymous",
            }
        )
    return out


def parse_tle_text(text: str) -> list[dict[str, str]]:
    """Split CelesTrak 3-line (or 2-line) TLE blocks. Pure — used by tests."""
    lines = [ln.rstrip() for ln in (text or "").splitlines() if ln.strip()]
    recs: list[dict[str, str]] = []
    i = 0
    while i < len(lines):
        a = lines[i].strip()
        b = lines[i + 1].strip() if i + 1 < len(lines) else ""
        c = lines[i + 2].strip() if i + 2 < len(lines) else ""
        if a.startswith("1 ") and b.startswith("2 "):
            recs.append({"name": f"SAT-{len(recs)+1}", "l1": a, "l2": b})
            i += 2
            continue
        if b.startswith("1 ") and c.startswith("2 "):
            recs.append({"name": a, "l1": b, "l2": c})
            i += 3
            continue
        i += 1
    return recs


def norad_from_tle_line1(l1: str) -> str:
    """Catalog number from a classic TLE line 1 (5 digits historically)."""
    a = (l1 or "").strip()
    if len(a) >= 7 and a.startswith("1"):
        token = a[2:8].strip().rstrip("UABC")
        digits = "".join(ch for ch in token if ch.isdigit())
        if digits:
            return str(int(digits))
    return ""


def omm_complete(obj: dict[str, Any]) -> bool:
    if not isinstance(obj, dict):
        return False
    for key in OMM_REQUIRED:
        val = obj.get(key)
        if val is None or val == "":
            return False
    return True


def parse_omm_json(payload: Any) -> list[dict[str, Any]]:
    """Normalize CelesTrak GP JSON (OMM) into sat records the UI can consume."""
    rows: list[Any]
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("omm") or payload.get("data") or payload.get("records") or [payload]
    else:
        return []
    out: list[dict[str, Any]] = []
    for raw in rows:
        rec = normalize_omm(raw)
        if rec:
            out.append(rec)
    return out


def normalize_omm(obj: Any) -> dict[str, Any] | None:
    """Map one CelesTrak OMM object. Pure — used by tests."""
    if not isinstance(obj, dict):
        return None
    norad_raw = obj.get("NORAD_CAT_ID", obj.get("norad"))
    try:
        norad_i = int(norad_raw)
    except (TypeError, ValueError):
        return None
    if norad_i <= 0:
        return None
    name = str(obj.get("OBJECT_NAME") or obj.get("name") or f"SAT-{norad_i}").strip()
    rec: dict[str, Any] = {
        "name": name,
        "norad": str(norad_i),
        "format": "omm" if omm_complete(obj) else "incomplete",
        "l1": str(obj.get("l1") or ""),
        "l2": str(obj.get("l2") or ""),
    }
    for key in OMM_KEYS:
        if key in obj and obj[key] is not None:
            rec[key] = obj[key]
    rec["OBJECT_NAME"] = rec.get("OBJECT_NAME") or name
    rec["NORAD_CAT_ID"] = norad_i
    return rec


def attach_tle_fallback(recs: list[dict[str, Any]], tle_recs: list[dict[str, str]]) -> None:
    """Classic TLE only for NORAD < 100000 when OMM fields are missing."""
    by_norad: dict[str, dict[str, str]] = {}
    for tle in tle_recs:
        nid = norad_from_tle_line1(tle.get("l1") or "")
        if nid:
            by_norad[nid] = tle
    for rec in recs:
        if omm_complete(rec) and rec.get("l1") and rec.get("l2"):
            continue
        try:
            norad_i = int(rec.get("norad") or rec.get("NORAD_CAT_ID") or 0)
        except (TypeError, ValueError):
            norad_i = 0
        if norad_i >= 100000:
            continue
        tle = by_norad.get(str(norad_i))
        if not tle:
            continue
        rec["l1"] = tle["l1"]
        rec["l2"] = tle["l2"]
        if not rec.get("name") or rec["name"].startswith("SAT-"):
            rec["name"] = tle["name"]
        if not omm_complete(rec):
            rec["format"] = "tle"


def _dedupe_sats(recs: list[dict[str, Any]], cap: int = 90) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for rec in recs:
        key = str(rec.get("norad") or rec.get("NORAD_CAT_ID") or rec.get("name") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(rec)
    unique.sort(
        key=lambda r: (
            0 if "ZARYA" in str(r.get("name") or "").upper() else (1 if "ISS" in str(r.get("name") or "").upper() else 2),
            str(r.get("name") or ""),
        )
    )
    return unique[:cap]


def normalize_usgs(payload: dict, *, fetched_at: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for feat in payload.get("features") or []:
        props = feat.get("properties") or {}
        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates") or []
        if len(coords) < 2:
            continue
        lon, lat = float(coords[0]), float(coords[1])
        depth_km = float(coords[2]) if len(coords) > 2 and coords[2] is not None else 0.0
        eid = str(feat.get("id") or props.get("code") or f"eq-{lat:.3f}-{lon:.3f}")
        mag = props.get("mag")
        try:
            mag_f = float(mag) if mag is not None else 0.0
        except (TypeError, ValueError):
            mag_f = 0.0
        out.append(
            {
                "id": f"eq-{eid}",
                "kind": "quake",
                "callsign": props.get("place") or eid,
                "lat": lat,
                "lon": lon,
                "alt_m": -depth_km * 1000.0,
                "heading": 0,
                "mag": mag_f,
                "time_ms": props.get("time"),
                "url": props.get("url") or "",
                "source": "usgs",
            }
        )
    return out


def _dedupe_flights(rows: list[dict[str, Any]], cap: int = 400) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for r in rows:
        i = r.get("id")
        if not i or i in seen:
            continue
        seen.add(i)
        out.append(r)
        if len(out) >= cap:
            break
    return out


async def fetch_flights(
    lat: float,
    lon: float,
    radius_nm: float = 400,
    include_mil: bool = True,
    client: Optional[httpx.AsyncClient] = None,
) -> dict[str, Any]:
    """Keyless flights: adsb.lol first, OpenSky anonymous bbox fallback."""
    lat = max(-89.9, min(89.9, float(lat)))
    lon = max(-179.9, min(179.9, float(lon)))
    radius_nm = max(25, min(1500, float(radius_nm)))
    key = f"flights:{lat:.2f}:{lon:.2f}:{int(radius_nm)}:{int(include_mil)}"
    hit = cache_get(key)
    if hit:
        return {**hit, "cached": True}

    own = client is None
    client = client or httpx.AsyncClient(follow_redirects=True)
    fetched_at = _now_ms()
    contacts: list[dict[str, Any]] = []
    source = "adsb.lol"
    errors: list[str] = []
    try:
        try:
            url = f"{ADSB_BASE}/v2/lat/{lat:.3f}/lon/{lon:.3f}/dist/{int(radius_nm)}"
            code, text = await _get_text(client, url, timeout=12.0)
            if code == 200 and text.lstrip().startswith("{"):
                import json

                contacts.extend(normalize_adsb(json.loads(text), fetched_at=fetched_at))
            else:
                errors.append(f"adsb.lol HTTP {code}")
        except Exception as e:
            errors.append(f"adsb.lol {e}")

        if include_mil:
            try:
                code, text = await _get_text(client, f"{ADSB_BASE}/v2/mil", timeout=12.0)
                if code == 200 and text.lstrip().startswith("{"):
                    import json

                    mil = normalize_adsb(json.loads(text), fetched_at=fetched_at)
                    for row in mil:
                        row["military"] = True
                    contacts.extend(mil)
            except Exception as e:
                errors.append(f"adsb.lol/mil {e}")

        if len(contacts) < 8:
            # OpenSky anonymous bbox — no account. Often slow / rate-limited.
            dlat = max(1.5, radius_nm / 60.0)
            dlon = max(1.5, radius_nm / (60.0 * max(0.2, math.cos(math.radians(lat)))))
            params = {
                "lamin": f"{lat - dlat:.3f}",
                "lamax": f"{lat + dlat:.3f}",
                "lomin": f"{lon - dlon:.3f}",
                "lomax": f"{lon + dlon:.3f}",
            }
            try:
                code, text = await _get_text(
                    client, OPENSKY_BASE, timeout=14.0, params=params
                )
                if code == 200 and text.lstrip().startswith("{"):
                    import json

                    extra = normalize_opensky(json.loads(text), fetched_at=fetched_at)
                    contacts.extend(extra)
                    if extra and not contacts:
                        source = "opensky-anonymous"
                    elif extra:
                        source = "adsb.lol+opensky-anonymous"
                else:
                    errors.append(f"opensky HTTP {code}")
            except Exception as e:
                errors.append(f"opensky {e}")

        civ = [c for c in contacts if not c.get("military")]
        mil = [c for c in contacts if c.get("military")]
        contacts = _dedupe_flights(civ, cap=320) + _dedupe_flights(mil, cap=120)
        status = "live" if contacts else "unavailable"
        payload = {
            "kind": "flights",
            "status": status,
            "source": source if contacts else "adsb.lol|opensky-anonymous",
            "freshness_ms": fetched_at,
            "label": "LIVE" if contacts else "UNAVAILABLE",
            "count": len(contacts),
            "contacts": contacts,
            "errors": errors[:6],
            "note": "Public ADS-B. Interpolated on the client between polls. No OpenSky account.",
        }
        cache_set(key, payload, ttl=12.0)
        return {**payload, "cached": False}
    finally:
        if own:
            await client.aclose()


async def fetch_tles(
    groups: Optional[list[str]] = None,
    client: Optional[httpx.AsyncClient] = None,
) -> dict[str, Any]:
    """CelesTrak GP OMM JSON, with classic TLE fallback for NORAD < 100000."""
    groups = [g for g in (groups or list(TLE_GROUPS)) if g in TLE_GROUPS and g != "active"]
    if not groups:
        groups = ["stations"]
    key = "omm:" + ",".join(groups)
    hit = cache_get(key)
    if hit:
        return {**hit, "cached": True}

    own = client is None
    client = client or httpx.AsyncClient(follow_redirects=True)
    fetched_at = _now_ms()
    recs: list[dict[str, Any]] = []
    errors: list[str] = []
    source = "celestrak"
    try:
        import json

        need_tle: set[str] = set()
        for g in groups:
            try:
                code, text = await _get_text(
                    client,
                    CELESTRAK_GP,
                    timeout=12.0,
                    params={"GROUP": g, "FORMAT": "json"},
                )
                if code == 200 and text.lstrip().startswith(("[", "{")):
                    parsed = parse_omm_json(json.loads(text))
                    recs.extend(parsed)
                    if any(not omm_complete(r) for r in parsed):
                        need_tle.add(g)
                    if not parsed:
                        need_tle.add(g)
                else:
                    errors.append(f"celestrak {g} HTTP {code}")
                    need_tle.add(g)
            except Exception as e:
                errors.append(f"celestrak {g} {e}")
                need_tle.add(g)

        if need_tle:
            tle_recs: list[dict[str, str]] = []
            for g in groups:
                if g not in need_tle:
                    continue
                try:
                    code, text = await _get_text(
                        client,
                        CELESTRAK_GP,
                        timeout=12.0,
                        params={"GROUP": g, "FORMAT": "tle"},
                    )
                    if code == 200 and "1 " in text and "2 " in text:
                        tle_recs.extend(parse_tle_text(text))
                    else:
                        errors.append(f"celestrak {g} tle HTTP {code}")
                except Exception as e:
                    errors.append(f"celestrak {g} tle {e}")
            if tle_recs:
                by_norad = {r.get("norad") for r in recs}
                for tle in tle_recs:
                    nid = norad_from_tle_line1(tle.get("l1") or "")
                    if not nid or nid in by_norad:
                        continue
                    try:
                        if int(nid) >= 100000:
                            continue
                    except ValueError:
                        continue
                    recs.append(
                        {
                            "name": tle["name"],
                            "norad": nid,
                            "format": "tle",
                            "l1": tle["l1"],
                            "l2": tle["l2"],
                            "NORAD_CAT_ID": int(nid),
                            "OBJECT_NAME": tle["name"],
                        }
                    )
                    by_norad.add(nid)
                attach_tle_fallback(recs, tle_recs)

        has_iss = any(
            str(r.get("norad")) == "25544" or "ISS" in str(r.get("name") or "").upper()
            for r in recs
        )
        if not has_iss:
            try:
                code, text = await _get_text(client, CELESTRAK_ISS_OMM, timeout=10.0)
                if code == 200 and text.lstrip().startswith(("[", "{")):
                    recs = parse_omm_json(json.loads(text)) + recs
                    has_iss = True
            except Exception as e:
                errors.append(f"iss omm {e}")
        if not has_iss:
            for url in (CELESTRAK_ISS, ARISS_ISS):
                try:
                    code, text = await _get_text(client, url, timeout=10.0)
                    if code == 200 and "1 " in text:
                        for tle in parse_tle_text(text):
                            nid = norad_from_tle_line1(tle.get("l1") or "") or "25544"
                            recs.insert(
                                0,
                                {
                                    "name": tle["name"],
                                    "norad": nid,
                                    "format": "tle",
                                    "l1": tle["l1"],
                                    "l2": tle["l2"],
                                    "NORAD_CAT_ID": int(nid) if nid.isdigit() else 25544,
                                    "OBJECT_NAME": tle["name"],
                                },
                            )
                        if "ariss" in url:
                            source = "celestrak+ariss"
                        break
                except Exception as e:
                    errors.append(f"iss {e}")

        unique = _dedupe_sats(recs)
        payload_format = "omm" if any(omm_complete(r) for r in unique) else (
            "tle" if unique else "omm"
        )

        payload = {
            "kind": "satellites",
            "status": "live" if unique else "unavailable",
            "source": source,
            "provider": "celestrak",
            "format": payload_format,
            "freshness_ms": fetched_at,
            "label": "LIVE" if unique else "UNAVAILABLE",
            "count": len(unique),
            "tles": unique,
            "omm": unique,
            "errors": errors[:6],
            "note": "CelesTrak GP OMM JSON. Positions via satellite.js json2satrec. Classic TLE fallback only for NORAD < 100000 when OMM is incomplete.",
        }
        cache_set(key, payload, ttl=30 * 60)
        return {**payload, "cached": False}
    finally:
        if own:
            await client.aclose()


async def fetch_quakes(client: Optional[httpx.AsyncClient] = None) -> dict[str, Any]:
    hit = cache_get("quakes")
    if hit:
        return {**hit, "cached": True}
    own = client is None
    client = client or httpx.AsyncClient(follow_redirects=True)
    fetched_at = _now_ms()
    try:
        try:
            code, text = await _get_text(client, USGS_QUAKES, timeout=12.0)
            if code != 200:
                raise RuntimeError(f"HTTP {code}")
            import json

            payload_in = json.loads(text)
            contacts = normalize_usgs(payload_in, fetched_at=fetched_at)
            payload = {
                "kind": "quakes",
                "status": "live",
                "source": "usgs",
                "freshness_ms": int(
                    (payload_in.get("metadata") or {}).get("generated") or fetched_at
                ),
                "label": "LIVE",
                "count": len(contacts),
                "contacts": contacts,
                "errors": [],
                "note": "USGS M2.5+ earthquakes, past 24h.",
            }
        except Exception as e:
            payload = {
                "kind": "quakes",
                "status": "unavailable",
                "source": "usgs",
                "freshness_ms": fetched_at,
                "label": "UNAVAILABLE",
                "count": 0,
                "contacts": [],
                "errors": [str(e)],
                "note": "USGS feed unreachable.",
            }
        cache_set("quakes", payload, ttl=5 * 60)
        return {**payload, "cached": False}
    finally:
        if own:
            await client.aclose()


def ships_stub() -> dict[str, Any]:
    from app.services.godseye_ais import ships_stub as _stub

    return _stub()


async def geocode(q: str, client: Optional[httpx.AsyncClient] = None) -> dict[str, Any]:
    q = (q or "").strip()
    if len(q) < 2:
        return {"results": [], "source": "nominatim"}
    key = f"geo:{q.lower()}"
    hit = cache_get(key)
    if hit:
        return hit
    own = client is None
    client = client or httpx.AsyncClient(follow_redirects=True)
    try:
        code, text = await _get_text(
            client,
            NOMINATIM,
            timeout=10.0,
            params={"q": q, "format": "json", "limit": "5", "polygon_geojson": "1"},
        )
        import json

        rows = json.loads(text) if code == 200 else []
        results = [
            {
                "label": r.get("display_name"),
                "lat": float(r["lat"]),
                "lon": float(r["lon"]),
                "geojson": r.get("geojson"),
                "kind": (r.get("type") or r.get("class") or "place"),
            }
            for r in rows
            if r.get("lat") and r.get("lon")
        ]
        payload = {"results": results, "source": "nominatim", "status": "ok" if results else "empty"}
        cache_set(key, payload, ttl=3600)
        return payload
    except Exception as e:
        return {"results": [], "source": "nominatim", "status": "unavailable", "errors": [str(e)]}
    finally:
        if own:
            await client.aclose()


def parse_firms_csv(text: str, cap: int = 700) -> list[dict[str, Any]]:
    """Parse NASA FIRMS VIIRS CSV. Pure — used by tests."""
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    if len(lines) < 2:
        return []
    header = [h.strip().lower() for h in lines[0].split(",")]
    idx = {name: i for i, name in enumerate(header)}
    lat_i = idx.get("latitude")
    lon_i = idx.get("longitude")
    if lat_i is None or lon_i is None:
        return []
    frp_i = idx.get("frp")
    conf_i = idx.get("confidence")
    date_i = idx.get("acq_date")
    time_i = idx.get("acq_time")
    rows: list[dict[str, Any]] = []
    for ln in lines[1:]:
        cols = ln.split(",")
        if max(lat_i, lon_i) >= len(cols):
            continue
        try:
            lat = float(cols[lat_i])
            lon = float(cols[lon_i])
        except ValueError:
            continue
        frp = 0.0
        if frp_i is not None and frp_i < len(cols):
            try:
                frp = float(cols[frp_i] or 0)
            except ValueError:
                frp = 0.0
        conf = cols[conf_i].strip() if conf_i is not None and conf_i < len(cols) else ""
        date = cols[date_i].strip() if date_i is not None and date_i < len(cols) else ""
        tod = cols[time_i].strip() if time_i is not None and time_i < len(cols) else ""
        rows.append(
            {
                "id": f"fire-{lat:.3f}-{lon:.3f}-{date}-{tod}",
                "kind": "fire",
                "callsign": f"FRP {frp:.0f}" if frp else "FIRE",
                "lat": lat,
                "lon": lon,
                "alt_m": 0.0,
                "heading": 0,
                "frp": frp,
                "confidence": conf,
                "source": "nasa-firms",
            }
        )
    rows.sort(key=lambda r: float(r.get("frp") or 0), reverse=True)
    return rows[:cap]


def _overpass_ql(lat: float, lon: float, radius_m: int) -> str:
    return f"""[out:json][timeout:22];
(
  node["landuse"="military"](around:{radius_m},{lat:.4f},{lon:.4f});
  way["landuse"="military"](around:{radius_m},{lat:.4f},{lon:.4f});
  node["military"](around:{radius_m},{lat:.4f},{lon:.4f});
  way["military"](around:{radius_m},{lat:.4f},{lon:.4f});
  node["waterway"="dam"](around:{radius_m},{lat:.4f},{lon:.4f});
  way["waterway"="dam"](around:{radius_m},{lat:.4f},{lon:.4f});
  node["man_made"="dam"](around:{radius_m},{lat:.4f},{lon:.4f});
  way["man_made"="dam"](around:{radius_m},{lat:.4f},{lon:.4f});
  node["telecom"="data_center"](around:{radius_m},{lat:.4f},{lon:.4f});
  way["telecom"="data_center"](around:{radius_m},{lat:.4f},{lon:.4f});
  node["building"="data_centre"](around:{radius_m},{lat:.4f},{lon:.4f});
  node["aeroway"="aerodrome"]["military"](around:{radius_m},{lat:.4f},{lon:.4f});
);
out center 90;
"""


def normalize_overpass(payload: dict) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for el in payload.get("elements") or []:
        tags = el.get("tags") or {}
        lat = el.get("lat") or (el.get("center") or {}).get("lat")
        lon = el.get("lon") or (el.get("center") or {}).get("lon")
        if lat is None or lon is None:
            continue
        name = tags.get("name") or tags.get("ref") or f"{el.get('type')}-{el.get('id')}"
        kind = "site"
        if tags.get("waterway") == "dam" or tags.get("man_made") == "dam":
            kind = "dam"
        elif tags.get("telecom") == "data_center" or tags.get("building") == "data_centre":
            kind = "datacenter"
        elif tags.get("landuse") == "military" or "military" in tags:
            kind = "installation"
        eid = f"osm-{el.get('type')}-{el.get('id')}"
        if eid in seen:
            continue
        seen.add(eid)
        out.append(
            {
                "id": eid,
                "kind": kind,
                "callsign": name,
                "lat": float(lat),
                "lon": float(lon),
                "alt_m": 0.0,
                "heading": 0,
                "osm_kind": kind,
                "source": "osm-overpass",
            }
        )
    rank = {"installation": 0, "datacenter": 1, "dam": 2, "site": 3}

    def _named(callsign: str) -> bool:
        n = str(callsign or "")
        return bool(n) and not n.startswith(("node-", "way-", "relation-"))

    out.sort(key=lambda r: (rank.get(r.get("kind"), 9), 0 if _named(r.get("callsign")) else 1, r.get("callsign") or ""))
    installs = [r for r in out if r.get("kind") != "dam"]
    dams = [r for r in out if r.get("kind") == "dam"][:24]
    return (installs + dams)[:80]


def normalize_radio(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for st in rows or []:
        try:
            lat = float(st.get("geo_lat") or 0)
            lon = float(st.get("geo_long") or 0)
        except (TypeError, ValueError):
            continue
        if not lat and not lon:
            continue
        sid = str(st.get("stationuuid") or st.get("changeuuid") or st.get("name") or "")
        if not sid:
            continue
        out.append(
            {
                "id": f"radio-{sid[:16]}",
                "kind": "radio",
                "callsign": str(st.get("name") or sid).split(" - ")[0][:36],
                "lat": lat,
                "lon": lon,
                "alt_m": 0.0,
                "heading": 0,
                "url": st.get("url_resolved") or st.get("url") or "",
                "codec": st.get("codec") or "",
                "country": st.get("country") or "",
                "source": "radio-browser",
            }
        )
    return out[:40]


async def fetch_fires(client: Optional[httpx.AsyncClient] = None) -> dict[str, Any]:
    hit = cache_get("fires")
    if hit:
        return {**hit, "cached": True}
    own = client is None
    client = client or httpx.AsyncClient(follow_redirects=True)
    fetched_at = _now_ms()
    try:
        try:
            code, text = await _get_text(client, FIRMS_CSV, timeout=20.0)
            if code != 200:
                raise RuntimeError(f"HTTP {code}")
            contacts = parse_firms_csv(text)
            payload = {
                "kind": "fires",
                "status": "live" if contacts else "unavailable",
                "source": "nasa-firms",
                "freshness_ms": fetched_at,
                "label": "LIVE" if contacts else "UNAVAILABLE",
                "count": len(contacts),
                "contacts": contacts,
                "errors": [],
                "note": "NASA FIRMS VIIRS 24h active fire detections. Public CSV, downsampled by FRP.",
            }
        except Exception as e:
            payload = {
                "kind": "fires",
                "status": "unavailable",
                "source": "nasa-firms",
                "freshness_ms": fetched_at,
                "label": "UNAVAILABLE",
                "count": 0,
                "contacts": [],
                "errors": [str(e)],
                "note": "NASA FIRMS CSV unreachable.",
            }
        cache_set("fires", payload, ttl=10 * 60)
        return {**payload, "cached": False}
    finally:
        if own:
            await client.aclose()


def _eonet_cat_id(cat: Any) -> str:
    if isinstance(cat, dict):
        return str(cat.get("id") or cat.get("title") or "").strip()
    return str(cat or "").strip()


def eonet_kind(categories: Any) -> str | None:
    """Map EONET categories to a contact kind. Wildfires are never storms."""
    rows = categories if isinstance(categories, list) else [categories]
    ids = [_eonet_cat_id(c).lower().replace(" ", "").replace("-", "") for c in rows if c]
    if any(i in EONET_SKIP for i in ids):
        return None
    for i in ids:
        kind = EONET_KIND.get(i)
        if kind:
            return kind
    return None


def ring_centroid(ring: Any) -> tuple[float, float] | None:
    """Average of an exterior ring [[lon, lat], ...]. Pure — used by tests."""
    if not isinstance(ring, (list, tuple)) or len(ring) < 1:
        return None
    slat = 0.0
    slon = 0.0
    n = 0
    for pt in ring:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            continue
        try:
            lon = float(pt[0])
            lat = float(pt[1])
        except (TypeError, ValueError):
            continue
        slon += lon
        slat += lat
        n += 1
    if n < 1:
        return None
    return slat / n, slon / n


def geom_centroid(geometry: Any) -> tuple[float, float] | None:
    """Point / Polygon / MultiPolygon → (lat, lon). Pure — used by tests."""
    if not isinstance(geometry, dict):
        return None
    gtype = str(geometry.get("type") or "")
    coords = geometry.get("coordinates")
    if gtype == "Point" and isinstance(coords, (list, tuple)) and len(coords) >= 2:
        try:
            return float(coords[1]), float(coords[0])
        except (TypeError, ValueError):
            return None
    if gtype == "Polygon" and isinstance(coords, (list, tuple)) and coords:
        return ring_centroid(coords[0])
    if gtype == "MultiPolygon" and isinstance(coords, (list, tuple)) and coords:
        first = coords[0]
        if isinstance(first, (list, tuple)) and first:
            return ring_centroid(first[0])
    if isinstance(coords, (list, tuple)) and len(coords) >= 2 and not isinstance(coords[0], (list, tuple)):
        try:
            return float(coords[1]), float(coords[0])
        except (TypeError, ValueError):
            return None
    return None


def latest_eonet_geometry(geoms: Any) -> dict[str, Any] | None:
    rows = [g for g in (geoms or []) if isinstance(g, dict)]
    for g in reversed(rows):
        if geom_centroid(g) or geom_centroid({"type": g.get("type") or "Point", "coordinates": g.get("coordinates")}):
            return g
    return None


def normalize_eonet(payload: dict) -> list[dict[str, Any]]:
    """NASA EONET open events → storm/volcano/flood/dust contacts. Skips wildfires."""
    out: list[dict[str, Any]] = []
    for ev in payload.get("events") or []:
        if not isinstance(ev, dict):
            continue
        kind = eonet_kind(ev.get("categories"))
        if not kind:
            continue
        geom = latest_eonet_geometry(ev.get("geometry"))
        if not geom:
            continue
        pos = geom_centroid(geom) or geom_centroid(
            {"type": geom.get("type") or "Point", "coordinates": geom.get("coordinates")}
        )
        if not pos:
            continue
        lat, lon = pos
        eid = str(ev.get("id") or f"{lat:.3f}-{lon:.3f}")
        cats = ev.get("categories") or []
        cat_title = ""
        if cats and isinstance(cats[0], dict):
            cat_title = str(cats[0].get("title") or cats[0].get("id") or "")
        mag = geom.get("magnitudeValue")
        try:
            mag_f = float(mag) if mag is not None and mag != "" else None
        except (TypeError, ValueError):
            mag_f = None
        row: dict[str, Any] = {
            "id": f"eonet-{eid}",
            "kind": kind,
            "callsign": str(ev.get("title") or eid).strip() or eid,
            "lat": lat,
            "lon": lon,
            "alt_m": 0.0,
            "heading": 0,
            "category": cat_title or kind,
            "source": "nasa-eonet",
        }
        if mag_f is not None:
            row["mag"] = mag_f
        if geom.get("date"):
            row["time"] = geom.get("date")
        if ev.get("link"):
            row["url"] = ev.get("link")
        out.append(row)
    return out[:60]


def nws_high_signal(event: str) -> bool:
    """Keep tornado/hurricane/flood/winter warnings — drop Small Craft / Beach Hazards."""
    return str(event or "").strip().lower() in NWS_KEEP


def normalize_nws(payload: dict) -> list[dict[str, Any]]:
    """NWS active alerts GeoJSON → high-signal alert contacts. Pure — used by tests."""
    out: list[dict[str, Any]] = []
    for feat in payload.get("features") or []:
        if not isinstance(feat, dict):
            continue
        props = feat.get("properties") or {}
        event = str(props.get("event") or "").strip()
        if not nws_high_signal(event):
            continue
        pos = geom_centroid(feat.get("geometry"))
        if not pos:
            continue
        lat, lon = pos
        raw_id = str(feat.get("id") or props.get("id") or f"{lat:.3f}-{lon:.3f}")
        short = raw_id.rsplit("/", 1)[-1]
        if short.startswith("urn:oid:"):
            short = short.split(":")[-1]
        area = str(props.get("areaDesc") or "").strip()
        title = str(props.get("headline") or event).strip()
        if area and event and event.lower() not in title.lower():
            title = f"{event} · {area.split(';')[0][:48]}"
        elif area and title == event:
            title = f"{event} · {area.split(';')[0][:48]}"
        row = {
            "id": f"nws-{short}"[:48],
            "kind": "alert",
            "callsign": title[:80] or event,
            "lat": lat,
            "lon": lon,
            "alt_m": 0.0,
            "heading": 0,
            "category": event,
            "area": area,
            "severity": str(props.get("severity") or ""),
            "source": "nws",
        }
        out.append(row)
        if len(out) >= 80:
            break
    return out


def _dedupe_contacts(rows: list[dict[str, Any]], cap: int = 140) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for r in rows:
        i = str(r.get("id") or "")
        if not i or i in seen:
            continue
        seen.add(i)
        out.append(r)
        if len(out) >= cap:
            break
    return out


async def fetch_storms(client: Optional[httpx.AsyncClient] = None) -> dict[str, Any]:
    """NASA EONET storms/volcanoes/floods/dust + high-signal NWS alerts."""
    hit = cache_get("storms")
    if hit:
        return {**hit, "cached": True}
    own = client is None
    client = client or httpx.AsyncClient(follow_redirects=True)
    fetched_at = _now_ms()
    contacts: list[dict[str, Any]] = []
    errors: list[str] = []
    sources: list[str] = []
    failed = 0
    try:
        import json

        try:
            code, text = await _get_text(
                client,
                EONET_EVENTS,
                timeout=14.0,
                params={
                    "status": "open",
                    "category": "severeStorms,volcanoes,floods,dustHaze",
                    "limit": "60",
                },
            )
            if code != 200:
                raise RuntimeError(f"HTTP {code}")
            body = json.loads(text) if text.lstrip().startswith("{") else {}
            eonet = normalize_eonet(body)
            contacts.extend(eonet)
            sources.append("nasa-eonet")
        except Exception as e:
            failed += 1
            errors.append(f"eonet {e}")

        try:
            code, text = await _get_text(
                client,
                NWS_ALERTS,
                timeout=14.0,
                params={"status": "actual", "message_type": "alert"},
                headers=_nws_headers(),
            )
            if code != 200:
                raise RuntimeError(f"HTTP {code}")
            body = json.loads(text) if text.lstrip().startswith("{") else {}
            nws = normalize_nws(body)
            contacts.extend(nws)
            sources.append("nws")
        except Exception as e:
            failed += 1
            errors.append(f"nws {e}")

        contacts = _dedupe_contacts(contacts)
        both_down = failed >= 2
        if both_down:
            status, label = "unavailable", "UNAVAILABLE"
        elif contacts:
            status, label = "live", "LIVE"
        else:
            status, label = "empty", "EMPTY"
        source = "+".join(sources) if sources else "nasa-eonet|nws"
        payload = {
            "kind": "storms",
            "status": status,
            "source": source,
            "freshness_ms": fetched_at,
            "label": label,
            "count": len(contacts),
            "contacts": contacts,
            "errors": errors[:6],
            "note": "NASA EONET open storms/volcanoes/floods/dust + NWS high-signal alerts. Not a fires layer.",
        }
        cache_set("storms", payload, ttl=STORMS_TTL_S)
        return {**payload, "cached": False}
    finally:
        if own:
            await client.aclose()


async def fetch_sites(
    lat: float,
    lon: float,
    radius_km: float = 80,
    client: Optional[httpx.AsyncClient] = None,
) -> dict[str, Any]:
    lat = max(-89.9, min(89.9, float(lat)))
    lon = max(-179.9, min(179.9, float(lon)))
    radius_m = int(max(15, min(180, float(radius_km))) * 1000)
    key = f"sites:{lat:.1f}:{lon:.1f}:{radius_m}"
    hit = cache_get(key)
    if hit:
        return {**hit, "cached": True}
    own = client is None
    client = client or httpx.AsyncClient(follow_redirects=True)
    fetched_at = _now_ms()
    try:
        try:
            last_err: Exception | None = None
            contacts: list[dict[str, Any]] = []
            for url in OVERPASS_MIRRORS:
                try:
                    r = await client.post(
                        url,
                        content=_overpass_ql(lat, lon, radius_m),
                        headers={**_headers(), "Content-Type": "text/plain"},
                        timeout=24.0,
                    )
                    if r.status_code != 200:
                        raise RuntimeError(f"HTTP {r.status_code}")
                    contacts = normalize_overpass(r.json())
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    continue
            if last_err and not contacts:
                raise last_err
            payload = {
                "kind": "sites",
                "status": "live" if contacts else "empty",
                "source": "osm-overpass",
                "freshness_ms": fetched_at,
                "label": "LIVE" if contacts else "EMPTY",
                "count": len(contacts),
                "contacts": contacts,
                "errors": [],
                "note": "OSM installations, dams, and data centers near the look point. Live Overpass, not a copied extract.",
            }
        except Exception as e:
            payload = {
                "kind": "sites",
                "status": "unavailable",
                "source": "osm-overpass",
                "freshness_ms": fetched_at,
                "label": "UNAVAILABLE",
                "count": 0,
                "contacts": [],
                "errors": [str(e)],
                "note": "Overpass unreachable.",
            }
        cache_set(key, payload, ttl=15 * 60)
        return {**payload, "cached": False}
    finally:
        if own:
            await client.aclose()


async def fetch_radio(
    lat: float,
    lon: float,
    client: Optional[httpx.AsyncClient] = None,
) -> dict[str, Any]:
    lat = max(-89.9, min(89.9, float(lat)))
    lon = max(-179.9, min(179.9, float(lon)))
    key = f"radio:{lat:.1f}:{lon:.1f}"
    hit = cache_get(key)
    if hit:
        return {**hit, "cached": True}
    own = client is None
    client = client or httpx.AsyncClient(follow_redirects=True)
    fetched_at = _now_ms()
    try:
        try:
            last_err: Exception | None = None
            contacts: list[dict[str, Any]] = []
            params = {
                "geo_lat": f"{lat:.3f}",
                "geo_long": f"{lon:.3f}",
                "geo_distance": "250000",
                "order": "clickcount",
                "reverse": "true",
                "limit": "24",
                "hidebroken": "true",
            }
            for url in RADIO_MIRRORS:
                try:
                    r = await client.get(url, params=params, headers=_headers(), timeout=12.0)
                    if r.status_code != 200:
                        raise RuntimeError(f"HTTP {r.status_code}")
                    contacts = normalize_radio(r.json())
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    continue
            if last_err and not contacts:
                raise last_err
            payload = {
                "kind": "radio",
                "status": "live" if contacts else "empty",
                "source": "radio-browser",
                "freshness_ms": fetched_at,
                "label": "LIVE" if contacts else "EMPTY",
                "count": len(contacts),
                "contacts": contacts,
                "errors": [],
                "note": "Public radio-browser.info stations near the look point.",
            }
        except Exception as e:
            payload = {
                "kind": "radio",
                "status": "unavailable",
                "source": "radio-browser",
                "freshness_ms": fetched_at,
                "label": "UNAVAILABLE",
                "count": 0,
                "contacts": [],
                "errors": [str(e)],
                "note": "radio-browser unreachable.",
            }
        cache_set(key, payload, ttl=20 * 60)
        return {**payload, "cached": False}
    finally:
        if own:
            await client.aclose()


def normalize_launches(payload: dict) -> list[dict[str, Any]]:
    """Launch Library 2 → pad contacts. Pure — used by tests."""
    out: list[dict[str, Any]] = []
    for row in payload.get("results") or []:
        pad = row.get("pad") or {}
        loc = pad.get("location") or {}
        try:
            lat = float(pad.get("latitude") or loc.get("latitude") or 0)
            lon = float(pad.get("longitude") or loc.get("longitude") or 0)
        except (TypeError, ValueError):
            continue
        if not lat and not lon:
            continue
        rocket = ((row.get("rocket") or {}).get("configuration") or {}).get("name") or ""
        status = (row.get("status") or {}).get("name") or ""
        mission = (row.get("mission") or {}).get("name") or ""
        lid = str(row.get("id") or row.get("slug") or "")
        if not lid:
            continue
        out.append(
            {
                "id": f"ll2-{lid[:18]}",
                "kind": "mission",
                "callsign": row.get("name") or mission or lid,
                "lat": lat,
                "lon": lon,
                "alt_m": 0.0,
                "heading": 90.0,
                "rocket": rocket,
                "status": status,
                "net": row.get("net") or "",
                "pad": pad.get("name") or "",
                "provider": (row.get("launch_service_provider") or {}).get("name") or "",
                "mission": mission,
                "reconstructed": True,
                "source": "launch-library-2",
            }
        )
    return out[:40]


def is_browser_video(url: str) -> bool:
    """HTML5 <video src> plays mp4/webm. Caltrans HLS playlists do not."""
    path = urlparse(str(url or "")).path.lower()
    return any(path.endswith(ext) for ext in BROWSER_VIDEO_SUFFIXES)


def still_is_placeholder(url: str, status: int, length: int | None, etag: str | None) -> bool:
    if status != 200:
        return False
    et = str(etag or "").strip().strip('"').lower()
    if et == AUSTIN_DEAD_STILL_ETAG:
        return True
    host = urlparse(str(url or "")).netloc.lower()
    if "austinmobility.io" in host and length == AUSTIN_DEAD_STILL_BYTES:
        return True
    return False


def allowed_cctv_media(url: str) -> bool:
    parsed = urlparse(str(url or ""))
    if parsed.scheme != "https" or parsed.netloc not in CCTV_MEDIA_HOSTS:
        return False
    if parsed.netloc == "s3-eu-west-1.amazonaws.com" and "jamcams.tfl.gov.uk" not in parsed.path:
        return False
    return True


def _video_or_blank(url: str) -> str:
    raw = str(url or "").strip()
    return raw if is_browser_video(raw) else ""


def normalize_tfl_cameras(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for place in rows or []:
        try:
            lat = float(place.get("lat"))
            lon = float(place.get("lon"))
        except (TypeError, ValueError):
            continue
        props = {p.get("key"): p.get("value") for p in (place.get("additionalProperties") or []) if p.get("key")}
        still = props.get("imageUrl") or props.get("fullUrl") or ""
        video = _video_or_blank(props.get("videoUrl") or "")
        pid = str(place.get("id") or place.get("naptanId") or "")
        if not pid:
            continue
        out.append(
            {
                "id": f"cctv-tfl-{pid[-16:]}",
                "kind": "cctv",
                "callsign": place.get("commonName") or pid,
                "lat": lat,
                "lon": lon,
                "alt_m": 12.0,
                "heading": 0,
                "still": still,
                "video": video,
                "city": "London",
                "source": "tfl-jamcam",
            }
        )
    return out


def normalize_austin_cameras(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Austin SODA traffic cameras. Pure — used by tests."""
    out: list[dict[str, Any]] = []
    for row in rows or []:
        loc = row.get("location") or {}
        coords = loc.get("coordinates") if isinstance(loc, dict) else None
        try:
            if coords and len(coords) >= 2:
                lon, lat = float(coords[0]), float(coords[1])
            else:
                lat = float(row.get("latitude") or row.get("lat") or 0)
                lon = float(row.get("longitude") or row.get("lon") or 0)
        except (TypeError, ValueError):
            continue
        if not lat and not lon:
            continue
        cid = str(row.get("camera_id") or row.get("id") or "")
        if not cid:
            continue
        still = str(row.get("screenshot_address") or "").strip()
        if not still and cid.isdigit():
            still = f"https://cctv.austinmobility.io/image/{cid}.jpg"
        heading = 0.0
        for key in ("primary_st", "location_name"):
            name = str(row.get(key) or "")
            if " / " in name or " SVRD" in name:
                heading = 90.0
                break
        out.append(
            {
                "id": f"cctv-atx-{cid}"[:28],
                "kind": "cctv",
                "callsign": str(row.get("location_name") or cid).strip() or cid,
                "lat": lat,
                "lon": lon,
                "alt_m": 14.0,
                "heading": heading,
                "still": still,
                "video": "",
                "city": "Austin",
                "source": "austin-open-data",
            }
        )
    return out


def _caltrans_events(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        cctv = data.get("cctv")
        if isinstance(cctv, dict):
            svc = cctv.get("service") or {}
            ev = svc.get("event") if isinstance(svc, dict) else cctv
            if isinstance(ev, list):
                return ev
            if isinstance(ev, dict):
                return [ev]
            return [cctv]
    ev = payload.get("cctv")
    if isinstance(ev, list):
        return ev
    if isinstance(ev, dict):
        return [ev]
    return []


def normalize_caltrans_cameras(payload: Any, district: str) -> list[dict[str, Any]]:
    """Caltrans CWWP2 CCTV. `data` is a list of `{cctv: {...}}` objects."""
    out: list[dict[str, Any]] = []
    for row in _caltrans_events(payload):
        ev = row.get("cctv") if isinstance(row, dict) and isinstance(row.get("cctv"), dict) else row
        if not isinstance(ev, dict):
            continue
        loc = ev.get("location") or {}
        try:
            lat = float(loc.get("latitude") or ev.get("latitude") or 0)
            lon = float(loc.get("longitude") or ev.get("longitude") or 0)
        except (TypeError, ValueError):
            continue
        if not lat and not lon:
            continue
        img_block = (ev.get("imageData") or {}).get("static")
        if isinstance(img_block, dict):
            img = img_block.get("currentImageURL") or img_block.get("stillUrl") or ""
        else:
            img = img_block or (ev.get("imageData") or {}).get("stillUrl") or ""
        heading = {
            "north": 0,
            "east": 90,
            "south": 180,
            "west": 270,
        }.get(str(loc.get("direction") or "").lower(), 0)
        eid = str(ev.get("id") or ev.get("index") or loc.get("locationName") or "")
        if not eid:
            continue
        out.append(
            {
                "id": f"cctv-ca-{district}-{eid}"[:36],
                "kind": "cctv",
                "callsign": loc.get("locationName") or ev.get("title") or eid,
                "lat": lat,
                "lon": lon,
                "alt_m": 14.0,
                "heading": heading,
                "still": img,
                "video": _video_or_blank((ev.get("imageData") or {}).get("streamingVideoURL") or ""),
                "city": "California",
                "source": "caltrans-cwwp2",
            }
        )
    return out


async def fetch_missions(client: Optional[httpx.AsyncClient] = None) -> dict[str, Any]:
    hit = cache_get("missions")
    if hit:
        return {**hit, "cached": True}
    own = client is None
    client = client or httpx.AsyncClient(follow_redirects=True)
    fetched_at = _now_ms()
    try:
        contacts: list[dict[str, Any]] = []
        errors: list[str] = []
        for path in ("upcoming", "previous"):
            try:
                r = await client.get(
                    f"{LAUNCH_LIBRARY}/{path}/",
                    params={"limit": "24", "mode": "normal"},
                    headers=_headers(),
                    timeout=14.0,
                )
                if r.status_code != 200:
                    raise RuntimeError(f"HTTP {r.status_code}")
                contacts.extend(normalize_launches(r.json()))
            except Exception as e:
                errors.append(f"{path}: {e}")
        seen: set[str] = set()
        uniq: list[dict[str, Any]] = []
        for c in contacts:
            if c["id"] in seen:
                continue
            seen.add(c["id"])
            uniq.append(c)
        payload = {
            "kind": "missions",
            "status": "live" if uniq else "unavailable",
            "source": "launch-library-2",
            "freshness_ms": fetched_at,
            "label": "LIVE" if uniq else "UNAVAILABLE",
            "count": len(uniq),
            "contacts": uniq,
            "errors": errors,
            "note": "Launch Library 2 pads, last/next ~30 days. Ascent replay is a labeled RECONSTRUCTED ESTIMATE.",
        }
        cache_set("missions", payload, ttl=15 * 60)
        return {**payload, "cached": False}
    finally:
        if own:
            await client.aclose()


def _assemble_cctv(contacts: list[dict[str, Any]], errors: list[str], fetched_at: int) -> dict[str, Any]:
    by_city: dict[str, list[dict[str, Any]]] = {}
    offline = 0
    for cam in contacts:
        if cam.get("live") is False:
            offline += 1
            continue
        by_city.setdefault(str(cam.get("city") or "other"), []).append(cam)
    mixed: list[dict[str, Any]] = []
    for rows in by_city.values():
        mixed.extend(rows[:320])
    return {
        "kind": "cctv",
        "status": "live" if mixed else "unavailable",
        "source": "tfl|austin-open-data|caltrans",
        "freshness_ms": fetched_at,
        "label": "LIVE" if mixed else "UNAVAILABLE",
        "count": len(mixed),
        "cities": {k: len(v) for k, v in by_city.items()},
        "offline": offline,
        "contacts": mixed[:960],
        "errors": errors[:8],
        "note": "Public city camera lists. Dead Austin stills and HLS-only Caltrans streams are dropped so the HUD shows working footage.",
    }


async def probe_austin_stills(client: httpx.AsyncClient, contacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop Austin cameras whose still is the shared 12 KB no-signal JPEG.

    Bound concurrency — 800 parallel HEADs time out on Railway and we keep the dead ones.
    """
    gate = asyncio.Semaphore(16)
    others = [c for c in contacts if c.get("city") != "Austin"]
    austin = [c for c in contacts if c.get("city") == "Austin"]

    async def one(cam: dict[str, Any]) -> dict[str, Any]:
        if not cam.get("still"):
            return cam
        async with gate:
            try:
                r = await client.head(cam["still"], headers=_headers(), timeout=8.0)
                length = r.headers.get("content-length")
                n = int(length) if length and str(length).isdigit() else None
                if r.status_code in (403, 405, 501) or (r.status_code == 200 and n is None):
                    r = await client.get(cam["still"], headers=_headers(), timeout=10.0)
                    n = len(r.content)
                if still_is_placeholder(cam["still"], r.status_code, n, r.headers.get("etag")):
                    return {**cam, "live": False}
                return {**cam, "live": True}
            except Exception:
                return {**cam, "live": True}

    probed = list(await asyncio.gather(*[one(c) for c in austin]))
    return others + probed


async def proxy_cctv_media(url: str):
    """Same-origin still so city CDNs cannot hotlink-block the cockpit."""
    from fastapi.responses import Response

    if not allowed_cctv_media(url):
        return Response(status_code=400, content=b"blocked")
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            r = await client.get(url, headers=_headers(), timeout=16.0)
    except Exception:
        return Response(status_code=502, content=b"upstream")
    length = int(r.headers.get("content-length") or len(r.content) or 0)
    if still_is_placeholder(url, r.status_code, length, r.headers.get("etag")):
        return Response(status_code=404, content=b"offline")
    if r.status_code != 200 or not r.content:
        return Response(status_code=404, content=b"missing")
    ct = (r.headers.get("content-type") or "image/jpeg").split(";")[0]
    if not (ct.startswith("image/") or ct in ("application/octet-stream",)):
        return Response(status_code=415, content=b"not-image")
    return Response(
        content=r.content,
        media_type=ct,
        headers={"Cache-Control": "public, max-age=20"},
    )


def imagery_bbox(lat: float, lon: float, zoom: int, width: int = 1280, height: int = 720) -> tuple[float, float, float, float]:
    """Geographic window for a satellite still at a Google-like zoom."""
    z = max(4, min(19, int(zoom)))
    deg_per_px = 360.0 / (256 * (2**z))
    clat = max(0.2, math.cos(math.radians(lat)))
    dlon = (width / 2.0) * deg_per_px / clat
    dlat = (height / 2.0) * deg_per_px
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def google_static_map_url(lat: float, lon: float, zoom: int, key: str, size: str = "640x360", scale: int = 2) -> str:
    from urllib.parse import urlencode

    q = urlencode(
        {
            "center": f"{float(lat):.5f},{float(lon):.5f}",
            "zoom": str(int(zoom)),
            "size": size,
            "scale": str(int(scale)),
            "maptype": "satellite",
            "key": key,
        }
    )
    return f"{GOOGLE_STATIC_MAP}?{q}"


def esri_imagery_url(lat: float, lon: float, zoom: int, width: int = 1280, height: int = 720) -> str:
    from urllib.parse import urlencode

    minx, miny, maxx, maxy = imagery_bbox(lat, lon, zoom, width, height)
    q = urlencode(
        {
            "bbox": f"{minx:.6f},{miny:.6f},{maxx:.6f},{maxy:.6f}",
            "bboxSR": "4326",
            "imageSR": "4326",
            "size": f"{int(width)},{int(height)}",
            "format": "jpg",
            "f": "image",
        }
    )
    return f"{ESRI_IMAGERY}?{q}"


def _still_ok(status: int, content: bytes, content_type: str) -> bool:
    if status != 200 or not content or len(content) < 8000:
        return False
    ct = (content_type or "").split(";")[0].lower()
    if ct.startswith("image/gif"):
        return False
    if content[:6] in (b"GIF87a", b"GIF89a"):
        return False
    return ct.startswith("image/") or ct in ("application/octet-stream", "")


async def proxy_sat_still(lat: float, lon: float, zoom: int = 15, settings: Any = None):
    """Same-origin satellite still — Google Static Maps, then ESRI World Imagery."""
    from fastapi.responses import Response

    lat = max(-85.0, min(85.0, float(lat)))
    lon = max(-179.9, min(179.9, float(lon)))
    zoom = max(4, min(19, int(zoom)))
    cache_key = f"satstill:{lat:.4f}:{lon:.4f}:{zoom}"
    hit = cache_get(cache_key)
    if isinstance(hit, dict) and hit.get("bytes"):
        return Response(
            content=hit["bytes"],
            media_type=hit.get("ct") or "image/jpeg",
            headers={"Cache-Control": "public, max-age=3600"},
        )
    key = ""
    if settings is not None:
        key = (getattr(settings, "google_maps_api_key", None) or "").strip()
    urls: list[str] = []
    if key:
        urls.append(google_static_map_url(lat, lon, zoom, key))
    urls.append(esri_imagery_url(lat, lon, zoom))
    last_status = 502
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            for url in urls:
                try:
                    r = await client.get(url, headers=_headers(), timeout=16.0)
                except Exception:
                    continue
                ct = (r.headers.get("content-type") or "image/jpeg").split(";")[0]
                if _still_ok(r.status_code, r.content, ct):
                    cache_set(cache_key, {"bytes": r.content, "ct": ct}, ttl=60 * 60)
                    return Response(
                        content=r.content,
                        media_type=ct or "image/jpeg",
                        headers={"Cache-Control": "public, max-age=3600"},
                    )
                last_status = r.status_code or last_status
    except Exception:
        last_status = 502
    return Response(status_code=last_status if last_status >= 400 else 502, content=b"missing")


async def _load_cctv_now(client: Optional[httpx.AsyncClient] = None) -> dict[str, Any]:
    own = client is None
    client = client or httpx.AsyncClient(
        follow_redirects=True,
        limits=httpx.Limits(max_connections=24, max_keepalive_connections=12),
    )
    fetched_at = _now_ms()
    errors: list[str] = []

    async def tfl() -> list[dict[str, Any]]:
        try:
            r = await client.get(TFL_JAMCAM, headers=_headers(), timeout=10.0)
            if r.status_code == 200:
                return normalize_tfl_cameras(r.json() if isinstance(r.json(), list) else [])
            errors.append(f"tfl HTTP {r.status_code}")
        except Exception as e:
            errors.append(f"tfl: {e}")
        return []

    async def austin() -> list[dict[str, Any]]:
        try:
            r = await client.get(
                AUSTIN_CAMERAS,
                params={"$limit": "800", "$where": "camera_status='TURNED_ON'"},
                headers=_headers(),
                timeout=10.0,
            )
            if r.status_code == 200:
                body = r.json()
                return normalize_austin_cameras(body if isinstance(body, list) else [])
            errors.append(f"austin HTTP {r.status_code}")
        except Exception as e:
            errors.append(f"austin: {e}")
        return []

    async def caltrans(url: str) -> list[dict[str, Any]]:
        district = url.split("/data/")[-1].split("/")[0]
        try:
            r = await client.get(url, headers=_headers(), timeout=10.0)
            if r.status_code == 200:
                return normalize_caltrans_cameras(r.json(), district)
            errors.append(f"{district} HTTP {r.status_code}")
        except Exception as e:
            errors.append(f"{district}: {e}")
        return []

    try:
        chunks = await asyncio.gather(tfl(), austin(), *[caltrans(u) for u in CALTRANS_CCTV])
        contacts: list[dict[str, Any]] = []
        for chunk in chunks:
            contacts.extend(chunk)
        contacts = await probe_austin_stills(client, contacts)
        payload = _assemble_cctv(contacts, errors, fetched_at)
        if payload["count"]:
            cache_set("cctv:v2", payload, ttl=30 * 60)
        return {**payload, "cached": False}
    finally:
        if own:
            await client.aclose()


_CCTV_REFRESHING = False


async def warm_cctv() -> None:
    """Fill the camera cache on boot so the rail is not 0 after a deploy."""
    try:
        await _load_cctv_now()
    except Exception:
        pass


async def fetch_cctv(client: Optional[httpx.AsyncClient] = None) -> dict[str, Any]:
    hit, fresh = cache_peek("cctv:v2")
    if hit and fresh:
        return {**hit, "cached": True}
    if hit:
        global _CCTV_REFRESHING
        if not _CCTV_REFRESHING:
            _CCTV_REFRESHING = True

            async def _bg() -> None:
                global _CCTV_REFRESHING
                try:
                    await _load_cctv_now()
                finally:
                    _CCTV_REFRESHING = False

            try:
                asyncio.get_running_loop().create_task(_bg())
            except RuntimeError:
                _CCTV_REFRESHING = False
        return {**hit, "cached": True, "stale": True}
    return await _load_cctv_now(client)


def tile_config(settings: Any) -> dict[str, Any]:
    """Expose optional client tile credentials only when actually set in env."""
    cesium = (getattr(settings, "cesium_ion_token", None) or "").strip()
    google = (getattr(settings, "google_maps_api_key", None) or "").strip()
    mode = "none"
    if google:
        mode = "google"
    elif cesium:
        mode = "cesium"
    from app.services.godseye_ais import ships_provider

    provider = ships_provider()
    cctv, _fresh = cache_peek("cctv:v2")
    cctv_count = int(cctv["count"]) if isinstance(cctv, dict) and cctv.get("count") else 0
    return {
        "tiles": mode,
        "cesiumIonToken": cesium or None,
        "googleMapsApiKey": google or None,
        "hasPhotoreal": bool(cesium or google),
        "hasShips": True,
        "shipsProvider": provider,
        "hasCctv": bool(cctv_count),
        "cctvCount": cctv_count or None,
        "note": "Photoreal 3D tiles load only when a real env token is present. Ships use keyless Open Waters AIS; AISSTREAM_API_KEY stays optional. Never invented.",
    }
