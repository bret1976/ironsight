/** WGS84 helpers for the GodsEye globe (meters, Three.js SphereGeometry frame). */

export const EARTH_R = 6_371_000

export function clamp(v, a, b) {
  return Math.max(a, Math.min(b, v))
}

export function deg(rad) {
  return (rad * 180) / Math.PI
}

export function rad(d) {
  return (d * Math.PI) / 180
}

/**
 * Match THREE.SphereGeometry UV/axis:
 * lon -180 at -X, lon 0 at +X, north +Y.
 */
export function latLonAltToXYZ(lat, lon, alt = 0) {
  const phi = rad(90 - lat)
  const theta = rad(lon + 180)
  const r = EARTH_R + alt
  const sinP = Math.sin(phi)
  return {
    x: -r * Math.cos(theta) * sinP,
    y: r * Math.cos(phi),
    z: r * Math.sin(theta) * sinP,
  }
}

export function xyzToLatLonAlt(x, y, z) {
  const r = Math.hypot(x, y, z) || 1
  const lat = deg(Math.asin(clamp(y / r, -1, 1)))
  // Inverse of SphereGeometry: x = R cos(lat) cos(lon), z = -R cos(lat) sin(lon)
  const lon = deg(Math.atan2(-z, x))
  return { lat, lon, alt: r - EARTH_R, r }
}

export function haversineKm(lat1, lon1, lat2, lon2) {
  const R = 6371
  const dLat = rad(lat2 - lat1)
  const dLon = rad(lon2 - lon1)
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(rad(lat1)) * Math.cos(rad(lat2)) * Math.sin(dLon / 2) ** 2
  return 2 * R * Math.asin(Math.min(1, Math.sqrt(a)))
}

export function lerp(a, b, t) {
  return a + (b - a) * t
}

export function lerpAngleDeg(a, b, t) {
  let d = ((b - a + 540) % 360) - 180
  return (a + d * t + 360) % 360
}

export function lerpLon(a, b, t) {
  let d = ((b - a + 540) % 360) - 180
  let v = a + d * t
  if (v > 180) v -= 360
  if (v < -180) v += 360
  return v
}

export function interpolateContact(prev, next, now) {
  if (!prev) return { ...next }
  const t0 = prev.t
  const t1 = next.t
  const span = Math.max(1, t1 - t0)
  const u = clamp((now - t0) / span, 0, 1.35)
  return {
    ...next,
    lat: lerp(prev.lat, next.lat, u),
    lon: lerpLon(prev.lon, next.lon, u),
    alt_m: lerp(prev.alt_m ?? 0, next.alt_m ?? 0, u),
    heading: lerpAngleDeg(prev.heading || 0, next.heading || 0, Math.min(1, u)),
    _interp: u,
  }
}

export function formatLatLon(lat, lon) {
  if (lat == null || lon == null || Number.isNaN(lat) || Number.isNaN(lon)) return '—'
  const ns = lat >= 0 ? 'N' : 'S'
  const ew = lon >= 0 ? 'E' : 'W'
  return `${Math.abs(lat).toFixed(3)}°${ns}  ${Math.abs(lon).toFixed(3)}°${ew}`
}

export function formatAlt(meters) {
  if (meters == null || Number.isNaN(meters)) return '—'
  if (Math.abs(meters) >= 1000) return `${(meters / 1000).toFixed(1)} km`
  return `${Math.round(meters)} m`
}

function dmsPart(v, pos, neg) {
  const a = Math.abs(v)
  const d = Math.floor(a)
  const mFloat = (a - d) * 60
  const m = Math.floor(mFloat)
  const s = (mFloat - m) * 60
  return `${d}°${String(m).padStart(2, '0')}'${s.toFixed(1).padStart(4, '0')}"${v >= 0 ? pos : neg}`
}

export function formatDms(lat, lon) {
  if (lat == null || lon == null || Number.isNaN(lat) || Number.isNaN(lon)) return '—'
  return `${dmsPart(lat, 'N', 'S')}  ${dmsPart(lon, 'E', 'W')}`
}

const MGRS_AZ = 'ABCDEFGHJKLMNPQRSTUVWXYZ'

export function formatMgrs(lat, lon) {
  if (lat == null || lon == null || Number.isNaN(lat) || Number.isNaN(lon)) return '—'
  let zone = Math.floor((lon + 180) / 6) + 1
  if (lat >= 56 && lat < 64 && lon >= 3 && lon < 12) zone = 32
  const bands = 'CDEFGHJKLMNPQRSTUVWX'
  const band = bands[Math.min(19, Math.max(0, Math.floor((lat + 80) / 8)))]
  const eRaw = ((lon - ((zone - 1) * 6 - 180) + 3) / 6) * 1e5
  const nRaw = ((lat + 80) % 8) / 8 * 1e5
  const col = MGRS_AZ[Math.abs(Math.floor(eRaw / 1e5)) % 24]
  const row = MGRS_AZ[Math.abs(Math.floor(nRaw / 1e5)) % 24]
  const e = String(Math.floor(Math.abs(eRaw) % 1e5)).padStart(5, '0').slice(0, 4)
  const n = String(Math.floor(Math.abs(nRaw) % 1e5)).padStart(5, '0').slice(0, 4)
  return `${zone}${band} ${col}${row} ${e} ${n}`
}

export function estimateGsd(altM) {
  if (altM == null || Number.isNaN(altM)) return '—'
  const g = Math.max(0.08, Math.abs(altM) * 0.00022)
  if (g >= 1000) return `${(g / 1000).toFixed(1)}km`
  if (g >= 10) return `${g.toFixed(0)}m`
  return `${g.toFixed(1)}m`
}

export function greatCirclePoints(lat1, lon1, lat2, lon2, n = 56) {
  const φ1 = rad(lat1)
  const λ1 = rad(lon1)
  const φ2 = rad(lat2)
  const λ2 = rad(lon2)
  const x1 = Math.cos(φ1) * Math.cos(λ1)
  const y1 = Math.cos(φ1) * Math.sin(λ1)
  const z1 = Math.sin(φ1)
  const x2 = Math.cos(φ2) * Math.cos(λ2)
  const y2 = Math.cos(φ2) * Math.sin(λ2)
  const z2 = Math.sin(φ2)
  const dot = clamp(x1 * x2 + y1 * y2 + z1 * z2, -1, 1)
  const ω = Math.acos(dot)
  const pts = []
  if (ω < 1e-6) {
    pts.push(latLonAltToXYZ(lat1, lon1, 900))
    return pts
  }
  const sinω = Math.sin(ω)
  for (let i = 0; i <= n; i++) {
    const t = i / n
    const a = Math.sin((1 - t) * ω) / sinω
    const b = Math.sin(t * ω) / sinω
    const x = a * x1 + b * x2
    const y = a * y1 + b * y2
    const z = a * z1 + b * z2
    const lat = deg(Math.atan2(z, Math.hypot(x, y)))
    const lon = deg(Math.atan2(y, x))
    pts.push(latLonAltToXYZ(lat, lon, 1400))
  }
  return pts
}

export function headingFromCamera(x, z) {
  return (deg(Math.atan2(x, z)) + 360) % 360
}

/** Local ENU-ish basis at a lat/lon on the SphereGeometry globe. */
export function localFrame(lat, lon) {
  const p = latLonAltToXYZ(lat, lon, 0)
  const rlen = Math.hypot(p.x, p.y, p.z) || 1
  const radial = { x: p.x / rlen, y: p.y / rlen, z: p.z / rlen }
  const east = { x: radial.z, y: 0, z: -radial.x }
  const elen = Math.hypot(east.x, east.y, east.z) || 1
  east.x /= elen
  east.z /= elen
  const north = {
    x: radial.y * east.z - radial.z * east.y,
    y: radial.z * east.x - radial.x * east.z,
    z: radial.x * east.y - radial.y * east.x,
  }
  return { radial, east, north }
}

/** Keep React state to the nearest N contacts so a 400-ship ingest cannot stall the HUD. */
export function nearestContacts(list, lat, lon, cap) {
  const rows = Array.isArray(list) ? list : []
  if (rows.length <= cap) return rows
  return rows
    .map((c) => ({ c, d: haversineKm(lat, lon, c.lat, c.lon) }))
    .sort((a, b) => a.d - b.d)
    .slice(0, cap)
    .map((r) => r.c)
}

export function headingVector(lat, lon, headingDeg) {
  const { east, north } = localFrame(lat, lon)
  const h = rad(headingDeg || 0)
  const c = Math.cos(h)
  const s = Math.sin(h)
  return {
    x: north.x * c + east.x * s,
    y: north.y * c + east.y * s,
    z: north.z * c + east.z * s,
  }
}
