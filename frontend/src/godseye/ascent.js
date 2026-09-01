/** Reconstructed launch ascent — labeled estimate, not telemetry. */

import { latLonAltToXYZ } from './geo.js'

export function reconstructedAscent(lat, lon, heading = 90, t01 = 0) {
  const t = Math.max(0, Math.min(1.15, t01))
  const pitch = 90 - Math.min(88, t * t * 92)
  const alt = t < 1 ? t * t * 185_000 : 185_000 + (t - 1) * 40_000
  const rangeKm = t * t * 420
  const h = (heading * Math.PI) / 180
  const dlat = (Math.cos(h) * rangeKm) / 111
  const dlon = (Math.sin(h) * rangeKm) / (111 * Math.max(0.2, Math.cos((lat * Math.PI) / 180)))
  return {
    lat: lat + dlat,
    lon: lon + dlon,
    alt_m: alt,
    heading,
    pitch,
    reconstructed: true,
  }
}

export function ascentTrail(lat, lon, heading = 90, n = 40) {
  const pts = []
  for (let i = 0; i <= n; i++) {
    const p = reconstructedAscent(lat, lon, heading, i / n)
    pts.push(latLonAltToXYZ(p.lat, p.lon, p.alt_m))
  }
  return pts
}
