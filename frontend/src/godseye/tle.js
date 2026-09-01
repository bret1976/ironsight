/** CelesTrak OMM / classic TLE parse + SGP4 propagate via satellite.js */

import * as satellite from 'satellite.js'

const OMM_REQUIRED = [
  'EPOCH',
  'MEAN_MOTION',
  'ECCENTRICITY',
  'INCLINATION',
  'RA_OF_ASC_NODE',
  'ARG_OF_PERICENTER',
  'MEAN_ANOMALY',
  'NORAD_CAT_ID',
  'BSTAR',
]

export function parseTleText(text) {
  const lines = String(text || '')
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean)
  const recs = []
  let i = 0
  while (i < lines.length) {
    const a = lines[i]
    const b = lines[i + 1] || ''
    const c = lines[i + 2] || ''
    if (a.startsWith('1 ') && b.startsWith('2 ')) {
      recs.push({ name: `SAT-${recs.length + 1}`, l1: a, l2: b })
      i += 2
      continue
    }
    if (b.startsWith('1 ') && c.startsWith('2 ')) {
      recs.push({ name: a, l1: b, l2: c })
      i += 3
      continue
    }
    i += 1
  }
  return recs
}

export function isOmm(rec) {
  if (!rec || typeof rec !== 'object') return false
  return OMM_REQUIRED.every((k) => rec[k] != null && rec[k] !== '')
}

export function makeSatrec(rec) {
  if (isOmm(rec) && typeof satellite.json2satrec === 'function') {
    try {
      return satellite.json2satrec(rec)
    } catch {
      // fall through to classic TLE
    }
  }
  try {
    if (rec?.l1 && rec?.l2) return satellite.twoline2satrec(rec.l1, rec.l2)
  } catch {
    return null
  }
  return null
}

export function propagateSatrec(satrec, date = new Date()) {
  if (!satrec) return null
  const pv = satellite.propagate(satrec, date)
  if (!pv?.position) return null
  const gmst = satellite.gstime(date)
  const gd = satellite.eciToGeodetic(pv.position, gmst)
  const lat = satellite.degreesLat(gd.latitude)
  const lon = satellite.degreesLong(gd.longitude)
  const alt_m = gd.height * 1000
  if (!Number.isFinite(lat) || !Number.isFinite(lon) || !Number.isFinite(alt_m)) return null
  return { lat, lon, alt_m }
}

export function noradId(rec) {
  if (rec?.NORAD_CAT_ID != null && rec.NORAD_CAT_ID !== '') return String(rec.NORAD_CAT_ID)
  if (rec?.norad != null && rec.norad !== '') return String(rec.norad)
  const m = String(rec?.l1 || '').match(/^1\s+(\d+)/)
  return m ? m[1] : rec?.name
}
