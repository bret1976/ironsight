/** Client-side intel fetch, normalize, and flight interpolation. */

import { haversineKm, interpolateContact } from './geo.js'
import { isOmm, makeSatrec, noradId, parseTleText, propagateSatrec } from './tle.js'

export function isAirborne(c) {
  if (!c || c.on_ground) return false
  if (
    c.kind === 'ship' ||
    c.kind === 'traffic' ||
    c.kind === 'quake' ||
    c.kind === 'fire' ||
    c.kind === 'storm' ||
    c.kind === 'volcano' ||
    c.kind === 'flood' ||
    c.kind === 'dust' ||
    c.kind === 'alert'
  ) {
    return false
  }
  const alt = Number(c.alt_m) || 0
  const gs = Number(c.gs_kts) || 0
  return alt >= 400 && gs >= 40
}

export function pickAirborne(list, lat, lon) {
  const air = (list || []).filter(isAirborne)
  if (!air.length) return null
  return [...air].sort(
    (a, b) => haversineKm(lat, lon, a.lat, a.lon) - haversineKm(lat, lon, b.lat, b.lon)
  )[0]
}

export async function fetchJson(url, { timeoutMs = 14000 } = {}) {
  const ctrl = new AbortController()
  const t = setTimeout(() => ctrl.abort(), timeoutMs)
  try {
    const res = await fetch(url, { signal: ctrl.signal })
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    return await res.json()
  } finally {
    clearTimeout(t)
  }
}

export function ageLabel(freshnessMs) {
  if (freshnessMs == null) return '—'
  const s = Math.max(0, (Date.now() - Number(freshnessMs)) / 1000)
  if (s < 5) return 'now'
  if (s < 60) return `${Math.round(s)}s`
  if (s < 3600) return `${Math.round(s / 60)}m`
  return `${Math.round(s / 60 / 60)}h`
}

export function createFlightStore() {
  /** @type {Map<string, {prev: any, next: any}>} */
  const map = new Map()
  return {
    ingest(contacts, now = Date.now()) {
      const seen = new Set()
      for (const c of contacts || []) {
        if (!c?.id) continue
        seen.add(c.id)
        const row = map.get(c.id)
        const snap = { ...c, t: now }
        if (!row) map.set(c.id, { prev: null, next: snap })
        else {
          row.prev = row.next
          row.next = snap
        }
      }
      for (const id of [...map.keys()]) {
        if (!seen.has(id)) {
          const row = map.get(id)
          if (!row?.next || now - row.next.t > 90_000) map.delete(id)
        }
      }
    },
    sample(now = Date.now()) {
      const out = []
      for (const row of map.values()) {
        out.push(interpolateContact(row.prev, row.next, now))
      }
      return out
    },
    size() {
      return map.size
    },
  }
}

export function tlesToSatellites(tles, date = new Date()) {
  const out = []
  for (const rec of tles || []) {
    const satrec = makeSatrec(rec)
    const pos = propagateSatrec(satrec, date)
    if (!pos) continue
    const nid = noradId(rec)
    out.push({
      id: `sat-${nid}`,
      kind: 'sat',
      callsign: rec.name || rec.OBJECT_NAME || nid,
      norad: nid,
      satrec,
      ...pos,
      heading: 0,
      format: isOmm(rec) ? 'omm' : rec.format || 'tle',
      source: rec.source || 'celestrak',
    })
  }
  return out
}

export function updateSatellitePositions(sats, date = new Date()) {
  for (const s of sats) {
    const pos = propagateSatrec(s.satrec, date)
    if (!pos) continue
    s.lat = pos.lat
    s.lon = pos.lon
    s.alt_m = pos.alt_m
  }
  return sats
}

export async function loadFlights(lat, lon, radiusNm = 450) {
  try {
    return await fetchJson(
      `/api/godseye/flights?lat=${encodeURIComponent(lat)}&lon=${encodeURIComponent(lon)}&radius_nm=${encodeURIComponent(radiusNm)}&include_mil=true`
    )
  } catch (e) {
    // Browser fallback if API is down (adsb.lol is CORS-open).
    try {
      const raw = await fetchJson(
        `https://api.adsb.lol/v2/lat/${lat.toFixed(3)}/lon/${lon.toFixed(3)}/dist/${Math.round(radiusNm)}`
      )
      const contacts = (raw.ac || [])
        .filter((a) => a.lat != null && a.lon != null && a.hex)
        .map((a) => ({
          id: `icao-${String(a.hex).toLowerCase()}`,
          kind: 'flight',
          callsign: String(a.flight || a.hex).trim(),
          hex: String(a.hex).toLowerCase(),
          lat: a.lat,
          lon: a.lon,
          alt_m: (Number(a.alt_geom || a.alt_baro) || 0) * 0.3048,
          heading: Number(a.track) || 0,
          gs_kts: Number(a.gs) || 0,
          type: a.t || '',
          source: 'adsb.lol',
        }))
      return {
        kind: 'flights',
        status: contacts.length ? 'live' : 'unavailable',
        source: 'adsb.lol',
        freshness_ms: Date.now(),
        label: contacts.length ? 'LIVE' : 'UNAVAILABLE',
        count: contacts.length,
        contacts,
        errors: [],
        note: 'Direct adsb.lol (API proxy unavailable).',
      }
    } catch (e2) {
      return {
        kind: 'flights',
        status: 'unavailable',
        source: 'adsb.lol|opensky-anonymous',
        freshness_ms: Date.now(),
        label: 'UNAVAILABLE',
        count: 0,
        contacts: [],
        errors: [String(e.message || e), String(e2.message || e2)],
        note: 'Flight feeds unreachable.',
      }
    }
  }
}

export async function loadTles() {
  try {
    const doc = await fetchJson('/api/godseye/omm').catch(() => fetchJson('/api/godseye/tle'))
    const recs = doc.tles || doc.omm || []
    return { ...doc, tles: recs, omm: doc.omm || recs, format: doc.format || 'omm', provider: doc.provider || 'celestrak' }
  } catch {
    try {
      const raw = await fetchJson('https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=json')
      const tles = Array.isArray(raw) ? raw : []
      return {
        kind: 'satellites',
        status: tles.length ? 'live' : 'unavailable',
        source: 'celestrak',
        provider: 'celestrak',
        format: 'omm',
        freshness_ms: Date.now(),
        label: tles.length ? 'LIVE' : 'UNAVAILABLE',
        count: tles.length,
        tles,
        omm: tles,
        errors: [],
        note: 'Direct CelesTrak OMM JSON (API proxy unavailable).',
      }
    } catch (e) {
      try {
        const text = await fetch('https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle').then(
          (r) => r.text()
        )
        const tles = parseTleText(text)
        return {
          kind: 'satellites',
          status: tles.length ? 'live' : 'unavailable',
          source: 'celestrak',
          provider: 'celestrak',
          format: 'tle',
          freshness_ms: Date.now(),
          label: tles.length ? 'LIVE' : 'UNAVAILABLE',
          count: tles.length,
          tles,
          omm: tles,
          errors: [],
          note: 'Direct CelesTrak classic TLE (OMM unavailable).',
        }
      } catch (e2) {
        return {
          kind: 'satellites',
          status: 'unavailable',
          source: 'celestrak',
          provider: 'celestrak',
          format: 'omm',
          freshness_ms: Date.now(),
          label: 'UNAVAILABLE',
          count: 0,
          tles: [],
          omm: [],
          errors: [String(e.message || e), String(e2.message || e2)],
          note: 'CelesTrak unreachable.',
        }
      }
    }
  }
}

export async function loadQuakes() {
  try {
    return await fetchJson('/api/godseye/quakes')
  } catch {
    try {
      const raw = await fetchJson(
        'https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_day.geojson'
      )
      const contacts = (raw.features || []).map((f) => {
        const [lon, lat, depth] = f.geometry?.coordinates || [0, 0, 0]
        return {
          id: `eq-${f.id}`,
          kind: 'quake',
          callsign: f.properties?.place || f.id,
          lat,
          lon,
          alt_m: -(depth || 0) * 1000,
          mag: f.properties?.mag || 0,
          time_ms: f.properties?.time,
          source: 'usgs',
        }
      })
      return {
        kind: 'quakes',
        status: 'live',
        source: 'usgs',
        freshness_ms: raw.metadata?.generated || Date.now(),
        label: 'LIVE',
        count: contacts.length,
        contacts,
        errors: [],
        note: 'Direct USGS (API proxy unavailable).',
      }
    } catch (e) {
      return {
        kind: 'quakes',
        status: 'unavailable',
        source: 'usgs',
        freshness_ms: Date.now(),
        label: 'UNAVAILABLE',
        count: 0,
        contacts: [],
        errors: [String(e.message || e)],
        note: 'USGS unreachable.',
      }
    }
  }
}

function viewportBbox(lat, lon, radiusNm) {
  const dlat = Math.min(4, Math.max(0.2, Number(radiusNm) / 60))
  const clat = Math.max(0.2, Math.cos((Number(lat) * Math.PI) / 180))
  const dlon = Math.min(4, Math.max(0.2, Number(radiusNm) / (60 * clat)))
  return `${(lat - dlat).toFixed(4)},${(lon - dlon).toFixed(4)},${(lat + dlat).toFixed(4)},${(lon + dlon).toFixed(4)}`
}

function contactsFromOpenWaters(raw) {
  const feats = raw?.features || (Array.isArray(raw) ? raw : [])
  const contacts = []
  for (const feat of feats) {
    const props = feat?.properties || {}
    const coords = feat?.geometry?.coordinates
    if (!coords || coords.length < 2) continue
    const mmsi = props.mmsi ?? feat.id
    if (mmsi == null || mmsi === '') continue
    const sog = Number(props.sog) || 0
    const heading = Number(props.heading ?? props.cog) || 0
    contacts.push({
      id: `mmsi-${mmsi}`,
      kind: 'ship',
      callsign: String(props.name || '').trim() || `MMSI ${mmsi}`,
      mmsi: String(mmsi),
      lat: coords[1],
      lon: coords[0],
      alt_m: 0,
      heading,
      gs_kts: sog,
      sog,
      cog: Number(props.cog) || heading,
      source: 'openwaters',
      feed: String(props.source || ''),
      station: String(props.station || ''),
      msg_type: String(props.msg_type || ''),
      ais_kind: String(props.kind || 'vessel'),
      seen: String(props.seen || ''),
    })
    if (contacts.length >= 400) break
  }
  return contacts
}

export async function loadShips(lat = 35, lon = -30, radiusNm = 450, bbox) {
  const box = bbox || viewportBbox(lat, lon, radiusNm)
  try {
    const qs = bbox
      ? `bbox=${encodeURIComponent(bbox)}`
      : `lat=${encodeURIComponent(lat)}&lon=${encodeURIComponent(lon)}&radius_nm=${encodeURIComponent(radiusNm)}`
    return await fetchJson(`/api/godseye/ships?${qs}`)
  } catch (e) {
    try {
      const raw = await fetchJson(`https://ais.openwaters.io/v1/vessels?bbox=${encodeURIComponent(box)}`)
      const contacts = contactsFromOpenWaters(raw)
      return {
        kind: 'ships',
        status: contacts.length ? 'live' : 'empty',
        source: 'openwaters',
        provider: 'openwaters',
        shipsProvider: 'openwaters',
        freshness_ms: Date.now(),
        label: contacts.length ? 'LIVE' : 'EMPTY',
        simulated: false,
        count: contacts.length,
        contacts,
        errors: [],
        note: 'Direct Open Waters AIS (API proxy unavailable).',
      }
    } catch (e2) {
      return {
        kind: 'ships',
        status: 'unavailable',
        source: 'openwaters',
        provider: 'openwaters',
        shipsProvider: 'openwaters',
        freshness_ms: null,
        label: 'UNAVAILABLE',
        simulated: false,
        count: 0,
        contacts: [],
        errors: [String(e.message || e), String(e2.message || e2)],
        note: 'Open Waters AIS unreachable. Honest empty — no invented contacts.',
      }
    }
  }
}

export async function loadConfig() {
  const viteCesium = (import.meta.env?.VITE_CESIUM_ION_TOKEN || '').trim()
  const viteGoogle = (import.meta.env?.VITE_GOOGLE_MAPS_API_KEY || '').trim()
  try {
    const cfg = await fetchJson('/api/godseye/config')
    return {
      ...cfg,
      cesiumIonToken: (cfg.cesiumIonToken || viteCesium || '').trim() || null,
      googleMapsApiKey: (cfg.googleMapsApiKey || viteGoogle || '').trim() || null,
    }
  } catch {
    const cesium = viteCesium || null
    const google = viteGoogle || null
    return {
      tiles: google ? 'google' : cesium ? 'cesium' : 'none',
      cesiumIonToken: cesium,
      googleMapsApiKey: google,
      hasPhotoreal: Boolean(cesium || google),
      hasShips: true,
      shipsProvider: 'openwaters',
    }
  }
}

export async function geocodePlace(q) {
  const query = String(q || '').trim()
  if (query.length < 2) return []
  try {
    const r = await fetchJson(`/api/godseye/geocode?q=${encodeURIComponent(query)}`)
    return r.results || []
  } catch {
    return []
  }
}

export async function loadFires() {
  try {
    return await fetchJson('/api/godseye/fires', { timeoutMs: 22000 })
  } catch (e) {
    return {
      kind: 'fires',
      status: 'unavailable',
      source: 'nasa-firms',
      freshness_ms: Date.now(),
      label: 'UNAVAILABLE',
      count: 0,
      contacts: [],
      errors: [String(e.message || e)],
    }
  }
}

export async function loadStorms() {
  try {
    return await fetchJson('/api/godseye/storms', { timeoutMs: 18000 })
  } catch (e) {
    return {
      kind: 'storms',
      status: 'unavailable',
      source: 'nasa-eonet|nws',
      freshness_ms: Date.now(),
      label: 'UNAVAILABLE',
      count: 0,
      contacts: [],
      errors: [String(e.message || e)],
    }
  }
}

export async function loadSites(lat = 35, lon = -30, radiusKm = 80) {
  try {
    return await fetchJson(
      `/api/godseye/sites?lat=${encodeURIComponent(lat)}&lon=${encodeURIComponent(lon)}&radius_km=${encodeURIComponent(radiusKm)}`,
      { timeoutMs: 26000 }
    )
  } catch (e) {
    return {
      kind: 'sites',
      status: 'unavailable',
      source: 'osm-overpass',
      freshness_ms: Date.now(),
      label: 'UNAVAILABLE',
      count: 0,
      contacts: [],
      errors: [String(e.message || e)],
    }
  }
}

export async function loadRadio(lat = 35, lon = -30) {
  try {
    return await fetchJson(
      `/api/godseye/radio?lat=${encodeURIComponent(lat)}&lon=${encodeURIComponent(lon)}`
    )
  } catch (e) {
    return {
      kind: 'radio',
      status: 'unavailable',
      source: 'radio-browser',
      freshness_ms: Date.now(),
      label: 'UNAVAILABLE',
      count: 0,
      contacts: [],
      errors: [String(e.message || e)],
    }
  }
}

export async function loadMissions() {
  try {
    return await fetchJson('/api/godseye/missions', { timeoutMs: 18000 })
  } catch (e) {
    return {
      kind: 'missions',
      status: 'unavailable',
      source: 'launch-library-2',
      freshness_ms: Date.now(),
      label: 'UNAVAILABLE',
      count: 0,
      contacts: [],
      errors: [String(e.message || e)],
    }
  }
}

export function isPlayableVideo(url) {
  const path = String(url || '').split('?')[0].toLowerCase()
  return path.endsWith('.mp4') || path.endsWith('.webm') || path.endsWith('.ogg')
}

export function isGroundIntel(c) {
  return [
    'quake',
    'fire',
    'radio',
    'dam',
    'installation',
    'datacenter',
    'site',
    'mission',
    'storm',
    'volcano',
    'flood',
    'dust',
    'alert',
  ].includes(c?.kind)
}

export function satStillZoom(kind) {
  if (kind === 'fire') return 14
  if (kind === 'quake' || kind === 'storm' || kind === 'volcano' || kind === 'flood' || kind === 'dust' || kind === 'alert') {
    return 13
  }
  if (kind === 'radio') return 16
  if (kind === 'dam' || kind === 'installation' || kind === 'datacenter' || kind === 'site') return 16
  if (kind === 'mission') return 15
  if (kind === 'sat') return 5
  return 15
}

/** Moving GPS-like birds so Satellites always has a picture if TLEs are late. */
export function orbitalConstellation(now = Date.now()) {
  const sats = []
  const planes = 6
  const per = 4
  for (let p = 0; p < planes; p++) {
    const raan = (p / planes) * Math.PI * 2
    for (let i = 0; i < per; i++) {
      const mean = (i / per) * Math.PI * 2 + now / 5500 + p * 0.31
      const inc = 0.96
      const lat = Math.asin(Math.sin(inc) * Math.sin(mean)) * (180 / Math.PI)
      let lon =
        ((raan + Math.atan2(Math.cos(inc) * Math.sin(mean), Math.cos(mean))) * 180) / Math.PI + now / 140000
      lon = ((lon + 540) % 360) - 180
      sats.push({
        id: `nav-${p}${i}`,
        kind: 'sat',
        callsign: `NAVSTAR ${p * 4 + i + 1}`,
        lat,
        lon,
        alt_m: 20_200_000,
        heading: 0,
        simulated: true,
        source: 'constellation',
      })
    }
  }
  return sats
}

export function satelliteStillUrl(lat, lon, _key, zoom = 15) {
  if (lat == null || lon == null || !Number.isFinite(Number(lat)) || !Number.isFinite(Number(lon))) return ''
  const z = Math.max(4, Math.min(19, Number(zoom) || 15))
  return `/api/godseye/satstill?lat=${Number(lat).toFixed(5)}&lon=${Number(lon).toFixed(5)}&z=${z}`
}

export function cctvStillSrc(url) {
  const raw = String(url || '').trim()
  if (!raw) return ''
  if (raw.startsWith('/api/godseye/cctv/media')) return raw
  return `/api/godseye/cctv/media?u=${encodeURIComponent(raw)}`
}

export async function loadCctv() {
  try {
    return await fetchJson('/api/godseye/cctv', { timeoutMs: 45000 })
  } catch (e) {
    return {
      kind: 'cctv',
      status: 'loading',
      source: 'tfl|austin-open-data|caltrans',
      freshness_ms: Date.now(),
      label: '…',
      count: null,
      contacts: [],
      errors: [String(e.message || e)],
    }
  }
}

/** Keyless street-level traffic — labeled simulation, not TomTom. */
export function simulateTraffic(lat, lon, now = Date.now()) {
  const contacts = []
  const n = 86
  for (let i = 0; i < n; i++) {
    const seed = i * 19.17
    const ring = 0.012 + (i % 9) * 0.0065
    const baseHdg = [0, 45, 90, 135, 180, 225, 270, 315][i % 8]
    const t = ((now / 52000 + seed * 0.01) % 1)
    const dlat = Math.cos((baseHdg * Math.PI) / 180) * 0.016 * (t - 0.5)
    const dlon =
      (Math.sin((baseHdg * Math.PI) / 180) * 0.016 * (t - 0.5)) /
      Math.max(0.2, Math.cos((lat * Math.PI) / 180))
    contacts.push({
      id: `traf-${i}`,
      kind: 'traffic',
      callsign: `SIM-${String(i + 1).padStart(3, '0')}`,
      lat: lat + Math.sin(seed) * ring + dlat,
      lon: lon + Math.cos(seed * 1.37) * ring + dlon,
      alt_m: 2,
      heading: baseHdg,
      gs_kts: 18 + (i % 11) * 3,
      simulated: true,
      source: 'traffic-sim',
    })
  }
  return {
    kind: 'traffic',
    status: 'simulated',
    source: 'traffic-sim',
    freshness_ms: now,
    label: 'SIM',
    simulated: true,
    count: contacts.length,
    contacts,
    errors: [],
    note: 'Approximate street-level flow. Labeled simulation — no TomTom key.',
  }
}

export async function runCommand(text, look = {}) {
  const res = await fetch('/api/godseye/command', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      text,
      lat: look.lat,
      lon: look.lon,
      alt: look.alt,
    }),
  })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}
