/** Each Data Layers click restages the cockpit — fly, swap layers, change the picture. */

import { EARTH_R, haversineKm } from './geo.js'

export const AIR_HUBS = [
  { id: 'LAX', lat: 33.9425, lon: -118.4081, alt: 12_000 },
  { id: 'AUS', lat: 30.1945, lon: -97.6699, alt: 12_000 },
  { id: 'JFK', lat: 40.6413, lon: -73.7781, alt: 12_000 },
  { id: 'LHR', lat: 51.47, lon: -0.4543, alt: 12_000 },
  { id: 'FRA', lat: 50.0379, lon: 8.5622, alt: 12_000 },
  { id: 'DXB', lat: 25.2532, lon: 55.3657, alt: 12_000 },
  { id: 'NRT', lat: 35.772, lon: 140.3929, alt: 12_000 },
  { id: 'SIN', lat: 1.3644, lon: 103.9915, alt: 12_000 },
]

export const SEA_HUBS = [
  { id: 'LA-LB', lat: 33.73, lon: -118.26, alt: 12_000 },
  { id: 'DOVER', lat: 51.05, lon: 1.4, alt: 12_000 },
  { id: 'SIN', lat: 1.26, lon: 103.85, alt: 12_000 },
  { id: 'ROTTERDAM', lat: 51.95, lon: 4.13, alt: 12_000 },
]

export const LAYER_SCENES = {
  flights: {
    layers: ['flights', 'military'],
    alt: 12_000,
    hubs: 'air',
    speak: 'Live flights. Tracking airborne contacts.',
    hint: 'Live contacts',
  },
  military: {
    layers: ['military'],
    alt: 36_000,
    hubs: 'air',
    pick: 'mil',
    speak: 'Military tracks. Close chase.',
    hint: 'Military picture',
  },
  ships: {
    layers: ['ships'],
    alt: 48_000,
    hubs: 'sea',
    speak: 'Surface vessels on the water.',
    hint: 'Approaches',
  },
  sats: {
    layers: ['sats'],
    lat: 18,
    lon: -24,
    alt: 8_200_000,
    speak: 'Orbital catalog. Earth with moving birds. Click one or hit ISS.',
    hint: 'Satellites · Earth orbit',
    pick: 'iss',
  },
  missions: {
    layers: ['sats', 'missions'],
    lat: 28.5721,
    lon: -80.648,
    alt: 16_000,
    speak: 'Space missions. Kennedy photoreal — still of the pad, or replay a reconstructed ascent.',
    hint: 'Space missions · Kennedy',
    pick: 'pad',
  },
  quakes: {
    layers: ['quakes'],
    lat: 2,
    lon: 128,
    alt: 8_200_000,
    pick: 'quake',
    speak: 'Earthquake — Earth in frame over the epicenter.',
    hint: 'Quake · Earth',
  },
  fires: {
    layers: ['fires'],
    lat: 37.2,
    lon: -119.4,
    alt: 8_200_000,
    pick: 'fire',
    speak: 'Fire — Earth in frame over the burn.',
    hint: 'Fire · Earth',
  },
  storms: {
    layers: ['storms'],
    lat: 22.0,
    lon: -60.0,
    alt: 8_200_000,
    pick: 'storm',
    speak: 'Storms and hazards — Earth in frame, EONET + NWS.',
    hint: 'Storms · Earth',
  },
  traffic: {
    layers: ['traffic'],
    alt: 18_000,
    hubs: 'air',
    speak: 'Street traffic. Simulated flow, labeled SIM.',
    hint: 'Street traffic',
  },
  cctv: {
    layers: ['cctv', 'traffic'],
    alt: 14_000,
    speak: 'Public cameras over the city. Click a cam for the still, drag to look around.',
    hint: 'Public cameras',
    city: true,
  },
  sites: {
    layers: ['sites'],
    alt: 8_500,
    hubs: 'air',
    pick: 'site',
    speak: 'Sites — bases, dams, and data centers on photoreal terrain.',
    hint: 'Sites · photoreal',
  },
  radio: {
    layers: ['radio'],
    alt: 9_000,
    hubs: 'air',
    pick: 'radio',
    speak: 'Public radio. Named stations on the map — tune one.',
    hint: 'Radio · photoreal',
  },
}

export function nearestHub(lat, lon, hubs, maxKm = 650) {
  let best = hubs[0]
  let bestD = Infinity
  for (const h of hubs) {
    const d = haversineKm(lat, lon, h.lat, h.lon)
    if (d < bestD) {
      best = h
      bestD = d
    }
  }
  return { hub: best, km: bestD, inRange: bestD <= maxKm }
}

/** Poll this instead of empty ocean so the rail never says 0 while traffic exists. */
export function coverageOrigin(lat, lon, hubs, maxKm = 650) {
  const { hub, inRange } = nearestHub(lat, lon, hubs, maxKm)
  return inRange ? { lat, lon, hub: null } : { lat: hub.lat, lon: hub.lon, hub }
}

/** How many 3D glyphs the city view can draw without locking the tab. */
export function glyphBudget(altM) {
  const alt = Number(altM) || 0
  if (alt > 400_000) return 28
  if (alt < 4_000) return 48
  if (alt < 20_000) return 32
  return 36
}

/** Globe glyphs/roster follow the focused rail — leftover ships must not ride along. */
export function contactMatchesFocus(focused, c) {
  if (!focused || !c) return true
  return layerContactPriority(focused, c) === 0
}

/** Detection callouts only for moving air/sea pictures — not fires/CCTV/storms. */
export function detectionForLayer(k) {
  return k === 'flights' || k === 'military' || k === 'ships'
}

/** Google 3D tiles only below this — orbital views page the planet and freeze the tab. */
export const TILES_MAX_ALT_M = 80_000

export function tilesAltitudeOk(altM) {
  return (Number(altM) || 0) < TILES_MAX_ALT_M
}

/** City-scale satstill zoom so a 2K marble is not one black texel. */
export function satstillZoomForAlt(altM) {
  const alt = Number(altM) || 12_000
  if (alt < 8_000) return 15
  if (alt < 16_000) return 13
  if (alt < 30_000) return 12
  if (alt < 50_000) return 11
  if (alt < 90_000) return 9
  return 7
}

/** Drape a satellite still when the marble would read as a flat color.
 * Stay under the tiles cap so fires/storms (180–220 km) show a globe, not a disc. */
export function lookPatchOk(altM) {
  const alt = Number(altM) || 0
  return alt > 400 && alt < TILES_MAX_ALT_M
}

/**
 * World meters for a globe glyph. City cameras must not scale sats/missions
 * into 2–80 km cubes or fires/storms into planet-eating discs.
 */
export function glyphScaleM(kind, dist, selected = false) {
  const d = Math.max(40, Number(dist) || 0)
  let target
  if (kind === 'mission') target = Math.min(80, Math.max(22, d * 0.002))
  else if (kind === 'sat') target = Math.min(2_400, Math.max(80, d * 0.0008))
  else if (kind === 'flight' || kind === 'mil') target = Math.min(9_000, Math.max(90, d * 0.011))
  else if (kind === 'ship') target = Math.min(4_500, Math.max(70, d * 0.01))
  else if (
    kind === 'quake' ||
    kind === 'fire' ||
    kind === 'storm' ||
    kind === 'volcano' ||
    kind === 'flood' ||
    kind === 'dust' ||
    kind === 'alert'
  ) {
    target = Math.min(22_000, Math.max(80, d * 0.0032))
  } else target = Math.min(280, Math.max(18, d * 0.004))
  return target * (selected ? 1.28 : 1)
}

/** Prefer the focused layer's contacts so the picture matches the rail. */
export function layerContactPriority(focused, c) {
  if (!focused || !c) return 1
  switch (focused) {
    case 'flights':
      return c.kind === 'flight' && !c.military ? 0 : 1
    case 'military':
      return c.military ? 0 : 1
    case 'ships':
      return c.kind === 'ship' ? 0 : 1
    case 'sats':
      return c.kind === 'sat' ? 0 : 1
    case 'missions':
      return c.kind === 'mission' ? 0 : 1
    case 'quakes':
      return c.kind === 'quake' ? 0 : 1
    case 'fires':
      return c.kind === 'fire' ? 0 : 1
    case 'storms':
      return ['storm', 'volcano', 'flood', 'dust', 'alert'].includes(c.kind) ? 0 : 1
    case 'traffic':
      return c.kind === 'traffic' ? 0 : 1
    case 'sites':
      return ['site', 'installation', 'dam', 'datacenter'].includes(c.kind) ? 0 : 1
    case 'radio':
      return c.kind === 'radio' ? 0 : 1
    case 'cctv':
      return c.kind === 'cctv' ? 0 : 1
    default:
      return 1
  }
}

export const SCENE_STILL_LAYERS = new Set(['quakes', 'fires', 'storms', 'sites', 'radio', 'missions', 'sats'])

export function sceneStillKind(focusedLayer) {
  if (focusedLayer === 'quakes') return 'quake'
  if (focusedLayer === 'fires') return 'fire'
  if (focusedLayer === 'storms') return 'storm'
  if (focusedLayer === 'sites') return 'site'
  if (focusedLayer === 'radio') return 'radio'
  if (focusedLayer === 'missions') return 'mission'
  if (focusedLayer === 'sats') return 'sat'
  return null
}

export function nearbyRadiusKm(altM, kind) {
  if (
    kind === 'sat' ||
    kind === 'sats' ||
    kind === 'mission' ||
    kind === 'missions' ||
    kind === 'mil' ||
    kind === 'military'
  ) {
    return 20_000
  }
  if (
    kind === 'quake' ||
    kind === 'quakes' ||
    kind === 'fire' ||
    kind === 'fires' ||
    kind === 'storm' ||
    kind === 'storms' ||
    kind === 'volcano' ||
    kind === 'flood' ||
    kind === 'dust' ||
    kind === 'alert'
  ) {
    return 12_000
  }
  if (kind === 'site' || kind === 'sites' || kind === 'radio') return 400
  const scaled = Math.max(160, Math.min(2_800, (altM || 12_000) * 0.014))
  return scaled
}

export function findIss(list) {
  return (list || []).find((s) => /iss|zarya/i.test(s.callsign || s.id)) || null
}

export function pickSite(list) {
  const rank = { installation: 0, datacenter: 1, dam: 2, site: 3 }
  const named = (s) => {
    const n = String(s || '')
    return n && !/^(node|way|relation)-/i.test(n)
  }
  return (
    [...(list || [])].sort((a, b) => {
      const d = (rank[a.kind] ?? 9) - (rank[b.kind] ?? 9)
      if (d) return d
      return Number(named(b.callsign)) - Number(named(a.callsign))
    })[0] || null
  )
}

export function pickRadio(list) {
  return (list || []).find((r) => r.url) || (list || [])[0] || null
}

/** Prefer a live tropical storm, then any EONET event, then an NWS alert. */
export function pickStorm(list) {
  const rank = { storm: 0, volcano: 1, flood: 2, dust: 3, alert: 4 }
  return (
    [...(list || [])].sort((a, b) => {
      const d = (rank[a.kind] ?? 9) - (rank[b.kind] ?? 9)
      if (d) return d
      return (Number(b.mag) || 0) - (Number(a.mag) || 0)
    })[0] || null
  )
}

/** Camera height that actually shows the craft, not a 90 km square. */
export function chaseAlt(contact) {
  const alt = Math.abs(contact?.alt_m || 0)
  if (contact?.kind === 'sat') return LAYER_SCENES.sats.alt
  if (contact?.kind === 'mission') return contact?.reconstructed ? Math.max(80_000, alt * 0.35 + 48_000) : 16_000
  if (contact?.kind === 'ship') return 28_000
  if (contact?.kind === 'flight' || contact?.military) return Math.max(8_000, alt + 6_000)
  if (
    contact?.kind === 'quake' ||
    contact?.kind === 'fire' ||
    contact?.kind === 'storm' ||
    contact?.kind === 'volcano' ||
    contact?.kind === 'flood' ||
    contact?.kind === 'dust' ||
    contact?.kind === 'alert'
  ) {
    return 420_000
  }
  if (
    contact?.kind === 'site' ||
    contact?.kind === 'installation' ||
    contact?.kind === 'dam' ||
    contact?.kind === 'datacenter'
  ) {
    return 7_400
  }
  if (contact?.kind === 'radio') return 8_200
  return 18_000
}

/**
 * After Satellites the target is Earth center, so OrbitControls.minDistance
 * becomes EARTH_R+90. A city snap is only a few km from the surface target —
 * update() then clamps the camera back to a globe. City dives must drop this first.
 */
export function flyMinDistance(altM) {
  return (Number(altM) || 0) > 1_200_000 ? EARTH_R + 90 : 90
}

/** Side throw so city dives are oblique — facades, not a nadir postage stamp. */
export function cinematicThrowM(altM) {
  const alt = Number(altM) || 12_000
  if (alt > 1_200_000) return 0
  return Math.min(alt * 1.15, alt * 0.55 + 9_000)
}

/**
 * Layer restage throw — enough to read streets, not a 12 km side-slip into a field.
 * The demo city look is an urban overview you can drag and zoom out of.
 */
export function overviewThrowM(altM) {
  const alt = Number(altM) || 12_000
  if (alt > 1_200_000) return 0
  return Math.min(alt * 0.28, 3_200)
}

/**
 * City overview for CCTV — urban-core altitude from the demo (look around, then zoom out).
 * Camera clicks must remount at this height. Do not dive under ~8 km on a layer click.
 */
export const CCTV_SCENE_ALT_M = 14_000

/** Layer rail restages a free look. Orbit/cockpit only after the user picks a contact or hits ORB/C. */
export function layerClickCamMode() {
  return 'free'
}

export function cityFov(altM, cockpit = false) {
  if (cockpit) return 68
  const alt = Number(altM) || 12_000
  if (alt < 4_000) return 62
  if (alt < 80_000) return 56
  return 48
}

export function siteFlyArrived(cameraPos, goalPos, altM) {
  if (!cameraPos || !goalPos) return false
  const dx = cameraPos.x - goalPos.x
  const dy = cameraPos.y - goalPos.y
  const dz = cameraPos.z - goalPos.z
  const dist = Math.hypot(dx, dy, dz)
  const alt = Number(altM) || 12_000
  return dist < Math.max(250, alt * 0.06)
}

const CITY_LAYERS = new Set([
  'flights',
  'military',
  'ships',
  'traffic',
  'cctv',
  'sites',
  'radio',
])
const SPACE_LAYERS = new Set(['sats', 'missions', 'quakes', 'fires', 'storms'])

/** Flights / sites / radio need street-to-city altitude. Orbit hashes must not keep them in space. */
/** Programmatic dives must beat a leaked OrbitControls drag. */
export function shouldApplySiteFly(siteFly, dragging) {
  if (!siteFly) return false
  if (siteFly.snap) return true
  return !dragging
}

export function isCityPicture(layers) {
  const rows = Array.isArray(layers) ? layers : []
  if (rows.some((k) => SPACE_LAYERS.has(k))) return false
  return rows.some((k) => CITY_LAYERS.has(k))
}

export function resolveSceneLook(scene, look, picks = {}) {
  if (!scene) return null
  let lat = look?.lat ?? 0
  let lon = look?.lon ?? 0
  let alt = scene.alt ?? look?.alt ?? 20_000
  let hint = scene.hint || scene.speak
  if (!scene.stay && scene.lat != null && scene.lon != null) {
    lat = scene.lat
    lon = scene.lon
  }
  if (scene.hubs === 'air') {
    const { hub, inRange } = nearestHub(lat, lon, AIR_HUBS, 650)
    if (!inRange) {
      lat = hub.lat
      lon = hub.lon
      hint = `${scene.hint} · ${hub.id}`
    }
    alt = scene.alt
  }
  if (scene.hubs === 'sea') {
    const { hub, inRange } = nearestHub(lat, lon, SEA_HUBS, 500)
    if (!inRange) {
      lat = hub.lat
      lon = hub.lon
      hint = `${scene.hint} · ${hub.id}`
    } else {
      lat = hub.lat
      lon = hub.lon
    }
    alt = scene.alt
  }
  // Contact picks update the right card via onPick. They must not restage the
  // camera — the 00:27 sat vanish and fire/storm punches were a second flyTo.
  return { lat, lon, alt, hint, speak: scene.speak }
}
