/** Shareable GodsEye view state in the URL hash. */

export const DEFAULT_LAYERS = ['flights', 'military']
export const ALL_LAYERS = [
  'flights',
  'military',
  'sats',
  'quakes',
  'ships',
  'fires',
  'storms',
  'sites',
  'radio',
  'missions',
  'cctv',
  'traffic',
]
export const SENSORS = ['rgb', 'crt', 'nvg', 'flir', 'ironbell', 'noir', 'snow']
export const CAM_MODES = ['free', 'orbit', 'cockpit']

export function parseGodsEyeState(hash = '', search = '') {
  const out = {
    lat: null,
    lon: null,
    alt: null,
    layers: [...DEFAULT_LAYERS],
    track: null,
    sensor: 'rgb',
    cam: 'free',
  }

  const q = new URLSearchParams(search.startsWith('?') ? search.slice(1) : search)
  if (q.get('godsLat')) out.lat = Number(q.get('godsLat'))
  if (q.get('godsLon')) out.lon = Number(q.get('godsLon'))
  if (q.get('godsAlt')) out.alt = Number(q.get('godsAlt'))
  if (q.get('godsLayers')) out.layers = splitLayers(q.get('godsLayers'))
  if (q.get('godsTrack')) out.track = q.get('godsTrack')
  if (q.get('godsSensor') && SENSORS.includes(q.get('godsSensor'))) out.sensor = q.get('godsSensor')
  if (q.get('godsCam') && CAM_MODES.includes(q.get('godsCam'))) out.cam = q.get('godsCam')

  const raw = String(hash || '')
  const idx = raw.indexOf('gods=')
  if (idx === -1) return sanitize(out)
  const body = decodeURIComponent(raw.slice(idx + 5).split('&')[0])
  const parts = body.split(',')
  if (parts[0] !== '' && !Number.isNaN(Number(parts[0]))) out.lat = Number(parts[0])
  if (parts[1] !== '' && !Number.isNaN(Number(parts[1]))) out.lon = Number(parts[1])
  if (parts[2] !== '' && !Number.isNaN(Number(parts[2]))) out.alt = Number(parts[2])
  if (parts[3]) out.layers = splitLayers(parts[3])
  if (parts[4] && parts[4] !== '-') out.track = parts[4]
  if (parts[5] && SENSORS.includes(parts[5])) out.sensor = parts[5]
  if (parts[6] && CAM_MODES.includes(parts[6])) out.cam = parts[6]
  return sanitize(out)
}

export function serializeGodsEyeState(state) {
  const lat = state.lat == null || Number.isNaN(state.lat) ? '' : Number(state.lat).toFixed(3)
  const lon = state.lon == null || Number.isNaN(state.lon) ? '' : Number(state.lon).toFixed(3)
  const alt = state.alt == null || Number.isNaN(state.alt) ? '' : Math.round(state.alt)
  const layers = (state.layers || DEFAULT_LAYERS).filter((l) => ALL_LAYERS.includes(l)).join('+')
  const track = state.track || '-'
  const sensor = SENSORS.includes(state.sensor) ? state.sensor : 'rgb'
  const cam = CAM_MODES.includes(state.cam) ? state.cam : 'free'
  return `#gods=${lat},${lon},${alt},${layers},${track},${sensor},${cam}`
}

export function splitLayers(s) {
  const parts = String(s || '')
    .split(/[+| ]/)
    .map((x) => x.trim())
    .filter((x) => ALL_LAYERS.includes(x))
  return parts.length ? parts : [...DEFAULT_LAYERS]
}

function sanitize(s) {
  if (s.lat != null && (Number.isNaN(s.lat) || s.lat < -90 || s.lat > 90)) s.lat = null
  if (s.lon != null && (Number.isNaN(s.lon) || s.lon < -180 || s.lon > 180)) s.lon = null
  if (s.alt != null && (Number.isNaN(s.alt) || s.alt < 0)) s.alt = null
  if (!Array.isArray(s.layers) || !s.layers.length) s.layers = [...DEFAULT_LAYERS]
  if (!CAM_MODES.includes(s.cam)) s.cam = 'free'
  return s
}
