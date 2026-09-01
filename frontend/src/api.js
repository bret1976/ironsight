const BASE = '/api'

function authHeaders() {
  const token = localStorage.getItem('ironsight_token')
  const h = {}
  if (token) h['Authorization'] = `Bearer ${token}`
  return h
}

async function req(path, opts = {}) {
  const headers = { ...authHeaders(), ...(opts.headers || {}) }
  const res = await fetch(`${BASE}${path}`, { ...opts, headers })
  if (!res.ok) {
    let detail = res.statusText
    try {
      const j = await res.json()
      detail = j.detail || j.message || JSON.stringify(j)
    } catch {}
    const err = new Error(detail)
    err.status = res.status
    throw err
  }
  if (res.status === 204) return null
  return res.json()
}

export const api = {
  health: () => req('/health'),
  plans: () => req('/plans'),
  register: (email, password, name) =>
    req('/auth/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password, name }),
    }),
  login: (email, password) =>
    req('/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    }),
  me: () => req('/auth/me'),
  forgotPassword: (email) =>
    req('/auth/forgot-password', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email }),
    }),
  resetPassword: (token, password) =>
    req('/auth/reset-password', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token, password }),
    }),
  changePassword: (old_password, new_password) =>
    req('/auth/change-password', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ old_password, new_password }),
    }),
  deleteAccount: () => req('/auth/account', { method: 'DELETE' }),
  checkout: (plan) =>
    req('/billing/checkout', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ plan }),
    }),
  portal: () => req('/billing/portal', { method: 'POST' }),
  devSetPlan: (plan) =>
    req('/billing/dev-set-plan', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ plan }),
    }),
  openDemo: () => req('/onboarding/demo-session', { method: 'POST' }),
  orgs: () => req('/orgs'),
  getOrg: (orgId) => req(`/orgs/${orgId}`),
  createOrg: (name) =>
    req('/orgs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    }),
  inviteOrg: (orgId, email, role = 'member') =>
    req(`/orgs/${orgId}/invites`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, role }),
    }),
  acceptInvite: (token) =>
    req('/orgs/accept-invite', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token }),
    }),
  support: (subject, body, email = '') =>
    req('/support', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ subject, body, email }),
    }),
  sla: () => req('/sla'),
  legal: (doc) => fetch(`/api/legal/${doc}`).then((r) => r.text()),
  reviewQueue: (id) => req(`/sessions/${id}/review-queue`),
  reviewShot: (id, shotId, classification, note = '') =>
    req(`/sessions/${id}/review/${shotId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ classification, note }),
    }),
  coachLock: (id, force = false) =>
    req(`/sessions/${id}/coach-lock`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ force }),
    }),
  coachUnlock: (id) => req(`/sessions/${id}/coach-unlock`, { method: 'POST' }),
  quality: (id) => req(`/sessions/${id}/quality`),
  orgDashboard: (orgId) => req(`/orgs/${orgId}/dashboard`),
  assignSession: (orgId, sessionId, assignee_id, role_hint = 'coach') =>
    req(`/orgs/${orgId}/sessions/${sessionId}/assign`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ assignee_id, role_hint }),
    }),
  createLane: (orgId, name, notes = '') =>
    req(`/orgs/${orgId}/lanes`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, notes }),
    }),
  bookLane: (orgId, body) =>
    req(`/orgs/${orgId}/bookings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  orgSla: (orgId) => req(`/orgs/${orgId}/sla`),
  acceptSla: (orgId) => req(`/orgs/${orgId}/sla/accept`, { method: 'POST' }),
  magicLink: (email) =>
    req('/auth/magic-link', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email }),
    }),
  magicConsume: (token) =>
    req('/auth/magic-link/consume', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token }),
    }),
  listSessions: () => req('/sessions'),
  createSession: (name) =>
    req('/sessions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    }),
  getSession: (id) => req(`/sessions/${id}`),
  deleteSession: (id) => req(`/sessions/${id}`, { method: 'DELETE' }),
  upload: async (id, { shooter, observer }) => {
    const fd = new FormData()
    if (shooter) fd.append('shooter', shooter)
    if (observer) fd.append('observer', observer)
    return req(`/sessions/${id}/upload`, { method: 'POST', body: fd })
  },
  process: (id) => req(`/sessions/${id}/process`, { method: 'POST' }),
  pointcloudUrl: (id) => `${BASE}/sessions/${id}/pointcloud`,
  splatUrl: (id) => `${BASE}/sessions/${id}/splat`,
  retrainGsplat: (id, opts = {}) => {
    const q = new URLSearchParams()
    if (opts.iters) q.set('iters', String(opts.iters))
    if (opts.quality) q.set('quality', opts.quality)
    const qs = q.toString()
    return req(`/sessions/${id}/gsplat${qs ? `?${qs}` : ''}`, { method: 'POST' })
  },
  mediaUrl: (id, kind, name) => `${BASE}/sessions/${id}/media/${kind}/${name}`,
  tracksAt: (id, t, camera = 'primary') =>
    req(`/sessions/${id}/tracks?t=${encodeURIComponent(t)}&camera=${encodeURIComponent(camera)}`),
  tracks: (id) => req(`/sessions/${id}/tracks`),
  rebuildTracks: (id) => req(`/sessions/${id}/tracks/rebuild`, { method: 'POST' }),
  radarUrl: (id) => `${BASE}/sessions/${id}/radar`,
  trainYolo: (epochs = 30) =>
    req(`/models/yolo/train?epochs=${epochs}`, { method: 'POST' }),
  cleanDualcam: (id) => req(`/sessions/${id}/dualcam/clean`, { method: 'POST' }),
  poseUrl: (id) => `${BASE}/sessions/${id}/pose`,
  wrap: (id) => req(`/sessions/${id}/wrap`),
  wrapExportUrl: (id, fmt = 'html') => {
    const token = localStorage.getItem('ironsight_token')
    const q = token ? `?token=${encodeURIComponent(token)}&fmt=${fmt}` : `?fmt=${fmt}`
    return `${BASE}/sessions/${id}/wrap/export${q}`
  },
  recomputeStats: (id) => req(`/sessions/${id}/stats/recompute`, { method: 'POST' }),
  annotate: (id, body) =>
    req(`/sessions/${id}/annotate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  runMappers: (id, includeHigh = false) =>
    req(`/sessions/${id}/mappers/run?include_high=${includeHigh ? 'true' : 'false'}`, {
      method: 'POST',
    }),
  mappers: (id) => req(`/sessions/${id}/mappers`),
  jobs: () => req('/jobs'),
  godseyeConfig: () => req('/godseye/config'),
  godseyeFlights: (lat, lon, radiusNm = 450) =>
    req(`/godseye/flights?lat=${encodeURIComponent(lat)}&lon=${encodeURIComponent(lon)}&radius_nm=${encodeURIComponent(radiusNm)}`),
  godseyeTle: () => req('/godseye/tle'),
  godseyeOmm: () => req('/godseye/omm'),
  godseyeQuakes: () => req('/godseye/quakes'),
  godseyeShips: (lat = 35, lon = -30, radiusNm = 450, bbox) =>
    req(
      bbox
        ? `/godseye/ships?bbox=${encodeURIComponent(bbox)}`
        : `/godseye/ships?lat=${encodeURIComponent(lat)}&lon=${encodeURIComponent(lon)}&radius_nm=${encodeURIComponent(radiusNm)}`
    ),
  godseyeGeocode: (q) => req(`/godseye/geocode?q=${encodeURIComponent(q)}`),
  godseyeFires: () => req('/godseye/fires'),
  godseyeStorms: () => req('/godseye/storms'),
  godseyeSites: (lat, lon, radiusKm = 80) =>
    req(`/godseye/sites?lat=${encodeURIComponent(lat)}&lon=${encodeURIComponent(lon)}&radius_km=${encodeURIComponent(radiusKm)}`),
  godseyeRadio: (lat, lon) =>
    req(`/godseye/radio?lat=${encodeURIComponent(lat)}&lon=${encodeURIComponent(lon)}`),
  godseyeCommand: (text, look = {}) =>
    req('/godseye/command', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, lat: look.lat, lon: look.lon, alt: look.alt }),
    }),
}

export function setToken(token) {
  if (token) localStorage.setItem('ironsight_token', token)
  else localStorage.removeItem('ironsight_token')
}

export function getToken() {
  return localStorage.getItem('ironsight_token')
}

export function sessionWsUrl(sessionId) {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  const token = getToken()
  const q = token ? `?token=${encodeURIComponent(token)}` : ''
  return `${proto}://${location.host}/api/ws/${sessionId}${q}`
}

export function resolveMediaPath(sessionId, absPath) {
  if (!absPath) return null
  const marker = `/sessions/${sessionId}/`
  const idx = absPath.replace(/\\/g, '/').indexOf(marker)
  if (idx === -1) {
    const m = absPath
      .replace(/\\/g, '/')
      .match(/\/sessions\/[^/]+\/(previews|frames|recon|audio|uploads|uploads_clean|gsplat|tracks)\/(.+)$/)
    if (!m) return null
    return `${BASE}/sessions/${sessionId}/media/${m[1]}/${m[2]}`
  }
  const rest = absPath.replace(/\\/g, '/').slice(idx + marker.length)
  const [kind, ...parts] = rest.split('/')
  return `${BASE}/sessions/${sessionId}/media/${kind}/${parts.join('/')}`
}
