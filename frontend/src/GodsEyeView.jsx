/**
 * Standalone Gods Eye cockpit — original implementation.
 *
 * Interaction ideas (live public layers, sensor looks, click-to-track,
 * cockpit chase, voice fly-to) inspired by a public spy-satellite demo.
 * No third-party repo, datasets, models, or CSS were copied.
 */
import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { Html, Line, OrbitControls } from '@react-three/drei'
import * as THREE from 'three'
import CockpitHud from './godseye/CockpitHud.jsx'
import { ascentTrail, reconstructedAscent } from './godseye/ascent.js'
import GlobeEarth, { SensorScreen } from './godseye/GlobeEarth.jsx'
import CctvMesh, { CCTV_CITIES } from './godseye/CctvMesh.jsx'
import ContactGlyph, { contactXYZ } from './godseye/ContactGlyph.jsx'
import {
  EARTH_R,
  greatCirclePoints,
  haversineKm,
  headingFromCamera,
  headingVector,
  latLonAltToXYZ,
  localFrame,
  nearestContacts,
  xyzToLatLonAlt,
} from './godseye/geo.js'
import {
  createFlightStore,
  geocodePlace,
  loadCctv,
  loadConfig,
  loadFires,
  loadFlights,
  loadStorms,
  loadMissions,
  loadQuakes,
  loadRadio,
  loadShips,
  loadSites,
  loadTles,
  isAirborne,
  isGroundIntel,
  orbitalConstellation,
  pickAirborne,
  runCommand,
  simulateTraffic,
  tlesToSatellites,
  updateSatellitePositions,
} from './godseye/layers.js'
import {
  AIR_HUBS,
  LAYER_SCENES,
  SEA_HUBS,
  chaseAlt,
  coverageOrigin,
  findIss,
  glyphBudget,
  layerContactPriority,
  pickRadio,
  pickSite,
  pickStorm,
  nearbyRadiusKm,
  nearestHub,
  isCityPicture,
  CCTV_SCENE_ALT_M,
  cinematicThrowM,
  overviewThrowM,
  cityFov,
  flyMinDistance,
  layerClickCamMode,
  resolveSceneLook,
  tilesAltitudeOk,
  contactMatchesFocus,
  detectionForLayer,
  lookPatchOk,
} from './godseye/scenes.js'
import LookPatch from './godseye/LookPatch.jsx'
import { DEFAULT_LAYERS, parseGodsEyeState, serializeGodsEyeState, SENSORS } from './godseye/urlState.js'

const TilesGlobe = React.lazy(() => import('./godseye/TilesGlobe.jsx'))

const HOME = { lat: 33.9425, lon: -118.4081, alt: 9_200 }
const RESET_LOOK = { lat: 18, lon: -24, alt: 16_000_000 }

function isStaleLook(initial) {
  if (initial?.lat == null || initial?.lon == null) return true
  if (initial.track) return false
  const alt = Number(initial.alt)
  if (!Number.isFinite(alt)) return true
  if (alt > 3_500_000) return true
  if (Math.abs(alt - EARTH_R) < 80_000) return true
  if (isCityPicture(initial.layers) && !tilesAltitudeOk(alt)) return true
  return false
}

function resetOrbitDeltas(controls) {
  if (!controls) return
  const delta = controls._sphericalDelta || controls.sphericalDelta
  if (delta?.set) delta.set(0, 0, 0)
  const pan = controls._panOffset || controls.panOffset
  if (pan?.set) pan.set(0, 0, 0)
  if (controls._scale != null) controls._scale = 1
  controls.scale = 1
}

/** three r170 stores spherical on _spherical. update() rebuilds it from camera − target. */
function syncOrbitFromCamera(controls, camera) {
  if (!controls || !camera) return
  resetOrbitDeltas(controls)
  const offset = new THREE.Vector3().copy(camera.position).sub(controls.target)
  if (offset.lengthSq() < 1e-8) return
  if (controls._quat) offset.applyQuaternion(controls._quat)
  const sph = controls._spherical || controls.spherical
  if (sph?.setFromVector3) sph.setFromVector3(offset)
  const sph0 = controls.spherical0
  if (sph0?.setFromVector3) sph0.setFromVector3(offset)
}
function ringPoints(lat, lon, km = 4, n = 48) {
  const pts = []
  const dlat = km / 111.0
  const dlon = km / (111.0 * Math.max(0.2, Math.cos((lat * Math.PI) / 180)))
  for (let i = 0; i <= n; i++) {
    const a = (i / n) * Math.PI * 2
    const p = latLonAltToXYZ(lat + Math.sin(a) * dlat, lon + Math.cos(a) * dlon, 900)
    pts.push(new THREE.Vector3(p.x, p.y, p.z))
  }
  return pts
}

function geojsonToLines(geojson) {
  if (!geojson) return []
  const rings = []
  const take = (coords) => {
    const pts = coords.map(([lon, lat]) => {
      const p = latLonAltToXYZ(lat, lon, 1200)
      return new THREE.Vector3(p.x, p.y, p.z)
    })
    if (pts.length > 1) rings.push(pts)
  }
  const walk = (g) => {
    if (!g) return
    if (g.type === 'Polygon') g.coordinates.forEach(take)
    else if (g.type === 'MultiPolygon') g.coordinates.forEach((poly) => poly.forEach(take))
    else if (g.type === 'LineString') take(g.coordinates)
    else if (g.type === 'MultiLineString') g.coordinates.forEach(take)
  }
  walk(geojson)
  return rings
}

function cinematicPose(lat, lon, alt, goal, tgt, throwM = cinematicThrowM(alt), heightScale = 0.78) {
  const p = latLonAltToXYZ(lat, lon, 0)
  const n = new THREE.Vector3(p.x, p.y, p.z).normalize()
  if ((alt || 0) > 1_200_000) {
    goal.copy(n).multiplyScalar(EARTH_R + alt)
    tgt.set(0, 0, 0)
    return
  }
  const worldUp = new THREE.Vector3(0, 1, 0)
  const east = new THREE.Vector3().crossVectors(worldUp, n)
  if (east.lengthSq() < 1e-8) east.set(1, 0, 0)
  east.normalize()
  const north = new THREE.Vector3().crossVectors(n, east).normalize()
  const throwAmt = throwM
  goal
    .copy(n)
    .multiplyScalar(EARTH_R + alt * heightScale)
    .addScaledVector(east, throwAmt * 0.82)
    .addScaledVector(north, throwAmt * 0.26)
  tgt.copy(n).multiplyScalar(EARTH_R)
}

function scenePose(lat, lon, alt, goal, tgt) {
  cinematicPose(lat, lon, alt, goal, tgt, overviewThrowM(alt), 0.92)
}

function lookXYZ(look) {
  const goal = new THREE.Vector3()
  const tgt = new THREE.Vector3()
  scenePose(look.lat, look.lon, look.alt || HOME.alt, goal, tgt)
  return goal
}

function orbitRing(alt, tiltDeg, n = 128) {
  const tilt = (tiltDeg * Math.PI) / 180
  const r = EARTH_R + alt
  const pts = []
  for (let i = 0; i <= n; i++) {
    const a = (i / n) * Math.PI * 2
    const x = Math.cos(a)
    const z = Math.sin(a)
    const y = -z * Math.sin(tilt)
    const zz = z * Math.cos(tilt)
    pts.push(new THREE.Vector3(x * r, y * r, zz * r))
  }
  return pts
}

function SatRings() {
  const rings = useMemo(
    () => [
      { pts: orbitRing(420_000, 51.6), color: '#f5d76e', opacity: 0.55 },
      { pts: orbitRing(780_000, 86), color: '#7ee0ff', opacity: 0.28 },
    ],
    []
  )
  return (
    <group>
      {rings.map((ring, i) => (
        <Line key={i} points={ring.pts} color={ring.color} transparent opacity={ring.opacity} lineWidth={1} />
      ))}
    </group>
  )
}

function Trail({ points, color = '#f5d76e' }) {
  const pts = useMemo(() => points.map((p) => new THREE.Vector3(p.x, p.y, p.z)), [points])
  if (pts.length < 2) return null
  return <Line points={pts} color={color} transparent opacity={0.7} lineWidth={1.6} />
}

function Annotation({ item }) {
  const lines = useMemo(() => {
    const fromJson = geojsonToLines(item.geojson)
    if (fromJson.length) return fromJson
    if (item.points) return [item.points.map((p) => new THREE.Vector3(p.x, p.y, p.z))]
    return [ringPoints(item.lat, item.lon, item.km || 3.2)]
  }, [item])
  return (
    <group>
      {lines.map((pts, i) => (
        <Line key={i} points={pts} color="#f5d76e" transparent opacity={0.85} lineWidth={1.4} />
      ))}
      <Html
        position={(() => {
          const p = latLonAltToXYZ(item.lat, item.lon, 4000)
          return [p.x, p.y, p.z]
        })()}
        pointerEvents="none"
        zIndexRange={[8, 0]}
        style={{ pointerEvents: 'none' }}
      >
        <div className="ge-float-tag">{item.label}</div>
      </Html>
    </group>
  )
}

function DetectionProjector({ contacts, selectedId, onBoxes }) {
  const { camera, size } = useThree()
  const last = useRef(0)
  const sig = useRef('')
  useLayoutEffect(() => {
    sig.current = ''
    onBoxes([])
  }, [onBoxes])
  useFrame(() => {
    const now = performance.now()
    if (now - last.current < 250) return
    last.current = now
    const v = new THREE.Vector3()
    const boxes = []
    const cap = Math.min(contacts.length, 24)
    for (let i = 0; i < cap; i++) {
      const c = contacts[i]
      const p = contactXYZ(c)
      v.set(p.x, p.y, p.z)
      v.project(camera)
      if (v.z > 1 || v.z < -1 || Math.abs(v.x) > 1.15 || Math.abs(v.y) > 1.15) continue
      const x = (v.x * 0.5 + 0.5) * size.width
      const y = (-v.y * 0.5 + 0.5) * size.height
      const near = Math.max(0.15, 1 - Math.abs(v.z))
      const w = 18 + near * 28
      const h = 14 + near * 20
      boxes.push({
        id: c.id,
        x,
        y,
        w,
        h,
        label: (c.callsign || c.id).slice(0, 14),
        alt_m: c.alt_m,
        gs_kts: c.gs_kts,
        kind: c.military ? 'mil' : c.kind,
        military: !!c.military,
        selected: c.id === selectedId,
        contact: c,
      })
    }
    const next = boxes.map((b) => `${b.id}:${b.x | 0}:${b.y | 0}:${b.selected ? 1 : 0}`).join('|')
    if (next === sig.current) return
    sig.current = next
    onBoxes(boxes)
  })
  return null
}

/** One-shot pose. Remount with sceneKey — never re-run when the user drags. */
function CameraBoot({ lat, lon, alt }) {
  const { camera, controls } = useThree()
  const pose = useRef({ lat, lon, alt })
  const hold = useRef(36)
  const apply = () => {
    const { lat: la, lon: lo, alt: al } = pose.current
    const goal = new THREE.Vector3()
    const tgt = new THREE.Vector3()
    scenePose(la, lo, al, goal, tgt)
    camera.position.copy(goal)
    camera.fov = cityFov(al, false)
    camera.near = 12
    camera.far = EARTH_R * 40
    camera.updateProjectionMatrix()
    camera.lookAt(tgt)
    if (!controls) return false
    if (controls) {
      controls.enabled = hold.current <= 1
      controls.minDistance = flyMinDistance(al)
      controls.maxDistance = EARTH_R * 8
      controls.target.copy(tgt)
      resetOrbitDeltas(controls)
      syncOrbitFromCamera(controls, camera)
    }
    return Boolean(controls)
  }
  useLayoutEffect(() => {
    apply()
  }, [camera, controls])
  useFrame(() => {
    if (hold.current <= 0) return
    if (apply()) hold.current -= 1
  })
  return null
}

function CameraRig({ tracked, camMode, draggingRef, holdRef }) {
  const { camera, controls } = useThree()
  const goal = useRef(new THREE.Vector3())
  const tgt = useRef(new THREE.Vector3())
  const orbitA = useRef(0)

  useFrame((_, dt) => {
    if (holdRef?.current > 0) {
      holdRef.current -= 1
      return
    }
    const dragging = draggingRef?.current

    if (tracked && camMode === 'cockpit' && !dragging) {
      const p = contactXYZ(tracked)
      const pos = new THREE.Vector3(p.x, p.y, p.z)
      const n = pos.clone().normalize()
      const fwd = headingVector(tracked.lat, tracked.lon, tracked.heading || 0)
      const f = new THREE.Vector3(fwd.x, fwd.y, fwd.z).normalize()
      const agl = Math.max(40, Math.abs(tracked.alt_m || 0))
      const lift = Math.max(28, agl * 0.045 + 36)
      const back = Math.max(70, agl * 0.11 + 90)
      goal.current.copy(pos).addScaledVector(f, -back).addScaledVector(n, lift)
      tgt.current.copy(pos).addScaledVector(f, Math.max(180, back * 3.2)).addScaledVector(n, -lift * 0.15)
      camera.position.lerp(goal.current, 1 - Math.pow(0.00008, dt))
      if (controls) {
        controls.target.lerp(tgt.current, 1 - Math.pow(0.0002, dt))
        controls.update()
      } else camera.lookAt(tgt.current)
      return
    }

    if (tracked && camMode === 'orbit' && controls && !dragging) {
      const p = contactXYZ(tracked)
      const pos = new THREE.Vector3(p.x, p.y, p.z)
      const n = pos.clone().normalize()
      const craft = tracked.kind === 'flight' || tracked.military || tracked.kind === 'ship'
      if (tracked.kind === 'sat') {
        tgt.current.set(0, 0, 0)
      } else if (tracked.kind === 'mission' && !tracked.reconstructed) {
        scenePose(tracked.lat, tracked.lon, chaseAlt(tracked), goal.current, tgt.current)
      } else if (craft) {
        tgt.current.copy(pos)
      } else {
        tgt.current.copy(n).multiplyScalar(EARTH_R)
      }
      controls.minDistance = flyMinDistance(chaseAlt(tracked))
      controls.target.lerp(tgt.current, 1 - Math.pow(0.04, dt))
      orbitA.current += dt * 0.16
      if (craft) {
        const fwd = headingVector(tracked.lat, tracked.lon, tracked.heading || 0)
        const f = new THREE.Vector3(fwd.x, fwd.y, fwd.z).normalize()
        const east = new THREE.Vector3().crossVectors(n, f)
        if (east.lengthSq() < 1e-8) east.crossVectors(n, new THREE.Vector3(0, 1, 0))
        east.normalize()
        const flying = isAirborne(tracked)
        const agl = Math.max(120, Math.abs(tracked.alt_m || 80))
        const lift = tracked.kind === 'ship' ? 2800 : flying ? Math.max(2800, agl * 0.18 + 2200) : 14_000
        const back = tracked.kind === 'ship' ? 7200 : flying ? Math.max(5200, agl * 0.42 + 3800) : 8_000
        goal.current
          .copy(pos)
          .addScaledVector(f, -back)
          .addScaledVector(n, lift)
          .addScaledVector(east, Math.sin(orbitA.current) * back * 0.42)
      } else if (tracked.kind === 'sat') {
        const lift = LAYER_SCENES.sats.alt
        const east = new THREE.Vector3().crossVectors(new THREE.Vector3(0, 1, 0), n)
        if (east.lengthSq() < 1e-8) east.set(1, 0, 0)
        east.normalize()
        const north = new THREE.Vector3().crossVectors(n, east).normalize()
        tgt.current.set(0, 0, 0)
        goal.current
          .copy(n)
          .multiplyScalar(EARTH_R + lift)
          .addScaledVector(east, Math.sin(orbitA.current) * lift * 0.08)
          .addScaledVector(north, Math.cos(orbitA.current) * lift * 0.05)
      } else if (tracked.kind === 'mission' && !tracked.reconstructed) {
        scenePose(tracked.lat, tracked.lon, chaseAlt(tracked), goal.current, tgt.current)
      } else {
        const lift = chaseAlt(tracked)
        const east = new THREE.Vector3().crossVectors(new THREE.Vector3(0, 1, 0), n)
        if (east.lengthSq() < 1e-8) east.set(1, 0, 0)
        east.normalize()
        const north = new THREE.Vector3().crossVectors(n, east).normalize()
        tgt.current.copy(n).multiplyScalar(EARTH_R)
        goal.current
          .copy(n)
          .multiplyScalar(EARTH_R + lift)
          .addScaledVector(east, Math.sin(orbitA.current) * lift * 0.22)
          .addScaledVector(north, Math.cos(orbitA.current) * lift * 0.16)
      }
      camera.position.lerp(goal.current, 1 - Math.pow(0.03, dt))
      controls.update()
      return
    }
  })

  return null
}

function ControlsLimiter({ sceneAlt, sceneLat, sceneLon, holdRef }) {
  const { camera, controls } = useThree()
  useFrame(() => {
    if (!controls) return
    const alt = Math.max(1_200, Number(sceneAlt) || 12_000)
    const city = alt <= 1_200_000
    if (holdRef?.current > 0) {
      const goal = new THREE.Vector3()
      const tgt = new THREE.Vector3()
      scenePose(sceneLat || 0, sceneLon || 0, alt, goal, tgt)
      camera.position.copy(goal)
      camera.fov = cityFov(alt, false)
      camera.updateProjectionMatrix()
      controls.enabled = false
      controls.minDistance = flyMinDistance(alt)
      controls.target.copy(tgt)
      resetOrbitDeltas(controls)
      syncOrbitFromCamera(controls, camera)
      return
    }
    controls.enabled = true
    const t = controls.target
    const tl = Math.hypot(t.x, t.y, t.z)
    // After Satellites the target is Earth-center. A city restage with
    // minDistance 90 then parks the camera inside the planet (black until zoom-out).
    if (city && tl < EARTH_R * 0.55) {
      const surf = latLonAltToXYZ(sceneLat || 0, sceneLon || 0, 0)
      t.set(surf.x, surf.y, surf.z)
    }
    controls.minDistance = flyMinDistance(alt)
    const cl = Math.hypot(camera.position.x, camera.position.y, camera.position.z)
    if (cl < EARTH_R + 120) {
      const goal = new THREE.Vector3()
      const tgt = new THREE.Vector3()
      scenePose(sceneLat || 0, sceneLon || 0, alt, goal, tgt)
      camera.position.copy(goal)
      t.copy(tgt)
      syncOrbitFromCamera(controls, camera)
    }
  })
  return null
}

function LookReporter({ onLook }) {
  const { camera } = useThree()
  const last = useRef(0)
  const prev = useRef(null)
  useFrame(() => {
    const now = performance.now()
    if (now - last.current < 300) return
    last.current = now
    const ll = xyzToLatLonAlt(camera.position.x, camera.position.y, camera.position.z)
    const heading = headingFromCamera(camera.position.x, camera.position.z)
    const p = prev.current
    if (
      p &&
      Math.abs(p.lat - ll.lat) < 0.00015 &&
      Math.abs(p.lon - ll.lon) < 0.00015 &&
      Math.abs(p.alt - ll.alt) < 20 &&
      Math.abs(p.heading - heading) < 1.2
    ) {
      return
    }
    prev.current = { ...ll, heading }
    onLook({ ...ll, heading })
  })
  return null
}

function speak(text) {
  if (!text || typeof window === 'undefined' || !window.speechSynthesis) return
  window.speechSynthesis.cancel()
  const u = new SpeechSynthesisUtterance(text)
  u.rate = 1.04
  u.pitch = 0.92
  window.speechSynthesis.speak(u)
}

function flyAltFor(query, hit) {
  const q = `${query} ${hit?.kind || ''} ${hit?.label || ''}`.toLowerCase()
  if (/\b(airport|lax|jfk|sfo|lhr|nrt|cdg|dxb|sin|ksc|pad)\b/.test(q)) return 8_500
  if (/\b(street|park|capitol|tower|bridge)\b/.test(q)) return 2_400
  if (/\b(state|country|ocean|globe)\b/.test(q)) return 1_800_000
  return 28_000
}

export default function GodsEyeView({ onEnterRange } = {}) {
  const initial = useMemo(
    () =>
      parseGodsEyeState(
        typeof location !== 'undefined' ? location.hash : '',
        typeof location !== 'undefined' ? location.search : ''
      ),
    []
  )
  const startLook = useMemo(() => {
    if (!isStaleLook(initial)) {
      return { lat: initial.lat, lon: initial.lon, alt: initial.alt || HOME.alt, heading: 0 }
    }
    return { ...HOME, heading: 0 }
  }, [initial])

  const [sensor, setSensor] = useState(initial.sensor || 'rgb')
  const [camMode, setCamMode] = useState(initial.cam || 'free')
  const [layersOn, setLayersOn] = useState(initial.layers || [...DEFAULT_LAYERS])
  const [trackedId, setTrackedId] = useState(initial.track || null)
  const [look, setLook] = useState(startLook)
  const [sceneLook, setSceneLook] = useState(startLook)
  const [sceneKey, setSceneKey] = useState(0)
  const [search, setSearch] = useState('')
  const [flyHint, setFlyHint] = useState('')
  const draggingRef = useRef(false)
  const [flights, setFlights] = useState([])
  const [military, setMilitary] = useState([])
  const [ships, setShips] = useState([])
  const [sats, setSats] = useState([])
  const [quakes, setQuakes] = useState([])
  const [fires, setFires] = useState([])
  const [storms, setStorms] = useState([])
  const [sites, setSites] = useState([])
  const [radios, setRadios] = useState([])
  const [missions, setMissions] = useState([])
  const [cctv, setCctv] = useState([])
  const [traffic, setTraffic] = useState([])
  const [annotations, setAnnotations] = useState([])
  const [measure, setMeasure] = useState(null)
  const [layerMeta, setLayerMeta] = useState({
    flights: { status: '…', label: '…', source: 'adsb.lol' },
    military: { status: '…', label: '…', source: 'adsb.lol/mil' },
    sats: { status: '…', label: '…', source: 'celestrak' },
    quakes: { status: '…', label: '…', source: 'usgs' },
    ships: { status: '…', label: '…', source: 'openwaters' },
    fires: { status: '…', label: '…', source: 'nasa-firms' },
    storms: { status: '…', label: '…', source: 'nasa-eonet+nws' },
    sites: { status: '…', label: '…', source: 'osm-overpass' },
    radio: { status: '…', label: '…', source: 'radio-browser' },
    missions: { status: '…', label: '…', source: 'launch-library-2' },
    cctv: { status: 'loading', label: '…', source: 'city-apis' },
    traffic: { status: 'simulated', label: 'SIM', source: 'traffic-sim' },
  })
  const [config, setConfig] = useState(null)
  const [tilesStatus, setTilesStatus] = useState({ active: false, mode: 'none' })
  const [tilesWanted, setTilesWanted] = useState(false)
  const [trail, setTrail] = useState([])
  const [latencyMs, setLatencyMs] = useState(null)
  const [listening, setListening] = useState(false)
  const [transcript, setTranscript] = useState('')
  const [speakLine, setSpeakLine] = useState('')
  const [radioUrl, setRadioUrl] = useState('')
  const [hudOn, setHudOn] = useState(true)
  const [detectionOn, setDetectionOn] = useState(false)
  const [boxes, setBoxes] = useState([])
  const [missionOpen, setMissionOpen] = useState(() => {
    if (typeof localStorage === 'undefined') return true
    if (initial.track) return false
    if (isStaleLook(initial)) return true
    return localStorage.getItem('gev-mission') !== '1'
  })
  const [cctvStill, setCctvStill] = useState(null)
  const [cctvCity, setCctvCity] = useState(null)
  const [focusedLayer, setFocusedLayer] = useState(null)
  const [viewshedOn, setViewshedOn] = useState(true)
  const [replay, setReplay] = useState(null)
  const [utc, setUtc] = useState('')
  const flightStore = useRef(createFlightStore())
  const milStore = useRef(createFlightStore())
  const shipStore = useRef(createFlightStore())
  const satRef = useRef([])
  const lookRef = useRef(look)
  const recRef = useRef(null)
  const audioRef = useRef(null)
  const camModeRef = useRef(camMode)
  const quakesRef = useRef([])
  const firesRef = useRef([])
  const stormsRef = useRef([])
  const sitesRef = useRef([])
  const radiosRef = useRef([])
  const missionsRef = useRef([])
  const focusedLayerRef = useRef(null)
  const pollOrigin = useRef({ lat: HOME.lat, lon: HOME.lon, alt: HOME.alt })
  const emptyHopRef = useRef(false)
  const pendingLockRef = useRef(null)
  const restageHoldRef = useRef(0)
  const satTickRef = useRef(0)
  lookRef.current = look
  camModeRef.current = camMode
  quakesRef.current = quakes
  firesRef.current = fires
  stormsRef.current = storms
  sitesRef.current = sites
  radiosRef.current = radios
  missionsRef.current = missions
  focusedLayerRef.current = focusedLayer

  const startPos = useMemo(() => lookXYZ(startLook), [startLook])

  useEffect(() => {
    document.body.classList.add('gods-eye')
    document.title = 'GODS EYE'
    const release = () => {
      draggingRef.current = false
    }
    window.addEventListener('pointerup', release)
    window.addEventListener('pointercancel', release)
    return () => {
      document.body.classList.remove('gods-eye')
      document.title = 'IronSight'
      window.removeEventListener('pointerup', release)
      window.removeEventListener('pointercancel', release)
    }
  }, [])

  useEffect(() => {
    const id = setInterval(() => {
      const d = new Date()
      const p = (n) => String(n).padStart(2, '0')
      setUtc(
        `${d.getUTCFullYear()}-${p(d.getUTCMonth() + 1)}-${p(d.getUTCDate())} ${p(d.getUTCHours())}:${p(d.getUTCMinutes())}:${p(d.getUTCSeconds())}Z`
      )
    }, 1000)
    return () => clearInterval(id)
  }, [])

  useEffect(() => {
    loadConfig()
      .then((cfg) => {
        setConfig(cfg)
        if (cfg?.cctvCount) {
          setLayerMeta((m) => ({
            ...m,
            cctv: {
              ...m.cctv,
              count: cfg.cctvCount,
              label: 'LIVE',
              status: 'live',
              source: 'tfl|austin-open-data|caltrans',
            },
          }))
        }
      })
      .catch(() => setConfig({ tiles: 'none' }))
  }, [])

  useEffect(() => {
    if (!layersOn.includes('flights') && !layersOn.includes('military')) {
      return undefined
    }
    let cancelled = false
    const poll = async () => {
      const t0 = performance.now()
      const origin = pollOrigin.current || lookRef.current
      const air = coverageOrigin(origin.lat || HOME.lat, origin.lon || HOME.lon, AIR_HUBS, 650)
      const f = await loadFlights(air.lat, air.lon, 450)
      if (cancelled) return
      const civ = (f.contacts || []).filter((c) => !c.military)
      const mil = (f.contacts || []).filter((c) => c.military)
      flightStore.current.ingest(civ)
      milStore.current.ingest(mil)
      setLayerMeta((m) => ({
        ...m,
        flights: { ...f, count: civ.length, contacts: civ, label: civ.length ? 'LIVE' : f.label },
        military: {
          ...f,
          kind: 'military',
          source: 'adsb.lol/mil',
          count: mil.length,
          contacts: mil,
          label: mil.length ? 'LIVE' : f.label,
        },
      }))
      if (civ.length === 0 && !emptyHopRef.current && focusedLayerRef.current === 'flights') {
        emptyHopRef.current = true
        const { hub } = nearestHub(origin.lat || HOME.lat, origin.lon || HOME.lon, AIR_HUBS, 0)
        pollOrigin.current = { lat: hub.lat, lon: hub.lon, alt: LAYER_SCENES.flights.alt }
        flyTo(hub.lat, hub.lon, LAYER_SCENES.flights.alt, `Live flights · ${hub.id}`)
        setSpeakLine('No ADS-B over that ocean. Jumping to live airspace.')
      }
      const want = pendingLockRef.current
      if (want === 'flight') {
        const pick = pickAirborne(civ, air.lat, air.lon)
        if (pick) {
          pendingLockRef.current = null
          setTrackedId(pick.id)
        }
      } else if (want === 'mil') {
        const pick = pickAirborne(mil, air.lat, air.lon) || pickAirborne(civ, air.lat, air.lon)
        if (pick) {
          pendingLockRef.current = null
          setTrackedId(pick.id)
        }
      }
      if (!cancelled) setLatencyMs(performance.now() - t0)
    }
    poll()
    const id = setInterval(poll, 12000)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [layersOn])

  useEffect(() => {
    if (layersOn.includes('flights') || layersOn.includes('military')) return undefined
    let cancelled = false
    const tick = () => {
      const look = lookRef.current
      const air = coverageOrigin(look.lat, look.lon, AIR_HUBS, 650)
      loadFlights(air.lat, air.lon, 450).then((f) => {
        if (cancelled) return
        const civ = (f.contacts || []).filter((c) => !c.military)
        const mil = (f.contacts || []).filter((c) => c.military)
        setLayerMeta((m) => ({
          ...m,
          flights: { ...f, count: civ.length, contacts: civ, label: civ.length ? 'LIVE' : f.label },
          military: {
            ...f,
            kind: 'military',
            source: 'adsb.lol/mil',
            count: mil.length,
            contacts: mil,
            label: mil.length ? 'LIVE' : f.label,
          },
        }))
      })
    }
    tick()
    const id = setInterval(tick, 20000)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [layersOn])

  useEffect(() => {
    if (!layersOn.includes('ships')) {
      return undefined
    }
    let cancelled = false
    const poll = async () => {
      const origin = pollOrigin.current || lookRef.current
      const sea = coverageOrigin(origin.lat || HOME.lat, origin.lon || HOME.lon, SEA_HUBS, 500)
      const s = await loadShips(sea.lat, sea.lon, 160)
      if (cancelled) return
      shipStore.current.ingest(s.contacts || [])
      setLayerMeta((m) => ({ ...m, ships: s }))
      if (pendingLockRef.current === 'ship' && (s.contacts || []).length) {
        const pick = [...s.contacts].sort(
          (a, b) => haversineKm(sea.lat, sea.lon, a.lat, a.lon) - haversineKm(sea.lat, sea.lon, b.lat, b.lon)
        )[0]
        pendingLockRef.current = null
        setTrackedId(pick.id)
      }
    }
    poll()
    const id = setInterval(poll, 20000)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [layersOn])

  useEffect(() => {
    if (!layersOn.includes('sats')) {
      return undefined
    }
    let cancelled = false
    loadTles().then((doc) => {
      if (cancelled) return
      setLayerMeta((m) => ({ ...m, sats: doc }))
      const list = tlesToSatellites(doc.tles || doc.omm || [])
      satRef.current = list
      setSats(list)
    })
    return () => {
      cancelled = true
    }
  }, [layersOn])

  useEffect(() => {
    if (!layersOn.includes('quakes')) {
      return undefined
    }
    let cancelled = false
    loadQuakes().then((doc) => {
      if (cancelled) return
      setLayerMeta((m) => ({ ...m, quakes: doc }))
      setQuakes(doc.contacts || [])
    })
    return () => {
      cancelled = true
    }
  }, [layersOn])

  useEffect(() => {
    if (!layersOn.includes('fires')) {
      return undefined
    }
    let cancelled = false
    loadFires().then((doc) => {
      if (cancelled) return
      setLayerMeta((m) => ({ ...m, fires: doc }))
      setFires((doc.contacts || []).slice(0, 80))
    })
    return () => {
      cancelled = true
    }
  }, [layersOn])

  useEffect(() => {
    if (!layersOn.includes('storms')) {
      return undefined
    }
    let cancelled = false
    loadStorms().then((doc) => {
      if (cancelled) return
      setLayerMeta((m) => ({ ...m, storms: doc }))
      setStorms(doc.contacts || [])
    })
    return () => {
      cancelled = true
    }
  }, [layersOn])

  useEffect(() => {
    if (!layersOn.includes('sites')) {
      setSites([])
      return undefined
    }
    let cancelled = false
    const run = () => {
      const origin = pollOrigin.current || lookRef.current
      const { lat, lon } = origin
      loadSites(lat || HOME.lat, lon || HOME.lon, (origin.alt || lookRef.current.alt) > 6_000_000 ? 160 : 90).then((doc) => {
        if (cancelled) return
        setLayerMeta((m) => ({ ...m, sites: doc }))
        setSites(doc.contacts || [])
      })
    }
    run()
    const id = setInterval(run, 90_000)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [layersOn])

  useEffect(() => {
    if (!layersOn.includes('radio')) {
      setRadios([])
      return undefined
    }
    let cancelled = false
    const run = () => {
      const origin = pollOrigin.current || lookRef.current
      const { lat, lon } = origin
      loadRadio(lat || HOME.lat, lon || HOME.lon).then((doc) => {
        if (cancelled) return
        setLayerMeta((m) => ({ ...m, radio: doc }))
        setRadios(doc.contacts || [])
      })
    }
    run()
    return () => {
      cancelled = true
    }
  }, [layersOn])

  useEffect(() => {
    if (!layersOn.includes('missions')) {
      return undefined
    }
    let cancelled = false
    loadMissions().then((doc) => {
      if (cancelled) return
      setLayerMeta((m) => ({ ...m, missions: doc }))
      setMissions(doc.contacts || [])
    })
    return () => {
      cancelled = true
    }
  }, [layersOn])

  useEffect(() => {
    if (!layersOn.includes('cctv')) return undefined
    let cancelled = false
    const pullCctv = () => {
      loadCctv().then((doc) => {
        if (cancelled) return
        if ((doc.contacts || []).length) {
          setLayerMeta((m) => ({ ...m, cctv: doc }))
          setCctv(doc.contacts || [])
        }
      })
    }
    pullCctv()
    const cctvTick = setInterval(pullCctv, 60_000)
    return () => {
      cancelled = true
      clearInterval(cctvTick)
    }
  }, [layersOn])

  useEffect(() => {
    const traf = simulateTraffic(HOME.lat, HOME.lon)
    setLayerMeta((m) => ({
      ...m,
      traffic: { status: 'simulated', label: 'SIM', count: traf.contacts.length, source: 'traffic-sim' },
    }))
    let cancelled = false
    const idle = window.setTimeout(() => {
      if (cancelled) return
      const metaOnly = (key, doc) => {
        if (cancelled) return
        setLayerMeta((m) => ({
          ...m,
          [key]: {
            ...m[key],
            status: doc.status || m[key]?.status,
            label: doc.label || m[key]?.label,
            source: doc.source || m[key]?.source,
            count: doc.count ?? (doc.contacts || []).length,
          },
        }))
      }
      loadMissions().then((doc) => metaOnly('missions', doc))
      loadQuakes().then((doc) => metaOnly('quakes', doc))
      loadFires().then((doc) => metaOnly('fires', doc))
      loadStorms().then((doc) => metaOnly('storms', doc))
    }, 5000)
    return () => {
      cancelled = true
      window.clearTimeout(idle)
    }
  }, [])

  useEffect(() => {
    const id = setInterval(() => {
      const origin = lookRef.current
      const city = (origin.alt || 0) < 20_000
      if (layersOn.includes('flights')) {
        setFlights(nearestContacts(flightStore.current.sample(Date.now()), origin.lat, origin.lon, city ? 64 : 40))
      }
      if (layersOn.includes('military')) {
        setMilitary(nearestContacts(milStore.current.sample(Date.now()), origin.lat, origin.lon, city ? 40 : 24))
      }
      if (layersOn.includes('ships')) {
        setShips(nearestContacts(shipStore.current.sample(Date.now()), origin.lat, origin.lon, city ? 60 : 36))
      }
      if (layersOn.includes('sats')) {
        if (satRef.current.length) {
          updateSatellitePositions(satRef.current)
          satTickRef.current += 1
          setSats([...satRef.current])
        } else {
          setSats(orbitalConstellation(Date.now()))
        }
      }
      if (layersOn.includes('traffic') && lookRef.current.alt < 25_000) {
        const doc = simulateTraffic(lookRef.current.lat, lookRef.current.lon)
        setTraffic(doc.contacts)
        setLayerMeta((m) => ({ ...m, traffic: doc }))
      } else if (!layersOn.includes('traffic')) {
        setTraffic([])
      }
      setReplay((cur) => {
        if (!cur?.playing) return cur
        const t = cur.t + (0.012 * (cur.rate || 1))
        return { ...cur, t: t > 1.12 ? 0 : t }
      })
    }, 500)
    return () => clearInterval(id)
  }, [layersOn])

  const replayContact = useMemo(() => {
    if (!replay) return null
    const pad = missions.find((m) => m.id === replay.id)
    if (!pad) return null
    const p = reconstructedAscent(pad.lat, pad.lon, pad.heading || 90, replay.t)
    return {
      ...pad,
      ...p,
      id: `${pad.id}-ascent`,
      kind: 'mission',
      callsign: `${pad.callsign} · ASCENT`,
      reconstructed: true,
    }
  }, [replay, missions])

  const contacts = useMemo(() => {
    const rows = []
    if (layersOn.includes('flights')) rows.push(...flights)
    if (layersOn.includes('military')) rows.push(...military)
    if (layersOn.includes('sats')) rows.push(...sats)
    if (layersOn.includes('quakes')) rows.push(...quakes)
    if (layersOn.includes('ships')) rows.push(...ships)
    if (layersOn.includes('fires')) rows.push(...fires)
    if (layersOn.includes('storms')) rows.push(...storms)
    if (layersOn.includes('sites')) rows.push(...sites)
    if (layersOn.includes('radio')) rows.push(...radios)
    if (layersOn.includes('missions')) rows.push(...missions)
    if (layersOn.includes('cctv')) rows.push(...cctv)
    if (layersOn.includes('traffic')) rows.push(...traffic)
    if (replayContact) rows.push(replayContact)
    return rows
  }, [flights, military, sats, quakes, ships, fires, storms, sites, radios, missions, cctv, traffic, layersOn, replayContact])

  const tracked = useMemo(() => {
    const hit = contacts.find((c) => c.id === trackedId)
    if (hit) return hit
    if (replayContact && trackedId === replayContact.id) return replayContact
    return null
  }, [contacts, trackedId, replayContact])

  useEffect(() => {
    if (!tracked) return
    if (!(tracked.kind === 'flight' || tracked.military)) return
    if (isAirborne(tracked)) return
    const pool = focusedLayerRef.current === 'military' ? military : flights
    const pick = pickAirborne(pool, tracked.lat, tracked.lon)
    if (pick && pick.id !== tracked.id) {
      setTrackedId(pick.id)
    }
  }, [tracked, flights, military])

  useEffect(() => {
    if (!tracked) {
      setTrail([])
      return
    }
    const p = contactXYZ(tracked)
    setTrail((prev) => {
      const last = prev[prev.length - 1]
      if (last && Math.hypot(last.x - p.x, last.y - p.y, last.z - p.z) < 400) return prev
      return [...prev.slice(-72), p]
    })
  }, [tracked])

  const rosterKm = nearbyRadiusKm(look.alt, focusedLayer === 'military' ? 'mil' : focusedLayer)
  const nearby = useMemo(() => {
    const origin = tracked || { lat: look.lat, lon: look.lon }
    const km = nearbyRadiusKm(look.alt, focusedLayer === 'military' ? 'mil' : focusedLayer)
    return contacts
      .map((c) => ({ ...c, _km: haversineKm(origin.lat, origin.lon, c.lat, c.lon) }))
      .filter((c) => c._km <= km)
      .filter((c) => c.id === tracked?.id || contactMatchesFocus(focusedLayer, c))
      .filter((c) => {
        if (c.id === tracked?.id) return true
        if ((focusedLayer === 'flights' || focusedLayer === 'military') && (c.kind === 'flight' || c.military)) {
          return isAirborne(c)
        }
        return true
      })
      .sort((a, b) => a._km - b._km)
  }, [contacts, tracked, look.lat, look.lon, look.alt, focusedLayer])
  const nearbyKm = useMemo(() => new Map(nearby.map((c) => [c.id, c._km])), [nearby])

  const writeHash = useCallback(
    (cam = lookRef.current) => {
      const hash = serializeGodsEyeState({
        lat: cam.lat,
        lon: cam.lon,
        alt: cam.alt,
        layers: layersOn,
        track: trackedId,
        sensor,
        cam: camMode,
      })
      if (typeof location !== 'undefined' && location.hash !== hash) {
        history.replaceState(null, '', `${location.pathname}${location.search}${hash}`)
      }
    },
    [layersOn, trackedId, sensor, camMode]
  )

  useEffect(() => {
    writeHash()
  }, [writeHash])

  useEffect(() => {
    const id = setInterval(() => writeHash(lookRef.current), 2000)
    return () => clearInterval(id)
  }, [writeHash])

  const flyCctvCity = (city) => {
    const hit = CCTV_CITIES.find((c) => c.id === city)
    if (!hit) return
    setCctvCity(city)
    setFocusedLayer('cctv')
    setLayersOn((cur) => {
      const next = cur.includes('cctv') ? cur : [...cur, 'cctv']
      return next.includes('traffic') ? next : [...next, 'traffic']
    })
    flyTo(hit.lat, hit.lon, hit.alt || CCTV_SCENE_ALT_M, `${city} cameras`)
    setSpeakLine(`${city} public cameras.`)
  }

  const stageLayer = (k) => {
    const scene = LAYER_SCENES[k]
    if (!scene) return
    setCctvStill(null)
    setTrackedId(null)
    setReplay(null)
    setBoxes([])
    setCamMode(layerClickCamMode())
    setDetectionOn(detectionForLayer(k))
    setHudOn(true)
    const keep = new Set(scene.layers || [])
    if (!keep.has('flights')) setFlights([])
    if (!keep.has('military')) setMilitary([])
    if (!keep.has('ships')) setShips([])
    if (!keep.has('sats')) setSats([])
    if (!keep.has('quakes')) setQuakes([])
    if (!keep.has('fires')) setFires([])
    if (!keep.has('storms')) setStorms([])
    if (!keep.has('sites')) setSites([])
    if (!keep.has('radio')) setRadios([])
    if (!keep.has('missions')) setMissions([])
    if (!keep.has('cctv')) setCctv([])
    if (!keep.has('traffic')) setTraffic([])
    setFocusedLayer(k)
    emptyHopRef.current = false
    pendingLockRef.current =
      k === 'flights'
        ? 'flight'
        : k === 'military'
          ? 'mil'
          : k === 'ships'
            ? 'ship'
            : k === 'quakes'
              ? 'quake'
              : k === 'fires'
                ? 'fire'
                : k === 'storms'
                  ? 'storm'
                  : k === 'sites'
                    ? 'site'
                    : k === 'radio'
                      ? 'radio'
                      : k === 'missions'
                        ? 'mission'
                        : null
    if (k !== 'sats' && k !== 'missions') setSensor('rgb')
    if (k === 'cctv') {
      setLayersOn(scene.layers)
      if (!(cctv || []).length) setSpeakLine('Acquiring public cameras.')
      const near = CCTV_CITIES.map((c) => ({
        c,
        d: haversineKm(lookRef.current.lat, lookRef.current.lon, c.lat, c.lon),
      })).sort((a, b) => a.d - b.d)[0]
      const pick = near && near.d < 80 ? near.c : CCTV_CITIES[0]
      flyCctvCity(pick.id)
      return
    }
    setLayersOn(scene.layers)
    const quake = [...(quakesRef.current || [])].sort((a, b) => (b.mag || 0) - (a.mag || 0))[0]
    const fire = [...(firesRef.current || [])].sort((a, b) => (b.frp || 0) - (a.frp || 0))[0]
    const storm = pickStorm(stormsRef.current)
    const site = pickSite(sitesRef.current)
    const radio = pickRadio(radiosRef.current)
    const mil = [...(milStore.current.sample(Date.now()) || [])].sort((a, b) => (b.gs_kts || 0) - (a.gs_kts || 0))[0]
    const pad = (missionsRef.current || [])[0] || null
    const look = resolveSceneLook(scene, lookRef.current)
    if (!look) return
    pollOrigin.current = { lat: look.lat, lon: look.lon, alt: look.alt }

    let flyLat = look.lat
    let flyLon = look.lon
    let flyAlt = look.alt
    let flyHint = look.hint
    let keepTrack = false

    if (k === 'traffic') {
      const traf = simulateTraffic(look.lat, look.lon)
      setTraffic(traf.contacts)
      setLayerMeta((m) => ({ ...m, traffic: traf }))
    }
    if (k === 'sats' && !satRef.current.length) {
      setSats(orbitalConstellation())
    }

    if (k === 'flights') {
      const pick = pickAirborne(flightStore.current.sample(Date.now()), look.lat, look.lon)
      if (pick) {
        setTrackedId(pick.id)
        keepTrack = true
        pendingLockRef.current = null
      }
    } else if (k === 'military') {
      const pick =
        pickAirborne(milStore.current.sample(Date.now()), look.lat, look.lon) ||
        (mil && isAirborne(mil) ? mil : null)
      if (pick) {
        setTrackedId(pick.id)
        keepTrack = true
        pendingLockRef.current = null
      }
    } else if (k === 'ships') {
      const pool = shipStore.current.sample(Date.now()) || []
      const pick = [...pool].sort(
        (a, b) => haversineKm(look.lat, look.lon, a.lat, a.lon) - haversineKm(look.lat, look.lon, b.lat, b.lon)
      )[0]
      if (pick) {
        setTrackedId(pick.id)
        keepTrack = true
        pendingLockRef.current = null
      }
    } else if (k === 'sats') {
      pendingLockRef.current = null
    } else if (k === 'missions') {
      if (pad) {
        setTrackedId(pad.id)
        keepTrack = true
        pendingLockRef.current = null
      } else {
        pendingLockRef.current = 'mission'
      }
    } else if (k === 'quakes' && quake) {
      setTrackedId(quake.id)
      keepTrack = true
      pendingLockRef.current = null
    } else if (k === 'fires' && fire) {
      setTrackedId(fire.id)
      keepTrack = true
      pendingLockRef.current = null
    } else if (k === 'storms' && storm) {
      setTrackedId(storm.id)
      keepTrack = true
      pendingLockRef.current = null
    } else if (k === 'sites' && site) {
      setTrackedId(site.id)
      keepTrack = true
      pendingLockRef.current = null
    } else if (k === 'radio' && radio) {
      setTrackedId(radio.id)
      keepTrack = true
      pendingLockRef.current = null
    }

    flyTo(flyLat, flyLon, flyAlt, flyHint, { keepTrack })
    setSpeakLine(look.speak)
  }

  const toggleLayer = (k) => {
    stageLayer(k)
  }

  const flyTo = (lat, lon, alt = 14_000, hint = '', opts = {}) => {
    draggingRef.current = false
    if (!opts.keepTrack) setTrackedId(null)
    if (!opts.keepMode) setCamMode(layerClickCamMode())
    const heading = lookRef.current.heading || 0
    const next = { lat, lon, alt, heading }
    lookRef.current = next
    setLook(next)
    setSceneLook({ lat, lon, alt })
    restageHoldRef.current = 48
    setSceneKey((n) => n + 1)
    pollOrigin.current = { lat, lon, alt }
    if (hint) setFlyHint(hint)
  }

  useEffect(() => {
    const want = pendingLockRef.current
    const lockOnly = (id) => {
      pendingLockRef.current = null
      if (id) setTrackedId(id)
    }
    if (want === 'quake' && quakes.length) {
      lockOnly([...quakes].sort((a, b) => (b.mag || 0) - (a.mag || 0))[0]?.id)
    } else if (want === 'fire' && fires.length) {
      lockOnly([...fires].sort((a, b) => (b.frp || 0) - (a.frp || 0))[0]?.id)
    } else if (want === 'storm' && storms.length) {
      lockOnly(pickStorm(storms)?.id)
    } else if (want === 'site' && sites.length) {
      lockOnly(pickSite(sites)?.id)
    } else if (want === 'radio' && radios.length) {
      lockOnly(pickRadio(radios)?.id)
    } else if (want === 'mission' && missions.length) {
      lockOnly(missions[0]?.id)
    } else if (want === 'iss') {
      const iss = findIss(sats) || findIss(satRef.current)
      if (!iss) return
      if (focusedLayerRef.current === 'sats') lockOnly(iss.id)
      else pendingLockRef.current = null
    }
  }, [quakes, fires, storms, sats, sites, radios, missions])

  const onSearch = async (q = search) => {
    const query = String(q || '').trim()
    if (query.length < 2) return
    setFlyHint('Geocoding…')
    const hits = await geocodePlace(query)
    if (!hits.length) {
      setFlyHint('No place match')
      return
    }
    const h = hits[0]
    flyTo(h.lat, h.lon, flyAltFor(query, h), h.label)
    return h
  }

  const annotatePlace = async (q) => {
    const hits = await geocodePlace(q)
    if (!hits.length) {
      setFlyHint('Nothing to mark')
      return
    }
    const h = hits[0]
    setAnnotations((prev) => [
      ...prev.slice(-11),
      { id: `ann-${Date.now()}`, label: h.label?.split(',')[0] || q, lat: h.lat, lon: h.lon, geojson: h.geojson },
    ])
    flyTo(h.lat, h.lon, 8_500, `Marked ${h.label?.split(',')[0] || q}`)
  }

  const measurePlaces = async (a, b) => {
    const [ha, hb] = await Promise.all([geocodePlace(a), geocodePlace(b)])
    if (!ha[0] || !hb[0]) {
      setFlyHint('Could not measure those places')
      return
    }
    const km = haversineKm(ha[0].lat, ha[0].lon, hb[0].lat, hb[0].lon)
    const midLat = (ha[0].lat + hb[0].lat) / 2
    const midLon = (ha[0].lon + hb[0].lon) / 2
    const pts = greatCirclePoints(ha[0].lat, ha[0].lon, hb[0].lat, hb[0].lon)
    setMeasure({ a: ha[0].label?.split(',')[0] || a, b: hb[0].label?.split(',')[0] || b, km })
    setAnnotations((prev) => [
      ...prev.slice(-11),
      {
        id: `meas-${Date.now()}`,
        label: `${km.toFixed(0)} km`,
        lat: midLat,
        lon: midLon,
        points: pts,
      },
    ])
    flyTo(midLat, midLon, Math.min(2_400_000, Math.max(80_000, km * 80)), `${km.toFixed(0)} km`)
  }

  const onReset = () => {
    setTrackedId(null)
    setReplay(null)
    flyTo(RESET_LOOK.lat, RESET_LOOK.lon, RESET_LOOK.alt, 'Full Earth')
    setSpeakLine('Full Earth.')
  }

  const onPick = (c) => {
    if (!c?.id) return
    pendingLockRef.current = null
    setReplay(null)
    setTrackedId(c.id)
    if (c.kind === 'cctv') {
      setCctvStill(c)
      return
    }
    const ground =
      c.kind === 'quake' ||
      c.kind === 'fire' ||
      c.kind === 'storm' ||
      c.kind === 'volcano' ||
      c.kind === 'flood' ||
      c.kind === 'dust' ||
      c.kind === 'alert' ||
      c.kind === 'site' ||
      c.kind === 'installation' ||
      c.kind === 'dam' ||
      c.kind === 'datacenter' ||
      c.kind === 'radio'
    if (ground) {
      setCamMode('free')
      setSpeakLine(c.callsign || c.id)
      const here = lookRef.current
      if (haversineKm(here.lat, here.lon, c.lat, c.lon) > 80) {
        flyTo(c.lat, c.lon, chaseAlt(c), c.callsign || c.id, { keepTrack: true })
      }
      return
    }
    if (c.kind === 'sat') {
      setCamMode('free')
      setSpeakLine(`Tracking ${c.callsign || c.id}.`)
      if ((lookRef.current.alt || 0) < 1_500_000) {
        flyTo(LAYER_SCENES.sats.lat, LAYER_SCENES.sats.lon, LAYER_SCENES.sats.alt, c.callsign || 'Sat', {
          keepTrack: true,
        })
      }
      return
    }
    if (c.kind === 'mission') {
      setCamMode('free')
      setSpeakLine(c.callsign || c.id)
      const here = lookRef.current
      if (haversineKm(here.lat, here.lon, c.lat, c.lon) > 80) {
        flyTo(c.lat, c.lon, chaseAlt(c), c.callsign || c.id, { keepTrack: true })
      }
      return
    }
    setCamMode(isAirborne(c) || c.kind === 'ship' ? 'orbit' : 'free')
  }

  const trackIss = () => {
    if (!layersOn.includes('sats')) setLayersOn((cur) => [...cur, 'sats'])
    const findIss = () =>
      sats.find((s) => /iss|zarya/i.test(s.callsign || s.id)) ||
      satRef.current.find((s) => /iss|zarya/i.test(s.callsign || s.id))
    const lock = (iss) => {
      setTrackedId(iss.id)
      setCamMode('free')
      if ((lookRef.current.alt || 0) < 1_500_000) {
        flyTo(LAYER_SCENES.sats.lat, LAYER_SCENES.sats.lon, LAYER_SCENES.sats.alt, iss.callsign || 'ISS', {
          keepTrack: true,
        })
      }
      setSpeakLine(`Tracking ${iss.callsign || 'ISS'}.`)
      speak(`Tracking ${iss.callsign || 'the ISS'}`)
    }
    const hit = findIss()
    if (hit) {
      lock(hit)
      return
    }
    setSpeakLine('Acquiring ISS…')
    let n = 0
    const id = setInterval(() => {
      const again = findIss()
      n += 1
      if (again) {
        clearInterval(id)
        lock(again)
      } else if (n >= 12) {
        clearInterval(id)
        setSpeakLine('ISS ephemeris still loading.')
      }
    }, 400)
  }

  const startReplay = (mission) => {
    setReplay({ id: mission.id, t: 0, rate: 1, playing: true })
    setTrackedId(`${mission.id}-ascent`)
    setCamMode('orbit')
    setSpeakLine(`Reconstructed ascent · ${mission.callsign}`)
  }

  const applyActions = async (actions) => {
    for (const a of actions || []) {
      if (a.type === 'reset') onReset()
      if (a.type === 'sensor' && a.sensor) setSensor(a.sensor)
      if (a.type === 'camera' && a.mode) setCamMode(a.mode)
      if (a.type === 'detection') setDetectionOn(a.on !== false)
      if (a.type === 'hud') setHudOn(a.on !== false)
      if (a.type === 'measure' && a.a && a.b) await measurePlaces(a.a, a.b)
      if (a.type === 'layer' && a.layer) {
        if (a.on === false) {
          setLayersOn((cur) => cur.filter((x) => x !== a.layer))
          setFocusedLayer((cur) => (cur === a.layer ? null : cur))
          if (a.layer === 'cctv') setCctvStill(null)
        } else {
          stageLayer(a.layer)
        }
      }
      if (a.type === 'track_iss') trackIss()
      if (a.type === 'track_nearest') {
        const kind = a.kind
        const pool = nearby.filter((c) => {
          if (kind === 'mil') return c.military || c.kind === 'mil'
          return c.kind === kind
        })
        const hit = pool[0] || contacts.find((c) => (kind === 'mil' ? c.military : c.kind === kind))
        if (hit) onPick(hit)
      }
      if (a.type === 'fly_to' && a.q) await onSearch(a.q)
      if (a.type === 'annotate' && a.q) await annotatePlace(a.q)
    }
  }

  const handleUtterance = async (text) => {
    const q = String(text || '').trim()
    if (!q) return
    const qLow = q.toLowerCase()
    const namedCity = CCTV_CITIES.find((c) => qLow === c.id.toLowerCase() || qLow.includes(c.id.toLowerCase()))
    if (namedCity && (qLow === namedCity.id.toLowerCase() || /cam|cctv|viewshed/.test(qLow))) {
      flyCctvCity(namedCity.id)
      return
    }
    if (/^(cctv|cameras|public cameras)$/i.test(q)) {
      flyCctvCity('Austin')
      return
    }
    setSpeakLine('Working…')
    try {
      const doc = await runCommand(q, lookRef.current)
      setSpeakLine(doc.speak || '')
      speak(doc.speak || '')
      await applyActions(doc.actions || [])
    } catch {
      const h = await onSearch(q)
      setSpeakLine(h ? `Flying to ${h.label}` : 'Command failed')
    }
  }

  const onVoice = () => {
    const SR = typeof window !== 'undefined' && (window.SpeechRecognition || window.webkitSpeechRecognition)
    if (!SR) {
      const typed = window.prompt('Voice is not available here. Type a command:')
      if (typed) handleUtterance(typed)
      return
    }
    if (recRef.current && listening) {
      recRef.current.stop()
      return
    }
    const rec = new SR()
    rec.lang = 'en-US'
    rec.interimResults = true
    rec.continuous = false
    recRef.current = rec
    rec.onstart = () => setListening(true)
    rec.onend = () => setListening(false)
    rec.onerror = () => setListening(false)
    rec.onresult = (ev) => {
      const last = ev.results[ev.results.length - 1]
      const text = last?.[0]?.transcript || ''
      setTranscript(text)
      if (last?.isFinal) handleUtterance(text)
    }
    rec.start()
  }

  const onTune = (station) => {
    if (!audioRef.current) audioRef.current = new Audio()
    if (radioUrl === station.url) {
      audioRef.current.pause()
      setRadioUrl('')
      return
    }
    audioRef.current.src = station.url
    audioRef.current.play().catch(() => setSpeakLine('Browser blocked the stream.'))
    setRadioUrl(station.url)
    setSpeakLine(`Tuned ${station.callsign}`)
    flyTo(station.lat, station.lon, 8_200, station.callsign)
  }

  const onMission = (id) => {
    try {
      localStorage.setItem('gev-mission', '1')
    } catch {
      /* ignore */
    }
    setMissionOpen(false)
    if (id === 'contacts') {
      stageLayer('flights')
      flyTo(HOME.lat, HOME.lon, HOME.alt, 'Live contacts · LAX')
      setSpeakLine('Live contacts over Los Angeles.')
    } else if (id === 'cctv') {
      setViewshedOn(true)
      setLayersOn(['cctv', 'traffic', 'flights'])
      setFocusedLayer('cctv')
      flyCctvCity('Austin')
      setSpeakLine('Public cameras over Austin.')
    } else if (id === 'space') {
      stageLayer('missions')
    } else if (id === 'enviro') {
      stageLayer('fires')
    }
  }

  useEffect(() => {
    const onKey = (e) => {
      if (e.target.matches?.('input, textarea')) return
      if (e.key >= '1' && e.key <= '7') {
        const next = SENSORS[Number(e.key) - 1]
        if (next) setSensor(next)
      } else if (e.key === 'h' || e.key === 'H') {
        setHudOn((v) => !v)
      } else if (e.key === 'd' || e.key === 'D') {
        setDetectionOn((v) => !v)
      } else if (e.key === 'c' || e.key === 'C') {
        setCamMode((m) => (m === 'cockpit' ? 'free' : 'cockpit'))
      } else if (e.key === 'Escape') {
        if (missionOpen) setMissionOpen(false)
        else if (cctvStill) setCctvStill(null)
        else if (camModeRef.current === 'cockpit') setCamMode('free')
        else if (trackedId) {
          setTrackedId(null)
          setReplay(null)
        }
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [missionOpen, cctvStill, trackedId])

  const tilesAvailable = Boolean(config?.hasPhotoreal || config?.googleMapsApiKey || config?.cesiumIonToken)
  // First paint: satellite marble only. Photoreal tiles are a second mesh stack — opt-in HUD chip.
  const tilesOn = Boolean(tilesWanted && tilesAvailable && tilesAltitudeOk(look.alt))
  const cockpit = camMode === 'cockpit' && !!tracked

  useEffect(() => {
    if (!tilesOn) setTilesStatus({ active: false, mode: 'none' })
  }, [tilesOn])

  useEffect(() => {
    const onLost = () => {
      setTilesWanted(false)
      setTilesStatus({ active: false, mode: 'none', error: 'webgl-lost' })
    }
    window.addEventListener('webglcontextlost', onLost, true)
    return () => window.removeEventListener('webglcontextlost', onLost, true)
  }, [])
  const replayTrail = useMemo(() => {
    if (!replay) return []
    const pad = missions.find((m) => m.id === replay.id)
    if (!pad) return []
    return ascentTrail(pad.lat, pad.lon, pad.heading || 90)
  }, [replay, missions])

  const detectContacts = useMemo(() => {
    if (!tilesAltitudeOk(look.alt) || focusedLayer === 'sats' || focusedLayer === 'missions') return []
    const pool = nearby.filter(
      (c) =>
        c.id !== trackedId &&
        contactMatchesFocus(focusedLayer, c) &&
        (isAirborne(c) || c.kind === 'ship')
    )
    if (cockpit) return pool.filter((c) => (c._km || 0) < 80).slice(0, 6)
    return pool.slice(0, 14)
  }, [nearby, look.alt, cockpit, trackedId, focusedLayer])

  const drawContacts = useMemo(() => {
    const origin = tracked || look
    const scored = contacts
      .filter((c) => c.kind !== 'cctv')
      .filter((c) => c.id === trackedId || contactMatchesFocus(focusedLayer, c))
      .filter((c) => {
        if (c.id === trackedId) return true
        if (c.kind === 'flight' || c.military) return isAirborne(c)
        return true
      })
      .map((c) => ({
        c,
        km: nearbyKm.get(c.id) ?? haversineKm(origin.lat, origin.lon, c.lat, c.lon),
        pri: layerContactPriority(focusedLayer, c),
      }))
      .sort((a, b) => a.pri - b.pri || a.km - b.km)
    const keep = []
    const seen = new Set()
    const take = (row) => {
      if (seen.has(row.c.id)) return
      seen.add(row.c.id)
      keep.push(row.c)
    }
    const locked = scored.find((row) => row.c.id === trackedId || row.c.id === replayContact?.id)
    if (locked) take(locked)
    for (const row of scored) {
      if (keep.length >= glyphBudget(look.alt)) break
      take(row)
    }
    return keep
  }, [contacts, nearbyKm, tracked, look, trackedId, replayContact, focusedLayer])

  return (
    <div className={`godseye ge-cockpit layout-probe-cockpit sensor-${sensor} ${cockpit ? 'is-cockpit' : ''}`}>
      <Canvas
        className="ge-globe"
        dpr={[1, 1.25]}
        camera={{
          position: [startPos.x, startPos.y, startPos.z],
          fov: cityFov(startLook.alt, cockpit),
          near: 12,
          far: EARTH_R * 40,
        }}
        gl={{ antialias: false, logarithmicDepthBuffer: true, alpha: false, powerPreference: 'high-performance' }}
        style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', touchAction: 'none', cursor: 'grab', zIndex: 0, pointerEvents: 'auto' }}
        onCreated={({ gl, camera }) => {
          gl.setClearColor('#010103', 1)
          const tgt = new THREE.Vector3()
          const goal = new THREE.Vector3()
          scenePose(startLook.lat, startLook.lon, startLook.alt, goal, tgt)
          camera.position.copy(goal)
          camera.lookAt(tgt)
        }}
      >
        <hemisphereLight args={['#e8f2ff', '#3a2e22', 1.25]} />
        <ambientLight intensity={1.2} />
        <directionalLight position={[EARTH_R * 4, EARTH_R * 2, EARTH_R]} intensity={2.8} color="#fff6e8" />
        <GlobeEarth
          sensor={sensor}
          showSurface={!(tilesOn && tilesStatus?.active && look.alt < 22_000)}
          scale={1}
        />
        {lookPatchOk(sceneLook.alt) && !(tilesOn && tilesStatus?.active) && (
          <LookPatch lat={sceneLook.lat} lon={sceneLook.lon} alt={sceneLook.alt} enabled />
        )}
        {tilesOn && (
          <React.Suspense fallback={null}>
            <TilesGlobe
              config={config}
              enabled={tilesOn}
              onStatus={setTilesStatus}
              quality={look.alt < 4_000 ? 16 : look.alt < 14_000 ? 18 : 22}
            />
          </React.Suspense>
        )}
        {layersOn.includes('sats') && <SatRings />}
        {layersOn.includes('cctv') && (
          <CctvMesh
            cameras={cctv}
            selectedId={tracked?.kind === 'cctv' ? tracked.id : null}
            onPick={onPick}
            lookAlt={look.alt}
            origin={look}
            viewshedOn={viewshedOn}
          />
        )}
        <group key={`glyphs-${focusedLayer || 'none'}-${layersOn.join('+')}`}>
        {drawContacts.map((c) => (
            <ContactGlyph
              key={c.id}
              contact={nearbyKm.has(c.id) ? { ...c, _km: nearbyKm.get(c.id) } : c}
              selected={c.id === trackedId || c.id === replayContact?.id}
              onPick={onPick}
              sensor={sensor}
              showLabel={
                c.id === trackedId ||
                (focusedLayer === 'traffic' && c.kind === 'traffic' && (nearbyKm.get(c.id) || 999) < 3) ||
                ((focusedLayer === 'flights' || focusedLayer === 'military' || focusedLayer === 'ships') &&
                  (c.kind === 'flight' || c.military || c.kind === 'ship') &&
                  (nearbyKm.get(c.id) || 999) < 80) ||
                ((focusedLayer === 'sats' || focusedLayer === 'missions') &&
                  (c.kind === 'sat' || c.kind === 'mission') &&
                  (c.id === trackedId ||
                    /iss|zarya/i.test(c.callsign || c.id) ||
                    c.kind === 'mission')) ||
                (focusedLayer === 'quakes' && c.kind === 'quake' && ((c.mag || 0) >= 5.5 || c.id === trackedId)) ||
                (focusedLayer === 'fires' && c.kind === 'fire' && ((c.frp || 0) >= 60 || c.id === trackedId)) ||
                (focusedLayer === 'storms' &&
                  ['storm', 'volcano', 'flood', 'dust', 'alert'].includes(c.kind) &&
                  (c.id === trackedId || c.kind === 'storm' || (c.mag || 0) >= 40)) ||
                (focusedLayer === 'sites' && isGroundIntel(c) && c.kind !== 'quake' && c.kind !== 'fire') ||
                (focusedLayer === 'radio' && c.kind === 'radio')
              }
            />
          ))}
        </group>
        {annotations.map((a) => (
          <Annotation key={a.id} item={a} />
        ))}
        {tracked && <Trail points={trail} />}
        {replayTrail.length > 1 && <Trail points={replayTrail} color="#ff9f43" />}
        <SensorScreen mode={sensor} />
        {detectionOn && (
          <DetectionProjector
            key={`det-${focusedLayer || 'none'}`}
            contacts={detectContacts}
            selectedId={trackedId}
            onBoxes={setBoxes}
          />
        )}
        <CameraRig tracked={tracked} camMode={camMode} draggingRef={draggingRef} holdRef={restageHoldRef} />
        <React.Fragment key={sceneKey}>
          <OrbitControls
            makeDefault
            enableDamping
            dampingFactor={0.1}
            enablePan={false}
            enableZoom
            enableRotate
            zoomSpeed={1.2}
            rotateSpeed={0.68}
            minPolarAngle={0.08}
            maxPolarAngle={Math.PI - 0.08}
            minDistance={flyMinDistance(sceneLook.alt)}
            maxDistance={EARTH_R * 8}
            enabled
            onStart={() => {
              draggingRef.current = true
              setCamMode((m) => (m === 'orbit' ? 'free' : m))
            }}
            onEnd={() => {
              draggingRef.current = false
            }}
          />
          <CameraBoot lat={sceneLook.lat} lon={sceneLook.lon} alt={sceneLook.alt} />
        </React.Fragment>
        <ControlsLimiter
          sceneAlt={sceneLook.alt}
          sceneLat={sceneLook.lat}
          sceneLon={sceneLook.lon}
          holdRef={restageHoldRef}
        />
        <LookReporter onLook={setLook} />
      </Canvas>
      <div
        className="ge-hud"
        onPointerDownCapture={(e) => {
          if (e.target.closest?.('button, input, select, textarea, a, .ge-layers, .ge-right-rail, .ge-roster-dock, .ge-card, .ge-strip, .ge-mission, .ge-cctv, .ge-tuner, .ge-search')) {
            draggingRef.current = false
          }
        }}
      >
        <div className="ge-scan" aria-hidden />
      <CockpitHud
        look={look}
        heading={look.heading || 0}
        sensor={sensor}
        setSensor={setSensor}
        layersOn={layersOn}
        toggleLayer={toggleLayer}
        focusedLayer={focusedLayer}
        layerMeta={layerMeta}
        tracked={tracked}
        nearby={nearby}
        rosterKm={rosterKm}
        onSelect={onPick}
        onClearTrack={() => {
          setTrackedId(null)
          setCamMode('free')
          setReplay(null)
        }}
        search={search}
        setSearch={setSearch}
        onSearch={() => handleUtterance(search)}
        onReset={onReset}
        tilesStatus={{
          ...tilesStatus,
          pending: Boolean(tilesOn && !tilesStatus?.active && !tilesStatus?.error),
          available: tilesAvailable,
          wanted: tilesWanted,
        }}
        onToggleTiles={() => {
          if (!tilesAvailable) return
          setTilesWanted((v) => !v)
        }}
        flyHint={flyHint}
        camMode={camMode}
        setCamMode={setCamMode}
        latencyMs={latencyMs}
        voiceOn={listening}
        onVoice={onVoice}
        listening={listening}
        transcript={transcript}
        speakLine={speakLine}
        mapsKey={config?.googleMapsApiKey || ''}
        onTrackIss={trackIss}
        onOrbit={() => {
          setCamMode((m) => (m === 'orbit' ? 'free' : 'orbit'))
        }}
        radioUrl={radioUrl}
        onTune={onTune}
        radios={radios}
        hudOn={hudOn}
        detectionOn={detectionOn}
        setDetectionOn={setDetectionOn}
        missionOpen={missionOpen}
        onMission={onMission}
        dismissMission={() => {
          try {
            localStorage.setItem('gev-mission', '1')
          } catch {
            /* ignore */
          }
          setMissionOpen(false)
        }}
        boxes={
          detectionOn
            ? boxes.filter((b) => contactMatchesFocus(focusedLayer, b.contact || b))
            : []
        }
        cctvStill={cctvStill}
        onCloseCctv={() => setCctvStill(null)}
        cctvCity={cctvCity}
        onCctvCity={flyCctvCity}
        viewshedOn={viewshedOn}
        setViewshedOn={setViewshedOn}
        measure={measure}
        replay={replay}
        onReplay={startReplay}
        onReplayRate={setReplay}
        utc={utc}
        onEnterRange={onEnterRange}
      />
      </div>
    </div>
  )
}
