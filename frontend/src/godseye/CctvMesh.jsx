import React, { useMemo } from 'react'
import { Line } from '@react-three/drei'
import * as THREE from 'three'
import { headingVector, latLonAltToXYZ, localFrame } from './geo.js'
import { CCTV_SCENE_ALT_M } from './scenes.js'

export const CCTV_CITIES = [
  { id: 'Austin', lat: 30.2672, lon: -97.7431, alt: CCTV_SCENE_ALT_M, blurb: 'Austin Mobility cameras' },
  { id: 'London', lat: 51.5074, lon: -0.1278, alt: CCTV_SCENE_ALT_M, blurb: 'TfL JamCams' },
  { id: 'California', lat: 34.0522, lon: -118.2437, alt: CCTV_SCENE_ALT_M, blurb: 'Caltrans roadside' },
]

function mountXYZ(cam) {
  return latLonAltToXYZ(cam.lat, cam.lon, Math.max(8, cam.alt_m || 14))
}

function viewshedLines(cam, rangeM = 220) {
  const origin = mountXYZ(cam)
  const { radial } = localFrame(cam.lat, cam.lon)
  const fwd = headingVector(cam.lat, cam.lon, cam.heading || 0)
  const f = new THREE.Vector3(fwd.x, fwd.y, fwd.z).normalize()
  const up = new THREE.Vector3(radial.x, radial.y, radial.z)
  const right = new THREE.Vector3().crossVectors(f, up).normalize()
  const o = new THREE.Vector3(origin.x, origin.y, origin.z)
  const far = o.clone().addScaledVector(f, rangeM).addScaledVector(up, -18)
  const corners = [
    far.clone().addScaledVector(right, 70).addScaledVector(up, 28),
    far.clone().addScaledVector(right, -70).addScaledVector(up, 28),
    far.clone().addScaledVector(right, -70).addScaledVector(up, -22),
    far.clone().addScaledVector(right, 70).addScaledVector(up, -22),
  ]
  const segs = []
  for (const c of corners) segs.push([o.clone(), c])
  segs.push([corners[0], corners[1], corners[2], corners[3], corners[0]])
  return { origin: o, far, corners, segs, quat: new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().lookAt(o, far, up)) }
}

function CamGlyph({ cam, selected, onPick, viewshed }) {
  const geo = useMemo(() => viewshedLines(cam), [cam])
  return (
    <group>
      <mesh
        position={geo.origin}
        onClick={(e) => {
          e.stopPropagation()
          onPick(cam)
        }}
        onPointerDown={(e) => {
          e.stopPropagation()
          onPick(cam)
        }}
      >
        <sphereGeometry args={[48, 12, 12]} />
        <meshBasicMaterial transparent opacity={0} depthWrite={false} depthTest={false} />
      </mesh>
      <mesh position={geo.origin}>
        <boxGeometry args={[10, 7, 16]} />
        <meshBasicMaterial color={selected ? '#7ee0ff' : '#f4f6f8'} transparent opacity={0.96} />
      </mesh>
      {(selected || viewshed) &&
        geo.segs.map((pts, i) => (
          <Line
            key={i}
            points={pts}
            color={selected ? '#e8eef2' : '#c5ced4'}
            dashed
            dashSize={10}
            gapSize={7}
            transparent
            opacity={selected ? 0.85 : 0.38}
            lineWidth={1}
          />
        ))}
      {selected && (
        <mesh position={geo.far} quaternion={geo.quat}>
          <planeGeometry args={[150, 84]} />
          <meshBasicMaterial color="#0a1218" transparent opacity={0.22} side={THREE.DoubleSide} />
        </mesh>
      )}
    </group>
  )
}

export default function CctvMesh({ cameras, selectedId, onPick, lookAlt, origin, viewshedOn }) {
  const nearby = useMemo(() => {
    if (!cameras?.length) return []
    if (lookAlt > 40_000) {
      const by = {}
      for (const cam of cameras) {
        const city = cam.city || 'other'
        ;(by[city] ||= []).push(cam)
      }
      const out = []
      for (const rows of Object.values(by)) out.push(...rows.slice(0, 8))
      return out
    }
    const ranked = cameras
      .map((c) => ({
        c,
        d: Math.hypot((c.lat - (origin?.lat || 0)) * 111, (c.lon - (origin?.lon || 0)) * 90),
      }))
      .sort((a, b) => a.d - b.d)
    return ranked.filter((r) => r.d < 16).slice(0, 22).map((r) => r.c)
  }, [cameras, lookAlt, origin])

  const shedIds = useMemo(() => {
    const ids = new Set()
    if (selectedId) ids.add(selectedId)
    if (viewshedOn) nearby.slice(0, 5).forEach((c) => ids.add(c.id))
    return ids
  }, [nearby, selectedId, viewshedOn])

  return (
    <group>
      {nearby.map((cam) => (
        <CamGlyph
          key={cam.id}
          cam={cam}
          selected={cam.id === selectedId}
          onPick={onPick}
          viewshed={shedIds.has(cam.id)}
        />
      ))}
    </group>
  )
}
