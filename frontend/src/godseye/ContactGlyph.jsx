import React, { useMemo, useRef } from 'react'
import { Html, Line } from '@react-three/drei'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { headingVector, latLonAltToXYZ, localFrame } from './geo.js'
import { isAirborne } from './layers.js'
import { glyphScaleM } from './scenes.js'

const KIND_COLOR = {
  flight: '#7ecbff',
  mil: '#ffd34d',
  sat: '#e8c547',
  quake: '#e5484d',
  ship: '#4dd0e1',
  fire: '#ff5a1f',
  storm: '#b794f6',
  volcano: '#e879f9',
  flood: '#818cf8',
  dust: '#f6ad55',
  alert: '#f6ad55',
  installation: '#c9a227',
  dam: '#7dd3fc',
  datacenter: '#c084fc',
  site: '#c9a227',
  radio: '#f0abfc',
  mission: '#f5d76e',
  cctv: '#f4f6f8',
  traffic: '#9ae6b4',
}

const UNIT = 1
const _fwd = new THREE.Vector3()
const _up = new THREE.Vector3()
const _aim = new THREE.Vector3()
const _dummy = new THREE.Object3D()

function contactXYZ(c) {
  const lift =
    c.kind === 'quake'
      ? 4000
      : c.kind === 'fire'
        ? 2500
        : c.kind === 'storm' || c.kind === 'volcano' || c.kind === 'flood' || c.kind === 'dust' || c.kind === 'alert'
          ? 3200
          : c.kind === 'traffic'
            ? 8
            : c.kind === 'ship'
              ? 55
              : c.kind === 'mission'
                ? 110
                : 0
  return latLonAltToXYZ(c.lat, c.lon, Math.max(0, c.alt_m || 0) + lift)
}

/** Airliner — nose along -Z so Object3D.lookAt points it at heading. */
function PlaneMesh({ color }) {
  const mat = useMemo(
    () => new THREE.MeshBasicMaterial({ color, depthWrite: false, transparent: true, opacity: 0.96 }),
    [color]
  )
  return (
    <group>
      <mesh material={mat} rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.11, 0.085, 1.55, 8]} />
      </mesh>
      <mesh material={mat} position={[0, 0, -0.96]} rotation={[-Math.PI / 2, 0, 0]}>
        <coneGeometry args={[0.085, 0.36, 8]} />
      </mesh>
      <mesh material={mat} position={[0, -0.01, 0.04]}>
        <boxGeometry args={[2.2, 0.04, 0.38]} />
      </mesh>
      <mesh material={mat} position={[0.7, -0.09, 0.04]} rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.05, 0.05, 0.32, 6]} />
      </mesh>
      <mesh material={mat} position={[-0.7, -0.09, 0.04]} rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.05, 0.05, 0.32, 6]} />
      </mesh>
      <mesh material={mat} position={[0, 0.24, 0.62]}>
        <boxGeometry args={[0.045, 0.4, 0.26]} />
      </mesh>
      <mesh material={mat} position={[0, 0.03, 0.7]}>
        <boxGeometry args={[0.58, 0.03, 0.16]} />
      </mesh>
    </group>
  )
}

function ShipMesh({ color }) {
  const mat = useMemo(
    () => new THREE.MeshBasicMaterial({ color, depthWrite: false, transparent: true, opacity: 0.95 }),
    [color]
  )
  return (
    <group>
      <mesh material={mat} position={[0, 0.08, 0]}>
        <boxGeometry args={[0.38, 0.22, 1.7]} />
      </mesh>
      <mesh material={mat} position={[0, 0.08, 0.78]} rotation={[Math.PI / 2, 0, 0]}>
        <coneGeometry args={[0.19, 0.36, 6]} />
      </mesh>
      <mesh material={mat} position={[0, 0.28, -0.16]}>
        <boxGeometry args={[0.26, 0.28, 0.5]} />
      </mesh>
    </group>
  )
}

function RingMesh({ color }) {
  return (
    <group>
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <circleGeometry args={[0.38, 24]} />
        <meshBasicMaterial color={color} transparent opacity={0.42} side={THREE.DoubleSide} depthWrite={false} />
      </mesh>
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.42, 0.92, 36]} />
        <meshBasicMaterial color={color} transparent opacity={0.94} side={THREE.DoubleSide} depthWrite={false} />
      </mesh>
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <ringGeometry args={[1.05, 1.22, 36]} />
        <meshBasicMaterial color={color} transparent opacity={0.38} side={THREE.DoubleSide} depthWrite={false} />
      </mesh>
    </group>
  )
}

/** Pad / tower — not a screen-filling octahedron. */
function PadMesh({ color }) {
  const mat = useMemo(
    () => new THREE.MeshBasicMaterial({ color, depthWrite: false, transparent: true, opacity: 0.96 }),
    [color]
  )
  return (
    <group>
      <mesh material={mat} position={[0, 0.06, 0]}>
        <boxGeometry args={[1.15, 0.12, 1.15]} />
      </mesh>
      <mesh material={mat} position={[0, 0.62, 0]}>
        <boxGeometry args={[0.16, 1.05, 0.16]} />
      </mesh>
      <mesh material={mat} position={[0.28, 0.28, 0.28]}>
        <boxGeometry args={[0.22, 0.36, 0.22]} />
      </mesh>
    </group>
  )
}

export default function ContactGlyph({ contact, selected, onPick, sensor, showLabel }) {
  const group = useRef()
  const flying = isAirborne(contact) || contact.kind === 'sat' || contact.reconstructed
  const color = KIND_COLOR[contact.military ? 'mil' : contact.kind] || '#fff'
  const tint = sensor === 'nvg' ? '#3dff8a' : sensor === 'flir' || sensor === 'ironbell' ? '#ff9a3c' : color
  const seaborne = contact.kind === 'ship'
  const moving = flying || seaborne || contact.kind === 'traffic' || contact.kind === 'flight' || contact.military

  useFrame(({ camera }) => {
    if (!group.current) return
    const p = contactXYZ(contact)
    group.current.position.set(p.x, p.y, p.z)
    const dist = camera.position.distanceTo(group.current.position)
    group.current.scale.setScalar(glyphScaleM(contact.military ? 'mil' : contact.kind, dist, selected))
    if (moving) {
      const { radial } = localFrame(contact.lat, contact.lon)
      const fwd = headingVector(contact.lat, contact.lon, contact.heading || 0)
      _fwd.set(fwd.x, fwd.y, fwd.z).normalize()
      _up.set(radial.x, radial.y, radial.z).normalize()
      _aim.copy(group.current.position).add(_fwd)
      _dummy.position.copy(group.current.position)
      _dummy.up.copy(_up)
      _dummy.lookAt(_aim)
      group.current.quaternion.copy(_dummy.quaternion)
    }
  })

  const altFt = contact.alt_m != null ? Math.round((contact.alt_m * 3.28084) / 100) * 100 : null
  const spd = contact.gs_kts != null ? Math.round(contact.gs_kts) : null
  const drop = flying && (contact.alt_m || 0) > 80
  const stem = useMemo(() => {
    if (!drop) return null
    const top = latLonAltToXYZ(contact.lat, contact.lon, contact.alt_m)
    const bot = latLonAltToXYZ(contact.lat, contact.lon, 0)
    return [
      [bot.x, bot.y, bot.z],
      [top.x, top.y, top.z],
    ]
  }, [drop, contact.lat, contact.lon, contact.alt_m])

  return (
    <group>
      {stem && (
        <Line points={stem} color={tint} transparent opacity={selected ? 0.55 : 0.2} lineWidth={1} />
      )}
      <group
        ref={group}
        onClick={(e) => {
          e.stopPropagation()
          onPick?.(contact)
        }}
        onPointerDown={(e) => {
          e.stopPropagation()
          onPick?.(contact)
        }}
      >
        <mesh>
          <sphereGeometry args={[contact.kind === 'mission' ? 2.8 : 1.15, 12, 12]} />
          <meshBasicMaterial transparent opacity={0} depthWrite={false} depthTest={false} />
        </mesh>
        {contact.kind === 'flight' || contact.military ? (
          <PlaneMesh color={tint} />
        ) : contact.kind === 'ship' ? (
          <ShipMesh color={tint} />
        ) : contact.kind === 'mission' ? (
          <PadMesh color={tint} />
        ) : contact.kind === 'quake' ||
          contact.kind === 'fire' ||
          contact.kind === 'storm' ||
          contact.kind === 'volcano' ||
          contact.kind === 'flood' ||
          contact.kind === 'dust' ||
          contact.kind === 'alert' ? (
          <RingMesh color={tint} />
        ) : (
          <mesh>
            <octahedronGeometry args={[0.48, 0]} />
            <meshBasicMaterial color={tint} transparent opacity={0.92} depthWrite={false} />
          </mesh>
        )}
        {showLabel && (
          <Html position={[0, 1.1, 0]} pointerEvents="none" style={{ pointerEvents: 'none' }} zIndexRange={[8, 0]}>
            <div className={`ge-track-tag ${selected ? 'on' : ''} ${contact.military ? 'mil' : contact.kind}`}>
              <b>{contact.callsign || contact.id}</b>
              {(altFt != null || spd != null) && (
                <i>
                  {altFt != null ? `ALT ${altFt >= 1000 ? `${Math.round(altFt / 1000)}K` : altFt}` : ''}
                  {altFt != null && spd != null ? ' · ' : ''}
                  {spd != null ? `${spd}KT` : ''}
                </i>
              )}
            </div>
          </Html>
        )}
      </group>
    </group>
  )
}

export { contactXYZ, KIND_COLOR, UNIT }
