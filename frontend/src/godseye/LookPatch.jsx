import React, { useEffect, useMemo, useState } from 'react'
import * as THREE from 'three'
import { latLonAltToXYZ, localFrame } from './geo.js'
import { satelliteStillUrl } from './layers.js'
import { lookPatchOk, satstillZoomForAlt } from './scenes.js'

/**
 * Local satellite still on the globe. City-scale marble is one dark texel;
 * this existing satstill feed is the ocean/street picture without a 3D-tile stack.
 */
export default function LookPatch({ lat, lon, alt, enabled }) {
  const zoom = satstillZoomForAlt(alt)
  const src = enabled && lookPatchOk(alt) && lat != null && lon != null ? satelliteStillUrl(lat, lon, '', zoom) : ''
  const [tex, setTex] = useState(null)

  useEffect(() => {
    if (!src) {
      setTex((prev) => {
        prev?.dispose?.()
        return null
      })
      return undefined
    }
    let cancelled = false
    const loader = new THREE.TextureLoader()
    loader.setCrossOrigin('anonymous')
    loader.load(
      src,
      (t) => {
        if (cancelled) {
          t.dispose()
          return
        }
        t.colorSpace = THREE.SRGBColorSpace
        t.anisotropy = 4
        t.needsUpdate = true
        setTex((prev) => {
          prev?.dispose?.()
          return t
        })
      },
      undefined,
      () => {
        if (!cancelled) {
          setTex((prev) => {
            prev?.dispose?.()
            return null
          })
        }
      }
    )
    return () => {
      cancelled = true
    }
  }, [src])

  useEffect(
    () => () => {
      tex?.dispose?.()
    },
    [tex]
  )

  const pose = useMemo(() => {
    if (lat == null || lon == null || !Number.isFinite(Number(lat)) || !Number.isFinite(Number(lon))) return null
    const p = latLonAltToXYZ(lat, lon, 220)
    const { east, north, radial } = localFrame(lat, lon)
    const degPerPx = 360 / (256 * 2 ** zoom)
    const clat = Math.max(0.2, Math.cos((Number(lat) * Math.PI) / 180))
    const widthM = Math.min(90_000, 1280 * degPerPx * 111_320 * clat)
    const heightM = Math.min(52_000, 720 * degPerPx * 110_540)
    const e = new THREE.Vector3(east.x, east.y, east.z).normalize()
    const n = new THREE.Vector3(north.x, north.y, north.z).normalize()
    const r = new THREE.Vector3(radial.x, radial.y, radial.z).normalize()
    const quat = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(e, n, r))
    return { position: [p.x, p.y, p.z], quat, widthM, heightM }
  }, [lat, lon, zoom])

  if (!enabled || !tex || !pose) return null
  return (
    <mesh position={pose.position} quaternion={pose.quat} renderOrder={2} raycast={() => null}>
      <planeGeometry args={[pose.widthM, pose.heightM]} />
      <meshBasicMaterial
        map={tex}
        toneMapped={false}
        depthWrite
        polygonOffset
        polygonOffsetFactor={-2}
        polygonOffsetUnits={-2}
      />
    </mesh>
  )
}
