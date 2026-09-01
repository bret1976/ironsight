import React, { Suspense, useEffect, useMemo, useState } from 'react'
import { DRACOLoader } from 'three/examples/jsm/loaders/DRACOLoader.js'

/**
 * Photoreal 3D tiles with the plugins Google meshes actually need.
 * ECEF (Z-up) is rotated -90° about X to match the Y-up SphereGeometry globe.
 */
const DRACO_PATH = 'https://www.gstatic.com/draco/versioned/decoders/1.5.7/'

function useDraco() {
  return useMemo(() => {
    const loader = new DRACOLoader()
    loader.setDecoderPath(DRACO_PATH)
    loader.setDecoderConfig({ type: 'js' })
    loader.preload()
    return loader
  }, [])
}

function unlitScene(scene) {
  if (!scene) return
  scene.traverse((obj) => {
    if (!obj.isMesh || !obj.material) return
    const mats = Array.isArray(obj.material) ? obj.material : [obj.material]
    for (const mat of mats) {
      if (!mat) continue
      if ('metalness' in mat) mat.metalness = 0
      if ('roughness' in mat) mat.roughness = 1
      if ('envMapIntensity' in mat) mat.envMapIntensity = 0
      mat.toneMapped = false
      mat.needsUpdate = true
    }
    obj.castShadow = false
    obj.receiveShadow = false
  })
}

export default function TilesGlobe({ config, enabled, onStatus, quality = 18 }) {
  const [mod, setMod] = useState(null)
  const token = (config?.googleMapsApiKey || config?.cesiumIonToken || '').trim()
  const mode = config?.googleMapsApiKey ? 'google' : config?.cesiumIonToken ? 'cesium' : 'none'
  const draco = useDraco()

  useEffect(() => {
    if (!enabled || !token || mode === 'none') {
      onStatus?.({ active: false, mode: 'none' })
      return
    }
    let cancelled = false
    // Let the HUD commit before the tiles parser takes the main thread.
    const arm = window.setTimeout(() => {
      import('3d-tiles-renderer/r3f')
        .then((r3f) => import('3d-tiles-renderer/plugins').then((plugins) => ({ r3f, plugins })))
        .then((loaded) => {
          if (!cancelled) {
            setMod(loaded)
            onStatus?.({ active: true, mode })
          }
        })
        .catch((e) => {
          console.warn('GodsEye 3D tiles failed to load', e)
          onStatus?.({ active: false, mode: 'none', error: String(e.message || e) })
        })
    }, 700)
    return () => {
      cancelled = true
      window.clearTimeout(arm)
    }
  }, [enabled, token, mode, onStatus])

  useEffect(() => () => draco.dispose(), [draco])

  if (!enabled || !token || !mod) return null

  const { TilesRenderer, TilesPlugin, TilesAttributionOverlay } = mod.r3f
  const { GoogleCloudAuthPlugin, CesiumIonAuthPlugin, GLTFExtensionsPlugin, TilesFadePlugin, UpdateOnChangePlugin, UnloadTilesPlugin } =
    mod.plugins

  return (
    <group rotation={[-Math.PI / 2, 0, 0]}>
      <Suspense fallback={null}>
        <TilesRenderer
          errorTarget={quality}
          maxDepth={18}
          maxTilesProcessed={16}
          loadSiblings={false}
          lruCache-minSize={80}
          lruCache-maxSize={220}
          lruCache-maxBytesSize={2.8e8}
          downloadQueue-maxJobsPerOrigin={2}
          parseQueue-maxJobs={1}
          onLoadModel={(ev) => unlitScene(ev?.scene || ev)}
        >
          {mode === 'google' ? (
            <TilesPlugin
              plugin={GoogleCloudAuthPlugin}
              args={{ apiToken: token, autoRefreshToken: true, useRecommendedSettings: false }}
            />
          ) : (
            <TilesPlugin plugin={CesiumIonAuthPlugin} args={{ apiToken: token, assetId: '2275207' }} />
          )}
          <TilesPlugin plugin={GLTFExtensionsPlugin} args={{ dracoLoader: draco, rtc: true }} />
          <TilesPlugin plugin={UpdateOnChangePlugin} />
          <TilesPlugin plugin={TilesFadePlugin} />
          <TilesPlugin plugin={UnloadTilesPlugin} />
          {TilesAttributionOverlay ? (
            <TilesAttributionOverlay
              style={{
                position: 'absolute',
                left: 12,
                bottom: 72,
                color: '#9ab',
                fontSize: 10,
                pointerEvents: 'none',
              }}
            />
          ) : null}
        </TilesRenderer>
      </Suspense>
    </group>
  )
}
