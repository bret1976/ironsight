import React, { useEffect, useMemo, useRef, useState } from 'react'
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d'
import { api, resolveMediaPath } from './api'

/**
 * 3DGS God's Eye + Splat vs Real Frame compare (as in Fable IronSight video).
 */
export default function SplatViewer({ sessionId, session, activeShot, compare = false }) {
  const containerRef = useRef(null)
  const viewerRef = useRef(null)
  const [status, setStatus] = useState('idle')
  const [error, setError] = useState(null)

  const gs = session?.gaussian_splat
  const splatUrl = sessionId ? api.splatUrl(sessionId) : null

  const realFrame = useMemo(() => {
    if (!session || !activeShot?.frame_paths?.length) {
      // fallback: first camera preview as still via video current — use first shot frame
      const any = (session?.shots || []).find((s) => s.frame_paths?.length)
      if (!any) return null
      return resolveMediaPath(session.id, any.frame_paths[0])
    }
    return resolveMediaPath(session.id, activeShot.frame_paths[0])
  }, [session, activeShot])

  const primaryPreview = useMemo(() => {
    const cam = session?.cameras?.[0]
    if (!cam) return null
    return resolveMediaPath(session.id, cam.preview || cam.path)
  }, [session])

  useEffect(() => {
    let disposed = false
    let viewer = null

    async function boot() {
      if (!containerRef.current || !splatUrl || !gs) {
        setStatus('waiting')
        return
      }
      setStatus('loading')
      setError(null)

      if (viewerRef.current) {
        try {
          viewerRef.current.dispose()
        } catch {}
        viewerRef.current = null
        containerRef.current.innerHTML = ''
      }

      try {
        viewer = new GaussianSplats3D.Viewer({
          rootElement: containerRef.current,
          cameraUp: [0, -1, 0],
          initialCameraPosition: [0, -2, -4],
          initialCameraLookAt: [0, 0, 0],
          sharedMemoryForWorkers: false,
          gpuAcceleratedSort: true,
          halfPrecisionCovariancesOnGPU: true,
        })
        viewerRef.current = viewer

        await viewer.addSplatScene(splatUrl, {
          showLoadingUI: false,
          progressiveLoad: true,
        })

        if (disposed) {
          viewer.dispose()
          return
        }
        setStatus('ready')
      } catch (e) {
        console.error('Splat viewer error', e)
        if (!disposed) {
          setError(String(e.message || e))
          setStatus('error')
        }
      }
    }

    boot()
    return () => {
      disposed = true
      if (viewerRef.current) {
        try {
          viewerRef.current.dispose()
        } catch {}
        viewerRef.current = null
      }
    }
  }, [sessionId, splatUrl, gs?.path, gs?.method, gs?.gaussian_count, compare])

  if (!gs) {
    return (
      <div className="overlay-empty">
        <h2>3D Gaussian Splatting</h2>
        <p>No splat trained yet for this session.</p>
        <p>Run the pipeline (COLMAP → 3DGS) or click Retrain 3DGS after reconstruction.</p>
      </div>
    )
  }

  const splatPane = (
    <div className={`splat-pane ${compare ? 'half' : 'full'}`}>
      {compare && <div className="pane-tag splat-tag">← Splat Reconstruction</div>}
      <div className="godseye-hud" style={{ pointerEvents: 'none' }}>
        {!compare && (
          <>
            <div className="corner tl" />
            <div className="corner tr" />
            <div className="corner bl" />
            <div className="corner br" />
          </>
        )}
        <div className="godseye-label">
          SPLAT · {gs.gaussian_count?.toLocaleString() || '?'} GAUSSIANS · {gs.method}
          {gs.device ? ` · ${String(gs.device).toUpperCase()}` : ''}
          {gs.num_iters ? ` · ${gs.num_iters} ITER` : ''}
          {gs.metal ? ' · METAL' : ''}
          {status === 'loading' ? ' · LOADING…' : ''}
        </div>
      </div>
      {error && (
        <div className="overlay-empty" style={{ position: 'absolute', inset: 0, zIndex: 6 }}>
          <h2>Splat Load Error</h2>
          <p>{error}</p>
        </div>
      )}
      {status === 'loading' && (
        <div className="splat-loading">LOADING GAUSSIAN SPLATS…</div>
      )}
      <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
    </div>
  )

  if (!compare) {
    return (
      <div className="splat-stage">
        {splatPane}
        <div className="xray-hint">
          Free-fly the reconstructed range — angles your cameras never filmed
        </div>
      </div>
    )
  }

  // Compare mode: Splat | Real Frame (video demo layout)
  return (
    <div className="splat-compare">
      {splatPane}
      <div className="splat-pane half real">
        <div className="pane-tag real-tag">Real Frame →</div>
        {realFrame ? (
          <img src={realFrame} alt="real frame" className="real-frame-img" />
        ) : primaryPreview ? (
          <video src={primaryPreview} className="real-frame-img" muted playsInline />
        ) : (
          <div className="overlay-empty">
            <p>No real frame for this shot.</p>
          </div>
        )}
        {activeShot && (
          <div className="real-frame-meta">
            SHOT {activeShot.id} · {activeShot.classification}
            {activeShot.target_label ? ` · ${activeShot.target_label}` : ''}
          </div>
        )}
      </div>
    </div>
  )
}
