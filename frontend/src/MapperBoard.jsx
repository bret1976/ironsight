import React, { useEffect, useMemo, useState } from 'react'
import { api } from './api'

/**
 * R&D multi-mapper experiment board (video: COLMAP vs Pi3 vs lingbot overnight board).
 */
export default function MapperBoard({ sessionId, session }) {
  const [board, setBoard] = useState(null)
  const [err, setErr] = useState(null)
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState('')

  const load = async () => {
    if (!sessionId) return
    try {
      const b = await api.mappers(sessionId)
      setBoard(b)
      setErr(null)
    } catch (e) {
      setBoard(session?.mapper_board || null)
      if (!session?.mapper_board) setErr(String(e.message || e))
    }
  }

  useEffect(() => {
    load()
  }, [sessionId, session?.mapper_board?.path])

  const run = async (high = false) => {
    setBusy(true)
    setMsg(high ? 'Running multi-mapper board (incl. COLMAP high)…' : 'Running multi-mapper board…')
    setErr(null)
    try {
      const b = await api.runMappers(sessionId, high)
      setBoard(b)
      setMsg(b.summary || 'Done')
    } catch (e) {
      setErr(String(e.message || e))
      setMsg('Failed')
    } finally {
      setBusy(false)
    }
  }

  const series = board?.topdown?.series || []
  const bounds = useMemo(() => {
    let minX = Infinity
    let maxX = -Infinity
    let minZ = Infinity
    let maxZ = -Infinity
    for (const s of series) {
      for (const [x, z] of s.points || []) {
        minX = Math.min(minX, x)
        maxX = Math.max(maxX, x)
        minZ = Math.min(minZ, z)
        maxZ = Math.max(maxZ, z)
      }
    }
    if (!Number.isFinite(minX)) return { minX: -1, maxX: 1, minZ: -1, maxZ: 1 }
    const pad = 0.1
    const dx = (maxX - minX) || 1
    const dz = (maxZ - minZ) || 1
    return {
      minX: minX - dx * pad,
      maxX: maxX + dx * pad,
      minZ: minZ - dz * pad,
      maxZ: maxZ + dz * pad,
    }
  }, [series])

  const colors = {
    colmap_low: '#4db8ff',
    colmap_medium: '#3dff8a',
    colmap_high: '#f0c14a',
    opencv_sift: '#ff9f43',
    pi3: '#b388ff',
    lingbot_map: '#ff4d5a',
  }

  const toSvg = (x, z) => {
    const w = 640
    const h = 360
    const { minX, maxX, minZ, maxZ } = bounds
    const px = ((x - minX) / (maxX - minX || 1)) * (w - 40) + 20
    const py = ((z - minZ) / (maxZ - minZ || 1)) * (h - 40) + 20
    return [px, py]
  }

  return (
    <div className="mapper-board">
      <div className="mapper-head">
        <div>
          <h2>MULTI-MAPPER EXPERIMENT BOARD</h2>
          <p className="mapper-sub">
            Trajectory agreement · sim3-aligned · primary / fallback selection (video R&D board)
          </p>
        </div>
        <div className="mapper-actions">
          <button className="btn primary" disabled={busy || !sessionId} onClick={() => run(false)}>
            {busy ? 'Running…' : 'Run Board'}
          </button>
          <button className="btn" disabled={busy || !sessionId} onClick={() => run(true)}>
            + COLMAP High
          </button>
        </div>
      </div>

      {msg && <div className="mapper-msg">{msg}</div>}
      {err && <div className="mapper-err">{err}</div>}

      {!board && !busy && (
        <div className="overlay-empty">
          <h2>No experiments yet</h2>
          <p>Run the board to compare COLMAP quality tiers vs OpenCV SfM.</p>
          <p style={{ color: 'var(--text-dim)', marginTop: 8 }}>
            Pi3 / lingbot-map appear as external research slots (not bundled).
          </p>
        </div>
      )}

      {board && (
        <>
          <div className="mapper-picks">
            <span>
              PRIMARY <strong>{board.primary || '—'}</strong>
            </span>
            <span>
              FALLBACK <strong>{board.fallback || '—'}</strong>
            </span>
            <span>
              REF <strong>{board.reference || '—'}</strong>
            </span>
          </div>

          <div className="mapper-layout">
            <svg className="mapper-svg" viewBox="0 0 640 360" preserveAspectRatio="xMidYMid meet">
              <rect x="0" y="0" width="640" height="360" fill="#050805" />
              <text x="16" y="22" fill="#6a8a72" fontSize="11" fontFamily="monospace">
                Camera trajectories, sim3-aligned (top-down X–Z)
              </text>
              {series.map((s) => {
                const pts = (s.points || []).map(([x, z]) => toSvg(x, z))
                if (pts.length < 2) return null
                const d = pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' ')
                return (
                  <path
                    key={s.id}
                    d={d}
                    fill="none"
                    stroke={colors[s.id] || '#3dff8a'}
                    strokeWidth={s.is_ref ? 2.5 : 1.5}
                    strokeDasharray={s.gate?.startsWith('FAIL') ? '4 3' : undefined}
                    opacity={0.9}
                  />
                )
              })}
            </svg>

            <div className="mapper-table-wrap">
              <table className="mapper-table">
                <thead>
                  <tr>
                    <th>Engine</th>
                    <th>Pts</th>
                    <th>RMSE</th>
                    <th>Drift</th>
                    <th>Time</th>
                    <th>Gate</th>
                  </tr>
                </thead>
                <tbody>
                  {(board.comparisons || board.methods || []).map((r) => (
                    <tr key={r.id} className={r.id === board.primary ? 'primary-row' : ''}>
                      <td>
                        <span className="swatch" style={{ background: colors[r.id] || '#888' }} />
                        {r.id}
                      </td>
                      <td>{r.point_count ?? '—'}</td>
                      <td>{r.rmse != null ? Number(r.rmse).toFixed(3) : '—'}</td>
                      <td>
                        {r.drift != null ? `${(Number(r.drift) * 100).toFixed(1)}%` : '—'}
                      </td>
                      <td>{r.elapsed_s != null ? `${r.elapsed_s}s` : '—'}</td>
                      <td className={String(r.gate || r.status || '').startsWith('PASS') ? 'ok' : 'bad'}>
                        {r.gate || r.status || '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="mapper-note">{board.summary}</p>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
