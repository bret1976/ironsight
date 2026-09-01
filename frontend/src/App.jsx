import React, { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, getToken, resolveMediaPath, sessionWsUrl, setToken } from './api'
import MapperBoard from './MapperBoard'
import Landing from './Landing'
import ReviewWizard from './ReviewWizard'
import TeamHub from './TeamHub'

const SplatViewer = lazy(() => import('./SplatViewer.jsx'))

function fmtTime(s) {
  if (s == null || Number.isNaN(s)) return '00:00.000'
  const m = Math.floor(s / 60)
  const sec = s - m * 60
  return `${String(m).padStart(2, '0')}:${sec.toFixed(3).padStart(6, '0')}`
}

function tagClass(c) {
  if (c === 'HIT') return 'hit'
  if (c === 'MISS') return 'miss'
  if (c === 'NOT_A_SHOT') return 'notshot'
  return 'unknown'
}

function tagLabel(c) {
  if (c === 'HIT') return '✓ HIT'
  if (c === 'MISS') return '✗ MISS'
  if (c === 'NOT_A_SHOT') return 'NOT-A-SHOT'
  return c || 'UNKNOWN'
}

function fmtShort(s) {
  if (s == null || Number.isNaN(s)) return '0.0s'
  if (s < 60) return `${s.toFixed(1)}s`
  return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`
}

function sessionModeLabel(s) {
  if (s?.gaussian_splat) return 'Side-by-side · 3D ready'
  if (s?.pointcloud) return 'Side-by-side · map built'
  if ((s?.cameras?.length || 0) > 1) return 'Side-by-side video'
  return 'Practice'
}

function friendlyError(msg) {
  const m = String(msg || '')
  if (/plan limit|402/i.test(m)) return 'You’ve used this plan’s free sessions this month. Upgrade to keep going.'
  if (/coach-ready|Seal/i.test(m)) return m
  if (/sample practice|public_demo|Demo/i.test(m)) return 'This is a sample. Click New practice to use your own video.'
  if (/Authentication required|401/i.test(m)) return 'Please sign in to continue.'
  if (/not your session|403/i.test(m)) return 'That practice belongs to another account.'
  return m
}

function colorClass(c) {
  const k = String(c || 'steel').toLowerCase()
  if (k.includes('red')) return 'color-red'
  if (k.includes('white')) return 'color-white'
  if (k.includes('blue')) return 'color-blue'
  if (k.includes('green')) return 'color-green'
  if (k.includes('amber') || k.includes('yellow')) return 'color-amber'
  return 'color-steel'
}

/**
 * Pixel bboxes from multi-view tracks (CV + VLM + CSRT + 3D proj).
 * Falls back to shot-level bbox / visible_targets when timeline empty.
 */
function TargetOverlay({ boxes, activeShot }) {
  const list = boxes || []
  if (!list.length) return null
  return (
    <div className="target-layer">
      {list.map((t, i) => {
        const b = t.bbox
        if (!b || b.length < 4) return null
        const [x, y, w, h] = b.map(Number)
        // Skip garbage boxes
        if (w < 0.005 || h < 0.005 || w > 0.95 || h > 0.95) return null
        const active =
          t.active ||
          (activeShot &&
            (activeShot.target_id === t.id ||
              activeShot.target_label === t.label ||
              activeShot.target_label === t.id))
        const cls = t.classification || (active ? activeShot?.classification : null)
        return (
          <div
            key={`${t.id || t.label || i}-${i}`}
            className={`target-box ${colorClass(t.color)} ${active ? 'active' : ''} ${
              cls === 'HIT' ? 'is-hit' : cls === 'MISS' ? 'is-miss' : ''
            } ${t.source === 'ghost3d' || t.source === 'proj3d' ? 'ghost3d' : ''}`}
            style={{
              left: `${x * 100}%`,
              top: `${y * 100}%`,
              width: `${w * 100}%`,
              height: `${h * 100}%`,
            }}
            title={`${t.label || t.id} · ${t.source || ''} · conf ${((t.conf || t.confidence || 0) * 100).toFixed(0)}%`}
          >
            <div className="box-frame">
              <span className="c tl" />
              <span className="c tr" />
              <span className="c bl" />
              <span className="c br" />
            </div>
            <div className="tlabel">
              {(t.id && String(t.id) !== 'null' ? t.id : t.label?.split?.(' ')?.[0]) || 'T'}
              {t.label ? ` · ${String(t.label).replace(/^(B\d+|T\d+)\s*/i, '')}` : ''}
              {cls === 'HIT' ? ' ✓' : cls === 'MISS' ? ' ✗' : ''}
            </div>
          </div>
        )
      })}
    </div>
  )
}

function boxesFromSession(session, t, camera) {
  /** Prefer dense track_timeline; else nearest shot bboxes. */
  const camKey = camera === 'observer' ? 'observer' : 'primary'
  const tl = session?.track_timeline?.[camKey]
  if (tl?.length) {
    let best = tl[0]
    let bestD = Math.abs(best.t - t)
    for (const row of tl) {
      const d = Math.abs(row.t - t)
      if (d < bestD) {
        best = row
        bestD = d
      }
    }
    if (bestD <= 0.75) return best.targets || []
  }
  // Shot-level
  const shots = session?.shots || []
  if (!shots.length) return []
  let best = shots[0]
  let bestD = Math.abs(best.timestamp_s - t)
  for (const s of shots) {
    const d = Math.abs(s.timestamp_s - t)
    if (d < bestD) {
      best = s
      bestD = d
    }
  }
  if (bestD > 1.25) return []
  const out = []
  if (best.bbox) {
    out.push({
      id: best.target_id,
      label: best.target_label,
      color: best.target_color || 'steel',
      bbox: best.bbox,
      conf: best.confidence,
      source: 'shot',
      classification: best.classification,
      active: true,
    })
  }
  for (const v of best.visible_targets || []) {
    if (v.bbox) out.push({ ...v, source: v.source || 'vlm' })
  }
  return out
}

export default function App({ onEnterGods } = {}) {
  const [screen, setScreen] = useState(() => {
    if (typeof location !== 'undefined' && /#(range|dual|studio)\b/i.test(location.hash)) return 'app'
    const stored = localStorage.getItem('ironsight_screen')
    if (stored === 'landing' || stored === 'app') return stored
    return 'app'
  })
  const [user, setUser] = useState(null)
  const [health, setHealth] = useState(null)
  const [sessions, setSessions] = useState([])
  const [session, setSession] = useState(null)
  const [stageView, setStageView] = useState('dual') // dual | splat | compare | frames | mappers
  const [openRail, setOpenRail] = useState(null) // null | 'left' | 'right' — overlay drawers on narrow range
  const [activeShotId, setActiveShotId] = useState(null)
  const [playhead, setPlayhead] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [pipelineProgress, setPipelineProgress] = useState(0)
  const [pipelineMsg, setPipelineMsg] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [showNew, setShowNew] = useState(false)
  const [newName, setNewName] = useState('My practice')
  const [shooterFile, setShooterFile] = useState(null)
  const [observerFile, setObserverFile] = useState(null)
  const [wsState, setWsState] = useState('idle')
  const [showLibrary, setShowLibrary] = useState(false)
  const [primaryBoxes, setPrimaryBoxes] = useState([])
  const [observerBoxes, setObserverBoxes] = useState([])
  const [annotateMode, setAnnotateMode] = useState(false)
  const [wrapCard, setWrapCard] = useState(null)
  const [drawBox, setDrawBox] = useState(null) // {x0,y0,x1,y1} norm during drag
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [helpOpen, setHelpOpen] = useState(false)
  const [reviewQueue, setReviewQueue] = useState(null)
  const [quality, setQuality] = useState(null)
  const [rangeDash, setRangeDash] = useState(null)
  const [showReviewWizard, setShowReviewWizard] = useState(false)
  const [showTeamHub, setShowTeamHub] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const [analyzeStep, setAnalyzeStep] = useState(null)
  const [pwOld, setPwOld] = useState('')
  const [pwNew, setPwNew] = useState('')
  const videoRef = useRef(null)
  const obsVideoRef = useRef(null)
  const wsRef = useRef(null)
  const lastTrackFetch = useRef(0)

  const refreshList = useCallback(async () => {
    try {
      const r = await api.listSessions()
      setSessions(r.sessions || [])
    } catch (e) {
      setError(friendlyError(e.message || e))
    }
  }, [])

  useEffect(() => {
    api.health().then(setHealth).catch((e) => setError(friendlyError(e.message || e)))
    // Load user if token present
    if (getToken()) {
      api
        .me()
        .then((r) => setUser(r.user ? { ...r.user, anonymous: r.anonymous } : null))
        .catch(() => setToken(null))
    }
    // Billing return
    const q = new URLSearchParams(location.search)
    if (q.get('billing') === 'success') {
      setScreen('app')
      localStorage.setItem('ironsight_screen', 'app')
      api.me().then((r) => setUser(r.user)).catch(() => {})
      window.history.replaceState({}, '', '/')
    }
    if (typeof location !== 'undefined' && /#(range|dual|studio)\b/i.test(location.hash)) {
      setScreen('app')
      localStorage.setItem('ironsight_screen', 'app')
      setStageView('dual')
    }
  }, [])

  useEffect(() => {
    if (screen !== 'app') return
    refreshList().then(async () => {
      try {
        const r = await api.listSessions()
        const list = r.sessions || []
        const pick =
          list.find((s) => s.id === '8b27bffb48a6') ||
          list.find((s) => s.status === 'ready' && s.gaussian_splat) ||
          list.find((s) => s.status === 'ready') ||
          list[0]
        if (pick) openSession(pick.id)
      } catch {}
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [screen, refreshList])

  useEffect(() => {
    if (!session?.id) return
    const url = sessionWsUrl(session.id)
    const ws = new WebSocket(url)
    wsRef.current = ws
    setWsState('connecting')
    ws.onopen = () => setWsState('live')
    ws.onclose = () => setWsState('closed')
    ws.onerror = () => setWsState('error')
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data)
        if (msg.type === 'snapshot' && msg.session) {
          setSession(msg.session)
        }
        if (msg.type === 'pipeline') {
          setPipelineProgress(msg.progress || 0)
          setPipelineMsg(msg.message || '')
          if (msg.data?.step != null || msg.step != null) {
            setAnalyzeStep({
              step: msg.data?.step || msg.step,
              total: msg.data?.steps_total || msg.steps_total || 5,
              label: msg.message,
            })
          }
          if (msg.stats) {
            setSession((prev) =>
              prev
                ? {
                    ...prev,
                    stats: msg.stats,
                    status: msg.status || prev.status,
                    stage: msg.stage || prev.stage,
                  }
                : prev
            )
          }
          if (['hit_miss', 'reconstruction_3d', 'gaussian_splat', 'done', 'error', 'coaching', 'finalize'].includes(msg.stage)) {
            api.getSession(session.id).then(setSession).catch(() => {})
          }
          if (msg.stage === 'done') {
            refreshList()
            refreshReview(session.id)
            setShowReviewWizard(true)
            setPipelineMsg('Done — fix unsure shots, then Seal as coach-ready')
            setAnalyzeStep({ step: 5, total: 5, label: 'Complete' })
          }
          if (msg.stage === 'error') refreshList()
        }
      } catch {}
    }
    const ping = setInterval(() => {
      if (ws.readyState === 1) ws.send('ping')
    }, 15000)
    return () => {
      clearInterval(ping)
      ws.close()
    }
  }, [session?.id, refreshList])

  const activeShot = useMemo(() => {
    if (!session?.shots?.length) return null
    if (activeShotId != null) {
      return session.shots.find((s) => s.id === activeShotId) || session.shots[0]
    }
    return session.shots[0]
  }, [session, activeShotId])

  const duration = session?.stats?.duration_s || session?.cameras?.[0]?.duration_s || 0
  const totalShots = session?.stats?.total_shots ?? session?.shots?.length ?? 0
  const shotIndex = activeShot
    ? (session.shots.findIndex((s) => s.id === activeShot.id) + 1)
    : 0

  const seekTo = (t) => {
    setPlayhead(t)
    if (videoRef.current) videoRef.current.currentTime = t
    if (obsVideoRef.current) {
      const offset = session?.sync?.offset_s || 0
      obsVideoRef.current.currentTime = Math.max(0, t + offset)
    }
  }

  const selectShot = (shot) => {
    setActiveShotId(shot.id)
    seekTo(shot.timestamp_s)
  }

  const refreshReview = useCallback(async (id) => {
    if (!id) return
    try {
      const q = await api.reviewQueue(id)
      setReviewQueue(q)
    } catch {
      setReviewQueue(null)
    }
  }, [])

  const openSession = async (id) => {
    setError(null)
    setBusy(true)
    try {
      const doc = await api.getSession(id)
      setSession(doc)
      setActiveShotId(doc.shots?.[0]?.id ?? null)
      setPipelineProgress(doc.status === 'ready' ? 1 : 0)
      setStageView('dual')
      setShowLibrary(false)
      await refreshReview(id)
      try {
        const q = await api.quality(id)
        setQuality(q)
      } catch {
        setQuality(null)
      }
    } catch (e) {
      setError(String(e.message || e))
    } finally {
      setBusy(false)
    }
  }

  const createAndMaybeUpload = async () => {
    setBusy(true)
    setError(null)
    try {
      const doc = await api.createSession(newName || 'My practice')
      setShowNew(false)
      const hadFiles = !!(shooterFile || observerFile)
      if (hadFiles) {
        await api.upload(doc.id, { shooter: shooterFile, observer: observerFile })
      }
      await refreshList()
      await openSession(doc.id)
      setShooterFile(null)
      setObserverFile(null)
      if (getToken()) {
        api.me().then((r) => setUser(r.user ? { ...r.user, anonymous: r.anonymous } : null)).catch(() => {})
      }
      // Super-friendly: jump straight into analyze when they uploaded video
      if (hadFiles) {
        setPipelineMsg('Video saved — starting analyze…')
        setTimeout(() => {
          // process via API directly using doc.id
          api
            .process(doc.id)
            .then(() => {
              setPipelineMsg('Analyzing your video…')
              openSession(doc.id)
            })
            .catch((e) => setError(friendlyError(e.message || e)))
        }, 400)
      }
    } catch (e) {
      const msg = friendlyError(e.message || e)
      setError(msg)
      if (e.status === 402 || /plan limit|upgrade/i.test(msg)) {
        setTimeout(() => {
          if (window.confirm(`${msg}\n\nOpen pricing to upgrade?`)) {
            setScreen('landing')
            localStorage.setItem('ironsight_screen', 'landing')
          }
        }, 50)
      }
    } finally {
      setBusy(false)
    }
  }

  const runPipeline = async () => {
    if (!session) return
    if (session.public_demo) {
      setError('This is a sample practice. Click New practice to analyze your own video.')
      return
    }
    setBusy(true)
    setError(null)
    setPipelineProgress(0)
    setAnalyzeStep({ step: 1, total: 5, label: 'Starting…' })
    setPipelineMsg('Starting analyze…')
    try {
      const r = await api.process(session.id)
      if (r?.queued) {
        setPipelineMsg('Queued for Team priority worker — keep this tab open')
      } else if (r?.plan_features && !r.plan_features.gsplat) {
        setPipelineMsg('Analyzing… (3D off on Free — upgrade for fly-around)')
      } else {
        setPipelineMsg('Analyzing your video…')
      }
      setStageView('dual')
    } catch (e) {
      const msg = friendlyError(e.message || e)
      setError(msg)
      if (e.status === 402 || /upgrade|plan/i.test(msg)) {
        setTimeout(() => {
          if (window.confirm(`${msg}\n\nOpen pricing?`)) {
            setScreen('landing')
            localStorage.setItem('ironsight_screen', 'landing')
          }
        }, 50)
      }
    } finally {
      setBusy(false)
    }
  }

  const classifyReview = async (shotId, classification) => {
    if (!session) return
    setBusy(true)
    try {
      const r = await api.reviewShot(session.id, shotId, classification)
      setReviewQueue(r.queue)
      const doc = await api.getSession(session.id)
      setSession(doc)
      const nextId = r.queue?.next_shot_id
      if (nextId != null) {
        const s = (doc.shots || []).find((x) => x.id === nextId)
        if (s) selectShot(s)
      }
      if (r.queue && r.queue.needs_review_count === 0) {
        setPipelineMsg('All checked — Seal as coach-ready')
      }
    } catch (e) {
      setError(friendlyError(e.message || e))
    } finally {
      setBusy(false)
    }
  }

  const sealCoach = async () => {
    if (!session) return
    setBusy(true)
    try {
      const r = await api.coachLock(session.id, false)
      setReviewQueue(r.queue)
      setPipelineMsg('Sealed — you can download the report')
      await openSession(session.id)
    } catch (e) {
      setError(friendlyError(e.message || e))
      setShowReviewWizard(true)
    } finally {
      setBusy(false)
    }
  }

  const retrainGsplat = async (quality = 'default') => {
    if (!session) return
    setBusy(true)
    setError(null)
    setPipelineMsg(`Metal 3DGS retrain (${quality})…`)
    try {
      await api.retrainGsplat(session.id, { quality })
      await openSession(session.id)
      setStageView('splat')
      setPipelineProgress(1)
      setPipelineMsg('3DGS ready')
    } catch (e) {
      setError(String(e.message || e))
      setPipelineMsg('3DGS retrain failed')
    } finally {
      setBusy(false)
    }
  }

  const togglePlay = () => {
    const v = videoRef.current
    if (!v) return
    if (v.paused) {
      v.play()
      obsVideoRef.current?.play?.()
      setPlaying(true)
    } else {
      v.pause()
      obsVideoRef.current?.pause?.()
      setPlaying(false)
    }
  }

  // Keyboard: ← → shots, Space play/pause, 1-5 modes
  useEffect(() => {
    const onKey = (e) => {
      if (e.target.matches?.('input, textarea')) return
      if (!session?.shots?.length && !['1', '2', '3', '4', '5', 'n'].includes(e.key)) return
      const idx = session?.shots?.findIndex((s) => s.id === activeShot?.id) ?? -1
      if (e.key === 'ArrowRight' || e.key === ']') {
        const next = session.shots[Math.min(session.shots.length - 1, idx + 1)]
        if (next) selectShot(next)
      } else if (e.key === 'ArrowLeft' || e.key === '[') {
        const prev = session.shots[Math.max(0, idx - 1)]
        if (prev) selectShot(prev)
      } else if (e.key === ' ' || e.key === 'k') {
        e.preventDefault()
        togglePlay()
      } else if (e.key === '1') {
        setOpenRail(null)
        setStageView('dual')
      } else if (e.key === '2') {
        setOpenRail(null)
        setStageView('splat')
      } else if (e.key === '3') {
        setOpenRail(null)
        setStageView('compare')
      } else if (e.key === '4') {
        setOpenRail(null)
        onEnterGods?.()
      } else if (e.key === '5') {
        setOpenRail(null)
        setStageView('frames')
      } else if (e.key === 'n') setShowNew(true)
      else if (e.key === 'l') setShowLibrary((v) => !v)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  const primaryPreview = useMemo(() => {
    const cam = session?.cameras?.[0]
    if (!cam?.preview && !cam?.path) return null
    return resolveMediaPath(session.id, cam.preview || cam.path)
  }, [session])

  const observerPreview = useMemo(() => {
    const cam = session?.cameras?.[1]
    if (!cam?.preview && !cam?.path) return null
    return resolveMediaPath(session.id, cam.preview || cam.path)
  }, [session])

  // Real multi-view boxes synced to playhead
  useEffect(() => {
    if (!session?.id || stageView !== 'dual') return
    const localP = boxesFromSession(session, playhead, 'primary')
    const localO = boxesFromSession(session, playhead, 'observer')
    setPrimaryBoxes(localP)
    setObserverBoxes(localO)

    // Periodic API refresh for full tracks.json accuracy
    const now = performance.now()
    if (now - lastTrackFetch.current < 180) return
    lastTrackFetch.current = now
    if (!session.tracks && !session.track_timeline) return
    api
      .tracksAt(session.id, playhead, 'primary')
      .then((r) => {
        if (r?.targets?.length) setPrimaryBoxes(r.targets)
      })
      .catch(() => {})
    if (session.cameras?.length > 1) {
      const ot = playhead + (session.sync?.offset_s || 0)
      api
        .tracksAt(session.id, ot, 'observer')
        .then((r) => {
          if (r?.targets?.length) setObserverBoxes(r.targets)
        })
        .catch(() => {})
    }
  }, [session, playhead, stageView])

  const activeFrames = useMemo(() => {
    if (!activeShot?.frame_paths?.length || !session) return []
    return activeShot.frame_paths
      .map((p) => resolveMediaPath(session.id, p))
      .filter(Boolean)
  }, [activeShot, session])

  const modeLabel =
    stageView === 'dual'
      ? 'Side-by-side'
      : stageView === 'splat'
      ? '3D fly-around'
      : stageView === 'compare'
      ? '3D vs real'
      : stageView === 'mappers'
      ? 'Lab board'
      : 'Shot photos'

  if (screen === 'landing') {
    return (
      <Landing
        user={user}
        onAuth={(u) => {
          setUser(u)
          setScreen('app')
          localStorage.setItem('ironsight_screen', 'app')
        }}
        onEnterApp={() => {
          setScreen('app')
          localStorage.setItem('ironsight_screen', 'app')
          setStageView('dual')
        }}
        onEnterGods={() => onEnterGods?.()}
      />
    )
  }

  const sessionName = session?.name || 'No practice open yet'
  const camCount = session?.cameras?.length || 0
  const hasVideo = Boolean(primaryPreview)
  const hasShots = Boolean(session?.shots?.length)
  const isReady = session?.status === 'ready'
  const guideStep = !hasVideo ? 1 : session?.status === 'processing' ? 2 : hasShots || isReady ? 3 : 2
  const techFooter = [
    camCount >= 1 ? `${camCount} cam` : null,
    session?.sync?.method || null,
    session?.gaussian_splat
      ? `${session.gaussian_splat.gaussian_count?.toLocaleString?.() || '?'} 3D points`
      : null,
  ]
    .filter(Boolean)
    .join(' · ')

  const layoutName = 'range'

  return (
    <div
      className="app-hud layout-range"
      data-layout={layoutName}
      data-stage={stageView}
    >
      <header className="hud-top">
        <div className="hud-top-left">
          <div className="hud-dot" />
          <span>IronSight</span>
        </div>
        <div className="hud-top-center" title={modeLabel}>
          <span>{sessionName}</span>
          {session?.status && (
            <>
              <span className="sep">·</span>
              <span className="mode">
                {session.status === 'processing'
                  ? 'Analyzing…'
                  : session.status === 'ready'
                  ? 'Ready to review'
                  : session.status}
              </span>
            </>
          )}
        </div>
        <div className="hud-top-right">
          {user && (
            <span className="account-chip" title={user.email || ''}>
              {(user.plan || 'free').toUpperCase()}
              {user.plan_details?.sessions_per_month != null
                ? ` · ${user.sessions_used_month || 0}/${user.plan_details.sessions_per_month}`
                : ''}
            </span>
          )}
          <button type="button" className="mini-btn" onClick={() => setHelpOpen(true)}>
            Help
          </button>
          {user?.plan === 'free' && (
            <button
              type="button"
              className="mini-btn primary"
              onClick={() => {
                setScreen('landing')
                localStorage.setItem('ironsight_screen', 'landing')
              }}
            >
              Upgrade
            </button>
          )}
          {user && !user.anonymous && user.plan !== 'free' && (
            <button
              type="button"
              className="mini-btn"
              onClick={async () => {
                try {
                  const r = await api.portal()
                  if (r.portal_url) window.location.href = r.portal_url
                } catch (e) {
                  setError(String(e.message || e))
                }
              }}
            >
              Billing
            </button>
          )}
          {user && !user.anonymous && (
            <button
              type="button"
              className="mini-btn"
              onClick={() => {
                setToken(null)
                setUser(null)
                setScreen('landing')
                localStorage.setItem('ironsight_screen', 'landing')
              }}
            >
              Log out
            </button>
          )}
          <button
            type="button"
            className="mini-btn"
            onClick={() => {
              setScreen('landing')
              localStorage.setItem('ironsight_screen', 'landing')
            }}
          >
            Home
          </button>
        </div>
      </header>

      <div className="guide-strip">
        <div className={`guide-step ${guideStep === 1 ? 'active' : guideStep > 1 ? 'done' : ''}`}>
          <span className="gn">1</span> Upload video
        </div>
        <div className={`guide-step ${guideStep === 2 ? 'active' : guideStep > 2 ? 'done' : ''}`}>
          <span className="gn">2</span> Analyze
        </div>
        <div className={`guide-step ${guideStep === 3 ? 'active' : ''}`}>
          <span className="gn">3</span> Review shots
        </div>
        <div className="guide-hint">
          {guideStep === 1 && 'Start with “New practice” or try the demo.'}
          {guideStep === 2 && 'Press “Analyze my video” — this can take a few minutes.'}
          {guideStep === 3 && 'Click shots on the right. Fix mistakes with Fix score.'}
        </div>
      </div>

      <div className="session-strip always">
        {sessions.map((s) => (
          <button
            key={s.id}
            type="button"
            className={session?.id === s.id ? 'active' : ''}
            onClick={() => openSession(s.id)}
            title={s.public_demo ? 'Sample practice' : s.id}
          >
            <span className="sn">
              {s.public_demo ? '⭐ ' : ''}
              {s.name || s.id}
            </span>
            <span className="sm">
              {s.stats?.total_shots ?? 0} shots
              {s.stats?.duration_s != null ? ` · ${fmtShort(s.stats.duration_s)}` : ''}
              {s.status === 'processing' ? ' · working…' : ''}
              {s.status === 'ready' ? ' · ready' : ''}
            </span>
          </button>
        ))}
        <button type="button" className="strip-add" onClick={() => setShowNew(true)}>
          + New practice
        </button>
        <button type="button" className="strip-add" onClick={() => setShowLibrary(true)}>
          My library
        </button>
        <button type="button" className="strip-add" onClick={() => setShowTeamHub(true)}>
          Team
        </button>
        <button type="button" className="strip-add" onClick={() => setShowSettings(true)}>
          Settings
        </button>
      </div>

      {session?.public_demo && (
        <div className="demo-banner">
          <strong>Sample practice</strong> — look around and try the shot list. To score{' '}
          <em>your</em> video, click <strong>New practice</strong> (Analyze is locked here).
          <button type="button" className="mini-btn primary" onClick={() => setShowNew(true)}>
            New practice
          </button>
        </div>
      )}

      {error && (
        <div className="error-toast" onClick={() => setError(null)}>
          {friendlyError(error)}
        </div>
      )}

      {/* BODY */}
      <div className="hud-body">
        {/* LEFT — overlay drawer on narrow range; hidden in cockpit */}
        <aside className={`hud-left ${openRail === 'left' ? 'is-open' : ''}`}>
          <div className="shot-big">
            <div className="label">This shot</div>
            <div className="num">
              {shotIndex || 0}
              <span>/{totalShots || 0}</span>
            </div>
          </div>
          <div className="hitmiss-row">
            <div className="ok">✓ Hit {session?.stats?.hits ?? 0}</div>
            <div className="bad">✗ Miss {session?.stats?.misses ?? 0}</div>
            {(session?.stats?.unknowns || 0) > 0 && (
              <div className="dim">? Unsure {session.stats.unknowns}</div>
            )}
            {(session?.stats?.needs_review || 0) > 0 && (
              <div className="dim warn-review">⚑ Check {session.stats.needs_review}</div>
            )}
          </div>
          <div className="acc-block" title={session?.stats?.accuracy_display || ''}>
            <div className="label">Your score</div>
            <div className={`val score-val trust-${session?.stats?.accuracy_trust || 'none'}`}>
              {session?.stats?.accuracy_display ||
                (session?.stats?.adjudicated
                  ? `${session.stats.accuracy ?? 0}%`
                  : 'Upload & analyze to score')}
            </div>
            {(session?.stats?.adjudicated != null || session?.stats?.total_shots) && (
              <div className="acc-sub">
                {session.stats.adjudicated ?? 0} of {session.stats.total_shots ?? 0} shots scored
              </div>
            )}
          </div>

          <div className="mini-actions">
            <div className="primary-actions">
              <button type="button" className="mini-btn primary" onClick={() => setShowNew(true)}>
                1 · New practice
              </button>
              <button
                type="button"
                className="mini-btn primary"
                disabled={!session || busy || session?.status === 'processing' || session?.public_demo}
                onClick={runPipeline}
                title={session?.public_demo ? 'Sample is read-only' : ''}
              >
                {session?.status === 'processing' ? 'Analyzing…' : '2 · Analyze my video'}
              </button>
              {session?.status === 'ready' && !session?.public_demo && (
                <button
                  type="button"
                  className="mini-btn primary"
                  onClick={() => setShowReviewWizard(true)}
                >
                  Check unsure shots
                </button>
              )}
              <button
                type="button"
                className="mini-btn primary"
                disabled={!session || busy}
                onClick={async () => {
                  setBusy(true)
                  try {
                    const w = await api.wrap(session.id)
                    setWrapCard(w)
                    if (w.needs_coach_seal) {
                      setShowReviewWizard(true)
                      setPipelineMsg('Fix shots & seal before full download')
                    }
                  } catch (e) {
                    setError(friendlyError(e.message || e))
                    if (/Seal|coach-ready/i.test(String(e.message || e))) {
                      setShowReviewWizard(true)
                    }
                  } finally {
                    setBusy(false)
                  }
                }}
              >
                3 · Get my report
              </button>
              <button
                type="button"
                className="mini-btn"
                disabled={busy}
                onClick={async () => {
                  setBusy(true)
                  try {
                    const r = await api.openDemo()
                    await refreshList()
                    if (r.session_id) await openSession(r.session_id)
                    setPipelineMsg('Demo loaded — click shots on the right to explore')
                  } catch (e) {
                    setError(String(e.message || e))
                  } finally {
                    setBusy(false)
                  }
                }}
              >
                Try the demo first
              </button>
            </div>

            <div className="transport">
              <button type="button" className="mini-btn" onClick={togglePlay} disabled={!primaryPreview}>
                {playing ? 'Pause' : 'Play'}
              </button>
              <button
                type="button"
                className="mini-btn"
                disabled={!session?.shots?.length}
                onClick={() => {
                  const idx = session.shots.findIndex((s) => s.id === activeShot?.id)
                  const prev = session.shots[Math.max(0, idx - 1)]
                  if (prev) selectShot(prev)
                }}
              >
                ← Prev
              </button>
              <button
                type="button"
                className="mini-btn"
                disabled={!session?.shots?.length}
                onClick={() => {
                  const idx = session.shots.findIndex((s) => s.id === activeShot?.id)
                  const next = session.shots[Math.min(session.shots.length - 1, idx + 1)]
                  if (next) selectShot(next)
                }}
              >
                Next →
              </button>
            </div>

            <button
              type="button"
              className={`mini-btn ${annotateMode ? 'primary' : ''}`}
              disabled={!session?.shots?.length}
              onClick={() => setAnnotateMode((v) => !v)}
            >
              {annotateMode ? 'Fixing score…' : 'Fix score'}
            </button>

            {reviewQueue && reviewQueue.needs_review_count > 0 && (
              <div className="review-nudge">
                Coach checklist: {reviewQueue.progress?.reviewed || 0}/
                {reviewQueue.total_shots || 0} done · {reviewQueue.needs_review_count} left
                {reviewQueue.next_shot_id != null && (
                  <button
                    type="button"
                    className="mini-btn primary"
                    style={{ marginTop: 6, width: '100%' }}
                    onClick={() => {
                      const s = session?.shots?.find((x) => x.id === reviewQueue.next_shot_id)
                      if (s) {
                        selectShot(s)
                        setAnnotateMode(true)
                      }
                    }}
                  >
                    Next unsure shot → #{reviewQueue.next_shot_id}
                  </button>
                )}
              </div>
            )}
            {reviewQueue?.coach_locked && (
              <div className="review-nudge" style={{ borderColor: 'rgba(94,234,212,0.4)', color: 'var(--teal)' }}>
                ✓ Coach-locked — safe to share wrap
              </div>
            )}
            {session?.shots?.length > 0 && !session?.public_demo && (
              <button
                type="button"
                className="mini-btn"
                disabled={busy}
                onClick={async () => {
                  setBusy(true)
                  try {
                    if (reviewQueue?.coach_locked) {
                      await api.coachUnlock(session.id)
                      setPipelineMsg('Unlocked for more edits')
                    } else {
                      const r = await api.coachLock(session.id, false)
                      setPipelineMsg('Coach lock on — report is coach-ready')
                      setReviewQueue(r.queue)
                    }
                    await openSession(session.id)
                  } catch (e) {
                    const msg = String(e.message || e)
                    setError(msg)
                    if (/still need review/i.test(msg)) {
                      setAnnotateMode(true)
                    }
                  } finally {
                    setBusy(false)
                  }
                }}
              >
                {reviewQueue?.coach_locked ? 'Unlock coach seal' : 'Seal as coach-ready'}
              </button>
            )}

            {quality && (
              <div className="team-panel">
                <div className="label">3D readiness · {quality.grade}</div>
                <div style={{ fontSize: 11, color: 'var(--text-dim)', lineHeight: 1.35 }}>
                  {quality.summary}
                </div>
                {(quality.tips || []).slice(0, 2).map((t, i) => (
                  <div key={i} style={{ fontSize: 11, color: 'var(--teal)', marginTop: 4 }}>
                    • {t}
                  </div>
                ))}
                {(quality.warnings || []).slice(0, 2).map((t, i) => (
                  <div key={i} style={{ fontSize: 11, color: 'var(--amber)', marginTop: 4 }}>
                    ! {t}
                  </div>
                ))}
                <button
                  type="button"
                  className="mini-btn"
                  style={{ marginTop: 6 }}
                  disabled={busy || !session}
                  onClick={async () => {
                    try {
                      const q = await api.quality(session.id)
                      setQuality(q)
                    } catch (e) {
                      setError(String(e.message || e))
                    }
                  }}
                >
                  Re-check video quality
                </button>
              </div>
            )}

            <button
              type="button"
              className="advanced-toggle"
              onClick={() => setShowAdvanced((v) => !v)}
            >
              {showAdvanced ? '▾ Hide extra tools' : '▸ More tools (3D, team…)'}
            </button>

            {showAdvanced && (
              <>
                <button type="button" className="mini-btn" onClick={() => setShowLibrary(true)}>
                  My library
                </button>
                {user && !user.anonymous && (
                  <div className="team-panel">
                    <div className="label">Team & help</div>
                    <button
                      type="button"
                      className="mini-btn"
                      disabled={busy || (user.plan !== 'team' && user.plan !== 'pro')}
                      title={
                        user.plan === 'free'
                          ? 'Needs Pro (3 seats) or Team (10)'
                          : 'Create org'
                      }
                      onClick={async () => {
                        setBusy(true)
                        try {
                          const r = await api.createOrg(`${user.name || 'Team'} Range`)
                          setPipelineMsg(
                            `Org ready · ${r.org?.seat_limit || '?'} seats (${user.plan})`
                          )
                          const me = await api.me()
                          setUser(me.user ? { ...me.user, anonymous: me.anonymous } : null)
                          if (r.org?.id) {
                            try {
                              setRangeDash(await api.orgDashboard(r.org.id))
                            } catch {}
                          }
                        } catch (e) {
                          setError(String(e.message || e))
                        } finally {
                          setBusy(false)
                        }
                      }}
                    >
                      {user.plan === 'pro' ? 'Coach team (3 seats)' : 'Range team (10 seats)'}
                    </button>
                    {user.org_id && (
                      <button
                        type="button"
                        className="mini-btn"
                        onClick={async () => {
                          try {
                            const d = await api.orgDashboard(user.org_id)
                            setRangeDash(d)
                            setPipelineMsg(
                              `Range: ${d.summary?.sessions || 0} practices · ${d.summary?.needs_review_shots || 0} need review`
                            )
                          } catch (e) {
                            setError(String(e.message || e))
                          }
                        }}
                      >
                        Range dashboard
                      </button>
                    )}
                    {user.org_id && user.plan === 'team' && (
                      <button
                        type="button"
                        className="mini-btn"
                        onClick={async () => {
                          const name = prompt('Lane name (e.g. Bay 1)')
                          if (!name) return
                          try {
                            await api.createLane(user.org_id, name)
                            setPipelineMsg(`Lane “${name}” added`)
                          } catch (e) {
                            setError(String(e.message || e))
                          }
                        }}
                      >
                        Add lane
                      </button>
                    )}
                    {user.org_id && (
                      <button
                        type="button"
                        className="mini-btn"
                        onClick={async () => {
                          try {
                            await api.acceptSla(user.org_id)
                            setPipelineMsg('Team SLA accepted')
                          } catch (e) {
                            setError(String(e.message || e))
                          }
                        }}
                      >
                        Accept SLA
                      </button>
                    )}
                    <button
                      type="button"
                      className="mini-btn"
                      onClick={async () => {
                        const subject = prompt('What do you need help with?')
                        if (!subject) return
                        const body = prompt('Describe the problem (include practice name if you can)')
                        if (!body) return
                        try {
                          const r = await api.support(subject, body)
                          setPipelineMsg(`Message sent · ticket ${r.ticket_id}`)
                        } catch (e) {
                          setError(String(e.message || e))
                        }
                      }}
                    >
                      Contact support
                    </button>
                  </div>
                )}
                <button
                  type="button"
                  className="mini-btn"
                  disabled={!session || !session.pointcloud || busy}
                  onClick={() => retrainGsplat('default')}
                >
                  Rebuild 3D view
                </button>
                <button
                  type="button"
                  className="mini-btn"
                  disabled={!session || busy || !session.shots?.length}
                  onClick={async () => {
                    setBusy(true)
                    setError(null)
                    setPipelineMsg('Rebuilding tracks…')
                    try {
                      await api.rebuildTracks(session.id)
                      await openSession(session.id)
                      setPipelineMsg('Tracks updated')
                    } catch (e) {
                      setError(String(e.message || e))
                    } finally {
                      setBusy(false)
                    }
                  }}
                >
                  Rebuild tracks
                </button>
              </>
            )}

            {annotateMode && activeShot && (
              <div className="annotate-panel">
                <div className="label">Set shot {activeShot.id} to…</div>
                <div className="ann-btns">
                  {['HIT', 'MISS', 'NOT_A_SHOT', 'UNKNOWN'].map((c) => (
                    <button
                      type="button"
                      key={c}
                      className={`mini-btn ${activeShot.classification === c ? 'primary' : ''}`}
                      onClick={async () => {
                        try {
                          await api.annotate(session.id, {
                            shot_id: activeShot.id,
                            classification: c,
                            bbox: drawBox
                              ? [
                                  Math.min(drawBox.x0, drawBox.x1),
                                  Math.min(drawBox.y0, drawBox.y1),
                                  Math.abs(drawBox.x1 - drawBox.x0),
                                  Math.abs(drawBox.y1 - drawBox.y0),
                                ]
                              : activeShot.bbox,
                          })
                          try {
                            await api.reviewShot(session.id, activeShot.id, c)
                          } catch {}
                          await openSession(session.id)
                          setDrawBox(null)
                        } catch (e) {
                          setError(String(e.message || e))
                        }
                      }}
                    >
                      {c === 'NOT_A_SHOT' ? 'NOT-SHOT' : c}
                    </button>
                  ))}
                </div>
                <p className="ann-hint">Optional: drag a box on the shooter video, then pick Hit/Miss.</p>
              </div>
            )}
            {(pipelineMsg || session?.status === 'processing' || analyzeStep) && (
              <>
                {analyzeStep && (
                  <div className="analyze-steps">
                    {[1, 2, 3, 4, 5].map((n) => (
                      <span
                        key={n}
                        className={
                          n < (analyzeStep.step || 0)
                            ? 'done'
                            : n === analyzeStep.step
                            ? 'active'
                            : ''
                        }
                      >
                        {n}
                      </span>
                    ))}
                  </div>
                )}
                <div style={{ color: 'var(--text-dim)', fontSize: 12, lineHeight: 1.35 }}>
                  {pipelineMsg || session?.stage}
                </div>
                <div className="progress-slim">
                  <div
                    style={{
                      width: `${Math.round(
                        (pipelineProgress || (session?.status === 'ready' ? 1 : 0)) * 100
                      )}%`,
                    }}
                  />
                </div>
              </>
            )}
          </div>
        </aside>

        {/* CENTER STAGE */}
        <section className="hud-stage">
          <div className="stage-modes">
            <button
              type="button"
              className={stageView === 'dual' ? 'active' : ''}
              onClick={() => {
                setOpenRail(null)
                setStageView('dual')
              }}
            >
              Side-by-side video
            </button>
            <button
              type="button"
              className={stageView === 'splat' ? 'active' : ''}
              onClick={() => {
                setOpenRail(null)
                setStageView('splat')
              }}
            >
              3D fly-around
            </button>
            <button
              type="button"
              className={stageView === 'compare' ? 'active' : ''}
              onClick={() => {
                setOpenRail(null)
                setStageView('compare')
              }}
            >
              3D vs real
            </button>
            <button
              type="button"
              onClick={() => {
                setOpenRail(null)
                onEnterGods?.()
              }}
            >
              See-through map
            </button>
            <button
              type="button"
              className={stageView === 'frames' ? 'active' : ''}
              onClick={() => {
                setOpenRail(null)
                setStageView('frames')
              }}
            >
              Shot photos
            </button>
            <button
              type="button"
              className={stageView === 'mappers' ? 'active' : ''}
              onClick={() => {
                setOpenRail(null)
                setStageView('mappers')
              }}
            >
              Lab board
            </button>
          </div>
          <button
            type="button"
            className={`rail-backdrop ${openRail ? 'is-on' : ''}`}
            aria-label="Close side panels"
            onClick={() => setOpenRail(null)}
          />
          <div className="rail-chips">
            <button
              type="button"
              className={`mini-btn ${openRail === 'left' ? 'primary' : ''}`}
              onClick={() => setOpenRail((v) => (v === 'left' ? null : 'left'))}
            >
              Score
            </button>
            <button
              type="button"
              className={`mini-btn ${openRail === 'right' ? 'primary' : ''}`}
              onClick={() => setOpenRail((v) => (v === 'right' ? null : 'right'))}
            >
              Shots
            </button>
          </div>

          {!hasVideo && stageView === 'dual' && (
            <div className="welcome-hero">
              <h2>Welcome — let’s score a practice</h2>
              <p>
                IronSight watches your range video, finds each bang, and labels hit or miss. You can
                fix anything the AI gets wrong, then share a short report.
              </p>
              <div className="welcome-cards">
                <div className="welcome-card">
                  <strong>1 · Upload</strong>
                  <span>Phone video from the shooter. A second side angle helps a lot.</span>
                </div>
                <div className="welcome-card">
                  <strong>2 · Analyze</strong>
                  <span>We listen for shots and score them. Grab a snack — big files take time.</span>
                </div>
                <div className="welcome-card">
                  <strong>3 · Review</strong>
                  <span>Jump shot-to-shot, fix labels, download your wrap report.</span>
                </div>
              </div>
              <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', justifyContent: 'center' }}>
                <button type="button" className="btn-glow" onClick={() => setShowNew(true)}>
                  New practice
                </button>
                <button
                  type="button"
                  className="btn-ghost"
                  disabled={busy}
                  onClick={async () => {
                    setBusy(true)
                    try {
                      const r = await api.openDemo()
                      await refreshList()
                      if (r.session_id) await openSession(r.session_id)
                    } catch (e) {
                      setError(String(e.message || e))
                    } finally {
                      setBusy(false)
                    }
                  }}
                >
                  Load demo
                </button>
              </div>
            </div>
          )}

          {activeShot && (stageView === 'dual' || stageView === 'compare') && (
            <div className={`hit-banner-center ${tagClass(activeShot.classification)}`}>
              {activeShot.classification === 'HIT'
                ? '✓ '
                : activeShot.classification === 'MISS'
                ? '✗ '
                : activeShot.classification === 'NOT_A_SHOT'
                ? '⊘ '
                : '? '}
              {activeShot.classification === 'NOT_A_SHOT'
                ? 'NOT-A-SHOT'
                : activeShot.classification}
              {activeShot.target_label && activeShot.classification !== 'NOT_A_SHOT'
                ? ` — ${activeShot.target_label}`
                : ''}
            </div>
          )}

          {stageView === 'dual' && hasVideo && (
            <div className="dual-sxs layout-probe-dual">
              <div className="cam-pane">
                <div className="cam-label-bar">
                  <span className="role">Your view (shooter)</span>
                  <span className="meta">
                    {session?.cameras?.[0]?.filename
                      ? session.cameras[0].filename.slice(0, 18)
                      : '—'}
                  </span>
                </div>
                {primaryPreview ? (
                  <>
                    <video
                      ref={videoRef}
                      src={primaryPreview}
                      playsInline
                      onTimeUpdate={(e) => {
                        setPlayhead(e.target.currentTime)
                        if (obsVideoRef.current && session?.sync) {
                          const want = e.target.currentTime + (session.sync.offset_s || 0)
                          if (Math.abs(obsVideoRef.current.currentTime - want) > 0.12) {
                            obsVideoRef.current.currentTime = Math.max(0, want)
                          }
                        }
                      }}
                      onPlay={() => {
                        setPlaying(true)
                        obsVideoRef.current?.play?.()
                      }}
                      onPause={() => {
                        setPlaying(false)
                        obsVideoRef.current?.pause?.()
                      }}
                      onEnded={() => setPlaying(false)}
                    />
                    <TargetOverlay
                      boxes={[
                        ...primaryBoxes,
                        // 3D→2D see-through ghosts (proj3d / session targets without live track)
                        ...((session?.targets || [])
                          .filter((t) => t.position_3d && !primaryBoxes.some((b) => b.id === t.id))
                          .slice(0, 8)
                          .map((t, i) => {
                            // fan estimated positions when no pixel track — mark as ghost through-wall
                            const n = Math.max(1, (session.targets || []).length)
                            const x = 0.2 + (i / n) * 0.55
                            return {
                              id: t.id,
                              label: t.label,
                              color: t.color,
                              bbox: [x, 0.35, 0.06, 0.1],
                              source: 'ghost3d',
                              conf: 0.4,
                            }
                          }) || []),
                      ]}
                      activeShot={activeShot}
                    />
                    {annotateMode && (
                      <div
                        className="annotate-draw"
                        onMouseDown={(e) => {
                          const r = e.currentTarget.getBoundingClientRect()
                          const x = (e.clientX - r.left) / r.width
                          const y = (e.clientY - r.top) / r.height
                          setDrawBox({ x0: x, y0: y, x1: x, y1: y })
                        }}
                        onMouseMove={(e) => {
                          if (!drawBox || drawBox.x0 == null) return
                          const r = e.currentTarget.getBoundingClientRect()
                          const x = (e.clientX - r.left) / r.width
                          const y = (e.clientY - r.top) / r.height
                          setDrawBox((d) => (d ? { ...d, x1: x, y1: y } : d))
                        }}
                        onMouseUp={() => {}}
                      >
                        {drawBox && (
                          <div
                            className="draw-rect"
                            style={{
                              left: `${Math.min(drawBox.x0, drawBox.x1) * 100}%`,
                              top: `${Math.min(drawBox.y0, drawBox.y1) * 100}%`,
                              width: `${Math.abs(drawBox.x1 - drawBox.x0) * 100}%`,
                              height: `${Math.abs(drawBox.y1 - drawBox.y0) * 100}%`,
                            }}
                          />
                        )}
                      </div>
                    )}
                  </>
                ) : (
                  <div className="cam-empty">
                    <h3>No shooter video yet</h3>
                    <p>Upload the video from your point of view.</p>
                    <button type="button" className="btn primary" style={{ marginTop: 12 }} onClick={() => setShowNew(true)}>
                      New practice
                    </button>
                  </div>
                )}
              </div>
              <div className="cam-pane">
                <div className="cam-label-bar">
                  <span className="role">Side view (optional)</span>
                  <span className="meta">
                    {session?.sync
                      ? `synced ${Number(session.sync.offset_s || 0).toFixed(2)}s`
                      : 'optional'}
                  </span>
                </div>
                {observerPreview ? (
                  <>
                    <video ref={obsVideoRef} src={observerPreview} playsInline muted />
                    <TargetOverlay boxes={observerBoxes} activeShot={activeShot} />
                  </>
                ) : (
                  <div className="cam-empty">
                    <h3>No side camera</h3>
                    <p>Still works with one camera. A second angle makes scoring and 3D better.</p>
                  </div>
                )}
              </div>
            </div>
          )}

          {stageView === 'splat' && session && (
            <Suspense fallback={null}>
              <SplatViewer sessionId={session.id} session={session} activeShot={activeShot} />
            </Suspense>
          )}

          {stageView === 'compare' && session && (
            <Suspense fallback={null}>
              <SplatViewer
                sessionId={session.id}
                session={session}
                activeShot={activeShot}
                compare
              />
            </Suspense>
          )}

          {stageView === 'mappers' && session && (
            <MapperBoard sessionId={session.id} session={session} />
          )}

          {stageView === 'frames' && (
            <div className="frames-grid">
              {activeFrames.length === 0 && (
                <div className="overlay-empty">
                  <h2>No frames</h2>
                  <p>Run pipeline to extract adjudication frames per shot.</p>
                </div>
              )}
              {activeFrames.map((src, i) => (
                <img key={i} src={src} alt={`frame-${i}`} />
              ))}
              {activeShot?.reasoning && (
                <div className="reason">
                  <strong>AI note · </strong>
                  {activeShot.reasoning}
                  {activeShot.confidence != null && (
                    <span style={{ color: 'var(--text-dim)' }}>
                      {' '}
                      · conf {(activeShot.confidence * 100).toFixed(0)}%
                    </span>
                  )}
                </div>
              )}
            </div>
          )}

          {session?.coaching &&
            !session?.coaching_locked &&
            user?.plan !== 'free' &&
            stageView === 'dual' && (
            <div className="coaching-toast">
              <strong>COACH · </strong>
              {session.coaching}
            </div>
          )}
          {user?.plan === 'free' && session?.status === 'ready' && stageView === 'dual' && (
            <div className="coaching-toast locked">
              <strong>PRO · </strong>
              Full coach debrief + wrap export unlock on Pro. Use Review to lock hit/miss truth.
            </div>
          )}
        </section>

        <aside className={`hud-right ${openRail === 'right' ? 'is-open' : ''}`}>
          <div className="shot-log-h">Shots · {session?.shots?.length || 0}</div>
          <div className="shot-log">
            {(session?.shots || []).map((s) => (
              <button
                type="button"
                key={s.id}
                className={`shot-item ${activeShot?.id === s.id ? 'active' : ''} ${
                  s.needs_review || s.classification === 'UNKNOWN' ? 'needs-review' : ''
                }`}
                onClick={() => selectShot(s)}
              >
                <span className="n">{String(s.id).padStart(2, '0')}</span>
                <span className="tgt">
                  <span className={`cdot ${colorClass(s.target_color)}`} />
                  {s.classification === 'NOT_A_SHOT'
                    ? 'not a shot'
                    : s.target_label || s.target_id || 'target?'}
                  {(s.needs_review || s.classification === 'UNKNOWN') && (
                    <span className="review-dot" title="Double-check this one">
                      {' '}
                      ⚑
                    </span>
                  )}
                </span>
                <span className={`tag ${tagClass(s.classification)}`}>{tagLabel(s.classification)}</span>
              </button>
            ))}
            {!session?.shots?.length && (
              <div style={{ padding: 14, color: 'var(--text-dim)', fontSize: 12, lineHeight: 1.45 }}>
                No shots yet.
                <br />
                <br />
                1. New practice + upload
                <br />
                2. Analyze my video
                <br />
                3. Shots show up here
              </div>
            )}
          </div>
        </aside>
      </div>

      {/* TIMELINE */}
      <div className="hud-timeline">
        <div className="tl-row">
          <div
            className="tl-track"
            onClick={(e) => {
              if (!duration) return
              const rect = e.currentTarget.getBoundingClientRect()
              const x = (e.clientX - rect.left) / rect.width
              seekTo(x * duration)
            }}
          >
            {(session?.shots || []).map((s) => (
              <div
                key={s.id}
                className={`tl-marker ${tagClass(s.classification)} ${
                  activeShot?.id === s.id ? 'active' : ''
                }`}
                style={{ left: `${duration ? (s.timestamp_s / duration) * 100 : 0}%` }}
                onClick={(e) => {
                  e.stopPropagation()
                  selectShot(s)
                }}
                title={`#${s.id} ${s.classification} ${fmtTime(s.timestamp_s)}`}
              >
                <div className="stem" />
                <div className="dot" />
                <div className="num">{s.id}</div>
              </div>
            ))}
            <div
              className="tl-playhead"
              style={{ left: `${duration ? (playhead / duration) * 100 : 0}%` }}
            />
          </div>
        </div>
        <div className="tl-meta">
          <span>
            {activeShot ? `Shot ${activeShot.id}` : 'Click the timeline or a shot'} · keys: ← → shots ·
            space play
          </span>
          <span>
            {fmtTime(playhead)} / {fmtTime(duration)}
          </span>
        </div>
      </div>

      <footer className="hud-footer">
        <span>
          IronSight <strong>studio</strong>
        </span>
        <span className="sep">|</span>
        <span className="footer-tech">{techFooter || 'Ready when you are'}</span>
        <span className="sep">|</span>
        <span className={wsState === 'live' ? 'live' : ''}>
          {wsState === 'live' ? 'Live' : 'Connecting…'}
        </span>
        <span className="grow" />
        <span>{session ? session.name || session.id : 'No practice open'}</span>
      </footer>

      {helpOpen && (
        <div className="modal-backdrop" onClick={() => setHelpOpen(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h3>How to use IronSight</h3>
            <p className="lib-sub">In plain English — like explaining it to a friend.</p>
            <ol style={{ paddingLeft: 18, color: 'var(--text-dim)', lineHeight: 1.55, fontSize: 14 }}>
              <li>
                <strong style={{ color: 'var(--text)' }}>New practice</strong> — name it and upload
                your video (and a side camera if you have one).
              </li>
              <li>
                <strong style={{ color: 'var(--text)' }}>Analyze my video</strong> — we find bangs
                and score hit/miss. Wait for “Ready to review.”
              </li>
              <li>
                <strong style={{ color: 'var(--text)' }}>Click shots</strong> on the right list or
                the timeline.
              </li>
              <li>
                <strong style={{ color: 'var(--text)' }}>Fix score</strong> if the AI was wrong.
              </li>
              <li>
                <strong style={{ color: 'var(--text)' }}>Get my report</strong> — shareable wrap for
                a coach.
              </li>
            </ol>
            <p style={{ marginTop: 14, fontSize: 13, color: 'var(--text-mute)' }}>
              Demo practice is read-only. 3D views are optional (Pro). Not military radar — just
              smart video review.
            </p>
            <div className="modal-actions">
              <button type="button" className="btn primary" onClick={() => setHelpOpen(false)}>
                Got it
              </button>
            </div>
          </div>
        </div>
      )}

      {wrapCard && (
        <div className="modal-backdrop" onClick={() => setWrapCard(null)}>
          <div className="modal wrap-card" onClick={(e) => e.stopPropagation()}>
            <div className="wrap-kicker">SESSION WRAP</div>
            <h2 className="wrap-title">{wrapCard.title}</h2>
            <div className="wrap-headline">{wrapCard.headline}</div>
            <div className="wrap-stats">
              <div>
                <div className="wk">HITS</div>
                <div className="wv">{wrapCard.stats?.hits}</div>
              </div>
              <div>
                <div className="wk">MISSES</div>
                <div className="wv red">{wrapCard.stats?.misses}</div>
              </div>
              <div>
                <div className="wk">SCORE</div>
                <div className="wv sm">
                  {wrapCard.stats?.accuracy_display || `${wrapCard.stats?.accuracy ?? 0}%`}
                </div>
              </div>
              <div>
                <div className="wk">BEST SPLIT</div>
                <div className="wv sm">
                  {wrapCard.stats?.best_split_s != null
                    ? `${wrapCard.stats.best_split_s}s`
                    : '—'}
                </div>
              </div>
            </div>
            {(wrapCard.stats?.needs_review > 0 || wrapCard.stats?.unknowns > 0) && (
              <div className="wrap-review">
                {wrapCard.stats.needs_review || wrapCard.stats.unknowns} need review · adjudicated{' '}
                {wrapCard.stats.adjudicated_rate ?? 0}%
              </div>
            )}
            {wrapCard.best_target && (
              <div className="wrap-target">
                Top target: <strong>{wrapCard.best_target.label}</strong> (
                {wrapCard.best_target.hits}H)
              </div>
            )}
            {wrapCard.cues?.length > 0 && (
              <ul className="wrap-cues">
                {wrapCard.cues.map((c, i) => (
                  <li key={i}>{c}</li>
                ))}
              </ul>
            )}
            {wrapCard.coaching && <div className="wrap-coach">{wrapCard.coaching}</div>}
            {wrapCard.coaching_locked && (
              <div className="wrap-locked">Full coach notes unlock on Pro.</div>
            )}
            <div className="wrap-share">{wrapCard.share_line}</div>
            {wrapCard.needs_coach_seal && (
              <div className="review-nudge">
                {wrapCard.export_blocked_reason ||
                  'Seal as coach-ready before downloading the full report.'}
              </div>
            )}
            {wrapCard.coach_locked && (
              <div className="review-nudge" style={{ color: 'var(--teal)' }}>
                ✓ Coach-sealed — safe to share
              </div>
            )}
            <div className="modal-actions">
              <button
                type="button"
                className="btn"
                onClick={() => {
                  navigator.clipboard?.writeText(wrapCard.share_line || '')
                }}
              >
                Copy share line
              </button>
              {session && wrapCard.export_allowed && (
                <a
                  className="btn primary"
                  href={api.wrapExportUrl(session.id, 'html')}
                  target="_blank"
                  rel="noreferrer"
                >
                  Download HTML
                </a>
              )}
              {session && wrapCard.export_allowed && (
                <a
                  className="btn"
                  href={api.wrapExportUrl(session.id, 'md')}
                  target="_blank"
                  rel="noreferrer"
                >
                  Download MD
                </a>
              )}
              {session && wrapCard.needs_coach_seal && (
                <button
                  type="button"
                  className="btn primary"
                  onClick={() => {
                    setWrapCard(null)
                    setShowReviewWizard(true)
                  }}
                >
                  Fix & seal first
                </button>
              )}
              {session && !wrapCard.export_allowed && !wrapCard.needs_coach_seal && (
                <button
                  type="button"
                  className="btn primary"
                  onClick={() => {
                    setWrapCard(null)
                    setScreen('landing')
                    localStorage.setItem('ironsight_screen', 'landing')
                  }}
                >
                  Upgrade for export
                </button>
              )}
              <button type="button" className="btn" onClick={() => setWrapCard(null)}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {showLibrary && (
        <div className="modal-backdrop" onClick={() => setShowLibrary(false)}>
          <div className="modal library" onClick={(e) => e.stopPropagation()}>
            <h3>My practices</h3>
            <p className="lib-sub">Everything you’ve uploaded or opened (including the demo).</p>
            <div className="lib-grid">
              {sessions.map((s) => (
                <button
                  key={s.id}
                  className={`lib-card ${session?.id === s.id ? 'active' : ''}`}
                  onClick={() => {
                    openSession(s.id)
                    setShowLibrary(false)
                  }}
                >
                  <div className="lib-name">{(s.name || s.id).toUpperCase()}</div>
                  <div className="lib-meta">
                    <span>{s.stats?.total_shots ?? 0} shots</span>
                    <span>{fmtShort(s.stats?.duration_s || 0)}</span>
                    <span>{s.status}</span>
                  </div>
                  <div className="lib-tags">
                    {s.stats?.accuracy_display ? (
                      <span className="ltag">{s.stats.accuracy_display}</span>
                    ) : s.stats?.accuracy != null ? (
                      <span className="ltag">SCORE {s.stats.accuracy}%</span>
                    ) : null}
                    {(s.gaussian_splat || s.status === 'ready') && (
                      <span className="ltag live">3DGS</span>
                    )}
                    <span className="ltag">{sessionModeLabel(s)}</span>
                  </div>
                  <div className="lib-id">{s.id}</div>
                </button>
              ))}
              {!sessions.length && (
                <p style={{ color: 'var(--text-dim)' }}>No sessions yet. Create one.</p>
              )}
            </div>
            <div className="modal-actions">
              <button className="btn" onClick={() => setShowLibrary(false)}>
                Close
              </button>
              <button
                className="btn primary"
                onClick={() => {
                  setShowLibrary(false)
                  setShowNew(true)
                }}
              >
                + New Session
              </button>
            </div>
          </div>
        </div>
      )}

      {showReviewWizard && session && (
        <div className="review-dock">
          <ReviewWizard
            session={session}
            queue={reviewQueue}
            activeShot={activeShot}
            busy={busy}
            onSelectShot={selectShot}
            onClassify={classifyReview}
            onSeal={sealCoach}
            onClose={() => setShowReviewWizard(false)}
          />
        </div>
      )}

      {showTeamHub && (
        <TeamHub
          user={user}
          session={session}
          onClose={() => setShowTeamHub(false)}
          onMessage={(m) => setPipelineMsg(m)}
          onError={(m) => setError(friendlyError(m))}
        />
      )}

      {showSettings && (
        <div className="modal-backdrop" onClick={() => setShowSettings(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h3>Settings</h3>
            <p className="lib-sub">Account · {user?.email || 'not signed in'}</p>
            <label>Current password</label>
            <input type="password" value={pwOld} onChange={(e) => setPwOld(e.target.value)} />
            <label>New password</label>
            <input type="password" value={pwNew} onChange={(e) => setPwNew(e.target.value)} />
            <div className="modal-actions">
              <button
                type="button"
                className="btn primary"
                disabled={busy || !pwOld || !pwNew}
                onClick={async () => {
                  setBusy(true)
                  try {
                    await api.changePassword(pwOld, pwNew)
                    setPipelineMsg('Password updated')
                    setPwOld('')
                    setPwNew('')
                    setShowSettings(false)
                  } catch (e) {
                    setError(friendlyError(e.message || e))
                  } finally {
                    setBusy(false)
                  }
                }}
              >
                Change password
              </button>
              <button
                type="button"
                className="btn"
                onClick={async () => {
                  if (!window.confirm('Delete your account permanently?')) return
                  try {
                    await api.deleteAccount()
                    setToken(null)
                    setUser(null)
                    setScreen('landing')
                    localStorage.setItem('ironsight_screen', 'landing')
                  } catch (e) {
                    setError(friendlyError(e.message || e))
                  }
                }}
              >
                Delete account
              </button>
              <button type="button" className="btn" onClick={() => setShowSettings(false)}>
                Close
              </button>
            </div>
            <p className="lib-sub" style={{ marginTop: 12 }}>
              Legal:{' '}
              <a href="/api/legal/terms" target="_blank" rel="noreferrer">
                Terms
              </a>{' '}
              ·{' '}
              <a href="/api/legal/privacy" target="_blank" rel="noreferrer">
                Privacy
              </a>{' '}
              ·{' '}
              <a href="/api/legal/refund" target="_blank" rel="noreferrer">
                Refunds
              </a>{' '}
              ·{' '}
              <a href="/api/legal/sla" target="_blank" rel="noreferrer">
                Team SLA
              </a>
            </p>
          </div>
        </div>
      )}

      {showNew && (
        <div className="modal-backdrop" onClick={() => setShowNew(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h3>New practice</h3>
            <p className="lib-sub">Give it a name, then pick your video files (phone MP4 is fine).</p>
            <label>Practice name</label>
            <input
              type="text"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              placeholder="Saturday steel · week 3"
            />
            <label>Your view (shooter / helmet / phone)</label>
            <input
              type="file"
              accept="video/*"
              onChange={(e) => setShooterFile(e.target.files?.[0] || null)}
            />
            {shooterFile && <span className="file-chip">{shooterFile.name}</span>}
            <label>Side view camera (optional but better)</label>
            <input
              type="file"
              accept="video/*"
              onChange={(e) => setObserverFile(e.target.files?.[0] || null)}
            />
            {observerFile && <span className="file-chip">{observerFile.name}</span>}
            <div className="modal-actions">
              <button type="button" className="btn" onClick={() => setShowNew(false)}>
                Cancel
              </button>
              <button type="button" className="btn primary" disabled={busy} onClick={createAndMaybeUpload}>
                {busy ? 'Uploading…' : 'Save & continue'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
