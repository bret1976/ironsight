import React from 'react'

/**
 * One-shot-at-a-time coach review — big Hit / Miss buttons.
 */
export default function ReviewWizard({
  session,
  queue,
  activeShot,
  onSelectShot,
  onClassify,
  onSeal,
  onClose,
  busy,
}) {
  const remaining = queue?.needs_review_count ?? 0
  const total = queue?.total_shots ?? 0
  const done = Math.max(0, total - remaining)
  const pct = total ? Math.round((done / total) * 100) : 0
  const nextId = queue?.next_shot_id
  const shot =
    activeShot ||
    (session?.shots || []).find((s) => s.id === nextId) ||
    (session?.shots || [])[0]

  if (!queue || remaining === 0) {
    return (
      <div className="review-wizard glass">
        <div className="rw-head">
          <h3>All shots checked ✓</h3>
          <button type="button" className="mini-btn" onClick={onClose}>
            Close
          </button>
        </div>
        <p className="rw-sub">Seal this practice so the report is coach-ready.</p>
        <button type="button" className="btn-glow full" disabled={busy} onClick={onSeal}>
          Seal as coach-ready
        </button>
      </div>
    )
  }

  return (
    <div className="review-wizard glass">
      <div className="rw-head">
        <h3>Check this shot</h3>
        <button type="button" className="mini-btn" onClick={onClose}>
          Minimize
        </button>
      </div>
      <div className="rw-progress">
        <div className="rw-bar">
          <div style={{ width: `${pct}%` }} />
        </div>
        <span>
          {done} of {total} checked · {remaining} left
        </span>
      </div>
      {shot && (
        <>
          <div className="rw-shot">
            <div className="rw-shot-id">Shot #{shot.id}</div>
            <div className="rw-shot-meta">
              Time {Number(shot.timestamp_s || 0).toFixed(2)}s
              {shot.target_label ? ` · ${shot.target_label}` : ''}
            </div>
            <div className={`rw-current tag ${String(shot.classification || '').toLowerCase()}`}>
              AI says: {shot.classification || 'UNKNOWN'}
              {shot.confidence != null
                ? ` (${Math.round(Number(shot.confidence) * 100)}%)`
                : ''}
            </div>
            {shot.reasoning && <p className="rw-reason">{shot.reasoning}</p>}
          </div>
          <div className="rw-actions">
            <button
              type="button"
              className="rw-hit"
              disabled={busy}
              onClick={() => onClassify(shot.id, 'HIT')}
            >
              ✓ Hit
            </button>
            <button
              type="button"
              className="rw-miss"
              disabled={busy}
              onClick={() => onClassify(shot.id, 'MISS')}
            >
              ✗ Miss
            </button>
            <button
              type="button"
              className="mini-btn"
              disabled={busy}
              onClick={() => onClassify(shot.id, 'NOT_A_SHOT')}
            >
              Not a shot
            </button>
            <button
              type="button"
              className="mini-btn"
              disabled={busy}
              onClick={() => onClassify(shot.id, 'UNKNOWN')}
            >
              Still unsure
            </button>
          </div>
          {nextId != null && shot.id !== nextId && (
            <button
              type="button"
              className="text-link"
              onClick={() => {
                const s = (session?.shots || []).find((x) => x.id === nextId)
                if (s) onSelectShot(s)
              }}
            >
              Jump to next unsure → #{nextId}
            </button>
          )}
        </>
      )}
    </div>
  )
}
