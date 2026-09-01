import React, { useEffect, useState } from 'react'
import { api } from './api'

/**
 * Full Team / Pro collab surface: invites, dashboard, lanes, SLA.
 */
export default function TeamHub({ user, session, onClose, onMessage, onError }) {
  const [orgs, setOrgs] = useState([])
  const [org, setOrg] = useState(null)
  const [dash, setDash] = useState(null)
  const [inviteEmail, setInviteEmail] = useState('')
  const [laneName, setLaneName] = useState('')
  const [bookTitle, setBookTitle] = useState('')
  const [bookStart, setBookStart] = useState('')
  const [bookEnd, setBookEnd] = useState('')
  const [sla, setSla] = useState(null)
  const [jobs, setJobs] = useState([])
  const [busy, setBusy] = useState(false)

  const load = async () => {
    try {
      const r = await api.orgs()
      const list = r.orgs || []
      setOrgs(list)
      const oid = user?.org_id || list[0]?.id
      if (oid) {
        try {
          const full = await api.getOrg(oid)
          setOrg(full)
        } catch {
          setOrg(list.find((x) => x.id === oid) || list[0] || null)
        }
        const d = await api.orgDashboard(oid)
        setDash(d)
        try {
          setSla(await api.orgSla(oid))
        } catch {}
      }
      try {
        const j = await api.jobs()
        setJobs(j.jobs || [])
      } catch {}
    } catch (e) {
      onError?.(String(e.message || e))
    }
  }

  useEffect(() => {
    load()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user?.org_id])

  const run = async (fn) => {
    setBusy(true)
    try {
      await fn()
      await load()
    } catch (e) {
      onError?.(String(e.message || e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal team-hub" onClick={(e) => e.stopPropagation()}>
        <div className="rw-head">
          <h3>Team & range</h3>
          <button type="button" className="mini-btn" onClick={onClose}>
            Close
          </button>
        </div>

        {!org && (
          <div className="team-panel">
            <p className="lib-sub">
              Pro = 3 seats · Team = 10 seats. Create a group to invite people.
            </p>
            <button
              type="button"
              className="btn-glow full"
              disabled={busy || (user?.plan !== 'pro' && user?.plan !== 'team')}
              onClick={() =>
                run(async () => {
                  const r = await api.createOrg(`${user?.name || 'My'} team`)
                  onMessage?.(`Created · ${r.org?.seat_limit} seats`)
                })
              }
            >
              Create {user?.plan === 'pro' ? 'coach team (3)' : 'range team'}
            </button>
          </div>
        )}

        {org && (
          <>
            <div className="team-panel">
              <div className="label">{org.name}</div>
              <div style={{ fontSize: 12, color: 'var(--text-dim)' }}>
                Seats {org.seat_used ?? org.members?.length ?? 0}/{org.seat_limit} · plan{' '}
                {org.plan}
              </div>
              {(org.members || []).map((m) => (
                <div key={m.user_id} style={{ fontSize: 12, marginTop: 4 }}>
                  {m.email || m.user_id} · {m.role}
                </div>
              ))}
            </div>

            <div className="team-panel">
              <div className="label">Invite a seat</div>
              <input
                placeholder="email@range.com"
                value={inviteEmail}
                onChange={(e) => setInviteEmail(e.target.value)}
              />
              <button
                type="button"
                className="mini-btn primary"
                disabled={busy || !inviteEmail}
                onClick={() =>
                  run(async () => {
                    const r = await api.inviteOrg(org.id, inviteEmail)
                    onMessage?.(r.link ? `Invite link: ${r.link}` : `Invited ${inviteEmail}`)
                    setInviteEmail('')
                  })
                }
              >
                Send invite
              </button>
            </div>

            {dash && (
              <div className="team-panel">
                <div className="label">Range dashboard</div>
                <div className="rw-progress">
                  <span>
                    {dash.summary?.sessions || 0} practices ·{' '}
                    {dash.summary?.needs_review_shots || 0} shots need review ·{' '}
                    {dash.summary?.coach_ready_sessions || 0} sealed
                  </span>
                </div>
                <p className="lib-sub">{dash.value_story}</p>
                <div className="lib-grid" style={{ maxHeight: 160 }}>
                  {(dash.sessions || []).slice(0, 8).map((s) => (
                    <div key={s.id} className="lib-card">
                      <div className="lib-name">{s.name || s.id}</div>
                      <div className="lib-meta">
                        <span>{s.shots} shots</span>
                        <span>{s.needs_review} to check</span>
                        <span>{s.coach_locked ? 'sealed' : s.status}</span>
                      </div>
                      {session?.id === s.id ? null : (
                        <button
                          type="button"
                          className="mini-btn"
                          style={{ marginTop: 6 }}
                          disabled={busy || !user?.id}
                          onClick={() =>
                            run(async () => {
                              await api.assignSession(org.id, s.id, user.id, 'coach')
                              onMessage?.(`Assigned ${s.name || s.id} to you`)
                            })
                          }
                        >
                          Assign to me
                        </button>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {user?.plan === 'team' && (
              <div className="team-panel">
                <div className="label">Lanes</div>
                <input
                  placeholder="Bay 1"
                  value={laneName}
                  onChange={(e) => setLaneName(e.target.value)}
                />
                <button
                  type="button"
                  className="mini-btn"
                  disabled={busy || !laneName}
                  onClick={() =>
                    run(async () => {
                      await api.createLane(org.id, laneName)
                      setLaneName('')
                      onMessage?.('Lane added')
                    })
                  }
                >
                  Add lane
                </button>
                {(dash?.lanes || []).map((l) => (
                  <div key={l.id} style={{ fontSize: 12, marginTop: 4 }}>
                    {l.name}
                  </div>
                ))}
                <div className="label" style={{ marginTop: 10 }}>
                  Book a lane
                </div>
                <input
                  placeholder="Title"
                  value={bookTitle}
                  onChange={(e) => setBookTitle(e.target.value)}
                />
                <input
                  type="datetime-local"
                  value={bookStart}
                  onChange={(e) => setBookStart(e.target.value)}
                />
                <input
                  type="datetime-local"
                  value={bookEnd}
                  onChange={(e) => setBookEnd(e.target.value)}
                />
                <button
                  type="button"
                  className="mini-btn"
                  disabled={busy || !bookTitle || !bookStart || !bookEnd || !dash?.lanes?.[0]}
                  onClick={() =>
                    run(async () => {
                      await api.bookLane(org.id, {
                        lane_id: dash.lanes[0].id,
                        title: bookTitle,
                        starts_at: new Date(bookStart).toISOString(),
                        ends_at: new Date(bookEnd).toISOString(),
                      })
                      setBookTitle('')
                      onMessage?.('Lane booked')
                    })
                  }
                >
                  Book first lane
                </button>
                {(dash?.bookings || []).slice(0, 5).map((b) => (
                  <div key={b.id} style={{ fontSize: 11, color: 'var(--text-dim)', marginTop: 4 }}>
                    {b.title} · {b.starts_at}
                  </div>
                ))}
              </div>
            )}

            <div className="team-panel">
              <div className="label">SLA</div>
              <a href="/api/legal/sla" target="_blank" rel="noreferrer" className="text-link">
                Read Team SLA contract
              </a>
              {sla && (
                <div style={{ fontSize: 11, color: 'var(--text-dim)', marginTop: 6 }}>
                  Acceptances: {sla.acceptances?.length || 0} · Live:{' '}
                  {sla.live_metrics?.status || '—'}
                </div>
              )}
              <button
                type="button"
                className="mini-btn"
                disabled={busy}
                onClick={() =>
                  run(async () => {
                    await api.acceptSla(org.id)
                    onMessage?.('SLA accepted')
                  })
                }
              >
                Accept SLA
              </button>
            </div>
          </>
        )}

        <div className="team-panel">
          <div className="label">Analyze jobs</div>
          {(jobs || []).length === 0 && (
            <div style={{ fontSize: 12, color: 'var(--text-dim)' }}>No jobs yet</div>
          )}
          {(jobs || []).slice(0, 8).map((j) => (
            <div key={j.id} style={{ fontSize: 11, marginTop: 4 }}>
              {j.kind} · {j.status} · {j.session_id?.slice?.(0, 8)}
              {j.priority ? ` · pri ${j.priority}` : ''}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
