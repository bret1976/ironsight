import React, { useEffect, useState } from 'react'
import { api, setToken } from './api'

/**
 * Marketing + auth — Linear Precision × Magic UI Bento.
 * Design refs: linear.app design system + magicui.design bento grid.
 */
export default function Landing({ onEnterApp, onEnterGods, user, onAuth }) {
  const [plans, setPlans] = useState([])
  const [mode, setMode] = useState('login')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [name, setName] = useState('')
  const [err, setErr] = useState(null)
  const [info, setInfo] = useState(null)
  const [busy, setBusy] = useState(false)
  const [stripeOk, setStripeOk] = useState(false)
  const [resetToken, setResetToken] = useState('')

  useEffect(() => {
    api
      .plans()
      .then((r) => {
        setPlans(r.plans || [])
        setStripeOk(!!r.stripe_configured)
      })
      .catch(() => {})
    const q = new URLSearchParams(location.search)
    if (q.get('reset')) {
      setResetToken(q.get('reset'))
      setMode('reset')
    }
    if (q.get('invite')) {
      setInfo('You have a team invite — sign in with the invited email.')
      sessionStorage.setItem('ironsight_invite', q.get('invite'))
    }
    if (q.get('magic')) {
      const tok = q.get('magic')
      setBusy(true)
      api
        .magicConsume(tok)
        .then((r) => {
          setToken(r.token)
          onAuth?.(r.user)
          onEnterApp?.()
          window.history.replaceState({}, '', '/')
        })
        .catch((ex) => setErr(String(ex.message || ex)))
        .finally(() => setBusy(false))
    }
  }, [])

  const submit = async (e) => {
    e.preventDefault()
    setBusy(true)
    setErr(null)
    setInfo(null)
    try {
      if (mode === 'forgot') {
        const r = await api.forgotPassword(email)
        setInfo(
          r.link
            ? `Reset link (dev): ${r.link}`
            : 'If that email is on file, we sent a reset link.'
        )
        return
      }
      if (mode === 'reset') {
        await api.resetPassword(resetToken, password)
        setInfo('Password updated — you can sign in now.')
        setMode('login')
        return
      }
      const r =
        mode === 'login'
          ? await api.login(email, password)
          : await api.register(email, password, name)
      setToken(r.token)
      onAuth?.(r.user)
      const inv = sessionStorage.getItem('ironsight_invite')
      if (inv) {
        try {
          await api.acceptInvite(inv)
          sessionStorage.removeItem('ironsight_invite')
        } catch {}
      }
      onEnterApp?.()
    } catch (ex) {
      setErr(String(ex.message || ex))
    } finally {
      setBusy(false)
    }
  }

  const upgrade = async (planId) => {
    if (!user || user.anonymous) {
      setMode('register')
      setErr('Make a free account first, then pick a plan.')
      return
    }
    setBusy(true)
    setErr(null)
    try {
      if (stripeOk) {
        const r = await api.checkout(planId)
        if (r.checkout_url) window.location.href = r.checkout_url
        return
      }
      const r = await api.devSetPlan(planId)
      onAuth?.(r.user)
      setInfo(`${(r.user?.plan || planId).toUpperCase()} plan is on. Open the studio to use it.`)
      onEnterApp?.()
    } catch (ex) {
      setErr(String(ex.message || ex))
    } finally {
      setBusy(false)
    }
  }

  const defaultPlans = [
    {
      id: 'free',
      name: 'Free',
      price_usd: 0,
      description: 'Try it with short clips',
      sessions_per_month: 3,
      max_video_minutes: 5,
      gsplat: false,
      coaching: false,
      wrap_export: false,
    },
    {
      id: 'pro',
      name: 'Pro',
      price_usd: 49,
      description: 'Coach workflow + 3 seats',
      sessions_per_month: 40,
      max_video_minutes: 30,
      gsplat: true,
      coaching: true,
      wrap_export: true,
      pose: true,
      seats: 3,
    },
    {
      id: 'team',
      name: 'Team',
      price_usd: 149,
      description: 'Range ops: seats, lanes, priority queue',
      sessions_per_month: 200,
      max_video_minutes: 90,
      gsplat: true,
      coaching: true,
      wrap_export: true,
      mappers: true,
      seats: 10,
      priority: true,
      lanes: true,
    },
  ]

  return (
    <div className="lp">
      <div className="lp-bg" aria-hidden />

      <header className="lp-nav">
        <div className="lp-brand">
          <span className="lp-mark" />
          <span className="lp-word">IronSight</span>
        </div>
        <nav className="lp-nav-links">
          <a href="#how">Product</a>
          <a href="#pricing">Pricing</a>
        </nav>
        <div className="lp-nav-cta">
          {user && !user.anonymous ? (
            <>
              <span className="lp-user-pill">
                {(user.plan || 'free').toUpperCase()} · {user.email?.split('@')[0]}
              </span>
              <button type="button" className="btn-glow" onClick={() => onEnterApp?.()}>
                Open studio
              </button>
            </>
          ) : (
            <>
              <button type="button" className="btn-ghost" onClick={() => setMode('login')}>
                Sign in
              </button>
              <button type="button" className="btn-glow" onClick={() => onEnterApp?.()}>
                Get started
              </button>
            </>
          )}
        </div>
      </header>

      <main className="lp-main">
        <section className="lp-hero">
          <p className="lp-eyebrow">Range video · hit / miss · coach-ready</p>
          <h1 className="lp-title">
            Did you hit?
            <br />
            <span className="lp-title-grad">See every shot clearly.</span>
          </h1>
          <p className="lp-lead">
            Upload practice video. IronSight finds each bang, scores hit or miss, syncs dual
            cameras, and gives you a clean review you can share with a coach.
          </p>
          <div className="lp-hero-actions">
            <button type="button" className="btn-glow lg" onClick={() => onEnterApp?.()}>
              Open free studio
            </button>
            <button type="button" className="btn-ghost lg" onClick={() => onEnterGods?.()}>
              See-through map
            </button>
          </div>
          <div className="lp-trust-row">
            <span>Phone video only</span>
            <span className="dot" />
            <span>Fix AI in one click</span>
            <span className="dot" />
            <span>Share a coach wrap</span>
          </div>

          <div className="lp-stats" aria-label="Highlights">
            <div className="lp-stat">
              <div className="lp-stat-v">Dual-cam</div>
              <div className="lp-stat-l">One locked timeline</div>
            </div>
            <div className="lp-stat">
              <div className="lp-stat-v">Hit / miss</div>
              <div className="lp-stat-l">AI + human seal</div>
            </div>
            <div className="lp-stat">
              <div className="lp-stat-v">3D bay</div>
              <div className="lp-stat-l">Pro fly-around</div>
            </div>
          </div>
        </section>

        {/* Product screenshot frame (Linear marketing rhythm) */}
        <div className="lp-product-frame" aria-hidden>
          <div className="lp-product-chrome">
            <i />
            <i />
            <i />
          </div>
          <div className="lp-product-body">
            <div className="lp-mock-side">
              <div className="lp-mock-line accent w80" />
              <div className="lp-mock-line w60" />
              <div className="lp-mock-line w80" />
              <div className="lp-mock-line w60" />
              <div className="lp-mock-line accent w80" />
            </div>
            <div className="lp-mock-stage">
              <div className="big">Shooter · Observer · Timeline</div>
              <div className="lp-mock-hit">
                <span>✓ Hit 14</span>
                <span>✗ Miss 3</span>
              </div>
            </div>
            <div className="lp-mock-side">
              <div className="lp-mock-line accent w80" />
              <div className="lp-mock-line w60" />
              <div className="lp-mock-line w80" />
              <div className="lp-mock-line w60" />
              <div className="lp-mock-line w80" />
            </div>
          </div>
        </div>

        {/* Magic UI Bento feature grid */}
        <section className="lp-how" id="how">
          <h2>Built for practice, not jargon</h2>
          <p className="lp-section-sub">
            Three steps. Clear results. You stay in control of every call.
          </p>
          <div className="lp-bento">
            <article className="lp-bento-card wide">
              <div className="lp-bento-ico">1</div>
              <h3>Upload & analyze</h3>
              <p>
                Drop shooter video (and an observer angle if you have one). Hit analyze — we find
                bangs, score hit / miss, and lock both cameras to one scrubber.
              </p>
            </article>
            <article className="lp-bento-card">
              <div className="lp-bento-ico">2</div>
              <h3>Review shots</h3>
              <p>Jump shot-to-shot. Fix anything the AI got wrong with one tap.</p>
            </article>
            <article className="lp-bento-card">
              <div className="lp-bento-ico">3</div>
              <h3>Share wrap</h3>
              <p>Export a short coach-ready report — not a wall of technical noise.</p>
            </article>
            <article className="lp-bento-card">
              <div className="lp-bento-ico">◎</div>
              <h3>Dual-cam sync</h3>
              <p>Shooter + side view stay locked. Scrub once, both move.</p>
            </article>
            <article className="lp-bento-card wide">
              <div className="lp-bento-ico">◈</div>
              <h3>3D bay on Pro</h3>
              <p>
                Reconstruct the range from video and fly around the bay. Optional quality checks
                keep the splat honest before you show a coach.
              </p>
            </article>
          </div>
        </section>

        <section className="lp-split">
          <div className="lp-card glass auth-panel">
            <h2>
              {mode === 'login' && 'Welcome back'}
              {mode === 'register' && 'Create free account'}
              {mode === 'forgot' && 'Reset password'}
              {mode === 'reset' && 'Choose a new password'}
            </h2>
            <p className="lp-muted">
              {mode === 'register'
                ? 'Takes about 20 seconds. Free plan includes 3 short sessions.'
                : 'Same account on phone and laptop.'}
            </p>
            <form onSubmit={submit} className="lp-form">
              {mode === 'register' && (
                <label>
                  Your name
                  <input
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder="Alex"
                    autoComplete="name"
                  />
                </label>
              )}
              {mode !== 'reset' && (
                <label>
                  Email
                  <input
                    type="email"
                    required
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="you@school.edu"
                    autoComplete="email"
                  />
                </label>
              )}
              {(mode === 'login' || mode === 'register' || mode === 'reset') && (
                <label>
                  Password (8+ characters)
                  <input
                    type="password"
                    required
                    minLength={8}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
                  />
                </label>
              )}
              {err && <div className="lp-err">{err}</div>}
              {info && <div className="lp-ok">{info}</div>}
              <button className="btn-glow full" type="submit" disabled={busy}>
                {busy
                  ? 'Working…'
                  : mode === 'login'
                  ? 'Sign in'
                  : mode === 'register'
                  ? 'Create account'
                  : mode === 'forgot'
                  ? 'Email me a link'
                  : 'Save password'}
              </button>
            </form>
            <div className="lp-form-links">
              <button
                type="button"
                className="text-link"
                onClick={() => setMode(mode === 'login' ? 'register' : 'login')}
              >
                {mode === 'login' ? 'New here? Create account' : 'Already have an account? Sign in'}
              </button>
              {mode === 'login' && (
                <button type="button" className="text-link" onClick={() => setMode('forgot')}>
                  Forgot password?
                </button>
              )}
              {mode === 'login' && (
                <button
                  type="button"
                  className="text-link"
                  onClick={async () => {
                    if (!email) {
                      setErr('Enter your email above, then click email sign-in link.')
                      return
                    }
                    setBusy(true)
                    setErr(null)
                    try {
                      const r = await api.magicLink(email)
                      setInfo(
                        r.link
                          ? `Dev magic link: ${r.link}`
                          : 'Check your email for a one-tap sign-in link (20 min).'
                      )
                    } catch (ex) {
                      setErr(String(ex.message || ex))
                    } finally {
                      setBusy(false)
                    }
                  }}
                >
                  Email me a sign-in link (no password)
                </button>
              )}
            </div>
          </div>

          <div className="lp-card glass feature-panel">
            <h2>What you get</h2>
            <ul className="lp-feature-list">
              <li>
                <span className="ico">◎</span>
                <div>
                  <strong>Hit / miss board</strong>
                  <p>Every bang listed with a clear result you can fix</p>
                </div>
              </li>
              <li>
                <span className="ico">◫</span>
                <div>
                  <strong>Two cameras, one timeline</strong>
                  <p>Shooter + side view stay locked together</p>
                </div>
              </li>
              <li>
                <span className="ico">◈</span>
                <div>
                  <strong>Optional 3D fly-around</strong>
                  <p>Pro reconstructs the range so you can look around</p>
                </div>
              </li>
              <li>
                <span className="ico">▤</span>
                <div>
                  <strong>Shareable wrap</strong>
                  <p>A short report for coaches — not jargon</p>
                </div>
              </li>
            </ul>
            <p className="lp-note">
              Video software only — not radar and not a military system. You (or your coach) make
              the final call.
            </p>
          </div>
        </section>

        <section className="lp-pricing" id="pricing">
          <h2>Simple pricing</h2>
          <p className="lp-section-sub">
            {stripeOk
              ? 'Secure card checkout is ready.'
              : 'Accounts work now; connect Stripe live keys to charge cards.'}
          </p>
          <div className="lp-price-grid">
            {(plans.length ? plans : defaultPlans).map((p) => (
              <div key={p.id} className={`lp-price-card ${p.id === 'pro' ? 'is-featured' : ''}`}>
                {p.id === 'pro' && <div className="lp-badge">Popular</div>}
                <h3>{p.name}</h3>
                <div className="lp-price">
                  ${p.price_usd}
                  <span>/mo</span>
                </div>
                <p className="lp-price-desc">{p.description}</p>
                <ul>
                  <li>{p.sessions_per_month} practice sessions / month</li>
                  <li>Videos up to {p.max_video_minutes || '—'} minutes</li>
                  <li>
                    {p.coaching !== false ? 'Full coach notes' : 'Scoreboard only (upgrade for notes)'}
                  </li>
                  <li>{p.wrap_export ? 'Download shareable report' : 'Report preview'}</li>
                  <li>{p.gsplat ? '3D fly-around + quality check' : 'No 3D fly-around yet'}</li>
                  <li>
                    {(p.seats || 1) > 1 ? `${p.seats} seats (invite helpers)` : '1 seat'}
                  </li>
                  {p.priority && <li>Priority analyze queue</li>}
                  {p.lanes && <li>Lane schedule + range dashboard</li>}
                </ul>
                {p.price_usd > 0 ? (
                  <button
                    type="button"
                    className="btn-glow full"
                    disabled={busy}
                    onClick={() => upgrade(p.id)}
                  >
                    {p.id === 'pro'
                      ? stripeOk
                        ? 'Start Pro'
                        : 'Choose Pro'
                      : stripeOk
                      ? `Get ${p.name}`
                      : `Choose ${p.name}`}
                  </button>
                ) : (
                  <button type="button" className="btn-ghost full" onClick={() => setMode('register')}>
                    Start free
                  </button>
                )}
              </div>
            ))}
          </div>
        </section>
      </main>

      <footer className="lp-foot">
        <span>IronSight — practice video, made clear</span>
        <div className="lp-legal">
          <a href="/api/legal/terms" target="_blank" rel="noreferrer">
            Terms
          </a>
          <a href="/api/legal/privacy" target="_blank" rel="noreferrer">
            Privacy
          </a>
          <a href="/api/legal/refund" target="_blank" rel="noreferrer">
            Refunds
          </a>
        </div>
      </footer>
    </div>
  )
}
