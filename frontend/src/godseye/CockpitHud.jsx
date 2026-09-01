import React, { useEffect, useMemo, useRef, useState } from 'react'
import { estimateGsd, formatAlt, formatDms, formatLatLon, formatMgrs } from './geo.js'
import { sceneStillKind, tilesAltitudeOk } from './scenes.js'
import { ageLabel, isGroundIntel, satStillZoom, satelliteStillUrl } from './layers.js'
import { serializeGodsEyeState } from './urlState.js'
import CctvFrame from './CctvFrame.jsx'

export const SENSOR_META = {
  rgb: { key: '1', code: 'EO', hint: 'Electro-optical daylight' },
  crt: { key: '2', code: 'CRT', hint: 'Scanline phosphor tube' },
  nvg: { key: '3', code: 'NVG', hint: 'Night-vision green phosphor' },
  flir: { key: '4', code: 'FLIR', hint: 'Infrared ironbow' },
  ironbell: { key: '5', code: 'BELL', hint: 'Hot iron-bell thermal' },
  noir: { key: '6', code: 'NOIR', hint: 'High-contrast monochrome' },
  snow: { key: '7', code: 'SNOW', hint: 'White-hot inverted look' },
}

export const LAYER_META = [
  { id: 'flights', name: 'Live flights', blurb: 'Airborne tracks over photoreal' },
  { id: 'military', name: 'Military', blurb: 'Mil ADS-B, same sky' },
  { id: 'ships', name: 'Vessels', blurb: 'AIS on the water' },
  { id: 'sats', name: 'Satellites', blurb: 'Orbit catalog · ISS' },
  { id: 'missions', name: 'Space missions', blurb: 'Pads and reconstructed ascent' },
  { id: 'quakes', name: 'Quakes', blurb: 'USGS epicenter · satellite dive' },
  { id: 'fires', name: 'Fires', blurb: 'NASA FIRMS · FLIR over the burn' },
  { id: 'storms', name: 'Storms', blurb: 'EONET storms + NWS alerts' },
  { id: 'traffic', name: 'Traffic', blurb: 'Simulated street flow' },
  { id: 'cctv', name: 'CCTV', blurb: 'City cameras on the map' },
  { id: 'sites', name: 'Sites', blurb: 'Bases, dams, data centers' },
  { id: 'radio', name: 'Radio', blurb: 'Public stations you can tune' },
]

const MISSIONS = [
  {
    id: 'contacts',
    title: 'Live Contacts',
    blurb: 'Light up the sky. Lock a live aircraft, ride the trail, drop into its cockpit.',
  },
  {
    id: 'cctv',
    title: 'Public cameras',
    blurb: 'Austin, London, California. Jump a live CCTV, see the feed in the city, cycle viewsheds.',
  },
  {
    id: 'space',
    title: 'Space Missions',
    blurb: 'Pads from Launch Library 2. Track the ISS or replay a reconstructed ascent.',
  },
  {
    id: 'enviro',
    title: 'Environmental',
    blurb: 'Fires go FLIR over the burn. Quakes dive the epicenter. Storms show EONET + NWS alerts.',
  },
]

function SatStill({ lat, lon, mapsKey, kind, label }) {
  const [ok, setOk] = useState(true)
  const src = satelliteStillUrl(lat, lon, mapsKey, satStillZoom(kind))
  if (!src) return <p className="ge-sat-miss">No coordinates for a satellite still.</p>
  if (!ok) {
    return <p className="ge-sat-miss">Photoreal globe is the live image. Still unavailable.</p>
  }
  return (
    <img
      className="ge-sat-still"
      alt={`Satellite ${label || kind}`}
      src={src}
      onError={() => setOk(false)}
    />
  )
}

function zulu(d = new Date()) {
  const p = (n) => String(n).padStart(2, '0')
  return `${d.getUTCFullYear()}-${p(d.getUTCMonth() + 1)}-${p(d.getUTCDate())} ${p(d.getUTCHours())}:${p(d.getUTCMinutes())}:${p(d.getUTCSeconds())}Z`
}

function sceneLine({ look, sensor, tracked, flights, tilesOn, camMode }) {
  const place = tracked?.callsign || (look?.alt < 80_000 ? 'street scale' : look?.alt < 400_000 ? 'city scale' : 'orbital scale')
  const live = flights > 0 ? `${flights} live tracks` : 'awaiting tracks'
  const optic = (SENSOR_META[sensor] || SENSOR_META.rgb).code
  const tiles = tilesOn ? 'photoreal' : 'marble'
  const cam = camMode === 'cockpit' ? 'cockpit' : camMode === 'orbit' ? 'orbit' : 'free look'
  return `${optic} · ${place} · ${live} · ${tiles} · ${cam}`.toUpperCase()
}

export default function CockpitHud({
  look,
  heading,
  sensor,
  setSensor,
  layersOn,
  toggleLayer,
  focusedLayer,
  layerMeta,
  tracked,
  nearby,
  rosterKm,
  onSelect,
  onClearTrack,
  search,
  setSearch,
  onSearch,
  onReset,
  tilesStatus,
  flyHint,
  camMode,
  setCamMode,
  latencyMs,
  voiceOn,
  onVoice,
  listening,
  transcript,
  speakLine,
  mapsKey,
  onTrackIss,
  onOrbit,
  radioUrl,
  onTune,
  radios,
  hudOn,
  detectionOn,
  setDetectionOn,
  missionOpen,
  onMission,
  dismissMission,
  boxes,
  cctvStill,
  onCloseCctv,
  cctvCity,
  onCctvCity,
  viewshedOn,
  setViewshedOn,
  measure,
  replay,
  onReplay,
  onReplayRate,
  utc,
  onEnterRange,
  onToggleTiles,
}) {
  const sm = SENSOR_META[sensor] || SENSOR_META.rgb
  const inputRef = useRef(null)
  const [consoleOpen, setConsoleOpen] = useState(false)
  const [needle, setNeedle] = useState(0.18)
  const [copied, setCopied] = useState(false)
  const copyTimer = useRef(null)
  const flightCount = layerMeta?.flights?.count || 0

  const onShare = async () => {
    const hash = serializeGodsEyeState({
      lat: look?.lat,
      lon: look?.lon,
      alt: look?.alt,
      layers: layersOn,
      track: tracked?.id,
      sensor,
      cam: camMode,
    })
    const origin = typeof window !== 'undefined' ? window.location.origin : ''
    const url = `${origin}/${hash}`
    try {
      if (navigator.clipboard?.writeText) await navigator.clipboard.writeText(url)
      else throw new Error('no clipboard')
    } catch {
      try {
        const ta = document.createElement('textarea')
        ta.value = url
        ta.setAttribute('readonly', '')
        ta.style.position = 'fixed'
        ta.style.left = '-9999px'
        document.body.appendChild(ta)
        ta.select()
        document.execCommand('copy')
        ta.remove()
      } catch {
        return
      }
    }
    setCopied(true)
    if (copyTimer.current) window.clearTimeout(copyTimer.current)
    copyTimer.current = window.setTimeout(() => setCopied(false), 1600)
  }

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === '/' && !e.target.matches?.('input, textarea')) {
        e.preventDefault()
        inputRef.current?.focus()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => {
      window.removeEventListener('keydown', onKey)
      if (copyTimer.current) window.clearTimeout(copyTimer.current)
    }
  }, [])

  const stations = useMemo(() => (radios || []).filter((r) => r.url), [radios])

  const onNeedle = (v) => {
    setNeedle(v)
    if (!stations.length) return
    const i = Math.min(stations.length - 1, Math.round(v * (stations.length - 1)))
    const st = stations[i]
    if (st) onTune(st)
  }

  const tilesOn = Boolean(tilesStatus?.active) && tilesAltitudeOk(look?.alt)
  const summary = sceneLine({
    look,
    sensor,
    tracked,
    flights: flightCount,
    tilesOn,
    camMode,
  })

  return (
    <>
      {hudOn && (
        <>
          <div className="ge-letterbox top" aria-hidden />
          <div className="ge-letterbox bot" aria-hidden />
          <div className="ge-class-bar" aria-hidden>
            <span>LIVE OSINT // PUBLIC SIGNAL // NO CLEARANCE</span>
            <span>GEV-{4000 + Math.abs(Math.round((look?.lat || 0) * 17)) % 900}</span>
            <span>PAGE 1/1</span>
          </div>

          <div className="ge-corners" aria-hidden>
            <div className="ge-c tl">
              <div className="ge-brkt">┌</div>
              <div>
                <div className="ge-wordmark">GODS EYE</div>
                <div className="ge-cls">LIVE OSINT // PUBLIC SIGNAL</div>
                <div className="ge-sys">OPS-{4100 + Math.abs(Math.round((look?.lon || 0) * 11)) % 90}</div>
                <div className="ge-mode">{sm.code}</div>
                <div className="ge-sum">{speakLine || summary}</div>
              </div>
            </div>
            <div className="ge-c tr">
              <div>
                <div className="ge-rec">
                  <i /> REC {utc || zulu()}
                </div>
                <div>BAND PAN · BITS 11 · LVL 1A</div>
              </div>
              <div className="ge-brkt">┐</div>
            </div>
            <div className="ge-c bl">
              <div className="ge-brkt">└</div>
              <div>
                <div>MGRS {formatMgrs(look?.lat, look?.lon)}</div>
                <div>{formatDms(look?.lat, look?.lon)}</div>
              </div>
            </div>
            <div className="ge-c br">
              <div>
                <div>GSD {estimateGsd(look?.alt)} · NIIRS {look?.alt < 20_000 ? '5+' : look?.alt < 200_000 ? '3' : '1'}</div>
                <div>ALT {formatAlt(look?.alt)} · HDG {Number(heading || 0).toFixed(0).padStart(3, '0')}°</div>
                <div>
                  {latencyMs != null ? `${Math.max(1, Math.round(latencyMs))} ms` : '— ms'} ·{' '}
                  {tilesOn ? 'PHOTOREAL' : tilesStatus?.pending ? 'TILES…' : 'MARBLE'}
                </div>
              </div>
              <div className="ge-brkt">┘</div>
            </div>
          </div>

          <div className="ge-reticle" aria-hidden>
            <i />
            <b />
            <em />
          </div>
        </>
      )}

      {detectionOn &&
        (boxes || []).map((box) => {
          const altFt = box.alt_m != null ? Math.round((box.alt_m * 3.28084) / 100) * 100 : null
          const spd = box.gs_kts != null ? Math.round(box.gs_kts) : null
          const altTxt = altFt == null ? '' : altFt >= 1000 ? `${Math.round(altFt / 1000)}K` : String(altFt)
          return (
            <button
              key={box.id}
              type="button"
              className={`ge-callout ${box.kind} ${box.selected ? 'lock' : ''} ${box.military ? 'mil' : ''}`}
              style={{ left: box.x, top: box.y }}
              onClick={() => box.contact && onSelect(box.contact)}
            >
              <span className="ge-callout-craft" aria-hidden>
                <svg viewBox="0 0 32 16" width="28" height="14">
                  <path
                    d="M2 8 L12 7.2 L14 3 H16 L15.2 7.1 L26 6.6 L28 5 H30 L29 8 L30 11 H28 L26 9.4 L15.2 8.9 L16 13 H14 L12 8.8 Z"
                    fill="currentColor"
                  />
                </svg>
              </span>
              <span>
                <strong>{box.label}</strong>
                {(altTxt || spd != null) && (
                  <em>
                    {altTxt ? `ALT ${altTxt}` : ''}
                    {altTxt && spd != null ? ' · ' : ''}
                    {spd != null ? `${spd}KT` : ''}
                  </em>
                )}
              </span>
            </button>
          )
        })}

      {missionOpen && (
        <div className="ge-mission">
          <p className="ge-mission-k">First run</p>
          <h2>Stage a mission</h2>
          <p className="ge-mission-lead">
            Photoreal Earth. Live tracks. Talk to it. Pick a start — or explore the globe yourself.
          </p>
          <div className="ge-mission-grid ge-mission-grid-4">
            {MISSIONS.map((m) => (
              <button key={m.id} type="button" onClick={() => onMission(m.id)}>
                <strong>{m.title}</strong>
                <span>{m.blurb}</span>
              </button>
            ))}
          </div>
          <button type="button" className="ge-text" onClick={dismissMission}>
            Explore manually
          </button>
        </div>
      )}

      <div className="ge-strip">
        <button type="button" className="ge-word" onClick={() => setConsoleOpen((v) => !v)}>
          GODS EYE
        </button>
        <button type="button" className="ge-word" onClick={() => inputRef.current?.focus()}>
          LOCATION
        </button>
        <form
          className="ge-search"
          onSubmit={(e) => {
            e.preventDefault()
            onSearch()
          }}
        >
          <input
            ref={inputRef}
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Austin, London, California, or a target…"
            aria-label="Location"
          />
        </form>
        <button type="button" className={`ge-mic ${listening || voiceOn ? 'on' : ''}`} onClick={onVoice}>
          {listening ? 'LISTEN' : 'MIC'}
        </button>
        <span className="ge-preset-lab">PRESETS</span>
        <div className="ge-digits">
          {Object.entries(SENSOR_META).map(([k, v]) => (
            <button key={k} type="button" className={sensor === k ? 'on' : ''} onClick={() => setSensor(k)}>
              {v.key}
            </button>
          ))}
        </div>
        <button type="button" className={camMode === 'cockpit' ? 'on' : ''} onClick={() => setCamMode(camMode === 'cockpit' ? 'free' : 'cockpit')}>
          C
        </button>
        <button type="button" className={detectionOn ? 'on' : ''} onClick={() => setDetectionOn((v) => !v)}>
          D
        </button>
        <button type="button" className={camMode === 'orbit' ? 'on' : ''} onClick={onOrbit}>
          ORB
        </button>
        <button type="button" onClick={onTrackIss}>
          ISS
        </button>
        <button type="button" onClick={onReset}>
          GLOBE
        </button>
        {onEnterRange && (
          <button type="button" onClick={onEnterRange} title="Range Command Center · dual-cam">
            RANGE
          </button>
        )}
        {onToggleTiles && tilesStatus?.available && (
          <button
            type="button"
            className={tilesStatus?.wanted ? 'on' : ''}
            onClick={onToggleTiles}
            title="Photoreal tiles are a second mesh stack — off on first paint"
          >
            {tilesOn ? 'PHOTOREAL' : tilesStatus?.pending ? 'TILES…' : 'TILES'}
          </button>
        )}
        <button type="button" className={copied ? 'on' : ''} onClick={onShare} title="Copy shareable #gods= link">
          {copied ? 'COPIED' : 'SHARE'}
        </button>
      </div>

      {hudOn && (
        <aside className="ge-layers">
          <div className="ge-kicker">Data layers · click to go</div>
          {LAYER_META.map((row) => {
            const m = layerMeta[row.id] || {}
            const on = layersOn.includes(row.id)
            const focused = focusedLayer === row.id
            const live = m.status === 'live' || m.status === 'empty' || m.status === 'simulated'
            return (
              <button
                key={row.id}
                type="button"
                className={`ge-layer ${on ? 'on' : ''} ${focused ? 'focus' : ''} ${row.id}`}
                title={row.blurb ? `${row.blurb}. Click to dive.` : `Jump the camera to ${row.name}`}
                onClick={() => toggleLayer(row.id)}
              >
                <span className="nm">{row.name}</span>
                <span className="cnt">{m.count != null ? m.count : '—'}</span>
                <span className={`tag ${live ? 'live' : 'off'}`}>{m.label || '…'}</span>
                <span className="go">{focused ? 'NOW' : 'GO'}</span>
                {focused && row.blurb ? <span className="src">{row.blurb}</span> : null}
              </button>
            )
          })}
          {layersOn.includes('cctv') && (
            <div className="ge-cities">
              {['Austin', 'London', 'California'].map((city) => (
                <button
                  key={city}
                  type="button"
                  className={cctvCity === city ? 'on' : ''}
                  onClick={() => onCctvCity?.(city)}
                >
                  {city}
                </button>
              ))}
              <button type="button" className={viewshedOn ? 'on' : ''} onClick={() => setViewshedOn?.((v) => !v)}>
                VIEWSHED
              </button>
            </div>
          )}
        </aside>
      )}

      {consoleOpen && (
        <aside className="ge-console">
          <p className="ge-note">
            {sm.hint}. Keys 1–7 sensors · H HUD · D detection · C cockpit · Esc out.
            {layerMeta.cctv?.count != null ? ` · ${layerMeta.cctv.count} public cameras.` : ''}
          </p>
          {LAYER_META.map((row) => {
            const m = layerMeta[row.id] || {}
            return (
              <p key={row.id} className="ge-note">
                {row.name}: {m.source || '—'} {m.freshness_ms ? `· ${ageLabel(m.freshness_ms)}` : ''}
              </p>
            )
          })}
        </aside>
      )}

      {(hudOn || tracked || (sceneStillKind(focusedLayer) && look?.lat != null)) && (
      <aside className="ge-right-rail">
      {hudOn && (
        <div className="ge-roster-dock">
          <div className="ge-kicker">
            Contacts · {rosterKm >= 10_000 ? 'global' : `${Math.round(rosterKm || 250)} km`}
          </div>
          <div className="ge-roster">
            {(nearby || []).slice(0, 12).map((c) => (
              <button
                key={c.id}
                type="button"
                className={`ge-roster-row ${tracked?.id === c.id ? 'lock' : ''}`}
                onPointerDown={(e) => {
                  e.stopPropagation()
                  onSelect(c)
                }}
                onClick={(e) => {
                  e.stopPropagation()
                  onSelect(c)
                }}
              >
                <span className={`dot ${c.military ? 'mil' : c.kind}`} />
                <span className="nm">{c.callsign || c.id}</span>
                <span className="dst">{c._km != null ? `${Math.round(c._km)} km` : ''}</span>
              </button>
            ))}
            {!nearby?.length && (
              <p className="ge-note">{focusedLayer ? 'Acquiring contacts for this picture.' : 'No contacts in radius.'}</p>
            )}
          </div>
        </div>
      )}

      {!tracked && sceneStillKind(focusedLayer) && look?.lat != null && (
        <div className="ge-card has-sat ge-card-only">
          <div className="ge-card-hd">
            <strong>
              {focusedLayer === 'sats'
                ? 'Satellites · Earth still'
                : focusedLayer === 'missions'
                  ? 'Space missions · pad still'
                  : `${focusedLayer} · satellite still`}
            </strong>
          </div>
          <SatStill
            key={`scene-${focusedLayer}-${look.lat.toFixed(3)}-${look.lon.toFixed(3)}`}
            lat={look.lat}
            lon={look.lon}
            mapsKey={mapsKey}
            kind={sceneStillKind(focusedLayer)}
            label={focusedLayer}
          />
        </div>
      )}

      {tracked && (
        <div className={`ge-card ge-card--tracked ${isGroundIntel(tracked) ? 'has-sat' : ''}`}>
          <div className="ge-card-hd">
            <strong>{tracked.callsign || tracked.id}</strong>
            <button type="button" className="ge-text" onClick={onClearTrack}>
              Unlock
            </button>
          </div>
          <div className="ge-kv">
            <span>Kind</span>
            <span>
              {tracked.military ? 'military' : tracked.kind}
              {tracked.simulated ? ' · SIM' : ''}
              {tracked.reconstructed ? ' · RECONSTRUCTED' : ''}
            </span>
            <span>Pos</span>
            <span>{formatLatLon(tracked.lat, tracked.lon)}</span>
            <span>Alt</span>
            <span>{formatAlt(tracked.alt_m)}</span>
            {tracked.gs_kts != null && (
              <>
                <span>GS</span>
                <span>{Math.round(tracked.gs_kts)} kt</span>
              </>
            )}
            {tracked.heading != null && (
              <>
                <span>HDG</span>
                <span>{Math.round(tracked.heading)}°</span>
              </>
            )}
            {tracked.mag != null && (
              <>
                <span>Mag</span>
                <span>{Number(tracked.mag).toFixed(1)}</span>
              </>
            )}
            {tracked.frp != null && (
              <>
                <span>FRP</span>
                <span>{Number(tracked.frp).toFixed(0)}</span>
              </>
            )}
            {tracked.category && (
              <>
                <span>Cat</span>
                <span>{tracked.category}</span>
              </>
            )}
            {tracked.severity && (
              <>
                <span>Sev</span>
                <span>{tracked.severity}</span>
              </>
            )}
            {tracked.area && (
              <>
                <span>Area</span>
                <span>{tracked.area}</span>
              </>
            )}
            {tracked.rocket && (
              <>
                <span>Lv</span>
                <span>{tracked.rocket}</span>
              </>
            )}
            {tracked.net && (
              <>
                <span>NET</span>
                <span>{String(tracked.net).replace('T', ' ').slice(0, 16)}</span>
              </>
            )}
            {tracked.city && (
              <>
                <span>City</span>
                <span>{tracked.city}</span>
              </>
            )}
            <span>Src</span>
            <span>{tracked.source || '—'}</span>
          </div>
          {isGroundIntel(tracked) || tracked.kind === 'sat' ? (
            <SatStill
              key={tracked.id}
              lat={tracked.lat}
              lon={tracked.lon}
              mapsKey={mapsKey}
              kind={tracked.kind}
              label={tracked.callsign || tracked.kind}
            />
          ) : null}
          {tracked.kind === 'flight' || tracked.kind === 'mil' || tracked.military ? (
            <button
              type="button"
              className={`ge-chip ${camMode === 'cockpit' ? 'on' : ''}`}
              onClick={() => setCamMode(camMode === 'cockpit' ? 'free' : 'cockpit')}
            >
              {camMode === 'cockpit' ? 'Exit cockpit' : 'Cockpit'}
            </button>
          ) : null}
          {tracked.kind === 'sat' || tracked.kind === 'mission' || tracked.kind === 'ship' ? (
            <button
              type="button"
              className={`ge-chip ${camMode === 'orbit' ? 'on' : ''}`}
              onClick={() => setCamMode(camMode === 'orbit' ? 'free' : 'orbit')}
            >
              {camMode === 'orbit' ? 'Free look' : 'Orbit track'}
            </button>
          ) : null}
          {tracked.kind === 'mission' && (
            <button type="button" className="ge-chip on" onClick={() => onReplay(tracked)}>
              {replay?.id === tracked.id ? 'Riding ascent' : 'Replay ascent'}
            </button>
          )}
          {tracked.kind === 'radio' && tracked.url && (
            <button type="button" className="ge-chip on" onClick={() => onTune(tracked)}>
              {radioUrl === tracked.url ? 'Playing' : 'Tune in'}
            </button>
          )}
        </div>
      )}
      </aside>
      )}

      {replay && (
        <div className="ge-replay">
          <span>RECONSTRUCTED ESTIMATE</span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={replay.t}
            onChange={(e) => onReplayRate?.({ ...replay, t: Number(e.target.value), playing: false })}
          />
          <button type="button" onClick={() => onReplayRate?.({ ...replay, rate: replay.rate === 1 ? 4 : replay.rate === 4 ? 0.25 : 1 })}>
            {replay.rate}×
          </button>
        </div>
      )}

      {layersOn.includes('radio') && stations.length > 0 && (
        <div className="ge-tuner">
          <div className="ge-tuner-scale" aria-hidden>
            {stations.slice(0, 24).map((st, i) => (
              <i key={st.id} style={{ left: `${(i / Math.max(1, stations.length - 1)) * 100}%` }} title={st.callsign} />
            ))}
          </div>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={needle}
            onChange={(e) => onNeedle(Number(e.target.value))}
            aria-label="Analog radio tuner"
          />
          <span>{stations[Math.min(stations.length - 1, Math.round(needle * (stations.length - 1)))]?.callsign || '—'}</span>
        </div>
      )}

      {measure && (
        <div className="ge-measure">
          {measure.a} → {measure.b} · {measure.km.toFixed(1)} km
        </div>
      )}

      {cctvStill && (
        <div className="ge-cctv">
          <div className="ge-card-hd">
            <strong>{cctvStill.callsign}</strong>
            <button type="button" className="ge-text" onClick={onCloseCctv}>
              Close
            </button>
          </div>
          {cctvStill.still || cctvStill.video ? (
            <CctvFrame cam={cctvStill} />
          ) : (
            <p className="ge-note">No still from this city feed.</p>
          )}
        </div>
      )}

      <footer className="ge-speakbar">
        {listening && transcript ? <em>{transcript}</em> : speakLine || flyHint || 'Drag the planet · scroll to dive · click to lock · 1–7 sensors'}
      </footer>
    </>
  )
}
