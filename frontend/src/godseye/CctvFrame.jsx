import React, { useEffect, useState } from 'react'
import { cctvStillSrc, isPlayableVideo } from './layers.js'

export default function CctvFrame({ cam, className = '' }) {
  const still = cctvStillSrc(cam?.still)
  const video = isPlayableVideo(cam?.video) ? cam.video : ''
  const [mode, setMode] = useState(video ? 'video' : still ? 'still' : 'miss')
  useEffect(() => {
    setMode(video ? 'video' : still ? 'still' : 'miss')
  }, [cam?.id, still, video])
  if (mode === 'video') {
    return (
      <video
        className={className}
        src={video}
        poster={still || undefined}
        autoPlay
        muted
        playsInline
        loop
        onError={() => setMode(still ? 'still' : 'miss')}
      />
    )
  }
  if (mode === 'still') {
    return <img className={className} src={still} alt={cam?.callsign || ''} onError={() => setMode('miss')} />
  }
  return <span className="miss">Camera offline</span>
}
