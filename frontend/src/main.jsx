import React, { lazy, Suspense, useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'
import GodsEyeView from './GodsEyeView'
import './styles.css'

/** Dual-cam / studio only. Bare `#gods=` share links stay on the globe. */
function isRangeHash() {
  return typeof location !== 'undefined' && /#(range|dual|studio)\b/i.test(location.hash)
}

/** Range Command Center + splat stay out of the globe first-paint chunk. */
const App = lazy(() => import('./App.jsx'))

function Root() {
  const [product, setProduct] = useState(() => (isRangeHash() ? 'range' : 'gods'))

  useEffect(() => {
    const sync = () => setProduct(isRangeHash() ? 'range' : 'gods')
    window.addEventListener('hashchange', sync)
    return () => window.removeEventListener('hashchange', sync)
  }, [])

  if (product === 'range') {
    return (
      <Suspense fallback={null}>
        <App
          onEnterGods={() => {
            history.replaceState(null, '', `${location.pathname}${location.search}`)
            setProduct('gods')
          }}
        />
      </Suspense>
    )
  }

  return (
    <GodsEyeView
      onEnterRange={() => {
        location.hash = 'range'
        setProduct('range')
      }}
    />
  )
}

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>
)
