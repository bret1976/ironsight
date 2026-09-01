import React, { useEffect, useMemo, useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'
import { EARTH_R } from './geo.js'

const DAY_URLS = [
  'https://unpkg.com/three-globe@2.44.1/example/img/earth-blue-marble.jpg',
  'https://cdn.jsdelivr.net/gh/mrdoob/three.js@r170/examples/textures/planets/earth_atmos_2048.jpg',
]

const earthVert = /* glsl */ `
  varying vec2 vUv;
  varying vec3 vNormalW;
  varying vec3 vPosW;
  void main() {
    vUv = uv;
    vec4 w = modelMatrix * vec4(position, 1.0);
    vPosW = w.xyz;
    vNormalW = normalize(mat3(modelMatrix) * normal);
    gl_Position = projectionMatrix * viewMatrix * w;
  }
`

const earthFrag = /* glsl */ `
  uniform sampler2D dayMap;
  uniform float hasMap;
  uniform int sensor;
  uniform vec3 sunDir;
  varying vec2 vUv;
  varying vec3 vNormalW;
  varying vec3 vPosW;

  vec3 ironbow(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c;
    if (t < 0.25) c = mix(vec3(0.02, 0.0, 0.08), vec3(0.15, 0.0, 0.55), t / 0.25);
    else if (t < 0.5) c = mix(vec3(0.15, 0.0, 0.55), vec3(0.85, 0.05, 0.15), (t - 0.25) / 0.25);
    else if (t < 0.75) c = mix(vec3(0.85, 0.05, 0.15), vec3(0.98, 0.75, 0.08), (t - 0.5) / 0.25);
    else c = mix(vec3(0.98, 0.75, 0.08), vec3(1.0, 0.98, 0.92), (t - 0.75) / 0.25);
    return c;
  }

  vec3 procedural(vec2 uv) {
    float lat = (0.5 - uv.y) * 3.14159265;
    float lon = (uv.x - 0.5) * 6.2831853;
    float ice = smoothstep(0.72, 0.92, abs(sin(lat)));
    float bands = 0.55 + 0.45 * sin(lat * 6.0 + sin(lon * 3.0));
      vec3 ocean = vec3(0.04, 0.18, 0.38);
      vec3 land = vec3(0.12, 0.22, 0.10);
    vec3 desert = vec3(0.35, 0.28, 0.12);
    float landMask = smoothstep(0.42, 0.62, bands * (0.7 + 0.3 * sin(lon * 7.0 + lat * 4.0)));
    vec3 base = mix(ocean, mix(land, desert, smoothstep(0.1, 0.45, abs(lat))), landMask);
    return mix(base, vec3(0.86, 0.9, 0.95), ice);
  }

  void main() {
    vec3 day = hasMap > 0.5 ? texture2D(dayMap, vUv).rgb : procedural(vUv);
    float oceanMask = 1.0 - smoothstep(0.28, 0.42, day.g);
    day = mix(day, max(day, vec3(0.05, 0.20, 0.40)), oceanMask * 0.72);
    vec3 n = normalize(vNormalW);
    float ndl = max(dot(n, normalize(sunDir)), 0.0);
    float wrap = ndl * 0.5 + 0.5;
    vec3 viewDir = normalize(cameraPosition - vPosW);
    vec3 halfV = normalize(normalize(sunDir) + viewDir);
    float spec = pow(max(dot(n, halfV), 0.0), 28.0) * 0.28;
    float ocean = 1.0 - smoothstep(0.28, 0.42, day.g);
    vec3 color = day * wrap + vec3(0.45, 0.55, 0.7) * spec * ocean;

    float lum = dot(color, vec3(0.25, 0.65, 0.10));
    if (sensor == 1) {
      color = vec3(0.02, 0.08, 0.03) + vec3(0.05, 1.15, 0.18) * lum;
    } else if (sensor == 2) {
      color = ironbow(pow(lum, 0.85));
    } else if (sensor == 3) {
      float g = lum;
      color = vec3(g * 0.35, g * 1.05, g * 0.28);
    } else if (sensor == 4) {
      color = ironbow(pow(lum, 0.52));
      color.r = min(1.0, color.r * 1.28 + 0.08);
      color.g = color.g * 0.72;
    } else if (sensor == 5) {
      float g = pow(lum, 1.15);
      color = vec3(g, g, g) * vec3(1.02, 1.0, 0.96);
    } else if (sensor == 6) {
      float g = pow(1.0 - lum, 0.85);
      color = vec3(0.92, 0.95, 1.0) * (0.25 + 0.75 * g);
    }
    gl_FragColor = vec4(color, 1.0);
  }
`

const atmosVert = /* glsl */ `
  varying vec3 vNormalW;
  varying vec3 vPosW;
  void main() {
    vec4 w = modelMatrix * vec4(position, 1.0);
    vPosW = w.xyz;
    vNormalW = normalize(mat3(modelMatrix) * normal);
    gl_Position = projectionMatrix * viewMatrix * w;
  }
`

const atmosFrag = /* glsl */ `
  uniform int sensor;
  varying vec3 vNormalW;
  varying vec3 vPosW;
  void main() {
    vec3 n = normalize(vNormalW);
    vec3 v = normalize(cameraPosition - vPosW);
    float f = pow(1.0 - abs(dot(n, v)), 2.4);
    vec3 col = vec3(0.25, 0.55, 1.0);
    if (sensor == 1) col = vec3(0.05, 0.9, 0.2);
    if (sensor == 2) col = vec3(0.95, 0.25, 0.05);
    if (sensor == 3) col = vec3(0.2, 0.95, 0.25);
    if (sensor == 4) col = vec3(1.0, 0.35, 0.02);
    if (sensor == 5) col = vec3(0.75, 0.75, 0.72);
    if (sensor == 6) col = vec3(0.85, 0.92, 1.0);
    gl_FragColor = vec4(col * f * 1.4, f * 0.85);
  }
`

function useDayTexture() {
  const tex = useMemo(() => {
    const loader = new THREE.TextureLoader()
    loader.setCrossOrigin('anonymous')
    const t = new THREE.Texture()
    t.colorSpace = THREE.SRGBColorSpace
    let i = 0
    const tryNext = () => {
      if (i >= DAY_URLS.length) return
      const url = DAY_URLS[i++]
      loader.load(
        url,
        (loaded) => {
          loaded.colorSpace = THREE.SRGBColorSpace
          loaded.anisotropy = 8
          loaded.needsUpdate = true
          t.image = loaded.image
          t.colorSpace = THREE.SRGBColorSpace
          t.needsUpdate = true
          t.userData.ready = true
        },
        undefined,
        tryNext
      )
    }
    tryNext()
    return t
  }, [])
  useEffect(() => () => tex.dispose(), [tex])
  return tex
}

function Stars() {
  const geo = useMemo(() => {
    const n = 2800
    const pos = new Float32Array(n * 3)
    for (let i = 0; i < n; i++) {
      const r = EARTH_R * (12 + Math.random() * 18)
      const a = Math.random() * Math.PI * 2
      const b = Math.acos(2 * Math.random() - 1)
      pos[i * 3] = r * Math.sin(b) * Math.cos(a)
      pos[i * 3 + 1] = r * Math.cos(b)
      pos[i * 3 + 2] = r * Math.sin(b) * Math.sin(a)
    }
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3))
    return g
  }, [])
  useEffect(() => () => geo.dispose(), [geo])
  return (
    <points geometry={geo} raycast={() => null}>
      <pointsMaterial color="#c8d2e8" size={18000} sizeAttenuation transparent opacity={0.85} depthWrite={false} />
    </points>
  )
}

export function sensorIdOf(mode) {
  if (mode === 'nvg') return 1
  if (mode === 'flir') return 2
  if (mode === 'crt') return 3
  if (mode === 'ironbell') return 4
  if (mode === 'noir') return 5
  if (mode === 'snow') return 6
  return 0
}

export default function GlobeEarth({ sensor = 'rgb', showSurface = true, scale = 1 }) {
  const day = useDayTexture()
  const sensorId = sensorIdOf(sensor)
  const uniforms = useMemo(
    () => ({
      dayMap: { value: day },
      hasMap: { value: 0 },
      sensor: { value: 0 },
      sunDir: { value: new THREE.Vector3(1, 0.35, 0.2).normalize() },
    }),
    [day]
  )
  const atmosU = useMemo(() => ({ sensor: { value: 0 } }), [])

  useFrame(({ camera }) => {
    uniforms.hasMap.value = day.userData.ready ? 1 : 0
    uniforms.sensor.value = sensorId
    atmosU.sensor.value = sensorId
    // Light the face in frame so a night-side restage is not a black disc.
    uniforms.sunDir.value.copy(camera.position).normalize()
  })

  return (
    <group>
      <Stars />
      {showSurface && (
        <mesh scale={scale} raycast={() => null}>
          <sphereGeometry args={[EARTH_R, 96, 96]} />
          <shaderMaterial vertexShader={earthVert} fragmentShader={earthFrag} uniforms={uniforms} />
        </mesh>
      )}
      <mesh scale={1.028} raycast={() => null}>
        <sphereGeometry args={[EARTH_R, 64, 64]} />
        <shaderMaterial
          vertexShader={atmosVert}
          fragmentShader={atmosFrag}
          uniforms={atmosU}
          transparent
          depthWrite={false}
          side={THREE.BackSide}
          blending={THREE.AdditiveBlending}
        />
      </mesh>
    </group>
  )
}

export function SensorScreen({ mode }) {
  const { size } = useThree()
  const mat = useRef()
  const uniforms = useMemo(
    () => ({
      uSensor: { value: 0 },
      uTime: { value: 0 },
      uRes: { value: new THREE.Vector2(1, 1) },
    }),
    []
  )
  useFrame((state) => {
    const id = sensorIdOf(mode)
    uniforms.uSensor.value = id
    uniforms.uTime.value = state.clock.elapsedTime
    uniforms.uRes.value.set(size.width, size.height)
    if (mat.current) mat.current.visible = id !== 0
  })
  return (
    <mesh renderOrder={40} frustumCulled={false} position={[0, 0, -0.6]} raycast={() => null}>
      <planeGeometry args={[2, 2]} />
      <shaderMaterial
        ref={mat}
        transparent
        depthTest={false}
        depthWrite={false}
        blending={THREE.NormalBlending}
        uniforms={uniforms}
        vertexShader={`varying vec2 vUv; void main(){ vUv=uv; gl_Position=vec4(position.xy, 0.0, 1.0); }`}
        fragmentShader={`
          uniform int uSensor;
          uniform float uTime;
          uniform vec2 uRes;
          varying vec2 vUv;
          void main() {
            if (uSensor == 0) discard;
            vec2 uv = vUv;
            float vig = smoothstep(0.95, 0.35, length(uv - 0.5));
            float scan = 0.0;
            if (uSensor == 3) {
              scan = 0.18 * step(0.5, fract(uv.y * uRes.y * 0.45 + uTime * 6.0));
              float ca = 0.004;
              gl_FragColor = vec4(0.08 + ca, 0.22, 0.05, 0.22 + scan * 0.35) * (0.55 + 0.45 * vig);
              return;
            }
            if (uSensor == 1) {
              float grain = fract(sin(dot(uv * uRes + uTime, vec2(12.9898, 78.233))) * 43758.5453);
              gl_FragColor = vec4(0.0, 0.18, 0.04, 0.16 + grain * 0.07) * vig;
              return;
            }
            if (uSensor == 4) {
              float scan = 0.08 * step(0.55, fract(uv.y * uRes.y * 0.22 + uTime * 2.0));
              gl_FragColor = vec4(0.22, 0.04, 0.0, 0.14 + scan) * vig;
              return;
            }
            if (uSensor == 5) {
              float grain = fract(sin(dot(uv * uRes + uTime, vec2(12.9898, 78.233))) * 43758.5453);
              gl_FragColor = vec4(0.02, 0.02, 0.02, 0.12 + grain * 0.06) * vig;
              return;
            }
            if (uSensor == 6) {
              gl_FragColor = vec4(0.72, 0.82, 0.95, 0.10) * vig;
              return;
            }
            gl_FragColor = vec4(0.12, 0.02, 0.0, 0.10) * vig;
          }
        `}
      />
    </mesh>
  )
}
