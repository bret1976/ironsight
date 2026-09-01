"""GodsEye voice/text command interpreter.

Rule-based first so the cockpit works without a model. Optional OpenRouter
pass turns natural language into the same action list.
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

from app.core.config import get_settings

SENSORS = {
    "rgb": "rgb",
    "eo": "rgb",
    "daylight": "rgb",
    "electro-optical": "rgb",
    "nvg": "nvg",
    "night vision": "nvg",
    "night-vision": "nvg",
    "flir": "flir",
    "thermal": "flir",
    "ironbow": "flir",
    "iron bell": "ironbell",
    "ironbell": "ironbell",
    "crt": "crt",
    "scanline": "crt",
    "noir": "noir",
    "black and white": "noir",
    "snow": "snow",
    "white hot": "snow",
}

LAYER_ALIASES = {
    "flight": "flights",
    "flights": "flights",
    "planes": "flights",
    "adsb": "flights",
    "military": "military",
    "mil": "military",
    "helicopter": "military",
    "helicopters": "military",
    "ships": "ships",
    "vessels": "ships",
    "ais": "ships",
    "sats": "sats",
    "satellites": "sats",
    "iss": "sats",
    "quakes": "quakes",
    "earthquakes": "quakes",
    "fires": "fires",
    "firms": "fires",
    "storms": "storms",
    "storm": "storms",
    "hazards": "storms",
    "hurricane": "storms",
    "alerts": "storms",
    "eonet": "storms",
    "sites": "sites",
    "installations": "sites",
    "dams": "sites",
    "radio": "radio",
    "missions": "missions",
    "launches": "missions",
    "space missions": "missions",
    "cctv": "cctv",
    "cameras": "cctv",
    "traffic cameras": "cctv",
    "traffic": "traffic",
    "congestion": "traffic",
}

FLY_RE = re.compile(
    r"\b(?:fly|go|take me|center|show|navigate|jump)\s+(?:to|over|into)?\s*(.+)$",
    re.I,
)
ANNOTATE_RE = re.compile(r"\b(?:annotate|outline|mark|highlight)\s+(.+)$", re.I)
MEASURE_RE = re.compile(
    r"(?:how far is\s+(.+?)\s+from\s+(.+))|(?:(?:how far|distance|measure)\b.*?\b(?:from|between)\s+(.+?)\s+(?:to|and)\s+(.+))$",
    re.I,
)


def parse_command(text: str) -> dict[str, Any]:
    """Pure interpreter. Returns {speak, actions}."""
    raw = (text or "").strip()
    t = raw.lower()
    actions: list[dict[str, Any]] = []
    speak = ""

    if not t:
        return {"speak": "Say a place, a layer, or track the ISS.", "actions": [], "source": "rules"}

    if re.search(r"\b(globe|earth|zoom out|reset|full earth)\b", t):
        actions.append({"type": "reset"})
        speak = "Returning to globe view."

    if re.search(r"\b(detection|bounding box|boxes)\b", t):
        on_det = not re.search(r"\b(off|hide|disable)\b", t)
        actions.append({"type": "detection", "on": on_det})
        speak = speak or ("Detection overlay on." if on_det else "Detection overlay off.")

    if re.search(r"\b(hud|tactical layout|heads.?up)\b", t) and re.search(
        r"\b(on|off|hide|show|toggle)\b", t
    ):
        on_hud = not re.search(r"\b(off|hide)\b", t)
        actions.append({"type": "hud", "on": on_hud})
        speak = speak or ("HUD on." if on_hud else "HUD off.")

    m = MEASURE_RE.search(raw)
    if m:
        a = (m.group(1) or m.group(3) or "").strip(" .?!")
        b = (m.group(2) or m.group(4) or "").strip(" .?!")
        if a and b:
            actions.append({"type": "measure", "a": a, "b": b})
            speak = speak or f"Measuring {a} to {b}."

    if re.search(r"\b(cockpit|chase|pilot view|from the track|enter cockpit)\b", t):
        actions.append({"type": "camera", "mode": "cockpit"})
        speak = "Cockpit mode."
    elif re.search(r"\b(orbit|orbiting|circle|spin around)\b", t):
        actions.append({"type": "camera", "mode": "orbit"})
        speak = "Orbiting the look point."
    elif re.search(r"\b(free orbit|unlock camera|map view)\b", t):
        actions.append({"type": "camera", "mode": "free"})
        speak = "Free orbit."

    if re.search(r"\b(iss|zarya|space station)\b", t) and "satellite" not in t[:8]:
        actions.append({"type": "track_iss"})
        speak = speak or "Tracking the ISS."

    if re.search(r"\bnearest (heli|helicopter|helo)\b", t):
        actions.append({"type": "track_nearest", "kind": "mil"})
        speak = speak or "Locking the nearest military contact."
    elif re.search(r"\bnearest (ship|vessel)\b", t):
        actions.append({"type": "track_nearest", "kind": "ship"})
        speak = speak or "Locking the nearest vessel."
    elif re.search(r"\bnearest (flight|plane|aircraft)\b", t):
        actions.append({"type": "track_nearest", "kind": "flight"})
        speak = speak or "Locking the nearest flight."

    for phrase, sensor in SENSORS.items():
        if phrase in t:
            actions.append({"type": "sensor", "sensor": sensor})
            speak = speak or f"{sensor.upper()} look."
            break

    on = not re.search(r"\b(off|disable|hide|remove)\b", t)
    for alias, layer in LAYER_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", t) and re.search(
            r"\b(on|off|enable|disable|show|hide|toggle|layer|contacts)\b", t
        ):
            actions.append({"type": "layer", "layer": layer, "on": on})
            speak = speak or f"{layer} {'on' if on else 'off'}."

    m = ANNOTATE_RE.search(raw)
    if m:
        place = m.group(1).strip(" .?!")
        if place:
            actions.append({"type": "annotate", "q": place})
            speak = speak or f"Marking {place}."

    m = FLY_RE.search(raw)
    if m and not any(a["type"] in ("track_iss", "annotate") for a in actions):
        place = m.group(1).strip(" .?!")
        place = re.sub(r"^(the|a|an)\s+", "", place, flags=re.I)
        if place and place not in ("it", "that", "this", "here"):
            place = re.sub(r"\s+and\s+(start\s+)?orbit.*$", "", place, flags=re.I).strip()
            actions.append({"type": "fly_to", "q": place})
            if re.search(r"\borbit", t):
                actions.append({"type": "camera", "mode": "orbit"})
            speak = speak or f"Flying to {place}."

    if not actions:
        # Bare place name — treat as fly-to.
        if len(raw) >= 3 and not re.search(r"\b(what|why|who|how|hello|hi)\b", t):
            actions.append({"type": "fly_to", "q": raw})
            speak = f"Flying to {raw}."
        else:
            speak = "I can fly you somewhere, track the ISS, switch FLIR or NVG, or open cockpit mode."

    return {"speak": speak, "actions": _dedupe_actions(actions), "source": "rules"}


def _dedupe_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for a in actions:
        key = json.dumps(a, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(a)
    return out


async def interpret_command(text: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Rules first, then optional LLM polish when OpenRouter is configured."""
    parsed = parse_command(text)
    settings = get_settings()
    key = (getattr(settings, "openrouter_api_key", None) or "").strip()
    if not key or not (text or "").strip():
        return {**parsed, "hasModel": bool(key)}

    look = context or {}
    system = (
        "You are the voice of a live spy-satellite globe. "
        "Return ONLY JSON: {\"speak\": string, \"actions\": [ ... ]}. "
        "Allowed action types: fly_to {q}, annotate {q}, measure {a,b}, sensor {sensor: rgb|nvg|flir|ironbell|crt|noir|snow}, "
        "layer {layer: flights|military|ships|sats|quakes|fires|storms|sites|radio|missions|cctv|traffic, on: bool}, "
        "track_iss, track_nearest {kind: flight|mil|ship|sat|mission|cctv}, camera {mode: free|orbit|cockpit}, "
        "detection {on: bool}, hud {on: bool}, reset. "
        "Keep speak to one short spoken sentence. Do not invent coordinates."
    )
    user = json.dumps(
        {
            "utterance": text,
            "look": {
                "lat": look.get("lat"),
                "lon": look.get("lon"),
                "alt": look.get("alt"),
            },
            "seed": parsed,
        }
    )
    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            r = await client.post(
                f"{settings.openrouter_base_url.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": getattr(settings, "coaching_model", None) or "google/gemini-2.5-flash",
                    "temperature": 0.2,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
            )
        if r.status_code >= 400:
            return {**parsed, "hasModel": True, "modelError": f"HTTP {r.status_code}"}
        body = r.json()
        content = (((body.get("choices") or [{}])[0].get("message") or {}).get("content")) or ""
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?", "", content).rstrip("`").strip()
        data = json.loads(content)
        actions = data.get("actions") if isinstance(data, dict) else None
        if not isinstance(actions, list):
            return {**parsed, "hasModel": True}
        clean: list[dict[str, Any]] = []
        for a in actions:
            if isinstance(a, dict) and a.get("type"):
                clean.append(a)
        speak = str(data.get("speak") or parsed["speak"])
        return {
            "speak": speak,
            "actions": clean or parsed["actions"],
            "source": "model",
            "hasModel": True,
        }
    except Exception as e:
        return {**parsed, "hasModel": True, "modelError": str(e)}
