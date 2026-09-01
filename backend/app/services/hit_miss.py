"""Hit / miss classification via real vision-language model (OpenRouter Gemini).

Sends frames extracted around each detected shot. No heuristics-as-truth, no mocks.
If the API key is missing, raises — the pipeline will surface a real error.
"""
from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any, Optional  # noqa: F401 used by type hints

from openai import OpenAI

from app.core.config import get_settings


SYSTEM_PROMPT = """You are IronSight shot adjudication AI for live-fire range analysis.
You receive still frames around a single audio-detected bang. Frames may include:
- SHOOTER / FPV (first person) — use for muzzle flash, recoil, sight picture
- OBSERVER / 3P (third person or side cam) — use for plate movement, spark, dust on steel

Prefer OBSERVER evidence for HIT vs MISS when both views exist.
Prefer SHOOTER evidence for NOT_A_SHOT (no muzzle/recoil = likely false audio).

Rules (choose the strongest justified class — avoid UNKNOWN when possible):
- HIT: any credible impact sign (spark, plate swing/rock, dust kick on target face, splash on steel, hole/change on paper, clear hit mark appearing).
- MISS: clear live-fire (muzzle/recoil) but impact on berm/ground/wall OR no target reaction when target face is clearly visible and steady.
- NOT_A_SHOT: no live-fire — speech, clap, door, rack, talking head, UI slides, no muzzle/recoil.
- UNKNOWN: only when frames are useless (extreme blur, totally wrong moment, fully occluded, night with zero signal).
- If targets are visible and fire is clear but impact ambiguous → MISS with lower confidence rather than UNKNOWN when berm/ground is the likely destinaton; HIT only with impact evidence.
- Identify target label if visible (B3, RED PLATE, WHITE PLATE, STEEL, etc.).
- Return STRICT JSON only, no markdown.
"""


def _client() -> OpenAI:
    s = get_settings()
    if s.openrouter_api_key:
        return OpenAI(
            api_key=s.openrouter_api_key,
            base_url=s.openrouter_base_url,
            default_headers={
                "HTTP-Referer": "https://ironsight.local",
                "X-Title": "IronSight",
            },
        )
    if s.xai_api_key:
        return OpenAI(api_key=s.xai_api_key, base_url=s.xai_base_url)
    raise RuntimeError(
        "No vision API key configured. Set OPENROUTER_API_KEY (preferred) or XAI_API_KEY."
    )


def _model() -> str:
    s = get_settings()
    if s.openrouter_api_key:
        return s.vision_model
    return "grok-2-vision-1212"


def _b64_image(path: Path) -> str:
    data = path.read_bytes()
    return base64.b64encode(data).decode("ascii")


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    # strip fences if model adds them
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            return json.loads(m.group(0))
        raise


def _normalize_cls(cls: str) -> str:
    cls = str(cls or "UNKNOWN").upper().replace("-", "_").replace(" ", "_")
    if cls in {"NOTASHOT", "FALSE_POSITIVE", "FALSEPOSITIVE", "NO_SHOT", "NOSHOT"}:
        return "NOT_A_SHOT"
    if cls not in {"HIT", "MISS", "NOT_A_SHOT", "UNKNOWN"}:
        return "UNKNOWN"
    return cls


def _parse_bbox(bbox: Any) -> Optional[list[float]]:
    if not bbox or len(bbox) < 4:
        return None
    try:
        return [float(max(0.0, min(1.0, float(x)))) for x in bbox[:4]]
    except (TypeError, ValueError):
        return None


def _parse_targets(data: dict[str, Any]) -> list[dict[str, Any]]:
    targets_out = []
    for t in data.get("targets") or []:
        if not isinstance(t, dict):
            continue
        tb = _parse_bbox(t.get("bbox") or t.get("box"))
        if not tb:
            continue
        targets_out.append(
            {
                "id": t.get("id") or t.get("target_id"),
                "label": t.get("label") or t.get("target_label"),
                "color": t.get("color") or t.get("target_color") or "steel",
                "bbox": tb,
                "confidence": float(t.get("confidence") or data.get("confidence") or 0.7),
            }
        )
    return targets_out


def _vlm_classify(
    paths: list[Path],
    *,
    shot_index: int,
    timestamp_s: float,
    context: Optional[str],
    pass_hint: str = "",
) -> dict[str, Any]:
    client = _client()
    n_obs = sum(1 for p in paths if "obs" in p.name.lower())
    n_sho = len(paths) - n_obs
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"Shot #{shot_index} at t={timestamp_s:.3f}s.\n"
                f"Context: {context or 'outdoor / indoor range live fire'}\n"
                f"Frame mix: ~{n_sho} shooter/FPV, ~{n_obs} observer/3P.\n"
                f"{pass_hint}\n"
                "Respond with JSON: "
                '{"classification":"HIT|MISS|NOT_A_SHOT|UNKNOWN","confidence":0-1,'
                '"target_id":"B# or null","target_label":"string or null",'
                '"target_color":"red|blue|white|steel|null",'
                '"bbox":[x,y,w,h],'
                '"targets":[{"id":"B3","label":"RED PLATE","color":"red","bbox":[x,y,w,h],"confidence":0-1}],'
                '"reasoning":"one sentence"}\n'
                "bbox is normalized [x,y,width,height] in 0..1 (engaged target). "
                "List ALL visible range targets in targets[]. "
                "Bias: use NOT_A_SHOT for false audio; prefer HIT/MISS over UNKNOWN when any impact or berm evidence exists."
            ),
        }
    ]
    # Prefer observer frames first in the prompt order (model attends early images more)
    ordered = sorted(paths, key=lambda p: (0 if "obs" in p.name.lower() else 1, p.name))
    for p in ordered[:8]:
        mime = "image/jpeg" if p.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{_b64_image(p)}"},
            }
        )

    resp = client.chat.completions.create(
        model=_model(),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        temperature=0.05,
        max_tokens=450,
    )
    raw = resp.choices[0].message.content or "{}"
    data = _parse_json(raw)
    cls = _normalize_cls(data.get("classification", "UNKNOWN"))
    conf = float(data.get("confidence") or 0.0)
    return {
        "classification": cls,
        "confidence": conf,
        "target_id": data.get("target_id"),
        "target_label": data.get("target_label"),
        "target_color": data.get("target_color"),
        "bbox": _parse_bbox(data.get("bbox")),
        "targets": _parse_targets(data),
        "reasoning": data.get("reasoning") or raw[:300],
        "model": _model(),
        "raw": data,
    }


def _yolo_assist(frame_paths: list[Path]) -> dict[str, Any]:
    """Cheap plate detector signal for review cues (does not invent HIT/MISS)."""
    try:
        from app.services import yolo_plates
    except Exception:
        return {"plates": 0}
    if not yolo_plates.weights_available():
        return {"plates": 0}
    plates = 0
    best = None
    for p in frame_paths[:6]:
        try:
            dets = yolo_plates.detect_plates_yolo_path(p)
        except Exception:
            continue
        if not isinstance(dets, list):
            continue
        for d in dets:
            plates += 1
            conf = float(d.get("confidence") or d.get("conf") or 0)
            if best is None or conf > best.get("confidence", 0):
                best = {
                    "confidence": conf,
                    "bbox": d.get("bbox"),
                    "label": d.get("label") or "PLATE",
                }
    return {"plates": plates, "best": best}


def classify_shot(
    frame_paths: list[str | Path],
    *,
    shot_index: int,
    timestamp_s: float,
    context: Optional[str] = None,
    retry_on_unknown: bool = True,
) -> dict[str, Any]:
    """Call real VLM on the shot frames and return structured adjudication.

    Reduces UNKNOWN via: observer-first framing, stricter prompts, second pass,
    and optional YOLO plate assist for needs_review tagging.
    """
    paths = [Path(p) for p in frame_paths if Path(p).exists()]
    if not paths:
        raise FileNotFoundError("No frame images available for hit/miss classification")

    result = _vlm_classify(
        paths,
        shot_index=shot_index,
        timestamp_s=timestamp_s,
        context=context,
        pass_hint="Pass 1: primary adjudication.",
    )

    # Second pass for UNKNOWN or very low-confidence adjudicated shots
    if retry_on_unknown and (
        result["classification"] == "UNKNOWN"
        or (
            result["classification"] in ("HIT", "MISS")
            and float(result.get("confidence") or 0) < 0.4
        )
    ):
        # Prefer observer-only if available; else all frames with stronger instruction
        obs = [p for p in paths if "obs" in p.name.lower()]
        retry_paths = obs if len(obs) >= 1 else paths
        try:
            retry = _vlm_classify(
                retry_paths,
                shot_index=shot_index,
                timestamp_s=timestamp_s,
                context=context,
                pass_hint=(
                    "Pass 2: previous pass was inconclusive. "
                    "Look harder for plate motion/spark (HIT) or berm impact (MISS). "
                    "Only return UNKNOWN if frames truly show nothing useful."
                ),
            )
            # Prefer decisive non-UNKNOWN; or higher confidence same class
            if retry["classification"] != "UNKNOWN":
                if result["classification"] == "UNKNOWN" or float(retry.get("confidence") or 0) >= float(
                    result.get("confidence") or 0
                ):
                    retry["reasoning"] = (
                        f"{retry.get('reasoning') or ''} [retry from: {result.get('classification')}]"
                    ).strip()
                    result = retry
        except Exception:
            pass

    yolo = _yolo_assist(paths)
    result["yolo_plates"] = yolo.get("plates") or 0
    if yolo.get("best") and not result.get("bbox"):
        bb = yolo["best"].get("bbox")
        if bb and len(bb) >= 4:
            # YOLO may return xyxy or xywhn — only use if looks normalized
            try:
                vals = [float(x) for x in bb[:4]]
                if max(vals) <= 1.5:
                    result["bbox"] = [
                        max(0.0, min(1.0, vals[0])),
                        max(0.0, min(1.0, vals[1])),
                        max(0.0, min(1.0, vals[2] if vals[2] < 1 else 0.1)),
                        max(0.0, min(1.0, vals[3] if vals[3] < 1 else 0.1)),
                    ]
                    result["bbox_source"] = "yolo"
            except (TypeError, ValueError):
                pass

    conf = float(result.get("confidence") or 0.0)
    cls = result["classification"]
    needs_review = cls == "UNKNOWN" or (
        cls in ("HIT", "MISS") and conf < 0.45
    )
    # If still UNKNOWN but YOLO sees plates + we have many frames, flag for coach review
    # (do not invent HIT — honesty over vanity)
    if cls == "UNKNOWN" and (yolo.get("plates") or 0) >= 1:
        needs_review = True
        result["reasoning"] = (
            (result.get("reasoning") or "")
            + " Plates visible (YOLO) but impact not confirmed — coach review recommended."
        ).strip()

    result["needs_review"] = needs_review
    result["confidence"] = conf
    return result


def detect_targets_in_frame(
    image_path: str | Path,
    *,
    context: Optional[str] = None,
) -> dict[str, Any]:
    """Detect all visible range targets with pixel-accurate normalized bboxes."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(path)

    client = _client()
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "You are IronSight multi-view target tracker.\n"
                f"Context: {context or 'live-fire range'}\n"
                "Detect EVERY visible range target (steel plates, silhouettes, poppers, cardboard).\n"
                "Return STRICT JSON only:\n"
                '{"targets":[{"id":"B3","label":"RED PLATE","color":"red|blue|white|steel|green",'
                '"bbox":[x,y,w,h],"confidence":0-1}]}\n'
                "bbox = normalized [x,y,width,height] with all values in 0..1 "
                "(x,y = top-left of box relative to image width/height).\n"
                "If no targets visible, return {\"targets\":[]}."
            ),
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{'jpeg' if path.suffix.lower() in {'.jpg', '.jpeg'} else 'png'};base64,{_b64_image(path)}"
            },
        },
    ]
    resp = client.chat.completions.create(
        model=_model(),
        messages=[
            {
                "role": "system",
                "content": (
                    "You locate shooting-range targets with tight bounding boxes. "
                    "JSON only. Normalized xywh bboxes must be accurate."
                ),
            },
            {"role": "user", "content": content},
        ],
        temperature=0.1,
        max_tokens=800,
    )
    raw = resp.choices[0].message.content or "{}"
    data = _parse_json(raw)
    targets = []
    for t in data.get("targets") or []:
        if not isinstance(t, dict):
            continue
        bbox = t.get("bbox") or t.get("box")
        if not bbox or len(bbox) < 4:
            continue
        try:
            bbox = [float(x) for x in bbox[:4]]
        except (TypeError, ValueError):
            continue
        targets.append(
            {
                "id": t.get("id") or t.get("target_id"),
                "label": t.get("label") or t.get("target_label"),
                "color": t.get("color") or t.get("target_color") or "steel",
                "bbox": bbox,
                "confidence": float(t.get("confidence") or 0.7),
                "source": "vlm",
            }
        )
    return {"targets": targets, "model": _model(), "raw": data}


def coach_session(
    shots_summary: list[dict[str, Any]],
    stats: dict[str, Any],
    focus: Optional[str] = None,
) -> str:
    """Post-run coaching from real LLM using actual shot log — not canned text."""
    client = _client()
    s = get_settings()
    model = s.coaching_model if s.openrouter_api_key else "grok-4.5"
    if not s.openrouter_api_key and s.xai_api_key:
        model = "grok-4.5"

    prompt = {
        "stats": stats,
        "shots": shots_summary[:80],
        "focus": focus or "overall performance, splits, and target transitions",
    }
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an elite USPSA/IDPA range coach embedded in IronSight. "
                    "Give a concise tactical debrief (120-200 words) from the real shot log. "
                    "No fluff, no invented stats. Reference actual shot numbers and results."
                ),
            },
            {"role": "user", "content": json.dumps(prompt)},
        ],
        temperature=0.4,
        max_tokens=500,
    )
    return (resp.choices[0].message.content or "").strip()
