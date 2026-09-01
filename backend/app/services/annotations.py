"""Human review / correction tool data + coach wrap export.

Corrections are stored per-session and used to:
1. Override classification/target on the shot log
2. Feed yolo_plates harvest as high-confidence labels
3. Rebuild honest stats after each fix
"""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.services import store
from app.services.scoring import compute_stats, recompute_session_stats


def annotations_path(session_id: str) -> Path:
    return store.session_dir(session_id) / "annotations.json"


def load_annotations(session_id: str) -> dict[str, Any]:
    p = annotations_path(session_id)
    if not p.exists():
        return {"session_id": session_id, "corrections": []}
    return json.loads(p.read_text())


def save_correction(
    session_id: str,
    *,
    shot_id: int,
    classification: Optional[str] = None,
    target_id: Optional[str] = None,
    target_label: Optional[str] = None,
    target_color: Optional[str] = None,
    bbox: Optional[list[float]] = None,
    note: Optional[str] = None,
) -> dict[str, Any]:
    """Upsert a human correction and apply it to the live session shot log."""
    doc = store.load_session(session_id)
    ann = load_annotations(session_id)
    corrections = ann.setdefault("corrections", [])

    payload = {
        "shot_id": int(shot_id),
        "classification": classification,
        "target_id": target_id,
        "target_label": target_label,
        "target_color": target_color,
        "bbox": bbox,
        "note": note,
    }
    # replace existing for this shot
    corrections = [c for c in corrections if int(c.get("shot_id", -1)) != int(shot_id)]
    corrections.append(payload)
    ann["corrections"] = corrections
    annotations_path(session_id).write_text(json.dumps(ann, indent=2))

    # Apply to session shots
    for sh in doc.get("shots") or []:
        if int(sh.get("id", -1)) != int(shot_id):
            continue
        if classification:
            sh["classification"] = classification
            sh["human_corrected"] = True
            sh["needs_review"] = False
            sh["confidence"] = max(float(sh.get("confidence") or 0), 0.95)
        if target_id is not None:
            sh["target_id"] = target_id
        if target_label is not None:
            sh["target_label"] = target_label
        if target_color is not None:
            sh["target_color"] = target_color
        if bbox is not None:
            sh["bbox"] = bbox
            sh["bbox_source"] = "human"
        if note:
            sh["review_note"] = note
        break

    stats = recompute_session_stats(doc)
    stats["human_corrections"] = len(corrections)
    doc["stats"] = stats
    store.save_session(doc)
    return {"ok": True, "correction": payload, "stats": stats, "annotations": ann}


def build_wrap_card(
    session_id: str,
    *,
    export: bool = True,
    watermark: bool = False,
) -> dict[str, Any]:
    """Session debrief card for coaches. Export = full shareable payload."""
    doc = store.load_session(session_id)
    # Always recompute honest stats for wrap
    stats = recompute_session_stats(doc)
    store.save_session(doc)
    shots = doc.get("shots") or []
    real = [s for s in shots if s.get("classification") in ("HIT", "MISS", "UNKNOWN")]

    targets = doc.get("targets") or []
    best_t = None
    if targets:
        best_t = max(targets, key=lambda t: (t.get("hits") or 0))

    shot_lines = []
    for s in real:
        mark = "✓" if s.get("classification") == "HIT" else (
            "✗" if s.get("classification") == "MISS" else "?"
        )
        rev = " · REVIEW" if s.get("needs_review") or s.get("classification") == "UNKNOWN" else ""
        hum = " · coach" if s.get("human_corrected") else ""
        shot_lines.append(
            {
                "id": s.get("id"),
                "t": s.get("timestamp_s"),
                "result": s.get("classification"),
                "target": s.get("target_label") or s.get("target_id"),
                "confidence": s.get("confidence"),
                "needs_review": bool(s.get("needs_review") or s.get("classification") == "UNKNOWN"),
                "human_corrected": bool(s.get("human_corrected")),
                "line": (
                    f"#{s.get('id')} {s.get('timestamp_s', 0):.2f}s {mark} "
                    f"{s.get('classification')} "
                    f"{s.get('target_label') or s.get('target_id') or ''}{rev}{hum}"
                ).strip(),
            }
        )

    coaching = doc.get("coaching") if not doc.get("coaching_locked") else None
    if watermark or not export:
        # Free: short wrap without full coaching export
        coaching_out = None
        share_extra = " · IronSight Free (upgrade for full coach export)"
    else:
        coaching_out = coaching
        share_extra = ""

    wrap = {
        "title": (doc.get("name") or "Range Session").upper(),
        "session_id": session_id,
        "headline": _headline(stats),
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "coach_locked": bool(doc.get("coach_locked")),
        "coach_ready": bool(doc.get("coach_locked") or doc.get("coach_ready")),
        "export_allowed": bool(export) and not watermark,
        "stats": {
            "shots": stats.get("total_shots") or len(real),
            "hits": stats.get("hits") or 0,
            "misses": stats.get("misses") or 0,
            "unknowns": stats.get("unknowns") or 0,
            "needs_review": stats.get("needs_review") or 0,
            "adjudicated": stats.get("adjudicated") or 0,
            "adjudicated_rate": stats.get("adjudicated_rate") or 0,
            "accuracy": stats.get("accuracy") or 0,
            "accuracy_display": stats.get("accuracy_display") or "—",
            "accuracy_trust": stats.get("accuracy_trust") or "none",
            "best_split_s": stats.get("best_split_s"),
            "avg_split_s": stats.get("avg_split_s"),
            "duration_s": stats.get("duration_s"),
            "not_a_shots": stats.get("not_a_shots") or 0,
            "human_corrections": stats.get("human_corrections") or 0,
        },
        "best_target": (
            {
                "label": best_t.get("label") or best_t.get("id"),
                "hits": best_t.get("hits"),
                "misses": best_t.get("misses"),
            }
            if best_t
            else None
        ),
        "shot_log": shot_lines if export else shot_lines[:5],
        "coaching": coaching_out,
        "coaching_locked": bool(doc.get("coaching_locked")),
        "cues": _cues(stats, real),
        "tech": {
            "cameras": len(doc.get("cameras") or []),
            "sync": (doc.get("sync") or {}).get("method"),
            "recon": (doc.get("pointcloud") or {}).get("method"),
            "gaussians": (doc.get("gaussian_splat") or {}).get("gaussian_count"),
            "tracks": (doc.get("tracks") or {}).get("method"),
            "pose": bool(doc.get("pose")),
        },
        "share_line": _share_line(doc, stats) + share_extra,
    }

    out = store.session_dir(session_id) / "wrap_card.json"
    out.write_text(json.dumps(wrap, indent=2))
    wrap["path"] = str(out)

    # Always write HTML export (watermarked if free)
    html_path = store.session_dir(session_id) / "wrap_export.html"
    html_path.write_text(_render_wrap_html(wrap, watermark=watermark or not export))
    wrap["html_path"] = str(html_path)

    md_path = store.session_dir(session_id) / "wrap_export.md"
    md_path.write_text(_render_wrap_md(wrap, watermark=watermark or not export))
    wrap["md_path"] = str(md_path)

    return wrap


def _cues(stats: dict, shots: list[dict]) -> list[str]:
    """Top actionable coaching cues from real log — no invented stats."""
    cues = []
    unk = int(stats.get("unknowns") or 0)
    if unk:
        cues.append(f"Review {unk} unscored shot(s) in the Command Center (tap Review / Annotate).")
    hits = int(stats.get("hits") or 0)
    misses = int(stats.get("misses") or 0)
    if hits + misses >= 3 and misses > hits:
        cues.append("Miss rate high on scored rounds — check sight picture and cadence on first targets.")
    if stats.get("best_split_s") and float(stats["best_split_s"]) > 2.5:
        cues.append("Best split is slow — work target-to-target transitions dry-fire.")
    if stats.get("avg_split_s") and float(stats["avg_split_s"]) < 1.0 and hits >= 3:
        cues.append("Fast splits — keep vision on steel through break.")
    if not cues:
        cues.append("Log is clean enough for a short debrief — mark any disputed calls as coach corrections.")
    return cues[:3]


def _headline(stats: dict) -> str:
    trust = stats.get("accuracy_trust") or "none"
    if trust in ("none", "unscored"):
        return "NEEDS REVIEW"
    if trust == "low":
        return "PARTIAL SCORE · REVIEW"
    acc = float(stats.get("accuracy") or 0)
    hits = int(stats.get("hits") or 0)
    if trust == "good" and acc >= 95 and hits >= 5:
        return "CLEAN RUN"
    if acc >= 80:
        return "SOLID SESSION"
    if acc >= 50:
        return "ROOM TO TIGHTEN"
    if hits == 0:
        return "WARM-UP / REVIEW"
    return "KEEP GRINDING"


def _share_line(doc: dict, stats: dict) -> str:
    name = doc.get("name") or "Range"
    disp = stats.get("accuracy_display") or f"{stats.get('accuracy', 0)}%"
    return (
        f"{name}: {stats.get('hits', 0)}H/{stats.get('misses', 0)}M "
        f"· {disp} · "
        f"best split {stats.get('best_split_s') or '—'}s · IronSight"
    )


def _render_wrap_md(wrap: dict, *, watermark: bool) -> str:
    s = wrap.get("stats") or {}
    lines = [
        f"# {wrap.get('title')}",
        f"**{wrap.get('headline')}**",
        "",
        f"- Hits: {s.get('hits')}  Misses: {s.get('misses')}  Unknown/Review: {s.get('unknowns')}",
        f"- Score: {s.get('accuracy_display')}",
        f"- Best split: {s.get('best_split_s') or '—'}s  Avg: {s.get('avg_split_s') or '—'}s",
        f"- Duration: {s.get('duration_s') or '—'}s",
        "",
        "## Shot log",
    ]
    for sh in wrap.get("shot_log") or []:
        lines.append(f"- {sh.get('line')}")
    if wrap.get("cues"):
        lines.append("")
        lines.append("## Cues")
        for c in wrap["cues"]:
            lines.append(f"- {c}")
    if wrap.get("coaching"):
        lines.append("")
        lines.append("## Coach notes")
        lines.append(wrap["coaching"])
    elif wrap.get("coaching_locked"):
        lines.append("")
        lines.append("_Full coaching notes require Pro._")
    lines.append("")
    lines.append(wrap.get("share_line") or "")
    if watermark:
        lines.append("")
        lines.append("_IronSight Free — upgrade for full export & coaching._")
    return "\n".join(lines) + "\n"


def _render_wrap_html(wrap: dict, *, watermark: bool) -> str:
    s = wrap.get("stats") or {}
    shot_html = "".join(
        f"<li>{html.escape(str(sh.get('line') or ''))}</li>" for sh in (wrap.get("shot_log") or [])
    )
    cues_html = "".join(f"<li>{html.escape(c)}</li>" for c in (wrap.get("cues") or []))
    coach = wrap.get("coaching") or (
        "<em>Full coaching notes require Pro.</em>" if wrap.get("coaching_locked") else ""
    )
    if coach and not coach.startswith("<"):
        coach = f"<p>{html.escape(coach)}</p>"
    wm = (
        "<p class='wm'>IronSight Free — upgrade for full export &amp; coaching.</p>"
        if watermark
        else ""
    )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>{html.escape(str(wrap.get('title') or 'Wrap'))}</title>
<style>
body{{font-family:ui-monospace,Menlo,monospace;background:#0b0f0c;color:#d7e7d7;padding:32px;max-width:720px;margin:0 auto}}
h1{{color:#3dff8a;letter-spacing:.08em;font-size:22px}}
.headline{{color:#9fefc0;margin-bottom:16px}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0}}
.card{{border:1px solid #1e3a28;padding:12px;background:#0f1612}}
.k{{font-size:10px;color:#6a8a72;letter-spacing:.12em}}
.v{{font-size:22px;color:#3dff8a}}
.share{{margin-top:24px;padding:12px;border:1px dashed #3dff8a55;color:#9fefc0}}
.wm{{opacity:.6;font-size:12px;margin-top:24px}}
ul{{line-height:1.6}}
</style></head><body>
<h1>{html.escape(str(wrap.get('title') or ''))}</h1>
<div class="headline">{html.escape(str(wrap.get('headline') or ''))}</div>
<div class="grid">
  <div class="card"><div class="k">HITS</div><div class="v">{s.get('hits')}</div></div>
  <div class="card"><div class="k">MISSES</div><div class="v">{s.get('misses')}</div></div>
  <div class="card"><div class="k">SCORE</div><div class="v" style="font-size:14px">{html.escape(str(s.get('accuracy_display') or ''))}</div></div>
  <div class="card"><div class="k">REVIEW</div><div class="v">{s.get('needs_review') or s.get('unknowns') or 0}</div></div>
</div>
<h3>Shot log</h3><ul>{shot_html}</ul>
<h3>Cues</h3><ul>{cues_html}</ul>
<h3>Coach notes</h3>{coach}
<div class="share">{html.escape(str(wrap.get('share_line') or ''))}</div>
{wm}
</body></html>
"""
