"""Honest session scoring — never claim 100% when most shots are UNKNOWN."""
from __future__ import annotations

from typing import Any, Optional


def compute_stats(shots: list[dict], duration_s: float = 0.0) -> dict[str, Any]:
    """Compute coach-safe accuracy and review metrics.

    Accuracy is only among HIT+MISS (adjudicated). When few shots are adjudicated,
    we surface that clearly so the UI never pretends the session is "100% clean".
    """
    hits = sum(1 for s in shots if s.get("classification") == "HIT")
    misses = sum(1 for s in shots if s.get("classification") == "MISS")
    unknowns = sum(1 for s in shots if s.get("classification") == "UNKNOWN")
    not_shots = sum(1 for s in shots if s.get("classification") == "NOT_A_SHOT")
    human = sum(1 for s in shots if s.get("human_corrected"))
    needs_review = sum(
        1
        for s in shots
        if s.get("classification") == "UNKNOWN"
        or s.get("needs_review")
        or (s.get("classification") in ("HIT", "MISS") and float(s.get("confidence") or 0) < 0.45)
    )
    known = hits + misses
    real = [s for s in shots if s.get("classification") in ("HIT", "MISS", "UNKNOWN")]
    confs = [
        float(s.get("confidence") or 0)
        for s in shots
        if s.get("classification") in ("HIT", "MISS") and s.get("confidence") is not None
    ]
    acc = (hits / known * 100.0) if known else 0.0
    adjudicated_rate = (known / len(real) * 100.0) if real else 0.0

    # Display rules: only call out accuracy when enough of the string was scored
    if not real:
        acc_display = "— no shots"
        acc_trust = "none"
    elif known == 0:
        acc_display = f"0 adjudicated · {unknowns} need review"
        acc_trust = "unscored"
    elif adjudicated_rate < 40.0:
        acc_display = f"{hits}H/{misses}M of {known} scored · {unknowns} need review"
        acc_trust = "low"
    elif adjudicated_rate < 70.0:
        acc_display = f"{round(acc, 0):.0f}% of scored ({known}/{len(real)}) · {unknowns} review"
        acc_trust = "partial"
    else:
        acc_display = f"{round(acc, 0):.0f}% ({hits}H/{misses}M)"
        acc_trust = "good" if unknowns == 0 else "partial"

    splits: list[float] = []
    ts = sorted(float(s["timestamp_s"]) for s in real if s.get("timestamp_s") is not None)
    for a, b in zip(ts, ts[1:]):
        splits.append(b - a)

    return {
        "total_shots": len(real),
        "total_events": len(shots),
        "hits": hits,
        "misses": misses,
        "unknowns": unknowns,
        "not_a_shots": not_shots,
        "adjudicated": known,
        "adjudicated_rate": round(adjudicated_rate, 1),
        "accuracy": round(acc, 1),
        "accuracy_display": acc_display,
        "accuracy_trust": acc_trust,
        "needs_review": needs_review,
        "human_corrections": human,
        "avg_confidence": round(sum(confs) / len(confs), 3) if confs else None,
        "duration_s": float(duration_s or 0.0),
        "best_split_s": round(min(splits), 3) if splits else None,
        "avg_split_s": round(sum(splits) / len(splits), 3) if splits else None,
    }


def recompute_session_stats(doc: dict[str, Any]) -> dict[str, Any]:
    duration = float((doc.get("stats") or {}).get("duration_s") or 0.0)
    if not duration:
        for cam in doc.get("cameras") or []:
            duration = max(duration, float(cam.get("duration_s") or 0.0))
    stats = compute_stats(doc.get("shots") or [], duration)
    doc["stats"] = stats
    return stats
