"""Classical gunshot detection — no ML, no mocks.

Two profiles:
  - range  (default): strict impulsive muzzle-blast model for pure range audio
  - mixed: more permissive for compressed multi-source video (talking head + clips)

Pure-range model features:
  band-pass → onset strength + short-time energy
  high crest factor + short attack width
  spectral flatness / high-band energy ratio (broadband crack)
  refractory merge + ranked top-N by gunshot score
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import librosa
import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, find_peaks

from app.core.config import get_settings

ShotProfile = Literal["range", "mixed", "auto"]


@dataclass
class DetectedShot:
    timestamp_s: float
    energy: float
    peak_db: float
    duration_s: float
    score: float = 0.0


def _bandpass(y: np.ndarray, sr: int, low: float, high: float) -> np.ndarray:
    nyq = 0.5 * sr
    lo = max(low / nyq, 1e-5)
    hi = min(high / nyq, 0.999)
    if lo >= hi:
        return y
    b, a = butter(4, [lo, hi], btype="band")
    if len(y) < 32:
        return y
    padlen = min(3 * max(len(a), len(b)), len(y) - 1)
    try:
        return filtfilt(b, a, y, padlen=padlen)
    except ValueError:
        return y


def load_audio(path: str | Path, sr: int = 22050) -> tuple[np.ndarray, int]:
    y, sr_out = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32), int(sr_out)


def extract_audio_from_video(video_path: str | Path, out_wav: str | Path, sr: int = 44100) -> Path:
    import subprocess

    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", str(sr), "-ac", "1",
        str(out_wav),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out_wav.exists():
        raise RuntimeError(f"ffmpeg audio extract failed: {proc.stderr[-800:]}")
    return out_wav


def _hop_peak_amp(y: np.ndarray, hop: int, n_frames: int, frame_length: int) -> np.ndarray:
    yp = np.pad(np.abs(y.astype(np.float64)), (0, frame_length))
    peaks = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        s = i * hop
        peaks[i] = yp[s : s + frame_length].max() if s < len(yp) else 0.0
    return peaks


def _infer_profile(y: np.ndarray, sr: int) -> ShotProfile:
    """Heuristic: pure range has long low-energy gaps between sparse impulses."""
    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    if len(rms) < 20:
        return "mixed"
    quiet = float(np.mean(rms < np.percentile(rms, 40)))
    peakiness = float(np.percentile(rms, 99) / (np.median(rms) + 1e-9))
    # Range: lots of quiet + very peaky
    if quiet > 0.35 and peakiness > 4.0:
        return "range"
    if peakiness > 6.0 and quiet > 0.2:
        return "range"
    return "mixed"


def detect_shots(
    audio_path: str | Path,
    *,
    sr: int = 22050,
    min_interval_s: Optional[float] = None,
    energy_percentile: Optional[float] = None,
    onset_delta: Optional[float] = None,
    max_shots: int = 80,
    profile: ShotProfile = "auto",
) -> list[DetectedShot]:
    """Detect impulsive shot-like events from real audio."""
    settings = get_settings()
    min_interval_s = min_interval_s if min_interval_s is not None else settings.shot_min_interval_s
    energy_percentile = energy_percentile if energy_percentile is not None else settings.shot_energy_percentile
    onset_delta = onset_delta if onset_delta is not None else settings.shot_onset_delta

    y, sr = load_audio(audio_path, sr=sr)
    if len(y) < sr // 4:
        return []

    if profile == "auto":
        profile = _infer_profile(y, sr)

    # Strict range profile: pure live-fire (still works on lightly compressed range mics)
    if profile == "range":
        min_interval_s = max(min_interval_s, 0.28)
        energy_percentile = max(energy_percentile, 99.3)
        onset_delta = max(onset_delta, 0.40)
        crest_min = 5.0
        max_dur = 0.32
        local_contrast = 2.8
        highband_min = 0.14
        score_keep_k = 0.5
    else:
        crest_min = 3.2
        max_dur = 0.55
        local_contrast = 2.2
        highband_min = 0.10
        score_keep_k = 0.35

    y_bp = _bandpass(y, sr, settings.shot_bandpass_low_hz, settings.shot_bandpass_high_hz)

    hop = 256
    frame_length = 2048
    onset_env = librosa.onset.onset_strength(y=y_bp, sr=sr, hop_length=hop, aggregate=np.median)
    times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=hop)
    rms = librosa.feature.rms(y=y_bp, frame_length=frame_length, hop_length=hop)[0]

    # High-band energy ratio (gunshot crack is broadband)
    try:
        S = np.abs(librosa.stft(y_bp, n_fft=frame_length, hop_length=hop)) ** 2
        freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_length)
        hi_mask = freqs >= 1500
        lo_mask = freqs < 1500
        hi_e = S[hi_mask].sum(axis=0) + 1e-12
        lo_e = S[lo_mask].sum(axis=0) + 1e-12
        high_ratio = hi_e / (hi_e + lo_e)
    except Exception:
        high_ratio = np.ones(len(onset_env)) * 0.25

    n = min(len(onset_env), len(rms), len(times), len(high_ratio))
    onset_env = onset_env[:n].astype(np.float64)
    rms = rms[:n].astype(np.float64)
    times = times[:n]
    high_ratio = high_ratio[:n].astype(np.float64)
    if n < 8:
        return []

    peaks_amp = _hop_peak_amp(y_bp, hop, n, frame_length)

    onset_med = float(np.median(onset_env))
    onset_mad = float(np.median(np.abs(onset_env - onset_med))) + 1e-9
    rms_med = float(np.median(rms))
    rms_p = float(np.percentile(rms, min(energy_percentile, 99.5)))
    rms_p95 = float(np.percentile(rms, 95))
    rms_p99 = float(np.percentile(rms, 99))

    distance = max(1, int(min_interval_s * sr / hop))
    if profile == "range":
        prominence = max(3.5 * onset_mad, float(np.std(onset_env)) * 0.9, 0.12)
        height = onset_med + max(4.0, 5.0 * onset_delta) * onset_mad
    else:
        prominence = max(2.5 * onset_mad, float(np.std(onset_env)) * 0.6, 0.08)
        height = onset_med + max(2.5, 3.0 * onset_delta) * onset_mad

    peak_idxs, _ = find_peaks(
        onset_env,
        distance=distance,
        prominence=prominence,
        height=height,
    )

    candidates: list[tuple[float, DetectedShot]] = []
    for p in peak_idxs:
        t = float(times[p])
        e = float(rms[p])
        pk = float(peaks_amp[p])
        on = float(onset_env[p])
        cr = pk / (e + 1e-6)
        hr = float(high_ratio[p])

        # --- gates ---
        if profile == "range":
            # Extreme energy + impulsive onset; allow slightly lower crest if energy is top 0.5%
            energy_ok = e >= max(rms_p * 0.45, rms_p99 * 0.4, rms_med * 2.5)
            onset_ok = on >= onset_med + 4.5 * onset_mad
            if not energy_ok:
                continue
            if not onset_ok:
                continue
            if cr < crest_min and e < rms_p99 * 0.7:
                continue
            if hr < highband_min and cr < crest_min:
                continue
        else:
            energy_ok = e >= max(rms_p * 0.35, rms_p95 * 0.55, rms_med * 1.8)
            onset_ok = on >= onset_med + 3.5 * onset_mad
            if not (energy_ok or (onset_ok and e >= rms_med * 1.2)):
                continue
            if not onset_ok and cr < crest_min:
                continue
            if hr < highband_min * 0.7:
                continue

        # Local contrast
        lo = max(0, p - 12)
        hi = min(n, p + 13)
        neighborhood = np.concatenate([onset_env[lo:p], onset_env[p + 1 : hi]])
        if len(neighborhood):
            if on < float(np.median(neighborhood)) + local_contrast * (
                float(np.std(neighborhood)) + onset_mad
            ):
                continue

        # Duration from half-height
        left = int(p)
        while left > 0 and onset_env[left] > on * 0.45:
            left -= 1
        right = int(p)
        while right < n - 1 and onset_env[right] > on * 0.45:
            right += 1
        dur = float(max(0.012, times[min(right, n - 1)] - times[max(left, 0)]))
        if dur > max_dur:
            continue
        # Range: reject very long "whoosh" and very weak short clicks
        if profile == "range" and dur < 0.012:
            continue

        peak_db = float(20.0 * np.log10(max(e, 1e-9)))
        # Gunshot score
        score = (
            (on / (onset_med + 1e-6))
            * (e / (rms_med + 1e-6))
            * (1.0 + 0.2 * min(cr, 15.0))
            * (1.0 + 1.5 * hr)
        )
        candidates.append(
            (
                score,
                DetectedShot(
                    timestamp_s=t,
                    energy=e,
                    peak_db=peak_db,
                    duration_s=min(dur, 0.25),
                    score=float(score),
                ),
            )
        )

    candidates.sort(key=lambda x: x[1].timestamp_s)
    merged: list[tuple[float, DetectedShot]] = []
    for score, s in candidates:
        if merged and (s.timestamp_s - merged[-1][1].timestamp_s) < min_interval_s:
            if score > merged[-1][0]:
                merged[-1] = (score, s)
            continue
        merged.append((score, s))

    if merged:
        scores = np.array([m[0] for m in merged], dtype=np.float64)
        if profile == "range":
            # Keep only top-scoring impulses clearly above median
            score_floor = float(np.percentile(scores, 55)) + score_keep_k * float(np.std(scores) + 1e-9)
            strong = [m for m in merged if m[0] >= score_floor]
            # Always keep at least the strongest few if any passed gates
            if len(strong) < min(3, len(merged)):
                strong = sorted(merged, key=lambda x: x[0], reverse=True)[: min(8, len(merged))]
                strong.sort(key=lambda x: x[1].timestamp_s)
            merged = strong
        else:
            score_floor = float(np.median(scores)) + score_keep_k * float(np.std(scores) + 1e-9)
            strong = [m for m in merged if m[0] >= score_floor]
            if len(strong) < min(5, len(merged)):
                strong = sorted(merged, key=lambda x: x[0], reverse=True)[: min(12, len(merged))]
                strong.sort(key=lambda x: x[1].timestamp_s)
            merged = strong

    if len(merged) > max_shots:
        top = sorted(merged, key=lambda x: x[0], reverse=True)[:max_shots]
        top.sort(key=lambda x: x[1].timestamp_s)
        merged = top

    return [s for _, s in merged]


def dual_cam_audio_sync(
    audio_a_path: str | Path,
    audio_b_path: str | Path,
    *,
    sr: int = 22050,
    max_lag_s: float = 10.0,
) -> dict:
    """Cross-correlate onset envelopes to recover dual-cam time offset."""
    y_a, sr = load_audio(audio_a_path, sr=sr)
    y_b, sr = load_audio(audio_b_path, sr=sr)

    hop = 256
    env_a = librosa.onset.onset_strength(y=y_a, sr=sr, hop_length=hop)
    env_b = librosa.onset.onset_strength(y=y_b, sr=sr, hop_length=hop)

    env_a = (env_a - np.mean(env_a)) / (np.std(env_a) + 1e-9)
    env_b = (env_b - np.mean(env_b)) / (np.std(env_b) + 1e-9)

    max_lag_frames = int(max_lag_s * sr / hop)
    corr = signal.correlate(env_a, env_b, mode="full")
    lags = signal.correlation_lags(len(env_a), len(env_b), mode="full")

    mask = (lags >= -max_lag_frames) & (lags <= max_lag_frames)
    corr_w = corr[mask]
    lags_w = lags[mask]
    if len(corr_w) == 0:
        return {"offset_s": 0.0, "offset_frames": 0.0, "peak_ratio": 0.0, "passed": False}

    peak_idx = int(np.argmax(corr_w))
    peak_val = float(corr_w[peak_idx])
    tmp = corr_w.copy()
    tmp[max(0, peak_idx - 5) : peak_idx + 6] = -np.inf
    second = float(np.max(tmp)) if np.isfinite(tmp).any() else 0.0
    peak_ratio = peak_val / (abs(second) + 1e-9) if second else float("inf")

    lag_frames = int(lags_w[peak_idx])
    offset_s = -float(lag_frames * hop / sr)

    details = {"full": {"offset_s": offset_s, "peak_ratio": peak_ratio}}
    mid = len(env_a) // 2
    if mid > 50 and len(env_b) > 100:
        for name, sa, ea, sb, eb in (
            ("first_half", 0, mid, 0, min(mid, len(env_b))),
            ("second_half", mid, len(env_a), min(mid, len(env_b)), len(env_b)),
        ):
            ca = env_a[sa:ea]
            cb = env_b[sb:eb]
            if len(ca) < 20 or len(cb) < 20:
                continue
            c = signal.correlate(ca - ca.mean(), cb - cb.mean(), mode="full")
            l = signal.correlation_lags(len(ca), len(cb), mode="full")
            m = (l >= -max_lag_frames) & (l <= max_lag_frames)
            if not m.any():
                continue
            pi = int(np.argmax(c[m]))
            lf = int(l[m][pi])
            details[name] = {"offset_s": float(-lf * hop / sr)}

    offsets = [details["full"]["offset_s"]]
    for k in ("first_half", "second_half"):
        if k in details:
            offsets.append(details[k]["offset_s"])
    spread = float(np.max(offsets) - np.min(offsets)) if len(offsets) > 1 else 0.0
    passed = peak_ratio >= 1.5 and spread < 0.05

    return {
        "offset_s": float(np.median(offsets)),
        "offset_frames": float(np.median(offsets) * 30.0),
        "peak_ratio": float(peak_ratio),
        "passed": bool(passed),
        "method": "audio_cross_correlation",
        "details": {**details, "spread_s": spread},
    }
