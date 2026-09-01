"""Trained YOLO plate detector for IronSight range targets.

Builds a YOLO dataset from weak labels (CV + VLM keyframe bboxes in tracks.json),
fine-tunes ultralytics YOLOv8n, and exposes detect() for the multi-view tracker.

Weights live at data/models/plates_yolo/best.pt
"""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from app.core.config import ROOT, DATA_DIR

MODELS_DIR = DATA_DIR / "models" / "plates_yolo"
DATASET_DIR = DATA_DIR / "datasets" / "plates_yolo"
WEIGHTS_PATH = MODELS_DIR / "best.pt"
LAST_META = MODELS_DIR / "train_meta.json"

# Single class: range plate / popper / silhouette
CLASS_NAMES = ["plate"]
_model_cache: Any = None


def weights_available() -> bool:
    return WEIGHTS_PATH.exists() and WEIGHTS_PATH.stat().st_size > 10_000


def _yolo_line(cls: int, bbox_xywh: list[float]) -> str:
    """bbox normalized xywh (top-left) → YOLO cxcywh."""
    x, y, w, h = [float(v) for v in bbox_xywh[:4]]
    w = max(1e-4, min(1.0, w))
    h = max(1e-4, min(1.0, h))
    cx = min(1.0, max(0.0, x + w / 2))
    cy = min(1.0, max(0.0, y + h / 2))
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def harvest_labels_from_sessions(
    sessions_root: Optional[Path] = None,
    *,
    min_conf: float = 0.45,
) -> list[dict[str, Any]]:
    """Collect (image_path, boxes) from tracks keyframes + shot frames with bboxes."""
    sessions_root = sessions_root or (DATA_DIR / "sessions")
    samples: list[dict[str, Any]] = []

    if not sessions_root.exists():
        return samples

    for sdir in sorted(sessions_root.iterdir()):
        if not sdir.is_dir():
            continue
        # tracks.json keyframes
        tracks_path = sdir / "tracks" / "tracks.json"
        if tracks_path.exists():
            try:
                doc = json.loads(tracks_path.read_text())
            except Exception:
                doc = {}
            for kf in doc.get("keyframes") or []:
                img = kf.get("frame")
                if not img or not Path(img).exists():
                    continue
                boxes = []
                for t in kf.get("targets") or []:
                    b = t.get("bbox")
                    conf = float(t.get("confidence") or t.get("conf") or 0.5)
                    if not b or len(b) < 4 or conf < min_conf:
                        continue
                    # drop motion/noise and tiny/huge
                    if max(b[2], b[3]) < 0.008 or max(b[2], b[3]) > 0.55:
                        continue
                    if str(t.get("label") or "").upper() == "MOTION":
                        continue
                    boxes.append({"bbox": b, "conf": conf, "source": t.get("source")})
                if boxes:
                    samples.append({"image": str(img), "boxes": boxes, "session": sdir.name})

        # shot frames with VLM visible_targets
        sess = sdir / "session.json"
        if sess.exists():
            try:
                sdoc = json.loads(sess.read_text())
            except Exception:
                sdoc = {}
            for sh in sdoc.get("shots") or []:
                # primary shot frames
                for fp in sh.get("frame_paths") or []:
                    if not Path(fp).exists():
                        continue
                    boxes = []
                    if sh.get("bbox"):
                        boxes.append(
                            {
                                "bbox": sh["bbox"],
                                "conf": float(sh.get("confidence") or 0.8),
                                "source": "shot",
                            }
                        )
                    for v in sh.get("visible_targets") or []:
                        if v.get("bbox"):
                            boxes.append(
                                {
                                    "bbox": v["bbox"],
                                    "conf": float(v.get("confidence") or 0.7),
                                    "source": "shot_vis",
                                }
                            )
                    if boxes:
                        samples.append(
                            {"image": str(fp), "boxes": boxes, "session": sdir.name}
                        )

    return samples


def build_yolo_dataset(
    samples: list[dict[str, Any]],
    out_dir: Path = DATASET_DIR,
    *,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Path:
    """Write YOLO-format dataset; returns path to data.yaml."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True)
        (out_dir / "labels" / split).mkdir(parents=True)

    rng = random.Random(seed)
    # dedupe by image path
    by_img: dict[str, list] = {}
    for s in samples:
        by_img.setdefault(s["image"], []).extend(s["boxes"])

    items = list(by_img.items())
    rng.shuffle(items)
    n_val = max(1, int(len(items) * val_ratio)) if len(items) > 5 else max(1, len(items) // 5)
    val_set = set(i for i, _ in items[:n_val])

    counts = {"train": 0, "val": 0}
    for img_path, boxes in items:
        src = Path(img_path)
        if not src.exists():
            continue
        split = "val" if img_path in val_set else "train"
        # unique name
        stem = f"{src.parent.name}_{src.stem}"[:80]
        dst_img = out_dir / "images" / split / f"{stem}{src.suffix or '.jpg'}"
        dst_lbl = out_dir / "labels" / split / f"{stem}.txt"
        try:
            shutil.copy2(src, dst_img)
        except Exception:
            continue
        lines = []
        for b in boxes:
            lines.append(_yolo_line(0, b["bbox"]))
        dst_lbl.write_text("\n".join(lines) + ("\n" if lines else ""))
        counts[split] += 1

    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(
        f"path: {out_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n  0: plate\n"
    )
    (out_dir / "counts.json").write_text(json.dumps(counts, indent=2))
    return yaml_path


def train_plate_yolo(
    *,
    epochs: int = 40,
    imgsz: int = 640,
    batch: int = 8,
    model_name: str = "yolov8n.pt",
    force: bool = False,
) -> dict[str, Any]:
    """Harvest labels → train YOLO → copy best.pt to MODELS_DIR."""
    from ultralytics import YOLO

    samples = harvest_labels_from_sessions()
    if len(samples) < 8:
        # Still train if we have at least a few unique images after build
        pass
    yaml_path = build_yolo_dataset(samples)
    counts = json.loads((DATASET_DIR / "counts.json").read_text())
    if counts.get("train", 0) < 4:
        raise RuntimeError(
            f"Not enough labeled images to train YOLO (train={counts.get('train')}). "
            "Run pipeline on range dual-cam first to harvest boxes."
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(model_name)
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(MODELS_DIR / "runs"),
        name="plates",
        exist_ok=True,
        verbose=True,
        patience=12,
        device="mps" if _mps_ok() else "cpu",
        workers=2,
    )

    # Find best.pt
    run_dir = MODELS_DIR / "runs" / "plates"
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        # ultralytics may nest
        candidates = list((MODELS_DIR / "runs").rglob("best.pt"))
        if not candidates:
            raise RuntimeError("YOLO training finished but best.pt not found")
        best = max(candidates, key=lambda p: p.stat().st_mtime)

    shutil.copy2(best, WEIGHTS_PATH)
    meta = {
        "weights": str(WEIGHTS_PATH),
        "epochs": epochs,
        "imgsz": imgsz,
        "train_images": counts.get("train"),
        "val_images": counts.get("val"),
        "samples_harvested": len(samples),
        "class_names": CLASS_NAMES,
        "source": "weak_labels_cv_vlm_shot",
    }
    LAST_META.write_text(json.dumps(meta, indent=2))
    global _model_cache
    _model_cache = None
    return meta


def _mps_ok() -> bool:
    try:
        import torch

        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def load_model(weights: Optional[Path] = None):
    global _model_cache
    path = Path(weights) if weights else WEIGHTS_PATH
    if not path.exists():
        return None
    if _model_cache is not None:
        return _model_cache
    from ultralytics import YOLO

    _model_cache = YOLO(str(path))
    return _model_cache


def detect_plates_yolo(
    frame_bgr: np.ndarray,
    *,
    conf: float = 0.25,
    iou: float = 0.45,
    weights: Optional[Path] = None,
) -> list[dict[str, Any]]:
    """Run trained YOLO; return normalized xywh detections."""
    model = load_model(weights)
    if model is None:
        return []
    h, w = frame_bgr.shape[:2]
    # ultralytics accepts BGR numpy
    res = model.predict(frame_bgr, conf=conf, iou=iou, verbose=False)
    if not res:
        return []
    r0 = res[0]
    boxes = getattr(r0, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    out = []
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    for i, (x1, y1, x2, y2) in enumerate(xyxy):
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        bbox = [
            float(max(0, x1) / w),
            float(max(0, y1) / h),
            float(min(w, bw) / w),
            float(min(h, bh) / h),
        ]
        c = float(confs[i])
        out.append(
            {
                "id": f"Y{i+1}",
                "label": "PLATE",
                "color": "steel",
                "bbox": bbox,
                "confidence": c,
                "source": "yolo",
            }
        )
    return out


def detect_plates_yolo_path(image_path: Path, **kwargs) -> list[dict[str, Any]]:
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    return detect_plates_yolo(img, **kwargs)
