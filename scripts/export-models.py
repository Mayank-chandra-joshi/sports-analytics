#!/usr/bin/env python3
"""Export the ONNX models the engine loads, offline after the first fetch.

Run once per machine (or once and copy `models/` around). Everything it
pulls is free to use; nothing is downloaded at engine runtime.

    python3 scripts/export-models.py                 # detectors (COCO baseline)
    python3 scripts/export-models.py --reid          # + OSNet appearance model
    python3 scripts/export-models.py --list          # what it would do

Needs `pip install ultralytics onnx onnxsim` (and `torch`, CPU build is fine).
The football-specific weights are NOT fetched here: train or download them
from Roboflow Universe, export with the same helper, then add a manifest
entry with `sha256sum`.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

DETECTORS = [
    ("yolo11n", "yolo11n-coco-640.onnx", "fast, CPU-friendly baseline"),
    ("yolo11s", "yolo11s-coco-640.onnx", "better recall on small/far players"),
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_detector(name: str, out: str, imgsz: int = 640) -> Path:
    from ultralytics import YOLO

    dst = MODELS / out
    if dst.is_file():
        print(f"  {out} already present — skipping")
        return dst
    m = YOLO(f"{name}.pt")
    p = m.export(format="onnx", imgsz=imgsz, opset=17, simplify=True, dynamic=False)
    MODELS.mkdir(exist_ok=True)
    shutil.copy(p, dst)
    return dst


def export_reid(out: str = "osnet-x1_0-256x128.onnx") -> Path | None:
    """OSNet from torchreid (the SoccerNet-tuned weights the POC uses)."""
    dst = MODELS / out
    if dst.is_file():
        print(f"  {out} already present — skipping")
        return dst
    try:
        import torch
        import torchreid  # vendored in ../sports-reid, or pip install torchreid
    except ImportError:
        print("  torchreid not importable — skipping ReID export", file=sys.stderr)
        print("  (the engine runs without it; appearance_weight simply has no input)")
        return None
    model = torchreid.models.build_model("osnet_x1_0", num_classes=1000, pretrained=True)
    model.eval()
    dummy = torch.zeros(1, 3, 256, 128)
    MODELS.mkdir(exist_ok=True)
    torch.onnx.export(model, dummy, str(dst), opset_version=17,
                      input_names=["input"], output_names=["features"])
    return dst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reid", action="store_true", help="also export the OSNet appearance model")
    ap.add_argument("--list", action="store_true", help="print what would be exported and exit")
    args = ap.parse_args()

    if args.list:
        for name, out, why in DETECTORS:
            print(f"{name:10} -> models/{out}   ({why})")
        print("osnet_x1_0 -> models/osnet-x1_0-256x128.onnx   (--reid)")
        return 0

    made: list[Path] = []
    print("exporting detectors…")
    for name, out, _ in DETECTORS:
        made.append(export_detector(name, out))
    if args.reid:
        print("exporting ReID…")
        p = export_reid()
        if p:
            made.append(p)

    print("\nAdd or update these entries in models/manifest.toml:\n")
    for p in made:
        print(f'  # {p.name}\n  sha256 = "{sha256(p)}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
