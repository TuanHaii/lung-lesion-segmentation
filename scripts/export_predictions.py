#!/usr/bin/env python3
"""Export per-slice test-set predictions for EVAL-03 .. EVAL-07.

This is the missing bridge between training checkpoints and the metric
scripts.  It writes one probability map per test sample:

    predictions/att_unet/<sample_id>.npy
    predictions/unet/<sample_id>.npy
    predictions/segnet/<sample_id>.npy

The metric scripts binarize these maps at threshold 0.5, so the same exported
file can also be reused for threshold sensitivity analysis.

Example
-------
python scripts/export_predictions.py \
    --checkpoint checkpoints/EXP-ATTUNET-001_best.pth \
    --model AttentionUNet \
    --output_dir predictions/att_unet
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.eval_loader import get_eval_dataloader  # noqa: E402
from src.evaluate import load_model_from_checkpoint  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Export test-set segmentation probabilities")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--model",
        required=True,
        choices=["AttentionUNet", "AttentionUnet", "attention_unet", "UNet", "Unet", "unet", "SegNet", "Segnet", "segnet"],
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--manifest", default="data/processed/test_manifest.json")
    parser.add_argument("--raw_data_dir", default=None)
    parser.add_argument("--lesion_label", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Debug: only export the first N slices")
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Save a 0/1 uint8 mask instead of a probability map",
    )
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f">> Device: {device}")

    model = load_model_from_checkpoint(args.checkpoint, args.model, device)
    loader = get_eval_dataloader(
        manifest_path=args.manifest,
        lesion_label=args.lesion_label,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        raw_data_dir=args.raw_data_dir,
        enforce_readonly=True,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported = 0
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if args.limit is not None and index >= args.limit:
                break
            images = batch["image"].to(device=device, dtype=torch.float32)
            probs = torch.sigmoid(model(images)).squeeze().cpu().numpy().astype(np.float32)
            sample_id = batch["sample_id"][0]
            if args.binary:
                payload = (probs >= 0.5).astype(np.uint8)
            else:
                payload = probs
            np.save(output_dir / f"{sample_id}.npy", payload)
            exported += 1
            if (index + 1) % 50 == 0:
                print(f"  ... {index + 1}/{len(loader)}")

    metadata = {
        "model": args.model,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split": "test",
        "manifest": str(Path(args.manifest).resolve()),
        "lesion_label": args.lesion_label,
        "threshold": 0.5,
        "binary": bool(args.binary),
        "exported_slices": exported,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
    }
    (output_dir / "predictions_manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[+] Exported {exported} predictions to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
