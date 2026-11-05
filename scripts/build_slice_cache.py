#!/usr/bin/env python3
"""Build a compact 2D slice cache for reproducible W5 retraining.

Loading the original NIfTI volumes needs ~7 GB of RAM (the full train+val
3D arrays), which is more than this machine can hold comfortably.  This
script applies the frozen v1.0 preprocessing once and stores 256x256 slices
in three small ``.npz`` files (~350 MB total):

    data/processed/slice_cache/train.npz
    data/processed/slice_cache/val.npz
    data/processed/slice_cache/test.npz

Each file contains: images (float16), masks (uint8), sample_ids,
patient_ids and lesion_present.  The training pipeline then applies the
same Albumentations transforms on these cached slices.

Example
-------
python scripts/build_slice_cache.py
python scripts/build_slice_cache.py --image_size 256 --lesion_label 1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.mask_binarize import binarize_mask  # noqa: E402
from src.data.transform import normalize_min_max, resize_ct_pair  # noqa: E402


def read_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build 2D slice cache for W5 retraining")
    parser.add_argument("--manifest", default="data/processed/clean_slices_manifest.csv")
    parser.add_argument("--splits", default="data/processed/splits_patient_level.json")
    parser.add_argument("--output_dir", default="data/processed/slice_cache")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--lesion_label", type=int, default=1)
    parser.add_argument("--splits_to_build", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    rows = read_manifest(Path(args.manifest))
    split_data = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    patient_splits = split_data["patient_splits"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = (args.image_size, args.image_size)

    for split in args.splits_to_build:
        patients = set(patient_splits[split])
        subset = [r for r in rows if r["patient_id"] in patients]
        # Group by volume so each NIfTI is loaded exactly once.
        subset.sort(key=lambda r: (r["source_volume_path"], int(r["slice_index"])))
        count = len(subset)
        print(f"\n>> {split}: {count} slices from {len(patients)} patients")

        images = np.empty((count, *target), dtype=np.float16)
        masks = np.empty((count, *target), dtype=np.uint8)
        sample_ids: list[str] = []
        patient_ids: list[str] = []
        lesion_present: list[int] = []

        current_volume = None
        ct_volume = None
        mask_volume = None
        for index, record in enumerate(subset):
            volume_path = record["source_volume_path"]
            if volume_path != current_volume:
                ct_volume = nib.load(volume_path).get_fdata(dtype=np.float32)
                mask_volume = nib.load(record["source_mask_path"]).get_fdata(dtype=np.float32)
                current_volume = volume_path
                print(f"   loaded {Path(volume_path).name}")

            z = int(record["slice_index"])
            image = normalize_min_max(ct_volume[:, :, z])
            mask = binarize_mask(mask_volume[:, :, z], lesion_label=args.lesion_label)
            image, mask = resize_ct_pair(image, mask, target_size=target)

            images[index] = image.astype(np.float16)
            masks[index] = mask.astype(np.uint8)
            sample_ids.append(record["sample_id"])
            patient_ids.append(record["patient_id"])
            lesion_present.append(int(record["lesion_present"]))

            if (index + 1) % 200 == 0:
                print(f"   ... {index + 1}/{count}")

        output = output_dir / f"{split}.npz"
        np.savez(
            output,
            images=images,
            masks=masks,
            sample_ids=np.asarray(sample_ids, dtype="<U64"),
            patient_ids=np.asarray(patient_ids, dtype="<U64"),
            lesion_present=np.asarray(lesion_present, dtype=np.uint8),
        )
        print(f"[+] {output}  ({output.stat().st_size / 1e6:.1f} MB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
