#!/usr/bin/env python3
"""
Task ID: EVAL-02 Bridge (Data & Evaluation Lead)
Tên script: scripts/sync_eval02_predictions.py
Mục đích:
- Quét thư mục kết quả suy luận thô raw_predictions_tensors_eval02/ của cả 3 mô hình:
  + Standard U-Net (Baseline): unet
  + SegNet (Baseline): segnet
  + Attention U-Net (Proposed): att_unet
- Chuẩn hóa cấu trúc lưu trữ về: predictions/{unet, segnet, att_unet}/.
- Đảm bảo tên file tương ứng khớp 1:1 với sample_id trong data/processed/test_manifest.json (265 lát cắt).
- Đồng bộ các mảng tensor xác suất (.npy), tạo binary mask (.png tại threshold 0.5) và sao chép predictions_manifest.json.
- Kiểm tra toàn vẹn dữ liệu (Integrity Verification) và báo cáo trạng thái.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

MODELS = {
    "att_unet": "Attention U-Net (Proposed)",
    "unet": "Standard U-Net (Baseline)",
    "segnet": "SegNet (Baseline)",
}


def load_manifest_sample_ids(manifest_path: Path) -> List[str]:
    """Tải danh sách sample_id từ test_manifest.json."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"[ERROR] Không tìm thấy manifest: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    slices = data.get("slices", [])
    sample_ids = [s["sample_id"] for s in slices]
    return sample_ids


def sync_model_predictions(
    raw_dir: Path,
    dest_dir: Path,
    expected_sample_ids: List[str],
    generate_png: bool = True,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Đồng bộ dự đoán của 1 mô hình từ raw sang dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    synced_npy = 0
    synced_png = 0
    missing_samples = []

    # 1. Đồng bộ file manifest nếu có
    raw_manifest = raw_dir / "predictions_manifest.json"
    if raw_manifest.exists():
        shutil.copy2(raw_manifest, dest_dir / "predictions_manifest.json")

    # 2. Đồng bộ từng sample_id
    for sid in expected_sample_ids:
        raw_npy = raw_dir / f"{sid}.npy"
        dest_npy = dest_dir / f"{sid}.npy"
        dest_png = dest_dir / f"{sid}.png"

        if not raw_npy.exists():
            missing_samples.append(sid)
            continue

        # Sao chép file .npy nếu chưa có hoặc khác kích thước
        if not dest_npy.exists() or dest_npy.stat().st_size != raw_npy.stat().st_size:
            shutil.copy2(raw_npy, dest_npy)
        synced_npy += 1

        # Tạo file .png nếu cần
        if generate_png and not dest_png.exists():
            arr = np.load(raw_npy)
            if arr.ndim > 2:
                arr = arr.squeeze()
            bin_mask = ((arr >= threshold) * 255).astype(np.uint8)
            Image.fromarray(bin_mask).save(dest_png)
            synced_png += 1
        elif dest_png.exists():
            synced_png += 1

    return {
        "expected": len(expected_sample_ids),
        "synced_npy": synced_npy,
        "synced_png": synced_png,
        "missing_count": len(missing_samples),
        "missing_samples": missing_samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="EVAL-02: Sync and standardize raw prediction tensors into predictions/"
    )
    parser.add_argument(
        "--raw_predictions_dir",
        type=Path,
        default=Path("raw_predictions_tensors_eval02"),
        help="Thư mục chứa raw predictions tensors",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("predictions"),
        help="Thư mục đích lưu trữ predictions chuẩn hóa",
    )
    parser.add_argument(
        "--manifest_path",
        type=Path,
        default=Path("data/processed/test_manifest.json"),
        help="Đường dẫn file test_manifest.json",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Ngưỡng nhị phân hóa (mặc định 0.5)",
    )
    args = parser.parse_args()

    print("======================================================================")
    print(" TASK EVAL-02: SYNC AND STANDARDIZE PREDICTION TENSORS")
    print("======================================================================")

    if not args.raw_predictions_dir.exists():
        print(f"[!] Lỗi: Không tìm thấy thư mục {args.raw_predictions_dir}", file=sys.stderr)
        return 1

    expected_sample_ids = load_manifest_sample_ids(args.manifest_path)
    print(f"[*] Loaded {len(expected_sample_ids)} expected test samples from manifest.")

    total_issues = 0
    for model_key, display_name in MODELS.items():
        raw_model_dir = args.raw_predictions_dir / model_key
        dest_model_dir = args.output_dir / model_key

        if not raw_model_dir.exists():
            print(f"[!] Lỗi: Không tìm thấy thư mục mô hình: {raw_model_dir}", file=sys.stderr)
            total_issues += 1
            continue

        stats = sync_model_predictions(
            raw_dir=raw_model_dir,
            dest_dir=dest_model_dir,
            expected_sample_ids=expected_sample_ids,
            generate_png=True,
            threshold=args.threshold,
        )

        print(f"[*] Model: {display_name} ({model_key})")
        print(f"    - NPY Tensors: {stats['synced_npy']}/{stats['expected']}")
        print(f"    - PNG Masks:   {stats['synced_png']}/{stats['expected']}")

        if stats["missing_count"] > 0:
            print(f"    [!] Cảnh báo: Thiếu {stats['missing_count']} samples: {stats['missing_samples'][:3]}...")
            total_issues += 1
        else:
            print("    [+] Khớp 1:1 hoàn toàn 265/265 mẫu test.")

    if total_issues > 0:
        print(f"[ERROR] Phát hiện {total_issues} vấn đề trong quá trình đồng bộ!", file=sys.stderr)
        return 1

    print("[SUCCESS] EVAL-02 sync completed successfully.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

