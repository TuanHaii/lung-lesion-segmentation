#!/usr/bin/env python3
"""
Task ID: EVAL-02 (Sprint W5 - Model Evaluation & Inference Pipeline)
Mô tả: Thực hiện đánh giá độc lập (Inference Pipeline) trên Test Set cho 3 mô hình:
       1. Attention U-Net (Proposed)
       2. Standard U-Net (Baseline)
       3. SegNet (Baseline)

Tự động thực hiện các bước:
1. Tải checkpoint tốt nhất của 3 mô hình (ưu tiên EXP-SEGNET-002 -> EXP-SEGNET-001).
2. Chạy suy luận (Inference) trên Independent Test Set (265 slices, 3 bệnh nhân test).
3. Đo đạc chính xác thời gian suy luận (Latency ms/slice) trên cùng một phần cứng.
4. Xuất các raw predictions tensors (.npy) vào predictions/{att_unet, unet, segnet}/ kèm manifest.
5. Tính toán các chỉ số phân đoạn theo chuẩn Dual-Reporting:
   - Chế độ 1: Lesion-containing Slices (GT > 0)
   - Chế độ 2: All Slices
   - Chế độ 3: Patient-level 3D Volume Aggregation
6. Lưu báo cáo chi tiết vào reports/{attunet, unet, segnet}/.
7. Tổng hợp bảng so sánh định lượng (CSV, Markdown, LaTeX) và kiểm định Wilcoxon.
"""

import os
import sys
import time
import json
import csv
import math
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.unet import UNet
from src.models.attention_unet import AttentionUNet
from src.models.segnet import SegNet
from src.data.eval_loader import get_eval_dataloader
from scripts.compare_models import generate_comparison_report


def count_parameters(model: nn.Module) -> int:
    """Đếm tổng số tham số có thể huấn luyện của mô hình."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_model(checkpoint_path: str, model_type: str, device: torch.device) -> nn.Module:
    """Tải và khởi tạo mô hình từ checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    norm_type = model_type.lower()
    if "attention" in norm_type or "att" in norm_type:
        model = AttentionUNet(in_channels=1, out_channels=1)
    elif "segnet" in norm_type:
        model = SegNet(in_channels=1, num_classes=1, init_features=32)
    elif "unet" in norm_type:
        model = UNet(in_channels=1, out_channels=1)
    else:
        raise ValueError(f"Loại mô hình không được hỗ trợ: {model_type}")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def compute_metrics_slice(tp: int, fp: int, fn: int, tn: int, eps: float = 1e-6) -> Dict[str, float]:
    """Tính các chỉ số phân đoạn cho một lát cắt tuân thủ tài liệu hướng dẫn."""
    is_gt_empty = (tp + fn) == 0
    is_pred_empty = (tp + fp) == 0

    if is_gt_empty and is_pred_empty:
        return {"dice": 1.0, "iou": 1.0, "precision": 1.0, "recall": 1.0}
    elif is_gt_empty and not is_pred_empty:
        return {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 1.0}
    elif not is_gt_empty and is_pred_empty:
        return {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}

    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    return {
        "dice": float(np.clip(dice, 0.0, 1.0)),
        "iou": float(np.clip(iou, 0.0, 1.0)),
        "precision": float(np.clip(precision, 0.0, 1.0)),
        "recall": float(np.clip(recall, 0.0, 1.0)),
    }


def evaluate_single_model(
    model_name: str,
    checkpoint_path: str,
    device: torch.device,
    test_loader,
    reports_dir: Path,
    predictions_dir: Path,
    threshold: float = 0.5,
    export_predictions: bool = True,
) -> Dict[str, Any]:
    """Chạy suy luận, đo độ trễ và lưu báo cáo cho 1 mô hình."""
    print(f"\n>> Đang đánh giá mô hình: [{model_name}]")
    print(f"   [+] Checkpoint : {checkpoint_path}")
    print(f"   [+] Báo cáo    : {reports_dir}")
    print(f"   [+] Predictions: {predictions_dir}")

    reports_dir.mkdir(parents=True, exist_ok=True)
    if export_predictions:
        predictions_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(checkpoint_path, model_name, device)
    param_count = count_parameters(model)
    ckpt_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)

    slice_records = []
    patient_accum = {}
    latencies = []

    # Warmup GPU
    if device.type == "cuda":
        dummy = torch.randn(1, 1, 256, 256, device=device)
        for _ in range(5):
            _ = model(dummy)
        torch.cuda.synchronize()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Inference {model_name}"):
            images = batch["image"].to(device=device, dtype=torch.float32)
            masks = batch["mask"].cpu().numpy()
            sample_id = batch["sample_id"][0]
            patient_id = batch["patient_id"][0]
            slice_idx = int(batch["slice_index"][0]) if "slice_index" in batch else 0

            # Đo latency chính xác
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000.0
            latencies.append(latency_ms)

            probs = torch.sigmoid(outputs).squeeze().cpu().numpy().astype(np.float32)
            pred_bin = (probs >= threshold).astype(np.uint8)
            gt_bin = (masks.squeeze() >= 0.5).astype(np.uint8)

            # Lưu prediction cả .npy (tensor xác suất) và .png (binary mask 0/255)
            if export_predictions:
                np.save(predictions_dir / f"{sample_id}.npy", probs)
                mask_uint8 = (pred_bin * 255).astype(np.uint8)
                Image.fromarray(mask_uint8).save(predictions_dir / f"{sample_id}.png")

            # Tính toán confusion
            p_flat = pred_bin.flatten().astype(bool)
            g_flat = gt_bin.flatten().astype(bool)

            tp = int(np.logical_and(p_flat, g_flat).sum())
            fp = int(np.logical_and(p_flat, np.logical_not(g_flat)).sum())
            fn = int(np.logical_and(np.logical_not(p_flat), g_flat).sum())
            tn = int(np.logical_and(np.logical_not(p_flat), np.logical_not(g_flat)).sum())

            has_lesion = 1 if (tp + fn) > 0 else 0
            metrics = compute_metrics_slice(tp, fp, fn, tn)

            slice_records.append({
                "sample_id": sample_id,
                "patient_id": patient_id,
                "slice_index": slice_idx,
                "lesion_present": has_lesion,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "dice": metrics["dice"],
                "iou": metrics["iou"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "latency_ms": round(latency_ms, 2)
            })

            # Tích luỹ cấp bệnh nhân
            if patient_id not in patient_accum:
                patient_accum[patient_id] = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "slices": 0}
            patient_accum[patient_id]["tp"] += tp
            patient_accum[patient_id]["fp"] += fp
            patient_accum[patient_id]["fn"] += fn
            patient_accum[patient_id]["tn"] += tn
            patient_accum[patient_id]["slices"] += 1

    # 1. Xuất file test_metrics_slice_level.csv
    slice_csv = reports_dir / "test_metrics_slice_level.csv"
    with open(slice_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sample_id", "patient_id", "slice_index", "lesion_present",
            "tp", "fp", "fn", "tn", "dice", "iou", "precision", "recall", "latency_ms"
        ])
        writer.writeheader()
        writer.writerows(slice_records)

    # 2. Xuất file test_metrics_patient_level.csv
    patient_records = []
    for pid, counts in patient_accum.items():
        pm = compute_metrics_slice(counts["tp"], counts["fp"], counts["fn"], counts["tn"])
        patient_records.append({
            "patient_id": pid,
            "total_slices": counts["slices"],
            "tp": counts["tp"],
            "fp": counts["fp"],
            "fn": counts["fn"],
            "tn": counts["tn"],
            "patient_dice": pm["dice"],
            "patient_iou": pm["iou"],
            "patient_precision": pm["precision"],
            "patient_recall": pm["recall"],
        })

    patient_csv = reports_dir / "test_metrics_patient_level.csv"
    with open(patient_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "patient_id", "total_slices", "tp", "fp", "fn", "tn",
            "patient_dice", "patient_iou", "patient_precision", "patient_recall"
        ])
        writer.writeheader()
        writer.writerows(patient_records)

    # 3. Xuất file predictions_manifest.json
    if export_predictions:
        manifest_meta = {
            "model_name": model_name,
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "parameters": param_count,
            "checkpoint_size_mb": round(ckpt_size_mb, 2),
            "total_samples": len(slice_records),
            "latency_mean_ms": round(float(np.mean(latencies)), 2),
            "latency_std_ms": round(float(np.std(latencies)), 2),
            "latency_min_ms": round(float(np.min(latencies)), 2),
            "latency_max_ms": round(float(np.max(latencies)), 2),
            "threshold": threshold,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "device": str(device)
        }
        with open(predictions_dir / "predictions_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest_meta, f, indent=2, ensure_ascii=False)

    avg_latency = float(np.mean(latencies))
    print(f"   ✓ Hoàn tất {len(slice_records)} lát cắt | Latency TB: {avg_latency:.2f} ms/slice | Params: {param_count/1e6:.2f}M")
    return {
        "model_name": model_name,
        "parameters": param_count,
        "checkpoint_size_mb": ckpt_size_mb,
        "latency_mean_ms": avg_latency,
    }


def find_checkpoint(candidates: List[str]) -> Optional[str]:
    """Tìm checkpoint đầu tiên tồn tại trong danh sách ứng viên."""
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def main():
    parser = argparse.ArgumentParser(description="Chạy lại EVAL-02 cho 3 mô hình trên Test Set")
    parser.add_argument("--device", default=None, help="cpu hoặc cuda")
    parser.add_argument("--manifest", default="data/processed/test_manifest.json", help="Đường dẫn test manifest")
    parser.add_argument("--raw_data_dir", default=None, help="Đường dẫn raw CT scans (nếu cần)")
    parser.add_argument("--no_export_raw", action="store_true", help="Không xuất file .npy predictions")
    args = parser.parse_args()

    # UTF-8 stdout cho Windows console
    if sys.stdout.encoding.lower() not in ["utf-8", "utf8"]:
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("   THỰC THI TASK EVAL-02: INFERENCE & COMPREHENSIVE EVALUATION TRÊN TEST SET")
    print("=" * 80)
    print(f">> Thiết bị tính toán : {device}")

    # Danh sách 3 mô hình cần đánh giá
    models_to_eval = [
        {
            "name": "AttentionUNet",
            "display": "Attention U-Net (Proposed)",
            "checkpoints": [
                "checkpoints/EXP-ATTUNET-001_best.pth",
                "/kaggle/working/lung-lesion-segmentation/checkpoints/EXP-ATTUNET-001_best.pth",
                "/kaggle/working/EXP-ATTUNET-001_best.pth",
            ],
            "report_dir": Path("reports/attunet"),
            "pred_dir": Path("predictions/att_unet"),
        },
        {
            "name": "UNet",
            "display": "Standard U-Net (Baseline)",
            "checkpoints": [
                "checkpoints/EXP-UNET-001_best.pth",
                "/kaggle/working/lung-lesion-segmentation/checkpoints/EXP-UNET-001_best.pth",
                "/kaggle/working/EXP-UNET-001_best.pth",
            ],
            "report_dir": Path("reports/unet"),
            "pred_dir": Path("predictions/unet"),
        },
        {
            "name": "SegNet",
            "display": "SegNet (Baseline)",
            "checkpoints": [
                "checkpoints/EXP-SEGNET-002_best.pth",
                "/kaggle/working/lung-lesion-segmentation/checkpoints/EXP-SEGNET-002_best.pth",
                "/kaggle/working/EXP-SEGNET-002_best.pth",
                "checkpoints/EXP-SEGNET-001_best.pth",
                "/kaggle/working/lung-lesion-segmentation/checkpoints/EXP-SEGNET-001_best.pth",
            ],
            "report_dir": Path("reports/segnet"),
            "pred_dir": Path("predictions/segnet"),
        },
    ]

    # Kiểm tra sự tồn tại của DataLoader
    test_loader = get_eval_dataloader(
        manifest_path=args.manifest,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        raw_data_dir=args.raw_data_dir,
        enforce_readonly=False,
    )
    print(f">> Khởi tạo Test DataLoader thành công: {len(test_loader)} lát cắt.")

    perf_summary = []
    # Chạy đánh giá từng mô hình
    for m in models_to_eval:
        ckpt_path = find_checkpoint(m["checkpoints"])
        if not ckpt_path:
            print(f"\n[⚠️ CẢNH BÁO] Không tìm thấy checkpoint cho {m['name']}! Bỏ qua mô hình này.")
            continue

        res = evaluate_single_model(
            model_name=m["name"],
            checkpoint_path=ckpt_path,
            device=device,
            test_loader=test_loader,
            reports_dir=m["report_dir"],
            predictions_dir=m["pred_dir"],
            export_predictions=not args.no_export_raw,
        )
        perf_summary.append(res)

    # Chạy so sánh mô hình tổng hợp (compare_models)
    print("\n" + "=" * 80)
    print(">> ĐANG TỔNG HỢP SO SÁNH VÀ XUẤT BẢNG KẾT QUẢ CUỐI CÙNG...")
    print("=" * 80)
    generate_comparison_report(reports_dir="reports", output_csv="reports/model_comparison_summary.csv")

    print("\n✓ ĐÃ HOÀN TẤT THỰC THI TASK EVAL-02 CHO TẤT CẢ CÁC MÔ HÌNH!")


if __name__ == "__main__":
    main()

