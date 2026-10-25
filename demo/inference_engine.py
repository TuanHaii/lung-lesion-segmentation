"""
Task ID: DEMO-01 (Inference Engine Module)
Mô tả: Engine nạp mô hình, tiền xử lý lát cắt CT phổi, suy luận Forward Pass
       và tính toán các thông số định lượng lâm sàng (Lesion Area, Burden %).
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.models.unet import UNet
from src.models.attention_unet import AttentionUNet
from src.models.segnet import SegNet

CT_WINDOWS = {
    "lung": (-1000, 400),
    "mediastinum": (-160, 240),
    "soft_tissue": (-100, 200),
    "bone": (-200, 1000),
    "raw": None
}


def apply_windowing(image: np.ndarray, window_name: str = "lung", custom_range: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """Áp dụng kỹ thuật Cửa sổ Hounsfield Unit (HU Windowing) và chuẩn hóa về [0, 1]."""
    if custom_range is not None:
        vmin, vmax = custom_range
    elif window_name in CT_WINDOWS and CT_WINDOWS[window_name] is not None:
        vmin, vmax = CT_WINDOWS[window_name]
    else:
        vmin, vmax = float(np.min(image)), float(np.max(image))

    windowed = np.clip(image, vmin, vmax)
    if vmax > vmin:
        windowed = (windowed - vmin) / (vmax - vmin)
    else:
        windowed = np.zeros_like(windowed, dtype=np.float32)
    return windowed.astype(np.float32)


def load_model(
    model_name: str = "AttentionUNet",
    checkpoint_path: Optional[str] = None,
    device: str = "cpu"
) -> Tuple[nn.Module, bool]:
    """Khởi tạo kiến trúc mô hình và nạp trọng số đã huấn luyện (nếu có).
    
    Returns:
        (model, has_checkpoint): Mô hình PyTorch và cờ kiểm tra có checkpoint thật hay không.
    """
    dev = torch.device(device)

    # 1. Khởi tạo kiến trúc
    if model_name in ["AttentionUNet", "Attention U-Net (Đề xuất)"]:
        model = AttentionUNet(in_channels=1, out_channels=1, features=[64, 128, 256, 512])
    elif model_name in ["UNet", "Standard U-Net (Baseline)"]:
        model = UNet(in_channels=1, out_channels=1, features=[64, 128, 256, 512])
    elif model_name in ["SegNet", "SegNet (Baseline)"]:
        model = SegNet(in_channels=1, num_classes=1, init_features=32)
    else:
        raise ValueError(f"Không hỗ trợ mô hình: {model_name}")

    has_checkpoint = False
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location=dev)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            else:
                state_dict = ckpt
            model.load_state_dict(state_dict)
            has_checkpoint = True
        except Exception as e:
            print(f"[!] Cảnh báo nạp checkpoint: {e}")

    model.to(dev)
    model.eval()
    return model, has_checkpoint


def preprocess_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (256, 256),
    window_name: str = "lung",
    custom_range: Optional[Tuple[float, float]] = None
) -> Tuple[torch.Tensor, np.ndarray]:
    """Tiền xử lý ảnh CT 2D: HU Windowing -> Resize -> Tensor (1, 1, H, W)."""
    # 1. Windowing
    ct_norm = apply_windowing(image, window_name=window_name, custom_range=custom_range)

    # 2. Resize về target_size (256, 256)
    if ct_norm.shape != target_size:
        pil_img = Image.fromarray((ct_norm * 255).astype(np.uint8))
        pil_resized = pil_img.resize(target_size, Image.Resampling.BILINEAR)
        ct_resized = np.array(pil_resized, dtype=np.float32) / 255.0
    else:
        ct_resized = ct_norm

    # 3. Chuyển thành PyTorch Tensor: [1, 1, H, W]
    tensor = torch.from_numpy(ct_resized).unsqueeze(0).unsqueeze(0).float()
    return tensor, ct_resized


@torch.no_grad()
def run_inference(
    model: nn.Module,
    input_tensor: torch.Tensor,
    threshold: float = 0.5,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Chạy suy luận forward pass và trả về heatmap xác suất, mask nhị phân và thời gian suy luận."""
    dev = torch.device(device)
    tensor = input_tensor.to(dev)

    start_t = time.perf_counter()
    logits = model(tensor)
    if torch.cuda.is_available() and dev.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start_t) * 1000.0

    # Chuyển qua Sigmoid để lấy xác suất [0, 1]
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    pred_mask = (probs >= threshold).astype(np.uint8)

    return {
        "probability_map": probs,
        "binary_mask": pred_mask,
        "latency_ms": latency_ms,
    }


def calculate_clinical_metrics(
    pred_mask: np.ndarray,
    ct_norm: np.ndarray,
    spacing: Tuple[float, float] = (1.0, 1.0)
) -> Dict[str, Any]:
    """Tính toán các chỉ số định lượng phục vụ chẩn đoán lâm sàng."""
    lesion_pixels = int(np.sum(pred_mask > 0))
    total_pixels = int(pred_mask.size)

    # Ước tính diện tích (cm2) nếu có pixel spacing (mm/pixel)
    pixel_area_cm2 = (spacing[0] * spacing[1]) / 100.0
    lesion_area_cm2 = lesion_pixels * pixel_area_cm2

    # Ước lượng vùng nhu mô phổi (vùng tối trong lung window)
    lung_approx = np.logical_and(ct_norm > 0.05, ct_norm < 0.65)
    lung_pixels = int(np.sum(lung_approx)) or (total_pixels // 3)

    burden_percent = min(100.0, (lesion_pixels / max(lung_pixels, 1)) * 100.0)

    # Phân loại mức độ tổn thương theo tiêu chuẩn y khoa
    if burden_percent < 5.0:
        severity = "Nhẹ (Mild)"
        severity_color = "green"
    elif burden_percent < 25.0:
        severity = "Trung bình (Moderate)"
        severity_color = "orange"
    else:
        severity = "Lan tỏa / Nặng (Severe)"
        severity_color = "red"

    return {
        "lesion_pixels": lesion_pixels,
        "lesion_area_cm2": lesion_area_cm2,
        "lung_burden_percent": burden_percent,
        "severity": severity,
        "severity_color": severity_color,
    }
