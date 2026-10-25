"""
Task ID: DEMO-01 (Visualization Tools Module)
Mô tả: Các tiện ích trực quan hóa ảnh y tế: tạo lớp phủ màu (Color Overlay),
       vẽ đường bao ranh giới (Contour Boundary), bản đồ nhiệt (Heatmap)
       và biểu đồ so sánh đa mô hình.
"""

from typing import Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None


def create_overlay(
    ct_norm: np.ndarray,
    mask: np.ndarray,
    color_rgb: Tuple[int, int, int] = (255, 50, 50),
    alpha: float = 0.45,
    draw_contour: bool = True
) -> np.ndarray:
    """Tạo ảnh RGB với lớp phủ màu bán trong suốt đè lên lát cắt CT."""
    # Chuyển ảnh xám 1 kênh thành ảnh RGB 3 kênh
    ct_uint8 = (np.clip(ct_norm, 0.0, 1.0) * 255).astype(np.uint8)
    ct_rgb = np.stack([ct_uint8, ct_uint8, ct_uint8], axis=-1)

    bin_mask = (mask > 0).astype(bool)
    if not np.any(bin_mask):
        return ct_rgb

    # Tạo màu lớp phủ
    overlay = ct_rgb.copy().astype(np.float32)
    for c in range(3):
        overlay[:, :, c][bin_mask] = (
            (1.0 - alpha) * overlay[:, :, c][bin_mask] + alpha * color_rgb[c]
        )
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Vẽ đường viền ranh giới (Contour) nếu có opencv
    if draw_contour and cv2 is not None:
        mask_u8 = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 0), thickness=2)

    return overlay


def create_probability_heatmap(prob_map: np.ndarray, colormap: str = "magma") -> np.ndarray:
    """Tạo ảnh RGB Heatmap xác suất dự đoán."""
    cmap = cm.get_cmap(colormap)
    heatmap_rgba = cmap(prob_map)
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
    return heatmap_rgb


def create_quad_comparison_figure(
    ct_norm: np.ndarray,
    prob_map: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: Optional[np.ndarray] = None,
    alpha: float = 0.45,
    title: str = "Kết quả phân đoạn tổn thương CT phổi"
) -> plt.Figure:
    """Vẽ bảng 4 ô so sánh trực quan toàn diện."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    # 1. CT Gốc
    axes[0].imshow(ct_norm, cmap="gray")
    axes[0].set_title("1. Lát cắt CT (Lung Window)", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # 2. Probability Heatmap
    im_prob = axes[1].imshow(prob_map, cmap="magma", vmin=0.0, vmax=1.0)
    axes[1].set_title("2. Bản đồ xác suất (Probability Map)", fontsize=11, fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im_prob, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Ground Truth vs Pred Mask
    if gt_mask is not None and np.any(gt_mask > 0):
        # Có Ground Truth -> Hiển thị Ground Truth
        axes[2].imshow(gt_mask, cmap="Reds")
        axes[2].set_title(f"3. Ground Truth ({int(np.sum(gt_mask > 0))} px)", fontsize=11, fontweight="bold")
    else:
        # Không có GT -> Hiển thị Binary Mask dự đoán
        axes[2].imshow(pred_mask, cmap="Reds")
        axes[2].set_title(f"3. Mask nhị phân ({int(np.sum(pred_mask > 0))} px)", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    # 4. Lớp phủ Overlay
    overlay_img = create_overlay(ct_norm, pred_mask, alpha=alpha, draw_contour=True)
    axes[3].imshow(overlay_img)
    axes[3].set_title("4. Chẩn đoán Overlay (Vàng: Viền)", fontsize=11, fontweight="bold")
    axes[3].axis("off")

    plt.tight_layout()
    return fig


def create_dual_model_figure(
    ct_norm: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    name_a: str = "Attention U-Net",
    name_b: str = "Standard U-Net",
    gt_mask: Optional[np.ndarray] = None
) -> plt.Figure:
    """Tạo biểu đồ so sánh trực tiếp 2 mô hình trên cùng lát cắt."""
    n_cols = 4 if gt_mask is not None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    fig.suptitle(f"So sánh định tính: {name_a} vs {name_b}", fontsize=14, fontweight="bold", y=1.02)

    col = 0
    # 1. CT Gốc
    axes[col].imshow(ct_norm, cmap="gray")
    axes[col].set_title("Lát cắt CT gốc", fontsize=11, fontweight="bold")
    axes[col].axis("off")
    col += 1

    # 2. Ground Truth nếu có
    if gt_mask is not None:
        axes[col].imshow(ct_norm, cmap="gray")
        overlay_gt = create_overlay(ct_norm, gt_mask, color_rgb=(0, 200, 0), alpha=0.5, draw_contour=True)
        axes[col].imshow(overlay_gt)
        axes[col].set_title("Ground Truth (Xanh lá)", fontsize=11, fontweight="bold")
        axes[col].axis("off")
        col += 1

    # 3. Model A
    overlay_a = create_overlay(ct_norm, mask_a, color_rgb=(255, 50, 50), alpha=0.5, draw_contour=True)
    axes[col].imshow(overlay_a)
    px_a = int(np.sum(mask_a > 0))
    axes[col].set_title(f"{name_a} ({px_a} px)", fontsize=11, fontweight="bold")
    axes[col].axis("off")
    col += 1

    # 4. Model B
    overlay_b = create_overlay(ct_norm, mask_b, color_rgb=(50, 100, 255), alpha=0.5, draw_contour=True)
    axes[col].imshow(overlay_b)
    px_b = int(np.sum(mask_b > 0))
    axes[col].set_title(f"{name_b} ({px_b} px)", fontsize=11, fontweight="bold")
    axes[col].axis("off")

    plt.tight_layout()
    return fig
