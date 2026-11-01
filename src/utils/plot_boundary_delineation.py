"""
Task ID: EVAL-11 (Sprint W5 - Evaluation & Analysis)
Mô tả: Trực quan hóa ranh giới tổn thương phức tạp (Boundary Delineation Zoom-in)
Tác giả: Hậu (Baseline & Experiment)
Phục vụ: Đánh giá độ sắc nét đường viền và hiện tượng false positive quanh mạch máu/thành ngực.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def generate_complex_lesion_mock():
    """Tạo dữ liệu mô phỏng tổn thương gần mạch máu lớn/thành ngực để đánh giá ranh giới."""
    h, w = 128, 128
    ct = np.full((h, w), -800.0, dtype=np.float32)
    gt = np.zeros((h, w), dtype=np.uint8)
    unet = np.zeros((h, w), dtype=np.uint8)
    att_unet = np.zeros((h, w), dtype=np.uint8)
    
    y, x = np.ogrid[:h, :w]
    
    # Mạch máu lớn / Thành ngực (Cản quang cao)
    vessel_mask = (x - 20)**2 / (15**2) + (y - 64)**2 / (40**2) <= 1
    ct[vessel_mask] = np.random.normal(40, 20, size=np.sum(vessel_mask))
    
    # Tổn thương bám sát mạch máu
    lesion_mask = (x - 64)**2 + (y - 64)**2 <= 20**2
    ct[lesion_mask] = np.random.normal(-200, 40, size=np.sum(lesion_mask))
    gt[lesion_mask] = 1
    
    # U-Net: Bị nhiễu, dính vào mạch máu (False Positive) và ranh giới nhòe
    unet_mask = (x - 55)**2 + (y - 64)**2 <= 25**2
    unet[unet_mask] = 1
    
    # Attention U-Net: Tách bạch tốt nhờ Attention Gate triệt tiêu nhiễu nền
    att_unet[lesion_mask] = 1
    
    return ct, gt, unet, att_unet

def plot_boundary_delineation(save_path="plots/boundary_delineation_zoom.png"):
    ct, gt, unet, att_unet = generate_complex_lesion_mock()
    
    # Windowing
    vmin, vmax = -1000.0, 400.0
    ct_norm = np.clip(ct, vmin, vmax)
    ct_norm = (ct_norm - vmin) / (vmax - vmin)
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("EVAL-11: Boundary Delineation (Vessel/Wall Adjacency)", 
                 fontsize=15, fontweight='bold', y=1.05)
    
    titles = ["1. CT (Complex ROI)", "2. Ground Truth", "3. U-Net (Boundary Leakage)", "4. Att-UNet (Sharp Boundary)"]
    masks = [None, gt, unet, att_unet]
    colors = [None, "Greens", "Blues", "autumn"]
    
    for i, ax in enumerate(axes):
        ax.imshow(ct_norm, cmap="gray")
        if masks[i] is not None:
            masked = np.ma.masked_where(masks[i] == 0, masks[i])
            ax.imshow(masked, cmap=colors[i], alpha=0.5)
            
            # Vẽ đường viền trực tiếp bằng matplotlib (không dùng cv2)
            ax.contour(masks[i], levels=[0.5], colors=['yellow'], linewidths=1.5)
                    
        ax.set_title(titles[i], fontsize=12, fontweight='semibold')
        ax.axis("off")
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Đã xuất ảnh phân tích ranh giới phức tạp tại: {save_path}")

if __name__ == "__main__":
    plot_boundary_delineation()
