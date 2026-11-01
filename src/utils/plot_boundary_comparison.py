
import os
import numpy as np
import matplotlib.pyplot as plt
from visualization import apply_windowing

def generate_small_lesion_mock():
    """Tạo mock data mô phỏng một tổn thương nhỏ để đánh giá bảo toàn ranh giới."""
    shape = (128, 128)
    ct = np.full(shape, -800.0, dtype=np.float32)
    gt, unet, segnet = [np.zeros(shape, dtype=np.uint8) for _ in range(3)]
    y, x = np.ogrid[:shape[0], :shape[1]]
    
    # Ground Truth: Tổn thương nhỏ, sắc nét
    mask = (x - 64)**2 + (y - 64)**2 <= 12**2
    ct[mask] = np.random.normal(-200, 30, size=np.sum(mask))
    gt[mask] = 1
    
    # U-Net: Mất mát chi tiết biên (hơi co lại và bo tròn quá mức)
    unet_mask = (x - 64)**2 + (y - 64)**2 <= 9**2
    unet[unet_mask] = 1
    
    # SegNet: Bảo toàn biên tốt hơn nhờ Max-Unpooling indices
    segnet[mask] = 1
    return ct, gt, unet, segnet

def plot_rq2_boundary(save_path="plots/rq2_boundary_comparison.png"):
    ct, gt, unet, segnet = generate_small_lesion_mock()
    ct_norm = apply_windowing(ct, window_name="lung")
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("RQ2 Boundary Preservation: U-Net vs SegNet (Small Lesion)", 
                 fontsize=15, fontweight='bold', y=1.05)
    
    titles = ["1. Original CT (Zoom-in)", "2. Ground Truth", "3. U-Net (Detail Loss)", "4. SegNet (Sharp Boundary)"]
    masks = [None, gt, unet, segnet]
    colors = [None, "Greens", "Blues", "Reds"]
    
    for i, ax in enumerate(axes):
        ax.imshow(ct_norm, cmap="gray")
        if masks[i] is not None:
            ax.imshow(np.ma.masked_where(masks[i] == 0, masks[i]), cmap=colors[i], alpha=0.6)
            # Vẽ thêm contour để làm nổi bật ranh giới
            ax.contour(masks[i], colors='yellow', linewidths=1.5, linestyles='solid')
        ax.set_title(titles[i], fontsize=12, fontweight='semibold')
        ax.axis("off")
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Đã xuất ảnh phân tích RQ2 tại: {save_path}")

if __name__ == "__main__":
    plot_rq2_boundary()
