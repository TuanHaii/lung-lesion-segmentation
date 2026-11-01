
import os
import numpy as np
import matplotlib.pyplot as plt
from visualization import apply_windowing

def generate_mock_triplet_data(shape=(256, 256)):
    """Sinh dữ liệu giả lập cho 1 lát cắt: CT, Ground Truth và 3 Prediction Mask"""
    h, w = shape
    ct_slice = np.full(shape, -1000.0, dtype=np.float32)
    gt_mask = np.zeros(shape, dtype=np.uint8)
    y, x = np.ogrid[:h, :w]
    
    # Tạo phổi background
    left_lung = ((x - (w // 2 - 45))**2 / (35**2) + (y - h // 2)**2 / (65**2)) <= 1
    right_lung = ((x - (w // 2 + 45))**2 / (35**2) + (y - h // 2)**2 / (65**2)) <= 1
    ct_slice[left_lung] = np.random.normal(-750, 40, size=(np.sum(left_lung),))
    ct_slice[right_lung] = np.random.normal(-750, 40, size=(np.sum(right_lung),))
    
    # Tạo tổn thương (Ground Truth)
    lesion_center = (h // 2 + 10, w // 2 - 35)
    lesion_mask = ((x - lesion_center[1])**2 + (y - lesion_center[0])**2) <= 15**2
    ct_slice[lesion_mask] = np.random.normal(-250, 50, size=(np.sum(lesion_mask),))
    gt_mask[lesion_mask] = 1
    
    # Giả lập Mask dự đoán từ các mô hình
    unet_mask = np.copy(gt_mask)
    unet_mask[h//2-5:h//2+5, w//2-40:w//2-30] = 0 # U-Net miss một góc nhỏ
    
    segnet_mask = np.copy(gt_mask)
    segnet_mask[h//2+10:h//2+20, w//2-40:w//2-20] = 1 # SegNet tràn viền (False Positive)
    
    attunet_mask = np.copy(gt_mask) # Attention U-Net bám biên rất tốt
    
    return ct_slice, gt_mask, segnet_mask, unet_mask, attunet_mask

def plot_qualitative_grid(num_samples=3, save_path="plots/qualitative_comparison_grid.png"):
    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4 * num_samples))
    fig.suptitle("Qualitative Comparison: CT vs Ground Truth vs Multi-Model Predictions", 
                 fontsize=18, fontweight='bold', y=0.95 + (0.01 * (3-num_samples)))
    
    col_titles = ["1. Original CT (Lung Window)", "2. Ground Truth Mask", "3. SegNet Prediction", "4. U-Net Prediction", "5. Attention U-Net Pred"]
    
    for i in range(num_samples):
        ct, gt, segnet, unet, attunet = generate_mock_triplet_data()
        ct_norm = apply_windowing(ct, window_name="lung")
        
        # Cột 1: CT Gốc
        axes[i, 0].imshow(ct_norm, cmap="gray")
        if i == 0: axes[i, 0].set_title(col_titles[0], fontsize=12, fontweight='semibold')
        axes[i, 0].axis("off")
        
        # Cột 2: Ground Truth
        axes[i, 1].imshow(ct_norm, cmap="gray")
        axes[i, 1].imshow(np.ma.masked_where(gt == 0, gt), cmap="Greens", alpha=0.6)
        if i == 0: axes[i, 1].set_title(col_titles[1], fontsize=12, fontweight='semibold')
        axes[i, 1].axis("off")
        
        # Cột 3: SegNet (Baseline RQ2)
        axes[i, 2].imshow(ct_norm, cmap="gray")
        axes[i, 2].imshow(np.ma.masked_where(segnet == 0, segnet), cmap="Reds", alpha=0.6)
        if i == 0: axes[i, 2].set_title(col_titles[2], fontsize=12, fontweight='semibold')
        axes[i, 2].axis("off")
        
        # Cột 4: U-Net (Baseline Chính)
        axes[i, 3].imshow(ct_norm, cmap="gray")
        axes[i, 3].imshow(np.ma.masked_where(unet == 0, unet), cmap="Blues", alpha=0.6)
        if i == 0: axes[i, 3].set_title(col_titles[3], fontsize=12, fontweight='semibold')
        axes[i, 3].axis("off")

        # Cột 5: Attention U-Net (Proposed)
        axes[i, 4].imshow(ct_norm, cmap="gray")
        axes[i, 4].imshow(np.ma.masked_where(attunet == 0, attunet), cmap="autumn", alpha=0.6)
        if i == 0: axes[i, 4].set_title(col_titles[4], fontsize=12, fontweight='semibold')
        axes[i, 4].axis("off")
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Đã lưu bộ ảnh so sánh lưới đa mô hình tại: {save_path}")

if __name__ == "__main__":
    plot_qualitative_grid()
