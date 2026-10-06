import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models.segnet import SegNet
from src.utils.visualization import generate_mock_ct_and_mask, apply_windowing

def run_demo():
    print("[1/4] Đang khởi tạo dữ liệu CT lát cắt mẫu...")
    ct_vol, mask_vol = generate_mock_ct_and_mask(shape=(256, 256, 40))
    
    # Chọn slice có tổn thương rõ nhất
    mask_counts = [np.sum(mask_vol[:, :, z] > 0) for z in range(ct_vol.shape[2])]
    slice_idx = int(np.argmax(mask_counts))
    ct_slice = ct_vol[:, :, slice_idx]
    gt_slice = mask_vol[:, :, slice_idx]

    # Preprocessing: Windowing [-1000, 400] HU -> [0, 1]
    ct_norm = apply_windowing(ct_slice, window_name="lung")

    print("[2/4] Đang nạp mô hình SegNet...")
    model = SegNet(in_channels=1, num_classes=1, init_features=32)
    model.eval()

    # Chuẩn bị Tensor: (1, 1, 256, 256)
    input_tensor = torch.from_numpy(ct_norm).unsqueeze(0).unsqueeze(0)

    print("[3/4] Đang thực hiện Forward Pass (Inference)...")
    with torch.no_grad():
        logits = model(input_tensor)
        pred_prob = torch.sigmoid(logits).squeeze().numpy()
        pred_mask = (pred_prob > 0.5).astype(np.uint8)

    print("[4/4] Đang trực quan hóa kết quả và lưu ảnh...")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"Chạy thử nghiệm SegNet Baseline | Slice #{slice_idx}", fontsize=14, fontweight="bold")

    axes[0].imshow(ct_norm, cmap="gray")
    axes[0].set_title("1. Lát cắt CT (Lung Window)")
    axes[0].axis("off")

    axes[1].imshow(gt_slice, cmap="Reds")
    axes[1].set_title(f"2. Ground Truth Mask ({int(np.sum(gt_slice))} px)")
    axes[1].axis("off")

    im_prob = axes[2].imshow(pred_prob, cmap="magma")
    axes[2].set_title("3. SegNet Probability Map")
    axes[2].axis("off")
    plt.colorbar(im_prob, ax=axes[2], fraction=0.046, pad=0.04)

    # Hiển thị Overlay: CT + Dự đoán nhị phân
    axes[3].imshow(ct_norm, cmap="gray")
    masked_pred = np.ma.masked_where(pred_mask == 0, pred_mask)
    axes[3].imshow(masked_pred, cmap="cool", alpha=0.5)
    axes[3].set_title("4. SegNet Pred Overlay (Binary)")
    axes[3].axis("off")

    plt.tight_layout()
    out_path = "evidence/demo_segnet_prediction.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Đã lưu ảnh kết quả tại: {out_path}")

if __name__ == "__main__":
    run_demo()
