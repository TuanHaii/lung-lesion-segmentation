import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

try:
    import cv2
except ImportError:
    cv2 = None

CT_WINDOWS = {
    "lung": (-1000, 400),
    "mediastinum": (-160, 240),
    "soft_tissue": (-100, 200),
    "liver": (-20, 180),
    "bone": (-200, 1000),
    "raw": None
}

def apply_windowing(image: np.ndarray, window_name: str = "lung", custom_range=None) -> np.ndarray:
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

def generate_mock_ct_and_mask(shape=(256, 256, 40)):
    h, w, d = shape
    ct = np.full(shape, -1000.0, dtype=np.float32)
    mask = np.zeros(shape, dtype=np.uint8)

    y, x = np.ogrid[:h, :w]
    body_center = (h // 2, w // 2)
    body_radius = min(h, w) // 2 - 20
    body_mask = (x - body_center[1])**2 + (y - body_center[0])**2 <= body_radius**2

    for z in range(d):
        ct[:, :, z][body_mask] = np.random.normal(40, 15, size=(np.sum(body_mask),))

    left_lung = ((x - (w // 2 - 45))**2 / (35**2) + (y - h // 2)**2 / (65**2)) <= 1
    right_lung = ((x - (w // 2 + 45))**2 / (35**2) + (y - h // 2)**2 / (65**2)) <= 1
    for z in range(5, d - 5):
        ct[:, :, z][left_lung] = np.random.normal(-750, 40, size=(np.sum(left_lung),))
        ct[:, :, z][right_lung] = np.random.normal(-750, 40, size=(np.sum(right_lung),))

    lesion_center = (h // 2 + 15, w // 2 - 40)
    lesion_radius = 18
    lesion_spatial = ((x - lesion_center[1])**2 + (y - lesion_center[0])**2) <= lesion_radius**2
    for z in range(12, 28):
        ct[:, :, z][lesion_spatial] = np.random.normal(-250, 50, size=(np.sum(lesion_spatial),))
        mask[:, :, z][lesion_spatial] = 1

    return ct, mask

def plot_ct_mask_and_histogram(ct_slice, mask_slice, slice_idx=0, window_name="lung", alpha=0.45, save_path=None):
    ct_norm = apply_windowing(ct_slice, window_name=window_name)
    bin_mask = (mask_slice > 0).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(f"Kiểm tra CT Slice & Ground Truth Mask | Slice #{slice_idx} | Window: {window_name.upper()}", fontsize=14, fontweight="bold", y=1.03)

    im1 = axes[0].imshow(ct_norm, cmap="gray")
    axes[0].set_title("1. CT Slice (Windowed)", fontsize=11, fontweight="semibold")
    axes[0].axis("off")
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].imshow(bin_mask, cmap="Reds", interpolation="nearest")
    axes[1].set_title(f"2. GT Mask (Voxels: {int(np.sum(bin_mask))})", fontsize=11, fontweight="semibold")
    axes[1].axis("off")

    axes[2].imshow(ct_norm, cmap="gray")
    masked_data = np.ma.masked_where(bin_mask == 0, bin_mask)
    axes[2].imshow(masked_data, cmap="autumn", alpha=alpha, interpolation="nearest")

    if cv2 is not None and np.any(bin_mask > 0):
        contours, _ = cv2.findContours(bin_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cnt = cnt.squeeze(axis=1)
            if len(cnt.shape) == 2:
                axes[2].plot(cnt[:, 0], cnt[:, 1], color="#00FFCC", linewidth=1.5)
    axes[2].set_title("3. Overlay (Mask + Contour)", fontsize=11, fontweight="semibold")
    axes[2].axis("off")

    ax_hist = axes[3]
    all_pixels = ct_slice.flatten()
    lesion_pixels = ct_slice[bin_mask > 0].flatten() if np.any(bin_mask > 0) else np.array([])
    valid_pixels = all_pixels[all_pixels >= -1000]

    ax_hist.hist(valid_pixels, bins=60, color="#6c757d", alpha=0.5, density=True, label="Full Slice (HU >= -1000)")
    if len(lesion_pixels) > 0:
        ax_hist.hist(lesion_pixels, bins=35, color="#d90429", alpha=0.75, density=True, label="Tổn thương (Lesion ROI)")
        mean_val = float(np.mean(lesion_pixels))
        ax_hist.axvline(mean_val, color="#8d0801", linestyle="--", linewidth=1.5, label=f"Mean ROI: {mean_val:.1f} HU")

    ax_hist.set_title("4. Phân bố cường độ điểm ảnh (HU)", fontsize=11, fontweight="semibold")
    ax_hist.set_xlabel("Hounsfield Units (HU)")
    ax_hist.set_ylabel("Mật độ (Density)")
    ax_hist.legend(loc="upper right", fontsize=8)
    ax_hist.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[SUCCESS] Đã xuất ảnh kiểm tra tại: {save_path}")
    plt.close()

def generate_sample_grid(ct_vol, mask_vol, grid_size=(3, 3), window_name="lung", alpha=0.5, save_path=None):
    rows, cols = grid_size
    num_samples = rows * cols
    mask_sums = [np.sum(mask_vol[:, :, z] > 0) for z in range(ct_vol.shape[2])]
    active_slices = [z for z, count in enumerate(mask_sums) if count > 0]
    indices = np.linspace(active_slices[0], active_slices[-1], num_samples, dtype=int) if active_slices else np.linspace(0, ct_vol.shape[2] - 1, num_samples, dtype=int)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()

    for idx, slice_idx in enumerate(indices):
        ax = axes[idx]
        ct_slice = ct_vol[:, :, slice_idx]
        m_slice = (mask_vol[:, :, slice_idx] > 0).astype(np.uint8)
        ct_norm = apply_windowing(ct_slice, window_name=window_name)

        ax.imshow(ct_norm, cmap="gray")
        if np.any(m_slice > 0):
            masked = np.ma.masked_where(m_slice == 0, m_slice)
            ax.imshow(masked, cmap="autumn", alpha=alpha, interpolation="nearest")
            if cv2 is not None:
                contours, _ = cv2.findContours(m_slice.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    cnt = cnt.squeeze(axis=1)
                    if len(cnt.shape) == 2:
                        ax.plot(cnt[:, 0], cnt[:, 1], color="#00FFCC", linewidth=1.2)

        area = int(np.sum(m_slice > 0))
        color = "#2b9348" if area > 0 else "#6c757d"
        ax.set_title(f"Slice #{slice_idx} | Lesion: {area} px", fontsize=10, fontweight="bold", color=color)
        ax.axis("off")

    fig.suptitle(f"Grid mẫu kiểm tra CT + Ground Truth Mask ({rows}x{cols}) | Window: {window_name.upper()}", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[SUCCESS] Đã lưu Grid ảnh mẫu tại: {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--output", type=str, default="evidence/W2_DATA-11_single_inspection.png")
    args = parser.parse_args()

    ct_vol, mask_vol = generate_mock_ct_and_mask(shape=(256, 256, 40))
    if args.grid:
        generate_sample_grid(ct_vol, mask_vol, grid_size=(3, 3), save_path=args.output)
    else:
        mask_counts = [np.sum(mask_vol[:, :, z] > 0) for z in range(ct_vol.shape[2])]
        selected_slice = int(np.argmax(mask_counts))
        plot_ct_mask_and_histogram(ct_vol[:, :, selected_slice], mask_vol[:, :, selected_slice], slice_idx=selected_slice, save_path=args.output)

if __name__ == "__main__":
    main()
