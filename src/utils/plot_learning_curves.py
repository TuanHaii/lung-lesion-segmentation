
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def generate_mock_logs(epochs=50):
    """Tạo dữ liệu huấn luyện giả lập cho 3 mô hình để dựng trước khung đồ thị"""
    np.random.seed(42)
    epochs_range = np.arange(1, epochs + 1)
    data = {}
    
    # 1. U-Net (Baseline)
    data['U-Net'] = {
        'train_loss': 0.6 * np.exp(-0.1 * epochs_range) + 0.15 + np.random.normal(0, 0.01, epochs),
        'val_loss': 0.6 * np.exp(-0.08 * epochs_range) + 0.20 + np.random.normal(0, 0.015, epochs),
        'val_dice': 0.76 - 0.6 * np.exp(-0.12 * epochs_range) + np.random.normal(0, 0.01, epochs)
    }
    
    # 2. SegNet (Comparative Model RQ2)
    data['SegNet'] = {
        'train_loss': 0.6 * np.exp(-0.08 * epochs_range) + 0.18 + np.random.normal(0, 0.01, epochs),
        'val_loss': 0.6 * np.exp(-0.07 * epochs_range) + 0.23 + np.random.normal(0, 0.015, epochs),
        'val_dice': 0.72 - 0.6 * np.exp(-0.1 * epochs_range) + np.random.normal(0, 0.01, epochs)
    }
    
    # 3. Attention U-Net (Proposed Model RQ1)
    data['Attention U-Net'] = {
        'train_loss': 0.6 * np.exp(-0.12 * epochs_range) + 0.10 + np.random.normal(0, 0.01, epochs),
        'val_loss': 0.6 * np.exp(-0.1 * epochs_range) + 0.15 + np.random.normal(0, 0.015, epochs),
        'val_dice': 0.82 - 0.6 * np.exp(-0.15 * epochs_range) + np.random.normal(0, 0.01, epochs)
    }
    return epochs_range, data

def smooth_curve(points, factor=0.75):
    """Làm mượt đường cong để dễ quan sát xu hướng hội tụ"""
    smoothed = []
    for point in points:
        if smoothed:
            smoothed.append(smoothed[-1] * factor + point * (1 - factor))
        else:
            smoothed.append(point)
    return np.array(smoothed)

def plot_learning_curves(save_path="plots/learning_curves_comparison.png"):
    epochs, data = generate_mock_logs()
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Động học học tập: U-Net vs SegNet vs Attention U-Net", fontsize=16, fontweight='bold', y=1.05)
    
    colors = {'U-Net': '#1f77b4', 'SegNet': '#ff7f0e', 'Attention U-Net': '#2ca02c'}
    
    # Panel 1: Training Loss
    for model in data:
        axes[0].plot(epochs, smooth_curve(data[model]['train_loss']), label=model, color=colors[model], linewidth=2.5)
        axes[0].plot(epochs, data[model]['train_loss'], color=colors[model], alpha=0.2)
    axes[0].set_title("1. Training Loss (BCE + Dice)", fontsize=13, fontweight='semibold')
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="upper right")
    
    # Panel 2: Validation Loss
    for model in data:
        axes[1].plot(epochs, smooth_curve(data[model]['val_loss']), label=model, color=colors[model], linewidth=2.5, linestyle='--')
        axes[1].plot(epochs, data[model]['val_loss'], color=colors[model], alpha=0.2, linestyle='--')
    axes[1].set_title("2. Validation Loss", fontsize=13, fontweight='semibold')
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="upper right")

    # Panel 3: Validation Dice Score
    for model in data:
        axes[2].plot(epochs, smooth_curve(data[model]['val_dice']), label=model, color=colors[model], linewidth=2.5)
        axes[2].plot(epochs, data[model]['val_dice'], color=colors[model], alpha=0.2)
    axes[2].set_title("3. Validation Dice Score", fontsize=13, fontweight='semibold')
    axes[2].set_xlabel("Epochs")
    axes[2].set_ylabel("Dice Coefficient")
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend(loc="lower right")
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Đã xuất biểu đồ so sánh Learning Curves tại: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="plots/learning_curves_comparison.png")
    args = parser.parse_args()
    plot_learning_curves(args.output)