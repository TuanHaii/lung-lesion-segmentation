import torch
import torch.optim as optim
import sys
import os

# Thêm thư mục gốc vào path để import các module trong src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.unet import UNet
from src.losses.hybrid_loss import HybridLoss
from src.utils.metrics import compute_metrics

def test_pipeline_sanity_check():
    print("=" * 60)
    print("SANITY CHECK PIPELINE: Forward -> Loss -> Backward -> Metric")
    print("=" * 60)

    # 1. Khởi tạo Dummy Data (Batch=2, Channel=1, H=256, W=256)
    batch_size = 2
    dummy_input = torch.randn(batch_size, 1, 256, 256)
    # Ground truth mask binary (0 hoặc 1)
    dummy_target = torch.randint(0, 2, (batch_size, 1, 256, 256)).float()

    print(f"[1/5] Dummy Input Shape : {dummy_input.shape}")
    print(f"[1/5] Dummy Target Shape: {dummy_target.shape}")

    # 2. Khởi tạo Model, Loss Function, Optimizer
    model = UNet(in_channels=1, out_channels=1)
    criterion = HybridLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 3. Step 1: Forward Pass
    model.train()
    logits = model(dummy_input)
    print(f"[2/5] Forward Pass OK   : Output Logits Shape = {logits.shape}")

    # 4. Step 2: Loss Computation & Backward Pass (Gradient Update)
    loss = criterion(logits, dummy_target)
    print(f"[3/5] Loss Computed     : Hybrid Loss Value = {loss.item():.4f}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("[4/5] Backward Pass OK  : Gradients calculated & Optimizer updated")

    # 5. Step 3: Compute Evaluation Metrics
    model.eval()
    with torch.no_grad():
        preds = model(dummy_input)
        metrics = compute_metrics(preds, dummy_target, threshold=0.5)

    print("[5/5] Metrics Evaluated :")
    for k, v in metrics.items():
        print(f"      - {k.upper():<10}: {v:.4f}")

    print("=" * 60)
    print(">> TEST PIPELINE PASSED SUCCESSFULLY! ALL SYSTEMS GO <<")
    print("=" * 60)

if __name__ == "__main__":
    test_pipeline_sanity_check()