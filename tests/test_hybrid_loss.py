import sys
import os
from pathlib import Path
import torch
try:
    import pytest
except ImportError:
    pytest = None

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.losses.hybrid_loss import HybridLoss




def test_hybrid_loss_gradient_flow():
    """Kiểm tra forward & backward pass, đảm bảo gradient chảy mượt mà và không NaN/Inf."""
    criterion = HybridLoss(bce_weight=0.5, dice_weight=0.5, smooth=1e-5)
    b, c, h, w = 4, 1, 256, 256

    logits = torch.randn((b, c, h, w), requires_grad=True)
    targets = torch.randint(0, 2, (b, c, h, w)).long()  # Giả lập targets dạng long tensor

    loss = criterion(logits, targets)
    loss.backward()

    assert not torch.isnan(loss), "Lỗi: Hybrid Loss trả về NaN!"
    assert not torch.isinf(loss), "Lỗi: Hybrid Loss trả về Inf!"
    assert logits.grad is not None, "Lỗi: Gradient không truyền về logits!"
    assert loss.item() > 0.0, "Lỗi: Giá trị Loss phải là số thực dương!"


def test_hybrid_loss_weight_scaling():
    """Kiểm tra xem các trọng số alpha, beta có tác động đúng tỷ lệ hay không."""
    criterion_pure_bce = HybridLoss(bce_weight=1.0, dice_weight=0.0)
    criterion_pure_dice = HybridLoss(bce_weight=0.0, dice_weight=1.0)
    criterion_hybrid = HybridLoss(bce_weight=0.5, dice_weight=0.5)

    b, c, h, w = 2, 1, 64, 64
    logits = torch.randn((b, c, h, w))
    targets = torch.randint(0, 2, (b, c, h, w)).float()

    l_bce = criterion_pure_bce(logits, targets)
    l_dice = criterion_pure_dice(logits, targets)
    l_hybrid = criterion_hybrid(logits, targets)

    expected_loss = 0.5 * l_bce + 0.5 * l_dice
    assert torch.isclose(l_hybrid, expected_loss, atol=1e-5), "Lỗi: Tính toán kết hợp tuyến tính bị sai!"


if __name__ == "__main__":
    test_hybrid_loss_gradient_flow()
    test_hybrid_loss_weight_scaling()
    print(">> ALL HYBRID LOSS TESTS PASSED SUCCESSFULLY! <<")