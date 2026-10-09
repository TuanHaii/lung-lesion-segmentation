import torch
import pytest
from src.losses.dice_loss import SoftDiceLoss


def test_soft_dice_loss_gradient_flow():
    """Kiểm tra forward/backward pass, đảm bảo gradient không bị NaN/Inf (RSK-03)."""
    criterion = SoftDiceLoss(smooth=1e-5)
    batch_size, channels, height, width = 4, 1, 256, 256

    logits = torch.randn((batch_size, channels, height, width), requires_grad=True)
    targets = torch.randint(0, 2, (batch_size, channels, height, width)).float()

    loss = criterion(logits, targets)
    loss.backward()

    assert not torch.isnan(loss), "Lỗi: Giá trị Loss trả về NaN!"
    assert not torch.isinf(loss), "Lỗi: Giá trị Loss trả về Inf!"
    assert logits.grad is not None, "Lỗi: Gradient không truyền ngược về Logits!"
    assert not torch.isnan(logits.grad).any(), "Lỗi: Gradient chứa giá trị NaN!"


def test_soft_dice_loss_perfect_prediction():
    """Kiểm tra trường hợp dự đoán hoàn hảo: Loss phải tiệm cận 0.0."""
    criterion = SoftDiceLoss(smooth=1e-5)
    batch_size, channels, height, width = 2, 1, 256, 256

    targets = torch.zeros((batch_size, channels, height, width)).float()
    targets[:, :, 50:150, 50:150] = 1.0  # Tạo tổn thương giả lập

    # Logits cực lớn cho vùng 1 và cực âm cho vùng 0
    logits = torch.where(targets == 1.0, torch.tensor(15.0), torch.tensor(-15.0))

    loss = criterion(logits, targets)
    assert loss.item() < 0.01, f"Lỗi: Loss dự đoán hoàn hảo phải tiệm cận 0, nhận được: {loss.item()}"


def test_soft_dice_loss_worst_prediction():
    """Kiểm tra trường hợp dự đoán nghịch đảo hoàn toàn: Loss phải tiệm cận 1.0."""
    criterion = SoftDiceLoss(smooth=1e-5)
    batch_size, channels, height, width = 2, 1, 256, 256

    targets = torch.zeros((batch_size, channels, height, width)).float()
    targets[:, :, 50:150, 50:150] = 1.0

    # Dự đoán ngược hoàn toàn với Ground Truth
    logits = torch.where(targets == 1.0, torch.tensor(-15.0), torch.tensor(15.0))

    loss = criterion(logits, targets)
    assert loss.item() > 0.95, f"Lỗi: Loss dự đoán sai hoàn toàn phải tiệm cận 1, nhận được: {loss.item()}"


def test_soft_dice_loss_empty_slices_edge_case():
    """
    Kiểm tra xử lý lát cắt rỗng không chứa tổn thương (GT = 0).
    Theo đặc tả: GT = 0 và Model đoán đúng (Pred ~ 0) thì Dice Score = 1.0 -> Loss = 0.0.
    """
    criterion = SoftDiceLoss(smooth=1e-5)
    batch_size, channels, height, width = 2, 1, 256, 256

    empty_targets = torch.zeros((batch_size, channels, height, width)).float()
    correct_empty_logits = torch.full((batch_size, channels, height, width), -15.0)

    loss = criterion(correct_empty_logits, empty_targets)
    assert loss.item() < 0.01, f"Lỗi: Lát cắt rỗng dự đoán đúng phải có loss gần 0, nhận được: {loss.item()}"