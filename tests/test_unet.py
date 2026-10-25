import sys
import os
from pathlib import Path
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.unet import UNet
from src.models.attention_unet import AttentionUNet


def test_unet_shape():
    dummy_input = torch.randn(2, 1, 256, 256)
    model = UNet(in_channels=1, out_channels=1)
    output = model(dummy_input)
    print(f"[TEST UNET] Input shape: {list(dummy_input.shape)} -> Output shape: {list(output.shape)}")
    assert output.shape == (2, 1, 256, 256), "Sai kích thước đầu ra UNet!"
    print("[SUCCESS] UNet sanity check PASSED!")


def test_attention_unet_shape():
    dummy_input = torch.randn(2, 1, 256, 256)
    model = AttentionUNet(in_channels=1, out_channels=1)
    output = model(dummy_input)
    print(f"[TEST ATT-UNET] Input shape: {list(dummy_input.shape)} -> Output shape: {list(output.shape)}")
    assert output.shape == (2, 1, 256, 256), "Sai kích thước đầu ra AttentionUNet!"
    print("[SUCCESS] AttentionUNet sanity check PASSED!")


if __name__ == "__main__":
    test_unet_shape()
    test_attention_unet_shape()
    print(">> ALL UNET & ATTENTION UNET TESTS PASSED! <<")
