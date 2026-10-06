import sys
import os
sys.path.insert(0, os.path.abspath("."))

import torch
from src.models.segnet import SegNet

def test_segnet_forward():
    print("[RUNNING] Bắt đầu kiểm thử SegNet...")
    batch_size = 2
    in_channels = 1
    height, width = 256, 256
    
    dummy_input = torch.randn(batch_size, in_channels, height, width)
    model = SegNet(in_channels=in_channels, num_classes=1, init_features=32)
    
    output = model(dummy_input)
    print(f"[TEST] Input shape:  {list(dummy_input.shape)}")
    print(f"[TEST] Output shape: {list(output.shape)}")
    
    assert output.shape == (batch_size, 1, height, width), "Output shape không khớp với kích thước ảnh đầu vào!"
    print("[SUCCESS] Unit test SegNet hoàn tất: Kích thước tensor hoàn toàn chính xác!")

if __name__ == "__main__":
    test_segnet_forward()