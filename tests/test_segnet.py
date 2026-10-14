import torch
from src.models.segnet import SegNet

def test_shape():
    # Giả lập input: Batch 2, 1 channel CT, kích thước 256x256
    dummy_input = torch.randn(2, 1, 256, 256)
    model = SegNet(in_channels=1, num_classes=1)
    output = model(dummy_input)
    print(f"[TEST] Input shape: {list(dummy_input.shape)}")
    print(f"[TEST] Output shape: {list(output.shape)}")
    assert output.shape == (2, 1, 256, 256), "Sai kích thước đầu ra!"
    print("[SUCCESS] Unit test SegNet hoàn tất: Kích thước tensor hoàn toàn chính xác!")

if __name__ == "__main__":
    test_shape()
