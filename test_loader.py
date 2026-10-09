import time
import torch
from src.data.dataset import get_dataloader


def test_loader_performance():
    print("=" * 65)
    print("BẮT ĐẦU KIỂM THỬ TOÀN DIỆN PYTORCH DATALOADER (DATA-10)")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Thiết bị kiểm thử: {device}")

    # 1. Khởi tạo 3 loader
    train_loader = get_dataloader(split="train", batch_size=16, num_workers=2, shuffle=True)
    val_loader = get_dataloader(split="val", batch_size=16, num_workers=2, shuffle=False)
    test_loader = get_dataloader(split="test", batch_size=1, num_workers=1, shuffle=False)

    print(f"-> Train loader: {len(train_loader.dataset)} slices | {len(train_loader)} batches (batch_size=16)")
    print(f"-> Val loader:   {len(val_loader.dataset)} slices | {len(val_loader)} batches (batch_size=16)")
    print(f"-> Test loader:  {len(test_loader.dataset)} slices | {len(test_loader)} batches (batch_size=1)")

    # 2. Kiểm tra shape và dải giá trị trên 1 batch mẫu của tập Train
    print("\n[1] Kiểm tra kích thước tensor và tính toàn vẹn dữ liệu...")
    images, masks, meta = next(iter(train_loader))

    assert images.shape == (16, 1, 256, 256), f"Sai kích thước batch ảnh: {images.shape}"
    assert masks.shape == (16, 1, 256, 256), f"Sai kích thước batch mask: {masks.shape}"
    assert images.dtype == torch.float32, f"Sai kiểu dữ liệu ảnh: {images.dtype}"
    assert masks.dtype == torch.float32, f"Sai kiểu dữ liệu mask: {masks.dtype}"

    # Kiểm tra dải giá trị Min-Max [0, 1] và mask nhị phân {0.0, 1.0}
    img_min, img_max = float(images.min()), float(images.max())
    mask_unique = torch.unique(masks).tolist()

    assert 0.0 <= img_min and img_max <= 1.0, f"Ảnh vi phạm dải [0, 1]: min={img_min}, max={img_max}"
    assert set(mask_unique).issubset({0.0, 1.0}), f"Mask vi phạm nhãn nhị phân: {mask_unique}"

    print(f"  + Image Tensor shape: {list(images.shape)} | Range: [{img_min:.3f}, {img_max:.3f}] -> ĐẠT")
    print(f"  + Mask Tensor shape:  {list(masks.shape)} | Unique labels: {mask_unique} -> ĐẠT")
    print(f"  + Metadata sample: ID={meta['sample_id'][0]}, Patient={meta['patient_id'][0]} -> ĐẠT")

    # 3. Đo tốc độ tải batch trên thiết bị (Throughput Benchmark)
    print("\n[2] Đo tốc độ tải batch (Throughput Benchmark trên 1 epoch Train)...")
    start_time = time.time()
    total_samples = 0

    for i, (imgs, msks, _) in enumerate(train_loader):
        if torch.cuda.is_available():
            imgs = imgs.to(device, non_blocking=True)
            msks = msks.to(device, non_blocking=True)
        total_samples += imgs.size(0)

    elapsed_time = time.time() - start_time
    throughput = total_samples / elapsed_time

    print(f"  + Tổng số lát cắt đã nạp: {total_samples} slices")
    print(f"  + Tổng thời gian thực thi: {elapsed_time:.2f} giây")
    print(f"  + Tốc độ nạp dữ liệu:      {throughput:.1f} slices/giây")

    print("\n" + "=" * 65)
    print("=> [DATA-10] PyTorch Dataset & DataLoader PASSED 100%!")
    print("=" * 65)


if __name__ == "__main__":
    test_loader_performance()