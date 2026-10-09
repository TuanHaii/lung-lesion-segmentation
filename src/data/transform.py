from pathlib import Path
import sys
from typing import Tuple, Union
import cv2
import numpy as np
import torch

# Đảm bảo nhận diện thư mục gốc của project
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.data.hu_window import apply_hu_window


def normalize_min_max(
    image: Union[np.ndarray, torch.Tensor],
    min_hu: float = -1000.0,
    max_hu: float = 400.0,
) -> Union[np.ndarray, torch.Tensor]:
    """Cắt ngưỡng HU [-1000, 400] và co dãn tuyến tính về đoạn [0, 1]."""
    clipped = apply_hu_window(image, min_hu=min_hu, max_hu=max_hu)
    denom = max_hu - min_hu

    if isinstance(clipped, np.ndarray):
        norm = (clipped - min_hu) / denom
        return np.clip(norm, 0.0, 1.0).astype(np.float32)
    elif isinstance(clipped, torch.Tensor):
        norm = (clipped - min_hu) / denom
        return torch.clamp(norm, 0.0, 1.0).to(dtype=torch.float32)
    else:
        raise TypeError(f"Kiểu dữ liệu không hỗ trợ: {type(image)}")


def resize_ct_pair(
    image: np.ndarray,
    mask: np.ndarray,
    target_size: Tuple[int, int] = (256, 256),
) -> Tuple[np.ndarray, np.ndarray]:
    """Resize đồng bộ cặp ảnh CT và Mask.

    - CT: Bilinear Interpolation (cv2.INTER_LINEAR)
    - Mask: Nearest Neighbor (cv2.INTER_NEAREST) bảo toàn nhãn rời rạc
    """
    dsize = (target_size[1], target_size[0])  # (width, height)
    resized_image = cv2.resize(image, dsize, interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(mask, dsize, interpolation=cv2.INTER_NEAREST)
    return resized_image, resized_mask


def preprocess_slice(
    image_slice: np.ndarray,
    mask_slice: np.ndarray,
    target_size: Tuple[int, int] = (256, 256),
) -> Tuple[np.ndarray, np.ndarray]:
    """Pipeline tiền xử lý toàn diện: HU clipping -> Min-Max [0, 1] -> Resizing."""
    norm_img = normalize_min_max(image_slice)
    res_img, res_mask = resize_ct_pair(
        norm_img, mask_slice, target_size=target_size
    )
    return res_img, res_mask


if __name__ == "__main__":
    print("Đang chạy Unit Test cho DATA-06 (Min-Max Scaling & Resizing)...")

    # 1. Kiểm tra dải Min-Max trên NumPy
    raw_np = np.array(
        [[-1500.0, -1000.0], [-300.0, 400.0], [1500.0, 0.0]], dtype=np.float32
    )
    norm_np = normalize_min_max(raw_np)
    assert norm_np.min() >= 0.0 and norm_np.max() <= 1.0, (
        "Lỗi Min-Max dải [0, 1] trên NumPy!"
    )
    assert np.isclose(norm_np[0, 1], 0.0) and np.isclose(norm_np[1, 1], 1.0), (
        "Lỗi tính toán chuẩn hóa tại biên!"
    )

    # 2. Kiểm tra dải Min-Max trên Tensor
    raw_ts = torch.tensor(raw_np)
    norm_ts = normalize_min_max(raw_ts)
    assert norm_ts.min() >= 0.0 and norm_ts.max() <= 1.0, (
        "Lỗi Min-Max dải [0, 1] trên Tensor!"
    )

    # 3. Kiểm tra bảo toàn nhãn rời rạc khi Resize Mask (Nearest Neighbor)
    test_img = np.random.uniform(-1000, 400, size=(512, 512)).astype(np.float32)
    test_mask = np.random.choice([0, 1, 2, 3], size=(512, 512)).astype(np.uint8)

    out_img, out_mask = resize_ct_pair(
        test_img, test_mask, target_size=(256, 256)
    )
    assert out_img.shape == (256, 256), "Sai kích thước ảnh CT sau resize!"
    assert out_mask.shape == (256, 256), "Sai kích thước mask sau resize!"

    unique_in = set(np.unique(test_mask))
    unique_out = set(np.unique(out_mask))
    assert unique_out.issubset(unique_in), (
        "LỖI NGHIÊM TRỌNG: Phép nội suy làm phát sinh nhãn lạ ngoài {0, 1, 2, 3}!"
    )

    print("=> [DATA-06] Unit test Min-Max scaling & Resizing PASSED 100%!")