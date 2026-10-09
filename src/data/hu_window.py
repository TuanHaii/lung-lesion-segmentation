from typing import Union
import numpy as np
import torch


def apply_hu_window(
    image: Union[np.ndarray, torch.Tensor],
    min_hu: float = -1000.0,
    max_hu: float = 400.0,
) -> Union[np.ndarray, torch.Tensor]:
    """Cắt dải Hounsfield Unit (HU) về cửa sổ nhu mô phổi [-1000, 400].

    Parameters
    ----------
    image : np.ndarray hoặc torch.Tensor
        Mảng ảnh CT 2D hoặc 3D chứa giá trị HU thô.
    min_hu : float, mặc định -1000.0
        Ngưỡng sàn (không khí / ngoài lồng ngực).
    max_hu : float, mặc định 400.0
        Ngưỡng trần (mô đặc / xương sườn).

    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        Ảnh sau khi clip ngưỡng cường độ, giữ nguyên cấu trúc đầu vào.
    """
    if isinstance(image, np.ndarray):
        return np.clip(image, min_hu, max_hu)
    elif isinstance(image, torch.Tensor):
        return torch.clamp(image, min=min_hu, max=max_hu)
    else:
        raise TypeError(
            f"Đầu vào phải là numpy.ndarray hoặc torch.Tensor, nhận được: {type(image)}"
        )


if __name__ == "__main__":
    print("Đang chạy Unit Test cho DATA-05 (HU Windowing)...")

    # 1. Kiểm thử trên NumPy array
    raw_np = np.array(
        [-1250.0, -1000.0, -450.0, 0.0, 400.0, 1050.0], dtype=np.float32
    )
    expected_np = np.array(
        [-1000.0, -1000.0, -450.0, 0.0, 400.0, 400.0], dtype=np.float32
    )
    res_np = apply_hu_window(raw_np)
    assert np.allclose(res_np, expected_np), (
        "Lỗi: HU clipping không đúng trên NumPy!"
    )

    # 2. Kiểm thử trên PyTorch Tensor
    raw_tensor = torch.tensor(raw_np)
    expected_tensor = torch.tensor(expected_np)
    res_tensor = apply_hu_window(raw_tensor)
    assert torch.allclose(res_tensor, expected_tensor), (
        "Lỗi: HU clipping không đúng trên PyTorch Tensor!"
    )

    # 3. Kiểm thử giữ nguyên giá trị trong dải hợp lệ
    in_range_arr = np.array([-800.0, -200.0, 150.0], dtype=np.float32)
    assert np.allclose(apply_hu_window(in_range_arr), in_range_arr), (
        "Lỗi: Giá trị hợp lệ bị biến đổi!"
    )

    print("=> [DATA-05] Unit test HU Windowing [-1000, 400] PASSED 100%!")