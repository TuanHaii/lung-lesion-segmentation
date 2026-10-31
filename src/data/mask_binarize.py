from typing import Union
import numpy as np
import torch


def binarize_mask(
    mask: Union[np.ndarray, torch.Tensor],
    lesion_label: int = 1,
    dtype: str = "uint8",
) -> Union[np.ndarray, torch.Tensor]:
    """Chuyển đổi nhãn phân đoạn về dạng nhị phân nghiêm ngặt {0, 1}.

    Tự động tương thích cả 2 chuẩn dataset:
    - Chuẩn Infection_Mask nhị phân {0, 1} -> Lấy nhãn 1
    - Chuẩn Lung_and_Infection_Mask đa lớp {0, 1, 2, 3} -> Lấy nhãn 3

    Parameters
    ----------
    mask : np.ndarray hoặc torch.Tensor
        Mask phân đoạn gốc chứa nhãn {0, 1} hoặc đa lớp {0, 1, 2, 3}.
    lesion_label : int, mặc định 1
        Mã nhãn đại diện cho vùng tổn thương nhiễm trùng (1 cho Infection_Mask, 3 cho Lung_and_Infection).
    dtype : str, mặc định 'uint8'
        Kiểu dữ liệu đầu ra mong muốn ('uint8' hoặc 'float32').

    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        Binary mask với 1 là tổn thương và 0 là nền / phổi lành.
    """
    if isinstance(mask, np.ndarray):
        max_val = float(np.max(mask)) if mask.size > 0 else 0.0
        # Tự động thích ứng nếu cấu hình truyền nhãn 3 nhưng file mask chỉ có {0, 1}
        if max_val <= 1.0 and lesion_label == 3:
            target_label = 1
        elif max_val >= 3.0 and lesion_label != 1:
            target_label = 3
        else:
            target_label = lesion_label

        binary = (mask == target_label).astype(
            np.float32 if dtype == "float32" else np.uint8
        )
        return binary

    elif isinstance(mask, torch.Tensor):
        target_dtype = torch.float32 if dtype == "float32" else torch.uint8
        max_val = float(torch.max(mask).item()) if mask.numel() > 0 else 0.0
        if max_val <= 1.0 and lesion_label == 3:
            target_label = 1
        elif max_val >= 3.0 and lesion_label != 1:
            target_label = 3
        else:
            target_label = lesion_label

        binary = (mask == target_label).to(target_dtype)
        return binary

    else:
        raise TypeError(
            f"Dữ liệu đầu vào phải là numpy.ndarray hoặc torch.Tensor, nhận được: {type(mask)}"
        )


if __name__ == "__main__":
    print("Đang chạy Unit Test cho DATA-07 (Mask Binarization)...")

    # 1. Kiểm thử trên NumPy array đa lớp {0, 1, 2, 3} với lesion_label=3
    raw_np = np.array(
        [[0, 1, 2], [3, 3, 0], [1, 2, 3]], dtype=np.int32
    )
    expected_np = np.array(
        [[0, 0, 0], [1, 1, 0], [0, 0, 1]], dtype=np.uint8
    )
    bin_np = binarize_mask(raw_np, lesion_label=3, dtype="uint8")

    assert np.array_equal(bin_np, expected_np), "Lỗi: Nhị phân hóa sai trên NumPy!"
    assert set(np.unique(bin_np)).issubset({0, 1}), "Lỗi: Xuất hiện nhãn ngoài {0, 1}!"
    assert bin_np.dtype == np.uint8, "Lỗi: Kiểu dữ liệu không phải uint8!"

    # 2. Kiểm thử tự động tương thích với bộ mask nhị phân {0, 1}
    raw_bin = np.array([[0, 1], [1, 0]], dtype=np.int32)
    bin_res1 = binarize_mask(raw_bin, lesion_label=1, dtype="uint8")
    bin_res3 = binarize_mask(raw_bin, lesion_label=3, dtype="uint8")  # Auto-fallback to 1
    assert np.array_equal(bin_res1, raw_bin), "Lỗi nhị phân hóa với lesion_label=1!"
    assert np.array_equal(bin_res3, raw_bin), "Lỗi auto-fallback sang 1 khi mask chỉ có {0, 1}!"

    # 3. Kiểm thử trên PyTorch Tensor với float32 (cho hàm loss)
    raw_ts = torch.tensor([[0, 2], [3, 1]], dtype=torch.long)
    expected_ts = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    bin_ts = binarize_mask(raw_ts, lesion_label=3, dtype="float32")

    assert torch.equal(bin_ts, expected_ts), "Lỗi: Nhị phân hóa sai trên Tensor!"
    assert bin_ts.dtype == torch.float32, "Lỗi: Kiểu dữ liệu không phải float32!"

    print("=> [DATA-07] Unit test Mask Binarization PASSED 100%!")
