from typing import Tuple, Union
import albumentations as A
import cv2
import numpy as np
import torch


def get_training_augmentation(img_size: Tuple[int, int] = (256, 256)) -> A.Compose:
    """Pipeline tăng cường dữ liệu dành riêng cho tập Training.
    
    Bao gồm:
    - Horizontal Flip (p=0.5)
    - Small Rotation (+-15 độ), Shift, Scale
    - Elastic Transform mô phỏng biến dạng mô mềm phổi
    """
    return A.Compose([
        # 1. Lật ngang (mô phỏng đối xứng phổi trái - phải)
        A.HorizontalFlip(p=0.5),

        # 2. Xoay góc hẹp +-15 độ và co giãn nhẹ (không làm mất đặc trưng)
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=15,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            value=0.0,
            mask_value=0,
            p=0.5,
        ),

        # 3. Biến dạng đàn hồi nhẹ mô phỏng độ giãn nở lồng ngực
        A.ElasticTransform(
            alpha=1,
            sigma=50,
            alpha_affine=10,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            value=0.0,
            mask_value=0,
            p=0.3,
        ),

        # 4. Đảm bảo kích thước đầu ra đúng chuẩn
        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR),
    ], additional_targets={"mask": "mask"})


def get_validation_augmentation(img_size: Tuple[int, int] = (256, 256)) -> A.Compose:
    """Pipeline dành cho tập Validation và Test: Tuyệt đối không biến dạng hình học."""
    return A.Compose([
        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR),
    ], additional_targets={"mask": "mask"})


def apply_augmentation(
    transform: A.Compose,
    image: np.ndarray,
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Thực thi pipeline biến đổi đồng thời trên cặp ảnh CT và Binary Mask."""
    # Đảm bảo mask ở dạng uint8 để tránh lỗi nội suy nhãn
    mask_in = mask.astype(np.uint8) if mask.dtype != np.uint8 else mask

    augmented = transform(image=image, mask=mask_in)
    aug_img = augmented["image"]
    aug_mask = augmented["mask"]

    # Ép ngưỡng cứng sau biến đổi để bảo toàn tính nhị phân {0, 1}
    aug_mask = (aug_mask > 0.5).astype(np.uint8)

    return aug_img, aug_mask


if __name__ == "__main__":
    print("Đang chạy Unit Test cho DATA-09 (Albumentations Pipeline)...")

    train_pipe = get_training_augmentation((256, 256))
    val_pipe = get_validation_augmentation((256, 256))

    # 1. Tạo cặp dữ liệu giả lập (ảnh CT normalized [0, 1], mask nhị phân {0, 1})
    dummy_img = np.random.uniform(0.0, 1.0, size=(256, 256)).astype(np.float32)
    dummy_mask = np.zeros((256, 256), dtype=np.uint8)
    dummy_mask[100:150, 100:150] = 1  # Vùng tổn thương giả lập

    # 2. Kiểm thử trên Training pipeline
    for seed in range(5):
        t_img, t_mask = apply_augmentation(train_pipe, dummy_img, dummy_mask)
        assert t_img.shape == (256, 256), f"Lỗi kích thước ảnh train: {t_img.shape}"
        assert t_mask.shape == (256, 256), f"Lỗi kích thước mask train: {t_mask.shape}"
        
        unique_mask_vals = set(np.unique(t_mask))
        assert unique_mask_vals.issubset({0, 1}), f"Lỗi: Mask sau biến đổi bị rò rỉ nhãn {unique_mask_vals}!"

    # 3. Kiểm thử trên Validation/Test pipeline (không làm thay đổi số lượng pixel mask)
    v_img, v_mask = apply_augmentation(val_pipe, dummy_img, dummy_mask)
    assert np.sum(v_mask) == np.sum(dummy_mask), "Validation pipeline làm biến đổi dữ liệu mask gốc!"

    print("=> [DATA-09] Unit test Data Augmentation PASSED 100%!")