import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.augmentations import (
    apply_augmentation,
    get_training_augmentation,
    get_validation_augmentation,
)
from src.data.hu_window import apply_hu_window
from src.data.mask_binarize import binarize_mask
from src.data.transform import normalize_min_max


class LungCTDataset(Dataset):
    """PyTorch Dataset đọc các lát cắt 2D CT phổi và Ground Truth Mask chuẩn hóa."""

    def __init__(
        self,
        manifest_path: str = "data/processed/clean_slices_manifest.csv",
        split_json_path: str = "data/processed/splits_patient_level.json",
        split: str = "train",
        target_size: Tuple[int, int] = (256, 256),
        lesion_label: int = 3,
        cache_volumes: bool = True,
    ):
        """Khởi tạo Dataset theo phân vùng tập dữ liệu.

        Parameters
        ----------
        manifest_path : str
            Đường dẫn file clean_slices_manifest.csv.
        split_json_path : str
            Đường dẫn file splits_patient_level.json.
        split : str
            Phân vùng dữ liệu: 'train', 'val', hoặc 'test'.
        target_size : tuple
            Kích thước ảnh sau chuẩn hóa (mặc định 256x256).
        lesion_label : int
            Mã nhãn tổn thương trên mask thô (mặc định 3).
        cache_volumes : bool
            Lưu cache các 3D volume vào RAM để tối ưu tốc độ đọc đĩa.
        """
        assert split in ["train", "val", "test"], f"Split '{split}' không hợp lệ!"
        self.split = split
        self.target_size = target_size
        self.lesion_label = lesion_label
        self.cache_volumes = cache_volumes

        # 1. Đọc danh sách mẫu thuộc split từ JSON phân vùng
        with open(split_json_path, "r", encoding="utf-8") as f:
            split_data = json.load(f)
        valid_sample_ids = set(split_data["slice_samples"][split])

        # 2. Đọc và lọc manifest
        self.samples: List[Dict] = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["sample_id"] in valid_sample_ids:
                    self.samples.append(row)

        if len(self.samples) == 0:
            raise ValueError(f"Không tìm thấy mẫu nào cho tập '{split}'!")

        # 3. Thiết lập pipeline biến đổi
        if self.split == "train":
            self.transform = get_training_augmentation(img_size=self.target_size)
        else:
            self.transform = get_validation_augmentation(img_size=self.target_size)

        # 4. Bộ nhớ đệm RAM cho 3D volumes
        self._ct_cache: Dict[str, np.ndarray] = {}
        self._mask_cache: Dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _get_volume_data(self, ct_path: str, mask_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Tải dữ liệu 3D NIfTI kèm cơ chế đệm bộ nhớ."""
        if self.cache_volumes:
            if ct_path not in self._ct_cache:
                ct_nii = nib.load(ct_path)
                mask_nii = nib.load(mask_path)
                self._ct_cache[ct_path] = ct_nii.get_fdata(dtype=np.float32)
                self._mask_cache[mask_path] = mask_nii.get_fdata(dtype=np.float32)
            return self._ct_cache[ct_path], self._mask_cache[mask_path]
        else:
            ct_nii = nib.load(ct_path)
            mask_nii = nib.load(mask_path)
            return ct_nii.get_fdata(dtype=np.float32), mask_nii.get_fdata(dtype=np.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        record = self.samples[idx]
        ct_path = record["source_volume_path"]
        mask_path = record["source_mask_path"]
        z = int(record["slice_index"])

        # 1. Trích xuất lát cắt 2D thô
        ct_volume, mask_volume = self._get_volume_data(ct_path, mask_path)
        raw_slice = ct_volume[:, :, z]
        raw_mask = mask_volume[:, :, z]

        # 2. Tiền xử lý: HU Clipping [-1000, 400] & Min-Max [0, 1]
        norm_img = normalize_min_max(raw_slice, min_hu=-1000.0, max_hu=400.0)

        # 3. Tiền xử lý: Nhị phân hóa Mask {0, 1}
        bin_mask = binarize_mask(raw_mask, lesion_label=self.lesion_label, dtype="uint8")

        # 4. Augmentation (nếu là train) / Resize 256x256 (nếu là val/test)
        aug_img, aug_mask = apply_augmentation(self.transform, norm_img, bin_mask)

        # 5. Chuyển thành PyTorch Tensor [C, H, W] với C=1
        tensor_img = torch.from_numpy(aug_img).unsqueeze(0).to(torch.float32)
        tensor_mask = torch.from_numpy(aug_mask).unsqueeze(0).to(torch.float32)

        metadata = {
            "sample_id": record["sample_id"],
            "patient_id": record["patient_id"],
            "slice_index": z,
            "lesion_present": int(record["lesion_present"]),
        }

        return tensor_img, tensor_mask, metadata


def get_dataloader(
    split: str = "train",
    batch_size: int = 16,
    shuffle: Optional[bool] = None,
    num_workers: int = 2,
    pin_memory: bool = True,
    manifest_path: str = "data/processed/clean_slices_manifest.csv",
    split_json_path: str = "data/processed/splits_patient_level.json",
    cache_volumes: bool = True,
) -> DataLoader:
    """Khởi tạo PyTorch DataLoader tiêu chuẩn hóa cho huấn luyện và đánh giá."""
    is_train = split == "train"
    if shuffle is None:
        shuffle = is_train

    dataset = LungCTDataset(
        manifest_path=manifest_path,
        split_json_path=split_json_path,
        split=split,
        cache_volumes=cache_volumes,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=is_train,
    )
    return loader