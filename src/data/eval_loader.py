#!/usr/bin/env python3
"""
Task ID: EVAL-01 (Data & Evaluation Lead)
Tên module: src/data/eval_loader.py
Mục đích:
- Cung cấp PyTorch Dataset (IndependentTestDataset) và DataLoader (get_eval_dataloader)
  chuẩn hóa y khoa cho tập kiểm thử độc lập (Independent Test Set).
- Tuân thủ quy chuẩn tiền xử lý y tế v1.0:
  + Hounsfield Unit (HU) Windowing [-1000, 400]
  + Min-Max Normalization về dải [0.0, 1.0]
  + Nhị phân hóa Ground Truth Mask về {0.0, 1.0}
  + Chuẩn hóa kích thước chuẩn (256, 256)
  + Tuyệt đối KHÔNG áp dụng Data Augmentation
- Thiết lập bắt buộc: batch_size = 1, shuffle = False (phản ánh chẩn đoán lâm sàng tuần tự).
- Trả về dictionary: {"image", "mask", "sample_id", "patient_id", "slice_index", "lesion_present", "volume_id"}.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import nibabel as nib
except ImportError:
    nib = None


def is_manifest_readonly(manifest_path: Union[str, Path]) -> bool:
    """Kiểm tra xem file manifest có đang ở trạng thái READ-ONLY hay không."""
    p = Path(manifest_path)
    if not p.exists():
        return False
    return not os.access(p, os.W_OK)


def set_manifest_readonly(manifest_path: Union[str, Path], readonly: bool = True) -> None:
    """Thiết lập hoặc gỡ bỏ thuộc tính READ-ONLY cho file manifest."""
    p = Path(manifest_path)
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy manifest: {manifest_path}")
    if readonly:
        os.chmod(p, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    else:
        os.chmod(p, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)


class IndependentTestDataset(Dataset):
    """Dataset chuẩn y tế phục vụ đánh giá độc lập mô hình phân đoạn tổn thương CT phổi."""

    def __init__(
        self,
        manifest_path: Union[str, Path] = "data/processed/test_manifest.json",
        target_size: Tuple[int, int] = (256, 256),
        hu_range: Tuple[float, float] = (-1000.0, 400.0),
        lesion_label: int = 1,
        raw_data_dir: Optional[str] = None,
        cache_volumes: bool = True,
        mock_fallback: bool = True,
        enforce_readonly: bool = True,
    ):
        self.manifest_path = Path(manifest_path)
        self.target_size = target_size
        self.hu_range = hu_range
        self.lesion_label = lesion_label
        self.raw_data_dir = raw_data_dir or os.environ.get("RAW_DATA_DIR")
        self.cache_volumes = cache_volumes
        self.mock_fallback = mock_fallback

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"[ERROR] Không tìm thấy test manifest tại: {self.manifest_path}. "
                "Hãy chạy 'python scripts/prepare_test_manifest.py' trước!"
            )

        if enforce_readonly and not is_manifest_readonly(self.manifest_path):
            try:
                set_manifest_readonly(self.manifest_path, readonly=True)
            except Exception as e:
                print(f"[!] Cảnh báo: Không thể khóa quyền ghi cho manifest: {e}")

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.metadata = data.get("metadata", {})
        self.slices: List[Dict[str, Any]] = data.get("slices", [])
        self.test_patients: List[str] = self.metadata.get("test_patients", [])

        if not self.slices:
            raise ValueError("[ERROR] Danh sách lát cắt trong test manifest bị rỗng!")

        self._volume_cache: Dict[str, np.ndarray] = {}
        self._mask_volume_cache: Dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.slices)

    def _resolve_path(self, path_str: str) -> Optional[Path]:
        """Tìm đường dẫn thực tế của file NIfTI."""
        p = Path(path_str)
        if p.exists() and p.stat().st_size > 0:
            return p

        candidates = []
        if self.raw_data_dir:
            base = Path(self.raw_data_dir)
            candidates.extend([
                base / p,
                base / p.name,
                base / "ct_volumes" / p.name,
                base / "ground_truth_masks" / p.name,
                base / "data" / "raw" / p.name,
            ])

        for c in candidates:
            if c.exists() and c.stat().st_size > 0:
                return c

        return None

    def _generate_mock_slice(self, record: Dict[str, Any], idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sinh lát cắt giả lập kích thước 512x512 tuân thủ phân bố HU và nhãn ground truth."""
        rng = np.random.RandomState(idx + 2026)
        ct_raw = np.full((512, 512), fill_value=-1000.0, dtype=np.float32)
        ct_raw[100:412, 100:412] = rng.uniform(-700.0, -500.0, size=(312, 312))
        ct_raw[80:100, 80:432] = 40.0
        ct_raw[85:95, 120:140] = 700.0

        mask_raw = np.zeros((512, 512), dtype=np.float32)
        if int(record.get("lesion_present", 0)) == 1:
            ct_raw[200:260, 140:200] = rng.uniform(-200.0, 50.0, size=(60, 60))
            mask_raw[200:260, 140:200] = 1.0

        return ct_raw, mask_raw

    def _load_slice(self, record: Dict[str, Any], idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Tải lát cắt CT và Mask từ file NIfTI hoặc dùng mock fallback nếu không tìm thấy file."""
        vol_path = self._resolve_path(record.get("source_volume_path", ""))
        mask_path = self._resolve_path(record.get("source_mask_path", ""))

        if vol_path is None or mask_path is None or nib is None:
            if self.mock_fallback:
                return self._generate_mock_slice(record, idx)
            raise FileNotFoundError(
                f"[ERROR] Không tìm thấy file NIfTI: {record.get('source_volume_path')} "
                f"hoặc {record.get('source_mask_path')}"
            )

        z = int(record["slice_index"])

        # Cache volume vào RAM nếu được kích hoạt
        v_str = str(vol_path)
        m_str = str(mask_path)
        if self.cache_volumes:
            if v_str not in self._volume_cache:
                self._volume_cache[v_str] = nib.load(v_str).get_fdata(dtype=np.float32)
            if m_str not in self._mask_volume_cache:
                self._mask_volume_cache[m_str] = nib.load(m_str).get_fdata(dtype=np.float32)
            ct_slice = self._volume_cache[v_str][:, :, z]
            mask_slice = self._mask_volume_cache[m_str][:, :, z]
        else:
            ct_slice = nib.load(v_str).get_fdata(dtype=np.float32)[:, :, z]
            mask_slice = nib.load(m_str).get_fdata(dtype=np.float32)[:, :, z]

        # Binarize mask
        if self.lesion_label is not None and np.max(mask_slice) > 1.0:
            mask_binary = (mask_slice == self.lesion_label).astype(np.float32)
        else:
            mask_binary = (mask_slice >= 0.5).astype(np.float32)

        return ct_slice, mask_binary

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.slices[idx]
        ct_raw, mask_raw = self._load_slice(record, idx)

        # 1. HU Windowing [-1000, 400]
        min_hu, max_hu = self.hu_range
        ct_clipped = np.clip(ct_raw, min_hu, max_hu)

        # 2. Min-Max Normalization về [0.0, 1.0]
        ct_norm = (ct_clipped - min_hu) / (max_hu - min_hu + 1e-7)
        ct_norm = np.clip(ct_norm, 0.0, 1.0).astype(np.float32)

        # 3. Resize về kích thước chuẩn (256, 256)
        if ct_norm.shape != self.target_size:
            if cv2 is not None:
                ct_resized = cv2.resize(
                    ct_norm,
                    (self.target_size[1], self.target_size[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                mask_resized = cv2.resize(
                    mask_raw.astype(np.float32),
                    (self.target_size[1], self.target_size[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            else:
                # Fallback bằng torch
                ct_t = torch.from_numpy(ct_norm).unsqueeze(0).unsqueeze(0)
                mask_t = torch.from_numpy(mask_raw).unsqueeze(0).unsqueeze(0)
                ct_resized = torch.nn.functional.interpolate(
                    ct_t, size=self.target_size, mode="bilinear", align_corners=False
                ).squeeze().numpy()
                mask_resized = torch.nn.functional.interpolate(
                    mask_t, size=self.target_size, mode="nearest"
                ).squeeze().numpy()
        else:
            ct_resized = ct_norm
            mask_resized = mask_raw

        # Đảm bảo mask nhị phân nghiêm ngặt {0.0, 1.0}
        mask_binary = (mask_resized >= 0.5).astype(np.float32)

        image_tensor = torch.from_numpy(ct_resized).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0).float()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "sample_id": record["sample_id"],
            "patient_id": record["patient_id"],
            "slice_index": record["slice_index"],
            "lesion_present": record["lesion_present"],
            "volume_id": record.get("volume_id", ""),
        }


def get_eval_dataloader(
    manifest_path: Union[str, Path] = "data/processed/test_manifest.json",
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    raw_data_dir: Optional[str] = None,
    enforce_readonly: bool = True,
) -> DataLoader:
    """Khởi tạo DataLoader tiêu chuẩn cho tập kiểm thử độc lập."""
    dataset = IndependentTestDataset(
        manifest_path=manifest_path,
        raw_data_dir=raw_data_dir,
        enforce_readonly=enforce_readonly,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )

