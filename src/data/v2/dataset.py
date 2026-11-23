"""Lung CT Dataset V2 for Dual-Source Architecture.

Key features:
1. Safe path resolution via SafePathResolver (strictly isolates CT and Mask, no unsafe fallback).
2. Dual-source preprocessing (Coronacases HU-window vs Radiopaedia direct scaling).
3. Synchronized spatial resize (256x256) and binary mask thresholding.
4. Augmentations strictly confined to 'train' split.
5. In-memory volume caching for optimal I/O efficiency.
6. Returns standardized tensors:
   image: [1, 256, 256] float32 in [0.0, 1.0]
   mask:  [1, 256, 256] float32 in {0.0, 1.0}
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import nibabel as nib
except ImportError:
    nib = None

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.augmentations import (
    apply_augmentation,
    get_training_augmentation,
    get_validation_augmentation,
)
from src.data.v2.manifest import load_manifest_v2
from src.data.v2.preprocessing.common import binarize_mask_v2, resize_ct_pair_v2
from src.data.v2.preprocessing.registry import get_preprocessor
from src.data.v2.resolver import SafePathResolver
from src.data.v2.source_detector import detect_source
from src.utils.seed import seed_worker


class LungCTDatasetV2(Dataset):
    """PyTorch Dataset V2 reading 2D slices from dual-source CT volumes."""

    def __init__(
        self,
        manifest_path: str = "data/processed_v2/clean_slices_manifest_v2.csv",
        split: str = "train",
        target_size: Tuple[int, int] = (256, 256),
        lesion_label: int = 1,
        cache_volumes: bool = True,
        raw_data_dir: Optional[str] = None,
        samples_override: Optional[List[Dict[str, Any]]] = None,
    ):
        assert split in ["train", "val", "test"], f"Invalid split: '{split}'"
        self.split = split
        self.target_size = target_size
        self.lesion_label = lesion_label
        self.cache_volumes = cache_volumes
        self.raw_data_dir = raw_data_dir

        self.resolver = SafePathResolver(raw_data_dir=raw_data_dir)

        if samples_override is not None:
            self.samples = [s for s in samples_override if s.get("split") == split]
        else:
            self.samples = load_manifest_v2(manifest_path, split=split)

        if len(self.samples) == 0:
            raise ValueError(f"LungCTDatasetV2: No samples found for split '{split}' in {manifest_path}!")

        # Setup transforms
        if self.split == "train":
            self.transform = get_training_augmentation(img_size=self.target_size)
        else:
            self.transform = get_validation_augmentation(img_size=self.target_size)

        # In-memory caching for 3D volumes
        self._ct_cache: Dict[str, np.ndarray] = {}
        self._mask_cache: Dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _load_volume_pair(self, ct_path_str: str, mask_path_str: str, patient_id: str, sample_id: str) -> Tuple[np.ndarray, np.ndarray]:
        if nib is None:
            raise ImportError("nibabel is required to load .nii / .nii.gz files: pip install nibabel")

        # Resolve paths safely
        resolved_ct, resolved_mask = self.resolver.resolve_pair(
            ct_path=ct_path_str,
            mask_path=mask_path_str,
            patient_id=patient_id,
            sample_id=sample_id,
        )

        resolved_ct_str = str(resolved_ct)
        resolved_mask_str = str(resolved_mask)

        if self.cache_volumes:
            if resolved_ct_str not in self._ct_cache:
                ct_nii = nib.load(resolved_ct_str)
                self._ct_cache[resolved_ct_str] = ct_nii.get_fdata(dtype=np.float32)
            if resolved_mask_str not in self._mask_cache:
                mask_nii = nib.load(resolved_mask_str)
                self._mask_cache[resolved_mask_str] = mask_nii.get_fdata(dtype=np.float32)

            return self._ct_cache[resolved_ct_str], self._mask_cache[resolved_mask_str]
        else:
            ct_nii = nib.load(resolved_ct_str)
            mask_nii = nib.load(resolved_mask_str)
            return ct_nii.get_fdata(dtype=np.float32), mask_nii.get_fdata(dtype=np.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        record = self.samples[idx]
        sample_id = record["sample_id"]
        patient_id = record["patient_id"]
        ct_path_str = record["source_volume_path"]
        mask_path_str = record["source_mask_path"]
        z = int(record["slice_index"])
        source = record.get("source") or detect_source(sample_id=sample_id, patient_id=patient_id, path=ct_path_str)

        # 1. Load 3D Volume slice
        ct_vol, mask_vol = self._load_volume_pair(ct_path_str, mask_path_str, patient_id, sample_id)
        raw_slice = ct_vol[:, :, z]
        raw_mask = mask_vol[:, :, z]

        # 2. Source-Specific Intensity Preprocessing
        preprocessor = get_preprocessor(source)
        norm_img = preprocessor.preprocess(raw_slice, metadata=record)

        # 3. Mask Binarization
        bin_mask = binarize_mask_v2(raw_mask, lesion_label=self.lesion_label)

        # 4. Augmentations (train) or Resize (val/test)
        if self.split == "train":
            aug_img, aug_mask = apply_augmentation(self.transform, norm_img, bin_mask.astype(np.uint8))
            aug_mask = (aug_mask > 0.5).astype(np.float32)
        else:
            aug_img, aug_mask = resize_ct_pair_v2(norm_img, bin_mask, target_size=self.target_size)
            aug_mask = (aug_mask > 0.5).astype(np.float32)

        # 5. Tensor conversion [1, H, W]
        tensor_img = torch.from_numpy(aug_img).unsqueeze(0).to(torch.float32)
        tensor_mask = torch.from_numpy(aug_mask).unsqueeze(0).to(torch.float32)

        metadata = {
            "sample_id": sample_id,
            "patient_id": patient_id,
            "source": source,
            "slice_index": z,
            "lesion_present": int(record["lesion_present"]),
            "split": self.split,
        }

        return tensor_img, tensor_mask, metadata


def get_dataloader_v2(
    manifest_path: str = "data/processed_v2/clean_slices_manifest_v2.csv",
    split: str = "train",
    batch_size: int = 16,
    target_size: Tuple[int, int] = (256, 256),
    lesion_label: int = 1,
    num_workers: int = 0,
    pin_memory: bool = False,
    raw_data_dir: Optional[str] = None,
    samples_override: Optional[List[Dict[str, Any]]] = None,
) -> DataLoader:
    """Create PyTorch DataLoader for Dataset V2."""
    dataset = LungCTDatasetV2(
        manifest_path=manifest_path,
        split=split,
        target_size=target_size,
        lesion_label=lesion_label,
        raw_data_dir=raw_data_dir,
        samples_override=samples_override,
    )

    shuffle = (split == "train")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
    )
    return loader
