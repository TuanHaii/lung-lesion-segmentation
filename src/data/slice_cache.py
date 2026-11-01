#!/usr/bin/env python3
"""Dataset and DataLoader backed by the compact 2D slice cache.

The cache is produced by ``scripts/build_slice_cache.py``.  It keeps the
exact same preprocessing as :class:`src.data.dataset.LungCTDataset`
(HU window, min-max scaling, binary lesion mask, 256x256 resize) but stores
the result as ``float16`` images instead of reloading 3D NIfTI volumes.

Samples are returned in the same ``(image, mask, metadata)`` format used by
the rest of the training pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.augmentations import (
    apply_augmentation,
    get_training_augmentation,
    get_validation_augmentation,
)


class CachedSliceDataset(Dataset):
    def __init__(
        self,
        cache_path: str | Path,
        augment: bool = False,
        target_size: Tuple[int, int] = (256, 256),
    ) -> None:
        data = np.load(str(cache_path), allow_pickle=False)
        self.images = data["images"]
        self.masks = data["masks"]
        self.sample_ids = [str(value) for value in data["sample_ids"]]
        self.patient_ids = [str(value) for value in data["patient_ids"]]
        self.lesion_present = data["lesion_present"]
        if len(self.sample_ids) != len(self.images):
            raise ValueError(f"Corrupt slice cache: {cache_path}")

        self.transform = (
            get_training_augmentation(img_size=target_size)
            if augment
            else get_validation_augmentation(img_size=target_size)
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        image = np.asarray(self.images[idx], dtype=np.float32)
        mask = np.asarray(self.masks[idx], dtype=np.uint8)
        image, mask = apply_augmentation(self.transform, image, mask)

        metadata = {
            "sample_id": self.sample_ids[idx],
            "patient_id": self.patient_ids[idx],
            "lesion_present": int(self.lesion_present[idx]),
        }
        return (
            torch.from_numpy(image).unsqueeze(0).to(torch.float32),
            torch.from_numpy(mask).unsqueeze(0).to(torch.float32),
            metadata,
        )


def get_cached_dataloader(
    cache_dir: str | Path,
    split: str,
    batch_size: int = 8,
    num_workers: int = 2,
    pin_memory: bool = True,
    target_size: Tuple[int, int] = (256, 256),
) -> DataLoader:
    cache_path = Path(cache_dir) / f"{split}.npz"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing slice cache '{cache_path}'. Run scripts/build_slice_cache.py first."
        )
    dataset = CachedSliceDataset(
        cache_path=cache_path,
        augment=(split == "train"),
        target_size=target_size,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
