"""Coronacases Preprocessor for Pipeline V2.

Applies standard pulmonary HU windowing [-1000.0, 400.0] and linear normalization
to [0.0, 1.0] for Coronacases CT scans (which are natively HU or HU-like int16/float32).
"""

from typing import Any, Dict, Optional
import numpy as np
from src.data.v2.preprocessing.base import BasePreprocessor


class CoronacasesPreprocessor(BasePreprocessor):
    """Preprocessor for Coronacases CT data with pulmonary HU windowing."""

    def __init__(
        self,
        min_hu: float = -1000.0,
        max_hu: float = 400.0,
    ):
        self.min_hu = float(min_hu)
        self.max_hu = float(max_hu)
        assert self.max_hu > self.min_hu, "max_hu must be greater than min_hu"
        self.denom = self.max_hu - self.min_hu

    def preprocess(
        self,
        image: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Window and scale Coronacases slice.

        Formula:
            clipped = np.clip(image, min_hu, max_hu)
            normalized = (clipped - min_hu) / (max_hu - min_hu)
        """
        if not isinstance(image, np.ndarray):
            image = np.asarray(image)

        # Sanity check: Ensure input is numeric
        if not np.issubdtype(image.dtype, np.number):
            raise TypeError(f"CoronacasesPreprocessor expects numeric array, got {image.dtype}")

        # Check for NaN / Inf
        if np.isnan(image).any() or np.isinf(image).any():
            image = np.nan_to_num(image, nan=self.min_hu, posinf=self.max_hu, neginf=self.min_hu)

        img_float = image.astype(np.float32)
        clipped = np.clip(img_float, self.min_hu, self.max_hu)
        normalized = (clipped - self.min_hu) / self.denom

        return np.clip(normalized, 0.0, 1.0).astype(np.float32)
