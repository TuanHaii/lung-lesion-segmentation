"""Radiopaedia Preprocessor for Pipeline V2.

Radiopaedia data consists of web-exported CT scans with intensities already windowed
into uint8 [0, 255].
Applying pulmonary HU windowing [-1000, 400] to Radiopaedia data would erroneously
compress intensities into [0.714, 0.896].
This preprocessor maps [0, 255] directly and linearly to [0.0, 1.0], preserving
tissue contrast and preventing intensity distortion.
"""

from typing import Any, Dict, Optional
import numpy as np
from src.data.v2.preprocessing.base import BasePreprocessor


class RadiopaediaPreprocessor(BasePreprocessor):
    """Preprocessor for Radiopaedia CT data with direct linear scaling."""

    def __init__(self, max_value: float = 255.0):
        self.max_value = float(max_value)
        assert self.max_value > 0.0, "max_value must be positive"

    def preprocess(
        self,
        image: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Linearly scale Radiopaedia slice [0, 255] to [0.0, 1.0]."""
        if not isinstance(image, np.ndarray):
            image = np.asarray(image)

        if not np.issubdtype(image.dtype, np.number):
            raise TypeError(f"RadiopaediaPreprocessor expects numeric array, got {image.dtype}")

        if np.isnan(image).any() or np.isinf(image).any():
            image = np.nan_to_num(image, nan=0.0, posinf=self.max_value, neginf=0.0)

        img_float = image.astype(np.float32)

        # Verify dynamic range: Radiopaedia is typically [0, 255]
        # If max is already <= 1.0 (e.g. pre-scaled), preserve it safely
        current_max = float(img_float.max()) if img_float.size > 0 else 0.0
        if current_max <= 1.0 and float(img_float.min()) >= 0.0:
            return np.clip(img_float, 0.0, 1.0).astype(np.float32)

        normalized = img_float / self.max_value
        return np.clip(normalized, 0.0, 1.0).astype(np.float32)
