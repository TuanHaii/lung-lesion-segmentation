"""Abstract Base Preprocessor for Pipeline V2."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np


class BasePreprocessor(ABC):
    """Abstract base class for CT intensity preprocessing."""

    @abstractmethod
    def preprocess(
        self,
        image: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Preprocess 2D raw CT image slice into normalized float32 [0.0, 1.0].

        Parameters
        ----------
        image : np.ndarray
            Raw 2D slice from CT volume.
        metadata : Optional[Dict[str, Any]]
            Optional slice or volume metadata.

        Returns
        -------
        np.ndarray
            Normalized 2D image in float32 with values bounded in [0.0, 1.0].
        """
        pass
