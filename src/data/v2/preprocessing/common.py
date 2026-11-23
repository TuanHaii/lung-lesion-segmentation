"""Common Spatial and Mask Preprocessing for Pipeline V2.

Synchronously resizes CT image and ground-truth mask, applying:
- Bilinear interpolation for continuous image intensity
- Nearest-neighbor interpolation for discrete mask labels
- Strict binary mask thresholding {0, 1}
"""

from typing import Tuple
try:
    import cv2
except ImportError:
    cv2 = None

import numpy as np
import torch
import torch.nn.functional as F


def resize_ct_pair_v2(
    image: np.ndarray,
    mask: np.ndarray,
    target_size: Tuple[int, int] = (256, 256),
) -> Tuple[np.ndarray, np.ndarray]:
    """Resize CT image and mask synchronously.

    Parameters
    ----------
    image : np.ndarray
        2D preprocessed CT image (float32).
    mask : np.ndarray
        2D binary or multi-class mask.
    target_size : Tuple[int, int]
        (H, W) or (height, width).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (resized_image, resized_mask)
    """
    target_h, target_w = target_size

    if cv2 is not None:
        # cv2.resize expects (width, height)
        dsize = (target_w, target_h)
        resized_img = cv2.resize(image, dsize, interpolation=cv2.INTER_LINEAR)
        resized_msk = cv2.resize(mask, dsize, interpolation=cv2.INTER_NEAREST)
        return resized_img.astype(np.float32), resized_msk
    else:
        # PyTorch F.interpolate fallback
        t_img = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
        t_msk = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
        r_img = F.interpolate(t_img, size=(target_h, target_w), mode="bilinear", align_corners=False)
        r_msk = F.interpolate(t_msk, size=(target_h, target_w), mode="nearest")
        return r_img.squeeze().numpy().astype(np.float32), r_msk.squeeze().numpy().astype(mask.dtype)


def binarize_mask_v2(
    mask: np.ndarray,
    lesion_label: int = 1,
) -> np.ndarray:
    """Strictly binarize mask according to lesion_label.

    Returns float32 array with strictly {0.0, 1.0}.
    """
    return (mask == lesion_label).astype(np.float32)
