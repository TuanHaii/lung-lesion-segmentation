"""Data V2 Package for Dual-Source CT Dataset."""

from src.data.v2.source_detector import detect_source, SourceIdentificationError
from src.data.v2.resolver import SafePathResolver, PathResolverError
from src.data.v2.manifest import create_manifest_v2, load_manifest_v2, MANIFEST_V2_COLUMNS
from src.data.v2.integrity import validate_dataset_integrity, IntegrityReport
from src.data.v2.dataset import LungCTDatasetV2, get_dataloader_v2
from src.data.v2.preprocessing import (
    BasePreprocessor,
    CoronacasesPreprocessor,
    RadiopaediaPreprocessor,
    get_preprocessor,
    resize_ct_pair_v2,
    binarize_mask_v2,
)

__all__ = [
    "detect_source",
    "SourceIdentificationError",
    "SafePathResolver",
    "PathResolverError",
    "create_manifest_v2",
    "load_manifest_v2",
    "MANIFEST_V2_COLUMNS",
    "validate_dataset_integrity",
    "IntegrityReport",
    "LungCTDatasetV2",
    "get_dataloader_v2",
    "BasePreprocessor",
    "CoronacasesPreprocessor",
    "RadiopaediaPreprocessor",
    "get_preprocessor",
    "resize_ct_pair_v2",
    "binarize_mask_v2",
]
