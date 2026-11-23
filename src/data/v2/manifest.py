"""Manifest V2 Builder and Loader.

Enriches the slice-level dataset manifest with:
- Strict source identification ('coronacases' or 'radiopaedia')
- Explicit preprocessing profile ('coronacases_hu_window' or 'radiopaedia_direct_scale')
- Partition assignment ('train', 'val', 'test')
- Preserves all original V1 manifest files without overwriting.
"""

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.data.v2.source_detector import detect_source


MANIFEST_V2_COLUMNS = [
    "sample_id",
    "patient_id",
    "source",
    "volume_id",
    "slice_index",
    "original_shape",
    "spacing_x",
    "spacing_y",
    "spacing_z",
    "lung_area_px",
    "lesion_present",
    "lesion_area_px",
    "source_volume_path",
    "source_mask_path",
    "preprocessing_profile",
    "split",
    "qa_status",
]


def create_manifest_v2(
    v1_manifest_path: str = "data/processed/clean_slices_manifest.csv",
    split_json_path: str = "data/processed/splits_patient_level.json",
    output_manifest_path: str = "data/processed_v2/clean_slices_manifest_v2.csv",
) -> List[Dict[str, Any]]:
    """Build Manifest V2 from V1 manifest and split JSON.

    Parameters
    ----------
    v1_manifest_path : str
        Source V1 manifest path (read-only).
    split_json_path : str
        Source V1 split json path (read-only).
    output_manifest_path : str
        Target V2 manifest destination (under data/processed_v2/).

    Returns
    -------
    List[Dict[str, Any]]
        List of enriched V2 sample records.
    """
    v1_path = Path(v1_manifest_path)
    split_path = Path(split_json_path)

    if not v1_path.exists():
        raise FileNotFoundError(f"V1 manifest not found at: {v1_path}")
    if not split_path.exists():
        raise FileNotFoundError(f"Split JSON not found at: {split_path}")

    with open(split_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    # Invert slice_samples mapping: sample_id -> split
    sample_to_split: Dict[str, str] = {}
    for split_name, sample_ids in split_data.get("slice_samples", {}).items():
        for sid in sample_ids:
            sample_to_split[sid] = split_name

    v2_records: List[Dict[str, Any]] = []

    with open(v1_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["sample_id"]
            pid = row["patient_id"]
            ct_path = row["source_volume_path"]
            mask_path = row["source_mask_path"]

            source = detect_source(sample_id=sid, patient_id=pid, path=ct_path)
            profile = (
                "coronacases_hu_window"
                if source == "coronacases"
                else "radiopaedia_direct_scale"
            )

            split = sample_to_split.get(sid, "unassigned")

            record: Dict[str, Any] = {
                "sample_id": sid,
                "patient_id": pid,
                "source": source,
                "volume_id": row["volume_id"],
                "slice_index": int(row["slice_index"]),
                "original_shape": row["original_shape"],
                "spacing_x": float(row["spacing_x"]),
                "spacing_y": float(row["spacing_y"]),
                "spacing_z": float(row["spacing_z"]),
                "lung_area_px": int(row["lung_area_px"]),
                "lesion_present": int(row["lesion_present"]),
                "lesion_area_px": int(row["lesion_area_px"]),
                "source_volume_path": ct_path,
                "source_mask_path": mask_path,
                "preprocessing_profile": profile,
                "split": split,
                "qa_status": row.get("qa_status", "PASSED"),
            }
            v2_records.append(record)

    out_p = Path(output_manifest_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with open(out_p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_V2_COLUMNS)
        writer.writeheader()
        writer.writerows(v2_records)

    return v2_records


def load_manifest_v2(
    manifest_path: str = "data/processed_v2/clean_slices_manifest_v2.csv",
    split: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load records from Manifest V2.

    Parameters
    ----------
    manifest_path : str
        Path to clean_slices_manifest_v2.csv.
    split : Optional[str]
        If provided ('train', 'val', 'test'), filters records by split.

    Returns
    -------
    List[Dict[str, Any]]
    """
    p = Path(manifest_path)
    if not p.exists():
        raise FileNotFoundError(f"Manifest V2 not found at: {p}")

    records = []
    with open(p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if split is not None and row["split"] != split:
                continue
            records.append(row)
    return records
