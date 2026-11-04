#!/usr/bin/env python3
"""Shared utilities for the W5 evaluation tasks (EVAL-03 .. EVAL-07).

The module deliberately keeps the metric definitions in one place so that the
four per-metric scripts (DSC, IoU, Precision, Recall) and the statistical
comparison script all use exactly the same edge-case rules.

Primary input format
--------------------
`data/processed/test_manifest.json` contains one record per test slice.  A
prediction can be stored as a probability map or a binary mask, one file per
sample id, under e.g. `predictions/att_unet/<sample_id>.npy`.

The loader also supports a fallback input: the slice-level CSV emitted by
`src/evaluate.py` (`reports/<model>/test_metrics_slice_level.csv`).  This is
useful while the Kaggle inference output is being downloaded, and it avoids
inventing numbers when predictions are not available yet.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - opencv is in requirements.txt
    cv2 = None

try:
    import nibabel as nib
except ImportError:  # pragma: no cover - nibabel is in requirements.txt
    nib = None

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.mask_binarize import binarize_mask  # noqa: E402
from src.data.transform import resize_ct_pair  # noqa: E402

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_DISPLAY = {
    "att_unet": "Attention U-Net (Proposed)",
    "unet": "Standard U-Net (Baseline)",
    "segnet": "SegNet (Baseline)",
}

MODEL_ALIASES = {
    "att_unet": "att_unet",
    "att-unet": "att_unet",
    "attention_unet": "att_unet",
    "attention-unet": "att_unet",
    "attentionunet": "att_unet",
    "attunet": "att_unet",
    "unet": "unet",
    "u-net": "unet",
    "standard_unet": "unet",
    "baseline_unet": "unet",
    "segnet": "segnet",
    "seg-net": "segnet",
}

DEFAULT_MODELS = ("att_unet", "unet", "segnet")
METRIC_NAMES = ("dice", "iou", "precision", "recall")


def canonical_model(name: str) -> str:
    key = name.strip().lower()
    if key not in MODEL_ALIASES:
        raise ValueError(
            f"Unknown model '{name}'. Use one of: {', '.join(sorted(MODEL_ALIASES))}"
        )
    return MODEL_ALIASES[key]


def display_model(name: str) -> str:
    return MODEL_DISPLAY.get(canonical_model(name), name)


# ---------------------------------------------------------------------------
# Manifest / path handling
# ---------------------------------------------------------------------------


def load_manifest(manifest_path: str | Path) -> List[Dict[str, Any]]:
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Test manifest not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    slices = data.get("slices")
    if not isinstance(slices, list) or not slices:
        raise ValueError(f"Manifest has no 'slices': {path}")
    return slices


def _candidate_paths(path_str: str, raw_data_dir: Optional[str]) -> Iterable[Path]:
    p = Path(path_str)
    yield p
    if raw_data_dir:
        base = Path(raw_data_dir)
        yield base / p
        yield base / p.name
        yield base / p.parent.name / p.name
        yield base / "data" / "raw" / p.parent.name / p.name
        yield base / str(p).replace("data/raw/", "").replace("data\\raw\\", "")
    kaggle_root = Path("/kaggle/input")
    if kaggle_root.exists():
        yield kaggle_root / p.name
        yield from kaggle_root.rglob(p.name)


def resolve_data_path(path_str: str, raw_data_dir: Optional[str] = None) -> Path:
    seen = set()
    for candidate in _candidate_paths(path_str, raw_data_dir):
        try:
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.exists() and candidate.stat().st_size > 0:
                return candidate
        except OSError:
            continue
    raise FileNotFoundError(
        f"Could not resolve data file '{path_str}'. "
        "Pass --raw_data_dir or set RAW_DATA_DIR."
    )


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


class GroundTruthLoader:
    """Load and cache the binary lesion mask for each test record."""

    def __init__(
        self,
        target_size: Tuple[int, int] = (256, 256),
        lesion_label: int = 3,
        raw_data_dir: Optional[str] = None,
    ) -> None:
        if nib is None:
            raise ImportError("nibabel is required to read the NIfTI ground truth masks")
        self.target_size = target_size
        self.lesion_label = lesion_label
        self.raw_data_dir = raw_data_dir
        self._cache: Dict[Path, np.ndarray] = {}

    def get(self, record: Dict[str, Any]) -> np.ndarray:
        mask_path = resolve_data_path(record["source_mask_path"], self.raw_data_dir)
        if mask_path not in self._cache:
            self._cache[mask_path] = nib.load(str(mask_path)).get_fdata(dtype=np.float32)

        volume = self._cache[mask_path]
        z = int(record["slice_index"])
        if z < 0 or z >= volume.shape[2]:
            raise IndexError(
                f"slice_index={z} is outside volume {mask_path.name} "
                f"(depth={volume.shape[2]})"
            )

        raw_slice = volume[:, :, z]
        binary = binarize_mask(
            raw_slice, lesion_label=self.lesion_label, dtype="uint8"
        )
        if binary.shape != self.target_size:
            # Reuse the exact nearest-neighbour rule used by the evaluation loader.
            _, binary = resize_ct_pair(
                binary.astype(np.float32),
                binary,
                target_size=self.target_size,
            )
        return binary.astype(np.uint8)


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------


def _find_model_dir(predictions_dir: Path, model: str) -> Optional[Path]:
    names = [model]
    for alias, canonical in MODEL_ALIASES.items():
        if canonical == model and alias not in names:
            names.append(alias)
    for name in names:
        candidate = predictions_dir / name
        if candidate.is_dir():
            return candidate
    # A single-model directory may be passed directly.
    if any(predictions_dir.glob("*.*")):
        return predictions_dir
    return None


def _find_prediction_file(model_dir: Path, sample_id: str) -> Optional[Path]:
    for suffix in (".npy", ".npz", ".png", ".tif", ".tiff", ".jpg", ".jpeg"):
        direct = model_dir / f"{sample_id}{suffix}"
        if direct.exists():
            return direct
        nested = model_dir / sample_id / f"pred{suffix}"
        if nested.exists():
            return nested
        nested = model_dir / sample_id / f"prediction{suffix}"
        if nested.exists():
            return nested
    matches = sorted(model_dir.rglob(f"{sample_id}.*"))
    for match in matches:
        if match.suffix.lower() in {".npy", ".npz", ".png", ".tif", ".tiff", ".jpg", ".jpeg"}:
            return match
    return None


def _as_2d(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3:
        # Prefer a single-channel map; otherwise take the first channel.
        if arr.shape[0] in (1, 3, 4) and arr.shape[0] < arr.shape[-1]:
            arr = arr[0] if arr.shape[0] == 1 else arr[0]
        elif arr.shape[-1] in (1, 3, 4):
            arr = arr[..., 0]
        else:
            arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D prediction map, got shape {arr.shape}")
    return arr


def _load_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            for key in ("prob", "probability", "pred", "prediction", "mask", "logits"):
                if key in data:
                    return data[key]
            return data[list(data.keys())[0]]
    if cv2 is None:
        raise ImportError("opencv is required to read image predictions")
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read prediction image: {path}")
    return image


def load_prediction_file(
    path: Path,
    target_size: Tuple[int, int] = (256, 256),
    threshold: float = 0.5,
) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    """Return (binary_mask, probability_or_None, source_description)."""
    array = _as_2d(_load_array(path)).astype(np.float32)

    # Convert logits to probabilities only when values clearly look like logits.
    if array.min() < 0.0 or array.max() > 1.0:
        if array.min() < -0.01 or array.max() > 1.01:
            # A 0/255 image is handled below instead of being sigmoid-ed.
            if array.min() >= 0.0 and array.max() <= 255.0:
                array = array / 255.0
            else:
                array = 1.0 / (1.0 + np.exp(-np.clip(array, -50.0, 50.0)))
    if array.max() > 1.0:
        array = array / 255.0

    probability = np.clip(array, 0.0, 1.0).astype(np.float32)
    if probability.shape != target_size:
        if cv2 is None:
            raise ImportError("opencv is required to resize prediction maps")
        probability = cv2.resize(
            probability,
            (target_size[1], target_size[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    binary = (probability >= float(threshold)).astype(np.uint8)
    return binary, probability, str(path)


class PredictionLoader:
    def __init__(
        self,
        predictions_dir: str | Path = "predictions",
        models: Sequence[str] = DEFAULT_MODELS,
        target_size: Tuple[int, int] = (256, 256),
        threshold: float = 0.5,
    ) -> None:
        self.root = Path(predictions_dir)
        self.models = [canonical_model(m) for m in models]
        self.target_size = target_size
        self.threshold = threshold
        self.model_dirs: Dict[str, Path] = {}
        for model in self.models:
            model_dir = _find_model_dir(self.root, model)
            if model_dir is None:
                raise FileNotFoundError(
                    f"No prediction directory for '{model}' under {self.root}. "
                    "Expected e.g. predictions/att_unet/<sample_id>.npy"
                )
            self.model_dirs[model] = model_dir

    def get(self, model: str, sample_id: str) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
        model = canonical_model(model)
        path = _find_prediction_file(self.model_dirs[model], sample_id)
        if path is None:
            raise FileNotFoundError(
                f"Missing prediction for sample '{sample_id}' in {self.model_dirs[model]}"
            )
        return load_prediction_file(path, target_size=self.target_size, threshold=self.threshold)


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------


def compute_counts(pred_binary: np.ndarray, gt_binary: np.ndarray) -> Dict[str, int]:
    pred = np.asarray(pred_binary).astype(bool)
    gt = np.asarray(gt_binary).astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def metrics_from_counts(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    eps: float = 1e-7,
) -> Dict[str, float]:
    """Metric definitions with the documented edge-case conventions."""
    gt_empty = (tp + fn) == 0
    pred_empty = (tp + fp) == 0

    if gt_empty and pred_empty:
        return {"dice": 1.0, "iou": 1.0, "precision": 1.0, "recall": 1.0}
    if gt_empty and not pred_empty:
        return {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 1.0}
    if not gt_empty and pred_empty:
        return {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}

    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    return {
        "dice": float(np.clip(dice, 0.0, 1.0)),
        "iou": float(np.clip(iou, 0.0, 1.0)),
        "precision": float(np.clip(precision, 0.0, 1.0)),
        "recall": float(np.clip(recall, 0.0, 1.0)),
    }


def mean_std_ci(values: Sequence[float], confidence: float = 0.95) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    mean = float(arr.mean())
    if n == 1:
        return {"n": 1, "mean": mean, "std": 0.0, "ci95_low": mean, "ci95_high": mean}
    std = float(arr.std(ddof=1))
    # Normal-approximation CI for the mean.  Kept explicit so the paper can
    # state exactly how the interval was obtained (n is small: 3 patients).
    z = 1.959963984540054 if confidence == 0.95 else 1.96
    half = z * std / math.sqrt(n)
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci95_low": float(max(0.0, mean - half)),
        "ci95_high": float(min(1.0, mean + half)),
    }


# ---------------------------------------------------------------------------
# Row construction from predictions or from src/evaluate.py CSV
# ---------------------------------------------------------------------------


def _row_from_arrays(
    model: str,
    record: Dict[str, Any],
    gt: np.ndarray,
    pred: np.ndarray,
    source: str,
) -> Dict[str, Any]:
    counts = compute_counts(pred, gt)
    metrics = metrics_from_counts(**counts)
    row: Dict[str, Any] = {
        "model": display_model(model),
        "model_key": canonical_model(model),
        "sample_id": record["sample_id"],
        "patient_id": record["patient_id"],
        "slice_index": int(record["slice_index"]),
        "lesion_present": int(record.get("lesion_present", int(gt.sum() > 0))),
        "tp": counts["tp"],
        "fp": counts["fp"],
        "fn": counts["fn"],
        "tn": counts["tn"],
        "gt_pixels": int(gt.sum()),
        "pred_pixels": int(pred.sum()),
        "prediction_source": source,
    }
    row.update(metrics)
    return row


def rows_from_predictions(
    manifest: Sequence[Dict[str, Any]],
    predictions: PredictionLoader,
    models: Sequence[str],
    ground_truth: GroundTruthLoader,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, record in enumerate(manifest):
        gt = ground_truth.get(record)
        for model in models:
            pred, _, source = predictions.get(canonical_model(model), record["sample_id"])
            rows.append(_row_from_arrays(model, record, gt, pred, source))
        if (idx + 1) % 50 == 0:
            print(f"  ... processed {idx + 1}/{len(manifest)} slices")
    return rows


def _read_metrics_csv(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    return rows


def rows_from_metrics_csv(
    reports_dir: str | Path,
    models: Sequence[str],
    manifest: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Fallback: read reports/<model>/test_metrics_slice_level.csv."""
    reports_dir = Path(reports_dir)
    manifest_by_id = {s["sample_id"]: s for s in manifest}
    rows: List[Dict[str, Any]] = []
    for model in models:
        model_key = canonical_model(model)
        candidates = [
            reports_dir / model_key,
            reports_dir / ("attunet" if model_key == "att_unet" else model_key),
            reports_dir / model,
        ]
        csv_path = next(
            (p / "test_metrics_slice_level.csv" for p in candidates if (p / "test_metrics_slice_level.csv").exists()),
            None,
        )
        if csv_path is None:
            raise FileNotFoundError(
                f"No predictions and no reports/<model>/test_metrics_slice_level.csv for '{model}'"
            )
        for raw in _read_metrics_csv(csv_path):
            record = manifest_by_id.get(raw["sample_id"], {})
            row: Dict[str, Any] = {
                "model": display_model(model),
                "model_key": model_key,
                "sample_id": raw["sample_id"],
                "patient_id": raw.get("patient_id", record.get("patient_id", "")),
                "slice_index": int(raw.get("slice_index", record.get("slice_index", 0))),
                "lesion_present": int(float(raw.get("lesion_present", record.get("lesion_present", 0)))),
                "tp": int(float(raw.get("tp", 0))),
                "fp": int(float(raw.get("fp", 0))),
                "fn": int(float(raw.get("fn", 0))),
                "tn": int(float(raw.get("tn", 0))),
                "gt_pixels": int(float(raw.get("tp", 0))) + int(float(raw.get("fn", 0))),
                "pred_pixels": int(float(raw.get("tp", 0))) + int(float(raw.get("fp", 0))),
                "prediction_source": str(csv_path),
            }
            counts = {k: row[k] for k in ("tp", "fp", "fn", "tn")}
            row.update(metrics_from_counts(**counts))
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Aggregation and output
# ---------------------------------------------------------------------------

DETAIL_FIELDS = [
    "model", "model_key", "sample_id", "patient_id", "slice_index",
    "lesion_present", "tp", "fp", "fn", "tn", "gt_pixels", "pred_pixels",
    "dice", "iou", "precision", "recall", "prediction_source",
]


def patient_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    by_model_patient: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (row["model_key"], row["patient_id"])
        by_model_patient.setdefault(key, []).append(row)

    for (model_key, patient_id), group in sorted(by_model_patient.items()):
        counts = {
            "tp": sum(int(r["tp"]) for r in group),
            "fp": sum(int(r["fp"]) for r in group),
            "fn": sum(int(r["fn"]) for r in group),
            "tn": sum(int(r["tn"]) for r in group),
        }
        metrics = metrics_from_counts(**counts)
        result.append({
            "model": display_model(model_key),
            "model_key": model_key,
            "patient_id": patient_id,
            "n_slices": len(group),
            "n_lesion_slices": sum(1 for r in group if int(r["lesion_present"]) == 1),
            **counts,
            **metrics,
        })
    return result


def summary_rows(rows: Sequence[Dict[str, Any]], patients: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for model_key in sorted({r["model_key"] for r in rows}):
        model_rows = [r for r in rows if r["model_key"] == model_key]
        model_patients = [r for r in patients if r["model_key"] == model_key]
        scopes = {
            "slice_all": model_rows,
            "slice_lesion": [r for r in model_rows if int(r["lesion_present"]) == 1],
        }
        for scope, subset in scopes.items():
            for metric in METRIC_NAMES:
                stats = mean_std_ci([float(r[metric]) for r in subset])
                result.append({
                    "model": display_model(model_key),
                    "model_key": model_key,
                    "level": scope,
                    "metric": metric,
                    **stats,
                })
        for metric in METRIC_NAMES:
            stats = mean_std_ci([float(r[metric]) for r in model_patients])
            result.append({
                "model": display_model(model_key),
                "model_key": model_key,
                "level": "patient",
                "metric": metric,
                **stats,
            })
    return result


def write_csv(path: str | Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    return path


def print_metric_table(metric: str, summary: Sequence[Dict[str, Any]]) -> None:
    print(f"\n== {metric.upper()} SUMMARY ==")
    print(f"{'Model':<34} {'Scope':<14} {'N':>4} {'Mean':>8} {'Std':>8} {'95% CI':>18}")
    for row in summary:
        if row["metric"] != metric:
            continue
        ci = f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}]"
        print(
            f"{row['model']:<34} {row['level']:<14} {row['n']:>4} "
            f"{row['mean']:>8.4f} {row['std']:>8.4f} {ci:>18}"
        )


def run_metric_cli(
    metric: str,
    description: str,
    default_output: str,
    argv: Optional[Sequence[str]] = None,
) -> int:
    """Common CLI used by compute_dsc/iou/precision/recall.py."""
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--manifest", default="data/processed/test_manifest.json")
    parser.add_argument("--predictions_dir", default="predictions")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--raw_data_dir", default=None)
    parser.add_argument("--lesion_label", type=int, default=3)
    parser.add_argument("--output", default=default_output)
    parser.add_argument("--patient_output", default=None)
    parser.add_argument("--summary_output", default=None)
    parser.add_argument(
        "--from_reports",
        action="store_true",
        help="Use reports/<model>/test_metrics_slice_level.csv instead of prediction maps",
    )
    parser.add_argument(
        "--allow_empty_gt",
        action="store_true",
        help="Do not fail when the test set has zero positive GT pixels (debug only)",
    )
    args = parser.parse_args(argv)

    models = [canonical_model(m) for m in args.models]
    manifest = load_manifest(args.manifest)

    use_reports = args.from_reports
    if not use_reports:
        try:
            predictions = PredictionLoader(
                args.predictions_dir,
                models=models,
                threshold=args.threshold,
            )
        except FileNotFoundError as exc:
            print(f"[!] {exc}")
            print("[!] Falling back to reports/<model>/test_metrics_slice_level.csv")
            use_reports = True

    if use_reports:
        rows = rows_from_metrics_csv("reports", models, manifest)
    else:
        ground_truth = GroundTruthLoader(
            lesion_label=args.lesion_label,
            raw_data_dir=args.raw_data_dir,
        )
        rows = rows_from_predictions(manifest, predictions, models, ground_truth)

    total_gt = sum(int(r["gt_pixels"]) for r in rows)
    if total_gt == 0 and not args.allow_empty_gt:
        raise SystemExit(
            "[X] Toàn bộ ground truth đều rỗng (0 pixel tổn thương). "
            "Kết quả sẽ là 1.0 giả tạo. Hãy kiểm tra lesion_label/dataset, "
            "hoặc dùng --allow_empty_gt nếu chỉ muốn kiểm thử script."
        )

    patients = patient_rows(rows)
    summary = summary_rows(rows, patients)

    output = write_csv(args.output, rows, DETAIL_FIELDS)
    file_prefix = "dsc" if metric == "dice" else metric
    patient_output = Path(args.patient_output) if args.patient_output else Path(args.output).with_name(
        f"metrics_{file_prefix}_patient.csv"
    )
    summary_output = Path(args.summary_output) if args.summary_output else Path(args.output).with_name(
        f"metrics_{file_prefix}_summary.csv"
    )
    write_csv(
        patient_output,
        patients,
        ["model", "model_key", "patient_id", "n_slices", "n_lesion_slices",
         "tp", "fp", "fn", "tn", "dice", "iou", "precision", "recall"],
    )
    write_csv(
        summary_output,
        summary,
        ["model", "model_key", "level", "metric", "n", "mean", "std", "ci95_low", "ci95_high"],
    )

    print_metric_table(metric, summary)
    print(f"\n[+] Detail  : {output}")
    print(f"[+] Patient : {patient_output}")
    print(f"[+] Summary : {summary_output}")
    return 0
