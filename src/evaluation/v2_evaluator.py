"""Independent Multi-Level Evaluation Pipeline V2.

Evaluates segmentation predictions with rigorous clinical & statistical standards:
1. Micro Global Pixel-Level Metrics: Aggregated across all test pixels (sum TP, FP, FN, TN).
2. Macro Slice-Level Metrics:
   - All slices (overall average).
   - Lesion-positive slices (non-empty ground truth).
   - Lesion-negative slices (empty ground truth specificity).
3. Patient-Level Metrics: Mean metrics grouped by patient.
4. Source-Level Metrics: Disaggregated for Coronacases vs Radiopaedia.
5. Strict Safety: Zero mock fallbacks. Fails explicitly if ground truth is absent.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


@dataclass
class SliceResult:
    sample_id: str
    patient_id: str
    source: str
    slice_index: int
    lesion_present: int
    tp: int
    fp: int
    fn: int
    tn: int
    dice: float
    iou: float
    precision: float
    recall: float


def compute_slice_confusion(
    pred_bin: np.ndarray,
    gt_bin: np.ndarray,
    smooth: float = 1e-6,
) -> Dict[str, Any]:
    """Calculate confusion matrix and metrics for a single 2D slice."""
    p_flat = (pred_bin > 0.5).reshape(-1).astype(np.int64)
    g_flat = (gt_bin > 0.5).reshape(-1).astype(np.int64)

    tp = int(np.sum((p_flat == 1) & (g_flat == 1)))
    fp = int(np.sum((p_flat == 1) & (g_flat == 0)))
    fn = int(np.sum((p_flat == 0) & (g_flat == 1)))
    tn = int(np.sum((p_flat == 0) & (g_flat == 0)))

    gt_pos = tp + fn
    pred_pos = tp + fp

    if gt_pos == 0:
        if pred_pos == 0:
            dice = 1.0
            iou = 1.0
            precision = 1.0
            recall = 1.0
        else:
            dice = 0.0
            iou = 0.0
            precision = 0.0
            recall = 1.0
    else:
        dice = float((2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth))
        iou = float((tp + smooth) / (tp + fp + fn + smooth))
        precision = float((tp + smooth) / (pred_pos + smooth))
        recall = float((tp + smooth) / (gt_pos + smooth))

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
    }


class EvaluatorV2:
    """Unified multi-level segmentation evaluation engine."""

    def __init__(self, threshold: float = 0.5, smooth: float = 1e-6):
        self.threshold = threshold
        self.smooth = smooth

    def evaluate_slices(self, slice_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate slice-level results across multiple grouping tiers.

        Each item in slice_items must contain:
        - 'sample_id', 'patient_id', 'source', 'slice_index', 'lesion_present'
        - 'pred': 2D binary numpy array or probabilities
        - 'gt': 2D binary ground truth numpy array
        """
        results: List[SliceResult] = []

        for item in slice_items:
            sid = item["sample_id"]
            pid = item["patient_id"]
            source = item["source"]
            sidx = int(item["slice_index"])
            lesion_p = int(item["lesion_present"])

            pred_arr = item["pred"]
            gt_arr = item["gt"]

            if pred_arr is None or gt_arr is None:
                raise ValueError(f"Missing pred or gt for sample '{sid}'! No mock data allowed.")

            pred_bin = (pred_arr >= self.threshold).astype(np.uint8)
            gt_bin = (gt_arr > 0.5).astype(np.uint8)

            conf = compute_slice_confusion(pred_bin, gt_bin, smooth=self.smooth)
            sr = SliceResult(
                sample_id=sid,
                patient_id=pid,
                source=source,
                slice_index=sidx,
                lesion_present=lesion_p,
                tp=conf["tp"],
                fp=conf["fp"],
                fn=conf["fn"],
                tn=conf["tn"],
                dice=conf["dice"],
                iou=conf["iou"],
                precision=conf["precision"],
                recall=conf["recall"],
            )
            results.append(sr)

        if not results:
            return {"error": "Empty results list"}

        return self._aggregate_report(results)

    def _calc_micro(self, res_list: List[SliceResult]) -> Dict[str, float]:
        total_tp = sum(r.tp for r in res_list)
        total_fp = sum(r.fp for r in res_list)
        total_fn = sum(r.fn for r in res_list)
        total_tn = sum(r.tn for r in res_list)

        denom_dice = 2.0 * total_tp + total_fp + total_fn
        dice = (2.0 * total_tp + self.smooth) / (denom_dice + self.smooth) if denom_dice > 0 else 1.0

        denom_iou = total_tp + total_fp + total_fn
        iou = (total_tp + self.smooth) / (denom_iou + self.smooth) if denom_iou > 0 else 1.0

        denom_prec = total_tp + total_fp
        precision = (total_tp + self.smooth) / (denom_prec + self.smooth) if denom_prec > 0 else 1.0

        denom_rec = total_tp + total_fn
        recall = (total_tp + self.smooth) / (denom_rec + self.smooth) if denom_rec > 0 else 1.0

        return {
            "micro_tp": total_tp,
            "micro_fp": total_fp,
            "micro_fn": total_fn,
            "micro_tn": total_tn,
            "micro_dice": float(dice),
            "micro_iou": float(iou),
            "micro_precision": float(precision),
            "micro_recall": float(recall),
        }

    def _calc_macro(self, res_list: List[SliceResult]) -> Dict[str, float]:
        n = len(res_list)
        if n == 0:
            return {"count": 0, "macro_dice": 0.0, "macro_iou": 0.0, "macro_precision": 0.0, "macro_recall": 0.0}
        return {
            "count": n,
            "macro_dice": float(np.mean([r.dice for r in res_list])),
            "macro_iou": float(np.mean([r.iou for r in res_list])),
            "macro_precision": float(np.mean([r.precision for r in res_list])),
            "macro_recall": float(np.mean([r.recall for r in res_list])),
        }

    def _aggregate_report(self, results: List[SliceResult]) -> Dict[str, Any]:
        # 1. Overall
        overall_micro = self._calc_micro(results)
        overall_macro = self._calc_macro(results)

        # 2. Lesion presence partition
        lesion_slices = [r for r in results if r.lesion_present == 1]
        normal_slices = [r for r in results if r.lesion_present == 0]

        lesion_macro = self._calc_macro(lesion_slices)
        normal_macro = self._calc_macro(normal_slices)

        # 3. Source partition
        coronacases_slices = [r for r in results if r.source == "coronacases"]
        radiopaedia_slices = [r for r in results if r.source == "radiopaedia"]

        source_metrics = {
            "coronacases": {
                **self._calc_micro(coronacases_slices),
                **self._calc_macro(coronacases_slices),
            },
            "radiopaedia": {
                **self._calc_micro(radiopaedia_slices),
                **self._calc_macro(radiopaedia_slices),
            },
        }

        # 4. Patient partition
        patient_groups: Dict[str, List[SliceResult]] = {}
        for r in results:
            patient_groups.setdefault(r.patient_id, []).append(r)

        patient_metrics: Dict[str, Any] = {}
        for pid, p_slices in sorted(patient_groups.items()):
            patient_metrics[pid] = {
                "source": p_slices[0].source,
                "total_slices": len(p_slices),
                "lesion_slices": sum(1 for s in p_slices if s.lesion_present == 1),
                **self._calc_macro(p_slices),
            }

        return {
            "overall": {
                "total_slices": len(results),
                "micro": overall_micro,
                "macro_all": overall_macro,
                "macro_lesion_positive": lesion_macro,
                "macro_lesion_negative": normal_macro,
            },
            "by_source": source_metrics,
            "by_patient": patient_metrics,
            "slice_details": [asdict(r) for r in results],
        }

    @torch.no_grad()
    def evaluate_dataloader(
        self,
        model: nn.Module,
        dataloader: Any,
        device: torch.device,
    ) -> Dict[str, Any]:
        """Run model inference over DataLoader and evaluate results."""
        model.eval()
        slice_items: List[Dict[str, Any]] = []

        for images, masks, metadata in dataloader:
            images = images.to(device, dtype=torch.float32)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            masks_np = masks.cpu().numpy()

            b_size = images.size(0)
            for b in range(b_size):
                p_2d = probs[b, 0]
                m_2d = masks_np[b, 0]
                slice_items.append({
                    "sample_id": metadata["sample_id"][b] if isinstance(metadata["sample_id"], (list, tuple)) else metadata["sample_id"],
                    "patient_id": metadata["patient_id"][b] if isinstance(metadata["patient_id"], (list, tuple)) else metadata["patient_id"],
                    "source": metadata["source"][b] if isinstance(metadata["source"], (list, tuple)) else metadata["source"],
                    "slice_index": int(metadata["slice_index"][b]),
                    "lesion_present": int(metadata["lesion_present"][b]),
                    "pred": p_2d,
                    "gt": m_2d,
                })

        return self.evaluate_slices(slice_items)
