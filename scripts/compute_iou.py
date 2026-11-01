#!/usr/bin/env python3
"""Task EVAL-04 - Jaccard Index (IoU).

IoU is computed from the same TP/FP/FN counts as DSC; the direct formula
TP / (TP + FP + FN) is used (equivalent to DSC / (2 - DSC)).

Examples
--------
python scripts/compute_iou.py --predictions_dir predictions
python scripts/compute_iou.py --from_reports
"""

from metric_utils import run_metric_cli

if __name__ == "__main__":
    raise SystemExit(
        run_metric_cli(
            metric="iou",
            description="EVAL-04: Jaccard Index (IoU) on the independent test set",
            default_output="results/metrics_iou_detail.csv",
        )
    )
