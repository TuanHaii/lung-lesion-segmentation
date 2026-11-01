#!/usr/bin/env python3
"""Task EVAL-05 - Precision (Positive Predictive Value).

Precision = TP / (TP + FP). It measures how many predicted lesion pixels are
actually lesion pixels (false-positive control).

Examples
--------
python scripts/compute_precision.py --predictions_dir predictions
python scripts/compute_precision.py --from_reports
"""

from metric_utils import run_metric_cli

if __name__ == "__main__":
    raise SystemExit(
        run_metric_cli(
            metric="precision",
            description="EVAL-05: Precision on the independent test set",
            default_output="results/metrics_precision_detail.csv",
        )
    )
