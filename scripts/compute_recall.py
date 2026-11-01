#!/usr/bin/env python3
"""Task EVAL-06 - Recall / Sensitivity.

Recall = TP / (TP + FN). It measures how much of the true lesion area is
captured (false-negative control); this is the most safety-critical metric in
the clinical discussion.

Examples
--------
python scripts/compute_recall.py --predictions_dir predictions
python scripts/compute_recall.py --from_reports
"""

from metric_utils import run_metric_cli

if __name__ == "__main__":
    raise SystemExit(
        run_metric_cli(
            metric="recall",
            description="EVAL-06: Recall / Sensitivity on the independent test set",
            default_output="results/metrics_recall_detail.csv",
        )
    )
