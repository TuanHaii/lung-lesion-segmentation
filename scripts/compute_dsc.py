#!/usr/bin/env python3
"""Task EVAL-03 - Dice Similarity Coefficient (DSC).

Computes DSC per test slice and per patient on the independent test set, then
reports mean, standard deviation and the 95% confidence interval.

Examples
--------
python scripts/compute_dsc.py --predictions_dir predictions
python scripts/compute_dsc.py --from_reports
"""

from metric_utils import run_metric_cli

if __name__ == "__main__":
    raise SystemExit(
        run_metric_cli(
            metric="dice",
            description="EVAL-03: Dice Similarity Coefficient on the independent test set",
            default_output="results/metrics_dsc_detail.csv",
        )
    )
