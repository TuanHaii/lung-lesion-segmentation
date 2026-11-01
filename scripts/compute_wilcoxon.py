#!/usr/bin/env python3
"""Task EVAL-07 - Quantitative comparison table + statistical tests.

Reads the four detail CSVs produced by EVAL-03 .. EVAL-06, builds the model
comparison table (DSC, IoU, Precision, Recall), and runs a paired Wilcoxon
signed-rank test and a paired t-test between Attention U-Net and each baseline.

The primary scope is lesion-containing slices (GT > 0), following the
project's dual-reporting convention.  A second CSV for all slices is also
written for the supplementary material.

Examples
--------
python scripts/compute_dsc.py --predictions_dir predictions
python scripts/compute_iou.py --predictions_dir predictions
python scripts/compute_precision.py --predictions_dir predictions
python scripts/compute_recall.py --predictions_dir predictions
python scripts/compute_wilcoxon.py
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "scripts"))

from metric_utils import (  # noqa: E402
    DEFAULT_MODELS,
    METRIC_NAMES,
    canonical_model,
    display_model,
    mean_std_ci,
    write_csv,
)

try:
    from scipy.stats import ttest_rel, wilcoxon
except ImportError:  # pragma: no cover
    ttest_rel = None
    wilcoxon = None

DETAIL_FILES = {
    "dice": "results/metrics_dsc_detail.csv",
    "iou": "results/metrics_iou_detail.csv",
    "precision": "results/metrics_precision_detail.csv",
    "recall": "results/metrics_recall_detail.csv",
}


def read_detail(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the corresponding compute_*.py script first."
        )
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def index_by_sample(rows: Sequence[Dict[str, str]], metric: str) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for row in rows:
        sample_id = row["sample_id"]
        try:
            result[sample_id] = float(row[metric])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid value for {metric} in sample {sample_id}") from exc
    return result


def paired_arrays(
    values_a: Dict[str, float],
    values_b: Dict[str, float],
) -> Tuple[List[float], List[float]]:
    common = sorted(set(values_a) & set(values_b))
    missing = (set(values_a) ^ set(values_b))
    if missing:
        print(f"[!] {len(missing)} sample_ids are not paired and were ignored.")
    return [values_a[s] for s in common], [values_b[s] for s in common]


def safe_wilcoxon(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    if wilcoxon is None or len(a) < 2:
        return float("nan"), float("nan")
    differences = [x - y for x, y in zip(a, b)]
    if all(abs(d) < 1e-12 for d in differences):
        return 0.0, 1.0
    try:
        stat, p_value = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        return float(stat), float(p_value)
    except ValueError as exc:
        print(f"[!] Wilcoxon failed: {exc}")
        return float("nan"), float("nan")


def safe_ttest(a: Sequence[float], b: Sequence[float]) -> float:
    if ttest_rel is None or len(a) < 2:
        return float("nan")
    differences = [x - y for x, y in zip(a, b)]
    if all(abs(d) < 1e-12 for d in differences):
        return 1.0
    try:
        _, p_value = ttest_rel(a, b)
        return float(p_value)
    except ValueError:
        return float("nan")


def build_table(scope: str, details: Dict[str, List[Dict[str, str]]]) -> List[Dict[str, object]]:
    # metric -> model -> sample -> value
    values: Dict[str, Dict[str, Dict[str, float]]] = {}
    patient_sets: Dict[str, set] = {}
    sample_sets: Dict[str, set] = {}

    for metric in METRIC_NAMES:
        values[metric] = {}
        for row in details[metric]:
            if scope == "lesion_slices" and int(float(row["lesion_present"])) != 1:
                continue
            model_key = canonical_model(row.get("model_key", row["model"]))
            values[metric].setdefault(model_key, {})[row["sample_id"]] = float(row[metric])
            patient_sets.setdefault(model_key, set()).add(row["patient_id"])
            sample_sets.setdefault(model_key, set()).add(row["sample_id"])

    table: List[Dict[str, object]] = []
    proposed = "att_unet"
    for model_key in sorted(values["dice"], key=lambda m: (m != proposed, m)):
        row: Dict[str, object] = {
            "scope": scope,
            "model": display_model(model_key),
            "model_key": model_key,
            "n_slices": len(sample_sets.get(model_key, set())),
            "n_patients": len(patient_sets.get(model_key, set())),
        }
        for metric in METRIC_NAMES:
            prefix = "dsc" if metric == "dice" else metric
            stats = mean_std_ci(list(values[metric][model_key].values()))
            row[f"{prefix}_mean"] = round(stats["mean"], 6)
            row[f"{prefix}_std"] = round(stats["std"], 6)
            row[f"{prefix}_ci95_low"] = round(stats["ci95_low"], 6)
            row[f"{prefix}_ci95_high"] = round(stats["ci95_high"], 6)
            row[f"{prefix}_n"] = stats["n"]

        # Paired tests against the proposed model.
        for metric in METRIC_NAMES:
            prefix = "dsc" if metric == "dice" else metric
            if model_key == proposed:
                row[f"wilcoxon_stat_{prefix}"] = ""
                row[f"wilcoxon_p_{prefix}"] = ""
                row[f"ttest_p_{prefix}"] = ""
                row[f"significant_{prefix}"] = ""
                continue
            a, b = paired_arrays(values[metric][proposed], values[metric][model_key])
            stat, p_w = safe_wilcoxon(a, b)
            p_t = safe_ttest(a, b)
            row[f"wilcoxon_stat_{prefix}"] = "" if math.isnan(stat) else round(stat, 4)
            row[f"wilcoxon_p_{prefix}"] = "" if math.isnan(p_w) else f"{p_w:.6g}"
            row[f"ttest_p_{prefix}"] = "" if math.isnan(p_t) else f"{p_t:.6g}"
            row[f"significant_{prefix}"] = (
                "" if math.isnan(p_w) else ("yes" if p_w < 0.05 else "no")
            )
        table.append(row)
    return table


def latex_table(rows: Sequence[Dict[str, object]], scope: str) -> str:
    lines = [
        "% Auto-generated by scripts/compute_wilcoxon.py (EVAL-07)",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Quantitative comparison on the independent test set"
        + (" (lesion-containing slices)" if scope == "lesion_slices" else " (all slices)")
        + "}",
        "\\label{tab:quantitative_comparison_" + scope + "}",
        "\\begin{tabular}{lcccc}",
        "\\hline",
        "\\textbf{Model} & \\textbf{DSC} & \\textbf{IoU} & \\textbf{Precision} & \\textbf{Recall} \\\\",
        "\\hline",
    ]
    for row in rows:
        cells = []
        for metric in METRIC_NAMES:
            prefix = "dsc" if metric == "dice" else metric
            mean = float(row[f"{prefix}_mean"])
            std = float(row[f"{prefix}_std"])
            cells.append(f"{mean:.4f} $\\pm$ {std:.4f}")
        lines.append(
            f"{row['model']} & " + " & ".join(cells) + " \\\\"
        )
    lines += ["\\hline", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="EVAL-07: comparison table + Wilcoxon/t-test")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument(
        "--output",
        default="results/quantitative_comparison_table.csv",
    )
    parser.add_argument(
        "--all_slices_output",
        default="results/quantitative_comparison_table_all_slices.csv",
    )
    parser.add_argument(
        "--latex_output",
        default="results/quantitative_comparison_table.tex",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    details = {
        metric: read_detail(results_dir / Path(path).name)
        for metric, path in DETAIL_FILES.items()
    }

    primary = build_table("lesion_slices", details)
    all_slices = build_table("all_slices", details)

    fields = [
        "scope", "model", "model_key", "n_slices", "n_patients",
        "dsc_mean", "dsc_std", "dsc_ci95_low", "dsc_ci95_high", "dsc_n",
        "iou_mean", "iou_std", "iou_ci95_low", "iou_ci95_high", "iou_n",
        "precision_mean", "precision_std", "precision_ci95_low", "precision_ci95_high", "precision_n",
        "recall_mean", "recall_std", "recall_ci95_low", "recall_ci95_high", "recall_n",
    ]
    for metric in METRIC_NAMES:
        prefix = "dsc" if metric == "dice" else metric
        fields += [
            f"wilcoxon_stat_{prefix}",
            f"wilcoxon_p_{prefix}",
            f"ttest_p_{prefix}",
            f"significant_{prefix}",
        ]

    write_csv(args.output, primary, fields)
    write_csv(args.all_slices_output, all_slices, fields)

    latex_path = Path(args.latex_output)
    latex_path.parent.mkdir(parents=True, exist_ok=True)
    latex_path.write_text(latex_table(primary, "lesion_slices"), encoding="utf-8")

    print(f"\n== EVAL-07 QUANTITATIVE COMPARISON (lesion-containing slices) ==")
    header = f"{'Model':<34} {'DSC':>17} {'IoU':>17} {'Precision':>17} {'Recall':>17} {'p(DSC)':>10}"
    print(header)
    for row in primary:
        p_value = row.get("wilcoxon_p_dsc", "")
        print(
            f"{row['model']:<34} "
            f"{row['dsc_mean']:>8.4f} ± {row['dsc_std']:<6.4f} "
            f"{row['iou_mean']:>8.4f} ± {row['iou_std']:<6.4f} "
            f"{row['precision_mean']:>8.4f} ± {row['precision_std']:<6.4f} "
            f"{row['recall_mean']:>8.4f} ± {row['recall_std']:<6.4f} "
            f"{str(p_value):>10}"
        )
    print(f"\n[+] Primary table : {args.output}")
    print(f"[+] All slices    : {args.all_slices_output}")
    print(f"[+] LaTeX table   : {args.latex_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
