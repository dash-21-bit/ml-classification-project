from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from ml_project.evaluate import ModelReport


def export_metrics_csv(reports: Iterable[ModelReport], out_path: Path) -> None:
    """
    Export model metrics to CSV.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in reports:
        rows.append({
            "model": r.model_name,
            "accuracy": r.accuracy,
            "precision": r.precision,
            "recall": r.recall,
            "f1": r.f1,
            "roc_auc": r.roc_auc,
        })

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
