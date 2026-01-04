from __future__ import annotations

from pathlib import Path

from ml_project.dataset import load_dataset
from ml_project.preprocess import make_splits
from ml_project.train import train_models
from ml_project.evaluate import evaluate_model
from ml_project.plots import save_confusion_matrix_png, save_roc_curve_png
from ml_project.export import export_metrics_csv

DATA = Path("data/raw/breast_cancer.csv")
METRICS_CSV = Path("reports/metrics.csv")

def main() -> None:
    # 1) Load
    df = load_dataset(DATA)

    # 2) Split
    split = make_splits(df, target_col="target", test_size=0.2, random_state=42)

    # 3) Train
    models = train_models(split.X_train, split.y_train, random_state=42)

    # 4) Evaluate both models
    lr_report = evaluate_model(models.logistic_regression, split.X_test, split.y_test, "LogisticRegression")
    rf_report = evaluate_model(models.random_forest, split.X_test, split.y_test, "RandomForest")

    # 5) Export metrics
    export_metrics_csv([lr_report, rf_report], METRICS_CSV)

    # 6) Save plots (PNG)
    save_confusion_matrix_png(lr_report.cm, Path("reports/figures/lr_confusion_matrix.png"), "LR Confusion Matrix")
    save_roc_curve_png(lr_report.fpr, lr_report.tpr, Path("reports/figures/lr_roc_curve.png"), "LR ROC Curve")

    save_confusion_matrix_png(rf_report.cm, Path("reports/figures/rf_confusion_matrix.png"), "RF Confusion Matrix")
    save_roc_curve_png(rf_report.fpr, rf_report.tpr, Path("reports/figures/rf_roc_curve.png"), "RF ROC Curve")

    # 7) Print final results
    print("\n=== Metrics saved ===")
    print(f"- {METRICS_CSV}")
    print("\n=== Figures saved ===")
    print("- reports/figures/lr_confusion_matrix.png")
    print("- reports/figures/lr_roc_curve.png")
    print("- reports/figures/rf_confusion_matrix.png")
    print("- reports/figures/rf_roc_curve.png")

    print("\nDone ✅")

if __name__ == "__main__":
    main()
