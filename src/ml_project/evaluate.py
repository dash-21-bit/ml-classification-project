from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


@dataclass
class ModelReport:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    cm: np.ndarray
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray


def evaluate_model(model, X_test, y_test, model_name: str) -> ModelReport:
    """
    Evaluate a trained classifier on the test set.
    Returns metrics, confusion matrix, and ROC curve arrays.
    """
    y_pred = model.predict(X_test)

    # Some models support predict_proba (LR and RF do)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="binary",
        zero_division=0,
    )

    cm = confusion_matrix(y_test, y_pred)

    roc_auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)

    return ModelReport(
        model_name=model_name,
        accuracy=float(acc),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        roc_auc=float(roc_auc),
        cm=cm,
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
    )
