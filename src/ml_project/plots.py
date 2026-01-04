from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def save_confusion_matrix_png(cm: np.ndarray, out_path: Path, title: str) -> None:
    """
    Save a confusion matrix heatmap-style image (simple matplotlib).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Put numbers inside cells
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_roc_curve_png(fpr, tpr, out_path: Path, title: str) -> None:
    """
    Save ROC curve as PNG.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")  # baseline diagonal
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
