from __future__ import annotations

from pathlib import Path
import pandas as pd
from sklearn.datasets import load_breast_cancer

OUT = Path("data/raw/breast_cancer.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

data = load_breast_cancer(as_frame=True)
df = data.frame.copy()

# The target column is called "target" in sklearn's frame
# We'll map it to labels to make it human readable
df["target_label"] = df["target"].map({0: "malignant", 1: "benign"})

df.to_csv(OUT, index=False)
print(f"Saved dataset to: {OUT} with shape {df.shape}")
