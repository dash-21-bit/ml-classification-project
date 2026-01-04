from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def make_splits(
    df: pd.DataFrame,
    target_col: str = "target",
    test_size: float = 0.2,
    random_state: int = 42,
) -> SplitData:
    """
    Create train/test splits.

    - df: full dataset
    - target_col: name of target column
    - test_size: fraction reserved for test set
    - random_state: ensures reproducibility
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Columns: {df.columns.tolist()}")

    # Features = all columns except target and any label column
    drop_cols = [target_col]
    if "target_label" in df.columns:
        drop_cols.append("target_label")

    X = df.drop(columns=drop_cols)
    y = df[target_col]

    # Stratify keeps class balance similar in train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return SplitData(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
