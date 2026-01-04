from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """
    Load the dataset CSV into a pandas DataFrame.
    """
    return pd.read_csv(csv_path)
