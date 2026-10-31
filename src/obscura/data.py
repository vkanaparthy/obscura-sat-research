"""Loading the competition CSVs. sample_id is always kept as text."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import GROUP_COL, ID_COL


def load_tables(root: Path):
    root = Path(root)
    train = pd.read_csv(root / "train.csv", dtype={ID_COL: str, GROUP_COL: str})
    test = pd.read_csv(root / "test.csv", dtype={ID_COL: str})
    sub = pd.read_csv(root / "sample_submission.csv", dtype={ID_COL: str})
    return train, test, sub


def base_feature_columns(test: pd.DataFrame) -> list[str]:
    return [c for c in test.columns if c != ID_COL]
