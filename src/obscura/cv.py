"""Site-grouped cross-validation. Test sites never appear in train, so never split a site."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold


def site_folds(df, group_col: str = "site_key", n_splits: int = 5):
    folds = list(GroupKFold(n_splits=n_splits).split(df, groups=df[group_col].values))
    for tr, va in folds:
        assert not set(df[group_col].values[tr]) & set(df[group_col].values[va]), "site leak"
    return folds


def fold_assignments(df, folds) -> np.ndarray:
    fold = np.full(len(df), -1)
    for k, (_, va) in enumerate(folds):
        fold[va] = k
    return fold
