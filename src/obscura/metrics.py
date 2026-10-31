"""Masked mean column-wise RMSE, the competition metric."""
from __future__ import annotations

import numpy as np

from .config import TARGETS


def masked_mcrmse(y_true, y_pred, targets=TARGETS):
    """Per target, RMSE over rows where y_true is not NaN; then the mean of those RMSEs.

    Returns (score, {target: rmse}).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    per = {}
    for j, t in enumerate(targets):
        m = ~np.isnan(y_true[:, j])
        per[t] = float(np.sqrt(np.mean((y_true[m, j] - y_pred[m, j]) ** 2))) if m.any() else float("nan")
    return float(np.nanmean(list(per.values()))), per
