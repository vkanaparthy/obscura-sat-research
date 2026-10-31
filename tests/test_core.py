import numpy as np
import pandas as pd

from obscura.config import BANDS, TARGETS
from obscura.cv import site_folds
from obscura.features import add_tabular_features, chip_features
from obscura.metrics import masked_mcrmse


def test_metric_ignores_missing():
    y = np.array([[1.0, np.nan, 0, 0, 0], [3.0, 2.0, 0, 0, 0]])
    p = np.array([[2.0, 99.0, 0, 0, 0], [3.0, 2.0, 0, 0, 0]])
    score, per = masked_mcrmse(y, p)
    assert per["t_no2"] == 0.0                       # the 99 on a blank target is not scored
    assert np.isclose(per["t_ch4"], np.sqrt(0.5))
    assert np.isclose(score, np.sqrt(0.5) / 5)


def test_site_folds_do_not_leak():
    df = pd.DataFrame({"site_key": [f"s{i % 12}" for i in range(60)]})
    for tr, va in site_folds(df, n_splits=4):
        assert not set(df.site_key[tr]) & set(df.site_key[va])


def test_chip_features_shape_and_finite():
    rng = np.random.default_rng(0)
    chip = rng.integers(0, 4000, (64, 64, len(BANDS))).astype(np.float32)
    chip[..., 12] = rng.integers(0, 12, (64, 64))
    chip[..., 13] = rng.integers(0, 100, (64, 64))
    f = chip_features(chip)
    assert len(f) > 60 and all(np.isfinite(v) for v in f.values())


def test_wind_components():
    df = pd.DataFrame({"wind_speed": [2.0], "wind_direction": [180.0]})  # wind from the south
    out = add_tabular_features(df)
    assert np.isclose(out.wind_v[0], 2.0) and abs(out.wind_u[0]) < 1e-9


def test_targets_list():
    assert TARGETS == ["t_ch4", "t_no2", "t_co", "t_aai", "t_firms"]
