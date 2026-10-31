"""Image-chip statistics and tabular feature engineering."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (BAND_IDX, BANDS, CLOUD_PROB_THRESHOLD, CLOUDY_SCL, FEATURE_INFO, ID_COL,
                     REFLECTANCE_SCALE, SCL_CLASSES, TARGETS, GROUP_COL)

EPS = 1e-6


def read_chip(path) -> np.ndarray:
    """Return the chip as float32 (H, W, 14)."""
    import tifffile

    a = tifffile.imread(path)
    if a.ndim == 3 and a.shape[0] == len(BANDS) and a.shape[-1] != len(BANDS):
        a = np.moveaxis(a, 0, -1)
    return a.astype(np.float32)


def clear_mask(chip: np.ndarray) -> np.ndarray:
    scl = chip[..., BAND_IDX["SCL"]].astype(np.int16)
    cld = chip[..., BAND_IDX["CLDPRB"]]
    return (cld < CLOUD_PROB_THRESHOLD) & ~np.isin(scl, CLOUDY_SCL)


def chip_features(chip: np.ndarray) -> dict:
    f: dict = {}
    scl = chip[..., BAND_IDX["SCL"]].astype(np.int16)
    cld = chip[..., BAND_IDX["CLDPRB"]]
    refl = chip[..., :12] * REFLECTANCE_SCALE
    clear = clear_mask(chip)
    f["img_clear_frac"] = float(clear.mean())
    f["img_cldprb_mean"] = float(cld.mean())
    for k, name in SCL_CLASSES.items():
        f[f"scl_{name}"] = float((scl == k).mean())

    px = refl[clear] if clear.sum() >= 100 else refl.reshape(-1, 12)
    for i, b in enumerate(BANDS[:12]):
        v = px[:, i]
        f[f"{b}_mean"], f[f"{b}_std"] = float(v.mean()), float(v.std())
        p50, p95, p99 = np.percentile(v, [50, 95, 99])
        f[f"{b}_p50"], f[f"{b}_p95"], f[f"{b}_p99"] = float(p50), float(p95), float(p99)

    g = lambda b: px[:, BAND_IDX[b]]
    indices = {
        "nbr": (g("B8") - g("B12")) / (g("B8") + g("B12") + EPS),
        "ndbi": (g("B11") - g("B8")) / (g("B11") + g("B8") + EPS),
        "swir_ratio": g("B12") / (g("B11") + EPS),
        "bright": px[:, 1:4].mean(1),
    }
    for name, v in indices.items():
        f[f"{name}_mean"], f[f"{name}_p95"] = float(v.mean()), float(np.percentile(v, 95))

    # Hot-surface / flare proxy on the whole chip (flares can be misflagged as cloud)
    b11, b12 = refl[..., BAND_IDX["B11"]], refl[..., BAND_IDX["B12"]]
    hot = (b12 > 0.5) & (b12 > b11)
    f["hot_frac"], f["hot_cnt"] = float(hot.mean()), float(hot.sum())
    f["b12_max"] = float(b12.max())
    f["b12_b11_p999"] = float(np.percentile(b12 / (b11 + EPS), 99.9))
    return f


def _one(img_dir, sid):
    path = Path(img_dir) / f"{sid}.tif"
    out = {ID_COL: sid}
    if path.exists():
        out.update(chip_features(read_chip(path)))
    return out


def build_image_features(sample_ids, img_dir, cache_path=None, n_jobs=-1) -> pd.DataFrame:
    if cache_path and os.path.exists(cache_path):
        return pd.read_csv(cache_path, dtype={ID_COL: str})
    from joblib import Parallel, delayed

    rows = Parallel(n_jobs=n_jobs, verbose=1)(delayed(_one)(img_dir, s) for s in sample_ids)
    df = pd.DataFrame(rows)
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
    return df


def add_tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {"wind_speed", "wind_direction"} <= set(df.columns):
        r = np.deg2rad(df["wind_direction"])
        df["wind_u"] = -df["wind_speed"] * np.sin(r)
        df["wind_v"] = -df["wind_speed"] * np.cos(r)
        df["wind_dir_sin"], df["wind_dir_cos"] = np.sin(r), np.cos(r)
    if {"temperature", "humidity"} <= set(df.columns):
        df["temp_x_hum"] = df["temperature"] * df["humidity"]
    present = [c for c in FEATURE_INFO if c in df.columns]
    df["n_weather_missing"] = df[present].isna().sum(1)
    return df


def build_features(train, test, img_feats=None):
    if img_feats is not None:
        train = train.merge(img_feats, on=ID_COL, how="left")
        test = test.merge(img_feats, on=ID_COL, how="left")
    train, test = add_tabular_features(train), add_tabular_features(test)
    feats = [c for c in test.columns if c not in [ID_COL, GROUP_COL, *TARGETS]]
    return train, test, feats
