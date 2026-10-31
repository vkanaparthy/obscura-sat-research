"""Train per-target LightGBM with site-grouped CV and write a submission.

Usage:
    python -m obscura.train --config configs/baseline.yaml
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config import ID_COL, TARGETS, find_data_root
from .cv import fold_assignments, site_folds
from .data import load_tables
from .features import build_features, build_image_features
from .metrics import masked_mcrmse


def run(cfg: dict) -> dict:
    import lightgbm as lgb

    root = find_data_root(cfg.get("data_dir"))
    out_dir = Path(cfg.get("output_dir", "outputs")) / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    train, test, sub = load_tables(root)

    img = None
    if cfg.get("use_image_features", True):
        ids = pd.concat([train[ID_COL], test[ID_COL]]).unique()
        img = build_image_features(ids, root / "images", cache_path=cfg.get("image_cache"))
    Xtr, Xte, feats = build_features(train, test, img)

    seed, n_folds = cfg.get("seed", 42), cfg.get("n_folds", 5)
    params = {**cfg["lgb_params"], "random_state": seed, "verbose": -1}
    Y = Xtr[TARGETS].values
    folds = site_folds(Xtr, n_splits=n_folds)
    oof_mean, oof = np.zeros_like(Y), np.zeros_like(Y)
    pred = np.zeros((len(Xte), len(TARGETS)))
    imp = pd.Series(0.0, index=feats)

    for k, (tr, va) in enumerate(folds):
        for j, t in enumerate(TARGETS):
            trm, vam = tr[~np.isnan(Y[tr, j])], va[~np.isnan(Y[va, j])]
            oof_mean[va, j] = np.nanmean(Y[tr, j])
            m = lgb.LGBMRegressor(**params)
            m.fit(Xtr.iloc[trm][feats], Y[trm, j], eval_set=[(Xtr.iloc[vam][feats], Y[vam, j])],
                  callbacks=[lgb.early_stopping(cfg.get("early_stopping", 100), verbose=False)])
            oof[va, j] = m.predict(Xtr.iloc[va][feats])
            pred[:, j] += m.predict(Xte[feats]) / n_folds
            imp += pd.Series(m.feature_importances_, index=feats)
        print(f"fold {k}: mean={masked_mcrmse(Y[va], oof_mean[va])[0]:.4f} "
              f"lgb={masked_mcrmse(Y[va], oof[va])[0]:.4f}")

    s_mean, per_mean = masked_mcrmse(Y, oof_mean)
    s_lgb, per_lgb = masked_mcrmse(Y, oof)

    # Fall back to the train mean for any target where the model did not help in CV
    final = pred.copy()
    fallback = []
    for j, t in enumerate(TARGETS):
        if cfg.get("mean_fallback", True) and per_lgb[t] >= per_mean[t]:
            final[:, j] = np.nanmean(Y[:, j])
            fallback.append(t)

    out = pd.DataFrame({ID_COL: Xte[ID_COL].astype(str)})
    out[TARGETS] = final
    out = sub[[ID_COL]].merge(out, on=ID_COL, how="left")
    assert len(out) == len(sub) and np.isfinite(out[TARGETS].values).all(), "invalid submission"
    out.to_csv(out_dir / "submission.csv", index=False)

    oof_df = pd.DataFrame(oof, columns=[f"oof_{t}" for t in TARGETS])
    oof_df.insert(0, ID_COL, Xtr[ID_COL].values)
    oof_df["fold"] = fold_assignments(Xtr, folds)
    oof_df.to_csv(out_dir / "oof.csv", index=False)
    imp.sort_values(ascending=False).to_csv(out_dir / "feature_importance.csv", header=["gain_count"])

    result = {"name": cfg["name"], "time": time.strftime("%Y-%m-%d %H:%M"), "n_features": len(feats),
              "cv_mcrmse": s_lgb, "cv_mean_baseline": s_mean, "per_target": per_lgb,
              "per_target_mean_baseline": per_mean, "mean_fallback_targets": fallback}
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    with open(Path(cfg.get("output_dir", "outputs")) / "experiments.jsonl", "a") as fh:
        fh.write(json.dumps(result) + "\n")
    print(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    run(cfg)


if __name__ == "__main__":
    main()
