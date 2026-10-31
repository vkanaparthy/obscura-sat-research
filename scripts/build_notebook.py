import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# OBSCURA baseline: site-grouped LightGBM on tabular + image features

**Pipeline**
1. Load data and the data dictionary (from the competition README).
2. Sanity-check the data: missing targets, missing weather, site counts.
3. Extract cloud-masked statistics from each 14-band Sentinel-2 chip.
4. Validate with `GroupKFold` on `site_key` (test sites never appear in train).
5. Train one LightGBM per target on rows where that target was measured.
6. Compare against the train-mean baseline with masked MCRMSE, then write `submission.csv`.

Rules reminder: no attempts to identify sites, locations or dates, and no outside satellite / weather / methane data.
""")

md(open("docs/data_dictionary.md").read().replace("# OBSCURA data dictionary","## Data definitions",1))

code(r"""
import os, glob, warnings
import numpy as np, pandas as pd
import tifffile
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")

def find_root():
    for c in [os.environ.get("OBSCURA_DIR", ""), "/kaggle/input/obscura", *glob.glob("/kaggle/input/*")]:
        if c and os.path.exists(os.path.join(c, "train.csv")):
            return c
    hits = glob.glob("/kaggle/input/**/train.csv", recursive=True)
    if hits:
        return os.path.dirname(hits[0])
    raise FileNotFoundError("train.csv not found: add the OBSCURA dataset to the notebook")

ROOT = find_root()
IMG_DIR = os.path.join(ROOT, "images")
OUT_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
TARGETS = ["t_ch4", "t_no2", "t_co", "t_aai", "t_firms"]
SEED, N_FOLDS = 42, 5
print("data:", ROOT)
""")

code(r"""
# Machine-readable data dictionary (same content as the table above)
BANDS = ["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B11","B12","SCL","CLDPRB"]
SCL_CLASSES = {0:"no_data",1:"saturated",2:"dark_shadow",3:"cloud_shadow",4:"vegetation",5:"bare_soil",
               6:"water",7:"unclassified",8:"cloud_med",9:"cloud_high",10:"cirrus",11:"snow_ice"}
TARGET_INFO = {
    "t_ch4":   "XCH4 enhancement (site mean - background median), z-score",
    "t_no2":   "Tropospheric NO2 column enhancement, z-score",
    "t_co":    "CO column enhancement, z-score",
    "t_aai":   "Absorbing aerosol index over site window, z-score",
    "t_firms": "Fire/flare thermal intensity, z-score; no detection = real low value",
}
FEATURE_INFO = {
    "month_sin": ("-", "cyclic month"), "month_cos": ("-", "cyclic month"),
    "temperature": ("degC", "ERA5-Land 2 m air temp, daily mean"),
    "humidity": ("%", "relative humidity, daily mean"),
    "wind_speed": ("m/s", "10 m wind near overpass"),
    "wind_direction": ("deg", "direction wind blows from, near overpass"),
    "precipitation": ("m/day", "total precipitation, daily sum"),
    "solar_rad": ("MJ/m2/day", "downward surface solar radiation"),
    "elevation": ("m", "GLO-30 mean elevation over chip"),
    "slope": ("deg", "mean terrain slope over chip"),
    "s2_cloud_frac": ("0-1", "fraction of chip with cloud prob >= 40"),
}
SPECTRAL_INDICES = ["ndvi","evi","savi","ndre","ndwi","mndwi","awei","nmdi","nbri","nbr2",
                    "bsi","bi","swir_ratio","ndii","hsfi","ohi","b11_mean","b12_mean"]
for c in SPECTRAL_INDICES:
    FEATURE_INFO[c] = ("-", "spectral index, mean over cloud-free chip pixels")

train = pd.read_csv(os.path.join(ROOT, "train.csv"), dtype={"sample_id": str, "site_key": str})
test  = pd.read_csv(os.path.join(ROOT, "test.csv"),  dtype={"sample_id": str})
sub   = pd.read_csv(os.path.join(ROOT, "sample_submission.csv"), dtype={"sample_id": str})

BASE_FEATS = [c for c in test.columns if c != "sample_id"]
unknown = [c for c in BASE_FEATS if c not in FEATURE_INFO]
print(train.shape, test.shape, "| sites:", train.site_key.nunique())
print("features:", len(BASE_FEATS), "| spectral/other (not in dictionary table):", unknown)
""")

code(r"""
# Data checks
summary = pd.DataFrame({
    "measured": train[TARGETS].notna().sum(),
    "missing_%": (train[TARGETS].isna().mean() * 100).round(1),
    "mean": train[TARGETS].mean().round(3),
    "std": train[TARGETS].std().round(3),
})
display(summary)
print("rows per site:"); display(train.site_key.value_counts().describe())
miss = pd.concat([train[BASE_FEATS].isna().mean(), test[BASE_FEATS].isna().mean()], axis=1, keys=["train", "test"])
display((miss[miss.max(axis=1) > 0] * 100).round(1))
display(train[TARGETS].corr().round(2))
""")

md(r"""
## Image features
For each chip: build a clear-sky mask (CLDPRB < 40 and SCL not cloud / shadow / cirrus / no-data), then take per-band means, std and high percentiles over clear pixels, SCL class fractions, and a few indices. The B12/B11 "hot pixel" counts are a rough proxy for flares (`t_firms`). Results are cached to `img_feats.csv`.
""")

code(r"""
CLOUDY_SCL = [0, 1, 3, 8, 9, 10]
B = {b: i for i, b in enumerate(BANDS)}

def chip_features(sid):
    path = os.path.join(IMG_DIR, f"{sid}.tif")
    f = {"sample_id": sid}
    if not os.path.exists(path):
        return f
    a = tifffile.imread(path)
    if a.ndim == 3 and a.shape[0] == 14 and a.shape[-1] != 14:
        a = np.moveaxis(a, 0, -1)                        # -> H, W, 14
    a = a.astype(np.float32)
    scl, cld = a[..., B["SCL"]].astype(np.int16), a[..., B["CLDPRB"]]
    refl = a[..., :12] * 1e-4
    clear = (cld < 40) & ~np.isin(scl, CLOUDY_SCL)
    f["img_clear_frac"] = clear.mean()
    f["img_cldprb_mean"] = cld.mean()
    for k, name in SCL_CLASSES.items():
        f[f"scl_{name}"] = (scl == k).mean()
    px = refl[clear] if clear.sum() >= 100 else refl.reshape(-1, 12)
    for i, b in enumerate(BANDS[:12]):
        v = px[:, i]
        f[f"{b}_mean"], f[f"{b}_std"] = v.mean(), v.std()
        f[f"{b}_p50"], f[f"{b}_p95"], f[f"{b}_p99"] = np.percentile(v, [50, 95, 99])
    eps = 1e-6
    g = lambda b: px[:, B[b]]
    for name, val in {
        "nbr":  (g("B8") - g("B12")) / (g("B8") + g("B12") + eps),
        "ndbi": (g("B11") - g("B8")) / (g("B11") + g("B8") + eps),
        "swir_ratio": g("B12") / (g("B11") + eps),
        "bright": px[:, 1:4].mean(1),
    }.items():
        f[f"{name}_mean"], f[f"{name}_p95"] = val.mean(), np.percentile(val, 95)
    # flare / hot-surface proxy over the whole chip (flares may be flagged as cloud)
    b11, b12 = refl[..., B["B11"]], refl[..., B["B12"]]
    hot = (b12 > 0.5) & (b12 > b11)
    f["hot_frac"], f["hot_cnt"] = hot.mean(), float(hot.sum())
    f["b12_max"] = float(b12.max())
    f["b12_b11_p999"] = float(np.percentile(b12 / (b11 + eps), 99.9))
    return f

cache = os.path.join(OUT_DIR, "img_feats.csv")
ids = pd.concat([train.sample_id, test.sample_id]).unique()
if os.path.exists(cache):
    img = pd.read_csv(cache, dtype={"sample_id": str})
else:
    img = pd.DataFrame(Parallel(n_jobs=-1, verbose=1)(delayed(chip_features)(s) for s in ids))
    img.to_csv(cache, index=False)
print(img.shape)
""")

code(r"""
def add_features(df):
    df = df.merge(img, on="sample_id", how="left")
    if {"wind_speed", "wind_direction"} <= set(df.columns):
        r = np.deg2rad(df.wind_direction)
        df["wind_u"], df["wind_v"] = -df.wind_speed * np.sin(r), -df.wind_speed * np.cos(r)
        df["wind_dir_sin"], df["wind_dir_cos"] = np.sin(r), np.cos(r)
    if {"temperature", "humidity"} <= set(df.columns):
        df["temp_x_hum"] = df.temperature * df.humidity
    df["n_weather_missing"] = df[[c for c in FEATURE_INFO if c in df]].isna().sum(1)
    return df

Xtr, Xte = add_features(train), add_features(test)
FEATS = [c for c in Xte.columns if c not in ["sample_id", "site_key"] + TARGETS]
print(len(FEATS), "features")
""")

md(r"""
## Validation: masked MCRMSE with site-grouped folds
""")

code(r"""
def masked_mcrmse(y_true, y_pred):
    # y_true may contain NaN (not measured); those cells are ignored
    scores = {}
    for j, t in enumerate(TARGETS):
        m = ~np.isnan(y_true[:, j])
        scores[t] = float(np.sqrt(np.mean((y_true[m, j] - y_pred[m, j]) ** 2))) if m.any() else np.nan
    return float(np.nanmean(list(scores.values()))), scores

PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=15, min_child_samples=40,
              feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=5.0, n_estimators=2000, verbose=-1, random_state=SEED)

Y = Xtr[TARGETS].values
groups = Xtr.site_key.values
folds = list(GroupKFold(n_splits=N_FOLDS).split(Xtr, groups=groups))
oof_mean = np.zeros_like(Y); oof_lgb = np.zeros_like(Y)
test_pred = np.zeros((len(Xte), len(TARGETS)))
best_iters = {t: [] for t in TARGETS}
importance = pd.Series(0.0, index=FEATS)

for k, (tr, va) in enumerate(folds):
    for j, t in enumerate(TARGETS):
        trm = tr[~np.isnan(Y[tr, j])]
        vam = va[~np.isnan(Y[va, j])]
        oof_mean[va, j] = np.nanmean(Y[tr, j])
        model = lgb.LGBMRegressor(**PARAMS)
        model.fit(Xtr.iloc[trm][FEATS], Y[trm, j],
                  eval_set=[(Xtr.iloc[vam][FEATS], Y[vam, j])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        oof_lgb[va, j] = model.predict(Xtr.iloc[va][FEATS])
        test_pred[:, j] += model.predict(Xte[FEATS]) / N_FOLDS
        best_iters[t].append(model.best_iteration_)
        importance += pd.Series(model.feature_importances_, index=FEATS)
    print(f"fold {k}: mean={masked_mcrmse(Y[va], oof_mean[va])[0]:.4f}  lgb={masked_mcrmse(Y[va], oof_lgb[va])[0]:.4f}")

s_mean, per_mean = masked_mcrmse(Y, oof_mean)
s_lgb, per_lgb = masked_mcrmse(Y, oof_lgb)
display(pd.DataFrame({"train_mean": per_mean, "lightgbm": per_lgb}).T.assign(MCRMSE=[s_mean, s_lgb]).round(4))
""")

md(r"""
Early stopping on the validation fold makes the OOF score slightly optimistic. The per-target table tells you where the model beats the mean; for any target where it doesn't, the cell below falls back to the train mean.
""")

code(r"""
final = test_pred.copy()
for j, t in enumerate(TARGETS):
    if per_lgb[t] >= per_mean[t]:
        final[:, j] = np.nanmean(Y[:, j])
        print(f"{t}: LightGBM did not beat the mean in CV -> using train mean")

out = pd.DataFrame({"sample_id": Xte.sample_id.astype(str)})
out[TARGETS] = final
out = sub[["sample_id"]].merge(out, on="sample_id", how="left")
assert len(out) == len(sub) and np.isfinite(out[TARGETS].values).all()
out.to_csv(os.path.join(OUT_DIR, "submission.csv"), index=False)
print(out.head()); print("saved submission.csv")

display(importance.sort_values(ascending=False).head(25).round(0))
""")

nb["cells"] = C
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}
nbf.write(nb, "notebooks/obscura_baseline.ipynb")
print("written")
