# OBSCURA satellite research

*Observing Blocked Satellite Columns Using Remote-sensing and Auxiliary data*

Code for the [OBSCURA: Satellite AI Research Challenge](https://obscura.devpost.com/) ([Kaggle](https://www.kaggle.com/competitions/obscura)): predict five atmospheric targets (CH4, NO2, CO, aerosol index, fire/flare intensity) around industrial sites from Sentinel-2 chips plus weather and terrain features.

- Metric: masked mean column-wise RMSE (lower is better)
- Validation: GroupKFold on `site_key`, since test facilities never appear in train
- Data definitions: [`docs/data_dictionary.md`](docs/data_dictionary.md)

## Layout

```
configs/            experiment configs (YAML)
data/               local data (git-ignored), see data/README.md
docs/               data dictionary, research log, report outline
notebooks/          obscura_baseline.ipynb, self-contained for Kaggle
outputs/            per-experiment submission, OOF, metrics (git-ignored)
scripts/            download_data.sh, submit.sh
src/obscura/        config, data loading, features, CV, metric, training
tests/              unit tests
```

## Quick start

```bash
pip install -e ".[dev]"
bash scripts/download_data.sh                  # needs ~/.kaggle/kaggle.json
python -m obscura.train --config configs/baseline.yaml
bash scripts/submit.sh lgb_baseline "lgb baseline"
pytest
```

Each run writes `outputs/<name>/{submission.csv, oof.csv, metrics.json, feature_importance.csv}` and appends a line to `outputs/experiments.jsonl`. Copy results into `docs/research_log.md`.

On Kaggle, upload `notebooks/obscura_baseline.ipynb`, add the competition data, and run all. It needs no install step.

## Baseline

Per-target LightGBM trained only on rows where that target was measured, using the tabular features, wind components, and cloud-masked chip statistics (per-band percentiles, SCL class fractions, SWIR hot-pixel flare proxies). Targets where it doesn't beat the train mean in CV fall back to the mean.

## Ideas to try

- Multi-task model sharing information across the five targets
- CNN / pretrained Sentinel-2 encoder on the chips (e.g. SSL4EO or SatMAE weights; check the rules on external models)
- Stronger regularization and target shrinkage for site/time shift
- Per-target feature selection; robust losses (Huber) for noisy targets

## Key dates (PKT; Pacific in brackets)

- Devpost submissions open: Nov 18, 2026
- Final deadline: Nov 24, 2026, 8:00 PM PKT (7:00 AM PT)

## Rules to remember

Don't commit or publish the dataset. Don't try to identify sites, locations or dates, and don't use outside satellite, weather or methane data.
