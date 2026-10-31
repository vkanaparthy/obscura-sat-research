# Research write-up outline (Devpost PDF)

Due with the Devpost project: window opens Nov 18, 2026; deadline **Nov 24, 2026, 8:00 PM PKT (7:00 AM Pacific)**. No video required.

1. **Project and team** — title, names, Kaggle team name, Kaggle profile URL(s).
2. **Problem understanding** — the five targets, z-scoring, blank ≠ zero, site/time shift between train and test.
3. **Data exploration** — missing-target rates, target correlations, missing weather, cloud cover, chip observations.
4. **Baselines** — train mean; LightGBM on tabular only; + image stats.
5. **Experiments and methodology** — features, models, hyperparameters (pull from `research_log.md`).
6. **Validation and results** — GroupKFold on `site_key`, per-target and mean CV, public/private LB, CV-vs-LB comparison.
7. **Findings and limitations** — what worked, what didn't, noise in targets, domain shift.
8. **Code and reproducibility** — repo link, Kaggle notebook link, `pip install -e .`, `python -m obscura.train --config ...`, seeds, library versions.

Don't include or publish the competition data, chips or targets.
