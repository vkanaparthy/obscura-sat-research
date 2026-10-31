# Research log

One entry per experiment. This feeds the Devpost research PDF (baselines, experiments, what did and didn't work). `outputs/experiments.jsonl` records scores automatically; write the reasoning here.

CV = 5-fold GroupKFold on `site_key`, masked MCRMSE (lower is better).

| date | name | change | CV MCRMSE | mean baseline | public LB | notes |
|---|---|---|---|---|---|---|
| 2026-10-30 | lgb_baseline | per-target LightGBM, tabular + chip stats | 0.8934 | 1.0090 | 1.12804 | large CV→LB gap; mean LB not yet measured |

### lgb_baseline (2026-10-30)
- **Hypothesis:** cloud-masked chip statistics plus weather/terrain features predict the five targets better than the train mean.
- **Change:** first model. One LightGBM per target, trained only on measured rows; 5-fold GroupKFold on `site_key`; fall back to the mean for any target that loses in CV (none did).
- **CV (per target):**

  | | t_ch4 | t_no2 | t_co | t_aai | t_firms | MCRMSE |
  |---|---|---|---|---|---|---|
  | train mean | 1.0058 | 1.0039 | 1.0019 | 1.0088 | 1.0245 | 1.0090 |
  | LightGBM | 0.9954 | 0.9211 | 0.9750 | 0.6024 | 0.9732 | 0.8934 |

- **Public LB:** 1.12804
- **Takeaway:** almost all of the CV gain comes from `t_aai` (1.01 → 0.60); the other targets barely move. CV is grouped by site but not by time, while test is from later periods, so CV likely overstates how well the model transfers. `t_aai` is a level, not a site-minus-background difference, so it is the target most exposed to time/region shifts. Next: score the train mean on the LB to see whether the model helps there at all, then test `t_aai` = mean.

## Entry template

### <name> (<date>)
- **Hypothesis:**
- **Change:**
- **CV (per target):** ch4 / no2 / co / aai / firms
- **Public LB:**
- **Takeaway:**
