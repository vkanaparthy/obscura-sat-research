# Research log

One entry per experiment. This feeds the Devpost research PDF (baselines, experiments, what did and didn't work). `outputs/experiments.jsonl` records scores automatically; write the reasoning here.

CV = 5-fold GroupKFold on `site_key`, masked MCRMSE (lower is better).

| date | name | change | CV MCRMSE | mean baseline | public LB | notes |
|---|---|---|---|---|---|---|
| | lgb_baseline | per-target LightGBM, tabular + chip stats | | | | |

## Entry template

### <name> (<date>)
- **Hypothesis:**
- **Change:**
- **CV (per target):** ch4 / no2 / co / aai / firms
- **Public LB:**
- **Takeaway:**
