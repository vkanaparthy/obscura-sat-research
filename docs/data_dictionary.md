# OBSCURA data dictionary

Sources: competition README (dataset pipeline v3.0.0) and the Kaggle Data tab. Machine-readable copy in `src/obscura/config.py`.

## Task

The data cover industrial facilities worldwide: refineries, gas plants, LNG terminals, coal power stations and mines, smelters and steelworks, petrochemical and chemical sites, oilfields and a few other kinds. **Each sample is one facility over a window of a few days.** For each sample: one image chip, 29 numeric features and (train only) five targets. Predict all five targets jointly.

All targets are measured over a fixed **10 km window centred on the site** (Sentinel-5P, ~1.1 km grid). Gas enhancements = window mean minus the median of a **30–75 km background ring**. Every target is a **z-score** using unpublished train-set statistics, so units are hidden and all targets share a comparable scale.

**Blank = not measured, not zero.** Satellite gas products are often unavailable (cloud, quality filtering). Blank targets are not scored, but every test row needs all five predictions. Targets are computed **over land only** (MODIS water mask).

### How often each target is blank (train)

| target | blank rate |
|---|---|
| `t_ch4` | about two thirds of rows |
| `t_no2`, `t_co` | a few percent |
| `t_aai`, `t_firms` | almost never |

`t_firms` at its **lowest value means no fire detected**: a real measurement, not a blank. Expect a large spike at the minimum (zero-inflated target).

### Target sources

| target | product |
|---|---|
| `t_ch4`, `t_no2`, `t_co`, `t_aai` | Copernicus Sentinel-5P TROPOMI, site average minus surrounding ring (aai: site window) |
| `t_firms` | NASA FIRMS thermal anomaly detections |

## Targets

| column | unit | meaning | notes |
|---|---|---|---|
| `t_ch4` | z-score | XCH4 (methane) enhancement: site mean − background median | |
| `t_no2` | z-score | tropospheric NO2 column enhancement (site − background) | |
| `t_co` | z-score | CO column enhancement (site − background) | |
| `t_aai` | z-score | absorbing aerosol index (smoke/dust) over the site window | level, not a background difference |
| `t_firms` | z-score | thermal-anomaly (fire/flare) intensity | no detection is a **real low value** |

## Identifiers

| column | meaning |
|---|---|
| `sample_id` | anonymized id, matches `images/<sample_id>.tif`. Keep as text. |
| `site_key` | train only. Anonymized site group, use for GroupKFold. Test sites never appear in train. |

## Tabular features

| column | unit | description | quantization |
|---|---|---|---|
| `month_sin`, `month_cos` | – | month of year on a circle (Dec next to Jan); no day or year column | |
| `temperature` | °C | ERA5-Land 2 m air temperature, daily mean | 0.5 |
| `humidity` | % | relative humidity from ERA5-Land dewpoint, daily mean | 2 |
| `wind_speed` | m/s | ERA5-Land 10 m wind near satellite overpass (mean of hourly speeds) | 0.5 |
| `wind_direction` | deg | meteorological direction (wind blows **from**) near overpass | 10 |
| `precipitation` | m/day | ERA5-Land total precipitation, daily sum | 0.0005 |
| `solar_rad` | MJ/m²/day | ERA5-Land downward surface solar radiation | 0.5 |
| `elevation` | m | Copernicus GLO-30 mean elevation over the chip | 50 |
| `slope` | deg | mean terrain slope over the chip | 1 |
| `s2_cloud_frac` | 0–1 | fraction of chip pixels with cloud probability ≥ 40 | |
| 18 spectral indices (below) | – | mean over the **cloud-free** pixels of the chip, clipped to a sensible range | |

**The 18 spectral indices:** `ndvi`, `evi`, `savi`, `ndre`, `ndwi`, `mndwi`, `awei`, `nmdi`, `nbri`, `nbr2`, `bsi`, `bi`, `swir_ratio`, `ndii`, `hsfi`, `ohi`, `b11_mean`, `b12_mean`.

29 features in total: 2 month + 6 weather + 2 terrain + 1 cloud fraction + 18 spectral.

- Values are **rounded on purpose** (so they can't be matched to public archives).
- No feature is computed from a target. No satellite gas values, quality flags or fire data are among the features.
- Weather features can be missing near coasts.
- Sources: weather ERA5-Land (ECMWF); elevation and slope Copernicus GLO-30.

## Image chips: `images/<sample_id>.tif`

One chip per sample, 4,439 in total (almost all of the ~9 GB). GeoTIFF-style file, readable with `rasterio` or `tifffile`: 14 layers of 384×384 px, uint16, no map position or metadata. **Not cloud-masked**; use SCL and CLDPRB as you see fit. Each facility is framed to fit its size, so ground size per pixel differs between sites and is not stated. **Very dark water can be clipped to zero in the blue band (B2).**

| idx | band | content | notes |
|---|---|---|---|
| 0 | B1 | coastal aerosol | 443 nm, 60 m |
| 1 | B2 | blue | 490 nm, 10 m |
| 2 | B3 | green | 560 nm, 10 m |
| 3 | B4 | red | 665 nm, 10 m |
| 4 | B5 | red edge 1 | 705 nm, 20 m |
| 5 | B6 | red edge 2 | 740 nm, 20 m |
| 6 | B7 | red edge 3 | 783 nm, 20 m |
| 7 | B8 | NIR | 842 nm, 10 m |
| 8 | B8A | narrow NIR | 865 nm, 20 m |
| 9 | B9 | water vapour | 945 nm, 60 m |
| 10 | B11 | SWIR 1 | 1610 nm, 20 m |
| 11 | B12 | SWIR 2 | 2190 nm, 20 m; hot flares are bright here |
| 12 | SCL | scene classification | see below |
| 13 | CLDPRB | s2cloudless cloud probability | 0–100 |

Indices above are 0-based (numpy); the Kaggle page numbers layers 1–14 (rasterio). B1–B12 are Sentinel-2 L2A surface reflectance ×10,000: divide by 10,000 to get 0–1.

SCL values (standard ESA L2A): 0 no data, 1 saturated/defective, 2 dark/shadow, 3 cloud shadow, 4 vegetation, 5 bare soil, 6 water, 7 unclassified, 8 cloud medium probability, 9 cloud high probability, 10 thin cirrus, 11 snow/ice.

## Files

| file | rows | contents |
|---|---|---|
| `train.csv` | 3218 | `sample_id`, `site_key`, five targets, features |
| `test.csv` | 1221 | `sample_id`, features |
| `sample_submission.csv` | 1221 | `sample_id` + five target columns |

## Metric

Masked MCRMSE: per target, RMSE over rows where the target was measured; average the five. Predicting the train mean scores ≈1.0 on train-like rows and higher on held-out facilities, because their targets are shifted. That is the baseline to beat.

## Splits and rules

- Train, public test and private test share nothing: different facilities **at least 150 km apart** and different, non-overlapping time periods. Train is earliest, public test later, private test latest.
- Test has no `site_key` and no way to link a test row to a training site. A model that scores well on a random split but badly on a `site_key` split has memorised sites.
- Ids are random codes; site names, countries, coordinates, dates, product ids and original units are not in any file.
- Do **not** try to identify sites, locations or dates, and do **not** use outside satellite, weather or methane data for any row. Violations invalidate a submission.

## Known limitations

- Enhancements are small relative to retrieval noise; negative z-scores are normal. Treat targets as noisy.
- Targets describe coarse column quantities, not facility emission rates.

## Attribution

Contains modified Copernicus Sentinel data (Sentinel-2, Sentinel-5P) 2022–2026, processed by ESA. FIRMS (NASA); ERA5-Land (C3S/ECMWF); Copernicus DEM GLO-30 (ESA). Data access via Google Earth Engine.
