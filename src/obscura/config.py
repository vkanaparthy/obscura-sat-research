"""Constants and data definitions for the OBSCURA competition."""
from __future__ import annotations

import glob
import os
from pathlib import Path

TARGETS = ["t_ch4", "t_no2", "t_co", "t_aai", "t_firms"]
ID_COL = "sample_id"
GROUP_COL = "site_key"

# Band order inside images/<sample_id>.tif (14 bands, uint16, 384x384)
BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12", "SCL", "CLDPRB"]
BAND_IDX = {b: i for i, b in enumerate(BANDS)}
REFLECTANCE_BANDS = BANDS[:12]  # x1e-4 -> surface reflectance
REFLECTANCE_SCALE = 1e-4

# Standard ESA Sentinel-2 L2A scene classification values
SCL_CLASSES = {
    0: "no_data", 1: "saturated", 2: "dark_shadow", 3: "cloud_shadow", 4: "vegetation",
    5: "bare_soil", 6: "water", 7: "unclassified", 8: "cloud_med", 9: "cloud_high",
    10: "cirrus", 11: "snow_ice",
}
CLOUDY_SCL = [0, 1, 3, 8, 9, 10]
CLOUD_PROB_THRESHOLD = 40  # same threshold the organizers use for s2_cloud_frac

TARGET_INFO = {
    "t_ch4": "XCH4 enhancement: site mean minus 30-75 km background median (z-score)",
    "t_no2": "Tropospheric NO2 column enhancement, site minus background (z-score)",
    "t_co": "CO column enhancement, site minus background (z-score)",
    "t_aai": "Absorbing aerosol index over the 10 km site window (z-score)",
    "t_firms": "Fire/flare thermal intensity; no detection is a real low value (z-score)",
}

# (unit, description) for the documented tabular features
FEATURE_INFO = {
    "month_sin": ("-", "cyclic month-of-year (sin)"),
    "month_cos": ("-", "cyclic month-of-year (cos)"),
    "temperature": ("degC", "ERA5-Land 2 m air temperature, daily mean (q 0.5)"),
    "humidity": ("%", "relative humidity from dewpoint, daily mean (q 2)"),
    "wind_speed": ("m/s", "10 m wind speed near overpass (q 0.5)"),
    "wind_direction": ("deg", "direction wind blows from, near overpass (q 10)"),
    "precipitation": ("m/day", "total precipitation, daily sum (q 0.0005)"),
    "solar_rad": ("MJ/m2/day", "downward surface solar radiation (q 0.5)"),
    "elevation": ("m", "Copernicus GLO-30 mean elevation over chip (q 50)"),
    "slope": ("deg", "mean terrain slope over chip (q 1)"),
    "s2_cloud_frac": ("0-1", "fraction of chip pixels with cloud prob >= 40"),
}
# 18 spectral indices: chip means over cloud-free pixels, clipped to a sensible range
SPECTRAL_INDICES = ["ndvi", "evi", "savi", "ndre", "ndwi", "mndwi", "awei", "nmdi", "nbri", "nbr2",
                    "bsi", "bi", "swir_ratio", "ndii", "hsfi", "ohi", "b11_mean", "b12_mean"]
for _c in SPECTRAL_INDICES:
    FEATURE_INFO[_c] = ("-", "spectral index, mean over cloud-free chip pixels")
EXPECTED_FEATURES = list(FEATURE_INFO)  # 29 columns in train.csv / test.csv
assert len(EXPECTED_FEATURES) == 29

FACILITY_TYPES = ["refinery", "gas plant", "LNG terminal", "coal power station", "coal mine", "smelter",
                  "steelworks", "petrochemical", "chemical", "oilfield", "other"]  # documented, not a column

WEATHER_COLS =["temperature", "humidity", "wind_speed", "wind_direction", "precipitation", "solar_rad"]


def find_data_root(explicit: str | None = None) -> Path:
    """Locate the folder containing train.csv (local data/ dir or Kaggle input)."""
    candidates = [explicit, os.environ.get("OBSCURA_DIR"), "data/raw", "/kaggle/input/obscura",
                  *glob.glob("/kaggle/input/*")]
    for c in candidates:
        if c and Path(c, "train.csv").exists():
            return Path(c)
    hits = glob.glob("/kaggle/input/**/train.csv", recursive=True)
    if hits:
        return Path(hits[0]).parent
    raise FileNotFoundError("train.csv not found. Put the data in data/raw/ or set OBSCURA_DIR.")
