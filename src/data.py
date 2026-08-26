"""Fetch and cache per-region hourly data from the open CarbonCast repository.

LiteCast (arXiv:2511.06187) evaluates on 2023 EIA/ENTSO-E energy-mix traces for
50 regions. Those exact 2023 files are not redistributed, but the CarbonCast
repo ships the same hourly data pipeline (carbon intensity, per-source energy
mix, weather forecasts, demand forecasts) for 13 regions for 2020-2021. We use
that real data as the substrate and flag the substitution in the report.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

RAW_BASE = "https://raw.githubusercontent.com/carbonfirst/CarbonCast/main/data"

REGIONS = [
    "AUS_QLD", "BPAT", "CISO", "DE", "ERCO", "ES", "FPL",
    "ISNE", "NL", "NYISO", "PJM", "PL", "SE",
]

# Direct (scope-2) carbon emission factors used by CarbonCast/LiteCast pipeline,
# in gCO2eq/kWh. Source: electricityMap config/co2eq_parameters_direct.json.
CEF_DIRECT = {
    "coal": 760.0, "nat_gas": 370.0, "nuclear": 0.0, "oil": 406.0,
    "hydro": 0.0, "solar": 0.0, "wind": 0.0, "other": 575.0,
}

MIX_COLUMNS = ["coal", "nat_gas", "nuclear", "oil", "hydro", "solar", "wind", "other"]

DEMAND_COLUMNS = ["avg_demand_production_forecast"]

WEATHER_COLUMNS = [
    "forecast_avg_wind_speed_wMean",
    "forecast_avg_temperature_wMean",
    "forecast_avg_dewpoint_wMean",
    "forecast_avg_dswrf_wMean",
    "forecast_avg_precipitation_wMean",
]


def download(cache_dir: str, region: str, kind: str) -> str:
    """Return the local path to a region's CSV, downloading it if needed."""
    os.makedirs(cache_dir, exist_ok=True)
    fname = f"{region}_{kind}.csv"
    dst = os.path.join(cache_dir, fname)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst
    url = f"{RAW_BASE}/{region}/{fname}"
    import urllib.request

    print(f"[data] fetching {url}")
    urllib.request.urlretrieve(url, dst)
    return dst


def load_region(cache_dir: str, region: str) -> dict[str, pd.DataFrame]:
    """Load carbon intensity + mix and the day-ahead exogenous forecast frame."""
    ci = pd.read_csv(download(cache_dir, region, "direct_emissions"),
                     parse_dates=["UTC time"]).set_index("UTC time").sort_index()
    da = pd.read_csv(download(cache_dir, region, "96hr_forecasts_DA"),
                     parse_dates=["UTC time"]).set_index("UTC time").sort_index()

    for col in MIX_COLUMNS + ["carbon_intensity"]:
        ci[col] = pd.to_numeric(ci[col], errors="coerce")
    ci = ci[~ci.index.duplicated()]

    da = da[~da.index.duplicated()]
    for col in da.columns:
        da[col] = pd.to_numeric(da[col], errors="coerce")
    da = da.ffill()

    return {"ci": ci, "da": da}


def carbon_intensity_from_mix(mix: pd.DataFrame, cef: dict[str, float]) -> pd.Series:
    """Weighted-average carbon intensity (gCO2eq/kWh) from per-source generation."""
    m = mix[MIX_COLUMNS].clip(lower=0.0)
    weights = pd.Series([cef[c] for c in MIX_COLUMNS], index=MIX_COLUMNS)
    total = m.sum(axis=1)
    ci = (m * weights).sum(axis=1) / total.replace(0, pd.NA)
    return ci.fillna(0.0)


def build_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Deterministic cyclical time features (hour of day, day of week)."""
    h = index.hour
    d = index.dayofweek
    return pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * h / 24),
        "hour_cos": np.cos(2 * np.pi * h / 24),
        "dow_sin": np.sin(2 * np.pi * d / 7),
        "dow_cos": np.cos(2 * np.pi * d / 7),
    }, index=index)

