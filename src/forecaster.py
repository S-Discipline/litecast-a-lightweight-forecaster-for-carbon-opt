"""Forecasters: LiteCast (SARIMAX with exogenous weather/demand), Persistence,
and the Oracle (perfect future knowledge)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .data import CEF_DIRECT, MIX_COLUMNS, WEATHER_COLUMNS, DEMAND_COLUMNS
from .metrics import mape as _mape, concordance_index as _ci

SARIMAX_ORDER = (1, 0, 1)
SARIMAX_SEASONAL = (1, 1, 1, 24)  # hourly data, daily seasonality (m=24)


def _harmonize(df: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    return df.reindex(idx)


def _exog_frame(da: pd.DataFrame, idx: pd.DatetimeIndex,
                use_weather: bool, use_demand: bool, use_time: bool) -> pd.DataFrame:
    feats = []
    if use_time:
        h = idx.hour
        d = idx.dayofweek
        feats.append(pd.DataFrame({
            "hour_sin": np.sin(2 * np.pi * h / 24),
            "hour_cos": np.cos(2 * np.pi * h / 24),
            "dow_sin": np.sin(2 * np.pi * d / 7),
            "dow_cos": np.cos(2 * np.pi * d / 7),
        }, index=idx))
    if use_weather:
        wcols = [c for c in WEATHER_COLUMNS if c in da.columns]
        if wcols:
            feats.append(da[wcols].reindex(idx))
    if use_demand:
        dcols = [c for c in DEMAND_COLUMNS if c in da.columns]
        if dcols:
            feats.append(da[dcols].reindex(idx))
    if not feats:
        return pd.DataFrame(index=idx)
    return pd.concat(feats, axis=1)


class LiteCast:
    """SARIMAX(1,0,1)(1,1,1,24) forecaster per the LiteCast paper.

    The model is refit on a rolling window of `train_days` days of historical
    carbon intensity (or per-source energy mix, aggregated via emission factors)
    and produces a multi-step hourly forecast of carbon intensity for the next
    `horizon` hours, using exogenous weather/demand forecast inputs when given.
    """

    def __init__(self, train_days: int = 7, horizon: int = 168,
                 use_weather: bool = True, use_demand: bool = True,
                 use_time: bool = True, order=SARIMAX_ORDER,
                 seasonal=SARIMAX_SEASONAL, verbose: bool = False):
        self.train_days = train_days
        self.horizon = horizon
        self.use_weather = use_weather
        self.use_demand = use_demand
        self.use_time = use_time
        self.order = order
        self.seasonal = seasonal
        self.verbose = verbose

    def _fit_forecast(self, y: pd.Series, exog_train: pd.DataFrame,
                      exog_future: pd.DataFrame) -> pd.Series:
        y = y.replace([np.inf, -np.inf], np.nan).ffill().fillna(y.mean())
        exog_train = exog_train.fillna(0.0)
        exog_future = exog_future.fillna(0.0)
        try:
            model = SARIMAX(y, exog=exog_train if len(exog_train.columns) else None,
                            order=self.order, seasonal_order=self.seasonal,
                            enforce_stationarity=False, enforce_invertibility=False)
            fit = model.fit(disp=False, maxiter=200)
            pred = fit.get_forecast(len(exog_future),
                                    exog=exog_future if len(exog_future.columns) else None)
            return pd.Series(pred.predicted_mean.values, index=exog_future.index)
        except Exception as e:  # noqa: BLE001 - fall back to persistence
            if self.verbose:
                print(f"  [litecast] SARIMAX failed ({e}); falling back to persistence")
            return y.iloc[-24:].mean().reindex(exog_future.index).ffill()

    def forecast(self, ci: pd.Series, da: pd.DataFrame, refit: bool = True) -> pd.Series:
        """Produce an hourly carbon-intensity forecast.

        `ci` is the historical carbon-intensity series; `da` carries the
        exogenous weather/demand forecast data. Returns a Series over the next
        `horizon` hours after the end of `ci`.
        """
        hist = ci.dropna()
        if len(hist) == 0:
            raise ValueError("empty history")
        last_t = hist.index[-1]
        future_idx = pd.date_range(start=last_t + pd.Timedelta(hours=1),
                                   periods=self.horizon, freq="h")
        train = hist.tail(self.train_days * 24)
        exog_train = _exog_frame(da, train.index, self.use_weather,
                                 self.use_demand, self.use_time)
        exog_future = _exog_frame(da, future_idx, self.use_weather,
                                  self.use_demand, self.use_time)
        return self._fit_forecast(train, exog_train, exog_future)


class MixLiteCast:
    """LiteCast variant that forecasts per-source energy-mix proportions with
    SARIMAX and aggregates them to carbon intensity via direct emission factors,
    mirroring the paper's energy-mix modeling choice."""

    def __init__(self, train_days: int = 7, horizon: int = 168,
                 use_weather: bool = True, use_demand: bool = True,
                 use_time: bool = True, verbose: bool = False):
        self.train_days = train_days
        self.horizon = horizon
        self.use_weather = use_weather
        self.use_demand = use_demand
        self.use_time = use_time
        self.verbose = verbose

    def forecast(self, ci: pd.DataFrame, da: pd.DataFrame) -> pd.Series:
        """ci must contain the per-source mix columns (or `carbon_intensity`)."""
        last_t = ci.index[-1]
        future_idx = pd.date_range(start=last_t + pd.Timedelta(hours=1),
                                   periods=self.horizon, freq="h")
        train = ci.tail(self.train_days * 24)
        exog_train = _exog_frame(da, train.index, self.use_weather,
                                 self.use_demand, self.use_time)
        exog_future = _exog_frame(da, future_idx, self.use_weather,
                                  self.use_demand, self.use_time)

        if MIX_COLUMNS[0] in ci.columns:
            # model each source independently, then aggregate via emission factors
            src_pred = {}
            for src in MIX_COLUMNS:
                y = train[src].astype(float).replace([np.inf, -np.inf], np.nan).ffill()
                if y.dropna().empty:
                    src_pred[src] = pd.Series(0.0, index=future_idx)
                    continue
                fc = LiteCast(self.train_days, self.horizon, self.use_weather,
                              self.use_demand, self.use_time,
                              verbose=self.verbose)._fit_forecast(
                    y, exog_train, exog_future)
                src_pred[src] = fc.clip(lower=0.0)
            pred_mix = pd.DataFrame(src_pred)
            total = pred_mix.sum(axis=1)
            ci_pred = (pred_mix * pd.Series([CEF_DIRECT[c] for c in MIX_COLUMNS],
                                            index=MIX_COLUMNS)).sum(axis=1) / total.replace(0, pd.NA)
            return ci_pred.fillna(0.0)
        else:
            return LiteCast(self.train_days, self.horizon, self.use_weather,
                            self.use_demand, self.use_time,
                            verbose=self.verbose).forecast(
                ci["carbon_intensity"], da)


class Persistence:
    """Reuses the previous day's observations as the forecast (naive baseline)."""

    def __init__(self, horizon: int = 168):
        self.horizon = horizon

    def forecast(self, ci: pd.Series, da: pd.DataFrame | None = None) -> pd.Series:
        hist = ci.dropna()
        last_t = hist.index[-1]
        future_idx = pd.date_range(start=last_t + pd.Timedelta(hours=1),
                                   periods=self.horizon, freq="h")
        prev_day = hist.tail(24)
        reps = int(np.ceil(self.horizon / 24))
        vals = np.tile(prev_day.values, reps)[:self.horizon]
        return pd.Series(vals, index=future_idx)


class Oracle:
    """Perfect future knowledge; used only for evaluation of maximum savings."""

    def __init__(self, horizon: int = 168):
        self.horizon = horizon

    def forecast(self, ci: pd.Series, future_actual: pd.Series | None = None) -> pd.Series:
        if future_actual is None:
            raise ValueError("Oracle needs the future actual series")
        return future_actual.iloc[:self.horizon]
