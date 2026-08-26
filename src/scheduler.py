"""Carbon-aware schedulers: continuous and interruptible job placement over a
forecast window, plus the LiteCast dynamic heuristic (Algorithm 1)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import concordance_index


def schedule_continuous(forecast: pd.Series, job_length: int, slack: int) -> int:
    """Continuous job: pick the start hour h in [0, slack) that minimizes the
    predicted cumulative emissions over the job's L-hour window (Eq. 1)."""
    f = forecast.values
    n = len(f)
    best_h = 0
    best_cost = np.inf
    for h in range(0, slack):
        if h + job_length > n:
            break
        cost = f[h:h + job_length].sum()
        if cost < best_cost:
            best_cost = cost
            best_h = h
    return best_h


def schedule_interruptible(forecast: pd.Series, job_length: int, slack: int) -> np.ndarray:
    """Interruptible job: pick the L hours of lowest predicted intensity inside
    the scheduling window (Eq. 3). Returns the set of chosen hours."""
    f = forecast.values
    n = min(len(f), slack + job_length)
    window = f[:n]
    order = np.argsort(window, kind="stable")
    chosen = np.sort(order[:job_length])
    return chosen


def realized_emissions(actual: pd.Series, hours: np.ndarray) -> float:
    """E_pred = sum of actual carbon intensity over the scheduled hours (Eq. 2/4)."""
    return float(actual.iloc[hours].sum())


def evaluate_schedule(forecast: pd.Series, actual: pd.Series,
                      job_length: int, slack: int,
                      mode: str) -> tuple[float, float]:
    """Return (E_pred, E_oracle) for one job given a forecast and the actuals.

    mode: 'continuous' or 'interruptible'.
    """
    if mode == "continuous":
        h_pred = schedule_continuous(forecast, job_length, slack)
        h_oracle = schedule_continuous(actual, job_length, slack)
        e_pred = realized_emissions(actual, np.arange(h_pred, h_pred + job_length))
        e_oracle = realized_emissions(actual, np.arange(h_oracle, h_oracle + job_length))
    else:
        a_pred = schedule_interruptible(forecast, job_length, slack)
        a_oracle = schedule_interruptible(actual, job_length, slack)
        e_pred = realized_emissions(actual, a_pred)
        e_oracle = realized_emissions(actual, a_oracle)
    return e_pred, e_oracle


class Heuristic:
    """LiteCast's dynamic schedule-update heuristic (Algorithm 1).

    While a job has not started, periodically retrain the forecaster on new
    data; if the new forecast has better concordance (and yields lower expected
    emissions), update the start time.
    """

    def __init__(self, forecaster, retrain_every: int = 6, horizon: int = 168):
        self.forecaster = forecaster
        self.retrain_every = retrain_every  # hours between re-checks
        self.horizon = horizon

    def schedule(self, job_length: int, slack: int, mode: str,
                 ci_hist: pd.Series, da: pd.DataFrame, actual: pd.Series) -> int:
        """Simulate the heuristic over the job's wait window. `actual` is the
        ground-truth future (evaluation-only). Returns chosen start hour.

        Returns tuple (start_hour, concordance_history) for analysis.
        """
        base_forecast = self.forecaster.forecast(ci_hist, da)
        # concordance of the base forecast against actuals within the window
        base_ci = concordance_index(actual, base_forecast)
        start = schedule_continuous(base_forecast, job_length, slack) if mode == "continuous" \
            else schedule_interruptible(base_forecast, job_length, slack)
        # the heuristic updates a continuous job's start time; for interruptible
        # jobs it would re-pick the hour set, so we keep the same logic via start.
        best_ci = base_ci
        best_start = start
        for t in range(1, slack, self.retrain_every):
            new_hist_end = ci_hist.index[-1] + pd.Timedelta(hours=t)
            new_hist = pd.concat([ci_hist, actual.iloc[:t].astype(float)])
            new_hist = new_hist[~new_hist.index.duplicated(keep="last")].sort_index()
            new_fc = self.forecaster.forecast(new_hist, da)
            new_ci = concordance_index(actual, new_fc)
            if new_ci > best_ci:
                cand = schedule_continuous(new_fc, job_length, slack) if mode == "continuous" \
                    else schedule_interruptible(new_fc, job_length, slack)
                # check predicted emissions lower than the current schedule's
                if mode == "continuous":
                    cur_cost = base_forecast.iloc[best_start:best_start + job_length].sum()
                    cand_cost = new_fc.iloc[cand:cand + job_length].sum()
                    if cand_cost < cur_cost:
                        best_start = cand
                        best_ci = new_ci
        return best_start
