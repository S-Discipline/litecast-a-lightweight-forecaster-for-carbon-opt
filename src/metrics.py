"""Evaluation metrics used by LiteCast: MAPE, concordance index, carbon
optimality (ratio of realized emissions to oracle emissions)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean absolute percentage error (%). Aligns on index, ignores NaNs."""
    df = pd.DataFrame({"y": y_true, "p": y_pred}).dropna()
    if len(df) == 0:
        return float("nan")
    eps = 1e-6
    return float((np.abs(df["y"] - df["p"]) / np.abs(df["y"]).clip(lower=eps)).mean() * 100.0)


def concordance_index(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Concordance index (Harrell's C): fraction of comparable pairs whose
    forecast ranking matches the ground-truth ranking. Ties in the forecast are
    counted as 0.5; ties in the ground truth are not comparable."""
    df = pd.DataFrame({"y": y_true.values, "p": y_pred.values}).dropna()
    if len(df) < 2:
        return float("nan")
    n = len(df)
    y = df["y"].to_numpy()
    p = df["p"].to_numpy()
    concordant = 0.0
    comparable = 0
    for i in range(n):
        for j in range(i + 1, n):
            if y[i] == y[j]:
                continue
            comparable += 1
            if p[i] == p[j]:
                concordant += 0.5
            elif (p[i] < p[j]) == (y[i] < y[j]):
                concordant += 1.0
    if comparable == 0:
        return float("nan")
    return float(concordant / comparable)


def realized_emissions(actual: np.ndarray, schedule_hours: np.ndarray) -> float:
    """Sum of actual carbon intensity over the scheduled hours (job consumes 1 kW)."""
    return float(actual[schedule_hours].sum())


def carbon_optimality(e_pred: float, e_oracle: float) -> float:
    """rho = E_pred / E_oracle, as a percentage. 100% = perfect (oracle)."""
    if e_oracle <= 0:
        return float("nan")
    return float(e_pred / e_oracle * 100.0)


def additional_emissions(e_pred: float, e_oracle: float) -> float:
    """(E_pred / E_oracle - 1) * 100, i.e. % above the oracle. 0% = perfect."""
    if e_oracle <= 0:
        return float("nan")
    return float((e_pred / e_oracle - 1.0) * 100.0)
