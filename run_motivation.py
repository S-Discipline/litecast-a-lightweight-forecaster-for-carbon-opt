"""Motivation experiment reproducing the paper's Fig 2 central claim.

The paper argues that for carbon-aware scheduling, preserving the *ranking*
(concordance) of a forecast matters more than pointwise accuracy (MAPE). Fig 2
shows that on ERCOT (Texas), a forecast with high MAPE but high concordance can
pick the oracle-optimal schedule, while low-MAPE but low-concordance forecasts
pick up to ~20%-worse schedules.

We reproduce this by taking a real 2021 ERCO carbon-intensity window and
constructing forecast variants with controlled concordance and MAPE:
  * oracle              - perfect (used for the reference schedule)
  * high-CI low-MAPE    - additive noise, keeps ordering mostly intact
  * low-CI high-MAPE    - scaled/shifted, breaks ordering
  * noise-only          - random shuffles with controlled permutation distance
Then we schedule a continuous job of length L over slack T with each forecast
and measure realized emissions relative to the oracle schedule.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data import load_region  # noqa: E402
from src.metrics import mape, concordance_index  # noqa: E402
from src.scheduler import schedule_continuous  # noqa: E402


def make_variants(actual: np.ndarray, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out = {}
    # 1) oracle
    out["oracle"] = actual.copy()
    # 2) high concordance, inflated MAPE: monotonic transform (scale+shift) keeps
    #    ordering essentially intact while pushing the pointwise error up
    out["highCI_highMAPE"] = actual * 0.7 + 40 + rng.normal(0, 2, actual.shape)
    # 3) low concordance, low MAPE: tiny additive noise then swap adjacent pairs
    #    to break ordering while keeping pointwise values near the truth
    base = actual + rng.normal(0, 2, actual.shape)
    flipped = base.copy()
    n_swap = int(0.6 * len(actual))
    for k in range(n_swap):
        i = rng.integers(0, len(actual) - 1)
        flipped[i], flipped[i + 1] = flipped[i + 1], flipped[i]
    out["lowCI_lowMAPE"] = flipped
    # 4) shuffled forecasts with controlled concordance (pairwise swaps)
    for frac in (0.2, 0.4, 0.6):
        shuffled = base.copy()
        swap_n = int(frac * len(actual) / 2)
        perm = rng.permutation(len(actual))
        for k in range(swap_n):
            i, j = perm[2 * k], perm[2 * k + 1]
            shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
        out[f"shuffle{int(frac*100)}"] = shuffled
    return out


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    cache_dir = os.path.expanduser(cfg.get("cache_dir", "~/litecast_data"))
    region = cfg.get("region", "ERCO")
    job_length = cfg.get("job_length", 3)
    slack = cfg.get("slack", 24)
    n_windows = cfg.get("n_windows", 60)
    seed = cfg.get("seed", 0)

    d = load_region(cache_dir, region)
    ci = d["ci"]["carbon_intensity"].astype(float)
    ci = ci[~ci.index.duplicated()].sort_index()

    # sample windows of L+T hours across 2021
    win_len = job_length + slack
    starts = pd.date_range("2021-01-01", "2021-11-30", freq="4D")[:n_windows]

    variants = {}
    for s in starts:
        w = ci.reindex(pd.date_range(s, periods=win_len, freq="h"))
        if w.notna().sum() < win_len:
            continue
        actual = w.values.astype(float)
        v = make_variants(actual, seed=seed)
        for name, fc in v.items():
            variants.setdefault(name, []).append(fc)

    rows = []
    print("\n" + "=" * 78)
    print(f"Motivation experiment (paper Fig 2): {region}, job length {job_length}h, slack {slack}h")
    print(f"n_windows={len(starts)} windows of {win_len}h across 2021")
    print("=" * 78)

    for name, fcs in variants.items():
        mapes, cis, extras = [], [], []
        for actual, fc in zip(
                [ci.reindex(pd.date_range(s, periods=win_len, freq="h")).values.astype(float) for s in starts],
                fcs):
            mapes.append(mape(pd.Series(actual), pd.Series(fc)))
            cis.append(concordance_index(pd.Series(actual), pd.Series(fc)))
            h_pred = schedule_continuous(pd.Series(fc), job_length, slack)
            h_oracle = schedule_continuous(pd.Series(actual), job_length, slack)
            e_pred = actual[h_pred:h_pred + job_length].sum()
            e_oracle = actual[h_oracle:h_oracle + job_length].sum()
            extras.append((e_pred / e_oracle - 1.0) * 100.0)
        rows.append({
            "variant": name,
            "MAPE%": float(np.nanmean(mapes)),
            "concordance": float(np.nanmean(cis)),
            "additional_emissions%": float(np.nanmean(extras)),
        })

    df = pd.DataFrame(rows).round(3)
    print(df.to_string(index=False))
    print("\nInterpretation: realized emissions should track concordance, not MAPE.")
    print("The oracle schedule has additional_emissions=0; a forecast with lower MAPE but")
    print("lower concordance should do worse than a higher-MAPE higher-concordance forecast.")
    print("=" * 78)


if __name__ == "__main__":
    main()
