# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo", "numpy", "pandas", "matplotlib", "tabulate"]
# ///
# Copyright 2026 OpenResearch

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        **Paper:** LiteCast: A Lightweight Forecaster for Carbon Optimizations
        (Joseph, Savadi & Souza — [arXiv:2511.06187](https://www.alphaxiv.org/abs/2511.06187)).

        **Reproduction scope:** minimal-scale, on real 2021 hourly grid data (CarbonCast dataset),
        run as `orx` experiments on a single vast.ai A10 instance (96 vCPU).

        **The claim being tested:** for carbon-aware scheduling, the *ranking* of a carbon-intensity
        forecast (concordance index) determines realized carbon savings — not its pointwise accuracy
        (MAPE). A lightweight SARIMAX forecaster with ~7 days of history should schedule within ~97% of
        the oracle's carbon optimality, beating the state-of-the-art CarbonCast deep baseline.

        This notebook opens with the reproduction's evidence, then explains the pipeline.
        """
    )
    return


@app.cell
def _(mo):
    mo.md("## Headline result: concordance, not MAPE, drives savings")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        We reproduced the paper's motivating experiment on ERCOT (Texas): a 3-hour job with 24h slack,
        scheduled over 60 real 2021 windows using forecast variants with *controlled* concordance and MAPE.
        The forecast with the highest pointwise error (16.1% MAPE) but preserved ranking (0.96 concordance)
        scheduled within **+0.05%** of the oracle — while numerically more accurate forecasts (1-5% MAPE)
        with degraded ranking cost **+1-3%** extra emissions.
        """
    )
    return


@app.cell
def _(mo):
    mo.image("images/motivation_concordance_vs_mape.png")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        The key mechanism: the scheduler consumes a ranking of hours, not raw values. A monotonic bias that
        preserves order (high MAPE, high concordance) leaves the chosen low-carbon hours unchanged; shuffled
        noise that breaks order (low MAPE, low concordance) picks sub-optimal hours even though its numbers
        are closer.
        """
    )
    return


@app.cell
def _(mo):
    mo.md("## How LiteCast works")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        LiteCast forecasts a grid's hourly average carbon intensity (gCO₂eq/kWh) with a
        **SARIMAX(1,0,1)(1,1,1,24)** seasonal time-series model — one AR/MA pair at lag 1 and one at lag 24
        (daily seasonality), regressed on hour-of-day, day-of-week, weather and demand forecasts. Crucially
        it refits on a rolling **7-day** window, so it is cheap and re-trainable.

        A carbon-aware scheduler then places flexible jobs using only the forecast's *ordering*:

        - **Continuous job** (length $L$, slack $T$): pick the start hour $h$ minimizing predicted
          cumulative emissions over $[h, h+L)$ (paper Eq. 1).
        - **Interruptible job**: pick the $L$ hours of lowest predicted intensity in the window (Eq. 3).

        Realized emissions are scored against the **actual** intensity. Optimality is
        $\rho = E_{pred} / E_{oracle}$, where $E_{oracle}$ is the schedule under perfect foresight.
        """
    )
    return


@app.cell
def _(mo):
    mo.md("## Reproduction evidence")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        **Scheduling optimality vs oracle (4 headline regions, weekly origins, 2021).** LiteCast beats
        persistence at short slack and stays within ~5-8% of the oracle there; long-slack optimality
        degrades (a divergence from the paper, which reports 97% for <24H jobs).
        """
    )
    return


@app.cell
def _(mo):
    mo.image("images/baseline_scheduling_optimality.png")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        **Forecast quality.** The paper reports LiteCast MAPE of 9% (California) to 49% (Denmark). Our
        direct-SARIMAX run produced 14.9% (Netherlands) to 42.7% (Germany) — the range shape matches, but
        absolute values are worse, and LiteCast's concordance came in just below persistence. The paper's
        per-source energy-mix modeling (which we did not use in these headline runs) is the most likely
        explanation.
        """
    )
    return


@app.cell
def _(mo):
    mo.image("images/baseline_forecast_quality.png")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        **LiteCast vs the real CarbonCast CNN-LSTM (Jul-Dec 2021, 13 regions).** Using CarbonCast's public
        pretrained weights, CarbonCast achieved higher concordance in 9/13 regions (0.729 vs 0.680). This
        diverges from the paper — but CarbonCast here had a large training-data advantage (a full prior year
        vs LiteCast's 7 days), and both forecasters still scheduled within 4-6% of the oracle, which is
        consistent with the paper's thesis that ranking quality — not raw accuracy — is what moves emissions.
        """
    )
    return


@app.cell
def _(mo):
    mo.image("images/carboncast_concordance.png")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Summary of the four experiments

        | Experiment | Paper claim | Observed | Assessment |
        |---|---|---|---|
        | Fig 2 motivation (ERCO) | concordance drives savings, not MAPE | high-CI forecast schedules within +0.05%; low-CI costs 1-3% | **Confirmed** |
        | Baseline scheduling | 97% optimality (<24H jobs) | 105-109% at 24H slack; beats persistence | **Partially aligned** |
        | Forecast quality | MAPE 9-49%, CI > CarbonCast | MAPE 14.9-42.7%, CI below persistence | **Diverged** |
        | vs CarbonCast | ~10% higher CI everywhere | CarbonCast higher in 9/13 regions | **Diverged** |
        | Heuristic (Alg 1) | ~22% → ~17% additional emissions | no improvement at minimal cadence | **Diverged at scale** |

        **Key substitutions:** 2021 CarbonCast data (paper: 2023 EIA/ENTSO-E, 50 regions); direct
        carbon-intensity SARIMAX (paper: per-source energy-mix); CarbonCast from pretrained checkpoints;
        Denmark → Netherlands proxy. Full analysis:
        [reproduction report](../reports/litecast-reproduction/report.md).
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ### Interactive (optional)

        The cells below re-derive the concordance-vs-MAPE mechanism on a synthetic window — cheap, no
        training required — so you can vary the "ranking damage" and watch the realized emissions move.
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    hours = pd.date_range("2021-06-01", periods=48, freq="h")
    actual = 200 + 60 * np.sin(2 * np.pi * hours.hour / 24) + rng.normal(0, 8, 48)
    df = pd.DataFrame({"hour": hours, "actual": actual})
    return df, hours, np, pd, rng


@app.cell
def _(df, np, pd, rng):
    mono_bias = df["actual"] * 0.7 + 40  # preserves ordering, inflates MAPE
    swap_n = 20
    shuffled = (df["actual"] + rng.normal(0, 2, 48)).values.copy()
    perm = rng.permutation(48)
    for k in range(swap_n):
        i, j = perm[2 * k], perm[2 * k + 1]
        shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
    df["highCI_highMAPE"] = mono_bias
    df["lowCI_lowMAPE"] = shuffled
    return mono_bias, shuffled, swap_n, perm


@app.cell
def _(df, np):
    def _mape(y_true, y_pred):
        return float(np.mean(np.abs(y_true - y_pred) / np.abs(y_true).clip(lower=1e-6)) * 100)

    def _concordance(y_true, y_pred):
        y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
        n = len(y)
        concordant = comparable = 0
        for i in range(n):
            for j in range(i + 1, n):
                if y[i] == y[j]:
                    continue
                comparable += 1
                concordant += 0.5 if p[i] == p[j] else float((p[i] < p[j]) == (y[i] < y[j]))
        return concordant / comparable if comparable else float("nan")

    job_length, slack = 3, 24
    rows = []
    for name in ("highCI_highMAPE", "lowCI_lowMAPE"):
        fc = df[name]
        m = _mape(df["actual"], fc)
        ci = _concordance(df["actual"], fc)
        h_pred = int(np.argmin([fc.iloc[h:h + job_length].sum() for h in range(slack)]))
        h_oracle = int(np.argmin([df["actual"].iloc[h:h + job_length].sum() for h in range(slack)]))
        e_pred = df["actual"].iloc[h_pred:h_pred + job_length].sum()
        e_oracle = df["actual"].iloc[h_oracle:h_oracle + job_length].sum()
        rows.append({"variant": name, "MAPE%": round(m, 1), "concordance": round(ci, 3),
                     "additional_emissions%": round((e_pred / e_oracle - 1) * 100, 2)})
    return _concordance, _mape, job_length, rows, slack


@app.cell
def _(mo, pd, rows):
    mo.md(f"**On one synthetic window:** {pd.DataFrame(rows).to_markdown(index=False)}")
    return


if __name__ == "__main__":
    app.run()
