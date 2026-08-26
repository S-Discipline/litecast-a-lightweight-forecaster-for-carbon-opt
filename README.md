# LiteCast: A Lightweight Forecaster for Carbon Optimizations — Reproduction

## Paper claim tested

This repository reproduces the central claim of **LiteCast: A Lightweight Forecaster for Carbon Optimizations**
(Joseph, Savadi & Souza, UC Santa Cruz, [arXiv:2511.06187](https://www.alphaxiv.org/abs/2511.06187)):

> For carbon-aware scheduling, a forecast's **concordance (rank-ordering quality)** drives realized
> carbon savings — not its pointwise accuracy (MAPE) — and a lightweight SARIMAX forecaster using
> only ~7 days of history can schedule within ~97% of the oracle's carbon optimality, outperforming
> the state-of-the-art deep-learning CarbonCast baseline.

## What was done

Implemented the full LiteCast pipeline from scratch — SARIMAX(1,0,1)(1,1,1,24) forecaster with
weather/demand/hour/day-of-week exogenous regressors, a continuous + interruptible carbon-aware
scheduler, the dynamic re-scheduling heuristic (Algorithm 1), and MAPE / concordance / carbon-optimality
metrics — then drove it through `orx` experiments on a **single vast.ai A10 instance (96 vCPU)** using
real 2021 hourly grid data (carbon intensity, energy mix, weather and demand forecasts) from the open
CarbonCast repository. The real pretrained CarbonCast CNN-LSTM was run as the SOTA baseline.

## Assessment (minimal-scale reproduction)

| Claim | Paper number | Observed number | Assessment |
|---|---|---|---|
| Concordance (not MAPE) drives savings | 32% MAPE / 95% CI forecast schedules optimally; low-CI costs up to 20% | 16% MAPE / 0.96 CI schedules within +0.05%; low-CI costs +1-3% | **Confirmed** |
| LiteCast near-oracle scheduling | 97% optimality (<24H jobs) | 105-109% optimality at 24H slack; beats persistence; 117-144% at long slack | **Partially aligned** |
| LiteCast MAPE | 9% (CA) to 49% (DK) | 14.9% (NL) to 42.7% (DE) | **Diverged** (same range shape, worse absolute) |
| LiteCast concordance > CarbonCast | ~10% higher everywhere, 15% in high-wind | CarbonCast higher in 9/13 regions (0.729 vs 0.680) | **Diverged** |
| Heuristic cuts emissions | ~22% → ~17% additional | no improvement at minimal cadence | **Diverged at this scope** |

**Downscaling / substitutions:** the paper's 2023 EIA/ENTSO-E data for 50 regions is not public; we used
the CarbonCast 2021 data (13 regions; Denmark → Netherlands proxy). LiteCast ran as the *direct*
carbon-intensity SARIMAX rather than the paper's per-source energy-mix modeling (the likely cause of the
forecast-quality divergence), and CarbonCast ran from its pretrained checkpoints rather than retrained for
the target year. Full details and per-claim analysis in the
[reproduction report](reports/litecast-reproduction/report.md).

## Experiment log

| Branch / experiment | Purpose or change | Exact run command | Assessment | Compute |
|---|---|---|---|---|
| `orx/baseline-litecast-sarimax-forecast-carbon-aware` | LiteCast SARIMAX forecast quality + scheduling vs oracle/persistence, 4 regions, 2021 | `bash run_experiment.sh` | Partially aligned (Claims 2-3) | vast.ai A10, 96 vCPU |
| `orx/carboncast-cnn-lstm-baseline-comparison` | LiteCast vs real pretrained CarbonCast CNN-LSTM, Jul-Dec 2021, 13 regions | `bash run_experiment.sh` | Diverged (Claim 4) | vast.ai A10, 96 vCPU |
| `orx/concordance-vs-mape-motivation-fig-2` | Controlled concordance/MAPE variants on ERCO (paper Fig 2) | `bash run_experiment.sh` | Confirmed (Claim 1) | vast.ai A10, 96 vCPU |
| `orx/litecast-heuristic-algorithm-1` | Dynamic re-scheduling heuristic (Algorithm 1) | `bash run_experiment.sh` | Diverged at scale (Claim 5) | vast.ai A10, 96 vCPU |
| `main` | Not run as an experiment (publication surface) | — | — | — |

Each branch runs `bash run_experiment.sh`, which dispatches on the `experiment:` key in `config.yaml`
(`baseline` / `carboncast` / `motivation` / `heuristic`).

## Try it yourself

```sh
pip install -r requirements.txt
python run.py --config configs/baseline.yaml            # baseline (LiteCast vs persistence vs oracle)
python run_carboncast.py --config configs/carboncast.yaml # vs real CarbonCast CNN-LSTM
python run_heuristic.py --config configs/heuristic.yaml  # Algorithm 1 heuristic
python run_motivation.py --config configs/motivation.yaml # paper Fig 2 concordance-vs-MAPE
```

Interactive tutorial: open `notebooks/litecast_reproduction.py` with `marimo edit` or `marimo run`.

---

*The upstream repository is empty; this reproduction project lives entirely in this tree.*
